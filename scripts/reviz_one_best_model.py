#!/usr/bin/env python
"""Regenerate best_model/*.png for ONE real-dataset cell via the production GPU path.

The stock load_best_model() only generates SIMULATION data, so it mismatches
real-dataset cells (WaDi/SWaT/PSM). This builds the model from best_model.pt and
the test set from the real DATASET_LOADERS, then runs the production path:
  Evaluator._compute_patch_scores_all_patches -> derive_pred_data (single-source
  new 4:1/no-FM score) -> BestModelVisualizer(pred_data, detailed_data).generate_all()

best epoch did NOT flip for these cells -> best_model.pt is the right epoch.
force_recon_only is gated on best_epoch (pre-warmup best -> recon-only viz score).

Usage: python scripts/reviz_one_best_model.py <cell_dir>   (one cell per process)
"""
import os, sys, json
os.environ.setdefault('MPLBACKEND', 'Agg')
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '4'
sys.path.insert(0, '/home/ykio/notebooks/TSMAE')
import numpy as np
import torch

CELL_TO_LOADER = {'PSM': 'PSM', 'WaDi/A1': 'WaDi_14days_A1', 'WaDi/A2': 'WaDi_14days_A2',
                  'SWaT/A1A2_full': 'swat_A1A2', 'SWaT/A1A2_excl22': 'swat_A1A2'}


def loader_key(cell_dir):
    for suffix, key in CELL_TO_LOADER.items():
        if cell_dir.replace('\\', '/').endswith('/' + suffix) or cell_dir.endswith(suffix):
            return key
    return None


def main():
    cell_dir = sys.argv[1].rstrip('/')
    from mae_anomaly import Config, set_seed
    from mae_anomaly.model import SelfDistilledMAEMultivariate
    from mae_anomaly.dataset_sliding import SlidingWindowDataset
    from mae_anomaly.datasets.loaders import get_dataset_loader
    from mae_anomaly.utils.experiment import resolve_test_stride
    from mae_anomaly.evaluator import Evaluator
    from mae_anomaly.visualization import BestModelVisualizer
    from mae_anomaly.visualization.base import derive_pred_data
    from mae_anomaly.scoring import is_prewarmup_epoch
    from torch.utils.data import DataLoader

    model_path = os.path.join(cell_dir, 'best_model.pt')
    if not os.path.exists(model_path):
        raise SystemExit(f"no best_model.pt in {cell_dir}")
    meta = json.load(open(os.path.join(cell_dir, 'experiment_metadata.json')))
    best_epoch = meta.get('timing', {}).get('best_epoch')

    # --- model + config from checkpoint ---
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    config = Config()
    for k, v in ckpt['config'].items():
        if hasattr(config, k):
            setattr(config, k, v)
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SelfDistilledMAEMultivariate(config)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(config.device).eval()

    # --- REAL test dataset via DATASET_LOADERS ---
    key = loader_key(cell_dir)
    if key is None:
        raise SystemExit(f"cannot map cell to loader: {cell_dir}")
    data = get_dataset_loader(key)()
    data_info = {}
    if len(data) == 6:
        signals, point_labels, anomaly_regions, feature_names, train_ratio, data_info = data
    elif len(data) == 5:
        signals, point_labels, anomaly_regions, feature_names, train_ratio = data
    else:
        signals, point_labels, anomaly_regions, feature_names = data
        train_ratio = 0.5
    config.num_features = signals.shape[1]
    set_seed(config.random_seed)
    test_dataset = SlidingWindowDataset(
        signals=signals, point_labels=point_labels, anomaly_regions=anomaly_regions,
        window_size=config.seq_length, stride=resolve_test_stride(config),
        mask_last_n=config.patch_size, split='test', train_ratio=train_ratio,
        seed=config.random_seed, run_boundaries=(data_info or {}).get('run_boundaries'),
        normalize_mode=config.normalize_mode,
        minmax_range=getattr(config, 'minmax_range', '0_1'),
        minmax_clamp_min=getattr(config, 'minmax_clamp_min', None),
        minmax_clamp_max=getattr(config, 'minmax_clamp_max', None))
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # --- GPU patch scores -> pred_data (production path) ---
    evaluator = Evaluator(model, config, test_loader, test_dataset=test_dataset)
    recon_p, disc_p, student_p, labels, sample_types, anomaly_types = \
        evaluator._compute_patch_scores_all_patches(collect_detail=True)
    full_detail = evaluator.detail_results
    fm_patches = getattr(evaluator, 'fm_patches', None)
    disc_per_feature = getattr(evaluator, 'disc_per_feature', None)

    total_test = len(test_dataset)
    VIZ_MAX = 10000
    viz_idx = (np.linspace(0, total_test - 1, VIZ_MAX, dtype=int)
               if total_test > VIZ_MAX else np.arange(total_test))
    pred_data = derive_pred_data(
        recon_p, disc_p, student_p, labels, sample_types, config, test_dataset,
        subset_indices=viz_idx, fm_patches=fm_patches,
        force_recon_only=is_prewarmup_epoch(config, best_epoch))
    detailed_data = {k: (v[viz_idx] if isinstance(v, np.ndarray) and v.ndim >= 1
                         and v.shape[0] == total_test else v)
                     for k, v in full_detail.items()}
    # disc_per_feature -> viz subset (final form; excl22 adds a keep_win filter below)
    disc_pf = (disc_per_feature[viz_idx]
               if (disc_per_feature is not None
                   and getattr(disc_per_feature, 'shape', [0])[0] == total_test)
               else None)

    # --- SWaT excl22: filter out region-22 windows/points (eval-masking viz) ---
    is_excl22 = cell_dir.replace('\\', '/').endswith('A1A2_excl22')
    if is_excl22:
        info = meta.get('excl_region22_info')
        if not isinstance(info, dict):
            raise SystemExit("excl22 cell but no excl_region22_info in metadata")
        from scripts.run_base_experiments import _filter_excl22_viz_data
        rs, re_ = int(info['region_start']), int(info['region_end'])
        pred_data, detailed_data, keep_win = _filter_excl22_viz_data(
            pred_data, detailed_data, config, rs, re_)
        if disc_pf is not None:
            disc_pf = disc_pf[keep_win]

    history = None
    thp = os.path.join(cell_dir, 'training_histories.json')
    if os.path.exists(thp):
        hd = json.load(open(thp))
        history = hd.get('0', hd if isinstance(hd, dict) else None)

    del model, evaluator, test_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    best_dir = os.path.join(cell_dir, 'visualization', 'best_model')
    os.makedirs(best_dir, exist_ok=True)
    vis = BestModelVisualizer(config=config, output_dir=best_dir,
                              pred_data=pred_data, detailed_data=detailed_data)
    if disc_pf is not None:
        try:
            vis.disc_per_feature = disc_pf
        except Exception:
            pass
    vis.generate_all(experiment_dir=cell_dir, history=history)
    print(f"BEST_MODEL_VIZ_OK {cell_dir} loader={key} best_epoch={best_epoch} "
          f"prewarmup={is_prewarmup_epoch(config, best_epoch)} excl22={is_excl22}")


if __name__ == '__main__':
    main()
