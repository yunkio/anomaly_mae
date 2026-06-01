#!/usr/bin/env python
"""Backfill best_model_score_contribution.png for experiments whose run was
RESUMED before the 2026-05-30 checkpoint off-by-one fix.

Those runs saved the per-epoch contribution-ratio history 1 entry short
(epoch=500 but epoch_recon_ratio_*=499), which crashed
plot_score_contribution_analysis's stackplot inside the (swallowed) _safe_plot,
so the PNG was never written. The viz code now length-aligns defensively, but
the already-completed experiments still lack the file. This script regenerates
ONLY that single plot, faithfully replicating the pipeline's inference + viz
path (real dataset loaders, real model, excl22 region filtering) — it does not
touch any other artefact.

Usage:
    python scripts/backfill_score_contribution.py
"""
import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')

from mae_anomaly import Config
from mae_anomaly.model import SelfDistilledMAEMultivariate
from mae_anomaly.dataset_sliding import SlidingWindowDataset, AnomalyRegion
from mae_anomaly.evaluator import Evaluator, find_swat_region_22
from mae_anomaly.utils.experiment import resolve_test_stride
from mae_anomaly.visualization import BestModelVisualizer, setup_style
from mae_anomaly.visualization.base import derive_pred_data
from torch.utils.data import DataLoader

# Import pipeline helpers from run_base_experiments
from scripts.run_base_experiments import get_dataset_loader, _filter_excl22_viz_data

# (dataset_dir, loader_key, is_excl22, model_path_override)
TARGETS = [
    ("results/experiments/271_lr_20260529_225351_baseline/SWaT/A1A2_full",  "swat_A1A2",      False, None),
    ("results/experiments/271_lr_20260529_225351_baseline/SWaT/A1A2_excl22","swat_A1A2",      True,
        "results/experiments/271_lr_20260529_225351_baseline/SWaT/A1A2_full/best_model.pt"),
    ("results/experiments/271_lr_20260529_225351_baseline/WaDi/A1",         "WaDi_14days_A1", False, None),
    ("results/experiments/274_20260529_100418_274canon_balsamp/WaDi/A1",    "WaDi_14days_A1", False, None),
]


def load_config_from_best_model(model_path):
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    config = Config()
    for k, v in ckpt["config"].items():
        if hasattr(config, k):
            setattr(config, k, v)
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    return config, ckpt


def backfill_one(dataset_dir, loader_key, is_excl22, model_path_override):
    tag = dataset_dir.split("results/experiments/")[-1]
    model_path = model_path_override or os.path.join(dataset_dir, "best_model.pt")
    if not os.path.exists(model_path):
        print(f"[SKIP] {tag}: no best_model.pt ({model_path})")
        return False

    print(f"\n===== {tag} (loader={loader_key}, excl22={is_excl22}) =====")
    config, ckpt = load_config_from_best_model(model_path)

    # --- Real data (same loaders as run_base_experiment) ---
    loader_fn = get_dataset_loader(loader_key)
    data = loader_fn()
    if len(data) >= 6:
        signals, point_labels, anomaly_regions, feature_names, train_ratio, data_info = data[:6]
    elif len(data) == 5:
        signals, point_labels, anomaly_regions, feature_names, train_ratio = data
        data_info = None
    else:
        signals, point_labels, anomaly_regions, feature_names = data
        train_ratio = config.sliding_window_train_ratio
        data_info = None
    run_boundaries = data_info.get("run_boundaries") if data_info else None
    config.num_features = signals.shape[1]

    # --- Test dataset (identical construction to run_base_experiment L2282) ---
    test_stride = resolve_test_stride(config)
    test_dataset = SlidingWindowDataset(
        signals=signals, point_labels=point_labels, anomaly_regions=anomaly_regions,
        window_size=config.seq_length, stride=test_stride, mask_last_n=config.patch_size,
        split="test", train_ratio=train_ratio, seed=config.random_seed,
        run_boundaries=run_boundaries,
        normalize_mode=config.normalize_mode,
        minmax_range=getattr(config, "minmax_range", "0_1"),
        minmax_clamp_min=getattr(config, "minmax_clamp_min", None),
        minmax_clamp_max=getattr(config, "minmax_clamp_max", None),
    )
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    print(f"  test windows: {len(test_dataset)}  stride={test_stride}")

    # --- Model + patch-score inference (same as run_base_experiment L3054) ---
    model = SelfDistilledMAEMultivariate(config).to(config.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    evaluator = Evaluator(model, config, test_loader, test_dataset=test_dataset)
    recon_p, disc_p, student_p, labels, sample_types, anomaly_types = \
        evaluator._compute_patch_scores_all_patches(collect_detail=True)
    fm_patches = getattr(evaluator, "fm_patches", None)

    # --- pred_data (same subsample as pipeline: linspace 10k) ---
    VIZ_MAX = 10000
    total_test = len(test_dataset)
    viz_idx = (np.linspace(0, total_test - 1, VIZ_MAX, dtype=int)
               if total_test > VIZ_MAX else np.arange(total_test))
    pred_data = derive_pred_data(
        recon_p, disc_p, student_p, labels, sample_types,
        config, test_dataset, subset_indices=viz_idx, fm_patches=fm_patches,
    )

    # --- excl22 filter (test-local region22, same as bg-worker L1706) ---
    if is_excl22:
        total_length = len(signals)
        train_end = int(total_length * train_ratio)
        test_regions = [
            AnomalyRegion(start=r.start - train_end, end=r.end - train_end, anomaly_type=r.anomaly_type)
            for r in anomaly_regions if r.start >= train_end
        ]
        region22 = find_swat_region_22(test_regions)
        if region22 is None:
            print("  [WARN] excl22: region22 not found — using unfiltered pred_data")
        else:
            _filt = _filter_excl22_viz_data(
                pred_data, None, config, int(region22.start), int(region22.end))
            pred_data = _filt[0]  # (pred, det[, keep_win]) — take filtered pred_data
            print(f"  excl22 filtered: region [{region22.start}, {region22.end})")

    # --- Generate ONLY the score-contribution plot ---
    viz_dir = os.path.join(dataset_dir, "visualization", "best_model")
    os.makedirs(viz_dir, exist_ok=True)
    setup_style()
    vis = BestModelVisualizer(pred_data=pred_data, config=config, output_dir=viz_dir)
    out_png = os.path.join(viz_dir, "best_model_score_contribution.png")
    vis.plot_score_contribution_analysis(experiment_dir=dataset_dir)

    if os.path.exists(out_png):
        print(f"  ✅ wrote {out_png}")
        return True
    print(f"  ❌ png not created")
    return False


def main():
    ok = 0
    for t in TARGETS:
        try:
            if backfill_one(*t):
                ok += 1
        except Exception as e:
            import traceback
            print(f"  ❌ FAILED: {e}")
            traceback.print_exc()
    print(f"\n=== backfill done: {ok}/{len(TARGETS)} generated ===")


if __name__ == "__main__":
    main()
