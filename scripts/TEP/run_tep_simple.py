"""TEP type-generalization experiment #12 — simple-baseline runner.

Runs pca_error + sensor_range on the pre-registered fold design
(temp/tep_design/80_experiment_design_final.md) as comparison anchors for the
upcoming MAE runs:

  - 4 contaminated folds (f_step/f_rand/f_ds/f_unk): train = FF 240 runs +
    seen-family faulty 60 runs (models are label-blind; folds differ only in
    train contamination composition)
  - ffonly: clean-normal reference train (B0 analogue, fold-independent scores,
    fold-dependent partition bookkeeping)

Evaluation = EXISTING baseline stack, unchanged: comparison.baseline_common.
compute_all_metrics (thin wrapper over mae_anomaly.evaluator.compute_full_metric_set
— identical metric set/conventions as every baseline experiment). Reported per
partition: full (unprefixed) + seen_/unseen_/exclhard_ prefixes (excl22-style),
plus per-fault lite metrics (per_fault_metrics.json).

Run-boundary policy: pca_error's 5-box smoothing is applied PER RUN (never
crosses a run seam; upstream semantics otherwise preserved — global median-IQR
statistics, first 5 rows of each run stay 0). sensor_range is pointwise (no
temporal mixing). Region objects never span runs by construction.

Usage:
  ~/anaconda3/envs/dc_vis/bin/python scripts/TEP/run_tep_simple.py            # all
  ~/anaconda3/envs/dc_vis/bin/python scripts/TEP/run_tep_simple.py --fold f_step --smoke
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tep_common import (
    DATA_DIR, EXCLUDED_HARD, EXPERIMENT_NUMBER, FOLDS, MODEL_PARAMS,
    NORMALIZE_MODE, REPO_ROOT, RESULTS_BASE, RUN_LEN, FAULT_ONSET_IDX,
    seen_faults, unseen_faults,
)

sys.path.insert(0, REPO_ROOT)
from comparison.baseline_common import (          # noqa: E402
    _aggregate_run_metrics, compute_all_metrics, save_epoch_metrics,
    save_metadata, save_scores_npz,
)
from comparison.baselines.pca_error.model import PCAError, _median_iqr_smooth  # noqa: E402
from comparison.baselines.sensor_range.model import SensorRangeDeviation       # noqa: E402
from comparison.baselines.l2_norm.model import L2Norm                          # noqa: E402
from comparison.baselines.nn_distance.model import NNDistance                  # noqa: E402
from comparison.baselines.random.model import RandomBaseline                   # noqa: E402
from mae_anomaly.dataset_sliding import AnomalyRegion                           # noqa: E402


# ---------------------------------------------------------------------------
# normalization — comparison Q1 standard (minmax, train-only fit, NO clip)
# ---------------------------------------------------------------------------
def fit_minmax(train_X: np.ndarray):
    mn = train_X.min(axis=0).astype(np.float64)
    mx = train_X.max(axis=0).astype(np.float64)
    denom = mx - mn
    denom[denom == 0] = 1.0  # constant-column guard (sklearn MinMaxScaler semantics)
    return mn, denom


def apply_minmax(X: np.ndarray, mn, denom) -> np.ndarray:
    return ((X.astype(np.float64) - mn) / denom).astype(np.float32)


# ---------------------------------------------------------------------------
# boundary-safe pca_error scoring
# ---------------------------------------------------------------------------
def pca_predict_boundary_safe(model: PCAError, test_X: np.ndarray,
                              run_starts: np.ndarray) -> np.ndarray:
    """Upstream PCAError.predict with smoothing applied per run segment.

    Identical to comparison/baselines/pca_error/model.py:predict except the
    5-box smoothing (the only temporally-mixing step) restarts at every run
    boundary, so no window ever mixes two simulation runs. The median-IQR
    normalization stays GLOBAL over the full test stream (upstream computes it
    over the whole array passed to predict — we pass the whole test set, same
    as the existing pipeline).
    """
    latent = model.pca.transform(test_X)
    recon = model.pca.inverse_transform(latent)
    per_feature_error = np.abs(test_X - recon)               # (n, f) paper-faithful

    # global median-IQR normalization (upstream data_utils.py:47-72, epsilon 1e-2)
    from scipy.stats import iqr
    med = np.median(per_feature_error, axis=0)
    spread = iqr(per_feature_error, axis=0)
    err = (per_feature_error - med) / (np.abs(spread) + 1e-2)

    # per-run 5-box smoothing (upstream loop semantics inside each run)
    smoothed = np.zeros_like(err)
    w = 5
    edges = list(run_starts) + [len(err)]
    for s, e in zip(edges[:-1], edges[1:]):
        seg = err[s:e]
        n = len(seg)
        out = np.zeros_like(seg)
        for i in range(w, n):
            lo, hi = i - w, min(i + w - 1, n)
            if hi > lo:
                out[i] = np.mean(seg[lo:hi], axis=0)
        smoothed[s:e] = out
    return smoothed.max(axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# partition machinery
# ---------------------------------------------------------------------------
def load_test():
    d = np.load(os.path.join(DATA_DIR, 'test_stream.npz'))
    with open(os.path.join(DATA_DIR, 'test_run_table.json')) as fp:
        run_table = json.load(fp)
    return d['X'], d['y'], d['fault_id'], d['run_boundaries'], run_table


def partition_eval(scores, y, run_table, fault_set, *, lite=False):
    """Slice whole runs (faults in fault_set + all FF runs), rebuild regions,
    run the standard metric stack on the partition-local arrays."""
    sel = [r for r in run_table if r['fault'] in fault_set or r['fault'] == 0]
    s_parts, y_parts, regions = [], [], []
    offset = 0
    for r in sel:
        n = r['end'] - r['start']
        s_parts.append(scores[r['start']:r['end']])
        y_parts.append(y[r['start']:r['end']])
        if r['fault'] != 0:
            regions.append(AnomalyRegion(start=offset + FAULT_ONSET_IDX,
                                         end=offset + n,
                                         anomaly_type=int(r['fault'])))
        offset += n
    sp = np.concatenate(s_parts)
    yp = np.concatenate(y_parts)
    return compute_all_metrics(sp, yp, regions, lite=lite), len(sp), len(regions)


def prefixed(metrics: dict, prefix: str) -> dict:
    return {f'{prefix}{k}': v for k, v in metrics.items()
            if not k.startswith('_') and k != 'epoch'}


# ---------------------------------------------------------------------------
# single experiment = (train stream, model) -> scores + partitioned metrics
# ---------------------------------------------------------------------------
def make_model(name: str):
    if name == 'pca_error':
        return PCAError(**MODEL_PARAMS['pca_error'])
    if name == 'sensor_range':
        return SensorRangeDeviation(**MODEL_PARAMS['sensor_range'])
    if name == 'l2_norm':
        return L2Norm(**MODEL_PARAMS['l2_norm'])
    if name == 'nn_distance':
        return NNDistance(**MODEL_PARAMS['nn_distance'])
    if name == 'random':
        return RandomBaseline(**MODEL_PARAMS['random'])
    raise ValueError(name)


def score_test(model_name, model, test_Xn, run_starts):
    if model_name == 'pca_error':
        return pca_predict_boundary_safe(model, test_Xn, run_starts)
    return model.predict(test_Xn)


def run_condition(train_npz, model_name: str, out_dir: Path,
                  test_X, test_y, fault_id, run_bounds, run_table,
                  fold_partitions: dict, experiment_name: str,
                  per_fault: bool, smoke: bool):
    """fold_partitions: {prefix: fault_set} evaluated on this score vector.
    '' (full) and 'exclhard_' are always evaluated.
    train_npz: filename in DATA_DIR, or a ready (n, F) train array (sweep use)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(train_npz, str):
        d = np.load(os.path.join(DATA_DIR, train_npz))
        train_X = d['X']
    else:
        train_X, train_npz = train_npz, '<in-memory>'

    t0 = time.time()
    mn, denom = fit_minmax(train_X)                     # label-free fit, full train
    train_Xn = apply_minmax(train_X, mn, denom)
    test_Xn = apply_minmax(test_X, mn, denom)
    model = make_model(model_name)
    model.fit(train_Xn)
    train_time = time.time() - t0

    run_starts = np.concatenate([[0], run_bounds])
    all_faults = sorted(set(r['fault'] for r in run_table) - {0})
    eval_sets = {'': set(all_faults), 'exclhard_': set(EXCLUDED_HARD)}
    eval_sets.update(fold_partitions)

    # Paper §4.2 convention (baseline_common.run_simple_baseline): the random
    # baseline reports mean±std over 5 independent draws; deterministic models
    # run once. scores.npz keeps the LAST draw (representative artifact).
    n_runs = 5 if model_name == 'random' else 1
    per_run_metrics, inference_time, eval_time = [], 0.0, 0.0
    scores = None
    part_info = {}
    for run_idx in range(n_runs):
        t0 = time.time()
        scores = score_test(model_name, model, test_Xn, run_starts)
        inference_time += time.time() - t0
        assert len(scores) == len(test_y) and np.isfinite(scores).all()

        m_all = {'epoch': 1}
        t0 = time.time()
        for prefix, fault_set in eval_sets.items():
            m, n_pts, n_reg = partition_eval(scores, test_y, run_table, fault_set,
                                             lite=smoke)
            m_all.update(prefixed(m, prefix) if prefix else
                         {k: v for k, v in m.items() if not k.startswith('_')})
            part_info[prefix or 'full'] = {'n_points': n_pts, 'n_regions': n_reg,
                                           'faults': sorted(fault_set)}
            if run_idx == n_runs - 1:
                print(f"    [{prefix or 'full':<10}] pak_auc_f1={m.get('pak_auc_f1', 0):.4f} "
                      f"prc={m.get('prc_auc', 0):.4f} vus_pr={m.get('vus_pr') or 0:.4f} "
                      f"({n_pts:,} pts, {n_reg} regions)")
        eval_time += time.time() - t0
        per_run_metrics.append(m_all)
        if n_runs > 1:
            print(f"    [draw {run_idx + 1}/{n_runs}] full pak_auc_f1="
                  f"{m_all.get('pak_auc_f1', 0):.4f}")
    metrics = _aggregate_run_metrics(per_run_metrics)

    pf = {}
    if per_fault and not smoke:
        # per-fault from the last draw (single-draw for random — noted in report)
        for f in all_faults:
            m, _, _ = partition_eval(scores, test_y, run_table, {f}, lite=True)
            pf[str(f)] = {k: m.get(k) for k in
                          ('pak_auc_f1', 'pak_auc_prc_auc', 'prc_auc', 'roc_auc',
                           'f1_t', 'aff_f1', 'pa_0_f1', 'pa_100_f1')}
        with open(out_dir / 'per_fault_metrics.json', 'w') as fp:
            json.dump(pf, fp, indent=2)

    save_scores_npz(out_dir, scores)
    save_epoch_metrics(out_dir, [metrics])
    save_metadata(out_dir, model_name, experiment_name, train_time, inference_time,
                  extra={'normalize_mode': NORMALIZE_MODE, 'n_runs': n_runs,
                         'model_params': {k: str(v) for k, v in MODEL_PARAMS[model_name].items()},
                         'train_npz': train_npz,
                         'eval_time': eval_time,
                         'boundary_policy': ('per-run smoothing' if model_name == 'pca_error'
                                             else 'pointwise (no temporal mixing)'),
                         'partitions': part_info})
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', default='all',
                    choices=['all'] + list(FOLDS) + ['ffonly'])
    ap.add_argument('--models', default='pca_error,sensor_range')
    ap.add_argument('--results-dir', default=None,
                    help='existing results dir to resume into')
    ap.add_argument('--smoke', action='store_true',
                    help='lite metrics (skip VUS), skip per-fault')
    args = ap.parse_args()

    models = args.models.split(',')
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        ts = time.strftime('%Y%m%d_%H%M%S')
        results_dir = Path(RESULTS_BASE) / f'{EXPERIMENT_NUMBER}_{ts}_tep_typegen_simple'
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results dir: {results_dir}")

    test_X, test_y, fault_id, run_bounds, run_table = load_test()
    print(f"Test stream: {test_X.shape}, anomaly {test_y.mean():.2%}, "
          f"runs {len(run_table)}")

    folds = list(FOLDS) if args.fold in ('all', 'ffonly') else [args.fold]
    do_contaminated = args.fold != 'ffonly'
    do_ffonly = args.fold in ('all', 'ffonly')

    # ---- contaminated folds (one train stream per fold) ----
    if do_contaminated:
        for fold in folds:
            parts = {'seen_': set(seen_faults(fold)),
                     'unseen_': set(unseen_faults(fold))}
            for model_name in models:
                print(f"\n=== {fold} / {model_name} (train: contaminated) ===")
                run_condition(f'train_{fold}.npz', model_name,
                              results_dir / fold / model_name,
                              test_X, test_y, fault_id, run_bounds, run_table,
                              parts, f'tep_typegen_{fold}',
                              per_fault=True, smoke=args.smoke)

    # ---- ffonly (clean-normal reference; fold-independent scores) ----
    if do_ffonly:
        parts = {}
        for fold in FOLDS:
            parts[f'{fold}_seen_'] = set(seen_faults(fold))
            parts[f'{fold}_unseen_'] = set(unseen_faults(fold))
        for model_name in models:
            print(f"\n=== ffonly / {model_name} (train: FF-only reference) ===")
            run_condition('train_ffonly.npz', model_name,
                          results_dir / 'ffonly' / model_name,
                          test_X, test_y, fault_id, run_bounds, run_table,
                          parts, 'tep_typegen_ffonly',
                          per_fault=True, smoke=args.smoke)

    print(f"\nAll done. Results: {results_dir}")


if __name__ == '__main__':
    main()
