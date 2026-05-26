"""
P15-lite — BOCPD (simplified, fast version)

BOCPD가 O(T²) → 큰 dataset에서 너무 slow. Lite version:
- 2 hazard × 2 prior_var = 4 combinations only
- Aggressive subsample (max 1000 points)
- 1 hybrid mode (z-sum with ref)
- Skip per-dataset best discovery
"""
import sys
from pathlib import Path
import numpy as np
import json
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from mae_anomaly.scripts.q3_exploration.core.data import (
    DatasetScores, iter_dataset_aliases, median_anomaly_segment_length, get_per_group
)
from mae_anomaly.scripts.q3_exploration.core.scoring import (
    per_channel_points, adaptive_combine, gauss, zscore
)
from mae_anomaly.scripts.q3_exploration.core.evaluation import (
    pak_auc_f1, wilcoxon_test, per_group_summary
)
from mae_anomaly.scripts.q3_exploration.core.postprocess import nlm_sigmoid_transform


def bocpd_minimal(signal, hazard_lambda=100, prior_var=1.0, max_run_length=200):
    """Minimal BOCPD with aggressive max_run_length cap.
    Returns: cp_prob (T,)
    """
    from scipy.stats import norm
    T = len(signal)
    s_mean, s_std = signal.mean(), signal.std() + 1e-9
    x = (signal - s_mean) / s_std

    hazard = 1.0 / hazard_lambda
    max_r = min(T, max_run_length)

    log_R = np.full(max_r + 1, -1e30)
    log_R[0] = 0.0
    cp_prob = np.zeros(T)

    mean_acc = np.zeros(max_r + 2)
    n_acc = np.zeros(max_r + 2, dtype=np.float64)

    for t in range(T):
        xt = x[t]

        # Use only first max_r+1 entries
        valid = log_R > -1e10
        if not valid.any():
            cp_prob[t] = 0
            continue

        # Predictive likelihood
        # For r=0 (or n_acc=0): N(0, prior_var)
        # Else: N(running mean, posterior_var)
        log_pred = np.full(max_r + 1, -1e30)
        for r in range(max_r + 1):
            if not valid[r]:
                continue
            if n_acc[r] < 1:
                log_pred[r] = norm.logpdf(xt, loc=0.0, scale=np.sqrt(prior_var))
            else:
                pm = mean_acc[r] * n_acc[r] / (1 + n_acc[r])
                pv = prior_var * (1.0 + 1.0 / (n_acc[r] + 1))
                log_pred[r] = norm.logpdf(xt, loc=pm, scale=np.sqrt(pv))

        log_growth = log_R + log_pred + np.log(max(1.0 - hazard, 1e-15))
        valid_cp = valid & (log_pred > -1e10)
        if valid_cp.any():
            log_cp = np.logaddexp.reduce(log_R[valid_cp] + log_pred[valid_cp] + np.log(max(hazard, 1e-15)))
        else:
            log_cp = -1e30

        new_log_R = np.full(max_r + 1, -1e30)
        new_log_R[0] = log_cp
        new_log_R[1:max_r + 1] = log_growth[:max_r]

        norm_const = np.logaddexp.reduce(new_log_R[new_log_R > -1e10])
        new_log_R -= norm_const
        log_R = new_log_R

        cp_prob[t] = np.exp(log_R[0])

        # Update sufficient stats
        new_mean = np.zeros(max_r + 2)
        new_n = np.zeros(max_r + 2)
        new_mean[1:max_r + 2] = (mean_acc[:max_r + 1] * n_acc[:max_r + 1] + xt) / (n_acc[:max_r + 1] + 1)
        new_n[1:max_r + 2] = n_acc[:max_r + 1] + 1
        mean_acc, n_acc = new_mean, new_n

    return cp_prob


def main():
    targets = iter_dataset_aliases()
    print(f"P15-lite — BOCPD (simplified), {len(targets)} datasets")

    # Reduced grid for speed
    configs = [
        (100, 1.0),   # baseline
        (300, 1.0),   # longer characteristic
        (100, 2.0),   # higher prior var
    ]

    all_results = {}
    t_start = time.time()

    for i, (alias, swat) in enumerate(targets, 1):
        ds = DatasetScores.load(alias, swat)
        if ds is None:
            continue
        pt_r, pt_d, pt_s, pt_f = per_channel_points(ds, stride=21)
        base_unsmoothed = adaptive_combine(pt_r, pt_d, pt_f, use_fm=True)
        median_seg = median_anomaly_segment_length(ds.regions)

        baseline = gauss(base_unsmoothed, 10)
        baseline_pak = pak_auc_f1(baseline, ds.point_labels, ds.regions, ds.eval_mask)
        ref_score = nlm_sigmoid_transform(gauss(base_unsmoothed, max(median_seg/5, 0.5)), T_factor=1.5)
        ref_pak = pak_auc_f1(ref_score, ds.point_labels, ds.regions, ds.eval_mask)

        # Aggressive subsample (target 800 points)
        target_points = 800
        subsample_stride = max(1, len(base_unsmoothed) // target_points)
        downsampled = base_unsmoothed[::subsample_stride]
        n_down = len(downsampled)

        dataset_result = {
            'baseline_pak': baseline_pak,
            'ref_pak': ref_pak,
            'median_seg': median_seg,
            'n_down': n_down,
            'variants': {},
        }

        t_ds = time.time()
        for hazard, pv in configs:
            adj_hazard = max(hazard / subsample_stride, 5)
            cp_down = bocpd_minimal(downsampled, hazard_lambda=adj_hazard, prior_var=pv,
                                      max_run_length=min(300, n_down))

            # Upsample
            cp_full = np.repeat(cp_down, subsample_stride)[:len(base_unsmoothed)]
            if len(cp_full) < len(base_unsmoothed):
                cp_full = np.concatenate([cp_full,
                                           np.zeros(len(base_unsmoothed) - len(cp_full))])

            # Smooth
            cp_smoothed = gauss(cp_full, max(median_seg / 5.0, 0.5))

            # Hybrid: z-sum with ref
            hybrid = zscore(ref_score) + zscore(cp_smoothed)
            pak_hybrid = pak_auc_f1(hybrid, ds.point_labels, ds.regions, ds.eval_mask)

            # CP weighted base
            cp_weighted = base_unsmoothed * (1.0 + cp_full)
            cp_weighted_smoothed = gauss(cp_weighted, max(median_seg / 5.0, 0.5))
            cp_weighted_nlm = nlm_sigmoid_transform(cp_weighted_smoothed, T_factor=1.5)
            pak_cp_weighted = pak_auc_f1(cp_weighted_nlm, ds.point_labels, ds.regions, ds.eval_mask)

            # Standalone smoothed cp_prob
            pak_standalone = pak_auc_f1(cp_smoothed, ds.point_labels, ds.regions, ds.eval_mask)

            key = f'h{hazard}_pv{pv}'
            dataset_result['variants'][key] = {
                'standalone': float(pak_standalone),
                'hybrid_sum': float(pak_hybrid),
                'cp_weighted': float(pak_cp_weighted),
            }

        all_results[alias] = dataset_result
        elapsed = time.time() - t_start
        ds_elapsed = time.time() - t_ds
        if i % 3 == 0 or i == len(targets):
            print(f"  [{i:2d}/{len(targets)}] {alias}: ds_time={ds_elapsed:.1f}s  total={elapsed:.0f}s", flush=True)

    out = Path(__file__).parent.parent / 'results' / 'P15_bocpd_lite.json'
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out}")

    # Analysis
    baseline_paks = [r['baseline_pak'] for r in all_results.values()]
    ref_paks = [r['ref_pak'] for r in all_results.values()]
    print(f"\n=== Aggregate ===")
    print(f"reference ref div5_T1.5: meanΔ={np.mean(ref_paks) - np.mean(baseline_paks):+.4f}")

    variants_summary = {}
    for hazard, pv in configs:
        key = f'h{hazard}_pv{pv}'
        for variant in ['standalone', 'hybrid_sum', 'cp_weighted']:
            full_key = f'{key}_{variant}'
            scores = [r['variants'][key][variant] for r in all_results.values()]
            deltas = np.array(scores) - np.array(baseline_paks)
            mean_d = float(deltas.mean())
            wins = int((deltas > 0).sum())
            losses = int((deltas < 0).sum())
            cata = int((deltas < -0.05).sum())
            p = wilcoxon_test(scores, baseline_paks, alternative='greater')
            variants_summary[full_key] = {
                'mean_delta': mean_d, 'wins': wins, 'losses': losses,
                'cata': cata, 'p_value': p,
            }

    sorted_v = sorted(variants_summary.items(), key=lambda x: -x[1]['mean_delta'])
    print(f"\n{'Variant':<32s} {'meanΔ':>10s} {'W/L':>9s} {'cata':>5s} {'p':>8s}")
    for name, s in sorted_v:
        print(f"{name:<32s} {s['mean_delta']:>+10.4f} {s['wins']:>2d}/{s['losses']:<2d}      {s['cata']:>5d} {s['p_value']:>8.4f}")

    best_name, best_data = sorted_v[0]
    print(f"\n=== BEST BOCPD: {best_name} ===")
    print(f"  mean Δ: {best_data['mean_delta']:+.4f}  vs ref: {best_data['mean_delta'] - (np.mean(ref_paks) - np.mean(baseline_paks)):+.4f}")


if __name__ == "__main__":
    main()
