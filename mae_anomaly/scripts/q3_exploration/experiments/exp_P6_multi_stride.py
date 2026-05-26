"""
P6 — Multi-Stride Aggregation Ensemble

현재 baseline은 stride=21에서 patch contributions를 subsample (LoO inference의 50 forwards 중 1/21 사용).
다른 stride에서 추출한 score가 다른 perspective를 줄 수 있다.

Strategy:
- Stride ∈ {1, 7, 14, 21, 42} 각각에서 score 추출
- 각 stride의 score를 z-norm 후 ensemble
- 평균 / 최대 / robust mean 등 ensemble strategy 검증

Note: 큰 stride는 less data per point (noise 증가), 작은 stride는 over-smoothing 위험.
"""
import sys
from pathlib import Path
import numpy as np
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from mae_anomaly.scripts.q3_exploration.core.data import (
    DatasetScores, iter_dataset_aliases, median_anomaly_segment_length, get_per_group
)
from mae_anomaly.scripts.q3_exploration.core.scoring import (
    per_channel_points, adaptive_combine, gauss, zscore, aggregate_K50, stride_subsample
)
from mae_anomaly.scripts.q3_exploration.core.evaluation import (
    pak_auc_f1, wilcoxon_test, per_group_summary
)
from mae_anomaly.scripts.q3_exploration.core.postprocess import nlm_sigmoid_transform


def per_channel_points_at_stride(ds, stride):
    """Stride parameter화."""
    r_s, ws_s = stride_subsample(ds.recon, ds.window_start_indices, stride)
    d_s, _ = stride_subsample(ds.disc, ds.window_start_indices, stride)
    f_s, _ = stride_subsample(ds.fm, ds.window_start_indices, stride)
    pt_r = aggregate_K50(r_s, ws_s, ds.num_patches, ds.patch_size, ds.total_length)
    pt_d = aggregate_K50(d_s, ws_s, ds.num_patches, ds.patch_size, ds.total_length)
    pt_f = aggregate_K50(f_s, ws_s, ds.num_patches, ds.patch_size, ds.total_length)
    return pt_r, pt_d, pt_f


def make_score_at_stride(ds, stride, sigma_factor, use_nlm=True, t_factor=1.5):
    """At given stride, compute final score."""
    pt_r, pt_d, pt_f = per_channel_points_at_stride(ds, stride)
    base = adaptive_combine(pt_r, pt_d, pt_f, use_fm=True)
    median_seg = median_anomaly_segment_length(ds.regions)
    sigma = max(median_seg / sigma_factor, 0.5)
    smoothed = gauss(base, sigma)
    if use_nlm:
        return nlm_sigmoid_transform(smoothed, T_factor=t_factor)
    return smoothed


def main():
    targets = iter_dataset_aliases()
    print(f"P6 — Multi-Stride Ensemble, {len(targets)} datasets")

    strides = [1, 7, 14, 21, 42, 63]
    sigma_factor = 5.0  # P2 best: div5.0
    t_factor = 1.5

    all_results = {}

    for i, (alias, swat) in enumerate(targets, 1):
        ds = DatasetScores.load(alias, swat)
        if ds is None:
            continue

        median_seg = median_anomaly_segment_length(ds.regions)

        # Baseline (stride=21, gauss10)
        pt_r, pt_d, pt_f = per_channel_points_at_stride(ds, 21)
        base = adaptive_combine(pt_r, pt_d, pt_f, use_fm=True)
        baseline = gauss(base, 10)
        baseline_pak = pak_auc_f1(baseline, ds.point_labels, ds.regions, ds.eval_mask)

        # Per-stride scores
        per_stride_scores = {}
        per_stride_paks = {}
        for s in strides:
            score = make_score_at_stride(ds, s, sigma_factor, use_nlm=True, t_factor=t_factor)
            per_stride_scores[s] = score
            per_stride_paks[s] = pak_auc_f1(score, ds.point_labels, ds.regions, ds.eval_mask)

        # Reference: div5.0_T1.5 at stride=21
        ref_pak = per_stride_paks[21]

        # Ensemble variants:
        # E1: z-norm mean of all strides
        scores_array = np.stack([zscore(per_stride_scores[s]) for s in strides], axis=0)
        e1_mean = scores_array.mean(axis=0)
        e1_pak = pak_auc_f1(e1_mean, ds.point_labels, ds.regions, ds.eval_mask)

        # E2: weighted mean (weight inverse of stride: smaller stride = more weight)
        weights = 1.0 / np.array(strides, dtype=np.float64)
        weights /= weights.sum()
        e2_weighted = (weights.reshape(-1, 1) * scores_array).sum(axis=0)
        e2_pak = pak_auc_f1(e2_weighted, ds.point_labels, ds.regions, ds.eval_mask)

        # E3: maximum across strides (extreme detection)
        e3_max = scores_array.max(axis=0)
        e3_pak = pak_auc_f1(e3_max, ds.point_labels, ds.regions, ds.eval_mask)

        # E4: median (robust)
        e4_median = np.median(scores_array, axis=0)
        e4_pak = pak_auc_f1(e4_median, ds.point_labels, ds.regions, ds.eval_mask)

        # E5: trimmed mean (drop top and bottom stride)
        sorted_s = np.sort(scores_array, axis=0)
        e5_trim = sorted_s[1:-1].mean(axis=0)
        e5_pak = pak_auc_f1(e5_trim, ds.point_labels, ds.regions, ds.eval_mask)

        # E6: small-stride only (1, 7, 14) - high resolution
        small_scores = np.stack([zscore(per_stride_scores[s]) for s in [1, 7, 14]], axis=0)
        e6_small_mean = small_scores.mean(axis=0)
        e6_pak = pak_auc_f1(e6_small_mean, ds.point_labels, ds.regions, ds.eval_mask)

        # E7: ensemble of 21 + 14 (close strides)
        close_scores = np.stack([zscore(per_stride_scores[s]) for s in [14, 21]], axis=0)
        e7_close = close_scores.mean(axis=0)
        e7_pak = pak_auc_f1(e7_close, ds.point_labels, ds.regions, ds.eval_mask)

        all_results[alias] = {
            'baseline_pak': baseline_pak,
            'ref_pak_stride21': ref_pak,
            'per_stride_paks': per_stride_paks,
            'e1_mean_pak': e1_pak,
            'e2_weighted_pak': e2_pak,
            'e3_max_pak': e3_pak,
            'e4_median_pak': e4_pak,
            'e5_trim_pak': e5_pak,
            'e6_small_mean_pak': e6_pak,
            'e7_close_pak': e7_pak,
        }

        if i % 10 == 0 or i == len(targets):
            print(f"  [{i:2d}/{len(targets)}] processed", flush=True)

    # Save
    out = Path(__file__).parent.parent / 'results' / 'P6_multi_stride.json'
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out}")

    # Analysis
    print("\n=== Per-stride score (div5.0_T1.5 base) ===")
    for s in strides:
        deltas = [r['per_stride_paks'][s] - r['baseline_pak'] for r in all_results.values()]
        baseline_s = [r['baseline_pak'] for r in all_results.values()]
        method_s = [r['per_stride_paks'][s] for r in all_results.values()]
        mean_d = np.mean(deltas)
        wins = sum(1 for d in deltas if d > 0)
        losses = sum(1 for d in deltas if d < 0)
        cata = sum(1 for d in deltas if d < -0.05)
        p = wilcoxon_test(method_s, baseline_s, alternative='greater')
        print(f"  stride={s:>2d}: meanΔ={mean_d:+.4f}  W/L={wins:2d}/{losses:2d}  cata={cata}  p={p:.4f}")

    print("\n=== Multi-stride ensembles ===")
    ensembles = ['e1_mean', 'e2_weighted', 'e3_max', 'e4_median', 'e5_trim', 'e6_small_mean', 'e7_close']
    for e in ensembles:
        key = f'{e}_pak'
        deltas = [r[key] - r['baseline_pak'] for r in all_results.values()]
        baseline_s = [r['baseline_pak'] for r in all_results.values()]
        method_s = [r[key] for r in all_results.values()]
        mean_d = np.mean(deltas)
        wins = sum(1 for d in deltas if d > 0)
        losses = sum(1 for d in deltas if d < 0)
        cata = sum(1 for d in deltas if d < -0.05)
        p = wilcoxon_test(method_s, baseline_s, alternative='greater')
        print(f"  {e:<18s}: meanΔ={mean_d:+.4f}  W/L={wins:2d}/{losses:2d}  cata={cata}  p={p:.4f}")


if __name__ == "__main__":
    main()
