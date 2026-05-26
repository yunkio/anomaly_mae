"""
P14 — Anomaly Boundary Refinement

Detected anomaly region의 정확한 start/end boundary를 refine.
Often anomaly boundary는 score sequence의 sharp transition에 있음.

Strategy:
1. 1st pass detection → coarse regions
2. Each region의 boundary 주변에서:
   - Gradient analysis (highest 1st derivative point near boundary)
   - Bayesian change point
   - Adjusted threshold sweep within local window
3. PA-K 평가에서 정확한 boundary가 detection이 더 stable

Note: pak_auc_f1 자체는 PA-K rule로 boundary precision에 robust.
본 실험은 boundary refinement가 다른 metric (best F1, affiliation)에 더 큰 영향을 줄지 검증.
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
    per_channel_points, adaptive_combine, gauss, zscore
)
from mae_anomaly.scripts.q3_exploration.core.evaluation import (
    pak_auc_f1, wilcoxon_test, per_group_summary
)
from mae_anomaly.scripts.q3_exploration.core.postprocess import nlm_sigmoid_transform


def refine_boundaries_v1_gradient(score, candidate_regions, search_radius=10):
    """V1: Boundary refinement via local gradient maxima.

    각 candidate region의 start/end 주변에서 가장 큰 score gradient point를 새 boundary로.
    """
    refined_score = score.copy()
    # For each candidate region, sharpen boundary
    for s, e in candidate_regions:
        # Refine start
        lo = max(0, s - search_radius)
        hi = min(len(score), s + search_radius)
        if hi > lo + 1:
            local_grad = np.abs(np.diff(score[lo:hi]))
            if len(local_grad) > 0:
                new_start = lo + np.argmax(local_grad) + 1
                # Boost score around new start
                refined_score[new_start:new_start+3] *= 1.1

        # Refine end
        lo = max(0, e - search_radius)
        hi = min(len(score), e + search_radius)
        if hi > lo + 1:
            local_grad = np.abs(np.diff(score[lo:hi]))
            if len(local_grad) > 0:
                new_end = lo + np.argmax(local_grad)
                refined_score[new_end-3:new_end] *= 1.1

    return refined_score


def refine_boundaries_v2_local_threshold(score, candidate_regions, search_radius=20):
    """V2: Local threshold adjustment.

    Each candidate region 주변에서 local statistics 기반으로 더 정확한 threshold 결정.
    """
    refined_score = score.copy()

    for s, e in candidate_regions:
        # Local window around region
        lo = max(0, s - search_radius)
        hi = min(len(score), e + search_radius)
        local = score[lo:hi]
        if len(local) < 5:
            continue

        # Local mean + 2*std as boundary signal
        local_mean = local.mean()
        local_std = local.std()
        local_threshold = local_mean + 2 * local_std

        # Boost points above local threshold
        boost_mask = (refined_score[lo:hi] > local_threshold)
        refined_score[lo:hi] = np.where(boost_mask,
                                          refined_score[lo:hi] * 1.05,
                                          refined_score[lo:hi])

    return refined_score


def refine_boundaries_v3_dilation(score, candidate_regions, dilation=3):
    """V3: Morphological dilation - extend score peaks slightly outward.

    Captures the fact that anomaly boundaries are often gradual.
    """
    from scipy.ndimage import maximum_filter1d
    refined_score = score.copy()

    if not candidate_regions:
        return refined_score

    # Region mask
    region_mask = np.zeros(len(score), dtype=bool)
    for s, e in candidate_regions:
        region_mask[s:e] = True

    # Dilate the region's max value outward
    dilated_score = maximum_filter1d(score, size=2 * dilation + 1)

    # Where dilated > original AND outside region (boundary extension)
    refined_score = np.where(dilated_score > refined_score, dilated_score, refined_score)
    return refined_score


def detect_candidate_regions(score, percentile=90, min_run=3):
    threshold = np.percentile(score, percentile)
    binary = score > threshold
    if binary.sum() == 0:
        return []
    diffs = np.diff(binary.astype(int))
    starts = np.where(diffs == 1)[0] + 1
    ends = np.where(diffs == -1)[0] + 1
    if binary[0]:
        starts = np.concatenate([[0], starts])
    if binary[-1]:
        ends = np.concatenate([ends, [len(binary)]])
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e - s >= min_run]


def main():
    targets = iter_dataset_aliases()
    print(f"P14 — Boundary Refinement, {len(targets)} datasets")

    all_results = {}

    for i, (alias, swat) in enumerate(targets, 1):
        ds = DatasetScores.load(alias, swat)
        if ds is None:
            continue
        pt_r, pt_d, pt_s, pt_f = per_channel_points(ds, stride=21)
        base_unsmoothed = adaptive_combine(pt_r, pt_d, pt_f, use_fm=True)
        median_seg = median_anomaly_segment_length(ds.regions)
        base_smoothed = gauss(base_unsmoothed, 10)
        baseline_pak = pak_auc_f1(base_smoothed, ds.point_labels, ds.regions, ds.eval_mask)

        # Reference
        ref_score = nlm_sigmoid_transform(gauss(base_unsmoothed, max(median_seg/5, 0.5)), T_factor=1.5)
        ref_pak = pak_auc_f1(ref_score, ds.point_labels, ds.regions, ds.eval_mask)

        # Detect candidate regions
        candidates = detect_candidate_regions(ref_score, percentile=90)

        # Apply refinements
        v1_score = refine_boundaries_v1_gradient(ref_score, candidates)
        v1_pak = pak_auc_f1(v1_score, ds.point_labels, ds.regions, ds.eval_mask)

        v2_score = refine_boundaries_v2_local_threshold(ref_score, candidates)
        v2_pak = pak_auc_f1(v2_score, ds.point_labels, ds.regions, ds.eval_mask)

        v3_score = refine_boundaries_v3_dilation(ref_score, candidates, dilation=3)
        v3_pak = pak_auc_f1(v3_score, ds.point_labels, ds.regions, ds.eval_mask)

        v3_d5_score = refine_boundaries_v3_dilation(ref_score, candidates, dilation=5)
        v3_d5_pak = pak_auc_f1(v3_d5_score, ds.point_labels, ds.regions, ds.eval_mask)

        all_results[alias] = {
            'baseline_pak': baseline_pak,
            'ref_pak': ref_pak,
            'v1_gradient_pak': v1_pak,
            'v2_local_thr_pak': v2_pak,
            'v3_dilate3_pak': v3_pak,
            'v3_dilate5_pak': v3_d5_pak,
            'n_candidates': len(candidates),
        }

        if i % 10 == 0 or i == len(targets):
            print(f"  [{i:2d}/{len(targets)}] processed", flush=True)

    out = Path(__file__).parent.parent / 'results' / 'P14_boundary_refinement.json'
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out}")

    print("\n=== Aggregate Δ vs baseline ===")
    baseline_paks = [r['baseline_pak'] for r in all_results.values()]
    for method_key, label in [('ref_pak', 'ref div5_T1.5'),
                                ('v1_gradient_pak', 'V1 gradient boost'),
                                ('v2_local_thr_pak', 'V2 local threshold'),
                                ('v3_dilate3_pak', 'V3 dilation d=3'),
                                ('v3_dilate5_pak', 'V3 dilation d=5')]:
        scores = [r[method_key] for r in all_results.values()]
        deltas = np.array(scores) - np.array(baseline_paks)
        mean_d = float(deltas.mean())
        wins = int((deltas > 0).sum())
        losses = int((deltas < 0).sum())
        cata = int((deltas < -0.05).sum())
        p = wilcoxon_test(scores, baseline_paks, alternative='greater')
        print(f"  {label:<25s}: meanΔ={mean_d:+.4f}  W/L={wins:2d}/{losses:2d}  cata={cata}  p={p:.4f}")

    print("\n=== Δ over ref div5_T1.5 ===")
    ref_paks = [r['ref_pak'] for r in all_results.values()]
    for method_key, label in [('v1_gradient_pak', 'V1'),
                                ('v2_local_thr_pak', 'V2'),
                                ('v3_dilate3_pak', 'V3-d3'),
                                ('v3_dilate5_pak', 'V3-d5')]:
        scores = [r[method_key] for r in all_results.values()]
        deltas = np.array(scores) - np.array(ref_paks)
        mean_d = float(deltas.mean())
        wins = int((deltas > 0).sum())
        losses = int((deltas < 0).sum())
        p = wilcoxon_test(scores, ref_paks, alternative='greater')
        print(f"  {label} - ref: meanΔ={mean_d:+.4f}  W/L={wins:2d}/{losses:2d}  p={p:.4f}")


if __name__ == "__main__":
    main()
