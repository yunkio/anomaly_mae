"""
P13 — Iterative Score Refinement

Strategy: anomaly detection의 1차 result를 활용해 2차 refinement.

1. Round 1: standard div5.0_T1.5로 detect → candidate anomaly regions
2. Round 2: detected regions를 활용해 second-pass refinement:
   - Locally refine score around candidate region boundaries
   - Estimate per-candidate σ (per-burst FWHM)
   - Apply per-burst smoothing kernel
3. Round 3 (optional): converged refinement

Hypothesis: 1st pass의 noise를 2nd pass에서 reduce 가능. 단 detected region이 false positive이면
2nd pass에서 amplify될 위험.
"""
import sys
from pathlib import Path
import numpy as np
import json
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths

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


def detect_candidate_regions(score, percentile=90, min_run=3):
    """1st-pass: detect candidate anomaly regions."""
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

    regions = []
    for s, e in zip(starts, ends):
        if e - s >= min_run:
            regions.append((int(s), int(e)))
    return regions


def per_region_sigma_estimate(region, score):
    """Per-region σ estimate (peak width).
    Returns sigma based on FWHM."""
    s, e = region
    region_score = score[s:e]
    if len(region_score) < 3:
        return max((e - s) / 3.0, 1.0)

    # Find peak within region
    peak_local = np.argmax(region_score) + s
    peak_val = score[peak_local]

    # Half-max
    half_max = peak_val * 0.5
    # Search left and right
    left = peak_local
    while left > 0 and score[left] > half_max:
        left -= 1
    right = peak_local
    while right < len(score) - 1 and score[right] > half_max:
        right += 1
    fwhm = right - left
    return max(fwhm / 2.355, 1.0)


def iterative_refine_v1(base_unsmoothed, median_seg, n_iters=3):
    """V1: First-pass with global σ, then re-smooth around detected regions
    using per-region σ.

    각 iteration에서:
    - Score 계산 (smoothed + NLM)
    - Candidate regions detect
    - 각 region 주변에 per-region σ로 local refinement (overlapping)
    """
    sigma_global = max(median_seg / 5.0, 0.5)

    # Round 1
    score = gauss(base_unsmoothed, sigma_global)
    final = nlm_sigmoid_transform(score, T_factor=1.5)

    for it in range(n_iters):
        # Detect candidates
        candidates = detect_candidate_regions(final, percentile=85)
        if not candidates:
            break

        # For each candidate, refine score in expanded window
        refined = final.copy()
        for s, e in candidates:
            # Estimate per-region σ
            sigma_local = per_region_sigma_estimate((s, e), final)
            sigma_local = max(min(sigma_local, sigma_global * 2), 1.0)

            # Expand window
            expand = int(sigma_local * 3)
            ws = max(0, s - expand)
            we = min(len(base_unsmoothed), e + expand)

            # Re-smooth this window only with per-region σ
            local_base = base_unsmoothed[ws:we]
            if len(local_base) < 5:
                continue
            local_smoothed = gaussian_filter1d(local_base.astype(np.float64),
                                                 sigma=sigma_local, mode='reflect')
            # Blend (weighted by Gaussian envelope to avoid sharp edges)
            window_size = we - ws
            envelope = np.exp(-((np.arange(window_size) - window_size/2) ** 2) /
                              (2 * (window_size / 4) ** 2))
            envelope /= envelope.max()

            # Apply z-norm to local for fair blending
            local_smoothed_z = (local_smoothed - local_smoothed.mean()) / (local_smoothed.std() + 1e-9)
            global_z = (refined[ws:we] - refined[ws:we].mean()) / (refined[ws:we].std() + 1e-9)
            blended_z = envelope * local_smoothed_z + (1 - envelope) * global_z
            # Back to original scale
            refined[ws:we] = (blended_z * refined[ws:we].std() + refined[ws:we].mean())

        # Re-apply NLM on refined score
        final = nlm_sigmoid_transform(refined, T_factor=1.5)

    return final


def iterative_refine_v2(base_unsmoothed, median_seg, n_iters=2):
    """V2: Multi-σ ensemble guided by 1st pass.

    1st pass detects candidates → 2nd pass uses 다중 σ smoothing in candidate regions only.
    """
    sigma_global = max(median_seg / 5.0, 0.5)
    score_v1 = gauss(base_unsmoothed, sigma_global)
    final_v1 = nlm_sigmoid_transform(score_v1, T_factor=1.5)

    candidates = detect_candidate_regions(final_v1, percentile=85)
    if not candidates:
        return final_v1

    # Multi-σ smoothed scores
    sigmas = [max(sigma_global / 2, 0.5), sigma_global, sigma_global * 2]
    smoothed_variants = [gauss(base_unsmoothed, s) for s in sigmas]
    nlm_variants = [nlm_sigmoid_transform(s, T_factor=1.5) for s in smoothed_variants]
    z_variants = [zscore(s) for s in nlm_variants]

    # In candidates: take max of variants (more aggressive detection)
    # Outside: use median (more conservative)
    final = np.median(np.stack(z_variants), axis=0)
    is_candidate = np.zeros_like(final, dtype=bool)
    for s, e in candidates:
        is_candidate[s:e] = True
    max_variants = np.maximum.reduce(z_variants)
    final[is_candidate] = max_variants[is_candidate]

    return final


def iterative_refine_v3(base_unsmoothed, median_seg, n_iters=3):
    """V3: Self-consistency iteration. Refine σ based on detected region widths.

    각 iteration:
    1. Current σ로 smooth + detect candidates
    2. Estimate new σ as median of candidate widths / 3
    3. Update σ for next iteration
    Converge가 빠르므로 보통 3 iter면 충분.
    """
    sigma = max(median_seg / 5.0, 0.5)
    history = [sigma]

    for it in range(n_iters):
        smoothed = gauss(base_unsmoothed, sigma)
        final = nlm_sigmoid_transform(smoothed, T_factor=1.5)
        candidates = detect_candidate_regions(final, percentile=90, min_run=2)

        if not candidates:
            break

        widths = [e - s for s, e in candidates]
        valid_widths = [w for w in widths if w > 2]
        if not valid_widths:
            break

        new_sigma = max(np.median(valid_widths) / 3.0, 0.5)
        # Damping to avoid oscillation
        sigma = 0.5 * sigma + 0.5 * new_sigma
        history.append(sigma)

    # Final eval with converged σ
    smoothed = gauss(base_unsmoothed, sigma)
    final = nlm_sigmoid_transform(smoothed, T_factor=1.5)
    return final, history


def main():
    targets = iter_dataset_aliases()
    print(f"P13 — Iterative Score Refinement, {len(targets)} datasets")

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

        # Reference: div5.0_T1.5
        ref_score = nlm_sigmoid_transform(gauss(base_unsmoothed, max(median_seg/5, 0.5)), T_factor=1.5)
        ref_pak = pak_auc_f1(ref_score, ds.point_labels, ds.regions, ds.eval_mask)

        # V1: Per-region refinement
        v1_score = iterative_refine_v1(base_unsmoothed, median_seg, n_iters=3)
        v1_pak = pak_auc_f1(v1_score, ds.point_labels, ds.regions, ds.eval_mask)

        # V2: Multi-σ ensemble in candidates
        v2_score = iterative_refine_v2(base_unsmoothed, median_seg)
        v2_pak = pak_auc_f1(v2_score, ds.point_labels, ds.regions, ds.eval_mask)

        # V3: Self-consistency σ iteration
        v3_score, sigma_history = iterative_refine_v3(base_unsmoothed, median_seg, n_iters=5)
        v3_pak = pak_auc_f1(v3_score, ds.point_labels, ds.regions, ds.eval_mask)

        all_results[alias] = {
            'baseline_pak': baseline_pak,
            'ref_pak_div5_T15': ref_pak,
            'v1_per_region_pak': v1_pak,
            'v2_multi_sigma_pak': v2_pak,
            'v3_self_consistency_pak': v3_pak,
            'v3_sigma_history': sigma_history,
            'median_seg': median_seg,
        }

        if i % 10 == 0 or i == len(targets):
            print(f"  [{i:2d}/{len(targets)}] processed", flush=True)

    # Save
    out = Path(__file__).parent.parent / 'results' / 'P13_iterative_refinement.json'
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out}")

    # Analysis
    print("\n=== Aggregate Δ vs baseline ===")
    baseline_paks = [r['baseline_pak'] for r in all_results.values()]
    for method_key, label in [('ref_pak_div5_T15', 'ref div5_T1.5'),
                                ('v1_per_region_pak', 'V1 per-region refine'),
                                ('v2_multi_sigma_pak', 'V2 multi-σ ensemble'),
                                ('v3_self_consistency_pak', 'V3 σ self-consistency')]:
        scores = [r[method_key] for r in all_results.values()]
        deltas = np.array(scores) - np.array(baseline_paks)
        mean_d = float(deltas.mean())
        wins = int((deltas > 0).sum())
        losses = int((deltas < 0).sum())
        cata = int((deltas < -0.05).sum())
        p = wilcoxon_test(scores, baseline_paks, alternative='greater')
        print(f"  {label:<25s}: meanΔ={mean_d:+.4f}  W/L={wins:2d}/{losses:2d}  cata={cata}  p={p:.4f}")

    # Δ each method over ref
    print("\n=== Δ over ref div5_T1.5 ===")
    ref_paks = [r['ref_pak_div5_T15'] for r in all_results.values()]
    for method_key, label in [('v1_per_region_pak', 'V1'),
                                ('v2_multi_sigma_pak', 'V2'),
                                ('v3_self_consistency_pak', 'V3')]:
        scores = [r[method_key] for r in all_results.values()]
        deltas = np.array(scores) - np.array(ref_paks)
        mean_d = float(deltas.mean())
        wins = int((deltas > 0).sum())
        losses = int((deltas < 0).sum())
        p = wilcoxon_test(scores, ref_paks, alternative='greater')
        print(f"  {label} - ref: meanΔ={mean_d:+.4f}  W/L={wins:2d}/{losses:2d}  p={p:.4f}")

    # V3 σ convergence analysis
    print("\n=== V3 σ self-consistency convergence ===")
    converged_count = 0
    for alias, r in all_results.items():
        hist = r['v3_sigma_history']
        if len(hist) >= 3 and abs(hist[-1] - hist[-2]) < 1.0:
            converged_count += 1
    print(f"  Converged (|σ_n - σ_{{n-1}}| < 1): {converged_count}/{len(all_results)}")

    # Average σ trajectory
    max_len = max(len(r['v3_sigma_history']) for r in all_results.values())
    sigma_means = []
    for it in range(max_len):
        vals = [r['v3_sigma_history'][it] for r in all_results.values()
                if it < len(r['v3_sigma_history'])]
        sigma_means.append(np.mean(vals))
    print(f"  Mean σ trajectory: {[f'{s:.1f}' for s in sigma_means]}")


if __name__ == "__main__":
    main()
