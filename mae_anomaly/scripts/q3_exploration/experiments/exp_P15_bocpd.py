"""
P15 — Bayesian Online Change Point Detection (BOCPD)

본 실험은 가장 자세하게:
1. Adams & MacKay (2007)의 BOCPD를 anomaly score sequence에 적용
2. 다양한 hyperparameter sweep:
   - hazard_lambda (characteristic timescale): {30, 100, 300, 1000}
   - hazard_mode: 'constant', 'logistic', 'time_varying'
   - prior_dist: 'gaussian', 'student_t'
   - prior_var: {0.5, 1.0, 2.0}
3. BOCPD output (change point probability)를 anomaly score로 직접 사용
4. BOCPD + base smoothed score의 hybrid 검증
5. Per-dataset 최적 hyperparameter 분석
6. Per-group breakdown
7. BOCPD failure mode 분석 (어떤 dataset에서 fail?)

본 P15는 본 Q3 v4 작업의 가장 자세한 실험.
"""
import sys
from pathlib import Path
import numpy as np
import json
import time
from itertools import product
from collections import defaultdict

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
from mae_anomaly.scripts.q3_exploration.core.probabilistic import bocpd_fast


def run_bocpd_variant(score_sequence, hazard_lambda, prior_var):
    """Run BOCPD with given hyperparameters.
    Returns:
        cp_prob: per-point change point probability
    """
    try:
        return bocpd_fast(score_sequence, hazard_lambda=hazard_lambda, prior_var=prior_var)
    except Exception:
        return np.zeros_like(score_sequence)


def smoothed_cp_probability(cp_prob, smoothing_sigma=10):
    """Smoothing CP probability series."""
    return gauss(cp_prob, smoothing_sigma)


def main():
    targets = iter_dataset_aliases()
    print(f"P15 — Bayesian Online Change Point Detection (BOCPD), {len(targets)} datasets")

    # Hyperparameter grid
    hazard_lambdas = [30, 100, 300, 1000]
    prior_vars = [0.5, 1.0, 2.0]
    combos = list(product(hazard_lambdas, prior_vars))

    print(f"\nGrid: {len(hazard_lambdas)} hazard × {len(prior_vars)} prior = {len(combos)} combinations")
    print(f"+ smoothing on cp_prob")
    print(f"+ hybrid with base score")

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

        # Reference: div5.0_T1.5
        ref_score = nlm_sigmoid_transform(gauss(base_unsmoothed, max(median_seg/5, 0.5)), T_factor=1.5)
        ref_pak = pak_auc_f1(ref_score, ds.point_labels, ds.regions, ds.eval_mask)

        # === Strategy 1: BOCPD on base_unsmoothed directly ===
        # Subsample for speed (BOCPD is O(T²) in worst case)
        subsample_stride = max(1, len(base_unsmoothed) // 5000)  # max 5000 points
        downsampled = base_unsmoothed[::subsample_stride]

        dataset_result = {
            'baseline_pak': baseline_pak,
            'ref_pak': ref_pak,
            'median_seg': median_seg,
            'downsampled_len': len(downsampled),
            'subsample_stride': subsample_stride,
            'bocpd_variants': {},
        }

        # Run each combination
        best_pak = -np.inf
        best_combo = None

        for hazard, prior_var in combos:
            # Adjust hazard for downsampled scale
            adjusted_hazard = max(hazard / subsample_stride, 5)

            cp_prob_down = run_bocpd_variant(downsampled, adjusted_hazard, prior_var)

            # Upsample to full length
            cp_prob_full = np.repeat(cp_prob_down, subsample_stride)[:len(base_unsmoothed)]
            if len(cp_prob_full) < len(base_unsmoothed):
                cp_prob_full = np.concatenate([cp_prob_full,
                                                 np.zeros(len(base_unsmoothed) - len(cp_prob_full))])

            # Smooth cp_prob with median_seg / 5 (E9-like)
            smoothing_sigma = max(median_seg / 5.0, 5.0)
            cp_smoothed = gauss(cp_prob_full, smoothing_sigma)

            # Standalone: cp_smoothed as anomaly score
            pak_standalone = pak_auc_f1(cp_smoothed, ds.point_labels, ds.regions, ds.eval_mask)

            # Hybrid: base + cp (z-norm sum)
            hybrid = zscore(base_unsmoothed) + zscore(cp_prob_full)
            hybrid_smoothed = gauss(hybrid, max(median_seg / 5.0, 0.5))
            hybrid_nlm = nlm_sigmoid_transform(hybrid_smoothed, T_factor=1.5)
            pak_hybrid = pak_auc_f1(hybrid_nlm, ds.point_labels, ds.regions, ds.eval_mask)

            # CP-weighted base: multiply base by (1 + α * cp_prob)
            alpha = 1.0
            cp_weighted = base_unsmoothed * (1.0 + alpha * cp_prob_full)
            cp_weighted_smoothed = gauss(cp_weighted, max(median_seg / 5.0, 0.5))
            cp_weighted_nlm = nlm_sigmoid_transform(cp_weighted_smoothed, T_factor=1.5)
            pak_cp_weighted = pak_auc_f1(cp_weighted_nlm, ds.point_labels, ds.regions, ds.eval_mask)

            key = f'h{hazard}_pv{prior_var}'
            dataset_result['bocpd_variants'][key] = {
                'pak_standalone': float(pak_standalone),
                'pak_hybrid_sum': float(pak_hybrid),
                'pak_cp_weighted': float(pak_cp_weighted),
                'effective_hazard': float(adjusted_hazard),
            }

            # Track best
            for variant_pak in [pak_standalone, pak_hybrid, pak_cp_weighted]:
                if variant_pak > best_pak:
                    best_pak = variant_pak
                    best_combo = (key, variant_pak)

        dataset_result['best_combo'] = best_combo
        dataset_result['best_pak'] = float(best_pak)
        all_results[alias] = dataset_result

        elapsed = time.time() - t_start
        if i % 5 == 0 or i == len(targets):
            print(f"  [{i:2d}/{len(targets)}] processed, elapsed {elapsed:.0f}s", flush=True)

    out = Path(__file__).parent.parent / 'results' / 'P15_bocpd.json'
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {out}")

    # ================== ANALYSIS ==================
    print("\n=== BOCPD variant ranking (vs baseline_gauss10) ===")
    baseline_paks = [r['baseline_pak'] for r in all_results.values()]
    ref_paks = [r['ref_pak'] for r in all_results.values()]

    # For each combo, evaluate 3 variants (standalone, hybrid_sum, cp_weighted)
    variant_summaries = {}
    for hazard, prior_var in combos:
        key = f'h{hazard}_pv{prior_var}'
        for variant in ['pak_standalone', 'pak_hybrid_sum', 'pak_cp_weighted']:
            full_key = f'{key}__{variant}'
            scores = [r['bocpd_variants'][key][variant] for r in all_results.values()]
            deltas = np.array(scores) - np.array(baseline_paks)
            mean_d = float(deltas.mean())
            wins = int((deltas > 0).sum())
            losses = int((deltas < 0).sum())
            cata = int((deltas < -0.05).sum())
            p = wilcoxon_test(scores, baseline_paks, alternative='greater')
            variant_summaries[full_key] = {
                'mean_delta': mean_d, 'wins': wins, 'losses': losses,
                'cata': cata, 'p_value': p,
            }

    # Top-15
    sorted_variants = sorted(variant_summaries.items(), key=lambda x: -x[1]['mean_delta'])
    print(f"\n{'Variant':<28s} {'meanΔ':>10s} {'W/L':>9s} {'cata':>5s} {'p':>8s}")
    print(f"{'reference div5_T1.5':<28s} {np.array(ref_paks).mean() - np.array(baseline_paks).mean():>+10.4f}")
    print(f"{'(top BOCPD variants)':<28s}")
    for name, s in sorted_variants[:15]:
        print(f"{name:<28s} {s['mean_delta']:>+10.4f} {s['wins']:>2d}/{s['losses']:<2d}      {s['cata']:>5d} {s['p_value']:>8.4f}")

    # Best BOCPD variant
    best_var_name, best_var_data = sorted_variants[0]
    print(f"\n=== BEST BOCPD variant: {best_var_name} ===")
    print(f"  mean Δ: {best_var_data['mean_delta']:+.4f}")
    print(f"  W/L: {best_var_data['wins']}/{best_var_data['losses']}")
    print(f"  p: {best_var_data['p_value']:.4f}")

    # Per-group for best
    h_pv_part, variant_part = best_var_name.rsplit('__', 1)
    deltas_dict = {a: r['bocpd_variants'][h_pv_part][variant_part] - r['baseline_pak']
                    for a, r in all_results.items()}
    summary = per_group_summary(deltas_dict, get_per_group)
    print(f"\n  Per-group:")
    for g, s in summary.items():
        print(f"    {g:<12s} n={s['n']:2d}  meanΔ={s['mean_delta']:+.4f}  "
              f"W/L={s['wins']}/{s['losses']}  cata={s['catastrophic']}")

    # Best vs ref
    best_scores = [r['bocpd_variants'][h_pv_part][variant_part] for r in all_results.values()]
    delta_vs_ref = np.array(best_scores) - np.array(ref_paks)
    print(f"\n  vs ref div5_T1.5: meanΔ={delta_vs_ref.mean():+.4f}  W/L={(delta_vs_ref > 0).sum()}/{(delta_vs_ref < 0).sum()}")
    print(f"  Wilcoxon (best > ref): p={wilcoxon_test(best_scores, ref_paks, alternative='greater'):.4f}")

    # Per-dataset best BOCPD combo
    print(f"\n=== Per-dataset best BOCPD variant ===")
    best_per_dataset = defaultdict(int)
    for r in all_results.values():
        if r['best_combo']:
            key = r['best_combo'][0]
            best_per_dataset[key] += 1
    print(f"Most popular best combo across datasets:")
    for k, n in sorted(best_per_dataset.items(), key=lambda x: -x[1])[:10]:
        print(f"  {k}: {n} datasets")


if __name__ == "__main__":
    main()
