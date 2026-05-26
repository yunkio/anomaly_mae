"""
P16 — Extreme Value Theory (EVT) Based Tail Modeling

Anomaly = extreme observations from background distribution.
본 P16은 GPD (Generalized Pareto Distribution) fit on threshold exceedances:

1. Fit GPD on training portion exceedances above percentile threshold
2. For each test point, compute p-value = survival function of GPD
3. Anomaly score = -log(p_value)

Variants:
- threshold_percentile ∈ {90, 95, 97.5, 99}
- POT applied on different base scores (gauss10, div5_T1.5)
- Hybrid: base + POT score
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
from mae_anomaly.scripts.q3_exploration.core.probabilistic import (
    POTAnomalyScore, pot_anomaly_score,
)


def main():
    targets = iter_dataset_aliases()
    print(f"P16 — EVT-based Tail Modeling, {len(targets)} datasets")

    threshold_percentiles = [85, 90, 92.5, 95, 97.5, 99]
    base_score_types = ['gauss10', 'div5_T1.5']

    all_results = {}

    for i, (alias, swat) in enumerate(targets, 1):
        ds = DatasetScores.load(alias, swat)
        if ds is None:
            continue

        pt_r, pt_d, pt_s, pt_f = per_channel_points(ds, stride=21)
        base_unsmoothed = adaptive_combine(pt_r, pt_d, pt_f, use_fm=True)
        median_seg = median_anomaly_segment_length(ds.regions)

        # Base scores
        baseline = gauss(base_unsmoothed, 10)
        baseline_pak = pak_auc_f1(baseline, ds.point_labels, ds.regions, ds.eval_mask)

        ref_score = nlm_sigmoid_transform(gauss(base_unsmoothed, max(median_seg/5, 0.5)), T_factor=1.5)
        ref_pak = pak_auc_f1(ref_score, ds.point_labels, ds.regions, ds.eval_mask)

        dataset_result = {
            'baseline_pak': baseline_pak, 'ref_pak': ref_pak,
            'median_seg': median_seg, 'variants': {},
        }

        # For each base + threshold combination
        for base_type in base_score_types:
            base_score = baseline if base_type == 'gauss10' else ref_score

            for thr_pct in threshold_percentiles:
                # POT applied self-fit
                pot_score = pot_anomaly_score(base_score, threshold_percentile=thr_pct)

                # Variant A: POT alone
                pot_smoothed = gauss(pot_score, max(median_seg / 5.0, 0.5))
                pak_pot_alone = pak_auc_f1(pot_smoothed, ds.point_labels, ds.regions, ds.eval_mask)

                # Variant B: base + POT (z-norm sum)
                hybrid = zscore(base_score) + zscore(pot_score)
                pak_hybrid = pak_auc_f1(hybrid, ds.point_labels, ds.regions, ds.eval_mask)

                # Variant C: POT applied on base then NLM
                pot_nlm = nlm_sigmoid_transform(pot_smoothed, T_factor=1.5)
                pak_pot_nlm = pak_auc_f1(pot_nlm, ds.point_labels, ds.regions, ds.eval_mask)

                key = f'{base_type}_thr{thr_pct}'
                dataset_result['variants'][key] = {
                    'pot_alone': float(pak_pot_alone),
                    'hybrid': float(pak_hybrid),
                    'pot_nlm': float(pak_pot_nlm),
                }

        all_results[alias] = dataset_result

        if i % 10 == 0 or i == len(targets):
            print(f"  [{i:2d}/{len(targets)}] processed", flush=True)

    out = Path(__file__).parent.parent / 'results' / 'P16_evt_tail.json'
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out}")

    # Analysis
    print("\n=== EVT variants ranking (vs baseline) ===")
    baseline_paks = [r['baseline_pak'] for r in all_results.values()]
    ref_paks = [r['ref_pak'] for r in all_results.values()]

    delta_ref = np.array(ref_paks) - np.array(baseline_paks)
    print(f"reference ref div5_T1.5 baseline:    meanΔ={delta_ref.mean():+.4f}\n")

    variants_summary = {}
    for base_type in base_score_types:
        for thr_pct in threshold_percentiles:
            key = f'{base_type}_thr{thr_pct}'
            for variant in ['pot_alone', 'hybrid', 'pot_nlm']:
                full_key = f'{key}__{variant}'
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

    sorted_variants = sorted(variants_summary.items(), key=lambda x: -x[1]['mean_delta'])
    print(f"{'Variant':<40s} {'meanΔ':>10s} {'W/L':>9s} {'cata':>5s} {'p':>8s}")
    for name, s in sorted_variants[:15]:
        print(f"{name:<40s} {s['mean_delta']:>+10.4f} {s['wins']:>2d}/{s['losses']:<2d}      {s['cata']:>5d} {s['p_value']:>8.4f}")

    # Best
    best_name, best_data = sorted_variants[0]
    print(f"\n=== BEST EVT variant: {best_name} ===")
    print(f"  mean Δ: {best_data['mean_delta']:+.4f}")
    print(f"  W/L: {best_data['wins']}/{best_data['losses']}")
    print(f"  p: {best_data['p_value']:.4f}")

    # Per-group
    base_thr, variant = best_name.rsplit('__', 1)
    deltas_dict = {a: r['variants'][base_thr][variant] - r['baseline_pak']
                    for a, r in all_results.items()}
    summary = per_group_summary(deltas_dict, get_per_group)
    print(f"\n  Per-group:")
    for g, s in summary.items():
        print(f"    {g:<12s} n={s['n']:2d}  meanΔ={s['mean_delta']:+.4f}  "
              f"W/L={s['wins']}/{s['losses']}  cata={s['catastrophic']}")


if __name__ == "__main__":
    main()
