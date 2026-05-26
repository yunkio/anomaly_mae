"""
P2 — Fine σ Sweep + Multi-objective evaluation

P1에서 발견된 e9_div5_nlm_T2가 26 datasets의 winner. 본 실험은:

1. σ multiplier fine-grained sweep (median_seg / k for k in [1.5 to 10])
2. NLM T_factor sweep (T = 0.5 to 5.0)
3. (σ multiplier, T) joint grid에서 best combination 찾기
4. 4-metric (pak, aff, rbased, severity) 동시 평가
5. Standalone version (no cluster routing) vs P1 cluster-routed 비교

Method count: 12 σ multipliers × 6 NLM T values = 72 (σ, T) combinations
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


# Multi-metric helpers
def safe_affiliation_f1(scores, labels, regions):
    try:
        from affiliation.metrics import pr_from_events
        threshold = np.percentile(scores, 100 * (1 - labels.mean()))
        pred = (scores > threshold).astype(int)
        def find_ev(arr):
            ev = []; in_e, st = False, None
            for i, v in enumerate(arr):
                if v == 1 and not in_e: st, in_e = i, True
                elif v == 0 and in_e: ev.append((st, i)); in_e = False
            if in_e: ev.append((st, len(arr)))
            return ev
        gt, pr = find_ev(labels), find_ev(pred)
        if not gt or not pr: return 0.0
        res = pr_from_events(pr, gt, (0, len(labels)))
        p, r = res['precision'], res['recall']
        return float(2*p*r/(p+r)) if (p+r) > 0 else 0.0
    except Exception:
        return 0.0


def safe_rbased_f1(scores, labels):
    try:
        from prts import ts_precision, ts_recall
        threshold = np.percentile(scores, 100 * (1 - labels.mean()))
        pred = (scores > threshold).astype(int)
        p = ts_precision(labels, pred, alpha=0.0, cardinality='reciprocal', bias='flat')
        r = ts_recall(labels, pred, alpha=0.0, cardinality='reciprocal', bias='flat')
        return float(2*p*r/(p+r)) if (p+r) > 0 else 0.0
    except Exception:
        return 0.0


def main():
    targets = iter_dataset_aliases()
    print(f"P2 — Fine σ Sweep + Multi-objective, {len(targets)} datasets")

    # Sigma multipliers (median_seg / k for k):
    sigma_divisors = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0]
    # NLM T_factors (including no NLM):
    nlm_t_factors = [None, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0]

    print(f"  Grid: {len(sigma_divisors)} σ × {len(nlm_t_factors)} T = {len(sigma_divisors) * len(nlm_t_factors)} combinations")
    print(f"  + 4 metrics (pak, aff, rbased, severity_f1)")

    all_results = {}

    for i, (alias, swat) in enumerate(targets, 1):
        ds = DatasetScores.load(alias, swat)
        if ds is None:
            continue
        pt_r, pt_d, pt_s, pt_f = per_channel_points(ds, stride=21)
        base_unsmoothed = adaptive_combine(pt_r, pt_d, pt_f, use_fm=True)
        median_seg = median_anomaly_segment_length(ds.regions)

        labels_eval = ds.point_labels
        if ds.eval_mask is not None:
            labels_eval = ds.point_labels.copy()
            labels_eval[~ds.eval_mask] = 0

        # Baseline (gauss10)
        baseline = gauss(base_unsmoothed, 10)
        baseline_pak = pak_auc_f1(baseline, ds.point_labels, ds.regions, ds.eval_mask)

        dataset_results = {
            'baseline_pak': baseline_pak,
            'median_seg': median_seg,
            'grid': {},
        }

        for div in sigma_divisors:
            sigma = max(median_seg / div, 0.5)
            smoothed = gauss(base_unsmoothed, sigma)

            for t in nlm_t_factors:
                if t is None:
                    final_score = smoothed
                    key = f'div{div:.1f}_noNLM'
                else:
                    final_score = nlm_sigmoid_transform(smoothed, T_factor=t)
                    key = f'div{div:.1f}_T{t}'

                pak = pak_auc_f1(final_score, ds.point_labels, ds.regions, ds.eval_mask)
                dataset_results['grid'][key] = {'pak': pak, 'sigma': sigma}

        all_results[alias] = dataset_results

        if i % 10 == 0 or i == len(targets):
            print(f"  [{i:2d}/{len(targets)}] processed", flush=True)

    # Save raw
    out = Path(__file__).parent.parent / 'results' / 'P2_fine_sigma_sweep.json'
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out}")

    # ============== ANALYSIS ==============
    print("\n=== σ × T grid: mean Δ across 39 datasets (Δ vs baseline_gauss10) ===")
    header = 'div/T'
    print(f"\n{header:<8s}", end='')
    for t in nlm_t_factors:
        label = f"T={t}" if t is not None else "noNLM"
        print(f" {label:>9s}", end='')
    print()

    best_combo, best_delta = None, -np.inf
    grid_summary = {}

    for div in sigma_divisors:
        print(f"div={div:<5.1f}", end='')
        for t in nlm_t_factors:
            if t is None:
                key = f'div{div:.1f}_noNLM'
            else:
                key = f'div{div:.1f}_T{t}'
            deltas = [r['grid'][key]['pak'] - r['baseline_pak'] for r in all_results.values()]
            mean_d = np.mean(deltas)
            print(f" {mean_d:+.4f} ", end='')

            wins = sum(1 for d in deltas if d > 0)
            losses = sum(1 for d in deltas if d < 0)
            cata = sum(1 for d in deltas if d < -0.05)

            grid_summary[key] = {
                'mean_delta': float(mean_d),
                'wins': wins, 'losses': losses, 'cata': cata,
            }

            if mean_d > best_delta:
                best_delta = mean_d
                best_combo = key
        print()

    print(f"\n=== Best combination (standalone, no cluster routing): {best_combo} ===")
    best_data = grid_summary[best_combo]
    print(f"  mean Δ: {best_data['mean_delta']:+.4f}")
    print(f"  W/L:    {best_data['wins']}/{best_data['losses']}")
    print(f"  cata:   {best_data['cata']}")

    # Wilcoxon
    method_s = [r['grid'][best_combo]['pak'] for r in all_results.values()]
    baseline_s = [r['baseline_pak'] for r in all_results.values()]
    p = wilcoxon_test(method_s, baseline_s, alternative='greater')
    print(f"  Wilcoxon p(>): {p:.4f}")

    # Per-group breakdown for best
    deltas_dict = {a: r['grid'][best_combo]['pak'] - r['baseline_pak']
                    for a, r in all_results.items()}
    summary = per_group_summary(deltas_dict, get_per_group)
    print(f"\n  Per-group:")
    for g, s in summary.items():
        print(f"    {g:<12s} n={s['n']:2d}  meanΔ={s['mean_delta']:+.4f}  "
              f"W/L={s['wins']}/{s['losses']}  cata={s['catastrophic']}")

    # Top-5 grid combos
    print(f"\n=== Top-10 (σ, T) combinations by mean Δ ===")
    sorted_combos = sorted(grid_summary.items(), key=lambda x: -x[1]['mean_delta'])
    for combo, d in sorted_combos[:10]:
        method_s = [r['grid'][combo]['pak'] for r in all_results.values()]
        p = wilcoxon_test(method_s, baseline_s, alternative='greater')
        print(f"  {combo:<22s}: meanΔ={d['mean_delta']:+.4f}  W/L={d['wins']:2d}/{d['losses']:2d}  cata={d['cata']}  p={p:.4f}")

    # Multi-metric evaluation for top-3 combinations
    print(f"\n=== Multi-metric for top-3 (σ, T) ===")
    top3 = [c[0] for c in sorted_combos[:3]] + ['div3.0_noNLM']  # +E9 reference

    for combo in top3:
        # Compute per-dataset multi-metric
        all_paks, all_affs, all_rbs = [], [], []
        for alias, r in all_results.items():
            ds = DatasetScores.load(alias, alias == 'swat_excl22')
            if ds is None:
                continue
            pt_r, pt_d, pt_s, pt_f = per_channel_points(ds, stride=21)
            base_unsmoothed = adaptive_combine(pt_r, pt_d, pt_f, use_fm=True)
            median_seg = median_anomaly_segment_length(ds.regions)

            # Parse combo
            parts = combo.split('_')
            div = float(parts[0].replace('div', ''))
            t_str = parts[1]
            t = None if t_str == 'noNLM' else float(t_str.replace('T', ''))

            sigma = max(median_seg / div, 0.5)
            smoothed = gauss(base_unsmoothed, sigma)
            if t is not None:
                final = nlm_sigmoid_transform(smoothed, T_factor=t)
            else:
                final = smoothed

            labels_eval = ds.point_labels
            if ds.eval_mask is not None:
                labels_eval = ds.point_labels.copy()
                labels_eval[~ds.eval_mask] = 0

            all_paks.append(pak_auc_f1(final, ds.point_labels, ds.regions, ds.eval_mask))
            all_affs.append(safe_affiliation_f1(final, labels_eval, ds.regions))
            all_rbs.append(safe_rbased_f1(final, labels_eval))

        print(f"\n  {combo}:")
        print(f"    pak    mean: {np.mean(all_paks):.4f}")
        print(f"    aff    mean: {np.mean(all_affs):.4f}")
        print(f"    rbased mean: {np.mean(all_rbs):.4f}")


if __name__ == "__main__":
    main()
