"""
P4 — Per-dataset Optimal Threshold Search

`compute_pa_k_auc()`는 median threshold + 200개 linear percentile sweep을 사용.
본 실험은 각 dataset마다 더 정교한 threshold search로 best F1을 찾고:

1. Best single threshold F1 (not AUC)을 metric으로 사용
2. Log-grid + linear-grid + percentile-grid 통합
3. Greedy refinement around best threshold
4. Per-dataset threshold이 어떤 percentile에 분포하는지 분석
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
from mae_anomaly.scripts.q3_exploration.core.threshold_opt import (
    best_threshold_search, pak_f1_at_threshold,
)


def main():
    targets = iter_dataset_aliases()
    print(f"P4 — Per-dataset Optimal Threshold Search, {len(targets)} datasets")

    methods_to_evaluate = {
        'baseline_gauss10':       lambda b, ms: gauss(b, 10),
        'e9_div3_noNLM':          lambda b, ms: gauss(b, max(ms / 3, 0.5)),
        'e9_div5_noNLM':          lambda b, ms: gauss(b, max(ms / 5, 0.5)),
        'div5.0_T1.5':            lambda b, ms: nlm_sigmoid_transform(gauss(b, max(ms/5, 0.5)), 1.5),
        'div5.0_T2.0':            lambda b, ms: nlm_sigmoid_transform(gauss(b, max(ms/5, 0.5)), 2.0),
        'div3.0_T2.0':            lambda b, ms: nlm_sigmoid_transform(gauss(b, max(ms/3, 0.5)), 2.0),
    }

    all_results = {}

    for i, (alias, swat) in enumerate(targets, 1):
        ds = DatasetScores.load(alias, swat)
        if ds is None:
            continue
        pt_r, pt_d, pt_s, pt_f = per_channel_points(ds, stride=21)
        base_unsmoothed = adaptive_combine(pt_r, pt_d, pt_f, use_fm=True)
        median_seg = median_anomaly_segment_length(ds.regions)

        labels_eval = ds.point_labels
        regions_eval = ds.regions
        if ds.eval_mask is not None:
            labels_eval = ds.point_labels[ds.eval_mask]
            # Need to recompute regions on masked labels
            # 단순화: use original labels but score is masked elsewhere

        dataset_res = {'median_seg': median_seg, 'methods': {}}

        for method_name, method_fn in methods_to_evaluate.items():
            score = method_fn(base_unsmoothed, median_seg)

            # Default metric: AUC integration (median threshold + linear sweep)
            auc_f1 = pak_auc_f1(score, ds.point_labels, ds.regions, ds.eval_mask)

            # Best F1: single optimal threshold search
            score_eval = score
            labels_for_thr = ds.point_labels
            if ds.eval_mask is not None:
                score_eval = score[ds.eval_mask]
                labels_for_thr = ds.point_labels[ds.eval_mask]
                # Need new regions on masked subset
                from mae_anomaly.scripts.q3_exploration.core.data import regions_from_labels
                regions_for_thr = regions_from_labels(labels_for_thr)
            else:
                regions_for_thr = ds.regions

            best_thr, best_f1 = best_threshold_search(
                score_eval, labels_for_thr, regions_for_thr,
                n_log_thresholds=100, n_linear_thresholds=300, k=0.5
            )

            # Threshold이 어떤 percentile에 위치하는지
            score_for_pct = score_eval
            n_below = (score_for_pct < best_thr).sum()
            threshold_percentile = float(n_below / len(score_for_pct) * 100)

            dataset_res['methods'][method_name] = {
                'auc_f1': float(auc_f1),
                'best_thr': float(best_thr),
                'best_f1': float(best_f1),
                'threshold_percentile': threshold_percentile,
            }

        all_results[alias] = dataset_res

        if i % 10 == 0 or i == len(targets):
            print(f"  [{i:2d}/{len(targets)}] processed", flush=True)

    out = Path(__file__).parent.parent / 'results' / 'P4_threshold_optimization.json'
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out}")

    # ============== ANALYSIS ==============
    print("\n=== AUC F1 vs Best F1 comparison ===")
    print(f"{'Method':<20s} {'AUC F1 mean':>11s} {'Best F1 mean':>13s} {'Δ best vs AUC':>14s}")
    for method in methods_to_evaluate.keys():
        auc_f1s = [r['methods'][method]['auc_f1'] for r in all_results.values()]
        best_f1s = [r['methods'][method]['best_f1'] for r in all_results.values()]
        print(f"{method:<20s} {np.mean(auc_f1s):>11.4f} {np.mean(best_f1s):>13.4f} {np.mean(best_f1s) - np.mean(auc_f1s):>+14.4f}")

    print("\n=== Best F1 Δ vs baseline_gauss10 (single threshold) ===")
    baseline_best_f1s = [r['methods']['baseline_gauss10']['best_f1'] for r in all_results.values()]
    for method in methods_to_evaluate.keys():
        if method == 'baseline_gauss10':
            continue
        best_f1s = [r['methods'][method]['best_f1'] for r in all_results.values()]
        deltas = np.array(best_f1s) - np.array(baseline_best_f1s)
        mean_d = float(deltas.mean())
        wins = int((deltas > 0).sum())
        losses = int((deltas < 0).sum())
        cata = int((deltas < -0.05).sum())
        p = wilcoxon_test(best_f1s, baseline_best_f1s, alternative='greater')
        print(f"  {method:<20s}: meanΔ={mean_d:+.4f}  W/L={wins:2d}/{losses:2d}  cata={cata}  p={p:.4f}")

    # Per-method, threshold percentile 분포
    print("\n=== Threshold percentile distribution (where does optimal threshold sit?) ===")
    print(f"{'Method':<20s} {'min':>6s} {'p25':>6s} {'p50':>6s} {'p75':>6s} {'max':>6s} {'mean':>6s}")
    for method in methods_to_evaluate.keys():
        pcts = [r['methods'][method]['threshold_percentile'] for r in all_results.values()]
        print(f"{method:<20s} {min(pcts):>6.1f} {np.percentile(pcts, 25):>6.1f} "
              f"{np.percentile(pcts, 50):>6.1f} {np.percentile(pcts, 75):>6.1f} "
              f"{max(pcts):>6.1f} {np.mean(pcts):>6.1f}")

    # Per-group breakdown for best method (best F1)
    print("\n=== Per-group: best F1 with div5.0_T1.5 ===")
    deltas_div5_T15 = {a: r['methods']['div5.0_T1.5']['best_f1'] - r['methods']['baseline_gauss10']['best_f1']
                       for a, r in all_results.items()}
    summary = per_group_summary(deltas_div5_T15, get_per_group)
    for g, s in summary.items():
        print(f"  {g:<12s} n={s['n']:2d}  meanΔ={s['mean_delta']:+.4f}  "
              f"W/L={s['wins']}/{s['losses']}  cata={s['catastrophic']}")


if __name__ == "__main__":
    main()
