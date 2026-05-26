"""
P17 — Gaussian Mixture Model on Score Distribution

Anomaly score 분포가 bimodal (normal + anomaly modes)이라고 가정.
GMM으로 2 components fit 후, 각 점의 anomaly mode posterior probability를 anomaly score로 사용.

Variants:
1. 2-component GMM on base_unsmoothed
2. 2-component GMM on smoothed (div5)
3. 2-component GMM on smoothed (div5_T1.5)
4. 3-component GMM (separate noise / normal / anomaly)
5. GMM-based posterior + hybrid with original score

Mechanism가 다른 angle: distribution-shape based vs intensity-based.
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
    fit_gmm_2component, fit_gmm_n_components, gmm_anomaly_posterior,
)


def main():
    targets = iter_dataset_aliases()
    print(f"P17 — Gaussian Mixture Model on Score Distribution, {len(targets)} datasets")

    all_results = {}

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

        # E9 smoothed (no NLM)
        e9_score = gauss(base_unsmoothed, max(median_seg/5, 0.5))

        # Variants
        results = {
            'baseline_pak': baseline_pak,
            'ref_pak': ref_pak,
            'median_seg': median_seg,
            'variants': {},
        }

        # V1: GMM 2-component on baseline
        try:
            posterior_v1 = gmm_anomaly_posterior(baseline)
            v1_smoothed = gauss(posterior_v1, max(median_seg/5, 0.5))
            results['variants']['gmm2_on_baseline'] = float(
                pak_auc_f1(v1_smoothed, ds.point_labels, ds.regions, ds.eval_mask))
        except Exception:
            results['variants']['gmm2_on_baseline'] = baseline_pak

        # V2: GMM on e9_score
        try:
            posterior_v2 = gmm_anomaly_posterior(e9_score)
            results['variants']['gmm2_on_e9'] = float(
                pak_auc_f1(posterior_v2, ds.point_labels, ds.regions, ds.eval_mask))
        except Exception:
            results['variants']['gmm2_on_e9'] = baseline_pak

        # V3: GMM on ref (div5_T1.5)
        try:
            posterior_v3 = gmm_anomaly_posterior(ref_score)
            results['variants']['gmm2_on_ref'] = float(
                pak_auc_f1(posterior_v3, ds.point_labels, ds.regions, ds.eval_mask))
        except Exception:
            results['variants']['gmm2_on_ref'] = baseline_pak

        # V4: 3-component GMM (noise + normal + anomaly)
        try:
            gmm3 = fit_gmm_n_components(e9_score, n_components=3)
            if gmm3 is not None:
                anom_idx = int(np.argmax(gmm3.means_.flatten()))
                posterior_v4 = gmm3.predict_proba(e9_score.reshape(-1, 1))[:, anom_idx]
                results['variants']['gmm3_on_e9'] = float(
                    pak_auc_f1(posterior_v4, ds.point_labels, ds.regions, ds.eval_mask))
            else:
                results['variants']['gmm3_on_e9'] = baseline_pak
        except Exception:
            results['variants']['gmm3_on_e9'] = baseline_pak

        # V5: Hybrid - base + GMM posterior z-norm sum
        try:
            hybrid = zscore(ref_score) + zscore(posterior_v3)
            results['variants']['hybrid_ref_plus_gmm'] = float(
                pak_auc_f1(hybrid, ds.point_labels, ds.regions, ds.eval_mask))
        except Exception:
            results['variants']['hybrid_ref_plus_gmm'] = baseline_pak

        # V6: GMM 2 with smoothing
        try:
            posterior_smoothed = gauss(posterior_v3, max(median_seg/10, 0.5))
            posterior_smoothed_nlm = nlm_sigmoid_transform(posterior_smoothed, T_factor=1.5)
            results['variants']['gmm2_ref_smoothed_nlm'] = float(
                pak_auc_f1(posterior_smoothed_nlm, ds.point_labels, ds.regions, ds.eval_mask))
        except Exception:
            results['variants']['gmm2_ref_smoothed_nlm'] = baseline_pak

        # V7: GMM 2-component with explicit fit on training portion only (split first 60%)
        try:
            split_idx = int(len(ref_score) * 0.6)
            train_scores = ref_score[:split_idx]
            posterior_v7_full = gmm_anomaly_posterior(ref_score, train_scores=train_scores)
            results['variants']['gmm2_train_fit'] = float(
                pak_auc_f1(posterior_v7_full, ds.point_labels, ds.regions, ds.eval_mask))
        except Exception:
            results['variants']['gmm2_train_fit'] = baseline_pak

        # V8: Mean of multiple GMM posteriors (ensemble)
        try:
            posteriors = []
            for seed in [42, 123, 7]:
                posteriors.append(gmm_anomaly_posterior(ref_score, random_state=seed))
            mean_post = np.mean(posteriors, axis=0)
            results['variants']['gmm_ensemble'] = float(
                pak_auc_f1(mean_post, ds.point_labels, ds.regions, ds.eval_mask))
        except Exception:
            results['variants']['gmm_ensemble'] = baseline_pak

        all_results[alias] = results

        if i % 10 == 0 or i == len(targets):
            print(f"  [{i:2d}/{len(targets)}] processed", flush=True)

    out = Path(__file__).parent.parent / 'results' / 'P17_gmm.json'
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out}")

    # Analysis
    print("\n=== GMM variants ranking ===")
    baseline_paks = [r['baseline_pak'] for r in all_results.values()]
    ref_paks = [r['ref_pak'] for r in all_results.values()]
    print(f"reference ref div5_T1.5: meanΔ={np.mean(ref_paks) - np.mean(baseline_paks):+.4f}\n")

    variant_names = list(all_results[list(all_results.keys())[0]]['variants'].keys())
    variant_summaries = {}
    for name in variant_names:
        scores = [r['variants'][name] for r in all_results.values()]
        deltas = np.array(scores) - np.array(baseline_paks)
        mean_d = float(deltas.mean())
        wins = int((deltas > 0).sum())
        losses = int((deltas < 0).sum())
        cata = int((deltas < -0.05).sum())
        p = wilcoxon_test(scores, baseline_paks, alternative='greater')
        variant_summaries[name] = {
            'mean_delta': mean_d, 'wins': wins, 'losses': losses,
            'cata': cata, 'p_value': p,
        }

    sorted_variants = sorted(variant_summaries.items(), key=lambda x: -x[1]['mean_delta'])
    print(f"{'Variant':<30s} {'meanΔ':>10s} {'W/L':>9s} {'cata':>5s} {'p':>8s}")
    for name, s in sorted_variants:
        print(f"{name:<30s} {s['mean_delta']:>+10.4f} {s['wins']:>2d}/{s['losses']:<2d}      {s['cata']:>5d} {s['p_value']:>8.4f}")

    # Best
    best_name, best_data = sorted_variants[0]
    print(f"\n=== BEST GMM variant: {best_name} ===")
    print(f"  mean Δ: {best_data['mean_delta']:+.4f}")
    print(f"  vs ref: {best_data['mean_delta'] - (np.mean(ref_paks) - np.mean(baseline_paks)):+.4f}")
    print(f"  p: {best_data['p_value']:.4f}")

    # Per-group
    deltas_dict = {a: r['variants'][best_name] - r['baseline_pak']
                    for a, r in all_results.items()}
    summary = per_group_summary(deltas_dict, get_per_group)
    print(f"\n  Per-group:")
    for g, s in summary.items():
        print(f"    {g:<12s} n={s['n']:2d}  meanΔ={s['mean_delta']:+.4f}  "
              f"W/L={s['wins']}/{s['losses']}  cata={s['catastrophic']}")


if __name__ == "__main__":
    main()
