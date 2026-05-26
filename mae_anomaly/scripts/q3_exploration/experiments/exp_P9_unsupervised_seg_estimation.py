"""
P9 — Unsupervised median_seg Estimator (CRITICAL)

본 작업 전체에서 가장 critical 한 미해결 문제: median_anomaly_segment_length를
labels 없이 estimate. 만약 unsupervised로 정확히 추정 가능하면 div5.0_T1.5
heuristic을 fully unsupervised로 deploy 가능 (+0.0212 양성 효과).

본 실험은:
1. 6개 base estimator로 39 datasets에서 estimate
2. 각 estimator의 (estimated vs true) median_seg correlation 측정
3. Estimator로부터 σ 계산 → 최종 pak_auc_f1 evaluation
4. 5개 ensemble 변종 (weighted geom mean, median, max-confidence 등) 비교
5. Per-dataset error analysis (어떤 dataset에서 estimator가 fail?)
"""
import sys
from pathlib import Path
import numpy as np
import json
from scipy.stats import pearsonr, spearmanr

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
from mae_anomaly.scripts.q3_exploration.core.segment_estimation import (
    PeakRunEstimator, PeakWidthEstimator, AutocorrelationEstimator,
    WaveletEstimator, KDEEstimator, ChangePointEstimator, EnsembleEstimator,
)


def evaluate_with_estimated_sigma(ds, base_unsmoothed, estimated_median_seg,
                                    sigma_divisor=5.0, use_nlm=True, T=1.5):
    """Apply div5.0_T1.5 with estimated σ."""
    sigma = max(estimated_median_seg / sigma_divisor, 0.5)
    smoothed = gauss(base_unsmoothed, sigma)
    if use_nlm:
        smoothed = nlm_sigmoid_transform(smoothed, T_factor=T)
    return pak_auc_f1(smoothed, ds.point_labels, ds.regions, ds.eval_mask)


def main():
    targets = iter_dataset_aliases()
    print(f"P9 — Unsupervised median_seg Estimation, {len(targets)} datasets")

    base_estimators = {
        'peak_run_p85': PeakRunEstimator(percentile=85),
        'peak_run_p90': PeakRunEstimator(percentile=90),
        'peak_run_p95': PeakRunEstimator(percentile=95),
        'peak_width': PeakWidthEstimator(prominence_percentile=90),
        'autocorr_e_half': AutocorrelationEstimator(target_decay=0.5),
        'autocorr_e_third': AutocorrelationEstimator(target_decay=0.367),
        'wavelet': WaveletEstimator(),
        'kde': KDEEstimator(),
        'change_point': ChangePointEstimator(),
    }

    ensemble_variants = {
        'ens_geom_mean': EnsembleEstimator(mode='weighted_geom_mean'),
        'ens_wmean': EnsembleEstimator(mode='weighted_mean'),
        'ens_median': EnsembleEstimator(mode='median'),
        'ens_max_conf': EnsembleEstimator(mode='max_confidence'),
    }

    all_results = {}

    print(f"\nN base estimators: {len(base_estimators)}")
    print(f"N ensemble variants: {len(ensemble_variants)}")

    for i, (alias, swat) in enumerate(targets, 1):
        ds = DatasetScores.load(alias, swat)
        if ds is None:
            continue

        pt_r, pt_d, pt_s, pt_f = per_channel_points(ds, stride=21)
        base_unsmoothed = adaptive_combine(pt_r, pt_d, pt_f, use_fm=True)

        true_median_seg = median_anomaly_segment_length(ds.regions)
        baseline_smoothed = gauss(base_unsmoothed, 10)
        baseline_pak = pak_auc_f1(baseline_smoothed,
                                   ds.point_labels, ds.regions, ds.eval_mask)

        # True median_seg 기반 (semi-supervised reference)
        ref_pak = evaluate_with_estimated_sigma(ds, base_unsmoothed, true_median_seg)

        result = {
            'alias': alias,
            'true_median_seg': true_median_seg,
            'baseline_pak': baseline_pak,
            'ref_pak_supervised': ref_pak,
            'estimators': {},
        }

        # Base estimators
        for name, est in base_estimators.items():
            try:
                m, c = est.estimate(base_unsmoothed)
                est_pak = evaluate_with_estimated_sigma(ds, base_unsmoothed, m)
                result['estimators'][name] = {
                    'estimated_median_seg': float(m),
                    'confidence': float(c),
                    'pak': float(est_pak),
                    'delta_baseline': float(est_pak - baseline_pak),
                }
            except Exception as e:
                result['estimators'][name] = {'estimated_median_seg': 10.0,
                                              'confidence': 0.0,
                                              'pak': baseline_pak,
                                              'delta_baseline': 0.0,
                                              'error': str(e)}

        # Ensemble variants
        for name, est in ensemble_variants.items():
            try:
                m, c, internals = est.estimate(base_unsmoothed)
                est_pak = evaluate_with_estimated_sigma(ds, base_unsmoothed, m)
                result['estimators'][name] = {
                    'estimated_median_seg': float(m),
                    'confidence': float(c),
                    'pak': float(est_pak),
                    'delta_baseline': float(est_pak - baseline_pak),
                }
            except Exception as e:
                result['estimators'][name] = {'estimated_median_seg': 10.0,
                                              'confidence': 0.0,
                                              'pak': baseline_pak,
                                              'delta_baseline': 0.0,
                                              'error': str(e)}

        all_results[alias] = result

        if i % 10 == 0 or i == len(targets):
            print(f"  [{i:2d}/{len(targets)}] processed", flush=True)

    # Save raw
    out = Path(__file__).parent.parent / 'results' / 'P9_unsupervised_seg.json'
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out}")

    # ============== ANALYSIS ==============
    print("\n=== Estimator correlation with TRUE median_seg ===")
    true_segs = np.array([r['true_median_seg'] for r in all_results.values()])
    log_true_segs = np.log(true_segs + 1)

    all_estimator_names = list(base_estimators.keys()) + list(ensemble_variants.keys())
    correlation_results = {}
    for name in all_estimator_names:
        estimated = np.array([r['estimators'][name]['estimated_median_seg']
                              for r in all_results.values()])
        log_estimated = np.log(estimated + 1)

        rho_pearson, _ = pearsonr(log_estimated, log_true_segs)
        rho_spearman, _ = spearmanr(estimated, true_segs)
        correlation_results[name] = {
            'pearson_log': float(rho_pearson),
            'spearman': float(rho_spearman),
            'mean_estimated': float(estimated.mean()),
            'mean_true': float(true_segs.mean()),
        }

    sorted_corrs = sorted(correlation_results.items(),
                          key=lambda x: -x[1]['spearman'])
    print(f"\n{'Estimator':<22s} {'Pearson(log)':>15s} {'Spearman':>10s}")
    for name, c in sorted_corrs:
        print(f"{name:<22s} {c['pearson_log']:>15.3f} {c['spearman']:>10.3f}")

    # Pak performance ranking
    print("\n=== pak_auc_f1 result with each estimator (div5.0_T1.5 base) ===")
    baseline_paks = [r['baseline_pak'] for r in all_results.values()]
    ref_paks = [r['ref_pak_supervised'] for r in all_results.values()]

    delta_ref = np.array(ref_paks) - np.array(baseline_paks)
    print(f"\nReference (supervised true median_seg): meanΔ={delta_ref.mean():+.4f}")
    print(f"{'Estimator':<22s} {'meanΔ':>10s} {'W/L':>8s} {'cata':>5s} {'p':>8s} {'capture':>9s}")

    estimator_summary = {}
    for name in all_estimator_names:
        deltas = np.array([r['estimators'][name]['delta_baseline']
                            for r in all_results.values()])
        scores = np.array([r['estimators'][name]['pak'] for r in all_results.values()])
        mean_d = float(deltas.mean())
        wins = int((deltas > 0).sum())
        losses = int((deltas < 0).sum())
        cata = int((deltas < -0.05).sum())
        p = wilcoxon_test(scores.tolist(), baseline_paks, alternative='greater')
        capture = mean_d / delta_ref.mean() * 100 if delta_ref.mean() > 0 else 0
        estimator_summary[name] = {
            'mean_delta': mean_d, 'wins': wins, 'losses': losses,
            'cata': cata, 'p_value': p, 'oracle_capture_pct': capture,
        }
        print(f"{name:<22s} {mean_d:>+10.4f} {wins:>2d}/{losses:<2d}     {cata:>5d} {p:>8.4f} {capture:>8.1f}%")

    # Best estimator analysis
    best_name = max(estimator_summary, key=lambda x: estimator_summary[x]['mean_delta'])
    print(f"\n=== BEST unsupervised estimator: {best_name} ===")
    best = estimator_summary[best_name]
    print(f"  mean Δ: {best['mean_delta']:+.4f}")
    print(f"  W/L:    {best['wins']}/{best['losses']}")
    print(f"  cata:   {best['cata']}")
    print(f"  p:      {best['p_value']:.4f}")
    print(f"  capture: {best['oracle_capture_pct']:.1f}% of supervised reference")

    # Per-group breakdown
    deltas_dict = {a: r['estimators'][best_name]['delta_baseline']
                    for a, r in all_results.items()}
    summary = per_group_summary(deltas_dict, get_per_group)
    print(f"\n  Per-group:")
    for g, s in summary.items():
        print(f"    {g:<12s} n={s['n']:2d}  meanΔ={s['mean_delta']:+.4f}  "
              f"W/L={s['wins']}/{s['losses']}  cata={s['catastrophic']}")

    # Per-dataset error analysis (where estimators fail?)
    print(f"\n=== Per-dataset estimator quality (top failures with {best_name}) ===")
    per_dataset_errors = []
    for alias, r in all_results.items():
        est_seg = r['estimators'][best_name]['estimated_median_seg']
        true_seg = r['true_median_seg']
        log_ratio = abs(np.log(est_seg / max(true_seg, 1e-9)))
        per_dataset_errors.append((alias, true_seg, est_seg, log_ratio,
                                    r['estimators'][best_name]['delta_baseline']))

    per_dataset_errors.sort(key=lambda x: -x[3])
    print(f"\n{'alias':<25s} {'true_med':>9s} {'est_med':>9s} {'log_ratio':>10s} {'Δ':>9s}")
    for alias, true_s, est_s, lr, dl in per_dataset_errors[:10]:
        print(f"{alias:<25s} {true_s:>9.1f} {est_s:>9.1f} {lr:>10.2f} {dl:>+9.4f}")

    # Save analysis
    analysis_out = Path(__file__).parent.parent / 'results' / 'P9_analysis.json'
    with open(analysis_out, 'w') as f:
        json.dump({
            'correlation_with_true': correlation_results,
            'estimator_summary': estimator_summary,
            'best_estimator': best_name,
            'reference_delta': float(delta_ref.mean()),
        }, f, indent=2)
    print(f"\nSaved analysis: {analysis_out}")


if __name__ == "__main__":
    main()
