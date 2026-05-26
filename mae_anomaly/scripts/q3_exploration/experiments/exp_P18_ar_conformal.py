"""
P18 — AR Residual + Conformal Prediction

두 가지 다른 mechanism의 hybrid:

1. AR(p) residual: score sequence를 AR(p) model로 predict 후 residual을 anomaly indicator로
   - p ∈ {3, 5, 10, 20}
2. Conformal prediction: training portion 기반 calibrated p-value
3. Hybrid: AR + Conformal + base score
4. AR + GMM/HMM state segmentation
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
    ConformalCalibrator, conformal_anomaly_score,
)
from mae_anomaly.scripts.q3_exploration.core.timeseries_models import (
    ar_residual_score, hmm_state_anomaly_score, state_persistence_score,
    spectral_subtract,
)


def main():
    targets = iter_dataset_aliases()
    print(f"P18 — AR Residual + Conformal Prediction, {len(targets)} datasets")

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

        e9_smoothed = gauss(base_unsmoothed, max(median_seg/5, 0.5))
        ref_score = nlm_sigmoid_transform(e9_smoothed, T_factor=1.5)
        ref_pak = pak_auc_f1(ref_score, ds.point_labels, ds.regions, ds.eval_mask)

        results = {
            'baseline_pak': baseline_pak,
            'ref_pak': ref_pak,
            'median_seg': median_seg,
            'variants': {},
        }

        # === AR Residual variants ===
        for p in [3, 5, 10, 20]:
            try:
                ar_res = ar_residual_score(base_unsmoothed, p=p)
                ar_smoothed = gauss(ar_res, max(median_seg / 5.0, 0.5))
                results['variants'][f'ar{p}_alone'] = float(
                    pak_auc_f1(ar_smoothed, ds.point_labels, ds.regions, ds.eval_mask))

                # AR + NLM
                ar_nlm = nlm_sigmoid_transform(ar_smoothed, T_factor=1.5)
                results['variants'][f'ar{p}_nlm'] = float(
                    pak_auc_f1(ar_nlm, ds.point_labels, ds.regions, ds.eval_mask))

                # Hybrid: ref + AR z-norm sum
                hybrid = zscore(ref_score) + zscore(ar_res)
                results['variants'][f'ref_plus_ar{p}'] = float(
                    pak_auc_f1(hybrid, ds.point_labels, ds.regions, ds.eval_mask))
            except Exception:
                pass

        # === Conformal variants ===
        for cal_pct in [50, 70, 80, 90]:
            try:
                cal_thr = np.percentile(base_unsmoothed, cal_pct)
                cal_scores = base_unsmoothed[base_unsmoothed <= cal_thr]

                conf_score = conformal_anomaly_score(base_unsmoothed, cal_scores=cal_scores)
                conf_smoothed = gauss(conf_score, max(median_seg / 5.0, 0.5))
                conf_nlm = nlm_sigmoid_transform(conf_smoothed, T_factor=1.5)
                results['variants'][f'conf{cal_pct}_alone'] = float(
                    pak_auc_f1(conf_nlm, ds.point_labels, ds.regions, ds.eval_mask))

                # Hybrid with ref
                hybrid = zscore(ref_score) + zscore(conf_score)
                results['variants'][f'ref_plus_conf{cal_pct}'] = float(
                    pak_auc_f1(hybrid, ds.point_labels, ds.regions, ds.eval_mask))
            except Exception:
                pass

        # === HMM-based state ===
        try:
            hmm_state = hmm_state_anomaly_score(e9_smoothed, transition_stay=0.95)
            results['variants']['hmm_state'] = float(
                pak_auc_f1(hmm_state, ds.point_labels, ds.regions, ds.eval_mask))

            # HMM smoothed
            hmm_smooth = gauss(hmm_state, max(median_seg/10, 0.5))
            results['variants']['hmm_state_smoothed'] = float(
                pak_auc_f1(hmm_smooth, ds.point_labels, ds.regions, ds.eval_mask))

            # Hybrid
            hybrid_hmm = zscore(ref_score) + zscore(hmm_state)
            results['variants']['ref_plus_hmm'] = float(
                pak_auc_f1(hybrid_hmm, ds.point_labels, ds.regions, ds.eval_mask))
        except Exception:
            pass

        # === State persistence ===
        try:
            pers = state_persistence_score(e9_smoothed, threshold_percentile=85, window=21)
            pers_smoothed = gauss(pers, max(median_seg/10, 0.5))
            results['variants']['persistence'] = float(
                pak_auc_f1(pers_smoothed, ds.point_labels, ds.regions, ds.eval_mask))

            hybrid_pers = zscore(ref_score) + zscore(pers)
            results['variants']['ref_plus_persistence'] = float(
                pak_auc_f1(hybrid_pers, ds.point_labels, ds.regions, ds.eval_mask))
        except Exception:
            pass

        # === Spectral subtract on base ===
        try:
            cleaned = spectral_subtract(base_unsmoothed)
            cleaned_smoothed = gauss(cleaned, max(median_seg / 5.0, 0.5))
            cleaned_nlm = nlm_sigmoid_transform(cleaned_smoothed, T_factor=1.5)
            results['variants']['spectral_subtract'] = float(
                pak_auc_f1(cleaned_nlm, ds.point_labels, ds.regions, ds.eval_mask))
        except Exception:
            pass

        # === Multi-method ensemble: ref + AR5 + HMM + persistence ===
        try:
            ar5 = ar_residual_score(base_unsmoothed, p=5)
            hmm = hmm_state_anomaly_score(e9_smoothed)
            pers = state_persistence_score(e9_smoothed)
            super_ensemble = (zscore(ref_score) + 0.5 * zscore(ar5)
                              + 0.3 * zscore(hmm) + 0.3 * zscore(pers))
            results['variants']['super_ensemble'] = float(
                pak_auc_f1(super_ensemble, ds.point_labels, ds.regions, ds.eval_mask))
        except Exception:
            pass

        all_results[alias] = results

        if i % 10 == 0 or i == len(targets):
            print(f"  [{i:2d}/{len(targets)}] processed", flush=True)

    out = Path(__file__).parent.parent / 'results' / 'P18_ar_conformal.json'
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out}")

    # Analysis
    print("\n=== AR + Conformal + HMM variants ranking ===")
    baseline_paks = [r['baseline_pak'] for r in all_results.values()]
    ref_paks = [r['ref_pak'] for r in all_results.values()]
    print(f"reference ref div5_T1.5: meanΔ={np.mean(ref_paks) - np.mean(baseline_paks):+.4f}\n")

    variant_names = sorted(all_results[list(all_results.keys())[0]]['variants'].keys())
    variant_summaries = {}
    for name in variant_names:
        scores = [r['variants'].get(name, r['baseline_pak']) for r in all_results.values()]
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
    print(f"{'Variant':<28s} {'meanΔ':>10s} {'W/L':>9s} {'cata':>5s} {'p':>8s}")
    for name, s in sorted_variants:
        print(f"{name:<28s} {s['mean_delta']:>+10.4f} {s['wins']:>2d}/{s['losses']:<2d}      {s['cata']:>5d} {s['p_value']:>8.4f}")

    # Best
    best_name, best_data = sorted_variants[0]
    print(f"\n=== BEST: {best_name} ===")
    print(f"  vs ref: {best_data['mean_delta'] - (np.mean(ref_paks) - np.mean(baseline_paks)):+.4f}")


if __name__ == "__main__":
    main()
