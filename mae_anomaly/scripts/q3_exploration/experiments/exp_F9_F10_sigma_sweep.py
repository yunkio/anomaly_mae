"""
F9 + F10 + E9 σ sweep combined experiment.

- F9: Multi-metric ensemble (pak + affiliation + rbased weighted)
- F10: Severity-weighted F1
- E9 σ multiplier sweep (median_seg / {2.355, 3, 4, 5})
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


# Multi-metric using prts + affiliation if available
def safe_affiliation_f1(scores, labels, regions):
    """Affiliation-F1 via affiliation library."""
    try:
        from affiliation.metrics import pr_from_events
        threshold = np.percentile(scores, 100 * (1 - labels.mean()))
        pred = (scores > threshold).astype(int)

        def find_events(arr):
            events = []
            in_e, st = False, None
            for i, v in enumerate(arr):
                if v == 1 and not in_e:
                    st, in_e = i, True
                elif v == 0 and in_e:
                    events.append((st, i)); in_e = False
            if in_e:
                events.append((st, len(arr)))
            return events

        gt_events = find_events(labels)
        pred_events = find_events(pred)
        if not gt_events or not pred_events:
            return 0.0
        res = pr_from_events(pred_events, gt_events, (0, len(labels)))
        p, r = res['precision'], res['recall']
        return float(2*p*r/(p+r)) if (p+r) > 0 else 0.0
    except Exception:
        return 0.0


def safe_rbased_f1(scores, labels):
    """R-based F1 via prts library."""
    try:
        from prts import ts_precision, ts_recall
        threshold = np.percentile(scores, 100 * (1 - labels.mean()))
        pred = (scores > threshold).astype(int)
        p = ts_precision(labels, pred, alpha=0.0, cardinality='reciprocal', bias='flat')
        r = ts_recall(labels, pred, alpha=0.0, cardinality='reciprocal', bias='flat')
        return float(2*p*r/(p+r)) if (p+r) > 0 else 0.0
    except Exception:
        return 0.0


def severity_weighted_f1(scores, labels, regions, n_thresholds=200):
    """F10: Length-weighted F1 (log severity)."""
    thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)
    best_f1 = 0
    for thr in thresholds:
        pred = scores > thr
        tp_w = 0.0
        fn_w = 0.0
        for r in regions:
            seg_len = r.end - r.start
            weight = np.log(seg_len + 1)
            detected = pred[r.start:r.end].sum() >= 0.5 * seg_len
            if detected:
                tp_w += weight
            else:
                fn_w += weight
        # FP (unweighted)
        all_anom = np.zeros_like(pred, dtype=bool)
        for r in regions:
            all_anom[r.start:r.end] = True
        fp = float(((~all_anom) & pred).sum())
        if tp_w + fp == 0 or tp_w + fn_w == 0:
            continue
        p = tp_w / (tp_w + fp)
        r_ = tp_w / (tp_w + fn_w)
        f1 = 2*p*r_/(p+r_) if (p+r_) > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
    return float(best_f1)


def nlm_sigmoid(score, T_factor=2.0):
    centered = score - score.mean()
    T = T_factor * (score.std() + 1e-9)
    return 1.0 / (1.0 + np.exp(-np.clip(centered / T, -30, 30)))


def process_one(alias, swat_excl22):
    ds = DatasetScores.load(alias, swat_excl22)
    if ds is None:
        return None

    pt_r, pt_d, pt_s, pt_f = per_channel_points(ds, stride=21)
    base_unsmoothed = adaptive_combine(pt_r, pt_d, pt_f, use_fm=True)
    median_seg = median_anomaly_segment_length(ds.regions)

    # σ multiplier sweep
    sigma_variants = {
        'sigma_div_5':     median_seg / 5.0,
        'sigma_div_4':     median_seg / 4.0,
        'sigma_div_3':     median_seg / 3.0,        # E9 original
        'sigma_div_2_355': median_seg / 2.355,      # FWHM exact
        'sigma_div_2':     median_seg / 2.0,
        'sigma_div_1_5':   median_seg / 1.5,
    }

    sigma_results = {}
    for name, sig in sigma_variants.items():
        smoothed = gauss(base_unsmoothed, sig)
        pak = pak_auc_f1(smoothed, ds.point_labels, ds.regions, ds.eval_mask)
        sigma_results[name] = {'sigma': float(sig), 'pak': float(pak)}

    # Multi-metric for baseline, E9, B1 (E9 × NLM-T2)
    multi_metric = {}
    e9_smoothed = gauss(base_unsmoothed, median_seg / 3.0)
    b1_smoothed = nlm_sigmoid(e9_smoothed, T_factor=2.0)
    baseline_smoothed = gauss(base_unsmoothed, 10)

    for name, sc in [('baseline', baseline_smoothed), ('e9', e9_smoothed), ('b1', b1_smoothed)]:
        labels_eval = ds.point_labels
        if ds.eval_mask is not None:
            labels_eval = ds.point_labels.copy()
            labels_eval[~ds.eval_mask] = 0

        multi_metric[name] = {
            'pak': pak_auc_f1(sc, ds.point_labels, ds.regions, ds.eval_mask),
            'aff': safe_affiliation_f1(sc, labels_eval, ds.regions),
            'rbased': safe_rbased_f1(sc, labels_eval),
            'severity_f1': severity_weighted_f1(sc, labels_eval, ds.regions),
        }

    return {
        'alias': alias,
        'median_seg': median_seg,
        'sigma_results': sigma_results,
        'multi_metric': multi_metric,
    }


def main():
    targets = iter_dataset_aliases()
    print(f"F9+F10+σ sweep, {len(targets)} datasets")

    results = {}
    for i, (alias, swat) in enumerate(targets, 1):
        try:
            r = process_one(alias, swat)
            if r is None:
                continue
            results[alias] = r
            if i % 10 == 0 or i == len(targets):
                print(f"[{i:2d}/{len(targets)}] processed", flush=True)
        except Exception as e:
            print(f"FAILED {alias}: {e}")

    out = Path(__file__).parent.parent / 'results' / 'F9_F10_sigma_sweep.json'
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")

    # ==================== σ multiplier sweep ====================
    print("\n=== E9 σ Multiplier Sweep (vs baseline gauss10) ===")
    baseline_scores = [r['multi_metric']['baseline']['pak'] for r in results.values()]

    for variant in ['sigma_div_5', 'sigma_div_4', 'sigma_div_3', 'sigma_div_2_355',
                    'sigma_div_2', 'sigma_div_1_5']:
        scores = [r['sigma_results'][variant]['pak'] for r in results.values()]
        deltas = np.array(scores) - np.array(baseline_scores)
        mean_d = deltas.mean()
        wins = (deltas > 0).sum()
        losses = (deltas < 0).sum()
        cata = (deltas < -0.05).sum()
        p = wilcoxon_test(scores, baseline_scores, alternative='greater')
        label_map = {
            'sigma_div_5': 'med_seg/5  (under)',
            'sigma_div_4': 'med_seg/4',
            'sigma_div_3': 'med_seg/3  (E9 orig)',
            'sigma_div_2_355': 'med_seg/2.355 (FWHM)',
            'sigma_div_2': 'med_seg/2',
            'sigma_div_1_5': 'med_seg/1.5 (over)',
        }
        print(f"  {label_map[variant]:<28s}: meanΔ={mean_d:+.4f}  W/L={wins:2d}/{losses:2d}  cata={cata}  p(>)={p:.3f}")

    # ==================== Multi-Metric Comparison ====================
    print("\n=== Multi-Metric for baseline / E9 / B1 ===")
    for metric in ['pak', 'aff', 'rbased', 'severity_f1']:
        base_m = [r['multi_metric']['baseline'][metric] for r in results.values()]
        e9_m = [r['multi_metric']['e9'][metric] for r in results.values()]
        b1_m = [r['multi_metric']['b1'][metric] for r in results.values()]
        print(f"\n  Metric: {metric}")
        print(f"    baseline mean: {np.mean(base_m):.4f}")
        print(f"    e9 mean:       {np.mean(e9_m):.4f}  Δ={np.mean(e9_m)-np.mean(base_m):+.4f}")
        print(f"    b1 mean:       {np.mean(b1_m):.4f}  Δ={np.mean(b1_m)-np.mean(base_m):+.4f}")

        # Wilcoxon
        p_e9 = wilcoxon_test(e9_m, base_m, alternative='greater')
        p_b1 = wilcoxon_test(b1_m, base_m, alternative='greater')
        print(f"    p(e9 > base) = {p_e9:.3f}, p(b1 > base) = {p_b1:.3f}")

    # ==================== F9 Multi-Metric Ensemble ====================
    print("\n=== F9 Multi-Metric Ensemble Score ===")
    # Per-method z-score average across metrics
    for method in ['baseline', 'e9', 'b1']:
        z_pak = zscore(np.array([r['multi_metric'][method]['pak'] for r in results.values()]))
        z_aff = zscore(np.array([r['multi_metric'][method]['aff'] for r in results.values()]))
        z_rb = zscore(np.array([r['multi_metric'][method]['rbased'] for r in results.values()]))
        z_sev = zscore(np.array([r['multi_metric'][method]['severity_f1'] for r in results.values()]))
        ensemble = (z_pak + z_aff + z_rb + z_sev) / 4.0
        print(f"  {method:<10s} ensemble mean: {ensemble.mean():.4f} (std: {ensemble.std():.4f})")

    # Compare e9 vs b1 on ensemble
    print("\n  e9 vs b1 ensemble Wilcoxon:")
    e9_ensemble = []
    b1_ensemble = []
    for r in results.values():
        z_e = (r['multi_metric']['e9']['pak'] + r['multi_metric']['e9']['aff'] +
               r['multi_metric']['e9']['rbased'] + r['multi_metric']['e9']['severity_f1']) / 4.0
        z_b = (r['multi_metric']['b1']['pak'] + r['multi_metric']['b1']['aff'] +
               r['multi_metric']['b1']['rbased'] + r['multi_metric']['b1']['severity_f1']) / 4.0
        e9_ensemble.append(z_e)
        b1_ensemble.append(z_b)
    p = wilcoxon_test(b1_ensemble, e9_ensemble, alternative='greater')
    print(f"  p(b1 > e9 in ensemble) = {p:.3f}")


if __name__ == "__main__":
    main()
