"""
Phase B — Hybrid Methods

E9 adapt_single (+0.0112) + NLM-T2 (+0.007) + Conditional routing 등의 hybrid 검증.

Hypothesis:
- E9는 smoothing scale (multi-scale frequency adaptation)
- NLM-T2는 distribution shape (tail compression)
- Conditional routing은 dataset characteristic adaptation
→ 이들이 직교 mechanism이면 additive effect 가능

Methods:
- B1: E9 × NLM-T2 (sigmoid tail compression after E9 smoothing)
- B2: Conditional adapt (median_seg 기반 routing: long-segment → larger σ cap)
- B3: 3-method ensemble (E9 + NLM-T2 + Z5 z-score average)
- B4: A3+NLM-T2 (Phase A의 unsupervised + NLM)
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
    point_score_from_loo, gauss, zscore, per_channel_points, adaptive_combine
)
from mae_anomaly.scripts.q3_exploration.core.evaluation import (
    pak_auc_f1, wilcoxon_test, per_group_summary
)


def nlm_sigmoid(score, T_factor=2.0):
    """NLM-T2: sigmoid((score - mean) / (T_factor * std)).
    Monotone tail compression."""
    centered = score - score.mean()
    T = T_factor * (score.std() + 1e-9)
    return 1.0 / (1.0 + np.exp(-np.clip(centered / T, -30, 30)))


def z5_pyramid(score, scales=[5, 25, 100, 400, 1600]):
    """1분기 Z5-Pyramid: 5-scale geometric mean."""
    s = np.maximum(score, 1e-10)
    log_s = np.log(s)
    smoothed = [gauss(log_s, max(W / 3.0, 0.5)) for W in scales]
    return np.exp(np.mean(np.stack(smoothed, axis=0), axis=0))


def hybrid_b1_e9_nlm(base_unsmoothed, sigma_adapt, T_factor=2.0):
    """B1: E9 smoothing + NLM-T2 sigmoid."""
    smoothed = gauss(base_unsmoothed, sigma_adapt)
    return nlm_sigmoid(smoothed, T_factor)


def hybrid_b2_conditional(base_unsmoothed, median_seg, threshold=300, sigma_cap=50):
    """B2: Conditional adapt — long-segment dataset에 σ cap."""
    sigma = median_seg / 3.0
    if median_seg > threshold:
        sigma = min(sigma, sigma_cap)
    return gauss(base_unsmoothed, sigma)


def hybrid_b3_diversity(base_unsmoothed, sigma_adapt):
    """B3: 3-method z-score ensemble (E9 + NLM + Z5)."""
    e9 = gauss(base_unsmoothed, sigma_adapt)
    nlm = nlm_sigmoid(e9, T_factor=2.0)
    z5 = z5_pyramid(base_unsmoothed)
    return (zscore(e9) + zscore(nlm) + zscore(z5)) / 3.0


def hybrid_b4_a3_nlm(base_unsmoothed):
    """B4: Phase A's A3 KDE-based σ + NLM-T2."""
    from scipy.stats import gaussian_kde

    sample = base_unsmoothed
    if len(sample) > 5000:
        idx = np.random.RandomState(0).choice(len(sample), 5000, replace=False)
        sample = sample[idx]

    try:
        kde = gaussian_kde(sample, bw_method='silverman')
        xs = np.linspace(sample.min(), sample.max(), 500)
        density = kde(xs)
        peak_idx = np.argmax(density)
        half_max = density[peak_idx] / 2.0

        left_mask = density[:peak_idx] < half_max
        left_idx = np.where(left_mask)[0][-1] if left_mask.sum() > 0 else 0
        right_mask = density[peak_idx:] < half_max
        right_idx = (peak_idx + np.where(right_mask)[0][0]) if right_mask.sum() > 0 else len(density) - 1

        fwhm_in_xs = xs[right_idx] - xs[left_idx]
        score_range = base_unsmoothed.max() - base_unsmoothed.min() + 1e-9
        fwhm_ratio = fwhm_in_xs / score_range
        sigma_a3 = 10.0 * (1 + fwhm_ratio * 5)
        sigma_a3 = max(min(sigma_a3, 200.0), 0.5)
    except Exception:
        sigma_a3 = 10.0

    smoothed = gauss(base_unsmoothed, sigma_a3)
    return nlm_sigmoid(smoothed, T_factor=2.0)


def hybrid_b5_adapt_z5_routing(base_unsmoothed, median_seg, threshold=200):
    """B5: Dataset-conditional Z5 vs E9 routing.

    Long-segment dataset (median_seg > 200) → Z5-Pyramid (Exathlon-specific best)
    Short/medium → E9 adapt
    """
    if median_seg > threshold:
        return z5_pyramid(base_unsmoothed)
    else:
        return gauss(base_unsmoothed, median_seg / 3.0)


def process_one(alias, swat_excl22):
    ds = DatasetScores.load(alias, swat_excl22)
    if ds is None:
        return None

    pt_r, pt_d, pt_s, pt_f = per_channel_points(ds, stride=21)
    base_unsmoothed = adaptive_combine(pt_r, pt_d, pt_f, use_fm=True)

    median_seg = median_anomaly_segment_length(ds.regions)
    sigma_adapt = median_seg / 3.0

    # Baseline (gauss10)
    baseline = gauss(base_unsmoothed, 10)
    baseline_pak = pak_auc_f1(baseline, ds.point_labels, ds.regions, ds.eval_mask)

    # E9 (semi-supervised ceiling)
    e9 = gauss(base_unsmoothed, sigma_adapt)
    e9_pak = pak_auc_f1(e9, ds.point_labels, ds.regions, ds.eval_mask)

    # B1-B5
    b1 = hybrid_b1_e9_nlm(base_unsmoothed, sigma_adapt)
    b1_pak = pak_auc_f1(b1, ds.point_labels, ds.regions, ds.eval_mask)

    b2 = hybrid_b2_conditional(base_unsmoothed, median_seg)
    b2_pak = pak_auc_f1(b2, ds.point_labels, ds.regions, ds.eval_mask)

    b3 = hybrid_b3_diversity(base_unsmoothed, sigma_adapt)
    b3_pak = pak_auc_f1(b3, ds.point_labels, ds.regions, ds.eval_mask)

    b4 = hybrid_b4_a3_nlm(base_unsmoothed)
    b4_pak = pak_auc_f1(b4, ds.point_labels, ds.regions, ds.eval_mask)

    b5 = hybrid_b5_adapt_z5_routing(base_unsmoothed, median_seg)
    b5_pak = pak_auc_f1(b5, ds.point_labels, ds.regions, ds.eval_mask)

    return {
        'alias': alias, 'baseline_pak': baseline_pak, 'e9_pak': e9_pak,
        'median_seg': median_seg, 'sigma_adapt': sigma_adapt,
        'b1_pak': b1_pak, 'b2_pak': b2_pak, 'b3_pak': b3_pak,
        'b4_pak': b4_pak, 'b5_pak': b5_pak,
    }


def main():
    targets = iter_dataset_aliases()
    print(f"Phase B — Hybrid Methods, {len(targets)} datasets")

    results = {}
    for i, (alias, swat) in enumerate(targets, 1):
        try:
            r = process_one(alias, swat)
            if r is None:
                continue
            results[alias] = r
            print(f"[{i:2d}/{len(targets)}] {alias:<25s} base={r['baseline_pak']:.4f} "
                  f"e9={r['e9_pak']:.4f} B1={r['b1_pak']:.4f} B2={r['b2_pak']:.4f} "
                  f"B3={r['b3_pak']:.4f} B4={r['b4_pak']:.4f} B5={r['b5_pak']:.4f}",
                  flush=True)
        except Exception as e:
            import traceback
            print(f"FAILED {alias}: {e}"); traceback.print_exc()

    out = Path(__file__).parent.parent / 'results' / 'phaseB_hybrid.json'
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")

    print("\n=== Aggregate Δ vs baseline ===")
    for method in ['e9', 'b1', 'b2', 'b3', 'b4', 'b5']:
        deltas = {a: r[f'{method}_pak'] - r['baseline_pak'] for a, r in results.items()}
        baseline_s = [r['baseline_pak'] for r in results.values()]
        method_s = [r[f'{method}_pak'] for r in results.values()]
        mean_d = np.mean(list(deltas.values()))
        wins = sum(1 for d in deltas.values() if d > 0)
        losses = sum(1 for d in deltas.values() if d < 0)
        cata = sum(1 for d in deltas.values() if d < -0.05)
        p = wilcoxon_test(method_s, baseline_s, alternative='greater')
        print(f"  {method:<5s}: meanΔ={mean_d:+.4f}  W/L={wins:2d}/{losses:2d}  cata={cata}  p(>)={p:.3f}")

    # Per-group for best
    print("\n=== Per-group for B2 (Conditional E9) ===")
    deltas_b2 = {a: r['b2_pak'] - r['baseline_pak'] for a, r in results.items()}
    summary = per_group_summary(deltas_b2, get_per_group)
    for g, s in summary.items():
        print(f"  {g:<12s} n={s['n']:2d}  meanΔ={s['mean_delta']:+.4f}  "
              f"W/L={s['wins']}/{s['losses']}  cata={s['catastrophic']}")

    print("\n=== Per-group for B5 (Z5/E9 routing) ===")
    deltas_b5 = {a: r['b5_pak'] - r['baseline_pak'] for a, r in results.items()}
    summary = per_group_summary(deltas_b5, get_per_group)
    for g, s in summary.items():
        print(f"  {g:<12s} n={s['n']:2d}  meanΔ={s['mean_delta']:+.4f}  "
              f"W/L={s['wins']}/{s['losses']}  cata={s['catastrophic']}")


if __name__ == "__main__":
    main()
