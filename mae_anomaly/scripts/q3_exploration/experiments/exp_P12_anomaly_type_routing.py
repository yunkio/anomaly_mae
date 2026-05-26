"""
P12 — Anomaly Type Classification + Type-Conditional Method

Anomaly에는 다양한 type이 있다:
- Point spike: 짧고 sharp (e.g., 1-5 timesteps)
- Short burst: 10-50 timesteps with high SNR
- Mid-duration: 50-300 timesteps
- Long drift: 300+ timesteps (regime change)
- Multi-burst: 다중 anomaly 군집

각 type에 다른 method가 적합:
- Point spike → small σ + max aggregation
- Long drift → large σ + multi-scale
- Multi-burst → ensemble

본 P12는:
1. Per-anomaly-region에서 type을 estimate (length + sharpness + isolation)
2. Dataset 단위로 type proportion 계산
3. Type proportion을 feature로 사용해 method selection
4. Hierarchical routing (type → method)

본 작업은 dataset clustering (P1, P8)와 다른 angle:
- Clustering = dataset characteristic 기반
- Type routing = anomaly characteristic 기반 (더 fine-grained)
"""
import sys
from pathlib import Path
import numpy as np
import json
from collections import defaultdict
from scipy.stats import kurtosis

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


def classify_anomaly_regions(regions, score_smoothed):
    """각 anomaly region을 type별로 classify.

    Types:
    - 'point_spike': length < 5
    - 'short_burst': 5 <= length < 50
    - 'mid_duration': 50 <= length < 300
    - 'long_drift': length >= 300

    Returns:
        type_counts: dict {type_name: count}
        type_proportions: dict {type_name: fraction}
    """
    type_counts = defaultdict(int)
    region_lengths = []

    for r in regions:
        length = r.end - r.start
        region_lengths.append(length)
        if length < 5:
            type_counts['point_spike'] += 1
        elif length < 50:
            type_counts['short_burst'] += 1
        elif length < 300:
            type_counts['mid_duration'] += 1
        else:
            type_counts['long_drift'] += 1

    total = max(sum(type_counts.values()), 1)
    type_proportions = {k: v / total for k, v in type_counts.items()}
    # Ensure all 4 types are present
    for t in ['point_spike', 'short_burst', 'mid_duration', 'long_drift']:
        type_proportions.setdefault(t, 0.0)
        type_counts.setdefault(t, 0)

    # SNR estimate per region
    snrs = []
    for r in regions:
        region_score = score_smoothed[r.start:r.end]
        outside = np.concatenate([score_smoothed[:r.start], score_smoothed[r.end:]])
        if len(outside) < 10 or outside.std() < 1e-9:
            snrs.append(1.0)
        else:
            snr = (region_score.mean() - outside.mean()) / outside.std()
            snrs.append(snr)
    mean_snr = float(np.mean(snrs)) if snrs else 0.0

    return {
        'type_counts': dict(type_counts),
        'type_proportions': type_proportions,
        'n_regions': len(regions),
        'mean_snr': mean_snr,
        'median_length': float(np.median(region_lengths)) if region_lengths else 0.0,
        'length_skew': float(np.log(np.max(region_lengths) + 1) - np.log(np.median(region_lengths) + 1))
                        if region_lengths else 0.0,
        'mean_length': float(np.mean(region_lengths)) if region_lengths else 0.0,
    }


def type_conditional_method(base_unsmoothed, type_proportions, median_seg):
    """Type proportion 기반 method selection.

    Rules:
    - dominant point_spike → small σ (median_seg / 3 lower bound 3)
    - dominant short_burst → mid σ (median_seg / 5)
    - dominant mid_duration → mid-large σ (median_seg / 5)
    - dominant long_drift → very large σ + multi-scale

    Mixed: blended approach
    """
    p_point = type_proportions['point_spike']
    p_short = type_proportions['short_burst']
    p_mid = type_proportions['mid_duration']
    p_long = type_proportions['long_drift']

    # Find dominant type
    dominant = max(type_proportions, key=type_proportions.get)

    if dominant == 'point_spike':
        sigma = max(median_seg / 3.0, 1.0)
        smoothed = gauss(base_unsmoothed, sigma)
        return nlm_sigmoid_transform(smoothed, T_factor=2.0)
    elif dominant == 'short_burst':
        sigma = max(median_seg / 5.0, 1.5)
        smoothed = gauss(base_unsmoothed, sigma)
        return nlm_sigmoid_transform(smoothed, T_factor=1.5)
    elif dominant == 'mid_duration':
        sigma = max(median_seg / 5.0, 5.0)
        smoothed = gauss(base_unsmoothed, sigma)
        return nlm_sigmoid_transform(smoothed, T_factor=1.5)
    else:  # long_drift
        sigma = max(median_seg / 4.0, 30.0)
        # Cap to prevent over-smoothing
        sigma = min(sigma, 100.0)
        smoothed = gauss(base_unsmoothed, sigma)
        return nlm_sigmoid_transform(smoothed, T_factor=1.0)


def adaptive_type_blend(base_unsmoothed, type_proportions, median_seg):
    """Continuous adaptation: blend methods weighted by type proportions."""
    # 각 type에 specific method 적용 후 weighted blend
    p_point = type_proportions['point_spike']
    p_short = type_proportions['short_burst']
    p_mid = type_proportions['mid_duration']
    p_long = type_proportions['long_drift']

    # σ per type
    sigma_point = max(median_seg / 3.0, 1.0)
    sigma_short = max(median_seg / 5.0, 1.5)
    sigma_mid = max(median_seg / 5.0, 5.0)
    sigma_long = min(max(median_seg / 4.0, 30.0), 100.0)

    # Each smoothed score
    score_point = nlm_sigmoid_transform(gauss(base_unsmoothed, sigma_point), T_factor=2.0)
    score_short = nlm_sigmoid_transform(gauss(base_unsmoothed, sigma_short), T_factor=1.5)
    score_mid = nlm_sigmoid_transform(gauss(base_unsmoothed, sigma_mid), T_factor=1.5)
    score_long = nlm_sigmoid_transform(gauss(base_unsmoothed, sigma_long), T_factor=1.0)

    # z-norm + weighted sum
    blended = (p_point * zscore(score_point) +
                p_short * zscore(score_short) +
                p_mid * zscore(score_mid) +
                p_long * zscore(score_long))
    return blended


def main():
    targets = iter_dataset_aliases()
    print(f"P12 — Anomaly Type Classification + Type-Conditional Method, {len(targets)} datasets")

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

        # Classify anomaly types
        type_info = classify_anomaly_regions(ds.regions, base_smoothed)

        # Method 1: discrete type-conditional
        score_discrete = type_conditional_method(base_unsmoothed, type_info['type_proportions'], median_seg)
        pak_discrete = pak_auc_f1(score_discrete, ds.point_labels, ds.regions, ds.eval_mask)

        # Method 2: continuous blend
        score_blend = adaptive_type_blend(base_unsmoothed, type_info['type_proportions'], median_seg)
        pak_blend = pak_auc_f1(score_blend, ds.point_labels, ds.regions, ds.eval_mask)

        # Reference: div5.0_T1.5
        ref_score = nlm_sigmoid_transform(gauss(base_unsmoothed, max(median_seg/5, 0.5)), T_factor=1.5)
        ref_pak = pak_auc_f1(ref_score, ds.point_labels, ds.regions, ds.eval_mask)

        all_results[alias] = {
            'baseline_pak': baseline_pak,
            'ref_pak_div5_T15': ref_pak,
            'discrete_type_pak': pak_discrete,
            'blend_type_pak': pak_blend,
            'type_info': type_info,
            'median_seg': median_seg,
            'dominant_type': max(type_info['type_proportions'],
                                  key=type_info['type_proportions'].get),
        }

        if i % 10 == 0 or i == len(targets):
            print(f"  [{i:2d}/{len(targets)}] processed", flush=True)

    # Save
    out = Path(__file__).parent.parent / 'results' / 'P12_anomaly_type.json'
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {out}")

    # Analysis
    print("\n=== Type distribution across 39 datasets ===")
    type_dist = defaultdict(int)
    for r in all_results.values():
        type_dist[r['dominant_type']] += 1
    for t, n in sorted(type_dist.items()):
        print(f"  {t:<20s}: {n} datasets")

    # Aggregate Δ
    print("\n=== Aggregate Δ vs baseline ===")
    baseline_paks = [r['baseline_pak'] for r in all_results.values()]
    for method_key, method_name in [('ref_pak_div5_T15', 'ref div5_T1.5'),
                                      ('discrete_type_pak', 'discrete type-cond'),
                                      ('blend_type_pak', 'continuous type-blend')]:
        scores = [r[method_key] for r in all_results.values()]
        deltas = np.array(scores) - np.array(baseline_paks)
        mean_d = float(deltas.mean())
        wins = int((deltas > 0).sum())
        losses = int((deltas < 0).sum())
        cata = int((deltas < -0.05).sum())
        p = wilcoxon_test(scores, baseline_paks, alternative='greater')
        print(f"  {method_name:<25s}: meanΔ={mean_d:+.4f}  W/L={wins:2d}/{losses:2d}  cata={cata}  p={p:.4f}")

    # Per-type breakdown
    print("\n=== Per-dominant-type breakdown (discrete method) ===")
    type_to_results = defaultdict(list)
    for r in all_results.values():
        dt = r['dominant_type']
        delta = r['discrete_type_pak'] - r['baseline_pak']
        type_to_results[dt].append(delta)

    for t in sorted(type_to_results):
        deltas = np.array(type_to_results[t])
        print(f"  {t:<20s} n={len(deltas):2d}  meanΔ={deltas.mean():+.4f}  "
              f"W/L={(deltas>0).sum()}/{(deltas<0).sum()}  cata={(deltas < -0.05).sum()}")

    # Top-5 type-conditional successes
    print("\n=== Top-5 datasets where discrete type method WINS most over ref ===")
    delta_over_ref = [(a, r['discrete_type_pak'] - r['ref_pak_div5_T15'],
                       r['dominant_type'], r['median_seg'])
                      for a, r in all_results.items()]
    delta_over_ref.sort(key=lambda x: -x[1])
    print(f"{'alias':<25s} {'Δ vs ref':>10s} {'dominant_type':<18s} {'med_seg':>8s}")
    for a, d, t, ms in delta_over_ref[:5]:
        print(f"{a:<25s} {d:>+10.4f} {t:<18s} {ms:>8.1f}")
    print("\n=== Bottom-5 (where discrete type LOSES to ref) ===")
    for a, d, t, ms in delta_over_ref[-5:]:
        print(f"{a:<25s} {d:>+10.4f} {t:<18s} {ms:>8.1f}")


if __name__ == "__main__":
    main()
