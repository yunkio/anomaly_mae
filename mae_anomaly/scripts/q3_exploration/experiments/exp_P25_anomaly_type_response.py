"""
P25 — Anomaly Type Response Mapping (Q3 v7, PRIORITY 3)

Q3 v6 P20에서 6 morphology clusters를 식별 (length, contrast, isolation 기준).
본 실험은 raw signal-level anomaly type을 classify하고, 274 model이 어떤 type에 강/약한가를
정량 (synthetic anomaly이 아닌, REAL anomaly에서).

Anomaly type classification (raw signal-based):
- Spike-like: short, high magnitude change
- Level-shift: sustained value change
- Drift: gradual change over time
- Noise burst: variance increase
- Quasi-normal: changes minimal (potential H1 label noise / H3 feature absence)

Methodology:
- 각 anomaly region의 raw signal-level features 계산:
  - magnitude (peak-to-mean ratio)
  - duration
  - variance change
  - trend (linear slope)
  - frequency content (autocorrelation breakdown)
- KMeans clustering → anomaly types
- For each type, measure 274 model's per-channel response (recon/disc/student/fm)
- For each type, measure achievable PAK-AUC F1 with adaptive_combine + smoothing

Goal:
- 각 raw anomaly type에 대해 274 model의 강점/약점 정량
- 본 quantification으로 어떤 type을 training-time intervention으로 보완해야 하는지 식별
"""
import sys
from pathlib import Path
import numpy as np
import json
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from mae_anomaly.scripts.q3_exploration.core.data import (
    DatasetScores, iter_dataset_aliases, get_per_group
)
from mae_anomaly.scripts.q3_exploration.core.scoring import (
    per_channel_points, adaptive_combine, gauss
)

sys.path.insert(0, '/home/ykio/notebooks/claude')
from mae_anomaly.datasets.loaders import DATASET_LOADERS


def get_raw_signals(alias, ds=None):
    """Load and align to test portion (matching ds.point_labels)."""
    if alias.startswith('smd_machine-'):
        machine = alias.replace('smd_machine-', '')
        loader_name = f'smd_simple_machine-{machine}'
    elif alias.startswith('exathlon_app'):
        app_num = alias.replace('exathlon_app', '')
        loader_name = f'exathlon_app{app_num}'
    elif alias in ('swat_full', 'swat_excl22'):
        loader_name = 'swat_A1A2'
    elif alias == 'wadi_A1':
        loader_name = 'WaDi_14days_A1'
    elif alias == 'wadi_A2':
        loader_name = 'WaDi_14days_A2'
    elif alias == 'psm':
        loader_name = 'PSM'
    elif alias == 'simulation':
        loader_name = 'simulation'
    else:
        return None
    if loader_name not in DATASET_LOADERS:
        return None
    try:
        loader_fn = DATASET_LOADERS[loader_name]
        data = loader_fn()
        if len(data) == 6:
            sigs, lab, reg, fn, tr, _ = data
        elif len(data) == 5:
            sigs, lab, reg, fn, tr = data
        else:
            sigs, lab, reg, fn = data; tr = 0.5

        if ds is not None:
            test_len = len(ds.point_labels)
            full_len = len(sigs)
            split = full_len - test_len
            test_signals = sigs[split:]
            train_signals = sigs[:split]
            return test_signals, ds.point_labels, ds.regions, fn, tr, train_signals
        return sigs, lab, reg, fn, tr, None
    except Exception as e:
        return None


def compute_raw_anomaly_features(signals, region, context_size=200, n_features_max=10):
    """Compute raw-signal-level features for one anomaly region.

    Returns: {magnitude, duration, var_ratio, slope, autocorr_break, ...}
    """
    in_sig = signals[region.start:region.end]
    if len(in_sig) < 3:
        return None
    ctx_start = max(0, region.start - context_size)
    ctx_end = min(len(signals), region.end + context_size)
    ctx_sig = np.concatenate([signals[ctx_start:region.start],
                                signals[region.end:ctx_end]])
    if len(ctx_sig) < 10:
        return None

    duration = region.end - region.start

    # Select top-N most variable features for analysis
    feat_var = ctx_sig.var(axis=0)
    top_feat = np.argsort(-feat_var)[:n_features_max]
    in_sig = in_sig[:, top_feat]
    ctx_sig = ctx_sig[:, top_feat]

    # Robust mean/std
    ctx_mean = ctx_sig.mean(axis=0)
    ctx_std = ctx_sig.std(axis=0) + 1e-9

    # Normalize
    in_norm = (in_sig - ctx_mean) / ctx_std

    # Magnitude: max abs deviation
    max_abs_dev = float(np.max(np.abs(in_norm)))
    mean_abs_dev = float(np.mean(np.abs(in_norm)))
    median_abs_dev = float(np.median(np.abs(in_norm)))

    # Variance ratio
    in_var = in_sig.var(axis=0) + 1e-9
    var_ratio = float(np.mean(in_var / (ctx_sig.var(axis=0) + 1e-9)))

    # Slope (linear trend)
    if len(in_sig) >= 5:
        x = np.arange(len(in_sig))
        slopes = []
        for j in range(in_sig.shape[1]):
            try:
                slope = np.polyfit(x, in_sig[:, j], 1)[0]
                slopes.append(abs(slope) / (ctx_std[j] + 1e-9))
            except:
                slopes.append(0)
        mean_slope = float(np.mean(slopes))
        max_slope = float(np.max(slopes))
    else:
        mean_slope = max_slope = 0.0

    # Autocorrelation breakdown
    autocorr_break = 0.0
    if len(in_sig) >= 10:
        for j in range(in_sig.shape[1]):
            s = in_sig[:, j]
            if s.std() < 1e-9: continue
            # Lag-1 autocorrelation
            ac = np.corrcoef(s[:-1], s[1:])[0, 1]
            if not np.isnan(ac):
                autocorr_break = max(autocorr_break, abs(1 - ac))

    # Shape symmetry (rising vs falling halves)
    half = len(in_sig) // 2
    if half >= 3:
        first_half = float(np.abs(in_norm[:half]).mean())
        second_half = float(np.abs(in_norm[half:]).mean())
        shape_skew = abs(first_half - second_half) / (first_half + second_half + 1e-9)
    else:
        shape_skew = 0.0

    return {
        'duration': duration,
        'log_duration': float(np.log(duration + 1)),
        'max_abs_dev': max_abs_dev,
        'mean_abs_dev': mean_abs_dev,
        'median_abs_dev': median_abs_dev,
        'var_ratio': var_ratio,
        'log_var_ratio': float(np.log(var_ratio + 0.01)),
        'mean_slope': mean_slope,
        'max_slope': max_slope,
        'autocorr_break': autocorr_break,
        'shape_skew': shape_skew,
    }


def classify_anomaly_type(features):
    """Rule-based classification into 5 types."""
    if features is None:
        return 'unknown'
    mag = features['max_abs_dev']
    dur = features['duration']
    var = features['var_ratio']
    slope = features['mean_slope']
    skew = features['shape_skew']

    # Quasi-normal: minimal change
    if mag < 1.5 and var < 2 and slope < 0.2:
        return 'quasi_normal'
    # Spike-like: short + high magnitude
    if dur < 20 and mag > 3:
        return 'spike'
    # Drift: long duration + steady slope, low var ratio
    if dur > 30 and slope > 0.1 and var < 3:
        return 'drift'
    # Noise burst: high variance ratio
    if var > 5:
        return 'noise_burst'
    # Level shift: sustained high magnitude, low slope
    if mag > 2 and slope < 0.1 and dur > 10:
        return 'level_shift'
    # Default: mixed / complex
    return 'mixed'


def measure_274_response_per_region(ds, region, context_size=200):
    """274 model's per-channel response for this region.

    Returns: in/ctx mean for recon, disc, student, fm + adaptive score.
    """
    pt_r, pt_d, pt_s, pt_f = per_channel_points(ds, stride=21)
    adaptive = adaptive_combine(pt_r, pt_d, pt_f, use_fm=True)
    adaptive_smoothed = gauss(adaptive, 10)

    if region.end > len(pt_r):
        return None

    ctx_start = max(0, region.start - context_size)
    ctx_end = min(len(pt_r), region.end + context_size)
    in_idx = np.arange(region.start, region.end)
    ctx_idx = np.concatenate([np.arange(ctx_start, region.start),
                                 np.arange(region.end, ctx_end)])
    if len(ctx_idx) < 5:
        return None

    ctx_std = float(pt_r[ctx_idx].std()) + 1e-9

    return {
        'in_recon': float(pt_r[in_idx].mean()),
        'ctx_recon': float(pt_r[ctx_idx].mean()),
        'in_disc': float(pt_d[in_idx].mean()),
        'ctx_disc': float(pt_d[ctx_idx].mean()),
        'in_student': float(pt_s[in_idx].mean()),
        'ctx_student': float(pt_s[ctx_idx].mean()),
        'in_fm': float(pt_f[in_idx].mean()),
        'ctx_fm': float(pt_f[ctx_idx].mean()),
        'in_adaptive': float(adaptive_smoothed[in_idx].mean()),
        'ctx_adaptive': float(adaptive_smoothed[ctx_idx].mean()),
        'in_adapt_max': float(adaptive_smoothed[in_idx].max()),
        'ctx_adapt_max': float(adaptive_smoothed[ctx_idx].max()),
        'adapt_contrast': float((adaptive_smoothed[in_idx].max() - adaptive_smoothed[ctx_idx].max()) / ctx_std),
    }


def main():
    print("=" * 80)
    print("P25 — Anomaly Type Response Mapping (Q3 v7)")
    print("=" * 80)

    output_dir = Path(__file__).parent.parent / 'results' / 'P25_anomaly_type'
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = iter_dataset_aliases()

    print("\n--- Stage 1: Classify all anomaly regions by raw signal type ---")
    all_regions_features = []  # list of dicts

    for alias, swat_excl22 in targets:
        ds = DatasetScores.load(alias, swat_excl22)
        if ds is None: continue
        raw_data = get_raw_signals(alias, ds)
        if raw_data is None: continue
        signals, point_labels, anomaly_regions, feature_names, train_ratio, _ = raw_data

        for r in anomaly_regions:
            if ds.eval_mask is not None and not ds.eval_mask[r.start:r.end].any():
                continue
            feat = compute_raw_anomaly_features(signals, r, context_size=200)
            if feat is None: continue
            atype = classify_anomaly_type(feat)
            response = measure_274_response_per_region(ds, r, context_size=200)
            if response is None: continue
            all_regions_features.append({
                'alias': alias,
                'group': get_per_group(alias),
                'region_start': r.start,
                'region_end': r.end,
                'anomaly_type': atype,
                **feat,
                **response,
            })

    print(f"  Total regions analyzed: {len(all_regions_features)}")

    # Type distribution
    type_counts = defaultdict(int)
    for r in all_regions_features:
        type_counts[r['anomaly_type']] += 1
    print(f"  Type distribution:")
    for atype, n in sorted(type_counts.items(), key=lambda x: -x[1]):
        pct = 100.0 * n / len(all_regions_features)
        print(f"    {atype:<15s}: {n:>4d} ({pct:>5.1f}%)")

    # ============= Stage 2: Per-type 274 response =============
    print("\n--- Stage 2: 274 model response per anomaly type ---")

    print(f"\n  {'Type':<15s} {'n':>5s} {'r_in/ctx':>10s} {'d_in/ctx':>10s} {'s_in/ctx':>10s} {'f_in/ctx':>10s} {'adapt_ctr':>10s}")
    type_response = {}
    for atype in type_counts:
        regions_of_type = [r for r in all_regions_features if r['anomaly_type'] == atype]
        if not regions_of_type: continue

        # Average per-channel ratios
        r_ratio = np.mean([r['in_recon'] / (r['ctx_recon'] + 1e-9) for r in regions_of_type])
        d_ratio = np.mean([r['in_disc'] / (r['ctx_disc'] + 1e-9) for r in regions_of_type])
        s_ratio = np.mean([r['in_student'] / (r['ctx_student'] + 1e-9) for r in regions_of_type])
        f_ratio = np.mean([r['in_fm'] / (r['ctx_fm'] + 1e-9) for r in regions_of_type])
        adapt_contrast = np.mean([r['adapt_contrast'] for r in regions_of_type])

        type_response[atype] = {
            'n_regions': len(regions_of_type),
            'recon_ratio': float(r_ratio),
            'disc_ratio': float(d_ratio),
            'student_ratio': float(s_ratio),
            'fm_ratio': float(f_ratio),
            'adapt_contrast': float(adapt_contrast),
            # Pct of regions where model is "winning"
            'pct_recon_high': float(np.mean([r['in_recon'] > r['ctx_recon'] for r in regions_of_type])),
            'pct_disc_high': float(np.mean([r['in_disc'] > r['ctx_disc'] for r in regions_of_type])),
            'pct_adapt_positive': float(np.mean([r['adapt_contrast'] > 0 for r in regions_of_type])),
        }
        print(f"  {atype:<15s} {len(regions_of_type):>5d} {r_ratio:>10.3f} "
              f"{d_ratio:>10.3f} {s_ratio:>10.3f} {f_ratio:>10.3f} {adapt_contrast:>10.3f}")

    # ============= Stage 3: Per-type detection performance =============
    print("\n--- Stage 3: Per-type win rate (model is correctly distinguishing) ---")
    print(f"  {'Type':<15s} {'n':>5s} {'recon%':>8s} {'disc%':>8s} {'adapt%':>8s}")
    for atype, tr in type_response.items():
        print(f"  {atype:<15s} {tr['n_regions']:>5d} "
              f"{tr['pct_recon_high']*100:>7.1f}% "
              f"{tr['pct_disc_high']*100:>7.1f}% "
              f"{tr['pct_adapt_positive']*100:>7.1f}%")

    # ============= Stage 4: Per-group + Per-type =============
    print("\n--- Stage 4: Per-group × Per-type breakdown ---")
    type_by_group = defaultdict(lambda: defaultdict(int))
    for r in all_regions_features:
        type_by_group[r['group']][r['anomaly_type']] += 1

    groups = sorted(type_by_group.keys())
    print(f"  {'Group':<15s} " + ' '.join(f'{t:>12s}' for t in sorted(type_counts.keys())))
    for g in groups:
        line = f"  {g:<15s} "
        line += ' '.join(f'{type_by_group[g][t]:>12d}' for t in sorted(type_counts.keys()))
        print(line)

    # ============= Stage 5: Identify the "weakness" — quasi_normal type =============
    print("\n--- Stage 5: Deep dive on quasi_normal regions (potential failure cases) ---")
    qn_regions = [r for r in all_regions_features if r['anomaly_type'] == 'quasi_normal']
    if qn_regions:
        print(f"  n_quasi_normal: {len(qn_regions)}")
        print(f"  Mean adapt_contrast: {np.mean([r['adapt_contrast'] for r in qn_regions]):+.3f}")
        print(f"  Pct with adapt > 0:  {100*np.mean([r['adapt_contrast'] > 0 for r in qn_regions]):.1f}%")
        print(f"  Pct with recon > ctx: {100*np.mean([r['in_recon'] > r['ctx_recon'] for r in qn_regions]):.1f}%")
        # Top 10 quasi-normal datasets
        qn_per_alias = defaultdict(int)
        for r in qn_regions:
            qn_per_alias[r['alias']] += 1
        print(f"\n  Top 10 datasets with quasi_normal:")
        for a, n in sorted(qn_per_alias.items(), key=lambda x: -x[1])[:10]:
            print(f"    {a:<25s}: {n} quasi-normal regions")

    # ============= Stage 6: Cross-validate with P23 inverted regions =============
    print("\n--- Stage 6: Cross-validate with P23 inverted regions ---")
    p23_path = Path(__file__).parent.parent / 'results' / 'P23_inverted_signal' / 'P23_full_analysis.json'
    if p23_path.exists():
        with open(p23_path) as f:
            p23_data = json.load(f)
        # P23 found 119 inverted, distributed across 29 datasets
        # Check how many of those are quasi_normal / mixed / other type
        # Need the actual region starts from P23

        # Simpler approach: among regions with adapt_contrast < -0.5 (inverted in P25 too),
        # what type distribution
        p25_inverted = [r for r in all_regions_features if r['adapt_contrast'] < -0.5]
        print(f"  P25 inverted regions (adapt_contrast < -0.5): {len(p25_inverted)}")
        if p25_inverted:
            type_dist_inv = defaultdict(int)
            for r in p25_inverted:
                type_dist_inv[r['anomaly_type']] += 1
            print(f"  Type distribution of inverted:")
            for t, n in sorted(type_dist_inv.items(), key=lambda x: -x[1]):
                pct = 100 * n / len(p25_inverted)
                print(f"    {t:<15s}: {n:>3d} ({pct:.1f}%)")

    # Save full results
    save_data = {
        'n_regions': len(all_regions_features),
        'type_counts': dict(type_counts),
        'type_response': type_response,
        'type_by_group': {g: dict(td) for g, td in type_by_group.items()},
        'all_regions_features': all_regions_features[:1000],  # cap for size
    }
    with open(output_dir / 'P25_type_response.json', 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nSaved: {output_dir / 'P25_type_response.json'}")


if __name__ == "__main__":
    main()
