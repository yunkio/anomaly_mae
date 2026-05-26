"""
P23 — Inverted Signal Anomaly Investigation (Q3 v7, PRIORITY 1)

Q3 v6 P20에서 식별된 cluster 3 (n=98) inverted contrast anomalies의 본질 deep investigation.

본 실험은 4 hypothesis를 정량 검증:

H1 (Label Noise): 본 regions는 실제로 normal이며 mislabeled
H2 (Reverse Learning): Model이 anomaly type을 잘 reconstruct (low recon error)
H3 (Feature Absence): Raw signal에 anomaly signal distinct하지 않음
H4 (Training Contamination): Train data에 anomaly-like patterns 자주 등장

For each hypothesis:
- Stage 1: Identify inverted regions across 39 datasets
- Stage 2: H1 check (label vs raw signal)
- Stage 3: H2 check (per-region recon analysis)
- Stage 4: H3 check (raw signal distance)
- Stage 5: H4 check (train data similarity)
- Stage 6: Aggregate evidence for each hypothesis
- Stage 7: Per-dataset diagnostic

Output: Comprehensive analysis + visualization for hypothesis ranking.
"""
import sys
from pathlib import Path
import numpy as np
import json
import time
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from mae_anomaly.scripts.q3_exploration.core.data import (
    DatasetScores, iter_dataset_aliases, median_anomaly_segment_length, get_per_group
)
from mae_anomaly.scripts.q3_exploration.core.scoring import (
    per_channel_points, adaptive_combine, gauss
)
from mae_anomaly.scripts.q3_exploration.core.inverted_signal_analysis import (
    identify_inverted_regions, per_region_recon_analysis,
    raw_signal_distance_analysis, find_similar_training_patterns,
)


# Load raw signals (datasets module from mae_anomaly)
sys.path.insert(0, '/home/ykio/notebooks/claude')
from mae_anomaly.datasets.loaders import DATASET_LOADERS


def get_raw_signals(alias, ds=None):
    """Load raw signals for a dataset alias and align to test portion."""
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
            signals, point_labels, anomaly_regions, feature_names, train_ratio, info = data
        elif len(data) == 5:
            signals, point_labels, anomaly_regions, feature_names, train_ratio = data
        else:
            signals, point_labels, anomaly_regions, feature_names = data
            train_ratio = 0.5

        # Align signals to test portion if ds provided
        if ds is not None:
            test_len = len(ds.point_labels)
            full_len = len(signals)
            split = full_len - test_len
            # train_signals[:split], test_signals = signals[split:]
            # But find_similar_training_patterns expects full signals + train_ratio.
            # We provide aligned wrapper data.
            test_signals = signals[split:]
            train_signals = signals[:split]
            # Reconstruct "full" array where indices are test-relative for raw distance,
            # and provide separate train_signals for H4
            return test_signals, ds.point_labels, ds.regions, train_ratio, train_signals
        return signals, point_labels, anomaly_regions, train_ratio, None
    except Exception as e:
        print(f"  Error loading {alias}: {e}")
        return None


def main():
    print("=" * 80)
    print("P23 — Inverted Signal Anomaly Investigation (Q3 v7)")
    print("=" * 80)

    output_dir = Path(__file__).parent.parent / 'results' / 'P23_inverted_signal'
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = iter_dataset_aliases()

    # ================ Stage 1: Identify inverted regions ================
    print("\n--- Stage 1: Identify inverted regions across 39 datasets ---")

    all_inverted = {}
    all_normal_anomalies = {}

    for alias, swat_excl22 in targets:
        ds = DatasetScores.load(alias, swat_excl22)
        if ds is None: continue
        pt_r, pt_d, pt_s, pt_f = per_channel_points(ds, stride=21)
        base = adaptive_combine(pt_r, pt_d, pt_f, use_fm=True)
        base_smoothed = gauss(base, 10)

        regions_info = identify_inverted_regions(base_smoothed, ds.regions,
                                                    ds.eval_mask, context_size=200,
                                                    min_contrast=-0.5)
        inverted = [r for r in regions_info if r['is_inverted']]
        normal = [r for r in regions_info if not r['is_inverted']]

        if inverted:
            all_inverted[alias] = inverted
            all_normal_anomalies[alias] = normal

    n_total_inverted = sum(len(v) for v in all_inverted.values())
    print(f"  Total inverted regions identified: {n_total_inverted} across {len(all_inverted)} datasets")
    print(f"  Per-dataset count distribution:")
    for alias, regions in sorted(all_inverted.items(), key=lambda x: -len(x[1]))[:10]:
        print(f"    {alias:<25s}: {len(regions)} inverted regions")

    # ================ Stage 2: H1 Label Noise — check label vs raw ================
    print("\n--- Stage 2: H1 Label Noise hypothesis check ---")
    print("  (Examining raw signal pattern in inverted regions vs normal)")

    h1_evidence = {}
    # Use median + robust statistics to avoid outliers
    for alias in list(all_inverted.keys())[:15]:
        ds = DatasetScores.load(alias, alias == 'swat_excl22')
        if ds is None: continue
        raw_data = get_raw_signals(alias, ds)
        if raw_data is None: continue
        signals, labels, _, _, _ = raw_data  # signals is test portion

        def signal_distance(ws_start, ws_end, ctx_size=200):
            in_sig = signals[ws_start:ws_end]
            if len(in_sig) < 3: return None
            ctx_start = max(0, ws_start - ctx_size)
            ctx_end = min(len(signals), ws_end + ctx_size)
            ctx_sig = np.concatenate([signals[ctx_start:ws_start],
                                        signals[ws_end:ctx_end]])
            if len(ctx_sig) < 10: return None
            in_mean = in_sig.mean(axis=0)
            ctx_mean = ctx_sig.mean(axis=0)
            ctx_std = ctx_sig.std(axis=0) + 1e-9
            return float(np.linalg.norm((in_mean - ctx_mean) / ctx_std))

        inv_regions = all_inverted[alias]
        inv_distances = [d for d in [signal_distance(r['region'].start, r['region'].end)
                                       for r in inv_regions] if d is not None]
        norm_distances = [d for d in [signal_distance(r['region'].start, r['region'].end)
                                        for r in all_normal_anomalies[alias][:50]]
                          if d is not None]

        if inv_distances and norm_distances:
            h1_evidence[alias] = {
                'n_inv': len(inv_distances),
                'n_norm': len(norm_distances),
                'inv_mean_distance': float(np.mean(inv_distances)),
                'inv_median_distance': float(np.median(inv_distances)),
                'norm_mean_distance': float(np.mean(norm_distances)),
                'norm_median_distance': float(np.median(norm_distances)),
                # H1 verdict: if inv_distance ≈ background ≈ 0, label noise likely
                'distance_ratio': float(np.mean(inv_distances) / (np.mean(norm_distances) + 1e-9)),
            }

    print(f"  H1 Label Noise evidence (10 datasets):")
    print(f"  {'Dataset':<25s} {'n_inv':>6s} {'inv_dist':>10s} {'norm_dist':>10s} {'ratio':>8s}")
    for alias, ev in h1_evidence.items():
        print(f"  {alias:<25s} {ev['n_inv']:>6d} {ev['inv_mean_distance']:>10.3f} {ev['norm_mean_distance']:>10.3f} {ev['distance_ratio']:>8.3f}")

    # Interpretation
    if h1_evidence:
        mean_ratio = np.mean([ev['distance_ratio'] for ev in h1_evidence.values()])
        print(f"\n  Mean distance ratio (inv / norm): {mean_ratio:.3f}")
        print(f"  H1 interpretation: ratio << 1 → inverted regions are NORMAL-like (label noise)")
        print(f"  ratio ~ 1 → inverted regions are AS DIFFERENT as normal anomalies (other hypotheses)")
        print(f"  ratio > 1 → inverted regions are EVEN MORE distinct than normal anomalies")

    # ================ Stage 3: H2 Reverse Learning — per-region recon analysis ================
    print("\n--- Stage 3: H2 Reverse Learning hypothesis check ---")
    print("  (Per-region recon error: model이 inverted anomaly를 더 잘 reconstruct하는가?)")

    h2_evidence = {}
    for alias, inv_regions in list(all_inverted.items())[:10]:
        ds = DatasetScores.load(alias, alias == 'swat_excl22')
        if ds is None: continue

        recon_results = per_region_recon_analysis(
            ds.recon, ds.disc, ds.fm, ds.window_start_indices,
            ds.regions, ds.num_patches, ds.patch_size, ds.total_length,
            ds.eval_mask
        )

        inv_region_starts = set(r['region'].start for r in inv_regions)
        inv_recon = [r for r in recon_results if r['region_start'] in inv_region_starts]
        norm_recon = [r for r in recon_results if r['region_start'] not in inv_region_starts]

        if not inv_recon or not norm_recon: continue

        # Recon ratio
        inv_recon_ratios = [r['recon_ratio'] for r in inv_recon]
        norm_recon_ratios = [r['recon_ratio'] for r in norm_recon]

        # Disc ratio (teacher-student gap)
        inv_disc_ratios = [r['disc_ratio'] for r in inv_recon]
        norm_disc_ratios = [r['disc_ratio'] for r in norm_recon]

        # Inverted count: how many of inv regions have all-channels-inverted
        inv_with_recon_inverted = sum(r['recon_inverted'] for r in inv_recon)
        inv_with_all_inverted = sum(r['all_channels_inverted'] for r in inv_recon)

        h2_evidence[alias] = {
            'n_inv': len(inv_recon),
            'inv_mean_recon_ratio': float(np.mean(inv_recon_ratios)),  # < 1 = anomaly recon is BETTER
            'norm_mean_recon_ratio': float(np.mean(norm_recon_ratios)),
            'inv_with_recon_inverted': inv_with_recon_inverted,
            'inv_with_all_inverted': inv_with_all_inverted,
            'inv_with_recon_inverted_pct': inv_with_recon_inverted / len(inv_recon) * 100,
        }

    print(f"  H2 Reverse Learning evidence:")
    print(f"  {'Dataset':<25s} {'n_inv':>6s} {'inv_R':>8s} {'norm_R':>8s} {'recInv%':>8s} {'allInv':>7s}")
    for alias, ev in h2_evidence.items():
        print(f"  {alias:<25s} {ev['n_inv']:>6d} {ev['inv_mean_recon_ratio']:>8.3f} "
              f"{ev['norm_mean_recon_ratio']:>8.3f} {ev['inv_with_recon_inverted_pct']:>7.1f}% "
              f"{ev['inv_with_all_inverted']:>7d}")
    print(f"\n  H2 interpretation: inv_R < 1 → model reconstructs anomaly BETTER than context")
    print(f"  This is consistent with reverse learning")

    # ================ Stage 4: H3 Feature Absence — raw signal distance ================
    print("\n--- Stage 4: H3 Feature Absence hypothesis check ---")
    print("  (Raw signal에서 anomaly position이 distinct한가?)")

    h3_evidence = {}
    for alias, inv_regions in list(all_inverted.items())[:15]:
        ds = DatasetScores.load(alias, alias == 'swat_excl22')
        if ds is None: continue
        raw_data = get_raw_signals(alias, ds)
        if raw_data is None: continue
        signals, _, _, _, _ = raw_data

        # Inverted regions
        inv_actual_regions = [r['region'] for r in inv_regions]
        inv_dist = raw_signal_distance_analysis(signals, inv_actual_regions,
                                                  context_size=200)

        # Normal anomalies
        norm_actual_regions = [r['region'] for r in all_normal_anomalies[alias][:50]]
        norm_dist = raw_signal_distance_analysis(signals, norm_actual_regions,
                                                   context_size=200)

        if not inv_dist or not norm_dist: continue

        h3_evidence[alias] = {
            'n_inv': len(inv_dist),
            'inv_max_std_diff': float(np.mean([r['max_std_dim_diff'] for r in inv_dist])),
            'inv_mean_std_diff': float(np.mean([r['mean_std_dim_diff'] for r in inv_dist])),
            'inv_var_ratio': float(np.mean([r['var_ratio'] for r in inv_dist])),
            'inv_wasserstein': float(np.mean([r['mean_wasserstein'] for r in inv_dist])),
            'norm_max_std_diff': float(np.mean([r['max_std_dim_diff'] for r in norm_dist])),
            'norm_mean_std_diff': float(np.mean([r['mean_std_dim_diff'] for r in norm_dist])),
            'norm_var_ratio': float(np.mean([r['var_ratio'] for r in norm_dist])),
            'norm_wasserstein': float(np.mean([r['mean_wasserstein'] for r in norm_dist])),
        }

    print(f"  H3 Feature Absence evidence:")
    print(f"  {'Dataset':<25s} {'n_inv':>6s} {'inv_max':>8s} {'norm_max':>9s} {'inv_w':>8s} {'norm_w':>8s}")
    for alias, ev in h3_evidence.items():
        print(f"  {alias:<25s} {ev['n_inv']:>6d} {ev['inv_max_std_diff']:>8.2f} "
              f"{ev['norm_max_std_diff']:>9.2f} {ev['inv_wasserstein']:>8.3f} {ev['norm_wasserstein']:>8.3f}")
    print(f"\n  H3 interpretation: inv_max << norm_max → features lack anomaly signal (H3 supported)")

    # ================ Stage 5: H4 Training Contamination ================
    print("\n--- Stage 5: H4 Training Contamination hypothesis check ---")
    print("  (Train data에서 anomaly-like patterns frequency)")

    h4_evidence = {}
    for alias, inv_regions in list(all_inverted.items())[:15]:
        ds = DatasetScores.load(alias, alias == 'swat_excl22')
        if ds is None: continue
        raw_data = get_raw_signals(alias, ds)
        if raw_data is None: continue
        test_signals, _, _, train_ratio, train_signals = raw_data
        if train_signals is None or len(train_signals) < 200: continue

        # H4: provide train_signals separately. Use a wrapper that
        # uses train_signals[: split] as train, test_signals as the
        # space where anomaly regions are defined.
        # Compute neighbors in train data vs in_centroid from test
        def contamination_for_region(region):
            in_sig = test_signals[region.start:region.end]
            if len(in_sig) < 3: return None
            region_len = region.end - region.start
            in_centroid = in_sig.mean(axis=0)
            # Sliding train windows
            n_windows = max(1, (len(train_signals) - region_len) // 10)
            if n_windows < 5: return None
            train_centroids = np.array([
                train_signals[i*10 : i*10 + region_len].mean(axis=0)
                for i in range(n_windows)
                if i*10 + region_len <= len(train_signals)
            ])
            if len(train_centroids) == 0: return None
            distances = np.linalg.norm(train_centroids - in_centroid, axis=1)
            # Use median test distance as scale
            test_centroids = []
            for i in range(min(100, (len(test_signals) - region_len) // 10)):
                test_centroids.append(test_signals[i*10 : i*10 + region_len].mean(axis=0))
            if len(test_centroids) > 0:
                test_centroids = np.array(test_centroids)
                test_dists = np.linalg.norm(test_centroids - in_centroid, axis=1)
                test_median = float(np.median(test_dists))
            else:
                test_median = float(np.median(distances))
            return {
                'region_start': region.start, 'region_end': region.end,
                'train_min_distance': float(distances.min()),
                'train_median_distance': float(np.median(distances)),
                'contamination_ratio': float((distances <= test_median).mean()),
            }

        contamination = [c for c in [contamination_for_region(r['region'])
                                      for r in inv_regions] if c]
        norm_contam = [c for c in [contamination_for_region(r['region'])
                                    for r in all_normal_anomalies[alias][:30]] if c]
        if not contamination: continue

        h4_evidence[alias] = {
            'n_inv': len(contamination),
            'inv_mean_contam': float(np.mean([r['contamination_ratio'] for r in contamination])),
            'inv_train_min': float(np.mean([r['train_min_distance'] for r in contamination])),
            'inv_train_median': float(np.mean([r['train_median_distance'] for r in contamination])),
            'norm_mean_contam': float(np.mean([r['contamination_ratio'] for r in norm_contam])) if norm_contam else 0,
            'norm_train_min': float(np.mean([r['train_min_distance'] for r in norm_contam])) if norm_contam else 0,
        }

    print(f"  H4 Training Contamination evidence:")
    print(f"  {'Dataset':<25s} {'n_inv':>6s} {'inv_contam':>11s} {'norm_contam':>12s} {'inv_min':>9s} {'norm_min':>9s}")
    for alias, ev in h4_evidence.items():
        print(f"  {alias:<25s} {ev['n_inv']:>6d} {ev['inv_mean_contam']:>11.3f} "
              f"{ev['norm_mean_contam']:>12.3f} {ev['inv_train_min']:>9.3f} {ev['norm_train_min']:>9.3f}")
    print(f"\n  H4 interpretation: inv_contam > norm_contam → train data has anomaly-like patterns (H4 supported)")

    # ================ Stage 6: Aggregate hypothesis evidence ================
    print("\n" + "=" * 80)
    print("STAGE 6: Aggregate Hypothesis Ranking")
    print("=" * 80)

    # H1 score: how much inverted regions are normal-like (closer to context)
    # H2 score: reverse-learning ratio (how often recon is inverted)
    # H3 score: feature absence (inv_max_std_diff << norm_max_std_diff)
    # H4 score: training contamination (inv_contam > norm_contam)

    aggregated = {}
    if h1_evidence:
        h1_ratios = [ev['distance_ratio'] for ev in h1_evidence.values()]
        aggregated['H1_label_noise'] = {
            'mean_distance_ratio': float(np.mean(h1_ratios)),
            'evidence_strength': float(1.0 / (np.mean(h1_ratios) + 1e-9)),  # higher = stronger
            'verdict': 'SUPPORTED' if np.mean(h1_ratios) < 0.5 else 'WEAK' if np.mean(h1_ratios) < 1.0 else 'NOT SUPPORTED',
        }

    if h2_evidence:
        h2_recon_ratios = [ev['inv_mean_recon_ratio'] for ev in h2_evidence.values()]
        h2_inv_pcts = [ev['inv_with_recon_inverted_pct'] for ev in h2_evidence.values()]
        aggregated['H2_reverse_learning'] = {
            'mean_recon_ratio': float(np.mean(h2_recon_ratios)),
            'mean_recon_inverted_pct': float(np.mean(h2_inv_pcts)),
            'evidence_strength': float((np.mean(h2_inv_pcts) - 50) / 50),  # > 0 if > 50% inverted
            'verdict': 'SUPPORTED' if np.mean(h2_inv_pcts) > 50 else 'NOT SUPPORTED',
        }

    if h3_evidence:
        # Ratio of inv vs norm distinctiveness
        h3_max_ratios = [ev['inv_max_std_diff'] / (ev['norm_max_std_diff'] + 1e-9)
                         for ev in h3_evidence.values()]
        aggregated['H3_feature_absence'] = {
            'mean_max_dist_ratio': float(np.mean(h3_max_ratios)),
            'evidence_strength': float(1.0 / (np.mean(h3_max_ratios) + 1e-9)),
            'verdict': 'SUPPORTED' if np.mean(h3_max_ratios) < 0.5 else 'WEAK' if np.mean(h3_max_ratios) < 1.0 else 'NOT SUPPORTED',
        }

    if h4_evidence:
        h4_contam_diffs = [ev['inv_mean_contam'] - ev['norm_mean_contam']
                            for ev in h4_evidence.values()]
        aggregated['H4_training_contamination'] = {
            'mean_contam_diff': float(np.mean(h4_contam_diffs)),
            'evidence_strength': float(np.mean(h4_contam_diffs)),
            'verdict': 'SUPPORTED' if np.mean(h4_contam_diffs) > 0.05 else 'WEAK' if np.mean(h4_contam_diffs) > 0 else 'NOT SUPPORTED',
        }

    print("\nHypothesis ranking:")
    ranked = sorted(aggregated.items(), key=lambda x: -x[1].get('evidence_strength', 0))
    for h_name, h_data in ranked:
        print(f"\n  {h_name}:")
        for k, v in h_data.items():
            if isinstance(v, str):
                print(f"    {k}: {v}")
            else:
                print(f"    {k}: {v:.3f}")

    # Save all evidence
    save_data = {
        'n_total_inverted': n_total_inverted,
        'n_datasets_with_inverted': len(all_inverted),
        'h1_evidence_per_dataset': h1_evidence,
        'h2_evidence_per_dataset': h2_evidence,
        'h3_evidence_per_dataset': h3_evidence,
        'h4_evidence_per_dataset': h4_evidence,
        'aggregated_hypothesis_ranking': aggregated,
        'per_dataset_inverted_count': {a: len(v) for a, v in all_inverted.items()},
    }
    with open(output_dir / 'P23_full_analysis.json', 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nSaved: {output_dir / 'P23_full_analysis.json'}")


if __name__ == "__main__":
    main()
