"""
P24 — Per-Feature Importance Analysis (Q3 v7, PRIORITY 2)

Q3 v6 P19 oracle channel mix가 +0.0428 ceiling을 보였지만, channel은 4개 (r/d/s/f)에 불과.
본 실험은 raw signal의 8 features 중 어떤 것이 anomaly detection에 가장 기여하는가를 정량.

Goal: Feature-level intervention 가능성을 정량
- 본 ceiling은 model-level channel mixing이 아닌 input-level feature selection으로 도달 가능한가?
- Per-dataset oracle feature subset의 closure 가능성

Hypothesis:
- H1: Top-K features (K=2-4)만으로도 ≥80% 성능 달성 가능 → input-level optimization 가능
- H2: Inverted anomaly regions는 specific feature set lacking → P23 H3 cross-check
- H3: 각 anomaly cluster (P20)에 대해 critical feature는 다르다 → multi-modal feature attribution
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
from mae_anomaly.scripts.q3_exploration.core.evaluation import pak_auc_f1
from mae_anomaly.scripts.q3_exploration.core.feature_attribution import (
    per_feature_anomaly_separation, per_feature_top_k_importance,
    feature_correlation_matrix, feature_dimensionality_reduction,
)

sys.path.insert(0, '/home/ykio/notebooks/claude')
from mae_anomaly.datasets.loaders import DATASET_LOADERS


def get_raw_signals(alias, ds=None):
    """Load raw signals for a dataset alias and align to test portion.

    Returns: (test_signals, test_point_labels, test_regions, feature_names, train_ratio, train_signals).
    test_signals length == ds.point_labels length.
    """
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

        # Align to test portion
        if ds is not None:
            test_len = len(ds.point_labels)
            full_len = len(signals)
            split = full_len - test_len
            train_signals = signals[:split]
            test_signals = signals[split:]
            test_labels = ds.point_labels
            return test_signals, test_labels, ds.regions, feature_names, train_ratio, train_signals
        else:
            return signals, point_labels, anomaly_regions, feature_names, train_ratio, None
    except Exception as e:
        print(f"  Error loading {alias}: {e}")
        return None


def simple_per_feature_score(signals, sigma=10):
    """Per-feature z-score based anomaly score (no training needed).

    각 feature를 independent z-score로 처리, max across features.
    """
    from scipy.ndimage import gaussian_filter1d, uniform_filter1d

    n_features = signals.shape[1]
    window_size = 200
    point_scores = np.zeros(signals.shape[0])

    for j in range(n_features):
        feat = signals[:, j]
        # Rolling mean + std
        rolling_mean = uniform_filter1d(feat, size=window_size, mode='reflect')
        rolling_var = uniform_filter1d(feat ** 2, size=window_size, mode='reflect') - rolling_mean ** 2
        rolling_std = np.sqrt(np.maximum(rolling_var, 1e-9))
        z = np.abs((feat - rolling_mean) / rolling_std)
        point_scores = np.maximum(point_scores, z)

    smoothed = gaussian_filter1d(point_scores, sigma=sigma, mode='reflect')
    return smoothed


def per_feature_subset_score(signals, feature_indices, sigma=10):
    """Score using only subset of features."""
    from scipy.ndimage import gaussian_filter1d, uniform_filter1d

    subset = signals[:, feature_indices]
    window_size = 200
    point_scores = np.zeros(signals.shape[0])

    for j in range(subset.shape[1]):
        feat = subset[:, j]
        rolling_mean = uniform_filter1d(feat, size=window_size, mode='reflect')
        rolling_var = uniform_filter1d(feat ** 2, size=window_size, mode='reflect') - rolling_mean ** 2
        rolling_std = np.sqrt(np.maximum(rolling_var, 1e-9))
        z = np.abs((feat - rolling_mean) / rolling_std)
        point_scores = np.maximum(point_scores, z)

    smoothed = gaussian_filter1d(point_scores, sigma=sigma, mode='reflect')
    return smoothed


def main():
    print("=" * 80)
    print("P24 — Per-Feature Importance Analysis (Q3 v7)")
    print("=" * 80)

    output_dir = Path(__file__).parent.parent / 'results' / 'P24_feature_importance'
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = iter_dataset_aliases()

    # ============= Stage 1: Per-dataset feature separation =============
    print("\n--- Stage 1: Per-feature anomaly separation across datasets ---")

    all_feature_sep = {}
    all_feature_corr = {}
    all_feature_counts = {}

    for alias, swat_excl22 in targets:
        ds = DatasetScores.load(alias, swat_excl22)
        if ds is None: continue
        raw_data = get_raw_signals(alias, ds)
        if raw_data is None: continue
        signals, point_labels, anomaly_regions, feature_names, train_ratio, _ = raw_data
        eval_mask = ds.eval_mask

        per_feat = per_feature_anomaly_separation(signals, point_labels, eval_mask)
        if not per_feat: continue

        # Top features
        seps = [f['separation'] for f in per_feat]
        all_feature_sep[alias] = seps
        all_feature_counts[alias] = signals.shape[1]

        # Correlation
        corrs = feature_correlation_matrix(signals, point_labels, eval_mask)
        all_feature_corr[alias] = corrs.tolist()

    print(f"  Analyzed {len(all_feature_sep)} datasets")
    print(f"  Feature counts: min={min(all_feature_counts.values())}, max={max(all_feature_counts.values())}")
    feature_dist = defaultdict(int)
    for n in all_feature_counts.values():
        feature_dist[n] += 1
    print(f"  Feature count distribution: {dict(feature_dist)}")

    # ============= Stage 2: Per-feature detection ceiling =============
    print("\n--- Stage 2: Per-feature subset detection ceiling ---")
    print("  (Each feature alone vs all features vs top-K)")

    subset_results = {}
    for alias, swat_excl22 in targets:
        ds = DatasetScores.load(alias, swat_excl22)
        if ds is None: continue
        raw_data = get_raw_signals(alias, ds)
        if raw_data is None: continue
        signals, point_labels, anomaly_regions, feature_names, train_ratio, _ = raw_data
        eval_mask = ds.eval_mask
        n_features = signals.shape[1]

        # All features
        all_score = simple_per_feature_score(signals, sigma=10)
        all_pak = pak_auc_f1(all_score, point_labels, anomaly_regions, eval_mask)

        # Per single feature
        per_feat_pak = []
        for j in range(n_features):
            single_score = per_feature_subset_score(signals, [j], sigma=10)
            single_pak = pak_auc_f1(single_score, point_labels, anomaly_regions, eval_mask)
            per_feat_pak.append(single_pak)

        # Top-K (K=3) by separation
        per_feat = per_feature_anomaly_separation(signals, point_labels, eval_mask)
        top_k = sorted(range(n_features), key=lambda j: -abs(per_feat[j]['separation']))[:3]
        topk_score = per_feature_subset_score(signals, top_k, sigma=10)
        topk_pak = pak_auc_f1(topk_score, point_labels, anomaly_regions, eval_mask)

        subset_results[alias] = {
            'n_features': n_features,
            'all_pak': float(all_pak),
            'best_single_pak': float(max(per_feat_pak)),
            'best_single_idx': int(np.argmax(per_feat_pak)),
            'top3_pak': float(topk_pak),
            'top3_indices': top_k,
            'per_feature_pak': per_feat_pak,
        }

    # ============= Stage 3: Comparison with baseline (274 model adaptive) =============
    print("\n--- Stage 3: vs baseline 274 model ---")

    baseline_results = {}
    for alias, swat_excl22 in targets:
        ds = DatasetScores.load(alias, swat_excl22)
        if ds is None: continue
        pt_r, pt_d, pt_s, pt_f = per_channel_points(ds, stride=21)
        base = adaptive_combine(pt_r, pt_d, pt_f, use_fm=True)
        base_smoothed = gauss(base, 10)
        base_pak = pak_auc_f1(base_smoothed, ds.point_labels, ds.regions, ds.eval_mask)
        baseline_results[alias] = float(base_pak)

    # ============= Stage 4: Cross-comparison: which method wins per dataset? =============
    print("\n--- Stage 4: Per-feature vs 274 model cross-comparison ---")
    cross_compare = {}
    n_top_k_wins = 0
    n_baseline_wins = 0
    n_close = 0
    for alias in set(subset_results.keys()) & set(baseline_results.keys()):
        feat_score = subset_results[alias]['top3_pak']
        base_score = baseline_results[alias]
        if feat_score > base_score + 0.05:
            cross_compare[alias] = 'feature_wins'
            n_top_k_wins += 1
        elif base_score > feat_score + 0.05:
            cross_compare[alias] = 'baseline_wins'
            n_baseline_wins += 1
        else:
            cross_compare[alias] = 'close'
            n_close += 1

    print(f"  Top-3 feature wins:  {n_top_k_wins}")
    print(f"  274 baseline wins:   {n_baseline_wins}")
    print(f"  Close (Δ<0.05):      {n_close}")

    # Per-group breakdown
    feat_better_per_group = defaultdict(int)
    for alias, who in cross_compare.items():
        g = get_per_group(alias)
        if who == 'feature_wins':
            feat_better_per_group[g] += 1
    print(f"\n  Where feature subset beats 274 baseline by >0.05:")
    for g, n in feat_better_per_group.items():
        print(f"    {g}: {n} datasets")

    # ============= Stage 5: Top features per dataset =============
    print("\n--- Stage 5: Top features per dataset ---")
    print(f"  {'Dataset':<25s} {'#feat':>6s} {'All':>7s} {'Top3':>7s} {'Best':>7s} {'274base':>8s}")
    for alias in sorted(subset_results.keys()):
        sr = subset_results[alias]
        base = baseline_results.get(alias, 0)
        print(f"  {alias:<25s} {sr['n_features']:>6d} {sr['all_pak']:>7.3f} "
              f"{sr['top3_pak']:>7.3f} {sr['best_single_pak']:>7.3f} {base:>8.3f}")

    # ============= Stage 6: Feature subset oracle ceiling =============
    print("\n--- Stage 6: Feature subset oracle ceiling (combinatorial) ---")
    print("  (For each dataset, find best 2-feature subset; aggregate gains)")

    oracle_results = {}
    deltas_vs_baseline = []
    for alias, swat_excl22 in targets:
        if alias not in baseline_results: continue
        if alias not in subset_results: continue

        ds = DatasetScores.load(alias, swat_excl22)
        if ds is None: continue
        raw_data = get_raw_signals(alias, ds)
        if raw_data is None: continue
        signals, point_labels, anomaly_regions, feature_names, train_ratio, _ = raw_data
        eval_mask = ds.eval_mask
        n_features = signals.shape[1]

        # Try 2-feature combinations (limit if too many)
        max_combo = 12
        if n_features > max_combo:
            # Top by single-feature pak
            per_feat_pak = subset_results[alias]['per_feature_pak']
            top_indices = sorted(range(n_features), key=lambda j: -per_feat_pak[j])[:max_combo]
        else:
            top_indices = list(range(n_features))

        best_pak = 0
        best_pair = None
        for i in range(len(top_indices)):
            for j in range(i+1, len(top_indices)):
                pair = [top_indices[i], top_indices[j]]
                pair_score = per_feature_subset_score(signals, pair, sigma=10)
                pair_pak = pak_auc_f1(pair_score, point_labels, anomaly_regions, eval_mask)
                if pair_pak > best_pak:
                    best_pak = pair_pak
                    best_pair = pair
        oracle_results[alias] = {
            'oracle_2feature_pak': float(best_pak),
            'best_pair_indices': best_pair,
            'delta_vs_baseline': float(best_pak - baseline_results[alias]),
        }
        deltas_vs_baseline.append(best_pak - baseline_results[alias])

    if deltas_vs_baseline:
        mean_delta = float(np.mean(deltas_vs_baseline))
        med_delta = float(np.median(deltas_vs_baseline))
        print(f"\n  Oracle 2-feature subset vs baseline:")
        print(f"    Mean Δ:    {mean_delta:+.4f}")
        print(f"    Median Δ:  {med_delta:+.4f}")
        print(f"    n datasets where Δ > 0.05: {sum(1 for d in deltas_vs_baseline if d > 0.05)}")
        print(f"    n datasets where Δ < -0.05: {sum(1 for d in deltas_vs_baseline if d < -0.05)}")

    # Save
    save_data = {
        'n_datasets': len(subset_results),
        'subset_results': subset_results,
        'baseline_results': baseline_results,
        'cross_compare': cross_compare,
        'oracle_2feature_results': oracle_results,
        'per_feature_separation': all_feature_sep,
        'feature_count_distribution': dict(feature_dist),
        'feature_correlation': all_feature_corr,
    }
    with open(output_dir / 'P24_feature_analysis.json', 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nSaved: {output_dir / 'P24_feature_analysis.json'}")


if __name__ == "__main__":
    main()
