"""
P20 + P21 — Anomaly Sub-type Discovery + Information-Theoretic Bound

P20: Hard datasets의 individual anomaly regions를 unsupervised clustering
     - Per-region feature: length, isolation, internal variability, shape
     - K-means clustering of anomalies
     - Per-cluster method preference 검증

P21: Information-theoretic bound estimation
     - Mutual information between score and labels (binning-based)
     - Bayes optimal error rate (k-NN density estimation)
     - Theoretical ceiling vs achievable
"""
import sys
from pathlib import Path
import numpy as np
import json
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from mae_anomaly.scripts.q3_exploration.core.data import (
    DatasetScores, iter_dataset_aliases, median_anomaly_segment_length, get_per_group
)
from mae_anomaly.scripts.q3_exploration.core.scoring import (
    per_channel_points, adaptive_combine, gauss, zscore
)
from mae_anomaly.scripts.q3_exploration.core.evaluation import pak_auc_f1
from mae_anomaly.scripts.q3_exploration.core.data_analysis import (
    anomaly_isolation_profile, score_label_alignment_metrics,
)
from mae_anomaly.scripts.q3_exploration.core.postprocess import nlm_sigmoid_transform


# ============= P20: Anomaly Sub-type Discovery =============

def extract_per_anomaly_features(ds, score_smoothed):
    """각 anomaly region의 7-feature vector."""
    profiles = anomaly_isolation_profile(score_smoothed, ds.regions, ds.eval_mask)
    features = []
    for p in profiles:
        # 7 features:
        # 1. log length, 2. isolation, 3. contrast, 4. internal_variability
        # 5. in_max / ctx_std (normalized peak), 6. log(in_max - in_mean), 7. position (relative)
        feat = [
            float(np.log10(p['length'] + 1)),
            p['isolation'],
            p['contrast'],
            p['internal_variability'],
            p['in_max'] / (p['ctx_std'] + 1e-9),
            float(np.log10(max(p['in_max'] - p['in_mean'], 1e-9))),
            p['start'] / max(len(score_smoothed), 1),  # relative position
        ]
        features.append(feat)
    return np.array(features), profiles


def run_P20():
    print("=" * 80)
    print("P20 — Anomaly Sub-type Discovery")
    print("=" * 80)

    targets = iter_dataset_aliases()
    all_anomaly_features = []
    all_anomaly_meta = []  # (alias, profile_dict)

    for alias, swat in targets:
        ds = DatasetScores.load(alias, swat)
        if ds is None: continue
        pt_r, pt_d, pt_s, pt_f = per_channel_points(ds, stride=21)
        base = adaptive_combine(pt_r, pt_d, pt_f, use_fm=True)
        base_smoothed = gauss(base, 10)
        features, profiles = extract_per_anomaly_features(ds, base_smoothed)
        if len(features) == 0: continue
        for i, p in enumerate(profiles):
            all_anomaly_features.append(features[i])
            all_anomaly_meta.append({'alias': alias, 'profile': p})

    X = np.array(all_anomaly_features)
    print(f"\nTotal anomaly regions: {len(X)}")
    print(f"Feature matrix: {X.shape}")

    # Clean & scale
    X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K=5 anomaly subtypes
    print(f"\n--- Anomaly Sub-type Clustering ---")
    for K in [3, 4, 5, 6]:
        km = KMeans(n_clusters=K, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)

        print(f"\nK={K}: cluster sizes {dict(zip(*np.unique(labels, return_counts=True)))}")
        for c in range(K):
            members_mask = labels == c
            members = [all_anomaly_meta[i] for i in np.where(members_mask)[0]]
            # Feature means
            feat_mean = X[members_mask].mean(axis=0)
            print(f"  Cluster {c} (n={members_mask.sum()}):")
            print(f"    log_length={feat_mean[0]:.2f}, isolation={feat_mean[1]:.2f}, "
                  f"contrast={feat_mean[2]:.2f}, int_var={feat_mean[3]:.2f}")
            print(f"    in_max/ctx_std={feat_mean[4]:.2f}, log(in_max-in_mean)={feat_mean[5]:.2f}")
            # Top datasets in this cluster
            dataset_counts = defaultdict(int)
            for m in members:
                dataset_counts[m['alias']] += 1
            top_ds = sorted(dataset_counts.items(), key=lambda x: -x[1])[:3]
            print(f"    Top datasets: {[(a, n) for a, n in top_ds]}")

    # Output
    output_path = Path(__file__).parent.parent / 'results' / 'P20_anomaly_subtype.json'
    output_path.parent.mkdir(exist_ok=True)
    final_labels = KMeans(n_clusters=5, random_state=42, n_init=10).fit_predict(X_scaled)
    # Per-dataset: distribution of subtype labels
    dataset_subtype_profile = defaultdict(lambda: defaultdict(int))
    for i, meta in enumerate(all_anomaly_meta):
        dataset_subtype_profile[meta['alias']][int(final_labels[i])] += 1

    save_data = {
        'n_anomaly_regions': len(X),
        'feature_names': ['log_length', 'isolation', 'contrast', 'internal_variability',
                          'in_max_over_ctx_std', 'log_in_max_minus_mean', 'relative_position'],
        'dataset_subtype_profile_K5': {a: dict(v) for a, v in dataset_subtype_profile.items()},
    }
    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved P20: {output_path}")


# ============= P21: Information-Theoretic Bound =============

def mutual_information_binned(scores, labels, n_bins=20):
    """MI(score, label) via binning."""
    # Bin scores
    bin_edges = np.percentile(scores, np.linspace(0, 100, n_bins + 1))
    bin_edges = np.unique(bin_edges)  # may collapse if degenerate
    if len(bin_edges) < 3:
        return 0.0
    bin_indices = np.digitize(scores, bin_edges[:-1])

    # Joint distribution
    n = len(scores)
    p_score_bins = np.zeros(len(bin_edges) - 1)
    p_label_given_score = np.zeros((len(bin_edges) - 1, 2))
    for b in range(1, len(bin_edges)):
        mask = bin_indices == b
        if mask.sum() == 0: continue
        p_score_bins[b - 1] = mask.sum() / n
        p_label_given_score[b - 1, 0] = (labels[mask] == 0).sum() / max(mask.sum(), 1)
        p_label_given_score[b - 1, 1] = (labels[mask] == 1).sum() / max(mask.sum(), 1)

    # Marginal P(label)
    p_label_normal = (labels == 0).mean()
    p_label_anom = (labels == 1).mean()
    if p_label_anom < 1e-9 or p_label_normal < 1e-9:
        return 0.0

    # H(label)
    H_label = -p_label_normal * np.log2(max(p_label_normal, 1e-15)) \
              - p_label_anom * np.log2(max(p_label_anom, 1e-15))

    # H(label | score)
    H_label_given_score = 0.0
    for b in range(len(bin_edges) - 1):
        p_n = p_label_given_score[b, 0]
        p_a = p_label_given_score[b, 1]
        h = 0.0
        if p_n > 1e-15: h -= p_n * np.log2(p_n)
        if p_a > 1e-15: h -= p_a * np.log2(p_a)
        H_label_given_score += p_score_bins[b] * h

    return max(0.0, H_label - H_label_given_score)


def bayes_optimal_error_rate(scores, labels, n_bins=50):
    """Bayes-optimal error rate estimate.

    Compute P(label=anom | score bin) for each bin.
    Optimal decision: predict argmax over labels.
    Error rate = P(misclassify).
    """
    bin_edges = np.percentile(scores, np.linspace(0, 100, n_bins + 1))
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 3:
        return 0.5
    bin_indices = np.digitize(scores, bin_edges[:-1])

    n = len(scores)
    total_error = 0.0
    for b in range(1, len(bin_edges)):
        mask = bin_indices == b
        if mask.sum() == 0: continue
        p_anom = (labels[mask] == 1).sum() / max(mask.sum(), 1)
        # Optimal: if p_anom > 0.5, predict anom (error = 1 - p_anom)
        # else predict normal (error = p_anom)
        bin_error = min(p_anom, 1 - p_anom)
        total_error += bin_error * mask.sum() / n
    return total_error


def run_P21():
    print("\n" + "=" * 80)
    print("P21 — Information-Theoretic Bound")
    print("=" * 80)

    targets = iter_dataset_aliases()
    all_metrics = {}

    for alias, swat in targets:
        ds = DatasetScores.load(alias, swat)
        if ds is None: continue

        pt_r, pt_d, pt_s, pt_f = per_channel_points(ds, stride=21)
        base = adaptive_combine(pt_r, pt_d, pt_f, use_fm=True)
        base_smoothed = gauss(base, 10)

        labels = ds.point_labels
        if ds.eval_mask is not None:
            labels_eval = labels[ds.eval_mask]
            score_eval = base_smoothed[ds.eval_mask]
        else:
            labels_eval = labels
            score_eval = base_smoothed

        if labels_eval.sum() == 0 or labels_eval.sum() == len(labels_eval):
            continue

        # H(label)
        p_anom = labels_eval.mean()
        H_label = -p_anom * np.log2(max(p_anom, 1e-15)) \
                  - (1 - p_anom) * np.log2(max(1 - p_anom, 1e-15))

        mi = mutual_information_binned(score_eval, labels_eval, n_bins=30)
        bayes_err = bayes_optimal_error_rate(score_eval, labels_eval, n_bins=50)

        # Anomaly base rate
        base_rate_error = min(p_anom, 1 - p_anom)  # always predict majority

        # MI / H(label) = information completeness ratio
        info_ratio = mi / max(H_label, 1e-9)

        # Also for div5+NLM-T1.5 score (best method-applied)
        median_seg = median_anomaly_segment_length(ds.regions)
        better_score = nlm_sigmoid_transform(gauss(base, max(median_seg/5, 0.5)), T_factor=1.5)
        if ds.eval_mask is not None:
            better_score_eval = better_score[ds.eval_mask]
        else:
            better_score_eval = better_score

        mi_better = mutual_information_binned(better_score_eval, labels_eval, n_bins=30)
        bayes_err_better = bayes_optimal_error_rate(better_score_eval, labels_eval, n_bins=50)
        info_ratio_better = mi_better / max(H_label, 1e-9)

        all_metrics[alias] = {
            'H_label_bits': float(H_label),
            'anomaly_rate': float(p_anom),
            'baseline_MI_bits': float(mi),
            'baseline_info_ratio': float(info_ratio),
            'baseline_bayes_err': float(bayes_err),
            'better_MI_bits': float(mi_better),
            'better_info_ratio': float(info_ratio_better),
            'better_bayes_err': float(bayes_err_better),
            'base_rate_error': float(base_rate_error),
        }

    # Print summary
    print(f"\n{'Dataset':<25s} {'H(L)':>6s} {'P(anom)':>8s} {'baseMI':>7s} {'b_ratio':>8s} {'baseBE':>7s} {'betterMI':>9s} {'b_ratio_bet':>11s}")
    aliases_sorted = sorted(all_metrics.keys(),
                              key=lambda a: -all_metrics[a]['better_info_ratio'])
    for alias in aliases_sorted[:25]:
        m = all_metrics[alias]
        print(f"{alias:<25s} {m['H_label_bits']:>6.3f} {m['anomaly_rate']:>8.3f} "
              f"{m['baseline_MI_bits']:>7.3f} {m['baseline_info_ratio']:>8.3f} "
              f"{m['baseline_bayes_err']:>7.3f} "
              f"{m['better_MI_bits']:>9.3f} {m['better_info_ratio']:>11.3f}")

    # Aggregate
    print(f"\n--- Aggregate (39 datasets) ---")
    keys = ['baseline_MI_bits', 'baseline_info_ratio', 'baseline_bayes_err',
            'better_MI_bits', 'better_info_ratio', 'better_bayes_err']
    for k in keys:
        vals = [m[k] for m in all_metrics.values()]
        print(f"  {k:<25s}: mean={np.mean(vals):.4f}  median={np.median(vals):.4f}")

    # Compare baseline → better gain
    delta_mi = [m['better_MI_bits'] - m['baseline_MI_bits'] for m in all_metrics.values()]
    delta_be = [m['baseline_bayes_err'] - m['better_bayes_err'] for m in all_metrics.values()]
    print(f"\n  Δ MI (better - baseline):     mean={np.mean(delta_mi):+.4f}")
    print(f"  Δ BE_reduction (baseline - better): mean={np.mean(delta_be):+.4f}")

    output_path = Path(__file__).parent.parent / 'results' / 'P21_info_theory.json'
    with open(output_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved P21: {output_path}")


if __name__ == "__main__":
    run_P20()
    run_P21()
