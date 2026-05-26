"""
P22 — Per-Dataset Adaptive Method Selector

Q3 v5 meta-analysis 발견: per-dataset best method가 30+ different methods에 분포.
본 P22는 dataset signature → best method classifier 학습:

1. Q3 v5의 method_clusters K=10 사용 → 10 cluster representatives만 candidate
2. Per-dataset signature (anomaly characteristics + score statistics) 추출
3. LOO classification: train on 38 datasets → predict best cluster for 1 test
4. Predicted cluster의 representative method 적용
5. Vs. uniform best method (P12) 비교

목표: P8 K=8 tri-routing (+0.0276) 능가 또는 동등 (cluster routing의 fine-grained version)
"""
import sys
from pathlib import Path
import numpy as np
import json
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut

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
from mae_anomaly.scripts.q3_exploration.core.meta_aggregation import load_all_results
from mae_anomaly.scripts.q3_exploration.core.meta_clustering import (
    compute_method_correlation_matrix, cluster_methods,
)
from mae_anomaly.scripts.q3_exploration.core.clustering import (
    extract_signature_unsupervised, extract_signature_supervised,
)


def main():
    print("P22 — Per-Dataset Adaptive Method Selector")

    # Load all meta results
    matrix = load_all_results()
    aliases, method_names, delta_matrix = matrix.to_matrix()
    print(f"Loaded {len(aliases)} datasets × {len(method_names)} methods")

    # Get method clusters K=10
    corr = compute_method_correlation_matrix(delta_matrix)
    cluster_labels, _ = cluster_methods(corr, n_clusters=10)

    # Cluster representatives (highest mean Δ per cluster)
    mean_deltas_method = np.nanmean(delta_matrix, axis=0)
    cluster_to_members = defaultdict(list)
    for i, c in enumerate(cluster_labels):
        cluster_to_members[c].append(i)
    cluster_representatives = {}
    for c, members in cluster_to_members.items():
        best_in_cluster = max(members, key=lambda i: mean_deltas_method[i])
        cluster_representatives[c] = {
            'index': best_in_cluster,
            'name': method_names[best_in_cluster],
            'mean_delta': float(mean_deltas_method[best_in_cluster]),
        }

    print(f"\n10 cluster representatives:")
    for c, info in sorted(cluster_representatives.items(),
                           key=lambda x: -x[1]['mean_delta']):
        print(f"  Cluster {c}: {info['name']:<40s}  mean Δ = {info['mean_delta']:+.4f}")

    n_clusters = len(cluster_representatives)
    cluster_ids_sorted = sorted(cluster_representatives.keys())

    # Per-dataset: which cluster's representative gives best Δ?
    print(f"\nFinding per-dataset best cluster representative...")
    rep_indices = [cluster_representatives[c]['index'] for c in cluster_ids_sorted]
    rep_delta_matrix = delta_matrix[:, rep_indices]  # (n_datasets, n_clusters)

    best_cluster_per_dataset = []
    best_delta_per_dataset = []
    for i, alias in enumerate(aliases):
        best_local_idx = int(np.argmax(rep_delta_matrix[i]))
        best_cluster_per_dataset.append(cluster_ids_sorted[best_local_idx])
        best_delta_per_dataset.append(float(rep_delta_matrix[i, best_local_idx]))

    # Per-dataset best of clusters
    print(f"\nBest cluster distribution across datasets:")
    cluster_counts = defaultdict(int)
    for c in best_cluster_per_dataset:
        cluster_counts[c] += 1
    for c, n in sorted(cluster_counts.items(), key=lambda x: -x[1]):
        info = cluster_representatives[c]
        print(f"  Cluster {c} ({info['name'][:30]}): {n} datasets")

    # Extract dataset signatures
    print(f"\nExtracting dataset signatures...")
    aliases_with_sig = []
    X_signatures = []
    y_best_cluster = []
    baseline_paks = []

    for i, alias in enumerate(aliases):
        is_swat_excl = (alias == 'swat_excl22')
        ds = DatasetScores.load(alias, is_swat_excl)
        if ds is None: continue

        pt_r, pt_d, pt_s, pt_f = per_channel_points(ds, stride=21)
        base = adaptive_combine(pt_r, pt_d, pt_f, use_fm=True)
        base_smoothed = gauss(base, 10)
        baseline_pak = pak_auc_f1(base_smoothed, ds.point_labels, ds.regions, ds.eval_mask)

        unsup = extract_signature_unsupervised(base_smoothed, pt_r, pt_d, pt_f)
        sup = extract_signature_supervised(ds.regions, ds.point_labels, baseline_pak)
        feature_vector = list(unsup.values()) + list(sup.values())

        aliases_with_sig.append(alias)
        X_signatures.append(feature_vector)
        y_best_cluster.append(best_cluster_per_dataset[i])
        baseline_paks.append(baseline_pak)

    X = np.array(X_signatures)
    y = np.array(y_best_cluster)
    baseline_paks = np.array(baseline_paks)

    print(f"\nX shape: {X.shape}")
    print(f"y unique: {np.unique(y)}")

    # LOO classification
    print(f"\n=== LOO Classifier evaluation ===")

    classifiers_to_test = {
        'rf_50_d3': RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42),
        'rf_100_d5': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        'rf_50_d_None': RandomForestClassifier(n_estimators=50, random_state=42),
    }

    for cls_name, cls_proto in classifiers_to_test.items():
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        loo = LeaveOneOut()
        predicted_clusters = []
        for train_idx, test_idx in loo.split(X_scaled):
            from sklearn.base import clone
            cls = clone(cls_proto)
            cls.fit(X_scaled[train_idx], y[train_idx])
            pred = cls.predict(X_scaled[test_idx])
            predicted_clusters.append(int(pred[0]))

        # Apply predicted cluster's representative
        pred_paks = []
        for i, alias in enumerate(aliases_with_sig):
            pred_cluster = predicted_clusters[i]
            rep_idx = cluster_representatives[pred_cluster]['index']
            alias_idx = aliases.index(alias)
            pred_paks.append(baseline_paks[i] + delta_matrix[alias_idx, rep_idx])

        pred_paks = np.array(pred_paks)
        deltas = pred_paks - baseline_paks
        mean_d = float(deltas.mean())
        wins = int((deltas > 0).sum())
        losses = int((deltas < 0).sum())
        cata = int((deltas < -0.05).sum())
        p = wilcoxon_test(pred_paks.tolist(), baseline_paks.tolist(), alternative='greater')

        # Accuracy of classifier
        n_correct = sum(p_c == y_c for p_c, y_c in zip(predicted_clusters, y))
        acc = n_correct / len(y)

        print(f"\n  {cls_name}:")
        print(f"    Classifier accuracy (LOO): {acc:.2%}")
        print(f"    Mean Δ: {mean_d:+.4f}")
        print(f"    W/L: {wins}/{losses}  cata={cata}")
        print(f"    p: {p:.4f}")

    # Upper bound: oracle cluster selection (always pick best)
    oracle_paks = baseline_paks.copy()
    for i, alias in enumerate(aliases_with_sig):
        alias_idx = aliases.index(alias)
        oracle_paks[i] = baseline_paks[i] + best_delta_per_dataset[alias_idx]
    oracle_deltas = oracle_paks - baseline_paks
    print(f"\n--- Oracle cluster selection (upper bound) ---")
    print(f"  Mean Δ: {oracle_deltas.mean():+.4f}")
    print(f"  W/L: {(oracle_deltas > 0).sum()}/{(oracle_deltas < 0).sum()}")

    # Comparison: P12 standalone (best single method)
    p12_idx = None
    for j, mname in enumerate(method_names):
        if mname == 'P12_blend_type_pak':
            p12_idx = j
            break
    if p12_idx is not None:
        p12_deltas = delta_matrix[[aliases.index(a) for a in aliases_with_sig], p12_idx]
        print(f"\n--- P12 standalone (best single method) ---")
        print(f"  Mean Δ: {p12_deltas.mean():+.4f}")
        print(f"  W/L: {(p12_deltas > 0).sum()}/{(p12_deltas < 0).sum()}")

    # Per-group breakdown for best classifier
    print(f"\n=== Per-group breakdown (rf_50_d3) ===")
    deltas_dict = {a: float(d) for a, d in zip(aliases_with_sig, deltas)}
    summary = per_group_summary(deltas_dict, get_per_group)
    for g, s in summary.items():
        print(f"  {g:<12s} n={s['n']:2d}  meanΔ={s['mean_delta']:+.4f}  "
              f"W/L={s['wins']}/{s['losses']}  cata={s['catastrophic']}")

    # Save
    output_path = Path(__file__).parent.parent / 'results' / 'P22_method_selector.json'
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({
            'cluster_representatives': cluster_representatives,
            'best_cluster_per_dataset': {a: int(c) for a, c in zip(aliases, best_cluster_per_dataset)},
            'oracle_mean_delta': float(oracle_deltas.mean()),
        }, f, indent=2)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
