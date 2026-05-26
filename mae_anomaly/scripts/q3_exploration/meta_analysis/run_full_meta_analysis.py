"""
Full Meta-Analysis — Stage 1+2+3+4 통합 분석.

본 script는 다음을 수행:

Stage 1: Load all results into single matrix (242 methods × 39 datasets)
Stage 2: Method-method correlation + hierarchical clustering
Stage 3: Dataset-dataset clustering by method response
Stage 4: Failure mode analysis + theoretical bounds + method redundancy

Output:
- meta_analysis/output/method_correlation_matrix.npz
- meta_analysis/output/method_cluster_assignments.json
- meta_analysis/output/dataset_cluster_assignments.json
- meta_analysis/output/method_redundancy_pairs.json
- meta_analysis/output/diversity_subset.json
- meta_analysis/output/hard_datasets.json
- meta_analysis/output/universal_winners.json
"""
import sys
from pathlib import Path
import numpy as np
import json
from collections import defaultdict, Counter

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from mae_anomaly.scripts.q3_exploration.core.meta_aggregation import load_all_results
from mae_anomaly.scripts.q3_exploration.core.meta_clustering import (
    compute_method_correlation_matrix, cluster_methods, cluster_datasets,
    method_redundancy_analysis, method_diversity_subset,
    failure_mode_analysis, universal_winners_analysis,
    per_dataset_best_method, per_method_best_dataset,
)


def get_per_group(alias):
    if alias.startswith('smd_'): return 'SMD'
    if alias.startswith('exathlon_'): return 'Exathlon'
    return 'Standalone'


def main():
    print("=" * 80)
    print("META-ANALYSIS — Stage 1: Result Aggregation")
    print("=" * 80)

    matrix = load_all_results()
    aliases, method_names, delta_matrix = matrix.to_matrix()

    print(f"\nAggregated matrix: {delta_matrix.shape[0]} datasets × {delta_matrix.shape[1]} methods")
    print(f"NaN ratio: {np.isnan(delta_matrix).mean():.2%}")
    print(f"Mean Δ across all cells: {np.nanmean(delta_matrix):+.4f}")
    print(f"Median Δ: {np.nanmedian(delta_matrix):+.4f}")
    print(f"Δ range: [{np.nanmin(delta_matrix):.4f}, {np.nanmax(delta_matrix):.4f}]")

    # Source distribution
    print(f"\nMethod sources:")
    src_counts = Counter()
    for m in matrix.methods.values():
        src_counts[m.source] += 1
    for src, n in sorted(src_counts.items()):
        print(f"  {src}: {n} methods")

    print(f"\nFamily distribution:")
    family_counts = Counter()
    for m in matrix.methods.values():
        family_counts[m.family] += 1
    for fam, n in sorted(family_counts.most_common()):
        if n >= 3:
            print(f"  {fam}: {n}")

    # ==================== Stage 2: Method-Method Clustering ====================
    print("\n" + "=" * 80)
    print("STAGE 2: Method-Method Correlation + Clustering")
    print("=" * 80)

    print("\nComputing pairwise Pearson correlation (242×242)...")
    corr_pearson = compute_method_correlation_matrix(delta_matrix, method='pearson')
    print(f"  Mean off-diagonal correlation: {np.mean(corr_pearson[np.triu_indices_from(corr_pearson, k=1)]):.3f}")
    print(f"  Median: {np.median(corr_pearson[np.triu_indices_from(corr_pearson, k=1)]):.3f}")

    # Hierarchical clustering
    for n_clusters in [5, 10, 15, 20]:
        labels, _ = cluster_methods(corr_pearson, n_clusters=n_clusters)
        cluster_to_methods = defaultdict(list)
        for i, c in enumerate(labels):
            cluster_to_methods[c].append(method_names[i])
        cluster_sizes = sorted([len(v) for v in cluster_to_methods.values()], reverse=True)
        print(f"\nK={n_clusters} clusters: sizes {cluster_sizes[:8]}{'...' if len(cluster_sizes) > 8 else ''}")
        # Show cluster of each top method
        mean_deltas = np.nanmean(delta_matrix, axis=0)
        top_5 = np.argsort(-mean_deltas)[:5]
        for idx in top_5:
            mname = method_names[idx]
            c = labels[idx]
            # Other top members of same cluster
            same_cluster = [(method_names[i], np.nanmean(delta_matrix[:, i]))
                            for i in np.where(labels == c)[0]]
            same_cluster.sort(key=lambda x: -x[1])
            cluster_top3 = [m[0] for m in same_cluster[:3]]
            print(f"  Method {mname} (mean Δ={mean_deltas[idx]:+.4f}) → cluster {c} (size {len(same_cluster)})")
            print(f"    Cluster top-3: {cluster_top3}")

    # Use K=10 for further analysis
    labels_10, _ = cluster_methods(corr_pearson, n_clusters=10)

    # Per-cluster representative
    cluster_to_methods = defaultdict(list)
    for i, c in enumerate(labels_10):
        cluster_to_methods[c].append((method_names[i], np.nanmean(delta_matrix[:, i])))

    print("\n--- K=10 Cluster Analysis ---")
    cluster_representatives = {}
    for c, members in sorted(cluster_to_methods.items()):
        members.sort(key=lambda x: -x[1])
        rep_method = members[0]  # highest mean Δ in cluster
        cluster_representatives[int(c)] = {
            'representative': rep_method[0],
            'representative_delta': float(rep_method[1]),
            'size': len(members),
            'top5_members': [{'name': m[0], 'delta': float(m[1])} for m in members[:5]],
        }
        print(f"\nCluster {c} (size={len(members)}, rep mean Δ={rep_method[1]:+.4f}):")
        for m in members[:5]:
            print(f"  {m[0]:<40s} mean Δ={m[1]:+.4f}")

    # ==================== Method Redundancy Analysis ====================
    print("\n" + "=" * 80)
    print("Method Redundancy Analysis")
    print("=" * 80)

    redundant_pairs = method_redundancy_analysis(delta_matrix, threshold=0.95)
    print(f"\nN highly redundant pairs (ρ > 0.95): {len(redundant_pairs)}")
    print(f"\nTop-15 redundant pairs:")
    for i, j, c in redundant_pairs[:15]:
        print(f"  {method_names[i][:30]:<30s} ~ {method_names[j][:30]:<30s} ρ={c:.4f}")

    redundant_90 = method_redundancy_analysis(delta_matrix, threshold=0.90)
    print(f"\nN pairs with ρ > 0.90: {len(redundant_90)}")
    redundant_80 = method_redundancy_analysis(delta_matrix, threshold=0.80)
    print(f"N pairs with ρ > 0.80: {len(redundant_80)}")

    # ==================== Diversity Subset ====================
    print("\n" + "=" * 80)
    print("Diversity-Maximizing Method Subset")
    print("=" * 80)

    for k_div in [5, 10, 15]:
        div_indices = method_diversity_subset(delta_matrix, k=k_div)
        print(f"\nTop-{k_div} diversity subset (greedy maximin):")
        for idx in div_indices:
            mname = method_names[idx]
            mean_d = np.nanmean(delta_matrix[:, idx])
            wins = ((delta_matrix[:, idx] > 0).sum())
            print(f"  {mname:<40s} mean Δ={mean_d:+.4f}  W={wins}/{delta_matrix.shape[0]}")

    # ==================== Stage 3: Dataset-Dataset Clustering ====================
    print("\n" + "=" * 80)
    print("STAGE 3: Dataset-Dataset Clustering by Method Response")
    print("=" * 80)

    for n_d_clusters in [3, 4, 5, 6, 8]:
        labels_d, _ = cluster_datasets(delta_matrix, n_clusters=n_d_clusters)
        cluster_to_aliases = defaultdict(list)
        for i, c in enumerate(labels_d):
            cluster_to_aliases[c].append(aliases[i])

        print(f"\nK={n_d_clusters} dataset clusters:")
        for c, ds_list in sorted(cluster_to_aliases.items()):
            # Cluster의 group composition
            groups = Counter([get_per_group(a) for a in ds_list])
            # Cluster의 mean Δ across methods
            cluster_delta = delta_matrix[[i for i, a in enumerate(aliases) if a in ds_list]]
            cluster_mean_delta = np.nanmean(cluster_delta)
            print(f"  Cluster {c} (n={len(ds_list)}, mean Δ across methods={cluster_mean_delta:+.4f}):")
            print(f"    groups: {dict(groups)}")
            if len(ds_list) <= 10:
                print(f"    datasets: {ds_list}")
            else:
                print(f"    sample: {ds_list[:6]}... (+{len(ds_list)-6})")

    # ==================== Stage 4: Failure Mode + Universal Winners ====================
    print("\n" + "=" * 80)
    print("STAGE 4: Failure Mode + Universal Winners")
    print("=" * 80)

    # Hard datasets
    hard_datasets = failure_mode_analysis(delta_matrix, aliases, method_names,
                                           n_methods_for_consensus=20)
    print(f"\n--- Top-10 Hardest Datasets (lowest top-20 method mean Δ) ---")
    print(f"{'Dataset':<25s} {'Hardness':>10s} {'top-3 winning methods':<60s}")
    for alias, hardness, method_deltas in hard_datasets[:10]:
        top3 = sorted(method_deltas.items(), key=lambda x: -x[1])[:3]
        top3_str = ', '.join([f"{m[:18]}={d:+.3f}" for m, d in top3])
        print(f"{alias:<25s} {hardness:>+10.4f} {top3_str:<60s}")

    print(f"\n--- Top-10 Easiest Datasets (highest top-20 method mean Δ) ---")
    for alias, hardness, method_deltas in hard_datasets[-10:]:
        top3 = sorted(method_deltas.items(), key=lambda x: -x[1])[:3]
        top3_str = ', '.join([f"{m[:18]}={d:+.3f}" for m, d in top3])
        print(f"{alias:<25s} {hardness:>+10.4f} {top3_str:<60s}")

    # Universal winners
    print("\n--- Universal Winners (high win rate across datasets) ---")
    winner_stats = universal_winners_analysis(delta_matrix, method_names, aliases)
    sorted_winners = sorted(winner_stats, key=lambda x: -x['win_rate'])
    print(f"{'Method':<40s} {'win_rate':>10s} {'meanΔ':>10s} {'worst':>10s} {'best':>10s}")
    for w in sorted_winners[:15]:
        print(f"{w['name']:<40s} {w['win_rate']:>10.2%} {w['mean_delta']:>+10.4f} "
              f"{w['worst_delta']:>+10.4f} {w['best_delta']:>+10.4f}")

    # Per-dataset best
    print("\n--- Per-Dataset Best Method ---")
    per_ds = per_dataset_best_method(delta_matrix, method_names, aliases)
    best_method_distribution = Counter([r['best_method'] for r in per_ds])
    print(f"Most popular per-dataset winners:")
    for method, n in best_method_distribution.most_common(15):
        print(f"  {method:<50s} : {n} datasets")

    # ==================== Save outputs ====================
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)

    # Method delta matrix CSV
    try:
        import pandas as pd
        df = matrix.to_pandas()
        df.to_csv(output_dir / 'delta_matrix.csv')
        print(f"\nSaved: {output_dir / 'delta_matrix.csv'}")
    except Exception as e:
        print(f"CSV save error: {e}")

    # Method correlation
    np.savez_compressed(output_dir / 'method_correlation.npz',
                         correlation=corr_pearson,
                         method_names=np.array(method_names),
                         aliases=np.array(aliases),
                         delta_matrix=delta_matrix)

    # Cluster assignments
    cluster_data = {
        'method_clusters_K10': {method_names[i]: int(labels_10[i]) for i in range(len(method_names))},
        'cluster_representatives_K10': cluster_representatives,
    }
    with open(output_dir / 'method_clusters.json', 'w') as f:
        json.dump(cluster_data, f, indent=2)

    dataset_clusters = {}
    for K in [3, 4, 5, 6, 8]:
        labels_d, _ = cluster_datasets(delta_matrix, n_clusters=K)
        dataset_clusters[f'K{K}'] = {aliases[i]: int(labels_d[i]) for i in range(len(aliases))}
    with open(output_dir / 'dataset_clusters.json', 'w') as f:
        json.dump(dataset_clusters, f, indent=2)

    # Hard datasets
    hard_data = [{'alias': h[0], 'hardness': float(h[1])} for h in hard_datasets]
    with open(output_dir / 'hard_datasets.json', 'w') as f:
        json.dump(hard_data, f, indent=2)

    # Universal winners
    with open(output_dir / 'universal_winners.json', 'w') as f:
        json.dump(sorted_winners, f, indent=2)

    # Diversity subset
    with open(output_dir / 'diversity_subsets.json', 'w') as f:
        json.dump({
            f'top_{k}': [{'idx': int(i), 'name': method_names[i],
                          'mean_delta': float(np.nanmean(delta_matrix[:, i]))}
                          for i in method_diversity_subset(delta_matrix, k=k)]
            for k in [5, 10, 15]
        }, f, indent=2)

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
