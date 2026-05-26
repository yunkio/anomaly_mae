"""
Meta-analysis visualization.

Generates:
- method_correlation_heatmap.png (242×242 correlation)
- method_dendrogram.png (hierarchical clustering tree)
- dataset_clustering.png
- universal_winners_scatter.png (win_rate vs mean Δ)
- hardness_distribution.png
"""
import sys
from pathlib import Path
import numpy as np
import json
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from mae_anomaly.scripts.q3_exploration.core.meta_aggregation import load_all_results
from mae_anomaly.scripts.q3_exploration.core.meta_clustering import (
    compute_method_correlation_matrix, cluster_methods, cluster_datasets,
)


def get_group(alias):
    if alias.startswith('smd_'): return 'SMD'
    if alias.startswith('exathlon_'): return 'Exathlon'
    return 'Standalone'


def main():
    matrix = load_all_results()
    aliases, method_names, delta_matrix = matrix.to_matrix()
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)

    print(f"Matrix: {delta_matrix.shape}")

    # ============ Method correlation heatmap (with hierarchical ordering) ============
    print("Plotting method correlation heatmap...")
    corr = compute_method_correlation_matrix(delta_matrix)
    n = corr.shape[0]
    dist = np.clip(1.0 - corr, 0, 2)
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0)
    Z = linkage(squareform(dist, checks=False), method='average')

    # Get leaf order
    from scipy.cluster.hierarchy import leaves_list
    leaf_order = leaves_list(Z)

    # Reorder correlation matrix
    corr_ordered = corr[leaf_order][:, leaf_order]

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(corr_ordered, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_title(f'Method-Method Correlation Matrix\n{n} methods, hierarchically ordered', fontsize=14)
    ax.set_xlabel('Method index (reordered)')
    ax.set_ylabel('Method index (reordered)')
    plt.colorbar(im, ax=ax, label='Pearson correlation')
    plt.tight_layout()
    plt.savefig(output_dir / 'method_correlation_heatmap.png', dpi=120, bbox_inches='tight')
    plt.close()

    # ============ Method dendrogram (K=10 cuts) ============
    print("Plotting method dendrogram...")
    labels_10, _ = cluster_methods(corr, n_clusters=10)
    fig, ax = plt.subplots(figsize=(20, 8))
    dendrogram(Z, labels=[f'{i}' for i in range(n)], ax=ax,
                color_threshold=Z[-9, 2], no_labels=True)
    ax.set_title(f'Method Hierarchical Clustering Dendrogram (K=10 cut)\n242 methods', fontsize=14)
    ax.set_xlabel('Method index')
    ax.set_ylabel('Distance (1 - correlation)')
    plt.tight_layout()
    plt.savefig(output_dir / 'method_dendrogram.png', dpi=120, bbox_inches='tight')
    plt.close()

    # ============ Method effect distribution ============
    print("Plotting method effect distribution...")
    mean_deltas = np.nanmean(delta_matrix, axis=0)
    win_rates = np.array([(delta_matrix[:, j] > 0).sum() / delta_matrix.shape[0]
                           for j in range(n)])
    catastrophic = np.array([(delta_matrix[:, j] < -0.05).sum() for j in range(n)])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # Mean Δ histogram
    axes[0].hist(mean_deltas, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(0, color='red', linestyle='--', label='no effect')
    axes[0].axvline(0.0276, color='green', linestyle='--', label='P8 best (+0.0276)')
    axes[0].set_xlabel('Mean Δ_pak vs baseline')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'Mean Δ distribution across 242 methods')
    axes[0].legend()

    # Win rate vs mean Δ scatter
    colors = ['green' if d > 0 else 'red' if d < -0.01 else 'gray' for d in mean_deltas]
    axes[1].scatter(mean_deltas, win_rates, c=colors, alpha=0.5, s=20)
    axes[1].set_xlabel('Mean Δ_pak')
    axes[1].set_ylabel('Win rate (fraction of datasets with Δ > 0)')
    axes[1].set_title(f'Method Effect Landscape')
    axes[1].axhline(0.5, color='black', linestyle=':', alpha=0.3)
    axes[1].axvline(0, color='black', linestyle=':', alpha=0.3)
    # Annotate top
    top5 = np.argsort(-mean_deltas)[:5]
    for idx in top5:
        axes[1].annotate(method_names[idx][:20], (mean_deltas[idx], win_rates[idx]),
                          fontsize=7, alpha=0.8)
    plt.tight_layout()
    plt.savefig(output_dir / 'method_effect_landscape.png', dpi=120, bbox_inches='tight')
    plt.close()

    # ============ Dataset clustering visualization ============
    print("Plotting dataset clustering...")
    labels_d, Z_d = cluster_datasets(delta_matrix, n_clusters=5)

    # Dataset signature for each: median Δ across methods
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Per-dataset bar chart (mean Δ across top-30 methods)
    mean_deltas_method = np.nanmean(delta_matrix, axis=0)
    top30_method_idx = np.argsort(-mean_deltas_method)[:30]
    per_dataset_top30_mean = np.nanmean(delta_matrix[:, top30_method_idx], axis=1)

    sorted_ds = np.argsort(per_dataset_top30_mean)
    colors_ds = []
    for i in sorted_ds:
        g = get_group(aliases[i])
        colors_ds.append({'SMD': 'tab:blue', 'Exathlon': 'tab:orange',
                          'Standalone': 'tab:green'}[g])

    axes[0].barh(range(len(sorted_ds)), [per_dataset_top30_mean[i] for i in sorted_ds],
                  color=colors_ds, edgecolor='black', linewidth=0.3)
    axes[0].set_yticks(range(len(sorted_ds)))
    axes[0].set_yticklabels([aliases[i] for i in sorted_ds], fontsize=7)
    axes[0].set_xlabel('Mean Δ (top-30 methods)')
    axes[0].set_title('Dataset Difficulty (mean across top-30 methods)')
    axes[0].axvline(0, color='red', linestyle='--', alpha=0.5)

    # Dataset dendrogram
    dendrogram(Z_d, labels=[aliases[i][:15] for i in range(len(aliases))], ax=axes[1],
                color_threshold=Z_d[-4, 2], leaf_font_size=7)
    axes[1].set_title(f'Dataset Hierarchical Clustering (K=5)')
    axes[1].set_ylabel('Distance')

    plt.tight_layout()
    plt.savefig(output_dir / 'dataset_clustering.png', dpi=120, bbox_inches='tight')
    plt.close()

    # ============ Per-cluster method patterns ============
    print("Plotting cluster heatmap...")
    fig, ax = plt.subplots(figsize=(14, 8))
    # For top-20 methods, plot their delta values per dataset
    top20_method_idx = np.argsort(-mean_deltas_method)[:20]
    plot_matrix = delta_matrix[:, top20_method_idx]

    # Order datasets by hardness
    ds_order = np.argsort(np.nanmean(plot_matrix, axis=1))

    im = ax.imshow(plot_matrix[ds_order], cmap='RdBu_r', vmin=-0.1, vmax=0.1,
                    aspect='auto')
    ax.set_xticks(range(20))
    ax.set_xticklabels([method_names[i][:25] for i in top20_method_idx],
                        rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(aliases)))
    ax.set_yticklabels([aliases[ds_order[i]][:25] for i in range(len(aliases))],
                        fontsize=7)
    ax.set_title('Top-20 Methods × 39 Datasets Delta Matrix\n(red=positive Δ, blue=negative)')
    plt.colorbar(im, ax=ax, label='Δ pak_auc_f1')
    plt.tight_layout()
    plt.savefig(output_dir / 'top20_methods_heatmap.png', dpi=120, bbox_inches='tight')
    plt.close()

    print(f"\nAll visualizations saved to: {output_dir}")
    for f in sorted(output_dir.iterdir()):
        if f.suffix == '.png':
            print(f"  {f.name}: {f.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
