"""
Meta-analysis: Method-method and Dataset-dataset clustering.

본 module:
- compute_method_correlation_matrix: pairwise correlation of method delta-vectors over datasets
- cluster_methods: hierarchical clustering based on correlation
- compute_dataset_response_matrix: each dataset의 method-response vector
- cluster_datasets: dataset preference clustering
- method_redundancy: information-theoretic redundancy estimation
- failure_mode_analysis: hard datasets identification
"""
import numpy as np
from typing import List, Tuple, Dict
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform


def compute_method_correlation_matrix(delta_matrix: np.ndarray, method='pearson'):
    """Pairwise correlation between methods over their delta vectors.

    Args:
        delta_matrix: (n_datasets, n_methods)
        method: 'pearson', 'spearman'

    Returns:
        corr: (n_methods, n_methods)
    """
    if method == 'pearson':
        # numpy corrcoef on columns
        corr = np.corrcoef(delta_matrix.T)
    elif method == 'spearman':
        from scipy.stats import spearmanr
        corr, _ = spearmanr(delta_matrix)
    else:
        raise ValueError(method)
    # Handle NaN
    corr = np.nan_to_num(corr, nan=0.0)
    return corr


def cluster_methods(corr_matrix: np.ndarray, n_clusters: int = 10,
                     linkage_method='average'):
    """Hierarchical clustering based on (1 - correlation) distance.

    Returns:
        cluster_labels: (n_methods,) array of cluster IDs
        linkage_matrix: for dendrogram
    """
    n = corr_matrix.shape[0]
    # Distance = 1 - correlation (clipped to [0, 2])
    dist = np.clip(1.0 - corr_matrix, 0, 2)
    # Symmetrize and make zero diagonal
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0)
    # Condensed form
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method=linkage_method)
    labels = fcluster(Z, t=n_clusters, criterion='maxclust')
    return labels, Z


def cluster_datasets(delta_matrix: np.ndarray, n_clusters: int = 5,
                      linkage_method='average', method='spearman'):
    """Cluster datasets by their method-response signature.

    Args:
        delta_matrix: (n_datasets, n_methods) — each row is dataset's method response
        n_clusters: target cluster count

    Returns:
        labels: (n_datasets,) array
        Z: linkage matrix
    """
    n_datasets = delta_matrix.shape[0]
    if method == 'spearman':
        from scipy.stats import spearmanr
        # Dataset-dataset correlation
        corr, _ = spearmanr(delta_matrix.T)
    else:
        corr = np.corrcoef(delta_matrix)
    corr = np.nan_to_num(corr, nan=0.0)

    dist = np.clip(1.0 - corr, 0, 2)
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method=linkage_method)
    labels = fcluster(Z, t=n_clusters, criterion='maxclust')
    return labels, Z


def method_redundancy_analysis(delta_matrix: np.ndarray, threshold=0.95):
    """Identify highly redundant method pairs.

    Args:
        delta_matrix: (n_datasets, n_methods)
        threshold: correlation threshold

    Returns:
        redundant_pairs: list of (i, j, corr) tuples
    """
    corr = compute_method_correlation_matrix(delta_matrix)
    n = corr.shape[0]
    redundant_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if corr[i, j] >= threshold:
                redundant_pairs.append((i, j, corr[i, j]))
    return sorted(redundant_pairs, key=lambda x: -x[2])


def method_diversity_subset(delta_matrix: np.ndarray, k=10):
    """Select k most diverse methods (maximizing pairwise distance).

    Greedy: start with method with highest mean Δ, then iteratively
    add method maximizing min-distance to selected set.
    """
    n_methods = delta_matrix.shape[1]
    corr = compute_method_correlation_matrix(delta_matrix)
    dist = 1.0 - corr

    # Start: method with highest mean Δ
    mean_deltas = np.nanmean(delta_matrix, axis=0)
    selected = [int(np.nanargmax(mean_deltas))]

    while len(selected) < k:
        # For each candidate, min distance to selected
        candidates = [i for i in range(n_methods) if i not in selected]
        if not candidates:
            break
        min_dists = np.array([min(dist[c, s] for s in selected) for c in candidates])
        best_idx = np.argmax(min_dists)
        selected.append(candidates[best_idx])

    return selected


def failure_mode_analysis(delta_matrix: np.ndarray, alias_list: List[str],
                            method_names: List[str], n_methods_for_consensus: int = 10):
    """Hard datasets identification.

    Hard = consistently fails (low / negative Δ) across many methods.

    Args:
        delta_matrix: (n_datasets, n_methods)
        n_methods_for_consensus: only consider top-n_methods (by mean Δ)

    Returns:
        hard_datasets: ordered list with hardness scores
    """
    # 평균 Δ 기준으로 top methods 선정 (winners)
    mean_deltas = np.nanmean(delta_matrix, axis=0)
    top_method_idx = np.argsort(-mean_deltas)[:n_methods_for_consensus]

    # 각 dataset에 대해 top methods의 평균 Δ
    top_delta_matrix = delta_matrix[:, top_method_idx]
    dataset_hardness = -np.nanmean(top_delta_matrix, axis=1)  # 더 negative = harder

    # Sort
    sorted_indices = np.argsort(-dataset_hardness)
    return [(alias_list[i], dataset_hardness[i],
             {method_names[j]: delta_matrix[i, j] for j in top_method_idx})
            for i in sorted_indices]


def universal_winners_analysis(delta_matrix: np.ndarray, method_names: List[str],
                                   alias_list: List[str], min_datasets=30):
    """Find methods that work on majority of datasets.

    Universal winner = positive Δ on most datasets (W/L ratio).
    """
    n_datasets, n_methods = delta_matrix.shape
    method_stats = []
    for j in range(n_methods):
        deltas = delta_matrix[:, j]
        wins = (deltas > 0).sum()
        losses = (deltas < 0).sum()
        n_valid = (~np.isnan(deltas)).sum()
        mean_delta = np.nanmean(deltas)
        median_delta = np.nanmedian(deltas)
        worst_delta = np.nanmin(deltas)
        best_delta = np.nanmax(deltas)
        method_stats.append({
            'name': method_names[j],
            'wins': int(wins),
            'losses': int(losses),
            'n_valid': int(n_valid),
            'mean_delta': float(mean_delta),
            'median_delta': float(median_delta),
            'worst_delta': float(worst_delta),
            'best_delta': float(best_delta),
            'win_rate': wins / max(n_valid, 1),
        })

    return method_stats


def per_dataset_best_method(delta_matrix: np.ndarray, method_names: List[str],
                              alias_list: List[str]):
    """Per-dataset: which method is best?"""
    n_datasets, n_methods = delta_matrix.shape
    result = []
    for i in range(n_datasets):
        deltas = delta_matrix[i]
        if np.all(np.isnan(deltas)):
            continue
        best_idx = np.nanargmax(deltas)
        result.append({
            'alias': alias_list[i],
            'best_method': method_names[best_idx],
            'best_delta': float(deltas[best_idx]),
        })
    return result


def per_method_best_dataset(delta_matrix: np.ndarray, method_names: List[str],
                              alias_list: List[str]):
    """Per-method: which dataset has the highest gain?"""
    n_datasets, n_methods = delta_matrix.shape
    result = []
    for j in range(n_methods):
        deltas = delta_matrix[:, j]
        if np.all(np.isnan(deltas)):
            continue
        best_idx = np.nanargmax(deltas)
        result.append({
            'method': method_names[j],
            'best_dataset': alias_list[best_idx],
            'best_delta': float(deltas[best_idx]),
        })
    return result
