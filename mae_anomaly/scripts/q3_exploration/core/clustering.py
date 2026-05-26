"""
Clustering utilities for dataset characterization.

Provides:
- DatasetSignatureExtractor: 8-feature unsupervised signature
- DatasetSignatureExtractorSupervised: extends with baseline_pak (uses labels)
- run_kmeans_clustering: KMeans with std scaling + log transforms
- get_cluster_methods: per-cluster best method discovery
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, kurtosis


@dataclass
class DatasetSignature:
    """Per-dataset characteristic signature.

    Features (raw):
    - Anomaly structure: median_seg, max_seg, std_seg, n_regions, anomaly_ratio
    - Score distribution: skewness, kurtosis, iqr, autocorr_lag1
    - Channel balance: recon_disc_ratio, disc_fm_ratio
    - Optional supervised: baseline_pak
    """
    alias: str
    features: Dict[str, float] = field(default_factory=dict)

    def feature_vector(self, keys: List[str]) -> np.ndarray:
        return np.array([self.features.get(k, 0.0) for k in keys])


def extract_signature_unsupervised(
    base_smoothed: np.ndarray,
    pt_r: np.ndarray, pt_d: np.ndarray, pt_f: np.ndarray,
) -> Dict[str, float]:
    """Pure unsupervised signature (labels 사용 안 함)."""
    # Score distribution shape
    skewness = float(skew(base_smoothed))
    kurt = float(kurtosis(base_smoothed))
    iqr = float(np.percentile(base_smoothed, 75) - np.percentile(base_smoothed, 25))
    std_to_mean = float(base_smoothed.std() / (abs(base_smoothed.mean()) + 1e-9))

    # Autocorrelation lag-1 (자기상관)
    s = base_smoothed - base_smoothed.mean()
    if s.std() > 1e-9 and len(s) > 1:
        autocorr_lag1 = float(np.mean(s[:-1] * s[1:]) / (s.var() + 1e-9))
    else:
        autocorr_lag1 = 0.0

    # Score peakedness: top 5% vs median
    top5_median_ratio = float(np.percentile(base_smoothed, 95) /
                               (np.median(base_smoothed) + 1e-9))

    # Channel balance
    recon_mean = pt_r.mean() + 1e-9
    disc_mean = pt_d.mean() + 1e-9
    fm_mean = pt_f.mean() + 1e-9
    recon_disc_ratio = float(recon_mean / disc_mean)
    disc_fm_ratio = float(disc_mean / fm_mean)

    # Sequence length (proxy for dataset scale)
    seq_len_log = float(np.log10(len(base_smoothed) + 1))

    # Peak frequency: how many high-percentile points
    high_threshold = np.percentile(base_smoothed, 90)
    n_peaks = (base_smoothed > high_threshold).sum()
    peak_density = float(n_peaks / len(base_smoothed))

    return {
        'skewness': skewness,
        'kurtosis': kurt,
        'iqr': iqr,
        'std_to_mean': std_to_mean,
        'autocorr_lag1': autocorr_lag1,
        'top5_median_ratio': top5_median_ratio,
        'recon_disc_ratio_log': float(np.log(recon_disc_ratio + 1e-9)),
        'disc_fm_ratio_log': float(np.log(disc_fm_ratio + 1e-9)),
        'seq_len_log': seq_len_log,
        'peak_density': peak_density,
    }


def extract_signature_supervised(
    regions,
    point_labels: np.ndarray,
    baseline_pak: float,
) -> Dict[str, float]:
    """Supervised signature (label-dependent)."""
    seg_lens = [r.end - r.start for r in regions]
    if not seg_lens:
        return {
            'median_seg_log': 0.0,
            'max_seg_log': 0.0,
            'std_seg_log': 0.0,
            'n_regions_log': 0.0,
            'anomaly_ratio': 0.0,
            'baseline_pak': baseline_pak,
        }
    return {
        'median_seg_log': float(np.log10(np.median(seg_lens) + 1)),
        'max_seg_log': float(np.log10(max(seg_lens) + 1)),
        'std_seg_log': float(np.log10(np.std(seg_lens) + 1) if len(seg_lens) > 1 else 0.0),
        'n_regions_log': float(np.log10(len(seg_lens) + 1)),
        'anomaly_ratio': float(point_labels.mean()),
        'baseline_pak': baseline_pak,
    }


def run_kmeans_clustering(
    signatures: Dict[str, DatasetSignature],
    feature_keys: List[str],
    n_clusters: int,
    random_state: int = 42,
):
    """KMeans clustering with standard scaling.

    Returns:
        (cluster_ids_dict, cluster_centers, X_scaled)
        cluster_ids_dict: {alias: cluster_id}
    """
    aliases = list(signatures.keys())
    X = np.array([signatures[a].feature_vector(feature_keys) for a in aliases])

    # Handle NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_ids = km.fit_predict(X_scaled)

    return dict(zip(aliases, cluster_ids.tolist())), km.cluster_centers_, X_scaled


def get_cluster_best_methods(
    cluster_ids_dict: Dict[str, int],
    method_scores_dict: Dict[str, Dict[str, float]],
    baseline_method: str = 'baseline_gauss10',
) -> Dict[int, str]:
    """Per-cluster best method discovery (cluster Δ 최대 method).

    Args:
        cluster_ids_dict: {alias: cluster_id}
        method_scores_dict: {alias: {method_name: score}}
        baseline_method: 비교 baseline

    Returns:
        {cluster_id: best_method_name}
    """
    cluster_to_aliases = {}
    for alias, cid in cluster_ids_dict.items():
        cluster_to_aliases.setdefault(cid, []).append(alias)

    best_methods = {}
    for cid, aliases in cluster_to_aliases.items():
        method_deltas = {}
        all_methods = list(next(iter(method_scores_dict.values())).keys())
        for method in all_methods:
            if method == baseline_method:
                continue
            deltas = [method_scores_dict[a][method] - method_scores_dict[a][baseline_method]
                      for a in aliases if method in method_scores_dict[a]]
            if deltas:
                method_deltas[method] = np.mean(deltas)
        if method_deltas:
            best_methods[cid] = max(method_deltas, key=method_deltas.get)
        else:
            best_methods[cid] = baseline_method
    return best_methods


def apply_cluster_routing(
    cluster_ids_dict: Dict[str, int],
    cluster_to_method: Dict[int, str],
    method_scores_dict: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """Apply cluster routing: 각 alias를 그 cluster의 best method로 routing.

    Returns:
        {alias: routed_score}
    """
    routed = {}
    for alias, cid in cluster_ids_dict.items():
        method = cluster_to_method.get(cid, 'baseline_gauss10')
        routed[alias] = method_scores_dict[alias].get(method, 0.0)
    return routed
