"""
Training Data Distribution Audit.

본 module은 train 데이터의 quality + anomaly contamination를 측정.

Hypothesis: train 데이터에 anomaly-like patterns가 contaminated → model은 본 patterns를 normal로 학습 →
test 시점에 본 patterns의 anomaly에 대해 reverse learning 또는 weak detection.

Methods:
- train_signal_stats: train data의 distribution statistics (mean, std, skew, kurt)
- train_unstable_periods: rolling std anomaly detection in train
- train_test_distribution_distance: train vs test KL/Wasserstein
- train_anomaly_density_estimate: train에 anomaly-like region 식별 + count
"""
import numpy as np
from typing import Dict, List, Tuple
from scipy.stats import skew, kurtosis, wasserstein_distance
from scipy.ndimage import uniform_filter1d


def train_signal_stats(train_signals):
    """Train data의 per-feature statistical summary."""
    if len(train_signals) < 100:
        return None
    n_features = train_signals.shape[1]

    per_feat = []
    for j in range(n_features):
        feat = train_signals[:, j]
        per_feat.append({
            'feature_idx': j,
            'mean': float(feat.mean()),
            'std': float(feat.std()),
            'skew': float(skew(feat)) if feat.std() > 1e-9 else 0.0,
            'kurt': float(kurtosis(feat)) if feat.std() > 1e-9 else 0.0,
            'min': float(feat.min()),
            'max': float(feat.max()),
            'iqr': float(np.percentile(feat, 75) - np.percentile(feat, 25)),
            'unique_ratio': float(len(np.unique(feat)) / len(feat)),
        })
    return per_feat


def train_unstable_periods(train_signals, window_size=200, std_threshold=3.0):
    """Train data에서 anomaly-like (불안정한) 시간 구간 식별.

    Returns: list of (start, end) for periods with abnormally high rolling std.
    """
    n_features = train_signals.shape[1]
    instability_score = np.zeros(len(train_signals))

    for j in range(n_features):
        feat = train_signals[:, j]
        rolling_mean = uniform_filter1d(feat, size=window_size, mode='reflect')
        rolling_var = uniform_filter1d(feat ** 2, size=window_size, mode='reflect') - rolling_mean ** 2
        rolling_std = np.sqrt(np.maximum(rolling_var, 1e-9))
        baseline_std = float(np.median(rolling_std))
        z = rolling_std / (baseline_std + 1e-9)
        instability_score = np.maximum(instability_score, z)

    # Identify periods above threshold
    above = instability_score > std_threshold
    unstable_periods = []
    i = 0
    while i < len(above):
        if above[i]:
            j = i
            while j < len(above) and above[j]:
                j += 1
            unstable_periods.append((i, j, float(instability_score[i:j].max())))
            i = j
        else:
            i += 1
    return unstable_periods, instability_score


def train_test_distribution_distance(train_signals, test_signals, n_features_subset=None):
    """Train vs test per-feature distribution distance.

    Returns: per-feature distance, summary.
    """
    n_features = train_signals.shape[1]
    if n_features_subset is not None and n_features > n_features_subset:
        # Top variable features
        variances = train_signals.var(axis=0)
        top_idx = np.argsort(-variances)[:n_features_subset]
        train_signals = train_signals[:, top_idx]
        test_signals = test_signals[:, top_idx]
        n_features = n_features_subset

    per_feat = []
    for j in range(n_features):
        tr_feat = train_signals[:, j]
        te_feat = test_signals[:, j]
        # Mean shift (in train std units)
        tr_std = tr_feat.std() + 1e-9
        mean_shift = float(abs(te_feat.mean() - tr_feat.mean()) / tr_std)
        # Variance ratio
        var_ratio = float((te_feat.var() + 1e-9) / (tr_feat.var() + 1e-9))
        # Wasserstein (1D)
        try:
            wass = float(wasserstein_distance(tr_feat[::10], te_feat[::10]))  # subsample
        except:
            wass = 0.0
        # Range overlap
        tr_min, tr_max = tr_feat.min(), tr_feat.max()
        te_min, te_max = te_feat.min(), te_feat.max()
        overlap_start = max(tr_min, te_min)
        overlap_end = min(tr_max, te_max)
        union_start = min(tr_min, te_min)
        union_end = max(tr_max, te_max)
        range_iou = float(max(0, overlap_end - overlap_start) / (union_end - union_start + 1e-9))

        per_feat.append({
            'feature_idx': j,
            'mean_shift': mean_shift,
            'var_ratio': var_ratio,
            'wasserstein': wass,
            'range_iou': range_iou,
        })
    return per_feat


def train_anomaly_density_estimate(train_signals, test_anomaly_regions, test_signals,
                                     context_size=200, similarity_threshold=0.5):
    """Train에서 test anomaly와 유사한 patterns의 빈도 추정.

    For each test anomaly region:
    - Extract centroid
    - Count train windows within similarity_threshold (raw distance metric)
    - Compare to base rate (random train windows)

    Returns: per-region contamination estimate.
    """
    total_train = len(train_signals)
    if total_train < 200: return []

    results = []
    for r in test_anomaly_regions:
        in_sig = test_signals[r.start:r.end]
        if len(in_sig) < 3: continue
        region_len = r.end - r.start
        in_centroid = in_sig.mean(axis=0)

        # Train windows sliding
        n_windows = max(1, (total_train - region_len) // 10)
        train_centroids = np.array([
            train_signals[i * 10 : i * 10 + region_len].mean(axis=0)
            for i in range(n_windows)
            if i * 10 + region_len <= total_train
        ])
        if len(train_centroids) == 0: continue

        # Distances to in_centroid
        distances = np.linalg.norm(train_centroids - in_centroid, axis=1)
        # Test set baseline: random non-anomaly test centroids
        # Use median train_centroid distance as baseline scale
        scale = float(np.median(distances)) + 1e-9
        normalized = distances / scale
        # Contamination = fraction of train windows that are "close" (<0.5 of median)
        contam_count = int((normalized < similarity_threshold).sum())
        contam_ratio = contam_count / len(train_centroids)

        results.append({
            'region_start': r.start, 'region_end': r.end,
            'train_min_distance': float(distances.min()),
            'train_median_distance': float(np.median(distances)),
            'train_max_distance': float(distances.max()),
            'contam_count': contam_count,
            'contam_ratio': float(contam_ratio),
            'n_train_windows': len(train_centroids),
        })
    return results
