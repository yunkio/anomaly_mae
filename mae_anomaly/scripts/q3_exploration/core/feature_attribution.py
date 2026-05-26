"""
Per-Feature Importance / Attribution Analysis.

본 module은 274 model에서 각 input feature가 anomaly detection에 얼마나 contribute하는지 측정.

Methods:
1. Feature variance importance: anomaly position에서 각 feature의 deviation
2. Per-feature anomaly separation: feature i alone으로 detection capability
3. Feature ablation: feature i 제거 시 score change
4. Per-feature reconstruction error: model이 어떤 feature를 잘 reconstruct하는가
"""
import numpy as np
from typing import Dict, List, Tuple, Optional


def per_feature_anomaly_separation(signals, point_labels, eval_mask=None):
    """각 feature 별 anomaly vs normal separation.

    Returns: per-feature dict with separation, snr metrics.
    """
    if eval_mask is not None:
        labels_m = point_labels[eval_mask]
        signals_m = signals[eval_mask]
    else:
        labels_m = point_labels
        signals_m = signals

    n_features = signals.shape[1]
    per_feat = []

    for j in range(n_features):
        feat = signals_m[:, j]
        in_anom = feat[labels_m == 1]
        out_anom = feat[labels_m == 0]

        if len(in_anom) < 3 or len(out_anom) < 10:
            per_feat.append({
                'feature_idx': j, 'separation': 0.0, 'in_mean': 0.0,
                'in_std': 0.0, 'out_mean': 0.0, 'out_std': 0.0,
                'variance_ratio': 0.0, 't_statistic': 0.0,
            })
            continue

        in_mean, in_std = float(in_anom.mean()), float(in_anom.std() + 1e-9)
        out_mean, out_std = float(out_anom.mean()), float(out_anom.std() + 1e-9)
        separation = (in_mean - out_mean) / out_std
        var_ratio = float(in_std ** 2 / (out_std ** 2 + 1e-9))

        # Welch's t-statistic
        se = np.sqrt(in_std**2 / len(in_anom) + out_std**2 / len(out_anom))
        t_stat = (in_mean - out_mean) / (se + 1e-9)

        per_feat.append({
            'feature_idx': j,
            'separation': float(separation),
            'in_mean': in_mean, 'in_std': in_std,
            'out_mean': out_mean, 'out_std': out_std,
            'variance_ratio': var_ratio,
            't_statistic': float(t_stat),
        })

    return per_feat


def per_feature_top_k_importance(signals, point_labels, eval_mask=None, K=5):
    """Top-K most discriminative features."""
    per_feat = per_feature_anomaly_separation(signals, point_labels, eval_mask)
    abs_seps = sorted(per_feat, key=lambda x: -abs(x['separation']))
    return abs_seps[:K]


def feature_correlation_matrix(signals, point_labels, eval_mask=None):
    """Feature-label correlation per feature."""
    if eval_mask is not None:
        labels_m = point_labels[eval_mask]
        signals_m = signals[eval_mask]
    else:
        labels_m = point_labels
        signals_m = signals

    corrs = []
    for j in range(signals.shape[1]):
        feat = signals_m[:, j]
        if feat.std() < 1e-9 or labels_m.std() < 1e-9:
            corrs.append(0.0)
            continue
        c = np.corrcoef(feat, labels_m.astype(float))[0, 1]
        corrs.append(float(c))
    return np.array(corrs)


def per_feature_subset_detection(signals, point_labels, regions, eval_mask=None,
                                    feature_subset_indices=None, sigma=10):
    """Feature subset만 사용해 detection 가능성 측정.

    Args:
        feature_subset_indices: list of feature indices. None = all features.

    Returns: pak_auc_f1 using only these features.
    """
    from scipy.ndimage import gaussian_filter1d
    from mae_anomaly.scripts.q3_exploration.core.evaluation import pak_auc_f1

    if feature_subset_indices is None:
        feature_subset_indices = list(range(signals.shape[1]))

    subset_signals = signals[:, feature_subset_indices]
    # Per-point anomaly score = max deviation from rolling mean
    window_size = 100
    point_scores = np.zeros(signals.shape[0])
    for j in range(subset_signals.shape[1]):
        feat = subset_signals[:, j]
        # Rolling mean
        from scipy.ndimage import uniform_filter1d
        rolling_mean = uniform_filter1d(feat, size=window_size, mode='reflect')
        rolling_std = np.array([
            feat[max(0, i-window_size//2):min(len(feat), i+window_size//2)].std() + 1e-9
            for i in range(0, len(feat), 50)
        ])
        # Interpolate std to full length
        rolling_std_full = np.interp(np.arange(len(feat)),
                                       np.arange(0, len(feat), 50),
                                       rolling_std)
        z = np.abs((feat - rolling_mean) / rolling_std_full)
        point_scores = np.maximum(point_scores, z)

    smoothed = gaussian_filter1d(point_scores, sigma=sigma, mode='reflect')
    return pak_auc_f1(smoothed, point_labels, regions, eval_mask)


def feature_ablation_analysis(base_score_func, signals, point_labels, regions,
                                eval_mask=None):
    """One-feature-out ablation: drop each feature, measure score change.

    Args:
        base_score_func: function (signals) → score sequence

    Returns: per-feature ablation impact.
    """
    from mae_anomaly.scripts.q3_exploration.core.evaluation import pak_auc_f1

    baseline_score = base_score_func(signals)
    baseline_pak = pak_auc_f1(baseline_score, point_labels, regions, eval_mask)

    impacts = []
    for j in range(signals.shape[1]):
        # Replace feature j with its mean
        signals_ablated = signals.copy()
        signals_ablated[:, j] = signals[:, j].mean()

        ablated_score = base_score_func(signals_ablated)
        ablated_pak = pak_auc_f1(ablated_score, point_labels, regions, eval_mask)
        impacts.append({
            'feature_idx': j,
            'baseline_pak': float(baseline_pak),
            'ablated_pak': float(ablated_pak),
            'impact': float(baseline_pak - ablated_pak),  # higher = more important
        })

    return impacts


def feature_dimensionality_reduction(signals, n_components=5, method='pca'):
    """Reduce signals to lower-dim representation.

    Args:
        method: 'pca', 'ica' (TODO)

    Returns: (reduced_signals, explained_variance_ratio)
    """
    if method == 'pca':
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(signals)
        return reduced, pca.explained_variance_ratio_
    else:
        raise NotImplementedError(method)
