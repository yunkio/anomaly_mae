"""
Inverted Signal Anomaly Deep Analysis (Q3 v7).

Q3 v6 P20에서 식별된 cluster 3 (n=98, 19%) inverted contrast anomalies:
- Anomaly position score < context score (inverted)
- 274 model이 본 anomaly type에 wrong signal 생성

본 module은 4 hypothesis를 검증하기 위한 utilities:
- H1 (Label noise): label vs raw signal consistency
- H2 (Reverse learning): per-region reconstruction quality
- H3 (Feature absence): feature space distance / variance
- H4 (Training contamination): train data similarity to anomaly patterns

Provides:
- identify_inverted_regions: cluster 3 후보 region 식별
- per_region_recon_analysis: per-region recon quality
- raw_signal_distance: anomaly vs normal raw signal distance
- find_similar_training_regions: train data에서 nearest neighbor
- inverted_region_features: 7+ characterization features per region
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance


def identify_inverted_regions(score_smoothed, regions, eval_mask=None,
                                context_size=200, min_contrast=-0.5):
    """Cluster 3 (inverted contrast) regions 식별.

    Returns list of (region, contrast, isolation, additional_metrics).
    """
    results = []
    for r in regions:
        if eval_mask is not None and not eval_mask[r.start:r.end].any():
            continue
        ctx_start = max(0, r.start - context_size)
        ctx_end = min(len(score_smoothed), r.end + context_size)
        in_region = score_smoothed[r.start:r.end]
        ctx = np.concatenate([score_smoothed[ctx_start:r.start],
                               score_smoothed[r.end:ctx_end]])
        if len(ctx) < 5 or len(in_region) == 0:
            continue

        in_max = float(in_region.max())
        in_mean = float(in_region.mean())
        in_min = float(in_region.min())
        ctx_max = float(ctx.max())
        ctx_mean = float(ctx.mean())
        ctx_std = float(ctx.std())

        contrast = (in_max - ctx_max) / (ctx_std + 1e-9)
        isolation = (in_max - ctx_mean) / (ctx_std + 1e-9)

        # Inverted criteria: contrast < min_contrast (negative)
        is_inverted = contrast < min_contrast

        results.append({
            'region': r,
            'in_max': in_max, 'in_mean': in_mean, 'in_min': in_min,
            'ctx_max': ctx_max, 'ctx_mean': ctx_mean, 'ctx_std': ctx_std,
            'contrast': contrast, 'isolation': isolation,
            'is_inverted': is_inverted,
            'length': r.end - r.start,
        })
    return results


def per_region_recon_analysis(per_window_recon, per_window_disc, per_window_fm,
                                window_start_indices, ds_regions, num_patches,
                                patch_size, total_length, eval_mask=None):
    """각 anomaly region의 per-channel reconstruction quality 측정.

    Hypothesis check (H2 Reverse Learning):
    - If model is "reverse-learning", anomaly region의 recon error는 NORMAL region보다 낮을 것
    - Normal context의 recon error는 anomaly region보다 높을 것

    Returns: per-region dict with recon/disc/fm in-region mean vs context mean.
    """
    # Aggregate to point-level
    from mae_anomaly.scripts.q3_exploration.core.scoring import aggregate_K50, stride_subsample

    mask = np.arange(len(window_start_indices)) % 21 == 0
    ws = window_start_indices[mask]
    pt_r = aggregate_K50(per_window_recon[mask], ws, num_patches, patch_size, total_length)
    pt_d = aggregate_K50(per_window_disc[mask], ws, num_patches, patch_size, total_length)
    pt_f = aggregate_K50(per_window_fm[mask], ws, num_patches, patch_size, total_length)

    results = []
    for r in ds_regions:
        if eval_mask is not None and not eval_mask[r.start:r.end].any():
            continue
        context_size = 200
        ctx_start = max(0, r.start - context_size)
        ctx_end = min(total_length, r.end + context_size)
        ctx_mask = np.zeros(total_length, dtype=bool)
        ctx_mask[ctx_start:r.start] = True
        ctx_mask[r.end:ctx_end] = True

        region_mask = np.zeros(total_length, dtype=bool)
        region_mask[r.start:r.end] = True

        if ctx_mask.sum() < 5 or region_mask.sum() == 0:
            continue

        in_recon = pt_r[region_mask].mean()
        ctx_recon = pt_r[ctx_mask].mean()
        in_disc = pt_d[region_mask].mean()
        ctx_disc = pt_d[ctx_mask].mean()
        in_fm = pt_f[region_mask].mean()
        ctx_fm = pt_f[ctx_mask].mean()

        # Inverted criteria: in_recon < ctx_recon (i.e., model reconstructs anomaly BETTER than normal)
        recon_inverted = in_recon < ctx_recon
        disc_inverted = in_disc < ctx_disc
        fm_inverted = in_fm < ctx_fm

        results.append({
            'region_start': r.start, 'region_end': r.end,
            'in_recon': float(in_recon), 'ctx_recon': float(ctx_recon),
            'recon_ratio': float(in_recon / (ctx_recon + 1e-9)),
            'in_disc': float(in_disc), 'ctx_disc': float(ctx_disc),
            'disc_ratio': float(in_disc / (ctx_disc + 1e-9)),
            'in_fm': float(in_fm), 'ctx_fm': float(ctx_fm),
            'fm_ratio': float(in_fm / (ctx_fm + 1e-9)),
            'recon_inverted': bool(recon_inverted),
            'disc_inverted': bool(disc_inverted),
            'fm_inverted': bool(fm_inverted),
            'all_channels_inverted': bool(recon_inverted and disc_inverted and fm_inverted),
        })

    return results


def raw_signal_distance_analysis(signals, ds_regions, eval_mask=None,
                                    context_size=200, n_features_subset=None):
    """Raw signal에서 anomaly vs normal distance 측정.

    Hypothesis check (H3 Feature Absence):
    - If features lack anomaly signal, raw distance (anomaly vs normal) will be SMALL
    - Compared to normal-normal distance

    Args:
        signals: (T, n_features) raw signal
        ds_regions: list of AnomalyRegion
        eval_mask: optional eval mask
        n_features_subset: optional feature subset (e.g., top-K most variable)

    Returns: per-region distance metrics.
    """
    if n_features_subset is not None and signals.shape[1] > n_features_subset:
        # Select top-K most variable features
        variances = signals.var(axis=0)
        top_idx = np.argsort(-variances)[:n_features_subset]
        signals = signals[:, top_idx]

    total_length = signals.shape[0]
    results = []

    for r in ds_regions:
        if eval_mask is not None and not eval_mask[r.start:r.end].any():
            continue
        # Region signal
        in_sig = signals[r.start:r.end]
        if len(in_sig) < 3:
            continue

        # Context
        ctx_start = max(0, r.start - context_size)
        ctx_end = min(total_length, r.end + context_size)
        ctx_sig = np.concatenate([signals[ctx_start:r.start],
                                    signals[r.end:ctx_end]])
        if len(ctx_sig) < 10:
            continue

        # Mahalanobis-like distance: (in_mean - ctx_mean) / ctx_std
        ctx_mean = ctx_sig.mean(axis=0)
        ctx_std = ctx_sig.std(axis=0) + 1e-9
        in_mean = in_sig.mean(axis=0)
        std_diff = np.abs(in_mean - ctx_mean) / ctx_std  # (n_features,)
        max_dim_diff = float(std_diff.max())
        mean_dim_diff = float(std_diff.mean())

        # Variance distinguishability
        in_var = in_sig.var(axis=0)
        ctx_var = ctx_sig.var(axis=0) + 1e-9
        var_ratio = float(np.mean(in_var / ctx_var))

        # Wasserstein distance (1D, per feature)
        try:
            wass_dists = [wasserstein_distance(in_sig[:, i], ctx_sig[:, i])
                          for i in range(signals.shape[1])]
            mean_wass = float(np.mean(wass_dists))
            max_wass = float(np.max(wass_dists))
        except Exception:
            mean_wass = max_wass = 0.0

        results.append({
            'region_start': r.start, 'region_end': r.end,
            'length': r.end - r.start,
            'max_std_dim_diff': max_dim_diff,
            'mean_std_dim_diff': mean_dim_diff,
            'var_ratio': var_ratio,
            'mean_wasserstein': mean_wass,
            'max_wasserstein': max_wass,
        })

    return results


def find_similar_training_patterns(signals, train_ratio, regions,
                                     context_size=200, n_top_neighbors=5):
    """Train data에서 anomaly region과 유사한 patterns 찾기.

    Hypothesis check (H4 Training Contamination):
    - Train data에 anomaly와 유사한 patterns가 자주 등장 → model이 본 patterns를 normal로 학습

    Args:
        signals: full signals
        train_ratio: train/test split point
        regions: anomaly regions (in test portion)

    Returns: per-region nearest-neighbor distances in train data.
    """
    total_length = signals.shape[0]
    split = int(train_ratio * total_length)
    train_signals = signals[:split]

    results = []
    for r in regions:
        in_sig = signals[r.start:r.end]
        if len(in_sig) < 3:
            continue
        region_len = r.end - r.start
        in_centroid = in_sig.mean(axis=0)

        # Sliding window comparison in train data
        n_train_windows = max(1, (split - region_len) // 10)  # stride=10
        train_centroids = np.array([
            train_signals[i * 10 : i * 10 + region_len].mean(axis=0)
            for i in range(n_train_windows)
            if i * 10 + region_len <= split
        ])
        if len(train_centroids) == 0:
            continue

        # Distances
        distances = np.linalg.norm(train_centroids - in_centroid, axis=1)
        sorted_dists = np.sort(distances)
        top_dists = sorted_dists[:n_top_neighbors]
        median_dist = float(np.median(distances))
        min_dist = float(distances.min())

        # How many train windows are within "anomaly threshold"?
        # Threshold = distance to nearest non-anomaly window centroid in test
        test_centroids = []
        if split < total_length - region_len:
            test_n = (total_length - split - region_len) // 10
            for i in range(min(test_n, 100)):
                pos = split + i * 10
                if pos + region_len <= total_length:
                    test_centroids.append(signals[pos:pos+region_len].mean(axis=0))
            test_centroids = np.array(test_centroids)
            if len(test_centroids) > 0:
                test_dists_to_in = np.linalg.norm(test_centroids - in_centroid, axis=1)
                test_median = float(np.median(test_dists_to_in))
            else:
                test_median = median_dist
        else:
            test_median = median_dist

        # Contamination indicator: ratio of train windows within test_median
        contamination_ratio = float((distances <= test_median).mean())

        results.append({
            'region_start': r.start, 'region_end': r.end,
            'train_min_distance': min_dist,
            'train_median_distance': median_dist,
            'train_top5_mean': float(np.mean(top_dists)),
            'test_median_distance': test_median,
            'contamination_ratio': contamination_ratio,
            'n_train_windows_checked': len(train_centroids),
        })

    return results
