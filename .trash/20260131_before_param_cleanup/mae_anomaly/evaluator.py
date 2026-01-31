"""
Evaluator for Self-Distilled MAE Anomaly Detection

Includes PA%K (Point-Adjust with K%) evaluation metric for time series anomaly detection.
PA%K is a segment-level adjustment that counts an anomaly segment as detected if at least
K% of its points are flagged as anomalies.

Point-level PA%K:
    For proper evaluation with stride=1 sliding windows, window-level scores are aggregated
    to point-level scores using one of three methods:
    - 'mean': Average of all window scores covering each timestep
    - 'median': Median of all window scores covering each timestep
    - 'voting': Majority vote of binary predictions (default)
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast
from typing import Dict, Tuple, List, Optional
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve
)

from .dataset_sliding import ANOMALY_TYPE_NAMES


def _build_aggregation_map(
    window_start_indices: np.ndarray,
    patch_size: int,
    num_patches: int,
    total_length: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precompute the geometry mapping from (window, patch) to timesteps.

    Returns:
        flat_t: valid timestep indices
        flat_wp: valid flat (w*num_patches + p) indices into patch_scores.ravel()
        point_coverage: (total_length,) coverage count per timestep
        covered: (total_length,) bool mask of covered timesteps
    """
    n_windows = len(window_start_indices)
    offsets = np.arange(patch_size)
    patch_indices = np.arange(num_patches)

    base_positions = window_start_indices[:, np.newaxis] + patch_indices[np.newaxis, :] * patch_size
    all_t = (base_positions[:, :, np.newaxis] + offsets[np.newaxis, np.newaxis, :]).ravel()

    # Flat index into patch_scores.ravel(): w * num_patches + p, repeated patch_size times
    w_grid = np.arange(n_windows)[:, np.newaxis, np.newaxis]
    p_grid = patch_indices[np.newaxis, :, np.newaxis]
    all_wp = np.broadcast_to(
        w_grid * num_patches + p_grid, (n_windows, num_patches, patch_size)
    ).ravel()

    valid = (all_t >= 0) & (all_t < total_length)
    flat_t = all_t[valid]
    flat_wp = all_wp[valid]
    point_coverage = np.bincount(flat_t, minlength=total_length).astype(int)
    covered = point_coverage > 0
    return flat_t, flat_wp, point_coverage, covered


def _aggregate_with_map(
    patch_scores_flat: np.ndarray,
    flat_t: np.ndarray,
    flat_wp: np.ndarray,
    point_coverage: np.ndarray,
    covered: np.ndarray,
    total_length: int,
    method: str = 'mean',
    threshold: Optional[float] = None,
) -> np.ndarray:
    """Aggregate patch scores to point-level using precomputed geometry map."""
    point_scores = np.full(total_length, np.nan)
    flat_s = patch_scores_flat[flat_wp]

    if method == 'mean':
        score_sum = np.bincount(flat_t, weights=flat_s, minlength=total_length)
        point_scores[covered] = score_sum[covered] / point_coverage[covered]
    elif method == 'voting':
        if threshold is None:
            raise ValueError("threshold is required for voting method")
        votes = np.bincount(flat_t, weights=(flat_s > threshold).astype(float), minlength=total_length)
        point_scores[covered] = (votes[covered] > point_coverage[covered] / 2).astype(float)
    elif method in ('median', 'max'):
        sort_idx = np.argsort(flat_t)
        sorted_t = flat_t[sort_idx]
        sorted_s = flat_s[sort_idx]
        splits = np.searchsorted(sorted_t, np.arange(total_length), side='left')
        splits_end = np.searchsorted(sorted_t, np.arange(total_length), side='right')
        agg_fn = np.median if method == 'median' else np.max
        for t in np.where(covered)[0]:
            point_scores[t] = agg_fn(sorted_s[splits[t]:splits_end[t]])
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
    return point_scores


def aggregate_patch_scores_to_point_level(
    patch_scores: np.ndarray,
    window_start_indices: np.ndarray,
    seq_length: int,
    patch_size: int,
    num_patches: int,
    total_length: int,
    method: str = 'voting',
    threshold: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate per-patch scores to point-level scores

    Each window has N patches, and each patch's score is assigned to the timesteps
    that patch covers. Each timestep is covered by ~seq_length windows.

    Args:
        patch_scores: (n_windows, num_patches) per-patch anomaly scores
        window_start_indices: (n_windows,) start index of each window in the time series
        seq_length: Window size (e.g., 100)
        patch_size: Size of each patch (e.g., 10)
        num_patches: Number of patches per window (e.g., 10)
        total_length: Total length of the time series
        method: Aggregation method ('mean', 'median', 'max', 'voting')
        threshold: Threshold for binary prediction (required for 'voting')

    Returns:
        point_scores: (total_length,) aggregated scores per timestep (NaN for no coverage)
        point_coverage: (total_length,) number of (window, patch) pairs covering each timestep
    """
    point_scores = np.full(total_length, np.nan)
    point_coverage = np.zeros(total_length, dtype=int)

    # Build flat arrays of (timestep_index, score) for all patch-timestep assignments
    # Each window w covers patches 0..num_patches-1, each patch covers patch_size timesteps
    n_windows = len(window_start_indices)

    # For each (window, patch), the covered timesteps are:
    #   start_idx + p_idx * patch_size + offset, for offset in [0, patch_size)
    # Vectorize: create all timestep indices and corresponding scores at once
    offsets = np.arange(patch_size)  # (patch_size,)
    patch_indices = np.arange(num_patches)  # (num_patches,)

    # Base positions: (n_windows, num_patches) -> start of each patch
    base_positions = window_start_indices[:, np.newaxis] + patch_indices[np.newaxis, :] * patch_size
    # (n_windows, num_patches)

    # All timestep indices: (n_windows, num_patches, patch_size)
    all_t_indices = base_positions[:, :, np.newaxis] + offsets[np.newaxis, np.newaxis, :]
    # Corresponding scores: (n_windows, num_patches, patch_size) - broadcast from patch_scores
    all_scores = np.broadcast_to(
        patch_scores[:, :, np.newaxis], (n_windows, num_patches, patch_size)
    )

    # Flatten
    flat_t = all_t_indices.ravel()
    flat_s = all_scores.ravel()

    # Filter valid indices (0 <= t < total_length)
    valid = (flat_t >= 0) & (flat_t < total_length)
    flat_t = flat_t[valid]
    flat_s = flat_s[valid]

    # Compute coverage via bincount
    point_coverage = np.bincount(flat_t, minlength=total_length).astype(int)
    covered = point_coverage > 0

    if method == 'mean':
        score_sum = np.bincount(flat_t, weights=flat_s, minlength=total_length)
        point_scores[covered] = score_sum[covered] / point_coverage[covered]
    elif method == 'voting':
        if threshold is None:
            raise ValueError("threshold is required for voting method")
        votes = np.bincount(flat_t, weights=(flat_s > threshold).astype(float), minlength=total_length)
        point_scores[covered] = (votes[covered] > point_coverage[covered] / 2).astype(float)
    elif method in ('median', 'max'):
        # For median/max, need per-timestep grouping (less common path)
        sort_idx = np.argsort(flat_t)
        sorted_t = flat_t[sort_idx]
        sorted_s = flat_s[sort_idx]
        splits = np.searchsorted(sorted_t, np.arange(total_length), side='left')
        splits_end = np.searchsorted(sorted_t, np.arange(total_length), side='right')
        agg_fn = np.median if method == 'median' else np.max
        for t in np.where(covered)[0]:
            point_scores[t] = agg_fn(sorted_s[splits[t]:splits_end[t]])
    else:
        raise ValueError(f"Unknown aggregation method: {method}")

    return point_scores, point_coverage


def compute_pa_k_adjusted_predictions(
    predictions: np.ndarray,
    labels: np.ndarray,
    k_percent: int = 20
) -> np.ndarray:
    """Compute PA%K (Point-Adjust with K%) adjusted predictions

    PA%K adjusts anomaly detection evaluation by considering segment-level detection:
    - If >= K% of an anomaly segment is detected, the ENTIRE segment is considered detected
    - This is more lenient and realistic for time series anomaly detection

    Args:
        predictions: Binary predictions (0/1) for each sample
        labels: True labels (0/1) for each sample
        k_percent: Detection threshold percentage (default 10%)

    Returns:
        Adjusted predictions array where anomaly segments are either all 1 or all 0
        based on whether K% threshold was met
    """
    n = len(labels)
    adjusted_preds = predictions.copy()

    # Find contiguous anomaly segments
    i = 0
    while i < n:
        if labels[i] == 1:  # Start of anomaly segment
            start = i
            while i < n and labels[i] == 1:
                i += 1
            end = i  # [start, end)

            # Check if >= K% of segment is detected
            segment_preds = predictions[start:end]
            detection_ratio = segment_preds.mean()

            if detection_ratio >= k_percent / 100:
                # Segment is detected - all points count as detected
                adjusted_preds[start:end] = 1
            else:
                # Segment not detected - all points count as not detected
                adjusted_preds[start:end] = 0
        else:
            i += 1

    return adjusted_preds


def compute_pa_k_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    k_percent: int = 20
) -> Dict[str, float]:
    """Compute PA%K (Point-Adjust with K%) metrics

    Args:
        predictions: Binary predictions (0/1) for each sample
        labels: True labels (0/1) for each sample
        k_percent: Detection threshold percentage (default 20%)

    Returns:
        Dict with 'pa_k_precision', 'pa_k_recall', 'pa_k_f1'
    """
    adjusted_preds = compute_pa_k_adjusted_predictions(predictions, labels, k_percent)

    pa_precision = precision_score(labels, adjusted_preds, zero_division=0)
    pa_recall = recall_score(labels, adjusted_preds, zero_division=0)
    pa_f1 = f1_score(labels, adjusted_preds, zero_division=0)

    return {
        'pa_k_precision': pa_precision,
        'pa_k_recall': pa_recall,
        'pa_k_f1': pa_f1
    }


def precompute_point_score_indices(
    window_start_indices: np.ndarray,
    seq_length: int,
    patch_size: int,
    total_length: int,
    num_patches: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute which (window, patch) indices cover each timestep.

    Returns flat arrays for vectorized voting instead of list-of-lists.

    Args:
        window_start_indices: (n_windows,) start index of each window
        seq_length: Window size (e.g., 100)
        patch_size: Size of each patch (e.g., 10)
        total_length: Total length of the time series
        num_patches: Number of patches per window

    Returns:
        flat_timesteps: (N,) timestep index for each (window, patch, offset) assignment
        flat_score_indices: (N,) flat index into scores.ravel() for each assignment
        point_coverage: (total_length,) count of covering windows/patches per timestep
    """
    n_windows = len(window_start_indices)
    offsets = np.arange(patch_size)
    patch_indices = np.arange(num_patches)

    # Base timestep positions: (n_windows, num_patches)
    base_positions = window_start_indices[:, np.newaxis] + patch_indices[np.newaxis, :] * patch_size

    # All timestep indices: (n_windows, num_patches, patch_size)
    all_t = base_positions[:, :, np.newaxis] + offsets[np.newaxis, np.newaxis, :]

    # Flat score indices: w_idx * num_patches + p_idx, broadcast to (n_windows, num_patches, patch_size)
    w_grid = np.arange(n_windows)[:, np.newaxis, np.newaxis]
    p_grid = patch_indices[np.newaxis, :, np.newaxis]
    all_score_idx = (w_grid * num_patches + p_grid) * np.ones(patch_size, dtype=int)[np.newaxis, np.newaxis, :]

    flat_t = all_t.ravel()
    flat_score_idx = all_score_idx.ravel()

    # Filter valid
    valid = (flat_t >= 0) & (flat_t < total_length)
    flat_t = flat_t[valid]
    flat_score_idx = flat_score_idx[valid]

    point_coverage = np.bincount(flat_t, minlength=total_length).astype(int)
    return flat_t, flat_score_idx, point_coverage


def vectorized_voting_for_all_thresholds(
    scores: np.ndarray,
    point_indices,  # (flat_timesteps, flat_score_indices) tuple or legacy list-of-lists
    point_coverage: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Compute majority-voted point-level binary scores for all thresholds.

    Args:
        scores: Patch scores. Shape (n_windows, n_patches).
        point_indices: Tuple (flat_timesteps, flat_score_indices) from precompute_point_score_indices
        point_coverage: From precompute_point_score_indices
        thresholds: (n_thresholds,) threshold values

    Returns:
        (n_thresholds, total_length) binary voted point scores
    """
    n_thresholds = len(thresholds)
    total_length = len(point_coverage)
    flat_scores = scores.ravel()

    # Unpack vectorized indices
    if isinstance(point_indices, tuple) and len(point_indices) == 2:
        flat_t, flat_score_idx = point_indices
    else:
        # Legacy fallback: convert list-of-lists to flat arrays
        n_patches = scores.shape[1]
        all_t = []
        all_si = []
        for t, indices in enumerate(point_indices):
            for w, p in indices:
                all_t.append(t)
                all_si.append(w * n_patches + p)
        flat_t = np.array(all_t, dtype=int)
        flat_score_idx = np.array(all_si, dtype=int)

    # Get all patch scores for all assignments: (N_assignments,)
    assigned_scores = flat_scores[flat_score_idx]

    # For each threshold, count votes per timestep using bincount
    result = np.zeros((n_thresholds, total_length), dtype=np.float64)
    half_coverage = point_coverage / 2.0

    for ti in range(n_thresholds):
        votes_binary = (assigned_scores > thresholds[ti]).astype(float)
        vote_counts = np.bincount(flat_t, weights=votes_binary, minlength=total_length)
        result[ti] = (vote_counts > half_coverage).astype(np.float64)

    return result


def _compute_single_pa_k_roc(args: tuple):
    """Compute PA%K ROC curve from pre-voted point scores.

    Args (tuple):
        point_scores_all: (n_thresholds, total_length) binary voted scores
        point_labels: (total_length,) or array-like ground truth labels
        anomaly_regions: list of anomaly region objects with .start, .end, .anomaly_type
        eval_type_mask: (total_length,) bool mask for which points to evaluate
        k_percent: PA%K threshold percentage
        anomaly_type: int or None (None = all types)
        return_mode: True → (fprs, tprs, auc, f1),
                     'all' → (fprs, tprs, auc, optimal_thresh_idx, f1)

    Returns:
        Tuple depending on return_mode
    """
    point_scores_all, point_labels, anomaly_regions, eval_type_mask, k_percent, anomaly_type, return_mode = args

    point_labels = np.asarray(point_labels)
    n_thresholds, total_length = point_scores_all.shape

    # Filter regions by anomaly_type if specified
    if anomaly_type is not None:
        regions = [r for r in anomaly_regions if r.anomaly_type == anomaly_type]
    else:
        regions = list(anomaly_regions)

    # Apply eval_type_mask: only keep points where mask is True
    masked_labels = point_labels.copy()
    masked_labels[~eval_type_mask] = 0  # Exclude masked-out anomaly points

    # Identify positive/negative counts
    n_positive = int(masked_labels.sum())
    n_negative = int((eval_type_mask & (point_labels == 0)).sum())

    if n_positive == 0 or n_negative == 0:
        empty_fprs = np.array([0.0, 1.0])
        empty_tprs = np.array([0.0, 1.0])
        if return_mode == 'all':
            return empty_fprs, empty_tprs, 0.5, 0, 0.0
        return empty_fprs, empty_tprs, 0.5, 0.0

    k_ratio = k_percent / 100.0
    tprs = np.zeros(n_thresholds)
    fprs = np.zeros(n_thresholds)
    f1s = np.zeros(n_thresholds)

    # Pre-extract region boundaries for vectorized segment processing
    region_starts = np.array([r.start for r in regions])
    region_ends = np.array([min(r.end, total_length) for r in regions])
    valid_regions = region_ends > region_starts
    region_starts = region_starts[valid_regions]
    region_ends = region_ends[valid_regions]
    region_lengths = region_ends - region_starts

    for t_idx in range(n_thresholds):
        preds = point_scores_all[t_idx].copy()

        # Vectorized PA%K segment adjustment using cumsum
        if len(region_starts) > 0:
            cumsum_preds = np.concatenate([[0], np.cumsum(preds)])
            region_sums = cumsum_preds[region_ends] - cumsum_preds[region_starts]
            detection_ratios = region_sums / region_lengths

            for r_idx in range(len(region_starts)):
                s, e = region_starts[r_idx], region_ends[r_idx]
                if detection_ratios[r_idx] >= k_ratio:
                    preds[s:e] = 1.0
                else:
                    preds[s:e] = 0.0

        # Apply mask
        preds[~eval_type_mask] = 0

        tp = int((preds * masked_labels).sum())
        fp = int((preds * (eval_type_mask & (point_labels == 0))).sum())
        fn = n_positive - tp

        tprs[t_idx] = tp / n_positive if n_positive > 0 else 0
        fprs[t_idx] = fp / n_negative if n_negative > 0 else 0

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tprs[t_idx]
        f1s[t_idx] = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Sort by FPR
    sorted_idx = np.argsort(fprs)
    fprs_sorted = fprs[sorted_idx]
    tprs_sorted = tprs[sorted_idx]

    auc = float(np.trapz(tprs_sorted, fprs_sorted))

    # Find optimal threshold (max F1)
    optimal_idx = int(np.argmax(f1s))
    best_f1 = float(f1s[optimal_idx])

    if return_mode == 'all':
        return fprs_sorted, tprs_sorted, auc, optimal_idx, best_f1
    return fprs_sorted, tprs_sorted, auc, best_f1


def _compute_voted_point_predictions(
    patch_scores: np.ndarray,
    point_indices_flat: tuple,
    point_coverage: np.ndarray,
    threshold: float,
    total_length: int,
) -> np.ndarray:
    """Compute majority-voted binary point predictions at a single threshold.

    For each timestamp, collect all patch scores covering it,
    count how many exceed threshold, and predict anomaly if majority votes yes.

    Args:
        patch_scores: (n_windows, num_patches) patch-level scores
        point_indices_flat: tuple of (flat_t, flat_score_idx) from precompute_point_score_indices
        point_coverage: (total_length,) coverage count per timestamp
        threshold: decision threshold
        total_length: total number of timestamps

    Returns:
        (total_length,) binary predictions
    """
    flat_t, flat_si = point_indices_flat
    flat_scores = patch_scores.ravel()
    binary = (flat_scores[flat_si] > threshold).astype(np.float64)

    # Sum votes per timestamp using bincount
    vote_counts = np.bincount(flat_t, weights=binary, minlength=total_length)

    # Majority vote: predict anomaly if more than half the patches vote yes
    preds = np.zeros(total_length, dtype=np.float64)
    covered = point_coverage > 0
    preds[covered] = (vote_counts[covered] > point_coverage[covered] / 2).astype(np.float64)

    return preds


def _compute_pa_k_f1_at_threshold(
    voted_preds: np.ndarray,
    point_labels: np.ndarray,
    anomaly_regions,
    k_percent: int,
    eval_mask: np.ndarray,
) -> float:
    """Compute PA%K F1 score from voted binary predictions at a fixed threshold.

    Args:
        voted_preds: (total_length,) binary predictions from voting
        point_labels: (total_length,) ground truth
        anomaly_regions: list of anomaly region objects
        k_percent: PA%K threshold percentage
        eval_mask: (total_length,) bool mask for which points to evaluate

    Returns:
        PA%K F1 score
    """
    preds = voted_preds.copy()
    total_length = len(preds)
    k_ratio = k_percent / 100.0

    # Apply PA%K segment adjustment
    for region in anomaly_regions:
        start = region.start
        end = min(region.end, total_length)
        if end <= start:
            continue
        detection_ratio = preds[start:end].mean()
        if detection_ratio >= k_ratio:
            preds[start:end] = 1.0
        else:
            preds[start:end] = 0.0

    # Apply eval mask
    preds[~eval_mask] = 0
    masked_labels = point_labels.copy()
    masked_labels[~eval_mask] = 0

    tp = float((preds * masked_labels).sum())
    fp = float((preds * (eval_mask & (point_labels == 0))).sum())
    fn = float(masked_labels.sum() - tp)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return f1


def compute_segment_pa_k_detection_rate(
    point_scores: np.ndarray,
    point_labels,
    anomaly_regions,
    anomaly_type: int,
    threshold: float = 0.5,
    k_percent: int = 20
) -> float:
    """Compute PA%K detection rate for segments of a specific anomaly type.

    Args:
        point_scores: (total_length,) point-level scores (can be binary voted)
        point_labels: (total_length,) ground truth labels
        anomaly_regions: list of anomaly region objects with .start, .end, .anomaly_type
        anomaly_type: which anomaly type to evaluate
        threshold: score threshold for binary prediction
        k_percent: PA%K threshold percentage

    Returns:
        Fraction of anomaly segments detected (0.0 to 1.0)
    """
    point_scores = np.asarray(point_scores)
    predictions = (point_scores > threshold).astype(int)

    regions = [r for r in anomaly_regions if r.anomaly_type == anomaly_type]
    if len(regions) == 0:
        return 0.0

    k_ratio = k_percent / 100.0
    detected = 0
    total = len(regions)

    for region in regions:
        start = region.start
        end = min(region.end, len(predictions))
        if end <= start:
            continue
        seg_preds = predictions[start:end]
        if seg_preds.mean() >= k_ratio:
            detected += 1

    return detected / total if total > 0 else 0.0


class DatasetMetadata:
    """Lightweight metadata-only substitute for SlidingWindowDataset.

    Holds only the attributes that Evaluator needs (point_labels,
    window_start_indices, anomaly_regions) without storing the full
    windows array, saving ~2-3 GB for large window sizes.
    """

    def __init__(self, point_labels, window_start_indices, anomaly_regions):
        self.point_labels = point_labels
        self.window_start_indices = window_start_indices
        self.anomaly_regions = anomaly_regions


class Evaluator:
    """Evaluator for anomaly detection

    Args:
        model: Trained model
        config: Model configuration
        test_loader: DataLoader for test data
        test_dataset: Optional SlidingWindowDataset (or DatasetMetadata) for
                     point-level PA%K evaluation. If provided, enables
                     point-level PA%K with window score aggregation.
    """

    def __init__(
        self,
        model,
        config,
        test_loader: DataLoader,
        test_dataset=None
    ):
        self.model = model
        self.config = config
        self.test_loader = test_loader
        self.test_dataset = test_dataset
        if self.model is not None:
            self.model.eval()

        # Mixed Precision Training (AMP) for inference
        self.use_amp = getattr(config, 'use_amp', False) and torch.cuda.is_available()

        # Point-level PA%K requires test_dataset with specific attributes
        self.can_compute_point_level_pa_k = (
            test_dataset is not None and
            hasattr(test_dataset, 'point_labels') and
            hasattr(test_dataset, 'window_start_indices')
        )

        # Cache for raw scores to avoid redundant forward passes
        self._cache = {}

    def clear_cache(self):
        """Clear cached scores (call when model or data changes)"""
        self._cache = {}

    def _build_cache_dict(
        self,
        recon_patches: np.ndarray,
        disc_patches: np.ndarray,
        student_recon_patches: np.ndarray,
        labels: np.ndarray,
        sample_types: np.ndarray,
        anomaly_types: np.ndarray,
    ) -> dict:
        """Build raw_scores cache dict from pre-computed patch-level scores.

        This is the shared logic used by both _get_cached_scores() (after GPU
        forward pass) and set_precomputed_patch_scores() (from external data).

        Args:
            recon_patches: (n_windows, num_patches) teacher reconstruction scores
            disc_patches: (n_windows, num_patches) discrepancy scores
            student_recon_patches: (n_windows, num_patches) student reconstruction scores
            labels: (n_windows,) window-level binary labels
            sample_types: (n_windows,) window-level sample type indicators
            anomaly_types: (n_windows,) window-level anomaly type indicators

        Returns:
            Complete cache dict with patch/window scores, labels, and derived metadata
        """
        # Derive window-level scores by averaging
        window_recon = recon_patches.mean(axis=1)
        window_disc = disc_patches.mean(axis=1)
        window_student_recon = student_recon_patches.mean(axis=1)

        # Compute patch labels if possible
        patch_labels = None
        if self.can_compute_point_level_pa_k:
            patch_labels = self._compute_patch_labels()

        # Compute patch-level sample_types based on patch labels (generalized approach)
        # sample_type: 2=anomaly (patch has anomaly), 1=disturbing (normal patch in anomaly window), 0=pure_normal
        n_windows, num_patches = recon_patches.shape
        patch_sample_types = np.zeros((n_windows, num_patches), dtype=np.int64)

        if patch_labels is not None:
            window_has_anomaly = (patch_labels.sum(axis=1) > 0)  # (n_windows,)
            patch_sample_types[patch_labels == 1] = 2
            # Vectorized: normal patches in anomaly windows → disturbing
            disturbing_mask = window_has_anomaly[:, np.newaxis] & (patch_labels == 0)
            patch_sample_types[disturbing_mask] = 1

        # Compute patch-level anomaly_types (inherit from window if patch has anomaly, else 0)
        patch_anomaly_types = np.zeros((n_windows, num_patches), dtype=np.int64)
        if patch_labels is not None:
            # Vectorized: broadcast window anomaly types to patches with anomaly
            patch_anomaly_types = np.where(
                patch_labels == 1,
                anomaly_types[:, np.newaxis],
                0
            )

        return {
            'patch_recon': recon_patches,
            'patch_disc': disc_patches,
            'patch_student_recon': student_recon_patches,
            'window_recon': window_recon,
            'window_disc': window_disc,
            'window_student_recon': window_student_recon,
            'labels': labels,
            'sample_types': sample_types,  # window-level
            'anomaly_types': anomaly_types,  # window-level
            'patch_labels': patch_labels,
            'patch_sample_types': patch_sample_types,  # (n_windows, num_patches)
            'patch_anomaly_types': patch_anomaly_types,  # (n_windows, num_patches)
        }

    def set_precomputed_patch_scores(
        self,
        recon_patches: np.ndarray,
        disc_patches: np.ndarray,
        student_recon_patches: np.ndarray,
        labels: np.ndarray,
        sample_types: np.ndarray,
        anomaly_types: np.ndarray,
    ):
        """Populate the evaluator cache with pre-computed patch scores.

        Use this to avoid a redundant GPU forward pass when patch scores
        have already been computed (e.g., by collect_all_visualization_data).

        Args:
            recon_patches: (n_windows, num_patches) teacher reconstruction scores
            disc_patches: (n_windows, num_patches) discrepancy scores
            student_recon_patches: (n_windows, num_patches) student reconstruction scores
            labels: (n_windows,) window-level binary labels
            sample_types: (n_windows,) window-level sample type indicators
            anomaly_types: (n_windows,) window-level anomaly type indicators
        """
        self._cache['raw_scores'] = self._build_cache_dict(
            recon_patches, disc_patches, student_recon_patches,
            labels, sample_types, anomaly_types,
        )

    def _get_cached_scores(self):
        """Get cached raw scores, computing if needed

        Returns:
            dict with 'patch_recon', 'patch_disc', 'window_recon', 'window_disc',
                 'labels', 'sample_types', 'anomaly_types', 'patch_labels',
                 'patch_sample_types', 'patch_anomaly_types'
                Patch arrays are (n_windows, num_patches), window arrays are (n_windows,)
        """
        cache_key = 'raw_scores'
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Compute patch-level scores once
        recon_patches, disc_patches, student_recon_patches, labels, sample_types, anomaly_types = self._compute_patch_scores_all_patches()

        result = self._build_cache_dict(
            recon_patches, disc_patches, student_recon_patches,
            labels, sample_types, anomaly_types,
        )

        self._cache[cache_key] = result
        return result

    def _apply_scoring_formula(self, recon: np.ndarray, disc: np.ndarray, scoring_mode: str) -> np.ndarray:
        """Apply scoring formula to raw recon/disc scores

        Args:
            recon: Raw reconstruction scores
            disc: Raw discrepancy scores
            scoring_mode: 'default', 'normalized', 'adaptive', or 'ratio_weighted'

        Returns:
            Combined anomaly scores
        """
        if scoring_mode == 'normalized':
            recon_mean, recon_std = recon.mean(), recon.std() + 1e-4
            disc_mean, disc_std = disc.mean(), disc.std() + 1e-4
            recon_z = (recon - recon_mean) / recon_std
            disc_z = (disc - disc_mean) / disc_std
            return recon_z + disc_z
        elif scoring_mode == 'adaptive':
            adaptive_lambda = recon.mean() / (disc.mean() + 1e-4)
            return recon + adaptive_lambda * disc
        elif scoring_mode == 'ratio_weighted':
            disc_median = np.median(disc) + 1e-4
            return recon * (1 + disc / disc_median)
        else:  # default
            return recon + self.config.lambda_disc * disc

    def _get_point_score_indices(self):
        """Get cached point score indices (geometry-only, scoring-mode independent)."""
        cache_key = 'point_score_indices'
        if cache_key not in self._cache:
            ws_indices = np.array(self.test_dataset.window_start_indices)
            total_len = len(self.test_dataset.point_labels)
            self._cache[cache_key] = precompute_point_score_indices(
                window_start_indices=ws_indices,
                seq_length=self.config.seq_length,
                patch_size=self.config.patch_size,
                total_length=total_len,
                num_patches=self.config.num_patches,
            )
        return self._cache[cache_key]

    def _get_aggregation_map(self):
        """Get cached aggregation map (geometry-only, scoring-mode independent)."""
        cache_key = 'aggregation_map'
        if cache_key not in self._cache:
            ws_indices = np.array(self.test_dataset.window_start_indices)
            total_len = len(self.test_dataset.point_labels)
            self._cache[cache_key] = _build_aggregation_map(
                ws_indices, self.config.patch_size,
                self.config.num_patches, total_len
            )
        return self._cache[cache_key]

    def _compute_patch_labels(self) -> np.ndarray:
        """Compute patch-level labels from point-level labels (vectorized)."""
        if not self.can_compute_point_level_pa_k:
            raise RuntimeError("Patch labels require test_dataset with point_labels and window_start_indices")

        point_labels = np.asarray(self.test_dataset.point_labels)
        window_start_indices = np.asarray(self.test_dataset.window_start_indices)
        patch_size = self.config.patch_size
        num_patches = self.config.num_patches
        n_windows = len(window_start_indices)
        total_len = len(point_labels)

        # Vectorized: build (n_windows, num_patches) start positions
        ws = window_start_indices[:, np.newaxis]  # (n_windows, 1)
        p_idx = np.arange(num_patches)[np.newaxis, :]  # (1, num_patches)
        starts = ws + p_idx * patch_size  # (n_windows, num_patches)
        ends = np.minimum(starts + patch_size, total_len)

        # Use cumulative sum trick to check if any point in [start, end) has anomaly
        cumsum = np.concatenate([[0], np.cumsum(point_labels)])
        patch_sums = cumsum[ends] - cumsum[starts]
        patch_labels = (patch_sums > 0).astype(int)

        return patch_labels

    def _compute_patch_scores_all_patches(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute per-patch recon/disc/student_recon scores by masking each patch one at a time

        Optimized: All patches processed in a single forward pass by expanding batch dimension.
        Returns per-patch scores to be used directly for point-level aggregation.

        Returns:
            recon_patch_scores: (n_windows, num_patches) per-patch reconstruction scores
            disc_patch_scores: (n_windows, num_patches) per-patch discrepancy scores
            student_recon_patch_scores: (n_windows, num_patches) per-patch student reconstruction scores
            labels: (n_windows,) window labels
            sample_types: (n_windows,) sample type indicators
            anomaly_types: (n_windows,) anomaly type indicators
        """
        all_recon_patches = []
        all_disc_patches = []
        all_student_recon_patches = []
        all_labels = []
        all_sample_types = []
        all_anomaly_types = []

        patch_size = self.config.patch_size
        num_patches = self.config.num_patches

        with torch.no_grad(), autocast('cuda', enabled=self.use_amp):
            for batch in self.test_loader:
                if len(batch) == 5:
                    sequences, window_labels, point_labels, sample_types, anomaly_types = batch
                elif len(batch) == 4:
                    sequences, window_labels, point_labels, sample_types = batch
                    anomaly_types = torch.zeros_like(window_labels)
                else:
                    sequences, window_labels, point_labels = batch
                    sample_types = torch.zeros_like(window_labels)
                    anomaly_types = torch.zeros_like(window_labels)

                sequences = sequences.to(self.config.device)
                batch_size, seq_length, num_features = sequences.shape

                # Process patches in batches for memory efficiency
                patch_batch_size = 4  # Fixed: process 4 patches at a time
                batch_recon_patches = torch.zeros(batch_size, num_patches, device=self.config.device)
                batch_disc_patches = torch.zeros(batch_size, num_patches, device=self.config.device)
                batch_student_recon_patches = torch.zeros(batch_size, num_patches, device=self.config.device)

                for batch_start in range(0, num_patches, patch_batch_size):
                    batch_end = min(batch_start + patch_batch_size, num_patches)
                    current_batch_patches = batch_end - batch_start

                    # Expand only for current patch batch
                    expanded = sequences.unsqueeze(1).expand(-1, current_batch_patches, -1, -1)
                    expanded = expanded.reshape(batch_size * current_batch_patches, seq_length, num_features)

                    # Create masks for current patch batch
                    masks = torch.ones(batch_size * current_batch_patches, seq_length, device=self.config.device)
                    for i, patch_idx in enumerate(range(batch_start, batch_end)):
                        start_pos = patch_idx * patch_size
                        end_pos = start_pos + patch_size
                        masks[i::current_batch_patches, start_pos:end_pos] = 0

                    # Forward pass for current batch
                    teacher_output, student_output, _ = self.model(expanded, masking_ratio=0.0, mask=masks)

                    # Compute errors
                    recon_error = ((teacher_output - expanded) ** 2).mean(dim=2)
                    student_recon_error = ((student_output - expanded) ** 2).mean(dim=2)
                    discrepancy = ((teacher_output - student_output) ** 2).mean(dim=2)

                    # Reshape to (B, current_batch_patches, S)
                    recon_error = recon_error.view(batch_size, current_batch_patches, seq_length)
                    student_recon_error = student_recon_error.view(batch_size, current_batch_patches, seq_length)
                    discrepancy = discrepancy.view(batch_size, current_batch_patches, seq_length)

                    # Extract scores for each patch's masked region
                    for i, patch_idx in enumerate(range(batch_start, batch_end)):
                        start_pos = patch_idx * patch_size
                        end_pos = start_pos + patch_size
                        batch_recon_patches[:, patch_idx] = recon_error[:, i, start_pos:end_pos].mean(dim=1)
                        batch_student_recon_patches[:, patch_idx] = student_recon_error[:, i, start_pos:end_pos].mean(dim=1)
                        batch_disc_patches[:, patch_idx] = discrepancy[:, i, start_pos:end_pos].mean(dim=1)

                    # Clear intermediate tensors
                    del expanded, masks, teacher_output, student_output, recon_error, student_recon_error, discrepancy

                all_recon_patches.append(batch_recon_patches.cpu().numpy())
                all_disc_patches.append(batch_disc_patches.cpu().numpy())
                all_student_recon_patches.append(batch_student_recon_patches.cpu().numpy())
                all_labels.append(window_labels.cpu().numpy())
                all_sample_types.append(sample_types.cpu().numpy())
                all_anomaly_types.append(anomaly_types.cpu().numpy())

        return (
            np.concatenate(all_recon_patches),          # (n_windows, num_patches)
            np.concatenate(all_disc_patches),            # (n_windows, num_patches)
            np.concatenate(all_student_recon_patches),   # (n_windows, num_patches)
            np.concatenate(all_labels),
            np.concatenate(all_sample_types),
            np.concatenate(all_anomaly_types)
        )

    def compute_anomaly_scores(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute anomaly scores for all samples in test_loader

        Masks each patch one at a time (all_patches mode), N forward passes per window.

        Supports different scoring modes via config.anomaly_score_mode:
        - 'default': recon + lambda_disc * disc
        - 'normalized': Z-score normalization (recon_z + disc_z)
        - 'adaptive': Auto-scaled lambda (recon + (mean_recon/mean_disc) * disc)
        - 'ratio_weighted': Ratio-based (recon * (1 + disc/median_disc))

        Uses caching to avoid redundant forward passes.

        Returns:
            scores: (n_samples,) anomaly scores (window-level)
            labels: (n_samples,) true labels
            sample_types: (n_samples,) sample type indicators
            anomaly_types: (n_samples,) anomaly type indicators
        """
        score_mode = getattr(self.config, 'anomaly_score_mode', 'default')

        # Get cached raw scores
        cached = self._get_cached_scores()

        recon_all = cached['window_recon']
        disc_all = cached['window_disc']
        labels = cached['labels']
        sample_types = cached['sample_types']
        anomaly_types = cached['anomaly_types']

        # Apply scoring formula
        scores = self._apply_scoring_formula(recon_all, disc_all, score_mode)

        return scores, labels, sample_types, anomaly_types

    def compute_detailed_losses(self) -> Dict[str, np.ndarray]:
        """Compute detailed losses for all samples in test_loader

        Returns patch-level data (n_windows x num_patches,) flattened.

        Sample_types are computed per-patch:
        - 0 = pure_normal: normal patch in normal window
        - 1 = disturbing: normal patch in anomaly-containing window
        - 2 = anomaly: patch containing anomaly

        Returns:
            Dictionary containing:
                reconstruction_loss: (n_samples,) reconstruction loss per sample
                discrepancy_loss: (n_samples,) discrepancy loss per sample
                total_loss: (n_samples,) total loss per sample
                labels: (n_samples,) true labels
                sample_types: (n_samples,) sample type indicators (patch-level)
                anomaly_types: (n_samples,) anomaly type indicators
        """
        cached = self._get_cached_scores()

        # Flatten patch-level data to 1D
        recon_loss = cached['patch_recon'].flatten()
        disc_loss = cached['patch_disc'].flatten()
        labels = cached['patch_labels'].flatten() if cached['patch_labels'] is not None else np.zeros_like(recon_loss)
        sample_types = cached['patch_sample_types'].flatten()
        anomaly_types = cached['patch_anomaly_types'].flatten()

        return {
            'reconstruction_loss': recon_loss,
            'discrepancy_loss': disc_loss,
            'total_loss': recon_loss + self.config.lambda_disc * disc_loss,
            'labels': labels,
            'sample_types': sample_types,
            'anomaly_types': anomaly_types
        }

    def get_performance_by_anomaly_type(self) -> Dict[str, Dict[str, float]]:
        """Compute performance metrics for each anomaly type at point-level.

        Uses mean-aggregated point-level scores and point-level threshold.
        PA%K uses voting with the point-level threshold.

        Only iterates over actual anomaly types (skips 'normal' at index 0).

        Returns:
            Dictionary with anomaly type names as keys, containing metrics for each type
        """
        score_mode = getattr(self.config, 'anomaly_score_mode', 'default')
        cache_key = f'anomaly_type_metrics_{score_mode}'

        if cache_key in self._cache:
            return self._cache[cache_key]

        if not (self.can_compute_point_level_pa_k and hasattr(self.test_dataset, 'anomaly_regions')):
            return {}

        # Get point-level scores and threshold
        cached = self._get_cached_scores()
        patch_recon = cached['patch_recon']
        patch_disc = cached['patch_disc']
        patch_scores = self._apply_scoring_formula(patch_recon, patch_disc, score_mode)

        ws_indices = np.array(self.test_dataset.window_start_indices)
        pt_labels = np.array(self.test_dataset.point_labels)
        total_len = len(pt_labels)
        anomaly_regions = self.test_dataset.anomaly_regions
        normal_mask = (pt_labels == 0)

        # Point-level scores via mean aggregation (cached geometry)
        flat_t, flat_wp, coverage, covered = self._get_aggregation_map()
        point_scores = _aggregate_with_map(patch_scores.ravel(), flat_t, flat_wp, coverage, covered, total_len, method='mean')
        point_scores = np.nan_to_num(point_scores, nan=0.0)

        # Point-level threshold
        if len(np.unique(pt_labels)) < 2:
            self._cache[cache_key] = {}
            return {}

        fpr, tpr, thresholds_arr = roc_curve(pt_labels, point_scores)
        optimal_idx = np.argmax(tpr - fpr)
        threshold = thresholds_arr[optimal_idx]
        point_predictions = (point_scores > threshold).astype(int)

        # Precompute voting data for PA%K (once, shared across all types)
        pt_flat_t, pt_flat_si, pt_coverage = self._get_point_score_indices()

        # PA%K AUROC: threshold sweep with voting (once, shared)
        flat = patch_scores.ravel()
        n_thresh = 100
        vote_thresholds = np.linspace(flat.min() - 0.01, flat.max() + 0.01, n_thresh)
        point_scores_all = vectorized_voting_for_all_thresholds(
            scores=patch_scores,
            point_indices=(pt_flat_t, pt_flat_si),
            point_coverage=pt_coverage,
            thresholds=vote_thresholds,
        )

        # PA%K F1: voted predictions at point-level threshold (once, shared)
        voted_preds = _compute_voted_point_predictions(
            patch_scores, (pt_flat_t, pt_flat_si), pt_coverage, threshold, total_len
        )

        # Build per-point anomaly_type array and precompute eval_type_masks
        point_anomaly_types = np.full(total_len, -1, dtype=int)  # -1 = normal
        for region in anomaly_regions:
            end = min(region.end, total_len)
            point_anomaly_types[region.start:end] = region.anomaly_type

        # Precompute eval_type_mask for each anomaly type (exclude other types' regions)
        # Start with all True, then for each type mask out other types
        base_mask = np.ones(total_len, dtype=bool)
        # Build mask of each type's region coverage
        type_region_masks = {}  # atype_idx -> bool mask of regions belonging to this type
        for atype_idx in range(1, len(ANOMALY_TYPE_NAMES)):
            type_region_masks[atype_idx] = (point_anomaly_types == atype_idx)

        # For each type, eval_type_mask = ~(union of other types' anomaly regions)
        all_anomaly_region_mask = (point_anomaly_types >= 1)  # all anomaly points

        results = {}

        # Skip index 0 ('normal') — only evaluate actual anomaly types
        for atype_idx in range(1, len(ANOMALY_TYPE_NAMES)):
            atype_name = ANOMALY_TYPE_NAMES[atype_idx]
            type_anomaly_mask = type_region_masks[atype_idx]

            if not type_anomaly_mask.any():
                continue

            # For per-type evaluation: all normal + only this type's anomaly points
            eval_mask = normal_mask | type_anomaly_mask
            type_scores = point_scores[eval_mask]
            type_labels = pt_labels[eval_mask]
            type_predictions = point_predictions[eval_mask]

            type_results = {
                'count': int(type_anomaly_mask.sum()),
                'mean_score': float(point_scores[type_anomaly_mask].mean()),
                'std_score': float(point_scores[type_anomaly_mask].std()),
            }

            if len(np.unique(type_labels)) > 1:
                type_results['roc_auc'] = float(roc_auc_score(type_labels, type_scores))
                type_results['precision'] = float(precision_score(type_labels, type_predictions, zero_division=0))
                type_results['recall'] = float(recall_score(type_labels, type_predictions, zero_division=0))
                type_results['f1_score'] = float(f1_score(type_labels, type_predictions, zero_division=0))
                # detection_rate = recall (point-wise fraction of anomaly points detected)
                type_results['detection_rate'] = type_results['recall']

                # PA%K: eval_type_mask excludes other anomaly types
                eval_type_mask = ~(all_anomaly_region_mask & ~type_anomaly_mask)

                type_regions = [r for r in anomaly_regions if r.anomaly_type == atype_idx]

                for k in [10, 20, 50, 80]:
                    # PA%K AUROC: sweep
                    roc_result = _compute_single_pa_k_roc((
                        point_scores_all, pt_labels, anomaly_regions,
                        eval_type_mask, k, atype_idx, True
                    ))
                    _, _, pa_auc, _ = roc_result
                    type_results[f'pa_{k}_roc_auc'] = float(pa_auc)

                    # PA%K F1: fixed threshold + voting
                    pa_f1 = _compute_pa_k_f1_at_threshold(
                        voted_preds, pt_labels, anomaly_regions,
                        k, eval_type_mask
                    )
                    type_results[f'pa_{k}_f1'] = float(pa_f1)

                    # PA%K segment detection rate: same voted_preds (single threshold) for all K
                    pa_det_rate = compute_segment_pa_k_detection_rate(
                        point_scores=voted_preds,
                        point_labels=pt_labels,
                        anomaly_regions=anomaly_regions,
                        anomaly_type=atype_idx,
                        threshold=0.5,
                        k_percent=k
                    )
                    type_results[f'pa_{k}_detection_rate'] = float(pa_det_rate)
            else:
                # All points in eval set are anomaly (no normal points have this type)
                type_results['detection_rate'] = float(type_predictions.mean())

                for k in [10, 20, 50, 80]:
                    pa_rate = compute_segment_pa_k_detection_rate(
                        point_scores=voted_preds,
                        point_labels=pt_labels,
                        anomaly_regions=anomaly_regions,
                        anomaly_type=atype_idx,
                        threshold=0.5,
                        k_percent=k
                    )
                    type_results[f'pa_{k}_detection_rate'] = float(pa_rate)

            results[atype_name] = type_results

        self._cache[cache_key] = results
        return results

    def evaluate(self) -> Dict[str, float]:
        """Evaluate and return metrics at point-level.

        All primary metrics (roc_auc, f1_score, precision, recall) are computed at
        point-level: patch scores are mean-aggregated to physical timestamps, and
        ground-truth point_labels are used directly.

        PA%K metrics use the point-level optimal threshold with majority voting
        for patch→point binary aggregation, then PA%K segment adjustment.

        PA%K AUROC sweeps thresholds (threshold-free metric) with voting.
        PA%K F1 uses the single point-level threshold.

        Uses caching to avoid redundant forward passes when called multiple times
        with different scoring modes.
        """
        score_mode = getattr(self.config, 'anomaly_score_mode', 'default')

        # Get cached raw scores
        cached = self._get_cached_scores()

        results = {}

        # Use patch-level scores from cache
        recon_patches = cached['patch_recon']
        disc_patches = cached['patch_disc']
        sample_types = cached['sample_types']

        # Compute patch-level anomaly scores using cached helper
        patch_scores = self._apply_scoring_formula(recon_patches, disc_patches, score_mode)

        # === Point-level aggregation ===
        if not (self.can_compute_point_level_pa_k and hasattr(self.test_dataset, 'anomaly_regions')):
            # Cannot compute point-level: return zeros
            return {
                'roc_auc': 0.0, 'precision': 0.0, 'recall': 0.0,
                'f1_score': 0.0, 'optimal_threshold': 0.0,
                'pa_10_f1': 0.0, 'pa_10_roc_auc': 0.0,
                'pa_20_f1': 0.0, 'pa_20_roc_auc': 0.0,
                'pa_50_f1': 0.0, 'pa_50_roc_auc': 0.0,
                'pa_80_f1': 0.0, 'pa_80_roc_auc': 0.0,
            }

        ws_indices = np.array(self.test_dataset.window_start_indices)
        pt_labels = np.array(self.test_dataset.point_labels)
        total_len = len(pt_labels)

        # Aggregate patch scores to point-level via mean (cached geometry)
        flat_t, flat_wp, coverage, covered = self._get_aggregation_map()
        point_scores = _aggregate_with_map(patch_scores.ravel(), flat_t, flat_wp, coverage, covered, total_len, method='mean')

        # Handle NaN (uncovered timestamps) — treat as 0 score
        point_scores = np.nan_to_num(point_scores, nan=0.0)

        # === Point-level ROC and threshold ===
        if len(np.unique(pt_labels)) > 1:
            roc_auc = roc_auc_score(pt_labels, point_scores)
            fpr, tpr, thresholds = roc_curve(pt_labels, point_scores)
            optimal_idx = np.argmax(tpr - fpr)
            threshold = thresholds[optimal_idx]

            predictions = (point_scores > threshold).astype(int)
            precision = precision_score(pt_labels, predictions, zero_division=0)
            recall = recall_score(pt_labels, predictions, zero_division=0)
            f1 = f1_score(pt_labels, predictions, zero_division=0)

            results = {
                'roc_auc': roc_auc,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'optimal_threshold': threshold,
            }

            # === PA%K metrics ===
            pa_k_values = [10, 20, 50, 80]

            # Precompute point indices for voting (cached)
            pt_flat_t, pt_flat_si, pt_coverage = self._get_point_score_indices()

            # PA%K AUROC: sweep thresholds with voting (threshold-free metric)
            flat = patch_scores.ravel()
            n_thresholds = 100
            thresholds_arr = np.linspace(flat.min() - 0.01, flat.max() + 0.01, n_thresholds)

            point_scores_all = vectorized_voting_for_all_thresholds(
                scores=patch_scores,
                point_indices=(pt_flat_t, pt_flat_si),
                point_coverage=pt_coverage,
                thresholds=thresholds_arr,
            )

            eval_mask = np.ones(total_len, dtype=bool)
            # Compute voted predictions once (K-independent)
            voted_preds = _compute_voted_point_predictions(
                patch_scores, (pt_flat_t, pt_flat_si), pt_coverage, threshold, total_len
            )
            for k in pa_k_values:
                # PA%K AUROC via threshold sweep
                roc_result = _compute_single_pa_k_roc((
                    point_scores_all, pt_labels, self.test_dataset.anomaly_regions,
                    eval_mask, k, None, True
                ))
                _, _, pa_auc, _ = roc_result
                results[f'pa_{k}_roc_auc'] = float(pa_auc)

                # PA%K F1: use point-level threshold with voting
                pa_f1 = _compute_pa_k_f1_at_threshold(
                    voted_preds, pt_labels, self.test_dataset.anomaly_regions,
                    k, eval_mask
                )
                results[f'pa_{k}_f1'] = float(pa_f1)
        else:
            results = {
                'roc_auc': 0.0, 'precision': 0.0, 'recall': 0.0,
                'f1_score': 0.0, 'optimal_threshold': 0.0,
                'pa_10_f1': 0.0, 'pa_10_roc_auc': 0.0,
                'pa_20_f1': 0.0, 'pa_20_roc_auc': 0.0,
                'pa_50_f1': 0.0, 'pa_50_roc_auc': 0.0,
                'pa_80_f1': 0.0, 'pa_80_roc_auc': 0.0,
            }

        # Disturbing normal performance (window-level, descriptive)
        # sample_type: 0=pure_normal, 1=disturbing_normal, 2=anomaly
        window_scores = self._apply_scoring_formula(
            cached['window_recon'], cached['window_disc'], score_mode
        )
        disturbing_mask = (sample_types == 0) | (sample_types == 1)
        if disturbing_mask.sum() > 0 and 'optimal_threshold' in results:
            disturbing_scores = window_scores[disturbing_mask]
            disturbing_labels = sample_types[disturbing_mask]

            if len(np.unique(disturbing_labels)) > 1:
                disturbing_roc_auc = roc_auc_score(disturbing_labels, disturbing_scores)
                d_predictions = (disturbing_scores > results['optimal_threshold']).astype(int)
                disturbing_precision = precision_score(disturbing_labels, d_predictions, zero_division=0)
                disturbing_recall = recall_score(disturbing_labels, d_predictions, zero_division=0)
                disturbing_f1 = f1_score(disturbing_labels, d_predictions, zero_division=0)

                results['disturbing_roc_auc'] = disturbing_roc_auc
                results['disturbing_precision'] = disturbing_precision
                results['disturbing_recall'] = disturbing_recall
                results['disturbing_f1'] = disturbing_f1
                results['n_pure_normal'] = int((sample_types == 0).sum())
                results['n_disturbing_normal'] = int((sample_types == 1).sum())
                results['n_anomaly'] = int((sample_types == 2).sum())

        return results

    def evaluate_by_score_type(self, score_type: str) -> Dict[str, float]:
        """Evaluate using a single score component at point-level.

        Same point-level logic as evaluate() but with an individual score component.

        Args:
            score_type: One of 'disc', 'teacher_recon', 'student_recon'

        Returns:
            Dict with roc_auc, f1_score, precision, recall, optimal_threshold,
            and pa_K_roc_auc, pa_K_f1 for K in [10, 20, 50, 80]
        """
        cached = self._get_cached_scores()

        if score_type == 'disc':
            patch_scores = cached['patch_disc']
        elif score_type == 'teacher_recon':
            patch_scores = cached['patch_recon']
        elif score_type == 'student_recon':
            patch_scores = cached['patch_student_recon']
        else:
            raise ValueError(f"Unknown score_type: {score_type}")

        if not (self.can_compute_point_level_pa_k and hasattr(self.test_dataset, 'anomaly_regions')):
            return {
                'roc_auc': 0.0, 'precision': 0.0, 'recall': 0.0,
                'f1_score': 0.0, 'optimal_threshold': 0.0,
                'pa_10_f1': 0.0, 'pa_10_roc_auc': 0.0,
                'pa_20_f1': 0.0, 'pa_20_roc_auc': 0.0,
                'pa_50_f1': 0.0, 'pa_50_roc_auc': 0.0,
                'pa_80_f1': 0.0, 'pa_80_roc_auc': 0.0,
            }

        ws_indices = np.array(self.test_dataset.window_start_indices)
        pt_labels = np.array(self.test_dataset.point_labels)
        total_len = len(pt_labels)

        # Aggregate to point-level via mean (cached geometry)
        flat_t, flat_wp, coverage, covered = self._get_aggregation_map()
        point_scores = _aggregate_with_map(patch_scores.ravel(), flat_t, flat_wp, coverage, covered, total_len, method='mean')
        point_scores = np.nan_to_num(point_scores, nan=0.0)

        results = {}

        if len(np.unique(pt_labels)) > 1:
            roc_auc_val = roc_auc_score(pt_labels, point_scores)
            fpr, tpr, thresholds = roc_curve(pt_labels, point_scores)
            optimal_idx = np.argmax(tpr - fpr)
            threshold = thresholds[optimal_idx]

            predictions = (point_scores > threshold).astype(int)
            results = {
                'roc_auc': float(roc_auc_val),
                'precision': float(precision_score(pt_labels, predictions, zero_division=0)),
                'recall': float(recall_score(pt_labels, predictions, zero_division=0)),
                'f1_score': float(f1_score(pt_labels, predictions, zero_division=0)),
                'optimal_threshold': float(threshold),
            }

            # PA%K metrics
            pa_k_values = [10, 20, 50, 80]
            pt_flat_t, pt_flat_si, pt_coverage = self._get_point_score_indices()

            # PA%K AUROC: threshold sweep with voting
            flat = patch_scores.ravel()
            n_thresholds = 100
            thresholds_arr = np.linspace(flat.min() - 0.01, flat.max() + 0.01, n_thresholds)

            point_scores_all = vectorized_voting_for_all_thresholds(
                scores=patch_scores,
                point_indices=(pt_flat_t, pt_flat_si),
                point_coverage=pt_coverage,
                thresholds=thresholds_arr,
            )

            eval_mask = np.ones(total_len, dtype=bool)
            # Compute voted predictions once (K-independent)
            voted_preds = _compute_voted_point_predictions(
                patch_scores, (pt_flat_t, pt_flat_si), pt_coverage, threshold, total_len
            )
            for k in pa_k_values:
                roc_result = _compute_single_pa_k_roc((
                    point_scores_all, pt_labels, self.test_dataset.anomaly_regions,
                    eval_mask, k, None, True
                ))
                _, _, pa_auc, _ = roc_result
                results[f'pa_{k}_roc_auc'] = float(pa_auc)

                # PA%K F1: point-level threshold + voting
                pa_f1 = _compute_pa_k_f1_at_threshold(
                    voted_preds, pt_labels, self.test_dataset.anomaly_regions,
                    k, eval_mask
                )
                results[f'pa_{k}_f1'] = float(pa_f1)
        else:
            results = {
                'roc_auc': 0.0, 'precision': 0.0, 'recall': 0.0,
                'f1_score': 0.0, 'optimal_threshold': 0.0,
                'pa_10_f1': 0.0, 'pa_10_roc_auc': 0.0,
                'pa_20_f1': 0.0, 'pa_20_roc_auc': 0.0,
                'pa_50_f1': 0.0, 'pa_50_roc_auc': 0.0,
                'pa_80_f1': 0.0, 'pa_80_roc_auc': 0.0,
            }

        return results
