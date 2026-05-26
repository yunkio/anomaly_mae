"""
Evaluator for Self-Distilled MAE Anomaly Detection

Includes PA%K (Point-Adjust with K%) evaluation metric for time series anomaly detection.
PA%K is a segment-level adjustment that counts an anomaly segment as detected if at least
K% of its points are flagged as anomalies.

All metrics (including PA%K) use mean-aggregated point-level scores.
Patch scores → mean aggregation to physical timestamps → threshold → PA%K adjustment.
"""

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.amp import autocast
from typing import Dict, Tuple, List, Optional, Any
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, average_precision_score,
    precision_recall_curve
)

from .dataset_sliding import ANOMALY_TYPE_NAMES


# ============================================================================
# F1_T (Time-series F1) - From QuoVadisTAD / TimeSeAD
# Source: https://github.com/ssarfraz/QuoVadisTAD
# Based on: Tatbul et al. (2018), Wagner et al. (2023)
# ============================================================================

def _compute_window_indices(binary_labels: np.ndarray) -> List[Tuple[int, int]]:
    """Compute a list of indices where anomaly windows begin and end.

    Args:
        binary_labels: 1-D array with 1 for anomaly, 0 for normal

    Returns:
        List of (start, end) tuples where end is exclusive
    """
    differences = np.diff(binary_labels, prepend=0)
    indices = np.nonzero(differences)[0]
    if len(indices) % 2 != 0:
        indices = np.append(indices, binary_labels.size)
    return [(indices[i], indices[i + 1]) for i in range(0, len(indices), 2)]


def _constant_bias_fn(inputs: np.ndarray) -> float:
    """Constant bias - average overlap."""
    if inputs.shape[0] == 0:
        return 0.0
    return np.sum(inputs) / inputs.shape[0]


def _improved_cardinality_fn(cardinality: int, gt_length: int) -> float:
    """Recall-consistent cardinality function from TimeSeAD (Wagner et al. 2023).

    Penalizes ground truth windows covered by many predictions.
    """
    if gt_length <= 1:
        return 1.0
    return ((gt_length - 1) / gt_length) ** (cardinality - 1)


def _compute_ts_overlap(
    preds: np.ndarray,
    pred_indices: List[Tuple[int, int]],
    gt_indices: List[Tuple[int, int]],
    alpha: float,
    use_window_weight: bool = False
) -> float:
    """Compute overlap score using two-pointer approach (O(n) complexity).

    From QuoVadisTAD/TimeSeAD implementation.
    """
    n_gt_windows = len(gt_indices)
    n_pred_windows = len(pred_indices)

    if n_gt_windows == 0:
        return 0.0

    total_score = 0.0
    total_gt_points = 0

    i = j = 0
    while i < n_gt_windows and j < n_pred_windows:
        gt_start, gt_end = gt_indices[i]
        window_length = gt_end - gt_start
        total_gt_points += window_length
        i += 1

        cardinality = 0
        while j < n_pred_windows and pred_indices[j][1] <= gt_start:
            j += 1
        while j < n_pred_windows and pred_indices[j][0] < gt_end:
            j += 1
            cardinality += 1

        if cardinality == 0:
            continue

        # The last predicted window that overlaps could also overlap the next window
        j -= 1

        cardinality_multiplier = _improved_cardinality_fn(cardinality, window_length)

        prediction_inside_ground_truth = preds[gt_start:gt_end]
        omega = _constant_bias_fn(prediction_inside_ground_truth)

        weight = window_length if use_window_weight else 1

        # Existence reward (if cardinality > 0 then this is certainly 1)
        total_score += alpha * weight
        # Overlap reward
        total_score += (1 - alpha) * cardinality_multiplier * omega * weight

    # Handle remaining GT windows (no overlapping predictions)
    while i < n_gt_windows:
        gt_start, gt_end = gt_indices[i]
        window_length = gt_end - gt_start
        total_gt_points += window_length
        i += 1

    denom = total_gt_points if use_window_weight else n_gt_windows
    if denom == 0:
        return 0.0

    return total_score / denom


def ts_precision_and_recall(
    labels: np.ndarray,
    predictions: np.ndarray,
    alpha: float = 0,
    weighted_precision: bool = True,
    label_ranges: Optional[List[Tuple[int, int]]] = None
) -> Tuple[float, float]:
    """Compute time-series precision and recall (Tatbul et al. 2018).

    Uses improved cardinality function from TimeSeAD (Wagner et al. 2023).

    Args:
        labels: Ground truth binary labels
        predictions: Binary predictions
        alpha: Weight for existence term (0 = pure overlap)
        weighted_precision: Weight precision by window length
        label_ranges: Pre-computed label ranges (optional, for efficiency)

    Returns:
        (precision, recall) tuple
    """
    has_anomalies = np.any(labels > 0)
    has_predictions = np.any(predictions > 0)

    if not has_predictions and not has_anomalies:
        return 1.0, 1.0
    elif not has_predictions or not has_anomalies:
        return 0.0, 0.0

    if label_ranges is None:
        label_ranges = _compute_window_indices(labels)
    pred_ranges = _compute_window_indices(predictions)

    # Recall: for each GT window, how much is covered by predictions
    recall = _compute_ts_overlap(
        predictions, pred_ranges, label_ranges,
        alpha, use_window_weight=False
    )

    # Precision: for each predicted window, how much overlaps with GT
    precision = _compute_ts_overlap(
        labels, label_ranges, pred_ranges,
        0, use_window_weight=weighted_precision
    )

    return precision, recall


def compute_f1_t_at_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    weighted_precision: bool = True
) -> Tuple[float, float, float]:
    """Compute F1_T (time-series F1) at a given threshold.

    Uses the same threshold as point-level F1 for consistency.

    Args:
        labels: Ground truth binary labels
        scores: Continuous anomaly scores
        threshold: Threshold for binary predictions (from point-level F1)
        weighted_precision: Weight precision by window length

    Returns:
        (f1_t, precision_t, recall_t)
    """
    predictions = (scores > threshold).astype(int)
    prec, rec = ts_precision_and_recall(
        labels, predictions,
        alpha=0,
        weighted_precision=weighted_precision
    )

    if prec + rec > 0:
        f1 = 2 * prec * rec / (prec + rec)
    else:
        f1 = 0.0

    return float(f1), float(prec), float(rec)


def find_f1_optimal_idx(fpr, tpr, labels):
    """Find threshold index that maximizes F1 score on ROC curve.

    Unlike Youden's J (argmax TPR-FPR), this accounts for class imbalance
    by optimizing precision-recall balance directly.
    """
    n_pos = np.sum(labels)
    n_neg = len(labels) - n_pos
    precision = (tpr * n_pos) / (tpr * n_pos + fpr * n_neg + 1e-10)
    recall = tpr
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    return np.argmax(f1)


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
) -> np.ndarray:
    """Aggregate patch scores to point-level using precomputed geometry map."""
    point_scores = np.full(total_length, np.nan)
    flat_s = patch_scores_flat[flat_wp]

    if method == 'mean':
        score_sum = np.bincount(flat_t, weights=flat_s, minlength=total_length)
        point_scores[covered] = score_sum[covered] / point_coverage[covered]
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
    method: str = 'mean',
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate per-patch scores to point-level scores via mean.

    Each window has N patches, and each patch's score is assigned to the timesteps
    that patch covers. Each timestep is covered by ~seq_length windows.

    Args:
        patch_scores: (n_windows, num_patches) per-patch anomaly scores
        window_start_indices: (n_windows,) start index of each window in the time series
        seq_length: Window size (e.g., 100)
        patch_size: Size of each patch (e.g., 10)
        num_patches: Number of patches per window (e.g., 10)
        total_length: Total length of the time series
        method: Aggregation method ('mean', 'median', 'max')

    Returns:
        point_scores: (total_length,) aggregated scores per timestep (NaN for no coverage)
        point_coverage: (total_length,) number of (window, patch) pairs covering each timestep
    """
    n_windows = len(window_start_indices)
    offsets = np.arange(patch_size)
    patch_indices = np.arange(num_patches)

    base_positions = window_start_indices[:, np.newaxis] + patch_indices[np.newaxis, :] * patch_size
    all_t_indices = base_positions[:, :, np.newaxis] + offsets[np.newaxis, np.newaxis, :]
    all_scores = np.broadcast_to(
        patch_scores[:, :, np.newaxis], (n_windows, num_patches, patch_size)
    )

    flat_t = all_t_indices.ravel()
    flat_s = all_scores.ravel()

    valid = (flat_t >= 0) & (flat_t < total_length)
    flat_t = flat_t[valid]
    flat_s = flat_s[valid]

    point_coverage = np.bincount(flat_t, minlength=total_length).astype(int)
    covered = point_coverage > 0

    point_scores = np.full(total_length, np.nan)
    if method == 'mean':
        score_sum = np.bincount(flat_t, weights=flat_s, minlength=total_length)
        point_scores[covered] = score_sum[covered] / point_coverage[covered]
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


def compute_pa_k_metrics_from_mean_scores(
    point_scores: np.ndarray,
    point_labels: np.ndarray,
    anomaly_regions,
    threshold: float,
    k_percent: int,
    eval_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute PA%K metrics using mean-aggregated continuous scores.

    1. Threshold → binary predictions
    2. PA%K segment adjustment
    3. Compute F1, F1_T, Precision, Recall

    Args:
        point_scores: (total_length,) continuous point-level scores
        point_labels: (total_length,) ground truth labels
        anomaly_regions: list of anomaly region objects
        threshold: decision threshold for binary predictions
        k_percent: PA%K threshold percentage (0-100)
        eval_mask: (total_length,) bool mask for evaluation scope

    Returns:
        Dict with pa_f1, pa_precision, pa_recall, pa_f1_t
    """
    total_length = len(point_labels)
    if eval_mask is None:
        eval_mask = np.ones(total_length, dtype=bool)

    predictions = (point_scores > threshold).astype(int)
    adjusted = _apply_pa_k_segment_adjustment(
        predictions, point_labels, anomaly_regions, k_percent, total_length
    )

    # Apply eval mask
    adjusted[~eval_mask] = 0
    masked_labels = point_labels.copy()
    masked_labels[~eval_mask] = 0

    tp = float((adjusted * masked_labels).sum())
    fp = float((adjusted * (eval_mask & (point_labels == 0))).sum())
    fn = float(masked_labels.sum() - tp)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # F1_T (time-series F1) on PA%K-adjusted predictions
    prec_t, rec_t = ts_precision_and_recall(masked_labels, adjusted, alpha=0)
    if prec_t + rec_t > 0:
        f1_t_val = 2 * prec_t * rec_t / (prec_t + rec_t)
    else:
        f1_t_val = 0.0

    return {
        'f1': f1, 'precision': precision, 'recall': recall, 'f1_t': f1_t_val,
    }


def compute_pa_k_roc_prc_from_mean_scores(
    point_scores: np.ndarray,
    point_labels: np.ndarray,
    anomaly_regions,
    k_percent: int,
    eval_mask: Optional[np.ndarray] = None,
    n_thresholds: int = 200,
) -> Dict[str, float]:
    """Compute PA%K ROC-AUC and PRC-AUC by sweeping thresholds on mean scores.

    Args:
        point_scores: (total_length,) continuous point-level scores
        point_labels: (total_length,) ground truth labels
        anomaly_regions: list of anomaly region objects
        k_percent: PA%K threshold percentage
        eval_mask: (total_length,) bool mask
        n_thresholds: number of thresholds to sweep

    Returns:
        Dict with roc_auc, prc_auc
    """
    total_length = len(point_labels)
    if eval_mask is None:
        eval_mask = np.ones(total_length, dtype=bool)

    masked_labels = point_labels.copy()
    masked_labels[~eval_mask] = 0

    n_positive = int(masked_labels.sum())
    n_negative = int((eval_mask & (point_labels == 0)).sum())

    if n_positive == 0 or n_negative == 0:
        return {'roc_auc': 0.5, 'prc_auc': 0.0}

    scores_range = point_scores[eval_mask]
    thresholds = np.linspace(
        scores_range.min() - 0.01, scores_range.max() + 0.01, n_thresholds
    )

    tprs = np.zeros(n_thresholds)
    fprs = np.zeros(n_thresholds)
    precisions = np.zeros(n_thresholds)

    # Pre-extract region boundaries
    region_starts = np.array([r.start for r in anomaly_regions])
    region_ends = np.array([min(r.end, total_length) for r in anomaly_regions])
    valid_regions = region_ends > region_starts
    region_starts = region_starts[valid_regions]
    region_ends = region_ends[valid_regions]
    region_lengths = (region_ends - region_starts).astype(float)
    k_ratio = k_percent / 100.0

    for ti, th in enumerate(thresholds):
        preds = (point_scores > th).astype(float)

        # PA%K segment adjustment using cumsum
        if len(region_starts) > 0:
            cumsum_preds = np.concatenate([[0], np.cumsum(preds)])
            region_sums = cumsum_preds[region_ends] - cumsum_preds[region_starts]
            detection_ratios = region_sums / region_lengths
            for r_idx in range(len(region_starts)):
                s, e = region_starts[r_idx], region_ends[r_idx]
                preds[s:e] = 1.0 if detection_ratios[r_idx] >= k_ratio else 0.0

        preds[~eval_mask] = 0
        tp = float((preds * masked_labels).sum())
        fp = float((preds * (eval_mask & (point_labels == 0))).sum())

        tprs[ti] = tp / n_positive
        fprs[ti] = fp / n_negative
        precisions[ti] = tp / (tp + fp) if (tp + fp) > 0 else 1.0

    # ROC AUC
    sorted_idx = np.argsort(fprs)
    roc_auc = float(np.trapz(tprs[sorted_idx], fprs[sorted_idx]))

    # PRC AUC — step-AP (sklearn-style average_precision_score)
    # Sort by threshold descending = recall ascending; prepend (rec=0, prec=1) endpoint;
    # AP = Σ (R_n − R_{n−1}) · P_n. trapz on PR curves is biased (overestimates).
    th_order = np.argsort(-thresholds)
    rec_step = np.concatenate([[0.0], tprs[th_order]])
    prec_step = np.concatenate([[1.0], precisions[th_order]])
    prc_auc = float(np.sum(np.diff(rec_step) * prec_step[1:]))
    prc_auc = max(0.0, min(1.0, prc_auc))

    return {'roc_auc': roc_auc, 'prc_auc': prc_auc}


def _apply_pa_k_segment_adjustment(
    predictions: np.ndarray,
    labels: np.ndarray,
    anomaly_regions,
    k_percent: int,
    total_length: int,
) -> np.ndarray:
    """Apply PA%K segment adjustment to binary predictions.

    If >= K% of an anomaly segment is detected, the entire segment is marked detected.
    Otherwise, the entire segment is marked undetected.
    """
    adjusted = predictions.copy()
    k_ratio = k_percent / 100.0

    region_starts = np.array([r.start for r in anomaly_regions])
    region_ends = np.array([min(r.end, total_length) for r in anomaly_regions])
    valid = region_ends > region_starts

    if valid.any():
        starts = region_starts[valid]
        ends = region_ends[valid]
        lengths = (ends - starts).astype(float)
        cumsum = np.concatenate([[0], np.cumsum(predictions)])
        sums = cumsum[ends] - cumsum[starts]
        ratios = sums / lengths

        for i in range(len(starts)):
            adjusted[starts[i]:ends[i]] = 1.0 if ratios[i] >= k_ratio else 0.0

    return adjusted


def _compute_threshold_dependent(
    s: np.ndarray, y: np.ndarray, pred: np.ndarray,
) -> Dict[str, float]:
    """Affiliation P/R/F1 + Range-based F1 at a given binarization (pred).

    Used by compute_extra_metrics (at optimal F1 threshold) and
    compute_ar_threshold_metric_set (at anomaly-ratio threshold).

    Provenance:
    - Affiliation: Huet et al. KDD 2022 (ahstat/affiliation-metrics-py).
    - R-based F1: Tatbul et al. NeurIPS 2018, official wrapper TSB-AD
      basic_metricor.metric_RF1 in fixed-threshold mode (preds=pred).
    """
    from affiliation.metrics import pr_from_events as _aff_pr
    from affiliation.generics import convert_vector_to_events as _aff_v2e
    from TSB_AD.evaluation.basic_metrics import basic_metricor as _rf1_grader_cls

    out = {
        'affiliation_precision': 0.0, 'affiliation_recall': 0.0,
        'affiliation_f1': 0.0,
        'r_based_f1': 0.0,
    }
    try:
        evt_pred = _aff_v2e(pred.tolist())
        evt_gt   = _aff_v2e(y.tolist())
        if len(evt_pred) > 0 and len(evt_gt) > 0:
            ar = _aff_pr(evt_pred, evt_gt, (0, len(s)))
            p  = float(ar.get('precision', 0.0))
            r_ = float(ar.get('recall', 0.0))
            out['affiliation_precision'] = p
            out['affiliation_recall']    = r_
            out['affiliation_f1'] = (2.0 * p * r_ / (p + r_)) if (p + r_) > 0 else 0.0
    except Exception as e:
        print(f"  [Aff warn] {type(e).__name__}: {e}")

    try:
        rf1 = _rf1_grader_cls().metric_RF1(y, s, preds=pred)
        out['r_based_f1'] = float(rf1)
    except Exception as e:
        print(f"  [RF1 warn] {type(e).__name__}: {e}")

    return out


def compute_extra_metrics(
    point_scores: np.ndarray,
    point_labels: np.ndarray,
    threshold: float,
    sliding_window: int = 100,
) -> Dict[str, float]:
    """VUS-PR, VUS-ROC (threshold-free) + Affiliation-F1, R-based F1 (at given threshold).

    VUS-PR / VUS-ROC: Paparrizos et al., "Volume Under the Surface", VLDB 2022.
        Official: https://github.com/TheDatumOrg/VUS (pip install vus)
    Affiliation-F1: Huet, Navarro, Rossi, "Local Evaluation of Time Series
        Anomaly Detection Algorithms", KDD 2022.
        Official: https://github.com/ahstat/affiliation-metrics-py
    R-based F1: Tatbul, Lee, Zdonik, Alam, Gottschlich, "Precision and Recall
        for Time Series", NeurIPS 2018.
        Official wrapper: TSB-AD's `basic_metricor.metric_RF1`
        (https://github.com/TheDatumOrg/TSB-AD).

    Args:
        point_scores: 1-D anomaly score per timestep.
        point_labels: 1-D binary ground truth (1 = anomaly).
        threshold: cutoff for converting scores -> binary predictions (used by
                   Affiliation + R-based F1 — VUS is threshold-free).
        sliding_window: VUS buffer length (typical 100 for TSAD benchmarks).

    Returns:
        dict with keys vus_roc, vus_pr, affiliation_precision, affiliation_recall,
        affiliation_f1, r_based_f1. Returns zeros for degenerate input.
    """
    from vus.metrics import get_metrics as _vus_get

    out = {
        'vus_roc': 0.0, 'vus_pr': 0.0,
        'affiliation_precision': 0.0, 'affiliation_recall': 0.0,
        'affiliation_f1': 0.0,
        'r_based_f1': 0.0,
    }

    s = np.asarray(point_scores, dtype=float).ravel()
    y = np.asarray(point_labels, dtype=int).ravel()
    if len(s) == 0 or len(s) != len(y):
        return out
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == len(y):
        return out  # degenerate

    pred = (s >= float(threshold)).astype(int)

    # VUS — official example normalizes scores to [0,1] via MinMaxScaler.
    smin, smax = float(s.min()), float(s.max())
    s_norm = (s - smin) / (smax - smin + 1e-12)
    try:
        vr = _vus_get(s_norm, y, metric='all', slidingWindow=sliding_window)
        out['vus_roc'] = float(vr.get('VUS_ROC', 0.0))
        out['vus_pr']  = float(vr.get('VUS_PR', 0.0))
    except Exception as e:
        print(f"  [VUS warn] {type(e).__name__}: {e}")

    # Aff + R-F1 at given threshold
    out.update(_compute_threshold_dependent(s, y, pred))
    return out


def compute_ar_threshold_metric_set(
    point_scores: np.ndarray,
    point_labels: np.ndarray,
    weighted_precision: bool = True,
) -> Dict[str, float]:
    """Recompute threshold-dependent metrics at anomaly-ratio threshold.

    AR threshold = (1 - anomaly_ratio)-th quantile of scores.
    Use case: when test data anomaly ratio is known (~5%), pick top-5% scores
    as positive predictions — independent of optimal-F1 threshold selection
    (which leaks ground truth into threshold choice).

    Saved keys (all suffixed `_ar`, plus anomaly_ratio + anomaly_ratio_threshold):
        f1_ar, precision_ar, recall_ar          (sklearn point-strict)
        f1_t_ar, precision_t_ar, recall_t_ar    (TS-window F1 via compute_f1_t_at_threshold)
        affiliation_precision_ar, _recall_ar, _f1_ar
        r_based_f1_ar
    PA%K family, prc_auc, vus_*, disc_snr are NOT recomputed (threshold-free
    or K-integrated → independent of single-threshold choice).
    """
    out = {
        'anomaly_ratio': 0.0,
        'anomaly_ratio_threshold': 0.0,
        'f1_ar': 0.0, 'precision_ar': 0.0, 'recall_ar': 0.0,
        'f1_t_ar': 0.0, 'precision_t_ar': 0.0, 'recall_t_ar': 0.0,
        'affiliation_precision_ar': 0.0, 'affiliation_recall_ar': 0.0,
        'affiliation_f1_ar': 0.0,
        'r_based_f1_ar': 0.0,
    }
    s = np.asarray(point_scores, dtype=float).ravel()
    y = np.asarray(point_labels, dtype=int).ravel()
    if len(s) == 0 or len(s) != len(y):
        return out
    ar = float(y.mean())
    out['anomaly_ratio'] = ar
    if ar <= 0 or ar >= 1:
        return out  # degenerate

    ar_th = float(np.quantile(s, 1.0 - ar))
    out['anomaly_ratio_threshold'] = ar_th

    pred = (s > ar_th).astype(int)

    # point-strict (sklearn)
    out['f1_ar']        = float(f1_score(y, pred, zero_division=0))
    out['precision_ar'] = float(precision_score(y, pred, zero_division=0))
    out['recall_ar']    = float(recall_score(y, pred, zero_division=0))

    # time-series F1 via existing helper
    f1_t_ar, prec_t_ar, rec_t_ar = compute_f1_t_at_threshold(
        y, s, ar_th, weighted_precision=weighted_precision
    )
    out['f1_t_ar']        = float(f1_t_ar)
    out['precision_t_ar'] = float(prec_t_ar)
    out['recall_t_ar']    = float(rec_t_ar)

    # Aff + R-F1 at AR threshold
    td = _compute_threshold_dependent(s, y, pred)
    out['affiliation_precision_ar'] = td['affiliation_precision']
    out['affiliation_recall_ar']    = td['affiliation_recall']
    out['affiliation_f1_ar']        = td['affiliation_f1']
    out['r_based_f1_ar']            = td['r_based_f1']
    return out


EXTRA_METRIC_KEYS = (
    'vus_roc', 'vus_pr',
    'affiliation_precision', 'affiliation_recall', 'affiliation_f1',
    'r_based_f1',
    # Anomaly-ratio threshold variants (saved to epoch_metrics.json only).
    'anomaly_ratio', 'anomaly_ratio_threshold',
    'f1_ar', 'precision_ar', 'recall_ar',
    'f1_t_ar', 'precision_t_ar', 'recall_t_ar',
    'affiliation_precision_ar', 'affiliation_recall_ar', 'affiliation_f1_ar',
    'r_based_f1_ar',
)

# K values used by all PA%K metric loops in this module.
PA_K_VALUES = list(range(0, 101, 5))

# All threshold-base scalar keys (roc/prc/F1/F1_T) — for zero-fill in degenerate case.
_BASE_SCALAR_KEYS = (
    'roc_auc', 'prc_auc', 'precision', 'recall', 'f1_score', 'optimal_threshold',
    'f1_t', 'precision_t', 'recall_t',
)

# All PA%K-integrated keys (matching compute_pa_k_auc scalar output).
PAK_AUC_KEYS = (
    'pak_auc_prc_auc', 'pak_auc_roc_auc', 'pak_auc_f1',
    'pak_auc_f1_t', 'pak_auc_precision', 'pak_auc_recall',
    'pak_auc_f1_raw', 'pak_auc_f1_t_raw',
    'pak_auc_precision_raw', 'pak_auc_recall_raw',
)


def _zero_metric_set() -> Dict[str, float]:
    """All-zeros dict matching compute_full_metric_set output schema.

    Used as fallback when input is degenerate (single-class labels, empty, etc.).
    """
    out = {k: 0.0 for k in _BASE_SCALAR_KEYS}
    for k in PA_K_VALUES:
        for sub in ('f1', 'precision', 'recall', 'roc_auc', 'prc_auc'):
            out[f'pa_{k}_{sub}'] = 0.0
    for k in PAK_AUC_KEYS:
        out[k] = 0.0
    for k in EXTRA_METRIC_KEYS:
        out[k] = 0.0
    return out


def compute_full_metric_set(
    point_scores: np.ndarray,
    point_labels: np.ndarray,
    anomaly_regions,
    eval_mask: Optional[np.ndarray] = None,
    n_thresholds: int = 200,
    sliding_window: int = 100,
) -> Dict[str, Any]:
    """SINGLE SOURCE OF TRUTH for per-epoch metric computation.

    Both pipelines call this:
    - `mae_anomaly/evaluator.py` (MAE training, via Evaluator's eval methods)
    - `comparison/baseline_common.py` (baseline training)

    To add a new metric across both pipelines: edit ONLY this function.

    Computes:
        - Threshold-based core: roc_auc, prc_auc, precision, recall, f1_score,
          optimal_threshold, f1_t, precision_t, recall_t. Computed on
          `point_scores[eval_mask]` + `point_labels[eval_mask]`.
        - Per-K PA%K (k=0..100 step 5): pa_{k}_{f1, precision, recall, roc_auc, prc_auc}
        - PA%K AUC integrated: pak_auc_{f1, prc_auc, roc_auc, f1_t, precision, recall,
          *_raw} + diagnostic `_per_k_*` arrays.
        - VUS-PR, VUS-ROC (Paparrizos VLDB 2022, threshold-free).
        - Affiliation P/R/F1 (Huet KDD 2022, at optimal threshold).
        - R-based F1 (Tatbul NeurIPS 2018, at optimal threshold).
        - Anomaly-ratio threshold variants (`_ar` suffix): same threshold-based
          metrics recomputed at `(1 - anomaly_rate)`-quantile threshold.

    Args:
        point_scores: 1-D anomaly score per timestep (full array).
        point_labels: 1-D binary ground truth (1=anomaly). Same length as point_scores.
        anomaly_regions: list of region objects with `.start`, `.end` attributes
            (already filtered if needed — e.g., SWaT excl22 removes Region 22).
        eval_mask: optional boolean mask (True = include). If None, all True.
            Threshold-based core metrics use `point_*[eval_mask]`; PA%K functions
            accept eval_mask directly; VUS and AR use full `point_*` (mask not
            respected by VUS algorithm, AR computed on full for ratio consistency).
        n_thresholds: PA%K AUC threshold sweep count (default 200, matches MAE).
        sliding_window: VUS buffer length (default 100).

    Returns:
        dict with ~133 scalar keys + 4 `_per_k_*` diagnostic lists. Underscore-
        prefixed keys may be filtered by callers that want JSON-clean output.

    Backwards compat: keys/values match prior MAE inline implementation
    (3 sites in this module) byte-for-byte except all values are float-cast
    for JSON safety. Verified via comparison against existing
    epoch_metrics.json from exp271 SWaT.
    """
    if eval_mask is None:
        eval_mask = np.ones(len(point_labels), dtype=bool)

    # ---- Degenerate case detection ----
    base_scores = point_scores[eval_mask]
    base_labels = point_labels[eval_mask]
    if len(base_labels) == 0 or len(np.unique(base_labels)) <= 1:
        return _zero_metric_set()

    # ---- Threshold-based core ----
    roc_auc = roc_auc_score(base_labels, base_scores)
    prc_auc = average_precision_score(base_labels, base_scores)
    fpr, tpr, thresholds = roc_curve(base_labels, base_scores)
    optimal_idx = find_f1_optimal_idx(fpr, tpr, base_labels)
    threshold = thresholds[optimal_idx]
    predictions = (base_scores > threshold).astype(int)

    f1_t, precision_t, recall_t = compute_f1_t_at_threshold(
        base_labels, base_scores, threshold
    )

    results: Dict[str, Any] = {
        'roc_auc': float(roc_auc),
        'prc_auc': float(prc_auc),
        'precision': float(precision_score(base_labels, predictions, zero_division=0)),
        'recall':    float(recall_score(base_labels, predictions, zero_division=0)),
        'f1_score':  float(f1_score(base_labels, predictions, zero_division=0)),
        'optimal_threshold': float(threshold),
        'f1_t':        float(f1_t),
        'precision_t': float(precision_t),
        'recall_t':    float(recall_t),
    }

    # ---- Per-K PA%K (granular) ----
    for k in PA_K_VALUES:
        pa_metrics = compute_pa_k_metrics_from_mean_scores(
            point_scores, point_labels, anomaly_regions, threshold, k, eval_mask
        )
        results[f'pa_{k}_f1']        = float(pa_metrics['f1'])
        results[f'pa_{k}_precision'] = float(pa_metrics['precision'])
        results[f'pa_{k}_recall']    = float(pa_metrics['recall'])
        pa_roc_prc = compute_pa_k_roc_prc_from_mean_scores(
            point_scores, point_labels, anomaly_regions, k, eval_mask
        )
        results[f'pa_{k}_roc_auc'] = float(pa_roc_prc['roc_auc'])
        results[f'pa_{k}_prc_auc'] = float(pa_roc_prc['prc_auc'])

    # ---- PA%K AUC integrated (+ _per_k_* diagnostic lists) ----
    pak_auc = compute_pa_k_auc(
        point_scores, point_labels, anomaly_regions, threshold, eval_mask, n_thresholds
    )
    results.update(pak_auc)

    # ---- VUS-PR/ROC + Affiliation-F1 + R-based F1 (at optimal threshold) ----
    # NOTE: VUS is threshold-free and operates on FULL point_scores/labels
    # (sliding-window VUS algorithm doesn't naturally support arbitrary masks).
    results.update(compute_extra_metrics(point_scores, point_labels, threshold, sliding_window))

    # ---- Anomaly-ratio threshold variants ----
    # NOTE: anomaly_ratio = full point_labels.mean(), preserving consistency
    # across pipelines. If you need mask-aware AR, derive from point_labels[eval_mask] externally.
    results.update(compute_ar_threshold_metric_set(point_scores, point_labels))

    return results


def compute_pa_k_auc(
    point_scores: np.ndarray,
    point_labels: np.ndarray,
    anomaly_regions,
    threshold: float,
    eval_mask: Optional[np.ndarray] = None,
    n_thresholds: int = 200,
) -> Dict[str, float]:
    """Compute PA%K AUC: sweep K=0,1,...,100 and integrate each metric curve.

    For each K in [0, 100], computes metrics in two modes:
      - **best** (best_f1_w_pa): Per-K threshold re-optimization via PRC threshold
        sweep. At each K, finds the threshold that maximizes F1 AFTER PA%K
        adjustment. Following Kim et al. (AAAI 2022, "Towards a Rigorous
        Evaluation of Time-series Anomaly Detection"), tadpak implementation.
      - **raw** (raw_f1_w_pa): Fixed threshold (pre-PA F1-optimal) applied across
        all K values. This is the legacy behavior for comparison.
      - PRC-AUC, ROC-AUC: threshold sweep (unchanged, already correct)
    Then integrates each metric over K using the trapezoidal rule,
    normalized to [0, 1] range.

    Args:
        point_scores: (total_length,) continuous point-level scores
        point_labels: (total_length,) ground truth labels
        anomaly_regions: list of anomaly region objects
        threshold: decision threshold for raw F1/Precision/Recall (pre-PA optimal)
        eval_mask: (total_length,) bool mask for evaluation scope
        n_thresholds: number of thresholds for ROC/PRC/F1 sweep

    Returns:
        Dict with pak_auc_{prc_auc, roc_auc, f1, f1_t, precision, recall}
        and pak_auc_{f1_raw, f1_t_raw, precision_raw, recall_raw}
    """
    # ===========================================================================
    # Vectorized implementation (~30x faster than the original loop-based code).
    # Verified bit-identical to the previous (fixed) loop-based version within
    # 1e-6 abs tolerance on synthetic + SMD + WaDi + SWaT/excl22 (see
    # temp/verify_vectorize_v3.py). Peak memory per call ≈ T × N bool array.
    # ===========================================================================
    import gc as _gc
    total_length = len(point_labels)
    if eval_mask is None:
        eval_mask = np.ones(total_length, dtype=bool)

    k_values = np.arange(0, 101)
    n_k = len(k_values)
    k_ratios = k_values.astype(np.float64) / 100.0

    # Regions (small)
    region_starts = np.array([r.start for r in anomaly_regions], dtype=np.int64)
    region_ends = np.array([min(r.end, total_length) for r in anomaly_regions], dtype=np.int64)
    valid = region_ends > region_starts
    v_starts = region_starts[valid]
    v_ends = region_ends[valid]
    v_lengths = (v_ends - v_starts).astype(np.float64)
    n_regions = int(len(v_starts))

    masked_labels = point_labels.copy().astype(np.int64)
    masked_labels[~eval_mask] = 0
    n_positive = int(masked_labels.sum())
    normal_eval_mask = eval_mask & (point_labels == 0)
    n_negative = int(normal_eval_mask.sum())
    has_both_classes = n_positive > 0 and n_negative > 0

    scores_range = point_scores[eval_mask]
    thresh_arr = np.linspace(scores_range.min() - 0.01, scores_range.max() + 0.01, n_thresholds)

    # (T, N) bool — largest array. e.g. SWaT (200×224K) = 45 MB.
    preds_raw_TN = (point_scores[None, :] > thresh_arr[:, None])

    # Per-region pred count via direct slice sum (no big cumsum buffer)
    if n_regions > 0:
        region_preds_TR = np.empty((n_thresholds, n_regions), dtype=np.int64)
        for _ri in range(n_regions):
            region_preds_TR[:, _ri] = preds_raw_TN[:, v_starts[_ri]:v_ends[_ri]].sum(axis=1)
        dr_TR = region_preds_TR.astype(np.float64) / v_lengths[None, :]
        del region_preds_TR
    else:
        dr_TR = np.zeros((n_thresholds, 0), dtype=np.float64)

    label_in_eval = ((point_labels == 1) & eval_mask)
    normal_in_eval = ((point_labels == 0) & eval_mask)
    if n_regions > 0:
        region_label_R = np.empty(n_regions, dtype=np.int64)
        region_normal_R = np.empty(n_regions, dtype=np.int64)
        for _ri in range(n_regions):
            region_label_R[_ri] = int(label_in_eval[v_starts[_ri]:v_ends[_ri]].sum())
            region_normal_R[_ri] = int(normal_in_eval[v_starts[_ri]:v_ends[_ri]].sum())
    else:
        region_label_R = np.zeros(0, dtype=np.int64)
        region_normal_R = np.zeros(0, dtype=np.int64)

    region_mask = np.zeros(total_length, dtype=bool)
    for _ri in range(n_regions):
        region_mask[v_starts[_ri]:v_ends[_ri]] = True
    outside_label = (~region_mask) & label_in_eval
    outside_normal = (~region_mask) & normal_in_eval
    del region_mask, label_in_eval, normal_in_eval

    TP_outside_T = (preds_raw_TN & outside_label[None, :]).sum(axis=1).astype(np.int64)
    FP_outside_T = (preds_raw_TN & outside_normal[None, :]).sum(axis=1).astype(np.int64)

    if n_regions > 0:
        detected_TKR = (dr_TR[:, None, :] >= k_ratios[None, :, None])
        det_int = detected_TKR.astype(np.int64)
        TP_inside_TK = det_int @ region_label_R
        FP_inside_TK = det_int @ region_normal_R
        del det_int
    else:
        detected_TKR = np.zeros((n_thresholds, n_k, 0), dtype=bool)
        TP_inside_TK = np.zeros((n_thresholds, n_k), dtype=np.int64)
        FP_inside_TK = np.zeros((n_thresholds, n_k), dtype=np.int64)

    TP_TK = TP_outside_T[:, None] + TP_inside_TK
    FP_TK = FP_outside_T[:, None] + FP_inside_TK
    del TP_inside_TK, FP_inside_TK

    TPR_TK = TP_TK.astype(np.float64) / max(n_positive, 1)
    FPR_TK = FP_TK.astype(np.float64) / max(n_negative, 1)
    denom_pr = (TP_TK + FP_TK).astype(np.float64)
    PREC_TK = np.where(denom_pr > 0, TP_TK / np.maximum(denom_pr, 1.0), 1.0)
    REC_TK = TPR_TK
    denom_f1 = PREC_TK + REC_TK
    F1_TK = np.where(denom_f1 > 0, 2.0 * PREC_TK * REC_TK / np.maximum(denom_f1, 1e-12), 0.0)
    del denom_pr, denom_f1

    # ---- RAW mode (fixed threshold from input) ----
    base_preds = (point_scores > threshold).astype(np.int64)
    if n_regions > 0:
        base_region_preds_R = np.empty(n_regions, dtype=np.int64)
        for _ri in range(n_regions):
            base_region_preds_R[_ri] = int(base_preds[v_starts[_ri]:v_ends[_ri]].sum())
        base_dr_R = base_region_preds_R.astype(np.float64) / v_lengths
        base_detected_KR = (base_dr_R[None, :] >= k_ratios[:, None])
        base_TP_K_inside = base_detected_KR.astype(np.int64) @ region_label_R
        base_FP_K_inside = base_detected_KR.astype(np.int64) @ region_normal_R
    else:
        base_detected_KR = np.zeros((n_k, 0), dtype=bool)
        base_TP_K_inside = np.zeros(n_k, dtype=np.int64)
        base_FP_K_inside = np.zeros(n_k, dtype=np.int64)
    base_TP_outside = int((base_preds * outside_label).sum())
    base_FP_outside = int((base_preds * outside_normal).sum())
    del outside_label, outside_normal

    f1s_raw = np.zeros(n_k)
    f1_ts_raw = np.zeros(n_k)
    precisions_raw = np.zeros(n_k)
    recalls_raw = np.zeros(n_k)
    for ki in range(n_k):
        tp = float(base_TP_outside + base_TP_K_inside[ki])
        fp = float(base_FP_outside + base_FP_K_inside[ki])
        fn = float(max(n_positive - tp, 0))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precisions_raw[ki] = prec
        recalls_raw[ki] = rec
        f1s_raw[ki] = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        adjusted = base_preds.astype(np.float64).copy()
        if n_regions > 0:
            for _ri in range(n_regions):
                adjusted[v_starts[_ri]:v_ends[_ri]] = 1.0 if base_detected_KR[ki, _ri] else 0.0
        adjusted[~eval_mask] = 0
        prec_t, rec_t = ts_precision_and_recall(masked_labels, adjusted.astype(int), alpha=0)
        f1_ts_raw[ki] = 2 * prec_t * rec_t / (prec_t + rec_t) if (prec_t + rec_t) > 0 else 0.0
        del adjusted

    # ---- BEST mode (per-K threshold sweep) ----
    prc_aucs = np.zeros(n_k)
    roc_aucs = np.zeros(n_k)
    f1s_best = np.zeros(n_k)
    f1_ts_best = np.zeros(n_k)
    precisions_best = np.zeros(n_k)
    recalls_best = np.zeros(n_k)

    if not has_both_classes:
        prc_aucs[:] = 0.0
        roc_aucs[:] = 0.5
        f1s_best[:] = f1s_raw
        f1_ts_best[:] = f1_ts_raw
        precisions_best[:] = precisions_raw
        recalls_best[:] = recalls_raw
    else:
        best_ti_K = np.argmax(F1_TK, axis=0)
        f1s_best = F1_TK[best_ti_K, np.arange(n_k)]
        precisions_best = PREC_TK[best_ti_K, np.arange(n_k)]
        recalls_best = REC_TK[best_ti_K, np.arange(n_k)]
        for ki in range(n_k):
            ti = int(best_ti_K[ki])
            best_preds_ki = preds_raw_TN[ti].astype(np.float64).copy()
            if n_regions > 0:
                for _ri in range(n_regions):
                    best_preds_ki[v_starts[_ri]:v_ends[_ri]] = 1.0 if detected_TKR[ti, ki, _ri] else 0.0
            best_preds_ki[~eval_mask] = 0
            prec_t_b, rec_t_b = ts_precision_and_recall(masked_labels, best_preds_ki.astype(int), alpha=0)
            f1_ts_best[ki] = 2 * prec_t_b * rec_t_b / (prec_t_b + rec_t_b) if (prec_t_b + rec_t_b) > 0 else 0.0
            del best_preds_ki

        # PRC AUC via step-AP per K, ROC AUC via trapz per K
        th_order = np.argsort(-thresh_arr)
        for ki in range(n_k):
            rec_step = np.concatenate([[0.0], REC_TK[th_order, ki]])
            prec_step = np.concatenate([[1.0], PREC_TK[th_order, ki]])
            ap = float(np.sum(np.diff(rec_step) * prec_step[1:]))
            prc_aucs[ki] = max(0.0, min(1.0, ap))
            si = np.argsort(FPR_TK[:, ki])
            roc_aucs[ki] = float(np.trapz(TPR_TK[si, ki], FPR_TK[si, ki]))

    # Free large arrays explicitly before building return dict
    del preds_raw_TN, F1_TK, PREC_TK, REC_TK, FPR_TK, TPR_TK, TP_TK, FP_TK
    del dr_TR, detected_TKR, masked_labels, normal_eval_mask
    _gc.collect()

    # Integrate over K with trapezoidal rule, normalized by K range (100).
    # Also expose per-K arrays so callers (e.g., bulk-recompute scripts) can
    # extract per-K metrics without re-running this whole function 21x.
    return {
        'pak_auc_prc_auc': float(np.trapz(prc_aucs, k_values) / 100.0),
        'pak_auc_roc_auc': float(np.trapz(roc_aucs, k_values) / 100.0),
        # best_f1_w_pa: per-K re-optimized threshold (primary metric)
        'pak_auc_f1': float(np.trapz(f1s_best, k_values) / 100.0),
        'pak_auc_f1_t': float(np.trapz(f1_ts_best, k_values) / 100.0),
        'pak_auc_precision': float(np.trapz(precisions_best, k_values) / 100.0),
        'pak_auc_recall': float(np.trapz(recalls_best, k_values) / 100.0),
        # raw_f1_w_pa: fixed threshold (legacy, for comparison)
        'pak_auc_f1_raw': float(np.trapz(f1s_raw, k_values) / 100.0),
        'pak_auc_f1_t_raw': float(np.trapz(f1_ts_raw, k_values) / 100.0),
        'pak_auc_precision_raw': float(np.trapz(precisions_raw, k_values) / 100.0),
        'pak_auc_recall_raw': float(np.trapz(recalls_raw, k_values) / 100.0),
        # Per-K arrays (indexed 0..100). Exposed for cheap downstream extraction.
        # Returned as Python lists so the dict stays JSON-serializable when
        # downstream callers dump epoch_metrics.json (avoids numpy serialize fail).
        '_per_k_prc_auc': prc_aucs.tolist(),
        '_per_k_roc_auc': roc_aucs.tolist(),
        '_per_k_f1': f1s_best.tolist(),
        '_per_k_values': k_values.tolist(),
    }


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
        self.use_amp = config.use_amp and torch.cuda.is_available()
        amp_dtype_name = getattr(config, 'amp_dtype', 'fp16').lower()
        if amp_dtype_name == 'fp16':
            self.amp_dtype = torch.float16
        elif amp_dtype_name == 'bf16':
            self.amp_dtype = torch.bfloat16
            # Match trainer's safety: bf16 requires CUDA capability >= 8.0 (Ampere/Ada/Hopper).
            if self.use_amp:
                _cc = torch.cuda.get_device_capability()
                if _cc < (8, 0):
                    raise RuntimeError(
                        f"amp_dtype='bf16' requires CUDA capability >= 8.0 (Ampere/Ada/Hopper); "
                        f"got sm_{_cc[0]}{_cc[1]}. Use amp_dtype='fp16' on older GPUs."
                    )
        else:
            raise ValueError(
                f"config.amp_dtype must be 'fp16' or 'bf16', got {amp_dtype_name!r}"
            )

        # Point-level PA%K requires test_dataset with specific attributes
        self.can_compute_point_level_pa_k = (
            test_dataset is not None and
            hasattr(test_dataset, 'point_labels') and
            hasattr(test_dataset, 'window_start_indices')
        )

        # Cache for raw scores to avoid redundant forward passes
        self._cache = {}
        # Per-feature discrepancy (populated by _compute_patch_scores_all_patches)
        self.disc_per_feature = None  # (n_windows, F) after inference
        # FM patches (populated by _compute_patch_scores_all_patches when use_feature_matching=True)
        self.fm_patches = None  # (n_windows, num_patches) or None

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
        fm_patches: Optional[np.ndarray] = None,
    ):
        """Populate the evaluator cache with pre-computed patch scores.

        Use this to avoid a redundant GPU forward pass when patch scores
        have already been computed (e.g., by _compute_patch_scores_all_patches).

        Args:
            recon_patches: (n_windows, num_patches) teacher reconstruction scores
            disc_patches: (n_windows, num_patches) discrepancy scores
            student_recon_patches: (n_windows, num_patches) student reconstruction scores
            labels: (n_windows,) window-level binary labels
            sample_types: (n_windows,) window-level sample type indicators
            anomaly_types: (n_windows,) window-level anomaly type indicators
            fm_patches: optional (n_windows, num_patches) feature-matching distance scores.
                Must be supplied when use_feature_matching=True so adaptive scoring
                in evaluate() includes FM contribution (mirrors npz save path).
        """
        self._cache['raw_scores'] = self._build_cache_dict(
            recon_patches, disc_patches, student_recon_patches,
            labels, sample_types, anomaly_types,
        )
        if fm_patches is not None:
            self.fm_patches = fm_patches

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

    def _get_cached_fm_scores(self) -> Optional[np.ndarray]:
        """Get cached FM window scores (mean of patch-level FM distances)."""
        if not getattr(self.config, 'use_feature_matching', False):
            return None
        # Ensure patch scores are computed (triggers _compute_patch_scores_all_patches)
        self._get_cached_scores()
        if self.fm_patches is not None:
            return self.fm_patches.mean(axis=1)  # (n_windows, num_patches) → (n_windows,)
        return None

    def _apply_scoring_formula(self, recon: np.ndarray, disc: np.ndarray, scoring_mode: str,
                               fm: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply scoring formula to raw recon/disc/fm scores

        Args:
            recon: Raw reconstruction scores (teacher)
            disc: Raw discrepancy scores (output-level)
            scoring_mode: 'default', 'adaptive', or 'ratio_weighted'
            fm: Raw feature matching distance scores (hidden-level), optional

        Returns:
            Combined anomaly scores

        Scoring formula (adaptive mode):
            scaled_disc = disc * (recon.mean() / disc.mean())
            scaled_fm   = fm   * (recon.mean() / fm.mean())
            student_error = (w_disc * scaled_disc + w_fm * scaled_fm) / (w_disc + w_fm)
            score = recon + student_error
        """
        if scoring_mode == 'adaptive':
            recon_mean = recon.mean() + 1e-4

            # Resolve eval weights (-1 = use training weights)
            w_disc = getattr(self.config, 'eval_disc_weight', -1.0)
            w_fm = getattr(self.config, 'eval_fm_weight', -1.0)
            if w_disc < 0:
                w_disc = 1.0
            if w_fm < 0:
                w_fm = getattr(self.config, 'fm_loss_weight', 1.0)
            # OD disabled → exclude untrained disc from scoring
            if not getattr(self.config, 'use_output_discrepancy', True):
                w_disc = 0.0

            # Scale disc to recon's scale
            scaled_disc = disc * (recon_mean / (disc.mean() + 1e-4))

            if fm is not None and getattr(self.config, 'use_feature_matching', False):
                # Scale FM to recon's scale
                scaled_fm = fm * (recon_mean / (fm.mean() + 1e-4))
                if w_disc + w_fm > 0:
                    student_error = (w_disc * scaled_disc + w_fm * scaled_fm) / (w_disc + w_fm)
                else:
                    student_error = np.zeros_like(recon)
            elif w_disc > 0:
                student_error = scaled_disc
            else:
                # Both OD and FM disabled — teacher recon only
                student_error = np.zeros_like(recon)

            return recon + student_error
        elif scoring_mode == 'ratio_weighted':
            disc_median = np.median(disc) + 1e-4
            return recon * (1 + disc / disc_median)
        else:  # default
            return recon + self.config.lambda_disc * disc

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

    def _compute_patch_scores_all_patches(self, collect_detail=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute per-patch recon/disc/student_recon scores by masking each patch one at a time

        Optimized: All patches processed in a single forward pass by expanding batch dimension.
        Returns per-patch scores to be used directly for point-level aggregation.

        When collect_detail=True, also collects per-timestep reconstruction data
        (feature 0 only) for visualization, stored in self.detail_results.

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
        self._all_fm_patches_list = []  # Reset on each call (prevent stale data on retry)
        all_labels = []
        all_sample_types = []
        all_anomaly_types = []
        all_disc_per_feature = []

        # Detail collectors (when collect_detail=True)
        if collect_detail:
            det_originals = []
            det_teacher_recons = []
            det_student_recons = []
            det_discrepancies = []
            det_point_labels = []
            _te_norm_sum = 0.0; _te_norm_cnt = 0
            _te_anom_sum = 0.0; _te_anom_cnt = 0
            _se_norm_sum = 0.0; _se_norm_cnt = 0
            _se_anom_sum = 0.0; _se_anom_cnt = 0

        patch_size = self.config.patch_size
        num_patches = self.config.num_patches

        with torch.no_grad(), autocast('cuda', enabled=self.use_amp, dtype=self.amp_dtype):
            for batch in tqdm(self.test_loader, desc="Patch scores", leave=False):
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
                # Memory per forward: batch_size * patch_batch_size * seq_len * features * 4 bytes
                # Reduce patch_batch_size for d_model>256 to prevent GPU OOM on large test sets (SWaT/WaDi)
                _use_complementary = getattr(self.config, 'eval_complementary_masking', False)
                patch_batch_size = min(num_patches, 2 if self.config.d_model > 256 else 10)
                batch_recon_patches = torch.zeros(batch_size, num_patches, device=self.config.device)
                batch_disc_patches = torch.zeros(batch_size, num_patches, device=self.config.device)
                batch_student_recon_patches = torch.zeros(batch_size, num_patches, device=self.config.device)
                batch_fm_patches = torch.zeros(batch_size, num_patches, device=self.config.device) if getattr(self.config, 'use_feature_matching', False) else None
                # Per-feature discrepancy: (B, num_patches, F) — window mean computed after loop
                batch_disc_per_feature = torch.zeros(batch_size, num_patches, num_features, device=self.config.device)

                # Detail: per-timestep reconstruction assembly
                if collect_detail:
                    teacher_recon_ts = torch.zeros(batch_size, seq_length, device=self.config.device)
                    student_recon_ts = torch.zeros(batch_size, seq_length, device=self.config.device)
                    disc_ts = torch.zeros(batch_size, seq_length, device=self.config.device)
                    teacher_err_ts = torch.zeros(batch_size, seq_length, device=self.config.device)
                    student_err_ts = torch.zeros(batch_size, seq_length, device=self.config.device)

                if _use_complementary:
                    # --- Complementary masking: K groups, no batch expansion ---
                    _K = getattr(self.config, 'eval_complementary_k', 7)
                    _perm = torch.randperm(num_patches)
                    _sizes = [num_patches // _K + (1 if i < num_patches % _K else 0) for i in range(_K)]
                    _groups = torch.split(_perm, _sizes)

                    for group in _groups:
                        group_patches = group.tolist()

                        # Create mask: 1=visible, 0=masked (for all group patches)
                        masks = torch.ones(batch_size, seq_length, device=self.config.device)
                        for patch_idx in group_patches:
                            start_pos = patch_idx * patch_size
                            end_pos = start_pos + patch_size
                            masks[:, start_pos:end_pos] = 0

                        # Single forward pass (no batch expansion needed)
                        teacher_output, student_output, _ = self.model(sequences, masking_ratio=0.0, mask=masks)

                        # FM distance (if enabled)
                        _fm_per_patch_comp = None
                        if batch_fm_patches is not None:
                            _t_hid = getattr(self.model, '_teacher_hidden', None)
                            _s_hid = getattr(self.model, '_student_hidden', None)
                            if _t_hid is not None and _s_hid is not None:
                                import torch.nn.functional as _F
                                _fm_metric = getattr(self.config, 'fm_distance_metric', 'cosine')
                                if _fm_metric == 'l2':
                                    _fm_per_patch_comp = ((_t_hid - _s_hid) ** 2).mean(dim=-1).transpose(0, 1)
                                else:
                                    _cos = _F.cosine_similarity(_t_hid, _s_hid, dim=-1)
                                    _fm_per_patch_comp = (1 - _cos).transpose(0, 1)

                        # Compute errors
                        _recon_full = (teacher_output - sequences) ** 2
                        _student_full = (student_output - sequences) ** 2
                        _disc_full = (teacher_output - student_output) ** 2
                        recon_error = _recon_full.mean(dim=2)
                        student_recon_error = _student_full.mean(dim=2)
                        discrepancy = _disc_full.mean(dim=2)

                        # Extract scores for each masked patch in this group
                        for patch_idx in group_patches:
                            start_pos = patch_idx * patch_size
                            end_pos = start_pos + patch_size
                            batch_recon_patches[:, patch_idx] = recon_error[:, start_pos:end_pos].mean(dim=1)
                            batch_student_recon_patches[:, patch_idx] = student_recon_error[:, start_pos:end_pos].mean(dim=1)
                            batch_disc_patches[:, patch_idx] = discrepancy[:, start_pos:end_pos].mean(dim=1)
                            batch_disc_per_feature[:, patch_idx, :] = _disc_full[:, start_pos:end_pos, :].mean(dim=1)
                            if _fm_per_patch_comp is not None:
                                batch_fm_patches[:, patch_idx] = _fm_per_patch_comp[:, patch_idx]

                            if collect_detail:
                                teacher_recon_ts[:, start_pos:end_pos] = teacher_output[:, start_pos:end_pos, 0]
                                student_recon_ts[:, start_pos:end_pos] = student_output[:, start_pos:end_pos, 0]
                                disc_ts[:, start_pos:end_pos] = discrepancy[:, start_pos:end_pos]
                                teacher_err_ts[:, start_pos:end_pos] = recon_error[:, start_pos:end_pos]
                                student_err_ts[:, start_pos:end_pos] = student_recon_error[:, start_pos:end_pos]

                        del _recon_full, _student_full, _disc_full, teacher_output, student_output
                        del recon_error, student_recon_error, discrepancy, masks

                else:
                    # --- Leave-one-out masking (default) ---
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

                        # FM distance per patch (if feature matching enabled)
                        _fm_per_patch = None
                        if batch_fm_patches is not None:
                            _t_hid = getattr(self.model, '_teacher_hidden', None)
                            _s_hid = getattr(self.model, '_student_hidden', None)
                            if _t_hid is not None and _s_hid is not None:
                                import torch.nn.functional as _F
                                _fm_metric = getattr(self.config, 'fm_distance_metric', 'cosine')
                                if _fm_metric == 'l2':
                                    _fm_dist = ((_t_hid - _s_hid) ** 2).mean(dim=-1).transpose(0, 1)
                                else:  # cosine
                                    _cos = _F.cosine_similarity(_t_hid, _s_hid, dim=-1)
                                    _fm_dist = (1 - _cos).transpose(0, 1)
                                _fm_dist = _fm_dist.view(batch_size, current_batch_patches, num_patches)
                                _fm_per_patch = _fm_dist

                        # Compute errors — keep full (B', L, F) for per-feature stats before reducing
                        _recon_full = (teacher_output - expanded) ** 2
                        _student_full = (student_output - expanded) ** 2
                        _disc_full = (teacher_output - student_output) ** 2
                        recon_error = _recon_full.mean(dim=2)
                        student_recon_error = _student_full.mean(dim=2)
                        discrepancy = _disc_full.mean(dim=2)

                        # Reshape to (B, current_batch_patches, S)
                        recon_error = recon_error.view(batch_size, current_batch_patches, seq_length)
                        student_recon_error = student_recon_error.view(batch_size, current_batch_patches, seq_length)
                        discrepancy = discrepancy.view(batch_size, current_batch_patches, seq_length)

                        # Per-feature discrepancy: reshape for per-patch extraction
                        _disc_full_4d = _disc_full.view(batch_size, current_batch_patches, seq_length, num_features)

                        # Detail: reshape outputs for feature-0 extraction before del
                        if collect_detail:
                            t_out = teacher_output.view(batch_size, current_batch_patches, seq_length, num_features)
                            s_out = student_output.view(batch_size, current_batch_patches, seq_length, num_features)

                        # Extract scores for each patch's masked region
                        for i, patch_idx in enumerate(range(batch_start, batch_end)):
                            start_pos = patch_idx * patch_size
                            end_pos = start_pos + patch_size
                            batch_recon_patches[:, patch_idx] = recon_error[:, i, start_pos:end_pos].mean(dim=1)
                            batch_student_recon_patches[:, patch_idx] = student_recon_error[:, i, start_pos:end_pos].mean(dim=1)
                            batch_disc_patches[:, patch_idx] = discrepancy[:, i, start_pos:end_pos].mean(dim=1)
                            batch_disc_per_feature[:, patch_idx, :] = _disc_full_4d[:, i, start_pos:end_pos, :].mean(dim=1)
                            if _fm_per_patch is not None:
                                batch_fm_patches[:, patch_idx] = _fm_per_patch[:, i, patch_idx]

                            if collect_detail:
                                teacher_recon_ts[:, start_pos:end_pos] = t_out[:, i, start_pos:end_pos, 0]
                                student_recon_ts[:, start_pos:end_pos] = s_out[:, i, start_pos:end_pos, 0]
                                disc_ts[:, start_pos:end_pos] = discrepancy[:, i, start_pos:end_pos]
                                teacher_err_ts[:, start_pos:end_pos] = recon_error[:, i, start_pos:end_pos]
                                student_err_ts[:, start_pos:end_pos] = student_recon_error[:, i, start_pos:end_pos]

                        # Clear intermediate tensors
                        del _recon_full, _student_full, _disc_full, _disc_full_4d
                        if collect_detail:
                            del expanded, masks, teacher_output, student_output, recon_error, student_recon_error, discrepancy, t_out, s_out
                        else:
                            del expanded, masks, teacher_output, student_output, recon_error, student_recon_error, discrepancy

                all_recon_patches.append(batch_recon_patches.cpu().numpy())
                all_disc_patches.append(batch_disc_patches.cpu().numpy())
                all_student_recon_patches.append(batch_student_recon_patches.cpu().numpy())
                if batch_fm_patches is not None:
                    self._all_fm_patches_list.append(batch_fm_patches.cpu().numpy())
                all_labels.append(window_labels.cpu().numpy())
                all_sample_types.append(sample_types.cpu().numpy())
                all_anomaly_types.append(anomaly_types.cpu().numpy())
                # Per-feature disc: (B, num_patches, F) → window mean (B, F)
                all_disc_per_feature.append(batch_disc_per_feature.mean(dim=1).cpu().numpy())

                # Detail: collect per-batch reconstruction data
                if collect_detail:
                    det_originals.append(sequences[:, :, 0].cpu().numpy())
                    det_teacher_recons.append(teacher_recon_ts.cpu().numpy())
                    det_student_recons.append(student_recon_ts.cpu().numpy())
                    det_discrepancies.append(disc_ts.cpu().numpy())
                    det_point_labels.append(point_labels.numpy())
                    # Error statistics (running sums)
                    te_cpu = teacher_err_ts.cpu()
                    se_cpu = student_err_ts.cpu()
                    norm_m = (window_labels == 0)
                    anom_m = (window_labels == 1)
                    if norm_m.any():
                        _te_norm_sum += te_cpu[norm_m].sum().item()
                        _te_norm_cnt += te_cpu[norm_m].numel()
                        _se_norm_sum += se_cpu[norm_m].sum().item()
                        _se_norm_cnt += se_cpu[norm_m].numel()
                    if anom_m.any():
                        _te_anom_sum += te_cpu[anom_m].sum().item()
                        _te_anom_cnt += te_cpu[anom_m].numel()
                        _se_anom_sum += se_cpu[anom_m].sum().item()
                        _se_anom_cnt += se_cpu[anom_m].numel()
                    del teacher_recon_ts, student_recon_ts, disc_ts, teacher_err_ts, student_err_ts, te_cpu, se_cpu

        # Assemble detail_results
        if collect_detail:
            self.detail_results = {
                'originals': np.concatenate(det_originals),           # (N, seq_length) feat 0
                'teacher_recons': np.concatenate(det_teacher_recons), # (N, seq_length) feat 0
                'student_recons': np.concatenate(det_student_recons), # (N, seq_length) feat 0
                'discrepancies': np.concatenate(det_discrepancies),   # (N, seq_length) timestep-level
                'point_labels': np.concatenate(det_point_labels),     # (N, seq_length)
                'labels': np.concatenate(all_labels),                 # (N,) — same ref as below
                'sample_types': np.concatenate(all_sample_types),     # (N,)
                'teacher_err_normal_mean': np.float64(_te_norm_sum / max(_te_norm_cnt, 1)),
                'teacher_err_anomaly_mean': np.float64(_te_anom_sum / max(_te_anom_cnt, 1)),
                'student_err_normal_mean': np.float64(_se_norm_sum / max(_se_norm_cnt, 1)),
                'student_err_anomaly_mean': np.float64(_se_anom_sum / max(_se_anom_cnt, 1)),
            }

        # Store per-feature discrepancy as instance attribute (accessed by callers)
        self.disc_per_feature = np.concatenate(all_disc_per_feature)  # (n_windows, F)

        # Store FM patches if feature matching is enabled
        if hasattr(self, '_all_fm_patches_list') and self._all_fm_patches_list:
            self.fm_patches = np.concatenate(self._all_fm_patches_list)  # (n_windows, num_patches)
            del self._all_fm_patches_list
        else:
            self.fm_patches = None

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
        - 'adaptive': Auto-scaled lambda (recon + (mean_recon/mean_disc) * disc)

        Uses caching to avoid redundant forward passes.

        Returns:
            scores: (n_samples,) anomaly scores (window-level)
            labels: (n_samples,) true labels
            sample_types: (n_samples,) sample type indicators
            anomaly_types: (n_samples,) anomaly type indicators
        """
        score_mode = self.config.anomaly_score_mode

        # Get cached raw scores
        cached = self._get_cached_scores()

        recon_all = cached['window_recon']
        disc_all = cached['window_disc']
        labels = cached['labels']
        sample_types = cached['sample_types']
        anomaly_types = cached['anomaly_types']

        # Apply scoring formula (with FM if available)
        fm_all = self._get_cached_fm_scores()
        scores = self._apply_scoring_formula(recon_all, disc_all, score_mode, fm=fm_all)

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

        Uses mean-aggregated point-level scores. PA%K also uses mean scores.
        """
        score_mode = self.config.anomaly_score_mode
        cache_key = f'anomaly_type_metrics_{score_mode}'

        if cache_key in self._cache:
            return self._cache[cache_key]

        if not (self.can_compute_point_level_pa_k and hasattr(self.test_dataset, 'anomaly_regions')):
            return {}

        cached = self._get_cached_scores()
        patch_recon = cached['patch_recon']
        patch_disc = cached['patch_disc']
        patch_fm = self.fm_patches if hasattr(self, 'fm_patches') else None
        patch_scores = self._apply_scoring_formula(patch_recon, patch_disc, score_mode, fm=patch_fm)

        pt_labels = np.array(self.test_dataset.point_labels)
        total_len = len(pt_labels)
        anomaly_regions = self.test_dataset.anomaly_regions
        normal_mask = (pt_labels == 0)

        flat_t, flat_wp, coverage, covered = self._get_aggregation_map()
        point_scores = _aggregate_with_map(patch_scores.ravel(), flat_t, flat_wp, coverage, covered, total_len, method='mean')
        point_scores = np.nan_to_num(point_scores, nan=0.0)

        if len(np.unique(pt_labels)) < 2:
            self._cache[cache_key] = {}
            return {}

        fpr, tpr, thresholds_arr = roc_curve(pt_labels, point_scores)
        optimal_idx = find_f1_optimal_idx(fpr, tpr, pt_labels)
        threshold = thresholds_arr[optimal_idx]
        point_predictions = (point_scores > threshold).astype(int)

        # Build per-point anomaly_type array
        point_anomaly_types = np.full(total_len, -1, dtype=int)
        for region in anomaly_regions:
            end = min(region.end, total_len)
            point_anomaly_types[region.start:end] = region.anomaly_type

        unique_atypes = sorted(set(
            r.anomaly_type for r in anomaly_regions if r.anomaly_type > 0
        ))

        type_region_masks = {}
        for atype_idx in unique_atypes:
            type_region_masks[atype_idx] = (point_anomaly_types == atype_idx)

        all_anomaly_region_mask = (point_anomaly_types >= 1)

        results = {}

        for atype_idx in unique_atypes:
            if atype_idx < len(ANOMALY_TYPE_NAMES):
                atype_name = ANOMALY_TYPE_NAMES[atype_idx]
            else:
                atype_name = f'fault_{atype_idx}'
            type_anomaly_mask = type_region_masks[atype_idx]

            if not type_anomaly_mask.any():
                continue

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
                type_results['prc_auc'] = float(average_precision_score(type_labels, type_scores))
                type_results['precision'] = float(precision_score(type_labels, type_predictions, zero_division=0))
                type_results['recall'] = float(recall_score(type_labels, type_predictions, zero_division=0))
                type_results['f1_score'] = float(f1_score(type_labels, type_predictions, zero_division=0))
                type_results['detection_rate'] = type_results['recall']

                eval_type_mask = ~(all_anomaly_region_mask & ~type_anomaly_mask)
                type_regions = [r for r in anomaly_regions if r.anomaly_type == atype_idx]

                for k in range(0, 101, 5):
                    # PA%K F1/Precision/Recall (mean-based)
                    pa_metrics = compute_pa_k_metrics_from_mean_scores(
                        point_scores, pt_labels, anomaly_regions, threshold, k, eval_type_mask
                    )
                    type_results[f'pa_{k}_f1'] = pa_metrics['f1']
                    type_results[f'pa_{k}_precision'] = pa_metrics['precision']
                    type_results[f'pa_{k}_recall'] = pa_metrics['recall']

                    # PA%K ROC-AUC / PRC-AUC (mean-based threshold sweep)
                    pa_roc_prc = compute_pa_k_roc_prc_from_mean_scores(
                        point_scores, pt_labels, anomaly_regions, k, eval_type_mask
                    )
                    type_results[f'pa_{k}_roc_auc'] = pa_roc_prc['roc_auc']
                    type_results[f'pa_{k}_prc_auc'] = pa_roc_prc['prc_auc']

                    # PA%K segment detection rate
                    pa_det_rate = compute_segment_pa_k_detection_rate(
                        point_scores=point_scores,
                        point_labels=pt_labels,
                        anomaly_regions=anomaly_regions,
                        anomaly_type=atype_idx,
                        threshold=threshold,
                        k_percent=k
                    )
                    type_results[f'pa_{k}_detection_rate'] = float(pa_det_rate)
            else:
                type_results['detection_rate'] = float(type_predictions.mean())
                for k in range(0, 101, 5):
                    pa_rate = compute_segment_pa_k_detection_rate(
                        point_scores=point_scores,
                        point_labels=pt_labels,
                        anomaly_regions=anomaly_regions,
                        anomaly_type=atype_idx,
                        threshold=threshold,
                        k_percent=k
                    )
                    type_results[f'pa_{k}_detection_rate'] = float(pa_rate)

            results[atype_name] = type_results

        self._cache[cache_key] = results
        return results

    def evaluate(self) -> Dict[str, float]:
        """Evaluate and return metrics at point-level.

        All metrics (including PA%K) use mean-aggregated point-level scores.
        PA%K F1/Precision/Recall use the point-level optimal threshold.
        PA%K ROC-AUC/PRC-AUC sweep thresholds on continuous mean scores.
        """
        score_mode = self.config.anomaly_score_mode
        cached = self._get_cached_scores()

        recon_patches = cached['patch_recon']
        disc_patches = cached['patch_disc']
        sample_types = cached['sample_types']
        _fm_p = self.fm_patches if hasattr(self, 'fm_patches') else None
        patch_scores = self._apply_scoring_formula(recon_patches, disc_patches, score_mode, fm=_fm_p)

        if not (self.can_compute_point_level_pa_k and hasattr(self.test_dataset, 'anomaly_regions')):
            return _zero_metric_set()

        pt_labels = np.array(self.test_dataset.point_labels)
        total_len = len(pt_labels)

        flat_t, flat_wp, coverage, covered = self._get_aggregation_map()
        point_scores = _aggregate_with_map(patch_scores.ravel(), flat_t, flat_wp, coverage, covered, total_len, method='mean')
        point_scores = np.nan_to_num(point_scores, nan=0.0)

        # === Single-source metric set (PA%K + AUC + VUS/Aff/RF1 + AR variants) ===
        # See compute_full_metric_set() docstring for the full key schema.
        anomaly_regions = self.test_dataset.anomaly_regions
        eval_mask = np.ones(total_len, dtype=bool)
        results = compute_full_metric_set(point_scores, pt_labels, anomaly_regions, eval_mask)
        threshold = results.get('optimal_threshold', 0.0)

        # Disturbing normal performance (window-level, descriptive)
        # sample_type: 0=pure_normal, 1=disturbing_normal, 2=anomaly
        _fm_w = self._get_cached_fm_scores()
        window_scores = self._apply_scoring_formula(
            cached['window_recon'], cached['window_disc'], score_mode, fm=_fm_w
        )
        window_scores = np.nan_to_num(window_scores, nan=0.0)
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
        All metrics use mean-aggregated point-level scores.

        Args:
            score_type: One of 'disc', 'teacher_recon', 'student_recon'
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
            return _zero_metric_set()

        pt_labels = np.array(self.test_dataset.point_labels)
        total_len = len(pt_labels)

        flat_t, flat_wp, coverage, covered = self._get_aggregation_map()
        point_scores = _aggregate_with_map(patch_scores.ravel(), flat_t, flat_wp, coverage, covered, total_len, method='mean')
        point_scores = np.nan_to_num(point_scores, nan=0.0)

        # === Single-source metric set (see compute_full_metric_set docstring) ===
        anomaly_regions = self.test_dataset.anomaly_regions
        eval_mask = np.ones(total_len, dtype=bool)
        results = compute_full_metric_set(point_scores, pt_labels, anomaly_regions, eval_mask)
        return results


# =============================================================================
# SWaT Region 22 Exclusion Metrics
# =============================================================================

def find_swat_largest_region(anomaly_regions):
    """Find the largest anomaly region (the problematic region 22 in SWaT)."""
    if not anomaly_regions:
        return None
    return max(anomaly_regions, key=lambda r: r.end - r.start)


def compute_metrics_with_exclusion(point_scores, pt_labels, anomaly_regions, excl_region):
    """Compute full metrics excluding a specific anomaly region.

    Args:
        point_scores: (total_len,) point-level anomaly scores
        pt_labels: (total_len,) binary labels
        anomaly_regions: list of AnomalyRegion (test-local coordinates)
        excl_region: AnomalyRegion to exclude (test-local coordinates)

    Returns:
        dict of metrics computed with eval_mask
    """
    total_len = len(pt_labels)
    eval_mask = np.ones(total_len, dtype=bool)
    eval_mask[excl_region.start:excl_region.end] = False

    masked_scores = point_scores[eval_mask]
    masked_labels = pt_labels[eval_mask]

    if len(np.unique(masked_labels)) <= 1:
        # Distinct from MAE eval methods: callers (excl22 path) expect empty dict here.
        return {}

    # Filter anomaly_regions: drop the excluded region.
    filtered_regions = [r for r in anomaly_regions
                        if not (r.start == excl_region.start and r.end == excl_region.end)]

    # === Single-source metric set (see compute_full_metric_set docstring) ===
    return compute_full_metric_set(point_scores, pt_labels, filtered_regions, eval_mask)
