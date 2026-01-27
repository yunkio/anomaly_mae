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


def aggregate_scores_to_point_level(
    window_scores: np.ndarray,
    window_start_indices: np.ndarray,
    seq_length: int,
    patch_size: int,
    total_length: int,
    method: str = 'voting',
    threshold: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate window-level scores to point-level scores (last_patch mode)

    Each timestep may be covered by multiple windows' last patches.
    This function aggregates the scores from all covering windows.

    Args:
        window_scores: (n_windows,) window-level anomaly scores
        window_start_indices: (n_windows,) start index of each window in the time series
        seq_length: Window size (e.g., 100)
        patch_size: Size of last patch (e.g., 10)
        total_length: Total length of the time series
        method: Aggregation method ('mean', 'median', 'max', 'voting')
        threshold: Threshold for binary prediction (required for 'voting')

    Returns:
        point_scores: (total_length,) aggregated scores per timestep (NaN for no coverage)
        point_coverage: (total_length,) number of windows covering each timestep
    """
    point_scores = np.full(total_length, np.nan)
    point_coverage = np.zeros(total_length, dtype=int)

    # Collect scores for each timestep
    point_score_lists = [[] for _ in range(total_length)]

    for score, start_idx in zip(window_scores, window_start_indices):
        # This window's last patch covers timesteps [start_idx + seq_length - patch_size, start_idx + seq_length)
        last_patch_start = start_idx + seq_length - patch_size
        last_patch_end = start_idx + seq_length

        for t in range(last_patch_start, min(last_patch_end, total_length)):
            point_score_lists[t].append(score)

    # Aggregate scores
    for t in range(total_length):
        if len(point_score_lists[t]) > 0:
            scores = np.array(point_score_lists[t])
            point_coverage[t] = len(scores)

            if method == 'mean':
                point_scores[t] = scores.mean()
            elif method == 'median':
                point_scores[t] = np.median(scores)
            elif method == 'max':
                point_scores[t] = scores.max()
            elif method == 'voting':
                if threshold is None:
                    raise ValueError("threshold is required for voting method")
                votes = (scores > threshold).sum()
                # Majority vote: if more than half predict anomaly, it's anomaly
                point_scores[t] = 1.0 if votes > len(scores) / 2 else 0.0
            else:
                raise ValueError(f"Unknown aggregation method: {method}")

    return point_scores, point_coverage


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
    """Aggregate per-patch scores to point-level scores (all_patches mode)

    Each window has N patches, and each patch's score is assigned to the timesteps
    that patch covers. This gives much higher coverage per timestep compared to
    last_patch mode.

    Coverage comparison:
    - last_patch: each timestep covered by ~patch_size windows
    - all_patches: each timestep covered by ~seq_length windows (N times more)

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

    # Collect scores for each timestep
    point_score_lists = [[] for _ in range(total_length)]

    for w_idx, start_idx in enumerate(window_start_indices):
        # For each patch in this window
        for p_idx in range(num_patches):
            patch_start = start_idx + p_idx * patch_size
            patch_end = patch_start + patch_size
            score = patch_scores[w_idx, p_idx]

            # Assign this patch's score to timesteps it covers
            for t in range(patch_start, min(patch_end, total_length)):
                if t >= 0:
                    point_score_lists[t].append(score)

    # Aggregate scores
    for t in range(total_length):
        if len(point_score_lists[t]) > 0:
            scores = np.array(point_score_lists[t])
            point_coverage[t] = len(scores)

            if method == 'mean':
                point_scores[t] = scores.mean()
            elif method == 'median':
                point_scores[t] = np.median(scores)
            elif method == 'max':
                point_scores[t] = scores.max()
            elif method == 'voting':
                if threshold is None:
                    raise ValueError("threshold is required for voting method")
                votes = (scores > threshold).sum()
                point_scores[t] = 1.0 if votes > len(scores) / 2 else 0.0
            else:
                raise ValueError(f"Unknown aggregation method: {method}")

    return point_scores, point_coverage


def compute_point_level_pa_k(
    window_scores: np.ndarray,
    window_start_indices: np.ndarray,
    point_labels: np.ndarray,
    seq_length: int,
    patch_size: int,
    method: str = 'voting',
    threshold: Optional[float] = None,
    k_values: List[int] = [10, 20, 50, 80]
) -> Dict[str, float]:
    """Compute point-level PA%K metrics with window score aggregation

    Args:
        window_scores: (n_windows,) window-level anomaly scores
        window_start_indices: (n_windows,) start index of each window
        point_labels: (total_length,) point-level ground truth labels
        seq_length: Window size (e.g., 100)
        patch_size: Size of last patch (e.g., 10)
        method: Aggregation method ('mean', 'median', 'max', 'voting')
        threshold: Threshold for binary prediction
        k_values: List of K% values for PA%K

    Returns:
        Dict with pa_{k}_f1 and pa_{k}_roc_auc for each k value
    """
    total_length = len(point_labels)

    # Aggregate window scores to point level
    point_scores, point_coverage = aggregate_scores_to_point_level(
        window_scores=window_scores,
        window_start_indices=window_start_indices,
        seq_length=seq_length,
        patch_size=patch_size,
        total_length=total_length,
        method=method,
        threshold=threshold
    )

    # Only use timesteps with coverage
    valid_mask = point_coverage > 0
    valid_point_scores = point_scores[valid_mask]
    valid_point_labels = point_labels[valid_mask]

    if len(valid_point_labels) == 0 or len(np.unique(valid_point_labels)) < 2:
        # No valid data or only one class
        return {f'pa_{k}_f1': 0.0 for k in k_values} | {f'pa_{k}_roc_auc': 0.0 for k in k_values}

    results = {}

    # For voting method, scores are already binary (0/1)
    if method == 'voting':
        valid_predictions = valid_point_scores.astype(int)
    else:
        # For mean/median/max, use threshold to binarize
        if threshold is None:
            # Use optimal threshold from ROC curve
            fpr, tpr, thresholds = roc_curve(valid_point_labels, valid_point_scores)
            optimal_idx = np.argmax(tpr - fpr)
            threshold = thresholds[optimal_idx]
        valid_predictions = (valid_point_scores > threshold).astype(int)

    for k in k_values:
        # Compute PA%K F1
        pa_metrics = compute_pa_k_metrics(valid_predictions, valid_point_labels, k_percent=k)
        results[f'pa_{k}_f1'] = pa_metrics['pa_k_f1']

        # Compute PA%K ROC-AUC
        # For voting, we need continuous scores for ROC-AUC
        # Re-aggregate with mean for ROC-AUC calculation
        if method == 'voting':
            mean_scores, _ = aggregate_scores_to_point_level(
                window_scores=window_scores,
                window_start_indices=window_start_indices,
                seq_length=seq_length,
                patch_size=patch_size,
                total_length=total_length,
                method='mean',
                threshold=None
            )
            valid_mean_scores = mean_scores[valid_mask]
            pa_roc_auc = compute_pa_k_roc_auc(valid_mean_scores, valid_point_labels, k_percent=k)
        else:
            pa_roc_auc = compute_pa_k_roc_auc(valid_point_scores, valid_point_labels, k_percent=k)

        results[f'pa_{k}_roc_auc'] = pa_roc_auc

    return results


def compute_point_level_pa_k_all_patches(
    patch_scores: np.ndarray,
    window_start_indices: np.ndarray,
    point_labels: np.ndarray,
    seq_length: int,
    patch_size: int,
    num_patches: int,
    method: str = 'voting',
    threshold: Optional[float] = None,
    k_values: List[int] = [10, 20, 50, 80]
) -> Dict[str, float]:
    """Compute point-level PA%K metrics using per-patch scores (all_patches mode)

    Each window has N patches, and each patch's score is used to aggregate
    point-level scores for the timesteps that patch covers.

    Args:
        patch_scores: (n_windows, num_patches) per-patch anomaly scores
        window_start_indices: (n_windows,) start index of each window
        point_labels: (total_length,) point-level ground truth labels
        seq_length: Window size (e.g., 100)
        patch_size: Size of each patch (e.g., 10)
        num_patches: Number of patches per window (e.g., 10)
        method: Aggregation method ('mean', 'median', 'max', 'voting')
        threshold: Threshold for binary prediction
        k_values: List of K% values for PA%K

    Returns:
        Dict with pa_{k}_f1 and pa_{k}_roc_auc for each k value
    """
    total_length = len(point_labels)

    # Aggregate patch scores to point level
    point_scores, point_coverage = aggregate_patch_scores_to_point_level(
        patch_scores=patch_scores,
        window_start_indices=window_start_indices,
        seq_length=seq_length,
        patch_size=patch_size,
        num_patches=num_patches,
        total_length=total_length,
        method=method,
        threshold=threshold
    )

    # Only use timesteps with coverage
    valid_mask = point_coverage > 0
    valid_point_scores = point_scores[valid_mask]
    valid_point_labels = point_labels[valid_mask]

    if len(valid_point_labels) == 0 or len(np.unique(valid_point_labels)) < 2:
        return {f'pa_{k}_f1': 0.0 for k in k_values} | {f'pa_{k}_roc_auc': 0.0 for k in k_values}

    results = {}

    if method == 'voting':
        valid_predictions = valid_point_scores.astype(int)
    else:
        if threshold is None:
            fpr, tpr, thresholds = roc_curve(valid_point_labels, valid_point_scores)
            optimal_idx = np.argmax(tpr - fpr)
            threshold = thresholds[optimal_idx]
        valid_predictions = (valid_point_scores > threshold).astype(int)

    for k in k_values:
        pa_metrics = compute_pa_k_metrics(valid_predictions, valid_point_labels, k_percent=k)
        results[f'pa_{k}_f1'] = pa_metrics['pa_k_f1']

        if method == 'voting':
            mean_scores, _ = aggregate_patch_scores_to_point_level(
                patch_scores=patch_scores,
                window_start_indices=window_start_indices,
                seq_length=seq_length,
                patch_size=patch_size,
                num_patches=num_patches,
                total_length=total_length,
                method='mean',
                threshold=None
            )
            valid_mean_scores = mean_scores[valid_mask]
            pa_roc_auc = compute_pa_k_roc_auc(valid_mean_scores, valid_point_labels, k_percent=k)
        else:
            pa_roc_auc = compute_pa_k_roc_auc(valid_point_scores, valid_point_labels, k_percent=k)

        results[f'pa_{k}_roc_auc'] = pa_roc_auc

    return results


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


def compute_pa_k_roc_auc(
    scores: np.ndarray,
    labels: np.ndarray,
    k_percent: int = 20,
    n_thresholds: int = 100
) -> float:
    """Compute PA%K adjusted ROC-AUC

    For each threshold:
    1. Binarize predictions (score > threshold -> 1)
    2. Apply PA%K adjustment to predictions
    3. Calculate TPR and FPR from adjusted predictions
    4. Build ROC curve and compute AUC

    Args:
        scores: Continuous anomaly scores
        labels: True binary labels (0/1)
        k_percent: Detection threshold percentage for PA%K
        n_thresholds: Number of thresholds to try

    Returns:
        PA%K adjusted ROC-AUC score
    """
    if len(np.unique(labels)) < 2:
        return 0.0

    # Generate thresholds from score range
    min_score, max_score = scores.min(), scores.max()
    thresholds = np.linspace(min_score - 0.01, max_score + 0.01, n_thresholds)

    tprs = []
    fprs = []

    n_positive = labels.sum()
    n_negative = len(labels) - n_positive

    for thresh in thresholds:
        # Binarize predictions
        preds = (scores > thresh).astype(int)

        # Apply PA%K adjustment
        adjusted_preds = compute_pa_k_adjusted_predictions(preds, labels, k_percent)

        # Calculate TPR and FPR from adjusted predictions
        tp = ((adjusted_preds == 1) & (labels == 1)).sum()
        fp = ((adjusted_preds == 1) & (labels == 0)).sum()

        tpr = tp / n_positive if n_positive > 0 else 0
        fpr = fp / n_negative if n_negative > 0 else 0

        tprs.append(tpr)
        fprs.append(fpr)

    # Convert to arrays and sort by FPR
    fprs = np.array(fprs)
    tprs = np.array(tprs)

    # Sort by FPR for proper AUC calculation
    sorted_indices = np.argsort(fprs)
    fprs_sorted = fprs[sorted_indices]
    tprs_sorted = tprs[sorted_indices]

    # Compute AUC using trapezoidal rule
    auc = np.trapz(tprs_sorted, fprs_sorted)

    return float(auc)


class Evaluator:
    """Evaluator for anomaly detection

    Args:
        model: Trained model
        config: Model configuration
        test_loader: DataLoader for test data
        test_dataset: Optional SlidingWindowDataset for point-level PA%K evaluation.
                     If provided, enables point-level PA%K with window score aggregation.
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

    def _get_cached_scores(self, inference_mode: str):
        """Get cached raw scores for the given inference mode, computing if needed

        Returns:
            For last_patch: dict with 'recon', 'disc', 'labels', 'sample_types', 'anomaly_types'
                           All are window-level (n_windows,)
            For all_patches: dict with 'patch_recon', 'patch_disc', 'window_recon', 'window_disc',
                            'labels', 'sample_types', 'anomaly_types', 'patch_labels',
                            'patch_sample_types', 'patch_anomaly_types'
                           Patch arrays are (n_windows, num_patches), window arrays are (n_windows,)
        """
        cache_key = f'raw_scores_{inference_mode}'
        if cache_key in self._cache:
            return self._cache[cache_key]

        if inference_mode == 'all_patches':
            # Compute patch-level scores once
            recon_patches, disc_patches, labels, sample_types, anomaly_types = self._compute_patch_scores_all_patches()

            # Derive window-level scores by averaging
            window_recon = recon_patches.mean(axis=1)
            window_disc = disc_patches.mean(axis=1)

            # Compute patch labels if possible
            patch_labels = None
            if self.can_compute_point_level_pa_k:
                patch_labels = self._compute_patch_labels()

            # Compute patch-level sample_types based on patch labels (generalized approach)
            # sample_type: 2=anomaly (patch has anomaly), 1=disturbing (normal patch in anomaly window), 0=pure_normal
            n_windows, num_patches = recon_patches.shape
            patch_sample_types = np.zeros((n_windows, num_patches), dtype=np.int64)

            if patch_labels is not None:
                # Determine which windows contain any anomaly
                window_has_anomaly = (patch_labels.sum(axis=1) > 0)  # (n_windows,)

                # Patches with anomaly → type 2
                patch_sample_types[patch_labels == 1] = 2

                # Normal patches in anomaly-containing windows → type 1 (disturbing)
                for w_idx in range(n_windows):
                    if window_has_anomaly[w_idx]:
                        for p_idx in range(num_patches):
                            if patch_labels[w_idx, p_idx] == 0:
                                patch_sample_types[w_idx, p_idx] = 1
                # Normal patches in normal windows stay 0 (pure_normal)

            # Compute patch-level anomaly_types (inherit from window if patch has anomaly, else 0)
            patch_anomaly_types = np.zeros((n_windows, num_patches), dtype=np.int64)
            if patch_labels is not None:
                for w_idx in range(n_windows):
                    for p_idx in range(num_patches):
                        if patch_labels[w_idx, p_idx] == 1:
                            patch_anomaly_types[w_idx, p_idx] = anomaly_types[w_idx]

            result = {
                'patch_recon': recon_patches,
                'patch_disc': disc_patches,
                'window_recon': window_recon,
                'window_disc': window_disc,
                'labels': labels,
                'sample_types': sample_types,  # window-level (for backward compatibility)
                'anomaly_types': anomaly_types,  # window-level
                'patch_labels': patch_labels,
                'patch_sample_types': patch_sample_types,  # (n_windows, num_patches)
                'patch_anomaly_types': patch_anomaly_types,  # (n_windows, num_patches)
            }
        else:  # last_patch
            recon, disc, labels, sample_types, anomaly_types = self._compute_raw_scores_last_patch()
            result = {
                'recon': recon,
                'disc': disc,
                'labels': labels,
                'sample_types': sample_types,
                'anomaly_types': anomaly_types,
            }

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

    def _compute_raw_scores_last_patch(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute raw recon/disc scores using last-patch masking (original method)

        Returns:
            recon_all, disc_all, labels, sample_types, anomaly_types
        """
        all_recon = []
        all_disc = []
        all_labels = []
        all_sample_types = []
        all_anomaly_types = []

        with torch.no_grad(), autocast('cuda', enabled=self.use_amp):
            for batch in self.test_loader:
                if len(batch) == 5:
                    sequences, last_patch_labels, point_labels, sample_types, anomaly_types = batch
                elif len(batch) == 4:
                    sequences, last_patch_labels, point_labels, sample_types = batch
                    anomaly_types = torch.zeros_like(last_patch_labels)
                else:
                    sequences, last_patch_labels, point_labels = batch
                    sample_types = torch.zeros_like(last_patch_labels)
                    anomaly_types = torch.zeros_like(last_patch_labels)

                sequences = sequences.to(self.config.device)
                batch_size, seq_length, num_features = sequences.shape

                mask = torch.ones(batch_size, seq_length, device=self.config.device)
                mask[:, -self.config.mask_last_n:] = 0

                teacher_output, student_output, _ = self.model(sequences, masking_ratio=0.0, mask=mask)

                recon_error = ((teacher_output - sequences) ** 2).mean(dim=2)
                discrepancy = ((teacher_output - student_output) ** 2).mean(dim=2)

                masked_positions = (mask == 0)
                recon_scores = (recon_error * masked_positions).sum(dim=1) / (masked_positions.sum(dim=1) + 1e-4)
                disc_scores = (discrepancy * masked_positions).sum(dim=1) / (masked_positions.sum(dim=1) + 1e-4)

                all_recon.append(recon_scores.cpu().numpy())
                all_disc.append(disc_scores.cpu().numpy())
                all_labels.append(last_patch_labels.cpu().numpy())
                all_sample_types.append(sample_types.cpu().numpy())
                all_anomaly_types.append(anomaly_types.cpu().numpy())

        return (
            np.concatenate(all_recon),
            np.concatenate(all_disc),
            np.concatenate(all_labels),
            np.concatenate(all_sample_types),
            np.concatenate(all_anomaly_types)
        )

    def _compute_patch_labels(self) -> np.ndarray:
        """Compute patch-level labels from point-level labels

        For all_patches mode sample-level metrics, each patch gets its own label:
        - 1 if the patch contains any anomaly timestep
        - 0 otherwise

        Returns:
            patch_labels: (n_windows, num_patches) binary labels per patch
        """
        if not self.can_compute_point_level_pa_k:
            raise RuntimeError("Patch labels require test_dataset with point_labels and window_start_indices")

        point_labels = self.test_dataset.point_labels
        window_start_indices = self.test_dataset.window_start_indices
        patch_size = self.config.patch_size
        num_patches = self.config.num_patches
        n_windows = len(window_start_indices)

        patch_labels = np.zeros((n_windows, num_patches), dtype=int)

        for w_idx, start_idx in enumerate(window_start_indices):
            for p_idx in range(num_patches):
                patch_start = start_idx + p_idx * patch_size
                patch_end = min(patch_start + patch_size, len(point_labels))

                # Label is 1 if any timestep in this patch is anomaly
                if patch_start < len(point_labels):
                    has_anomaly = point_labels[patch_start:patch_end].any()
                    patch_labels[w_idx, p_idx] = 1 if has_anomaly else 0

        return patch_labels

    def _compute_patch_scores_all_patches(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute per-patch recon/disc scores by masking each patch one at a time

        Optimized: All patches processed in a single forward pass by expanding batch dimension.
        Returns per-patch scores to be used directly for point-level aggregation.

        Returns:
            recon_patch_scores: (n_windows, num_patches) per-patch reconstruction scores
            disc_patch_scores: (n_windows, num_patches) per-patch discrepancy scores
            labels: (n_windows,) last_patch labels
            sample_types: (n_windows,) sample type indicators
            anomaly_types: (n_windows,) anomaly type indicators
        """
        all_recon_patches = []
        all_disc_patches = []
        all_labels = []
        all_sample_types = []
        all_anomaly_types = []

        patch_size = self.config.patch_size
        num_patches = self.config.num_patches

        with torch.no_grad(), autocast('cuda', enabled=self.use_amp):
            for batch in self.test_loader:
                if len(batch) == 5:
                    sequences, last_patch_labels, point_labels, sample_types, anomaly_types = batch
                elif len(batch) == 4:
                    sequences, last_patch_labels, point_labels, sample_types = batch
                    anomaly_types = torch.zeros_like(last_patch_labels)
                else:
                    sequences, last_patch_labels, point_labels = batch
                    sample_types = torch.zeros_like(last_patch_labels)
                    anomaly_types = torch.zeros_like(last_patch_labels)

                sequences = sequences.to(self.config.device)
                batch_size, seq_length, num_features = sequences.shape

                # Process patches in batches for memory efficiency
                patch_batch_size = 4  # Fixed: process 4 patches at a time
                batch_recon_patches = torch.zeros(batch_size, num_patches, device=self.config.device)
                batch_disc_patches = torch.zeros(batch_size, num_patches, device=self.config.device)

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
                    discrepancy = ((teacher_output - student_output) ** 2).mean(dim=2)

                    # Reshape to (B, current_batch_patches, S)
                    recon_error = recon_error.view(batch_size, current_batch_patches, seq_length)
                    discrepancy = discrepancy.view(batch_size, current_batch_patches, seq_length)

                    # Extract scores for each patch's masked region
                    for i, patch_idx in enumerate(range(batch_start, batch_end)):
                        start_pos = patch_idx * patch_size
                        end_pos = start_pos + patch_size
                        batch_recon_patches[:, patch_idx] = recon_error[:, i, start_pos:end_pos].mean(dim=1)
                        batch_disc_patches[:, patch_idx] = discrepancy[:, i, start_pos:end_pos].mean(dim=1)

                    # Clear intermediate tensors
                    del expanded, masks, teacher_output, student_output, recon_error, discrepancy

                all_recon_patches.append(batch_recon_patches.cpu().numpy())
                all_disc_patches.append(batch_disc_patches.cpu().numpy())
                all_labels.append(last_patch_labels.cpu().numpy())
                all_sample_types.append(sample_types.cpu().numpy())
                all_anomaly_types.append(anomaly_types.cpu().numpy())

        return (
            np.concatenate(all_recon_patches),  # (n_windows, num_patches)
            np.concatenate(all_disc_patches),   # (n_windows, num_patches)
            np.concatenate(all_labels),
            np.concatenate(all_sample_types),
            np.concatenate(all_anomaly_types)
        )

    def _compute_raw_scores_all_patches(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute window-level scores from all patches (for backward compatibility)

        Returns window-level scores by averaging all patch scores.
        For point-level PA%K, use _compute_patch_scores_all_patches() instead.

        Returns:
            recon_all, disc_all, labels, sample_types, anomaly_types
        """
        recon_patches, disc_patches, labels, sample_types, anomaly_types = self._compute_patch_scores_all_patches()

        # Aggregate patch scores to window-level
        recon_all = recon_patches.mean(axis=1)  # (n_windows,)
        disc_all = disc_patches.mean(axis=1)

        return recon_all, disc_all, labels, sample_types, anomaly_types

    def compute_anomaly_scores(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute anomaly scores for all samples in test_loader

        Supports different inference modes via config.inference_mode:
        - 'last_patch': Mask only last patch (faster, original method)
        - 'all_patches': Mask each patch one at a time, N forward passes (more robust)

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
        inference_mode = getattr(self.config, 'inference_mode', 'last_patch')
        score_mode = getattr(self.config, 'anomaly_score_mode', 'default')

        # Get cached raw scores
        cached = self._get_cached_scores(inference_mode)

        if inference_mode == 'all_patches':
            recon_all = cached['window_recon']
            disc_all = cached['window_disc']
            labels = cached['labels']
            sample_types = cached['sample_types']
            anomaly_types = cached['anomaly_types']
        else:
            recon_all = cached['recon']
            disc_all = cached['disc']
            labels = cached['labels']
            sample_types = cached['sample_types']
            anomaly_types = cached['anomaly_types']

        # Apply scoring formula
        scores = self._apply_scoring_formula(recon_all, disc_all, score_mode)

        return scores, labels, sample_types, anomaly_types

    def compute_detailed_losses(self) -> Dict[str, np.ndarray]:
        """Compute detailed losses for all samples in test_loader

        Respects config.inference_mode:
        - 'last_patch': Returns window-level data (n_windows,)
        - 'all_patches': Returns patch-level data (n_windows × num_patches,) flattened

        For 'all_patches' mode, sample_types are computed per-patch:
        - 0 = pure_normal: normal patch in normal window
        - 1 = disturbing: normal patch in anomaly-containing window
        - 2 = anomaly: patch containing anomaly

        Returns:
            Dictionary containing:
                reconstruction_loss: (n_samples,) reconstruction loss per sample
                discrepancy_loss: (n_samples,) discrepancy loss per sample
                total_loss: (n_samples,) total loss per sample
                labels: (n_samples,) true labels
                sample_types: (n_samples,) sample type indicators (patch-level for all_patches)
                anomaly_types: (n_samples,) anomaly type indicators
        """
        inference_mode = getattr(self.config, 'inference_mode', 'last_patch')
        cached = self._get_cached_scores(inference_mode)

        if inference_mode == 'all_patches':
            # Flatten patch-level data to 1D
            recon_loss = cached['patch_recon'].flatten()
            disc_loss = cached['patch_disc'].flatten()
            labels = cached['patch_labels'].flatten() if cached['patch_labels'] is not None else np.zeros_like(recon_loss)
            sample_types = cached['patch_sample_types'].flatten()
            anomaly_types = cached['patch_anomaly_types'].flatten()
        else:  # last_patch
            recon_loss = cached['recon']
            disc_loss = cached['disc']
            labels = cached['labels']
            sample_types = cached['sample_types']
            anomaly_types = cached['anomaly_types']

        return {
            'reconstruction_loss': recon_loss,
            'discrepancy_loss': disc_loss,
            'total_loss': recon_loss + self.config.lambda_disc * disc_loss,
            'labels': labels,
            'sample_types': sample_types,
            'anomaly_types': anomaly_types
        }

    def get_performance_by_anomaly_type(self) -> Dict[str, Dict[str, float]]:
        """Compute performance metrics for each anomaly type

        Returns:
            Dictionary with anomaly type names as keys, containing metrics for each type
        """
        scores, labels, sample_types, anomaly_types = self.compute_anomaly_scores()

        # Get optimal threshold from overall performance
        if len(np.unique(labels)) > 1:
            fpr, tpr, thresholds = roc_curve(labels, scores)
            optimal_idx = np.argmax(tpr - fpr)
            threshold = thresholds[optimal_idx]
        else:
            threshold = np.median(scores)

        results = {}

        for atype_idx, atype_name in enumerate(ANOMALY_TYPE_NAMES):
            type_mask = (anomaly_types == atype_idx)
            if type_mask.sum() == 0:
                continue

            type_scores = scores[type_mask]
            type_labels = labels[type_mask]
            type_predictions = (type_scores > threshold).astype(int)

            type_results = {
                'count': int(type_mask.sum()),
                'mean_score': float(type_scores.mean()),
                'std_score': float(type_scores.std()),
            }

            # Compute metrics if both classes exist
            if len(np.unique(type_labels)) > 1:
                type_results['roc_auc'] = float(roc_auc_score(type_labels, type_scores))
                type_results['precision'] = float(precision_score(type_labels, type_predictions, zero_division=0))
                type_results['recall'] = float(recall_score(type_labels, type_predictions, zero_division=0))
                type_results['f1_score'] = float(f1_score(type_labels, type_predictions, zero_division=0))
                # PA%K metrics for all K values (10, 20, 50, 80)
                for k in [10, 20, 50, 80]:
                    pa_metrics = compute_pa_k_metrics(type_predictions, type_labels, k_percent=k)
                    pa_roc_auc = compute_pa_k_roc_auc(type_scores, type_labels, k_percent=k)
                    type_results[f'pa_{k}_f1'] = float(pa_metrics['pa_k_f1'])
                    type_results[f'pa_{k}_roc_auc'] = float(pa_roc_auc)
            else:
                # For anomaly types (all labels are 1), compute detection rate
                if type_labels.sum() == len(type_labels):  # All anomalies
                    type_results['detection_rate'] = float(type_predictions.mean())
                    # PA%K detection rate for all K values
                    for k in [10, 20, 50, 80]:
                        adjusted = compute_pa_k_adjusted_predictions(type_predictions, type_labels, k_percent=k)
                        type_results[f'pa_{k}_detection_rate'] = float(adjusted.mean())
                else:  # All normal
                    type_results['false_positive_rate'] = float(type_predictions.mean())

            results[atype_name] = type_results

        return results

    def evaluate(self) -> Dict[str, float]:
        """Evaluate and return metrics including disturbing normal performance

        Sample-level metrics (roc_auc, f1_score, precision, recall):
        - last_patch mode: computed at window level (n_windows samples)
        - all_patches mode: computed at patch level (n_windows × num_patches samples)
          Each patch gets its own label (1 if patch contains anomaly, 0 otherwise)

        PA%K metrics are computed at point-level if test_dataset is provided,
        otherwise falls back to sample-level computation.

        Uses caching to avoid redundant forward passes when called multiple times
        with different scoring modes.
        """
        inference_mode = getattr(self.config, 'inference_mode', 'last_patch')
        score_mode = getattr(self.config, 'anomaly_score_mode', 'default')

        # Get cached raw scores (computes only once per inference_mode)
        cached = self._get_cached_scores(inference_mode)

        results = {}

        if inference_mode == 'all_patches' and self.can_compute_point_level_pa_k:
            # All patches mode: use patch-level scores and labels from cache
            recon_patches = cached['patch_recon']
            disc_patches = cached['patch_disc']
            patch_labels = cached['patch_labels']
            labels = cached['labels']
            sample_types = cached['sample_types']

            # Compute patch-level anomaly scores using cached helper
            patch_scores = self._apply_scoring_formula(recon_patches, disc_patches, score_mode)

            # Flatten for sample-level metrics: (n_windows × num_patches,)
            sample_scores = patch_scores.flatten()
            sample_labels = patch_labels.flatten()

            # Window-level scores for disturbing normal analysis
            window_scores = self._apply_scoring_formula(
                cached['window_recon'], cached['window_disc'], score_mode
            )
        else:
            # Last patch mode: use window-level scores and labels from cache
            recon = cached['recon']
            disc = cached['disc']
            labels = cached['labels']
            sample_types = cached['sample_types']

            sample_scores = self._apply_scoring_formula(recon, disc, score_mode)
            sample_labels = labels
            window_scores = sample_scores  # Same as sample_scores for last_patch

        # Overall performance (sample-level)
        if len(np.unique(sample_labels)) > 1:
            roc_auc = roc_auc_score(sample_labels, sample_scores)
            fpr, tpr, thresholds = roc_curve(sample_labels, sample_scores)
            optimal_idx = np.argmax(tpr - fpr)
            threshold = thresholds[optimal_idx]

            predictions = (sample_scores > threshold).astype(int)
            precision = precision_score(sample_labels, predictions, zero_division=0)
            recall = recall_score(sample_labels, predictions, zero_division=0)
            f1 = f1_score(sample_labels, predictions, zero_division=0)

            results = {
                'roc_auc': roc_auc,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'optimal_threshold': threshold,
            }

            # PA%K metrics: use point-level if available, otherwise sample-level
            pa_k_values = [10, 20, 50, 80]

            if self.can_compute_point_level_pa_k:
                aggregation_method = getattr(self.config, 'point_aggregation_method', 'voting')

                if inference_mode == 'all_patches':
                    # All patches mode: reuse patch_scores already computed above
                    # Use per-patch scores for point-level PA%K
                    point_pa_metrics = compute_point_level_pa_k_all_patches(
                        patch_scores=patch_scores,
                        window_start_indices=self.test_dataset.window_start_indices,
                        point_labels=self.test_dataset.point_labels,
                        seq_length=self.config.seq_length,
                        patch_size=self.config.patch_size,
                        num_patches=self.config.num_patches,
                        method=aggregation_method,
                        threshold=threshold,
                        k_values=pa_k_values
                    )
                else:
                    # Last patch mode: use window-level scores (original behavior)
                    point_pa_metrics = compute_point_level_pa_k(
                        window_scores=window_scores,
                        window_start_indices=self.test_dataset.window_start_indices,
                        point_labels=self.test_dataset.point_labels,
                        seq_length=self.config.seq_length,
                        patch_size=self.config.patch_size,
                        method=aggregation_method,
                        threshold=threshold,
                        k_values=pa_k_values
                    )
                results.update(point_pa_metrics)
            else:
                # Fallback to sample-level PA%K
                for k in pa_k_values:
                    f1_metrics = compute_pa_k_metrics(predictions, sample_labels, k_percent=k)
                    roc_auc_k = compute_pa_k_roc_auc(sample_scores, sample_labels, k_percent=k)
                    results[f'pa_{k}_f1'] = f1_metrics['pa_k_f1']
                    results[f'pa_{k}_roc_auc'] = roc_auc_k
        else:
            results = {
                'roc_auc': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'optimal_threshold': 0.0,
                'pa_10_f1': 0.0,
                'pa_10_roc_auc': 0.0,
                'pa_20_f1': 0.0,
                'pa_20_roc_auc': 0.0,
                'pa_50_f1': 0.0,
                'pa_50_roc_auc': 0.0,
                'pa_80_f1': 0.0,
                'pa_80_roc_auc': 0.0,
            }

        # Disturbing normal performance
        # sample_type: 0=pure_normal, 1=disturbing_normal, 2=anomaly
        # IMPORTANT: Use the GLOBAL threshold (from entire dataset), not a separate threshold
        disturbing_mask = (sample_types == 0) | (sample_types == 1)
        if disturbing_mask.sum() > 0:
            disturbing_scores = window_scores[disturbing_mask]
            disturbing_labels = sample_types[disturbing_mask]

            if len(np.unique(disturbing_labels)) > 1:
                # ROC-AUC is threshold-free, so compute it normally
                disturbing_roc_auc = roc_auc_score(disturbing_labels, disturbing_scores)

                # Use GLOBAL threshold (from entire dataset) for predictions
                # This ensures fair comparison - we evaluate how well the model
                # distinguishes disturbing normal from pure normal using the same
                # threshold that was chosen for overall anomaly detection
                d_predictions = (disturbing_scores > threshold).astype(int)
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
