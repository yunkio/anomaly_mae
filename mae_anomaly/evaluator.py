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
    """Aggregate window-level scores to point-level scores

    Each timestep may be covered by multiple windows' last patches.
    This function aggregates the scores from all covering windows.

    Args:
        window_scores: (n_windows,) window-level anomaly scores
        window_start_indices: (n_windows,) start index of each window in the time series
        seq_length: Window size (e.g., 100)
        patch_size: Size of last patch (e.g., 10)
        total_length: Total length of the time series
        method: Aggregation method ('mean', 'median', 'voting')
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
            elif method == 'voting':
                if threshold is None:
                    raise ValueError("threshold is required for voting method")
                votes = (scores > threshold).sum()
                # Majority vote: if more than half predict anomaly, it's anomaly
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
        method: Aggregation method ('mean', 'median', 'voting')
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
        # For mean/median, use threshold to binarize
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

        # Point-level PA%K requires test_dataset with specific attributes
        self.can_compute_point_level_pa_k = (
            test_dataset is not None and
            hasattr(test_dataset, 'point_labels') and
            hasattr(test_dataset, 'window_start_indices')
        )

    def compute_anomaly_scores(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute anomaly scores for all samples in test_loader

        Supports different scoring modes via config.anomaly_score_mode:
        - 'default': recon + lambda_disc * disc
        - 'normalized': Z-score normalization (recon_z + disc_z)
        - 'adaptive': Auto-scaled lambda (recon + (mean_recon/mean_disc) * disc)
        - 'ratio_weighted': Ratio-based (recon * (1 + disc/median_disc))

        Returns:
            scores: (n_samples,) anomaly scores
            labels: (n_samples,) true labels
            sample_types: (n_samples,) sample type indicators
            anomaly_types: (n_samples,) anomaly type indicators
        """
        all_recon = []
        all_disc = []
        all_labels = []
        all_sample_types = []
        all_anomaly_types = []

        # First pass: collect all raw values
        with torch.no_grad():
            for batch in self.test_loader:
                # Support 3-tuple, 4-tuple, and 5-tuple returns
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

                # Create mask for last n positions
                mask = torch.ones(batch_size, seq_length, device=self.config.device)
                mask[:, -self.config.mask_last_n:] = 0

                teacher_output, student_output, _ = self.model(sequences, masking_ratio=0.0, mask=mask)

                # Compute raw reconstruction error and discrepancy
                recon_error = ((teacher_output - sequences) ** 2).mean(dim=2)
                discrepancy = ((teacher_output - student_output) ** 2).mean(dim=2)

                # Compute per-sample scores on masked positions
                masked_positions = (mask == 0)
                recon_scores = (recon_error * masked_positions).sum(dim=1) / (masked_positions.sum(dim=1) + 1e-8)
                disc_scores = (discrepancy * masked_positions).sum(dim=1) / (masked_positions.sum(dim=1) + 1e-8)

                all_recon.append(recon_scores.cpu().numpy())
                all_disc.append(disc_scores.cpu().numpy())
                all_labels.append(last_patch_labels.cpu().numpy())
                all_sample_types.append(sample_types.cpu().numpy())
                all_anomaly_types.append(anomaly_types.cpu().numpy())

        recon_all = np.concatenate(all_recon)
        disc_all = np.concatenate(all_disc)
        labels = np.concatenate(all_labels)
        sample_types = np.concatenate(all_sample_types)
        anomaly_types = np.concatenate(all_anomaly_types)

        # Compute final scores based on scoring mode
        score_mode = getattr(self.config, 'anomaly_score_mode', 'default')

        if score_mode == 'normalized':
            # Z-score normalization: both components have same scale
            recon_mean, recon_std = recon_all.mean(), recon_all.std() + 1e-8
            disc_mean, disc_std = disc_all.mean(), disc_all.std() + 1e-8
            recon_z = (recon_all - recon_mean) / recon_std
            disc_z = (disc_all - disc_mean) / disc_std
            scores = recon_z + disc_z

        elif score_mode == 'adaptive':
            # Auto-scale lambda based on mean values
            adaptive_lambda = recon_all.mean() / (disc_all.mean() + 1e-8)
            scores = recon_all + adaptive_lambda * disc_all

        elif score_mode == 'ratio_weighted':
            # Ratio-based: use disc relative to median
            disc_median = np.median(disc_all) + 1e-8
            scores = recon_all * (1 + disc_all / disc_median)

        else:  # default
            scores = recon_all + self.config.lambda_disc * disc_all

        return scores, labels, sample_types, anomaly_types

    def compute_detailed_losses(self) -> Dict[str, np.ndarray]:
        """Compute detailed losses for all samples in test_loader

        Returns:
            Dictionary containing:
                reconstruction_loss: (n_samples,) reconstruction loss per sample
                discrepancy_loss: (n_samples,) discrepancy loss per sample
                total_loss: (n_samples,) total loss per sample
                labels: (n_samples,) true labels
                sample_types: (n_samples,) sample type indicators
                anomaly_types: (n_samples,) anomaly type indicators
        """
        all_recon_loss = []
        all_disc_loss = []
        all_labels = []
        all_sample_types = []
        all_anomaly_types = []

        with torch.no_grad():
            for batch in self.test_loader:
                # Support 3-tuple, 4-tuple, and 5-tuple returns
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

                # Create mask for last n positions
                mask = torch.ones(batch_size, seq_length, device=self.config.device)
                mask[:, -self.config.mask_last_n:] = 0

                teacher_output, student_output, _ = self.model(sequences, masking_ratio=0.0, mask=mask)

                # Compute reconstruction loss (on masked positions)
                masked_positions = (mask == 0)
                recon_error = ((teacher_output - sequences) ** 2).mean(dim=2)
                recon_loss = (recon_error * masked_positions).sum(dim=1) / (masked_positions.sum(dim=1) + 1e-8)

                # Compute discrepancy loss (on masked positions)
                disc_error = ((teacher_output - student_output) ** 2).mean(dim=2)
                disc_loss = (disc_error * masked_positions).sum(dim=1) / (masked_positions.sum(dim=1) + 1e-8)

                all_recon_loss.append(recon_loss.cpu().numpy())
                all_disc_loss.append(disc_loss.cpu().numpy())
                all_labels.append(last_patch_labels.cpu().numpy())
                all_sample_types.append(sample_types.cpu().numpy())
                all_anomaly_types.append(anomaly_types.cpu().numpy())

        recon_loss = np.concatenate(all_recon_loss)
        disc_loss = np.concatenate(all_disc_loss)
        labels = np.concatenate(all_labels)
        sample_types = np.concatenate(all_sample_types)
        anomaly_types = np.concatenate(all_anomaly_types)

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

        Sample-level metrics (roc_auc, f1_score, precision, recall) are computed
        at the window level as before.

        PA%K metrics are computed at point-level if test_dataset is provided,
        otherwise falls back to sample-level computation.
        """
        scores, labels, sample_types, anomaly_types = self.compute_anomaly_scores()

        results = {}

        # Overall performance (sample-level)
        if len(np.unique(labels)) > 1:
            roc_auc = roc_auc_score(labels, scores)
            fpr, tpr, thresholds = roc_curve(labels, scores)
            optimal_idx = np.argmax(tpr - fpr)
            threshold = thresholds[optimal_idx]

            predictions = (scores > threshold).astype(int)
            precision = precision_score(labels, predictions, zero_division=0)
            recall = recall_score(labels, predictions, zero_division=0)
            f1 = f1_score(labels, predictions, zero_division=0)

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
                # Point-level PA%K with window score aggregation
                aggregation_method = getattr(self.config, 'point_aggregation_method', 'voting')
                point_pa_metrics = compute_point_level_pa_k(
                    window_scores=scores,
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
                    f1_metrics = compute_pa_k_metrics(predictions, labels, k_percent=k)
                    roc_auc_k = compute_pa_k_roc_auc(scores, labels, k_percent=k)
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
            disturbing_scores = scores[disturbing_mask]
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
