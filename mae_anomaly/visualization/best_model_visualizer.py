"""
Best Model Visualizer - Detailed Model Analysis Visualizations

This module provides visualizations for:
- ROC curve and PR curve
- Score distribution
- Confusion matrix
- Score components (reconstruction vs discrepancy)
- Teacher-student comparison
- Reconstruction examples
- Detection examples (TP, TN, FP, FN)
- Summary statistics
- Anomaly type analysis
- Case studies
"""

import os
import glob
import json
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve, f1_score, average_precision_score
from scipy.stats import gaussian_kde
from tqdm import tqdm

from torch.utils.data import Subset

from mae_anomaly import Config, ANOMALY_TYPE_NAMES, ANOMALY_CATEGORY
from mae_anomaly.evaluator import (
    aggregate_patch_scores_to_point_level,
    compute_segment_pa_k_detection_rate,
    compute_pa_k_metrics_from_mean_scores,
    compute_pa_k_roc_prc_from_mean_scores,
    compute_pa_k_auc,
    find_f1_optimal_idx,
    _build_aggregation_map,
    _aggregate_with_map,
)
from .base import (
    get_anomaly_colors, SAMPLE_TYPE_NAMES, SAMPLE_TYPE_COLORS,
    VIS_COLORS, VIS_MARKERS, VIS_LINESTYLES,
    compute_score_contributions,
)

def _safe_lim(lo, hi, fallback_range=1.0):
    """Ensure axis limits are valid (not NaN, Inf, or degenerate).

    Returns (lo, hi) with guaranteed lo < hi and finite values.
    """
    if np.isnan(lo) or np.isinf(lo):
        lo = 0.0
    if np.isnan(hi) or np.isinf(hi):
        hi = lo + fallback_range
    if hi <= lo:
        hi = lo + fallback_range
    return float(lo), float(hi)


def _percentile_bins(arrays, n_bins=50, clip_pct=99.5):
    """Compute percentile-based histogram bin edges robust to extreme outliers.

    Args:
        arrays: list of 1-D arrays to compute joint percentile range over
        n_bins: number of bins
        clip_pct: upper percentile to clip at (avoids long tails dominating bins)

    Returns:
        1-D array of bin edges (length n_bins + 1)
    """
    combined = np.concatenate([a for a in arrays if len(a) > 0])
    if len(combined) == 0:
        return np.linspace(0, 1, n_bins + 1)
    lo = 0.0
    hi = float(np.percentile(combined, clip_pct))
    if hi <= lo:
        hi = float(combined.max()) + 1e-8
    return np.linspace(lo, hi, n_bins + 1)


def _unwrap_subset(dataset):
    """Unwrap torch Subset to get the underlying dataset with custom attributes."""
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return dataset


def _get_subset_window_indices(dataset):
    """Get window_start_indices respecting Subset filtering.

    Returns:
        Tuple of (base_dataset, window_start_indices_array)
        - base_dataset: the unwrapped dataset with anomaly_regions, point_labels, etc.
        - window_start_indices_array: indices corresponding to actual predictions
          (filtered by Subset indices if applicable)
    """
    base = _unwrap_subset(dataset)
    all_indices = np.array(base.window_start_indices)
    if isinstance(dataset, Subset):
        subset_idx = np.array(dataset.indices)
        return base, all_indices[subset_idx]
    return base, all_indices


class BestModelVisualizer:
    """Visualize best model analysis"""

    def __init__(self, model=None, config: Config = None, test_loader=None, output_dir: str = '',
                 pred_data: Dict = None, detailed_data: Dict = None):
        """Initialize BestModelVisualizer.

        Args:
            model: Trained model (optional if pred_data provided)
            config: Model configuration
            test_loader: Test data loader (optional if pred_data provided)
            output_dir: Output directory for visualizations
            pred_data: Pre-computed predictions (optional, skips GPU inference if provided)
            detailed_data: Pre-computed detailed data (optional, skips GPU inference if provided)
        """
        self.model = model
        self.config = config
        self.test_loader = test_loader
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.num_patches = getattr(config, 'num_patches', 10)

        # Collect prediction data (or use pre-computed data for efficiency)
        if pred_data is not None:
            self.pred_data = pred_data
        else:
            print("  Collecting model predictions...")
            self.pred_data = collect_predictions(model, test_loader, config)

        # === OPTIMIZATION: Lazy Loading for detailed_data ===
        # Store pre-computed detailed_data or None for lazy loading
        self._detailed_data = detailed_data
        self._detailed_data_loaded = detailed_data is not None

        # Compute and store threshold (only needs pred_data)
        self._compute_threshold()

        # === OPTIMIZATION: Pre-compute cached values (lazy for detailed_data-dependent caches) ===
        self._sample_type_masks = None
        self._init_pred_caches()

        # If detailed_data was pre-computed, initialize its caches now
        if self._detailed_data_loaded and self._detailed_data is not None:
            self._init_detailed_caches()

    @property
    def detailed_data(self):
        """Lazy-load detailed_data when first accessed."""
        if not self._detailed_data_loaded:
            print("  [Lazy] Collecting detailed data...")
            self._detailed_data = collect_detailed_data(self.model, self.test_loader, self.config)
            self._detailed_data_loaded = True
            # Initialize sample type masks now that we have detailed_data
            self._init_detailed_caches()
        return self._detailed_data

    def _init_detailed_caches(self):
        """Initialize caches that depend on detailed_data (called lazily)."""
        if self._detailed_data is None:
            return
        self._sample_type_masks = {
            'pure_normal': self._detailed_data['sample_types'] == 0,
            'disturbing': self._detailed_data['sample_types'] == 1,
            'anomaly': self._detailed_data['sample_types'] == 2,
        }

    def _get_sample_type_masks(self):
        """Get sample type masks from pred_data (patch-level, unified for all modes).

        This is the canonical source for sample_type masks used in statistical visualizations.
        (n_windows × num_patches) samples with per-patch sample_types
        """
        return self._pred_sample_type_masks

    def _get_detailed_sample_type_masks(self):
        """Get sample type masks from detailed_data (window-level, for time series indexing).

        Note: This is only used for indexing into window-level detailed_data arrays.
        For statistical analysis, use _get_sample_type_masks() instead.
        """
        if self._sample_type_masks is None:
            _ = self.detailed_data  # Trigger lazy load
        return self._sample_type_masks

    def _init_pred_caches(self):
        """Pre-compute cached values that only need pred_data (not detailed_data)."""
        # Cache predictions
        self._predictions = (self.pred_data['scores'] >= self.threshold).astype(int)

        # Cache label masks
        self._label_masks = {
            'normal': self.pred_data['labels'] == 0,
            'anomaly': self.pred_data['labels'] == 1,
        }

        # Cache sample type masks from pred_data (patch-level, unified for all modes)
        # This is the canonical source for sample_type masks
        self._pred_sample_type_masks = {
            'pure_normal': self.pred_data['sample_types'] == 0,
            'disturbing': self.pred_data['sample_types'] == 1,
            'anomaly': self.pred_data['sample_types'] == 2,
        }

        # Cache ROC data (used in multiple plots)
        fpr, tpr, thresholds = roc_curve(self.pred_data['labels'], self.pred_data['scores'])
        self._roc_data = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}
        self._optimal_idx = find_f1_optimal_idx(fpr, tpr, self.pred_data['labels'])

    # === OPTIMIZATION: Data sampling for KDE and heatmaps ===
    MAX_SAMPLES_KDE = 3000  # Max samples for KDE computation
    MAX_SAMPLES_HEATMAP = 50  # Max rows for heatmap visualization

    def _sample_for_kde(self, data: np.ndarray) -> np.ndarray:
        """Sample data for KDE computation to reduce computation time."""
        if len(data) <= self.MAX_SAMPLES_KDE:
            return data
        indices = np.random.choice(len(data), self.MAX_SAMPLES_KDE, replace=False)
        return data[indices]

    def _sample_for_heatmap(self, data: np.ndarray) -> np.ndarray:
        """Sample data for heatmap visualization to reduce rendering time."""
        if len(data) <= self.MAX_SAMPLES_HEATMAP:
            return data
        indices = np.random.choice(len(data), self.MAX_SAMPLES_HEATMAP, replace=False)
        return data[indices]

    def _compute_threshold(self):
        """Compute optimal threshold from current scores using F1-optimal ROC analysis."""
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(self.pred_data['labels'], self.pred_data['scores'])
        optimal_idx = find_f1_optimal_idx(fpr, tpr, self.pred_data['labels'])
        self.threshold = thresholds[optimal_idx]

    def recompute_scores(self, scoring_mode: str):
        """Recompute anomaly scores with different scoring mode (CPU only, fast).

        Recomputes patch-level scores, then re-aggregates to point-level.
        The raw recon_errors and discrepancies are preserved.

        Args:
            scoring_mode: One of 'default', 'adaptive'
        """
        recon = self.pred_data['recon_errors']
        disc = self.pred_data['discrepancies']
        if scoring_mode not in ('default', 'adaptive', 'ratio_weighted'):
            raise ValueError(f"Unknown scoring_mode: {scoring_mode}")
        # SINGLE SOURCE: mae_anomaly.scoring. pred_data does not currently carry
        # an FM array, so fm=None drops FM consistently with pre-refactor viz
        # behavior. Phase 2 will route FM through PatchScoresBundle.
        # The caller mutates self.config.anomaly_score_mode below, so set it
        # first so compute_score sees the requested mode.
        prev_mode = getattr(self.config, 'anomaly_score_mode', 'default')
        self.config.anomaly_score_mode = scoring_mode
        try:
            from mae_anomaly.scoring import compute_score
            patch_scores = compute_score(recon, disc, None, self.config,
                                         force_recon_only=False)
        finally:
            # restore in case the assignment below changes nothing (it does,
            # but defensive).
            self.config.anomaly_score_mode = prev_mode

        self.pred_data['patch_scores'] = patch_scores
        self.config.anomaly_score_mode = scoring_mode

        # Re-aggregate to point-level if possible
        if 'window_start_indices' in self.pred_data and 'point_labels' in self.pred_data:
            from mae_anomaly.evaluator import _build_aggregation_map, _aggregate_with_map
            n_windows = self.pred_data['n_windows']
            num_patches = self.pred_data['num_patches']
            patch_size = self.pred_data['patch_size']
            total_len = self.pred_data['total_length']
            ws_indices = self.pred_data['window_start_indices']

            # Build geometry map once, reuse for all 4 aggregations
            flat_t, flat_wp, coverage, covered = _build_aggregation_map(
                ws_indices, patch_size, num_patches, total_len
            )

            def _agg(arr_flat):
                ps = _aggregate_with_map(arr_flat, flat_t, flat_wp, coverage, covered, total_len, method='mean')
                return np.nan_to_num(ps, nan=0.0)

            point_scores = _agg(patch_scores)
            self.pred_data['point_scores'] = point_scores
            self.pred_data['scores'] = point_scores

            # Re-aggregate component scores to point-level (reuse same map)
            self.pred_data['point_recon'] = _agg(self.pred_data['recon_errors'])
            self.pred_data['point_disc'] = _agg(self.pred_data['discrepancies'])
            self.pred_data['point_student'] = _agg(self.pred_data['student_errors'])
        else:
            self.pred_data['scores'] = patch_scores

        # Recompute threshold and refresh all cached ROC data / predictions
        self._compute_threshold()
        self._init_pred_caches()

    def _patch_idx_to_window_idx(self, patch_idx: int) -> int:
        """Convert patch-level index to window-level index.

        pred_data has shape (n_windows * num_patches,)
        but detailed_data has shape (n_windows,).
        """
        return patch_idx // self.num_patches

    def _get_masked_region(self, patch_idx: int, seq_len: int) -> tuple:
        """Get the masked region (start, end) for a given patch_idx.

        Args:
            patch_idx: Index from pred_data (patch-level in all_patches mode)
            seq_len: Length of the time series (for validation)

        Returns:
            (mask_start, mask_end): Start and end indices of the masked region
        """
        patch_size = self.config.patch_size

        # Each patch_idx corresponds to masking a specific patch
        masked_patch = patch_idx % self.num_patches
        mask_start = masked_patch * patch_size
        mask_end = min((masked_patch + 1) * patch_size, seq_len)

        return mask_start, mask_end

    def _highlight_masked_region(self, ax, patch_idx: int, seq_len: int,
                                  color=None, alpha=0.2, label='Masked'):
        """Highlight the masked region for a given patch_idx.

        Args:
            ax: matplotlib axis
            patch_idx: Index from pred_data
            seq_len: Length of the time series
            color: Color for the shaded region (default: VIS_COLORS['masked_region'])
            alpha: Transparency
            label: Label for legend (set to None to skip legend)
        """
        if color is None:
            color = VIS_COLORS['masked_region']

        mask_start, mask_end = self._get_masked_region(patch_idx, seq_len)
        ax.axvspan(mask_start, mask_end, alpha=alpha, color=color, label=label)

    def _highlight_anomaly_regions(self, ax, point_labels, color=None, alpha=0.2, label='Anomaly Region'):
        # Use VIS_COLORS if color not specified
        if color is None:
            color = VIS_COLORS['anomaly_region']
        """Highlight anomaly regions with shaded areas

        Args:
            ax: matplotlib axis
            point_labels: array of point-level labels (1=anomaly, 0=normal)
            color: color for shading
            alpha: transparency
            label: label for legend (only shown for first region)
        """
        if point_labels is None:
            return

        # Find contiguous anomaly regions
        in_anomaly = False
        start_idx = 0
        first_region = True

        for i, label_val in enumerate(point_labels):
            if label_val == 1 and not in_anomaly:
                # Start of anomaly region
                start_idx = i
                in_anomaly = True
            elif label_val == 0 and in_anomaly:
                # End of anomaly region
                region_label = label if first_region else None
                ax.axvspan(start_idx, i - 1, alpha=alpha, color=color, label=region_label)
                in_anomaly = False
                first_region = False

        # Handle case where anomaly extends to end
        if in_anomaly:
            region_label = label if first_region else None
            ax.axvspan(start_idx, len(point_labels) - 1, alpha=alpha, color=color, label=region_label)

    def plot_roc_curve(self):
        """Plot ROC curve"""
        # Use cached ROC data
        fpr = self._roc_data['fpr']
        tpr = self._roc_data['tpr']
        thresholds = self._roc_data['thresholds']
        roc_auc = auc(fpr, tpr)

        # Use cached optimal index
        optimal_idx = self._optimal_idx
        optimal_threshold = thresholds[optimal_idx]

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(fpr, tpr, color=VIS_COLORS['anomaly'], lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], color=VIS_COLORS['reference'], lw=2, linestyle='--')
        ax.scatter(fpr[optimal_idx], tpr[optimal_idx], s=100, c=VIS_COLORS['threshold'], zorder=5,
                  label=f'Best F1 (threshold={optimal_threshold:.4f})')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'best_model_roc_curve.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - best_model_roc_curve.png")

    def plot_prc_curve(self):
        """Plot Precision-Recall curve"""
        labels = self.pred_data['labels']
        scores = self.pred_data['scores']

        prec, rec, thresholds = precision_recall_curve(labels, scores)
        # Use average_precision_score for proper PRC-AUC (consistent with evaluator.py)
        prc_auc = average_precision_score(labels, scores)

        # Find F1-optimal point on PR curve
        f1_scores = 2 * prec * rec / (prec + rec + 1e-10)
        optimal_idx = np.argmax(f1_scores)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(rec, prec, color=VIS_COLORS['anomaly'], lw=2, label=f'PR curve (AUC = {prc_auc:.4f})')

        # Baseline: fraction of positives
        baseline = labels.sum() / len(labels)
        ax.axhline(y=baseline, color=VIS_COLORS['reference'], lw=2, linestyle='--', label=f'Baseline ({baseline:.4f})')

        ax.scatter(rec[optimal_idx], prec[optimal_idx], s=100, c=VIS_COLORS['threshold'], zorder=5,
                  label=f'Best F1 (F1={f1_scores[optimal_idx]:.4f})')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'best_model_prc_curve.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - best_model_prc_curve.png")

    def plot_confusion_matrix(self):
        """Plot confusion matrix"""
        # Use cached values
        optimal_threshold = self._roc_data['thresholds'][self._optimal_idx]
        predictions = self._predictions
        cm = confusion_matrix(self.pred_data['labels'], predictions)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Normal', 'Anomaly'],
                   yticklabels=['Normal', 'Anomaly'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix (threshold={optimal_threshold:.4f})', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'best_model_confusion_matrix.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - best_model_confusion_matrix.png")

    def plot_reconstruction_examples(self, num_examples: int = 3):
        """Show reconstruction examples

        Shows assembled reconstruction where each position was
        reconstructed when its patch was masked. All patches are shown as reconstructed.
        """
        normal_idx = np.where(self.detailed_data['labels'] == 0)[0]
        anomaly_idx = np.where(self.detailed_data['labels'] == 1)[0]

        np.random.seed(42)
        normal_samples = np.random.choice(normal_idx, min(num_examples, len(normal_idx)), replace=False)
        anomaly_samples = np.random.choice(anomaly_idx, min(num_examples, len(anomaly_idx)), replace=False)
        all_samples = list(normal_samples) + list(anomaly_samples)
        sample_labels = ['Normal'] * len(normal_samples) + ['Anomaly'] * len(anomaly_samples)

        fig, axes = plt.subplots(len(all_samples), 3, figsize=(15, 4 * len(all_samples)))

        for row, (idx, label) in enumerate(zip(all_samples, sample_labels)):
            original = self.detailed_data['originals'][idx]
            teacher = self.detailed_data['teacher_recons'][idx]
            student = self.detailed_data['student_recons'][idx]
            disc = self.detailed_data['discrepancies'][idx]
            point_labels = self.detailed_data['point_labels'][idx]

            seq_len = len(original)
            x = np.arange(seq_len)

            # Determine representative patch_idx for masked region highlighting
            # Find the most relevant patch to highlight
            # For anomaly samples: find patch containing anomaly
            # For normal samples: use last patch
            if label == 'Anomaly' and point_labels.sum() > 0:
                # Find first anomaly point and determine its patch
                anomaly_start = np.where(point_labels == 1)[0][0]
                highlighted_patch = anomaly_start // self.config.patch_size
            else:
                # For normal samples, highlight the last patch
                highlighted_patch = self.num_patches - 1
            # Compute patch_idx for _highlight_masked_region
            representative_patch_idx = idx * self.num_patches + highlighted_patch

            # Original vs Teacher
            ax = axes[row, 0] if len(all_samples) > 1 else axes[0]
            self._highlight_anomaly_regions(ax, point_labels, alpha=0.3, label='Anomaly')
            ax.plot(x, original, 'b-', label='Original', alpha=0.8)
            ax.plot(x, teacher, 'g--', label='Teacher', alpha=0.8)
            self._highlight_masked_region(ax, representative_patch_idx, seq_len, label='Masked')
            ax.set_title(f'{label} - Original vs Teacher (Patch {highlighted_patch})')
            ax.legend(fontsize=8)

            # Original vs Student
            ax = axes[row, 1] if len(all_samples) > 1 else axes[1]
            self._highlight_anomaly_regions(ax, point_labels, alpha=0.3, label='Anomaly')
            ax.plot(x, original, 'b-', label='Original', alpha=0.8)
            ax.plot(x, student, 'r--', label='Student', alpha=0.8)
            self._highlight_masked_region(ax, representative_patch_idx, seq_len, label='Masked')
            ax.set_title(f'{label} - Original vs Student (Patch {highlighted_patch})')
            ax.legend(fontsize=8)

            # Discrepancy
            ax = axes[row, 2] if len(all_samples) > 1 else axes[2]
            self._highlight_anomaly_regions(ax, point_labels, alpha=0.3, label='Anomaly')
            ax.plot(x, disc, color=VIS_COLORS['student'], lw=2)
            self._highlight_masked_region(ax, representative_patch_idx, seq_len, label='Masked')
            ax.set_title(f'{label} - Discrepancy Profile (Patch {highlighted_patch})')
            ax.axhline(y=disc.mean(), color=VIS_COLORS['disturbing'], linestyle='--', label=f'Mean: {disc.mean():.4f}')
            ax.legend(fontsize=8)

        # Unify y-axis: within each row cols 0-1 (signal), col 2 (discrepancy) across rows
        n_samples = len(all_samples)
        for row in range(n_samples):
            ax0 = axes[row, 0] if n_samples > 1 else axes[0]
            ax1 = axes[row, 1] if n_samples > 1 else axes[1]
            _y0, _y1 = ax0.get_ylim(), ax1.get_ylim()
            _shared = (min(_y0[0], _y1[0]), max(_y0[1], _y1[1]))
            ax0.set_ylim(_shared)
            ax1.set_ylim(_shared)
        disc_axes = [axes[row, 2] if n_samples > 1 else axes[2] for row in range(n_samples)]
        _disc_ylims = [ax.get_ylim() for ax in disc_axes]
        _shared_disc = (min(y[0] for y in _disc_ylims), max(y[1] for y in _disc_ylims))
        for ax in disc_axes:
            ax.set_ylim(_shared_disc)

        plt.suptitle('Reconstruction Examples', fontsize=12, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'best_model_reconstruction.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - best_model_reconstruction.png")

    def _point_idx_to_window_idx(self, point_idx: int) -> int:
        """Find a window containing the given point timestamp.

        Returns the window index in detailed_data that contains this point.
        Falls back to 0 if no matching window found.
        """
        if 'window_start_indices' in self.pred_data:
            ws = self.pred_data['window_start_indices']
            seq_len = self.pred_data.get('seq_length', self.config.seq_length)
            # Find windows that contain this point
            for w_idx, start in enumerate(ws):
                if start <= point_idx < start + seq_len:
                    if w_idx < len(self.detailed_data['originals']):
                        return w_idx
        # Fallback: use modular mapping
        n_windows = len(self.detailed_data['originals'])
        return min(point_idx % n_windows, n_windows - 1)

    def plot_detection_examples(self):
        """Show TP, TN, FP, FN examples with anomaly region highlighted.

        Uses point-level predictions and scores.
        """
        threshold = self._roc_data['thresholds'][self._optimal_idx]
        predictions = self._predictions
        labels = self.pred_data['labels']

        # Find examples (point-level indices)
        tp_idx = np.where((predictions == 1) & (labels == 1))[0]
        tn_idx = np.where((predictions == 0) & (labels == 0))[0]
        fp_idx = np.where((predictions == 1) & (labels == 0))[0]
        fn_idx = np.where((predictions == 0) & (labels == 1))[0]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        examples = [
            (tp_idx, 'True Positive', VIS_COLORS['true_positive']),
            (tn_idx, 'True Negative', VIS_COLORS['true_negative']),
            (fp_idx, 'False Positive', VIS_COLORS['false_positive']),
            (fn_idx, 'False Negative', VIS_COLORS['false_negative'])
        ]

        for ax, (indices, title, color) in zip(axes.flatten(), examples):
            if len(indices) > 0:
                point_idx = indices[len(indices) // 2]  # Pick middle example
                window_idx = self._point_idx_to_window_idx(point_idx)
                original = self.detailed_data['originals'][window_idx]
                window_point_labels = self.detailed_data['point_labels'][window_idx]
                x = np.arange(len(original))

                self._highlight_anomaly_regions(ax, window_point_labels, alpha=0.3, label='Anomaly Region')

                ax.plot(x, original, color=color, lw=2, label='Signal')
                ax.set_title(f'{title}\nPoint score: {self.pred_data["scores"][point_idx]:.4f}, '
                           f'Threshold: {threshold:.4f}', fontweight='bold')
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Value')
                ax.legend(fontsize=8, loc='upper right')
            else:
                ax.text(0.5, 0.5, f'No {title} examples', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title, fontweight='bold')

        # Unify y-axis across all subplots (all show signal values)
        _data_axes = [ax for ax in axes.flatten() if len(ax.lines) > 0]
        if _data_axes:
            _all_ylims = [ax.get_ylim() for ax in _data_axes]
            _shared = (min(y[0] for y in _all_ylims), max(y[1] for y in _all_ylims))
            for ax in _data_axes:
                ax.set_ylim(_shared)

        plt.suptitle('Detection Examples (Point-Level)',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'best_model_detection_examples.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - best_model_detection_examples.png")

    def plot_summary_statistics(self):
        """Plot summary statistics"""
        normal_mask = self.detailed_data['labels'] == 0
        anomaly_mask = self.detailed_data['labels'] == 1

        # Use pre-computed error statistics (flat keys from memory-optimized collection)
        teacher_normal = float(self.detailed_data['teacher_err_normal_mean'])
        teacher_anomaly = float(self.detailed_data['teacher_err_anomaly_mean'])
        student_normal = float(self.detailed_data['student_err_normal_mean'])
        student_anomaly = float(self.detailed_data['student_err_anomaly_mean'])

        disc_normal = self.detailed_data['discrepancies'][normal_mask].mean()
        disc_anomaly = self.detailed_data['discrepancies'][anomaly_mask].mean()

        # ROC-AUC
        fpr, tpr, _ = roc_curve(self.pred_data['labels'], self.pred_data['scores'])
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(12, 10))
        ax.axis('off')

        text = f"""
╔════════════════════════════════════════════════════════════════════╗
║                      BEST MODEL SUMMARY                             ║
╠════════════════════════════════════════════════════════════════════╣
║  Model Configuration                                                ║
║    - Margin: {self.config.margin:.2f}                                               ║
║    - Lambda (disc): {self.config.lambda_disc:.2f}                                        ║
║    - Margin Type: {getattr(self.config, 'margin_type', 'hinge'):<10}                                  ║
║    - Patchify Mode: {getattr(self.config, 'patchify_mode', 'linear'):<12}                              ║
╠════════════════════════════════════════════════════════════════════╣
║  Sample Counts                                                      ║
║    - Normal: {normal_mask.sum():>6}                                             ║
║    - Anomaly: {anomaly_mask.sum():>5}                                             ║
╠════════════════════════════════════════════════════════════════════╣
║  Reconstruction Errors                           Normal    Anomaly  ║
║    - Teacher Error:                             {teacher_normal:.4f}    {teacher_anomaly:.4f}   ║
║    - Student Error:                             {student_normal:.4f}    {student_anomaly:.4f}   ║
╠════════════════════════════════════════════════════════════════════╣
║  Discrepancy (Masked Region)                                        ║
║    - Normal Mean:  {disc_normal:.6f}                                       ║
║    - Anomaly Mean: {disc_anomaly:.6f}                                       ║
║    - Separation Ratio: {disc_anomaly / (disc_normal + 1e-8):.2f}x                                     ║
╠════════════════════════════════════════════════════════════════════╣
║  Performance Metrics                                                ║
║    - ROC-AUC: {roc_auc:.4f}                                             ║
╚════════════════════════════════════════════════════════════════════╝
        """

        ax.text(0.1, 0.9, text, fontsize=11, family='monospace',
               verticalalignment='top', transform=ax.transAxes)

        plt.savefig(os.path.join(self.output_dir, 'best_model_summary.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - best_model_summary.png")

    def _compute_anomaly_type_metrics(self):
        """Compute per-anomaly-type detection metrics for all scoring methods.

        Returns a dict keyed by scoring method name, each containing:
            anomaly_types, detection_rates, pa_{10,20,50,80}_rates,
            mean_scores, counts, normal_mean_score, disturbing_mean_score
        """
        labels = self.pred_data['labels']
        combined_scores = self.pred_data['scores']
        point_recon = self.pred_data.get('point_recon', combined_scores)
        point_disc = self.pred_data.get('point_disc', combined_scores)
        point_student = self.pred_data.get('point_student', combined_scores)

        # Build point-level anomaly_types from anomaly_regions
        raw_dataset = self.test_loader.dataset if hasattr(self.test_loader, 'dataset') else None
        base_dataset = _unwrap_subset(raw_dataset) if raw_dataset is not None else None
        total_length = len(labels)
        anomaly_types_arr = np.zeros(total_length, dtype=int)
        if base_dataset is not None and hasattr(base_dataset, 'anomaly_regions'):
            for region in base_dataset.anomaly_regions:
                start = region.start
                end = min(region.end, total_length)
                anomaly_types_arr[start:end] = region.anomaly_type

        scoring_methods = [
            ('Anomaly Score', combined_scores, self.pred_data.get('patch_scores')),
            ('Discrepancy', point_disc, self.pred_data.get('discrepancies')),
            ('Teacher Recon', point_recon, self.pred_data.get('recon_errors')),
            ('Student Recon', point_student, self.pred_data.get('student_errors')),
        ]

        # Check segment-based PA%K availability
        can_use_segment_pa_k = (
            base_dataset is not None and
            hasattr(base_dataset, 'anomaly_regions') and
            hasattr(base_dataset, 'point_labels') and
            hasattr(base_dataset, 'window_start_indices') and
            len(base_dataset.anomaly_regions) > 0
        )

        # Pre-compute shared aggregation map for mean-based point-level scores
        agg_map = None
        window_start_indices = None
        full_length = total_length
        num_patches = getattr(self.config, 'num_patches', 10)
        n_windows = 0
        if can_use_segment_pa_k:
            full_length = len(base_dataset.point_labels)
            _, window_start_indices = _get_subset_window_indices(raw_dataset)
            n_windows = len(window_start_indices)
            agg_map = _build_aggregation_map(
                window_start_indices, self.config.patch_size,
                num_patches, full_length,
            )

        # Compute per-method metrics
        all_method_metrics = {}
        for score_name, scores, patch_scores_flat in scoring_methods:
            # Optimal threshold
            if len(np.unique(labels)) > 1:
                fpr, tpr, thresholds = roc_curve(labels, scores)
                optimal_idx = find_f1_optimal_idx(fpr, tpr, labels)
                threshold = thresholds[optimal_idx]
            else:
                threshold = np.median(scores)
            predictions = (scores > threshold).astype(int)

            # Compute mean-aggregated point-level scores for PA%K
            point_scores_mean = None
            if can_use_segment_pa_k and patch_scores_flat is not None and len(patch_scores_flat) == n_windows * num_patches:
                flat_t, flat_wp, coverage, covered = agg_map
                point_scores_mean = _aggregate_with_map(
                    patch_scores_flat, flat_t, flat_wp, coverage, covered,
                    full_length, method='mean'
                )
                point_scores_mean = np.nan_to_num(point_scores_mean, nan=0.0)

            # Per-type metrics
            anomaly_types_list = []
            detection_rates = []
            pa_10_rates, pa_20_rates, pa_50_rates, pa_80_rates = [], [], [], []
            mean_scores_list = []
            counts = []

            for atype_idx in range(1, 10):
                atype_name = ANOMALY_TYPE_NAMES[atype_idx] if atype_idx < len(ANOMALY_TYPE_NAMES) else f'type_{atype_idx}'
                type_mask = (anomaly_types_arr == atype_idx)
                if type_mask.sum() == 0:
                    continue

                type_scores = scores[type_mask]
                type_labels = labels[type_mask]
                type_predictions = predictions[type_mask]
                anomaly_types_list.append(atype_name)
                mean_scores_list.append(float(type_scores.mean()))
                counts.append(int(type_mask.sum()))

                anomaly_sample_mask = (type_labels == 1)
                if anomaly_sample_mask.sum() > 0:
                    detection_rates.append(float(type_predictions[anomaly_sample_mask].mean()) * 100)
                    if can_use_segment_pa_k and point_scores_mean is not None:
                        for k, rate_list in [(10, pa_10_rates), (20, pa_20_rates),
                                             (50, pa_50_rates), (80, pa_80_rates)]:
                            pa_rate = compute_segment_pa_k_detection_rate(
                                point_scores=point_scores_mean,
                                point_labels=base_dataset.point_labels,
                                anomaly_regions=base_dataset.anomaly_regions,
                                anomaly_type=atype_idx, threshold=threshold, k_percent=k
                            )
                            rate_list.append(float(pa_rate) * 100)
                    else:
                        pa_10_rates.append(0)
                        pa_20_rates.append(0)
                        pa_50_rates.append(0)
                        pa_80_rates.append(0)
                else:
                    detection_rates.append(0)
                    pa_10_rates.append(0)
                    pa_20_rates.append(0)
                    pa_50_rates.append(0)
                    pa_80_rates.append(0)

            # Normal / disturbing normal mean scores
            normal_mask = (labels == 0) & (anomaly_types_arr == 0)
            disturbing_mask = (labels == 0) & (anomaly_types_arr > 0)
            normal_mean_score = float(scores[normal_mask].mean()) if normal_mask.sum() > 0 else 0.0
            disturbing_mean_score = float(scores[disturbing_mask].mean()) if disturbing_mask.sum() > 0 else 0.0

            all_method_metrics[score_name] = {
                'anomaly_types': anomaly_types_list,
                'detection_rates': detection_rates,
                'pa_10_rates': pa_10_rates,
                'pa_20_rates': pa_20_rates,
                'pa_50_rates': pa_50_rates,
                'pa_80_rates': pa_80_rates,
                'mean_scores': mean_scores_list,
                'counts': counts,
                'normal_mean_score': normal_mean_score,
                'disturbing_mean_score': disturbing_mean_score,
            }

        return all_method_metrics

    def plot_performance_by_anomaly_type(self, anomaly_type_metrics: Dict = None):
        """Plot detection performance by anomaly type (combined anomaly score only).

        Args:
            anomaly_type_metrics: Pre-computed metrics from _compute_anomaly_type_metrics()
        """
        if anomaly_type_metrics is None or 'Anomaly Score' not in anomaly_type_metrics:
            print("  ! Skipping performance_by_anomaly_type (no metrics)")
            return

        m = anomaly_type_metrics['Anomaly Score']
        anomaly_types_original = m['anomaly_types']
        if len(anomaly_types_original) == 0:
            print("  ! Skipping performance_by_anomaly_type (no anomaly types found)")
            return

        anomaly_types_display = [a.replace('_', '\n') for a in anomaly_types_original]
        detection_rates = m['detection_rates']
        pa_10_detection_rates = m['pa_10_rates']
        pa_20_detection_rates = m['pa_20_rates']
        pa_50_detection_rates = m['pa_50_rates']
        pa_80_detection_rates = m['pa_80_rates']
        mean_scores = m['mean_scores']
        counts = m['counts']
        normal_mean_score = m['normal_mean_score']
        disturbing_mean_score = m['disturbing_mean_score']
        display_scores = mean_scores
        display_normal = normal_mean_score
        display_disturbing = disturbing_mean_score

        # Create 3x3 subplot grid (9 total subplots)
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        anomaly_colors = get_anomaly_colors()
        colors = [anomaly_colors.get(atype, VIS_COLORS['reference']) for atype in anomaly_types_original]

        # ===== Row 1: Detection Rates by Evaluation Method =====
        # Row 1, Col 1: Point-wise Detection Rate
        ax = axes[0, 0]
        bars = ax.bar(anomaly_types_display, detection_rates, color=colors, alpha=0.8, edgecolor=VIS_COLORS['baseline'])
        ax.set_ylabel('Detection Rate (%)')
        ax.set_title('Point-wise Detection Rate', fontweight='bold')
        ax.set_ylim(0, 110)
        plt.setp(ax.get_xticklabels(), fontsize=7, rotation=0, ha='center')
        for bar, rate in zip(bars, detection_rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{rate:.0f}%', ha='center', va='bottom', fontsize=7)

        # Row 1, Col 2: PA%10 Detection Rate
        ax = axes[0, 1]
        bars = ax.bar(anomaly_types_display, pa_10_detection_rates, color=colors, alpha=0.8, edgecolor=VIS_COLORS['baseline'])
        ax.set_ylabel('Detection Rate (%)')
        ax.set_title('PA%10 Detection Rate (lenient)', fontweight='bold')
        ax.set_ylim(0, 110)
        plt.setp(ax.get_xticklabels(), fontsize=7, rotation=0, ha='center')
        for bar, rate in zip(bars, pa_10_detection_rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{rate:.0f}%', ha='center', va='bottom', fontsize=7)

        # Row 1, Col 3: PA%80 Detection Rate
        ax = axes[0, 2]
        bars = ax.bar(anomaly_types_display, pa_80_detection_rates, color=colors, alpha=0.8, edgecolor=VIS_COLORS['baseline'])
        ax.set_ylabel('Detection Rate (%)')
        ax.set_title('PA%80 Detection Rate (strict)', fontweight='bold')
        ax.set_ylim(0, 110)
        plt.setp(ax.get_xticklabels(), fontsize=7, rotation=0, ha='center')
        for bar, rate in zip(bars, pa_80_detection_rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{rate:.0f}%', ha='center', va='bottom', fontsize=7)

        # ===== Row 2: PA%K Comparison (grouped bar chart) =====
        x = np.arange(len(anomaly_types_display))
        width = 0.18

        # Row 2, Col 1: All PA%K comparison
        ax = axes[1, 0]
        bars1 = ax.bar(x - 2*width, detection_rates, width, label='Point-wise', color=colors, alpha=0.3)
        bars2 = ax.bar(x - width, pa_10_detection_rates, width, label='PA%10', color=colors, alpha=0.5)
        bars3 = ax.bar(x, pa_20_detection_rates, width, label='PA%20', color=colors, alpha=0.7)
        bars4 = ax.bar(x + width, pa_50_detection_rates, width, label='PA%50', color=colors, alpha=0.85)
        bars5 = ax.bar(x + 2*width, pa_80_detection_rates, width, label='PA%80', color=colors, alpha=1.0)
        ax.set_ylabel('Detection Rate (%)')
        ax.set_title('All PA%K Methods Comparison', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(anomaly_types_display, fontsize=7, rotation=0, ha='center')
        ax.set_ylim(0, 115)
        ax.legend(loc='upper right', fontsize=7)

        # Row 2, Col 2: PA%10 vs PA%80 (lenient vs strict)
        ax = axes[1, 1]
        width = 0.35
        bars1 = ax.bar(x - width/2, pa_10_detection_rates, width, label='PA%10 (lenient)',
                       color=colors, alpha=0.5, edgecolor=VIS_COLORS['baseline'])
        bars2 = ax.bar(x + width/2, pa_80_detection_rates, width, label='PA%80 (strict)',
                       color=colors, alpha=1.0, edgecolor=VIS_COLORS['baseline'])
        ax.set_ylabel('Detection Rate (%)')
        ax.set_title('PA%10 vs PA%80: Lenient vs Strict', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(anomaly_types_display, fontsize=7, rotation=0, ha='center')
        ax.set_ylim(0, 115)
        ax.legend(loc='upper right', fontsize=8)

        # Row 2, Col 3: Mean Anomaly Score (includes normal/disturbing reference)
        ax = axes[1, 2]

        display_scores = mean_scores
        display_normal = normal_mean_score
        display_disturbing = disturbing_mean_score

        # Bar chart for anomaly types only
        bars = ax.bar(anomaly_types_display, display_scores, color=colors, alpha=0.8, edgecolor=VIS_COLORS['baseline'])
        ax.set_ylabel('Mean Anomaly Score')
        ax.set_title('Mean Score by Sample Type', fontweight='bold')
        plt.setp(ax.get_xticklabels(), fontsize=7, rotation=0, ha='center')
        for bar, score in zip(bars, display_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                   f'{score:.4f}', ha='center', va='bottom', fontsize=6, rotation=45)

        # Add dashed horizontal lines for normal and disturbing (like threshold)
        ax.axhline(y=display_normal, color=VIS_COLORS['pure_normal'], linestyle='--', linewidth=2,
                   label=f'Pure Normal ({display_normal:.4f})')
        ax.axhline(y=display_disturbing, color=VIS_COLORS['disturbing'], linestyle='--', linewidth=2,
                   label=f'Disturbing Normal ({display_disturbing:.4f})')
        ax.legend(loc='upper right', fontsize=7)

        # ===== Row 3: Summary Statistics =====
        # Row 3, Col 1: Detection Rate Drop (PA%10 - PA%80)
        ax = axes[2, 0]
        rate_drop = [pa10 - pa80 for pa10, pa80 in zip(pa_10_detection_rates, pa_80_detection_rates)]
        bars = ax.bar(anomaly_types_display, rate_drop, color=colors, alpha=0.8, edgecolor=VIS_COLORS['baseline'])
        ax.set_ylabel('Rate Drop (%)')
        ax.set_title('Detection Consistency (PA%10 - PA%80)', fontweight='bold')
        plt.setp(ax.get_xticklabels(), fontsize=7, rotation=0, ha='center')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        for bar, drop in zip(bars, rate_drop):
            # Position text above bar, with minimum height to avoid overlap with x-axis
            text_y = max(bar.get_height(), 0) + 0.5
            ax.text(bar.get_x() + bar.get_width()/2, text_y,
                   f'{drop:.0f}%', ha='center', va='bottom', fontsize=7)

        # Row 3, Col 2: Sample Distribution (pie chart)
        ax = axes[2, 1]
        ax.pie(counts, labels=anomaly_types_original, colors=colors,
               autopct='%1.1f%%', startangle=90)
        ax.set_title('Sample Distribution', fontweight='bold')

        # Row 3, Col 3: Summary Table
        ax = axes[2, 2]
        ax.axis('off')
        # Create summary statistics
        avg_pointwise = np.mean(detection_rates)
        avg_pa10 = np.mean(pa_10_detection_rates)
        avg_pa80 = np.mean(pa_80_detection_rates)
        summary_text = f"""
Detection Rate Summary (Avg across anomaly types)

Point-wise:  {avg_pointwise:.1f}%
PA%10:       {avg_pa10:.1f}%
PA%20:       {np.mean(pa_20_detection_rates):.1f}%
PA%50:       {np.mean(pa_50_detection_rates):.1f}%
PA%80:       {avg_pa80:.1f}%

Consistency Gap: {avg_pa10 - avg_pa80:.1f}%
(Lower = more consistent detection)
"""
        ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_title('Summary Statistics', fontweight='bold')

        plt.suptitle('Performance Analysis by Anomaly Type (9 Subplots)', fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_by_anomaly_type.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - performance_by_anomaly_type.png")

    def plot_score_distribution_by_type(self, experiment_dir: str = None):
        """Plot violin charts comparing reconstruction and discrepancy score distributions
        by normal and each anomaly pattern type.

        Creates subplots for each pattern type, with each subplot showing both
        reconstruction score and discrepancy score (with scoring mode applied) as violins.

        Args:
            experiment_dir: Path to experiment directory containing best_model_detailed.csv
        """
        detailed_csv = None
        if experiment_dir:
            csv_path = os.path.join(experiment_dir, 'best_model_detailed.csv')
            if os.path.exists(csv_path):
                detailed_csv = pd.read_csv(csv_path)

        if detailed_csv is None:
            print("  ! Skipping score_distribution_by_type (no detailed CSV found)")
            return

        # Get anomaly type names including normal
        from mae_anomaly import SLIDING_ANOMALY_TYPE_NAMES
        anomaly_colors = get_anomaly_colors()

        # Prepare data for violin plots
        # Order: normal first, then anomaly types in SLIDING_ANOMALY_TYPE_NAMES order
        type_order = ['normal'] + [name for name in SLIDING_ANOMALY_TYPE_NAMES[1:]]
        type_order = [t for t in type_order if t in detailed_csv['anomaly_type_name'].unique()]

        if len(type_order) == 0:
            print("  ! Skipping score_distribution_by_type (no anomaly types found)")
            return

        # Get scoring mode and compute weighted scores
        scoring_mode = getattr(self.config, 'anomaly_score_mode', 'default')
        lambda_disc = getattr(self.config, 'lambda_disc', 0.5)

        # Compute scores based on scoring mode.
        # SINGLE SOURCE: mae_anomaly.scoring.compute_adaptive_components for the
        # adaptive branch; the default/ratio_weighted branches fall back to the
        # simple expressions inline (no FM involvement). This call site only
        # builds per-axis numbers for a score-distribution boxplot, so we want
        # the raw recon_scores and the rescaled disc_scores (== student_error
        # when fm=None) such that recon_scores + disc_scores == total score.
        recon_raw = detailed_csv['reconstruction_loss'].values
        disc_raw = detailed_csv['discrepancy_loss'].values

        if scoring_mode == 'adaptive':
            from mae_anomaly.scoring import compute_adaptive_components
            comps = compute_adaptive_components(recon_raw, disc_raw, None, self.config,
                                                force_recon_only=False)
            recon_scores = recon_raw
            disc_scores = comps['student_error']
        else:  # default / ratio_weighted fallback for the boxplot
            recon_scores = recon_raw
            disc_scores = lambda_disc * disc_raw

        detailed_csv['recon_score'] = recon_scores
        detailed_csv['disc_score'] = disc_scores

        # Create figure with single horizontal row for easy comparison
        n_types = len(type_order)
        fig, axes = plt.subplots(1, n_types, figsize=(3 * n_types, 5), sharey=True)
        if n_types == 1:
            axes = [axes]

        # Find global y-axis limits (0.01 ~ 99.99 percentile for near-full range)
        all_scores = np.concatenate([recon_scores, disc_scores])
        y_min = np.percentile(all_scores, 0.01)
        y_max = np.percentile(all_scores, 99.99)
        y_min, y_max = _safe_lim(y_min, y_max)
        y_margin = (y_max - y_min) * 0.05  # 5% margin on each side
        y_min = y_min - y_margin
        y_max = y_max + y_margin

        for idx, atype in enumerate(type_order):
            ax = axes[idx]
            type_data = detailed_csv[detailed_csv['anomaly_type_name'] == atype]
            type_color = anomaly_colors.get(atype, VIS_COLORS['reference'])

            if len(type_data) == 0:
                ax.set_visible(False)
                continue

            # OPT-B: Subsample for violin KDE (converges at ~10K samples)
            MAX_VIOLIN = 10_000
            recon_vals = type_data['recon_score'].values.copy()
            disc_vals = type_data['disc_score'].values.copy()
            if len(recon_vals) > MAX_VIOLIN:
                sub_idx = np.random.choice(len(recon_vals), MAX_VIOLIN, replace=False)
                recon_vals = recon_vals[sub_idx]
                disc_vals = disc_vals[sub_idx]
            # Add tiny noise if all values identical (violin KDE requires variance)
            if np.std(recon_vals) < 1e-10:
                recon_vals = recon_vals + np.random.normal(0, 1e-8, len(recon_vals))
            if np.std(disc_vals) < 1e-10:
                disc_vals = disc_vals + np.random.normal(0, 1e-8, len(disc_vals))
            plot_data = pd.DataFrame({
                'Score': np.concatenate([recon_vals, disc_vals]),
                'Component': ['Recon'] * len(recon_vals) + ['Disc'] * len(disc_vals)
            })

            # Draw violin plot with both components
            sns.violinplot(x='Component', y='Score', data=plot_data, ax=ax,
                          palette={'Recon': VIS_COLORS['normal'], 'Disc': VIS_COLORS['anomaly']},
                          inner='box', cut=0)

            # Set title and labels
            display_name = atype.replace('_', ' ').title()
            ax.set_title(f'{display_name}\n(n={len(type_data)})', fontweight='bold', fontsize=10,
                        color=type_color)
            ax.set_xlabel('')
            ax.set_ylabel('Score' if idx == 0 else '')
            ax.set_ylim(y_min, y_max)

            # Add mean values as text (positioned near top of plot)
            recon_mean = type_data['recon_score'].mean()
            disc_mean = type_data['disc_score'].mean()
            text_y = y_max - (y_max - y_min) * 0.05  # 5% from top
            ax.text(0, text_y, f'{recon_mean:.4f}', ha='center', va='top', fontsize=8, color=VIS_COLORS['normal'])
            ax.text(1, text_y, f'{disc_mean:.4f}', ha='center', va='top', fontsize=8, color=VIS_COLORS['anomaly'])

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=VIS_COLORS['normal'], label='Recon Score'),
            Patch(facecolor=VIS_COLORS['anomaly'], label='Disc Score')
        ]
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.99))

        scoring_label = {'default': 'Default', 'adaptive': 'Adaptive'}.get(scoring_mode, scoring_mode)
        plt.suptitle(f'Score Distribution by Anomaly Type ({scoring_label} Scoring)\n'
                    '(Reconstruction vs Discrepancy Score per Pattern)',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'score_distribution_by_type.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - score_distribution_by_type.png")

    def plot_pure_vs_disturbing_normal(self):
        """Compare pure normal vs disturbing normal in detail

        Uses patch-level data from pred_data for unified analysis across all inference modes.
        sample_types are computed per-patch:
        - pure_normal (0): normal patch in normal window
        - disturbing (1): normal patch in anomaly-containing window
        - anomaly (2): patch containing anomaly
        """
        # Use patch-level sample type masks from pred_data
        masks = self._get_sample_type_masks()
        pure_normal_mask = masks['pure_normal']
        disturbing_mask = masks['disturbing']
        anomaly_mask = masks['anomaly']

        # Use pred_data for patch-level statistics (unified for all inference modes)
        # recon_errors = teacher reconstruction error (teacher vs original)
        # student_errors = student reconstruction error (student vs original)
        # discrepancies = teacher-student difference
        teacher_errors = self.pred_data['recon_errors']
        student_errors = self.pred_data['student_errors']
        discrepancies = self.pred_data['discrepancies']

        # Compute scores for each sample type using pred_data
        def compute_scores(mask):
            teacher = teacher_errors[mask]
            student = student_errors[mask]
            disc = discrepancies[mask]
            total = teacher + self.config.lambda_disc * disc
            return teacher, student, disc, total

        pure_teacher, pure_student, pure_disc, pure_total = compute_scores(pure_normal_mask)
        dist_teacher, dist_student, dist_disc, dist_total = compute_scores(disturbing_mask)
        anom_teacher, anom_student, anom_disc, anom_total = compute_scores(anomaly_mask)

        # Build list of present sample types (some may be empty for block-split datasets)
        type_info = []  # (label, short_label, color, teacher, student, disc, total, count)
        if len(pure_total) > 0:
            type_info.append(('Pure Normal', 'Pure\nNormal', VIS_COLORS['normal'],
                              pure_teacher, pure_student, pure_disc, pure_total, int(pure_normal_mask.sum())))
        if len(dist_total) > 0:
            type_info.append(('Disturbing Normal', 'Disturbing\nNormal', VIS_COLORS['disturbing'],
                              dist_teacher, dist_student, dist_disc, dist_total, int(disturbing_mask.sum())))
        if len(anom_total) > 0:
            type_info.append(('Anomaly', 'Anomaly', VIS_COLORS['anomaly'],
                              anom_teacher, anom_student, anom_disc, anom_total, int(anomaly_mask.sum())))

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Total score distribution (percentile-based bins for outlier robustness)
        ax = axes[0, 0]
        total_bins = _percentile_bins([t[6] for t in type_info])
        for label, _, color, _, _, _, total, count in type_info:
            ax.hist(total, bins=total_bins, alpha=0.6, label=f'{label} (n={count})', color=color, density=True)
        ax.set_xlabel('Total Score (Recon + λ·Disc)')
        ax.set_ylabel('Density')
        ax.set_title('Score Distribution by Sample Type', fontweight='bold')
        ax.legend()

        # 2. Box plot comparison
        ax = axes[0, 1]
        MAX_BOX = 50_000
        def _sub(arr, max_n=MAX_BOX):
            return arr[np.random.choice(len(arr), min(len(arr), max_n), replace=False)] if len(arr) > max_n else arr
        box_data = [_sub(t[6]) for t in type_info]
        box_labels = [t[1] for t in type_info]
        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
        for patch, t in zip(bp['boxes'], type_info):
            patch.set_facecolor(t[2])
            patch.set_alpha(0.7)
        ax.set_ylabel('Total Score')
        ax.set_title('Score Box Plot', fontweight='bold')

        # 3. Discrepancy comparison (key metric, percentile-based bins)
        ax = axes[0, 2]
        disc_bins = _percentile_bins([t[5] for t in type_info])
        for label, _, color, _, _, disc, _, _ in type_info:
            ax.hist(disc, bins=disc_bins, alpha=0.6, label=label, color=color, density=True)
        ax.set_xlabel('Discrepancy (Teacher-Student)')
        ax.set_ylabel('Density')
        ax.set_title('Discrepancy Distribution', fontweight='bold')
        ax.legend()

        # 4. Teacher vs Student scatter
        ax = axes[1, 0]
        MAX_SCATTER = 50_000
        def _subsample(x, y, max_n=MAX_SCATTER):
            if len(x) > max_n:
                idx = np.random.choice(len(x), max_n, replace=False)
                return x[idx], y[idx]
            return x, y
        all_maxes = []
        for label, _, color, teacher, student, _, _, _ in type_info:
            ts, ss = _subsample(teacher, student)
            ax.scatter(ts, ss, alpha=0.5, label=label, color=color, s=20)
            all_maxes.extend([teacher.max(), student.max()])
        if all_maxes:
            max_val = max(all_maxes)
            ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='y=x')
        ax.set_xlabel('Teacher Error')
        ax.set_ylabel('Student Error')
        ax.set_title('Teacher vs Student Error', fontweight='bold')
        ax.legend()

        # 5. Mean comparison bar chart
        ax = axes[1, 1]
        n_types = len(type_info)
        x = np.arange(n_types)
        width = 0.25

        means_teacher = [t[3].mean() for t in type_info]
        means_student = [t[4].mean() for t in type_info]
        means_disc = [t[5].mean() for t in type_info]

        ax.bar(x - width, means_teacher, width, label='Teacher Error', color=VIS_COLORS['teacher'])
        ax.bar(x, means_student, width, label='Student Error', color=VIS_COLORS['student'])
        ax.bar(x + width, means_disc, width, label='Discrepancy', color=VIS_COLORS['anomaly'])

        ax.set_xticks(x)
        ax.set_xticklabels([t[1] for t in type_info])
        ax.set_ylabel('Mean Value')
        ax.set_title('Mean Error Components', fontweight='bold')
        ax.legend()

        # 6. Statistics summary
        ax = axes[1, 2]
        ax.axis('off')

        # Safe mean helper for possibly-empty arrays
        def _safe_mean(arr):
            return arr.mean() if len(arr) > 0 else 0.0

        pure_n = int(pure_normal_mask.sum())
        dist_n = int(disturbing_mask.sum())
        anom_n = int(anomaly_mask.sum())
        pure_mean = _safe_mean(pure_total)
        dist_mean = _safe_mean(dist_total)
        anom_mean = _safe_mean(anom_total)
        pure_disc_mean = _safe_mean(pure_disc)
        dist_disc_mean = _safe_mean(dist_disc)
        anom_disc_mean = _safe_mean(anom_disc)

        stats_text = f"""
Pure Normal vs Disturbing Normal Analysis
═══════════════════════════════════════════════════════

Sample Counts:
  • Pure Normal:      {pure_n:>6}
  • Disturbing Normal:{dist_n:>6}
  • Anomaly:          {anom_n:>6}

Mean Total Score:
  • Pure Normal:      {pure_mean:.6f}
  • Disturbing Normal:{dist_mean:.6f}
  • Anomaly:          {anom_mean:.6f}

Mean Discrepancy (Key Metric):
  • Pure Normal:      {pure_disc_mean:.6f}
  • Disturbing Normal:{dist_disc_mean:.6f}
  • Anomaly:          {anom_disc_mean:.6f}

Separation Analysis:
  • Anom/Pure ratio:  {anom_mean / (pure_mean + 1e-8):.2f}x
  • Anom/Dist ratio:  {anom_mean / (dist_mean + 1e-8):.2f}x
  • Dist/Pure ratio:  {dist_mean / (pure_mean + 1e-8):.2f}x
        """

        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        plt.suptitle('Pure Normal vs Disturbing Normal Comparison\n'
                    '(Disturbing = normal patch in anomaly-containing window)',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'pure_vs_disturbing_normal.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - pure_vs_disturbing_normal.png")

    def plot_discrepancy_trend(self):
        """Plot discrepancy analysis using patch-level data

        Uses patch-level data from pred_data for unified analysis across all inference modes.
        Each sample is a prediction unit (one patch per masked position).

        sample_types are computed per-patch:
        - pure_normal (0): normal patch in normal window
        - disturbing (1): normal patch in anomaly-containing window
        - anomaly (2): patch containing anomaly
        """
        # Use patch-level sample type masks from pred_data
        masks = self._get_sample_type_masks()
        pure_normal_mask = masks['pure_normal']
        disturbing_mask = masks['disturbing']
        anomaly_mask = masks['anomaly']

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Get patch-level discrepancies from pred_data
        discrepancies = self.pred_data['discrepancies']
        pure_disc = discrepancies[pure_normal_mask]
        dist_disc = discrepancies[disturbing_mask]
        anom_disc = discrepancies[anomaly_mask]

        # 1. Discrepancy histogram by sample type (percentile-based bins)
        ax = axes[0, 0]
        bins = _percentile_bins([pure_disc, dist_disc, anom_disc])

        ax.hist(pure_disc, bins=bins, alpha=0.6, label=f'Pure Normal (n={len(pure_disc)})',
                color=VIS_COLORS['normal'], density=True)
        ax.hist(dist_disc, bins=bins, alpha=0.6, label=f'Disturbing (n={len(dist_disc)})',
                color=VIS_COLORS['disturbing'], density=True)
        ax.hist(anom_disc, bins=bins, alpha=0.6, label=f'Anomaly (n={len(anom_disc)})',
                color=VIS_COLORS['anomaly'], density=True)

        ax.set_xlabel('Discrepancy (Teacher-Student)')
        ax.set_ylabel('Density')
        ax.set_title('Discrepancy Distribution by Sample Type\n(Per-Patch Level)', fontweight='bold')
        ax.legend(fontsize=9)

        # 2. Box plot comparison
        ax = axes[0, 1]
        data_to_plot = [pure_disc, dist_disc, anom_disc]
        labels = ['Pure\nNormal', 'Disturbing\nNormal', 'Anomaly']
        sample_colors = [VIS_COLORS['normal'], VIS_COLORS['disturbing'], VIS_COLORS['anomaly']]

        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], sample_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        # Add mean markers
        means = [d.mean() if len(d) > 0 else 0 for d in data_to_plot]
        ax.scatter([1, 2, 3], means, color=VIS_COLORS['baseline'], marker='D', s=50, zorder=5, label='Mean')

        ax.set_ylabel('Discrepancy')
        ax.set_title('Discrepancy Box Plot by Sample Type', fontweight='bold')
        ax.legend()

        # 3. Discrepancy CDF comparison
        ax = axes[1, 0]

        for data, label, color in [(pure_disc, 'Pure Normal', VIS_COLORS['normal']),
                                    (dist_disc, 'Disturbing', VIS_COLORS['disturbing']),
                                    (anom_disc, 'Anomaly', VIS_COLORS['anomaly'])]:
            if len(data) > 0:
                sorted_data = np.sort(data)
                cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                ax.plot(sorted_data, cdf, label=label, color=color, lw=2)

        ax.set_xlabel('Discrepancy')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title('Discrepancy CDF by Sample Type', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Statistics summary
        ax = axes[1, 1]
        ax.axis('off')

        def safe_stats(arr):
            if len(arr) == 0:
                return 0, 0, 0, 0, 0
            return len(arr), arr.mean(), arr.std(), np.median(arr), arr.max()

        n_pure, mean_pure, std_pure, med_pure, max_pure = safe_stats(pure_disc)
        n_dist, mean_dist, std_dist, med_dist, max_dist = safe_stats(dist_disc)
        n_anom, mean_anom, std_anom, med_anom, max_anom = safe_stats(anom_disc)

        stats_text = f"""
Discrepancy Analysis (Per-Patch Level)
══════════════════════════════════════════════════════

Sample Counts:
  • Pure Normal:      {n_pure:>8}
  • Disturbing:       {n_dist:>8}
  • Anomaly:          {n_anom:>8}

Mean Discrepancy:
  • Pure Normal:      {mean_pure:.6f} ± {std_pure:.6f}
  • Disturbing:       {mean_dist:.6f} ± {std_dist:.6f}
  • Anomaly:          {mean_anom:.6f} ± {std_anom:.6f}

Median Discrepancy:
  • Pure Normal:      {med_pure:.6f}
  • Disturbing:       {med_dist:.6f}
  • Anomaly:          {med_anom:.6f}

Separation Ratios:
  • Anom/Pure:        {mean_anom / (mean_pure + 1e-8):.2f}x
  • Anom/Disturbing:  {mean_anom / (mean_dist + 1e-8):.2f}x
  • Disturbing/Pure:  {mean_dist / (mean_pure + 1e-8):.2f}x
        """

        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        plt.suptitle('Discrepancy Analysis (Teacher-Student Difference)',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'discrepancy_trend.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - discrepancy_trend.png")

    def plot_case_study_gallery(self, experiment_dir: str = None):
        """Generate qualitative case studies showing representative examples for each category.

        Shows one detailed example for each:
        - True Positive (correctly detected anomaly)
        - True Negative (correctly identified normal)
        - False Positive (false alarm)
        - False Negative (missed anomaly)
        - For each anomaly type (spike, memory_leak, noise, drift, network_congestion)
        """
        # Get predictions using pred_data (point-level scores and labels)
        threshold = self._get_optimal_threshold()
        scores = self._get_scores()
        predictions = (scores >= threshold).astype(int)
        labels = self.pred_data['labels']

        # Find examples for each category (point-level)
        tp_idx = np.where((labels == 1) & (predictions == 1))[0]
        tn_idx = np.where((labels == 0) & (predictions == 0))[0]
        fp_idx = np.where((labels == 0) & (predictions == 1))[0]
        fn_idx = np.where((labels == 1) & (predictions == 0))[0]

        categories = [
            ('True Positive', tp_idx, VIS_COLORS['true_positive']),
            ('True Negative', tn_idx, VIS_COLORS['true_negative']),
            ('False Positive', fp_idx, VIS_COLORS['false_positive']),
            ('False Negative', fn_idx, VIS_COLORS['false_negative'])
        ]

        fig, axes = plt.subplots(4, 3, figsize=(18, 20))

        for row, (cat_name, indices, color) in enumerate(categories):
            if len(indices) == 0:
                for col in range(3):
                    axes[row, col].text(0.5, 0.5, f'No {cat_name} found',
                                       ha='center', va='center', transform=axes[row, col].transAxes)
                    axes[row, col].set_title(f'{cat_name}')
                continue

            # Select a representative sample (median score among category)
            cat_scores = scores[indices]
            point_idx = indices[np.argsort(cat_scores)[len(cat_scores)//2]]
            window_idx = self._point_idx_to_window_idx(point_idx)

            # Column 1: Time series with reconstruction
            ax = axes[row, 0]
            original = self.detailed_data['originals'][window_idx]
            teacher_recon = self.detailed_data['teacher_recons'][window_idx]
            student_recon = self.detailed_data['student_recons'][window_idx]
            point_labels = self.detailed_data['point_labels'][window_idx]

            ax.plot(original, 'b-', lw=1.2, alpha=0.8, label='Original')
            ax.plot(teacher_recon, 'g--', lw=1.5, alpha=0.7, label='Teacher')
            ax.plot(student_recon, 'r:', lw=1.5, alpha=0.7, label='Student')

            # Highlight anomaly and masked regions
            anomaly_region = np.where(point_labels == 1)[0]
            if len(anomaly_region) > 0:
                ax.axvspan(anomaly_region[0], anomaly_region[-1], alpha=0.2, color=VIS_COLORS['anomaly_region'], label='Anomaly')
            ax.set_title(f'{cat_name}: Time Series', fontweight='bold', color=color)
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Value')
            if row == 0:
                ax.legend(fontsize=7, loc='upper right')

            # Column 2: Discrepancy profile
            ax = axes[row, 1]
            discrepancy = self.detailed_data['discrepancies'][window_idx]
            ax.fill_between(range(len(discrepancy)), discrepancy, alpha=0.6, color=VIS_COLORS['student'])
            ax.plot(discrepancy, color=VIS_COLORS['student_dark'], lw=1)

            if len(anomaly_region) > 0:
                ax.axvspan(anomaly_region[0], anomaly_region[-1], alpha=0.2, color=VIS_COLORS['anomaly_region'])

            ax.set_title(f'{cat_name}: Discrepancy (|T-S|)', fontweight='bold', color=color)
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Discrepancy')
            ax.legend(fontsize=8)

            # Column 3: Statistics
            ax = axes[row, 2]
            ax.axis('off')

            sample_score = scores[point_idx]

            stats_text = f"""
{cat_name} Case Study
═══════════════════════════════════

Point Index:  {point_idx} (Window: {window_idx})
True Label:   {'Anomaly' if labels[point_idx] == 1 else 'Normal'}
Prediction:   {'Anomaly' if predictions[point_idx] == 1 else 'Normal'}

Score Analysis:
  • Total Score:    {sample_score:.6f}
  • Threshold:      {threshold:.6f}
  • Margin:         {sample_score - threshold:+.6f}

Window Discrepancy: {np.mean(discrepancy):.6f}

Anomaly in Window:
  • Points: {len(anomaly_region)} / {len(original)}
            """

            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor=color, alpha=0.15))

        # Unify y-axis: col 0 (time series) across rows, col 1 (discrepancy) across rows
        for col_idx in range(2):
            _col_axes = [axes[row, col_idx] for row in range(4)
                         if len(axes[row, col_idx].lines) > 0 or len(axes[row, col_idx].collections) > 0]
            if _col_axes:
                _all_ylims = [ax.get_ylim() for ax in _col_axes]
                _shared = (min(y[0] for y in _all_ylims), max(y[1] for y in _all_ylims))
                for ax in _col_axes:
                    ax.set_ylim(_shared)

        plt.suptitle('Case Study Gallery: Representative Examples for Each Outcome\n'
                    '(Yellow=Masked Region, Red=Anomaly Region)',
                    fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'case_study_gallery.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - case_study_gallery.png")

    def plot_hardest_samples(self):
        """Analyze the hardest-to-detect samples (lowest margin FN and FP)."""
        threshold = self._get_optimal_threshold()
        scores = self._get_scores()
        labels = self.pred_data['labels']
        predictions = (scores >= threshold).astype(int)

        # FN: anomalies with lowest scores (furthest below threshold)
        fn_mask = (labels == 1) & (predictions == 0)
        fn_indices = np.where(fn_mask)[0]
        if len(fn_indices) > 0:
            fn_scores = scores[fn_mask]
            fn_sorted = fn_indices[np.argsort(fn_scores)]  # Lowest score first
        else:
            fn_sorted = np.array([])

        # FP: normals with highest scores (furthest above threshold)
        fp_mask = (labels == 0) & (predictions == 1)
        fp_indices = np.where(fp_mask)[0]
        if len(fp_indices) > 0:
            fp_scores = scores[fp_mask]
            fp_sorted = fp_indices[np.argsort(fp_scores)[::-1]]  # Highest score first
        else:
            fp_sorted = np.array([])

        fig, axes = plt.subplots(4, 3, figsize=(18, 20))

        # Top 2 hardest FN
        for row in range(2):
            if row < len(fn_sorted):
                idx = fn_sorted[row]
                self._plot_sample_detail(axes[row], idx, f'Hardest FN #{row+1}', VIS_COLORS['false_negative'], threshold)
            else:
                for col in range(3):
                    axes[row, col].text(0.5, 0.5, f'No FN #{row+1}', ha='center', va='center')
                    axes[row, col].axis('off')

        # Top 2 hardest FP
        for row in range(2, 4):
            fp_row = row - 2
            if fp_row < len(fp_sorted):
                idx = fp_sorted[fp_row]
                self._plot_sample_detail(axes[row], idx, f'Hardest FP #{fp_row+1}', VIS_COLORS['false_positive'], threshold)
            else:
                for col in range(3):
                    axes[row, col].text(0.5, 0.5, f'No FP #{fp_row+1}', ha='center', va='center')
                    axes[row, col].axis('off')

        # Unify y-axis: col 0 (time series) across rows, col 1 (discrepancy) across rows
        for col_idx in range(2):
            _col_axes = [axes[row, col_idx] for row in range(4)
                         if len(axes[row, col_idx].lines) > 0 or len(axes[row, col_idx].collections) > 0]
            if _col_axes:
                _all_ylims = [ax.get_ylim() for ax in _col_axes]
                _shared = (min(y[0] for y in _all_ylims), max(y[1] for y in _all_ylims))
                for ax in _col_axes:
                    ax.set_ylim(_shared)

        plt.suptitle('Hardest Samples Analysis\n'
                    'FN: Anomalies with lowest scores (hardest to detect)\n'
                    'FP: Normals with highest scores (most confusing)',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'hardest_samples.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - hardest_samples.png")

    def _plot_sample_detail(self, axes_row, point_idx, title_prefix, color, threshold):
        """Helper to plot detailed sample analysis in a row of 3 axes.

        Args:
            point_idx: Point-level index from pred_data
        """
        # Convert point index to window index for detailed_data access
        window_idx = self._point_idx_to_window_idx(point_idx)

        original = self.detailed_data['originals'][window_idx]
        teacher_recon = self.detailed_data['teacher_recons'][window_idx]
        student_recon = self.detailed_data['student_recons'][window_idx]
        discrepancy = self.detailed_data['discrepancies'][window_idx]
        point_labels = self.detailed_data['point_labels'][window_idx]
        score = self._get_scores()[point_idx]
        label = self.pred_data['labels'][point_idx]
        sample_type_label = 'anomaly' if label == 1 else 'normal'

        anomaly_region = np.where(point_labels == 1)[0]

        # Column 1: Time series
        ax = axes_row[0]
        ax.plot(original, 'b-', lw=1.2, alpha=0.8, label='Original')
        ax.plot(teacher_recon, 'g--', lw=1.5, alpha=0.7, label='Teacher')
        ax.plot(student_recon, 'r:', lw=1.5, alpha=0.7, label='Student')
        if len(anomaly_region) > 0:
            ax.axvspan(anomaly_region[0], anomaly_region[-1], alpha=0.2, color=VIS_COLORS['anomaly_region'])
        ax.set_title(f'{title_prefix}: Time Series', fontweight='bold', color=color)
        ax.legend(fontsize=7)

        # Column 2: Discrepancy
        ax = axes_row[1]
        ax.fill_between(range(len(discrepancy)), discrepancy, alpha=0.6, color=VIS_COLORS['student'])
        ax.plot(discrepancy, color=VIS_COLORS['student_dark'], lw=1)
        if len(anomaly_region) > 0:
            ax.axvspan(anomaly_region[0], anomaly_region[-1], alpha=0.2, color=VIS_COLORS['anomaly_region'])
        ax.set_title(f'{title_prefix}: Discrepancy', fontweight='bold', color=color)

        # Column 3: Stats
        ax = axes_row[2]
        ax.axis('off')

        margin = score - threshold

        stats_text = f"""
{title_prefix}
═══════════════════════════════

Point Idx:  {point_idx}
Window Idx: {window_idx}
Type:  {sample_type_label}
Label: {'Anomaly' if label == 1 else 'Normal'}

Score:     {score:.6f}
Threshold: {threshold:.6f}
Margin:    {margin:+.6f}
        """

        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.15))

    def plot_learning_curve(self, history: Dict):
        """Plot learning curves for training losses.

        Uses consistent color/marker scheme:
        - Colors: Normal=blue, Anomaly=red, Teacher=green, Student=purple
        - Markers: Discrepancy=square, Teacher recon=circle, Student recon=triangle

        Args:
            history: Training history dictionary with keys:
                - epoch: list of epoch numbers
                - train_rec_loss: teacher reconstruction loss per epoch
                - train_disc_loss: discrepancy loss per epoch
                - train_normal_loss: normal discrepancy loss (optional)
                - train_anomaly_loss: anomaly discrepancy loss (optional)
                - train_teacher_recon_normal/anomaly: teacher recon by type
                - train_student_recon_normal/anomaly: student recon by type
        """
        if history is None:
            print("  ! Skipping learning_curve (no history provided)")
            return

        epochs = history.get('epoch', [])
        if len(epochs) == 0:
            print("  ! Skipping learning_curve (empty history)")
            return

        # Check for detailed metrics
        has_detailed = 'train_teacher_recon_normal' in history
        has_discriminator = len(history.get('train_d_loss', [])) > 0

        # Create comprehensive figure (3x3 if discriminator, 2x3 otherwise)
        n_rows = 3 if has_discriminator else 2
        fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows))

        # Get warmup epochs from history (count epochs where disc_loss is 0)
        warmup_epochs = 1
        if 'train_disc_loss' in history:
            for i, d in enumerate(history['train_disc_loss']):
                if d > 0:
                    warmup_epochs = i
                    break

        # Style constants for consistency
        c_normal = VIS_COLORS['normal']      # Blue
        c_anomaly = VIS_COLORS['anomaly']    # Red
        c_teacher = VIS_COLORS['teacher']    # Green
        c_student = VIS_COLORS['student']    # Purple
        c_total = VIS_COLORS['total']        # Green
        m_disc = VIS_MARKERS['discrepancy']  # Square
        m_teacher = VIS_MARKERS['teacher_recon']  # Circle
        m_student = VIS_MARKERS['student_recon']  # Triangle

        # 1. Teacher Reconstruction Loss (Normal vs Anomaly)
        ax = axes[0, 0]
        if has_detailed and 'train_teacher_recon_normal' in history:
            ax.plot(epochs, history['train_teacher_recon_normal'],
                   color=c_normal, ls='-', lw=2, marker=m_teacher, ms=4, label='Normal')
            ax.plot(epochs, history['train_teacher_recon_anomaly'],
                   color=c_anomaly, ls='-', lw=2, marker=m_teacher, ms=4, label='Anomaly')
        else:
            ax.plot(epochs, history['train_rec_loss'],
                   color=c_teacher, ls='-', lw=2, marker=m_teacher, ms=4, label='Teacher Recon')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss (log scale)')
        ax.set_yscale('log')
        ax.set_title('Teacher Reconstruction Loss (○)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')
        if warmup_epochs > 0:
            ax.axvspan(0.5, warmup_epochs + 0.5, alpha=0.2, color=VIS_COLORS['masked_region'])

        # 2. Student Reconstruction Loss (Normal vs Anomaly)
        ax = axes[0, 1]
        if has_detailed and 'train_student_recon_normal' in history:
            ax.plot(epochs, history['train_student_recon_normal'],
                   color=c_normal, ls='-', lw=2, marker=m_student, ms=4, label='Normal')
            ax.plot(epochs, history['train_student_recon_anomaly'],
                   color=c_anomaly, ls='-', lw=2, marker=m_student, ms=4, label='Anomaly')
        else:
            ax.plot(epochs, history.get('train_student_recon_loss', [0]*len(epochs)),
                   color=c_student, ls='-', lw=2, marker=m_student, ms=4, label='Student Recon')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss (log scale)')
        ax.set_yscale('log')
        ax.set_title('Student Reconstruction Loss (△)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')
        if warmup_epochs > 0:
            ax.axvspan(0.5, warmup_epochs + 0.5, alpha=0.2, color=VIS_COLORS['masked_region'])

        # 3. Discrepancy Loss (Normal vs Anomaly)
        ax = axes[0, 2]
        if 'train_normal_loss' in history and 'train_anomaly_loss' in history:
            ax.plot(epochs, history['train_normal_loss'],
                   color=c_normal, ls='-', lw=2, marker=m_disc, ms=4, label='Normal (minimize)')
            ax.plot(epochs, history['train_anomaly_loss'],
                   color=c_anomaly, ls='-', lw=2, marker=m_disc, ms=4, label='Anomaly (margin)')
        ax.plot(epochs, history['train_disc_loss'],
               color=c_total, ls='--', lw=1.5, label='Total')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Discrepancy Loss (log scale)')
        ax.set_yscale('log')
        ax.set_title('Discrepancy Loss (□)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')
        if warmup_epochs > 0:
            ax.axvspan(0.5, warmup_epochs + 0.5, alpha=0.2, color=VIS_COLORS['masked_region'])

        # 4. Teacher vs Student (Normal samples)
        ax = axes[1, 0]
        if has_detailed:
            ax.plot(epochs, history['train_teacher_recon_normal'],
                   color=c_teacher, ls='-', lw=2, marker=m_teacher, ms=4, label='Teacher (○)')
            ax.plot(epochs, history['train_student_recon_normal'],
                   color=c_student, ls='--', lw=2, marker=m_student, ms=4, label='Student (△)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss (log scale)')
        ax.set_yscale('log')
        ax.set_title('Normal Data: Teacher vs Student', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')
        if warmup_epochs > 0:
            ax.axvspan(0.5, warmup_epochs + 0.5, alpha=0.2, color=VIS_COLORS['masked_region'])

        # 5. Teacher vs Student (Anomaly samples)
        ax = axes[1, 1]
        if has_detailed:
            ax.plot(epochs, history['train_teacher_recon_anomaly'],
                   color=c_teacher, ls='-', lw=2, marker=m_teacher, ms=4, label='Teacher (○)')
            ax.plot(epochs, history['train_student_recon_anomaly'],
                   color=c_student, ls='--', lw=2, marker=m_student, ms=4, label='Student (△)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss (log scale)')
        ax.set_yscale('log')
        ax.set_title('Anomaly Data: Teacher vs Student', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')
        if warmup_epochs > 0:
            ax.axvspan(0.5, warmup_epochs + 0.5, alpha=0.2, color=VIS_COLORS['masked_region'])

        # 6. All Losses Combined
        ax = axes[1, 2]
        ax.plot(epochs, history['train_rec_loss'],
               color=c_teacher, ls='-', lw=2, marker=m_teacher, ms=3, label='Teacher Recon (○)')
        ax.plot(epochs, history['train_disc_loss'],
               color=c_anomaly, ls='-', lw=2, marker=m_disc, ms=3, label='Discrepancy (□)')
        ax.plot(epochs, history['train_loss'],
               color=VIS_COLORS['baseline'], ls='--', lw=1.5, label='Total Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (log scale)')
        ax.set_yscale('log')
        ax.set_title('All Losses Combined', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')
        if warmup_epochs > 0:
            ax.axvspan(0.5, warmup_epochs + 0.5, alpha=0.2, color=VIS_COLORS['masked_region'], label='Warm-up')

        # 7-9. Discriminator metrics (only when use_discriminator=True)
        if has_discriminator:
            d_epochs = epochs[:len(history['train_d_loss'])]

            # 7. D Loss + Accuracy (dual y-axis)
            ax = axes[2, 0]
            ax.plot(d_epochs, history['train_d_loss'],
                   color='#E65100', ls='-', lw=2, label='D Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('D Loss', color='#E65100')
            ax.tick_params(axis='y', labelcolor='#E65100')
            ax.grid(True, alpha=0.3)
            ax2 = ax.twinx()
            ax2.plot(d_epochs, history['train_d_real_acc'],
                    color='#1565C0', ls='--', lw=1.5, label='Real Acc')
            ax2.plot(d_epochs, history['train_d_fake_acc'],
                    color='#C62828', ls='--', lw=1.5, label='Fake Acc')
            ax2.set_ylabel('Accuracy', color='gray')
            ax2.set_ylim([0, 1.05])
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=8)
            ax.set_title('Discriminator Loss & Accuracy', fontweight='bold')

            # 8. Adversarial Loss
            ax = axes[2, 1]
            ax.plot(d_epochs, history['train_adv_loss'],
                   color='#7B1FA2', ls='-', lw=2, marker='D', ms=3, label='Adv Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Adversarial Loss')
            ax.set_title('Student Adversarial Loss', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # 9. Adaptive Lambda
            ax = axes[2, 2]
            ax.plot(d_epochs, history['train_adaptive_lambda'],
                   color='#00695C', ls='-', lw=2, marker='*', ms=4, label='λ_adv')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Adaptive λ')
            ax.set_title('Adaptive Lambda (Gradient Balance)', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Unify y-axis for same-metric subplot pairs
        # Row 0 cols 0-1: Teacher/Student Reconstruction (both MSE Loss)
        _ylim_00 = axes[0, 0].get_ylim()
        _ylim_01 = axes[0, 1].get_ylim()
        _shared_row0 = (min(_ylim_00[0], _ylim_01[0]), max(_ylim_00[1], _ylim_01[1]))
        axes[0, 0].set_ylim(_shared_row0)
        axes[0, 1].set_ylim(_shared_row0)
        # Row 1 cols 0-1: Normal/Anomaly Teacher-vs-Student (both MSE Loss)
        _ylim_10 = axes[1, 0].get_ylim()
        _ylim_11 = axes[1, 1].get_ylim()
        _shared_row1 = (min(_ylim_10[0], _ylim_11[0]), max(_ylim_10[1], _ylim_11[1]))
        axes[1, 0].set_ylim(_shared_row1)
        axes[1, 1].set_ylim(_shared_row1)

        # Add legend for color/marker scheme
        legend_text = ('Color: Blue=Normal, Red=Anomaly, Green=Teacher, Purple=Student\n'
                      'Marker: ○=Teacher Recon, △=Student Recon, □=Discrepancy')
        plt.suptitle(f'Learning Curves\n(Yellow: Warm-up epochs = {warmup_epochs}, Teacher only)\n{legend_text}',
                    fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'learning_curve.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - learning_curve.png")

    def _get_optimal_threshold(self):
        """Get the F1-optimal threshold from ROC curve."""
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(self.pred_data['labels'], self.pred_data['scores'])
        optimal_idx = find_f1_optimal_idx(fpr, tpr, self.pred_data['labels'])
        return thresholds[optimal_idx]

    def _get_scores(self):
        """Get scores array (from pred_data or compute from detailed_data)."""
        return self.pred_data['scores']

    def plot_score_contribution_analysis(self, experiment_dir: str = None):
        """Plot detailed analysis of score contributions from reconstruction and discrepancy

        Creates a comprehensive figure showing:
        (A) Category-wise average score breakdown (stacked bar)
        (B) Contribution ratio percentages by category
        (C) Scatter plot of recon vs disc colored by category
        (D) Weighted contribution violin plots
        (E) Recon contribution KDE by category
        (F) Disc contribution KDE by category
        (G) Normal contribution ratio trend over epochs (if history available)
        (H) Disturbing normal contribution ratio trend over epochs (if history available)
        (I) Anomaly contribution ratio trend over epochs (if history available)

        The epoch-wise contribution trends (G, H, I) show how the ratio of reconstruction
        vs discrepancy contribution changes during training for each sample type.
        All three plots share the same Y-axis range (0-100%) for comparison.

        Args:
            experiment_dir: Path to experiment directory for loading training history
        """
        # Load training history if available
        history = None
        if experiment_dir:
            history_path = os.path.join(experiment_dir, 'training_histories.json')
            if os.path.exists(history_path):
                with open(history_path) as f:
                    history_data = json.load(f)
                    history = history_data.get('0', {})

        # Check for per-sample-type contribution ratio history (new format)
        has_contrib_history = (history is not None and
                              'epoch_recon_ratio_normal' in history and
                              'epoch_recon_ratio_disturbing' in history and
                              'epoch_recon_ratio_anomaly' in history)

        # Compute contributions
        recon_all = self.pred_data['recon_errors']
        disc_all = self.pred_data['discrepancies']
        sample_types = self.pred_data['sample_types']

        contrib_data = compute_score_contributions(recon_all, disc_all, self.config)
        recon_contrib = contrib_data['recon_contrib']
        disc_contrib = contrib_data['disc_contrib']

        # Category masks
        normal_mask = sample_types == 0     # Pure normal
        disturbing_mask = sample_types == 1  # Disturbing normal
        anomaly_mask = sample_types == 2     # Anomaly

        # Category stats
        categories = ['Normal', 'Disturbing', 'Anomaly']
        masks = [normal_mask, disturbing_mask, anomaly_mask]
        cat_colors = [VIS_COLORS['normal'], VIS_COLORS['disturbing'], VIS_COLORS['anomaly']]

        # Check for absolute score history (new format)
        has_abs_score_history_check = (history is not None and
                                      'epoch_recon_score_normal' in history and
                                      'epoch_recon_score_disturbing' in history and
                                      'epoch_recon_score_anomaly' in history)

        # Create figure - 4x3 if absolute score history, 3x3 if only ratio history, else 2x3
        if has_abs_score_history_check:
            fig = plt.figure(figsize=(18, 24))
            gs = GridSpec(4, 3, figure=fig, hspace=0.32, wspace=0.28)
        elif has_contrib_history:
            fig = plt.figure(figsize=(18, 18))
            gs = GridSpec(3, 3, figure=fig, hspace=0.32, wspace=0.28)
        else:
            fig = plt.figure(figsize=(18, 12))
            gs = GridSpec(2, 3, figure=fig, hspace=0.32, wspace=0.28)

        # === (A) Top-Left: Stacked Bar - Average Score Breakdown (IMPROVED) ===
        ax1 = fig.add_subplot(gs[0, 0])

        # Calculate mean contributions per category
        recon_means = [recon_contrib[m].mean() if m.sum() > 0 else 0 for m in masks]
        disc_means = [disc_contrib[m].mean() if m.sum() > 0 else 0 for m in masks]
        totals = [r + d for r, d in zip(recon_means, disc_means)]
        max_total = max(totals) if totals else 1
        _, max_total = _safe_lim(0, max_total)

        y_pos = np.arange(len(categories))
        bar_height = 0.6

        # Stacked horizontal bars
        bars1 = ax1.barh(y_pos, recon_means, bar_height, label='Reconstruction',
                         color=[c + '99' for c in cat_colors], edgecolor='black', linewidth=1)
        bars2 = ax1.barh(y_pos, disc_means, bar_height, left=recon_means, label='Discrepancy',
                         color=cat_colors, edgecolor='black', linewidth=1, hatch='///')

        # Improved x-axis limit to give space for labels
        ax1.set_xlim(-0.05 * max_total, max_total * 1.25)

        # Add value labels with improved positioning
        for i, (r, d) in enumerate(zip(recon_means, disc_means)):
            total = r + d
            # Total label with proper offset
            ax1.text(total + max_total * 0.03, i, f'{total:.4f}',
                    va='center', fontsize=10, fontweight='bold')
            # Inner labels only if bar segment is wide enough
            if r > max_total * 0.08:
                ax1.text(r/2, i, f'{r:.3f}', va='center', ha='center', fontsize=9, color='white')
            if d > max_total * 0.08:
                ax1.text(r + d/2, i, f'{d:.3f}', va='center', ha='center', fontsize=9, color='white')

        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(categories)
        ax1.set_xlabel('Anomaly Score (Weighted Contribution)')
        ax1.set_title('(A) Average Score Breakdown by Category', fontsize=12, fontweight='bold')
        ax1.legend(loc='lower right')
        ax1.grid(axis='x', alpha=0.3)

        # === (B) Top-Middle: Contribution Ratio (%) ===
        ax2 = fig.add_subplot(gs[0, 1])

        # Calculate mean ratios per category
        recon_ratios = [contrib_data['recon_ratio'][m].mean() * 100 if m.sum() > 0 else 0 for m in masks]
        disc_ratios = [contrib_data['disc_ratio'][m].mean() * 100 if m.sum() > 0 else 0 for m in masks]

        x = np.arange(len(categories))
        width = 0.35

        bars1 = ax2.bar(x - width/2, recon_ratios, width, label='Recon %',
                        color='#85C1E9', edgecolor='black', linewidth=1)
        bars2 = ax2.bar(x + width/2, disc_ratios, width, label='Disc %',
                        color='#F1948A', edgecolor='black', linewidth=1)

        # Add value labels
        for bar, val in zip(bars1, recon_ratios):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        for bar, val in zip(bars2, disc_ratios):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax2.set_xticks(x)
        ax2.set_xticklabels(categories)
        ax2.set_ylabel('Contribution (%)')
        ax2.set_ylim(0, 110)
        ax2.set_title('(B) Contribution Ratio by Category', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.grid(axis='y', alpha=0.3)
        ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5)

        # === (C) Top-Right: Scatter Plot ===
        ax3 = fig.add_subplot(gs[0, 2])

        for mask, color, label in zip(masks, cat_colors, categories):
            if mask.sum() > 0:
                ax3.scatter(recon_all[mask], disc_all[mask],
                           c=color, alpha=0.4, s=20, label=f'{label} (n={mask.sum()})')

        ax3.set_xlabel('Reconstruction Loss')
        ax3.set_ylabel('Discrepancy Loss')
        ax3.set_title('(C) Reconstruction vs Discrepancy by Category', fontsize=12, fontweight='bold')
        ax3.legend(loc='upper right')
        ax3.grid(alpha=0.3)

        # Add reference lines for means
        for mask, color in zip(masks, cat_colors):
            if mask.sum() > 0:
                ax3.axvline(recon_all[mask].mean(), color=color, linestyle='--', alpha=0.5)
                ax3.axhline(disc_all[mask].mean(), color=color, linestyle=':', alpha=0.5)

        # === (D) Bottom-Left: Violin Plots with WEIGHTED CONTRIBUTION ===
        ax4 = fig.add_subplot(gs[1, 0])

        # Prepare data for violin plot - using weighted contributions
        violin_data = []
        violin_labels = []
        violin_colors = []

        for cat, mask, color in zip(categories, masks, cat_colors):
            if mask.sum() > 0:
                r_data = recon_contrib[mask]
                d_data = disc_contrib[mask]
                # Add tiny noise if all values identical (violinplot KDE requires variance)
                if np.std(r_data) < 1e-10:
                    r_data = r_data + np.random.normal(0, 1e-8, len(r_data))
                if np.std(d_data) < 1e-10:
                    d_data = d_data + np.random.normal(0, 1e-8, len(d_data))
                violin_data.append(r_data)
                violin_labels.append(f'{cat}\nRecon')
                violin_colors.append(color + '80')
                violin_data.append(d_data)
                violin_labels.append(f'{cat}\nDisc')
                violin_colors.append(color)

        if violin_data:
            parts = ax4.violinplot(violin_data, positions=range(len(violin_data)),
                                   showmeans=True, showmedians=True)
        else:
            parts = {'bodies': []}

        # Color the violin plots
        for i, (pc, color) in enumerate(zip(parts['bodies'], violin_colors)):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)

        ax4.set_xticks(range(len(violin_labels)))
        ax4.set_xticklabels(violin_labels, fontsize=9)
        ax4.set_ylabel('Weighted Contribution')
        ax4.set_title('(D) Contribution Distribution by Category & Type', fontsize=12, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)

        # === (E) Bottom-Middle: Recon Contribution KDE ===
        ax5 = fig.add_subplot(gs[1, 1])
        _KDE_MAX_SAMPLES = 10000  # Subsample for KDE performance

        def _plot_kde(ax, data_array, mask, color, label):
            """Plot KDE for a category, handling degenerate data (zero variance)."""
            if mask.sum() <= 10:
                return
            data = data_array[mask]
            if np.std(data) < 1e-10:
                # All values nearly identical — show as vertical line with annotation
                ax.axvline(data.mean(), color=color, linewidth=2, label=f'{label} (={data.mean():.4f})')
                return
            q1, q99 = np.percentile(data, [1, 99])
            data_clean = data[(data >= q1) & (data <= q99)]
            if len(data_clean) > 10 and np.std(data_clean) > 1e-10:
                kde_data = data_clean if len(data_clean) <= _KDE_MAX_SAMPLES else data_clean[np.linspace(0, len(data_clean)-1, _KDE_MAX_SAMPLES, dtype=int)]
                kde = gaussian_kde(kde_data)
                x_range = np.linspace(data_clean.min(), data_clean.max(), 200)
                ax.plot(x_range, kde(x_range), color=color, linewidth=2, label=label)
                ax.fill_between(x_range, kde(x_range), alpha=0.3, color=color)
            ax.axvline(data.mean(), color=color, linestyle='--', alpha=0.7, linewidth=1.5)

        for mask, color, label in zip(masks, cat_colors, categories):
            _plot_kde(ax5, recon_contrib, mask, color, label)

        ax5.set_xlabel('Recon Contribution')
        ax5.set_ylabel('Density')
        ax5.set_title('(E) Reconstruction Contribution Distribution', fontsize=12, fontweight='bold')
        ax5.legend(loc='upper right')
        ax5.grid(alpha=0.3)

        # === (F) Bottom-Right: Disc Contribution KDE ===
        ax6 = fig.add_subplot(gs[1, 2])

        for mask, color, label in zip(masks, cat_colors, categories):
            _plot_kde(ax6, disc_contrib, mask, color, label)

        ax6.set_xlabel('Disc Contribution')
        ax6.set_ylabel('Density')
        ax6.set_title('(F) Discrepancy Contribution Distribution', fontsize=12, fontweight='bold')
        ax6.legend(loc='upper right')
        ax6.grid(alpha=0.3)

        # === Row 3: Per-Sample-Type Contribution Ratio Trends (if history available) ===
        # Check for absolute score history (new format)
        has_abs_score_history = (history is not None and
                                'epoch_recon_score_normal' in history and
                                'epoch_recon_score_disturbing' in history and
                                'epoch_recon_score_anomaly' in history)

        # Check for raw score history (for recalculation with current scoring mode)
        has_raw_score_history = (history is not None and
                                'epoch_raw_recon_normal' in history and
                                'epoch_raw_disc_normal' in history)

        # Helper function to recalculate contributions from raw scores
        def recalculate_contributions_from_raw(history, score_mode, lambda_disc):
            """Recalculate contribution scores from raw values using current scoring mode.

            This is needed because training history may have been saved with 'default' mode
            even when experiment config specifies 'adaptive' mode.
            """
            raw_recon_normal = np.array(history['epoch_raw_recon_normal'])
            raw_recon_disturbing = np.array(history['epoch_raw_recon_disturbing'])
            raw_recon_anomaly = np.array(history['epoch_raw_recon_anomaly'])
            raw_disc_normal = np.array(history['epoch_raw_disc_normal'])
            raw_disc_disturbing = np.array(history['epoch_raw_disc_disturbing'])
            raw_disc_anomaly = np.array(history['epoch_raw_disc_anomaly'])

            n_epochs = len(raw_recon_normal)
            result = {
                'recon_ratio_normal': [], 'disc_ratio_normal': [],
                'recon_ratio_disturbing': [], 'disc_ratio_disturbing': [],
                'recon_ratio_anomaly': [], 'disc_ratio_anomaly': [],
                'recon_score_normal': [], 'disc_score_normal': [],
                'recon_score_disturbing': [], 'disc_score_disturbing': [],
                'recon_score_anomaly': [], 'disc_score_anomaly': [],
            }

            for i in range(n_epochs):
                # Compute per-epoch raw values
                raw_recons = [raw_recon_normal[i], raw_recon_disturbing[i], raw_recon_anomaly[i]]
                raw_discs = [raw_disc_normal[i], raw_disc_disturbing[i], raw_disc_anomaly[i]]

                # Compute overall mean for adaptive lambda (weighted by test set ratios: 65%, 15%, 20%)
                overall_recon_mean = 0.65 * raw_recon_normal[i] + 0.15 * raw_recon_disturbing[i] + 0.20 * raw_recon_anomaly[i]
                overall_disc_mean = 0.65 * raw_disc_normal[i] + 0.15 * raw_disc_disturbing[i] + 0.20 * raw_disc_anomaly[i]

                for j, (sample_type, recon, disc) in enumerate(zip(
                    ['normal', 'disturbing', 'anomaly'], raw_recons, raw_discs
                )):
                    if score_mode == 'adaptive':
                        adaptive_lambda = overall_recon_mean / (overall_disc_mean + 1e-8)
                        recon_contrib = recon
                        disc_contrib = adaptive_lambda * disc
                    else:  # default
                        recon_contrib = recon
                        disc_contrib = lambda_disc * disc

                    total = recon_contrib + disc_contrib + 1e-8
                    recon_ratio = (recon_contrib / total) * 100
                    disc_ratio = (disc_contrib / total) * 100

                    result[f'recon_ratio_{sample_type}'].append(recon_ratio)
                    result[f'disc_ratio_{sample_type}'].append(disc_ratio)
                    result[f'recon_score_{sample_type}'].append(recon_contrib)
                    result[f'disc_score_{sample_type}'].append(disc_contrib)

            # Convert to numpy arrays
            for key in result:
                result[key] = np.array(result[key])

            return result

        if has_contrib_history:
            epochs = history.get('epoch', list(range(len(history['epoch_recon_ratio_normal']))))

            # Recalculate contributions from raw scores if available (fixes scoring mode mismatch)
            if has_raw_score_history:
                score_mode = getattr(self.config, 'anomaly_score_mode', 'default')
                lambda_disc = getattr(self.config, 'lambda_disc', 0.5)
                recalc = recalculate_contributions_from_raw(history, score_mode, lambda_disc)
                recon_ratio_normal = recalc['recon_ratio_normal']
                disc_ratio_normal = recalc['disc_ratio_normal']
                recon_ratio_disturbing = recalc['recon_ratio_disturbing']
                disc_ratio_disturbing = recalc['disc_ratio_disturbing']
                recon_ratio_anomaly = recalc['recon_ratio_anomaly']
                disc_ratio_anomaly = recalc['disc_ratio_anomaly']
            else:
                # Fallback to stored values (may not reflect current scoring mode)
                recon_ratio_normal = np.array(history['epoch_recon_ratio_normal'])
                disc_ratio_normal = np.array(history['epoch_disc_ratio_normal'])
                recon_ratio_disturbing = np.array(history['epoch_recon_ratio_disturbing'])
                disc_ratio_disturbing = np.array(history['epoch_disc_ratio_disturbing'])
                recon_ratio_anomaly = np.array(history['epoch_recon_ratio_anomaly'])
                disc_ratio_anomaly = np.array(history['epoch_disc_ratio_anomaly'])

            # Length-align epochs vs ratio arrays (2026-05-30 defensive fix).
            # Runs RESUMED before the trainer/checkpoint off-by-one fix saved the
            # contribution-ratio history 1 entry short (e.g. epoch=500 but
            # epoch_recon_ratio_*=499). matplotlib stackplot requires equal-length
            # operands, so trim every series to the common length. No effect on
            # consistent histories. (The save-/load-side fixes in
            # run_base_experiments.py prevent the shortfall for new runs; this keeps
            # the chart renderable for already-completed experiments.)
            _ratio_arrays = [recon_ratio_normal, disc_ratio_normal,
                             recon_ratio_disturbing, disc_ratio_disturbing,
                             recon_ratio_anomaly, disc_ratio_anomaly]
            _nL = min([len(epochs)] + [len(a) for a in _ratio_arrays])
            epochs = np.asarray(epochs)[:_nL]
            recon_ratio_normal = recon_ratio_normal[:_nL]
            disc_ratio_normal = disc_ratio_normal[:_nL]
            recon_ratio_disturbing = recon_ratio_disturbing[:_nL]
            disc_ratio_disturbing = disc_ratio_disturbing[:_nL]
            recon_ratio_anomaly = recon_ratio_anomaly[:_nL]
            disc_ratio_anomaly = disc_ratio_anomaly[:_nL]

            # Common styling
            colors_stack = ['#85C1E9', '#F1948A']  # Light blue for Recon, Light red for Disc

            # === (G) Normal Contribution Ratio Trend ===
            ax7 = fig.add_subplot(gs[2, 0])
            ax7.stackplot(epochs, recon_ratio_normal, disc_ratio_normal,
                         labels=['Recon %', 'Disc %'],
                         colors=colors_stack, alpha=0.8)
            ax7.set_xlabel('Epoch')
            ax7.set_ylabel('Contribution (%)')
            ax7.set_ylim(0, 100)
            ax7.set_title(f'(G) Normal Contribution Trend (n={normal_mask.sum()})', fontsize=12, fontweight='bold')
            ax7.legend(loc='upper right', fontsize=9)
            ax7.grid(alpha=0.3)
            ax7.axhline(y=50, color='gray', linestyle='--', alpha=0.5)

            # === (H) Disturbing Normal Contribution Ratio Trend ===
            ax8 = fig.add_subplot(gs[2, 1])
            ax8.stackplot(epochs, recon_ratio_disturbing, disc_ratio_disturbing,
                         labels=['Recon %', 'Disc %'],
                         colors=colors_stack, alpha=0.8)
            ax8.set_xlabel('Epoch')
            ax8.set_ylabel('Contribution (%)')
            ax8.set_ylim(0, 100)
            ax8.set_title(f'(H) Disturbing Normal Trend (n={disturbing_mask.sum()})', fontsize=12, fontweight='bold')
            ax8.legend(loc='upper right', fontsize=9)
            ax8.grid(alpha=0.3)
            ax8.axhline(y=50, color='gray', linestyle='--', alpha=0.5)

            # === (I) Anomaly Contribution Ratio Trend ===
            ax9 = fig.add_subplot(gs[2, 2])
            ax9.stackplot(epochs, recon_ratio_anomaly, disc_ratio_anomaly,
                         labels=['Recon %', 'Disc %'],
                         colors=colors_stack, alpha=0.8)
            ax9.set_xlabel('Epoch')
            ax9.set_ylabel('Contribution (%)')
            ax9.set_ylim(0, 100)
            ax9.set_title(f'(I) Anomaly Contribution Trend (n={anomaly_mask.sum()})', fontsize=12, fontweight='bold')
            ax9.legend(loc='upper right', fontsize=9)
            ax9.grid(alpha=0.3)
            ax9.axhline(y=50, color='gray', linestyle='--', alpha=0.5)

        # === Row 4: Per-Sample-Type Absolute Score Trends (if history available) ===
        if has_abs_score_history:
            epochs = np.array(history.get('epoch', list(range(len(history['epoch_recon_score_normal'])))))

            # Get absolute scores by sample type
            # Recalculate from raw scores if available (fixes scoring mode mismatch)
            if has_raw_score_history:
                score_mode = getattr(self.config, 'anomaly_score_mode', 'default')
                lambda_disc = getattr(self.config, 'lambda_disc', 0.5)
                recalc = recalculate_contributions_from_raw(history, score_mode, lambda_disc)
                recon_score_normal = recalc['recon_score_normal']
                disc_score_normal = recalc['disc_score_normal']
                recon_score_disturbing = recalc['recon_score_disturbing']
                disc_score_disturbing = recalc['disc_score_disturbing']
                recon_score_anomaly = recalc['recon_score_anomaly']
                disc_score_anomaly = recalc['disc_score_anomaly']
            else:
                # Fallback to stored values (may not reflect current scoring mode)
                recon_score_normal = np.array(history['epoch_recon_score_normal'])
                disc_score_normal = np.array(history['epoch_disc_score_normal'])
                recon_score_disturbing = np.array(history['epoch_recon_score_disturbing'])
                disc_score_disturbing = np.array(history['epoch_disc_score_disturbing'])
                recon_score_anomaly = np.array(history['epoch_recon_score_anomaly'])
                disc_score_anomaly = np.array(history['epoch_disc_score_anomaly'])

            # Length-align epochs vs score arrays (2026-05-30 defensive fix — same
            # off-by-one as the ratio block above; epoch_mask below would otherwise
            # index a shorter array). No effect on consistent histories.
            _score_arrays = [recon_score_normal, disc_score_normal,
                             recon_score_disturbing, disc_score_disturbing,
                             recon_score_anomaly, disc_score_anomaly]
            _nL2 = min([len(epochs)] + [len(a) for a in _score_arrays])
            epochs = np.asarray(epochs)[:_nL2]
            recon_score_normal = recon_score_normal[:_nL2]
            disc_score_normal = disc_score_normal[:_nL2]
            recon_score_disturbing = recon_score_disturbing[:_nL2]
            disc_score_disturbing = disc_score_disturbing[:_nL2]
            recon_score_anomaly = recon_score_anomaly[:_nL2]
            disc_score_anomaly = disc_score_anomaly[:_nL2]

            # Filter to show only epoch >= 5
            start_epoch = 5
            epoch_mask = epochs >= start_epoch
            epochs_filtered = epochs[epoch_mask]
            recon_normal_f = recon_score_normal[epoch_mask]
            disc_normal_f = disc_score_normal[epoch_mask]
            recon_disturbing_f = recon_score_disturbing[epoch_mask]
            disc_disturbing_f = disc_score_disturbing[epoch_mask]
            recon_anomaly_f = recon_score_anomaly[epoch_mask]
            disc_anomaly_f = disc_score_anomaly[epoch_mask]

            # Compute unified y-axis limits across all three plots (from epoch 5)
            all_total_scores = np.concatenate([
                recon_normal_f + disc_normal_f,
                recon_disturbing_f + disc_disturbing_f,
                recon_anomaly_f + disc_anomaly_f
            ])
            y_max = all_total_scores.max() * 1.1
            _, y_max = _safe_lim(0, y_max)

            # Common styling for stacked area
            colors_abs_stack = ['#85C1E9', '#F1948A']  # Light blue for Recon, Light red for Disc

            # === (J) Normal Absolute Score Trend (Area) ===
            ax10 = fig.add_subplot(gs[3, 0])
            ax10.stackplot(epochs_filtered, recon_normal_f, disc_normal_f,
                          labels=['Recon Score', 'Disc Score'],
                          colors=colors_abs_stack, alpha=0.8)
            ax10.set_xlabel('Epoch')
            ax10.set_ylabel('Anomaly Score')
            ax10.set_ylim(0, y_max)
            ax10.set_xlim(start_epoch, epochs_filtered[-1] if len(epochs_filtered) > 0 else start_epoch)
            ax10.set_title(f'(J) Normal Absolute Score (n={normal_mask.sum()})', fontsize=12, fontweight='bold')
            ax10.legend(loc='upper right', fontsize=9)
            ax10.grid(alpha=0.3)

            # === (K) Disturbing Normal Absolute Score Trend (Area) ===
            ax11 = fig.add_subplot(gs[3, 1])
            ax11.stackplot(epochs_filtered, recon_disturbing_f, disc_disturbing_f,
                          labels=['Recon Score', 'Disc Score'],
                          colors=colors_abs_stack, alpha=0.8)
            ax11.set_xlabel('Epoch')
            ax11.set_ylabel('Anomaly Score')
            ax11.set_ylim(0, y_max)
            ax11.set_xlim(start_epoch, epochs_filtered[-1] if len(epochs_filtered) > 0 else start_epoch)
            ax11.set_title(f'(K) Disturbing Absolute Score (n={disturbing_mask.sum()})', fontsize=12, fontweight='bold')
            ax11.legend(loc='upper right', fontsize=9)
            ax11.grid(alpha=0.3)

            # === (L) Anomaly Absolute Score Trend (Area) ===
            ax12 = fig.add_subplot(gs[3, 2])
            ax12.stackplot(epochs_filtered, recon_anomaly_f, disc_anomaly_f,
                          labels=['Recon Score', 'Disc Score'],
                          colors=colors_abs_stack, alpha=0.8)
            ax12.set_xlabel('Epoch')
            ax12.set_ylabel('Anomaly Score')
            ax12.set_ylim(0, y_max)
            ax12.set_xlim(start_epoch, epochs_filtered[-1] if len(epochs_filtered) > 0 else start_epoch)
            ax12.set_title(f'(L) Anomaly Absolute Score (n={anomaly_mask.sum()})', fontsize=12, fontweight='bold')
            ax12.legend(loc='upper right', fontsize=9)
            ax12.grid(alpha=0.3)

        # Add overall title with scoring mode info
        score_mode = contrib_data['score_mode']
        mode_params = contrib_data['mode_params']

        if score_mode == 'default':
            param_str = f"λ_disc={mode_params.get('lambda_disc', 0.5):.2f}"
        elif score_mode == 'adaptive':
            param_str = f"adaptive_λ={mode_params.get('adaptive_lambda', 0):.3f}"
        elif score_mode == 'ratio_weighted':
            param_str = f"disc_median={mode_params.get('disc_median', 0):.4f}"
        else:
            param_str = ""

        fig.suptitle(f'Score Contribution Analysis\nScoring Mode: {score_mode} ({param_str})',
                    fontsize=14, fontweight='bold', y=0.99 if has_contrib_history else 0.98)

        # Sample counts
        n_normal = normal_mask.sum()
        n_disturbing = disturbing_mask.sum()
        n_anomaly = anomaly_mask.sum()
        fig.text(0.02, 0.01, f'Samples: Normal={n_normal}, Disturbing={n_disturbing}, Anomaly={n_anomaly}',
                fontsize=10, ha='left')

        plt.tight_layout(rect=[0.01, 0.02, 0.99, 0.96])
        plt.savefig(os.path.join(self.output_dir, 'best_model_score_contribution.png'),
                   dpi=150)
        plt.close()
        print("  - best_model_score_contribution.png")

    def plot_performance_by_pa_k(self):
        """Plot PA%K performance analysis with varying K (mean-based).

        Creates two subplots:
        1. All PA%K Methods Comparison: Grouped bar chart comparing F1 scores
           at K=0, 10, 20, 50, 80, 100 for each scoring method.
        2. F1 Score vs K: Line plot showing how F1 score changes as K varies
           from 0 to 100 (step=5), for each scoring method. Includes AUF1
           (Area Under F1 curve) in legend.

        Uses mean-aggregated point-level scores and optimal threshold per method.
        """
        raw_dataset = self.test_loader.dataset if hasattr(self.test_loader, 'dataset') else None
        base_dataset = _unwrap_subset(raw_dataset) if raw_dataset is not None else None
        can_use = (
            base_dataset is not None and
            hasattr(base_dataset, 'anomaly_regions') and
            hasattr(base_dataset, 'point_labels') and
            hasattr(base_dataset, 'window_start_indices') and
            len(base_dataset.anomaly_regions) > 0
        )
        if not can_use:
            print("  - performance_by_PA_K.png (SKIPPED: no segment info)")
            return

        # Prepare data
        patch_recon = self.pred_data['recon_errors']
        patch_student = self.pred_data['student_errors']
        patch_disc = self.pred_data['discrepancies']
        patch_combined = self.pred_data['patch_scores']

        total_length = len(base_dataset.point_labels)
        point_labels = np.array(base_dataset.point_labels)
        anomaly_regions = base_dataset.anomaly_regions
        _, window_start_indices = _get_subset_window_indices(raw_dataset)

        n_windows = self.pred_data.get('n_windows', len(window_start_indices))
        num_patches = self.pred_data.get('num_patches', getattr(self.config, 'num_patches', 10))

        scoring_methods = [
            ('Anomaly Score', patch_combined.reshape(n_windows, num_patches), VIS_COLORS['total']),
            ('Discrepancy', patch_disc.reshape(n_windows, num_patches), VIS_COLORS['discrepancy']),
            ('Teacher Recon', patch_recon.reshape(n_windows, num_patches), VIS_COLORS['teacher']),
            ('Student Recon', patch_student.reshape(n_windows, num_patches), VIS_COLORS['student']),
        ]

        # Build aggregation map
        flat_t, flat_wp, coverage, covered = _build_aggregation_map(
            window_start_indices, self.config.patch_size, num_patches, total_length,
        )
        eval_mask = np.ones(total_length, dtype=bool)

        # K values
        k_bar = [0, 10, 20, 50, 80, 100]
        k_line = list(range(0, 101, 5))
        all_k = sorted(set(k_bar + k_line))

        # Compute metrics at each K for each scoring method
        method_results = {}
        for name, patch_scores, color in scoring_methods:
            # Mean-aggregate to point-level
            point_scores = _aggregate_with_map(
                patch_scores.ravel(), flat_t, flat_wp, coverage, covered, total_length, method='mean'
            )
            point_scores = np.nan_to_num(point_scores, nan=0.0)

            if len(np.unique(point_labels)) < 2:
                method_results[name] = {k: (0.0, 0.0, 0.0) for k in all_k}
                continue

            # Find optimal threshold
            from sklearn.metrics import roc_curve as sk_roc_curve
            fpr, tpr, thresholds = sk_roc_curve(point_labels, point_scores)
            optimal_idx = find_f1_optimal_idx(fpr, tpr, point_labels)
            threshold = thresholds[optimal_idx]

            k_metrics = {}
            for k in all_k:
                pa_rp = compute_pa_k_roc_prc_from_mean_scores(
                    point_scores, point_labels, anomaly_regions, k, eval_mask
                )
                pa_m = compute_pa_k_metrics_from_mean_scores(
                    point_scores, point_labels, anomaly_regions, threshold, k, eval_mask
                )
                k_metrics[k] = (pa_rp['roc_auc'], pa_rp['prc_auc'], pa_m['f1'])
            method_results[name] = k_metrics

        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

        # --- Subplot 1: Grouped bar chart ---
        x = np.arange(len(k_bar))
        n_methods = len(scoring_methods)
        width = 0.8 / n_methods

        for i, (name, _, color) in enumerate(scoring_methods):
            f1_vals = [method_results[name][k][2] for k in k_bar]
            offset = (i - (n_methods - 1) / 2) * width
            bars = ax1.bar(x + offset, f1_vals, width, label=name, color=color, alpha=0.8)
            for bar, val in zip(bars, f1_vals):
                if val > 0.01:
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                            f'{val:.3f}', ha='center', va='bottom', fontsize=7, rotation=45)

        ax1.set_xlabel('PA%K')
        ax1.set_ylabel('Optimal F1 Score')
        ax1.set_title('All PA%K Methods Comparison', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'K={k}' for k in k_bar])
        max_f1 = max(
            max(method_results[name][k][2] for k in k_bar)
            for name, _, _ in scoring_methods
        )
        _, max_f1_safe = _safe_lim(0, max_f1)
        ax1.set_ylim(0, min(1.05, max_f1_safe * 1.2 + 0.05))
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3, axis='y')

        # --- Subplot 2: F1 vs K line plot ---
        for name, _, color in scoring_methods:
            f1_vals = [method_results[name][k][2] for k in k_line]
            auf1 = np.trapz(f1_vals, k_line) / 100.0
            ax2.plot(k_line, f1_vals, color=color, lw=2, marker='o', markersize=3,
                    label=f'{name} (AUF1={auf1:.4f})')

        ax2.set_xlabel('K (PA%K)')
        ax2.set_ylabel('Optimal F1 Score')
        ax2.set_title('F1 Score with Varying K', fontweight='bold')
        ax2.set_xlim(-2, 102)
        max_f1_line = max(
            max(method_results[name][k][2] for k in k_line)
            for name, _, _ in scoring_methods
        )
        _, max_f1_line_safe = _safe_lim(0, max_f1_line)
        ax2.set_ylim(0, min(1.05, max_f1_line_safe * 1.15 + 0.02))
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3)

        for k_ref in [10, 20, 50, 80]:
            ax2.axvline(x=k_ref, color='gray', linestyle=':', alpha=0.3)

        scoring_mode = getattr(self.config, 'anomaly_score_mode', 'default')
        fig.suptitle(f'PA%K Performance Analysis (Scoring: {scoring_mode})',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_by_PA_K.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()
        print("  - performance_by_PA_K.png")

    def plot_pa_k_auc_summary(self, experiment_dir: str = None):
        """Plot PA%K AUC summary: K vs metric curves + Adaptive vs Teacher bar chart.

        Creates a 2x3 grid of K-sweep curves (PRC-AUC, ROC-AUC, F1, F1_T, Precision, Recall)
        and a separate bar chart comparing Adaptive vs Teacher PA%K AUC.

        Reads PA%K AUC data from experiment_metadata.json.
        """
        if experiment_dir is None:
            print("  - pa_k_auc_summary.png (SKIPPED: no experiment_dir)")
            return

        meta_path = os.path.join(experiment_dir, 'experiment_metadata.json')
        if not os.path.exists(meta_path):
            print("  - pa_k_auc_summary.png (SKIPPED: no metadata)")
            return

        with open(meta_path, 'r') as f:
            metadata = json.load(f)

        metrics = metadata.get('metrics', {})
        teacher_metrics = metadata.get('teacher_recon_metrics', {})

        # Check if PA%K AUC data exists
        if 'pak_auc_prc_auc' not in metrics:
            print("  - pa_k_auc_summary.png (SKIPPED: no PA%K AUC data)")
            return

        # Collect PA%K per-K data (step=5) for curve plots
        k_values = list(range(0, 101, 5))
        metric_names = ['prc_auc', 'roc_auc', 'f1', 'precision', 'recall']
        metric_labels = ['PRC-AUC', 'ROC-AUC', 'F1', 'Precision', 'Recall']

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes_flat = axes.flatten()

        for i, (mname, mlabel) in enumerate(zip(metric_names, metric_labels)):
            ax = axes_flat[i]
            # Adaptive
            if mname in ('prc_auc', 'roc_auc'):
                adaptive_vals = [metrics.get(f'pa_{k}_{mname}', 0) for k in k_values]
                teacher_vals = [teacher_metrics.get(f'pa_{k}_{mname}', 0) for k in k_values]
            else:
                adaptive_vals = [metrics.get(f'pa_{k}_{mname}', 0) for k in k_values]
                teacher_vals = [teacher_metrics.get(f'pa_{k}_{mname}', 0) for k in k_values]

            adaptive_auc = metrics.get(f'pak_auc_{mname}', 0)
            teacher_auc = teacher_metrics.get(f'pak_auc_{mname}', 0)

            # For F1 metrics, show best vs raw AUC in legend
            if mname in ('f1', 'precision', 'recall'):
                adaptive_raw = metrics.get(f'pak_auc_{mname}_raw', 0)
                teacher_raw = teacher_metrics.get(f'pak_auc_{mname}_raw', 0)
                a_label = f'Adaptive (best={adaptive_auc:.4f}, raw={adaptive_raw:.4f})'
                t_label = f'Teacher (best={teacher_auc:.4f}, raw={teacher_raw:.4f})'
            else:
                a_label = f'Adaptive (AUC={adaptive_auc:.4f})'
                t_label = f'Teacher (AUC={teacher_auc:.4f})'

            ax.plot(k_values, adaptive_vals, color='#e74c3c', lw=2, marker='o', markersize=3,
                   label=a_label)
            ax.plot(k_values, teacher_vals, color='#3498db', lw=2, marker='s', markersize=3,
                   label=t_label)
            ax.fill_between(k_values, adaptive_vals, alpha=0.1, color='#e74c3c')
            ax.fill_between(k_values, teacher_vals, alpha=0.1, color='#3498db')

            ax.set_xlabel('K (%)')
            ax.set_ylabel(mlabel)
            ax.set_title(f'{mlabel} vs K', fontweight='bold')
            ax.set_xlim(-2, 102)
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)

        # Last subplot: PA%K AUC bar comparison (best + raw for F1 metrics)
        ax_bar = axes_flat[5]
        auc_metrics = ['prc_auc', 'roc_auc', 'f1', 'f1_t', 'precision', 'recall']
        auc_labels = ['PRC', 'ROC', 'F1\n(best)', 'F1_T\n(best)', 'Prec\n(best)', 'Rec\n(best)']
        adaptive_aucs = [metrics.get(f'pak_auc_{m}', 0) for m in auc_metrics]
        teacher_aucs = [teacher_metrics.get(f'pak_auc_{m}', 0) for m in auc_metrics]

        # Raw F1 metrics (fixed threshold, for comparison)
        raw_metrics = ['f1_raw', 'f1_t_raw', 'precision_raw', 'recall_raw']
        raw_labels = ['F1\n(raw)', 'F1_T\n(raw)', 'Prec\n(raw)', 'Rec\n(raw)']
        adaptive_raws = [metrics.get(f'pak_auc_{m}', 0) for m in raw_metrics]
        teacher_raws = [teacher_metrics.get(f'pak_auc_{m}', 0) for m in raw_metrics]

        all_labels = auc_labels + raw_labels
        all_adaptive = adaptive_aucs + adaptive_raws
        all_teacher = teacher_aucs + teacher_raws

        x = np.arange(len(all_labels))
        width = 0.35
        bars1 = ax_bar.bar(x - width/2, all_adaptive, width, label='Adaptive', color='#e74c3c', alpha=0.8)
        bars2 = ax_bar.bar(x + width/2, all_teacher, width, label='Teacher', color='#3498db', alpha=0.8)

        for bar, val in zip(bars1, all_adaptive):
            if val > 0.01:
                ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=7)
        for bar, val in zip(bars2, all_teacher):
            if val > 0.01:
                ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=7)

        # Visual separator between best and raw
        ax_bar.axvline(x=5.5, color='gray', linestyle=':', alpha=0.5)

        ax_bar.set_ylabel('PA%K AUC')
        ax_bar.set_title('PA%K AUC: Adaptive vs Teacher (best + raw)', fontweight='bold')
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(all_labels, fontsize=7)
        ax_bar.legend(fontsize=9)
        ax_bar.grid(True, alpha=0.3, axis='y')

        fig.suptitle('PA%K AUC Summary (K=0..100)', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'pa_k_auc_summary.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()
        print("  - pa_k_auc_summary.png")

    # =================================================================
    # Feature-level visualizations (use self.disc_per_feature)
    # =================================================================

    def plot_feature_dominance(self):
        """Per-feature discrepancy share + Herfindahl index by sample type.

        Shows which features dominate the anomaly score. High Herfindahl = few
        features dominating (potential feature dominance problem).
        """
        dpf = getattr(self, 'disc_per_feature', None)
        if dpf is None or dpf.ndim != 2:
            print("  - Skipping feature_dominance.png (no disc_per_feature)")
            return

        n_windows, n_features = dpf.shape
        sample_types = self.pred_data.get('sample_types')
        if sample_types is None:
            return

        # Map window-level sample types (pred_data is patch-level, need window-level)
        n_patches = self.num_patches
        if len(sample_types) == n_windows * n_patches:
            # Patch-level → window-level: take the max per window (anomaly > disturbing > normal)
            win_types = sample_types.reshape(n_windows, n_patches).max(axis=1)
        elif len(sample_types) == n_windows:
            win_types = sample_types
        else:
            print(f"  - Skipping feature_dominance.png (sample_types shape mismatch: {len(sample_types)} vs {n_windows})")
            return

        all_types = [(0, 'Normal', VIS_COLORS['normal']),
                     (1, 'Disturbing', VIS_COLORS['disturbing']),
                     (2, 'Anomaly', VIS_COLORS['anomaly'])]
        # Only show panels for types that have data
        present_types = [(t_idx, t_name, t_color) for t_idx, t_name, t_color in all_types
                         if (win_types == t_idx).sum() > 0]
        if len(present_types) == 0:
            print("  - Skipping feature_dominance.png (no windows with valid sample types)")
            return

        fig, axes = plt.subplots(1, len(present_types), figsize=(6 * len(present_types), 6))
        if len(present_types) == 1:
            axes = [axes]

        for ax, (t_idx, t_name, t_color) in zip(axes, present_types):
            mask = (win_types == t_idx)
            subset = dpf[mask]  # (n_subset, F)
            feature_means = subset.mean(axis=0)  # (F,)
            total = feature_means.sum() + 1e-10
            shares = feature_means / total  # fractional share

            # Herfindahl index: sum of squared shares (1/F = uniform, 1.0 = single feature)
            hhi = float((shares ** 2).sum())

            bars = ax.bar(range(n_features), shares * 100, color=t_color, alpha=0.7)
            ax.set_xlabel('Feature Index')
            ax.set_ylabel('Disc Share (%)')
            ax.set_title(f'{t_name} (n={mask.sum()}, HHI={hhi:.3f})', fontweight='bold')
            ax.set_xticks(range(n_features))
            ax.grid(True, alpha=0.3, axis='y')

            # Highlight dominant features (>2x uniform)
            uniform_share = 100.0 / n_features
            for i, bar in enumerate(bars):
                if shares[i] * 100 > 2 * uniform_share:
                    bar.set_edgecolor('red')
                    bar.set_linewidth(2)

        fig.suptitle('Per-Feature Discrepancy Dominance', fontsize=14, fontweight='bold')
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, 'feature_dominance.png'), dpi=150)
        plt.close(fig)
        print("  - feature_dominance.png")

    def plot_feature_extremes(self):
        """Max/mean ratio per feature by sample type — identifies extreme outlier features."""
        dpf = getattr(self, 'disc_per_feature', None)
        if dpf is None or dpf.ndim != 2:
            print("  - Skipping feature_extremes.png (no disc_per_feature)")
            return

        n_windows, n_features = dpf.shape
        sample_types = self.pred_data.get('sample_types')
        if sample_types is None:
            return

        n_patches = self.num_patches
        if len(sample_types) == n_windows * n_patches:
            win_types = sample_types.reshape(n_windows, n_patches).max(axis=1)
        elif len(sample_types) == n_windows:
            win_types = sample_types
        else:
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(n_features)
        width = 0.25
        type_names = ['Normal', 'Disturbing', 'Anomaly']
        type_colors_list = [VIS_COLORS['normal'], VIS_COLORS['disturbing'], VIS_COLORS['anomaly']]

        for i, (t_idx, t_name, t_color) in enumerate(zip([0, 1, 2], type_names, type_colors_list)):
            mask = (win_types == t_idx)
            if mask.sum() == 0:
                continue
            subset = dpf[mask]
            feat_mean = subset.mean(axis=0) + 1e-10
            feat_max = subset.max(axis=0)
            ratio = feat_max / feat_mean
            ax.bar(x + i * width, ratio, width, label=t_name, color=t_color, alpha=0.7)

        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Max / Mean Ratio')
        ax.set_title('Per-Feature Extreme Value Ratio (Max/Mean)', fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels([f'F{i}' for i in range(n_features)], fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='10x threshold')

        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, 'feature_extremes.png'), dpi=150)
        plt.close(fig)
        print("  - feature_extremes.png")

    def plot_feature_profile(self):
        """Per-feature discrepancy box plot by sample type — compact distribution overview."""
        dpf = getattr(self, 'disc_per_feature', None)
        if dpf is None or dpf.ndim != 2:
            print("  - Skipping feature_profile.png (no disc_per_feature)")
            return

        n_windows, n_features = dpf.shape
        sample_types = self.pred_data.get('sample_types')
        if sample_types is None:
            return

        n_patches = self.num_patches
        if len(sample_types) == n_windows * n_patches:
            win_types = sample_types.reshape(n_windows, n_patches).max(axis=1)
        elif len(sample_types) == n_windows:
            win_types = sample_types
        else:
            return

        # Subsample for performance (box plots with >100K points are slow)
        MAX_PER_TYPE = 20000
        fig, axes = plt.subplots(1, n_features, figsize=(max(3 * n_features, 12), 5), sharey=True)
        if n_features == 1:
            axes = [axes]

        type_names = ['Norm', 'Dist', 'Anom']
        type_colors_list = [VIS_COLORS['normal'], VIS_COLORS['disturbing'], VIS_COLORS['anomaly']]

        # Clip at p99.5 for y-axis
        clip_val = float(np.percentile(dpf, 99.5))
        _, clip_val = _safe_lim(0, clip_val)

        for f_idx, ax in enumerate(axes):
            box_data = []
            labels = []
            colors = []
            for t_idx, t_name, t_color in zip([0, 1, 2], type_names, type_colors_list):
                mask = (win_types == t_idx)
                if mask.sum() == 0:
                    continue
                vals = dpf[mask, f_idx]
                if len(vals) > MAX_PER_TYPE:
                    vals = vals[np.random.choice(len(vals), MAX_PER_TYPE, replace=False)]
                box_data.append(vals)
                labels.append(t_name)
                colors.append(t_color)

            if box_data:
                bp = ax.boxplot(box_data, labels=labels, patch_artist=True, showfliers=False)
                for patch, c in zip(bp['boxes'], colors):
                    patch.set_facecolor(c)
                    patch.set_alpha(0.6)
            ax.set_title(f'F{f_idx}', fontsize=9)
            ax.set_ylim(bottom=0, top=clip_val)
            ax.grid(True, alpha=0.3, axis='y')

        axes[0].set_ylabel('Discrepancy')
        fig.suptitle('Per-Feature Discrepancy Profile by Sample Type', fontsize=14, fontweight='bold')
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, 'feature_profile.png'), dpi=150)
        plt.close(fig)
        print("  - feature_profile.png")

    def _select_best_epoch_for_viz(self, epochs):
        """Best epoch by pak_auc_f1. For official runs, FORCE selection to post-warmup
        epochs only (epoch > teacher_only_warmup_epochs): during warmup the student /
        output-discrepancy is untrained, so a pre-warmup 'best' is misleading for the
        score/discrepancy visualizations. Falls back to all epochs if no post-warmup
        eval exists. (Affects VISUALIZATION only — training/metric best-epoch unchanged.)"""
        cands = list(epochs)
        if bool(getattr(self.config, 'official', False)):
            warm = int(getattr(self.config, 'teacher_only_warmup_epochs', -1) or -1)
            if warm < 0:
                warm = int(getattr(self.config, 'num_epochs', 0) or 0) // 2
            post = [e for e in epochs if int(e.get('epoch', 0)) > warm]
            if post:
                cands = post
        return max(cands, key=lambda e: e.get('pak_auc_f1', 0))

    def plot_anomaly_threshold(self, experiment_dir: str = None):
        """Plot anomaly score timeline with each component on its own subplot.

        Components are scaled to their actual contribution to anomaly_score
        (single source: mae_anomaly.scoring.compute_adaptive_components). Since
        2026-06-01 FM is NOT part of the score and the discrepancy is
        down-weighted to recon:disc = ratio:1 (default 4:1):
            scaled_disc = disc × (recon.mean / disc.mean) / ratio   (== student_error)
            anomaly_score = recon + scaled_disc

        Saved as anomaly_threshold.png in self.output_dir.

        Source data: epoch_scores/epoch_XXX_scores.npz (best epoch).
        Point labels: test_loader.dataset.point_labels.
        """
        # 1. Resolve dataset directory (experiment_dir param ≠ exp root here —
        #    output_dir is .../<dataset_sub>/visualization/best_model/).
        dataset_dir = os.path.dirname(os.path.dirname(self.output_dir))

        # 2. Find best epoch + threshold
        metrics_path = os.path.join(dataset_dir, 'epoch_metrics.json')
        if not os.path.exists(metrics_path):
            print(f"  [anomaly_threshold] no epoch_metrics.json at {dataset_dir}")
            return
        epoch_data = json.load(open(metrics_path))
        best = self._select_best_epoch_for_viz(epoch_data['epochs'])
        best_epoch = best['epoch']
        threshold = best.get('optimal_threshold', None)
        pak_auc_f1 = best.get('pak_auc_f1', 0.0)
        pak_auc_prc = best.get('pak_auc_prc_auc', None)

        # 3. Load scores npz
        npz_path = os.path.join(dataset_dir, 'epoch_scores',
                                 f'epoch_{best_epoch:03d}_scores.npz')
        if not os.path.exists(npz_path):
            # fallback: most recent npz
            candidates = sorted(glob.glob(os.path.join(dataset_dir, 'epoch_scores',
                                                        'epoch_*_scores.npz')))
            if not candidates:
                print(f"  [anomaly_threshold] no epoch_scores npz")
                return
            npz_path = candidates[-1]
        scores = np.load(npz_path)
        adaptive_score = scores['adaptive_score']
        recon = scores['teacher_recon_error']
        disc = scores['discrepancy_error']
        fm = scores['fm_error'] if 'fm_error' in scores.files else None

        # 4-5. SINGLE SOURCE: mae_anomaly.scoring.compute_adaptive_components.
        # This chart needs each component separately *with their effective
        # weights folded in* so that recon + scaled_disc(+scaled_fm) == score.
        # compute_adaptive_components returns scaled_disc/scaled_fm at unit
        # weight; we apply the same w/(w_disc+w_fm) folding the chart already
        # did so the visual interpretation is unchanged.
        from mae_anomaly.scoring import compute_adaptive_components
        comps = compute_adaptive_components(recon, disc, fm, self.config,
                                            force_recon_only=False)
        w_disc = comps['w_disc']
        w_fm = comps.get('w_fm', 0.0)
        ratio = comps.get('recon_disc_ratio', 4.0)
        # 2026-06-01: FM dropped from the anomaly score. The disc panel shows the
        # ACTUAL down-weighted contribution (student_error == scaled_disc / ratio)
        # so that recon + scaled_disc == anomaly_score holds exactly.
        use_fm = False
        scaled_disc = comps['student_error']
        scaled_fm = None

        # 6. Get test point labels — prefer npz-embedded (always present in new
        # format), fallback to test_loader (legacy path)
        if 'point_labels' in scores.files:
            point_labels = np.asarray(scores['point_labels'])
        else:
            try:
                test_dataset = self.test_loader.dataset
                point_labels = np.array(test_dataset.point_labels)
            except Exception:
                point_labels = None

        m = len(adaptive_score)
        if point_labels is not None:
            m = min(m, len(point_labels))
        adaptive_score = adaptive_score[:m]
        recon = recon[:m]
        scaled_disc = scaled_disc[:m]
        if scaled_fm is not None:
            scaled_fm = scaled_fm[:m]
        if point_labels is not None:
            point_labels = point_labels[:m]

        # [official] Use the actually-scored values: for official runs the score is
        # official_score (causal) and its disc contribution is (official_score - recon)
        # = 0.25·disc·s_t — matching the metric/threshold (and anomaly_threshold_test_event).
        # Differs from the adaptive scale-match (recon.mean/disc.mean / ratio).
        _is_official = bool(getattr(self.config, 'official', False)) and ('official_score' in scores.files)
        if _is_official:
            score_plot = np.asarray(scores['official_score'])[:m]
            disc_plot = score_plot - recon
            score_label = 'anomaly score (official causal)'
            disc_label = 'official disc contribution (= score − recon)'
        else:
            score_plot = adaptive_score
            disc_plot = scaled_disc
            score_label = 'anomaly_score (= recon + scaled_disc)'
            disc_label = f'scaled_disc  [recon:disc = {ratio:.0f}:1]'

        # 7. Build anomaly regions
        test_regions = []
        if point_labels is not None:
            in_anom = False
            start = 0
            for i, lbl in enumerate(point_labels):
                if lbl == 1 and not in_anom:
                    start = i
                    in_anom = True
                elif lbl == 0 and in_anom:
                    test_regions.append((start, i))
                    in_anom = False
            if in_anom:
                test_regions.append((start, m))

        # 8. Detection ratio per region (anomaly_score subplot only)
        if threshold is not None:
            det_ratios = []
            for s, e in test_regions:
                seg = score_plot[s:e]
                if len(seg) == 0:
                    det_ratios.append(0.0)
                else:
                    det_ratios.append(float((seg >= threshold).sum()) / float(len(seg)))
        else:
            det_ratios = [0.0] * len(test_regions)

        # Greedy 1-D non-overlap label layout (within a row)
        def _layout_label_positions(centers, label_width, x_min, x_max):
            n = len(centers)
            if n == 0:
                return []
            order = sorted(range(n), key=lambda i: centers[i])
            positions = [None] * n
            cur = x_min - label_width
            for idx in order:
                new_pos = max(centers[idx], cur + label_width)
                positions[idx] = new_pos
                cur = new_pos
            if positions[order[-1]] > x_max:
                cur = x_max + label_width
                for idx in reversed(order):
                    positions[idx] = min(positions[idx], cur - label_width)
                    cur = positions[idx]
            return positions

        # 9. Plot (3 or 4 subplots)
        n_subplots = 4 if scaled_fm is not None else 3
        fig, axes = plt.subplots(n_subplots, 1, figsize=(16, 3 * n_subplots), sharex=True)
        x = np.arange(m)

        panels = [
            (score_label, score_plot, 'black', threshold),
            ('recon  (weight = 1)', recon, 'tab:blue', None),
            (disc_label, disc_plot, 'tab:green', None),
        ]

        for idx_ax, (ax, (label, series, color, thr)) in enumerate(zip(axes, panels)):
            is_score_panel = (idx_ax == 0)
            # Shade regions
            for ri, (s, e) in enumerate(test_regions):
                if is_score_panel and threshold is not None:
                    alpha = 0.15 + 0.50 * det_ratios[ri]
                else:
                    alpha = 0.25
                ax.axvspan(s, e, alpha=alpha, color='red', zorder=1)
            ax.plot(x, series, color=color, linewidth=0.7, alpha=0.9, zorder=3)
            if thr is not None:
                ax.axhline(thr, color='black', linestyle='--', linewidth=1.0, alpha=0.7,
                           label=f'threshold={thr:.4f}')

            # Detection-ratio annotations
            if is_score_panel and threshold is not None and test_regions:
                n_reg = len(test_regions)
                if n_reg <= 30:
                    fontsize = 8; lw_frac = 0.025
                elif n_reg <= 60:
                    fontsize = 7; lw_frac = 0.018
                else:
                    fontsize = 6; lw_frac = 0.014
                label_width = m * lw_frac
                rows_needed = max(2, int(np.ceil(n_reg * label_width / (m * 0.9))))
                rows_needed = min(rows_needed, 6)

                y_min, y_max = ax.get_ylim()
                margin = (y_max - y_min) * (0.18 + 0.06 * rows_needed)
                new_top = y_max + margin
                ax.set_ylim(top=new_top)
                row_ys = [new_top - margin * (0.12 + 0.78 * (r + 0.5) / rows_needed)
                          for r in range(rows_needed)]

                centers = [(s + e) / 2.0 for s, e in test_regions]
                row_indices = {r: [] for r in range(rows_needed)}
                for i in range(n_reg):
                    row_indices[i % rows_needed].append(i)
                label_x_by_idx = {}
                for r, idxs in row_indices.items():
                    pos = _layout_label_positions(
                        [centers[i] for i in idxs], label_width, 0, m)
                    for k, i in enumerate(idxs):
                        label_x_by_idx[i] = pos[k]

                for ri, ((s, e), ratio) in enumerate(zip(test_regions, det_ratios)):
                    cx = centers[ri]
                    lx = label_x_by_idx[ri]
                    ly = row_ys[ri % rows_needed]
                    ax.plot([cx, lx], [y_max, ly - margin * 0.04],
                            color='gray', linewidth=0.35, alpha=0.5, zorder=8)
                    ax.text(lx, ly, f'{ratio:.2f}',
                            ha='center', va='center',
                            fontsize=fontsize, color='black', zorder=10,
                            bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                                      edgecolor='lightgray', alpha=0.95, linewidth=0.4))
                anom_patch = mpatches.Patch(color='red', alpha=0.4,
                                            label='anomaly (opacity ∝ detection ratio)')
                handles, _ = ax.get_legend_handles_labels()
                ax.legend(handles=[anom_patch] + handles, loc='upper left',
                          fontsize=9, framealpha=0.9)

            ax.set_ylabel(label, fontsize=9)
            ax.grid(alpha=0.3)

        axes[-1].set_xlabel('Test point index')
        axes[0].set_xlim(0, m)

        # FPR(non-anomaly): false positive rate on non-anomaly timesteps at the
        # best-epoch optimal threshold. Aligns with comparison baselines' viz.
        fpr_normal = None
        if (threshold is not None and np.isfinite(threshold)
                and point_labels is not None):
            normal_mask = (point_labels == 0)
            n_normal = int(normal_mask.sum())
            if n_normal > 0:
                n_fp = int(((score_plot >= threshold) & normal_mask).sum())
                fpr_normal = n_fp / n_normal

        prc_str = f', pak_auc_prc={pak_auc_prc:.4f}' if pak_auc_prc is not None else ''
        fpr_str = f', FPR(non-anomaly)={fpr_normal:.4f}' if fpr_normal is not None else ''
        fig.suptitle(
            f'Anomaly Threshold Timeline — best_epoch={best_epoch}, '
            f'pak_auc_f1={pak_auc_f1:.4f}{prc_str}{fpr_str}', fontsize=12)
        plt.tight_layout()
        out_path = os.path.join(self.output_dir, 'anomaly_threshold.png')
        plt.savefig(out_path, dpi=110, bbox_inches='tight')
        plt.close()
        print(f"  - anomaly_threshold.png")

    def plot_anomaly_threshold_test_event(self, experiment_dir: str = None):
        """Test Event Timeline view -> anomaly_threshold_test_event.png.

        For the anomaly score AND each component (recon, disc) draws an image1-style
        Test Event Timeline (score line + threshold + ground-truth shading) plus
        gt / prediction / overlap event tracks. The threshold is set from the TEST
        anomaly ratio: thr = quantile(score, 1 - ar), ar = mean(point_labels==1)
        (so exactly `ar` fraction of points are flagged as positive). Anomaly score =
        official causal score for official runs (else adaptive). npz-based, no GPU.
        """
        from matplotlib.lines import Line2D
        dataset_dir = os.path.dirname(os.path.dirname(self.output_dir))
        metrics_path = os.path.join(dataset_dir, 'epoch_metrics.json')
        if not os.path.exists(metrics_path):
            print(f"  [test_event] no epoch_metrics.json"); return
        epoch_data = json.load(open(metrics_path))
        best = self._select_best_epoch_for_viz(epoch_data['epochs'])
        best_epoch = best['epoch']
        npz_path = os.path.join(dataset_dir, 'epoch_scores', f'epoch_{best_epoch:03d}_scores.npz')
        if not os.path.exists(npz_path):
            cands = sorted(glob.glob(os.path.join(dataset_dir, 'epoch_scores', 'epoch_*_scores.npz')))
            if not cands:
                print(f"  [test_event] no epoch_scores npz"); return
            npz_path = cands[-1]
        scores = np.load(npz_path)
        if 'point_labels' not in scores.files:
            print(f"  [test_event] no point_labels in npz"); return
        point_labels = np.asarray(scores['point_labels']).astype(int)
        m = len(point_labels)
        adaptive = np.asarray(scores['adaptive_score'])[:m]
        recon = np.asarray(scores['teacher_recon_error'])[:m]
        disc_raw = np.asarray(scores['discrepancy_error'])[:m]
        fm = np.asarray(scores['fm_error'])[:m] if 'fm_error' in scores.files else None
        _is_official = bool(getattr(self.config, 'official', False)) and ('official_score' in scores.files)

        # Use the EXACT components of the score the metrics were actually computed on:
        #  - official run: anomaly score = official_score (= recon + w·disc·s_t, causal,
        #    w=0.25, s_t = (R_tr+Σrecon)/(D_tr+Σdisc) train-normal-seeded cumulative ratio).
        #    The disc contribution is recovered exactly as (official_score − recon)
        #    = w·disc·s_t — NOT the adaptive scale-match (recon.mean/disc.mean / ratio),
        #    which uses TEST means + a fixed /4 and differs from the official scaling.
        #  - non-official run: adaptive score = recon + scaled_disc (compute_adaptive_components).
        if _is_official:
            score = np.asarray(scores['official_score'])[:m]
            disc_contrib = score - recon
            series = [
                ('anomaly score (official causal = recon + 0.25·disc·s_t)', score, 'black'),
                ('recon  (weight = 1)', recon, 'tab:blue'),
                ('official disc contribution (= score − recon = 0.25·disc·s_t)', disc_contrib, 'tab:green'),
            ]
        else:
            from mae_anomaly.scoring import compute_adaptive_components
            comps = compute_adaptive_components(recon, disc_raw, fm, self.config, force_recon_only=False)
            scaled_disc = np.asarray(comps['student_error'])[:m]
            ratio = comps.get('recon_disc_ratio', 4.0)
            series = [
                ('anomaly score (= recon + scaled_disc)', adaptive, 'black'),
                ('recon  (weight = 1)', recon, 'tab:blue'),
                (f'scaled_disc  [recon:disc = {ratio:.0f}:1]', scaled_disc, 'tab:green'),
            ]

        ar = float((point_labels == 1).mean())
        gt_mask = (point_labels == 1)
        x = np.arange(m)

        def _mask_regions(mask):
            regs = []; ins = False; st = 0
            for i, v in enumerate(mask):
                if v and not ins:
                    st = i; ins = True
                elif not v and ins:
                    regs.append((st, i)); ins = False
            if ins:
                regs.append((st, len(mask)))
            return regs
        gt_regions = _mask_regions(gt_mask)

        n = len(series)
        fig, all_axes = plt.subplots(2 * n, 1, figsize=(16, 3.4 * n), sharex=True,
                                     gridspec_kw={'height_ratios': [2.0, 0.7] * n})
        for si, (name, s, scolor) in enumerate(series):
            ax_t = all_axes[2 * si]; ax_k = all_axes[2 * si + 1]
            # threshold from the TEST anomaly ratio
            thr = float(np.quantile(s, 1.0 - ar)) if 0.0 < ar < 1.0 else float(np.max(s) + 1.0)
            pred_mask = (s >= thr)
            pred_regions = _mask_regions(pred_mask)
            overlap_regions = _mask_regions(pred_mask & gt_mask)
            for a, b in gt_regions:
                ax_t.axvspan(a, b, color='red', alpha=0.12, zorder=1)
            ax_t.plot(x, s, color=scolor, linewidth=0.7, zorder=3)
            ax_t.axhline(thr, color='black', linestyle='--', linewidth=1.0, alpha=0.8, zorder=4)
            ax_t.set_ylabel('Score', fontsize=9)
            ax_t.set_title(f'{name}  —  AR-threshold (ar={ar:.4f}) = {thr:.4g}', fontsize=10)
            ax_t.legend(handles=[
                mpatches.Patch(color='red', alpha=0.25, label='ground truth event'),
                Line2D([], [], color=scolor, label='score'),
                Line2D([], [], color='black', linestyle='--', label='threshold (AR)'),
            ], loc='upper right', fontsize=8, framealpha=0.9)
            ax_t.grid(alpha=0.3)
            for yi, (regs, color) in enumerate([(gt_regions, 'red'), (pred_regions, 'tab:blue'),
                                                (overlap_regions, 'tab:purple')]):
                yy = 2 - yi
                for a, b in regs:
                    ax_k.broken_barh([(a, max(1, b - a))], (yy - 0.35, 0.7), facecolors=color)
            ax_k.set_yticks([2, 1, 0]); ax_k.set_yticklabels(['gt', 'pred', 'overlap'], fontsize=8)
            ax_k.set_ylim(-0.6, 2.6)
            ax_k.legend(handles=[
                mpatches.Patch(color='red', label='ground truth'),
                mpatches.Patch(color='tab:blue', label='prediction'),
                mpatches.Patch(color='tab:purple', label='overlap'),
            ], loc='upper right', ncol=3, fontsize=8, framealpha=0.9)
            ax_k.grid(alpha=0.2, axis='x')

        all_axes[-1].set_xlabel('Test point index')
        all_axes[0].set_xlim(0, m)
        fig.suptitle(f'Test Event Timeline (AR threshold) — best_epoch={best_epoch}, '
                     f'anomaly_ratio={ar:.4f}', fontsize=12)
        plt.tight_layout()
        out_path = os.path.join(self.output_dir, 'anomaly_threshold_test_event.png')
        plt.savefig(out_path, dpi=110, bbox_inches='tight')
        plt.close()
        print(f"  - anomaly_threshold_test_event.png")

    def generate_all(self, experiment_dir: str = None, history: Dict = None):
        """Generate all best model visualizations

        Args:
            experiment_dir: Path to experiment directory for loading detailed results
            history: Training history dictionary for learning curve visualization
        """
        print("\n  Generating Best Model Visualizations...")

        def _safe_plot(name, fn):
            try:
                fn()
            except Exception as e:
                print(f"  [VIZ WARNING] {name} failed: {e}")

        _safe_plot('roc_curve', self.plot_roc_curve)
        _safe_plot('prc_curve', self.plot_prc_curve)
        _safe_plot('confusion_matrix', self.plot_confusion_matrix)
        _safe_plot('anomaly_threshold', lambda: self.plot_anomaly_threshold(experiment_dir))
        _safe_plot('anomaly_threshold_test_event', lambda: self.plot_anomaly_threshold_test_event(experiment_dir))
        _safe_plot('score_contribution', lambda: self.plot_score_contribution_analysis(experiment_dir))
        _safe_plot('reconstruction_examples', self.plot_reconstruction_examples)
        _safe_plot('detection_examples', self.plot_detection_examples)
        _safe_plot('summary_statistics', self.plot_summary_statistics)
        _safe_plot('learning_curve', lambda: self.plot_learning_curve(history))
        _safe_plot('pure_vs_disturbing', self.plot_pure_vs_disturbing_normal)
        _safe_plot('discrepancy_trend', self.plot_discrepancy_trend)

        # Qualitative case studies
        _safe_plot('case_study_gallery', lambda: self.plot_case_study_gallery(experiment_dir))
        _safe_plot('hardest_samples', self.plot_hardest_samples)

        # Anomaly type analysis (requires detailed results from experiment)
        anomaly_type_metrics = self._compute_anomaly_type_metrics()
        _safe_plot('performance_by_type', lambda: self.plot_performance_by_anomaly_type(anomaly_type_metrics))
        _safe_plot('score_dist_by_type', lambda: self.plot_score_distribution_by_type(experiment_dir))

        # ROC and PRC curve comparisons (different score types)
        _safe_plot('roc_comparison', self.plot_roc_curve_comparison)
        _safe_plot('prc_comparison', self.plot_prc_curve_comparison)
        _safe_plot('roc_pa80_comparison', self.plot_roc_curve_pa80_comparison)

        # PA%K AUC summary (requires experiment_metadata.json)
        _safe_plot('pak_auc_summary', lambda: self.plot_pa_k_auc_summary(experiment_dir))

        # Feature-level analysis (requires disc_per_feature from evaluator)
        _safe_plot('feature_dominance', self.plot_feature_dominance)
        _safe_plot('feature_extremes', self.plot_feature_extremes)
        _safe_plot('feature_profile', self.plot_feature_profile)

        # FM vs OD loss contribution trend over training
        _safe_plot('fm_od_contribution_trend', lambda: self.plot_fm_od_contribution_trend(experiment_dir, history))

        # GRL contribution trend over training
        _safe_plot('grl_contribution_trend', lambda: self.plot_grl_contribution_trend(experiment_dir, history))

        # SCAD contribution trend over training (mirror of GRL plot)
        _safe_plot('scad_contribution_trend', lambda: self.plot_scad_contribution_trend(experiment_dir, history))

    def plot_fm_od_contribution_trend(self, experiment_dir: str = None, history: Dict = None):
        """Plot epoch-wise Feature Matching (FM) loss vs Object Detection (OD) loss trends.

        Creates a 1x3 figure:
        (A) FM vs OD absolute values over epochs
        (B) FM / (FM + OD) ratio over epochs
        (C) fm_adaptive_lambda over epochs (if available)

        OD loss = train_normal_loss + train_anomaly_loss.
        FM loss = train_fm_loss.

        Args:
            experiment_dir: Path to experiment directory for loading training history
            history: Optional pre-loaded history dict
        """
        # Load history if not provided
        if history is None and experiment_dir:
            history_path = os.path.join(experiment_dir, 'training_histories.json')
            if os.path.exists(history_path):
                with open(history_path) as f:
                    history_data = json.load(f)
                    history = history_data.get('0', {})

        if history is None:
            print("  - Skipping FM_OD_contribution_trend.png (no history)")
            return

        fm_loss = history.get('train_fm_loss')
        normal_loss = history.get('train_normal_loss')
        anomaly_loss = history.get('train_anomaly_loss')

        if not fm_loss or not normal_loss or not anomaly_loss:
            print("  - Skipping FM_OD_contribution_trend.png (missing loss keys)")
            return

        epochs = np.array(history.get('epoch', list(range(1, len(fm_loss) + 1))))
        fm = np.array(fm_loss)
        od = np.array(normal_loss) + np.array(anomaly_loss)

        # Check if FM loss is all zeros (feature matching disabled)
        if fm.max() < 1e-10:
            print("  - Skipping FM_OD_contribution_trend.png (FM loss all zeros — feature matching disabled)")
            return

        has_adaptive_lambda = bool(history.get('train_fm_adaptive_lambda'))
        n_cols = 3 if has_adaptive_lambda else 2
        fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))
        if n_cols == 1:
            axes = [axes]

        # (A) FM vs OD absolute values
        ax = axes[0]
        ax.plot(epochs, fm, color='#E74C3C', linewidth=2, label='FM Loss', marker='o', markersize=3)
        ax.plot(epochs, od, color='#3498DB', linewidth=2, label='OD Loss (Normal+Anomaly)', marker='s', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('(A) FM vs OD Loss', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # (B) FM / (FM + OD) ratio
        ax = axes[1]
        total = fm + od + 1e-10
        ratio = fm / total * 100
        ax.plot(epochs, ratio, color='#9B59B6', linewidth=2, marker='D', markersize=3)
        ax.fill_between(epochs, 0, ratio, color='#9B59B6', alpha=0.15)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('FM / (FM+OD) %')
        ax.set_title('(B) FM Contribution Ratio', fontweight='bold')
        ax.set_ylim(0, max(100, ratio.max() * 1.1))
        ax.grid(True, alpha=0.3)

        # (C) fm_adaptive_lambda (if available)
        if has_adaptive_lambda:
            ax = axes[2]
            adaptive_lambda = np.array(history['train_fm_adaptive_lambda'])
            ax.plot(epochs[:len(adaptive_lambda)], adaptive_lambda, color='#F39C12',
                    linewidth=2, marker='^', markersize=3)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Adaptive Lambda')
            ax.set_title('(C) FM Adaptive Lambda', fontweight='bold')
            ax.grid(True, alpha=0.3)

        fig.suptitle('Feature Matching vs Object Detection Loss Trend', fontsize=14, fontweight='bold')
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, 'FM_OD_contribution_trend.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("  - FM_OD_contribution_trend.png")

    def plot_grl_contribution_trend(self, experiment_dir: str = None, history: Dict = None):
        """Plot epoch-wise GRL classifier contribution to total loss.

        Creates a 1x3 figure:
        (A) Main loss vs GRL contribution (grl_cls_loss * grl_effective_weight)
        (B) GRL / (Main + GRL) contribution ratio
        (C) GRL classifier metrics (balanced_acc, anomaly_acc, normal_acc)
        """
        if history is None and experiment_dir:
            history_path = os.path.join(experiment_dir, 'training_histories.json')
            if os.path.exists(history_path):
                with open(history_path) as f:
                    history_data = json.load(f)
                    history = history_data.get('0', {})

        if history is None:
            print("  - Skipping GRL_contribution_trend.png (no history)")
            return

        grl_cls = history.get('train_grl_cls_loss')
        grl_ew = history.get('train_grl_effective_weight')
        main_loss = history.get('train_loss')

        if not grl_cls or not grl_ew or not main_loss:
            print("  - Skipping GRL_contribution_trend.png (missing GRL keys)")
            return

        grl_cls = np.array(grl_cls)
        grl_ew = np.array(grl_ew)
        main = np.array(main_loss)

        if grl_cls.max() < 1e-10:
            print("  - Skipping GRL_contribution_trend.png (GRL inactive)")
            return

        epochs = np.array(history.get('epoch', list(range(1, len(grl_cls) + 1))))
        grl_contrib = grl_cls * grl_ew

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # (A) Main loss vs GRL contribution
        ax = axes[0]
        ax.plot(epochs, main, color='#3498DB', linewidth=2, label='Main Loss', marker='o', markersize=2)
        ax.plot(epochs, grl_contrib, color='#E74C3C', linewidth=2, label='GRL Contribution', marker='s', markersize=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('(A) Main vs GRL Loss', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # (B) GRL contribution ratio
        ax = axes[1]
        total = main + grl_contrib + 1e-10
        ratio = grl_contrib / total * 100
        ax.plot(epochs, ratio, color='#9B59B6', linewidth=2, marker='D', markersize=2)
        ax.fill_between(epochs, 0, ratio, color='#9B59B6', alpha=0.15)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('GRL / (Main+GRL) %')
        ax.set_title('(B) GRL Contribution Ratio', fontweight='bold')
        ax.set_ylim(0, min(100, max(ratio.max() * 1.1, 10)))
        ax.grid(True, alpha=0.3)

        # (C) GRL classifier metrics + degeneracy gap.
        # balanced_acc=0.5 is deceptive — it coexists with full degeneracy
        # (normal=0, anomaly=1). |normal-anomaly| gap disambiguates: gap≈0 with
        # balanced high = genuine; gap≈1 = degenerate (classifier ignores one class).
        ax = axes[2]
        bacc = history.get('train_grl_balanced_acc')
        aacc = history.get('train_grl_anomaly_acc')
        nacc = history.get('train_grl_normal_acc')
        if bacc and aacc and nacc:
            _aacc = np.array(aacc); _nacc = np.array(nacc)
            _gap = history.get('train_grl_acc_gap')
            gap = np.array(_gap) if _gap else np.abs(_nacc - _aacc)
            ax.plot(epochs, np.array(bacc), color='#2ECC71', linewidth=2, label='Balanced Acc', marker='o', markersize=2)
            ax.plot(epochs, _aacc, color='#E74C3C', linewidth=1.5, label='Anomaly Acc', linestyle='--', alpha=0.7)
            ax.plot(epochs, _nacc, color='#3498DB', linewidth=1.5, label='Normal Acc', linestyle='--', alpha=0.7)
            ax.plot(epochs, gap, color='#E65100', linewidth=2.5, label='|N−A| Gap (degeneracy)', marker='o', markersize=2)
            ax.fill_between(epochs, -0.05, 1.05, where=(gap >= 0.8),
                            color='#E65100', alpha=0.10, label='Degenerate (gap≥0.8)')
            ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
            ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy / Gap')
        ax.set_title('(C) GRL Metrics + Degeneracy Gap', fontweight='bold')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        fig.suptitle('GRL Contribution Trend', fontsize=14, fontweight='bold')
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, 'GRL_contribution_trend.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("  - GRL_contribution_trend.png")

    def plot_scad_contribution_trend(self, experiment_dir: str = None, history: Dict = None):
        """Plot epoch-wise SCAD contribution to total loss (mirrors plot_grl_contribution_trend).

        Creates a 1x3 figure:
        (A) Main loss vs SCAD contribution (scad_loss × scad_effective_weight)
        (B) SCAD / (Main + SCAD) contribution ratio
        (C) Diagnostic stack: z_separation, adaptive λ, cluster variances
        """
        if history is None and experiment_dir:
            history_path = os.path.join(experiment_dir, 'training_histories.json')
            if os.path.exists(history_path):
                with open(history_path) as f:
                    history_data = json.load(f)
                    history = history_data.get('0', {})

        if history is None:
            print("  - Skipping SCAD_contribution_trend.png (no history)")
            return

        scad_loss = history.get('train_scad_loss')
        scad_eff = history.get('train_scad_effective_weight')
        main_loss = history.get('train_loss')

        if not scad_loss or not scad_eff or not main_loss:
            print("  - Skipping SCAD_contribution_trend.png (missing SCAD keys)")
            return

        scad_loss = np.array(scad_loss)
        scad_eff = np.array(scad_eff)
        main = np.array(main_loss)

        if scad_loss.max() < 1e-10:
            print("  - Skipping SCAD_contribution_trend.png (SCAD inactive)")
            return

        epochs = np.array(history.get('epoch', list(range(1, len(scad_loss) + 1))))
        scad_contrib = scad_loss * scad_eff

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # (A) Main vs SCAD Contribution
        ax = axes[0]
        ax.plot(epochs, main, color='#1565C0', lw=2, label='Main Loss', marker='o', ms=2)
        ax.plot(epochs, scad_contrib, color='#D32F2F', lw=2, label='SCAD Contribution', marker='s', ms=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('(A) Main vs SCAD Loss', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # (B) Contribution Ratio
        ax = axes[1]
        total = main + scad_contrib + 1e-10
        ratio = scad_contrib / total * 100
        ax.plot(epochs, ratio, color='#7B1FA2', lw=2, marker='D', ms=2)
        ax.fill_between(epochs, 0, ratio, color='#7B1FA2', alpha=0.15)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('SCAD / (Main+SCAD) %')
        ax.set_title('(B) SCAD Contribution Ratio', fontweight='bold')
        ax.set_ylim(0, min(100, max(ratio.max() * 1.1, 10)))
        ax.grid(True, alpha=0.3)

        # (C) Diagnostic stack (separation + adaptive λ + variances)
        ax = axes[2]
        sep = history.get('train_scad_z_separation', [])
        if sep:
            ax.plot(epochs, np.array(sep), color='#2E7D32', lw=2,
                    label='Z Separation', marker='o', ms=2)
            ax.set_ylabel('Z Separation', color='#2E7D32')
            ax.tick_params(axis='y', labelcolor='#2E7D32')

        avar = history.get('train_scad_z_anom_var', [])
        nvar = history.get('train_scad_z_norm_var', [])
        alam = history.get('train_scad_adaptive_lambda', [])
        if alam:
            ax2 = ax.twinx()
            ax2.plot(epochs, np.array(alam), color='#E65100', lw=1.5, ls='--',
                     label='Adaptive λ', marker='*', ms=3, alpha=0.7)
            if avar:
                ax2.plot(epochs, np.array(avar), color='#C62828', lw=1, ls=':',
                         label='Anom Var', alpha=0.5)
            if nvar:
                ax2.plot(epochs, np.array(nvar), color='#1565C0', lw=1, ls=':',
                         label='Norm Var', alpha=0.5)
            ax2.set_ylabel('λ / Variance', color='#E65100')
            ax2.tick_params(axis='y', labelcolor='#E65100')
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='upper left')

        ax.set_xlabel('Epoch')
        ax.set_title('(C) SCAD Diagnostic Stack', fontweight='bold')
        ax.grid(True, alpha=0.3)

        fig.suptitle('SCAD Contribution Trend', fontsize=14, fontweight='bold')
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, 'SCAD_contribution_trend.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("  - SCAD_contribution_trend.png")

    def plot_roc_curve_comparison(self):
        """Plot ROC curves comparing different scoring methods.

        Compares:
        - Anomaly Score (combined): recon + lambda * disc (based on scoring mode)
        - Discrepancy Only: teacher-student difference
        - Teacher Recon Only: teacher reconstruction error
        - Student Recon Only: student reconstruction error
        """
        labels = self.pred_data['labels']
        combined_scores = self.pred_data['scores']
        # Use point-level component scores if available, else patch-level
        recon_errors = self.pred_data.get('point_recon', self.pred_data['recon_errors'])
        student_errors = self.pred_data.get('point_student', self.pred_data['student_errors'])
        discrepancies = self.pred_data.get('point_disc', self.pred_data['discrepancies'])

        fig, ax = plt.subplots(figsize=(10, 10))

        # Define scoring methods and their colors
        scoring_methods = [
            ('Anomaly Score (combined)', combined_scores, VIS_COLORS['total']),
            ('Discrepancy Only', discrepancies, VIS_COLORS['discrepancy']),
            ('Teacher Recon Only', recon_errors, VIS_COLORS['teacher']),
            ('Student Recon Only', student_errors, VIS_COLORS['student']),
        ]

        # Track offset for annotations to avoid overlap
        annotation_offsets = [
            (0.02, 0.02),   # Combined
            (-0.18, -0.08),  # Discrepancy
            (0.02, -0.08),  # Teacher
            (-0.18, 0.02),  # Student
        ]

        # Short names for annotations
        short_names = ['Anomaly', 'Discr.', 'Teacher', 'Student']

        # Plot ROC curve for each method
        for idx, (name, scores, color) in enumerate(scoring_methods):
            if len(np.unique(labels)) > 1:
                fpr, tpr, thresholds = roc_curve(labels, scores)
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC={roc_auc:.4f})')

                # Find F1-optimal point
                optimal_idx = find_f1_optimal_idx(fpr, tpr, labels)
                optimal_threshold = thresholds[optimal_idx]
                predictions = (scores > optimal_threshold).astype(int)
                f1 = f1_score(labels, predictions)

                # Mark optimal point
                opt_fpr = fpr[optimal_idx]
                opt_tpr = tpr[optimal_idx]
                ax.scatter(opt_fpr, opt_tpr, s=80, color=color, zorder=5, marker='o')

                # Add annotation near optimal point with method name, AUC, and F1
                offset_x, offset_y = annotation_offsets[idx]
                short_name = short_names[idx]
                ax.annotate(f'{short_name}\nAUC={roc_auc:.3f}\nF1={f1:.3f}',
                           xy=(opt_fpr, opt_tpr),
                           xytext=(opt_fpr + offset_x, opt_tpr + offset_y),
                           fontsize=8, color=color, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor=color),
                           arrowprops=dict(arrowstyle='->', color=color, lw=0.5))

        # Reference line
        ax.plot([0, 1], [0, 1], color=VIS_COLORS['reference'], lw=2, linestyle='--', label='Random')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')

        scoring_mode = getattr(self.config, 'anomaly_score_mode', 'default')
        ax.set_title(f'ROC Curve Comparison\n(Scoring Mode: {scoring_mode})', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'best_model_roc_curve_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - best_model_roc_curve_comparison.png")

    def plot_prc_curve_comparison(self):
        """Plot Precision-Recall curves comparing different scoring methods."""
        labels = self.pred_data['labels']
        combined_scores = self.pred_data['scores']
        recon_errors = self.pred_data.get('point_recon', self.pred_data['recon_errors'])
        student_errors = self.pred_data.get('point_student', self.pred_data['student_errors'])
        discrepancies = self.pred_data.get('point_disc', self.pred_data['discrepancies'])

        fig, ax = plt.subplots(figsize=(10, 10))

        scoring_methods = [
            ('Anomaly Score (combined)', combined_scores, VIS_COLORS['total']),
            ('Discrepancy Only', discrepancies, VIS_COLORS['discrepancy']),
            ('Teacher Recon Only', recon_errors, VIS_COLORS['teacher']),
            ('Student Recon Only', student_errors, VIS_COLORS['student']),
        ]

        annotation_offsets = [
            (0.02, 0.02), (-0.18, -0.08), (0.02, -0.08), (-0.18, 0.02),
        ]
        short_names = ['Anomaly', 'Discr.', 'Teacher', 'Student']

        for idx, (name, scores, color) in enumerate(scoring_methods):
            if len(np.unique(labels)) > 1:
                prec, rec, thresholds = precision_recall_curve(labels, scores)
                # sklearn returns recall in descending order, reverse for plotting only
                rec = rec[::-1]
                prec = prec[::-1]
                # Use step-AP (sklearn average_precision_score), not trapz/auc, for PR-AUC
                prc_auc_val = average_precision_score(labels, scores)
                ax.plot(rec, prec, color=color, lw=2, label=f'{name} (AUC={prc_auc_val:.4f})')

                # Find F1-optimal point
                f1_scores = 2 * prec * rec / (prec + rec + 1e-10)
                opt_idx = np.argmax(f1_scores)
                opt_rec = rec[opt_idx]
                opt_prec = prec[opt_idx]
                f1 = f1_scores[opt_idx]
                ax.scatter(opt_rec, opt_prec, s=80, color=color, zorder=5, marker='o')

                offset_x, offset_y = annotation_offsets[idx]
                short_name = short_names[idx]
                ax.annotate(f'{short_name}\nAUC={prc_auc_val:.3f}\nF1={f1:.3f}',
                           xy=(opt_rec, opt_prec),
                           xytext=(opt_rec + offset_x, opt_prec + offset_y),
                           fontsize=8, color=color, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor=color),
                           arrowprops=dict(arrowstyle='->', color=color, lw=0.5))

        # Baseline
        baseline = labels.sum() / len(labels)
        ax.axhline(y=baseline, color=VIS_COLORS['reference'], lw=2, linestyle='--', label=f'Baseline ({baseline:.4f})')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')

        scoring_mode = getattr(self.config, 'anomaly_score_mode', 'default')
        ax.set_title(f'Precision-Recall Curve Comparison\n(Scoring Mode: {scoring_mode})', fontsize=12, fontweight='bold')
        ax.legend(loc='lower left', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'best_model_prc_curve_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - best_model_prc_curve_comparison.png")

    def plot_roc_curve_pa80_comparison(self):
        """Plot PA%80 PRC-AUC bar comparison for different scoring methods.

        Uses mean-aggregated point-level scores with PA%80 segment adjustment.
        Compares Adaptive, Discrepancy, Teacher, Student scoring methods.
        """
        raw_dataset = self.test_loader.dataset if hasattr(self.test_loader, 'dataset') else None
        base_dataset = _unwrap_subset(raw_dataset) if raw_dataset is not None else None
        can_use = (
            base_dataset is not None and
            hasattr(base_dataset, 'anomaly_regions') and
            hasattr(base_dataset, 'point_labels') and
            hasattr(base_dataset, 'window_start_indices') and
            len(base_dataset.anomaly_regions) > 0
        )

        if not can_use:
            print("  - best_model_roc_curve_PA80_comparison.png (SKIPPED: no segment info)")
            return

        patch_recon = self.pred_data['recon_errors']
        patch_student = self.pred_data['student_errors']
        patch_disc = self.pred_data['discrepancies']
        patch_combined = self.pred_data['patch_scores']

        total_length = len(base_dataset.point_labels)
        point_labels = np.array(base_dataset.point_labels)
        anomaly_regions = base_dataset.anomaly_regions
        _, window_start_indices = _get_subset_window_indices(raw_dataset)

        n_windows = self.pred_data.get('n_windows', len(window_start_indices))
        num_patches = self.pred_data.get('num_patches', getattr(self.config, 'num_patches', 10))

        scoring_methods = [
            ('Anomaly Score', patch_combined.reshape(n_windows, num_patches), VIS_COLORS['total']),
            ('Discrepancy', patch_disc.reshape(n_windows, num_patches), VIS_COLORS['discrepancy']),
            ('Teacher Recon', patch_recon.reshape(n_windows, num_patches), VIS_COLORS['teacher']),
            ('Student Recon', patch_student.reshape(n_windows, num_patches), VIS_COLORS['student']),
        ]

        flat_t, flat_wp, coverage, covered = _build_aggregation_map(
            window_start_indices, self.config.patch_size, num_patches, total_length,
        )
        eval_mask = np.ones(total_length, dtype=bool)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        method_names = []
        roc_aucs = []
        prc_aucs = []
        colors = []

        for name, patch_scores, color in scoring_methods:
            point_scores = _aggregate_with_map(
                patch_scores.ravel(), flat_t, flat_wp, coverage, covered, total_length, method='mean'
            )
            point_scores = np.nan_to_num(point_scores, nan=0.0)

            pa_rp = compute_pa_k_roc_prc_from_mean_scores(
                point_scores, point_labels, anomaly_regions, 80, eval_mask
            )
            method_names.append(name)
            roc_aucs.append(pa_rp['roc_auc'])
            prc_aucs.append(pa_rp['prc_auc'])
            colors.append(color)

        x = np.arange(len(method_names))
        bars1 = ax1.bar(x, roc_aucs, color=colors, alpha=0.8)
        for bar, val in zip(bars1, roc_aucs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax1.set_ylabel('PA%80 ROC-AUC')
        ax1.set_title('PA%80 ROC-AUC by Scoring Method', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(method_names, rotation=15)
        _, roc_hi = _safe_lim(0, max(roc_aucs))
        ax1.set_ylim(0, min(1.05, roc_hi * 1.15 + 0.02))
        ax1.grid(True, alpha=0.3, axis='y')

        bars2 = ax2.bar(x, prc_aucs, color=colors, alpha=0.8)
        for bar, val in zip(bars2, prc_aucs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax2.set_ylabel('PA%80 PRC-AUC')
        ax2.set_title('PA%80 PRC-AUC by Scoring Method', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(method_names, rotation=15)
        _, prc_hi = _safe_lim(0, max(prc_aucs))
        ax2.set_ylim(0, min(1.05, prc_hi * 1.15 + 0.02))
        ax2.grid(True, alpha=0.3, axis='y')

        scoring_mode = getattr(self.config, 'anomaly_score_mode', 'default')
        fig.suptitle(f'PA%80 AUC Comparison (Mean-Based, Scoring: {scoring_mode})',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'best_model_roc_curve_PA80_comparison.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()
        print("  - best_model_roc_curve_PA80_comparison.png")


# =============================================================================
# TrainingProgressVisualizer - Training Progress Analysis
# =============================================================================

