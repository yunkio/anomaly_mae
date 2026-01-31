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
import json
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve, f1_score
from scipy.stats import gaussian_kde
from tqdm import tqdm

from torch.utils.data import Subset

from mae_anomaly import Config, ANOMALY_TYPE_NAMES, ANOMALY_CATEGORY
from mae_anomaly.evaluator import (
    compute_pa_k_adjusted_predictions,
    aggregate_patch_scores_to_point_level,
    compute_segment_pa_k_detection_rate,
    precompute_point_score_indices,
    vectorized_voting_for_all_thresholds,
    _compute_single_pa_k_roc,
    _compute_voted_point_predictions,
)
from .base import (
    get_anomaly_colors, SAMPLE_TYPE_NAMES, SAMPLE_TYPE_COLORS,
    VIS_COLORS, VIS_MARKERS, VIS_LINESTYLES,
    collect_predictions, collect_detailed_data,
    compute_score_contributions,
)

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

    def __init__(self, model, config: Config, test_loader, output_dir: str,
                 pred_data: Dict = None, detailed_data: Dict = None):
        """Initialize BestModelVisualizer.

        Args:
            model: Trained model
            config: Model configuration
            test_loader: Test data loader
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
        self._optimal_idx = np.argmax(tpr - fpr)

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
        """Compute optimal threshold from current scores using ROC analysis."""
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(self.pred_data['labels'], self.pred_data['scores'])
        optimal_idx = np.argmax(tpr - fpr)
        self.threshold = thresholds[optimal_idx]

    def recompute_scores(self, scoring_mode: str):
        """Recompute anomaly scores with different scoring mode (CPU only, fast).

        Recomputes patch-level scores, then re-aggregates to point-level.
        The raw recon_errors and discrepancies are preserved.

        Args:
            scoring_mode: One of 'default', 'adaptive', 'normalized'
        """
        recon = self.pred_data['recon_errors']
        disc = self.pred_data['discrepancies']
        lambda_disc = getattr(self.config, 'lambda_disc', 0.5)

        if scoring_mode == 'default':
            patch_scores = recon + lambda_disc * disc
        elif scoring_mode == 'adaptive':
            adaptive_lambda = recon.mean() / (disc.mean() + 1e-4)
            patch_scores = recon + adaptive_lambda * disc
        elif scoring_mode == 'normalized':
            recon_z = (recon - recon.mean()) / (recon.std() + 1e-4)
            disc_z = (disc - disc.mean()) / (disc.std() + 1e-4)
            patch_scores = recon_z + disc_z
        else:
            raise ValueError(f"Unknown scoring_mode: {scoring_mode}")

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
                  label=f'Optimal (threshold={optimal_threshold:.4f})')
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

    def plot_performance_by_anomaly_type(self, experiment_dir: str = None):
        """Plot detection performance by anomaly type

        Args:
            experiment_dir: Path to experiment directory containing anomaly_type_metrics.json
        """
        metrics_json = None
        if experiment_dir:
            json_path = os.path.join(experiment_dir, 'anomaly_type_metrics.json')
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    metrics_json = json.load(f)

        if metrics_json is None:
            print("  ! Skipping performance_by_anomaly_type (no metrics JSON found)")
            return

        # Extract detection rates for anomaly types
        anomaly_types_original = []  # Original names for color lookup
        anomaly_types_display = []   # Display names with line breaks
        detection_rates = []  # Point-wise detection rate
        pa_10_detection_rates = []  # PA%10
        pa_20_detection_rates = []  # PA%20
        pa_50_detection_rates = []  # PA%50
        pa_80_detection_rates = []  # PA%80
        mean_scores = []
        counts = []

        # Extract normal/disturbing scores for reference
        normal_mean_score = 0.0
        disturbing_mean_score = 0.0

        for atype, metrics in metrics_json.items():
            if atype == 'normal':
                normal_mean_score = metrics.get('mean_score', 0)
                continue  # Skip normal for detection rate plots
            if atype == 'disturbing_normal':
                disturbing_mean_score = metrics.get('mean_score', 0)
                continue  # Skip disturbing for detection rate plots
            anomaly_types_original.append(atype)
            anomaly_types_display.append(atype.replace('_', '\n'))
            detection_rates.append(metrics.get('detection_rate', 0) * 100)
            # PA%K detection rates for all K values
            default_rate = metrics.get('detection_rate', 0)
            pa_10_detection_rates.append(metrics.get('pa_10_detection_rate', default_rate) * 100)
            pa_20_detection_rates.append(metrics.get('pa_20_detection_rate', default_rate) * 100)
            pa_50_detection_rates.append(metrics.get('pa_50_detection_rate', default_rate) * 100)
            pa_80_detection_rates.append(metrics.get('pa_80_detection_rate', default_rate) * 100)
            mean_scores.append(metrics.get('mean_score', 0))
            counts.append(metrics.get('count', 0))

        if len(anomaly_types_original) == 0:
            print("  ! Skipping performance_by_anomaly_type (no anomaly types found)")
            return

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

        # For normalized mode, shift all scores so minimum is 0
        scoring_mode = getattr(self.config, 'anomaly_score_mode', 'default')
        all_scores_for_shift = mean_scores + [normal_mean_score, disturbing_mean_score]
        score_min = min(all_scores_for_shift) if scoring_mode == 'normalized' else 0

        # Shift scores for normalized mode
        display_scores = [s - score_min for s in mean_scores]
        display_normal = normal_mean_score - score_min
        display_disturbing = disturbing_mean_score - score_min

        # Bar chart for anomaly types only
        bars = ax.bar(anomaly_types_display, display_scores, color=colors, alpha=0.8, edgecolor=VIS_COLORS['baseline'])
        ax.set_ylabel('Mean Anomaly Score')
        title_suffix = ' (shifted to 0)' if scoring_mode == 'normalized' else ''
        ax.set_title(f'Mean Score by Sample Type{title_suffix}', fontweight='bold')
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

        # Compute scores based on scoring mode
        recon_raw = detailed_csv['reconstruction_loss'].values
        disc_raw = detailed_csv['discrepancy_loss'].values

        if scoring_mode == 'default':
            # Total score = recon + lambda_disc * disc
            recon_scores = recon_raw
            disc_scores = lambda_disc * disc_raw
        elif scoring_mode == 'adaptive':
            # Total score = recon + adaptive_lambda * disc
            # where adaptive_lambda = mean(recon) / mean(disc)
            recon_mean = recon_raw.mean() + 1e-8
            disc_mean = disc_raw.mean() + 1e-8
            adaptive_weight = recon_mean / disc_mean
            recon_scores = recon_raw
            disc_scores = adaptive_weight * disc_raw
        else:  # normalized
            # Total score = recon_z + disc_z (same as evaluator.py)
            # Note: z-scores can be negative, which is correct
            recon_mean, recon_std = recon_raw.mean(), recon_raw.std() + 1e-8
            disc_mean, disc_std = disc_raw.mean(), disc_raw.std() + 1e-8
            recon_scores = (recon_raw - recon_mean) / recon_std
            disc_scores = (disc_raw - disc_mean) / disc_std

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

            # Prepare data for violin plot
            plot_data = pd.DataFrame({
                'Score': np.concatenate([type_data['recon_score'].values, type_data['disc_score'].values]),
                'Component': ['Recon'] * len(type_data) + ['Disc'] * len(type_data)
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

        scoring_label = {'default': 'Default', 'adaptive': 'Adaptive', 'normalized': 'Normalized'}.get(scoring_mode, scoring_mode)
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

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Total score distribution
        ax = axes[0, 0]
        ax.hist(pure_total, bins=30, alpha=0.6, label=f'Pure Normal (n={pure_normal_mask.sum()})', color=VIS_COLORS['normal'], density=True)
        ax.hist(dist_total, bins=30, alpha=0.6, label=f'Disturbing Normal (n={disturbing_mask.sum()})', color=VIS_COLORS['disturbing'], density=True)
        ax.hist(anom_total, bins=30, alpha=0.6, label=f'Anomaly (n={anomaly_mask.sum()})', color=VIS_COLORS['anomaly'], density=True)
        ax.set_xlabel('Total Score (Recon + λ·Disc)')
        ax.set_ylabel('Density')
        ax.set_title('Score Distribution by Sample Type', fontweight='bold')
        ax.legend()

        # 2. Box plot comparison
        ax = axes[0, 1]
        box_data = [pure_total, dist_total, anom_total]
        labels = ['Pure\nNormal', 'Disturbing\nNormal', 'Anomaly']
        bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
        sample_colors = [VIS_COLORS['normal'], VIS_COLORS['disturbing'], VIS_COLORS['anomaly']]
        for patch, color in zip(bp['boxes'], sample_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel('Total Score')
        ax.set_title('Score Box Plot', fontweight='bold')

        # 3. Discrepancy comparison (key metric)
        ax = axes[0, 2]
        ax.hist(pure_disc, bins=30, alpha=0.6, label='Pure Normal', color=VIS_COLORS['normal'], density=True)
        ax.hist(dist_disc, bins=30, alpha=0.6, label='Disturbing Normal', color=VIS_COLORS['disturbing'], density=True)
        ax.hist(anom_disc, bins=30, alpha=0.6, label='Anomaly', color=VIS_COLORS['anomaly'], density=True)
        ax.set_xlabel('Discrepancy (Teacher-Student)')
        ax.set_ylabel('Density')
        ax.set_title('Discrepancy Distribution', fontweight='bold')
        ax.legend()

        # 4. Teacher vs Student scatter
        ax = axes[1, 0]
        ax.scatter(pure_teacher, pure_student, alpha=0.5, label='Pure Normal', color=VIS_COLORS['normal'], s=20)
        ax.scatter(dist_teacher, dist_student, alpha=0.5, label='Disturbing Normal', color=VIS_COLORS['disturbing'], s=20)
        ax.scatter(anom_teacher, anom_student, alpha=0.5, label='Anomaly', color=VIS_COLORS['anomaly'], s=20)
        max_val = max(np.max([pure_teacher.max(), dist_teacher.max(), anom_teacher.max()]),
                      np.max([pure_student.max(), dist_student.max(), anom_student.max()]))
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='y=x')
        ax.set_xlabel('Teacher Error')
        ax.set_ylabel('Student Error')
        ax.set_title('Teacher vs Student Error', fontweight='bold')
        ax.legend()

        # 5. Mean comparison bar chart
        ax = axes[1, 1]
        x = np.arange(3)
        width = 0.25

        means_teacher = [pure_teacher.mean(), dist_teacher.mean(), anom_teacher.mean()]
        means_student = [pure_student.mean(), dist_student.mean(), anom_student.mean()]
        means_disc = [pure_disc.mean(), dist_disc.mean(), anom_disc.mean()]

        # Use consistent colors: Teacher=green, Student=purple, Discrepancy=anomaly(red)
        bars1 = ax.bar(x - width, means_teacher, width, label='Teacher Error', color=VIS_COLORS['teacher'])
        bars2 = ax.bar(x, means_student, width, label='Student Error', color=VIS_COLORS['student'])
        bars3 = ax.bar(x + width, means_disc, width, label='Discrepancy', color=VIS_COLORS['anomaly'])

        ax.set_xticks(x)
        ax.set_xticklabels(['Pure\nNormal', 'Disturbing\nNormal', 'Anomaly'])
        ax.set_ylabel('Mean Value')
        ax.set_title('Mean Error Components', fontweight='bold')
        ax.legend()

        # 6. Statistics summary
        ax = axes[1, 2]
        ax.axis('off')

        stats_text = f"""
Pure Normal vs Disturbing Normal Analysis
═══════════════════════════════════════════════════════

Sample Counts:
  • Pure Normal:      {pure_normal_mask.sum():>6}
  • Disturbing Normal:{disturbing_mask.sum():>6}
  • Anomaly:          {anomaly_mask.sum():>6}

Mean Total Score:
  • Pure Normal:      {pure_total.mean():.6f}
  • Disturbing Normal:{dist_total.mean():.6f}
  • Anomaly:          {anom_total.mean():.6f}

Mean Discrepancy (Key Metric):
  • Pure Normal:      {pure_disc.mean():.6f}
  • Disturbing Normal:{dist_disc.mean():.6f}
  • Anomaly:          {anom_disc.mean():.6f}

Separation Analysis:
  • Anom/Pure ratio:  {anom_total.mean() / (pure_total.mean() + 1e-8):.2f}x
  • Anom/Dist ratio:  {anom_total.mean() / (dist_total.mean() + 1e-8):.2f}x
  • Dist/Pure ratio:  {dist_total.mean() / (pure_total.mean() + 1e-8):.2f}x
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

        # 1. Discrepancy histogram by sample type
        ax = axes[0, 0]
        bins = np.linspace(0, max(discrepancies.max(), 0.1), 50)

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

        # Create comprehensive 2x3 figure for detailed view
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

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
        """Get the optimal threshold from ROC curve."""
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(self.pred_data['labels'], self.pred_data['scores'])
        optimal_idx = np.argmax(tpr - fpr)
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
                violin_data.append(recon_contrib[mask])
                violin_labels.append(f'{cat}\nRecon')
                violin_colors.append(color + '80')

                violin_data.append(disc_contrib[mask])
                violin_labels.append(f'{cat}\nDisc')
                violin_colors.append(color)

        parts = ax4.violinplot(violin_data, positions=range(len(violin_data)),
                               showmeans=True, showmedians=True)

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

        for mask, color, label in zip(masks, cat_colors, categories):
            if mask.sum() > 10:  # Need enough samples for KDE
                data = recon_contrib[mask]
                # Remove outliers for better visualization
                q1, q99 = np.percentile(data, [1, 99])
                data_clean = data[(data >= q1) & (data <= q99)]
                if len(data_clean) > 10:
                    kde_data = data_clean if len(data_clean) <= _KDE_MAX_SAMPLES else data_clean[np.linspace(0, len(data_clean)-1, _KDE_MAX_SAMPLES, dtype=int)]
                    kde = gaussian_kde(kde_data)
                    x_range = np.linspace(data_clean.min(), data_clean.max(), 200)
                    ax5.plot(x_range, kde(x_range), color=color, linewidth=2, label=label)
                    ax5.fill_between(x_range, kde(x_range), alpha=0.3, color=color)
                # Add mean line
                ax5.axvline(data.mean(), color=color, linestyle='--', alpha=0.7, linewidth=1.5)

        ax5.set_xlabel('Recon Contribution')
        ax5.set_ylabel('Density')
        ax5.set_title('(E) Reconstruction Contribution Distribution', fontsize=12, fontweight='bold')
        ax5.legend(loc='upper right')
        ax5.grid(alpha=0.3)

        # === (F) Bottom-Right: Disc Contribution KDE ===
        ax6 = fig.add_subplot(gs[1, 2])

        for mask, color, label in zip(masks, cat_colors, categories):
            if mask.sum() > 10:  # Need enough samples for KDE
                data = disc_contrib[mask]
                # Remove outliers for better visualization
                q1, q99 = np.percentile(data, [1, 99])
                data_clean = data[(data >= q1) & (data <= q99)]
                if len(data_clean) > 10:
                    kde_data = data_clean if len(data_clean) <= _KDE_MAX_SAMPLES else data_clean[np.linspace(0, len(data_clean)-1, _KDE_MAX_SAMPLES, dtype=int)]
                    kde = gaussian_kde(kde_data)
                    x_range = np.linspace(data_clean.min(), data_clean.max(), 200)
                    ax6.plot(x_range, kde(x_range), color=color, linewidth=2, label=label)
                    ax6.fill_between(x_range, kde(x_range), alpha=0.3, color=color)
                # Add mean line
                ax6.axvline(data.mean(), color=color, linestyle='--', alpha=0.7, linewidth=1.5)

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
            even when experiment config specifies 'adaptive' or 'normalized' mode.
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
                    elif score_mode == 'normalized':
                        # For normalized, we need std which we don't have per-epoch
                        # Use simple scaling based on means as approximation
                        # This gives equal weight to both components
                        recon_contrib = recon
                        disc_contrib = (overall_recon_mean / (overall_disc_mean + 1e-8)) * disc
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
        elif score_mode == 'normalized':
            param_str = f"z-score normalized"
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
                   dpi=150, bbox_inches='tight')
        plt.close()
        print("  - best_model_score_contribution.png")

    def generate_all(self, experiment_dir: str = None, history: Dict = None):
        """Generate all best model visualizations

        Args:
            experiment_dir: Path to experiment directory for loading detailed results
            history: Training history dictionary for learning curve visualization
        """
        print("\n  Generating Best Model Visualizations...")
        self.plot_roc_curve()
        self.plot_confusion_matrix()
        self.plot_score_contribution_analysis(experiment_dir)
        self.plot_reconstruction_examples()
        self.plot_detection_examples()
        self.plot_summary_statistics()
        self.plot_learning_curve(history)
        self.plot_pure_vs_disturbing_normal()
        self.plot_discrepancy_trend()

        # Qualitative case studies
        self.plot_case_study_gallery(experiment_dir)
        self.plot_hardest_samples()

        # Anomaly type analysis (requires detailed results from experiment)
        self.plot_performance_by_anomaly_type(experiment_dir)
        self.plot_score_distribution_by_type(experiment_dir)

        # Anomaly type score trends over epochs (requires history with epoch_anomaly_type_scores)
        self.plot_score_contribution_epoch_trends(experiment_dir, history)

        # ROC curve comparison (different score types)
        self.plot_roc_curve_comparison()
        self.plot_roc_curve_pa80_comparison()

        # Performance by anomaly type comparison (different score types)
        self.plot_performance_by_anomaly_type_comparison(self.output_dir)

    def plot_score_contribution_epoch_trends(self, experiment_dir: str = None, history: Dict = None):
        """Plot epoch-wise score contribution breakdown for each anomaly type

        Creates stacked area plots showing how reconstruction and discrepancy
        scores contribute to total score over epochs, for normal and each anomaly type.
        Similar to plots (J)-(L) in score_contribution_analysis but broken down by
        individual anomaly types.

        Features:
        - One stacked area subplot per type (normal + 6 anomaly types)
        - Unified y-axis across all plots for comparison
        - X-axis starts from epoch 5 (after initial training stabilization)
        - Shows recon score (blue) and disc score (red) stacked

        Args:
            experiment_dir: Path to experiment directory for loading training history
            history: Optional pre-loaded history dict
        """
        from mae_anomaly import SLIDING_ANOMALY_TYPE_NAMES

        # Load history if not provided
        if history is None and experiment_dir:
            history_path = os.path.join(experiment_dir, 'training_histories.json')
            if os.path.exists(history_path):
                with open(history_path) as f:
                    history_data = json.load(f)
                    history = history_data.get('0', {})

        # Check if epoch_anomaly_type_scores is available
        if history is None or 'epoch_anomaly_type_scores' not in history:
            print("  - Skipping score contribution epoch trends (no epoch_anomaly_type_scores in history)")
            return

        epoch_scores = history['epoch_anomaly_type_scores']
        if not epoch_scores:
            return

        epochs = np.array(history.get('epoch', list(range(1, len(epoch_scores) + 1))))

        # Collect anomaly types that appear in the data
        all_types = set()
        for epoch_data in epoch_scores:
            all_types.update(epoch_data.keys())

        # Order: normal first, then anomaly types in order
        anomaly_type_order = ['normal'] + [name for name in SLIDING_ANOMALY_TYPE_NAMES if name in all_types and name != 'normal']
        anomaly_type_order = [t for t in anomaly_type_order if t in all_types]

        n_types = len(anomaly_type_order)
        if n_types == 0:
            return

        # Get scoring mode for title
        score_mode = getattr(self.config, 'anomaly_score_mode', 'default')
        lambda_disc = getattr(self.config, 'lambda_disc', 0.5)

        # Collect data for each type
        # History was saved with 'default' mode (lambda_disc * disc), need to recalculate
        # for other scoring modes
        type_data = {t: {'recon': [], 'disc': [], 'raw_disc': [], 'count': []} for t in anomaly_type_order}

        for epoch_data in epoch_scores:
            for atype in anomaly_type_order:
                if atype in epoch_data:
                    recon = epoch_data[atype].get('recon_score', 0)
                    disc_weighted = epoch_data[atype].get('disc_score', 0)
                    # Reverse default weighting to get raw disc
                    raw_disc = disc_weighted / lambda_disc if lambda_disc > 0 else disc_weighted
                    type_data[atype]['recon'].append(recon)
                    type_data[atype]['raw_disc'].append(raw_disc)
                    type_data[atype]['count'].append(epoch_data[atype].get('count', 0))
                else:
                    type_data[atype]['recon'].append(0)
                    type_data[atype]['raw_disc'].append(0)
                    type_data[atype]['count'].append(0)

        # Compute weighted disc based on current scoring mode
        if score_mode == 'normalized':
            # Z-score normalization: compute all z-scores first, then shift globally
            n_epochs = len(epochs)
            z_scores = {}  # {atype: {'recon_z': [], 'disc_z': []}}

            # First pass: compute z-scores for all types
            for atype in anomaly_type_order:
                recon_arr = np.array(type_data[atype]['recon'])
                raw_disc_arr = np.array(type_data[atype]['raw_disc'])
                z_scores[atype] = {'recon_z': [], 'disc_z': []}

                for i in range(n_epochs):
                    all_recon = [type_data[t]['recon'][i] for t in anomaly_type_order]
                    all_disc = [type_data[t]['raw_disc'][i] for t in anomaly_type_order]
                    recon_mean, recon_std = np.mean(all_recon), np.std(all_recon) + 1e-8
                    disc_mean, disc_std = np.mean(all_disc), np.std(all_disc) + 1e-8
                    recon_z = (recon_arr[i] - recon_mean) / recon_std
                    disc_z = (raw_disc_arr[i] - disc_mean) / disc_std
                    z_scores[atype]['recon_z'].append(recon_z)
                    z_scores[atype]['disc_z'].append(disc_z)

            # Find global minimum across all types and epochs
            all_z_values = []
            for atype in anomaly_type_order:
                all_z_values.extend(z_scores[atype]['recon_z'])
                all_z_values.extend(z_scores[atype]['disc_z'])
            global_min = min(all_z_values) if all_z_values else 0

            # Second pass: shift all values by global minimum
            for atype in anomaly_type_order:
                type_data[atype]['recon'] = [z - global_min for z in z_scores[atype]['recon_z']]
                type_data[atype]['disc'] = [z - global_min for z in z_scores[atype]['disc_z']]

        else:
            # Non-normalized modes: adaptive or default
            for atype in anomaly_type_order:
                recon_arr = np.array(type_data[atype]['recon'])
                raw_disc_arr = np.array(type_data[atype]['raw_disc'])

                if score_mode == 'adaptive':
                    # Per-epoch adaptive lambda
                    disc_weighted = []
                    for i in range(len(recon_arr)):
                        # Use overall epoch means (approximate from available data)
                        all_recon = np.mean([type_data[t]['recon'][i] for t in anomaly_type_order if type_data[t]['recon'][i] > 0])
                        all_disc = np.mean([type_data[t]['raw_disc'][i] for t in anomaly_type_order if type_data[t]['raw_disc'][i] > 0])
                        adaptive_lambda = all_recon / (all_disc + 1e-8)
                        disc_weighted.append(adaptive_lambda * raw_disc_arr[i])
                    type_data[atype]['disc'] = disc_weighted
                else:  # default
                    type_data[atype]['disc'] = list(lambda_disc * raw_disc_arr)

        # Filter to show only epoch >= 5
        start_epoch = 5
        epoch_mask = epochs >= start_epoch
        if not epoch_mask.any():
            print("  - Skipping score contribution epoch trends (not enough epochs)")
            return

        epochs_filtered = epochs[epoch_mask]

        # Filter data for each type and compute unified y-axis
        type_data_filtered = {}
        all_total_scores = []

        for atype in anomaly_type_order:
            recon_f = np.array(type_data[atype]['recon'])[epoch_mask]
            disc_f = np.array(type_data[atype]['disc'])[epoch_mask]
            type_data_filtered[atype] = {'recon': recon_f, 'disc': disc_f}
            all_total_scores.extend(recon_f + disc_f)

        # Unified y-axis limit
        y_max = max(all_total_scores) * 1.1 if all_total_scores else 1

        # Define colors for each type
        type_colors = {
            'normal': VIS_COLORS['normal'],
            'spike': '#E74C3C',
            'memory_leak': '#9B59B6',
            'cpu_saturation': '#3498DB',
            'network_congestion': '#F39C12',
            'cascading_failure': '#1ABC9C',
            'resource_contention': '#34495E',
            'point_spike': '#E91E63'
        }

        # Colors for stacked area (recon = light blue, disc = light red)
        colors_stack = ['#85C1E9', '#F1948A']

        # Create figure - 2 rows, up to 4 columns
        n_cols = min(4, n_types)
        n_rows = (n_types - 1) // n_cols + 1

        fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows))

        for idx, atype in enumerate(anomaly_type_order):
            ax = fig.add_subplot(n_rows, n_cols, idx + 1)

            recon_f = type_data_filtered[atype]['recon']
            disc_f = type_data_filtered[atype]['disc']
            count = type_data[atype]['count'][0] if type_data[atype]['count'] else 0
            color = type_colors.get(atype, '#95A5A6')

            # Stacked area plot
            ax.stackplot(epochs_filtered, recon_f, disc_f,
                        labels=['Recon Score', 'Disc Score'],
                        colors=colors_stack, alpha=0.8)

            # Add border line at top showing total score
            ax.plot(epochs_filtered, recon_f + disc_f, color=color, linewidth=2)

            ax.set_xlabel('Epoch')
            ax.set_ylabel('Anomaly Score')
            ax.set_ylim(0, y_max)
            ax.set_xlim(start_epoch, epochs_filtered[-1] if len(epochs_filtered) > 0 else start_epoch)
            ax.set_title(f'{atype}\n(n={count})', fontsize=11, fontweight='bold', color=color)
            ax.grid(alpha=0.3)

            # Only show legend on first plot
            if idx == 0:
                ax.legend(loc='upper right', fontsize=9)

        # Build title with scoring mode info
        fig.suptitle(f'Score Contribution Trends by Anomaly Type (Epoch ≥ 5)\nScoring Mode: {score_mode}',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'best_model_score_contribution_trends.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()
        print("  - best_model_score_contribution_trends.png")

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

                # Find optimal point and compute F1
                optimal_idx = np.argmax(tpr - fpr)
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

    def plot_roc_curve_pa80_comparison(self):
        """Plot PA%80 ROC curves comparing different scoring methods.

        Uses voting threshold variation approach: for each threshold T, the voting
        aggregation converts window scores > T to 1 (vote for anomaly), then majority
        vote determines the point-level prediction. This directly evaluates the voting
        mechanism at different operating points.

        Compares:
        - Anomaly Score (combined): recon + lambda * disc (based on scoring mode)
        - Discrepancy Only: teacher-student difference
        - Teacher Recon Only: teacher reconstruction error
        - Student Recon Only: student reconstruction error
        """
        # Check if we can use segment-based PA%K (handle Subset wrapping)
        raw_dataset = self.test_loader.dataset if hasattr(self.test_loader, 'dataset') else None
        base_dataset = _unwrap_subset(raw_dataset) if raw_dataset is not None else None
        can_use_segment_pa_k = (
            base_dataset is not None and
            hasattr(base_dataset, 'anomaly_regions') and
            hasattr(base_dataset, 'point_labels') and
            hasattr(base_dataset, 'window_start_indices') and
            len(base_dataset.anomaly_regions) > 0
        )

        if not can_use_segment_pa_k:
            print("  - best_model_roc_curve_PA80_comparison.png (SKIPPED: no segment info)")
            return

        # Use PATCH-level scores for voting (not point-level)
        patch_recon = self.pred_data['recon_errors']
        patch_student = self.pred_data['student_errors']
        patch_disc = self.pred_data['discrepancies']
        patch_combined = self.pred_data['patch_scores']

        # Get point-level data (use base dataset for signal-level attrs, subset-aware for window indices)
        total_length = len(base_dataset.point_labels)
        point_labels = np.array(base_dataset.point_labels)
        anomaly_regions = base_dataset.anomaly_regions
        _, window_start_indices = _get_subset_window_indices(raw_dataset)

        # Use pred_data dimensions (may differ from dataset if collected separately)
        n_windows = self.pred_data.get('n_windows', len(window_start_indices))
        num_patches = self.pred_data.get('num_patches', getattr(self.config, 'num_patches', 10))

        # Convert patch-level scores to 2D (n_windows, num_patches) for voting
        def prepare_scores(scores):
            return scores.reshape(n_windows, num_patches)

        # Scoring methods to compare
        scoring_methods = [
            ('Anomaly Score (combined)', prepare_scores(patch_combined), VIS_COLORS['total']),
            ('Discrepancy Only', prepare_scores(patch_disc), VIS_COLORS['discrepancy']),
            ('Teacher Recon Only', prepare_scores(patch_recon), VIS_COLORS['teacher']),
            ('Student Recon Only', prepare_scores(patch_student), VIS_COLORS['student']),
        ]

        fig, ax = plt.subplots(figsize=(10, 10))

        # Track offset for annotations
        annotation_offsets = [
            (0.02, 0.02),
            (-0.18, -0.08),
            (0.02, -0.08),
            (-0.18, 0.02),
        ]
        short_names = ['Anomaly', 'Discr.', 'Teacher', 'Student']

        # OPTIMIZATION: Precompute point_indices once (shared across all scoring methods)
        n_thresholds = 100  # High resolution for smooth ROC curves
        pt_flat_t, pt_flat_si, point_coverage = precompute_point_score_indices(
            window_start_indices=window_start_indices,
            seq_length=self.config.seq_length,
            patch_size=self.config.patch_size,
            total_length=total_length,
            num_patches=num_patches,
        )

        # Create evaluation mask (no type filtering for overall PA%80)
        eval_type_mask = np.ones(total_length, dtype=bool)

        for idx, (name, scores, color) in enumerate(scoring_methods):
            # OPTIMIZATION: Use vectorized voting with precomputed indices
            min_score, max_score = scores.min(), scores.max()
            thresholds = np.linspace(min_score - 0.01, max_score + 0.01, n_thresholds)

            # Vectorized voting for all thresholds at once
            point_scores_all = vectorized_voting_for_all_thresholds(
                scores=scores,
                point_indices=(pt_flat_t, pt_flat_si),
                point_coverage=point_coverage,
                thresholds=thresholds,
            )

            # Compute ROC using the helper function
            result = _compute_single_pa_k_roc((
                point_scores_all,
                point_labels,
                anomaly_regions,
                eval_type_mask,
                80,  # k_percent
                None,  # anomaly_type (None = all types)
                True  # return_optimal_f1
            ))

            fprs_sorted, tprs_sorted, pa80_auc, pa80_f1 = result

            ax.plot(fprs_sorted, tprs_sorted, color=color, lw=2,
                   label=f'{name} (AUC={pa80_auc:.4f}, F1={pa80_f1:.4f})')

            # Find optimal point (max TPR - FPR)
            if len(fprs_sorted) > 0:
                optimal_idx = np.argmax(tprs_sorted - fprs_sorted)
                opt_fpr = fprs_sorted[optimal_idx]
                opt_tpr = tprs_sorted[optimal_idx]
                ax.scatter(opt_fpr, opt_tpr, s=80, color=color, zorder=5, marker='o')

                # Annotation with AUC and F1
                offset_x, offset_y = annotation_offsets[idx]
                short_name = short_names[idx]
                ax.annotate(f'{short_name}\nAUC={pa80_auc:.3f}\nF1={pa80_f1:.3f}',
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
        ax.set_ylabel('True Positive Rate (PA%80 Adjusted)')

        scoring_mode = getattr(self.config, 'anomaly_score_mode', 'default')
        ax.set_title(f'PA%80 ROC Curve Comparison (Voting Threshold)\n(Segment-Based, Scoring Mode: {scoring_mode})', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'best_model_roc_curve_PA80_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - best_model_roc_curve_PA80_comparison.png")

    def plot_performance_by_anomaly_type_comparison(self, experiment_dir: str = None):
        """Plot detection performance by anomaly type comparing different scoring methods.

        Creates a 4x4 grid:
        - Rows: 4 scoring methods (Combined, Discrepancy, Teacher Recon, Student Recon)
        - Cols: Point-wise Detection Rate, All PA%K, PA%80 Detection Rate, Mean Score

        Args:
            experiment_dir: Path to experiment directory (optional, for compatibility)
        """
        labels = self.pred_data['labels']  # point-level
        combined_scores = self.pred_data['scores']  # point-level
        point_recon = self.pred_data.get('point_recon', combined_scores)
        point_disc = self.pred_data.get('point_disc', combined_scores)
        point_student = self.pred_data.get('point_student', combined_scores)

        # Build point-level anomaly_types from anomaly_regions
        raw_dataset = self.test_loader.dataset if hasattr(self.test_loader, 'dataset') else None
        base_dataset = _unwrap_subset(raw_dataset) if raw_dataset is not None else None
        total_length = len(labels)
        anomaly_types = np.zeros(total_length, dtype=int)
        if base_dataset is not None and hasattr(base_dataset, 'anomaly_regions'):
            for region in base_dataset.anomaly_regions:
                start = region.start
                end = min(region.end, total_length)
                anomaly_types[start:end] = region.anomaly_type

        # Build point-level sample_types from point_labels
        # 0=pure normal, 1=disturbing normal (not used at point-level), 2=anomaly
        point_sample_types = np.zeros(total_length, dtype=int)
        point_sample_types[labels == 1] = 2

        # Scoring methods: point-level scores for threshold/detection,
        # patch-level scores stored separately for voting
        scoring_methods = [
            ('Anomaly Score', combined_scores),
            ('Discrepancy', point_disc),
            ('Teacher Recon', point_recon),
            ('Student Recon', point_student),
        ]
        # Patch-level scores for voting (keyed by scoring method name)
        patch_scores_by_method = {
            'Anomaly Score': self.pred_data.get('patch_scores'),
            'Discrepancy': self.pred_data.get('discrepancies'),
            'Teacher Recon': self.pred_data.get('recon_errors'),
            'Student Recon': self.pred_data.get('student_errors'),
        }

        # Check if test_dataset has anomaly_regions for segment-based PA%K (handle Subset)
        can_use_segment_pa_k = (
            base_dataset is not None and
            hasattr(base_dataset, 'anomaly_regions') and
            hasattr(base_dataset, 'point_labels') and
            hasattr(base_dataset, 'window_start_indices') and
            len(base_dataset.anomaly_regions) > 0
        )

        # Pre-compute voting-based PA%K metrics for each scoring method
        # Uses a SINGLE point-level threshold per scoring method for all K values.
        # PA%K only varies the segment detection criterion (K%), not the threshold.
        pa_k_voted_preds_by_method = {}
        if can_use_segment_pa_k:
            total_length = len(base_dataset.point_labels)
            num_patches = getattr(self.config, 'num_patches', 10)
            _, window_start_indices = _get_subset_window_indices(raw_dataset)
            n_windows = len(window_start_indices)
            point_labels_full = np.array(base_dataset.point_labels)

            # Precompute point_indices ONCE (shared across all methods)
            pt_flat_t, pt_flat_si, point_coverage = precompute_point_score_indices(
                window_start_indices=window_start_indices,
                seq_length=self.config.seq_length,
                patch_size=self.config.patch_size,
                total_length=total_length,
                num_patches=num_patches,
            )

            # For each scoring method, compute voted predictions at that method's
            # single point-level optimal threshold
            for score_name, point_scores_method in scoring_methods:
                patch_scores_flat = patch_scores_by_method.get(score_name)
                if patch_scores_flat is None or len(patch_scores_flat) != n_windows * num_patches:
                    continue
                window_scores = patch_scores_flat.reshape(n_windows, num_patches)

                # Find point-level optimal threshold for this scoring method
                if len(np.unique(labels)) > 1:
                    fpr_m, tpr_m, thresholds_m = roc_curve(labels, point_scores_method)
                    optimal_idx_m = np.argmax(tpr_m - fpr_m)
                    method_threshold = thresholds_m[optimal_idx_m]
                else:
                    method_threshold = np.median(point_scores_method)

                # Aggregate patch scores to point-level via mean for threshold
                point_scores_agg, _ = aggregate_patch_scores_to_point_level(
                    window_scores, window_start_indices, self.config.seq_length,
                    self.config.patch_size, num_patches, total_length, method='mean'
                )
                point_scores_agg = np.nan_to_num(point_scores_agg, nan=0.0)

                # Compute voted binary predictions at the single threshold
                voted_preds = _compute_voted_point_predictions(
                    window_scores, (pt_flat_t, pt_flat_si), point_coverage,
                    method_threshold, total_length
                )

                pa_k_voted_preds_by_method[score_name] = voted_preds

        fig, axes = plt.subplots(4, 4, figsize=(24, 20))
        anomaly_colors = get_anomaly_colors()

        # Store baseline (Anomaly Score) detection rates for reference lines
        baseline_detection_rates = None
        baseline_pa_80_rates = None
        baseline_anomaly_types_list = None

        for row_idx, (score_name, scores) in enumerate(scoring_methods):
            # Compute optimal threshold for this scoring method
            if len(np.unique(labels)) > 1:
                fpr, tpr, thresholds = roc_curve(labels, scores)
                optimal_idx = np.argmax(tpr - fpr)
                threshold = thresholds[optimal_idx]
            else:
                threshold = np.median(scores)

            predictions = (scores > threshold).astype(int)

            # Collect metrics per anomaly type
            anomaly_types_list = []
            detection_rates = []
            pa_10_rates = []
            pa_20_rates = []
            pa_50_rates = []
            pa_80_rates = []
            mean_scores_list = []
            normal_mean_score = 0.0
            disturbing_mean_score = 0.0

            # Get normal mean scores (point-level: label==0 is normal)
            normal_mask = (labels == 0)
            anomaly_mask = (labels == 1)
            if normal_mask.sum() > 0:
                normal_mean_score = float(scores[normal_mask].mean())
            if anomaly_mask.sum() > 0:
                disturbing_mean_score = float(scores[anomaly_mask].mean())

            # Process each anomaly type (skip normal=0)
            for atype_idx in range(1, 10):  # anomaly types 1-9
                atype_name = ANOMALY_TYPE_NAMES[atype_idx] if atype_idx < len(ANOMALY_TYPE_NAMES) else f'type_{atype_idx}'
                type_mask = (anomaly_types == atype_idx)

                if type_mask.sum() == 0:
                    continue

                type_scores = scores[type_mask]
                type_labels = labels[type_mask]
                type_predictions = predictions[type_mask]

                anomaly_types_list.append(atype_name)

                # Detection rate: compute for actual anomaly samples (label==1) only
                anomaly_sample_mask = (type_labels == 1)
                if anomaly_sample_mask.sum() > 0:
                    # Detection rate = fraction of actual anomalies that were predicted as anomalies
                    det_rate = float(type_predictions[anomaly_sample_mask].mean())
                    detection_rates.append(det_rate * 100)

                    # PA%K detection rates — single threshold for all K values.
                    # PA%K only varies the segment detection criterion (K%),
                    # not the binary prediction threshold.
                    if can_use_segment_pa_k and score_name in pa_k_voted_preds_by_method:
                        voted_preds = pa_k_voted_preds_by_method[score_name]

                        for k, rate_list in [(10, pa_10_rates), (20, pa_20_rates),
                                             (50, pa_50_rates), (80, pa_80_rates)]:
                            pa_rate = compute_segment_pa_k_detection_rate(
                                point_scores=voted_preds,
                                point_labels=base_dataset.point_labels,
                                anomaly_regions=base_dataset.anomaly_regions,
                                anomaly_type=atype_idx,
                                threshold=0.5,
                                k_percent=k
                            )
                            rate_list.append(float(pa_rate) * 100)
                    else:
                        # Fallback: sample-level PA%K (approximate)
                        for k, rate_list in [(10, pa_10_rates), (20, pa_20_rates),
                                             (50, pa_50_rates), (80, pa_80_rates)]:
                            adjusted = compute_pa_k_adjusted_predictions(type_predictions, type_labels, k_percent=k)
                            rate_list.append(float(adjusted[anomaly_sample_mask].mean()) * 100)
                else:
                    detection_rates.append(0)
                    pa_10_rates.append(0)
                    pa_20_rates.append(0)
                    pa_50_rates.append(0)
                    pa_80_rates.append(0)

                mean_scores_list.append(float(type_scores.mean()))

            if len(anomaly_types_list) == 0:
                continue

            # Store baseline (first row = Anomaly Score) for reference lines
            if row_idx == 0:
                baseline_detection_rates = detection_rates.copy()
                baseline_pa_80_rates = pa_80_rates.copy()
                baseline_anomaly_types_list = anomaly_types_list.copy()

            colors = [anomaly_colors.get(at, VIS_COLORS['reference']) for at in anomaly_types_list]
            display_names = [at.replace('_', '\n') for at in anomaly_types_list]
            x = np.arange(len(display_names))

            # Column 1: Point-wise Detection Rate
            ax = axes[row_idx, 0]
            bars = ax.bar(display_names, detection_rates, color=colors, alpha=0.8, edgecolor=VIS_COLORS['baseline'])
            ax.set_ylabel('Detection Rate (%)')
            ax.set_title(f'{score_name}: Point-wise Detection Rate', fontweight='bold', fontsize=10)
            ax.set_ylim(0, 110)
            for bar, rate in zip(bars, detection_rates):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                       f'{rate:.0f}%', ha='center', va='bottom', fontsize=6)
            ax.tick_params(axis='x', labelsize=7)

            # Add baseline reference markers for rows 1-3 (not row 0)
            if row_idx > 0 and baseline_detection_rates is not None:
                for i, (atype_name, baseline_rate) in enumerate(zip(baseline_anomaly_types_list, baseline_detection_rates)):
                    if atype_name in anomaly_types_list:
                        idx = anomaly_types_list.index(atype_name)
                        ax.hlines(y=baseline_rate, xmin=idx-0.4, xmax=idx+0.4,
                                 colors=VIS_COLORS['total'], linestyles='dashed', linewidth=1.5, alpha=0.7)
                if row_idx == 1:
                    ax.plot([], [], color=VIS_COLORS['total'], linestyle='--', linewidth=1.5, label='Anomaly Score')
                    ax.legend(loc='lower right', fontsize=6)

            # Column 2: All PA%K Methods Comparison
            ax = axes[row_idx, 1]
            width = 0.18
            bars1 = ax.bar(x - 2*width, detection_rates, width, label='Point-wise', color=colors, alpha=0.3)
            bars2 = ax.bar(x - width, pa_10_rates, width, label='PA%10', color=colors, alpha=0.5)
            bars3 = ax.bar(x, pa_20_rates, width, label='PA%20', color=colors, alpha=0.7)
            bars4 = ax.bar(x + width, pa_50_rates, width, label='PA%50', color=colors, alpha=0.85)
            bars5 = ax.bar(x + 2*width, pa_80_rates, width, label='PA%80', color=colors, alpha=1.0)
            ax.set_ylabel('Detection Rate (%)')
            ax.set_title(f'{score_name}: All PA%K Comparison', fontweight='bold', fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels(display_names, fontsize=6)
            ax.set_ylim(0, 115)
            if row_idx == 0:
                ax.legend(loc='upper right', fontsize=6)

            # Column 3: PA%80 Detection Rate
            ax = axes[row_idx, 2]
            bars = ax.bar(display_names, pa_80_rates, color=colors, alpha=0.8, edgecolor=VIS_COLORS['baseline'])
            ax.set_ylabel('Detection Rate (%)')
            ax.set_title(f'{score_name}: PA%80 Detection Rate (strict)', fontweight='bold', fontsize=10)
            ax.set_ylim(0, 110)
            for bar, rate in zip(bars, pa_80_rates):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                       f'{rate:.0f}%', ha='center', va='bottom', fontsize=6)
            ax.tick_params(axis='x', labelsize=7)

            # Add baseline reference markers for rows 1-3 (not row 0)
            if row_idx > 0 and baseline_pa_80_rates is not None:
                for i, (atype_name, baseline_rate) in enumerate(zip(baseline_anomaly_types_list, baseline_pa_80_rates)):
                    if atype_name in anomaly_types_list:
                        idx = anomaly_types_list.index(atype_name)
                        ax.hlines(y=baseline_rate, xmin=idx-0.4, xmax=idx+0.4,
                                 colors=VIS_COLORS['total'], linestyles='dashed', linewidth=1.5, alpha=0.7)
                if row_idx == 1:
                    ax.plot([], [], color=VIS_COLORS['total'], linestyle='--', linewidth=1.5, label='Anomaly Score')
                    ax.legend(loc='lower right', fontsize=6)

            # Column 4: Mean Score by Sample Type
            ax = axes[row_idx, 3]

            # Shift for normalized mode - add small padding so minimum is visible
            scoring_mode = getattr(self.config, 'anomaly_score_mode', 'default')
            all_scores_for_shift = mean_scores_list + [normal_mean_score, disturbing_mean_score]
            if scoring_mode == 'normalized':
                score_min = min(all_scores_for_shift)
                score_range = max(all_scores_for_shift) - score_min
                padding = score_range * 0.05  # 5% padding so minimum bar is visible
            else:
                score_min = 0
                padding = 0

            display_scores = [s - score_min + padding for s in mean_scores_list]
            display_normal = normal_mean_score - score_min + padding
            display_disturbing = disturbing_mean_score - score_min + padding

            bars = ax.bar(display_names, display_scores, color=colors, alpha=0.8, edgecolor=VIS_COLORS['baseline'])
            ax.set_ylabel('Mean Score')
            title_suffix = ' (shifted)' if scoring_mode == 'normalized' else ''
            ax.set_title(f'{score_name}: Mean Score{title_suffix}', fontweight='bold', fontsize=10)

            # Fix: Use percentage of y-axis range for text offset
            y_max = max(display_scores + [display_normal, display_disturbing]) * 1.15
            text_offset = y_max * 0.02
            for bar, score in zip(bars, display_scores):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + text_offset,
                       f'{score:.4f}', ha='center', va='bottom', fontsize=5, rotation=45)
            ax.tick_params(axis='x', labelsize=7)
            ax.set_ylim(0, y_max)

            # Reference lines - show original values in legend
            ax.axhline(y=display_normal, color=VIS_COLORS['pure_normal'], linestyle='--', linewidth=1.5,
                       label=f'Pure Normal ({normal_mean_score:.4f})')
            ax.axhline(y=display_disturbing, color=VIS_COLORS['disturbing'], linestyle='--', linewidth=1.5,
                       label=f'Disturbing ({disturbing_mean_score:.4f})')
            if row_idx == 0:
                ax.legend(loc='upper right', fontsize=6)

        scoring_mode = getattr(self.config, 'anomaly_score_mode', 'default')
        plt.suptitle(f'Performance by Anomaly Type - Scoring Method Comparison\n(Scoring Mode: {scoring_mode})',
                    fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_by_anomaly_type_comparison.png'),
                   dpi=150, bbox_inches='tight')
        plt.close('all')
        # Free large intermediate arrays to prevent memory leak
        import gc; gc.collect()
        print("  - performance_by_anomaly_type_comparison.png")


# =============================================================================
# TrainingProgressVisualizer - Training Progress Analysis
# =============================================================================

