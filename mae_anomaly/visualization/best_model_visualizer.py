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

from mae_anomaly import Config, ANOMALY_TYPE_NAMES, ANOMALY_CATEGORY
from .base import (
    get_anomaly_colors, SAMPLE_TYPE_NAMES, SAMPLE_TYPE_COLORS,
    VIS_COLORS, VIS_MARKERS, VIS_LINESTYLES,
    collect_predictions, collect_detailed_data,
    compute_score_contributions,
)

class BestModelVisualizer:
    """Visualize best model analysis"""

    def __init__(self, model, config: Config, test_loader, output_dir: str):
        self.model = model
        self.config = config
        self.test_loader = test_loader
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Check inference mode
        self.inference_mode = getattr(config, 'inference_mode', 'last_patch')
        self.num_patches = getattr(config, 'num_patches', 10)

        # Collect data
        print("  Collecting model predictions...")
        self.pred_data = collect_predictions(model, test_loader, config)
        print("  Collecting detailed data...")
        self.detailed_data = collect_detailed_data(model, test_loader, config)

    def _patch_idx_to_window_idx(self, patch_idx: int) -> int:
        """Convert patch-level index to window-level index for all_patches mode.

        In all_patches mode, pred_data has shape (n_windows * num_patches,)
        but detailed_data has shape (n_windows,).
        """
        if self.inference_mode == 'all_patches':
            return patch_idx // self.num_patches
        return patch_idx

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
        fpr, tpr, thresholds = roc_curve(self.pred_data['labels'], self.pred_data['scores'])
        roc_auc = auc(fpr, tpr)

        # Find optimal threshold
        optimal_idx = np.argmax(tpr - fpr)
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
        fpr, tpr, thresholds = roc_curve(self.pred_data['labels'], self.pred_data['scores'])
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        predictions = (self.pred_data['scores'] >= optimal_threshold).astype(int)
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
        """Show reconstruction examples"""
        normal_idx = np.where(self.detailed_data['labels'] == 0)[0]
        anomaly_idx = np.where(self.detailed_data['labels'] == 1)[0]

        np.random.seed(42)
        normal_samples = np.random.choice(normal_idx, min(num_examples, len(normal_idx)), replace=False)
        anomaly_samples = np.random.choice(anomaly_idx, min(num_examples, len(anomaly_idx)), replace=False)
        all_samples = list(normal_samples) + list(anomaly_samples)
        sample_labels = ['Normal'] * len(normal_samples) + ['Anomaly'] * len(anomaly_samples)

        fig, axes = plt.subplots(len(all_samples), 3, figsize=(15, 4 * len(all_samples)))

        for row, (idx, label) in enumerate(zip(all_samples, sample_labels)):
            original = self.detailed_data['originals'][idx, :, 0]
            teacher = self.detailed_data['teacher_recons'][idx, :, 0]
            student = self.detailed_data['student_recons'][idx, :, 0]
            mask = self.detailed_data['masks'][idx]
            disc = self.detailed_data['discrepancies'][idx]
            point_labels = self.detailed_data['point_labels'][idx]

            seq_len = len(original)
            x = np.arange(seq_len)

            # In all_patches mode, mask is all zeros (each patch was masked in turn)
            # Use mask_last_n for visualization consistency
            if self.inference_mode == 'all_patches':
                # Don't show masked region for all_patches mode (all positions evaluated)
                mask_start, mask_end = seq_len, seq_len
            else:
                masked_region = np.where(mask == 0)[0]
                if len(masked_region) > 0:
                    mask_start, mask_end = masked_region[0], masked_region[-1]
                else:
                    mask_start, mask_end = seq_len, seq_len

            # Original vs Teacher
            ax = axes[row, 0] if len(all_samples) > 1 else axes[0]
            self._highlight_anomaly_regions(ax, point_labels, alpha=0.3, label='Anomaly')
            ax.plot(x, original, 'b-', label='Original', alpha=0.8)
            ax.plot(x, teacher, 'g--', label='Teacher', alpha=0.8)
            ax.axvspan(mask_start, mask_end, alpha=0.2, color=VIS_COLORS['masked_region'], label='Masked')
            ax.set_title(f'{label} - Original vs Teacher')
            ax.legend(fontsize=8)

            # Original vs Student
            ax = axes[row, 1] if len(all_samples) > 1 else axes[1]
            self._highlight_anomaly_regions(ax, point_labels, alpha=0.3, label='Anomaly')
            ax.plot(x, original, 'b-', label='Original', alpha=0.8)
            ax.plot(x, student, 'r--', label='Student', alpha=0.8)
            ax.axvspan(mask_start, mask_end, alpha=0.2, color=VIS_COLORS['masked_region'], label='Masked')
            ax.set_title(f'{label} - Original vs Student')
            ax.legend(fontsize=8)

            # Discrepancy
            ax = axes[row, 2] if len(all_samples) > 1 else axes[2]
            self._highlight_anomaly_regions(ax, point_labels, alpha=0.3, label='Anomaly')
            ax.plot(x, disc, color=VIS_COLORS['student'], lw=2)
            ax.axvspan(mask_start, mask_end, alpha=0.2, color=VIS_COLORS['masked_region'], label='Masked')
            ax.axhline(y=disc.mean(), color=VIS_COLORS['disturbing'], linestyle='--', label=f'Mean: {disc.mean():.4f}')
            ax.set_title(f'{label} - Discrepancy Profile')
            ax.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'best_model_reconstruction.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - best_model_reconstruction.png")

    def plot_detection_examples(self):
        """Show TP, TN, FP, FN examples with anomaly region and masked region highlighted"""
        fpr, tpr, thresholds = roc_curve(self.pred_data['labels'], self.pred_data['scores'])
        optimal_idx = np.argmax(tpr - fpr)
        threshold = thresholds[optimal_idx]

        predictions = (self.pred_data['scores'] >= threshold).astype(int)
        labels = self.pred_data['labels']

        # Find examples
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
                patch_idx = indices[0]
                window_idx = self._patch_idx_to_window_idx(patch_idx)
                original = self.detailed_data['originals'][window_idx, :, 0]
                point_labels = self.detailed_data['point_labels'][window_idx]
                x = np.arange(len(original))

                # Highlight anomaly regions first (so they appear behind the line)
                self._highlight_anomaly_regions(ax, point_labels, alpha=0.3, label='Anomaly Region')

                # Highlight masked region (only for last_patch mode)
                if self.inference_mode != 'all_patches':
                    mask_start = len(original) - self.config.mask_last_n
                    ax.axvspan(mask_start, len(original), alpha=0.2, color=VIS_COLORS['masked_region'], label='Masked Region')

                ax.plot(x, original, color=color, lw=2, label='Signal')
                ax.set_title(f'{title}\nScore: {self.pred_data["scores"][patch_idx]:.4f}, '
                           f'Threshold: {threshold:.4f}', fontweight='bold')
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Value')
                ax.legend(fontsize=8, loc='upper right')
            else:
                ax.text(0.5, 0.5, f'No {title} examples', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title, fontweight='bold')

        plt.suptitle('Detection Examples\n(Red=Anomaly Region, Yellow=Masked Region)',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'best_model_detection_examples.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - best_model_detection_examples.png")

    def plot_summary_statistics(self):
        """Plot summary statistics"""
        normal_mask = self.detailed_data['labels'] == 0
        anomaly_mask = self.detailed_data['labels'] == 1
        mask_inverse = 1 - self.detailed_data['masks']

        # Compute statistics
        teacher_normal = self.detailed_data['teacher_errors'][normal_mask].mean()
        teacher_anomaly = self.detailed_data['teacher_errors'][anomaly_mask].mean()
        student_normal = self.detailed_data['student_errors'][normal_mask].mean()
        student_anomaly = self.detailed_data['student_errors'][anomaly_mask].mean()

        disc_normal = (self.detailed_data['discrepancies'][normal_mask] * mask_inverse[normal_mask]).sum() / \
                     (mask_inverse[normal_mask].sum() + 1e-8)
        disc_anomaly = (self.detailed_data['discrepancies'][anomaly_mask] * mask_inverse[anomaly_mask]).sum() / \
                      (mask_inverse[anomaly_mask].sum() + 1e-8)

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

    def plot_loss_by_anomaly_type(self, experiment_dir: str = None):
        """Plot loss distributions by anomaly type

        Args:
            experiment_dir: Path to experiment directory containing best_model_detailed.csv
        """
        # Try to load detailed results from file
        detailed_csv = None
        if experiment_dir:
            csv_path = os.path.join(experiment_dir, 'best_model_detailed.csv')
            if os.path.exists(csv_path):
                detailed_csv = pd.read_csv(csv_path)

        if detailed_csv is None:
            print("  ! Skipping loss_by_anomaly_type (no detailed CSV found)")
            return

        # Get unique anomaly types present in data
        anomaly_types_present = detailed_csv['anomaly_type_name'].unique()

        # Setup colors dynamically from base module
        colors = get_anomaly_colors()

        n_types = len(ANOMALY_TYPE_NAMES)
        n_cols = 4
        n_rows = (n_types + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        axes = axes.flatten()

        for idx, atype in enumerate(ANOMALY_TYPE_NAMES):
            if idx >= len(axes):
                break
            ax = axes[idx]
            type_data = detailed_csv[detailed_csv['anomaly_type_name'] == atype]

            if len(type_data) == 0:
                ax.text(0.5, 0.5, f'No {atype} samples', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(atype.replace('_', ' ').title(), fontweight='bold')
                continue

            # Create violin plot for both loss types
            data_to_plot = [
                type_data['reconstruction_loss'].values,
                type_data['discrepancy_loss'].values
            ]

            parts = ax.violinplot(data_to_plot, positions=[1, 2], showmeans=True, showmedians=True)

            # Color the violins
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors.get(atype, VIS_COLORS['reference']))
                pc.set_alpha(0.7)

            ax.set_xticks([1, 2])
            ax.set_xticklabels(['Recon Loss', 'Discrepancy'])
            ax.set_title(f"{atype.replace('_', ' ').title()} (n={len(type_data)})", fontweight='bold')
            ax.set_ylabel('Loss Value')

            # Add statistics
            recon_mean = type_data['reconstruction_loss'].mean()
            disc_mean = type_data['discrepancy_loss'].mean()
            ax.text(0.95, 0.95, f'R: {recon_mean:.4f}\nD: {disc_mean:.4f}',
                   transform=ax.transAxes, ha='right', va='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Compute global y-limits for consistent comparison across anomaly types
        all_recon = detailed_csv['reconstruction_loss'].values
        all_disc = detailed_csv['discrepancy_loss'].values
        global_max = max(all_recon.max(), all_disc.max())
        global_min = min(0, all_recon.min(), all_disc.min())

        # Apply unified y-axis to all subplots
        for idx in range(min(len(ANOMALY_TYPE_NAMES), len(axes))):
            axes[idx].set_ylim(global_min, global_max * 1.1)

        # Hide empty subplots
        for idx in range(len(ANOMALY_TYPE_NAMES), len(axes)):
            axes[idx].axis('off')

        plt.suptitle('Loss Distribution by Anomaly Type (unified y-axis)', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'loss_by_anomaly_type.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - loss_by_anomaly_type.png")

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

        for atype, metrics in metrics_json.items():
            if atype == 'normal':
                continue  # Skip normal for this plot
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
        for bar, rate in zip(bars, detection_rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{rate:.0f}%', ha='center', va='bottom', fontsize=7)

        # Row 1, Col 2: PA%10 Detection Rate
        ax = axes[0, 1]
        bars = ax.bar(anomaly_types_display, pa_10_detection_rates, color=colors, alpha=0.8, edgecolor=VIS_COLORS['baseline'])
        ax.set_ylabel('Detection Rate (%)')
        ax.set_title('PA%10 Detection Rate (lenient)', fontweight='bold')
        ax.set_ylim(0, 110)
        for bar, rate in zip(bars, pa_10_detection_rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{rate:.0f}%', ha='center', va='bottom', fontsize=7)

        # Row 1, Col 3: PA%80 Detection Rate
        ax = axes[0, 2]
        bars = ax.bar(anomaly_types_display, pa_80_detection_rates, color=colors, alpha=0.8, edgecolor=VIS_COLORS['baseline'])
        ax.set_ylabel('Detection Rate (%)')
        ax.set_title('PA%80 Detection Rate (strict)', fontweight='bold')
        ax.set_ylim(0, 110)
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
        ax.set_xticklabels(anomaly_types_display, fontsize=7)
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
        ax.set_xticklabels(anomaly_types_display, fontsize=7)
        ax.set_ylim(0, 115)
        ax.legend(loc='upper right', fontsize=8)

        # Row 2, Col 3: Mean Anomaly Score
        ax = axes[1, 2]
        bars = ax.bar(anomaly_types_display, mean_scores, color=colors, alpha=0.8, edgecolor=VIS_COLORS['baseline'])
        ax.set_ylabel('Mean Anomaly Score')
        ax.set_title('Mean Score by Anomaly Type', fontweight='bold')
        for bar, score in zip(bars, mean_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                   f'{score:.4f}', ha='center', va='bottom', fontsize=7)

        # ===== Row 3: Summary Statistics =====
        # Row 3, Col 1: Detection Rate Drop (PA%10 - PA%80)
        ax = axes[2, 0]
        rate_drop = [pa10 - pa80 for pa10, pa80 in zip(pa_10_detection_rates, pa_80_detection_rates)]
        bars = ax.bar(anomaly_types_display, rate_drop, color=colors, alpha=0.8, edgecolor=VIS_COLORS['baseline'])
        ax.set_ylabel('Rate Drop (%)')
        ax.set_title('Detection Consistency (PA%10 - PA%80)', fontweight='bold')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        for bar, drop in zip(bars, rate_drop):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
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

    def plot_value_vs_pattern_comparison(self, experiment_dir: str = None):
        """Compare detection performance: value-based vs pattern-based anomalies

        Value-based anomalies (types 1-7): Values deviate from normal range
        Pattern-based anomalies (types 8-10): Values in normal range, patterns differ

        This visualization helps answer: Is the model detecting based on unusual VALUES
        or based on unusual PATTERNS?

        Args:
            experiment_dir: Path to experiment directory containing detailed CSV and metrics JSON
        """
        # Load detailed CSV
        detailed_csv = None
        if experiment_dir:
            csv_path = os.path.join(experiment_dir, 'best_model_detailed.csv')
            if os.path.exists(csv_path):
                detailed_csv = pd.read_csv(csv_path)

        if detailed_csv is None:
            print("  ! Skipping value_vs_pattern_comparison (no detailed CSV found)")
            return

        # Add category column based on ANOMALY_CATEGORY
        def get_category(row):
            atype_name = row['anomaly_type_name']
            if atype_name == 'normal':
                return 'normal'
            atype_idx = ANOMALY_TYPE_NAMES.index(atype_name) if atype_name in ANOMALY_TYPE_NAMES else -1
            return ANOMALY_CATEGORY.get(atype_idx, 'unknown')

        detailed_csv['category'] = detailed_csv.apply(get_category, axis=1)

        # Separate data by category
        normal_data = detailed_csv[detailed_csv['category'] == 'normal']
        value_data = detailed_csv[detailed_csv['category'] == 'value']
        pattern_data = detailed_csv[detailed_csv['category'] == 'pattern']

        if len(value_data) == 0 and len(pattern_data) == 0:
            print("  ! Skipping value_vs_pattern_comparison (no anomaly data)")
            return

        # Create figure with 2x2 layout
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Colors for categories
        cat_colors = {
            'normal': VIS_COLORS['normal'],
            'value': VIS_COLORS['anomaly'],  # Red for value-based
            'pattern': VIS_COLORS['discrepancy']  # Blue for pattern-based
        }

        # ===== Plot 1: Score Distribution Comparison =====
        ax = axes[0, 0]
        if len(normal_data) > 0:
            ax.hist(normal_data['total_loss'], bins=50, alpha=0.5, label=f'Normal (n={len(normal_data)})',
                   color=cat_colors['normal'], density=True)
        if len(value_data) > 0:
            ax.hist(value_data['total_loss'], bins=50, alpha=0.5, label=f'Value-based (n={len(value_data)})',
                   color=cat_colors['value'], density=True)
        if len(pattern_data) > 0:
            ax.hist(pattern_data['total_loss'], bins=50, alpha=0.5, label=f'Pattern-based (n={len(pattern_data)})',
                   color=cat_colors['pattern'], density=True)
        ax.set_xlabel('Anomaly Score')
        ax.set_ylabel('Density')
        ax.set_title('Score Distribution: Value vs Pattern Anomalies', fontweight='bold')
        ax.legend()

        # ===== Plot 2: Box Plot Comparison =====
        ax = axes[0, 1]
        categories = []
        scores = []
        for cat_name, cat_data in [('Normal', normal_data), ('Value-based', value_data), ('Pattern-based', pattern_data)]:
            if len(cat_data) > 0:
                categories.extend([cat_name] * len(cat_data))
                scores.extend(cat_data['total_loss'].values)

        if len(categories) > 0:
            plot_df = pd.DataFrame({'Category': categories, 'Score': scores})
            box_colors = [cat_colors['normal'], cat_colors['value'], cat_colors['pattern']]
            box_colors = box_colors[:len(plot_df['Category'].unique())]
            bp = ax.boxplot([plot_df[plot_df['Category'] == cat]['Score'].values
                            for cat in plot_df['Category'].unique()],
                           labels=plot_df['Category'].unique(), patch_artist=True)
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        ax.set_ylabel('Anomaly Score')
        ax.set_title('Score Comparison by Category', fontweight='bold')

        # ===== Plot 3: Detection Rate Bar Chart =====
        ax = axes[1, 0]
        # Calculate detection rates using threshold from self.threshold
        threshold = self.threshold

        detection_rates = []
        category_names = []
        sample_counts = []

        for cat_name, cat_data, cat_key in [('Normal\n(FPR)', normal_data, 'normal'),
                                              ('Value-based', value_data, 'value'),
                                              ('Pattern-based', pattern_data, 'pattern')]:
            if len(cat_data) > 0:
                if cat_key == 'normal':
                    # For normal, detection rate means FPR (false positives)
                    rate = (cat_data['total_loss'] >= threshold).mean() * 100
                else:
                    # For anomalies, detection rate means TPR
                    rate = (cat_data['total_loss'] >= threshold).mean() * 100
                detection_rates.append(rate)
                category_names.append(cat_name)
                sample_counts.append(len(cat_data))

        if len(category_names) > 0:
            colors = [cat_colors.get('normal' if 'Normal' in cat else 'value' if 'Value' in cat else 'pattern', 'gray')
                     for cat in category_names]
            bars = ax.bar(category_names, detection_rates, color=colors, alpha=0.8, edgecolor=VIS_COLORS['baseline'])

            for bar, rate, count in zip(bars, detection_rates, sample_counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{rate:.1f}%\n(n={count})', ha='center', va='bottom', fontsize=10)

        ax.set_ylabel('Detection Rate (%)')
        ax.set_title('Detection Rate Comparison\n(Value-based vs Pattern-based)', fontweight='bold')
        ax.set_ylim(0, 115)

        # ===== Plot 4: Mean Score Components =====
        ax = axes[1, 1]
        categories_plot = []
        recon_means = []
        disc_means = []

        for cat_name, cat_data in [('Normal', normal_data), ('Value-based', value_data), ('Pattern-based', pattern_data)]:
            if len(cat_data) > 0:
                categories_plot.append(cat_name)
                recon_means.append(cat_data['reconstruction_loss'].mean())
                disc_means.append(cat_data['discrepancy_loss'].mean())

        if len(categories_plot) > 0:
            x = np.arange(len(categories_plot))
            width = 0.35
            ax.bar(x - width/2, recon_means, width, label='Reconstruction', color=VIS_COLORS['reconstruction'], alpha=0.8)
            ax.bar(x + width/2, disc_means, width, label='Discrepancy', color=VIS_COLORS['discrepancy'], alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(categories_plot)
            ax.set_ylabel('Mean Loss')
            ax.set_title('Loss Components: Value vs Pattern Anomalies', fontweight='bold')
            ax.legend()

        # Add annotation explaining the significance
        fig.text(0.5, 0.02,
                'Value-based: Anomalies with unusual VALUES (types 1-7: spike, memory_leak, etc.)\n'
                'Pattern-based: Anomalies with unusual PATTERNS but normal values (types 8-10: correlation_inversion, temporal_flatline, frequency_shift)',
                ha='center', fontsize=10, style='italic',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        plt.suptitle('Value-based vs Pattern-based Anomaly Detection', fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0.08, 1, 0.96])
        plt.savefig(os.path.join(self.output_dir, 'value_vs_pattern_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - value_vs_pattern_comparison.png")

    def plot_loss_scatter_by_anomaly_type(self, experiment_dir: str = None):
        """Scatter plot of reconstruction vs discrepancy loss colored by anomaly type

        Args:
            experiment_dir: Path to experiment directory containing best_model_detailed.csv
        """
        detailed_csv = None
        if experiment_dir:
            csv_path = os.path.join(experiment_dir, 'best_model_detailed.csv')
            if os.path.exists(csv_path):
                detailed_csv = pd.read_csv(csv_path)

        if detailed_csv is None:
            print("  ! Skipping loss_scatter_by_anomaly_type (no detailed CSV found)")
            return

        # Setup colors dynamically from base module
        colors = get_anomaly_colors()

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: All types scatter
        ax = axes[0]
        for atype in ANOMALY_TYPE_NAMES:
            type_data = detailed_csv[detailed_csv['anomaly_type_name'] == atype]
            if len(type_data) > 0:
                ax.scatter(type_data['reconstruction_loss'], type_data['discrepancy_loss'],
                          alpha=0.6, label=atype.replace('_', ' ').title(),
                          color=colors.get(atype, VIS_COLORS['reference']), s=30)

        ax.set_xlabel('Reconstruction Loss')
        ax.set_ylabel('Discrepancy Loss')
        ax.set_title('Loss Components by Anomaly Type', fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)

        # Right: Normal vs All Anomalies
        ax = axes[1]
        normal_data = detailed_csv[detailed_csv['anomaly_type_name'] == 'normal']
        anomaly_data = detailed_csv[detailed_csv['anomaly_type_name'] != 'normal']

        ax.scatter(normal_data['reconstruction_loss'], normal_data['discrepancy_loss'],
                  alpha=0.5, label='Normal', color=VIS_COLORS['normal'], s=30)
        ax.scatter(anomaly_data['reconstruction_loss'], anomaly_data['discrepancy_loss'],
                  alpha=0.5, label='Anomaly', color=VIS_COLORS['anomaly'], s=30)

        ax.set_xlabel('Reconstruction Loss')
        ax.set_ylabel('Discrepancy Loss')
        ax.set_title('Normal vs Anomaly Loss Distribution', fontweight='bold')
        ax.legend(loc='upper right')

        plt.suptitle('Loss Scatter Analysis', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'loss_scatter_by_anomaly_type.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - loss_scatter_by_anomaly_type.png")

    def plot_sample_type_analysis(self, experiment_dir: str = None):
        """Analyze performance across sample types (pure_normal, disturbing_normal, anomaly)

        Args:
            experiment_dir: Path to experiment directory containing best_model_detailed.csv
        """
        detailed_csv = None
        if experiment_dir:
            csv_path = os.path.join(experiment_dir, 'best_model_detailed.csv')
            if os.path.exists(csv_path):
                detailed_csv = pd.read_csv(csv_path)

        if detailed_csv is None:
            print("  ! Skipping sample_type_analysis (no detailed CSV found)")
            return

        # Use dynamic sample type names and colors from base module
        sample_type_names = SAMPLE_TYPE_NAMES
        colors = SAMPLE_TYPE_COLORS

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Score distribution by sample type
        ax = axes[0, 0]
        for stype in [0, 1, 2]:
            type_data = detailed_csv[detailed_csv['sample_type'] == stype]
            if len(type_data) > 0:
                ax.hist(type_data['total_loss'], bins=30, alpha=0.6,
                       label=f"{sample_type_names[stype]} (n={len(type_data)})",
                       color=colors[stype], density=True)
        ax.set_xlabel('Total Loss (Anomaly Score)')
        ax.set_ylabel('Density')
        ax.set_title('Score Distribution by Sample Type', fontweight='bold')
        ax.legend()

        # 2. Box plot by sample type
        ax = axes[0, 1]
        box_data = []
        box_labels = []
        for stype in [0, 1, 2]:
            type_data = detailed_csv[detailed_csv['sample_type'] == stype]
            if len(type_data) > 0:
                box_data.append(type_data['total_loss'].values)
                box_labels.append(sample_type_names[stype])

        if box_data:
            bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor(colors[i])
                patch.set_alpha(0.7)
        ax.set_ylabel('Total Loss')
        ax.set_title('Score Box Plot by Sample Type', fontweight='bold')

        # 3. Reconstruction vs Discrepancy by sample type
        ax = axes[1, 0]
        for stype in [0, 1, 2]:
            type_data = detailed_csv[detailed_csv['sample_type'] == stype]
            if len(type_data) > 0:
                ax.scatter(type_data['reconstruction_loss'], type_data['discrepancy_loss'],
                          alpha=0.5, label=sample_type_names[stype], color=colors[stype], s=30)
        ax.set_xlabel('Reconstruction Loss')
        ax.set_ylabel('Discrepancy Loss')
        ax.set_title('Loss Components by Sample Type', fontweight='bold')
        ax.legend()

        # 4. Mean losses comparison
        ax = axes[1, 1]
        x = np.arange(3)
        width = 0.35

        recon_means = []
        disc_means = []
        labels = []

        for stype in [0, 1, 2]:
            type_data = detailed_csv[detailed_csv['sample_type'] == stype]
            if len(type_data) > 0:
                recon_means.append(type_data['reconstruction_loss'].mean())
                disc_means.append(type_data['discrepancy_loss'].mean())
                labels.append(sample_type_names[stype])

        if recon_means:
            x = np.arange(len(labels))
            ax.bar(x - width/2, recon_means, width, label='Reconstruction', color=VIS_COLORS['normal'], alpha=0.8)
            ax.bar(x + width/2, disc_means, width, label='Discrepancy', color=VIS_COLORS['anomaly'], alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
        ax.set_ylabel('Mean Loss')
        ax.set_title('Mean Losses by Sample Type', fontweight='bold')
        ax.legend()

        plt.suptitle('Sample Type Analysis\n(Pure Normal = completely normal, '
                    'Disturbing Normal = anomaly outside last patch, Anomaly = anomaly in last patch)',
                    fontsize=12, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'sample_type_analysis.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - sample_type_analysis.png")

    def plot_pure_vs_disturbing_normal(self):
        """Compare pure normal vs disturbing normal in detail"""
        pure_normal_mask = self.detailed_data['sample_types'] == 0
        disturbing_mask = self.detailed_data['sample_types'] == 1
        anomaly_mask = self.detailed_data['sample_types'] == 2

        mask_inverse = 1 - self.detailed_data['masks']

        # Compute scores for each sample type
        def compute_scores(mask):
            teacher_err = (self.detailed_data['teacher_errors'][mask] * mask_inverse[mask]).sum(axis=1) / (mask_inverse[mask].sum(axis=1) + 1e-8)
            student_err = (self.detailed_data['student_errors'][mask] * mask_inverse[mask]).sum(axis=1) / (mask_inverse[mask].sum(axis=1) + 1e-8)
            disc = (self.detailed_data['discrepancies'][mask] * mask_inverse[mask]).sum(axis=1) / (mask_inverse[mask].sum(axis=1) + 1e-8)
            total = teacher_err + self.config.lambda_disc * disc
            return teacher_err, student_err, disc, total

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
                    '(Disturbing Normal = anomaly exists but NOT in last patch)',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'pure_vs_disturbing_normal.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - pure_vs_disturbing_normal.png")

    def plot_discrepancy_trend(self):
        """Plot discrepancy trends across time steps

        Note: Discrepancy is computed on ALL timesteps, but only the last mask_last_n
        positions are masked during evaluation. At unmasked positions, teacher-student
        discrepancy should be similar across sample types. The key difference appears
        in the MASKED region (last patch).
        """
        # Select representative samples
        pure_normal_mask = self.detailed_data['sample_types'] == 0
        disturbing_mask = self.detailed_data['sample_types'] == 1
        anomaly_mask = self.detailed_data['sample_types'] == 2

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        seq_length = self.detailed_data['discrepancies'].shape[1]
        mask_last_n = getattr(self.config, 'mask_last_n', 10)
        t = np.arange(seq_length)

        # Get data
        pure_disc = self.detailed_data['discrepancies'][pure_normal_mask]
        dist_disc = self.detailed_data['discrepancies'][disturbing_mask]
        anom_disc = self.detailed_data['discrepancies'][anomaly_mask]

        # 1. Mean discrepancy with std bands - FOCUSED ON LAST PATCH
        ax = axes[0, 0]

        pure_mean = pure_disc.mean(axis=0)
        dist_mean = dist_disc.mean(axis=0)
        anom_mean = anom_disc.mean(axis=0)

        pure_std = pure_disc.std(axis=0)
        dist_std = dist_disc.std(axis=0)
        anom_std = anom_disc.std(axis=0)

        # Plot with std bands
        ax.plot(t, pure_mean, label=f'Pure Normal (n={pure_normal_mask.sum()})', color=VIS_COLORS['normal'], lw=2)
        ax.fill_between(t, pure_mean - pure_std, pure_mean + pure_std, color=VIS_COLORS['normal'], alpha=0.2)

        ax.plot(t, dist_mean, label=f'Disturbing Normal (n={disturbing_mask.sum()})', color=VIS_COLORS['disturbing'], lw=2)
        ax.fill_between(t, dist_mean - dist_std, dist_mean + dist_std, color=VIS_COLORS['disturbing'], alpha=0.2)

        ax.plot(t, anom_mean, label=f'Anomaly (n={anomaly_mask.sum()})', color=VIS_COLORS['anomaly'], lw=2)
        ax.fill_between(t, anom_mean - anom_std, anom_mean + anom_std, color=VIS_COLORS['anomaly'], alpha=0.2)

        # Highlight masked region (last patch)
        ax.axvspan(seq_length - mask_last_n, seq_length, alpha=0.3, color=VIS_COLORS['masked_region'], label='Masked Region')

        ax.set_xlabel('Time Step')
        ax.set_ylabel('Mean Discrepancy (±1 std)')
        ax.set_title('Discrepancy Trend by Sample Type\n(Shaded: ±1 std)', fontweight='bold')
        ax.legend(fontsize=8)

        # 2. ZOOMED IN: Last Patch Region Only
        ax = axes[0, 1]
        t_last = np.arange(seq_length - mask_last_n, seq_length)

        pure_last = pure_disc[:, -mask_last_n:]
        dist_last = dist_disc[:, -mask_last_n:]
        anom_last = anom_disc[:, -mask_last_n:]

        # Use consistent colors from VIS_COLORS (Normal=blue, Anomaly=red, Disturbing=orange)
        ax.plot(t_last, pure_last.mean(axis=0), label='Pure Normal',
               color=VIS_COLORS['normal'], lw=2, marker='o', ms=4)
        ax.fill_between(t_last, pure_last.mean(axis=0) - pure_last.std(axis=0),
                       pure_last.mean(axis=0) + pure_last.std(axis=0), color=VIS_COLORS['normal'], alpha=0.2)

        ax.plot(t_last, dist_last.mean(axis=0), label='Disturbing Normal',
               color=VIS_COLORS['disturbing'], lw=2, marker='s', ms=4)
        ax.fill_between(t_last, dist_last.mean(axis=0) - dist_last.std(axis=0),
                       dist_last.mean(axis=0) + dist_last.std(axis=0), color=VIS_COLORS['disturbing'], alpha=0.2)

        ax.plot(t_last, anom_last.mean(axis=0), label='Anomaly',
               color=VIS_COLORS['anomaly'], lw=2, marker='^', ms=4)
        ax.fill_between(t_last, anom_last.mean(axis=0) - anom_last.std(axis=0),
                       anom_last.mean(axis=0) + anom_last.std(axis=0), color=VIS_COLORS['anomaly'], alpha=0.2)

        ax.set_xlabel('Time Step')
        ax.set_ylabel('Mean Discrepancy')
        ax.set_title(f'ZOOMED: Last {mask_last_n} Steps (Masked Region)\nThis is where differences matter!', fontweight='bold')
        ax.legend()

        # 3. Discrepancy heatmap comparison (side by side: normal vs anomaly)
        ax = axes[1, 0]

        # Take 25 samples each for comparison
        n_samples = min(25, pure_normal_mask.sum(), anomaly_mask.sum())
        combined = np.vstack([pure_disc[:n_samples], anom_disc[:n_samples]])
        combined_labels = ['Normal'] * n_samples + ['Anomaly'] * n_samples

        sns.heatmap(combined, ax=ax, cmap='Reds', cbar_kws={'label': 'Discrepancy'})
        ax.axvline(x=seq_length - mask_last_n, color='white', linestyle='--', lw=2)
        ax.axhline(y=n_samples, color='white', linestyle='-', lw=3)
        ax.text(seq_length - mask_last_n - 5, n_samples/2, 'Normal', ha='right', va='center', fontsize=10, color='white', fontweight='bold')
        ax.text(seq_length - mask_last_n - 5, n_samples + n_samples/2, 'Anomaly', ha='right', va='center', fontsize=10, color='white', fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Sample Index')
        ax.set_title('Discrepancy Heatmap: Normal vs Anomaly', fontweight='bold')

        # 4. Last patch discrepancy distribution (bar chart for clarity)
        ax = axes[1, 1]
        last_patch_pure = pure_disc[:, -mask_last_n:].mean(axis=1)
        last_patch_dist = dist_disc[:, -mask_last_n:].mean(axis=1)
        last_patch_anom = anom_disc[:, -mask_last_n:].mean(axis=1)

        # Box plot for better comparison
        data_to_plot = [last_patch_pure, last_patch_dist, last_patch_anom]
        labels = ['Pure Normal', 'Disturbing Normal', 'Anomaly']
        sample_colors = [VIS_COLORS['normal'], VIS_COLORS['disturbing'], VIS_COLORS['anomaly']]

        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], sample_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        # Add mean markers
        means = [d.mean() for d in data_to_plot]
        ax.scatter([1, 2, 3], means, color=VIS_COLORS['baseline'], marker='D', s=50, zorder=5, label='Mean')

        ax.set_ylabel('Mean Discrepancy (Last Patch)')
        ax.set_title('Last Patch Discrepancy Distribution\n(Masked Region Only)', fontweight='bold')

        # Add statistics text
        stats_text = f"Mean ± Std:\n"
        stats_text += f"Pure Normal: {means[0]:.4f} ± {last_patch_pure.std():.4f}\n"
        stats_text += f"Disturbing: {means[1]:.4f} ± {last_patch_dist.std():.4f}\n"
        stats_text += f"Anomaly: {means[2]:.4f} ± {last_patch_anom.std():.4f}"
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.suptitle('Discrepancy Trend Analysis (Teacher-Student Difference)', fontsize=14, fontweight='bold', y=1.02)
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
        # Get predictions using pred_data (has scores)
        threshold = self._get_optimal_threshold()
        scores = self._get_scores()
        predictions = (scores >= threshold).astype(int)
        labels = self.pred_data['labels']
        sample_types = self.detailed_data['sample_types']

        # Find examples for each category
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
            patch_idx = indices[np.argsort(cat_scores)[len(cat_scores)//2]]
            window_idx = self._patch_idx_to_window_idx(patch_idx)

            # Column 1: Time series with reconstruction
            ax = axes[row, 0]
            original = self.detailed_data['originals'][window_idx, :, 0]
            teacher_recon = self.detailed_data['teacher_recons'][window_idx, :, 0]
            student_recon = self.detailed_data['student_recons'][window_idx, :, 0]
            point_labels = self.detailed_data['point_labels'][window_idx]

            ax.plot(original, 'b-', lw=1.2, alpha=0.8, label='Original')
            ax.plot(teacher_recon, 'g--', lw=1.5, alpha=0.7, label='Teacher')
            ax.plot(student_recon, 'r:', lw=1.5, alpha=0.7, label='Student')

            # Highlight anomaly and masked regions
            anomaly_region = np.where(point_labels == 1)[0]
            if len(anomaly_region) > 0:
                ax.axvspan(anomaly_region[0], anomaly_region[-1], alpha=0.2, color=VIS_COLORS['anomaly_region'], label='Anomaly')
            if self.inference_mode != 'all_patches':
                mask_start = len(original) - self.config.mask_last_n
                ax.axvspan(mask_start, len(original), alpha=0.2, color=VIS_COLORS['masked_region'], label='Masked')

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
            if self.inference_mode != 'all_patches':
                ax.axvspan(mask_start, len(original), alpha=0.2, color=VIS_COLORS['masked_region'])
            ax.axhline(y=np.mean(discrepancy[-self.config.mask_last_n:]), color=VIS_COLORS['threshold'], linestyle='--',
                      label=f'Masked Mean: {np.mean(discrepancy[-self.config.mask_last_n:]):.4f}')

            ax.set_title(f'{cat_name}: Discrepancy (|T-S|)', fontweight='bold', color=color)
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Discrepancy')
            ax.legend(fontsize=8)

            # Column 3: Statistics
            ax = axes[row, 2]
            ax.axis('off')

            sample_score = scores[patch_idx]
            teacher_err = np.mean((original[-self.config.mask_last_n:] - teacher_recon[-self.config.mask_last_n:])**2)
            student_err = np.mean((original[-self.config.mask_last_n:] - student_recon[-self.config.mask_last_n:])**2)
            masked_disc = np.mean(discrepancy[-self.config.mask_last_n:])

            stype = sample_types[window_idx]
            stype_name = ['Pure Normal', 'Disturbing Normal', 'Anomaly'][stype]

            stats_text = f"""
{cat_name} Case Study
═══════════════════════════════════

Sample Index: {window_idx}
Sample Type:  {stype_name}
True Label:   {'Anomaly' if labels[patch_idx] == 1 else 'Normal'}
Prediction:   {'Anomaly' if predictions[patch_idx] == 1 else 'Normal'}

Score Analysis:
  • Total Score:    {sample_score:.6f}
  • Threshold:      {threshold:.6f}
  • Margin:         {sample_score - threshold:+.6f}

Masked Region Metrics:
  • Teacher MSE:    {teacher_err:.6f}
  • Student MSE:    {student_err:.6f}
  • Discrepancy:    {masked_disc:.6f}

Anomaly Location:
  • Points: {len(anomaly_region)} / {len(original)}
  • In Mask: {sum(point_labels[-self.config.mask_last_n:])}
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

    def plot_anomaly_type_case_studies(self, experiment_dir: str = None):
        """Generate detailed case studies for each anomaly type.

        Shows representative TP and FN examples for each anomaly type.
        Dynamically handles any number of anomaly types from ANOMALY_TYPE_NAMES.
        """
        if experiment_dir is None:
            print("  ! Skipping anomaly_type_case_studies (need experiment_dir)")
            return

        # Load detailed results with anomaly types
        detailed_path = os.path.join(experiment_dir, 'best_model_detailed_results.csv')
        if not os.path.exists(detailed_path):
            print(f"  ! Skipping anomaly_type_case_studies (no detailed results)")
            return

        df = pd.read_csv(detailed_path)
        threshold = self._get_optimal_threshold()

        # Use actual anomaly type names from dataset (skip 'normal' at index 0)
        anomaly_types = ANOMALY_TYPE_NAMES[1:]  # e.g., ['spike', 'memory_leak', ...]
        anomaly_type_map = {i: name for i, name in enumerate(ANOMALY_TYPE_NAMES) if i > 0}

        n_types = len(anomaly_types)
        fig, axes = plt.subplots(n_types, 4, figsize=(20, 5 * n_types))
        if n_types == 1:
            axes = axes.reshape(1, -1)  # Ensure 2D array

        for row, atype in enumerate(anomaly_types):
            # Find samples of this anomaly type
            type_mask = df['anomaly_type_name'] == atype if 'anomaly_type_name' in df.columns else np.zeros(len(df), dtype=bool)

            if type_mask.sum() == 0:
                # Try numeric type
                type_num = anomaly_types.index(atype) + 1
                if 'anomaly_type' in df.columns:
                    type_mask = df['anomaly_type'] == type_num

            if type_mask.sum() == 0:
                for col in range(4):
                    axes[row, col].text(0.5, 0.5, f'No {atype} samples',
                                       ha='center', va='center', transform=axes[row, col].transAxes)
                continue

            type_indices = np.where(type_mask)[0]
            type_scores = df.loc[type_mask, 'total_score'].values if 'total_score' in df.columns else self._get_scores()[type_indices]
            type_preds = (type_scores >= threshold).astype(int)

            tp_mask = type_preds == 1
            fn_mask = type_preds == 0

            # Column 1: TP example time series
            ax = axes[row, 0]
            if tp_mask.sum() > 0:
                tp_local_idx = np.where(tp_mask)[0][0]
                tp_idx = type_indices[tp_local_idx]

                if tp_idx < len(self.detailed_data['originals']):
                    original = self.detailed_data['originals'][tp_idx, :, 0]
                    teacher_recon = self.detailed_data['teacher_recons'][tp_idx, :, 0]
                    point_labels = self.detailed_data['point_labels'][tp_idx]

                    ax.plot(original, 'b-', lw=1.2, alpha=0.8)
                    ax.plot(teacher_recon, 'g--', lw=1.5, alpha=0.7)

                    anomaly_region = np.where(point_labels == 1)[0]
                    if len(anomaly_region) > 0:
                        ax.axvspan(anomaly_region[0], anomaly_region[-1], alpha=0.2, color=VIS_COLORS['anomaly_region'])
                    ax.axvspan(len(original) - self.config.mask_last_n, len(original), alpha=0.2, color=VIS_COLORS['masked_region'])

                ax.set_title(f'{atype.upper()}: TP Example', fontweight='bold', color=VIS_COLORS['teacher'])
            else:
                ax.text(0.5, 0.5, 'No TP', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{atype.upper()}: TP Example', fontweight='bold')
            ax.set_xlabel('Time Step')
            if row == 0:
                ax.set_ylabel('Value')

            # Column 2: TP discrepancy
            ax = axes[row, 1]
            if tp_mask.sum() > 0 and tp_idx < len(self.detailed_data['discrepancies']):
                discrepancy = self.detailed_data['discrepancies'][tp_idx]
                ax.fill_between(range(len(discrepancy)), discrepancy, alpha=0.6, color=VIS_COLORS['teacher'])
                if len(anomaly_region) > 0:
                    ax.axvspan(anomaly_region[0], anomaly_region[-1], alpha=0.2, color=VIS_COLORS['anomaly_region'])
                ax.axvspan(len(discrepancy) - self.config.mask_last_n, len(discrepancy), alpha=0.2, color=VIS_COLORS['masked_region'])
            ax.set_title(f'{atype.upper()}: TP Discrepancy', fontweight='bold', color=VIS_COLORS['teacher'])
            ax.set_xlabel('Time Step')

            # Column 3: FN example time series
            ax = axes[row, 2]
            if fn_mask.sum() > 0:
                fn_local_idx = np.where(fn_mask)[0][0]
                fn_idx = type_indices[fn_local_idx]

                if fn_idx < len(self.detailed_data['originals']):
                    original = self.detailed_data['originals'][fn_idx, :, 0]
                    teacher_recon = self.detailed_data['teacher_recons'][fn_idx, :, 0]
                    point_labels = self.detailed_data['point_labels'][fn_idx]

                    ax.plot(original, 'b-', lw=1.2, alpha=0.8)
                    ax.plot(teacher_recon, 'g--', lw=1.5, alpha=0.7)

                    anomaly_region_fn = np.where(point_labels == 1)[0]
                    if len(anomaly_region_fn) > 0:
                        ax.axvspan(anomaly_region_fn[0], anomaly_region_fn[-1], alpha=0.2, color=VIS_COLORS['anomaly_region'])
                    ax.axvspan(len(original) - self.config.mask_last_n, len(original), alpha=0.2, color=VIS_COLORS['masked_region'])

                ax.set_title(f'{atype.upper()}: FN Example', fontweight='bold', color=VIS_COLORS['anomaly'])
            else:
                ax.text(0.5, 0.5, 'No FN (All Detected!)', ha='center', va='center', transform=ax.transAxes, color=VIS_COLORS['teacher'])
                ax.set_title(f'{atype.upper()}: FN Example', fontweight='bold')
            ax.set_xlabel('Time Step')

            # Column 4: FN discrepancy
            ax = axes[row, 3]
            if fn_mask.sum() > 0 and fn_idx < len(self.detailed_data['discrepancies']):
                discrepancy = self.detailed_data['discrepancies'][fn_idx]
                ax.fill_between(range(len(discrepancy)), discrepancy, alpha=0.6, color=VIS_COLORS['anomaly'])
                if len(anomaly_region_fn) > 0:
                    ax.axvspan(anomaly_region_fn[0], anomaly_region_fn[-1], alpha=0.2, color=VIS_COLORS['anomaly_region'])
                ax.axvspan(len(discrepancy) - self.config.mask_last_n, len(discrepancy), alpha=0.2, color=VIS_COLORS['masked_region'])
            ax.set_title(f'{atype.upper()}: FN Discrepancy', fontweight='bold', color=VIS_COLORS['anomaly'])
            ax.set_xlabel('Time Step')

        plt.suptitle('Anomaly Type Case Studies: TP vs FN Examples\n'
                    '(Yellow=Masked Region, Red=Anomaly Region, Green=Reconstruction)',
                    fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'anomaly_type_case_studies.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - anomaly_type_case_studies.png")

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

    def _plot_sample_detail(self, axes_row, patch_idx, title_prefix, color, threshold):
        """Helper to plot detailed sample analysis in a row of 3 axes.

        Args:
            patch_idx: Index from pred_data (patch-level in all_patches mode)
        """
        # Convert patch index to window index for detailed_data access
        window_idx = self._patch_idx_to_window_idx(patch_idx)

        original = self.detailed_data['originals'][window_idx, :, 0]
        teacher_recon = self.detailed_data['teacher_recons'][window_idx, :, 0]
        student_recon = self.detailed_data['student_recons'][window_idx, :, 0]
        discrepancy = self.detailed_data['discrepancies'][window_idx]
        point_labels = self.detailed_data['point_labels'][window_idx]
        score = self._get_scores()[patch_idx]
        label = self.pred_data['labels'][patch_idx]
        sample_type = self.detailed_data['sample_types'][window_idx]

        anomaly_region = np.where(point_labels == 1)[0]
        mask_start = len(original) - self.config.mask_last_n

        # Column 1: Time series
        ax = axes_row[0]
        ax.plot(original, 'b-', lw=1.2, alpha=0.8, label='Original')
        ax.plot(teacher_recon, 'g--', lw=1.5, alpha=0.7, label='Teacher')
        ax.plot(student_recon, 'r:', lw=1.5, alpha=0.7, label='Student')
        if len(anomaly_region) > 0:
            ax.axvspan(anomaly_region[0], anomaly_region[-1], alpha=0.2, color=VIS_COLORS['anomaly_region'])
        if self.inference_mode != 'all_patches':
            ax.axvspan(mask_start, len(original), alpha=0.2, color=VIS_COLORS['masked_region'])
        ax.set_title(f'{title_prefix}: Time Series', fontweight='bold', color=color)
        ax.legend(fontsize=7)

        # Column 2: Discrepancy
        ax = axes_row[1]
        ax.fill_between(range(len(discrepancy)), discrepancy, alpha=0.6, color=VIS_COLORS['student'])
        ax.plot(discrepancy, color=VIS_COLORS['student_dark'], lw=1)
        if len(anomaly_region) > 0:
            ax.axvspan(anomaly_region[0], anomaly_region[-1], alpha=0.2, color=VIS_COLORS['anomaly_region'])
        if self.inference_mode != 'all_patches':
            ax.axvspan(mask_start, len(original), alpha=0.2, color=VIS_COLORS['masked_region'])
        ax.set_title(f'{title_prefix}: Discrepancy', fontweight='bold', color=color)

        # Column 3: Stats
        ax = axes_row[2]
        ax.axis('off')

        stype_name = ['Pure Normal', 'Disturbing Normal', 'Anomaly'][sample_type]
        margin = score - threshold

        stats_text = f"""
{title_prefix}
═══════════════════════════════

Index: {window_idx}
Type:  {stype_name}
Label: {'Anomaly' if label == 1 else 'Normal'}

Score:     {score:.6f}
Threshold: {threshold:.6f}
Margin:    {margin:+.6f}

Anomaly in masked region:
  {sum(point_labels[-self.config.mask_last_n:])} / {self.config.mask_last_n} points
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
        ax.set_ylabel('MSE Loss')
        ax.set_title('Teacher Reconstruction Loss (○)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
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
        ax.set_ylabel('MSE Loss')
        ax.set_title('Student Reconstruction Loss (△)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
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
        ax.set_ylabel('Discrepancy Loss')
        ax.set_title('Discrepancy Loss (□)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
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
        ax.set_ylabel('MSE Loss')
        ax.set_title('Normal Data: Teacher vs Student', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
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
        ax.set_ylabel('MSE Loss')
        ax.set_title('Anomaly Data: Teacher vs Student', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
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
        ax.set_ylabel('Loss')
        ax.set_title('All Losses Combined', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
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

        for mask, color, label in zip(masks, cat_colors, categories):
            if mask.sum() > 10:  # Need enough samples for KDE
                data = recon_contrib[mask]
                # Remove outliers for better visualization
                q1, q99 = np.percentile(data, [1, 99])
                data_clean = data[(data >= q1) & (data <= q99)]
                if len(data_clean) > 10:
                    kde = gaussian_kde(data_clean)
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
                    kde = gaussian_kde(data_clean)
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

        if has_contrib_history:
            epochs = history.get('epoch', list(range(len(history['epoch_recon_ratio_normal']))))

            # Get contribution ratios by sample type
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
        self.plot_anomaly_type_case_studies(experiment_dir)
        self.plot_hardest_samples()

        # Anomaly type analysis (requires detailed results from experiment)
        self.plot_loss_by_anomaly_type(experiment_dir)
        self.plot_performance_by_anomaly_type(experiment_dir)
        self.plot_value_vs_pattern_comparison(experiment_dir)  # Value vs pattern anomaly comparison
        self.plot_loss_scatter_by_anomaly_type(experiment_dir)
        self.plot_sample_type_analysis(experiment_dir)

        # Anomaly type score trends over epochs (requires history with epoch_anomaly_type_scores)
        self.plot_anomaly_type_score_trends(experiment_dir, history)
        self.plot_score_contribution_epoch_trends(experiment_dir, history)

    def plot_anomaly_type_score_trends(self, experiment_dir: str = None, history: Dict = None):
        """Plot epoch-wise score trends for each anomaly type

        Creates a figure showing how reconstruction and discrepancy scores evolve
        during training for each anomaly type (including normal).

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
            print("  - Skipping anomaly type score trends (no epoch_anomaly_type_scores in history)")
            return

        epoch_scores = history['epoch_anomaly_type_scores']
        if not epoch_scores:
            return

        epochs = history.get('epoch', list(range(1, len(epoch_scores) + 1)))

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

        # Create figure with 2 rows: recon_score, disc_score
        # Columns: one per anomaly type
        n_cols = min(4, n_types)  # Max 4 columns
        n_rows = ((n_types - 1) // n_cols + 1) * 2  # 2 metrics per type group

        fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows))

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

        # Collect data for each type
        type_data = {t: {'recon': [], 'disc': [], 'count': []} for t in anomaly_type_order}

        for epoch_data in epoch_scores:
            for atype in anomaly_type_order:
                if atype in epoch_data:
                    type_data[atype]['recon'].append(epoch_data[atype].get('recon_score', 0))
                    type_data[atype]['disc'].append(epoch_data[atype].get('disc_score', 0))
                    type_data[atype]['count'].append(epoch_data[atype].get('count', 0))
                else:
                    type_data[atype]['recon'].append(0)
                    type_data[atype]['disc'].append(0)
                    type_data[atype]['count'].append(0)

        # Compute unified y-axis limits
        all_recon = []
        all_disc = []
        for atype in anomaly_type_order:
            all_recon.extend(type_data[atype]['recon'])
            all_disc.extend(type_data[atype]['disc'])

        y_min_recon = min(0, min(all_recon) if all_recon else 0) * 1.1
        y_max_recon = max(all_recon) * 1.1 if all_recon else 1
        y_min_disc = min(0, min(all_disc) if all_disc else 0) * 1.1
        y_max_disc = max(all_disc) * 1.1 if all_disc else 1

        # Plot recon scores (odd rows) and disc scores (even rows)
        for idx, atype in enumerate(anomaly_type_order):
            col = idx % n_cols
            row_group = idx // n_cols

            color = type_colors.get(atype, '#95A5A6')
            count = type_data[atype]['count'][0] if type_data[atype]['count'] else 0

            # Recon score plot
            ax_recon = fig.add_subplot(n_rows, n_cols, row_group * 2 * n_cols + col + 1)
            ax_recon.plot(epochs, type_data[atype]['recon'], color=color, linewidth=2, marker='o', markersize=3)
            ax_recon.set_xlabel('Epoch')
            ax_recon.set_ylabel('Recon Score')
            ax_recon.set_ylim(y_min_recon, y_max_recon)
            ax_recon.set_title(f'{atype}\n(n={count}) - Recon', fontsize=10, fontweight='bold')
            ax_recon.grid(alpha=0.3)
            ax_recon.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

            # Disc score plot
            ax_disc = fig.add_subplot(n_rows, n_cols, (row_group * 2 + 1) * n_cols + col + 1)
            ax_disc.plot(epochs, type_data[atype]['disc'], color=color, linewidth=2, marker='s', markersize=3)
            ax_disc.set_xlabel('Epoch')
            ax_disc.set_ylabel('Disc Score')
            ax_disc.set_ylim(y_min_disc, y_max_disc)
            ax_disc.set_title(f'{atype}\n(n={count}) - Disc', fontsize=10, fontweight='bold')
            ax_disc.grid(alpha=0.3)
            ax_disc.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        fig.suptitle('Anomaly Type Score Trends Over Epochs', fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'best_model_anomaly_type_trends.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()
        print("  - best_model_anomaly_type_trends.png")

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

        # Collect data for each type
        type_data = {t: {'recon': [], 'disc': [], 'count': []} for t in anomaly_type_order}

        for epoch_data in epoch_scores:
            for atype in anomaly_type_order:
                if atype in epoch_data:
                    type_data[atype]['recon'].append(epoch_data[atype].get('recon_score', 0))
                    type_data[atype]['disc'].append(epoch_data[atype].get('disc_score', 0))
                    type_data[atype]['count'].append(epoch_data[atype].get('count', 0))
                else:
                    type_data[atype]['recon'].append(0)
                    type_data[atype]['disc'].append(0)
                    type_data[atype]['count'].append(0)

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

        fig.suptitle('Score Contribution Trends by Anomaly Type (Epoch ≥ 5)',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'best_model_score_contribution_trends.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()
        print("  - best_model_score_contribution_trends.png")


# =============================================================================
# TrainingProgressVisualizer - Training Progress Analysis
# =============================================================================

