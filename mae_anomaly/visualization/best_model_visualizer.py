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
from tqdm import tqdm

from mae_anomaly import Config, ANOMALY_TYPE_NAMES
from .base import (
    get_anomaly_colors, SAMPLE_TYPE_NAMES, SAMPLE_TYPE_COLORS,
    VIS_COLORS, VIS_MARKERS, VIS_LINESTYLES,
    collect_predictions, collect_detailed_data,
)

class BestModelVisualizer:
    """Visualize best model analysis"""

    def __init__(self, model, config: Config, test_loader, output_dir: str):
        self.model = model
        self.config = config
        self.test_loader = test_loader
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Collect data
        print("  Collecting model predictions...")
        self.pred_data = collect_predictions(model, test_loader, config)
        print("  Collecting detailed data...")
        self.detailed_data = collect_detailed_data(model, test_loader, config)

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

    def plot_score_distribution(self):
        """Plot anomaly score distribution"""
        normal_mask = self.pred_data['labels'] == 0
        anomaly_mask = self.pred_data['labels'] == 1

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        ax = axes[0]
        ax.hist(self.pred_data['scores'][normal_mask], bins=50, alpha=0.6,
               label='Normal', color=VIS_COLORS['normal'], density=True)
        ax.hist(self.pred_data['scores'][anomaly_mask], bins=50, alpha=0.6,
               label='Anomaly', color=VIS_COLORS['anomaly'], density=True)
        ax.axvline(self.pred_data['scores'][normal_mask].mean(), color=VIS_COLORS['normal'], linestyle='--', lw=2)
        ax.axvline(self.pred_data['scores'][anomaly_mask].mean(), color=VIS_COLORS['anomaly'], linestyle='--', lw=2)
        ax.set_xlabel('Anomaly Score')
        ax.set_ylabel('Density')
        ax.set_title('Anomaly Score Distribution', fontsize=12, fontweight='bold')
        ax.legend()

        # Box plot
        ax = axes[1]
        box_data = [self.pred_data['scores'][normal_mask], self.pred_data['scores'][anomaly_mask]]
        bp = ax.boxplot(box_data, labels=['Normal', 'Anomaly'], patch_artist=True)
        bp['boxes'][0].set_facecolor(VIS_COLORS['normal'])
        bp['boxes'][1].set_facecolor(VIS_COLORS['anomaly'])
        ax.set_ylabel('Anomaly Score')
        ax.set_title('Anomaly Score Box Plot', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'best_model_score_distribution.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - best_model_score_distribution.png")

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

    def plot_score_components(self):
        """Plot reconstruction error vs discrepancy"""
        normal_mask = self.pred_data['labels'] == 0
        anomaly_mask = self.pred_data['labels'] == 1

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Scatter plot
        ax = axes[0]
        ax.scatter(self.pred_data['recon_errors'][normal_mask],
                  self.pred_data['discrepancies'][normal_mask],
                  alpha=0.5, label='Normal', color=VIS_COLORS['normal'], s=30)
        ax.scatter(self.pred_data['recon_errors'][anomaly_mask],
                  self.pred_data['discrepancies'][anomaly_mask],
                  alpha=0.5, label='Anomaly', color=VIS_COLORS['anomaly'], s=30)
        ax.set_xlabel('Reconstruction Error')
        ax.set_ylabel('Discrepancy')
        ax.set_title('Reconstruction Error vs Discrepancy', fontsize=12, fontweight='bold')
        ax.legend()

        # Component contributions
        ax = axes[1]
        x = np.arange(2)
        width = 0.35

        normal_recon = self.pred_data['recon_errors'][normal_mask].mean()
        normal_disc = self.pred_data['discrepancies'][normal_mask].mean()
        anomaly_recon = self.pred_data['recon_errors'][anomaly_mask].mean()
        anomaly_disc = self.pred_data['discrepancies'][anomaly_mask].mean()

        bars1 = ax.bar(x - width/2, [normal_recon, anomaly_recon], width, label='Recon Error', color=VIS_COLORS['normal'])
        bars2 = ax.bar(x + width/2, [normal_disc, anomaly_disc], width, label='Discrepancy', color=VIS_COLORS['anomaly'])

        ax.set_xticks(x)
        ax.set_xticklabels(['Normal', 'Anomaly'])
        ax.set_ylabel('Mean Value')
        ax.set_title('Score Component Contributions', fontsize=12, fontweight='bold')
        ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'best_model_score_components.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - best_model_score_components.png")

    def plot_teacher_student_comparison(self):
        """Compare teacher and student reconstructions"""
        normal_mask = self.detailed_data['labels'] == 0
        anomaly_mask = self.detailed_data['labels'] == 1
        mask_inverse = 1 - self.detailed_data['masks']

        # Per-sample mean errors
        teacher_mean = (self.detailed_data['teacher_errors'] * mask_inverse).sum(axis=1) / (mask_inverse.sum(axis=1) + 1e-8)
        student_mean = (self.detailed_data['student_errors'] * mask_inverse).sum(axis=1) / (mask_inverse.sum(axis=1) + 1e-8)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Scatter plot
        ax = axes[0]
        ax.scatter(teacher_mean[normal_mask], student_mean[normal_mask],
                  alpha=0.6, label='Normal', color=VIS_COLORS['normal'], s=30)
        ax.scatter(teacher_mean[anomaly_mask], student_mean[anomaly_mask],
                  alpha=0.6, label='Anomaly', color=VIS_COLORS['anomaly'], s=30)

        max_val = max(teacher_mean.max(), student_mean.max())
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='y=x')

        ax.set_xlabel('Teacher Reconstruction Error')
        ax.set_ylabel('Student Reconstruction Error')
        ax.set_title('Teacher vs Student Error', fontsize=12, fontweight='bold')
        ax.legend()

        # Discrepancy distribution
        ax = axes[1]
        disc_mean = (self.detailed_data['discrepancies'] * mask_inverse).sum(axis=1) / (mask_inverse.sum(axis=1) + 1e-8)

        ax.hist(disc_mean[normal_mask], bins=30, alpha=0.6, label='Normal', color=VIS_COLORS['normal'], density=True)
        ax.hist(disc_mean[anomaly_mask], bins=30, alpha=0.6, label='Anomaly', color=VIS_COLORS['anomaly'], density=True)
        ax.set_xlabel('Mean Discrepancy')
        ax.set_ylabel('Density')
        ax.set_title('Discrepancy Distribution', fontsize=12, fontweight='bold')
        ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'best_model_teacher_student_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - best_model_teacher_student_comparison.png")

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
                idx = indices[0]
                original = self.detailed_data['originals'][idx, :, 0]
                point_labels = self.detailed_data['point_labels'][idx]
                x = np.arange(len(original))

                # Highlight anomaly regions first (so they appear behind the line)
                self._highlight_anomaly_regions(ax, point_labels, alpha=0.3, label='Anomaly Region')

                # Highlight masked region (last patch)
                mask_start = len(original) - self.config.mask_last_n
                ax.axvspan(mask_start, len(original), alpha=0.2, color=VIS_COLORS['masked_region'], label='Masked Region')

                ax.plot(x, original, color=color, lw=2, label='Signal')
                ax.set_title(f'{title}\nScore: {self.pred_data["scores"][idx]:.4f}, '
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

        # Hide empty subplots
        for idx in range(len(ANOMALY_TYPE_NAMES), len(axes)):
            axes[idx].axis('off')

        plt.suptitle('Loss Distribution by Anomaly Type', fontsize=14, fontweight='bold', y=1.02)
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
        anomaly_types = []
        detection_rates = []
        mean_scores = []
        counts = []

        for atype, metrics in metrics_json.items():
            if atype == 'normal':
                continue  # Skip normal for this plot
            anomaly_types.append(atype.replace('_', '\n'))
            detection_rates.append(metrics.get('detection_rate', 0) * 100)
            mean_scores.append(metrics.get('mean_score', 0))
            counts.append(metrics.get('count', 0))

        if len(anomaly_types) == 0:
            print("  ! Skipping performance_by_anomaly_type (no anomaly types found)")
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Detection rate bar chart
        ax = axes[0]
        anomaly_colors = get_anomaly_colors()
        colors = [anomaly_colors.get(atype, VIS_COLORS['reference']) for atype in anomaly_types]
        bars = ax.bar(anomaly_types, detection_rates, color=colors, alpha=0.8, edgecolor=VIS_COLORS['baseline'])
        ax.set_ylabel('Detection Rate (%)')
        ax.set_title('Detection Rate by Anomaly Type', fontweight='bold')
        ax.set_ylim(0, 105)

        # Add value labels
        for bar, rate in zip(bars, detection_rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)

        # Mean score bar chart
        ax = axes[1]
        bars = ax.bar(anomaly_types, mean_scores, color=colors, alpha=0.8, edgecolor=VIS_COLORS['baseline'])
        ax.set_ylabel('Mean Anomaly Score')
        ax.set_title('Mean Score by Anomaly Type', fontweight='bold')

        for bar, score in zip(bars, mean_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                   f'{score:.4f}', ha='center', va='bottom', fontsize=8)

        # Sample count pie chart
        ax = axes[2]
        ax.pie(counts, labels=[t.replace('\n', ' ') for t in anomaly_types], colors=colors,
               autopct='%1.1f%%', startangle=90)
        ax.set_title('Sample Distribution', fontweight='bold')

        plt.suptitle('Performance Analysis by Anomaly Type', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_by_anomaly_type.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - performance_by_anomaly_type.png")

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

    def plot_hypothesis_verification(self):
        """Verify hypotheses about why disturbing normal might outperform pure normal.

        4 Hypotheses to verify:
        1. Anomaly pattern in window provides "hint" (increases discrepancy)
        2. Anomaly → Normal transition is unstable
        3. Pure normal has high variance leading to False Positives
        4. Data distribution issue (most training data is pure_normal)
        """
        pure_normal_mask = self.detailed_data['sample_types'] == 0
        disturbing_mask = self.detailed_data['sample_types'] == 1
        anomaly_mask = self.detailed_data['sample_types'] == 2

        mask_inverse = 1 - self.detailed_data['masks']
        seq_length = self.detailed_data['discrepancies'].shape[1]
        mask_last_n = getattr(self.config, 'mask_last_n', 10)

        # Compute scores for each sample
        def compute_sample_scores(sample_mask):
            teacher_err = (self.detailed_data['teacher_errors'][sample_mask] * mask_inverse[sample_mask]).sum(axis=1) / (mask_inverse[sample_mask].sum(axis=1) + 1e-8)
            disc = (self.detailed_data['discrepancies'][sample_mask] * mask_inverse[sample_mask]).sum(axis=1) / (mask_inverse[sample_mask].sum(axis=1) + 1e-8)
            total = teacher_err + self.config.lambda_disc * disc
            return teacher_err, disc, total

        pure_recon, pure_disc, pure_total = compute_sample_scores(pure_normal_mask)
        dist_recon, dist_disc, dist_total = compute_sample_scores(disturbing_mask)
        anom_recon, anom_disc, anom_total = compute_sample_scores(anomaly_mask)

        # Get global threshold from all data
        threshold = self._get_optimal_threshold()

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Hypothesis 1: Anomaly ratio vs Score (for disturbing normal)
        ax = axes[0, 0]
        # For disturbing normal samples, compute anomaly ratio in window (excluding last patch)
        dist_point_labels = self.detailed_data['point_labels'][disturbing_mask]
        dist_anomaly_ratio = (dist_point_labels[:, :-mask_last_n] > 0).mean(axis=1)

        scatter = ax.scatter(dist_anomaly_ratio, dist_total, alpha=0.5, c=dist_disc,
                            cmap='RdYlGn_r', s=20)
        plt.colorbar(scatter, ax=ax, label='Discrepancy')
        ax.axhline(y=threshold, color=VIS_COLORS['anomaly_region'], linestyle='--', lw=2, label=f'Threshold={threshold:.4f}')
        ax.set_xlabel('Anomaly Ratio in Window (excluding last patch)')
        ax.set_ylabel('Total Score')
        ax.set_title('H1: Does anomaly in window increase score?\n(Disturbing Normal samples)', fontweight='bold')
        ax.legend()

        # Add correlation coefficient
        corr = np.corrcoef(dist_anomaly_ratio, dist_total)[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # 2. Hypothesis 2: Distance from anomaly end to last patch
        ax = axes[0, 1]
        # For disturbing normal, find the distance from last anomaly point to last patch start
        dist_distances = []
        for i, pl in enumerate(dist_point_labels):
            anomaly_positions = np.where(pl > 0)[0]
            if len(anomaly_positions) > 0:
                last_anomaly_pos = anomaly_positions[-1]
                distance = (seq_length - mask_last_n) - last_anomaly_pos
                dist_distances.append(max(0, distance))  # 0 if anomaly touches last patch boundary
            else:
                dist_distances.append(seq_length)  # No anomaly, maximum distance
        dist_distances = np.array(dist_distances)

        scatter = ax.scatter(dist_distances, dist_total, alpha=0.5, c=dist_disc,
                            cmap='RdYlGn_r', s=20)
        plt.colorbar(scatter, ax=ax, label='Discrepancy')
        ax.axhline(y=threshold, color=VIS_COLORS['anomaly_region'], linestyle='--', lw=2, label=f'Threshold')
        ax.set_xlabel('Distance from Last Anomaly to Last Patch Start')
        ax.set_ylabel('Total Score')
        ax.set_title('H2: Does recent anomaly affect score?\n(Disturbing Normal samples)', fontweight='bold')
        ax.legend()

        # Add correlation
        corr = np.corrcoef(dist_distances, dist_total)[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # 3. Hypothesis 3: Error variance comparison
        ax = axes[0, 2]
        # Compare variance of scores across sample types
        data_for_violin = [pure_total, dist_total, anom_total]
        labels = ['Pure\nNormal', 'Disturbing\nNormal', 'Anomaly']
        sample_colors = [VIS_COLORS['normal'], VIS_COLORS['disturbing'], VIS_COLORS['anomaly']]

        parts = ax.violinplot(data_for_violin, positions=[0, 1, 2], showmeans=True, showmedians=True)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(sample_colors[i])
            pc.set_alpha(0.7)

        ax.axhline(y=threshold, color=VIS_COLORS['threshold'], linestyle='--', lw=2, label=f'Global Threshold')
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(labels)
        ax.set_ylabel('Total Score')
        ax.set_title('H3: Score variance comparison\n(Higher variance → more FPs)', fontweight='bold')
        ax.legend()

        # Add variance values
        vars_text = f"Variance:\n  Pure: {pure_total.var():.6f}\n  Dist: {dist_total.var():.6f}\n  Anom: {anom_total.var():.6f}"
        ax.text(0.95, 0.95, vars_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', horizontalalignment='right',
               fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # 4. Classification metrics with GLOBAL threshold
        ax = axes[1, 0]
        # Apply global threshold to all sample types
        pure_fp = (pure_total > threshold).sum() / len(pure_total) * 100  # False Positive Rate
        dist_fp = (dist_total > threshold).sum() / len(dist_total) * 100  # False Positive Rate (should be low)
        anom_tp = (anom_total > threshold).sum() / len(anom_total) * 100  # True Positive Rate

        sample_colors = [VIS_COLORS['normal'], VIS_COLORS['disturbing'], VIS_COLORS['anomaly']]
        bars = ax.bar(['Pure Normal\n(FP Rate)', 'Disturbing Normal\n(FP Rate)', 'Anomaly\n(TP Rate)'],
                      [pure_fp, dist_fp, anom_tp], color=sample_colors)
        ax.set_ylabel('Rate (%)')
        ax.set_title('Classification Rates with Global Threshold', fontweight='bold')

        # Add values on bars
        for bar, val in zip(bars, [pure_fp, dist_fp, anom_tp]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=10)

        # 5. Reconstruction error by position (temporal analysis)
        ax = axes[1, 1]
        t = np.arange(seq_length)

        pure_recon_temporal = self.detailed_data['teacher_errors'][pure_normal_mask].mean(axis=0)
        dist_recon_temporal = self.detailed_data['teacher_errors'][disturbing_mask].mean(axis=0)
        anom_recon_temporal = self.detailed_data['teacher_errors'][anomaly_mask].mean(axis=0)

        ax.plot(t, pure_recon_temporal, label='Pure Normal', color=VIS_COLORS['normal'], lw=2)
        ax.plot(t, dist_recon_temporal, label='Disturbing Normal', color=VIS_COLORS['disturbing'], lw=2)
        ax.plot(t, anom_recon_temporal, label='Anomaly', color=VIS_COLORS['anomaly'], lw=2)
        ax.axvspan(seq_length - mask_last_n, seq_length, alpha=0.2, color=VIS_COLORS['reference'], label='Last Patch (masked)')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Mean Reconstruction Error')
        ax.set_title('Temporal Reconstruction Error\n(Where does error come from?)', fontweight='bold')
        ax.legend()

        # 6. Summary statistics
        ax = axes[1, 2]
        ax.axis('off')

        summary_text = f"""
Hypothesis Verification Summary
═══════════════════════════════════════════════════════

Global Threshold: {threshold:.6f}

H1: Anomaly Ratio Correlation
  • Corr(anomaly_ratio, score): {np.corrcoef(dist_anomaly_ratio, dist_total)[0,1]:.3f}
  • Interpretation: {'Positive' if corr > 0.1 else 'Weak/None'} correlation
    → Anomaly in window {'DOES' if corr > 0.1 else 'does NOT'} increase score

H2: Distance from Anomaly Effect
  • Corr(distance, score): {np.corrcoef(dist_distances, dist_total)[0,1]:.3f}
  • Interpretation: Recent anomaly {'affects' if np.corrcoef(dist_distances, dist_total)[0,1] < -0.1 else 'does NOT affect'} last patch

H3: Variance Analysis
  • Pure Normal variance:      {pure_total.var():.6f}
  • Disturbing Normal variance:{dist_total.var():.6f}
  • Anomaly variance:          {anom_total.var():.6f}
  • Pure > Dist variance: {pure_total.var() > dist_total.var()}
    → {'Higher pure variance → more FPs' if pure_total.var() > dist_total.var() else 'Similar variance'}

Classification with Global Threshold:
  • Pure Normal FP Rate:      {pure_fp:.1f}%
  • Disturbing Normal FP Rate:{dist_fp:.1f}%
  • Anomaly TP Rate:          {anom_tp:.1f}%
        """

        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        plt.suptitle('Hypothesis Verification: Why might Disturbing Normal outperform Pure Normal?',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'hypothesis_verification.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - hypothesis_verification.png")

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
            median_idx = indices[np.argsort(cat_scores)[len(cat_scores)//2]]

            # Column 1: Time series with reconstruction
            ax = axes[row, 0]
            original = self.detailed_data['originals'][median_idx, :, 0]
            teacher_recon = self.detailed_data['teacher_recons'][median_idx, :, 0]
            student_recon = self.detailed_data['student_recons'][median_idx, :, 0]
            point_labels = self.detailed_data['point_labels'][median_idx]

            ax.plot(original, 'b-', lw=1.2, alpha=0.8, label='Original')
            ax.plot(teacher_recon, 'g--', lw=1.5, alpha=0.7, label='Teacher')
            ax.plot(student_recon, 'r:', lw=1.5, alpha=0.7, label='Student')

            # Highlight anomaly and masked regions
            anomaly_region = np.where(point_labels == 1)[0]
            if len(anomaly_region) > 0:
                ax.axvspan(anomaly_region[0], anomaly_region[-1], alpha=0.2, color=VIS_COLORS['anomaly_region'], label='Anomaly')
            mask_start = len(original) - self.config.mask_last_n
            ax.axvspan(mask_start, len(original), alpha=0.2, color=VIS_COLORS['masked_region'], label='Masked')

            ax.set_title(f'{cat_name}: Time Series', fontweight='bold', color=color)
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Value')
            if row == 0:
                ax.legend(fontsize=7, loc='upper right')

            # Column 2: Discrepancy profile
            ax = axes[row, 1]
            discrepancy = self.detailed_data['discrepancies'][median_idx]
            ax.fill_between(range(len(discrepancy)), discrepancy, alpha=0.6, color=VIS_COLORS['student'])
            ax.plot(discrepancy, color=VIS_COLORS['student_dark'], lw=1)

            if len(anomaly_region) > 0:
                ax.axvspan(anomaly_region[0], anomaly_region[-1], alpha=0.2, color=VIS_COLORS['anomaly_region'])
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

            sample_score = scores[median_idx]
            teacher_err = np.mean((original[-self.config.mask_last_n:] - teacher_recon[-self.config.mask_last_n:])**2)
            student_err = np.mean((original[-self.config.mask_last_n:] - student_recon[-self.config.mask_last_n:])**2)
            masked_disc = np.mean(discrepancy[-self.config.mask_last_n:])

            stype = sample_types[median_idx]
            stype_name = ['Pure Normal', 'Disturbing Normal', 'Anomaly'][stype]

            stats_text = f"""
{cat_name} Case Study
═══════════════════════════════════

Sample Index: {median_idx}
Sample Type:  {stype_name}
True Label:   {'Anomaly' if labels[median_idx] == 1 else 'Normal'}
Prediction:   {'Anomaly' if predictions[median_idx] == 1 else 'Normal'}

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

    def plot_feature_contribution_analysis(self):
        """Analyze which features contribute most to anomaly detection.

        Shows per-feature reconstruction error and discrepancy patterns.
        """
        # Get per-feature statistics
        num_features = self.detailed_data['originals'].shape[2]
        threshold = self._get_optimal_threshold()
        scores = self._get_scores()
        predictions = (scores >= threshold).astype(int)
        labels = self.pred_data['labels']

        tp_mask = (labels == 1) & (predictions == 1)
        fn_mask = (labels == 1) & (predictions == 0)
        tn_mask = (labels == 0) & (predictions == 0)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Per-feature reconstruction error for TP vs FN
        ax = axes[0, 0]
        feature_errors_tp = []
        feature_errors_fn = []
        feature_errors_tn = []

        for f in range(num_features):
            original_f = self.detailed_data['originals'][:, -self.config.mask_last_n:, f]
            teacher_f = self.detailed_data['teacher_recons'][:, -self.config.mask_last_n:, f]
            mse_f = ((original_f - teacher_f) ** 2).mean(axis=1)

            feature_errors_tp.append(mse_f[tp_mask].mean() if tp_mask.sum() > 0 else 0)
            feature_errors_fn.append(mse_f[fn_mask].mean() if fn_mask.sum() > 0 else 0)
            feature_errors_tn.append(mse_f[tn_mask].mean() if tn_mask.sum() > 0 else 0)

        x = np.arange(num_features)
        width = 0.25
        ax.bar(x - width, feature_errors_tp, width, label='TP (Detected Anomaly)', color=VIS_COLORS['teacher'])
        ax.bar(x, feature_errors_fn, width, label='FN (Missed Anomaly)', color=VIS_COLORS['anomaly'])
        ax.bar(x + width, feature_errors_tn, width, label='TN (Normal)', color=VIS_COLORS['normal'])
        ax.set_xticks(x)
        ax.set_xticklabels([f'F{i}' for i in range(num_features)])
        ax.set_xlabel('Feature')
        ax.set_ylabel('Mean Teacher MSE (Masked Region)')
        ax.set_title('Per-Feature Reconstruction Error', fontweight='bold')
        ax.legend()

        # 2. Per-feature discrepancy
        ax = axes[0, 1]
        feature_disc_tp = []
        feature_disc_fn = []
        feature_disc_tn = []

        for f in range(num_features):
            teacher_f = self.detailed_data['teacher_recons'][:, -self.config.mask_last_n:, f]
            student_f = self.detailed_data['student_recons'][:, -self.config.mask_last_n:, f]
            disc_f = np.abs(teacher_f - student_f).mean(axis=1)

            feature_disc_tp.append(disc_f[tp_mask].mean() if tp_mask.sum() > 0 else 0)
            feature_disc_fn.append(disc_f[fn_mask].mean() if fn_mask.sum() > 0 else 0)
            feature_disc_tn.append(disc_f[tn_mask].mean() if tn_mask.sum() > 0 else 0)

        ax.bar(x - width, feature_disc_tp, width, label='TP', color=VIS_COLORS['teacher'])
        ax.bar(x, feature_disc_fn, width, label='FN', color=VIS_COLORS['anomaly'])
        ax.bar(x + width, feature_disc_tn, width, label='TN', color=VIS_COLORS['normal'])
        ax.set_xticks(x)
        ax.set_xticklabels([f'F{i}' for i in range(num_features)])
        ax.set_xlabel('Feature')
        ax.set_ylabel('Mean Discrepancy (Masked Region)')
        ax.set_title('Per-Feature Discrepancy', fontweight='bold')
        ax.legend()

        # 3. Feature importance ranking
        ax = axes[0, 2]
        # Importance = how much a feature contributes to TP-TN separation
        importance = np.array(feature_errors_tp) - np.array(feature_errors_tn)
        sorted_idx = np.argsort(importance)[::-1]

        importance_colors = [VIS_COLORS['true_positive'] if imp > 0 else VIS_COLORS['false_negative'] for imp in importance[sorted_idx]]
        bars = ax.barh([f'Feature {i}' for i in sorted_idx], importance[sorted_idx], color=importance_colors)
        ax.axvline(x=0, color=VIS_COLORS['baseline'], linestyle='-', lw=1)
        ax.set_xlabel('Importance (TP Error - TN Error)')
        ax.set_title('Feature Importance Ranking', fontweight='bold')

        # 4. Example: Most important feature for a TP
        ax = axes[1, 0]
        if tp_mask.sum() > 0:
            most_important_f = sorted_idx[0]
            tp_idx = np.where(tp_mask)[0][0]

            original = self.detailed_data['originals'][tp_idx, :, most_important_f]
            teacher = self.detailed_data['teacher_recons'][tp_idx, :, most_important_f]
            student = self.detailed_data['student_recons'][tp_idx, :, most_important_f]

            ax.plot(original, 'b-', lw=1.2, label='Original')
            ax.plot(teacher, 'g--', lw=1.5, label='Teacher')
            ax.plot(student, 'r:', lw=1.5, label='Student')
            ax.axvspan(len(original) - self.config.mask_last_n, len(original), alpha=0.2, color=VIS_COLORS['masked_region'])
            ax.set_title(f'Most Important Feature (F{most_important_f}) - TP Example', fontweight='bold')
            ax.legend()
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')

        # 5. Example: Least important feature
        ax = axes[1, 1]
        if tp_mask.sum() > 0:
            least_important_f = sorted_idx[-1]
            tp_idx = np.where(tp_mask)[0][0]

            original = self.detailed_data['originals'][tp_idx, :, least_important_f]
            teacher = self.detailed_data['teacher_recons'][tp_idx, :, least_important_f]
            student = self.detailed_data['student_recons'][tp_idx, :, least_important_f]

            ax.plot(original, 'b-', lw=1.2, label='Original')
            ax.plot(teacher, 'g--', lw=1.5, label='Teacher')
            ax.plot(student, 'r:', lw=1.5, label='Student')
            ax.axvspan(len(original) - self.config.mask_last_n, len(original), alpha=0.2, color=VIS_COLORS['masked_region'])
            ax.set_title(f'Least Important Feature (F{least_important_f}) - Same TP', fontweight='bold')
            ax.legend()
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')

        # 6. Summary
        ax = axes[1, 2]
        ax.axis('off')

        summary_text = f"""
Feature Contribution Analysis
═══════════════════════════════════════════════════

Feature Importance Ranking:
{''.join([f'  {i+1}. Feature {sorted_idx[i]}: {importance[sorted_idx[i]]:+.6f}' + chr(10) for i in range(min(5, num_features))])}

Interpretation:
  • Positive = Feature helps detect anomalies
  • Negative = Feature hurts detection

Key Insight:
  Features with high TP error but low TN error
  are most discriminative for anomaly detection.

Detection Statistics:
  • TP samples: {tp_mask.sum()}
  • FN samples: {fn_mask.sum()}
  • TN samples: {tn_mask.sum()}
        """

        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        plt.suptitle('Feature Contribution Analysis: Which Features Drive Anomaly Detection?',
                    fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'feature_contribution_analysis.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - feature_contribution_analysis.png")

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

    def _plot_sample_detail(self, axes_row, idx, title_prefix, color, threshold):
        """Helper to plot detailed sample analysis in a row of 3 axes."""
        original = self.detailed_data['originals'][idx, :, 0]
        teacher_recon = self.detailed_data['teacher_recons'][idx, :, 0]
        student_recon = self.detailed_data['student_recons'][idx, :, 0]
        discrepancy = self.detailed_data['discrepancies'][idx]
        point_labels = self.detailed_data['point_labels'][idx]
        score = self._get_scores()[idx]
        label = self.pred_data['labels'][idx]
        sample_type = self.detailed_data['sample_types'][idx]

        anomaly_region = np.where(point_labels == 1)[0]
        mask_start = len(original) - self.config.mask_last_n

        # Column 1: Time series
        ax = axes_row[0]
        ax.plot(original, 'b-', lw=1.2, alpha=0.8, label='Original')
        ax.plot(teacher_recon, 'g--', lw=1.5, alpha=0.7, label='Teacher')
        ax.plot(student_recon, 'r:', lw=1.5, alpha=0.7, label='Student')
        if len(anomaly_region) > 0:
            ax.axvspan(anomaly_region[0], anomaly_region[-1], alpha=0.2, color=VIS_COLORS['anomaly_region'])
        ax.axvspan(mask_start, len(original), alpha=0.2, color=VIS_COLORS['masked_region'])
        ax.set_title(f'{title_prefix}: Time Series', fontweight='bold', color=color)
        ax.legend(fontsize=7)

        # Column 2: Discrepancy
        ax = axes_row[1]
        ax.fill_between(range(len(discrepancy)), discrepancy, alpha=0.6, color=VIS_COLORS['student'])
        ax.plot(discrepancy, color=VIS_COLORS['student_dark'], lw=1)
        if len(anomaly_region) > 0:
            ax.axvspan(anomaly_region[0], anomaly_region[-1], alpha=0.2, color=VIS_COLORS['anomaly_region'])
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

Index: {idx}
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

    def plot_learning_curve(self, history: Dict, use_student_recon: bool = False):
        """Plot learning curves for training losses.

        Uses consistent color/marker scheme:
        - Colors: Normal=blue, Anomaly=red, Teacher=green, Student=purple
        - Markers: Discrepancy=square, Teacher recon=circle, Student recon=triangle

        Args:
            history: Training history dictionary with keys:
                - epoch: list of epoch numbers
                - train_rec_loss: teacher reconstruction loss per epoch
                - train_disc_loss: discrepancy loss per epoch
                - train_student_recon_loss: student reconstruction loss (optional)
                - train_normal_loss: normal discrepancy loss (optional)
                - train_anomaly_loss: anomaly discrepancy loss (optional)
                - train_teacher_recon_normal/anomaly: teacher recon by type
                - train_student_recon_normal/anomaly: student recon by type
            use_student_recon: Whether student reconstruction loss is enabled
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

    def generate_all(self, experiment_dir: str = None, history: Dict = None, use_student_recon: bool = False):
        """Generate all best model visualizations

        Args:
            experiment_dir: Path to experiment directory for loading detailed results
            history: Training history dictionary for learning curve visualization
            use_student_recon: Whether student reconstruction loss is enabled
        """
        print("\n  Generating Best Model Visualizations...")
        self.plot_roc_curve()
        self.plot_score_distribution()
        self.plot_confusion_matrix()
        self.plot_score_components()
        self.plot_teacher_student_comparison()
        self.plot_reconstruction_examples()
        self.plot_detection_examples()
        self.plot_summary_statistics()
        self.plot_learning_curve(history, use_student_recon)
        self.plot_pure_vs_disturbing_normal()
        self.plot_discrepancy_trend()
        self.plot_hypothesis_verification()

        # Qualitative case studies
        self.plot_case_study_gallery(experiment_dir)
        self.plot_anomaly_type_case_studies(experiment_dir)
        self.plot_feature_contribution_analysis()
        self.plot_hardest_samples()

        # Anomaly type analysis (requires detailed results from experiment)
        self.plot_loss_by_anomaly_type(experiment_dir)
        self.plot_performance_by_anomaly_type(experiment_dir)
        self.plot_loss_scatter_by_anomaly_type(experiment_dir)
        self.plot_sample_type_analysis(experiment_dir)


# =============================================================================
# TrainingProgressVisualizer - Training Progress Analysis
# =============================================================================

