"""
Training Progress Visualizer - Training Progress Analysis

This module provides visualizations for:
- Training dynamics over epochs
- Score trajectories
- Threshold evolution
- Performance evolution
- Learning curves
- Anomaly type learning progress
"""

import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score as sklearn_f1_score
from tqdm import tqdm

from mae_anomaly import (
    Config, SelfDistilledMAEMultivariate, set_seed,
    SlidingWindowTimeSeriesGenerator, SlidingWindowDataset,
    ANOMALY_TYPE_NAMES, SelfDistillationLoss, Trainer,
)
from .base import get_anomaly_colors, SAMPLE_TYPE_COLORS, SAMPLE_TYPE_NAMES, VIS_COLORS

class TrainingProgressVisualizer:
    """Visualize how the model learns over training epochs

    Re-trains the best model configuration and collects data at checkpoints
    to visualize:
    - Score distribution evolution
    - Sample trajectory (how each sample's score changes)
    - Metrics evolution (ROC-AUC, F1 over epochs)
    - Late bloomer analysis (samples that were FN early but TP later)
    - Anomaly type learning speed
    """

    def __init__(self, best_config: Dict, output_dir: str,
                 checkpoint_epochs: List[int] = None,
                 num_train: int = 2000, num_test: int = 500, max_epochs: int = 100):
        """Initialize TrainingProgressVisualizer

        Args:
            best_config: Best model configuration dict
            output_dir: Output directory for visualizations
            checkpoint_epochs: Epochs at which to collect data
            num_train: Number of training samples (Stage 2 default: 2000)
            num_test: Number of test samples (Stage 2 default: 500)
            max_epochs: Maximum epochs to train (Stage 2 default: 100)
        """
        self.best_config = best_config
        self.output_dir = output_dir
        self.num_train = num_train
        self.num_test = num_test
        self.max_epochs = max_epochs

        if checkpoint_epochs is None:
            self.checkpoint_epochs = [0, 5, 10, 20, 40, 60, 80, 100]
        else:
            self.checkpoint_epochs = checkpoint_epochs

        os.makedirs(output_dir, exist_ok=True)

        # Will be populated by retrain_with_checkpoints()
        self.checkpoint_data = {}
        self.config = None

    def retrain_with_checkpoints(self):
        """Re-train the model and collect data at checkpoint epochs"""
        from mae_anomaly.trainer import Trainer
        from mae_anomaly.evaluator import Evaluator

        print("\n  Re-training model with checkpoints...")
        print(f"  Checkpoints at epochs: {self.checkpoint_epochs}")

        # Create config from best_config
        self.config = Config()
        for key, value in self.best_config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # Set Stage 2 settings
        self.config.num_train_samples = self.num_train
        self.config.num_test_samples = self.num_test
        self.config.num_epochs = self.max_epochs

        # Create datasets with same seeds as run_experiments.py
        set_seed(self.config.random_seed)
        generator = SlidingWindowTimeSeriesGenerator(
            total_length=self.config.sliding_window_total_length,
            num_features=self.config.num_features,
            interval_scale=self.config.anomaly_interval_scale,
            seed=self.config.random_seed
        )
        signals, point_labels, anomaly_regions = generator.generate()

        train_dataset = SlidingWindowDataset(
            signals=signals,
            point_labels=point_labels,
            anomaly_regions=anomaly_regions,
            window_size=self.config.seq_length,
            stride=self.config.sliding_window_stride,
            mask_last_n=10,
            split='train',
            train_ratio=0.5,
            seed=self.config.random_seed
        )

        test_dataset = SlidingWindowDataset(
            signals=signals,
            point_labels=point_labels,
            anomaly_regions=anomaly_regions,
            window_size=self.config.seq_length,
            stride=self.config.sliding_window_stride,
            mask_last_n=10,
            split='test',
            train_ratio=0.5,
            target_counts={
                'pure_normal': int(self.config.num_test_samples * self.config.test_ratio_pure_normal),
                'disturbing_normal': int(self.config.num_test_samples * self.config.test_ratio_disturbing_normal),
                'anomaly': int(self.config.num_test_samples * self.config.test_ratio_anomaly)
            },
            seed=self.config.random_seed
        )

        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)

        # Create model and trainer
        model = SelfDistilledMAEMultivariate(self.config)
        model = model.to(self.config.device)

        trainer = Trainer(model, self.config, train_loader, test_loader, verbose=False)
        evaluator = Evaluator(model, self.config, test_loader)

        # Collect data at epoch 0 (before training)
        print(f"  Collecting data at epoch 0 (before training)...")
        self._collect_checkpoint_data(0, model, test_loader, evaluator)

        # Train and collect at checkpoints
        for epoch in range(self.max_epochs):
            # Train one epoch (Trainer.train_epoch takes epoch index)
            trainer.train_epoch(epoch)
            trainer.scheduler.step()

            actual_epoch = epoch + 1
            if actual_epoch in self.checkpoint_epochs:
                print(f"  Collecting data at epoch {actual_epoch}...")
                self._collect_checkpoint_data(actual_epoch, model, test_loader, evaluator)

        print(f"  Training complete! Collected {len(self.checkpoint_data)} checkpoints.")

    def _collect_checkpoint_data(self, epoch: int, model, test_loader, evaluator):
        """Collect data at a checkpoint epoch"""
        model.eval()
        device = self.config.device

        # Compute scores using evaluator (same method as run_experiments.py)
        scores, labels, sample_types, anomaly_types = evaluator.compute_anomaly_scores()

        # Compute metrics
        metrics = evaluator.evaluate()

        # Collect detailed data for a subset of samples (for reconstruction visualization)
        detailed_samples = self._collect_sample_details(model, test_loader, num_samples=20)

        self.checkpoint_data[epoch] = {
            'scores': scores,
            'labels': labels,
            'sample_types': sample_types,
            'anomaly_types': anomaly_types,
            'roc_auc': metrics['roc_auc'],
            'f1_score': metrics['f1_score'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'detailed_samples': detailed_samples
        }

    def _collect_sample_details(self, model, test_loader, num_samples: int = 20) -> Dict:
        """Collect detailed reconstruction data for a subset of samples"""
        model.eval()
        device = self.config.device

        originals = []
        teacher_recons = []
        student_recons = []
        labels = []
        sample_types = []
        point_labels = []

        collected = 0
        with torch.no_grad():
            for batch in test_loader:
                sequences, last_patch_labels, pt_labels, st, _ = batch
                sequences = sequences.to(device)
                batch_size, seq_length, num_features = sequences.shape

                # Create mask for last n positions
                mask = torch.ones(batch_size, seq_length, device=device)
                mask[:, -self.config.mask_last_n:] = 0

                teacher_output, student_output, _ = model(sequences, masking_ratio=0.0, mask=mask)

                for i in range(batch_size):
                    if collected >= num_samples:
                        break
                    originals.append(sequences[i].cpu().numpy())
                    teacher_recons.append(teacher_output[i].cpu().numpy())
                    student_recons.append(student_output[i].cpu().numpy())
                    labels.append(last_patch_labels[i].item())
                    sample_types.append(st[i].item())
                    point_labels.append(pt_labels[i].numpy())
                    collected += 1

                if collected >= num_samples:
                    break

        return {
            'originals': np.array(originals),
            'teacher_recons': np.array(teacher_recons),
            'student_recons': np.array(student_recons),
            'labels': np.array(labels),
            'sample_types': np.array(sample_types),
            'point_labels': np.array(point_labels)
        }

    def plot_score_evolution(self):
        """Plot how score distributions evolve over training"""
        epochs_to_plot = [e for e in self.checkpoint_epochs if e in self.checkpoint_data]

        # Use 2x3 grid for 6 checkpoints
        n_plots = min(6, len(epochs_to_plot))
        if n_plots < 6:
            indices = list(range(n_plots))
        else:
            # Select 6 evenly spaced epochs
            indices = [0, 1, 2, len(epochs_to_plot)//2, -2, -1]
            indices = sorted(set([i % len(epochs_to_plot) for i in indices]))[:6]

        selected_epochs = [epochs_to_plot[i] for i in indices[:n_plots]]

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for idx, epoch in enumerate(selected_epochs):
            if idx >= 6:
                break
            ax = axes[idx]
            data = self.checkpoint_data[epoch]

            normal_mask = data['labels'] == 0
            anomaly_mask = data['labels'] == 1

            # Histogram
            ax.hist(data['scores'][normal_mask], bins=30, alpha=0.6,
                   label=f'Normal (n={normal_mask.sum()})', color=VIS_COLORS['normal'], density=True)
            ax.hist(data['scores'][anomaly_mask], bins=30, alpha=0.6,
                   label=f'Anomaly (n={anomaly_mask.sum()})', color=VIS_COLORS['anomaly'], density=True)

            # Add vertical lines for means
            ax.axvline(data['scores'][normal_mask].mean(), color=VIS_COLORS['normal'], linestyle='--', lw=2)
            ax.axvline(data['scores'][anomaly_mask].mean(), color=VIS_COLORS['anomaly'], linestyle='--', lw=2)

            ax.set_xlabel('Anomaly Score')
            ax.set_ylabel('Density')
            ax.set_title(f'Epoch {epoch} (ROC-AUC: {data["roc_auc"]:.4f})', fontsize=12, fontweight='bold')
            ax.legend(fontsize=8)

        # Hide unused axes
        for idx in range(n_plots, 6):
            axes[idx].axis('off')

        plt.suptitle('Score Distribution Evolution During Training\n'
                    '(Normal=Blue, Anomaly=Red, Dashed=Mean)',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'score_evolution.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - score_evolution.png")

    def plot_sample_trajectories(self):
        """Plot how individual sample scores change over training"""
        epochs = sorted(self.checkpoint_data.keys())
        n_samples = len(self.checkpoint_data[epochs[0]]['scores'])

        # Build score matrix: (n_samples, n_epochs)
        score_matrix = np.zeros((n_samples, len(epochs)))
        for i, epoch in enumerate(epochs):
            score_matrix[:, i] = self.checkpoint_data[epoch]['scores']

        labels = self.checkpoint_data[epochs[0]]['labels']
        sample_types = self.checkpoint_data[epochs[0]]['sample_types']

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Left: All samples with color by label
        ax = axes[0]
        for i in range(n_samples):
            if labels[i] == 0:
                color = VIS_COLORS['normal']
                alpha = 0.1
            else:
                color = VIS_COLORS['anomaly']
                alpha = 0.3
            ax.plot(epochs, score_matrix[i, :], color=color, alpha=alpha, lw=0.5)

        # Add mean trajectories
        normal_mean = score_matrix[labels == 0].mean(axis=0)
        anomaly_mean = score_matrix[labels == 1].mean(axis=0)
        ax.plot(epochs, normal_mean, color=VIS_COLORS['normal_dark'], lw=3, label='Normal Mean')
        ax.plot(epochs, anomaly_mean, color=VIS_COLORS['anomaly_dark'], lw=3, label='Anomaly Mean')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Anomaly Score')
        ax.set_title('Sample Score Trajectories (Normal=Blue, Anomaly=Red)', fontweight='bold')
        ax.legend()

        # Right: By sample type (Pure Normal, Disturbing Normal, Anomaly)
        ax = axes[1]
        colors = SAMPLE_TYPE_COLORS
        labels_map = SAMPLE_TYPE_NAMES

        for stype in [0, 1, 2]:
            mask = sample_types == stype
            if mask.sum() > 0:
                mean_traj = score_matrix[mask].mean(axis=0)
                std_traj = score_matrix[mask].std(axis=0)
                ax.plot(epochs, mean_traj, color=colors[stype], lw=2, label=f'{labels_map[stype]} (n={mask.sum()})')
                ax.fill_between(epochs, mean_traj - std_traj, mean_traj + std_traj,
                               color=colors[stype], alpha=0.2)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Anomaly Score')
        ax.set_title('Mean Score Trajectory by Sample Type (±1 std)', fontweight='bold')
        ax.legend()

        plt.suptitle('Learning Trajectories: How Scores Evolve During Training',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'sample_trajectories.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - sample_trajectories.png")

    def plot_metrics_evolution(self):
        """Plot how metrics evolve over training"""
        epochs = sorted(self.checkpoint_data.keys())

        roc_aucs = [self.checkpoint_data[e]['roc_auc'] for e in epochs]
        f1_scores = [self.checkpoint_data[e]['f1_score'] for e in epochs]
        precisions = [self.checkpoint_data[e]['precision'] for e in epochs]
        recalls = [self.checkpoint_data[e]['recall'] for e in epochs]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # ROC-AUC
        ax = axes[0, 0]
        ax.plot(epochs, roc_aucs, 'o-', color=VIS_COLORS['anomaly'], lw=2, markersize=8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('ROC-AUC')
        ax.set_title('ROC-AUC Evolution', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([min(0.5, min(roc_aucs) - 0.05), 1.0])

        # F1 Score
        ax = axes[0, 1]
        ax.plot(epochs, f1_scores, 'o-', color=VIS_COLORS['teacher'], lw=2, markersize=8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('F1-Score')
        ax.set_title('F1-Score Evolution', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Precision and Recall
        ax = axes[1, 0]
        ax.plot(epochs, precisions, 'o-', color=VIS_COLORS['normal'], lw=2, markersize=8, label='Precision')
        ax.plot(epochs, recalls, 's-', color=VIS_COLORS['student'], lw=2, markersize=8, label='Recall')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_title('Precision & Recall Evolution', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # All metrics together
        ax = axes[1, 1]
        ax.plot(epochs, roc_aucs, 'o-', lw=2, label='ROC-AUC')
        ax.plot(epochs, f1_scores, 's-', lw=2, label='F1-Score')
        ax.plot(epochs, precisions, '^-', lw=2, label='Precision')
        ax.plot(epochs, recalls, 'v-', lw=2, label='Recall')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_title('All Metrics Combined', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.suptitle('Model Performance Metrics Over Training',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'metrics_evolution.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - metrics_evolution.png")

    def plot_late_bloomer_analysis(self):
        """Analyze samples that changed classification status over training.

        Late bloomers include:
        - Anomalies: FN → TP (missed at start, detected at end)
        - Normals: FP → TN (false alarm at start, correct at end)

        Uses per-epoch optimal thresholds for fair comparison.
        """
        epochs = sorted(self.checkpoint_data.keys())
        first_epoch = epochs[0]
        last_epoch = epochs[-1]

        first_data = self.checkpoint_data[first_epoch]
        last_data = self.checkpoint_data[last_epoch]

        # Calculate per-epoch optimal thresholds
        def get_optimal_threshold(data):
            fpr, tpr, thresholds = roc_curve(data['labels'], data['scores'])
            optimal_idx = np.argmax(tpr - fpr)
            return thresholds[optimal_idx]

        first_threshold = get_optimal_threshold(first_data)
        last_threshold = get_optimal_threshold(last_data)

        # Get predictions using per-epoch thresholds
        first_preds = (first_data['scores'] >= first_threshold).astype(int)
        last_preds = (last_data['scores'] >= last_threshold).astype(int)

        # Late bloomer anomalies: FN → TP
        late_bloomer_anomalies = np.where(
            (first_data['labels'] == 1) &  # True anomaly
            (first_preds == 0) &  # Predicted normal at first
            (last_preds == 1)  # Predicted anomaly at last
        )[0]

        # Late bloomer normals: FP → TN (false alarm corrected)
        late_bloomer_normals = np.where(
            (first_data['labels'] == 0) &  # True normal
            (first_preds == 1) &  # Predicted anomaly at first (FP)
            (last_preds == 0)  # Predicted normal at last (TN)
        )[0]

        # Persistent FN: Still missed at last epoch
        persistent_fn = np.where(
            (first_data['labels'] == 1) &  # True anomaly
            (last_preds == 0)  # Still predicted normal at last
        )[0]

        # Early correct anomalies: TP from the start
        early_correct_anomalies = np.where(
            (first_data['labels'] == 1) &
            (first_preds == 1) &
            (last_preds == 1)
        )[0]

        print(f"  Found {len(late_bloomer_anomalies)} late bloomer anomalies (FN→TP)")
        print(f"  Found {len(late_bloomer_normals)} late bloomer normals (FP→TN)")

        # Build score matrix and threshold trajectory
        n_samples = len(self.checkpoint_data[epochs[0]]['scores'])
        score_matrix = np.zeros((n_samples, len(epochs)))
        thresholds_by_epoch = []
        for i, epoch in enumerate(epochs):
            data = self.checkpoint_data[epoch]
            score_matrix[:, i] = data['scores']
            thresholds_by_epoch.append(get_optimal_threshold(data))

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))

        # 1. Late bloomer anomaly trajectories (FN → TP)
        ax = axes[0, 0]
        ax.plot(epochs, thresholds_by_epoch, 'k--', lw=2, alpha=0.7, label='Threshold')
        if len(late_bloomer_anomalies) > 0:
            for i in late_bloomer_anomalies[:15]:
                ax.plot(epochs, score_matrix[i, :], alpha=0.6, lw=1.5)
            ax.set_title(f'Late Bloomer Anomalies (FN→TP): n={len(late_bloomer_anomalies)}', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No FN→TP transitions found', ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Late Bloomer Anomalies (FN→TP)', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Anomaly Score')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        # 2. Late bloomer normals trajectories (FP → TN)
        ax = axes[0, 1]
        ax.plot(epochs, thresholds_by_epoch, 'k--', lw=2, alpha=0.7, label='Threshold')
        if len(late_bloomer_normals) > 0:
            for i in late_bloomer_normals[:15]:
                ax.plot(epochs, score_matrix[i, :], alpha=0.6, lw=1.5, color=VIS_COLORS['disturbing'])
            ax.set_title(f'Late Bloomer Normals (FP→TN): n={len(late_bloomer_normals)}', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No FP→TN transitions found', ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Late Bloomer Normals (FP→TN)', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Anomaly Score')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        # 3. Persistent FN trajectories
        ax = axes[0, 2]
        ax.plot(epochs, thresholds_by_epoch, 'k--', lw=2, alpha=0.7, label='Threshold')
        if len(persistent_fn) > 0:
            for i in persistent_fn[:15]:
                ax.plot(epochs, score_matrix[i, :], alpha=0.6, lw=1.5, color=VIS_COLORS['anomaly'])
            ax.set_title(f'Persistent FN (Always Missed): n={len(persistent_fn)}', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No persistent FN found', ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Persistent FN Trajectories', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Anomaly Score')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        # 4. Detection metrics evolution with per-epoch thresholds
        ax = axes[1, 0]
        detection_rates = []
        false_alarm_rates = []
        for i, epoch in enumerate(epochs):
            data = self.checkpoint_data[epoch]
            thresh = thresholds_by_epoch[i]
            preds = (data['scores'] >= thresh).astype(int)
            anomaly_mask = data['labels'] == 1
            normal_mask = data['labels'] == 0
            if anomaly_mask.sum() > 0:
                detection_rates.append((preds[anomaly_mask] == 1).sum() / anomaly_mask.sum() * 100)
            else:
                detection_rates.append(0)
            if normal_mask.sum() > 0:
                false_alarm_rates.append((preds[normal_mask] == 1).sum() / normal_mask.sum() * 100)
            else:
                false_alarm_rates.append(0)

        ax.plot(epochs, detection_rates, 'o-', color=VIS_COLORS['anomaly'], lw=2, markersize=8, label='Detection Rate (TPR)')
        ax.plot(epochs, false_alarm_rates, 's-', color=VIS_COLORS['normal'], lw=2, markersize=8, label='False Alarm Rate (FPR)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Rate (%)')
        ax.set_title('Detection & False Alarm Rate Over Training', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])

        # 5. Threshold evolution
        ax = axes[1, 1]
        ax.plot(epochs, thresholds_by_epoch, 'o-', color=VIS_COLORS['teacher'], lw=2, markersize=8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Optimal Threshold')
        ax.set_title('Optimal Threshold Evolution', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 6. Summary statistics
        ax = axes[1, 2]
        ax.axis('off')

        # Count all transitions
        total_anomalies = (first_data['labels'] == 1).sum()
        total_normals = (first_data['labels'] == 0).sum()

        tp_at_start = ((first_data['labels'] == 1) & (first_preds == 1)).sum()
        tp_at_end = ((last_data['labels'] == 1) & (last_preds == 1)).sum()
        fp_at_start = ((first_data['labels'] == 0) & (first_preds == 1)).sum()
        fp_at_end = ((last_data['labels'] == 0) & (last_preds == 1)).sum()

        summary_text = f"""
Late Bloomer Analysis Summary (Per-Epoch Thresholds)
═══════════════════════════════════════════════════════════

Thresholds:
  • Epoch {first_epoch}: {first_threshold:.4f}
  • Epoch {last_epoch}: {last_threshold:.4f}

Anomaly Detection (n={total_anomalies}):
  ┌─────────────────────────────────────────┐
  │ Epoch {first_epoch:3d}    │ Epoch {last_epoch:3d}    │ Count  │
  ├─────────────────────────────────────────┤
  │ TP         │ TP         │ {len(early_correct_anomalies):5d}  │ (Always Correct)
  │ FN         │ TP         │ {len(late_bloomer_anomalies):5d}  │ (Late Bloomer)
  │ FN         │ FN         │ {len(persistent_fn):5d}  │ (Persistent Miss)
  └─────────────────────────────────────────┘

Normal Classification (n={total_normals}):
  • FP→TN (Late Bloomer):  {len(late_bloomer_normals):5d}
  • FP at start:          {fp_at_start:5d}
  • FP at end:            {fp_at_end:5d}

Performance Improvement:
  • Detection Rate: {detection_rates[0]:.1f}% → {detection_rates[-1]:.1f}%  ({detection_rates[-1]-detection_rates[0]:+.1f}%)
  • False Alarm:    {false_alarm_rates[0]:.1f}% → {false_alarm_rates[-1]:.1f}%  ({false_alarm_rates[-1]-false_alarm_rates[0]:+.1f}%)
        """

        ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        plt.suptitle('Late Bloomer Analysis: Samples That Changed Classification Over Training\n'
                    '(Using Per-Epoch Optimal Thresholds)',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'late_bloomer_analysis.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - late_bloomer_analysis.png")

    def plot_anomaly_type_learning(self):
        """Plot how detection improves for each anomaly type over training"""
        epochs = sorted(self.checkpoint_data.keys())
        last_data = self.checkpoint_data[epochs[-1]]

        # Get threshold
        fpr, tpr, thresholds = roc_curve(last_data['labels'], last_data['scores'])
        optimal_idx = np.argmax(tpr - fpr)
        threshold = thresholds[optimal_idx]

        # Use actual anomaly type names from dataset
        anomaly_type_names = ANOMALY_TYPE_NAMES  # ['normal', 'spike', 'memory_leak', ...]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 1. Detection rate by anomaly type over epochs
        ax = axes[0]
        # Generate enough colors for all anomaly types from get_anomaly_colors()
        anomaly_colors = get_anomaly_colors()
        colors = [anomaly_colors.get(atype, VIS_COLORS['reference']) for atype in anomaly_type_names]

        for atype_idx, atype_name in enumerate(anomaly_type_names[1:], start=1):  # Skip normal
            detection_rates = []
            for epoch in epochs:
                data = self.checkpoint_data[epoch]
                mask = data['anomaly_types'] == atype_idx
                if mask.sum() > 0:
                    preds = (data['scores'][mask] >= threshold).astype(int)
                    detection_rates.append(preds.sum() / mask.sum() * 100)
                else:
                    detection_rates.append(np.nan)

            ax.plot(epochs, detection_rates, 'o-', lw=2, markersize=6,
                   label=f'{atype_name} (n={mask.sum()})', color=colors[atype_idx])

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Detection Rate (%)')
        ax.set_title('Detection Rate by Anomaly Type Over Training', fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])

        # 2. Mean score by anomaly type over epochs
        ax = axes[1]

        for atype_idx, atype_name in enumerate(anomaly_type_names[1:], start=1):
            mean_scores = []
            for epoch in epochs:
                data = self.checkpoint_data[epoch]
                mask = data['anomaly_types'] == atype_idx
                if mask.sum() > 0:
                    mean_scores.append(data['scores'][mask].mean())
                else:
                    mean_scores.append(np.nan)

            ax.plot(epochs, mean_scores, 'o-', lw=2, markersize=6,
                   label=atype_name, color=colors[atype_idx])

        # Add normal for comparison
        normal_scores = []
        for epoch in epochs:
            data = self.checkpoint_data[epoch]
            mask = data['labels'] == 0
            if mask.sum() > 0:
                normal_scores.append(data['scores'][mask].mean())
            else:
                normal_scores.append(np.nan)
        ax.plot(epochs, normal_scores, 'o--', lw=2, markersize=6, label='Normal', color=VIS_COLORS['normal'])

        ax.axhline(y=threshold, color=VIS_COLORS['threshold'], linestyle='--', lw=2, alpha=0.7, label='Threshold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mean Anomaly Score')
        ax.set_title('Mean Score by Anomaly Type Over Training', fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.suptitle('Anomaly Type Learning: Which Types Are Easier/Harder to Detect?',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'anomaly_type_learning.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - anomaly_type_learning.png")

    def plot_reconstruction_evolution(self):
        """Show how reconstruction quality improves for specific samples.

        Enhanced version showing:
        - Original signal
        - Teacher reconstruction
        - Student reconstruction
        - Teacher-Student discrepancy (key for anomaly detection)
        """
        epochs = sorted(self.checkpoint_data.keys())

        # Select epochs to show
        if len(epochs) >= 4:
            show_epochs = [epochs[0], epochs[len(epochs)//3], epochs[2*len(epochs)//3], epochs[-1]]
        else:
            show_epochs = epochs

        # Get detailed samples
        first_details = self.checkpoint_data[epochs[0]]['detailed_samples']

        # Find anomaly samples
        anomaly_indices = np.where(first_details['labels'] == 1)[0]

        if len(anomaly_indices) == 0:
            print("  ! Skipping reconstruction_evolution (no anomaly samples)")
            return

        # Select up to 2 anomaly samples for better visualization
        selected_indices = anomaly_indices[:2]

        # Create figure: 2 rows per sample (reconstruction + discrepancy)
        n_samples = len(selected_indices)
        fig, axes = plt.subplots(n_samples * 2, len(show_epochs), figsize=(5*len(show_epochs), 3*n_samples*2))

        if n_samples == 1:
            axes = axes.reshape(2, -1)

        for sample_num, sample_idx in enumerate(selected_indices):
            original = first_details['originals'][sample_idx, :, 0]  # First feature
            point_labels = first_details['point_labels'][sample_idx]

            for col, epoch in enumerate(show_epochs):
                details = self.checkpoint_data[epoch]['detailed_samples']

                teacher_recon = details['teacher_recons'][sample_idx, :, 0]
                student_recon = details['student_recons'][sample_idx, :, 0]

                # Row 1: Reconstruction comparison
                ax = axes[sample_num * 2, col]

                ax.plot(original, 'b-', alpha=0.7, lw=1.2, label='Original')
                ax.plot(teacher_recon, 'g--', alpha=0.8, lw=1.5, label='Teacher')
                ax.plot(student_recon, 'r:', alpha=0.8, lw=1.5, label='Student')

                # Highlight anomaly region
                anomaly_region = np.where(point_labels == 1)[0]
                if len(anomaly_region) > 0:
                    ax.axvspan(anomaly_region[0], anomaly_region[-1], alpha=0.15, color=VIS_COLORS['anomaly_region'])

                # Highlight last patch (masked region)
                ax.axvspan(len(original) - self.config.mask_last_n, len(original),
                          alpha=0.15, color=VIS_COLORS['masked_region'])

                # Compute metrics in masked region
                masked_region = slice(-self.config.mask_last_n, None)
                teacher_mse = np.mean((original[masked_region] - teacher_recon[masked_region])**2)
                student_mse = np.mean((original[masked_region] - student_recon[masked_region])**2)

                ax.set_title(f'Epoch {epoch}\nT-MSE: {teacher_mse:.4f}, S-MSE: {student_mse:.4f}', fontsize=9)
                if col == 0:
                    ax.set_ylabel(f'Sample {sample_idx}\nReconstruction', fontsize=9)
                if sample_num == 0 and col == len(show_epochs) - 1:
                    ax.legend(fontsize=7, loc='upper right')

                # Row 2: Discrepancy (Teacher - Student)
                ax = axes[sample_num * 2 + 1, col]

                # Point-level discrepancy
                discrepancy = np.abs(teacher_recon - student_recon)
                ax.fill_between(range(len(discrepancy)), discrepancy, alpha=0.6, color=VIS_COLORS['student'], label='|T-S| Discrepancy')
                ax.plot(discrepancy, color=VIS_COLORS['student_dark'], lw=1)

                # Highlight regions
                if len(anomaly_region) > 0:
                    ax.axvspan(anomaly_region[0], anomaly_region[-1], alpha=0.15, color=VIS_COLORS['anomaly_region'])
                ax.axvspan(len(original) - self.config.mask_last_n, len(original),
                          alpha=0.15, color=VIS_COLORS['masked_region'])

                # Compute discrepancy in masked region
                masked_disc = np.mean(discrepancy[masked_region])
                full_disc = np.mean(discrepancy)

                ax.set_title(f'Discrepancy\nMasked: {masked_disc:.4f}, Full: {full_disc:.4f}', fontsize=9)
                if col == 0:
                    ax.set_ylabel(f'Sample {sample_idx}\nDiscrepancy', fontsize=9)
                if sample_num == n_samples - 1:
                    ax.set_xlabel('Time Step')
                if sample_num == 0 and col == len(show_epochs) - 1:
                    ax.legend(fontsize=7, loc='upper right')

        plt.suptitle('Reconstruction & Discrepancy Evolution for Anomaly Samples\n'
                    '(Yellow=Masked Region, Red=Anomaly Region)\n'
                    'Key Insight: Discrepancy should increase in masked anomaly regions as training progresses',
                    fontsize=12, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'reconstruction_evolution.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - reconstruction_evolution.png")

    def plot_decision_boundary_evolution(self):
        """Show how the decision boundary (threshold) evolves"""
        epochs = sorted(self.checkpoint_data.keys())

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 1. Optimal threshold over epochs
        ax = axes[0]
        thresholds = []
        for epoch in epochs:
            data = self.checkpoint_data[epoch]
            fpr, tpr, ths = roc_curve(data['labels'], data['scores'])
            optimal_idx = np.argmax(tpr - fpr)
            thresholds.append(ths[optimal_idx])

        ax.plot(epochs, thresholds, 'o-', color=VIS_COLORS['teacher'], lw=2, markersize=8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Optimal Threshold')
        ax.set_title('Optimal Threshold Evolution', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 2. Score separation over epochs
        ax = axes[1]
        separations = []
        for epoch in epochs:
            data = self.checkpoint_data[epoch]
            normal_mean = data['scores'][data['labels'] == 0].mean()
            anomaly_mean = data['scores'][data['labels'] == 1].mean()
            separations.append(anomaly_mean - normal_mean)

        ax.plot(epochs, separations, 'o-', color=VIS_COLORS['student'], lw=2, markersize=8)
        ax.axhline(y=0, color=VIS_COLORS['baseline'], linestyle='-', lw=1)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score Separation (Anomaly Mean - Normal Mean)')
        ax.set_title('Score Separation Evolution', fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.suptitle('Decision Boundary Analysis: How Separation Improves',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'decision_boundary_evolution.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - decision_boundary_evolution.png")

    def plot_late_bloomer_case_studies(self):
        """Generate detailed case studies of late bloomer samples.

        Shows time series, reconstruction, and discrepancy evolution for
        samples that transitioned from incorrect to correct classification.
        """
        epochs = sorted(self.checkpoint_data.keys())
        first_epoch = epochs[0]
        last_epoch = epochs[-1]

        first_data = self.checkpoint_data[first_epoch]
        last_data = self.checkpoint_data[last_epoch]

        # Per-epoch thresholds
        def get_optimal_threshold(data):
            fpr, tpr, thresholds = roc_curve(data['labels'], data['scores'])
            optimal_idx = np.argmax(tpr - fpr)
            return thresholds[optimal_idx]

        first_threshold = get_optimal_threshold(first_data)
        last_threshold = get_optimal_threshold(last_data)

        first_preds = (first_data['scores'] >= first_threshold).astype(int)
        last_preds = (last_data['scores'] >= last_threshold).astype(int)

        # Find late bloomer anomalies (FN → TP)
        late_bloomer_anomalies = np.where(
            (first_data['labels'] == 1) &
            (first_preds == 0) &
            (last_preds == 1)
        )[0]

        if len(late_bloomer_anomalies) == 0:
            print("  ! Skipping late_bloomer_case_studies (no late bloomers found)")
            return

        # Get detailed samples - check how many we have
        first_details = self.checkpoint_data[epochs[0]]['detailed_samples']
        n_detailed = len(first_details['originals'])

        # Filter late bloomers to only those in detailed samples
        valid_late_bloomers = [idx for idx in late_bloomer_anomalies if idx < n_detailed]

        if len(valid_late_bloomers) == 0:
            print(f"  ! Skipping late_bloomer_case_studies (late bloomers not in detailed samples, need idx < {n_detailed})")
            return

        # Select up to 3 late bloomers with most interesting score trajectories
        n_samples = min(3, len(valid_late_bloomers))
        selected_indices = valid_late_bloomers[:n_samples]

        print(f"  Plotting {n_samples} late bloomer case studies...")

        # Select epochs to show
        if len(epochs) >= 4:
            show_epochs = [epochs[0], epochs[len(epochs)//3], epochs[2*len(epochs)//3], epochs[-1]]
        else:
            show_epochs = epochs

        fig, axes = plt.subplots(n_samples * 3, len(show_epochs), figsize=(5*len(show_epochs), 4*n_samples*3))

        if n_samples == 1:
            axes = axes.reshape(3, -1)

        for sample_num, sample_idx in enumerate(selected_indices):
            # Get sample data from first epoch (sample_idx is already validated)
            original = first_details['originals'][sample_idx, :, 0]
            point_labels = first_details['point_labels'][sample_idx]
            anomaly_region = np.where(point_labels == 1)[0]

            for col, epoch in enumerate(show_epochs):
                details = self.checkpoint_data[epoch]['detailed_samples']

                teacher_recon = details['teacher_recons'][sample_idx, :, 0]
                student_recon = details['student_recons'][sample_idx, :, 0]

                # Get threshold and prediction for this epoch
                data = self.checkpoint_data[epoch]
                epoch_thresh = get_optimal_threshold(data)
                epoch_pred = 'Detected' if data['scores'][sample_idx] >= epoch_thresh else 'Missed'
                epoch_score = data['scores'][sample_idx]

                # Row 1: Time series with reconstruction
                ax = axes[sample_num * 3, col]
                ax.plot(original, 'b-', lw=1.2, alpha=0.8, label='Original')
                ax.plot(teacher_recon, 'g--', lw=1.5, alpha=0.7, label='Teacher')
                ax.plot(student_recon, 'r:', lw=1.5, alpha=0.7, label='Student')

                if len(anomaly_region) > 0:
                    ax.axvspan(anomaly_region[0], anomaly_region[-1], alpha=0.2, color=VIS_COLORS['anomaly_region'])
                ax.axvspan(len(original) - self.config.mask_last_n, len(original), alpha=0.2, color=VIS_COLORS['masked_region'])

                pred_color = VIS_COLORS['true_positive'] if epoch_pred == 'Detected' else VIS_COLORS['false_negative']
                ax.set_title(f'Epoch {epoch}: {epoch_pred}\nScore: {epoch_score:.4f} (Thresh: {epoch_thresh:.4f})',
                           fontsize=9, color=pred_color, fontweight='bold')
                if col == 0:
                    ax.set_ylabel(f'Late Bloomer #{sample_num+1}\nTime Series', fontsize=9)
                if sample_num == 0 and col == len(show_epochs) - 1:
                    ax.legend(fontsize=7, loc='upper right')

                # Row 2: Discrepancy
                ax = axes[sample_num * 3 + 1, col]
                discrepancy = np.abs(teacher_recon - student_recon)
                ax.fill_between(range(len(discrepancy)), discrepancy, alpha=0.6, color=VIS_COLORS['student'])
                ax.plot(discrepancy, color=VIS_COLORS['student_dark'], lw=1)

                if len(anomaly_region) > 0:
                    ax.axvspan(anomaly_region[0], anomaly_region[-1], alpha=0.2, color=VIS_COLORS['anomaly_region'])
                ax.axvspan(len(original) - self.config.mask_last_n, len(original), alpha=0.2, color=VIS_COLORS['masked_region'])

                masked_disc = np.mean(discrepancy[-self.config.mask_last_n:])
                ax.set_title(f'Discrepancy (Masked Mean: {masked_disc:.4f})', fontsize=9)
                if col == 0:
                    ax.set_ylabel(f'Late Bloomer #{sample_num+1}\n|T-S|', fontsize=9)

                # Row 3: Reconstruction error profile
                ax = axes[sample_num * 3 + 2, col]
                teacher_err = np.abs(original - teacher_recon)
                student_err = np.abs(original - student_recon)

                ax.plot(teacher_err, 'g-', lw=1.2, alpha=0.8, label='Teacher Error')
                ax.plot(student_err, 'r-', lw=1.2, alpha=0.8, label='Student Error')

                if len(anomaly_region) > 0:
                    ax.axvspan(anomaly_region[0], anomaly_region[-1], alpha=0.2, color=VIS_COLORS['anomaly_region'])
                ax.axvspan(len(original) - self.config.mask_last_n, len(original), alpha=0.2, color=VIS_COLORS['masked_region'])

                if col == 0:
                    ax.set_ylabel(f'Late Bloomer #{sample_num+1}\nRecon Error', fontsize=9)
                if sample_num == n_samples - 1:
                    ax.set_xlabel('Time Step')
                if sample_num == 0 and col == len(show_epochs) - 1:
                    ax.legend(fontsize=7, loc='upper right')

        plt.suptitle('Late Bloomer Case Studies: Samples That Learned to Be Detected\n'
                    '(FN at start → TP at end)\n'
                    'Yellow=Masked Region, Red=Anomaly Region',
                    fontsize=12, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'late_bloomer_case_studies.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - late_bloomer_case_studies.png")

    def generate_all(self):
        """Generate all training progress visualizations"""
        print("\n  Generating Training Progress Visualizations...")

        # First, retrain with checkpoints
        self.retrain_with_checkpoints()

        # Then generate all plots
        self.plot_score_evolution()
        self.plot_sample_trajectories()
        self.plot_metrics_evolution()
        self.plot_late_bloomer_analysis()
        self.plot_late_bloomer_case_studies()  # NEW: Detailed case studies
        self.plot_anomaly_type_learning()
        self.plot_reconstruction_evolution()
        self.plot_decision_boundary_evolution()


# =============================================================================
# Main Function
# =============================================================================

