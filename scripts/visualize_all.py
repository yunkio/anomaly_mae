"""
Unified Visualization Script for Self-Distilled MAE Anomaly Detection

This script generates ALL visualizations for experiments:
1. Data visualizations - Understanding the dataset and anomaly types
2. Architecture visualizations - Model pipeline, patchify modes, self-distillation
3. Stage 1 visualizations - Quick search results analysis
4. Stage 2 visualizations - Full training comparison
5. Best model visualizations - Detailed model analysis

Usage:
    python scripts/visualize_all.py --experiment-dir <path_to_experiment>
    python scripts/visualize_all.py --model-path <path_to_best_model.pt>
    python scripts/visualize_all.py  # Auto-finds latest experiment

Output structure:
    results/experiments/YYYYMMDD_HHMMSS/visualization/
        ├── data/           # DataVisualizer outputs
        ├── architecture/   # ArchitectureVisualizer outputs
        ├── stage1/         # Stage 1 results visualization
        ├── stage2/         # Stage 2 results visualization
        └── best_model/     # Best model analysis
"""
import sys
sys.path.insert(0, '/home/ykio/notebooks/claude')

import os
import json
import glob
from datetime import datetime
from dataclasses import asdict
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve, f1_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from mae_anomaly import (
    Config, SelfDistilledMAEMultivariate, set_seed,
    SlidingWindowTimeSeriesGenerator, SlidingWindowDataset,
    SLIDING_ANOMALY_TYPE_NAMES, ANOMALY_TYPE_NAMES
)


# =============================================================================
# Utility Functions
# =============================================================================

def setup_style():
    """Setup matplotlib style"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['figure.dpi'] = 150


def find_latest_experiment(base_dir: str = 'results/experiments') -> Optional[str]:
    """Find the most recent experiment directory"""
    if not os.path.exists(base_dir):
        return None

    # Find directories with timestamp pattern
    exp_dirs = []
    for d in os.listdir(base_dir):
        full_path = os.path.join(base_dir, d)
        if os.path.isdir(full_path):
            # Check if it has required files
            if os.path.exists(os.path.join(full_path, 'quick_search_results.csv')) or \
               os.path.exists(os.path.join(full_path, 'best_model.pt')):
                exp_dirs.append(full_path)

    if not exp_dirs:
        return None

    # Sort by modification time
    exp_dirs.sort(key=os.path.getmtime, reverse=True)
    return exp_dirs[0]


def load_experiment_data(experiment_dir: str) -> Dict:
    """Load all experiment data from directory"""
    data = {
        'experiment_dir': experiment_dir,
        'quick_results': None,
        'full_results': None,
        'histories': None,
        'metadata': None,
        'best_config': None,
        'model_path': None,
    }

    # Load CSV results
    quick_path = os.path.join(experiment_dir, 'quick_search_results.csv')
    if os.path.exists(quick_path):
        data['quick_results'] = pd.read_csv(quick_path)
        print(f"  Loaded quick_search_results.csv: {len(data['quick_results'])} rows")

    full_path = os.path.join(experiment_dir, 'full_search_results.csv')
    if os.path.exists(full_path):
        data['full_results'] = pd.read_csv(full_path)
        print(f"  Loaded full_search_results.csv: {len(data['full_results'])} rows")

    # Load training histories
    hist_path = os.path.join(experiment_dir, 'training_histories.json')
    if os.path.exists(hist_path):
        with open(hist_path, 'r') as f:
            data['histories'] = json.load(f)
        print(f"  Loaded training_histories.json: {len(data['histories'])} experiments")

    # Load metadata
    meta_path = os.path.join(experiment_dir, 'experiment_metadata.json')
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            data['metadata'] = json.load(f)
        print(f"  Loaded experiment_metadata.json")

    # Load best config
    config_path = os.path.join(experiment_dir, 'best_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            data['best_config'] = json.load(f)
        print(f"  Loaded best_config.json")

    # Check for best model
    model_path = os.path.join(experiment_dir, 'best_model.pt')
    if os.path.exists(model_path):
        data['model_path'] = model_path
        print(f"  Found best_model.pt")

    return data


def load_best_model(model_path: str, num_test: int = 2000) -> Tuple:
    """Load saved best model and create test dataloader

    Uses SlidingWindowDataset for consistency with run_experiments.py
    """
    print(f"Loading model from: {model_path}")

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    # Reconstruct config
    config = Config()
    saved_config = checkpoint['config']
    for key, value in saved_config.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Create model
    model = SelfDistilledMAEMultivariate(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config.device)
    model.eval()

    # Print info
    metrics = checkpoint.get('metrics', {})
    print(f"  Config: margin={config.margin}, lambda_disc={config.lambda_disc}, "
          f"margin_type={getattr(config, 'margin_type', 'hinge')}")
    print(f"  Metrics: ROC-AUC={metrics.get('roc_auc', 0):.4f}, F1={metrics.get('f1_score', 0):.4f}")

    # Generate sliding window dataset
    set_seed(config.random_seed)
    generator = SlidingWindowTimeSeriesGenerator(
        total_length=config.sliding_window_total_length,
        num_features=config.num_features,
        interval_scale=config.anomaly_interval_scale,
        seed=config.random_seed
    )
    signals, point_labels, anomaly_regions = generator.generate()

    # Create test dataset
    test_dataset = SlidingWindowDataset(
        signals=signals,
        point_labels=point_labels,
        anomaly_regions=anomaly_regions,
        window_size=config.seq_length,
        stride=config.sliding_window_stride,
        mask_last_n=config.mask_last_n,
        split='test',
        train_ratio=0.5,
        target_counts={
            'pure_normal': config.test_target_pure_normal,
            'disturbing_normal': config.test_target_disturbing_normal,
            'anomaly': config.test_target_anomaly
        },
        seed=config.random_seed
    )
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    print(f"  Test dataset: {len(test_dataset)} samples")

    return model, config, test_loader, metrics


def collect_predictions(model, dataloader, config) -> Dict:
    """Collect model predictions and scores

    Uses the same evaluation method as evaluator.py:
    - Mask only last mask_last_n positions (not random masking)
    - Use MSE (squared error) for consistency with training
    """
    model.eval()
    device = config.device

    all_scores = []
    all_labels = []
    all_recon_errors = []
    all_discrepancies = []
    all_sample_types = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting predictions"):
            sequences, last_patch_labels, point_labels, sample_types, _ = batch
            sequences = sequences.to(device)
            batch_size, seq_length, num_features = sequences.shape

            # Create mask for last n positions (same as evaluator.py)
            mask = torch.ones(batch_size, seq_length, device=device)
            mask[:, -config.mask_last_n:] = 0

            # Forward pass with fixed mask (masking_ratio=0.0 means use provided mask)
            teacher_output, student_output, _ = model(sequences, masking_ratio=0.0, mask=mask)

            # Compute errors using MSE (same as evaluator.py)
            recon_error = ((teacher_output - sequences) ** 2).mean(dim=2)
            discrepancy = ((teacher_output - student_output) ** 2).mean(dim=2)

            # Compute scores on masked positions only
            masked_positions = (mask == 0)
            recon = (recon_error * masked_positions).sum(dim=1) / (masked_positions.sum(dim=1) + 1e-8)
            disc = (discrepancy * masked_positions).sum(dim=1) / (masked_positions.sum(dim=1) + 1e-8)
            scores = recon + config.lambda_disc * disc

            all_scores.append(scores.cpu().numpy())
            all_labels.append(last_patch_labels.numpy())
            all_recon_errors.append(recon.cpu().numpy())
            all_discrepancies.append(disc.cpu().numpy())
            all_sample_types.append(sample_types.numpy())

    return {
        'scores': np.concatenate(all_scores),
        'labels': np.concatenate(all_labels),
        'recon_errors': np.concatenate(all_recon_errors),
        'discrepancies': np.concatenate(all_discrepancies),
        'sample_types': np.concatenate(all_sample_types)
    }


def collect_detailed_data(model, dataloader, config) -> Dict:
    """Collect detailed data for analysis

    Uses the same evaluation method as evaluator.py:
    - Mask only last mask_last_n positions (not random masking)
    - Use MSE (squared error) for consistency with training
    """
    model.eval()
    device = config.device

    all_teacher_errors = []
    all_student_errors = []
    all_discrepancies = []
    all_masks = []
    all_labels = []
    all_point_labels = []
    all_originals = []
    all_teacher_recons = []
    all_student_recons = []
    all_sample_types = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting detailed data"):
            sequences, last_patch_labels, point_labels, sample_types, _ = batch
            sequences = sequences.to(device)
            batch_size, seq_length, num_features = sequences.shape

            # Create mask for last n positions (same as evaluator.py)
            mask = torch.ones(batch_size, seq_length, device=device)
            mask[:, -config.mask_last_n:] = 0

            # Forward pass with fixed mask
            teacher_output, student_output, _ = model(sequences, masking_ratio=0.0, mask=mask)

            # Compute errors using MSE (same as evaluator.py)
            teacher_error = ((teacher_output - sequences) ** 2).mean(dim=-1)
            student_error = ((student_output - sequences) ** 2).mean(dim=-1)
            discrepancy = ((teacher_output - student_output) ** 2).mean(dim=-1)

            all_teacher_errors.append(teacher_error.cpu().numpy())
            all_student_errors.append(student_error.cpu().numpy())
            all_discrepancies.append(discrepancy.cpu().numpy())
            all_masks.append(mask.cpu().numpy())
            all_labels.append(last_patch_labels.numpy())
            all_point_labels.append(point_labels.numpy())
            all_originals.append(sequences.cpu().numpy())
            all_teacher_recons.append(teacher_output.cpu().numpy())
            all_student_recons.append(student_output.cpu().numpy())
            all_sample_types.append(sample_types.numpy())

    return {
        'teacher_errors': np.concatenate(all_teacher_errors),
        'student_errors': np.concatenate(all_student_errors),
        'discrepancies': np.concatenate(all_discrepancies),
        'masks': np.concatenate(all_masks),
        'labels': np.concatenate(all_labels),
        'point_labels': np.concatenate(all_point_labels),
        'originals': np.concatenate(all_originals),
        'teacher_recons': np.concatenate(all_teacher_recons),
        'student_recons': np.concatenate(all_student_recons),
        'sample_types': np.concatenate(all_sample_types),
    }


# =============================================================================
# DataVisualizer - Dataset and Anomaly Type Visualizations
# =============================================================================

class DataVisualizer:
    """Visualize dataset characteristics and anomaly types"""

    def __init__(self, output_dir: str, config: Config = None):
        self.output_dir = output_dir
        self.config = config or Config()
        os.makedirs(output_dir, exist_ok=True)

    def plot_anomaly_types(self):
        """Visualize different anomaly types in the dataset"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Generate example sequences for each anomaly type
        np.random.seed(42)
        seq_length = 100
        t = np.linspace(0, 4*np.pi, seq_length)

        # 1. Normal - clean sinusoid
        normal = np.sin(t) + 0.1 * np.random.randn(seq_length)
        ax = axes[0, 0]
        ax.plot(t, normal, 'b-', lw=1.5)
        ax.set_title('Normal Pattern', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_ylim(-3, 3)

        # 2. Point anomaly - spike
        point_anomaly = normal.copy()
        point_anomaly[70:75] = 2.5
        ax = axes[0, 1]
        ax.plot(t, point_anomaly, 'b-', lw=1.5)
        ax.axvspan(t[70], t[74], alpha=0.3, color='red')
        ax.scatter(t[70:75], point_anomaly[70:75], c='red', s=50, zorder=5)
        ax.set_title('Point Anomaly (Spike)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_ylim(-3, 3)

        # 3. Contextual anomaly - level shift
        contextual = normal.copy()
        contextual[60:] += 1.0
        ax = axes[0, 2]
        ax.plot(t, contextual, 'b-', lw=1.5)
        ax.axvspan(t[60], t[-1], alpha=0.3, color='red')
        ax.set_title('Contextual Anomaly (Level Shift)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_ylim(-3, 3)

        # 4. Collective anomaly - unusual pattern
        collective = normal.copy()
        collective[50:80] = 0.5 * np.sin(3*t[50:80]) + np.sin(10*t[50:80]) * 0.3
        ax = axes[1, 0]
        ax.plot(t, collective, 'b-', lw=1.5)
        ax.axvspan(t[50], t[79], alpha=0.3, color='red')
        ax.set_title('Collective Anomaly (Unusual Pattern)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_ylim(-3, 3)

        # 5. Frequency anomaly
        freq_anomaly = normal.copy()
        freq_anomaly[40:70] = np.sin(4*t[40:70]) + 0.1 * np.random.randn(30)
        ax = axes[1, 1]
        ax.plot(t, freq_anomaly, 'b-', lw=1.5)
        ax.axvspan(t[40], t[69], alpha=0.3, color='red')
        ax.set_title('Frequency Anomaly', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_ylim(-3, 3)

        # 6. Trend anomaly
        trend_anomaly = normal.copy()
        trend_anomaly[50:] += np.linspace(0, 1.5, 50)
        ax = axes[1, 2]
        ax.plot(t, trend_anomaly, 'b-', lw=1.5)
        ax.axvspan(t[50], t[-1], alpha=0.3, color='red')
        ax.set_title('Trend Anomaly (Drift)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_ylim(-3, 3)

        plt.suptitle('Types of Anomalies in Time Series Data', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'anomaly_types.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - anomaly_types.png")

    def plot_sample_types(self):
        """Visualize different sample types (normal, disturbing normal, anomaly)"""
        # Create sample dataset using sliding window
        set_seed(42)
        generator = SlidingWindowTimeSeriesGenerator(
            total_length=50000,  # Small dataset for visualization
            num_features=1,
            interval_scale=1.5,
            seed=42
        )
        signals, point_labels, anomaly_regions = generator.generate()
        dataset = SlidingWindowDataset(
            signals=signals,
            point_labels=point_labels,
            anomaly_regions=anomaly_regions,
            window_size=self.config.seq_length,
            stride=10,
            mask_last_n=10,
            split='test',
            train_ratio=0.5,
            seed=42
        )

        # Collect samples by type (with point labels for disturbing normal)
        normal_samples = []
        disturbing_samples = []  # (seq, point_labels)
        anomaly_samples = []

        for i in range(len(dataset)):
            seq, label, point_labels, sample_type, _ = dataset[i]
            if sample_type == 0:  # pure normal
                normal_samples.append(seq[:, 0].numpy())
            elif sample_type == 1:  # disturbing normal
                disturbing_samples.append((seq[:, 0].numpy(), point_labels.numpy()))
            else:  # anomaly
                anomaly_samples.append(seq[:, 0].numpy())

            if len(normal_samples) >= 5 and len(disturbing_samples) >= 5 and len(anomaly_samples) >= 5:
                break

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        x = np.arange(self.config.seq_length)

        # Normal samples
        ax = axes[0]
        for i, seq in enumerate(normal_samples[:5]):
            ax.plot(x, seq, alpha=0.7, label=f'Sample {i+1}')
        ax.set_title('Pure Normal Samples', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.axvspan(x[-10], x[-1], alpha=0.2, color='yellow', label='Last Patch')
        ax.legend(fontsize=8)

        # Disturbing normal samples - show anomaly regions
        ax = axes[1]
        for i, (seq, point_labels) in enumerate(disturbing_samples[:5]):
            line, = ax.plot(x, seq, alpha=0.7, label=f'Sample {i+1}')
            # Find and highlight anomaly regions in this sample
            anomaly_mask = point_labels > 0
            if anomaly_mask.any():
                # Find contiguous anomaly regions
                diff = np.diff(np.concatenate([[0], anomaly_mask.astype(int), [0]]))
                starts = np.where(diff == 1)[0]
                ends = np.where(diff == -1)[0]
                for start, end in zip(starts, ends):
                    ax.axvspan(start, end - 1, alpha=0.15, color=line.get_color())
        ax.set_title('Disturbing Normal Samples\n(Anomaly in window, but last patch normal)',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.axvspan(x[-10], x[-1], alpha=0.2, color='green', label='Normal Last Patch')
        ax.legend(fontsize=8)

        # Anomaly samples
        ax = axes[2]
        for i, seq in enumerate(anomaly_samples[:5]):
            ax.plot(x, seq, alpha=0.7, label=f'Sample {i+1}')
        ax.set_title('Anomaly Samples\n(Anomaly in last patch)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.axvspan(x[-10], x[-1], alpha=0.2, color='red', label='Anomaly Last Patch')
        ax.legend(fontsize=8)

        plt.suptitle('Sample Types in Dataset', fontsize=14, fontweight='bold', y=1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'sample_types.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - sample_types.png")

    def plot_feature_examples(self):
        """Visualize multivariate feature examples"""
        set_seed(42)
        generator = SlidingWindowTimeSeriesGenerator(
            total_length=50000,
            num_features=self.config.num_features,
            interval_scale=1.5,
            seed=42
        )
        signals, point_labels, anomaly_regions = generator.generate()
        dataset = SlidingWindowDataset(
            signals=signals,
            point_labels=point_labels,
            anomaly_regions=anomaly_regions,
            window_size=self.config.seq_length,
            stride=10,
            mask_last_n=10,
            split='test',
            train_ratio=0.5,
            seed=42
        )

        # Get one normal and one anomaly sample
        normal_seq = None
        anomaly_seq = None
        for i in range(len(dataset)):
            seq, label, _, _, _ = dataset[i]
            if label == 0 and normal_seq is None:
                normal_seq = seq.numpy()
            elif label == 1 and anomaly_seq is None:
                anomaly_seq = seq.numpy()
            if normal_seq is not None and anomaly_seq is not None:
                break

        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        x = np.arange(self.config.seq_length)

        # Normal sample features
        ax = axes[0]
        for f in range(min(5, self.config.num_features)):
            ax.plot(x, normal_seq[:, f], alpha=0.8, label=f'Feature {f+1}')
        ax.axvspan(x[-10], x[-1], alpha=0.2, color='yellow')
        ax.set_title('Normal Sample - Multiple Features', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.legend(loc='upper right')

        # Anomaly sample features
        ax = axes[1]
        for f in range(min(5, self.config.num_features)):
            ax.plot(x, anomaly_seq[:, f], alpha=0.8, label=f'Feature {f+1}')
        ax.axvspan(x[-10], x[-1], alpha=0.2, color='red')
        ax.set_title('Anomaly Sample - Multiple Features', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'feature_examples.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - feature_examples.png")

    def plot_dataset_statistics(self):
        """Visualize dataset statistics"""
        set_seed(42)
        generator = SlidingWindowTimeSeriesGenerator(
            total_length=200000,  # Larger for better statistics
            num_features=self.config.num_features,
            interval_scale=1.5,
            seed=42
        )
        signals, point_labels, anomaly_regions = generator.generate()
        dataset = SlidingWindowDataset(
            signals=signals,
            point_labels=point_labels,
            anomaly_regions=anomaly_regions,
            window_size=self.config.seq_length,
            stride=10,
            mask_last_n=10,
            split='test',
            train_ratio=0.5,
            target_counts={'pure_normal': 600, 'disturbing_normal': 150, 'anomaly': 250},
            seed=42
        )

        labels = []
        sample_types = []

        for i in range(len(dataset)):
            _, label, _, st, _ = dataset[i]
            labels.append(label)
            sample_types.append(st)

        labels = np.array(labels)
        sample_types = np.array(sample_types)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Label distribution
        ax = axes[0]
        label_counts = [np.sum(labels == 0), np.sum(labels == 1)]
        bars = ax.bar(['Normal', 'Anomaly'], label_counts, color=['#3498DB', '#E74C3C'])
        ax.set_title('Label Distribution', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count')
        for bar, count in zip(bars, label_counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                   f'{count}\n({count/len(labels)*100:.1f}%)', ha='center', va='bottom')

        # Sample type distribution
        ax = axes[1]
        type_counts = [np.sum(sample_types == 0), np.sum(sample_types == 1), np.sum(sample_types == 2)]
        type_labels = ['Pure Normal', 'Disturbing Normal', 'Anomaly']
        colors = ['#3498DB', '#F39C12', '#E74C3C']
        bars = ax.bar(type_labels, type_counts, color=colors)
        ax.set_title('Sample Type Distribution', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=15)
        for bar, count in zip(bars, type_counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                   f'{count}', ha='center', va='bottom')

        # Pie chart
        ax = axes[2]
        ax.pie(type_counts, labels=type_labels, colors=colors, autopct='%1.1f%%',
               startangle=90, explode=[0, 0.05, 0])
        ax.set_title('Sample Type Proportions', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'dataset_statistics.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - dataset_statistics.png")

    def plot_anomaly_generation_rules(self):
        """Visualize the rules for generating each anomaly type"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        np.random.seed(42)

        seq_length = self.config.seq_length
        t = np.arange(seq_length)
        mask_last_n = getattr(self.config, 'mask_last_n', 5)

        # Base normal signal
        base_signal = 0.5 + 0.3 * np.sin(t * 0.1) + 0.05 * np.random.randn(seq_length)

        anomaly_info = [
            ('Spike (DDoS Attack)', 'spike',
             'CPU + Network + ResponseTime spike\nWidth: 3-10 timesteps\nIntensity: +0.3~0.5',
             lambda s: self._add_spike(s.copy(), seq_length, mask_last_n)),
            ('Memory Leak', 'memory_leak',
             'Memory gradual increase\nDisk I/O increases (swapping)\nContinues to end',
             lambda s: self._add_memory_leak(s.copy(), seq_length)),
            ('Noise Burst', 'noise',
             'All features: random noise\nNoise std: 0.2\nLength: 20-40 timesteps',
             lambda s: self._add_noise(s.copy(), seq_length, mask_last_n)),
            ('Drift', 'drift',
             'CPU + ResponseTime gradual increase\nLinear drift: +0.2~0.4\nLength: 30-50 timesteps',
             lambda s: self._add_drift(s.copy(), seq_length, mask_last_n)),
            ('Network Congestion', 'network_congestion',
             'Network + ResponseTime spike\nPersists to end\nIntensity: +0.3~0.5',
             lambda s: self._add_network_congestion(s.copy(), seq_length)),
            ('Disturbing Normal', 'disturbing',
             'Anomaly OUTSIDE last patch\nLast patch remains normal\nTests model robustness',
             lambda s: self._add_disturbing(s.copy(), seq_length, mask_last_n))
        ]

        for idx, (title, atype, description, func) in enumerate(anomaly_info):
            ax = axes[idx // 3, idx % 3]

            # Generate anomaly signal and get anomaly region
            signal, anomaly_start, anomaly_end = func(base_signal.copy())

            # Plot
            ax.plot(t, base_signal, 'b-', alpha=0.3, lw=1, label='Original')
            ax.plot(t, signal, 'r-', lw=2, label='With Anomaly')

            # Highlight last patch
            ax.axvspan(seq_length - mask_last_n, seq_length, alpha=0.2, color='yellow',
                      label=f'Last Patch (n={mask_last_n})')

            # Highlight anomaly region
            if anomaly_start is not None:
                ax.axvspan(anomaly_start, anomaly_end, alpha=0.3, color='red', label='Anomaly Region')

            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Value')
            ax.legend(fontsize=8, loc='upper left')

            # Add description box
            ax.text(0.98, 0.02, description, transform=ax.transAxes, fontsize=8,
                   verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.suptitle('Anomaly Generation Rules\n(All anomalies must overlap with last patch except Disturbing Normal)',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'anomaly_generation_rules.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - anomaly_generation_rules.png")

    def _add_spike(self, signal, seq_length, mask_last_n):
        """Add spike anomaly"""
        spike_width = 7
        spike_pos = seq_length - mask_last_n - spike_width // 2
        signal[spike_pos:spike_pos + spike_width] += 0.4
        return signal, spike_pos, spike_pos + spike_width

    def _add_memory_leak(self, signal, seq_length):
        """Add memory leak anomaly"""
        start_pos = seq_length // 3
        leak = np.linspace(0, 0.4, seq_length - start_pos)
        signal[start_pos:] += leak
        return signal, start_pos, seq_length

    def _add_noise(self, signal, seq_length, mask_last_n):
        """Add noise anomaly"""
        noise_length = 25
        noise_start = max(0, seq_length - mask_last_n - noise_length // 2)
        actual_end = min(noise_start + noise_length, seq_length)
        actual_length = actual_end - noise_start
        signal[noise_start:actual_end] += np.random.normal(0, 0.2, actual_length)
        return signal, noise_start, actual_end

    def _add_drift(self, signal, seq_length, mask_last_n):
        """Add drift anomaly"""
        drift_length = 40
        drift_start = max(0, seq_length - mask_last_n - drift_length // 2)
        actual_end = min(drift_start + drift_length, seq_length)
        actual_length = actual_end - drift_start
        drift = np.linspace(0, 0.3, actual_length)
        signal[drift_start:actual_end] += drift
        return signal, drift_start, actual_end

    def _add_network_congestion(self, signal, seq_length):
        """Add network congestion anomaly"""
        change_point = seq_length // 2
        signal[change_point:] += 0.35
        return signal, change_point, seq_length

    def _add_disturbing(self, signal, seq_length, mask_last_n):
        """Add disturbing normal (anomaly outside last patch)"""
        spike_width = 7
        spike_pos = seq_length // 3
        signal[spike_pos:spike_pos + spike_width] += 0.4
        return signal, spike_pos, spike_pos + spike_width

    def plot_feature_correlations(self):
        """Visualize correlations between features"""
        set_seed(42)
        generator = SlidingWindowTimeSeriesGenerator(
            total_length=100000,
            num_features=self.config.num_features,
            interval_scale=1.5,
            seed=42
        )
        signals, point_labels, anomaly_regions = generator.generate()
        dataset = SlidingWindowDataset(
            signals=signals,
            point_labels=point_labels,
            anomaly_regions=anomaly_regions,
            window_size=self.config.seq_length,
            stride=10,
            mask_last_n=10,
            split='test',
            train_ratio=0.5,
            seed=42
        )

        # Collect all data
        all_data = []
        for i in range(len(dataset)):
            seq, _, _, _, _ = dataset[i]
            all_data.append(seq.numpy())
        all_data = np.array(all_data)  # (N, T, F)

        # Flatten to (N*T, F) for correlation
        flat_data = all_data.reshape(-1, self.config.num_features)

        feature_names = ['CPU', 'Memory', 'DiskIO', 'Network', 'ResponseTime'][:self.config.num_features]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1. Correlation matrix
        ax = axes[0]
        corr_matrix = np.corrcoef(flat_data.T)
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=ax,
                   xticklabels=feature_names, yticklabels=feature_names,
                   vmin=-1, vmax=1, center=0)
        ax.set_title('Feature Correlation Matrix', fontsize=12, fontweight='bold')

        # 2. Pairwise scatter (CPU vs Memory)
        ax = axes[1]
        sample_idx = np.random.choice(len(flat_data), min(5000, len(flat_data)), replace=False)
        ax.scatter(flat_data[sample_idx, 0], flat_data[sample_idx, 1], alpha=0.3, s=5)
        ax.set_xlabel('CPU')
        ax.set_ylabel('Memory')
        ax.set_title(f'CPU vs Memory (corr={corr_matrix[0, 1]:.3f})', fontsize=12, fontweight='bold')

        # 3. Feature description
        ax = axes[2]
        ax.axis('off')

        desc_text = """
Feature Generation Rules (Server Monitoring Simulation)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CPU (Feature 0):
  • Base pattern: sinusoidal + noise
  • Formula: base + amp * sin(freq * t) + noise(0, 0.05)

Memory (Feature 1):
  • Correlated with CPU (slower variation)
  • Formula: base + 0.3 * CPU + 0.2 * sin(0.5 * freq * t) + noise

DiskIO (Feature 2):
  • Correlated with Memory (spiky, Poisson spikes)
  • Formula: base + 0.2 * Memory + poisson_spikes + noise

Network (Feature 3):
  • Bursty traffic pattern
  • Formula: amp * |sin(freq * t)| + noise

ResponseTime (Feature 4):
  • Correlated with CPU and Network
  • Formula: base + 0.3 * CPU + 0.2 * Network + noise
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """
        ax.text(0.05, 0.95, desc_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        plt.suptitle('Feature Correlations and Generation Rules', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'feature_correlations.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - feature_correlations.png")

    def plot_experiment_settings(self):
        """Visualize experiment settings for reproducibility"""
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.axis('off')

        settings_text = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         EXPERIMENT SETTINGS                                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Data Configuration                                                           ║
║    • Sequence Length: {self.config.seq_length:<10}                                            ║
║    • Number of Features: {self.config.num_features:<7}                                            ║
║    • Train Anomaly Ratio: {self.config.train_anomaly_ratio:<6}                                          ║
║    • Test Anomaly Ratio: {self.config.test_anomaly_ratio:<7}                                          ║
║    • Disturbing Ratio: 0.2 (20% of normal samples)                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Stage 1 (Quick Search)                                                       ║
║    • Epochs: 15                                                               ║
║    • Train Samples: 1000                                                      ║
║    • Test Samples: 400                                                        ║
║    • Purpose: Rapid hyperparameter screening                                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Stage 2 (Full Training)                                                      ║
║    • Epochs: 100                                                              ║
║    • Train Samples: 2000                                                      ║
║    • Test Samples: 500                                                        ║
║    • Top-k from Stage 1: 150                                                  ║
║    • Purpose: Full training of promising configurations                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Random Seeds                                                                 ║
║    • Train Dataset: seed=42                                                   ║
║    • Test Dataset: seed=43                                                    ║
║    • Ensures identical data across all experiments                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Model Configuration                                                          ║
║    • Batch Size: {self.config.batch_size:<11}                                             ║
║    • Learning Rate: {self.config.learning_rate:<8}                                             ║
║    • Model Dimension (d_model): {self.config.d_model:<5}                                        ║
║    • Encoder Layers: {self.config.num_encoder_layers:<7}                                             ║
║    • Attention Heads (nhead): {self.config.nhead:<6}                                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """

        ax.text(0.05, 0.95, settings_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))

        plt.savefig(os.path.join(self.output_dir, 'experiment_settings.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - experiment_settings.png")

    def generate_all(self):
        """Generate all data visualizations"""
        print("\n  Generating Data Visualizations...")
        self.plot_anomaly_types()
        self.plot_sample_types()
        self.plot_feature_examples()
        self.plot_dataset_statistics()
        self.plot_anomaly_generation_rules()
        self.plot_feature_correlations()
        self.plot_experiment_settings()


# =============================================================================
# ArchitectureVisualizer - Model Architecture Visualizations
# =============================================================================

class ArchitectureVisualizer:
    """Visualize model architecture and concepts"""

    def __init__(self, output_dir: str, config: Config = None):
        self.output_dir = output_dir
        self.config = config or Config()
        os.makedirs(output_dir, exist_ok=True)

    def plot_model_pipeline(self):
        """Visualize the overall model pipeline"""
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 8)
        ax.axis('off')

        # Define colors
        colors = {
            'input': '#3498DB',
            'encoder': '#27AE60',
            'teacher': '#E74C3C',
            'student': '#9B59B6',
            'loss': '#F39C12',
            'mask': '#95A5A6'
        }

        # Input
        rect = mpatches.FancyBboxPatch((0.5, 3), 2, 2, boxstyle="round,pad=0.05",
                                        facecolor=colors['input'], edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(1.5, 4, 'Input\nSequence\n(T, F)', ha='center', va='center', fontsize=10, fontweight='bold', color='white')

        # Patchify + Embed
        rect = mpatches.FancyBboxPatch((3.5, 3), 2, 2, boxstyle="round,pad=0.05",
                                        facecolor=colors['mask'], edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(4.5, 4, 'Patchify\n+\nEmbed', ha='center', va='center', fontsize=10, fontweight='bold')

        # Masking
        rect = mpatches.FancyBboxPatch((6.5, 3), 2, 2, boxstyle="round,pad=0.05",
                                        facecolor=colors['mask'], edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(7.5, 4, 'Random\nMasking', ha='center', va='center', fontsize=10, fontweight='bold')

        # Encoder
        rect = mpatches.FancyBboxPatch((9.5, 3), 2, 2, boxstyle="round,pad=0.05",
                                        facecolor=colors['encoder'], edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(10.5, 4, 'Shared\nEncoder', ha='center', va='center', fontsize=10, fontweight='bold', color='white')

        # Teacher Decoder (top)
        rect = mpatches.FancyBboxPatch((12.5, 5.5), 2, 2, boxstyle="round,pad=0.05",
                                        facecolor=colors['teacher'], edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(13.5, 6.5, 'Teacher\nDecoder', ha='center', va='center', fontsize=10, fontweight='bold', color='white')

        # Student Decoder (bottom)
        rect = mpatches.FancyBboxPatch((12.5, 0.5), 2, 2, boxstyle="round,pad=0.05",
                                        facecolor=colors['student'], edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(13.5, 1.5, 'Student\nDecoder', ha='center', va='center', fontsize=10, fontweight='bold', color='white')

        # Arrows
        arrow_style = dict(arrowstyle='->', color='black', lw=2)
        ax.annotate('', xy=(3.4, 4), xytext=(2.6, 4), arrowprops=arrow_style)
        ax.annotate('', xy=(6.4, 4), xytext=(5.6, 4), arrowprops=arrow_style)
        ax.annotate('', xy=(9.4, 4), xytext=(8.6, 4), arrowprops=arrow_style)

        # Split to teacher/student
        ax.annotate('', xy=(12.4, 6.5), xytext=(11.6, 4.5), arrowprops=arrow_style)
        ax.annotate('', xy=(12.4, 1.5), xytext=(11.6, 3.5), arrowprops=arrow_style)

        # Loss box
        rect = mpatches.FancyBboxPatch((15, 3), 1, 2, boxstyle="round,pad=0.05",
                                        facecolor=colors['loss'], edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(15.5, 4, 'Loss', ha='center', va='center', fontsize=10, fontweight='bold', color='white')

        # Arrows to loss
        ax.annotate('', xy=(14.9, 4.5), xytext=(14.6, 5.5), arrowprops=arrow_style)
        ax.annotate('', xy=(14.9, 3.5), xytext=(14.6, 2.5), arrowprops=arrow_style)

        # Labels
        ax.text(15.5, 5.3, 'Recon', ha='center', fontsize=8)
        ax.text(15.5, 2.7, 'Disc', ha='center', fontsize=8)

        # Title
        ax.set_title('Self-Distilled MAE Pipeline', fontsize=14, fontweight='bold', y=1.02)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'model_pipeline.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - model_pipeline.png")

    def plot_patchify_modes(self):
        """Visualize different patchify modes"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Sample data
        np.random.seed(42)
        seq_length = 100
        t = np.arange(seq_length)
        signal = np.sin(t * 0.1) + 0.5 * np.sin(t * 0.3) + 0.1 * np.random.randn(seq_length)

        num_patches = 10
        patch_size = seq_length // num_patches

        titles = ['CNN-First Mode', 'Patch-CNN Mode', 'Linear Mode']
        descriptions = [
            'CNN on full sequence\n(potential information leakage)',
            'Patchify first, then CNN per patch\n(no cross-patch leakage)',
            'Patchify then linear embedding\n(original MAE style)'
        ]
        colors = ['#4682B4', '#CD5C5C', '#228B22']

        for idx, (ax, title, desc, color) in enumerate(zip(axes, titles, descriptions, colors)):
            ax.plot(t, signal, 'b-', alpha=0.5, lw=1)

            # Draw patches
            for p in range(num_patches):
                start = p * patch_size
                end = (p + 1) * patch_size
                ax.axvspan(start, end, alpha=0.2 if p % 2 == 0 else 0.1, color=color)
                ax.axvline(x=start, color='gray', linestyle='--', alpha=0.5)

            ax.set_title(f'{title}\n{desc}', fontsize=11, fontweight='bold')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Value')

            # Add processing indicator
            if idx == 0:
                ax.annotate('', xy=(80, 1.5), xytext=(20, 1.5),
                           arrowprops=dict(arrowstyle='<->', color='red', lw=2))
                ax.text(50, 1.7, 'CNN sees all', ha='center', fontsize=9, color='red')
            elif idx == 1:
                ax.annotate('', xy=(patch_size-1, 1.5), xytext=(0, 1.5),
                           arrowprops=dict(arrowstyle='<->', color='green', lw=2))
                ax.text(patch_size/2, 1.7, 'CNN per patch', ha='center', fontsize=9, color='green')

        plt.suptitle('Patchify Mode Comparison', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'patchify_modes.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - patchify_modes.png")

    def plot_masking_visualization(self):
        """Visualize the masking process"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        np.random.seed(42)
        seq_length = 100
        t = np.arange(seq_length)
        signal = np.sin(t * 0.1) + 0.3 * np.cos(t * 0.2)

        num_patches = 10
        patch_size = seq_length // num_patches

        # 1. Original sequence
        ax = axes[0, 0]
        ax.plot(t, signal, 'b-', lw=2)
        ax.set_title('1. Original Sequence', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')

        # 2. Patchified
        ax = axes[0, 1]
        colors = plt.cm.tab10(np.arange(num_patches))
        for p in range(num_patches):
            start = p * patch_size
            end = (p + 1) * patch_size
            ax.plot(t[start:end], signal[start:end], color=colors[p], lw=2)
            ax.axvline(x=start, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(f'2. Patchified ({num_patches} patches)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')

        # 3. Masked
        ax = axes[1, 0]
        mask_ratio = 0.5
        n_masked = int(num_patches * mask_ratio)
        masked_patches = np.random.choice(num_patches, n_masked, replace=False)

        for p in range(num_patches):
            start = p * patch_size
            end = (p + 1) * patch_size
            if p in masked_patches:
                ax.axvspan(start, end, alpha=0.3, color='red')
                ax.plot(t[start:end], signal[start:end], 'r--', lw=1, alpha=0.5)
            else:
                ax.plot(t[start:end], signal[start:end], 'b-', lw=2)
            ax.axvline(x=start, color='gray', linestyle='--', alpha=0.5)

        ax.set_title(f'3. Masked ({mask_ratio*100:.0f}% masking ratio)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')

        # Legend
        patches = [mpatches.Patch(color='blue', label='Visible'),
                   mpatches.Patch(color='red', alpha=0.3, label='Masked')]
        ax.legend(handles=patches)

        # 4. Reconstruction target
        ax = axes[1, 1]
        for p in range(num_patches):
            start = p * patch_size
            end = (p + 1) * patch_size
            if p in masked_patches:
                ax.axvspan(start, end, alpha=0.3, color='red')
                ax.plot(t[start:end], signal[start:end], 'g-', lw=2, label='Target' if p == masked_patches[0] else '')
            else:
                ax.plot(t[start:end], signal[start:end], 'b-', lw=2, alpha=0.3)
            ax.axvline(x=start, color='gray', linestyle='--', alpha=0.5)

        ax.set_title('4. Reconstruction Target (masked regions)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')

        plt.suptitle('MAE Masking Process', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'masking_visualization.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - masking_visualization.png")

    def plot_self_distillation_concept(self):
        """Visualize the self-distillation concept"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: Normal sample - teacher and student similar
        ax = axes[0]
        np.random.seed(42)
        t = np.arange(100)
        original = np.sin(t * 0.1) + 0.1 * np.random.randn(100)
        teacher_recon = original + 0.05 * np.random.randn(100)
        student_recon = original + 0.08 * np.random.randn(100)

        ax.plot(t, original, 'b-', lw=2, label='Original')
        ax.plot(t, teacher_recon, 'g--', lw=2, label='Teacher', alpha=0.8)
        ax.plot(t, student_recon, 'r:', lw=2, label='Student', alpha=0.8)

        # Highlight small discrepancy
        ax.fill_between(t, teacher_recon, student_recon, alpha=0.2, color='purple')

        ax.set_title('Normal Sample\n(Small Teacher-Student Discrepancy)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.legend()

        # Right: Anomaly sample - teacher and student differ
        ax = axes[1]
        original_anomaly = np.sin(t * 0.1)
        original_anomaly[70:85] = 1.5  # Anomaly
        original_anomaly += 0.1 * np.random.randn(100)

        teacher_recon_a = original_anomaly.copy()
        teacher_recon_a[70:85] = np.sin(t[70:85] * 0.1) + 0.1  # Teacher reconstructs normal pattern

        student_recon_a = original_anomaly.copy()
        student_recon_a[70:85] = np.sin(t[70:85] * 0.1) + 0.5  # Student fails more

        ax.plot(t, original_anomaly, 'b-', lw=2, label='Original (Anomaly)')
        ax.plot(t, teacher_recon_a, 'g--', lw=2, label='Teacher', alpha=0.8)
        ax.plot(t, student_recon_a, 'r:', lw=2, label='Student', alpha=0.8)

        # Highlight large discrepancy
        ax.fill_between(t, teacher_recon_a, student_recon_a, alpha=0.3, color='purple')
        ax.axvspan(70, 85, alpha=0.2, color='red')

        ax.set_title('Anomaly Sample\n(Large Teacher-Student Discrepancy)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.legend()

        plt.suptitle('Self-Distillation for Anomaly Detection\nTeacher (trained) vs Student (weaker decoder)',
                     fontsize=14, fontweight='bold', y=1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'self_distillation_concept.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - self_distillation_concept.png")

    def plot_margin_types(self):
        """Visualize different margin types for discrepancy loss"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        x = np.linspace(-0.5, 2, 200)
        margin = 0.5

        # Hinge loss
        ax = axes[0]
        hinge = np.maximum(0, margin - x)
        ax.plot(x, hinge, 'b-', lw=2)
        ax.axvline(x=margin, color='r', linestyle='--', label=f'margin={margin}')
        ax.fill_between(x, 0, hinge, alpha=0.2)
        ax.set_title('Hinge Loss\nmax(0, m - d)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Discrepancy (d)')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.set_ylim(-0.1, 1.5)
        ax.grid(True, alpha=0.3)

        # Softplus loss
        ax = axes[1]
        softplus = np.log(1 + np.exp(margin - x))
        ax.plot(x, softplus, 'g-', lw=2)
        ax.axvline(x=margin, color='r', linestyle='--', label=f'margin={margin}')
        ax.fill_between(x, 0, softplus, alpha=0.2, color='green')
        ax.set_title('Softplus Loss\nlog(1 + exp(m - d))', fontsize=12, fontweight='bold')
        ax.set_xlabel('Discrepancy (d)')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.set_ylim(-0.1, 1.5)
        ax.grid(True, alpha=0.3)

        # Dynamic margin concept
        ax = axes[2]
        # Show multiple margins
        for m in [0.3, 0.5, 0.7, 1.0]:
            hinge = np.maximum(0, m - x)
            ax.plot(x, hinge, lw=1.5, label=f'm={m}', alpha=0.7)
        ax.set_title('Dynamic Margin\nAdaptive margin based on input', fontsize=12, fontweight='bold')
        ax.set_xlabel('Discrepancy (d)')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.set_ylim(-0.1, 1.5)
        ax.grid(True, alpha=0.3)

        plt.suptitle('Margin Types for Discrepancy Loss', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'margin_types.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - margin_types.png")

    def plot_loss_components(self):
        """Visualize the loss function components"""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')

        # Title
        ax.text(0.5, 0.95, 'Loss Function Components', ha='center', va='top',
                fontsize=16, fontweight='bold', transform=ax.transAxes)

        # Total Loss formula
        ax.text(0.5, 0.85, r'$\mathcal{L}_{total} = \mathcal{L}_{recon} + \lambda \cdot \mathcal{L}_{disc}$',
                ha='center', va='center', fontsize=14, transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Reconstruction Loss
        ax.text(0.25, 0.65, 'Reconstruction Loss', ha='center', va='center',
                fontsize=12, fontweight='bold', transform=ax.transAxes)
        ax.text(0.25, 0.55, r'$\mathcal{L}_{recon} = \frac{1}{|M|} \sum_{i \in M} ||\hat{x}_i^{teacher} - x_i||^2$',
                ha='center', va='center', fontsize=11, transform=ax.transAxes)
        ax.text(0.25, 0.45, 'Teacher reconstructs\nmasked patches',
                ha='center', va='center', fontsize=10, transform=ax.transAxes, style='italic')

        # Discrepancy Loss
        ax.text(0.75, 0.65, 'Discrepancy Loss', ha='center', va='center',
                fontsize=12, fontweight='bold', transform=ax.transAxes)
        ax.text(0.75, 0.55, r'$\mathcal{L}_{disc} = \max(0, m - ||\hat{x}^{teacher} - \hat{x}^{student}||)$',
                ha='center', va='center', fontsize=11, transform=ax.transAxes)
        ax.text(0.75, 0.45, 'Push Teacher-Student\ndiscrepancy above margin',
                ha='center', va='center', fontsize=10, transform=ax.transAxes, style='italic')

        # Intuition box
        rect = mpatches.FancyBboxPatch((0.1, 0.1), 0.8, 0.25, boxstyle="round,pad=0.02",
                                        facecolor='lightblue', edgecolor='black', linewidth=1,
                                        transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(0.5, 0.28, 'Intuition:', ha='center', va='center',
                fontsize=11, fontweight='bold', transform=ax.transAxes)
        ax.text(0.5, 0.2, '- For Normal data: Both Teacher and Student reconstruct well\n'
                         '  -> Small discrepancy (good reconstruction, low disc loss)',
                ha='center', va='center', fontsize=10, transform=ax.transAxes)
        ax.text(0.5, 0.13, '- For Anomaly data: Teacher learns normal patterns better than Student\n'
                         '  -> Large discrepancy (anomaly signal!)',
                ha='center', va='center', fontsize=10, transform=ax.transAxes)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'loss_components.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - loss_components.png")

    def generate_all(self):
        """Generate all architecture visualizations"""
        print("\n  Generating Architecture Visualizations...")
        self.plot_model_pipeline()
        self.plot_patchify_modes()
        self.plot_masking_visualization()
        self.plot_self_distillation_concept()
        self.plot_margin_types()
        self.plot_loss_components()


# =============================================================================
# ExperimentVisualizer - Stage 1 Results Visualizations
# =============================================================================

class ExperimentVisualizer:
    """Visualize Stage 1 (Quick Search) results"""

    def __init__(self, results_df: pd.DataFrame, param_keys: List[str], output_dir: str):
        self.results_df = results_df
        self.param_keys = param_keys
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_heatmaps(self, metric: str = 'roc_auc'):
        """Generate heatmaps for parameter pairs"""
        numeric_params = [p for p in self.param_keys if self.results_df[p].dtype in ['float64', 'int64']]

        if len(numeric_params) < 2:
            print("  - Skipping heatmaps (insufficient numeric parameters)")
            return

        n_params = len(numeric_params)
        n_plots = n_params * (n_params - 1) // 2
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_plots > 1 else [axes]

        idx = 0
        for i, param1 in enumerate(numeric_params):
            for j, param2 in enumerate(numeric_params):
                if i >= j:
                    continue

                pivot = self.results_df.pivot_table(
                    values=metric, index=param1, columns=param2, aggfunc='mean'
                )

                ax = axes[idx]
                sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax, cbar_kws={'label': metric})
                ax.set_title(f'{param1} vs {param2}')
                idx += 1

        # Hide unused axes
        for ax in axes[idx:]:
            ax.axis('off')

        plt.suptitle(f'Parameter Pair Heatmaps ({metric.upper()})', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'heatmaps_{metric}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  - heatmaps_{metric}.png")

    def plot_parallel_coordinates(self):
        """Generate parallel coordinates plot"""
        df = self.results_df.copy()

        # Select numeric columns for parallel coordinates
        numeric_cols = [c for c in self.param_keys if df[c].dtype in ['float64', 'int64']]
        if len(numeric_cols) < 2:
            print("  - Skipping parallel coordinates (insufficient numeric parameters)")
            return

        # Normalize columns
        df_norm = df.copy()
        for col in numeric_cols:
            col_min = df[col].min()
            col_max = df[col].max()
            if col_max > col_min:
                df_norm[col] = (df[col] - col_min) / (col_max - col_min)
            else:
                df_norm[col] = 0.5

        fig, ax = plt.subplots(figsize=(12, 6))

        # Color by ROC-AUC
        cmap = plt.cm.RdYlGn
        norm = plt.Normalize(df['roc_auc'].min(), df['roc_auc'].max())

        for idx, row in df_norm.iterrows():
            values = [row[col] for col in numeric_cols]
            color = cmap(norm(df.loc[idx, 'roc_auc']))
            ax.plot(range(len(numeric_cols)), values, color=color, alpha=0.3, lw=0.8)

        # Highlight top 10
        top_10_idx = df.nlargest(10, 'roc_auc').index
        for idx in top_10_idx:
            values = [df_norm.loc[idx, col] for col in numeric_cols]
            ax.plot(range(len(numeric_cols)), values, color='red', alpha=0.8, lw=2)

        ax.set_xticks(range(len(numeric_cols)))
        ax.set_xticklabels(numeric_cols, rotation=45, ha='right')
        ax.set_ylabel('Normalized Value')
        ax.set_title('Parallel Coordinates (colored by ROC-AUC, red=top 10)', fontsize=12, fontweight='bold')

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='ROC-AUC')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'parallel_coordinates.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - parallel_coordinates.png")

    def plot_parameter_importance(self):
        """Generate parameter importance box plots"""
        fig, axes = plt.subplots(2, 4, figsize=(16, 10))
        axes = axes.flatten()

        for idx, param in enumerate(self.param_keys):
            if idx >= len(axes):
                break

            ax = axes[idx]
            groups = self.results_df.groupby(param)['roc_auc'].apply(list).to_dict()

            labels = list(groups.keys())
            data = list(groups.values())

            bp = ax.boxplot(data, labels=[str(l)[:10] for l in labels], patch_artist=True)

            # Color boxes
            colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)

            ax.set_title(f'{param}', fontsize=11, fontweight='bold')
            ax.set_ylabel('ROC-AUC')
            ax.tick_params(axis='x', rotation=45)

        # Hide unused axes
        for ax in axes[len(self.param_keys):]:
            ax.axis('off')

        plt.suptitle('Parameter Importance (ROC-AUC Distribution)', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'parameter_importance.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - parameter_importance.png")

    def plot_top_k_comparison(self, k: int = 10):
        """Plot top k configurations comparison"""
        actual_k = min(k, len(self.results_df))
        top_k = self.results_df.nlargest(actual_k, 'roc_auc')

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Bar chart
        ax = axes[0]
        x = np.arange(actual_k)
        width = 0.35

        bars1 = ax.bar(x - width/2, top_k['roc_auc'], width, label='ROC-AUC', color='#3498DB')
        bars2 = ax.bar(x + width/2, top_k['f1_score'], width, label='F1-Score', color='#E74C3C')

        ax.set_xlabel('Configuration Rank')
        ax.set_ylabel('Score')
        ax.set_title(f'Top {actual_k} Configurations', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'#{i+1}' for i in range(actual_k)])
        ax.legend()
        ax.set_ylim(0, 1.1)

        # Add values
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

        # Table
        ax = axes[1]
        ax.axis('off')

        table_data = []
        for i, (_, row) in enumerate(top_k.iterrows()):
            config_str = ', '.join([f"{p}={row[p]}" for p in self.param_keys[:4]])
            table_data.append([
                f'#{i+1}',
                f'{row["roc_auc"]:.4f}',
                f'{row["f1_score"]:.4f}',
                config_str[:50] + '...' if len(config_str) > 50 else config_str
            ])

        table = ax.table(
            cellText=table_data,
            colLabels=['Rank', 'ROC-AUC', 'F1', 'Configuration'],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax.set_title(f'Top {k} Configuration Details', fontsize=12, fontweight='bold', y=0.95)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'top_k_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - top_k_comparison.png")

    def plot_metric_distributions(self):
        """Plot metric distributions"""
        metrics = ['roc_auc', 'f1_score', 'precision', 'recall']
        available_metrics = [m for m in metrics if m in self.results_df.columns]

        fig, axes = plt.subplots(1, len(available_metrics), figsize=(4*len(available_metrics), 4))
        if len(available_metrics) == 1:
            axes = [axes]

        for ax, metric in zip(axes, available_metrics):
            data = self.results_df[metric].dropna()
            ax.hist(data, bins=30, color='#3498DB', edgecolor='black', alpha=0.7)
            ax.axvline(data.mean(), color='red', linestyle='--', lw=2, label=f'Mean: {data.mean():.4f}')
            ax.axvline(data.median(), color='green', linestyle='--', lw=2, label=f'Median: {data.median():.4f}')
            ax.set_title(metric.upper(), fontsize=11, fontweight='bold')
            ax.set_xlabel('Score')
            ax.set_ylabel('Count')
            ax.legend(fontsize=8)

        plt.suptitle('Metric Distributions', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'metric_distributions.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - metric_distributions.png")

    def plot_metric_correlations(self):
        """Plot metric correlations"""
        metrics = ['roc_auc', 'f1_score', 'precision', 'recall']
        if 'disturbing_roc_auc' in self.results_df.columns:
            metrics.append('disturbing_roc_auc')

        available_metrics = [m for m in metrics if m in self.results_df.columns]
        corr = self.results_df[available_metrics].corr()

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax,
                   vmin=-1, vmax=1, center=0)
        ax.set_title('Metric Correlations', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'metric_correlations.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - metric_correlations.png")

    def plot_categorical_comparison(self, param: str, metric: str = 'roc_auc'):
        """Plot comparison for a categorical parameter"""
        if param not in self.results_df.columns:
            return

        groups = self.results_df.groupby(param)[metric]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Bar chart with error bars
        ax = axes[0]
        means = groups.mean()
        stds = groups.std()
        x = range(len(means))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=plt.cm.Set2(np.linspace(0, 1, len(means))))
        ax.set_xticks(x)
        ax.set_xticklabels(means.index, rotation=45, ha='right')
        ax.set_ylabel(metric)
        ax.set_title(f'{param} Mean {metric} (with std)', fontsize=11, fontweight='bold')

        # Box plot
        ax = axes[1]
        data = [self.results_df[self.results_df[param] == v][metric].values for v in means.index]
        bp = ax.boxplot(data, labels=means.index, patch_artist=True)
        for patch, color in zip(bp['boxes'], plt.cm.Set2(np.linspace(0, 1, len(means)))):
            patch.set_facecolor(color)
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylabel(metric)
        ax.set_title(f'{param} Distribution', fontsize=11, fontweight='bold')

        # Violin plot
        ax = axes[2]
        positions = range(len(means.index))
        parts = ax.violinplot(data, positions=positions, showmeans=True, showmedians=True)
        ax.set_xticks(positions)
        ax.set_xticklabels(means.index, rotation=45, ha='right')
        ax.set_ylabel(metric)
        ax.set_title(f'{param} Violin Plot', fontsize=11, fontweight='bold')

        plt.suptitle(f'{param} Comparison ({metric})', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{param}_comparison_{metric}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  - {param}_comparison_{metric}.png")

    def plot_summary_dashboard(self):
        """Create a comprehensive summary dashboard"""
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # 1. ROC-AUC distribution
        ax = fig.add_subplot(gs[0, 0])
        ax.hist(self.results_df['roc_auc'], bins=30, color='#3498DB', edgecolor='black', alpha=0.7)
        ax.axvline(self.results_df['roc_auc'].mean(), color='red', linestyle='--', lw=2)
        ax.set_title('ROC-AUC Distribution', fontweight='bold')
        ax.set_xlabel('ROC-AUC')
        ax.set_ylabel('Count')

        # 2. Top 10 bar chart
        ax = fig.add_subplot(gs[0, 1])
        top_n = min(10, len(self.results_df))
        top_10 = self.results_df.nlargest(top_n, 'roc_auc')
        ax.barh(range(top_n), top_10['roc_auc'].values[::-1], color='#27AE60')
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([f'#{i+1}' for i in range(top_n-1, -1, -1)])
        ax.set_xlabel('ROC-AUC')
        ax.set_title('Top 10 Configurations', fontweight='bold')

        # 3. Summary statistics
        ax = fig.add_subplot(gs[0, 2])
        ax.axis('off')
        stats_text = f"""
        Total Configurations: {len(self.results_df)}

        ROC-AUC:
          Best: {self.results_df['roc_auc'].max():.4f}
          Mean: {self.results_df['roc_auc'].mean():.4f}
          Std: {self.results_df['roc_auc'].std():.4f}

        F1-Score:
          Best: {self.results_df['f1_score'].max():.4f}
          Mean: {self.results_df['f1_score'].mean():.4f}
        """
        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_title('Summary Statistics', fontweight='bold')

        # 4-6. Parameter importance for first 3 numeric params
        numeric_params = [p for p in self.param_keys if self.results_df[p].dtype in ['float64', 'int64']][:3]
        for i, param in enumerate(numeric_params):
            ax = fig.add_subplot(gs[1, i])
            groups = self.results_df.groupby(param)['roc_auc'].apply(list).to_dict()
            bp = ax.boxplot(list(groups.values()), labels=[str(k) for k in groups.keys()], patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('#3498DB')
            ax.set_title(f'{param}', fontweight='bold')
            ax.set_ylabel('ROC-AUC')

        # 7-9. Categorical params
        cat_params = ['patchify_mode', 'margin_type', 'force_mask_anomaly']
        for i, param in enumerate(cat_params):
            if param in self.results_df.columns:
                ax = fig.add_subplot(gs[2, i])
                groups = self.results_df.groupby(param)['roc_auc']
                means = groups.mean()
                stds = groups.std()
                colors = plt.cm.Set2(np.linspace(0, 1, len(means)))
                ax.bar(range(len(means)), means, yerr=stds, capsize=5, color=colors)
                ax.set_xticks(range(len(means)))
                ax.set_xticklabels(means.index, rotation=45, ha='right')
                ax.set_title(f'{param}', fontweight='bold')
                ax.set_ylabel('ROC-AUC')

        plt.suptitle('Stage 1 Summary Dashboard', fontsize=16, fontweight='bold', y=1.02)
        plt.savefig(os.path.join(self.output_dir, 'stage1_summary_dashboard.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - stage1_summary_dashboard.png")

    def generate_all(self):
        """Generate all Stage 1 visualizations"""
        print("\n  Generating Stage 1 Visualizations...")
        self.plot_heatmaps()
        self.plot_parallel_coordinates()
        self.plot_parameter_importance()
        self.plot_top_k_comparison()
        self.plot_metric_distributions()
        self.plot_metric_correlations()

        # Categorical comparisons
        for param in ['patchify_mode', 'margin_type']:
            if param in self.param_keys:
                self.plot_categorical_comparison(param)

        self.plot_summary_dashboard()


# =============================================================================
# Stage2Visualizer - Stage 2 Results Visualizations
# =============================================================================

class Stage2Visualizer:
    """Visualize Stage 2 (Full Training) results"""

    def __init__(self, full_df: pd.DataFrame, quick_df: pd.DataFrame,
                 histories: Dict, output_dir: str):
        self.full_df = full_df
        self.quick_df = quick_df
        self.histories = histories
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_quick_vs_full(self):
        """Compare quick search vs full training performance"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Scatter plot
        ax = axes[0]
        ax.scatter(self.full_df['quick_roc_auc'], self.full_df['roc_auc'],
                  alpha=0.6, s=50, c='#3498DB')

        # Diagonal line
        min_val = min(self.full_df['quick_roc_auc'].min(), self.full_df['roc_auc'].min())
        max_val = max(self.full_df['quick_roc_auc'].max(), self.full_df['roc_auc'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='y=x')

        ax.set_xlabel('Quick Search ROC-AUC')
        ax.set_ylabel('Full Training ROC-AUC')
        ax.set_title('Quick vs Full Performance', fontsize=12, fontweight='bold')
        ax.legend()

        # Improvement distribution
        ax = axes[1]
        improvements = self.full_df['roc_auc'] - self.full_df['quick_roc_auc']
        ax.hist(improvements, bins=30, color='#27AE60', edgecolor='black', alpha=0.7)
        ax.axvline(improvements.mean(), color='red', linestyle='--', lw=2,
                  label=f'Mean: {improvements.mean():.4f}')
        ax.axvline(0, color='black', linestyle='-', lw=1)
        ax.set_xlabel('ROC-AUC Improvement')
        ax.set_ylabel('Count')
        ax.set_title('Improvement Distribution', fontsize=12, fontweight='bold')
        ax.legend()

        # Top 10 comparison
        ax = axes[2]
        top_n = min(10, len(self.full_df))
        top_10_full = self.full_df.nlargest(top_n, 'roc_auc')
        top_10_improve = self.full_df.nlargest(top_n, 'roc_auc_improvement')

        x = np.arange(top_n)
        width = 0.35

        ax.bar(x - width/2, top_10_full['roc_auc'].values, width,
              label='Top by Final ROC-AUC', color='#3498DB')
        ax.bar(x + width/2, top_10_improve['roc_auc'].values, width,
              label='Top by Improvement', color='#E74C3C')

        ax.set_xlabel('Rank')
        ax.set_ylabel('Final ROC-AUC')
        ax.set_title(f'Top {top_n} by Different Criteria', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'#{i+1}' for i in range(top_n)])
        ax.legend()

        plt.suptitle('Quick Search vs Full Training Comparison', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'stage2_quick_vs_full.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - stage2_quick_vs_full.png")

    def plot_selection_criterion_analysis(self):
        """Analyze performance by selection criterion"""
        if 'selection_criterion' not in self.full_df.columns:
            print("  - Skipping selection criterion analysis (column not found)")
            return

        criteria_groups = self.full_df.groupby('selection_criterion')['roc_auc']

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Box plot
        ax = axes[0]
        criteria_data = {name: group.values for name, group in criteria_groups}
        bp = ax.boxplot(criteria_data.values(), labels=criteria_data.keys(), patch_artist=True)

        colors = plt.cm.Set3(np.linspace(0, 1, len(criteria_data)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax.tick_params(axis='x', rotation=45)
        ax.set_ylabel('ROC-AUC')
        ax.set_title('Performance by Selection Criterion', fontsize=12, fontweight='bold')

        # Bar chart of counts and mean performance
        ax = axes[1]
        means = criteria_groups.mean()
        counts = criteria_groups.count()

        x = np.arange(len(means))
        width = 0.4

        ax2 = ax.twinx()

        bars1 = ax.bar(x - width/2, means, width, label='Mean ROC-AUC', color='#3498DB')
        bars2 = ax2.bar(x + width/2, counts, width, label='Count', color='#E74C3C', alpha=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels(means.index, rotation=45, ha='right')
        ax.set_ylabel('Mean ROC-AUC', color='#3498DB')
        ax2.set_ylabel('Count', color='#E74C3C')
        ax.set_title('Selection Criteria Statistics', fontsize=12, fontweight='bold')

        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'stage2_selection_criterion.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - stage2_selection_criterion.png")

    def plot_learning_curves(self, top_k: int = 10):
        """Plot learning curves for top experiments"""
        if not self.histories:
            print("  - Skipping learning curves (no history data)")
            return

        # Get top k experiment IDs
        top_k_df = self.full_df.nlargest(top_k, 'roc_auc')

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Training loss
        ax = axes[0]
        for _, row in top_k_df.iterrows():
            exp_id = str(int(row['combination_id']))
            if exp_id in self.histories:
                history = self.histories[exp_id]
                if 'train_loss' in history:
                    ax.plot(history['train_loss'], alpha=0.7,
                           label=f"#{int(row.get('stage2_rank', 0))} (ROC={row['roc_auc']:.3f})")

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Training Loss')
        ax.set_title('Training Loss Curves (Top 10)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')

        # Validation loss if available
        ax = axes[1]
        has_val = False
        for _, row in top_k_df.iterrows():
            exp_id = str(int(row['combination_id']))
            if exp_id in self.histories:
                history = self.histories[exp_id]
                if 'val_loss' in history:
                    ax.plot(history['val_loss'], alpha=0.7,
                           label=f"#{int(row.get('stage2_rank', 0))}")
                    has_val = True

        if has_val:
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Validation Loss')
            ax.set_title('Validation Loss Curves (Top 10)', fontsize=12, fontweight='bold')
            ax.legend(fontsize=8, loc='upper right')
        else:
            ax.axis('off')
            ax.text(0.5, 0.5, 'No validation loss data available',
                   ha='center', va='center', transform=ax.transAxes)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'learning_curves.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - learning_curves.png")

    def plot_summary_dashboard(self):
        """Create Stage 2 summary dashboard"""
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Quick vs Full scatter
        ax = fig.add_subplot(gs[0, 0])
        ax.scatter(self.full_df['quick_roc_auc'], self.full_df['roc_auc'],
                  alpha=0.6, s=50, c='#3498DB')
        min_val = min(self.full_df['quick_roc_auc'].min(), self.full_df['roc_auc'].min())
        max_val = max(self.full_df['quick_roc_auc'].max(), self.full_df['roc_auc'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        ax.set_xlabel('Quick Search ROC-AUC')
        ax.set_ylabel('Full Training ROC-AUC')
        ax.set_title('Quick vs Full Performance', fontweight='bold')

        # 2. Improvement histogram
        ax = fig.add_subplot(gs[0, 1])
        improvements = self.full_df['roc_auc'] - self.full_df['quick_roc_auc']
        ax.hist(improvements, bins=20, color='#27AE60', edgecolor='black', alpha=0.7)
        ax.axvline(improvements.mean(), color='red', linestyle='--', lw=2)
        ax.axvline(0, color='black', linestyle='-', lw=1)
        ax.set_xlabel('ROC-AUC Improvement')
        ax.set_ylabel('Count')
        ax.set_title(f'Improvement (mean={improvements.mean():.4f})', fontweight='bold')

        # 3. Top 10 table
        ax = fig.add_subplot(gs[1, 0])
        ax.axis('off')
        top_10 = self.full_df.nlargest(10, 'roc_auc')

        table_data = []
        for i, (_, row) in enumerate(top_10.iterrows()):
            table_data.append([
                f'#{i+1}',
                f'{row["roc_auc"]:.4f}',
                f'{row["quick_roc_auc"]:.4f}',
                f'{row["roc_auc"] - row["quick_roc_auc"]:+.4f}',
                str(row.get('selection_criterion', 'N/A'))[:20]
            ])

        table = ax.table(
            cellText=table_data,
            colLabels=['Rank', 'Final ROC', 'Quick ROC', 'Improve', 'Criterion'],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax.set_title('Top 10 Stage 2 Results', fontweight='bold', y=0.95)

        # 4. Statistics
        ax = fig.add_subplot(gs[1, 1])
        ax.axis('off')

        stats_text = f"""
        Stage 2 Summary
        ═══════════════════════════════════════

        Total Models Trained: {len(self.full_df)}

        Final ROC-AUC:
          Best:   {self.full_df['roc_auc'].max():.4f}
          Mean:   {self.full_df['roc_auc'].mean():.4f}
          Std:    {self.full_df['roc_auc'].std():.4f}

        Improvement from Quick Search:
          Mean:   {improvements.mean():+.4f}
          Max:    {improvements.max():+.4f}
          Min:    {improvements.min():+.4f}

        Models Improved: {(improvements > 0).sum()} / {len(improvements)}
        """

        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        plt.suptitle('Stage 2 Summary Dashboard', fontsize=16, fontweight='bold', y=1.02)
        plt.savefig(os.path.join(self.output_dir, 'stage2_summary_dashboard.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - stage2_summary_dashboard.png")

    def plot_hyperparameter_impact(self, param: str, metric: str = 'roc_auc'):
        """Plot detailed impact of a single hyperparameter on Stage 2 results

        Args:
            param: Hyperparameter name
            metric: Metric to analyze
        """
        if param not in self.full_df.columns:
            print(f"  ! Skipping {param} impact (column not found)")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        groups = self.full_df.groupby(param)

        # 1. Box plot of final ROC-AUC
        ax = axes[0, 0]
        group_data = {str(k): v[metric].values for k, v in groups}
        bp = ax.boxplot(group_data.values(), labels=group_data.keys(), patch_artist=True)
        colors = plt.cm.Set2(np.linspace(0, 1, len(group_data)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        ax.set_xlabel(param)
        ax.set_ylabel(metric)
        ax.set_title(f'{param} vs {metric.upper()} (Full Training)', fontweight='bold')
        ax.tick_params(axis='x', rotation=45)

        # 2. Mean + Std comparison
        ax = axes[0, 1]
        means = groups[metric].mean()
        stds = groups[metric].std()
        x = np.arange(len(means))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in means.index], rotation=45, ha='right')
        ax.set_xlabel(param)
        ax.set_ylabel(f'Mean {metric}')
        ax.set_title(f'{param} Mean ± Std', fontweight='bold')

        # Add value labels
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.005,
                   f'{mean:.4f}', ha='center', va='bottom', fontsize=8)

        # 3. Improvement analysis (Quick -> Full)
        ax = axes[1, 0]
        improvements = self.full_df.groupby(param).apply(
            lambda x: (x['roc_auc'] - x['quick_roc_auc']).mean()
        )
        colors_imp = ['#27AE60' if imp > 0 else '#E74C3C' for imp in improvements]
        bars = ax.bar(range(len(improvements)), improvements, color=colors_imp, alpha=0.8, edgecolor='black')
        ax.set_xticks(range(len(improvements)))
        ax.set_xticklabels([str(k) for k in improvements.index], rotation=45, ha='right')
        ax.axhline(y=0, color='black', linestyle='-', lw=1)
        ax.set_xlabel(param)
        ax.set_ylabel('Mean Improvement')
        ax.set_title(f'{param} Mean Improvement (Quick → Full)', fontweight='bold')

        for bar, imp in zip(bars, improvements):
            y_pos = bar.get_height() + 0.002 if imp >= 0 else bar.get_height() - 0.01
            ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                   f'{imp:+.4f}', ha='center', va='bottom' if imp >= 0 else 'top', fontsize=8)

        # 4. Statistics table
        ax = axes[1, 1]
        ax.axis('off')

        table_data = []
        for param_val in means.index:
            param_data = self.full_df[self.full_df[param] == param_val]
            table_data.append([
                str(param_val),
                f"{param_data[metric].mean():.4f}",
                f"{param_data[metric].std():.4f}",
                f"{param_data[metric].max():.4f}",
                f"{param_data[metric].min():.4f}",
                f"{len(param_data)}"
            ])

        table = ax.table(
            cellText=table_data,
            colLabels=[param, 'Mean', 'Std', 'Max', 'Min', 'Count'],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)
        ax.set_title(f'{param} Statistics', fontweight='bold', y=0.95)

        plt.suptitle(f'Hyperparameter Impact: {param}', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        # Save to hyperparameter-specific file
        plt.savefig(os.path.join(self.output_dir, f'hyperparam_{param}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  - hyperparam_{param}.png")

    def plot_all_hyperparameters(self):
        """Generate separate visualization for each hyperparameter"""
        # margin and lambda_disc are fixed, not in grid search
        hyperparams = [
            'masking_ratio', 'masking_strategy', 'num_patches',
            'margin_type', 'force_mask_anomaly', 'patch_level_loss', 'patchify_mode'
        ]

        for param in hyperparams:
            if param in self.full_df.columns:
                self.plot_hyperparameter_impact(param)

    def plot_hyperparameter_interactions(self):
        """Plot interactions between key hyperparameters"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Key interactions to visualize (margin and lambda_disc are fixed)
        interactions = [
            ('masking_ratio', 'num_patches'),
            ('masking_strategy', 'patchify_mode'),
            ('margin_type', 'patchify_mode'),
            ('masking_ratio', 'masking_strategy'),
            ('num_patches', 'patchify_mode'),
            ('patchify_mode', 'patch_level_loss')
        ]

        for idx, (param1, param2) in enumerate(interactions):
            ax = axes[idx // 3, idx % 3]

            if param1 not in self.full_df.columns or param2 not in self.full_df.columns:
                ax.text(0.5, 0.5, f'{param1} or {param2} not found',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{param1} × {param2}', fontweight='bold')
                continue

            # Create pivot table
            try:
                pivot = self.full_df.pivot_table(
                    values='roc_auc', index=param1, columns=param2, aggfunc='mean'
                )
                sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax, cbar_kws={'label': 'ROC-AUC'})
                ax.set_title(f'{param1} × {param2}', fontweight='bold')
            except Exception as e:
                ax.text(0.5, 0.5, f'Cannot create heatmap:\n{str(e)[:30]}',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{param1} × {param2}', fontweight='bold')

        plt.suptitle('Hyperparameter Interactions (Stage 2 ROC-AUC)', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'hyperparameter_interactions.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - hyperparameter_interactions.png")

    def plot_best_config_summary(self):
        """Visualize the best configuration details"""
        best_row = self.full_df.loc[self.full_df['roc_auc'].idxmax()]

        fig, ax = plt.subplots(figsize=(14, 10))
        ax.axis('off')

        # margin=0.5, lambda_disc=0.5 are fixed
        hyperparams = [
            ('masking_ratio', '마스킹 비율', '마스킹할 패치의 비율'),
            ('masking_strategy', '마스킹 전략', 'patch 또는 feature_wise'),
            ('num_patches', '패치 수', '시퀀스를 나눌 패치 개수'),
            ('margin_type', '마진 타입', 'Loss 계산 방식'),
            ('force_mask_anomaly', '이상 강제 마스킹', '이상 영역 마스킹 여부'),
            ('patch_level_loss', '패치 레벨 Loss', '패치 단위 Loss 계산'),
            ('patchify_mode', '패치화 모드', 'CNN/Linear 패치 임베딩'),
            ('mask_last_n', '마지막 N 마스킹', '마지막 N 타임스텝 마스킹'),
        ]

        # Build summary text
        summary_lines = [
            "╔══════════════════════════════════════════════════════════════════════════════╗",
            "║                        BEST MODEL CONFIGURATION                              ║",
            "╠══════════════════════════════════════════════════════════════════════════════╣",
        ]

        for param, name, desc in hyperparams:
            if param in best_row.index:
                value = best_row[param]
                if isinstance(value, float):
                    value_str = f"{value:.4f}" if value < 1 else f"{value:.2f}"
                else:
                    value_str = str(value)
                line = f"║  {name:<20} = {value_str:<12} ({desc})"
                summary_lines.append(f"{line:<78}║")

        summary_lines.append("╠══════════════════════════════════════════════════════════════════════════════╣")
        summary_lines.append("║                           PERFORMANCE METRICS                                ║")
        summary_lines.append("╠══════════════════════════════════════════════════════════════════════════════╣")

        metrics = ['roc_auc', 'f1_score', 'precision', 'recall', 'quick_roc_auc', 'roc_auc_improvement']
        metric_names = ['ROC-AUC', 'F1-Score', 'Precision', 'Recall', 'Quick ROC-AUC', 'Improvement']

        for metric, name in zip(metrics, metric_names):
            if metric in best_row.index:
                value = best_row[metric]
                if pd.notna(value):
                    line = f"║  {name:<20} = {value:.4f}"
                    summary_lines.append(f"{line:<78}║")

        summary_lines.append("╚══════════════════════════════════════════════════════════════════════════════╝")

        summary_text = "\n".join(summary_lines)

        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

        plt.savefig(os.path.join(self.output_dir, 'best_config_summary.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  - best_config_summary.png")

    def generate_all(self):
        """Generate all Stage 2 visualizations"""
        print("\n  Generating Stage 2 Visualizations...")
        self.plot_quick_vs_full()
        self.plot_selection_criterion_analysis()
        self.plot_learning_curves()
        self.plot_summary_dashboard()
        self.plot_all_hyperparameters()
        self.plot_hyperparameter_interactions()
        self.plot_best_config_summary()


# =============================================================================
# BestModelVisualizer - Best Model Analysis Visualizations
# =============================================================================

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

    def _highlight_anomaly_regions(self, ax, point_labels, color='red', alpha=0.2, label='Anomaly Region'):
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
        ax.plot(fpr, tpr, color='#E74C3C', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        ax.scatter(fpr[optimal_idx], tpr[optimal_idx], s=100, c='green', zorder=5,
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
               label='Normal', color='#3498DB', density=True)
        ax.hist(self.pred_data['scores'][anomaly_mask], bins=50, alpha=0.6,
               label='Anomaly', color='#E74C3C', density=True)
        ax.axvline(self.pred_data['scores'][normal_mask].mean(), color='#3498DB', linestyle='--', lw=2)
        ax.axvline(self.pred_data['scores'][anomaly_mask].mean(), color='#E74C3C', linestyle='--', lw=2)
        ax.set_xlabel('Anomaly Score')
        ax.set_ylabel('Density')
        ax.set_title('Anomaly Score Distribution', fontsize=12, fontweight='bold')
        ax.legend()

        # Box plot
        ax = axes[1]
        box_data = [self.pred_data['scores'][normal_mask], self.pred_data['scores'][anomaly_mask]]
        bp = ax.boxplot(box_data, labels=['Normal', 'Anomaly'], patch_artist=True)
        bp['boxes'][0].set_facecolor('#3498DB')
        bp['boxes'][1].set_facecolor('#E74C3C')
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
                  alpha=0.5, label='Normal', color='#3498DB', s=30)
        ax.scatter(self.pred_data['recon_errors'][anomaly_mask],
                  self.pred_data['discrepancies'][anomaly_mask],
                  alpha=0.5, label='Anomaly', color='#E74C3C', s=30)
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

        bars1 = ax.bar(x - width/2, [normal_recon, anomaly_recon], width, label='Recon Error', color='#3498DB')
        bars2 = ax.bar(x + width/2, [normal_disc, anomaly_disc], width, label='Discrepancy', color='#E74C3C')

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
                  alpha=0.6, label='Normal', color='#3498DB', s=30)
        ax.scatter(teacher_mean[anomaly_mask], student_mean[anomaly_mask],
                  alpha=0.6, label='Anomaly', color='#E74C3C', s=30)

        max_val = max(teacher_mean.max(), student_mean.max())
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='y=x')

        ax.set_xlabel('Teacher Reconstruction Error')
        ax.set_ylabel('Student Reconstruction Error')
        ax.set_title('Teacher vs Student Error', fontsize=12, fontweight='bold')
        ax.legend()

        # Discrepancy distribution
        ax = axes[1]
        disc_mean = (self.detailed_data['discrepancies'] * mask_inverse).sum(axis=1) / (mask_inverse.sum(axis=1) + 1e-8)

        ax.hist(disc_mean[normal_mask], bins=30, alpha=0.6, label='Normal', color='#3498DB', density=True)
        ax.hist(disc_mean[anomaly_mask], bins=30, alpha=0.6, label='Anomaly', color='#E74C3C', density=True)
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
            self._highlight_anomaly_regions(ax, point_labels, color='red', alpha=0.3, label='Anomaly')
            ax.plot(x, original, 'b-', label='Original', alpha=0.8)
            ax.plot(x, teacher, 'g--', label='Teacher', alpha=0.8)
            ax.axvspan(mask_start, mask_end, alpha=0.2, color='yellow', label='Masked')
            ax.set_title(f'{label} - Original vs Teacher')
            ax.legend(fontsize=8)

            # Original vs Student
            ax = axes[row, 1] if len(all_samples) > 1 else axes[1]
            self._highlight_anomaly_regions(ax, point_labels, color='red', alpha=0.3, label='Anomaly')
            ax.plot(x, original, 'b-', label='Original', alpha=0.8)
            ax.plot(x, student, 'r--', label='Student', alpha=0.8)
            ax.axvspan(mask_start, mask_end, alpha=0.2, color='yellow', label='Masked')
            ax.set_title(f'{label} - Original vs Student')
            ax.legend(fontsize=8)

            # Discrepancy
            ax = axes[row, 2] if len(all_samples) > 1 else axes[2]
            self._highlight_anomaly_regions(ax, point_labels, color='red', alpha=0.3, label='Anomaly')
            ax.plot(x, disc, 'purple', lw=2)
            ax.axvspan(mask_start, mask_end, alpha=0.2, color='yellow', label='Masked')
            ax.axhline(y=disc.mean(), color='orange', linestyle='--', label=f'Mean: {disc.mean():.4f}')
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
            (tp_idx, 'True Positive', '#27AE60'),
            (tn_idx, 'True Negative', '#3498DB'),
            (fp_idx, 'False Positive', '#F39C12'),
            (fn_idx, 'False Negative', '#E74C3C')
        ]

        for ax, (indices, title, color) in zip(axes.flatten(), examples):
            if len(indices) > 0:
                idx = indices[0]
                original = self.detailed_data['originals'][idx, :, 0]
                point_labels = self.detailed_data['point_labels'][idx]
                x = np.arange(len(original))

                # Highlight anomaly regions first (so they appear behind the line)
                self._highlight_anomaly_regions(ax, point_labels, color='red', alpha=0.3, label='Anomaly Region')

                # Highlight masked region (last patch)
                mask_start = len(original) - self.config.mask_last_n
                ax.axvspan(mask_start, len(original), alpha=0.2, color='yellow', label='Masked Region')

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
║    - Patchify Mode: {getattr(self.config, 'patchify_mode', 'cnn_first'):<12}                              ║
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

        # Setup colors (7 anomaly types including normal)
        colors = {
            'normal': '#3498DB',
            'spike': '#E74C3C',
            'memory_leak': '#F39C12',
            'cpu_saturation': '#9B59B6',
            'network_congestion': '#E67E22',
            'cascading_failure': '#1ABC9C',
            'resource_contention': '#16A085'
        }

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
                pc.set_facecolor(colors.get(atype, '#95A5A6'))
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
        colors = ['#E74C3C', '#F39C12', '#9B59B6', '#1ABC9C', '#E67E22'][:len(anomaly_types)]
        bars = ax.bar(anomaly_types, detection_rates, color=colors, alpha=0.8, edgecolor='black')
        ax.set_ylabel('Detection Rate (%)')
        ax.set_title('Detection Rate by Anomaly Type', fontweight='bold')
        ax.set_ylim(0, 105)

        # Add value labels
        for bar, rate in zip(bars, detection_rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)

        # Mean score bar chart
        ax = axes[1]
        bars = ax.bar(anomaly_types, mean_scores, color=colors, alpha=0.8, edgecolor='black')
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

        colors = {
            'normal': '#3498DB',
            'spike': '#E74C3C',
            'memory_leak': '#F39C12',
            'noise': '#9B59B6',
            'drift': '#1ABC9C',
            'network_congestion': '#E67E22'
        }

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: All types scatter
        ax = axes[0]
        for atype in ANOMALY_TYPE_NAMES:
            type_data = detailed_csv[detailed_csv['anomaly_type_name'] == atype]
            if len(type_data) > 0:
                ax.scatter(type_data['reconstruction_loss'], type_data['discrepancy_loss'],
                          alpha=0.6, label=atype.replace('_', ' ').title(),
                          color=colors.get(atype, '#95A5A6'), s=30)

        ax.set_xlabel('Reconstruction Loss')
        ax.set_ylabel('Discrepancy Loss')
        ax.set_title('Loss Components by Anomaly Type', fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)

        # Right: Normal vs All Anomalies
        ax = axes[1]
        normal_data = detailed_csv[detailed_csv['anomaly_type_name'] == 'normal']
        anomaly_data = detailed_csv[detailed_csv['anomaly_type_name'] != 'normal']

        ax.scatter(normal_data['reconstruction_loss'], normal_data['discrepancy_loss'],
                  alpha=0.5, label='Normal', color='#3498DB', s=30)
        ax.scatter(anomaly_data['reconstruction_loss'], anomaly_data['discrepancy_loss'],
                  alpha=0.5, label='Anomaly', color='#E74C3C', s=30)

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

        sample_type_names = {0: 'Pure Normal', 1: 'Disturbing Normal', 2: 'Anomaly'}
        colors = {0: '#3498DB', 1: '#F39C12', 2: '#E74C3C'}

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
            ax.bar(x - width/2, recon_means, width, label='Reconstruction', color='#3498DB', alpha=0.8)
            ax.bar(x + width/2, disc_means, width, label='Discrepancy', color='#E74C3C', alpha=0.8)
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
        ax.hist(pure_total, bins=30, alpha=0.6, label=f'Pure Normal (n={pure_normal_mask.sum()})', color='#3498DB', density=True)
        ax.hist(dist_total, bins=30, alpha=0.6, label=f'Disturbing Normal (n={disturbing_mask.sum()})', color='#F39C12', density=True)
        ax.hist(anom_total, bins=30, alpha=0.6, label=f'Anomaly (n={anomaly_mask.sum()})', color='#E74C3C', density=True)
        ax.set_xlabel('Total Score (Recon + λ·Disc)')
        ax.set_ylabel('Density')
        ax.set_title('Score Distribution by Sample Type', fontweight='bold')
        ax.legend()

        # 2. Box plot comparison
        ax = axes[0, 1]
        box_data = [pure_total, dist_total, anom_total]
        labels = ['Pure\nNormal', 'Disturbing\nNormal', 'Anomaly']
        bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
        colors = ['#3498DB', '#F39C12', '#E74C3C']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel('Total Score')
        ax.set_title('Score Box Plot', fontweight='bold')

        # 3. Discrepancy comparison (key metric)
        ax = axes[0, 2]
        ax.hist(pure_disc, bins=30, alpha=0.6, label='Pure Normal', color='#3498DB', density=True)
        ax.hist(dist_disc, bins=30, alpha=0.6, label='Disturbing Normal', color='#F39C12', density=True)
        ax.hist(anom_disc, bins=30, alpha=0.6, label='Anomaly', color='#E74C3C', density=True)
        ax.set_xlabel('Discrepancy (Teacher-Student)')
        ax.set_ylabel('Density')
        ax.set_title('Discrepancy Distribution', fontweight='bold')
        ax.legend()

        # 4. Teacher vs Student scatter
        ax = axes[1, 0]
        ax.scatter(pure_teacher, pure_student, alpha=0.5, label='Pure Normal', color='#3498DB', s=20)
        ax.scatter(dist_teacher, dist_student, alpha=0.5, label='Disturbing Normal', color='#F39C12', s=20)
        ax.scatter(anom_teacher, anom_student, alpha=0.5, label='Anomaly', color='#E74C3C', s=20)
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

        bars1 = ax.bar(x - width, means_teacher, width, label='Teacher Error', color='#27AE60')
        bars2 = ax.bar(x, means_student, width, label='Student Error', color='#9B59B6')
        bars3 = ax.bar(x + width, means_disc, width, label='Discrepancy', color='#E74C3C')

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
        """Plot discrepancy trends across time steps"""
        # Select representative samples
        pure_normal_mask = self.detailed_data['sample_types'] == 0
        disturbing_mask = self.detailed_data['sample_types'] == 1
        anomaly_mask = self.detailed_data['sample_types'] == 2

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Mean discrepancy trend by sample type
        ax = axes[0, 0]
        seq_length = self.detailed_data['discrepancies'].shape[1]
        t = np.arange(seq_length)

        pure_mean = self.detailed_data['discrepancies'][pure_normal_mask].mean(axis=0)
        dist_mean = self.detailed_data['discrepancies'][disturbing_mask].mean(axis=0)
        anom_mean = self.detailed_data['discrepancies'][anomaly_mask].mean(axis=0)

        ax.plot(t, pure_mean, label='Pure Normal', color='#3498DB', lw=2)
        ax.plot(t, dist_mean, label='Disturbing Normal', color='#F39C12', lw=2)
        ax.plot(t, anom_mean, label='Anomaly', color='#E74C3C', lw=2)

        # Highlight last patch
        mask_last_n = getattr(self.config, 'mask_last_n', 5)
        ax.axvspan(seq_length - mask_last_n, seq_length, alpha=0.2, color='gray', label='Last Patch')

        ax.set_xlabel('Time Step')
        ax.set_ylabel('Mean Discrepancy')
        ax.set_title('Discrepancy Trend by Sample Type', fontweight='bold')
        ax.legend()

        # 2. Discrepancy heatmap for anomaly samples
        ax = axes[0, 1]
        anomaly_disc = self.detailed_data['discrepancies'][anomaly_mask][:50]  # First 50
        sns.heatmap(anomaly_disc, ax=ax, cmap='Reds', cbar_kws={'label': 'Discrepancy'})
        ax.axvline(x=seq_length - mask_last_n, color='white', linestyle='--', lw=2)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Sample Index')
        ax.set_title('Anomaly Samples - Discrepancy Heatmap', fontweight='bold')

        # 3. Discrepancy heatmap for normal samples
        ax = axes[1, 0]
        normal_disc = self.detailed_data['discrepancies'][pure_normal_mask][:50]  # First 50
        sns.heatmap(normal_disc, ax=ax, cmap='Reds', cbar_kws={'label': 'Discrepancy'})
        ax.axvline(x=seq_length - mask_last_n, color='white', linestyle='--', lw=2)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Sample Index')
        ax.set_title('Pure Normal Samples - Discrepancy Heatmap', fontweight='bold')

        # 4. Last patch discrepancy distribution
        ax = axes[1, 1]
        last_patch_pure = self.detailed_data['discrepancies'][pure_normal_mask, -mask_last_n:].mean(axis=1)
        last_patch_dist = self.detailed_data['discrepancies'][disturbing_mask, -mask_last_n:].mean(axis=1)
        last_patch_anom = self.detailed_data['discrepancies'][anomaly_mask, -mask_last_n:].mean(axis=1)

        ax.hist(last_patch_pure, bins=30, alpha=0.6, label='Pure Normal', color='#3498DB', density=True)
        ax.hist(last_patch_dist, bins=30, alpha=0.6, label='Disturbing Normal', color='#F39C12', density=True)
        ax.hist(last_patch_anom, bins=30, alpha=0.6, label='Anomaly', color='#E74C3C', density=True)
        ax.set_xlabel('Mean Discrepancy (Last Patch)')
        ax.set_ylabel('Density')
        ax.set_title('Last Patch Discrepancy Distribution', fontweight='bold')
        ax.legend()

        plt.suptitle('Discrepancy Trend Analysis', fontsize=14, fontweight='bold', y=1.02)
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
        ax.axhline(y=threshold, color='red', linestyle='--', lw=2, label=f'Threshold={threshold:.4f}')
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
        ax.axhline(y=threshold, color='red', linestyle='--', lw=2, label=f'Threshold')
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
        colors = ['#3498DB', '#F39C12', '#E74C3C']

        parts = ax.violinplot(data_for_violin, positions=[0, 1, 2], showmeans=True, showmedians=True)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)

        ax.axhline(y=threshold, color='green', linestyle='--', lw=2, label=f'Global Threshold')
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

        bars = ax.bar(['Pure Normal\n(FP Rate)', 'Disturbing Normal\n(FP Rate)', 'Anomaly\n(TP Rate)'],
                      [pure_fp, dist_fp, anom_tp], color=colors)
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

        ax.plot(t, pure_recon_temporal, label='Pure Normal', color='#3498DB', lw=2)
        ax.plot(t, dist_recon_temporal, label='Disturbing Normal', color='#F39C12', lw=2)
        ax.plot(t, anom_recon_temporal, label='Anomaly', color='#E74C3C', lw=2)
        ax.axvspan(seq_length - mask_last_n, seq_length, alpha=0.2, color='gray', label='Last Patch (masked)')
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
            ('True Positive', tp_idx, '#27AE60'),
            ('True Negative', tn_idx, '#3498DB'),
            ('False Positive', fp_idx, '#E67E22'),
            ('False Negative', fn_idx, '#E74C3C')
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
                ax.axvspan(anomaly_region[0], anomaly_region[-1], alpha=0.2, color='red', label='Anomaly')
            mask_start = len(original) - self.config.mask_last_n
            ax.axvspan(mask_start, len(original), alpha=0.2, color='yellow', label='Masked')

            ax.set_title(f'{cat_name}: Time Series', fontweight='bold', color=color)
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Value')
            if row == 0:
                ax.legend(fontsize=7, loc='upper right')

            # Column 2: Discrepancy profile
            ax = axes[row, 1]
            discrepancy = self.detailed_data['discrepancies'][median_idx]
            ax.fill_between(range(len(discrepancy)), discrepancy, alpha=0.6, color='#9B59B6')
            ax.plot(discrepancy, color='#8E44AD', lw=1)

            if len(anomaly_region) > 0:
                ax.axvspan(anomaly_region[0], anomaly_region[-1], alpha=0.2, color='red')
            ax.axvspan(mask_start, len(original), alpha=0.2, color='yellow')
            ax.axhline(y=np.mean(discrepancy[-self.config.mask_last_n:]), color='green', linestyle='--',
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

        Shows representative TP and FN examples for each anomaly type:
        spike, memory_leak, noise, drift, network_congestion
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

        anomaly_types = ['spike', 'memory_leak', 'noise', 'drift', 'network_congestion']
        anomaly_type_map = {1: 'spike', 2: 'memory_leak', 3: 'noise', 4: 'drift', 5: 'network_congestion'}

        fig, axes = plt.subplots(5, 4, figsize=(20, 25))

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
                        ax.axvspan(anomaly_region[0], anomaly_region[-1], alpha=0.2, color='red')
                    ax.axvspan(len(original) - self.config.mask_last_n, len(original), alpha=0.2, color='yellow')

                ax.set_title(f'{atype.upper()}: TP Example', fontweight='bold', color='#27AE60')
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
                ax.fill_between(range(len(discrepancy)), discrepancy, alpha=0.6, color='#27AE60')
                if len(anomaly_region) > 0:
                    ax.axvspan(anomaly_region[0], anomaly_region[-1], alpha=0.2, color='red')
                ax.axvspan(len(discrepancy) - self.config.mask_last_n, len(discrepancy), alpha=0.2, color='yellow')
            ax.set_title(f'{atype.upper()}: TP Discrepancy', fontweight='bold', color='#27AE60')
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
                        ax.axvspan(anomaly_region_fn[0], anomaly_region_fn[-1], alpha=0.2, color='red')
                    ax.axvspan(len(original) - self.config.mask_last_n, len(original), alpha=0.2, color='yellow')

                ax.set_title(f'{atype.upper()}: FN Example', fontweight='bold', color='#E74C3C')
            else:
                ax.text(0.5, 0.5, 'No FN (All Detected!)', ha='center', va='center', transform=ax.transAxes, color='#27AE60')
                ax.set_title(f'{atype.upper()}: FN Example', fontweight='bold')
            ax.set_xlabel('Time Step')

            # Column 4: FN discrepancy
            ax = axes[row, 3]
            if fn_mask.sum() > 0 and fn_idx < len(self.detailed_data['discrepancies']):
                discrepancy = self.detailed_data['discrepancies'][fn_idx]
                ax.fill_between(range(len(discrepancy)), discrepancy, alpha=0.6, color='#E74C3C')
                if len(anomaly_region_fn) > 0:
                    ax.axvspan(anomaly_region_fn[0], anomaly_region_fn[-1], alpha=0.2, color='red')
                ax.axvspan(len(discrepancy) - self.config.mask_last_n, len(discrepancy), alpha=0.2, color='yellow')
            ax.set_title(f'{atype.upper()}: FN Discrepancy', fontweight='bold', color='#E74C3C')
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
        ax.bar(x - width, feature_errors_tp, width, label='TP (Detected Anomaly)', color='#27AE60')
        ax.bar(x, feature_errors_fn, width, label='FN (Missed Anomaly)', color='#E74C3C')
        ax.bar(x + width, feature_errors_tn, width, label='TN (Normal)', color='#3498DB')
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

        ax.bar(x - width, feature_disc_tp, width, label='TP', color='#27AE60')
        ax.bar(x, feature_disc_fn, width, label='FN', color='#E74C3C')
        ax.bar(x + width, feature_disc_tn, width, label='TN', color='#3498DB')
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

        colors = ['#27AE60' if imp > 0 else '#E74C3C' for imp in importance[sorted_idx]]
        bars = ax.barh([f'Feature {i}' for i in sorted_idx], importance[sorted_idx], color=colors)
        ax.axvline(x=0, color='black', linestyle='-', lw=1)
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
            ax.axvspan(len(original) - self.config.mask_last_n, len(original), alpha=0.2, color='yellow')
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
            ax.axvspan(len(original) - self.config.mask_last_n, len(original), alpha=0.2, color='yellow')
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
                self._plot_sample_detail(axes[row], idx, f'Hardest FN #{row+1}', '#E74C3C', threshold)
            else:
                for col in range(3):
                    axes[row, col].text(0.5, 0.5, f'No FN #{row+1}', ha='center', va='center')
                    axes[row, col].axis('off')

        # Top 2 hardest FP
        for row in range(2, 4):
            fp_row = row - 2
            if fp_row < len(fp_sorted):
                idx = fp_sorted[fp_row]
                self._plot_sample_detail(axes[row], idx, f'Hardest FP #{fp_row+1}', '#E67E22', threshold)
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
            ax.axvspan(anomaly_region[0], anomaly_region[-1], alpha=0.2, color='red')
        ax.axvspan(mask_start, len(original), alpha=0.2, color='yellow')
        ax.set_title(f'{title_prefix}: Time Series', fontweight='bold', color=color)
        ax.legend(fontsize=7)

        # Column 2: Discrepancy
        ax = axes_row[1]
        ax.fill_between(range(len(discrepancy)), discrepancy, alpha=0.6, color='#9B59B6')
        ax.plot(discrepancy, color='#8E44AD', lw=1)
        if len(anomaly_region) > 0:
            ax.axvspan(anomaly_region[0], anomaly_region[-1], alpha=0.2, color='red')
        ax.axvspan(mask_start, len(original), alpha=0.2, color='yellow')
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

    def _get_optimal_threshold(self):
        """Get the optimal threshold from ROC curve."""
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(self.pred_data['labels'], self.pred_data['scores'])
        optimal_idx = np.argmax(tpr - fpr)
        return thresholds[optimal_idx]

    def _get_scores(self):
        """Get scores array (from pred_data or compute from detailed_data)."""
        return self.pred_data['scores']

    def generate_all(self, experiment_dir: str = None):
        """Generate all best model visualizations

        Args:
            experiment_dir: Path to experiment directory for loading detailed results
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
                'pure_normal': self.config.test_target_pure_normal,
                'disturbing_normal': self.config.test_target_disturbing_normal,
                'anomaly': self.config.test_target_anomaly
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
                   label=f'Normal (n={normal_mask.sum()})', color='#3498DB', density=True)
            ax.hist(data['scores'][anomaly_mask], bins=30, alpha=0.6,
                   label=f'Anomaly (n={anomaly_mask.sum()})', color='#E74C3C', density=True)

            # Add vertical lines for means
            ax.axvline(data['scores'][normal_mask].mean(), color='#3498DB', linestyle='--', lw=2)
            ax.axvline(data['scores'][anomaly_mask].mean(), color='#E74C3C', linestyle='--', lw=2)

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
                color = '#3498DB'
                alpha = 0.1
            else:
                color = '#E74C3C'
                alpha = 0.3
            ax.plot(epochs, score_matrix[i, :], color=color, alpha=alpha, lw=0.5)

        # Add mean trajectories
        normal_mean = score_matrix[labels == 0].mean(axis=0)
        anomaly_mean = score_matrix[labels == 1].mean(axis=0)
        ax.plot(epochs, normal_mean, color='#2980B9', lw=3, label='Normal Mean')
        ax.plot(epochs, anomaly_mean, color='#C0392B', lw=3, label='Anomaly Mean')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Anomaly Score')
        ax.set_title('Sample Score Trajectories (Normal=Blue, Anomaly=Red)', fontweight='bold')
        ax.legend()

        # Right: By sample type (Pure Normal, Disturbing Normal, Anomaly)
        ax = axes[1]
        colors = {0: '#3498DB', 1: '#F39C12', 2: '#E74C3C'}
        labels_map = {0: 'Pure Normal', 1: 'Disturbing Normal', 2: 'Anomaly'}

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
        ax.plot(epochs, roc_aucs, 'o-', color='#E74C3C', lw=2, markersize=8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('ROC-AUC')
        ax.set_title('ROC-AUC Evolution', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([min(0.5, min(roc_aucs) - 0.05), 1.0])

        # F1 Score
        ax = axes[0, 1]
        ax.plot(epochs, f1_scores, 'o-', color='#27AE60', lw=2, markersize=8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('F1-Score')
        ax.set_title('F1-Score Evolution', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Precision and Recall
        ax = axes[1, 0]
        ax.plot(epochs, precisions, 'o-', color='#3498DB', lw=2, markersize=8, label='Precision')
        ax.plot(epochs, recalls, 's-', color='#9B59B6', lw=2, markersize=8, label='Recall')
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
                ax.plot(epochs, score_matrix[i, :], alpha=0.6, lw=1.5, color='#F39C12')
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
                ax.plot(epochs, score_matrix[i, :], alpha=0.6, lw=1.5, color='#E74C3C')
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

        ax.plot(epochs, detection_rates, 'o-', color='#E74C3C', lw=2, markersize=8, label='Detection Rate (TPR)')
        ax.plot(epochs, false_alarm_rates, 's-', color='#3498DB', lw=2, markersize=8, label='False Alarm Rate (FPR)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Rate (%)')
        ax.set_title('Detection & False Alarm Rate Over Training', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])

        # 5. Threshold evolution
        ax = axes[1, 1]
        ax.plot(epochs, thresholds_by_epoch, 'o-', color='#27AE60', lw=2, markersize=8)
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

        # Anomaly type names (from dataset.py)
        anomaly_type_names = ['normal', 'spike', 'memory_leak', 'noise', 'drift', 'network_congestion']

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 1. Detection rate by anomaly type over epochs
        ax = axes[0]
        colors = ['#3498DB', '#E74C3C', '#F39C12', '#9B59B6', '#1ABC9C', '#E67E22']

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
        ax.plot(epochs, normal_scores, 'o--', lw=2, markersize=6, label='Normal', color='#3498DB')

        ax.axhline(y=threshold, color='green', linestyle='--', lw=2, alpha=0.7, label='Threshold')
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
                    ax.axvspan(anomaly_region[0], anomaly_region[-1], alpha=0.15, color='red')

                # Highlight last patch (masked region)
                ax.axvspan(len(original) - self.config.mask_last_n, len(original),
                          alpha=0.15, color='yellow')

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
                ax.fill_between(range(len(discrepancy)), discrepancy, alpha=0.6, color='#9B59B6', label='|T-S| Discrepancy')
                ax.plot(discrepancy, color='#8E44AD', lw=1)

                # Highlight regions
                if len(anomaly_region) > 0:
                    ax.axvspan(anomaly_region[0], anomaly_region[-1], alpha=0.15, color='red')
                ax.axvspan(len(original) - self.config.mask_last_n, len(original),
                          alpha=0.15, color='yellow')

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

        ax.plot(epochs, thresholds, 'o-', color='#27AE60', lw=2, markersize=8)
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

        ax.plot(epochs, separations, 'o-', color='#9B59B6', lw=2, markersize=8)
        ax.axhline(y=0, color='black', linestyle='-', lw=1)
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
                    ax.axvspan(anomaly_region[0], anomaly_region[-1], alpha=0.2, color='red')
                ax.axvspan(len(original) - self.config.mask_last_n, len(original), alpha=0.2, color='yellow')

                pred_color = '#27AE60' if epoch_pred == 'Detected' else '#E74C3C'
                ax.set_title(f'Epoch {epoch}: {epoch_pred}\nScore: {epoch_score:.4f} (Thresh: {epoch_thresh:.4f})',
                           fontsize=9, color=pred_color, fontweight='bold')
                if col == 0:
                    ax.set_ylabel(f'Late Bloomer #{sample_num+1}\nTime Series', fontsize=9)
                if sample_num == 0 and col == len(show_epochs) - 1:
                    ax.legend(fontsize=7, loc='upper right')

                # Row 2: Discrepancy
                ax = axes[sample_num * 3 + 1, col]
                discrepancy = np.abs(teacher_recon - student_recon)
                ax.fill_between(range(len(discrepancy)), discrepancy, alpha=0.6, color='#9B59B6')
                ax.plot(discrepancy, color='#8E44AD', lw=1)

                if len(anomaly_region) > 0:
                    ax.axvspan(anomaly_region[0], anomaly_region[-1], alpha=0.2, color='red')
                ax.axvspan(len(original) - self.config.mask_last_n, len(original), alpha=0.2, color='yellow')

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
                    ax.axvspan(anomaly_region[0], anomaly_region[-1], alpha=0.2, color='red')
                ax.axvspan(len(original) - self.config.mask_last_n, len(original), alpha=0.2, color='yellow')

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

def main():
    """Main entry point for visualization generation"""
    import argparse

    parser = argparse.ArgumentParser(description='Generate all visualizations for MAE experiments')
    parser.add_argument('--experiment-dir', type=str, default=None,
                       help='Path to experiment directory')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to best_model.pt file')
    parser.add_argument('--num-test', type=int, default=500,
                       help='Number of test samples for best model analysis')
    parser.add_argument('--skip-data', action='store_true',
                       help='Skip data visualizations')
    parser.add_argument('--skip-architecture', action='store_true',
                       help='Skip architecture visualizations')
    parser.add_argument('--skip-experiments', action='store_true',
                       help='Skip experiment result visualizations')
    parser.add_argument('--skip-model', action='store_true',
                       help='Skip best model visualizations')
    parser.add_argument('--retrain', action='store_true',
                       help='Re-train best model to visualize learning progress (takes ~10 min)')
    args = parser.parse_args()

    print("="*80)
    print(" " * 15 + "UNIFIED VISUALIZATION SCRIPT")
    print("="*80)

    setup_style()

    # Find experiment directory
    experiment_dir = None
    if args.experiment_dir:
        experiment_dir = args.experiment_dir
    elif args.model_path:
        experiment_dir = os.path.dirname(args.model_path)
    else:
        experiment_dir = find_latest_experiment()
        if experiment_dir:
            print(f"\nAuto-detected experiment: {experiment_dir}")

    if not experiment_dir or not os.path.exists(experiment_dir):
        print("ERROR: No experiment directory found.")
        print("Please specify --experiment-dir or run experiments first.")
        return

    # Create visualization output directory
    vis_dir = os.path.join(experiment_dir, 'visualization')
    os.makedirs(vis_dir, exist_ok=True)

    print(f"\nExperiment directory: {experiment_dir}")
    print(f"Visualization output: {vis_dir}")

    # Load experiment data
    print("\nLoading experiment data...")
    exp_data = load_experiment_data(experiment_dir)

    # Default config
    config = Config()
    if exp_data['best_config']:
        for key, value in exp_data['best_config'].items():
            if hasattr(config, key):
                setattr(config, key, value)

    # Get param keys (margin and lambda_disc are fixed, not in grid search)
    param_keys = ['masking_ratio', 'masking_strategy', 'num_patches',
                  'margin_type', 'force_mask_anomaly', 'patch_level_loss', 'patchify_mode']

    # 1. Data Visualizations
    if not args.skip_data:
        data_dir = os.path.join(vis_dir, 'data')
        data_vis = DataVisualizer(data_dir, config)
        data_vis.generate_all()

    # 2. Architecture Visualizations
    if not args.skip_architecture:
        arch_dir = os.path.join(vis_dir, 'architecture')
        arch_vis = ArchitectureVisualizer(arch_dir, config)
        arch_vis.generate_all()

    # 3. Stage 1 Visualizations
    if not args.skip_experiments and exp_data['quick_results'] is not None:
        stage1_dir = os.path.join(vis_dir, 'stage1')
        stage1_vis = ExperimentVisualizer(exp_data['quick_results'], param_keys, stage1_dir)
        stage1_vis.generate_all()

    # 4. Stage 2 Visualizations
    if not args.skip_experiments and exp_data['full_results'] is not None:
        stage2_dir = os.path.join(vis_dir, 'stage2')
        stage2_vis = Stage2Visualizer(
            exp_data['full_results'],
            exp_data['quick_results'],
            exp_data['histories'] or {},
            stage2_dir
        )
        stage2_vis.generate_all()

    # 5. Best Model Visualizations
    if not args.skip_model and exp_data['model_path']:
        best_dir = os.path.join(vis_dir, 'best_model')
        model, config, test_loader, _ = load_best_model(exp_data['model_path'], args.num_test)
        best_vis = BestModelVisualizer(model, config, test_loader, best_dir)
        best_vis.generate_all(experiment_dir=experiment_dir)

    # 6. Training Progress Visualizations (requires re-training)
    if args.retrain and exp_data['best_config']:
        progress_dir = os.path.join(vis_dir, 'training_progress')
        progress_vis = TrainingProgressVisualizer(
            best_config=exp_data['best_config'],
            output_dir=progress_dir,
            checkpoint_epochs=[0, 5, 10, 20, 40, 60, 80, 100],
            num_train=2000,  # Stage 2 settings
            num_test=500,
            max_epochs=100
        )
        progress_vis.generate_all()

    print("\n" + "="*80)
    print(" " * 20 + "ALL VISUALIZATIONS COMPLETE!")
    print(f" " * 15 + f"Results saved to: {vis_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
