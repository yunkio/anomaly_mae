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

from mae_anomaly import Config, MultivariateTimeSeriesDataset, SelfDistilledMAEMultivariate, set_seed
from mae_anomaly.dataset import ANOMALY_TYPE_NAMES


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


def load_best_model(model_path: str, num_test: int = 500) -> Tuple:
    """Load saved best model and create test dataloader"""
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

    # Create test dataset
    set_seed(config.random_seed)
    test_dataset = MultivariateTimeSeriesDataset(
        num_samples=num_test,
        seq_length=config.seq_length,
        num_features=config.num_features,
        anomaly_ratio=0.3,
        seed=43
    )
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    return model, config, test_loader, metrics


def collect_predictions(model, dataloader, config) -> Dict:
    """Collect model predictions and scores"""
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

            teacher_output, student_output, mask = model(sequences)

            # Compute errors
            mask_inverse = 1 - mask
            teacher_error = torch.abs(teacher_output - sequences).mean(dim=-1)
            student_error = torch.abs(student_output - sequences).mean(dim=-1)
            discrepancy = torch.abs(teacher_error - student_error)

            # Masked region scores
            recon = (teacher_error * mask_inverse).sum(dim=1) / (mask_inverse.sum(dim=1) + 1e-8)
            disc = (discrepancy * mask_inverse).sum(dim=1) / (mask_inverse.sum(dim=1) + 1e-8)
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
    """Collect detailed data for analysis"""
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

            teacher_output, student_output, mask = model(sequences)

            teacher_error = torch.abs(teacher_output - sequences).mean(dim=-1)
            student_error = torch.abs(student_output - sequences).mean(dim=-1)
            discrepancy = torch.abs(teacher_error - student_error)

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
        # Create sample dataset
        set_seed(42)
        dataset = MultivariateTimeSeriesDataset(
            num_samples=200,
            seq_length=self.config.seq_length,
            num_features=1,
            anomaly_ratio=0.3,
            seed=42
        )

        # Collect samples by type
        normal_samples = []
        disturbing_samples = []
        anomaly_samples = []

        for i in range(len(dataset)):
            seq, label, _, sample_type, _ = dataset[i]
            if sample_type == 0:  # pure normal
                normal_samples.append(seq[:, 0].numpy())
            elif sample_type == 1:  # disturbing normal
                disturbing_samples.append(seq[:, 0].numpy())
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

        # Disturbing normal samples
        ax = axes[1]
        for i, seq in enumerate(disturbing_samples[:5]):
            ax.plot(x, seq, alpha=0.7, label=f'Sample {i+1}')
        ax.set_title('Disturbing Normal Samples\n(Anomaly pattern, but last patch normal)',
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
        dataset = MultivariateTimeSeriesDataset(
            num_samples=10,
            seq_length=self.config.seq_length,
            num_features=self.config.num_features,
            anomaly_ratio=0.3,
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
        dataset = MultivariateTimeSeriesDataset(
            num_samples=1000,
            seq_length=self.config.seq_length,
            num_features=self.config.num_features,
            anomaly_ratio=0.3,
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

    def generate_all(self):
        """Generate all data visualizations"""
        print("\n  Generating Data Visualizations...")
        self.plot_anomaly_types()
        self.plot_sample_types()
        self.plot_feature_examples()
        self.plot_dataset_statistics()


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

    def generate_all(self):
        """Generate all Stage 2 visualizations"""
        print("\n  Generating Stage 2 Visualizations...")
        self.plot_quick_vs_full()
        self.plot_selection_criterion_analysis()
        self.plot_learning_curves()
        self.plot_summary_dashboard()


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
        """Show TP, TN, FP, FN examples"""
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
                ax.plot(x, original, color=color, lw=2, label='Signal')
                ax.set_title(f'{title}\nScore: {self.pred_data["scores"][idx]:.4f}, '
                           f'Threshold: {threshold:.4f}', fontweight='bold')
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Value')
                ax.legend(fontsize=8, loc='upper right')
            else:
                ax.text(0.5, 0.5, f'No {title} examples', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title, fontweight='bold')

        plt.suptitle('Detection Examples', fontsize=14, fontweight='bold', y=1.02)
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

        # Setup colors
        colors = {
            'normal': '#3498DB',
            'spike': '#E74C3C',
            'memory_leak': '#F39C12',
            'noise': '#9B59B6',
            'drift': '#1ABC9C',
            'network_congestion': '#E67E22'
        }

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, atype in enumerate(ANOMALY_TYPE_NAMES):
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

        # Anomaly type analysis (requires detailed results from experiment)
        self.plot_loss_by_anomaly_type(experiment_dir)
        self.plot_performance_by_anomaly_type(experiment_dir)
        self.plot_loss_scatter_by_anomaly_type(experiment_dir)
        self.plot_sample_type_analysis(experiment_dir)


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

    # Get param keys
    param_keys = ['masking_ratio', 'num_patches', 'margin', 'lambda_disc',
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

    print("\n" + "="*80)
    print(" " * 20 + "ALL VISUALIZATIONS COMPLETE!")
    print(f" " * 15 + f"Results saved to: {vis_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
