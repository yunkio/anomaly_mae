"""
Base utilities for visualization module

This module provides:
- Common imports and setup
- Color palettes for anomaly types, features, and sample types
- Utility functions for loading experiment data and models
- Data collection functions for model analysis
"""

import os
import json
from typing import Dict, List, Tuple, Optional

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
    ANOMALY_TYPE_NAMES, FEATURE_NAMES,
)
from mae_anomaly.dataset_sliding import ANOMALY_TYPE_CONFIGS


# =============================================================================
# Color Palettes (Dynamic)
# =============================================================================

# Base color palette (enough for any number of categories)
_BASE_COLORS = [
    '#3498DB',  # Blue
    '#E74C3C',  # Red
    '#F39C12',  # Orange
    '#9B59B6',  # Purple
    '#E67E22',  # Dark Orange
    '#1ABC9C',  # Teal
    '#16A085',  # Dark Teal
    '#E91E63',  # Pink
    '#8E44AD',  # Dark Purple
    '#27AE60',  # Green
    '#2980B9',  # Dark Blue
    '#C0392B',  # Dark Red
]


def get_anomaly_colors() -> Dict[str, str]:
    """Generate consistent colors for all anomaly types

    Returns:
        Dict mapping anomaly type name to hex color
    """
    return {name: _BASE_COLORS[i % len(_BASE_COLORS)]
            for i, name in enumerate(ANOMALY_TYPE_NAMES)}


def get_feature_colors() -> Dict[str, str]:
    """Generate consistent colors for all features

    Returns:
        Dict mapping feature name to hex color
    """
    return {name: _BASE_COLORS[i % len(_BASE_COLORS)]
            for i, name in enumerate(FEATURE_NAMES)}


# Sample type constants
SAMPLE_TYPE_NAMES = {0: 'Pure Normal', 1: 'Disturbing Normal', 2: 'Anomaly'}
SAMPLE_TYPE_COLORS = {0: '#3498DB', 1: '#F39C12', 2: '#E74C3C'}


# =============================================================================
# Style Setup
# =============================================================================

def setup_style():
    """Setup matplotlib style for consistent visualizations"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['figure.dpi'] = 150


# =============================================================================
# Experiment Data Loading
# =============================================================================

def find_latest_experiment(base_dir: str = 'results/experiments') -> Optional[str]:
    """Find the most recent experiment directory

    Args:
        base_dir: Base directory containing experiment folders

    Returns:
        Path to the latest experiment directory, or None if not found
    """
    if not os.path.exists(base_dir):
        return None

    # Find directories with required files
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
    """Load all experiment data from directory

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        Dict containing experiment data (results, histories, metadata, etc.)
    """
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

    Args:
        model_path: Path to saved model checkpoint
        num_test: Number of test samples (used for info only)

    Returns:
        Tuple of (model, config, test_loader, metrics)
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


# =============================================================================
# Data Collection Functions
# =============================================================================

def collect_predictions(model, dataloader, config) -> Dict:
    """Collect model predictions and scores

    Uses the same evaluation method as evaluator.py:
    - Mask only last mask_last_n positions (not random masking)
    - Use MSE (squared error) for consistency with training

    Args:
        model: Trained model
        dataloader: Test data loader
        config: Model configuration

    Returns:
        Dict containing scores, labels, recon_errors, discrepancies, sample_types
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

    Args:
        model: Trained model
        dataloader: Test data loader
        config: Model configuration

    Returns:
        Dict containing detailed data (errors, reconstructions, etc.)
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
# Anomaly Injection Functions for Visualization
# =============================================================================

def get_anomaly_type_info() -> Dict:
    """Get information about each anomaly type for visualization

    Dynamically includes all anomaly types from ANOMALY_TYPE_NAMES.
    Known types have detailed info; unknown types get auto-generated info.

    Returns:
        Dict mapping anomaly type name to info dict with:
        - description: Human-readable description
        - affected_features: List of feature names affected
        - length_range: (min, max) duration
        - characteristics: Key characteristics for visualization
    """
    # Known anomaly type info (for detailed descriptions)
    known_info = {
        'spike': {
            'description': 'Traffic Spike / DDoS Attack',
            'affected_features': ['CPU', 'Network', 'ResponseTime'],
            'characteristics': 'Sudden spike in multiple metrics',
        },
        'memory_leak': {
            'description': 'Memory Leak',
            'affected_features': ['Memory', 'DiskIO'],
            'characteristics': 'Gradual increase, continues to end',
        },
        'cpu_saturation': {
            'description': 'CPU Saturation',
            'affected_features': ['CPU', 'ThreadCount'],
            'characteristics': 'Sustained high CPU, oscillation',
        },
        'network_congestion': {
            'description': 'Network Congestion',
            'affected_features': ['Network', 'ResponseTime', 'QueueLength'],
            'characteristics': 'Sustained high network load',
        },
        'cascading_failure': {
            'description': 'Cascading Failure',
            'affected_features': ['ErrorRate', 'ResponseTime', 'CPU'],
            'characteristics': 'Progressive failure across services',
        },
        'resource_contention': {
            'description': 'Resource Contention',
            'affected_features': ['CPU', 'Memory', 'ThreadCount'],
            'characteristics': 'Oscillating contention pattern',
        },
        'point_spike': {
            'description': 'Point Spike (True Point Anomaly)',
            'affected_features': ['2+ random features'],
            'characteristics': 'Very short (3-5 timesteps), random features',
        },
    }

    # Build info dict for all anomaly types in ANOMALY_TYPE_NAMES
    info = {}
    for i, name in enumerate(ANOMALY_TYPE_NAMES[1:], start=1):  # Skip 'normal'
        if name in known_info:
            info[name] = known_info[name].copy()
        else:
            # Auto-generate info for unknown anomaly types
            info[name] = {
                'description': name.replace('_', ' ').title(),
                'affected_features': ['Multiple features'],
                'characteristics': 'Anomaly pattern',
            }

        # Add length_range from ANOMALY_TYPE_CONFIGS if available
        if i in ANOMALY_TYPE_CONFIGS:
            info[name]['length_range'] = ANOMALY_TYPE_CONFIGS[i]['length_range']

    return info
