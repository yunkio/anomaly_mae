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
    NormalDataComplexity,
    ANOMALY_TYPE_NAMES, FEATURE_NAMES,
)
from mae_anomaly.dataset_sliding import ANOMALY_TYPE_CONFIGS
from mae_anomaly.evaluator import aggregate_patch_scores_to_point_level


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

    Value-based anomalies (types 1-6): Use warm colors (red/orange tones)
    Pattern-based anomalies (types 7-9): Use cool colors (blue/purple tones)

    Returns:
        Dict mapping anomaly type name to hex color
    """
    # Pattern-based anomalies get distinct cool colors for visual distinction
    pattern_colors = {
        'correlation_inversion': '#2980B9',   # Strong Blue
        'temporal_flatline': '#8E44AD',       # Purple
        'frequency_shift': '#1F618D',         # Dark Blue
    }

    colors = {}
    for i, name in enumerate(ANOMALY_TYPE_NAMES):
        if name in pattern_colors:
            colors[name] = pattern_colors[name]
        else:
            colors[name] = _BASE_COLORS[i % len(_BASE_COLORS)]
    return colors


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
# Consistent Visualization Style (Learning Curves & Comparisons)
# =============================================================================
# Color scheme: Normal=blue tones, Anomaly=red tones
# Marker scheme: Grouped by loss type for cross-plot consistency

# Colors for data types (Normal vs Anomaly)
VIS_COLORS = {
    # Primary data types
    'normal': '#3498DB',      # Blue for normal data
    'pure_normal': '#2ECC71', # Green for pure normal (in normal window)
    'anomaly': '#E74C3C',     # Red for anomaly data
    'disturbing': '#F39C12',  # Orange for disturbing normal
    # Model components
    'teacher': '#27AE60',     # Green for teacher model
    'student': '#9B59B6',     # Purple for student model
    'discrepancy': '#E67E22', # Orange for discrepancy (distinct from student)
    'reconstruction': '#27AE60',  # Green for reconstruction
    'total': '#3498DB',       # Blue for totals/combined (distinct from teacher)
    # Region highlighting
    'anomaly_region': '#E74C3C',  # Red for anomaly region highlight
    'masked_region': '#F1C40F',   # Yellow for masked region highlight
    'normal_region': '#27AE60',   # Green for normal region highlight
    # Darker variants (for emphasis/mean lines)
    'normal_dark': '#2980B9',     # Dark blue
    'anomaly_dark': '#C0392B',    # Dark red
    'student_dark': '#8E44AD',    # Dark purple
    # Detection outcomes (TP/TN/FP/FN)
    'true_positive': '#27AE60',   # Green - correct detection
    'true_negative': '#3498DB',   # Blue - correct normal
    'false_positive': '#F39C12',  # Orange - false alarm
    'false_negative': '#E74C3C',  # Red - missed detection
    # General purpose
    'baseline': 'black',
    'reference': 'gray',
    'threshold': '#27AE60',       # Green for threshold lines
}

# Markers for loss types (consistent across all plots)
VIS_MARKERS = {
    'discrepancy': 's',       # Square for discrepancy loss
    'teacher_recon': 'o',     # Circle for teacher reconstruction
    'student_recon': '^',     # Triangle for student reconstruction
    'total': 'D',             # Diamond for total/combined
}

# Line styles
VIS_LINESTYLES = {
    'solid': '-',
    'dashed': '--',
    'dotted': ':',
}


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


def load_best_model(model_path: str, num_test: int = 2000, use_complexity: bool = False) -> Tuple:
    """Load saved best model and create test dataloader

    Uses SlidingWindowDataset for consistency with run_experiments.py

    Args:
        model_path: Path to saved model checkpoint
        num_test: Number of test samples (used for info only)
        use_complexity: Whether to use normal data complexity (default: False for temp experiments)

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
    complexity = NormalDataComplexity(enable_complexity=use_complexity)
    print(f"  Normal data complexity: {'ENABLED' if use_complexity else 'DISABLED'}")
    generator = SlidingWindowTimeSeriesGenerator(
        total_length=config.sliding_window_total_length,
        num_features=config.num_features,
        interval_scale=config.anomaly_interval_scale,
        complexity=complexity,
        seed=config.random_seed
    )
    signals, point_labels, anomaly_regions = generator.generate()

    # Create test dataset - no downsampling, stride=1 for PA%K evaluation
    test_dataset = SlidingWindowDataset(
        signals=signals,
        point_labels=point_labels,
        anomaly_regions=anomaly_regions,
        window_size=config.seq_length,
        stride=config.sliding_window_test_stride,  # Test stride=1 for PA%K
        mask_last_n=config.patch_size,
        split='test',
        train_ratio=config.sliding_window_train_ratio,
        seed=config.random_seed
    )
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    print(f"  Test dataset: {len(test_dataset)} samples")

    return model, config, test_loader, test_dataset, metrics



def compute_score_contributions(
    recon_all: np.ndarray,
    disc_all: np.ndarray,
    config
) -> Dict:
    """Compute score contributions for different scoring modes

    Calculates how much each component (reconstruction, discrepancy) contributes
    to the final anomaly score, based on the scoring mode.

    Args:
        recon_all: Array of reconstruction errors per sample
        disc_all: Array of discrepancy values per sample
        config: Model configuration with anomaly_score_mode and lambda_disc

    Returns:
        Dict containing:
            scores: Final anomaly scores
            recon_contrib: Absolute contribution from reconstruction
            disc_contrib: Absolute contribution from discrepancy
            recon_ratio: Ratio of recon contribution (0-1)
            disc_ratio: Ratio of disc contribution (0-1)
            score_mode: The scoring mode used
            mode_params: Dict of mode-specific parameters
    """
    score_mode = getattr(config, 'anomaly_score_mode', 'default')
    lambda_disc = getattr(config, 'lambda_disc', 0.5)

    mode_params = {'score_mode': score_mode}

    # OD disabled → zero out disc contribution
    od_enabled = getattr(config, 'use_output_discrepancy', True)
    if not od_enabled and not getattr(config, 'use_feature_matching', False):
        disc_all = np.zeros_like(disc_all)

    if score_mode == 'adaptive':
        # Auto-scale lambda based on mean values
        adaptive_lambda = recon_all.mean() / (disc_all.mean() + 1e-8)
        recon_contrib = recon_all
        disc_contrib = adaptive_lambda * disc_all
        scores = recon_contrib + disc_contrib

        total = scores + 1e-8
        recon_ratio = recon_contrib / total
        disc_ratio = disc_contrib / total

        mode_params['adaptive_lambda'] = adaptive_lambda

    elif score_mode == 'ratio_weighted':
        # Ratio-based: use disc relative to median
        disc_median = np.median(disc_all) + 1e-8
        recon_contrib = recon_all
        disc_contrib = recon_all * (disc_all / disc_median)  # Multiplicative factor
        scores = recon_all * (1 + disc_all / disc_median)

        # For ratio calculation, use additive interpretation
        total = recon_contrib + disc_contrib + 1e-8
        recon_ratio = recon_contrib / total
        disc_ratio = disc_contrib / total

        mode_params['disc_median'] = disc_median

    else:  # default
        recon_contrib = recon_all
        disc_contrib = lambda_disc * disc_all
        scores = recon_contrib + disc_contrib

        total = scores + 1e-8
        recon_ratio = recon_contrib / total
        disc_ratio = disc_contrib / total

        mode_params['lambda_disc'] = lambda_disc

    return {
        'scores': scores,
        'recon_contrib': recon_contrib,
        'disc_contrib': disc_contrib,
        'recon_ratio': recon_ratio,
        'disc_ratio': disc_ratio,
        'score_mode': score_mode,
        'mode_params': mode_params
    }


# collect_detailed_data and collect_all_visualization_data were removed.
# Reconstruction data is now collected by evaluator._compute_patch_scores_all_patches(collect_detail=True).
# pred_data is derived via derive_pred_data() below.


def derive_pred_data(
    recon_patches: np.ndarray,
    disc_patches: np.ndarray,
    student_patches: np.ndarray,
    labels: np.ndarray,
    sample_types: np.ndarray,
    config,
    test_dataset,
    subset_indices: Optional[np.ndarray] = None,
) -> Dict:
    """Derive pred_data dict from evaluator's patch scores (pure numpy, no GPU).

    Converts evaluator's (n_windows, num_patches) arrays into the format
    BestModelVisualizer expects, replacing the need for collect_predictions().

    Args:
        recon_patches: (n_windows, num_patches) teacher reconstruction scores
        disc_patches: (n_windows, num_patches) discrepancy scores
        student_patches: (n_windows, num_patches) student reconstruction scores
        labels: (n_windows,) window labels
        sample_types: (n_windows,) window sample types
        config: Config object
        test_dataset: SlidingWindowDataset (needs point_labels, window_start_indices)
        subset_indices: optional indices to subsample for viz (e.g. 10k)

    Returns:
        Dict matching collect_predictions() output format
    """
    num_patches = config.num_patches
    patch_size = config.patch_size
    seq_length = config.seq_length

    # Subsample if needed
    if subset_indices is not None:
        recon_p = recon_patches[subset_indices]
        disc_p = disc_patches[subset_indices]
        student_p = student_patches[subset_indices]
    else:
        recon_p = recon_patches
        disc_p = disc_patches
        student_p = student_patches

    n_windows = len(recon_p)

    # Flatten to 1D (patch-level)
    recon_flat = recon_p.flatten()
    disc_flat = disc_p.flatten()
    student_flat = student_p.flatten()

    # OD disabled → zero out disc in scoring
    od_enabled = getattr(config, 'use_output_discrepancy', True)
    if not od_enabled and not getattr(config, 'use_feature_matching', False):
        disc_flat = np.zeros_like(disc_flat)

    # Apply scoring formula (same as collect_predictions)
    score_mode = getattr(config, 'anomaly_score_mode', 'default')
    if score_mode == 'adaptive':
        adaptive_lambda = recon_flat.mean() / (disc_flat.mean() + 1e-4)
        scores_flat = recon_flat + adaptive_lambda * disc_flat
    elif score_mode == 'ratio_weighted':
        disc_median = np.median(disc_flat) + 1e-4
        scores_flat = recon_flat * (1 + disc_flat / disc_median)
    else:
        scores_flat = recon_flat + config.lambda_disc * disc_flat

    # Compute patch-level labels from test_dataset
    ws_all = np.array(test_dataset.window_start_indices)
    pt_labels_full = np.array(test_dataset.point_labels)
    total_len = len(pt_labels_full)
    ws_indices = ws_all[subset_indices] if subset_indices is not None else ws_all

    patch_labels = np.zeros((n_windows, num_patches), dtype=np.int64)
    for p_idx in range(num_patches):
        start_pos = p_idx * patch_size
        end_pos = min(start_pos + patch_size, seq_length)
        for w_idx in range(n_windows):
            ws = ws_indices[w_idx]
            if pt_labels_full[ws + start_pos:ws + end_pos].any():
                patch_labels[w_idx, p_idx] = 1

    # Per-patch sample_types
    window_has_anomaly = (patch_labels.sum(axis=1) > 0)
    patch_sample_types = np.zeros_like(patch_labels)
    patch_sample_types[patch_labels == 1] = 2  # anomaly
    patch_sample_types[window_has_anomaly[:, np.newaxis] & (patch_labels == 0)] = 1  # disturbing

    result = {
        'patch_scores': scores_flat,
        'patch_labels': patch_labels.flatten(),
        'recon_errors': recon_flat,
        'student_errors': student_flat,
        'discrepancies': disc_flat,
        'sample_types': patch_sample_types.flatten(),
        'n_windows': n_windows,
        'num_patches': num_patches,
        'patch_size': patch_size,
        'seq_length': seq_length,
    }

    # Point-level aggregation
    patch_scores_2d = scores_flat.reshape(n_windows, num_patches)
    patch_recon_2d = recon_flat.reshape(n_windows, num_patches)
    patch_disc_2d = disc_flat.reshape(n_windows, num_patches)
    patch_student_2d = student_flat.reshape(n_windows, num_patches)

    point_scores, _ = aggregate_patch_scores_to_point_level(
        patch_scores_2d, ws_indices, seq_length, patch_size, num_patches, total_len, method='mean')
    point_recon, _ = aggregate_patch_scores_to_point_level(
        patch_recon_2d, ws_indices, seq_length, patch_size, num_patches, total_len, method='mean')
    point_disc, _ = aggregate_patch_scores_to_point_level(
        patch_disc_2d, ws_indices, seq_length, patch_size, num_patches, total_len, method='mean')
    point_student, _ = aggregate_patch_scores_to_point_level(
        patch_student_2d, ws_indices, seq_length, patch_size, num_patches, total_len, method='mean')

    point_scores = np.nan_to_num(point_scores, nan=0.0)
    point_recon = np.nan_to_num(point_recon, nan=0.0)
    point_disc = np.nan_to_num(point_disc, nan=0.0)
    point_student = np.nan_to_num(point_student, nan=0.0)

    result.update({
        'point_scores': point_scores,
        'point_labels': pt_labels_full,
        'point_recon': point_recon,
        'point_disc': point_disc,
        'point_student': point_student,
        'scores': point_scores,
        'labels': pt_labels_full,
        'window_start_indices': ws_indices,
        'total_length': total_len,
        'patch_recon_2d': patch_recon_2d,
        'patch_disc_2d': patch_disc_2d,
        'patch_student_2d': patch_student_2d,
    })

    return result


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
