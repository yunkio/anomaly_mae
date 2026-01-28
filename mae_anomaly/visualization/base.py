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
        mask_last_n=config.mask_last_n,
        split='test',
        train_ratio=config.sliding_window_train_ratio,
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
    - Respects config.inference_mode ('last_patch' or 'all_patches')
    - Use MSE (squared error) for consistency with training

    For 'last_patch' mode:
    - Mask only last mask_last_n positions
    - Use last_patch_labels as labels
    - One score per window

    For 'all_patches' mode:
    - Mask each patch one at a time (N forward passes)
    - Compute patch-level labels from point_labels
    - One score per patch (N scores per window)

    Args:
        model: Trained model
        dataloader: Test data loader
        config: Model configuration

    Returns:
        Dict containing scores, labels, recon_errors, discrepancies, sample_types
    """
    model.eval()
    device = config.device
    inference_mode = getattr(config, 'inference_mode', 'last_patch')
    patch_size = getattr(config, 'patch_size', 10)
    num_patches = getattr(config, 'num_patches', 10)

    all_labels = []
    all_recon_errors = []      # Teacher reconstruction error (teacher vs original)
    all_student_errors = []    # Student reconstruction error (student vs original)
    all_discrepancies = []     # Teacher-student difference
    all_sample_types = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Collecting predictions ({inference_mode})"):
            sequences, last_patch_labels, point_labels, sample_types, _ = batch
            sequences = sequences.to(device)
            batch_size, seq_length, num_features = sequences.shape

            if inference_mode == 'all_patches':
                # All patches mode: process patches in batches for memory efficiency
                patch_batch_size = 2  # Fixed: process 2 patches at a time (reduced for GPU memory)

                # Initialize result tensors
                patch_recon_scores = torch.zeros(batch_size, num_patches, device=device)
                patch_student_scores = torch.zeros(batch_size, num_patches, device=device)
                patch_disc_scores = torch.zeros(batch_size, num_patches, device=device)

                # Process patches in batches
                for batch_start in range(0, num_patches, patch_batch_size):
                    batch_end = min(batch_start + patch_batch_size, num_patches)
                    current_batch_patches = batch_end - batch_start

                    # Expand only for current patch batch
                    expanded = sequences.unsqueeze(1).expand(-1, current_batch_patches, -1, -1)
                    expanded = expanded.reshape(batch_size * current_batch_patches, seq_length, num_features)

                    # Create masks for current patch batch
                    masks = torch.ones(batch_size * current_batch_patches, seq_length, device=device)
                    for i, patch_idx in enumerate(range(batch_start, batch_end)):
                        start_pos = patch_idx * patch_size
                        end_pos = start_pos + patch_size
                        masks[i::current_batch_patches, start_pos:end_pos] = 0

                    # Forward pass for current batch
                    teacher_output, student_output, _ = model(expanded, masking_ratio=0.0, mask=masks)

                    # Compute errors
                    recon_error = ((teacher_output - expanded) ** 2).mean(dim=2)  # Teacher vs original
                    student_error = ((student_output - expanded) ** 2).mean(dim=2)  # Student vs original
                    discrepancy = ((teacher_output - student_output) ** 2).mean(dim=2)  # Teacher vs Student

                    # Reshape to (B, current_batch_patches, S)
                    recon_error = recon_error.view(batch_size, current_batch_patches, seq_length)
                    student_error = student_error.view(batch_size, current_batch_patches, seq_length)
                    discrepancy = discrepancy.view(batch_size, current_batch_patches, seq_length)

                    # Extract scores for each patch's masked region
                    for i, patch_idx in enumerate(range(batch_start, batch_end)):
                        start_pos = patch_idx * patch_size
                        end_pos = start_pos + patch_size
                        patch_recon_scores[:, patch_idx] = recon_error[:, i, start_pos:end_pos].mean(dim=1)
                        patch_student_scores[:, patch_idx] = student_error[:, i, start_pos:end_pos].mean(dim=1)
                        patch_disc_scores[:, patch_idx] = discrepancy[:, i, start_pos:end_pos].mean(dim=1)

                    # Clear intermediate tensors
                    del expanded, masks, teacher_output, student_output, recon_error, student_error, discrepancy

                # Compute patch-level labels from point_labels
                patch_labels = torch.zeros(batch_size, num_patches, dtype=torch.long)
                for p_idx in range(num_patches):
                    start_pos = p_idx * patch_size
                    end_pos = min(start_pos + patch_size, seq_length)
                    patch_has_anomaly = point_labels[:, start_pos:end_pos].any(dim=1)
                    patch_labels[:, p_idx] = patch_has_anomaly.long()

                # Flatten to 1D (each patch is a sample) and convert to numpy
                recon_flat = patch_recon_scores.flatten().cpu().numpy()
                student_flat = patch_student_scores.flatten().cpu().numpy()
                disc_flat = patch_disc_scores.flatten().cpu().numpy()
                labels = patch_labels.flatten()

                # Raw scores collected; scoring formula applied globally after concatenation

                # Compute per-patch sample_types based on patch labels (generalized approach)
                # sample_type: 2=anomaly (patch has anomaly), 1=disturbing (normal patch in anomaly window), 0=pure_normal
                window_has_anomaly = (point_labels.sum(dim=1) > 0).unsqueeze(1)  # (B, 1)
                patch_sample_types = torch.zeros_like(patch_labels)
                patch_sample_types[patch_labels == 1] = 2  # patches with anomaly
                # Normal patches in windows that have anomaly elsewhere → disturbing
                normal_in_anomaly_window = (patch_labels == 0) & window_has_anomaly
                patch_sample_types[normal_in_anomaly_window] = 1
                # Remaining (normal patches in normal windows) stay 0 (pure_normal)
                sample_types_expanded = patch_sample_types.flatten()

                all_labels.append(labels.numpy())
                all_recon_errors.append(recon_flat)
                all_student_errors.append(student_flat)
                all_discrepancies.append(disc_flat)
                all_sample_types.append(sample_types_expanded.numpy())

            else:
                # Last patch mode: mask only last patch (original behavior)
                mask = torch.ones(batch_size, seq_length, device=device)
                mask[:, -config.mask_last_n:] = 0

                # Forward pass with fixed mask
                teacher_output, student_output, _ = model(sequences, masking_ratio=0.0, mask=mask)

                # Compute errors using MSE
                recon_error = ((teacher_output - sequences) ** 2).mean(dim=2)  # Teacher vs original
                student_error = ((student_output - sequences) ** 2).mean(dim=2)  # Student vs original
                discrepancy = ((teacher_output - student_output) ** 2).mean(dim=2)  # Teacher vs Student

                # Compute scores on masked positions only
                masked_positions = (mask == 0)
                recon = (recon_error * masked_positions).sum(dim=1) / (masked_positions.sum(dim=1) + 1e-8)
                student = (student_error * masked_positions).sum(dim=1) / (masked_positions.sum(dim=1) + 1e-8)
                disc = (discrepancy * masked_positions).sum(dim=1) / (masked_positions.sum(dim=1) + 1e-8)

                # Raw scores collected; scoring formula applied globally after concatenation
                recon_np = recon.cpu().numpy()
                disc_np = disc.cpu().numpy()

                all_labels.append(last_patch_labels.numpy())
                all_recon_errors.append(recon_np)
                all_student_errors.append(student.cpu().numpy())
                all_discrepancies.append(disc_np)
                all_sample_types.append(sample_types.numpy())

    # Concatenate all raw scores
    recon_all = np.concatenate(all_recon_errors)
    disc_all = np.concatenate(all_discrepancies)
    student_all = np.concatenate(all_student_errors)
    labels_all = np.concatenate(all_labels)
    sample_types_all = np.concatenate(all_sample_types)

    # Apply scoring formula GLOBALLY (consistent with evaluator._apply_scoring_formula)
    score_mode = getattr(config, 'anomaly_score_mode', 'default')
    if score_mode == 'normalized':
        recon_z = (recon_all - recon_all.mean()) / (recon_all.std() + 1e-4)
        disc_z = (disc_all - disc_all.mean()) / (disc_all.std() + 1e-4)
        scores_all = recon_z + disc_z
    elif score_mode == 'adaptive':
        adaptive_lambda = recon_all.mean() / (disc_all.mean() + 1e-4)
        scores_all = recon_all + adaptive_lambda * disc_all
    elif score_mode == 'ratio_weighted':
        disc_median = np.median(disc_all) + 1e-4
        scores_all = recon_all * (1 + disc_all / disc_median)
    else:  # default
        scores_all = recon_all + config.lambda_disc * disc_all

    return {
        'scores': scores_all,
        'labels': labels_all,
        'recon_errors': recon_all,           # Teacher reconstruction error
        'student_errors': student_all,       # Student reconstruction error
        'discrepancies': disc_all,           # Teacher-student difference
        'sample_types': sample_types_all
    }


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

    if score_mode == 'normalized':
        # Z-score normalization
        recon_mean, recon_std = recon_all.mean(), recon_all.std() + 1e-8
        disc_mean, disc_std = disc_all.mean(), disc_all.std() + 1e-8
        recon_z = (recon_all - recon_mean) / recon_std
        disc_z = (disc_all - disc_mean) / disc_std
        scores = recon_z + disc_z

        # Min-shift for visualization: shift both to start from 0
        # This preserves relative differences while making interpretation intuitive
        min_val = min(recon_z.min(), disc_z.min())
        recon_contrib = recon_z - min_val
        disc_contrib = disc_z - min_val

        # Contribution ratios based on shifted (non-negative) values
        total = recon_contrib + disc_contrib + 1e-8
        recon_ratio = recon_contrib / total
        disc_ratio = disc_contrib / total

        mode_params.update({
            'recon_mean': recon_mean, 'recon_std': recon_std,
            'disc_mean': disc_mean, 'disc_std': disc_std,
            'min_shift': min_val
        })

    elif score_mode == 'adaptive':
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


def collect_detailed_data(model, dataloader, config) -> Dict:
    """Collect detailed data for analysis

    Uses the same evaluation method as evaluator.py:
    - Respects config.inference_mode ('last_patch' or 'all_patches')
    - Use MSE (squared error) for consistency with training

    Note: For 'all_patches' mode, this function returns aggregated data across
    all patches. For per-patch detailed data, use collect_predictions() instead.

    Args:
        model: Trained model
        dataloader: Test data loader
        config: Model configuration

    Returns:
        Dict containing detailed data (errors, reconstructions, etc.)
    """
    model.eval()
    device = config.device
    inference_mode = getattr(config, 'inference_mode', 'last_patch')
    patch_size = getattr(config, 'patch_size', 10)
    num_patches = getattr(config, 'num_patches', 10)

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
        for batch in tqdm(dataloader, desc=f"Collecting detailed data ({inference_mode})"):
            sequences, last_patch_labels, point_labels, sample_types, _ = batch
            sequences = sequences.to(device)
            batch_size, seq_length, num_features = sequences.shape

            if inference_mode == 'all_patches':
                # All patches mode: process patches in batches for memory efficiency
                patch_batch_size = 2  # Fixed: process 2 patches at a time (reduced for GPU memory)

                # Initialize final result tensors
                teacher_error_final = torch.zeros(batch_size, seq_length, device=device)
                student_error_final = torch.zeros(batch_size, seq_length, device=device)
                discrepancy_final = torch.zeros(batch_size, seq_length, device=device)
                teacher_recon_final = torch.zeros_like(sequences)
                student_recon_final = torch.zeros_like(sequences)

                # Process patches in batches
                for batch_start in range(0, num_patches, patch_batch_size):
                    batch_end = min(batch_start + patch_batch_size, num_patches)
                    current_batch_patches = batch_end - batch_start

                    # Expand only for current patch batch
                    expanded = sequences.unsqueeze(1).expand(-1, current_batch_patches, -1, -1)
                    expanded = expanded.reshape(batch_size * current_batch_patches, seq_length, num_features)

                    # Create masks for current patch batch
                    masks = torch.ones(batch_size * current_batch_patches, seq_length, device=device)
                    for i, patch_idx in enumerate(range(batch_start, batch_end)):
                        start_pos = patch_idx * patch_size
                        end_pos = start_pos + patch_size
                        masks[i::current_batch_patches, start_pos:end_pos] = 0

                    # Forward pass for current batch
                    teacher_output, student_output, _ = model(expanded, masking_ratio=0.0, mask=masks)

                    # Compute errors
                    teacher_error = ((teacher_output - expanded) ** 2).mean(dim=-1)
                    student_error = ((student_output - expanded) ** 2).mean(dim=-1)
                    discrepancy = ((teacher_output - student_output) ** 2).mean(dim=-1)

                    # Reshape to (B, current_batch_patches, S, F)
                    teacher_output = teacher_output.view(batch_size, current_batch_patches, seq_length, num_features)
                    student_output = student_output.view(batch_size, current_batch_patches, seq_length, num_features)
                    teacher_error = teacher_error.view(batch_size, current_batch_patches, seq_length)
                    student_error = student_error.view(batch_size, current_batch_patches, seq_length)
                    discrepancy = discrepancy.view(batch_size, current_batch_patches, seq_length)

                    # Assemble results for this batch of patches
                    for i, patch_idx in enumerate(range(batch_start, batch_end)):
                        start_pos = patch_idx * patch_size
                        end_pos = start_pos + patch_size
                        teacher_error_final[:, start_pos:end_pos] = teacher_error[:, i, start_pos:end_pos]
                        student_error_final[:, start_pos:end_pos] = student_error[:, i, start_pos:end_pos]
                        discrepancy_final[:, start_pos:end_pos] = discrepancy[:, i, start_pos:end_pos]
                        teacher_recon_final[:, start_pos:end_pos] = teacher_output[:, i, start_pos:end_pos]
                        student_recon_final[:, start_pos:end_pos] = student_output[:, i, start_pos:end_pos]

                    # Clear intermediate tensors to free memory
                    del expanded, masks, teacher_output, student_output, teacher_error, student_error, discrepancy

                # Combined mask (all positions were masked at some point)
                combined_mask = torch.zeros(batch_size, seq_length, device=device)

                all_teacher_errors.append(teacher_error_final.cpu().numpy())
                all_student_errors.append(student_error_final.cpu().numpy())
                all_discrepancies.append(discrepancy_final.cpu().numpy())
                all_masks.append(combined_mask.cpu().numpy())
                all_labels.append(last_patch_labels.numpy())
                all_point_labels.append(point_labels.numpy())
                all_originals.append(sequences.cpu().numpy())
                all_teacher_recons.append(teacher_recon_final.cpu().numpy())
                all_student_recons.append(student_recon_final.cpu().numpy())
                all_sample_types.append(sample_types.numpy())

            else:
                # Last patch mode (original behavior)
                mask = torch.ones(batch_size, seq_length, device=device)
                mask[:, -config.mask_last_n:] = 0

                # Forward pass with fixed mask
                teacher_output, student_output, _ = model(sequences, masking_ratio=0.0, mask=mask)

                # Compute errors using MSE
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

    # All outputs are window-level regardless of inference_mode
    # (collect_predictions handles patch-level labels/scores for metrics)
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


def collect_all_visualization_data(model, dataloader, config) -> Tuple[Dict, Dict]:
    """Collect both prediction and detailed data in a SINGLE forward pass.

    This function eliminates duplicate forward passes by collecting both:
    - Prediction data (scores, labels, errors for metrics)
    - Detailed data (reconstructions for visualization)

    Performance improvement: ~2x faster than calling collect_predictions()
    and collect_detailed_data() separately for all_patches mode.

    Args:
        model: Trained model
        dataloader: Test data loader
        config: Model configuration

    Returns:
        Tuple of (pred_data, detailed_data) where:
        - pred_data: Dict with scores, labels, recon_errors, student_errors, discrepancies, sample_types
        - detailed_data: Dict with teacher_errors, student_errors, discrepancies, masks,
                        labels, point_labels, originals, teacher_recons, student_recons, sample_types
    """
    model.eval()
    device = config.device
    inference_mode = getattr(config, 'inference_mode', 'last_patch')
    patch_size = getattr(config, 'patch_size', 10)
    num_patches = getattr(config, 'num_patches', 10)

    # Prediction data collectors
    all_scores = []
    all_pred_labels = []
    all_recon_errors = []
    all_student_errors_pred = []
    all_discrepancies_pred = []
    all_sample_types_pred = []

    # Detailed data collectors
    all_teacher_errors_det = []
    all_student_errors_det = []
    all_discrepancies_det = []
    all_masks = []
    all_det_labels = []
    all_point_labels = []
    all_originals = []
    all_teacher_recons = []
    all_student_recons = []
    all_sample_types_det = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Collecting all data ({inference_mode})"):
            sequences, last_patch_labels, point_labels, sample_types, _ = batch
            sequences = sequences.to(device)
            batch_size, seq_length, num_features = sequences.shape

            if inference_mode == 'all_patches':
                # All patches mode: process patches in batches for memory efficiency
                patch_batch_size = 2

                # Initialize prediction result tensors
                patch_recon_scores = torch.zeros(batch_size, num_patches, device=device)
                patch_student_scores = torch.zeros(batch_size, num_patches, device=device)
                patch_disc_scores = torch.zeros(batch_size, num_patches, device=device)

                # Initialize detailed result tensors
                teacher_error_final = torch.zeros(batch_size, seq_length, device=device)
                student_error_final = torch.zeros(batch_size, seq_length, device=device)
                discrepancy_final = torch.zeros(batch_size, seq_length, device=device)
                teacher_recon_final = torch.zeros_like(sequences)
                student_recon_final = torch.zeros_like(sequences)

                # Process patches in batches (SINGLE forward pass per batch)
                for batch_start in range(0, num_patches, patch_batch_size):
                    batch_end = min(batch_start + patch_batch_size, num_patches)
                    current_batch_patches = batch_end - batch_start

                    # Expand for current patch batch
                    expanded = sequences.unsqueeze(1).expand(-1, current_batch_patches, -1, -1)
                    expanded = expanded.reshape(batch_size * current_batch_patches, seq_length, num_features)

                    # Create masks for current patch batch
                    masks = torch.ones(batch_size * current_batch_patches, seq_length, device=device)
                    for i, patch_idx in enumerate(range(batch_start, batch_end)):
                        start_pos = patch_idx * patch_size
                        end_pos = start_pos + patch_size
                        masks[i::current_batch_patches, start_pos:end_pos] = 0

                    # SINGLE forward pass for this batch
                    teacher_output, student_output, _ = model(expanded, masking_ratio=0.0, mask=masks)

                    # Compute errors (used for both prediction and detailed data)
                    recon_error = ((teacher_output - expanded) ** 2).mean(dim=2)
                    student_error = ((student_output - expanded) ** 2).mean(dim=2)
                    discrepancy = ((teacher_output - student_output) ** 2).mean(dim=2)

                    # Reshape errors
                    recon_error_view = recon_error.view(batch_size, current_batch_patches, seq_length)
                    student_error_view = student_error.view(batch_size, current_batch_patches, seq_length)
                    discrepancy_view = discrepancy.view(batch_size, current_batch_patches, seq_length)

                    # Reshape outputs for detailed data
                    teacher_output_view = teacher_output.view(batch_size, current_batch_patches, seq_length, num_features)
                    student_output_view = student_output.view(batch_size, current_batch_patches, seq_length, num_features)

                    # Extract data for each patch
                    for i, patch_idx in enumerate(range(batch_start, batch_end)):
                        start_pos = patch_idx * patch_size
                        end_pos = start_pos + patch_size

                        # Prediction data: patch-level scores
                        patch_recon_scores[:, patch_idx] = recon_error_view[:, i, start_pos:end_pos].mean(dim=1)
                        patch_student_scores[:, patch_idx] = student_error_view[:, i, start_pos:end_pos].mean(dim=1)
                        patch_disc_scores[:, patch_idx] = discrepancy_view[:, i, start_pos:end_pos].mean(dim=1)

                        # Detailed data: full sequence reconstruction
                        teacher_error_final[:, start_pos:end_pos] = recon_error_view[:, i, start_pos:end_pos]
                        student_error_final[:, start_pos:end_pos] = student_error_view[:, i, start_pos:end_pos]
                        discrepancy_final[:, start_pos:end_pos] = discrepancy_view[:, i, start_pos:end_pos]
                        teacher_recon_final[:, start_pos:end_pos] = teacher_output_view[:, i, start_pos:end_pos]
                        student_recon_final[:, start_pos:end_pos] = student_output_view[:, i, start_pos:end_pos]

                    # Clear intermediate tensors
                    del expanded, masks, teacher_output, student_output, recon_error, student_error, discrepancy

                # === Prediction data processing ===
                patch_labels = torch.zeros(batch_size, num_patches, dtype=torch.long)
                for p_idx in range(num_patches):
                    start_pos = p_idx * patch_size
                    end_pos = min(start_pos + patch_size, seq_length)
                    patch_has_anomaly = point_labels[:, start_pos:end_pos].any(dim=1)
                    patch_labels[:, p_idx] = patch_has_anomaly.long()

                scores = (patch_recon_scores + config.lambda_disc * patch_disc_scores).flatten()
                labels_flat = patch_labels.flatten()
                recon_flat = patch_recon_scores.flatten()
                student_flat = patch_student_scores.flatten()
                disc_flat = patch_disc_scores.flatten()

                window_has_anomaly = (point_labels.sum(dim=1) > 0).unsqueeze(1)
                patch_sample_types = torch.zeros_like(patch_labels)
                patch_sample_types[patch_labels == 1] = 2
                normal_in_anomaly_window = (patch_labels == 0) & window_has_anomaly
                patch_sample_types[normal_in_anomaly_window] = 1
                sample_types_expanded = patch_sample_types.flatten()

                all_scores.append(scores.cpu().numpy())
                all_pred_labels.append(labels_flat.numpy())
                all_recon_errors.append(recon_flat.cpu().numpy())
                all_student_errors_pred.append(student_flat.cpu().numpy())
                all_discrepancies_pred.append(disc_flat.cpu().numpy())
                all_sample_types_pred.append(sample_types_expanded.numpy())

                # === Detailed data processing ===
                combined_mask = torch.zeros(batch_size, seq_length, device=device)
                all_teacher_errors_det.append(teacher_error_final.cpu().numpy())
                all_student_errors_det.append(student_error_final.cpu().numpy())
                all_discrepancies_det.append(discrepancy_final.cpu().numpy())
                all_masks.append(combined_mask.cpu().numpy())
                all_det_labels.append(last_patch_labels.numpy())
                all_point_labels.append(point_labels.numpy())
                all_originals.append(sequences.cpu().numpy())
                all_teacher_recons.append(teacher_recon_final.cpu().numpy())
                all_student_recons.append(student_recon_final.cpu().numpy())
                all_sample_types_det.append(sample_types.numpy())

            else:
                # Last patch mode: SINGLE forward pass
                mask = torch.ones(batch_size, seq_length, device=device)
                mask[:, -config.mask_last_n:] = 0

                teacher_output, student_output, _ = model(sequences, masking_ratio=0.0, mask=mask)

                # Compute errors
                recon_error = ((teacher_output - sequences) ** 2).mean(dim=2)
                student_error = ((student_output - sequences) ** 2).mean(dim=2)
                discrepancy = ((teacher_output - student_output) ** 2).mean(dim=2)

                # === Prediction data ===
                masked_positions = (mask == 0)
                recon = (recon_error * masked_positions).sum(dim=1) / (masked_positions.sum(dim=1) + 1e-8)
                student = (student_error * masked_positions).sum(dim=1) / (masked_positions.sum(dim=1) + 1e-8)
                disc = (discrepancy * masked_positions).sum(dim=1) / (masked_positions.sum(dim=1) + 1e-8)
                scores = recon + config.lambda_disc * disc

                all_scores.append(scores.cpu().numpy())
                all_pred_labels.append(last_patch_labels.numpy())
                all_recon_errors.append(recon.cpu().numpy())
                all_student_errors_pred.append(student.cpu().numpy())
                all_discrepancies_pred.append(disc.cpu().numpy())
                all_sample_types_pred.append(sample_types.numpy())

                # === Detailed data ===
                teacher_error_det = ((teacher_output - sequences) ** 2).mean(dim=-1)
                student_error_det = ((student_output - sequences) ** 2).mean(dim=-1)
                discrepancy_det = ((teacher_output - student_output) ** 2).mean(dim=-1)

                all_teacher_errors_det.append(teacher_error_det.cpu().numpy())
                all_student_errors_det.append(student_error_det.cpu().numpy())
                all_discrepancies_det.append(discrepancy_det.cpu().numpy())
                all_masks.append(mask.cpu().numpy())
                all_det_labels.append(last_patch_labels.numpy())
                all_point_labels.append(point_labels.numpy())
                all_originals.append(sequences.cpu().numpy())
                all_teacher_recons.append(teacher_output.cpu().numpy())
                all_student_recons.append(student_output.cpu().numpy())
                all_sample_types_det.append(sample_types.numpy())

    pred_data = {
        'scores': np.concatenate(all_scores),
        'labels': np.concatenate(all_pred_labels),
        'recon_errors': np.concatenate(all_recon_errors),
        'student_errors': np.concatenate(all_student_errors_pred),
        'discrepancies': np.concatenate(all_discrepancies_pred),
        'sample_types': np.concatenate(all_sample_types_pred)
    }

    detailed_data = {
        'teacher_errors': np.concatenate(all_teacher_errors_det),
        'student_errors': np.concatenate(all_student_errors_det),
        'discrepancies': np.concatenate(all_discrepancies_det),
        'masks': np.concatenate(all_masks),
        'labels': np.concatenate(all_det_labels),
        'point_labels': np.concatenate(all_point_labels),
        'originals': np.concatenate(all_originals),
        'teacher_recons': np.concatenate(all_teacher_recons),
        'student_recons': np.concatenate(all_student_recons),
        'sample_types': np.concatenate(all_sample_types_det),
    }

    return pred_data, detailed_data


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
