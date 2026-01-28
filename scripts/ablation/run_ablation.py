#!/usr/bin/env python
"""
Unified Ablation Study Runner
=============================

Runs ablation experiments based on configuration files.

Usage:
    python scripts/ablation/run_ablation.py --config configs/20260126_160417_phase2.py
    python scripts/ablation/run_ablation.py --config configs/20260126_005356_phase1.py --skip-existing

Config file format:
    PHASE_NAME: str           - Phase identifier
    PHASE_DESCRIPTION: str    - Description of the phase
    BASE_CONFIG: dict         - Base configuration (optional)
    EXPERIMENTS: list         - List of experiment configurations
    SCORING_MODES: list       - Scoring modes to evaluate (default: ['default'])
    INFERENCE_MODES: list     - Inference modes (default: ['last_patch'])
"""

import os
import sys
import json
import time
import shutil
import importlib.util
import multiprocessing as mp
from copy import deepcopy
from datetime import datetime
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Suppress transformer nested_tensor warning (batch_first=False is intentional for model compatibility)
warnings.filterwarnings('ignore', message='.*enable_nested_tensor.*')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mae_anomaly import (
    Config, set_seed,
    SlidingWindowTimeSeriesGenerator, SlidingWindowDataset,
    NormalDataComplexity,
    SelfDistilledMAEMultivariate, SelfDistillationLoss,
    Trainer, Evaluator, SLIDING_ANOMALY_TYPE_NAMES
)


# =============================================================================
# Dataset Statistics and Documentation
# =============================================================================

def save_dataset_info(
    output_dir: str,
    config: Config,
    signals: np.ndarray,
    point_labels: np.ndarray,
    anomaly_regions: list,
    train_ratio: float
):
    """Save dataset statistics to dataset.md file."""
    total_length = len(signals)
    train_end = int(total_length * train_ratio)
    test_length = total_length - train_end

    # Count anomaly timestamps by type
    anomaly_type_per_point = np.zeros(total_length, dtype=int)
    for region in anomaly_regions:
        atype = region.anomaly_type
        if isinstance(atype, str):
            atype_idx = SLIDING_ANOMALY_TYPE_NAMES.index(atype)
        else:
            atype_idx = atype
        anomaly_type_per_point[region.start:region.end] = atype_idx

    # Train/Test timestamp counts by type
    train_ts_counts = {name: 0 for name in SLIDING_ANOMALY_TYPE_NAMES[1:]}
    test_ts_counts = {name: 0 for name in SLIDING_ANOMALY_TYPE_NAMES[1:]}

    for t in range(total_length):
        atype_idx = anomaly_type_per_point[t]
        if atype_idx > 0:
            atype = SLIDING_ANOMALY_TYPE_NAMES[atype_idx]
            if t < train_end:
                train_ts_counts[atype] += 1
            else:
                test_ts_counts[atype] += 1

    train_total_anomaly = sum(train_ts_counts.values())
    test_total_anomaly = sum(test_ts_counts.values())

    # Region counts by type
    train_region_counts = {name: 0 for name in SLIDING_ANOMALY_TYPE_NAMES[1:]}
    test_region_counts = {name: 0 for name in SLIDING_ANOMALY_TYPE_NAMES[1:]}

    for region in anomaly_regions:
        atype = region.anomaly_type
        if isinstance(atype, int):
            atype = SLIDING_ANOMALY_TYPE_NAMES[atype]
        if region.start < train_end:
            train_region_counts[atype] += 1
        else:
            test_region_counts[atype] += 1

    # Window distribution (train)
    train_dataset = SlidingWindowDataset(
        signals=signals,
        point_labels=point_labels,
        anomaly_regions=anomaly_regions,
        window_size=config.seq_length,
        stride=config.sliding_window_stride,
        mask_last_n=config.mask_last_n,
        split='train',
        train_ratio=train_ratio,
        seed=config.random_seed
    )

    train_pure = train_disturb = train_anom = 0
    for i in range(len(train_dataset)):
        _, _, _, sample_type, _ = train_dataset[i]
        if sample_type == 0:
            train_pure += 1
        elif sample_type == 1:
            train_disturb += 1
        else:
            train_anom += 1

    # Window distribution (test)
    test_dataset = SlidingWindowDataset(
        signals=signals,
        point_labels=point_labels,
        anomaly_regions=anomaly_regions,
        window_size=config.seq_length,
        stride=config.sliding_window_test_stride,
        mask_last_n=config.mask_last_n,
        split='test',
        train_ratio=train_ratio,
        seed=config.random_seed
    )

    test_pure = test_disturb = test_anom = 0
    for i in range(len(test_dataset)):
        _, _, _, sample_type, _ = test_dataset[i]
        if sample_type == 0:
            test_pure += 1
        elif sample_type == 1:
            test_disturb += 1
        else:
            test_anom += 1

    # Window distribution for window_size=500 (for comparison)
    window_size_500 = 500
    mask_last_n_500 = 50  # 10% of window size

    train_dataset_500 = SlidingWindowDataset(
        signals=signals,
        point_labels=point_labels,
        anomaly_regions=anomaly_regions,
        window_size=window_size_500,
        stride=config.sliding_window_stride,
        mask_last_n=mask_last_n_500,
        split='train',
        train_ratio=train_ratio,
        seed=config.random_seed
    )

    train_pure_500 = train_disturb_500 = train_anom_500 = 0
    for i in range(len(train_dataset_500)):
        _, _, _, sample_type, _ = train_dataset_500[i]
        if sample_type == 0:
            train_pure_500 += 1
        elif sample_type == 1:
            train_disturb_500 += 1
        else:
            train_anom_500 += 1

    test_dataset_500 = SlidingWindowDataset(
        signals=signals,
        point_labels=point_labels,
        anomaly_regions=anomaly_regions,
        window_size=window_size_500,
        stride=config.sliding_window_test_stride,
        mask_last_n=mask_last_n_500,
        split='test',
        train_ratio=train_ratio,
        seed=config.random_seed
    )

    test_pure_500 = test_disturb_500 = test_anom_500 = 0
    for i in range(len(test_dataset_500)):
        _, _, _, sample_type, _ = test_dataset_500[i]
        if sample_type == 0:
            test_pure_500 += 1
        elif sample_type == 1:
            test_disturb_500 += 1
        else:
            test_anom_500 += 1

    # Generate markdown content
    md_content = f"""# Dataset Information

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Configuration

| Parameter | Value |
|-----------|-------|
| `sliding_window_total_length` | {config.sliding_window_total_length:,} |
| `sliding_window_train_ratio` | {train_ratio:.4f} |
| `sliding_window_stride` (train) | {config.sliding_window_stride} |
| `sliding_window_test_stride` | {config.sliding_window_test_stride} |
| `anomaly_interval_scale` | {config.anomaly_interval_scale} |
| `seq_length` (window_size) | {config.seq_length} |
| `mask_last_n` | {config.mask_last_n} |
| `num_features` | {config.num_features} |
| `random_seed` | {config.random_seed} |

---

## Timestamp Statistics

| Split | Total | Normal | Anomaly | Anomaly % |
|-------|-------|--------|---------|-----------|
| **Train** | {train_end:,} | {train_end - train_total_anomaly:,} | {train_total_anomaly:,} | {100*train_total_anomaly/train_end:.2f}% |
| **Test** | {test_length:,} | {test_length - test_total_anomaly:,} | {test_total_anomaly:,} | {100*test_total_anomaly/test_length:.2f}% |

---

## Train Data

### Anomaly Type별 Timestamp 길이

| Anomaly Type | Timestamps |
|--------------|------------|
"""

    for atype, count in train_ts_counts.items():
        md_content += f"| {atype} | {count:,} |\n"
    md_content += f"| **Total** | **{train_total_anomaly:,}** |\n"

    md_content += f"""
### Anomaly Type별 Region 갯수

| Anomaly Type | Regions |
|--------------|---------|
"""

    for atype, count in train_region_counts.items():
        md_content += f"| {atype} | {count} |\n"
    md_content += f"| **Total** | **{sum(train_region_counts.values())}** |\n"

    md_content += f"""
### Window 분포 (window_size={config.seq_length}, stride={config.sliding_window_stride})

| Window Type | Count | Ratio |
|-------------|-------|-------|
| pure_normal | {train_pure:,} | {100*train_pure/len(train_dataset):.2f}% |
| disturbing_normal | {train_disturb:,} | {100*train_disturb/len(train_dataset):.2f}% |
| anomaly | {train_anom:,} | {100*train_anom/len(train_dataset):.2f}% |
| **Total** | **{len(train_dataset):,}** | 100% |

### Window 분포 (window_size=500, stride={config.sliding_window_stride})

| Window Type | Count | Ratio |
|-------------|-------|-------|
| pure_normal | {train_pure_500:,} | {100*train_pure_500/len(train_dataset_500):.2f}% |
| disturbing_normal | {train_disturb_500:,} | {100*train_disturb_500/len(train_dataset_500):.2f}% |
| anomaly | {train_anom_500:,} | {100*train_anom_500/len(train_dataset_500):.2f}% |
| **Total** | **{len(train_dataset_500):,}** | 100% |

---

## Test Data

### Anomaly Type별 Timestamp 길이

| Anomaly Type | Timestamps |
|--------------|------------|
"""

    for atype, count in test_ts_counts.items():
        md_content += f"| {atype} | {count:,} |\n"
    md_content += f"| **Total** | **{test_total_anomaly:,}** |\n"

    md_content += f"""
### Anomaly Type별 Region 갯수

| Anomaly Type | Regions |
|--------------|---------|
"""

    for atype, count in test_region_counts.items():
        md_content += f"| {atype} | {count} |\n"
    md_content += f"| **Total** | **{sum(test_region_counts.values())}** |\n"

    md_content += f"""
### Window 분포 (window_size={config.seq_length}, stride={config.sliding_window_test_stride})

| Window Type | Count | Ratio |
|-------------|-------|-------|
| pure_normal | {test_pure:,} | {100*test_pure/len(test_dataset):.2f}% |
| disturbing_normal | {test_disturb:,} | {100*test_disturb/len(test_dataset):.2f}% |
| anomaly | {test_anom:,} | {100*test_anom/len(test_dataset):.2f}% |
| **Total** | **{len(test_dataset):,}** | 100% |

### Window 분포 (window_size=500, stride={config.sliding_window_test_stride})

| Window Type | Count | Ratio |
|-------------|-------|-------|
| pure_normal | {test_pure_500:,} | {100*test_pure_500/len(test_dataset_500):.2f}% |
| disturbing_normal | {test_disturb_500:,} | {100*test_disturb_500/len(test_dataset_500):.2f}% |
| anomaly | {test_anom_500:,} | {100*test_anom_500/len(test_dataset_500):.2f}% |
| **Total** | **{len(test_dataset_500):,}** | 100% |
"""

    # Save to file
    dataset_md_path = os.path.join(output_dir, 'dataset.md')
    with open(dataset_md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)

    print(f"  Dataset info saved to: {dataset_md_path}")


# =============================================================================
# Loss Statistics and Separation Metrics
# =============================================================================

def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size between two groups.

    Cohen's d = (mean1 - mean2) / pooled_std
    Measures how well-separated two distributions are.
    |d| < 0.2: negligible, 0.2-0.5: small, 0.5-0.8: medium, > 0.8: large
    """
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return 0.0

    mean1, mean2 = group1.mean(), group2.mean()
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std < 1e-10:
        return 0.0

    return (mean2 - mean1) / pooled_std  # Positive if mean2 > mean1


def compute_loss_statistics(detailed_losses: Dict) -> Dict:
    """Compute detailed loss statistics including per-category metrics and separation.

    Args:
        detailed_losses: Dict from evaluator.compute_detailed_losses()

    Returns:
        Dict with loss statistics and separation metrics
    """
    recon_loss = detailed_losses['reconstruction_loss']
    disc_loss = detailed_losses['discrepancy_loss']
    sample_types = detailed_losses['sample_types']

    # Sample type masks
    pure_normal_mask = sample_types == 0
    disturbing_mask = sample_types == 1
    anomaly_mask = sample_types == 2
    normal_mask = (sample_types == 0) | (sample_types == 1)  # pure + disturbing

    stats = {}

    # Overall losses
    stats['reconstruction_loss'] = float(recon_loss.mean())
    stats['discrepancy_loss'] = float(disc_loss.mean())

    # Per-category reconstruction losses
    stats['recon_normal'] = float(recon_loss[normal_mask].mean()) if normal_mask.sum() > 0 else 0.0
    stats['recon_anomaly'] = float(recon_loss[anomaly_mask].mean()) if anomaly_mask.sum() > 0 else 0.0
    stats['recon_pure_normal'] = float(recon_loss[pure_normal_mask].mean()) if pure_normal_mask.sum() > 0 else 0.0
    stats['recon_disturbing'] = float(recon_loss[disturbing_mask].mean()) if disturbing_mask.sum() > 0 else 0.0

    # Per-category discrepancy losses
    stats['disc_normal'] = float(disc_loss[normal_mask].mean()) if normal_mask.sum() > 0 else 0.0
    stats['disc_anomaly'] = float(disc_loss[anomaly_mask].mean()) if anomaly_mask.sum() > 0 else 0.0
    stats['disc_pure_normal'] = float(disc_loss[pure_normal_mask].mean()) if pure_normal_mask.sum() > 0 else 0.0
    stats['disc_disturbing'] = float(disc_loss[disturbing_mask].mean()) if disturbing_mask.sum() > 0 else 0.0

    # Standard deviations (for separation metrics)
    stats['recon_normal_std'] = float(recon_loss[normal_mask].std()) if normal_mask.sum() > 1 else 0.0
    stats['recon_anomaly_std'] = float(recon_loss[anomaly_mask].std()) if anomaly_mask.sum() > 1 else 0.0
    stats['disc_normal_std'] = float(disc_loss[normal_mask].std()) if normal_mask.sum() > 1 else 0.0
    stats['disc_anomaly_std'] = float(disc_loss[anomaly_mask].std()) if anomaly_mask.sum() > 1 else 0.0
    stats['disc_disturbing_std'] = float(disc_loss[disturbing_mask].std()) if disturbing_mask.sum() > 1 else 0.0

    # Ratio metrics (higher = better separation)
    stats['disc_ratio'] = stats['disc_anomaly'] / (stats['disc_normal'] + 1e-8)
    stats['disc_ratio_disturbing'] = stats['disc_anomaly'] / (stats['disc_disturbing'] + 1e-8)
    stats['recon_ratio'] = stats['recon_anomaly'] / (stats['recon_normal'] + 1e-8)

    # Cohen's d separation metrics (higher = better separation)
    # Positive values mean anomaly has higher values than normal (expected behavior)
    if normal_mask.sum() > 0 and anomaly_mask.sum() > 0:
        stats['disc_cohens_d_normal_vs_anomaly'] = compute_cohens_d(
            disc_loss[normal_mask], disc_loss[anomaly_mask]
        )
        stats['recon_cohens_d_normal_vs_anomaly'] = compute_cohens_d(
            recon_loss[normal_mask], recon_loss[anomaly_mask]
        )
    else:
        stats['disc_cohens_d_normal_vs_anomaly'] = 0.0
        stats['recon_cohens_d_normal_vs_anomaly'] = 0.0

    if disturbing_mask.sum() > 0 and anomaly_mask.sum() > 0:
        stats['disc_cohens_d_disturbing_vs_anomaly'] = compute_cohens_d(
            disc_loss[disturbing_mask], disc_loss[anomaly_mask]
        )
        stats['recon_cohens_d_disturbing_vs_anomaly'] = compute_cohens_d(
            recon_loss[disturbing_mask], recon_loss[anomaly_mask]
        )
    else:
        stats['disc_cohens_d_disturbing_vs_anomaly'] = 0.0
        stats['recon_cohens_d_disturbing_vs_anomaly'] = 0.0

    return stats


# =============================================================================
# Background Visualization Support
# =============================================================================

_background_viz_processes = []
_max_concurrent_viz = 1


def _cleanup_finished_processes():
    """Remove finished processes from the list."""
    global _background_viz_processes
    _background_viz_processes = [p for p in _background_viz_processes if p.is_alive()]


def _wait_for_slot():
    """Wait until a slot is available for a new visualization process."""
    global _background_viz_processes
    while True:
        _cleanup_finished_processes()
        if len(_background_viz_processes) < _max_concurrent_viz:
            break
        time.sleep(1)


def _background_viz_worker(args):
    """Worker function for background visualization."""
    model_path, scoring_modes, inference_modes, output_dirs, num_test = args

    import traceback
    import sys

    try:
        import matplotlib
        matplotlib.use('Agg')

        from mae_anomaly.visualization import (
            setup_style, load_best_model, BestModelVisualizer,
            collect_all_visualization_data
        )

        setup_style()
    except Exception as e:
        print(f"[BG-VIZ ERROR] Import failed: {e}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return False

    try:
        print(f"[BG-VIZ] Loading model from {model_path}...", flush=True)
        model, config, test_loader, _ = load_best_model(model_path, num_test)

        pred_cache = {}
        detailed_cache = {}

        for inference_mode in inference_modes:
            print(f"[BG-VIZ] Collecting all data for {inference_mode}...", flush=True)
            config.inference_mode = inference_mode
            # Use merged function for ~2x speedup (single forward pass instead of two)
            pred_cache[inference_mode], detailed_cache[inference_mode] = collect_all_visualization_data(
                model, test_loader, config
            )

        invariant_plots = ['best_model_reconstruction.png', 'learning_curve.png', 'case_study_gallery.png']

        for inference_mode in inference_modes:
            inference_suffix = 'last' if inference_mode == 'last_patch' else 'all'
            config.inference_mode = inference_mode
            pred_data = pred_cache[inference_mode]
            detailed_data = detailed_cache[inference_mode]

            first_viz_dir = None

            for idx, scoring_mode in enumerate(scoring_modes):
                result_key = f"{scoring_mode}_{inference_suffix}"
                exp_dir = output_dirs.get(result_key)
                if not exp_dir:
                    continue

                viz_dir = os.path.join(exp_dir, 'visualization', 'best_model')
                os.makedirs(viz_dir, exist_ok=True)
                print(f"[BG-VIZ] Generating visualizations for {result_key} -> {viz_dir}", flush=True)

                history_path = os.path.join(exp_dir, 'training_histories.json')
                history = None
                if os.path.exists(history_path):
                    with open(history_path) as f:
                        history_data = json.load(f)
                        history = history_data.get('0', {})

                visualizer = BestModelVisualizer(
                    model, config, test_loader, viz_dir,
                    pred_data=pred_data.copy(),
                    detailed_data=detailed_data
                )
                visualizer.recompute_scores(scoring_mode)

                if idx == 0:
                    first_viz_dir = viz_dir
                    visualizer.generate_all(experiment_dir=exp_dir, history=history)
                else:
                    for plot_file in invariant_plots:
                        src = os.path.join(first_viz_dir, plot_file)
                        if os.path.exists(src):
                            shutil.copy2(src, viz_dir)
                    visualizer.plot_roc_curve()
                    visualizer.plot_confusion_matrix()
                    visualizer.plot_score_contribution_analysis(exp_dir)
                    visualizer.plot_detection_examples()
                    visualizer.plot_summary_statistics()
                    visualizer.plot_pure_vs_disturbing_normal()
                    visualizer.plot_discrepancy_trend()
                    visualizer.plot_hardest_samples()
                    visualizer.plot_performance_by_anomaly_type(exp_dir)
                    visualizer.plot_score_distribution_by_type(exp_dir)
                    visualizer.plot_score_contribution_epoch_trends(exp_dir, history)

                print(f"[BG-VIZ] Completed {result_key}", flush=True)

        print(f"[BG-VIZ] All visualizations completed successfully!", flush=True)
        return True

    except Exception as e:
        print(f"[BG-VIZ ERROR] Visualization failed: {e}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return False


def run_visualization_background(
    model_path: str,
    scoring_modes: List[str],
    inference_modes: List[str],
    output_dirs: Dict[str, str],
    num_test: int = 500
):
    """Start visualization in background process (non-blocking)."""
    _wait_for_slot()

    ctx = mp.get_context('spawn')
    args = (model_path, scoring_modes, inference_modes, output_dirs, num_test)
    process = ctx.Process(target=_background_viz_worker, args=(args,))
    process.start()
    _background_viz_processes.append(process)
    return process


def wait_for_background_visualizations():
    """Wait for all background visualization processes to complete."""
    global _background_viz_processes
    for process in _background_viz_processes:
        process.join()
    _background_viz_processes = []


# =============================================================================
# Config Loading
# =============================================================================

def load_config_module(config_path: str):
    """Load configuration from a Python file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    spec = importlib.util.spec_from_file_location("ablation_config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Validate required fields
    if not hasattr(module, 'EXPERIMENTS'):
        raise ValueError("Config must define EXPERIMENTS list")

    # Get optional fields with defaults
    base_config = getattr(module, 'BASE_CONFIG', {})

    # Determine mask_settings: respect config file, don't force both modes
    if hasattr(module, 'MASK_SETTINGS'):
        mask_settings = module.MASK_SETTINGS
    elif 'mask_after_encoder' in base_config:
        # Use BASE_CONFIG's mask_after_encoder as the single setting
        mask_settings = [base_config['mask_after_encoder']]
    else:
        # Default to False (mask_before) if nothing specified
        mask_settings = [False]

    config = {
        'phase_name': getattr(module, 'PHASE_NAME', 'ablation'),
        'phase_description': getattr(module, 'PHASE_DESCRIPTION', ''),
        'base_config': base_config,
        'experiments': module.EXPERIMENTS,
        'scoring_modes': getattr(module, 'SCORING_MODES', ['default']),
        'inference_modes': getattr(module, 'INFERENCE_MODES', ['last_patch']),
        'mask_settings': mask_settings,
    }

    return config


# =============================================================================
# Single Experiment Runner
# =============================================================================

class SingleExperimentRunner:
    """Run a single experiment configuration."""

    def __init__(
        self,
        exp_config: Dict,
        output_base_dir: str,
        signals: np.ndarray,
        point_labels: np.ndarray,
        anomaly_regions: list,
        train_ratio: float = None,  # None = use config.sliding_window_train_ratio
        point_aggregation_method: str = 'voting',
        scoring_modes: List[str] = None,
        inference_modes: List[str] = None
    ):
        self.exp_config = exp_config
        self.output_base_dir = output_base_dir
        self.signals = signals
        self.point_labels = point_labels
        self.anomaly_regions = anomaly_regions
        self._train_ratio_override = train_ratio  # Store override value
        self.point_aggregation_method = point_aggregation_method
        self.scoring_modes = scoring_modes or ['default']
        self.inference_modes = inference_modes or ['last_patch']

    def _create_config(self) -> Config:
        """Create Config object from experiment configuration."""
        config = Config()

        for key, value in self.exp_config.items():
            if key == 'name':
                continue
            if hasattr(config, key):
                setattr(config, key, value)

        config.point_aggregation_method = self.point_aggregation_method

        return config

    @property
    def train_ratio(self) -> float:
        """Get train ratio from override or config default."""
        if self._train_ratio_override is not None:
            return self._train_ratio_override
        return Config().sliding_window_train_ratio

    def run(self) -> Dict[str, Dict]:
        """Run the experiment with all scoring/inference mode combinations."""
        config = self._create_config()
        set_seed(config.random_seed)

        # No downsampling for test - use all windows with stride=1 for PA%K evaluation

        train_dataset = SlidingWindowDataset(
            signals=self.signals,
            point_labels=self.point_labels,
            anomaly_regions=self.anomaly_regions,
            window_size=config.seq_length,
            stride=config.sliding_window_stride,  # Train stride (default 11)
            mask_last_n=config.mask_last_n,
            split='train',
            train_ratio=self.train_ratio,
            seed=config.random_seed
        )

        test_dataset = SlidingWindowDataset(
            signals=self.signals,
            point_labels=self.point_labels,
            anomaly_regions=self.anomaly_regions,
            window_size=config.seq_length,
            stride=config.sliding_window_test_stride,  # Test stride=1 for PA%K
            mask_last_n=config.mask_last_n,
            split='test',
            train_ratio=self.train_ratio,
            seed=config.random_seed
        )

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

        # Set primary scoring mode BEFORE training
        config.anomaly_score_mode = self.scoring_modes[0]

        train_start = time.time()
        model = SelfDistilledMAEMultivariate(config)
        trainer = Trainer(model, config, train_loader, test_loader, verbose=False)
        trainer.train()
        train_time = time.time() - train_start

        history = trainer.history

        all_results = {}
        exp_name = self.exp_config.get('name', 'experiment')
        mask_suffix = 'mask_after' if config.mask_after_encoder else 'mask_before'

        for inference_mode in self.inference_modes:
            inference_suffix = 'all' if inference_mode == 'all_patches' else 'last'
            config.inference_mode = inference_mode

            evaluator = Evaluator(model, config, test_loader, test_dataset=test_dataset)

            for scoring_mode in self.scoring_modes:
                config.anomaly_score_mode = scoring_mode
                # Note: Don't clear cache here - raw scores (recon, disc) are independent of scoring_mode
                # Clearing cache would cause redundant forward passes

                eval_start = time.time()
                metrics = evaluator.evaluate()
                eval_time = time.time() - eval_start

                # Compute metrics by score type (disc only, teacher recon only, student recon only)
                disc_metrics = evaluator.evaluate_by_score_type('disc')
                teacher_recon_metrics = evaluator.evaluate_by_score_type('teacher_recon')
                student_recon_metrics = evaluator.evaluate_by_score_type('student_recon')

                result_key = f"{scoring_mode}_{inference_suffix}"
                output_dir = os.path.join(
                    self.output_base_dir,
                    f"{exp_name}_{mask_suffix}_{result_key}"
                )
                os.makedirs(output_dir, exist_ok=True)

                # Save model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': asdict(config),
                    'metrics': metrics
                }, os.path.join(output_dir, 'best_model.pt'))

                # Save config
                with open(os.path.join(output_dir, 'best_config.json'), 'w') as f:
                    json.dump(asdict(config), f, indent=2)

                # Save training history
                with open(os.path.join(output_dir, 'training_histories.json'), 'w') as f:
                    json.dump({'0': history}, f, indent=2)

                # Save detailed losses and compute loss statistics
                detailed_losses = evaluator.compute_detailed_losses()
                detailed_df = pd.DataFrame({
                    'reconstruction_loss': detailed_losses['reconstruction_loss'],
                    'discrepancy_loss': detailed_losses['discrepancy_loss'],
                    'total_loss': detailed_losses['total_loss'],
                    'label': detailed_losses['labels'],
                    'sample_type': detailed_losses['sample_types'],
                    'anomaly_type': detailed_losses['anomaly_types'],
                    'anomaly_type_name': [SLIDING_ANOMALY_TYPE_NAMES[int(at)] for at in detailed_losses['anomaly_types']]
                })
                detailed_df.to_csv(os.path.join(output_dir, 'best_model_detailed.csv'), index=False)

                # Compute loss statistics and separation metrics
                loss_stats = compute_loss_statistics(detailed_losses)

                # Save anomaly type metrics
                anomaly_type_metrics = evaluator.get_performance_by_anomaly_type()
                with open(os.path.join(output_dir, 'anomaly_type_metrics.json'), 'w') as f:
                    json.dump(anomaly_type_metrics, f, indent=2)

                # Save experiment metadata
                metadata = {
                    'experiment_name': exp_name,
                    'scoring_mode': scoring_mode,
                    'inference_mode': inference_mode,
                    'train_time': train_time,
                    'inference_time': eval_time,
                    'metrics': metrics,
                    'loss_stats': loss_stats,
                    'timestamp': datetime.now().isoformat()
                }
                with open(os.path.join(output_dir, 'experiment_metadata.json'), 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)

                # Build comprehensive result dict (matching Phase 1 format + new metrics)
                config_dict = asdict(config)
                all_results[result_key] = {
                    'output_dir': output_dir,
                    'metrics': metrics,
                    'disc_metrics': disc_metrics,
                    'teacher_recon_metrics': teacher_recon_metrics,
                    'student_recon_metrics': student_recon_metrics,
                    'loss_stats': loss_stats,
                    'config': config_dict,
                    'train_time': train_time,
                    'eval_time': eval_time,
                    'inference_mode': inference_mode,
                    'inference_suffix': inference_suffix
                }

        return all_results


# =============================================================================
# Main Runner
# =============================================================================

def run_ablation_study(
    config_path: str,
    output_dir: Optional[str] = None,
    skip_existing: bool = True,
    enable_viz: bool = True,
    background_viz: bool = True,
    experiments_filter: Optional[List[str]] = None
):
    """Run ablation study from config file."""

    # Load configuration
    ablation_config = load_config_module(config_path)
    phase_name = ablation_config['phase_name']
    experiments = ablation_config['experiments']
    scoring_modes = ablation_config['scoring_modes']
    inference_modes = ablation_config['inference_modes']
    # mask_settings: loaded from config, respects BASE_CONFIG['mask_after_encoder']
    mask_settings = ablation_config['mask_settings']

    print(f"{'='*80}")
    print(f"Ablation Study: {phase_name}")
    print(f"Description: {ablation_config['phase_description']}")
    print(f"{'='*80}")
    print(f"Total experiments: {len(experiments)}")
    print(f"Mask settings: {['mask_before' if not m else 'mask_after' for m in mask_settings]}")
    print(f"Scoring modes: {scoring_modes}")
    print(f"Inference modes: {inference_modes}")
    print(f"Skip existing: {skip_existing}")
    print(f"Background visualization: {background_viz}")

    # Filter experiments if specified
    if experiments_filter:
        experiments = [e for e in experiments if e['name'] in experiments_filter]
        print(f"Filtered to {len(experiments)} experiments")

    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(
            PROJECT_ROOT, 'results', 'experiments',
            f'{timestamp}_{phase_name}'
        )
    os.makedirs(output_dir, exist_ok=True)
    info_dir = os.path.join(output_dir, '000_ablation_info')
    os.makedirs(info_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print(f"Info directory: {info_dir}")
    print()

    # Generate dataset
    print("Generating dataset...", flush=True)
    base_config = Config()
    set_seed(base_config.random_seed)

    complexity = NormalDataComplexity(enable_complexity=False)
    generator = SlidingWindowTimeSeriesGenerator(
        total_length=base_config.sliding_window_total_length,
        num_features=base_config.num_features,
        interval_scale=base_config.anomaly_interval_scale,
        complexity=complexity,
        seed=base_config.random_seed
    )
    signals, point_labels, anomaly_regions = generator.generate()
    print(f"  Signal shape: {signals.shape}")

    # Save dataset information to 000_ablation_info/dataset.md
    print("Saving dataset info...", flush=True)
    save_dataset_info(
        output_dir=info_dir,
        config=base_config,
        signals=signals,
        point_labels=point_labels,
        anomaly_regions=anomaly_regions,
        train_ratio=base_config.sliding_window_train_ratio
    )
    print()

    # Run experiments - iterate over mask_settings (before_mask first, then after_mask)
    all_results = []
    total_runs = len(experiments) * len(mask_settings)
    run_count = 0

    for mask_after_encoder in mask_settings:
        mask_suffix = 'mask_after' if mask_after_encoder else 'mask_before'
        print(f"\n{'='*80}")
        print(f" ROUND: mask_after_encoder = {mask_after_encoder} ({mask_suffix})")
        print(f"{'='*80}\n")

        for exp_config in tqdm(experiments, desc=f"Experiments ({mask_suffix})"):
            exp_name = exp_config['name']
            run_count += 1

            # Create modified config with current mask_after_encoder setting
            exp_config_modified = deepcopy(exp_config)
            exp_config_modified['mask_after_encoder'] = mask_after_encoder

            # Check if all outputs exist
            all_exist = True
            for inference_mode in inference_modes:
                inference_suffix = 'all' if inference_mode == 'all_patches' else 'last'
                for scoring_mode in scoring_modes:
                    result_key = f"{scoring_mode}_{inference_suffix}"
                    exp_dir = os.path.join(output_dir, f"{exp_name}_{mask_suffix}_{result_key}")
                    if not os.path.exists(exp_dir):
                        all_exist = False
                        break
                if not all_exist:
                    break

            if skip_existing and all_exist:
                print(f"  Skipping {exp_name}_{mask_suffix} (all outputs exist)")
                continue

            try:
                runner = SingleExperimentRunner(
                    exp_config=exp_config_modified,
                    output_base_dir=output_dir,
                    signals=signals,
                    point_labels=point_labels,
                    anomaly_regions=anomaly_regions,
                    scoring_modes=scoring_modes,
                    inference_modes=inference_modes
                )

                results = runner.run()

                # Start background visualization
                if enable_viz and results:
                    first_result = next(iter(results.values()))
                    model_path = os.path.join(first_result['output_dir'], 'best_model.pt')
                    output_dirs_map = {k: v['output_dir'] for k, v in results.items()}

                    if background_viz:
                        print(f"  Starting background visualization...", flush=True)
                        run_visualization_background(
                            model_path=model_path,
                            scoring_modes=scoring_modes,
                            inference_modes=inference_modes,
                            output_dirs=output_dirs_map,
                            num_test=500
                        )

                # Record results (matching Phase 1 format + new separation metrics)
                for result_key, result in results.items():
                    scoring_mode = result_key.split('_')[0]
                    inference_suffix = result_key.split('_')[1]
                    cfg = result['config']
                    loss_stats = result['loss_stats']

                    record = {
                        # Experiment identifiers
                        'experiment': f"{exp_name}_{mask_suffix}_{result_key}",
                        'base_experiment': f"{exp_name}_{mask_suffix}",
                        'scoring_mode': scoring_mode,
                        'inference_mode': result['inference_mode'],
                        'inference_suffix': inference_suffix,
                        'mask_after_encoder': mask_after_encoder,

                        # Timing
                        'train_time': result['train_time'],
                        'inference_time': result['eval_time'],

                        # Config parameters
                        'force_mask_anomaly': cfg.get('force_mask_anomaly'),
                        'margin_type': cfg.get('margin_type'),
                        'masking_ratio': cfg.get('masking_ratio'),
                        'masking_strategy': cfg.get('masking_strategy'),
                        'seq_length': cfg.get('seq_length'),
                        'num_patches': cfg.get('num_patches'),
                        'patch_size': cfg.get('patch_size'),
                        'patch_level_loss': cfg.get('patch_level_loss'),
                        'patchify_mode': cfg.get('patchify_mode'),
                        'shared_mask_token': cfg.get('shared_mask_token'),
                        'd_model': cfg.get('d_model'),
                        'nhead': cfg.get('nhead'),
                        'num_encoder_layers': cfg.get('num_encoder_layers'),
                        'num_teacher_decoder_layers': cfg.get('num_teacher_decoder_layers'),
                        'num_student_decoder_layers': cfg.get('num_student_decoder_layers'),
                        'num_shared_decoder_layers': cfg.get('num_shared_decoder_layers'),
                        'dim_feedforward': cfg.get('dim_feedforward'),
                        'dropout': cfg.get('dropout'),
                        'cnn_channels': str(cfg.get('cnn_channels')),
                        'anomaly_loss_weight': cfg.get('anomaly_loss_weight'),
                        'num_epochs': cfg.get('num_epochs'),
                        'mask_last_n': cfg.get('mask_last_n'),
                        'margin': cfg.get('margin'),
                        'lambda_disc': cfg.get('lambda_disc'),
                        'dynamic_margin_k': cfg.get('dynamic_margin_k'),
                        'learning_rate': cfg.get('learning_rate'),
                        'weight_decay': cfg.get('weight_decay'),
                        'teacher_only_warmup_epochs': cfg.get('teacher_only_warmup_epochs'),
                        'warmup_epochs': cfg.get('warmup_epochs'),

                        # Evaluation metrics
                        **result['metrics'],

                        # Loss statistics (per-category means)
                        'reconstruction_loss': loss_stats['reconstruction_loss'],
                        'discrepancy_loss': loss_stats['discrepancy_loss'],
                        'recon_normal': loss_stats['recon_normal'],
                        'recon_anomaly': loss_stats['recon_anomaly'],
                        'recon_pure_normal': loss_stats['recon_pure_normal'],
                        'recon_disturbing': loss_stats['recon_disturbing'],
                        'disc_normal': loss_stats['disc_normal'],
                        'disc_anomaly': loss_stats['disc_anomaly'],
                        'disc_pure_normal': loss_stats['disc_pure_normal'],
                        'disc_disturbing': loss_stats['disc_disturbing'],

                        # Ratio metrics (higher = better separation)
                        'disc_ratio': loss_stats['disc_ratio'],
                        'disc_ratio_disturbing': loss_stats['disc_ratio_disturbing'],
                        'recon_ratio': loss_stats['recon_ratio'],

                        # Cohen's d separation metrics (higher = better separation)
                        'disc_cohens_d_normal_vs_anomaly': loss_stats['disc_cohens_d_normal_vs_anomaly'],
                        'disc_cohens_d_disturbing_vs_anomaly': loss_stats['disc_cohens_d_disturbing_vs_anomaly'],
                        'recon_cohens_d_normal_vs_anomaly': loss_stats['recon_cohens_d_normal_vs_anomaly'],
                        'recon_cohens_d_disturbing_vs_anomaly': loss_stats['recon_cohens_d_disturbing_vs_anomaly'],

                        # Standard deviations
                        'disc_normal_std': loss_stats['disc_normal_std'],
                        'disc_anomaly_std': loss_stats['disc_anomaly_std'],
                        'disc_disturbing_std': loss_stats['disc_disturbing_std'],

                        # Discrepancy-only metrics
                        'disc_only_roc_auc': result.get('disc_metrics', {}).get('roc_auc', 0),
                        'disc_only_f1_score': result.get('disc_metrics', {}).get('f1_score', 0),
                        'disc_only_pa_20_roc_auc': result.get('disc_metrics', {}).get('pa_20_roc_auc', 0),
                        'disc_only_pa_20_f1': result.get('disc_metrics', {}).get('pa_20_f1', 0),
                        'disc_only_pa_50_roc_auc': result.get('disc_metrics', {}).get('pa_50_roc_auc', 0),
                        'disc_only_pa_50_f1': result.get('disc_metrics', {}).get('pa_50_f1', 0),
                        'disc_only_pa_80_roc_auc': result.get('disc_metrics', {}).get('pa_80_roc_auc', 0),
                        'disc_only_pa_80_f1': result.get('disc_metrics', {}).get('pa_80_f1', 0),

                        # Teacher recon-only metrics
                        'teacher_recon_roc_auc': result.get('teacher_recon_metrics', {}).get('roc_auc', 0),
                        'teacher_recon_f1_score': result.get('teacher_recon_metrics', {}).get('f1_score', 0),
                        'teacher_recon_pa_20_roc_auc': result.get('teacher_recon_metrics', {}).get('pa_20_roc_auc', 0),
                        'teacher_recon_pa_20_f1': result.get('teacher_recon_metrics', {}).get('pa_20_f1', 0),
                        'teacher_recon_pa_50_roc_auc': result.get('teacher_recon_metrics', {}).get('pa_50_roc_auc', 0),
                        'teacher_recon_pa_50_f1': result.get('teacher_recon_metrics', {}).get('pa_50_f1', 0),
                        'teacher_recon_pa_80_roc_auc': result.get('teacher_recon_metrics', {}).get('pa_80_roc_auc', 0),
                        'teacher_recon_pa_80_f1': result.get('teacher_recon_metrics', {}).get('pa_80_f1', 0),

                        # Student recon-only metrics
                        'student_recon_roc_auc': result.get('student_recon_metrics', {}).get('roc_auc', 0),
                        'student_recon_f1_score': result.get('student_recon_metrics', {}).get('f1_score', 0),
                        'student_recon_pa_20_roc_auc': result.get('student_recon_metrics', {}).get('pa_20_roc_auc', 0),
                        'student_recon_pa_20_f1': result.get('student_recon_metrics', {}).get('pa_20_f1', 0),
                        'student_recon_pa_50_roc_auc': result.get('student_recon_metrics', {}).get('pa_50_roc_auc', 0),
                        'student_recon_pa_50_f1': result.get('student_recon_metrics', {}).get('pa_50_f1', 0),
                        'student_recon_pa_80_roc_auc': result.get('student_recon_metrics', {}).get('pa_80_roc_auc', 0),
                        'student_recon_pa_80_f1': result.get('student_recon_metrics', {}).get('pa_80_f1', 0),

                        # Output directory
                        'output_dir': result['output_dir'],
                    }
                    all_results.append(record)

                # Save summary incrementally to 000_ablation_info/
                summary_df = pd.DataFrame(all_results)
                summary_path = os.path.join(info_dir, 'summary_results.csv')
                summary_df.to_csv(summary_path, index=False)

                # Print summary with separation metrics
                for result_key, result in results.items():
                    roc_auc = result['metrics'].get('roc_auc', 0)
                    disc_d = result['loss_stats'].get('disc_cohens_d_normal_vs_anomaly', 0)
                    print(f"    {result_key}: ROC-AUC={roc_auc:.4f}, disc_d={disc_d:.2f}")

            except Exception as e:
                print(f"  ERROR in {exp_name}_{mask_suffix}: {e}")
                import traceback
                traceback.print_exc()

    # Save summary to 000_ablation_info/
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_path = os.path.join(info_dir, 'summary_results.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to: {summary_path}")

    # Wait for visualizations
    if enable_viz and background_viz and _background_viz_processes:
        print(f"\n{'='*60}")
        print("Waiting for background visualizations to complete...")
        print(f"{'='*60}")
        wait_for_background_visualizations()
        print("All background visualizations completed.")

    print(f"\n{'='*80}")
    print("Ablation study complete!")
    print(f"{'='*80}")

    return output_dir


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run ablation study from config file')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file (e.g., configs/20260126_160417_phase2.py)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: results/experiments/{timestamp}_{phase})')
    parser.add_argument('--no-skip', action='store_true',
                        help='Do not skip existing experiments')
    parser.add_argument('--no-viz', action='store_true',
                        help='Disable visualization')
    parser.add_argument('--no-background-viz', action='store_true',
                        help='Run visualization synchronously')
    parser.add_argument('--experiments', type=str, nargs='+', default=None,
                        help='Run only specific experiments by name')

    args = parser.parse_args()

    run_ablation_study(
        config_path=args.config,
        output_dir=args.output_dir,
        skip_existing=not args.no_skip,
        enable_viz=not args.no_viz,
        background_viz=not args.no_background_viz,
        experiments_filter=args.experiments
    )
