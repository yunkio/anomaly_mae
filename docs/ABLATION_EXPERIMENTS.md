# Ablation Study Framework

This document describes the ablation study framework for the Self-Distilled MAE model.

## Quick Start

```bash
# Run unified Phase 1 experiments (170 experiments × 6 variants = 1020 results)
python scripts/ablation/run_ablation.py --config configs/20260127_052220_phase1.py

# Run specific experiments only
python scripts/ablation/run_ablation.py --config configs/20260127_052220_phase1.py \
    --experiments 001_default 022_d_model_128

# Skip existing and disable visualization
python scripts/ablation/run_ablation.py --config configs/20260127_052220_phase1.py \
    --no-viz

# Run with background visualization (default)
python scripts/ablation/run_ablation.py --config configs/20260127_052220_phase1.py
```

## Directory Structure

```
scripts/ablation/
├── run_ablation.py                              # Unified runner (use this)
├── configs/                                      # Configuration files
│   └── 20260127_052220_phase1.py               # Unified Phase 1: 170 experiments
└── run_ablation_experiments_*.py               # [DEPRECATED] Old scripts
```

## Creating New Ablation Configs

Create a new Python file in `configs/` with the following structure:

```python
"""
Phase N Ablation Study Configuration
"""

from copy import deepcopy
from typing import Dict, List

# Phase metadata
PHASE_NAME = "phase_n"
PHASE_DESCRIPTION = "Description of this phase"
CREATED_AT = "2026-01-27 12:00:00"

# Evaluation modes
SCORING_MODES = ['default', 'adaptive', 'normalized']

# Base configuration
BASE_CONFIG = {
    'd_model': 64,
    'nhead': 2,
    'masking_ratio': 0.2,
    # ... other settings
}

def get_experiments() -> List[Dict]:
    """Define experiment configurations."""
    base = deepcopy(BASE_CONFIG)
    experiments = []

    exp = deepcopy(base)
    exp['name'] = '01_experiment_name'
    exp['masking_ratio'] = 0.10
    experiments.append(exp)

    # ... more experiments

    return experiments

EXPERIMENTS = get_experiments()
```

## Command Line Options

| Option | Description |
|--------|-------------|
| `--config` | Path to config file (required) |
| `--output-dir` | Custom output directory |
| `--no-skip` | Re-run all experiments (don't skip existing) |
| `--no-viz` | Disable visualization |
| `--no-background-viz` | Run visualization synchronously |
| `--experiments` | Run only specific experiments by name |

## Unified Phase 1: Comprehensive Ablation Study (170 Experiments)

The unified Phase 1 combines architecture exploration and focused optimization into a single comprehensive ablation study.

**Total Experiments**: 170
**Variants per Experiment**: 2 (mask_after_encoder) × 3 (scoring_mode) = 6
**Total Results**: 170 × 6 = **1020**

### Base Configuration

```python
BASE_CONFIG = {
    'd_model': 64,
    'nhead': 2,
    'masking_ratio': 0.2,
    'num_encoder_layers': 1,
    'num_teacher_decoder_layers': 2,
    'num_student_decoder_layers': 1,
    'dim_feedforward': 256,
    'cnn_channels': (32, 64),
    'patchify_mode': 'patch_cnn',
    'margin_type': 'dynamic',
    'force_mask_anomaly': True,
    'shared_mask_token': False,
    # ... other settings
}
```

### Experiment Groups

| Group | Experiments | Focus |
|-------|-------------|-------|
| 1 | 001-010 | Window Size & Patch Variations |
| 2 | 011-020 | Encoder/Decoder Depth |
| 3 | 021-030 | Model Capacity (d_model, nhead, FFN) |
| 4 | 031-040 | Masking Ratio |
| 5 | 041-050 | Loss Parameters |
| 6 | 051-060 | Training Parameters |
| 7 | 061-070 | Combined Optimal Configurations |
| 8 | 071-085 | Maximize disc_ratio with d_model=128 |
| 9 | 086-100 | Maximize disc_ratio AND t_ratio |
| 10 | 101-120 | Best Model Variations (Scoring & Window) |
| 11 | 121-140 | Window Size vs Model Depth Relationship |
| 12 | 141-155 | PA%80 Optimization Experiments |
| 13 | 156-170 | Additional Explorations |

### Key Experiments

**Architecture Exploration (001-070)**:
- Window sizes: 100, 200, 500, 1000, 2000 timesteps
- Patch sizes: 5, 10, 20 timesteps
- Encoder layers: 1, 2, 3, 4
- Decoder configs: t2s1, t3s1, t4s1, t2s2, t3s2, t4s2
- d_model: 16, 32, 64, 128, 256
- nhead: 1, 2, 4, 8, 16
- Masking ratios: 0.1 to 0.8

**Focused Optimization (071-170)**:
- d_model=128 variations with different nhead, masking_ratio
- Window 500 with various decoder depths
- PA%80 optimization with nhead=16, t4s1, masking_ratio=0.10
- Learning rate and dropout variations

## Scoring Modes

| Mode | Formula | Description |
|------|---------|-------------|
| `default` | recon + λ×disc | Fixed weight (λ=0.5) |
| `adaptive` | recon + (μ_recon/μ_disc)×disc | Dynamic weight |
| `normalized` | z(recon) + z(disc) | Z-score normalization |

## Output Structure

```
results/experiments/{timestamp}_phase1/
├── summary_results.csv                    # All metrics summary
├── {exp_name}_mask_before_default/
│   ├── best_model.pt                      # Model checkpoint
│   ├── best_config.json                   # Config
│   ├── training_histories.json            # Training history
│   ├── best_model_detailed.csv            # Per-sample losses
│   ├── anomaly_type_metrics.json          # Per-type metrics
│   ├── experiment_metadata.json           # Experiment info
│   └── visualization/
│       └── best_model/
│           ├── best_model_roc_curve.png
│           ├── best_model_confusion_matrix.png
│           ├── best_model_score_contribution.png
│           └── ...
├── {exp_name}_mask_before_adaptive/
│   └── ...
├── {exp_name}_mask_before_normalized/
│   └── ...
└── ...
```

## Key Metrics

| Metric | Description | Goal |
|--------|-------------|------|
| `disc_ratio` | disc_anomaly / disc_normal | Higher |
| `t_ratio` | t_recon_anomaly / t_recon_normal | Higher |
| `roc_auc` | ROC Area Under Curve | Higher |
| `f1_score` | F1 Score | Higher |
| `pa_k_f1` | Point-Adjusted F1 at K% | Higher |

---

**Status**: ✅ Framework implemented (170 experiments unified)
