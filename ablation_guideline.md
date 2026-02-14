# MAE Ablation Experiment Guide

**Purpose**: Complete reference for running MAE ablation experiments.

---

## Summary

### Quick Reference
```
┌─────────────────────────────────────────────────────────────────┐
│  TODO LIST:      Create BEFORE experiment, update as TOP PRIORITY│
│  TRAIN STRIDE:   11 (14days), 3 (50:50)                         │
│  EPOCHS:         50                                              │
│  EXECUTION:      Always foreground (NO conda run, NO &, NO nohup)│
│  WORKFLOW:       Run ablation → Verify results → Add to comparison│
└─────────────────────────────────────────────────────────────────┘
```

### Section Index
| Section | When to Use |
|---------|-------------|
| [1. Critical Rules](#1-critical-rules) | Before ANY experiment |
| [2. Workflow](#2-workflow) | Running experiments |
| [3. Monitoring](#3-monitoring) | During experiment execution |
| [4. Results Structure](#4-results-structure) | Understanding output |
| [5. New Dataset](#5-new-dataset) | Adding new datasets |
| [6. Troubleshooting](#6-troubleshooting) | When errors occur |

---

## 1. Critical Rules

### 1.1 Mandatory Parameters

**Reference**: `scripts/run_wadi_14days_ablation.py` line 47

| Parameter | 14days Split | 50:50 Split |
|-----------|-------------|-------------|
| `TRAIN_STRIDE` | **11** | **3** |
| `test_stride` | **1** | **1** |
| `num_epochs` | **50** | **50** |

### 1.2 Execution Rules
- **ALWAYS** run in foreground to see live output
- **NEVER** use: `conda run`, `&`, `nohup`
- **ALWAYS** activate environment first: `conda activate dc_vis`

### 1.3 Todo List Management (CRITICAL)
```
⚠️  BEFORE starting ANY experiment:
    1. Create a todo list with ALL steps
    2. Mark current task as in_progress
    3. Update todo list as TOP PRIORITY at every step
    4. Mark completed tasks immediately
    5. NEVER proceed without updating todo list first
```

**Why**: Experiments are long-running (40 configs × ~35min = ~24 hours). Without todo list:
- Progress is forgotten between monitoring intervals
- Steps are skipped or duplicated
- State is lost if interrupted

---

## 2. Workflow

### 2.1 Available Scripts

**Reference**: Verify with `ls scripts/run_*ablation*.py`

| Experiment Type | Script | Results Directory |
|-----------------|--------|-------------------|
| WaDi 50:50 | `scripts/run_wadi_ablation.py` | `results/WaDi/{A1,A2}/` |
| WaDi 14days | `scripts/run_wadi_14days_ablation.py` | `results/WaDi/{A1,A2}_14days/` |
| WaDi swap | `scripts/run_wadi_ablation.py --swap` | `results/WaDi/{A1,A2}_swap/` |
| SWaT A1+A2 | `scripts/run_swat_ablation.py` | `results/SWaT/` |

### 2.2 Standard Commands
```bash
conda activate dc_vis

# Full ablation (40 experiments = 8 arch × 5 window/patch)
python scripts/run_wadi_14days_ablation.py --scenario A1

# Single experiment
python scripts/run_wadi_14days_ablation.py --scenario A1 --only w500_p5_td4_sd2

# Resume from specific index
python scripts/run_wadi_14days_ablation.py --scenario A1 --start-from 20
```

### 2.3 Command Line Arguments

**Reference**: `scripts/run_wadi_ablation.py` argparse section

| Argument | Description |
|----------|-------------|
| `--scenario` | A1 or A2 |
| `--start-from` | Start from experiment index (0-based) |
| `--only` | Run only specific experiment name |
| `--swap` | Swap train/test split (50:50 only) |

### 2.4 Experiment Grid

**Reference**: `scripts/run_wadi_ablation.py` lines 49-66

**8 Architecture Variants** × **5 Window/Patch Configs** = **40 experiments**

Architecture variants:
- `td2_sd1`, `td3_sd1`, `td4_sd1`, `td4_sd2` (decoder depth)
- `d64` (reduced model dimension)
- `enc2`, `enc3`, `enc4` (encoder depth)

Window/Patch configs:
- `w500_p5`, `w100_p5`, `w500_p10`, `w100_p10`, `w500_p20`

**Naming**: `{window}_{patch}_{arch}` (e.g., `w500_p5_td4_sd2`)

### 2.5 After Ablation: Add to Comparison

**Reference**: `comparison/add_mae_14days_results.py`

```bash
# After ablation complete, add best results to comparison
python comparison/add_mae_14days_results.py --scenario A1

# Discover available configs and metrics
python comparison/add_mae_14days_results.py --scenario A1 --discover

# Verify
python -c "import json; print(json.load(open('comparison/results/WaDi_A1_14days/results.json'))['models'].keys())"
```

---

## 3. Monitoring

### 3.1 Monitoring Commands
```bash
# GPU status
nvidia-smi

# Check running process
ps aux | grep "run_wadi" | grep python

# Count completed experiments
ls results/WaDi/A1_14days/ | wc -l
```

### 3.2 Progress Report Format
```
#   Experiment           Status       Time    ROC     PRC     F1_T
--------------------------------------------------------------------
1   w500_p5_td2_sd1      ✅          32min   0.958   0.650   0.649
2   w500_p5_td3_sd1      ✅          34min   0.945   0.620   0.625
3   w500_p5_td4_sd1      🔄 Ep 35/50 -       -       -       -
4   w500_p5_td4_sd2      ⏳          -       -       -       -
```

**Status**: ✅ Completed | 🔄 In progress | ⏳ Pending | ❌ Failed

### 3.3 Monitoring Frequency
| Phase | Check Interval | Action |
|-------|----------------|--------|
| Pipeline start | Every 1 min | Verify first experiment running |
| During training | Every 10 min | Check GPU usage, update table |
| Between experiments | Every 30 min | Count completed, update progress |

**Report progress table at each check interval.**

### 3.4 Stuck Detection

A process is stuck if:
- GPU memory full but no training progress
- No new files in results directory for >45 minutes
- CPU at 100% but no GPU activity

```bash
# Monitor file changes
watch -n 60 "ls -lt results/WaDi/A1_14days/ | head -5"
```

---

## 4. Results Structure

### 4.1 Directory Layout

**Verify with**: `ls results/WaDi/A1_14days/`

```
results/WaDi/{scenario}/
├── dataset.md                                    # Dataset info
├── ablation_summary_{timestamp}.json             # Summary
└── {timestamp}_{config_name}/                    # One per experiment
    ├── experiment_metadata.json                  # Config + all metrics
    ├── best_config.json                          # Model configuration
    ├── best_model.pt                             # Model checkpoint
    ├── best_model_detailed.csv                   # Per-sample scores
    ├── anomaly_type_metrics.json                 # Per-anomaly-type metrics
    ├── training_histories.json                   # Training history
    └── visualization/best_model/                 # Plots
```

### 4.2 experiment_metadata.json Key Fields

**Reference**: `results/WaDi/A1_14days/{exp}/experiment_metadata.json`

| Field | Description |
|-------|-------------|
| `metrics.roc_auc` | Combined scoring ROC-AUC |
| `metrics.prc_auc` | Combined scoring PRC-AUC |
| `metrics.f1_t` | Threshold-adjusted F1 |
| `teacher_recon_metrics.*` | Teacher-only scoring |
| `disc_metrics.*` | Discrepancy-only scoring |
| `loss_stats.disc_SNR` | Discrepancy signal-to-noise ratio |

### 4.3 Quick Results Check
```bash
# View metrics for specific experiment
python -c "
import json
m = json.load(open('results/WaDi/A1_14days/{exp}/experiment_metadata.json'))
print(f\"ROC: {m['metrics']['roc_auc']:.4f}\")
print(f\"PRC: {m['metrics']['prc_auc']:.4f}\")
print(f\"F1_T: {m['metrics']['f1_t']:.4f}\")
"
```

---

## 5. New Dataset

### 5.1 Code Reuse (CRITICAL)

**NEVER duplicate shared functions.** Import from `run_wadi_14days_ablation.py`:

```python
from scripts.run_wadi_14days_ablation import (
    make_config, free_gpu, mem_status, run_single_experiment,
)
```

**Shared functions** (parameterized, do NOT re-implement):
| Function | Purpose | Dataset-specific params |
|----------|---------|------------------------|
| `make_config(overrides)` | Config defaults | None |
| `free_gpu()` | GPU cleanup | None |
| `mem_status()` | Memory status | None |
| `run_single_experiment(...)` | Train+inference+spawn | `train_stride`, `dataset_name`, `background_processes` |
| `_cpu_eval_viz_worker(...)` | Background eval+viz | `dataset_name` (via run_single_experiment) |

**Dataset-specific code only** (implement in new script):
- `load_{dataset}_combined()` — data loading
- `save_dataset_info()` — dataset.md text
- `EXPERIMENTS` list — experiment configs
- `main()` — CLI args, orchestration

### 5.2 Reference Scripts

| Script | Location | Reference File |
|--------|----------|----------------|
| Ablation Runner | `scripts/run_{dataset}_ablation.py` | `scripts/run_wadi_14days_ablation.py` |
| MAE Integration | `comparison/add_mae_{exp}_results.py` | `comparison/add_mae_14days_results.py` |

**Example**: `scripts/run_swat_ablation.py` — imports shared functions, only defines SWaT data loading + experiments.

### 5.3 Steps for New Dataset
1. Create ablation script → `scripts/run_{dataset}_ablation.py`
   - Import shared functions from `run_wadi_14days_ablation.py`
   - Implement only: data loading, dataset info, experiment list, main
2. Create MAE integration script → `comparison/add_mae_{exp}_results.py`
3. Run ablation: `python scripts/run_{dataset}_ablation.py`
4. Add to comparison: `python comparison/add_mae_{exp}_results.py`

---

## 6. Troubleshooting

| Problem | Solution |
|---------|----------|
| **OOM during training** | Reduce `batch_size` (default 256 → 128) |
| **OOM during eval** | Already handled (VIZ_MAX_SAMPLES=10000) |
| **Stuck process** | `kill -9 PID`, resume with `--start-from` |
| **Missing experiment** | Run with `--only {exp_name}` |
| **Background eval slow** | Normal - runs at nice=19 priority |

### 6.1 Resume After Failure
```bash
# Find last completed experiment index
ls results/WaDi/A1_14days/ | wc -l

# Resume from next index
python scripts/run_wadi_14days_ablation.py --scenario A1 --start-from 25
```

### 6.2 Rerun Single Experiment
```bash
# Delete failed experiment directory first
rm -rf results/WaDi/A1_14days/{timestamp}_{config_name}/

# Rerun specific config
python scripts/run_wadi_14days_ablation.py --scenario A1 --only w500_p5_td4_sd2
```

---

## Checklist Before Running

- [ ] **Todo list created with ALL experiment steps**
- [ ] `conda activate dc_vis`
- [ ] Correct script selected (14days vs 50:50 vs swap)
- [ ] Running in foreground (NO &, NO nohup)
- [ ] Monitoring ready (nvidia-smi, watch)
- [ ] Sufficient disk space (~10GB per experiment)

---

## 7. Unified Config System

### 7.1 Usage

**Single command for all datasets:**
```bash
python scripts/ablation/run_ablation.py --config scripts/ablation/configs/<your_config>.py
```

### 7.2 Critical File Reference

**Implementation (modify these if adding features):**

| File | Lines | Purpose |
|------|-------|---------|
| `mae_anomaly/datasets/loaders.py` | All | Dataset loaders + `DATASET_LOADERS` registry |
| `scripts/ablation/run_ablation.py` | 1337-1354 | Config merge logic |
| `scripts/ablation/run_ablation.py` | 1386-1425 | Dataset loading by type |
| `scripts/ablation/run_ablation.py` | 1106-1125 | Config application to model |

**Config Templates (copy these to start):**

| Template | Use For |
|----------|---------|
| `scripts/ablation/configs/simulation_test.py` | Quick testing (1 epoch) |
| `scripts/ablation/configs/swat_A1A2_test.py` | SWaT experiments |
| `scripts/ablation/configs/wadi_14days_A1_test.py` | WaDi experiments |

### 7.3 Config File Format

```python
# scripts/ablation/configs/my_experiment.py

DATASET_TYPE = 'swat_A1A2'  # Required - see section 7.4

PHASE_NAME = "my_experiment"
PHASE_DESCRIPTION = "Brief description"

# Applied to ALL experiments
BASE_CONFIG = {
    'seq_length': 500,
    'patch_size': 5,
    'num_patches': 100,
    'num_epochs': 50,
    'learning_rate': 2e-3,
    'batch_size': 256,
    'sliding_window_stride': 11,      # Train stride
    'sliding_window_test_stride': 1,   # Test stride
    # ... other config params
}

# Experiment variations (each inherits BASE_CONFIG)
EXPERIMENTS = [
    {'name': 'baseline', 'config': {}},  # Uses BASE_CONFIG
    {'name': 'exp2', 'config': {'num_epochs': 100}},  # Override
]

SCORING_MODES = ['default']
```

### 7.4 Dataset Types

See `mae_anomaly/datasets/loaders.py` DATASET_LOADERS for complete list:

| `DATASET_TYPE` | Description |
|----------------|-------------|
| `'simulation'` | Generated time series (default) |
| `'swat_A1A2'` | SWaT A1 + A2 combined |
| `'swat_A1A2_swap'` | SWaT with swapped A2 halves |
| `'wadi_14days_A1'` | WaDi 14 days + A1 |
| `'wadi_14days_A2'` | WaDi 14 days + A2 |
| `'wadi_A2'` | WaDi A2 only |

### 7.5 Quick Start

```bash
# 1. Copy template
cp scripts/ablation/configs/simulation_test.py scripts/ablation/configs/my_exp.py

# 2. Edit: DATASET_TYPE, BASE_CONFIG, EXPERIMENTS

# 3. Run
python scripts/ablation/run_ablation.py --config scripts/ablation/configs/my_exp.py
```

**Quick test (1 epoch):**
```python
BASE_CONFIG = {
    'num_epochs': 1,
    'sliding_window_total_length': 27500,  # For simulation only
    'sliding_window_test_stride': 11,      # Fast test
}
```

---

*Last updated: 2026-02-15*
