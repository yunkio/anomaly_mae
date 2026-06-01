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

**Reference**: Config files in `scripts/ablation/configs/`

| Parameter | 14days Split | 50:50 Split |
|-----------|-------------|-------------|
| `sliding_window_stride` | **11** | **3** |
| `sliding_window_test_stride` | **3** | **3** |
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

### 2.1 Unified Config System

**Single Entry Point**: `scripts/ablation/run_ablation.py`

**Available Dataset Types** (defined in config files):

| Dataset Type | Config Example | Results Directory |
|--------------|----------------|-------------------|
| Simulation | `simulation_test.py` | `results/experiments/YYYYMMDD_HHMMSS/` |
| SWaT A1+A2 | `swat_A1A2_test.py` | `results/experiments/YYYYMMDD_HHMMSS/` |
| SWaT swap | Custom config with `DATASET_TYPE='swat_A1A2_swap'` | `results/experiments/YYYYMMDD_HHMMSS/` |
| WaDi 14days A1 | `wadi_14days_A1_test.py` | `results/experiments/YYYYMMDD_HHMMSS/` |
| WaDi 14days A2 | Custom config with `DATASET_TYPE='wadi_14days_A2'` | `results/experiments/YYYYMMDD_HHMMSS/` |
| WaDi A2 only | Custom config with `DATASET_TYPE='wadi_A2'` | `results/experiments/YYYYMMDD_HHMMSS/` |

**Config Files Location**: `scripts/ablation/configs/`

### 2.2 Standard Commands
```bash
conda activate dc_vis

# Run ablation with config file
python scripts/ablation/run_ablation.py --config scripts/ablation/configs/wadi_14days_A1_test.py

# Create custom config from template
cp scripts/ablation/configs/wadi_14days_A1_test.py scripts/ablation/configs/my_experiment.py
# Edit my_experiment.py: DATASET_TYPE, BASE_CONFIG, EXPERIMENTS
python scripts/ablation/run_ablation.py --config scripts/ablation/configs/my_experiment.py

# Resume from specific experiment index
python scripts/ablation/run_ablation.py --config <config>.py --start-from 20

# Run specific experiment only
python scripts/ablation/run_ablation.py --config <config>.py --only <exp_name>
```

### 2.3 Command Line Arguments

**Reference**: `scripts/ablation/run_ablation.py` argparse section

| Argument | Description |
|----------|-------------|
| `--config` | Path to config file (required) |
| `--start-from` | Start from experiment index (0-based) |
| `--only` | Run only specific experiment name |

### 2.4 Experiment Configuration

**Reference**: Config file `EXPERIMENTS` list in `scripts/ablation/configs/*.py`

**Config Structure**:
```python
DATASET_TYPE = 'wadi_14days_A1'  # Dataset selection

BASE_CONFIG = {
    'seq_length': 500,
    'patch_size': 5,
    'num_patches': 100,
    'd_model': 128,
    'num_encoder_layers': 2,
    'num_teacher_decoder_layers': 4,
    'num_student_decoder_layers': 1,
    'num_epochs': 50,
    'learning_rate': 2e-3,
    'sliding_window_stride': 11,
    'sliding_window_test_stride': 3,
    # ... other params
}

EXPERIMENTS = [
    {'name': 'baseline', 'config': {}},  # Uses BASE_CONFIG
    {'name': 'deeper_enc', 'config': {'num_encoder_layers': 4}},
    {'name': 'larger_model', 'config': {'d_model': 256}},
    # ... define your experiments
]

SCORING_MODES = ['default']  # or ['default', 'teacher_only', 'disc_only']
```

**Typical Ablation Grid Examples**:
- Architecture: encoder layers (1, 2, 3, 4), decoder depth (td2, td3, td4), student depth (sd1, sd2)
- Model size: d_model (64, 128, 256)
- Window/Patch: (w500_p5, w100_p5, w500_p10, w100_p10, w500_p20)

### 2.5 After Ablation: Add to Comparison

**After running ablation** with `scripts/ablation/run_ablation.py`, add best results to comparison framework.

**Reference**: Dataset-specific scripts in `comparison/`

```bash
# WaDi 14days results
python comparison/add_mae_14days_results.py --scenario A1

# SWaT results
python comparison/add_mae_swat_results.py

# Discover available configs and metrics
python comparison/add_mae_14days_results.py --scenario A1 --discover

# Verify results added
python -c "import json; print(json.load(open('comparison/results/WaDi_A1_14days/results.json'))['models'].keys())"
```

**Note**: Results are stored in `results/experiments/YYYYMMDD_HHMMSS/` by the unified ablation system, then imported to comparison framework by these scripts.

---

## 3. Monitoring

### 3.1 Monitoring Commands
```bash
# GPU status
nvidia-smi

# Check running process
ps aux | grep "run_ablation" | grep python

# Count completed experiments (check experiment subdirectories)
ls results/experiments/ | tail -5  # Show latest 5 experiments
ls results/experiments/YYYYMMDD_HHMMSS/*/ | wc -l  # Count configs in specific experiment
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
# Monitor file changes (update path to your experiment)
watch -n 60 "ls -lt results/experiments/YYYYMMDD_HHMMSS/ | head -5"
```

---

## 4. Results Structure

### 4.1 Directory Layout

**Two result structures** depending on experiment type:

**1. Dataset-Specific Ablation** (데이터셋 단위 ablation 테스트)

**Path**: `results/{Dataset}/{scenario}/`

```
results/SWaT/A1A2/
├── dataset.md
├── ablation_summary_{timestamp}.json
└── {timestamp}_{config_name}/
    ├── experiment_metadata.json
    ├── best_config.json
    ├── best_model.pt
    ├── best_model_detailed.csv
    ├── anomaly_type_metrics.json
    ├── training_histories.json
    └── visualization/best_model/

results/WaDi/A1_14days/
└── ... (same structure)
```

**2. Model-Level Experiments** (모델 단위로 모든 데이터셋 테스트)

**Path**: `results/experiments/YYYYMMDD_HHMMSS_{description}/`

```
results/experiments/20260215_043952_simulation_test/
├── 000_ablation_info/
│   ├── config.py                     # Config snapshot
│   ├── dataset.md                    # Dataset info
│   └── ablation_summary.json         # Summary
│
├── 001_{exp_name}/                   # Per-experiment results
│   ├── experiment_metadata.json
│   ├── best_config.json
│   ├── best_model.pt
│   ├── best_model_detailed.csv
│   ├── anomaly_type_metrics.json
│   ├── training_histories.json
│   └── visualization/best_model/
│
└── ... (one directory per experiment)
```

### 4.2 experiment_metadata.json Key Fields

**Reference**: `results/experiments/YYYYMMDD_HHMMSS/XXX_{exp}/experiment_metadata.json`

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
m = json.load(open('results/experiments/YYYYMMDD_HHMMSS/XXX_{exp_name}/experiment_metadata.json'))
print(f\"ROC: {m['metrics']['roc_auc']:.4f}\")
print(f\"PRC: {m['metrics']['prc_auc']:.4f}\")
print(f\"F1_T: {m['metrics']['f1_t']:.4f}\")
"

# View summary of all experiments in a run
python -c "
import json
s = json.load(open('results/experiments/YYYYMMDD_HHMMSS/000_ablation_info/ablation_summary.json'))
for exp in s['experiments']:
    print(f\"{exp['name']}: ROC={exp['metrics']['roc_auc']:.4f}, PRC={exp['metrics']['prc_auc']:.4f}\")
"
```

---

## 5. New Dataset

### 5.1 Add Dataset Loader

**Add to**: `mae_anomaly/datasets/loaders.py`

```python
def load_my_dataset():
    """Load My Dataset data.

    Returns:
        tuple: (signals, point_labels, anomaly_regions, metadata,
                test_signals, test_point_labels) OR
               (signals, point_labels, anomaly_regions, metadata)
    """
    # Load your data
    # Return either 4 or 6 values depending on pre-split data
    pass

# Add to registry
DATASET_LOADERS = {
    # ... existing loaders
    'my_dataset': load_my_dataset,
}
```

### 5.2 Create Config File

**Create**: `scripts/ablation/configs/my_dataset_experiment.py`

```python
DATASET_TYPE = 'my_dataset'  # Match registry key

PHASE_NAME = "my_dataset_experiment"
PHASE_DESCRIPTION = "Description of experiment"

BASE_CONFIG = {
    'seq_length': 500,
    'patch_size': 5,
    'num_patches': 100,
    'd_model': 128,
    'num_encoder_layers': 2,
    'num_teacher_decoder_layers': 4,
    'num_student_decoder_layers': 1,
    'num_epochs': 50,
    'learning_rate': 2e-3,
    'batch_size': 256,
    'sliding_window_stride': 11,      # Adjust for your data
    'sliding_window_test_stride': 3,
    # ... other params
}

EXPERIMENTS = [
    {'name': 'baseline', 'config': {}},
    {'name': 'deeper_enc', 'config': {'num_encoder_layers': 4}},
    # ... your experiments
]

SCORING_MODES = ['default']
```

### 5.3 Steps for New Dataset

1. **Add dataset loader** to `mae_anomaly/datasets/loaders.py`
   - Implement `load_my_dataset()` function
   - Add to `DATASET_LOADERS` registry

2. **Create config file** in `scripts/ablation/configs/`
   - Set `DATASET_TYPE` to match registry key
   - Define `BASE_CONFIG` and `EXPERIMENTS`

3. **Run ablation**:
   ```bash
   python scripts/ablation/run_ablation.py --config scripts/ablation/configs/my_dataset_experiment.py
   ```

4. **Add to comparison** (optional):
   - Create `comparison/add_mae_my_dataset_results.py` if needed
   - Or use existing comparison scripts

**Reference**: See existing loaders in `mae_anomaly/datasets/loaders.py` and configs in `scripts/ablation/configs/`

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
ls results/experiments/YYYYMMDD_HHMMSS/ | grep -E "^[0-9]+" | wc -l

# Resume from next index
python scripts/ablation/run_ablation.py --config <config>.py --start-from 25
```

### 6.2 Rerun Single Experiment
```bash
# Delete failed experiment directory first
rm -rf results/experiments/YYYYMMDD_HHMMSS/XXX_{exp_name}/

# Rerun specific experiment
python scripts/ablation/run_ablation.py --config <config>.py --only {exp_name}
```

---

## Checklist Before Running

- [ ] **Todo list created with ALL experiment steps**
- [ ] `conda activate dc_vis`
- [ ] Config file prepared (DATASET_TYPE, BASE_CONFIG, EXPERIMENTS)
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
    'sliding_window_test_stride': 3,   # Test stride
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
| `'PSM'` | Pooled Server Metrics (eBay), single stream, train_stride=21 |
| `'exathlon_app{1,2,4,5,6,9}'` × 6 | Exathlon Spark cluster traces, per-app (TimeSeAD 6-app convention), 19 FScustom features |

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
