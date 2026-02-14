# Baseline Comparison Pipeline Guide

**Purpose**: Complete reference for running baseline comparison experiments.

---

## Summary

### Quick Reference
```
┌─────────────────────────────────────────────────────────────────┐
│  TODO LIST:      Create BEFORE experiment, update as TOP PRIORITY│
│  ALL DL MODELS:  epochs=10, train_stride=11                     │
│  EXECUTION:      Always foreground (NO conda run, NO &, NO nohup)│
│  WORKFLOW:       Run baselines → Add MAE → Verify results       │
└─────────────────────────────────────────────────────────────────┘
```

### Section Index
| Section | When to Use |
|---------|-------------|
| [1. Critical Rules](#1-critical-rules) | Before ANY experiment |
| [2. Models](#2-models) | Model parameter reference |
| [3. Workflow](#3-workflow) | Running experiments |
| [4. Monitoring](#4-monitoring) | During experiment execution |
| [5. New Experiment](#5-new-experiment) | Adding new datasets |
| [6. Results Structure](#6-results-structure) | Understanding output |
| [7. Troubleshooting](#7-troubleshooting) | When errors occur |

---

## 1. Critical Rules

**Reference**: Baseline comparison scripts in `comparison/` directory

### 1.1 Mandatory Parameters
| Parameter | Value | Applies To |
|-----------|-------|------------|
| `epochs` | **10** | ALL deep learning models |
| `train_stride` | **11** | ALL deep learning models |

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

**Why**: Experiments are long-running. Without todo list tracking:
- Progress is forgotten between monitoring intervals
- Steps are skipped or duplicated
- State is lost if interrupted

---

## 2. Models

**Reference**: `comparison/run_wadi_14days.py` lines 50-200 (model definitions)

### All Baselines (Unified Reference)
| Category | Model | Key Parameters |
|----------|-------|----------------|
| **Simple** | random | seed=42 |
| | sensor_range | count_sensors=False |
| | pca_error | n_components='auto' |
| | l2_norm | ord=2, normalize=True |
| | nn_distance | distance='euclidean', subsample=10000 |
| **Neural** | mlp | seq_len=5, embedding_dim=32, batch_size=512 |
| | mlpmixer | seq_len=5, embedding_dim=128, lr=0.0002 |
| | transformer | seq_len=5, embedding_dim=128, num_heads=1 |
| | gcn_lstm | seq_len=5, gcn_out=10, lstm=64, batch_size=100 |
| **SOTA** | anomaly_transformer | win_size=100, d_model=512, n_heads=8, e_layers=3 |
| | tranad | seq_len=10, d_model=256 |
| | usad | seq_len=5, hidden_dim=100, z_dim=20 |
| | dagmm | seq_len=5, latent_dim=4, n_gmm=4 |
| | gdn | seq_len=5, embed_dim=64, top_k=20 |
| | omnianomaly | seq_len=100, hidden_dim=100, z_dim=8 |

**Note**: ALL Neural and SOTA models use `epochs=10, train_stride=11`

---

## 3. Workflow

**Reference Scripts**:
- MAE Ablation: `scripts/ablation/run_ablation.py` with config files
- Baseline Runner: `comparison/run_wadi_14days.py`
- MAE Integration: `comparison/add_mae_14days_results.py`

### Pipeline Flow
```
Step 0: MAE Ablation                    Step 1: Run Baselines      Step 2: Add MAE        Step 3: Verify
(./scripts/ablation/)                   (./comparison/)            (./comparison/)
────────────────────────────            ─────────────────          ─────────────          ─────────
python scripts/ablation/run_ablation.py python run_{exp}.py   →   python add_mae_...  →  results.json
  --config configs/{config}.py     →         │                         │
     │                                       ▼                         ▼
     ▼                                  comparison/results/       (adds MAE metrics
results/{Dataset}/{scenario}/             └── {exp}/                to results.json)
  └── experiment_metadata.json               ├── {model}/
                                             └── results.json
```

### Standard Commands
```bash
conda activate dc_vis

# Step 0: Run MAE ablation FIRST
python scripts/ablation/run_ablation.py --config configs/{config}.py

# Step 1: Run baselines (in ./comparison/)
python comparison/run_{experiment}.py --scenario {scenario}

# Step 2: Add MAE results (copies from results/{Dataset}/{scenario}/ to comparison/results/)
python comparison/add_mae_{experiment}_results.py --scenario {scenario}

# Step 3: Verify
python -c "import json; print(json.load(open('comparison/results/{exp}/results.json'))['models'].keys())"
```

### Command Line Arguments (Baseline Script)
**Reference**: `comparison/run_wadi_14days.py` argparse section

| Argument | Description |
|----------|-------------|
| `--only-simple` | Only simple baselines (fast test) |
| `--only-neural` | Only neural baselines |
| `--skip-at` | Skip Anomaly Transformer |
| `--skip-sota` | Skip all SOTA models |

---

## 4. Monitoring

### 4.1 Monitoring Commands
```bash
# GPU status
nvidia-smi

# Check running process
ps aux | grep "run_" | grep python

# Check completed models
ls comparison/results/{EXPERIMENT_NAME}/
```

### 4.2 Progress Report Format
```
#   Model                Status       ROC     PRC      F1    F1_T
--------------------------------------------------------------------
1   random               ✅          0.504   0.038   0.074   0.074
2   sensor_range         ✅          0.483   0.117   0.162   0.362
3   mlp                  🔄 Ep 3/10  -       -       -       -
4   transformer          ⏳          -       -       -       -
```

**Status**: ✅ Completed | 🔄 In progress | ⏳ Pending | ⏭️ Skipped | ❌ Failed

### 4.3 Stuck Detection
A process is stuck if:
- CPU usage is 0% but process running
- No file changes in >5 minutes
- GPU memory full but no progress

```bash
# Check CPU usage
ps aux | grep "run_" | awk '{print $3}'

# Monitor file changes
watch -n 10 "ls -la comparison/results/{exp}/"
```

### 4.4 Monitoring Frequency
| Phase | Check Interval | Action |
|-------|----------------|--------|
| Pipeline start | Every 30 sec | Verify first model running |
| First model training | Every 1 min | Check progress, update table |
| Stable operation | Every 5 min | Update progress table |
| Long operations (SOTA) | Every 10 min | Check GPU, update table |

**Report progress table at each check interval.**

---

## 5. New Experiment

### 5.1 Required Files

**Reference**: Existing implementations for WaDi/SWaT

| File Type | Location | Reference File |
|-----------|----------|----------------|
| Data Loader | `mae_anomaly/datasets/loaders.py` | See `load_wadi()`, `load_swat()` functions |
| Config File | `scripts/ablation/configs/{config}.py` | `scripts/ablation/configs/20260131_121350_phase2.py` |
| Baseline Runner | `comparison/run_{experiment}.py` | `comparison/run_wadi_14days.py` |
| MAE Integration | `comparison/add_mae_{exp}_results.py` | `comparison/add_mae_14days_results.py` |

### 5.2 Data Loader Requirements

**Reference**: `mae_anomaly/datasets/loaders.py`

Add a new loader function:
```python
def load_{dataset}(scenario: str = 'default') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load {dataset} dataset.

    Returns:
        train_X, train_y, test_X, test_y
    """
    # Load and preprocess data
    # Return splits
```

### 5.3 Steps for New Experiment
1. Add data loader function → `mae_anomaly/datasets/loaders.py`
2. Create config file → `scripts/ablation/configs/{config}.py`
3. Create baseline runner → `comparison/run_{experiment}.py`
4. Create MAE integration script → `comparison/add_mae_{experiment}_results.py`
5. Create EXPERIMENTS.md → `comparison/results/{exp}/`
6. Run: MAE ablation → Baselines → Add MAE → Verify

---

## 6. Results Structure

### Directory Layout

**Verify with**: `ls -la comparison/results/WaDi_A1_14days/`

```
comparison/results/{EXPERIMENT}/
├── results.json           # Combined results (includes MAE after add_mae script)
├── EXPERIMENTS.md         # Documentation (required)
└── {model_name}/          # One per baseline model
    ├── metadata.json      # Model config + metrics
    ├── scores.npy         # Anomaly scores
    └── model/             # Weights (neural only)
```

**Note**: MAE results are NOT stored in separate directories. They are added directly to `results.json` by the `add_mae_*_results.py` script.

### metadata.json Format

**Reference**: `comparison/results/WaDi_A1_14days/mlp/metadata.json`

```json
{
  "model_name": "mlp",
  "timestamp": "2026-02-09T23:57:17.170867",
  "metrics": {
    "point_level": {
      "prc_auc": 0.264,
      "roc_auc": 0.806,
      "f1_score": 0.346,
      "recall": 0.244,
      "precision": 0.595
    },
    "f1_t": {
      "f1_t": 0.398,
      "precision_t": 0.470,
      "recall_t": 0.345
    },
    "pa_k": {
      "pa_10": { "prc_auc": ..., "roc_auc": ..., "f1_score": ..., "recall": ..., "precision": ... },
      "pa_20": { ... },
      "pa_50": { ... },
      "pa_80": { ... },
      "pa_100": { ... }
    },
    "timing": {
      "train_time": 7.12,
      "inference_time": 0.24
    }
  }
}
```

### EXPERIMENTS.md Template (REQUIRED)

**Reference**: `comparison/results/WaDi_A1_14days/EXPERIMENTS.md`

**Required Sections**:
1. **Dataset Description** - Source, Description, Purpose
2. **Data Statistics** - Features, Train/Test Samples, Anomaly Ratio, Segments
3. **Data Split** - Diagram showing train/test split
4. **Model Configurations**
   - Simple Baselines (5 models)
   - Neural Baselines (4 models, with epochs=10, train_stride=11)
   - SOTA Models (6 models, with epochs=10, train_stride=11)
5. **Results**
   - Baseline Results table (all 15 models)
   - MAE Models config table (Config Name, Window, Patch, Encoder, TD, SD, Scoring)
   - MAE Results table
6. **Key Findings** - Summary of best performers
7. **Running the Experiment** - Commands
8. **Results Location** - Directory structure

**Results Table Format** (8 metric columns):
```
| Model | ROC | PRC | F1 | F1_T | PA20_F1 | PA20_PRC | PA80_F1 | PA80_PRC |
```

---

## 7. Troubleshooting

| Problem | Solution |
|---------|----------|
| **OOM** | Reduce batch_size, or use `--only-simple` first |
| **Stuck** | Kill process (`kill -9 PID`), resume (script skips completed) |
| **Invalid JSON** | Verify: `python -c "import json; json.load(open('path'))"` |
| **Missing model** | Delete model dir, rerun script |

---

## Checklist Before Running

- [ ] **Todo list created with ALL experiment steps**
- [ ] `conda activate dc_vis`
- [ ] ALL DL models: epochs=10, train_stride=11
- [ ] Running in foreground (NO &, NO nohup)
- [ ] Monitoring ready (nvidia-smi, ps aux)
- [ ] EXPERIMENTS.md template prepared

---

*Last updated: 2026-02-15*
