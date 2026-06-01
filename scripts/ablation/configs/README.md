# Ablation Config Files

## Migration from run_*.py scripts

As of 2026-02-15, all `scripts/run_*.py` scripts have been migrated to a unified config-based system using `run_ablation.py`.

### New System

**Single Entry Point:**
```bash
python scripts/ablation/run_ablation.py --config scripts/ablation/configs/<config_file>.py
```

**Dataset Types:**
- `simulation` - Generated time series data (default)
- `swat_A1A2` - SWaT A1 (normal) + A2 (attack) combined
- `swat_A1A2_swap` - SWaT with swapped A2 halves
- `wadi_14days_A1` - WaDi 14 days + A1 attack
- `wadi_14days_A2` - WaDi 14 days + A2 attack
- `wadi_A2` - WaDi A2 attack only

### Config File Format

```python
# Dataset selection
DATASET_TYPE = 'swat_A1A2'  # or 'simulation', 'wadi_14days_A1', etc.

# Phase metadata
PHASE_NAME = "my_experiment"
PHASE_DESCRIPTION = "Description of experiment"

# Base configuration (applied to all experiments)
BASE_CONFIG = {
    # Model architecture    'seq_length': 500,
    'num_epochs': 50,
    'learning_rate': 2e-3,
    # ... other config params
}

# Experiment variations
EXPERIMENTS = [
    {'name': 'exp1', 'config': {}},  # Uses BASE_CONFIG
    {'name': 'exp2', 'config': {'num_epochs': 100}},  # Overrides
]

# Scoring modes
SCORING_MODES = ['default']
```

### Old Scripts (Archived)

The following scripts have been replaced by configs:

| Old Script | New Config Example | Dataset Type |
|-----------|-------------------|--------------|
| `run_mae_baseline.py` | `simulation_test.py` | `simulation` |
| `run_swat_ablation.py` | `swat_A1A2_test.py` | `swat_A1A2` |
| `run_wadi_14days_ablation.py` | `wadi_14days_A1_test.py` | `wadi_14days_A1` |

All old scripts are backed up in `.trash/20260215_run_scripts/`.

### Benefits

- ✅ Single codebase (~4,700 lines eliminated)
- ✅ Easy experiment reproduction (just share config file)
- ✅ No code duplication
- ✅ Centralized bug fixes
- ✅ Type-safe config validation
