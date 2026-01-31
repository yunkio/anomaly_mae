# Backup: 2026-01-31 Parameter Cleanup

## What was removed

### 1. `masking_strategy` parameter
- **Was**: `'patch'` or `'feature_wise'` option in Config
- **Decision**: Fixed to `'patch'`. Feature-wise masking showed no benefit in Phase 1 ablation (experiment 040_feature_wise_mask). The `feature_wise_masking()` method in model.py and all related branching code were deleted.

### 2. `'normalized'` scoring mode
- **Was**: One of three scoring modes: `['default', 'adaptive', 'normalized']`
- **Decision**: Removed `'normalized'`. Phase 1 analysis showed normalized scoring consistently underperformed adaptive and default. Only `['default', 'adaptive']` retained.

## Files modified
- `mae_anomaly/config.py` - Removed field and comments
- `mae_anomaly/model.py` - Removed `feature_wise_masking()` method and branching
- `mae_anomaly/evaluator.py` - Removed normalized scoring branch
- `mae_anomaly/trainer.py` - Removed normalized scoring branch
- `mae_anomaly/visualization/base.py` - Removed normalized branches
- `mae_anomaly/visualization/best_model_visualizer.py` - Removed normalized branches
- `mae_anomaly/visualization/stage2_visualizer.py` - Removed masking_strategy display
- `scripts/ablation/run_ablation.py` - Removed masking_strategy from records
- `scripts/ablation/configs/20260127_052220_phase1.py` - Removed masking_strategy, normalized
- `scripts/profile_pipeline.py` - Removed masking_strategy, normalized
- `scripts/visualize_all.py` - Removed masking_strategy from param_keys
