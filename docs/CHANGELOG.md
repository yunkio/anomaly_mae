# Changelog

## 2026-01-24 (Update 24): Per-Feature Min-Max Normalization

### Summary

Replaced data clipping (`np.clip(signals, 0, 1)`) with per-feature min-max normalization. This preserves relative anomaly magnitudes and eliminates boundary artifacts.

### Changes

#### 1. Added Normalization Function

**Modified Files**:
- `mae_anomaly/dataset_sliding.py`

**New Function**:
```python
def _normalize_per_feature(signals: np.ndarray) -> np.ndarray:
    """Per-feature min-max normalization to [0, 1] range.

    This is preferred over clipping because:
    1. Preserves relative magnitude of anomalies (spikes won't be capped)
    2. No artificial saturation at boundaries
    3. More realistic simulation of real-world data preprocessing
    """
    signals = signals.copy()
    for f in range(signals.shape[1]):
        min_val = signals[:, f].min()
        max_val = signals[:, f].max()
        if max_val - min_val > 1e-8:
            signals[:, f] = (signals[:, f] - min_val) / (max_val - min_val)
        else:
            signals[:, f] = 0.5
    return signals.astype(np.float32)
```

---

#### 2. Replaced Clipping with Normalization

**Locations Changed**:

| Method | Before | After |
|--------|--------|-------|
| `_generate_simple_normal_series()` | `np.clip(signals, 0, 1)` | `_normalize_per_feature(signals)` |
| `generate()` | `np.clip(signals, 0, 1)` | `_normalize_per_feature(signals)` |

---

#### 3. Why This Change?

| Aspect | Clipping | Min-Max Normalization |
|--------|----------|----------------------|
| Spike anomalies | Capped at 1.0 (info loss) | Full magnitude preserved |
| Boundary behavior | Flat saturation | Natural distribution |
| Relative magnitudes | Distorted | Preserved exactly |
| Real-world similarity | Artificial | Matches preprocessing |

---

### Documentation Updates

- **DATASET.md**: Added "Data Normalization" section, updated Safety Constraints table
- **CHANGELOG.md**: This entry

---

## 2026-01-24 (Update 23): Dataset Visualization Improvements

### Summary

Improved dataset visualization quality by using dedicated datasets for plotting (without anomaly contamination), added before/after comparisons at same window positions, and cleaned up redundant/misleading visualizations.

### Changes

#### 1. Added `inject_anomalies` Parameter to Generator

**Modified Files**:
- `mae_anomaly/dataset_sliding.py`

**New Parameter**:
```python
def generate(self, inject_anomalies: bool = True) -> Tuple[...]:
    """
    Args:
        inject_anomalies: If True (default), inject anomalies.
                          If False, return pure normal data.
    """
```

This allows visualization code to generate clean normal data for complexity feature demonstrations.

---

#### 2. Improved Dataset Visualizations

**Modified Files**:
- `mae_anomaly/visualization/data_visualizer.py`

**Changes**:

| Function | Change |
|----------|--------|
| `plot_anomaly_generation_rules()` | Show only 1 example per anomaly type (was 2) |
| `plot_normal_complexity_features()` | Uses `inject_anomalies=False` for clean comparison |
| `plot_complexity_comparison()` | Uses `inject_anomalies=False` for clean comparison |
| `plot_complexity_vs_anomaly()` | **Completely redesigned**: Before/after comparison at same window position |
| `plot_dataset_statistics()` | **Removed** (hardcoded values were misleading) |

**New `plot_complexity_vs_anomaly()` Design**:
- Row 1: Complexity features (gray=before, blue=after) at same window position
- Row 2: Anomaly injection (gray=before, red=after) at same window position
- Allows clear visualization of what each feature/anomaly actually changes

---

#### 3. Stage 1 Visualization Cleanup

**Modified Files**:
- `mae_anomaly/visualization/experiment_visualizer.py`

**Changes**:

| Function | Change |
|----------|--------|
| `plot_metric_correlations()` | **Removed** (not useful for hyperparameter analysis) |
| `plot_parallel_coordinates()` | **Added interpretation guide** panel explaining how to read the plot |

---

### Documentation Updates

- **VISUALIZATIONS.md**: Updated tables and usage examples
- **CHANGELOG.md**: This entry

---

## 2026-01-24 (Update 22): Comprehensive Visualization Style Consistency

### Summary

Extended VIS_COLORS with additional semantic color keys and applied consistent styling across ALL visualization files, eliminating hardcoded color values.

### Changes

#### 1. Extended VIS_COLORS Constants

**Modified Files**:
- `mae_anomaly/visualization/base.py`

**New Color Keys Added**:
```python
VIS_COLORS = {
    # Primary data types (existing)
    'normal': '#3498DB',
    'anomaly': '#E74C3C',
    'disturbing': '#F39C12',
    'teacher': '#27AE60',
    'student': '#9B59B6',
    'total': '#2ECC71',
    # Region highlighting (NEW)
    'anomaly_region': '#E74C3C',
    'masked_region': '#F1C40F',
    'normal_region': '#27AE60',
    # Darker variants (NEW)
    'normal_dark': '#2980B9',
    'anomaly_dark': '#C0392B',
    'student_dark': '#8E44AD',
    # Detection outcomes (NEW)
    'true_positive': '#27AE60',
    'true_negative': '#3498DB',
    'false_positive': '#F39C12',
    'false_negative': '#E74C3C',
    # General purpose (NEW)
    'baseline': 'black',
    'reference': 'gray',
    'threshold': '#27AE60',
}
```

---

#### 2. Applied VIS_COLORS Across All Visualizers

**Modified Files**:
- `mae_anomaly/visualization/best_model_visualizer.py`
- `mae_anomaly/visualization/experiment_visualizer.py`
- `mae_anomaly/visualization/stage2_visualizer.py`
- `mae_anomaly/visualization/training_visualizer.py`
- `mae_anomaly/visualization/data_visualizer.py`
- `mae_anomaly/visualization/architecture_visualizer.py`

**Changes**:
- Replaced ALL hardcoded hex color values (e.g., `'#3498DB'`) with `VIS_COLORS['normal']`
- Replaced ALL hardcoded color names (e.g., `'red'`, `'yellow'`) with `VIS_COLORS` keys
- Added VIS_COLORS import to files that were missing it
- Used semantic color keys (e.g., `'anomaly_region'` for highlighting anomalies)

---

### Documentation Updates

- **VISUALIZATIONS.md**: Updated VIS_COLORS table with all new keys
- **CHANGELOG.md**: This entry

---

## 2026-01-24 (Update 21): Self-Distillation Training Improvements

### Summary

Added encoder gradient detachment for student decoder, configurable warm-up epochs, detailed learning curve visualization, and consistent color/marker scheme across all visualizations.

### Changes

#### 1. Encoder Gradient Detachment for Student Decoder

**Modified Files**:
- `mae_anomaly/model.py`

**Changes**:
- Student decoder now receives `.detach()`ed encoder output
- Encoder is only updated by teacher reconstruction loss
- Prevents student's conflicting objectives from corrupting encoder representations

**Implementation**:
```python
# In forward():
if self.config.use_student:
    student_latent = latent.detach()  # Detach encoder output
    student_output = self.student_decoder(student_latent)
```

---

#### 2. Configurable Teacher-Only Warm-up Epochs

**Modified Files**:
- `mae_anomaly/config.py`
- `mae_anomaly/trainer.py`
- `mae_anomaly/loss.py`

**New Parameter**:
- `teacher_only_warmup_epochs: int = 1` (default)

**Changes**:
- First N epochs train only teacher model (no discrepancy/student loss)
- Added `teacher_only` parameter to loss function
- Allows teacher to learn basic reconstruction before introducing discrepancy

---

#### 3. Detailed Learning Curve Visualization

**Modified Files**:
- `mae_anomaly/loss.py`
- `mae_anomaly/trainer.py`
- `mae_anomaly/visualization/best_model_visualizer.py`
- `scripts/visualize_all.py`

**New Metrics Tracked**:
- `train_teacher_recon_normal`: Teacher recon loss on normal samples
- `train_teacher_recon_anomaly`: Teacher recon loss on anomaly samples
- `train_student_recon_normal`: Student recon loss on normal samples
- `train_student_recon_anomaly`: Student recon loss on anomaly samples

**New Visualization**: `learning_curve.png`
- 2x3 grid showing detailed loss breakdown:
  - Teacher Reconstruction (Normal vs Anomaly)
  - Student Reconstruction (Normal vs Anomaly)
  - Discrepancy Loss (Normal vs Anomaly)
  - Normal Data: Teacher vs Student
  - Anomaly Data: Teacher vs Student
  - All Losses Combined

---

#### 4. Consistent Visualization Color/Marker Scheme

**Modified Files**:
- `mae_anomaly/visualization/base.py`
- `mae_anomaly/visualization/__init__.py`
- `mae_anomaly/visualization/best_model_visualizer.py`

**New Style Constants** (in `base.py`):
```python
VIS_COLORS = {
    'normal': '#3498DB',      # Blue for normal data
    'anomaly': '#E74C3C',     # Red for anomaly data
    'disturbing': '#F39C12',  # Orange for disturbing normal
    'teacher': '#27AE60',     # Green for teacher model
    'student': '#9B59B6',     # Purple for student model
    'total': '#2ECC71',       # Green for totals
}

VIS_MARKERS = {
    'discrepancy': 's',       # Square for discrepancy loss
    'teacher_recon': 'o',     # Circle for teacher reconstruction
    'student_recon': '^',     # Triangle for student reconstruction
    'total': 'D',             # Diamond for total/combined
}
```

**Applied to**:
- `plot_learning_curve()`: Full color/marker scheme
- `plot_discrepancy_trend()`: Consistent colors
- `plot_pure_vs_disturbing_normal()`: Consistent colors for bar charts

---

### Documentation Updates

- **ARCHITECTURE.md**: Added encoder gradient detachment and warm-up epochs documentation
- **VISUALIZATIONS.md**: Added VIS_COLORS/VIS_MARKERS documentation and learning_curve.png
- **CHANGELOG.md**: This entry

---

## 2026-01-23 (Update 20): Quick Search Dataset Configuration

### Changes
- `quick_length`: 100,000 → 200,000 timesteps
- `quick_train_ratio`: 0.3 → 0.2 (20% train, 80% test)
- "Anomaly Types" → "Anomaly Types (samples)" for clarity
- Removed sample count warning messages

### Files Modified
- `scripts/run_experiments.py`
- `mae_anomaly/dataset_sliding.py`

---

## 2026-01-23 (Update 19): Enhanced Dataset Statistics Display

### Changes
- Now displays **3 dataset views**: Train Set (Raw), Test Set (Raw), Test Set (Downsampled)
- Each view shows **Anomaly Types** distribution (per sample, not per region)
- Clearer output format for experiment monitoring

### Output Format
```
[Quick Dataset - Train Set (Raw)]
  - Pure Normal: X,XXX (XX.X%)
  - Anomaly: XXX (X.X%)
  Anomaly Types:
    - spike: XX
    - memory_leak: XX
    ...

[Quick Dataset - Test Set (Raw)]
  ...

[Quick Dataset - Test Set (Downsampled to 65%:15%:25%)]
  ...
```

### Files Modified
- `scripts/run_experiments.py`

---

## 2026-01-23 (Update 18): Train/Test Set Composition Fix

### Problem
- Only test set statistics were displayed, train set was missing
- Test set ratios were hardcoded as absolute counts (1200:300:500)

### Changes

#### 1. Train/Test Statistics Display
- Now shows both **Train Set (Raw)** and **Test Set (Raw)** statistics
- Train set: no downsampling, natural distribution (~5% anomaly from interval_scale)
- Test set: shows raw distribution + target ratio info

#### 2. Test Set Ratio-Based Downsampling
- **Before**: Hardcoded counts (1200:300:500 = 60:15:25)
- **After**: Ratio-based (65:15:25) scaled to `num_test_samples`
- Config now uses `test_ratio_*` instead of `test_target_*`

#### 3. Dataset Composition
| Split | Pure Normal | Disturbing | Anomaly | Downsampling |
|-------|-------------|------------|---------|--------------|
| Train | Natural | Natural | ~5% | None |
| Test | 65% | 15% | 25% | Yes |

### Files Modified
- `mae_anomaly/config.py`
- `scripts/run_experiments.py`

---

## 2026-01-23 (Update 17): Fix Anomaly Ratio in Quick Search

### Problem
- Previous fix scaled interval proportionally: `quick_interval_scale = base * (quick/full)`
- This reduced interval → more frequent anomalies → 19% anomaly ratio (too high)

### Solution
- Use same `interval_scale` for both quick and full search
- Anomaly ratio determined by interval_scale, not data length
- Consistent ~5% anomaly ratio regardless of dataset size

### Files Modified
- `scripts/run_experiments.py`

---

## 2026-01-23 (Update 16): Quick Search Dataset Size Increase

### Changes
- `quick_length`: 66000 → 100000 (more data for quick search)
- Warning threshold: 200 → 300 (suppress warnings when samples >= 300)

### Files Modified
- `scripts/run_experiments.py`
- `mae_anomaly/dataset_sliding.py`

---

## 2026-01-23 (Update 15): Reduce Periodicity in Complex Normal Data

### Summary

Improved normal data generation to be less strictly periodic, making anomaly detection more challenging and realistic.

### Changes

#### 1. Remove Hard Clipping
- **Before**: Normal data was clipped to `[0.05, 0.70]` range
- **After**: No clipping - natural value distribution
- Reason: Hard clipping made normal data unrealistically bounded and easy to classify

#### 2. Irrational Frequency Ratios
- **Before**: `freq2 ≈ freq1/10`, `freq3 ≈ freq1/50` (integer-like ratios)
- **After**: `freq2 = freq1/(π×[2.8-3.5])`, `freq3 = freq1/(π²×[1.5-2.5])`
- Reason: Integer ratios cause beat patterns to repeat; irrational ratios (π-based) prevent exact repetition

#### 3. Phase Jitter
- **New feature**: Slowly-varying phase offset added to sinusoidal components
- Parameters: `enable_phase_jitter=True`, `phase_jitter_sigma=0.002`, `phase_jitter_smoothing=500`
- Applied with decreasing weight per frequency: fast (1.0), medium (0.7), slow (0.4)
- Result: Even with same frequencies, patterns drift over time

### New NormalDataComplexity Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_phase_jitter` | True | Enable phase jitter |
| `phase_jitter_sigma` | 0.002 | Random walk step size |
| `phase_jitter_smoothing` | 500 | Smoothing window |

### Files Modified
- `mae_anomaly/dataset_sliding.py`
- `docs/DATASET.md`
- `docs/CHANGELOG.md`

---

## 2026-01-23 (Update 14): Experiment Configuration Updates

### Summary

Simplified num_patches options, doubled full search dataset, and improved warning thresholds.

### Changes

#### 1. `num_patches` Grid Reduction
- **Before**: `[10, 25, 50]` (3 values)
- **After**: `[10, 25]` (2 values)
- Reason: 50 patches = 2 timesteps per patch, too granular for effective pattern learning

#### 2. Full Search Dataset Size Doubled
- **Before**: `full_length = 220000`
- **After**: `full_length = 440000`
- Provides more training data for Stage 2 full search

#### 3. Warning Threshold for Sample Count
- Warnings now only appear when sample count < 200 (previously: any shortage)
- Reduces noise during quick searches with limited data

#### 4. Grid Combinations
- **Before**: 2×2×3×3×2×2×2×2×2 = 1152 combinations
- **After**: 2×2×2×3×2×2×2×2×2 = 768 combinations

### Files Modified
- `scripts/run_experiments.py`
- `mae_anomaly/dataset_sliding.py`
- `docs/ABLATION_STUDIES.md`
- `docs/VISUALIZATIONS.md`

---

## 2026-01-23 (Update 13): MAE Architecture Enhancements

### Summary

Added two new architecture parameters for standard MAE masking and separate mask tokens, along with experiment infrastructure improvements.

### New Parameters

#### 1. `mask_after_encoder` (config.py)
- **False (default)**: Mask tokens go through encoder (current behavior)
- **True**: Standard MAE - encode visible patches only, insert mask tokens before decoder

**Implementation**:
- Added `_encode_visible_only()` method: Encodes only visible patches
- Added `_insert_mask_tokens_and_unshuffle()` method: Inserts mask tokens at correct positions
- Modified `forward()` to support both modes

#### 2. `shared_mask_token` (config.py)
- **True (default)**: Single mask token shared between teacher/student
- **False**: Separate learnable mask tokens for teacher and student decoders

**Implementation**:
- Added `_get_mask_token(for_decoder)` method to retrieve appropriate token
- Separate `teacher_mask_token` and `student_mask_token` when not shared

### Experiment Changes

**Modified Files**:
- `scripts/run_experiments.py`
- `scripts/visualize_all.py`

**Parameter Grid Updates**:
```python
DEFAULT_PARAM_GRID = {
    # ... existing parameters ...
    'mask_after_encoder': [False, True],
    'shared_mask_token': [True, False],
}
# Total combinations: 2*2*3*3*2*2*2*2*2 = 1152
```

**Dataset Size Changes**:
- `quick_length`: 200000 → 66000 (1/3 reduction)
- `full_length`: 440000 → 220000 (1/2 reduction)
- `full_epochs`: fixed at 2

**Stage 2 Selection Updates**:
- Added `mask_after_encoder` (top 5 per value)
- Added `shared_mask_token` (top 5 per value)

**Output Cleanup**:
- Removed "Train: X, Test: Y" from Stage 1/2 headers (values were outdated)

### Documentation Updates

**Modified Files**:
- `docs/ARCHITECTURE.md`: Added MAE Masking Architecture and Mask Token Configuration sections
- `docs/ABLATION_STUDIES.md`: Added sections 8 (Mask After Encoder) and 9 (Shared Mask Token)

---

## 2026-01-23 (Update 12.2): Complexity Visualization

### Summary

Added 3 new visualization functions to explain NormalDataComplexity features.

### Changes

**Modified Files**:
- `mae_anomaly/visualization/data_visualizer.py`

**New Visualizations**:
1. `plot_normal_complexity_features()` - Shows each of 6 complexity features individually
2. `plot_complexity_comparison()` - Simple vs Complex normal data side-by-side
3. `plot_complexity_vs_anomaly()` - Why complexity features don't resemble anomalies

**Output Files**:
- `normal_complexity_features.png` - 6-panel feature explanation
- `complexity_comparison.png` - Simple vs Complex comparison
- `complexity_vs_anomaly.png` - Complexity vs Anomaly discrimination

---

## 2026-01-23 (Update 12.1): Experiment Integration

### Summary

- Exported `NormalDataComplexity` from `mae_anomaly` package
- Updated `run_experiments.py` to use complexity features by default
- Added `--no-complexity` CLI flag to disable complexity features

### Changes

**Modified Files**:
- `mae_anomaly/__init__.py`: Export `NormalDataComplexity`
- `scripts/run_experiments.py`: Use complexity by default, add CLI flag

**Usage**:
```bash
# Default: with complexity (recommended)
python scripts/run_experiments.py

# Without complexity (simple patterns)
python scripts/run_experiments.py --no-complexity
```

---

## 2026-01-23 (Update 12): Normal Data Complexity Features

### Summary

Added 6 configurable complexity features to make normal data more realistic and challenging for anomaly detection models. All features are designed to NOT be confused with anomaly patterns.

### Changes

#### 1. NormalDataComplexity Configuration

**Modified Files**:
- `mae_anomaly/dataset_sliding.py`

**Added**:
- `NormalDataComplexity` dataclass with on/off switches for each feature
- All features enabled by default, individually toggleable

```python
@dataclass
class NormalDataComplexity:
    enable_complexity: bool = True
    enable_regime_switching: bool = True
    enable_multi_scale_periodicity: bool = True
    enable_heteroscedastic_noise: bool = True
    enable_varying_correlations: bool = True
    enable_drift: bool = True
    enable_normal_bumps: bool = True
    # ... detailed parameters for each
```

---

#### 2. Six Complexity Features Implemented

| Feature | Description | Transition Time |
|---------|-------------|-----------------|
| **Regime Switching** | Different operational states | 1500 timesteps |
| **Multi-Scale Periodicity** | 3 overlapping frequencies | Continuous |
| **Heteroscedastic Noise** | Load-dependent variance | Continuous |
| **Time-Varying Correlations** | Slowly changing correlations | Period 15000 ts |
| **Bounded Drift (O-U)** | Mean-reverting random walk | Continuous |
| **Normal Bumps** | Small, gradual load increases | Gaussian envelope |

---

#### 3. Safety Constraints

All complexity features enforce strict constraints to distinguish from anomalies:

| Constraint | Value | Reason |
|------------|-------|--------|
| Transition time | >= 1000 ts | Anomalies are 3-150 ts |
| Value range | [0.05, 0.70] | Anomalies push to 0.7-1.0 |
| Bump magnitude | max 0.10 | Spike adds 0.3-0.6 |
| Bump duration | 100-300 ts | Spike is 10-25 ts |

---

#### 4. Documentation Updated

**Modified Files**:
- `docs/DATASET.md`

**Added**:
- New section "Normal Data Complexity Features"
- Detailed documentation for each feature
- Configuration examples
- Safety constraints explanation

---

### Usage

```python
from mae_anomaly.dataset_sliding import NormalDataComplexity, SlidingWindowTimeSeriesGenerator

# Full complexity (default)
complexity = NormalDataComplexity()

# Simple mode
complexity = NormalDataComplexity(enable_complexity=False)

# Custom
complexity = NormalDataComplexity(
    enable_regime_switching=True,
    enable_normal_bumps=False,
)

generator = SlidingWindowTimeSeriesGenerator(
    total_length=440000,
    complexity=complexity,
    seed=42
)
```

---

## 2026-01-23 (Update 11): Visualization Quality Improvements

### Changes

#### 1. Removed Redundant anomaly_types Visualization

**Modified Files**:
- `mae_anomaly/visualization/data_visualizer.py`

**Changes**:
- Removed `plot_anomaly_types()` from `generate_all()` - redundant with `plot_anomaly_generation_rules()`
- The `anomaly_generation_rules.png` provides more informative visualization using actual dataset samples

---

#### 2. Improved feature_examples Visualization

**Modified Files**:
- `mae_anomaly/visualization/data_visualizer.py`

**Changes**:
- Now displays ALL 8 features (was hardcoded to 5)
- Uses actual `FEATURE_NAMES` for labels (CPU_Usage, Memory_Usage, etc.)
- Dynamic subplot layout based on feature count

---

#### 3. Improved sample_types Visualization with Diverse Sampling

**Modified Files**:
- `mae_anomaly/visualization/data_visualizer.py`

**Changes**:
- Added `select_diverse()` function to randomly sample from shuffled data
- Prevents showing overlapping/similar samples due to stride=10
- Ensures visual diversity in sample type comparison

---

#### 4. Improved patchify_modes as Conceptual Flow Diagrams

**Modified Files**:
- `mae_anomaly/visualization/architecture_visualizer.py`

**Changes**:
- Complete rewrite of `plot_patchify_modes()`
- Now shows conceptual processing pipeline with boxes and arrows
- Three modes clearly differentiated:
  - **CNN-First**: Input → CNN → Patchify → Embed
  - **Patch-CNN**: Input → Patchify → CNN (per patch) → Embed
  - **Linear (MAE)**: Input → Patchify → Linear Projection
- Removed meaningless bar chart comparison

---

#### 5. Improved discrepancy_trend Visualization

**Modified Files**:
- `mae_anomaly/visualization/best_model_visualizer.py`

**Changes**:
- Added standard deviation bands (mean ± std shading)
- Added zoomed view of last patch region (masked region)
- Added box plots showing discrepancy distribution by sample type
- Added statistics text box with mean ± std values
- More informative for analyzing masked region behavior

---

#### 6. Fixed METRIC_COLUMNS in Stage2Visualizer

**Modified Files**:
- `mae_anomaly/visualization/stage2_visualizer.py`

**Changes**:
- Added missing metrics to `METRIC_COLUMNS`:
  - `disturbing_roc_auc`, `disturbing_f1`, `disturbing_precision`, `disturbing_recall`
  - `quick_roc_auc`, `quick_f1`, `quick_disturbing_roc_auc`
  - `roc_auc_improvement`, `selection_criterion`, `stage2_rank`
- Prevents metrics from being incorrectly treated as hyperparameters

---

### Benefits

1. **Cleaner visualizations**: Removed redundant plots, improved clarity
2. **More informative**: All features shown with proper names
3. **Better diversity**: Sample type visualization shows varied data
4. **Conceptual clarity**: Patchify modes now explain the processing pipeline
5. **Statistical rigor**: Discrepancy trend includes uncertainty bands
6. **Correct hyperparameter analysis**: Metrics no longer appear as hyperparameters in Stage 2 plots

---

## 2026-01-23 (Update 10): Dynamic Hyperparameter and Configuration Management

### Changes

#### 1. Dynamic param_keys in visualize_all.py

**Modified Files**:
- `scripts/visualize_all.py`

**Before**: Hardcoded list of hyperparameter keys
```python
param_keys = ['masking_ratio', 'masking_strategy', 'num_patches', ...]
```

**After**: Dynamically extracted from experiment metadata or results
```python
if exp_data['metadata'] and 'param_grid' in exp_data['metadata']:
    param_keys = list(exp_data['metadata']['param_grid'].keys())
else:
    # Fallback: extract from results DataFrame
    param_keys = [c for c in columns if c not in metric_cols]
```

---

#### 2. Dynamic Hyperparameter Lists in stage2_visualizer.py

**Modified Files**:
- `mae_anomaly/visualization/stage2_visualizer.py`

**Changes**:
- Added `METRIC_COLUMNS` class constant for known metric columns
- Added `_get_hyperparam_columns()` helper method
- `plot_all_hyperparameters()`: Now uses dynamic hyperparameter detection
- `plot_hyperparameter_interactions()`: Dynamically generates interaction pairs
- `plot_best_config_summary()`: Uses dynamic hyperparams with fallback descriptions

---

#### 3. Dynamic Categorical Parameters in experiment_visualizer.py

**Modified Files**:
- `mae_anomaly/visualization/experiment_visualizer.py`

**Changes**:
- Added `_get_categorical_params()` helper method
- `plot_summary_dashboard()`: Uses dynamically detected categorical params
- `generate_all()`: Uses dynamic categorical params for comparisons

---

#### 4. Robust get_anomaly_type_info in base.py

**Modified Files**:
- `mae_anomaly/visualization/base.py`

**Changes**:
- `get_anomaly_type_info()` now handles unknown anomaly types gracefully
- Auto-generates descriptions for new anomaly types not in known_info dict
- Always includes all types from `ANOMALY_TYPE_NAMES`

---

### Benefits

1. **No manual updates needed**: Adding new hyperparameters to `DEFAULT_PARAM_GRID` automatically includes them in visualizations
2. **No sync issues**: New anomaly types are automatically handled with auto-generated descriptions
3. **Reduced maintenance**: Less hardcoded values = fewer places to update when configuration changes
4. **Better error handling**: Fallback mechanisms prevent crashes from missing data

---

## 2026-01-23 (Update 9): Visualization Module Modularization

### Changes

#### 1. Modular Visualization Package

**New Directory Structure**:
```
mae_anomaly/
└── visualization/
    ├── __init__.py              # Module exports
    ├── base.py                  # Common utilities, colors, data loading
    ├── data_visualizer.py       # DataVisualizer class
    ├── architecture_visualizer.py  # ArchitectureVisualizer class
    ├── experiment_visualizer.py # ExperimentVisualizer (Stage 1)
    ├── stage2_visualizer.py     # Stage2Visualizer class
    ├── best_model_visualizer.py # BestModelVisualizer class
    └── training_visualizer.py   # TrainingProgressVisualizer class
```

**Modified Files**:
- [scripts/visualize_all.py](../scripts/visualize_all.py): Reduced from ~4900 lines to ~166 lines
- [mae_anomaly/visualization/](../mae_anomaly/visualization/): New modular package

**Benefits**:
- Cleaner, more maintainable code structure
- Each visualizer class in its own file
- Common utilities centralized in `base.py`
- Easy to extend with new visualizers

---

#### 2. Dynamic Color Management

**Modified Files**:
- `mae_anomaly/visualization/base.py`
- `mae_anomaly/visualization/best_model_visualizer.py`
- `mae_anomaly/visualization/training_visualizer.py`

**Changes**:
- Created `get_anomaly_colors()` function that dynamically generates colors for all anomaly types
- Created `SAMPLE_TYPE_COLORS` and `SAMPLE_TYPE_NAMES` constants
- Replaced all hardcoded color dictionaries with dynamic functions
- Colors now automatically adapt when anomaly types are added/removed

**Before** (hardcoded):
```python
colors = {
    'normal': '#3498DB',
    'spike': '#E74C3C',
    # ... manually maintained
}
```

**After** (dynamic):
```python
from mae_anomaly.visualization import get_anomaly_colors
colors = get_anomaly_colors()  # Automatically includes all anomaly types
```

---

#### 3. Dynamic plot_anomaly_generation_rules

**Modified Files**:
- `mae_anomaly/visualization/data_visualizer.py`

**Changes**:
- `plot_anomaly_generation_rules()` now dynamically generates visualizations based on `ANOMALY_TYPE_NAMES`
- Uses actual dataset examples instead of synthetic simulation
- Automatically adapts grid size based on number of anomaly types
- Gets anomaly info (length_range, characteristics) from `ANOMALY_TYPE_CONFIGS`

---

#### 4. Usage Update

**New Import Pattern**:
```python
# Old (from script)
from scripts.visualize_all import DataVisualizer, load_best_model

# New (from module)
from mae_anomaly.visualization import (
    DataVisualizer,
    ArchitectureVisualizer,
    ExperimentVisualizer,
    Stage2Visualizer,
    BestModelVisualizer,
    TrainingProgressVisualizer,
    setup_style,
    load_best_model,
    get_anomaly_colors,
)
```

**Running visualizations** (unchanged):
```bash
python scripts/visualize_all.py  # Still works the same way
```

---

## 2026-01-23 (Update 8): Point Spike Duration Change and Visualization Fixes

### Changes

#### 1. Point Spike Duration Change

**Modified Files**:
- [mae_anomaly/dataset_sliding.py](../mae_anomaly/dataset_sliding.py)
- [docs/DATASET.md](DATASET.md)

**Changes**:
- Point spike duration: (1, 3) → **(3, 5)** timesteps
- Still the shortest anomaly type, but more detectable

```python
# Before
7: {'length_range': (1, 3), 'interval_mean': 4000}

# After
7: {'length_range': (3, 5), 'interval_mean': 4000}
```

---

#### 2. Visualization Color Map Update

**Modified Files**:
- [scripts/visualize_all.py](../scripts/visualize_all.py)

**Changes**:
- Updated `plot_loss_by_anomaly_type()` colors: Added `point_spike` color
- Updated `plot_loss_scatter_by_anomaly_type()` colors: Fixed outdated anomaly type names (`noise`, `drift` → actual types)

**Before** (incorrect):
```python
colors = {
    'normal': '#3498DB',
    'spike': '#E74C3C',
    'memory_leak': '#F39C12',
    'noise': '#9B59B6',        # ← Wrong
    'drift': '#1ABC9C',         # ← Wrong
    'network_congestion': '#E67E22'
}
```

**After** (correct):
```python
colors = {
    'normal': '#3498DB',
    'spike': '#E74C3C',
    'memory_leak': '#F39C12',
    'cpu_saturation': '#9B59B6',
    'network_congestion': '#E67E22',
    'cascading_failure': '#1ABC9C',
    'resource_contention': '#16A085',
    'point_spike': '#E91E63',
}
```

---

#### 3. Anomaly-Type Performance Comparison Verification

**Existing Functions (Best Model)**:
- `plot_loss_by_anomaly_type()`: Loss distribution per anomaly type ✓
- `plot_performance_by_anomaly_type()`: Detection rate & mean score per type ✓
- `plot_loss_scatter_by_anomaly_type()`: Loss scatter per type ✓
- `plot_anomaly_type_case_studies()`: TP/FN examples per type ✓

**Existing Functions (Training Progress)**:
- `plot_anomaly_type_learning()`: Detection rate over epochs per type ✓

**Stage 1/2**: Designed for hyperparameter comparison, not anomaly-type analysis (by design)

---

## 2026-01-23 (Update 7): Point Spike Anomaly and Dataset Statistics

### Changes

#### 1. New Anomaly Type: Point Spike

**Modified Files**:
- [mae_anomaly/dataset_sliding.py](../mae_anomaly/dataset_sliding.py)
- [docs/DATASET.md](DATASET.md)

**New Anomaly Type**:
- **point_spike** (type 7): True point anomaly lasting only 3-5 timesteps
- **Unique characteristic**: 2+ random features spike simultaneously
- Makes threshold-based detection on individual features less effective

```python
# Point spike configuration
7: {'length_range': (3, 5), 'interval_mean': 4000}

# Injection logic
def _inject_point_spike(self, signals, start, end):
    # Select 2+ random features
    num_features_to_spike = np.random.randint(2, self.num_features + 1)
    features_to_spike = np.random.choice(self.num_features, num_features_to_spike, replace=False)
    # Apply spike magnitude +0.3 to +0.6 to each selected feature
```

---

#### 2. Dataset Statistics Output

**Modified Files**:
- [scripts/run_experiments.py](../scripts/run_experiments.py)

**New Feature**: When running experiments, dataset statistics are now printed:

```
[Quick Dataset Statistics - Test Set (Raw)]
Sample Types:
  - Pure Normal:       XXXX (XX.X%)
  - Disturbing Normal: XXX (XX.X%)
  - Anomaly:           XXX (XX.X%)
  - Total:             XXXX

Anomaly Types (region count):
  - spike: XX
  - memory_leak: XX
  - cpu_saturation: XX
  - network_congestion: XX
  - cascading_failure: XX
  - resource_contention: XX
  - point_spike: XX
```

---

#### 3. Visualization Code Update

**Modified Files**:
- [scripts/visualize_all.py](../scripts/visualize_all.py)

**Changes**:
- `plot_anomaly_type_case_studies()`: Now dynamically uses `ANOMALY_TYPE_NAMES` instead of hardcoded list
- `plot_anomaly_type_learning()`: Now dynamically uses `ANOMALY_TYPE_NAMES` instead of hardcoded list
- Handles any number of anomaly types automatically

---

## 2026-01-23 (Update 6): Reduce Full Search Epochs

### Changes

- Changed `full_epochs` default from **3 to 2** for faster experimentation
- Updated files:
  - [scripts/run_experiments.py](../scripts/run_experiments.py): Function parameter and argparse default
  - [README.md](../README.md): Experiment settings table
  - [docs/ABLATION_STUDIES.md](ABLATION_STUDIES.md): Stage 2 description
  - [docs/VISUALIZATIONS.md](VISUALIZATIONS.md): Settings table

---

## 2026-01-23 (Update 5): Threshold Fix and Hypothesis Verification

### Changes

#### 1. Disturbing Normal Evaluation Fix

**Modified Files**:
- [mae_anomaly/evaluator.py](../mae_anomaly/evaluator.py)

**Problem**:
- Disturbing normal evaluation was using a **separate threshold** calculated only from pure_normal and disturbing_normal samples
- This was incorrect - should use the **global threshold** from the entire dataset

**Fix**:
- Now uses the global optimal threshold (calculated from all samples) for disturbing normal evaluation
- ROC-AUC is threshold-free, so no change needed there
- Precision/Recall/F1 now use the same threshold as overall evaluation

**Before** (incorrect):
```python
d_fpr, d_tpr, d_thresholds = roc_curve(disturbing_labels, disturbing_scores)
d_optimal_idx = np.argmax(d_tpr - d_fpr)
d_threshold = d_thresholds[d_optimal_idx]  # Separate threshold!
d_predictions = (disturbing_scores > d_threshold).astype(int)
```

**After** (correct):
```python
# Use GLOBAL threshold (from entire dataset)
d_predictions = (disturbing_scores > threshold).astype(int)
```

---

#### 2. Hypothesis Verification Visualization

**Modified Files**:
- [scripts/visualize_all.py](../scripts/visualize_all.py)
- [docs/VISUALIZATIONS.md](../docs/VISUALIZATIONS.md)

**New Visualization**: `hypothesis_verification.png`

Verifies 4 hypotheses about why disturbing normal might outperform pure normal:

1. **H1: Anomaly Hint** - Does anomaly in window increase score?
   - Scatter plot of anomaly ratio vs total score

2. **H2: Transition Effect** - Does recent anomaly affect last patch?
   - Scatter plot of distance from anomaly to last patch vs score

3. **H3: Variance Analysis** - Does pure normal have higher variance?
   - Violin plot comparing score distributions

4. **H4: Classification Rates** - How do FP/TP rates compare with global threshold?
   - Bar chart of classification rates

---

#### 3. Quick Search Epoch Reduction

**Modified Files**:
- [scripts/run_experiments.py](../scripts/run_experiments.py)
- [README.md](../README.md)
- [docs/ABLATION_STUDIES.md](../docs/ABLATION_STUDIES.md)
- [docs/VISUALIZATIONS.md](../docs/VISUALIZATIONS.md)

**Changes**:
- Stage 1 (Quick Search) epochs: 2 → **1**

**Rationale**:
- Single epoch sufficient for quick screening of 432 combinations
- Significantly reduces experiment time while maintaining ranking quality

**Updated Settings**:
| Stage | Epochs |
|-------|--------|
| Stage 1 (Quick) | 1 |
| Stage 2 (Full) | 3 |

---

## 2026-01-23 (Update 4): Estimated Time Display

### Changes

#### Time Estimation Feature

**Modified Files**:
- [scripts/run_experiments.py](../scripts/run_experiments.py)

**Changes**:
- Added time estimation based on first model training time
- Displays estimated time for Quick Search, Full Search, and Total
- Considers dataset size, epochs, and model count differences

**Output Format**:
```
>>> Estimated Time (based on 1st model: X.Xs) <<<
  Quick Search: XX분 (432 models × X.Xs)
  Full Search:  XX분 (~60 models × X.Xs)
  Total:        XX분
  (Quick remaining: XX분)
```

**Calculation**:
- Quick Search: `first_model_time × n_models`
- Full Search: `first_model_time × (full_train/quick_train) × (full_epochs/quick_epochs) × n_stage2_models`
  - `full_train/quick_train = 22,000/6,000 ≈ 3.67`
  - `full_epochs/quick_epochs = 3/2 = 1.5`

---

## 2026-01-23 (Update 3): Stage 2 Selection Reduction and Epoch Fine-tuning

### Changes

#### 1. Quick Search Epoch Reduction

**Changes**:
- Stage 1 epochs: 5 → **3**

**Rationale**:
- Further speed up quick search screening
- 3 epochs sufficient to identify promising configurations

---

#### 2. Stage 2 Selection Criteria Reduction

**Modified Files**:
- [scripts/run_experiments.py](../scripts/run_experiments.py)

**Changes**:
- Per-parameter top models: 10 → **5**
- Overall ROC-AUC top models: 30 → **10**
- Disturbing ROC-AUC top models: 20 → **5**
- Expected Stage 2 models: ~150 → **~50-70** (after deduplication)

**Rationale**:
- Faster full training while maintaining diverse coverage
- Still covers all parameter values with representative models

---

#### 3. Stage 2 Model Count Display

**Changes**:
- Added print statement showing Stage 2 model count during experiment execution
- Format: `>>> Stage 2 will train {N} models (from {M} Stage 1 combinations) <<<`

---

## 2026-01-23 (Update 2): Two-Stage Dataset and Epoch Configuration

### Changes

#### 1. Separate Datasets for Quick/Full Search

**Modified Files**:
- [scripts/run_experiments.py](../scripts/run_experiments.py)

**Changes**:
- Stage 1 (Quick Search): 200,000 timesteps, train_ratio=0.3 → ~6,000 train, ~14,000 test
- Stage 2 (Full Search): 2,200,000 timesteps, train_ratio=0.5 → ~110,000 train, ~110,000 test
- Test set always uses target_counts 1200:300:500 (total 2,000)

**Rationale**:
- Quick search needs fast iteration (small train set)
- Test set composition should be consistent across stages for fair comparison

---

#### 2. Epoch Count Reduction

**Changes**:
- Stage 1 epochs: 15 → **5**
- Stage 2 epochs: 100 → **30**

**Rationale**:
- Faster experimentation while maintaining reasonable training quality
- Quick search only needs to identify promising configurations

---

## 2026-01-23: Dataset Migration and Hyperparameter Grid Cleanup

### Major Changes

#### 1. Dataset Migration to SlidingWindowDataset

**Modified Files**:
- [mae_anomaly/dataset.py](../mae_anomaly/dataset.py) → Deprecated
- [mae_anomaly/dataset_sliding.py](../mae_anomaly/dataset_sliding.py) → Primary dataset
- [scripts/run_experiments.py](../scripts/run_experiments.py)
- [scripts/visualize_all.py](../scripts/visualize_all.py)

**Changes**:
- Replaced `MultivariateTimeSeriesDataset` with `SlidingWindowTimeSeriesGenerator` and `SlidingWindowDataset`
- New dataset features:
  - Continuous sliding window extraction from long time series
  - 8 correlated server metrics (CPU, Memory, DiskIO, Network, ResponseTime, ThreadCount, ErrorRate, QueueLength)
  - 6 realistic anomaly types: spike, memory_leak, cpu_saturation, network_congestion, cascading_failure, resource_contention
  - Three sample types: pure_normal, disturbing_normal, anomaly
  - Train/test split by time (no data leakage)

---

#### 2. Fixed Hyperparameters (margin, lambda_disc)

**Modified Files**:
- [scripts/run_experiments.py](../scripts/run_experiments.py)
- [scripts/visualize_all.py](../scripts/visualize_all.py)
- [mae_anomaly/config.py](../mae_anomaly/config.py)

**Changes**:
- `margin` and `lambda_disc` are now fixed at 0.5 (not in hyperparameter grid)
- Reduced hyperparameter search space from 2592 to 288 combinations
- Grid now includes: `masking_ratio`, `masking_strategy`, `num_patches`, `margin_type`, `force_mask_anomaly`, `patch_level_loss`, `patchify_mode`

**Rationale**:
- Preliminary experiments showed margin=0.5 and lambda_disc=0.5 perform well across configurations
- Reducing search space allows more thorough exploration of other hyperparameters

---

#### 3. Stage 2 Selection Criteria Update

**Modified Files**:
- [scripts/run_experiments.py](../scripts/run_experiments.py)

**Changes**:
- New selection criteria for Stage 2 (150 diverse candidates):
  - Per-parameter top 10 (e.g., best for each masking_ratio value)
  - Overall top 30 by ROC-AUC
  - Top 20 by disturbing normal ROC-AUC
- Added `masking_strategy` to selection criteria

---

#### 4. num_features Updated (5 → 8)

**Modified Files**:
- [mae_anomaly/config.py](../mae_anomaly/config.py)
- [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md)
- [docs/DATASET.md](../docs/DATASET.md)

**Changes**:
- Default `num_features` changed from 5 to 8
- All documentation diagrams updated to reflect (batch, 100, 8) dimensions

---

#### 5. Visualization Bug Fixes

**Modified Files**:
- [scripts/visualize_all.py](../scripts/visualize_all.py)

**Fixes**:
- Updated `param_keys` to remove `margin` and `lambda_disc`
- Added `ANOMALY_TYPE_NAMES` import for visualization
- Fixed `plot_loss_by_anomaly_type` subplot grid (2x3 → dynamic for 7 anomaly types)
- Updated multiple places where margin/lambda_disc were referenced

---

### Documentation Updates

- [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md): Updated num_features (5→8), all dimension examples
- [docs/VISUALIZATIONS.md](../docs/VISUALIZATIONS.md): Updated param_keys, CSV columns, removed margin/lambda_disc
- [docs/DATASET.md](../docs/DATASET.md): Complete documentation for SlidingWindowDataset
- [docs/CHANGELOG.md](../docs/CHANGELOG.md): This entry

---

## 2026-01-22 (Update 3): Qualitative Case Studies and Late Bloomer Fix

### Major Changes

#### 1. Late Bloomer Algorithm Fix

**Modified Files**:
- [scripts/visualize_all.py](../scripts/visualize_all.py)

**Issue Found**:
- Late bloomer analysis used final epoch's threshold for all epochs
- At epoch 0, the model hasn't learned, so all scores are similar
- Using the final threshold at epoch 0 produces incorrect predictions

**Fixes**:
- Implemented per-epoch optimal threshold calculation
- Late bloomers now correctly identified as samples that changed from incorrect to correct classification
- Added two categories:
  - **Late Bloomer Anomalies (FN→TP)**: Missed at start, detected at end
  - **Late Bloomer Normals (FP→TN)**: False alarm at start, correct at end

---

#### 2. Reconstruction Evolution Enhancement

**Modified Files**:
- [scripts/visualize_all.py](../scripts/visualize_all.py)

**Changes**:
- Added Student reconstruction alongside Teacher (was Teacher-only)
- Added discrepancy visualization (|Teacher - Student|)
- Shows both reconstruction and discrepancy evolution over epochs
- Key insight: Discrepancy should increase in masked anomaly regions as training progresses

---

#### 3. Qualitative Case Study Visualizations

**New Files** (in `best_model/`):
- `case_study_gallery.png`: Representative TP/TN/FP/FN examples with detailed analysis
- `anomaly_type_case_studies.png`: Per-anomaly-type TP vs FN comparison
- `feature_contribution_analysis.png`: Which features drive anomaly detection
- `hardest_samples.png`: Analysis of hardest-to-detect samples (lowest margin FN/FP)

**New Files** (in `training_progress/`):
- `late_bloomer_case_studies.png`: Detailed time series evolution for late bloomers

**New Methods**:
- `BestModelVisualizer.plot_case_study_gallery()`: Median examples for each outcome
- `BestModelVisualizer.plot_anomaly_type_case_studies()`: Per-type TP/FN comparison
- `BestModelVisualizer.plot_feature_contribution_analysis()`: Feature importance ranking
- `BestModelVisualizer.plot_hardest_samples()`: Hardest FN and FP analysis
- `TrainingProgressVisualizer.plot_late_bloomer_case_studies()`: Detailed late bloomer evolution

---

### Documentation Updates

- [docs/VISUALIZATIONS.md](../docs/VISUALIZATIONS.md): Added new visualizations and updated descriptions
- [docs/CHANGELOG.md](../docs/CHANGELOG.md): This changelog entry

---

## 2026-01-22 (Update 2): Visualization Enhancements and Consistency Fixes

### Major Changes

#### 1. Visualization Data Consistency Fix

**Modified Files**:
- [scripts/visualize_all.py](../scripts/visualize_all.py)

**Issue Found**:
- `visualize_all.py` used different evaluation settings than `run_experiments.py`:
  - `anomaly_ratio=0.3` instead of `config.test_anomaly_ratio=0.25`
  - Random masking instead of fixed last-patch masking
  - MAE (absolute error) instead of MSE (squared error)

**Fixes**:
- Changed `anomaly_ratio` to use `config.test_anomaly_ratio` (0.25)
- Changed `collect_predictions()` and `collect_detailed_data()` to use same evaluation as `evaluator.py`:
  - Fixed mask: `mask[:, -config.mask_last_n:] = 0`
  - Forward with `masking_ratio=0.0` and explicit mask
  - MSE computation: `((output - input) ** 2).mean(dim=2)`

---

#### 2. New Data Visualizations

**New Files**:
- `data/anomaly_generation_rules.png`: Detailed rules for each anomaly type
- `data/feature_correlations.png`: Feature correlation matrix and generation rules
- `data/experiment_settings.png`: Experiment settings summary (Stage 1/2)

**Changes**:
- Added `plot_anomaly_generation_rules()`: Shows how each anomaly type is generated
- Added `plot_feature_correlations()`: Shows inter-feature correlations
- Added `plot_experiment_settings()`: Summarizes experiment settings for reproducibility

---

#### 3. Stage 2 Per-Hyperparameter Visualizations

**New Files** (in `stage2/`):
- `hyperparam_masking_ratio.png`
- `hyperparam_num_patches.png`
- `hyperparam_margin.png`
- `hyperparam_lambda_disc.png`
- `hyperparam_margin_type.png`
- `hyperparam_force_mask_anomaly.png`
- `hyperparam_patch_level_loss.png`
- `hyperparam_patchify_mode.png`
- `hyperparameter_interactions.png`
- `best_config_summary.png`

**Changes**:
- Added `plot_hyperparameter_impact()`: Per-hyperparameter detailed analysis
- Added `plot_all_hyperparameters()`: Generate all per-hyperparameter plots
- Added `plot_hyperparameter_interactions()`: Interaction heatmaps
- Added `plot_best_config_summary()`: Best config with Korean descriptions

---

#### 4. Best Model Analysis Improvements

**New Files**:
- `best_model/pure_vs_disturbing_normal.png`: Pure Normal vs Disturbing Normal comparison
- `best_model/discrepancy_trend.png`: Discrepancy trend analysis across time steps

**Changes**:
- Added `plot_pure_vs_disturbing_normal()`: Detailed comparison of sample types
- Added `plot_discrepancy_trend()`: Time-step level discrepancy analysis

---

### Documentation Updates

- [docs/VISUALIZATIONS.md](../docs/VISUALIZATIONS.md): Complete rewrite with all new visualizations

---

## 2026-01-22: Project Cleanup and Patchify Mode

### Major Changes

#### 1. Patchify Mode Feature

**Modified Files**:
- [mae_anomaly/model.py](../mae_anomaly/model.py)
- [mae_anomaly/config.py](../mae_anomaly/config.py)

**Changes**:
- Added `patchify_mode` configuration option with 2 modes:
  - `linear`: Direct patchify + linear projection (MAE original style)
  - `patch_cnn`: Patchify first, then CNN per patch (no cross-patch leakage)
- Updated model to support both patchify modes

**Benefits**:
- Flexibility to test different patchification strategies
- `patch_cnn` mode prevents information leakage across patches
- Better control over local feature extraction

---

#### 2. Visualization Refactoring

**Modified Files**:
- [scripts/visualize_all.py](../scripts/visualize_all.py) (NEW)
- [scripts/run_experiments.py](../scripts/run_experiments.py) (refactored)

**Changes**:
- Moved all visualization code from `run_experiments.py` to dedicated `visualize_all.py`
- Created 5 visualization classes:
  - `DataVisualizer`: Data distribution and sample visualization
  - `ArchitectureVisualizer`: Model architecture diagrams
  - `ExperimentVisualizer`: Stage 1 (Quick Search) results
  - `Stage2Visualizer`: Stage 2 (Full Training) results
  - `BestModelVisualizer`: Best model analysis
- `run_experiments.py` now only handles training and saves results to CSV
- Fixed shape mismatch bugs when data has fewer than 10 rows

---

#### 3. Project Cleanup

**Deleted Files**:
- `tests/integration/` (obsolete test files using old module)
- `REFACTORING_COMPLETE.md`, `REFACTORING_PLAN.md`
- `docs/bugfixes/`, `docs/implementation/`, `docs/analysis/`

**Updated Files**:
- [README.md](../README.md) - Complete rewrite for current structure
- [examples/basic_usage.py](../examples/basic_usage.py) - Updated imports and examples
- [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) - Added patchify_mode documentation
- [docs/ABLATION_STUDIES.md](../docs/ABLATION_STUDIES.md) - Added patchify_mode experiments

---

### Documentation Updates

- README.md now reflects current project structure
- Added patchify mode examples in basic_usage.py
- Architecture documentation includes all 3 patchify modes
- Ablation studies documentation includes patchify_mode experiments

---

## 2026-01-14: Architecture and Training Updates

### Major Changes

#### 1. Architecture: Transformer → 1D-CNN + Transformer Hybrid

**Modified Files**:
- [mae_anomaly/model.py](../mae_anomaly/model.py)

**Changes**:
- Added 2-layer 1D-CNN before Transformer:
  - Conv1: num_features (5) → d_model//2 (32), kernel=3
  - Conv2: d_model//2 (32) → d_model (64), kernel=3
  - BatchNorm + ReLU after each layer
- Updated patch embedding to work with CNN features:
  - New method: `patchify_cnn()` for CNN output
  - Processes (batch, d_model, seq_length) → (batch, num_patches, d_model*patch_size)
- Updated forward pass:
  - Input → CNN → Patchify → Transformer
  - CNN adds ~6,912 parameters
  - Total parameters: ~513K (was ~505K)

**Benefits**:
- Better local feature extraction
- Combines CNN (local) + Transformer (global) strengths
- Improved representation learning

---

#### 2. Best Model Selection: Match Evaluation Criterion

**Modified Files**:
- [scripts/run_experiments.py](../scripts/run_experiments.py)

**Changes**:
- Best model selection now matches evaluation metric:
  - **Baseline**: Uses total loss (reconstruction + discrepancy)
  - **TeacherOnly**: Uses teacher reconstruction loss
  - **StudentOnly**: Uses student reconstruction loss
- Model selection based on ROC-AUC during grid search

**Rationale**:
- Previous: All experiments used reconstruction loss for model selection
- Issue: Baseline evaluation uses discrepancy, but model selected on reconstruction
- Fix: Model selection criterion now matches what we optimize for in each ablation

---

### Documentation Updates

#### Created Files:
1. [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md)
   - Complete architecture documentation
   - Component-by-component breakdown
   - Parameter counts and pipeline diagram
   - Design rationale and comparisons

#### Updated Files:
1. [docs/ABLATION_STUDIES.md](../docs/ABLATION_STUDIES.md)
   - Added architecture overview section
   - Updated best model selection notes
   - Clarified evaluation criteria for each ablation

---

---

## Previous Updates (2026-01-14)

### Data and Ablation Updates

1. **Data Size Increase** (5x):
   - Train: 1,000 → 5,000 samples
   - Test: 300 → 1,500 samples

2. **Best Model Checkpointing**:
   - Track training loss during epochs
   - Save model at lowest loss epoch
   - Restore best model after training

3. **Masking Strategy Ablation**:
   - Added patch masking (same-time across features)
   - Added feature-wise masking (independent per feature)
   - Tests importance of cross-feature temporal coherence

4. **Removed Redundant Ablations**:
   - Removed NoDiscrepancy (redundant with TeacherOnly)
   - Removed NoMasking (replaced with more informative experiments)

5. **Cleanup**:
   - Deleted old experiment results
   - Removed unused folders
   - Regenerated visualizations

---

## File Structure

```
mae_anomaly/
├── model.py            [MODIFIED] - Added 1D-CNN layers, patchify modes
├── config.py           [MODIFIED] - Added patchify_mode, margin_type options
├── loss.py             - Self-distillation loss
├── trainer.py          - Training loop
├── evaluator.py        - Evaluation metrics
└── dataset.py          - Synthetic dataset generation

scripts/
├── run_experiments.py  - Two-stage grid search experiment runner
└── visualize_all.py    - Comprehensive visualization generator

docs/
├── ARCHITECTURE.md     - Architecture documentation
├── ABLATION_STUDIES.md - Ablation study documentation
├── VISUALIZATIONS.md   - Visualization guide
└── CHANGELOG.md        - This file
```

---

## Summary

**Total Changes**:
1. ✅ Transformer → 1D-CNN + Transformer hybrid architecture
2. ✅ Best model selection matches evaluation criterion
3. ✅ Comprehensive architecture documentation
4. ✅ Testing and verification scripts

**Status**: All changes implemented, tested, and documented.

**Next Steps**: Run full experiments with updated architecture and training logic.
