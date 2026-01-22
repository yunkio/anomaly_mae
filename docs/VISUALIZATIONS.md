# Visualization Documentation

**Last Updated**: 2026-01-23
**Status**: Complete

---

## Overview

This project generates comprehensive visualizations through a modular visualization package (`mae_anomaly.visualization`) with a unified entry script (`scripts/visualize_all.py`). Visualizations are organized into six main categories:

1. **Data Visualizations** - Dataset characteristics, anomaly generation rules, feature correlations
2. **Architecture Visualizations** - Model pipeline, patchify modes, self-distillation concept
3. **Stage 1 Visualizations** - Quick search (hyperparameter screening) results
4. **Stage 2 Visualizations** - Full training results with per-hyperparameter analysis
5. **Best Model Visualizations** - Detailed analysis including pure/disturbing normal comparison
6. **Training Progress Visualizations** - Model learning evolution over training epochs

---

## Module Structure

The visualization code is organized in `mae_anomaly/visualization/`:

```
mae_anomaly/visualization/
├── __init__.py                  # Module exports
├── base.py                      # Common utilities and color functions
├── data_visualizer.py           # DataVisualizer class
├── architecture_visualizer.py   # ArchitectureVisualizer class
├── experiment_visualizer.py     # ExperimentVisualizer (Stage 1)
├── stage2_visualizer.py         # Stage2Visualizer class
├── best_model_visualizer.py     # BestModelVisualizer class
└── training_visualizer.py       # TrainingProgressVisualizer class
```

### Dynamic Color Management

All visualizers use dynamic color functions from `base.py`:

```python
from mae_anomaly.visualization import (
    get_anomaly_colors,       # Returns dict mapping anomaly types to colors
    get_feature_colors,       # Returns dict mapping feature names to colors
    SAMPLE_TYPE_COLORS,       # {0: '#3498DB', 1: '#F39C12', 2: '#E74C3C'}
    SAMPLE_TYPE_NAMES,        # {0: 'Pure Normal', 1: 'Disturbing Normal', 2: 'Anomaly'}
)
```

This ensures consistency when anomaly types or features change.

---

## Scripts

### 1. run_experiments.py (Experiment Runner)

Runs experiments and saves results. **Does NOT generate visualizations.**

```bash
# Run full experiment pipeline
python scripts/run_experiments.py

# Custom parameters
python scripts/run_experiments.py --quick-epochs 10 --full-epochs 50

# Output
results/experiments/YYYYMMDD_HHMMSS/
├── quick_search_results.csv
├── full_search_results.csv
├── best_model.pt
├── best_config.json
├── training_histories.json
└── experiment_metadata.json
```

### 2. visualize_all.py (Unified Visualization Script)

Generates ALL visualizations from saved experiment results.

```bash
# Auto-find latest experiment
python scripts/visualize_all.py

# Specify experiment directory
python scripts/visualize_all.py --experiment-dir results/experiments/20260122_120000

# Skip certain visualizations
python scripts/visualize_all.py --skip-data --skip-architecture

# Output
results/experiments/YYYYMMDD_HHMMSS/visualization/
├── data/           # DataVisualizer outputs
├── architecture/   # ArchitectureVisualizer outputs
├── stage1/         # Stage 1 results visualization
├── stage2/         # Stage 2 results visualization (incl. per-hyperparameter)
└── best_model/     # Best model analysis
```

---

## Experiment Settings Consistency

Visualizations use the **same evaluation method** as `run_experiments.py`:

| Setting | Stage 1 | Stage 2 | Visualization |
|---------|---------|---------|---------------|
| Time Series Length | 200,000 | 440,000 | N/A |
| Train Ratio | 0.3 | 0.5 | N/A |
| Train Samples | ~6,000 | ~22,000 | N/A |
| Test Samples | 2,000 | 2,000 | 2,000 |
| Test Target | 1200:300:500 | 1200:300:500 | 1200:300:500 |
| Epochs | 1 | 2 | N/A (uses saved model) |
| Masking | Last `mask_last_n` | Last `mask_last_n` | Last `mask_last_n` |
| Error Metric | MSE | MSE | MSE |

---

## Visualization Categories

### 1. Data Visualizations (`data/`)

Understanding the dataset, anomaly types, and experiment settings.

| File | Description |
|------|-------------|
| `sample_types.png` | Comparison of pure normal, disturbing normal, and anomaly samples (diverse sampling for variety) |
| `feature_examples.png` | All 8 features with actual FEATURE_NAMES (CPU, Memory, DiskIO, etc.) for normal and anomaly samples |
| `dataset_statistics.png` | Label and sample type distributions |
| `anomaly_generation_rules.png` | Detailed rules for each anomaly type with actual dataset examples |
| `feature_correlations.png` | Feature correlation matrix and generation rules explanation |
| `experiment_settings.png` | Experiment settings summary (Stage 1/2 epochs, data counts, seeds) |

**Note**: `anomaly_types.png` was removed (redundant with `anomaly_generation_rules.png`).

### 2. Architecture Visualizations (`architecture/`)

Understanding the model architecture and concepts.

| File | Description |
|------|-------------|
| `model_pipeline.png` | Overall Self-Distilled MAE pipeline diagram |
| `patchify_modes.png` | Conceptual flow diagrams showing processing pipeline differences (CNN-First, Patch-CNN, Linear) |
| `masking_visualization.png` | Step-by-step visualization of the masking process |
| `self_distillation_concept.png` | Teacher vs Student behavior on normal/anomaly data |
| `margin_types.png` | Hinge, softplus, and dynamic margin loss functions |
| `loss_components.png` | Reconstruction loss and discrepancy loss explanation |

### 3. Stage 1 Visualizations (`stage1/`)

Quick search results analysis (hyperparameter screening).

| File | Description |
|------|-------------|
| `heatmaps_roc_auc.png` | Parameter pair heatmaps showing mean ROC-AUC |
| `parallel_coordinates.png` | Parallel coordinates plot for all combinations |
| `parameter_importance.png` | Box plots showing each parameter's impact on ROC-AUC |
| `top_k_comparison.png` | Bar chart and table comparing top 10 configurations |
| `metric_distributions.png` | Histograms of ROC-AUC, F1, Precision, Recall |
| `metric_correlations.png` | Correlation matrix between all metrics |
| `patchify_mode_comparison_roc_auc.png` | Detailed comparison of patchify modes |
| `margin_type_comparison_roc_auc.png` | Detailed comparison of margin types |
| `stage1_summary_dashboard.png` | Comprehensive dashboard with all key insights |

### 4. Stage 2 Visualizations (`stage2/`)

Full training results on selected diverse candidates, **with per-hyperparameter analysis**.

| File | Description |
|------|-------------|
| `stage2_quick_vs_full.png` | Quick search vs full training performance comparison |
| `stage2_selection_criterion.png` | Analysis of performance by selection criterion |
| `learning_curves.png` | Training loss curves for top experiments |
| `stage2_summary_dashboard.png` | Comprehensive Stage 2 summary dashboard |
| `hyperparam_masking_ratio.png` | masking_ratio impact on Stage 2 ROC-AUC |
| `hyperparam_masking_strategy.png` | masking_strategy impact on Stage 2 ROC-AUC |
| `hyperparam_num_patches.png` | num_patches impact on Stage 2 ROC-AUC |
| `hyperparam_margin_type.png` | margin_type impact on Stage 2 ROC-AUC |
| `hyperparam_force_mask_anomaly.png` | force_mask_anomaly impact on Stage 2 ROC-AUC |
| `hyperparam_patch_level_loss.png` | patch_level_loss impact on Stage 2 ROC-AUC |
| `hyperparam_patchify_mode.png` | patchify_mode impact on Stage 2 ROC-AUC |
| `hyperparameter_interactions.png` | **NEW**: Heatmaps showing hyperparameter interactions |
| `best_config_summary.png` | **NEW**: Best model configuration summary with Korean descriptions |

### 5. Best Model Visualizations (`best_model/`)

Detailed analysis of the single best performing model, including qualitative case studies.

| File | Description |
|------|-------------|
| `best_model_roc_curve.png` | ROC curve with AUC and optimal threshold |
| `best_model_score_distribution.png` | Anomaly score distributions by label |
| `best_model_confusion_matrix.png` | Confusion matrix with accuracy metrics |
| `best_model_score_components.png` | Reconstruction error vs discrepancy scatter |
| `best_model_teacher_student_comparison.png` | Teacher vs Student error analysis |
| `best_model_reconstruction.png` | Original vs Teacher vs Student reconstruction |
| `best_model_detection_examples.png` | TP, TN, FP, FN example time series |
| `best_model_summary.png` | Summary statistics and configuration |
| `pure_vs_disturbing_normal.png` | Detailed comparison of pure normal vs disturbing normal |
| `discrepancy_trend.png` | Discrepancy trend with std bands, zoomed last-patch view, and box plots by sample type |
| `hypothesis_verification.png` | **NEW**: Verification of hypotheses about disturbing normal performance |
| `case_study_gallery.png` | Representative TP/TN/FP/FN case studies with detailed analysis |
| `anomaly_type_case_studies.png` | **NEW**: Per-anomaly-type case studies (TP vs FN) |
| `feature_contribution_analysis.png` | **NEW**: Which features contribute most to detection |
| `hardest_samples.png` | **NEW**: Analysis of hardest-to-detect samples |
| `loss_by_anomaly_type.png` | Loss distributions by anomaly type |
| `performance_by_anomaly_type.png` | Detection rate and mean score by anomaly type |
| `loss_scatter_by_anomaly_type.png` | Recon vs discrepancy scatter colored by anomaly type |
| `sample_type_analysis.png` | Sample type (pure/disturbing/anomaly) analysis |

### 6. Training Progress Visualizations (`training_progress/`)

Visualizations showing how the model learns over training epochs (requires `--retrain` flag).

| File | Description |
|------|-------------|
| `score_evolution.png` | Score distribution changes at checkpoint epochs |
| `sample_trajectories.png` | Individual sample score trajectories over training |
| `metrics_evolution.png` | ROC-AUC, F1, Precision, Recall over epochs |
| `late_bloomer_analysis.png` | **UPDATED**: Samples that changed classification (per-epoch thresholds) |
| `late_bloomer_case_studies.png` | **NEW**: Detailed case studies of late bloomer samples |
| `anomaly_type_learning.png` | Detection rate by anomaly type over epochs |
| `reconstruction_evolution.png` | **UPDATED**: Teacher & Student reconstruction + discrepancy evolution |
| `decision_boundary_evolution.png` | Threshold and score separation over epochs |

---

## Visualization Classes

All visualizers are now imported from the `mae_anomaly.visualization` module:

```python
from mae_anomaly.visualization import (
    # Base utilities
    setup_style,
    find_latest_experiment,
    load_experiment_data,
    load_best_model,
    collect_predictions,
    collect_detailed_data,
    get_anomaly_colors,
    get_feature_colors,
    SAMPLE_TYPE_NAMES,
    SAMPLE_TYPE_COLORS,
    # Visualizers
    DataVisualizer,
    ArchitectureVisualizer,
    ExperimentVisualizer,
    Stage2Visualizer,
    BestModelVisualizer,
    TrainingProgressVisualizer,
)
```

### DataVisualizer

```python
from mae_anomaly.visualization import DataVisualizer

data_vis = DataVisualizer(output_dir='output/data', config=config)
data_vis.plot_sample_types()           # Diverse sampling for variety
data_vis.plot_feature_examples()       # All 8 features with FEATURE_NAMES
data_vis.plot_dataset_statistics()
data_vis.plot_anomaly_generation_rules()  # Dynamic: uses ANOMALY_TYPE_NAMES
data_vis.plot_feature_correlations()
data_vis.plot_experiment_settings()
data_vis.generate_all()  # Generate all (excludes redundant anomaly_types)
```

### ArchitectureVisualizer

```python
from mae_anomaly.visualization import ArchitectureVisualizer

arch_vis = ArchitectureVisualizer(output_dir='output/architecture', config=config)
arch_vis.plot_model_pipeline()
arch_vis.plot_patchify_modes()
arch_vis.plot_masking_visualization()
arch_vis.plot_self_distillation_concept()
arch_vis.plot_margin_types()
arch_vis.plot_loss_components()
arch_vis.generate_all()  # Generate all
```

### ExperimentVisualizer (Stage 1)

```python
from mae_anomaly.visualization import ExperimentVisualizer
import pandas as pd

results_df = pd.read_csv('results/experiments/20260122/quick_search_results.csv')
param_keys = ['masking_ratio', 'masking_strategy', 'num_patches',
              'margin_type', 'force_mask_anomaly', 'patch_level_loss', 'patchify_mode']

stage1_vis = ExperimentVisualizer(results_df, param_keys, output_dir='output/stage1')
stage1_vis.plot_heatmaps()
stage1_vis.plot_parallel_coordinates()
stage1_vis.plot_parameter_importance()
stage1_vis.plot_top_k_comparison(k=10)
stage1_vis.plot_metric_distributions()
stage1_vis.plot_metric_correlations()
stage1_vis.plot_categorical_comparison('patchify_mode')
stage1_vis.plot_summary_dashboard()
stage1_vis.generate_all()  # Generate all
```

### Stage2Visualizer

```python
from mae_anomaly.visualization import Stage2Visualizer
import pandas as pd
import json

full_df = pd.read_csv('results/experiments/20260122/full_search_results.csv')
quick_df = pd.read_csv('results/experiments/20260122/quick_search_results.csv')
with open('results/experiments/20260122/training_histories.json') as f:
    histories = json.load(f)

stage2_vis = Stage2Visualizer(full_df, quick_df, histories, output_dir='output/stage2')
stage2_vis.plot_quick_vs_full()
stage2_vis.plot_selection_criterion_analysis()
stage2_vis.plot_learning_curves(top_k=10)
stage2_vis.plot_summary_dashboard()
stage2_vis.plot_all_hyperparameters()
stage2_vis.plot_hyperparameter_interactions()
stage2_vis.plot_best_config_summary()
stage2_vis.generate_all()  # Generate all
```

### BestModelVisualizer

```python
from mae_anomaly.visualization import BestModelVisualizer, load_best_model

model, config, test_loader, metrics = load_best_model('results/experiments/20260122/best_model.pt')

best_vis = BestModelVisualizer(model, config, test_loader, output_dir='output/best_model')
best_vis.plot_roc_curve()
best_vis.plot_score_distribution()
best_vis.plot_confusion_matrix()
best_vis.plot_score_components()
best_vis.plot_teacher_student_comparison()
best_vis.plot_reconstruction_examples(num_examples=3)
best_vis.plot_detection_examples()
best_vis.plot_summary_statistics()
best_vis.plot_pure_vs_disturbing_normal()
best_vis.plot_discrepancy_trend()
best_vis.plot_hypothesis_verification()
# Qualitative case studies
best_vis.plot_case_study_gallery()
best_vis.plot_anomaly_type_case_studies()
best_vis.plot_feature_contribution_analysis()
best_vis.plot_hardest_samples()
best_vis.generate_all()  # Generate all
```

### TrainingProgressVisualizer

```python
from mae_anomaly.visualization import TrainingProgressVisualizer
import json

with open('results/experiments/20260122/best_config.json') as f:
    best_config = json.load(f)

progress_vis = TrainingProgressVisualizer(best_config, output_dir='output/training_progress')
progress_vis.generate_all()  # Re-trains model and generates all plots

# Individual plots (after retrain_with_checkpoints())
progress_vis.plot_score_evolution()
progress_vis.plot_sample_trajectories()
progress_vis.plot_metrics_evolution()
progress_vis.plot_late_bloomer_analysis()
progress_vis.plot_late_bloomer_case_studies()
progress_vis.plot_anomaly_type_learning()
progress_vis.plot_reconstruction_evolution()
progress_vis.plot_decision_boundary_evolution()
```

---

## Stage 2 Selection Criteria

Stage 2 selects ~50-70 diverse candidates for full training using a 3-phase approach:

**Phase 1**: Per-parameter top 5 (ensures coverage of all parameter values)
**Phase 2**: Top 10 by overall ROC-AUC (excluding Phase 1 selections)
**Phase 3**: Top 5 by disturbing ROC-AUC (excluding Phase 1, 2 selections)

| Criterion | Count | Description |
|-----------|-------|-------------|
| overall_roc_auc | 10 | Top by overall ROC-AUC (Phase 1 제외) |
| disturbing_roc_auc | 5 | Top by disturbing normal ROC-AUC (Phase 1, 2 제외) |
| force_mask_anomaly=True | 5 | Best with forced anomaly masking |
| force_mask_anomaly=False | 5 | Best without forced anomaly masking |
| patch_level_loss=True | 5 | Best with patch-level loss |
| patch_level_loss=False | 5 | Best without patch-level loss |
| margin_type=dynamic | 5 | Best with dynamic margin |
| margin_type=softplus | 5 | Best with softplus margin |
| margin_type=hinge | 5 | Best with hinge margin |
| patchify_mode=cnn_first | 5 | Best with CNN-first patchify |
| patchify_mode=patch_cnn | 5 | Best with patch-CNN patchify |
| patchify_mode=linear | 5 | Best with linear patchify (MAE style) |
| masking_strategy=patch | 5 | Best with patch masking strategy |
| masking_strategy=feature_wise | 5 | Best with feature-wise masking strategy |
| masking_ratio (각 값) | 5 | Best for each masking ratio value |
| num_patches (각 값) | 5 | Best for each num_patches value |

---

## CSV Output Format

### quick_search_results.csv (Stage 1)

| Column | Description |
|--------|-------------|
| combination_id | Unique identifier for parameter combination |
| masking_ratio | Masking ratio (0.4, 0.7) |
| masking_strategy | Masking strategy (patch, feature_wise) |
| num_patches | Number of patches (10, 25, 50) |
| margin_type | Margin type (hinge, softplus, dynamic) |
| force_mask_anomaly | Whether to force mask anomaly patches (True/False) |
| patch_level_loss | Whether to use patch-level loss (True/False) |
| patchify_mode | Patchify mode (cnn_first, patch_cnn, linear) |
| roc_auc | ROC-AUC score |
| f1_score | F1 score |
| precision | Precision |
| recall | Recall |
| disturbing_roc_auc | ROC-AUC on disturbing normal samples |
| disturbing_f1 | F1 on disturbing normal samples |

**Note**: `margin` and `lambda_disc` are fixed at 0.5 and not included in hyperparameter search.

### full_search_results.csv (Stage 2)

Additional columns:

| Column | Description |
|--------|-------------|
| stage2_rank | Rank in Stage 2 training order |
| selection_criterion | Why this model was selected |
| quick_roc_auc | ROC-AUC from quick search |
| quick_f1 | F1 from quick search |
| roc_auc_improvement | Improvement from quick to full training |

---

## Color Scheme

Consistent colors across all visualizations:

| Element | Color |
|---------|-------|
| Normal samples | Blue (#3498DB) |
| Disturbing normal | Orange (#F39C12) |
| Anomaly samples | Red (#E74C3C) |
| Teacher model | Green |
| Student model | Red/Orange |
| Masked region | Yellow (alpha=0.2) |
| Threshold line | Green (dashed) |

### Patchify Mode Colors

| Mode | Color |
|------|-------|
| cnn_first | Steel Blue |
| patch_cnn | Coral |
| linear | Forest Green |

---

## Workflow

```
1. Run experiments (no visualization):
   python scripts/run_experiments.py

2. Generate all visualizations:
   python scripts/visualize_all.py

3. Or generate specific visualizations:
   python scripts/visualize_all.py --skip-data --skip-architecture
```

---

**Status**: Documentation complete. Visualization module modularized (Update 9). All visualizers use dynamic color management.
