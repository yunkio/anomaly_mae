# Visualization Documentation

**Last Updated**: 2026-01-22
**Status**: Complete

---

## Overview

This project generates comprehensive visualizations through a unified script (`visualize_all.py`). Visualizations are organized into five main categories:

1. **Data Visualizations** - Dataset characteristics and anomaly types
2. **Architecture Visualizations** - Model pipeline, patchify modes, self-distillation concept
3. **Stage 1 Visualizations** - Quick search (hyperparameter screening) results
4. **Stage 2 Visualizations** - Full training results on selected candidates
5. **Best Model Visualizations** - Detailed analysis of the best performing model

---

## Scripts

### 1. run_experiments.py (Experiment Runner)

Runs experiments and saves results. **Does NOT generate visualizations.**

```bash
# Run full experiment pipeline
python scripts/run_experiments.py

# Custom parameters
python scripts/run_experiments.py --quick-epochs 10 --full-epochs 50 --top-k 100

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
├── stage2/         # Stage 2 results visualization
└── best_model/     # Best model analysis
```

---

## Visualization Categories

### 1. Data Visualizations (`data/`)

Understanding the dataset and anomaly types.

| File | Description |
|------|-------------|
| `anomaly_types.png` | Examples of different anomaly types (point, contextual, collective, frequency, trend) |
| `sample_types.png` | Comparison of pure normal, disturbing normal, and anomaly samples |
| `feature_examples.png` | Multivariate feature visualization for normal and anomaly samples |
| `dataset_statistics.png` | Label and sample type distributions |

### 2. Architecture Visualizations (`architecture/`)

Understanding the model architecture and concepts.

| File | Description |
|------|-------------|
| `model_pipeline.png` | Overall Self-Distilled MAE pipeline diagram |
| `patchify_modes.png` | Comparison of cnn_first, patch_cnn, and linear modes |
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

Full training results on selected diverse candidates.

| File | Description |
|------|-------------|
| `stage2_quick_vs_full.png` | Quick search vs full training performance comparison |
| `stage2_selection_criterion.png` | Analysis of performance by selection criterion |
| `learning_curves.png` | Training loss curves for top experiments |
| `stage2_summary_dashboard.png` | Comprehensive Stage 2 summary dashboard |

### 5. Best Model Visualizations (`best_model/`)

Detailed analysis of the single best performing model.

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

---

## Visualization Classes

### DataVisualizer

```python
from scripts.visualize_all import DataVisualizer

data_vis = DataVisualizer(output_dir='output/data', config=config)
data_vis.plot_anomaly_types()
data_vis.plot_sample_types()
data_vis.plot_feature_examples()
data_vis.plot_dataset_statistics()
data_vis.generate_all()  # Generate all
```

### ArchitectureVisualizer

```python
from scripts.visualize_all import ArchitectureVisualizer

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
from scripts.visualize_all import ExperimentVisualizer
import pandas as pd

results_df = pd.read_csv('results/experiments/20260122/quick_search_results.csv')
param_keys = ['masking_ratio', 'num_patches', 'margin', 'lambda_disc',
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
from scripts.visualize_all import Stage2Visualizer
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
stage2_vis.generate_all()  # Generate all
```

### BestModelVisualizer

```python
from scripts.visualize_all import BestModelVisualizer, load_best_model

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
best_vis.generate_all()  # Generate all
```

---

## Stage 2 Selection Criteria

Stage 2 selects 150 diverse candidates for full training:

| Criterion | Count | Description |
|-----------|-------|-------------|
| overall_roc_auc | 30 | Top by overall ROC-AUC |
| disturbing_roc_auc | 20 | Top by disturbing normal ROC-AUC |
| force_mask_anomaly=True | 10 | Best with forced anomaly masking |
| patch_level_loss=True | 10 | Best with patch-level loss |
| margin_type=dynamic | 10 | Best with dynamic margin |
| margin_type=softplus | 10 | Best with softplus margin |
| margin_type=hinge | 10 | Best with hinge margin |
| patchify_mode=cnn_first | 10 | Best with CNN-first patchify |
| patchify_mode=patch_cnn | 10 | Best with patch-CNN patchify |
| patchify_mode=linear | 10 | Best with linear patchify (MAE style) |
| masking_strategy=patch | 10 | Best with patch masking strategy |
| masking_strategy=feature_wise | 10 | Best with feature-wise masking strategy |

---

## CSV Output Format

### quick_search_results.csv (Stage 1)

| Column | Description |
|--------|-------------|
| combination_id | Unique identifier for parameter combination |
| masking_ratio | Masking ratio (0.4, 0.7) |
| masking_strategy | Masking strategy (patch, feature_wise) |
| num_patches | Number of patches (10, 25, 50) |
| margin | Margin value (0.25, 0.5, 1.0) |
| lambda_disc | Discrepancy loss weight (0.3, 0.5, 0.7) |
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

**Status**: Documentation complete. All visualizations implemented and tested.
