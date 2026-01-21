# Ablation Studies Documentation

**Last Updated**: 2026-01-22
**Status**: ✅ Verified and Logically Correct

---

## Overview

This document describes the ablation studies used to validate the Self-Distilled MAE architecture for time series anomaly detection.

## Architecture

**Model**: 1D-CNN + Transformer hybrid architecture
- **1D-CNN**: Two convolutional layers for local feature extraction
  - Conv1: num_features → d_model//2 (kernel_size=3, padding=1)
  - Conv2: d_model//2 → d_model (kernel_size=3, padding=1)
- **Transformer**: Encoder-decoder architecture with self-distillation
  - Encoder: 3 layers, 4 attention heads
  - Teacher Decoder: 4 layers
  - Student Decoder: 1 layer

**Pipeline**:
1. Input (batch, 100, 5) → 1D-CNN → (batch, 64, 100)
2. CNN features → Patch embedding → (batch, 25, 64)
3. Patches + Positional encoding → Transformer encoder
4. Latent → Teacher/Student decoders → Output projection
5. Reconstruction (batch, 100, 5)

---

## Ablation Configurations

### 1. Baseline (Full Model)

**Configuration**:
- `use_masking`: ✓ True
- `use_discrepancy_loss`: ✓ True
- `use_teacher`: ✓ True
- `use_student`: ✓ True

**Training**:
- Loss: Reconstruction + Discrepancy
- Random patch masking during training (40% of patches masked)
- Both teacher and student models trained

**Evaluation**:
- Metric: Teacher-Student Discrepancy
- Mask last patch and compute difference between teacher and student outputs

**Purpose**:
Full model with all components. This is the proposed architecture.

---

### 2. Patch Size Experiments

**Configuration**:
Different numbers of patches tested with baseline configuration:
- 10 patches (patch size = 10)
- 25 patches (patch size = 4) - **Default**
- 50 patches (patch size = 2)

**Purpose**:
Test the impact of patch granularity on anomaly detection performance.
- Fewer patches: Larger context per patch, coarser masking
- More patches: Finer granularity, more precise localization

**Hypothesis**:
There exists an optimal patch size that balances:
- Sufficient context for reconstruction
- Fine enough granularity for anomaly localization
- Model capacity and computational efficiency

---

### 3. Masking Ratio Experiments

**Configuration**:
Different masking ratios tested:
- 40% masking
- 70% masking

**Purpose**:
Test the optimal amount of masking during training.
- Lower masking: Easier reconstruction, less regularization
- Higher masking: Harder reconstruction, more regularization

---

### 4. Margin Experiments

**Configuration**:
Different margin values tested with baseline configuration:
- margin = 0.25
- margin = 0.5 (default)
- margin = 1.0

**Purpose**:
Test the impact of margin value on discrepancy learning.
- Lower margin: Teacher-student gap can be smaller
- Higher margin: Forces larger separation on anomalies

---

### 5. Margin Type Experiments

**Configuration**:
Three different margin loss types:
- **hinge** (default): `ReLU(margin - discrepancy)`
- **softplus**: `Softplus(margin - discrepancy)` - smoother gradient
- **dynamic**: `ReLU(mu + k*sigma - discrepancy)` - adaptive margin

**Purpose**:
Test which margin formulation works best for self-distillation.

---

### 6. Lambda_disc Experiments

**Configuration**:
Different discrepancy loss weights tested with baseline configuration:
- λ_disc = 0.3
- λ_disc = 0.5 (default)
- λ_disc = 0.7

**Training**:
- All use full baseline configuration (teacher + student + discrepancy loss)
- Same architecture and hyperparameters
- Only difference: weight of discrepancy loss in total loss

**Loss Function**:
```python
total_loss = reconstruction_loss + λ_disc * discrepancy_loss
```

**Purpose**:
Test the optimal balance between reconstruction and discrepancy objectives.
- **Low λ_disc**: Prioritizes reconstruction accuracy
- **High λ_disc**: Prioritizes teacher-student separation on anomalies

**Hypothesis**:
- Too low λ_disc: Student learns to mimic teacher too well, reducing discrepancy signal
- Too high λ_disc: May compromise reconstruction quality
- Optimal λ_disc: Balances reconstruction and anomaly sensitivity

---

### 7. Force Mask Anomaly Experiments

**Configuration**:
- `force_mask_anomaly=False` (default)
- `force_mask_anomaly=True`

**Purpose**:
Test whether forcing anomaly patches to be masked during training improves detection.
- When True: Patches containing anomalies are always masked
- Ensures model learns to reconstruct normal patterns only

---

### 8. Patch-Level vs Window-Level Loss

**Configuration**:
- `patch_level_loss=True` (default): Compute discrepancy per patch
- `patch_level_loss=False`: Compute discrepancy at window level

**Purpose**:
Test whether fine-grained (patch-level) or coarse (window-level) discrepancy loss is more effective.

---

### 9. Patchify Mode Experiments

**Configuration**:
Three different patchify modes tested:
- **linear** (default): Patchify first, then linear projection (no CNN)
- **cnn_first**: CNN on full sequence, then patchify
- **patch_cnn**: Patchify first, then CNN per patch

**Structure Comparison**:

| Mode | Flow | Cross-Patch Information |
|------|------|------------------------|
| linear | Input → Patchify → Linear | None |
| cnn_first | Input → CNN → Patchify → Linear | Yes (via CNN receptive field) |
| patch_cnn | Input → Patchify → CNN (per patch) → Linear | None |

**Purpose**:
Test the impact of different patchification strategies on anomaly detection:

1. **linear**: Baseline approach following original MAE paper
   - No local feature extraction before patching
   - Simplest architecture

2. **cnn_first**: Local features extracted globally before patching
   - CNN sees full sequence, can share information across patches
   - May help when local patterns benefit from global context
   - **Potential Issue**: Information leakage across patches may compromise masking

3. **patch_cnn**: Independent CNN per patch
   - No cross-patch information leakage
   - Maintains strict patch independence for masking
   - Combines local feature extraction with MAE-style masking

**Hypothesis**:
- `patch_cnn` may perform best due to combining local feature extraction without violating masking assumptions
- `cnn_first` may underperform if cross-patch leakage makes masked patches predictable
- `linear` provides a clean baseline

---

## Comparison Table

### Patch Size Comparison

| Configuration | Num Patches | Patch Size | Purpose |
|--------------|-------------|------------|---------|
| **10 patches** | 10 | 10 | Coarse granularity, large context |
| **25 patches (Default)** | 25 | 4 | Balanced granularity |
| **50 patches** | 50 | 2 | Fine granularity, precise localization |

---

## Evaluation Strategy

**All ablations use the same evaluation protocol**:

1. **Data**: Same test dataset with 25% anomaly ratio
2. **Last Patch Masking**: Mask only the last patch (time steps 96-100 for patch_size=4)
3. **Anomaly Score**: Compute error metric on the masked last patch
4. **Label**: Binary label based on whether last patch contains anomaly
5. **Metrics**: ROC-AUC, Precision, Recall, F1-Score

**Key Point**: Only the training configuration differs. Evaluation is consistent across all ablations for fair comparison.

---

## What Each Ablation Tests

### Patch Size Impact
**Question**: What is the optimal patch granularity for anomaly detection?

- Too few patches (large patch size): May lack localization precision
- Too many patches (small patch size): May lack context, harder to reconstruct
- Optimal: Balance between context and granularity

### Masking Strategy Impact
**Question**: Does feature-wise masking outperform patch masking?

- Patch masking: All features masked at same patches
- Feature-wise masking: Each feature masked at different patches

### Patchify Mode Impact
**Question**: How does local feature extraction affect anomaly detection?

- Linear: Pure MAE style, no CNN preprocessing
- CNN-first: Global CNN before patching (potential information leakage)
- Patch-CNN: Independent CNN per patch (no leakage)

---

## Expected Results

Based on the MAE and self-distillation literature:

1. **Patch size shows non-monotonic relationship** - optimal balance exists
2. **Masking strategy may depend on anomaly type** - point vs collective anomalies
3. **Patch-CNN may outperform CNN-first** - due to preserved masking assumptions

---

## Code Implementation

All ablations are tested via grid search in `scripts/run_experiments.py`:

```python
# Parameter grid for ablation experiments
DEFAULT_PARAM_GRID = {
    'masking_ratio': [0.4, 0.7],
    'masking_strategy': ['patch', 'feature_wise'],
    'num_patches': [10, 25, 50],
    'margin': [0.25, 0.5, 1.0],
    'lambda_disc': [0.3, 0.5, 0.7],
    'margin_type': ['hinge', 'softplus', 'dynamic'],
    'force_mask_anomaly': [False, True],
    'patch_level_loss': [True, False],
    'patchify_mode': ['cnn_first', 'patch_cnn', 'linear'],
}

# Two-stage grid search
runner = ExperimentRunner(param_grid=DEFAULT_PARAM_GRID)
results = runner.run_grid_search(
    quick_epochs=15,      # Stage 1: quick screening
    full_epochs=100,      # Stage 2: full training on top-k
    top_k=150             # Number of candidates for Stage 2
)
```

Evaluation logic in `mae_anomaly/evaluator.py`:

```python
# Anomaly score: reconstruction error + discrepancy
recon_error = ((teacher_output - sequences) ** 2).mean(dim=2)
discrepancy = ((teacher_output - student_output) ** 2).mean(dim=2)
error = recon_error + self.config.lambda_disc * discrepancy
```

---

## Grid Search Parameters

The experiment runner uses a two-stage grid search approach:

**Stage 1: Quick Screening** (15 epochs default)
- All parameter combinations evaluated quickly
- Top candidates selected for full training

**Stage 2: Full Training** (50 epochs)
- Top candidates from Stage 1
- Full training with early stopping

**Default Parameter Grid**:
```python
DEFAULT_PARAM_GRID = {
    'masking_ratio': [0.4, 0.7],
    'masking_strategy': ['patch', 'feature_wise'],
    'num_patches': [10, 25, 50],
    'margin': [0.25, 0.5, 1.0],
    'lambda_disc': [0.3, 0.5, 0.7],
    'margin_type': ['hinge', 'softplus', 'dynamic'],
    'force_mask_anomaly': [False, True],
    'patch_level_loss': [True, False],
    'patchify_mode': ['cnn_first', 'patch_cnn', 'linear'],
}
# Total combinations: 2*2*3*3*3*3*2*2*3 = 3888
```

---

## Notes

1. **Architecture**: 1D-CNN + Transformer hybrid for better local feature extraction.

2. **Patch Size Experiments**: Tests 3 configurations (10, 25, 50 patches) to find optimal granularity.

3. **Masking Strategies**: Both patch and feature-wise masking are tested.

4. **Dataset Size**: 10000 train samples, 2500 test samples.

5. **Last Patch Only**: All evaluations focus on predicting the last patch to simulate real-time anomaly detection.

6. **Same Random Seed**: All experiments use the same random seed (42) for reproducibility.

---

**Status**: ✅ All ablation studies verified and ready for experimentation.
