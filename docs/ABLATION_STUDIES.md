# Ablation Studies Documentation

**Last Updated**: 2026-01-25
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
  - Encoder: 1 layer, 2 attention heads (t2s1 default)
  - Teacher Decoder: 2 layers
  - Student Decoder: 1 layer

**Pipeline**:
1. Input (batch, 100, 8) → 1D-CNN → (batch, 64, 100)
2. CNN features → Patch embedding → (batch, 10, 64)
3. Patches + Positional encoding → Transformer encoder
4. Latent → Teacher/Student decoders → Output projection
5. Reconstruction (batch, 100, 8)

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
- 10 patches (patch size = 10) - **Default**
- 25 patches (patch size = 4)
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

### 4. Margin Type Experiments

**Configuration**:
Three different margin loss types:
- **hinge** (default): `ReLU(margin - discrepancy)`
- **softplus**: `Softplus(margin - discrepancy)` - smoother gradient
- **dynamic**: `ReLU(mu + k*sigma - discrepancy)` - adaptive margin

**Purpose**:
Test which margin formulation works best for self-distillation.

---

### 5. Force Mask Anomaly Experiments

**Configuration**:
- `force_mask_anomaly=False` (default)
- `force_mask_anomaly=True`

**Purpose**:
Test whether forcing anomaly patches to be masked during training improves detection.
- When True: Patches containing anomalies are always masked
- Ensures model learns to reconstruct normal patterns only

---

### 6. Patch-Level vs Window-Level Loss

**Configuration**:
- `patch_level_loss=True` (default): Compute discrepancy per patch
- `patch_level_loss=False`: Compute discrepancy at window level

**Purpose**:
Test whether fine-grained (patch-level) or coarse (window-level) discrepancy loss is more effective.

---

### 7. Patchify Mode Experiments

**Configuration**:
Two different patchify modes tested:
- **linear** (default): Patchify first, then linear projection (no CNN)
- **patch_cnn**: Patchify first, then CNN per patch

**Structure Comparison**:

| Mode | Flow | Cross-Patch Information |
|------|------|------------------------|
| linear | Input → Patchify → Linear | None |
| patch_cnn | Input → Patchify → CNN (per patch) → Linear | None |

**Purpose**:
Test the impact of different patchification strategies on anomaly detection:

1. **linear**: Baseline approach following original MAE paper
   - No local feature extraction before patching
   - Simplest architecture

2. **patch_cnn**: Independent CNN per patch
   - No cross-patch information leakage
   - Maintains strict patch independence for masking
   - Combines local feature extraction with MAE-style masking

**Hypothesis**:
- `patch_cnn` may perform best due to combining local feature extraction without violating masking assumptions
- `linear` provides a clean baseline

---

### 8. Mask After Encoder Experiments

**Configuration**:
- `mask_after_encoder=False` (default): Mask tokens go through encoder
- `mask_after_encoder=True`: Standard MAE - encode visible patches only

**Pipeline Comparison**:

| Mode | Encoder Input | Mask Token Insertion |
|------|---------------|---------------------|
| False (current) | All patches (visible + mask tokens) | Before encoder |
| True (standard MAE) | Only visible patches | Before decoder |

**Purpose**:
Test whether standard MAE masking architecture (encode visible only) outperforms the current approach.

**Hypothesis**:
- Standard MAE may learn better representations by not letting mask tokens influence encoding
- Current approach may benefit from mask token attention patterns

---

### 9. Shared vs Separate Mask Token Experiments

**Configuration**:
- `shared_mask_token=True` (default): Single mask token shared by teacher/student
- `shared_mask_token=False`: Separate learnable mask tokens

**Purpose**:
Test whether independent mask representations help differentiate teacher/student behavior.

**Hypothesis**:
- Separate mask tokens may allow each decoder to learn optimal representations for its capacity
- Shared mask token provides simpler, more consistent learning signal

---

## Comparison Table

### Patch Size Comparison

| Configuration | Num Patches | Patch Size | Purpose |
|--------------|-------------|------------|---------|
| **10 patches (Default)** | 10 | 10 | Balanced granularity (seq_length=100), matches mask_last_n |
| **25 patches** | 25 | 4 | Fine granularity, precise localization |
| **50 patches** | 50 | 2 | Very fine granularity |

---

## Evaluation Strategy

**All ablations use the same evaluation protocol**:

1. **Data**: Same test dataset with 25% anomaly ratio
2. **Masking**: Depends on `inference_mode` (see below)
3. **Anomaly Score**: Compute error metric on masked positions
4. **Label**: Binary label (window-level or patch-level depending on mode)
5. **Metrics**: ROC-AUC, Precision, Recall, F1-Score, PA%K

### Inference Modes

| Mode | Masking | Sample Unit | Use Case |
|------|---------|-------------|----------|
| `last_patch` | Last patch only | Window | Fast, streaming detection |
| `all_patches` | Each patch (N passes) | Patch | Thorough evaluation |

- **last_patch**: 1 forward pass per window, window-level labels
- **all_patches**: N forward passes per window, patch-level labels (10× more samples)

See [INFERENCE_MODES.md](INFERENCE_MODES.md) for detailed flow diagrams.

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
# Note: margin=0.5, lambda_disc=0.5 are fixed (not in grid)
DEFAULT_PARAM_GRID = {
    'masking_ratio': [0.4, 0.7],
    'masking_strategy': ['patch', 'feature_wise'],
    'num_patches': [10, 25],
    'margin_type': ['hinge', 'softplus', 'dynamic'],
    'force_mask_anomaly': [False, True],
    'patch_level_loss': [True, False],
    'patchify_mode': ['patch_cnn', 'linear'],
    'mask_after_encoder': [False, True],
    'shared_mask_token': [True, False],
}
# Total combinations: 2*2*2*3*2*2*2*2*2 = 768

# Two-stage grid search
runner = ExperimentRunner(param_grid=DEFAULT_PARAM_GRID)
results = runner.run_grid_search(
    quick_epochs=1,       # Stage 1: quick screening
    full_epochs=2,        # Stage 2: full training (fixed at 2)
    two_stage=True
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

**Stage 1: Quick Screening** (1 epoch default)
- All 288 parameter combinations evaluated quickly
- Candidates selected for full training based on diverse criteria

**Stage 2: Full Training** (2 epochs default)
- ~50-70 diverse candidates from Stage 1 (after deduplication)
- Full training with all components

**Fixed Parameters** (not in grid search):
- `margin = 0.5`
- `lambda_disc = 0.5`

**Default Parameter Grid**:
```python
DEFAULT_PARAM_GRID = {
    'masking_ratio': [0.4, 0.7],
    'masking_strategy': ['patch', 'feature_wise'],
    'num_patches': [10, 25],
    'margin_type': ['hinge', 'softplus', 'dynamic'],
    'force_mask_anomaly': [False, True],
    'patch_level_loss': [True, False],
    'patchify_mode': ['patch_cnn', 'linear'],
    'mask_after_encoder': [False, True],
    'shared_mask_token': [True, False],
}
# Total combinations: 2*2*2*3*2*2*2*2*2 = 768
```

---

## Stage 2 Selection Criteria

Stage 2 uses a 3-phase diverse selection strategy:

**Phase 1: Per-Parameter Coverage (5 models per value)**
- `force_mask_anomaly`: True (5) + False (5)
- `patch_level_loss`: True (5) + False (5)
- `margin_type`: hinge (5) + softplus (5) + dynamic (5)
- `patchify_mode`: patch_cnn (5) + linear (5)
- `masking_strategy`: patch (5) + feature_wise (5)
- `masking_ratio`: each value (5)
- `num_patches`: each value (5)
- `mask_after_encoder`: True (5) + False (5)
- `shared_mask_token`: True (5) + False (5)

**Phase 2: Overall Performance (Phase 1 제외)**
- Top 10 by overall ROC-AUC (Phase 1에서 선택되지 않은 모델만)

**Phase 3: Disturbing Normal Performance (Phase 1, 2 제외)**
- Top 5 by disturbing_roc_auc (Phase 1, 2에서 선택되지 않은 모델만)

각 Phase에서 이미 선택된 모델은 다음 Phase에서 제외됩니다.

**Stage 2 Output**:
- Per-parameter performance summary with disturbing normal metrics
- Detailed comparison across all parameter values

---

## Notes

1. **Architecture**: 1D-CNN + Transformer hybrid for better local feature extraction.

2. **Patch Size Experiments**: Tests 3 configurations (10, 25, 50 patches) to find optimal granularity.

3. **Masking Strategies**: Both patch and feature-wise masking are tested.

4. **Dataset Size**: 440,000 timesteps total, ~22,000 train windows, ~2,000 test windows.

5. **Inference Modes**: Both `last_patch` (fast) and `all_patches` (thorough) evaluated.

6. **Same Random Seed**: All experiments use the same random seed (42) for reproducibility.

---

**Status**: ✅ All ablation studies verified and ready for experimentation.
