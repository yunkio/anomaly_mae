# Model Architecture Documentation

**Last Updated**: 2026-01-23
**Model**: 1D-CNN + Transformer Self-Distilled MAE

---

## Overview

The model uses a hybrid 1D-CNN + Transformer architecture for multivariate time series anomaly detection with self-distillation.

---

## Architecture Components

### 1. 1D Convolutional Layers

**Purpose**: Extract local temporal features from raw input

**Structure**:
```
Input: (batch, num_features=8, seq_length=100)
↓
Conv1d(8 → 32, kernel=3, padding=1) + BatchNorm + ReLU
↓
Conv1d(32 → 64, kernel=3, padding=1) + BatchNorm + ReLU
↓
Output: (batch, d_model=64, seq_length=100)
```

**Benefits**:
- Captures local temporal patterns
- Reduces feature dimensionality
- Provides translation invariance

---

### 2. Patchify Modes

**Purpose**: Convert input into patch-level representations

The model supports 3 different patchify modes, controlled by `config.patchify_mode`:

#### 2.1 Linear Mode (`patchify_mode='linear'`)

**Default MAE-style approach**: Patchify first, then linear projection

```
Input: (batch, 100, 8)
↓
Patchify: (batch, 25, 4*8=32)
↓
Linear(32 → 64)
↓
Patches: (batch, 25, 64)
```

**Characteristics**:
- No CNN layers used
- Simplest approach, following original MAE paper
- Linear projection from raw patch values

---

#### 2.2 Patch CNN Mode (`patchify_mode='patch_cnn'`)

**Patchify first, then CNN per patch (no cross-patch leakage)**

```
Input: (batch, 100, 8)
↓
Patchify: (batch, 25, 4, 8) → (batch*25, 8, 4)
↓
Conv1d(8 → 32, kernel=3, padding=1) + BatchNorm + ReLU
Conv1d(32 → 64, kernel=3, padding=1) + BatchNorm + ReLU
↓
(batch*25, 64, 4)
↓
Flatten + Linear: (batch*25, 256) → (batch, 25, 64)
```

**Characteristics**:
- Each patch processed independently
- No information leakage between patches
- Stricter separation aligns with masking objectives
- Better for MAE pretraining where masked patches should be unpredictable

---

### Patchify Mode Comparison

| Mode | CNN Position | Cross-Patch Info | Best For |
|------|--------------|------------------|----------|
| linear | None | No | Baseline, simplest |
| patch_cnn | After patchify | No | Strict MAE-style masking |

---

### 3. Patch Embedding (Legacy Note)

**Note**: In `patch_cnn` mode, CNN output is projected to patch embeddings. In `linear` mode, raw patches are directly projected.

**Details**:
- 25 patches per sequence (default)
- Each patch covers 4 time steps (patch_size = seq_length / num_patches)
- Patch size balances context and granularity

---

### 3. Positional Encoding

**Purpose**: Add position information to patches

**Structure**:
- Sinusoidal encoding
- Max length: 5000
- Dimension: 64

**Formula**:
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Benefits**:
- Model learns position-aware representations
- No additional parameters (pre-computed)

---

### 4. Transformer Encoder

**Purpose**: Process patches and learn global context

**Structure**:
- Layers: 3
- Attention heads: 4
- d_model: 64
- Feedforward dim: 256
- Dropout: 0.1

**Details**:
- Multi-head self-attention captures dependencies
- Feedforward network adds non-linearity
- Layer normalization stabilizes training

---

### 5. Teacher Decoder

**Purpose**: Heavy decoder for accurate reconstruction

**Structure**:
- Layers: 4
- Attention heads: 4
- d_model: 64
- Feedforward dim: 256
- Dropout: 0.1

**Details**:
- More capacity for complex patterns
- Used in discrepancy computation
- Cross-attention to encoder outputs

---

### 6. Student Decoder

**Purpose**: Lightweight decoder for efficient anomaly detection

**Structure**:
- Layers: 1
- Attention heads: 4
- d_model: 64
- Feedforward dim: 256
- Dropout: 0.1

**Details**:
- Less capacity → struggles on anomalies
- Discrepancy with teacher reveals anomalies
- More efficient for deployment

---

### 7. Output Projection

**Purpose**: Convert decoder output back to time series

**Structure**:
```
Decoder output: (batch, 25, 64)
↓
Linear(64 → 32) per patch (patch_size * num_features = 4 * 8)
↓
Unpatchify: (batch, 100, 8)
```

**Details**:
- Reconstructs original input dimensions
- Applied to both teacher and student outputs

---

## Full Pipeline

The pipeline varies based on `patchify_mode`:

### Linear Mode (default)
```
Input: (batch, 100, 8)
    ↓
[Patchify]
    ↓ (batch, 25, 32)
[Linear Embedding]
    ↓ (batch, 25, 64)
[Random Patch Masking (40%)]
    ↓
[Positional Encoding]
    ↓
[Transformer Encoder (3 layers)]
    ↓
[Teacher Decoder (4 layers)] | [Student Decoder (1 layer)]
    ↓                           ↓
[Output Projection]         [Output Projection]
    ↓                           ↓
[Unpatchify]                [Unpatchify]
    ↓                           ↓
Teacher Output              Student Output
(batch, 100, 8)            (batch, 100, 8)
    ↓                           ↓
        [Discrepancy Computation]
                 ↓
         Anomaly Score
```

### CNN First Mode
```
Input: (batch, 100, 8)
    ↓
[1D-CNN Layers (full sequence)]
    ↓ (batch, 64, 100)
[Patchify CNN Features]
    ↓ (batch, 25, 256)
[Linear Projection]
    ↓ (batch, 25, 64)
[Random Patch Masking → Encoder → Decoders → Output]
```

### Patch CNN Mode
```
Input: (batch, 100, 8)
    ↓
[Patchify]
    ↓ (batch, 25, 4, 8)
[1D-CNN per patch (independent)]
    ↓ (batch, 25, 64)
[Random Patch Masking → Encoder → Decoders → Output]
```

---

## Masking Strategies

### Training Time

The model supports two masking strategies, controlled by `config.masking_strategy`:

**Patch Masking** (`masking_strategy='patch'`, default):
- Randomly mask 40% of patches (configurable via `masking_ratio`)
- All features masked at same time points
- Preserves cross-feature temporal coherence
- Suitable for detecting anomalies that affect multiple features simultaneously

**Feature-wise Masking** (`masking_strategy='feature_wise'`):
- Each feature masked independently at different **patches**
- Different patches masked for each feature
- Tests importance of cross-feature relationships
- Useful for detecting anomalies that affect individual features
- `force_mask_anomaly` supported: anomaly patches are masked for ALL features

```
Patch Masking Example (3 features, 5 patches):
     P0   P1   P2   P3   P4
F0:  ██   ░░   ██   ░░   ██    (patches 1,3 masked)
F1:  ██   ░░   ██   ░░   ██    (same patches)
F2:  ██   ░░   ██   ░░   ██    (same patches)

Feature-wise Masking Example (3 features, 5 patches):
     P0   P1   P2   P3   P4
F0:  ██   ░░   ██   ░░   ██    (patches 1,3 masked)
F1:  ░░   ██   ░░   ██   ██    (patches 0,2 masked)
F2:  ██   ██   ░░   ░░   ██    (patches 2,3 masked)

(██ = visible, ░░ = masked)
Each feature independently selects which patches to mask.
```

### Inference Time

**Last Patch Masking**:
- Mask only the last patch (time steps 96-100 for patch_size=4)
- Model predicts missing values
- Anomaly score based on reconstruction/discrepancy

---

## Self-Distillation Mechanism

### Training Loss

**Reconstruction Loss**:
```python
L_rec = MSE(teacher_out, original) + MSE(student_out, original)
```

**Discrepancy Loss** (with margin types):

1. **Hinge (default)**:
```python
L_disc = ReLU(margin - |teacher_error - student_error|)
```

2. **Softplus**:
```python
L_disc = Softplus(margin - |teacher_error - student_error|)
```

3. **Dynamic**:
```python
# Margin adapts based on normal samples' discrepancy distribution
dynamic_margin = mu + k * sigma
L_disc = ReLU(dynamic_margin - discrepancy)
```

**Total Loss**:
```python
L_total = L_rec + λ_disc * L_disc
```

**Hyperparameters**:
- margin = 0.5 (default)
- λ_disc = 0.5 (default)
- masking_ratio = 0.4 (default)

### Evaluation Metric

**Baseline** (use_discrepancy_loss=True):
```python
anomaly_score = MSE(teacher_out, original) + λ * MSE(teacher_out - student_out)
```

**TeacherOnly** (use_student=False):
```python
anomaly_score = MSE(teacher_out - original)
```

**StudentOnly** (use_teacher=False):
```python
anomaly_score = MSE(student_out - original)
```

---

## Loss Configuration

### Patch-Level vs Window-Level Loss

**Patch-Level Loss** (`patch_level_loss=True`, default):
- Compute discrepancy per patch
- Apply margin loss per patch
- Average across patches

**Window-Level Loss** (`patch_level_loss=False`):
- Average discrepancy across all patches first
- Apply margin loss once on the average
- Single scalar loss per sample

### Force Mask Anomaly

**force_mask_anomaly=True**:
- During training, force mask patches containing anomalies
- Ensures model learns to reconstruct normal patterns
- Prevents learning anomaly-specific patterns

---

## Design Choices

### Why 1D-CNN before Transformer?

1. **Local feature extraction**: CNNs excel at capturing local patterns
2. **Dimensionality reduction**: Maps 5 features → 64 channels
3. **Translation invariance**: Useful for time series
4. **Complementary**: CNN captures local, Transformer captures global

### Why Patch-based Processing?

1. **Computational efficiency**: 25 patches vs 100 tokens
2. **Context preservation**: Each patch contains 4 time steps
3. **Masking granularity**: Coarse enough for reconstruction task
4. **MAE-inspired**: Follows successful MAE design

### Why Self-Distillation?

1. **Anomaly sensitivity**: Student struggles more on anomalies
2. **Discrepancy signal**: Teacher-student gap indicates anomalies
3. **Regularization**: Prevents overfitting to anomalies
4. **Efficiency**: Student model lighter for deployment

---

## Key Advantages

1. **Hybrid Architecture**: Combines CNN (local) + Transformer (global)
2. **Self-Distillation**: Uses discrepancy for anomaly detection
3. **Patch-based**: Efficient processing with sufficient context
4. **Flexible Masking**: Supports multiple masking strategies
5. **Ablation-ready**: Easy to disable components for experiments

---

## Default Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| seq_length | 100 | Input sequence length |
| num_features | 8 | Multivariate features (server metrics) |
| d_model | 64 | Model dimension |
| num_patches | 25 | Number of patches |
| patch_size | 4 | Time steps per patch |
| patchify_mode | linear | Patchify mode (linear/patch_cnn) |
| masking_strategy | patch | Masking strategy (patch/feature_wise) |
| masking_ratio | 0.4 | Training masking ratio |
| num_encoder_layers | 3 | Encoder layers |
| num_teacher_decoder_layers | 4 | Teacher decoder layers |
| num_student_decoder_layers | 1 | Student decoder layers |
| margin | 0.5 | Discrepancy margin (fixed) |
| lambda_disc | 0.5 | Discrepancy loss weight (fixed) |
| margin_type | hinge | Margin loss type |
| patch_level_loss | True | Loss computation level |

---

**Status**: ✅ Architecture implemented and tested
