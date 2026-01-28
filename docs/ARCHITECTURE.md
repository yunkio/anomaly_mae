# Model Architecture Documentation

**Last Updated**: 2026-01-28
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

The model supports 2 different patchify modes, controlled by `config.patchify_mode`:

#### 2.1 Linear Mode (`patchify_mode='linear'`)

**MAE-style approach**: Patchify first, then linear projection

```
Input: (batch, 100, 8)
↓
Patchify: (batch, 10, 10*8=80)
↓
Linear(32 → 64)
↓
Patches: (batch, 10, 64)
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
Patchify: (batch, 10, 10, 8) → (batch*10, 8, 10)
↓
Conv1d(8 → 32, kernel=3, padding=1) + BatchNorm + ReLU
Conv1d(32 → 64, kernel=3, padding=1) + BatchNorm + ReLU
↓
(batch*10, 64, 10)
↓
Flatten + Linear: (batch*10, 640) → (batch, 10, 64)
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
- 10 patches per sequence (default)
- Each patch covers 10 time steps (patch_size = seq_length / num_patches = 100 / 10)
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
- Layers: 1
- Attention heads: 2
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
- Layers: 2
- Attention heads: 2
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
- Attention heads: 2
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
Decoder output: (batch, 10, 64)
↓
Linear(64 → 80) per patch (patch_size * num_features = 10 * 8)
↓
Unpatchify: (batch, 100, 8)
```

**Details**:
- Reconstructs original input dimensions
- Applied to both teacher and student outputs

---

## Full Pipeline

The pipeline varies based on `patchify_mode`:

### Linear Mode
```
Input: (batch, 100, 8)
    ↓
[Patchify]
    ↓ (batch, 10, 80)
[Linear Embedding]
    ↓ (batch, 10, 64)
[Random Patch Masking (20%)]
    ↓
[Positional Encoding]
    ↓
[Transformer Encoder (1 layer)]
    ↓
[Teacher Decoder (2 layers)] | [Student Decoder (1 layer)]
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

### Patch CNN Mode (default)
```
Input: (batch, 100, 8)
    ↓
[Patchify]
    ↓ (batch, 10, 10, 8)
[1D-CNN per patch (independent)]
    ↓ (batch, 10, 64)
[Random Patch Masking → Encoder → Decoders → Output]
```

---

## Masking Strategies

### Training Time

The model supports two masking strategies, controlled by `config.masking_strategy`:

**Patch Masking** (`masking_strategy='patch'`, default):
- Randomly mask 20% of patches (configurable via `masking_ratio`)
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

The model supports two inference modes, controlled by `config.inference_mode`:

**Last Patch Mode** (`inference_mode='last_patch'`, default):
- Mask only the last patch (time steps 90-99 for patch_size=10)
- Single forward pass per window
- Window-level score for sample metrics (ROC-AUC, F1, etc.)
- Suitable for real-time/streaming scenarios

**All Patches Mode** (`inference_mode='all_patches'`):
- Mask each patch one at a time (N forward passes per window)
- Per-patch scores with patch-level labels
- Each patch is a sample for metrics (N× more samples)
- More thorough evaluation, 10× coverage per timestep

See [INFERENCE_MODES.md](INFERENCE_MODES.md) for detailed flow diagrams.

---

## MAE Masking Architecture

The model supports two masking architectures, controlled by `config.mask_after_encoder`:

### Standard Mode (`mask_after_encoder=False`, default)

**Current behavior**: Mask tokens are inserted before encoder

```
Input: (batch, 100, 8)
    ↓
[Embed Input] → (num_patches, batch, d_model)
    ↓
[Insert Mask Tokens at masked positions]
    ↓
[Positional Encoding]
    ↓
[Encoder processes ALL patches including mask tokens]
    ↓
[Decoder]
    ↓
Output
```

**Characteristics**:
- Mask tokens participate in encoder attention
- Encoder sees full sequence length
- Simpler implementation

### MAE-Style Mode (`mask_after_encoder=True`)

**Standard MAE approach**: Encode visible patches only, insert mask tokens before decoder

```
Input: (batch, 100, 8)
    ↓
[Embed Input] → (num_patches, batch, d_model)
    ↓
[Remove masked patches (keep visible only)]
    ↓
[Positional Encoding (visible patches only)]
    ↓
[Encoder processes ONLY visible patches]
    ↓
[Insert mask tokens at masked positions]
    ↓
[Decoder]
    ↓
Output
```

**Characteristics**:
- Encoder is more efficient (processes fewer tokens)
- Follows original MAE paper design
- Mask tokens don't influence encoder representations
- Better separation between visible and masked information

---

## Mask Token Configuration

The model supports shared or separate mask tokens, controlled by `config.shared_mask_token`:

### Shared Mode (`shared_mask_token=True`, default)

**Single mask token**: Both teacher and student decoders use the same learnable mask token

```python
self.mask_token = nn.Parameter(...)  # Shared between teacher/student
```

**Characteristics**:
- Simpler model (fewer parameters)
- Teacher and student see identical masked representations
- Default behavior

### Separate Mode (`shared_mask_token=False`)

**Separate mask tokens**: Teacher and student decoders have independent mask tokens

```python
self.teacher_mask_token = nn.Parameter(...)  # For teacher decoder
self.student_mask_token = nn.Parameter(...)  # For student decoder
```

**Characteristics**:
- Each decoder can learn its own mask representation
- More flexibility in reconstruction approach
- May help differentiate teacher/student behavior on masked regions

---

## Self-Distillation Mechanism

### Encoder Gradient Detachment

**Student decoder does NOT update encoder**:
- Student decoder receives `.detach()`ed encoder output
- Only the teacher reconstruction loss updates the encoder
- This ensures the encoder learns to represent normal patterns (via teacher) without being corrupted by the student's conflicting objectives

**Implementation**:
```python
# In model.py forward():
if self.config.use_student and self.student_decoder is not None:
    if self.mask_after_encoder:
        student_latent = self._insert_mask_tokens_and_unshuffle(
            latent_visible.detach(), ids_restore, seq_len, student_mask_token
        )
    else:
        student_latent = latent.detach()  # Detach encoder output
    student_output = self.student_decoder(student_latent)
```

### Warm-up Epochs

**Teacher-only warm-up period**:
- First N epochs train only the teacher model (no discrepancy/student loss)
- Controlled by `teacher_only_warmup_epochs` (default=3)
- Allows teacher to learn basic reconstruction before introducing discrepancy

**Implementation**:
```python
# In trainer.py train():
teacher_warmup = getattr(self.config, 'teacher_only_warmup_epochs', 3)
for epoch in range(self.config.num_epochs):
    teacher_only = (epoch < teacher_warmup)  # True during warm-up
    epoch_losses = self.train_epoch(epoch, teacher_only=teacher_only)
```

### Training Loss

**Reconstruction Loss**:
```python
L_rec = MSE(teacher_out, original) + MSE(student_out, original)
```

**Discrepancy Loss** (with margin types):

1. **Hinge**:
```python
L_disc = ReLU(margin - |teacher_error - student_error|)
```

2. **Softplus**:
```python
L_disc = Softplus(margin - |teacher_error - student_error|)
```

3. **Dynamic** (default):
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
- masking_ratio = 0.2 (default)

### Evaluation Metric

**Baseline** (use_discrepancy_loss=True):
```python
anomaly_score = MSE(teacher_out, original) + λ * MSE(teacher_out - student_out)
```

**Per-Component Scoring** (`evaluate_by_score_type()`):
```python
# Individual score components for CSV columns
disc_only_score = MSE(teacher_out - student_out)
teacher_recon_score = MSE(teacher_out - original)
student_recon_score = MSE(student_out - original)
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
2. **Dimensionality reduction**: Maps 8 features → 64 channels
3. **Translation invariance**: Useful for time series
4. **Complementary**: CNN captures local, Transformer captures global

### Why Patch-based Processing?

1. **Computational efficiency**: 10 patches vs 100 tokens
2. **Context preservation**: Each patch contains 10 time steps
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
| nhead | 2 | Number of attention heads |
| dim_feedforward | 256 | FFN dimension (4x d_model) |
| num_patches | 10 | Number of patches (seq_length / patch_size) |
| patch_size | 10 | Time steps per patch (fixed) |
| patchify_mode | patch_cnn | Patchify mode (patch_cnn/linear) |
| cnn_channels | (32, 64) | CNN channels (d_model//2, d_model) |
| masking_strategy | patch | Masking strategy (patch/feature_wise) |
| masking_ratio | 0.2 | Training masking ratio |
| mask_after_encoder | False | Mask tokens go through encoder (non-standard) |
| shared_mask_token | True | Share mask token between teacher/student |
| num_encoder_layers | 1 | Encoder layers |
| num_teacher_decoder_layers | 2 | Teacher decoder layers (t2s1) |
| num_student_decoder_layers | 1 | Student decoder layers |
| margin | 0.5 | Discrepancy margin (fixed) |
| lambda_disc | 0.5 | Discrepancy loss weight (fixed) |
| margin_type | dynamic | Margin loss type (dynamic/hinge/softplus) |
| dynamic_margin_k | 1.5 | k for dynamic margin (mu + k*sigma) |
| patch_level_loss | True | Loss computation level |
| learning_rate | 5e-3 | Learning rate |
| weight_decay | 1e-5 | Weight decay |
| teacher_only_warmup_epochs | 3 | Epochs for teacher-only training |
| warmup_epochs | 10 | Learning rate warm-up epochs |
| num_shared_decoder_layers | 0 | Shared layers between decoders |
| anomaly_loss_weight | 1.0 | Weight for anomaly samples in loss |
| use_amp | True | Mixed Precision Training (1.2x speedup, 40% memory ↓) |

---

**Status**: ✅ Architecture implemented and tested
