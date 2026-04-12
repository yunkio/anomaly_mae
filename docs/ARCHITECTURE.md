# Model Architecture Documentation

**Last Updated**: 2026-04-13
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
Input: (batch, num_features=8, seq_length=500)
↓
Conv1d(8 → 64, kernel=3, padding=1) + BatchNorm + ReLU
↓
Conv1d(64 → 128, kernel=3, padding=1) + BatchNorm + ReLU
↓
Output: (batch, d_model=128, seq_length=500)
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
Input: (batch, 500, 8)
↓
Patchify: (batch, 100, 5*8=40)
↓
Linear(40 → 128)
↓
Patches: (batch, 100, 128)
```

**Characteristics**:
- No CNN layers used
- Simplest approach, following original MAE paper
- Linear projection from raw patch values

---

#### 2.2 Patch CNN Mode (`patchify_mode='patch_cnn'`)

**Patchify first, then CNN per patch (no cross-patch leakage)**

```
Input: (batch, 500, 8)
↓
Patchify: (batch, 100, 5, 8) → (batch*100, 8, 5)
↓
Conv1d(8 → 64, kernel=3, padding=1) + BatchNorm + ReLU
Conv1d(64 → 128, kernel=3, padding=1) + BatchNorm + ReLU
↓
(batch*100, 128, 5)
↓
Flatten + Linear: (batch*100, 640) → (batch, 100, 128)
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
- 100 patches per sequence (default)
- Each patch covers 5 time steps (patch_size = seq_length / num_patches = 500 / 100)
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
- Layers: 2 (enc2, optimal from ablation study)
- Attention heads: 8
- d_model: 128
- Feedforward dim: 512
- Dropout: 0.15
- **Pre-Norm + GELU + eps=1e-6** (matching original MAE paper)
- Final LayerNorm after encoder stack

**Details**:
- `norm_first=True`: Pre-LayerNorm architecture (more stable training)
- `activation='gelu'`: GELU activation in feedforward layers
- `layer_norm_eps=1e-6`: Matching original MAE implementation
- Multi-head self-attention captures dependencies

---

### 5. Teacher Decoder

**Purpose**: Heavy decoder for accurate reconstruction

**Structure**:
- Layers: 4 (td4, optimal from ablation study)
- Attention heads: 8
- d_model: 128
- Feedforward dim: 512
- Dropout: 0.15
- Pre-Norm + GELU + eps=1e-6 (same as encoder)

**Details**:
- Deep decoder for high-quality reconstruction
- Used in discrepancy computation
- Cross-attention to encoder outputs

---

### 6. Student Decoder

**Purpose**: Lightweight decoder for efficient anomaly detection

**Structure**:
- Layers: 1 (sd1, shallow for better discrepancy signal)
- Attention heads: 8
- d_model: 128
- Feedforward dim: 512
- Dropout: 0.15

**Details**:
- Shallow decoder creates larger capacity gap with teacher
- Discrepancy with teacher reveals anomalies
- Decoder layer count can be varied in ablation studies

---

### 7. Output Projection

**Purpose**: Convert decoder output back to time series

**Structure**:
```
Decoder output: (batch, 100, 128)
↓
Linear(128 → 40) per patch (patch_size * num_features = 5 * 8)
↓
Unpatchify: (batch, 500, 8)
```

**Details**:
- Reconstructs original input dimensions
- Applied to both teacher and student outputs

---

## Full Pipeline

The pipeline varies based on `patchify_mode`:

### Linear Mode
```
Input: (batch, 500, 8)
    ↓
[Patchify]
    ↓ (batch, 100, 40)
[Linear Embedding]
    ↓ (batch, 100, 128)
[Random Patch Masking (15%)]
    ↓
[Positional Encoding]
    ↓
[Transformer Encoder (2 layers)]
    ↓
[Teacher Decoder (4 layers)] | [Student Decoder (1 layer)]
    ↓                           ↓
[Output Projection]         [Output Projection]
    ↓                           ↓
[Unpatchify]                [Unpatchify]
    ↓                           ↓
Teacher Output              Student Output
(batch, 500, 8)            (batch, 500, 8)
    ↓                           ↓
        [Discrepancy Computation]
                 ↓
         Anomaly Score
```

### Patch CNN Mode (default)
```
Input: (batch, 500, 8)
    ↓
[Patchify]
    ↓ (batch, 100, 5, 8)
[1D-CNN per patch (independent)]
    ↓ (batch, 100, 128)
[Random Patch Masking → Encoder → Decoders → Output]
```

---

## Masking Strategy

### Training Time

**Patch Masking** (default):
- Randomly mask 15% of patches (configurable via `masking_ratio`)
- All features masked at same time points (same patches masked across all features)
- Preserves cross-feature temporal coherence
- Suitable for detecting anomalies that affect multiple features simultaneously

```
Patch Masking Example (3 features, 5 patches):
     P0   P1   P2   P3   P4
F0:  ██   ░░   ██   ░░   ██    (patches 1,3 masked)
F1:  ██   ░░   ██   ░░   ██    (same patches)
F2:  ██   ░░   ██   ░░   ██    (same patches)

(██ = visible, ░░ = masked)
```

### Inference Time

During inference, each patch is masked one at a time with N forward passes per window:
- Mask each patch position independently
- Compute reconstruction error and discrepancy for masked positions
- Per-patch scores (n_windows × num_patches) computed during inference
- **Point-level aggregation**: patch scores are mean-aggregated to physical timestamps for evaluation metrics
- All metrics (ROC-AUC, F1, precision, recall, PA%K) use mean-aggregated point-level scores
- PA%K AUC: K=0..100 sweep → integrated scalar per metric (robustness measure)

See [INFERENCE_MODES.md](INFERENCE_MODES.md) for detailed flow diagrams.

---

## MAE Masking Architecture

The model supports two masking architectures, controlled by `config.mask_after_encoder`:

### Standard Mode (`mask_after_encoder=False`)

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

### MAE-Style Mode (`mask_after_encoder=True`, default)

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

### Shared Mode (`shared_mask_token=True`)

**Single mask token**: Both teacher and student decoders use the same learnable mask token

```python
self.mask_token = nn.Parameter(...)  # Shared between teacher/student
```

**Characteristics**:
- Simpler model (fewer parameters)
- Teacher and student see identical masked representations
- Default behavior

### Separate Mode (`shared_mask_token=False`, default)

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
- Controlled by `teacher_only_warmup_epochs` (default=-1, auto: `num_epochs // 2`)
- Allows teacher to learn basic reconstruction before introducing discrepancy
- When `freeze_teacher_after_warmup=True`: encoder/teacher frozen at warmup end (method C: eval + no_grad)

**Implementation**:
```python
# In trainer.py __init__():
if config.teacher_only_warmup_epochs < 0:
    config.teacher_only_warmup_epochs = config.num_epochs // 2  # Auto: 25 for 50 epochs
```

**Epoch Offset (Train Augmentation)**:

When `epoch_offset=True`, each epoch's train window start positions are shifted by a random offset from `[0, stride)`. Offsets are sampled without replacement (non-replacement within each cycle of `stride` epochs), so over `stride` epochs all possible offsets are covered exactly once. Test windows are always fixed at offset=0.

```python
# epoch_offset=True, stride=21:
# Epoch 0: windows at [7, 28, 49, ...]   (random offset 7)
# Epoch 1: windows at [14, 35, 56, ...]  (random offset 14)
# ...
# After 21 epochs: all offsets 0-20 used exactly once → full stride=1 coverage
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

4. **None** (unbounded):
```python
L_disc = -discrepancy  # No margin, unbounded maximization
```

**Anomaly Loss Direction** (`anomaly_loss_direction`):
- `'maximize'` (default): Push anomaly discrepancy UP (standard)
- `'minimize'`: Push anomaly discrepancy DOWN (same as normal, for ablation)

**Total Loss**:
```python
L_total = L_rec + normal_loss_weight * L_normal + anomaly_loss_weight * L_anomaly
```

**Hyperparameters**:
- margin = 0.5 (default)
- λ_disc = 2.0 (default)
- masking_ratio = 0.15 (default)
- normal_loss_weight = 1.0, anomaly_loss_weight = 2.0

### Adversarial Discriminator (Optional)

**Purpose**: Prevent student decoder's "noise strategy" — adding noise to artificially inflate discrepancy. Forces structurally different output.

**Architecture** (`PatchDiscriminator`):
```
Input: (batch, num_features, patch_size)
↓
SpectralNorm(Conv1d(num_features → 64, k=3)) + LeakyReLU(0.2)
↓
SpectralNorm(Conv1d(64 → 32, k=3)) + LeakyReLU(0.2)
↓
AdaptiveAvgPool1d(1) + Flatten
↓
SpectralNorm(Linear(32 → 1))
↓
Output: (batch, 1) logit (real/fake)
```

**Training Flow** (when `use_discriminator=True`):
```
1. Forward: teacher_out, student_out, mask = model(x)
2. Base loss: L_rec + L_disc (normal + anomaly with margin)
3. Extract patches: real (original) & fake (student output) from masked regions
4. D Step (epoch ≥ disc_warmup):
   - D trains on ALL patches (normal + anomaly): BCE(D(real), 1) + BCE(D(fake.detach()), 0)
5. Student Adversarial (anomaly patches only):
   - adv_loss = BCE(D(anomaly_fake), 1)  — fool D
   - λ_adv = adaptive_lambda(normal_loss, anomaly_disc_forward, adv_loss)
   - L_total += λ_adv * adv_loss
6. Model backward: L_total.backward() → optimizer.step()
```

**Key Design Decisions**:
- D distinguishes real vs fake (NOT normal vs anomaly)
- TTUR: D learning rate = main_lr × `disc_lr_ratio` (4×), β1=0
- Adaptive λ uses anomaly discrepancy WITHOUT margin reversal (prevents signal cancellation)
- Spectral Normalization on all D layers (Lipschitz constraint)
- Single AMP GradScaler shared between D and model optimizers

### Gradient Reversal Layer (Optional)

**Purpose**: Adversarial feature suppression — force encoder/student to produce representations that cannot distinguish normal from anomaly.

**Two modes** (`grl_mode`):
1. **Classifier** (DANN-style): `AnomalyClassifierHead` with GRL gradient reversal
2. **WDGRL**: `WassersteinCritic` with gradient penalty (no GRL needed, minimax via separate optimizer)

**Key config**: `use_grl=True`, `grl_target_mode` (patch/window), `grl_balanced_sampling`, `grl_use_focal`, `grl_cls_lr_ratio`

**Mutual exclusion**: `use_grl` and `use_discriminator` cannot both be True.

### Feature Matching Loss (Optional)

**Purpose**: Align teacher and student hidden representations on normal patches.

```python
FM_loss = 1 - cosine_similarity(teacher_hidden, student_hidden)  # masked normal patches
```

**Key config**: `use_feature_matching=True`, `fm_distance_metric` (cosine/l2), `fm_adaptive_lambda`, `fm_loss_weight`

Can be combined with or replace output discrepancy (`use_output_discrepancy=False` disables OD, FM-only training).

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

### Point-Level Aggregation

All scoring formulas above produce **patch-level** scores (n_windows × num_patches). For evaluation:

1. **Mean aggregation**: Each timestamp's score = mean of all patch scores covering it
2. **Point-level ROC/threshold**: ROC curve computed from (point_labels, point_scores), threshold via F1-optimal
3. **Primary metrics**: F1, precision, recall computed from point-level binary predictions
4. **PA%K F1**: Mean-aggregated scores → threshold → PA%K segment adjustment → F1/Precision/Recall/F1_T
5. **PA%K ROC/PRC-AUC**: Threshold sweep on mean scores → PA%K adjustment per threshold → AUC
6. **PA%K AUC**: Sweep K=0..100 for each metric, integrate → single robustness scalar per metric
   - `pak_auc_f1` (best_f1_w_pa): Per-K threshold re-optimization after PA%K segment adjustment (Kim et al., AAAI 2022). Used for best epoch selection.
   - `pak_auc_f1_raw` (raw_f1_w_pa): Fixed pre-PA threshold, legacy comparison metric.
7. **Loss statistics** (disc_ratio, Cohen's d): Remain patch-level (describe model behavior)

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
- During training, anomaly patches are **prioritized** for masking within a fixed budget
- Masking budget = `round(num_patches * masking_ratio)` — always exactly this many patches are masked per sample
- If anomaly patches ≤ budget: all anomaly patches are masked, remaining slots filled with random normal patches
- If anomaly patches > budget: only `budget` anomaly patches are masked (randomly selected), excess remain visible as encoder context
- This maintains uniform masking count across the batch, which is required by `_encode_visible_only` (standard MAE encoder)
- Ensures model primarily learns to reconstruct normal patterns while preserving batch-level masking invariants

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
| seq_length | 500 | Input sequence length |
| num_features | 8 | Multivariate features (server metrics) |
| d_model | 128 | Model dimension (or `'dynamic'` for auto-selection) |
| nhead | 8 | Number of attention heads |
| dim_feedforward | 512 | FFN dimension (auto: 4× d_model if not overridden) |
| num_patches | 100 | Number of patches (seq_length / patch_size) |
| patch_size | 5 | Time steps per patch (fixed) |
| patchify_mode | patch_cnn | Patchify mode (patch_cnn/linear) |
| cnn_channels | (64, 128) | CNN channels (d_model//2, d_model) |
| masking_ratio | 0.15 | Training masking ratio |
| mask_after_encoder | True | Standard MAE: encode visible only, insert mask before decoder |
| shared_mask_token | False | Separate mask tokens for teacher/student |
| num_encoder_layers | 2 | Encoder layers (enc2) |
| num_teacher_decoder_layers | 4 | Teacher decoder layers (td4) |
| num_student_decoder_layers | 1 | Student decoder layers (sd1) |
| margin | 0.5 | Discrepancy margin (fixed) |
| lambda_disc | 2.0 | Discrepancy loss weight |
| margin_type | dynamic | Margin type (dynamic/hinge/softplus/none) |
| anomaly_loss_direction | maximize | Anomaly disc direction (maximize/minimize) |
| normal_loss_weight | 1.0 | Normal discrepancy loss weight |
| dynamic_margin_k | 2.0 | k for dynamic margin (mu + k*sigma) |
| patch_level_loss | True | Loss computation level |
| learning_rate | 1e-3 | Learning rate |
| weight_decay | 1e-3 | Weight decay (bias/norm excluded) |
| teacher_only_warmup_epochs | -1 | Auto: num_epochs // 2 |
| warmup_epochs | 10 | LR linear warmup epochs |
| epoch_offset | True | Non-replacement random train window offset per epoch |
| normalize_mode | zscore | Per-feature z-score (train-only fit) or minmax |
| best_epoch_metric | pak_auc_f1 | Best epoch selection metric |
| num_shared_decoder_layers | 0 | Shared layers between decoders |
| anomaly_loss_weight | 2.0 | Weight for anomaly samples in loss |
| use_amp | True | Mixed Precision Training (1.2x speedup, 40% memory ↓) |
| use_discriminator | False | Enable adversarial discriminator for student decoder |
| disc_lr_ratio | 4.0 | D learning rate = main_lr × ratio (TTUR) |
| adaptive_lambda | True | Gradient magnitude balancing (VQGAN-style) |
| disc_warmup_epochs | 10 | Epoch to start D training |
| disc_channels | (64, 32) | Discriminator 1D CNN channels |
| adv_loss_weight | 1.0 | Adversarial loss weight (disc:adv ratio) |
| use_grl | False | Enable GRL adversarial training |
| use_feature_matching | False | Enable feature matching loss |

### Optimizer & LR Schedule

- **AdamW**: betas=(0.9, 0.99), bias/LayerNorm params excluded from weight decay
- **LR Schedule**: LinearLR warmup (1e-4 → 1.0, `warmup_epochs`) + CosineAnnealingLR (remaining epochs)
- **D optimizer** (when `use_discriminator=True`): AdamW, betas=(0.0, 0.99), no weight decay, CosineAnnealingLR from disc_warmup
- **WDGRL critic** (when `grl_mode='wdgrl'`): Adam, betas=(0.5, 0.999), separate LR

### Dynamic d_model (Set C)

When `d_model='dynamic'`, the model dimension is auto-selected per-dataset based on `num_features`:

```python
# resolve_dynamic_d_model(num_features, patch_size)
raw = patch_size * num_features
d_model = min(d for d in [128, 192, 256, 384, 512] if d >= raw)
# Capped at 512; dim_feedforward = 4 * d_model (auto-computed)
```

Set C config: `patch_size=10, patchify_mode='linear', d_model='dynamic'`. See `set_guideline.md` for details.

---

**Status**: ✅ Architecture implemented and tested
