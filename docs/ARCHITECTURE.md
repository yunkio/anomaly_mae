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

**Asymmetric decoder WIDTH (`decoder_half_dim`, 2026-06-15)**: by default the
decoder width = encoder `d_model` (only DEPTH is asymmetric: e4 > td3 > sd2).
Set `decoder_half_dim=True` for MAE-style WIDTH asymmetry — teacher/student/shared
decoders, mask tokens, output projections, and GRL/SCAD heads run at `d_model//2`
(decoder `dim_feedforward = (d_model//2)*4`, nhead reused), while the **encoder
stays `d_model`**. A `decoder_embed = Linear(d_model, d_model//2)` narrows the
encoder latent (teacher-side; student detaches it). Works with dynamic `d_model`
(always //2 of resolved). Requires `mask_after_encoder` + transformer-enc-dec,
incompatible with teacher-output-EMA; nhead-non-divisible or odd d_model raise
`ValueError`. Default (False) → `decoder_embed=Identity`, byte-identical. FM and
discrepancy are unchanged (both decoders share the same width).

**GRL attachment layer (`grl_attach_layer`, 2026-06-15)**: `'last'` (default) attaches
the GRL classifier to the student decoder's FINAL hidden (h2, just before output
projection). `'first'` attaches it to the FIRST student-decoder layer's output (h1),
so adversarial invariance acts on an intermediate representation and the final layer
specializes for reconstruction (reconstruction/FM still use h2). Falls back to 'last'
if student depth ≤ 1; byte-identical when 'last'.

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

**Dynamic warm-up early-stop** (`use_teacher_warmup_early_stop=True`, opt-in; default-off byte-identical):
End the teacher-only warm-up *dynamically* instead of at the fixed `teacher_only_warmup_epochs`
(which then acts as an upper bound). On trigger, model+optimizer+scheduler are reverted to the
metric's best epoch before the student joins. Two metrics:
- `teacher_warmup_early_stop_metric='recon_snr'` (default): **maximize** train recon SNR
  (`(recon_a−recon_n)/(σ_a+σ_n+ε)`, anomaly↔normal separation); trigger when the strict-max best
  is not improved for `teacher_warmup_early_stop_patience` (10) epochs, after
  `teacher_warmup_early_stop_min_epochs` (50). Revert to the highest-SNR epoch.
- `teacher_warmup_early_stop_metric='train_loss'` (2026-06-20, peak_reversal): **minimize** epoch
  train_loss (= teacher reconstruction during warm-up). From `teacher_warmup_es_min_epoch` (20),
  every `teacher_warmup_es_check_interval` (5) epochs compare to the best-low; a check worsening
  by more than `teacher_warmup_es_relative_threshold` (1%) counts a reversal, and
  `teacher_warmup_es_patience_checks` (2) consecutive reversals trigger a revert to the
  lowest-train_loss check epoch. Targets the overfitting-onset point (recon rising), distinct from
  recon_snr's separation-peak. The two metric branches are mutually `elif`-separated so the
  recon_snr path is byte-identical.

**`train_recon_snr` logging** (2026-06-20): the train recon SNR above is computed and stored in
`training_histories.json` as `train_recon_snr` for **every run** (not only early-stop runs), all
epochs; `None` when undefined (no anomaly/normal samples that epoch). Always detached/no-grad →
training byte-identical.

**Implementation**:
```python
# In trainer.py __init__():
if config.teacher_only_warmup_epochs < 0:
    config.teacher_only_warmup_epochs = config.num_epochs // 2  # Auto: 25 for 50 epochs
```

**Pre-warmup anomaly score = recon-only (2026-06-01)**:

The warm-up gate masks *training* (the student decoder/discrepancy/FM are skipped
in `model.forward(teacher_only=True)`), but **evaluation must mask the score too**.
During warm-up the student is frozen / random-initialised, so its output-discrepancy
(`disc`) and feature-matching (`fm`) signals are noise that must NOT enter the
adaptive anomaly score. The eval-side gate:

```python
# mae_anomaly/scoring.py — single source for the window predicate
def is_prewarmup_epoch(config, epoch):  # epoch is 1-indexed (ep = trained+1)
    return epoch is not None and config.teacher_only_warmup_epochs > 0 \
           and epoch <= config.teacher_only_warmup_epochs

# scoring functions take a REQUIRED keyword-only force_recon_only.
# True  → w_disc=0 AND fm_active=False → student_error=0 → score == recon (exact).
# False → legacy full adaptive score (post-warmup, byte-for-byte unchanged).
compute_score(recon, disc, fm, config, force_recon_only=is_prewarmup_epoch(config, ep))

# Evaluator carries the flag per eval:
evaluator.set_eval_context(epoch=ep)   # sets self._force_recon_only
metrics = evaluator.evaluate(...)      # _apply_scoring_formula forwards the flag
```

Zeroing `w_disc` alone is insufficient — the FM term is a separate branch
(`fm_active and fm is not None`), so recon-only requires **both** `w_disc=0` and
`fm_active=False`. The raw per-epoch npz arrays (`teacher_recon_error`,
`discrepancy_error`, `fm_error`) are always saved un-gated, so offline recompute
keeps full information; only the composed `adaptive_score` is gated. See
`docs/POST_MORTEMS/2026-06-01_prewarmup_student_score_leak.md`.

**Epoch Offset (Train Augmentation)**:

When `epoch_offset=True`, each epoch's train window start positions are shifted by a random offset from `[0, stride)`. Offsets are sampled without replacement (non-replacement within each cycle of `stride` epochs), so over `stride` epochs all possible offsets are covered exactly once. Test windows are always fixed at offset=0.

```python
# epoch_offset=True, stride=21:
# Epoch 0: windows at [7, 28, 49, ...]   (random offset 7)
# Epoch 1: windows at [14, 35, 56, ...]  (random offset 14)
# ...
# After 21 epochs: all offsets 0-20 used exactly once → full stride=1 coverage
```

**Official MAE-mode (`official=True`, default-off)**:

A single config flag that switches a SEPARATE code path (default `False` → byte-identical). It lays `CANON_271` (the full canonical 271 config) as the base for anything not explicitly passed, lets the user's `config_override` keys win over it, then FORCES the bundle below (last writer in `make_config` → beats preset/override-string/dataset_def). All new behavior sits behind `if getattr(config, 'official', False)` guards; `apply_official_overrides()` short-circuits with `if not official: return config`.

| # | Forced behavior | Where |
|---|---|---|
| 1 | `epoch_offset=False` + train `stride=1` (the local `train_stride`, which the datasets actually read — not just the config field) | `apply_official_overrides` + run_base local |
| 2 | `num_epochs` default **30** (overridable), `teacher_only_warmup_epochs` default **`num_epochs//2`** (overridable); early-stop off forced. The two are defaults set BEFORE the user-override merge (explicit `num_epochs`/`teacher_only_warmup_epochs` win); NOT forced by `apply_official_overrides` | run_base official build |
| 3 | **Per-iteration LR** (MAE `util/lr_sched.py`): linear warmup 0→peak over `w=teacher_only_warmup_epochs`, half-cosine→`min_lr=0` over `[w, num_epochs)`, `e=epoch+batch/len(loader)`; param-group ratios preserved; per-epoch `scheduler.step()` skipped | `trainer.py _official_lr_now` |
| 4 | Model-only checkpoint EVERY epoch → `official_epochs/epoch_NNN.pt` (separate namespace; ~3.3 GB/dataset at d_model=512). **Keep-option** `official_keep_checkpoints` (global, default True) + `official_ckpt_overrides='k1:false,k2:true'` (per-dataset, unlisted→global): `False` skips `official_epochs/` writes, runs eval+viz, then deletes best/best_checkpoint/latest at the end (→ only metrics+npz+viz remain) | `run_base post_epoch_save_callback` + end-cleanup |
| 5 | `eval_interval=1` (per-experiment local; global `EVAL_INTERVAL` untouched) | `run_base` local |
| 6 | **Causal/online anomaly score** (`scoring.py`, single source): seed `R_tr=Σrecon_tr[norm]`, `D_tr=Σdisc_tr[norm]`; `s_t=(R_tr+cumsum recon)/(D_tr+cumsum disc+ε)`; `score_t=recon_t+0.25·disc_t·s_t`. Prefix-only cumsum ⇒ no future/label use. Per-epoch train-inference yields `R_tr/D_tr`; best epoch picked on this score's `pak_auc_f1`; metrics/VUS/excl22/viz all consistent; npz gains `official_score` (keeps `adaptive_score`) | `scoring.compute_official_causal_score`, `run_base _evaluate_all_parallel` |
| 7 | Thorough single-knob seeding (`set_seed_official`: +PYTHONHASHSEED + DataLoader generator/worker_init_fn; keeps `cudnn.benchmark`, no determinism-algos) | `config.set_seed_official` |

Usage: `--set C --dataset <key> --config-override "official=True <optional deltas>"`. See CHANGELOG (2026-06-22) for the verification record.

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

**Collected diagnostic — anomaly output discrepancy** (`train_anomaly_disc_forward`, 2026-06-20):
The forward/un-margined mean discrepancy on anomaly patches (`anomaly_disc_forward` = mean of
`(teacher_out.detach() − student_out)²` over masked anomaly patches) is **always computed**,
outside the `disable_anomaly_loss` gate. It is now also **logged per-epoch into
`training_histories.json`** (`loss_dict` → `epoch_losses` → `history['train_anomaly_disc_forward']`),
including **GRL/SCAD paths** where the anomaly maximize-loss (`L_anomaly`) is disabled (=0). This is
**collect-only** — a detached scalar that never enters `L_total`/gradients, so training stays
byte-identical. It is the symmetric counterpart to the already-logged normal discrepancy
(`train_normal_loss`) and records the actual anomaly-detection signal during training regardless of
objective. `0.0` sentinel during teacher-only warmup / when discrepancy is off (same convention as `dis`).

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

### GRL ↔ MSE Loss Balancing (`loss_balance_mode`, 2026-06-14)

**Problem**: the GRL classifier BCE loss (O(1–20)) and the self-distillation MSE
discrepancy loss (O(0.5→1e-4)) live on fundamentally different scales; summing
them in one optimizer needs scale-matching (Axis-A). A single mutually-exclusive
enum `loss_balance_mode` selects the strategy:

| mode | mechanism | source |
|---|---|---|
| `adaptive_lambda_legacy` (**default**) | grad-norm ratio λ + prev-epoch carry (current behavior, **byte-identical**) | in-house |
| `fixed` | constant weight (`fixed_grl_weight`<0 → `grl_loss_weight`) | — |
| `mse_norm_dann` | normalize BCE by EMA(\|BCE\|) → O(1) + Ganin λ ramp | Ganin et al. 2016 |
| `relobralo` | softmax of loss ratios + Bernoulli lookback + EMA | Bischof & Kraus, arXiv:2110.09813 |
| `famo` | log-loss simplex balancer (O(1), streaming `w` update) | Liu et al., NeurIPS 2023, arXiv:2306.03792 |
| `uwso` | tempered-softmax over 1/L (closed-form, no learned σ) | Kirchdorfer et al., arXiv:2408.07985 |

**Invariants**: recon is the absolute anchor (never renormalized — only the BCE
term / disc ratio is rescaled, except `famo` which rebalances both tasks); all
weights are stop-gradient; never multiplied with the Ganin reversal ramp
(`model._grl_lambda`); defaults calibrated for 271 (ep250→350, GRL's first ~100
epochs). The 4 new modes require `use_grl=True` + `grl_mode='classifier'` and are
mutually exclusive with `use_scad` (`Trainer.__init__` raises `ValueError`).

**Code**: dispatch + helpers (`_lbm_apply`, `_lbm_{mse_norm_dann,relobralo,uwso,famo}`)
and `_lbm_state_dict`/`_lbm_load_state_dict` in `trainer.py`; 20 fields in
`config.py`; checkpoint `lbm_state` in `run_base_experiments.py`. Full per-mode
parameter reference: Notion subpage "loss_balance_mode — GRL·MSE 손실 균형 6 선택지".

### Feature Matching Loss (Optional)

**Purpose**: Align teacher and student hidden representations on normal patches.

```python
FM_loss = 1 - cosine_similarity(teacher_hidden, student_hidden)  # masked normal patches
```

**Key config**: `use_feature_matching=True`, `fm_distance_metric` (cosine/l2), `fm_adaptive_lambda`, `fm_loss_weight`

Can be combined with or replace output discrepancy (`use_output_discrepancy=False` disables OD, FM-only training).

#### FM ↔ OD Loss Balancing (`fm_balance_mode`, 2026-06-17)

When `fm_adaptive_lambda=True`, FM is excluded from `total_loss` in `loss.py` and re-added by
the trainer with an adaptive weight. By default that weight is the legacy grad-norm ratio
`λ = (‖∇_student OD‖ / ‖∇_student FM‖).clamp(0,10)` — which is already an **OD↔FM** balancer
(teacher reconstruction has zero gradient on the student decoder, so it cancels). `fm_balance_mode`
replaces *how* that FM weight is computed with a value-based multi-task balancer on the **OD↔FM pair
only** (reconstruction stays in `total_loss` at fixed weight 1, never in the balanced pair):

```
total = recon (w=1)  +  OD (anchor)  +  w·FM      # w from the chosen balancer
```

enum `fm_balance_mode` (default `'none'` = byte-identical to exp271; **GRL is NOT affected** — it keeps
its own `loss_balance_mode`):
- `none` — legacy grad-norm-ratio FM λ (prev-epoch lag).
- `relobralo` — ReLoBRaLo loss-ratio softmax (Bischof & Kraus 2110.09813); reuses `relobralo_*`. OD anchored, FM relative weight, clamped [0,10].
- `famo` — FAMO log-loss simplex (Liu et al. NeurIPS 2023, 2306.03792); reweights BOTH OD and FM via `recon + logcomb([OD,FM])` (recon rebuilt from `loss_tensors['reconstruction_loss']`, ordering-independent).
- `uwso` — UW-SO inverse-loss tempered softmax (Kirchdorfer 2408.07985), **MSE↔MSE scale-free variant**: 1/L normalized by its mean before temperature (raw 1/L≈1e3 for MSE≈1e-3 saturates the softmax), `fm_uwso_temperature/loss_floor/ema_beta`, rel clamped [0,10].

`mse_norm_dann` is intentionally NOT offered for FM (its Ganin adversarial ramp is meaningless for two
cooperative MSE losses). Requires `fm_adaptive_lambda=True` + `use_feature_matching=True`, mutually
exclusive with `use_scad`. Balancer runtime state (relobralo EMA + independent RNG, uwso EMA, famo
optimizer) round-trips through the checkpoint `lbm_state` (back-compat with old checkpoints).
Implementation: `trainer.py` `_fm_balance_apply` / `_fm_relobralo` / `_fm_uwso` / `_fm_famo`. Experiments
exp327 (relobralo) / exp328 (famo) / exp329 (uwso).

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

**force_mask_all_anomaly=True** (opt-in, 2026-06-22, exp337; requires force_mask_anomaly + use_masking; default False = byte-identical):
- Removes the "excess remain visible" behavior: masks **ALL** anomaly patches per-sample even above budget, so the encoder never sees an anomaly patch.
- Per-sample masked count `K_s = min(max(budget, n_anomaly_patches), num_patches-1)` (cap keeps ≥1 visible; a fully-anomalous window keeps exactly 1 visible).
- This makes the per-sample masked count **ragged** (variable across the batch), which the MAE visible-only encoder (`mask_after_encoder=True`) normally forbids. Preserved without changing the architecture via **padding + `src_key_padding_mask`**: `_encode_visible_only` pads over-masked samples up to the batch-max visible count and key-pads the extras (encoder ignores them); `_insert_mask_tokens_and_unshuffle` overwrites those padded rows with the mask token before the standard `ids_restore` unshuffle. When the mask is uniform (eval, default, no over-budget sample) both functions take their original path → byte-identical.
- Trade-off: only the per-sample over-budget masking deviates (no batch-wide over-masking of normals); but the masked normal/anomaly partition shifts, so `train_recon_snr`, recon/discrepancy splits, and GRL/SCAD/FM populations legitimately change under this flag (OFF runs unaffected).

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

When `d_model='dynamic'`, the model dimension is auto-selected per-dataset based on `num_features` and `patch_size`:

```python
# resolve_dynamic_d_model(num_features, patch_size)
raw = patch_size * num_features
d_model = min(d for d in [64, 96, 128, 192, 256, 384, 512] if d >= raw)
# Capped at 512; dim_feedforward = 4 * d_model (auto-computed)
```

All candidates are divisible by `nhead=8` (64/8=8, 96/8=12, 128/8=16, …). The
64/96 candidates were added (2026-05-21) so that low-F datasets at small
`patch_size` (e.g., simulation F=8 with `patch_size=5`, raw=40 → d_model=64)
get a tight model size instead of being over-provisioned at 128.

**Note**: For Set C (`patch_size=10`), simulation (F=8, raw=80) now selects
`d_model=96` (was 128); all other supported datasets (F≥19) remain unchanged.

### Consistency Validation (added 2026-05-21)

`make_config()` and `Trainer.__init__` now raise `ValueError` when:
- `seq_length % patch_size != 0` — sequence length must be divisible by patch size
- `seq_length != patch_size * num_patches` — explicit num_patches must match

`SelfDistilledMAEMultivariate.__init__` includes an `assert` for the
divisibility check as a final safeguard on direct instantiation paths.

Set C config: `patch_size=10, patchify_mode='linear', d_model='dynamic'`. See `set_guideline.md` for details.

---

**Status**: ✅ Architecture implemented and tested
