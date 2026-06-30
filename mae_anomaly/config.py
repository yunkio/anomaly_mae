"""
Configuration classes for MAE anomaly detection
"""

import os
import random
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Configuration for MAE anomaly detection experiments"""
    # Data parameters
    seq_length: int = 500
    num_features: int = 8  # Multivariate: 8 features (expanded for sliding window dataset)

    # Sliding window dataset parameters
    use_sliding_window_dataset: bool = True  # Use new sliding window dataset
    sliding_window_total_length: int = 275000  # Total length (220K train + 55K test)
    sliding_window_stride: int = 21  # Stride for train window extraction (overlapping windows)
    # Phase 6 (2026-05-29): sentinel -1 means "auto: resolves to (num_patches - 1)
    # at the call site". Any positive int overrides. The previous fixed default
    # of 21 is preserved when the user explicitly sets it via config-override.
    sliding_window_test_stride: int = -1  # Stride for test window extraction (-1 = num_patches-1)
    epoch_offset: bool = True  # Non-replacement random train window offset each epoch (cycles through [0, stride))
    sliding_window_train_ratio: float = 0.8  # Train ratio (220K/275K = 0.8, test = 55K)
    normalize_mode: str = 'minmax'  # Normalization mode for input signals (exp271 canonical default 2026-05-27)
    # - 'minmax': Per-feature min-max scaling (target range controlled by minmax_range) — default (exp271)
    # - 'zscore': Per-feature z-score standardization (mean=0, std=1)
    minmax_range: str = '0_1'  # Target output range when normalize_mode='minmax'
    # - '0_1' (default): scale train to [0, 1], tight-clip test to [0, 1]
    # - 'neg1_1' (NPSR-style): scale train to [-1, 1], test gets test-only clamp via
    #   minmax_clamp_min/max (default ±4) — preserves test outlier info.
    minmax_clamp_min: float = -4.0  # Test-only clamp lower bound (used only when minmax_range='neg1_1')
    minmax_clamp_max: float = 4.0   # Test-only clamp upper bound (used only when minmax_range='neg1_1')
    anomaly_interval_scale: float = 0.75  # Scale factor for anomaly intervals (2x frequency, ~13% anomaly)

    # Model parameters
    d_model: int = 128
    nhead: int = 8
    num_encoder_layers: int = 2  # enc2 (optimal from ablation study)
    num_teacher_decoder_layers: int = 4  # td4 decoder (optimal from ablation study)
    num_student_decoder_layers: int = 1  # sd1 (shallow student for better discrepancy)
    num_shared_decoder_layers: int = 0  # Shared decoder layers before teacher/student decoders
    # - 0: No shared decoder (default)
    # - >0: Shared decoder trained with teacher, separate mask tokens for student
    decoder_half_dim: bool = False  # [2026-06-15] MAE-style asymmetric decoder width
    # - False: decoder width = encoder d_model (default, byte-identical)
    # - True : decoder(teacher/student/shared)·mask token·output proj·GRL/SCAD head = d_model//2,
    #          decoder dim_feedforward = (d_model//2)*4, nhead reused; encoder stays d_model.
    #          A decoder_embed Linear(d_model→d_model//2) narrows the latent. Works with dynamic
    #          d_model (always //2 of resolved). Requires mask_after_encoder + transformer-enc-dec.
    dim_feedforward: int = 512  # 4 * d_model
    dropout: float = 0.15
    masking_ratio: float = 0.15
    masking_ratio_min: float = -1.0  # Random masking range min (-1 = disabled, use fixed masking_ratio)
    masking_ratio_max: float = -1.0  # Random masking range max (-1 = disabled)
    # When both >= 0: sample uniform(min, max) per batch. Overrides masking_ratio during training.
    num_patches: int = 100  # seq_length / patch_size (dynamically computed when window size changes)
    patch_size: int = 5  # Fixed patch size; num_patches = seq_length / patch_size
    patchify_mode: str = 'patch_cnn'  # 'patch_cnn', 'linear'
    # - 'patch_cnn': Patchify first, then CNN per patch (no cross-patch leakage)
    # - 'linear': Patchify then linear embedding (MAE original style, no CNN)
    mask_after_encoder: bool = True  # Standard MAE masking architecture
    # - False: Mask tokens go through encoder (current behavior)
    # - True: Encode visible patches only, insert mask tokens before decoder (standard MAE)
    masking_strategy: str = 'patch'  # 'patch' (existing), 'feature_wise' (training-only ablation)
    # - 'patch': mask full temporal patch tokens exactly as before.
    # - 'feature_wise': during training, mask patch-feature cells in the raw input before
    #   embedding and compute loss only on those masked feature cells. Evaluation still uses
    #   the existing patch leave-one-out masks, so official scoring/inference is unchanged.
    shared_mask_token: bool = False  # Share mask token between teacher and student
    # - True: Single mask token shared (current behavior)
    # - False: Separate mask tokens for teacher and student decoders

    # MAE architecture variants (ablation study options)
    use_transformer_encoder_decoder: bool = True  # Use TransformerEncoder for decoder (MAE-style)
    # - True: TransformerEncoder (self-attention only, no cross-attention with encoder output)
    # - False: TransformerDecoder (cross-attention with encoder output via memory)
    use_flatten_linear_embedding: bool = True  # Use flatten+linear for patch embedding
    # - True: Flatten patch then linear projection (preserves patch structure)
    # - False: Mean pooling over patch dimension (simple averaging)

    # CNN architecture for patch_cnn mode
    cnn_channels: tuple = None  # (mid_channels, out_channels) for patch_cnn
    # - None: Auto-scale with d_model (d_model//2, d_model) — recommended
    # - (64, 128): Fixed CNN channels (only correct when d_model=128)
    cnn_kernel_size: int = 3  # Kernel size for patch_cnn Conv1d layers
    # - 3: Default (receptive field = 3 per layer)
    # - 5: Wider receptive field (better for larger patch_size)

    # RevIN (Reversible Instance Normalization, Kim et al. ICLR'22) — per-window normalization
    # Applied INSIDE model.forward(): normalize at input → encoder/decoder operate in normalized space → denormalize at output.
    # Use with normalize_mode='zscore' (loader-level standardization) for upstream-consistent behavior
    # (matches PatchTST / ModernTCN / CATCH / TimesNet pattern).
    use_revin: bool = False
    revin_affine: bool = True  # Learnable per-feature γ (weight) and β (bias)
    revin_eps: float = 1e-5  # Variance floor for numerical stability
    revin_visible_only: bool = False  # Stats from non-anomaly timesteps during training
    # - False (default): mean/std over all timesteps in window (matches all 4 baselines)
    # - True: exclude anomaly timesteps using point_labels (training only;
    #         inference uses full window since point_labels not available)

    # Loss parameters
    margin: float = 0.5
    lambda_disc: float = 2.0
    margin_type: str = 'dynamic'  # 'hinge' (relu), 'softplus', 'dynamic', 'none'
    dynamic_margin_k: float = 2.0  # k for dynamic margin (mu + k*sigma)
    patch_level_loss: bool = True  # True=patch-level, False=window-level discrepancy loss

    # Discrepancy loss parameters
    anomaly_loss_weight: float = 2.0  # Weight multiplier for anomaly discrepancy loss
    anomaly_loss_direction: str = 'maximize'  # Anomaly discrepancy loss direction
    # - 'maximize': Push anomaly disc UP (default, relu(margin - disc))
    # - 'minimize': Push anomaly disc DOWN (same direction as normal, for Exp 170)
    # - 1.0: Default (equal weight)
    # - 2.0/3.0/5.0: Stronger interference on anomaly samples
    normal_loss_weight: float = 1.0  # Weight multiplier for normal discrepancy loss
    # - 1.0: Default
    # - 3.0+: Stronger suppression of normal disc → lower normal baseline → better disc SNR
    student_recon_weight: float = 0.0  # [NOT YET IMPLEMENTED] Weight for student direct reconstruction loss
    # - 0.0: Disabled (default)
    # - 0.5+: Student directly learns to reconstruct normal input (planned, not yet in loss.py)

    # Anomaly score computation mode
    anomaly_score_mode: str = 'adaptive'
    # - 'default': recon + lambda_disc * disc (original)
    # - 'adaptive': Auto-scaled lambda with individual normalization (recon + student_error)
    # - 'ratio_weighted': recon * (1 + disc / median_disc)

    # GRL (Gradient Reversal Layer) parameters
    grl_disable_anomaly_loss: bool = True  # Disable anomaly_loss when GRL is active
    # - True: anomaly_loss=0 when use_grl=True (default, current behavior)
    # - False: anomaly_loss + GRL simultaneous (adversarial equilibrium)
    use_grl: bool = True  # Enable GRL for anomaly-aware student training (exp271 canonical default 2026-05-27)
    # - True: GRL classifier on student hidden → anomaly_loss disabled, GRL handles disc generation — default (exp271)
    # - False: No GRL
    grl_cls_hidden: int = 0  # Classifier hidden dim (0 = auto: d_model // 2)
    grl_loss_weight: float = 0.2  # GRL loss weight multiplier applied to adaptive lambda (exp271 canonical default 2026-05-27)
    grl_target_mode: str = 'window'  # GRL classifier target granularity (exp271 canonical default 2026-05-27)
    # - 'window': Target = has_anomaly_in_window (window-level label, all patches same) — default (exp271)
    # - 'patch': Target = patch_has_anomaly (per-patch label)
    grl_pos_weight: float = 19.0  # GRL classifier pos_weight for class imbalance
    # - >0: Fixed value (default 19.0 ≈ 95%/5% normal/anomaly ratio)
    # - Automatically set from actual dataset anomaly ratio by run_base_experiments.py
    # - Ignored when grl_balanced_sampling=True
    grl_balanced_sampling: bool = False  # Balanced downsampling for GRL classifier
    # - False: Use all masked patches with pos_weight (default, existing behavior)
    # - True: Downsample normal patches to match anomaly count → 1:1 balanced loss
    #   pos_weight is ignored. Prevents class-imbalance-driven classifier collapse.
    grl_mode: str = 'classifier'  # GRL adversarial training mode
    # - 'classifier': Binary classifier + GRL gradient reversal (DANN-style, default)
    # - 'wdgrl': Wasserstein critic with gradient penalty (WDGRL, Shen et al. 2018)
    #   No GRL needed — minimax via separate critic optimizer.
    #   More stable than classifier mode: no saturation, no class-imbalance collapse.
    grl_use_focal: bool = True  # Use focal loss for GRL classifier
    # - True: Focal loss (default, existing behavior) — down-weights easy examples
    # - False: Standard BCE loss — removes focal+pos_weight non-standard interaction
    grl_cls_lr_ratio: float = 0.1  # GRL classifier LR as fraction of main LR (exp271 canonical default 2026-05-27)
    # - 0.1: Slower classifier convergence to prevent collapse — default (exp271)
    # - 1.0: Same LR as main model
    #   Only applies to classifier mode (grl_mode='classifier')
    grl_cls_arch: str = 'default'  # GRL classifier architecture
    # - 'default': 1-layer MLP (LayerNorm → d_model//2 → GELU → Dropout(0.1) → 1)
    # - '2layer': 2-layer MLP (LayerNorm → hidden → GELU → Drop(0.2) → hidden//2 → GELU → Drop(0.2) → 1)
    # - 'dann': DANN-style 3-layer (d_model → d_model*2 → ReLU → Dropout(0.5) → d_model*2 → ReLU → Dropout(0.5) → 1)
    grl_cls_hidden: int = 0  # GRL classifier hidden dimension (0 = auto: d_model//2 for default, d_model for 2layer)
    grl_attach_layer: str = 'last'  # [2026-06-15] Where the GRL classifier reads from
    # - 'last' : student decoder FINAL hidden (h2), just before output proj (default, byte-identical)
    # - 'first': student decoder FIRST-layer output (h1) — invariance on an intermediate rep, the
    #            final layer specializes for reconstruction. Falls back to 'last' if student depth ≤ 1.
    grl_adaptive_lambda: bool = True  # Adaptive λ for GRL gradient balancing
    # - True: Auto-computed λ to balance GRL vs main gradients (default, VQGAN-style)
    # - False: Fixed weight — loss += grl_loss_weight * grl_cls_loss (no adaptive scaling)
    wdgrl_k_critic: int = 5  # Critic update steps per main update (WDGRL only)
    wdgrl_gp_weight: float = 10.0  # Gradient penalty weight (WDGRL only)
    wdgrl_critic_lr: float = 1e-4  # Critic optimizer learning rate (WDGRL only)

    # ── loss_balance_mode (2026-06-14): how the GRL classifier BCE loss is scale-matched
    #    against the teacher-student MSE discrepancy loss (Axis-A scale matching, NOT
    #    gradient-direction conflict). SINGLE enum → exactly one active (mutually exclusive,
    #    incl. legacy adaptive_lambda). DEFAULT 'adaptive_lambda_legacy' is byte-identical
    #    to the current behavior; new modes are reachable ONLY by explicit config.
    #    Only intercepts grl_mode='classifier'; WDGRL/SCAD/FM paths are untouched.
    loss_balance_mode: str = 'adaptive_lambda_legacy'
    #   ∈ {adaptive_lambda_legacy, fixed, mse_norm_dann, relobralo, famo, uwso}
    #   - adaptive_lambda_legacy: current dispatch (grl_adaptive_lambda True→VQGAN-ratio / False→fixed)
    #   - fixed: loss += (fixed_grl_weight if >=0 else grl_loss_weight) * grl_cls_loss  (no scaling)
    #   - mse_norm_dann: BCE/EMA(|BCE|) magnitude-match + Ganin deterministic ramp (runaway-proof)
    #   - relobralo: Bischof&Kraus 2110.09813 — loss-ratio softmax + random lookback + EMA
    #   - famo: Liu et al. NeurIPS2023 2306.03792 — log-loss simplex (O(1), needs post-step re-forward)
    #   - uwso: Kirchdorfer et al. 2408.07985 (IJCV2025) — analytic tempered-softmax inverse-loss
    # mse_norm_dann params (calibration: 271, GRL active ep250-500, prioritize ep250-350 ≈ 100ep window)
    mse_norm_ema_beta: float = 0.1     # EMA new-fraction for running |loss| scale (≈10-step memory)
    mse_norm_eps: float = 1e-8         # divide-by-zero guard
    mse_norm_log_variant: bool = False # True: weight log(BCE) instead of BCE/EMA (for >1000x swings)
    dann_ramp_gamma: float = 10.0      # Ganin logistic ramp steepness (λ_p=2/(1+exp(-γp))-1)
    dann_ramp_horizon: int = 100       # ramp window in epochs: p=clamp((epoch-warmup)/horizon,0,1)
    # relobralo params (Bischof&Kraus; α/ρ recalibrated 0.999→0.99 for the ~100-epoch GRL window)
    relobralo_T: float = 1.0           # softmax temperature
    relobralo_alpha: float = 0.99      # EMA of historical weights
    relobralo_rho: float = 0.99        # Bernoulli prob of keeping prev weight vs onset re-anchor
    relobralo_eps: float = 1e-12       # ratio div guard
    relobralo_update_freq: str = 'epoch'  # 'epoch' (state update once/epoch) | 'step' (per batch)
    # famo params (official repo Cranial-XIX/FAMO defaults)
    famo_gamma: float = 0.01           # weight-decay on logits w (repo famo.py default)
    famo_w_lr: float = 0.025           # Adam lr for the logit optimizer
    famo_max_norm: float = 0.0         # 0 = reuse trainer's grad clip (avoid double-clip)
    famo_reforward: bool = True        # True: post-step same-batch re-forward (faithful); False: next-batch approx
    # uwso params (Kirchdorfer et al. Eq.4; T mid of NYUv2~2-3 / Cityscapes~20-48 for BCE-vs-MSE gap)
    uwso_temperature: float = 4.0      # tempered-softmax temperature on (1/L)
    uwso_loss_floor_mse: float = 1e-3  # cap 1/L_mse blow-up as MSE→1e-4
    uwso_loss_floor_bce: float = 1e-2  # symmetric floor for BCE
    uwso_ema_beta: float = 0.9         # EMA new-fraction for loss smoothing (1.0 = paper-faithful, off)
    # fixed mode param
    fixed_grl_weight: float = -1.0     # <0 sentinel → use grl_loss_weight; else this explicit weight

    # Feature Matching Loss (independent of GRL)
    use_feature_matching: bool = True  # Enable feature matching loss (exp271 canonical default 2026-05-27)
    # cosine(teacher_hidden, student_hidden) on masked normal patches
    fm_adaptive_lambda: bool = True  # Adaptive λ for FM-OD gradient balancing (exp271 canonical default 2026-05-27)
    # - True: FM weight = adaptive_lambda * fm_loss_weight (gradient-balanced) — default (exp271)
    # - False: FM weight = fm_loss_weight (fixed, 1:1)
    fm_distance_metric: str = 'l2'  # Distance metric for feature matching (exp271 canonical default 2026-05-27)
    # - 'l2': L2 distance (magnitude + direction) — default (exp271)
    # - 'cosine': 1 - cosine_similarity (direction-only)
    use_output_discrepancy: bool = True  # Enable output discrepancy loss (normal_loss + anomaly_loss)
    # - True: OD loss active (default, existing behavior)
    # - False: OD loss zeroed, only FM loss contributes to discrepancy_loss
    #   Discrepancy values are still computed for metrics/logging, but don't affect training.
    #   At inference, disc component is excluded from scoring (w_disc=0).
    fm_loss_weight: float = 1.0  # FM:disc training weight ratio (1.0 = equal)

    # ── fm_balance_mode (2026-06-17): replace FM's adaptive-λ COMPUTATION with a
    #    multi-task loss balancer applied to the OD↔FM pair ONLY (reconstruction is
    #    NOT in the balanced pair — it stays in total_loss at fixed weight 1, exactly
    #    as exp271). This is the value-based analogue of the legacy FM adaptive λ,
    #    which is already an OD↔FM balancer (‖∇_student OD‖/‖∇_student FM‖; teacher
    #    recon has zero gradient on the student decoder, so it cancels — see line 218).
    #    DEFAULT 'none' is byte-identical to exp271. GRL is NOT touched here — it keeps
    #    loss_balance_mode (271='adaptive_lambda_legacy'). Requires fm_adaptive_lambda=True
    #    + use_feature_matching=True (FM excluded from loss.py total, added in trainer).
    #    NOTE: mse_norm_dann is intentionally NOT offered for FM — its defining Ganin
    #    adversarial ramp has no meaning for two cooperative MSE distillation losses.
    fm_balance_mode: str = 'none'
    #   ∈ {none, relobralo, famo, uwso}
    #   - none: legacy — FM λ = (‖∇_student(OD)‖/‖∇_student FM‖).clamp(0,10), prev-epoch lag
    #   - relobralo: OD↔FM loss-ratio softmax (reuses relobralo_* params; ratio-based, scale-agnostic)
    #   - famo: OD↔FM log-loss simplex (reuses famo_* params; reweights BOTH OD and FM)
    #   - uwso: OD↔FM tempered-softmax inverse-loss (fm_uwso_* params; MSE↔MSE re-tuned)
    # fm_uwso params — re-tuned from the GRL uwso defaults for MSE↔MSE. Because OD and FM
    #   are both tiny MSE-like losses (~1e-3), the raw 1/L (~1e3) saturates the softmax, so
    #   _fm_uwso normalizes 1/L by its MEAN before the temperature → T is SCALE-FREE (acts on
    #   the loss ratio only). ONE shared MSE floor for both terms; rel clamped to [0,10].
    fm_uwso_temperature: float = 1.0   # scale-free temp on mean-normalized 1/L (lower = sharper)
    fm_uwso_loss_floor: float = 1e-4   # shared 1/L floor for BOTH OD and FM (both MSE-like)
    fm_uwso_ema_beta: float = 0.9      # EMA new-fraction for loss smoothing (matches GRL uwso)

    # --- SCAD (Supervised Contrastive Anomaly Discrimination) parameters ---
    use_scad: bool = False  # Enable SCAD loss for direct anomaly discrimination
    # - False: SCAD disabled (default, GRL or no anomaly-aware loss used)
    # - True: SCAD active on student hidden → replaces GRL classifier
    # Mutual exclusion: use_scad ⊕ use_grl (enforced in trainer.py validation)

    scad_form: str = 'A'  # SCAD loss variant
    # - 'A': Free-energy log-sum-exp (default, smooth hard negative mining)
    # - 'B': Margin hinge (BGAD-spirit, semi-push outside margin only)
    # - 'C': One-sided thresholded negative repulsion (2026-06-15) — U detached,
    #        push anchors to cos ≤ gamma. C(one_sided=False, gamma=-margin) ≡ B(margin).

    scad_d_proj: int = 128  # Projection head output dim (SimCLR convention)
    scad_temperature: float = 0.1  # Temperature for log-sum-exp (Form A only)
    scad_margin: float = 0.3  # Cosine margin for hinge (Form B only)
    scad_gamma: float = 0.0  # Form C threshold: penalize cos(z_a, z_u) > gamma (default 0 = decorrelate)
    scad_one_sided: bool = True  # Form C: detach U (negatives) so only anomaly anchors move

    scad_loss_weight: float = 0.5  # SCAD base multiplier × adaptive λ × ramp
    # Smaller than fm_loss_weight (1.0) since SCAD is new; larger than
    # grl_loss_weight (0.2) since direct optimization vs adversarial

    scad_adaptive_lambda: bool = True  # VQGAN-style gradient balancing
    scad_ramp_up: str = 'sigmoid'  # 'sigmoid' (Ganin) | 'linear' | 'none'

    scad_patch_label_mode: str = 'patch'  # 'patch' (recommended) | 'window'
    # - 'patch': y_p = 1 iff patch p contains ≥1 anomaly timestep (precise)
    # - 'window': window-mode (legacy, same as GRL's noisy mode)

    scad_use_memory_bank: bool = False  # Optional: anomaly memory queue
    scad_memory_bank_size: int = 1024  # FIFO queue size for anomaly hidden

    scad_proj_head_arch: str = 'default'  # SCAD projection head architecture
    # - 'default': 2-layer MLP (LN → Linear → GELU → Linear) — SimCLR convention
    # - 'linear': 1-layer (LN → Linear) — mitigates absorption issue
    # - 'deep': 3-layer MLP — more expressive (absorption risk)

    scad_apply_space: str = 'projection'  # Where the SCAD repulsion is measured/applied
    # - 'projection' (default): cos repulsion on z = scad_head(student_hidden) — current SCAD-C
    # - 'hidden_final': H-SCAD-C — cos repulsion on h = L2norm(LayerNorm(student_hidden))
    #   DIRECTLY on the final student decoder hidden. No projection head (0 new params),
    #   parameter-free LN (removes per-token mean/DC). Better aligned with the score path
    #   (student_hidden → student_output_projection → output discrepancy) than projection z,
    #   so it avoids projection-only absorption. Same Form-C math & 6 scad_c_* metrics.

    # Inference scoring weight override (-1 = use training weight)
    eval_disc_weight: float = -1.0  # Disc weight at inference (-1 → 1.0)
    eval_fm_weight: float = -1.0    # FM weight at inference (-1 → fm_loss_weight) — UNUSED in score since 2026-06-01

    # 2026-06-01 lambda change: adaptive anomaly score uses teacher recon :
    # output-discrepancy = score_recon_disc_ratio : 1 (default 4:1), with the
    # recon-scale correction kept. FM is NOT part of the anomaly score (it stays
    # a training loss only). See mae_anomaly/scoring.compute_adaptive_components.
    score_recon_disc_ratio: float = 4.0

    # Inference: Complementary Masking
    eval_complementary_masking: bool = False  # Use K-group complementary masking at inference
    # - False: Leave-one-out masking (default, existing behavior)
    # - True: Partition patches into K non-overlapping groups (~15% each), 7 forward passes
    eval_complementary_k: int = 7  # Number of complementary groups (100/7 ≈ 14-15 patches per group)

    # Teacher Freeze
    freeze_teacher_after_warmup: bool = False  # Freeze encoder/teacher after warmup (method C: eval + no_grad)
    # - False: All parameters trainable throughout (default)
    # - True: Freeze at teacher_only_warmup_epochs (forced to num_epochs // 2)
    freeze_encoder_only: bool = False  # Freeze encoder only (teacher/student decoders keep training)
    # - False: No effect (default)
    # - True: Freeze encoder+patch_embed at warmup. Decoders, output projections, mask tokens stay trainable.
    #   Mutually exclusive with freeze_teacher_after_warmup.

    # Training: Masking Ratio Annealing
    masking_ratio_anneal: bool = False  # Linear anneal masking_ratio after warmup
    # - False: Fixed masking_ratio throughout (default)
    # - True: After warmup, linearly decrease from masking_ratio to 1/num_patches (1 patch)

    # Discriminator (Adversarial Realism) parameters
    use_discriminator: bool = False  # Enable adversarial discriminator for student decoder
    # - False: No discriminator (default, existing behavior unchanged)
    # - True: 1D CNN discriminator judges student output realism
    d_grad_student_layers: str = 'all'  # [NOT YET IMPLEMENTED] Adversarial gradient propagation scope
    # - 'all': Adversarial gradient flows through entire student decoder (current behavior always)
    # - 'last': Only last transformer block + output projection (planned, not yet implemented)
    disc_lr_ratio: float = 4.0  # D learning rate = main_lr * disc_lr_ratio (TTUR)
    adaptive_lambda: bool = True  # Adaptive λ via gradient magnitude balancing (VQGAN-style)
    # - True: λ_adv auto-computed per step to balance discrepancy vs adversarial gradients
    # - False: Fixed λ_adv = 1.0
    adv_loss_weight: float = 1.0  # Adversarial loss weight multiplier
    # - 1.0: Default (discrepancy:adversarial = 1:1 after adaptive λ balancing)
    # - 0.5/0.2/0.1: Reduce adversarial influence (discrepancy:adversarial = 1:x)
    disc_warmup_epochs: int = 10  # Epoch to start D training (matches LR warmup)
    disc_channels: tuple = (64, 32)  # Discriminator 1D CNN channels (c1, c2)

    # Training parameters
    batch_size: int = 256  # Batch size for training
    num_epochs: int = 50
    eval_interval: int = -1  # Per-epoch eval frequency. -1 = auto (1 official / 5 else; TEP_typegen → 3 in run_base).
    # >0 → eval every N epochs (+ always final epoch); explicit override always wins. (single def; dup below removed 2026-06-23)
    learning_rate: float = 1e-3  # Default learning rate (halved from 2e-3 for training stability)
    weight_decay: float = 1e-3
    warmup_epochs: int = 10
    teacher_only_warmup_epochs: int = -1  # First N epochs train teacher only (-1 = auto: num_epochs // 2)

    # === [신규 2026-06-01] Teacher output EMA ===
    # student의 output discrepancy 표적을 live teacher 대신 weight-EMA로 평탄화한 teacher 출력으로 교체해
    # student가 매 epoch 움직이는 teacher를 쫓을 때의 불안정성을 완화. 활성 조건: post-warmup(epoch>warmup)
    # AND freeze_teacher_after_warmup=False (teacher가 계속 학습되는 경우)에만. 적용 채널: output discrepancy만
    # (feature matching 제외). EMA 대상 모듈: teacher_decoder + teacher_output_projection (+ teacher_mask_token);
    # encoder 제외. 추론/anomaly score(scoring.py)·dynamic_margin은 live teacher 사용(EMA는 student 학습 표적 전용).
    # 기본 False → 271 등 기존 동작 불변.
    use_teacher_output_ema: bool = False
    teacher_output_ema_momentum: float = 0.996  # step별 EMA: θ_ema ← m·θ_ema + (1−m)·θ_teacher. 유효 평균창 ≈ 1/(1−m) step.

    # === [신규 2026-06-01] Teacher warmup early-stop (recon_snr 기반 동적 단축) ===
    # warmup 길이를 고정값이 아니라 teacher의 (학습 데이터 대상) recon_snr plateau에 따라 동적으로 조기 종료.
    # warmup 중 매 epoch train recon_snr을 추적하여 patience epoch 동안 최고치 갱신이 없으면(strict) 거기서
    # warmup을 종료하고, model+optimizer를 recon_snr 최고 epoch 시점으로 revert한 뒤 student 학습을 시작한다.
    # teacher_only_warmup_epochs는 이때 상한(upper bound)으로 작동(plateau가 안 오면 기존처럼 그 값에서 종료).
    # 기본 False → 기존 고정 warmup 동작 그대로.
    use_teacher_warmup_early_stop: bool = False
    teacher_warmup_early_stop_patience: int = 10    # recon_snr 무갱신 epoch 수 → 트리거
    teacher_warmup_early_stop_min_epochs: int = 50  # 이 epoch 이전엔 트리거 금지(최소 warmup floor)
    teacher_warmup_early_stop_metric: str = 'recon_snr'  # 'recon_snr' | 'train_loss'
    # --- metric='train_loss' (peak_reversal, 2026-06-20 spec pseudo-code) 전용 상수 ---
    # 매 check_interval epoch마다(min_epoch 이상) epoch train_loss를 확인: 최저점 대비
    # relative_threshold(=1%) 초과 악화 check가 patience_checks회 연속 누적되면 종료하고
    # train_loss가 가장 낮았던 check epoch으로 model+optimizer+scheduler rollback.
    teacher_warmup_es_check_interval: int = 5       # train_loss 확인 주기(epoch)
    teacher_warmup_es_patience_checks: int = 2      # 연속 reversal check 수 → 트리거(≈10ep @int=5)
    teacher_warmup_es_relative_threshold: float = 0.01  # best-low 대비 상대 악화 임계(1%)
    teacher_warmup_es_min_epoch: int = 20           # 이 epoch 이전엔 트리거 금지(train_loss 모드)

    best_epoch_metric: str = 'pak_auc_f1'  # Metric for best epoch selection
    # - 'pak_auc_f1': PA%K AUC of F1 with per-K threshold re-optimization (best_f1_w_pa, recommended)
    #   Per-K threshold sweep after PA%K segment adjustment (Kim et al., AAAI 2022 tadpak method)
    # - 'pak_auc_f1_raw': PA%K AUC of F1 with fixed threshold (raw_f1_w_pa, legacy comparison)
    # - 'prc_auc': Precision-Recall AUC (legacy default)
    # (2026-06-23) eval_interval은 위 Training-parameters 그룹에 단일 정의(default -1=auto). 여기 있던
    # 중복 정의(=5)가 그걸 가려 default가 5로 깨져 있던 것을 제거 — 이제 -1(auto: official=1 / else 5; TEP=3).
    use_amp: bool = True  # Mixed Precision Training (Automatic Mixed Precision)
    # - True: Use float16 for forward pass, float32 for loss/gradients (faster on Tensor Core GPUs)
    # - False: Use float32 everywhere (more stable, required for older GPUs)
    amp_dtype: str = 'bf16'  # AMP compute dtype. One of 'fp16' | 'bf16'.
    # Default changed 2026-05-27: 'fp16' -> 'bf16' (RTX 30/40 / A100 / H100).
    # bf16 has the fp32 exponent range -> safer for SCAD logsumexp / focal exp(-bce);
    # no GradScaler needed. Requires CUDA capability >= 8.0; trainer.py raises
    # RuntimeError on older GPUs (silent fallback intentionally avoided).
    # To reproduce pre-flip fp16 results: pass amp_dtype='fp16' via config_override
    # or Config(amp_dtype='fp16').

    # Ablation flags
    use_discrepancy_loss: bool = True
    use_teacher: bool = True
    use_student: bool = True
    use_masking: bool = True
    force_mask_anomaly: bool = True  # Prioritize masking anomaly patches during training
    force_mask_all_anomaly: bool = False  # [exp337] Force-mask ALL anomaly patches per-sample,
    # even above the fixed budget, so the encoder never sees an anomaly patch. Requires
    # force_mask_anomaly=True AND use_masking=True. Default False = byte-identical to exp271.
    # Per-sample masked count = min(max(budget, n_anomaly_patches), num_patches-1) → produces a
    # RAGGED mask (variable count per sample), handled by _encode_visible_only's padding +
    # src_key_padding_mask path so mask_after_encoder=True is preserved. A fully-anomalous window
    # keeps exactly 1 visible patch (encoder needs >=1).
    # - True: Anomaly patches are masked first within fixed masking budget (masking_ratio).
    #   If anomaly patches exceed the budget, excess remain visible as encoder context.
    #   Masking count is always exactly round(num_patches * masking_ratio) per sample.
    # - False: Random masking only (no anomaly-aware prioritization)
    blind_train_labels: bool = False  # [label-blind control] Zero ALL TRAIN-split point
    # labels at the dataset boundary so every downstream consumer (force_mask_anomaly,
    # GRL classifier, anomaly_loss, dynamic-margin normal-selection) sees a label-free
    # train stream. Test labels + anomaly_regions (eval) are UNTOUCHED. Used for the TEP
    # type-gen condition B (matched label-blind control). Default False = unchanged.
    train_label_mask_frac: float = 0.0  # [label-mask, 2026-06-24; rank-fix 2026-06-30] unlabel
    # the chronologically-LAST `frac` of the TRAIN ANOMALY timepoints (by rank among the train
    # anomaly points), NOT the back frac of the timeline. 0.0=off (no-op, byte-identical),
    # 1.0=all train anomaly labels zeroed (≡ blind_train_labels=True), 0.5=latest 50% of the
    # train anomaly timepoints unlabeled. (Old position-based semantics wiped ~all anomalies at
    # the smallest frac because these datasets pack every train anomaly into the back few % of
    # the timeline; rank-based gives the intended graded sweep.) Applied ONLY to the training
    # train_dataset (via .copy(), so the shared array / test split / best_epoch_train_scores keep
    # TRUE labels). Normal labels are never touched. Test labels + eval UNTOUCHED.
    train_exclude_anomaly_segments: bool = False  # [exclude-anomaly 2026-06-24] REMOVE anomaly-
    # labeled timesteps from the TRAIN signal entirely (not mask — splice out), inserting run
    # boundaries at the splice junctions so windows never span a removed gap (existing boundaries
    # preserved). Applied ONLY to the training train_dataset (fancy-index copy → shared array /
    # test split / best_epoch_train_scores UNTOUCHED; normalization scaler unchanged). Default
    # False = no-op, byte-identical.

    # Reproducibility
    random_seed: int = 42
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── Official MAE-mode override bundle (2026-06-22) ────────────────────────
    # When official=True a SEPARATE code path applies a fixed bundle that
    # OVERRIDES any conflicting config (see apply_official_overrides + the
    # `getattr(config,'official',False)` guards in trainer/scoring/run_base).
    # Default False ⇒ byte-identical to today. min_lr is the cosine-decay floor
    # used only by the official per-iteration LR schedule (matches the legacy
    # CosineAnnealingLR eta_min=0, so it is inert when official=False).
    official: bool = False
    min_lr: float = 0.0
    # [official] Per-epoch checkpoint persistence (official mode only).
    # official_keep_checkpoints: GLOBAL default. True = keep official_epochs/ (every
    # epoch) + the run's best/last checkpoints (current behavior). False ('저장 안함')
    # = SKIP official_epochs/ writes, still run eval + visualization normally, then
    # DELETE best_model/best/latest checkpoints at the end (end state: only metrics,
    # epoch_scores npz, and visualizations remain). official_ckpt_overrides: per-
    # dataset override as 'key1:false,key2:true' — overrides the global for those
    # dataset keys only (anything unlisted uses official_keep_checkpoints).
    official_keep_checkpoints: bool = True
    official_ckpt_overrides: str = ''


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
    torch.backends.cudnn.benchmark = True  # Auto-tune convolution algorithms for speed


# =============================================================================
# Official MAE-mode (2026-06-22)
# =============================================================================
# CANON_271 = the COMPLETE effective config of the canonical Exp 271
# (`271canon_baseline`). It is the single source of truth for "when official=True,
# every config not explicitly overridden defaults to 271". Derived verbatim from
# exp271's queue config_override string + the Set C geometry (seq=500/p=10/np=50)
# that 271 inherits from the preset. The official run path lays this down as the
# BASE, lets the user's explicit per-experiment overrides win over it, then
# apply_official_overrides() FORCES the official bundle on top. Keep in sync with
# configs/queue_dedup_renumbered_v6.json exp271 if 271 ever changes.
CANON_271 = {
    'seq_length': 500, 'patch_size': 10, 'num_patches': 50,
    'num_encoder_layers': 4, 'num_teacher_decoder_layers': 3,
    'num_student_decoder_layers': 2, 'dim_feedforward': 2048,
    'patchify_mode': 'linear', 'cnn_kernel_size': 3, 'd_model': 512, 'nhead': 8,
    'mask_after_encoder': True, 'num_epochs': 500,
    'teacher_only_warmup_epochs': 250, 'epoch_offset': True,
    'anomaly_score_mode': 'adaptive', 'normalize_mode': 'minmax',
    'minmax_range': '0_1', 'minmax_clamp_min': -4.0, 'minmax_clamp_max': 4.0,
    'use_feature_matching': True, 'fm_loss_weight': 1.0, 'fm_distance_metric': 'l2',
    'fm_adaptive_lambda': True, 'use_grl': True, 'grl_loss_weight': 0.2,
    'grl_target_mode': 'window', 'grl_cls_lr_ratio': 0.1,
    'grl_balanced_sampling': False, 'grl_use_focal': True, 'use_scad': False,
    'use_output_discrepancy': True, 'dynamic_margin_k': 6, 'anomaly_loss_weight': 2.0,
    'lambda_disc': 2.0, 'eval_disc_weight': -1.0, 'eval_fm_weight': -1.0,
    'force_mask_anomaly': True, 'amp_dtype': 'bf16', 'batch_size': 1024,
    'learning_rate': 0.001, 'sliding_window_test_stride': -1,
}

# The OFFICIAL bundle. apply_official_overrides FORCES the truly-fixed items
# (epoch_offset off, train stride 1, early-stop off). num_epochs (default 30) and
# teacher_only_warmup_epochs (default num_epochs//2) are OVERRIDABLE — they are set
# as defaults in run_base's official overrides build BEFORE the user-override merge
# (so a user's explicit num_epochs / teacher_only_warmup_epochs wins), and are NOT
# forced here. The per-iteration LR warmup horizon w = teacher_only_warmup_epochs;
# cosine decays over [w, num_epochs). Other official behaviors (per-iteration LR,
# every-epoch save, eval_interval=1, causal score, thorough seeding) are read off
# `config.official` directly in the trainer / run_base / scoring subsystems.
def apply_official_overrides(config: 'Config') -> 'Config':
    """Force the truly-fixed official items onto `config`. No-op (byte-identical)
    when config.official is falsy. MUST be the LAST writer so these win over
    preset / override-string / dataset_def / CANON_271. (num_epochs and
    teacher_only_warmup_epochs are deliberately NOT forced here — they are
    user-overridable defaults applied in run_base before the user merge.)"""
    if not getattr(config, 'official', False):
        return config
    config.epoch_offset = False              # (1) no train epoch-offset
    config.sliding_window_stride = 1         # (1) train stride = 1 (display/field; run_base also forces the local)
    config.use_teacher_warmup_early_stop = False  # fixed-warmup mode ⇒ disable runtime shortening
    return config


def set_seed_official(seed: int) -> 'torch.Generator':
    """Thorough, SINGLE-knob reproducible seeding for official mode WITHOUT the
    determinism slowdown. Everything derives from `seed`, so changing this one
    value changes every RNG stream. Keeps cudnn.benchmark=True and does NOT call
    torch.use_deterministic_algorithms / cudnn.deterministic (speed preserved).
    Returns a torch.Generator (seeded from `seed`) for the train DataLoader."""
    set_seed(seed)  # random / numpy / torch / cuda(+all); cudnn flags unchanged (benchmark on)
    os.environ['PYTHONHASHSEED'] = str(seed)  # best-effort (full effect needs launcher export)
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def official_worker_init_fn(worker_id: int) -> None:
    """Per-worker deterministic seeding derived from the main-process torch seed
    (which set_seed_official set from the single master seed). Defensive: today
    num_workers=0 so this is moot, but it makes any future worker use reproducible."""
    base = (torch.initial_seed() + worker_id) % (2 ** 31)
    np.random.seed(base)
    random.seed(base)
