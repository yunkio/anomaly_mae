"""
Configuration classes for MAE anomaly detection
"""

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
    sliding_window_test_stride: int = 21  # Stride for test window extraction
    epoch_offset: bool = True  # Non-replacement random train window offset each epoch (cycles through [0, stride))
    sliding_window_train_ratio: float = 0.8  # Train ratio (220K/275K = 0.8, test = 55K)
    normalize_mode: str = 'zscore'  # Normalization mode for input signals
    # - 'zscore': Per-feature z-score standardization (mean=0, std=1) — default
    # - 'minmax': Per-feature min-max scaling to [0, 1] with clip
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
    dim_feedforward: int = 512  # 4 * d_model
    dropout: float = 0.15
    masking_ratio: float = 0.15
    num_patches: int = 100  # seq_length / patch_size (dynamically computed when window size changes)
    patch_size: int = 5  # Fixed patch size; num_patches = seq_length / patch_size
    patchify_mode: str = 'patch_cnn'  # 'patch_cnn', 'linear'
    # - 'patch_cnn': Patchify first, then CNN per patch (no cross-patch leakage)
    # - 'linear': Patchify then linear embedding (MAE original style, no CNN)
    mask_after_encoder: bool = True  # Standard MAE masking architecture
    # - False: Mask tokens go through encoder (current behavior)
    # - True: Encode visible patches only, insert mask tokens before decoder (standard MAE)
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
    use_grl: bool = False  # Enable GRL for anomaly-aware student training
    # - False: No GRL (default, existing behavior unchanged)
    # - True: GRL classifier on student hidden → anomaly_loss disabled, GRL handles disc generation
    grl_cls_hidden: int = 0  # Classifier hidden dim (0 = auto: d_model // 2)
    grl_loss_weight: float = 1.0  # GRL loss weight multiplier applied to adaptive lambda
    grl_target_mode: str = 'patch'  # GRL classifier target granularity
    # - 'patch': Target = patch_has_anomaly (current, per-patch label)
    # - 'window': Target = has_anomaly_in_window (window-level label, all patches same)
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
    grl_cls_lr_ratio: float = 1.0  # GRL classifier LR as fraction of main LR
    # - 1.0: Same LR as main model (default, existing behavior)
    # - <1.0: Slower classifier convergence to prevent collapse (e.g., 0.1)
    #   Only applies to classifier mode (grl_mode='classifier')
    wdgrl_k_critic: int = 5  # Critic update steps per main update (WDGRL only)
    wdgrl_gp_weight: float = 10.0  # Gradient penalty weight (WDGRL only)
    wdgrl_critic_lr: float = 1e-4  # Critic optimizer learning rate (WDGRL only)

    # Feature Matching Loss (independent of GRL)
    use_feature_matching: bool = False  # Enable feature matching loss
    # cosine(teacher_hidden, student_hidden) on masked normal patches
    fm_adaptive_lambda: bool = False  # Adaptive λ for FM-OD gradient balancing
    # - False: FM weight = fm_loss_weight (fixed, default 1:1)
    # - True: FM weight = adaptive_lambda * fm_loss_weight (gradient-balanced)
    fm_distance_metric: str = 'cosine'  # Distance metric for feature matching
    # - 'cosine': 1 - cosine_similarity (default, direction-only)
    # - 'l2': L2 distance (magnitude + direction)
    use_output_discrepancy: bool = True  # Enable output discrepancy loss (normal_loss + anomaly_loss)
    # - True: OD loss active (default, existing behavior)
    # - False: OD loss zeroed, only FM loss contributes to discrepancy_loss
    #   Discrepancy values are still computed for metrics/logging, but don't affect training.
    #   At inference, disc component is excluded from scoring (w_disc=0).
    fm_loss_weight: float = 1.0  # FM:disc training weight ratio (1.0 = equal)

    # Inference scoring weight override (-1 = use training weight)
    eval_disc_weight: float = -1.0  # Disc weight at inference (-1 → 1.0)
    eval_fm_weight: float = -1.0    # FM weight at inference (-1 → fm_loss_weight)

    # Teacher Freeze
    freeze_teacher_after_warmup: bool = False  # Freeze encoder/teacher after warmup (method C: eval + no_grad)
    # - False: All parameters trainable throughout (default)
    # - True: Freeze at teacher_only_warmup_epochs (forced to num_epochs // 2)

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
    learning_rate: float = 1e-3  # Default learning rate (halved from 2e-3 for training stability)
    weight_decay: float = 1e-3
    warmup_epochs: int = 10
    teacher_only_warmup_epochs: int = -1  # First N epochs train teacher only (-1 = auto: num_epochs // 2)
    best_epoch_metric: str = 'pak_auc_f1'  # Metric for best epoch selection
    # - 'pak_auc_f1': PA%K AUC of F1 with per-K threshold re-optimization (best_f1_w_pa, recommended)
    #   Per-K threshold sweep after PA%K segment adjustment (Kim et al., AAAI 2022 tadpak method)
    # - 'pak_auc_f1_raw': PA%K AUC of F1 with fixed threshold (raw_f1_w_pa, legacy comparison)
    # - 'prc_auc': Precision-Recall AUC (legacy default)
    eval_interval: int = 5  # Epoch interval for lightweight test evaluation (contrib ratios)
    # - 1: Every epoch (most detailed, slower)
    # - 5: Every 5 epochs (default, good balance)
    use_amp: bool = True  # Mixed Precision Training (Automatic Mixed Precision)
    # - True: Use float16 for forward pass, float32 for loss/gradients (faster on Tensor Core GPUs)
    # - False: Use float32 everywhere (more stable, required for older GPUs)

    # Ablation flags
    use_discrepancy_loss: bool = True
    use_teacher: bool = True
    use_student: bool = True
    use_masking: bool = True
    force_mask_anomaly: bool = True  # Prioritize masking anomaly patches during training
    # - True: Anomaly patches are masked first within fixed masking budget (masking_ratio).
    #   If anomaly patches exceed the budget, excess remain visible as encoder context.
    #   Masking count is always exactly round(num_patches * masking_ratio) per sample.
    # - False: Random masking only (no anomaly-aware prioritization)

    # Reproducibility
    random_seed: int = 42
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


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
