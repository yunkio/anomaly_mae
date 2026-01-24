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
    seq_length: int = 100
    num_features: int = 8  # Multivariate: 8 features (expanded for sliding window dataset)
    num_train_samples: int = 10000  # (for legacy dataset)
    num_test_samples: int = 2500  # (for legacy dataset)
    train_anomaly_ratio: float = 0.05  # ~5% anomaly in train
    test_anomaly_ratio: float = 0.25

    # Sliding window dataset parameters
    use_sliding_window_dataset: bool = True  # Use new sliding window dataset
    sliding_window_total_length: int = 440000  # Total length of long time series (1/5 of original 2.2M)
    sliding_window_stride: int = 11  # Stride for window extraction (overlapping windows)
    anomaly_interval_scale: float = 1.5  # Scale factor for anomaly intervals (tuned for ~5% anomaly)

    # Test set target ratios (for downsampling)
    # Ratios: pure_normal=65%, disturbing_normal=15%, anomaly=25%
    test_ratio_pure_normal: float = 0.65
    test_ratio_disturbing_normal: float = 0.15
    test_ratio_anomaly: float = 0.25

    # Model parameters
    d_model: int = 64
    nhead: int = 4
    num_encoder_layers: int = 3
    num_teacher_decoder_layers: int = 4
    num_student_decoder_layers: int = 1
    num_shared_decoder_layers: int = 0  # Shared decoder layers before teacher/student decoders
    # - 0: No shared decoder (default)
    # - >0: Shared decoder trained with teacher, separate mask tokens for student
    dim_feedforward: int = 256
    dropout: float = 0.1
    masking_ratio: float = 0.4
    masking_strategy: str = 'patch'  # 'patch' or 'feature_wise'
    # - 'patch': Mask entire patches (all features at same time points)
    # - 'feature_wise': Mask each feature independently (different time points per feature)
    num_patches: int = 10  # 10 patches per sequence
    patch_size: int = 10  # seq_length / num_patches = 100 / 10 = 10
    patchify_mode: str = 'linear'  # 'patch_cnn', 'linear'
    # - 'patch_cnn': Patchify first, then CNN per patch (no cross-patch leakage)
    # - 'linear': Patchify then linear embedding (MAE original style, no CNN)
    mask_after_encoder: bool = False  # Standard MAE masking architecture
    # - False: Mask tokens go through encoder (current behavior)
    # - True: Encode visible patches only, insert mask tokens before decoder (standard MAE)
    shared_mask_token: bool = True  # Share mask token between teacher and student
    # - True: Single mask token shared (current behavior)
    # - False: Separate mask tokens for teacher and student decoders

    # CNN architecture for patch_cnn mode
    cnn_channels: tuple = None  # (mid_channels, out_channels) for patch_cnn, None=default based on d_model
    # - None: Use default (d_model//2, d_model)
    # - (16, 32): Smaller CNN
    # - (64, 128): Larger CNN

    # Loss parameters
    margin: float = 0.5
    lambda_disc: float = 0.5
    margin_type: str = 'hinge'  # 'hinge' (relu), 'softplus', 'dynamic'
    dynamic_margin_k: float = 3.0  # k for dynamic margin (mu + k*sigma)
    patch_level_loss: bool = True  # True=patch-level, False=window-level discrepancy loss

    # Student loss parameters
    use_student_reconstruction_loss: bool = False  # Add reconstruction loss to student
    # - False: Student only learns via discrepancy loss (default)
    # - True: Student also has reconstruction loss (opposite direction for anomaly)
    anomaly_loss_weight: float = 1.0  # Weight multiplier for anomaly discrepancy loss
    # - 1.0: Default (equal weight)
    # - 2.0/3.0/5.0: Stronger interference on anomaly samples

    # Anomaly score computation mode
    anomaly_score_mode: str = 'default'
    # - 'default': recon + lambda_disc * disc (original)
    # - 'normalized': Z-score normalization (recon_z + disc_z)
    # - 'adaptive': Auto-scaled lambda (recon + (mean_recon/mean_disc) * disc)
    # - 'ratio_weighted': Ratio-based (recon * (1 + disc/median_disc))

    # Training parameters
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    warmup_epochs: int = 10
    teacher_only_warmup_epochs: int = 1  # First N epochs train teacher only (no discrepancy/student loss)

    # Inference parameters
    mask_last_n: int = 10  # Last 1 patch (= patch_size)

    # Point-level PA%K aggregation method
    point_aggregation_method: str = 'voting'  # 'mean', 'median', 'voting'
    # - 'mean': Average of window scores for each timestep
    # - 'median': Median of window scores for each timestep
    # - 'voting': Majority vote of binary predictions (default)

    # Ablation flags
    use_discrepancy_loss: bool = True
    use_teacher: bool = True
    use_student: bool = True
    use_masking: bool = True
    force_mask_anomaly: bool = False  # Force mask patches containing anomalies during training

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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
