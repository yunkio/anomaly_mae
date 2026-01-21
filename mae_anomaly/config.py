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
    num_features: int = 5  # Multivariate: 5 features
    num_train_samples: int = 10000  # 5x increase
    num_test_samples: int = 2500  # 5x increase
    train_anomaly_ratio: float = 0.05
    test_anomaly_ratio: float = 0.25

    # Model parameters
    d_model: int = 64
    nhead: int = 4
    num_encoder_layers: int = 3
    num_teacher_decoder_layers: int = 4
    num_student_decoder_layers: int = 1
    dim_feedforward: int = 256
    dropout: float = 0.1
    masking_ratio: float = 0.4
    masking_strategy: str = 'patch'  # 'patch' or 'feature_wise'
    # - 'patch': Mask entire patches (all features at same time points)
    # - 'feature_wise': Mask each feature independently (different time points per feature)
    num_patches: int = 25  # 25 patches per sequence
    patch_size: int = 4  # seq_length / num_patches = 100 / 25 = 4
    patchify_mode: str = 'linear'  # 'cnn_first', 'patch_cnn', 'linear'
    # - 'cnn_first': CNN on full sequence, then patchify (information leakage across patches)
    # - 'patch_cnn': Patchify first, then CNN per patch (no cross-patch leakage)
    # - 'linear': Patchify then linear embedding (MAE original style, no CNN)

    # Loss parameters
    margin: float = 0.5
    lambda_disc: float = 0.5
    margin_type: str = 'hinge'  # 'hinge' (relu), 'softplus', 'dynamic'
    dynamic_margin_k: float = 3.0  # k for dynamic margin (mu + k*sigma)
    patch_level_loss: bool = True  # True=patch-level, False=window-level discrepancy loss

    # Training parameters
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    warmup_epochs: int = 10

    # Inference parameters
    mask_last_n: int = 4  # Last 1 patch (patch_size)

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
