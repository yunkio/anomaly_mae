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
    num_train_samples: int = 2000
    num_test_samples: int = 500
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
    masking_ratio: float = 0.6
    masking_strategy: str = 'patch'  # 'patch', 'token', 'temporal', or 'feature_wise'
    patch_size: int = 10  # Number of time steps per patch (for patch-based masking)

    # Loss parameters
    margin: float = 1.0
    lambda_disc: float = 0.5

    # Training parameters
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    warmup_epochs: int = 10

    # Inference parameters
    mask_last_n: int = 10

    # Ablation flags
    use_discrepancy_loss: bool = True
    use_teacher: bool = True
    use_student: bool = True
    use_masking: bool = True

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
