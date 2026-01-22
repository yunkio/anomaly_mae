"""
MAE Anomaly Detection Package

Self-Distilled Masked Autoencoder for Time Series Anomaly Detection
"""

from .config import Config, set_seed
from .dataset_sliding import (
    SlidingWindowTimeSeriesGenerator,
    SlidingWindowDataset,
    NormalDataComplexity,
    FEATURE_NAMES,
    ANOMALY_TYPE_NAMES as SLIDING_ANOMALY_TYPE_NAMES,
    ANOMALY_TYPE_NAMES,
)
from .model import PositionalEncoding, SelfDistilledMAEMultivariate
from .loss import SelfDistillationLoss
from .trainer import Trainer
from .evaluator import Evaluator

__all__ = [
    # Config
    'Config',
    'set_seed',
    # Data (sliding window dataset)
    'SlidingWindowTimeSeriesGenerator',
    'SlidingWindowDataset',
    'NormalDataComplexity',
    'FEATURE_NAMES',
    'ANOMALY_TYPE_NAMES',
    'SLIDING_ANOMALY_TYPE_NAMES',  # Alias for backwards compatibility
    # Model
    'PositionalEncoding',
    'SelfDistilledMAEMultivariate',
    # Loss
    'SelfDistillationLoss',
    # Training
    'Trainer',
    'Evaluator',
]
