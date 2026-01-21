"""
MAE Anomaly Detection Package

Self-Distilled Masked Autoencoder for Time Series Anomaly Detection
"""

from .config import Config, set_seed
from .dataset import MultivariateTimeSeriesDataset, ANOMALY_TYPE_NAMES, ANOMALY_TYPES
from .model import PositionalEncoding, SelfDistilledMAEMultivariate
from .loss import SelfDistillationLoss
from .trainer import Trainer
from .evaluator import Evaluator

__all__ = [
    # Config
    'Config',
    'set_seed',
    # Data
    'MultivariateTimeSeriesDataset',
    'ANOMALY_TYPE_NAMES',
    'ANOMALY_TYPES',
    # Model
    'PositionalEncoding',
    'SelfDistilledMAEMultivariate',
    # Loss
    'SelfDistillationLoss',
    # Training
    'Trainer',
    'Evaluator',
]
