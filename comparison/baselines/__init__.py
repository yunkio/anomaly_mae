"""
Baseline models for Time Series Anomaly Detection

Based on QuoVadisTAD (arXiv:2405.02678)

Simple Baselines:
- Random: Uniform random scores
- SensorRangeDeviation: Deviation from training range
- PCAError: PCA reconstruction error
- L2Norm: L2 norm of feature vectors
- NNDistance: 1-NN distance to training data
Neural Baselines (QuoVadisTAD):
- MLPBaseline: 1-Layer MLP
- MLPMixerBaseline: Single Block MLPMixer
- TransformerBaseline: Single Transformer Block
- GCNLSTMBaseline: 1-Layer GCN-LSTM

Deep Learning SOTA:
- AnomalyTransformerBaseline: Anomaly Transformer (ICLR 2022)
- TranADBaseline: TranAD (VLDB 2022)
- USADBaseline: USAD (KDD 2020)
- DAGMMBaseline: DAGMM (ICLR 2018) — TranAD-author impl (2026-05-25)
- GDNBaseline: GDN (AAAI 2021)
- OmniAnomalyBaseline: OmniAnomaly (KDD 2019)
"""

# Simple baselines
from .random import RandomBaseline
from .sensor_range import SensorRangeDeviation
from .pca_error import PCAError
from .l2_norm import L2Norm
from .nn_distance import NNDistance
# Neural baselines (QuoVadisTAD)
from .mlp import MLPBaseline
from .mlpmixer import MLPMixerBaseline
from .transformer import TransformerBaseline
from .gcn_lstm import GCNLSTMBaseline

# Deep learning SOTA
# anomaly_transformer: 2026-05-22 dir 손실 (git untracked였고 백업 없음). legacy 결과는
# 기존 result dir에 모두 보존되어 있어 신규 학습 불요. import 격리로 다른 모델 영향 X.
try:
    from .anomaly_transformer import AnomalyTransformerBaseline
    _HAS_AT_MODULE = True
except ImportError:
    AnomalyTransformerBaseline = None
    _HAS_AT_MODULE = False
from .tranad import TranADBaseline
from .usad import USADBaseline
from .dagmm import DAGMMBaseline
from .gdn import GDNBaseline
from .omnianomaly import OmniAnomalyBaseline

# New SOTA additions (2026-05-19 batch — 7 models, post 3-model removal 2026-05-26)
# Phase 1: TFMAE + NPSR (MAE-direct competitor + F1* philosophy)
from .tfmae import TFMAEBaseline
# Phase 2: TimesNet + DCdetector (Top-tier SOTA)
from .timesnet import TimesNetBaseline
from .dcdetector import DCdetectorBaseline
# Phase 3: MEMTO (memory-guided Transformer)
from .memto import MEMTOBaseline
# Phase 3: ModernTCN (large-kernel convolutional TSA)
from .moderntcn import ModernTCNBaseline
# Phase 5: CATCH (channel-aware freq reconstruction)
from .catch import CATCHBaseline
# Phase 1: NPSR (point + sequence reconstruction with nominality score)
from .npsr import NPSRBaseline

# Evaluator: DEPRECATED — use mae_anomaly.evaluator directly
# from .evaluator import ...  # Removed: all metric computation now uses mae_anomaly.evaluator

__all__ = [
    # Simple baselines
    'RandomBaseline',
    'SensorRangeDeviation',
    'PCAError',
    'L2Norm',
    'NNDistance',
    # Neural baselines (QuoVadisTAD)
    'MLPBaseline',
    'MLPMixerBaseline',
    'TransformerBaseline',
    'GCNLSTMBaseline',
    # Deep learning SOTA
    'AnomalyTransformerBaseline',
    'TranADBaseline',
    'USADBaseline',
    'DAGMMBaseline',
    'GDNBaseline',
    'OmniAnomalyBaseline',
    # 2026-05-19 batch (7 models)
    'TFMAEBaseline',
    'TimesNetBaseline',
    'DCdetectorBaseline',
    'MEMTOBaseline',
    'ModernTCNBaseline',
    'CATCHBaseline',
    'NPSRBaseline',
]
