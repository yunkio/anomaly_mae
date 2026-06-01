"""
TFMAE for Baseline Comparison

Based on: "Temporal-Frequency Masked Autoencoders for Time Series Anomaly Detection"
Paper: ICDE 2024, https://ieeexplore.ieee.org/document/10597757
Original code: https://github.com/LMissher/TFMAE (MIT License)
"""

from .model import MTFA, my_kl_loss
from .wrapper import TFMAEBaseline

__all__ = ["MTFA", "TFMAEBaseline", "my_kl_loss"]
