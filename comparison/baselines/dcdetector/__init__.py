"""
DCdetector for Baseline Comparison

Based on: "DCdetector: Dual Attention Contrastive Representation Learning for Time Series Anomaly Detection"
Paper: KDD 2023, https://arxiv.org/abs/2306.10347
Original code: https://github.com/DAMO-DI-ML/KDD2023-DCdetector (no LICENSE — attribution only)
"""

from .model import DCdetector, my_kl_loss
from .wrapper import DCdetectorBaseline

__all__ = ["DCdetector", "DCdetectorBaseline", "my_kl_loss"]
