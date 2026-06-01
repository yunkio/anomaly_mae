"""ModernTCN baseline (ICLR 2024 Spotlight): pure convolution structure for TSA tasks."""

from .model import ModernTCN
from .wrapper import ModernTCNBaseline

__all__ = ["ModernTCNBaseline", "ModernTCN"]
