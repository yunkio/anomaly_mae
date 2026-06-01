"""
TimesNet for Baseline Comparison

Based on: "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis"
Paper: ICLR 2023, https://openreview.net/pdf?id=ju_Uqw384Oq
Original code: https://github.com/thuml/Time-Series-Library (MIT License)
"""

from .model import TimesNet
from .wrapper import TimesNetBaseline

__all__ = ["TimesNet", "TimesNetBaseline"]
