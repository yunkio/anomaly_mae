"""Data loaders for comparison experiments.

All experiments use UnifiedLoader which wraps MAE raw loaders + z-score normalization.
"""
from .unified_loader import UnifiedLoader
