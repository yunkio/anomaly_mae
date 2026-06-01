"""CATCH baseline (ICLR 2025): channel-aware tokenization with frequency reconstruction.

Vendored from https://github.com/decisionintelligence/CATCH (TAB framework,
ts_benchmark/baselines/catch/). The model architecture in `model.py` is bit-identical
to upstream; the wrapper in `wrapper.py` adapts the per-sample sliding-window interface
expected by `comparison.baseline_common.run_sota_baseline_with_epoch_eval()`.
"""

from .model import CATCH, CATCHModel
from .wrapper import CATCHBaseline

__all__ = ["CATCHBaseline", "CATCH", "CATCHModel"]
