"""Experiment utilities for configuration and execution."""

import os
import re

from mae_anomaly import Config
from mae_anomaly.config import apply_official_overrides


# Dynamic d_model candidates (must be divisible by nhead=8)
# 64 and 96 added to allow smaller d_model for low-F datasets at small patch_size.
# 768 is INTENDED (not a bug) for HIGH-F datasets: raw = patch*F can exceed 512.
#   e.g. patch_size=10 → MSL F=55 raw=550→768, WaDi F=123 raw=1230→768.
#   Low/medium-F datasets still resolve to the old [128..512] choices.
# ⚠️ d_model=768 carries a heavy per-epoch host-RAM cost (~30GB by ep500 on fast
#   1-batch datasets like MSL) that is INDEPENDENT of batch_size — recover an OOM
#   via resume-from-checkpoint, do NOT cap d_model to 512 (768 is correct).
D_MODEL_CANDIDATES = [64, 96, 128, 192, 256, 384, 512, 768]


def resolve_test_stride(config) -> int:
    """Resolve the effective sliding-window test stride.

    Phase 6 (2026-05-29): one-line formula
        S = W // 10 - 1
    chosen so that:

    * coverage = W / S = 10·W / (W − 10) ≈ 10–11 for W ∈ [100, 1000].
    * S ends in the digit 9, which is automatically coprime to every
      standard patch size used in this codebase ({2, 4, 5, 8, 10, 20}).
      The coprime property guarantees that, as the covering window
      index varies, the patch-position residue ``(k·S) mod P`` cycles
      through all ``P`` values — so the ~10 patch scores averaged into
      each timestep's point score sample maximally diverse contexts.

    Override: any positive ``sliding_window_test_stride`` in the config
    is used verbatim (escape hatch for legacy reproduction).
    """
    raw = int(getattr(config, 'sliding_window_test_stride', -1))
    if raw > 0:
        return raw
    W = int(getattr(config, 'seq_length', 1))
    return max(1, W // 10 - 1)


def resolve_dynamic_d_model(num_features: int, patch_size: int) -> int:
    """Select d_model >= raw patch info from candidate list.

    Chooses the smallest candidate in `D_MODEL_CANDIDATES`
    (= [64, 96, 128, 192, 256, 384, 512, 768]) that is >= `patch_size * num_features`.
    Caps at 768 (max candidate).

    For patch_size=10: low/medium-F datasets match the prior [128..512] list, but
    HIGH-F datasets now resolve to 768 (MSL F=55 → raw 550 → 768; WaDi F=123 → 768).
    768 is intended — it is NOT clamped to 512.

    For smaller patch_sizes (e.g., 5) on low-F datasets (e.g., F=8, raw=40),
    the function now returns 64 instead of 128, providing finer adaptation.

    Args:
        num_features: Number of input features.
        patch_size: Patch size (timesteps per patch).

    Returns:
        Selected d_model value.
    """
    raw = patch_size * num_features
    for d in D_MODEL_CANDIDATES:
        if d >= raw:
            return d
    return D_MODEL_CANDIDATES[-1]  # cap at 768 (max candidate)


def get_next_experiment_number(experiments_dir: str) -> int:
    """Get the next experiment number by scanning existing directories.

    Directories are expected to follow the pattern: {N}_{timestamp}_{suffix}
    where N is an integer.

    Returns:
        Next available integer (max existing + 1, or 0 if none).
    """
    if not os.path.isdir(experiments_dir):
        return 0

    max_num = -1
    for entry in os.listdir(experiments_dir):
        if not os.path.isdir(os.path.join(experiments_dir, entry)):
            continue
        match = re.match(r'^(\d+)_', entry)
        if match:
            max_num = max(max_num, int(match.group(1)))

    return max_num + 1


def make_numbered_experiment_dir(experiments_dir: str, suffix: str) -> str:
    """Create a numbered experiment directory.

    Args:
        experiments_dir: Parent directory (e.g., results/experiments/)
        suffix: Directory suffix (e.g., '20260224_120000_phase1')

    Returns:
        Full path like results/experiments/9_20260224_120000_phase1
    """
    os.makedirs(experiments_dir, exist_ok=True)
    num = get_next_experiment_number(experiments_dir)
    dirname = f"{num}_{suffix}"
    full_path = os.path.join(experiments_dir, dirname)
    os.makedirs(full_path, exist_ok=True)
    return full_path


def make_config(overrides: dict) -> Config:
    """Create Config with defaults + overrides.

    All defaults come from the Config dataclass (config.py).
    Only pass overrides for values that differ from Config defaults.

    If 'dim_feedforward' is not in overrides, it is auto-computed as 4 * d_model
    after all overrides are applied.

    Args:
        overrides: Dictionary of config parameters to override

    Returns:
        Config object with applied overrides
    """
    config = Config()

    # Apply overrides
    for k, v in overrides.items():
        if k == 'name':
            continue
        if hasattr(config, k):
            setattr(config, k, v)

    # Official MAE-mode: FORCE the official bundle as the LAST writer so it wins
    # over every preset / override-string / dataset_def / CANON_271 value. No-op
    # (byte-identical) when config.official is False. Placed AFTER the merge loop
    # and BEFORE dim_feedforward + consistency checks (official touches no geometry).
    config = apply_official_overrides(config)

    # Auto-compute dim_feedforward = 4 * d_model if not explicitly overridden
    if 'dim_feedforward' not in overrides:
        config.dim_feedforward = config.d_model * 4

    # Consistency validation: seq_length must be divisible by patch_size and
    # equal to patch_size * num_patches. Silent inconsistency previously masked
    # by `self.num_patches = seq_length // patch_size` in model.py.
    if config.seq_length % config.patch_size != 0:
        raise ValueError(
            f"seq_length ({config.seq_length}) must be divisible by "
            f"patch_size ({config.patch_size}); got remainder "
            f"{config.seq_length % config.patch_size}."
        )
    if config.seq_length != config.patch_size * config.num_patches:
        raise ValueError(
            f"Inconsistent patch configuration: "
            f"seq_length ({config.seq_length}) != "
            f"patch_size ({config.patch_size}) * num_patches "
            f"({config.num_patches}) = {config.patch_size * config.num_patches}. "
            f"Expected num_patches = "
            f"{config.seq_length // config.patch_size}."
        )

    return config
