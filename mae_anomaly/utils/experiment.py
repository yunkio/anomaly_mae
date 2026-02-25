"""Experiment utilities for configuration and execution."""

import os
import re

from mae_anomaly import Config


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

    return config
