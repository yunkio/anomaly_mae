"""Dataset loaders and utilities for various datasets (SWaT, WaDi, SMD, PSM, Simulation, TEP, SMAP, MSL)."""

from .loaders import (
    get_dataset_loader,
    DATASET_LOADERS,
    load_swat_combined,
    load_swat_combined_swap,
    load_wadi_14days_combined,
    load_wadi_a2,
    load_simulation,
    load_tep,
    TEP_FAULT_NAMES,
    load_smd,
    load_smd_block_split,
    load_psm,
    SMD_MACHINE_NAMES,
    # SMAP / MSL (NASA Telemanom, Hundman et al. KDD 2018) — added 2026-05-26
    load_smap_combined,    # Pattern A — all-channels concat
    load_msl_combined,     # Pattern A
    load_smap_simple,      # Pattern B — per-channel (SMD/Exathlon-style)
    load_msl_simple,       # Pattern B
    SMAP_CHANNEL_NAMES,
    MSL_CHANNEL_NAMES,
)
from .noisy import NoisyLabelSlidingWindowDataset

__all__ = [
    'get_dataset_loader',
    'DATASET_LOADERS',
    'load_swat_combined',
    'load_swat_combined_swap',
    'load_wadi_14days_combined',
    'load_wadi_a2',
    'load_simulation',
    'load_tep',
    'TEP_FAULT_NAMES',
    'load_smd',
    'load_smd_block_split',
    'load_psm',
    'SMD_MACHINE_NAMES',
    'load_smap_combined',
    'load_msl_combined',
    'load_smap_simple',
    'load_msl_simple',
    'SMAP_CHANNEL_NAMES',
    'MSL_CHANNEL_NAMES',
    'NoisyLabelSlidingWindowDataset',
]
