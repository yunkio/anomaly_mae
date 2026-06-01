#!/usr/bin/env python
"""
Experiment configurations for baseline comparison experiments.

All experiments use UnifiedLoader (comparison.data.unified_loader) which calls
MAE raw loaders + z-score normalization directly. No duplicate loading code.

Each config maps an experiment name to:
- loader_kwargs: Parameters for UnifiedLoader
- results_dir_name: Output directory under comparison/results/
- model_preset: Hyperparameter set ('wadi_14days' or 'general')
- train_stride: Training stride for neural models (None=default, 3=SWaT)
- has_excl22: Whether to compute excl22 metrics (exclude largest test anomaly region)
- segment_aware_training: Whether neural models need segment-aware windowing (normalonly)
- all_models_list: Models to track in status table
"""

# ============================================================
# Model lists
# ============================================================

# Standard 15 baselines (Simple 5 + Neural 3 + SOTA 7)
# Single source of truth for Notion `Baseline Comparison` page Section 1.
STANDARD_BASELINES = [
    'random', 'sensor_range', 'pca_error', 'l2_norm', 'nn_distance',
    'mlp', 'mlpmixer', 'transformer', 'gcn_lstm',
    'anomaly_transformer', 'tranad', 'usad', 'dagmm', 'gdn', 'omnianomaly',
    # === 2026-05-19 batch (7 active SOTA, post 3-model removal 2026-05-26) ===
    'tfmae', 'timesnet', 'dcdetector', 'memto', 'moderntcn',
    'catch', 'npsr',
]

# Simulation: same 15-baseline set as STANDARD_BASELINES
SIMULATION_BASELINES = STANDARD_BASELINES

# Weakly-supervised baselines (2026-05-29/30 SSL official-repo porting) — Q1-ONLY.
# Deliberately NOT added to STANDARD_BASELINES: those run under BOTH Q1 (full) and
# Q3 (normalonly), but weak-supervised models require labeled anomalies and are N/A
# under Q3 (their wrappers raise RuntimeError on all-zero train_y). Run these on Q1
# experiments via `--model <name>` (or a Q1-only model list); the runner dispatches
# them through `run_weak_sota_baseline_with_epoch_eval` (forwards train_y). Keeping
# this as a separate list leaves every existing EXPERIMENT_CONFIGS entry untouched.
WEAK_SUPERVISED_BASELINES = ['wetas', 'treemil', 'nrdetector', 'deepmil']


# ============================================================
# Experiment Configurations (base: WaDi×4 + SWaT×2 + simulation×2 + PSM×2 = 10)
# Plus SMD 28 machines × 2 variants = 56 generated below
# Plus Exathlon 6 apps × 2 variants = 12 generated below
# ============================================================

EXPERIMENT_CONFIGS = {
    # ==================== WaDi (14days normal + attack) ====================
    # Matches MAE #40 structure: WaDi/A1, WaDi/A2
    "wadi_14days_A1": {
        "loader_kwargs": {"dataset": "wadi_14days", "scenario": "A1"},
        "results_dir_name": "WaDi/A1",
        "dataset_name": "WaDi A1",
        "model_preset": "default",
        "train_stride": None,
        "has_excl22": False,
        "segment_aware_training": False,
        "all_models_list": STANDARD_BASELINES,
    },
    "wadi_14days_A1_normalonly": {
        "loader_kwargs": {"dataset": "wadi_14days", "scenario": "A1", "variant": "normalonly"},
        "results_dir_name": "WaDi/A1",
        "dataset_name": "WaDi A1 NormalOnly",
        "model_preset": "default",
        "train_stride": None,
        "has_excl22": False,
        "segment_aware_training": True,
        "all_models_list": STANDARD_BASELINES,
    },
    "wadi_14days_A2": {
        "loader_kwargs": {"dataset": "wadi_14days", "scenario": "A2"},
        "results_dir_name": "WaDi/A2",
        "dataset_name": "WaDi A2",
        "model_preset": "default",
        "train_stride": None,
        "has_excl22": False,
        "segment_aware_training": False,
        "all_models_list": STANDARD_BASELINES,
    },
    "wadi_14days_A2_normalonly": {
        "loader_kwargs": {"dataset": "wadi_14days", "scenario": "A2", "variant": "normalonly"},
        "results_dir_name": "WaDi/A2",
        "dataset_name": "WaDi A2 NormalOnly",
        "model_preset": "default",
        "train_stride": None,
        "has_excl22": False,
        "segment_aware_training": True,
        "all_models_list": STANDARD_BASELINES,
    },

    # ==================== WaDi A1 ====================
    "wadi_A1": {
        "loader_kwargs": {"dataset": "wadi", "scenario": "A1"},
        "results_dir_name": "WaDi_A1",
        "dataset_name": "WaDi A1",
        "model_preset": "default",
        "train_stride": None,
        "has_excl22": False,
        "segment_aware_training": False,
        "all_models_list": STANDARD_BASELINES,
        "add_mae_command": "python comparison/add_mae_results.py --experiment wadi_A1",
    },
    "wadi_A1_normalonly": {
        "loader_kwargs": {"dataset": "wadi", "scenario": "A1", "variant": "normalonly"},
        "results_dir_name": "WaDi_A1_normalonly",
        "dataset_name": "WaDi A1 NormalOnly",
        "model_preset": "default",
        "train_stride": None,
        "has_excl22": False,
        "segment_aware_training": True,
        "all_models_list": STANDARD_BASELINES,
        "add_mae_command": "python comparison/add_mae_results.py --experiment wadi_A1_normalonly",
    },

    # ==================== WaDi A2 ====================
    "wadi_A2": {
        "loader_kwargs": {"dataset": "wadi", "scenario": "A2"},
        "results_dir_name": "WaDi_A2",
        "dataset_name": "WaDi A2",
        "model_preset": "default",
        "train_stride": None,
        "has_excl22": False,
        "segment_aware_training": False,
        "all_models_list": STANDARD_BASELINES,
        "add_mae_command": "python comparison/add_mae_results.py --experiment wadi_A2",
    },
    "wadi_A2_normalonly": {
        "loader_kwargs": {"dataset": "wadi", "scenario": "A2", "variant": "normalonly"},
        "results_dir_name": "WaDi_A2_normalonly",
        "dataset_name": "WaDi A2 NormalOnly",
        "model_preset": "default",
        "train_stride": None,
        "has_excl22": False,
        "segment_aware_training": True,
        "all_models_list": STANDARD_BASELINES,
        "add_mae_command": "python comparison/add_mae_results.py --experiment wadi_A2_normalonly",
    },

    # ==================== SWaT A1+A2 ====================
    # Matches MAE #40 structure: SWaT/A1A2_full + SWaT/A1A2_excl22
    "swat_a1a2": {
        "loader_kwargs": {"dataset": "swat"},
        "results_dir_name": "SWaT/A1A2_full",
        "dataset_name": "SWaT A1+A2",
        "model_preset": "default",
        "train_stride": None,
        "has_excl22": True,
        "segment_aware_training": False,
        "all_models_list": STANDARD_BASELINES,
    },
    "swat_a1a2_normalonly": {
        "loader_kwargs": {"dataset": "swat", "variant": "normalonly"},
        "results_dir_name": "SWaT/A1A2_full",
        "dataset_name": "SWaT A1A2 NormalOnly",
        "model_preset": "default",
        "train_stride": None,
        "has_excl22": True,
        "segment_aware_training": True,
        "all_models_list": STANDARD_BASELINES,
    },

    # ==================== Simulation ====================
    # Matches MAE #40 structure: simulation/simulation
    "simulation": {
        "loader_kwargs": {
            "dataset": "simulation",
            "total_length": 275000, "train_ratio": 0.8,
            "num_features": 8, "interval_scale": 0.75, "seed": 42,
        },
        "results_dir_name": "simulation/simulation",
        "dataset_name": "Simulation (Phase2 conditions)",
        "model_preset": "default",
        "train_stride": None,
        "has_excl22": False,
        "segment_aware_training": False,
        "all_models_list": SIMULATION_BASELINES,
    },
    "simulation_sim_normalonly": {
        "loader_kwargs": {
            "dataset": "simulation", "variant": "normalonly",
            "total_length": 275000, "train_ratio": 0.8,
            "num_features": 8, "interval_scale": 0.75, "seed": 42,
        },
        "results_dir_name": "simulation/simulation",
        "dataset_name": "Simulation NormalOnly (simple)",
        "model_preset": "default",
        "train_stride": None,
        "has_excl22": False,
        "segment_aware_training": True,
        "all_models_list": SIMULATION_BASELINES,
    },
    "simulation_complex": {
        "loader_kwargs": {
            "dataset": "simulation_complex",
            "total_length": 275000, "train_ratio": 0.8,
            "num_features": 8, "seed": 42,
        },
        "results_dir_name": "simulation_complex",
        "dataset_name": "Simulation Complex (complexity=True)",
        "model_preset": "default",
        "train_stride": None,
        "has_excl22": False,
        "segment_aware_training": False,
        "all_models_list": STANDARD_BASELINES,
        "add_mae_command": "python comparison/add_mae_results.py --experiment simulation_complex",
    },
    "simulation_normalonly": {
        "loader_kwargs": {
            # NOTE: Uses simulation_complex (complexity=True) — legacy behavior
            "dataset": "simulation_complex", "variant": "normalonly",
            "total_length": 275000, "train_ratio": 0.8,
            "num_features": 8, "interval_scale": 0.75, "seed": 42,
        },
        "results_dir_name": "simulation_normalonly",
        "dataset_name": "Simulation NormalOnly (complexity=True)",
        "model_preset": "default",
        "train_stride": None,
        "has_excl22": False,
        "segment_aware_training": True,
        "all_models_list": SIMULATION_BASELINES,
        # epoch override 제거: 전 모델 20 epoch 통일 (preset 'default')
        "add_mae_command": "python comparison/add_mae_results.py --experiment simulation_normalonly",
    },

    # ==================== PSM (Pooled Server Metrics, eBay) ====================
    # Single contiguous stream — same simple 50/50 split pattern as SMD/SWaT:
    # train = orig train file + front 50% of test, test = back 50% of test
    "psm": {
        "loader_kwargs": {"dataset": "psm"},
        "results_dir_name": "PSM",
        "dataset_name": "PSM",
        "model_preset": "default",
        "train_stride": None,
        "has_excl22": False,
        "segment_aware_training": False,
        "all_models_list": STANDARD_BASELINES,
    },
    "psm_normalonly": {
        "loader_kwargs": {"dataset": "psm", "variant": "normalonly"},
        "results_dir_name": "PSM",
        "dataset_name": "PSM NormalOnly",
        "model_preset": "default",
        "train_stride": None,
        "has_excl22": False,
        "segment_aware_training": True,
        "all_models_list": STANDARD_BASELINES,
    },

    # ==================== SMAP (NASA Telemanom, 54 channels) ====================
    # QuoVadis pattern: per-channel split + time-concat. Each channel split via
    # SMD-style safe cut at ~50% of test (margin=10), pushed outside anomaly
    # regions. `run_boundaries` marks every discontinuity (orig_train↔test_front
    # inside each channel + inter-channel joins) so segment-aware windowing in
    # `_apply_normalonly()` and SOTA wrappers cannot bridge non-adjacent
    # recordings.
    "smap": {
        "loader_kwargs": {"dataset": "smap"},
        "results_dir_name": "SMAP",
        "dataset_name": "SMAP",
        "model_preset": "default",
        "train_stride": None,
        "has_excl22": False,
        "segment_aware_training": False,
        "all_models_list": STANDARD_BASELINES,
    },
    "smap_normalonly": {
        "loader_kwargs": {"dataset": "smap", "variant": "normalonly"},
        "results_dir_name": "SMAP",
        "dataset_name": "SMAP NormalOnly",
        "model_preset": "default",
        "train_stride": None,
        "has_excl22": False,
        "segment_aware_training": True,
        "all_models_list": STANDARD_BASELINES,
    },

    # ==================== MSL (NASA Telemanom, 27 channels) ====================
    # Same processing as SMAP. Feature dim 55 (vs SMAP's 25).
    "msl": {
        "loader_kwargs": {"dataset": "msl"},
        "results_dir_name": "MSL",
        "dataset_name": "MSL",
        "model_preset": "default",
        "train_stride": None,
        "has_excl22": False,
        "segment_aware_training": False,
        "all_models_list": STANDARD_BASELINES,
    },
    "msl_normalonly": {
        "loader_kwargs": {"dataset": "msl", "variant": "normalonly"},
        "results_dir_name": "MSL",
        "dataset_name": "MSL NormalOnly",
        "model_preset": "default",
        "train_stride": None,
        "has_excl22": False,
        "segment_aware_training": True,
        "all_models_list": STANDARD_BASELINES,
    },
    # SMD / Exathlon 'concat' (all machines / all apps combined into one stream;
    # per-segment test-cut, run_boundaries on every segment + seam). Mirror smap/msl.
    "smd_concat": {
        "loader_kwargs": {"dataset": "smd_concat"},
        "results_dir_name": "SMD/concat",
        "dataset_name": "SMD concat",
        "model_preset": "default",
        "train_stride": None,
        "has_excl22": False,
        "segment_aware_training": False,
        "all_models_list": STANDARD_BASELINES,
    },
    "smd_concat_normalonly": {
        "loader_kwargs": {"dataset": "smd_concat", "variant": "normalonly"},
        "results_dir_name": "SMD/concat",
        "dataset_name": "SMD concat NormalOnly",
        "model_preset": "default",
        "train_stride": None,
        "has_excl22": False,
        "segment_aware_training": True,
        "all_models_list": STANDARD_BASELINES,
    },
    "exathlon_concat": {
        "loader_kwargs": {"dataset": "exathlon_concat"},
        "results_dir_name": "Exathlon/concat",
        "dataset_name": "Exathlon concat",
        "model_preset": "default",
        "train_stride": None,
        "has_excl22": False,
        "segment_aware_training": False,
        "all_models_list": STANDARD_BASELINES,
    },
    "exathlon_concat_normalonly": {
        "loader_kwargs": {"dataset": "exathlon_concat", "variant": "normalonly"},
        "results_dir_name": "Exathlon/concat",
        "dataset_name": "Exathlon concat NormalOnly",
        "model_preset": "default",
        "train_stride": None,
        "has_excl22": False,
        "segment_aware_training": True,
        "all_models_list": STANDARD_BASELINES,
    },
}


# ============================================================
# SMD Simple Split Experiments (28 machines)
# Train = orig train + front 50% test, Test = back 50% test
# ============================================================

# Single source of truth: mae_anomaly.datasets.loaders.SMD_MACHINE_NAMES
from mae_anomaly.datasets.loaders import SMD_MACHINE_NAMES

for _machine in SMD_MACHINE_NAMES:
    # Non-normalonly (for Q1/Q2)
    _key = f"smd_{_machine}"
    EXPERIMENT_CONFIGS[_key] = {
        "loader_kwargs": {
            "dataset": "smd_simple",
            "machine": _machine,
        },
        "results_dir_name": f"SMD/{_machine}",
        "dataset_name": f"SMD {_machine}",
        "model_preset": "default",
        "train_stride": None,
        "has_excl22": False,
        "segment_aware_training": False,
        "all_models_list": STANDARD_BASELINES,
    }

    # Normalonly (for Q3/Q4)
    _key_no = f"smd_{_machine}_normalonly"
    EXPERIMENT_CONFIGS[_key_no] = {
        "loader_kwargs": {
            "dataset": "smd_simple",
            "machine": _machine,
            "variant": "normalonly",
        },
        "results_dir_name": f"SMD/{_machine}",
        "dataset_name": f"SMD {_machine} NormalOnly",
        "model_preset": "default",
        "train_stride": None,
        "has_excl22": False,
        "segment_aware_training": True,
        "all_models_list": STANDARD_BASELINES,
    }

del _machine, _key, _key_no  # Clean up loop variables


# ============================================================
# Exathlon Experiments (6 apps × standard + normalonly)
# Apps {1, 2, 4, 5, 6, 9} per TimeSeAD 6-app convention.
# Per-app: train = all undisturbed + first floor(N_dist/2) disturbed (sorted by trace_id),
#          test = remaining disturbed.
# ============================================================

from mae_anomaly.datasets.loaders import EXATHLON_APP_IDS

for _app in EXATHLON_APP_IDS:
    # Non-normalonly (for Q1/Q2)
    _key = f"exathlon_app{_app}"
    EXPERIMENT_CONFIGS[_key] = {
        "loader_kwargs": {
            "dataset": "exathlon",
            "app": _app,
        },
        "results_dir_name": f"Exathlon/app{_app}",
        "dataset_name": f"Exathlon app{_app}",
        "model_preset": "default",
        "train_stride": None,
        "has_excl22": False,
        "segment_aware_training": False,
        "all_models_list": STANDARD_BASELINES,
    }
    # Normalonly (for Q3/Q4)
    _key_no = f"exathlon_app{_app}_normalonly"
    EXPERIMENT_CONFIGS[_key_no] = {
        "loader_kwargs": {
            "dataset": "exathlon",
            "app": _app,
            "variant": "normalonly",
        },
        "results_dir_name": f"Exathlon/app{_app}",
        "dataset_name": f"Exathlon app{_app} NormalOnly",
        "model_preset": "default",
        "train_stride": None,
        "has_excl22": False,
        "segment_aware_training": True,
        "all_models_list": STANDARD_BASELINES,
    }

del _app, _key, _key_no  # Clean up loop variables


# ============================================================
# SMAP per-channel Experiments (Pattern B — SMD/Exathlon-style per-entity)
# 54 channels × {standard, normalonly}
# Train = orig train + front 50% test, Test = back 50% test
# UnifiedLoader fits min-max / z-score on this single channel's train portion
# only — per-channel scaler (Telemanom/OmniAnomaly entity-level convention).
# Pattern A entries (`smap`, `smap_normalonly`) above remain available.
# ============================================================

from mae_anomaly.datasets.loaders import SMAP_CHANNEL_NAMES

for _ch in SMAP_CHANNEL_NAMES:
    _key = f"smap_{_ch}"
    EXPERIMENT_CONFIGS[_key] = {
        "loader_kwargs": {
            "dataset": "smap_simple",
            "channel": _ch,
        },
        "results_dir_name": f"SMAP/{_ch}",
        "dataset_name": f"SMAP {_ch}",
        "model_preset": "default",
        "train_stride": None,
        "has_excl22": False,
        "segment_aware_training": False,
        "all_models_list": STANDARD_BASELINES,
    }

    _key_no = f"smap_{_ch}_normalonly"
    EXPERIMENT_CONFIGS[_key_no] = {
        "loader_kwargs": {
            "dataset": "smap_simple",
            "channel": _ch,
            "variant": "normalonly",
        },
        "results_dir_name": f"SMAP/{_ch}",
        "dataset_name": f"SMAP {_ch} NormalOnly",
        "model_preset": "default",
        "train_stride": None,
        "has_excl22": False,
        "segment_aware_training": True,
        "all_models_list": STANDARD_BASELINES,
    }

del _ch, _key, _key_no


# ============================================================
# MSL per-channel Experiments (Pattern B)
# 27 channels × {standard, normalonly}
# Same processing as SMAP per-channel.
# ============================================================

from mae_anomaly.datasets.loaders import MSL_CHANNEL_NAMES

for _ch in MSL_CHANNEL_NAMES:
    _key = f"msl_{_ch}"
    EXPERIMENT_CONFIGS[_key] = {
        "loader_kwargs": {
            "dataset": "msl_simple",
            "channel": _ch,
        },
        "results_dir_name": f"MSL/{_ch}",
        "dataset_name": f"MSL {_ch}",
        "model_preset": "default",
        "train_stride": None,
        "has_excl22": False,
        "segment_aware_training": False,
        "all_models_list": STANDARD_BASELINES,
    }

    _key_no = f"msl_{_ch}_normalonly"
    EXPERIMENT_CONFIGS[_key_no] = {
        "loader_kwargs": {
            "dataset": "msl_simple",
            "channel": _ch,
            "variant": "normalonly",
        },
        "results_dir_name": f"MSL/{_ch}",
        "dataset_name": f"MSL {_ch} NormalOnly",
        "model_preset": "default",
        "train_stride": None,
        "has_excl22": False,
        "segment_aware_training": True,
        "all_models_list": STANDARD_BASELINES,
    }

del _ch, _key, _key_no


# ============================================================
# Loader module/class (unified for all experiments)
# ============================================================

LOADER_MODULE = "comparison.data.unified_loader"
LOADER_CLASS = "UnifiedLoader"


def get_experiment_config(experiment_name: str) -> dict:
    """Get experiment configuration by name."""
    if experiment_name not in EXPERIMENT_CONFIGS:
        available = sorted(EXPERIMENT_CONFIGS.keys())
        raise ValueError(
            f"Unknown experiment: '{experiment_name}'\n"
            f"Available experiments:\n  " + "\n  ".join(available)
        )
    return EXPERIMENT_CONFIGS[experiment_name]


def list_experiments():
    """Print all available experiments."""
    print(f"\n{'Experiment':<30} {'Dataset':<40} {'Preset':<15} {'Special'}")
    print("-" * 110)

    for name, cfg in sorted(EXPERIMENT_CONFIGS.items()):
        special = []
        if cfg.get('has_excl22'):
            special.append('excl22')
        if cfg.get('segment_aware_training'):
            special.append('normalonly')
        if cfg.get('train_stride'):
            special.append(f'stride={cfg["train_stride"]}')

        special_str = ', '.join(special) if special else '-'
        print(f"{name:<30} {cfg.get('dataset_name', ''):<40} {cfg['model_preset']:<15} {special_str}")

    print(f"\nTotal: {len(EXPERIMENT_CONFIGS)} experiments")
