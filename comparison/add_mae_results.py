#!/usr/bin/env python
"""
Unified MAE Results Integration Script

Adds MAE model results from ablation experiments to comparison results.json.
Replaces all individual add_mae_*.py scripts.

Modes:
  1. Auto mode (default): Uses hardcoded best configs if known, otherwise
     auto-discovers best MAE config by PRC-AUC from the source directory.
  2. Discover mode (--discover): Lists all available MAE configs with metrics.
  3. Manual mode (--mae-dir + --name): Adds a single specific MAE config.

Usage:
    conda activate dc_vis

    # Auto-discover and add best MAE results
    python comparison/add_mae_results.py --experiment wadi_14days_A1
    python comparison/add_mae_results.py --experiment swat_a1a2
    python comparison/add_mae_results.py --experiment swat_a1a2_normal50

    # Discover available MAE configs
    python comparison/add_mae_results.py --experiment wadi_14days_A1 --discover

    # Manual: add specific MAE config
    python comparison/add_mae_results.py --experiment wadi_A1 \
        --mae-dir results/WaDi/A1/20260202_222231 --name mae_custom --scoring combined

Equivalent commands (old → new):
    python comparison/add_mae_14days_results.py --scenario A1
      → python comparison/add_mae_results.py --experiment wadi_14days_A1

    python comparison/add_mae_14days_results.py --scenario A2 --discover
      → python comparison/add_mae_results.py --experiment wadi_14days_A2 --discover

    python comparison/add_mae_swat_results.py
      → python comparison/add_mae_results.py --experiment swat_a1a2

    python comparison/add_mae_swat_results.py --discover
      → python comparison/add_mae_results.py --experiment swat_a1a2 --discover

    python comparison/add_mae_swat_normal50_results.py
      → python comparison/add_mae_results.py --experiment swat_a1a2_normal50

    python comparison/add_mae_wadi_14days_normal50_results.py --scenario A1
      → python comparison/add_mae_results.py --experiment wadi_14days_normal50_A1

    python comparison/add_mae_wadi_14days_normal50_results.py --scenario A2 --discover
      → python comparison/add_mae_results.py --experiment wadi_14days_normal50_A2 --discover

    python comparison/add_mae_results.py --experiment WaDi_A1 --mae-dir 20260202_222231 --name mae_default
      → python comparison/add_mae_results.py --experiment wadi_A1 --mae-dir results/WaDi/A1/20260202_222231 --name mae_default
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# MAE Source Directory Mapping
# ============================================================
# Maps experiment name → relative path (from PROJECT_ROOT) where MAE results live.
# Each directory contains subdirectories named by config timestamp, each with
# experiment_metadata.json.

MAE_SOURCE_DIRS = {
    # WaDi 14days
    "wadi_14days_A1": "results/WaDi/A1_14days",
    "wadi_14days_A2": "results/WaDi/A2_14days",
    # WaDi 14days Normal50
    "wadi_14days_normal50_A1": "results/WaDi/A1_14days_normal50",
    "wadi_14days_normal50_A2": "results/WaDi/A2_14days_normal50",
    # WaDi A1/A2 variants
    "wadi_A1": "results/WaDi/A1",
    "wadi_A1_normalonly": "results/WaDi/A1_normalonly",
    "wadi_A1_swap": "results/WaDi/A1_swap",
    "wadi_A1_normalonly_swap": "results/WaDi/A1_normalonly_swap",
    "wadi_A2": "results/WaDi/A2",
    "wadi_A2_normalonly": "results/WaDi/A2_normalonly",
    "wadi_A2_swap": "results/WaDi/A2_swap",
    "wadi_A2_normalonly_swap": "results/WaDi/A2_normalonly_swap",
    # SWaT
    "swat_a1a2": "results/SWaT/A1A2",
    "swat_a1a2_normal50": "results/SWaT/A1A2_normal50",
    "swat_a1a2_normalonly": "results/SWaT/A1A2_normalonly",
    "swat_a1a2_swap": "results/SWaT/A1A2_swap",
    "swat_a1a2_swap_normalonly": "results/SWaT/A1A2_swap_normalonly",
    # Simulation
    "simulation": "results/Simulation/phase2",
    "simulation_complex": "results/Simulation/complex",
    "simulation_normalonly": "results/Simulation/normalonly",
    "simulation_complex_normal50": "results/Simulation/complex_normal50",
    "simulation_normal_50": "results/Simulation/normal_50",
    # PSM (Pooled Server Metrics, eBay) — single contiguous stream
    "psm": "results/PSM",
    # Exathlon — 6 Spark apps (retrofit from PSM-trained MAE models)
    # Each app dir is a symlink farm pointing to results/experiments/{EXP}_*/Exathlon/app{N}/
    "exathlon_app1": "results/Exathlon/app1",
    "exathlon_app2": "results/Exathlon/app2",
    "exathlon_app4": "results/Exathlon/app4",
    "exathlon_app5": "results/Exathlon/app5",
    "exathlon_app6": "results/Exathlon/app6",
    "exathlon_app9": "results/Exathlon/app9",
}


# ============================================================
# Best MAE Configurations (hardcoded from ablation analysis)
# ============================================================
# Experiments not listed here will use auto-discovery (best by PRC).

BEST_MAE_CONFIGS = {
    "wadi_14days_A1": {
        "combined": [
            {"config": "20260208_080838_w500_p5_td4_sd2", "name": "mae_w500_p5_td4_sd2"},
        ],
        "teacher": [
            {"config": "20260208_135908_w500_p10_td3_sd1", "name": "mae_teacher_w500_p10_td3_sd1"},
        ],
    },
    "wadi_14days_A2": {
        "combined": [
            {"config": "20260209_113528_w500_p20_enc3", "name": "mae_w500_p20_enc3"},
        ],
        "teacher": [
            {"config": "20260209_113528_w500_p20_enc3", "name": "mae_teacher_w500_p20_enc3"},
        ],
    },
    "swat_a1a2": {
        "combined": [
            {"config": "20260212_005359_w500_p5_td4_sd2", "name": "mae_w500_p5_td4_sd2"},
            {"config": "20260211_231332_w500_p5_default", "name": "mae_w500_p5_default"},
        ],
        "teacher": [
            {"config": "20260212_005359_w500_p5_td4_sd2", "name": "mae_teacher_w500_p5_td4_sd2"},
        ],
    },
}

# Experiments needing excl22 metrics for MAE models (exclude largest test anomaly region)
EXCL22_EXPERIMENTS = {"swat_a1a2", "swat_a1a2_normalonly"}

PA_K_VALUES = [10, 20, 50, 80, 100]

# Scoring type → metadata key mapping
SCORING_METRICS_KEY = {
    "combined": "metrics",
    "total": "metrics",
    "teacher": "teacher_recon_metrics",
    "teacher_recon": "teacher_recon_metrics",
    "disc": "disc_metrics",
}


# ============================================================
# Metric Loading
# ============================================================

def load_mae_metrics(exp_dir: Path, scoring_type: str = "combined") -> dict:
    """Load MAE metrics from experiment_metadata.json.

    Args:
        exp_dir: Experiment directory containing experiment_metadata.json
        scoring_type: "combined"/"total" → metrics, "teacher"/"teacher_recon" → teacher_recon_metrics,
                      "disc" → disc_metrics

    Returns:
        Metrics dict in results.json format, or None if not found
    """
    metadata_path = exp_dir / "experiment_metadata.json"
    if not metadata_path.exists():
        print(f"  WARNING: {metadata_path} not found")
        return None

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    metrics_key = SCORING_METRICS_KEY.get(scoring_type, "metrics")
    raw_metrics = metadata.get(metrics_key, {})

    if not raw_metrics:
        # Fallback: try "metrics" key
        if metrics_key != "metrics":
            print(f"  WARNING: {metrics_key} not found, falling back to 'metrics'")
            raw_metrics = metadata.get("metrics", {})
        if not raw_metrics:
            print(f"  WARNING: No metrics found for scoring_type={scoring_type}")
            return None

    result = {
        "point_level": {
            "prc_auc": raw_metrics.get("prc_auc", 0.0),
            "roc_auc": raw_metrics.get("roc_auc", 0.0),
            "f1_score": raw_metrics.get("f1_score", 0.0),
            "recall": raw_metrics.get("recall", 0.0),
            "precision": raw_metrics.get("precision", 0.0),
        },
        "f1_t": {
            "f1_t": raw_metrics.get("f1_t", raw_metrics.get("pa_0_f1", raw_metrics.get("f1_score", 0.0))),
            "precision_t": raw_metrics.get("precision_t", raw_metrics.get("precision", 0.0)),
            "recall_t": raw_metrics.get("recall_t", raw_metrics.get("recall", 0.0)),
        },
        "pa_k": {}
    }

    for k in PA_K_VALUES:
        pa_key = f"pa_{k}"
        result["pa_k"][pa_key] = {
            "prc_auc": raw_metrics.get(f"pa_{k}_prc_auc", 0.0),
            "roc_auc": raw_metrics.get(f"pa_{k}_roc_auc", 0.0),
            "f1_score": raw_metrics.get(f"pa_{k}_f1", 0.0),
            "recall": raw_metrics.get(f"pa_{k}_recall", 0.0),
            "precision": raw_metrics.get(f"pa_{k}_precision", 0.0),
        }

    result["timing"] = {
        "train_time": metadata.get("train_time", 0.0),
        "inference_time": metadata.get("inference_time", 0.0),
    }

    result["mae_info"] = {
        "config_name": exp_dir.name,
        "scoring_type": scoring_type,
        "source_dir": str(exp_dir),
    }

    return result


def load_excl22_metrics(mae_source_dir: Path, config_name: str,
                        scoring_type: str = "combined") -> dict:
    """Load excl22 metrics from eval_excl_region21.json.

    excl22 = exclude largest test anomaly region.
    File produced by scripts/eval_exclude_region21.py which:
    1. Removes windows overlapping the largest region
    2. Recomputes adaptive_lambda from clean windows
    3. Recomputes scores with new lambda
    4. Aggregates to point-level from clean windows
    5. Computes threshold and PRC fresh

    Args:
        mae_source_dir: Base directory containing eval_excl_region21.json
        config_name: Experiment directory name to match
        scoring_type: "combined" → excl_metrics, "teacher" → excl_recon_metrics

    Returns:
        Excl metrics dict, or None
    """
    excl_path = mae_source_dir / "eval_excl_region21.json"
    if not excl_path.exists():
        print(f"  WARNING: {excl_path} not found")
        print(f"  Run: python scripts/eval_exclude_region21.py")
        return None

    with open(excl_path, 'r') as f:
        all_excl = json.load(f)

    match = next((e for e in all_excl if e["exp_name"] == config_name), None)
    if match is None:
        print(f"  WARNING: Config {config_name} not found in eval_excl_region21.json")
        return None

    # Select metrics key based on scoring type
    if scoring_type in ("teacher", "teacher_recon"):
        em = match.get("excl_recon_metrics", {})
    else:
        em = match.get("excl_metrics", {})

    if not em:
        print(f"  WARNING: No excl metrics for scoring_type={scoring_type}")
        return None

    result = {
        "point_level": {
            "prc_auc": em.get("prc_auc", 0.0),
            "roc_auc": em.get("roc_auc", 0.0),
            "f1_score": em.get("f1", em.get("f1_score", 0.0)),
            "recall": em.get("recall", 0.0),
            "precision": em.get("precision", 0.0),
        },
        "f1_t": {
            "f1_t": em.get("f1_t", 0.0),
            "precision_t": em.get("precision_t", 0.0),
            "recall_t": em.get("recall_t", 0.0),
        },
        "pa_k": {},
    }

    # Only combined scoring has adaptive_lambda and window info
    if scoring_type in ("combined", "total"):
        result["adaptive_lambda"] = {
            "full": match.get("full_lambda", 0.0),
            "excl": match.get("excl_lambda", 0.0),
        }
        result["windows"] = {
            "total": match.get("n_windows_total", 0),
            "removed": match.get("n_windows_removed", 0),
            "kept": match.get("n_windows_kept", 0),
        }

    return result


# ============================================================
# Resolve target results directory
# ============================================================

def _get_target_results_dir(experiment: str) -> Path:
    """Get target comparison results directory for an experiment."""
    try:
        from comparison.experiment_configs import EXPERIMENT_CONFIGS
        exp_config = EXPERIMENT_CONFIGS.get(experiment, {})
        results_dir_name = exp_config.get('results_dir_name', experiment)
    except ImportError:
        results_dir_name = experiment

    return PROJECT_ROOT / "comparison" / "results" / results_dir_name


def _load_or_create_results(results_path: Path, experiment_name: str) -> dict:
    """Load existing results.json or create a new one."""
    if results_path.exists():
        with open(results_path, 'r') as f:
            results = json.load(f)
        # Handle old format without "models" key
        if "models" not in results:
            results = {
                "experiment": experiment_name,
                "dataset": results.get("dataset", {}),
                "timestamp": datetime.now().isoformat(),
                "models": results,
            }
        return results

    print(f"WARNING: {results_path} not found. Creating new results.json")
    return {
        "experiment": experiment_name,
        "dataset": {},
        "timestamp": datetime.now().isoformat(),
        "models": {},
    }


# ============================================================
# Discover Mode
# ============================================================

def discover_configs(experiment: str, mae_source_dir: Path):
    """Discover all MAE configs and show their metrics."""
    if not mae_source_dir.exists():
        print(f"No results found at {mae_source_dir}")
        return

    print(f"\n{'='*60}")
    print(f"Available MAE Configs for {experiment}")
    print(f"{'='*60}")

    # Load excl22 data if available
    excl_data = {}
    if experiment in EXCL22_EXPERIMENTS:
        excl_path = mae_source_dir / "eval_excl_region21.json"
        if excl_path.exists():
            with open(excl_path, 'r') as f:
                for entry in json.load(f):
                    excl_data[entry["exp_name"]] = entry

    configs = []
    for exp_dir in mae_source_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        metadata_path = exp_dir / "experiment_metadata.json"
        if not metadata_path.exists():
            continue

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        metrics = metadata.get("metrics", {})
        teacher_metrics = metadata.get("teacher_recon_metrics", {})
        excl = excl_data.get(exp_dir.name, {})
        excl_metrics = excl.get("excl_metrics", {})

        entry = {
            "config": exp_dir.name,
            "combined_roc": metrics.get("roc_auc", 0.0),
            "combined_prc": metrics.get("prc_auc", 0.0),
            "combined_f1t": metrics.get("f1_t", 0.0),
            "teacher_roc": teacher_metrics.get("roc_auc", 0.0),
            "teacher_prc": teacher_metrics.get("prc_auc", 0.0),
            "teacher_f1t": teacher_metrics.get("f1_t", 0.0),
        }

        if excl_metrics:
            entry["excl_prc"] = excl_metrics.get("prc_auc", 0.0)
            entry["excl_f1t"] = excl_metrics.get("f1_t", 0.0)

        configs.append(entry)

    configs.sort(key=lambda x: x["combined_prc"], reverse=True)

    has_excl = any("excl_prc" in c for c in configs)

    if has_excl:
        print(f"\n{'Config':<40} {'Comb_PRC':>9} {'Comb_F1T':>9} {'Teach_PRC':>10} "
              f"{'Excl_PRC':>9} {'Excl_F1T':>9}")
        print("-" * 100)
        for cfg in configs:
            print(f"{cfg['config']:<40} {cfg['combined_prc']:>9.4f} {cfg['combined_f1t']:>9.4f} "
                  f"{cfg['teacher_prc']:>10.4f} {cfg.get('excl_prc', 0):>9.4f} "
                  f"{cfg.get('excl_f1t', 0):>9.4f}")
    else:
        print(f"\n{'Config':<45} {'C_ROC':>7} {'C_PRC':>7} {'C_F1T':>7} "
              f"{'T_ROC':>7} {'T_PRC':>7} {'T_F1T':>7}")
        print("-" * 95)
        for cfg in configs:
            print(f"{cfg['config']:<45} {cfg['combined_roc']:>7.4f} {cfg['combined_prc']:>7.4f} "
                  f"{cfg['combined_f1t']:>7.4f} {cfg['teacher_roc']:>7.4f} "
                  f"{cfg['teacher_prc']:>7.4f} {cfg['teacher_f1t']:>7.4f}")

    print("-" * (100 if has_excl else 95))
    print(f"Total configs: {len(configs)}")


# ============================================================
# Add MAE Results
# ============================================================

def add_mae_results(experiment: str, mae_source_dir: Path):
    """Add MAE results to comparison results.json.

    Uses hardcoded BEST_MAE_CONFIGS if available, otherwise auto-discovers best by PRC.
    """
    target_results_dir = _get_target_results_dir(experiment)
    results_path = target_results_dir / "results.json"
    results = _load_or_create_results(results_path, experiment)
    has_excl = experiment in EXCL22_EXPERIMENTS

    print(f"\n{'='*60}")
    print(f"Adding MAE Results to {experiment}")
    print(f"{'='*60}")

    best_configs = BEST_MAE_CONFIGS.get(experiment)

    if best_configs:
        _add_from_hardcoded(results, mae_source_dir, best_configs, has_excl)
    else:
        _add_from_auto_discover(results, mae_source_dir, has_excl)

    results["timestamp"] = datetime.now().isoformat()

    target_results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {results_path}")

    _print_summary(results, has_excl)


def _add_from_hardcoded(results: dict, mae_source_dir: Path, best_configs: dict,
                        has_excl: bool):
    """Add MAE results using hardcoded best config names."""
    # Add combined scoring MAE configs
    for cfg in best_configs.get("combined", []):
        config_name = cfg["config"]
        model_name = cfg["name"]
        exp_dir = mae_source_dir / config_name

        print(f"\nAdding {model_name} (combined scoring)...")
        metrics = load_mae_metrics(exp_dir, "combined")
        if metrics:
            if has_excl:
                excl_metrics = load_excl22_metrics(mae_source_dir, config_name, "combined")
                if excl_metrics:
                    metrics["excl22"] = excl_metrics

            results["models"][model_name] = metrics
            print(f"  ROC-AUC: {metrics['point_level']['roc_auc']:.4f}")
            print(f"  PRC-AUC: {metrics['point_level']['prc_auc']:.4f}")
            print(f"  F1_T: {metrics['f1_t']['f1_t']:.4f}")
            if has_excl and "excl22" in metrics:
                print(f"  [Excl22] PRC-AUC: {metrics['excl22']['point_level']['prc_auc']:.4f}")

    # Add teacher-only scoring MAE configs
    for cfg in best_configs.get("teacher", []):
        config_name = cfg["config"]
        model_name = cfg["name"]
        exp_dir = mae_source_dir / config_name

        print(f"\nAdding {model_name} (teacher scoring)...")
        metrics = load_mae_metrics(exp_dir, "teacher")
        if metrics:
            if has_excl:
                excl_metrics = load_excl22_metrics(mae_source_dir, config_name, "teacher")
                if excl_metrics:
                    metrics["excl22"] = excl_metrics

            results["models"][model_name] = metrics
            print(f"  ROC-AUC: {metrics['point_level']['roc_auc']:.4f}")
            print(f"  PRC-AUC: {metrics['point_level']['prc_auc']:.4f}")
            print(f"  F1_T: {metrics['f1_t']['f1_t']:.4f}")


def _add_from_auto_discover(results: dict, mae_source_dir: Path, has_excl: bool):
    """Auto-discover best MAE config by combined PRC and add both scoring types."""
    if not mae_source_dir.exists():
        print(f"No MAE results found at {mae_source_dir}")
        return

    exp_dirs = [d for d in mae_source_dir.iterdir()
                if d.is_dir() and (d / "experiment_metadata.json").exists()]

    if not exp_dirs:
        print(f"No MAE experiment results found in {mae_source_dir}")
        return

    # Find best by combined PRC
    best_dir = None
    best_prc = -1
    for d in exp_dirs:
        with open(d / "experiment_metadata.json", 'r') as f:
            m = json.load(f)
        prc = m.get("metrics", {}).get("prc_auc", 0)
        if prc > best_prc:
            best_prc = prc
            best_dir = d

    print(f"  Best config: {best_dir.name} (PRC={best_prc:.4f})")

    # Add combined scoring
    combined_metrics = load_mae_metrics(best_dir, "combined")
    if combined_metrics:
        if has_excl:
            excl_metrics = load_excl22_metrics(mae_source_dir, best_dir.name, "combined")
            if excl_metrics:
                combined_metrics["excl22"] = excl_metrics

        results["models"]["mae_adaptive"] = combined_metrics
        print(f"  mae_adaptive: ROC={combined_metrics['point_level']['roc_auc']:.4f} "
              f"PRC={combined_metrics['point_level']['prc_auc']:.4f}")

    # Add teacher-only scoring
    teacher_metrics = load_mae_metrics(best_dir, "teacher")
    if teacher_metrics:
        if has_excl:
            excl_metrics = load_excl22_metrics(mae_source_dir, best_dir.name, "teacher")
            if excl_metrics:
                teacher_metrics["excl22"] = excl_metrics

        results["models"]["mae_teacheronly"] = teacher_metrics
        print(f"  mae_teacheronly: ROC={teacher_metrics['point_level']['roc_auc']:.4f} "
              f"PRC={teacher_metrics['point_level']['prc_auc']:.4f}")


def add_mae_manual(experiment: str, mae_dir: Path, model_name: str, scoring_type: str):
    """Manually add a single MAE config to results."""
    target_results_dir = _get_target_results_dir(experiment)
    results_path = target_results_dir / "results.json"
    results = _load_or_create_results(results_path, experiment)
    has_excl = experiment in EXCL22_EXPERIMENTS

    print(f"\n{'='*60}")
    print(f"Adding {model_name} ({scoring_type}) to {experiment}")
    print(f"{'='*60}")

    metrics = load_mae_metrics(mae_dir, scoring_type)
    if metrics:
        if has_excl:
            mae_source_dir = mae_dir.parent
            excl_metrics = load_excl22_metrics(mae_source_dir, mae_dir.name, scoring_type)
            if excl_metrics:
                metrics["excl22"] = excl_metrics

        results["models"][model_name] = metrics
        results["timestamp"] = datetime.now().isoformat()

        target_results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"  ROC-AUC: {metrics['point_level']['roc_auc']:.4f}")
        print(f"  PRC-AUC: {metrics['point_level']['prc_auc']:.4f}")
        print(f"  F1_T: {metrics['f1_t']['f1_t']:.4f}")
        print(f"\nAdded {model_name} to {results_path}")
    else:
        print("  ERROR: Could not load metrics")


# ============================================================
# Summary Printing
# ============================================================

def _print_summary(results: dict, has_excl: bool = False):
    """Print results summary table sorted by PRC-AUC."""
    print(f"\n{'='*90}")
    print("UPDATED RESULTS SUMMARY (sorted by PRC-AUC)")
    print("=" * 90)

    sorted_models = sorted(results["models"].items(),
                          key=lambda x: x[1]['point_level']['prc_auc'],
                          reverse=True)

    if has_excl:
        print(f"\n{'Model':<35} {'ROC':>7} {'PRC':>7} {'F1_T':>7} | "
              f"{'ROC_ex':>7} {'PRC_ex':>7} {'F1T_ex':>7}")
        print("-" * 90)

        for model_name, metrics in sorted_models:
            pl = metrics['point_level']
            f1t = metrics['f1_t']['f1_t']

            excl = metrics.get('excl22', {})
            epl = excl.get('point_level', {})
            ef1t = excl.get('f1_t', {}).get('f1_t', 0.0)

            excl_str = (f"{epl.get('roc_auc', 0.0):>7.4f} {epl.get('prc_auc', 0.0):>7.4f} "
                        f"{ef1t:>7.4f}") if epl else "    -       -       -"

            print(f"{model_name:<35} {pl['roc_auc']:>7.4f} {pl['prc_auc']:>7.4f} "
                  f"{f1t:>7.4f} | {excl_str}")

        print("-" * 90)
    else:
        print(f"\n{'Model':<35} {'ROC-AUC':>10} {'PRC-AUC':>10} {'F1_T':>10}")
        print("-" * 70)

        for model_name, metrics in sorted_models:
            pl = metrics['point_level']
            f1t = metrics['f1_t']['f1_t']
            print(f"{model_name:<35} {pl['roc_auc']:>10.4f} {pl['prc_auc']:>10.4f} "
                  f"{f1t:>10.4f}")

        print("-" * 70)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Add MAE results to comparison results.json',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python comparison/add_mae_results.py --experiment wadi_14days_A1
  python comparison/add_mae_results.py --experiment swat_a1a2
  python comparison/add_mae_results.py --experiment swat_a1a2_normal50 --discover
  python comparison/add_mae_results.py --experiment wadi_A1 \\
      --mae-dir results/WaDi/A1/20260202_222231 --name mae_custom --scoring combined

Scoring types:
  combined/total    → metrics (adaptive/total_loss scoring)
  teacher/teacher_recon → teacher_recon_metrics (reconstruction only)
  disc              → disc_metrics (discrepancy only, legacy)
        """
    )

    parser.add_argument('--experiment', '-e', type=str, required=True,
                       help='Experiment name (e.g., wadi_14days_A1, swat_a1a2)')
    parser.add_argument('--discover', action='store_true',
                       help='Discover and show available MAE configs')

    # Manual mode args
    parser.add_argument('--mae-dir', type=str, default=None,
                       help='Manual: path to specific MAE experiment directory')
    parser.add_argument('--name', type=str, default=None,
                       help='Manual: model name in results.json')
    parser.add_argument('--scoring', type=str, default='combined',
                       choices=['combined', 'total', 'teacher', 'teacher_recon', 'disc'],
                       help='Scoring type (default: combined)')

    args = parser.parse_args()

    # Resolve MAE source directory
    mae_source_rel = MAE_SOURCE_DIRS.get(args.experiment)
    if mae_source_rel:
        mae_source_dir = PROJECT_ROOT / mae_source_rel
    elif args.mae_dir:
        # For manual mode with unknown experiment, mae-dir is required
        mae_source_dir = None  # Will use mae_dir directly
    else:
        print(f"ERROR: No known MAE source for experiment '{args.experiment}'")
        print("Use --mae-dir to specify the MAE results directory")
        print(f"\nKnown experiments: {', '.join(sorted(MAE_SOURCE_DIRS.keys()))}")
        return

    if args.discover:
        if mae_source_dir is None:
            print("ERROR: Cannot discover without a known MAE source directory")
            return
        discover_configs(args.experiment, mae_source_dir)
    elif args.mae_dir:
        # Manual mode
        if not args.name:
            print("ERROR: --name is required when using --mae-dir")
            return
        mae_dir = Path(args.mae_dir)
        if not mae_dir.is_absolute():
            mae_dir = PROJECT_ROOT / mae_dir
        add_mae_manual(args.experiment, mae_dir, args.name, args.scoring)
    else:
        # Auto mode (hardcoded or discover best)
        if mae_source_dir is None:
            print("ERROR: No known MAE source. Use --mae-dir for manual mode")
            return
        add_mae_results(args.experiment, mae_source_dir)


if __name__ == "__main__":
    main()
