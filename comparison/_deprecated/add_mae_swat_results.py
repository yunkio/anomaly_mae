#!/usr/bin/env python
"""
Add MAE Results to SWaT Comparison

This script adds MAE model results from the ablation study to the comparison results.
It extracts MAE configurations and adds them to results.json, including excl-R#21 metrics.

Data sources:
- Normal metrics: results/SWaT/A1A2/{config}/experiment_metadata.json
- Excl-R#21 metrics: results/SWaT/A1A2/eval_excl_region21.json
  (produced by scripts/eval_exclude_region21.py)

Usage:
    conda activate dc_vis
    python comparison/add_mae_swat_results.py
    python comparison/add_mae_swat_results.py --discover
    python comparison/add_mae_swat_results.py --all
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np


# Best MAE configurations (to be updated after experiments complete)
# Config dir name → display name
BEST_MAE_CONFIGS = {
    # Combined scoring (recon + adaptive_lambda * disc)
    "combined": [
        {"config": "20260212_005359_w500_p5_td4_sd2", "name": "mae_w500_p5_td4_sd2"},
        {"config": "20260211_231332_w500_p5_default", "name": "mae_w500_p5_default"},
    ],
    # Teacher-only scoring (recon only)
    "teacher": [
        {"config": "20260212_005359_w500_p5_td4_sd2", "name": "mae_teacher_w500_p5_td4_sd2"},
    ],
}

# PA%K values matching the comparison framework
PA_K_VALUES = [10, 20, 50, 80, 100]


def load_mae_metrics(config_name: str, scoring_type: str = "combined") -> dict:
    """
    Load MAE metrics from experiment_metadata.json.

    Args:
        config_name: Experiment directory name under results/SWaT/
        scoring_type: "combined" or "teacher"

    Returns:
        Metrics dictionary in results.json format, or None if not found
    """
    mae_dir = PROJECT_ROOT / "results" / "SWaT" / "A1A2" / config_name
    metadata_path = mae_dir / "experiment_metadata.json"

    if not metadata_path.exists():
        print(f"  WARNING: {metadata_path} not found")
        return None

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Select metrics based on scoring type
    if scoring_type == "teacher":
        raw_metrics = metadata.get("teacher_recon_metrics", {})
    else:
        raw_metrics = metadata.get("metrics", {})

    if not raw_metrics:
        print(f"  WARNING: No metrics found for scoring_type={scoring_type}")
        return None

    # Convert to results.json format
    result = {
        "point_level": {
            "prc_auc": raw_metrics.get("prc_auc", 0.0),
            "roc_auc": raw_metrics.get("roc_auc", 0.0),
            "f1_score": raw_metrics.get("f1_score", 0.0),
            "recall": raw_metrics.get("recall", 0.0),
            "precision": raw_metrics.get("precision", 0.0),
        },
        "f1_t": {
            "f1_t": raw_metrics.get("f1_t", 0.0),
            "precision_t": raw_metrics.get("precision_t", 0.0),
            "recall_t": raw_metrics.get("recall_t", 0.0),
        },
        "pa_k": {}
    }

    # PA%K metrics
    for k in PA_K_VALUES:
        pa_key = f"pa_{k}"
        result["pa_k"][pa_key] = {
            "prc_auc": raw_metrics.get(f"pa_{k}_prc_auc", 0.0),
            "roc_auc": raw_metrics.get(f"pa_{k}_roc_auc", 0.0),
            "f1_score": raw_metrics.get(f"pa_{k}_f1", 0.0),
            "recall": raw_metrics.get(f"pa_{k}_recall", 0.0),
            "precision": raw_metrics.get(f"pa_{k}_precision", 0.0),
        }

    # Timing
    result["timing"] = {
        "train_time": metadata.get("train_time", 0.0),
        "inference_time": metadata.get("inference_time", 0.0),
    }

    # MAE-specific info
    result["mae_info"] = {
        "config_name": config_name,
        "scoring_type": scoring_type,
        "source_dir": str(mae_dir),
    }

    return result


def load_excl_r21_metrics(config_name: str) -> dict:
    """
    Load excl-R#21 metrics from eval_excl_region21.json.

    This file is produced by scripts/eval_exclude_region21.py which properly:
    1. Removes windows overlapping R#21
    2. Recomputes adaptive_lambda from clean windows
    3. Recomputes scores with new lambda
    4. Aggregates to point-level from clean windows
    5. Computes threshold and PRC fresh

    Args:
        config_name: Experiment directory name under results/SWaT/

    Returns:
        Excl metrics dict in results.json format, or None
    """
    excl_path = PROJECT_ROOT / "results" / "SWaT" / "A1A2" / "eval_excl_region21.json"

    if not excl_path.exists():
        print(f"  WARNING: {excl_path} not found")
        print(f"  Run: python scripts/eval_exclude_region21.py")
        return None

    with open(excl_path, 'r') as f:
        all_excl = json.load(f)

    # Find matching experiment
    match = None
    for entry in all_excl:
        if entry["exp_name"] == config_name:
            match = entry
            break

    if match is None:
        print(f"  WARNING: Config {config_name} not found in eval_excl_region21.json")
        return None

    em = match["excl_metrics"]

    # Map to results.json format
    # Note: eval_exclude_region21.py doesn't compute PA%K for excl data
    result = {
        "point_level": {
            "prc_auc": em.get("prc_auc", 0.0),
            "roc_auc": em.get("roc_auc", 0.0),
            "f1_score": em.get("f1", 0.0),
            "recall": em.get("recall", 0.0),
            "precision": em.get("precision", 0.0),
        },
        "f1_t": {
            "f1_t": em.get("f1_t", 0.0),
            "precision_t": em.get("precision_t", 0.0),
            "recall_t": em.get("recall_t", 0.0),
        },
        "pa_k": {},
        "adaptive_lambda": {
            "full": match.get("full_lambda", 0.0),
            "excl": match.get("excl_lambda", 0.0),
        },
        "windows": {
            "total": match.get("n_windows_total", 0),
            "removed": match.get("n_windows_removed", 0),
            "kept": match.get("n_windows_kept", 0),
        },
    }

    return result


def add_mae_to_results():
    """Add MAE results to comparison results.json."""
    results_dir = PROJECT_ROOT / "comparison" / "results" / "SWaT_A1A2"
    results_path = results_dir / "results.json"

    # Load existing results
    if results_path.exists():
        with open(results_path, 'r') as f:
            results = json.load(f)
    else:
        print(f"WARNING: {results_path} not found. Creating new results.json")
        results = {
            "experiment": "SWaT",
            "dataset": {},
            "timestamp": datetime.now().isoformat(),
            "models": {},
        }

    print(f"\n{'='*60}")
    print(f"Adding MAE Results to SWaT Comparison")
    print(f"{'='*60}")

    # Add combined scoring MAE configs
    for cfg in BEST_MAE_CONFIGS.get("combined", []):
        config_name = cfg["config"]
        model_name = cfg["name"]

        print(f"\nAdding {model_name} (combined scoring)...")
        metrics = load_mae_metrics(config_name, "combined")
        if metrics:
            # Load excl-R#21 metrics
            excl_metrics = load_excl_r21_metrics(config_name)
            if excl_metrics:
                metrics["excl_r21"] = excl_metrics

            results["models"][model_name] = metrics
            print(f"  ROC-AUC: {metrics['point_level']['roc_auc']:.4f}")
            print(f"  PRC-AUC: {metrics['point_level']['prc_auc']:.4f}")
            print(f"  F1_T: {metrics['f1_t']['f1_t']:.4f}")
            if excl_metrics:
                print(f"  [Excl-R#21] PRC-AUC: {excl_metrics['point_level']['prc_auc']:.4f}")

    # Add teacher-only scoring MAE configs
    for cfg in BEST_MAE_CONFIGS.get("teacher", []):
        config_name = cfg["config"]
        model_name = cfg["name"]

        print(f"\nAdding {model_name} (teacher scoring)...")
        metrics = load_mae_metrics(config_name, "teacher")
        if metrics:
            # Note: excl-R#21 for teacher scoring comes from excl_recon_metrics
            excl_path = PROJECT_ROOT / "results" / "SWaT" / "A1A2" / "eval_excl_region21.json"
            if excl_path.exists():
                with open(excl_path, 'r') as f:
                    all_excl = json.load(f)
                match = next((e for e in all_excl if e["exp_name"] == config_name), None)
                if match and "excl_recon_metrics" in match:
                    rm = match["excl_recon_metrics"]
                    metrics["excl_r21"] = {
                        "point_level": {
                            "prc_auc": rm.get("prc_auc", 0.0),
                            "roc_auc": rm.get("roc_auc", 0.0),
                            "f1_score": rm.get("f1", 0.0),
                            "recall": rm.get("recall", 0.0),
                            "precision": rm.get("precision", 0.0),
                        },
                        "f1_t": {
                            "f1_t": rm.get("f1_t", 0.0),
                            "precision_t": rm.get("precision_t", 0.0),
                            "recall_t": rm.get("recall_t", 0.0),
                        },
                        "pa_k": {},
                    }

            results["models"][model_name] = metrics
            print(f"  ROC-AUC: {metrics['point_level']['roc_auc']:.4f}")
            print(f"  PRC-AUC: {metrics['point_level']['prc_auc']:.4f}")
            print(f"  F1_T: {metrics['f1_t']['f1_t']:.4f}")

    # Update timestamp
    results["timestamp"] = datetime.now().isoformat()

    # Save
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {results_path}")

    # Print summary
    _print_summary(results)


def _print_summary(results: dict):
    """Print results summary table."""
    print("\n" + "=" * 90)
    print("UPDATED RESULTS SUMMARY (sorted by PRC-AUC)")
    print("=" * 90)

    print(f"\n{'Model':<35} {'ROC':>7} {'PRC':>7} {'F1_T':>7} | {'ROC_ex':>7} {'PRC_ex':>7} {'F1T_ex':>7}")
    print("-" * 90)

    sorted_models = sorted(results["models"].items(),
                          key=lambda x: x[1]['point_level']['prc_auc'],
                          reverse=True)

    for model_name, metrics in sorted_models:
        pl = metrics['point_level']
        f1t = metrics['f1_t']['f1_t']

        excl = metrics.get('excl_r21', {})
        epl = excl.get('point_level', {})
        ef1t = excl.get('f1_t', {}).get('f1_t', 0.0)

        excl_str = (f"{epl.get('roc_auc', 0.0):>7.4f} {epl.get('prc_auc', 0.0):>7.4f} "
                    f"{ef1t:>7.4f}") if epl else "    -       -       -"

        print(f"{model_name:<35} {pl['roc_auc']:>7.4f} {pl['prc_auc']:>7.4f} "
              f"{f1t:>7.4f} | {excl_str}")

    print("-" * 90)


def discover_mae_configs():
    """Discover all MAE configs and show their metrics."""
    mae_base_dir = PROJECT_ROOT / "results" / "SWaT" / "A1A2"

    if not mae_base_dir.exists():
        print(f"No MAE results found at {mae_base_dir}")
        return

    print(f"\n{'='*60}")
    print(f"Available MAE Configs for SWaT A1A2")
    print(f"{'='*60}")

    # Load excl-R#21 data if available
    excl_data = {}
    excl_path = mae_base_dir / "eval_excl_region21.json"
    if excl_path.exists():
        with open(excl_path, 'r') as f:
            for entry in json.load(f):
                excl_data[entry["exp_name"]] = entry

    configs = []
    for exp_dir in mae_base_dir.iterdir():
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

        configs.append({
            "config": exp_dir.name,
            "combined_prc": metrics.get("prc_auc", 0.0),
            "combined_f1t": metrics.get("f1_t", 0.0),
            "teacher_prc": teacher_metrics.get("prc_auc", 0.0),
            "excl_prc": excl_metrics.get("prc_auc", 0.0),
            "excl_f1t": excl_metrics.get("f1_t", 0.0),
        })

    configs.sort(key=lambda x: x["combined_prc"], reverse=True)

    print(f"\n{'Config':<40} {'Comb_PRC':>9} {'Comb_F1T':>9} {'Teach_PRC':>10} "
          f"{'Excl_PRC':>9} {'Excl_F1T':>9}")
    print("-" * 100)

    for cfg in configs:
        print(f"{cfg['config']:<40} {cfg['combined_prc']:>9.4f} {cfg['combined_f1t']:>9.4f} "
              f"{cfg['teacher_prc']:>10.4f} {cfg['excl_prc']:>9.4f} {cfg['excl_f1t']:>9.4f}")

    print("-" * 100)
    print(f"Total configs: {len(configs)}")


def main():
    parser = argparse.ArgumentParser(description='Add MAE results to SWaT comparison')
    parser.add_argument('--discover', action='store_true',
                       help='Discover and show available MAE configs')
    parser.add_argument('--all', action='store_true',
                       help='Add all available MAE configs (not just best)')
    args = parser.parse_args()

    if args.discover:
        discover_mae_configs()
    else:
        add_mae_to_results()


if __name__ == "__main__":
    main()
