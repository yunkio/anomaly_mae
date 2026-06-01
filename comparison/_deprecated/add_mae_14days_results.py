#!/usr/bin/env python
"""
Add MAE Results to WaDi 14days Comparison

This script adds MAE model results from the ablation study to the comparison results.
It extracts the best performing MAE configurations and adds them to results.json.

Best MAE configs (from ablation analysis):
- A1_14days: w500_p5_td4_sd2 (Combined PRC: 0.6501)
- A2_14days: w500_p20_enc3 (Combined PRC: 0.6268)

For teacher-only scoring, the best configs may differ.

Usage:
    conda activate dc_vis
    python comparison/add_mae_14days_results.py --scenario A1
    python comparison/add_mae_14days_results.py --scenario A2
    python comparison/add_mae_14days_results.py --scenario A1 --all-top5
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd


# Best MAE configurations from ablation analysis (by PRC)
BEST_MAE_CONFIGS = {
    "A1": {
        # Top configs by Combined PRC
        "combined": [
            {"config": "20260208_080838_w500_p5_td4_sd2", "name": "mae_w500_p5_td4_sd2"},
        ],
        # Top configs by Teacher PRC
        "teacher": [
            {"config": "20260208_135908_w500_p10_td3_sd1", "name": "mae_teacher_w500_p10_td3_sd1"},
        ],
    },
    "A2": {
        "combined": [
            {"config": "20260209_113528_w500_p20_enc3", "name": "mae_w500_p20_enc3"},
        ],
        "teacher": [
            {"config": "20260209_113528_w500_p20_enc3", "name": "mae_teacher_w500_p20_enc3"},
        ],
    },
}


def load_mae_metrics(scenario: str, config_name: str, scoring_type: str = "combined") -> dict:
    """
    Load MAE metrics from experiment_metadata.json.

    Args:
        scenario: "A1" or "A2"
        config_name: Experiment directory name
        scoring_type: "combined" or "teacher"

    Returns:
        Metrics dictionary in results.json format
    """
    mae_dir = PROJECT_ROOT / "results" / "WaDi" / f"{scenario}_14days" / config_name
    metadata_path = mae_dir / "experiment_metadata.json"

    if not metadata_path.exists():
        print(f"  WARNING: {metadata_path} not found")
        return None

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Select metrics based on scoring type
    if scoring_type == "combined":
        metrics_key = "metrics"
    elif scoring_type == "teacher":
        metrics_key = "teacher_recon_metrics"
    else:
        metrics_key = "metrics"

    raw_metrics = metadata.get(metrics_key, {})
    if not raw_metrics:
        print(f"  WARNING: No {metrics_key} found in metadata")
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
            "f1_t": raw_metrics.get("f1_t", raw_metrics.get("f1_score", 0.0)),
            "precision_t": raw_metrics.get("precision", 0.0),
            "recall_t": raw_metrics.get("recall", 0.0),
        },
        "pa_k": {}
    }

    # PA%K metrics
    for k in [10, 20, 50, 80, 100]:
        pa_key = f"pa_{k}"
        result["pa_k"][pa_key] = {
            "prc_auc": raw_metrics.get(f"pa_{k}_prc_auc", 0.0),
            "roc_auc": raw_metrics.get(f"pa_{k}_roc_auc", 0.0),
            "f1_score": raw_metrics.get(f"pa_{k}_f1", 0.0),
            "recall": 0.0,
            "precision": 0.0,
        }

    # Timing (from metadata)
    result["timing"] = {
        "train_time": metadata.get("train_time", 0.0),
        "inference_time": metadata.get("inference_time", 0.0),
    }

    # Additional MAE-specific info
    result["mae_info"] = {
        "config_name": config_name,
        "scoring_type": scoring_type,
        "source_dir": str(mae_dir),
    }

    return result


def add_mae_to_results(scenario: str, add_all_top5: bool = False):
    """Add MAE results to comparison results.json."""
    experiment_name = f"WaDi_{scenario}_14days"
    results_dir = PROJECT_ROOT / "comparison" / "results" / experiment_name
    results_path = results_dir / "results.json"

    # Load existing results
    if results_path.exists():
        with open(results_path, 'r') as f:
            results = json.load(f)
    else:
        print(f"WARNING: {results_path} not found. Creating new results.json")
        results = {
            "experiment": experiment_name,
            "dataset": {},
            "timestamp": datetime.now().isoformat(),
            "models": {},
        }

    print(f"\n{'='*60}")
    print(f"Adding MAE Results to {experiment_name}")
    print(f"{'='*60}")

    # Get best configs for this scenario
    best_configs = BEST_MAE_CONFIGS.get(scenario, {})

    # Add combined scoring MAE
    for cfg in best_configs.get("combined", []):
        config_name = cfg["config"]
        model_name = cfg["name"]

        print(f"\nAdding {model_name} (combined scoring)...")
        metrics = load_mae_metrics(scenario, config_name, "combined")
        if metrics:
            results["models"][model_name] = metrics
            print(f"  ROC-AUC: {metrics['point_level']['roc_auc']:.4f}")
            print(f"  PRC-AUC: {metrics['point_level']['prc_auc']:.4f}")
            print(f"  F1_T: {metrics['f1_t']['f1_t']:.4f}")

    # Add teacher-only scoring MAE
    for cfg in best_configs.get("teacher", []):
        config_name = cfg["config"]
        model_name = cfg["name"]

        print(f"\nAdding {model_name} (teacher scoring)...")
        metrics = load_mae_metrics(scenario, config_name, "teacher")
        if metrics:
            results["models"][model_name] = metrics
            print(f"  ROC-AUC: {metrics['point_level']['roc_auc']:.4f}")
            print(f"  PRC-AUC: {metrics['point_level']['prc_auc']:.4f}")
            print(f"  F1_T: {metrics['f1_t']['f1_t']:.4f}")

    # Update timestamp
    results["timestamp"] = datetime.now().isoformat()

    # Save
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {results_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("UPDATED RESULTS SUMMARY (sorted by PRC-AUC)")
    print("=" * 60)

    print(f"\n{'Model':<35} {'ROC-AUC':>10} {'PRC-AUC':>10} {'F1_T':>10}")
    print("-" * 70)

    sorted_models = sorted(results["models"].items(),
                          key=lambda x: x[1]['point_level']['prc_auc'],
                          reverse=True)

    for model_name, metrics in sorted_models:
        pl = metrics['point_level']
        f1t = metrics['f1_t']['f1_t']
        print(f"{model_name:<35} {pl['roc_auc']:>10.4f} {pl['prc_auc']:>10.4f} {f1t:>10.4f}")

    print("-" * 70)


def discover_and_show_mae_configs(scenario: str):
    """Discover all MAE configs for a scenario and show their metrics."""
    mae_base_dir = PROJECT_ROOT / "results" / "WaDi" / f"{scenario}_14days"

    if not mae_base_dir.exists():
        print(f"No MAE results found at {mae_base_dir}")
        return

    print(f"\n{'='*60}")
    print(f"Available MAE Configs for {scenario}_14days")
    print(f"{'='*60}")

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

        configs.append({
            "config": exp_dir.name,
            "combined_prc": metrics.get("prc_auc", 0.0),
            "combined_f1": metrics.get("f1_t", metrics.get("f1_score", 0.0)),
            "teacher_prc": teacher_metrics.get("prc_auc", 0.0),
            "teacher_f1": teacher_metrics.get("f1_t", teacher_metrics.get("f1_score", 0.0)),
        })

    # Sort by combined PRC
    configs.sort(key=lambda x: x["combined_prc"], reverse=True)

    print(f"\n{'Config':<40} {'Comb_PRC':>10} {'Comb_F1':>10} {'Teach_PRC':>10} {'Teach_F1':>10}")
    print("-" * 90)

    for cfg in configs[:10]:  # Show top 10
        print(f"{cfg['config']:<40} {cfg['combined_prc']:>10.4f} {cfg['combined_f1']:>10.4f} "
              f"{cfg['teacher_prc']:>10.4f} {cfg['teacher_f1']:>10.4f}")

    print("-" * 90)
    print(f"Total configs: {len(configs)}")


def main():
    parser = argparse.ArgumentParser(description='Add MAE results to WaDi 14days comparison')
    parser.add_argument('--scenario', type=str, default='A1', choices=['A1', 'A2'],
                       help='WaDi scenario (A1 or A2)')
    parser.add_argument('--discover', action='store_true',
                       help='Discover and show available MAE configs')
    parser.add_argument('--all-top5', action='store_true',
                       help='Add top 5 MAE configs instead of just best')
    args = parser.parse_args()

    if args.discover:
        discover_and_show_mae_configs(args.scenario)
    else:
        add_mae_to_results(args.scenario, args.all_top5)


if __name__ == "__main__":
    main()
