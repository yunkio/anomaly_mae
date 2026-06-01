#!/usr/bin/env python
"""
Add MAE Normal50 Results to WaDi 14days Comparison

Extracts MAE results from results/WaDi/{A1,A2}_14days_normal50/ and adds both
adaptive (anomaly_score) and teacher_recon scoring to comparison results.json.

Usage:
    conda activate dc_vis
    python comparison/add_mae_wadi_14days_normal50_results.py --scenario A1
    python comparison/add_mae_wadi_14days_normal50_results.py --scenario A2
    python comparison/add_mae_wadi_14days_normal50_results.py --scenario A1 --discover
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_mae_metrics(exp_dir: Path, scoring_type: str = "combined") -> dict:
    """Load MAE metrics from experiment_metadata.json."""
    metadata_path = exp_dir / "experiment_metadata.json"
    if not metadata_path.exists():
        print(f"  WARNING: {metadata_path} not found")
        return None

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    if scoring_type == "combined":
        raw_metrics = metadata.get("metrics", {})
    elif scoring_type == "teacher":
        raw_metrics = metadata.get("teacher_recon_metrics", {})
    else:
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
            "f1_t": raw_metrics.get("f1_t", raw_metrics.get("f1_score", 0.0)),
            "precision_t": raw_metrics.get("precision", 0.0),
            "recall_t": raw_metrics.get("recall", 0.0),
        },
        "pa_k": {}
    }

    for k in [10, 20, 50, 80, 100]:
        pa_key = f"pa_{k}"
        result["pa_k"][pa_key] = {
            "prc_auc": raw_metrics.get(f"pa_{k}_prc_auc", 0.0),
            "roc_auc": raw_metrics.get(f"pa_{k}_roc_auc", 0.0),
            "f1_score": raw_metrics.get(f"pa_{k}_f1", 0.0),
            "recall": 0.0,
            "precision": 0.0,
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


def discover_configs(scenario: str):
    """Discover all MAE configs and show their metrics."""
    mae_dir = PROJECT_ROOT / "results" / "WaDi" / f"{scenario}_14days_normal50"
    if not mae_dir.exists():
        print(f"No results found at {mae_dir}")
        return

    print(f"\n{'='*60}")
    print(f"Available MAE Configs for WaDi {scenario} 14days Normal50")
    print(f"{'='*60}")

    configs = []
    for exp_dir in mae_dir.iterdir():
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
            "combined_roc": metrics.get("roc_auc", 0.0),
            "combined_prc": metrics.get("prc_auc", 0.0),
            "combined_f1t": metrics.get("f1_t", 0.0),
            "teacher_roc": teacher_metrics.get("roc_auc", 0.0),
            "teacher_prc": teacher_metrics.get("prc_auc", 0.0),
            "teacher_f1t": teacher_metrics.get("f1_t", 0.0),
        })

    configs.sort(key=lambda x: x["combined_prc"], reverse=True)

    print(f"\n{'Config':<45} {'C_ROC':>7} {'C_PRC':>7} {'C_F1T':>7} {'T_ROC':>7} {'T_PRC':>7} {'T_F1T':>7}")
    print("-" * 95)
    for cfg in configs:
        print(f"{cfg['config']:<45} {cfg['combined_roc']:>7.4f} {cfg['combined_prc']:>7.4f} "
              f"{cfg['combined_f1t']:>7.4f} {cfg['teacher_roc']:>7.4f} {cfg['teacher_prc']:>7.4f} "
              f"{cfg['teacher_f1t']:>7.4f}")
    print(f"\nTotal: {len(configs)} configs")


def add_mae_results(scenario: str):
    """Add MAE results to comparison results.json."""
    experiment_name = f"WaDi_{scenario}_14days_normal50"
    results_dir = PROJECT_ROOT / "comparison" / "results" / experiment_name
    results_path = results_dir / "results.json"
    mae_base_dir = PROJECT_ROOT / "results" / "WaDi" / f"{scenario}_14days_normal50"

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

    exp_dirs = [d for d in mae_base_dir.iterdir()
                if d.is_dir() and (d / "experiment_metadata.json").exists()]

    if not exp_dirs:
        print(f"No MAE experiment results found in {mae_base_dir}")
        return

    # Use the best by combined PRC
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
        results["models"]["mae_adaptive"] = combined_metrics
        print(f"  mae_adaptive: ROC={combined_metrics['point_level']['roc_auc']:.4f} "
              f"PRC={combined_metrics['point_level']['prc_auc']:.4f}")

    # Add teacher-only scoring
    teacher_metrics = load_mae_metrics(best_dir, "teacher")
    if teacher_metrics:
        results["models"]["mae_teacheronly"] = teacher_metrics
        print(f"  mae_teacheronly: ROC={teacher_metrics['point_level']['roc_auc']:.4f} "
              f"PRC={teacher_metrics['point_level']['prc_auc']:.4f}")

    results["timestamp"] = datetime.now().isoformat()

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {results_path}")

    # Summary
    print(f"\n{'Model':<35} {'ROC-AUC':>10} {'PRC-AUC':>10} {'F1_T':>10}")
    print("-" * 70)
    sorted_models = sorted(results["models"].items(),
                          key=lambda x: x[1]['point_level']['prc_auc'], reverse=True)
    for name, m in sorted_models:
        pl = m['point_level']
        print(f"{name:<35} {pl['roc_auc']:>10.4f} {pl['prc_auc']:>10.4f} {m['f1_t']['f1_t']:>10.4f}")


def main():
    parser = argparse.ArgumentParser(description='Add MAE results to WaDi 14days Normal50 comparison')
    parser.add_argument('--scenario', type=str, default='A1', choices=['A1', 'A2'])
    parser.add_argument('--discover', action='store_true')
    args = parser.parse_args()

    if args.discover:
        discover_configs(args.scenario)
    else:
        add_mae_results(args.scenario)


if __name__ == "__main__":
    main()
