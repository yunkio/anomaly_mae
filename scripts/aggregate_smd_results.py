#!/usr/bin/env python3
"""Aggregate SMD baseline results across 28 machines.

For each model:
1. Load best-epoch metrics from 28 machines
2. Average across machines
3. Save to SMD/results/{model}/results.csv

Usage:
    python scripts/aggregate_smd_results.py --queue-dir /path/to/Q3_dir
"""

import os
import sys
import argparse
import json
import csv
import numpy as np
from pathlib import Path

# Single source of truth for SMD machine names
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mae_anomaly.datasets.loaders import SMD_MACHINE_NAMES as SMD_MACHINES

ALL_MODELS = [
    'random', 'sensor_range', 'pca_error', 'l2_norm', 'nn_distance',
    'mlp', 'mlpmixer', 'transformer', 'gcn_lstm',
    'anomaly_transformer', 'tranad', 'usad', 'dagmm', 'gdn', 'omnianomaly',
]

# Metrics to exclude from averaging (internal timing etc.)
SKIP_KEYS = {'epoch', '_inference_time', '_eval_time', '_train_time'}


def get_best_epoch_metrics(em_path):
    """Load epoch_metrics.json and return best epoch's metrics dict."""
    try:
        data = json.load(open(em_path))
        epochs = data.get('epochs', [])
        if not epochs:
            return None
        best = max(epochs, key=lambda e: e.get('pak_auc_f1', 0) or 0)
        return best
    except Exception:
        return None


def aggregate(queue_dir):
    """Aggregate results for all models."""
    smd_base = queue_dir / 'SMD'
    results_base = smd_base / 'results'

    for model in ALL_MODELS:
        print(f"\n{'='*50}")
        print(f"  Model: {model}")
        print(f"{'='*50}")

        machine_metrics = []  # List of (machine_name, best_metrics_dict)

        for machine in SMD_MACHINES:
            em = smd_base / machine / model / 'epoch_metrics.json'
            metrics = get_best_epoch_metrics(em)

            if metrics is None:
                print(f"  {machine}: MISSING")
                continue

            machine_metrics.append((machine, metrics))
            print(f"  {machine}: OK (pak_f1={metrics.get('pak_auc_f1', 0):.4f})")

        if not machine_metrics:
            print(f"  NO RESULTS for {model}")
            continue

        # Average across machines
        all_keys = set()
        for _, mm in machine_metrics:
            all_keys.update(mm.keys())

        global_avg = {}
        for key in sorted(all_keys):
            if key in SKIP_KEYS:
                continue
            vals = [mm.get(key) for _, mm in machine_metrics if mm.get(key) is not None]
            if vals and all(isinstance(v, (int, float)) for v in vals):
                global_avg[key] = sum(vals) / len(vals)

        # Save results.csv
        model_dir = results_base / model
        model_dir.mkdir(parents=True, exist_ok=True)
        csv_path = model_dir / 'results.csv'

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            metric_keys = sorted(global_avg.keys())
            writer.writerow(['machine'] + metric_keys)

            # Per-machine rows
            for machine_name, mm in machine_metrics:
                row = [machine_name]
                for key in metric_keys:
                    val = mm.get(key)
                    if val is not None:
                        row.append(f"{val:.6f}" if isinstance(val, float) else str(val))
                    else:
                        row.append("")
                writer.writerow(row)

            # Average row
            avg_row = ['AVERAGE']
            for key in metric_keys:
                val = global_avg.get(key)
                if val is not None:
                    avg_row.append(f"{val:.6f}" if isinstance(val, float) else str(val))
                else:
                    avg_row.append("")
            writer.writerow(avg_row)

        print(f"\n  Saved: {csv_path}")
        print(f"  Machines: {len(machine_metrics)}/{len(SMD_MACHINES)}")
        print(f"  AVERAGE pak_auc_f1: {global_avg.get('pak_auc_f1', 0):.4f}")
        print(f"  AVERAGE pak_auc_prc_auc: {global_avg.get('pak_auc_prc_auc', 0):.4f}")


def main():
    parser = argparse.ArgumentParser(description='Aggregate SMD baseline results')
    parser.add_argument('--queue-dir', type=str, required=True,
                        help='Queue experiment directory (e.g., Q3 dir)')
    args = parser.parse_args()

    queue_dir = Path(args.queue_dir)
    if not queue_dir.exists():
        print(f"ERROR: {queue_dir} does not exist")
        return 1

    aggregate(queue_dir)
    print("\n\nDone!")
    return 0


if __name__ == '__main__':
    exit(main())
