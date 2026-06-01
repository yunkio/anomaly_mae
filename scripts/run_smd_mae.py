#!/usr/bin/env python3
"""Add SMD K=6 block split experiments to existing MAE experiment directories.

Reads config from an existing experiment's best_config.json, then runs all 56
SMD datasets (28 machines × 2 parities) using the same config.

This script imports all functionality from run_base_experiments.py — no direct
ML implementation here.

Usage:
    conda activate dc_vis

    # Add SMD results to experiment 102
    python scripts/run_smd_mae.py --experiment-dir results/experiments/102_...

    # Multiple experiments
    python scripts/run_smd_mae.py --experiment-dir results/experiments/102_... results/experiments/114_...

    # Skip completed (already existing) machine/parity dirs
    python scripts/run_smd_mae.py --experiment-dir results/experiments/102_... --skip-existing

    # Aggregate only (no training)
    python scripts/run_smd_mae.py --experiment-dir results/experiments/102_... --aggregate-only

    # Specific machines only
    python scripts/run_smd_mae.py --experiment-dir results/experiments/102_... --machines machine-1-1 machine-1-2

    # Don't wait for background processes (pipeline mode)
    python scripts/run_smd_mae.py --experiment-dir results/experiments/102_... --no-wait
"""

import os
import sys
import json
import time
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Import all functionality from the main experiment script
from scripts.run_base_experiments import (
    run_base_experiment,
    aggregate_smd_results,
    SMD_DATASETS,
    _background_processes,
)
from mae_anomaly.utils import mem_status


def reconstruct_config_preset(experiment_dir):
    """Reconstruct config_preset from an existing experiment directory.

    Reads best_config.json from available sub-experiments, detects dynamic d_model
    by comparing d_model values across datasets, and builds a config_preset dict
    compatible with run_base_experiment().

    Args:
        experiment_dir: Path to existing experiment directory.

    Returns:
        config_preset dict with 'overrides', 'suffix', 'description' keys.
    """
    # Find all available best_config.json files
    config_paths = {}
    for subdir in ['simulation/simulation', 'SWaT/A1A2_full', 'WaDi/A1', 'WaDi/A2']:
        path = os.path.join(experiment_dir, subdir, 'best_config.json')
        if os.path.exists(path):
            config_paths[subdir] = path

    if not config_paths:
        raise FileNotFoundError(
            f"No best_config.json found in {experiment_dir}. "
            "Need at least one existing sub-experiment."
        )

    # Read configs and detect dynamic d_model
    configs = {}
    d_models = set()
    for subdir, path in config_paths.items():
        with open(path) as f:
            configs[subdir] = json.load(f)
        d_models.add(configs[subdir].get('d_model', 128))

    is_dynamic = len(d_models) > 1
    print(f"  d_model values: {d_models} → {'dynamic' if is_dynamic else 'fixed'}")

    # Use the first available config as base
    base_config = next(iter(configs.values()))

    # Build overrides from base_config (all fields)
    overrides = dict(base_config)

    # For dynamic d_model: set to 'dynamic' and let run_base_experiment resolve per-dataset
    # Also remove dim_feedforward so make_config auto-computes it as 4 * d_model
    if is_dynamic:
        overrides['d_model'] = 'dynamic'
        overrides.pop('dim_feedforward', None)

    # Remove dataset-specific fields that will be set by run_base_experiment
    overrides.pop('num_features', None)

    config_preset = {
        'suffix': 'smd_addition',
        'description': f"SMD addition (from {os.path.basename(experiment_dir)})",
        'overrides': overrides,
    }

    print(f"  Config preset reconstructed: d_model={'dynamic' if is_dynamic else overrides['d_model']}, "
          f"patchify={overrides.get('patchify_mode', 'linear')}, "
          f"enc={overrides.get('num_encoder_layers', '?')}, "
          f"disc={'ON' if overrides.get('use_discriminator') else 'OFF'}")

    return config_preset


def run_smd_for_experiment(experiment_dir, config_preset, args):
    """Run all SMD datasets for a single experiment directory.

    Args:
        experiment_dir: Path to existing experiment directory.
        config_preset: Reconstructed config_preset dict.
        args: Parsed command-line arguments.
    """
    # Filter to requested machines
    if args.machines:
        machine_set = set(args.machines)
        datasets = [d for d in SMD_DATASETS
                     if any(m in d['key'] for m in machine_set)]
    else:
        datasets = list(SMD_DATASETS)

    # Filter out already-completed datasets
    if args.skip_existing:
        remaining = []
        for d in datasets:
            result_dir = os.path.join(experiment_dir, d['results_subdir'])
            em_path = os.path.join(result_dir, 'epoch_metrics.json')
            if os.path.exists(em_path):
                print(f"  Skipping {d['key']} (already exists)")
            else:
                remaining.append(d)
        datasets = remaining

    if not datasets:
        print("  No datasets to run (all completed or filtered out)")
        return

    print(f"\n{'='*80}")
    print(f"SMD Experiments: {len(datasets)} datasets")
    print(f"  Experiment dir: {experiment_dir}")
    print(f"  Memory: {mem_status()}")
    print(f"{'='*80}")

    results = []
    total_start = time.time()

    for i, dataset_def in enumerate(datasets):
        print(f"\n{'#'*80}")
        print(f"# [{i+1}/{len(datasets)}] {dataset_def['key']}")
        print(f"{'#'*80}")

        # Wait if too many background processes
        while len([p for _, p in _background_processes if p.is_alive()]) >= 10:
            print(f"  Waiting for background processes... ({mem_status()})")
            time.sleep(10)

        try:
            progress_info = f"[{i+1}/{len(datasets)}]"
            result = run_base_experiment(
                dataset_def, config_preset, experiment_dir, progress_info,
                save_weights=args.save_weights,
            )
            results.append(result)
            pt = result['pure_train_time']
            ee = result['epoch_eval_time']
            print(f"  Completed: {result['key']} (train={pt:.0f}s eval={ee:.0f}s, "
                  f"bestScore={result['best_score']:.4f}@ep{result['best_epoch']}, "
                  f"bestPRC={result['best_prc']:.4f})")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Wait for background processes
    if args.no_wait:
        bg_pids = [p.pid for _, p in _background_processes if p.is_alive()]
        print(f"\n  --no-wait: {len(bg_pids)} background processes still running")
    else:
        print(f"\n  Waiting for {len(_background_processes)} background processes...")
        for name, p in _background_processes:
            p.join(timeout=600)
            if p.is_alive():
                print(f"    {name}: still running (timeout)")
                p.terminate()
            else:
                print(f"    {name}: completed")

    total_time = time.time() - total_start
    print(f"\n{'='*80}")
    print(f"SMD EXPERIMENTS COMPLETE ({total_time/60:.1f} min, {len(results)}/{len(datasets)} succeeded)")
    print(f"{'='*80}")

    if results:
        print(f"\n{'Dataset':<35} {'Train(s)':>8} {'BestEp':>6} {'BestScore':>10} {'BestPRC':>8}")
        print("-" * 75)
        for r in results:
            print(f"{r['key']:<35} {r['pure_train_time']:>8.0f} {r['best_epoch']:>6} "
                  f"{r['best_score']:>10.4f} {r['best_prc']:>8.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Add SMD experiments to existing MAE experiment directories'
    )
    parser.add_argument('--experiment-dir', type=str, nargs='+', required=True,
                        help='One or more existing experiment directories')
    parser.add_argument('--machines', type=str, nargs='+', default=None,
                        help='Specific machine IDs to run (e.g., machine-1-1 machine-2-3)')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip machine/parity dirs that already have epoch_metrics.json')
    parser.add_argument('--aggregate-only', action='store_true',
                        help='Only run aggregation (no training)')
    parser.add_argument('--no-wait', action='store_true',
                        help='Skip waiting for background CPU processes')
    parser.add_argument('--save-weights', action='store_true', default=False,
                        help='Save model weights (checkpoints, best_model.pt). Default: off')

    args = parser.parse_args()

    for exp_dir in args.experiment_dir:
        exp_dir = os.path.abspath(exp_dir)
        print(f"\n{'#'*80}")
        print(f"# Experiment: {os.path.basename(exp_dir)}")
        print(f"{'#'*80}")

        if not os.path.isdir(exp_dir):
            print(f"  ERROR: {exp_dir} does not exist")
            continue

        if not args.aggregate_only:
            # Reconstruct config from existing experiment
            print("  Reconstructing config from existing experiment...")
            try:
                config_preset = reconstruct_config_preset(exp_dir)
            except FileNotFoundError as e:
                print(f"  ERROR: {e}")
                continue

            # Run SMD experiments
            run_smd_for_experiment(exp_dir, config_preset, args)

        # Aggregate results
        print("\n  Running SMD aggregation...")
        aggregate_smd_results(exp_dir)

    print("\n\nAll done!")


if __name__ == "__main__":
    main()
