#!/usr/bin/env python
"""
Unified Baseline Comparison Runner

Single entry point for all baseline experiments.
Results saved in MAE-compatible format (epoch_metrics.json, scores.npz).

Usage:
    conda activate dc_vis

    # List all available experiments
    python comparison/run_baseline.py --list-experiments

    # Run all baselines for an experiment
    python comparison/run_baseline.py --experiment simulation --model all

    # Run a single model
    python comparison/run_baseline.py --experiment swat_a1a2 --model random

    # Show current status
    python comparison/run_baseline.py --experiment simulation_complex --list

    # Filter models
    python comparison/run_baseline.py --experiment wadi_14days_A1 --model all --only-simple
    python comparison/run_baseline.py --experiment swat_a1a2 --model all --skip-sota

    # Override epochs
    python comparison/run_baseline.py --experiment wadi_14days_A1 --model all --neural-epochs 5

    # DL baselines with per-epoch eval (default for neural/SOTA)
    python comparison/run_baseline.py --experiment simulation --model mlp --eval-interval 2
"""

import sys
import argparse
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from comparison.baseline_common import (
    BASELINE_MODELS,
    SIMPLE_MODELS,
    NEURAL_MODELS,
    SOTA_MODELS,
    run_simple_baseline,
    run_dl_baseline_with_epoch_eval,
    run_dl_baseline,
    run_sota_baseline_with_epoch_eval,
    print_status,
    print_summary_sorted,
    create_model,
    is_model_available,
    filter_models,
    load_data_from_config,
    generate_excl22_directory,
)
from comparison.visualization import (
    plot_baseline_epoch_metrics,
    plot_baseline_prc_curve,
)
from comparison.experiment_configs import (
    EXPERIMENT_CONFIGS,
    get_experiment_config,
    list_experiments,
)


# ============================================================
# Visualization Helper
# ============================================================

def _generate_model_visualization(model_dir: Path, model_name: str, test_y: np.ndarray):
    """Generate visualization for a completed model.

    - DL models (with multiple epochs): epoch_metrics plots
    - All models: PRC curve from best epoch scores
    """
    import json

    metrics_file = model_dir / 'epoch_metrics.json'
    if not metrics_file.exists():
        return

    with open(metrics_file, 'r') as f:
        data = json.load(f)

    epochs = data.get('epochs', [])
    if not epochs:
        return

    # Epoch metrics plots (only for DL models with >1 epoch)
    if len(epochs) > 1:
        viz_dir = model_dir / 'visualization' / 'epoch_metrics'
        plot_baseline_epoch_metrics(epochs, str(viz_dir))
        print(f"  [VIZ] {model_name}: epoch_metrics ({len(epochs)} epochs)")

    # PRC curve from scores.npz
    scores_file = model_dir / 'scores.npz'
    if scores_file.exists():
        scores = np.load(str(scores_file))['anomaly_score']
        min_len = min(len(scores), len(test_y))
        viz_dir = model_dir / 'visualization' / 'best_model'
        plot_baseline_prc_curve(scores[:min_len], test_y[:min_len], str(viz_dir), model_name)
        print(f"  [VIZ] {model_name}: best_model_prc_curve")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Unified baseline comparison runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--experiment', '-e', type=str, default=None,
                       help='Experiment name (use --list-experiments to see all)')
    parser.add_argument('--model', '-m', type=str, default=None,
                       help='Model to run: model_name or "all"')
    parser.add_argument('--list', '-l', action='store_true',
                       help='List current status and exit')
    parser.add_argument('--list-experiments', action='store_true',
                       help='List all available experiments and exit')

    # Model filter flags
    parser.add_argument('--only-simple', action='store_true',
                       help='Only run simple baselines')
    parser.add_argument('--only-neural', action='store_true',
                       help='Only run neural + SOTA baselines')
    parser.add_argument('--skip-at', action='store_true',
                       help='Skip Anomaly Transformer')
    parser.add_argument('--skip-sota', action='store_true',
                       help='Skip all SOTA models')

    # Epoch overrides
    parser.add_argument('--neural-epochs', type=int, default=None,
                       help='Epochs for neural baselines')
    parser.add_argument('--sota-epochs', type=int, default=None,
                       help='Epochs for SOTA models')
    parser.add_argument('--at-epochs', type=int, default=None,
                       help='Epochs for Anomaly Transformer')

    # Eval control
    parser.add_argument('--eval-interval', type=int, default=1,
                       help='Evaluate every N epochs for DL models (default: 1)')
    parser.add_argument('--max-eval-workers', type=int, default=10,
                       help='Max CPU eval threads (default: 10)')
    parser.add_argument('--no-epoch-eval', action='store_true',
                       help='Skip per-epoch eval for DL models (only final eval)')

    # Output
    parser.add_argument('--output-base', type=str, default=None,
                       help='Override output base directory (results go to output-base/{experiment_dir_name}/)')

    # Normalization
    parser.add_argument('--normalize-mode', type=str, default=None,
                       choices=['zscore', 'minmax'],
                       help='Override normalization mode (default: zscore)')

    # Other
    parser.add_argument('--nn-subsample', type=int, default=None,
                       help='Subsample size for 1-NN Distance')
    parser.add_argument('--force', action='store_true',
                       help='Re-run even if results already exist')

    args = parser.parse_args()

    # List experiments mode
    if args.list_experiments:
        list_experiments()
        return

    if args.experiment is None:
        parser.print_help()
        print("\n\nUse --list-experiments to see all available experiments.")
        return

    # Load experiment config
    config = get_experiment_config(args.experiment)
    experiment_name = config['results_dir_name']
    if args.output_base:
        results_dir = Path(args.output_base) / experiment_name
    else:
        results_dir = PROJECT_ROOT / "comparison" / "results" / experiment_name
    all_models = config['all_models_list']

    # List status mode
    if args.list:
        print_status(results_dir, all_models, experiment_name)
        return

    if args.model is None:
        print(f"Usage: python comparison/run_baseline.py --experiment {args.experiment} --model <model_name|all>")
        print(f"Available models: {', '.join(all_models)}")
        print("\nCurrent status:")
        print_status(results_dir, all_models, experiment_name)
        return

    # ========== Load Data ==========
    # Models whose upstream anomaly_detection pipeline assumes StandardScaler-normalized
    # input (and apply internal RevIN/instance-norm on top). For these we pass raw data
    # so the wrapper can run its own StandardScaler — matches upstream line-by-line.
    # Other 18 baselines use the externally-applied --normalize-mode (Q1/Q3 = minmax).
    SELF_NORMALIZING_SOTA = {"timesnet", "moderntcn", "dcdetector", "catch", "tfmae"}

    if args.model in SELF_NORMALIZING_SOTA:
        effective_norm = 'none'
        norm_mode = 'none'
        print(f"  [override] {args.model} uses internal StandardScaler — passing raw data (normalize_mode=none)")
    else:
        effective_norm = args.normalize_mode
        norm_mode = args.normalize_mode or 'zscore'

    print("=" * 70)
    print(f"Experiment: {experiment_name}")
    print(f"Dataset: {config.get('dataset_name', experiment_name)}")
    print(f"Normalization: {norm_mode}")
    print("=" * 70)

    loader = load_data_from_config(config, normalize_mode=effective_norm)

    # Get data from UnifiedLoader
    train_X, train_y = loader.get_train_data()
    test_X, test_y = loader.get_test_data()
    anomaly_regions = loader.get_test_anomaly_regions()
    n_features = loader.n_features

    # SWaT excl22 (exclude largest test anomaly region)
    excl_region = None
    if config.get('has_excl22'):
        excl_region = loader.excl_region  # Set by _identify_excl22_region()
        if excl_region is not None:
            excl_start, excl_end = loader.get_excl22_test_range()
            print(f"  Excl22 region (test-local): [{excl_start:,}, {excl_end:,}) "
                  f"= {excl_end - excl_start:,} pts")

    # Dataset summary
    print(f"\n  Train: {len(train_X):,} samples (anomaly: {train_y.mean():.2%})")
    print(f"  Test: {len(test_X):,} samples (anomaly: {test_y.mean():.2%})")
    print(f"  Features: {n_features}")
    print(f"  Anomaly regions: {len(anomaly_regions)}")

    if config.get('loader_kwargs', {}).get('variant') == 'normalonly':
        print(f"  Variant: normalonly")
        if hasattr(loader, 'original_train_length'):
            print(f"  Original train: {loader.original_train_length:,}")
            print(f"  Normal segments: {len(loader.normal_segments)}")

    # ========== Determine Models to Run ==========
    if args.model == 'all':
        models_to_run = list(all_models)
    else:
        models_to_run = [args.model]

    # Apply filters
    if args.model == 'all':
        models_to_run = filter_models(
            models_to_run,
            only_simple=args.only_simple,
            only_neural=args.only_neural,
            skip_at=args.skip_at,
            skip_sota=args.skip_sota,
        )

    # Epoch overrides
    epoch_overrides = {}
    if args.neural_epochs is not None:
        epoch_overrides['neural_epochs'] = args.neural_epochs
    elif config.get('default_neural_epochs'):
        epoch_overrides['neural_epochs'] = config['default_neural_epochs']
    if args.sota_epochs is not None:
        epoch_overrides['sota_epochs'] = args.sota_epochs
    if args.at_epochs is not None:
        epoch_overrides['at_epochs'] = args.at_epochs

    results_dir.mkdir(parents=True, exist_ok=True)
    print_status(results_dir, all_models, experiment_name)

    # ========== Run Models ==========
    is_segment_aware = config.get('segment_aware_training', False)

    for model_name in models_to_run:
        # Skip if already completed (unless --force)
        model_dir = results_dir / model_name
        if not args.force and (model_dir / 'epoch_metrics.json').exists():
            print(f"\n[SKIP] {model_name} already has results (use --force to re-run)")
            continue

        if not is_model_available(model_name):
            print(f"\n[SKIP] {model_name} not available (import failed)")
            continue

        try:
            model = create_model(
                model_name, n_features, config['model_preset'],
                epoch_overrides, config.get('train_stride'), args.nn_subsample,
            )
            output_dir = results_dir / model_name

            if model_name in SIMPLE_MODELS:
                # ---- Simple baseline: no epochs ----
                run_simple_baseline(
                    model, train_X, test_X, test_y, anomaly_regions,
                    model_name, output_dir, experiment_name,
                    excl_region=excl_region,
                )

            elif model_name in NEURAL_MODELS and not args.no_epoch_eval:
                # ---- Neural baseline with per-epoch eval ----
                # For segment-aware (normalonly): pre-build windows from segments
                sa_windows, sa_targets = None, None
                if is_segment_aware:
                    seq_len = getattr(model, 'seq_len', 5)
                    sa_windows, sa_targets = loader.create_windows_from_segments(seq_len=seq_len)
                run_dl_baseline_with_epoch_eval(
                    model, train_X, test_X, test_y, anomaly_regions,
                    model_name, output_dir, experiment_name,
                    excl_region=excl_region,
                    eval_interval=args.eval_interval,
                    max_eval_workers=args.max_eval_workers,
                    train_windows=sa_windows,
                    train_targets=sa_targets,
                )

            elif model_name in SOTA_MODELS and not args.no_epoch_eval:
                # ---- SOTA with per-epoch eval via callback ----
                run_sota_baseline_with_epoch_eval(
                    model, train_X, test_X, test_y, anomaly_regions,
                    model_name, output_dir, experiment_name,
                    excl_region=excl_region,
                    eval_interval=args.eval_interval,
                    max_eval_workers=args.max_eval_workers,
                )

            else:
                # ---- no-epoch-eval mode: fit then single eval ----
                run_dl_baseline(
                    model, train_X, test_X, test_y, anomaly_regions,
                    model_name, output_dir, experiment_name,
                    excl_region=excl_region,
                )

            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # ---- Visualization ----
            try:
                _generate_model_visualization(output_dir, model_name, test_y)
            except Exception as viz_e:
                print(f"  [VIZ] {model_name} visualization failed: {viz_e}")

        except Exception as e:
            print(f"\n[ERROR] {model_name} failed: {e}")
            traceback.print_exc()
            continue

        print_status(results_dir, all_models, experiment_name)

    # ========== Post-processing: SWaT excl22 directory ==========
    if config.get('has_excl22'):
        generate_excl22_directory(results_dir)

    # ========== Final Summary ==========
    print("\n" + "=" * 70)
    print("FINAL STATUS")
    print("=" * 70)
    print_status(results_dir, all_models, experiment_name)
    print_summary_sorted(results_dir, all_models, excl22=config.get('has_excl22', False))
    print("\nDone!")


if __name__ == "__main__":
    main()
