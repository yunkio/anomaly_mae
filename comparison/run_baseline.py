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
    WEAK_SUPERVISED_MODELS,
    run_simple_baseline,
    run_dl_baseline_with_epoch_eval,
    run_dl_baseline,
    run_sota_baseline_with_epoch_eval,
    run_weak_sota_baseline_with_epoch_eval,
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
    plot_baseline_anomaly_threshold,
)
from comparison.experiment_configs import (
    EXPERIMENT_CONFIGS,
    get_experiment_config,
    list_experiments,
)


# ============================================================
# Visualization Helper
# ============================================================

def _generate_model_visualization(model_dir: Path, model_name: str, test_y: np.ndarray,
                                    anomaly_regions=None, experiment_name: str = ''):
    """Generate visualization for a completed model.

    - DL models (with multiple epochs): epoch_metrics plots
    - All models: PRC curve from best epoch scores + anomaly_threshold timeline
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

    # PRC curve from scores.npz + anomaly threshold plot (best epoch)
    scores_file = model_dir / 'scores.npz'
    if scores_file.exists():
        scores = np.load(str(scores_file))['anomaly_score']
        min_len = min(len(scores), len(test_y))
        viz_dir = model_dir / 'visualization' / 'best_model'
        plot_baseline_prc_curve(scores[:min_len], test_y[:min_len], str(viz_dir), model_name)
        print(f"  [VIZ] {model_name}: best_model_prc_curve")

        # Anomaly threshold timeline plot (MAE-style for baselines)
        try:
            best = max(epochs, key=lambda e: e.get('pak_auc_f1', 0))
            best_threshold = best.get('optimal_threshold', None)
            plot_baseline_anomaly_threshold(
                scores=scores[:min_len],
                labels=test_y[:min_len],
                regions=anomaly_regions,
                threshold=best_threshold,
                output_path=str(viz_dir / 'anomaly_threshold.png'),
                model_name=model_name,
                dataset_name=experiment_name,
                extra={
                    'best_epoch': best.get('epoch'),
                    'pak_auc_f1': best.get('pak_auc_f1', 0.0),
                    'prc_auc': best.get('prc_auc', 0.0),
                },
            )
            print(f"  [VIZ] {model_name}: anomaly_threshold (best epoch={best.get('epoch')})")
        except Exception as viz_e:
            print(f"  [VIZ] {model_name}: anomaly_threshold failed: {viz_e}")


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
                       help='Re-run even if results already exist (precedence: --force > --resume)')
    parser.add_argument('--resume', action='store_true',
                       help=('Resume weak SSL models (deepmil/wetas/treemil/nrdetector/'
                             'nrdetector_full) from checkpoints/last.pt if it exists. '
                             'Other models ignore this flag (their wrappers do not '
                             'implement set_resume()). If the target sota_epochs is '
                             'already reached, the entry is auto-skipped. Mutually '
                             'exclusive with --force (--force wins if both are passed).'))

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
    SELF_NORMALIZING_SOTA = {"timesnet", "moderntcn", "dcdetector", "catch", "tfmae", "anomaly_transformer", "memto", "npsr"}
    # Weakly-supervised baselines self-normalize per their OWN original-paper method
    # (per-recording StandardScaler for WETAS/TreeMIL/DeepMIL-lineage; per-split z-score
    # StandardScaler for NRdetector — see baselines/<m>/wrapper.py). Pass raw data so each
    # wrapper applies its paper-faithful scaler instead of the external pipeline minmax.
    # (2026-05-30 normalization-fidelity rework — fixes the global-minmax deviation.)
    SELF_NORMALIZING_WEAK = {"wetas", "treemil", "nrdetector", "nrdetector_full", "deepmil"}
    # Remaining ~14 baselines use the externally-applied --normalize-mode (Q1/Q3 = minmax;
    # their original papers also use MinMaxScaler — QuoVadis/USAD/TranAD/DAGMM/GDN/OmniAnomaly).

    if args.model in SELF_NORMALIZING_SOTA or args.model in SELF_NORMALIZING_WEAK:
        effective_norm = 'none'
        norm_mode = 'none'
        print(f"  [override] {args.model} uses internal scaler (original-paper normalization) — passing raw data (normalize_mode=none)")
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
    failed_models = []  # collected silently-skipped models for end-of-run exit code

    for model_name in models_to_run:
        # Skip if already completed (unless --force or --resume with more epochs to go).
        model_dir = results_dir / model_name
        if not args.force and (model_dir / 'epoch_metrics.json').exists():
            if args.resume:
                # Resume only meaningful for weak SSL models (others have no
                # checkpoint infrastructure — their wrappers don't implement
                # set_resume()). Distinguish the two cases in the message so
                # the operator knows whether resume was tried or skipped.
                is_weak_ssl = model_name in WEAK_SUPERVISED_MODELS
                last_ckpt = model_dir / 'checkpoints' / 'last.pt'
                target_epochs = args.sota_epochs if args.sota_epochs is not None else None
                if not is_weak_ssl:
                    print(f"\n[SKIP] {model_name} already has results "
                          f"(--resume not applicable: only weak SSL models support resume)",
                          flush=True)
                    continue
                if last_ckpt.exists() and target_epochs is not None:
                    try:
                        import torch as _torch
                        _ck = _torch.load(last_ckpt, map_location='cpu')
                        ck_epoch = int(_ck.get('epoch', 0))
                        if ck_epoch >= target_epochs:
                            print(f"\n[SKIP] {model_name} resume already at epoch "
                                  f"{ck_epoch} ≥ target {target_epochs}", flush=True)
                            continue
                        print(f"\n[RESUME] {model_name} from epoch {ck_epoch} → target {target_epochs}",
                              flush=True)
                        # fall through to training loop (resume=True is forwarded
                        # to run_weak_sota_baseline_with_epoch_eval below).
                    except Exception as ck_e:
                        print(f"\n[WARN] {model_name} could not inspect last.pt ({ck_e}); "
                              f"falling back to SKIP", flush=True)
                        continue
                else:
                    # No checkpoint to resume from — treat the existing
                    # epoch_metrics.json as a completed run (consistent with
                    # the pre-resume behaviour).
                    print(f"\n[SKIP] {model_name} already has results "
                          f"(no checkpoint to resume — completed before resume infra "
                          f"was added, or --sota-epochs not specified)",
                          flush=True)
                    continue
            else:
                print(f"\n[SKIP] {model_name} already has results (use --force to re-run "
                      f"or --resume to continue)", flush=True)
                continue

        if not is_model_available(model_name):
            print(f"\n[SKIP] {model_name} not available (import failed)", flush=True)
            continue

        try:
            # Model creation is split out: TypeError here = preset/wrapper signature
            # mismatch, which used to silently skip the model. Now it surfaces as
            # an [ERROR] line AND a nonzero exit at the end of the run so the queue
            # runner / monitoring layer can detect it instead of treating an empty
            # output dir as "in progress".
            try:
                model = create_model(
                    model_name, n_features, config['model_preset'],
                    epoch_overrides, config.get('train_stride'), args.nn_subsample,
                )
            except TypeError as type_e:
                print(
                    f"\n[ERROR] {model_name} factory create_model TypeError — "
                    f"likely MODEL_PRESETS key not accepted by wrapper __init__. "
                    f"Fix the wrapper signature or the preset. Details:",
                    flush=True,
                )
                traceback.print_exc()
                failed_models.append((model_name, 'TypeError', str(type_e)))
                continue

            # Dataset-specific memory-pressure override:
            # - dcdetector dual-attention OOM-thrashes on WaDi 1.3M train at default
            #   batch_size=128 (GPU mem hits 98% and forward stalls). Halve.
            # - catch on WaDi: halved (64) still stalled in preprocessing 5h+ at GPU 98%.
            #   Quartered (32) to give more headroom + avoid CPU thrashing on 123 features.
            _WADI_BATCH_DIVISORS = {'dcdetector': 2, 'catch': 4}
            divisor = _WADI_BATCH_DIVISORS.get(model_name)
            if divisor and 'wadi' in args.experiment.lower():
                if hasattr(model, 'batch_size'):
                    _orig_bs = model.batch_size
                    model.batch_size = _orig_bs // divisor
                    print(
                        f"[CONFIG] {model_name} × {args.experiment}: "
                        f"batch_size {_orig_bs} → {model.batch_size} (GPU mem pressure, divisor={divisor})",
                        flush=True,
                    )

            output_dir = results_dir / model_name

            if model_name in SIMPLE_MODELS:
                # ---- Simple baseline: no epochs ----
                run_simple_baseline(
                    model, train_X, test_X, test_y, anomaly_regions,
                    model_name, output_dir, experiment_name,
                    excl_region=excl_region,
                )

            # Boundary safety (all multi-segment datasets): drop sliding
            # windows that span a `run_boundaries` discontinuity. Three
            # discontinuity classes covered uniformly:
            #   (a) cross-channel concat — SMAP/MSL Pattern A (54/27 channels
            #       concatenated in time; A-1 end ↔ A-2 start would otherwise
            #       mix two physically distinct telemetry channels)
            #   (b) cross-recording — SWaT A1↔A2, WaDi 14days↔attack,
            #       PSM/SMD orig_train↔test_front (temporally non-contiguous
            #       blocks from different recording periods)
            #   (c) cross-trace — Exathlon trace junctions (separate Spark
            #       application executions)
            # In every case the boundary marks a true temporal/physical
            # discontinuity, and a window spanning it manufactures a fake
            # "continuous" sample. The MAE main pipeline already enforces
            # this via SlidingWindowDataset; this branch extends the same
            # guarantee to comparison/baseline standard variants.
            has_run_boundaries = bool((loader.data_info or {}).get('run_boundaries'))

            if model_name in NEURAL_MODELS and not args.no_epoch_eval:
                # ---- Neural baseline with per-epoch eval ----
                #   - normalonly variant: anomaly-removed segments (existing)
                #   - standard variant with run_boundaries: boundary-only
                #     segments, anomaly kept (NEW, applies to every dataset
                #     that ships a non-empty run_boundaries list)
                #   - standard variant without run_boundaries (simulation):
                #     original full-train sliding (unchanged)
                sa_windows, sa_targets = None, None
                if is_segment_aware:
                    seq_len = getattr(model, 'seq_len', 5)
                    sa_windows, sa_targets = loader.create_windows_from_segments(seq_len=seq_len)
                elif has_run_boundaries:
                    seq_len = getattr(model, 'seq_len', 5)
                    sa_windows, sa_targets = loader.create_train_windows_boundary_safe(
                        seq_len=seq_len,
                        stride=getattr(model, 'train_stride', 1) or 1,
                    )
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
                # Same boundary-safety policy as the DL branch above. All 14
                # SOTA wrappers accept `train_segments=...` and filter windows
                # that would cross a boundary.
                sa_segments = None
                if is_segment_aware:
                    sa_segments = [(s.start, s.end) for s in loader.normal_segments]
                elif has_run_boundaries:
                    sa_segments = loader.get_boundary_train_segments()
                # Per-source-FILE NORM segments (SEPARATE from window-safety
                # sa_segments): multi-file -> per entity; single-file -> whole-array.
                norm_tr, norm_te = loader.get_file_norm_segments()
                run_sota_baseline_with_epoch_eval(
                    model, train_X, test_X, test_y, anomaly_regions,
                    model_name, output_dir, experiment_name,
                    excl_region=excl_region,
                    eval_interval=args.eval_interval,
                    max_eval_workers=args.max_eval_workers,
                    train_segments=sa_segments,
                    norm_train_segments=norm_tr,
                    test_segments=norm_te,
                )

            elif model_name in WEAK_SUPERVISED_MODELS and not args.no_epoch_eval:
                # ---- Weakly-supervised SOTA (2026-05-29/30 SSL porting) ----
                # Same execution shape as the SOTA branch, but forwards train_y
                # (point labels for the train split) so the wrapper can build its
                # own weak window/bag labels (max over window, train-split-only,
                # leak-free). Q1-only: the wrapper raises RuntimeError under Q3
                # (normalonly → all-zero train_y → no positive bag).
                sa_segments = None
                if is_segment_aware:
                    sa_segments = [(s.start, s.end) for s in loader.normal_segments]
                elif has_run_boundaries:
                    sa_segments = loader.get_boundary_train_segments()
                # Per-source-FILE NORM segments (SEPARATE from window-safety
                # sa_segments): multi-file -> per entity; single-file -> whole-array.
                norm_tr, norm_te = loader.get_file_norm_segments()
                run_weak_sota_baseline_with_epoch_eval(
                    model, train_X, train_y, test_X, test_y, anomaly_regions,
                    model_name, output_dir, experiment_name,
                    excl_region=excl_region,
                    eval_interval=args.eval_interval,
                    max_eval_workers=args.max_eval_workers,
                    train_segments=sa_segments,
                    norm_train_segments=norm_tr,
                    test_segments=norm_te,
                    resume=args.resume,
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
                _generate_model_visualization(output_dir, model_name, test_y,
                                              anomaly_regions=anomaly_regions,
                                              experiment_name=experiment_name)
            except Exception as viz_e:
                print(f"  [VIZ] {model_name} visualization failed: {viz_e}")

        except Exception as e:
            print(f"\n[ERROR] {model_name} failed: {e}", flush=True)
            traceback.print_exc()
            failed_models.append((model_name, type(e).__name__, str(e)))
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
    if failed_models:
        print("\n" + "=" * 70)
        print(f"FAILED MODELS: {len(failed_models)}")
        print("=" * 70)
        for m, etype, emsg in failed_models:
            print(f"  ❌ {m}: {etype}: {emsg}")
        print(
            "\nExiting with code 3 so the queue runner / monitoring layer detects\n"
            "the silent-fail and the missing model dir is not mistaken for\n"
            "an in-progress entry.",
            flush=True,
        )
        sys.exit(3)
    print("\nDone!")


if __name__ == "__main__":
    main()
