#!/usr/bin/env python
"""
Re-evaluation script with pipelined GPU→CPU parallelism
========================================================

Loads already-trained models and re-runs evaluation only (no training, no viz).
GPU forward passes run sequentially; CPU-heavy scoring/metrics run in parallel.

Usage:
    # Profile mode: compare sequential vs parallel (small subset)
    python scripts/profile_re_eval.py --mode both --exp-range 1-5

    # Full re-evaluation with parallel pipeline
    python scripts/profile_re_eval.py --mode parallel --exp-range 1-170 --workers 8

    # Re-evaluate specific experiments
    python scripts/profile_re_eval.py --mode parallel --exp-range 50-60
"""

import os
import sys
import json
import time
import warnings
import builtins
from pathlib import Path
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore', message='.*enable_nested_tensor.*')

# Force flush on all prints for real-time output
_original_print = builtins.print
def _flush_print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    _original_print(*args, **kwargs)
builtins.print = _flush_print

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mae_anomaly import (
    Config, set_seed,
    SlidingWindowTimeSeriesGenerator, SlidingWindowDataset,
    NormalDataComplexity,
    SelfDistilledMAEMultivariate,
    Evaluator, SLIDING_ANOMALY_TYPE_NAMES
)
from scripts.ablation.run_ablation import compute_loss_statistics

# =============================================================================
# Constants
# =============================================================================

RESULTS_DIR = PROJECT_ROOT / 'results' / 'experiments' / '20260128_012500_phase1'
SCORING_MODES = ['default', 'adaptive', 'normalized']


# =============================================================================
# Helpers
# =============================================================================

def discover_base_experiments(exp_range: Optional[str] = None) -> List[Dict]:
    """Discover base experiments from directories, optionally filtered by range."""
    all_dirs = sorted(os.listdir(RESULTS_DIR))

    # Parse experiment range
    if exp_range:
        parts = exp_range.split('-')
        exp_min, exp_max = int(parts[0]), int(parts[1])
        valid_prefixes = [f"{i:03d}" for i in range(exp_min, exp_max + 1)]
    else:
        valid_prefixes = None

    base_exps = {}
    for d in all_dirs:
        parts = d.rsplit('_', 2)
        if len(parts) < 3 or parts[-1] != 'all':
            continue

        scoring_key = f"{parts[-2]}_{parts[-1]}"
        base_name = d[:-(len(scoring_key) + 1)]

        if valid_prefixes is not None:
            exp_num = base_name.split('_')[0]
            if exp_num not in valid_prefixes:
                continue

        if base_name not in base_exps:
            # Use the first scoring variant dir for model path
            base_exps[base_name] = {
                'base_name': base_name,
                'model_path': str(RESULTS_DIR / d / 'best_model.pt'),
                'output_dirs': {},
            }
        base_exps[base_name]['output_dirs'][scoring_key] = str(RESULTS_DIR / d)

    return list(base_exps.values())


def generate_dataset():
    """Generate the shared dataset."""
    config = Config()
    set_seed(config.random_seed)
    complexity = NormalDataComplexity(enable_complexity=False)
    generator = SlidingWindowTimeSeriesGenerator(
        total_length=config.sliding_window_total_length,
        num_features=config.num_features,
        interval_scale=config.anomaly_interval_scale,
        complexity=complexity,
        seed=config.random_seed
    )
    signals, point_labels, anomaly_regions = generator.generate()
    return signals, point_labels, anomaly_regions


def load_model_and_config(model_path: str):
    """Load model and config from checkpoint."""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    config = Config()
    for key, value in checkpoint['config'].items():
        if hasattr(config, key):
            setattr(config, key, value)
    model = SelfDistilledMAEMultivariate(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config.device)
    model.eval()
    return model, config


def create_test_dataset_and_loader(config, signals, point_labels, anomaly_regions):
    """Create test dataset and loader for a given config."""
    test_dataset = SlidingWindowDataset(
        signals=signals,
        point_labels=point_labels,
        anomaly_regions=anomaly_regions,
        window_size=config.seq_length,
        stride=config.sliding_window_test_stride,
        mask_last_n=config.patch_size,
        split='test',
        train_ratio=config.sliding_window_train_ratio,
        seed=config.random_seed
    )
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    return test_dataset, test_loader


def eval_one_scoring_mode(evaluator, config, scoring_mode, test_dataset, verbose=True, log_prefix=""):
    """Run full evaluation for one scoring mode."""
    config.anomaly_score_mode = scoring_mode
    t0 = time.time()

    def _log(step):
        if verbose:
            elapsed = time.time() - t0
            print(f"    {log_prefix}[{scoring_mode}] {step} ({elapsed:.1f}s)")

    _log("evaluate() ...")
    metrics = evaluator.evaluate()
    _log(f"evaluate() done -> ROC={metrics.get('roc_auc',0):.4f}")

    _log("by_score_type(disc) ...")
    disc_metrics = evaluator.evaluate_by_score_type('disc')
    _log("by_score_type(teacher_recon) ...")
    teacher_recon_metrics = evaluator.evaluate_by_score_type('teacher_recon')
    _log("by_score_type(student_recon) ...")
    student_recon_metrics = evaluator.evaluate_by_score_type('student_recon')

    _log("detailed_losses + stats ...")
    detailed_losses = evaluator.compute_detailed_losses()
    loss_stats = compute_loss_statistics(detailed_losses)
    anomaly_type_metrics = evaluator.get_performance_by_anomaly_type()
    _log(f"DONE total={time.time()-t0:.1f}s")

    return {
        'metrics': metrics,
        'disc_metrics': disc_metrics,
        'teacher_recon_metrics': teacher_recon_metrics,
        'student_recon_metrics': student_recon_metrics,
        'loss_stats': loss_stats,
        'anomaly_type_metrics': anomaly_type_metrics,
    }


def build_record(base_name, scoring_mode, result, config_dict):
    """Build a summary record matching run_ablation format."""
    cfg = config_dict
    loss_stats = result['loss_stats']
    mask_after = 'mask_after' in base_name
    result_key = f"{scoring_mode}_all"

    record = {
        'experiment': f"{base_name}_{result_key}",
        'base_experiment': base_name,
        'scoring_mode': scoring_mode,
        'mask_after_encoder': mask_after,
        'train_time': 0,
        'inference_time': 0,

        'force_mask_anomaly': cfg.get('force_mask_anomaly'),
        'margin_type': cfg.get('margin_type'),
        'masking_ratio': cfg.get('masking_ratio'),
        'masking_strategy': cfg.get('masking_strategy'),
        'seq_length': cfg.get('seq_length'),
        'num_patches': cfg.get('num_patches'),
        'patch_size': cfg.get('patch_size'),
        'patch_level_loss': cfg.get('patch_level_loss'),
        'patchify_mode': cfg.get('patchify_mode'),
        'shared_mask_token': cfg.get('shared_mask_token'),
        'd_model': cfg.get('d_model'),
        'nhead': cfg.get('nhead'),
        'num_encoder_layers': cfg.get('num_encoder_layers'),
        'num_teacher_decoder_layers': cfg.get('num_teacher_decoder_layers'),
        'num_student_decoder_layers': cfg.get('num_student_decoder_layers'),
        'num_shared_decoder_layers': cfg.get('num_shared_decoder_layers'),
        'dim_feedforward': cfg.get('dim_feedforward'),
        'dropout': cfg.get('dropout'),
        'cnn_channels': str(cfg.get('cnn_channels')),
        'anomaly_loss_weight': cfg.get('anomaly_loss_weight'),
        'num_epochs': cfg.get('num_epochs'),
        'margin': cfg.get('margin'),
        'lambda_disc': cfg.get('lambda_disc'),
        'dynamic_margin_k': cfg.get('dynamic_margin_k'),
        'learning_rate': cfg.get('learning_rate'),
        'weight_decay': cfg.get('weight_decay'),
        'teacher_only_warmup_epochs': cfg.get('teacher_only_warmup_epochs'),
        'warmup_epochs': cfg.get('warmup_epochs'),

        **result['metrics'],

        'reconstruction_loss': loss_stats['reconstruction_loss'],
        'discrepancy_loss': loss_stats['discrepancy_loss'],
        'recon_normal': loss_stats['recon_normal'],
        'recon_anomaly': loss_stats['recon_anomaly'],
        'recon_pure_normal': loss_stats['recon_pure_normal'],
        'recon_disturbing': loss_stats['recon_disturbing'],
        'disc_normal': loss_stats['disc_normal'],
        'disc_anomaly': loss_stats['disc_anomaly'],
        'disc_pure_normal': loss_stats['disc_pure_normal'],
        'disc_disturbing': loss_stats['disc_disturbing'],
        'disc_ratio': loss_stats['disc_ratio'],
        'disc_ratio_disturbing': loss_stats['disc_ratio_disturbing'],
        'recon_ratio': loss_stats['recon_ratio'],
        'disc_cohens_d_normal_vs_anomaly': loss_stats['disc_cohens_d_normal_vs_anomaly'],
        'disc_cohens_d_disturbing_vs_anomaly': loss_stats['disc_cohens_d_disturbing_vs_anomaly'],
        'recon_cohens_d_normal_vs_anomaly': loss_stats['recon_cohens_d_normal_vs_anomaly'],
        'recon_cohens_d_disturbing_vs_anomaly': loss_stats['recon_cohens_d_disturbing_vs_anomaly'],
        'disc_normal_std': loss_stats['disc_normal_std'],
        'disc_anomaly_std': loss_stats['disc_anomaly_std'],
        'disc_disturbing_std': loss_stats['disc_disturbing_std'],

        'disc_only_roc_auc': result.get('disc_metrics', {}).get('roc_auc', 0),
        'disc_only_f1_score': result.get('disc_metrics', {}).get('f1_score', 0),
        'disc_only_pa_20_roc_auc': result.get('disc_metrics', {}).get('pa_20_roc_auc', 0),
        'disc_only_pa_20_f1': result.get('disc_metrics', {}).get('pa_20_f1', 0),
        'disc_only_pa_50_roc_auc': result.get('disc_metrics', {}).get('pa_50_roc_auc', 0),
        'disc_only_pa_50_f1': result.get('disc_metrics', {}).get('pa_50_f1', 0),
        'disc_only_pa_80_roc_auc': result.get('disc_metrics', {}).get('pa_80_roc_auc', 0),
        'disc_only_pa_80_f1': result.get('disc_metrics', {}).get('pa_80_f1', 0),

        'teacher_recon_roc_auc': result.get('teacher_recon_metrics', {}).get('roc_auc', 0),
        'teacher_recon_f1_score': result.get('teacher_recon_metrics', {}).get('f1_score', 0),
        'teacher_recon_pa_20_roc_auc': result.get('teacher_recon_metrics', {}).get('pa_20_roc_auc', 0),
        'teacher_recon_pa_20_f1': result.get('teacher_recon_metrics', {}).get('pa_20_f1', 0),
        'teacher_recon_pa_50_roc_auc': result.get('teacher_recon_metrics', {}).get('pa_50_roc_auc', 0),
        'teacher_recon_pa_50_f1': result.get('teacher_recon_metrics', {}).get('pa_50_f1', 0),
        'teacher_recon_pa_80_roc_auc': result.get('teacher_recon_metrics', {}).get('pa_80_roc_auc', 0),
        'teacher_recon_pa_80_f1': result.get('teacher_recon_metrics', {}).get('pa_80_f1', 0),

        'student_recon_roc_auc': result.get('student_recon_metrics', {}).get('roc_auc', 0),
        'student_recon_f1_score': result.get('student_recon_metrics', {}).get('f1_score', 0),
        'student_recon_pa_20_roc_auc': result.get('student_recon_metrics', {}).get('pa_20_roc_auc', 0),
        'student_recon_pa_20_f1': result.get('student_recon_metrics', {}).get('pa_20_f1', 0),
        'student_recon_pa_50_roc_auc': result.get('student_recon_metrics', {}).get('pa_50_roc_auc', 0),
        'student_recon_pa_50_f1': result.get('student_recon_metrics', {}).get('pa_50_f1', 0),
        'student_recon_pa_80_roc_auc': result.get('student_recon_metrics', {}).get('pa_80_roc_auc', 0),
        'student_recon_pa_80_f1': result.get('student_recon_metrics', {}).get('pa_80_f1', 0),
    }
    return record


# =============================================================================
# CPU worker for parallel mode
# =============================================================================

def _cpu_eval_worker(args):
    """Worker: CPU-only evaluation using pre-cached scores from GPU forward pass."""
    (cached_scores, config_dict, base_name, scoring_modes,
     signals, point_labels, anomaly_regions_serialized) = args

    warnings.filterwarnings('ignore')

    # Reconstruct config
    config = Config()
    for key, value in config_dict.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Reconstruct anomaly_regions
    from mae_anomaly.dataset_sliding import AnomalyRegion
    anomaly_regions = []
    for r in anomaly_regions_serialized:
        region = AnomalyRegion(
            start=r['start'], end=r['end'],
            anomaly_type=r['anomaly_type'],
        )
        anomaly_regions.append(region)

    # Create test dataset/loader
    test_dataset = SlidingWindowDataset(
        signals=signals,
        point_labels=point_labels,
        anomaly_regions=anomaly_regions,
        window_size=config.seq_length,
        stride=config.sliding_window_test_stride,
        mask_last_n=config.patch_size,
        split='test',
        train_ratio=config.sliding_window_train_ratio,
        seed=config.random_seed
    )
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # Create evaluator with dummy model (cache pre-populated, model never called)
    dummy_model = SelfDistilledMAEMultivariate(config)
    dummy_model.eval()
    evaluator = Evaluator(dummy_model, config, test_loader, test_dataset=test_dataset)
    evaluator._cache['raw_scores'] = cached_scores

    records = []
    timings = {}

    for sm in scoring_modes:
        t0 = time.time()
        # No verbose logging in workers (avoids interleaved output)
        result = eval_one_scoring_mode(evaluator, config, sm, test_dataset, verbose=False)
        timings[f"{base_name}_{sm}"] = time.time() - t0
        record = build_record(base_name, sm, result, config_dict)
        records.append(record)

    return records, timings


# =============================================================================
# Sequential mode
# =============================================================================

def run_sequential(base_experiments, signals, point_labels, anomaly_regions, verbose=True):
    """Run all evaluations sequentially."""
    all_records = []
    timings = {'load': {}, 'forward': {}, 'eval': {}}

    total_start = time.time()
    n_total = len(base_experiments)

    for i, exp in enumerate(base_experiments):
        base_name = exp['base_name']
        model_path = exp['model_path']
        elapsed_total = time.time() - total_start
        print(f"\n  [{i+1}/{n_total}] {base_name} (elapsed: {elapsed_total:.0f}s)")

        print(f"    Loading model...", end=' ')
        t0 = time.time()
        model, config = load_model_and_config(model_path)
        test_dataset, test_loader = create_test_dataset_and_loader(
            config, signals, point_labels, anomaly_regions
        )
        t_load = time.time() - t0
        timings['load'][base_name] = t_load
        print(f"done ({t_load:.1f}s) seq_len={config.seq_length} test={len(test_dataset)}")

        print(f"    GPU forward pass...", end=' ')
        evaluator = Evaluator(model, config, test_loader, test_dataset=test_dataset)
        t0 = time.time()
        evaluator._get_cached_scores()
        t_forward = time.time() - t0
        timings['forward'][base_name] = t_forward
        print(f"done ({t_forward:.1f}s)")

        for j, sm in enumerate(SCORING_MODES):
            t0 = time.time()
            result = eval_one_scoring_mode(evaluator, config, sm, test_dataset,
                                           verbose=verbose,
                                           log_prefix=f"sm {j+1}/{len(SCORING_MODES)} ")
            t_eval = time.time() - t0
            timings['eval'][f"{base_name}_{sm}"] = t_eval
            record = build_record(base_name, sm, result, asdict(config))
            all_records.append(record)

        del model, evaluator
        torch.cuda.empty_cache()

        eval_total = sum(timings['eval'][f'{base_name}_{sm}'] for sm in SCORING_MODES)
        print(f"    DONE: load={t_load:.1f}s fwd={t_forward:.1f}s eval={eval_total:.1f}s total={t_load+t_forward+eval_total:.1f}s")

    total_time = time.time() - total_start
    return all_records, timings, total_time


# =============================================================================
# Parallel mode (pipelined GPU→CPU)
# =============================================================================

def run_parallel(base_experiments, signals, point_labels, anomaly_regions, max_workers=8):
    """GPU forward passes sequential, CPU eval in parallel process pool."""
    all_records = []
    timings = {'load': {}, 'forward': {}, 'eval': {}}
    errors = []

    # Serialize anomaly_regions for pickling
    anomaly_regions_ser = [
        {'start': r.start, 'end': r.end, 'anomaly_type': r.anomaly_type}
        for r in anomaly_regions
    ]

    total_start = time.time()
    n_total = len(base_experiments)
    n_gpu_done = 0
    n_cpu_done = 0

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = []

        for i, exp in enumerate(base_experiments):
            base_name = exp['base_name']
            model_path = exp['model_path']
            elapsed = time.time() - total_start
            print(f"  [{i+1}/{n_total}] {base_name} (elapsed: {elapsed:.0f}s)", end=' ')

            try:
                # Load model
                t0 = time.time()
                model, config = load_model_and_config(model_path)
                test_dataset, test_loader = create_test_dataset_and_loader(
                    config, signals, point_labels, anomaly_regions
                )
                t_load = time.time() - t0
                timings['load'][base_name] = t_load

                # GPU forward pass
                evaluator = Evaluator(model, config, test_loader, test_dataset=test_dataset)
                t0 = time.time()
                cached = evaluator._get_cached_scores()
                t_forward = time.time() - t0
                timings['forward'][base_name] = t_forward

                # Copy cached arrays for worker
                cached_copy = {
                    k: v.copy() if isinstance(v, np.ndarray) else v
                    for k, v in cached.items()
                }

                # Submit CPU work
                worker_args = (
                    cached_copy, asdict(config), base_name, SCORING_MODES,
                    signals, point_labels, anomaly_regions_ser
                )
                fut = pool.submit(_cpu_eval_worker, worker_args)
                futures.append((base_name, fut))
                n_gpu_done += 1

                # Free GPU
                del model, evaluator
                torch.cuda.empty_cache()

                print(f"load={t_load:.1f}s fwd={t_forward:.1f}s -> pool (seq={config.seq_length} p={config.num_patches})")

            except Exception as e:
                print(f"ERROR: {e}")
                errors.append((base_name, str(e)))

            # Periodically collect completed futures to show progress
            newly_done = []
            for bn, fut in futures:
                if fut.done() and bn not in timings['eval'] and f"{bn}_default" not in timings['eval']:
                    try:
                        records, worker_timings = fut.result()
                        all_records.extend(records)
                        timings['eval'].update(worker_timings)
                        n_cpu_done += 1
                        newly_done.append(bn)
                    except Exception as e:
                        errors.append((bn, str(e)))
                        n_cpu_done += 1
                        newly_done.append(bn)
            if newly_done:
                print(f"    [CPU done: {', '.join(newly_done)}] ({n_cpu_done}/{n_gpu_done} collected)")

        # Collect remaining futures
        print(f"\n  Collecting remaining CPU workers... ({n_cpu_done}/{n_gpu_done} done so far)")
        for base_name, fut in futures:
            if f"{base_name}_default" in timings['eval']:
                continue  # Already collected
            try:
                records, worker_timings = fut.result()
                all_records.extend(records)
                timings['eval'].update(worker_timings)
                n_cpu_done += 1
                total_eval = sum(worker_timings.values())
                elapsed = time.time() - total_start
                print(f"    [{n_cpu_done}/{n_gpu_done}] {base_name} CPU done: {total_eval:.1f}s (elapsed: {elapsed:.0f}s)")
            except Exception as e:
                print(f"    ERROR {base_name}: {e}")
                errors.append((base_name, str(e)))
                n_cpu_done += 1

    total_time = time.time() - total_start

    if errors:
        print(f"\n  ERRORS ({len(errors)}):")
        for bn, err in errors:
            print(f"    {bn}: {err}")

    return all_records, timings, total_time


# =============================================================================
# Reporting
# =============================================================================

def print_timing_summary(label, timings, total_time, n_experiments):
    """Print concise timing summary."""
    total_load = sum(timings['load'].values())
    total_fwd = sum(timings['forward'].values())
    total_eval = sum(timings['eval'].values())

    print(f"\n{'='*60}")
    print(f" {label} SUMMARY")
    print(f"{'='*60}")
    print(f"  Experiments:  {n_experiments}")
    print(f"  Total load:   {total_load:.1f}s")
    print(f"  Total fwd:    {total_fwd:.1f}s")
    print(f"  Total eval:   {total_eval:.1f}s")
    print(f"  Sum of parts: {total_load + total_fwd + total_eval:.1f}s")
    print(f"  Wall clock:   {total_time:.1f}s")
    if label == 'PARALLEL':
        overlap = (total_load + total_fwd + total_eval) - total_time
        print(f"  Overlap saved: {overlap:.1f}s")


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Re-evaluate trained models with parallel CPU pipeline')
    parser.add_argument('--mode', choices=['sequential', 'parallel', 'both'], default='parallel')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--exp-range', type=str, default='1-170',
                        help='Experiment range, e.g. "1-170" or "1-5"')
    parser.add_argument('--verbose', action='store_true',
                        help='Show per-step eval timing in sequential mode')
    args = parser.parse_args()

    print("="*60)
    print(" Phase 1 Re-evaluation (Parallel GPU→CPU Pipeline)")
    print("="*60)
    print(f"  Mode: {args.mode}, Workers: {args.workers}, Range: {args.exp_range}")

    # Discover experiments
    base_experiments = discover_base_experiments(args.exp_range)
    print(f"\n  Found {len(base_experiments)} base experiments")
    if len(base_experiments) <= 20:
        for exp in base_experiments:
            print(f"    {exp['base_name']}")

    # Generate dataset
    print("\n  Generating dataset...", end=' ')
    t0 = time.time()
    signals, point_labels, anomaly_regions = generate_dataset()
    print(f"done ({time.time()-t0:.1f}s) shape={signals.shape}")

    results = {}

    if args.mode in ('sequential', 'both'):
        print(f"\n{'='*60}")
        print(f" SEQUENTIAL MODE")
        print(f"{'='*60}")
        records_seq, timings_seq, total_seq = run_sequential(
            base_experiments, signals, point_labels, anomaly_regions,
            verbose=args.verbose
        )
        print_timing_summary('SEQUENTIAL', timings_seq, total_seq, len(base_experiments))

        df = pd.DataFrame(records_seq)
        out_path = RESULTS_DIR / '000_ablation_info' / 'summary_results_sequential.csv'
        os.makedirs(out_path.parent, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")
        results['sequential'] = (total_seq, records_seq, timings_seq)

    if args.mode in ('parallel', 'both'):
        print(f"\n{'='*60}")
        print(f" PARALLEL MODE (workers={args.workers})")
        print(f"{'='*60}")
        records_par, timings_par, total_par = run_parallel(
            base_experiments, signals, point_labels, anomaly_regions,
            max_workers=args.workers
        )
        print_timing_summary('PARALLEL', timings_par, total_par, len(base_experiments))

        df = pd.DataFrame(records_par)
        out_path = RESULTS_DIR / '000_ablation_info' / 'summary_results.csv'
        os.makedirs(out_path.parent, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")
        results['parallel'] = (total_par, records_par, timings_par)

    # Comparison
    if args.mode == 'both' and 'sequential' in results and 'parallel' in results:
        seq_t = results['sequential'][0]
        par_t = results['parallel'][0]
        print(f"\n{'='*60}")
        print(f" COMPARISON")
        print(f"{'='*60}")
        print(f"  Sequential: {seq_t:.1f}s")
        print(f"  Parallel:   {par_t:.1f}s")
        print(f"  Speedup:    {seq_t/par_t:.2f}x")

    print(f"\nDone!")


if __name__ == '__main__':
    main()
