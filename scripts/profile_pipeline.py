#!/usr/bin/env python
"""
Pipeline Profiling Script (Granular)
=====================================

Profiles the evaluation → visualization → summary pipeline for experiments 001-005.
Uses existing trained models from phase1. Measures time and memory for each function.

GPU→CPU parallel handoff: GPU forward pass in main process, CPU eval in
ProcessPoolExecutor(max_workers=NUM_WORKERS), visualization in background processes.

All evaluation/visualization functionality is imported from original scripts.
Only timing/memory instrumentation is added here.

Usage:
    python scripts/profile_pipeline.py
"""

import os
import sys
import json
import time
import gc
import tracemalloc
import builtins
import warnings
import traceback as tb_module
import multiprocessing as mp
import shutil
from copy import deepcopy
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Force flush on all prints
_original_print = builtins.print
def _flush_print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    _original_print(*args, **kwargs)
builtins.print = _flush_print

warnings.filterwarnings('ignore', message='.*enable_nested_tensor.*')

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mae_anomaly import (
    Config, set_seed,
    SlidingWindowTimeSeriesGenerator, SlidingWindowDataset,
    NormalDataComplexity,
    SelfDistilledMAEMultivariate, SelfDistillationLoss,
    Trainer, Evaluator, SLIDING_ANOMALY_TYPE_NAMES
)
from mae_anomaly.visualization import (
    setup_style, BestModelVisualizer,
    collect_all_visualization_data,
)

# Import helpers from run_ablation
sys.path.insert(0, str(PROJECT_ROOT / 'scripts' / 'ablation'))
from run_ablation import (
    compute_loss_statistics,
    load_config_module,
    _cpu_eval_worker,
    _collect_done_futures,
    _background_viz_worker,
    run_visualization_background,
    wait_for_background_visualizations,
)


# =============================================================================
# Profiling Utilities
# =============================================================================

class StepProfiler:
    """Tracks time and memory for each pipeline step."""

    def __init__(self):
        self.records = []
        self._start_time = None
        self._start_gpu_mem = None
        self._start_cpu_mem = None
        self._current_step = None
        tracemalloc.start()

    def start(self, step_name: str, experiment: str = ""):
        self._current_step = step_name
        self._experiment = experiment
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._start_time = time.time()
        self._start_gpu_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        self._start_cpu_mem = tracemalloc.get_traced_memory()[0]

    def end(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - self._start_time
        gpu_mem_now = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        gpu_mem_peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        cpu_mem_now = tracemalloc.get_traced_memory()[0]
        cpu_mem_peak = tracemalloc.get_traced_memory()[1]

        record = {
            'experiment': self._experiment,
            'step': self._current_step,
            'time_sec': round(elapsed, 3),
            'gpu_mem_delta_mb': round((gpu_mem_now - self._start_gpu_mem) / 1024**2, 1),
            'gpu_mem_peak_mb': round(gpu_mem_peak / 1024**2, 1),
            'cpu_mem_delta_mb': round((cpu_mem_now - self._start_cpu_mem) / 1024**2, 1),
            'cpu_mem_peak_mb': round(cpu_mem_peak / 1024**2, 1),
        }
        self.records.append(record)

        status = (
            f"  [{self._current_step}] {elapsed:.1f}s"
            f" | GPU: {record['gpu_mem_delta_mb']:+.1f}MB (peak {record['gpu_mem_peak_mb']:.0f}MB)"
            f" | CPU: {record['cpu_mem_delta_mb']:+.1f}MB (peak {record['cpu_mem_peak_mb']:.0f}MB)"
        )
        print(status)
        return record

    def report(self):
        """Print formatted profiling report."""
        df = pd.DataFrame(self.records)
        print("\n" + "=" * 100)
        print(" PROFILING REPORT")
        print("=" * 100)

        # Summary by step type
        step_summary = df.groupby('step').agg(
            count=('time_sec', 'count'),
            total_time=('time_sec', 'sum'),
            mean_time=('time_sec', 'mean'),
            max_time=('time_sec', 'max'),
            mean_gpu_peak=('gpu_mem_peak_mb', 'mean'),
            max_gpu_peak=('gpu_mem_peak_mb', 'max'),
        ).sort_values('total_time', ascending=False)

        print("\n--- Time by Step (sorted by total time) ---")
        print(step_summary.to_string())

        # Per-experiment breakdown
        if 'experiment' in df.columns and df['experiment'].nunique() > 1:
            exp_summary = df.groupby('experiment').agg(
                total_time=('time_sec', 'sum'),
                max_gpu_peak=('gpu_mem_peak_mb', 'max'),
            ).sort_values('total_time', ascending=False)
            print("\n--- Total Time by Experiment ---")
            print(exp_summary.to_string())

        total = df['time_sec'].sum()
        print(f"\n--- TOTAL GPU-SIDE PIPELINE TIME: {total:.1f}s ({total/60:.1f}min) ---")
        print("  (CPU eval + visualization run in parallel, not included in GPU total)")
        print("=" * 100)

        return df


def _time_func(name, func, *args, **kwargs):
    """Time a function call and print result."""
    t0 = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - t0
    print(f"    [{name}] {elapsed:.2f}s")
    return result, elapsed


def _check_gpu_memory(label=""):
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"  GPU Memory ({label}): {alloc:.0f}MB allocated, {reserved:.0f}MB reserved")


def _check_cpu_memory(label=""):
    """Print current CPU memory usage from tracemalloc."""
    current, peak = tracemalloc.get_traced_memory()
    print(f"  CPU Memory ({label}): {current/1024**2:.0f}MB current, {peak/1024**2:.0f}MB peak")


# =============================================================================
# Profiled CPU Eval Worker (wraps original with timing)
# =============================================================================

def _profiled_cpu_eval_worker(args):
    """CPU eval worker with granular timing. Uses original _cpu_eval_worker internals.

    Wraps original evaluation code with per-step timing. All actual computation
    is from mae_anomaly.evaluator and run_ablation.compute_loss_statistics.
    """
    import tracemalloc as _tm
    _tm.start()

    (cached_scores, config_dict, exp_name, mask_suffix, scoring_modes,
     signals, point_labels, anomaly_regions_ser,
     output_base_dir, train_time, history, save_outputs) = args

    warnings.filterwarnings('ignore')
    pid = os.getpid()
    print(f"  [CPU-EVAL pid={pid}] Starting {exp_name}_{mask_suffix}")

    # Reconstruct config
    from mae_anomaly import Config
    config = Config()
    for key, value in config_dict.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Reconstruct anomaly_regions
    from mae_anomaly.dataset_sliding import AnomalyRegion
    anomaly_regions = [
        AnomalyRegion(start=r['start'], end=r['end'], anomaly_type=r['anomaly_type'])
        for r in anomaly_regions_ser
    ]

    # Create lightweight metadata
    from mae_anomaly.evaluator import DatasetMetadata
    total_length = len(signals)
    train_end = int(total_length * config.sliding_window_train_ratio)
    test_length = total_length - train_end
    stride = config.sliding_window_test_stride
    window_size = config.seq_length
    n_windows = max(0, (test_length - window_size) // stride + 1)
    window_start_indices = [train_end + i * stride for i in range(n_windows)]

    dataset_meta = DatasetMetadata(
        point_labels=point_labels,
        window_start_indices=window_start_indices,
        anomaly_regions=anomaly_regions,
    )

    # Create evaluator with pre-populated cache
    t0 = time.time()
    dummy_model = SelfDistilledMAEMultivariate(config)
    dummy_model.eval()
    evaluator = Evaluator(dummy_model, config, test_loader=None, test_dataset=dataset_meta)
    evaluator._cache['raw_scores'] = cached_scores
    print(f"  [CPU-EVAL pid={pid}] Evaluator setup: {time.time()-t0:.2f}s")

    records = []
    timings = {}

    for scoring_mode in scoring_modes:
        config.anomaly_score_mode = scoring_mode
        result_key = f"{scoring_mode}_all"
        sm_label = f"{exp_name}_{mask_suffix}_{result_key}"
        print(f"  [CPU-EVAL pid={pid}] Scoring mode: {scoring_mode}")

        sm_timings = {}

        t0 = time.time()
        metrics = evaluator.evaluate()
        sm_timings['evaluate'] = time.time() - t0
        print(f"    [evaluate] {sm_timings['evaluate']:.2f}s")

        t0 = time.time()
        disc_metrics = evaluator.evaluate_by_score_type('disc')
        sm_timings['evaluate_disc'] = time.time() - t0
        print(f"    [evaluate_disc] {sm_timings['evaluate_disc']:.2f}s")

        t0 = time.time()
        teacher_recon_metrics = evaluator.evaluate_by_score_type('teacher_recon')
        sm_timings['evaluate_teacher_recon'] = time.time() - t0
        print(f"    [evaluate_teacher_recon] {sm_timings['evaluate_teacher_recon']:.2f}s")

        t0 = time.time()
        student_recon_metrics = evaluator.evaluate_by_score_type('student_recon')
        sm_timings['evaluate_student_recon'] = time.time() - t0
        print(f"    [evaluate_student_recon] {sm_timings['evaluate_student_recon']:.2f}s")

        t0 = time.time()
        detailed_losses = evaluator.compute_detailed_losses()
        sm_timings['compute_detailed_losses'] = time.time() - t0
        print(f"    [compute_detailed_losses] {sm_timings['compute_detailed_losses']:.2f}s")

        t0 = time.time()
        loss_stats = compute_loss_statistics(detailed_losses)
        sm_timings['compute_loss_statistics'] = time.time() - t0
        print(f"    [compute_loss_statistics] {sm_timings['compute_loss_statistics']:.2f}s")

        t0 = time.time()
        anomaly_type_metrics = evaluator.get_performance_by_anomaly_type()
        sm_timings['get_performance_by_anomaly_type'] = time.time() - t0
        print(f"    [get_performance_by_anomaly_type] {sm_timings['get_performance_by_anomaly_type']:.2f}s")

        output_dir = os.path.join(output_base_dir, f"{exp_name}_{mask_suffix}_{result_key}")

        if save_outputs:
            os.makedirs(output_dir, exist_ok=True)

            t0 = time.time()
            with open(os.path.join(output_dir, 'best_config.json'), 'w') as f:
                json.dump(config_dict, f, indent=2)

            if history is not None:
                with open(os.path.join(output_dir, 'training_histories.json'), 'w') as f:
                    json.dump({'0': history}, f, indent=2)

            detailed_df = pd.DataFrame({
                'reconstruction_loss': detailed_losses['reconstruction_loss'],
                'discrepancy_loss': detailed_losses['discrepancy_loss'],
                'total_loss': detailed_losses['total_loss'],
                'label': detailed_losses['labels'],
                'sample_type': detailed_losses['sample_types'],
                'anomaly_type': detailed_losses['anomaly_types'],
                'anomaly_type_name': [SLIDING_ANOMALY_TYPE_NAMES[int(at)] for at in detailed_losses['anomaly_types']]
            })
            detailed_df.to_csv(os.path.join(output_dir, 'best_model_detailed.csv'), index=False)

            with open(os.path.join(output_dir, 'anomaly_type_metrics.json'), 'w') as f:
                json.dump(anomaly_type_metrics, f, indent=2)

            metadata = {
                'experiment_name': exp_name,
                'scoring_mode': scoring_mode,
                'train_time': train_time,
                'inference_time': 0,
                'metrics': metrics,
                'loss_stats': loss_stats,
                'timestamp': datetime.now().isoformat()
            }
            with open(os.path.join(output_dir, 'experiment_metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2, default=str)

            sm_timings['save_outputs'] = time.time() - t0
            print(f"    [save_outputs] {sm_timings['save_outputs']:.2f}s")

        timings[sm_label] = sm_timings

        # Build record (same as original _cpu_eval_worker)
        mask_after = config.mask_after_encoder
        record = {
            'experiment': f"{exp_name}_{mask_suffix}_{result_key}",
            'base_experiment': f"{exp_name}_{mask_suffix}",
            'scoring_mode': scoring_mode,
            'mask_after_encoder': mask_after,
            'train_time': train_time,
            'inference_time': 0,
            'force_mask_anomaly': config_dict.get('force_mask_anomaly'),
            'margin_type': config_dict.get('margin_type'),
            'masking_ratio': config_dict.get('masking_ratio'),
            'masking_strategy': config_dict.get('masking_strategy'),
            'seq_length': config_dict.get('seq_length'),
            'num_patches': config_dict.get('num_patches'),
            'patch_size': config_dict.get('patch_size'),
            'patch_level_loss': config_dict.get('patch_level_loss'),
            'patchify_mode': config_dict.get('patchify_mode'),
            'shared_mask_token': config_dict.get('shared_mask_token'),
            'd_model': config_dict.get('d_model'),
            'nhead': config_dict.get('nhead'),
            'num_encoder_layers': config_dict.get('num_encoder_layers'),
            'num_teacher_decoder_layers': config_dict.get('num_teacher_decoder_layers'),
            'num_student_decoder_layers': config_dict.get('num_student_decoder_layers'),
            'num_shared_decoder_layers': config_dict.get('num_shared_decoder_layers'),
            'dim_feedforward': config_dict.get('dim_feedforward'),
            'dropout': config_dict.get('dropout'),
            'cnn_channels': str(config_dict.get('cnn_channels')),
            'anomaly_loss_weight': config_dict.get('anomaly_loss_weight'),
            'num_epochs': config_dict.get('num_epochs'),
            'margin': config_dict.get('margin'),
            'lambda_disc': config_dict.get('lambda_disc'),
            'dynamic_margin_k': config_dict.get('dynamic_margin_k'),
            'learning_rate': config_dict.get('learning_rate'),
            'weight_decay': config_dict.get('weight_decay'),
            'teacher_only_warmup_epochs': config_dict.get('teacher_only_warmup_epochs'),
            'warmup_epochs': config_dict.get('warmup_epochs'),
            **metrics,
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
            'disc_only_roc_auc': disc_metrics.get('roc_auc', 0),
            'disc_only_f1_score': disc_metrics.get('f1_score', 0),
            'disc_only_pa_20_roc_auc': disc_metrics.get('pa_20_roc_auc', 0),
            'disc_only_pa_20_f1': disc_metrics.get('pa_20_f1', 0),
            'disc_only_pa_50_roc_auc': disc_metrics.get('pa_50_roc_auc', 0),
            'disc_only_pa_50_f1': disc_metrics.get('pa_50_f1', 0),
            'disc_only_pa_80_roc_auc': disc_metrics.get('pa_80_roc_auc', 0),
            'disc_only_pa_80_f1': disc_metrics.get('pa_80_f1', 0),
            'teacher_recon_roc_auc': teacher_recon_metrics.get('roc_auc', 0),
            'teacher_recon_f1_score': teacher_recon_metrics.get('f1_score', 0),
            'teacher_recon_pa_20_roc_auc': teacher_recon_metrics.get('pa_20_roc_auc', 0),
            'teacher_recon_pa_20_f1': teacher_recon_metrics.get('pa_20_f1', 0),
            'teacher_recon_pa_50_roc_auc': teacher_recon_metrics.get('pa_50_roc_auc', 0),
            'teacher_recon_pa_50_f1': teacher_recon_metrics.get('pa_50_f1', 0),
            'teacher_recon_pa_80_roc_auc': teacher_recon_metrics.get('pa_80_roc_auc', 0),
            'teacher_recon_pa_80_f1': teacher_recon_metrics.get('pa_80_f1', 0),
            'student_recon_roc_auc': student_recon_metrics.get('roc_auc', 0),
            'student_recon_f1_score': student_recon_metrics.get('f1_score', 0),
            'student_recon_pa_20_roc_auc': student_recon_metrics.get('pa_20_roc_auc', 0),
            'student_recon_pa_20_f1': student_recon_metrics.get('pa_20_f1', 0),
            'student_recon_pa_50_roc_auc': student_recon_metrics.get('pa_50_roc_auc', 0),
            'student_recon_pa_50_f1': student_recon_metrics.get('pa_50_f1', 0),
            'student_recon_pa_80_roc_auc': student_recon_metrics.get('pa_80_roc_auc', 0),
            'student_recon_pa_80_f1': student_recon_metrics.get('pa_80_f1', 0),
            'output_dir': output_dir,
        }
        records.append(record)

    # Print timing summary for this worker
    mem_current, mem_peak = _tm.get_traced_memory()
    _tm.stop()
    print(f"  [CPU-EVAL pid={pid}] DONE {exp_name}_{mask_suffix} "
          f"| CPU mem: {mem_current/1024**2:.0f}MB current, {mem_peak/1024**2:.0f}MB peak")
    for sm_label, sm_timings in timings.items():
        total_sm = sum(sm_timings.values())
        print(f"    {sm_label}: total={total_sm:.1f}s "
              + " | ".join(f"{k}={v:.1f}s" for k, v in sm_timings.items()))

    return records, {}


# =============================================================================
# Profiled Background Viz Worker (wraps original with per-plot timing)
# =============================================================================

def _profiled_viz_worker(args):
    """Background viz worker with per-plot timing. Uses all original BestModelVisualizer code."""
    config_dict, pred_data, detailed_data, scoring_modes, output_dirs = args

    import tracemalloc as _tm
    _tm.start()

    try:
        import matplotlib
        matplotlib.use('Agg')

        from mae_anomaly.visualization import setup_style, BestModelVisualizer
        from mae_anomaly import Config
        from mae_anomaly.config import set_seed
        from mae_anomaly.dataset_sliding import (
            SlidingWindowDataset, SlidingWindowTimeSeriesGenerator,
            NormalDataComplexity,
        )
        from torch.utils.data import DataLoader

        setup_style()
    except Exception as e:
        print(f"[BG-VIZ ERROR] Import failed: {e}", file=sys.stderr, flush=True)
        tb_module.print_exc()
        return False

    try:
        pid = os.getpid()

        # Reconstruct config on CPU
        config = Config()
        for k, v in config_dict.items():
            if hasattr(config, k):
                setattr(config, k, v)
        config.device = 'cpu'

        # Reconstruct test_loader for BestModelVisualizer
        base_cfg = Config()
        set_seed(base_cfg.random_seed)
        complexity = NormalDataComplexity(enable_complexity=False)
        generator = SlidingWindowTimeSeriesGenerator(
            total_length=base_cfg.sliding_window_total_length,
            num_features=base_cfg.num_features,
            interval_scale=base_cfg.anomaly_interval_scale,
            complexity=complexity,
            seed=base_cfg.random_seed
        )
        signals, point_labels, anomaly_regions = generator.generate()
        test_dataset = SlidingWindowDataset(
            signals=signals, point_labels=point_labels,
            anomaly_regions=anomaly_regions,
            window_size=config.seq_length,
            stride=config.sliding_window_test_stride,
            mask_last_n=config.patch_size, split='test',
            train_ratio=config.sliding_window_train_ratio,
            seed=config.random_seed
        )
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

        invariant_plots = ['best_model_reconstruction.png', 'learning_curve.png', 'case_study_gallery.png']
        first_viz_dir = None

        for idx, scoring_mode in enumerate(scoring_modes):
            result_key = f"{scoring_mode}_all"
            exp_dir = output_dirs.get(result_key)
            if not exp_dir:
                continue

            viz_dir = os.path.join(exp_dir, 'visualization', 'best_model')
            os.makedirs(viz_dir, exist_ok=True)
            print(f"[BG-VIZ pid={pid}] Generating visualizations for {result_key} -> {viz_dir}", flush=True)

            history_path = os.path.join(exp_dir, 'training_histories.json')
            history = None
            if os.path.exists(history_path):
                with open(history_path) as f:
                    history_data = json.load(f)
                    history = history_data.get('0', {})

            t0 = time.time()
            visualizer = BestModelVisualizer(
                None, config, test_loader, viz_dir,
                pred_data=pred_data.copy(),
                detailed_data=detailed_data
            )
            visualizer.recompute_scores(scoring_mode)
            print(f"  [BG-VIZ pid={pid}] Visualizer init + recompute: {time.time()-t0:.2f}s", flush=True)

            # Define plot sequence (same as generate_all + scoring-variant plots)
            if idx == 0:
                first_viz_dir = viz_dir
                plot_calls = [
                    ('plot_roc_curve', lambda: visualizer.plot_roc_curve()),
                    ('plot_confusion_matrix', lambda: visualizer.plot_confusion_matrix()),
                    ('plot_score_contribution_analysis', lambda: visualizer.plot_score_contribution_analysis(exp_dir)),
                    ('plot_reconstruction_examples', lambda: visualizer.plot_reconstruction_examples()),
                    ('plot_detection_examples', lambda: visualizer.plot_detection_examples()),
                    ('plot_summary_statistics', lambda: visualizer.plot_summary_statistics()),
                    ('plot_learning_curve', lambda: visualizer.plot_learning_curve(history)),
                    ('plot_pure_vs_disturbing_normal', lambda: visualizer.plot_pure_vs_disturbing_normal()),
                    ('plot_discrepancy_trend', lambda: visualizer.plot_discrepancy_trend()),
                    ('plot_case_study_gallery', lambda: visualizer.plot_case_study_gallery(exp_dir)),
                    ('plot_hardest_samples', lambda: visualizer.plot_hardest_samples()),
                    ('plot_performance_by_anomaly_type', lambda: visualizer.plot_performance_by_anomaly_type(exp_dir)),
                    ('plot_score_distribution_by_type', lambda: visualizer.plot_score_distribution_by_type(exp_dir)),
                    ('plot_score_contribution_epoch_trends', lambda: visualizer.plot_score_contribution_epoch_trends(exp_dir, history)),
                    ('plot_roc_curve_comparison', lambda: visualizer.plot_roc_curve_comparison()),
                    ('plot_roc_curve_pa80_comparison', lambda: visualizer.plot_roc_curve_pa80_comparison()),
                    ('plot_performance_by_anomaly_type_comparison', lambda: visualizer.plot_performance_by_anomaly_type_comparison(viz_dir)),
                ]
            else:
                # Copy invariant plots from first viz dir
                for plot_file in invariant_plots:
                    src = os.path.join(first_viz_dir, plot_file)
                    if os.path.exists(src):
                        shutil.copy2(src, viz_dir)

                plot_calls = [
                    ('plot_roc_curve', lambda: visualizer.plot_roc_curve()),
                    ('plot_confusion_matrix', lambda: visualizer.plot_confusion_matrix()),
                    ('plot_score_contribution_analysis', lambda: visualizer.plot_score_contribution_analysis(exp_dir)),
                    ('plot_detection_examples', lambda: visualizer.plot_detection_examples()),
                    ('plot_summary_statistics', lambda: visualizer.plot_summary_statistics()),
                    ('plot_pure_vs_disturbing_normal', lambda: visualizer.plot_pure_vs_disturbing_normal()),
                    ('plot_discrepancy_trend', lambda: visualizer.plot_discrepancy_trend()),
                    ('plot_hardest_samples', lambda: visualizer.plot_hardest_samples()),
                    ('plot_performance_by_anomaly_type', lambda: visualizer.plot_performance_by_anomaly_type(exp_dir)),
                    ('plot_score_distribution_by_type', lambda: visualizer.plot_score_distribution_by_type(exp_dir)),
                    ('plot_score_contribution_epoch_trends', lambda: visualizer.plot_score_contribution_epoch_trends(exp_dir, history)),
                    ('plot_roc_curve_comparison', lambda: visualizer.plot_roc_curve_comparison()),
                    ('plot_roc_curve_pa80_comparison', lambda: visualizer.plot_roc_curve_pa80_comparison()),
                    ('plot_performance_by_anomaly_type_comparison', lambda: visualizer.plot_performance_by_anomaly_type_comparison(exp_dir)),
                ]

            # Execute each plot with timing
            plot_timings = {}
            for plot_name, plot_fn in plot_calls:
                t0 = time.time()
                try:
                    plot_fn()
                except Exception as e:
                    print(f"  [BG-VIZ pid={pid}] ERROR in {plot_name}: {e}", flush=True)
                    tb_module.print_exc()
                elapsed = time.time() - t0
                plot_timings[plot_name] = elapsed
                print(f"    [{plot_name}] {elapsed:.2f}s", flush=True)

            total_sm = sum(plot_timings.values())
            print(f"  [BG-VIZ pid={pid}] {result_key} total: {total_sm:.1f}s", flush=True)

        mem_current, mem_peak = _tm.get_traced_memory()
        _tm.stop()
        print(f"[BG-VIZ pid={pid}] ALL DONE | CPU mem: {mem_current/1024**2:.0f}MB current, {mem_peak/1024**2:.0f}MB peak", flush=True)
        return True

    except Exception as e:
        print(f"[BG-VIZ ERROR] Visualization failed: {e}", file=sys.stderr, flush=True)
        tb_module.print_exc()
        return False


# =============================================================================
# Main Profiling Pipeline
# =============================================================================

EXPERIMENT_DIR = str(PROJECT_ROOT / 'results' / 'experiments' / '20260128_012500_phase1')
CONFIG_PATH = str(PROJECT_ROOT / 'scripts' / 'ablation' / 'configs' / '20260127_052220_phase1.py')
EXP_RANGE = (1, 5)
SCORING_MODES = ['default', 'adaptive', 'normalized']
NUM_WORKERS = 2


def main():
    profiler = StepProfiler()
    setup_style()
    total_start = time.time()

    print("=" * 80)
    print(" PIPELINE PROFILING (Granular): Experiments 001-005 (GPU→CPU parallel)")
    print(f" Experiment dir: {EXPERIMENT_DIR}")
    print(f" CPU eval workers: {NUM_WORKERS}")
    print("=" * 80)

    _check_gpu_memory("initial")

    # =========================================================================
    # Step 1: Dataset generation
    # =========================================================================
    print("\n[SETUP] Dataset Generation")
    profiler.start("dataset_generation", "setup")
    base_config = Config()
    set_seed(base_config.random_seed)
    complexity = NormalDataComplexity(enable_complexity=False)
    generator = SlidingWindowTimeSeriesGenerator(
        total_length=base_config.sliding_window_total_length,
        num_features=base_config.num_features,
        interval_scale=base_config.anomaly_interval_scale,
        complexity=complexity,
        seed=base_config.random_seed
    )
    signals, point_labels, anomaly_regions = generator.generate()
    profiler.end()
    print(f"  Signal shape: {signals.shape}")

    # Load config to get experiment list
    ablation_config = load_config_module(CONFIG_PATH)
    experiments = ablation_config['experiments']
    mask_settings = ablation_config['mask_settings']

    # Filter to 001-005
    valid_prefixes = {f"{i:03d}" for i in range(EXP_RANGE[0], EXP_RANGE[1] + 1)}
    experiments = [e for e in experiments if e['name'].split('_')[0] in valid_prefixes]
    print(f"  Filtered to {len(experiments)} base experiments")

    info_dir = os.path.join(EXPERIMENT_DIR, '000_ablation_info')
    os.makedirs(info_dir, exist_ok=True)

    # Serialize anomaly_regions for pickling to worker processes
    anomaly_regions_ser = [
        {'start': r.start, 'end': r.end, 'anomaly_type': r.anomaly_type}
        for r in anomaly_regions
    ]

    all_records = []
    collected_labels = set()
    errors = []
    total_runs = len(experiments) * len(mask_settings)
    run_count = 0
    n_gpu_done = 0

    # Track background viz processes manually (since we use custom worker)
    viz_processes = []

    # =========================================================================
    # GPU→CPU parallel pipeline (same as _run_parallel_mode)
    # =========================================================================
    print(f"\n{'='*80}")
    print(f" PIPELINE: GPU forward → CPU eval (workers={NUM_WORKERS}) → Background viz")
    print(f"{'='*80}")

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = []  # list of (exp_label, future)

        for mask_after_encoder in mask_settings:
            mask_suffix = 'mask_after' if mask_after_encoder else 'mask_before'
            print(f"\n{'='*80}")
            print(f" ROUND: {mask_suffix}")
            print(f"{'='*80}")

            for exp_config in experiments:
                exp_name = exp_config['name']
                run_count += 1
                exp_label = f"{exp_name}_{mask_suffix}"
                elapsed = time.time() - total_start
                print(f"\n--- [{run_count}/{total_runs}] {exp_label} (elapsed: {elapsed:.0f}s) ---")

                exp_config_modified = deepcopy(exp_config)
                exp_config_modified['mask_after_encoder'] = mask_after_encoder

                config = Config()
                for key, value in exp_config_modified.items():
                    if key == 'name':
                        continue
                    if hasattr(config, key):
                        setattr(config, key, value)
                config.point_aggregation_method = 'voting'
                config.anomaly_score_mode = SCORING_MODES[0]

                first_dir = os.path.join(EXPERIMENT_DIR, f"{exp_name}_{mask_suffix}_{SCORING_MODES[0]}_all")
                model_path = os.path.join(first_dir, 'best_model.pt')
                if not os.path.exists(model_path):
                    print(f"  SKIP: {model_path} not found")
                    continue

                try:
                    # --- Model load (GPU) ---
                    profiler.start("model_load", exp_label)
                    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                    for key, value in checkpoint['config'].items():
                        if hasattr(config, key):
                            setattr(config, key, value)
                    model = SelfDistilledMAEMultivariate(config)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model = model.to(config.device)
                    model.eval()
                    profiler.end()

                    # Load training history
                    hist_path = os.path.join(first_dir, 'training_histories.json')
                    history = None
                    if os.path.exists(hist_path):
                        with open(hist_path) as f:
                            hist_data = json.load(f)
                            history = hist_data.get('0')

                    # --- Test dataset creation ---
                    profiler.start("test_dataset_creation", exp_label)
                    test_dataset = SlidingWindowDataset(
                        signals=signals, point_labels=point_labels,
                        anomaly_regions=anomaly_regions,
                        window_size=config.seq_length,
                        stride=config.sliding_window_test_stride,
                        mask_last_n=config.patch_size, split='test',
                        train_ratio=config.sliding_window_train_ratio,
                        seed=config.random_seed
                    )
                    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
                    profiler.end()

                    _check_gpu_memory(f"before forward {exp_label}")

                    # --- Single GPU forward pass (eval scores + viz data) ---
                    profiler.start("gpu_forward_pass", exp_label)
                    evaluator = Evaluator(model, config, test_loader, test_dataset=test_dataset)
                    viz_pred_data, viz_detailed_data = collect_all_visualization_data(
                        model, test_loader, config
                    )

                    # Populate evaluator cache from viz data (no redundant forward pass)
                    n_win = viz_pred_data['n_windows']
                    num_p = viz_pred_data['num_patches']
                    evaluator.set_precomputed_patch_scores(
                        recon_patches=viz_pred_data['recon_errors'].reshape(n_win, num_p),
                        disc_patches=viz_pred_data['discrepancies'].reshape(n_win, num_p),
                        student_recon_patches=viz_pred_data['student_errors'].reshape(n_win, num_p),
                        labels=viz_detailed_data['labels'],
                        sample_types=viz_detailed_data['sample_types'],
                        anomaly_types=viz_detailed_data['anomaly_types'],
                    )
                    cached = evaluator._get_cached_scores()
                    profiler.end()

                    _check_gpu_memory(f"after forward {exp_label}")

                    # Copy cached numpy arrays for worker process
                    cached_copy = {
                        k: v.copy() if isinstance(v, np.ndarray) else v
                        for k, v in cached.items()
                    }
                    config_dict = asdict(config)

                    # Ensure scoring dirs exist
                    for sm in SCORING_MODES:
                        result_key = f"{sm}_all"
                        sm_dir = os.path.join(EXPERIMENT_DIR, f"{exp_name}_{mask_suffix}_{result_key}")
                        os.makedirs(sm_dir, exist_ok=True)

                    # --- Submit CPU eval to pool (profiled worker) ---
                    profiler.start("cpu_eval_submit", exp_label)
                    worker_args = (
                        cached_copy, config_dict, exp_name, mask_suffix,
                        SCORING_MODES, signals, point_labels, anomaly_regions_ser,
                        EXPERIMENT_DIR, 0, history, True
                    )
                    fut = pool.submit(_profiled_cpu_eval_worker, worker_args)
                    futures.append((exp_label, fut))
                    n_gpu_done += 1
                    profiler.end()

                    # Free GPU memory
                    del model, evaluator, test_loader, test_dataset, cached, cached_copy
                    torch.cuda.empty_cache()
                    gc.collect()

                    _check_gpu_memory(f"after cleanup {exp_label}")

                    # --- Start background visualization (profiled worker) ---
                    profiler.start("viz_launch", exp_label)
                    output_dirs_map = {
                        f"{sm}_all": os.path.join(EXPERIMENT_DIR, f"{exp_name}_{mask_suffix}_{sm}_all")
                        for sm in SCORING_MODES
                    }

                    # Wait for slot (max 1 concurrent viz)
                    viz_processes = [p for p in viz_processes if p.is_alive()]
                    while len(viz_processes) >= 1:
                        time.sleep(1)
                        viz_processes = [p for p in viz_processes if p.is_alive()]

                    ctx = mp.get_context('spawn')
                    viz_args = (config_dict, viz_pred_data, viz_detailed_data, SCORING_MODES, output_dirs_map)
                    viz_proc = ctx.Process(target=_profiled_viz_worker, args=(viz_args,))
                    viz_proc.start()
                    viz_processes.append(viz_proc)
                    profiler.end()

                    # Free viz data after launching background process
                    del viz_pred_data, viz_detailed_data
                    gc.collect()

                except Exception as e:
                    print(f"  ERROR: {e}")
                    errors.append((exp_label, str(e)))
                    tb_module.print_exc()

                # Periodically collect completed futures
                _collect_done_futures(futures, all_records, info_dir, errors, collected_labels)

        # Collect all remaining futures
        print(f"\n  Collecting remaining CPU workers... ({len(collected_labels)}/{n_gpu_done} done so far)")
        profiler.start("cpu_eval_wait", "cleanup")
        for exp_label, fut in futures:
            if exp_label in collected_labels:
                continue
            try:
                records, _ = fut.result()
                all_records.extend(records)
                collected_labels.add(exp_label)
                elapsed = time.time() - total_start
                print(f"    [{len(collected_labels)}/{n_gpu_done}] {exp_label} CPU done (elapsed: {elapsed:.0f}s)")
            except Exception as e:
                print(f"    ERROR {exp_label}: {e}")
                errors.append((exp_label, str(e)))
                collected_labels.add(exp_label)
                tb_module.print_exc()
        profiler.end()

    # Wait for background visualizations
    print(f"\n  Waiting for background visualizations... ({len(viz_processes)} processes)")
    profiler.start("viz_wait", "cleanup")
    for proc in viz_processes:
        proc.join()
    # Also wait for any processes started by the original run_visualization_background
    wait_for_background_visualizations()
    profiler.end()
    print("  All background visualizations completed.")

    # =========================================================================
    # Summary CSV generation
    # =========================================================================
    print(f"\n{'='*80}")
    print(" Summary CSV Generation")
    profiler.start("summary_csv", "summary")
    if all_records:
        summary_df = pd.DataFrame(all_records)
        summary_path = os.path.join(info_dir, 'summary_results.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"  Saved {len(all_records)} records to {summary_path}")
    else:
        print("  WARNING: No records collected!")
    profiler.end()

    # =========================================================================
    # Profiling Report
    # =========================================================================
    total_wall_time = time.time() - total_start
    profile_df = profiler.report()

    print(f"\n--- WALL CLOCK TIME: {total_wall_time:.1f}s ({total_wall_time/60:.1f}min) ---")
    print(f"  (includes GPU + overlapping CPU eval + visualization)")
    if errors:
        print(f"\n  ERRORS ({len(errors)}):")
        for label, err in errors:
            print(f"    {label}: {err}")
    else:
        print("\n  No errors!")

    # Save profiling results
    profile_path = os.path.join(info_dir, 'profiling_results.csv')
    profile_df.to_csv(profile_path, index=False)
    print(f"\nProfiling results saved to: {profile_path}")

    tracemalloc.stop()


if __name__ == "__main__":
    main()
