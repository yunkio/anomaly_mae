#!/usr/bin/env python3
"""
Profile script to measure timing of Phase 1 experiment re-evaluation and visualization.

Tests 6 models: 001_default_mask_after_{adaptive,default,normalized}_{all,last}

Measures:
1. Model loading
2. Test dataset creation
3. Evaluator creation
4. Forward pass (cache scores)
5. evaluate() for each scoring mode
6. get_performance_by_anomaly_type()
7. evaluate_by_score_type() for disc, teacher_recon, student_recon
8. collect_predictions()
9. Create visualizer
10. All visualization functions from generate_all():
    - plot_roc_curve
    - plot_confusion_matrix
    - plot_score_contribution_analysis
    - plot_reconstruction_examples
    - plot_detection_examples
    - plot_summary_statistics
    - plot_learning_curve
    - plot_pure_vs_disturbing_normal
    - plot_discrepancy_trend
    - plot_case_study_gallery
    - plot_anomaly_type_case_studies
    - plot_hardest_samples
    - plot_performance_by_anomaly_type
    - plot_score_distribution_by_type
    - plot_score_contribution_epoch_trends
    - plot_roc_curve_comparison
    - plot_roc_curve_pa80_comparison
    - plot_performance_by_anomaly_type_comparison

Usage:
    python scripts/profile_phase1_update.py
"""

import os
import sys
import time
import json
import torch
import numpy as np
import pandas as pd
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mae_anomaly import (
    Config, SelfDistilledMAEMultivariate, set_seed,
    SlidingWindowTimeSeriesGenerator, SlidingWindowDataset,
    NormalDataComplexity,
)
from mae_anomaly.evaluator import Evaluator
from mae_anomaly.visualization import BestModelVisualizer
from mae_anomaly.visualization.base import collect_predictions
from torch.utils.data import DataLoader


def load_model_and_config(exp_dir):
    """Load model and config from experiment directory."""
    model_path = os.path.join(exp_dir, 'best_model.pt')
    if not os.path.exists(model_path):
        return None, None

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    config = Config()
    saved_config = checkpoint.get('config', {})
    for key, value in saved_config.items():
        if hasattr(config, key):
            setattr(config, key, value)

    model = SelfDistilledMAEMultivariate(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config.device)
    model.eval()

    return model, config


def create_test_loader(config):
    """Create test data loader."""
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

    test_dataset = SlidingWindowDataset(
        signals=signals,
        point_labels=point_labels,
        anomaly_regions=anomaly_regions,
        window_size=config.seq_length,
        stride=config.sliding_window_test_stride,
        mask_last_n=config.mask_last_n,
        split='test',
        train_ratio=config.sliding_window_train_ratio,
        seed=config.random_seed
    )

    return DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False), test_dataset


def profile_single_experiment(exp_dir, exp_name, inference_mode):
    """Profile a single experiment with detailed timing breakdown."""
    timings = {}
    vis_times = {}

    print(f"\n{'='*70}")
    print(f"Experiment: {exp_name}")
    print(f"Inference mode: {inference_mode}")
    print(f"{'='*70}")

    # === 1. Load model ===
    t0 = time.time()
    model, config = load_model_and_config(exp_dir)
    timings['1_load_model'] = time.time() - t0
    print(f"  1. Load model: {timings['1_load_model']:.2f}s")

    if model is None:
        print("  ERROR: Model not found")
        return None

    # Set inference mode
    config.inference_mode = inference_mode

    # === 2. Create test loader ===
    t0 = time.time()
    test_loader, test_dataset = create_test_loader(config)
    timings['2_create_test_loader'] = time.time() - t0
    print(f"  2. Create test loader: {timings['2_create_test_loader']:.2f}s")
    print(f"     - Test dataset size: {len(test_dataset)}")

    # === 3. Create evaluator ===
    t0 = time.time()
    evaluator = Evaluator(model, config, test_loader, test_dataset=test_dataset)
    timings['3_create_evaluator'] = time.time() - t0
    print(f"  3. Create evaluator: {timings['3_create_evaluator']:.2f}s")

    # === 4. Forward pass (cache scores) ===
    t0 = time.time()
    _ = evaluator._get_cached_scores(inference_mode)
    timings['4_forward_pass'] = time.time() - t0
    print(f"  4. Forward pass (cached): {timings['4_forward_pass']:.2f}s")

    # === 5. Evaluate metrics (3 scoring modes) ===
    scoring_modes = ['default', 'adaptive', 'normalized']
    eval_times = {}

    for scoring_mode in scoring_modes:
        config.anomaly_score_mode = scoring_mode
        t0 = time.time()
        metrics = evaluator.evaluate()
        eval_times[scoring_mode] = time.time() - t0

    timings['5_evaluate_3_modes'] = sum(eval_times.values())
    print(f"  5. Evaluate (3 modes): {timings['5_evaluate_3_modes']:.2f}s")
    for mode, t in eval_times.items():
        print(f"     - {mode}: {t:.2f}s")

    # === 6. Get anomaly type metrics ===
    t0 = time.time()
    anomaly_type_metrics = evaluator.get_performance_by_anomaly_type()
    timings['6_anomaly_type_metrics'] = time.time() - t0
    print(f"  6. Anomaly type metrics: {timings['6_anomaly_type_metrics']:.2f}s")

    # === 7. Evaluate by score type (disc, teacher_recon, student_recon) ===
    score_type_times = {}
    for score_type in ['disc', 'teacher_recon', 'student_recon']:
        t0 = time.time()
        _ = evaluator.evaluate_by_score_type(score_type)
        score_type_times[score_type] = time.time() - t0

    timings['7_score_type_metrics'] = sum(score_type_times.values())
    print(f"  7. Score type metrics: {timings['7_score_type_metrics']:.2f}s")
    for st, t in score_type_times.items():
        print(f"     - {st}: {t:.2f}s")

    # === 8. Collect predictions for visualization ===
    t0 = time.time()
    pred_data = collect_predictions(model, test_loader, config)
    timings['8_collect_predictions'] = time.time() - t0
    print(f"  8. Collect predictions: {timings['8_collect_predictions']:.2f}s")

    # === 9. Create visualizer ===
    vis_output_dir = os.path.join(exp_dir, 'visualization', 'best_model')
    os.makedirs(vis_output_dir, exist_ok=True)

    t0 = time.time()
    visualizer = BestModelVisualizer(
        model, config, test_loader, vis_output_dir,
        pred_data=pred_data
    )
    timings['9_create_visualizer'] = time.time() - t0
    print(f"  9. Create visualizer: {timings['9_create_visualizer']:.2f}s")

    # === 10. Generate ALL visualizations (individual timing) ===
    print(f"\n  --- Visualization Functions ---")

    # Load history if available
    history = None
    history_path = os.path.join(exp_dir, 'training_history.json')
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)

    # All visualization functions from generate_all() + comparison functions
    visualization_functions = [
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
        ('plot_anomaly_type_case_studies', lambda: visualizer.plot_anomaly_type_case_studies(exp_dir)),
        ('plot_hardest_samples', lambda: visualizer.plot_hardest_samples()),
        ('plot_performance_by_anomaly_type', lambda: visualizer.plot_performance_by_anomaly_type(exp_dir)),
        ('plot_score_distribution_by_type', lambda: visualizer.plot_score_distribution_by_type(exp_dir)),
        ('plot_score_contribution_epoch_trends', lambda: visualizer.plot_score_contribution_epoch_trends(exp_dir, history)),
        ('plot_roc_curve_comparison', lambda: visualizer.plot_roc_curve_comparison()),
        ('plot_roc_curve_pa80_comparison', lambda: visualizer.plot_roc_curve_pa80_comparison()),
        ('plot_performance_by_anomaly_type_comparison', lambda: visualizer.plot_performance_by_anomaly_type_comparison(vis_output_dir)),
    ]

    for name, func in visualization_functions:
        t0 = time.time()
        try:
            func()
        except Exception as e:
            print(f"      - {name}: ERROR ({e})")
            vis_times[name] = 0
            continue
        vis_times[name] = time.time() - t0
        print(f"      - {name}: {vis_times[name]:.2f}s")

    timings['10_visualizations'] = sum(vis_times.values())
    print(f"\n  10. Visualizations Total: {timings['10_visualizations']:.2f}s")

    # === Total ===
    total = sum(timings.values())
    timings['TOTAL'] = total
    print(f"\n  TOTAL: {total:.2f}s")

    # Add sub-timings for detailed analysis
    timings['eval_default'] = eval_times.get('default', 0)
    timings['eval_adaptive'] = eval_times.get('adaptive', 0)
    timings['eval_normalized'] = eval_times.get('normalized', 0)
    timings['score_disc'] = score_type_times.get('disc', 0)
    timings['score_teacher_recon'] = score_type_times.get('teacher_recon', 0)
    timings['score_student_recon'] = score_type_times.get('student_recon', 0)

    # Add individual visualization timings
    for name, t in vis_times.items():
        timings[f'vis_{name}'] = t

    return timings


def main():
    base_dir = '/home/ykio/notebooks/claude/results/experiments/20260128_012500_phase1'

    # 6 target experiments (mask_before)
    target_experiments = [
        '001_default_mask_before_adaptive_all',
        '001_default_mask_before_adaptive_last',
        '001_default_mask_before_default_all',
        '001_default_mask_before_default_last',
        '001_default_mask_before_normalized_all',
        '001_default_mask_before_normalized_last',
    ]

    print("=" * 70)
    print("PHASE 1 UPDATE PROFILING")
    print("=" * 70)
    print(f"Target experiments: {len(target_experiments)}")
    print(f"Base directory: {base_dir}")

    all_timings = {}
    total_start = time.time()

    for exp_name in target_experiments:
        exp_dir = os.path.join(base_dir, exp_name)

        # Determine inference mode from experiment name
        if exp_name.endswith('_all'):
            inference_mode = 'all_patches'
        else:
            inference_mode = 'last_patch'

        timings = profile_single_experiment(exp_dir, exp_name, inference_mode)
        if timings:
            all_timings[exp_name] = timings

    total_time = time.time() - total_start

    # === Summary Table ===
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)

    # Group by inference mode
    all_patch_exps = [e for e in target_experiments if e.endswith('_all')]
    last_patch_exps = [e for e in target_experiments if e.endswith('_last')]

    # Main steps summary
    main_steps = [
        '1_load_model', '2_create_test_loader', '3_create_evaluator',
        '4_forward_pass', '5_evaluate_3_modes', '6_anomaly_type_metrics',
        '7_score_type_metrics', '8_collect_predictions', '9_create_visualizer',
        '10_visualizations', 'TOTAL'
    ]

    print(f"\n{'Step':<30} {'all_patches (avg)':>18} {'last_patch (avg)':>18} {'Ratio':>10}")
    print("-" * 80)

    for step in main_steps:
        all_avg = np.mean([all_timings[e].get(step, 0) for e in all_patch_exps if e in all_timings])
        last_avg = np.mean([all_timings[e].get(step, 0) for e in last_patch_exps if e in all_timings])
        ratio = all_avg / last_avg if last_avg > 0 else 0
        print(f"{step:<30} {all_avg:>15.2f}s {last_avg:>15.2f}s {ratio:>9.1f}x")

    # Detailed visualization breakdown
    print(f"\n--- Visualization Breakdown ---")
    vis_functions = [
        'plot_roc_curve', 'plot_confusion_matrix', 'plot_score_contribution_analysis',
        'plot_reconstruction_examples', 'plot_detection_examples', 'plot_summary_statistics',
        'plot_learning_curve', 'plot_pure_vs_disturbing_normal', 'plot_discrepancy_trend',
        'plot_case_study_gallery', 'plot_anomaly_type_case_studies', 'plot_hardest_samples',
        'plot_performance_by_anomaly_type', 'plot_score_distribution_by_type',
        'plot_score_contribution_epoch_trends', 'plot_roc_curve_comparison',
        'plot_roc_curve_pa80_comparison', 'plot_performance_by_anomaly_type_comparison'
    ]

    print(f"{'Visualization':<45} {'all_patches':>14} {'last_patch':>14} {'Ratio':>10}")
    print("-" * 85)

    for func_name in vis_functions:
        key = f'vis_{func_name}'
        all_avg = np.mean([all_timings[e].get(key, 0) for e in all_patch_exps if e in all_timings])
        last_avg = np.mean([all_timings[e].get(key, 0) for e in last_patch_exps if e in all_timings])
        ratio = all_avg / last_avg if last_avg > 0 else 0
        print(f"{func_name:<45} {all_avg:>11.2f}s {last_avg:>11.2f}s {ratio:>9.1f}x")

    # Per-experiment table
    print(f"\n--- Per-Experiment Totals ---")
    print(f"{'Experiment':<50} {'Time':>12}")
    print("-" * 65)
    for exp_name in target_experiments:
        if exp_name in all_timings:
            t = all_timings[exp_name]['TOTAL']
            print(f"{exp_name:<50} {t:>10.2f}s")

    print(f"\n{'GRAND TOTAL':<50} {total_time:>10.2f}s")
    print(f"{'Average per experiment':<50} {total_time/len(target_experiments):>10.2f}s")

    # Save detailed results to JSON
    output_path = os.path.join(base_dir, '000_ablation_info', 'profile_results.json')

    # Convert numpy types to native Python types
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_path, 'w') as f:
        json.dump(convert_to_native(all_timings), f, indent=2)
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == '__main__':
    main()
