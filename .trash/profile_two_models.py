#!/usr/bin/env python3
"""
Profile 2 experiments to compare all_patches vs last_patch timing.

Targets:
  - 001_default_mask_before_adaptive_all  (all_patches)
  - 001_default_mask_before_adaptive_last (last_patch)

Measures every step: model load, test data, evaluator, forward pass,
evaluate (3 modes), anomaly_type_metrics, collect_predictions,
visualizer creation, and all individual visualization functions.

Also checks PA%K consistency between evaluator and visualization.
"""

import os
import sys
import time
import json
import torch
import numpy as np
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mae_anomaly import (
    Config, SelfDistilledMAEMultivariate, set_seed,
    SlidingWindowTimeSeriesGenerator, SlidingWindowDataset,
    NormalDataComplexity,
)
from mae_anomaly.evaluator import (
    Evaluator,
    precompute_point_score_indices,
    vectorized_voting_for_all_thresholds,
    _compute_single_pa_k_roc,
    compute_segment_pa_k_detection_rate,
)
from mae_anomaly.visualization import BestModelVisualizer
from mae_anomaly.visualization.base import collect_predictions
from torch.utils.data import DataLoader


def load_model_and_config(exp_dir):
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


def profile_experiment(exp_dir, exp_name, inference_mode):
    """Profile a single experiment end-to-end."""
    timings = OrderedDict()
    vis_times = OrderedDict()

    print(f"\n{'='*70}")
    print(f"Experiment: {exp_name}")
    print(f"Inference mode: {inference_mode}")
    print(f"{'='*70}")

    # 1. Load model
    t0 = time.time()
    model, config = load_model_and_config(exp_dir)
    timings['1_load_model'] = time.time() - t0
    print(f"  1. Load model: {timings['1_load_model']:.2f}s")
    if model is None:
        print("  ERROR: Model not found")
        return None
    config.inference_mode = inference_mode

    # 2. Create test loader
    t0 = time.time()
    test_loader, test_dataset = create_test_loader(config)
    timings['2_create_test_loader'] = time.time() - t0
    print(f"  2. Create test loader: {timings['2_create_test_loader']:.2f}s (size={len(test_dataset)})")

    # 3. Create evaluator
    t0 = time.time()
    evaluator = Evaluator(model, config, test_loader, test_dataset=test_dataset)
    timings['3_create_evaluator'] = time.time() - t0
    print(f"  3. Create evaluator: {timings['3_create_evaluator']:.2f}s")

    # 4. Forward pass (cache scores)
    t0 = time.time()
    cached = evaluator._get_cached_scores(inference_mode)
    timings['4_forward_pass'] = time.time() - t0
    print(f"  4. Forward pass: {timings['4_forward_pass']:.2f}s")

    # 5. Evaluate (3 scoring modes)
    scoring_modes = ['default', 'adaptive', 'normalized']
    eval_times = {}
    eval_results = {}
    for scoring_mode in scoring_modes:
        config.anomaly_score_mode = scoring_mode
        t0 = time.time()
        metrics = evaluator.evaluate()
        eval_times[scoring_mode] = time.time() - t0
        eval_results[scoring_mode] = metrics
        pa20 = metrics.get('pa_20_roc_auc', 'N/A')
        pa20_str = f"{pa20:.4f}" if isinstance(pa20, (int, float)) else str(pa20)
        print(f"     {scoring_mode}: {eval_times[scoring_mode]:.2f}s (pa_20_roc_auc={pa20_str})")

    timings['5_evaluate_3_modes'] = sum(eval_times.values())
    print(f"  5. Evaluate total: {timings['5_evaluate_3_modes']:.2f}s")

    # 6. Anomaly type metrics
    evaluator._cache = {k: v for k, v in evaluator._cache.items() if k.startswith('raw_scores_')}
    config.anomaly_score_mode = 'adaptive'
    t0 = time.time()
    anomaly_metrics = evaluator.get_performance_by_anomaly_type()
    timings['6_anomaly_type_metrics'] = time.time() - t0
    print(f"  6. Anomaly type metrics: {timings['6_anomaly_type_metrics']:.2f}s ({len(anomaly_metrics)} types)")

    # 7. Collect predictions
    t0 = time.time()
    pred_data = collect_predictions(model, test_loader, config)
    timings['7_collect_predictions'] = time.time() - t0
    print(f"  7. Collect predictions: {timings['7_collect_predictions']:.2f}s")

    # 8. Create visualizer
    vis_output_dir = os.path.join(exp_dir, 'visualization', 'best_model')
    os.makedirs(vis_output_dir, exist_ok=True)
    t0 = time.time()
    visualizer = BestModelVisualizer(model, config, test_loader, vis_output_dir, pred_data=pred_data)
    timings['8_create_visualizer'] = time.time() - t0
    print(f"  8. Create visualizer: {timings['8_create_visualizer']:.2f}s")

    # 9. All visualization functions
    print(f"\n  --- Visualization Functions ---")

    history = None
    history_path = os.path.join(exp_dir, 'training_history.json')
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)

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
            print(f"     {name}: ERROR ({type(e).__name__}: {e})")
            vis_times[name] = -1
            continue
        vis_times[name] = time.time() - t0
        print(f"     {name}: {vis_times[name]:.2f}s")

    timings['9_visualizations'] = sum(t for t in vis_times.values() if t > 0)
    print(f"\n  9. Visualizations total: {timings['9_visualizations']:.2f}s")

    # === PA%K CONSISTENCY CHECK ===
    print(f"\n  --- PA%K Consistency Check ---")
    check_pa_k_consistency(evaluator, config, test_dataset, inference_mode, cached, eval_results)

    # Total
    total = sum(v for v in timings.values() if isinstance(v, (int, float)))
    timings['TOTAL'] = total
    print(f"\n  TOTAL: {total:.2f}s")

    # Add sub-timings
    for mode, t in eval_times.items():
        timings[f'eval_{mode}'] = t
    for name, t in vis_times.items():
        timings[f'vis_{name}'] = t

    return timings


def check_pa_k_consistency(evaluator, config, test_dataset, inference_mode, cached, eval_results):
    """Check if evaluator PA%K matches voting-threshold approach."""

    if not evaluator.can_compute_point_level_pa_k:
        print("  Cannot check: no point-level PA%K support")
        return

    # Get evaluator's PA%K values (from evaluate())
    config.anomaly_score_mode = 'adaptive'
    eval_metrics = eval_results.get('adaptive', evaluator.evaluate())

    evaluator_pa20_auc = eval_metrics.get('pa_20_roc_auc', None)
    evaluator_pa80_auc = eval_metrics.get('pa_80_roc_auc', None)
    print(f"  Evaluator pa_20_roc_auc: {evaluator_pa20_auc}")
    print(f"  Evaluator pa_80_roc_auc: {evaluator_pa80_auc}")

    # Compute voting-threshold PA%K (the correct approach)
    window_start_indices = np.array(test_dataset.window_start_indices)
    point_labels = np.array(test_dataset.point_labels)
    total_length = len(point_labels)
    num_patches = config.num_patches

    if inference_mode == 'all_patches':
        # Get patch-level scores
        patch_recon = cached['patch_recon']
        patch_disc = cached['patch_disc']
        # Apply adaptive scoring
        adaptive_lambda = patch_recon.mean() / (patch_disc.mean() + 1e-4)
        scores = patch_recon + adaptive_lambda * patch_disc
        is_patch = True
        n_windows = scores.shape[0]
    else:
        recon = cached['recon']
        disc = cached['disc']
        adaptive_lambda = recon.mean() / (disc.mean() + 1e-4)
        scores = recon + adaptive_lambda * disc
        is_patch = False
        n_windows = len(scores)

    # Precompute indices
    point_indices, point_coverage = precompute_point_score_indices(
        window_start_indices=window_start_indices,
        seq_length=config.seq_length,
        patch_size=config.patch_size,
        total_length=total_length,
        num_patches=num_patches if is_patch else None,
        is_patch_scores=is_patch
    )

    # Generate thresholds
    flat_scores = scores.ravel()
    min_s, max_s = flat_scores.min(), flat_scores.max()
    n_thresholds = 100
    thresholds = np.linspace(min_s - 0.01, max_s + 0.01, n_thresholds)

    # Vectorized voting
    point_scores_all = vectorized_voting_for_all_thresholds(
        scores=scores,
        point_indices=point_indices,
        point_coverage=point_coverage,
        thresholds=thresholds,
        is_patch_scores=is_patch
    )

    # Compute voting-threshold PA%K ROC-AUC
    eval_mask = np.ones(total_length, dtype=bool)
    for k in [20, 80]:
        result = _compute_single_pa_k_roc((
            point_scores_all, point_labels, test_dataset.anomaly_regions,
            eval_mask, k, None, True
        ))
        _, _, voting_auc, voting_f1 = result
        evaluator_auc = eval_metrics.get(f'pa_{k}_roc_auc', None)
        evaluator_f1 = eval_metrics.get(f'pa_{k}_f1', None)
        diff_auc = abs(voting_auc - evaluator_auc) if evaluator_auc is not None else 'N/A'
        diff_f1 = abs(voting_f1 - evaluator_f1) if evaluator_f1 is not None else 'N/A'
        match_auc = "MATCH" if isinstance(diff_auc, float) and diff_auc < 0.01 else "MISMATCH"
        match_f1 = "MATCH" if isinstance(diff_f1, float) and diff_f1 < 0.01 else "MISMATCH"
        print(f"  PA%{k} ROC-AUC: evaluator={evaluator_auc:.4f}, voting={voting_auc:.4f}, diff={diff_auc if isinstance(diff_auc, str) else f'{diff_auc:.4f}'} [{match_auc}]")
        print(f"  PA%{k} F1:      evaluator={evaluator_f1:.4f}, voting={voting_f1:.4f}, diff={diff_f1 if isinstance(diff_f1, str) else f'{diff_f1:.4f}'} [{match_f1}]")


def main():
    base_dir = '/home/ykio/notebooks/claude/results/experiments/20260128_012500_phase1'

    experiments = [
        ('001_default_mask_before_adaptive_all', 'all_patches'),
        ('001_default_mask_before_adaptive_last', 'last_patch'),
    ]

    print("=" * 70)
    print("PROFILING: 2 Models (all_patches vs last_patch)")
    print("=" * 70)

    all_timings = {}
    total_start = time.time()

    for exp_name, inf_mode in experiments:
        exp_dir = os.path.join(base_dir, exp_name)
        timings = profile_experiment(exp_dir, exp_name, inf_mode)
        if timings:
            all_timings[exp_name] = timings

    total_time = time.time() - total_start

    # Summary comparison
    print("\n" + "=" * 70)
    print("COMPARISON: all_patches vs last_patch")
    print("=" * 70)

    all_name = experiments[0][0]
    last_name = experiments[1][0]

    if all_name in all_timings and last_name in all_timings:
        main_steps = [
            '1_load_model', '2_create_test_loader', '3_create_evaluator',
            '4_forward_pass', '5_evaluate_3_modes', '6_anomaly_type_metrics',
            '7_collect_predictions', '8_create_visualizer', '9_visualizations', 'TOTAL'
        ]

        print(f"\n{'Step':<30} {'all_patches':>14} {'last_patch':>14} {'Ratio':>10}")
        print("-" * 72)
        for step in main_steps:
            a = all_timings[all_name].get(step, 0)
            l = all_timings[last_name].get(step, 0)
            ratio = a / l if l > 0 else 0
            print(f"{step:<30} {a:>11.2f}s {l:>11.2f}s {ratio:>9.1f}x")

        # Top 5 slowest visualization functions
        print(f"\n--- Top 5 Slowest Visualizations ---")
        all_vis = [(k.replace('vis_', ''), v) for k, v in all_timings[all_name].items()
                   if k.startswith('vis_') and v > 0]
        all_vis.sort(key=lambda x: x[1], reverse=True)
        for name, t in all_vis[:5]:
            last_t = all_timings[last_name].get(f'vis_{name}', 0)
            ratio = t / last_t if last_t > 0 else 0
            print(f"  {name:<45} all={t:.2f}s  last={last_t:.2f}s  ratio={ratio:.1f}x")

    print(f"\nGRAND TOTAL: {total_time:.2f}s")

    # Save results
    output_path = os.path.join(base_dir, '000_ablation_info', 'profile_two_models.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def to_native(obj):
        if isinstance(obj, dict):
            return {k: to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_path, 'w') as f:
        json.dump(to_native(all_timings), f, indent=2)
    print(f"Results saved to: {output_path}")


if __name__ == '__main__':
    main()
