#!/usr/bin/env python3
"""
Profile the full evaluation + visualization pipeline.
Measures timing for each step for both inference modes.
"""

import os
import sys
import time
import json
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mae_anomaly import (
    Config, SelfDistilledMAEMultivariate, set_seed,
    SlidingWindowTimeSeriesGenerator, SlidingWindowDataset,
    NormalDataComplexity,
)
from mae_anomaly.evaluator import Evaluator
from mae_anomaly.visualization import BestModelVisualizer
from mae_anomaly.visualization.base import collect_predictions, collect_detailed_data
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


def profile_pipeline(exp_dir, inference_mode):
    """Profile the full pipeline for a given inference mode."""
    timings = {}

    print(f"\n{'='*70}")
    print(f"Profiling: {os.path.basename(exp_dir)}")
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

    # Set inference mode
    config.inference_mode = inference_mode

    # 2. Create test loader
    t0 = time.time()
    test_loader, test_dataset = create_test_loader(config)
    timings['2_create_test_loader'] = time.time() - t0
    print(f"  2. Create test loader: {timings['2_create_test_loader']:.2f}s")
    print(f"     - Test dataset size: {len(test_dataset)}")

    # 3. Create evaluator
    t0 = time.time()
    evaluator = Evaluator(model, config, test_loader, test_dataset=test_dataset)
    timings['3_create_evaluator'] = time.time() - t0
    print(f"  3. Create evaluator: {timings['3_create_evaluator']:.2f}s")

    # 4. Forward pass (cache scores)
    t0 = time.time()
    _ = evaluator._get_cached_scores(inference_mode)
    timings['4_forward_pass'] = time.time() - t0
    print(f"  4. Forward pass (cached): {timings['4_forward_pass']:.2f}s")

    # 5. Evaluate metrics (3 scoring modes)
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

    # 6. Get anomaly type metrics (includes PA%K ROC-AUC, F1, Detection Rate)
    t0 = time.time()
    anomaly_type_metrics = evaluator.get_performance_by_anomaly_type()
    timings['6_anomaly_type_metrics'] = time.time() - t0
    print(f"  6. Anomaly type metrics: {timings['6_anomaly_type_metrics']:.2f}s")

    # 7. Save results to JSON (simulated)
    t0 = time.time()
    # Simulate saving metrics (convert numpy types)
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    _ = json.dumps(convert_to_native(metrics))
    _ = json.dumps(convert_to_native(anomaly_type_metrics))
    timings['7_save_results'] = time.time() - t0
    print(f"  7. Save results: {timings['7_save_results']:.2f}s")

    # === VISUALIZATION ===
    print(f"\n  --- Visualization ---")

    # 8. Collect predictions
    t0 = time.time()
    pred_data = collect_predictions(model, test_loader, config)
    timings['8_collect_predictions'] = time.time() - t0
    print(f"  8. Collect predictions: {timings['8_collect_predictions']:.2f}s")

    # 9. Create visualizer
    vis_output_dir = os.path.join(exp_dir, 'visualization', 'best_model')
    os.makedirs(vis_output_dir, exist_ok=True)

    t0 = time.time()
    visualizer = BestModelVisualizer(
        model, config, test_loader, vis_output_dir,
        pred_data=pred_data
    )
    timings['9_create_visualizer'] = time.time() - t0
    print(f"  9. Create visualizer: {timings['9_create_visualizer']:.2f}s")

    # 10. Generate visualizations (individual timing)
    vis_times = {}

    t0 = time.time()
    visualizer.plot_roc_curve()
    vis_times['roc_curve'] = time.time() - t0

    t0 = time.time()
    visualizer.plot_roc_curve_comparison()
    vis_times['roc_curve_comparison'] = time.time() - t0

    t0 = time.time()
    visualizer.plot_roc_curve_pa80_comparison()
    vis_times['roc_curve_pa80_comparison'] = time.time() - t0

    t0 = time.time()
    visualizer.plot_performance_by_anomaly_type(exp_dir)
    vis_times['performance_by_anomaly_type'] = time.time() - t0

    t0 = time.time()
    visualizer.plot_performance_by_anomaly_type_comparison(vis_output_dir)
    vis_times['performance_by_anomaly_type_comparison'] = time.time() - t0

    timings['10_visualizations'] = sum(vis_times.values())
    print(f"  10. Visualizations: {timings['10_visualizations']:.2f}s")
    for name, t in vis_times.items():
        print(f"      - {name}: {t:.2f}s")

    # Total
    total = sum(timings.values())
    timings['TOTAL'] = total
    print(f"\n  TOTAL: {total:.2f}s")

    return timings


def main():
    # Use a specific experiment directory
    base_dir = '/home/ykio/notebooks/claude/results/experiments/20260128_012500_phase1'

    # Find experiments for each inference mode
    all_patches_exp = None
    last_patch_exp = None

    for d in sorted(os.listdir(base_dir)):
        if d == '000_ablation_info':
            continue
        if d.endswith('_all') and all_patches_exp is None:
            all_patches_exp = os.path.join(base_dir, d)
        elif d.endswith('_last') and last_patch_exp is None:
            last_patch_exp = os.path.join(base_dir, d)

        if all_patches_exp and last_patch_exp:
            break

    print("="*70)
    print("FULL PIPELINE PROFILING")
    print("="*70)

    results = {}

    # Profile last_patch mode
    if last_patch_exp:
        results['last_patch'] = profile_pipeline(last_patch_exp, 'last_patch')

    # Profile all_patches mode
    if all_patches_exp:
        results['all_patches'] = profile_pipeline(all_patches_exp, 'all_patches')

    # Summary comparison
    print("\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)

    if 'last_patch' in results and 'all_patches' in results:
        lp = results['last_patch']
        ap = results['all_patches']

        print(f"\n{'Step':<45} {'last_patch':>12} {'all_patches':>12} {'Ratio':>10}")
        print("-" * 80)

        for key in lp.keys():
            lp_time = lp[key]
            ap_time = ap[key]
            ratio = ap_time / lp_time if lp_time > 0 else 0
            print(f"{key:<45} {lp_time:>10.2f}s {ap_time:>10.2f}s {ratio:>9.1f}x")


if __name__ == '__main__':
    main()
