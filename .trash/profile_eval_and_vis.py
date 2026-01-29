#!/usr/bin/env python3
"""
Profile evaluation + visualization pipeline for 1 model.
Measures every step matching run_ablation.py flow + full BestModelVisualizer.

Usage:
    python scripts/profile_eval_and_vis.py
"""

import os
import sys
import time
import json
import torch
import numpy as np

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
    model_path = os.path.join(exp_dir, 'best_model.pt')
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    config = Config()
    for key, value in checkpoint.get('config', {}).items():
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
        signals=signals, point_labels=point_labels, anomaly_regions=anomaly_regions,
        window_size=config.seq_length, stride=config.sliding_window_test_stride,
        mask_last_n=config.mask_last_n, split='test',
        train_ratio=config.sliding_window_train_ratio, seed=config.random_seed
    )
    return DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False), test_dataset


def profile_model(exp_dir, exp_name, inference_mode, skip_anomaly_type_in_eval=False):
    timings = {}

    print(f"\n{'='*70}")
    print(f"  {exp_name} ({inference_mode})")
    print(f"{'='*70}")

    # === EVALUATION PHASE (run_ablation.py steps) ===
    print("\n  --- EVALUATION PHASE ---")

    t0 = time.time()
    model, config = load_model_and_config(exp_dir)
    config.inference_mode = inference_mode
    timings['eval_load_model'] = time.time() - t0

    t0 = time.time()
    test_loader, test_dataset = create_test_loader(config)
    timings['eval_create_test_loader'] = time.time() - t0

    t0 = time.time()
    evaluator = Evaluator(model, config, test_loader, test_dataset=test_dataset)
    timings['eval_create_evaluator'] = time.time() - t0

    # evaluate() x3 scoring modes
    for sm in ['default', 'adaptive', 'normalized']:
        config.anomaly_score_mode = sm
        t0 = time.time()
        metrics = evaluator.evaluate()
        timings[f'eval_{sm}'] = time.time() - t0

    # evaluate_by_score_type() x3
    for st in ['disc', 'teacher_recon', 'student_recon']:
        t0 = time.time()
        _ = evaluator.evaluate_by_score_type(st)
        timings[f'eval_score_{st}'] = time.time() - t0

    # get_performance_by_anomaly_type()
    if skip_anomaly_type_in_eval:
        timings['eval_anomaly_type_metrics'] = 0.0
    else:
        t0 = time.time()
        anomaly_type_metrics = evaluator.get_performance_by_anomaly_type()
        timings['eval_anomaly_type_metrics'] = time.time() - t0

    eval_total = sum(v for k, v in timings.items() if k.startswith('eval_'))
    timings['EVAL_TOTAL'] = eval_total

    for k, v in timings.items():
        if k.startswith('eval_'):
            print(f"    {k:<35} {v:>8.2f}s")
    print(f"    {'EVAL_TOTAL':<35} {eval_total:>8.2f}s")

    # === VISUALIZATION PHASE ===
    print("\n  --- VISUALIZATION PHASE ---")

    # collect_predictions
    t0 = time.time()
    pred_data = collect_predictions(model, test_loader, config)
    timings['vis_collect_predictions'] = time.time() - t0

    # Create visualizer
    vis_dir = os.path.join(exp_dir, 'visualization', 'best_model')
    os.makedirs(vis_dir, exist_ok=True)

    t0 = time.time()
    visualizer = BestModelVisualizer(model, config, test_loader, vis_dir, pred_data=pred_data)
    timings['vis_create_visualizer'] = time.time() - t0

    # Load history
    history = None
    history_path = os.path.join(exp_dir, 'training_histories.json')
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)

    # Individual visualization functions
    vis_functions = [
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
        ('plot_performance_by_anomaly_type_comparison', lambda: visualizer.plot_performance_by_anomaly_type_comparison(vis_dir)),
    ]

    for name, func in vis_functions:
        t0 = time.time()
        try:
            func()
        except Exception as e:
            print(f"    vis_{name}: ERROR ({e})")
            timings[f'vis_{name}'] = 0
            continue
        timings[f'vis_{name}'] = time.time() - t0

    vis_total = sum(v for k, v in timings.items() if k.startswith('vis_'))
    timings['VIS_TOTAL'] = vis_total

    for k, v in timings.items():
        if k.startswith('vis_'):
            print(f"    {k:<35} {v:>8.2f}s")
    print(f"    {'VIS_TOTAL':<35} {vis_total:>8.2f}s")

    grand_total = eval_total + vis_total
    timings['GRAND_TOTAL'] = grand_total

    print(f"\n    {'EVAL_TOTAL':<35} {eval_total:>8.2f}s")
    print(f"    {'VIS_TOTAL':<35} {vis_total:>8.2f}s")
    print(f"    {'GRAND_TOTAL':<35} {grand_total:>8.2f}s")

    return timings


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['before', 'after'], default='before')
    args = parser.parse_args()

    base_dir = '/home/ykio/notebooks/claude/results/experiments/20260128_012500_phase1'
    exp_name = '001_default_mask_before_adaptive_all'
    exp_dir = os.path.join(base_dir, exp_name)
    inference_mode = 'all_patches'

    skip = (args.mode == 'after')
    timings = profile_model(exp_dir, exp_name, inference_mode, skip_anomaly_type_in_eval=skip)

    # Save
    output_path = os.path.join(base_dir, '000_ablation_info', f'profile_eval_vis_{args.mode}.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def to_native(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return obj

    with open(output_path, 'w') as f:
        json.dump({k: to_native(v) for k, v in timings.items()}, f, indent=2)
    print(f"\nSaved to: {output_path}")


if __name__ == '__main__':
    main()
