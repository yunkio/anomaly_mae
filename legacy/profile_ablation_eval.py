#!/usr/bin/env python3
"""
Profile the evaluation portion of the ablation pipeline for 1 model.
Measures each step that run_ablation.py performs after training.
"""

import os
import sys
import time
import json
import torch
import numpy as np
import pandas as pd

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


def profile_one_model(exp_dir, exp_name, inference_mode):
    """Profile evaluation steps matching run_ablation.py flow."""
    timings = {}

    print(f"\n{'='*70}")
    print(f"  {exp_name} ({inference_mode})")
    print(f"{'='*70}")

    # 1. Load model
    t0 = time.time()
    model, config = load_model_and_config(exp_dir)
    config.inference_mode = inference_mode
    timings['load_model'] = time.time() - t0

    # 2. Create test loader
    t0 = time.time()
    test_loader, test_dataset = create_test_loader(config)
    timings['create_test_loader'] = time.time() - t0

    # 3. Create evaluator
    t0 = time.time()
    evaluator = Evaluator(model, config, test_loader, test_dataset=test_dataset)
    timings['create_evaluator'] = time.time() - t0

    # 4. evaluate() for 3 scoring modes (as in run_ablation.py)
    scoring_modes = ['default', 'adaptive', 'normalized']
    for sm in scoring_modes:
        config.anomaly_score_mode = sm
        t0 = time.time()
        metrics = evaluator.evaluate()
        timings[f'evaluate_{sm}'] = time.time() - t0

    # 5. evaluate_by_score_type() x3 (as in run_ablation.py)
    for st in ['disc', 'teacher_recon', 'student_recon']:
        t0 = time.time()
        _ = evaluator.evaluate_by_score_type(st)
        timings[f'score_type_{st}'] = time.time() - t0

    # 6. get_performance_by_anomaly_type() (as in run_ablation.py)
    t0 = time.time()
    anomaly_type_metrics = evaluator.get_performance_by_anomaly_type()
    timings['anomaly_type_metrics'] = time.time() - t0

    # Print summary
    total = sum(timings.values())
    timings['TOTAL'] = total

    print(f"\n  {'Step':<35} {'Time':>10}")
    print(f"  {'-'*47}")
    for k, v in timings.items():
        print(f"  {k:<35} {v:>8.2f}s")

    return timings


def main():
    base_dir = '/home/ykio/notebooks/claude/results/experiments/20260128_012500_phase1'

    experiments = [
        ('001_default_mask_before_adaptive_all', 'all_patches'),
        ('001_default_mask_before_adaptive_last', 'last_patch'),
    ]

    all_timings = {}
    for exp_name, inf_mode in experiments:
        exp_dir = os.path.join(base_dir, exp_name)
        timings = profile_one_model(exp_dir, exp_name, inf_mode)
        all_timings[exp_name] = timings

    # Final comparison
    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")
    print(f"\n  {'Step':<35} {'all_patches':>12} {'last_patch':>12}")
    print(f"  {'-'*61}")
    keys = list(all_timings[experiments[0][0]].keys())
    for k in keys:
        v1 = all_timings[experiments[0][0]][k]
        v2 = all_timings[experiments[1][0]][k]
        print(f"  {k:<35} {v1:>10.2f}s {v2:>10.2f}s")


if __name__ == '__main__':
    main()
