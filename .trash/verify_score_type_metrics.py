#!/usr/bin/env python3
"""
Verify evaluate_by_score_type() produces values consistent with visualization's
plot_roc_curve_comparison() (which uses raw score components directly).

Tests 2 models: all_patches and last_patch.
Compares: evaluator's sample-level roc_auc per score type vs sklearn roc_auc on pred_data.
"""

import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mae_anomaly import (
    Config, SelfDistilledMAEMultivariate, set_seed,
    SlidingWindowTimeSeriesGenerator, SlidingWindowDataset,
    NormalDataComplexity,
)
from mae_anomaly.evaluator import Evaluator
from mae_anomaly.visualization.base import collect_predictions
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score


def load_model_and_config(exp_dir):
    model_path = os.path.join(exp_dir, 'best_model.pt')
    if not os.path.exists(model_path):
        return None, None
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


def verify_experiment(exp_dir, exp_name, inference_mode):
    print(f"\n{'='*70}")
    print(f"Experiment: {exp_name}")
    print(f"Inference mode: {inference_mode}")
    print(f"{'='*70}")

    model, config = load_model_and_config(exp_dir)
    if model is None:
        print("  ERROR: Model not found")
        return

    config.inference_mode = inference_mode
    test_loader, test_dataset = create_test_loader(config)
    evaluator = Evaluator(model, config, test_loader, test_dataset=test_dataset)

    # === 1. evaluate_by_score_type() timing ===
    print("\n--- evaluate_by_score_type() timing ---")
    score_types = ['disc', 'teacher_recon', 'student_recon']
    evaluator_results = {}

    for st in score_types:
        t0 = time.time()
        metrics = evaluator.evaluate_by_score_type(st)
        elapsed = time.time() - t0
        evaluator_results[st] = metrics
        print(f"  {st}: {elapsed:.2f}s | roc_auc={metrics['roc_auc']:.6f} | pa_20_roc_auc={metrics.get('pa_20_roc_auc', 'N/A')}")

    # === 2. Visualization's approach (collect_predictions → sklearn roc_auc) ===
    print("\n--- Visualization consistency check ---")
    pred_data = collect_predictions(model, test_loader, config)

    vis_scores = {
        'disc': pred_data['discrepancies'],
        'teacher_recon': pred_data['recon_errors'],
        'student_recon': pred_data['student_errors'],
    }
    labels = pred_data['labels']

    print(f"\n  {'Score Type':<20} {'Evaluator ROC':>14} {'Vis ROC':>14} {'Diff':>12} {'Match':>8}")
    print(f"  {'-'*68}")

    all_match = True
    for st in score_types:
        eval_roc = evaluator_results[st]['roc_auc']
        vis_roc = roc_auc_score(labels, vis_scores[st])
        diff = abs(eval_roc - vis_roc)
        match = diff < 1e-6
        if not match:
            all_match = False
        print(f"  {st:<20} {eval_roc:>14.6f} {vis_roc:>14.6f} {diff:>12.8f} {'OK' if match else 'MISMATCH':>8}")

    # === 3. CSV column values (non-zero check) ===
    print(f"\n--- CSV column values (should be non-zero) ---")
    for st in score_types:
        m = evaluator_results[st]
        prefix = {'disc': 'disc_only', 'teacher_recon': 'teacher_recon', 'student_recon': 'student_recon'}[st]
        print(f"  {prefix}_roc_auc = {m['roc_auc']:.6f}")
        print(f"  {prefix}_f1_score = {m['f1_score']:.6f}")
        for k in [20, 50, 80]:
            print(f"  {prefix}_pa_{k}_roc_auc = {m.get(f'pa_{k}_roc_auc', 0):.6f}")
            print(f"  {prefix}_pa_{k}_f1 = {m.get(f'pa_{k}_f1', 0):.6f}")

    print(f"\n{'='*70}")
    print(f"RESULT: {'ALL MATCH' if all_match else 'MISMATCH DETECTED'}")
    print(f"{'='*70}")
    return all_match


def main():
    base_dir = '/home/ykio/notebooks/claude/results/experiments/20260128_012500_phase1'

    experiments = [
        ('001_default_mask_before_adaptive_all', 'all_patches'),
        ('001_default_mask_before_adaptive_last', 'last_patch'),
    ]

    results = []
    for exp_name, inference_mode in experiments:
        exp_dir = os.path.join(base_dir, exp_name)
        match = verify_experiment(exp_dir, exp_name, inference_mode)
        results.append((exp_name, match))

    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    for exp_name, match in results:
        print(f"  {exp_name}: {'PASS' if match else 'FAIL'}")

    all_pass = all(m for _, m in results)
    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")


if __name__ == '__main__':
    main()
