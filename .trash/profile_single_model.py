#!/usr/bin/env python3
"""
Single model profiling to compare before/after optimization timing.

Tests: 001_default_mask_before_adaptive_all (inference_mode: all_patches)

Measures only inference-related steps:
- Forward pass
- evaluate() for 3 scoring modes
- get_performance_by_anomaly_type()
- evaluate_by_score_type() for 3 score types
"""

import os
import sys
import time
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mae_anomaly import (
    Config, SelfDistilledMAEMultivariate, set_seed,
    SlidingWindowTimeSeriesGenerator, SlidingWindowDataset,
    NormalDataComplexity,
)
from mae_anomaly.evaluator import Evaluator
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


def profile_inference(exp_dir, exp_name, inference_mode):
    """Profile inference steps only."""
    print(f"\n{'='*70}")
    print(f"SINGLE MODEL PROFILING (Optimized Version)")
    print(f"{'='*70}")
    print(f"Experiment: {exp_name}")
    print(f"Inference mode: {inference_mode}")
    print(f"{'='*70}")

    # Load model
    print("\n1. Loading model...")
    t0 = time.time()
    model, config = load_model_and_config(exp_dir)
    load_time = time.time() - t0
    print(f"   Load time: {load_time:.2f}s")

    if model is None:
        print("ERROR: Model not found")
        return

    config.inference_mode = inference_mode

    # Create test loader
    print("\n2. Creating test loader...")
    t0 = time.time()
    test_loader, test_dataset = create_test_loader(config)
    loader_time = time.time() - t0
    print(f"   Test loader time: {loader_time:.2f}s")
    print(f"   Test dataset size: {len(test_dataset)}")

    # Create evaluator
    print("\n3. Creating evaluator...")
    t0 = time.time()
    evaluator = Evaluator(model, config, test_loader, test_dataset=test_dataset)
    eval_create_time = time.time() - t0
    print(f"   Evaluator creation time: {eval_create_time:.2f}s")

    # Forward pass (cache scores)
    print("\n4. Forward pass (cache scores)...")
    t0 = time.time()
    _ = evaluator._get_cached_scores(inference_mode)
    forward_time = time.time() - t0
    print(f"   Forward pass time: {forward_time:.2f}s")

    # Evaluate with 3 scoring modes
    print("\n5. Evaluate (3 scoring modes)...")
    print(f"   can_compute_point_level_pa_k: {evaluator.can_compute_point_level_pa_k}")
    scoring_modes = ['default', 'adaptive', 'normalized']
    eval_times = {}

    for scoring_mode in scoring_modes:
        config.anomaly_score_mode = scoring_mode
        t0 = time.time()
        metrics = evaluator.evaluate()
        eval_times[scoring_mode] = time.time() - t0
        pa_20_val = metrics.get('pa_20_roc_auc', 'N/A')
        pa_20_str = f"{pa_20_val:.4f}" if isinstance(pa_20_val, (int, float)) else str(pa_20_val)
        print(f"   {scoring_mode}: {eval_times[scoring_mode]:.2f}s (pa_20_roc_auc={pa_20_str})")

    total_eval_time = sum(eval_times.values())
    print(f"   Total: {total_eval_time:.2f}s")

    # Get anomaly type metrics (THE MAIN BOTTLENECK)
    print("\n6. Get anomaly type metrics (main bottleneck)...")

    # Clear cache to force recomputation
    evaluator._cache.clear()
    _ = evaluator._get_cached_scores(inference_mode)

    print(f"   Cache keys before: {list(evaluator._cache.keys())}")

    t0 = time.time()
    anomaly_metrics = evaluator.get_performance_by_anomaly_type()
    anomaly_type_time = time.time() - t0
    print(f"   First call: {anomaly_type_time:.2f}s")

    # Debug: verify metrics are computed
    print(f"   Anomaly types processed: {len(anomaly_metrics)}")
    for atype, metrics in list(anomaly_metrics.items())[:3]:
        print(f"     - {atype}: count={metrics.get('count', 'N/A')}, pa_20_roc_auc={metrics.get('pa_20_roc_auc', 'N/A')}")

    print(f"   Cache keys after: {list(evaluator._cache.keys())}")

    # Test caching (should be instant)
    t0 = time.time()
    _ = evaluator.get_performance_by_anomaly_type()
    cached_time = time.time() - t0
    print(f"   Cached call: {cached_time:.4f}s")

    total_score_type_time = 0  # Skip evaluate_by_score_type (not relevant for optimization)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    total_inference = forward_time + total_eval_time + anomaly_type_time + total_score_type_time

    print(f"\n{'Step':<40} {'Time':>12}")
    print("-"*55)
    print(f"{'Forward pass':<40} {forward_time:>10.2f}s")
    print(f"{'Evaluate (3 modes)':<40} {total_eval_time:>10.2f}s")
    print(f"{'Anomaly type metrics':<40} {anomaly_type_time:>10.2f}s")
    print(f"{'Score type metrics (3 types)':<40} {total_score_type_time:>10.2f}s")
    print("-"*55)
    print(f"{'TOTAL INFERENCE':<40} {total_inference:>10.2f}s")

    # Comparison with previous profiling results
    print("\n" + "="*70)
    print("COMPARISON WITH PREVIOUS PROFILING (all_patches mode)")
    print("="*70)

    # Previous results from the profiling run
    prev_forward = 1.42  # approximate from previous run
    prev_eval = 0.28  # 5_evaluate_3_modes
    prev_anomaly_type = 22.33  # 6_anomaly_type_metrics (MAIN BOTTLENECK)
    prev_score_type = 0.07  # 7_score_type_metrics

    print(f"\n{'Step':<35} {'Before':>10} {'After':>10} {'Improvement':>12}")
    print("-"*70)
    print(f"{'Forward pass':<35} {prev_forward:>8.2f}s {forward_time:>8.2f}s {prev_forward/forward_time:>10.1f}x")
    print(f"{'Evaluate (3 modes)':<35} {prev_eval:>8.2f}s {total_eval_time:>8.2f}s {prev_eval/total_eval_time if total_eval_time > 0 else 0:>10.1f}x")
    print(f"{'Anomaly type metrics':<35} {prev_anomaly_type:>8.2f}s {anomaly_type_time:>8.2f}s {prev_anomaly_type/anomaly_type_time:>10.1f}x")
    print(f"{'Score type metrics':<35} {prev_score_type:>8.2f}s {total_score_type_time:>8.2f}s {prev_score_type/total_score_type_time if total_score_type_time > 0 else 0:>10.1f}x")

    prev_total = prev_forward + prev_eval + prev_anomaly_type + prev_score_type
    print("-"*70)
    print(f"{'TOTAL':<35} {prev_total:>8.2f}s {total_inference:>8.2f}s {prev_total/total_inference:>10.1f}x")


def main():
    base_dir = '/home/ykio/notebooks/claude/results/experiments/20260128_012500_phase1'
    exp_name = '001_default_mask_before_adaptive_all'
    exp_dir = os.path.join(base_dir, exp_name)
    inference_mode = 'all_patches'

    profile_inference(exp_dir, exp_name, inference_mode)


if __name__ == '__main__':
    main()
