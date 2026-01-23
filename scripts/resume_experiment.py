"""
Resume experiment from saved results (quick_search_results.csv, full_search_results.csv, best_config.json, best_model.pt)

This script generates the remaining files that weren't saved due to an error:
- best_model_detailed.csv
- anomaly_type_metrics.json
- experiment_metadata.json
"""

import sys
sys.path.insert(0, '/home/ykio/notebooks/claude')

import os
import json
from datetime import datetime
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from mae_anomaly import (
    Config, set_seed,
    SlidingWindowTimeSeriesGenerator, SlidingWindowDataset,
    NormalDataComplexity,
    SelfDistilledMAEMultivariate,
    Evaluator, SLIDING_ANOMALY_TYPE_NAMES
)


def resume_experiment(experiment_dir: str):
    """Resume experiment and generate remaining files"""

    print(f"\nResuming experiment from: {experiment_dir}")
    print("="*60)

    # Load best config
    config_path = os.path.join(experiment_dir, 'best_config.json')
    with open(config_path, 'r') as f:
        best_params = json.load(f)
    print(f"Loaded best_config.json: {best_params}")

    # Create Config object with best parameters
    config = Config()
    for key, value in best_params.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Derived parameters
    if 'patch_size' in best_params:
        config.patch_size = best_params['patch_size']
    if 'num_patches' in best_params:
        config.num_patches = best_params['num_patches']
    if 'mask_last_n' in best_params:
        config.mask_last_n = best_params['mask_last_n']

    set_seed(config.random_seed)

    # Load best model
    model_path = os.path.join(experiment_dir, 'best_model.pt')
    checkpoint = torch.load(model_path, map_location=config.device, weights_only=False)

    model = SelfDistilledMAEMultivariate(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config.device)
    model.eval()

    best_metrics = checkpoint.get('metrics', {})
    print(f"Loaded best_model.pt")
    print(f"  Best metrics: {best_metrics}")

    # Generate full dataset (same as original experiment)
    print("\nGenerating full search dataset...")
    full_length = config.sliding_window_total_length
    full_train_ratio = 0.5

    complexity = NormalDataComplexity(enable_complexity=True)
    generator = SlidingWindowTimeSeriesGenerator(
        total_length=full_length,
        num_features=config.num_features,
        interval_scale=config.anomaly_interval_scale,
        complexity=complexity,
        seed=config.random_seed
    )
    signals, point_labels, anomaly_regions = generator.generate()
    print(f"  Generated: {len(signals):,} timesteps, {len(anomaly_regions)} anomaly regions")

    # Create test dataset
    print("\nCreating test dataset...")
    total_test = config.num_test_samples
    target_counts = {
        'pure_normal': int(total_test * config.test_ratio_pure_normal),
        'disturbing_normal': int(total_test * config.test_ratio_disturbing_normal),
        'anomaly': int(total_test * config.test_ratio_anomaly)
    }
    print(f"  Target counts: {target_counts}")

    test_dataset = SlidingWindowDataset(
        signals=signals,
        point_labels=point_labels,
        anomaly_regions=anomaly_regions,
        window_size=config.seq_length,
        stride=config.sliding_window_stride,
        mask_last_n=config.mask_last_n,
        split='test',
        train_ratio=full_train_ratio,
        target_counts=target_counts,
        seed=config.random_seed
    )
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    print(f"  Test samples: {len(test_dataset)}")

    # Create evaluator and get detailed results
    print("\nGenerating detailed results...")
    evaluator = Evaluator(model, config, test_loader)
    detailed_losses = evaluator.compute_detailed_losses()

    # Save detailed CSV
    detailed_df = pd.DataFrame({
        'reconstruction_loss': detailed_losses['reconstruction_loss'],
        'discrepancy_loss': detailed_losses['discrepancy_loss'],
        'total_loss': detailed_losses['total_loss'],
        'label': detailed_losses['labels'],
        'sample_type': detailed_losses['sample_types'],
        'anomaly_type': detailed_losses['anomaly_types'],
        'anomaly_type_name': [SLIDING_ANOMALY_TYPE_NAMES[int(at)] for at in detailed_losses['anomaly_types']]
    })
    detailed_path = os.path.join(experiment_dir, 'best_model_detailed.csv')
    detailed_df.to_csv(detailed_path, index=False)
    print(f"  Saved: best_model_detailed.csv ({len(detailed_df)} samples)")

    # Get and save anomaly type metrics
    anomaly_type_metrics = evaluator.get_performance_by_anomaly_type()

    # Add summary statistics per anomaly type
    for atype_name in anomaly_type_metrics.keys():
        atype_mask = detailed_df['anomaly_type_name'] == atype_name
        if atype_mask.sum() > 0:
            anomaly_type_metrics[atype_name]['mean_reconstruction_loss'] = float(
                detailed_df.loc[atype_mask, 'reconstruction_loss'].mean()
            )
            anomaly_type_metrics[atype_name]['std_reconstruction_loss'] = float(
                detailed_df.loc[atype_mask, 'reconstruction_loss'].std()
            )
            anomaly_type_metrics[atype_name]['mean_discrepancy_loss'] = float(
                detailed_df.loc[atype_mask, 'discrepancy_loss'].mean()
            )
            anomaly_type_metrics[atype_name]['std_discrepancy_loss'] = float(
                detailed_df.loc[atype_mask, 'discrepancy_loss'].std()
            )

    metrics_path = os.path.join(experiment_dir, 'anomaly_type_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(anomaly_type_metrics, f, indent=2)
    print(f"  Saved: anomaly_type_metrics.json ({len(anomaly_type_metrics)} types)")

    # Load search results to get counts
    quick_results_path = os.path.join(experiment_dir, 'quick_search_results.csv')
    full_results_path = os.path.join(experiment_dir, 'full_search_results.csv')

    quick_results = pd.read_csv(quick_results_path) if os.path.exists(quick_results_path) else pd.DataFrame()
    full_results = pd.read_csv(full_results_path) if os.path.exists(full_results_path) else pd.DataFrame()

    # Save experiment metadata
    metadata = {
        'param_grid': best_params,
        'total_combinations': len(quick_results) if len(quick_results) > 0 else 'unknown',
        'stage1_results_count': len(quick_results),
        'stage2_results_count': len(full_results),
        'best_metrics': best_metrics,
        'timestamp': datetime.now().isoformat()
    }
    metadata_path = os.path.join(experiment_dir, 'experiment_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"  Saved: experiment_metadata.json")

    print("\n" + "="*60)
    print("Experiment resume complete!")
    print(f"Output directory: {experiment_dir}")
    print("\nGenerated files:")
    print("  - best_model_detailed.csv")
    print("  - anomaly_type_metrics.json")
    print("  - experiment_metadata.json")
    print("\nTo generate visualizations:")
    print(f"  python scripts/visualize_all.py --experiment-dir {experiment_dir}")
    print("="*60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Resume experiment from saved results')
    parser.add_argument('--experiment-dir', type=str, required=True,
                        help='Path to experiment directory containing best_config.json and best_model.pt')
    args = parser.parse_args()

    resume_experiment(args.experiment_dir)
