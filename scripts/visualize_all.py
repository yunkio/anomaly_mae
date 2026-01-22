"""
Unified Visualization Script for Self-Distilled MAE Anomaly Detection

This script generates ALL visualizations for experiments:
1. Data visualizations - Understanding the dataset and anomaly types
2. Architecture visualizations - Model pipeline, patchify modes, self-distillation
3. Stage 1 visualizations - Quick search results analysis
4. Stage 2 visualizations - Full training comparison
5. Best model visualizations - Detailed model analysis

Usage:
    python scripts/visualize_all.py --experiment-dir <path_to_experiment>
    python scripts/visualize_all.py --model-path <path_to_best_model.pt>
    python scripts/visualize_all.py  # Auto-finds latest experiment

Output structure:
    results/experiments/YYYYMMDD_HHMMSS/visualization/
        ├── data/           # DataVisualizer outputs
        ├── architecture/   # ArchitectureVisualizer outputs
        ├── stage1/         # Stage 1 results visualization
        ├── stage2/         # Stage 2 results visualization
        └── best_model/     # Best model analysis
"""
import sys
sys.path.insert(0, '/home/ykio/notebooks/claude')

import os
import argparse

from mae_anomaly import Config
from mae_anomaly.visualization import (
    setup_style,
    find_latest_experiment,
    load_experiment_data,
    load_best_model,
    DataVisualizer,
    ArchitectureVisualizer,
    ExperimentVisualizer,
    Stage2Visualizer,
    BestModelVisualizer,
    TrainingProgressVisualizer,
)


def main():
    """Main entry point for visualization generation"""
    parser = argparse.ArgumentParser(description='Generate all visualizations for MAE experiments')
    parser.add_argument('--experiment-dir', type=str, default=None,
                       help='Path to experiment directory')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to best_model.pt file')
    parser.add_argument('--num-test', type=int, default=500,
                       help='Number of test samples for best model analysis')
    parser.add_argument('--skip-data', action='store_true',
                       help='Skip data visualizations')
    parser.add_argument('--skip-architecture', action='store_true',
                       help='Skip architecture visualizations')
    parser.add_argument('--skip-experiments', action='store_true',
                       help='Skip experiment result visualizations')
    parser.add_argument('--skip-model', action='store_true',
                       help='Skip best model visualizations')
    parser.add_argument('--retrain', action='store_true',
                       help='Re-train best model to visualize learning progress (takes ~10 min)')
    args = parser.parse_args()

    print("="*80)
    print(" " * 15 + "UNIFIED VISUALIZATION SCRIPT")
    print("="*80)

    setup_style()

    # Find experiment directory
    experiment_dir = None
    if args.experiment_dir:
        experiment_dir = args.experiment_dir
    elif args.model_path:
        experiment_dir = os.path.dirname(args.model_path)
    else:
        experiment_dir = find_latest_experiment()
        if experiment_dir:
            print(f"\nAuto-detected experiment: {experiment_dir}")

    if not experiment_dir or not os.path.exists(experiment_dir):
        print("ERROR: No experiment directory found.")
        print("Please specify --experiment-dir or run experiments first.")
        return

    # Create visualization output directory
    vis_dir = os.path.join(experiment_dir, 'visualization')
    os.makedirs(vis_dir, exist_ok=True)

    print(f"\nExperiment directory: {experiment_dir}")
    print(f"Visualization output: {vis_dir}")

    # Load experiment data
    print("\nLoading experiment data...")
    exp_data = load_experiment_data(experiment_dir)

    # Default config
    config = Config()
    if exp_data['best_config']:
        for key, value in exp_data['best_config'].items():
            if hasattr(config, key):
                setattr(config, key, value)

    # Get param keys dynamically from metadata or results
    param_keys = None
    if exp_data['metadata'] and 'param_grid' in exp_data['metadata']:
        param_keys = list(exp_data['metadata']['param_grid'].keys())
    elif exp_data['quick_results'] is not None:
        # Fallback: extract from results DataFrame (exclude metric columns)
        metric_cols = {'combination_id', 'roc_auc', 'f1_score', 'precision', 'recall',
                       'disturbing_roc_auc', 'disturbing_f1', 'quick_roc_auc',
                       'roc_auc_improvement', 'selection_criterion', 'stage2_rank'}
        param_keys = [c for c in exp_data['quick_results'].columns if c not in metric_cols]

    if not param_keys:
        print("WARNING: Could not determine param_keys, using defaults")
        param_keys = ['masking_ratio', 'masking_strategy', 'num_patches',
                      'margin_type', 'force_mask_anomaly', 'patch_level_loss', 'patchify_mode']

    # 1. Data Visualizations
    if not args.skip_data:
        data_dir = os.path.join(vis_dir, 'data')
        data_vis = DataVisualizer(data_dir, config)
        data_vis.generate_all()

    # 2. Architecture Visualizations
    if not args.skip_architecture:
        arch_dir = os.path.join(vis_dir, 'architecture')
        arch_vis = ArchitectureVisualizer(arch_dir, config)
        arch_vis.generate_all()

    # 3. Stage 1 Visualizations
    if not args.skip_experiments and exp_data['quick_results'] is not None:
        stage1_dir = os.path.join(vis_dir, 'stage1')
        stage1_vis = ExperimentVisualizer(exp_data['quick_results'], param_keys, stage1_dir)
        stage1_vis.generate_all()

    # 4. Stage 2 Visualizations
    if not args.skip_experiments and exp_data['full_results'] is not None:
        stage2_dir = os.path.join(vis_dir, 'stage2')
        stage2_vis = Stage2Visualizer(
            exp_data['full_results'],
            exp_data['quick_results'],
            exp_data['histories'] or {},
            stage2_dir
        )
        stage2_vis.generate_all()

    # 5. Best Model Visualizations
    if not args.skip_model and exp_data['model_path']:
        best_dir = os.path.join(vis_dir, 'best_model')
        model, config, test_loader, _ = load_best_model(exp_data['model_path'], args.num_test)
        best_vis = BestModelVisualizer(model, config, test_loader, best_dir)
        best_vis.generate_all(experiment_dir=experiment_dir)

    # 6. Training Progress Visualizations (requires re-training)
    if args.retrain and exp_data['best_config']:
        progress_dir = os.path.join(vis_dir, 'training_progress')
        progress_vis = TrainingProgressVisualizer(
            best_config=exp_data['best_config'],
            output_dir=progress_dir,
            checkpoint_epochs=[0, 5, 10, 20, 40, 60, 80, 100],
            num_train=2000,  # Stage 2 settings
            num_test=500,
            max_epochs=100
        )
        progress_vis.generate_all()

    print("\n" + "="*80)
    print(" " * 20 + "ALL VISUALIZATIONS COMPLETE!")
    print(f" " * 15 + f"Results saved to: {vis_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
