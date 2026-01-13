"""
Quick test to verify visualization fix
Tests that all visualization methods work with nested metrics
"""

import torch
from torch.utils.data import DataLoader
from multivariate_mae_experiments import (
    Config,
    SelfDistilledMAEMultivariate,
    MultivariateTimeSeriesDataset,
    Trainer,
    Evaluator,
    ExperimentRunner
)

def test_visualization():
    """Test that visualizations work with nested metrics structure"""

    print("="*80)
    print("Testing Visualizations with nested metrics structure")
    print("="*80)

    # Create small datasets
    train_dataset = MultivariateTimeSeriesDataset(
        num_samples=50,
        seq_length=100,
        num_features=5,
        anomaly_ratio=0.05,
        is_train=True
    )

    test_dataset = MultivariateTimeSeriesDataset(
        num_samples=20,
        seq_length=100,
        num_features=5,
        anomaly_ratio=0.25,
        is_train=False
    )

    # Create config
    config = Config()
    config.num_epochs = 2  # Very short test
    config.batch_size = 8

    # Create experiment runner
    runner = ExperimentRunner(config)

    print("\nRunning multiple experiments for visualization test...")

    # Run baseline
    baseline_config = config
    runner.run_single_experiment(baseline_config, "Baseline", train_dataset, test_dataset)

    # Run one ablation
    ablation_config = Config()
    ablation_config.num_epochs = 2
    ablation_config.use_teacher = False
    runner.run_single_experiment(ablation_config, "Ablation: StudentOnly", train_dataset, test_dataset)

    # Run one hyperparameter variant
    hp_config = Config()
    hp_config.num_epochs = 2
    hp_config.masking_ratio = 0.75
    runner.run_single_experiment(hp_config, "MaskRatio=0.75", train_dataset, test_dataset)

    print("\n" + "="*80)
    print("Generating visualizations...")
    print("="*80)

    try:
        runner.generate_visualizations()
        print("\n✅ All visualizations generated successfully!")
        print("✅ No KeyError!")
    except Exception as e:
        print(f"\n❌ Visualization failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "="*80)
    print("TEST PASSED - Visualizations work correctly!")
    print("="*80)
    return True

if __name__ == "__main__":
    success = test_visualization()
    exit(0 if success else 1)
