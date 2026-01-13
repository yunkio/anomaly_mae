"""
Quick test to verify the experiment fix
Tests that the nested metrics structure is properly handled
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

def test_experiment_runner():
    """Test that ExperimentRunner correctly handles nested metrics"""

    print("="*80)
    print("Testing ExperimentRunner with nested metrics structure")
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
    config.num_epochs = 3  # Short test
    config.batch_size = 8

    # Create experiment runner
    runner = ExperimentRunner(config)

    # Run a single experiment
    print("\nRunning single experiment...")
    result = runner.run_single_experiment(
        config=config,
        experiment_name="Test Experiment",
        train_dataset=train_dataset,
        test_dataset=test_dataset
    )

    print("\n" + "="*80)
    print("VERIFICATION")
    print("="*80)

    # Verify the result structure
    assert 'experiment_name' in result
    assert 'metrics' in result
    assert 'config' in result
    assert 'history' in result

    metrics = result['metrics']

    # Verify nested structure
    assert 'sequence' in metrics
    assert 'point' in metrics
    assert 'combined' in metrics

    # Verify each level has required metrics
    for level in ['sequence', 'point', 'combined']:
        assert 'roc_auc' in metrics[level]
        assert 'f1_score' in metrics[level]
        assert 'precision' in metrics[level]
        assert 'recall' in metrics[level]

    print("✅ All checks passed!")
    print("\nMetrics structure:")
    print(f"  Sequence-Level: ROC-AUC={metrics['sequence']['roc_auc']:.4f}, F1={metrics['sequence']['f1_score']:.4f}")
    print(f"  Point-Level: ROC-AUC={metrics['point']['roc_auc']:.4f}, F1={metrics['point']['f1_score']:.4f}")
    print(f"  Combined: ROC-AUC={metrics['combined']['roc_auc']:.4f}, F1={metrics['combined']['f1_score']:.4f}")
    print(f"  Weights: Sequence={metrics['combined']['seq_weight']:.4f}, Point={metrics['combined']['point_weight']:.4f}")

    print("\n" + "="*80)
    print("TEST PASSED - No KeyError!")
    print("="*80)

if __name__ == "__main__":
    test_experiment_runner()
