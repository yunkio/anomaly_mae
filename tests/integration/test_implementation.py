"""
Comprehensive test script for the point-level anomaly detection implementation
Tests dataset, model, training, and 3-way evaluation
"""

import torch
from multivariate_mae_experiments import (
    Config, SelfDistilledMAEMultivariate,
    MultivariateTimeSeriesDataset, Trainer, Evaluator
)
from torch.utils.data import DataLoader


def test_dataset():
    """Test dataset creation and point-level labels"""
    print('=' * 80)
    print('TEST 1: Dataset and Point-Level Labels')
    print('=' * 80)

    dataset = MultivariateTimeSeriesDataset(
        num_samples=20,
        seq_length=100,
        num_features=5,
        anomaly_ratio=0.5,
        is_train=True,
        seed=42
    )

    print(f'\n✓ Dataset created successfully')
    print(f'  - Total samples: {len(dataset)}')
    print(f'  - Data shape: {dataset.data.shape}')
    print(f'  - Sequence labels shape: {dataset.seq_labels.shape}')
    print(f'  - Point labels shape: {dataset.point_labels.shape}')

    # Test __getitem__
    seq, seq_label, point_label = dataset[0]
    print(f'\n✓ __getitem__ returns 3 values:')
    print(f'  - Sequence: {seq.shape}')
    print(f'  - Seq label: {seq_label.shape} (value: {seq_label.item()})')
    print(f'  - Point labels: {point_label.shape}')

    # Verify point labels for anomalous sequences
    anomaly_indices = [i for i in range(len(dataset)) if dataset.seq_labels[i] == 1]
    if anomaly_indices:
        idx = anomaly_indices[0]
        point_labels = dataset.point_labels[idx]
        num_anomalous_points = point_labels.sum()
        print(f'\n✓ Point-level labels verified:')
        print(f'  - Anomalous sequence index: {idx}')
        print(f'  - Number of anomalous points: {num_anomalous_points}/100')
        print(f'  - Anomalous time steps: {list(point_labels.nonzero()[0])}')

    # Test DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    sequences, seq_labels, point_labels = next(iter(loader))
    print(f'\n✓ DataLoader works correctly:')
    print(f'  - Batch sequences: {sequences.shape}')
    print(f'  - Batch seq_labels: {seq_labels.shape}')
    print(f'  - Batch point_labels: {point_labels.shape}')

    return True


def test_model_forward():
    """Test model creation and forward pass"""
    print('\n' + '=' * 80)
    print('TEST 2: Model Creation and Forward Pass')
    print('=' * 80)

    config = Config()
    config.device = 'cpu'
    config.masking_strategy = 'patch'

    # Create model
    model = SelfDistilledMAEMultivariate(config)
    print(f'\n✓ Model created successfully')
    print(f'  - Masking strategy: {config.masking_strategy}')
    print(f'  - Patch size: {config.patch_size}')
    print(f'  - d_model: {config.d_model}')

    # Create test input
    batch_size = 4
    x = torch.randn(batch_size, config.seq_length, config.num_features)

    # Forward pass
    model.eval()
    with torch.no_grad():
        teacher_output, student_output, mask = model(x)

    print(f'\n✓ Forward pass successful:')
    print(f'  - Input: {x.shape}')
    print(f'  - Teacher output: {teacher_output.shape}')
    print(f'  - Student output: {student_output.shape}')
    print(f'  - Mask: {mask.shape}')

    return True


def test_training():
    """Test training process"""
    print('\n' + '=' * 80)
    print('TEST 3: Training Process')
    print('=' * 80)

    # Create config
    config = Config()
    config.device = 'cpu'
    config.num_train_samples = 50
    config.num_test_samples = 20
    config.num_epochs = 3
    config.masking_strategy = 'patch'

    # Create datasets
    train_dataset = MultivariateTimeSeriesDataset(
        num_samples=config.num_train_samples,
        seq_length=config.seq_length,
        num_features=config.num_features,
        anomaly_ratio=config.train_anomaly_ratio,
        is_train=True,
        seed=42
    )

    test_dataset = MultivariateTimeSeriesDataset(
        num_samples=config.num_test_samples,
        seq_length=config.seq_length,
        num_features=config.num_features,
        anomaly_ratio=config.test_anomaly_ratio,
        is_train=False,
        seed=43
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    print(f'\n✓ Datasets created:')
    print(f'  - Train samples: {len(train_dataset)} (anomaly ratio: {config.train_anomaly_ratio})')
    print(f'  - Test samples: {len(test_dataset)} (anomaly ratio: {config.test_anomaly_ratio})')

    # Create and train model
    model = SelfDistilledMAEMultivariate(config)
    trainer = Trainer(model, config, train_loader, test_loader)

    print(f'\n✓ Training for {config.num_epochs} epochs...')
    trainer.train()
    print(f'\n✓ Training completed successfully')

    return model, config, test_loader


def test_3way_evaluation(model, config, test_loader):
    """Test 3-way evaluation (sequence/point/combined)"""
    print('\n' + '=' * 80)
    print('TEST 4: 3-Way Evaluation')
    print('=' * 80)

    evaluator = Evaluator(model, config, test_loader)
    results = evaluator.evaluate()

    # Verify results structure
    print('\n' + '=' * 80)
    print('VERIFICATION OF RESULTS STRUCTURE')
    print('=' * 80)

    assert 'sequence' in results, "Missing 'sequence' in results"
    assert 'point' in results, "Missing 'point' in results"
    assert 'combined' in results, "Missing 'combined' in results"

    print(f'\n✓ Results structure verified:')
    print(f'  - Sequence metrics: {list(results["sequence"].keys())}')
    print(f'  - Point metrics: {list(results["point"].keys())}')
    print(f'  - Combined metrics: {list(results["combined"].keys())}')

    # Check combined weights
    seq_weight = results['combined']['seq_weight']
    point_weight = results['combined']['point_weight']
    print(f'\n✓ Combined weights:')
    print(f'  - Sequence weight: {seq_weight:.4f}')
    print(f'  - Point weight: {point_weight:.4f}')
    print(f'  - Sum: {seq_weight + point_weight:.4f} (should be 1.0)')

    assert abs(seq_weight + point_weight - 1.0) < 1e-6, "Weights don't sum to 1.0"

    # Display final results
    print('\n' + '=' * 80)
    print('FINAL PERFORMANCE SUMMARY')
    print('=' * 80)
    print(f'\nSequence-Level Performance:')
    print(f'  - ROC-AUC: {results["sequence"]["roc_auc"]:.4f}')
    print(f'  - F1-Score: {results["sequence"]["f1_score"]:.4f}')

    print(f'\nPoint-Level Performance:')
    print(f'  - ROC-AUC: {results["point"]["roc_auc"]:.4f}')
    print(f'  - F1-Score: {results["point"]["f1_score"]:.4f}')

    print(f'\nCombined Performance:')
    print(f'  - ROC-AUC: {results["combined"]["roc_auc"]:.4f}')
    print(f'  - F1-Score: {results["combined"]["f1_score"]:.4f}')

    return True


def main():
    """Run all tests"""
    print('\n' + '=' * 80)
    print('COMPREHENSIVE IMPLEMENTATION TEST')
    print('Point-Level Anomaly Detection for Multivariate Time Series')
    print('=' * 80)

    try:
        # Test 1: Dataset
        test_dataset()

        # Test 2: Model
        test_model_forward()

        # Test 3: Training
        model, config, test_loader = test_training()

        # Test 4: 3-way evaluation
        test_3way_evaluation(model, config, test_loader)

        # Final summary
        print('\n' + '=' * 80)
        print('ALL TESTS PASSED SUCCESSFULLY!')
        print('=' * 80)
        print('\n✓ Dataset creation with point-level labels')
        print('✓ Model forward pass')
        print('✓ Training process')
        print('✓ 3-way evaluation (sequence/point/combined)')
        print('\nThe implementation is working correctly!')
        print('=' * 80)

    except Exception as e:
        print('\n' + '=' * 80)
        print('TEST FAILED!')
        print('=' * 80)
        print(f'\nError: {type(e).__name__}: {str(e)}')
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
