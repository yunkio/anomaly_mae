"""
Quick test script to verify Self-Distilled MAE implementation
This runs a minimal version to check if everything works correctly
"""

import sys
import numpy as np

def test_imports():
    """Test if all required packages are installed"""
    print("Testing imports...")
    required_packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'sklearn': 'scikit-learn',
        'tqdm': 'tqdm'
    }

    missing_packages = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - MISSING")
            missing_packages.append(name)

    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("\nPlease install PyTorch first:")
        print("  pip install torch torchvision torchaudio")
        print("\nThen install other dependencies:")
        print("  pip install -r requirements.txt")
        return False

    print("\n✓ All packages installed successfully!\n")
    return True


def test_components():
    """Test individual components"""
    import torch
    from self_distilled_mae_anomaly_detection import (
        Config, TimeSeriesAnomalyDataset, SelfDistilledMAE,
        SelfDistillationLoss, set_seed
    )

    print("Testing components...")

    # Set seed
    set_seed(42)
    print("  ✓ Random seed set")

    # Test config
    config = Config()
    config.num_epochs = 2  # Quick test
    config.num_train_samples = 100
    config.num_test_samples = 50
    print("  ✓ Configuration created")

    # Test dataset
    print("\n  Testing dataset generation...")
    train_dataset = TimeSeriesAnomalyDataset(
        num_samples=10,
        seq_length=50,
        anomaly_ratio=0.3,
        is_train=True,
        seed=42
    )
    sequence, label = train_dataset[0]
    assert sequence.shape == (50, 1), f"Expected shape (50, 1), got {sequence.shape}"
    assert label in [0, 1], f"Expected label in [0, 1], got {label}"
    print(f"    ✓ Dataset shape: {sequence.shape}, Label: {label}")

    # Test model
    print("\n  Testing model architecture...")
    config.seq_length = 50
    model = SelfDistilledMAE(config)

    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 50, 1)
    teacher_out, student_out, mask, latent = model(x)

    assert teacher_out.shape == (batch_size, 50, 1)
    assert student_out.shape == (batch_size, 50, 1)
    assert mask.shape == (batch_size, 50)
    print(f"    ✓ Teacher output shape: {teacher_out.shape}")
    print(f"    ✓ Student output shape: {student_out.shape}")
    print(f"    ✓ Mask shape: {mask.shape}")
    print(f"    ✓ Latent shape: {latent.shape}")

    # Test loss function
    print("\n  Testing loss function...")
    criterion = SelfDistillationLoss(margin=1.0, lambda_disc=0.5)
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    loss, loss_dict = criterion(
        teacher_out, student_out, x, mask, labels, warmup_factor=0.5
    )

    assert loss.item() > 0, "Loss should be positive"
    assert 'reconstruction_loss' in loss_dict
    assert 'discrepancy_loss' in loss_dict
    print(f"    ✓ Total loss: {loss.item():.4f}")
    print(f"    ✓ Reconstruction loss: {loss_dict['reconstruction_loss']:.4f}")
    print(f"    ✓ Discrepancy loss: {loss_dict['discrepancy_loss']:.4f}")

    print("\n✓ All components tested successfully!\n")
    return True


def test_mini_training():
    """Test a minimal training loop"""
    import torch
    from torch.utils.data import DataLoader
    from self_distilled_mae_anomaly_detection import (
        Config, TimeSeriesAnomalyDataset, SelfDistilledMAE,
        Trainer, set_seed
    )

    print("Testing mini training loop (2 epochs, 100 samples)...")

    set_seed(42)

    # Minimal config
    config = Config()
    config.num_epochs = 2
    config.num_train_samples = 100
    config.num_test_samples = 50
    config.batch_size = 16

    # Create datasets
    train_dataset = TimeSeriesAnomalyDataset(
        num_samples=config.num_train_samples,
        seq_length=config.seq_length,
        anomaly_ratio=0.05,
        is_train=True,
        seed=42
    )

    test_dataset = TimeSeriesAnomalyDataset(
        num_samples=config.num_test_samples,
        seq_length=config.seq_length,
        anomaly_ratio=0.25,
        is_train=False,
        seed=43
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # Create model
    model = SelfDistilledMAE(config)

    # Create trainer
    trainer = Trainer(model, config, train_loader, test_loader)

    # Train
    print("\n  Training...")
    trainer.train()

    print("\n✓ Mini training completed successfully!\n")
    return True


def main():
    """Run all tests"""
    print("=" * 80)
    print("Self-Distilled MAE - Quick Test Suite")
    print("=" * 80 + "\n")

    # Test 1: Imports
    if not test_imports():
        sys.exit(1)

    # Test 2: Components
    try:
        test_components()
    except Exception as e:
        print(f"\n✗ Component test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Test 3: Mini training
    try:
        test_mini_training()
    except Exception as e:
        print(f"\n✗ Training test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("=" * 80)
    print("✓ All tests passed! The implementation is working correctly.")
    print("=" * 80)
    print("\nYou can now run the full training:")
    print("  python self_distilled_mae_anomaly_detection.py")
    print()


if __name__ == "__main__":
    main()
