"""
Example Usage: Self-Distilled MAE for Anomaly Detection

This script demonstrates how to use the implementation programmatically
for custom workflows, experiments, or integration into larger systems.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from mae_anomaly import (
    Config, set_seed,
    SlidingWindowTimeSeriesGenerator, SlidingWindowDataset,
    SelfDistilledMAEMultivariate, Trainer, Evaluator
)


def create_datasets(config, total_length=100000):
    """Helper function to create sliding window datasets"""
    generator = SlidingWindowTimeSeriesGenerator(
        total_length=total_length,
        num_features=config.num_features,
        interval_scale=config.anomaly_interval_scale,
        seed=config.random_seed
    )
    signals, point_labels, anomaly_regions = generator.generate()

    train_dataset = SlidingWindowDataset(
        signals=signals,
        point_labels=point_labels,
        anomaly_regions=anomaly_regions,
        window_size=config.seq_length,
        stride=config.sliding_window_stride,
        mask_last_n=10,
        split='train',
        train_ratio=0.5,
        seed=config.random_seed
    )

    test_dataset = SlidingWindowDataset(
        signals=signals,
        point_labels=point_labels,
        anomaly_regions=anomaly_regions,
        window_size=config.seq_length,
        stride=config.sliding_window_stride,
        mask_last_n=10,
        split='test',
        train_ratio=0.5,
        target_counts={'pure_normal': 120, 'disturbing_normal': 30, 'anomaly': 50},
        seed=config.random_seed
    )

    return train_dataset, test_dataset, signals, point_labels, anomaly_regions


# ============================================================================
# Example 1: Basic Usage with Default Settings
# ============================================================================

def example_1_basic_usage():
    """Train and evaluate with default configuration"""
    print("=" * 80)
    print("Example 1: Basic Usage")
    print("=" * 80)

    # Initialize
    set_seed(42)
    config = Config()
    config.num_epochs = 10  # Quick demo

    # Create datasets
    train_dataset, test_dataset, _, _, _ = create_datasets(config, total_length=100000)

    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Test dataset: {len(test_dataset)} samples")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize model
    model = SelfDistilledMAEMultivariate(config)

    # Train
    trainer = Trainer(model, config, train_loader, test_loader)
    trainer.train()

    # Evaluate
    evaluator = Evaluator(model, config, test_loader)
    metrics = evaluator.evaluate()

    print(f"\nFinal Results:")
    print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"  F1-Score: {metrics['f1_score']:.4f}")


# ============================================================================
# Example 2: Custom Configuration with Patchify Mode
# ============================================================================

def example_2_custom_config():
    """Use custom hyperparameters and patchify mode"""
    print("\n" + "=" * 80)
    print("Example 2: Custom Configuration with Patchify Mode")
    print("=" * 80)

    set_seed(42)

    # Custom configuration
    config = Config()
    config.d_model = 128  # Larger embedding
    config.num_encoder_layers = 4  # Deeper encoder
    config.num_teacher_decoder_layers = 6  # Even deeper teacher
    config.masking_ratio = 0.5  # 50% masking
    config.patchify_mode = 'patch_cnn'  # Use patch CNN mode
    config.num_epochs = 5

    print(f"Custom Config:")
    print(f"  d_model: {config.d_model}")
    print(f"  Encoder layers: {config.num_encoder_layers}")
    print(f"  Teacher decoder layers: {config.num_teacher_decoder_layers}")
    print(f"  Masking ratio: {config.masking_ratio}")
    print(f"  Patchify mode: {config.patchify_mode}")

    # Quick training
    train_dataset, test_dataset, _, _, _ = create_datasets(config, total_length=50000)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = SelfDistilledMAEMultivariate(config)
    trainer = Trainer(model, config, train_loader, test_loader)
    trainer.train()


# ============================================================================
# Example 3: Compare Patchify Modes
# ============================================================================

def example_3_compare_patchify_modes():
    """Compare different patchify modes"""
    print("\n" + "=" * 80)
    print("Example 3: Compare Patchify Modes")
    print("=" * 80)

    set_seed(42)

    # Create shared datasets
    config = Config()
    config.num_epochs = 5

    train_dataset, test_dataset, _, _, _ = create_datasets(config, total_length=80000)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    results = {}

    for mode in ['linear', 'cnn_first', 'patch_cnn']:
        print(f"\n--- Training with patchify_mode='{mode}' ---")
        set_seed(42)

        config = Config()
        config.num_epochs = 5
        config.patchify_mode = mode

        model = SelfDistilledMAEMultivariate(config)
        trainer = Trainer(model, config, train_loader, test_loader)
        trainer.train()

        evaluator = Evaluator(model, config, test_loader)
        metrics = evaluator.evaluate()

        results[mode] = metrics['roc_auc']
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")

    print("\n--- Summary ---")
    for mode, auc in results.items():
        print(f"  {mode}: {auc:.4f}")


# ============================================================================
# Example 4: Single Sample Inference
# ============================================================================

def example_4_single_sample_inference():
    """Perform inference on a single time series"""
    print("\n" + "=" * 80)
    print("Example 4: Single Sample Inference")
    print("=" * 80)

    set_seed(42)
    config = Config()

    # Create a trained model (using small dataset for demo)
    train_dataset, test_dataset, _, _, _ = create_datasets(config, total_length=50000)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = SelfDistilledMAEMultivariate(config)
    config.num_epochs = 5
    trainer = Trainer(model, config, train_loader, test_loader)
    trainer.train()

    # Inference on single sample
    model.eval()
    with torch.no_grad():
        # Get a sample
        sample_sequence, sample_label, _, _, _ = train_dataset[0]
        sample_sequence = sample_sequence.unsqueeze(0).to(config.device)

        # Create mask for last patch
        mask = torch.ones(1, config.num_patches, device=config.device)
        mask[:, -1] = 0  # Mask last patch

        # Forward pass
        teacher_out, student_out, _, _ = model(sample_sequence, mask=mask)

        # Compute anomaly score (discrepancy in masked region)
        discrepancy = (teacher_out - student_out) ** 2
        anomaly_score = discrepancy.mean().item()

        print(f"Sample Label: {'Anomaly' if sample_label == 1 else 'Normal'}")
        print(f"Anomaly Score: {anomaly_score:.6f}")
        print(f"Higher score = More anomalous")


# ============================================================================
# Example 5: Model Save/Load
# ============================================================================

def example_5_save_load_model():
    """Save and load trained model"""
    print("\n" + "=" * 80)
    print("Example 5: Save and Load Model")
    print("=" * 80)

    set_seed(42)
    config = Config()
    config.num_epochs = 5

    # Train model
    train_dataset, test_dataset, _, _, _ = create_datasets(config, total_length=50000)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = SelfDistilledMAEMultivariate(config)
    trainer = Trainer(model, config, train_loader, test_loader)
    trainer.train()

    # Save model
    save_path = 'self_distilled_mae_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
    }, save_path)
    print(f"Model saved to {save_path}")

    # Load model
    checkpoint = torch.load(save_path)
    loaded_config = checkpoint['config']

    new_model = SelfDistilledMAEMultivariate(loaded_config)
    new_model.load_state_dict(checkpoint['model_state_dict'])
    new_model.eval()
    print(f"Model loaded from {save_path}")

    # Verify loaded model works
    evaluator = Evaluator(new_model, loaded_config, test_loader)
    metrics = evaluator.evaluate()

    print(f"Inference successful")
    print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")


# ============================================================================
# Example 6: Visualize Anomaly Detection
# ============================================================================

def example_6_visualize_detection():
    """Visualize anomaly detection on custom sequences"""
    print("\n" + "=" * 80)
    print("Example 6: Visualize Anomaly Detection")
    print("=" * 80)

    set_seed(42)
    config = Config()
    config.num_epochs = 10

    # Train model
    train_dataset, test_dataset, _, _, _ = create_datasets(config, total_length=100000)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = SelfDistilledMAEMultivariate(config)
    trainer = Trainer(model, config, train_loader, test_loader)
    trainer.train()

    # Get scores
    evaluator = Evaluator(model, config, test_loader)
    scores, labels = evaluator.compute_anomaly_scores()

    # Find examples
    normal_idx = np.where(labels == 0)[0][0]
    anomaly_idx = np.where(labels == 1)[0][0]

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, (sample_idx, title) in enumerate([(normal_idx, 'Normal'), (anomaly_idx, 'Anomaly')]):
        # Get sample
        sample, label, _, _, _ = test_dataset[sample_idx]
        sample = sample.unsqueeze(0).to(config.device)

        # Create mask for last patch
        mask = torch.ones(1, config.num_patches, device=config.device)
        mask[:, -1] = 0

        # Forward
        model.eval()
        with torch.no_grad():
            teacher_out, student_out, _, _ = model(sample, mask=mask)

        # Plot first feature
        sample_np = sample.cpu().numpy()[0, :, 0]
        teacher_np = teacher_out.cpu().numpy()[0, :, 0]
        student_np = student_out.cpu().numpy()[0, :, 0]

        t = np.arange(config.seq_length)

        # Time series
        axes[idx, 0].plot(t, sample_np, label='Original', linewidth=2)
        axes[idx, 0].plot(t, teacher_np, label='Teacher', linewidth=2, alpha=0.7)
        axes[idx, 0].plot(t, student_np, label='Student', linewidth=2, alpha=0.7)
        axes[idx, 0].axvspan(config.seq_length - config.patch_size, config.seq_length, alpha=0.2, color='gray')
        axes[idx, 0].set_title(f'{title} Sample - Reconstructions (Feature 0)')
        axes[idx, 0].legend()
        axes[idx, 0].grid(True, alpha=0.3)

        # Discrepancy
        discrepancy = (teacher_np - student_np) ** 2
        axes[idx, 1].plot(t, discrepancy, color='red', linewidth=2)
        axes[idx, 1].axvspan(config.seq_length - config.patch_size, config.seq_length, alpha=0.2, color='gray')
        axes[idx, 1].set_title(f'{title} Sample - Discrepancy (Score: {scores[sample_idx]:.4f})')
        axes[idx, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('example_anomaly_detection.png', dpi=150, bbox_inches='tight')
    print("Visualization saved to example_anomaly_detection.png")
    plt.show()


# ============================================================================
# Main: Run All Examples
# ============================================================================

def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print("Self-Distilled MAE - Usage Examples")
    print("=" * 80 + "\n")

    # Uncomment the examples you want to run:

    example_1_basic_usage()  # Basic training and evaluation
    # example_2_custom_config()  # Custom hyperparameters with patchify mode
    # example_3_compare_patchify_modes()  # Compare linear vs cnn_first vs patch_cnn
    # example_4_single_sample_inference()  # Single sequence inference
    # example_5_save_load_model()  # Model persistence
    # example_6_visualize_detection()  # Visualization

    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
