"""
Example Usage: Self-Distilled MAE for Anomaly Detection

This script demonstrates how to use the implementation programmatically
for custom workflows, experiments, or integration into larger systems.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from self_distilled_mae_anomaly_detection import (
    Config, set_seed, TimeSeriesAnomalyDataset,
    SelfDistilledMAE, Trainer, Evaluator
)


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
    train_dataset = TimeSeriesAnomalyDataset(
        num_samples=500,
        seq_length=config.seq_length,
        anomaly_ratio=0.05,
        is_train=True,
        seed=42
    )

    test_dataset = TimeSeriesAnomalyDataset(
        num_samples=200,
        seq_length=config.seq_length,
        anomaly_ratio=0.25,
        is_train=False,
        seed=43
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize model
    model = SelfDistilledMAE(config)

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
# Example 2: Custom Configuration
# ============================================================================

def example_2_custom_config():
    """Use custom hyperparameters"""
    print("\n" + "=" * 80)
    print("Example 2: Custom Configuration")
    print("=" * 80)

    set_seed(42)

    # Custom configuration
    config = Config()
    config.d_model = 128  # Larger embedding
    config.num_encoder_layers = 4  # Deeper encoder
    config.num_teacher_decoder_layers = 6  # Even deeper teacher
    config.masking_ratio = 0.75  # More aggressive masking
    config.margin = 2.0  # Larger margin for discrepancy
    config.num_epochs = 5

    print(f"Custom Config:")
    print(f"  d_model: {config.d_model}")
    print(f"  Encoder layers: {config.num_encoder_layers}")
    print(f"  Teacher decoder layers: {config.num_teacher_decoder_layers}")
    print(f"  Masking ratio: {config.masking_ratio}")

    # Quick training
    train_dataset = TimeSeriesAnomalyDataset(200, config.seq_length, 0.05, True, 42)
    test_dataset = TimeSeriesAnomalyDataset(100, config.seq_length, 0.25, False, 43)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = SelfDistilledMAE(config)
    trainer = Trainer(model, config, train_loader, test_loader)
    trainer.train()


# ============================================================================
# Example 3: Single Sample Inference
# ============================================================================

def example_3_single_sample_inference():
    """Perform inference on a single time series"""
    print("\n" + "=" * 80)
    print("Example 3: Single Sample Inference")
    print("=" * 80)

    set_seed(42)
    config = Config()

    # Create a trained model (using small dataset for demo)
    train_dataset = TimeSeriesAnomalyDataset(200, config.seq_length, 0.05, True, 42)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

    model = SelfDistilledMAE(config)
    config.num_epochs = 5
    trainer = Trainer(model, config, train_loader, test_loader)
    trainer.train()

    # Inference on single sample
    model.eval()
    with torch.no_grad():
        # Get a sample
        sample_sequence, sample_label = train_dataset[0]
        sample_sequence = sample_sequence.unsqueeze(0).to(config.device)  # (1, seq_len, 1)

        # Create mask for last 10 positions
        mask = torch.ones(1, config.seq_length, device=config.device)
        mask[:, -10:] = 0

        # Forward pass
        teacher_out, student_out, _, _ = model(sample_sequence, mask=mask)

        # Compute anomaly score
        masked_positions = (mask == 0).unsqueeze(-1)
        discrepancy = ((teacher_out - student_out) ** 2) * masked_positions
        anomaly_score = discrepancy.sum() / masked_positions.sum()

        print(f"Sample Label: {'Anomaly' if sample_label == 1 else 'Normal'}")
        print(f"Anomaly Score: {anomaly_score.item():.6f}")
        print(f"Higher score = More anomalous")


# ============================================================================
# Example 4: Batch Inference with Custom Threshold
# ============================================================================

def example_4_batch_inference():
    """Process multiple sequences and use custom threshold"""
    print("\n" + "=" * 80)
    print("Example 4: Batch Inference with Custom Threshold")
    print("=" * 80)

    set_seed(42)
    config = Config()

    # Train model
    train_dataset = TimeSeriesAnomalyDataset(200, config.seq_length, 0.05, True, 42)
    test_dataset = TimeSeriesAnomalyDataset(100, config.seq_length, 0.30, False, 43)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = SelfDistilledMAE(config)
    config.num_epochs = 5
    trainer = Trainer(model, config, train_loader, test_loader)
    trainer.train()

    # Batch inference
    evaluator = Evaluator(model, config, test_loader)
    scores, labels = evaluator.compute_anomaly_scores()

    # Use custom threshold
    custom_threshold = np.percentile(scores, 70)  # Top 30% are anomalies

    predictions = (scores > custom_threshold).astype(int)

    # Calculate metrics manually
    true_positives = np.sum((predictions == 1) & (labels == 1))
    false_positives = np.sum((predictions == 1) & (labels == 0))
    false_negatives = np.sum((predictions == 0) & (labels == 1))

    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    print(f"Custom Threshold: {custom_threshold:.4f}")
    print(f"Detected Anomalies: {predictions.sum()} / {len(predictions)}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")


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
    train_dataset = TimeSeriesAnomalyDataset(200, config.seq_length, 0.05, True, 42)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

    model = SelfDistilledMAE(config)
    trainer = Trainer(model, config, train_loader, test_loader)
    trainer.train()

    # Save model
    save_path = 'self_distilled_mae_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
    }, save_path)
    print(f"✓ Model saved to {save_path}")

    # Load model
    checkpoint = torch.load(save_path)
    loaded_config = checkpoint['config']

    new_model = SelfDistilledMAE(loaded_config)
    new_model.load_state_dict(checkpoint['model_state_dict'])
    new_model.eval()
    print(f"✓ Model loaded from {save_path}")

    # Verify loaded model works
    test_dataset = TimeSeriesAnomalyDataset(50, config.seq_length, 0.25, False, 43)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    evaluator = Evaluator(new_model, loaded_config, test_loader)
    scores, labels = evaluator.compute_anomaly_scores()

    print(f"✓ Inference on {len(scores)} samples successful")
    print(f"  Mean score: {scores.mean():.4f}")
    print(f"  Std score: {scores.std():.4f}")


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
    train_dataset = TimeSeriesAnomalyDataset(500, config.seq_length, 0.05, True, 42)
    test_dataset = TimeSeriesAnomalyDataset(100, config.seq_length, 0.30, False, 43)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = SelfDistilledMAE(config)
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
        sample, label = test_dataset[sample_idx]
        sample = sample.unsqueeze(0).to(config.device)

        # Create mask
        mask = torch.ones(1, config.seq_length, device=config.device)
        mask[:, -10:] = 0

        # Forward
        model.eval()
        with torch.no_grad():
            teacher_out, student_out, _, _ = model(sample, mask=mask)

        # Plot
        sample_np = sample.cpu().numpy()[0, :, 0]
        teacher_np = teacher_out.cpu().numpy()[0, :, 0]
        student_np = student_out.cpu().numpy()[0, :, 0]

        t = np.arange(config.seq_length)

        # Time series
        axes[idx, 0].plot(t, sample_np, label='Original', linewidth=2)
        axes[idx, 0].plot(t, teacher_np, label='Teacher', linewidth=2, alpha=0.7)
        axes[idx, 0].plot(t, student_np, label='Student', linewidth=2, alpha=0.7)
        axes[idx, 0].axvspan(config.seq_length - 10, config.seq_length, alpha=0.2, color='gray')
        axes[idx, 0].set_title(f'{title} Sample - Reconstructions')
        axes[idx, 0].legend()
        axes[idx, 0].grid(True, alpha=0.3)

        # Discrepancy
        discrepancy = (teacher_np - student_np) ** 2
        axes[idx, 1].plot(t, discrepancy, color='red', linewidth=2)
        axes[idx, 1].axvspan(config.seq_length - 10, config.seq_length, alpha=0.2, color='gray')
        axes[idx, 1].set_title(f'{title} Sample - Discrepancy (Score: {scores[sample_idx]:.4f})')
        axes[idx, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('example_anomaly_detection.png', dpi=150, bbox_inches='tight')
    print("✓ Visualization saved to example_anomaly_detection.png")
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

    # example_1_basic_usage()  # Basic training and evaluation
    # example_2_custom_config()  # Custom hyperparameters
    # example_3_single_sample_inference()  # Single sequence inference
    # example_4_batch_inference()  # Batch processing with custom threshold
    # example_5_save_load_model()  # Model persistence
    example_6_visualize_detection()  # Visualization

    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
