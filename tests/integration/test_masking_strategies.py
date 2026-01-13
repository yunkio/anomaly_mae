"""
Test that different masking strategies produce different results
"""

import torch
from multivariate_mae_experiments import (
    Config,
    SelfDistilledMAEMultivariate,
    MultivariateTimeSeriesDataset,
    Trainer,
    Evaluator,
    ExperimentRunner
)

print("="*80)
print("Masking Strategy 차이 검증")
print("="*80)

# Create small test dataset
test_dataset = MultivariateTimeSeriesDataset(
    num_samples=20,
    seq_length=100,
    num_features=5,
    anomaly_ratio=0.25,
    is_train=False
)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)

# Test different masking strategies
strategies = ['token', 'temporal', 'feature_wise']
results = {}

for strategy in strategies:
    print(f"\n테스트 중: {strategy} masking")

    config = Config()
    config.masking_strategy = strategy
    config.num_epochs = 0  # No training, just test forward pass

    model = SelfDistilledMAEMultivariate(config)
    model.eval()

    # Get one batch
    batch = next(iter(test_loader))
    x, seq_labels, point_labels = batch

    # Forward pass
    with torch.no_grad():
        teacher_output, student_output, mask = model(x)

    # Record mask pattern
    mask_np = mask.cpu().numpy()

    results[strategy] = {
        'mask_mean': mask_np.mean(),
        'mask_std': mask_np.std(),
        'mask_shape': mask_np.shape,
        'num_masked': (mask_np == 0).sum(),
        'num_unmasked': (mask_np == 1).sum()
    }

    print(f"  Mask shape: {mask_np.shape}")
    print(f"  Masking ratio: {(mask_np == 0).sum() / mask_np.size:.4f}")
    print(f"  Mask mean: {mask_np.mean():.4f}")
    print(f"  Mask std: {mask_np.std():.4f}")

print("\n" + "="*80)
print("결과 비교")
print("="*80)

# Compare token vs temporal
token_mask = results['token']['mask_mean']
temporal_mask = results['temporal']['mask_mean']
feature_mask = results['feature_wise']['mask_mean']

print(f"\nToken masking mean: {token_mask:.4f}")
print(f"Temporal masking mean: {temporal_mask:.4f}")
print(f"Feature-wise masking mean: {feature_mask:.4f}")

if abs(token_mask - temporal_mask) < 0.01:
    print("\n⚠️  WARNING: Token과 Temporal masking이 매우 유사합니다!")
    print("  두 전략이 다르게 동작하지 않을 수 있습니다.")
else:
    print("\n✅ Token과 Temporal masking이 다르게 동작합니다!")

print("\n" + "="*80)
print("세부 실험 실행 (3 epochs)")
print("="*80)

# Run quick experiments with different strategies
train_dataset = MultivariateTimeSeriesDataset(
    num_samples=50,
    seq_length=100,
    num_features=5,
    anomaly_ratio=0.05,
    is_train=True
)

quick_results = {}

for strategy in ['token', 'temporal']:
    print(f"\n실험: {strategy} masking (3 epochs)")

    config = Config()
    config.masking_strategy = strategy
    config.num_epochs = 3
    config.batch_size = 8

    model = SelfDistilledMAEMultivariate(config)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Train
    trainer = Trainer(model, config, train_loader, test_loader)
    trainer.train()

    # Evaluate
    evaluator = Evaluator(model, config, test_loader)
    metrics = evaluator.evaluate()

    quick_results[strategy] = {
        'combined_f1': metrics['combined']['f1_score'],
        'sequence_f1': metrics['sequence']['f1_score'],
        'point_f1': metrics['point']['f1_score']
    }

    print(f"  Combined F1: {metrics['combined']['f1_score']:.4f}")

print("\n" + "="*80)
print("최종 검증")
print("="*80)

token_f1 = quick_results['token']['combined_f1']
temporal_f1 = quick_results['temporal']['combined_f1']

print(f"\nToken masking F1: {token_f1:.4f}")
print(f"Temporal masking F1: {temporal_f1:.4f}")
print(f"차이: {abs(token_f1 - temporal_f1):.4f}")

if abs(token_f1 - temporal_f1) < 0.001:
    print("\n⚠️  경고: 결과가 여전히 매우 유사합니다!")
    print("  코드 수정이 제대로 적용되지 않았을 수 있습니다.")
else:
    print("\n✅ Token과 Temporal masking이 다른 결과를 생성합니다!")
    print("  코드 수정이 성공적으로 적용되었습니다!")
