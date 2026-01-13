"""
Verify that Token and Temporal masking produce visually different mask patterns
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from multivariate_mae_experiments import Config, SelfDistilledMAEMultivariate

print("="*80)
print("Detailed Mask Pattern Verification")
print("="*80)

# Create a small test input
batch_size = 4
seq_len = 20
num_features = 5  # Must match Config.num_features

x = torch.randn(batch_size, seq_len, num_features)

# Test Token masking
print("\n1. Token Masking (BERT style - independent position masking)")
print("-" * 80)
config_token = Config()
config_token.masking_strategy = 'token'
config_token.masking_ratio = 0.6

model_token = SelfDistilledMAEMultivariate(config_token)
model_token.eval()

with torch.no_grad():
    _, _, mask_token = model_token(x)

mask_token_np = mask_token.cpu().numpy()
print(f"Mask shape: {mask_token_np.shape}")
print(f"Masking ratio: {(mask_token_np == 0).sum() / mask_token_np.size:.4f}")
print(f"\nFirst 10x10 section of mask (1=keep, 0=mask):")
print(mask_token_np[:10, :10].astype(int))

# Test Temporal masking
print("\n2. Temporal Masking (mask entire time steps)")
print("-" * 80)
config_temporal = Config()
config_temporal.masking_strategy = 'temporal'
config_temporal.masking_ratio = 0.6

model_temporal = SelfDistilledMAEMultivariate(config_temporal)
model_temporal.eval()

with torch.no_grad():
    _, _, mask_temporal = model_temporal(x)

mask_temporal_np = mask_temporal.cpu().numpy()
print(f"Mask shape: {mask_temporal_np.shape}")
print(f"Masking ratio: {(mask_temporal_np == 0).sum() / mask_temporal_np.size:.4f}")
print(f"\nFirst 10x10 section of mask (1=keep, 0=mask):")
print(mask_temporal_np[:10, :10].astype(int))

# Visualize the difference
print("\n3. Visual Comparison")
print("-" * 80)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Token masking
im1 = axes[0].imshow(mask_token_np[:, :50].T, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
axes[0].set_title('Token Masking\n(Independent position masking)', fontsize=12)
axes[0].set_xlabel('Batch Index')
axes[0].set_ylabel('Time Step')
plt.colorbar(im1, ax=axes[0], label='1=Keep, 0=Mask')

# Temporal masking
im2 = axes[1].imshow(mask_temporal_np[:, :50].T, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
axes[1].set_title('Temporal Masking\n(Entire time steps masked together)', fontsize=12)
axes[1].set_xlabel('Batch Index')
axes[1].set_ylabel('Time Step')
plt.colorbar(im2, ax=axes[1], label='1=Keep, 0=Mask')

plt.tight_layout()
plt.savefig('mask_pattern_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved mask_pattern_comparison.png")

# Check if patterns are different
print("\n4. Statistical Comparison")
print("-" * 80)
print(f"Token mask - unique rows: {len(np.unique(mask_token_np, axis=1))}")
print(f"Temporal mask - unique rows: {len(np.unique(mask_temporal_np, axis=1))}")

# Check if temporal masking has same pattern across batch dimension
temporal_same_across_batch = np.all(mask_temporal_np == mask_temporal_np[0:1, :], axis=0).sum() / mask_temporal_np.shape[1]
print(f"\nTemporal masking - ratio of time steps with same mask across all batches: {temporal_same_across_batch:.2%}")

token_same_across_batch = np.all(mask_token_np == mask_token_np[0:1, :], axis=0).sum() / mask_token_np.shape[1]
print(f"Token masking - ratio of time steps with same mask across all batches: {token_same_across_batch:.2%}")

if temporal_same_across_batch > 0.9:
    print("\n✅ Temporal masking correctly masks entire time steps uniformly across batch!")
else:
    print("\n⚠️  Temporal masking might not be working as expected")

if token_same_across_batch < 0.3:
    print("✅ Token masking correctly varies independently for each position!")
else:
    print("⚠️  Token masking might not be working as expected")

print("\n" + "="*80)
print("Verification Complete!")
print("="*80)
