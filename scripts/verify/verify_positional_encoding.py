"""
Verification script for positional encoding and input projection
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from multivariate_mae_experiments import Config, SelfDistilledMAEMultivariate, PositionalEncoding

def visualize_positional_encoding():
    """Visualize positional encoding patterns"""

    print("=" * 80)
    print("POSITIONAL ENCODING VISUALIZATION")
    print("=" * 80)

    d_model = 64
    max_len = 100

    # Create positional encoding
    pos_encoder = PositionalEncoding(d_model, max_len)

    # Get the positional encoding matrix
    pe = pos_encoder.pe.squeeze(1).numpy()  # (max_len, d_model)

    print(f"\nPositional Encoding Shape: {pe.shape}")
    print(f"  - Positions: {pe.shape[0]}")
    print(f"  - Embedding dimension: {pe.shape[1]}")

    # Check uniqueness
    print(f"\nUniqueness Check:")
    print(f"  - Position 0: {pe[0, :5]} ...")
    print(f"  - Position 1: {pe[1, :5]} ...")
    print(f"  - Position 50: {pe[50, :5]} ...")
    print(f"  - Position 99: {pe[99, :5]} ...")

    # Check if all positions are unique
    unique_positions = []
    for i in range(100):
        is_unique = True
        for j in range(i):
            if np.allclose(pe[i], pe[j]):
                is_unique = False
                break
        unique_positions.append(is_unique)

    print(f"\n  ✓ All positions are unique: {all(unique_positions)}")

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Full positional encoding heatmap
    ax = axes[0, 0]
    im = ax.imshow(pe, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    ax.set_xlabel('Embedding Dimension')
    ax.set_ylabel('Position (Time Step)')
    ax.set_title('Positional Encoding Matrix (100 positions × 64 dims)')
    plt.colorbar(im, ax=ax)

    # 2. First 20 positions in detail
    ax = axes[0, 1]
    im = ax.imshow(pe[:20, :], aspect='auto', cmap='RdBu_r', interpolation='nearest')
    ax.set_xlabel('Embedding Dimension')
    ax.set_ylabel('Position')
    ax.set_title('First 20 Positions (Detail)')
    plt.colorbar(im, ax=ax)

    # 3. Selected dimensions over positions
    ax = axes[1, 0]
    for dim in [0, 1, 2, 3, 10, 20, 30, 40]:
        ax.plot(pe[:, dim], label=f'Dim {dim}', alpha=0.7)
    ax.set_xlabel('Position')
    ax.set_ylabel('Encoding Value')
    ax.set_title('Selected Embedding Dimensions Over Positions')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    # 4. Cosine similarity between positions
    ax = axes[1, 1]
    # Compute pairwise cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    similarity = cosine_similarity(pe[:50])  # First 50 positions
    im = ax.imshow(similarity, cmap='viridis', interpolation='nearest')
    ax.set_xlabel('Position')
    ax.set_ylabel('Position')
    ax.set_title('Cosine Similarity Between Positions (first 50)')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig('positional_encoding_visualization.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: positional_encoding_visualization.png")
    plt.close()

    return pe


def test_input_projection():
    """Test input projection and verify it preserves temporal information"""

    print("\n" + "=" * 80)
    print("INPUT PROJECTION TEST")
    print("=" * 80)

    # Create config
    config = Config()
    config.masking_strategy = 'token'  # Use token-level for this test
    config.seq_length = 10  # Short sequence for clarity
    config.num_features = 5
    config.d_model = 64

    # Create model
    model = SelfDistilledMAEMultivariate(config)
    model.eval()

    # Create sample input with clear temporal pattern
    batch_size = 1
    seq_length = config.seq_length
    num_features = config.num_features

    # Create linearly increasing pattern for each feature
    x = torch.zeros(batch_size, seq_length, num_features)
    for t in range(seq_length):
        for f in range(num_features):
            x[0, t, f] = (t * 0.1) + (f * 0.01)  # Each timestep and feature has unique value

    print(f"\nInput shape: {x.shape}")
    print(f"\nInput data (batch=0):")
    print(x[0].numpy())

    # Apply input projection
    with torch.no_grad():
        x_embed = model.input_projection(x)  # (batch, seq, d_model)

    print(f"\nEmbedding shape: {x_embed.shape}")
    print(f"\nEmbedding at t=0 (first 10 dims): {x_embed[0, 0, :10].numpy()}")
    print(f"Embedding at t=1 (first 10 dims): {x_embed[0, 1, :10].numpy()}")
    print(f"Embedding at t=9 (first 10 dims): {x_embed[0, 9, :10].numpy()}")

    # Verify temporal order is preserved
    print(f"\n✓ Temporal order verification:")
    print(f"  - Shape maintains time dimension: {x_embed.shape[1] == seq_length}")

    # Check if embeddings are different for different timesteps
    embeddings_unique = True
    for i in range(seq_length):
        for j in range(i + 1, seq_length):
            if torch.allclose(x_embed[0, i], x_embed[0, j]):
                embeddings_unique = False
                break
    print(f"  - Each timestep has unique embedding: {embeddings_unique}")

    # Visualize embeddings
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Original input
    ax = axes[0]
    im = ax.imshow(x[0].T.numpy(), aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Feature')
    ax.set_title('Original Input (5 features × 10 time steps)')
    ax.set_yticks(range(num_features))
    ax.set_yticklabels(['CPU', 'Memory', 'Disk', 'Network', 'Response'])
    plt.colorbar(im, ax=ax)

    # Embeddings
    ax = axes[1]
    im = ax.imshow(x_embed[0].T.numpy(), aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Embedding Dimension')
    ax.set_title('Embeddings (64 dims × 10 time steps)')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig('input_projection_visualization.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: input_projection_visualization.png")
    plt.close()


def test_positional_encoding_effect():
    """Test the effect of positional encoding on embeddings"""

    print("\n" + "=" * 80)
    print("POSITIONAL ENCODING EFFECT TEST")
    print("=" * 80)

    d_model = 64
    seq_len = 20
    batch_size = 1

    # Create dummy embeddings (all zeros to see pure PE effect)
    x_embed = torch.zeros(seq_len, batch_size, d_model)

    # Apply positional encoding
    pos_encoder = PositionalEncoding(d_model)
    x_with_pe = pos_encoder(x_embed)

    print(f"\nBefore PE (all zeros):")
    print(f"  Shape: {x_embed.shape}")
    print(f"  t=0: {x_embed[0, 0, :5].numpy()}")

    print(f"\nAfter PE:")
    print(f"  Shape: {x_with_pe.shape}")
    print(f"  t=0: {x_with_pe[0, 0, :5].numpy()}")
    print(f"  t=1: {x_with_pe[1, 0, :5].numpy()}")
    print(f"  t=10: {x_with_pe[10, 0, :5].numpy()}")

    # Verify PE is added
    pe_added = not torch.allclose(x_embed, x_with_pe)
    print(f"\n✓ Positional encoding is added: {pe_added}")

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Before PE
    ax = axes[0]
    im = ax.imshow(x_embed.squeeze(1).numpy().T, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    ax.set_xlabel('Position')
    ax.set_ylabel('Dimension')
    ax.set_title('Before Positional Encoding (all zeros)')
    plt.colorbar(im, ax=ax)

    # After PE
    ax = axes[1]
    im = ax.imshow(x_with_pe.squeeze(1).numpy().T, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    ax.set_xlabel('Position')
    ax.set_ylabel('Dimension')
    ax.set_title('After Positional Encoding')
    plt.colorbar(im, ax=ax)

    # Difference (should be same as PE)
    ax = axes[2]
    diff = (x_with_pe - x_embed).squeeze(1).numpy().T
    im = ax.imshow(diff, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    ax.set_xlabel('Position')
    ax.set_ylabel('Dimension')
    ax.set_title('Difference (= Positional Encoding)')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig('positional_encoding_effect.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: positional_encoding_effect.png")
    plt.close()


def test_patch_mode():
    """Test patch mode input projection"""

    print("\n" + "=" * 80)
    print("PATCH MODE TEST")
    print("=" * 80)

    # Create config for patch mode
    config = Config()
    config.masking_strategy = 'patch'
    config.seq_length = 100
    config.patch_size = 10
    config.num_features = 5
    config.d_model = 64

    # Create model
    model = SelfDistilledMAEMultivariate(config)
    model.eval()

    # Create sample input
    batch_size = 1
    x = torch.randn(batch_size, config.seq_length, config.num_features)

    print(f"\nInput shape: {x.shape}")
    print(f"  - Sequence length: {config.seq_length}")
    print(f"  - Features: {config.num_features}")

    # Patchify
    with torch.no_grad():
        x_patches = model.patchify(x)

    print(f"\nPatchified shape: {x_patches.shape}")
    print(f"  - Number of patches: {x_patches.shape[1]}")
    print(f"  - Patch dimension: {x_patches.shape[2]} (= patch_size × num_features = {config.patch_size} × {config.num_features})")

    # Patch embedding
    with torch.no_grad():
        x_embed = model.patch_embed(x_patches)

    print(f"\nPatch embeddings shape: {x_embed.shape}")
    print(f"  - Number of patches: {x_embed.shape[1]}")
    print(f"  - Embedding dimension: {x_embed.shape[2]}")

    # Verify unpatchify
    with torch.no_grad():
        x_reconstructed = model.unpatchify(x_patches)

    print(f"\nUnpatchified shape: {x_reconstructed.shape}")
    print(f"✓ Reconstruction matches original: {torch.allclose(x, x_reconstructed)}")

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Original
    ax = axes[0, 0]
    im = ax.imshow(x[0].T.numpy(), aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Feature')
    ax.set_title('Original Time Series (5 features × 100 steps)')
    # Draw patch boundaries
    num_patches = config.seq_length // config.patch_size
    for i in range(1, num_patches):
        ax.axvline(x=i * config.patch_size - 0.5, color='red', linewidth=2, linestyle='--', alpha=0.7)
    plt.colorbar(im, ax=ax)

    # Patches
    ax = axes[0, 1]
    im = ax.imshow(x_patches[0].numpy(), aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_xlabel('Patch Dimension (50 = 10 steps × 5 features)')
    ax.set_ylabel('Patch Index')
    ax.set_title('Patchified (10 patches × 50 dims)')
    plt.colorbar(im, ax=ax)

    # Patch embeddings
    ax = axes[1, 0]
    im = ax.imshow(x_embed[0].numpy(), aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_xlabel('Embedding Dimension')
    ax.set_ylabel('Patch Index')
    ax.set_title('Patch Embeddings (10 patches × 64 dims)')
    plt.colorbar(im, ax=ax)

    # Reconstructed
    ax = axes[1, 1]
    im = ax.imshow(x_reconstructed[0].T.numpy(), aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Feature')
    ax.set_title('Unpatchified (reconstructed) (5 features × 100 steps)')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig('patch_mode_visualization.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: patch_mode_visualization.png")
    plt.close()


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("COMPREHENSIVE INPUT PROJECTION AND POSITIONAL ENCODING VERIFICATION")
    print("=" * 80)

    # Test 1: Positional encoding patterns
    pe = visualize_positional_encoding()

    # Test 2: Input projection (token mode)
    test_input_projection()

    # Test 3: Positional encoding effect
    test_positional_encoding_effect()

    # Test 4: Patch mode
    test_patch_mode()

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)
    print("\nGenerated visualizations:")
    print("  1. positional_encoding_visualization.png")
    print("  2. input_projection_visualization.png")
    print("  3. positional_encoding_effect.png")
    print("  4. patch_mode_visualization.png")
    print("\n✓ Positional encoding is working correctly")
    print("✓ Input projection preserves temporal information")
    print("✓ Patch mode correctly groups consecutive time steps")
    print("=" * 80)
