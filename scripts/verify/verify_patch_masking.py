"""
Verification script to check patch masking ratio correctness
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from multivariate_mae_experiments import Config, SelfDistilledMAEMultivariate

def verify_patch_masking():
    """Verify that patch masking masks the intended ratio of data"""

    print("=" * 80)
    print("PATCH MASKING RATIO VERIFICATION")
    print("=" * 80)

    # Test configurations
    test_cases = [
        {"masking_ratio": 0.6, "patch_size": 10, "seq_length": 100},
        {"masking_ratio": 0.75, "patch_size": 10, "seq_length": 100},
        {"masking_ratio": 0.5, "patch_size": 20, "seq_length": 100},
    ]

    batch_size = 32
    num_features = 5
    num_trials = 100

    results = []

    for test_case in test_cases:
        print(f"\n{'=' * 80}")
        print(f"Test Case: {test_case}")
        print(f"{'=' * 80}")

        # Create config with patch masking
        config = Config()
        config.masking_ratio = test_case["masking_ratio"]
        config.patch_size = test_case["patch_size"]
        config.seq_length = test_case["seq_length"]
        config.num_features = num_features
        config.masking_strategy = 'patch'

        # Calculate expected values
        num_patches = config.seq_length // config.patch_size
        expected_masked_patches = int(num_patches * config.masking_ratio)
        expected_unmasked_patches = num_patches - expected_masked_patches
        expected_masked_timesteps = expected_masked_patches * config.patch_size
        expected_unmasked_timesteps = expected_unmasked_patches * config.patch_size

        print(f"\nExpected:")
        print(f"  Total patches: {num_patches}")
        print(f"  Masked patches: {expected_masked_patches} ({config.masking_ratio * 100:.1f}%)")
        print(f"  Unmasked patches: {expected_unmasked_patches} ({(1 - config.masking_ratio) * 100:.1f}%)")
        print(f"  Masked time steps: {expected_masked_timesteps} ({expected_masked_timesteps / config.seq_length * 100:.1f}%)")
        print(f"  Unmasked time steps: {expected_unmasked_timesteps} ({expected_unmasked_timesteps / config.seq_length * 100:.1f}%)")

        # Create model
        model = SelfDistilledMAEMultivariate(config)
        model.eval()

        # Run multiple trials
        masked_ratios = []
        patch_mask_ratios = []

        with torch.no_grad():
            for trial in range(num_trials):
                # Create random input
                x = torch.randn(batch_size, config.seq_length, num_features)

                # Forward pass
                _, _, mask = model(x, masking_ratio=config.masking_ratio)

                # mask shape: (batch, seq_length) after expansion
                # Calculate actual masked ratio
                masked_ratio = (1 - mask).mean().item()  # 0 = masked, 1 = unmasked
                masked_ratios.append(masked_ratio)

                # Check patch-level masking
                # Reshape to patches to verify entire patches are masked
                mask_patches = mask.reshape(batch_size, num_patches, config.patch_size)
                # A patch is masked if all its time steps are masked (value = 0)
                patch_is_masked = (mask_patches.sum(dim=-1) == 0).float()  # (batch, num_patches)
                patch_mask_ratio = patch_is_masked.mean().item()
                patch_mask_ratios.append(patch_mask_ratio)

        # Calculate statistics
        actual_masked_ratio = np.mean(masked_ratios)
        actual_patch_mask_ratio = np.mean(patch_mask_ratios)
        std_masked_ratio = np.std(masked_ratios)
        std_patch_mask_ratio = np.std(patch_mask_ratios)

        print(f"\nActual (averaged over {num_trials} trials):")
        print(f"  Time-step masking ratio: {actual_masked_ratio:.4f} ± {std_masked_ratio:.4f} "
              f"({actual_masked_ratio * 100:.2f}%)")
        print(f"  Patch masking ratio: {actual_patch_mask_ratio:.4f} ± {std_patch_mask_ratio:.4f} "
              f"({actual_patch_mask_ratio * 100:.2f}%)")

        # Check if it matches expected
        expected_ratio = config.masking_ratio
        ratio_diff = abs(actual_masked_ratio - expected_ratio)
        patch_ratio_diff = abs(actual_patch_mask_ratio - expected_ratio)

        print(f"\nVerification:")
        print(f"  Expected masking ratio: {expected_ratio:.4f} ({expected_ratio * 100:.1f}%)")
        print(f"  Time-step ratio error: {ratio_diff:.4f} ({ratio_diff * 100:.2f}%)")
        print(f"  Patch ratio error: {patch_ratio_diff:.4f} ({patch_ratio_diff * 100:.2f}%)")

        if ratio_diff < 0.01 and patch_ratio_diff < 0.01:
            print(f"  ✓ PASS: Masking ratio is correct!")
        else:
            print(f"  ✗ FAIL: Masking ratio deviates from expected!")

        # Verify that masking is truly patch-based (contiguous blocks)
        # Sample one sequence to visualize
        x_sample = torch.randn(1, config.seq_length, num_features)
        _, _, mask_sample = model(x_sample, masking_ratio=config.masking_ratio)
        mask_sample = mask_sample[0].numpy()  # (seq_length,)

        # Check patch boundaries
        mask_patches = mask_sample.reshape(num_patches, config.patch_size)
        patches_fully_masked = np.all(mask_patches == 0, axis=1).sum()
        patches_fully_unmasked = np.all(mask_patches == 1, axis=1).sum()
        patches_partially_masked = num_patches - patches_fully_masked - patches_fully_unmasked

        print(f"\nPatch coherence check (single sample):")
        print(f"  Fully masked patches: {patches_fully_masked}/{num_patches}")
        print(f"  Fully unmasked patches: {patches_fully_unmasked}/{num_patches}")
        print(f"  Partially masked patches: {patches_partially_masked}/{num_patches}")

        if patches_partially_masked == 0:
            print(f"  ✓ PASS: All patches are either fully masked or fully unmasked (true patch masking)")
        else:
            print(f"  ✗ FAIL: Some patches are partially masked (not true patch masking)")

        results.append({
            "config": test_case,
            "expected_ratio": expected_ratio,
            "actual_timestep_ratio": actual_masked_ratio,
            "actual_patch_ratio": actual_patch_mask_ratio,
            "patch_coherence": patches_partially_masked == 0
        })

    # Visualization
    print(f"\n{'=' * 80}")
    print("CREATING VISUALIZATIONS")
    print(f"{'=' * 80}")

    fig, axes = plt.subplots(2, len(test_cases), figsize=(5 * len(test_cases), 8))

    for idx, test_case in enumerate(test_cases):
        config = Config()
        config.masking_ratio = test_case["masking_ratio"]
        config.patch_size = test_case["patch_size"]
        config.seq_length = test_case["seq_length"]
        config.num_features = num_features
        config.masking_strategy = 'patch'

        model = SelfDistilledMAEMultivariate(config)
        model.eval()

        # Generate sample
        x_sample = torch.randn(1, config.seq_length, num_features)
        _, _, mask_sample = model(x_sample, masking_ratio=config.masking_ratio)
        mask_sample = mask_sample[0].numpy()  # (seq_length,)

        # Plot 1: Time-step level mask
        ax1 = axes[0, idx] if len(test_cases) > 1 else axes[0]
        ax1.imshow(mask_sample.reshape(1, -1), cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax1.set_title(f'Mask (ratio={test_case["masking_ratio"]}, patch={test_case["patch_size"]})')
        ax1.set_xlabel('Time Step')
        ax1.set_yticks([])
        ax1.grid(False)

        # Add patch boundaries
        num_patches = config.seq_length // config.patch_size
        for i in range(1, num_patches):
            ax1.axvline(x=i * config.patch_size - 0.5, color='blue', linewidth=2, linestyle='--', alpha=0.5)

        # Plot 2: Patch-level view
        ax2 = axes[1, idx] if len(test_cases) > 1 else axes[1]
        mask_patches = mask_sample.reshape(num_patches, config.patch_size)
        patch_status = mask_patches.mean(axis=1)  # 0 = fully masked, 1 = fully unmasked

        ax2.bar(range(num_patches), patch_status, color=['red' if p == 0 else 'green' for p in patch_status])
        ax2.set_xlabel('Patch Index')
        ax2.set_ylabel('Mask Value (0=Masked, 1=Unmasked)')
        ax2.set_title('Patch-level Masking')
        ax2.set_ylim([0, 1.1])
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('patch_masking_verification.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: patch_masking_verification.png")

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")

    all_passed = all(
        abs(r["actual_timestep_ratio"] - r["expected_ratio"]) < 0.01 and
        abs(r["actual_patch_ratio"] - r["expected_ratio"]) < 0.01 and
        r["patch_coherence"]
        for r in results
    )

    if all_passed:
        print("\n✓ ALL TESTS PASSED!")
        print("  - Masking ratios match expected values")
        print("  - Patches are fully masked or unmasked (no partial masking)")
        print("  - Patch-based masking is implemented correctly")
    else:
        print("\n✗ SOME TESTS FAILED!")
        print("  Please review the results above for details")

    return results

if __name__ == '__main__':
    verify_patch_masking()
