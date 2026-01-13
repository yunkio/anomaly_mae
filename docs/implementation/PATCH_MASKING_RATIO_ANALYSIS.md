# Patch Masking Ratio Analysis

## Issue Discovered

When using patch-based masking, the actual masking ratio can deviate from the intended ratio when the number of patches multiplied by the masking ratio doesn't result in a whole number.

## Test Results

### Test Case 1: masking_ratio=0.6, patch_size=10, seq_length=100
- **Expected**: 6 patches masked (60.0%)
- **Actual**: 6 patches masked (60.0%)
- **Status**: ✓ PASS

### Test Case 2: masking_ratio=0.75, patch_size=10, seq_length=100
- **Expected**: 7.5 patches masked → rounds to 8
- **Actual**: 8 patches masked (80.0%)
- **Deviation**: +5% from intended 75%
- **Status**: ✗ FAIL (acceptable limitation)

### Test Case 3: masking_ratio=0.5, patch_size=20, seq_length=100
- **Expected**: 2.5 patches masked → rounds to 2 (banker's rounding)
- **Actual**: 3 patches masked (60.0%)
- **Deviation**: +10% from intended 50%
- **Status**: ✗ FAIL (acceptable limitation)

## Root Cause

The calculation is:
```python
num_keep = round(seq_len * (1 - masking_ratio))
num_masked = seq_len - num_keep
```

For 10 patches with 0.75 masking ratio:
- num_keep = round(10 * 0.25) = round(2.5) = 2 (banker's rounding)
- num_masked = 10 - 2 = 8 patches (80% instead of 75%)

For 5 patches with 0.5 masking ratio:
- num_keep = round(5 * 0.5) = round(2.5) = 2 (banker's rounding)
- num_masked = 5 - 2 = 3 patches (60% instead of 50%)

## Comparison with Vision MAE

Vision Transformer MAE (He et al., 2021) uses a similar approach. With 196 patches and 75% masking:
- num_keep = 196 * 0.25 = 49 patches exactly
- num_masked = 147 patches exactly

The issue only occurs when the number of patches is small and the ratio doesn't divide evenly.

## Solutions Considered

### Option 1: Use int() with floor (current in original code)
```python
num_keep = int(seq_len * (1 - masking_ratio))
```
- Always rounds down
- Test 2: int(2.5) = 2 → 8 masked (80%)
- Test 3: int(2.5) = 2 → 3 masked (60%)

### Option 2: Use round() (current implementation)
```python
num_keep = round(seq_len * (1 - masking_ratio))
```
- Uses banker's rounding (round half to even)
- Test 2: round(2.5) = 2 → 8 masked (80%)
- Test 3: round(2.5) = 2 → 3 masked (60%)

### Option 3: Use ceil() for masking
```python
import math
num_masked = math.ceil(seq_len * masking_ratio)
num_keep = seq_len - num_masked
```
- Always rounds up the number of masked patches
- Test 2: ceil(7.5) = 8 → 8 masked (80%)
- Test 3: ceil(2.5) = 3 → 3 masked (60%)

### Option 4: Accept the limitation
- Document that masking ratio is approximate when num_patches is small
- In practice, with seq_length=100 and patch_size=10, we have 10 patches
- Most common ratios (0.5, 0.6, 0.75) will have slight deviations
- This matches Vision MAE behavior (they just happen to use 196 patches which divides better)

## Recommendation

**Accept the limitation and document it.**

Reasoning:
1. **This is inherent to patch-based masking** - you can't mask fractional patches
2. **Vision MAE has the same limitation** - they just use more patches (196) which divides better
3. **Impact is minimal** - deviations are small (5-10%) and only occur with certain ratios
4. **Alternative ratios work perfectly**:
   - 0.6 with 10 patches → exactly 6 masked
   - 0.7 with 10 patches → exactly 7 masked
   - 0.8 with 10 patches → exactly 8 masked

## Verification Results

✓ **Patch coherence**: All patches are either fully masked or fully unmasked (no partial masking)
✓ **Masking works correctly**: The mechanism properly masks contiguous blocks
⚠ **Ratio deviation**: Small deviations occur when num_patches * ratio is not a whole number

## Conclusion

The patch masking implementation is **correct**. The ratio deviations are an inherent limitation of patch-based masking with small numbers of patches, and match the behavior of Vision MAE.

For our experiments with:
- seq_length = 100
- patch_size = 10
- num_patches = 10
- masking_ratio = 0.6

We get exactly 60% masking (6 patches masked), which is perfect.

If higher precision is needed, we could:
1. Use larger sequences to get more patches
2. Adjust patch_size to make num_patches divisible by the desired ratio
3. Use token-level masking instead (which is why we're adding it to ablation studies)
