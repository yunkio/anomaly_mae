# Masking Implementation Analysis

## Current Implementation: Token-Level Random Masking

### Architecture Overview

```
Input: (batch_size, seq_length=100, num_features=5)
  ↓
Input Projection: Linear(num_features=5 → d_model=64)
  ↓
Tokens: (batch_size, seq_length=100, d_model=64)
  ↓
Random Token-Level Masking (60% masked)
  ↓
Transformer Encoder/Decoder
```

### Current Masking Strategy

**File**: `multivariate_mae_experiments.py:388-473`

#### Temporal Masking (Default)
- Each **time step** is treated as a **token**
- Tokens are **randomly** masked (not contiguous)
- Mask shape: `(seq_len=100, batch_size)`
- All features at the same time step are masked together

#### Feature-wise Masking
- Each time step independently masked per feature
- Mask shape: `(seq_len=100, batch_size, d_model)`
- More flexible but still token-level

### Key Characteristics

✅ **What it does:**
- Token = individual time step
- 100 time steps → 100 tokens
- Random selection of which tokens to mask
- No contiguous blocks (patches)

❌ **What it does NOT do:**
- No patch tokenization
- No grouping of consecutive time steps
- No reduction of sequence length

## Comparison: Token-Level vs Patch-Based

### Current (Token-Level)
```
Time: [0][1][2][3][4][5][6][7][8][9]
Mask: [X][ ][X][ ][ ][X][X][ ][X][ ]
      ↑ Each time step masked independently
```

### Patch-Based (ViT/MAE Style)
```
Time: [0 1][2 3][4 5][6 7][8 9]
Patch: [P0 ][P1 ][P2 ][P3 ][P4 ]
Mask:  [X  ][ X ][ X ][   ][   ]
       ↑ Entire patches masked together
```

## Verification Results

### Masking Pattern Analysis

**Sample Mask (first 50 tokens):**
```
[0 0 0 0 1 1 0 0 0 1 0 1 0 0 1 1 0 1 0 0 0 1 0 1 1 1 0 0 1 0 0 0 1 0 0 1 0 0 0 1 0 1 0 0 1 0 1 0 1 0]
```
- `0` = masked (60 tokens)
- `1` = visible (40 tokens)
- Pattern: **Random** (not contiguous)
- Longest consecutive masked: ~6 tokens

### Statistics
- Masking ratio: 60% (60/100 tokens)
- Granularity: **Time-step level**
- Randomness: **High** (non-contiguous)

## To Implement Patch-Based Masking

### Required Changes

1. **Patch Embedding Layer**
   ```python
   # Group consecutive time steps
   patch_size = 10
   num_patches = seq_length // patch_size  # 100 → 10
   
   # Reshape: (batch, seq_length, features) → (batch, num_patches, patch_size*features)
   x = x.reshape(batch, num_patches, patch_size * num_features)
   
   # Project to d_model
   self.patch_embedding = nn.Linear(patch_size * num_features, d_model)
   ```

2. **Patch-Level Masking**
   ```python
   # Mask entire patches
   num_masked = int(num_patches * masking_ratio)
   masked_patch_indices = random.sample(range(num_patches), num_masked)
   
   # Apply mask to whole patches
   for idx in masked_patch_indices:
       tokens[idx] = mask_token
   ```

3. **Reduced Sequence Length**
   - Current: 100 tokens
   - Patch-based (size=10): 10 tokens
   - 10x reduction in computational cost

## Implications

### Current Implementation (Token-Level)
**Pros:**
- ✅ Fine-grained masking
- ✅ More masking diversity
- ✅ Better for small sequences

**Cons:**
- ❌ No computational efficiency gain
- ❌ Not aligned with MAE paper methodology
- ❌ Harder to learn local structure

### Patch-Based Approach
**Pros:**
- ✅ Computational efficiency (fewer tokens)
- ✅ Learns local temporal structure
- ✅ Aligned with MAE/ViT methodology
- ✅ Better scalability

**Cons:**
- ❌ Coarser granularity
- ❌ Needs careful patch size selection
- ❌ May lose fine-grained temporal info

## Recommendation

For **true MAE-style** masking on time series:
1. Implement patch tokenization
2. Use patch_size = 4-10 for seq_length=100
3. Mask entire patches (not individual time steps)
4. This will align with the original MAE paper

Current implementation is **token-level random masking**, 
which is closer to **BERT-style** masking than **MAE-style** patch masking.

---
Generated: 2025-12-29
