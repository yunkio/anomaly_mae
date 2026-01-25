# Inference Modes: Score Computation Flow

This document explains the complete inference flow for Self-Distilled MAE anomaly detection, covering both `last_patch` and `all_patches` inference modes.

## Overview

The inference process computes anomaly scores for each window by:
1. **Masking**: Hiding certain positions from the model
2. **Forward pass**: Getting teacher and student reconstructions
3. **Error computation**: Computing reconstruction error and discrepancy
4. **Score aggregation**: Combining patch-level scores into window-level scores

---

## 1. Last Patch Mode (`inference_mode='last_patch'`)

The original, faster inference method that masks only the last patch.

### Step 1: Masking
```
Window: [P0][P1][P2][P3][P4][P5][P6][P7][P8][P9]
Mask:   [ 1][ 1][ 1][ 1][ 1][ 1][ 1][ 1][ 1][ 0]
                                              ↑
                                         Last patch masked
```

- **Visible positions** (mask=1): Patches P0-P8 (90 timesteps)
- **Masked positions** (mask=0): Patch P9 (10 timesteps)

### Step 2: Forward Pass (Single)
```python
mask = torch.ones(batch_size, seq_length)
mask[:, -patch_size:] = 0  # Mask last 10 positions

teacher_output, student_output, _ = model(sequences, masking_ratio=0.0, mask=mask)
```

The model:
1. Encodes visible positions (mask=1)
2. Predicts masked positions using learned mask tokens
3. Returns teacher and student reconstructions

### Step 3: Error Computation
```python
# Per-position reconstruction error (averaged across features)
recon_error = ((teacher_output - sequences) ** 2).mean(dim=2)  # (batch, 100)

# Per-position discrepancy (teacher vs student)
discrepancy = ((teacher_output - student_output) ** 2).mean(dim=2)  # (batch, 100)
```

### Step 4: Window-Level Aggregation
```python
masked_positions = (mask == 0)  # Only last 10 positions

# Average error only on masked positions
recon_score = (recon_error * masked_positions).sum(dim=1) / masked_positions.sum(dim=1)
disc_score = (discrepancy * masked_positions).sum(dim=1) / masked_positions.sum(dim=1)
```

**Result**: One score per window, computed from the last patch only.

### Diagram
```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Window (100 timesteps)                         │
├──────────────────────────────────────────────────────────┬──────────────┤
│           Visible Patches (P0-P8, 90 timesteps)          │  Masked P9   │
│                         ↓                                │      ↓       │
│                    Encoder                               │  [MASK] tok  │
│                         ↓                                │      ↓       │
│              ┌─────────────────────────────────────────────────────┐    │
│              │                    Decoder                          │    │
│              │         ┌───────────┬───────────┐                   │    │
│              │         │  Teacher  │  Student  │                   │    │
│              │         └─────┬─────┴─────┬─────┘                   │    │
│              └───────────────┼───────────┼─────────────────────────┘    │
│                              ↓           ↓                               │
│                        T_output     S_output                            │
│                              │           │                               │
│                              └─────┬─────┘                               │
│                                    ↓                                     │
│             ┌───────────────────────────────────────────────────────┐   │
│             │    Compute Errors (ONLY on masked P9 positions)       │   │
│             │    recon = MSE(T_output[90:100], input[90:100])       │   │
│             │    disc  = MSE(T_output[90:100], S_output[90:100])    │   │
│             └───────────────────────────────────────────────────────┘   │
│                                    ↓                                     │
│                         Window Score = recon + λ*disc                   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. All Patches Mode (`inference_mode='all_patches'`)

A more robust inference method that evaluates each patch's reconstruction ability.

### Step 1: Iterative Masking (N forward passes)
For each patch position, create a separate mask:

```
Pass 0: Mask P0
Window: [P0][P1][P2][P3][P4][P5][P6][P7][P8][P9]
Mask:   [ 0][ 1][ 1][ 1][ 1][ 1][ 1][ 1][ 1][ 1]
         ↑
    Compute errors for positions 0-9

Pass 1: Mask P1
Window: [P0][P1][P2][P3][P4][P5][P6][P7][P8][P9]
Mask:   [ 1][ 0][ 1][ 1][ 1][ 1][ 1][ 1][ 1][ 1]
              ↑
    Compute errors for positions 10-19

...

Pass 9: Mask P9
Window: [P0][P1][P2][P3][P4][P5][P6][P7][P8][P9]
Mask:   [ 1][ 1][ 1][ 1][ 1][ 1][ 1][ 1][ 1][ 0]
                                              ↑
    Compute errors for positions 90-99
```

### Step 2: Forward Pass Loop
```python
recon_accum = torch.zeros(batch_size, seq_length)
disc_accum = torch.zeros(batch_size, seq_length)
count_accum = torch.zeros(batch_size, seq_length)

for patch_idx in range(num_patches):  # 10 iterations
    start_pos = patch_idx * patch_size
    end_pos = start_pos + patch_size

    # Create mask for this patch
    mask = torch.ones(batch_size, seq_length)
    mask[:, start_pos:end_pos] = 0

    teacher_output, student_output, _ = model(sequences, masking_ratio=0.0, mask=mask)

    # Compute errors for masked positions only
    recon_error = ((teacher_output - sequences) ** 2).mean(dim=2)
    discrepancy = ((teacher_output - student_output) ** 2).mean(dim=2)

    # Accumulate only for masked positions
    masked_positions = (mask == 0).float()
    recon_accum += recon_error * masked_positions
    disc_accum += discrepancy * masked_positions
    count_accum += masked_positions
```

### Step 3: Per-Timestep Score
```python
# Each timestep was masked exactly once, so count_accum = 1 everywhere
recon_per_step = recon_accum / (count_accum + 1e-8)  # (batch, 100)
disc_per_step = disc_accum / (count_accum + 1e-8)    # (batch, 100)
```

### Step 4: Per-Patch Score Storage (NOT window aggregation)
```python
# Store per-patch scores: (batch, num_patches)
patch_recon_scores[patch_idx] = recon_error[:, start:end].mean(dim=1)
patch_disc_scores[patch_idx] = discrepancy[:, start:end].mean(dim=1)
```

**Key Difference**: Unlike last_patch mode, we do NOT aggregate to window-level score.
Instead, per-patch scores are passed directly to point-level aggregation.

### Step 5: Direct Point-Level Aggregation
```python
# Each patch's score is assigned to timesteps that patch covers
for window_idx, start_idx in enumerate(window_start_indices):
    for patch_idx in range(num_patches):
        patch_start = start_idx + patch_idx * patch_size
        patch_end = patch_start + patch_size
        score = patch_scores[window_idx, patch_idx]

        # Assign to all timesteps in this patch
        for t in range(patch_start, patch_end):
            point_score_lists[t].append(score)
```

**Result**: Each timestep receives scores from ALL windows that contain it (not just those where it's in the last patch).

### Diagram
```
┌─────────────────────────────────────────────────────────────────────────┐
│                       All-Patches Inference                             │
│                                                                         │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │  Pass 0: [MASK][  ][  ][  ][  ][  ][  ][  ][  ][  ]            │   │
│   │          → patch_score[0] for timesteps 0-9                     │   │
│   ├────────────────────────────────────────────────────────────────┤   │
│   │  Pass 1: [  ][MASK][  ][  ][  ][  ][  ][  ][  ][  ]            │   │
│   │          → patch_score[1] for timesteps 10-19                   │   │
│   ├────────────────────────────────────────────────────────────────┤   │
│   │                           ...                                   │   │
│   ├────────────────────────────────────────────────────────────────┤   │
│   │  Pass 9: [  ][  ][  ][  ][  ][  ][  ][  ][  ][MASK]            │   │
│   │          → patch_score[9] for timesteps 90-99                   │   │
│   └────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │         Per-Patch Scores: (n_windows, num_patches)             │   │
│   │   Window 0: [s0][s1][s2][s3][s4][s5][s6][s7][s8][s9]          │   │
│   │   Window 1: [s0][s1][s2][s3][s4][s5][s6][s7][s8][s9]          │   │
│   │   ...                                                          │   │
│   └────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                    │
│            Direct Point-Level Aggregation (skip window score)           │
│            Each timestep aggregates scores from all covering patches    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Window-to-Point Level Aggregation

After computing window-level scores, we aggregate them to point-level for PA%K evaluation.

### Sliding Window Overlap
With stride=1, each timestep is covered by multiple windows:
```
Time:        t0  t1  t2  t3  t4  t5  t6  t7  t8  t9  t10 ...
Window 0:    [====== last patch ======]
Window 1:        [====== last patch ======]
Window 2:            [====== last patch ======]
...
```

Each timestep t is covered by the "last patch" of multiple windows.

### Aggregation Methods

```python
for t in range(total_length):
    scores_covering_t = [window_scores[w] for w in windows_covering_timestep_t]

    if method == 'mean':
        point_score[t] = mean(scores_covering_t)
    elif method == 'median':
        point_score[t] = median(scores_covering_t)
    elif method == 'max':
        point_score[t] = max(scores_covering_t)
    elif method == 'voting':
        votes = sum(s > threshold for s in scores_covering_t)
        point_score[t] = 1 if votes > len(scores_covering_t)/2 else 0
```

### Point-Level Score Diagram
```
┌─────────────────────────────────────────────────────────────────────────┐
│   Window Scores → Point Scores (stride=1, patch_size=10)                │
│                                                                         │
│   Window 0: score=0.8  covers timesteps [90-99]                         │
│   Window 1: score=0.7  covers timesteps [91-100]                        │
│   Window 2: score=0.9  covers timesteps [92-101]                        │
│   ...                                                                   │
│                                                                         │
│   Timestep 95 is covered by windows 0,1,2,3,4,5 (6 windows)            │
│                                                                         │
│   point_score[95] = aggregate([0.8, 0.7, 0.9, ...])                    │
│                   = 0.82  (if method='mean')                            │
│                   = 0.80  (if method='median')                          │
│                   = 0.90  (if method='max')                             │
│                   = 1     (if method='voting' and >50% above threshold) │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Scoring Modes

After computing raw `recon` and `disc` scores, final anomaly score is computed:

| Mode | Formula | Description |
|------|---------|-------------|
| `default` | `recon + λ * disc` | Fixed lambda weighting |
| `normalized` | `z(recon) + z(disc)` | Z-score normalization |
| `adaptive` | `recon + (μ_recon/μ_disc) * disc` | Auto-scaled lambda |
| `ratio_weighted` | `recon * (1 + disc/median_disc)` | Ratio-based |

---

## 5. Comparison Summary

| Aspect | Last Patch | All Patches |
|--------|-----------|-------------|
| Forward passes | 1 | N (=num_patches) |
| Masked positions | Last patch only | Each patch once |
| Score source | Last 10 timesteps | All 100 timesteps |
| Point aggregation | Window score → last patch timesteps | Patch scores → each patch's timesteps |
| Coverage per timestep | ~patch_size windows | ~seq_length windows (10x more) |
| Speed | Fast | 10x slower |
| Robustness | May miss early anomalies | Evaluates entire window |
| Use case | Real-time, streaming | Batch evaluation |

### Sample-Level Metrics

The granularity of sample-level metrics (ROC-AUC, F1, precision, recall) differs between modes:

| Mode | Sample Unit | Label Definition | Number of Samples |
|------|------------|------------------|-------------------|
| Last Patch | Window | 1 if last patch contains anomaly | n_windows |
| All Patches | Patch | 1 if that patch contains anomaly | n_windows × num_patches |

**Last Patch Mode:**
- Each window is one sample
- Label is 1 if the last patch contains any anomaly timestep
- Evaluation: window_score > threshold vs window_label

**All Patches Mode:**
- Each patch is one sample (10× more samples)
- Each patch gets its own label based on whether it contains any anomaly
- Evaluation: patch_score > threshold vs patch_label
- More granular evaluation: distinguishes patches within same window

### Key Architectural Difference

```
Last Patch Mode:
  Window → Window Score → Point Aggregation (last patch only)
  Coverage: timestep t gets scores from ~10 windows
  Sample-level: 1 window = 1 sample

All Patches Mode:
  Window → Per-Patch Scores → Direct Point Aggregation (all patches)
  Coverage: timestep t gets scores from ~100 windows (all containing windows)
  Sample-level: 1 patch = 1 sample (n_windows × num_patches total)
```

### When to use which?

- **Last Patch**: When anomalies are expected in recent timesteps (streaming prediction)
- **All Patches**: When anomalies can appear anywhere in the window (comprehensive evaluation with higher coverage)
