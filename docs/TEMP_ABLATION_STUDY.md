# TEMP Ablation Study: Architecture and Loss Parameter Experiments

This document describes the ablation study conducted to analyze the impact of various architecture and loss parameters on the Self-Distilled MAE model for anomaly detection.

## Overview

- **Total Experiments**: 42 (21 configurations x 2 mask_after_encoder settings)
- **Epochs per Experiment**: 10
- **Normal Data Complexity**: Disabled (simple patterns)
- **Dataset**: Full dataset with train_ratio=0.5

## Base Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| force_mask_anomaly | True | Force mask patches containing anomalies |
| margin_type | dynamic | Dynamic margin based on batch statistics |
| masking_ratio | 0.5 | 50% of patches masked |
| masking_strategy | patch | Mask entire patches |
| seq_length | 100 | Default window size |
| num_patches | 10 | 10 patches per window (patch_size=10) |
| patch_level_loss | True | Patch-level discrepancy loss |
| patchify_mode | patch_cnn | CNN-based patch embedding |
| shared_mask_token | False | Separate mask tokens for teacher/student |
| d_model | 32 | Embedding dimension |
| num_encoder_layers | 2 | Transformer encoder layers |
| num_teacher_decoder_layers | 4 | Teacher decoder layers |
| num_student_decoder_layers | 1 | Student decoder layers |
| dim_feedforward | 128 | FFN dimension (4 x d_model) |

## Experiment Configurations

### 1. Default and Shared Decoder

| # | Name | Changes from Base |
|---|------|-------------------|
| 1 | default | Base configuration |
| 2 | shared_decoder | shared=1, teacher=3, student=1 (shared trained with teacher) |

### 2. Window Size Experiments

| # | Name | Changes from Base |
|---|------|-------------------|
| 3 | window_200 | seq_length=200, num_patches=20, mask_last_n=20 |
| 4 | window_500 | seq_length=500, num_patches=50, mask_last_n=50 |

### 3. Encoder Depth Experiments

| # | Name | Changes from Base |
|---|------|-------------------|
| 5 | encoder_1 | num_encoder_layers=1 |
| 6 | encoder_3 | num_encoder_layers=3 |

### 4. Decoder Configuration Experiments

| # | Name | Changes from Base |
|---|------|-------------------|
| 7 | decoder_t4s2 | teacher=4, student=2 |
| 8 | decoder_t3s2 | teacher=3, student=2 |
| 9 | decoder_t3s1 | teacher=3, student=1 |
| 10 | decoder_t2s1 | teacher=2, student=1 |

### 5. Model Size Experiments

| # | Name | Changes from Base |
|---|------|-------------------|
| 11 | d_model_16 | d_model=16, ffn=64, cnn=(8,16) |
| 12 | d_model_8 | d_model=8, nhead=2, ffn=32, cnn=(4,8) |
| 13 | d_model_64 | d_model=64, ffn=256, cnn=(32,64) |
| 14 | d_model_4 | d_model=4, nhead=2, ffn=16, cnn=(4,4) |

### 6. CNN Architecture Experiments

| # | Name | Changes from Base |
|---|------|-------------------|
| 15 | cnn_small | cnn_channels=(4,8) vs default (16,32) |
| 16 | cnn_large | cnn_channels=(64,128) vs default (16,32) |

### 7. Loss Function Experiments

| # | Name | Changes from Base |
|---|------|-------------------|
| 17 | window_level_loss | patch_level_loss=False |
| 18 | student_recon_loss | use_student_reconstruction_loss=True |
| 19 | anomaly_weight_2x | anomaly_loss_weight=2.0 |
| 20 | anomaly_weight_3x | anomaly_loss_weight=3.0 |
| 21 | anomaly_weight_5x | anomaly_loss_weight=5.0 |

## New Features Implemented

### 1. Shared Decoder (`num_shared_decoder_layers`)

A shared decoder that is trained together with the teacher decoder. The architecture becomes:

- **Teacher path**: Encoder -> Shared Decoder -> Teacher Decoder -> Output
- **Student path**: Encoder -> Student Decoder -> Output (no shared decoder)

The shared decoder uses the same mask tokens as the teacher, while the student uses its own separate mask tokens.

### 2. Student Reconstruction Loss (`use_student_reconstruction_loss`)

Adds reconstruction loss to the student decoder with opposite learning direction for anomalies:

- **Normal samples**: Student learns to reconstruct (minimize MSE)
- **Anomaly samples**: Student learns to NOT reconstruct (maximize MSE)

This encourages the student to become worse at reconstructing anomalies, potentially increasing the teacher-student discrepancy.

### 3. Anomaly Loss Weight (`anomaly_loss_weight`)

Multiplier for the anomaly portion of the discrepancy loss. Higher values (2x, 3x, 5x) increase the interference signal for anomaly samples, potentially leading to larger discrepancies.

### 4. CNN Channels (`cnn_channels`)

Configurable CNN architecture for patch embedding in `patch_cnn` mode:

- **Default**: (d_model//2, d_model) = (16, 32) for d_model=32
- **Small**: (4, 8) - reduced capacity
- **Large**: (64, 128) - increased capacity

### 5. Window Size Variations

Testing the impact of different context window sizes:

- **Window 100** (default): 10 patches of size 10
- **Window 200**: 20 patches of size 10, mask_last_n=20
- **Window 500**: 50 patches of size 10, mask_last_n=50

## Masking Strategy Comparison

Each experiment is run twice with different masking strategies:

1. **mask_after_encoder=False** (mask_before): Mask tokens go through the encoder
2. **mask_after_encoder=True** (mask_after): Standard MAE style - encode visible patches only, insert mask tokens before decoder

## Evaluation Metrics

For each experiment, we report:

| Metric | Description |
|--------|-------------|
| roc_auc | Overall ROC-AUC score |
| f1_score | F1 score at optimal threshold |
| precision | Precision at optimal threshold |
| recall | Recall at optimal threshold |
| teacher_recon_loss | Teacher reconstruction MSE on test set |
| student_recon_loss | Student reconstruction MSE on test set |
| disturbing_roc_auc | ROC-AUC for disturbing normal vs pure normal |

## Running the Experiments

```bash
# Activate environment
conda activate dc_vis

# Run all 42 experiments
python scripts/run_temp_experiments.py --output-dir results/experiments/temp
```

## Output Structure

```
results/experiments/temp/
├── summary_results.csv          # All 42 experiments summary
├── 01_default_mask_before/
│   ├── best_model.pt
│   ├── best_config.json
│   ├── best_model_detailed.csv
│   ├── anomaly_type_metrics.json
│   ├── training_histories.json
│   ├── experiment_metadata.json
│   └── visualization/
├── 01_default_mask_after/
│   └── ...
├── 02_shared_decoder_mask_before/
│   └── ...
└── ...
```

## Analysis Guidelines

### Key Questions to Answer

1. **Window size**: Does larger context improve or hurt performance?
2. **Encoder depth**: Does more encoder layers improve performance?
3. **Decoder balance**: What's the optimal teacher-student decoder ratio?
4. **Shared decoder**: Does a shared decoder help or hurt?
5. **Model capacity**: How does d_model and CNN size affect performance?
6. **Loss granularity**: Is patch-level loss better than window-level?
7. **Student learning**: Does student reconstruction loss help?
8. **Anomaly interference**: Does stronger anomaly weight improve detection?
9. **Masking strategy**: Which masking approach (before/after encoder) works better?

### Expected Insights

- Deeper teacher with shallow student typically creates larger discrepancy
- Shared decoder may reduce teacher-student gap (hypothesis to test)
- Smaller models may generalize better with limited data
- Patch-level loss provides finer gradient signals
- Student reconstruction loss may amplify discrepancy for anomalies
- Higher anomaly weights may lead to faster separation but potential instability
- Larger windows provide more context but may dilute anomaly signals
