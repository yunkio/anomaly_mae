# Ablation Study: Architecture and Loss Parameter Experiments

This document describes the ablation study designed to analyze the impact of various architecture and loss parameters on normal vs anomaly discrepancy in the Self-Distilled MAE model.

## Primary Goal

**Maximize the anomaly/normal discrepancy ratio (disc_ratio)** and **teacher reconstruction ratio (t_ratio)** while maintaining good detection performance (ROC-AUC, F1).

## Experiment Overview

- **Total Experiments**: 70 configurations
- **Phases**: Each experiment runs with mask_after_encoder=False, then mask_after_encoder=True
- **Scoring Modes**: Each model tested with default, adaptive, and normalized scoring
- **Inference Modes**: Each model tested with last_patch and all_patches
- **Epochs per Experiment**: 30
- **Teacher Warmup Epochs**: 3
- **Normal Data Complexity**: Disabled (simple patterns)

**Total Output Combinations**: 70 configs × 2 phases × 3 scoring × 2 inference = 840 evaluations

## Base Configuration (Optimized Defaults)

| Parameter | Value | Description |
|-----------|-------|-------------|
| force_mask_anomaly | True | Force mask patches containing anomalies |
| margin_type | dynamic | Use dynamic margin based on batch statistics |
| masking_ratio | 0.2 | 20% of patches masked |
| masking_strategy | patch | Mask entire patches |
| seq_length | 100 | Window size |
| num_patches | 10 | seq_length / patch_size |
| patch_size | 10 | Timesteps per patch (fixed) |
| patch_level_loss | True | Compute loss at patch level |
| patchify_mode | patch_cnn | Use CNN for patch embedding |
| shared_mask_token | False | Separate mask tokens for teacher/student |
| d_model | 64 | Transformer embedding dimension |
| nhead | 2 | Number of attention heads |
| num_encoder_layers | 1 | Encoder layers |
| num_teacher_decoder_layers | 2 | Teacher decoder layers (t2s1) |
| num_student_decoder_layers | 1 | Student decoder layers |
| dim_feedforward | 256 | FFN hidden dimension (4x d_model) |
| cnn_channels | (32, 64) | CNN channels (d_model//2, d_model) |
| dropout | 0.1 | Dropout rate |
| learning_rate | 2e-3 | Initial learning rate |
| weight_decay | 1e-5 | L2 regularization |
| dynamic_margin_k | 1.5 | k for dynamic margin (mu + k*sigma) |

## Experiments 1-70

### GROUP 1 (01-10): Window Size & Patch Variations

| ID | Name | Description | Key Changes |
|----|------|-------------|-------------|
| 01 | default | Optimized baseline | - |
| 02 | window_200 | Larger window | seq_length=200, num_patches=20 |
| 03 | window_500 | Much larger window | seq_length=500, num_patches=50 |
| 04 | window_1000_p20 | Very large window (safe) | seq_length=1000, patch_size=20, num_patches=50 |
| 05 | window_1000_p10 | Very large window (risky) | seq_length=1000, num_patches=100 |
| 06 | patch_5 | Finer patches | patch_size=5, num_patches=20 |
| 07 | patch_20 | Coarser patches | patch_size=20, num_patches=5 |
| 08 | w500_p5 | Large window + fine patches | seq_length=500, patch_size=5, num_patches=100 |
| 09 | w500_p20 | Large window + coarse patches | seq_length=500, patch_size=20, num_patches=25 |
| 10 | window_2000_p20 | Extra large window | seq_length=2000, patch_size=20, num_patches=100 |

### GROUP 2 (11-20): Encoder/Decoder Depth Variations

| ID | Name | Description | Key Changes |
|----|------|-------------|-------------|
| 11 | encoder_2 | Deeper encoder | num_encoder_layers=2 |
| 12 | encoder_3 | Much deeper encoder | num_encoder_layers=3 |
| 13 | encoder_4 | Very deep encoder | num_encoder_layers=4 |
| 14 | encoder_0 | No encoder | num_encoder_layers=0 |
| 15 | decoder_t3s1 | Deeper teacher | num_teacher_decoder_layers=3 |
| 16 | decoder_t4s1 | Much deeper teacher | num_teacher_decoder_layers=4 |
| 17 | decoder_t2s2 | Equal depth | num_student_decoder_layers=2 |
| 18 | decoder_t3s2 | Deeper both | teacher=3, student=2 |
| 19 | decoder_t4s2 | Much deeper both | teacher=4, student=2 |
| 20 | shared_decoder | Shared + separate | shared=1, teacher=1, student=1 |

### GROUP 3 (21-30): Model Capacity

| ID | Name | Description | Key Changes |
|----|------|-------------|-------------|
| 21 | d_model_32 | Smaller model | d_model=32, FFN=128, cnn=(16,32) |
| 22 | d_model_128 | Larger model | d_model=128, nhead=4, FFN=512, cnn=(64,128) |
| 23 | d_model_256 | Very large model | d_model=256, nhead=8, FFN=1024, cnn=(128,256) |
| 24 | d_model_16 | Very small model | d_model=16, nhead=2, FFN=64, cnn=(8,16) |
| 25 | nhead_1 | Single head | nhead=1 |
| 26 | nhead_4 | More heads | nhead=4 |
| 27 | nhead_8 | Many heads | nhead=8 |
| 28 | d128_nhead_16 | Large + many heads | d_model=128, nhead=16, FFN=512 |
| 29 | ffn_128 | Smaller FFN | dim_feedforward=128 |
| 30 | ffn_512 | Larger FFN | dim_feedforward=512 |

### GROUP 4 (31-40): Masking Ratio Variations

| ID | Name | Description | Key Changes |
|----|------|-------------|-------------|
| 31 | mask_0.10 | Ultra-low masking | masking_ratio=0.1 |
| 32 | mask_0.15 | Very low masking | masking_ratio=0.15 |
| 33 | mask_0.25 | Slightly above default | masking_ratio=0.25 |
| 34 | mask_0.30 | Moderate masking | masking_ratio=0.3 |
| 35 | mask_0.40 | Standard masking | masking_ratio=0.4 |
| 36 | mask_0.50 | High masking | masking_ratio=0.5 |
| 37 | mask_0.60 | Higher masking | masking_ratio=0.6 |
| 38 | mask_0.70 | Very high masking | masking_ratio=0.7 |
| 39 | mask_0.80 | Ultra-high masking | masking_ratio=0.8 |
| 40 | feature_wise_mask | Feature-wise masking | masking_strategy='feature_wise' |

### GROUP 5 (41-50): Loss Parameters

| ID | Name | Description | Key Changes |
|----|------|-------------|-------------|
| 41 | lambda_0.1 | Very weak discrepancy | lambda_disc=0.1 |
| 42 | lambda_0.25 | Weak discrepancy | lambda_disc=0.25 |
| 43 | lambda_1.0 | Equal weight | lambda_disc=1.0 |
| 44 | lambda_2.0 | Strong discrepancy | lambda_disc=2.0 |
| 45 | lambda_3.0 | Very strong | lambda_disc=3.0 |
| 46 | k_1.0 | Very tight margin | dynamic_margin_k=1.0 |
| 47 | k_2.0 | Moderate margin | dynamic_margin_k=2.0 |
| 48 | k_2.5 | Slightly loose | dynamic_margin_k=2.5 |
| 49 | k_3.0 | Loose margin | dynamic_margin_k=3.0 |
| 50 | k_4.0 | Very loose margin | dynamic_margin_k=4.0 |

### GROUP 6 (51-60): Training Parameters

| ID | Name | Description | Key Changes |
|----|------|-------------|-------------|
| 51 | lr_5e-4 | Lower learning rate | learning_rate=5e-4 |
| 52 | lr_1e-3 | Standard LR | learning_rate=1e-3 |
| 53 | lr_3e-3 | Higher LR | learning_rate=3e-3 |
| 54 | lr_5e-3 | Very high LR | learning_rate=5e-3 |
| 55 | dropout_0.0 | No dropout | dropout=0.0 |
| 56 | dropout_0.2 | Higher dropout | dropout=0.2 |
| 57 | dropout_0.3 | Very high dropout | dropout=0.3 |
| 58 | wd_0 | No weight decay | weight_decay=0 |
| 59 | wd_1e-4 | Higher weight decay | weight_decay=1e-4 |
| 60 | anomaly_weight_2.0 | 2x anomaly weight | anomaly_loss_weight=2.0 |

### GROUP 7 (61-70): Combined Optimal Configurations

Focus on maximizing disc_ratio and t_ratio.

| ID | Name | Description | Key Changes |
|----|------|-------------|-------------|
| 61 | w500_d128 | Large window + capacity | seq_length=500, d_model=128, nhead=4 |
| 62 | w500_enc2 | Large window + deep encoder | seq_length=500, encoder=2 |
| 63 | w1000_p20_d128 | Very large window + capacity | seq_length=1000, patch_size=20, d_model=128 |
| 64 | mask0.15_d128 | Ultra-low mask + capacity | masking_ratio=0.15, d_model=128 |
| 65 | mask0.1_t3s1 | Ultra-low mask + deep teacher | masking_ratio=0.1, teacher=3 |
| 66 | nhead1_d128 | Single head + large model | nhead=1, d_model=128 |
| 67 | anom3_lambda1 | Strong anomaly focus | anomaly_weight=3.0, lambda_disc=1.0 |
| 68 | w500_mask0.15 | Large window + low mask | seq_length=500, masking_ratio=0.15 |
| 69 | w1000_p20_mask0.15 | Very large + low mask | seq_length=1000, patch_size=20, mask=0.15 |
| 70 | d128_enc2_t3s1 | Capacity + depth | d_model=128, encoder=2, teacher=3 |

## Key Metrics

- **disc_ratio**: discrepancy_anomaly / discrepancy_normal (higher is better)
- **t_ratio**: teacher_recon_anomaly / teacher_recon_normal (higher is better)
- **ROC-AUC**: Area under ROC curve
- **F1 Score**: Harmonic mean of precision and recall
- **PA%K F1**: Point-Adjust F1 at K% tolerance

## Scoring Modes

Each model is evaluated with three scoring methods:
- **default**: recon + lambda_disc * disc
- **adaptive**: recon + (mean_recon/mean_disc) * disc
- **normalized**: z-score normalized (recon_z + disc_z)

## Inference Modes

Each model is evaluated with two inference methods:

| Mode | Masking | Sample Unit | Coverage |
|------|---------|-------------|----------|
| `last_patch` | Last patch only | Window | ~10 windows per timestep |
| `all_patches` | Each patch (N passes) | Patch | ~100 windows per timestep |

**last_patch**: Fast (1 forward pass), window-level evaluation
**all_patches**: Thorough (N forward passes), patch-level evaluation with 10× coverage

See [INFERENCE_MODES.md](INFERENCE_MODES.md) for detailed documentation.

---

**Status**: ✅ Experiment configurations defined
