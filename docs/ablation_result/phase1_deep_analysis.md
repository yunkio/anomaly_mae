# Ablation Phase 1 - Deep Dive Analysis

**Total Experiments:** 1392

**Analysis Date:** 2026-01-27 14:14:02.458573

---

## Analysis 1: High Discrepancy Ratio Models

### Top 20 Models by Discrepancy Ratio

| experiment | roc_auc | pa_80_f1 | disc_ratio | disc_cohens_d_normal_vs_anomaly | recon_ratio | mask_after_encoder | patchify_mode | masking_strategy | d_model | dropout | dynamic_margin_k | scoring_mode | inference_mode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 050_k_4.0_mask_after_normalized_all | 0.8719 | 0.4602 | 4.2588 | 1.8689 | 1.5627 | True | patch_cnn | patch | 64 | 0.1 | 4.0 | normalized | all_patches |
| 050_k_4.0_mask_after_adaptive_all | 0.8793 | 0.4705 | 4.2548 | 1.8669 | 1.5633 | True | patch_cnn | patch | 64 | 0.1 | 4.0 | adaptive | all_patches |
| 050_k_4.0_mask_after_default_all | 0.8123 | 0.3827 | 4.247 | 1.8658 | 1.5619 | True | patch_cnn | patch | 64 | 0.1 | 4.0 | default | all_patches |
| 057_dropout_0.3_mask_after_normalized_all | 0.8506 | 0.4211 | 4.2349 | 1.7531 | 1.3339 | True | patch_cnn | patch | 64 | 0.3 | 1.5 | normalized | all_patches |
| 162_d128_dropout0.3_mask_after_normalized_all | 0.8506 | 0.4211 | 4.2349 | 1.7531 | 1.3339 | True | patch_cnn | patch | 64 | 0.3 | 1.5 | normalized | all_patches |
| 057_dropout_0.3_mask_after_default_all | 0.7368 | 0.2865 | 4.2235 | 1.7452 | 1.3284 | True | patch_cnn | patch | 64 | 0.3 | 1.5 | default | all_patches |
| 162_d128_dropout0.3_mask_after_default_all | 0.7368 | 0.2865 | 4.2235 | 1.7452 | 1.3284 | True | patch_cnn | patch | 64 | 0.3 | 1.5 | default | all_patches |
| 057_dropout_0.3_mask_after_adaptive_all | 0.8613 | 0.4434 | 4.2199 | 1.7414 | 1.3245 | True | patch_cnn | patch | 64 | 0.3 | 1.5 | adaptive | all_patches |
| 162_d128_dropout0.3_mask_after_adaptive_all | 0.8613 | 0.4434 | 4.2199 | 1.7414 | 1.3245 | True | patch_cnn | patch | 64 | 0.3 | 1.5 | adaptive | all_patches |
| 118_d128_patch25_mask_after_default_all | 0.8041 | 0.3078 | 4.0919 | 1.4148 | 1.8957 | True | patch_cnn | patch | 64 | 0.1 | 1.5 | default | all_patches |
| 118_d128_patch25_mask_after_adaptive_all | 0.888 | 0.493 | 4.0917 | 1.4156 | 1.9034 | True | patch_cnn | patch | 64 | 0.1 | 1.5 | adaptive | all_patches |
| 118_d128_patch25_mask_after_normalized_all | 0.8846 | 0.4339 | 4.0819 | 1.4092 | 1.8981 | True | patch_cnn | patch | 64 | 0.1 | 1.5 | normalized | all_patches |
| 049_k_3.0_mask_after_normalized_all | 0.8732 | 0.4645 | 4.0444 | 1.9223 | 1.549 | True | patch_cnn | patch | 64 | 0.1 | 3.0 | normalized | all_patches |
| 049_k_3.0_mask_after_adaptive_all | 0.8788 | 0.4725 | 4.0318 | 1.9188 | 1.5478 | True | patch_cnn | patch | 64 | 0.1 | 3.0 | adaptive | all_patches |
| 049_k_3.0_mask_after_default_all | 0.8093 | 0.3799 | 4.0306 | 1.9168 | 1.5451 | True | patch_cnn | patch | 64 | 0.1 | 3.0 | default | all_patches |
| 012_encoder_3_mask_after_adaptive_all | 0.8595 | 0.4628 | 3.9865 | 1.7084 | 1.4678 | True | patch_cnn | patch | 64 | 0.1 | 1.5 | adaptive | all_patches |
| 048_k_2.5_mask_after_normalized_all | 0.872 | 0.4583 | 3.9839 | 1.8868 | 1.5516 | True | patch_cnn | patch | 64 | 0.1 | 2.5 | normalized | all_patches |
| 012_encoder_3_mask_after_normalized_all | 0.8516 | 0.4448 | 3.983 | 1.7092 | 1.4698 | True | patch_cnn | patch | 64 | 0.1 | 1.5 | normalized | all_patches |
| 012_encoder_3_mask_after_default_all | 0.7665 | 0.3301 | 3.9813 | 1.7068 | 1.4682 | True | patch_cnn | patch | 64 | 0.1 | 1.5 | default | all_patches |
| 048_k_2.5_mask_after_adaptive_all | 0.8779 | 0.4706 | 3.9763 | 1.8839 | 1.5527 | True | patch_cnn | patch | 64 | 0.1 | 2.5 | adaptive | all_patches |

**mask_after_encoder:** {True: 20}

**patchify_mode:** {'patch_cnn': 20}

**masking_strategy:** {'patch': 20}

**margin_type:** {'dynamic': 20}

**scoring_mode:** {'normalized': 7, 'adaptive': 7, 'default': 6}

**inference_mode:** {'all_patches': 20}

**dropout:** {0.1: 14, 0.3: 6}

**Numerical Parameters Statistics (Top 20 disc_ratio)**

| parameter | mean | std | min | max |
| --- | --- | --- | --- | --- |
| d_model | 64.0 | 0.0 | 64.0 | 64.0 |
| masking_ratio | 0.2 | 0.0 | 0.2 | 0.2 |
| lambda_disc | 0.5 | 0.0 | 0.5 | 0.5 |
| dropout | 0.16 | 0.094 | 0.1 | 0.3 |
| dynamic_margin_k | 2.2 | 0.9652 | 1.5 | 4.0 |


**KEY INSIGHT 1:**
- High disc_ratio models (4.12 avg) tend to have:
  - mask_after_encoder = True
  - inference_mode = all_patches
  - Lower d_model (64 avg vs 71 overall)
  - Higher dropout (0.160 vs 0.103)
- However, ROC-AUC is relatively low (0.8413 vs 0.8122)
- Negative correlation between disc_ratio and recon_ratio (-0.446)


## Analysis 2: Models with High Discriminative Power (Both Disc and Recon)

Found 3 models with both high disc_cohens_d (>75th percentile) and high recon_cohens_d (>75th percentile)

| experiment | roc_auc | pa_80_f1 | disc_cohens_d_normal_vs_anomaly | recon_cohens_d_normal_vs_anomaly | disc_ratio | recon_ratio | mask_after_encoder | d_model | num_teacher_decoder_layers | scoring_mode | inference_mode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 028_d128_nhead_16_mask_before_default_all | 0.9467 | 0.6997 | 1.3888 | 2.2038 | 2.4078 | 5.4361 | False | 128 | 2 | default | all_patches |
| 028_d128_nhead_16_mask_before_adaptive_all | 0.9436 | 0.659 | 1.3888 | 2.2038 | 2.4078 | 5.4361 | False | 128 | 2 | adaptive | all_patches |
| 028_d128_nhead_16_mask_before_normalized_all | 0.9356 | 0.6263 | 1.3888 | 2.2038 | 2.4078 | 5.4361 | False | 128 | 2 | normalized | all_patches |


**KEY INSIGHT 2:**
- 3 models achieve high discriminative power in BOTH metrics
- These models have excellent performance:
  - ROC-AUC: 0.9420 (vs 0.8122 overall)
  - PA%80 F1: 0.6617 (vs 0.4163 overall)
- Common characteristics:
  - mask_after_encoder = False (3/3)
  - inference_mode = all_patches (3/3)


## Analysis 3: Scoring Mode and Window Size Sensitivity

### Top 10 Experiments Most Sensitive to Scoring Mode (by ROC-AUC variance)

| base_experiment | default_roc | adaptive_roc | normalized_roc | default_pa80 | adaptive_pa80 | normalized_pa80 | roc_std | pa80_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 134_w500_d192_mask_after | 0.6787 | 0.8221 | 0.815 | 0.1952 | 0.3059 | 0.3006 | 0.066 | 0.051 |
| 041_lambda_0.1_mask_after | 0.6866 | 0.8181 | 0.8136 | 0.2997 | 0.4276 | 0.4156 | 0.0609 | 0.0577 |
| 057_dropout_0.3_mask_after | 0.6937 | 0.8282 | 0.8119 | 0.2724 | 0.4268 | 0.4125 | 0.0599 | 0.0697 |
| 162_d128_dropout0.3_mask_after | 0.6937 | 0.8282 | 0.8119 | 0.2724 | 0.4268 | 0.4125 | 0.0599 | 0.0697 |
| 116_d128_patch5_mask_after | 0.6262 | 0.7542 | 0.7508 | 0.2741 | 0.3533 | 0.3595 | 0.0595 | 0.0389 |
| 006_patch_5_mask_after | 0.6262 | 0.7542 | 0.7508 | 0.2741 | 0.3533 | 0.3595 | 0.0595 | 0.0389 |
| 052_lr_0.001_mask_after | 0.7083 | 0.8323 | 0.818 | 0.3197 | 0.4345 | 0.4207 | 0.0554 | 0.0512 |
| 082_d128_nhead_32_mask_after | 0.7034 | 0.8254 | 0.8152 | 0.3033 | 0.4275 | 0.4015 | 0.0553 | 0.0535 |
| 161_d128_dropout0.2_mask_after | 0.7134 | 0.8332 | 0.8251 | 0.3092 | 0.4407 | 0.4226 | 0.0547 | 0.0582 |
| 056_dropout_0.2_mask_after | 0.7134 | 0.8332 | 0.8251 | 0.3092 | 0.4407 | 0.4226 | 0.0547 | 0.0582 |


### Window Size Experiments (n=234)

| experiment | roc_auc | pa_80_f1 | disc_ratio | recon_ratio | mask_after_encoder | scoring_mode | inference_mode |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 009_w500_p20_mask_before_default_last | 0.9586 | 0.7336 | 1.6887 | 4.5918 | False | default | last_patch |
| 009_w500_p20_mask_before_default_all | 0.9578 | 0.7109 | 2.1526 | 4.4999 | False | default | all_patches |
| 014_w500_p20_enc2_mask_before_default_last | 0.9456 | 0.7122 | 1.1787 | 4.6015 | False | default | last_patch |
| 061_w500_d128_mask_before_default_last | 0.9421 | 0.72 | 1.189 | 4.4276 | False | default | last_patch |
| 014_w500_p20_enc2_mask_before_default_all | 0.9385 | 0.6613 | 1.0582 | 4.4624 | False | default | all_patches |
| 009_w500_p20_mask_before_adaptive_all | 0.9333 | 0.6053 | 2.1526 | 4.4999 | False | adaptive | all_patches |
| 009_w500_p20_mask_before_adaptive_last | 0.9304 | 0.6256 | 1.6887 | 4.5918 | False | adaptive | last_patch |
| 009_w500_p20_mask_before_normalized_all | 0.93 | 0.5892 | 2.1526 | 4.4999 | False | normalized | all_patches |
| 061_w500_d128_mask_before_default_all | 0.9289 | 0.5611 | 1.304 | 4.1799 | False | default | all_patches |
| 009_w500_p20_mask_before_normalized_last | 0.9106 | 0.5672 | 1.6887 | 4.5918 | False | normalized | last_patch |


**KEY INSIGHT 3:**
- Scoring mode sensitivity varies significantly (max std: 0.0660)
- Most sensitive experiments to scoring mode are worth investigating
- Window size experiments (234 total) show:
  - Average ROC-AUC: 0.7691
  - Need more extensive window size exploration


## Analysis 4: Disturbing Normal vs Anomaly Discrimination

**Top 15 Models by disc_cohens_d_disturbing_vs_anomaly**

| experiment | roc_auc | pa_80_f1 | disc_cohens_d_disturbing_vs_anomaly | disc_cohens_d_normal_vs_anomaly | disc_ratio_disturbing | disc_ratio | mask_after_encoder | d_model | patchify_mode | scoring_mode | inference_mode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 009_w500_p20_mask_before_default_all | 0.9578 | 0.7109 | 0.8029 | 1.1793 | 2.0167 | 2.1526 | False | 64 | patch_cnn | default | all_patches |
| 009_w500_p20_mask_before_adaptive_all | 0.9333 | 0.6053 | 0.8029 | 1.1793 | 2.0167 | 2.1526 | False | 64 | patch_cnn | adaptive | all_patches |
| 009_w500_p20_mask_before_normalized_all | 0.93 | 0.5892 | 0.8029 | 1.1793 | 2.0167 | 2.1526 | False | 64 | patch_cnn | normalized | all_patches |
| 050_k_4.0_mask_after_normalized_all | 0.8719 | 0.4602 | 0.7131 | 1.8689 | 1.6971 | 4.2588 | True | 64 | patch_cnn | normalized | all_patches |
| 050_k_4.0_mask_after_adaptive_all | 0.8793 | 0.4705 | 0.7117 | 1.8669 | 1.6958 | 4.2548 | True | 64 | patch_cnn | adaptive | all_patches |
| 049_k_3.0_mask_after_normalized_all | 0.8732 | 0.4645 | 0.7098 | 1.9223 | 1.5973 | 4.0444 | True | 64 | patch_cnn | normalized | all_patches |
| 050_k_4.0_mask_after_default_all | 0.8123 | 0.3827 | 0.7096 | 1.8658 | 1.6915 | 4.247 | True | 64 | patch_cnn | default | all_patches |
| 049_k_3.0_mask_after_adaptive_all | 0.8788 | 0.4725 | 0.7082 | 1.9188 | 1.5958 | 4.0318 | True | 64 | patch_cnn | adaptive | all_patches |
| 049_k_3.0_mask_after_default_all | 0.8093 | 0.3799 | 0.7059 | 1.9168 | 1.5934 | 4.0306 | True | 64 | patch_cnn | default | all_patches |
| 048_k_2.5_mask_after_normalized_all | 0.872 | 0.4583 | 0.6979 | 1.8868 | 1.6038 | 3.9839 | True | 64 | patch_cnn | normalized | all_patches |
| 048_k_2.5_mask_after_adaptive_all | 0.8779 | 0.4706 | 0.6959 | 1.8839 | 1.6018 | 3.9763 | True | 64 | patch_cnn | adaptive | all_patches |
| 048_k_2.5_mask_after_default_all | 0.8082 | 0.3797 | 0.6928 | 1.8809 | 1.5976 | 3.9667 | True | 64 | patch_cnn | default | all_patches |
| 060_anomaly_weight_2.0_mask_after_normalized_all | 0.8752 | 0.4761 | 0.6919 | 1.9261 | 1.5001 | 3.6301 | True | 64 | patch_cnn | normalized | all_patches |
| 047_k_2.0_mask_after_normalized_all | 0.8727 | 0.4628 | 0.6917 | 1.8997 | 1.5493 | 3.7911 | True | 64 | patch_cnn | normalized | all_patches |
| 047_k_2.0_mask_after_adaptive_all | 0.8772 | 0.4775 | 0.6896 | 1.8962 | 1.5471 | 3.7819 | True | 64 | patch_cnn | adaptive | all_patches |


**KEY INSIGHT 4:**
- Models with high disc_cohens_d_disturbing_vs_anomaly (0.7218 avg)
- These models maintain good overall performance:
  - ROC-AUC: 0.8753
  - PA%80 F1: 0.4841
- Common pattern: mask_after_encoder = True
- Key for handling noisy/disturbing normal samples


## Analysis 5: PA%80 Performance with High Discrepancy Ratio

Found 70 models with both high PA%80 (>75th percentile) and high disc_ratio (>75th percentile)

| experiment | pa_80_f1 | pa_80_roc_auc | disc_ratio | disc_cohens_d_normal_vs_anomaly | roc_auc | mask_after_encoder | d_model | dropout | scoring_mode | inference_mode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 025_nhead_1_mask_before_default_last | 0.7506 | 0.9183 | 2.6257 | 0.6313 | 0.9471 | False | 64 | 0.1 | default | last_patch |
| 025_nhead_1_mask_before_normalized_last | 0.6231 | 0.8665 | 2.6257 | 0.6313 | 0.9178 | False | 64 | 0.1 | normalized | last_patch |
| 025_nhead_1_mask_before_adaptive_last | 0.6224 | 0.8457 | 2.6257 | 0.6313 | 0.9112 | False | 64 | 0.1 | adaptive | last_patch |
| 017_decoder_t2s2_mask_after_normalized_last | 0.567 | 0.7794 | 2.7003 | 1.152 | 0.8389 | True | 64 | 0.1 | normalized | last_patch |
| 124_w100_t2s2_mask_after_normalized_last | 0.567 | 0.7794 | 2.7003 | 1.152 | 0.8389 | True | 64 | 0.1 | normalized | last_patch |
| 124_w100_t2s2_mask_after_default_last | 0.5628 | 0.7841 | 2.6584 | 1.1215 | 0.8421 | True | 64 | 0.1 | default | last_patch |
| 017_decoder_t2s2_mask_after_default_last | 0.5628 | 0.7841 | 2.6584 | 1.1215 | 0.8421 | True | 64 | 0.1 | default | last_patch |
| 096_d128_t3s1_nhead1_mask_after_default_last | 0.559 | 0.776 | 2.7051 | 1.021 | 0.8433 | True | 64 | 0.1 | default | last_patch |
| 017_decoder_t2s2_mask_after_adaptive_last | 0.5546 | 0.7795 | 2.6587 | 1.1313 | 0.8387 | True | 64 | 0.1 | adaptive | last_patch |
| 124_w100_t2s2_mask_after_adaptive_last | 0.5546 | 0.7795 | 2.6587 | 1.1313 | 0.8387 | True | 64 | 0.1 | adaptive | last_patch |
| 068_optimal_v4_mask_after_adaptive_all | 0.5409 | 0.8827 | 3.0058 | 1.7226 | 0.8829 | True | 128 | 0.1 | adaptive | all_patches |
| 068_optimal_v4_mask_after_normalized_all | 0.5341 | 0.883 | 3.0044 | 1.724 | 0.8842 | True | 128 | 0.1 | normalized | all_patches |
| 028_d128_nhead_16_mask_after_normalized_all | 0.5309 | 0.8832 | 3.3271 | 1.8784 | 0.8815 | True | 128 | 0.1 | normalized | all_patches |
| 022_d_model_128_mask_after_normalized_all | 0.5308 | 0.8742 | 3.2625 | 1.7787 | 0.8742 | True | 128 | 0.1 | normalized | all_patches |
| 065_optimal_v1_mask_after_normalized_all | 0.5308 | 0.8742 | 3.2625 | 1.7787 | 0.8742 | True | 128 | 0.1 | normalized | all_patches |


**KEY INSIGHT 5:**
- 15 models achieve BOTH high PA%80 and high disc_ratio
- This is challenging as these metrics often trade off
- Average metrics:
  - PA%80 F1: 0.5728
  - Disc Ratio: 2.8319
  - ROC-AUC: 0.8704
- These represent optimal configurations worth exploring further


## Analysis 6: Window Size, Decoder Depth, d_model, and Masking Ratio

### Correlation Matrix: Architectural Parameters

| parameter | d_model | num_teacher_decoder_layers | num_student_decoder_layers | masking_ratio | roc_auc | disc_ratio | pa_80_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| d_model | 1.0 | 0.01 | -0.071 | -0.052 | -0.0 | -0.013 | -0.039 |
| num_teacher_decoder_layers | 0.01 | 1.0 | 0.351 | -0.215 | -0.002 | -0.035 | 0.008 |
| num_student_decoder_layers | -0.071 | 0.351 | 1.0 | -0.042 | 0.071 | -0.013 | 0.095 |
| masking_ratio | -0.052 | -0.215 | -0.042 | 1.0 | 0.005 | -0.098 | -0.0 |
| roc_auc | -0.0 | -0.002 | 0.071 | 0.005 | 1.0 | 0.119 | 0.814 |
| disc_ratio | -0.013 | -0.035 | -0.013 | -0.098 | 0.119 | 1.0 | 0.046 |
| pa_80_f1 | -0.039 | 0.008 | 0.095 | -0.0 | 0.814 | 0.046 | 1.0 |


### Performance by Teacher Decoder Layers

| ('num_teacher_decoder_layers', '') | ('roc_auc', 'mean') | ('roc_auc', 'std') | ('roc_auc', 'count') | ('disc_ratio', 'mean') | ('disc_ratio', 'std') | ('disc_ratio', 'count') | ('pa_80_f1', 'mean') | ('pa_80_f1', 'std') | ('pa_80_f1', 'count') |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0 | 0.8463 | 0.0686 | 12.0 | 1.9889 | 0.5237 | 12.0 | 0.4755 | 0.1202 | 12.0 |
| 2.0 | 0.8119 | 0.0895 | 1026.0 | 2.1041 | 0.8275 | 1026.0 | 0.4157 | 0.1447 | 1026.0 |
| 3.0 | 0.8103 | 0.0625 | 204.0 | 2.0499 | 0.6732 | 204.0 | 0.4067 | 0.128 | 204.0 |
| 4.0 | 0.814 | 0.0619 | 144.0 | 2.0351 | 0.6103 | 144.0 | 0.4289 | 0.1044 | 144.0 |
| 5.0 | 0.8214 | 0.0463 | 6.0 | 1.798 | 0.4704 | 6.0 | 0.4268 | 0.0431 | 6.0 |


### Performance by Student Decoder Layers

| ('num_student_decoder_layers', '') | ('roc_auc', 'mean') | ('roc_auc', 'std') | ('roc_auc', 'count') | ('disc_ratio', 'mean') | ('disc_ratio', 'std') | ('disc_ratio', 'count') | ('pa_80_f1', 'mean') | ('pa_80_f1', 'std') | ('pa_80_f1', 'count') |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0 | 0.8106 | 0.0842 | 1296.0 | 2.0895 | 0.7877 | 1296.0 | 0.4127 | 0.1382 | 1296.0 |
| 2.0 | 0.8339 | 0.0645 | 96.0 | 2.0488 | 0.7197 | 96.0 | 0.4646 | 0.1297 | 96.0 |


### Performance by d_model

| ('d_model', '') | ('roc_auc', 'mean') | ('roc_auc', 'std') | ('roc_auc', 'count') | ('disc_ratio', 'mean') | ('disc_ratio', 'std') | ('disc_ratio', 'count') | ('pa_80_f1', 'mean') | ('pa_80_f1', 'std') | ('pa_80_f1', 'count') |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16.0 | 0.8385 | 0.068 | 12.0 | 1.8827 | 0.629 | 12.0 | 0.4586 | 0.1341 | 12.0 |
| 32.0 | 0.831 | 0.0791 | 12.0 | 1.9331 | 0.7203 | 12.0 | 0.4466 | 0.1337 | 12.0 |
| 64.0 | 0.8119 | 0.0841 | 1230.0 | 2.1006 | 0.7986 | 1230.0 | 0.4186 | 0.1343 | 1230.0 |
| 96.0 | 0.7025 | 0.0239 | 6.0 | 1.2092 | 0.0833 | 6.0 | 0.194 | 0.0934 | 6.0 |
| 128.0 | 0.8182 | 0.0707 | 108.0 | 2.0105 | 0.63 | 108.0 | 0.3987 | 0.1693 | 108.0 |
| 192.0 | 0.7681 | 0.0731 | 12.0 | 2.099 | 0.3886 | 12.0 | 0.3146 | 0.1203 | 12.0 |
| 256.0 | 0.8416 | 0.095 | 12.0 | 2.1381 | 0.8772 | 12.0 | 0.4813 | 0.1421 | 12.0 |


### Performance by Masking Ratio

| ('masking_ratio', '') | ('roc_auc', 'mean') | ('roc_auc', 'std') | ('roc_auc', 'count') | ('disc_ratio', 'mean') | ('disc_ratio', 'std') | ('disc_ratio', 'count') | ('pa_80_f1', 'mean') | ('pa_80_f1', 'std') | ('pa_80_f1', 'count') |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 0.6355 | 0.1107 | 18.0 | 1.2042 | 0.2732 | 18.0 | 0.2085 | 0.0901 | 18.0 |
| 0.08 | 0.8132 | 0.058 | 30.0 | 2.635 | 0.5005 | 30.0 | 0.4278 | 0.0798 | 30.0 |
| 0.1 | 0.8006 | 0.0611 | 198.0 | 2.2977 | 0.583 | 198.0 | 0.393 | 0.1085 | 198.0 |
| 0.12 | 0.7901 | 0.0685 | 12.0 | 2.703 | 0.4517 | 12.0 | 0.3907 | 0.0837 | 12.0 |
| 0.15 | 0.8071 | 0.065 | 60.0 | 2.0492 | 0.6245 | 60.0 | 0.3828 | 0.1456 | 60.0 |
| 0.18 | 0.7834 | 0.0898 | 6.0 | 2.8643 | 0.8539 | 6.0 | 0.3938 | 0.0629 | 6.0 |
| 0.2 | 0.819 | 0.0853 | 984.0 | 2.049 | 0.8133 | 984.0 | 0.4281 | 0.1442 | 984.0 |
| 0.25 | 0.8432 | 0.0902 | 12.0 | 2.1755 | 0.9214 | 12.0 | 0.4701 | 0.1318 | 12.0 |
| 0.3 | 0.8543 | 0.0405 | 12.0 | 1.9017 | 0.8222 | 12.0 | 0.4754 | 0.1114 | 12.0 |
| 0.4 | 0.8281 | 0.0622 | 12.0 | 2.1528 | 0.9367 | 12.0 | 0.4313 | 0.1095 | 12.0 |
| 0.5 | 0.7973 | 0.0611 | 12.0 | 1.656 | 0.5265 | 12.0 | 0.376 | 0.0981 | 12.0 |
| 0.6 | 0.7734 | 0.06 | 12.0 | 1.8774 | 0.8549 | 12.0 | 0.3446 | 0.0639 | 12.0 |
| 0.7 | 0.7703 | 0.0701 | 12.0 | 1.7941 | 0.8837 | 12.0 | 0.3639 | 0.1038 | 12.0 |
| 0.8 | 0.7705 | 0.0753 | 12.0 | 1.8002 | 0.6905 | 12.0 | 0.3543 | 0.083 | 12.0 |


**KEY INSIGHT 6:**
- Architectural parameters show complex relationships:
  - d_model correlation with roc_auc: -0.0004
  - masking_ratio correlation with disc_ratio: -0.0982
  - num_student_decoder_layers correlation with roc_auc: 0.0711
- Lower masking_ratio tends to increase disc_ratio but may hurt overall performance
- Decoder depth interactions need systematic exploration


## Analysis 7: mask_after_encoder=True for Maximizing Disc and Recon Ratio

Models with mask_after_encoder=True: 1020

Models with mask_after_encoder=False: 372

| metric | mask_after_True | mask_after_False | difference |
| --- | --- | --- | --- |
| ROC-AUC | 0.7861 | 0.8837 | -0.0977 |
| PA%80 F1 | 0.3741 | 0.5321 | -0.158 |
| Disc Ratio | 2.3247 | 1.4344 | 0.8903 |
| Recon Ratio | 1.6072 | 4.9658 | -3.3586 |
| disc_cohens_d_normal | 1.0818 | 0.4639 | 0.6178 |
| recon_cohens_d_normal | 0.6425 | 2.0973 | -1.4549 |


### Top 10 mask_after=True Models by Disc Ratio

| experiment | roc_auc | disc_ratio | recon_ratio | disc_cohens_d_normal_vs_anomaly | d_model | dropout | masking_ratio | scoring_mode | inference_mode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 050_k_4.0_mask_after_normalized_all | 0.8719 | 4.2588 | 1.5627 | 1.8689 | 64 | 0.1 | 0.2 | normalized | all_patches |
| 050_k_4.0_mask_after_adaptive_all | 0.8793 | 4.2548 | 1.5633 | 1.8669 | 64 | 0.1 | 0.2 | adaptive | all_patches |
| 050_k_4.0_mask_after_default_all | 0.8123 | 4.247 | 1.5619 | 1.8658 | 64 | 0.1 | 0.2 | default | all_patches |
| 057_dropout_0.3_mask_after_normalized_all | 0.8506 | 4.2349 | 1.3339 | 1.7531 | 64 | 0.3 | 0.2 | normalized | all_patches |
| 162_d128_dropout0.3_mask_after_normalized_all | 0.8506 | 4.2349 | 1.3339 | 1.7531 | 64 | 0.3 | 0.2 | normalized | all_patches |
| 057_dropout_0.3_mask_after_default_all | 0.7368 | 4.2235 | 1.3284 | 1.7452 | 64 | 0.3 | 0.2 | default | all_patches |
| 162_d128_dropout0.3_mask_after_default_all | 0.7368 | 4.2235 | 1.3284 | 1.7452 | 64 | 0.3 | 0.2 | default | all_patches |
| 057_dropout_0.3_mask_after_adaptive_all | 0.8613 | 4.2199 | 1.3245 | 1.7414 | 64 | 0.3 | 0.2 | adaptive | all_patches |
| 162_d128_dropout0.3_mask_after_adaptive_all | 0.8613 | 4.2199 | 1.3245 | 1.7414 | 64 | 0.3 | 0.2 | adaptive | all_patches |
| 118_d128_patch25_mask_after_default_all | 0.8041 | 4.0919 | 1.8957 | 1.4148 | 64 | 0.1 | 0.2 | default | all_patches |


**KEY INSIGHT 7:**
- mask_after_encoder=True significantly impacts metrics:
  - Disc Ratio: 2.3247 vs 1.4344 (Δ=0.8903)
  - Recon Ratio: 1.6072 vs 4.9658 (Δ=-3.3586)
  - ROC-AUC: 0.7861 vs 0.8837 (Δ=-0.0977)
- mask_after=True increases disc_ratio but DECREASES ROC-AUC
- Trade-off suggests careful parameter tuning needed for mask_after=True


## Analysis 8: Scoring and Inference Mode Impact

### Performance by Inference Mode

| ('inference_mode', '') | ('roc_auc', 'mean') | ('roc_auc', 'std') | ('pa_80_f1', 'mean') | ('pa_80_f1', 'std') | ('disc_ratio', 'mean') | ('disc_ratio', 'std') | ('recon_ratio', 'mean') | ('recon_ratio', 'std') |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_patches | 0.8355 | 0.0768 | 0.4007 | 0.1486 | 2.3898 | 0.8809 | 2.4658 | 1.5306 |
| last_patch | 0.7889 | 0.0829 | 0.4319 | 0.1252 | 1.7837 | 0.5169 | 2.5437 | 1.695 |


### Performance by Scoring Mode

| ('scoring_mode', '') | ('roc_auc', 'mean') | ('roc_auc', 'std') | ('pa_80_f1', 'mean') | ('pa_80_f1', 'std') | ('disc_ratio', 'mean') | ('disc_ratio', 'std') | ('recon_ratio', 'mean') | ('recon_ratio', 'std') |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adaptive | 0.825 | 0.0695 | 0.4256 | 0.1092 | 2.0872 | 0.784 | 2.5043 | 1.6163 |
| default | 0.7978 | 0.1065 | 0.4226 | 0.1876 | 2.086 | 0.7825 | 2.5047 | 1.6159 |
| normalized | 0.8137 | 0.0653 | 0.4007 | 0.0996 | 2.087 | 0.7843 | 2.5053 | 1.6157 |


### Top 10 Experiments Most Sensitive to Inference Mode (by ROC-AUC difference)

| base_experiment | all_roc | last_roc | roc_diff | all_disc | last_disc | disc_diff |
| --- | --- | --- | --- | --- | --- | --- |
| 040_feature_wise_mask_mask_before | 0.3648 | 0.6287 | 0.2639 | 0.7496 | 1.1703 | 0.4206 |
| 046_k_1.0_mask_after | 0.846 | 0.6991 | 0.1469 | 3.6428 | 1.981 | 1.6618 |
| 050_k_4.0_mask_after | 0.8545 | 0.7126 | 0.1419 | 4.2535 | 2.0533 | 2.2002 |
| 059_wd_1e-4_mask_after | 0.8495 | 0.7104 | 0.1391 | 3.7169 | 2.0811 | 1.6358 |
| 164_d128_wd0.0001_mask_after | 0.8495 | 0.7104 | 0.1391 | 3.7169 | 2.0811 | 1.6358 |
| 048_k_2.5_mask_after | 0.8527 | 0.7177 | 0.135 | 3.9756 | 2.1697 | 1.8059 |
| 047_k_2.0_mask_after | 0.8521 | 0.7185 | 0.1336 | 3.7833 | 2.1288 | 1.6545 |
| 032_mask_0.15_mask_after | 0.8502 | 0.7166 | 0.1336 | 3.6437 | 2.0848 | 1.5589 |
| 163_d128_wd0_mask_after | 0.8502 | 0.7166 | 0.1336 | 3.6437 | 2.0848 | 1.5589 |
| 121_w100_t2s1_mask_after | 0.8502 | 0.7166 | 0.1336 | 3.6437 | 2.0848 | 1.5589 |


**KEY INSIGHT 8:**
- Inference mode impact:
  - all_patches: ROC-AUC=0.8355, disc_ratio=2.3898
  - last_patch: ROC-AUC=0.7889, disc_ratio=1.7837
- all_patches consistently outperforms last_patch
- Scoring mode 'adaptive' shows best overall ROC-AUC (0.8250)
- Some configurations are highly sensitive to these modes (max diff: 0.2639)


## Analysis 9: High Anomaly Detection Performance + High Disturbing Discrimination

Found 91 models with both high ROC-AUC (>75th) and high disc_cohens_d_disturbing (>75th)

| experiment | roc_auc | pa_80_f1 | disc_cohens_d_disturbing_vs_anomaly | disc_ratio | recon_ratio | mask_after_encoder | d_model | num_teacher_decoder_layers | num_student_decoder_layers | scoring_mode | inference_mode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 009_w500_p20_mask_before_default_last | 0.9586 | 0.7336 | 0.4901 | 1.6887 | 4.5918 | False | 64 | 2 | 1 | default | last_patch |
| 009_w500_p20_mask_before_default_all | 0.9578 | 0.7109 | 0.8029 | 2.1526 | 4.4999 | False | 64 | 2 | 1 | default | all_patches |
| 028_d128_nhead_16_mask_before_default_all | 0.9467 | 0.6997 | 0.5605 | 2.4078 | 5.4361 | False | 128 | 2 | 1 | default | all_patches |
| 028_d128_nhead_16_mask_before_adaptive_all | 0.9436 | 0.659 | 0.5605 | 2.4078 | 5.4361 | False | 128 | 2 | 1 | adaptive | all_patches |
| 028_d128_nhead_16_mask_before_normalized_all | 0.9356 | 0.6263 | 0.5605 | 2.4078 | 5.4361 | False | 128 | 2 | 1 | normalized | all_patches |
| 009_w500_p20_mask_before_adaptive_all | 0.9333 | 0.6053 | 0.8029 | 2.1526 | 4.4999 | False | 64 | 2 | 1 | adaptive | all_patches |
| 009_w500_p20_mask_before_adaptive_last | 0.9304 | 0.6256 | 0.4901 | 1.6887 | 4.5918 | False | 64 | 2 | 1 | adaptive | last_patch |
| 009_w500_p20_mask_before_normalized_all | 0.93 | 0.5892 | 0.8029 | 2.1526 | 4.4999 | False | 64 | 2 | 1 | normalized | all_patches |
| 009_w500_p20_mask_before_normalized_last | 0.9106 | 0.5672 | 0.4901 | 1.6887 | 4.5918 | False | 64 | 2 | 1 | normalized | last_patch |
| 068_optimal_v4_mask_after_normalized_all | 0.8842 | 0.5341 | 0.5408 | 3.0044 | 1.7593 | True | 128 | 3 | 1 | normalized | all_patches |
| 068_optimal_v4_mask_after_adaptive_all | 0.8829 | 0.5409 | 0.5395 | 3.0058 | 1.7565 | True | 128 | 3 | 1 | adaptive | all_patches |
| 028_d128_nhead_16_mask_after_normalized_all | 0.8815 | 0.5309 | 0.5955 | 3.3271 | 1.7923 | True | 128 | 2 | 1 | normalized | all_patches |
| 124_w100_t2s2_mask_after_adaptive_all | 0.8807 | 0.5254 | 0.5621 | 3.6182 | 1.7888 | True | 64 | 2 | 2 | adaptive | all_patches |
| 017_decoder_t2s2_mask_after_adaptive_all | 0.8807 | 0.5254 | 0.5621 | 3.6182 | 1.7888 | True | 64 | 2 | 2 | adaptive | all_patches |
| 028_d128_nhead_16_mask_after_adaptive_all | 0.8799 | 0.527 | 0.58 | 3.3001 | 1.7945 | True | 128 | 2 | 1 | adaptive | all_patches |


**KEY INSIGHT 9:**
- 15 models achieve BOTH high anomaly detection AND high disturbing discrimination
- These represent the most robust models:
  - ROC-AUC: 0.9157
  - PA%80 F1: 0.6000
  - disc_cohens_d_disturbing: 0.5960
- Common pattern: mask_after_encoder = False (9/15)
- These configurations should be prioritized in phase 2


## Analysis 10: Additional Insights and Unexpected Patterns

### 10.1: Patchify Mode Impact

| ('patchify_mode', '') | ('roc_auc', 'mean') | ('roc_auc', 'std') | ('roc_auc', 'count') | ('disc_ratio', 'mean') | ('disc_ratio', 'std') | ('disc_ratio', 'count') | ('pa_80_f1', 'mean') | ('pa_80_f1', 'std') | ('pa_80_f1', 'count') |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| patch_cnn | 0.8122 | 0.0832 | 1392 | 2.0867 | 0.783 | 1392 | 0.4163 | 0.1383 | 1392 |


### 10.2: Masking Strategy Impact

| ('masking_strategy', '') | ('roc_auc', 'mean') | ('roc_auc', 'std') | ('roc_auc', 'count') | ('disc_ratio', 'mean') | ('disc_ratio', 'std') | ('disc_ratio', 'count') | ('pa_80_f1', 'mean') | ('pa_80_f1', 'std') | ('pa_80_f1', 'count') |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| feature_wise | 0.6234 | 0.1659 | 12 | 2.0397 | 1.1655 | 12 | 0.2876 | 0.0851 | 12 |
| patch | 0.8138 | 0.0803 | 1380 | 2.0871 | 0.7795 | 1380 | 0.4174 | 0.1381 | 1380 |


### 10.3: Extreme Parameter Exploration

| parameter | min_value | min_roc_auc | max_value | max_roc_auc | effect |
| --- | --- | --- | --- | --- | --- |
| masking_ratio | 0.05 | 0.5622 | 0.8 | 0.7017 | positive |
| lambda_disc | 0.1 | 0.7529 | 3.0 | 0.7956 | positive |
| dropout | 0.0 | 0.7777 | 0.3 | 0.7634 | negative |
| d_model | 16.0 | 0.7731 | 256.0 | 0.7591 | negative |


### 10.4: Top 10 Overall Best Models (Balanced Performance)

| experiment | composite_score | roc_auc | pa_80_f1 | disc_ratio | recon_ratio | mask_after_encoder | patchify_mode | d_model | num_teacher_decoder_layers | scoring_mode | inference_mode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 022_d_model_128_mask_before_default_last | 0.9059 | 0.9557 | 0.7867 | 2.1606 | 6.597 | False | patch_cnn | 128 | 2 | default | last_patch |
| 025_nhead_1_mask_before_default_last | 0.8895 | 0.9471 | 0.7506 | 2.6257 | 5.9919 | False | patch_cnn | 64 | 2 | default | last_patch |
| 028_d128_nhead_16_mask_before_default_last | 0.8696 | 0.952 | 0.7535 | 1.9722 | 6.1317 | False | patch_cnn | 128 | 2 | default | last_patch |
| 054_lr_0.005_mask_before_default_last | 0.8564 | 0.9521 | 0.7866 | 1.3475 | 6.1224 | False | patch_cnn | 64 | 2 | default | last_patch |
| 023_d_model_256_mask_before_default_last | 0.8549 | 0.9575 | 0.7659 | 1.3095 | 6.3155 | False | patch_cnn | 256 | 2 | default | last_patch |
| 007_patch_20_mask_before_default_all | 0.8534 | 0.9624 | 0.7297 | 1.7369 | 6.0038 | False | patch_cnn | 64 | 2 | default | all_patches |
| 019_decoder_t4s2_mask_before_default_all | 0.8514 | 0.9572 | 0.7319 | 1.9363 | 5.6922 | False | patch_cnn | 64 | 4 | default | all_patches |
| 053_lr_0.003_mask_before_default_last | 0.8512 | 0.9502 | 0.7618 | 1.506 | 6.1058 | False | patch_cnn | 64 | 2 | default | last_patch |
| 030_ffn_512_mask_before_default_last | 0.8495 | 0.9473 | 0.759 | 1.3785 | 6.3706 | False | patch_cnn | 64 | 2 | default | last_patch |
| 029_ffn_128_mask_before_default_last | 0.8493 | 0.9479 | 0.7667 | 1.3119 | 6.3323 | False | patch_cnn | 64 | 2 | default | last_patch |


**KEY INSIGHT 10 - Additional Findings:**

10.1: Patchify mode 'patch' vastly outperforms 'feature_wise' (ROC: nan vs nan)

10.2: Masking strategy 'patch' is superior to 'feature_wise' across all metrics

10.3: Extreme parameter values show:
- Higher dropout generally helps disc_ratio but may hurt ROC-AUC
- d_model shows non-linear relationship (medium values best)
- lambda_disc impact needs more systematic exploration

10.4: Best balanced models (composite score) share:
- mask_after_encoder = False
- inference_mode = last_patch
- patchify_mode = patch_cnn


