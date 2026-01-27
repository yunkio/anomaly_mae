# Phase 1 Ablation Study Analysis Report

**Generated:** 2026-01-27 14:30:15

**Total Experiments:** 1398

## Overview

- Total experiments: 1398
- Inference modes: 2 (all_patches, last_patch)
- Scoring modes: 3 (adaptive, default, normalized)
- ROC-AUC range: 0.3643 - 0.9624
- Average ROC-AUC: 0.8126 ± 0.0834

## Summary Tables

### Table 1: Top 10 Models by ROC-AUC

| Rank | Model | ROC-AUC | F1 | PA20 AUC | PA50 AUC | PA80 AUC | disc_ratio | t_ratio | disc_d | recon_d | Inf Mode | Score |
|------|-------|---------|-----|----------|----------|----------|------------|---------|--------|---------|----------|-------|
| 1 | 007_patch_20 | 0.9624 | 0.846 | 0.987 | 0.980 | 0.960 | 1.74 | 6.00 | 0.97 | 1.95 | all_patches | default |
| 2 | 009_w500_p20 | 0.9586 | 0.883 | 0.979 | 0.969 | 0.926 | 1.69 | 4.59 | 0.73 | 2.08 | last_patch | default |
| 3 | 009_w500_p20 | 0.9578 | 0.740 | 0.994 | 0.989 | 0.965 | 2.15 | 4.50 | 1.18 | 3.77 | all_patches | default |
| 4 | 023_d_model_256 | 0.9575 | 0.848 | 0.981 | 0.969 | 0.921 | 1.31 | 6.32 | 0.41 | 2.11 | last_patch | default |
| 5 | 019_decoder_t4s2 | 0.9572 | 0.812 | 0.993 | 0.988 | 0.960 | 1.94 | 5.69 | 0.90 | 2.30 | all_patches | default |
| 6 | 063_w500_p20_d128 | 0.9567 | 0.657 | 0.992 | 0.985 | 0.961 | 1.61 | 5.50 | 1.05 | 3.49 | all_patches | default |
| 7 | 016_decoder_t4s1 | 0.9564 | 0.805 | 0.994 | 0.988 | 0.959 | 1.97 | 5.60 | 0.82 | 2.31 | all_patches | default |
| 8 | 022_d_model_128 | 0.9557 | 0.864 | 0.983 | 0.969 | 0.930 | 2.16 | 6.60 | 0.83 | 2.15 | last_patch | default |
| 9 | 015_decoder_t3s1 | 0.9555 | 0.814 | 0.993 | 0.986 | 0.958 | 1.73 | 5.67 | 0.73 | 2.23 | all_patches | default |
| 10 | 018_decoder_t3s2 | 0.9532 | 0.808 | 0.993 | 0.987 | 0.957 | 1.67 | 5.72 | 0.65 | 2.20 | all_patches | default |

### Table 2: Top 10 Models by Discrepancy Ratio

| Rank | Model | disc_ratio | ROC-AUC | F1 | PA80 AUC | disc_d | disc_d_disturb | t_ratio | recon_d | Inf Mode | Score |
|------|-------|------------|---------|-----|----------|--------|----------------|---------|---------|----------|-------|
| 1 | 050_k_4.0 | 4.259 | 0.8719 | 0.581 | 0.870 | 1.87 | 0.71 | 1.56 | 0.53 | all_patches | normalized |
| 2 | 050_k_4.0 | 4.255 | 0.8793 | 0.577 | 0.882 | 1.87 | 0.71 | 1.56 | 0.53 | all_patches | adaptive |
| 3 | 050_k_4.0 | 4.247 | 0.8123 | 0.517 | 0.773 | 1.87 | 0.71 | 1.56 | 0.53 | all_patches | default |
| 4 | 057_dropout_0.3 | 4.235 | 0.8506 | 0.536 | 0.845 | 1.75 | 0.59 | 1.33 | 0.33 | all_patches | normalized |
| 5 | 162_d128_dropout0.3 | 4.235 | 0.8506 | 0.536 | 0.845 | 1.75 | 0.59 | 1.33 | 0.33 | all_patches | normalized |
| 6 | 057_dropout_0.3 | 4.223 | 0.7368 | 0.418 | 0.655 | 1.75 | 0.58 | 1.33 | 0.32 | all_patches | default |
| 7 | 162_d128_dropout0.3 | 4.223 | 0.7368 | 0.418 | 0.655 | 1.75 | 0.58 | 1.33 | 0.32 | all_patches | default |
| 8 | 057_dropout_0.3 | 4.220 | 0.8613 | 0.563 | 0.862 | 1.74 | 0.58 | 1.32 | 0.32 | all_patches | adaptive |
| 9 | 162_d128_dropout0.3 | 4.220 | 0.8613 | 0.563 | 0.862 | 1.74 | 0.58 | 1.32 | 0.32 | all_patches | adaptive |
| 10 | 118_d128_patch25 | 4.092 | 0.8041 | 0.520 | 0.741 | 1.41 | 0.11 | 1.90 | 1.05 | all_patches | default |

### Table 3: Top 10 Models by Teacher Reconstruction Ratio

| Rank | Model | t_ratio | ROC-AUC | F1 | PA80 AUC | disc_ratio | disc_d | recon_d | Inf Mode | Score |
|------|-------|---------|---------|-----|----------|------------|--------|---------|----------|-------|
| 1 | 022_d_model_128 | 6.597 | 0.9277 | 0.771 | 0.888 | 2.16 | 0.83 | 2.15 | last_patch | adaptive |
| 2 | 022_d_model_128 | 6.597 | 0.9557 | 0.864 | 0.930 | 2.16 | 0.83 | 2.15 | last_patch | default |
| 3 | 022_d_model_128 | 6.597 | 0.9182 | 0.754 | 0.866 | 2.16 | 0.83 | 2.15 | last_patch | normalized |
| 4 | 030_ffn_512 | 6.371 | 0.8971 | 0.723 | 0.835 | 1.38 | 0.40 | 2.00 | last_patch | adaptive |
| 5 | 030_ffn_512 | 6.371 | 0.9473 | 0.842 | 0.915 | 1.38 | 0.40 | 2.00 | last_patch | default |
| 6 | 030_ffn_512 | 6.371 | 0.8559 | 0.657 | 0.771 | 1.38 | 0.40 | 2.00 | last_patch | normalized |
| 7 | 029_ffn_128 | 6.332 | 0.8954 | 0.719 | 0.837 | 1.31 | 0.36 | 2.09 | last_patch | adaptive |
| 8 | 029_ffn_128 | 6.332 | 0.9479 | 0.848 | 0.917 | 1.31 | 0.36 | 2.09 | last_patch | default |
| 9 | 029_ffn_128 | 6.332 | 0.8566 | 0.619 | 0.782 | 1.31 | 0.36 | 2.09 | last_patch | normalized |
| 10 | 023_d_model_256 | 6.316 | 0.9159 | 0.763 | 0.857 | 1.31 | 0.41 | 2.11 | last_patch | adaptive |

## Deep Analysis: 10 Focus Areas

### Focus Area 1: High Discrepancy Ratio Characteristics

**Objective:** Identify characteristics of models with high disc_cohens_d_normal_vs_anomaly

**Key Findings:**

- Top 50 models have disc_cohens_d_normal_vs_anomaly ranging from 1.875 to 1.926
- Average ROC-AUC in top 50: 0.8596
- Average disc_ratio in top 50: 3.682
- Inference mode distribution:
  - all_patches: 50 (100.0%)

**Top 5 Models:**

| Model | disc_d | ROC-AUC | disc_ratio | t_ratio | Inf Mode | Score |
|-------|--------|---------|------------|---------|----------|-------|
| 060_anomaly_weight_2.0 | 1.926 | 0.8752 | 3.63 | 1.55 | all_patches | normalized |
| 049_k_3.0 | 1.922 | 0.8732 | 4.04 | 1.55 | all_patches | normalized |
| 060_anomaly_weight_2.0 | 1.922 | 0.8068 | 3.61 | 1.55 | all_patches | default |
| 060_anomaly_weight_2.0 | 1.921 | 0.8787 | 3.61 | 1.55 | all_patches | adaptive |
| 049_k_3.0 | 1.919 | 0.8788 | 4.03 | 1.55 | all_patches | adaptive |

### Focus Area 2: High Disc AND High Recon Cohen's d

**Objective:** Find models with both high disc_cohens_d AND recon_cohens_d

**Key Findings:**

- Found 3 models with both disc_d > 1.332 AND recon_d > 1.734
- Average ROC-AUC: 0.9420
- Average disc_ratio: 2.408
- Average t_ratio: 5.436
- Average PA%80 ROC-AUC: 0.9508

**Inference Mode Distribution:**
- all_patches: 3 (100.0%)

**Top 10 Models:**

| Model | ROC-AUC | disc_d | recon_d | disc_ratio | t_ratio | PA80 AUC |
|-------|---------|--------|---------|------------|---------|----------|
| 028_d128_nhead_16 | 0.9467 | 1.39 | 2.20 | 2.41 | 5.44 | 0.953 |
| 028_d128_nhead_16 | 0.9436 | 1.39 | 2.20 | 2.41 | 5.44 | 0.954 |
| 028_d128_nhead_16 | 0.9356 | 1.39 | 2.20 | 2.41 | 5.44 | 0.946 |

### Focus Area 3: Scoring Mode and Window Size Effects

**Objective:** Understand how scoring mode and window size affect performance

**Scoring Mode Comparison:**

| Scoring Mode | Avg ROC-AUC | Avg disc_ratio | Avg t_ratio | Count |
|--------------|-------------|----------------|-------------|-------|
| default | 0.7985 | 2.083 | 2.517 | 466 |
| adaptive | 0.8254 | 2.084 | 2.517 | 466 |
| normalized | 0.8140 | 2.084 | 2.518 | 466 |

**Inference Mode Comparison:**

| Inference Mode | Avg ROC-AUC | Avg disc_ratio | Avg t_ratio | Count |
|----------------|-------------|----------------|-------------|-------|
| all_patches | 0.8359 | 2.386 | 2.479 | 699 |
| last_patch | 0.7893 | 1.781 | 2.555 | 699 |

### Focus Area 4: Disturbing Normal vs Anomaly Separation

**Objective:** Identify models that separate disturbing normal from anomaly well

**Key Findings:**

- Top disc_cohens_d_disturbing_vs_anomaly: 0.803
- Average ROC-AUC in top 20: 0.8811
- Average disc_ratio_2 (disturbing/anomaly) in top 20: 1.638

**Top 10 Models:**

| Model | disc_d_disturb | ROC-AUC | disc_ratio_2 | disc_ratio_1 | Inf Mode |
|-------|----------------|---------|--------------|--------------|----------|
| 009_w500_p20 | 0.803 | 0.9333 | 2.02 | 2.15 | all_patches |
| 009_w500_p20 | 0.803 | 0.9578 | 2.02 | 2.15 | all_patches |
| 009_w500_p20 | 0.803 | 0.9300 | 2.02 | 2.15 | all_patches |
| 050_k_4.0 | 0.713 | 0.8719 | 1.70 | 4.26 | all_patches |
| 050_k_4.0 | 0.712 | 0.8793 | 1.70 | 4.25 | all_patches |
| 049_k_3.0 | 0.710 | 0.8732 | 1.60 | 4.04 | all_patches |
| 050_k_4.0 | 0.710 | 0.8123 | 1.69 | 4.25 | all_patches |
| 049_k_3.0 | 0.708 | 0.8788 | 1.60 | 4.03 | all_patches |
| 049_k_3.0 | 0.706 | 0.8093 | 1.59 | 4.03 | all_patches |
| 063_w500_p20_d128 | 0.705 | 0.9410 | 1.48 | 1.61 | all_patches |

### Focus Area 5: High PA%80 with High Disc Ratio

**Objective:** Find models with both high PA%80 performance and high disc_ratio

**Key Findings:**

- Found 135 models with PA%80 > 0.852 AND disc_ratio > 2.623
- Average ROC-AUC: 0.8722
- Average PA%80 ROC-AUC: 0.8699
- Average disc_ratio: 3.402

**Top 10 Models:**

| Model | PA80 AUC | ROC-AUC | disc_ratio | t_ratio | Inf Mode | Score |
|-------|----------|---------|------------|---------|----------|-------|
| 025_nhead_1 | 0.9183 | 0.9471 | 2.63 | 5.99 | last_patch | default |
| 017_decoder_t2s2 | 0.8846 | 0.8807 | 3.62 | 1.79 | all_patches | adaptive |
| 124_w100_t2s2 | 0.8846 | 0.8807 | 3.62 | 1.79 | all_patches | adaptive |
| 028_d128_nhead_16 | 0.8832 | 0.8815 | 3.33 | 1.79 | all_patches | normalized |
| 068_optimal_v4 | 0.8830 | 0.8842 | 3.00 | 1.76 | all_patches | normalized |
| 017_decoder_t2s2 | 0.8828 | 0.8794 | 3.62 | 1.79 | all_patches | normalized |
| 124_w100_t2s2 | 0.8828 | 0.8794 | 3.62 | 1.79 | all_patches | normalized |
| 068_optimal_v4 | 0.8827 | 0.8829 | 3.01 | 1.76 | all_patches | adaptive |
| 050_k_4.0 | 0.8819 | 0.8793 | 4.25 | 1.56 | all_patches | adaptive |
| 049_k_3.0 | 0.8818 | 0.8788 | 4.03 | 1.55 | all_patches | adaptive |

### Focus Area 6: Window Size, Depth, and Masking Ratio

**Analysis:** Parameter extraction needed for detailed analysis. Recommend examining experiment names manually for patterns.

### Focus Area 7: Mask After Optimization

**Analysis:** Extract mask_after experiments and optimize for high disc + t_ratio.

### Focus Area 8: Scoring/Inference Sensitivity

**Key Finding:** Default scoring outperforms adaptive and normalized on average.
all_patches inference slightly better (0.8359 vs 0.7893).

### Focus Area 9: High Performance + Disturbing Separation

**Found 9 models with ROC > 0.945 AND high disturbing separation.**

### Focus Area 10: Additional Insights

**Key Insights:**

1. Overall average ROC-AUC: 0.8126 ± 0.0834
2. Best single model: 007_patch_20 (0.9624)
3. disc_ratio and ROC-AUC correlation: 0.114
4. t_ratio and ROC-AUC correlation: 0.650

## Key Insights and Hypotheses

### Insight 1: Discrepancy Ratio Alone Insufficient

**Observation:** disc_ratio_1 correlation with ROC-AUC is 0.114, indicating that high discrepancy ratio alone doesn't guarantee good performance.

**Hypothesis:** Models need BOTH good discrepancy (separation) AND good reconstruction (t_ratio) to achieve high ROC-AUC.

### Insight 2: Cohen's d Metrics Are Better Predictors

**Observation:** disc_cohens_d correlation with ROC-AUC is 0.137, while recon_cohens_d correlation is 0.600.

**Hypothesis:** Cohen's d metrics (effect size) are better performance indicators than raw ratios.

### Insight 3: Inference Mode Matters

**Observation:** all_patches achieves 0.8359 vs last_patch 0.7893.

**Hypothesis:** all_patches provides more robust aggregation of patch-level information.

### Insight 4: Scoring Mode Effects

**Observation:** default (0.7985) > adaptive (0.8254) > normalized (0.8140).

**Hypothesis:** Default scoring (simple averaging) works best for this dataset.

### Insight 5: High t_ratio Models

**Observation:** Models with high t_ratio (top 50) average ROC-AUC: 0.9164.

**Hypothesis:** Teacher reconstruction ratio is a strong indicator of model quality.

## Phase 2 Experiment Recommendations

### Priority 1: Maximize Disc + Recon Cohen's d (30 experiments)

**Goal:** Find parameter combinations that maximize both disc_cohens_d and recon_cohens_d.

- Base on top 0 models from Phase 1
- Focus on all_patches inference mode (0 of top use this)
- Focus on default scoring mode
- Vary: d_model, masking_ratio, decoder_depth

### Priority 2: Disturbing Normal Separation (25 experiments)

**Goal:** Improve disc_cohens_d_disturbing_vs_anomaly while maintaining high ROC-AUC.

- Focus on models with high disc_ratio_2
- Experiment with different patch sizes
- Test window size variations (500, 1000)

### Priority 3: PA%80 Optimization (25 experiments)

**Goal:** Maximize PA%80 performance for practical deployment.

- Start from high PA%80 + disc_ratio models
- Test scoring mode combinations
- Experiment with ensemble approaches

### Priority 4: Window Size + Depth Relationships (20 experiments)

**Goal:** Understand optimal model capacity for different window sizes.

- Window 500: vary decoder_depth (2, 3, 4, 5, 6)
- Window 500: vary d_model (96, 128, 192, 256, 320)
- Window 1000: test if larger capacity helps

### Priority 5: Mask After + Lambda Optimization (20 experiments)

**Goal:** Optimize discrepancy loss weighting for mask_after models.

- Test lambda_disc values: 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0
- Combine with high-performing base configurations

### Priority 6: Teacher-Student Ratio Exploration (15 experiments)

**Goal:** Find optimal teacher-student loss balance.

- Test ratios: t1s1, t2s1, t3s1, t4s1, t5s1
- Test ratios: t2s2, t3s2, t4s2

### Priority 7: Masking Ratio Fine-tuning (15 experiments)

**Goal:** Find optimal masking ratio for different model sizes.

- d_model=128: test masking [0.05, 0.10, 0.15, 0.20, 0.25]
- d_model=256: test masking [0.60, 0.70, 0.75, 0.80, 0.85]

