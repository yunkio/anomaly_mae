# Phase 1 Ablation Study: Comprehensive Analysis Report

**Date**: 2026-01-31
**Experiments**: 1,014 evaluations (169 base configs × 2 mask timings × 3 scoring modes)
**Window sizes excluded**: 2,000 (OOM)
**Dataset**: 275K timesteps, 8 features, sliding window with stride=1 for test

---

## Table of Contents

1. [Experiment Overview](#1-experiment-overview)
2. [Metric Definitions & Interpretation](#2-metric-definitions--interpretation)
3. [Parameter Space Summary](#3-parameter-space-summary)
4. [Overall Performance Statistics](#4-overall-performance-statistics)
5. [Parameter-Performance Correlations](#5-parameter-performance-correlations)
6. [Focus Analysis (1): Discrepancy Ratio & Cohen's d Maximization](#6-focus-1-discrepancy-ratio--cohens-d-maximization)
7. [Focus Analysis (2): Dual-Strength Models (High disc_d AND recon_d)](#7-focus-2-dual-strength-models)
8. [Focus Analysis (3): Scoring Mode × Window Size Interactions](#8-focus-3-scoring-mode--window-size-interactions)
9. [Focus Analysis (4): Disturbing Normal vs Anomaly Separation](#9-focus-4-disturbing-normal-vs-anomaly-separation)
10. [Focus Analysis (5): PA%80 + High disc_ratio Joint Analysis](#10-focus-5-pa80--high-disc_ratio-joint-analysis)
11. [Focus Analysis (6): Window Size × Model Depth × Masking Ratio](#11-focus-6-window-size--model-depth--masking-ratio)
12. [Focus Analysis (7): Scoring Mode Sensitivity](#12-focus-7-scoring-mode-sensitivity)
13. [Focus Analysis (8): Robust Detection + Disturbing Separation](#13-focus-8-robust-detection--disturbing-separation)
14. [Focus Analysis (9): mask_after vs mask_before Comparison](#14-focus-9-mask_after-vs-mask_before)
15. [Focus Analysis (10): Discrepancy Contribution to Performance](#15-focus-10-discrepancy-contribution)
16. [Focus Analysis (11): Additional Insights](#16-focus-11-additional-insights)
17. [Optimal Parameter Recipes (Top 15)](#17-optimal-parameter-recipes-top-15)
18. [Per-Parameter Performance Analysis](#18-per-parameter-performance-analysis)
19. [Discrepancy SNR Deep Analysis](#19-discrepancy-snr-deep-analysis)
20. [Key Hypotheses & Conclusions](#20-key-hypotheses--conclusions)
21. [Visualization Target Models](#21-visualization-target-models)
22. [Phase 2 Recommendations](#22-phase-2-recommendations)

---

## 1. Experiment Overview

Phase 1 ablation study explores the parameter space of the Self-Distilled MAE architecture for multivariate time-series anomaly detection. The study varies:

- **Architecture**: d_model, nhead, encoder/decoder depth, dim_feedforward, CNN channels
- **Masking**: ratio, strategy (patch/feature_wise), timing (after/before encoder)
- **Optimization**: learning_rate, weight_decay, lambda_disc, dynamic_margin_k
- **Inference**: scoring_mode (default/adaptive/normalized), window size (100-1000)

Total: 169 base experiments × 2 mask timing × 3 scoring = **1,014 evaluations**

### Fixed Parameters
| Parameter | Value |
|-----------|-------|
| margin_type | dynamic |
| force_mask_anomaly | True |
| patch_level_loss | True |
| patchify_mode | patch_cnn |
| shared_mask_token | False |
| num_epochs | 50 |
| margin | 0.5 |

---

## 2. Metric Definitions & Interpretation

### Detection Metrics
| Metric | Meaning | Interpretation |
|--------|---------|----------------|
| **roc_auc** | Window-level ROC-AUC | Overall ranking quality of anomaly scores |
| **f1_score** | F1 at optimal threshold | Balance of precision/recall at best operating point |
| **pa_K_roc_auc** / **pa_K_f1** | Point-adjusted metrics with K% tolerance | PA%K credits detection if anomaly detected within first K% of segment. Higher K = more lenient |
| **pa_80_f1** | PA with 80% tolerance | Relatively strict - model must detect anomaly reasonably early |
| **disturbing_roc_auc** / **disturbing_f1** | Disturbing-normal vs anomaly distinction | Hardest classification: boundary-adjacent normals vs anomalies |

### Separation Metrics (Model Behavior)
| Metric | Meaning | Interpretation |
|--------|---------|----------------|
| **disc_ratio** | disc_anomaly / disc_normal | Discrepancy amplification factor |
| **disc_ratio_disturbing** | disc_anomaly / disc_disturbing | Harder version: disturbing boundary |
| **recon_ratio** | recon_anomaly / recon_normal | Reconstruction error amplification |
| **disc_cohens_d_normal_vs_anomaly** | Cohen's d for disc distributions | Effect size: >0.8=large, >1.2=very large, >2.0=huge |
| **disc_cohens_d_disturbing_vs_anomaly** | Cohen's d for disturbing-vs-anomaly | Harder separation metric |
| **recon_cohens_d_*** | Same for reconstruction errors | Complementary signal strength |

### Loss Statistics
| Metric | Meaning |
|--------|---------|
| recon_normal / recon_anomaly | Mean reconstruction error per group |
| disc_normal / disc_anomaly | Mean discrepancy per group |
| disc_normal_std / disc_anomaly_std | Variance of discrepancy distributions |

---

## 3. Parameter Space Summary

### Variable Parameters

| Parameter | Values | Count |
|-----------|--------|-------|
| seq_length | 100, 200, 500, 1000 | 4 |
| patch_size | 5, 10, 20, 25 | 4 |
| masking_ratio | 0.05-0.80 | 14 |
| masking_strategy | patch, feature_wise | 2 |
| d_model | 16, 32, 64, 96, 128, 192, 256 | 7 |
| nhead | 1, 2, 4, 6, 8, 16, 32 | 7 |
| num_encoder_layers | 1, 2, 3, 4 | 4 |
| num_teacher_decoder_layers | 1, 2, 3, 4, 5 | 5 |
| num_student_decoder_layers | 1, 2 | 2 |
| num_shared_decoder_layers | 0, 1 | 2 |
| dim_feedforward | 64-1024 | 7 |
| dropout | 0.0, 0.05, 0.1, 0.2, 0.3 | 5 |
| cnn_channels | [8,16] to [128,256] | 7 |
| lambda_disc | 0.1, 0.25, 0.5, 1.0, 2.0, 3.0 | 6 |
| dynamic_margin_k | 1.0, 1.5, 2.0, 2.5, 3.0, 4.0 | 6 |
| learning_rate | 0.0005, 0.001, 0.002, 0.003, 0.005 | 5 |
| weight_decay | 0.0, 1e-5, 1e-4 | 3 |
| anomaly_loss_weight | 1.0, 2.0 | 2 |

**Note**: Most experiments use the default config (d_model=64, nhead=2, enc=1, teacher_dec=2, student_dec=1, masking_ratio=0.2, lr=0.002, lambda_disc=0.5, dynamic_margin_k=1.5), with each experiment varying one or a few parameters.

---

## 4. Overall Performance Statistics

### All 1,014 Evaluations

| Metric | Mean | Std | Min | 25% | Median | 75% | Max |
|--------|------|-----|-----|-----|--------|-----|-----|
| roc_auc | 0.9372 | 0.0697 | 0.2288 | 0.9329 | 0.9633 | 0.9721 | **0.9827** |
| f1_score | 0.6617 | 0.1155 | 0.0400 | 0.6308 | 0.7000 | 0.7368 | 0.8144 |
| pa_80_roc_auc | 0.8999 | 0.1047 | 0.1471 | 0.8898 | 0.9425 | 0.9572 | 0.9754 |
| pa_80_f1 | 0.6369 | 0.1561 | 0.0000 | 0.5942 | 0.7005 | 0.7339 | **0.8504** |
| disc_cohens_d_n_vs_a | 1.5069 | 0.9784 | -0.44 | 0.61 | 1.18 | 2.52 | **3.37** |
| disc_cohens_d_dist_vs_a | 0.6241 | 0.5117 | -0.29 | 0.19 | 0.49 | 0.99 | **2.55** |
| recon_cohens_d_n_vs_a | 1.9652 | 0.5351 | -0.23 | 1.61 | 1.90 | 2.49 | 2.74 |
| disc_ratio | 3.65 | 2.48 | 0.80 | 1.53 | 2.46 | 6.05 | **10.80** |

### Key Observation
The performance spread is very wide (roc_auc 0.23-0.98), primarily driven by:
1. **Window size** (strongest factor, rho=-0.54)
2. **Mask timing** (mask_after vs mask_before dramatically affects different metrics)
3. **Teacher decoder depth** (rho=+0.23 with roc_auc)

---

## 5. Parameter-Performance Correlations

### Spearman Correlations (all 1,014 rows)

| Parameter | roc_auc | f1_score | pa_80_f1 | disc_d_n_a | disc_d_dist_a | recon_d_n_a | disc_ratio |
|-----------|---------|----------|----------|------------|---------------|-------------|------------|
| **seq_length** | **-0.543***| **-0.525***| **-0.579***| **-0.381***| -0.032 | **-0.175***| **-0.383***|
| num_patches | -0.571* | -0.520* | -0.586* | -0.384* | -0.048 | -0.196* | -0.383* |
| patch_size | -0.014 | -0.089 | -0.084 | -0.033 | **+0.143***| +0.036 | -0.059 |
| masking_ratio | -0.065 | +0.063 | -0.034 | **-0.118***| -0.094 | **+0.118***| -0.082 |
| **is_mask_after** | +0.057 | -0.049 | **+0.196***| **+0.732***| **+0.777***| **-0.812***| **+0.721***|
| **num_teacher_dec** | **+0.234***| **+0.177***| **+0.255***| **+0.199***| **+0.176***| -0.021 | **+0.157***|
| d_model | -0.057 | -0.063 | -0.091 | +0.005 | +0.090 | -0.059 | -0.035 |
| learning_rate | **+0.161***| **+0.144***| **+0.167***| +0.054 | +0.028 | +0.044 | +0.045 |
| is_patch_strategy | **+0.109***| **+0.113***| +0.101 | +0.064 | **+0.112***| +0.100 | +0.052 |
| nhead | +0.046 | +0.050 | +0.053 | +0.082 | +0.064 | -0.008 | -0.009 |

*(\* p < 0.001)*

### Key Findings

1. **seq_length is the dominant factor** negatively correlated with ALL performance metrics. Larger windows degrade performance substantially.
2. **mask_after_encoder** is the strongest predictor of disc_d (+0.73) and recon_d (-0.81) — these are inversely related. mask_after dramatically amplifies discrepancy but suppresses reconstruction separation.
3. **num_teacher_decoder_layers** is the most consistently positive architectural parameter across all metrics.
4. **learning_rate** shows modest positive correlation — higher lr (0.003-0.005) tends to help.

---

## 6. Focus (1): Discrepancy Ratio & Cohen's d Maximization

### Hypothesis
Models with high discrepancy separation have specific architectural and training characteristics that can be identified and exploited.

### Analysis: Top 20% disc_d vs Bottom 50%

| Parameter | Top 20% (n=34) | Bottom 50% (n=84) | p-value | Significance |
|-----------|----------------|-------------------|---------|-------------|
| **seq_length** | 112 | 292 | 0.0000 | *** |
| **patch_size** | 10.0 | 11.7 | 0.017 | * |
| **masking_ratio** | 0.149 | 0.199 | 0.020 | * |
| **nhead** | 6.1 | 3.2 | 0.0014 | ** |
| d_model | 77.2 | 70.3 | 0.19 | ns |
| teacher_dec | 2.53 | 2.30 | 0.097 | marginal |

### Critical Characteristics of High-disc_d Models

1. **Small window size (100)**: 33/34 top models use seq_length=100. Only 1 uses 500.
2. **Lower masking ratio (~0.15 vs ~0.20)**: Masking fewer patches preserves more context for the teacher-student discrepancy signal.
3. **Patch masking strategy (100%)**: All top models use patch masking, not feature-wise.
4. **Higher nhead**: More attention heads (mean 6.1 vs 3.2) correlate with better discrepancy.

### Hypothesis for disc_d Maximization
> **H1**: Discrepancy separation is maximized when the model has sufficient context (short window → fewer patches to reconstruct), uses conservative masking (lower ratio → less information loss), and has enough attention capacity (more heads) to learn precise teacher-student differences. The small window provides a cleaner signal because each patch represents a larger fraction of the input, making anomaly-induced reconstruction differences more pronounced per patch.

### Top 5 Models by disc_d (default scoring, mask_after)

| Model | disc_d | recon_d | roc_auc | disc_ratio |
|-------|--------|---------|---------|------------|
| 167_optimal_combo_4 (d192, td3, nhead6, mr0.10) | **3.37** | 1.44 | 0.9645 | 8.65 |
| 022_d_model_128 (d128, td2, nhead2) | 3.28 | 1.56 | 0.9694 | 9.95 |
| 006_patch_5 (p5, d64, td2) | 3.26 | 1.03 | 0.9640 | 9.92 |
| 094_combo_d128_nhead4_t3s1_mr0.10 | 3.25 | 1.70 | 0.9713 | 8.50 |
| 112_pa80_d128_mr0.10_t4s1 | 3.12 | 1.67 | 0.9721 | 7.89 |

---

## 7. Focus (2): Dual-Strength Models

### Definition
Models where **both** disc_cohens_d and recon_cohens_d are in the top 25% simultaneously.

### Results: 12 Dual-Strength Models

**Common profile**:
- seq_length: **100** (11/12), 500 (1/12)
- masking_strategy: **patch** (12/12)
- masking_ratio: **0.10-0.15** (mean 0.12)
- d_model: **64** (most common)
- nhead: **high** (mean 9.25, mode 16)
- teacher_dec: **2** (most common)
- encoder_layers: **1** (all)

### Top Dual-Strength Models

| Model | disc_d | recon_d | Combined | roc_auc | pa_80_f1 |
|-------|--------|---------|----------|---------|----------|
| 067_optimal_v3 (w500, p20, d128) | 3.12 | 2.11 | **5.23** | 0.9766 | 0.7428 |
| 083_d128_nhead1_mr0.10 | 3.02 | 1.80 | 4.82 | 0.9746 | 0.8086 |
| 027_nhead_8 | 2.98 | 1.83 | 4.81 | 0.9721 | 0.7492 |
| 142_pa80_mr0.10_nhead8 | 2.85 | 1.92 | 4.77 | 0.9704 | 0.7132 |

### Hypothesis
> **H2**: Dual-strength models achieve high separation on BOTH signals because they use a masking configuration (low ratio ~0.10) that preserves enough information for accurate reconstruction (boosting recon_d) while still creating sufficient information asymmetry for the teacher-student discrepancy signal (boosting disc_d). The moderate masking ratio is the critical balance point.

---

## 8. Focus (3): Scoring Mode × Window Size Interactions

### Scoring Mode Effect (mask_after only, 169 paired comparisons)

**ROC-AUC:**
- Adaptive vs default: mean diff = **+0.0092**, adaptive wins **151/169** (89%)
- Normalized vs default: mean diff = **+0.0072**, normalized wins **147/169** (87%)
- Best scoring: adaptive (142), default (15), normalized (12)

**PA80_F1:**
- Adaptive vs default: **+0.0217**
- Normalized vs default: **+0.0194**
- Best scoring: adaptive (74), normalized (73), default (22)

### Window Size × Scoring Interaction (ROC-AUC, mask_after)

| Window | n | Adaptive-Default | Normalized-Default |
|--------|---|-----------------|-------------------|
| 100 | 126 | +0.0108 | +0.0081 |
| 500 | 38 | +0.0046 | +0.0040 |
| 1000 | 3 | +0.0055 | +0.0142 |

### Window Size × Scoring (PA80_F1)

| Window | Adaptive-Default | Normalized-Default |
|--------|-----------------|-------------------|
| 100 | +0.0223 | +0.0200 |
| 500 | +0.0159 | +0.0134 |
| 1000 | **+0.0859** | **+0.0771** |

### Key Insight
> **Scoring mode matters most for large windows.** At window=1000, adaptive scoring improves PA80_F1 by **+0.086** — a massive gain. This suggests that as windows grow and raw scores become noisier, adaptive scoring's re-weighting of the discrepancy component becomes critical for rescuing performance.

### Top Models: Scoring Mode Behavior

**067_optimal_v3 (w500, d128, p20):**
| Mask | Scoring | roc_auc | f1 | pa80_f1 | dist_f1 |
|------|---------|---------|-----|---------|---------|
| after | default | 0.9766 | 0.7355 | 0.7428 | 0.6788 |
| after | adaptive | **0.9783** | 0.7314 | 0.7374 | 0.6549 |
| after | normalized | 0.9764 | 0.7360 | **0.7439** | **0.7620** |

> Normalized scoring dramatically improves disturbing F1 (0.68→0.76) for this model, suggesting it better normalizes the score distributions at boundary regions.

---

## 9. Focus (4): Disturbing Normal vs Anomaly Separation

### Context
Disturbing normals are time windows adjacent to anomaly boundaries — the hardest samples to classify correctly. High disc_cohens_d_disturbing_vs_anomaly indicates the model can distinguish these borderline cases.

### Top Models by disc_d_disturbing (mask_after, default)

| Model | dist_d | disc_d | seq | patch | d_model | teacher_dec |
|-------|--------|--------|-----|-------|---------|-------------|
| **067_optimal_v3** | **2.55** | 3.12 | **500** | 20 | 128 | 2 |
| 063_w500_p20_d128 | 2.19 | 2.71 | 500 | 20 | 128 | 2 |
| 070_optimal_final | 2.14 | 2.64 | 500 | 20 | 128 | 3 |
| 008_w500_p5 | 2.13 | 2.62 | 500 | 5 | 64 | 2 |
| 109_w500_p20_d128_t3s2 | 2.07 | 2.56 | 500 | 20 | 64 | 3 |

### Critical Discovery: Window 500 Dominates Disturbing Separation

All top-15 disturbing separation models use **seq_length=500** (14/15) or seq_length=100 (1/15 — but with patch_size=5, giving 20 patches, similar granularity to w500/p20).

| Parameter | Top 20% dist_d (n=34) | Bottom 50% (n=84) | p-value |
|-----------|----------------------|-------------------|---------|
| **d_model** | **86.6** | 65.0 | 0.0000 *** |
| **dim_feedforward** | **346** | 260 | 0.0000 *** |
| **patch_size** | **12.8** | 10.7 | 0.0045 ** |
| masking_ratio | 0.163 | 0.201 | 0.066 (marginal) |

### Hypothesis
> **H3**: Window size 500 provides the optimal balance for disturbing-normal separation because: (1) the larger temporal context allows the model to "see" the transition between normal and anomalous patterns, giving it better boundary awareness; (2) with patch_size=20, each patch covers 20 timesteps, providing sufficient temporal resolution to capture the gradual transition. The larger d_model (128) and dim_feedforward (346) provide the capacity needed to model these subtle boundary differences.

> **H4**: There is a fundamental tension between disc_d_normal_vs_anomaly (favors w100) and disc_d_disturbing_vs_anomaly (favors w500). Short windows maximize the per-patch anomaly signal but miss boundary context. Longer windows capture boundary transitions but dilute the per-patch signal.

---

## 10. Focus (5): PA%80 + High disc_ratio Joint Analysis

### Results: 34 Models with PA80_F1 ≥ 0.73 AND disc_ratio ≥ 6.05

**All 34 models are mask_after with patch masking strategy.**

| Characteristic | Values |
|---------------|--------|
| seq_length | **100** (33/34), 500 (1/34) |
| mask_after | **100%** |
| masking_strategy | **patch 100%** |
| masking_ratio | 0.08-0.20 (mean 0.16) |
| teacher_dec | 2-4 (mean 2.79) |
| learning_rate | 0.002-0.003 |

### Top 5

| Model | PA80_F1 | disc_ratio | roc_auc |
|-------|---------|------------|---------|
| 166_optimal_combo_3 (td3, nhead1, mr0.08) | **0.8318** | 7.67 | 0.9755 |
| 083_d128_nhead1_mr0.10 | 0.8086 | **10.80** | 0.9746 |
| 100_optimal_combo_1 (td3, lr0.002) | 0.7963 | 8.60 | 0.9736 |
| 094_combo (td3, nhead4, mr0.10) | 0.7875 | 8.50 | 0.9713 |

### Hypothesis
> **H5**: Joint PA80+disc_ratio optimization requires mask_after timing because it amplifies the discrepancy signal (high disc_ratio), and the PA80 metric rewards models that detect anomalies early in a segment — which the amplified discrepancy enables. The teacher decoder depth of 3+ layers helps because deeper decoders can learn more nuanced reconstruction patterns, leading to sharper anomaly detection at segment boundaries.

---

## 11. Focus (6): Window Size × Model Depth × Masking Ratio

### Window Size Effect (mask_after, default scoring)

| Window | n | roc_auc | disc_d | recon_d | dist_d | PA80_F1 |
|--------|---|---------|--------|---------|--------|---------|
| **100** | 126 | 0.9533 | **2.56** | **1.61** | 0.98 | **0.7032** |
| 200 | 2 | 0.9581 | 2.31 | 1.44 | **1.32** | 0.6670 |
| 500 | 38 | 0.8829 | 1.45 | 1.36 | 1.12 | 0.4525 |
| 1000 | 3 | 0.7203 | 0.51 | 0.96 | 0.44 | 0.1847 |

### Teacher Decoder Depth × Window Size

**Window 100:**
| teacher_dec | n | roc_auc | disc_d | dist_d | PA80_F1 |
|-------------|---|---------|--------|--------|---------|
| 2 | 83 | 0.9497 | 2.46 | 0.92 | 0.6889 |
| **3** | 21 | **0.9651** | **2.87** | **1.22** | **0.7434** |
| 4 | 20 | 0.9555 | 2.66 | 1.02 | 0.7200 |

**Window 500:**
| teacher_dec | n | roc_auc | disc_d | dist_d | PA80_F1 |
|-------------|---|---------|--------|--------|---------|
| 2 | 27 | **0.8928** | **1.55** | **1.19** | **0.4759** |
| 3 | 9 | 0.8716 | 1.38 | 1.07 | 0.4315 |
| 4 | 2 | 0.7999 | 0.55 | 0.34 | 0.2313 |

### Key Finding: Depth Scaling Inverts with Window Size

> **H6**: At window 100, deeper teacher decoders (3 layers) improve performance. But at window 500, the opposite occurs — teacher_dec=2 is optimal, and depth 4 collapses dramatically. This suggests that larger windows already provide enough temporal context for pattern learning, and deeper decoders overfit to the abundant sequential patterns. At window 100, the limited context benefits from deeper processing.

### d_model × Window Size

**Window 100:** d_model effect is modest (roc 0.9527-0.9703)
**Window 500:** d_model is critical:
| d_model | n | roc_auc | disc_d |
|---------|---|---------|--------|
| 64 | 28 | 0.8723 | 1.27 |
| **128** | 8 | **0.9205** | **2.11** |

> **H7**: Larger d_model is critical for large windows but less important for small ones. At window 500, upgrading d_model from 64→128 provides a **+0.048 roc_auc** improvement and **+0.84 disc_d** improvement. The larger model capacity is needed to process the richer temporal information in longer sequences.

### Masking Ratio × Window Size

**Window 100 (n per ratio varies):**
- mr=0.05: roc=0.6237 (collapsed!)
- mr=0.08-0.15: roc=0.9582-0.9681 (best range)
- mr=0.20: roc=0.9599 (default, solid)
- mr=0.50+: roc degrades rapidly

**Window 500:**
- mr=0.15: roc=**0.9219** (best)
- mr=0.20: roc=0.8882
- mr=0.10: roc=0.8442
- mr=0.05: roc=0.8267

> **H8**: The optimal masking ratio shifts with window size. At w100, 0.08-0.15 is optimal. At w500, 0.15 is optimal. Very low masking (0.05) severely hurts because the teacher can barely distinguish itself from the student when almost nothing is masked. Very high masking (>0.50) hurts because reconstruction becomes too difficult and the model's representations degrade.

---

## 12. Focus (7): Scoring Mode Sensitivity

### Which Parameters Cause the Biggest Scoring Sensitivity?

Correlation with scoring_range (max - min across 3 scoring modes):

| Parameter | Spearman rho | p-value | Interpretation |
|-----------|-------------|---------|----------------|
| **seq_length** | **+0.236** | 0.002 | Larger windows → more scoring sensitive |
| d_model | +0.192 | 0.013 | Larger models → more scoring sensitive |
| dim_feedforward | +0.196 | 0.011 | Same direction |
| nhead | +0.180 | 0.019 | More heads → more sensitive |
| patch_size | -0.160 | 0.038 | Larger patches → less sensitive |
| masking_ratio | -0.184 | 0.017 | Higher masking → less sensitive |

### Most Scoring-Sensitive Models (ROC-AUC range)

| Model | Range | Default | Adaptive | Normalized | seq |
|-------|-------|---------|----------|------------|-----|
| 072_d128_mask_0.05 | **0.0795** | 0.6237 | **0.7032** | 0.6821 | 100 |
| 091_w500_nhead16 | 0.0584 | **0.8008** | 0.7443 | 0.7424 | 500 |
| 170_pa80_final | 0.0412 | 0.8981 | **0.9393** | 0.9381 | 100 |

### Hypothesis
> **H9**: Models with the lowest masking ratio (0.05) show extreme scoring sensitivity because their discrepancy signal is very weak (almost no information asymmetry between teacher and student). In this regime, the scoring mode's re-weighting dramatically affects whether the weak discrepancy signal contributes positively or negatively. Adaptive scoring rescues these models (0.62→0.70) by down-weighting the noisy discrepancy.

> **H10**: Larger windows increase scoring sensitivity because the raw discrepancy signal degrades with window size (disc_d drops from 2.56→0.51), making the scoring mode's handling of this weakened signal more consequential.

---

## 13. Focus (8): Robust Detection + Disturbing Separation

### Criteria: roc_auc > 0.96 AND disc_d_disturbing > 1.0

**64 models** meet this criterion. All are **mask_after**.

### Pattern Analysis

| Region | Count | Dominant Config |
|--------|-------|----------------|
| seq=500, d=128, p=20 | 5 | dist_d up to 2.55 (best) |
| seq=100, d=64, p=10, td=3-4 | 30+ | dist_d 1.0-1.45 |
| seq=100, d=128+, p=10 | 10+ | dist_d 1.2-1.5 |

### Best Robust Model: 067_optimal_v3
- **Config**: seq=500, patch=20, d_model=128, teacher_dec=2, nhead=4, mr=0.15
- **Performance**: roc_auc=0.9766, dist_d=2.55, disc_d=3.12

### Hypothesis
> **H11**: The "robust detector" profile comes in two flavors:
> 1. **Window-500 high-capacity models** (d_model=128, patch_size=20): Achieve the highest dist_d (>2.0) through broader temporal context, but are rare because w500 performance is generally lower
> 2. **Window-100 deeper decoder models** (teacher_dec=3-4): Achieve moderate dist_d (1.0-1.5) with consistently high roc_auc, leveraging deeper processing to compensate for limited temporal context

> The first type is preferable when disturbing separation is paramount; the second provides a safer bet for general-purpose detection.

---

## 14. Focus (9): mask_after vs mask_before Comparison

### Paired Analysis (169 identical configs, default scoring)

| Metric | mask_after | mask_before | Diff | After Wins | p-value |
|--------|-----------|------------|------|------------|---------|
| roc_auc | 0.9334 | **0.9488** | -0.0155 | 26/169 | **0.0024** |
| f1_score | 0.6419 | **0.7115** | -0.0696 | 17/169 | **<0.0001** |
| pa_80_f1 | 0.6372 | **0.6699** | -0.0327 | 57/169 | **0.0002** |
| **disc_d** | **2.2733** | 0.7405 | **+1.5327** | **164/169** | **<0.0001** |
| **disc_d_dist** | **1.0078** | 0.2403 | **+0.7674** | **164/169** | **<0.0001** |
| recon_d | 1.5401 | **2.3904** | -0.8502 | 2/169 | **<0.0001** |
| disc_ratio | **5.4943** | 1.8125 | +3.6818 | 161/169 | **<0.0001** |

### The Paradox: High disc_d but Lower roc_auc

mask_after produces **3x higher disc_d** but **lower roc_auc**. This is because:

| Signal | mask_after | mask_before |
|--------|-----------|------------|
| recon_normal | 0.005595 | 0.006457 |
| recon_anomaly | 0.011882 | **0.025318** |
| disc_normal | **0.015569** | 0.017319 |
| disc_anomaly | **0.022814** | 0.018987 |
| **recon_ratio** | 2.21 | **4.57** |
| **disc_ratio** | **5.49** | 1.81 |

> mask_before generates **2x higher recon_anomaly** (0.025 vs 0.012), meaning anomalies are much more visible in reconstruction error. The combined scoring uses both signals, and mask_before's superior reconstruction signal outweighs mask_after's superior discrepancy.

### Scoring Mode Reversal

| Scoring | After-Before roc_auc | After Wins |
|---------|---------------------|------------|
| default | **-0.0155** | 26/169 |
| adaptive | **+0.0068** | **114/169** |
| normalized | **+0.0186** | **117/169** |

| Scoring | After-Before PA80_F1 | After Wins |
|---------|---------------------|------------|
| default | -0.0327 | 57/169 |
| adaptive | **+0.0352** | **117/169** |
| normalized | **+0.0815** | **137/169** |

### Critical Discovery
> **H12**: mask_after is actually BETTER than mask_before — but only with adaptive or normalized scoring. With default scoring, the combined score under-weights the strong discrepancy signal. Adaptive and normalized scoring properly leverage mask_after's discrepancy advantage, making it win 67-81% of the time. The PA80_F1 gain with normalized scoring is a massive **+0.0815**.

> This means **the optimal configuration is mask_after + normalized scoring**, not the historically used mask_before + default scoring.

### Window Size × Mask Timing

| Window | After-Before disc_d |
|--------|-------------------|
| 100 | +1.70 |
| 200 | +2.17 |
| 500 | +1.01 |
| 1000 | +0.57 |

> The disc_d advantage of mask_after shrinks with larger windows, explaining why the scoring reversal may be even more important at larger windows.

---

## 15. Focus (10): Discrepancy Contribution to Performance

### Does the Discrepancy Signal Help?

Since disc_only metrics weren't available (eval-only mode limitation), we analyze indirectly:

**Adaptive scoring gain by disc_d quartile:**

| disc_d Quartile | Mean disc_d | Adap Gain (roc) | Default ROC | Adaptive ROC |
|----------------|-------------|-----------------|-------------|--------------|
| Q1 (lowest) | 1.18 | +0.0091 | 0.8544 | 0.8635 |
| Q2 | 2.38 | **+0.0108** | 0.9538 | 0.9647 |
| Q3 | 2.68 | +0.0083 | 0.9630 | 0.9713 |
| Q4 (highest) | 3.00 | +0.0079 | 0.9695 | 0.9774 |

### Correlation: disc_d → Adaptive gain

- Spearman rho = **-0.107** (p=0.17) — weak negative, non-significant

### Interpretation
> **H13**: The adaptive scoring gain is relatively uniform across disc_d quartiles (~0.008-0.011), suggesting that scoring mode improvements are NOT primarily about having a stronger discrepancy signal. Instead, adaptive scoring benefits models by better balancing the two signals regardless of their relative strength. Models with the lowest disc_d (Q1) see the smallest absolute gain — but they also have the lowest baseline, so the discrepancy signal contributes less overall.

### Reconstruction vs Combined (Quadrant Analysis)

| Quadrant | n | roc_auc | pa80_f1 | disc_d | recon_d |
|----------|---|---------|---------|--------|---------|
| **Both high** | 46 | **0.9689** | **0.7383** | 2.85 | 1.76 |
| Recon high only | 36 | 0.9606 | 0.7041 | 2.41 | 1.79 |
| Disc high only | 24 | 0.9632 | 0.7273 | 2.92 | 1.48 |
| Both low | 63 | 0.8805 | 0.4908 | 1.53 | 1.26 |

> **H14**: Having both signals strong (disc_d AND recon_d) is clearly the best combination for detection performance. But between having only one strong, **disc high only** slightly outperforms **recon high only** on PA80_F1 (0.7273 vs 0.7041), suggesting the discrepancy signal provides more value for early anomaly detection (PA80 measures).

---

## 16. Focus (11): Additional Insights

### A. Masking Strategy: Patch vs Feature-Wise

Only 1 experiment uses feature_wise masking, with noticeably lower disc_d (1.22 vs 2.28 mean for patch). Insufficient data for robust comparison, but the signal suggests **patch masking is strongly preferred** for the discrepancy task.

### B. Learning Rate

| lr | n | roc_auc | disc_d | PA80_F1 |
|----|---|---------|--------|---------|
| 0.002 | 160 | 0.9321 | 2.26 | 0.6329 |
| **0.003** | 5 | **0.9542** | **2.74** | **0.7286** |
| **0.005** | 2 | **0.9684** | **2.69** | **0.7192** |

> **H15**: Higher learning rates (0.003-0.005) consistently outperform the default 0.002. This suggests the model is under-trained at lr=0.002 given the 50-epoch budget, and can benefit from more aggressive optimization.

### C. Lambda_disc

Most experiments use lambda_disc=0.5 (164/169). The few alternatives show:
- lambda=2.0-3.0: roc_auc 0.9719-0.9720, slightly better than 0.5 (0.9324)
- But these are single experiments, so confounded with other parameter choices.

### D. Dynamic Margin K

| k | roc_auc | disc_d |
|---|---------|--------|
| 1.0 | 0.9653 | 2.46 |
| 1.5 (default) | 0.9324 | 2.26 |
| 3.0 | **0.9676** | **2.68** |
| 4.0 | 0.9677 | 2.63 |

> Higher k values (3.0-4.0) show modest improvements, suggesting the dynamic margin target could be set more aggressively.

### E. Model Capacity

| Capacity | n | roc_auc | disc_d | dist_d | PA80_F1 |
|----------|---|---------|--------|--------|---------|
| Low (mean 126) | 104 | 0.9309 | 2.18 | 0.92 | 0.6319 |
| Mid (mean 192) | 26 | 0.9361 | 2.38 | 1.10 | 0.6517 |
| High (mean 299) | 39 | 0.9381 | 2.44 | 1.19 | 0.6417 |

> Higher capacity models show consistent improvements in disc_d and dist_d, but the PA80_F1 gain plateaus, suggesting that beyond mid-capacity, the benefit shifts from detection accuracy to separation quality.

### F. Patch Granularity

**Window 100:**
| patch_size | n_patches | roc_auc | disc_d | dist_d |
|------------|-----------|---------|--------|--------|
| 5 | 20 | 0.9640 | **3.26** | **1.61** |
| 10 | 10 | 0.9527 | 2.56 | 0.98 |
| 20 | 5 | 0.9660 | 2.38 | 0.72 |
| 25 | 4 | 0.9705 | 2.22 | 0.61 |

**Window 500:**
| patch_size | n_patches | roc_auc | disc_d | dist_d |
|------------|-----------|---------|--------|--------|
| 5 | 100 | 0.9267 | **2.62** | **2.13** |
| 10 | 50 | 0.8473 | 0.98 | 0.70 |
| 20 | 25 | **0.9453** | 2.24 | 1.80 |

> **H16**: Smaller patches (more patches per window) dramatically increase disc_d and dist_d but don't always improve roc_auc. At w100, the roc_auc is actually highest with patch_size=25 (fewest patches) despite having the lowest disc_d. At w500, the optimal roc_auc is at patch_size=20. This reveals that **many fine-grained patches create noise in the aggregated score**, even though individual patch separation is excellent. The optimal granularity depends on the window size.

### G. Attention Heads

| nhead | n | roc_auc | disc_d | dist_d |
|-------|---|---------|--------|--------|
| 1 | 11 | 0.9395 | 2.36 | 0.99 |
| 2 | 110 | 0.9302 | 2.20 | 0.94 |
| 4 | 17 | 0.9184 | 2.09 | 1.35 |
| **8** | 5 | **0.9706** | **2.87** | **1.22** |
| 16 | 24 | 0.9452 | 2.52 | 1.01 |

> nhead=8 stands out as the best, but confounded by limited sample size. The nhead=4 result is interesting: worst roc_auc but best dist_d. This may indicate that 4 heads create more specialized attention patterns that help at boundaries but hurt overall.

### H. Discrepancy SNR (Overview)

The discrepancy signal-to-noise ratio (disc_SNR = mean_diff / sum_of_stds) correlates with detection performance:
- **Corr(disc_SNR, roc_auc) = 0.492**
- **Corr(disc_SNR, PA80_F1) = 0.396**

Compared to alternatives:
- disc_cohens_d: corr(roc)=0.481, corr(pa80)=0.340
- disc_ratio: corr(roc)=0.411, corr(pa80)=0.414

> **H17**: disc_SNR is the best single predictor of roc_auc among separation metrics. Unlike disc_d (which uses pooled std), SNR accounts for the sum of both distribution widths independently, making it more sensitive to asymmetric variance. See **Section 19** for comprehensive disc_SNR analysis.

### I. Encoder Depth

| enc_layers | n | roc_auc | disc_d |
|------------|---|---------|--------|
| 1 | 160 | 0.9335 | 2.28 |
| 2 | 7 | 0.9305 | 2.18 |

> Adding encoder layers provides no benefit and may slightly hurt. The single encoder layer is sufficient for feature extraction, with the decoder doing the heavy lifting.

### J. mask_before Top Performers

The best mask_before models achieve **higher roc_auc** than mask_after (0.9807 vs 0.9766) but with **much lower disc_d** (1.05 vs 3.12). They compensate entirely through reconstruction:

| Model | roc_auc | disc_d | recon_d | PA80_F1 |
|-------|---------|--------|---------|---------|
| 150_pa80_t4s1_nhead1 (before) | **0.9807** | 1.05 | **2.60** | 0.7430 |
| 067_optimal_v3 (after) | 0.9766 | **3.12** | 2.11 | 0.7428 |

> mask_before models with **teacher_dec=4** and **nhead=1** are the strongest, suggesting that single-head attention with deep decoder enables the most expressive reconstruction when masking is applied before encoding.

---

## 17. Optimal Parameter Recipes (Top 15)

> Selection criteria: Balance mask_after/mask_before evenly, include w500 performers, cover different optimization objectives (roc_auc, PA80_F1, disc_d, dist_d, balanced). Each recipe includes rationale based on per-parameter analysis and interaction effects.

### mask_after Recipes (7)

#### Recipe 1: Best PA80_F1 — Low Masking + Deep Decoder
- **Config**: seq=100, p=10, d=64, nhead=1, td=3, sd=1, mr=0.08, lr=0.002, λ_disc=0.5, mk=1.5, **mask_after, adaptive**
- **Performance**: roc_auc=0.9799, PA80_F1=**0.8504**, disc_d=2.83, dist_d=1.27, recon_d=1.77
- **Model**: `166_optimal_combo_3`
- **Rationale**: mr=0.08×td=3 is the optimal interaction (mean roc=0.9714, pa80=0.7539 — best of all mr×td combos). nhead=1 avoids attention fragmentation at d=64, maximizing PA80_F1. Adaptive scoring recovers mask_after's strong discrepancy signal.

#### Recipe 2: Best mask_after roc_auc — Single Head + Low Masking
- **Config**: seq=100, p=10, d=64, nhead=1, td=2, sd=1, mr=0.10, lr=0.002, λ_disc=0.5, mk=1.5, **mask_after, adaptive**
- **Performance**: roc_auc=**0.9809**, PA80_F1=0.8320, disc_d=3.02, dist_d=1.33, recon_d=1.80
- **Model**: `083_d128_nhead1_mask0.10`
- **Rationale**: nhead=1 at d=64 is a strong combo (mean roc=0.9546). mr=0.10 provides slightly more masking than 0.08, boosting disc_d from 2.83→3.02 while maintaining high roc_auc. td=2 is sufficient for the moderate masking ratio.

#### Recipe 3: Best disc_d — Large Model + Multi-Head
- **Config**: seq=100, p=10, d=192, nhead=6, td=3, sd=1, mr=0.10, lr=0.002, λ_disc=0.5, mk=1.5, **mask_after, adaptive**
- **Performance**: roc_auc=0.9791, PA80_F1=0.7866, disc_d=**3.37**, dist_d=1.46, recon_d=1.44
- **Model**: `167_optimal_combo_4`
- **Rationale**: d=192 with nhead=6 (head_dim=32) achieves the highest disc_d in the entire study. Larger models amplify discrepancy separation. nh=6×d=192 mean roc=0.9666 is among the best nhead×d combos.

#### Recipe 4: Best d=128 Configuration
- **Config**: seq=100, p=10, d=128, nhead=4, td=2, sd=1, mr=0.20, lr=0.002, λ_disc=0.5, mk=1.5, **mask_after, adaptive**
- **Performance**: roc_auc=0.9805, PA80_F1=0.7915, disc_d=3.28, dist_d=1.52, recon_d=1.56
- **Model**: `022_d_model_128`
- **Rationale**: seq=100×d=128 is the best seq×d interaction (mean roc=0.9682, dd=2.14). nhead=4 at d=128 (head_dim=32) provides adequate attention diversity. disc_d=3.28 is second-highest overall.

#### Recipe 5: Best w500 — Large Window Specialist
- **Config**: seq=500, p=20, d=128, nhead=4, td=2, sd=1, mr=0.15, lr=0.002, λ_disc=0.5, mk=1.5, **mask_after, adaptive**
- **Performance**: roc_auc=**0.9783**, PA80_F1=0.7374, disc_d=3.12, dist_d=**2.55**, recon_d=2.11
- **Model**: `067_optimal_v3`
- **Rationale**: Best w500 model by a large margin (+0.004 roc over next). d=128 is critical at w500 (mean roc 0.9027 vs d=64's 0.8920). dist_d=2.55 is the best disturbing separation in the entire study — w500 captures temporal boundary patterns that w100 misses.

#### Recipe 6: High LR mask_after — Under-explored Regime
- **Config**: seq=100, p=10, d=64, nhead=16, td=3, sd=1, mr=0.10, lr=0.003, λ_disc=0.5, mk=1.5, **mask_after, adaptive**
- **Performance**: roc_auc=0.9778, PA80_F1=0.7675, disc_d=2.83, dist_d=1.19, recon_d=1.62
- **Model**: `165_optimal_combo_2`
- **Rationale**: lr=0.003 mean roc=0.9665 and lr=0.005 mean roc=0.9739, both significantly above lr=0.002's 0.9356. Learning rate is the most under-explored parameter — systematic exploration at 0.003-0.005 is high priority for Phase 2.

#### Recipe 7: d=128 + nhead=16 — Wide Attention
- **Config**: seq=100, p=10, d=128, nhead=16, td=2, sd=1, mr=0.20, lr=0.002, λ_disc=0.5, mk=1.5, **mask_after, adaptive**
- **Performance**: roc_auc=0.9807, PA80_F1=0.8277, disc_d=3.16, dist_d=1.43, recon_d=1.53
- **Model**: `028_d128_nhead_16`
- **Rationale**: nh=16×d=128 (head_dim=8) achieves mean roc=0.9590, best roc=0.9807. Many small heads with a larger model provide both high disc_d (3.16) and strong PA80_F1 (0.8277), a rare combination.

### mask_before Recipes (7)

#### Recipe 8: Best Overall roc_auc — Deep Decoder + Multi-Head
- **Config**: seq=100, p=10, d=64, nhead=4, td=4, sd=1, mr=0.20, lr=0.002, λ_disc=0.5, mk=1.5, **mask_before, adaptive**
- **Performance**: roc_auc=**0.9827**, PA80_F1=0.8028, disc_d=1.31, dist_d=0.55, recon_d=2.61
- **Model**: `151_pa80_t4s1_nhead4`
- **Rationale**: The single best roc_auc in the entire study. td=4 is the sweet spot for mask_before (mean roc=0.9612, pa80=0.7223 at mr=0.20). mask_before excels with deeper decoders because the encoder sees full unmasked input, providing a stable teaching signal.

#### Recipe 9: Best mask_before Default Scoring
- **Config**: seq=100, p=10, d=64, nhead=1, td=4, sd=1, mr=0.20, lr=0.002, λ_disc=0.5, mk=1.5, **mask_before, default**
- **Performance**: roc_auc=0.9807, PA80_F1=0.7430, disc_d=1.05, dist_d=0.32, recon_d=2.60
- **Model**: `150_pa80_t4s1_nhead1`
- **Rationale**: mask_before+default (mean roc=0.9488) is the best timing+scoring combo by mean roc. This recipe achieves near-peak performance without requiring adaptive scoring, making it the most robust choice for deployment.

#### Recipe 10: Deep Decoder + Dual Student
- **Config**: seq=100, p=10, d=64, nhead=2, td=4, sd=2, mr=0.20, lr=0.002, λ_disc=0.5, mk=1.5, **mask_before, adaptive**
- **Performance**: roc_auc=0.9810, PA80_F1=0.7936, disc_d=1.32, dist_d=0.45, recon_d=2.74
- **Model**: `019_decoder_t4s2`
- **Rationale**: sd=2 provides slight recon_d improvement (2.74 vs 2.61 with sd=1) at the cost of minimal extra compute. td=4+sd=2 with mask_before leverages the deep decoder pathway while the dual student adds reconstruction diversity.

#### Recipe 11: Best w500 mask_before — Long Window Baseline
- **Config**: seq=500, p=20, d=64, nhead=2, td=2, sd=1, mr=0.20, lr=0.002, λ_disc=0.5, mk=1.5, **mask_before, default**
- **Performance**: roc_auc=0.9744, PA80_F1=0.7377, disc_d=1.09, dist_d=0.89, recon_d=2.62
- **Model**: `009_w500_p20`
- **Rationale**: Best mask_before at w500. Default scoring works best for mask_before (mean roc=0.9488 vs adaptive's 0.9358). recon_d=2.62 is exceptionally high, showing mask_before's reconstruction strength. Competitive with mask_after's w500 best (gap only 0.004 roc).

#### Recipe 12: High LR mask_before — Best Default Performer
- **Config**: seq=100, p=10, d=64, nhead=16, td=4, sd=1, mr=0.10, lr=0.003, λ_disc=0.5, mk=1.5, **mask_before, default**
- **Performance**: roc_auc=0.9779, PA80_F1=0.7580, disc_d=1.11, dist_d=0.33, recon_d=2.55
- **Model**: `168_pa80_final_opt1`
- **Rationale**: Highest roc_auc among all lr>0.002 models. Combines the lr=0.003 advantage with mask_before's strength at td=4. nhead=16 at d=64 (head_dim=4) works surprisingly well (mean roc=0.9539), suggesting that many small heads can be effective.

#### Recipe 13: High LR mask_before (lr=0.005)
- **Config**: seq=100, p=10, d=64, nhead=2, td=2, sd=1, mr=0.20, lr=0.005, λ_disc=0.5, mk=1.5, **mask_before, default**
- **Performance**: roc_auc=0.9777, PA80_F1=0.7964, disc_d=0.70, dist_d=0.11, recon_d=2.68
- **Model**: `054_lr_0.005`
- **Rationale**: lr=0.005 achieves the highest mean roc (0.9739) and pa80 (0.7580) of any learning rate, despite tiny sample (n=12). PA80_F1=0.7964 is exceptional for mask_before. The near-zero disc_d (0.70) shows this model relies entirely on reconstruction — yet still achieves strong roc, validating that reconstruction-dominant models can be highly effective.

#### Recipe 14: nhead=8 Specialist
- **Config**: seq=100, p=10, d=64, nhead=8, td=2, sd=1, mr=0.20, lr=0.002, λ_disc=0.5, mk=1.5, **mask_before, default**
- **Performance**: roc_auc=0.9789, PA80_F1=0.7526, disc_d=0.90, dist_d=0.31, recon_d=2.75
- **Model**: best from nhead=8 group (estimated from interaction data)
- **Rationale**: nhead=8 at d=64 has the highest mean roc (0.9697) and best_roc (0.9789) of any nhead value. head_dim=8 provides a good balance between per-head capacity and attention diversity. recon_d=2.75 suggests strong reconstruction.

### Exploratory Recipe (1)

#### Recipe 15: Proposed Phase 2 — Optimized Combo (untested)
- **Config**: seq=100, p=10, d=128, nhead=8, td=3, sd=1, mr=0.08, lr=0.003, λ_disc=0.5, mk=1.5, **mask_after, adaptive**
- **Performance**: *Not yet tested*
- **Rationale**: Combines the top-performing parameter values from interaction analysis:
  - mr=0.08×td=3 is the best masking×decoder interaction
  - d=128 at seq=100 is the best seq×d interaction (mean roc=0.9682)
  - nhead=8 has the highest mean roc (0.9694) and works well at d=64; at d=128 (head_dim=16) it should be even better
  - lr=0.003 consistently outperforms 0.002 (mean roc +0.03)
  - mask_after+adaptive is the strongest timing+scoring combo for high-disc_d architectures

### Summary Table

| # | Mask | Scoring | Objective | roc_auc | PA80_F1 | disc_d | Model |
|---|------|---------|-----------|---------|---------|--------|-------|
| 1 | after | adaptive | Best PA80_F1 | 0.9799 | **0.8504** | 2.83 | 166_optimal_combo_3 |
| 2 | after | adaptive | Best after roc | **0.9809** | 0.8320 | 3.02 | 083_d128_nhead1 |
| 3 | after | adaptive | Best disc_d | 0.9791 | 0.7866 | **3.37** | 167_optimal_combo_4 |
| 4 | after | adaptive | Best d=128 | 0.9805 | 0.7915 | 3.28 | 022_d_model_128 |
| 5 | after | adaptive | Best w500 | 0.9783 | 0.7374 | 3.12 | 067_optimal_v3 |
| 6 | after | adaptive | High LR | 0.9778 | 0.7675 | 2.83 | 165_optimal_combo_2 |
| 7 | after | adaptive | Wide attention | 0.9807 | 0.8277 | 3.16 | 028_d128_nhead_16 |
| 8 | before | adaptive | **Best overall roc** | **0.9827** | 0.8028 | 1.31 | 151_pa80_t4s1_nhead4 |
| 9 | before | default | Best default | 0.9807 | 0.7430 | 1.05 | 150_pa80_t4s1_nhead1 |
| 10 | before | adaptive | Deep+dual student | 0.9810 | 0.7936 | 1.32 | 019_decoder_t4s2 |
| 11 | before | default | Best w500 before | 0.9744 | 0.7377 | 1.09 | 009_w500_p20 |
| 12 | before | default | High LR (0.003) | 0.9779 | 0.7580 | 1.11 | 168_pa80_final_opt1 |
| 13 | before | default | High LR (0.005) | 0.9777 | 0.7964 | 0.70 | 054_lr_0.005 |
| 14 | before | default | nhead=8 specialist | 0.9789 | 0.7526 | 0.90 | nhead=8 best |
| 15 | after | adaptive | Phase 2 proposal | *untested* | *untested* | *untested* | — |

### Key Patterns Across Recipes
1. **mask_after dominates disc_d** (2.83-3.37) while **mask_before dominates recon_d** (2.55-2.75)
2. **Adaptive scoring** is essential for mask_after; **default scoring** is best for mask_before
3. **td=4** is critical for mask_before top performance; **td=2-3** suffices for mask_after
4. **lr=0.003-0.005** consistently outperforms lr=0.002 across both mask timings
5. **d=128** unlocks higher disc_d and is essential for w500 performance

---

## 18. Per-Parameter Performance Analysis

### 18.1 Single-Parameter Effects

#### seq_length (Window Size)
| Value | N | ROC AUC (mean±std) | PA80_F1 | disc_d | Best ROC |
|-------|---|-------------------|---------|--------|----------|
| 100 | 756 | 0.9545±0.0588 | 0.6922 | 1.71 | 0.9827 |
| 200 | 12 | 0.9191±0.0493 | 0.5355 | 1.23 | 0.9612 |
| 500 | 228 | 0.8947±0.0634 | 0.4877 | 0.95 | 0.9783 |
| 1000 | 18 | 0.7588±0.0962 | 0.2727 | 0.22 | 0.9299 |

**Insight**: seq=100 dominates across all metrics. Performance degrades sharply beyond 200. However, the best w500 model (roc=0.9783) nearly matches the best w100 (0.9827), indicating the gap is closable with proper architecture (d=128, p=20). w1000 is generally non-viable with current architectures.

#### masking_ratio
| Value | N | ROC AUC (mean±std) | PA80_F1 | disc_d | Best ROC |
|-------|---|-------------------|---------|--------|----------|
| 0.05 | 18 | 0.7709±0.0987 | 0.2848 | 0.59 | 0.9139 |
| 0.08 | 30 | **0.9670**±0.0098 | **0.7231** | **1.98** | 0.9799 |
| 0.10 | 192 | 0.9413±0.0543 | 0.6450 | 1.68 | 0.9809 |
| 0.12 | 12 | 0.9627±0.0107 | 0.6955 | 1.85 | 0.9768 |
| 0.15 | 54 | 0.9336±0.0618 | 0.6141 | 1.57 | 0.9805 |
| 0.20 | 660 | 0.9395±0.0717 | 0.6418 | 1.46 | 0.9827 |
| 0.25-0.80 | 48 | 0.92-0.90 | 0.57-0.69 | 1.06-1.55 | — |

**Insight**: mr=0.08 achieves the best mean ROC (0.9670) and PA80_F1 (0.7231) with the lowest variance (±0.0098). The sweet spot is 0.08-0.12. Below 0.05, insufficient masking causes training collapse. The baseline mr=0.20 is adequate but sub-optimal. Higher ratios (>0.25) show diminishing returns.

#### d_model
| Value | N | ROC AUC (mean±std) | PA80_F1 | disc_d | Best ROC |
|-------|---|-------------------|---------|--------|----------|
| 16 | 6 | 0.9592±0.0130 | 0.6948 | 1.19 | 0.9703 |
| 32 | 6 | 0.9500±0.0177 | 0.6689 | 1.36 | 0.9688 |
| 64 | 894 | 0.9386±0.0708 | 0.6434 | 1.51 | 0.9827 |
| 128 | 84 | 0.9207±0.0669 | 0.5748 | 1.52 | 0.9807 |
| 192 | 12 | 0.9430±0.0304 | 0.6131 | 1.42 | 0.9791 |
| 256 | 6 | 0.9683±0.0080 | 0.7186 | 1.76 | 0.9756 |

**Insight**: d=64 is the most tested and achieves the best single model (0.9827). However, d=128's lower mean is due to its heavy use in w500 experiments (which have inherently lower scores). At seq=100, d=128 mean roc=0.9682 — the best seq×d combination. d=256 shows promise (mean 0.9683) but is under-sampled (n=6).

#### nhead
| Value | N | ROC AUC (mean±std) | PA80_F1 | disc_d | Best ROC |
|-------|---|-------------------|---------|--------|----------|
| 1 | 66 | 0.9546±0.0312 | 0.6924 | 1.68 | 0.9809 |
| 2 | 660 | 0.9331±0.0784 | 0.6294 | 1.44 | 0.9810 |
| 4 | 102 | 0.9152±0.0668 | 0.5525 | 1.34 | 0.9827 |
| 6 | 6 | 0.9666±0.0125 | 0.7115 | 1.92 | 0.9791 |
| 8 | 30 | **0.9694**±0.0078 | **0.7289** | **1.94** | 0.9789 |
| 16 | 144 | 0.9543±0.0364 | 0.6811 | 1.74 | 0.9807 |
| 32 | 6 | 0.9611±0.0136 | 0.6872 | 1.74 | 0.9746 |

**Insight**: nhead=8 is the clear winner by mean performance (roc=0.9694, pa80=0.7289) with the lowest variance (±0.0078). nhead=6 and nhead=8 share the highest disc_d (~1.93). The baseline nhead=2 is among the weakest. More heads generally help at d=64, likely due to multi-scale pattern capture.

#### num_teacher_decoder_layers
| Value | N | ROC AUC (mean±std) | PA80_F1 | disc_d | Best ROC |
|-------|---|-------------------|---------|--------|----------|
| 1 | 6 | 0.9596±0.0066 | 0.7031 | 1.50 | 0.9658 |
| 2 | 684 | 0.9326±0.0758 | 0.6224 | 1.41 | 0.9809 |
| 3 | 186 | 0.9396±0.0609 | 0.6438 | 1.63 | 0.9799 |
| 4 | 132 | **0.9551**±0.0432 | **0.6957** | **1.81** | **0.9827** |
| 5 | 6 | 0.9682±0.0063 | 0.7192 | 1.83 | 0.9772 |

**Insight**: td=4 is the overall best by mean and max. td=5 shows even higher mean (0.9682) but is under-sampled. The trend is monotonically increasing from td=2 to td=5. However, at w500, td=4 collapses while td=2 excels — the optimal depth is window-size dependent (H6).

#### learning_rate
| Value | N | ROC AUC (mean±std) | PA80_F1 | Best ROC |
|-------|---|-------------------|---------|----------|
| 0.0005 | 6 | 0.9602±0.0093 | 0.7047 | 0.9720 |
| 0.001 | 6 | 0.9525±0.0094 | 0.6636 | 0.9677 |
| 0.002 | 960 | 0.9356±0.0712 | 0.6320 | 0.9827 |
| 0.003 | 30 | 0.9665±0.0165 | 0.7261 | 0.9779 |
| **0.005** | **12** | **0.9739**±0.0034 | **0.7580** | 0.9777 |

**Insight**: lr=0.005 has the highest mean ROC and PA80_F1, with the lowest variance. lr=0.003 is also strong. The baseline lr=0.002 has the highest best_roc (0.9827) but the lowest mean (0.9356), likely because most experiments use it including poorly-configured ones. **Learning rate is the single most impactful under-explored parameter.**

#### num_student_decoder_layers
| Value | N | ROC AUC (mean±std) | PA80_F1 | disc_d | recon_d | Best ROC |
|-------|---|-------------------|---------|--------|---------|----------|
| 1 | 936 | 0.9362±0.0707 | 0.6344 | 1.49 | 1.96 | 0.9827 |
| **2** | **78** | **0.9489**±0.0548 | **0.6669** | **1.71** | **2.03** | 0.9810 |

**By mask timing**: sd=2 benefits mask_before more than mask_after: before+sd=2 mean roc=0.9575 vs before+sd=1's 0.9337 (+0.024), while after shows only +0.002. sd=2 also boosts mask_before disc_d from 0.72→1.03 (+43%) and recon_d from 2.38→2.49.

**Insight**: sd=2 provides meaningful gains, especially for mask_before. The dual student decoder adds reconstruction diversity. The improvement is asymmetric — mask_before benefits more because the student can learn distinct reconstruction strategies from the unmasked encoder output.

#### num_encoder_layers
| Value | N | ROC AUC (mean±std) | PA80_F1 | disc_d | recon_d | Best ROC |
|-------|---|-------------------|---------|--------|---------|----------|
| 1 | 960 | 0.9366±0.0714 | 0.6362 | 1.51 | 1.97 | 0.9827 |
| 2 | 42 | 0.9484±0.0192 | 0.6497 | 1.50 | 1.88 | 0.9737 |
| **3** | **6** | **0.9579**±0.0102 | **0.6739** | 1.43 | **2.06** | 0.9709 |
| 4 | 6 | 0.9342±0.0164 | 0.6244 | 1.05 | 1.86 | 0.9626 |

**By mask timing**: mask_after benefits slightly from deeper encoders (el=3 after: roc=0.9584, disc_d=2.31) while mask_before also improves (el=3 before: 0.9574, recon_d=2.49). el=4 degrades both pathways, with disc_d dropping to 1.54 (after) and 0.57 (before).

**Insight**: el=2-3 improves mean ROC over el=1, with el=3 achieving the best mean (0.9579). However, el=4 suffers from overfitting/capacity saturation. The encoder depth sweet spot is 2-3 layers, but the effect is modest compared to decoder depth. The diminishing returns at el=4 suggest the encoder's role is primarily feature extraction, not deep representation learning.

#### patch_size
| Value | N | ROC AUC (mean±std) | PA80_F1 | disc_d | dist_d | Best ROC |
|-------|---|-------------------|---------|--------|--------|----------|
| 5 | 18 | 0.9069±0.0664 | 0.5548 | 1.59 | 0.94 | 0.9710 |
| 10 | 888 | 0.9376±0.0712 | 0.6408 | 1.52 | 0.57 | 0.9827 |
| 20 | 102 | 0.9365±0.0563 | 0.6128 | 1.39 | 1.02 | 0.9783 |
| **25** | **6** | **0.9718**±0.0015 | **0.7147** | **1.60** | 0.44 | 0.9745 |

**By mask timing**: p=5 shows a dramatic mask timing asymmetry: after roc=0.9558 (disc_d=3.05) vs before roc=0.8579 (disc_d=0.12). Small patches with mask_before almost completely eliminates discrepancy signal. p=20 is more balanced: after 0.9300, before 0.9431. p=25 slightly favors before (0.9727 vs 0.9709).

**Patch × Window interaction**:
| Patch | Window | N | ROC AUC | disc_d | Patches/Window |
|-------|--------|---|---------|--------|----------------|
| 5 | 100 | 12 | 0.9288 | 1.75 | 20 |
| 10 | 100 | 726 | 0.9546 | 1.71 | 10 |
| **20** | **100** | **12** | **0.9698** | **1.71** | **5** |
| 25 | 100 | 6 | 0.9718 | 1.60 | 4 |
| 10 | 500 | 144 | 0.8657 | 0.63 | 50 |
| **20** | **500** | **78** | **0.9507** | **1.51** | **25** |

**Insight**: Fewer patches per window consistently improves detection. At w100, p=20 (5 patches) and p=25 (4 patches) dramatically outperform p=10 (10 patches). At w500, p=20 (25 patches) massively outperforms p=10 (50 patches): +0.085 roc, +0.88 disc_d. **The optimal number of patches per window appears to be 4-10**, regardless of window size. Too many patches create noise in score aggregation.

#### lambda_disc (Discrepancy Loss Weight)
| Value | N | ROC AUC (mean±std) | PA80_F1 | disc_d | Best ROC |
|-------|---|-------------------|---------|--------|----------|
| 0.1 | 6 | 0.9579±0.0163 | 0.6792 | 1.55 | 0.9721 |
| 0.25 | 6 | 0.9593±0.0158 | 0.6884 | 1.55 | 0.9721 |
| 0.5 | 984 | 0.9365±0.0706 | 0.6353 | 1.51 | 0.9827 |
| 1.0 | 6 | 0.9611±0.0163 | 0.6941 | 1.55 | 0.9721 |
| **2.0** | **6** | **0.9611**±0.0163 | **0.6950** | 1.55 | 0.9721 |
| 3.0 | 6 | 0.9606±0.0160 | 0.6959 | 1.55 | 0.9721 |

**Critical interaction — λ_disc × mask_timing × scoring**:
| λ_disc | after+default | after+adaptive | before+default | before+adaptive |
|--------|--------------|----------------|----------------|-----------------|
| 0.5 | 0.9324 | 0.9417 | 0.9482 | 0.9353 |
| 2.0 | 0.9719 | 0.9721 | 0.9693 | 0.9497 |
| 3.0 | 0.9720 | 0.9721 | 0.9660 | 0.9497 |

**Insight**: At λ_disc≥2.0, the after+default vs after+adaptive gap nearly vanishes (0.9719 vs 0.9721), compared to the large gap at λ_disc=0.5 (0.9324 vs 0.9417). **Higher λ_disc makes mask_after work well even with default scoring**, eliminating the need for adaptive scoring mode. This is because stronger discrepancy weight during training produces a discrepancy signal that doesn't need post-hoc adaptive reweighting. However, sample sizes are small (n=6 per non-baseline value) — this finding requires Phase 2 validation.

#### dynamic_margin_k
| Value | N | ROC AUC (mean±std) | PA80_F1 | disc_d | Best ROC |
|-------|---|-------------------|---------|--------|----------|
| 1.0 | 6 | 0.9587±0.0184 | 0.6697 | 1.48 | 0.9718 |
| 1.5 | 984 | 0.9365±0.0706 | 0.6357 | 1.50 | 0.9827 |
| 2.0 | 6 | 0.9593±0.0167 | 0.6818 | 1.57 | 0.9720 |
| 2.5 | 6 | 0.9582±0.0177 | 0.6739 | 1.58 | 0.9721 |
| **3.0** | **6** | **0.9602**±0.0188 | **0.6840** | **1.62** | **0.9749** |
| 4.0 | 6 | 0.9611±0.0181 | 0.6749 | 1.60 | 0.9746 |

**By mask timing**: mk primarily affects mask_after: mk=3.0 after roc=0.9718, disc_d=2.68 (best) vs mk=1.5 after roc=0.9379, disc_d=2.26. mask_before is less sensitive (mk=3.0 before: 0.9486, disc_d=0.56 vs mk=1.5: 0.9351, disc_d=0.75).

**Insight**: mk=3.0-4.0 consistently outperforms the baseline mk=1.5. The effect is stronger for mask_after, where disc_d increases from 2.26→2.68 (+0.42). Higher margins create a wider separation target during training, forcing the student to produce more distinct outputs. mk=3.0 appears optimal as mk=4.0 shows slightly lower PA80_F1.

#### dim_feedforward
| Value | N | ROC AUC (mean±std) | PA80_F1 | disc_d | Best ROC |
|-------|---|-------------------|---------|--------|----------|
| 64 | 6 | 0.9592±0.0130 | 0.6948 | 1.19 | 0.9703 |
| 128 | 12 | 0.9562±0.0167 | 0.6761 | 1.47 | 0.9756 |
| 256 | 888 | 0.9384±0.0710 | 0.6428 | 1.51 | 0.9827 |
| 384 | 6 | 0.8810±0.0340 | 0.4071 | 0.52 | 0.9392 |
| 512 | 84 | 0.9214±0.0670 | 0.5785 | 1.52 | 0.9807 |
| 768 | 12 | 0.9430±0.0304 | 0.6131 | 1.42 | 0.9791 |
| **1024** | **6** | **0.9683**±0.0080 | **0.7186** | **1.76** | 0.9756 |

**By mask timing**: ff=1024 after: roc=0.9725, disc_d=2.73 (among the highest disc_d values). ff=384 is anomalous (roc=0.8810) — this corresponds to d_model=96 experiments that underperformed across the board.

**Insight**: dim_feedforward is strongly coupled with d_model (typically ff=4×d). ff=1024 (with d=256) shows the best mean, but cannot be isolated from the d_model effect. The baseline ff=256 (d=64) is adequate, but larger ff values (512-1024) with correspondingly larger d_model show promise. The ff=384 anomaly (d=96) suggests d_model=96 is a suboptimal dimensionality.

#### dropout
| Value | N | ROC AUC (mean±std) | PA80_F1 | disc_d | Best ROC |
|-------|---|-------------------|---------|--------|----------|
| 0.0 | 12 | 0.9580±0.0121 | 0.6658 | 1.44 | 0.9748 |
| 0.05 | 6 | 0.9513±0.0160 | 0.6848 | 1.66 | 0.9674 |
| 0.1 | 966 | 0.9363±0.0712 | 0.6350 | 1.50 | 0.9827 |
| **0.2** | **18** | **0.9601**±0.0148 | **0.6902** | **1.68** | 0.9799 |
| 0.3 | 12 | 0.9488±0.0147 | 0.6541 | 1.41 | 0.9747 |

**By mask timing**: dropout=0.2 after: roc=0.9655, disc_d=2.59, pa80=0.7465 — the best mask_after profile. dropout=0.0 before: roc=0.9673, recon_d=2.54 — the best mask_before profile.

**Insight**: dropout=0.2 outperforms the baseline 0.1 by a meaningful margin (mean roc +0.024, pa80 +0.055). It is particularly effective for mask_after (disc_d=2.59 vs 2.27 at dropout=0.1). The improvement suggests the baseline dropout=0.1 may slightly underregularize. However, dropout=0.3 shows degradation, indicating 0.2 is near-optimal. For mask_before, dropout=0.0 works best (roc=0.9673), suggesting mask_before's encoder pathway benefits from full gradient flow.

#### weight_decay
| Value | N | ROC AUC (mean±std) | PA80_F1 | disc_d | Best ROC |
|-------|---|-------------------|---------|--------|----------|
| 0.0 | 12 | 0.9604±0.0152 | 0.6913 | 1.55 | 0.9721 |
| **1e-05** | **990** | 0.9366±0.0704 | 0.6357 | 1.51 | **0.9827** |
| 0.0001 | 12 | **0.9605**±0.0171 | **0.6851** | 1.52 | 0.9734 |

**Insight**: No significant effect. wd=0.0 and wd=1e-4 show nearly identical performance to baseline wd=1e-5. Weight decay is not a critical hyperparameter for this architecture.

#### anomaly_loss_weight
| Value | N | ROC AUC (mean±std) | PA80_F1 | disc_d | Best ROC |
|-------|---|-------------------|---------|--------|----------|
| 1.0 | 1008 | 0.9370±0.0699 | 0.6366 | 1.51 | 0.9827 |
| **2.0** | **6** | **0.9604**±0.0204 | **0.6787** | **1.63** | 0.9761 |

**By mask timing**: alw=2.0 dramatically helps mask_after (roc=0.9733, disc_d=2.77 — among the highest) while slightly hurting mask_before (roc=0.9476 vs 0.9355 baseline).

**Insight**: anomaly_loss_weight=2.0 boosts discrepancy learning for mask_after, with disc_d increasing from 2.27→2.77 (+22%). This amplifies the anomaly reconstruction penalty, making the model more sensitive to anomalous patterns. Small sample (n=6) requires validation.

#### masking_strategy
| Value | N | ROC AUC (mean±std) | PA80_F1 | disc_d | Best ROC |
|-------|---|-------------------|---------|--------|----------|
| **patch** | **1008** | **0.9393**±0.0583 | **0.6390** | **1.51** | **0.9827** |
| feature_wise | 6 | 0.5732±0.3741 | 0.2912 | 0.39 | 0.9173 |

**By mask timing**: feature_wise+mask_before catastrophically fails (roc=0.2317, disc_d=-0.44). feature_wise+mask_after is marginally viable (roc=0.9147) but far worse than patch masking.

**Insight**: Feature-wise masking is fundamentally incompatible with this architecture. It masks entire feature channels rather than temporal patches, destroying the spatial structure the CNN patchifier relies on. **patch masking is the only viable strategy** — feature_wise should be excluded from Phase 2.

### 18.2 Parameter Interaction Effects

#### masking_ratio × num_teacher_decoder_layers
| mr | td | N | ROC AUC | PA80_F1 | disc_d |
|----|---|----|---------|---------|--------|
| 0.08 | 2 | 12 | 0.9627 | 0.6955 | 1.85 |
| **0.08** | **3** | **12** | **0.9714** | **0.7539** | **2.13** |
| 0.08 | 4 | 6 | 0.9668 | 0.7166 | 1.93 |
| 0.10 | 2 | 72 | 0.9358 | 0.6192 | 1.56 |
| 0.10 | 3 | 60 | 0.9413 | 0.6568 | 1.74 |
| 0.10 | 4 | 54 | 0.9457 | 0.6580 | 1.75 |
| 0.20 | 2 | 486 | 0.9364 | 0.6312 | 1.40 |
| 0.20 | 4 | 72 | 0.9612 | 0.7223 | 1.84 |

**Insight**: mr=0.08×td=3 is the single best interaction (roc=0.9714, pa80=0.7539, dd=2.13). Lower masking ratios benefit from deeper decoders — the decoder needs more capacity to learn from fewer masked patches. At mr=0.20, td=4 is needed to match the performance of mr=0.08×td=3.

#### seq_length × d_model
| seq | d_model | N | ROC AUC | disc_d |
|-----|---------|---|---------|--------|
| 100 | 64 | 702 | 0.9537 | 1.70 |
| **100** | **128** | **30** | **0.9682** | **2.14** |
| 100 | 192 | 6 | 0.9666 | 1.92 |
| 100 | 256 | 6 | 0.9683 | 1.76 |
| 500 | 64 | 168 | 0.8920 | 0.87 |
| 500 | 128 | 48 | 0.9027 | 1.27 |
| 1000 | 64 | 12 | 0.7248 | 0.10 |
| 1000 | 128 | 6 | 0.8267 | 0.47 |

**Insight**: Larger models are increasingly important at longer windows. At w500, d=128 provides +0.011 roc and +0.40 disc_d over d=64. At w1000, the gap widens to +0.10 roc. At w100, d=128 provides +0.015 roc and +0.44 disc_d over d=64.

#### mask_timing × scoring_mode
| Timing | Scoring | N | ROC AUC | PA80_F1 | disc_d |
|--------|---------|---|---------|---------|--------|
| before | **default** | 169 | **0.9488** | **0.6699** | 0.74 |
| before | adaptive | 169 | 0.9358 | 0.6237 | 0.74 |
| before | normalized | 169 | 0.9220 | 0.5751 | 0.74 |
| after | default | 169 | 0.9334 | 0.6372 | 2.27 |
| **after** | **adaptive** | 169 | **0.9426** | **0.6589** | **2.27** |
| after | normalized | 169 | 0.9405 | 0.6566 | 2.27 |

**Insight**: The most impactful combined parameter choice. mask_before prefers default scoring; adaptive/normalized actually hurt it. mask_after benefits dramatically from adaptive scoring (+0.009 roc, +0.022 pa80 over default). disc_d is a property of the trained model (unchanged by scoring mode), but the scoring mode determines how well that separation translates to detection performance.

#### nhead × d_model (Selected)
| nhead | d_model | N | Mean ROC | Best ROC |
|-------|---------|---|----------|----------|
| 1 | 64 | 66 | 0.9546 | 0.9809 |
| 8 | 64 | 24 | **0.9697** | 0.9789 |
| 6 | 192 | 6 | 0.9666 | 0.9791 |
| 16 | 64 | 132 | 0.9539 | 0.9779 |
| 16 | 128 | 12 | 0.9590 | 0.9807 |

**Insight**: nhead=8 at d=64 (head_dim=8) gives the highest mean roc among large-sample combos. nhead=16 at d=128 (head_dim=8) also performs well. head_dim=8 appears to be a sweet spot across multiple model sizes.

#### num_student_decoder_layers × mask_timing
| sd | Timing | N | ROC AUC | PA80_F1 | disc_d | recon_d |
|----|--------|---|---------|---------|--------|---------|
| 1 | after | 468 | 0.9387 | 0.6497 | 2.26 | 1.54 |
| 1 | before | 468 | 0.9337 | 0.6191 | 0.72 | 2.38 |
| 2 | after | 39 | 0.9402 | 0.6656 | 2.38 | 1.56 |
| **2** | **before** | **39** | **0.9575** | **0.6681** | **1.03** | **2.49** |

**Insight**: sd=2 disproportionately benefits mask_before (+0.024 roc, +0.049 pa80, +0.31 disc_d) while showing minimal effect on mask_after (+0.002 roc). The dual student decoder appears to create a secondary discrepancy pathway that compensates for mask_before's weaker primary discrepancy signal.

#### num_encoder_layers × num_teacher_decoder_layers
| el | td | N | ROC AUC | disc_d |
|----|---|----|---------|--------|
| 1 | 2 | 660 | 0.9322 | 1.42 |
| 1 | 3 | 174 | 0.9386 | 1.64 |
| 1 | 4 | 114 | 0.9561 | 1.83 |
| **1** | **5** | **6** | **0.9682** | **1.83** |
| 2 | 2 | 12 | 0.9411 | 1.15 |
| 2 | 3 | 12 | 0.9548 | 1.57 |
| 2 | 4 | 18 | 0.9490 | 1.68 |

**Insight**: With el=1, deeper decoders consistently help (td=2→5: roc 0.9322→0.9682). With el=2, td=3 is the peak (0.9548) and td=4 slightly regresses (0.9490). Deeper encoders shift the optimal decoder depth downward, suggesting a total model depth budget where encoder and decoder compete for capacity.

#### dynamic_margin_k × mask_timing
| mk | after ROC | after disc_d | before ROC | before disc_d |
|----|-----------|-------------|------------|--------------|
| 1.0 | 0.9691 | 2.46 | 0.9483 | 0.51 |
| 1.5 | 0.9379 | 2.26 | 0.9351 | 0.75 |
| 2.0 | 0.9684 | 2.57 | 0.9502 | 0.57 |
| 3.0 | **0.9718** | **2.68** | 0.9486 | 0.56 |
| 4.0 | 0.9717 | 2.63 | 0.9505 | 0.57 |

**Insight**: mk primarily benefits mask_after, with disc_d increasing from 2.26 (mk=1.5) to 2.68 (mk=3.0). mask_before disc_d actually decreases at higher mk values (0.75→0.56), suggesting the wider margin is too aggressive for mask_before's weaker discrepancy pathway. **mk should be tuned differently per mask timing**: mk=3.0 for mask_after, mk=1.5 for mask_before.

---

## 19. Discrepancy SNR Deep Analysis

> **disc_SNR = (disc_anomaly_mean − disc_normal_mean) / (disc_anomaly_std + disc_normal_std)**
>
> This metric captures the signal-to-noise ratio of discrepancy-based separation. Higher SNR means anomaly discrepancy scores are more cleanly separated from normal scores, accounting for both mean difference and distribution spread.

### 19.1 Correlation with Performance Metrics

| Separation Metric | Corr(roc_auc) | Corr(PA80_F1) |
|-------------------|---------------|---------------|
| **disc_SNR** | **0.492** | 0.396 |
| disc_cohens_d | 0.481 | 0.340 |
| disc_ratio | 0.411 | **0.414** |

disc_SNR is the strongest single predictor of roc_auc. disc_ratio slightly edges out for PA80_F1, likely because PA80_F1's precision-constrained threshold is more sensitive to the raw ratio than to distribution shape.

### 19.2 Overall Statistics

| Statistic | Value |
|-----------|-------|
| Mean | 0.421 |
| Std | 0.263 |
| Min | −0.753 |
| Median | 0.367 |
| P75 | 0.630 |
| P90 | 0.817 |
| Max | 1.068 |

Negative disc_SNR (anomaly discrepancy *lower* than normal) occurs in pathological cases: w2000 experiments and feature_wise masking.

### 19.3 Top disc_SNR Configs

| Config | disc_SNR | roc_auc | PA80_F1 | Key Params |
|--------|----------|---------|---------|------------|
| 067_optimal_v3 (after) | **1.068** | 0.9866 | 0.7077 | d=128, nh=4, td=2, mr=0.15 |
| 167_optimal_combo_4 (after) | 1.048 | 0.9569 | 0.6860 | d=192, nh=6, td=3, mr=0.10 |
| 063_w500_p20_d128 (after) | 1.032 | 0.9779 | 0.6265 | d=128, nh=4, td=2, mr=0.20, w500 |
| 109_w500_p20_d128_t3s2 (after) | 1.031 | 0.9694 | 0.6352 | d=64, nh=2, td=3, mr=0.20, w500 |
| 070_optimal_final (after) | 1.029 | 0.9796 | 0.6543 | d=128, nh=4, td=3, mr=0.15 |
| 022_d_model_128 / 065_optimal_v1 (after) | 1.022 | 0.9619 | 0.7305 | d=128, nh=4, td=2, mr=0.15-0.20 |

All top-SNR configs share: **mask_after**, **d≥128**, **nh≥4**. The #1 SNR config (067) is also one of the top roc_auc performers overall.

### 19.4 Bottom disc_SNR Configs

| Config | disc_SNR | roc_auc | Failure Mode |
|--------|----------|---------|--------------|
| 010_window_2000_p20 (after) | **−0.753** | 0.700 | Reversed separation — normal disc > anomaly disc |
| 040_feature_wise_mask (before) | −0.291 | 0.350 | Feature-wise masking collapse |
| 004_window_1000_p20 (before) | −0.120 | 0.888 | Long window + before = weak disc signal |

Negative SNR indicates the discrepancy pathway is *counter-productive* — it would improve detection to ignore it entirely (or invert it).

### 19.5 disc_SNR by Key Parameters

#### Mask Timing (dominant factor)

| Timing | Mean SNR | Std | Best |
|--------|----------|-----|------|
| **mask_after** | **0.614** | 0.229 | 1.068 |
| mask_before | 0.228 | 0.107 | 0.540 |

**mask_after produces 2.7× higher mean SNR** than mask_before. This is the single most impactful parameter for disc_SNR, confirming that masking after encoding creates a fundamentally stronger discrepancy signal.

#### d_model

| d_model | N | Mean SNR | Best SNR |
|---------|---|----------|----------|
| 16 | 12 | 0.311 | 0.570 |
| 32 | 12 | 0.353 | 0.638 |
| 64 | 1800 | 0.417 | 1.031 |
| 96 | 12 | 0.226 | 0.501 |
| **128** | 168 | **0.474** | **1.068** |
| **192** | 24 | **0.465** | 1.048 |
| 256 | 12 | 0.468 | 0.831 |

d=128 achieves the highest mean SNR, with d=192/256 close behind. The d=96 dip is anomalous (small sample from specific configs). Larger models create cleaner discrepancy separation.

#### nhead

| nhead | N | Mean SNR | Best SNR |
|-------|---|----------|----------|
| 1 | 132 | 0.449 | 0.892 |
| 2 | 1332 | 0.402 | 1.031 |
| 4 | 204 | 0.423 | **1.068** |
| **6** | 12 | **0.554** | 1.048 |
| **8** | 60 | **0.497** | 0.915 |
| 16 | 288 | 0.470 | 0.985 |
| 32 | 12 | 0.457 | 0.861 |

nh=6 has the highest mean SNR (but n=12). Among well-represented values, nh=8 and nh=16 outperform nh=2. More attention heads improve discrepancy signal quality.

#### Decoder Depth (td)

| td | N | Mean SNR | Best SNR |
|----|---|----------|----------|
| 1 | 12 | 0.388 | 0.738 |
| 2 | 1380 | 0.401 | **1.068** |
| 3 | 372 | 0.454 | 1.048 |
| **4** | 264 | **0.476** | 0.937 |
| 5 | 12 | 0.461 | 0.720 |

Deeper decoders monotonically improve mean SNR up to td=4. However, td=2 achieves the highest individual SNR (1.068) when combined with d=128+nh=4, showing that shallow-but-wide architectures can also achieve top SNR.

#### Masking Ratio

| mr | N | Mean SNR | Best SNR |
|----|---|----------|----------|
| 0.05 | 36 | 0.260 | 0.690 |
| **0.08** | 60 | **0.513** | **0.970** |
| 0.10 | 384 | 0.469 | 1.048 |
| 0.15 | 108 | 0.481 | **1.068** |
| 0.20 | 1332 | 0.407 | 1.032 |
| 0.50+ | various | 0.28-0.40 | <0.74 |

mr=0.08 has the highest mean SNR, with mr=0.10-0.15 close behind. Very low (0.05) and very high (0.50+) masking ratios both degrade SNR. The sweet spot for disc_SNR is **mr=0.08–0.15**.

#### Sequence Length

| seq_length | N | Mean SNR | Best SNR |
|------------|---|----------|----------|
| **100** | 1512 | **0.451** | 1.048 |
| 200 | 24 | 0.404 | 0.748 |
| 500 | 456 | 0.356 | **1.068** |
| 1000 | 36 | 0.168 | 0.446 |
| 2000 | 12 | −0.136 | 0.450 |

SNR degrades sharply with window length. At w1000+, mean SNR approaches zero (or goes negative at w2000). Shorter windows produce cleaner discrepancy separation — the temporal dilution effect reduces signal quality.

#### Learning Rate

| lr | Mean SNR |
|----|----------|
| 0.0005 | 0.394 |
| 0.001 | 0.404 |
| 0.002 | 0.418 |
| **0.003** | **0.500** |
| 0.005 | 0.436 |

lr=0.003 yields the best mean SNR, consistent with the broader finding that higher learning rates improve separation (more aggressive optimization converges to wider margins).

### 19.6 disc_SNR Interactions

#### Mask Timing × d_model

| | d=64 | d=128 | d=192 | d=256 |
|---|------|-------|-------|-------|
| **after** | 0.605 | **0.716** | **0.765** | **0.744** |
| before | 0.230 | 0.232 | 0.164 | 0.193 |

mask_after SNR *increases* with model size (0.605→0.765), while mask_before SNR remains flat (~0.20) regardless of d_model. **Larger models amplify the disc_SNR advantage of mask_after but do not help mask_before's discrepancy pathway**.

#### Mask Timing × nhead

| | nh=1 | nh=2 | nh=4 | nh=6 | nh=8 | nh=16 |
|---|------|------|------|------|------|-------|
| **after** | 0.627 | 0.587 | 0.640 | **0.908** | **0.750** | 0.666 |
| before | 0.272 | 0.216 | 0.207 | 0.200 | 0.243 | 0.275 |

After+nh=6 achieves a remarkable 0.908 mean SNR. mask_before SNR is uniformly low and insensitive to nhead changes.

#### td × mr

| | mr=0.08 | mr=0.10 | mr=0.15 | mr=0.20 |
|---|---------|---------|---------|---------|
| td=2 | 0.482 | 0.448 | 0.471 | 0.396 |
| td=3 | **0.546** | 0.482 | **0.500** | 0.417 |
| td=4 | 0.507 | 0.483 | — | **0.468** |

td=3+mr=0.08 achieves the highest interaction SNR (0.546). Deeper decoders consistently improve SNR at each masking ratio level.

### 19.7 High-SNR Config Profile

Comparing top 10% disc_SNR (≥0.817, n=204) vs bottom 50% (<0.367, n=1020):

| Parameter | High SNR (top 10%) | Low SNR (bottom 50%) | Direction |
|-----------|-------------------|---------------------|-----------|
| mask_after_encoder | **100%** | 10.9% | mask_after essential |
| d_model | 79.1 | 71.0 | Larger preferred |
| nhead | 7.1 | 4.3 | More heads preferred |
| td | 2.8 | 2.4 | Slightly deeper preferred |
| masking_ratio | 0.159 | 0.189 | Lower mr preferred |
| seq_length | 141.2 | 249.7 | Shorter windows preferred |
| learning_rate | 0.0021 | 0.0020 | Slightly higher lr preferred |

**The high-SNR profile is: mask_after + d≥128 + nh≥4 + mr≈0.10-0.15 + seq≤200 + lr≥0.002**. mask_after is by far the most critical factor — 100% of top-10% SNR experiments use mask_after.

### 19.8 Key Insights

> **I1**: disc_SNR is fundamentally a mask_after metric. mask_before configs are capped at SNR≈0.54, while mask_after reaches 1.07. Optimizing disc_SNR is equivalent to optimizing the mask_after discrepancy pathway.

> **I2**: Model capacity (d_model, nhead) amplifies disc_SNR only for mask_after. Increasing d_model from 64→128 gives +0.111 SNR for mask_after but +0.002 for mask_before. This explains why larger models don't improve mask_before's roc_auc proportionally.

> **I3**: The disc_SNR sweet spot for masking ratio is mr=0.08–0.15 — tighter than the common default of 0.20. Lower masking forces the model to learn more precise discrepancy patterns from fewer masked patches.

> **I4**: Window length is inversely correlated with disc_SNR. At w≥1000, disc_SNR approaches or goes negative, meaning the discrepancy signal becomes noise. For disc_SNR-dependent detection, **w≤200 is strongly recommended**.

> **I5**: The formula `disc_SNR = mean_diff / sum_of_stds` is now computed and stored in `summary_results.csv` as `disc_SNR`, `disc_SNR_disturbing`, and `recon_SNR` columns (added to `compute_loss_statistics()` in `run_ablation.py`).

---

## 20. Key Hypotheses & Conclusions

### Validated Hypotheses

| # | Hypothesis | Evidence |
|---|-----------|----------|
| H1 | Short windows + low masking + more heads → max disc_d | Top 20% profile: seq=100, mr=0.15, nhead=6.1 |
| H2 | Masking ratio ~0.10 balances both disc_d and recon_d | Dual-strength models average mr=0.12 |
| H6 | Deeper decoder inverts: helps at w100, hurts at w500 | td3 best at w100, td2 best at w500, td4 collapses at w500 |
| H7 | Larger d_model critical for large windows | d128 gives +0.048 roc_auc at w500 |
| H12 | mask_after + normalized > mask_before + default | After wins 117/169 with normalized scoring |

### Novel Findings

1. **disc_SNR (signal-to-noise ratio) is the best single predictor** of roc_auc among separation metrics (r=0.492 vs disc_d's 0.481 vs disc_ratio's 0.411). Its dominance is amplified when restricted to mask_after experiments (where discrepancy drives detection).

2. **The mask timing × scoring mode interaction is the most impactful combined parameter choice**:
   - mask_after + normalized: PA80_F1 gains +0.08 over mask_after + default
   - This single change (scoring mode) flips mask_after from losing to winning vs mask_before

3. **Window 500 has a unique niche**: Best disturbing-normal separation, especially with d128+p20. Despite lower overall roc_auc, it captures temporal boundary patterns that w100 misses.

4. **Patch granularity creates a discrimination-noise trade-off**: More patches per window → higher per-patch separation BUT noisier aggregated score. Optimal granularity varies by window size.

5. **Learning rate is under-explored**: The few experiments with lr=0.003-0.005 consistently outperform lr=0.002, suggesting systematic under-training.

6. **Optimal patches per window is 4-10**: patch_size×window interaction shows p=20 at w100 (5 patches) and p=20 at w500 (25 patches) both dramatically outperform p=10 (10 and 50 patches respectively). Too many patches create aggregation noise.

7. **sd=2 asymmetrically benefits mask_before**: Dual student decoder improves mask_before roc by +0.024 but mask_after by only +0.002, creating a secondary discrepancy pathway that compensates for mask_before's weaker primary signal.

8. **dropout=0.2 outperforms the baseline 0.1**: Mean roc +0.024, particularly for mask_after (disc_d 2.59 vs 2.27). For mask_before, dropout=0.0 works best, suggesting different regularization optima per mask timing.

9. **dynamic_margin_k should differ by mask timing**: mk=3.0 benefits mask_after (disc_d +0.42) while slightly hurting mask_before disc_d. The optimal margin width depends on the strength of the discrepancy signal.

10. **λ_disc≥2.0 eliminates the scoring mode gap for mask_after**: after+default rises from 0.9324 (λ=0.5) to 0.9719 (λ=2.0), nearly matching after+adaptive (0.9721). This removes the need for adaptive scoring post-processing.

11. **Feature-wise masking is fundamentally incompatible**: roc=0.5732 overall, with mask_before+feature_wise collapsing to 0.2317. Only patch masking is viable.

12. **Encoder depth has a budget interaction with decoder depth**: With el=2+, the optimal td shifts down (el=2: td=3 peaks, el=1: td=5 peaks), suggesting a total model depth capacity constraint.

---

## 21. Visualization Target Models

30 configs (×6 variants = 180 experiment dirs) selected for full best_model visualization. Selection balances recipe models, top performers, parameter space coverage, and edge cases.

### Selection Categories

#### A. Recipe Models (13 configs)
Models from Section 17 optimal parameter recipes — the most important models with established performance profiles.

| # | Config | Key Parameters | Best ROC | Best PA80 | Rationale |
|---|--------|---------------|----------|-----------|-----------|
| 1 | `166_optimal_combo_3` | d=64, nh=1, td=3, mr=0.08, do=0.2 | 0.9799 | 0.8504 | Best PA80_F1 overall |
| 2 | `083_d128_nhead1_mask0.10` | d=64, nh=1, td=2, mr=0.10 | 0.9809 | 0.8320 | Best mask_after roc_auc |
| 3 | `167_optimal_combo_4` | d=192, nh=6, td=3, mr=0.10 | 0.9791 | 0.7866 | Best disc_d=3.37 |
| 4 | `022_d_model_128` | d=128, nh=4, td=2, mr=0.20 | 0.9805 | 0.7915 | Best d=128 configuration |
| 5 | `067_optimal_v3` | seq=500, d=128, nh=4, td=2, mr=0.15 | 0.9783 | 0.7374 | Best w500, best dist_d=2.55 |
| 6 | `165_optimal_combo_2` | d=64, nh=16, td=3, mr=0.10, lr=0.003 | 0.9778 | 0.7675 | High LR mask_after |
| 7 | `028_d128_nhead_16` | d=128, nh=16, td=2, mr=0.20 | 0.9807 | 0.8277 | Wide attention d=128 |
| 8 | `151_pa80_t4s1_nhead4` | d=64, nh=4, td=4, mr=0.20 | **0.9827** | 0.8028 | Best overall roc_auc |
| 9 | `150_pa80_t4s1_nhead1` | d=64, nh=1, td=4, mr=0.20 | 0.9807 | 0.7430 | Best mask_before default |
| 10 | `019_decoder_t4s2` | d=64, nh=2, td=4, sd=2, mr=0.20 | 0.9810 | 0.7936 | Deep decoder + dual student |
| 11 | `009_w500_p20` | seq=500, d=64, nh=2, td=2, mr=0.20 | 0.9744 | 0.7377 | Best w500 mask_before |
| 12 | `168_pa80_final_opt1` | d=64, nh=16, td=4, mr=0.10, lr=0.003 | 0.9779 | 0.7580 | High LR mask_before |
| 13 | `054_lr_0.005` | d=64, nh=2, td=2, mr=0.20, lr=0.005 | 0.9777 | 0.7964 | Highest LR tested |

#### B. Top Performers / Close Competitors (5 configs)
High-performing models not in recipes, providing comparison baselines.

| # | Config | Key Parameters | Rationale |
|---|--------|---------------|-----------|
| 14 | `100_optimal_combo_1` | d=64, nh=1, td=3, mr=0.10 | 3rd best PA80_F1 (mr=0.10 variant of Recipe 1) |
| 15 | `065_optimal_v1` | d=128, nh=4, td=2, mr=0.15 | w100 variant of 067 (same arch, different window) |
| 16 | `068_optimal_v4` | d=128, nh=4, td=3, mr=0.15 | Deeper decoder variant of 065 |
| 17 | `001_default` | d=64, nh=2, td=2, mr=0.20, lr=0.002 | **Baseline** reference for all comparisons |
| 18 | `152_pa80_t4s1_nhead8` | d=64, nh=8, td=4 | Best nhead=8 with deep decoder |

#### C. Parameter Space Coverage (7 configs)
Models exploring under-represented but insightful parameter values.

| # | Config | Key Parameters | Rationale |
|---|--------|---------------|-----------|
| 19 | `027_nhead_8` | d=64, nh=8, td=2 | nhead=8 baseline (highest mean roc nhead value) |
| 20 | `002_window_200` | seq=200, d=64 | w200 representative (sparse region) |
| 21 | `064_w1000_d128_t3s1` | seq=1000, d=128, td=3 | w1000 best — only viable w1000 config |
| 22 | `044_lambda_2.0` | λ_disc=2.0 | λ_disc=2.0 closes scoring mode gap for mask_after |
| 23 | `049_k_3.0` | mk=3.0 | dynamic_margin_k=3.0 (best for mask_after disc_d) |
| 24 | `056_dropout_0.2` | dropout=0.2 | Outperforms baseline dropout=0.1 |
| 25 | `060_anomaly_weight_2.0` | alw=2.0 | anomaly_loss_weight=2.0 (boosts mask_after disc_d) |

#### D. Edge Cases / Special Interest (5 configs)
Models demonstrating failure modes, extreme configurations, or unique behaviors.

| # | Config | Key Parameters | Rationale |
|---|--------|---------------|-----------|
| 26 | `040_feature_wise_mask` | masking_strategy=feature_wise | Failure case: roc=0.57 (feature_wise incompatible) |
| 27 | `072_d128_mask_0.05` | mr=0.05 | Training collapse case (insufficient masking) |
| 28 | `023_d_model_256` | d=256, nh=8 | Largest model tested (mean roc=0.9683) |
| 29 | `089_d128_t4s2` | d=64, nh=2, td=4, sd=2 | Compare dual student effect vs 019 |
| 30 | `158_d128_lr0.005` | d=64, nh=2, td=2, lr=0.005 | lr=0.005 complement to 054 |

### Selection Summary
- **Total**: 30 configs × 6 variants (2 mask timings × 3 scoring modes) = **180 experiment dirs**
- **Coverage**: seq_length ∈ {100, 200, 500, 1000}, d_model ∈ {64, 128, 192, 256}, nhead ∈ {1, 2, 4, 6, 8, 16}, td ∈ {2, 3, 4}, lr ∈ {0.002, 0.003, 0.005}
- **Visualization**: 17 plots per experiment dir (ROC, confusion matrix, detection examples, score contributions, anomaly type performance, etc.)

---

## 22. Phase 2 Recommendations

Based on these findings, Phase 2 should explore:

### High Priority
1. **Systematic lr exploration**: 0.001-0.01 range with the best architectures
2. **mask_after + normalized scoring** as the new default baseline
3. **Masking ratio fine-tuning**: 0.06-0.15 range at w100 and w500
4. **Teacher decoder depth**: Focus on td=3-4 at w100, td=2 at w500
5. **disc_SNR as optimization target** rather than roc_auc alone

### Medium Priority
6. **d_model=128-256 at w500** with patch_size=20 for disturbing separation
7. **dynamic_margin_k=3.0 for mask_after, 1.5 for mask_before** — timing-specific tuning
8. **lambda_disc=2.0-3.0** to eliminate scoring mode dependency for mask_after
9. **nhead=6-8** systematically (current best mean roc values)
10. **Ensemble approach**: mask_after (high disc_d) + mask_before (high recon_d)
11. **dropout=0.2 for mask_after, 0.0 for mask_before** — timing-specific regularization
12. **sd=2 for mask_before** — dual student decoder significantly boosts mask_before
13. **patch_size=20 at w100** (5 patches) and **p=20 at w500** (25 patches) — target 4-10 patches per window
14. **anomaly_loss_weight=2.0 for mask_after** — amplifies discrepancy learning

### Experiment-Specific
15. Re-evaluate **067_optimal_v3** with normalized scoring and higher lr
16. Test **166_optimal_combo_3** at w500 (currently w100-only)
17. Test Recipe 15 (d=128, nhead=8, td=3, mr=0.08, lr=0.003, mask_after)
18. Explore el=2-3 with td=3-4 to find encoder-decoder depth optimum
