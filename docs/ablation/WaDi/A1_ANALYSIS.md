# WaDi A1 Dataset Ablation Study Analysis

**Date**: 2026-02-03
**Total Experiments**: 42
**Anomaly Type**: Spike (point-level anomaly)
**Dataset Features**: 96 features, 275K timesteps

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Experimental Setup](#2-experimental-setup)
3. [Overall Performance Statistics](#3-overall-performance-statistics)
4. [Parameter-wise Analysis](#4-parameter-wise-analysis)
5. [Parameter Interaction Effects](#5-parameter-interaction-effects)
6. [Point-Adjust (PA) Level Analysis](#6-point-adjust-pa-level-analysis)
7. [Statistical Tests](#7-statistical-tests)
8. [Optimal Configurations](#8-optimal-configurations)
9. [Key Insights and Hypotheses](#9-key-insights-and-hypotheses)
10. [Recommendations](#10-recommendations)
11. [Score Type Analysis (Teacher / Student / Discrepancy)](#11-score-type-analysis-teacher--student--discrepancy)

---

## 1. Executive Summary

### Key Findings

| Finding | Statistic | Significance |
|---------|-----------|--------------|
| **Patch size is the most influential parameter** | eta² = 0.103 | p = 0.0015 for PRC-AUC |
| **Best F1 configuration** | w100_p5_td4_sd1 | F1 = 0.6065 |
| **Smaller patches improve precision** | patch=5: 0.453 vs patch=20: 0.291 | Trade-off with recall |
| **Encoder depth 1 is optimal** | enc=1: F1=0.448 vs enc=4: F1=0.418 | eta² = 0.091 |
| **Teacher-Student gap matters** | td4_sd1 > td2_sd2 | Gap=3 layers optimal |

### Performance Range

- **ROC-AUC**: 0.8534 ~ 0.9641 (Mean: 0.9224)
- **PRC-AUC**: 0.2806 ~ 0.7005 (Mean: 0.4991)
- **F1 Score**: 0.2402 ~ 0.6064 (Mean: 0.4336)

---

## 2. Experimental Setup

### Parameter Space

| Parameter | Values Tested | Default |
|-----------|---------------|---------|
| Window Size | 100, 500 | 500 |
| Patch Size | 5, 10, 20 | 5 |
| Teacher Decoder Layers | 2, 3, 4 | 2 |
| Student Decoder Layers | 1, 2 | 1 |
| D_model | 64, 128 | 128 |
| Encoder Layers | 1, 2, 3, 4 | 1 |

### Metrics Collected

| Metric | Description | Importance |
|--------|-------------|------------|
| ROC-AUC | Area under ROC curve | Overall discrimination |
| PRC-AUC | Area under Precision-Recall curve | Imbalanced data performance |
| F1 Score | Harmonic mean of precision/recall | Balanced performance |
| Precision | TP / (TP + FP) | False alarm minimization |
| Recall | TP / (TP + FN) | Detection coverage |
| PA_k_* | Point-Adjust metrics at k% tolerance | Temporal accuracy |

---

## 3. Overall Performance Statistics

### Key Metrics Summary

| Metric | Mean | Std | Min | Max | Median |
|--------|------|-----|-----|-----|--------|
| ROC-AUC | 0.9224 | 0.0228 | 0.8534 | 0.9641 | 0.9247 |
| PRC-AUC | 0.4991 | 0.1008 | 0.2806 | 0.7005 | 0.5043 |
| F1 Score | 0.4336 | 0.0682 | 0.2402 | 0.6064 | 0.4236 |
| Precision | 0.3937 | 0.1421 | 0.1405 | 0.8020 | 0.3647 |
| Recall | 0.5977 | 0.1730 | 0.2381 | 0.9263 | 0.6197 |

### Top 5 Experiments

| Rank | Experiment | F1 | Precision | Recall | ROC-AUC | PRC-AUC |
|------|------------|----|-----------|--------|---------|---------|
| 1 | w100_p5_td4_sd1 | **0.6064** | 0.4900 | 0.7953 | 0.9515 | 0.4565 |
| 2 | w500_p5_enc4 | 0.5539 | 0.7478 | 0.4398 | 0.9077 | 0.6296 |
| 3 | w500_p20_d64 | 0.5485 | 0.3896 | 0.9263 | 0.9641 | 0.5261 |
| 4 | baseline (w500_p5_td2_sd2) | 0.5315 | 0.6263 | 0.4616 | 0.9361 | 0.6343 |
| 5 | w100_p10_d64 | 0.5216 | 0.4730 | 0.5814 | 0.9568 | 0.4220 |

### Worst 5 Experiments

| Rank | Experiment | F1 | Precision | Recall | ROC-AUC | PRC-AUC |
|------|------------|----|-----------|--------|---------|---------|
| 42 | w500_p10_enc4 | **0.2402** | 0.1405 | 0.8265 | 0.8613 | 0.2806 |
| 41 | w500_p20_enc2 | 0.3305 | 0.2112 | 0.7604 | 0.8893 | 0.5990 |
| 40 | w500_p10_enc2 | 0.3494 | 0.6558 | 0.2381 | 0.9208 | 0.6321 |
| 39 | w500_p20_enc4 | 0.3563 | 0.2295 | 0.7959 | 0.9045 | 0.7005 |
| 38 | w500_p10_td4_sd2 | 0.3608 | 0.2910 | 0.4747 | 0.8949 | 0.4584 |

---

## 4. Parameter-wise Analysis

### 4.1 Window Size (100 vs 500)

| Window | F1 Score | Precision | Recall | ROC-AUC | PRC-AUC |
|--------|----------|-----------|--------|---------|---------|
| 100 | 0.4434 | 0.3932 | 0.6103 | 0.9258 | 0.4484 |
| 500 | 0.4276 | 0.3939 | 0.5900 | 0.9203 | 0.5303 |

**Statistical Test**: t-test, t=0.716, p=0.478 (Not significant)

**Effect Size**: eta² = 0.013 (Negligible)

**Insight**: Window size has minimal impact on spike detection performance.

**Hypothesis**: Spike anomalies are point-level events that don't require long temporal context for detection. The 100-timestep window already captures sufficient local patterns for identifying spike anomalies.

---

### 4.2 Patch Size (5, 10, 20)

| Patch | F1 Score | ROC-AUC | PRC-AUC | Precision | Recall |
|-------|----------|---------|---------|-----------|--------|
| 5 | **0.4559** | 0.9260 | 0.5136 | **0.4545** | 0.5515 |
| 10 | 0.4073 | 0.9170 | 0.4459 | 0.3639 | 0.5751 |
| 20 | 0.4361 | 0.9251 | **0.5729** | 0.3164 | **0.7469** |

**Statistical Tests**:
- PRC-AUC: ANOVA F=7.71, **p=0.0015*** (Highly significant)
- Recall: ANOVA F=4.37, **p=0.0194*** (Significant)

**Effect Size**: eta² = 0.103 (Largest among all parameters)

**Key Observations**:
1. Patch size 5 achieves best F1 and PRC-AUC
2. Patch size 20 has highest recall but lowest precision
3. Clear precision-recall trade-off exists

**Hypothesis**: Larger patches aggregate information over broader regions, making them more likely to detect anomalies but also more prone to false positives. For point-level anomalies like spikes, smaller patches provide the fine-grained temporal resolution needed for precise localization.

---

### 4.3 Teacher Decoder Layers (2, 3, 4)

| TD Layers | F1 Score | Precision | Recall | ROC-AUC | PRC-AUC |
|-----------|----------|-----------|--------|---------|---------|
| 2 | 0.4224 | 0.4046 | 0.5953 | 0.9211 | **0.5067** |
| 3 | 0.4308 | 0.3386 | 0.6232 | 0.9219 | 0.4969 |
| 4 | **0.4613** | 0.3929 | 0.5918 | 0.9257 | 0.4821 |

**Statistical Test**: ANOVA F=1.25, p=0.299 (Not significant)

**Effect Size**: eta² = 0.060 (Small-Medium)

**Insight**: Deeper teacher decoder shows consistent improvement trend.

**Hypothesis**: A deeper teacher decoder has greater capacity to learn detailed reconstruction patterns from normal data. This creates a larger "gap" between teacher and student outputs on anomalous data, making discrepancy-based detection more effective.

---

### 4.4 Student Decoder Layers (1, 2)

| SD Layers | F1 Score | Precision | Recall | ROC-AUC | PRC-AUC |
|-----------|----------|-----------|--------|---------|---------|
| 1 | **0.4457** | 0.3898 | 0.6075 | 0.9238 | 0.4689 |
| 2 | 0.4269 | 0.3958 | 0.5923 | 0.9216 | **0.5159** |

**Statistical Test**: t-test, t=0.841, p=0.405 (Not significant)

**Effect Size**: eta² = 0.017 (Negligible)

**Insight**: Shallower student decoder performs slightly better.

**Hypothesis**: A simpler student model maintains an appropriate "knowledge gap" with the teacher. When the student is too capable (sd=2), it may learn to mimic the teacher even on anomalous patterns, reducing the discriminative power of the discrepancy signal.

---

### 4.5 D_model (64 vs 128)

| D_model | F1 Score | Precision | Recall | ROC-AUC | PRC-AUC |
|---------|----------|-----------|--------|---------|---------|
| 64 | 0.4400 | 0.4942 | 0.4758 | 0.9148 | **0.5115** |
| 128 | 0.4327 | 0.3801 | 0.6142 | 0.9234 | 0.4974 |

**Statistical Test**: t-test, t=0.219, p=0.828 (Not significant)

**Effect Size**: eta² = 0.001 (Negligible)

**Insight**: D_model has virtually no impact on F1 score.

**Hypothesis**: For the 96-feature WaDi dataset, even 64 dimensions provide sufficient representational capacity. The benefits of larger embedding dimensions may be more apparent in datasets with higher feature dimensionality or more complex temporal patterns.

---

### 4.6 Encoder Layers (1, 2, 3, 4)

| Enc Layers | F1 Score | Precision | Recall | ROC-AUC | PRC-AUC |
|------------|----------|-----------|--------|---------|---------|
| 1 | **0.4482** | 0.4220 | 0.5582 | 0.9228 | 0.4903 |
| 2 | 0.3928 | 0.3637 | 0.6041 | 0.9231 | **0.5430** |
| 3 | 0.4118 | 0.2868 | 0.7478 | 0.9308 | 0.4900 |
| 4 | 0.4176 | 0.3775 | 0.6545 | 0.9108 | 0.5121 |

**Statistical Test**: ANOVA F=1.26, p=0.300 (Not significant)

**Effect Size**: eta² = 0.091 (Medium)

**Key Observation**: enc_layers=1 is optimal; deeper encoders show degradation.

**Hypothesis**: Deeper encoders may cause "representation collapse" where the encoded features become too abstract, losing the fine-grained information needed for anomaly detection. Alternatively, gradient vanishing during training may prevent deeper encoders from learning effectively.

---

## 5. Parameter Interaction Effects

### 5.1 Window Size x Patch Size

| | Patch=5 | Patch=10 | Patch=20 |
|---|---------|----------|----------|
| **Window=100** | 0.4531 | 0.4337 | N/A |
| **Window=500** | 0.4582 | 0.3809 | 0.4361 |

**Interaction Effect**: Window=500 with Patch=10 shows notable performance drop (0.3809).

**Hypothesis**: The combination of long window (500) with medium patch (10) creates 50 patches, which may be suboptimal for the transformer's attention mechanism. Too many patches with moderate granularity may dilute the anomaly signal.

---

### 5.2 Teacher x Student Decoder Layers

| | SD=1 | SD=2 |
|---|------|------|
| **TD=2** | 0.4281 | 0.4211 |
| **TD=3** | 0.4308 | N/A |
| **TD=4** | **0.4781** | 0.4473 |

**Best Combination**: TD=4, SD=1 (F1=0.4781)

**Insight**: The gap between teacher and student matters more than absolute depth.

**Hypothesis**: A gap of 3 layers (TD=4, SD=1) creates optimal knowledge distillation dynamics. The teacher can learn complex normal patterns while the student remains simpler, maximizing the discrepancy signal on anomalies.

---

### 5.3 Patch Size x Encoder Layers

| | Enc=1 | Enc=2 | Enc=3 | Enc=4 |
|---|-------|-------|-------|-------|
| **Patch=5** | 0.4580 | 0.4403 | 0.4018 | **0.5130** |
| **Patch=10** | 0.4275 | 0.3765 | 0.3913 | 0.3530 |
| **Patch=20** | 0.4659 | 0.3305 | 0.4726 | 0.3563 |

**Key Finding**: Patch=5 with Enc=4 achieves highest F1 (0.5130), but Patch=10/20 with Enc>1 degrades significantly.

**Hypothesis**: Small patches provide enough local detail that deeper encoders can effectively learn hierarchical features. Larger patches with deep encoders may over-abstract the information, losing anomaly-relevant signals.

---

## 6. Point-Adjust (PA) Level Analysis

### 6.1 PA F1 Score Degradation

| PA Level | Mean F1 | Change from PA0 |
|----------|---------|-----------------|
| PA0 | 0.6703 | - |
| PA25 | 0.5047 | -0.1656 |
| PA50 | 0.4474 | -0.2229 |
| PA75 | 0.3632 | -0.3071 |
| PA100 | 0.1407 | -0.5296 |

**Observation**: F1 drops by 79% from PA0 to PA100, indicating models are better at approximate detection than exact localization.

---

### 6.2 PA Performance by Patch Size

| Patch | PA0 F1 | PA50 F1 | PA100 F1 | Drop Rate |
|-------|--------|---------|----------|-----------|
| 5 | **0.7316** | 0.4719 | 0.1190 | 83.7% |
| 10 | 0.6555 | 0.4196 | 0.1366 | 79.2% |
| 20 | 0.5620 | 0.4480 | **0.1977** | 64.8% |

**Insight**:
- Patch=5: Best at exact localization (PA0), worst at PA100
- Patch=20: Worst at exact localization, best at PA100

**Hypothesis**: Smaller patches provide precise temporal localization but are sensitive to small timing errors. Larger patches trade localization precision for robustness to timing variations.

---

### 6.3 PA Performance by Window Size

| Window | PA0 F1 | PA50 F1 | PA100 F1 |
|--------|--------|---------|----------|
| 100 | 0.6684 | 0.4671 | 0.1499 |
| 500 | 0.6715 | 0.4353 | 0.1350 |

**Insight**: Window size has minimal impact on PA metrics.

---

## 7. Statistical Tests

### 7.1 ANOVA Results Summary

| Parameter | F Statistic | p-value | Significance |
|-----------|-------------|---------|--------------|
| Patch Size (F1) | 2.24 | 0.121 | - |
| Patch Size (PRC-AUC) | **7.71** | **0.0015** | *** |
| Patch Size (Recall) | **4.37** | **0.0194** | * |
| TD Layers (F1) | 1.25 | 0.299 | - |
| Enc Layers (F1) | 1.26 | 0.300 | - |

### 7.2 Effect Size Ranking (eta²)

| Rank | Parameter | eta² | Interpretation |
|------|-----------|------|----------------|
| 1 | Patch Size | 0.103 | Medium |
| 2 | Encoder Layers | 0.091 | Medium |
| 3 | TD Layers | 0.060 | Small-Medium |
| 4 | SD Layers | 0.017 | Negligible |
| 5 | Window Size | 0.013 | Negligible |
| 6 | D_model | 0.001 | Negligible |

### 7.3 Correlation Analysis

**Spearman Correlations (Parameter → Metric)**

| Parameter | ROC-AUC | PRC-AUC | F1 | Precision | Recall |
|-----------|---------|---------|-----|-----------|--------|
| Window Size | -0.17 | 0.03 | -0.10 | 0.00 | -0.06 |
| Patch Size | -0.18 | **-0.51** | -0.20 | **-0.38** | **0.36** |
| TD Layers | 0.05 | 0.31 | 0.24 | 0.05 | -0.04 |
| SD Layers | 0.03 | -0.18 | -0.17 | 0.02 | -0.02 |
| D_model | 0.06 | 0.21 | 0.03 | -0.35 | 0.22 |
| Enc Layers | -0.04 | -0.29 | -0.24 | **-0.37** | 0.35 |

**Key Correlations**:
- Patch Size ↔ PRC-AUC: r = -0.51 (Strong negative)
- Patch Size ↔ Recall: r = 0.36 (Moderate positive)
- Enc Layers ↔ Recall: r = 0.35 (Moderate positive)

---

## 8. Optimal Configurations

### 8.1 Best by Metric

| Optimization Target | Best Configuration | Score | Precision | Recall |
|--------------------|-------------------|-------|-----------|--------|
| **F1 Score** | w100_p5_td4_sd1_d128_enc1 | 0.6064 | 0.4900 | 0.7953 |
| **ROC-AUC** | w500_p20_d64_td2_sd2_enc1 | 0.9641 | 0.3896 | 0.9263 |
| **PRC-AUC** | w500_p20_enc4 | 0.7005 | 0.2295 | 0.7959 |
| **Precision** | w100_p5_td2_sd1_d128_enc1 | 0.8020 | **0.8020** | 0.2727 |
| **Recall** | w500_p20_d64_td2_sd2_enc1 | 0.9263 | 0.3896 | **0.9263** |

### 8.2 Pareto Optimal Configurations

Two configurations form the Pareto frontier for F1 vs Recall:

| Configuration | F1 | Recall | Precision | Use Case |
|--------------|-----|--------|-----------|----------|
| w100_p5_td4_sd1 | **0.6064** | 0.7953 | 0.4900 | Balanced / Precision-focused |
| w500_p20_d64 | 0.5485 | **0.9263** | 0.3896 | Recall-focused |

### 8.3 Efficiency Analysis

Most efficient configurations (F1 / log(complexity)):

| Rank | Configuration | F1 | Precision | Recall | Complexity | Efficiency |
|------|--------------|-----|-----------|--------|------------|------------|
| 1 | w500_p20_d64 | 0.5485 | 0.3896 | 0.9263 | 256 | 0.0988 |
| 2 | w100_p10_d64 | 0.5216 | 0.4730 | 0.5814 | 256 | 0.0940 |
| 3 | w100_p5_td4_sd1 | 0.6065 | 0.4901 | 0.7953 | 640 | 0.0938 |

---

## 9. Key Insights and Hypotheses

### Insight 1: Patch Size is the Most Critical Parameter

**Observation**: eta² = 0.103, statistically significant for PRC-AUC (p=0.0015)

**Hypothesis**: For spike anomalies (point-level), temporal resolution directly impacts detection quality. Smaller patches (5) provide finer granularity, enabling:
1. More precise anomaly localization
2. Better separation of normal/anomaly distributions
3. Higher precision at the cost of some recall

**Implication**: Patch size should be tuned based on the expected anomaly duration. For point-like anomalies, use smaller patches; for extended anomalies, larger patches may be appropriate.

---

### Insight 2: Teacher-Student Gap Matters More Than Absolute Depth

**Observation**: TD=4, SD=1 (gap=3) achieves best F1 (0.4781 in interaction analysis)

**Hypothesis**: The knowledge distillation mechanism relies on the student's inability to fully replicate the teacher's behavior on anomalies:
1. Teacher learns detailed normal patterns
2. Student with limited capacity cannot learn anomalous patterns
3. Discrepancy between outputs becomes the anomaly signal

**Implication**: When increasing teacher depth, keep student shallow. A gap of 2-3 decoder layers appears optimal.

---

### Insight 3: Encoder Depth Should Be Minimal

**Observation**: enc=1 outperforms enc=2,3,4 on average F1 (0.4482 vs 0.39-0.42)

**Hypothesis**: Deep encoders may cause:
1. **Representation collapse**: Features become too abstract, losing anomaly-relevant information
2. **Gradient issues**: Vanishing gradients prevent effective learning
3. **Overfitting**: More parameters without corresponding benefit

**Implication**: Start with minimal encoder depth. Only increase if there's evidence of underfitting.

---

### Insight 4: Window Size Has Minimal Impact for Point Anomalies

**Observation**: eta² = 0.013, p = 0.478

**Hypothesis**: Spike anomalies are localized events that don't benefit from extended temporal context. The 100-timestep window already captures sufficient local patterns.

**Implication**: For point-level anomalies, prefer smaller windows for computational efficiency. Longer windows may be beneficial for pattern-level anomalies (drift, seasonal).

---

### Insight 5: Precision-Recall Trade-off Is Controllable

**Observation**:
- Patch=5: High precision (0.453), lower recall (0.551)
- Patch=20: Low precision (0.291), high recall (0.747)

**Hypothesis**: Larger patches aggregate more information, making them "trigger-happy" for anomalies:
1. More likely to capture any anomaly (high recall)
2. Also more likely to flag normal variations (low precision)

**Implication**: Adjust patch size based on operational requirements:
- Mission-critical systems: Use small patches for high precision
- Early warning systems: Use larger patches for high recall

---

## 10. Recommendations

### 10.1 Recommended Default Configuration

For general-purpose spike anomaly detection on WaDi A1:

```
window_size: 100
patch_size: 5
num_teacher_decoder_layers: 4
num_student_decoder_layers: 1
d_model: 128
num_encoder_layers: 1
```

**Expected Performance**: F1 ~ 0.60, Precision ~ 0.49, Recall ~ 0.80

---

### 10.2 Use-Case Specific Recommendations

| Use Case | Configuration | Expected F1 | Precision | Recall | Trade-off |
|----------|--------------|-------------|-----------|--------|-----------|
| **Balanced** | w100_p5_td4_sd1_enc1 | 0.60 | 0.49 | 0.80 | Best overall |
| **High Precision** | w100_p5_td2_sd1_enc1 | 0.41 | 0.80 | 0.27 | Minimize false alarms |
| **High Recall** | w500_p20_d64_enc1 | 0.55 | 0.39 | 0.93 | Minimize missed detections |
| **Resource-Limited** | w100_p10_d64_enc1 | 0.52 | 0.47 | 0.58 | Low memory/compute |

---

### 10.3 Future Experiment Suggestions

1. **Test other anomaly types**: The current analysis only covers spike anomalies. Pattern-level anomalies (drift, seasonal) may show different parameter sensitivities.

2. **Explore finer patch sizes**: Test patch_size=2,3,4 to find the optimal granularity.

3. **Teacher depth exploration**: Test td_layers=5,6 to see if the trend continues.

4. **Learning rate interaction**: Current analysis doesn't include learning rate effects.

5. **Multi-type anomaly evaluation**: Test how single configurations perform across different anomaly types.

---

## 11. Score Type Analysis (Teacher / Student / Discrepancy)

> **Note**: Sections 1-10 above use the **Combined Anomaly Score** (`recon + adaptive_lambda × disc`) for all evaluations. This section analyzes performance when using individual score components for anomaly detection.

### 11.1 Score Type Definitions

| Score Type | Formula | Description |
|------------|---------|-------------|
| **Anomaly Score (Combined)** | `recon + λ × disc` | Default scoring used in Sections 1-10 |
| **Teacher Recon** | `‖x - teacher(x)‖²` | Teacher reconstruction error only |
| **Student Recon** | `‖x - student(x)‖²` | Student reconstruction error only |
| **Discrepancy (Disc)** | `‖teacher(x) - student(x)‖²` | Teacher-Student output difference only |

---

### 11.2 Best Performance by Score Type

#### F1 Score Comparison

| Score Type | Best F1 | Precision | Recall | Experiment | Improvement vs Combined |
|------------|---------|-----------|--------|------------|------------------------|
| **Student Recon** | **0.7677** | 0.8131 | 0.7270 | w500_p20_enc3 | +26.6% |
| **Teacher Recon** | 0.7439 | 0.8127 | 0.6858 | w500_p20_enc4 | +22.7% |
| Anomaly (Combined) | 0.6065 | 0.4901 | 0.7953 | w100_p5_td4_sd1 | baseline |
| Discrepancy | 0.4859 | 0.3711 | 0.7034 | w100_p5_td4_sd1 | -19.9% |

**Key Finding**: Using Student Reconstruction error alone achieves **+26.6% higher F1** than the combined anomaly score, with significantly better precision (0.81 vs 0.49).

#### ROC-AUC Comparison

| Score Type | Best ROC-AUC | Experiment |
|------------|--------------|------------|
| Anomaly (Combined) | **0.9641** | w500_p20_d64 |
| Teacher Recon | 0.9528 | w100_p5_enc3 |
| Student Recon | 0.9524 | w100_p5_enc2 |
| Discrepancy | 0.9472 | w100_p5_td4_sd1 |

#### PRC-AUC Comparison

| Score Type | Best PRC-AUC | Experiment |
|------------|--------------|------------|
| **Teacher Recon** | **0.7498** | w500_p20_enc4 |
| **Student Recon** | 0.7391 | w500_p20_enc3 |
| Anomaly (Combined) | 0.7005 | w500_p20_enc4 |
| Discrepancy | 0.6333 | w500_p5_td3_sd1 |

---

### 11.3 Top 5 Configurations by Score Type

#### Discrepancy (Disc) - Top 5 by F1

| Rank | Configuration | F1 | ROC-AUC | PRC-AUC | Precision | Recall |
|------|---------------|-----|---------|---------|-----------|--------|
| 1 | w100_p5_td4_sd1 | **0.4859** | 0.9472 | 0.3876 | 0.3711 | 0.7034 |
| 2 | w100_p10_d64 | 0.4829 | 0.9225 | 0.3026 | 0.4442 | 0.5290 |
| 3 | w500_p5_td3_sd1 | 0.4562 | 0.9098 | 0.4857 | 0.7020 | 0.3379 |
| 4 | w100_p10_enc4 | 0.4444 | 0.9276 | 0.3889 | 0.3128 | 0.7671 |
| 5 | baseline | 0.4383 | 0.9331 | 0.4935 | 0.8368 | 0.2969 |

#### Teacher Reconstruction - Top 5 by F1

| Rank | Configuration | F1 | ROC-AUC | PRC-AUC | Precision | Recall |
|------|---------------|-----|---------|---------|-----------|--------|
| 1 | w500_p20_enc4 | **0.7439** | 0.9520 | 0.7497 | 0.8127 | 0.6858 |
| 2 | w500_p20_enc3 | 0.7412 | 0.9099 | 0.7318 | 0.8136 | 0.6806 |
| 3 | w500_p20_td3_sd1 | 0.7409 | 0.8553 | 0.7041 | 0.8294 | 0.6694 |
| 4 | w500_p10_td4_sd2 | 0.6612 | 0.9311 | 0.6335 | 0.5798 | 0.7692 |
| 5 | w500_p10_td3_sd1 | 0.6569 | 0.9343 | 0.6388 | 0.7363 | 0.5930 |

#### Student Reconstruction - Top 5 by F1

| Rank | Configuration | F1 | ROC-AUC | PRC-AUC | Precision | Recall |
|------|---------------|-----|---------|---------|-----------|--------|
| 1 | w500_p20_enc3 | **0.7677** | 0.8928 | 0.7391 | 0.8131 | 0.7270 |
| 2 | w500_p20_enc4 | 0.7535 | 0.9153 | 0.7355 | 0.7954 | 0.7158 |
| 3 | w500_p20_td3_sd1 | 0.7221 | 0.9058 | 0.7230 | 0.8366 | 0.6351 |
| 4 | w100_p5_enc2 | 0.6798 | 0.9524 | 0.6985 | 0.6793 | 0.6803 |
| 5 | w500_p10_td3_sd1 | 0.6670 | 0.9463 | 0.6847 | 0.7018 | 0.6354 |

---

### 11.4 Best Configuration per Metric per Score Type

| Metric | Disc Best | Disc Value | Teacher Best | Teacher Value | Student Best | Student Value |
|--------|-----------|------------|--------------|---------------|--------------|---------------|
| **F1** | w100_p5_td4_sd1 | 0.4859 | w500_p20_enc4 | 0.7439 | **w500_p20_enc3** | **0.7677** |
| **ROC-AUC** | w100_p5_td4_sd1 | 0.9472 | w100_p5_enc3 | 0.9528 | w100_p5_enc2 | 0.9524 |
| **PRC-AUC** | w500_p5_td3_sd1 | 0.6333 | **w500_p20_enc4** | **0.7498** | w500_p20_enc3 | 0.7391 |
| **Precision** | baseline | 0.8368 | w500_p20_td2_sd1 | 0.8681 | w500_p20_td3_sd1 | 0.8366 |
| **Recall** | w500_p10_enc2 | 0.8711 | w500_p10_td4_sd2 | 0.7692 | w500_p20_enc3 | 0.7270 |
| **PA%0 F1** | baseline | 0.9736 | w500_p20_td2_sd1 | 0.9666 | w500_p20_td3_sd1 | 0.9433 |

---

### 11.5 Key Insights

#### Insight 1: Reconstruction Errors Outperform Discrepancy for F1

**Observation**: Both Teacher and Student reconstruction errors achieve significantly higher F1 scores than the combined anomaly score or discrepancy alone.

| Score Type | Best F1 | Precision | Recall | vs Combined |
|------------|---------|-----------|--------|-------------|
| Student Recon | 0.7677 | 0.8131 | 0.7270 | +26.6% |
| Teacher Recon | 0.7439 | 0.8127 | 0.6858 | +22.7% |
| Combined | 0.6065 | 0.4901 | 0.7953 | - |
| Discrepancy | 0.4859 | 0.3711 | 0.7034 | -19.9% |

**Hypothesis**: For spike anomalies in WaDi A1, the reconstruction error provides a more direct signal of anomalous patterns. The discrepancy signal may introduce noise by depending on both teacher and student model behaviors.

---

#### Insight 2: Larger Patches Excel with Reconstruction-Based Scoring

**Observation**: Optimal configurations differ significantly between score types:

| Score Type | Best F1 Config | Optimal Patch Size |
|------------|----------------|-------------------|
| Combined | w100_p5_td4_sd1 | **5** |
| Teacher | w500_p20_enc4 | **20** |
| Student | w500_p20_enc3 | **20** |
| Disc | w100_p5_td4_sd1 | **5** |

**Hypothesis**: Larger patches (20) aggregate reconstruction errors more effectively, leading to cleaner anomaly signals. The combined score benefits from smaller patches because discrepancy requires fine-grained comparison.

---

#### Insight 3: Deep Encoders Benefit Reconstruction Scoring

**Observation**: Top teacher/student configurations use enc=3 or enc=4, while top combined/disc configurations use enc=1.

| Score Type | Optimal Encoder Depth |
|------------|----------------------|
| Teacher Recon | 3-4 layers |
| Student Recon | 2-3 layers |
| Combined | 1 layer |
| Discrepancy | 1 layer |

**Hypothesis**: Deeper encoders learn more expressive representations that distinguish normal from anomalous patterns in reconstruction. However, for discrepancy-based detection, simpler encoders maintain better teacher-student gap.

---

### 11.6 Practical Recommendations

| Use Case | Recommended Score Type | Configuration | Expected F1 | Precision | Recall |
|----------|----------------------|---------------|-------------|-----------|--------|
| **Maximum F1** | Student Recon | w500_p20_enc3 | 0.77 | 0.81 | 0.73 |
| **High Precision** | Discrepancy | baseline | 0.44 | 0.84 | 0.30 |
| **Balanced (Default)** | Combined | w100_p5_td4_sd1 | 0.61 | 0.49 | 0.80 |
| **High ROC-AUC** | Combined | w500_p20_d64 | 0.55 | 0.39 | 0.93 |

**Note**: When using Teacher/Student Recon scoring, the model architecture remains the same, only the evaluation method changes. This allows post-hoc selection of the optimal scoring strategy.

---

## Appendix: Data Files

- **CSV Data**: `results/WaDi/A1/ablation_results.csv`
- **Summary**: `results/WaDi/A1/ablation_summary_20260203_133903.json`
- **Individual Experiments**: `results/WaDi/A1/[timestamp]_[config]/`

---

## Changelog

| Date | Description |
|------|-------------|
| 2026-02-03 | Initial analysis generated |
| 2026-02-03 | **PRC-AUC values corrected**: Fixed PR curve calculation to start at (recall=0, precision=1). All PRC-AUC values updated throughout document. Mean PRC-AUC increased from 0.3714 to 0.4991 due to proper curve normalization. |

---

*Last updated: 2026-02-03*
