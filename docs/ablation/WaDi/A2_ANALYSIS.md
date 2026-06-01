# WaDi A2 Dataset Ablation Study Analysis

**Date**: 2026-02-05
**Total Experiments**: 40
**Anomaly Types**: Multiple (7 attack segments)
**Dataset Features**: 96 features, 172,803 timesteps

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Experimental Setup](#2-experimental-setup)
3. [Overall Performance Statistics](#3-overall-performance-statistics)
4. [Parameter-wise Analysis](#4-parameter-wise-analysis)
5. [Parameter Interaction Effects](#5-parameter-interaction-effects)
6. [Comparison with Baseline Models](#6-comparison-with-baseline-models)
7. [A1 vs A2 Dataset Comparison](#7-a1-vs-a2-dataset-comparison)
8. [Key Insights and Hypotheses](#8-key-insights-and-hypotheses)
9. [Recommendations](#9-recommendations)

---

## 1. Executive Summary

### Key Findings

| Finding | Statistic | Significance |
|---------|-----------|--------------|
| **Best F1 configuration** | w100_p5_td3_sd1 | F1 = 0.5728 |
| **td3_sd1 outperforms all others** | Mean F1 = 0.5345 | +13.3% vs next best |
| **Patch size 20 achieves best recall** | Mean recall = 0.637 | But lower precision |
| **Window size has minimal impact** | w100: F1=0.437 vs w500: F1=0.432 | Not significant |
| **Deep encoders (enc4) improve recall** | Mean recall = 0.644 | Trade-off with precision |

### Performance Range

- **ROC-AUC**: 0.8188 ~ 0.9396 (Mean: 0.8874)
- **PRC-AUC**: 0.1964 ~ 0.6146 (Mean: 0.3579)
- **F1 Score**: 0.2485 ~ 0.5728 (Mean: 0.4365)

### Comparison with A1

| Metric | A1 Best | A2 Best | Difference |
|--------|---------|---------|------------|
| F1 Score | 0.6065 (w100_p5_td4_sd1) | 0.5728 (w100_p5_td3_sd1) | -5.6% |
| ROC-AUC | 0.9641 | 0.9396 | -2.5% |
| Best Decoder | td4_sd1 | td3_sd1 | Different optimal |

---

## 2. Experimental Setup

### Parameter Space

| Parameter | Values Tested | Combinations |
|-----------|---------------|--------------|
| Window Size | 100, 500 | 2 |
| Patch Size | 5, 10, 20 | 3 (5 configs) |
| Architecture | td2_sd1, td3_sd1, td4_sd1, td4_sd2, d64, enc2, enc3, enc4 | 8 |

**Window/Patch Configurations**: w500_p5, w100_p5, w500_p10, w100_p10, w500_p20

**Total Experiments**: 5 window/patch configs × 8 arch variants = 40

### Dataset Characteristics

| Attribute | Value |
|-----------|-------|
| Total Timesteps | 172,803 |
| Features | 96 |
| Train/Test Ratio | 50:50 |
| Train Anomaly Ratio | 7.68% |
| Test Anomaly Ratio | 3.87% |
| Attack Segments | 7 |

---

## 3. Overall Performance Statistics

### Key Metrics Summary

| Metric | Mean | Std | Min | Max | Median |
|--------|------|-----|-----|-----|--------|
| ROC-AUC | 0.8874 | 0.0268 | 0.8188 | 0.9396 | 0.8849 |
| PRC-AUC | 0.3579 | 0.1205 | 0.1964 | 0.6146 | 0.3291 |
| F1 Score | 0.4365 | 0.0836 | 0.2485 | 0.5728 | 0.4239 |
| Precision | 0.4085 | 0.1568 | 0.1463 | 0.8501 | 0.3657 |
| Recall | 0.5629 | 0.1412 | 0.2657 | 0.8232 | 0.5939 |

### Top 10 Experiments

| Rank | Experiment | F1 | Precision | Recall | ROC-AUC | PRC-AUC |
|------|------------|----|-----------|--------|---------|---------|
| 1 | **w100_p5_td3_sd1** | **0.5728** | 0.5411 | 0.6083 | **0.9396** | 0.5752 |
| 2 | w500_p5_td3_sd1 | 0.5715 | 0.6733 | 0.4964 | 0.9134 | **0.6146** |
| 3 | w500_p20_enc4 | 0.5599 | 0.5363 | 0.5856 | 0.9171 | 0.5521 |
| 4 | w500_p20_enc2 | 0.5471 | 0.4834 | 0.6302 | 0.9235 | 0.3851 |
| 5 | w100_p10_td3_sd1 | 0.5351 | 0.5566 | 0.5153 | 0.8575 | 0.4744 |
| 6 | w100_p10_d64 | 0.5316 | 0.4383 | 0.6753 | 0.9115 | 0.3695 |
| 7 | w500_p10_td3_sd1 | 0.5207 | 0.5729 | 0.4773 | 0.8892 | 0.4395 |
| 8 | w500_p20_d64 | 0.5144 | 0.4323 | 0.6349 | 0.9155 | 0.3265 |
| 9 | w100_p5_td4_sd1 | 0.5116 | 0.4465 | 0.5990 | 0.8858 | 0.5179 |
| 10 | w100_p5_enc3 | 0.5051 | 0.4679 | 0.5488 | 0.9134 | 0.3628 |

### Worst 5 Experiments

| Rank | Experiment | F1 | Precision | Recall | ROC-AUC | PRC-AUC |
|------|------------|----|-----------|--------|---------|---------|
| 40 | w100_p5_td2_sd1 | **0.2485** | 0.1463 | 0.8232 | 0.8483 | 0.1964 |
| 39 | w500_p5_td4_sd1 | 0.2863 | 0.1887 | 0.5931 | 0.8728 | 0.2055 |
| 38 | w500_p10_td4_sd2 | 0.3077 | 0.2044 | 0.6218 | 0.8698 | 0.3291 |
| 37 | w100_p5_enc4 | 0.3100 | 0.2010 | 0.6768 | 0.8769 | 0.3418 |
| 36 | w100_p10_enc3 | 0.3114 | 0.3540 | 0.2780 | 0.8636 | 0.2859 |

---

## 4. Parameter-wise Analysis

### 4.1 Window Size (100 vs 500)

| Window | F1 Score | Precision | Recall | ROC-AUC | PRC-AUC |
|--------|----------|-----------|--------|---------|---------|
| 100 | 0.4371 | 0.4120 | 0.5637 | 0.8841 | 0.3649 |
| 500 | 0.4319 | 0.4041 | 0.5587 | 0.8895 | 0.3682 |

**Insight**: Window size has minimal impact on A2 performance, similar to A1. The difference is within noise margin.

**Hypothesis**: A2's multiple attack types don't require longer temporal context. Both windows capture sufficient patterns for detection.

---

### 4.2 Patch Size (5, 10, 20)

| Patch | F1 Score | ROC-AUC | PRC-AUC | Precision | Recall |
|-------|----------|---------|---------|-----------|--------|
| 5 | 0.4289 | 0.8927 | **0.4134** | **0.4497** | 0.5374 |
| 10 | 0.4186 | 0.8761 | 0.3215 | 0.3741 | 0.5460 |
| 20 | **0.4746** | **0.8992** | 0.3645 | 0.3885 | **0.6368** |

**Key Observations**:
1. Patch=20 achieves best F1 (0.4746) and recall (0.6368)
2. Patch=5 maintains best precision (0.4497) and PRC-AUC (0.4134)
3. Patch=10 is worst across all metrics

**Hypothesis**: Unlike A1 (point anomalies), A2's extended attack segments benefit from larger patches that aggregate more temporal information.

---

### 4.3 Architecture Variants

| Variant | F1 Score | ROC-AUC | PRC-AUC | Precision | Recall |
|---------|----------|---------|---------|-----------|--------|
| **td3_sd1** | **0.5345** | 0.8969 | **0.4978** | **0.5419** | 0.5527 |
| d64 | 0.4516 | **0.8967** | 0.3255 | 0.3816 | 0.5826 |
| enc4 | 0.4347 | 0.8996 | 0.3919 | 0.3471 | **0.6439** |
| enc2 | 0.4321 | 0.8848 | 0.3688 | 0.4911 | 0.4840 |
| td4_sd2 | 0.4297 | 0.8797 | 0.3637 | 0.5033 | 0.5073 |
| enc3 | 0.4080 | 0.8960 | 0.3540 | 0.3537 | 0.5247 |
| td2_sd1 | 0.3954 | 0.8702 | 0.3217 | 0.3097 | 0.6296 |
| td4_sd1 | 0.3857 | 0.8751 | 0.3117 | 0.3294 | 0.5608 |

**Key Observations**:
1. **td3_sd1 dominates**: Best F1, PRC-AUC, and precision
2. **td4_sd1 underperforms**: Unlike A1, deeper teacher hurts on A2
3. **enc4 has highest recall** (0.6439) but poor precision

**Hypothesis**: A2's multi-attack nature requires balanced teacher capacity. TD=4 may overfit to specific attack patterns, while TD=3 generalizes better.

---

## 5. Parameter Interaction Effects

### 5.1 Window Size × Patch Size (F1 Score)

| Window | Patch=5 | Patch=10 | Patch=20 |
|--------|---------|----------|----------|
| 100 | 0.4204 | 0.4538 | N/A |
| 500 | 0.4375 | 0.3835 | **0.4746** |

**Key Finding**: w500_p20 achieves best performance. w500_p10 shows notable degradation.

---

### 5.2 Window × Arch Variant (Best F1)

| Architecture | Best w100 | Best w500 | Better Window |
|--------------|-----------|-----------|---------------|
| td3_sd1 | **0.5728** | 0.5715 | w100 (slight) |
| enc4 | 0.3100 | **0.5599** | w500 (large) |
| enc2 | 0.4116 | **0.5471** | w500 |
| d64 | 0.5316 | 0.5144 | w100 |

**Insight**: Deep encoders (enc4) strongly benefit from longer windows, while td3_sd1 is robust across both.

---

## 6. Comparison with Baseline Models

### 6.1 Overall Ranking (All Models)

| Rank | Model | F1 | ROC-AUC | PRC-AUC | Precision | Recall |
|------|-------|-----|---------|---------|-----------|--------|
| 1 | tranad | **0.6059** | 0.7659 | 0.4492 | **0.8138** | 0.4826 |
| 2 | mae_teacher | 0.5898 | 0.8520 | 0.5612 | 0.7480 | 0.4868 |
| 3 | **w100_p5_td3_sd1** | 0.5728 | **0.9396** | 0.5752 | 0.5411 | 0.6083 |
| 4 | mae_tuned | 0.5715 | 0.9134 | **0.6146** | 0.6733 | 0.4964 |
| 5 | **w500_p5_td3_sd1** | 0.5715 | 0.9134 | **0.6146** | 0.6733 | 0.4964 |
| 6 | **w500_p20_enc4** | 0.5599 | 0.9171 | 0.5521 | 0.5363 | 0.5856 |
| 7 | nn_distance | 0.5182 | 0.8801 | 0.4791 | 0.6480 | 0.4318 |
| 8 | anomaly_transformer | 0.4712 | 0.8181 | 0.2708 | 0.4135 | 0.5476 |
| 9 | pca_error | 0.4549 | 0.8415 | 0.2844 | 0.4714 | 0.4396 |
| 10 | mlpmixer | 0.4416 | 0.7873 | 0.2459 | 0.3665 | 0.5554 |

### 6.2 Metric-Specific Leaders

| Metric | Best Model | Score | Type |
|--------|------------|-------|------|
| **F1 Score** | tranad | 0.6059 | Baseline |
| **ROC-AUC** | w100_p5_td3_sd1 | 0.9396 | **Ablation** |
| **PRC-AUC** | mae_tuned / w500_p5_td3_sd1 | 0.6146 | Tie |
| **Precision** | tranad | 0.8138 | Baseline |
| **Recall** | random | 0.9557 | Baseline (trivial) |

### 6.3 Ablation vs Baselines Summary

| Comparison | Ablation Best | Baseline Best | Winner |
|------------|---------------|---------------|--------|
| F1 Score | 0.5728 | 0.6059 | Baseline (+5.5%) |
| ROC-AUC | 0.9396 | 0.9134 | **Ablation (+2.9%)** |
| PRC-AUC | 0.6146 | 0.6146 | Tie |

---

## 7. A1 vs A2 Dataset Comparison

### 7.1 Dataset Characteristics

| Attribute | A1 | A2 |
|-----------|-----|-----|
| Timesteps | 275,000 | 172,803 |
| Train Anomaly % | ~5% | 7.68% |
| Test Anomaly % | ~5% | 3.87% |
| Attack Type | Spike (point-level) | Multiple (7 segments) |

### 7.2 Optimal Configuration Comparison

| Parameter | A1 Optimal | A2 Optimal | Same? |
|-----------|------------|------------|-------|
| Window | 100 | 100 | Yes |
| Patch | 5 | 5 | Yes |
| Teacher Decoder | 4 layers | 3 layers | **No** |
| Student Decoder | 1 layer | 1 layer | Yes |
| Best F1 | 0.6065 | 0.5728 | A1 better |

### 7.3 Performance Comparison

| Metric | A1 Mean | A2 Mean | Difference |
|--------|---------|---------|------------|
| F1 | 0.4336 | 0.4365 | +0.7% |
| ROC-AUC | 0.9224 | 0.8874 | -3.8% |
| PRC-AUC | 0.4991 | 0.3579 | -28.3% |
| Precision | 0.3937 | 0.4085 | +3.8% |
| Recall | 0.5977 | 0.5629 | -5.8% |

### 7.4 Key Differences

1. **A2 has lower ROC-AUC and PRC-AUC** despite similar F1
2. **Optimal teacher depth differs**: TD=4 for A1 vs TD=3 for A2
3. **A2 shows worse worst-case**: Min F1=0.2485 vs A1 Min F1=0.2402
4. **Patch size effect differs**: A2 benefits more from p=20

---

## 8. Key Insights and Hypotheses

### Insight 1: TD=3 is Optimal for Multi-Attack Scenarios

**Observation**: td3_sd1 (F1=0.5345) outperforms td4_sd1 (F1=0.3857) by 38.6%.

**Hypothesis**: A2 contains 7 different attack segments with varying patterns. A shallower teacher (TD=3) maintains better generalization across diverse anomaly types, while deeper teachers (TD=4) may overspecialize.

**Implication**: For datasets with heterogeneous anomaly types, prefer moderate teacher depth.

---

### Insight 2: Larger Patches Benefit Extended Anomalies

**Observation**: Patch=20 achieves best mean F1 (0.4746) vs Patch=5 (0.4289).

**Hypothesis**: A2's attack segments span extended time periods. Larger patches aggregate this temporal information more effectively than fine-grained patches optimized for point anomalies.

**Implication**: Match patch size to expected anomaly duration. A1 spikes → small patches; A2 segments → larger patches.

---

### Insight 3: Ablation Models Excel at Discrimination (ROC-AUC)

**Observation**: Best ablation ROC-AUC (0.9396) exceeds best baseline (0.9134).

**Hypothesis**: The self-distilled MAE architecture provides superior ranking capability, even when threshold selection for F1 optimization is suboptimal.

**Implication**: For applications where ranking matters (e.g., prioritized investigation), ablation models may be preferred over tranad despite lower F1.

---

### Insight 4: Deep Encoders Trade Precision for Recall

**Observation**: enc4 has highest recall (0.6439) but lowest precision (0.3471).

**Hypothesis**: Deeper encoders learn more abstract representations that "trigger" on broader patterns, increasing sensitivity at the cost of specificity.

**Implication**: Use enc4 for high-recall requirements; use td3_sd1 for balanced detection.

---

## 9. Recommendations

### 9.1 Recommended Default Configuration

For general-purpose anomaly detection on WaDi A2:

```
window_size: 100
patch_size: 5
num_teacher_decoder_layers: 3
num_student_decoder_layers: 1
d_model: 128
num_encoder_layers: 1
```

**Expected Performance**: F1 ~ 0.57, Precision ~ 0.54, Recall ~ 0.61

---

### 9.2 Use-Case Specific Recommendations

| Use Case | Configuration | Expected F1 | Precision | Recall | Trade-off |
|----------|--------------|-------------|-----------|--------|-----------|
| **Balanced** | w100_p5_td3_sd1 | 0.57 | 0.54 | 0.61 | Best overall |
| **High Precision** | w500_p5_enc2 | 0.44 | 0.85 | 0.30 | Minimize false alarms |
| **High Recall** | w100_p5_td2_sd1 | 0.25 | 0.15 | 0.82 | Maximize detection |
| **High ROC-AUC** | w100_p5_td3_sd1 | 0.57 | 0.54 | 0.61 | Best discrimination |

---

### 9.3 When to Use Baselines vs Ablation

| Scenario | Recommended | Reasoning |
|----------|-------------|-----------|
| Maximum F1 required | tranad | 5.5% higher F1 |
| Ranking/prioritization | w100_p5_td3_sd1 | 2.9% higher ROC-AUC |
| PRC-AUC critical | mae_tuned or w500_p5_td3_sd1 | Tied at 0.6146 |
| Interpretability | Ablation models | Teacher/student decomposition |

---

## Appendix: Data Files

- **Experiment Results**: `results/WaDi/A2/[timestamp]_[config]/`
- **Baseline Comparison**: `comparison/results/WaDi_A2/results.json`
- **Ablation Summary**: `results/WaDi/A2/ablation_summary_20260205_103120.json`

---

## Changelog

| Date | Description |
|------|-------------|
| 2026-02-05 | Initial analysis generated with 40 experiments |

---

*Last updated: 2026-02-05*
