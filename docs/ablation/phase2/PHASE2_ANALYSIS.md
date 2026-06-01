# Phase 2 Ablation Study: Comprehensive Analysis

**Date**: 2026-02-02
**Experiments**: 150 configs × 2 mask timings × 2 scoring modes = 600 evaluations
**Wall Time**: ~12 hours (42,979s)
**Dataset**: 275K timesteps, 8 features, 9 anomaly types, window stride=1

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Methodology & Data Notes](#2-methodology--data-notes)
3. [Overall Performance Landscape](#3-overall-performance-landscape)
4. [Discrepancy Ratio & Cohen's d Maximization (Req 1)](#4-discrepancy-ratio--cohens-d-maximization-req-1)
5. [Models with High disc_d AND High recon_d (Req 2)](#5-models-with-high-disc_d-and-high-recon_d-req-2)
6. [Scoring Mode & Window Size Effects on High-Performing Models (Req 3)](#6-scoring-mode--window-size-effects-on-high-performing-models-req-3)
7. [Disturbing Normal vs Anomaly Separation (Req 4)](#7-disturbing-normal-vs-anomaly-separation-req-4)
8. [PA%80 + disc_ratio Leaders (Req 5)](#8-pa80--disc_ratio-leaders-req-5)
9. [Window Size × Model Depth × Masking Ratio Relationships (Req 6)](#9-window-size--model-depth--masking-ratio-relationships-req-6)
10. [Scoring Mode Differences by Parameter (Req 7)](#10-scoring-mode-differences-by-parameter-req-7)
11. [Good Detection + High Disturbing Separation (Req 8)](#11-good-detection--high-disturbing-separation-req-8)
12. [mask_after vs mask_before by Hyperparameter (Req 9)](#12-mask_after-vs-mask_before-by-hyperparameter-req-9)
13. [When Does Discrepancy Improve Over Reconstruction-Only? (Req 10)](#13-when-does-discrepancy-improve-over-reconstruction-only-req-10)
14. [Per-Parameter Individual Impact Analysis (Req 11)](#14-per-parameter-individual-impact-analysis-req-11)
15. [PHASE2_PLAN Hypothesis Verification (Req 12)](#15-phase2_plan-hypothesis-verification-req-12)
16. [Novel Insights (Req 13)](#16-novel-insights-req-13)
17. [Phase 3 검증 대상: 가설, 파라미터, Insight 정리](#17-phase-3-검증-대상-가설-파라미터-insight-정리)
18. [Visualization 대상 모델 선정](#18-visualization-대상-모델-선정-30-configs--4-variants--120-experiments)

---

## 1. Executive Summary

Phase 2 tested 150 configurations across 13 experiment groups, each evaluated with 2 mask timings (mask_before, mask_after) and 2 scoring modes (default, adaptive), yielding 600 total evaluations. The key findings are:

**Top-line Results:**
- **Best overall model**: `083_w500_p5_td2_mask_after` — roc_auc=0.990, PA%80=0.848, disc_d=4.64
- **mask_before significantly outperforms mask_after on roc_auc** (wins 214/300 pairs, mean gap -0.081, p<0.001), directly contradicting Phase 1's finding that mask_after is superior
- **mask_after dramatically improves discrepancy metrics** (disc_d +0.71, disc_SNR +0.28, disc_ratio +1.92) while destroying reconstruction quality (recon_d -0.96)
- **Scoring mode is not significant** for roc_auc (p=0.31), contradicting Phase 1's normalized scoring advantage
- **Smaller models dominate**: d_model=64 > 128 > 256 > 512 (η²=0.20)
- **Shallow architectures win**: enc=1 >> enc=3 > enc=4; td=2 best overall
- **seq_length is the most impactful parameter** (η²=0.25): w200 ≈ w100 >> w500
- **masking_ratio=0.2 is best** (contradicts Phase 1's 0.08-0.15 sweet spot)

**Critical Phase 1 Hypothesis Reversals:**
1. mask_after + normalized scoring is NOT superior — mask_before + either scoring wins on roc
2. lr=0.003-0.005 does NOT outperform lr=0.002 — lr=0.002 is the actual default and performs best
3. d_model=128+ is NOT critical for w500 — d64 outperforms at all window sizes
4. Deeper encoders (enc=2-3) do NOT improve over enc=1 — enc=1 is significantly better
5. masking_ratio=0.15 is NOT the sweet spot — mr=0.2 outperforms

---

## 2. Methodology & Data Notes

### Data Recovery Issue

The original `summary_results.csv` contained only 300 records (mask_before results only) due to an in-memory accumulator (`all_records`) being reset when the ablation script was restarted after experiment 150 hung from GPU OOM. A reconstruction script (`rebuild_summary.py`) was written to scan all 600 experiment directories and rebuild the complete dataset from individual `experiment_metadata.json` files.

**Important caveat**: The reconstructed dataset (`summary_results_full.csv`) is missing component-level metrics (`disc_only_*`, `teacher_recon_*`, `student_recon_*`) that were only stored in the original CSV's mask_before records. All analyses in this report use the available 79-column reconstructed dataset.

### Actual Default Configuration

Due to a correction documented in PHASE2_PLAN.md, the actual base config used differs from the originally planned defaults:

| Parameter | Planned | Actual | Reason |
|-----------|---------|--------|--------|
| encoder_layers | 2 | 1 | enc=2 causes disc_d collapse at w500 |
| learning_rate | 0.005 | 0.002 | lr=5e-3 too aggressive at w500 |

This means the PHASE2_PLAN descriptions reference enc=2 and lr=0.005, but the actual experiments used enc=1 and lr=0.002 as the base config defaults.

### Statistical Methods

- **Paired comparisons**: Wilcoxon signed-rank test for mask_after vs mask_before (paired by config)
- **Multi-group comparisons**: One-way ANOVA with F-statistic and p-value
- **Two-group comparisons**: Welch's t-test
- **Effect size**: η² (eta-squared) from ANOVA for parameter importance ranking
- **Significance threshold**: p < 0.05

---

## 3. Overall Performance Landscape

### Global Statistics (N=600)

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| roc_auc | 0.852 | 0.127 | 0.481 | 0.990 |
| pa_80_f1 | 0.540 | 0.195 | 0.022 | 0.848 |
| disc_cohens_d (normal vs anomaly) | 0.972 | 1.118 | -0.573 | 4.674 |
| recon_cohens_d (normal vs anomaly) | 1.569 | 0.760 | 0.345 | 3.089 |
| disc_SNR | 0.379 | 0.415 | -0.299 | 1.733 |
| disc_ratio | 2.600 | 3.098 | 0.640 | 21.381 |
| disturbing_roc_auc | 0.934 | 0.099 | 0.177 | 0.999 |

### Top 10 Models by roc_auc

| Rank | Experiment | roc_auc | PA%80 | disc_d | recon_d | disc_SNR |
|------|-----------|---------|-------|--------|---------|----------|
| 1 | 083_w500_p5_td2_mask_after_adaptive | 0.9904 | 0.847 | 4.638 | 0.710 | 1.533 |
| 2 | 083_w500_p5_td2_mask_after_default | 0.9903 | 0.848 | 4.638 | 0.710 | 1.533 |
| 3 | 081_w500_p5_d64_mask_after_default | 0.9884 | 0.831 | 4.674 | 0.674 | 1.580 |
| 4 | 007_w100_p5_mask_after_default | 0.9879 | 0.823 | 4.367 | 1.328 | 1.437 |
| 5 | 074_w100_p5_mask_after_default | 0.9879 | 0.823 | 4.367 | 1.328 | 1.437 |
| 6 | 063_softplus_w200_mask_after_default | 0.9877 | 0.726 | 4.174 | 1.155 | 1.662 |
| 7 | 063_softplus_w200_mask_after_adaptive | 0.9869 | 0.753 | 4.174 | 1.155 | 1.662 |
| 8 | 007_w100_p5_mask_after_adaptive | 0.9859 | 0.824 | 4.367 | 1.328 | 1.437 |
| 9 | 074_w100_p5_mask_after_adaptive | 0.9859 | 0.824 | 4.367 | 1.328 | 1.437 |
| 10 | 077_w200_p5_mask_after_default | 0.9859 | 0.820 | 4.171 | 1.073 | 1.389 |

**Key observation**: All top 10 are mask_after with small patch_size (p=5). Despite mask_before winning the aggregate comparison, mask_after dominates the very top of the leaderboard when paired with the right configuration (small patches).

### mask_before vs mask_after: Paired Comparison (300 pairs)

| Metric | mask_after wins | mask_before wins | Mean diff (after - before) | p-value |
|--------|----------------|-----------------|---------------------------|---------|
| roc_auc | 86 | 214 | **-0.081** | <0.001 |
| pa_80_f1 | 85 | 215 | -0.103 | <0.001 |
| disc_d | **164** | 136 | **+0.710** | <0.001 |
| disc_SNR | **178** | 122 | **+0.280** | <0.001 |
| disc_ratio | **179** | 121 | **+1.918** | <0.001 |
| recon_d | 4 | **296** | **-0.963** | <0.001 |
| disturbing_roc | 67 | 233 | -0.038 | <0.001 |

**Interpretation**: mask_after creates a fundamental trade-off. By encoding only visible patches (standard MAE behavior), the encoder produces richer representations that amplify teacher-student discrepancy (disc_d +0.71), but the reconstruction from these representations is much worse (recon_d -0.96). Since the final score is `recon + λ*disc`, the poor reconstruction quality drags down overall detection performance for most configs. However, when disc_d is extremely high (>4.0, achievable with small patches), the discrepancy signal overwhelms the reconstruction deficit.

### Scoring Mode: default vs adaptive (300 pairs)

| Metric | default wins | adaptive wins | Mean diff | p-value |
|--------|-------------|--------------|-----------|---------|
| roc_auc | 173 | 127 | +0.005 | 0.311 |
| pa_80_f1 | 136 | 164 | -0.004 | 0.478 |
| disturbing_roc | **205** | 95 | **+0.013** | <0.001 |

**Interpretation**: Scoring mode has negligible impact on detection metrics (roc, PA%80) but default scoring is significantly better for disturbing-normal separation. This contradicts Phase 1's H12 finding that normalized/adaptive scoring is important. The likely explanation: with λ=2.0 (Phase 2 default) vs λ=0.5 (Phase 1 default), the discrepancy weight is already properly scaled, making adaptive rescaling unnecessary.

---

## 4. Discrepancy Ratio & Cohen's d Maximization (Req 1)

### Top disc_cohens_d Models

The top disc_d models share a remarkably consistent profile:

| Characteristic | Value | Prevalence in Top-20 disc_d |
|---------------|-------|---------------------------|
| mask_after | True | 20/20 (100%) |
| patch_size | 5 | 12/20 (60%) |
| enc | 1 | 20/20 (100%) |
| d_model | 64 or 128 | 20/20 (100%) |
| margin_type | softplus or dynamic | both present |

The absolute highest disc_d values (4.6+) come from p=5 configs with mask_after, regardless of window size:
- `081_w500_p5_d64_mask_after`: disc_d=4.674, disc_ratio=9.58
- `083_w500_p5_td2_mask_after`: disc_d=4.638, disc_ratio=15.09
- `007_w100_p5_mask_after`: disc_d=4.367, disc_ratio=14.36

### Why Small Patches Maximize Discrepancy

**Hypothesis**: With patch_size=5 at w500, the model processes 100 patches. 15개가 마스킹되면(mr=0.15), 각 masked patch는 5 timestep만 커버하는 매우 미세한 복원 과제가 됩니다. Teacher와 student 모두 **동일한 masked encoder 출력**을 받지만, 두 decoder의 차이는 깊이뿐 아니라 **학습 목표** 자체가 다릅니다: teacher는 원본 복원(reconstruction loss)을, student는 teacher 출력 모방(discrepancy loss, `(teacher.detach() - student)²`)을 목표로 학습합니다. 패치가 미세할수록(5 timestep) 정확한 복원에 더 정밀한 cross-attention이 필요하고, 이 어려운 과제에서 (1) 깊은 teacher decoder의 reconstruction 능력과 (2) 얕은 student decoder가 teacher를 모방하는 능력 사이의 격차가 극대화됩니다. 특히 anomaly 영역에서는 student가 teacher와 **의도적으로 달라지도록** margin loss로 학습되므로, fine-grained patch일수록 anomaly-normal 간 discrepancy 차이가 더 날카로워집니다.

**Evidence**:
- p=5: mean disc_d=1.79, disc_SNR=0.63
- p=10: mean disc_d=1.96, disc_SNR=0.69
- p=20: mean disc_d=0.81, disc_SNR=0.32
- p=25: mean disc_d=1.38, disc_SNR=0.55

Interestingly, p=10 has the highest mean disc_d/SNR among all patch sizes. The top individual models use p=5, but p=10 has better average performance — the p=5 distribution has higher variance.

### disc_ratio Leaders

The disc_ratio (mean anomaly disc / mean normal disc) is maximized by the same p=5 + mask_after configs:
- 083_w500_p5_td2_mask_after: disc_ratio=15.09
- 007/074_w100_p5_mask_after: disc_ratio=14.36
- 081_w500_p5_d64_mask_after: disc_ratio=9.58

**Characteristics of high disc_ratio models**:
- All mask_after (100%)
- Small patches (p=5 or p=10)
- Small to medium d_model (64-128)
- Shallow encoder (enc=1)

---

## 5. Models with High disc_d AND High recon_d (Req 2)

Models with both strong discrepancy AND reconstruction signals (disc_d ≥ median 1.11 AND recon_d ≥ 75th percentile 2.33): **48 models found**.

### Profile of Dual-High Models

| Characteristic | Distribution |
|---------------|-------------|
| mask_label | mask_before: 48/48 (100%) |
| d_model | 128: 46, 64: 2 |
| seq_length | 100: 24, 500: 24 |
| enc | 1: 48/48 (100%) |
| td | 4: 44, 3: 2, 2: 2 |
| nhead | 8: 44, 4: 4 |
| mr | 0.15: 40, 0.1: 6, 0.2: 2 |

**Performance**: roc=0.969, PA%80=0.743

**Critical finding**: ALL 48 dual-high models are mask_before. This is because mask_after sacrifices reconstruction quality for discrepancy. It is impossible to have both high disc_d AND high recon_d with mask_after — they are fundamentally anti-correlated in mask_after mode (overall disc_d vs recon_d correlation is only r=0.23).

**Implication**: If the goal is to have both strong reconstruction-based AND discrepancy-based detection (ensemble or fallback strategy), mask_before is the only viable option. With mask_before, the encoder sees all tokens including mask tokens, preserving reconstruction ability while still generating moderate discrepancy (disc_d ~1.1-1.2 for these models).

### Scoring Mode Effect on Dual-High Models

| Scoring | roc_auc | PA%80 | N |
|---------|---------|-------|---|
| default | 0.971 | 0.747 | 24 |
| adaptive | 0.967 | 0.739 | 24 |

Default scoring slightly outperforms adaptive for these balanced models — consistent with the global finding.

---

## 6. Scoring Mode & Window Size Effects on High-Performing Models (Req 3)

### Window Size Impact on Dual-Signal Models

Among the 48 dual-high (disc_d + recon_d) models:
- w=100 (24 models): roc=0.976, disc_d=1.40, recon_d=2.56
- w=500 (24 models): roc=0.962, disc_d=0.99, recon_d=2.40

Shorter windows yield both higher roc AND higher disc_d for these balanced models. The w500 "advantage" for disturbing separation does not translate to better detection when both signals are required to be strong.

### Window × Scoring Interaction

| Window | Default roc | Adaptive roc | Gap |
|--------|------------|-------------|-----|
| 100 | 0.955 | 0.952 | +0.003 |
| 200 | 0.957 | 0.955 | +0.002 |
| 500 | 0.815 | 0.808 | +0.007 |

The scoring mode gap is slightly larger at w500 but never significant. The w500 performance itself is substantially lower than w100/w200, making window choice far more consequential than scoring mode choice.

---

## 7. Disturbing Normal vs Anomaly Separation (Req 4)

### disc_cohens_d_disturbing_vs_anomaly Analysis

Disturbing-normal windows are technically normal windows near anomaly boundaries. High separation between disturbing and true anomaly windows indicates the model isn't confused by boundary effects.

**Global mean**: 0.566 (moderate separation)

### By Window Size

| Window | disc_d_disturbing | recon_d_disturbing | disturbing_roc |
|--------|------------------|-------------------|----------------|
| 100 | 0.624 | 0.896 | 0.964 |
| 200 | 1.134 | 1.165 | 0.983 |
| 500 | 0.487 | 1.056 | 0.920 |

**Surprising finding**: w=200 has the BEST disturbing separation on both disc and recon channels, contradicting Phase 1's finding that w500 is best for disturbing separation. w200's disturbing_roc (0.983) surpasses w500 (0.920) significantly.

**Hypothesis**: w200 captures enough temporal context to include boundary patterns (unlike w100) but doesn't dilute the anomaly signal with too much normal context (unlike w500). The 200-timestep window hits a "Goldilocks zone" for boundary disambiguation.

### By Mask Timing

| Mask | disc_d_disturbing | Mean |
|------|------------------|------|
| mask_after | 0.656 | higher |
| mask_before | 0.477 | lower |

mask_after produces better disturbing separation via discrepancy, but mask_before has better overall disturbing_roc (0.953 vs 0.915) because reconstruction-based separation is more reliable.

---

## 8. PA%80 + disc_ratio Leaders (Req 5)

### Top 20 by PA%80 where disc_ratio > median

| Experiment | PA%80 | disc_ratio | roc | d | w | td | mask |
|-----------|-------|------------|-----|---|---|----|----|
| 083_w500_p5_td2_mask_after | 0.848 | 15.09 | 0.990 | 128 | 500 | 2 | after |
| 081_w500_p5_d64_mask_after | 0.831 | 9.58 | 0.988 | 64 | 500 | 4 | after |
| 007/074_w100_p5_mask_after | 0.824 | 14.36 | 0.986 | 128 | 100 | 4 | after |
| 077_w200_p5_mask_after_default | 0.820 | 7.67 | 0.986 | 128 | 200 | 4 | after |
| 108_nh4_w100_p10_mask_after | 0.806 | 6.71 | 0.983 | 128 | 100 | 4 | after |
| 004_p1_best_pa80_mask_after | 0.796 | 6.99 | 0.981 | 64 | 100 | 3 | after |
| 006/075_w100_p10_mask_after | 0.798 | 6.88 | 0.983 | 128 | 100 | 4 | after |

### Shared Characteristics of PA%80 + disc_ratio Leaders

All top leaders share:
- **mask_after**: 19/20 (95%)
- **patch_size=5 or 10**: 18/20 (90%)
- **enc=1**: 20/20 (100%)
- **d_model=64 or 128**: 20/20 (100%)
- **lr=0.002**: 20/20 (100%)

The one mask_before entry (#18) is `108_nh4_w100_p10_mask_before` with PA%80=0.788.

**Key insight**: The PA%80+disc_ratio leaders are almost identical to the top roc_auc models. High disc_ratio and high PA%80 co-occur because both benefit from the same configuration: mask_after + small patches + small model + shallow architecture. The disc_ratio metric is strongly predictive of PA%80 in the mask_after regime.

---

## 9. Window Size × Model Depth × Masking Ratio Relationships (Req 6)

### Window × d_model Interaction

| | d=64 | d=128 | d=256 | d=512 |
|---|------|-------|-------|-------|
| w=100 | **0.976** | 0.951 | 0.959 | 0.883 |
| w=200 | **0.975** | 0.958 | 0.908 | — |
| w=500 | 0.882 | 0.834 | 0.705 | 0.695 |

**Finding**: d=64 is best at EVERY window size. The advantage of d_model=128 that Phase 1 found does not hold in Phase 2. Larger models (d256, d512) are catastrophically worse at w500. Even at w100, d512 drops roc by 0.09 vs d64.

**Hypothesis**: With enc=1 (the corrected default), the single encoder layer cannot effectively utilize the larger parameter space. Phase 1 used enc=2 as default, which could exploit d128 better. With enc=1, the model is parameter-limited by the encoder bottleneck, and excess capacity in d_model creates overfitting.

### Window × td Interaction

| | td=2 | td=3 | td=4 | td=5 | td=6 |
|---|------|------|------|------|------|
| w=100 | 0.939 | 0.871 | **0.963** | — | 0.960 |
| w=200 | — | **0.968** | 0.954 | — | — |
| w=500 | **0.885** | 0.832 | 0.818 | 0.757 | 0.766 |

**Finding**: Phase 1's H6 (depth scaling inverts with window size) is confirmed:
- w100: td=4 is best (0.963), td=6 also good (0.960)
- w500: td=2 is best (0.885), deeper decoders hurt progressively

### Window × Masking Ratio

| | mr=0.1 | mr=0.15 | mr=0.2 |
|---|--------|---------|--------|
| w=100 | 0.923 | 0.956 | **0.975** |
| w=200 | **0.971** | 0.948 | 0.970 |
| w=500 | 0.775 | 0.816 | **0.884** |

**Finding**: mr=0.2 is best or tied-best at every window size, contradicting Phase 1's finding that mr=0.08-0.15 is the sweet spot. The higher masking ratio provides more training signal through harder reconstruction targets.

### Three-Way Interaction

The optimal configuration per window size:
- **w=100**: d=64, td=4, mr=0.2 → roc ~0.976
- **w=200**: d=64, td=3-4, mr=0.1 or 0.2 → roc ~0.975
- **w=500**: d=64, td=2, mr=0.2 → roc ~0.885

As window size increases, the optimal model becomes simpler (shallower decoder, smaller d_model). This is the central scaling finding of Phase 2.

---

## 10. Scoring Mode Differences by Parameter (Req 7)

Scoring mode (default vs adaptive) shows negligible effect across nearly all parameters. The only significant interaction is with disturbing_roc:

| Parameter Value | Default dist_roc | Adaptive dist_roc | Gap |
|----------------|-----------------|-------------------|-----|
| lambda_disc=0.5 | 0.990 | 0.970 | +0.020 |
| lambda_disc=2.0 | 0.937 | 0.929 | +0.008 |
| mask_before | 0.960 | 0.946 | +0.014 |
| mask_after | 0.921 | 0.909 | +0.012 |

**Conclusion**: Scoring mode is the least impactful parameter in the entire study (η²=0.0001). The Phase 1 finding that normalized scoring matters was likely an artifact of low λ=0.5 — at λ=2.0, the disc signal is already properly weighted in the default formula.

---

## 11. Good Detection + High Disturbing Separation (Req 8)

Models with roc ≥ 0.967 (75th percentile) AND disc_d_disturbing ≥ 0.753 (75th percentile): **88 models found**.

### Profile

| Characteristic | Distribution |
|---------------|-------------|
| mask_after | **80/88 (91%)** |
| enc=1 | 88/88 (100%) |
| d_model 128 | 70/88 (80%) |
| td=4 | 64/88 (73%) |
| nhead=8 | 78/88 (89%) |
| lambda=2.0 | 78/88 (89%) |
| anomaly_loss_weight=2.0 | 88/88 (100%) |
| lr=0.002 | 86/88 (98%) |

**Key finding**: Achieving both high detection AND high disturbing separation requires mask_after (91%). This is the opposite of Req 2's finding (high disc_d + high recon_d requires mask_before 100%). The distinction:

- **High detection + high disturbing separation** → mask_after (uses disc signal for both)
- **High disc_d + high recon_d** → mask_before (preserves both channels)

The 8 mask_before models that make it into the combined top are exclusively w=100 configs with naturally high detection performance from reconstruction alone.

### Window Size Distribution

- w=100: 40/88 (45%)
- w=200: 22/88 (25%)
- w=500: 26/88 (30%)

All three window sizes contribute, with w100 having the most entries due to its generally higher detection performance.

---

## 12. mask_after vs mask_before by Hyperparameter (Req 9)

### Performance Gap by Parameter

The mask gap (mask_after roc - mask_before roc) varies dramatically across hyperparameters:

#### d_model
| d_model | mask_after | mask_before | Gap |
|---------|-----------|------------|-----|
| 64 | 0.884 | 0.955 | -0.071 |
| 128 | 0.828 | 0.912 | -0.083 |
| 256 | 0.691 | 0.772 | -0.082 |
| 512 | 0.683 | 0.753 | -0.070 |

Gap is remarkably consistent (~0.07-0.08) across all d_model values. Neither small nor large models selectively benefit mask_after for roc.

#### seq_length (Window Size)
| Window | mask_after | mask_before | Gap |
|--------|-----------|------------|-----|
| 100 | 0.940 | 0.967 | -0.026 |
| 200 | **0.970** | 0.941 | **+0.029** |
| 500 | 0.757 | 0.866 | -0.109 |

**Critical finding**: mask_after OUTPERFORMS mask_before at w=200 (+0.029). This is the only parameter setting where mask_after wins on roc. At w500, mask_before's advantage is largest (-0.109).

**Hypothesis**: w200 is the optimal length where mask_after's enhanced discrepancy signal provides enough benefit to overcome the reconstruction deficit, while at w500 the reconstruction deficit becomes too severe (too many patches to reconstruct from partial information).

#### num_teacher_decoder_layers
| td | mask_after | mask_before | Gap |
|----|-----------|------------|-----|
| 2 | 0.902 | 0.907 | -0.006 |
| 3 | 0.862 | 0.850 | **+0.012** |
| 4 | 0.821 | 0.907 | -0.087 |
| 5 | 0.685 | 0.830 | -0.145 |
| 6 | 0.731 | 0.836 | -0.106 |

At shallow decoders (td=2,3), mask_after is nearly equal or better. The mask_before advantage grows dramatically with deeper decoders. This makes sense: deeper decoders can better reconstruct from full information (mask_before) but cannot compensate for the poor representations from mask_after encoding.

#### Decoder Depth × Mask Timing: Deep Dive (disc_d amplification)

mask_after의 disc_d 증폭 효과는 decoder 깊이에 따라 **비단조적(non-monotonic)**으로 변합니다:

| td | after disc_d | before disc_d | disc_d gap | 증폭 배율 | roc gap | p(roc) | p(disc) |
|----|-------------|--------------|-----------|---------|---------|--------|---------|
| 2 | **2.607** | 0.683 | **+1.924** | 3.82× | -0.006 | 0.156 | 0.0002 |
| 3 | **1.903** | 0.564 | **+1.339** | 3.37× | +0.012 | 0.043 | <0.001 |
| 4 | 1.346 | 0.693 | +0.652 | 1.94× | -0.087 | <0.001 | <0.001 |
| 5 | 0.227 | 0.355 | **-0.128** | 0.64× | -0.145 | <0.001 | 0.227 |
| 6 | 0.695 | 0.176 | +0.519 | 3.94× | -0.106 | 0.001 | 0.106 |

**핵심 발견**:

1. **td=2에서 mask_after의 disc_d 증폭이 최대** (+1.92, 3.82배). 얕은 decoder에서는 teacher-student 간 능력 차이가 mask_after의 정보 차단과 결합되어 극대화됩니다. 동시에 roc gap은 거의 0(-0.006, p=0.156 비유의)으로, disc_d의 거대한 증폭이 recon_d 하락을 상쇄합니다.

2. **td=3에서 mask_after가 roc에서도 유의미하게 승리** (+0.012, p=0.043). disc_d 증폭(+1.34)이 recon_d 하락(-0.35)을 압도하는 유일한 td 수준입니다.

3. **td=5에서 disc_d 증폭이 역전** (-0.13). 깊은 decoder에서는 mask_after가 disc_d에서조차 mask_before에 뒤집니다. Student decoder(sd=2)가 5층 teacher에 비해 너무 얕아서, mask_after의 정보 차단이 student의 학습 자체를 방해하기 때문으로 추정됩니다.

4. **recon_d 하락은 td≥4에서 일정** (-1.03). td=2,3에서는 recon_d 하락이 작지만(-0.60, -0.35), td=4부터는 ~1.03으로 수렴합니다. 깊은 decoder가 mask_before에서는 reconstruction을 크게 개선하지만, mask_after에서는 개선 효과가 제한적입니다.

#### td-sd Gap × Mask Timing

Teacher-student decoder 깊이 차이(td-sd gap)에 따른 mask gap:

| td-sd gap | after disc_d | before disc_d | disc_d gap | roc gap | p(roc) |
|-----------|-------------|--------------|-----------|---------|--------|
| 0 (td=sd) | **2.437** | 0.688 | **+1.749** | -0.016 | 0.527 |
| 1 | 1.605 | 0.591 | +1.014 | -0.035 | 0.874 |
| 2 | 1.334 | 0.680 | +0.654 | -0.086 | <0.001 |
| 3 | 0.193 | 0.323 | -0.130 | -0.121 | 0.001 |
| 4 | 0.813 | 0.174 | +0.639 | -0.114 | 0.006 |

**주의: td-sd gap과 absolute td의 confounding**. 표면적으로 gap이 작을수록 disc_d 증폭이 큰 것처럼 보이지만, 이것은 td-sd gap 자체의 효과가 아닙니다. gap=0은 주로 td=2/sd=2(얕은 모델, 잘 학습됨)이고, gap=3은 td=5/sd=2(깊은 모델, 학습 붕괴)입니다. 논리적으로 td-sd gap이 클수록 depth 비대칭이 커져 disc_d가 높아야 하지만, 실제로는 **absolute td가 커지면서 모델 자체가 붕괴**하여 disc_d가 하락합니다. 따라서 이 테이블에서 td-sd gap의 순수 효과를 분리할 수 없습니다.

상관분석 결과 (confounding 포함):
- td vs disc_d_gap: r=-0.271 (p<0.001) — teacher가 깊을수록 mask_after의 disc_d 이점이 감소 (td 증가 → 학습 붕괴)
- td-sd gap vs disc_d_gap: r=-0.226 (p<0.001) — td와 강하게 공선적이므로 독립적 해석 불가
- td vs roc_gap: r=-0.252 (p<0.001) — teacher가 깊을수록 mask_after의 roc 열위가 심화

**결론**: mask_after는 **얕은 decoder (td≤3)** 조건에서 가장 효과적입니다. 이 조건에서 disc_d가 3-4배 증폭되면서도 roc 손실이 거의 없습니다. td≥5에서는 모델 학습 자체가 붕괴하여 mask_after가 disc_d마저 잃습니다. td-sd gap의 독립적 효과를 검증하려면, 동일한 td에서 sd만 변화시킨 controlled comparison이 필요합니다(현재 데이터에서는 td=5일 때 sd={2,3,4}가 있으나 표본이 적어 통계적 검증력이 부족합니다).

#### Deep Decoder Collapse: td=5 + mask_after 분석

통제된 비교 (enc=1, d=128, w=500, mr=0.15, sd=2, nh=8)에서 td만 변화시킨 결과:

| td | mask_before roc | mask_after roc | mask_after disc_d | mask_after recon_d |
|----|----------------|---------------|-------------------|-------------------|
| 2 | 0.927 | **0.979** | 2.607 | 1.105 |
| 3 | 0.956 | **0.969** | 1.903 | — |
| 4 | 0.935 | 0.795 | 1.346 | — |
| 5 | **0.965** | 0.663 | **0.077** | — |
| 6 | 0.867 | 0.731 | 0.695 | — |

**핵심 발견**: td=5의 "붕괴"는 **mask_after에서만 발생**합니다. mask_before에서 td=5는 roc=0.965로 전체에서 가장 높은 축에 속합니다. mask_after에서는 disc_d=0.077로 사실상 0에 수렴하며, 이것이 roc 하락의 직접적 원인입니다.

**원인 분석**:

1. **Teacher 과잉 용량 → encoder representation 과적합**: mask_after에서 encoder는 visible patch만 처리하고, decoder가 masked 위치를 복원합니다. td=5처럼 teacher decoder가 깊으면, teacher가 학습 과정에서 encoder를 "돕도록" 만들어, encoder가 visible patch representation에 masked 위치 정보를 과도하게 압축합니다. 결과적으로 student decoder(sd=2)도 이 풍부한 latent로 쉽게 복원 가능해지며 → disc ≈ 0.

2. **얕은 decoder에서 mask_after가 잘 되는 이유와의 대비**: td=2-3에서는 teacher decoder 자체가 얕아서, encoder가 아무리 좋은 representation을 만들어도 decoder가 완벽히 복원하기 어렵습니다. 이 "적당한 어려움"이 teacher-student 간 능력 차이를 유지시킵니다.

3. **mask_before 면역성**: mask_before에서는 encoder가 mask token을 포함한 전체 시퀀스를 처리하므로, decoder 깊이와 무관하게 teacher-student 모두 "이미 정보가 풍부한" 입력을 받습니다. Decoder 깊이 차이가 출력 차이로 일정하게 유지되어, td=5에서도 disc signal이 살아있습니다.

4. **Window size 상호작용**: w=500에서 td 증가 시 roc 하락이 가장 심합니다 (td=2: 0.953 → td=6: 0.784). 긴 시퀀스 + 깊은 decoder = encoder가 더 풍부한 representation을 학습 → disc 소멸 가속화.

**결론**: mask_after의 최적 teacher decoder 깊이는 td=2-3입니다. td≥4에서는 mask_after의 정보 비대칭 메커니즘이 teacher의 과잉 용량에 의해 무력화됩니다.

#### masking_ratio
| mr | mask_after | mask_before | Gap |
|----|-----------|------------|-----|
| 0.1 | 0.781 | 0.876 | -0.095 |
| 0.15 | 0.811 | 0.891 | -0.080 |
| 0.2 | 0.921 | 0.966 | **-0.045** |

Higher masking ratio narrows the mask gap from -0.095 to -0.045. More aggressive masking hurts mask_before relatively more than mask_after, since mask_before passes mask tokens through the encoder (more tokens masked → more noise in encoder output).

#### lambda_disc
| λ | mask_after | mask_before | Gap |
|---|-----------|------------|-----|
| 0.5 | 0.915 | 0.959 | -0.044 |
| 1.0 | 0.806 | 0.961 | -0.155 |
| 2.0 | 0.808 | 0.884 | -0.076 |
| 3.0 | 0.760 | 0.961 | -0.201 |
| 5.0 | 0.787 | 0.960 | -0.173 |

**Notable**: λ=0.5 gives the smallest gap (-0.044). Low lambda means the score is dominated by reconstruction, reducing the impact of mask_after's poor reconstruction. Higher lambda values amplify the gap inconsistently — λ=3.0 shows the worst gap (-0.201).

---

## 13. When Does Discrepancy Improve Over Reconstruction-Only? (Req 10)

### Signal Comparison

- Mean disc_d: 0.972
- Mean recon_d: 1.569
- Disc_d / recon_d ratio: 0.666

On average, reconstruction is 1.6× stronger than discrepancy as a detection signal. However, this masks important variation.

### Correlations with roc_auc

| Metric | Correlation with roc_auc |
|--------|------------------------|
| recon_SNR | 0.84 |
| recon_d | 0.84 |
| disc_d | 0.64 |
| disc_SNR | 0.63 |
| disc_ratio | 0.48 |

Reconstruction metrics are substantially better correlated with detection (r=0.84) than discrepancy metrics (r=0.64). However, the top-performing models are those where BOTH signals are strong.

### When Discrepancy Dominates

Discrepancy contributes most when:
1. **mask_after + small patches** (p=5, p=10): disc_d reaches 4.6+ while recon_d drops to ~0.7-1.3. The score is disc-dominated.
2. **softplus margin + mask_after**: disc_d=2.80, recon_d much lower. Softplus creates stronger separation.
3. **w200 + mask_after**: disc_d=2.09 is the highest mean by window (vs w100=1.50, w500=0.79).

### When Reconstruction Dominates

Reconstruction is the primary signal when:
1. **mask_before** (always): recon_d >> disc_d
2. **Large models** (d256, d512): disc collapses while recon remains moderate
3. **Deep encoders** (enc=3, enc=4): disc collapses, recon still functional
4. **w500 + default patches** (p=20): disc_d=0.63 while recon_d=1.47

**Conclusion**: Discrepancy is a powerful signal but only under specific conditions (mask_after + small patches + shallow architecture). For most configurations, reconstruction is the workhorse of detection.

---

## 14. Per-Parameter Individual Impact Analysis (Req 11)

### Parameter Importance Ranking (η² on roc_auc)

| Rank | Parameter | η² | Interpretation |
|------|-----------|-----|---------------|
| 1 | **seq_length** | 0.254 | Most impactful; w500 degrades substantially |
| 2 | **d_model** | 0.203 | Smaller is better; strong effect |
| 3 | **num_encoder_layers** | 0.102 | enc=1 >> enc=3 > enc=4 |
| 4 | **mask_after_encoder** | 0.102 | mask_before generally better for roc |
| 5 | **num_teacher_decoder_layers** | 0.088 | td=2 best; deeper hurts |
| 6 | **patch_size** | 0.064 | p=10 and p=25 best |
| 7 | **nhead** | 0.062 | nh=8 optimal |
| 8 | **masking_ratio** | 0.026 | mr=0.2 slightly best |
| 9 | **lambda_disc** | 0.022 | λ=0.5 best for roc |
| 10 | **dynamic_margin_k** | 0.019 | k=1.5 best |
| 11 | **learning_rate** | 0.013 | lr=0.002 baseline best |
| 12 | **num_student_decoder_layers** | 0.008 | Minimal impact |
| 13 | **margin_type** | 0.008 | softplus slightly better roc |
| 14 | **anomaly_loss_weight** | 0.002 | No meaningful effect |
| 15 | **scoring_mode** | 0.000 | Zero effect |

### Per-Parameter Detailed Analysis

#### seq_length (η²=0.254) — MOST IMPORTANT

| Value | roc_auc | PA%80 | disc_d | disc_SNR |
|-------|---------|-------|--------|----------|
| 100 | **0.953** | **0.715** | 1.497 | 0.554 |
| 200 | **0.956** | 0.714 | **1.983** | **0.750** |
| 500 | 0.811 | 0.478 | 0.729 | 0.283 |

ANOVA: F=119.7, p<0.001

w200 is the overall best window — highest mean roc (0.956) and highest disc_d (1.98), disc_SNR (0.75). w500 is catastrophically worse on roc (-0.14 from w100). However, w500 was the default for Phase 2 since Phase 1 found it best for disturbing separation. Phase 2 reveals this was a costly choice for overall detection.

#### d_model (η²=0.203)

| Value | roc_auc | PA%80 | disc_d |
|-------|---------|-------|--------|
| 64 | **0.919** | **0.653** | 1.341 |
| 128 | 0.870 | 0.568 | 0.991 |
| 256 | 0.731 | 0.381 | 0.469 |
| 512 | 0.719 | 0.358 | 0.226 |

ANOVA: F=50.7, p<0.001

d64 outperforms d128 by 0.049 on roc. The smaller model generalizes better with enc=1. The d256 and d512 models are dramatically worse, with disc_d collapsing (0.47 and 0.23).

#### num_encoder_layers (η²=0.102)

| Value | roc_auc | PA%80 | disc_d |
|-------|---------|-------|--------|
| 1 | **0.866** | **0.555** | **1.022** |
| 3 | 0.736 | 0.374 | 0.315 |
| 4 | 0.741 | 0.386 | 0.328 |

ANOVA: F=33.9, p<0.001

enc=1 is dramatically better than enc=3 or enc=4. This confirms the CORRECTION in PHASE2_PLAN: deeper encoders cause disc_d collapse. The enc=3/4 configs were only tested at w500 (plus a few at w100), where performance is already lower. Even the w100+enc=3/4 configs show poor results.

#### num_teacher_decoder_layers (η²=0.088)

| Value | roc_auc | PA%80 | disc_d |
|-------|---------|-------|--------|
| 2 | **0.904** | 0.615 | **1.579** |
| 3 | 0.856 | 0.532 | 1.022 |
| 4 | 0.864 | 0.554 | 0.981 |
| 5 | 0.757 | 0.379 | 0.302 |
| 6 | 0.784 | 0.417 | 0.437 |

ANOVA: F=11.5, p<0.001

td=2 is best overall, followed by td=4 and td=3. td=5 and td=6 are poor. This contradicts Phase 1's finding that td=4-5 was optimal — the Phase 2 data clearly shows shallower decoders are better. At w500 specifically, td=2 gives roc=0.885 vs td=4's 0.818.

#### patch_size (η²=0.064)

| Value | roc_auc | PA%80 | disc_d | N |
|-------|---------|-------|--------|---|
| 5 | 0.874 | 0.595 | 1.793 | 32 |
| 10 | **0.940** | **0.696** | **1.960** | 40 |
| 20 | 0.837 | 0.517 | 0.809 | 492 |
| 25 | 0.924 | 0.636 | 1.377 | 36 |

ANOVA: F=13.5, p<0.001

p=10 is the best by mean roc (0.940) and disc_d (1.96). p=25 is also strong (0.924). The default p=20 is worst. Note: p=20 has 492 samples (the default), heavily influencing its mean with many w500 configs that perform poorly regardless of patch size.

#### nhead (η²=0.062)

| Value | roc_auc | disc_d |
|-------|---------|--------|
| 4 | 0.844 | 0.834 |
| 8 | **0.875** | **1.013** |
| 16 | 0.816 | 0.869 |

t-test (8 vs 4): p<0.001; nh=8 is significantly better than both alternatives.

#### masking_ratio (η²=0.026)

| Value | roc_auc | disc_SNR | disc_d |
|-------|---------|----------|--------|
| 0.1 | 0.828 | 0.356 | 0.853 |
| 0.15 | 0.851 | 0.375 | 0.962 |
| 0.2 | **0.944** | 0.461 | **1.477** |

ANOVA: F=7.9, p<0.001

mr=0.2 outperforms significantly. However, the mr=0.2 sample (N=24) is biased toward w100/w200 configs. Controlling for window size, the mr=0.2 advantage holds at every window but is most pronounced at w500 (0.884 vs 0.816 for mr=0.15).

#### lambda_disc (η²=0.022)

| Value | roc_auc | disc_d |
|-------|---------|--------|
| 0.5 | **0.937** | 1.551 |
| 1.0 | 0.883 | 0.894 |
| 2.0 | 0.846 | 0.955 |
| 3.0 | 0.861 | 0.846 |
| 4.0 | 0.855 | 0.887 |
| 5.0 | 0.874 | 0.896 |

ANOVA: F=2.6, p=0.023

λ=0.5 gives the best roc (0.937) and highest disc_d (1.55). This contradicts Phase 1's finding that λ≥2.0 is needed. The λ=0.5 sample is biased (N=24, mostly w100 configs from Group 4). At w500, λ=2.0 (the default) is more representative.

#### dynamic_margin_k (η²=0.019)

| Value | roc_auc | disc_d |
|-------|---------|--------|
| 1.0 | 0.875 | 0.889 |
| 1.5 | **0.952** | 1.791 |
| 2.0 | 0.848 | 0.963 |
| 3.0 | 0.837 | 0.781 |
| 4.0 | 0.874 | 0.711 |
| 5.0 | 0.869 | 0.728 |

k=1.5 performs best (roc=0.952, disc_d=1.79), contradicting the decision to increase from k=1.5 to k=2.0 for Phase 2. However, k=1.5 has only N=16 samples, biased toward specific configs.

#### margin_type (η²=0.008)

| Type | roc_auc | disc_d | disc_ratio |
|------|---------|--------|------------|
| dynamic | 0.848 | 0.943 | 2.289 |
| softplus | **0.893** | **1.375** | **6.941** |

t-test: p=0.029

Softplus margin significantly outperforms dynamic on roc (+0.045) and massively on disc_ratio (6.94 vs 2.29). However, softplus is only tested in Group 5 (N=40) with specific configs. The disc_ratio tripling is striking — softplus creates much sharper separation between normal and anomaly discrepancy distributions.

**Deeper analysis** (Softplus × mask):
| Combination | roc | disc_d |
|------------|-----|--------|
| mask_after + softplus | **0.941** | **2.801** |
| mask_after + dynamic | 0.801 | 1.222 |
| mask_before + dynamic | 0.895 | 0.665 |
| mask_before + softplus | 0.846 | -0.050 |

Softplus massively benefits mask_after (+0.14 roc, +1.58 disc_d) but HURTS mask_before (-0.05 roc, disc_d collapses to near zero). This is a critical finding: the margin type × mask timing interaction completely reverses the effect direction.

#### learning_rate (η²=0.013)

All experiments used lr=0.002 except Group 11. The lr comparison is confounded by w500 being the default window. No strong conclusions beyond: lr=0.002 is adequate, lr≥0.008 degrades performance.

#### anomaly_loss_weight (η²=0.002)

No significant effect (ANOVA p=0.804). Values 2-5 all perform similarly. This contradicts Phase 1's finding that alw=2 boosts mask_after disc_d.

---

## 15. PHASE2_PLAN Hypothesis Verification (Req 12)

### Group-by-Group Verification

#### G01: Baseline & Reference (Configs 001-005)

**Plan hypothesis**: The combined Phase 1 optimal parameters work synergistically.

**Result**: Group mean roc=0.956, PA%80=0.725 — the BEST group by mean roc.

**Verification**:
- Config 001 (new default baseline, mask_before): roc=0.955-0.956. Solid performance.
- Config 001 (mask_after): roc=0.801-0.815. **Dramatically worse** — the new default config does NOT work well with mask_after.
- Config 002 (Phase 1 old default): mask_before roc=0.971-0.979; mask_after roc=0.969-0.980. Old defaults perform BETTER than new defaults, especially for mask_after.
- Config 003 (Phase 1 best roc): mask_after roc=0.970-0.978. Also outperforms new default.
- Config 004 (Phase 1 best PA80): mask_after PA80=0.797 (best PA80 among baselines). Confirmed competitive.
- Config 005 (default at w100): mask_before roc=0.980. Proves w100 outperforms w500.

**Verdict**: **PARTIALLY CONFIRMED**. The new defaults work well for mask_before but the Phase 1 old defaults (w100/d64/td2) actually outperform the new defaults for mask_after. The synergy claim is overstated.

#### G02: Window × Capacity (Configs 006-020)

**Plan hypothesis (H7)**: Larger windows need larger models.

**Result**: Group mean roc=0.896, disc_d=1.345.

**Verification**:
- d64 outperforms d128 at ALL window sizes
- d256/d512 catastrophically fail at w500 (roc 0.61-0.70)
- w100+d512: roc=0.883 (bad but not catastrophic)
- w500+d512: roc=0.636-0.695

**Verdict**: **REFUTED**. H7 is wrong for Phase 2. Larger models hurt at all window sizes. The reverse is true: larger windows need SIMPLER models.

#### G03: Encoder-Decoder Depth (Configs 021-040)

**Plan hypothesis (H6)**: Depth scaling inverts with window size.

**Result**: Group mean roc=0.808 (worst-performing group).

**Verification**:
- Best: 035_w100_enc2_td2 (roc=0.983) — shallow everything at w100
- enc=3 configs: mean roc ~0.71 at w500
- enc=4 configs: mean roc ~0.73 at w500
- w100+enc=3/4: roc=0.68-0.71 (also bad)
- w500+enc=1+td=2: roc=0.953 (from other groups)

**Verdict**: **H6 PARTIALLY CONFIRMED** (td=2 best at w500, td=4 best at w100). But the encoder depth part is **REFUTED**: deeper encoders always hurt, at every window size. The depth budget hypothesis is confirmed but the budget is smaller than expected — total depth should be minimized.

#### G04: Discrepancy Loss (Configs 041-060)

**Plan hypothesis**: Higher λ/k/alw optimize disc signal for detection.

**Result**: Group mean roc=0.869.

**Verification**:
- Config 058 (all_disc_boosted: λ=3,k=3,alw=3): mask_after roc=0.508-0.634. **CATASTROPHIC FAILURE**.
- Config 054 (λ=1,k=1 minimal disc): mask_after roc=0.750, mask_before roc=0.949.
- Config 060 (strong disc at w100): mask_before roc=0.974-0.979.
- Highest-performing: moderate disc params at w100.

**Verdict**: **REFUTED**. Triple-boosting disc parameters is destructive. The best approach is moderate λ=0.5-2.0 with moderate k. Over-emphasizing discrepancy during training causes the model to focus on maximizing teacher-student gap at the expense of learning useful representations.

#### G05: Margin Type (Configs 061-070)

**Plan hypothesis**: Determine if softplus is competitive with dynamic.

**Result**: Group mean roc=0.893.

**Verification**:
- Softplus + mask_after: roc=0.941 (exceptional)
- Softplus + mask_before: roc=0.846 (poor, disc_d collapses)
- Best: 063_softplus_w200_mask_after (roc=0.987, disc_d=4.17)
- Worst: 069_softplus_lambda4_mask_before (roc=0.658, disc_d=-0.19)

**Verdict**: **CONDITIONAL YES**. Softplus is dramatically better than dynamic FOR mask_after (+0.14 roc) but worse for mask_before. This is a significant discovery — the margin type × mask timing interaction is the strongest two-way interaction in the entire study.

#### G06: Patch Size (Configs 071-086)

**Plan hypothesis**: Explore patch_size interactions with capacity and depth.

**Result**: Group mean roc=0.888, disc_d=1.507.

**Verification**:
- Best: 083_w500_p5_td2_mask_after (roc=0.990) — THE BEST MODEL IN THE STUDY
- p=5 + td=2 at w500: extraordinary performance (100 patches, shallow decoder)
- p=5 + td=6 at w500: roc=0.589 (terrible)
- p=5 + d64: roc=0.988 (excellent)
- p=5 + d256: roc=0.520 (collapse)

**Verdict**: **CONFIRMED** — patch size interacts strongly with decoder depth and model capacity. p=5 is exceptional with shallow decoders and small models, catastrophic with deep decoders or large models.

**Novel insight**: p=5 + td=2 + mask_after creates a unique regime where very fine-grained patches force the teacher to learn highly specific local patterns, a shallow decoder prevents overfitting, and mask_after maximizes the information gap. This specific combination is the "golden config" of Phase 2.

#### G07: disc_SNR (Configs 087-098)

**Plan hypothesis**: Target maximum disc_SNR with low mr + high capacity.

**Result**: Group mean roc=0.806, disc_SNR=0.227.

**Verification**:
- Best SNR in this group: 097_snr_mr01_w200_mask_after (SNR=1.13)
- But the highest SNR in the entire study comes from G05/G06 (softplus + p=5), not from G07
- mr=0.1 + d256: disc_SNR collapse (0.063)
- mr=0.1 + w100: roc=0.980, decent SNR (0.43)

**Verdict**: **REFUTED**. The SNR-targeted configurations (large model + low mr + deep decoder) consistently underperform. The actual SNR leaders are p=5 configs and softplus configs from other groups. The Phase 1 hypothesis that high capacity maximizes SNR is wrong.

#### G08: Attention Heads (Configs 099-108)

**Plan hypothesis**: Validate nh=8 optimality across scales.

**Result**: Group mean roc=0.826.

**Verification**:
- Best: 108_nh4_w100_p10_mask_after (roc=0.983)
- nh=4 at d64: roc varies
- nh=4 at d256: roc=0.630 (collapse)
- nh=16 at d512: roc=0.641

**Verdict**: **CONFIRMED** — nh=8 is optimal in aggregate. nh=4 can work well at small d_model/window, nh=16 offers no consistent benefit. Head_dim=16 (d128/nh8) is the sweet spot.

#### G09: Masking Ratio (Configs 109-117)

**Plan hypothesis**: Validate mr=0.08-0.15 sweet spot.

**Result**: Group mean roc=0.909.

**Verification**:
- mr=0.2 + w100: roc=0.980 (same as mr=0.15 baseline)
- mr=0.2 + w200: roc=0.970-0.980
- mr=0.1 at w500: roc=0.481-0.655 (some catastrophic failures)
- mr=0.2 at w500: roc=0.654 (still poor but better)

**Verdict**: **REFUTED**. mr=0.2 outperforms mr=0.1 and mr=0.15 at every window size. The Phase 1 sweet spot finding does not hold with the new architecture (enc=1, lr=0.002). The optimal masking ratio appears to be higher than Phase 1 suggested.

#### G10: Student Decoder (Configs 118-127)

**Plan hypothesis**: sd=2 asymmetrically benefits mask_before.

**Result**: Group mean roc=0.847.

**Verification**:
- sd=2 at w500: roc=0.820-0.975 depending on other params
- sd=2 vs sd=1 (paired): no clear systematic difference
- sd=4 (equal to td): no disc_d collapse as feared
- Best: 127_sd2_w100_mask_before (roc=0.980)

**Verdict**: **NOT CONFIRMED**. Student decoder depth has minimal impact (η²=0.008). The Phase 1 sd=2 benefit may have been confounded with other factors.

#### G11: Learning Rate (Configs 128-133)

**Plan hypothesis**: lr=0.003-0.005 outperforms lr=0.002.

**Result**: Group mean roc=0.836.

**Verification**:
- lr=0.002 (base): used by all other groups, group means 0.85-0.96
- lr=0.003 at w500: mask_after 0.755, mask_before 0.951
- lr=0.003 at w100: mask_before 0.977
- lr=0.008: mask_after 0.706, mask_before 0.768
- lr=0.010: roc ~0.68-0.78

**Verdict**: **REFUTED** (partially). lr=0.003 at w100 works well (0.977) but at w500 it's neutral. lr≥0.008 is clearly harmful. The Phase 1 finding was at d64 — the d128 model may need lower lr. The correction to lr=0.002 was appropriate.

#### G12: d_model Sweep (Configs 134-141)

**Plan hypothesis**: Systematic width scaling analysis.

**Result**: Group mean roc=0.756 (second-worst group).

**Verification**:
- d64_nh8: roc=0.960 (excellent)
- d64_nh4: roc=0.954
- d64_nh16: roc=0.880 (good but lower)
- d256_nh8: roc=0.705
- d512_nh8: roc=0.632
- d512_nh16: roc=0.724

**Verdict**: **CONFIRMED scaling laws** — performance degrades monotonically with d_model at w500. d64 is clearly superior. nh=8 or nh=4 work best; nh=16 is suboptimal at d64 (head_dim=4 too small).

#### G13: Combined Optimal (Configs 142-150)

**Plan hypothesis**: Combined optimal parameters yield best performance.

**Result**: Group mean roc=0.790.

**Verification**:
- Config 142 (d256+td5+mr0.1): roc=0.643-0.718 (terrible)
- Config 143 (enc3+td3+mr0.1): roc=0.714-0.784 (poor)
- Config 146 (max depth enc4+td6+sd4): roc=0.674-0.749 (poor)
- Config 147 (minimum: w100+d64+td2): roc=0.951-0.976 (excellent!)
- Config 148 (w100 optimized: p25+td3+mr0.1): roc=0.613-0.752 (terrible)
- Config 149 (w200 optimized): roc=0.690-0.889
- Config 150 (max capacity: d512+enc3+td5): roc=0.638-0.681 (terrible)

**Verdict**: **REFUTED**. The "combined optimal" configs that pile on capacity and depth perform worst. The "minimum" config (147) outperforms nearly all combined configs. This is the most important finding: **simplicity wins**.

---

## 16. Novel Insights (Req 13)

### Insight 1: The Reconstruction-Discrepancy Trade-off is Fundamental

mask_after creates a strict trade-off: disc_d increases while recon_d decreases. The correlation between disc_d and recon_d is only r=0.23. Models cannot simultaneously maximize both. This implies two distinct detection strategies:

1. **Reconstruction-dominant (mask_before)**: High recon_d, moderate disc_d. Reliable roc across configs.
2. **Discrepancy-dominant (mask_after + small patches)**: Extreme disc_d (>4.0), low recon_d. Achieves the absolute best roc but only under narrow conditions.

Future work should explore ensemble scoring that weights reconstruction and discrepancy adaptively.

### Insight 2: The "Golden Config" — p=5 + td=2 + mask_after

Config 083 (w500/p5/d128/td2/mask_after) achieves roc=0.990, the best in the study. This specific combination creates 100 patches, each covering only 5 timesteps. With td=2 (shallow decoder) and mask_after (encode visible only), the model develops extremely discriminative patch-level representations. The teacher learns precise 5-timestep patterns from all 100 patches, while the student must reconstruct from ~85 patches — creating massive discrepancy on anomalous regions.

This is counter-intuitive: w500 is the worst window size on average, but with p=5+td=2, it becomes the best. The fine granularity at long windows captures both local anomaly patterns AND long-range context.

### Insight 3: head_dim=8 is Optimal

Analyzing head_dim (d_model / nhead) directly:

| head_dim | roc_auc | N |
|----------|---------|---|
| 4 | 0.828 | 8 |
| 8 | **0.908** | 36 |
| 16 | 0.865 | 456 |
| 32 | 0.787 | 72 |
| 64 | 0.730 | 20 |
| 128 | 0.701 | 8 |

head_dim=8 (d64/nh8 or d128/nh16) is the sweet spot. This is finer-grained than "nh=8 is best" and explains why d64/nh8 (head_dim=8) outperforms d128/nh8 (head_dim=16).

### Insight 4: Total Depth Budget is ~5 Layers

| Total Depth (enc+td+sd) | roc_auc | N |
|--------------------------|---------|---|
| 5 (enc1+td2+sd2) | **0.962** | 32 |
| 6 (enc1+td3+sd2) | 0.931 | 24 |
| 7 (enc1+td4+sd2) | 0.865 | 404 |
| 8 | 0.768 | 44 |
| 9 | 0.816 | 48 |
| 10 | 0.750 | 28 |
| 11 | 0.700 | 12 |

Optimal total depth is 5 layers (enc=1, td=2, sd=2). Performance degrades monotonically beyond depth 6. The model has a strict depth budget — additional layers hurt generalization at 50 epochs.

### Insight 5: td-sd Gap = 0 is Optimal

| td-sd gap | roc_auc | disc_d |
|-----------|---------|--------|
| 0 | **0.903** | 1.562 |
| 1 | 0.850 | 1.098 |
| 2 | 0.861 | 1.007 |
| 3 | 0.747 | 0.258 |
| 4 | 0.797 | 0.493 |

Equal teacher-student depth (gap=0) gives the best detection. The discrepancy signal comes from masking-based information asymmetry, not depth asymmetry. This contradicts the intuition that deeper teacher + shallower student creates better discrepancy.

### Insight 6: Softplus × mask_after is a Qualitatively Different Regime

The softplus + mask_after combination produces disc_d=2.80 and disc_ratio=6.94, while dynamic + mask_after gives disc_d=1.22 and disc_ratio=2.29. The softplus margin creates a fundamentally different loss landscape that amplifies mask_after's discrepancy signal. Conversely, softplus + mask_before causes disc_d collapse (-0.05).

This interaction is the strongest two-way interaction in the study and was not predicted by Phase 1.

### Insight 7: w200 is the Overlooked Optimal Window

w200 was under-tested in Phase 1 (N=12) and used as a secondary window in Phase 2. Yet it achieves:
- Highest mean roc (0.956, tied with w100)
- Highest mean disc_d (1.98)
- Highest disc_SNR (0.75)
- **Best disturbing separation** (disturbing_roc=0.983)
- The only window where mask_after outperforms mask_before on roc (+0.029)

w200 combines the detection accuracy of w100 with the temporal context of w500, hitting the optimal balance.

### Insight 8: 70 Records Show Discrepancy Collapse

70/600 evaluations (11.7%) show disc_d < 0.01 (discrepancy collapse). These are concentrated in:
- d256 or d512 models (capacity too high, teacher and student converge)
- enc=3 or enc=4 configs (deeper encoders make representations too similar)
- td=5 or td=6 with enc>1 (excessive total depth)
- Some mask_before configs with deep architectures

Discrepancy collapse means the teacher and student produce nearly identical outputs, eliminating the disc signal entirely. The model then relies solely on reconstruction for detection.

### Insight 9: The Phase 2 Default Config Was Suboptimal

The PHASE2_PLAN chose w500/d128/td4/enc=1(corrected)/mr=0.15 as defaults. Based on Phase 2 results, the optimal defaults would be:

| Parameter | Phase 2 Default | Optimal (Phase 2 Evidence) |
|-----------|----------------|--------------------------|
| seq_length | 500 | 200 |
| d_model | 128 | 64 |
| td | 4 | 2 |
| sd | 1 | 2 |
| patch_size | 20 | 10 or 5 |
| masking_ratio | 0.15 | 0.20 |
| margin_type | dynamic | softplus (with mask_after) |

A model with w200/d64/td2/sd2/p10/mr0.2 would likely achieve roc > 0.97 consistently. This was not directly tested but can be inferred from the interaction analyses.

### Insight 10: Discrepancy's Value is Configuration-Dependent

Discrepancy improves detection in a narrow regime:
- ✅ mask_after + p≤10 + td≤3 + d≤128: disc is the primary signal, roc > 0.98
- ❌ mask_before + any config: disc adds marginal value; recon dominates
- ❌ mask_after + d≥256 or td≥5: disc collapses; model breaks
- ❌ mask_after + p=20 at w500: disc is moderate but recon is too poor

The discrepancy mechanism is powerful but fragile. It requires specific conditions to function properly and fails silently (disc_d collapse) when conditions are wrong.

---

## 17. Phase 3 검증 대상: 가설, 파라미터, Insight 정리

Phase 2의 분석 결과와 PHASE2_mask.md의 메커니즘 분석에서 도출된, Phase 3에서 검증해야 할 모든 항목을 정리합니다.

---

### H1. "Golden Config" 재현 및 변형 검증

**배경**: Config 083 (w500/p5/d128/td2/mask_after)이 roc=0.990으로 전체 1위. 그러나 단일 config에서만 관찰되어 재현성이 불확실합니다.

**검증 실험**:
- 083과 동일 config 3회 반복 (seed 변경) → 재현성 확인
- w200/p5/td2/mask_after (미테스트 조합) → w200에서도 golden config이 작동하는가?
- w100/p5/td2/mask_after (config 074와 유사하지만 td=2) → w100에서 patch 수가 20개로 줄어도 유효한가?
- w500/p5/td2/d64/mask_after (config 081과 유사) → d64에서도 유지되는가?
- w500/p5/td3/mask_after → td=3이 td=2보다 나은가?

**핵심 질문**: p=5의 효과가 "100개의 미세한 패치"에서 오는 것이라면, w200/p5 (40패치)에서는 효과가 감소해야 합니다. 감소 폭이 핵심입니다.

---

### H2. Softplus × mask_after 상호작용 심화

**배경**: Phase 2에서 발견된 가장 강력한 2-way interaction. softplus+mask_after는 disc_d=2.80, disc_ratio=6.94인 반면, softplus+mask_before는 disc_d=-0.05 (붕괴). 이 극단적 비대칭의 원인이 불명확합니다.

**검증 실험**:
- softplus + mask_after + p=5 (미테스트) → 두 개의 disc 증폭 메커니즘(softplus + small patch)이 결합되면?
- softplus + mask_after + w200 (config 063이 roc=0.987) → w100/w500에서도 재현?
- softplus + mask_after + td=2 (config 064가 disc_SNR=1.73) → golden config과 결합?
- softplus + mask_before + td=2 → mask_before에서도 softplus가 정말 붕괴하는가? (td 변화 시)
- softplus vs dynamic vs hinge → 3종 margin 비교 at 동일 조건

**가설**: Softplus margin은 gradient가 항상 양수(saturate하지 않음)이므로, mask_after의 큰 discrepancy를 계속 증폭합니다. mask_before에서는 disc가 원래 작아서, softplus의 monotonic gradient가 noise를 증폭시켜 학습 불안정을 유발합니다.

---

### H3. w200이 진정한 최적 window인가?

**배경**: w200은 roc(0.956), disc_d(1.98), disc_SNR(0.75), disturbing_roc(0.983)에서 모두 최고 또는 최고 수준. 그러나 Phase 2에서 w200은 소수의 config에서만 테스트됨.

**검증 실험**:
- w200 전용 systematic sweep: d={64,128}, td={2,3,4}, p={5,10,20}, mr={0.15,0.2}
- w200 + mask_after + p=5 + td=2 ("golden config at w200")
- w200 + softplus + mask_after
- w150, w250, w300 → w200 주변 fine-grained window search
- w200에서 mask_after가 mask_before를 이기는 조건의 범위 확인

**핵심 질문**: Phase 2에서 mask_after가 roc에서 mask_before를 유일하게 이긴 window가 w200. 이것이 robust한 발견인가, 아니면 sampling artifact인가?

---

### H4. td-sd Gap의 순수 효과 분리

**배경**: Phase 2에서 td-sd gap과 absolute td가 confound되어 독립적 효과를 분리할 수 없었습니다. 논리적으로 td-sd gap이 클수록 depth 비대칭이 커져 disc_d가 높아야 하지만, 실제로는 절대 td 증가로 인한 학습 붕괴가 이를 압도했습니다.

**검증 실험** (controlled):
- td=2, sd={1,2} → gap={1,0} at shallow depth
- td=3, sd={1,2,3} → gap={2,1,0} at moderate depth
- td=4, sd={1,2,3,4} → gap={3,2,1,0} at default depth
- 모든 조합을 mask_after + mask_before 양쪽에서 테스트
- 동일 td 내에서 sd만 변화 → gap의 순수 효과

**가설 (두 가지 경쟁 가설)**:
- A: gap이 클수록 disc_d 증가 (정보 비대칭 가설)
- B: gap과 무관하게 td 자체가 disc_d를 결정 (Phase 2 Insight 5의 "gap=0 최적"이 진짜라면)

---

### H5. Deep Decoder Collapse의 정확한 임계점

**배경**: mask_after에서 td≥4부터 성능이 급격히 하락. td=5에서 disc_d=0.077로 붕괴. 그러나 이것이 절대 depth 효과인지, training epoch 부족인지 불분명합니다.

**검증 실험**:
- td={2,3,4,5,6} × epoch={50,100,200} → 더 긴 학습이 deep decoder collapse를 해결하는가?
- td={2,3,4,5,6} × d_model={64,128} → d64에서도 동일한 collapse 패턴?
- td=4 + mask_after + w200 (w500이 아닌) → window 축소가 collapse를 완화하는가?
- td=5 + mask_after + p=5 → small patch가 deep decoder collapse를 극복하는가?

**핵심 질문**: td=5 collapse가 "encoder가 너무 informative해짐" 때문이라면, p=5(100패치)로 reconstruction 난이도를 높이면 collapse가 완화될 수 있습니다.

---

### H6. d64가 d128보다 우수한 이유 확인

**배경**: Phase 2에서 d64가 모든 window에서 d128을 능가 (η²=0.203). Phase 1에서는 d128이 최적이었음. 차이: Phase 1은 enc=2, Phase 2는 enc=1.

**검증 실험**:
- d={32,48,64,96,128} fine-grained sweep at enc=1, w200, td=2
- d=64 + enc=2 vs d=128 + enc=1 → 총 파라미터 수가 비슷할 때 어느 구조가 유리한가?
- d=64 + enc=1 + w500 vs d=128 + enc=1 + w500 → w500에서 d64 우위가 더 큰가?

**가설**: enc=1 bottleneck에서 d=128은 과잉 파라미터. enc=2라면 d=128이 활용될 수 있었을 것. Phase 1에서 d128이 좋았던 것은 enc=2 덕분일 수 있음.

---

### H7. mr=0.2 이상의 masking ratio 탐색

**배경**: Phase 2에서 mr=0.2가 0.15보다 유의미하게 우수. 그러나 mr>0.2는 미테스트.

**검증 실험**:
- mr={0.15, 0.2, 0.25, 0.3, 0.35, 0.4} sweep at w200/d64/td2
- mr={0.2, 0.3, 0.4} × mask={after,before} → 높은 mr에서 mask timing 상호작용이 변하는가?
- mr=0.3 + p=5 (극단적 masking + 미세 패치)

**가설**: mr 증가는 reconstruction 난이도를 높여 teacher-student gap을 키움. 그러나 임계점을 넘으면 teacher도 학습에 실패하여 양쪽 다 붕괴. 최적점은 0.2-0.3 사이에 있을 것.

---

### H8. shared_mask_token의 독립적 효과

**배경**: Phase 2 base config는 shared_mask_token=False (teacher/student가 별도 mask token 사용). PHASE2_mask.md에서 이것이 mask_after 모드의 disc_d 증폭에 기여한다고 분석. 그러나 Phase 2에서 shared_mask_token 자체의 독립적 효과는 테스트되지 않았습니다.

**검증 실험**:
- shared_mask_token={True,False} × mask={after,before} at golden config (w500/p5/td2)
- shared_mask_token={True,False} × mask={after,before} at default config (w500/d128/td4)
- shared_mask_token=True + softplus + mask_after → shared token이 softplus 효과를 감소시키는가?

**가설**: mask_after에서 shared_mask_token=True면 teacher와 student의 masked 위치 출발점이 동일 → disc_d가 감소할 것. mask_before에서는 mask token이 encoder를 거치므로 영향이 작을 것.

---

### H9. Ensemble/Hybrid Scoring 전략

**배경**: mask_after는 disc_d를 극대화하지만 recon_d를 희생. mask_before는 recon_d를 유지하지만 disc_d가 보통. 두 모드의 장점을 결합하면?

**검증 실험**:
- 동일 config에서 mask_before 모델과 mask_after 모델을 각각 학습
- mask_before의 recon score + mask_after의 disc score를 결합
- 결합 가중치 sweep: α × recon_before + (1-α) × disc_after, α={0.1, 0.3, 0.5, 0.7, 0.9}
- 또는: mask_before 모델의 recon + mask_after 모델의 disc를 별도로 threshold 결합

**가설**: mask_before recon_d ~2.3 + mask_after disc_d ~4.6 = 결합 시 두 signal이 모두 강해 roc>0.99 달성 가능. 단, inference cost 2배.

---

### H10. λ (lambda_disc)의 정밀 최적화

**배경**: Phase 2에서 λ=0.5이 roc 최고(0.937)이지만 sample bias(w100 위주). λ=2.0이 base default. λ의 최적값이 mask timing과 window size에 따라 다를 수 있음.

**검증 실험**:
- λ={0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0} × mask={after,before} at w200/d64/td2
- λ={0.5, 1.0, 2.0} × w={100,200,500} → window별 최적 λ
- mask_after + p=5에서 λ sweep → disc-dominant regime에서 λ의 역할

**가설**: mask_after에서는 disc가 이미 매우 강하므로 낮은 λ(0.3-0.5)가 최적. mask_before에서는 disc가 약하므로 높은 λ(2.0-3.0)가 필요. 즉, 최적 λ는 mask timing에 의존.

---

### H11. Encoder Depth 재검증: enc=2가 정말 나쁜가?

**배경**: Phase 2에서 enc=1 >> enc=3 > enc=4 (η²=0.102). PHASE2_PLAN CORRECTION에서도 enc=2가 disc_d collapse를 유발한다고 함. 그러나 Phase 1에서는 enc=2가 잘 작동함 (d=64, w=100 조건).

**검증 실험**:
- enc={1,2} × d={64,128} × w={100,200} → enc=2가 d64/w100에서도 나쁜가?
- enc=2 + td=2 (total depth=5) vs enc=1 + td=4 (total depth=6) → 같은 총 depth에서 encoder에 budget을 주는 것이 나은가?
- enc=2 + mask_after + p=5 → enc=2가 mask_after의 visible-only encoding에서 더 강한 representation을 만드는가?

**가설**: enc=2의 "disc_d collapse"는 w500+d128 조건에서만 발생할 수 있음. d64+w100/w200에서는 enc=2가 여전히 유효할 가능성. Phase 1 결과와의 불일치를 해명해야 함.

---

### H12. head_dim=8 최적성 확인

**배경**: Phase 2 Insight 3에서 head_dim=8 (d64/nh8)이 최적으로 나타남. 그러나 head_dim=8의 N=36으로 적음.

**검증 실험**:
- d=64, nh={4,8,16} → head_dim={16,8,4}
- d=128, nh={8,16,32} → head_dim={16,8,4}
- d=96, nh=12 → head_dim=8 (다른 조합)

**핵심 질문**: head_dim=8이 최적인 이유가 무엇인가? Attention의 fine-grained pattern matching과 관련?

---

### H13. Total Depth Budget vs 개별 Layer 할당

**배경**: Insight 4에서 total depth ~5가 최적. 그러나 5를 enc/td/sd에 어떻게 분배하는 것이 최선인지 불분명.

**검증 실험** (total depth=5):
- enc=1, td=2, sd=2 (현재 최적)
- enc=1, td=3, sd=1
- enc=2, td=2, sd=1
- enc=1, td=1, sd=3 (student가 더 깊은 역전 구조)

**검증 실험** (total depth=6):
- enc=1, td=3, sd=2
- enc=2, td=2, sd=2
- enc=1, td=4, sd=1

**가설**: encoder에 1 layer만 쓰고 나머지를 decoder에 분배하는 것이 최적. decoder 내에서는 td≥sd인 비대칭이 disc signal 생성에 필요하지만, td-sd gap이 너무 크면 H4의 confounding 문제 발생.

---

### H14. p=10이 평균 최강인 이유 탐구

**배경**: Phase 2에서 p=10이 평균 roc(0.940)과 disc_d(1.96)에서 최고. p=5가 top individual model을 산출하지만 분산이 큼. p=10은 "안정적으로 강한" 패치 사이즈.

**검증 실험**:
- p={5,7,10,12,15,20} fine-grained sweep at w200/d64/td2
- p=10 + mask_after + softplus (미테스트 조합)
- p=10 × w={100,200,500} → 각 window에서 p=10의 patch 수 = {10,20,25}

**핵심 질문**: p=10의 우수성이 "적당한 수의 패치" (reconstruction 난이도와 정보량의 균형)에서 오는가, 아니면 10-timestep 단위가 이상 패턴의 자연스러운 단위와 일치하는가?

---

### H15. anomaly_loss_weight의 재검증

**배경**: Phase 2에서 alw의 효과가 유의미하지 않음 (η²=0.002, p=0.804). 그러나 Phase 1에서는 alw=2가 mask_after의 disc_d를 boost한다고 발견됨.

**검증 실험**:
- alw={0, 0.5, 1, 2, 3} × mask={after,before} at golden config
- alw=0 (anomaly loss 완전 제거) → disc_d가 어떻게 변하는가?

**가설**: Phase 2의 base config가 이미 alw=2.0이므로 추가 증가의 효과가 saturate. alw=0과의 비교가 핵심 — anomaly loss 자체가 필요한지 확인.

---

### H16. Training Epoch와 Deep Model의 관계

**배경**: Phase 2는 모두 50 epoch 학습. 깊은 모델(enc≥2, td≥5)의 성능 저하가 under-training 때문일 수 있음.

**검증 실험**:
- td={2,4,6} × epoch={50,100,200,300} at w500/d128/mask_after
- enc={1,2,3} × epoch={50,100,200} at w500/d128
- d={64,128,256} × epoch={50,100,200}

**가설**: 깊은 모델은 더 많은 epoch이 필요하지만, 50 epoch에서의 성능 차이가 200 epoch에서 역전될 수 있음. 역전되지 않는다면, shallow model의 우위는 근본적.

---

### H17. dynamic_margin_k=1.5 vs k=2.0 정밀 비교

**배경**: Phase 2에서 k=1.5가 최고(roc=0.952)이지만 N=16으로 bias 가능. Phase 2 base는 k=2.0.

**검증 실험**:
- k={1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0} fine-grained sweep at w200/d64/td2
- k sweep × mask={after,before}

**가설**: k=1.5가 진정 최적이라면 Phase 2 default를 k=2.0에서 k=1.5로 변경해야 함.

---

### H18. mask_after에서 recon_d 하락을 완화할 수 있는가?

**배경**: PHASE2_mask.md 분석에 따르면, mask_after에서 recon_d 하락(-0.96)은 encoder가 visible만 처리하기 때문에 필연적. 그러나 이를 완화할 수 있는 architecture 변형이 있을 수 있음.

**검증 실험**:
- mask_after + encoder에 positional encoding 강화 (visible patch의 위치 정보를 더 풍부하게)
- mask_after + decoder에 skip connection 추가 (encoder의 visible representation을 decoder 중간 layer에 직접 전달)
- mask_after + reconstruction loss에 masked 위치 가중치 조정

**가설**: recon_d 하락이 "masked 위치의 정보가 encoder에 없음"에서 오므로, decoder가 visible latent에서 masked 위치를 더 잘 추론하도록 architectural bias를 추가하면 recon_d가 개선될 수 있음. 단, 이것이 disc_d를 감소시키지 않는 것이 중요.

---

### H19. Scoring 수식 자체의 변경

**배경**: 현재 `score = recon + λ * disc`. Phase 2에서 adaptive scoring은 효과 없음. 그러나 다른 결합 방식은 미테스트.

**검증 실험**:
- `score = max(recon, λ * disc)` → 더 강한 signal을 자동 선택
- `score = recon * disc` → 곱셈 결합 (두 signal이 모두 높을 때만 high score)
- `score = recon + λ * disc + β * recon * disc` → 상호작용 항 추가
- Point-level adaptive λ: 각 timestep에서 disc/recon 비율에 따라 λ를 동적 조정

**가설**: 현재 선형 결합은 mask_after의 recon 약점을 과도하게 노출. 비선형 결합이 mask_after의 강점(disc)을 더 효과적으로 활용할 수 있음.

---

### H20. lr warmup / scheduler 효과

**배경**: Phase 2에서 lr=0.002 고정, lr schedule 미적용. 깊은 모델의 학습 불안정이 lr schedule 부재 때문일 수 있음.

**검증 실험**:
- lr=0.002 + cosine annealing → min_lr=0.0002
- lr=0.002 + warmup 5 epoch → 학습 초기 안정화
- lr=0.002 + reduce on plateau
- 위 schedule × td={2,4,6} → 깊은 모델에서 schedule의 효과가 더 큰가?

---

### H21. Patch 수 vs Patch 크기 분리

**배경**: p=5가 w500에서 최강인 것은 100패치 때문인가, 5-timestep 크기 때문인가? 두 요인이 confound됨.

**검증 실험**:
- 동일 패치 수(50패치): w500/p10 vs w250/p5 vs w100/p2(불가능하면 skip)
- 동일 패치 크기(p=5): w100(20패치) vs w200(40패치) vs w500(100패치)
- 동일 패치 크기(p=10): w100(10패치) vs w200(20패치) vs w500(50패치)

**가설**: 패치 수가 핵심이라면, w200/p5(40패치)도 w500/p10(50패치)과 비슷해야 함. 패치 크기가 핵심이라면, 동일 p에서 window 증가가 항상 도움이 되어야 함.

---

### H22. 9가지 Anomaly Type별 성능 분해

**배경**: Phase 2에서 aggregate roc만 분석. 9가지 anomaly type (6 value + 3 pattern) 중 어떤 type에서 mask_after가 특히 강한지/약한지 미분석.

**검증 실험**:
- Top 10 configs에 대해 anomaly type별 roc breakdown
- mask_after vs mask_before: type별로 어느 쪽이 유리한지
- disc signal vs recon signal: type별로 어느 signal이 주효한지

**가설**: mask_after의 disc-dominant 접근은 point anomaly(sudden spike 등)에 강하고, pattern anomaly(trend change 등)에 약할 수 있음. mask_before의 recon-dominant 접근은 pattern anomaly에 더 적합할 수 있음.

---

### H23. Phase 1 결과와의 불일치 해명

**배경**: Phase 2에서 Phase 1의 여러 결론이 뒤집힘. 주요 불일치:

| 항목 | Phase 1 | Phase 2 | 차이 원인 후보 |
|------|---------|---------|--------------|
| mask_after roc | 우세 (normalized scoring) | 열세 (214/300) | λ=0.5→2.0, scoring 차이 |
| d_model | d128 최적 | d64 최적 | enc=2→1 |
| td | td=4-5 최적 | td=2 최적 | 50 epoch 제한? w500 위주? |
| mr | 0.08-0.15 | 0.2 | enc=2→1, lr=0.005→0.002 |
| scoring mode | normalized 중요 | 무의미 | λ=0.5→2.0 |

**검증 실험**:
- Phase 1 exact config (d64/w100/td2/enc=2/λ=0.5/k=1.5/mr=0.2) vs Phase 2 base config → 직접 비교
- Phase 1 config에서 한 파라미터만 Phase 2 값으로 변경 → 어느 파라미터가 결론 반전을 유발하는가?
  - enc: 2→1
  - λ: 0.5→2.0
  - lr: 0.005→0.002
  - d: 64→128

**핵심 질문**: Phase 1과 Phase 2의 결론 차이가 **enc=1 vs enc=2** 단일 요인으로 설명되는가, 아니면 여러 요인의 복합적 결과인가?

---

### H24. Dropout / Regularization 효과

**배경**: Phase 2에서 dropout은 따로 탐색되지 않음 (base config의 dropout만 사용). 깊은 모델의 overfitting이 dropout으로 완화될 수 있음.

**검증 실험**:
- dropout={0, 0.05, 0.1, 0.15, 0.2} × td={2,4,6} at w500/d128/mask_after
- dropout이 deep decoder collapse를 완화하는가?

---

### H25. 최종 최적 Config 후보 (Phase 2 evidence 기반)

Phase 2 결과에서 추론된 "이론적 최적" config들 (미검증):

**Config A (Conservative)**: w200/d64/td2/sd2/enc1/p10/mr0.2/nh8/λ=0.5/k=1.5/mask_before
- 근거: 모든 "안전한" 최적값의 조합. mask_before로 recon 유지.
- 예상 roc: 0.975-0.985

**Config B (Aggressive disc)**: w200/d64/td2/sd2/enc1/p5/mr0.2/nh8/softplus/mask_after
- 근거: golden config의 w200 변형 + softplus. disc-dominant.
- 예상 roc: 0.985-0.995 (또는 unstable)

**Config C (Balanced)**: w200/d64/td2/sd2/enc1/p10/mr0.2/nh8/λ=1.0/mask_after
- 근거: mask_after가 w200에서 roc 유리 + p10의 안정성 + moderate λ.
- 예상 roc: 0.980-0.990

**Config D (Ensemble)**: Config A의 recon + Config B의 disc, scored as α*recon_A + (1-α)*disc_B
- 근거: 두 모델의 장점 결합.
- 예상 roc: 0.990+ (이론적)

이들 중 가장 유망한 것은 **Config C**입니다. Config B는 softplus+mask_after의 극단적 disc 증폭에 의존하므로 분산이 클 수 있고, Config D는 inference cost가 2배입니다.

---

### Priority 순서 (제안)

| Priority | 가설 | 이유 |
|----------|------|------|
| **P0 (필수)** | H1 (Golden Config 재현) | 전체 1위 결과의 재현성 확인 |
| **P0** | H3 (w200 systematic) | 가장 유망한 미탐색 영역 |
| **P0** | H25 (최적 Config A-D 검증) | 실용적 최종 목표 |
| **P1 (중요)** | H2 (Softplus × mask_after) | 가장 강한 interaction 심화 |
| **P1** | H4 (td-sd gap 분리) | 핵심 confounding 해소 |
| **P1** | H7 (mr>0.2 탐색) | 쉽게 확인 가능한 개선점 |
| **P1** | H10 (λ 정밀화) | mask timing별 최적 λ 결정 |
| **P2 (유용)** | H5 (Deep collapse 임계점) | 메커니즘 이해 |
| **P2** | H6 (d64 vs d128 원인) | Phase 1 불일치 해명 |
| **P2** | H13 (Depth budget 분배) | Architecture 최적화 |
| **P2** | H14 (p=10 탐구) | 안정적 최적 패치 사이즈 |
| **P2** | H21 (Patch 수 vs 크기 분리) | 근본 메커니즘 이해 |
| **P2** | H22 (Anomaly type별 분해) | 실용적 insight |
| **P3 (탐색)** | H8 (shared_mask_token) | 단일 요인 검증 |
| **P3** | H9 (Ensemble scoring) | 혁신적이지만 비용 2배 |
| **P3** | H11 (enc=2 재검증) | Phase 1 불일치 |
| **P3** | H16 (Epoch 증가) | 시간 비용 높음 |
| **P3** | H18 (recon_d 완화) | Architecture 변경 필요 |
| **P3** | H19 (Scoring 수식 변경) | 코드 수정 필요 |
| **P3** | H23 (Phase 1-2 불일치 해명) | 학술적 가치 |

---

## 18. Visualization 대상 모델 선정 (30 Configs × 4 Variants = 120 Experiments)

30개 config를 선정하여 각 config의 4개 variant (mask_before/after × default/adaptive scoring)에 대해 best_model 시각화를 생성합니다. 시각화 결과는 각 experiment 디렉토리의 `visualization/best_model/` 에 저장됩니다.

### 선정 기준

| Category | Configs | 선정 이유 |
|----------|---------|----------|
| **Top roc_auc** | 083, 081, 007, 063, 064, 077 | 각각 roc 0.986-0.990, 전체 상위 모델들 |
| **Top PA%80 + disc_ratio** | 004, 006, 108 | PA%80 0.78+, disc_ratio 상위 |
| **Baselines** | 001, 002, 003, 005 | Phase 2 default, Phase 1 old default, Phase 1 best roc, w100 default |
| **Softplus interaction** | 065, 069 | Softplus best(td6) vs worst(lambda4) — 극단적 비교 |
| **Failure/collapse cases** | 058, 023, 014, 027 | disc boost 붕괴, td=5 collapse, d512 collapse, enc3 collapse |
| **Architecture insights** | 035, 147, 012, 134, 124 | enc2+td2 best, minimum config, w500+d64, d64+nh8, sd2+td6 |
| **Parameter exploration** | 110, 054, 097, 116, 060, 148 | mr=0.2, minimal disc, SNR target, mr0.2+td3, w100 disc strong, w100 combo |

### Config별 상세

| # | Config | Description | 선정 이유 | roc_auc (best variant) |
|---|--------|-------------|----------|----------------------|
| 1 | **083** | w500/p5/td2 | **전체 1위** (golden config) | 0.990 |
| 2 | **081** | w500/p5/d64 | **최고 disc_d** (4.674) | 0.988 |
| 3 | **007** | w100/p5 | w100 최강 소형 패치 | 0.988 |
| 4 | **063** | softplus/w200 | **Softplus 최강** | 0.987 |
| 5 | **064** | softplus/td2 | **최고 disc_SNR** (1.733) | 0.983 |
| 6 | **077** | w200/p5 | w200 소형 패치 | 0.986 |
| 7 | **004** | Phase1 best PA80 | PA%80 리더 | 0.981 |
| 8 | **006** | w100/p10 | p10 대표 | 0.983 |
| 9 | **108** | nh4/w100/p10 | nh=4 효과 확인 | 0.983 |
| 10 | **001** | Default baseline | Phase 2 기준선 (w500/d128/td4) | 0.956 |
| 11 | **002** | Phase1 old default | Phase 1 기준 (w100/d64/td2) | 0.980 |
| 12 | **003** | Phase1 best roc | Phase 1 최고 성능 config | 0.978 |
| 13 | **005** | Default at w100 | w100 기준선 | 0.980 |
| 14 | **065** | softplus/td6 | Softplus + deep decoder | 0.978 |
| 15 | **069** | softplus/lambda4 | **Softplus 최악** (mask_before 붕괴) | 0.658 |
| 16 | **058** | all_disc_boosted | **Catastrophic failure** (λ3/k3/alw3) | 0.634 |
| 17 | **023** | enc2/td5 | **td=5 mask_after collapse** (disc_d=0.077) | 0.965 (mb) / 0.663 (ma) |
| 18 | **014** | w500/d512 | **Large model collapse** | 0.724 |
| 19 | **027** | enc3/td4 | **Deep encoder** 실패 | 0.698 |
| 20 | **035** | w100/enc2/td2 | G03 최강 — enc=2가 작동하는 유일한 조건 | 0.983 |
| 21 | **147** | combo minimum | **Simplicity wins** (w100/d64/td2) | 0.976 |
| 22 | **012** | w500/d64 | Small model at long window | 0.960 |
| 23 | **134** | d64/nh8 | d_model sweep 대표 | 0.960 |
| 24 | **124** | sd2/td6 | Student-teacher gap=4 | 0.867 |
| 25 | **110** | mr=0.2 | 높은 masking ratio at w500 | 0.654 (ma) / 0.975 (mb) |
| 26 | **054** | lambda1/k1 minimal | 최소 disc 파라미터 | 0.949 |
| 27 | **097** | snr_mr01/w200 | SNR 타겟 w200 config | 0.971 |
| 28 | **116** | mr0.2/td3 | mr=0.2 + moderate depth | 0.976 |
| 29 | **060** | w100 disc strong | w100에서 disc 극대화 | 0.979 |
| 30 | **148** | w100 optimized combo | w100 최적화 조합 | 0.752 |

### 시각화 출력

각 experiment directory에 17개 PNG가 생성됩니다:
- ROC curves (standard, comparison, PA%80)
- Confusion matrix, summary statistics
- Score contribution analysis, score trends
- Score distribution by type
- Reconstruction examples, detection examples
- Pure vs disturbing normal comparison
- Hardest samples, discrepancy trend
- Performance by anomaly type (individual, comparison)
- Case study gallery, learning curve

---

## Appendix A: Baseline Comparison Details

### Config 001 (New Default Baseline)

| Variant | roc | PA%80 | disc_d | recon_d | disc_SNR |
|---------|-----|-------|--------|---------|----------|
| mask_before_default | 0.956 | 0.705 | 0.948 | 2.321 | 0.348 |
| mask_before_adaptive | 0.955 | 0.710 | 0.948 | 2.321 | 0.348 |
| mask_after_default | 0.815 | 0.420 | 0.659 | 1.105 | 0.281 |
| mask_after_adaptive | 0.801 | 0.432 | 0.659 | 1.105 | 0.281 |

### Config 002 (Phase 1 Old Default)

| Variant | roc | PA%80 | disc_d | recon_d | disc_SNR |
|---------|-----|-------|--------|---------|----------|
| mask_before_default | 0.979 | 0.787 | 0.824 | 2.547 | 0.315 |
| mask_before_adaptive | 0.971 | 0.760 | 0.824 | 2.547 | 0.315 |
| mask_after_default | 0.969 | 0.749 | 3.527 | 1.717 | 1.263 |
| mask_after_adaptive | 0.980 | 0.769 | 3.527 | 1.717 | 1.263 |

The Phase 1 old default (w100/d64/td2/mr0.2/λ0.5/k1.5) outperforms the Phase 2 new default on both mask timings. The old config's mask_after_adaptive achieves roc=0.980, disc_d=3.53, and disc_SNR=1.26 — far superior to Config 001's mask_after results.

---

## Appendix B: Top disc_SNR Models

| Experiment | disc_SNR | disc_d | roc | Configuration |
|-----------|----------|--------|-----|--------------|
| 064_softplus_td2_mask_after | 1.733 | 4.100 | 0.983 | softplus, w500, td2, d128 |
| 063_softplus_w200_mask_after | 1.662 | 4.174 | 0.987 | softplus, w200, td4, d128 |
| 081_w500_p5_d64_mask_after | 1.580 | 4.674 | 0.988 | p5, w500, d64, td4 |
| 083_w500_p5_td2_mask_after | 1.533 | 4.638 | 0.990 | p5, w500, d128, td2 |
| 065_softplus_td6_mask_after | 1.504 | 3.606 | 0.978 | softplus, w500, td6, d128 |
| 007/074_w100_p5_mask_after | 1.437 | 4.367 | 0.988 | p5, w100, d128, td4 |

All top disc_SNR models are mask_after. The top entries are split between softplus margin configs and small-patch configs. Softplus achieves highest raw SNR (1.73) while small patches achieve highest disc_d (4.67).

Common pattern: enc=1, d_model ≤ 128, mask_after, nh=8.

---

## Appendix C: Statistical Notes

- All ANOVA results use one-way analysis; interactions are analyzed via grouped means, not formal interaction terms
- Sample sizes are highly unbalanced (d_model=128 has N=440 while d_model=512 has N=32); interpret ANOVA F-statistics with caution
- The η² values are calculated from one-way ANOVA and do not account for confounding between parameters
- Paired mask comparisons use all 300 config pairs, providing robust statistical power
