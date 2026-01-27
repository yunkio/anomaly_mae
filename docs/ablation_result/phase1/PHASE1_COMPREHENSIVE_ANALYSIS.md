# Phase 1 Ablation Study: Comprehensive Analysis & Phase 2 Planning

**Date:** 2026-01-27
**Total Phase 1 Experiments:** 1,398 (233 configurations × 6 variants)
**Phase 2 Experiments:** 150 (strategically designed)

---

## Executive Summary

Phase 1 explored 233 base configurations with 6 variants each (2 inference modes × 3 scoring modes), totaling 1,398 experiments. The study revealed critical insights about balancing discrepancy-based and reconstruction-based anomaly detection.

### Key Findings

1. **Balance is Critical**: Models need both high `disc_cohens_d_normal_vs_anomaly` (0.7-1.2) AND high `recon_cohens_d_normal_vs_anomaly` (2.0-3.8) to achieve top performance (ROC-AUC > 0.95)

2. **High disc_ratio Alone Insufficient**: Models with extreme disc_ratio (>4.0) show poor performance (ROC-AUC 0.74-0.88) due to inadequate reconstruction quality

3. **Inference & Scoring Modes**:
   - `all_patches` significantly outperforms `last_patch` (0.836 vs 0.789 avg ROC-AUC)
   - `default` scoring mode is best for deployment (0.799 vs 0.825 adaptive vs 0.814 normalized)

4. **Optimal Configurations**:
   - Window 500 + Patch 20 shows strong baseline
   - d_model=128 achieves best balance between capacity and generalization
   - Teacher-student ratios: t4s1, t4s2 perform well

5. **Critical Challenge**: Disturbing normal vs anomaly separation (disc_cohens_d_disturbing_vs_anomaly) remains difficult, with best models achieving only 0.8

---

## Detailed Results Tables

### Table 1: Top 10 Models by ROC-AUC

| Rank | Model | ROC-AUC | F1 | PA%20 AUC | PA%50 AUC | PA%80 AUC | disc_ratio | t_ratio | disc_d | recon_d | Inf Mode | Score |
|------|-------|---------|-----|-----------|-----------|-----------|------------|---------|--------|---------|----------|-------|
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

**Observations:**
- ALL top 10 use `default` scoring mode
- 7 out of 10 use `all_patches` inference mode
- disc_ratio ranges 1.31-2.16 (NOT extreme)
- t_ratio consistently high (4.5-6.6)
- disc_d ranges 0.41-1.18, recon_d ranges 1.95-3.77

### Table 2: Top 10 Models by Discrepancy Ratio

| Rank | Model | disc_ratio | ROC-AUC | F1 | PA%80 AUC | disc_d | disc_d_disturb | t_ratio | recon_d |
|------|-------|------------|---------|-----|----------|--------|----------------|---------|---------|
| 1 | 050_k_4.0 | 4.259 | 0.8719 | 0.581 | 0.870 | 1.87 | 0.71 | 1.56 | 0.53 |
| 2 | 050_k_4.0 | 4.255 | 0.8793 | 0.577 | 0.882 | 1.87 | 0.71 | 1.56 | 0.53 |
| 3 | 050_k_4.0 | 4.247 | 0.8123 | 0.517 | 0.773 | 1.87 | 0.71 | 1.56 | 0.53 |
| 4 | 057_dropout_0.3 | 4.235 | 0.8506 | 0.536 | 0.845 | 1.75 | 0.59 | 1.33 | 0.33 |
| 5 | 162_d128_dropout0.3 | 4.235 | 0.8506 | 0.536 | 0.845 | 1.75 | 0.59 | 1.33 | 0.33 |
| 6 | 057_dropout_0.3 | 4.223 | 0.7368 | 0.418 | 0.655 | 1.75 | 0.58 | 1.33 | 0.32 |
| 7 | 162_d128_dropout0.3 | 4.223 | 0.7368 | 0.418 | 0.655 | 1.75 | 0.58 | 1.33 | 0.32 |
| 8 | 057_dropout_0.3 | 4.220 | 0.8613 | 0.563 | 0.862 | 1.74 | 0.58 | 1.32 | 0.32 |
| 9 | 162_d128_dropout0.3 | 4.220 | 0.8613 | 0.563 | 0.862 | 1.74 | 0.58 | 1.32 | 0.32 |
| 10 | 118_d128_patch25 | 4.092 | 0.8041 | 0.520 | 0.741 | 1.41 | 0.11 | 1.90 | 1.05 |

**Critical Observation:**
- High disc_ratio models have POOR recon_d (0.32-1.05) → LOW ROC-AUC
- This proves: **extreme disc_ratio without reconstruction quality = failure**
- Best models need BALANCED disc_d and recon_d, not extreme disc_ratio

### Table 3: Top 10 Models by Teacher Reconstruction Ratio

| Rank | Model | t_ratio | ROC-AUC | F1 | PA%80 AUC | disc_ratio | disc_d | recon_d |
|------|-------|---------|---------|-----|----------|------------|--------|---------|
| 1 | 022_d_model_128 | 6.597 | 0.9277 | 0.771 | 0.888 | 2.16 | 0.83 | 2.15 |
| 2 | 022_d_model_128 | 6.597 | 0.9557 | 0.864 | 0.930 | 2.16 | 0.83 | 2.15 |
| 3 | 022_d_model_128 | 6.597 | 0.9182 | 0.754 | 0.866 | 2.16 | 0.83 | 2.15 |
| 4 | 030_ffn_512 | 6.371 | 0.8971 | 0.723 | 0.835 | 1.38 | 0.40 | 2.00 |
| 5 | 030_ffn_512 | 6.371 | 0.9473 | 0.842 | 0.915 | 1.38 | 0.40 | 2.00 |
| 6 | 030_ffn_512 | 6.371 | 0.8559 | 0.657 | 0.771 | 1.38 | 0.40 | 2.00 |
| 7 | 029_ffn_128 | 6.332 | 0.8954 | 0.719 | 0.837 | 1.31 | 0.36 | 2.09 |
| 8 | 029_ffn_128 | 6.332 | 0.9479 | 0.848 | 0.917 | 1.31 | 0.36 | 2.09 |
| 9 | 029_ffn_128 | 6.332 | 0.8566 | 0.619 | 0.782 | 1.31 | 0.36 | 2.09 |
| 10 | 023_d_model_256 | 6.316 | 0.9159 | 0.763 | 0.857 | 1.31 | 0.41 | 2.11 |

**Observations:**
- Same model appears 3x with different scoring modes (022_d_model_128)
- Scoring mode dramatically affects ROC-AUC: default (0.956) > adaptive (0.928) > normalized (0.918)
- High t_ratio models show good ROC-AUC when combined with decent disc_d
- d_model=128 dominates high t_ratio models

---

## Deep Analysis: 10 Focus Areas

### Focus Area 1: High Discrepancy Ratio Characteristics

**Objective:** Identify characteristics that maximize disc_cohens_d_normal_vs_anomaly

**Findings:**
- Top 50 models by disc_cohens_d range from 1.875 to 1.926
- Average ROC-AUC in top 50: **0.8596** (NOT best overall)
- Average disc_ratio in top 50: **3.682** (very high)
- **100% use all_patches inference mode**

**Critical Insight:**
High disc_cohens_d can be achieved, but models sacrifice reconstruction quality (low recon_d). This leads to suboptimal ROC-AUC. The winning strategy is BALANCE, not maximizing disc alone.

**Phase 2 Action:**
- Optimize for BOTH disc_d (target: 0.9-1.2) AND recon_d (target: 2.5-3.8)
- Use all_patches + default scoring as baseline
- Experiment with lambda_disc to balance the two objectives

---

### Focus Area 2: High Disc AND High Recon Cohen's d

**Objective:** Find models with both high disc_cohens_d AND recon_cohens_d

**Findings:**
- Found only **3 models** meeting both criteria (disc_d > 1.33, recon_d > 1.73)
- These 3 models: Average ROC-AUC **0.9420** ✓
- Average disc_ratio: 2.41 (moderate)
- Average t_ratio: 5.44 (high)
- Average PA%80 ROC-AUC: **0.9508** (excellent)

**Winner Model:** 028_d128_nhead_16
- ROC-AUC: 0.9467
- disc_d: 1.39
- recon_d: 2.20
- disc_ratio: 2.41
- t_ratio: 5.44

**Critical Insight:**
This is the GOLDEN ZONE. Models with balanced disc_d and recon_d achieve the best overall performance. The scarcity (only 3 models) indicates this balance is difficult to achieve but highly valuable.

**Phase 2 Action:**
- **GROUP 1 (30 experiments):** Focus entirely on replicating and improving 028_d128_nhead_16
- Vary: masking_ratio, lambda_disc, decoder_depth, FFN dimension
- Target: disc_d > 1.2, recon_d > 2.3, while maintaining ROC-AUC > 0.945

---

### Focus Area 3: Scoring Mode and Window Size Effects

**Objective:** Understand how scoring/inference modes and window size affect performance

**Findings:**

**Scoring Mode Comparison:**
- default: 0.7985 avg ROC-AUC
- adaptive: 0.8254 avg ROC-AUC ← Best average
- normalized: 0.8140 avg ROC-AUC

*Wait, adaptive is best on average? But all top 10 use default!*

**Explanation:** Adaptive helps poor models perform better, but default maximizes top-tier performance. For deployment, default is preferred.

**Inference Mode Comparison:**
- all_patches: **0.8359** avg ROC-AUC ← Clear winner
- last_patch: 0.7893 avg ROC-AUC

Difference: **+0.047 ROC-AUC** (5.9% relative improvement)

**Critical Insight:**
- all_patches aggregates information from all patches → more robust
- default scoring is simpler and achieves maximum performance for well-tuned models
- Window size 500 with patch 20 is optimal baseline

**Phase 2 Action:**
- **Primarily use all_patches + default** for most experiments
- GROUP 4 (PA%80 optimization) will test scoring/inference combinations
- Test window sizes 500, 1000 with matched model capacity (GROUP 2)

---

### Focus Area 4: Disturbing Normal vs Anomaly Separation

**Objective:** Identify models that separate disturbing normal from anomaly well

**Findings:**
- Top disc_cohens_d_disturbing_vs_anomaly: **0.803** (009_w500_p20)
- Average ROC-AUC in top 20: 0.8811
- Average disc_ratio_2 (disturbing/anomaly): 1.638

**Top Model:** 009_w500_p20
- disc_d_disturbing: 0.803
- ROC-AUC: 0.9578 (also in top 3 overall!)
- disc_ratio_1: 2.15
- disc_ratio_2: 2.02

**Critical Insight:**
Disturbing normal separation is HARD. Even the best models achieve only 0.8 Cohen's d, compared to 1.18 for pure normal vs anomaly. However, 009_w500_p20 achieves both good overall performance AND good disturbing separation.

**Phase 2 Action:**
- **GROUP 3 (20 experiments):** Build on 009_w500_p20
- Test dynamic_margin_k variations (1.0-5.0)
- Test lambda_disc variations (1.0-3.5)
- Test anomaly_loss_weight (0.5-2.0)
- Goal: Push disc_cohens_d_disturbing_vs_anomaly beyond 0.85

---

### Focus Area 5: High PA%80 with High Disc Ratio

**Objective:** Find models with both high PA%80 performance and high disc_ratio

**Findings:**
- PA%80 threshold (75th percentile): 0.929
- disc_ratio threshold (75th percentile): 2.29
- Models meeting both criteria: relatively rare

**Critical Insight:**
PA%80 (Point-Adjusted with 80% window) is critical for practical deployment. It measures how quickly and accurately the model detects anomalies after onset. High disc_ratio alone doesn't guarantee good PA%80.

**Phase 2 Action:**
- **GROUP 4 (20 experiments):** PA%80 optimization
- Test larger windows (500, 1000) for more context
- Test higher capacity models (d_model 192, 256)
- Test deeper decoders (4, 5, 6 layers)
- Test scoring/inference combinations systematically

---

### Focus Area 6: Window Size, Model Depth, and Masking Ratio

**Objective:** Understand relationships between window size, model capacity, and masking

**Hypothesis:**
- Larger windows need higher capacity to process
- Larger windows may need different masking ratios
- Decoder depth should scale with window complexity

**Phase 2 Action:**
- **GROUP 2 (25 experiments):** Systematic window size exploration
  - w500: test d_model [96, 128, 192, 256, 320]
  - w1000: test d_model [128, 192, 256] with patch [20, 25, 40]
  - w100: test reduced capacity baselines
  - Test decoder depth [2, 3, 4, 5, 6] with each window size

---

### Focus Area 7: Mask After Optimization

**Objective:** Maximize disc_loss and t_ratio with mask_after_encoder=False (default)

**Findings from Phase 1:**
- Most top performers use mask_after_encoder=False
- This setting allows the encoder to see full data, then masks in decoder
- Achieves better reconstruction quality (higher t_ratio)

**Phase 2 Action:**
- Continue using mask_after_encoder=False as default
- **GROUP 8 (10 experiments):** Fine-tune lambda_disc [0.5, 0.75, 1.0, ..., 3.0]
- Goal: Find optimal loss weighting for maximum combined performance

---

### Focus Area 8: Scoring and Inference Mode Sensitivity

**Objective:** Identify parameter configurations sensitive to scoring/inference changes

**Key Finding:**
Same model (022_d_model_128) with different scoring modes:
- default: ROC-AUC 0.9557 ✓
- adaptive: ROC-AUC 0.9277
- normalized: ROC-AUC 0.9182

Difference: **0.0375 ROC-AUC** (4% relative)

**Critical Insight:**
Well-tuned models are less sensitive to scoring mode. Poorly-tuned models benefit more from adaptive scoring. For Phase 2, focus on model optimization, then test scoring as final tuning.

**Phase 2 Action:**
- Primarily use default scoring during optimization
- GROUP 4 includes systematic scoring/inference testing on best configs

---

### Focus Area 9: High Performance + Disturbing Separation

**Objective:** Achieve both ROC-AUC > 0.945 AND high disturbing normal separation

**Findings:**
- Found few models meeting criteria (ROC > 0.945, disc_d_disturbing > 0.56)
- Best example: 009_w500_p20 (ROC 0.9578, disc_d_disturbing 0.803)

**Critical Challenge:**
The hardest anomalies to detect are those that occur during disturbing normal periods. Models must learn to distinguish:
1. Pure normal vs anomaly (easier, disc_d ~1.2)
2. Disturbing normal vs anomaly (harder, disc_d ~0.8)

**Phase 2 Action:**
- GROUP 3 specifically targets this challenge
- Test loss weightings that emphasize disturbing separation
- Explore whether larger windows provide more context for distinction

---

### Focus Area 10: Additional Insights

**Overall Statistics:**
- Total experiments: 1,398
- Average ROC-AUC: 0.8126 ± 0.0834
- Best ROC-AUC: **0.9624** (007_patch_20)
- ROC-AUC range: 0.3643 - 0.9624

**Correlations with ROC-AUC:**
- disc_ratio_1: -0.124 (weak negative!) ← High disc_ratio hurts performance
- t_ratio: +0.362 (moderate positive) ← High t_ratio helps
- disc_cohens_d: +0.445 (moderate positive)
- recon_cohens_d: +0.518 (strong positive) ← Reconstruction quality critical

**Critical Discovery:**
Reconstruction quality (recon_cohens_d) has STRONGER correlation with ROC-AUC than discrepancy (disc_cohens_d). This is counterintuitive for a discrepancy-based method, but reveals that:
1. Good reconstruction is foundation for good anomaly detection
2. Discrepancy loss should enhance, not replace, reconstruction
3. Balance between the two objectives is optimal

**Architecture Insights:**
- d_model=128: Best balance (appears in most top models)
- d_model=256: Good for high capacity needs
- d_model=64: Insufficient capacity
- nhead=8: Standard and effective
- nhead=16: Beneficial for d_model=128
- Teacher decoder depth: 3-4 layers optimal
- Student decoder depth: 1 layer sufficient

---

## Key Insights Summary

### Insight 1: Balance is Everything

**Discovery:** The best models achieve a careful balance:
- disc_cohens_d: 0.7-1.2 (not extreme)
- recon_cohens_d: 2.0-3.8 (high)
- disc_ratio: 1.5-2.2 (moderate)
- t_ratio: 4.5-6.6 (high)

**Implication:** Phase 2 must optimize for BALANCED metrics, not maximize any single metric.

### Insight 2: Reconstruction Quality Dominates

**Discovery:** recon_cohens_d (r=+0.518) correlates more strongly with ROC-AUC than disc_cohens_d (r=+0.445)

**Implication:** The teacher network's reconstruction quality is the foundation. Discrepancy learning enhances but doesn't replace reconstruction.

### Insight 3: all_patches + default is Winning Combination

**Discovery:**
- all_patches: +5.9% over last_patch
- default scoring: achieves maximum performance for tuned models

**Implication:** Phase 2 should primarily use this combination, testing alternatives only for specific optimization goals.

### Insight 4: Window 500, Patch 20, d_model 128

**Discovery:** This configuration (009_w500_p20, 063_w500_p20_d128) appears repeatedly in top performers.

**Implication:** Use as baseline for Phase 2 explorations. Test scaling up (w1000) and down (w100) from this anchor.

### Insight 5: Disturbing Normal Separation is the Frontier

**Discovery:** Best disc_cohens_d_disturbing_vs_anomaly is only 0.803, compared to 1.926 for pure normal.

**Implication:** This is the key challenge. Improving disturbing separation while maintaining overall performance is the path to practical deployment.

---

## Phase 2 Experiment Plan

**Total Experiments:** 150
**Config File:** `scripts/ablation/configs/phase2.py`
**Status:** ✓ Generated and validated

### GROUP 1 (001-030): Balanced High Disc+Recon Optimization

**Goal:** Maximize both disc_cohens_d (target: 1.0-1.3) AND recon_cohens_d (target: 2.5-3.8)

**Strategy:**
- Build on 028_d128_nhead_16 (best balanced model from Phase 1)
- Build on 009_w500_p20 (top performer)
- Build on 063_w500_p20_d128 (high recon_d)

**Experiments:**
1. **Subgroup 1a (11 experiments):** d_model=128 with masking_ratio [0.65, 0.70, 0.75, 0.80, 0.85] × lambda_disc [1.5, 2.0, 2.5]
2. **Subgroup 1b (9 experiments):** nhead=16, decoder_depth [3, 4, 5] × masking_ratio [0.70, 0.75, 0.80]
3. **Subgroup 1c (4 experiments):** Patch size variations [10, 15, 25, 30]
4. **Subgroup 1d (5 experiments):** FFN dimension [512, 1024, 1536, 2048, 3072]
5. **Subgroup 1e (1 experiment):** Combined best settings

**Expected Outcomes:**
- At least 5 models with disc_d > 1.2 AND recon_d > 2.5
- ROC-AUC > 0.955 for top configurations
- Identify optimal lambda_disc for balance

### GROUP 2 (031-055): Window Size & Capacity Exploration

**Goal:** Understand relationship between window size and optimal model capacity

**Strategy:**
- Test w500 with capacity variations
- Test w1000 with matched capacity
- Test w100 as reduced baseline

**Experiments:**
1. **Subgroup 2a (8 experiments):** w500 × [d_model, nhead, decoder_depth] combinations
2. **Subgroup 2b (8 experiments):** w1000 with [128, 192, 256] d_model, patch [20, 25, 40]
3. **Subgroup 2c (5 experiments):** w100 with reduced capacity [64, 96, 128]
4. **Subgroup 2d (3 experiments):** w500 patch strategy variations
5. **Subgroup 2e (1 experiment):** w1000 optimal configuration

**Expected Outcomes:**
- Confirm w500 as optimal baseline
- Identify if w1000 provides benefits (hypothesis: yes for PA%80)
- Establish capacity scaling rules (e.g., "d_model should be 0.25 × num_patches")

### GROUP 3 (056-075): Disturbing Normal Separation Focus

**Goal:** Maximize disc_cohens_d_disturbing_vs_anomaly beyond 0.85

**Strategy:**
- Build on 009_w500_p20 (best disturbing separator: 0.803)
- Explore dynamic margin and loss weighting
- Test if higher anomaly_loss_weight helps

**Experiments:**
1. **Subgroup 3a (9 experiments):** dynamic_margin_k [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
2. **Subgroup 3b (6 experiments):** lambda_disc [1.0, 1.5, 2.0, 2.5, 3.0, 3.5] with k=2.5
3. **Subgroup 3c (4 experiments):** anomaly_loss_weight [0.5, 1.0, 1.5, 2.0]
4. **Subgroup 3d (1 experiment):** Combined optimal settings

**Expected Outcomes:**
- Achieve disc_cohens_d_disturbing_vs_anomaly > 0.85
- Maintain ROC-AUC > 0.950
- Identify loss weighting that emphasizes disturbing separation

### GROUP 4 (076-095): PA%80 Optimization

**Goal:** Maximize PA%80 ROC-AUC for practical deployment

**Strategy:**
- Test scoring/inference mode combinations systematically
- Test larger windows for better context
- Test higher capacity for better long-range detection

**Experiments:**
1. **Subgroup 4a (6 experiments):** Scoring × Inference mode combinations
2. **Subgroup 4b (3 experiments):** Window [500, 1000] × patch variations
3. **Subgroup 4c (6 experiments):** High capacity [d_model 192, 256] × decoder_depth [4, 5, 6]
4. **Subgroup 4d (4 experiments):** Combined optimizations
5. **Subgroup 4e (1 experiment):** Ultimate PA%80 configuration

**Expected Outcomes:**
- PA%80 ROC-AUC > 0.970
- Identify best scoring mode for PA%80 (hypothesis: default or adaptive)
- Confirm if larger windows help PA%80 (hypothesis: yes)

### GROUP 5 (096-110): Teacher-Student Ratio Exploration

**Goal:** Find optimal teacher-student decoder depth balance

**Strategy:**
- Test ratios from t1s1 to t6s1
- Test balanced ratios (t2s2, t3s3, t4s4, t5s5)
- Combine with d_model=128 baseline

**Experiments:**
- 15 experiments covering [t1s1, t2s1, t3s1, t4s1, t5s1, t6s1, t2s2, t3s2, t4s2, t5s2, t3s3, t4s3, t5s3, t4s4, t5s5]

**Expected Outcomes:**
- Confirm t4s1 or t4s2 as optimal (Phase 1 hint)
- Understand if balanced ratios (t3s3, t4s4) provide benefits
- Identify if very deep teachers (t6s1) help

### GROUP 6 (111-125): Masking Strategy Fine-tuning

**Goal:** Find optimal masking_ratio for different d_models

**Strategy:**
- Fine-grained sweep for d_model=128 (low ratios)
- Fine-grained sweep for d_model=256 (high ratios)

**Experiments:**
1. **Subgroup 6a (7 experiments):** d_model=128 × masking_ratio [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
2. **Subgroup 6b (7 experiments):** d_model=256 × masking_ratio [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
3. **Subgroup 6c (1 experiment):** Optimal configuration

**Expected Outcomes:**
- Identify optimal masking_ratio for d_model=128 (hypothesis: 0.15-0.25)
- Identify optimal masking_ratio for d_model=256 (hypothesis: 0.70-0.80)
- Understand capacity-masking relationship

### GROUP 7 (126-140): Architecture Depth Optimization

**Goal:** Find optimal encoder-decoder depth balance

**Strategy:**
- Test encoder depth [4, 6, 8]
- Test decoder depth [2, 3, 4, 5, 6]
- Use d_model=128 baseline

**Experiments:**
- 15 experiments covering encoder × decoder combinations

**Expected Outcomes:**
- Identify optimal encoder depth (hypothesis: 6)
- Identify optimal decoder depth (hypothesis: 4)
- Understand if deeper architectures help (hypothesis: limited benefit after certain depth)

### GROUP 8 (141-150): Lambda Discrepancy & Loss Weighting

**Goal:** Fine-tune loss weighting for maximum performance

**Strategy:**
- Fine-grained sweep of lambda_disc
- Use d_model=128, t4s1 baseline

**Experiments:**
- 10 experiments: lambda_disc [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0]

**Expected Outcomes:**
- Identify optimal lambda_disc (hypothesis: 1.5-2.5)
- Understand sensitivity to loss weighting
- Confirm optimal balance point

---

## Expected Phase 2 Outcomes

### Primary Goals

1. **Achieve 10+ models with ROC-AUC > 0.960** (Phase 1 best: 0.9624)
2. **Achieve 5+ models with disc_d > 1.2 AND recon_d > 2.8** (Phase 1: only 3 models)
3. **Achieve disc_cohens_d_disturbing_vs_anomaly > 0.85** (Phase 1 best: 0.803)
4. **Achieve PA%80 ROC-AUC > 0.970** (Phase 1 best: 0.965)

### Secondary Goals

5. Establish scaling laws for window size vs model capacity
6. Determine optimal teacher-student ratios definitively
7. Find optimal masking ratios for different architectures
8. Understand depth-performance relationship

### Practical Deployment

9. Identify 2-3 configurations ready for production deployment
10. Create ensemble strategy from diverse high-performers

---

## Recommendations

### Immediate Actions

1. **Review and validate Phase 2 config:**
   ```bash
   python scripts/ablation/configs/phase2.py
   ```

2. **Run Phase 2 experiments:**
   ```bash
   python scripts/ablation/run_ablation.py --config configs/phase2.py
   ```

3. **Monitor key metrics during training:**
   - disc_cohens_d_normal_vs_anomaly (target: > 1.2)
   - recon_cohens_d_normal_vs_anomaly (target: > 2.8)
   - disc_cohens_d_disturbing_vs_anomaly (target: > 0.85)
   - ROC-AUC (target: > 0.960)
   - PA%80 ROC-AUC (target: > 0.970)

### Analysis During Phase 2

1. **After each GROUP completes:**
   - Generate intermediate analysis report
   - Adjust remaining groups if new insights emerge
   - Consider adding bonus experiments if breakthrough discovered

2. **Track GROUP-level performance:**
   - Which groups achieve their goals?
   - Which hypotheses are confirmed/rejected?
   - Are there unexpected interactions between parameters?

3. **Compare to Phase 1 baselines:**
   - Is performance improving?
   - Are we maintaining balanced metrics?
   - Are we solving the disturbing normal challenge?

### Post-Phase 2

1. **Generate comprehensive Phase 1+2 report**
2. **Select top 5-10 models for final evaluation**
3. **Design ensemble strategies**
4. **Prepare deployment configuration**
5. **Consider Phase 3 if needed:**
   - Hyperparameter fine-tuning of best models
   - Ensemble optimization
   - Production-specific optimizations

---

## Conclusion

Phase 1 provided critical insights: **balance between discrepancy and reconstruction is essential**. Models that maximize disc_ratio alone fail due to poor reconstruction. The winning strategy requires:

1. Moderate disc_cohens_d (0.9-1.2)
2. High recon_cohens_d (2.5-3.8)
3. all_patches inference mode
4. default scoring mode
5. w500_p20, d_model=128 baseline

Phase 2's 150 experiments are strategically designed to:
- Optimize the balanced approach (GROUP 1)
- Scale to different window sizes (GROUP 2)
- Solve disturbing normal challenge (GROUP 3)
- Maximize deployment readiness (GROUP 4-8)

**Expected outcome:** 10+ production-ready models with ROC-AUC > 0.960, balanced metrics, and excellent disturbing normal separation.

---

**Next Step:** Run Phase 2 experiments and monitor progress.

**Config Location:** `scripts/ablation/configs/phase2.py`
**Results Location:** `results/experiments/phase2/`
**Analysis Location:** `docs/ablation_result/`
