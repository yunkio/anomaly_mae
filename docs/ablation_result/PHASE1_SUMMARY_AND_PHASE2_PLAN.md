# Ablation Study: Phase 1 Summary & Phase 2 Plan

**Date:** 2026-01-27
**Total Phase 1 Experiments:** 1,392
**Planned Phase 2 Experiments:** 150

---

## Executive Summary

Phase 1 explored a wide range of hyperparameters across 1,392 experiments. The analysis revealed critical insights about model architecture, masking strategies, and performance trade-offs. Based on these findings, Phase 2 is designed with 150 targeted experiments organized into 7 thematic tracks.

### Key Findings

1. **Best Overall Performance:** `mask_after_encoder=False` with `d_model=128`, `nhead=16`, achieving **ROC-AUC=0.9624**
2. **Highest Discrepancy Ratio:** `mask_after_encoder=True` with `dynamic_margin_k=4.0`, achieving **disc_ratio=4.26** (but ROC-AUC=0.87)
3. **Trade-off Identified:** High disc_ratio correlates negatively with ROC-AUC (r=-0.45 with recon_ratio)
4. **Inference Mode:** `all_patches` consistently outperforms `last_patch` (0.8355 vs 0.7889 ROC-AUC)
5. **Window Size:** Larger windows (w=500) show excellent performance (ROC-AUC=0.9586), warrant further exploration

---

## Phase 1 Analysis Results

### Top 10 Models by Performance Metric

See detailed tables in:
- [phase1_top_models_tables.md](phase1_top_models_tables.md) - Top 10 models by ROC-AUC, disc_ratio, and teacher_ratio
- [phase1_deep_analysis.md](phase1_deep_analysis.md) - Comprehensive 10-point analysis

### Critical Insights by Analysis Point

#### 1. High Discrepancy Ratio Models

**Characteristics:**
- `mask_after_encoder = True` (100%)
- `d_model = 64` (concentrated)
- `dropout = 0.1-0.3` (higher values)
- `dynamic_margin_k = 2.2-4.0` (varied)
- `inference_mode = all_patches` (100%)

**Performance:**
- Disc Ratio: **4.26 average** (top 20)
- ROC-AUC: **0.84 average** (lower than overall 0.81)
- PA%80 F1: **0.46 average**

**Key Finding:** High disc_ratio achievable but at cost of ROC-AUC

#### 2. Models with High Disc AND Recon Cohen's d

**Only 3 models** achieved both high discriminative power:
- Experiment: `028_d128_nhead_16_mask_before_*`
- `d_model = 128`, `nhead = 16`, `mask_after = False`
- ROC-AUC: **0.9420**
- PA%80 F1: **0.6617**
- These represent the sweet spot for balanced performance

**Key Finding:** Simultaneous high disc and recon Cohen's d is rare but achievable with specific configs

#### 3. Scoring Mode & Window Size Sensitivity

**Scoring Mode Impact:**
- `adaptive`: Best overall ROC-AUC (0.8250)
- `default`: Most variable performance
- `normalized`: Stable mid-range performance
- Max sensitivity (std): **0.0660** ROC-AUC variance

**Window Size Results (234 experiments):**
- `w500_p20`: ROC-AUC **0.9586** (2nd best overall)
- Average window experiments: ROC-AUC 0.7691
- Larger windows need more exploration

**Key Finding:** Scoring mode matters significantly; larger windows promising but underexplored

#### 4. Disturbing Normal vs Anomaly Discrimination

**Top Models (disc_cohens_d_disturbing_vs_anomaly):**
- `w500_p20` experiments: **0.8029** (highest)
- `mask_after=True` with high k: **0.71** average
- Maintains good overall performance (ROC-AUC: 0.8753)

**Common Characteristics:**
- `mask_after_encoder`: Mixed (12 True, 3 False)
- `patchify_mode = patch_cnn` (100%)
- `inference_mode = all_patches` (100%)

**Key Finding:** Window size and margin k critical for disturbing normal discrimination

#### 5. High PA%80 with High Disc Ratio

**70 models** achieved both (>75th percentile):
- Average PA%80 F1: **0.5728**
- Average disc_ratio: **2.8319**
- Average ROC-AUC: **0.8704**

**Finding:** PA%80 and disc_ratio can be jointly optimized with careful parameter selection

#### 6. Architectural Parameter Relationships

**Correlations with ROC-AUC:**
- `num_student_decoder_layers`: **+0.071** (small positive)
- `d_model`: **-0.0004** (essentially zero)
- `masking_ratio`: **+0.005** (essentially zero)

**Performance by d_model:**
- 64: 0.8119 (most common, 1230 experiments)
- 128: 0.8182 (good balance)
- 256: 0.8416 (best, but only 12 experiments)
- 96: 0.7025 (worst, 6 experiments)

**Performance by masking_ratio:**
- 0.08: ROC-AUC 0.8132, disc_ratio 2.64 (good balance)
- 0.2: ROC-AUC 0.8190, disc_ratio 2.05 (most common)
- 0.3: ROC-AUC 0.8543, disc_ratio 1.90 (higher ratio better)

**Key Finding:** Weak correlations suggest parameter interactions; non-linear relationships

#### 7. mask_after_encoder Impact

**Comparison:**

| Metric | mask_after=True | mask_after=False | Difference |
|--------|----------------|------------------|------------|
| ROC-AUC | 0.7861 | **0.8837** | **-0.0977** ⬇️ |
| PA%80 F1 | 0.3741 | **0.5321** | **-0.1580** ⬇️ |
| Disc Ratio | **2.3247** | 1.4344 | **+0.8903** ⬆️ |
| Recon Ratio | 1.6072 | **4.9658** | **-3.3586** ⬇️ |
| disc_cohens_d | **1.0818** | 0.4639 | **+0.6178** ⬆️ |
| recon_cohens_d | 0.6425 | **2.0973** | **-1.4549** ⬇️ |

**Key Finding:** `mask_after=True` increases disc_ratio at severe cost to overall performance

#### 8. Scoring & Inference Mode Sensitivity

**Inference Mode:**
- `all_patches`: ROC-AUC **0.8355**, disc_ratio 2.39
- `last_patch`: ROC-AUC 0.7889, disc_ratio 1.78
- Difference: **+0.0466** ROC-AUC for all_patches

**Most Sensitive Experiments:**
- `feature_wise` masking: **0.2639** ROC-AUC difference between modes
- High k experiments: **0.14-0.15** difference

**Key Finding:** `all_patches` superior; some configs highly mode-sensitive

#### 9. High Performance + High Disturbing Discrimination

**91 models** achieved both (>75th percentile):
- ROC-AUC: **0.9157**
- PA%80 F1: **0.6000**
- disc_cohens_d_disturbing: **0.5960**

**Common Pattern:**
- `mask_after_encoder = False` (9/15 top models)
- `patchify_mode = patch_cnn` (100%)
- `inference_mode = all_patches` (12/15)

**Key Finding:** Best overall models maintain strong disturbing discrimination

#### 10. Additional Insights

**Patchify & Masking Strategy:**
- `patch` vastly superior to `feature_wise` (0.8138 vs 0.6234 ROC-AUC)

**Extreme Parameter Effects:**
- Higher `dropout` → higher disc_ratio, lower ROC-AUC
- `d_model`: Non-linear (medium values best)
- `lambda_disc`: Needs systematic exploration (weak correlation)

**Best Balanced Models (Composite Score):**
- All have `mask_after = False`
- All have `patchify_mode = patch_cnn`
- Most use `inference_mode = last_patch`

---

## Phase 2 Experimental Design

Based on Phase 1 insights, Phase 2 consists of **150 experiments** organized into **7 focused tracks**.

### Track 1: Balanced Performance Optimization (30 experiments)

**Goal:** Further optimize `mask_after=False` configurations

**Rationale:** Best overall models use mask_before; systematically explore parameter space

**Experiments:**
1. Vary `d_model`: 64, 96, 128, 160, 192, 256 (6 configs × modes)
2. Vary `num_teacher_decoder_layers`: 1-6 (6 configs × modes)
3. Vary `num_student_decoder_layers`: 1-4 (4 configs × modes)
4. Vary `nhead`: 1, 4, 8, 16, 32 (5 configs × modes)
5. Vary `learning_rate`: 0.0005-0.01 (5 configs × modes)
6. Vary `dim_feedforward`: 128-1024 (4 configs × modes)

### Track 2: Window Size Exploration (25 experiments)

**Goal:** Systematically explore larger windows

**Rationale:** `w500_p20` achieved ROC-AUC=0.9586; larger windows underexplored

**Experiments:**
1. Window 500 with patch sizes: 10, 20, 25, 50, 100 (5 configs)
2. Window 1000 with patch sizes: 10, 20, 25, 50, 100 (5 configs)
3. Window 1500 with patch sizes: 10, 20, 25, 50, 150 (5 configs)
4. Window scaling with d_model: (200, 96), (500, 128), (1000, 160), (1500, 192), (2000, 256) (5 configs)
5. Window + encoder depth combinations (5 configs)

### Track 3: High Disc Ratio with Better ROC-AUC (25 experiments)

**Goal:** Improve ROC-AUC while maintaining high disc_ratio

**Rationale:** disc_ratio=4.26 achievable but ROC-AUC too low (0.84)

**Experiments:**
1. High k (4.0) with varying dropout: 0.05-0.3 (5 configs)
2. High k (4.0) with varying d_model: 64-192 (5 configs)
3. Varying k (2.0-4.0) with optimal d_model=128 (5 configs)
4. High k with decoder variations (5 configs)
5. High k with masking_ratio variations (5 configs)

### Track 4: Disturbing Normal Discrimination (20 experiments)

**Goal:** Optimize disturbing normal vs anomaly separation

**Rationale:** Critical for real-world deployment; w500 showed promise

**Experiments:**
1. Window 500 with varying k: 1.5-3.5 (5 configs)
2. Window 500 with varying d_model: 64-192 (5 configs)
3. Window 1000 with varying k: 1.5-3.5 (5 configs)
4. Encoder depth variations for disturbing detection (5 configs)

### Track 5: Architectural Depth Exploration (20 experiments)

**Goal:** Find optimal encoder-decoder architecture

**Rationale:** Weak correlations suggest unexplored depth combinations

**Experiments:**
1. Deep encoders: 3-8 layers (5 configs)
2. Deep teacher decoders: 3-8 layers (5 configs)
3. Deep student decoders: 2-6 layers (5 configs)
4. Balanced deep architectures (5 configs)

### Track 6: Masking Ratio Fine-tuning (15 experiments)

**Goal:** Find optimal masking ratio for different scenarios

**Rationale:** 0.08-0.3 range promising; needs fine-grained search

**Experiments:**
1. Fine-grained ratios: 0.05-0.3 (10 configs)
2. Masking ratio with mask_after=True (5 configs)

### Track 7: Lambda_disc Systematic Exploration (15 experiments)

**Goal:** Determine optimal lambda_disc values

**Rationale:** Weak correlation suggests non-linear relationship

**Experiments:**
1. Lambda_disc with mask_before: 0.05-3.0 (8 configs)
2. Lambda_disc with mask_after: 0.3-3.0 (7 configs)

---

## Recommendations for Phase 2 Execution

### Prioritization

**High Priority Tracks (Run First):**
1. **Track 1** - Balanced Performance Optimization
2. **Track 2** - Window Size Exploration
3. **Track 4** - Disturbing Normal Discrimination

**Medium Priority:**
4. **Track 3** - High Disc Ratio Optimization
5. **Track 6** - Masking Ratio Fine-tuning

**Lower Priority (If Resources Available):**
6. **Track 5** - Architectural Depth
7. **Track 7** - Lambda_disc Exploration

### Expected Outcomes

**Track 1:** Should find ROC-AUC > 0.97 configurations

**Track 2:** Identify optimal window size for different scenarios; expect ROC-AUC improvements with larger windows

**Track 3:** Achieve disc_ratio > 3.5 with ROC-AUC > 0.90

**Track 4:** Maximize disc_cohens_d_disturbing_vs_anomaly while maintaining high ROC-AUC

**Track 5:** Discover if deeper architectures beneficial

**Track 6:** Find masking_ratio sweet spot for each scenario

**Track 7:** Establish lambda_disc guidelines

### Evaluation Metrics

For each experiment, prioritize:

1. **Primary Metrics:**
   - ROC-AUC (anomaly detection performance)
   - PA%80 F1 (point-adjusted performance)
   - disc_cohens_d_disturbing_vs_anomaly (disturbing discrimination)

2. **Secondary Metrics:**
   - disc_ratio (discrepancy ratio)
   - recon_ratio (teacher reconstruction ratio)
   - disc_cohens_d_normal_vs_anomaly (normal discrimination)

3. **Composite Score:**
   - 0.4 × ROC-AUC + 0.3 × PA%80 + 0.15 × disc_ratio + 0.15 × recon_ratio (normalized)

### Success Criteria

Phase 2 will be considered successful if we achieve:

1. ✅ ROC-AUC ≥ 0.97 (improvement from 0.9624)
2. ✅ PA%80 F1 ≥ 0.80 (improvement from 0.7867)
3. ✅ disc_ratio ≥ 3.0 with ROC-AUC ≥ 0.92
4. ✅ disc_cohens_d_disturbing_vs_anomaly ≥ 0.85 with ROC-AUC ≥ 0.93

---

## Configuration Files

**Phase 2 Experiments:** `scripts/ablation/configs/phase2/20260127_141642_phase2.py`

**Analysis Results:**
- Top Models Tables: `docs/ablation_result/phase1_top_models_tables.md`
- Deep Analysis: `docs/ablation_result/phase1_deep_analysis.md`
- Summary Statistics: `docs/ablation_result/phase1_statistics.json`

---

## Next Steps

1. **Review Phase 2 Configurations:** Validate experiment configurations in phase2 config file
2. **Resource Allocation:** Ensure sufficient compute for 150 experiments
3. **Run High Priority Tracks:** Execute Tracks 1, 2, 4 first
4. **Monitor Progress:** Track metrics during training
5. **Iterative Analysis:** Analyze results after each track completion
6. **Phase 3 Planning:** Use Phase 2 insights for potential refinement phase

---

## Appendix: Phase 1 Statistics

**Total Experiments:** 1,392

**Parameter Coverage:**
- mask_after_encoder: True (1,020), False (372)
- d_model: 16, 32, 64, 96, 128, 192, 256
- masking_ratio: 0.05-0.8 (14 values)
- dropout: 0.0-0.3 (4 values)
- dynamic_margin_k: 1.0-4.0 (5 values)
- num_encoder_layers: 1-4
- num_teacher_decoder_layers: 1-5
- num_student_decoder_layers: 1-2
- scoring_mode: default, adaptive, normalized
- inference_mode: all_patches, last_patch

**Performance Ranges:**
- ROC-AUC: 0.3643 - 0.9624
- PA%80 F1: 0.0422 - 0.7867
- disc_ratio: 0.7203 - 4.2588
- recon_ratio: 1.0147 - 6.5970

**Best Overall Configuration:**
```python
{
    'experiment': '007_patch_20_mask_before_default_all',
    'roc_auc': 0.9624,
    'pa_80_f1': 0.7297,
    'disc_ratio': 1.7369,
    'recon_ratio': 6.0038,
    'mask_after_encoder': False,
    'd_model': 64,
    'num_patches': 20,
    'inference_mode': 'all_patches',
    'scoring_mode': 'default'
}
```

---

**Generated:** 2026-01-27
**Analysis by:** Claude Sonnet 4.5
