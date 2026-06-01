# Phase 2 Ablation Study: Experiment Plan

**Date**: 2026-01-31
**Total Configs**: 150
**Evaluations per Config**: 2 scoring (default, normalized) × 2 mask timing (mask_before, mask_after) = 4
**Total Evaluations**: 600
**Based on**: Phase 1 Analysis (1,014 evaluations, 169 base configs)

> **⚠️ CORRECTION (2026-01-31)**: Diagnostic testing revealed that `num_encoder_layers=2` causes disc_d collapse at w500 (disc_d: 2.44→0.21, roc: 0.9855→0.9095). Combined enc2+td4 is catastrophic (roc=0.7592). The deeper encoder makes teacher/student outputs too similar, killing discriminative power. Additionally, `lr=5e-3` is too aggressive at w500 (roc drops 0.040). **Corrected defaults**: `num_encoder_layers=1` (not 2), `learning_rate=2e-3` (not 5e-3). The config file `20260131_121350_phase2.py` has been updated accordingly. Experiment descriptions below still reference old defaults (enc2, lr0.005) but the actual configs use corrected values.

---

## Phase 1 Key Insights Summary

### Core Findings
1. **mask_after + normalized scoring is superior** to mask_before + default (H12). mask_after wins 117/169 with normalized scoring on PA80_F1 (+0.0815 gain).
2. **disc_SNR is the best single predictor** of roc_auc (r=0.492), surpassing disc_d (0.481) and disc_ratio (0.411).
3. **Window 500 uniquely excels at disturbing-normal separation** (dist_d=2.55 best), capturing temporal boundary patterns w100 misses (H3, H4).
4. **Deeper teacher decoder helps at w100 but hurts at w500** — depth scaling inverts with window size (H6).
5. **d_model=128+ is critical for w500** (+0.048 roc, +0.84 disc_d vs d64) (H7).
6. **Masking ratio sweet spot is 0.08–0.15** for disc_SNR. mr=0.20 is adequate but sub-optimal (H8).
7. **lr=0.003–0.005 consistently outperforms lr=0.002** — learning rate was the most under-explored parameter (H15).
8. **nhead=8 is the best by mean performance** (roc=0.9694, pa80=0.7289) with lowest variance.
9. **lambda_disc≥2.0 eliminates the scoring mode gap** for mask_after (after+default rises from 0.9324 to 0.9719).
10. **Optimal patches per window is ~4–25** — too many patches create aggregation noise (H16).
11. **sd=2 asymmetrically benefits mask_before** (+0.024 roc vs +0.002 for mask_after).
12. **Encoder depth el=2–3 modestly improves** over el=1, but el=4 degrades. Encoder-decoder depth budget exists.
13. **dynamic_margin_k=3.0 best for mask_after** (disc_d +0.42 vs k=1.5), while hurting mask_before.
14. **anomaly_loss_weight=2.0 boosts mask_after** disc_d by +22%.
15. **Patch masking is the only viable strategy** — feature_wise collapses.

### Unresolved Questions for Phase 2
- Can discrepancy loss be made **consistently** useful for detection (not just separation)?
- Does the mask_after advantage hold at larger model scales and deeper encoders?
- Is softplus margin competitive with dynamic margin?
- How does patch_size interact with model capacity and window size at scale?
- Can deeper encoders (el=3–4) compensate for shallower decoders?
- Does the w500 + d128 advantage extend to d256/d512?

---

## New Default Parameters

These defaults are derived from Phase 1 optimal findings and serve as both the Phase 2 baseline and the new model defaults.

| Parameter | Value | Phase 1 Default | Rationale |
|-----------|-------|-----------------|-----------|
| window_size | 500 | 100 | Best disturbing separation; validated by H3 |
| patch_size | 20 | 10 | Optimal for w500 (25 patches); H16 |
| masking_ratio | 0.15 | 0.20 | SNR sweet spot 0.08–0.15; H8 |
| d_model | 128 | 64 | Critical for w500 performance; H7 |
| dim_feedforward | 512 | 256 | d_model × 4 |
| nhead | 8 | 2 | Best mean roc (0.9694); lowest variance |
| encoder_layers | 2 | 1 | el=2–3 improves over el=1 |
| teacher_decoder | 4 | 2 | Best overall by mean and max; monotonically improves td=2→5 |
| student_decoder | 1 | 1 | Baseline; sd=2 tested separately |
| learning_rate | 0.005 | 0.002 | Highest mean roc (0.9739); H15 |
| margin_type | dynamic | dynamic | Unchanged; softplus tested |
| dynamic_margin_k | 2.0 | 1.5 | Higher k helps mask_after; k=2.0 as moderate increase |
| lambda_disc | 2.0 | 0.5 | Eliminates scoring mode gap for mask_after |
| dropout | 0.15 | 0.1 | Between 0.1 (too low) and 0.2 (phase1 best mask_after) |
| weight_decay | 1e-05 | 1e-05 | No significant effect found |
| anomaly_loss_weight | 2 | 1 | Boosts mask_after disc_d +22% |
| scoring | normalized | default | H12: mask_after + normalized is optimal |

### Fixed Parameters (not varied)
| Parameter | Value |
|-----------|-------|
| patchify_mode | patch_cnn |
| masking_strategy | patch |
| force_mask_anomaly | True |
| patch_level_loss | True |
| shared_mask_token | False |
| num_shared_decoder_layers | 0 |
| num_epochs | 50 |
| margin | 0.5 |
| warmup_epochs | 10 |
| teacher_only_warmup_epochs | 3 |
| dropout | 0.15 |

### Parameter Space Constraints

| Parameter | Allowed Values | Notes |
|-----------|---------------|-------|
| window_size | 100, 200, 500 | w1000 excluded (non-viable in phase1) |
| masking_ratio | 0.1, 0.15, 0.2 | Narrowed from phase1's 0.05–0.80 |
| d_model | 64, 128, 256, 512 | dim_feedforward = d_model × 4 |
| patch_size | 5, 10, 20, 25 | Must divide window_size evenly |
| nhead | 4, 8, 16 | **Note: nhead=12 excluded** — not divisible by any allowed d_model |
| teacher_decoder | 2, 3, 4, 5, 6 | |
| student_decoder | 1, 2, 3, 4 | Must be ≤ teacher_decoder |
| encoder_layers | 2, 3, 4 | |
| margin_type | dynamic, softplus | |
| anomaly_loss_weight | ≥ 2 | |
| lambda_disc | unconstrained | |
| dynamic_margin_k | unconstrained | |
| learning_rate | unconstrained | |

**nhead=12 Incompatibility**: 64/12=5.33, 128/12=10.67, 256/12=21.33, 512/12=42.67 — none are integers. Standard transformer requires d_model % nhead == 0. nhead=12 is excluded from all experiments. Effective nhead options: {4, 8, 16}.

---

## Notation

All experiments use the default parameters unless explicitly overridden. Each experiment description lists only the parameters that differ from default.

**Shorthand**: w=window_size, p=patch_size, d=d_model, ff=dim_feedforward, nh=nhead, enc=encoder_layers, td=teacher_decoder, sd=student_decoder, mr=masking_ratio, lr=learning_rate, λ=lambda_disc, k=dynamic_margin_k, alw=anomaly_loss_weight, mt=margin_type

---

## Experiment List (150 Configs)

### Group 1: Baseline & Reference (Configs 001–005)

Establishes the new default baseline and compares against Phase 1 top performers to quantify the improvement from the new default parameter set.

---

#### Config 001 — New Default Baseline
**Changes**: None (all default parameters)
**Rationale**: Establishes the Phase 2 baseline. This is the new default config combining all Phase 1 insights: w500/p20/d128/nh8/enc2/td4/lr0.005/mr0.15/λ2.0/k2.0/alw2. Every other experiment is compared against this.
**Validates**: Whether the combined Phase 1 optimal parameters work synergistically or interfere with each other.

#### Config 002 — Phase 1 Old Default
**Changes**: w=100, p=10, d=64, ff=256, nh=2, enc=2, td=2, lr=0.002, mr=0.2, λ=0.5, k=1.5, alw=1 (note: alw=1 violates ≥2 constraint, use alw=2), dropout=0.15
**Rationale**: The Phase 1 baseline config adapted to Phase 2 constraints (enc=2 instead of 1, alw=2 instead of 1, dropout=0.15). Quantifies how much the new defaults improve over the old approach.
**Validates**: Magnitude of improvement from Phase 1 → Phase 2 default optimization.

#### Config 003 — Phase 1 Best roc_auc Adapted
**Changes**: w=100, p=10, d=64, ff=256, nh=4, td=4, mr=0.2, lr=0.002, λ=0.5, k=1.5
**Rationale**: Phase 1's single best roc_auc model (151_pa80_t4s1_nhead4, roc=0.9827) adapted to Phase 2 constraints. Tests whether the new scoring/mask evaluation framework changes the ranking.
**Validates**: Whether Phase 1 champions remain competitive under Phase 2 evaluation.

#### Config 004 — Phase 1 Best PA80_F1 Adapted
**Changes**: w=100, p=10, d=64, ff=256, nh=4, td=3, mr=0.1, lr=0.002, λ=0.5, k=1.5
**Rationale**: Phase 1's best PA80_F1 model (166_optimal_combo_3, PA80=0.8504) adapted. Tests whether early detection advantage persists.
**Validates**: Stability of PA80_F1 optimization across evaluation frameworks.

#### Config 005 — Default at w100
**Changes**: w=100, p=20
**Rationale**: The new default architecture (d128/nh8/enc2/td4/lr0.005) but at w100 with p=20 (5 patches). Tests whether the larger model designed for w500 is also effective at w100, or if it overfits the smaller input.
**Validates**: H7 inverse — is a w500-optimized architecture oversized for w100?

---

### Group 2: Window Size × Model Capacity (Configs 006–020)

Tests H7 (larger windows need larger models) and explores the scaling relationship between window size and model capacity. Also validates H3 (w500 for disturbing separation) and H4 (tension between disc_d and dist_d across window sizes).

---

#### Config 006 — w100, p=10
**Changes**: w=100, p=10
**Rationale**: w100 with 10 patches (vs default 25 patches at w500). Tests the classic w100/p10 setup with the new larger architecture.
**Validates**: Whether the new default architecture (d128/nh8/td4) provides gains at w100/p10.

#### Config 007 — w100, p=5
**Changes**: w=100, p=5
**Rationale**: w100 with 20 patches. Phase 1 showed p=5 at w100 gives high disc_d (3.26) but not highest roc. Tests this with the stronger architecture.
**Validates**: Patch granularity effect at w100 with large model — does d128 handle the noise from 20 patches better than d64?

#### Config 008 — w100, p=25
**Changes**: w=100, p=25
**Rationale**: w100 with only 4 patches. Phase 1 showed p=25/w100 had highest roc (0.9718) despite lowest disc_d. Tests with new architecture.
**Validates**: Whether fewer patches consistently improve roc even with larger models (H16).

#### Config 009 — w200, p=20
**Changes**: w=200, p=20
**Rationale**: w200 with 10 patches. w200 was severely under-tested in Phase 1 (only n=12). Tests the intermediate window size with default architecture.
**Validates**: Whether w200 fills the performance gap between w100 and w500 linearly or shows distinct characteristics.

#### Config 010 — w200, p=10
**Changes**: w=200, p=10
**Rationale**: w200 with 20 patches. More patches at intermediate window. Compares with 009 to measure patch granularity effect at w200.
**Validates**: Optimal patch count at w200; whether the ~10 patches/window optimum holds.

#### Config 011 — w200, p=25
**Changes**: w=200, p=25
**Rationale**: w200 with only 8 patches. Fewest patches at w200.
**Validates**: Whether p=25's "fewer patches = higher roc" trend extends to w200.

#### Config 012 — w500, d=64, ff=256
**Changes**: d=64, ff=256
**Rationale**: Deliberately undersized model for w500. Phase 1 showed d64 at w500 gives roc=0.8920 vs d128's 0.9027. Tests with all other new defaults (higher lr, λ, k, alw).
**Validates**: H7 — whether the new training params (lr=0.005, λ=2.0) can compensate for smaller model at w500.

#### Config 013 — w500, d=256, ff=1024
**Changes**: d=256, ff=1024
**Rationale**: Oversized model for w500. Phase 1 had only n=6 at d=256. Tests whether even larger capacity further improves w500.
**Validates**: Whether d256 provides additional gains beyond d128 at w500, or if returns diminish.

#### Config 014 — w500, d=512, ff=2048
**Changes**: d=512, ff=2048
**Rationale**: Very large model at w500. Completely untested in Phase 1. Tests upper bound of model capacity scaling.
**Validates**: Scaling law — does performance continue to improve with d512, or does overfitting dominate at 50 epochs?

#### Config 015 — w100, d=64, ff=256
**Changes**: w=100, p=20, d=64, ff=256
**Rationale**: Small model at w100 with p=20 (5 patches). The "minimal" configuration. Phase 1's most common setup was similar (w100/d64) but with p=10.
**Validates**: Lower bound performance with new training params at small window.

#### Config 016 — w100, d=256, ff=1024
**Changes**: w=100, p=20, d=256, ff=1024
**Rationale**: Large model at small window. Tests whether excess capacity helps or hurts at w100.
**Validates**: H7 inverse — overfitting risk with large model / small window.

#### Config 017 — w100, d=512, ff=2048
**Changes**: w=100, p=20, d=512, ff=2048
**Rationale**: Very large model at small window. Expected to overfit but quantifies the degree.
**Validates**: Upper limit of model size at w100; establishes overfitting boundary.

#### Config 018 — w200, d=64, ff=256
**Changes**: w=200, p=20, d=64, ff=256
**Rationale**: Small model at w200. Tests whether d64 is sufficient for the intermediate window.
**Validates**: Minimum viable model capacity for w200.

#### Config 019 — w200, d=256, ff=1024
**Changes**: w=200, p=20, d=256, ff=1024
**Rationale**: Large model at w200. Tests capacity scaling at intermediate window.
**Validates**: Whether w200 benefits from large models like w500 does.

#### Config 020 — w500, d=256, ff=1024, nh=16
**Changes**: d=256, ff=1024, nh=16
**Rationale**: Large model with many heads at w500. nh=16 at d256 gives head_dim=16 (same as default d128/nh8). Tests whether scaling both width and heads helps.
**Validates**: Whether the head_dim=16 sweet spot extends to larger models.

---

### Group 3: Encoder-Decoder Depth Ratio (Configs 021–040)

Tests H6 (depth scaling inverts with window size), the encoder-decoder depth budget hypothesis, and the novel idea of deeper encoders relative to decoders (user item 6). Phase 1 found el=1 was baseline with el=2-3 slightly better, and decoder depth was the primary driver. This group systematically explores the enc×td space.

---

#### Config 021 — enc=2, td=2
**Changes**: td=2
**Rationale**: Shallow decoder with default encoder. Phase 1 showed td=2 is optimal at w500 (H6). Tests whether this holds with the new larger architecture.
**Validates**: H6 — whether shallow decoder is still best at w500 with d128/nh8.

#### Config 022 — enc=2, td=3
**Changes**: td=3
**Rationale**: Moderate decoder depth. td=3 was optimal at w100 in Phase 1.
**Validates**: td=3 performance at w500 with improved baseline.

#### Config 023 — enc=2, td=5
**Changes**: td=5
**Rationale**: Deeper than default. Phase 1 showed td=5 had highest mean roc (0.9682) but was under-sampled (n=6).
**Validates**: Whether td=5 outperforms td=4 with proper architecture support (d128/nh8).

#### Config 024 — enc=2, td=6
**Changes**: td=6
**Rationale**: Deepest decoder. Completely untested in Phase 1. Probes whether the monotonic td improvement continues beyond 5.
**Validates**: Upper bound of decoder depth benefit.

#### Config 025 — enc=3, td=2
**Changes**: enc=3, td=2
**Rationale**: Deep encoder, shallow decoder. Novel configuration — Phase 1 showed enc×td interact (total depth budget). Tests whether a deeper encoder can compensate for a shallower decoder.
**Validates**: Encoder-decoder depth budget hypothesis; whether encoder depth can substitute for decoder depth.

#### Config 026 — enc=3, td=3
**Changes**: enc=3, td=3
**Rationale**: Balanced 3+3 configuration. Phase 1 showed el=2+td=3 peaked at roc=0.9548. Tests deeper encoder with moderate decoder.
**Validates**: Whether balanced depth outperforms the asymmetric default (enc2/td4).

#### Config 027 — enc=3, td=4
**Changes**: enc=3
**Rationale**: Deep encoder with default decoder. Phase 1 showed el=3 mean roc=0.9579 (best encoder depth).
**Validates**: Whether adding encoder depth on top of td=4 provides additive benefit or hits the depth budget.

#### Config 028 — enc=3, td=5
**Changes**: enc=3, td=5
**Rationale**: Deep both. Total depth = 8 layers. Tests whether very deep models help at w500.
**Validates**: Total depth scaling at w500; overfitting risk with 8 transformer layers.

#### Config 029 — enc=3, td=6
**Changes**: enc=3, td=6
**Rationale**: Maximum encoder + maximum decoder combination. Total depth = 9.
**Validates**: Extreme depth behavior; likely diminishing returns or degradation.

#### Config 030 — enc=4, td=2
**Changes**: enc=4, td=2
**Rationale**: Very deep encoder, shallow decoder. The most extreme "encoder-heavy" configuration. Phase 1 showed el=4 degrades (roc=0.9342), but that was with el=4+td=2 at d=64. Tests at d=128.
**Validates**: Whether d128 rescues the deep encoder configuration that failed at d=64 (user item 6).

#### Config 031 — enc=4, td=3
**Changes**: enc=4, td=3
**Rationale**: Deep encoder, moderate decoder. Total depth = 7.
**Validates**: enc=4 with deeper decoder; whether the Phase 1 el=4 degradation was due to insufficient decoder.

#### Config 032 — enc=4, td=4
**Changes**: enc=4
**Rationale**: Deep encoder with default decoder. Total depth = 8.
**Validates**: enc=4+td=4 at d=128 — the most "encoder-dominant" with equal depth allocation.

#### Config 033 — enc=4, td=5
**Changes**: enc=4, td=5
**Rationale**: Very deep everywhere. Total depth = 9.
**Validates**: Whether enc=4 works when paired with sufficiently deep decoder.

#### Config 034 — enc=4, td=6
**Changes**: enc=4, td=6
**Rationale**: Maximum depth configuration. Total = 10 layers.
**Validates**: Absolute upper bound of depth at this model width.

#### Config 035 — enc=2, td=2, w=100
**Changes**: w=100, p=20, td=2
**Rationale**: Shallow decoder at w100. Phase 1 showed td=2 is baseline at w100.
**Validates**: H6 at w100 — confirms depth scaling direction differs from w500.

#### Config 036 — enc=2, td=4, w=100
**Changes**: w=100, p=20
**Rationale**: Default decoder at w100. Same as Config 005 (intentional duplication for depth analysis grouping).
**Validates**: td=4 at w100 with new architecture — does the Phase 1 finding (td=3 best at w100) still hold?

#### Config 037 — enc=2, td=6, w=100
**Changes**: w=100, p=20, td=6
**Rationale**: Deepest decoder at w100. Tests extreme depth at small window.
**Validates**: Whether td=6 at w100 overfits or continues the td improvement trend.

#### Config 038 — enc=3, td=4, w=100
**Changes**: w=100, p=20, enc=3
**Rationale**: Deep encoder at w100. Compares with 036 (enc=2) to measure encoder depth contribution at small window.
**Validates**: Encoder depth benefit at w100.

#### Config 039 — enc=4, td=4, w=100
**Changes**: w=100, p=20, enc=4
**Rationale**: Very deep encoder at w100. Total depth = 8 with small input.
**Validates**: Whether deep encoder overfits at w100.

#### Config 040 — enc=4, td=2, w=100
**Changes**: w=100, p=20, enc=4, td=2
**Rationale**: Encoder-heavy at w100. Tests the "encoder does the work, decoder is minimal" approach.
**Validates**: Encoder-dominant architecture viability at small window (user item 6).

---

### Group 4: Discrepancy Loss Optimization (Configs 041–060)

Directly addresses user item (1): finding settings where discrepancy loss is maximally useful for detection. Also addresses item (5) on disc_SNR. Varies lambda_disc, dynamic_margin_k, and anomaly_loss_weight to optimize the discrepancy signal.

---

#### Config 041 — λ=0.5 (low disc weight)
**Changes**: λ=0.5
**Rationale**: Phase 1 baseline lambda. Tests whether the Phase 1 finding (λ=0.5 requires adaptive scoring to compensate) persists with the new architecture.
**Validates**: λ=2.0 vs λ=0.5 gap magnitude with improved architecture.

#### Config 042 — λ=1.0
**Changes**: λ=1.0
**Rationale**: Intermediate disc weight. Maps the λ response curve between 0.5 and 2.0.
**Validates**: Linearity of lambda_disc effect on performance.

#### Config 043 — λ=3.0
**Changes**: λ=3.0
**Rationale**: Higher disc weight. Phase 1 showed λ=3.0 gave roc=0.9720 (near-identical to λ=2.0's 0.9721) with n=6.
**Validates**: Whether λ=3.0 provides marginal gains over λ=2.0 at scale.

#### Config 044 — λ=4.0
**Changes**: λ=4.0
**Rationale**: Aggressive disc weight. Untested in Phase 1. Tests whether very high λ over-emphasizes discrepancy at the cost of reconstruction quality.
**Validates**: Upper bound of useful lambda_disc.

#### Config 045 — λ=5.0
**Changes**: λ=5.0
**Rationale**: Very aggressive disc weight. Extreme test — does the model still train stably when disc loss dominates?
**Validates**: Stability boundary of lambda_disc.

#### Config 046 — k=1.0 (low margin)
**Changes**: k=1.0
**Rationale**: Lowest margin target. Phase 1 showed k=1.0 gives roc=0.9691 (after), competitive but lower disc_d.
**Validates**: Whether tight margins produce a more precise but weaker disc signal.

#### Config 047 — k=1.5
**Changes**: k=1.5
**Rationale**: Phase 1 default margin. Establishes comparison with the new k=2.0 default.
**Validates**: k=2.0 vs k=1.5 improvement magnitude.

#### Config 048 — k=3.0
**Changes**: k=3.0
**Rationale**: Phase 1's best mask_after disc_d (2.68). Tests with improved architecture.
**Validates**: Whether k=3.0's disc_d advantage persists and translates to detection.

#### Config 049 — k=4.0
**Changes**: k=4.0
**Rationale**: Aggressive margin. Phase 1 showed k=4.0 slightly lower PA80_F1 than k=3.0 despite similar roc.
**Validates**: Whether k=4.0 over-stretches the margin target.

#### Config 050 — k=5.0
**Changes**: k=5.0
**Rationale**: Extreme margin. Untested. Probes upper bound.
**Validates**: Margin target saturation point.

#### Config 051 — λ=3.0, k=3.0
**Changes**: λ=3.0, k=3.0
**Rationale**: Both disc parameters elevated. Tests synergy between high lambda and high margin.
**Validates**: Whether λ and k amplify each other or show diminishing interaction.

#### Config 052 — λ=4.0, k=3.0
**Changes**: λ=4.0, k=3.0
**Rationale**: Very high disc weight with high margin. Pushes the disc optimization further.
**Validates**: Combined upper-range behavior.

#### Config 053 — λ=3.0, k=4.0
**Changes**: λ=3.0, k=4.0
**Rationale**: Reversed emphasis from 052: moderate λ, high k.
**Validates**: Whether k or λ is more impactful when both are elevated.

#### Config 054 — λ=1.0, k=1.0 (minimal disc)
**Changes**: λ=1.0, k=1.0
**Rationale**: Weak discrepancy configuration. The model should rely more on reconstruction.
**Validates**: Lower bound of disc contribution; isolates reconstruction-only performance.

#### Config 055 — alw=3
**Changes**: alw=3
**Rationale**: Higher anomaly loss weight. Phase 1 showed alw=2 boosts mask_after disc_d +22%. Tests further increase.
**Validates**: Whether alw scaling continues to help beyond 2.

#### Config 056 — alw=4
**Changes**: alw=4
**Rationale**: Aggressive anomaly emphasis.
**Validates**: alw scaling curve — diminishing returns?

#### Config 057 — alw=5
**Changes**: alw=5
**Rationale**: Very aggressive anomaly emphasis.
**Validates**: Upper bound of anomaly_loss_weight benefit.

#### Config 058 — λ=3.0, k=3.0, alw=3
**Changes**: λ=3.0, k=3.0, alw=3
**Rationale**: All discrepancy-related parameters elevated together. Maximum disc emphasis.
**Validates**: Whether triple-boosted disc parameters create a qualitatively different detection regime.

#### Config 059 — λ=0.5, k=1.0, alw=2 (disc-minimal at w500)
**Changes**: λ=0.5, k=1.0
**Rationale**: Near-minimum disc contribution. At w500, disc_SNR is already lower than w100. Tests whether minimal disc still helps.
**Validates**: Whether disc signal contributes anything meaningful at w500 with weak params.

#### Config 060 — λ=3.0, k=3.0, w=100
**Changes**: λ=3.0, k=3.0, w=100, p=20
**Rationale**: Strong disc params at w100, where disc_SNR is naturally higher. Tests whether elevated λ/k provide more benefit at small windows.
**Validates**: Window size × disc parameter interaction.

---

### Group 5: Margin Type — Dynamic vs Softplus (Configs 061–070)

Directly addresses user item (3): whether softplus margin provides meaningful differences from dynamic margin.

---

#### Config 061 — Softplus (default otherwise)
**Changes**: mt=softplus
**Rationale**: Direct comparison with Config 001 (dynamic). Everything else identical.
**Validates**: Softplus vs dynamic at the default configuration — the primary comparison.

#### Config 062 — Softplus, w=100
**Changes**: mt=softplus, w=100, p=20
**Rationale**: Softplus at small window. Tests whether margin type interacts with window size.
**Validates**: Margin type × window size interaction.

#### Config 063 — Softplus, w=200
**Changes**: mt=softplus, w=200, p=20
**Rationale**: Softplus at intermediate window.
**Validates**: Margin type consistency across window sizes.

#### Config 064 — Softplus, td=2
**Changes**: mt=softplus, td=2
**Rationale**: Softplus with shallow decoder. Margin type may interact with decoder depth (shallower decoder → less capacity to exploit margin signal).
**Validates**: Margin type × decoder depth interaction.

#### Config 065 — Softplus, td=6
**Changes**: mt=softplus, td=6
**Rationale**: Softplus with deepest decoder.
**Validates**: Whether softplus benefits more from depth than dynamic.

#### Config 066 — Softplus, d=64, ff=256
**Changes**: mt=softplus, d=64, ff=256
**Rationale**: Softplus with small model. Capacity may affect margin type utility.
**Validates**: Margin type × model capacity interaction.

#### Config 067 — Softplus, d=256, ff=1024
**Changes**: mt=softplus, d=256, ff=1024
**Rationale**: Softplus with large model.
**Validates**: Whether softplus scales differently with capacity than dynamic.

#### Config 068 — Softplus, λ=0.5
**Changes**: mt=softplus, λ=0.5
**Rationale**: Softplus with low disc weight. The margin type determines how the disc loss pushes separation; at low λ, the difference may be more or less visible.
**Validates**: Margin type × lambda_disc interaction.

#### Config 069 — Softplus, λ=4.0
**Changes**: mt=softplus, λ=4.0
**Rationale**: Softplus with high disc weight.
**Validates**: Whether softplus tolerates aggressive disc weight better than dynamic.

#### Config 070 — Softplus, enc=3, td=5
**Changes**: mt=softplus, enc=3, td=5
**Rationale**: Softplus with deep model (total 8 layers). Tests margin type with high-capacity architecture.
**Validates**: Comprehensive softplus assessment at scale.

---

### Group 6: Patch Size Interactions (Configs 071–086)

Directly addresses user item (4): how patch_size interacts with other parameters and when these interactions are significant.

---

#### Config 071 — p=5, w=500 (100 patches)
**Changes**: p=5
**Rationale**: Many patches at w500. Phase 1 showed p=5/w500 has high disc_d (2.62) but moderate roc (0.9267). Tests with d128.
**Validates**: Whether d128 handles the noise from 100 patches better than Phase 1's d64.

#### Config 072 — p=10, w=500 (50 patches)
**Changes**: p=10
**Rationale**: Medium patches at w500. Phase 1 showed p=10/w500 was worst (roc=0.8473). Tests whether new architecture rescues this.
**Validates**: Whether the p=10/w500 underperformance is architectural or fundamental.

#### Config 073 — p=25, w=500 (20 patches)
**Changes**: p=25
**Rationale**: Fewest patches at w500. 20 patches is within the optimal range (4–25).
**Validates**: Whether p=25 outperforms p=20 at w500 (fewer but coarser patches).

#### Config 074 — p=5, w=100 (20 patches)
**Changes**: w=100, p=5
**Rationale**: Many fine patches at w100. Phase 1 showed highest disc_d (3.26) but p=5/w100 had roc=0.9640 (not the best).
**Validates**: Patch granularity × window trade-off at w100.

#### Config 075 — p=10, w=100 (10 patches)
**Changes**: w=100, p=10
**Rationale**: Standard Phase 1 setup with new architecture.
**Validates**: Reference point for w100 patch size comparison.

#### Config 076 — p=25, w=100 (4 patches)
**Changes**: w=100, p=25
**Rationale**: Fewest possible patches. Phase 1 showed p=25/w100 had highest roc (0.9718) despite lowest disc_d.
**Validates**: Whether 4-patch models achieve highest roc at w100 with d128.

#### Config 077 — p=5, w=200 (40 patches)
**Changes**: w=200, p=5
**Rationale**: Many patches at w200.
**Validates**: Patch granularity effect at intermediate window.

#### Config 078 — p=10, w=200 (20 patches)
**Changes**: w=200, p=10
**Rationale**: Medium patches at w200.
**Validates**: Comparison with p=20/w200 (Config 009).

#### Config 079 — p=5, d=256, ff=1024
**Changes**: p=5, d=256, ff=1024
**Rationale**: Many patches (100) with large model. Tests whether high capacity can handle the noise from many fine patches at w500.
**Validates**: Patch_size × d_model interaction — can large models handle fine granularity? (user item 4)

#### Config 080 — p=10, d=256, ff=1024
**Changes**: p=10, d=256, ff=1024
**Rationale**: Medium patches with large model at w500.
**Validates**: Whether d256 rescues p=10/w500 (which failed at d64).

#### Config 081 — p=5, d=64, ff=256
**Changes**: p=5, d=64, ff=256
**Rationale**: Many patches with small model at w500. Expected to struggle.
**Validates**: Lower bound — small model with too many patches.

#### Config 082 — p=25, d=64, ff=256
**Changes**: p=25, d=64, ff=256
**Rationale**: Few patches with small model at w500.
**Validates**: Whether p=25 compensates for small model capacity at w500.

#### Config 083 — p=5, td=2
**Changes**: p=5, td=2
**Rationale**: Many patches with shallow decoder. Shallow decoder may struggle with 100-patch sequences.
**Validates**: Patch_size × decoder depth interaction.

#### Config 084 — p=5, td=6
**Changes**: p=5, td=6
**Rationale**: Many patches with deepest decoder. Tests whether deep decoder handles fine granularity.
**Validates**: Whether deep decoders help process many patches (user item 4).

#### Config 085 — p=25, td=2
**Changes**: p=25, td=2
**Rationale**: Few patches with shallow decoder. Simpler input may not need deep decoder.
**Validates**: Whether fewer patches allow shallower decoder (interaction with H6).

#### Config 086 — p=25, td=6
**Changes**: p=25, td=6
**Rationale**: Few patches with deepest decoder. Potentially overfitting since input is simple.
**Validates**: Overfitting risk with deep decoder on few-patch input.

---

### Group 7: disc_SNR Optimization (Configs 087–098)

Directly addresses user item (5): disc_SNR as a critical metric. Phase 1 established that high-SNR configs share mask_after + d≥128 + nh≥4 + mr≈0.10–0.15. This group tests configurations targeting maximum SNR.

---

#### Config 087 — mr=0.1, nh=4 (SNR-focused lower heads)
**Changes**: mr=0.1, nh=4
**Rationale**: Lower masking with fewer heads. Phase 1 SNR sweet spot is mr=0.08–0.15 with nh≥4. Tests nh=4 boundary.
**Validates**: Minimum nhead for high SNR at mr=0.1.

#### Config 088 — mr=0.1, nh=16 (SNR-focused many heads)
**Changes**: mr=0.1, nh=16
**Rationale**: Lower masking with maximum heads. nh=16/d128 = head_dim=8.
**Validates**: Whether more heads further improve SNR at low masking.

#### Config 089 — mr=0.1, d=256, ff=1024 (large model SNR)
**Changes**: mr=0.1, d=256, ff=1024
**Rationale**: Phase 1 showed mask_after SNR increases with model size (0.605→0.765 for d64→d192). Tests d256.
**Validates**: Whether SNR continues to scale with model size beyond d192.

#### Config 090 — mr=0.1, d=256, ff=1024, nh=16
**Changes**: mr=0.1, d=256, ff=1024, nh=16
**Rationale**: Maximum capacity + low masking. head_dim=16 at d256. Target: highest possible SNR.
**Validates**: Upper bound of disc_SNR achievable.

#### Config 091 — mr=0.1, td=5
**Changes**: mr=0.1, td=5
**Rationale**: Lower masking with deeper decoder. Phase 1 showed td×mr interact: deeper decoders help at lower mr.
**Validates**: td=5 + mr=0.1 synergy for SNR.

#### Config 092 — mr=0.1, td=6
**Changes**: mr=0.1, td=6
**Rationale**: Deepest decoder with low masking. Tests whether td=6 + mr=0.1 is optimal for SNR.
**Validates**: Maximum decoder depth + low masking for SNR.

#### Config 093 — mr=0.1, enc=3, td=5
**Changes**: mr=0.1, enc=3, td=5
**Rationale**: Deep encoder + deep decoder + low masking. Comprehensive depth + SNR configuration.
**Validates**: Whether encoder depth contributes to SNR (Phase 1 showed mask_after+el=3: disc_d=2.31).

#### Config 094 — mr=0.1, d=256, ff=1024, td=5
**Changes**: mr=0.1, d=256, ff=1024, td=5
**Rationale**: Large + deep + low masking. Near-maximum configuration for SNR.
**Validates**: Combined capacity and depth scaling for disc_SNR.

#### Config 095 — d=256, ff=1024, nh=16 (default mr=0.15)
**Changes**: d=256, ff=1024, nh=16
**Rationale**: Maximum width at default masking. Separates the mr effect from the capacity effect on SNR.
**Validates**: Whether mr=0.1 is necessary or if large capacity alone maximizes SNR.

#### Config 096 — mr=0.1, w=100 (SNR at small window)
**Changes**: mr=0.1, w=100, p=20
**Rationale**: Phase 1 showed w100 has higher mean SNR (0.451 vs 0.356 at w500). Tests mr=0.1 at w100 with new architecture.
**Validates**: Whether w100 + mr=0.1 achieves the absolute highest SNR values.

#### Config 097 — mr=0.1, w=200
**Changes**: mr=0.1, w=200, p=20
**Rationale**: SNR at intermediate window.
**Validates**: w200 SNR behavior with optimized masking.

#### Config 098 — lr=0.003, mr=0.1 (Phase 1 best SNR lr)
**Changes**: lr=0.003, mr=0.1
**Rationale**: Phase 1 showed lr=0.003 had the highest mean SNR (0.500). Tests whether lr=0.003 beats lr=0.005 specifically for SNR.
**Validates**: Optimal learning rate for disc_SNR — higher lr doesn't always mean higher SNR.

---

### Group 8: Attention Heads & Model Width (Configs 099–108)

Validates Phase 1 finding that nhead=8 is optimal and explores head count effects across model sizes. Phase 1 had limited samples for nh≥6.

---

#### Config 099 — nh=4
**Changes**: nh=4
**Rationale**: Fewer heads (head_dim=32 at d128). Phase 1 showed nh=4 had worst roc (0.9152) but best dist_d among w100 configs.
**Validates**: Whether nh=4's disturbing separation advantage appears at w500.

#### Config 100 — nh=16
**Changes**: nh=16
**Rationale**: Many heads (head_dim=8 at d128). Phase 1 showed nh=16 mean roc=0.9543 (slightly below nh=8's 0.9694).
**Validates**: Whether nh=16 underperforms nh=8 at w500 as it did in aggregate Phase 1 results.

#### Config 101 — nh=4, d=64, ff=256
**Changes**: nh=4, d=64, ff=256
**Rationale**: Fewer heads, small model. head_dim=16.
**Validates**: nh=4 at small scale; comparison with d128 (Config 099).

#### Config 102 — nh=4, d=256, ff=1024
**Changes**: nh=4, d=256, ff=1024
**Rationale**: Fewer heads, large model. head_dim=64 (very large per-head capacity).
**Validates**: Whether large head_dim with few heads is competitive.

#### Config 103 — nh=16, d=64, ff=256
**Changes**: nh=16, d=64, ff=256
**Rationale**: Many heads, small model. head_dim=4 (very small). Phase 1 showed nh=16/d64 worked surprisingly well.
**Validates**: Whether head_dim=4 is viable or if it's too small.

#### Config 104 — nh=16, d=256, ff=1024
**Changes**: nh=16, d=256, ff=1024
**Rationale**: Many heads, large model. head_dim=16.
**Validates**: nh=16 at d256; whether scaling both helps.

#### Config 105 — nh=4, d=512, ff=2048
**Changes**: nh=4, d=512, ff=2048
**Rationale**: Few heads, very large model. head_dim=128. Tests extreme per-head capacity.
**Validates**: Upper limit of head_dim scaling.

#### Config 106 — nh=16, d=512, ff=2048
**Changes**: nh=16, d=512, ff=2048
**Rationale**: Many heads, very large model. head_dim=32.
**Validates**: Comprehensive nh=16 at maximum scale.

#### Config 107 — nh=8, w=100
**Changes**: w=100, p=20
**Rationale**: Default heads at w100. Same as Config 005 (cross-reference for grouping).
**Validates**: nh=8 optimality at w100.

#### Config 108 — nh=4, w=100, p=10
**Changes**: nh=4, w=100, p=10
**Rationale**: Fewer heads at w100 with 10 patches. Phase 1 showed nh=4 had interesting dist_d behavior.
**Validates**: nh=4 characteristic at w100 (dist_d advantage).

---

### Group 9: Masking Ratio (Configs 109–117)

Validates Phase 1 findings on masking ratio sweet spot (0.08–0.15) within the allowed range {0.1, 0.15, 0.2}. Tests mr interactions with other parameters.

---

#### Config 109 — mr=0.1
**Changes**: mr=0.1
**Rationale**: Lower masking. Phase 1 showed mr=0.10 close to optimal for disc_SNR and near-best for overall roc.
**Validates**: mr=0.10 vs default mr=0.15 at w500.

#### Config 110 — mr=0.2
**Changes**: mr=0.2
**Rationale**: Higher masking. Phase 1 default was 0.20. Tests whether the mr reduction from 0.20→0.15 is justified.
**Validates**: mr=0.15 vs mr=0.20 improvement magnitude.

#### Config 111 — mr=0.1, w=100
**Changes**: mr=0.1, w=100, p=20
**Rationale**: Lower masking at small window. Phase 1 showed mr=0.08–0.15 optimal at w100.
**Validates**: mr=0.10 at w100 with new architecture.

#### Config 112 — mr=0.2, w=100
**Changes**: mr=0.2, w=100, p=20
**Rationale**: Higher masking at small window.
**Validates**: mr range effect at w100.

#### Config 113 — mr=0.1, w=200
**Changes**: mr=0.1, w=200, p=20
**Rationale**: Lower masking at intermediate window.
**Validates**: mr optimization at w200.

#### Config 114 — mr=0.2, w=200
**Changes**: mr=0.2, w=200, p=20
**Rationale**: Higher masking at w200.
**Validates**: Window × mr interaction at w200.

#### Config 115 — mr=0.1, td=3
**Changes**: mr=0.1, td=3
**Rationale**: Phase 1 showed mr=0.08×td=3 was the single best interaction. Tests mr=0.1×td=3 (closest allowed).
**Validates**: mr × td interaction — does td=3 + mr=0.1 outperform default td=4 + mr=0.15?

#### Config 116 — mr=0.2, td=3
**Changes**: mr=0.2, td=3
**Rationale**: Higher masking with moderate decoder.
**Validates**: Whether td=3 needs lower mr or works with higher mr too.

#### Config 117 — mr=0.1, td=5
**Changes**: mr=0.1, td=5
**Rationale**: Low masking with deep decoder. Same as Config 091 (cross-reference for grouping).
**Validates**: mr × deep decoder interaction.

---

### Group 10: Student Decoder Depth (Configs 118–127)

Validates Phase 1 finding that sd=2 asymmetrically benefits mask_before. Explores the full sd range {1,2,3,4} across architectures.

---

#### Config 118 — sd=2
**Changes**: sd=2
**Rationale**: Dual student decoder. Phase 1 showed sd=2 gives mask_before +0.024 roc, +0.31 disc_d, but minimal mask_after effect.
**Validates**: sd=2 benefit magnitude with new architecture at w500.

#### Config 119 — sd=3
**Changes**: sd=3
**Rationale**: Triple-layer student. Untested in Phase 1. Reduces teacher-student asymmetry (td=4, sd=3 → small gap).
**Validates**: Whether further student depth helps or if the teacher-student gap is needed.

#### Config 120 — sd=4
**Changes**: sd=4
**Rationale**: Student equals teacher (td=4, sd=4). No depth asymmetry — discrepancy comes only from information asymmetry (masking).
**Validates**: Whether depth asymmetry contributes to discrepancy, or if masking alone suffices.

#### Config 121 — sd=2, td=5
**Changes**: sd=2, td=5
**Rationale**: Wider teacher-student gap with deeper teacher.
**Validates**: Whether larger td-sd gap (5-2=3) increases disc_d.

#### Config 122 — sd=3, td=5
**Changes**: sd=3, td=5
**Rationale**: Moderate gap with deep teacher.
**Validates**: td-sd gap effect at td=5.

#### Config 123 — sd=4, td=5
**Changes**: sd=4, td=5
**Rationale**: Small gap with deep teacher (5-4=1).
**Validates**: Minimal depth asymmetry with deep teacher.

#### Config 124 — sd=2, td=6
**Changes**: sd=2, td=6
**Rationale**: Maximum gap (6-2=4) with deepest teacher.
**Validates**: Whether maximum depth asymmetry maximizes discrepancy.

#### Config 125 — sd=4, td=6
**Changes**: sd=4, td=6
**Rationale**: Deep student with deepest teacher (gap=2).
**Validates**: Whether deep student captures more while still maintaining discrepancy from deeper teacher.

#### Config 126 — sd=2, td=2
**Changes**: sd=2, td=2
**Rationale**: Equal depth student and teacher. No depth asymmetry with shallow models.
**Validates**: Whether masking-only asymmetry works at shallow depth.

#### Config 127 — sd=2, w=100
**Changes**: sd=2, w=100, p=20
**Rationale**: Dual student at w100. Tests whether sd=2's mask_before benefit appears at small window.
**Validates**: sd=2 × window size interaction.

---

### Group 11: Learning Rate (Configs 128–133)

Phase 1's most under-explored parameter. The new default lr=0.005 was validated by only n=12 experiments. This group confirms it and tests boundaries.

---

#### Config 128 — lr=0.003
**Changes**: lr=0.003
**Rationale**: Phase 1 showed lr=0.003 had highest mean SNR (0.500) and strong roc (0.9665). Tests whether lr=0.003 beats lr=0.005 for SNR.
**Validates**: Optimal lr for combined roc + SNR.

#### Config 129 — lr=0.002
**Changes**: lr=0.002
**Rationale**: Phase 1 default. Quantifies the lr improvement from 0.002→0.005.
**Validates**: lr=0.005 improvement magnitude.

#### Config 130 — lr=0.001
**Changes**: lr=0.001
**Rationale**: Low learning rate. Phase 1 showed lr=0.001 at roc=0.9525. With larger model, may need lower lr.
**Validates**: Whether the larger d128 model needs lower lr than d64 did.

#### Config 131 — lr=0.008
**Changes**: lr=0.008
**Rationale**: Higher than default. Untested territory.
**Validates**: Whether lr can be pushed beyond 0.005 for further gains.

#### Config 132 — lr=0.01
**Changes**: lr=0.01
**Rationale**: High learning rate. Tests stability boundary.
**Validates**: Upper limit of viable learning rate.

#### Config 133 — lr=0.003, w=100
**Changes**: lr=0.003, w=100, p=20
**Rationale**: Lower lr at small window. Smaller input may benefit from less aggressive optimization.
**Validates**: lr × window size interaction.

---

### Group 12: d_model Full Sweep (Configs 134–141)

Systematic exploration of model width scaling. dim_feedforward is always d_model × 4.

---

#### Config 134 — d=64, ff=256, nh=8
**Changes**: d=64, ff=256
**Rationale**: Small model. head_dim=8. Tests the minimal width at w500 with all new training params.
**Validates**: Whether d64 is viable at w500 with optimized training.

#### Config 135 — d=64, ff=256, nh=4
**Changes**: d=64, ff=256, nh=4
**Rationale**: Small model with fewer heads. head_dim=16.
**Validates**: nh=4 at d64 — different head_dim from Config 134.

#### Config 136 — d=64, ff=256, nh=16
**Changes**: d=64, ff=256, nh=16
**Rationale**: Small model with many heads. head_dim=4.
**Validates**: Whether many tiny heads work at d64.

#### Config 137 — d=256, ff=1024, nh=8
**Changes**: d=256, ff=1024
**Rationale**: Large model. head_dim=32. Same as Config 013 (cross-reference).
**Validates**: d256 scaling with default heads.

#### Config 138 — d=512, ff=2048, nh=8
**Changes**: d=512, ff=2048
**Rationale**: Very large model. head_dim=64. Same as Config 014 (cross-reference).
**Validates**: d512 scaling.

#### Config 139 — d=512, ff=2048, nh=16
**Changes**: d=512, ff=2048, nh=16
**Rationale**: Very large model with many heads. head_dim=32.
**Validates**: Whether nh=16 helps at d512.

#### Config 140 — d=512, ff=2048, nh=4
**Changes**: d=512, ff=2048, nh=4
**Rationale**: Very large model with few heads. head_dim=128.
**Validates**: Extreme head_dim at maximum model width.

#### Config 141 — d=256, ff=1024, nh=4
**Changes**: d=256, ff=1024, nh=4
**Rationale**: Large model with few heads. head_dim=64.
**Validates**: Whether d256 prefers more or fewer heads.

---

### Group 13: Combined Optimal & Hypothesis Verification (Configs 142–150)

Cross-cutting experiments that combine multiple Phase 1 insights to test synergies and verify key hypotheses.

---

#### Config 142 — Phase 2 Recipe: d=256, nh=8, td=5, mr=0.1, lr=0.005
**Changes**: d=256, ff=1024, td=5, mr=0.1
**Rationale**: Combines Phase 1's best individual parameter values: large model + deep decoder + low masking + high lr. A candidate for best overall.
**Validates**: Whether combining all optimal single-parameter choices yields the best combined performance.

#### Config 143 — Phase 2 Recipe: d=128, enc=3, td=3, mr=0.1
**Changes**: enc=3, td=3, mr=0.1
**Rationale**: Phase 1 found mr=0.08×td=3 was the best interaction, and enc=3 was the best encoder depth. Tests this combo with d128.
**Validates**: mr × td × enc triple interaction.

#### Config 144 — w500 Disturbing Specialist: d=256, p=20, nh=16
**Changes**: d=256, ff=1024, nh=16
**Rationale**: Targets maximum disturbing-normal separation at w500. Phase 1's best dist_d model was w500/d128/p20. Scales up capacity.
**Validates**: Whether d256 further improves w500's disturbing separation advantage (H3).

#### Config 145 — Disc-Recon Balance: λ=1.0, k=2.0, sd=2
**Changes**: λ=1.0, sd=2
**Rationale**: Lower disc weight with dual student. Tests a "balanced" approach: moderate disc emphasis + stronger student for reconstruction.
**Validates**: H14 — whether balancing both signals (via sd=2 for recon + moderate λ for disc) outperforms disc-heavy defaults.

#### Config 146 — Maximum Depth: enc=4, td=6, sd=4
**Changes**: enc=4, td=6, sd=4
**Rationale**: All decoders and encoder at maximum depth. Total = 14 layers. Extreme depth test.
**Validates**: Absolute depth ceiling; expected to overfit but establishes the boundary.

#### Config 147 — Minimum Config: w=100, d=64, nh=4, enc=2, td=2, sd=1
**Changes**: w=100, p=20, d=64, ff=256, nh=4, td=2
**Rationale**: Smallest viable config in Phase 2 parameter space. Establishes the lower performance bound.
**Validates**: Lower performance bound; how much the parameter optimization matters.

#### Config 148 — w100 Optimized: w=100, p=25, nh=4, td=3, mr=0.1
**Changes**: w=100, p=25, nh=4, td=3, mr=0.1
**Rationale**: w100-specific optimization combining Phase 1 insights: p=25 (best roc at w100), td=3 (best at w100), mr=0.1 (SNR sweet spot). nh=4 for dist_d.
**Validates**: Whether a w100-optimized config can rival the w500 default on overall metrics.

#### Config 149 — w200 Optimized: w=200, p=25, td=3, mr=0.1
**Changes**: w=200, p=25, td=3, mr=0.1
**Rationale**: w200-specific optimization. p=25 gives 8 patches (within optimal range). td=3 may be sweet spot for intermediate window.
**Validates**: w200 potential when specifically optimized.

#### Config 150 — High-Capacity SNR Target: d=512, nh=16, enc=3, td=5, mr=0.1
**Changes**: d=512, ff=2048, nh=16, enc=3, td=5, mr=0.1
**Rationale**: Maximum everything targeting disc_SNR. d512/nh16/enc3/td5 with low masking. The heaviest model in the study.
**Validates**: Whether brute-force capacity maximizes SNR, or if there are diminishing returns.

---

## Summary Statistics

| Group | Configs | Primary Focus | Key Phase 1 Hypotheses |
|-------|---------|---------------|----------------------|
| 1. Baseline & Reference | 001–005 | Baseline establishment | All |
| 2. Window × Capacity | 006–020 | Scaling laws | H3, H4, H7 |
| 3. Enc-Dec Depth | 021–040 | Depth allocation | H6, depth budget |
| 4. Disc Loss Optimization | 041–060 | Disc signal tuning | H13, H14, SNR |
| 5. Margin Type | 061–070 | Softplus vs dynamic | User item (3) |
| 6. Patch Size | 071–086 | Granularity effects | H16, user item (4) |
| 7. disc_SNR | 087–098 | SNR maximization | H17, I1–I4, user item (5) |
| 8. Attention Heads | 099–108 | Head count scaling | Phase 1 nh findings |
| 9. Masking Ratio | 109–117 | MR fine-tuning | H8, I3 |
| 10. Student Decoder | 118–127 | SD depth effects | Phase 1 sd=2 finding |
| 11. Learning Rate | 128–133 | LR optimization | H15 |
| 12. d_model Sweep | 134–141 | Width scaling | H7, capacity scaling |
| 13. Combined Optimal | 142–150 | Synergy testing | Cross-cutting |

### Coverage Matrix

| Parameter | Values Tested | Default |
|-----------|--------------|---------|
| window_size | 100, 200, 500 | 500 |
| patch_size | 5, 10, 20, 25 | 20 |
| masking_ratio | 0.1, 0.15, 0.2 | 0.15 |
| d_model | 64, 128, 256, 512 | 128 |
| nhead | 4, 8, 16 | 8 |
| encoder_layers | 2, 3, 4 | 2 |
| teacher_decoder | 2, 3, 4, 5, 6 | 4 |
| student_decoder | 1, 2, 3, 4 | 1 |
| learning_rate | 0.001, 0.002, 0.003, 0.005, 0.008, 0.01 | 0.005 |
| lambda_disc | 0.5, 1.0, 2.0, 3.0, 4.0, 5.0 | 2.0 |
| dynamic_margin_k | 1.0, 1.5, 2.0, 3.0, 4.0, 5.0 | 2.0 |
| anomaly_loss_weight | 2, 3, 4, 5 | 2 |
| margin_type | dynamic, softplus | dynamic |

### Key Analysis Plan

After running all 600 evaluations, the following analyses should be performed:

1. **mask_before vs mask_after**: For each config, compare the 2 mask timings at each scoring mode. Determine which configs show the largest mask_after advantage and which (if any) still favor mask_before.

2. **Scoring mode interaction**: For each mask timing, compare default vs normalized scoring. Validate H12 at scale.

3. **disc_SNR leaderboard**: Rank all 600 evaluations by disc_SNR. Identify the top-10 SNR configs and their shared characteristics.

4. **Discrepancy contribution**: Compare configs with high λ/k/alw (strong disc) vs low λ/k (weak disc) to determine when disc signal helps detection.

5. **Scaling laws**: Plot roc_auc vs d_model, window_size, total_depth for each mask timing.

6. **Interaction heatmaps**: enc×td, patch×window, d_model×window, nh×d_model interaction effects.

7. **Softplus assessment**: Paired comparison of all softplus configs (061–070) with their dynamic counterparts.

8. **Pareto frontier**: Identify configs on the Pareto frontier of roc_auc vs computational cost (parameter count × sequence length).
