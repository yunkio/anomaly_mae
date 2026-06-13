---
phase: 8
agent: final-reviewer-2
persona: TSAD domain expert + adversarial top-tier reviewer
directives: [R18, R1]
target_venue: Elsevier journal (IEEE TKDE / Pattern Recognition class)
last_modified: 2026-06-11
independence_note: Written without reading FINAL_AUDIT_reviewer1.md
---

# FINAL AUDIT — Reviewer 2 (TSAD Domain Expert)

**Paper:** Label-Aware Masked Autoencoding with Gradient Reversal for Multivariate Time Series Anomaly Detection (CSMAD)

---

## Summary

This paper formalizes the *contaminated semi-supervised* MTSAD setting, introduces a benchmark protocol that injects the chronological prefix of each test stream into training, and proposes CSMAD — an asymmetric Teacher–Student masked autoencoder with three label-integration paths: anomaly-priority masking, a normal-patch-restricted output-discrepancy loss, and gradient-reversal suppression of anomaly-discriminative information from the Student's hidden states. The architecture is carefully engineered, the related-work discussion is honest about the closest precedent (NRdetector / SDMAE), and the disclosure of methodological limitations (epoch asymmetry, test-set model selection, single-seed runs) is commendably transparent. The manuscript reads at a high technical level; the notation is consistent and the appendix is structurally complete, though all numerical cells remain as [X.XX] placeholders awaiting experimental results.

My assessment focuses on whether the paper's argumentative and structural foundations would survive peer review even after numbers are filled in. Several load-bearing structural issues remain unresolved.

---

## Scores (1 = lowest, 5 = highest)

| Dimension | Score | Rationale |
|---|---|---|
| **Novelty** | 3.5 / 5 | The contaminated semi-supervised benchmark protocol is a genuine contribution. The GRL applied inside a masked autoencoder self-distillation architecture for TSAD is novel in combination. The individual ingredients (asymmetric T–S decoder from SDMAE, GRL from Ganin et al., focal-style BCE) are all adopted. The novelty bar clears "solid incremental" but does not reach "conceptual breakthrough." |
| **Soundness** | 2.5 / 5 | Three technical concerns touch the core evaluation validity: (1) test-set model selection is an acknowledged oracle bias whose magnitude is unknown and unmitigated; (2) the anomaly-ratio threshold for Affiliation F1 / threshold-dependent metrics derives from evaluation-set ground truth, which is another oracle; (3) the justification that GRL suppression closes a "contextual leakage" pathway through visible patches is asserted but not demonstrated empirically in isolation. |
| **Clarity** | 4.0 / 5 | Writing quality and structural organization are strong. The dual-lambda GRL formulation is technically precise. The pseudocode is correct and complete. Minor clarity deficiencies exist at the GRL target ambiguity and the focal-variant novelty claim. |
| **Significance** | 3.0 / 5 | The benchmark protocol contribution (contaminated re-split) is likely the most impactful element and is genuinely useful to the community. CSMAD's performance claim is currently unverifiable. The 50x inference overhead of leave-one-out masking substantially limits deployment significance. |

---

## Strengths

**S1 — Benchmark Protocol is a Real Contribution.**
The contaminated re-split protocol is principled, reproducible (fixed 50% rule with documented boundary adjustment), and applied uniformly to all 27 baselines. Acknowledging NRdetector's precedent (7:3 split) and explaining the departure is good scholarship. This protocol will be independently useful regardless of CSMAD's numerical performance.

**S2 — Multi-Metric Design is Defensible Against PA-F1 Criticism.**
Explicitly excluding oracle-threshold PA F1 from all rankings, adopting PA%K-AUC (which integrates over the tolerance spectrum), and adding VUS-PR (Liu et al. 2024 NeurIPS benchmark endorsement) and Affiliation F1 constitutes a mature response to the Schmidl et al. and Liu et al. evaluation criticisms. The five-metric suite with three orthogonal failure-mode perspectives is well argued.

**S3 — Epoch Asymmetry Disclosure is Exemplary.**
Explicitly disclosing the 500/50/10 epoch imbalance in the main text, committing to a budget-sensitivity analysis (Appendix B.2), and providing a contaminated-training no-excision condition (Appendix B.1) shows methodological rigor. Few MTSAD papers treat this asymmetry so transparently.

**S4 — Dual-Lambda GRL Formulation is Technically Sound.**
The separation of lambda_GRL (adaptive gradient-norm-ratio scaling) from lambda_rev (sigmoid-scheduled reversal coefficient) prevents the adversarial term from collapsing or dominating. The formal derivation in Appendix C.1 is correct, and the stop-gradient on the encoder input to both Student and GRL is architecturally consistent with the claim that the encoder is trained exclusively by the Teacher's objective.

**S5 — SWaT excl22 Design is Methodologically Honest.**
Masking region 22 (83.75% of test anomaly mass, a single 35,900-timestep event) and running both full and excl22 evaluations with identical scores/models prevents one pathological event from dominating all rankings. Applying the same mask to every baseline is fair. This is a non-trivial methodological contribution.

**S6 — Appendix Completeness.**
Complete hyperparameter table (Table A.1), per-entity dataset statistics (Table A.4 with documented SMAP/MSL split shifts), notation summary (Table C.2), and pseudocode (Algorithm C.1) provide enough detail for reproduction, modulo the GPU model placeholder in Appendix A.1.

---

## Weaknesses

### REJECT-LEVEL (would prevent acceptance at TKDE / Pattern Recognition tier)

---

**W1 — Test-Set Model Selection Creates Unquantified Oracle Bias in Primary Metric.**

- **Finding ID:** FB-IND-02-01
- **Severity:** CRITICAL
- **Location:** Section 4.1.2 ("Test-set model selection"), p. 15; Algorithm C.1 lines 21–22.

The paper explicitly states: "Best-epoch selection for CSMAD and all 26 baselines evaluates PA%K-AUC F1 on the test split, as no separate validation split exists in this protocol." The paper then acknowledges "absolute estimates may be optimistically biased" and declares the bias uniform across all methods. However, this argument is insufficient at the top-journal level for two reasons.

First, the bias is *not* uniform across methods because CSMAD is evaluated every 5 epochs over 500 epochs (100 evaluation checkpoints), while unsupervised baselines are evaluated every epoch over 10 epochs (10 checkpoints). CSMAD has 10x more opportunities to cherry-pick the best test-set epoch. The disclosure in Section 4.1.2 and Table A.2 acknowledges the epoch budget difference but does not acknowledge that more frequent evaluation checkpoints directly amplify oracle bias regardless of absolute epoch count. This asymmetry is structural, not merely a budget difference.

Second, PA%K-AUC F1 is also the best-epoch selection criterion. Using the primary ranking metric as the selection criterion on the same set of data it will be ranked on is circular optimization. The paper's defense — "relative rankings are unaffected" — is only valid if all methods have the same number of selection opportunities, which they do not.

*Why it matters:* TKDE reviewers routinely reject papers where the primary comparison metric doubles as a held-out-test selection criterion without a clean separation between model selection and evaluation. The magnitude of the resulting inflation is unknown and cannot be inferred post-hoc.

*Recommendation:* Either (a) use a held-out validation prefix for model selection (a temporal prefix of the training split, chronologically earlier), (b) use a fixed last-epoch checkpoint, or (c) provide empirical bounds on selection bias by reporting mean-over-checkpoints performance alongside best-checkpoint performance for representative methods.

---

**W2 — Anomaly-Ratio Threshold Uses Evaluation-Set Ground Truth (Oracle Threshold for All Threshold-Dependent Metrics).**

- **Finding ID:** FB-IND-02-02
- **Severity:** CRITICAL
- **Location:** Section 4.1.2 ("Inference and threshold"), p. 15; Appendix A.2.

The threshold is set as the (1-alpha) quantile of the score distribution, where alpha is "the measured anomaly fraction of the evaluation span." This alpha is derived from evaluation-set ground-truth labels. Any threshold-dependent metric (Affiliation F1; and indirectly, all PA%K-AUC metrics at fixed-K since each per-K threshold is re-optimized) computed at this threshold uses information that would not be available in deployment. The paper distinguishes this from AnomalyTransformer's validation-split mechanism but does not acknowledge that using evaluation ground-truth labels for the threshold — even post-hoc — constitutes a form of oracle access.

The paper does note PA%K-AUC and VUS metrics are "threshold-free" or sweep over thresholds; these are less affected. But Affiliation F1 is explicitly computed at the anomaly-ratio threshold and included in the five primary metrics. This metric is therefore inflated by oracle knowledge for all methods.

*Why it matters:* If one method's score distribution is more concentrated (lower score variance) than another's, the oracle threshold benefits the less spread method disproportionately. Furthermore, comparing against baselines in the published literature that reported results under a non-oracle threshold is not a fair comparison.

*Recommendation:* For Affiliation F1, either (a) report at the PA%K-AUC-optimal threshold (already computed), (b) report at a threshold from a separate temporal validation segment, or (c) clearly restrict the "five primary metrics" claim to the four genuinely threshold-free metrics and downgrade Affiliation F1 to a supplementary measure.

---

**W3 — GRL Suppression Claim: The "Contextual Leakage" Pathway is Asserted, Not Demonstrated.**

- **Finding ID:** FB-IND-02-03
- **Severity:** MAJOR
- **Location:** Section 3.5 ("Why gradient reversal is necessary beyond loss bifurcation"), p. 11–12.

The paper argues that excluding anomalous patches from L_OD is insufficient because "visible patches of an anomalous window still carry the surrounding anomalous context" which the Student can exploit to indirectly reconstruct anomalous patterns, thereby reducing the discrepancy. GRL is claimed to close this route by suppressing anomaly-discriminative information in the Student's hidden states.

This is a coherent theoretical argument. However, the ablation (Table 3, Row 2: w/o GRL, OD-exclusion retained) only tests the net contribution of GRL over OD-exclusion, not whether the specific contextual-leakage mechanism is actually occurring. The paper never directly demonstrates that: (a) the Student's hidden states do contain anomaly-discriminative information before GRL is applied; (b) GRL measurably reduces that information (e.g., via a probing classifier accuracy on Student hidden states w/ vs. w/o GRL); or (c) the discrepancy at anomalous patches specifically rises (not just overall performance) when GRL is active.

Without this mechanistic evidence, the "contextual leakage" narrative is a post-hoc rationalization of an observed marginal ablation gain. A sufficiently small gain in Row 2 of Table 3 (once numbers are filled) would make this argument look like motivated design rather than a principled architectural choice.

*Why it matters:* The GRL pathway is positioned as the third orthogonal contribution and the key differentiator from SDMAE. If the ablation gain is, say, 0.5 PA%K-AUC F1 points, reviewers will ask whether the complexity of GRL (dual-lambda, warmup phasing, focal BCE variant) is justified. The justification must rest on more than ablation delta.

*Recommendation:* Add a probing analysis: train a simple linear or MLP classifier on frozen Student hidden states from models trained w/ and w/o GRL, measuring how well anomaly labels can be recovered. If GRL succeeds, probing accuracy should drop. This analysis belongs in Appendix B.5 or the qualitative analysis section.

---

**W4 — Single-Seed Evaluation Across 113 Entities with No Confidence Intervals.**

- **Finding ID:** FB-IND-02-04
- **Severity:** MAJOR
- **Location:** Section 4.1.2, p. 15; Appendix A.1.

The paper states: "one seed-42 run per entity. We report no cross-seed variance or confidence intervals for the main results." This applies to all 113 entities and all 26 baselines in the primary comparison. In a journal submission (as opposed to a conference paper), single-seed results without any uncertainty quantification fail the reproducibility bar at TKDE and Pattern Recognition.

The argument that 113 entities provide statistical coverage is incomplete: macro-averaged metrics across entities can still be dominated by a handful of high-anomaly-mass entities (SWaT excl22 excluded, but other entities with high test AR, e.g., MSL 16.72% and PSM 30.63%, will dominate family averages). A single seed on those dominant entities makes the family average volatile.

*Why it matters:* TKDE explicitly requires statistical significance for comparison claims. A paper stating CSMAD achieves the highest PA%K-AUC F1 on [N] of six dataset families without any variance estimate or significance test cannot be distinguished from a favorable random seed.

*Recommendation:* Run at minimum 3 seeds on a representative subset (e.g., 2 entities per family) and report mean ± std. Apply a Wilcoxon signed-rank test across entities for the claim that CSMAD outperforms the best unsupervised baseline. The per-entity Table A.8 provides the necessary per-entity breakdown for this test even from single seeds.

---

### MAJOR (significant weaknesses, likely revision-requiring)

---

**W5 — Test-Prefix Incorporation Introduces Advantageous Training Data Not Available to Baselines Under the Anomaly-Excised Condition.**

- **Finding ID:** FB-IND-02-05
- **Severity:** MAJOR
- **Location:** Section 4.1.4 ("Comparison conditions"), p. 16–17.

Under the anomaly-excised condition, unsupervised baselines train on the original training file with labeled anomaly regions removed, while CSMAD trains on the original training file *plus* the test prefix (with anomalies present). The paper notes this reduces baseline training volume by 0.52%–6.20% and defers to the "protocol-effect block" of Table 2 to decouple these effects.

However, the protocol-effect analysis only partially decouples the effects: it compares CSMAD (standard clean-train, label paths inactive) vs. unsupervised baselines (standard clean-train), which removes the label effect. What it does not isolate is the additional training data volume contributed by the test prefix for normal (non-anomalous) windows. For PSM (6.20% train AR), the test prefix adds approximately 7% more training data in normal context. For WaDi A2 (0.76% train AR), the test prefix is 86,402/2 ≈ 43,201 additional timesteps, which is substantial relative to the original 870,972-point training set (≈5% more data, predominantly normal). The protocol-effect block does not test "CSMAD trained only on original training file, no label paths" vs. "unsupervised baselines trained on original training file," which would be the clean control.

*Why it matters:* If CSMAD's improvement under the contaminated protocol includes a data-volume component that benefits normal reconstruction (the Teacher's objective is purely reconstruction), reviewers will correctly argue that the observed gain conflates label exploitation with training data augmentation.

*Recommendation:* Add a condition in the protocol-effect block: CSMAD trained on original training file only (no test prefix), all label paths inactive. This isolates whether the asymmetric T–S decoder alone (without more data and without labels) explains part of the protocol-condition gain.

---

**W6 — Baseline Epoch Budget Asymmetry is Not Adequately Mitigated by the Sensitivity Analysis.**

- **Finding ID:** FB-IND-02-06
- **Severity:** MAJOR
- **Location:** Section 4.1.2, p. 15; Appendix B.2.

Unsupervised baselines train for 10 epochs; CSMAD trains for 500 epochs. The paper's mitigation is an epoch-budget sensitivity analysis (Table B.2) that re-trains "representative unsupervised baselines" (Anomaly Transformer, TranAD) at 50 and 100 epochs and re-trains CSMAD at a reduced budget. All budget cells in Table B.2 remain [X.XX] placeholders.

Even assuming the analysis shows baselines do not improve beyond 10 epochs (plausible for many reconstruction-based methods), the analysis covers only 2 of 22 unsupervised baselines. The 7 recent baselines (TFMAE, NPSR, TimesNet, DCdetector, MEMTO, ModernTCN, CATCH) were originally published with training budgets in the tens to hundreds of epochs in their source papers, but are run for only 10 epochs here. Table A.3 lists epochs as [X.XX] for all neural baselines — these are the actual epoch counts used and they remain unknown to the reader.

*Why it matters:* TFMAE (ICDE 2024), CATCH (ICLR 2025), and MEMTO (NeurIPS 2023) are the most relevant recent baselines. If CATCH or TFMAE are underperforming because they are trained for 10 epochs when their published results use 50–200 epochs, CSMAD's advantage may be partially an artifact of differential training budgets rather than a genuine architectural benefit.

*Recommendation:* Fill in Table A.3 epoch columns. Confirm that each recent baseline was run for at least the epoch count reported in its original paper, or explicitly acknowledge and quantify where it was not.

---

**W7 — The "Three Orthogonal Mechanisms" Claim is Overstated; Orthogonality is Not Demonstrated.**

- **Finding ID:** FB-IND-02-07
- **Severity:** MAJOR
- **Location:** Contributions list (p. 4, contribution 2); Section 3.3–3.5; Abstract.

The paper consistently describes the three mechanisms as "orthogonal." In the strict sense, orthogonality would mean the mechanisms' effects are additive and non-redundant. The ablation (Table 3) removes each mechanism independently and measures the drop. This is a standard ablation design, not an orthogonality demonstration. In particular:

- Anomaly-priority masking and L_OD are not orthogonal: L_OD operates only on the masked patches labeled normal (P_n). If anomaly-priority masking ensures all anomalous patches are in M, then P_n = M ∩ {normal patches}. Removing anomaly-priority masking changes the composition of P_n (now stochastic), which changes the gradient signal of L_OD. The two mechanisms interact through their shared dependence on M.
- Similarly, GRL receives input from the Student hidden states of all masked patches, which includes the anomaly-priority-selected patches. The GRL gradient therefore depends on which patches were masked.

The mechanisms operate at different processing stages (masking, loss, gradient), which is why the paper uses "orthogonal" in a loose architectural sense. However, the paper's framing implies functional independence, which is not true. A reviewer familiar with ablation methodology will flag the interaction confounding.

*Recommendation:* Replace "orthogonal" with "complementary" or "operating at distinct processing stages" throughout. The ablation is fine; the terminology overclaims.

---

### MODERATE (addressable without re-running experiments)

---

**W8 — The Focal-Variant for L_cls is Presented as a Novel Design Contribution But Is Inadequately Motivated.**

- **Finding ID:** FB-IND-02-08
- **Severity:** MODERATE
- **Location:** Section 3.5; Appendix C.1 (Eq. C.3).

The paper introduces a focal-style BCE variant where the modulating factor p_t := e^{-l_i} is derived from the pos-weight-adjusted BCE loss l_i rather than from the raw prediction probability. The paper states "this formulation is introduced as part of the present design rather than adopted from prior work." This is a non-trivial claim: it asserts the novelty of a loss modification. However, no ablation compares this variant against standard focal loss or standard BCE with pos-weight. The claim of superiority is implied but not tested.

*Recommendation:* Either (a) add a one-row ablation comparing standard focal BCE vs. this variant, or (b) downgrade the novelty framing to "we use a practical variant of focal loss adapted to the class-imbalance structure of MTSAD" without claiming independent novelty.

---

**W9 — The Contaminated Benchmark Protocol Has a Temporal Information Asymmetry That Is Not Fully Analyzed.**

- **Finding ID:** FB-IND-02-09
- **Severity:** MODERATE
- **Location:** Section 4.1.1, p. 13–14; Appendix A.3.

The paper acknowledges that "the anomaly type distribution of the incorporated prefix may differ from that of the evaluation suffix." This is a genuine threat to validity: if the training prefix contains predominantly one anomaly type and the evaluation suffix contains predominantly another, the model is being tested on a distribution shift that does not reflect the stated goal of "exposing labeled anomalies that are absent from the original training splits." For SWaT, WaDi, and PSM — the datasets with the most detailed published anomaly-type information — the paper does not verify whether the prefix and suffix contain comparable anomaly diversity.

*Why it matters:* If prefix anomaly types are systematically easier (e.g., point anomalies) while suffix types are harder (e.g., contextual anomalies), CSMAD's GRL training signal and the model's generalization claims are limited to "generalize across easy anomaly types," which is not the advertised claim.

*Recommendation:* For SWaT (which has published attack descriptions for all 36 events), verify whether the 50% temporal split puts at least one instance of each attack category in each half. Report this in Appendix A.3. If the split is severely imbalanced, adjust the discussion of the protocol's limitations accordingly.

---

**W10 — Score Combination Ratio c=4 is a Fixed Hyper-parameter Without Ablation in the Main Text.**

- **Finding ID:** FB-IND-02-10
- **Severity:** MODERATE
- **Location:** Eq. (5), p. 13; Appendix B.4 (parameter sensitivity).

The score combination ratio c=4 (setting discrepancy contribution to one quarter of reconstruction after adaptive scaling) is the only scoring hyperparameter. Its sensitivity is deferred to Appendix B.4 (Figure B.1, currently placeholder). Because c mediates the relative weight of the Teacher reconstruction score vs. the Teacher–Student discrepancy score, its value determines how much benefit the Student architecture contributes to detection. If the sensitivity curve shows that c → infinity (pure reconstruction) or c → 1 (equal weight) performs comparably to c=4, the architectural benefit of having a Student at all is weakened.

*Recommendation:* Ensure the parameter sensitivity figure (Figure B.1) is visible in the final manuscript and explicitly discuss the sensitivity range that maintains competitive performance. If c=4 is near-flat over [2, 8], state this directly.

---

**W11 — GRL Operates on Window-Level Labels Broadcast to All Masked Patches, Creating a Within-Window Ambiguity.**

- **Finding ID:** FB-IND-02-11
- **Severity:** MODERATE
- **Location:** Section 3.5, p. 11; Eq. C.3.

The GRL classifier head predicts whether the enclosing window contains an anomaly (y^w), and the paper notes: "strictly, the target indicates an anomaly within the masked region, which coincides with y^w under anomaly-priority masking." This is not exactly true: under anomaly-priority masking, anomalous patches are in M (masked), so the masked region is predominantly anomalous. But the normal patches drawn into M as the remaining slots (when |anomalous patches| < |M|) also receive y^w = 1 as their GRL target, even though those individual patches are not anomalous. The classifier is therefore trained with a noisy per-patch label (y^w broadcast), not a true patch-level anomaly label (y^p), and some normal patches are supervised with positive target.

This is disclosed in a parenthetical "strictly..." but is not discussed as a limitation. For datasets with low anomaly density (SMAP: 0.70% train AR, MSL: 1.70% train AR), the majority of patches in a positive window are normal, so the broadcast label noise is substantial.

*Recommendation:* Either (a) use y^p_i as the GRL target when available (patch-level label is defined in the problem formulation), or (b) acknowledge this as a limitation and discuss whether using y^p instead of y^w changes the GRL behavior in a sensitivity row of Table B.4.

---

### MINOR

---

**W12 — SMAP/MSL Contain Predominantly Command-Channel Anomalies; Treating Them as Multivariate is Potentially Misleading.**

- **Finding ID:** FB-IND-02-12
- **Severity:** MINOR
- **Location:** Appendix C.1 (Table C.1); Section 4.1.1.

SMAP (25 = 1 telemetry + 24 command channels) and MSL (55 = 1 telemetry + 54 command channels) have anomaly labels primarily on the single telemetry channel, as is well documented in the original Hundman et al. dataset paper. Including them as "multivariate" datasets without noting that the anomaly structure is essentially univariate conflates different problem complexities. The paper should note this in the dataset discussion and discuss whether the multivariate correlation-learning argument central to CSMAD's motivation applies to these datasets.

---

**W13 — "GPU model" Placeholder in Appendix A.1 Affects Reproducibility.**

- **Finding ID:** FB-IND-02-13
- **Severity:** MINOR
- **Location:** Appendix A.1, p. 32; Algorithm C.1.

"All experiments run on [GPU model]" is a placeholder that must be filled. Wall-clock results in Table B.3 report seconds per entity, and without the GPU model these numbers are uninterpretable. This is a standard reproducibility requirement.

---

**W14 — The Claim About "First Architecture Combining Masked-Reconstruction Self-Distillation with Gradient Reversal" May Be Falsifiable by Domain Adaptation Literature.**

- **Finding ID:** FB-IND-02-14
- **Severity:** MINOR
- **Location:** Section 1, p. 4 (introduction paragraph); Section 2.3 footnote.

The paper claims CSMAD is "to our knowledge, the first architecture combining masked-reconstruction self-distillation with gradient reversal." This narrowly constructed claim is likely correct but could be challenged by domain adaptation work using masked autoencoders with adversarial alignment (e.g., masked feature alignment with GRL for domain shift, which appears in NLP and computer vision adaptation literature circa 2022–2024). The qualifier "to our knowledge" partially hedges this, but a targeted search for "masked autoencoder gradient reversal domain adaptation" would be prudent before submission.

---

**W15 — Citation [45] (Xu et al. 2018 KPIVAE) Is Used Only to Define Oracle PA F1 Threshold Selection, But the Current Manuscript Excludes That Oracle From Rankings.**

- **Finding ID:** FB-IND-02-15
- **Severity:** MINOR
- **Location:** Section 4.1.3, p. 16; Reference list [45].

The KPIVAE paper (Xu et al. WWW 2018) is cited as the source of conventional PA (K=0) F1 and its oracle-threshold variant. However, the present paper explicitly excludes oracle-threshold PA F1 from all rankings. The citation is used appropriately for historical attribution, but a reviewer may note that comparing against methods that published results using the oracle PA F1 (e.g., Anomaly Transformer, TranAD) and then excluding that metric from rankings creates an incompatibility: readers cannot assess CSMAD against the published numbers of those prior works. The paper's appendix (Appendix A.5) promises oracle PA F1 for comparability, but Table A.7 is currently fully [X.XX] placeholders with a note "cells [X.XX] pending queue."

---

## Argument Completeness Audit (MECE Assessment)

The paper claims four contributions. Assessing whether each is self-contained and mutually non-redundant:

1. **Contaminated semi-supervised benchmark protocol** — self-contained, well-formalized, and independently useful. This contribution would hold even if CSMAD performed at par with baselines. MECE: PASS.

2. **Three-path label integration** — the three paths are described as orthogonal but interact through shared dependence on M (see W7). The contribution is real but the "orthogonal" framing is imprecise. MECE: PARTIAL FAIL (terminology issue, not a structural gap).

3. **Asymmetric T–S decoder architecture** — the design (3L Teacher, 2L Student, independent parallel branches from shared encoder) is a direct adaptation of SDMAE from video to MTSAD with the addition of the GRL branch. The claim that this is architecturally novel relative to SDMAE rests on three differences stated in footnote 1 (independent branches vs. branching within Teacher, GRL absent from SDMAE, labeled anomaly use). These distinctions are clear and the contribution is adequately scoped. MECE: PASS.

4. **Extensive empirical evaluation** — currently entirely [X.XX]. Not assessable as a complete contribution at this stage. By design this is a placeholder state; the structure of the evaluation is sound. MECE: CONDITIONALLY PASS (pending numbers).

**Missing contribution: the adaptive scaling of the discrepancy component (Eq. 4–5) and the dual-lambda GRL formulation are engineering contributions not listed as named contributions.** This is fine for a journal paper; they are correctly embedded in the methodology description rather than over-claimed as contributions.

---

## Contribution Persuasiveness

The paper's strongest argument is the benchmark protocol: it cleanly identifies a gap (existing benchmarks have no labeled anomalies in training), proposes a principled fix (chronological prefix injection), and evaluates all 27 methods under the same protocol. This is independently publishable as a benchmark/protocol contribution.

CSMAD's algorithmic novelty is solid incremental: taking SDMAE's video anomaly detector and augmenting it with three label-integration mechanisms for a new domain and setting. The GRL addition is the most novel element and the GRL-necessity argument is intellectually honest (Section 3.5 argumentation is well-structured). The contribution would be more persuasive with mechanistic evidence (W3) and would survive numerical review if the margin over the best unsupervised competitor is consistent across dataset families (not driven by one outlier family).

---

## Experimental Design Persuasiveness

**Strengths of the design:**
- Uniform evaluation pipeline (shared loading layer, shared metric computation) eliminates implementation-dependent discrepancies.
- 26 baselines across four tiers (simple, legacy deep, recent deep, weakly supervised) is comprehensive.
- Five metrics with complementary failure modes is well-designed.
- Anomaly-excised condition for unsupervised baselines correctly gives them the best available use of labels under their paradigm.

**Weaknesses of the design:**
- Test-set model selection oracle (W1) — unmitigated.
- Anomaly-ratio threshold oracle for Affiliation F1 (W2) — unmitigated.
- Single seed without confidence intervals (W4) — unmitigated.
- Epoch asymmetry for recent baselines not fully addressed (W6).
- Protocol-effect decoupling is incomplete (W5).

The experimental design is more carefully constructed than the median MTSAD paper but has three structural oracle issues that will attract critical reviewer attention at any top venue.

---

## Citation Quality Spot-Check (5 citations)

1. **[14] Liu & Paparrizos, NeurIPS 2024 ("Elephant in the room")** — correctly cited as endorsing VUS-PR and criticizing MTSAD benchmark practices. The reference is used appropriately and the citation data (NeurIPS 2024 Datasets & Benchmarks) is correct.

2. **[42] Kim et al., AAAI 2022 (PA%K)** — correctly cited as the source of the PA%K parameterization. The paper correctly notes K=0 recovers conventional PA, which matches the original paper's formulation.

3. **[32] Ristea et al., CVPR 2024 (SDMAE)** — correctly cited and the architectural distinction from CSMAD (independent parallel branches vs. branching within Teacher) is accurately described. The CVPR 2024 citation data is correct.

4. **[36] Ganin et al., JMLR 2016 (GRL)** — correctly cited for both the GRL mechanism and the sigmoid reversal schedule. The schedule formula in Eq. C.1 matches the published formula in Ganin et al. (2016).

5. **[5] Wang et al., KDD 2025 (NRdetector)** — pages listed as 1551–1562 and DOI provided. The paper is the most important comparison point and is consistently referenced. Bib note confirms DOI was verified via Crossref. Accurate.

Overall citation quality: HIGH. No citation errors identified in the spot check.

---

## Placeholder Completeness Assessment

All numerical cells in Tables 2, 3, A.6, A.7, A.8, B.1, B.2, B.3, B.4 are [X.XX] placeholders. All four figures (Fig. 1–4, Fig. B.1) are placeholder boxes with detailed layout descriptions. Several inline numerical claims in Sections 4.2–4.4 use [X.XX] with PH:NUM-* tags (28 tagged placeholders identified in sec4_experiments.tex; 36 in appendix_A.tex; 20 in appendix_B.tex).

**Structural completeness:** The placeholder discipline is unusually clean. Every [X.XX] has a corresponding PH:NUM tag and the surrounding prose is written to accept any number, using hedges like "[gradually / monotonically]" (PH:NUM-027) to flag cases where the direction needs confirmation. No placeholder void creates a structural argument gap that would fail even after filling — except for:

- Table A.3: LR, Batch, Epochs columns for all neural baselines are [X.XX]. Until these are filled, the epoch-asymmetry analysis (W6) cannot be fully evaluated by reviewers.
- Table B.2: All cells [X.XX] — the epoch-budget sensitivity analysis is entirely pending, making the disclosure of epoch asymmetry currently unmitigated.
- Algorithm C.1 is titled "(pseudocode placeholder)" — this is unusual language and should be clarified as either a placeholder name or a confirmed algorithm rendering.

---

## Verdict

**Recommendation: Major Revision (equivalent to "Weak Reject" at a conference venue; revise-and-resubmit at journal tier)**

This is a carefully constructed paper with a genuine benchmark contribution and an interesting algorithmic design. The related-work discussion is honest, the notation is clean, and the methodological disclosures are more transparent than is typical for the field. However, three structural issues that will be raised by any expert reviewer remain unmitigated: (1) test-set model selection oracle with asymmetric checkpoint counts, (2) evaluation-set ground-truth anomaly ratio used as threshold for Affiliation F1, and (3) absence of mechanistic evidence for the GRL contextual-leakage suppression claim. These are not placeholder issues — they are design issues that require either new experiments or a significant re-framing of claims. Additionally, single-seed evaluation without statistical tests fails the reproducibility standard for a journal submission. The paper is closer to acceptance than rejection, but these issues would lead to a mandatory revision request at TKDE, Pattern Recognition, or equivalent.

---

## Reject-Reason Summary (for tracking)

**Reject-level findings: 4**
1. W1: Test-set model selection with asymmetric checkpoint counts creates unquantified, non-uniform oracle bias in the primary ranking metric.
2. W2: Anomaly-ratio threshold derived from evaluation-set ground truth creates oracle access for all threshold-dependent metrics (Affiliation F1).
3. W4: Single-seed evaluation across 113 entities without confidence intervals or significance testing fails journal reproducibility standards.
4. W3: GRL contextual-leakage suppression claim is asserted without mechanistic demonstration; ablation alone does not establish the stated causal mechanism.

**Core weakness in 3 lines:**
The primary comparison metric (PA%K-AUC F1) is also the model-selection criterion evaluated on the test set, with CSMAD having 10x more selection opportunities than unsupervised baselines — this is unmitigated oracle bias. The anomaly-ratio threshold is derived from evaluation ground truth, making Affiliation F1 an oracle-threshold metric. Single-seed results without statistical tests cannot support the claim of consistent outperformance across six dataset families at the journal level.
