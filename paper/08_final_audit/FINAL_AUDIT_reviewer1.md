---
phase: 8
agent: final-reviewer-1
directives: [R18, R1]
persona: Elsevier senior reviewer (Knowledge-Based Systems / Pattern Recognition tier), first reading
last_modified: 2026-06-11
---

# PEER REVIEW — FINAL AUDIT REPORT

**Paper title**: Label-Aware Masked Autoencoding with Gradient Reversal for Multivariate Time Series Anomaly Detection

**Submitted to**: [JOURNAL NAME — to be filled]

**Review round**: Simulated first submission (Phase 8 audit)

**Reviewer**: Final-Reviewer-1 (adversarial, KBS/PR-tier senior)

---

## 1. Summary

This paper formalizes a "contaminated semi-supervised" setting for multivariate time series anomaly detection (MTSAD), in which a small fraction of training observations carry anomaly labels alongside an unlabeled majority — a configuration the authors argue is realistic but unaddressed by existing benchmarks. They propose CSMAD (Contaminated Semi-supervised Masked Anomaly Detector), which integrates labeled anomaly information into a masked-autoencoder self-distillation architecture through three mechanisms: (i) anomaly-priority masking that preferentially masks labeled patches during training, (ii) an output-discrepancy loss restricted to normal patches, and (iii) a gradient-reversal layer (GRL) attached to the Student decoder's hidden states that adversarially erases anomaly-discriminative content from the learned representation. The architecture is asymmetric: a 3-layer Teacher decoder provides a stable reconstruction reference against which a 2-layer Student decoder fails more severely on anomalous patterns. Evaluation follows a contaminated benchmark protocol in which the chronological first half of each standard benchmark's test file is spliced into training, creating labeled-anomaly-bearing training sets. The authors evaluate against 26 baselines (22 unsupervised, 4 weakly supervised) on six dataset families under five metrics. At submission, all experimental result cells remain as [X.XX] placeholders — the manuscript structure, design rationale, and protocol are complete but experimental execution is pending.

---

## 2. Scores

| Dimension   | Score (1–5) | One-line rationale |
|-------------|-------------|-------------------|
| Novelty     | 4           | Combining masked-reconstruction self-distillation with GRL in a contaminated semi-supervised MTSAD setting is a genuine first; closest prior work (NRdetector) uses a pipeline with fixed backbone, not end-to-end gradient shaping. The contaminated benchmark protocol is also a useful contribution. Score is not 5 because gradient reversal for domain adaptation is decades old (Ganin 2016) and the time-series adaptation is technically incremental once the setting is accepted. |
| Soundness   | 3           | The problem formulation and architecture rationale are rigorous and internally consistent. Three concerns keep this from 4: (a) the GRL necessity argument has a structural gap (detailed in Weaknesses W-2); (b) the test-prefix contamination protocol introduces a train-test distributional shift whose severity is acknowledged but not controlled for; (c) the 10x/50x epoch asymmetry between baselines and CSMAD is acknowledged but not fully addressed — the epoch-budget sensitivity analysis (Appendix B.2) is itself pending. |
| Clarity     | 4           | Writing is dense but precise. Notation is consistently defined in Table C.2 and cross-referenced. The placeholder-box mechanism is unconventional but clearly labeled, so a reader can follow the logic even without results. The "why GRL is necessary beyond loss bifurcation" argument in Section 3.5 is the clearest presentation of GRL motivation I have seen in an anomaly-detection paper. Minor: Section 4.1.2 discloses epoch asymmetry transparently; some readers will expect this in a limitation paragraph, not buried in implementation details. |
| Significance| 3           | If CSMAD achieves the claimed competitive performance, the contribution is significant: it opens the contaminated semi-supervised MTSAD track, provides a reusable benchmark protocol, and introduces a principled adversarial mechanism for anomaly-feature suppression. The significance is currently conditional on results that do not exist in this submission. The setting formalization alone (contribution 1) would be a useful standalone contribution to the community. |

---

## 3. Strengths

**S-1. Well-motivated novel setting.** The contaminated semi-supervised framing is clearly distinguished from contamination-resilient and contamination-resistant detection. The footnote on p. 7 making this terminological distinction explicit is exactly the kind of precision this community needs. The motivation from industrial operational logs (fault records) is concrete and credible.

**S-2. Rigorous benchmark protocol design.** The decision to splice the chronological test prefix into training — rather than creating a synthetic label injection — is methodologically sound. The boundary-aware windowing, SWaT dual-evaluation, and SMAP/MSL split-shift documentation (Table A.5) show unusual care for reproducibility. The protocol precedent (NRdetector's 7:3 re-split) is cited appropriately.

**S-3. The GRL necessity argument is structurally complete.** Section 3.5 distinguishes operating in the loss space (loss bifurcation only) from operating in the gradient/representation space (GRL), and gives a concrete pathway argument for why loss bifurcation alone fails: the visible normal patches of an anomalous window still carry anomalous context through the shared encoder. This is a genuine theoretical contribution beyond engineering choices.

**S-4. Transparency about evaluation caveats.** The paper proactively discloses: epoch asymmetry and a dedicated sensitivity appendix; best-epoch selection on the test split (acknowledged as potentially optimistically biased); oracle PA F1 excluded from rankings; SWaT region-22 dominance effect and the excl22 counterpart. This is a notably honest presentation for a venue paper.

**S-5. Multi-metric evaluation philosophy.** Adopting PA%K-AUC F1 (primary), VUS-PR, VUS-ROC, Affiliation F1, and AUC-PR simultaneously, while explicitly justifying the exclusion of plain PA F1 from rankings, reflects current best practice from the NeurIPS 2024 benchmark critique [14] that the paper itself cites.

---

## 4. Weaknesses

### BLOCKER-grade weaknesses (reject-level if not addressed)

**W-1 [BLOCKER — Missing experimental results, all quantitative cells].** Every result cell in the paper is a [X.XX] placeholder. Tables 2, 3, A.6, A.7, A.8, B.1, B.2, B.3, B.4, and Figure 3 contain no numerical values. No claim of competitive performance, graceful degradation, or ablation gain is substantiated. For a KBS/PR submission, this is not a revision matter — it is a fundamental incompleteness that prevents acceptance. The manuscript is an unusually well-designed protocol and architecture draft, but it is not a complete paper by any standard journal criterion.

Location: Sections 4.2, 4.3, 4.4, 4.5 and all corresponding appendix tables.

**W-2 [BLOCKER — GRL necessity is asserted but not empirically closed].** The prose argument in Section 3.5 for why GRL is needed beyond loss bifurcation is logically sound, but the ablation (Table 3, Row 2) that would empirically confirm it is itself unresolved ([X.XX] cells). The claim "CSMAD is the first end-to-end MTSAD model that integrates labeled anomalies into the gradient of a masked-reconstruction self-distillation objective through gradient reversal" (p. 7) depends on GRL providing a measurable gain, which cannot currently be evaluated. If Row 2 vs Row 1 in Table 3 shows a negligible or negative margin for GRL, Contribution 3 and a significant part of Contribution 2 collapse.

Location: Section 3.5, Table 3 Row 2, Section 4.3 Row-2 paragraph.

**W-3 [BLOCKER — Test-set model selection introduces oracle bias that is acknowledged but unresolved].** Section 4.1.2 states: "Best-epoch selection for CSMAD and all 26 baselines evaluates PA%K-AUC F1 on the test split, as no separate validation split exists in this protocol. [...] absolute estimates may be optimistically biased." This bias applies to every reported number in the paper. The paper defers quantification to future work. For a submission to KBS/PR, this is not a minor limitation — it means the absolute performance figures cannot be used to draw reliable conclusions about competitive standing. The bias affects all 26 baselines equally, so relative rankings may survive, but the protocol should be repaired or the bias quantified. The epoch-budget sensitivity appendix (Appendix B.2) partially addresses epoch fairness, not the selection oracle.

Location: Section 4.1.2, "Test-set model selection" paragraph.

### Major revision-grade weaknesses

**W-4 [Major — Epoch asymmetry: 500 vs 10/50 epochs, not justified as budget-neutral].** CSMAD trains for 500 epochs with a 250-epoch warmup; most unsupervised baselines train for 10 epochs. The paper acknowledges this and commits to an epoch-budget sensitivity analysis (Appendix B.2), but that analysis is also a placeholder. The stated rationale ("convergence characteristics: CSMAD requires the 250-epoch warmup before the Student activates") is circular — it justifies the warmup by design necessity, not by evidence. A reviewer cannot assess whether the 50x epoch multiplier buys CSMAD a free-lunch margin. At KBS/PR, this requires actual numbers in B.2 before acceptance.

Location: Section 4.1.2, Appendix B.2.

**W-5 [Major — Train-test temporal distribution shift from contaminated protocol is bounded but not characterized].** The protocol splices the test-file prefix into training. The paper correctly notes that "the anomaly type distribution of the incorporated prefix may differ from that of the evaluation suffix." This is a non-trivial validity concern: if the prefix contains all easy anomaly types and the suffix contains hard ones (or vice versa), comparison conclusions are dataset-specific. The boundary-shift analysis (Table A.5) addresses SMAP/MSL split geometry but not anomaly-type distribution shift. A type-level analysis or at least an argument from the dataset structure would strengthen confidence in the protocol.

Location: Section 4.1.1 last paragraph, Table A.5.

**W-6 [Major — Contribution Bullet 4 "Extensive empirical evaluation" is premature].** Contribution 4 claims "competitive performance against [N] baselines under five evaluation metrics." This contribution is stated in the present tense as a completed empirical fact in the Introduction and Abstract, despite being entirely placeholder. The paper is systematically careful to mark placeholders elsewhere; the contribution bullets should either be stated as intended rather than achieved, or qualified with "we will demonstrate."

Location: Section 1, Contribution Bullet 4; Abstract sentence 6.

**W-7 [Major — Leave-one-out inference cost is O(N) forward passes per window but the inference cost table (B.3) is also a placeholder].** Section 3.6 acknowledges "approximately N = 50 forward passes" and Section 5 states "approximately 50x more forward-pass computation." Table B.3 (the actual measured overhead) is a placeholder. Without measured numbers, the reviewer cannot evaluate whether the practical deployment cost is acceptable. This is not cosmetic — inference cost is a primary practical concern for industrial sensor monitoring applications.

Location: Section 3.6, Section 5, Table B.3.

### Minor revision-grade weaknesses

**W-8 [Minor — The focal loss variant (Eq. C.3) is novel but uncited and underspecified].** The paper introduces a modified focal loss where the modulating factor p_t := exp(-l_i) is derived from the pos-weight-adjusted BCE rather than the raw prediction confidence. This is explicitly called out as "introduced as part of the present design rather than adopted from prior work." That is credible, but it also means the variant needs either an ablation or a theoretical justification for why this specific form is better than the standard focal loss for the GRL head's training stability. Currently there is neither.

Location: Appendix C.1, "Classification loss, exact form."

**W-9 [Minor — Contribution Bullet 3 rests on an appendix ablation that is also pending].** The asymmetric decoder design (3-layer Teacher, 2-layer Student) is described as a distinct contribution and "quantified in Appendix B.5." That quantification is in Table B.4, which is entirely [X.XX]. If the symmetric decoder ablation shows negligible difference, Contribution Bullet 3 reduces to an architectural choice rather than a validated inductive bias. This is not a blocker — the design rationale is sound regardless — but the contribution is overstated at this stage.

Location: Section 1 Contribution Bullet 3, Appendix B.5.

**W-10 [Minor — Single-seed evaluation across 113 entities].** Section 4.1.2 explicitly states "single run per entity" and notes the random-score baseline is the only multi-run average. For a method involving adversarial training (GRL), which is known to have training instability, single-seed results across 113 entities without confidence intervals is a reproducibility concern. This is acknowledged but should be elevated to a formal limitation in Section 5.

Location: Section 4.1.2, "Architecture and training" paragraph.

**W-11 [Minor — The "contaminated semi-supervised" label in Table 1 ("Train AR") could mislead].** Train AR figures (0.52%–6.20%) reflect anomalies sourced exclusively from the incorporated test prefix, not the original training file. This is stated clearly in the caption, but the footnote clarification "SMD per-machine values pending" in the main body of Table 1 appears inconsistently throughout the paper, sometimes stated and sometimes not. The inconsistency across Table 1, Table A.4, and the related protocol text is minor but should be harmonized.

Location: Table 1, Table A.4.

**W-12 [Minor — Abstract truncation / split across pages].** The abstract is split across pages 2 and 3, with the final sentence beginning on page 2 and terminating on page 3. This is a formatting artifact of the preprint build and will likely resolve in the final layout, but it makes first-reading navigation awkward.

Location: Abstract, pages 2–3.

---

## 5. Detailed Examination of Six Mandatory Items

### R1-i: Related Work / Contribution / Experiments MECE Coverage

**Related work**: The three-subsection structure (MTSAD families, Label-informed / PU / Weakly supervised, MAE + self-distillation) is well-organized and covers the direct ancestors. The NRdetector comparison is the most important and is handled precisely. A gap: the paper does not discuss anomaly detection methods that use contrastive or self-supervised learning with data augmentation (e.g., CPC-based time-series methods, COCA), which arguably address a related "limited label" challenge through a different route. This omission does not constitute a flaw given the scope, but a reviewer could ask for acknowledgment.

**Contributions vs. Experiments**: Contributions 1 (contaminated protocol) and 2 (three-path integration) map to Sections 4.1.1 and 4.2–4.3 respectively. Contribution 3 (asymmetric decoder) maps to Appendix B.5, which is a demotion that the paper handles honestly. Contribution 4 (extensive evaluation) maps to all pending result tables. The MECE check passes structurally; the issue is completeness of execution, not design.

**Experiments coverage**: The six families cover industrial control (SWaT, WaDi, PSM), IT infrastructure (SMD), and spacecraft telemetry (SMAP, MSL). This is a broad and representative selection. The ablation targets (Rows 2–4 of Table 3) are well-chosen for isolating each of the three label pathways. The protocol-effect block in Table 2 directly tests whether CSMAD's gain is due to architecture or merely the extra training data — this is the single most important experiment and its design is correct.

**Verdict**: MECE coverage is satisfactory given scope. No critical gap identified in experimental design; the gap is purely in execution.

### R1-ii: Argument Completeness — Contaminated Setting Justification, Protocol Defense, GRL Necessity

**Contaminated setting justification**: The motivation is well-founded. The three-signal observation (positions / patch loss exclusion / representational content) is the strongest single justification in the paper and directly motivates the three contributions.

**Protocol defense**: The choice of a 50/50 temporal split (rather than NRdetector's 70/30 or other ratios) is not directly justified — the paper simply states "temporal midpoint." A reviewer will ask: why 50/50 rather than 70/30 or 80/20? A sensitivity analysis over split ratios would strengthen this. This is a minor gap but could become a major concern if the split ratio significantly affects which methods appear to benefit.

**GRL necessity argument** (the most critical argumentation node): The Section 3.5 argument is logically complete. The pathway described — visible normal patches in an anomalous window carry anomalous context through the shared encoder, which the Student can exploit indirectly — is mechanistically specific and convincing. The stop-gradient on the encoder from the Student branch means the encoder is only optimized by the Teacher, which is a key architectural safeguard that is correctly described. The remaining vulnerability (Student decoder itself can memorize anomalous patterns from the visible patches it processes) is exactly what GRL closes. The argument is structurally complete. Its status as BLOCKER-level weakness is purely empirical (Table 3 Row 2 is pending), not logical.

### R1-iii: Contribution Persuasiveness (4 Bullets)

- **Bullet 1 (contaminated setting + protocol)**: Persuasive and novel. The protocol is already partially validated by the structural completeness of Table 1 and the split-shift analysis. Grade: well-supported.
- **Bullet 2 (three-path label integration)**: Persuasive in architecture design; mechanistic arguments are sound. Grade: conditionally supported pending ablation.
- **Bullet 3 (asymmetric T/S decoder)**: The design rationale in Section 3.4 ("Why the capacity gap matters") is clear. However, this is partially a retelling of the SDMAE design, adapted to multivariate time series. The SDMAE paper (Ristea et al., [32]) already uses asymmetric capacity in a masked-autoencoder context. The novelty here is the contaminated semi-supervised wrapper, not the asymmetry per se. The bullet could be read as overclaiming.
- **Bullet 4 (extensive evaluation)**: Overclaiming in present tense. Stated as achieved; execution is pending. Must be qualified.

### R1-iv: Experimental Narrative Persuasiveness (Placeholder-Conditional Assessment)

The experimental design decisions are well-chosen:
- The anomaly-excised baseline condition is a principled upper bound for unsupervised baselines: it grants them the best possible use of the label information, so CSMAD wins against this generous condition rather than a disadvantaged one. This is methodologically correct.
- The protocol-effect block in Table 2 correctly distinguishes architecture contribution from data contribution by holding architecture constant and varying only whether labeled anomalies appear in training. This is a key design choice that many papers get wrong.
- The label sparsity sweep (Figure 3) correctly sweeps region-granularity (not timestamp-granularity), matching the operational reality of fault-event logging.
- The qualitative decomposition (Figure 4) — four aligned traces — is a standard but useful diagnostic.

Weakness: the narrative in Section 4.2 is written as if results exist ("CSMAD achieves the highest PA%K-AUC F1 on [N] of the six dataset families"), when they do not. This is structurally necessary for a manuscript template, but a careful reader notices the tense inconsistency.

### R1-v: Citation Integrity Spot Check (5 citations)

The following five citations were verified against the refs.bib file:

1. **[5] wang2025nrdetector** (NRdetector, KDD 2025, pp. 1551–1562): Entry present, pages 1551–1562 per the bib comment "Crossref DOI query." DOI 10.1145/3690624.3709257 matches the KDD 2025 proceedings. The paper's characterization of NRdetector as a pipeline with fixed backbone is consistent with the NRdetector abstract description — citation appropriate and accurate. Pass.

2. **[14] liu2024elephant** (NeurIPS 2024 Datasets & Benchmarks track): Entry present. Authors Liu and Paparrizos, editors include Globersons/Mackey/Belgrave. The paper correctly cites this as criticizing evaluation practices in MTSAD. Pass.

3. **[32] ristea2024sdmae** (CVPR 2024, pp. 15984–15995): Entry present. The paper correctly attributes the asymmetric masked autoencoder self-distillation design and correctly distinguishes from SDMAE: (a) independent parallel Teacher/Student branches vs. SDMAE's branching within Teacher; (b) no GRL in SDMAE; (c) labeled anomalies present in CSMAD training. The distinction is accurate. Pass.

4. **[36] ganin2016dann** (JMLR 2016, GRL): Entry present. The sigmoid schedule (Eq. C.1) is correctly attributed and the exact schedule form matches the original Ganin et al. formulation. Pass.

5. **[42] kim2022rigorous** / **[43] paparrizos2022vus** (PA%K and VUS): Both entries present with correct DOIs (AAAI 2022 and VLDB 2022 respectively). The paper's description of PA%K-AUC F1 as integrating over K ∈ {0,...,100} and VUS as sweeping threshold and temporal tolerance is consistent with the cited papers. Pass.

Citation integrity: all five spot-checked entries are present, attributed correctly, and used in appropriate context. No integrity concern detected.

### R1-vi: Placeholder Caption and Specification Completeness

All four main-body figure placeholders (FIG-1 through FIG-4) and the appendix FIG-B1 include: (a) visible placeholder box with bold [FIG-k PLACEHOLDER] header, (b) content specification describing layout, panels, axes, and size assumption, and (c) a complete production-quality caption in the \caption{} command. The captions are complete enough that a graphic designer could produce the figure from the specification alone.

Tables with placeholders (Tables 2, 3, B.1, B.2, B.3, B.4, A.6, A.7, A.8) all have complete headers and row/column structure in place; only numeric cells are [X.XX]. The "cells [X.XX] pending experimental queue" notes inside tables are correct and visible.

All 31 numeric placeholders (NUM-001 through NUM-031) are registered in PLACEHOLDER_REGISTRY.md with expected-value sources, sync groups, and location tags. The registry is complete and cross-checked against the manuscript as of 2026-06-11.

One structural inconsistency: the introduction's contribution bullet 4 ("Extensive empirical evaluation") is written as an accomplished claim in the present tense, but the registry correctly tracks this as pending. The disconnect between the contribution-bullets style (assertive present) and the result-section style (hedged [X.XX]) is a known template convention but could cause confusion for the journal's editorial check.

---

## 6. Verdict

**Decision: Major Revision** (conditional on full experimental execution)

In practice, given that all experimental results are pending and the epoch-bias control (Appendix B.2) is also a placeholder, this submission would receive a **Reject** at KBS/PR in its current state, with a "resubmit with full results" recommendation rather than a revise-and-resubmit. I am recording Major Revision here to distinguish between the quality of the scientific design — which is unusually high — and the completeness status, which is disqualifying for initial submission.

**If I were to formally reject, the stated reasons would be:**

1. All quantitative experimental results are absent. No claims of competitive performance, graceful degradation, or ablation gain can be evaluated. A paper cannot be accepted on the basis of experimental design alone.
2. The epoch asymmetry (500 vs. 10 epochs) is acknowledged but the sensitivity analysis intended to address it is itself pending. This leaves the central comparison potentially invalid.
3. Test-set model selection (no validation split) is acknowledged as "optimistically biased" but not quantified. Combined with the epoch asymmetry, the comparison's integrity is doubly uncertain.

**If results are filled and the above three blockers are addressed, the remaining path to acceptance requires:**

- Appendix B.2 (epoch sensitivity) must contain real numbers showing that extended baseline training does not close the CSMAD margin.
- Table 3 Row 2 must confirm GRL provides a meaningful, positive gain beyond OD-exclusion alone.
- The 50/50 split ratio justification or a brief sensitivity note.
- Upgrade Contribution Bullet 4 tense to future/conditional.
- Add single-seed limitation to Section 5.

---

## 7. Compile and Static Check Summary

- PDF compiles cleanly to 46 pages. Output confirmed in main.log: "Output written on main.pdf (46 pages, 543271 bytes)."
- No LaTeX errors, undefined references, or undefined citations in the compile log.
- Two minor warnings: duplicate PDF destination identifier (page.1, pdfTeX ext4 warning — cosmetic, typical in elsarticle preprint mode) and two float specifier upgrades (`h` to `ht`). Neither is a blocker.
- All section files present: sec1_intro.tex, sec2_related.tex, sec3_method.tex, sec4_experiments.tex, sec5_conclusion.tex, appendix_A.tex, appendix_B.tex, appendix_C.tex.
- refs.bib contains 49 entries; all spot-checked citations resolved correctly.
- overleaf_package.zip present in /home/ykio/notebooks/TSMAE/paper/07_latex/. Package is ready for upload.
- PLACEHOLDER_REGISTRY.md is synchronized with the LaTeX source as of 2026-06-11 (v3-r1 scan: 31/31 NUM, 2 TXT, 4+1 FIG, 3 body TAB + 8 appendix TAB, 1 ALG — zero unmatched markers per registry §6).

---

## 8. Summary for Editor

This is a technically sophisticated paper addressing a genuine gap in the MTSAD literature. The problem setting is well-motivated, the architecture rationale is unusually rigorous, and the evaluation protocol design shows care rarely seen in this field. However, the manuscript is incomplete: all experimental results, ablations, and cost measurements are placeholders. The authors appear to be aware of and have correctly registered every outstanding item. The paper should be resubmitted after experimental completion. Upon completion, the primary remaining scientific questions are: (a) does GRL provide measurable gain beyond loss bifurcation; (b) is the performance advantage robust to epoch equalization; and (c) does test-set model selection bias materially distort the rankings.
