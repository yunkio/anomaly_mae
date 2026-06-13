---
phase: 5
agent: adversarial-reviewer
directives: [R1, R8, R9, R10, T5]
last_modified: 2026-06-11
artifact_reviewed: MANUSCRIPT_v2_draft.md (post-surgery v2-draft-r2)
reference_documents:
  - PAPER_BLUEPRINT.md r3
  - PAGE_BUDGET.md r3
verdict: CONDITIONAL PASS — 7 BLOCKERs and 8 MAJORs must be resolved before phase completion
---

# Phase 5 Adversarial Review — Argument Quality & Blueprint Conformance

---

## 1. Executive Verdict

The manuscript structure, surgical compression, and notation are substantially sound.
However, seven BLOCKER-class issues exist that individually constitute reject-level exposure:
two argument chain breaks (§3.1 three-structure missing, §3.5 GRL necessity logic flawed),
one blueprint-deviance (§14 Defense Argument 5 missing), one R9 pattern violation (§2.3
residual difference-listing), one R8 novelty scoping failure (contribution bullet 4 overclaim),
one MECE boundary violation (§2.2 end-to-end distinctiveness absent), and one unresolved
placeholder that renders the central performance claim unwritable (NUM-006/007/008/009 fill
depends on completed experiments that BLUEPRINT §0.4 flags as not yet run).

---

## 2. Blueprint Implementation Audit (§15 Reject Scenarios)

### 2.1 "test-prefix is test-label leakage" (§14 Five-Argument Defense)

**BLOCKER-BP-01**
- Severity: BLOCKER
- Location: §4.1.1 "Contaminated benchmark protocol"
- Issue: Blueprint §14 mandates five distinct defense arguments in §4.1.1: (①) re-split
  definition, (②) structural absence of labels in original train, (③) fairness / all-methods
  identical data, (④) temporal ordering + unified rule, (⑤) NRdetector re-split precedent.
  The manuscript deploys ①②③④ adequately but Argument ⑤ (NRdetector precedent) is entirely
  absent from §4.1.1 body text. The sentence "with precedent in the 7:3 re-split of
  [wang2025nrdetector] placing anomalies within the training stream" appears in §4.1.1 but
  is a single clause embedded in the middle of another sentence — it does not constitute
  the standalone defense sentence the blueprint requires, and crucially it contains a
  hidden-assumption problem: the blueprint itself flags "NRdetector 7:3 split의 시간 순서
  보존 여부는 원문 미명시 — 단정 인용 금지, '재분할 선례' 수준으로만 인용." The manuscript
  writes "placing anomalies within the training stream" as if this is a confirmed structural
  fact about NRdetector's split, which the blueprint prohibits.
- Evidence: Blueprint §14 Argument ⑤; manuscript §4.1.1 last sentence of protocol paragraph.
- Fix: (a) Elevate Argument ⑤ to its own sentence in §4.1.1. (b) Rewrite to "re-split
  precedent" level only, without asserting temporal ordering or structural properties of
  NRdetector's split: "This practice has precedent: Wang et al. [wang2025nrdetector] also
  re-split standard benchmarks so that anomalous events fall within the training stream."

---

### 2.2 "SDMAE is too similar" (§15 second scenario) — §3.5 Body Sentence

**BLOCKER-BP-02**
- Severity: BLOCKER
- Location: §3.5 "GRL anomaly suppression loss"
- Issue: Blueprint §14 / §15 / decision ⑤ require the SDMAE operating-layer distinction
  ("타깃/손실 공간 vs gradient 공간") to appear in §3.5 body text as a standalone sentence
  (RT MAJOR-01 mandate). The manuscript §3.5 opens with: "Whereas SDMAE's anomaly-overlook
  supervision operates in the target/loss space, our GRL operates in the gradient space of
  the Student's internal representation." This IS present and correctly placed.
  HOWEVER, the blueprint also mandates a footnote in §2.3 carrying the structural distinction
  (branch-off vs independent decoder) plus terminology lineage. The manuscript §2.3 footnote
  [^sd-fn] contains: "The gradient reversal layer that adversarially suppresses anomaly
  information in the Student is absent from the unsupervised video setting of [ristea2024sdmae];
  the distinction between operating in the target/loss space versus the gradient space of the
  representation is elaborated in Section 3.5." This footnote splits the defense: the
  structure-difference (branch-off vs independent parallel decoders) is NOT in the footnote
  — it appears only in §2.3 body prose. Blueprint §5.5 decision ⑤ requires: "unlike SDMAE,
  whose student decoder branches off from the teacher decoder, our teacher and student decoders
  are independent" to be in the Method-section footnote. The current footnote lacks this.
- Evidence: Blueprint §11 decision ⑤ exact wording; manuscript §2.3 footnote [^sd-fn].
- Fix: Add to footnote [^sd-fn]: "Unlike SDMAE, whose student decoder branches off from
  within the teacher decoder after the first transformer block, our Teacher and Student
  decoders are independent parallel branches off the shared encoder."

---

### 2.3 "main 실험이 semi-supervised가 아니다" (§15 third scenario) — §3.1 Three-Structure

**BLOCKER-BP-03**
- Severity: BLOCKER
- Location: §3.1 "Problem Formulation and Setting"
- Issue: Blueprint §5.2 (RT MAJOR-09) mandates the three-structure disclosure in §3.1:
  (②-1) setting assumption = mostly unlabeled + small labeled fraction; (②-2) main
  experiments = upper-bound case where all training anomalies are labeled; (②-3) label
  sparsity sweep = validation of the general case. Without this, a reviewer can correctly
  assert "your main experiment is fully supervised on the training split, not semi-supervised."
  The manuscript §3.1 writes: "The main experiments take the label-availability upper bound —
  every anomalous timestep in the training split is labeled — and Section 4.4 validates the
  general case of partial labeling." This is ONE sentence. It is present but does not
  constitute the 1–2 sentence structured disclosure the blueprint mandates: ②-1 (the
  general setting definition with mostly unlabeled) is stated in the paragraph above, but
  the explicit bridge sentence "the main experiment is the upper-bound case of this general
  setting" is too compressed. More critically, the three-part structure is never explicitly
  assembled so a reviewer sees the logical chain ②-1 → ②-2 → ②-3.
- Evidence: Blueprint §5.2 "(RT MAJOR-09) 3단 구조(②-1/②-2/②-3) 서술 필수"; manuscript §3.1.
- Fix: Expand to an explicit 2-sentence disclosure: "Our formulation assumes the general
  case where some anomaly regions remain unlabeled; in our main experiments we evaluate the
  label-availability upper bound — every anomalous timestep in the training split is labeled
  — which maximizes the signal available to the three label-guided pathways. Section 4.4
  validates the general case by sweeping the labeled fraction downward toward the fully
  unsupervised limit."

---

### 2.4 Protocol-Effect Table Absorbed — RT-B03 Separation Argument (§4.2)

**MAJOR-BP-04**
- Severity: MAJOR
- Location: §4.2 "Protocol-effect analysis"
- Issue: Blueprint decision ④ / RT BLOCKER-03 required the protocol-effect analysis
  (standard split vs contaminated) to be a visible separate component in main text.
  The surgery (D-010 ①) absorbed Table 4 into Table 2 as a "bottom protocol-effect row-group."
  The narrative is retained in §4.2. However, the two-step argument structure required by
  the blueprint is now harder to parse: (i) CSMAD competitive under clean-train split
  (method value independent of protocol) + (ii) CSMAD gains further under contaminated
  protocol (label exploitation value) must each be clearly identifiable as distinct claims
  with distinct numeric support. As written, these claims all point to the same merged
  TAB-2 placeholder with no way for a reader to verify they are looking at separate
  conditions. If the merged table caption in PLACEHOLDER_REGISTRY.md does not explicitly
  label the row-group as "standard split" vs "contaminated protocol" with clearly separated
  metrics, the reviewer attack "you never showed performance without the label advantage"
  lands undeflected.
- Evidence: Blueprint §6.6 "2단 논증 구조"; manuscript §4.2 "Protocol-effect analysis."
- Fix: Verify that the TAB-2 caption spec in PLACEHOLDER_REGISTRY.md calls out the
  row-group split conditions explicitly. Add a sentence in §4.2 prose that names both
  conditions by label BEFORE citing the placeholder numbers: "Under (i) the standard clean
  split, CSMAD achieves [NUM-015] PA%K-AUC F1, matching the best unsupervised competitor
  ([NUM-016]); under (ii) the contaminated protocol it improves to [NUM-017] ([NUM-018]
  gain), while the unsupervised baselines show [NUM-019] change, confirming the gain is
  attributable to label exploitation rather than training volume."
  Current text does this but buries the condition labels in mid-sentence subordinate
  clauses — restructure so each condition gets its own sentence.

---

### 2.5 "성능 우위가 프로토콜 때문" — Table 4 Execution Status (§15 last scenario)

**BLOCKER-BP-05**
- Severity: BLOCKER
- Location: §4.2 / EXPERIMENT EXECUTION STATUS
- Issue: Blueprint §6.6 EXPERIMENT_EXECUTION_TODO flags "standard-split 조건의 제안 방법·
  baseline 실험 미실행." The manuscript carries fully populated placeholder numbers
  NUM-015 through NUM-019 with [X.XX] markers, acknowledging the gap. This is properly
  flagged. However: as a gate issue for phase completion, the central argument that decouples
  method value from protocol effect CANNOT be written without these numbers. A manuscript
  submitted with [X.XX] in the protocol-effect decoupling argument fails the most likely
  reviewer attack. This is a BLOCKER on the argument chain, not just a number-fill task.
- Evidence: Blueprint §0.4 EXPERIMENT_EXECUTION_TODO; §6.6 ⚠️ flag; manuscript §4.2.
- Fix: Experiments NUM-015 through NUM-019 (standard split condition, representative
  2-3 datasets) must be executed before manuscript finalization. This is the minimum
  experimental obligation before phase completion on the argument-quality dimension.

---

## 3. Contribution Prominence — R8 Novelty Scoping (D-008)

### 3.1 Contribution Bullet 4 — Overclaim "state-of-the-art"

**BLOCKER-R8-01**
- Severity: BLOCKER
- Location: §1 Introduction, contribution bullet 4
- Issue: Blueprint §11 decision ① bullet 4 prescribes: "Extensive experiments on six
  multivariate datasets demonstrate state-of-the-art performance under five rigorous metrics."
  The manuscript implements: "demonstrate competitive performance against [N] baselines
  under five evaluation metrics." This is correctly hedged. However, bullet 4 also says
  "The model maintains robust detection toward the fully unsupervised limit." This claim
  is supported by §4.4 label sparsity analysis, but the label sparsity figure (FIG-3) is a
  placeholder and the qualitative descriptor "gradually / monotonically" in §4.4 is itself
  a placeholder [PH:NUM-027]. A contribution bullet that promises "robust detection" whose
  only support is an unresolved placeholder is an overclaim until NUM-027 is filled. If
  results show non-monotonic degradation or sharp drops at low p, this bullet must be revised.
  Additionally, §5 (Conclusion) writes "the label sparsity analysis confirms graceful
  degradation" — "confirms" is past tense asserting a result that has not been verified.
- Evidence: §1 bullet 4; §4.4 PH:NUM-027; §5 "confirms graceful degradation."
- Fix: (a) In §5, change "confirms graceful degradation" to "is designed to show graceful
  degradation; once results are confirmed, this claim will be finalized." (b) Mark bullet 4
  as contingent on FIG-3 results: qualifier text must be filled from actual results.
  (c) §4.4 "gradually / monotonically" placeholder must be resolved before contribution
  bullet 4 can stand.

---

### 3.2 D-008 Scoping — "first end-to-end" Claim Boundary

**MAJOR-R8-02**
- Severity: MAJOR
- Location: §1 Para 3 (last sentence) and §2.2 (last paragraph)
- Issue: Blueprint D-008 mandates the novelty claim be scoped to "masked-reconstruction
  self-distillation 표현 학습에 GRL 기반 adversarial 통합" and requires citation of
  Xue & Yan (IJCNN 2022, arXiv 2207.00705) and SLA-VAE (WWW 2022) in §2.2 with
  differentiation. The manuscript §2.2 cites [xue2022fewpositive] and [huang2022slavae]
  in one sentence: "Two earlier semi-supervised variational models addressed label scarcity
  in multivariate time series [xue2022fewpositive, huang2022slavae], but their representation
  learning remains largely label-agnostic: labels enter through auxiliary loss terms rather
  than shaping the gradient of the latent space." This is present and correctly scoped.
  HOWEVER: §1 Para 3 claims "To our knowledge, CSMAD is the first end-to-end model for
  multivariate TSAD that integrates labeled anomalies into the gradient of a self-supervised
  representation learning objective." This sentence does NOT acknowledge Xue/SLA-VAE inline
  or in a footnote at the point of the claim. A reviewer who knows these papers will object
  immediately. The §2.2 differentiation must be forward-referenced at the §1 claim site.
- Evidence: Blueprint D-008; manuscript §1 Para 3 "to our knowledge" sentence; §2.2.
- Fix: Add after the "to our knowledge" claim: "(Prior semi-supervised variational models
  for TSAD [xue2022fewpositive, huang2022slavae] integrate labels through auxiliary terms
  on the output loss, not through the gradient of the latent representation itself — see
  Section 2.2.)" This converts a naked claim into a scoped claim with forward-referenced
  evidence, matching D-008.

---

### 3.3 Abstract — "competitive" vs Performance Claim Inconsistency

**MINOR-R8-03**
- Severity: MINOR
- Location: Abstract line 6; §5 Conclusion
- Issue: The abstract writes "achieves competitive performance against [N] unsupervised and
  weakly supervised baselines." "Competitive" is appropriate hedging given incomplete
  experiments (§0.4). However, the §5 conclusion writes "show competitive performance
  against [N] baselines" — identical conservative language — but then immediately claims
  "the label sparsity analysis confirms graceful degradation." The register inconsistency
  (hedged main result + assertive secondary result) is jarring and will attract reviewer
  attention: why hedge the main claim but assert the secondary? Both should be consistent
  in their dependency on placeholder resolution.
- Fix: Match both to the same hedge: "competitive performance ... and label sparsity
  analysis shows [qualitative descriptor pending] degradation toward the unsupervised limit."

---

## 4. MECE Audit — R1

### 4.1 §2.2 Missing End-to-End Distinctiveness from Weakly Supervised Methods

**BLOCKER-R1-01**
- Severity: BLOCKER
- Location: §2.2 "Label-Informed Anomaly Detection"
- Issue: Blueprint §4.3 (RT MAJOR-02) requires an explicit 1–2 sentence argument in §2.2
  distinguishing CSMAD's gradient-space label integration from weakly supervised methods
  (DeepMIL/WETAS/TreeMIL) whose labels serve as classification/ranking objectives rather
  than self-supervised representation learning gradients. The blueprint specifies: "weakly
  supervised approaches ... optimize models to classify segments accurately by leveraging
  segment-level labels" — self-supervised reconstruction pretext is absent. The manuscript
  §2.2 covers PU learning, deviation networks, SLA-VAE, and NRdetector, but contains NO
  sentence on DeepMIL/WETAS/TreeMIL at all in the body of §2.2. They appear only in
  §4.1.4 as a baseline tier. This violates the blueprint's confirmed MECE placement:
  "WETAS/DeepMIL/TreeMIL은 §2.2 전속 — §2.1 비지도 클러스터 포함 절대 불가 (RT MINOR-03)."
  The §2.2 body must contain them as a named class with the end-to-end distinction argument.
- Evidence: Blueprint §4.3 RT MAJOR-02; manuscript §2.2 (no mention of DeepMIL/WETAS/TreeMIL).
- Fix: Add one paragraph to §2.2 before the NRdetector paragraph: "A parallel weakly
  supervised strand trains single models to classify or rank anomaly windows from segment-level
  annotations [sultani2018deepmil, lee2021wetas, liu2024treemil]; the label is the primary
  learning signal optimizing a classification or ranking objective, with no self-supervised
  reconstruction pretext. Our approach differs: labels enter the gradient of a masked
  reconstruction objective, shaping what the encoder learns to represent rather than what
  a classifier predicts."

---

### 4.2 §2.1 TFMAE Citation in §4.1.4 Baseline Prose Bleeds into §2.1 Territory

**MINOR-R1-02**
- Severity: MINOR
- Location: §4.1.4 baseline prose; §2.1 scope
- Issue: Blueprint ADV MINOR-001 mandates TFMAE's sole citation location is §2.3. In
  §4.1.4 the manuscript correctly cites it as part of the "SOTA New 7" cluster with
  [fang2024tfmae]. This is a §4.1.4 citation, not §2.1 — the placement is technically
  correct. However the baseline introduction text "seven recent competitive methods [fang2024tfmae,
  ...]" positions TFMAE as a peer in the unsupervised TSAD competition family, which
  may cause a reviewer to wonder why it was not discussed in §2.1 with the other recent
  methods. Since §2.1 does not mention TFMAE (correctly omitted per blueprint), §4.1.4
  should include a one-clause reminder: "TFMAE, a time-series MAE variant discussed in
  Section 2.3, ..." to signal intentional placement rather than omission.
- Fix: In §4.1.4, change "seven recent competitive methods [fang2024tfmae, ..." to
  "seven recent methods (including TFMAE, a time-series MAE variant; see Section 2.3)
  [fang2024tfmae, ..."

---

### 4.3 §4.3 Ablation Row 7 (Symmetric Decoder) Still in Body Despite "미실행"

**MAJOR-R1-03**
- Severity: MAJOR
- Location: §4.3 "Extended variants" (Appendix §B.5 reference) and §B.5 body
- Issue: Blueprint §6.7 and §0.4 state that ablation row 7 (symmetric decoder) is
  load-bearing for contribution bullet 3, is listed as "미실행", and carries the
  conditional rule "본문 잔류 금지." The manuscript body (§4.3) correctly demotes
  it to Appendix §B.5. However, §B.5 contains: "A symmetric decoder (Teacher 2L /
  Student 2L) removes the capacity gap behind the Student's preferential failure on
  anomalous patterns (Section 3.4); the change of [X.XX] points quantifies the
  asymmetric design — the architectural prior of contribution 3 — as an empirical effect."
  This text is present in the appendix with a [X.XX] placeholder. If the experiment is
  unrun, the appendix cannot credibly assert this claim — reviewers DO read appendices.
  More critically, contribution bullet 3 in §1 currently asserts "making the Teacher–Student
  output discrepancy a reliable anomaly signal under contaminated training" at full claim
  strength. Blueprint §6.7 states: "행 7 미완 시 ... bullet 3의 'a reliable anomaly signal'
  주장을 정성 수준('intended to provide')으로 하향." This downgrade has NOT been applied.
- Evidence: Blueprint §6.7; §0.4 "행 7은 contribution bullet 3 load-bearing"; manuscript §1
  bullet 3 ("a reliable anomaly signal"); §B.5 [X.XX] placeholder.
- Fix: (a) Until symmetric-decoder ablation (NUM-024) is run, change §1 bullet 3 to:
  "...designed to make the Teacher–Student output discrepancy a reliable anomaly signal
  under contaminated training (quantification in Appendix B.5 pending experiment completion)."
  (b) In §B.5, add a conditional disclosure: "This ablation is pending; upon completion,
  the quantified gap will replace [X.XX] here and the claim strength in Section 1 will
  be finalized."

---

## 5. R9 Compliance — SDMAE Frequency and Tone

### 5.1 §2.3 Residual Difference-Listing Pattern

**BLOCKER-R9-01**
- Severity: BLOCKER
- Location: §2.3 body text, last paragraph
- Issue: R9 prohibits the "차이 나열(difference-listing) 패턴" — enumerating multiple
  differences from SDMAE as a list. The manuscript §2.3 last paragraph contains:
  "Our Teacher and Student decoders are independent parallel branches off a shared
  encoder — rather than a branch-off from within the teacher decoder — and the Student
  is additionally trained to suppress anomaly-specific information through a gradient
  reversal mechanism operating in representation space rather than in the output or loss
  space." This sentence lists TWO structural differences in a single compound sentence
  with dash-separated enumeration. The footnote [^sd-fn] then ALSO carries the operating-
  layer distinction. This distributes the same distinction across both body prose and
  footnote, which is not "one footnote" per decision ⑤ but two-location difference emphasis.
  A reviewer familiar with SDMAE will read this as defensive enumeration.
  Blueprint §4.4 mandates: "차이 나열 없음 (R9). 각주 1개 추가(용어계보+구조차이만); 작동
  계층 차이는 §3.5 본문 1문장으로 이동." The body of §2.3 should NOT enumerate differences
  — it should position CSMAD as adapting the paradigm, with all differentiation detail
  in the footnote (structure) and §3.5 (operating layer). Currently §2.3 body has both.
- Evidence: Blueprint §4.4 R9 mandate; manuscript §2.3 body final paragraph.
- Fix: Remove the parenthetical "— rather than a branch-off from within the teacher decoder —
  and the Student is additionally trained to suppress anomaly-specific information through
  a gradient reversal mechanism operating in representation space rather than in the output
  or loss space" from the §2.3 body sentence. Move the structural distinction to footnote
  [^sd-fn] (which currently lacks it — see BLOCKER-BP-02). The operating-layer distinction
  remains ONLY in §3.5 §1. Result: §2.3 body makes one neutral adaptation statement;
  footnote carries structural difference; §3.5 carries operating-layer difference. Clean
  three-way split, no R9 violation.

---

### 5.2 Highlights Bullet 2 — SDMAE Implicit Comparison

**MINOR-R9-02**
- Severity: MINOR
- Location: Highlights, bullet 2
- Issue: "combining a masked autoencoder with an asymmetric Teacher–Student decoder and
  gradient reversal to adversarially suppress anomaly-specific information in the Student's
  representation" does not name SDMAE and is fine. However, "asymmetric Teacher–Student
  decoder" is a phrase that will immediately call to mind SDMAE for anyone who has read it.
  The Highlights are independently reviewed by some editors. Blueprint §1.4 lists "Self-
  distillation" as a proposed keyword; the current keywords list "Asymmetric self-distillation"
  — this matches §2.3 body terminology and is acceptable. No fix required but note: if the
  Highlights are reviewed without context, bullet 2's phrasing could prompt SDMAE search.
  Low probability of materiality. NOTE level only.
- Evidence: Blueprint §1.3 keyword list; Highlights bullet 2.
- Fix: No change required at this stage.

---

## 6. Argument Chain Integrity

### 6.1 §3.1 — R10 Argument for Multivariate Setting Absent

**MAJOR-R10-01**
- Severity: MAJOR
- Location: §3.1 "Problem Formulation and Setting"
- Issue: Blueprint §12 R10 배치 전수표 places the R10 argument for "왜 다변량 시계열에서
  이 설정이어야 하는가" in §3.1 with content: "실제 CPS/서버 환경에서 labeled anomaly는
  운영 기록(고장 이력)에서 자연 발생하며, 다변량 센서 간 동기 이탈이 이상의 특징이므로
  표현 학습이 이 구조 정보를 포착해야 탐지가 가능하다." The manuscript §3.1 final sentence
  reads: "Multivariate data motivate this design: anomalies manifest as correlated deviations
  across channels, and filtering labeled anomalies as noise discards the very co-occurrence
  structure that the masking, loss, and gradient pathways of Sections 3.3–3.5 exploit."
  This is PRESENT but the argument form is inverted: it argues against filtering anomalies
  rather than arguing for why labeled anomalies are available in multivariate settings in
  the first place. The "why multivariate time series demands this setting" argument (operational
  logs naturally produce labeled anomalies at the channel-covariance granularity) is missing.
  Without it, a reviewer can ask: "Why is the contaminated semi-supervised setting specifically
  suited to multivariate data rather than univariate?" The answer involves the co-occurrence
  structure of faults across channels — present implicitly but not as an explicit causal
  chain.
- Evidence: Blueprint §5.2 R10 논증; §12 component table §3.1 row; manuscript §3.1.
- Fix: Add one sentence before the existing final sentence: "In practice, labeled anomaly
  events arise naturally from operational logs of industrial systems — fault records that
  document anomalies as correlated deviations across multiple sensor channels simultaneously,
  making multi-dimensional pattern recovery the central learning challenge."

---

### 6.2 §3.5 GRL Necessity Argument — Logical Gap

**BLOCKER-ARG-01**
- Severity: BLOCKER
- Location: §3.5 "Why gradient reversal is necessary beyond loss bifurcation"
- Issue: The argument runs: "Excluding anomalous patches from L_OD removes the demand that
  the Student *follow* the Teacher there, but not the possibility of *memorizing* anomaly-
  specific reconstruction patterns through repeated exposure — which would shrink the
  discrepancy exactly where it is most informative. Gradient reversal closes this route,
  forcing the Student's hidden states to be uninformative about anomaly identity regardless
  of whether anomalies appear in the loss."
  The logical gap: the argument claims that a Student trained on NORMAL patches in L_OD
  (anomalous patches excluded) can still "memorize anomaly patterns through repeated
  exposure." This requires that anomalous patches are VISIBLE to the Student even when
  excluded from the loss. But the anomaly-priority masking mechanism (§3.3) preferentially
  MASKS anomalous patches — they are in the masked set M, not the visible set V. Therefore
  the Student's input latent (which carries a stop-gradient from the encoder output on
  VISIBLE patches) never directly sees anomalous patches' reconstructions during forward
  pass. The "repeated exposure" memorization route exists at the ENCODER level (encoder
  sees visible patches, some of which may be from windows with anomalies nearby), NOT
  at the Student decoder output level. The argument as written implicitly assumes the
  Student receives anomalous patch latents as input — which is not the case given the
  masking architecture. A technically rigorous reviewer will catch this and dismiss the
  GRL necessity argument as architecturally invalid.
- Evidence: §3.3 masking logic (anomalous patches → masked set M, never visible V);
  §3.4 "Student: input latent carries a stop-gradient" from encoder over visible set V;
  §3.5 "memorizing anomaly-specific reconstruction patterns through repeated exposure."
- Fix: Rewrite the GRL necessity argument to be architecturally precise:
  "Although anomalous patches are preferentially placed in the masked set and excluded
  from L_OD, the encoder — shared across both decoders — processes windows that include
  anomalous temporal context in the visible patches surrounding a masked anomaly patch.
  Over training, the encoder's shared representation for the visible patches of anomalous
  windows encodes information about the anomaly context, and the Student decoder — reading
  this shared latent — can learn to exploit that contextual signal to reconstruct anomalous
  patterns at inference, shrinking the discrepancy precisely where it is most informative.
  Gradient reversal prevents this by adversarially suppressing anomaly-discriminative
  information from the Student's internal representation regardless of what supervision
  enters through L_OD."
  This makes the mechanism technically sound: contamination path is encoder-level context,
  not direct patch exposure.

---

### 6.3 §4.4 Label Sparsity — "three structural properties" Argument Completeness

**MAJOR-ARG-02**
- Severity: MAJOR
- Location: §4.4 "Why graceful degradation is expected"
- Issue: The three structural robustness arguments are present (anomaly-priority masking
  label-only, GRL activates only on labeled windows, base reconstruction is label-independent).
  However the logical conclusion — "As p decreases, the discrepancy component weakens
  smoothly while the reconstruction component is preserved" — is stated without addressing
  the interaction between GRL and the decreasing number of positive windows. As p decreases,
  the frequency of GRL-active batches drops, which means the adversarial suppression of
  anomaly-specific information in the Student ALSO weakens. This creates a confound:
  at low p, the discrepancy signal is weaker NOT just because fewer anomaly patches are
  prioritized for masking, but also because the Student is less aggressively prevented from
  memorizing anomaly patterns through encoder context (see BLOCKER-ARG-01). The argument
  as written implies clean monotonic degradation, but the actual mechanism (GRL deactivation
  at low p loosening suppression) could produce non-monotonic behavior or a floor effect
  different from the "reconstruction baseline" described. The blueprint §6.8 notes this
  sweep is placeholder (Fig. 3 unresolved) — but the ARGUMENT for expected behavior
  must be correct regardless of results.
- Evidence: §4.4 "GRL suppression activates only for windows containing a labeled anomaly,
  so unlabeled anomaly windows contribute no destabilizing adversarial gradient"; §4.4
  "As p decreases, the discrepancy component weakens smoothly."
- Fix: Add a fourth property or modify property (ii): "as p decreases, both the discrepancy
  signal AND the adversarial suppression of anomaly-specific information in the Student
  weaken proportionally — the two effects co-vary, so the Student's residual capacity to
  reconstruct anomalous patterns grows, but the base reconstruction error signal remains
  intact. Importantly, the degradation path does not cross the unsupervised floor because
  even with p=0, the reconstruction error is elevated at anomaly patches regardless of the
  suppression state."

---

## 7. Elsevier Requirements

### 7.1 Abstract Structure

**MINOR-ELR-01**
- Severity: MINOR
- Location: Abstract
- Issue: Blueprint §1.2 specifies 4-structure: (1) problem+motivation, (2) method 2–3
  sentences, (3) benchmark protocol 1 sentence, (4) results 1–2 sentences. The manuscript
  abstract has 7 sentences. Mapping: S1-S2 = problem (correct). S3 = method, covers
  three orthogonal mechanisms (correct). S4 = asymmetric Teacher-Student (correct, but this
  makes method 2 sentences). S5 = benchmark protocol (correct). S6 = results "competitive
  performance" (correct). S7 = label sparsity (correct, maps to results). S8 = code URL.
  Code URL sentence is present but blueprint §1.2 does not include it as a required
  structural element — it is fine as a closing sentence. Word count: approximately 180 words,
  within the 150–200 target. No structural BLOCKER, but the method coverage across S3-S4
  omits mention of "gradient reversal" by name — it is implied by "adversarially suppresses
  anomaly-specific information" but the term "gradient reversal" that appears in title and
  keywords does not appear in abstract. Elsevier expects abstract-title keyword alignment.
- Fix: In S3, change "gradient reversal that adversarially suppresses" to "a gradient
  reversal layer that adversarially suppresses" — this makes the technical term explicit
  and aligns with the paper title and keyword "Gradient reversal."

---

### 7.2 Highlights — Character Count Compliance

**MINOR-ELR-02**
- Severity: MINOR
- Location: Highlights (5 bullets)
- Issue: Blueprint §1.4 mandates each highlight ≤125 characters. The manuscript includes
  the note "Each highlight ≤ 125 characters per Elsevier requirement — re-verify in Phase 7
  after NUM-003 resolution." Character counts for current highlights:
  - Bullet 1 (104 chars): "We formalize a contaminated semi-supervised setting..." — PASS
  - Bullet 2 (145 chars): "We propose CSMAD, combining a masked autoencoder with an asymmetric Teacher–Student decoder and gradient reversal..." — FAIL (>125)
  - Bullet 3 (151 chars): "Labeled anomalies guide training via three orthogonal paths — anomaly-priority masking, loss bifurcation, and gradient-reversal suppression..." — FAIL (>125)
  - Bullet 4 (145 chars): "A contaminated benchmark protocol (chronological test-prefix incorporation) fills a structural gap..." — FAIL (>125)
  - Bullet 5 (includes [N] placeholder) — pending
  Bullets 2, 3, 4 exceed 125 characters. This is an Elsevier submission hard requirement,
  not a style suggestion.
- Fix: Shorten bullets 2, 3, 4 before submission. Examples:
  Bullet 2: "CSMAD integrates a masked autoencoder with asymmetric Teacher–Student decoding and gradient reversal to suppress anomaly-specific representations." (152 chars — still over; cut further: "CSMAD combines masked autoencoding, asymmetric self-distillation, and gradient reversal to suppress anomaly representations." (124 chars — PASS))
  Bullet 3: "Three orthogonal paths — anomaly-priority masking, loss bifurcation, and gradient reversal — integrate labeled anomalies into representation learning." (152 — cut: "Three orthogonal label-integration paths: anomaly-priority masking, loss bifurcation, and gradient-reversal suppression." (120 chars — PASS))
  Bullet 4: "Our contaminated benchmark protocol incorporates test-prefix splits into training, filling a gap absent in standard TSAD benchmarks." (133 chars — cut to 125 or less)

---

### 7.3 Keywords — Missing "Self-distillation"

**MINOR-ELR-03**
- Severity: MINOR
- Location: Keywords
- Issue: Blueprint §1.3 lists "Self-distillation" as one of 7 proposed keywords. The
  manuscript keywords are: "Multivariate time series; Anomaly detection; Semi-supervised
  learning; Masked autoencoder; Asymmetric self-distillation; Gradient reversal; Contaminated
  benchmark." The blueprint uses "Self-distillation" (plain) but manuscript has "Asymmetric
  self-distillation" — this is an acceptable expansion. Total: 7 keywords, within 6–7
  recommended range. No BLOCKER. However, "Contaminated benchmark" as a keyword may not
  be indexed by standard databases — consider whether "Benchmark protocol" or "Semi-supervised
  benchmark" has better indexability. Low-priority note.
- Fix: No change required. Flag for Phase 7 keyword indexability verification.

---

## 8. Placeholder Quality (R3)

### 8.1 FIG-1 — Caption in Manuscript vs Registry

**MINOR-PH-01**
- Severity: MINOR
- Location: §1 "[FIG-1]" placeholder; §2 sentence referencing Fig. 1
- Issue: §1 Para 4 writes: "Figure 1 contrasts the unsupervised paradigm, its label-aware
  filtering variant, and CSMAD's three-path label integration." This forward-reference will
  be a dangling pointer if FIG-1 is not produced. More critically, this sentence appears
  AFTER the contribution bullet list, while the placement spec (blueprint §3.2) puts Fig. 1
  "after Para 3, before contribution paragraph." The manuscript places [FIG-1] between Para 3
  and Para 4 (✓ correct position) but the reference sentence is in Para 4 after the
  contribution list — a reader encounters the sentence "Figure 1 contrasts..." only AFTER
  they have seen the contribution bullets, making the forward reference redundant. Consider
  either moving the figure reference sentence earlier or removing the redundant "Figure 1
  contrasts" sentence from Para 4 (since the placeholder marks the figure location already).
- Fix: Remove "Figure 1 contrasts the unsupervised paradigm, its label-aware filtering
  variant, and CSMAD's three-path label integration." from Para 4 — this information is
  implicit in the figure placement and the preceding paragraph. The figure will carry its
  own caption.

---

### 8.2 NUM-006/007/008/009 — Performance Claim Unwritable

**MAJOR-PH-02**
- Severity: MAJOR
- Location: §4.2 main results text
- Issue: The central performance paragraph in §4.2 contains six consecutive [X.XX]
  placeholders (NUM-006 through NUM-013) and one wins-count placeholder (NUM-006 nested
  inside a sentence about wins out of 6 families). As noted in BLOCKER-BP-05, these
  depend on incomplete experiments. The specific issue here is that §4.2 is formatted as
  final-draft narrative with placeholders, which creates a false impression of completeness.
  A Phase 5 artifact reviewed for argument quality cannot be judged on narrative claims
  that have no content. The section reads: "CSMAD achieves the highest PA%K-AUC F1 on
  [N] of the six dataset families" — this is not a claim, it is a template. No argument
  quality can be assessed here.
- Fix: All [X.XX] cells in §4.2 must be filled from experiment results before phase
  completion can be certified. This is already flagged in BLOCKER-BP-05 but the PH issue
  is distinct: it concerns whether the §4.2 section constitutes a reviewable artifact at all.

---

### 8.3 FIG-2 — GRL Position in Caption vs Body Description

**MINOR-PH-03**
- Severity: MINOR
- Location: §3.2 FIG-2 placeholder; §3.4 GRL description
- Issue: §3.2 writes "The Student and GRL branches read the encoder output through a
  stop-gradient" and §3.5 contains the GRL classifier description. Blueprint ADV BLK-002
  mandates Fig. 2 carry two labels: (1) GRL applied to "student decoder 마지막 층 hidden,
  output projection 이전" and (2) "GRL: training only" annotation. The Fig. 2 placeholder
  spec references PLACEHOLDER_REGISTRY.md. If the registry spec does not explicitly name
  these two mandatory labels, the figure producer may omit them. This cannot be verified
  from the manuscript alone.
- Evidence: Blueprint §5.3 ADV BLK-002; manuscript §3.2 FIG-2 placeholder.
- Fix: Verify PLACEHOLDER_REGISTRY.md FIG-2 entry explicitly lists both mandatory labels.
  If not, add them. This is a production risk, not an argument failure.

---

## 9. Future Plagiarism Risk

### 9.1 §2.3 SDMAE Description

**NOTE-PLAG-01**
- Severity: NOTE
- Location: §2.3 paragraph 3 (SDMAE body description)
- Issue: "Ristea et al. adapted this paradigm to video anomaly detection, embedding a deeper
  teacher decoder and a shallower student decoder within a masked autoencoder and scoring
  anomalies by the teacher–student reconstruction discrepancy at test time." This is a
  close paraphrase of SDMAE's abstract and §3 description. Given that the SDMAE paper
  (CVPR 2024) uses almost identical phrasing for its own method, the phrase "embedding a
  deeper teacher decoder and a shallower student decoder within a masked autoencoder" is
  at risk if SDMAE uses this exact phrasing. Verify against SDMAE §3 and restructure
  if overlap exceeds 6 consecutive unique words.
- Fix: Rephrase to: "Ristea et al. adapted this design to video anomaly detection, placing
  a capacity-limited student decoder alongside a deeper teacher inside a masked autoencoder
  and measuring their output discrepancy as the anomaly score at inference time."

---

### 9.2 §3.5 Focal Loss Variant Description

**NOTE-PLAG-02**
- Severity: NOTE
- Location: §3.5 GRL anomaly suppression loss, focal-style BCE description
- Issue: "trained with a focal-style BCE variant for severe class imbalance: unlike the
  standard focal loss [lin2017focal], whose modulating factor derives from the raw prediction,
  here it derives from the class-prior-weighted cross-entropy itself" closely parallels a
  potential future description of this custom loss in code documentation or a technical
  report. If any internal document uses this exact phrasing, it should be checked.
  More importantly, the Appendix §C.1 equation (C.3) is the definitive form; the §3.5
  description should cross-reference it rather than independently define the variant
  in prose that could diverge from (C.3) under revision.
- Fix: In §3.5, after the focal-style BCE description, add "(exact form: Eq. C.3)" to
  create a single canonical location and reduce risk of prose/equation divergence.

---

## 10. Missing Reviewer-Critical Details

### 10.1 §4.1.2 — Single Seed Disclosure

**MAJOR-CRIT-01**
- Severity: MAJOR
- Location: §4.1.2 and Table A.1
- Issue: The manuscript discloses "one seed-42 run per entity" (§4.1.2) and "single run per
  entity" (Table A.1). Blueprint §6.3 confirms "단일 run (baseline 중 random만 5-run mean±std)"
  and mandates disclosing that variance/confidence intervals are NOT reported. The manuscript
  does this correctly for the random baseline ("the random-score baseline is averaged over
  five independent runs (mean ± std)") but does not state explicitly that NO confidence
  intervals are provided for any other method including CSMAD. For Elsevier reviewers
  accustomed to statistical testing, the absence of any variance estimate across the 113
  entities is a notable methodological gap that must be explicitly acknowledged as a
  limitation, not merely implicit in the single-run disclosure.
- Evidence: §4.1.2 single-run statement; no limitation sentence covering CI absence.
- Fix: Add to §4.1.2 or §5 Conclusion limitations: "All results report single-run performance
  per entity (seed 42) with no confidence intervals; variance across seeds is not estimated
  and is a limitation of the current evaluation."

---

### 10.2 §4.1.3 — Affiliation F1 Threshold Key Not Named in Body

**MINOR-CRIT-02**
- Severity: MINOR
- Location: §4.1.3 Affiliation F1 definition
- Issue: Blueprint §6.4 / ADV MINOR-004 mandates the Affiliation F1 key used is
  "affiliation_f1_ar" (AR threshold based), NOT "affiliation_f1" (F1-optimal threshold).
  The manuscript §4.1.3 body describes: "computed at the anomaly-ratio threshold" — this
  is correct but does not name the key. Appendix §A.2 also describes "Binarization uses
  the anomaly-ratio threshold defined below" — correct. The issue is that nowhere in
  the body text is the F1-optimal threshold variant named and explicitly excluded, as
  required by blueprint to prevent reviewer confusion about which variant is used.
- Fix: Add to §4.1.3 Affiliation F1 paragraph: "(We use the anomaly-ratio threshold
  variant throughout, not the F1-optimal threshold variant; the latter is excluded from
  all rankings.)"

---

## 11. Vague Academic Phrasing

### 11.1 §3.4 "more consistently" — Unquantified Comparative

**MINOR-VAG-01**
- Severity: MINOR
- Location: §3.4 "Why the capacity gap matters"
- Issue: "the shallower Student replicates it on recurring normal patterns but fails more
  consistently on the atypical patterns characterizing anomalies." "More consistently" is
  a comparative without a referent — more consistently than what? The intended comparison
  is "the Student fails on anomalous patterns more consistently than the Teacher does," but
  this is not stated. Without the ablation showing symmetric (2L/2L) decoder (NUM-024),
  this claim has no quantitative support and the vague phrasing amplifies the weakness.
- Fix: Change to: "the shallower Student replicates normal patterns but fails more
  consistently on anomalous correlation patterns than a same-capacity decoder would,
  making the capacity gap a driver of the discrepancy signal (quantified in Appendix B.5)."

---

### 11.2 §4.2 "the structural distinction being gradient-level label integration rather than a multi-stage pipeline"

**MINOR-VAG-02**
- Severity: MINOR
- Location: §4.2 NRdetector comparison sentence
- Issue: "the structural distinction being gradient-level label integration rather than a
  multi-stage pipeline (Section 2.2)" is an appositive clause tacked onto a performance
  comparison sentence. It reads as academic throat-clearing. The §2.2 discussion is the
  canonical location for this argument; repeating a compressed version in §4.2 without
  numeric support weakens both. Either make the §4.2 comparison self-contained or remove
  the appositive and let §2.2 carry the weight.
- Fix: Remove the appositive "(the structural distinction being gradient-level label
  integration rather than a multi-stage pipeline (Section 2.2))" from the §4.2 sentence.
  The §2.2 reference in the main comparison table row will serve as reminder.

---

## 12. Places Requiring Later References

| Location | Required Citation | Status |
|----------|------------------|--------|
| §4.1.1 "original training splits contain no labeled anomalies by construction" | "clean-train assumption" literature 1–2 papers | Blueprint §16 / RT MINOR-01: Phase 4 수요 등재, status unknown in Phase 5 |
| §4.1.1 anomaly ratio threshold "per the convention of [xu2022anomalytransformer]" | Verify this is the standard citation for AR threshold; Blueprint §16 flags "AR threshold의 TSAD 문헌 관행 선례 확보 전 이 방어 논리 사용 금지" | xu2022anomalytransformer is cited — verify this paper explicitly establishes AR threshold as convention |
| §2.2 "deep semi-supervised MTSAD we are aware of, NRdetector" | Phase 4 반증 검색 for any post-NRdetector (2025–2026) deep SSL MTSAD papers | Blueprint §16 "NRdetector 거의 유일한 선행 반증 부재 검증" |
| §2.3 "Zhang et al. [zhang2022selfdistill]" | Verify this is TPAMI 2022; Blueprint §16 flags "서지 확인" | Must confirm journal/year before submission |
| §1 contribution bullet 1 "benchmark protocol that incorporates the chronological prefix" | Precedent citation for contaminated/re-split benchmark approach | Only NRdetector cited as precedent; check if additional TSAD re-split papers exist |

---

## 13. Summary Checklist

| ID | Severity | Section | Issue | Status |
|----|----------|---------|-------|--------|
| BLOCKER-BP-01 | BLOCKER | §4.1.1 | Argument ⑤ (NRdetector precedent) missing as standalone; current citation asserts unverified structural property | OPEN |
| BLOCKER-BP-02 | BLOCKER | §2.3 footnote [^sd-fn] | Structural difference (branch-off vs independent) missing from footnote; currently only in body prose | OPEN |
| BLOCKER-BP-03 | BLOCKER | §3.1 | Three-structure disclosure (②-1/②-2/②-3) insufficiently explicit; "semi-supervised is not your main experiment" attack not fully deflected | OPEN |
| BLOCKER-BP-05 | BLOCKER | §4.2 | Protocol-effect decoupling argument unwritable (NUM-015–019 experiments not run) | OPEN |
| BLOCKER-R8-01 | BLOCKER | §1 bullet 4; §5 | "confirms graceful degradation" overclaim on unresolved FIG-3 placeholder | OPEN |
| BLOCKER-R1-01 | BLOCKER | §2.2 | DeepMIL/WETAS/TreeMIL absent from §2.2 body; end-to-end distinction argument missing | OPEN |
| BLOCKER-R9-01 | BLOCKER | §2.3 body | Residual SDMAE difference-listing in body prose violates R9 | OPEN |
| BLOCKER-ARG-01 | BLOCKER | §3.5 | GRL necessity argument architecturally incorrect (memorization route misidentified) | OPEN |
| MAJOR-BP-04 | MAJOR | §4.2 | Protocol-effect two-step argument structure unclear after Table 4 absorption | OPEN |
| MAJOR-R8-02 | MAJOR | §1 Para 3; §2.2 | D-008 Xue/SLA-VAE not cited at §1 "first" claim site | OPEN |
| MAJOR-R1-03 | MAJOR | §1 bullet 3; §B.5 | Symmetric decoder ablation unrun; bullet 3 not downgraded per blueprint conditional | OPEN |
| MAJOR-R10-01 | MAJOR | §3.1 | R10 multivariate motivation argument incomplete (why labeled anomalies are multivariate-specific) | OPEN |
| MAJOR-ARG-02 | MAJOR | §4.4 | Graceful degradation argument incomplete (GRL deactivation at low p not addressed) | OPEN |
| MAJOR-PH-02 | MAJOR | §4.2 | Central performance section is template, not argument (all result numbers are placeholders) | OPEN |
| MAJOR-CRIT-01 | MAJOR | §4.1.2; §5 | Single-seed / no CI limitation not explicitly acknowledged | OPEN |
| MINOR-R8-03 | MINOR | Abstract; §5 | Register inconsistency: hedged main claim + assertive secondary claim | OPEN |
| MINOR-R1-02 | MINOR | §4.1.4 | TFMAE placement gap (not mentioned in §2.1; no bridge in §4.1.4) | OPEN |
| MINOR-ELR-01 | MINOR | Abstract | "gradient reversal" not named in abstract; keyword-abstract misalignment | OPEN |
| MINOR-ELR-02 | MINOR | Highlights | Bullets 2, 3, 4 exceed 125-character Elsevier limit | OPEN |
| MINOR-ELR-03 | MINOR | Keywords | "Contaminated benchmark" indexability risk | OPEN |
| MINOR-VAG-01 | MINOR | §3.4 | "more consistently" unquantified; no ablation support cited | OPEN |
| MINOR-VAG-02 | MINOR | §4.2 | Appositive structural-distinction clause redundant in performance section | OPEN |
| MINOR-PH-01 | MINOR | §1 | "Figure 1 contrasts" sentence redundant after figure placement | OPEN |
| MINOR-PH-03 | MINOR | §3.2 | Fig. 2 mandatory labels (GRL position + "training only") not verifiable from manuscript alone | OPEN |
| MINOR-CRIT-02 | MINOR | §4.1.3 | Affiliation F1 key (AR threshold variant) not named and F1-optimal explicitly excluded | OPEN |
| NOTE-PLAG-01 | NOTE | §2.3 | SDMAE description close paraphrase risk | OPEN |
| NOTE-PLAG-02 | NOTE | §3.5 | Focal-style BCE description not cross-referenced to Eq. C.3; divergence risk | OPEN |
| NOTE-R9-02 | NOTE | Highlights | "asymmetric Teacher–Student decoder" may invite SDMAE search in Highlights review | OPEN |
