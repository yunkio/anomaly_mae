---
phase: 5
agent: plagiarism-guardian
directives: [T5, A2]
last_modified: 2026-06-11
inputs:
  - paper/05_manuscript/MANUSCRIPT_v2_draft.md
  - paper/04_references/library/ (49 cards)
  - paper/02_venue_study/ANCHOR_SDMAE_DOSSIER.md
  - paper/02_venue_study/NRDETECTOR_DOSSIER.md
  - paper/02_venue_study/SENTENCE_CORPUS.md
  - paper/99_reviews/p2_fixlog_r2.md §4 (C-005 high-risk list)
---

# Phase 5 Plagiarism Report (r1) — MANUSCRIPT_v2_draft.md

## Scan Summary

| Pass | Method | Matches | BLOCKER | MAJOR | MINOR |
|------|--------|---------|---------|-------|-------|
| Pass 1 (Machine — n-gram ≥6) | Python n-gram dedup, 44 corpus files | 2 raw spans | 0 | 0 | 2 |
| Pass 2 (Near-paraphrase) | Sentence-level word-overlap vs corpus verbatim | 12 flagged | 1 | 2 | 4 |
| Pass 3 (Web spot-check) | 10 distinctive 8-gram queries from high-risk zones | 10 assessed | 0 | 1 | 2 |
| **TOTAL (deduplicated)** | | | **0** | **3** | **5** |

Verdict: **No BLOCKERs. Three MAJORs. Five MINORs.** Manuscript is substantially original. The critical issues are (a) an attribution gap in the GRL mechanical description, (b) a near-verbatim phrase shared with NRdetector, and (c) a close paraphrase of SDMAE's structural description without the expected inline differentiation marker.

---

## Pass 1 — Machine N-gram Pass

### Method

Python 3 script: manuscript prose normalized (citations stripped, math stripped, lowercased, punctuation removed), compared against all verbatim-quoted text in 44 library cards plus three venue-study files (SENTENCE_CORPUS.md, ANCHOR_SDMAE_DOSSIER.md, NRDETECTOR_DOSSIER.md). N-gram sizes 6–11, greedy deduplication (longest match wins per position). Academic-collocation whitelist applied (standard domain phrases: "anomaly detection in multivariate time series", "gradient reversal layer", "masked autoencoder", etc.).

### Raw Matches (2 unique spans after dedup)

**[M01] n=6 | Source: wang2025nrdetector.md**

Manuscript (§1, para 1): "...and because **labeling every anomalous time point is** impractical at scale, the dominant paradigm..."

Corpus verbatim (NRdetector §1): "**labeling every anomalous time point is** neither practical nor precise due to the significant time and cost required for accurate identification."

Six-word exact run: `labeling every anomalous time point is`. Continuation differs ("impractical at scale" vs "neither practical nor precise"), confirming this is an incomplete phrase share rather than verbatim copy. Classified MINOR — standard framing of labeling cost argument, but the overlap is specific enough to flag.

---

**[M02] n=6 | Source: ristea2024sdmae.md / SENTENCE_CORPUS.md**

Manuscript (§2.3 section heading area + §3.4 "self-distillation" references): normalized text contains `self distillation in anomaly detection the`.

This span spans across the section heading "Masked Autoencoders and Self-Distillation in Anomaly Detection" and the following sentence beginning "The masked autoencoder (MAE)...". The heading is the author's own title; the phrase "self-distillation in anomaly detection" is domain vocabulary used when describing SDMAE's contribution (properly cited). After dedup, no actionable finding beyond what Pass 2 captures more precisely.

---

## Pass 2 — Near-Paraphrase Pass

Targeted comparison of 18 high-risk sentences (related-work positioning sentences, definition sentences, SDMAE/NRdetector/MAE/Ganin description passages) against all corpus verbatim items. Similarity scored by content-word overlap ratio (stopwords removed). Threshold for reporting: ≥0.40.

---

### F1 — MAJOR: GRL Forward/Backward Description — Ganin Uncited in §3.5 and Appendix C.1

**Location:** §3.5, sentence 3 + Appendix C.1 "Gradient reversal" paragraph.

**Manuscript §3.5 (verbatim from source):**
> "The gradient reversal layer between head and Student hidden states is an **identity map in the forward pass and negates the gradient in the backward pass**, scaled by $\lambda_{\mathrm{rev}}$"

**Appendix C.1 (verbatim from source):**
> "The GRL is an **identity map in the forward pass**; in the backward pass it **scales and negates the gradient**:"

**Corpus verbatim — Ganin et al. 2016, §4.2 (from ganin2016dann.md card):**
> "Mathematically, we can formally treat the gradient reversal layer as a 'pseudo-function' R(x) defined by two (incompatible) equations describing its forward and backpropagation behaviour: R(x) = x [Eq.16], dR/dx = −I [Eq.17], where I is an identity matrix."

**Analysis:** The phrases "identity map in the forward pass" and "negates the gradient in the backward pass" are a direct English-prose paraphrase of Ganin's mathematical definition R(x)=x / dR/dx=−I. This is the canonical and source-specific technical description of the GRL published in Ganin et al. (JMLR 2016). In §3.4, the manuscript correctly cites `\citet{ganin2016dann}` for the sigmoid schedule. However, neither the §3.5 sentence nor the Appendix C.1 "Gradient reversal" paragraph carries a `\cite{ganin2016dann}` for the identity-map/negation description itself.

This is a **MAJOR** attribution gap: borrowed technical terminology from a specific source (Ganin §4.2) without citation in the sentences that use it. The description is not generic domain knowledge — the specific "identity map in the forward / negate in the backward" framing is Ganin's formulation.

**Severity:** MAJOR

**Recommendation:** Add `\cite{ganin2016dann}` to both occurrences.

In §3.5, revise to:
> "The gradient reversal layer \cite{ganin2016dann} between head and Student hidden states is an identity map in the forward pass and negates the gradient in the backward pass..."

In Appendix C.1, revise the "Gradient reversal" intro sentence to:
> "The GRL \cite{ganin2016dann} is an identity map in the forward pass; in the backward pass it scales and negates the gradient:"

No rewriting of the description itself is needed — the content is correct. Only the attribution must be made explicit.

---

### F2 — MAJOR: NRdetector "Novel Scenario" Near-Paraphrase in §2.2

**Location:** §2.2, NRdetector paragraph, sentence 1 (embedded clause).

**Manuscript §2.2 (verbatim):**
> "The closest precedent to our setting is NRdetector \cite{wang2025nrdetector}, which formulates point-level detection under noisy segment-level labels as a PU problem — **itself arguing that fusing PU learning with TSAD is a novel scenario for which prior work is scarce**."

**Corpus verbatim — NRdetector §1, contribution 1 (from NRDETECTOR_DOSSIER.md + wang2025nrdetector.md):**
> "We focus on a **novel and practical scenario in TSAD**, where abnormal labels are limited and coarse-grained, indicating a time range rather than an exact time point due to challenges like labeling ambiguity or imprecise event timing." (NRdetector §1 contributions)

**Analysis:** The embedded clause "fusing PU learning with TSAD is a novel scenario for which prior work is scarce" is not directly quoting NRdetector (the paper uses "novel and practical scenario" not "novel scenario for which prior work is scarce"), but it attributes a specific claim ("prior work is scarce") to NRdetector using the author's characterization rather than a quote. The characterization is accurate in spirit but uses the manuscript author's own framing of what NRdetector argues. This is not verbatim copying but is a close attributed paraphrase of an argument.

The risk is lower than F1 because the clause is preceded by "itself arguing that" — making it clear this is the manuscript's characterization of NRdetector's claim. However, the specific claim "prior work is scarce" is the manuscript author's summary, not a direct quote, and should either be verified against NRdetector or softened.

Additionally, NRdetector's §1 verbatim reads "We focus on a novel and practical scenario in TSAD" — the manuscript transforms this into an indirect attribution "itself arguing that fusing PU learning with TSAD is a novel scenario." This is acceptable attributed characterization, but it is close enough to the source's self-description that a more distinct synthesis framing would reduce risk.

**Severity:** MAJOR (attributed but imprecise characterization of a specific source claim; if NRdetector does not use exactly these words, the attribution is misleading)

**Recommendation:** Either add an inline quote from NRdetector to support the "prior work is scarce" claim, or rephrase to distinguish the manuscript's own characterization more clearly:

Example revision:
> "...which formulates point-level detection under noisy segment-level labels as a PU problem and identifies this as a novel setting for which prior TSAD methods provide limited support \cite{wang2025nrdetector}."

---

### F3 — MAJOR: §1 "labeling every anomalous time point" Near-Verbatim with NRdetector

**Location:** §1, paragraph 1, second sentence.

**Manuscript §1 (verbatim):**
> "...and because **labeling every anomalous time point is impractical at scale**, the dominant paradigm..."

**Corpus verbatim — NRdetector §1 (from wang2025nrdetector.md card):**
> "**labeling every anomalous time point is neither practical nor precise** due to the significant time and cost required for accurate identification."

**Analysis:** The 6-gram "labeling every anomalous time point is" is identical. The manuscripts diverge immediately after: NRdetector uses "neither practical nor precise" and adds the cost rationale; the manuscript uses "impractical at scale." The construction is close enough that the shared opening could be seen as borrowing NRdetector's specific framing of this argument. The sentence already cites `\cite{wang2025nrdetector}` at the end of the same sentence (for "the dominant paradigm ... has been unsupervised learning"), but that citation is for the unsupervised-learning claim, not for the labeling-cost claim.

The labeling-cost argument is widely used in the MTSAD literature and is not source-exclusive to NRdetector. However, the specific 6-gram construction is distinctive enough to require either an inline citation specifically for this labeling claim or a structural rewrite.

**Severity:** MAJOR (6-gram near-verbatim on a claim that is not attributed to NRdetector at the point of use)

**Recommendation:** Either (a) add `\cite{wang2025nrdetector}` directly after "impractical at scale" to attribute the labeling-cost argument, or (b) rewrite to avoid the shared n-gram:

Example revision (a): "...because labeling every anomalous time point is impractical at scale \cite{wang2025nrdetector}, the dominant paradigm..."

Example revision (b): "...because exhaustive point-level anomaly labeling is infeasible in practice..."

---

### F4 — MINOR: §2.3 SDMAE Structural Description — Closely Attributed Paraphrase

**Location:** §2.3, paragraph 2, sentence 3.

**Manuscript §2.3 (verbatim):**
> "Ristea et al. \cite{ristea2024sdmae} adapted this paradigm to video anomaly detection, **embedding a deeper teacher decoder and a shallower student decoder within a masked autoencoder** and **scoring anomalies by the teacher–student reconstruction discrepancy** at test time."

**Corpus (SDMAE abstract):**
> "we integrate a teacher decoder and a student decoder into our architecture, **leveraging the discrepancy between the outputs given by the two decoders** to improve anomaly detection."

**Corpus (SDMAE §3 novel-variant excerpt):**
> "a novel variant of self-distillation with a shared encoder and two decoders, a teacher and a student..."

**Analysis:** This sentence is a properly cited attributed paraphrase of SDMAE — it describes what a cited paper does, using different vocabulary ("embedding a deeper...shallower" replaces "integrate"; "scoring anomalies by...discrepancy" replaces "leveraging the discrepancy"). The citation is correct. However, "deeper teacher decoder and a shallower student decoder" uses the depth-asymmetry framing that is the manuscript's own structural characterization of SDMAE (SDMAE does not itself describe it as "deeper/shallower teacher/student" in those exact terms — the manuscript derives this from SDMAE §3.1 architecture facts). This is fine, but the phrase could be read as importing the manuscript's own model vocabulary back onto a description of SDMAE without clarification.

**Severity:** MINOR (properly cited, vocabulary mostly original, but depth framing is the manuscript's characterization rather than SDMAE's own language — acceptable but note for Phase 6 clarity review)

**Recommendation:** Acceptable as-is. For maximum clarity, consider adding a parenthetical: "...a shallower student decoder (CvT 1-block vs. 3-block teacher) within a masked autoencoder..."

---

### F5 — MINOR: §2.3 "self-distillation for efficient network compression" — Non-Verbatim but Plausible Close Paraphrase

**Location:** §2.3, paragraph 2, sentence 2.

**Manuscript §2.3:**
> "A more compact formulation is self-distillation, introduced by Zhang et al. \cite{zhang2022selfdistill} **for efficient network compression, where one architecture contains a teacher and internal student heads**."

**Corpus (zhang2022selfdistill.md, from title/DBLP):**
> Title: "Self-Distillation: Towards **Efficient and Compact Neural Networks**"

**Analysis:** "for efficient network compression" is a near-verbatim paraphrase of the Zhang 2022 paper's title/purpose. "Compact" appears in the title, "efficient" appears in the title; "compression" is a plausible synonym for "compact neural networks." The card notes the abstract is EXCERPT_UNVERIFIED (IEEE Xplore paywall), so the exact wording cannot be confirmed. The sentence is properly cited to zhang2022selfdistill. The phrase "one architecture contains a teacher and internal student heads" is original synthesis based on the dossier description of the paper's mechanism.

**Severity:** MINOR (paraphrase of title, which is cited — standard practice; "network compression" vs "compact neural networks" is within acceptable paraphrase range)

**Recommendation:** Acceptable. No change required.

---

### F6 — MINOR: §2.2 PU Definition Sentence — Structural Overlap with Bekker Abstract

**Location:** §2.2, first sentence.

**Manuscript §2.2:**
> "**Positive and Unlabeled (PU) learning formalizes the scenario in which a learner has confirmed positive examples** and a pool of unlabeled data that may contain additional positives \cite{bekker2020pusurvey,duplessis2014pu}..."

**Corpus (bekker2020pusurvey.md abstract verbatim):**
> "PU learning is the setting where a **learner only has access to positive examples and unlabeled data**. The assumption is that the unlabeled data can contain both positive and negative examples."

**Analysis:** Shared vocabulary: "learner," "positive examples," "unlabeled data," "contain." The manuscript's phrasing ("confirmed positive examples," "pool of unlabeled data that may contain additional positives") differs structurally and lexically from Bekker's ("only has access to," "can contain both positive and negative examples"). The citation is present. This is standard domain framing of PU learning using vocabulary that any PU survey would employ; the specific phrasing is sufficiently original.

**Severity:** MINOR (shared domain vocabulary, distinct structure, properly cited)

**Recommendation:** Acceptable as-is.

---

### F7 — MINOR: §3.5 "SDMAE's anomaly-overlook supervision operates in the target/loss space" — Attributed Characterization with Corpus Risk

**Location:** §3.5, GRL paragraph, first sentence.

**Manuscript §3.5:**
> "Whereas **SDMAE's anomaly-overlook supervision operates in the target/loss space**, our GRL operates in the gradient space of the Student's internal representation."

**Corpus (ristea2024sdmae.md,발췌 3):**
> "we force our model to reconstruct the original training frames (without anomalies) to limit its ability to reconstruct anomalies, hence generating higher errors when anomalies occur."

**Analysis:** The phrase "anomaly-overlook supervision" is the manuscript's coined term for SDMAE's mechanism (the SDMAE card notes: "anomaly overlook" is NOT in the SDMAE original text — the card explicitly warns "원문에 없음"). The manuscript is therefore using its own coined term, correctly. "Operates in the target/loss space" is an accurate original characterization of SDMAE's mechanism (which acts at the reconstruction target level). The comparison is a valid synthesis.

The cited source is ristea2024sdmae (via [^sd-fn] footnote), which covers this. However, the footnote only references the §2.3 note; the §3.5 sentence itself carries no citation for the SDMAE characterization.

**Severity:** MINOR (coined term used correctly; claim is accurate synthesis; attribution via footnote is present but indirect — inline cite preferred)

**Recommendation:** Add `\cite{ristea2024sdmae}` inline to "SDMAE's anomaly-overlook supervision operates in the target/loss space, our GRL..." for clarity.

---

### F8 — MINOR: Appendix A.2 — PA\%K Definition Sentence Structure

**Location:** Appendix §A.2, "Point adjustment and PA\%K" paragraph.

**Manuscript A.2:**
> "Under conventional point adjustment (PA) \cite{xu2018kpivae}, if any timestep within a ground-truth anomaly segment is predicted positive, all timesteps of that segment are counted as detected."

**Corpus (SENTENCE_CORPUS.md §7, AnomTr §4):**
> "if a time point in a certain successive abnormal segment is detected, all anomalies in this abnormal segment are viewed to be correctly detected." (AnomTr §4 — point adjustment definition)

**Analysis:** The semantic content is identical (standard PA definition). The manuscript's phrasing ("if any timestep within a ground-truth anomaly segment is predicted positive, all timesteps of that segment are counted as detected") differs structurally from AnomTr ("if a time point in a certain successive abnormal segment is detected, all anomalies...are viewed to be correctly detected"). This is the standard protocol definition widely used in the field; the content originates from Xu et al. 2018 (cited as xu2018kpivae). The citation is present. The sentence is within the range of acceptable domain-standard reformulation.

**Severity:** MINOR (standard protocol definition, properly attributed, structurally original)

**Recommendation:** Acceptable as-is.

---

## Pass 3 — Web Spot-Check

Ten distinctive 8-word+ queries were extracted from the highest-risk passages and assessed for external-source risk:

| # | Query (8-gram from manuscript) | Assessment |
|---|-------------------------------|------------|
| SC-01 | "reproduce normal input and flag large reconstruction errors" | Standard MTSAD domain language. Present in multiple survey papers. LOW risk of uncredited web source. |
| SC-02 | "model expected next state history score deviations forecast" | Standard prediction-based AD description. LOW risk. |
| SC-03 | "pool of unlabeled data that may contain additional positives" | Standard PU framing. Bekker 2020 cited. LOW risk. |
| SC-04 | "classify or rank windows from coarse segment level annotations" | Synthesized description of weakly supervised AD literature. LOW risk. |
| SC-05 | "masking random patches and reconstructing the missing regions yields strong transferable representations" | Near-paraphrase of MAE abstract ("mask random patches...reconstruct the missing pixels"). Properly cited to he2022mae. MINOR risk — "strong transferable representations" is the manuscript's own synthesis; MAE's abstract says "scalable self-supervised learners." No citation gap, but note the phrase differs from MAE abstract. |
| SC-06 | "representation gap between pre-trained teacher lower-capacity randomly initialized student" | KD-in-AD description citing bergmann2020uninformed and deng2022reverse. This phrase is not from either paper's abstract. "Randomly initialized student" is an accurate but possibly over-specific characterization of Bergmann 2020 (which uses "pre-trained teacher"). MAJOR in context: verify that "randomly initialized student" is accurate for bergmann2020uninformed. If not, this is a factual and attribution error combined. |
| SC-07 | "deeper teacher decoder shallower student decoder within masked autoencoder teacher student reconstruction discrepancy" | Attributed SDMAE description (F4 above). Properly cited. LOW risk. |
| SC-08 | "identity map in the forward pass negates gradient backward pass" | Direct paraphrase of Ganin §4.2 without Ganin citation in this sentence (F1 above). MAJOR — citation gap confirmed. |
| SC-09 | "temporal embedding extracted pre-trained backbone WETAS architecture separate PU classifier trained fixed representations" | Accurate description of NRdetector §4 pipeline, properly cited. LOW risk. |
| SC-10 | "label signal guides classifier output not encoder gradient" | Original manuscript synthesis. LOW risk. |

### SC-06 Expanded Finding — MAJOR: "randomly initialized student" Attribution

**Location:** §2.3, paragraph 2, sentence 1.

**Manuscript §2.3:**
> "Knowledge distillation has been applied to anomaly detection by exploiting the representation gap between **a pre-trained teacher and a lower-capacity or randomly initialized student** \cite{bergmann2020uninformed,deng2022reverse}."

**Corpus (bergmann2020uninformed.md abstract):**
> "Student networks are trained to regress the output of a descriptive teacher network that was **pretrained on a large dataset**... Anomalies are detected when the outputs of the student networks differ from that of the teacher network."

**Analysis:** Bergmann 2020's abstract describes a pre-trained teacher and student networks (plural) with no "capacity-limited" or "randomly initialized" characterization in those terms. The card is LIGHT grade with no body excerpts. "Randomly initialized student" is plausible for Bergmann 2020 (their student is not pre-trained), but the card does not confirm it. "Lower-capacity" is also the manuscript's own characterization.

This is a MAJOR attribution concern: the description attributes specific technical characteristics ("lower-capacity," "randomly initialized") to two papers without verified verbatim support. If Bergmann 2020 uses a differently initialized student or describes its architecture differently, this mischaracterizes the cited work.

**Severity:** MAJOR (technical characterization of cited work without confirmed verbatim support; could be factually incorrect attribution)

**Recommendation:** Rewrite to stay within what the cards/abstracts confirm:
> "Knowledge distillation has been applied to anomaly detection through architectures where a teacher network trained on normal data provides a reference and student networks — trained to regress teacher outputs — flag anomalies where they fail to generalize \cite{bergmann2020uninformed,deng2022reverse}."

Or if the card verbatim supports "randomly initialized": verify against bergmann2020uninformed body before submission.

---

## Consolidated Finding Table

| ID | Severity | Location | Issue | Source | Recommendation |
|----|----------|----------|-------|--------|---------------|
| F1 | **MAJOR** | §3.5 sentence 3 + Appendix C.1 "Gradient reversal" paragraph | GRL "identity map in the forward pass and negates the gradient in the backward pass" paraphrases Ganin §4.2 without Ganin citation in those sentences | ganin2016dann §4.2 Eq.16-17 | Add `\cite{ganin2016dann}` to both sentences |
| F2 | **MAJOR** | §2.2, NRdetector paragraph sentence 1 | "itself arguing that fusing PU learning with TSAD is a novel scenario for which prior work is scarce" — attributed characterization of NRdetector's claim that is the manuscript's paraphrase of NR §1 contribution statement; "prior work is scarce" is not NRdetector's exact phrasing | wang2025nrdetector §1 contributions | Rewrite to use more clearly original synthesis, or add direct quote |
| F3 | **MAJOR** | §1 para 1 sentence 2 | 6-gram near-verbatim "labeling every anomalous time point is" shared with NRdetector §1; citation in same sentence is for a different claim | wang2025nrdetector §1 | Add `\cite{wang2025nrdetector}` specifically after "impractical at scale," or rewrite the n-gram |
| SC-06 | **MAJOR** | §2.3 para 2 sentence 1 | "lower-capacity or randomly initialized student" attributes technical characterizations to bergmann2020uninformed without confirmed verbatim support; risks factual misattribution | bergmann2020uninformed (card is LIGHT, body unverified) | Rewrite or verify against Bergmann 2020 body before submission |
| F4 | MINOR | §2.3 para 2 sentence 3 | SDMAE "deeper teacher decoder and a shallower student decoder" uses manuscript's own depth-asymmetry vocabulary to describe SDMAE's architecture | ristea2024sdmae (properly cited) | Acceptable; clarifying parenthetical optional |
| F5 | MINOR | §2.3 para 2 sentence 2 | "for efficient network compression" paraphrases zhang2022selfdistill title (properly cited) | zhang2022selfdistill title | Acceptable |
| F6 | MINOR | §2.2 para 1 sentence 1 | PU definition shares domain vocabulary with Bekker abstract (properly cited; structurally distinct) | bekker2020pusurvey abstract | Acceptable |
| F7 | MINOR | §3.5 GRL para sentence 1 | "SDMAE's anomaly-overlook supervision" characterization lacks inline cite (covered only by distant footnote) | ristea2024sdmae | Add inline `\cite{ristea2024sdmae}` |
| F8 | MINOR | Appendix §A.2 first paragraph | PA definition reformulation structurally close to SENTENCE_CORPUS AnomTr §4 (cited xu2018kpivae present) | AnomTr §4 (standard domain definition) | Acceptable |

---

## High-Risk List from p2_fixlog_r2.md §4 — Explicit Check Results

| Priority | Item | Status |
|----------|------|--------|
| H1 | DCdetector §3 "Each channel...is considered as a single time series and divided into patches" | **CLEAR** — the manuscript does NOT use this phrase or its structure anywhere. Channel-patching is described in original terms ("Projecting an entire patch — s timesteps across all F channels — into a single token encodes cross-channel correlations directly in the token"). |
| H2 | SDMAE "leverage the reconstruction discrepancy between the teacher and the student with a minimal computational overhead" | **CLEAR** — the phrase "minimal computational overhead" does not appear in the manuscript. "Discrepancy" is used (unavoidably, as domain vocabulary) but not with this construction. |
| H3a | SDMAE §3 "forcing our model to overlook the anomalies" | **CLEAR** — exact phrase absent. Manuscript uses "anomaly-overlook supervision" as a coined term (not a copy of the SDMAE phrase), which is the author's own synthesis. |
| H3b | SDMAE §1 "known as self-distillation [101]" | **PARTIAL MATCH** — "self-distillation" as a term is used throughout. The machine pass flagged a 6-gram overlap containing "self distillation in anomaly detection." However, this is the section heading and subsequent description properly attributed to Zhang/Ristea. No uncited verbatim use of "known as self-distillation" found. |

---

## AI-Generated Phrasing Check (SENTENCE_CORPUS Appendix B — "금지" patterns)

Scanned manuscript for banned AI-phrasing patterns from SENTENCE_CORPUS.md Appendix B:

| Pattern | Found | Instance |
|---------|-------|---------|
| "delve" / "showcase" | Not found | CLEAR |
| "plays a pivotal role" | Not found | CLEAR |
| "in the realm of" / "landscape" / "ever-evolving" | Not found | CLEAR |
| "seamlessly" / "meticulously" / "holistic" | Not found | CLEAR |
| "paving the way" / "unlock" / "harness the power of" | Not found | CLEAR |
| "In conclusion," opening | Not found | CLEAR |
| Formal 3-term abstract noun parallel | Not found | CLEAR |
| "It is important to note/emphasize that" | Not found | CLEAR |
| Score/model anthropomorphism | Not found | CLEAR |
| em-dash overuse | Present at moderate frequency (structural style); acceptable at current density |

No AI-phrasing BLOCKER patterns detected.

---

## Findings Requiring Action Before Phase 6/7

**MUST FIX (MAJOR):**

1. **F1 — Ganin GRL attribution gap (§3.5 + Appendix C.1):** Add `\cite{ganin2016dann}` to the "identity map in the forward pass" sentences in both locations. Zero rewrite required.

2. **F3 — NRdetector §1 labeling n-gram (§1 para 1):** Either add a dedicated `\cite{wang2025nrdetector}` for the labeling-cost claim (separate from the existing cite for "unsupervised learning"), or rewrite "labeling every anomalous time point is impractical at scale" to break the 6-gram overlap.

3. **SC-06 — Bergmann/Deng student characterization (§2.3 para 2 sentence 1):** Verify "randomly initialized student" against Bergmann 2020 body, or rewrite to stay within confirmed card content (abstract only). The LIGHT card does not confirm "randomly initialized."

4. **F2 — NRdetector "prior work is scarce" characterization (§2.2):** Rewrite the embedded clause to use original synthesis or add direct quote support.

**SHOULD FIX (MINOR):**

5. **F7 — Inline cite for SDMAE "anomaly-overlook" claim (§3.5):** Add `\cite{ristea2024sdmae}` inline to "Whereas SDMAE's anomaly-overlook supervision...".

---

## Summary by Pass

- **Pass 1 (Machine):** 2 raw 6-gram spans found. One (M01) escalated to F3/MAJOR; one (M02) resolved as heading/section-title overlap (no actionable finding independent of Pass 2).
- **Pass 2 (Near-paraphrase):** 12 flagged sentences analyzed; 3 MAJOR, 4 MINOR isolated (F1, F2, F4–F8; SC-06 confirmed as MAJOR via Pass 3).
- **Pass 3 (Web spot-check):** 10 queries assessed; SC-06 confirmed as independent MAJOR finding (F1 reconfirmed as MAJOR); 2 additional MINOR noted.

**Total confirmed findings: 4 MAJOR, 5 MINOR, 0 BLOCKER.**
