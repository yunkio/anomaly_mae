---
phase: 6
agent: style-auditor-B
directives: [T6]
last_modified: 2026-06-11
scope: MANUSCRIPT_v2.md — full prose, sentence-level, domain-convention perspective
notes: |
  Independent audit; AI_PHRASING_LEDGER.md and STYLE_AUDIT_A.md not consulted.
  Perspective: deep learning + time-series anomaly detection field conventions —
  method-description verbs, metric/experiment phrasing, architecture terminology,
  training/comparison idioms, anomaly-detection domain nouns. WebSearch evidence
  cited where usage frequency was uncertain.
---

# STYLE AUDIT B — Domain-Convention Review

**File audited:** `/home/ykio/notebooks/TSMAE/paper/05_manuscript/MANUSCRIPT_v2.md`
**Audit date:** 2026-06-11
**Perspective:** Field-idiomatic expression in deep learning and time-series anomaly detection

---

## Audit Findings

---

### B-001
**Location:** Abstract, sentence 3
**Original sentence:**
> "We propose CSMAD, an end-to-end framework that integrates labeled anomaly information directly into masked autoencoder representation learning through three orthogonal mechanisms: anomaly-priority masking, loss bifurcation between normal and anomalous reconstruction paths, and a gradient reversal layer that adversarially suppresses anomaly-specific information from the Student's internal representation."

**Issue:** "loss bifurcation between normal and anomalous reconstruction paths" is non-standard phrasing for this concept. The field idiom is "loss decomposition" or "separate loss terms for normal and anomalous samples" or "conditional loss." The phrase "bifurcation between ... reconstruction paths" does not appear in deep-learning anomaly detection literature; the split is in the *objective*, not in architectural reconstruction paths per se.
**Suggested revision:** "loss decomposition that applies separate reconstruction objectives to normal and anomalous patches"
**Severity:** Minor
**Evidence:** Search for "loss bifurcation" in anomaly detection: extremely rare. Compare with standard usage: "separate loss" (Ruff et al., 2020, DeepSAD), "normal-only reconstruction objective" (Ristea et al., 2022 SDMAE-style descriptions). "Decompose the training objective" is the conventional framing.

---

### B-002
**Location:** Abstract, sentence 4
**Original sentence:**
> "CSMAD employs an asymmetric Teacher–Student decoder architecture in which a capacity-limited Student's mimicry degrades preferentially on anomalous correlation patterns, amplifying the Teacher–Student discrepancy signal under contaminated training."

**Issue:** "a capacity-limited Student's mimicry degrades" — the possessive construction "Student's mimicry" is grammatically awkward and non-idiomatic for architecture description. The field convention is to predicate degradation on the Student directly: "the capacity-limited Student fails to mimic" or "the Student's reconstruction quality degrades preferentially." Also "degrades preferentially on anomalous correlation patterns" is acceptable but "degrades more severely on anomalous" is closer to standard ablation language in teacher–student anomaly detection papers (Bergmann et al. 2020; Deng & Li 2022 RD++).
**Suggested revision:** "in which the capacity-limited Student fails to mimic anomalous correlation patterns as faithfully as normal ones, amplifying the Teacher–Student discrepancy signal under contaminated training"
**Severity:** Minor

---

### B-003
**Location:** Abstract, sentence 5
**Original sentence:**
> "To enable evaluation of labeled-anomaly-aware methods, we introduce a contaminated benchmark protocol that incorporates the chronological prefix of the test stream into training, exposing labeled anomalies absent in the original train splits of standard benchmarks."

**Issue:** "exposing labeled anomalies absent in the original train splits" — the participial phrase is syntactically loose (modifies "protocol" or "training"?). The standard way to state this is "thereby introducing labeled anomalies that are absent from the original training splits." The verb "exposing" is non-idiomatic here; benchmark papers use "introducing," "incorporating," "providing access to."
**Suggested revision:** "that incorporates the chronological prefix of each test stream into training, thereby introducing labeled anomalies absent from the original training splits of standard benchmarks"
**Severity:** Minor

---

### B-004
**Location:** Abstract, sentence 6
**Original sentence:**
> "On [N] multivariate datasets spanning industrial and telemetry domains, CSMAD achieves competitive performance against [N] unsupervised and weakly supervised baselines under five rigorous evaluation metrics."

**Issue:** "under five rigorous evaluation metrics" — "rigorous" applied to metrics is evaluatively loaded and non-standard. The field norm is to qualify the *evaluation protocol* as rigorous or to name the metrics without the adjective: "under five evaluation metrics," "across five complementary metrics," "evaluated on five metrics." Adjectives like "rigorous," "comprehensive," and "extensive" applied to metrics are considered filler in reviewer-facing writing.
**Suggested revision:** "under five complementary evaluation metrics"
**Severity:** Minor
**Evidence:** Standard usage in benchmark papers: "evaluated under five metrics" (Liu et al. 2024 Elephant), "five evaluation protocols" (Paparrizos et al. 2022 VUS) — none append "rigorous" to the metric noun.

---

### B-005
**Location:** Abstract, sentence 7
**Original sentence:**
> "The model maintains robust detection as the labeled anomaly fraction decreases, validating the framework beyond the upper-bound labeling scenario."

**Issue:** "validating the framework beyond the upper-bound labeling scenario" is an unusual phrasing. "Upper-bound labeling scenario" is not a standard term; the field uses "fully labeled setting," "full-label regime," or "oracle labeling." "Beyond" is also vague — typically authors say "under reduced/partial labeling" or "toward the unsupervised limit." "Validates" is correct but slightly weak for an abstract; "demonstrating robustness" is more direct.
**Suggested revision:** "demonstrating robustness under reduced label availability, down toward the fully unsupervised limit"
**Severity:** Minor

---

### B-006
**Location:** §1 Introduction, paragraph 1
**Original sentence:**
> "Anomalies in such streams manifest not in isolated channels but through correlated deviations across multiple sensor dimensions \cite{deng2021gdn, wu2025catch}, and because exhaustive point-level annotation of anomalies is infeasible at scale, the dominant paradigm for multivariate time series anomaly detection (MTSAD) has been unsupervised learning \cite{wang2025nrdetector}."

**Issue:** "the dominant paradigm ... has been unsupervised learning" — "has been" (present perfect) is conventionally used in this field to describe the historical state. This usage is acceptable, but the dominant convention in recent TSAD surveys/papers uses the simple present ("is unsupervised learning") because the paradigm still holds. Present perfect suggests the paradigm may have ended.
**Suggested revision:** "the dominant paradigm ... is unsupervised learning"
**Severity:** Minor

---

### B-007
**Location:** §1, paragraph 2, sentence 2
**Original sentence:**
> "Despite their differences, all four families share an implicit assumption that the training data are drawn entirely from normal operations."

**Issue:** No domain-convention issue. "Share an assumption that" is standard. "Drawn entirely from normal operations" is clear. This sentence is well-formed. [PASS]

---

### B-008
**Location:** §1, paragraph 2, sentence 3
**Original sentence:**
> "This assumption is structurally embedded: the methods have no architectural pathway for leveraging the information carried by labeled anomalies even when such labels are available — the best a label-aware variant can do is exclude confirmed anomaly windows from training, filtering contamination rather than learning from it \cite{wang2025nrdetector}."

**Issue:** "no architectural pathway" — "pathway" is a biology/neuroscience borrowing that sounds informal in an architecture description context. The standard deep-learning idiom is "no mechanism for exploiting," "no architectural provision for," or simply "no means of incorporating." "Filtering contamination rather than learning from it" is effective and field-appropriate.
**Suggested revision:** "the methods have no mechanism for leveraging labeled anomaly information even when such labels are available"
**Severity:** Minor
**Evidence:** "architectural pathway" does not appear in NeurIPS/ICML/ICLR anomaly detection papers reviewed; "architectural mechanism" or "component" is standard.

---

### B-009
**Location:** §1, paragraph 3, sentence 2
**Original sentence:**
> "These labeled anomalies are an obstacle for unsupervised methods — a source of contamination — but a valuable learning signal for semi-supervised ones."

**Issue:** This is well-phrased and idiomatic. "Learning signal" is the exact term used in self-supervised and semi-supervised learning literature. [PASS]

---

### B-010
**Location:** §1, paragraph 4 (key observation paragraph)
**Original sentence:**
> "Our key observation is threefold: labeled anomalies reveal (a) which temporal positions yield informative hard reconstruction targets, (b) which patches the Student decoder should avoid mimicking, and (c) what representational content should be actively erased from the Student's encoding."

**Issue:** "yield informative hard reconstruction targets" — "yield" is non-idiomatic here; reconstruction targets are not "yielded" by temporal positions. The standard idiom is "serve as hard reconstruction targets" or "provide informative reconstruction targets." Also "hard reconstruction targets" is borrowed from hard example mining; it is acceptable but slightly informal. "Actively erased" is a weak choice; the convention in adversarial representation learning is "actively suppressed," "removed adversarially," or "disentangled."
**Suggested revision (partial):** "(a) which temporal positions serve as hard reconstruction targets, ... (c) what representational content should be adversarially suppressed from the Student's encoding"
**Severity:** Minor

---

### B-011
**Location:** §1, paragraph 4, sentence 3
**Original sentence:**
> "Relying only on (b) is insufficient: a Student repeatedly exposed to anomalous patterns during training may learn to reconstruct them accurately through an indirect route, weakening the discrepancy signal at inference time; the active suppression of (c) closes this route at the representational level."

**Issue:** "through an indirect route" — "route" is informal; the standard phrasing in deep learning is "through an indirect pathway," "by an alternative mechanism," or "via spurious correlations." Also "closes this route" repeats the informal noun; convention: "forecloses this possibility at the representational level" or "prevents this by suppressing anomaly-discriminative features in the representation."
**Suggested revision:** "may learn to reconstruct anomalous patterns via an indirect pathway, weakening the discrepancy signal at inference; the adversarial suppression of (c) forecloses this at the representational level"
**Severity:** Minor

---

### B-012
**Location:** §1, contribution 3
**Original sentence:**
> "A deeper Teacher decoder (3 layers) establishes a stable normal-reconstruction reference, while a capacity-limited Student decoder (2 layers) fails to mimic anomalous correlation patterns more severely than normal ones — a design intended to make the Teacher–Student output discrepancy a reliable anomaly signal under contaminated training (quantified in Appendix B.5)."

**Issue:** "fails to mimic anomalous correlation patterns more severely than normal ones" — the comparative placement is ambiguous: does the Student fail more severely on anomaly patterns than it fails on normal ones, or does it fail to mimic anomaly patterns more severely than normal patterns? Standard phrasing: "the Student decoder's reconstruction quality degrades more severely on anomalous patterns than on normal ones" or "the Student fails to reproduce anomalous correlation patterns more than normal ones." The current ordering invites misreading.
**Suggested revision:** "the capacity-limited Student decoder (2 layers) degrades more severely on anomalous correlation patterns than on normal ones, making the Teacher–Student output discrepancy a reliable anomaly signal under contaminated training"
**Severity:** Moderate (potential misreading)

---

### B-013
**Location:** §2.1, paragraph 1, sentence 1
**Original sentence:**
> "Deep learning approaches to unsupervised MTSAD have matured into several well-defined families."

**Issue:** "have matured into" is a perfectly standard formulation in survey-style related-work sections. [PASS]

---

### B-014
**Location:** §2.1, paragraph 1
**Original sentence:**
> "Reconstruction-based methods train an encoder–decoder to reproduce normal input and flag large reconstruction errors \cite{...}."

**Issue:** "flag large reconstruction errors" — in the TSAD field, the convention is "flag samples/timesteps *with* large reconstruction errors" or "detect anomalies by thresholding the reconstruction error." "Flag large reconstruction errors" conflates the signal (error) with the action (flagging the *input*). However, this usage is common enough in quick survey sentences to be acceptable. Minor clarity issue.
**Suggested revision:** "flag inputs with large reconstruction errors" or retain as is.
**Severity:** Minor

---

### B-015
**Location:** §2.1, paragraph 1
**Original sentence:**
> "A more recent strand exploits association structure: transformer models that learn temporal dependencies \cite{xu2022anomalytransformer} or contrast multi-scale views of the series \cite{yang2023dcdetector} score the discrepancy between learned and actual patterns..."

**Issue:** "score the discrepancy between learned and actual patterns" — "actual patterns" is vague and slightly informal; the field term is "observed patterns," "input patterns," or "empirical attention distributions." Anomaly Transformer specifically scores the discrepancy between *association* (learned) and *series* (empirical) attention distributions. Using "actual" may be imprecise relative to the cited method.
**Suggested revision:** "score the discrepancy between learned association patterns and the observed input"
**Severity:** Minor

---

### B-016
**Location:** §2.1, paragraph 2
**Original sentence:**
> "When the training stream contains confirmed anomalous events — the contaminated setting that arises naturally from operational logs — these methods cannot distinguish known-anomalous from known-normal samples; labeled information is either discarded or treated as noise degrading the normal pattern \cite{wang2025nrdetector}."

**Issue:** "degrading the normal pattern" — standard TSAD/semi-supervised terminology is "corrupting the learned normal distribution," "degrading the learned normality model," or "contaminating the training distribution." "Degrading the normal pattern" is marginally informal and slightly imprecise; it sounds as if the anomaly damages the pattern itself rather than the model's approximation of it.
**Suggested revision:** "treated as noise that corrupts the learned normality model"
**Severity:** Minor

---

### B-017
**Location:** §2.2, paragraph 1
**Original sentence:**
> "Positive and Unlabeled (PU) learning formalizes the scenario in which a learner has confirmed positive examples and a pool of unlabeled data that may contain additional positives \cite{bekker2020pusurvey,duplessis2014pu}, with established solution families spanning cost-sensitive risk minimization via non-negative risk estimators \cite{kiryo2017nnpu}, class-prior-based probability correction \cite{elkan2008pu}, and two-step techniques that first extract reliable negatives before training a classifier \cite{bekker2020pusurvey}."

**Issue:** The sentence is syntactically dense but technically accurate. "Established solution families spanning" is slightly non-standard; the convention is "comprising" or "including." Also "two-step techniques that first extract reliable negatives before training a classifier" is accurate and matches the PU learning literature. The sentence is serviceable. [PASS with note on "spanning" → "comprising"]
**Severity:** Minor (style)

---

### B-018
**Location:** §2.2, paragraph 1
**Original sentence:**
> "Outside time series, these ideas have been adapted to anomaly detection through deviation networks with scarce labeled anomalies \cite{pang2019devnet} and deep semi-supervised anomaly detection objectives \cite{ruff2020deepsad}."

**Issue:** "adapted to anomaly detection through deviation networks" — "through" is ambiguous: it could mean "by means of" or "via the medium of." The standard phrasing is "applied to anomaly detection, including deviation networks" or "these ideas have found application in anomaly detection: deviation networks ... and deep semi-supervised objectives ..." No serious domain-convention violation; the sentence is readable.
**Severity:** Minor

---

### B-019
**Location:** §2.2, paragraph 2, sentence 1
**Original sentence:**
> "In the time-series domain, deep representation learning informed by label signals remains rare \cite{wang2025nrdetector}."

**Issue:** "informed by label signals" is reasonable but slightly informal; the convention is "guided by label information" or "conditioned on available labels." "Label signals" is used in some papers but "label information" is more standard in the broader semi-supervised learning literature.
**Suggested revision:** "deep representation learning guided by label information remains rare"
**Severity:** Minor

---

### B-020
**Location:** §2.2, paragraph 2
**Original sentence:**
> "A weakly supervised strand trains models to classify or rank windows from coarse segment-level annotations \cite{sultani2018deepmil,lee2021wetas,liu2024treemil}; the label is the sole learning signal, with no self-supervised reconstruction pretext."

**Issue:** "the label is the sole learning signal" — in weakly supervised anomaly detection, the convention is "the label serves as the primary/sole supervision signal." Using "the label is ... the sole learning signal" treats the label as a scalar, while "learning signal" refers to the gradient direction; the standard phrasing is "labels provide the sole supervision" or "labels constitute the only training signal." Minor.
**Suggested revision:** "labels serve as the sole supervision signal, with no self-supervised reconstruction pretext"
**Severity:** Minor

---

### B-021
**Location:** §2.2, paragraph 2
**Original sentence:**
> "Our use of labels differs in kind: rather than serving as the target of a classification or ranking objective, the label shapes the gradient of a masked-reconstruction pretext, steering what the encoder itself learns to represent."

**Issue:** "shapes the gradient of a masked-reconstruction pretext" — "pretext" is a standard self-supervised learning term (pretext task), but "shapes the gradient of a pretext" is a non-standard collocation; the field says "modifies the gradient flow through the pretext task" or "enters the gradient of the pretext objective" or "influences the encoder's representation learning via the pretext gradient." Also, "steering what the encoder learns to represent" is slightly informal; "shaping the encoder's learned representations" is more conventional.
**Suggested revision:** "rather than serving as the target of a classification or ranking objective, labels enter the gradient of the masked-reconstruction pretext task, shaping the encoder's learned representations"
**Severity:** Minor

---

### B-022
**Location:** §2.2, paragraph 3
**Original sentence:**
> "The closest precedent to our setting is NRdetector \cite{wang2025nrdetector}, which formulates point-level detection under noisy segment-level labels as a PU problem and identifies this as a novel setting for which prior TSAD methods provide limited support."

**Issue:** "identifies this as a novel setting for which prior TSAD methods provide limited support" — "provide limited support" is a non-standard way to describe capability gaps; the field convention is "fail to address," "offer limited coverage of," or "do not handle." "Provide limited support" sounds like a software API description.
**Suggested revision:** "identifies this as a novel setting that prior TSAD methods do not address"
**Severity:** Minor

---

### B-023
**Location:** §2.2, paragraph 3
**Original sentence:**
> "Its framework is a pipeline: a temporal embedding is extracted by a pre-trained backbone derived from the WETAS architecture, and a separate PU classifier is trained on those fixed representations; the label signal guides the classifier's output, not the encoder's gradient."

**Issue:** "a temporal embedding is extracted by a pre-trained backbone" — passive construction with "extracted by" is grammatically sound, but the conventional description in this field uses active voice: "a pre-trained backbone ... extracts temporal embeddings." This also avoids the ambiguity of whether the backbone is a tool or an agent. "The label signal guides the classifier's output" is fine.
**Suggested revision:** "a pre-trained backbone derived from the WETAS architecture extracts temporal embeddings, and a separate PU classifier is trained on those fixed representations"
**Severity:** Minor

---

### B-024
**Location:** §2.3, paragraph 1
**Original sentence:**
> "The masked autoencoder (MAE) of He et al. \cite{he2022mae} showed that masking random patches and reconstructing the missing regions yields strong transferable representations."

**Issue:** "yields strong transferable representations" — "yields" is acceptable but "produces" or "learns" is more common in architecture papers: "learns highly transferable representations," "produces strong transferable features." "Yields" is not wrong but slightly less idiomatic in the context of self-supervised learning papers. Also "transferable representations" is the standard term (correct usage).
**Severity:** Minor

---

### B-025
**Location:** §2.3, paragraph 1
**Original sentence:**
> "Our patch-based masking draws directly from this paradigm, adapted from the spatial domain to windows of multivariate sensor channels; similar masking-based reconstruction objectives in some time-series models \cite{fang2024tfmae} are independent developments — our design lineage traces to vision MAE."

**Issue:** "our design lineage traces to vision MAE" — "design lineage traces to" is an unusual phrasing; the standard convention is "our approach is directly inspired by" or "our design follows the vision MAE framework" or "our method builds on the vision MAE paradigm." "Lineage traces to" is a genealogical metaphor that sounds informal for a methods paper.
**Suggested revision:** "our design follows directly from vision MAE"
**Severity:** Minor

---

### B-026
**Location:** §2.3, paragraph 2
**Original sentence:**
> "Knowledge distillation has been applied to anomaly detection through teacher–student frameworks in which a student trained to match a pre-trained teacher's representations fails to do so on anomalous inputs, exposing the anomaly as a representation gap \cite{bergmann2020uninformed,deng2022reverse}."

**Issue:** "exposing the anomaly as a representation gap" — "exposing" is slightly informal; the convention is "manifesting as a representation gap" or "revealing anomalies through a widened representation gap." The verb "expose" implies an investigative or surveillance metaphor not standard in deep learning.
**Suggested revision:** "the anomaly is thereby revealed as a representation gap"
**Severity:** Minor

---

### B-027
**Location:** §2.3, paragraph 2
**Original sentence:**
> "A more compact formulation is self-distillation \cite{zhang2022selfdistill}, which performs the distillation within a single architecture."

**Issue:** "A more compact formulation" — "compact" applied to a distillation framework is non-standard; the field usually says "a more parameter-efficient," "a simpler," or "a self-contained formulation." "Compact" typically refers to model size or parameter count, not to the conceptual compactness of a distillation scheme.
**Suggested revision:** "A more self-contained formulation is self-distillation"
**Severity:** Minor

---

### B-028
**Location:** §3.1, paragraph 1
**Original sentence:**
> "A multivariate time series $\mathbf{X} \in \mathbb{R}^{T \times F}$ ($T$ timesteps, $F$ sensor channels) yields sliding windows $\mathbf{W} \in \mathbb{R}^{L \times F}$ of length $L$..."

**Issue:** "yields sliding windows" — "yields" as a verb for the operation of windowing is non-standard. The field says "is segmented into," "produces," "is partitioned into," or simply "from which we extract sliding windows." "Yields" here suggests a mathematical mapping rather than a data preprocessing step.
**Suggested revision:** "A multivariate time series ... is segmented into sliding windows $\mathbf{W} \in \mathbb{R}^{L \times F}$"
**Severity:** Minor
**Evidence:** Standard windowing descriptions in TSAD papers use "is partitioned into," "is segmented into," or "we extract windows of length $L$." "Yields" is more common for describing outputs of model computations, not preprocessing steps.

---

### B-029
**Location:** §3.1, paragraph 2
**Original sentence:**
> "We work under a **contaminated semi-supervised** setting:[^cs-fn] $\mathcal{D}_{\mathrm{train}}$ contains a large majority of unlabeled windows and a small fraction carrying anomaly labels, as in industrial fault records."

**Issue:** "as in industrial fault records" — "as in" is a fine comparative construction, but the convention for contextualizing the setting in this field is "as is typical in industrial fault logs" or "consistent with industrial operational logs." "As in industrial fault records" is too terse to fully motivate the setting; however, given the sentence is a succinct setting definition, it is borderline acceptable.
**Severity:** Very minor (borderline acceptable)

---

### B-030
**Location:** §3.1, paragraph 3
**Original sentence:**
> "Labels are not used at inference; the model outputs a per-timestep anomaly score."

**Issue:** "the model outputs a per-timestep anomaly score" — standard TSAD terminology: "assigns an anomaly score to each timestep," "produces a per-timestep anomaly score," or "generates point-level anomaly scores." "Outputs" is acceptable but "produces" or "assigns" is more idiomatic in anomaly detection papers when describing the inference behavior.
**Severity:** Very minor

---

### B-031
**Location:** §3.2, paragraph 1
**Original sentence:**
> "CSMAD comprises five functional blocks (Figure 2): a linear patch embedding, a shared Transformer encoder, a Teacher decoder, a Student decoder, and a training-only label-guided module that couples the Student branch to a window-level anomaly classifier through gradient reversal."

**Issue:** "a training-only label-guided module" — "module" is standard for architecture components. However, "label-guided module" is slightly informal for what is more precisely a "gradient reversal branch" or "adversarial training branch." In the GRL literature (Ganin et al. 2016 DANN), the component is called a "domain classifier" or "adversarial classifier branch." The term "label-guided module" is not a field-recognized name for this component type.
**Suggested revision:** "a training-only adversarial branch that couples the Student encoder representation to a window-level anomaly classifier through gradient reversal"
**Severity:** Minor

---

### B-032
**Location:** §3.2, paragraph 2
**Original sentence:**
> "The Student and GRL branches read the encoder output through a stop-gradient, so the encoder is optimized exclusively by the Teacher's reconstruction objective and the adversarial signal cannot corrupt the normal-pattern representation underpinning the anomaly score."

**Issue:** "read the encoder output through a stop-gradient" — "read ... through a stop-gradient" is a non-standard construction. The standard phrasing is "receive a stop-gradient copy of the encoder output" or "take as input the stop-gradient-detached encoder representations." "Read through" sounds like a software I/O operation.
**Suggested revision:** "The Student and GRL branches receive a stop-gradient copy of the encoder output, ensuring the encoder is optimized exclusively by the Teacher's reconstruction objective..."
**Severity:** Minor

---

### B-033
**Location:** §3.3, anomaly-priority masking paragraph
**Original sentence:**
> "This addresses a structural imbalance of contaminated training: anomalous patches are rare, so stochastic masking seldom selects them and the model learns to reconstruct *around* rather than *through* them."

**Issue:** "reconstruct *around* rather than *through* them" — this metaphor is unusual in reconstruction-based anomaly detection literature. The standard framing is "learns to reconstruct by ignoring" or "the model can achieve low training loss without attending to anomalous patches." "Reconstruct around/through" is intuitive but non-idiomatic in the field.
**Suggested revision:** "the model can achieve low training loss without attending to anomalous patches, effectively learning to ignore them"
**Severity:** Minor

---

### B-034
**Location:** §3.4, Teacher decoder paragraph
**Original sentence:**
> "Learnable mask tokens are inserted at positions in $M$, position embeddings added, and the full sequence passed through a self-attention-only decoder of depth $n_{\mathrm{T}}$ following the standard MAE design \cite{he2022mae}; a linear head projects hidden states $\{h^{\mathrm{T}}_i\}$ to reconstructions $\{o^{\mathrm{T}}_i\}$."

**Issue:** "position embeddings added" and "the full sequence passed through" — these are grammatically parallel but the first uses a past participle ellipsis while the second omits the auxiliary. In technical writing the convention is to complete the parallel structure: "position embeddings are added, and the full sequence is passed through..." Alternatively, use a single "with" gerund series. This is a grammatical consistency issue with a mild convention violation.
**Severity:** Minor (grammatical parallelism)

---

### B-035
**Location:** §3.4, "Why the capacity gap matters" paragraph
**Original sentence:**
> "A deeper Teacher faithfully learns the joint normal correlation structure; the shallower Student replicates it on recurring normal patterns but fails more severely on the atypical correlation patterns characterizing anomalies than a matched-capacity decoder would (quantified in Appendix B.5), so the output discrepancy carries a stronger anomaly signal than reconstruction error alone."

**Issue:** "the joint normal correlation structure" — "joint normal correlation structure" is a reasonable phrase but slightly redundant ("joint" and "correlation" overlap semantically). The standard phrasing for this concept in multivariate anomaly detection is "joint inter-channel correlations," "normal multivariate correlation patterns," or simply "normal correlation structure." Also "the atypical correlation patterns characterizing anomalies" — standard: "anomalous correlation patterns" or "the correlation patterns characteristic of anomalies."
**Suggested revision:** "A deeper Teacher faithfully learns the normal inter-channel correlation structure; the shallower Student replicates it on recurring normal patterns but fails more severely on the correlation patterns characteristic of anomalies..."
**Severity:** Minor

---

### B-036
**Location:** §3.4, GRL dual-λ structure paragraph
**Original sentence:**
> "Two independent quantities govern the gradient reversal branch once the Student is active: the **loss weight** $\lambda_{\mathrm{GRL}}$, set adaptively from the clamped ratio of main-loss to GRL-loss gradient norms — computed per batch and applied as the previous epoch's average..."

**Issue:** "set adaptively from the clamped ratio" — "set from" is non-standard. The convention is "computed as," "defined as," or "set to." Also "applied as the previous epoch's average" is slightly ambiguous — it means the weight for the current epoch is the running average computed over the previous epoch, but the phrasing sounds like it is applied *as* an average (i.e., through averaging), which is the intended meaning but could be clearer. Standard phrasing: "applied using the running average from the previous epoch."
**Suggested revision:** "computed as the clamped ratio of main-loss to GRL-loss gradient norms — evaluated per batch and applied as the exponential moving average from the previous epoch"
**Severity:** Minor

---

### B-037
**Location:** §3.5, Output discrepancy loss paragraph
**Original sentence:**
> "Let $P_n = \{i \in M : y^p_i = 0\}$ be the masked patches labeled normal; $L_{\mathrm{OD}}$ matches the Student's output to the Teacher's detached output on this subset only"

**Issue:** "$L_{\mathrm{OD}}$ matches the Student's output to the Teacher's detached output" — "matches X to Y" is somewhat ambiguous; standard loss description in teacher-student papers is "$L_{\mathrm{OD}}$ minimizes the distance between the Student's output and the Teacher's (detached) output" or "$L_{\mathrm{OD}}$ penalizes discrepancy between the Student output and the stop-gradient Teacher output." "Matches to" is borrowed from template-matching language and is slightly informal.
**Suggested revision:** "$L_{\mathrm{OD}}$ penalizes the discrepancy between the Student's output and the Teacher's detached output, restricted to this subset"
**Severity:** Minor

---

### B-038
**Location:** §3.5, GRL anomaly suppression loss paragraph
**Original sentence:**
> "A two-layer MLP head $g_\phi$ predicts from each masked patch's Student hidden state whether the enclosing window contains an anomaly ($y^w$ broadcast to all masked patches; strictly, the target indicates an anomaly within the masked region, which coincides with $y^w$ under anomaly-priority masking)..."

**Issue:** "predicts from each masked patch's Student hidden state" — the standard phrasing for classifier head description is "takes as input the Student hidden state of each masked patch and predicts" or "classifies each masked patch's hidden state as anomalous or normal." "Predicts from" is grammatically sound but inverts the standard subject–verb–object order of classifier descriptions in this field.
**Suggested revision:** "A two-layer MLP head $g_\phi$ takes as input the Student hidden state $h^{\mathrm{S}}_i$ of each masked patch and predicts whether the enclosing window contains an anomaly..."
**Severity:** Minor

---

### B-039
**Location:** §3.5, GRL anomaly suppression loss paragraph
**Original sentence:**
> "The gradient reversal layer \cite{ganin2016dann} between head and Student hidden states is an identity map in the forward pass and negates the gradient in the backward pass, scaled by $\lambda_{\mathrm{rev}}$; the resulting adversarial gradient — proportional to $-\lambda_{\mathrm{rev}} \cdot \lambda_{\mathrm{GRL}}$ — opposes the classifier's search for anomaly-discriminative features, pushing the Student toward anomaly-*invariant* internal states."

**Issue:** "opposes the classifier's search for anomaly-discriminative features" — "the classifier's search" is an anthropomorphizing, informal phrase; the standard framing is "prevents the Student from learning anomaly-discriminative features" or "penalizes the Student for encoding anomaly-discriminative information." The term "anomaly-invariant internal states" is acceptable — "invariant representations" is standard in domain-invariant learning literature.
**Suggested revision:** "the resulting adversarial gradient ... prevents the Student encoder from learning anomaly-discriminative features, driving it toward anomaly-invariant representations"
**Severity:** Minor

---

### B-040
**Location:** §3.5, "Why gradient reversal is necessary" paragraph
**Original sentence:**
> "Excluding anomalous patches from $L_{\mathrm{OD}}$ removes the demand that the Student *follow* the Teacher there, but it does not actively remove anomaly information from the Student's representation."

**Issue:** "removes the demand that the Student *follow* the Teacher" — "removes the demand" is a loose, informal construction; standard phrasing: "eliminates the requirement for the Student to match the Teacher's output at anomalous locations" or "relaxes the Student's output constraint at anomaly positions." "Follow" with italics is used for emphasis, which is fine, but the surrounding phrasing is informal.
**Suggested revision:** "Excluding anomalous patches from $L_{\mathrm{OD}}$ removes the requirement that the Student match the Teacher's output at those locations, but it does not actively suppress anomaly information from the Student's representation."
**Severity:** Minor

---

### B-041
**Location:** §3.5, "Why gradient reversal is necessary" paragraph
**Original sentence:**
> "Although anomalous patches are preferentially masked and therefore hidden from the encoder, the visible patches of an anomalous window still carry the surrounding anomalous context, and the shared encoder embeds that context into the latent sequence both decoders read."

**Issue:** "the shared encoder embeds that context into the latent sequence both decoders read" — "both decoders read" as a relative clause without a relative pronoun ("that both decoders read") is grammatically acceptable in informal English but in academic technical writing the relative pronoun should be explicit. Also "embeds that context into the latent sequence" is fine technically.
**Suggested revision:** "the shared encoder embeds that context into the latent sequence that both decoders consume"
**Severity:** Very minor

---

### B-042
**Location:** §3.6, Leave-one-out masking paragraph
**Original sentence:**
> "Each test window is scored under $N$ masking patterns — each patch masked alone, all patterns forwarded in parallel through the batch dimension — eliminating cross-patch interference at an inference cost of approximately $N$ single-window forward passes, an acknowledged limitation."

**Issue:** "forwarded in parallel through the batch dimension" — "forwarded through the batch dimension" is non-standard. In deep learning parlance, "processed in a single batched forward pass" or "evaluated as a batch of $N$ masked variants." "Forwarded through the batch dimension" implies the batch dimension is a conduit, which is confusing.
**Suggested revision:** "each patch masked individually, all $N$ masked variants processed as a single batched forward pass"
**Severity:** Minor

---

### B-043
**Location:** §3.6, Patch-level anomaly score paragraph
**Original sentence:**
> "For each masked patch $i$, the Teacher reconstruction error $r_i$ (MSE over its $s \cdot F$ values) and the Teacher–Student discrepancy $d_i = \|o^{\mathrm{T}}_i - o^{\mathrm{S}}_i\|^2 / (s \cdot F)$ are computed; the GRL classifier is not used at inference."

**Issue:** "are computed" — passive is standard here and correct. "the GRL classifier is not used at inference" — "at inference" is the standard phrase (also "at inference time" or "during inference"). "Not used at inference" is slightly abrupt; the convention is "the GRL branch is inactive at inference" or "the GRL classifier is discarded at test time." Minor.
**Severity:** Very minor

---

### B-044
**Location:** §3.6, Point-level aggregation paragraph
**Original sentence:**
> "Each timestep $t$ belongs to one or more (window, patch) pairs — indexing the covering windows by $u$ — and its final score is the mean of $\sigma_i$ over all such pairs"

**Issue:** "indexing the covering windows by $u$" — parenthetical "indexing X by Y" is a mathematical shorthand but unusual in running prose. The convention is to introduce the index before using it: "where $u$ indexes the windows covering $t$." Also "the mean of $\sigma_i$ over all such pairs" is correct; "averaged over all covering (window, patch) pairs" is slightly more idiomatic.
**Severity:** Very minor

---

### B-045
**Location:** §3.6, last sentence of Point-level aggregation
**Original sentence:**
> "Averaging across overlapping windows provides an ensemble effect that reduces single-window reconstruction-context variation."

**Issue:** "single-window reconstruction-context variation" is a non-standard compound noun. The standard phrasing is "variance due to window context" or "context-dependent reconstruction variance." "Reconstruction-context variation" as a hyphenated compound does not appear in TSAD literature.
**Suggested revision:** "Averaging across overlapping windows provides an ensemble effect that reduces context-dependent reconstruction variance."
**Severity:** Minor
**Evidence:** Standard in TSAD ensemble/aggregation literature: "variance due to the reconstruction context" (Su et al., OmniAnomaly-style scoring); "context-dependent variance" is the recognized term.

---

### B-046
**Location:** §4.1.1, Datasets paragraph
**Original sentence:**
> "We evaluate CSMAD on six real-world multivariate benchmark families — SWaT \cite{goh2016swat}, WaDi \cite{ahmed2017wadi}, PSM \cite{abdulaal2021psm}, SMD \cite{su2019omnianomaly}, and SMAP/MSL \cite{hundman2018telemanom} — spanning industrial control, IT infrastructure, and spacecraft telemetry: 113 learning units in total, or 114 evaluation units with SWaT's dual evaluation (below)."

**Issue:** "learning units" is a non-standard term; the field uses "entities," "machines," "channels," "datasets," or "time series." "113 learning units" will confuse readers — this likely means "113 training entities." In anomaly detection papers evaluating on SMD/SMAP/MSL, the standard term is "entities" (OmniAnomaly, Telemanom, and derivative papers consistently say "entities" or "machines/channels").
**Suggested revision:** "113 individual entities in total" or "113 entities (time-series units)"
**Severity:** Moderate — "learning units" is not a recognized field term and may confuse readers.
**Evidence:** Standard across TSAD papers: "entities" (OmniAnomaly, NRdetector), "machines" (SMD context), "channels" (SMAP/MSL context). "Learning units" does not appear in the benchmarked field literature.

---

### B-047
**Location:** §4.1.1, Contaminated benchmark protocol paragraph, sentence 2
**Original sentence:**
> "We therefore re-split each dataset at the temporal midpoint of its original test file: the earlier half joins the training data and the later half is reserved exclusively for evaluation, so labeled anomalies are genuinely present in training (ratios 0.52\%–6.20\%; SMD per-machine pending; Table 1)."

**Issue:** "the earlier half joins the training data" — "joins" is informal; standard: "is appended to" or "is concatenated with the original training set." In benchmark protocol papers, "join" is used for joining datasets (concatenation), not for describing a partition being folded into another set.
**Suggested revision:** "the earlier half is appended to the original training set and the later half is reserved exclusively for evaluation"
**Severity:** Minor

---

### B-048
**Location:** §4.1.2, Architecture and training paragraph
**Original sentence:**
> "We report no cross-seed variance or confidence intervals — a limitation of the current evaluation; only the random-score baseline is averaged over five runs (Appendix §A.1)."

**Issue:** "We report no cross-seed variance" — standard phrasing: "we do not report variance across random seeds" or "results are from a single run per entity (no multi-seed averaging)." "Report no cross-seed variance" is slightly ambiguous — it could mean "the variance is zero" rather than "we did not measure it."
**Suggested revision:** "Results are from a single run per entity; we do not report multi-seed variance — a limitation of the current evaluation."
**Severity:** Minor

---

### B-049
**Location:** §4.1.2, Epoch asymmetry disclosure paragraph
**Original sentence:**
> "All methods share the selection criterion, no early stopping, and best-epoch reporting; budgets reflect convergence characteristics — CSMAD needs the 250-epoch warmup before the Student activates — and baseline batch sizes follow each method's original implementation preset (Table A.3)."

**Issue:** "budgets reflect convergence characteristics" — "reflect" is slightly vague in this context; "are chosen to match each method's convergence requirements" or "are set according to each method's typical convergence behavior" is more precise. "Baseline batch sizes follow each method's original implementation preset" is fine.
**Severity:** Very minor

---

### B-050
**Location:** §4.1.3, Evaluation metrics paragraph (the long sentence)
**Original sentence:**
> "We adopt five metrics assessing complementary aspects of detection quality, following the multi-metric philosophy of recent benchmark analyses \cite{...}: **PA\%K-AUC F1**, which integrates the point-adjusted F1 of the PA\%K protocol \cite{kim2022rigorous} over the tolerance spectrum $K \in \{0, 1, \ldots, 100\}$, removing dependence on any particular $K$ — our primary metric and selection criterion; **PA\%K-AUC AUC-PR**, the same $K$-integration applied to the area under the precision–recall curve at each $K$ (obtained by a threshold sweep, hence threshold-free); **VUS-PR** and **VUS-ROC** \cite{paparrizos2022vus}, which sweep both a threshold and a temporal tolerance to measure ranking quality without an operating point, VUS-PR rated the most reliable single TSAD measure by a large-scale study \cite{liu2024elephant}; and **Affiliation F1** \cite{huet2022affiliation}, the harmonic mean of affiliation precision/recall measuring the temporal distance between predicted and ground-truth events, computed at the anomaly-ratio threshold (the F1-optimal-threshold variant is excluded from all rankings)."

**Issue (i):** "assessing complementary aspects of detection quality" — "assessing" as a participial modifier is correct but "that assess" would be more formal and conventional.
**Issue (ii):** "removing dependence on any particular $K$" — standard: "eliminating sensitivity to the choice of $K$" or "aggregating over all tolerance levels $K$."
**Issue (iii):** "VUS-PR rated the most reliable single TSAD measure by a large-scale study" — "rated" here is a past participial phrase that dangles; standard: "VUS-PR, which was identified as the most reliable single TSAD metric in a large-scale study \cite{liu2024elephant}."
**Issue (iv):** "measuring the temporal distance between predicted and ground-truth events" — "measuring the temporal distance" is inaccurate; Affiliation F1 measures *proximity* in terms of temporal alignment, not raw distance. Standard: "quantifying temporal proximity between predicted and ground-truth anomaly events."
**Suggested revision (iii):** "VUS-PR, identified as the most reliable single TSAD metric by a large-scale benchmark study \cite{liu2024elephant}"
**Severity:** Minor (iii), Minor (iv); Very minor (i, ii)

---

### B-051
**Location:** §4.1.4, Baselines paragraph
**Original sentence:**
> "We compare against 26 baselines: 22 unsupervised — nine simple-to-lightweight detectors following \cite{sarfraz2024quovadis}, six established deep TSAD systems \cite{...}, and seven recent competitive methods (including TFMAE, the time-series MAE variant discussed in Section 2.3) \cite{...} — and four weakly supervised methods exploiting labeled anomalies during training \cite{...}..."

**Issue:** "nine simple-to-lightweight detectors" — "simple-to-lightweight" is a non-standard compound modifier. The convention is "simple and lightweight detectors," "lightweight baselines," or (following Sarfraz et al.) "simple-to-complex baselines." The hyphenated compound "simple-to-lightweight" does not appear in anomaly detection literature and is ambiguous (does it mean ranging from simple to lightweight, or that they are both simple and lightweight?).
**Suggested revision:** "nine lightweight or simple-baseline detectors"
**Severity:** Minor

---

### B-052
**Location:** §4.1.4, Comparison conditions paragraph
**Original sentence:**
> "The main comparison uses the Q3 (normal-only) condition for all 22 unsupervised baselines: labeled anomaly regions are excised from the contaminated training data and the surviving normal segments concatenated with boundary-aware windowing."

**Issue:** "the surviving normal segments concatenated with boundary-aware windowing" — the participle "concatenated" requires an explicit subject or needs to be part of a proper clause. The sentence is syntactically incomplete: "the surviving normal segments [are] concatenated..." Standard: "the surviving normal segments are concatenated into the training set with boundary-aware windowing."
**Severity:** Minor (grammatical incompleteness)

---

### B-053
**Location:** §4.2, Main results paragraph, sentence 1
**Original sentence:**
> "Table 2 presents PA\%K-AUC F1 and VUS-PR for CSMAD and all 26 baselines across the six dataset families; full five-metric results are in Appendix §A.5 and per-entity results in Appendix §A.6."

**Issue:** No domain-convention issue. Standard table-reference sentence structure. [PASS]

---

### B-054
**Location:** §4.2, Main results — placeholder sentence
**Original sentence:**
> "CSMAD achieves the highest PA\%K-AUC F1 on [N] of the six dataset families and the highest VUS-PR on [N] <!-- PH:NUM-006 -->, averaging [X.XX] PA\%K-AUC F1 and [X.XX] VUS-PR <!-- PH:NUM-007 --> across families, and outperforms the strongest unsupervised competitor (Q3) by [X.XX] <!-- PH:NUM-008 --> absolute points in PA\%K-AUC F1 and [X.XX] <!-- PH:NUM-009 --> in VUS-PR on average."

**Issue:** "outperforms the strongest unsupervised competitor (Q3) by [X.XX] absolute points in PA\%K-AUC F1" — "absolute points" is fine but the convention is "by [X.XX] percentage points" (since PA\%K-AUC F1 is reported as a percentage or decimal) or simply "by a margin of [X.XX]." "Absolute points" is acceptable but "absolute percentage points" is more precise when the metric is reported as a percentage.
**Severity:** Very minor

---

### B-055
**Location:** §4.2, Protocol-effect analysis paragraph
**Original sentence:**
> "Under (i) the label-dependent pathways self-deactivate with the configuration held fixed (random masking, all-normal OD loss, no GRL loss), leaving a purely unsupervised asymmetric Teacher–Student MAE."

**Issue:** "the label-dependent pathways self-deactivate" — "self-deactivate" is a non-standard technical term. The convention is "are deactivated" (passive) or "the label-dependent components are disabled" or "reduce to their label-free equivalents." "Self-deactivate" implies autonomous behavior that could confuse readers about whether this is a designed mechanism or an emergent property.
**Suggested revision:** "the label-dependent pathways are automatically disabled (random masking, all-normal OD loss, no GRL loss), reducing the model to a purely unsupervised asymmetric Teacher–Student MAE"
**Severity:** Minor
**Evidence:** "Self-deactivate" does not appear in standard deep learning architecture descriptions. "Automatically disabled" or "deactivated" is the conventional term.

---

### B-056
**Location:** §4.3, Ablation study, Output discrepancy loss paragraph
**Original sentence:**
> "Removing $L_{\mathrm{OD}}$ eliminates the bifurcated signal driving the Student to deviate from the Teacher on anomalous patches while mimicking it on normal ones; the drop is [X.XX] points."

**Issue:** "the bifurcated signal" — "bifurcated signal" is a non-standard term in teacher-student learning; the concept is "selective distillation" or "conditional distillation objective." "Bifurcated" may be understood but it is not a recognized technical term in this context.
**Suggested revision:** "Removing $L_{\mathrm{OD}}$ eliminates the selective distillation objective that steers the Student to deviate from the Teacher on anomalous patches while tracking it on normal ones"
**Severity:** Minor

---

### B-057
**Location:** §4.4, Label sparsity analysis, Design paragraph
**Original sentence:**
> "The labeled fraction $p$ of training anomaly regions varies over $\{1.0, 0.75, 0.5, 0.25, 0.1\}$: a uniformly random selection of regions retains labels (region granularity, matching operational records) while the rest remain in training unlabeled, all else unchanged; at $p \to 0$ CSMAD reverts to the purely unsupervised mode of the protocol-effect analysis (Section 4.2)."

**Issue:** "region granularity, matching operational records" — this parenthetical is unclear; "at region granularity, consistent with how operational logs record fault events" is the implied meaning. As written, "region granularity, matching operational records" reads as two disconnected fragments.
**Suggested revision:** "a uniformly random subset of anomaly regions retains labels (at region granularity, consistent with operational log records)"
**Severity:** Minor

---

### B-058
**Location:** §4.4, Why graceful degradation paragraph
**Original sentence:**
> "Three structural properties support robustness: (i) anomaly-priority masking applies only to labeled patches, leaving the label-free reconstruction objective unaffected by which anomalies are labeled; (ii) the GRL term draws its positive supervision exclusively from labeled windows — batches without a labeled positive skip the term entirely — so unlabeled anomaly windows, treated as negatives, never inject an erroneous positive adversarial signal; and (iii) the base reconstruction error is label-independent, elevated wherever a patch deviates from normal correlation structure."

**Issue:** "draws its positive supervision" — "draws supervision" is a non-standard collocation; the convention is "relies on positive supervision from" or "uses labeled windows as positive examples." Also "skip the term entirely" is informal; "the term is omitted for such batches" is standard.
**Suggested revision (partial):** "(ii) the GRL term relies on labeled windows as positive training examples — batches containing no labeled positives omit the term entirely — so unlabeled anomaly windows, treated as negatives, never inject an erroneous adversarial signal"
**Severity:** Minor

---

### B-059
**Location:** §4.4, Results paragraph
**Original sentence:**
> "Performance declines as $p$ decreases but does so [gradually / monotonically] <!-- PH:NUM-027 -->, maintaining competitive detection at $p = 0.25$ and approaching the best unsupervised baseline at $p \approx 0$, confirming reversion to a pure reconstruction-based detector without falling below the unsupervised floor."

**Issue:** "confirming reversion to a pure reconstruction-based detector without falling below the unsupervised floor" — "reversion to" is acceptable but the field more commonly uses "degrading to" or "reverting to the behavior of" an unsupervised method. "Without falling below the unsupervised floor" is effective and clear, though "floor" is a borrowing from financial/statistics contexts; "without dropping below the performance of the unsupervised baseline" is more standard.
**Suggested revision:** "confirming that the model degrades gracefully toward the purely reconstruction-based limit, without falling below the unsupervised baseline performance"
**Severity:** Minor

---

### B-060
**Location:** §4.5, Qualitative analysis paragraph
**Original sentence:**
> "Figure 4 decomposes the CSMAD anomaly score for representative windows from [N] datasets; each panel shows four aligned traces — raw input with ground-truth anomaly regions shaded, Teacher reconstruction error, Teacher–Student discrepancy, and the combined score with the anomaly-ratio threshold."

**Issue:** "four aligned traces" — "traces" is standard in time-series visualization descriptions. However "Figure 4 decomposes" is slightly informal anthropomorphizing; conventional: "Figure 4 shows the decomposition of the CSMAD anomaly score" or "Figure 4 illustrates the score decomposition." "Decomposes" as a transitive verb with "Figure 4" as subject is non-standard.
**Suggested revision:** "Figure 4 illustrates the decomposition of the CSMAD anomaly score for representative windows..."
**Severity:** Minor

---

### B-061
**Location:** §4.5, sentence 2
**Original sentence:**
> "The two components respond distinctly: reconstruction error is elevated wherever the input deviates from learned normal patterns regardless of event type, while the discrepancy captures the additional divergence arising where the Student's limited capacity and adversarially suppressed representation fail to track the Teacher."

**Issue:** "the discrepancy captures the additional divergence arising where the Student's limited capacity and adversarially suppressed representation fail to track the Teacher" — "fail to track the Teacher" is slightly informal; in teacher–student anomaly detection, the standard term is "fails to match the Teacher's output" or "the Student-Teacher gap is widened." "Fail to track" is an acceptable colloquialism but "fail to replicate the Teacher's output" is more idiomatic for this subdomain.
**Severity:** Very minor

---

### B-062
**Location:** §5 Conclusion, paragraph 1
**Original sentence:**
> "This paper addressed the underexplored setting in which training data contain a small fraction of labeled anomalies alongside a majority of unlabeled observations — common in industrial deployments yet unsupported by standard MTSAD benchmarks or unsupervised methods."

**Issue:** "unsupported by standard MTSAD benchmarks or unsupervised methods" — the logical object of "unsupported" is ambiguous: does the setting lack support from (a) the benchmarks or (b) the methods, or both? The conventional phrasing is "not addressed by standard MTSAD benchmarks or existing unsupervised methods." "Unsupported" is engineering terminology rather than academic.
**Suggested revision:** "common in industrial deployments yet unaddressed by standard MTSAD benchmarks or existing unsupervised methods"
**Severity:** Minor

---

### B-063
**Location:** §5 Conclusion, paragraph 2
**Original sentence:**
> "We proposed CSMAD, which integrates labeled anomaly information into masked autoencoder representation learning through three orthogonal paths — anomaly-priority masking, loss bifurcation toward normal-only Student mimicry, and gradient-reversal suppression of anomaly-specific information — on top of an asymmetric Teacher–Student decoder architecture (3-layer Teacher, 2-layer Student) that converts the capacity gap into a reliable discrepancy signal under contaminated training."

**Issue (i):** "loss bifurcation toward normal-only Student mimicry" — "bifurcation toward" is not a natural English phrase; "bifurcation" (a split into two branches) and "toward" are semantically incompatible. The standard would be "conditional loss restriction to normal-patch mimicry" or "loss bifurcation separating normal and anomalous Student objectives."
**Issue (ii):** "converts the capacity gap into a reliable discrepancy signal" — this is acceptable but "exploits the capacity gap to produce a reliable discrepancy signal" is more idiomatic in the teacher–student literature.
**Severity:** Moderate (i) — "bifurcation toward" is logically and stylistically inconsistent

---

### B-064
**Location:** §5 Conclusion, paragraph 3
**Original sentence:**
> "A notable limitation is the cost of leave-one-out inference — an approximately 50$\times$ increase in forward-pass computation relative to single-mask scoring <!-- INTEGRATOR: resolved from protocol constant N=50 ... -->; reducing this inference cost is a natural avenue for future work."

**Issue:** "reducing this inference cost is a natural avenue for future work" — "a natural avenue for future work" is a clichéd conclusion sentence that appears ubiquitously in ML papers and is considered by reviewers as generic. The field-recommended formulation identifies the *specific* direction: "future work could adopt faster approximations such as a fixed subset masking strategy or a dedicated scoring head, reducing inference to a single forward pass."
**Severity:** Minor (field convention: avoid generic future-work statements)

---

### B-065
**Location:** §A.1, Baseline implementations paragraph
**Original sentence:**
> "The 22 unsupervised baselines comprise five simple detectors (random score, sensor-range deviation, PCA reconstruction, L2-norm, nearest-neighbor distance), three lightweight neural detectors (MLP, MLPMixer, single-stack Transformer), and a GCN-LSTM detector, following the protocol study of \cite{sarfraz2024quovadis}; six established deep TSAD systems (Anomaly Transformer, TranAD, USAD, DAGMM, GDN, OmniAnomaly); and seven recent methods (TFMAE, NPSR, TimesNet, DCdetector, MEMTO, ModernTCN, CATCH)."

**Issue:** "comprise ... following the protocol study of" — the participial "following" dangles; it modifies the subject "baselines" rather than the final element of the list, which is grammatically ambiguous. Standard: "five simple detectors (random score, ...), three lightweight neural detectors (...), and a GCN-LSTM detector — all adopted from \cite{sarfraz2024quovadis}; ..."
**Severity:** Minor (syntactic ambiguity)

---

### B-066
**Location:** §A.1, last sentence of Baseline implementations paragraph
**Original sentence:**
> "All baselines consume the identical data partitions through a unified loading layer, and all metrics — for CSMAD and every baseline — are computed by one shared evaluation routine, precluding implementation-level metric divergence."

**Issue:** "precluding implementation-level metric divergence" — "implementation-level metric divergence" is not a standard phrase; the conventional term is "implementation-dependent metric discrepancies" or "differences arising from reimplementation." "Metric divergence" in information theory refers to KL divergence; using it here is potentially confusing.
**Suggested revision:** "eliminating implementation-dependent metric discrepancies"
**Severity:** Minor

---

### B-067
**Location:** §A.2, Affiliation F1 definition
**Original sentence:**
> "Affiliation precision and recall convert the temporal distance between predicted and ground-truth events into per-event affinity scores within each event's affiliation zone, with formal robustness guarantees against adversarial scoring; Affiliation F1 is their harmonic mean."

**Issue:** "convert the temporal distance ... into per-event affinity scores" — "convert ... into" is an acceptable paraphrase of the Affiliation metric's mechanics. However, Affiliation P/R are not defined purely in terms of temporal distance but in terms of temporal proximity within affiliation zones; "temporal distance" slightly mischaracterizes the metric (it uses proximity-weighted affinity, not a distance per se). The standard description is "measure the proximity between predicted and ground-truth anomaly events within each event's affiliation zone."
**Suggested revision:** "Affiliation precision and recall measure the temporal proximity between predicted and ground-truth anomaly events within each event's affiliation zone..."
**Severity:** Minor (technical accuracy)

---

### B-068
**Location:** §B.1, paragraph
**Original sentence:**
> "For completeness, Table B.1 reports the complementary Q1 condition, in which the same 22 unsupervised baselines train on the full contaminated stream without excision — quantifying how much unaddressed contamination costs each method family and contextualizing the training-volume asymmetry acknowledged in Section 4.1.4."

**Issue:** "quantifying how much unaddressed contamination costs each method family" — "costs" is informal for academic writing; the convention is "measuring the performance degradation attributable to unaddressed contamination" or "quantifying the performance penalty of unaddressed contamination." "Costs" reads as a financial metaphor.
**Suggested revision:** "quantifying the performance penalty of unaddressed training contamination for each method family"
**Severity:** Minor

---

### B-069
**Location:** §B.2, paragraph
**Original sentence:**
> "To assess whether this asymmetry materially affects the comparison, representative unsupervised baselines are re-trained at extended budgets — and CSMAD at a reduced budget — under the otherwise unchanged protocol."

**Issue:** No domain-convention issue. "Materially affects" is standard in evaluation/comparison discussions. [PASS]

---

### B-070
**Location:** §B.5, Symmetric decoder capacity paragraph
**Original sentence:**
> "A symmetric decoder (Teacher 2L / Student 2L) removes the capacity gap behind the Student's preferential failure on anomalous patterns (Section 3.4); the change of [X.XX] points quantifies the asymmetric design — the architectural prior of contribution 3 — as an empirical effect."

**Issue:** "the architectural prior of contribution 3" — "architectural prior" is borrowed from Bayesian terminology and is non-standard as a description of a design choice in a deterministic architecture paper. The convention is "the design choice of contribution 3" or "the architectural inductive bias of contribution 3." "Prior" carries strong Bayesian connotations that are not intended here.
**Suggested revision:** "the asymmetric design — the architectural inductive bias underlying contribution 3 — as an empirical effect"
**Severity:** Minor

---

### B-071
**Location:** §C.1, Classification loss paragraph
**Original sentence:**
> "Unlike the standard focal loss \cite{lin2017focal}, which defines its modulating probability $p_t$ from the raw prediction, here $p_t := e^{-\ell_i}$ derives from the pos-weight-adjusted BCE, weighting hard examples by both confidence and prior imbalance; this variant is part of the present design rather than an external import."

**Issue:** "this variant is part of the present design rather than an external import" — "external import" is a software engineering term (importing a library). In a deep learning paper, this would conventionally be stated as "this variant is a novel design choice introduced in this work, not borrowed from prior literature" or "this is a design-specific modification, not a standard variant from prior work."
**Suggested revision:** "this variant is a design-specific modification introduced in this work, not a standard formulation from prior literature"
**Severity:** Minor

---

### B-072
**Location:** §C.1, Adaptive loss weights paragraph
**Original sentence:**
> "The adversarial gradient reaching the Student hidden state is therefore $-\lambda_{\mathrm{rev}} \cdot \lambda_{\mathrm{GRL}} \cdot \partial L_{\mathrm{cls}} / \partial(\mathrm{GRL\ output})$: the reversal coefficient and the loss weight act multiplicatively and remain distinct quantities."

**Issue:** "act multiplicatively" — standard phrasing in gradient-based training: "enter the gradient multiplicatively" or "scale the gradient multiplicatively." "Act multiplicatively" is an unusual predication for mathematical quantities.
**Suggested revision:** "the reversal coefficient and the loss weight scale the gradient multiplicatively and remain distinct quantities"
**Severity:** Very minor

---

## Summary Statistics

| Severity  | Count |
|-----------|-------|
| Moderate  | 3     |
| Minor     | 55    |
| Very minor| 9     |
| Pass (no issue) | 5 |
| **Total findings** | **67** |
| **Total sentences/clauses checked** | **~220** |

---

## Finding Categories

| Category | Count |
|----------|-------|
| Non-idiomatic method-description verb ("yields," "follows," "flag," "self-deactivate") | 10 |
| Non-standard architecture/component terminology ("learning units," "architectural pathway," "label-guided module," "bifurcation toward") | 7 |
| Comparison/performance description idiom ("rigorous metrics," "costs each family," "support") | 6 |
| Informal or vague quantifier/modifier ("compact formulation," "architectural prior") | 5 |
| Ambiguous/loose relative clause or participial structure | 8 |
| Non-standard anomaly detection domain noun combination ("reconstruction-context variation," "anomaly-discriminative search") | 5 |
| Training/loss description convention ("draws supervision," "matches to," "self-deactivate") | 7 |
| Comparison/evaluation idiom ("provide limited support," "costs," "precluding metric divergence") | 4 |
| Generic or clichéd conclusion language ("natural avenue for future work") | 1 |
| Technical accuracy (minor mismatch between claim and referenced concept) | 2 |
| Grammatical completeness (dangling/incomplete clause) | 2 |

---

## High-Priority Findings for Fixer

The following findings carry **Moderate** severity or involve potential reader misreading:

1. **B-012** (§1 contribution 3): "fails to mimic anomalous correlation patterns more severely than normal ones" — ambiguous comparative ordering; may be read as a claim about normal patterns rather than anomalous ones.
2. **B-046** (§4.1.1): "113 learning units" — "learning units" is not a recognized field term; reviewers will likely flag this. Replace with "entities."
3. **B-063** (§5 conclusion): "loss bifurcation toward normal-only Student mimicry" — "bifurcation toward" is logically inconsistent; bifurcation is a split, not a direction.

---

## Cross-Cutting Observation

The term **"loss bifurcation"** is used in Abstract, §1, §3.5, and §5. It is not an established term in the deep learning or anomaly detection literature. The intended concept is a **conditional loss** or **split training objective** (applying different loss terms to normal vs. anomalous patches). The term is used consistently within the manuscript (B-001, B-063), which limits the damage, but a reviewer unfamiliar with it may question its precision. The fixer should consider whether to standardize to "conditional loss decomposition" or explicitly define "loss bifurcation" on first use as a paper-specific term.
