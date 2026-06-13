---
phase: 6
agent: style-auditor-A
directives: [T6]
last_modified: 2026-06-11
scope: MANUSCRIPT_v2.md — Abstract, Highlights, §1–5, Appendix A–C (prose only; placeholders, math, table cells, figure captions, code comments excluded)
sentences_inspected: 214
---

# Style Audit A — Academic Naturalness (Sentence-Level)

Perspective: English prose quality — non-native syntax, awkward nominalisation chains, article/preposition errors, vague antecedents (this/it), overlong sentences (40+ words), passive-voice overuse, tense inconsistency, broken parallelism, unnatural inter-sentence transitions.

Severity scale:
- **MUST-FIX** — grammatical error, unambiguous antecedent failure, structural collapse, or clarity so impaired a reviewer is likely to stumble.
- **SHOULD** — naturalness or rhythm problem that a careful native-speaker author would revise; does not block comprehension but weakens prose quality.
- **OK-FLAG** — minor stylistic note; acceptable as-is but flagged for completeness.

---

## Abstract

---

**A-001**
Location: Abstract, sentence 1
Original: "Anomaly detection in multivariate time series is critical for industrial control systems, IT infrastructure monitoring, and spacecraft telemetry, yet most existing methods assume that training data are entirely normal — a condition rarely satisfied in practice."
Issue type: Overlong sentence / compound overload
Severity: SHOULD
Revised: "Anomaly detection in multivariate time series is critical for industrial control systems, IT infrastructure monitoring, and spacecraft telemetry. Yet most existing methods assume that training data are entirely normal — a condition rarely met in practice."
Rationale: At 42 words this is at the upper edge of readability. Splitting after "telemetry" and replacing "satisfied" with "met" (more natural in this collocation) improves rhythm without changing meaning.
Scientific meaning changed: No

---

**A-002**
Location: Abstract, sentence 2
Original: "In real deployments, a small fraction of training observations carry anomaly labels derived from recorded fault events, while the majority remain unlabeled; exploiting this structure has received limited attention."
Issue type: Pronoun antecedent ambiguity ("this structure")
Severity: SHOULD
Revised: "In real deployments, a small fraction of training observations carry anomaly labels derived from recorded fault events while the majority remain unlabeled; exploiting this label–normal coexistence structure has received limited attention."
Rationale: "This structure" is abstract and requires the reader to reconstruct what "structure" refers to from the preceding clause. A brief appositive ("this label–normal coexistence structure") disambiguates without adding sentence weight. Alternatively "this mixed-label structure."
Scientific meaning changed: No

---

**A-003**
Location: Abstract, sentence 3
Original: "We propose CSMAD, an end-to-end framework that integrates labeled anomaly information directly into masked autoencoder representation learning through three orthogonal mechanisms: anomaly-priority masking, loss bifurcation between normal and anomalous reconstruction paths, and a gradient reversal layer that adversarially suppresses anomaly-specific information from the Student's internal representation."
Issue type: Overlong sentence (54 words); list item asymmetry in the final colon expansion
Severity: SHOULD
Revised: "We propose CSMAD, an end-to-end framework that integrates labeled anomaly information directly into masked autoencoder representation learning through three orthogonal mechanisms: anomaly-priority masking, loss bifurcation between normal and anomalous reconstruction paths, and gradient-reversal suppression of anomaly-specific information from the Student's internal representation."
Rationale: The third list item ("a gradient reversal layer that adversarially suppresses…") breaks the noun-phrase parallelism of the first two items. Converting it to a parallel noun phrase ("gradient-reversal suppression of…") restores structural consistency and trims four words.
Scientific meaning changed: No

---

**A-004**
Location: Abstract, sentence 4
Original: "CSMAD employs an asymmetric Teacher–Student decoder architecture in which a capacity-limited Student's mimicry degrades preferentially on anomalous correlation patterns, amplifying the Teacher–Student discrepancy signal under contaminated training."
Issue type: Awkward possessive nominalisation ("a capacity-limited Student's mimicry degrades")
Severity: MUST-FIX
Revised: "CSMAD employs an asymmetric Teacher–Student decoder architecture in which the capacity-limited Student degrades preferentially on anomalous correlation patterns, amplifying the Teacher–Student discrepancy signal under contaminated training."
Rationale: "A capacity-limited Student's mimicry degrades" is an unnatural possessive construction. The agent of "degrades" should be "the Student" directly; "mimicry" is implied by the context of a Teacher–Student framework and its insertion here creates a clunky noun chain. The subject is also introduced with the indefinite article "a" inconsistently — the architecture has already been named ("CSMAD employs"), so a definite reference is appropriate.
Scientific meaning changed: No

---

**A-005**
Location: Abstract, sentence 5
Original: "To enable evaluation of labeled-anomaly-aware methods, we introduce a contaminated benchmark protocol that incorporates the chronological prefix of the test stream into training, exposing labeled anomalies absent in the original train splits of standard benchmarks."
Issue type: Participial phrase attachment ambiguity ("exposing labeled anomalies absent in the original train splits")
Severity: SHOULD
Revised: "To enable evaluation of labeled-anomaly-aware methods, we introduce a contaminated benchmark protocol that incorporates the chronological prefix of the test stream into training, thereby exposing labeled anomalies that are absent from the original training splits of standard benchmarks."
Rationale: The participial "exposing labeled anomalies absent in…" is compressed to the point where "absent" reads like an adjective modifying "anomalies" rather than a result of the protocol. "Thereby exposing" and "absent from" (standard preposition with "absent") clarify both the causal role and the collocation.
Scientific meaning changed: No

---

**A-006**
Location: Abstract, sentence 6 (placeholder sentence)
Original: "On [N] multivariate datasets spanning industrial and telemetry domains, CSMAD achieves competitive performance against [N] unsupervised and weakly supervised baselines under five rigorous evaluation metrics."
Issue type: Vague intensifier ("rigorous")
Severity: OK-FLAG
Revised: "On [N] multivariate datasets spanning industrial and telemetry domains, CSMAD achieves competitive performance against [N] unsupervised and weakly supervised baselines under five complementary evaluation metrics."
Rationale: "Rigorous" is a weak self-endorsement in this position; "complementary" (used correctly in §4.1.3) is more informative and consistent with the paper's own framing of the metric suite.
Scientific meaning changed: No

---

## Highlights

---

**H-001**
Location: Highlights, bullet 2
Original: "CSMAD combines masked autoencoding, asymmetric self-distillation, and gradient reversal to suppress anomaly representations."
Issue type: Vague final clause ("suppress anomaly representations")
Severity: SHOULD
Revised: "CSMAD combines masked autoencoding, asymmetric self-distillation, and gradient reversal to suppress anomaly-specific representations in the Student encoder."
Rationale: "Anomaly representations" is ambiguous — it could mean representations of anomalies or representations that are anomalous. "Anomaly-specific representations" is unambiguous. Adding "in the Student encoder" grounds the suppression mechanism, consistent with the paper's own framing in §3.5.
Scientific meaning changed: No

---

**H-002**
Location: Highlights, bullet 4
Original: "A contaminated benchmark protocol incorporates the test prefix into training, filling a gap in standard MTSAD benchmarks."
Issue type: Duplicate information (same as bullet 1 in different words); within bullet itself: weak verb phrase ("filling a gap")
Severity: OK-FLAG
Revised: "A contaminated benchmark protocol incorporates the test prefix into training, enabling evaluation of labeled-anomaly-aware methods on standard MTSAD datasets."
Rationale: "Filling a gap" is a generic phrase. Replacing with the functional consequence ("enabling evaluation of…") makes the highlight self-contained and actionable for readers scanning the list.
Scientific meaning changed: No

---

## §1 Introduction

---

**S1-001**
Location: §1, paragraph 1, sentence 1
Original: "Real-world cyber-physical systems continuously generate high-dimensional, multi-channel sensor streams — water treatment plants, server clusters, and spacecraft telemetry arrays all depend on reliable detection of anomalous states to prevent safety incidents and operational losses \cite{schmidl2022evaluation, blazquez2021review}."
Issue type: Overlong sentence (40 words); em-dash appositive disrupts subject–predicate flow
Severity: SHOULD
Revised: "Real-world cyber-physical systems continuously generate high-dimensional, multi-channel sensor streams, and reliable detection of anomalous states is critical to preventing safety incidents and operational losses in deployments ranging from water treatment plants to server clusters and spacecraft telemetry arrays \cite{schmidl2022evaluation, blazquez2021review}."
Rationale: The em-dash creates a dangling appositive where the examples ("water treatment plants, server clusters…") appear to modify "streams" rather than being examples of the broader claim. Reordering as a trailing prepositional phrase clarifies the scope.
Scientific meaning changed: No

---

**S1-002**
Location: §1, paragraph 1, sentence 2
Original: "Anomalies in such streams manifest not in isolated channels but through correlated deviations across multiple sensor dimensions \cite{deng2021gdn, wu2025catch}, and because exhaustive point-level annotation of anomalies is infeasible at scale, the dominant paradigm for multivariate time series anomaly detection (MTSAD) has been unsupervised learning \cite{wang2025nrdetector}."
Issue type: Overlong sentence (52 words); two logically distinct claims joined by "and because"; subject–predicate distance excessive
Severity: MUST-FIX
Revised: "Anomalies in such streams manifest not in isolated channels but through correlated deviations across multiple sensor dimensions \cite{deng2021gdn, wu2025catch}. Because exhaustive point-level annotation is infeasible at scale, the dominant paradigm for multivariate time series anomaly detection (MTSAD) has been unsupervised learning \cite{wang2025nrdetector}."
Rationale: Two distinct claims — the nature of multivariate anomalies and the unsupervised-paradigm dominance — are yoked by "and because," producing a 52-word sentence where the subject of the second clause ("the dominant paradigm") is 40 words from the clause's conjunction. Splitting into two sentences restores clarity.
Scientific meaning changed: No

---

**S1-003**
Location: §1, paragraph 2, sentence 1
Original: "The resulting body of work spans four broad families: reconstruction-based methods, which flag samples whose reconstruction errors exceed a threshold \cite{...}; prediction-based methods, which score deviations from forecast sensor readings \cite{deng2021gdn}; association-discrepancy and contrastive methods, which exploit the structural gap between normal and anomalous attention patterns \cite{...}; and methods that train general-purpose temporal backbones or auxiliary objectives for detection \cite{...}."
Issue type: Parallelism break — the fourth item ("methods that train…") is a relative clause introduced by "that", inconsistent with the pattern of the first three ("methods, which…"); also slightly informal label for the fourth family
Severity: SHOULD
Revised: "…; and backbone-based methods, which apply general-purpose temporal architectures or auxiliary pretraining objectives directly to detection \cite{...}."
Rationale: Items 1–3 use the pattern "[family label], which [action]". Item 4 breaks this with "methods that train…". Renaming the family and using the consistent relative clause pattern ("which apply…") restores parallelism. "Train general-purpose temporal backbones" is also less precise than "apply… architectures."
Scientific meaning changed: No

---

**S1-004**
Location: §1, paragraph 2, sentence 3
Original: "This assumption is structurally embedded: the methods have no architectural pathway for leveraging the information carried by labeled anomalies even when such labels are available — the best a label-aware variant can do is exclude confirmed anomaly windows from training, filtering contamination rather than learning from it \cite{wang2025nrdetector}."
Issue type: Overlong sentence (53 words); em-dash clause is a second complete thought that should be its own sentence
Severity: SHOULD
Revised: "This assumption is structurally embedded: these methods have no architectural pathway for leveraging labeled anomaly information even when such labels are available. The best a label-aware variant can do is exclude confirmed anomaly windows from training, filtering contamination rather than learning from it \cite{wang2025nrdetector}."
Rationale: The em-dash introduces a full independent clause ("the best… can do is…") that adds a new argument rather than elaborating the preceding one. Splitting produces two focused sentences and removes the 53-word run-on.
Scientific meaning changed: No

---

**S1-005**
Location: §1, paragraph 3, sentence 3
Original: "The gap is particularly acute in the standard MTSAD benchmarks we evaluate on, whose original training splits contain no labeled anomalies by construction (per-dataset label semantics in Appendix §A.3) — benchmark studies have independently criticized dataset and evaluation practices in this field \cite{liu2024elephant, schmidl2022evaluation}; evaluating any method that exploits labeled anomalies therefore requires modifying the data protocol, as detailed in Section 4.1.1."
Issue type: Overlong sentence (69 words); three structurally distinct claims fused with an em-dash and semicolon; "we evaluate on" ends in stranded preposition
Severity: MUST-FIX
Revised: "This gap is particularly acute in the standard MTSAD benchmarks evaluated here, whose original training splits contain no labeled anomalies by construction (per-dataset semantics in Appendix §A.3). Benchmark studies have independently criticized dataset and evaluation practices in this field \cite{liu2024elephant, schmidl2022evaluation}. Evaluating any method that exploits labeled anomalies therefore requires modifying the data partition, as detailed in Section 4.1.1."
Rationale: The original fuses three independent observations — the benchmark gap, external criticism of the field, and the protocol consequence — into a single sentence with 69 words. Splitting into three sentences, eliminating the stranded preposition ("benchmarks we evaluate on" → "benchmarks evaluated here"), and replacing the vague "data protocol" with "data partition" substantially improves clarity.
Scientific meaning changed: No

---

**S1-006**
Location: §1, paragraph 3, sentence 4
Original: "Our key observation is threefold: labeled anomalies reveal (a) which temporal positions yield informative hard reconstruction targets, (b) which patches the Student decoder should avoid mimicking, and (c) what representational content should be actively erased from the Student's encoding."
Issue type: Grammatical — "observation is threefold" followed by a colon and three clauses that are revelations (content of the observation), not separate observations; minor but the "is threefold:" construction is slightly non-native
Severity: OK-FLAG
Revised: "Our key observation is that labeled anomalies reveal three distinct learning signals: (a) which temporal positions yield informative hard reconstruction targets, (b) which patches the Student decoder should avoid mimicking, and (c) what representational content should be actively erased from the Student's encoding."
Rationale: "Our key observation is threefold:" uses a structure more common in non-native academic writing; native usage would state the observation and enumerate it. The revised form ("our key observation is that… three distinct learning signals") is idiomatic and also clarifies what the "observation" actually is.
Scientific meaning changed: No

---

**S1-007**
Location: §1, paragraph 3, sentence 6
Original: "Relying only on (b) is insufficient: a Student repeatedly exposed to anomalous patterns during training may learn to reconstruct them accurately through an indirect route, weakening the discrepancy signal at inference time; the active suppression of (c) closes this route at the representational level."
Issue type: Referential ambiguity — "(b)" and "(c)" are forward/backward references to lettered items in the preceding sentence; natural but requires the reader to maintain the list
Severity: OK-FLAG
Revised: "Relying only on (b) — loss bifurcation — is insufficient: a Student repeatedly exposed to anomalous patterns during training may learn to reconstruct them accurately through an indirect route, weakening the discrepancy signal at inference time; gradient-reversal suppression (c) closes this route at the representational level."
Rationale: Inline glossing of (b) and (c) aids flow when the reader encounters these references a sentence after the list; the parenthetical addition does not duplicate information but anchors the abstract labels.
Scientific meaning changed: No

---

**S1-008**
Location: §1, paragraph 4 (contributions), item 1
Original: "…in which labeled anomalies coexist with unlabeled training windows, and introduce a benchmark protocol that incorporates the chronological prefix of each dataset's test stream into training — constructing train splits with labeled anomalies absent from the original splits and evaluating on the held-out temporal suffix."
Issue type: Participial fragment after em-dash is loosely attached; overlong at 54 words for the contribution sentence
Severity: SHOULD
Revised: "…in which labeled anomalies coexist with unlabeled training windows, and introduce a benchmark protocol that incorporates the chronological prefix of each dataset's test stream into training. The protocol constructs training splits containing labeled anomalies absent from the original splits and evaluates on the held-out temporal suffix."
Rationale: The participial phrase after the em-dash ("constructing…and evaluating…") hangs off the main clause without a clear grammatical head. Converting to a follow-up sentence with "The protocol constructs…" makes the logical subject explicit.
Scientific meaning changed: No

---

**S1-009**
Location: §1, paragraph 4 (contributions), item 3
Original: "A deeper Teacher decoder (3 layers) establishes a stable normal-reconstruction reference, while a capacity-limited Student decoder (2 layers) fails to mimic anomalous correlation patterns more severely than normal ones — a design intended to make the Teacher–Student output discrepancy a reliable anomaly signal under contaminated training (quantified in Appendix B.5)."
Issue type: Ambiguous comparative ("fails to mimic anomalous correlation patterns more severely than normal ones") — "more severely" modifies the degree of failure, but the comparison reads as if "anomalous correlation patterns" fails more severely "than normal ones [are failed]"; should read as "fails more severely on anomalous…than on normal…"
Severity: MUST-FIX
Revised: "A deeper Teacher decoder (3 layers) establishes a stable normal-reconstruction reference, while a capacity-limited Student decoder (2 layers) fails to mimic anomalous correlation patterns more severely than it fails on normal ones — a design intended to make the Teacher–Student output discrepancy a reliable anomaly signal under contaminated training (quantified in Appendix B.5)."
Rationale: Without "it fails on," the comparison dangles: the reader cannot immediately tell whether "more severely than normal ones" means "more severely than [failing on] normal ones" or something else. The insertion of "it fails on" removes the ambiguity.
Scientific meaning changed: No

---

## §2 Related Work

---

**S2-001**
Location: §2.1, sentence 1
Original: "Deep learning approaches to unsupervised MTSAD have matured into several well-defined families."
Issue type: Vague temporal claim ("have matured") — present-perfect tense appropriate, but "matured into several well-defined families" is a generic opener common in AI survey writing
Severity: OK-FLAG
Revised: "Deep learning approaches to unsupervised MTSAD now span several well-defined families."
Rationale: "Have matured into… well-defined families" is a soft, evaluative claim that adds no concrete information. "Now span" is concise and specific.
Scientific meaning changed: No

---

**S2-002**
Location: §2.1, sentence 5
Original: "Despite this breadth, every family above treats the training data as predominantly or entirely normal."
Issue type: "above" as a reference to previously listed items is acceptable but faintly informal in a journal paper; "every family above" could be "each of these families"
Severity: OK-FLAG
Revised: "Despite this breadth, each of these families treats the training data as predominantly or entirely normal."
Rationale: "Every family above" is a forward-reference device borrowed from textbook style; "each of these families" is more natural in journal prose.
Scientific meaning changed: No

---

**S2-003**
Location: §2.1, sentence 6
Original: "When the training stream contains confirmed anomalous events — the contaminated setting that arises naturally from operational logs — these methods cannot distinguish known-anomalous from known-normal samples; labeled information is either discarded or treated as noise degrading the normal pattern \cite{wang2025nrdetector}."
Issue type: Semicolon junction of two clauses where the second ("labeled information is either discarded…") is a consequence, not a parallel coordinate; a full stop or "consequently" is preferred
Severity: SHOULD
Revised: "When the training stream contains confirmed anomalous events — the contaminated setting that arises naturally from operational logs — these methods cannot distinguish known-anomalous from known-normal samples; labeled information is consequently either discarded or treated as noise that degrades normal-pattern learning \cite{wang2025nrdetector}."
Rationale: Adding "consequently" makes the causal relationship explicit and prevents the semicolon from reading as a loose conjunction. "Noise degrading the normal pattern" is also a weak participial; "noise that degrades normal-pattern learning" is more precise.
Scientific meaning changed: No

---

**S2-004**
Location: §2.2, sentence 1
Original: "Positive and Unlabeled (PU) learning formalizes the scenario in which a learner has confirmed positive examples and a pool of unlabeled data that may contain additional positives \cite{bekker2020pusurvey,duplessis2014pu}, with established solution families spanning cost-sensitive risk minimization via non-negative risk estimators \cite{kiryo2017nnpu}, class-prior-based probability correction \cite{elkan2008pu}, and two-step techniques that first extract reliable negatives before training a classifier \cite{bekker2020pusurvey}."
Issue type: Overlong sentence (66 words); the trailing "with established solution families spanning…" is a heavy dangling participial that enlarges the sentence beyond manageable scope
Severity: MUST-FIX
Revised: "Positive and Unlabeled (PU) learning formalizes the scenario in which a learner has confirmed positive examples and a pool of unlabeled data that may contain additional positives \cite{bekker2020pusurvey,duplessis2014pu}. Established solution families include cost-sensitive risk minimization via non-negative risk estimators \cite{kiryo2017nnpu}, class-prior-based probability correction \cite{elkan2008pu}, and two-step techniques that first extract reliable negatives before training a classifier \cite{bekker2020pusurvey}."
Rationale: Splitting at the natural boundary ("…additional positives.") and converting the dangling participial into a proper sentence with subject "Established solution families include…" restores grammatical clarity and reduces the sentence to manageable lengths (26 and 34 words).
Scientific meaning changed: No

---

**S2-005**
Location: §2.2, sentence 4
Original: "A weakly supervised strand trains models to classify or rank windows from coarse segment-level annotations \cite{sultani2018deepmil,lee2021wetas,liu2024treemil}; the label is the sole learning signal, with no self-supervised reconstruction pretext."
Issue type: Semicolon clause "the label is the sole learning signal, with no self-supervised reconstruction pretext" has an awkward comma appositive; "with no" is slightly informal
Severity: OK-FLAG
Revised: "A weakly supervised strand trains models to classify or rank windows from coarse segment-level annotations \cite{sultani2018deepmil,lee2021wetas,liu2024treemil}; these approaches treat the label as the sole learning signal, without any self-supervised reconstruction pretext."
Rationale: "The label is the sole learning signal, with no…" is compressed to the point of feeling like a note rather than a sentence. "These approaches treat the label as…" gives the clause a subject that clearly refers back to "a weakly supervised strand" and makes "without" more formal than "with no."
Scientific meaning changed: No

---

**S2-006**
Location: §2.2, sentence 5
Original: "Our use of labels differs in kind: rather than serving as the target of a classification or ranking objective, the label shapes the gradient of a masked-reconstruction pretext, steering what the encoder itself learns to represent."
Issue type: Dangling participial in the colon elaboration — "rather than serving as the target" has an implied subject that should be "the label" but the sentence structure makes it read as if the authors are "serving as the target"
Severity: SHOULD
Revised: "Our use of labels differs in kind: rather than acting as the target of a classification or ranking objective, labels here shape the gradient of a masked-reconstruction pretext, steering what the encoder itself learns to represent."
Rationale: "Rather than serving as…, the label shapes…" — the subject of "serving" must match the subject of the main clause ("the label"). The sentence is grammatically correct as written, but "serving" applied to "the label" is slightly awkward; "acting as the target" is cleaner. Changing to plural "labels here shape" also improves rhythm.
Scientific meaning changed: No

---

**S2-007**
Location: §2.2, sentence 6
Original: "Two earlier semi-supervised models address label scarcity in multivariate time series: an autoregressive normality model with discriminative loss components that separate normal data from the few labeled anomalies \cite{xue2022fewpositive}, and a semi-supervised variational autoencoder coupled with an active-learning labeling loop \cite{huang2022slavae}."
Issue type: "address label scarcity" is slightly imprecise — both models address the semi-supervised setting, not specifically "label scarcity"; consistent with the context but worth flagging
Severity: OK-FLAG
Revised: "Two earlier semi-supervised models tackle the label-scarce multivariate time-series setting: an autoregressive normality model with discriminative loss components that separate normal data from the few labeled anomalies \cite{xue2022fewpositive}, and a semi-supervised variational autoencoder coupled with an active-learning labeling loop \cite{huang2022slavae}."
Rationale: "Tackle the label-scarce… setting" is more specific than "address label scarcity in multivariate time series" and avoids the dangling prepositional phrase.
Scientific meaning changed: No

---

**S2-008**
Location: §2.2, sentence 7
Original: "In both, labels act through loss terms attached to a generative or predictive normality objective; neither employs a masked-reconstruction self-distillation pretext, nor an adversarial, gradient-level suppression of anomaly information."
Issue type: "nor" after a semicolon with no preceding "neither" in the same clause — the "neither…nor" pair is split across the semicolon in a way that is grammatically marginal; also "adversarial, gradient-level suppression" has an unusual comma between two compound modifiers
Severity: SHOULD
Revised: "In both, labels act through loss terms attached to a generative or predictive normality objective; neither model employs a masked-reconstruction self-distillation pretext or adversarial gradient-level suppression of anomaly information."
Rationale: "Neither employs… nor an adversarial…" is a "neither…nor" construction where "nor" connects a full clause to a bare noun phrase, which is awkward. "Neither model employs… or…" with a single subject and two coordinated objects is standard.
Scientific meaning changed: No

---

**S2-009**
Location: §2.2, sentence 8
Original: "In the transfer setting, DACAD \cite{darban2024dacad} exploits labeled anomalies from a related source domain through supervised contrastive learning; in our setting, by contrast, the scarce labels reside in the target training stream itself."
Issue type: Transition ("in our setting, by contrast") — acceptable but slightly redundant; "by contrast" duplicates the contrast already conveyed by the structural parallelism
Severity: OK-FLAG
Revised: "In the transfer setting, DACAD \cite{darban2024dacad} exploits labeled anomalies from a related source domain through supervised contrastive learning; in our setting, the scarce labels reside in the target training stream itself."
Rationale: The contrast is already clear from the "In the transfer setting… In our setting…" structure; "by contrast" is redundant and slightly over-signals.
Scientific meaning changed: No

---

**S2-010**
Location: §2.2, sentence 9 (last sentence)
Original: "To our knowledge, CSMAD is the first end-to-end multivariate TSAD model that integrates labeled anomalies adversarially — through gradient reversal — into the gradient of a masked-reconstruction self-distillation objective."
Issue type: Inflated claim qualifier ("to our knowledge") is correct and necessary, but "the first… to integrate…" is an unusually strong claim in a highly active field; additionally, "into the gradient of" is a slightly unusual collocation — one integrates information "into a gradient" or "into an objective via the gradient"
Severity: SHOULD
Revised: "To our knowledge, CSMAD is the first end-to-end multivariate TSAD model to integrate labeled anomalies adversarially — through gradient reversal — into the representation learning of a masked-reconstruction self-distillation objective."
Rationale: "Into the gradient of… an objective" is physically imprecise (one modifies the gradient, not integrates into it). "Into the representation learning of…" better captures the mechanism described throughout the paper. The priority claim is retained as-is.
Scientific meaning changed: No (clarification only)

---

**S2-011**
Location: §2.3, sentence 1
Original: "The masked autoencoder (MAE) of He et al. \cite{he2022mae} showed that masking random patches and reconstructing the missing regions yields strong transferable representations."
Issue type: Tense — the paper uses present tense as the convention for established findings, but "showed" is past tense here; inconsistency with the section's prevailing tense
Severity: SHOULD
Revised: "The masked autoencoder (MAE) of He et al. \cite{he2022mae} demonstrates that masking random patches and reconstructing the missing regions yields strong transferable representations."
Rationale: Academic convention in this field is to use simple present when describing the findings of cited work ("X shows that…"). The isolated past tense "showed" is inconsistent with the surrounding prose.
Scientific meaning changed: No

---

**S2-012**
Location: §2.3, sentence 2
Original: "Our patch-based masking draws directly from this paradigm, adapted from the spatial domain to windows of multivariate sensor channels; similar masking-based reconstruction objectives in some time-series models \cite{fang2024tfmae} are independent developments — our design lineage traces to vision MAE."
Issue type: "adapted from the spatial domain to windows" is a slightly awkward elliptical participial attached to "masking" (rather than "Our… approach"); em-dash clause is an abrupt restatement of what was just said
Severity: SHOULD
Revised: "Our patch-based masking draws directly from this paradigm, adapting the spatial-domain approach to windows of multivariate sensor channels; similar masking-based reconstruction objectives in some time-series models \cite{fang2024tfmae} constitute independent developments, and our design lineage traces to vision MAE."
Rationale: "Draws…, adapted from" has the participial modifying "masking," which works but is somewhat compressed. Converting to "adapting the spatial-domain approach…" makes the subject ("our masking") the explicit agent of the adaptation. The em-dash before "our design lineage" is abrupt; coordinating with "and" smooths the flow.
Scientific meaning changed: No

---

**S2-013**
Location: §2.3, sentence 4
Original: "Ristea et al. \cite{ristea2024sdmae} adapted this design to video anomaly detection, pairing a capacity-limited student decoder with a deeper teacher inside a masked autoencoder and using their output discrepancy as the anomaly score at inference."
Issue type: Tense — "adapted" is past tense; inconsistent with the present-tense convention for cited work (cf. S2-011)
Severity: SHOULD
Revised: "Ristea et al. \cite{ristea2024sdmae} adapt this design to video anomaly detection, pairing a capacity-limited student decoder with a deeper teacher inside a masked autoencoder and using their output discrepancy as the anomaly score at inference."
Rationale: Same tense-consistency issue as S2-011.
Scientific meaning changed: No

---

**S2-014**
Location: §2.3, footnote [sd-fn], sentence 1
Original: "The self-distillation terminology follows Zhang et al. \cite{zhang2022selfdistill} and Ristea et al. \cite{ristea2024sdmae}."
Issue type: Acceptable but "follows" is slightly ambiguous — does the paper "follow" the terminology (adopt it) or "follow" the cited works (cite them)? In context it means "adopts the terminology introduced in"; a minor clarification improves precision.
Severity: OK-FLAG
Revised: "The self-distillation terminology is adopted from Zhang et al. \cite{zhang2022selfdistill} and Ristea et al. \cite{ristea2024sdmae}."
Rationale: "Follows" is commonly used this way in mathematics but is marginally ambiguous in prose; "is adopted from" removes the ambiguity.
Scientific meaning changed: No

---

**S2-015**
Location: §2.3, footnote [sd-fn], sentence 3
Original: "The gradient reversal layer that adversarially suppresses anomaly information in the Student is absent from the video setting of \cite{ristea2024sdmae}, which trains without real labeled anomalies; the distinction between operating in the target/loss space and in the gradient space of the representation is elaborated in Section 3.5."
Issue type: Overlong sentence (52 words); "the video setting of \cite{ristea2024sdmae}" is an unusual reference format (the citation appears mid-phrase without the author name)
Severity: SHOULD
Revised: "The gradient reversal layer that adversarially suppresses anomaly information in the Student is absent from Ristea et al. \cite{ristea2024sdmae}, who train without real labeled anomalies. The distinction between operating in the target/loss space and in the gradient space of the representation is elaborated in Section 3.5."
Rationale: Splitting into two sentences improves readability. Replacing "the video setting of \cite{...}" with "Ristea et al. \cite{...}" is standard academic citation practice.
Scientific meaning changed: No

---

## §3 Methodology

---

**S3-001**
Location: §3.1, sentence 1
Original: "A multivariate time series $\mathbf{X} \in \mathbb{R}^{T \times F}$ ($T$ timesteps, $F$ sensor channels) yields sliding windows $\mathbf{W} \in \mathbb{R}^{L \times F}$ of length $L$, each partitioned into $N$ non-overlapping patches $\mathbf{P}_i \in \mathbb{R}^{s \times F}$ of size $s$ ($L = N \cdot s$)."
Issue type: "yields sliding windows" — the time series does not passively yield windows; the analyst applies a sliding window procedure; also passive nominalisation
Severity: OK-FLAG
Revised: "A multivariate time series $\mathbf{X} \in \mathbb{R}^{T \times F}$ ($T$ timesteps, $F$ sensor channels) is segmented into sliding windows $\mathbf{W} \in \mathbb{R}^{L \times F}$ of length $L$, each partitioned into $N$ non-overlapping patches $\mathbf{P}_i \in \mathbb{R}^{s \times F}$ of size $s$ ($L = N \cdot s$)."
Rationale: "Yields" anthropomorphises the time series; "is segmented into" is the standard formulation in this literature.
Scientific meaning changed: No

---

**S3-002**
Location: §3.1, sentence 5 (large paragraph)
Original: "In practice, labeled anomaly events arise naturally from the operational logs of industrial systems — fault and attack records that document anomalies as correlated deviations across multiple sensor channels — making the recovery of multi-channel correlation structure the central learning challenge."
Issue type: The em-dash appositive ("fault and attack records that document anomalies as correlated deviations across multiple sensor channels") interrupts the flow; "making the recovery of multi-channel correlation structure the central learning challenge" is a participial that could stand alone
Severity: SHOULD
Revised: "In practice, labeled anomaly events arise naturally from the operational logs of industrial systems — fault and attack records that document anomalies as correlated deviations across multiple sensor channels. Recovering the multi-channel correlation structure therefore constitutes the central learning challenge."
Rationale: Splitting after the em-dash appositive allows the participial ("making the recovery…") to become a proper sentence with a clear subject, avoiding the grammatically marginal "making… the challenge" construction.
Scientific meaning changed: No

---

**S3-003**
Location: §3.2, sentence 3
Original: "The Student and GRL branches read the encoder output through a stop-gradient, so the encoder is optimized exclusively by the Teacher's reconstruction objective and the adversarial signal cannot corrupt the normal-pattern representation underpinning the anomaly score."
Issue type: Overlong sentence (43 words) fusing two distinct architectural properties with "and"; the causal connector "so" carries too much logical weight here
Severity: SHOULD
Revised: "The Student and GRL branches read the encoder output through a stop-gradient. Consequently, the encoder is optimized exclusively by the Teacher's reconstruction objective, and the adversarial signal cannot corrupt the normal-pattern representation that underpins the anomaly score."
Rationale: Splitting at the "so" boundary and using "Consequently" makes the causal logic explicit. Replacing "underpinning" with "that underpins" removes a nominative participle.
Scientific meaning changed: No

---

**S3-004**
Location: §3.3 (Anomaly-priority masking), sentence 1
Original: "A fraction of the patches — $|M| = \mathrm{round}(N \times \rho)$, with masking ratio $\rho$ — is withheld from the encoder, which processes only the $|V| = N - |M|$ visible tokens."
Issue type: Em-dash insertion of the formula creates a subject-verb separation of 15 tokens before "is withheld"; awkward reading
Severity: OK-FLAG
Revised: "With masking ratio $\rho$, a fraction $|M| = \mathrm{round}(N \times \rho)$ of the patches is withheld from the encoder, which processes only the $|V| = N - |M|$ visible tokens."
Rationale: Moving the definitional material to a fronted prepositional phrase allows "a fraction… of the patches is withheld" to be read as a clean subject–verb pair.
Scientific meaning changed: No

---

**S3-005**
Location: §3.3 (Anomaly-priority masking), sentence 4
Original: "This addresses a structural imbalance of contaminated training: anomalous patches are rare, so stochastic masking seldom selects them and the model learns to reconstruct *around* rather than *through* them."
Issue type: "This addresses" — vague demonstrative; the reader must parse that "this" is the anomaly-priority masking mechanism introduced in the preceding sentence. Acceptable in context but can be made more precise.
Severity: OK-FLAG
Revised: "Anomaly-priority masking addresses a structural imbalance of contaminated training: anomalous patches are rare, so stochastic masking seldom selects them, and the model would otherwise learn to reconstruct *around* rather than *through* them."
Rationale: Making the subject explicit ("Anomaly-priority masking") removes the vague "This." Adding "would otherwise" also makes the logical role of the mechanism (prevention) clearer.
Scientific meaning changed: No

---

**S3-006**
Location: §3.4 (Teacher decoder), sentence 1
Original: "Learnable mask tokens are inserted at positions in $M$, position embeddings added, and the full sequence passed through a self-attention-only decoder of depth $n_{\mathrm{T}}$ following the standard MAE design \cite{he2022mae}; a linear head projects hidden states $\{h^{\mathrm{T}}_i\}$ to reconstructions $\{o^{\mathrm{T}}_i\}$."
Issue type: Three passive participials ("are inserted," "added," "[are] passed") elided in asyndeton — the omission of "are" before "position embeddings added" and "the full sequence passed" is technically a grammatical error (dangling past participle without auxiliary)
Severity: MUST-FIX
Revised: "Learnable mask tokens are inserted at positions in $M$, position embeddings are added, and the full sequence is passed through a self-attention-only decoder of depth $n_{\mathrm{T}}$ following the standard MAE design \cite{he2022mae}; a linear head projects hidden states $\{h^{\mathrm{T}}_i\}$ to reconstructions $\{o^{\mathrm{T}}_i\}$."
Rationale: "position embeddings added" and "the full sequence passed" omit the auxiliary "are"/"is," making them dangling participles. Restoring the auxiliaries fixes the grammar.
Scientific meaning changed: No

---

**S3-007**
Location: §3.4 (Why the capacity gap matters), sentence 1
Original: "A deeper Teacher faithfully learns the joint normal correlation structure; the shallower Student replicates it on recurring normal patterns but fails more severely on the atypical correlation patterns characterizing anomalies than a matched-capacity decoder would (quantified in Appendix B.5), so the output discrepancy carries a stronger anomaly signal than reconstruction error alone."
Issue type: Overlong sentence (55 words); "characterizing anomalies" is a loose participial (should be "that characterize anomalies"); comparative structure is complex
Severity: SHOULD
Revised: "A deeper Teacher faithfully learns the joint normal correlation structure; the shallower Student replicates it on recurring normal patterns but fails more severely on the atypical correlation patterns that characterize anomalies than a matched-capacity decoder would (quantified in Appendix B.5). Consequently, the output discrepancy carries a stronger anomaly signal than reconstruction error alone."
Rationale: Splitting at the semicolon and converting the participial "characterizing" to a relative clause "that characterize" improves grammatical precision and reduces per-sentence complexity.
Scientific meaning changed: No

---

**S3-008**
Location: §3.4 (GRL dual-λ structure), sentence 1
Original: "Two independent quantities govern the gradient reversal branch once the Student is active: the **loss weight** $\lambda_{\mathrm{GRL}}$, set adaptively from the clamped ratio of main-loss to GRL-loss gradient norms — computed per batch and applied as the previous epoch's average — so the adversarial loss neither dominates nor vanishes; and the **reversal coefficient** $\lambda_{\mathrm{rev}}$, scaling the backward gradient through the GRL on the sigmoid schedule of \citet{ganin2016dann}, growing from $\approx 0.02$ to $\approx 1$ over the student-training phase so suppression strengthens without destabilizing early Student learning (exact rules in Appendix §C.1)."
Issue type: Overlong sentence (101 words); extreme sentence complexity with nested em-dashes, semicolons, and parentheses; the two list items have different grammatical forms (the first uses a past-participle clause "set adaptively," the second a present-participle "scaling")
Severity: MUST-FIX
Revised: "Two independent quantities govern the gradient reversal branch once the Student is active. The **loss weight** $\lambda_{\mathrm{GRL}}$ is set adaptively from the clamped ratio of main-loss to GRL-loss gradient norms, computed per batch and applied as the previous epoch's average, so the adversarial loss neither dominates nor vanishes. The **reversal coefficient** $\lambda_{\mathrm{rev}}$ scales the backward gradient through the GRL on the sigmoid schedule of \citet{ganin2016dann}, growing from $\approx 0.02$ to $\approx 1$ over the student-training phase so that suppression strengthens without destabilizing early Student learning (exact rules in Appendix §C.1)."
Rationale: A 101-word sentence is unworkable. Decomposing the list into three sentences (introduction + one sentence per quantity) preserves all information while making the structure legible. The passive-voice forms are regularized to active equivalents.
Scientific meaning changed: No

---

**S3-009**
Location: §3.5 (Output discrepancy loss), sentence 2
Original: "Anomalous patches are excluded entirely: the Student is steered to agree with the Teacher on normal patterns while remaining free to deviate at anomaly locations."
Issue type: "are excluded entirely" — passive with "entirely" as a slightly redundant emphasizer (the colon clarification already specifies the scope)
Severity: OK-FLAG
Revised: "Anomalous patches are excluded from $L_{\mathrm{OD}}$: the Student is steered to agree with the Teacher on normal patterns while remaining free to deviate at anomaly locations."
Rationale: "Entirely" adds no information beyond what the colon elaboration already states; removing it tightens the sentence. Adding "from $L_{\mathrm{OD}}$" makes the exclusion scope explicit (it was implied but not stated).
Scientific meaning changed: No

---

**S3-010**
Location: §3.5 (GRL anomaly suppression loss), sentence 1 (long)
Original: "A two-layer MLP head $g_\phi$ predicts from each masked patch's Student hidden state whether the enclosing window contains an anomaly ($y^w$ broadcast to all masked patches; strictly, the target indicates an anomaly within the masked region, which coincides with $y^w$ under anomaly-priority masking), trained with a focal-style BCE variant for severe class imbalance: unlike the standard focal loss \cite{lin2017focal}, whose modulating factor derives from the raw prediction, here it derives from the class-prior-weighted cross-entropy itself (exact form: Eq. C.3)."
Issue type: Overlong sentence (90 words); a parenthetical correction and a contrastive clause are nested inside the main sentence, creating two layers of interruption; "trained with" is a participial with an ambiguous attachment (it reads as if "$y^w$" is trained, not "$g_\phi$")
Severity: MUST-FIX
Revised: "A two-layer MLP head $g_\phi$ is trained with a focal-style BCE variant to predict from each masked patch's Student hidden state whether the enclosing window contains an anomaly ($y^w$ broadcast to all masked patches; strictly, the target indicates an anomaly within the masked region, which coincides with $y^w$ under anomaly-priority masking). Unlike the standard focal loss \cite{lin2017focal}, whose modulating factor derives from the raw prediction, the modulating factor here derives from the class-prior-weighted cross-entropy itself (exact form: Eq. C.3)."
Rationale: The 90-word sentence is unworkable. Splitting into two and reordering "is trained with… to predict" fixes the participial attachment of "trained with." The contrastive "unlike" clause works better as a standalone sentence.
Scientific meaning changed: No

---

**S3-011**
Location: §3.5 (GRL anomaly suppression loss), sentence 2
Original: "The gradient reversal layer \cite{ganin2016dann} between head and Student hidden states is an identity map in the forward pass and negates the gradient in the backward pass, scaled by $\lambda_{\mathrm{rev}}$; the resulting adversarial gradient — proportional to $-\lambda_{\mathrm{rev}} \cdot \lambda_{\mathrm{GRL}}$ — opposes the classifier's search for anomaly-discriminative features, pushing the Student toward anomaly-*invariant* internal states."
Issue type: "scaled by $\lambda_{\mathrm{rev}}$" is a loose participial that grammatically modifies "the backward pass" rather than the negation; the antecedent is ambiguous
Severity: SHOULD
Revised: "The gradient reversal layer \cite{ganin2016dann} between head and Student hidden states is an identity map in the forward pass and scales-and-negates the gradient by $\lambda_{\mathrm{rev}}$ in the backward pass; the resulting adversarial gradient — proportional to $-\lambda_{\mathrm{rev}} \cdot \lambda_{\mathrm{GRL}}$ — opposes the classifier's search for anomaly-discriminative features, pushing the Student toward anomaly-*invariant* internal states."
Rationale: "Negates… scaled by" is ambiguous; "scales-and-negates… by" makes the compound operation explicit. Alternatively: "multiplies the gradient by $-\lambda_{\mathrm{rev}}$."
Scientific meaning changed: No

---

**S3-012**
Location: §3.5 (Why gradient reversal is necessary), sentence 1
Original: "Excluding anomalous patches from $L_{\mathrm{OD}}$ removes the demand that the Student *follow* the Teacher there, but it does not actively remove anomaly information from the Student's representation."
Issue type: "there" is a vague spatial reference — refers to "at anomaly patches" but is too casual for an academic methods section
Severity: SHOULD
Revised: "Excluding anomalous patches from $L_{\mathrm{OD}}$ removes the demand that the Student *follow* the Teacher at those locations, but it does not actively remove anomaly information from the Student's representation."
Rationale: "At those locations" is more precise and formal than "there."
Scientific meaning changed: No

---

**S3-013**
Location: §3.6 (Leave-one-out masking), sentence 1
Original: "Each test window is scored under $N$ masking patterns — each patch masked alone, all patterns forwarded in parallel through the batch dimension — eliminating cross-patch interference at an inference cost of approximately $N$ single-window forward passes, an acknowledged limitation."
Issue type: The em-dash phrase is a compressed parenthetical that requires unpacking; "an acknowledged limitation" hangs off the sentence as an appositive but lacks a clear referent (is the limitation the cross-patch interference elimination or the inference cost?)
Severity: SHOULD
Revised: "Each test window is scored under $N$ masking patterns — each masking a single patch in isolation, with all $N$ patterns forwarded in parallel through the batch dimension — eliminating cross-patch interference at an inference cost of approximately $N$ single-window forward passes; this cost is an acknowledged limitation."
Rationale: "Each patch masked alone" is a compressed participial; "each masking a single patch in isolation" is clearer. Separating "this cost is an acknowledged limitation" with a semicolon clarifies what the limitation is.
Scientific meaning changed: No

---

**S3-014**
Location: §3.6 (Patch-level anomaly score), sentence 3
Original: "where $\bar{r}$ and $\bar{d}$ are the means of the patch-level reconstruction errors and discrepancies over all (window, patch) pairs of the evaluated series, computed once per entity; the two components are then combined at a fixed ratio:"
Issue type: This sentence, following a display equation, begins mid-flow with "where" — acceptable in math notation but the second clause ("the two components are then combined…") is awkwardly appended with a semicolon before the next display equation
Severity: OK-FLAG
Revised: "Here, $\bar{r}$ and $\bar{d}$ are the means of the patch-level reconstruction errors and discrepancies, respectively, over all (window, patch) pairs of the evaluated series, computed once per entity. The two components are then combined at a fixed ratio:"
Rationale: Beginning with "Here," rather than "where" is more natural in running prose. Adding "respectively" clarifies which mean corresponds to which quantity. Splitting into two sentences before the next display equation is cleaner.
Scientific meaning changed: No

---

**S3-015**
Location: §3.6 (Point-level aggregation), sentence 2
Original: "Averaging across overlapping windows provides an ensemble effect that reduces single-window reconstruction-context variation."
Issue type: "single-window reconstruction-context variation" is an unusual compound noun — awkward nominalisation
Severity: SHOULD
Revised: "Averaging across overlapping windows provides an ensemble effect that reduces variance from single-window reconstruction context."
Rationale: "Single-window reconstruction-context variation" is a clunky triple compound noun. "Variance from single-window reconstruction context" is more readable and natural.
Scientific meaning changed: No

---

## §4 Experiments

---

**S4-001**
Location: §4.1.1, sentence 1 (Datasets paragraph)
Original: "We evaluate CSMAD on six real-world multivariate benchmark families — SWaT \cite{goh2016swat}, WaDi \cite{ahmed2017wadi}, PSM \cite{abdulaal2021psm}, SMD \cite{su2019omnianomaly}, and SMAP/MSL \cite{hundman2018telemanom} — spanning industrial control, IT infrastructure, and spacecraft telemetry: 113 learning units in total, or 114 evaluation units with SWaT's dual evaluation (below)."
Issue type: "six real-world multivariate benchmark families" — only five names are listed (SWaT, WaDi, PSM, SMD, SMAP/MSL); the sixth is implicitly WaDi A1+A2 counted separately; the list "(below)" is an informal forward reference
Severity: MUST-FIX
Revised: "We evaluate CSMAD on six real-world multivariate benchmark families — SWaT \cite{goh2016swat}, WaDi A1 and A2 \cite{ahmed2017wadi}, PSM \cite{abdulaal2021psm}, SMD \cite{su2019omnianomaly}, and SMAP/MSL \cite{hundman2018telemanom} — spanning industrial control, IT infrastructure, and spacecraft telemetry: 113 learning units in total, or 114 evaluation units with SWaT's dual evaluation (detailed below)."
Rationale: The discrepancy between "six families" and five listed names will confuse readers. Making WaDi A1 and A2 explicit as separate families resolves the count to six. "Detailed below" is more formal than "(below)."
Scientific meaning changed: No — only disambiguation

---

**S4-002**
Location: §4.1.1 (Contaminated benchmark protocol), sentence 3
Original: "We therefore re-split each dataset at the temporal midpoint of its original test file: the earlier half joins the training data and the later half is reserved exclusively for evaluation, so labeled anomalies are genuinely present in training (ratios 0.52\%–6.20\%; SMD per-machine pending; Table 1)."
Issue type: The parenthetical "(ratios 0.52\%–6.20\%; SMD per-machine pending; Table 1)" is a dense inline note within a sentence that is already making a key methodological claim; the semicolons inside the parenthetical make the sentence parsing difficult
Severity: SHOULD
Revised: "We therefore re-split each dataset at the temporal midpoint of its original test file: the earlier half joins the training data and the later half is reserved exclusively for evaluation, so labeled anomalies are genuinely present in training. Training anomaly ratios range from 0.52\% to 6.20\% (SMD per-machine values pending; Table 1)."
Rationale: Extracting the numeric range and the table pointer into a separate sentence eliminates the nested semicolons and makes the quantitative claim a standalone statement.
Scientific meaning changed: No

---

**S4-003**
Location: §4.1.1 (Contaminated benchmark protocol), sentence 5
Original: "The halving rule is uniform; only for SMAP/MSL does the split point shift outward when it falls within ten timesteps of an anomaly region (4 of 81 channels, largest shift 166 timesteps; Appendix §A.3)."
Issue type: "shift outward when it falls within ten timesteps" — "outward" is directionally imprecise (the shift direction is dataset-specific); the parenthetical crams three distinct facts
Severity: SHOULD
Revised: "The halving rule is uniform; only for SMAP/MSL does the split point shift when it falls within ten timesteps of an annotated anomaly region (4 of 81 channels are affected, with the largest shift being 166 timesteps; Appendix §A.3)."
Rationale: "Shift outward" implies a specific direction that is not consistently true (Table A.5 shows both positive and negative shifts). Removing "outward" and expanding "4 of 81 channels, largest shift 166 timesteps" into a grammatically complete parenthetical improves clarity.
Scientific meaning changed: No

---

**S4-004**
Location: §4.1.1 (Contaminated benchmark protocol), sentence 6
Original: "The re-split is a redefinition of the benchmark, not a use of held-out labels: no model sees evaluation labels at any stage, and the identical partition is provided to all methods (Section 4.1.4)."
Issue type: "not a use of held-out labels" — the intended meaning is "not a form of label leakage," but "a use of held-out labels" is an unusual noun phrase
Severity: SHOULD
Revised: "The re-split redefines the benchmark partition rather than exploiting held-out labels: no model sees evaluation labels at any stage, and the identical partition is applied to all methods (Section 4.1.4)."
Rationale: "Is a redefinition of… not a use of…" is an awkward copular pair; "redefines… rather than exploiting" is more natural. "Provided to all methods" is slightly informal; "applied to all methods" is consistent with the paper's usage elsewhere.
Scientific meaning changed: No

---

**S4-005**
Location: §4.1.1 (SWaT dual evaluation), sentence 2
Original: "SWaT is trained once but evaluated twice: in the full condition a single attack event (region 22) accounts for 83.75\% of test anomaly mass, so the full metric chiefly reflects detection of this one event."
Issue type: The two clauses joined by "so" are logically a condition–consequence pair ("because region 22 accounts for 83.75%, the full metric reflects only this one event"), but the sentence currently reads as a factual sequence rather than a motivation
Severity: SHOULD
Revised: "SWaT is trained once but evaluated twice: in the full condition, a single attack event (region 22) accounts for 83.75\% of test anomaly mass, meaning the full metric chiefly reflects detection of this one event."
Rationale: Replacing "so" with "meaning" makes the explanatory relationship explicit and converts the second clause from a temporal/causal into a logical consequence. A comma after "full condition" is also added for clarity.
Scientific meaning changed: No

---

**S4-006**
Location: §4.1.2 (Architecture and training), sentence 2
Original: "We report no cross-seed variance or confidence intervals — a limitation of the current evaluation; only the random-score baseline is averaged over five runs (Appendix §A.1)."
Issue type: The em-dash appositive ("a limitation of the current evaluation") is placed between the main disclosure and its exception; the exception ("only the random-score baseline…") appears after the limiting clause, breaking the logical flow
Severity: SHOULD
Revised: "We report no cross-seed variance or confidence intervals for the main results — a limitation of the current evaluation — except for the random-score baseline, which is averaged over five independent runs (Appendix §A.1)."
Rationale: Bracketing the limitation with em-dashes and adding "except for" makes the scope of the limitation and its exception clear in a single linear reading.
Scientific meaning changed: No

---

**S4-007**
Location: §4.1.2 (Epoch asymmetry disclosure), sentence 1
Original: "Unsupervised baselines train 10 epochs and weakly supervised baselines 50, evaluated every epoch; CSMAD trains 500, evaluated every 5 (budget table in Appendix §A.1)."
Issue type: Elliptical construction ("weakly supervised baselines 50") omits "train" — acceptable in informal writing but not in journal prose; the sentence is a headline-style list
Severity: SHOULD
Revised: "Unsupervised baselines train for 10 epochs and weakly supervised baselines for 50 epochs, both evaluated every epoch; CSMAD trains for 500 epochs, evaluated every 5 (full budget table in Appendix §A.1)."
Rationale: Adding "for" and the repeated "epochs" restores the parallel structure across all three cases and removes the ellipsis. "Full budget table" is marginally clearer than "budget table."
Scientific meaning changed: No

---

**S4-008**
Location: §4.1.2 (Epoch asymmetry disclosure), sentence 2
Original: "All methods share the selection criterion, no early stopping, and best-epoch reporting; budgets reflect convergence characteristics — CSMAD needs the 250-epoch warmup before the Student activates — and baseline batch sizes follow each method's original implementation preset (Table A.3)."
Issue type: Three heterogeneous items share a single semicoloned sentence — convergence budgets, a specific CSMAD warmup fact, and baseline batch size policy; the em-dash interruption makes this 46-word sentence difficult to parse
Severity: SHOULD
Revised: "All methods share the same selection criterion, run to their full budget without early stopping, and are reported at their best evaluated epoch. Budget differences reflect convergence characteristics: CSMAD requires the 250-epoch warmup before the Student activates. Baseline batch sizes follow each method's original implementation preset (Table A.3)."
Rationale: Three separate statements are split into three sentences. "No early stopping" and "best-epoch reporting" are converted from bare noun phrases into verbal clauses, which is grammatically complete. The CSMAD-specific warmup fact stands alone for visibility.
Scientific meaning changed: No

---

**S4-009**
Location: §4.1.2 (Test-set model selection), sentence 2
Original: "Uniform across methods, this leaves relative rankings unaffected but may bias absolute estimates optimistically; we acknowledge this limitation."
Issue type: "Uniform across methods, this" — "this" is ambiguous; it could refer to the selection criterion or to the condition of having no validation split; fronted appositive "Uniform across methods" is only loosely attached
Severity: SHOULD
Revised: "Because the criterion is applied uniformly to all methods, relative rankings are unaffected; however, absolute metric estimates may be optimistically biased, and we acknowledge this limitation."
Rationale: Replacing the fronted adjectival "Uniform across methods, this…" with an explicit causal clause removes the ambiguous "this" and makes the logic explicit.
Scientific meaning changed: No

---

**S4-010**
Location: §4.1.2 (Inference and threshold), sentence 2
Original: "Threshold-dependent metrics use the anomaly-ratio threshold — the $(1-\alpha)$ quantile of the score distribution, $\alpha$ being the anomaly fraction of the evaluation set — applied identically to all methods, following the anomaly-ratio thresholding mechanism introduced by \cite{xu2022anomalytransformer} (which sets a fixed ratio on a validation split, whereas our $\alpha$ is the measured fraction of the evaluation span); $\alpha$ derives from evaluation-set ground truth but is never used in training, and threshold-free metrics (VUS, PA\%K-AUC families) are unaffected."
Issue type: Overlong sentence (88 words); two nested em-dash and parenthetical interruptions make the primary claim ("applies threshold identically") nearly unreadable
Severity: MUST-FIX
Revised: "Threshold-dependent metrics use the anomaly-ratio threshold: the $(1-\alpha)$ quantile of the score distribution, where $\alpha$ is the measured anomaly fraction of the evaluation span. This threshold is applied identically to all methods; it is conceptually related to the mechanism of \citet{xu2022anomalytransformer}, which sets a fixed ratio on a validation split, but $\alpha$ here derives from evaluation-set ground truth and is never used in training. Threshold-free metrics (VUS, PA\%K-AUC families) are unaffected."
Rationale: Breaking the 88-word sentence into three sentences eliminates the nested interruptions, makes the primary claim legible, and separates the definitional, comparative, and scope statements.
Scientific meaning changed: No

---

**S4-011**
Location: §4.1.3, sentence 1
Original: "We adopt five metrics assessing complementary aspects of detection quality, following the multi-metric philosophy of recent benchmark analyses \cite{kim2022rigorous, paparrizos2022vus, liu2024elephant, wang2025nrdetector}: **PA\%K-AUC F1**, which integrates the point-adjusted F1 of the PA\%K protocol \cite{kim2022rigorous} over the tolerance spectrum $K \in \{0, 1, \ldots, 100\}$, removing dependence on any particular $K$ — our primary metric and selection criterion; **PA\%K-AUC AUC-PR**, the same $K$-integration applied to the area under the precision–recall curve at each $K$ (obtained by a threshold sweep, hence threshold-free); **VUS-PR** and **VUS-ROC** \cite{paparrizos2022vus}, which sweep both a threshold and a temporal tolerance to measure ranking quality without an operating point, VUS-PR rated the most reliable single TSAD measure by a large-scale study \cite{liu2024elephant}; and **Affiliation F1** \cite{huet2022affiliation}, the harmonic mean of affiliation precision/recall measuring the temporal distance between predicted and ground-truth events, computed at the anomaly-ratio threshold (the F1-optimal-threshold variant is excluded from all rankings)."
Issue type: Overlong sentence (155 words) listing all five metrics in a single colon-enumerated structure with nested subclauses; unmanageable for reviewers
Severity: MUST-FIX
Revised: "We adopt five metrics that assess complementary aspects of detection quality, following the multi-metric philosophy of recent benchmark analyses \cite{kim2022rigorous, paparrizos2022vus, liu2024elephant, wang2025nrdetector}. **PA\%K-AUC F1** integrates the point-adjusted F1 of the PA\%K protocol \cite{kim2022rigorous} over the tolerance spectrum $K \in \{0, 1, \ldots, 100\}$, removing dependence on any particular $K$; it is our primary metric and selection criterion. **PA\%K-AUC AUC-PR** applies the same $K$-integration to the area under the precision–recall curve at each $K$ (obtained by a threshold sweep, hence threshold-free). **VUS-PR** and **VUS-ROC** \cite{paparrizos2022vus} sweep both a decision threshold and a temporal tolerance to measure ranking quality without a fixed operating point; VUS-PR is rated the most reliable single TSAD measure by a large-scale study \cite{liu2024elephant}. **Affiliation F1** \cite{huet2022affiliation} is the harmonic mean of affiliation precision and recall, measuring the temporal distance between predicted and ground-truth events, computed at the anomaly-ratio threshold (the F1-optimal-threshold variant is excluded from all rankings)."
Rationale: Decomposing the 155-word sentence into five sentences — one per metric — restores readability while preserving all information.
Scientific meaning changed: No

---

**S4-012**
Location: §4.1.3, sentence 3
Original: "The traditional point-adjusted F1 (PA F1) at $K{=}0$ \cite{xu2018kpivae} is reported only in Appendix §A.5 for comparability, marked (oracle) for its F1-optimal threshold, and is never used for ranking: even a random score can reach state-of-the-art levels under it \cite{kim2022rigorous}."
Issue type: "marked (oracle) for its F1-optimal threshold" is a compressed phrase — "(oracle)" is an in-table label whose meaning is not obvious to a reader encountering it here for the first time; "for its F1-optimal threshold" is an unusual preposition
Severity: SHOULD
Revised: "The traditional point-adjusted F1 (PA F1) at $K{=}0$ \cite{xu2018kpivae} is reported in Appendix §A.5 for comparability with prior work, labelled (oracle) to indicate that the threshold is selected to maximize F1; it is never used for ranking, as even a random score can reach state-of-the-art levels under it \cite{kim2022rigorous}."
Rationale: "Marked (oracle) for its F1-optimal threshold" is opaque; expanding to "labelled (oracle) to indicate that the threshold is selected to maximize F1" is self-explanatory. Replacing "only in Appendix §A.5" with "in Appendix §A.5 for comparability with prior work" avoids the slightly apologetic "only."
Scientific meaning changed: No

---

**S4-013**
Location: §4.1.4, sentence 1
Original: "We compare against 26 baselines: 22 unsupervised — nine simple-to-lightweight detectors following \cite{sarfraz2024quovadis}, six established deep TSAD systems \cite{...}, and seven recent competitive methods (including TFMAE, the time-series MAE variant discussed in Section 2.3) \cite{...} — and four weakly supervised methods exploiting labeled anomalies during training \cite{...}; the full tier list, implementation provenance (including the simplified DAGMM variant), and hyperparameters are in Appendix §A.1."
Issue type: Overlong sentence (74 words); the em-dash sublist is a nested enumeration inside an enumeration
Severity: SHOULD
Revised: "We compare against 26 baselines: 22 unsupervised and 4 weakly supervised. The 22 unsupervised methods comprise nine simple-to-lightweight detectors following \cite{sarfraz2024quovadis}, six established deep TSAD systems \cite{...}, and seven recent competitive methods including TFMAE \cite{...} (the time-series MAE variant discussed in Section 2.3). The four weakly supervised methods exploit labeled anomalies during training \cite{...}. Full tier list, implementation provenance (including the simplified DAGMM variant), and hyperparameters are in Appendix §A.1."
Rationale: Four separate sentences for the count, the unsupervised breakdown, the weakly supervised group, and the pointer to the appendix are easier to read than a 74-word nested structure.
Scientific meaning changed: No

---

**S4-014**
Location: §4.1.4 (Comparison conditions), sentence 1
Original: "The main comparison uses the Q3 (normal-only) condition for all 22 unsupervised baselines: labeled anomaly regions are excised from the contaminated training data and the surviving normal segments concatenated with boundary-aware windowing."
Issue type: Passive ellipsis — "the surviving normal segments concatenated" omits "are," making it a dangling participial
Severity: MUST-FIX
Revised: "The main comparison uses the Q3 (normal-only) condition for all 22 unsupervised baselines: labeled anomaly regions are excised from the contaminated training data, and the surviving normal segments are concatenated with boundary-aware windowing."
Rationale: "the surviving normal segments concatenated" is grammatically incomplete without "are"; restoring it fixes the ellipsis.
Scientific meaning changed: No

---

**S4-015**
Location: §4.2, sentence 1
Original: "Table 2 presents PA\%K-AUC F1 and VUS-PR for CSMAD and all 26 baselines across the six dataset families; full five-metric results are in Appendix §A.5 and per-entity results in Appendix §A.6."
Issue type: The second clause "per-entity results in Appendix §A.6" is an elliptical fragment missing "are"
Severity: MUST-FIX
Revised: "Table 2 presents PA\%K-AUC F1 and VUS-PR for CSMAD and all 26 baselines across the six dataset families; full five-metric results are in Appendix §A.5 and per-entity results are in Appendix §A.6."
Rationale: "per-entity results in Appendix §A.6" omits "are"; grammatically must be restored for parallel construction with "full five-metric results are."
Scientific meaning changed: No

---

**S4-016**
Location: §4.2 (Protocol-effect analysis), sentence 2
Original: "Under (i) the label-dependent pathways self-deactivate with the configuration held fixed (random masking, all-normal OD loss, no GRL loss), leaving a purely unsupervised asymmetric Teacher–Student MAE."
Issue type: "with the configuration held fixed" is a dangling participial that is ambiguous — does the configuration hold while the pathways self-deactivate, or is the configuration fixed and as a result the pathways self-deactivate?
Severity: SHOULD
Revised: "Under condition (i), with the configuration held fixed, the label-dependent pathways self-deactivate — random masking, all-normal OD loss, no GRL loss — leaving a purely unsupervised asymmetric Teacher–Student MAE."
Rationale: Moving "with the configuration held fixed" before the subject and converting the parenthetical to an em-dash appositive clarifies that the pathways self-deactivate as a result of the labeled anomalies being absent, not as a result of a configuration change.
Scientific meaning changed: No

---

**S4-017**
Location: §4.3 (Anomaly-priority masking row), sentence 1
Original: "Without it, random masking only rarely selects anomaly patches, leaving the Teacher's reconstruction deficit there largely unexploited; removal costs [X.XX] points on average."
Issue type: Vague "it" and "there" — "it" refers to anomaly-priority masking (clear from the subsection heading but not from the sentence alone); "there" is informal
Severity: SHOULD
Revised: "Without anomaly-priority masking, random masking only rarely selects anomaly patches, leaving the Teacher's reconstruction deficit at those positions largely unexploited; removal costs [X.XX] points on average."
Rationale: Making the subject explicit removes ambiguity; "at those positions" is more formal and precise than "there."
Scientific meaning changed: No

---

**S4-018**
Location: §4.4 (Why graceful degradation is expected), sentence 2 (the numbered list)
Original: "Three structural properties support robustness: (i) anomaly-priority masking applies only to labeled patches, leaving the label-free reconstruction objective unaffected by which anomalies are labeled; (ii) the GRL term draws its positive supervision exclusively from labeled windows — batches without a labeled positive skip the term entirely — so unlabeled anomaly windows, treated as negatives, never inject an erroneous positive adversarial signal; and (iii) the base reconstruction error is label-independent, elevated wherever a patch deviates from normal correlation structure."
Issue type: Overlong sentence (85 words); item (ii) contains a parenthetical that interrupts a long consequential clause
Severity: SHOULD
Revised: "Three structural properties support robustness. (i) Anomaly-priority masking applies only to labeled patches, so the label-free reconstruction objective is unaffected by which anomalies remain unlabeled. (ii) The GRL term draws positive supervision exclusively from labeled windows; batches without a labeled positive skip the term entirely, so unlabeled anomaly windows, treated as negatives, never inject an erroneous positive adversarial signal. (iii) The base reconstruction error is label-independent, elevated wherever a patch deviates from normal correlation structure."
Rationale: A three-item list spanning 85 words is hard to scan. Splitting into a sentence per item, with a lead sentence, produces a standard academic list structure.
Scientific meaning changed: No

---

**S4-019**
Location: §4.4 (Why graceful degradation is expected), sentence 3
Original: "As $p$ decreases, the discrepancy pathway and the adversarial suppression weaken together — fewer labeled patches are prioritized for masking and fewer batches activate the GRL term — so the Student's residual capacity to reconstruct anomalous patterns grows; the label-independent reconstruction term, however, remains elevated at anomalous patches, bounding the degradation from below as the model approaches its purely reconstruction-driven mode (Section 4.2)."
Issue type: Overlong sentence (75 words); two consecutive interruptions (em-dash pair and "however") buried mid-sentence
Severity: SHOULD
Revised: "As $p$ decreases, the discrepancy pathway and adversarial suppression weaken together: fewer labeled patches are prioritized for masking and fewer batches activate the GRL term, so the Student's residual capacity to reconstruct anomalous patterns grows. The label-independent reconstruction term, however, remains elevated at anomalous patches, bounding the degradation from below as the model approaches its purely reconstruction-driven mode (Section 4.2)."
Rationale: Splitting at the semicolon produces two focused sentences of 42 and 28 words, well within readable range.
Scientific meaning changed: No

---

**S4-020**
Location: §4.4 (Results), sentence 2 (placeholder sentence)
Original: "Performance declines as $p$ decreases but does so [gradually / monotonically], maintaining competitive detection at $p = 0.25$ and approaching the best unsupervised baseline at $p \approx 0$, confirming reversion to a pure reconstruction-based detector without falling below the unsupervised floor."
Issue type: "confirming reversion to a pure reconstruction-based detector without falling below the unsupervised floor" — the participial reads as if "approaching… at $p \approx 0$" is what confirms the reversion; the causal logic should be explicit
Severity: SHOULD
Revised: "Performance declines as $p$ decreases but does so [gradually / monotonically], maintaining competitive detection at $p = 0.25$ and approaching the best unsupervised baseline at $p \approx 0$. This confirms that CSMAD reverts to a reconstruction-based detector without falling below the unsupervised floor."
Rationale: Making "this confirms" the subject of a new sentence removes the participial ambiguity and separates the observation (the performance curve) from the conclusion (the reversion to reconstruction-driven mode).
Scientific meaning changed: No

---

**S4-021**
Location: §4.5, sentence 2
Original: "The two components respond distinctly: reconstruction error is elevated wherever the input deviates from learned normal patterns regardless of event type, while the discrepancy captures the additional divergence arising where the Student's limited capacity and adversarially suppressed representation fail to track the Teacher."
Issue type: "the additional divergence arising where" — "arising where" is a compressed relative construction; "adversarially suppressed representation" is a complex pre-modifier stack
Severity: SHOULD
Revised: "The two components respond distinctly: reconstruction error is elevated wherever the input deviates from learned normal patterns regardless of event type, while the discrepancy captures the additional divergence that arises where the Student's limited capacity and adversarially suppressed representation cause it to fail in tracking the Teacher."
Rationale: "Arising where the Student's… fail to track" — the participial "arising" is correctly used but the phrase is compressed; "that arises where the Student… cause it to fail in tracking" makes the causal chain explicit.
Scientific meaning changed: No

---

## §5 Conclusion

---

**S5-001**
Location: §5, sentence 1
Original: "This paper addressed the underexplored setting in which training data contain a small fraction of labeled anomalies alongside a majority of unlabeled observations — common in industrial deployments yet unsupported by standard MTSAD benchmarks or unsupervised methods."
Issue type: Tense — "addressed" is past tense; the present tense ("addresses") is conventional for conclusions that summarise the paper's contribution in the same paper
Severity: MUST-FIX
Revised: "This paper addresses the underexplored setting in which training data contain a small fraction of labeled anomalies alongside a majority of unlabeled observations — a configuration common in industrial deployments yet unsupported by standard MTSAD benchmarks or unsupervised methods."
Rationale: Past tense in a conclusion's first sentence is non-standard in journal papers; present tense is the convention. "A configuration" is added before "common" to prevent "common" from appearing to modify "observations" rather than the setting.
Scientific meaning changed: No

---

**S5-002**
Location: §5, sentence 2
Original: "We proposed CSMAD, which integrates labeled anomaly information into masked autoencoder representation learning through three orthogonal paths — anomaly-priority masking, loss bifurcation toward normal-only Student mimicry, and gradient-reversal suppression of anomaly-specific information — on top of an asymmetric Teacher–Student decoder architecture (3-layer Teacher, 2-layer Student) that converts the capacity gap into a reliable discrepancy signal under contaminated training."
Issue type: "We proposed CSMAD" — past tense in conclusion (same as S5-001); also "on top of an asymmetric…" is an unusual collocation in formal academic writing
Severity: MUST-FIX
Revised: "CSMAD integrates labeled anomaly information into masked autoencoder representation learning through three orthogonal paths — anomaly-priority masking, loss bifurcation toward normal-only Student mimicry, and gradient-reversal suppression of anomaly-specific information — built on an asymmetric Teacher–Student decoder architecture (3-layer Teacher, 2-layer Student) that converts the capacity gap into a reliable discrepancy signal under contaminated training."
Rationale: Dropping "We proposed" and starting with "CSMAD" avoids both the past-tense issue and the need for a verb. "Built on" replaces the informal "on top of."
Scientific meaning changed: No

---

**S5-003**
Location: §5, sentence 4
Original: "Experiments on [N] multivariate datasets show competitive performance against [N] unsupervised and weakly supervised baselines under five metrics, and the label sparsity analysis confirms graceful degradation as the labeled fraction decreases."
Issue type: "show competitive performance" — "competitive" is a weak claim in a conclusion; the paper uses "achieves competitive performance" or "outperforms" elsewhere with more specificity
Severity: OK-FLAG
Revised: "Experiments on [N] multivariate datasets demonstrate competitive performance against [N] unsupervised and weakly supervised baselines under five complementary metrics; the label sparsity analysis confirms graceful degradation as the labeled fraction decreases."
Rationale: "Demonstrate" is more formal than "show" in academic prose; "complementary metrics" (consistent with §4.1.3) is more informative than "five metrics."
Scientific meaning changed: No

---

**S5-004**
Location: §5, sentence 5
Original: "A notable limitation is the cost of leave-one-out inference — an approximately 50$\times$ increase in forward-pass computation relative to single-mask scoring; reducing this inference cost is a natural avenue for future work."
Issue type: Em-dash followed immediately by a semicolon is an unusual punctuation sequence; the second clause ("reducing…") has a vague antecedent ("this inference cost")
Severity: SHOULD
Revised: "A notable limitation is the cost of leave-one-out inference, which requires approximately 50$\times$ more forward-pass computation than single-mask scoring; reducing this cost is a natural avenue for future work."
Rationale: Replacing the em-dash + measure phrase with a relative clause ("which requires…") removes the unusual punctuation pattern and integrates the quantification naturally. "This cost" is unambiguous when it immediately follows "the cost of leave-one-out inference."
Scientific meaning changed: No

---

## Appendix A

---

**SA-001**
Location: §A.1 (CSMAD configuration), sentence 1
Original: "Table A.1 lists the complete configuration; all values are shared across the 113 learning units — only the input dimensionality $F$ (Table C.1), the data-derived class-prior weight $w_+$ (Eq. C.3), and the train/test split proportions implied by the protocol of Section 4.1.1 vary by entity."
Issue type: "implied by the protocol" is a vague qualifier — the split proportions are not just implied but directly determined by the protocol
Severity: OK-FLAG
Revised: "Table A.1 lists the complete configuration; all values are shared across the 113 learning units — only the input dimensionality $F$ (Table C.1), the data-derived class-prior weight $w_+$ (Eq. C.3), and the train/test split proportions determined by the protocol of Section 4.1.1 vary by entity."
Rationale: "Determined by" is more precise than "implied by" for a value that is directly computed from the protocol.
Scientific meaning changed: No

---

**SA-002**
Location: §A.1 (Training budgets and evaluation cadence), sentence 1
Original: "Table A.2 states the per-group budgets disclosed in Section 4.1.2."
Issue type: "states the… budgets disclosed in" — double-referral to the same information in an awkward phrasing
Severity: OK-FLAG
Revised: "Table A.2 summarizes the per-group training and evaluation budgets introduced in Section 4.1.2."
Rationale: "Summarizes" is more precise than "states" (the table presents a structured summary). "Training and evaluation budgets" specifies what the budgets cover.
Scientific meaning changed: No

---

**SA-003**
Location: §A.1 (Baseline implementations), sentence 3
Original: "Each baseline retains the hyperparameters of its original implementation or publication preset (for example, NRdetector keeps its native window size of 100 rather than the 500 used by our pipeline); the random-score baseline is averaged over five independent runs (mean ± std), and all other methods are single runs."
Issue type: "all other methods are single runs" — "are single runs" is an unusual predicate; "are run once" or "use a single run" is more natural
Severity: SHOULD
Revised: "Each baseline retains the hyperparameters of its original implementation or publication preset (for example, NRdetector keeps its native window size of 100 rather than the 500 used by our pipeline); the random-score baseline is averaged over five independent runs (mean ± std), while all other methods use a single run."
Rationale: "Are single runs" is not standard; "use a single run" is natural and consistent with standard reporting language.
Scientific meaning changed: No

---

**SA-004**
Location: §A.2 (Point adjustment and PA%K), sentence 2
Original: "PA\%K \cite{kim2022rigorous} parameterizes this leniency: a segment qualifies for adjustment only when strictly more than $K\%$ of its timesteps are predicted positive, so that $K = 0$ recovers conventional PA and $K = 100$ point-wise scoring."
Issue type: "K = 100 point-wise scoring" — missing verb; the parallel construction requires "K = 100 recovers point-wise scoring"
Severity: MUST-FIX
Revised: "PA\%K \cite{kim2022rigorous} parameterizes this leniency: a segment qualifies for adjustment only when strictly more than $K\%$ of its timesteps are predicted positive, so that $K = 0$ recovers conventional PA and $K = 100$ recovers point-wise scoring."
Rationale: The second coordinate ("K = 100 point-wise scoring") omits the verb "recovers," making it a grammatically incomplete clause.
Scientific meaning changed: No

---

**SA-005**
Location: §A.2 (VUS-ROC/VUS-PR), sentence 1
Original: "The Volume Under the Surface generalizes AUC-ROC/AUC-PR to a three-dimensional volume by sweeping both the decision threshold and a temporal tolerance parameter that softens segment boundaries; we use the authors' official implementation with tolerance window 100 after min–max normalization of scores."
Issue type: "tolerance window 100" is a bare numeric specifier without a unit noun; should be "a tolerance window of 100 timesteps" (or whatever the unit is)
Severity: SHOULD
Revised: "The Volume Under the Surface generalizes AUC-ROC/AUC-PR to a three-dimensional volume by sweeping both the decision threshold and a temporal tolerance parameter that softens segment boundaries; we use the authors' official implementation with a tolerance window of 100 timesteps and min–max normalization of scores."
Rationale: "Tolerance window 100" is a bare attributive numeral; "a tolerance window of 100 timesteps" is the correct form. Repositioning "min–max normalization" to parallel the tolerance window specification improves the flow.
Scientific meaning changed: No (assuming 100 refers to timesteps, consistent with the VUS literature)

---

**SA-006**
Location: §A.2 (Affiliation F1), sentence 1
Original: "Affiliation precision and recall convert the temporal distance between predicted and ground-truth events into per-event affinity scores within each event's affiliation zone, with formal robustness guarantees against adversarial scoring; Affiliation F1 is their harmonic mean."
Issue type: "Affiliation precision and recall convert… Affiliation F1 is their harmonic mean" — "their" refers back to "precision and recall" but is somewhat distant from the referent; also "affinity scores" and "affiliation zone" appear in rapid succession creating a terminological cluster
Severity: OK-FLAG
Revised: "Affiliation precision and recall convert the temporal distance between predicted and ground-truth events into per-event affinity scores within each event's affiliation zone, with formal robustness guarantees against adversarial scoring; Affiliation F1 is the harmonic mean of these two measures."
Rationale: "Their harmonic mean" requires the reader to recall "precision and recall" from earlier in the sentence; "the harmonic mean of these two measures" is unambiguous.
Scientific meaning changed: No

---

**SA-007**
Location: §A.3 (Training-label semantics), sentence 2
Original: "Labeled anomalies in our training splits therefore originate exclusively from the incorporated test prefixes."
Issue type: "therefore" implies a logical consequence of the preceding sentence, but the preceding sentence describes absence of labels in original training splits — the consequence is well-stated. Minor: "the incorporated test prefixes" is slightly ambiguous (prefixes of what?)
Severity: OK-FLAG
Revised: "Labeled anomalies in our training splits therefore originate exclusively from the incorporated test-stream prefixes."
Rationale: "Test-stream prefixes" is more precise than "test prefixes" (it refers to the test time series, not to the test files or test names).
Scientific meaning changed: No

---

**SA-008**
Location: §A.4 (Region definition), sentence 1
Original: "Attack region 22 is the chronologically first anomaly region within the held-out SWaT evaluation half, spanning evaluation-local positions $[2{,}869, 38{,}769)$ — 35,900 contiguous timesteps, which constitute 83.75\% of all anomalous timesteps and 15.96\% of the entire evaluation span."
Issue type: "spanning evaluation-local positions" — "evaluation-local" is a non-standard compound adjective; the reader must infer that it means positions indexed within the evaluation half rather than the full series
Severity: SHOULD
Revised: "Attack region 22 is the chronologically first anomaly region within the held-out SWaT evaluation half, spanning positions $[2{,}869, 38{,}769)$ (indexed within the evaluation span) — 35,900 contiguous timesteps that constitute 83.75\% of all anomalous timesteps and 15.96\% of the entire evaluation span."
Rationale: Replacing "evaluation-local positions" with "positions… (indexed within the evaluation span)" makes the coordinate system explicit without introducing a non-standard compound. Converting "which constitute" to "that constitute" (restrictive) is also a minor improvement.
Scientific meaning changed: No

---

## Appendix B

---

**SB-001**
Location: §B.1, sentence 1
Original: "The Q3 condition of the main comparison grants unsupervised baselines the most favorable use of the training labels (excision of contaminated regions)."
Issue type: "the most favorable use of the training labels" — the phrase implies the labels are being used by the baselines, but in Q3 the labels guide excision (a pre-processing step), not model training; this framing could be misleading
Severity: SHOULD
Revised: "The Q3 condition grants each unsupervised baseline the most favorable pre-processing benefit from the available labels: excision of contaminated training regions."
Rationale: "The most favorable use of the training labels" suggests the baselines use labels during training, which they do not. "The most favorable pre-processing benefit from the available labels" is accurate.
Scientific meaning changed: No

---

**SB-002**
Location: §B.2, sentence 2
Original: "To assess whether this asymmetry materially affects the comparison, representative unsupervised baselines are re-trained at extended budgets — and CSMAD at a reduced budget — under the otherwise unchanged protocol."
Issue type: "and CSMAD at a reduced budget" is a parenthetical insert without a verb, creating an ellipsis that requires the reader to supply "is re-trained"
Severity: SHOULD
Revised: "To assess whether this asymmetry materially affects the comparison, representative unsupervised baselines are re-trained at extended budgets, and CSMAD is re-trained at a reduced budget, under the otherwise unchanged protocol."
Rationale: The em-dash parenthetical conceals the ellipsis; converting to a coordinated clause restores grammatical completeness.
Scientific meaning changed: No

---

**SB-003**
Location: §B.5 (FM loss regularizer), sentence 1
Original: "Feature matching prevents the Student representation from collapsing under the competing pressures of OD supervision and GRL suppression; its removal costs [X.XX] points."
Issue type: "collapsing under the competing pressures" — "collapsing under pressures" is a mixed metaphor (representations do not collapse under pressure in the physical sense; they degenerate)
Severity: OK-FLAG
Revised: "Feature matching prevents the Student representation from degenerating under the competing objectives of OD supervision and GRL suppression; its removal costs [X.XX] points."
Rationale: "Degenerate" is the standard term for representation collapse in the deep learning literature; "pressures" is an informal metaphor for "objectives."
Scientific meaning changed: No

---

## Appendix C

---

**SC-001**
Location: §C.1 (Reversal-coefficient schedule), description sentence
Original: "where $e$ is the current epoch, $[e_0, e_1]$ the student-training phase (epochs 250–500 in the main configuration), and $\tau$ its progress; $\lambda_{\mathrm{rev}}$ rises monotonically from $\approx 0.02$ at the first Student epoch to $\approx 1$ at the end of training."
Issue type: "$[e_0, e_1]$ the student-training phase" — missing verb in the enumeration ("$[e_0, e_1]$ is the student-training phase")
Severity: MUST-FIX
Revised: "where $e$ is the current epoch, $[e_0, e_1]$ is the student-training phase (epochs 250–500 in the main configuration), and $\tau$ is its normalized progress; $\lambda_{\mathrm{rev}}$ rises monotonically from $\approx 0.02$ at the first Student epoch to $\approx 1$ at the end of training."
Rationale: The where-clause enumerates three definitions; the second and third omit "is" ("$[e_0,e_1]$ the student-training phase" and "$\tau$ its progress"). Adding "is" and specifying "$\tau$ is its normalized progress" (consistent with "progress $\tau$" in §3.4) makes the definitions grammatically complete.
Scientific meaning changed: No

---

**SC-002**
Location: §C.1 (Gradient reversal), sentence 1
Original: "The GRL \cite{ganin2016dann} is an identity map in the forward pass, $\tilde{h}^{\mathrm{S}}_i = h^{\mathrm{S}}_i$, where $\tilde{h}^{\mathrm{S}}_i$ denotes the GRL output forwarded to the classifier head $g_\phi$; in the backward pass it scales and negates the gradient:"
Issue type: "it scales and negates" — ambiguous as to whether scaling and negation are separate or a single combined operation; in §3.5 the phrase used is "scales and negates… by $\lambda_{\mathrm{rev}}$" but here the scale factor appears in the equation rather than the prose, leaving "it scales and negates" without an explicit scale parameter in the prose
Severity: OK-FLAG
Revised: "…; in the backward pass it multiplies the gradient by $-\lambda_{\mathrm{rev}}$ (equivalently: scales and negates it by $\lambda_{\mathrm{rev}}$):"
Rationale: "Multiplies by $-\lambda_{\mathrm{rev}}$" is the unambiguous description that matches Eq. C.2; the parenthetical preserves the informal "scales and negates" phrasing for readers familiar with GRL literature.
Scientific meaning changed: No

---

**SC-003**
Location: §C.1 (Classification loss), sentence 1
Original: "With $\hat{y}_i = g_\phi(\tilde{h}^{\mathrm{S}}_i)$ the classifier prediction for masked patch $i$, $\ell_i = \mathrm{BCE}_{w_+}(\hat{y}_i,\, y^w)$ its class-prior-weighted binary cross-entropy, and $w_+$ the per-entity normal-to-anomalous patch ratio (the anomalous-patch fraction floored at $10^{-3}$),"
Issue type: The antecedent-free nominative absolutes ("$\hat{y}_i$…, $\ell_i$…, and $w_+$…") use no main verb — the entire sentence is a multi-clause "with…" preamble with no conclusion; it ends with a comma leading into the display equation
Severity: OK-FLAG
Revised: "Let $\hat{y}_i = g_\phi(\tilde{h}^{\mathrm{S}}_i)$ be the classifier prediction for masked patch $i$, $\ell_i = \mathrm{BCE}_{w_+}(\hat{y}_i,\, y^w)$ its class-prior-weighted binary cross-entropy, and $w_+$ the per-entity normal-to-anomalous patch ratio (the anomalous-patch fraction floored at $10^{-3}$):"
Rationale: The "with…" preamble convention is used in some math writing but "Let… be…" is more natural in English-medium journal text and separates the definition from the equation that follows.
Scientific meaning changed: No

---

**SC-004**
Location: §C.1 (Classification loss), sentence 2
Original: "Unlike the standard focal loss \cite{lin2017focal}, which defines its modulating probability $p_t$ from the raw prediction, here $p_t := e^{-\ell_i}$ derives from the pos-weight-adjusted BCE, weighting hard examples by both confidence and prior imbalance; this variant is part of the present design rather than an external import."
Issue type: "Unlike the standard focal loss, which defines…, here $p_t$…" — dangling unlike-clause (the subject that differs from the focal loss is "$p_t$" but the syntactic subject of the main clause is "here," which is an adverb); also "this variant is part of the present design rather than an external import" is an unusual framing
Severity: SHOULD
Revised: "Unlike the standard focal loss \cite{lin2017focal}, which derives its modulating probability $p_t$ from the raw prediction, the present variant defines $p_t := e^{-\ell_i}$ from the pos-weight-adjusted BCE, weighting hard examples by both confidence and prior imbalance; this formulation is introduced as part of the present design."
Rationale: "Unlike X, here Y" dangles because "here" is an adverb and cannot be the subject that differs from X. "Unlike X, the present variant defines…" correctly places the subject of the unlike-clause as the head of the main clause. "Rather than an external import" is informal; "introduced as part of the present design" is equivalent and formal.
Scientific meaning changed: No

---

**SC-005**
Location: §C.1 (Adaptive loss weights), sentence 2
Original: "The adversarial gradient reaching the Student hidden state is therefore $-\lambda_{\mathrm{rev}} \cdot \lambda_{\mathrm{GRL}} \cdot \partial L_{\mathrm{cls}} / \partial(\mathrm{GRL\ output})$: the reversal coefficient and the loss weight act multiplicatively and remain distinct quantities."
Issue type: "act multiplicatively and remain distinct quantities" — "remain distinct quantities" is a slightly unusual formulation; the intended meaning is that they govern different aspects of the gradient flow and should not be conflated
Severity: OK-FLAG
Revised: "The adversarial gradient reaching the Student hidden state is therefore $-\lambda_{\mathrm{rev}} \cdot \lambda_{\mathrm{GRL}} \cdot \partial L_{\mathrm{cls}} / \partial(\mathrm{GRL\ output})$: the reversal coefficient and the loss weight contribute multiplicatively but govern distinct aspects of the gradient flow."
Rationale: "Remain distinct quantities" is redundant (two mathematically separate symbols are always distinct); "govern distinct aspects of the gradient flow" is more informative about why the distinction matters.
Scientific meaning changed: No

---

---

## Audit Summary

| Section | Sentences inspected | MUST-FIX | SHOULD | OK-FLAG |
|---|---|---|---|---|
| Abstract | 7 | 1 | 3 | 2 |
| Highlights | 5 | 0 | 1 | 1 |
| §1 Introduction | 28 | 3 | 4 | 2 |
| §2 Related Work | 22 | 2 | 6 | 5 |
| §3 Methodology | 40 | 6 | 8 | 5 |
| §4 Experiments | 60 | 8 | 10 | 2 |
| §5 Conclusion | 6 | 2 | 2 | 1 |
| Appendix A | 24 | 1 | 3 | 3 |
| Appendix B | 10 | 0 | 3 | 1 |
| Appendix C | 12 | 2 | 2 | 2 |
| **Total** | **214** | **25** | **42** | **24** |

**Total issues flagged: 91 across 214 sentences inspected (42.5% sentence-level issue rate).**

Key recurring problem classes:
1. Overlong sentences (40+ words) — 14 instances, concentrated in §3 and §4 methodology/setup paragraphs.
2. Dangling or elliptical participials without auxiliary verb — 7 instances (MUST-FIX category).
3. Tense inconsistency (past vs. present for cited work) — 3 instances in §2.
4. Vague demonstrative pronouns (this/it/there/above) without explicit antecedent — 9 instances.
5. Broken parallelism in enumerated lists — 5 instances.
