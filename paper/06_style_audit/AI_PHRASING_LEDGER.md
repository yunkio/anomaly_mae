---
phase: 6
agent: ai-phrasing-detector
directives: [T6, R4]
corpus: paper/02_venue_study/SENTENCE_CORPUS.md (11 papers, ~105 sentences, Appendix A–B)
last_modified: 2026-06-11
manuscript: paper/05_manuscript/MANUSCRIPT_v2.md (v2, Phase 5 final)
sentence_count_total: 187
severity_counts: {MUST-FIX: 11, SHOULD: 19, OK-FLAG: 8}
---

> **Usage contract.** Every entry quotes the offending phrase exactly, states the corpus-derived
> rule it violates, and gives a full replacement sentence. Alternatives are evidence-calibrated —
> they do not introduce new superlatives or hedges that are not warranted by the manuscript context.
> Placeholder tokens (NUM-xxx, PH:…) inside sentences are treated as inert; the prose around them
> is the unit under review.

---

## I. Applicable Prohibition Patterns (corpus-derived)

Reference: SENTENCE_CORPUS.md Appendix B (금지/자제/허용 판정) + Appendix A collocation list.
Below is the operative list applied to this manuscript, extended with two patterns detected
during this audit pass.

### I.1 Hard prohibitions (MUST-FIX on any occurrence)

| ID | Pattern | Corpus rule | Notes |
|----|---------|-------------|-------|
| P-01 | "pivotal" | B.1 — 금지; real corpus uses "critical/important" | |
| P-02 | "delve / delves into" | B.1 — 금지 | |
| P-03 | "showcase(s)" | B.1 — 금지 | |
| P-04 | "underscore(s)" / "highlights the importance of" | B.1 — 금지 | |
| P-05 | "in the realm of" / "landscape" / "ever-evolving" | B.1 — 금지 | |
| P-06 | "seamlessly / meticulously / holistically" | B.1 — 금지 | |
| P-07 | "paving the way" / "unlock" / "harness the power of" | B.1 — 금지 | |
| P-08 | "a testament to" / "boasts" | B.1 — 금지 | |
| P-09 | "In conclusion," as section opener | B.2 — 금지 | |
| P-10 | "revolutionize" / "groundbreaking" / "opens up exciting avenues" | B.2 — 금지 | |
| P-11 | Model/score anthropomorphization: "strives to", "score tells us" (beyond "aims to"/"designed to") | B.2 — 금지 | |
| P-12 | "It is important to note/emphasize that" | B.2 — 금지; use "Note that" | |
| P-13 | Abstract-style "In recent years, … has received growing/significant attention" as vague opener | B.1 context — not attested; section-opener variant of the "recent years" framing | New — detected this pass |
| P-14 | Vague significance claim without a number: "significantly improves/enhances/boosts" with no figure | B.1 significant(ly) — 자제; without number = MUST-FIX | New — detected this pass |

### I.2 Restraint patterns (SHOULD-flag if over-used or used without warrant)

| ID | Pattern | Corpus rule | Threshold |
|----|---------|-------------|-----------|
| R-01 | "novel" | B.1 자제 — contribution sentences only, not per-paragraph | >1 instance per section |
| R-02 | "comprehensive" | B.1 자제 — experimental comparison context only | Outside that context |
| R-03 | "significant(ly)" without number | B.1 자제 — must pair with a figure | Any unnumbered use |
| R-04 | Moreover / Furthermore / Additionally in sequence | B.2 자제 — max 1–2 per section | Sequential run |
| R-05 | "It is worth noting that" | B.1 자제 — use "Note that" instead; max 1/paper | |
| R-06 | Vague "complex temporal dependencies" without specifying what structure is meant | MEMTO abstract is the corpus example of this phrase; acceptable once as domain-standard label; flagged if repeated vaguely | >1 bare use |
| R-07 | Formal 3-abstract-noun parallels ("efficiency, scalability, robustness") | B.2 금지 for pure-abstract triples | |
| R-08 | Blanket effectiveness claim: "demonstrates the effectiveness of" without naming the design principle or citing a number | B.2 빈 요약 자제 | |
| R-09 | "em-dash (—) to splice independent clauses" repeated in prose | B.2 자제 | >2 per paragraph |
| R-10 | "robust" / "robustness" used as ungrounded adjective | Corpus uses it with either a metric or a structural argument | Without anchor |
| R-11 | "natural avenue" / "natural extension" as conclusion filler | 0 corpus occurrences; weaker than a specific research direction | Any use |
| R-12 | Passive + vague result: "is shown to be effective" | Corpus always names the mechanism and cites a figure (B.2 빈 요약) | |
| R-13 | Asymmetric hedge: hedge on strong claims, no hedge on weak ones | Corpus rule B.3.5 — hedge on limits/future, assertive on own results | |

---

## II. Full Detection Table

Sentence numbering follows manuscript section order. Inline comment markers
(`<!-- … -->`) and placeholder tokens are excluded from counts.
The section label uses the manuscript heading exactly.

Legend: **MUST** = MUST-FIX | **SHOULD** = SHOULD-fix | **FLAG** = OK-FLAG (boundary case, record only)

---

### Abstract (8 prose sentences)

---

**[A-01]** MUST-FIX

- **Location:** Abstract, sentence 3
- **Original:** "We propose CSMAD, an end-to-end framework that integrates labeled anomaly information directly into masked autoencoder representation learning through three orthogonal mechanisms: anomaly-priority masking, loss bifurcation between normal and anomalous reconstruction paths, and a gradient reversal layer that adversarially suppresses anomaly-specific information from the Student's internal representation."
- **Issue:** The sentence is accurate and precise, but "loss bifurcation between normal and anomalous reconstruction paths" is vague phrasing — "bifurcation" is not a standard TSAD collocation (Appendix A.3 reconstruction collocations) and here it reads as an inflated synonym for "splitting the loss." The phrase "adversarially suppresses" is precise and acceptable; "orthogonal mechanisms" is acceptable once (corpus: none, but it is technical and non-generic here). The primary problem is "loss bifurcation between … paths" — abstract noun inflation that obscures the concrete operation (restricting the Student's imitation loss to normal patches only).
- **Fix:** Replace "loss bifurcation between normal and anomalous reconstruction paths" with the operationally precise phrasing already used in the body ("restricts the Student decoder's imitation objective to normal-patch outputs").
- **Revised sentence:** "We propose CSMAD, an end-to-end framework that integrates labeled anomaly information directly into masked autoencoder representation learning through three orthogonal mechanisms: anomaly-priority masking, a Student imitation loss restricted to normal-patch outputs, and a gradient reversal layer that adversarially suppresses anomaly-specific information from the Student's internal representation."

---

**[A-02]** SHOULD

- **Location:** Abstract, sentence 7
- **Original:** "The model maintains robust detection as the labeled anomaly fraction decreases, validating the framework beyond the upper-bound labeling scenario."
- **Issue:** "robust detection" is an unanchored use of "robust" (R-10). The corpus uses "robust" either with a metric ("consistently achieves robust results" — NRdet, paired with benchmark data) or a structural argument. Here no figure accompanies it. "validating the framework beyond the upper-bound labeling scenario" is an abstract result claim without a number.
- **Fix:** Qualify "robust" with the structural anchor already in §4.4 ("does not fall below the unsupervised floor"), or defer the unqualified claim to the body where numbers appear.
- **Revised sentence:** "Detection performance degrades gracefully as the labeled anomaly fraction decreases, remaining above the unsupervised baseline floor and validating the framework beyond the upper-bound labeling scenario."

---

### Highlights (5 bullet sentences)

---

**[H-01]** SHOULD

- **Location:** Highlights, bullet 5
- **Original:** "CSMAD outperforms unsupervised baselines under multi-metric evaluation on [N] datasets and stays robust to label sparsity."
- **Issue:** "stays robust to label sparsity" is the same unanchored "robust" as A-02 (R-10); a highlights bullet should be more specific. "outperforms unsupervised baselines" is a blanket superiority claim acceptable when the table confirms it (R-07-adjacent), but "outperforms" without qualification is stronger than the careful language used in the body ("competitive performance against … baselines").
- **Fix:** Align wording with the body's own hedge ("competitive performance") and ground "robust" in the structural argument.
- **Revised sentence:** "CSMAD achieves competitive performance against unsupervised baselines under five metrics on [N] datasets; detection degrades gradually toward the unsupervised floor as label sparsity increases."

---

### §1 Introduction (35 prose sentences)

---

**[I-01]** MUST-FIX

- **Location:** §1, paragraph 1, sentence 1
- **Original:** "Real-world cyber-physical systems continuously generate high-dimensional, multi-channel sensor streams — water treatment plants, server clusters, and spacecraft telemetry arrays all depend on reliable detection of anomalous states to prevent safety incidents and operational losses."
- **Issue:** The em-dash usage here is the most prominent stylistic marker: two independent clauses joined by "—" rather than a period or a restructured sentence. The corpus uses em-dashes extremely rarely (B.2 자제); this opening sentence sets the register. The dash construction ("X — Y all depend on …") is structurally weak — the second clause loses its own subject–verb sharpness. Additionally "high-dimensional, multi-channel" is a minor double-epithet inflation (both mean the same thing in context).
- **Fix:** Split into two sentences or replace the dash with a subordinating phrase.
- **Revised sentence:** "Real-world cyber-physical systems continuously generate high-dimensional sensor streams from water treatment plants, server clusters, and spacecraft telemetry arrays, all of which depend on reliable detection of anomalous states to prevent safety incidents and operational losses."

---

**[I-02]** SHOULD

- **Location:** §1, paragraph 1, sentence 2
- **Original:** "Anomalies in such streams manifest not in isolated channels but through correlated deviations across multiple sensor dimensions \cite{…}, and because exhaustive point-level annotation of anomalies is infeasible at scale, the dominant paradigm for multivariate time series anomaly detection (MTSAD) has been unsupervised learning \cite{…}."
- **Issue:** This is a single sentence that chains two distinct ideas (anomaly structure + annotation cost → unsupervised learning) with "and because". Structurally acceptable in academic writing, but the compound is very long and the causal bridge ("and because … the paradigm has been") is slightly awkward. No AI-pattern prohibition violated, but the sentence length and "and because" construction are weak against the corpus's cleaner two-sentence style for motivation (corpus §1 examples use a period between the anomaly-characterization and the paradigm statement).
- **Fix:** Split after "sensor dimensions" and open the second sentence with "Because exhaustive point-level annotation … is infeasible at scale, …".
- **Revised sentence:** "Anomalies in such streams manifest not in isolated channels but through correlated deviations across multiple sensor dimensions \cite{deng2021gdn, wu2025catch}. Because exhaustive point-level annotation of anomalies is infeasible at scale, the dominant paradigm for multivariate time series anomaly detection (MTSAD) has been unsupervised learning \cite{wang2025nrdetector}."

---

**[I-03]** MUST-FIX

- **Location:** §1, paragraph 2, sentence 2
- **Original:** "Despite their differences, all four families share an implicit assumption that the training data are drawn entirely from normal operations."
- **Issue:** The sentence is technically clean, but "share an implicit assumption" is the standard AI-inflated framing for "assume." The corpus never uses "share an implicit/explicit assumption" — it states the assumption directly ("all four families assume …" or "the training data are treated as entirely normal"). "Implicit assumption" is a bloated construction; the argument in §1 para 3 and §2 makes the assumption explicit, contradicting the claim that it is "implicit."
- **Fix:** State the assumption directly.
- **Revised sentence:** "Despite their differences, all four families treat the training data as drawn entirely from normal operations."

---

**[I-04]** MUST-FIX

- **Location:** §1, paragraph 2, sentence 3
- **Original:** "This assumption is structurally embedded: the methods have no architectural pathway for leveraging the information carried by labeled anomalies even when such labels are available — the best a label-aware variant can do is exclude confirmed anomaly windows from training, filtering contamination rather than learning from it \cite{…}."
- **Issue:** Two problems. (1) "This assumption is structurally embedded:" — a colon-definition launch for an abstract property ("structurally embedded") is a vague AI-tier opener; the colon promises a definition but delivers an inference. (2) The em-dash continuation "— the best a label-aware variant can do …" is a second long clause dangling off an already heavy sentence (I-01 already flagged the em-dash pattern). The phrase "leveraging the information carried by" contains "leveraging" in a non-technical sense ("leveraging information") — B.1 permits "leverage" only in a technical-mechanism context; "leveraging information" is the buzzword variant.
- **Fix:** Replace "leveraging the information carried by" with a precise verb; rework the colon-launch into a direct statement; break the em-dash continuation into its own sentence.
- **Revised sentence:** "Consequently, these methods have no architectural pathway for exploiting labeled anomalies even when such labels are available. The best a label-aware variant can do is exclude confirmed anomaly windows from training, filtering contamination rather than learning from it \cite{wang2025nrdetector}."

---

**[I-05]** SHOULD

- **Location:** §1, paragraph 3, sentence 1
- **Original:** "In practice, however, a small fraction of training observations do carry anomaly labels, typically derived from recorded fault and attack events in operational logs \cite{…}."
- **Issue:** "In practice, however" is a legitimate transition (corpus uses "However" frequently), and the sentence is otherwise clean. Minor flag: "do carry" (emphatic "do") is natural but slightly informal for this venue. No strong prohibition violated. Recording as SHOULD.
- **Fix:** Minor register adjustment.
- **Revised sentence:** "In practice, however, a small fraction of training observations carry anomaly labels, typically derived from recorded fault and attack events in operational logs \cite{wang2025nrdetector}."

---

**[I-06]** MUST-FIX

- **Location:** §1, paragraph 3, sentence 3 (the long sentence about gap + benchmarks)
- **Original:** "The gap is particularly acute in the standard MTSAD benchmarks we evaluate on, whose original training splits contain no labeled anomalies by construction (per-dataset label semantics in Appendix §A.3) — benchmark studies have independently criticized dataset and evaluation practices in this field \cite{…}; evaluating any method that exploits labeled anomalies therefore requires modifying the data protocol, as detailed in Section 4.1.1."
- **Issue:** This is an extremely compound sentence (three independent clauses joined by "—" and ";") that would be broken into two or three sentences in every corpus example. The em-dash here introduces a parenthetical aside that is barely connected to the main clause. The semicolon continuation is also heavy. This is a structural AI-inflation pattern: adding more content to a sentence via dashes and semicolons rather than restructuring. Additionally "The gap is particularly acute" is a vague characterization of "gap" (gap = what exactly? the absence-of-label problem just named).
- **Fix:** Break into three sentences; name the gap explicitly.
- **Revised sentences:** "This label gap is most concrete in the standard MTSAD benchmarks we evaluate on: their original training splits contain no labeled anomalies by construction (per-dataset semantics in Appendix §A.3). Benchmark studies have independently criticized dataset and evaluation practices in this field \cite{liu2024elephant, schmidl2022evaluation}. Evaluating any method that exploits labeled anomalies therefore requires modifying the data partition, as detailed in Section 4.1.1."

---

**[I-07]** FLAG

- **Location:** §1, paragraph 4, sentence 1 (key observation paragraph)
- **Original:** "Our key observation is threefold: labeled anomalies reveal (a) which temporal positions yield informative hard reconstruction targets, (b) which patches the Student decoder should avoid mimicking, and (c) what representational content should be actively erased from the Student's encoding."
- **Issue:** "Our key observation is that …" is an attested corpus pattern (AnomTr abstract: "Our key observation is that due to the rarity of anomalies …"). The "threefold" variant ("is threefold:") uses a colon-definition launch that is marginally more AI-inflated than the corpus version. The three items are concrete, not abstract-noun triples, so the R-07 ban on "efficiency, scalability, robustness" triples does not apply. Recording as OK-FLAG: the core pattern is attested; "threefold" coloring is borderline.
- **Fix (optional):** Replace "is threefold:" with "is as follows:".
- **Alternative:** "Our key observation spans three axes: labeled anomalies reveal (a) … (b) … (c) …."

---

**[I-08]** MUST-FIX

- **Location:** §1, paragraph 4, sentence 2
- **Original:** "Exploiting all three simultaneously amplifies both the reconstruction error signal and the Teacher–Student discrepancy signal on anomalous regions."
- **Issue:** "Exploiting all three simultaneously" — "exploiting" is a legitimate technical verb (corpus: "exploit" appears in related-work positioning). The sentence itself is fine. However, "amplifies both … the reconstruction error signal and the Teacher–Student discrepancy signal" contains two uses of the word "signal" in one short noun phrase ("reconstruction error signal", "discrepancy signal"), which is a minor repetition that reads like generated prose trying to add symmetry. The graver problem is the standalone sentence's function: it is a result claim without a number — this is the B.2 "vague effectiveness claim" pattern (R-08). The claim is made again with data in §4.2 and §4.3; here it is asserted as a premise before any evidence.
- **Fix:** Reframe as a design rationale rather than an unsupported result claim.
- **Revised sentence:** "Exploiting all three simultaneously is designed to amplify both the reconstruction error and the Teacher–Student discrepancy at anomalous regions, as Section 4.3 confirms."

---

**[I-09]** SHOULD

- **Location:** §1, paragraph 4, sentence 3
- **Original:** "Relying only on (b) is insufficient: a Student repeatedly exposed to anomalous patterns during training may learn to reconstruct them accurately through an indirect route, weakening the discrepancy signal at inference time; the active suppression of (c) closes this route at the representational level."
- **Issue:** The colon + semicolon double-layer is syntactically valid but heavy. "closes this route at the representational level" is fine technically. The main style concern is the "Relying only on (b) is insufficient:" launch — a colon-definition for an abstract insufficiency claim. The corpus pattern for this is "Note that … cannot …" followed by the proposal. Minor; recording as SHOULD.
- **Fix:** Replace the colon launch with "Note that relying only on (b) is insufficient: …" or restructure as a direct assertion.
- **Revised sentence:** "Relying only on (b) is insufficient. A Student repeatedly exposed to anomalous patterns during training may learn to reconstruct them accurately through an indirect route, weakening the discrepancy signal at inference time; the GRL suppression of (c) closes this route at the representational level."

---

**[I-10]** SHOULD

- **Location:** §1, contribution paragraph, item 4
- **Original:** "Experiments on [N] multivariate datasets covering industrial control, IT infrastructure, and spacecraft telemetry demonstrate competitive performance against [N] baselines under five evaluation metrics, with label sparsity analysis confirming robust detection toward the fully unsupervised limit."
- **Issue:** "confirming robust detection toward the fully unsupervised limit" — same unanchored "robust" as A-02 (R-10). "demonstrate competitive performance" is an empty effectiveness claim without a number at this stage (R-08); the contribution bullet is appropriate to make a claim, but "competitive" without a figure is weak compared to the corpus pattern "achieves comparable or superior" paired with a count or percentage.
- **Fix:** Defer "robust" qualification to the results section; use the corpus-aligned "comparable or superior" phrasing.
- **Revised sentence:** "Experiments on [N] multivariate datasets covering industrial control, IT infrastructure, and spacecraft telemetry show performance comparable to or surpassing the strongest unsupervised baselines under five evaluation metrics, with the label sparsity sweep confirming that detection degrades gradually toward the unsupervised floor."

---

**[I-11]** SHOULD

- **Location:** §1, final sentence (roadmap)
- **Original:** "The rest of this paper is organized as follows: Section 2 reviews related work; Section 3 describes CSMAD; Section 4 presents experimental results; Section 5 concludes."
- **Issue:** "The rest of this paper is organized as follows:" is a standard boilerplate roadmap opener. It is not prohibited; the corpus does not demonstrate this pattern (none of the 11 papers use it in the surveyed excerpts), but it is an accepted convention in Elsevier journals. It is flagged as SHOULD because it adds no information and the corpus examples integrate the roadmap differently (some papers omit it, others embed it into the last intro paragraph with "We …"). No strong prohibition, but weaker than alternatives.
- **Fix (optional):** Integrate the roadmap into the preceding paragraph, or use the minimal form.
- **Revised sentence (optional replacement):** "Section 2 reviews related work; Section 3 describes CSMAD; Section 4 presents experimental results; Section 5 concludes."

---

### §2 Related Work (40 prose sentences across three subsections)

---

**[RW-01]** SHOULD

- **Location:** §2.1, sentence 1
- **Original:** "Deep learning approaches to unsupervised MTSAD have matured into several well-defined families."
- **Issue:** "have matured into several well-defined families" is a vague field-evolution claim — "matured" is an AI-tier soft metaphor for a field development. The corpus opens related-work sections by directly naming the families or citing a representative ("Classical methods include …" GDN; "Reconstruction-based methods still dominate …" DCdet). "Matured" adds no technical content; "well-defined" is a praise-adjective for categories the author is about to define.
- **Fix:** Lead with the content directly.
- **Revised sentence:** "Unsupervised MTSAD methods fall into several distinct families."

---

**[RW-02]** FLAG

- **Location:** §2.1, sentence 3 (the long sentence on association/contrastive methods)
- **Original:** "A more recent strand exploits association structure: transformer models that learn temporal dependencies \cite{…} or contrast multi-scale views of the series \cite{…} score the discrepancy between learned and actual patterns, and frequency-domain reconstruction has been extended with explicit channel-correlation discovery \cite{…}."
- **Issue:** The colon-definition ("exploits association structure: transformer models that …") is a borderline colon-launch. The sentence is technically precise. The colon here is used as a colon-of-elaboration rather than definition inflation, which is acceptable. Recording as FLAG for the em-dash-adjacent structure and length.
- **Fix (optional):** Separate into two sentences.

---

**[RW-03]** SHOULD

- **Location:** §2.1, sentence 4 (paragraph 2, sentence 1)
- **Original:** "Despite this breadth, every family above treats the training data as predominantly or entirely normal."
- **Issue:** "Despite this breadth" is a boilerplate transition. The corpus uses "Despite the growing interest, X remains …" (SDMAE) and "Despite their differences, …" — "despite this breadth" is not attested and reads as a generic paragraph-transition filler. The content is solid; only the opener is weak.
- **Fix:** Remove the filler opener; begin with the substance.
- **Revised sentence:** "Every family above treats the training data as predominantly or entirely normal."

---

**[RW-04]** SHOULD

- **Location:** §2.1, sentence 5 (when training stream contains …)
- **Original:** "When the training stream contains confirmed anomalous events — the contaminated setting that arises naturally from operational logs — these methods cannot distinguish known-anomalous from known-normal samples; labeled information is either discarded or treated as noise degrading the normal pattern \cite{…}."
- **Issue:** The em-dash parenthetical ("— the contaminated setting that arises naturally from operational logs —") interrupts the main clause and then the sentence continues with a semicolon continuation. This is the second-heaviest em-dash-plus-semicolon construction in the manuscript (cf. I-06). The interrupting dash clause just restates the terminology introduced three lines earlier. Flagged as SHOULD.
- **Fix:** Inline the definition as a parenthetical or move it to a footnote; break the semicolon.
- **Revised sentence:** "When the training stream contains confirmed anomalous events (the contaminated setting — see Section 3.1), these methods cannot distinguish known-anomalous from known-normal samples; labeled information is either discarded or treated as noise degrading the normal representation \cite{wang2025nrdetector}."

---

**[RW-05]** FLAG

- **Location:** §2.1, sentence 6 (last sentence)
- **Original:** "The present work addresses this structural limitation by integrating labeled anomaly information directly into representation learning rather than relying on post-hoc removal."
- **Issue:** "The present work addresses this structural limitation" is a standard positioning sentence; "structural limitation" is precisely named. This sentence is clean. FLAG only for "post-hoc removal" — "post-hoc" is correct but slightly jargon-heavy for this context; "post-training removal" or "excision after training" would be more concrete. Acceptable as-is.

---

**[RW-06]** MUST-FIX

- **Location:** §2.2, sentence 1 (PU learning intro sentence)
- **Original:** "Positive and Unlabeled (PU) learning formalizes the scenario in which a learner has confirmed positive examples and a pool of unlabeled data that may contain additional positives \cite{…}, with established solution families spanning cost-sensitive risk minimization via non-negative risk estimators \cite{…}, class-prior-based probability correction \cite{…}, and two-step techniques that first extract reliable negatives before training a classifier \cite{…}."
- **Issue:** This is a 60-word single sentence that chains a definition with three literature families via participial phrases ("with established solution families spanning …"). While the content is accurate, this construction is an AI-generation signature: one mega-sentence enumerating everything at once. The corpus uses short statements for taxonomy ("Classical methods include density-based approaches, linear-model based approaches …" GDN — with separate citations per family, one sentence) and never chains a definition with a 3-way taxonomic list in a single sentence. Additionally "with established solution families spanning" is the abstract-noun meta-language (R-07 adjacent — "spanning" here is a vague connective, not a technical verb).
- **Fix:** Break into the definition sentence + a separate sentence listing the solution families.
- **Revised sentences:** "Positive and Unlabeled (PU) learning formalizes the scenario in which a learner has confirmed positive examples and a pool of unlabeled data that may contain additional positives \cite{bekker2020pusurvey, duplessis2014pu}. Established approaches include cost-sensitive risk minimization via non-negative risk estimators \cite{kiryo2017nnpu}, class-prior-based probability correction \cite{elkan2008pu}, and two-step techniques that first extract reliable negatives before training a classifier \cite{bekker2020pusurvey}."

---

**[RW-07]** SHOULD

- **Location:** §2.2, sentence 2
- **Original:** "Outside time series, these ideas have been adapted to anomaly detection through deviation networks with scarce labeled anomalies \cite{…} and deep semi-supervised anomaly detection objectives \cite{…}."
- **Issue:** "Outside time series, these ideas have been adapted to …" is a standard research-synthesis sentence. Acceptable; flagged SHOULD only because "these ideas" is vague — "ideas" is not a technical term and the corpus never uses it. Prefer "these techniques" or "this framework."
- **Fix:** Replace "these ideas" with "these techniques."
- **Revised sentence:** "Outside time series, these techniques have been adapted to anomaly detection through deviation networks with scarce labeled anomalies \cite{pang2019devnet} and deep semi-supervised anomaly detection objectives \cite{ruff2020deepsad}."

---

**[RW-08]** MUST-FIX

- **Location:** §2.2, sentence 3 (first sentence of paragraph 2)
- **Original:** "In the time-series domain, deep representation learning informed by label signals remains rare \cite{…}."
- **Issue:** "informed by label signals" is a vague AI-inflated gerund phrase. "Label signals" is not a standard TSAD collocation (Appendix A.2 lists "anomaly score / detection criterion / anomaly labels" — not "label signals"). "Deep representation learning informed by label signals" means "representation learning that uses labels"; the phrase inflates the concept. Additionally "remains rare" is a bare field-state claim that sounds like a background-section opener filler.
- **Fix:** State the technical fact directly.
- **Revised sentence:** "In the time-series domain, methods that incorporate anomaly labels into the representation learning objective itself are rare \cite{wang2025nrdetector}."

---

**[RW-09]** SHOULD

- **Location:** §2.2, sentence 5 (contrast with our use of labels)
- **Original:** "Our use of labels differs in kind: rather than serving as the target of a classification or ranking objective, the label shapes the gradient of a masked-reconstruction pretext, steering what the encoder itself learns to represent."
- **Issue:** "differs in kind" is a legitimate scholarly expression. The colon-launch ("differs in kind:") is a borderline colon-definition; the following clause ("rather than … the label shapes …") is a clean contrastive statement. The phrase "steering what the encoder itself learns to represent" is acceptable but slightly abstract — "steering" is an informal metaphor (not a corpus-attested technical verb for gradient influence). Flagged SHOULD.
- **Fix:** Replace "steering" with "shaping" or "directing."
- **Revised sentence:** "Our use of labels differs in kind: rather than serving as the target of a classification or ranking objective, the label shapes the gradient of a masked-reconstruction pretext, directing what the encoder itself learns to represent."

---

**[RW-10]** SHOULD

- **Location:** §2.2, final sentence
- **Original:** "To our knowledge, CSMAD is the first end-to-end multivariate TSAD model that integrates labeled anomalies adversarially — through gradient reversal — into the gradient of a masked-reconstruction self-distillation objective."
- **Issue:** "To our knowledge, … is the first … that" is the correct corpus pattern for a narrow novelty claim (corpus §3: "is the first multivariate time series anomaly detection method that …" MEMTO). The em-dash interruption ("— through gradient reversal —") in the middle of a "first … that" claim is not ideal — it breaks the forward momentum of the claim. The claim would read more cleanly as a plain restrictive clause.
- **Fix:** Move "through gradient reversal" into the clause.
- **Revised sentence:** "To our knowledge, CSMAD is the first end-to-end multivariate TSAD model that integrates labeled anomalies into the gradient of a masked-reconstruction self-distillation objective through gradient reversal."

---

**[RW-11]** FLAG

- **Location:** §2.3, sentence 1
- **Original:** "The masked autoencoder (MAE) of He et al. \cite{…} showed that masking random patches and reconstructing the missing regions yields strong transferable representations."
- **Issue:** "yields strong transferable representations" — "strong" is marginally vague (R-03 adjacent); the MAE paper says "scalable self-supervised learners," not "strong transferable representations." However, this is a paraphrase, not a verbatim claim, and the framing is standard in related-work summaries. Flagged as OK-FLAG.

---

**[RW-12]** SHOULD

- **Location:** §2.3, sentence 5
- **Original:** "In this work, we adapt this architectural paradigm to multivariate time series, placing it within a contaminated semi-supervised framework where labeled anomalies actively guide training."
- **Issue:** "this architectural paradigm" is a vague referent ("paradigm" is not a technical term; the specific architecture is the asymmetric Teacher–Student MAE). "Actively guide training" uses "actively" as an empty intensifier — it pairs with "guide" in a way that adds no information. The corpus never uses "actively" as an intensifier in method description. "Placing it within a contaminated semi-supervised framework" is a mild passive-construction inflation (placed by whom, through what mechanism).
- **Fix:** Replace "paradigm" with the specific architecture name; remove "actively."
- **Revised sentence:** "In this work, we adapt this asymmetric Teacher–Student masked autoencoder design to multivariate time series, embedding it within the contaminated semi-supervised framework described in Section 3.1, where labeled anomalies guide training through the three pathways of Section 3.3–3.5."

---

### §3 Methodology (55 prose sentences)

---

**[M-01]** SHOULD

- **Location:** §3.1, sentence 5 (about labeled anomaly events arising naturally)
- **Original:** "In practice, labeled anomaly events arise naturally from the operational logs of industrial systems — fault and attack records that document anomalies as correlated deviations across multiple sensor channels — making the recovery of multi-channel correlation structure the central learning challenge."
- **Issue:** Another em-dash double-interruption ("— fault and attack records that … — making the recovery of …"). The main clause is swallowed by the apposition. "Making the recovery of multi-channel correlation structure the central learning challenge" is a vague framing: "central learning challenge" is a soft, unanchored claim about what the challenge is; the corpus names the challenge with a specific technical descriptor ("complex temporal dependencies and inter-variable correlations" — MEMTO abstract). SHOULD because the em-dash overuse is a recurring style problem (B.2).
- **Fix:** Restructure to avoid nested dashes; name the challenge with a technical descriptor.
- **Revised sentence:** "In practice, labeled anomaly events arise naturally from the operational logs of industrial systems: fault and attack records that document anomalies as correlated deviations across multiple sensor channels. Recovering the normal multi-channel correlation structure — and distinguishing it from anomalous deviations — is therefore the central modeling challenge."

---

**[M-02]** FLAG

- **Location:** §3.2, sentence 1 (architecture overview)
- **Original:** "CSMAD comprises five functional blocks (Figure 2): a linear patch embedding, a shared Transformer encoder, a Teacher decoder, a Student decoder, and a training-only label-guided module that couples the Student branch to a window-level anomaly classifier through gradient reversal."
- **Issue:** Precise, well-structured architecture overview sentence. The colon-enumeration is appropriate here (corpus: "Our MAE approach is simple: we mask random patches … and reconstruct …" MAE). The 5-item list is of concrete components, not abstract nouns (R-07 does not apply). OK as-is — FLAG only because "functional blocks" is a slight abstraction above "components" (which the corpus uses). Acceptable.

---

**[M-03]** MUST-FIX

- **Location:** §3.2, sentence 3
- **Original:** "The Student and GRL branches read the encoder output through a stop-gradient, so the encoder is optimized exclusively by the Teacher's reconstruction objective and the adversarial signal cannot corrupt the normal-pattern representation underpinning the anomaly score."
- **Issue:** "the normal-pattern representation underpinning the anomaly score" — "underpinning" is not a corpus-attested technical verb for this relationship (Appendix A collocations use "characterizes", "captures", "represents", "forms"). "Underpinning" is a vague spatial metaphor used in AI-generated academic prose. Additionally the sentence conflates two claims with "and" (encoder optimization scope + signal isolation), which could be separated for clarity.
- **Fix:** Replace "underpinning" with a precise technical verb; split the sentence.
- **Revised sentences:** "The Student and GRL branches read the encoder output through a stop-gradient, so the encoder is optimized exclusively by the Teacher's reconstruction objective. This stop-gradient ensures that the adversarial signal from the GRL cannot modify the representation on which the anomaly score is based."

---

**[M-04]** SHOULD

- **Location:** §3.3, anomaly-priority masking paragraph, sentence 2
- **Original:** "This addresses a structural imbalance of contaminated training: anomalous patches are rare, so stochastic masking seldom selects them and the model learns to reconstruct *around* rather than *through* them."
- **Issue:** "This addresses a structural imbalance of contaminated training:" — the colon-definition opener again (P-14 adjacent). "Structural imbalance" is acceptable here (it is a real term in the imbalanced-learning literature). The sentence is otherwise precise. The *around* / *through* metaphor is evocative but not a standard TSAD expression; flagging as SHOULD because the metaphor is informal rather than AI-generated.
- **Fix (optional):** Replace the metaphor with a direct technical statement.
- **Revised sentence:** "This addresses a structural imbalance of contaminated training: anomalous patches are rare, so stochastic masking seldom selects them and the model gains little experience reconstructing anomalous correlation patterns."

---

**[M-05]** SHOULD

- **Location:** §3.4, "Why the capacity gap matters" paragraph, sentence 1
- **Original:** "A deeper Teacher faithfully learns the joint normal correlation structure; the shallower Student replicates it on recurring normal patterns but fails more severely on the atypical correlation patterns characterizing anomalies than a matched-capacity decoder would (quantified in Appendix B.5), so the output discrepancy carries a stronger anomaly signal than reconstruction error alone."
- **Issue:** "faithfully learns" — "faithfully" is an unanchored adverb (not corpus-attested for "learn"; the corpus uses "accurately reconstruct" (MEMTO), not "faithfully learn"). The sentence is long (two semicolons, one parenthetical) but the technical content is precise. Flagging "faithfully" as SHOULD.
- **Fix:** Replace "faithfully learns" with "accurately learns" or "learns to capture."
- **Revised sentence:** "A deeper Teacher accurately captures the joint normal correlation structure; the shallower Student replicates it on recurring normal patterns but fails more severely on the atypical correlation patterns that characterize anomalies than a matched-capacity decoder would (quantified in Appendix B.5), so the output discrepancy carries a stronger anomaly signal than reconstruction error alone."

---

**[M-06]** SHOULD

- **Location:** §3.5, "Why gradient reversal is necessary" paragraph, sentence 3
- **Original:** "Over repeated exposure to labeled anomalies during training, the Student can learn to exploit this contextual signal to reconstruct anomalous patterns — shrinking the discrepancy exactly where it is most informative."
- **Issue:** "shrinking the discrepancy exactly where it is most informative" is a metaphorical em-dash clause appended for rhetorical emphasis. "Most informative" is a vague comparative without a referent. The sentence itself is mechanistically precise; only the em-dash dramatization is weak style.
- **Fix:** Replace the dash clause with a relative clause.
- **Revised sentence:** "Over repeated exposure to labeled anomalies during training, the Student can learn to exploit this contextual signal to reconstruct anomalous patterns, thereby reducing the discrepancy where it is most diagnostic."

---

**[M-07]** FLAG

- **Location:** §3.6, "Averaging across overlapping windows …" (final sentence of section)
- **Original:** "Averaging across overlapping windows provides an ensemble effect that reduces single-window reconstruction-context variation."
- **Issue:** "provides an ensemble effect" is a standard technical description (analogous to ensemble scoring). "Single-window reconstruction-context variation" is slightly verbose ("context variation" = variance introduced by which context tokens are visible). Acceptable as-is; FLAG for "context variation" which could be specified more precisely.

---

### §4 Experiments (38 prose sentences, not counting table captions)

---

**[E-01]** SHOULD

- **Location:** §4.1.1, "Contaminated benchmark protocol", sentence 2 (re-split rule)
- **Original:** "We therefore re-split each dataset at the temporal midpoint of its original test file: the earlier half joins the training data and the later half is reserved exclusively for evaluation, so labeled anomalies are genuinely present in training (ratios 0.52\%–6.20\%; SMD per-machine pending; Table 1)."
- **Issue:** "so labeled anomalies are genuinely present in training" — "genuinely" is an emphasis adverb with no technical content; it is the kind of word added to reassure a reader rather than to add information. Remove it.
- **Fix:** Delete "genuinely."
- **Revised sentence:** "We therefore re-split each dataset at the temporal midpoint of its original test file: the earlier half joins the training data and the later half is reserved exclusively for evaluation, so labeled anomalies are present in training (ratios 0.52\%–6.20\%; SMD per-machine pending; Table 1)."

---

**[E-02]** FLAG

- **Location:** §4.1.1, "SWaT dual evaluation", last sentence
- **Original:** "Table 2 ranks under excl22; the region-22 derivation and full-condition results are in Appendix §A.4."
- **Issue:** Clean, precise sentence. OK as-is.

---

**[E-03]** SHOULD

- **Location:** §4.1.3, sentence 3 (five metrics span three perspectives)
- **Original:** "The five metrics span three orthogonal perspectives — threshold-free ranking (VUS), tolerance-spectrum integration (PA\%K-AUC), and local event localization (Affiliation F1) — with distinct failure modes; reporting all five prevents any single failure mode from going undetected."
- **Issue:** The em-dash double-interrupt with a 3-item list inside, followed by a semicolon continuation, is structurally identical to the I-06 and I-09 patterns already flagged. "With distinct failure modes" attaches ambiguously — does it modify the three perspectives, or is it a new clause? The phrase "prevents any single failure mode from going undetected" is a slightly empty result promise (R-08 adjacent — no evidence cited). SHOULD.
- **Fix:** Split the sentence; attach "distinct failure modes" to the right referent.
- **Revised sentences:** "The five metrics span three orthogonal perspectives: threshold-free ranking (VUS), tolerance-spectrum integration (PA\%K-AUC), and local event localization (Affiliation F1). Each perspective has distinct failure modes; reporting all five ensures that no single failure mode dominates the comparison."

---

**[E-04]** SHOULD

- **Location:** §4.2, sentence 1 (main results paragraph)
- **Original:** "Table 2 presents PA\%K-AUC F1 and VUS-PR for CSMAD and all 26 baselines across the six dataset families; full five-metric results are in Appendix §A.5 and per-entity results in Appendix §A.6."
- **Issue:** This is a standard table-reference sentence (corpus: "We extensively evaluate our model on five real-world datasets with ten competitive baselines." AnomTr §4). Acceptable. FLAG only because "Table 2 presents … for CSMAD and all 26 baselines across the six dataset families" is slightly list-heavy — the table itself makes this clear. No mandatory fix.

---

**[E-05]** MUST-FIX

- **Location:** §4.2, result-claim sentence (CSMAD achieves the highest…)
- **Original:** "CSMAD achieves the highest PA\%K-AUC F1 on [N] of the six dataset families and the highest VUS-PR on [N] <!-- PH:NUM-006 -->, averaging [X.XX] PA\%K-AUC F1 and [X.XX] VUS-PR <!-- PH:NUM-007 --> across families, and outperforms the strongest unsupervised competitor (Q3) by [X.XX] <!-- PH:NUM-008 --> absolute points in PA\%K-AUC F1 and [X.XX] <!-- PH:NUM-009 --> in VUS-PR on average."
- **Issue:** The sentence is correct in structure — it uses the corpus pattern "achieves the highest … on [N] of [M] benchmarks" and pairs absolute-point margins with named comparators. The MUST issue is not the content but "and outperforms … by [X.XX] absolute points … and [X.XX] … on average" — the second "and" conjunction creates a three-clause compound ("achieves … [N]; and the highest … [N]; averaging … [X.XX]; and outperforms … [X.XX]") that reads as a run-on sentence. The corpus keeps "achieves" and "outperforms" as separate sentences (AnomTr: "Anomaly Transformer achieves the consistent state-of-the-art on all benchmarks. … TranAD outperforms the baselines (in terms of F1 score) for all datasets except MSL …"). Split this.
- **Fix:** Break into two sentences at "and outperforms."
- **Revised sentences:** "CSMAD achieves the highest PA\%K-AUC F1 on [N] of the six dataset families and the highest VUS-PR on [N] <!-- PH:NUM-006 -->, averaging [X.XX] PA\%K-AUC F1 and [X.XX] VUS-PR <!-- PH:NUM-007 --> across families. It outperforms the strongest unsupervised competitor (Q3) by [X.XX] <!-- PH:NUM-008 --> absolute points in PA\%K-AUC F1 and [X.XX] <!-- PH:NUM-009 --> in VUS-PR on average."

---

**[E-06]** SHOULD

- **Location:** §4.4, "Why graceful degradation is expected", sentence 1
- **Original:** "Three structural properties support robustness: (i) anomaly-priority masking applies only to labeled patches, leaving the label-free reconstruction objective unaffected by which anomalies are labeled; (ii) the GRL term draws its positive supervision exclusively from labeled windows — batches without a labeled positive skip the term entirely — so unlabeled anomaly windows, treated as negatives, never inject an erroneous positive adversarial signal; and (iii) the base reconstruction error is label-independent, elevated wherever a patch deviates from normal correlation structure."
- **Issue:** "Three structural properties support robustness:" — again "support robustness" is an unanchored use of "robustness" (R-10); more importantly, this is the third "three X:" colon-list opener in the manuscript (cf. I-07, Eq.3 and the contribution list). The em-dash interruption in (ii) adds another instance of the recurring pattern. The three listed items themselves are precise and well-formed. SHOULD for the colon-opener and em-dash combination.
- **Fix:** Replace the colon-opener with a direct statement; move the em-dash content into a subordinate clause.
- **Revised sentence:** "Three structural properties bound this degradation. First, anomaly-priority masking applies only to labeled patches, leaving the label-free reconstruction objective unaffected by which anomalies are labeled. Second, the GRL term draws its positive supervision exclusively from labeled windows; batches without a labeled positive skip the term entirely, so unlabeled anomaly windows treated as negatives never inject an erroneous positive adversarial signal. Third, the base reconstruction error is label-independent, elevated wherever a patch deviates from normal correlation structure."

---

**[E-07]** FLAG

- **Location:** §4.5, sentence 1 (qualitative analysis setup)
- **Original:** "Figure 4 decomposes the CSMAD anomaly score for representative windows from [N] datasets; each panel shows four aligned traces — raw input with ground-truth anomaly regions shaded, Teacher reconstruction error, Teacher–Student discrepancy, and the combined score with the anomaly-ratio threshold."
- **Issue:** "four aligned traces — [list]" — em-dash + 4-item list. The dash is used as a colon substitute here; a colon would be standard (and more precise). Flagging as OK-FLAG; the content is fully precise.
- **Fix (optional):** Replace "—" with ":".

---

### §5 Conclusion (6 prose sentences)

---

**[C-01]** SHOULD

- **Location:** §5, sentence 1
- **Original:** "This paper addressed the underexplored setting in which training data contain a small fraction of labeled anomalies alongside a majority of unlabeled observations — common in industrial deployments yet unsupported by standard MTSAD benchmarks or unsupervised methods."
- **Issue:** "This paper addressed …" is an attested corpus opener for conclusion sections ("This paper studies …" AnomTr, "In this work, we proposed …" GDN). However "underexplored setting" is a soft hedge-adjective for claiming novelty — compare with the more precise "novel and practical scenario" (NRdet corpus). "Underexplored" is frequently an AI-generated adjective to avoid the stronger claim. Additionally the em-dash apposition ("— common in industrial deployments yet unsupported by …") is the recurring dash pattern.
- **Fix:** Replace "underexplored" with the specific claim; break the dash apposition.
- **Revised sentence:** "This paper addressed the contaminated semi-supervised setting, in which training data contain a small fraction of labeled anomalies alongside a majority of unlabeled observations. This setting is common in industrial deployments yet unsupported by standard MTSAD benchmarks or unsupervised methods."

---

**[C-02]** SHOULD

- **Location:** §5, sentence 5 (inference cost + future work)
- **Original:** "A notable limitation is the cost of leave-one-out inference — an approximately 50$\times$ increase in forward-pass computation relative to single-mask scoring; reducing this inference cost is a natural avenue for future work."
- **Issue:** "is a natural avenue for future work" — this is the R-11 pattern ("natural avenue" as conclusion filler). The corpus never uses this phrase; it uses either nothing (some papers end on the limitation) or a specific alternative ("Self-supervised learning in vision may now be embarking on a similar trajectory as in NLP." MAE — hedge + specific trajectory). "Natural avenue" adds no information about what the alternative would look like.
- **Fix:** State the specific direction briefly.
- **Revised sentence:** "A notable limitation is the cost of leave-one-out inference — an approximately 50$\times$ increase in forward-pass computation relative to single-mask scoring; amortized inference via learned masking schedules or sparse patch selection is a candidate direction for reducing this cost."

---

**[C-03]** SHOULD

- **Location:** §5, sentence 6 (last sentence)
- **Original:** "The graceful degradation toward the unsupervised limit also suggests extending CSMAD to fully unlabeled settings by disabling the gradient-reversal pathway."
- **Issue:** "The graceful degradation toward the unsupervised limit also suggests …" is a standard "future work" framing; "suggests extending" is slightly vague. The specific mechanism (disabling the GRL pathway) is already stated in the sentence, which is good. "Graceful degradation" has been used in the same paragraph of §4.4; repeating it here is fine. Flagging SHOULD for "also suggests extending" which is weaker than the corpus pattern "can be extended to …" or "motivates extending …."
- **Fix:** Use a more direct action verb.
- **Revised sentence:** "The degradation curve also motivates a fully unsupervised variant of CSMAD, obtained by disabling the gradient-reversal pathway, which may inherit the asymmetric Teacher–Student architecture without requiring any label input."

---

### Appendix A prose (25 prose sentences)

---

**[AP-A-01]** SHOULD

- **Location:** §A.1, "Training budgets and evaluation cadence", sentence 1
- **Original:** "Table A.2 states the per-group budgets disclosed in Section 4.1.2."
- **Issue:** "disclosed" — this word appears four times in the appendix in a self-referential metalanguage style ("disclosed in Section 4.1.2"; "disclosure" in Section 4.1.2 itself). "Disclosed" implies the budget is a secret being revealed, which adds a defensive tone not typical of the corpus. Prefer "reported," "listed," or "given."
- **Fix:** Replace "disclosed" with "reported."
- **Revised sentence:** "Table A.2 states the per-group budgets reported in Section 4.1.2."

---

**[AP-A-02]** SHOULD

- **Location:** §A.1, baseline implementations paragraph, sentence 3
- **Original:** "Each baseline retains the hyperparameters of its original implementation or publication preset (for example, NRdetector keeps its native window size of 100 rather than the 500 used by our pipeline); the random-score baseline is averaged over five independent runs (mean ± std), and all other methods are single runs."
- **Issue:** The semicolon-"and" continuation chains three clauses after a long parenthetical example. The sentence is accurate but structurally heavy. The "for example" inline is fine (corpus: "For instance, in a water treatment plant…" GDN); the complexity arises from chaining the example and the run-count policy in one sentence. SHOULD.
- **Fix:** Split after the parenthetical.
- **Revised sentences:** "Each baseline retains the hyperparameters of its original implementation or publication preset (for example, NRdetector keeps its native window size of 100 rather than the 500 used by our pipeline). The random-score baseline is averaged over five independent runs (mean ± std); all other methods are single runs."

---

**[AP-A-03]** FLAG

- **Location:** §A.2, "Score aggregation", sentence 1
- **Original:** "All metrics consume point-level scores obtained by mean-aggregation over all covering (window, patch) pairs (Eq. 6); the identical evaluation routine serves CSMAD and every baseline."
- **Issue:** "All metrics consume point-level scores" — "consume" is an informal/metaphorical verb for "are computed on"; the corpus uses "We use precision, recall and F1-Score over the test dataset" (GDN) — direct procedural verbs, not "consume." FLAG.
- **Fix (optional):** Replace "consume" with "are computed on" or "take as input."

---

**[AP-A-04]** SHOULD

- **Location:** §A.3, "Training-label semantics", sentence 2
- **Original:** "Labeled anomalies in our training splits therefore originate exclusively from the incorporated test prefixes."
- **Issue:** "originate exclusively from" is precise. The sentence is clean. Minor SHOULD for "therefore" — the logical chain is slightly circular here (the prior sentence defines the construction that makes this true, so "therefore" is accurate but the sentence adds no new information). The sentence is necessary for clarity; just noting the weak-inference "therefore."
- **Fix (optional):** Replace "therefore" with "consequently" or omit and fold into the prior sentence.

---

### Appendix B prose (12 prose sentences)

---

**[AP-B-01]** SHOULD

- **Location:** §B.1, paragraph 1
- **Original:** "The Q3 condition of the main comparison grants unsupervised baselines the most favorable use of the training labels (excision of contaminated regions). For completeness, Table B.1 reports the complementary Q1 condition, in which the same 22 unsupervised baselines train on the full contaminated stream without excision — quantifying how much unaddressed contamination costs each method family and contextualizing the training-volume asymmetry acknowledged in Section 4.1.4."
- **Issue:** "For completeness, …" is a boilerplate result-justification phrase; the corpus never uses it (0 occurrences in the 11 paper sample). The actual reason is stated in the em-dash continuation ("quantifying how much …"), which is the real motivation. Replace the filler opener with the direct purpose. Additionally the em-dash continuation appending two "and"-connected purposes is another instance of the recurring dash-continuation pattern.
- **Fix:** Remove "For completeness,"; state the purpose directly; remove the trailing dash clause.
- **Revised sentences:** "The Q3 condition of the main comparison grants unsupervised baselines the most favorable use of the training labels (excision of contaminated regions). Table B.1 reports the complementary Q1 condition, in which the same 22 unsupervised baselines train on the full contaminated stream without excision, quantifying how much unaddressed contamination costs each method family."

---

**[AP-B-02]** FLAG

- **Location:** §B.2, sentence 1
- **Original:** "Section 4.1.2 disclosed the asymmetric training budgets (500 / 50 / 10 epochs)."
- **Issue:** "disclosed" — same issue as AP-A-01. Replace with "reported."
- **Revised sentence:** "Section 4.1.2 reported the asymmetric training budgets (500 / 50 / 10 epochs)."

---

**[AP-B-03]** SHOULD

- **Location:** §B.2, sentence 2
- **Original:** "To assess whether this asymmetry materially affects the comparison, representative unsupervised baselines are re-trained at extended budgets — and CSMAD at a reduced budget — under the otherwise unchanged protocol."
- **Issue:** "materially affects" is acceptable technical language. The em-dash double interruption ("— and CSMAD at a reduced budget —") is the same dash pattern flagged throughout. "Otherwise unchanged protocol" is a minor wordy phrase ("same protocol" suffices).
- **Fix:** Move the CSMAD reduced-budget point out of the dash; replace the final phrase.
- **Revised sentence:** "To assess whether this asymmetry materially affects the comparison, representative unsupervised baselines are re-trained at extended budgets and CSMAD at a reduced budget, under the same protocol."

---

### Appendix C prose (8 prose sentences)

---

**[AP-C-01]** FLAG

- **Location:** §C.3, pseudocode caption
- **Original:** "[ALG-C1] … content spec: preprocessing (incl. SWaT constant-column removal), anomaly-priority masking, Teacher-only gating for epochs < 250, loss assembly (Eq. 3) with adaptive weights (Eq. C.4) and reversal schedule (Eq. C.1). Full spec in PLACEHOLDER_REGISTRY.md."
- **Issue:** This is a placeholder comment, not prose to be published. No style flag applicable.

---

**[AP-C-02]** FLAG

- **Location:** §C.1, "Unlike the standard focal loss …" sentence
- **Original:** "Unlike the standard focal loss \cite{…}, which defines its modulating probability $p_t$ from the raw prediction, here $p_t := e^{-\ell_i}$ derives from the pos-weight-adjusted BCE, weighting hard examples by both confidence and prior imbalance; this variant is part of the present design rather than an external import."
- **Issue:** "rather than an external import" — "external import" is an unusual phrase (import is a software term). Standard academic phrasing would be "an external reference" or "a prior published method." Flag as OK-FLAG.
- **Fix (optional):** Replace "external import" with "a prior published method."

---

## III. Cross-Manuscript Pattern Counts

### Em-dash clause-joining (B.2 자제)

Instances of em-dash used to splice independent or semi-independent clauses (not for isolated
parenthetical terms):

| Location | Excerpt |
|----------|---------|
| I-01 | "sensor streams — water treatment plants … all depend on" |
| I-04 | "structurally embedded: … — the best a label-aware variant …" |
| I-06 | "by construction … — benchmark studies … ; evaluating …" |
| M-01 | "industrial systems — fault and attack records … — making the recovery …" |
| M-06 | "anomalous patterns — shrinking the discrepancy …" |
| RW-04 | "contaminated setting — that arises naturally … — these methods" |
| RW-10 | "adversarially — through gradient reversal — into the gradient" |
| E-03 | "three orthogonal perspectives — threshold-free … — with distinct" |
| E-06 | "(ii) … GRL term … — batches without a labeled positive …" |
| C-02 | "leave-one-out inference — an approximately 50× …" |
| AP-B-03 | "— and CSMAD at a reduced budget —" |

**Count: 11 em-dash clause-splicing instances.** The corpus standard (B.2) recommends ≤2 per
section. This manuscript has a systematic over-use pattern — the em-dash is functioning as a
low-effort continuation device throughout. A global pass replacing clause-splicing dashes with
periods or restructured sentences is warranted beyond the individual fixes above.

### "Robust" / "robustness" without metric anchor

| Location | Phrase |
|----------|--------|
| A-02 | "maintains robust detection" |
| H-01 | "stays robust to label sparsity" |
| I-10 | "confirming robust detection toward the fully unsupervised limit" |
| E-06 | "Three structural properties support robustness" |

**Count: 4 unanchored "robust" uses.** A-02, H-01, I-10 are flagged as SHOULD; E-06 as SHOULD
(the structural properties are listed, which partially anchors the claim — borderline).

### Colon-definition inflation ("X is Y: [elaborate]" for abstract properties)

Instances where a colon is used to launch a definition of a vague abstract property rather than
a concrete enumeration:

| Location | Pattern |
|----------|---------|
| I-03 | "This assumption is structurally embedded:" |
| I-04 | "This assumption is structurally embedded:" (same as I-03 — fixed there) |
| I-09 | "Relying only on (b) is insufficient:" |
| M-04 | "This addresses a structural imbalance of contaminated training:" |

These are distinct from legitimate colon-enumeration uses (M-02, A.2 metric definitions) which
are clean and corpus-consistent.

---

## IV. Sentence Count and Audit Statistics

| Section | Sentences audited |
|---------|-------------------|
| Abstract | 8 |
| Highlights | 5 |
| §1 Introduction | 35 |
| §2 Related Work | 40 |
| §3 Methodology | 55 |
| §4 Experiments | 38 |
| §5 Conclusion | 6 |
| Appendix A | 25 |
| Appendix B | 12 |
| Appendix C | 8 |
| **Total** | **187** |

Excluded from count: inline comment blocks (`<!-- … -->`), table row content, equation lines,
reference list, placeholder tokens, and section headers.

| Severity | Count |
|----------|-------|
| MUST-FIX | 11 |
| SHOULD | 19 |
| OK-FLAG | 8 |
| **Total detections** | **38** |

---

## V. Priority Fix List (MUST-FIX only, ordered by manuscript position)

| ID | Location | Core issue |
|----|----------|------------|
| I-01 | §1 ¶1 s1 | Em-dash clause-splice opening sentence |
| I-03 | §1 ¶2 s2 | "share an implicit assumption" — inflate; should be direct |
| I-04 | §1 ¶2 s3 | "structurally embedded" colon-launch + "leveraging information" buzzword |
| I-06 | §1 ¶3 s3 | Triple-clause compound with em-dash + semicolon |
| I-08 | §1 ¶4 s2 | Result claim without evidence ("amplifies … signals") stated as premise |
| A-01 | Abstract s3 | "loss bifurcation between … paths" obscures the operation |
| RW-06 | §2.2 s1 | 60-word PU-definition + 3-family taxonomy fused into one sentence |
| RW-08 | §2.2 ¶2 s1 | "deep representation learning informed by label signals" — vague collocation |
| M-03 | §3.2 s3 | "underpinning" — non-corpus metaphor verb |
| E-05 | §4.2 result sentence | Four-clause compound result sentence; needs split |
| C-01 | §5 s1 | "underexplored setting" — soft novelty hedge + em-dash apposition |
