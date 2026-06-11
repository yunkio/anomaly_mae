---
phase: 7
agent: latex-engineer
version: v1
last_modified: 2026-06-11
basis: MANUSCRIPT_v3.md (frozen source); p6_recheck_r2.md D-011 carry-overs (R2-M1..M3, R2-m1..m5)
policy: "All prose changes beyond layout conversion must be logged here. Sentence deletions,
         compressions, and new-sentence creation outside D-011 scope are prohibited."
---

# PROSE DIFF LOG — Phase 7 LaTeX conversion

## 1. D-011 carry-over changes applied (R2-m2 through R2-m5 + R2-M1..M3)

These are the 4 MINOR + 3 MAJOR items from p6_recheck_r2.md that were flagged as Phase 7
touch-ups.

### R2-M1 (MAJOR, §1) — Applied during initial LaTeX writing
- **Location**: sec1_intro.tex §1 para 4, sentence 3
- **v3 source**: "Exploiting all three simultaneously **is designed to amplify** both the
  reconstruction error and the Teacher--Student discrepancy at anomalous regions (Section 4.3)."
- **Conversion**: "**The design exploits** all three simultaneously to amplify both the
  reconstruction error and the Teacher--Student discrepancy at anomalous regions
  (Section~\ref{sec:ablation})."
- **Rationale**: R2-M1 fix — gerund subject + "is designed to" creates action/object mismatch.
  Subject changed to "The design" (a design-object), "exploits" makes it active.

### R2-M2 (MAJOR, §A.1) — Applied during initial LaTeX writing
- **Location**: appendix_A.tex, baseline implementations paragraph
- **v3 source**: "… nine detectors adopted from the protocol study of \cite{sarfraz2024quovadis}
  — five simple detectors (…), three lightweight neural detectors (…), and a GCN-LSTM detector
  **— six established** deep MTSAD systems (…), and seven recent methods (…)."
- **Conversion**: "nine detectors adopted from the protocol study of~\cite{sarfraz2024quovadis}
  (five simple detectors: …; three lightweight neural detectors: …; and a GCN-LSTM detector);
  **six established** deep MTSAD systems (…); and seven recent methods (…)."
- **Rationale**: R2-M2 fix — internal dash-pair closed by the third colon creating ambiguous
  parsing. Outer enumeration restored to semicolons; inner sub-list moved to parentheses.

### R2-M3 (MAJOR, Abstract) — Applied during initial LaTeX writing
- **Location**: main.tex frontmatter, abstract paragraph 4
- **v3 source**: "we **introduce** a contaminated benchmark protocol that incorporates …,
  thereby **introducing** labeled anomalies …"
- **Conversion**: "we **introduce** a contaminated benchmark protocol that incorporates …,
  thereby **exposing** labeled anomalies …"
- **Rationale**: R2-M3 fix — introduce/introducing echo in same sentence. "exposing" (audit
  original) restored.

### R2-m2 (MINOR, §2.3) — Applied during initial LaTeX writing
- **Location**: sec2_related.tex §2.3, sentence about TFMAE
- **v3 source**: "… constitute independent developments, **and** our design follows directly
  from vision MAE."
- **Conversion**: "… constitute independent developments, **whereas** our design follows
  directly from vision MAE."
- **Rationale**: R2-m2 — "and" weakens contrastive relationship; "whereas" restores it.

### R2-m3 (MINOR, §4.1.1) — Applied post-draft
- **Location**: sec4_experiments.tex, contaminated benchmark protocol paragraph
- **v3 source**: "4 of 81 channels are affected, with the largest shift **being** 166
  timesteps"
- **Conversion**: "4 of 81 channels are affected; largest shift: 166 timesteps"
- **Rationale**: R2-m3 — "with X being Y" absolute construction is colloquial; inline
  parenthetical form adopted.

### R2-m4 (MINOR, §A.2) — Applied post-draft
- **Location**: appendix_A.tex, Affiliation F1 paragraph
- **v3 source**: "Affiliation precision and recall **measure** the temporal proximity …,
  **converted into** per-event affinity scores …"
- **Conversion**: "Affiliation precision and recall **convert** temporal proximity …
  **into** per-event affinity scores …"
- **Rationale**: R2-m4 — "converted into" is a dangling participial modifying "proximity"
  rather than the subject; active voice "convert … into" repairs the attachment.

### R2-m5 (MINOR, §2.2) — Applied post-draft
- **Location**: sec2_related.tex §2.2, sentence about xue2022fewpositive/huang2022slavae
- **v3 source**: "neither model employs a masked-reconstruction self-distillation pretext
  **or** adversarial …"
- **Conversion**: "neither model employs a masked-reconstruction self-distillation pretext
  **nor** adversarial …"
- **Rationale**: R2-m5 — formal grammar requires "nor" under negation scope of "neither".

---

## 2. Layout-conversion changes (not prose changes — recorded for completeness)

These are structural transformations from Markdown to LaTeX that are inherently formatting-only
and do not alter scientific meaning or sentence content:

| Location | Transformation | Type |
|---|---|---|
| All `\S x.y` cross-references | Markdown §-links → `\ref{...}` | Layout |
| All `\cite{...}` | Unchanged (verbatim from v3) | None |
| All equations (1)–(6), C.1–C.5 | `$$...$$\tag{n}` → `\begin{equation}\label{}\end{equation}` | Layout |
| Markdown `---` section dividers | Removed (LaTeX sections implicit) | Layout |
| Markdown footnote `[^fn-id]:` | → `\footnote{...}` inline | Layout |
| Markdown bold `**text**` | → `\textbf{text}` | Layout |
| Markdown tables (appendix) | → `tabular` environments | Layout |
| `\mathbf{W} \in \mathbb{R}^{L \times F}` etc. | Verified identical to v3 source | None |
| HTML comments `<!-- ... -->` | Stripped (PH markers preserved as `% PH:` comments) | Layout |

---

## 3. Placeholder markers — conversion verification

All PH: markers from v3 are preserved as `% PH:` LaTeX comments adjacent to the `{[X.XX]}`
or `{[N]}` tokens. Total count verified:

| Type | v3 count | LaTeX count | Status |
|---|---|---|---|
| NUM placeholders (001–031) | 31 | 31 | Preserved |
| TXT placeholders (TXT-001, TXT-002) | 4 occurrences | 4 occurrences | Preserved |
| FIG markers (FIG-1..4, FIG-B1) | 5 | 5 | As TikZ boxes |
| TAB body markers (TAB-1..3) | 3 | 3 | As tabular skeletons |
| TAB appendix markers (A3,A6,A7,A8,B1..B4) | 8 | 8 | As tabular skeletons |
| ALG marker (ALG-C1) | 1 | 1 | As algorithm2e block |

---

## 3b. r2 (pdf-qa fix round, 2026-06-11) — layout-only changes, zero prose edits

The PDF QA r1 fix round touched no English sentence. For audit completeness, the only two
changes with any rendered-text effect are recorded here; both are citation/cross-reference
repairs that restore the manuscript's intended rendering, not prose edits:

| Location | Before (rendered) | After (rendered) | Class |
|---|---|---|---|
| sec3_method.tex §3.4, appendix_C.tex §C.1 | "**(author?)** [36]" (broken `\citet` under elsarticle-num) | "Ganin et al. [36]" (literal text + `\cite`) | Citation repair (QA MAJOR-3 prescribed form) |
| ~29 sites "Appendix~\ref{…}" | "Appendix **Appendix** A.3" | "Appendix A.3" (`\ref` alone; elsarticle emits the word) | Cross-reference de-duplication (QA MAJOR-4) |

All other r2 changes are float/format level: sidewaystable→`table*` (TAB-2), single-column→
`table*` promotions + `adjustbox` max-width caps (12 tables), `\allowbreak` insertions inside
table cells/math lists (no word changes), appendix counter resets (A.1/B.1/C.1 numbering),
hardcoded float numbers → `\ref` (Table A.1/A.3/A.4/B.4 ×5 sites), `algorithm`→`algorithm*`,
figure placement options `[t]`→`[tbp]`, `\BodyFloatBarrier` at §1/§3/§4 ends, placeholder box
heights to REGISTRY assumptions (FIG-2 5.0cm, FIG-3 4.0cm, FIG-4 3.5cm), and removal of the
empty affiliation fields (", , , , ," artifact). **Zero sentence additions, deletions, or
rewordings.**

## 4. Prohibition compliance

> Scope note (2026-06-11): the statements below describe the Phase 7 **conversion round**
> (v1) and the r2 pdf-qa fix round only. The subsequent D-013 compression round
> (orchestrator-directed, scoped) is exempt from the zero-compression policy and is fully
> logged in §5.

- **Zero sentence deletions** from MANUSCRIPT_v3.md body or appendix prose.
- **Zero sentence compressions** beyond natural markdown→LaTeX formatting.
- **Zero new scientific claims or sentences** invented.
- **All 7 D-011 items** (3 MAJOR + 4 MINOR) applied and recorded above.
- Numerical values: unchanged from v3 (all real values from protocol truth; placeholders
  `[X.XX]`/`[N]` untouched).

---

## 5. D-013 — prose-compression round (2026-06-11, prose-compressor)

**Authorization**: orchestrator directive D-013 (targeted ~0.3p recovery after r2 gate FAIL at
9.30p). This round is an explicit, scoped exemption from the §4 zero-compression policy (which
governed the Phase 7 conversion round only). Meaning and all mandated argumentation
(R13 protocol motivation, R28 excl22, R29 metric complementarity + PA-F1 critique, R30
threshold defense, R31 fairness, R32 robustness logic, R10 component reasoning, R21 lineage
footnote, D-008 scoping sentences) are untouched; all placeholder markers and all 48 \cite
keys verified intact post-edit.

**Result**: 5p body 9.30p → **8.997p** (−0.30p); preprint 49 → 46 PDF pages; words removed
(rendered prose) ≈ **219** + TAB-1 single-column reconversion (~0.1p) + float-column dead-space
fix (layout, §5.6).

### 5.1 §1 Introduction (sec1_intro.tex) — −80 words

| # | Before | After | Δw |
|---|---|---|---|
| 1.1 | "Real-world cyber-physical systems continuously generate high-dimensional sensor streams from water treatment plants, server clusters, and spacecraft telemetry arrays, all of which depend on reliable detection of anomalous states to prevent safety incidents and operational losses~\cite{schmidl2022evaluation,blazquez2021review}." | "Real-world cyber-physical systems --- water treatment plants, server clusters, spacecraft telemetry arrays --- continuously generate high-dimensional sensor streams whose reliable anomaly detection prevents safety incidents and operational losses~\cite{…}." | −9 |
| 1.2 | "The resulting body of work spans four broad families: reconstruction-based methods, which flag samples whose reconstruction errors exceed a threshold~\cite{…}; prediction-based methods, which score deviations from forecast sensor readings~\cite{…}; association-discrepancy and contrastive methods, which exploit the structural gap between normal and anomalous attention patterns~\cite{…}; and backbone-based methods, which apply general-purpose temporal architectures or auxiliary training objectives directly to detection~\cite{…}. Despite their differences, all four families treat the training data as drawn entirely from normal operations. Consequently, these methods have no mechanism for exploiting labeled anomalies even when such labels are available. The best a label-aware variant can do is exclude confirmed anomaly windows from training, filtering contamination rather than learning from it~\cite{wang2025nrdetector}." | "The resulting body of work spans four broad families --- reconstruction-based~\cite{…}, prediction-based~\cite{…}, association-discrepancy and contrastive~\cite{…}, and backbone-based~\cite{…} methods (Section~\ref{sec:related_mtsad}) --- that, despite their differences, all treat the training data as drawn entirely from normal operations. These methods consequently have no mechanism for exploiting labeled anomalies even when such labels are available: the best a label-aware variant can do is exclude confirmed anomaly windows from training, filtering contamination rather than learning from it~\cite{wang2025nrdetector}." (per-family one-line definitions live in full form in §2.1; pointer added; R11 anchor sentence preserved verbatim) | −44 |
| 1.3 | "These labeled anomalies are an obstacle for unsupervised methods --- a source of contamination --- but a valuable learning signal for semi-supervised ones." | "These labeled anomalies are contamination for unsupervised methods but a valuable learning signal for semi-supervised ones." | −5 |
| 1.4 | "…anomaly labels, typically derived from recorded fault and attack events…" | "…anomaly labels, typically from recorded fault and attack events…" | −1 |
| 1.5 | "…we propose \textbf{CSMAD} (…), a single end-to-end framework that integrates labeled anomaly information directly into the representation learning of a masked autoencoder." | "…we propose \textbf{CSMAD} (…), an end-to-end framework that integrates labeled anomaly information directly into masked autoencoder representation learning." (matches abstract phrasing; claim unchanged) | −4 |
| 1.6 | "Section~\ref{sec:related} reviews related work; Section~\ref{sec:method} describes CSMAD; Section~\ref{sec:experiments} presents experimental results; Section~\ref{sec:conclusion} concludes." | (deleted — pure navigation roadmap, duplicated by headings; the only whole-sentence deletion of this round besides 2.3b) | −17 |

### 5.2 §2 Related Work (sec2_related.tex) — −89 words

| # | Before | After | Δw |
|---|---|---|---|
| 2.1 | "Deep learning approaches to unsupervised MTSAD fall into several well-defined families. Reconstruction-based methods train an encoder--decoder to reproduce normal input and flag inputs with large reconstruction errors~\cite{…}. Prediction-based methods model the expected next state from history and score deviations from the forecast~\cite{deng2021gdn}." | "Deep learning approaches to unsupervised MTSAD fall into several well-defined families: reconstruction-based methods flag inputs that an encoder--decoder trained to reproduce normal data reconstructs poorly~\cite{…}; prediction-based methods score deviations from a forecast of the expected next state~\cite{deng2021gdn}." | −5 |
| 2.2 | "A more recent strand exploits association structure: transformer models that learn temporal dependencies~\cite{xu2022anomalytransformer} or contrast multi-scale views of the series~\cite{yang2023dcdetector} score the discrepancy between learned and observed patterns, and frequency-domain reconstruction has been extended with explicit channel-correlation discovery~\cite{wu2025catch}." | "A more recent strand scores the discrepancy between learned and observed association structure, via transformer-learned temporal dependencies~\cite{…} or multi-scale contrastive views~\cite{…}; frequency-domain reconstruction has been extended with explicit channel-correlation discovery~\cite{…}." | −8 |
| 2.3a | "All of these families, however, treat the training data as predominantly or entirely normal. When the training stream contains confirmed anomalous events (the contaminated setting of Section~\ref{sec:problem_formulation}, arising naturally from operational logs), these methods cannot distinguish known-anomalous from known-normal samples; labeled information is consequently either discarded or treated as noise that corrupts the learned normality model~\cite{wang2025nrdetector}." | "All of these families, however, treat the training data as predominantly or entirely normal: when the training stream contains confirmed anomalous events (the contaminated setting of Section~\ref{sec:problem_formulation}), labeled information is either discarded or treated as noise that corrupts the learned normality model~\cite{wang2025nrdetector}." ("arising naturally from operational logs" stated in §1 and §3.1) | −18 |
| 2.3b | "The present work addresses this structural limitation by integrating labeled anomaly information directly into representation learning rather than relying on post-hoc removal." (intermediate form: "The present work instead integrates…") | (deleted — transition duplicate of §1 closing and §2.2 R1-01 differentiation sentence, both retained) | −19 |
| 2.4 | "Positive and Unlabeled (PU) learning formalizes the scenario in which a learner has confirmed positive examples and a pool of unlabeled data that may contain additional positives~\cite{…}. Established solution families include cost-sensitive risk minimization via non-negative risk estimators~\cite{kiryo2017nnpu}, class-prior-based probability correction~\cite{elkan2008pu}, and two-step techniques that first extract reliable negatives before training a classifier~\cite{bekker2020pusurvey}. Outside time series, these techniques have been adapted to anomaly detection through deviation networks with scarce labeled anomalies~\cite{pang2019devnet} and deep semi-supervised anomaly detection objectives~\cite{ruff2020deepsad}." | "Positive and Unlabeled (PU) learning formalizes learning from confirmed positive examples plus unlabeled data that may contain additional positives~\cite{…}; established solutions include non-negative risk estimators~\cite{kiryo2017nnpu}, class-prior-based probability correction~\cite{elkan2008pu}, and two-step techniques that extract reliable negatives before training a classifier~\cite{bekker2020pusurvey}. Outside time series, these techniques have been adapted to anomaly detection through deviation networks with scarce labeled anomalies~\cite{pang2019devnet} and deep semi-supervised objectives~\cite{ruff2020deepsad}." (all 6 keys retained; attributions unchanged) | −16 |
| 2.5 | "…annotations~\cite{…}; these approaches treat the label as the sole supervision signal…" | "…annotations~\cite{…}, treating the label as the sole supervision signal…" | −2 |
| 2.6 | "Two earlier semi-supervised models address the label-scarce multivariate time-series setting: … and a semi-supervised variational autoencoder coupled with an active-learning labeling loop~\cite{huang2022slavae}. In both, … neither model employs a masked-reconstruction…" | "Two earlier semi-supervised models address label-scarce multivariate time series: … and a variational autoencoder coupled with an active-learning labeling loop~\cite{huang2022slavae}. In both, … neither employs a masked-reconstruction…" ("semi-supervised" qualifier kept at sentence head) | −3 |
| 2.7 | "The closest precedent to our setting is NRdetector~\cite{…} … a pre-trained backbone derived from the WETAS architecture extracts temporal embeddings…" | "The closest precedent is NRdetector~\cite{…} … a pre-trained WETAS-derived backbone extracts temporal embeddings…" | −6 |
| 2.8 | "Our patch-based masking draws directly from this paradigm, adapting the spatial-domain approach to windows of multivariate sensor channels; similar masking-based reconstruction objectives…" | "Our patch-based masking adapts this spatial-domain paradigm to windows of multivariate sensor channels; similar masking-based reconstruction objectives…" (lineage clause "whereas our design follows directly from vision MAE" retained verbatim — R22/PARTIAL-14 fix untouched) | −5 |
| 2.9 | "Knowledge distillation has been applied to anomaly detection through teacher--student frameworks in which a student trained to match…" | "Knowledge distillation has been applied to anomaly detection through teacher--student frameworks: a student trained to match…" | −2 |
| 2.10 | "In this work, we adapt this asymmetric teacher--student masked autoencoder design to multivariate time series, embedding it within the contaminated semi-supervised framework described in Section~\ref{sec:problem_formulation}, …" | "We adapt this asymmetric teacher--student masked autoencoder design to multivariate time series, embedding it within the contaminated semi-supervised framework of Section~\ref{sec:problem_formulation}, …" (R21 lineage footnote attached to this sentence is byte-identical) | −5 |

### 5.3 §3 Methodology (sec3_method.tex) — −2 words

| # | Before | After | Δw |
|---|---|---|---|
| 3.1 | "Anomaly-priority masking is a training-time mechanism; at test time windows are scored under the deterministic leave-one-out masking of Section~\ref{sec:scoring}, with no label input." | "It is a training-time mechanism; at test time windows are scored under the deterministic leave-one-out masking of Section~\ref{sec:scoring}, with no label input." (antecedent = the mechanism named in the preceding sentence; B-7 phrasing "training-time mechanism; at test time … deterministic leave-one-out" preserved) | −2 |

### 5.4 §4 Experiments (sec4_experiments.tex) — −48 words (prose)

| # | Before | After | Δw |
|---|---|---|---|
| 4.1 | "…and seven recent competitive methods, including TFMAE, the time-series MAE variant discussed in Section~\ref{sec:related_mae}~\cite{…}." | "…and seven recent competitive methods, including TFMAE (Section~\ref{sec:related_mae})~\cite{…}." | −6 |
| 4.2 | "…; full five-metric results are in \ref{…} and per-entity results are in \ref{…}." | "…; full five-metric and per-entity results are in \ref{…} and \ref{…}." | −2 |
| 4.3 | "Formal definitions and computation details are in \ref{sec:appendix_metrics}." | "Formal definitions are in \ref{sec:appendix_metrics}." | −3 |
| 4.4 | "Without anomaly-priority masking, random masking only rarely selects anomaly patches, leaving the Teacher's reconstruction deficit at those positions largely unexploited; removal costs [X.XX] points on average." | "Without it, random masking rarely selects anomaly patches, leaving the Teacher's reconstruction deficit at those positions largely unexploited; removal costs [X.XX] points on average." (paragraph heading names the mechanism) | −2 |
| 4.5 | "Further ablations --- removing the feature-matching regularizer, removing the Teacher-only warmup, and a symmetric (2-layer/2-layer) decoder ---…" | "Further ablations --- removing the feature-matching regularizer or the Teacher-only warmup, and a symmetric (2-layer/2-layer) decoder ---…" | −1 |
| 4.6 | "The main protocol is the upper bound of label availability (every training anomaly region is labeled), whereas realistic deployments…" | "The main protocol labels every training anomaly region --- the upper bound of label availability --- whereas realistic deployments…" | −2 |
| 4.7 | "…retains labels (at region granularity, consistent with how operational logs record fault events) while the rest remain in training unlabeled…" | "…retains labels (region granularity, as operational logs record faults) while the rest remain in training unlabeled…" | −4 |
| 4.8 | "As $p$ decreases, the discrepancy pathway and the adversarial suppression weaken together: fewer labeled patches are prioritized for masking and fewer batches activate the GRL term, so the Student's residual capacity to reconstruct anomalous patterns grows. The label-independent reconstruction term, however, remains elevated at anomalous patches, bounding the degradation from below as the model approaches its purely reconstruction-driven mode (Section~\ref{sec:main_results}). This sweep differs from the label-noise sweep of~\cite{wang2025nrdetector}, which varies the rate of \emph{incorrect} segment labels rather than the rate at which true events are recorded at all." (+ "batches without a labeled positive omit the term entirely") | "As $p$ decreases, the discrepancy pathway and the adversarial suppression weaken together (fewer patches are priority-masked, fewer batches activate the GRL term) and the Student's residual capacity to reconstruct anomalous patterns grows; the reconstruction term, however, remains elevated at anomalous patches, bounding the degradation from below as the model approaches its purely reconstruction-driven mode (Section~\ref{sec:main_results}). This sweep differs from the label-noise sweep of~\cite{wang2025nrdetector}, which varies the rate of \emph{incorrect} segment labels, not the rate at which true events are recorded." (+ "batches without a labeled positive omit it entirely") — R32 3-property logic, ARG-02 covariation ("weaken together … bounding the degradation from below"), and B-8 property-(ii) semantics all preserved; "label-independent" qualifier remains where introduced (property 3) | −9 |
| 4.9 | "Figure~\ref{fig:decomp} illustrates the decomposition of the CSMAD anomaly score for representative windows from [N] datasets; each panel shows four aligned traces: raw input with ground-truth anomaly regions shaded, Teacher reconstruction error, Teacher--Student discrepancy, and the combined score with the anomaly-ratio threshold." | "Figure~\ref{fig:decomp} decomposes the CSMAD anomaly score for representative windows from [N] datasets into four aligned traces: raw input, Teacher reconstruction error, Teacher--Student discrepancy, and the thresholded combined score." (full per-trace descriptions remain verbatim in the adjacent complete caption; PH:NUM-028 marker untouched) | −13 |
| 4.10 | "The two components respond distinctly: reconstruction error is elevated wherever the input deviates from learned normal patterns regardless of event type, while the discrepancy captures the additional divergence that arises where the Student's limited capacity and adversarially suppressed representation fail to replicate the Teacher's output." | "The two components respond distinctly: reconstruction error rises wherever the input deviates from learned normal patterns regardless of event type, while the discrepancy captures the additional divergence where the Student's capacity-limited, adversarially suppressed representation fails to replicate the Teacher." | −6 |

### 5.5 TAB-1 (tab:datasets) cell/format abbreviation — caption kept complete-form

- `table*[t]` (full-width band) → single-column `table[t]` + `\scriptsize` + `\tabcolsep` 2pt + adjustbox cap (~0.1p band recovery).
- Headers: "Dataset family"→"Family", "\#Train pts"→"\#Train", "\#Test pts"→"\#Test", "Train AR (\%)"→"Train AR", "Test AR (\%)"→"Test AR" — the "(\%)" moved into the caption definitions ("anomaly ratio (\%)").
- SWaT cell: "19.05 (full) / 3.68 (excl22)" → "19.05\,/\,3.68$^{\dagger}$"; dagger defined in caption: "SWaT is evaluated under both full and excl22 conditions ($\dagger$: full\,/\,excl22)".
- All numeric values byte-identical; no rounding or thousands abbreviation applied.

### 5.6 Layout-only companion change (zero prose)

`main.tex` / `main_5p_measure.tex` / `main_3p_measure.tex`: added
`\floatpagefraction=.90, \topfraction=.92, \bottomfraction=.60, \textfraction=.07`.
Reason: after compression, FIG-3+FIG-4 formed a float-only column on 5p printed p9 with
`\@fpsep`/`\@fpbot` rubber stretch (~0.2p dead space) that absorbed the prose savings;
the raised float-page fraction forces them back to `[t]` placement with text beneath.
Placeholder box heights were NOT reduced (kept at REGISTRY assumptions — no measurement gaming).

### 5.7 Statistics, re-measurement, gate verdict

| Quantity | Value |
|---|---|
| Words removed (rendered prose) | ≈219 (§1 −80, §2 −89, §3 −2, §4 −48) |
| Whole-sentence deletions | 2 (§1 roadmap; §2.1 transition duplicate) — all other changes are compressions |
| TAB-1 | full-width band → single column (~0.1p) |
| 5p body endpoint (pdftotext -bbox) | §5 ends "…(to be released upon acceptance)." at printed **p9, right column, yMax 762.8pt** = 99.4% of the 84.8–766.8pt text block |
| **5p body measured** | 8 + 0.5 + 0.5×0.994 = **8.997p** (r2: 9.30p; Δ −0.30p) |
| Gate (8.5 ≤ body ≤ 9.0) | **PASS** (margin to upper bound ≈ 0.003p — endpoint sits on the last line of p9; fragile to any future text growth) |
| Builds | latexmk error-free ×3; 5p Overfull 1 (1.9pt output routine, pre-existing); preprint Overfull 10 (all prose lines ≤16.5pt, ≤ r2 set); pages 5p 21→19, preprint 49→46, 3p 26→25 |
| Integrity | `??` 0, "(author?)" 0, undefined refs 0 (3 builds); 48 unique \cite keys unchanged; PH markers unchanged (sec4 22, sec1 2, sec5 3, main 4, appendices 5); appendix pages visually verified — no overlap/truncation |

### §5.5 보완 기재 (미니 감사 r1 후속, 2026-06-11 orchestrator)

- TAB-1 단일컬럼 복귀 시 **Source 열(데이터셋 원논문 \cite 5건) 제거** — 동일 인용은 §4.1 본문 및 Table A.4에 보존되어 인용 무손실 (미니 감사 무손상 검증 일치). diff log 누락 보완.
- sec4 TAB-1 SMD 행 하드코딩 "\S A.3" → `Appendix~\ref{sec:appendix_dataset}` 환원 (미니 감사 권고).

### §5.6 측정 무결성 기록 (2026-06-11 orchestrator)
- 미니 감사 권고의 SMD 셀 `\ref` 환원은 float 배치 임계점 이동을 유발해 **원문 "(\S A.3)"로 재환원** (8.997p 측정 상태 보존). 하드코딩 "A.3" 라벨은 appendix 구조 변경 시 갱신 필요 — Phase 8 핸드오프 노트에 등재.
- 분량 판정 확정: 5p 빌드의 PDF p.1은 Highlights 별면(미산입), printed p.1(타이틀·초록)~p.9(Conclusion 종점, 우측 컬럼 ~97%) = **본문 ≈8.99p → R6 게이트 PASS (8.5 ≤ x ≤ 9.0)**.

---

## 6. D-014 (a) — Appendix B.2 선택-기회 비대칭 공개 보강 (2026-06-11, p8 spec-fixer)

**Authorization**: DECISION_LOG D-014 ②(a) — "best-epoch 선택-기회 비대칭(≈100 vs 10회)의 명시적
공개를 Appendix B.2에 보강 (본문 분량 무영향 — 0.003p 여유 보호; §7-3 미니 감사 경유)".
Appendix 한정 신규 2문장 — 본문(§1–§5) 무접촉.

### 6.1 추가 문장 (appendix_B.tex §B.2 lead 문단, 2문장)

| # | 신규 문장 (원문) | 삽입 위치 |
|---|---|---|
| 6-S1 | "Because every method is reported at its best evaluated epoch, the budget asymmetry also entails an asymmetry in selection opportunities: under the evaluation cadence of Section~\ref{sec:impl}, CSMAD is evaluated at 100 checkpoints (every 5 of 500 epochs), versus 50 and 10 for the weakly supervised and unsupervised baselines (every epoch)." | 기존 문장 1("Section~\ref{sec:impl} reported the asymmetric training budgets (500\,/\,50\,/\,10 epochs).") 직후 |
| 6-S2 | "These runs keep the evaluation cadence fixed, so the number of evaluated checkpoints scales with each budget and the sweep probes the selection-frequency effect together with the training-length effect." | 기존 문장 2("To assess whether this asymmetry … otherwise unchanged protocol.") 직후 — 문단 말미 |

### 6.2 미니 감사 3종 (신규 2문장 대상)

| 검사 | 판정 | 근거 |
|---|---|---|
| ① ai-phrasing | **PASS** | SENTENCE_CORPUS 부록 B 금지/자제 패턴 grep 0건 (delve/showcase/pivotal/…/Moreover 연쇄/em-dash/의인화/"It is worth noting" 전부 무검출 — 신규 문장 내 `---` 0개; 스캔 히트 2건은 인접한 LaTeX 주석 "% ---- Table B.2 placeholder ----"로 산문 아님). 구체 수치 결합 선언문 — corpus 양성 신호 B.3-1/4 부합 |
| ② plagiarism | **PASS** | 변별 n-gram 8종("selection opportunities", "evaluated checkpoints", "selection-frequency", "training-length", "evaluation cadence", "best evaluated epoch", "budget asymmetry", "asymmetry in selection") × 02_venue_study(corpus 105문장+dossier) + 04_references/library 52 cards 전체 grep — **0건 일치**. "best evaluated epoch"는 본문 §4.1.2 자기 원고 표현의 의도적 재사용(용어 일관성) |
| ③ method-truth | **PASS** | 100회 = [271c] metadata `timing.num_evals=100` 실측 (`config.eval_interval=5` × `num_epochs=500`); 10/50회 = EXPERIMENT_PROTOCOL_TRUTH r4 §④-3 (unsup 10ep·weak 50ep, baseline eval 매 epoch `baseline_common.py:943`); "cadence fixed + checkpoints scales with budget" = exp298/299 실측 (`eval_interval=5` 유지, `num_evals` 60/40 — budget 비례 ✓) + 명세 TAB-B2 ⑤(baseline 50/100 run 매 epoch eval 의무)와 정합. 수치 발명 0건 — 전부 프로토콜 상수 유도값 |

### 6.3 본문 무영향 검증 (재컴파일 2종)

| 항목 | 변경 전 | 변경 후 | 판정 |
|---|---|---|---|
| main.pdf (preprint) 총 페이지 | 46 | 46 | 불변 |
| main_5p_measure.pdf 총 페이지 | 19 | 19 | 불변 |
| 5p 본문 종점 (§5 "…acceptance).") | printed p.9 우측 컬럼 yMax 762.8pt (§5.7 기록) | printed p.9 (PDF p.10) 우측 컬럼, 단어 "ceptance)." yMax **762.842847pt** — 동일 좌표 | **본문 무영향 (8.997p 보존)** |
| 신규 문장 렌더 위치 | — | PDF p.15 (appendix §B.2 — 변경 전과 동일 페이지) | appendix 한정 |
| 빌드 오류 | 0 | 0 (`!` 라인 0, undefined ref 0, "??" 0) | PASS |
| Overfull | 5p 1건 / preprint 10건 (§5.7 기록) | 5p 1건 / preprint 10건 | 회귀 없음 |
