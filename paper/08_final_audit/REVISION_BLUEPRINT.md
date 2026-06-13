---
phase: 5
agent: revision-architect
directives: [R8, R10, R35, R17]
last_modified: 2026-06-13
inputs: [KBS_AIMS_SCOPE.md, REVISION_AUDIT.md, 271_CONFIG_TRUTH.md(r4), RESEARCH_SYNTHESIS.md, PAGE_BUDGET.md]
canonical_manuscript: paper/07_latex/{main,sec1_intro,sec2_related,sec3_method,sec4_experiments,sec5_conclusion,appendix_A,appendix_B,appendix_C}.tex + highlights.txt
method_truth: 271_CONFIG_TRUTH.md (r4) — single source; demotion changes PROMINENCE only, never factual existence
---

# CSMAD Revision Blueprint — Directives (1)–(4)

> A section-by-section, location-keyed edit plan. Every edit is **length-neutral or compressing**
> (body is at 8.997p in `main_5p_measure.tex`, top edge of the 8.5–9.0p R6 window — see §LENGTH).
> Every interaction/time-series claim is cross-checked against `271_CONFIG_TRUTH.md` r4 (citations
> in-line as §VI/§VIII). No performance number is ever written — all stay `[X.XX]`/`[N]` with their
> `PH:NUM-xxx` comments (A8). No copying of reference text; citations use `refs.bib` keys only.
> The three measurement twins (`main.tex`, `main_3p_measure.tex`, `main_5p_measure.tex`) must receive
> **identical** frontmatter edits or the R6 number drifts.

---

## 0. DECIDED CONTRIBUTION STRUCTURE (the spine; apply everywhere)

This is the single ordering that ALL banner locations must obey (abstract, highlights, Fig-1,
intro bullets, intro narrative, conclusion). It implements directives (2)+(3)+(4) at once.

| # | Contribution (claim it OWNS) | Directive served | KBS lever (KBS_AIMS_SCOPE §) |
|---|------------------------------|------------------|------------------------------|
| **C1** | **Problem + setting (practical).** The *contaminated semi-supervised* MTSAD setting — sparse recorded fault/attack labels coexist with unlabeled operational data — and a chronology-respecting benchmark protocol that exposes labeled anomalies absent from standard splits. | (3a) practical, (3 setting) | §2A practical significance + §2B prediction/warning systems (HIGHEST) |
| **C2** | **New mechanism (novelty — the headline).** To our knowledge, the **first** framework to integrate sparse anomaly labels into masked self-distillation by **adversarially suppressing anomaly information in the representation via gradient reversal** — labels shape the *representation itself*, unlike prior semi-supervised TS work that attaches labels to a generative/predictive loss (`xue2022fewpositive`, `huang2022slavae`) or delegates representation learning to a label-agnostic backbone (NRdetector, `wang2025nrdetector`). Anomaly-priority masking is folded in here as the enabling step. | (2 demote masking), (3a genuinely new) | §2C novelty-justified-by-problem + §2D knowledge-as-signal (HIGH) |
| **C3** | **Synergistic, time-series-aware design (interaction — the directive-4 fix).** An asymmetric Teacher–Student decoder whose discrepancy signal is *preserved and amplified by the interaction* of loss bifurcation and gradient-reversal suppression; co-designed around multivariate time-series structure (cross-sensor correlation in the patch token, windowed patches, chronological leakage-free evaluation). Components are **co-designed, not concatenated** — remove one and the discrepancy degrades. | (4 interaction), (4 TS-aware), (3a new framing) | §2C + §2D + §2E (HIGH) |
| **C4** | **Cross-domain evaluation (generality + rigor).** [N] datasets across industrial control, IT infrastructure, and spacecraft telemetry; [N] baselines; five complementary metrics; graceful degradation to the unsupervised floor as labels become sparse — evidence of robustness and deployability. | (3b practical value) | §2E generality + §2A deployability + §2F rigor (MEDIUM) |

**Why this ordering (justification).** The old bullet #2 ("Three-path label integration") was a
list, not a claim, and fronted the weakest item (masking); KBS explicitly *non*-rewards "incremental
loss-term tweaks presented as the headline" (§2 "does NOT reward"). The new spine makes each bullet
own a claim, leads with KBS's highest-weight lever (practical/deployment, C1) and its strongest
novelty hook (C2), and elevates the *interaction* to a first-class contribution (C3) — which is the
directive-4 deliverable and was previously invisible at bullet altitude. C4 names the differentiators
(cross-domain + graceful degradation) instead of the table-stakes word "extensive."

**Register guardrails (all four bullets + all prose).** "competitive with / comparable to," never
"we beat all"; keep the scoped qualifier "to our knowledge, the first … in the contaminated
semi-supervised MTSAD setting" on every novelty mention (already at `sec1_intro.tex:62`,
`sec2_related.tex:33`); no AI-tells (delve / showcase / pivotal / "In conclusion" / em-dash overuse);
no fabricated numbers. DO NOT claim (KBS §5): real-time/latency, interpretability, on-device,
security, or any numeric superiority margin. The ~50× leave-one-out inference cost stays disclosed
(`sec3_method.tex:251`, `sec5_conclusion.tex:25-28`).

---

## 1. ABSTRACT — `main.tex:82-112` (+ twins `main_3p_measure.tex`, `main_5p_measure.tex`)

The abstract already opens with the KBS-native practical hook (lines 83-89: "rarely met in
practice", "recorded fault events") — **keep and sharpen, do not weaken**. Two edits:

**Edit A1 (sentence at lines 90-94) — demote masking + kill "orthogonal" [H1, I3].**
Current: "...integrates labeled anomaly information directly into masked autoencoder representation
learning through **three orthogonal mechanisms: anomaly-priority masking**, a Student imitation loss
restricted to normal patches, and gradient-reversal suppression...".
Rewrite to (a) replace "three orthogonal mechanisms" with an *interaction* phrasing, (b) lead the
named novelty on the representation-level pair, (c) name masking last as the enabling step. Target
sentence (drafter may polish wording, must keep meaning + length):
> "...integrates labeled anomaly information directly into the masked-reconstruction objective: a
> Student imitation loss restricted to normal patches and, as the central novelty, gradient-reversal
> suppression that removes anomaly-specific information from the Student's representation, with
> anomaly-priority masking exposing the labeled positions on which the two act."

This keeps three mechanisms factually present (method-truth safe — all three are real label-entry
points, §VI ACTIVE rows) while reordering prominence and signalling interaction ("on which the two
act"), not orthogonality.

**Edit A2 (sentence at lines 95-98) — make the asymmetric decoder the CONSEQUENCE of the mechanisms [I3, flow-smooth D-1].**
Current sentence states the asymmetric decoder as a *separate* fact. Reconnect causally so the
abstract reads mechanisms → why the capacity gap yields a reliable signal. Suggested join:
> "These label-guided mechanisms work *because of* the asymmetric Teacher–Student decoder: the
> capacity-limited Student mimics the Teacher less faithfully on anomalous correlation patterns than
> on normal ones, so suppressing anomaly information keeps that discrepancy sharp and amplifies it
> under contaminated training."

Grounded: T3/S2 asymmetric decoder (§VIII Architecture), GRL suppression direction (§VIII GRL
Details, `model.py:129-144`), OD-normal-only (§VI, `loss.py:259-261`). "less faithfully on anomalous
correlation patterns" echoes the multivariate framing (cross-sensor correlation, T1/T2).

**Keep unchanged:** the protocol sentence (99-102), the cross-domain/competitive sentence (103-105,
keep "competitive" hedge), the graceful-degradation sentence (108-109), code-availability (110).
The `PH:NUM-001/002` comments stay.

**Length:** net zero (reorder + reconnect, no added clauses). Re-measure twins after edit.

---

## 2. HIGHLIGHTS — `main.tex:118-125` AND `highlights.txt` (+ twins) — KBS GFA: exactly 5 bullets, ≤85 chars each

**Edit H-1 (bullet 2, the masking-fronting bullet) [H2].**
Current (both files): "Three label paths: anomaly-priority masking, loss bifurcation, gradient
reversal." — this fronts masking as path #1. Replace with a **novelty/interaction-led** bullet that
keeps "gradient reversal" concrete (it is the indexable novelty) and drops masking from the banner:
> `Gradient-reversal suppression makes labels shape the representation, not just the loss.`
> (char count: 81 — within ≤85; verify in BOTH files.)

Alternative if drafter prefers the interaction angle:
> `Co-designed label paths interact to sharpen the Teacher-Student anomaly gap.` (75 chars)

Pick ONE; apply identically to `main.tex:120` and `highlights.txt:4`. The other four bullets stay:
- B1 (contaminated setting) — keep (`main.tex:119`).
- B3 (asymmetric Teacher–Student amplifies discrepancy) — keep (`main.tex:121`).
- B4 (contaminated benchmark / test prefixes) — keep (`main.tex:122`).
- B5 (competitive + graceful decay) — keep (`main.tex:123`).

**Result:** masking removed from highlights entirely (it is an implementation detail, not a headline
— directive 2); the five-bullet count, ≤85-char rule, and the `PH:NUM-003` comment are preserved.
Update the verification comment at `main.tex:116-117` to the new char counts.

**Length:** highlights are outside the 9p body count (PAGE_BUDGET §4); no body impact. Still keep
≤85 chars (KBS hard constraint).

---

## 3. KEYWORDS — `main.tex:131-134` — NO CHANGE

Already 6 keywords, masking-as-strategy is NOT a keyword (audit §A "Keyword line — no masking term").
The present set — `Multivariate time series \sep Anomaly detection \sep Semi-supervised learning \sep
Masked autoencoder \sep Asymmetric self-distillation \sep Gradient reversal` — already reflects the
desired prominence (gradient reversal + asymmetric self-distillation present; "masking" absent). Keep
exactly. Do not add a 7th (the comment at 129-130 documents why "Contaminated benchmark" was dropped
to hold 6 — preserve that constraint).

---

## 4. INTRODUCTION — `sec1_intro.tex`

### 4.1 Intro narrative — `sec1_intro.tex:8-25` (mostly KEEP; two targeted touches)

The narrative spine is strong and KBS-aligned: lines 8-13 (problem + unsupervised lineage + the
"can only filter, not learn" limitation), 15-19 (practical fault-log hook + benchmark label gap),
21-25 (the three-signal key observation + the necessity argument + NRdetector contrast). **Preserve
the practical framing and the NRdetector contrast — they are the C1/C2 anchors.**

**Edit N-1 (key-observation sentence, lines 21-25) — de-front masking, promote synergy [H6, C].**
Lines 21-25 are *the seed of the interaction story* and must stay, but signal (a)=masking should read
as the *enabling* signal while (b)/(c) carry the weight (the existing lines 23-25 already argue that
(b) alone is insufficient and (c) closes the pathway — that is the synergy). Minimal edit: keep the
three-signal enumeration but add a half-clause making the dependency explicit, e.g. after line 22 add
within the existing sentence budget: "...exploits all three together — (a) merely surfaces the
labeled positions so that (b) and (c) can act on them." No new sentence; fold into the existing
"exploits all three simultaneously" clause (line 22). This converts "simultaneously" (which reads as
*independent*) into "together … act on them" (which reads as *interacting*).
Grounded: §VIII Masking (anomaly-first surfaces positions), §VI OD-normal-only, §VIII GRL suppression.

**Keep:** the NRdetector sentence (25) and the "Relying only on (b) is insufficient … closes this
pathway" argument (23-24) verbatim — this is the citeable interaction/novelty contrast.

### 4.2 Figure 1 placeholder body + caption — `sec1_intro.tex:40-42` and `:55-57` [H3, H4]

**Edit F1.** Both the placeholder body (40-42) and the caption (55-57) list "anomaly-priority
masking, loss bifurcation, and gradient-reversal suppression" with masking FIRST. Reorder so the
representation-level mechanisms lead and masking is last (enabling step). New ordering (apply in BOTH
spots identically): "...through co-designed label paths — loss bifurcation and gradient-reversal
suppression, enabled by anomaly-priority masking — turning contamination into a learning signal."
"co-designed" + the em-dash-free recast supports C3 and matches the spine ordering. (Watch
em-dash budget: the caption already uses `---`; keep total reasonable, no overuse.)

### 4.3 "First" sentence — `sec1_intro.tex:61-63` — KEEP (it is the C2 anchor)

Lines 61-63 already state the scoped novelty correctly ("to our knowledge, the first architecture
combining masked-reconstruction self-distillation with gradient reversal to adversarially suppress
anomaly-specific information … in a contaminated semi-supervised MTSAD setting") and the prior-work
contrast (loss-attached labels, `xue2022fewpositive`, `huang2022slavae`). **Do not weaken the
qualifier.** This sentence becomes the prose home of C2; the bullet (below) restates it as a claim.

### 4.4 CONTRIBUTION BULLETS — `sec1_intro.tex:66-84` — full restructure to the §0 spine [H5, B.5]

Replace the four `\item` bodies with the C1–C4 claims. The `\begin{enumerate}` block and the four
`PH:NUM-004/005` comments stay (re-home the comments under C4). Concretely:

- **C1 bullet** (was #1, lines 67-69) — KEEP largely as-is; it already leads with the setting +
  protocol. Light touch: add the practical clause so it *owns* the deployment value, e.g. open with
  "Motivated by operational logs that record a few fault/attack events, we formalize the
  *contaminated semi-supervised* setting…". Title stays "Contaminated semi-supervised setting and
  benchmark protocol." (Grounded: §III split constants are real; practical motivation = intro 15.)

- **C2 bullet** (replaces old #2 "Three-path label integration", lines 71-76) — **retitle and
  recast.** New title: "**Label-driven representation suppression via gradient reversal.**" Body
  (one claim, not a list): "CSMAD is, to our knowledge, the first framework to integrate sparse
  anomaly labels into masked self-distillation by *adversarially suppressing anomaly information in
  the Student's representation* (gradient reversal), so the labels shape the representation rather
  than only attaching to a loss; anomaly-priority masking surfaces the labeled positions this acts
  on." Masking demoted to a trailing enabling clause — **factually present, not headline** (§VIII
  Masking, §VIII GRL Details). Keep the scoped "to our knowledge, the first."

- **C3 bullet** (NEW interaction bullet, absorbs old #2-as-system + old #3-architecture, lines 71-78)
  — Title: "**Synergistic, time-series-aware design.**" Body: "The asymmetric Teacher–Student decoder
  (3-layer Teacher, 2-layer Student) makes the output discrepancy the anomaly signal; loss
  bifurcation and gradient-reversal suppression *interact* to keep the capacity-limited Student poor
  at anomalies, preserving and amplifying that discrepancy — the components are co-designed, not
  concatenated. The design targets multivariate time-series structure: each patch token encodes
  cross-sensor correlations, and evaluation respects chronological order." (Grounded: T3/S2 §VIII;
  OD-normal-only §VI; GRL suppression §VIII; linear patch token over s×F §VIII Architecture +
  `sec3_method.tex:106-107`; chronological protocol §III.) This is the **directive-4 bullet**.

- **C4 bullet** (recast old #4 "Extensive empirical evaluation", lines 80-83) — Title:
  "**Cross-domain evaluation and label-sparsity robustness.**" Body names generality (industrial /
  IT / telemetry), [N] baselines, five metrics, and graceful degradation to the unsupervised floor.
  Drop the word "Extensive." Keep `PH:NUM-004/005`.

**Flow-smooth (D-2):** §3's subsection structure (masking / decoders / label-training) stays — it is
the correct *mechanical* decomposition. The intro now promises synergy; §3.2 (synthesis paragraph,
§7 below) delivers it; §3.3–3.5 give the parts. No §3 heading renames needed.

**Length:** old #2 was a 3-item list (~6 lines); new C2 (claim) + C3 (interaction) together must not
exceed old #2+#3 line count. C2 is shorter than old #2; C3 ≈ old #3 + one TS clause. Net target = old
#2+#3 ± 0. If C3 grows, trim C4 to a single sentence (it is the most compressible). See §LENGTH.

---

## 5. RELATED WORK — `sec2_related.tex` (light)

**Edit R-1 — `sec2_related.tex:32`** "three orthogonal mechanisms that shape what the model learns to
represent" → change "orthogonal" to "co-designed" (or "complementary") to stop signalling
non-interaction at this positioning-critical sentence. The surrounding contrast (NRdetector delegates
to a label-agnostic backbone; CSMAD puts labels in the encoder gradient, lines 30-33) is the C2
anchor — **keep verbatim**, including the "first end-to-end MTSAD model" scoped claim at line 33.

**Keep:** lines 25-28 (labels enter the pretext gradient, not a classification target; the
`xue2022fewpositive`/`huang2022slavae` loss-attached contrast; DACAD transfer contrast) — all are
C2-supporting and already correctly framed. No masking emphasis here (good).

---

## 6. METHOD — `sec3_method.tex`

### 6.1 Problem Formulation — `sec3_method.tex:8-39` — KEEP (strongest TS statement) [T3]

Lines 31-39 already state the multivariate rationale cleanly ("fault and attack records that document
anomalies as correlated deviations across multiple sensor channels"; "Recovering the normal
multi-channel correlation structure … is the central learning challenge"). **Preserve verbatim.** The
fix is to make the §3.2 synthesis and the C3 bullet *echo* this so the TS-aware rationale is not
stranded here. The footnote distinguishing "contaminated semi-supervised" from
contamination-resilient/-resistant (18-22) is load-bearing — keep.

### 6.2 SYNTHESIS / INTERACTION PARAGRAPH — `sec3_method.tex:87-97` (Overall Architecture) — **PRIMARY DIRECTIVE-4 EDIT** [I1, C.3]

This is the **single highest-leverage edit in the whole revision.** Lines 87-90 are a parts list
("CSMAD comprises five functional blocks: a linear patch embedding, a shared encoder, a Teacher
decoder, a Student decoder, and a training-only adversarial branch") — the canonical concatenation
symptom. Lines 91-97 already describe the stop-gradient isolation. **Add a 2-sentence synthesis at the
end of the subsection** (after line 97) stating the interaction explicitly:

> "These blocks are co-designed rather than stacked. The asymmetric capacity gap makes the
> Teacher–Student discrepancy informative only if the Student stays poor at anomalies; loss
> bifurcation (§\ref{sec:label_training}) removes the training pull toward anomalous patches, and
> gradient-reversal suppression closes the residual representational pathway by which the Student
> could still learn them, while anomaly-priority masking (§\ref{sec:masking}) surfaces the labeled
> positions on which both act. The discrepancy signal the score reads (§\ref{sec:scoring}) is thus
> preserved and amplified by the interaction of the components, not by any one alone."

**Length offset:** tighten the parts-list sentence (87-90) — e.g. drop the appositive re-description
of the adversarial branch (it is defined in §3.5) — to recover the 2 sentences added. Net ≈ 0.

**Grounded (every clause):** T3/S2 §VIII Architecture; OD on normal patches only, anomaly side zeroed
§VI + `loss.py:259-261`; GRL **suppresses** anomaly information (NOT "learns discriminative features"
— corrected direction, §VIII GRL Details, `model.py:129-144`); masking surfaces positions §VIII
Masking (`force_mask_anomaly=True`); score = recon + scaled-disc/4 §VIII Anomaly Score. **Wording
guard:** use "suppress anomaly information," never "learn discriminative features" (C.4 flag).

### 6.3 Patch embedding — `sec3_method.tex:102-108` — KEEP + reference from synthesis [T1]

Lines 106-107 ("Projecting an entire patch — s timesteps across all F channels — into a single token
encodes cross-channel correlations directly in the token") is the strongest existing TS-aware
sentence. **Keep verbatim**; the C3 bullet and §6.2 synthesis now lean on it for the
"cross-sensor correlation" claim.

### 6.4 Anomaly-priority masking paragraph — `sec3_method.tex:110-123` — KEEP factual, add ONE linking clause [D1]

This is the correct, factual home for masking (must stay accurate — it is genuinely 1 of 3 label-entry
points, §VIII Masking). It already reads as a mechanism addressing "a structural imbalance of
contaminated training," not a standalone selling point. **One light edit:** at the end of line 121,
append a half-clause linking it to the other paths: "…so that the discrepancy and suppression
mechanisms of §\ref{sec:label_training} have anomalous targets to act on." Supports synergy at zero
net length. Keep the masking-ratio formula (Eq. context), the `|M|=round(N×ρ)` definition (111), and
the leave-one-out/no-label-at-test note (122-123). Do NOT expand or re-emphasize.

### 6.5 "Why the capacity gap matters" — `sec3_method.tex:143-152` — add multivariate half-clause [T2]

Lines 145-146 already say the Student "fails more severely on the correlation patterns characteristic
of anomalies." Make the *why-MTS* explicit with a half-clause: the discrepancy is diagnostic
*precisely because MTS anomalies are cross-sensor correlation breaks, not single-channel spikes*, that
a shallow decoder cannot reproduce. Pull the rationale from intro line 9 ("correlated deviations
across multiple sensor dimensions"). One clause, no new sentence. Grounded: §VIII Architecture +
intro:9. Keep the self-distillation citation (150-152, `zhang2022selfdistill`, `ristea2024sdmae`).

### 6.6 Label-Guided Training opener — `sec3_method.tex:170-175` — recast to dependency chain + forward-ref [I2]

Line 173 ("Three loss components couple labeled anomaly information to the model at different levels")
reads as three *independent* levels. Recast the opener to frame them as a **dependency chain** and add
a one-clause forward-reference to the §3.5 necessity argument (lines 220-232) so the interaction is
visible before the equations:
> "Three loss components couple labeled anomaly information to the model as a dependency chain rather
> than independent terms: loss bifurcation removes the training pull toward anomalies, feature
> matching keeps the Student honest in hidden space, and gradient-reversal suppression closes the
> residual representational pathway that bifurcation alone leaves open (argued in the paragraph 'Why
> gradient reversal is necessary beyond loss bifurcation')."

**Flow-smooth (D-4):** keep the *full* argument at lines 220-232 (do NOT duplicate it) — this opener
gives only the one-clause forward-reference. Grounded: §VI OD-normal-only; §VIII GRL suppression;
necessity argument already in 220-232. Keep Eqs. (lod), (lfm), (lcls), (ltotal) and the GRL dual-λ
paragraph (160-168) verbatim — the λ_GRL / λ_rev distinction is method-truth (§VIII GRL Details r4),
do not collapse it.

### 6.7 Scoring — `sec3_method.tex:244-285` — KEEP + one TS-framing clause [T4]

The chronological/leave-one-out machinery is correct and load-bearing. **Light framing edit:** at the
point-level aggregation note (line 284, "ensemble effect"), label the overlap aggregation as
exploiting *temporal context* (one clause). The leave-one-out ~50× cost limitation (251) stays
disclosed. The score formula (Eqs. dscale, sigma, agg) is method-truth (§VIII Anomaly Score) — keep
exactly; recon + scaled-disc/4, c=4. Do not introduce `lambda_disc`/`recon+2·disc` (that is the
diagnostic CSV path, NOT the score — §VII #21).

---

## 7. EXPERIMENTS — `sec4_experiments.tex`

### 7.1 Ablation table + framing — `sec4_experiments.tex:340-385` — KEEP rows, reframe as MINOR component study [D4, D5]

**Keep the masking ablation row** (Row 3, `:357` "w/o anomaly-priority masking") and its discussion
paragraph (`:364-367`) — this is legitimate evidence and in fact *supports the synergy story*
(removing a component degrades the signal). Do NOT delete or expand.

**Reframe (one sentence, the ablation subsection topic sentence):** add/adjust the lead so the
ablation is framed as evidence for the **interaction**, not a parade of co-equal headline tricks. The
existing caption note (344-348) and Row-2 framing ("isolating the net effect of active adversarial
suppression") are good. Add at the subsection opening (before Table 3): "The ablations test whether
the components *interact* as designed: each removal should degrade the Teacher–Student discrepancy the
score depends on." This converts the ablation from "feature checklist" to "interaction evidence" (C3
support) without touching any number. The masking paragraph (364-367) stays as a *minor component*
study — secondary by location; do not promote its framing.

**No number edits anywhere** — Rows stay `[X.XX]`, `PH:NUM-021/022/023` intact.

### 7.2 Main-results / protocol framing — `sec4_experiments.tex` (§4.1.1 protocol, §4.2) — KEEP

Keep the protocol-precedent sentence (NRdetector also re-splits, audit B.4 flag) so the protocol does
not read as self-serving. Keep "competitive" hedging. The five-metrics "three orthogonal perspectives"
phrase at `:161` is a **legitimate** non-component use of "orthogonal" (metric families) — **do NOT
change it** (only the component-level "orthogonal" at abstract/intro/related/conclusion is the
problem).

### 7.3 Sparsity analysis — `sec4_experiments.tex:387-414+` — KEEP [D6]

The "three structural properties bound this degradation" argument (402-414) includes masking as one of
three graceful-degradation arguments (404-405) — factual and secondary; keep. This section is a C4
strength (label-sparsity robustness → deployability). No change beyond ensuring it is referenced by
the C4 bullet.

---

## 8. CONCLUSION — `sec5_conclusion.tex:8-32`

**Edit Con-1 (recap sentence, lines 12-17) — demote masking + kill "orthogonal" + state interaction [H7, I4].**
Current: "CSMAD integrates labeled anomaly information … through **three orthogonal paths** —
**anomaly-priority masking**, loss bifurcation …, and gradient-reversal suppression … — built on an
asymmetric Teacher–Student decoder … that converts the capacity gap into a reliable discrepancy
signal." Rewrite so (a) masking is no longer fronted, (b) "orthogonal" → interaction phrasing, (c) the
paths and the capacity gap *together* (not separately) produce the signal:
> "CSMAD integrates labeled anomaly information into masked autoencoder representation learning
> through co-designed, interacting paths — loss bifurcation that restricts Student mimicry to normal
> patches and gradient-reversal suppression of anomaly information, with anomaly-priority masking
> surfacing the labeled positions they act on — so that, together with the asymmetric Teacher–Student
> decoder (3-layer Teacher, 2-layer Student), the capacity gap becomes a reliable, amplified
> discrepancy signal under contaminated training."

This matches the §0 spine ordering uniformly (D-5 consistency). Grounded as in §6.2.

**Keep:** the practical "common in industrial deployments yet unaddressed" framing (8-11), the
benchmark-protocol sentence (18-19), the competitive/graceful-degradation results sentence (20-22,
with `PH:NUM-029/030`), the ~50× leave-one-out limitation (25-28), the unsupervised-variant future
work (29-30), code link (31). No AI-tell ("In conclusion" must NOT be introduced).

**Length:** recast is ≈ same length (it removes "three orthogonal paths" listing overhead and adds the
"together with … capacity gap" clause). Net ≈ 0.

---

## 9. MASKING-DEMOTION SUMMARY (the 7 HEADLINE spots, status after edits)

| # | Location | Action | After |
|---|----------|--------|-------|
| H1 | abstract `main.tex:90-94` (+twins) | reorder + "orthogonal"→interaction | masking named last, enabling step |
| H2 | highlights `main.tex:120` + `highlights.txt:4` (+twins) | replace bullet | **masking removed from highlights** |
| H3 | Fig-1 body `sec1_intro.tex:40-42` | reorder | masking last, "co-designed" |
| H4 | Fig-1 caption `sec1_intro.tex:55-57` | reorder (match H3) | masking last |
| H5 | intro contribution bullet `sec1_intro.tex:71-76` | retitle→C2/C3, recast | masking = trailing enabling clause |
| H6 | intro key-observation `sec1_intro.tex:21-22` | de-front (a), promote (b)/(c) | masking = enabling signal |
| H7 | conclusion recap `sec5_conclusion.tex:13` | reorder + "orthogonal"→interaction | masking surfaces positions, not headline |

**Method-truth preserved (factual masking, all KEPT):** D1 method paragraph (`sec3_method.tex:110-123`,
+1 linking clause), D2 Fig-2 mechanics (`:53,71`), D3 window-label coincidence (`:208`), D4 ablation
row (`sec4_experiments.tex:357`), D5 ablation paragraph (`:364-367`), D6 sparsity argument (`:404-405`),
D7 appendix config/pseudocode/Eq (`appendix_A.tex:38`, `appendix_C.tex:70-77,130`). Masking remains a
real, accurately described component — only its PROMINENCE drops (directive 2 + hard constraint).
"orthogonal" component-uses fixed at 4 spots (abstract, intro bullet, related:32, conclusion);
metric-perspective use at `sec4_experiments.tex:161` left intact.

---

## 10. INTERACTION / SYNTHESIS PARAGRAPH — WHERE IT GOES

**Primary synthesis paragraph:** `sec3_method.tex:97` (end of §3.2 Overall Architecture) — the new
2-sentence co-design paragraph in §6.2 above. This is the natural synthesis location (the parts list
was the gap) and the highest-leverage directive-4 edit.

**Supporting interaction statements (so the story is visible at every altitude):**
- Bullet **C3** (`sec1_intro.tex` contributions) — interaction as a first-class contribution.
- Intro key-observation (`sec1_intro.tex:22`) — "act on them" replaces "simultaneously."
- §3.5 opener (`sec3_method.tex:173`) — dependency-chain recast + forward-ref.
- Abstract (`main.tex:95-98`) — asymmetric decoder as the *consequence* of the mechanisms.
- Conclusion recap (`sec5_conclusion.tex:12-17`) — paths + capacity gap *together*.
- Ablation topic sentence (`sec4_experiments.tex` §4.3) — ablations as interaction *evidence*.

**Time-series-aware emphasis** lives at: problem formulation (`:31-39`, kept), patch-token cross-sensor
sentence (`:106-107`, kept + referenced), capacity-gap multivariate half-clause (`:145-146`, T2),
chronological/overlap framing (`:284`, T4), and echoed in C1/C3 bullets.

---

## 11. KBS-FIT REGISTER (apply throughout; from KBS_AIMS_SCOPE §4.4, §5)

**Lean into (legitimate, mapped to scope):** "applications-oriented," "operational logs,"
"monitoring / early-warning" (→ scope "prediction systems and warning systems"), "turns recorded
fault/attack events into a usable learning signal" (→ practical significance + the knowledge-based
hook), "exploits the sparse anomaly labels recorded in operational logs as a knowledge signal,"
"co-designed rather than concatenated," "the components interact," "designed around the characteristics
of multivariate time series — cross-sensor correlation, windowed patch reconstruction, and
chronological leakage-free evaluation," "competitive with state-of-the-art … across industrial, IT,
and telemetry domains," "degrades gracefully toward the unsupervised floor."

**Avoid (over-claim / wrong-journal / AI-tells):** unqualified "first ever" (always keep the scoped
"to our knowledge, the first … in the contaminated semi-supervised MTSAD setting"); "optimal," "we
solve," "we beat all"; "pure theory" framing; delve / showcase / pivotal / "In conclusion";
em-dash overuse. **Never claim** real-time/latency, interpretability, on-device, security, or any
numeric margin (no support in method truth or placeholder results).

---

## 12. LENGTH DISCIPLINE (R6 — body 8.5–9.0p, currently 8.997p @5p; top edge)

Every edit above is a **reorder, recast, or one-clause add with a matching offset**. Net target = 0.
Growth risks and their offsets:

1. **§3.2 synthesis (+2 sentences)** — offset by tightening the parts-list sentence `:87-90` (drop the
   adversarial-branch appositive). **Net ≈ 0.**
2. **Contribution C3 (interaction bullet)** — old #2 (3-item list) shrinks to C2 (1 claim); the saved
   lines fund C3. If C3 + C2 > old #2+#3, **trim C4 to one sentence** (most compressible — "extensive"
   removed anyway).
3. **§3.5 opener recast + forward-ref clause** — replace, do not append; net ≈ 0 (the full argument
   already exists at 220-232; only a one-clause pointer is added).
4. **Method half-clauses (D1, T2, T4)** — ~3 short clauses total; offset by the parts-list trim and by
   removing "three orthogonal mechanisms"/"three orthogonal paths" listing overhead in abstract +
   conclusion.

**If the body grows past 9.0p after edits, trim in this order** (PAGE_BUDGET §4 §9 compression ladder
analogue): (a) C4 bullet → single sentence; (b) ablation "Extended variants" paragraph
(`sec4_experiments.tex:382-385`) → fold into the caption note; (c) §4.2 analysis text → 150 words.
**Re-measure `main_5p_measure.tex` after the abstract + intro edits** (the frontmatter twins must be
edited identically — abstract A1/A2 and highlights H-1 in all three of `main.tex`,
`main_3p_measure.tex`, `main_5p_measure.tex` — or R6 drifts). Highlights/keywords/declarations are
outside the body count but the 5-bullet ≤85-char and 6-keyword KBS rules stay hard.

---

## 13. HARD-CONSTRAINT CHECKLIST (verify before handing to drafter)

- [ ] No performance number written; all `[X.XX]`/`[N]` + `PH:NUM-xxx` preserved (A8).
- [ ] Masking factually present in method (§6.4 D1 kept) + ablation (D4) + appendix (D7); demoted in
      prominence only at the 7 HEADLINE spots. Never removed.
- [ ] GRL described as **suppressing anomaly information**, never "learning discriminative features"
      (corrected direction, §VIII GRL Details r4).
- [ ] λ_GRL / λ_rev dual-λ distinction kept (§3.5, `:160-168`) — not collapsed.
- [ ] Score formula = recon + scaled-disc/4 (c=4); no `lambda_disc`/`recon+2·disc` (diagnostic CSV
      path, not the score — §VII #21).
- [ ] Scoped "to our knowledge, the first … contaminated semi-supervised MTSAD" qualifier kept on
      every novelty mention.
- [ ] 5 highlights ≤85 chars (both files), 6 keywords, 5 declaration sections, `journal{KBS}`, flat
      structure — untouched by these edits.
- [ ] Citations only from `refs.bib`; no near-paraphrase of reference text.
- [ ] No AI-tells; no em-dash overuse; `paper_legacy/` never opened.
- [ ] All three measure-twins edited identically for frontmatter; body re-measured 8.5–9.0p.
