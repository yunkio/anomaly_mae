---
phase: 5
agent: revision-auditor
directives: [R8, R10, R35]
last_modified: 2026-06-13
---

# CSMAD Manuscript Revision Audit — Directives (1)–(4)

> Canonical manuscript: `paper/07_latex/{main,sec1_intro,sec2_related,sec3_method,sec4_experiments,sec5_conclusion,appendix_A,appendix_B,appendix_C}.tex` + `highlights.txt`.
> Method truth: `271_CONFIG_TRUTH.md` (r4). Journal-fit truth: `08_final_audit/KBS_AIMS_SCOPE.md` (phase 8, 2026-06-13) + `KBS_FORMAT_REQUIREMENTS.md`.
> All result numbers stay `[X.XX]`/`[N]` placeholders (A8). Protocol constants (113/114 entities, 50% split, 5 metrics, region-22 83.75%, train AR 0.52–6.20%) are real.
> Body length: **8.997p** in `main_5p_measure.tex` (R6 window 8.5–9.0p, top edge) — every proposed edit is length-neutral or compressing. No additions of length.

This audit is a revision MAP, not a rewrite. Each item gives `file:line`, a classification, and a concrete proposed action. Directives (2)–(4) are addressed against the manuscript text; directive (1) (Notion readability) is a downstream publishing task noted in §E.

---

## (A) ANOMALY-PRIORITY MASKING OVER-EMPHASIS MAP

**Method truth (must stay accurate, never removed):** Anomaly-priority masking is genuinely one of the three points where labels enter config-271 training (`force_mask_anomaly=True`, `271_CONFIG_TRUTH.md` §VIII Masking; §VI ACTIVE row). It masks anomalous patches first within the 8-of-50 budget. Demotion changes only **prominence**, never factual presence. It remains: one method paragraph, Eq. (masking_rule), one architecture-figure mention, one ablation row, the config table, and the pseudocode comment.

Below, every occurrence in body + frontmatter, classified **HEADLINE** (must demote/remove from a banner position) vs **LEGITIMATE-DETAIL** (keep, ensure secondary).

### HEADLINE occurrences — demote / remove from banner position (7)

| # | file:line | Current text (excerpt) | Class | Proposed action |
|---|-----------|------------------------|-------|-----------------|
| H1 | `main.tex:92` (abstract) | "...three orthogonal mechanisms: **anomaly-priority masking**, a Student imitation loss restricted to normal patches, and gradient-reversal suppression..." | HEADLINE | **Demote (reorder).** Keep "three orthogonal mechanisms" but list the representation-level pair first; masking becomes the third (enabling) item. Lead the named novelty on self-distillation + gradient reversal. Mirror this reorder in `main_3p_measure.tex:91-92` and `main_5p_measure.tex:91-92` (measurement twins — must stay in sync or the R6 number drifts). |
| H2 | `main.tex:120` (highlights bullet 2) + `highlights.txt:4` | "Three label paths: **anomaly-priority masking**, loss bifurcation, gradient reversal." | HEADLINE | **Demote/recast.** This bullet fronts masking as path #1 of a KBS highlight (max 5 bullets). Recast around the interaction or the gradient-reversal novelty, e.g. an interaction-led bullet. Masking named last or folded. Keep ≤85 chars; update **both** `main.tex` and `highlights.txt` (separate submission file) and the two measure-twins (`main_3p_measure.tex:120`, `main_5p_measure.tex:120`). |
| H3 | `sec1_intro.tex:42` (Fig-1 placeholder body) | "...three paths --- **anomaly-priority masking**, loss bifurcation, and gradient-reversal suppression --- turning contamination into a learning signal." | HEADLINE | **Demote (reorder).** Fig-1 right-panel lists masking first. Reorder so representation-level mechanisms lead; masking last. |
| H4 | `sec1_intro.tex:56` (Fig-1 caption) | identical three-path list, masking first | HEADLINE | **Demote (reorder).** Same reorder as H3; caption and placeholder body must match. |
| H5 | `sec1_intro.tex:71-76` (contribution bullet #2 title + body) | Title: "**Three-path label integration** into masked autoencoder representation learning." Body lists "(i) *anomaly-priority masking*..." first. | HEADLINE | **Recast title + reorder.** Per directive (3)+(4), retitle this contribution toward the *new mechanism / synergy* (gradient-reversal suppression as the lead). Within the enumerated (i)/(ii)/(iii), masking may stay as one item but should not lead; consider ordering (i) loss bifurcation, (ii) gradient-reversal suppression, (iii) anomaly-priority masking (enabling step). See §B for the contribution-set rewrite. |
| H6 | `sec1_intro.tex:21-22` ("key observation" sentence) | "...three distinct learning signals: (a) which temporal positions serve as informative hard reconstruction targets [=masking], (b) which patches the Student should avoid mimicking, (c) what representational content should be adversarially suppressed." | HEADLINE (soft) | **Keep but de-front (a).** This sentence is actually the *seed of the interaction story* (it is followed by lines 23–25 explaining why (b) alone is insufficient and (c) closes the pathway). Keep the three-signal framing but ensure (a)=masking reads as the enabling signal, with (b)/(c) carrying the weight. This is the spot to **promote** the synergy (see §C). |
| H7 | `sec5_conclusion.tex:13` (conclusion contribution recap) | "...through three orthogonal paths --- **anomaly-priority masking**, loss bifurcation that restricts Student mimicry to normal patches, and gradient-reversal suppression..." | HEADLINE | **Demote (reorder).** Conclusion recap fronts masking. Reorder to match the revised abstract/intro ordering; masking third. |

### LEGITIMATE-DETAIL occurrences — keep, ensure clearly secondary (7)

| # | file:line | Context | Class | Proposed action |
|---|-----------|---------|-------|-----------------|
| D1 | `sec3_method.tex:110-123` (`\paragraph{Anomaly-priority masking.}` + motivation para) | Method definition, masking-ratio formula, "addresses a structural imbalance of contaminated training" | LEGITIMATE-DETAIL | **Keep.** This is the correct, factual home. Verify it reads as a mechanism *enabling* the discrepancy, not as a standalone selling point. One light edit: append a half-clause linking it to the other two paths ("...so that the discrepancy and suppression mechanisms of §3.5 have anomalous targets to act on") — supports the synergy story (§C) at zero net length. |
| D2 | `sec3_method.tex:53,71` (Fig-2 placeholder + caption) | "anomaly-priority masking withholds \|M\|=8 patches (anomalous first)" | LEGITIMATE-DETAIL | **Keep.** Architecture-figure mechanics; factual. No change. |
| D3 | `sec3_method.tex:208` | "...which coincides with $y^w$ under anomaly-priority masking" | LEGITIMATE-DETAIL | **Keep.** Technical justification for the window-label coincidence; load-bearing. No change. |
| D4 | `sec4_experiments.tex:357` (ablation Table 3, Row 3) | "3. w/o anomaly-priority masking" | LEGITIMATE-DETAIL | **Keep.** Ablation evidence is legitimate and in fact *supports* the synergy claim (removing it degrades the signal). No change. |
| D5 | `sec4_experiments.tex:364-367` (`\paragraph{Anomaly-priority masking (Row~3).}`) | ablation discussion | LEGITIMATE-DETAIL | **Keep.** Secondary by location (ablation subsection). No change; do not expand. |
| D6 | `sec4_experiments.tex:404-405` (sparsity, "First, anomaly-priority masking applies only to labeled patches...") | one of three graceful-degradation arguments | LEGITIMATE-DETAIL | **Keep.** Structural-property argument; factual and secondary. No change. |
| D7 | `appendix_A.tex:38` (config table) + `appendix_C.tex:130` (pseudocode comment) + `appendix_C.tex:70-77` (Eq. masking_rule) | appendix mechanics | LEGITIMATE-DETAIL | **Keep all.** Appendix is the correct place for full detail. No change. |

### Keyword line — no masking term present (good)
`main.tex:133`: keywords are `Masked autoencoder \sep Asymmetric self-distillation \sep Gradient reversal`. Masking is **not** elevated to a keyword. No change needed; this already reflects the desired prominence ordering.

### Counts
- **HEADLINE (demote/remove): 7** occurrences (H1–H7) — abstract ×1, highlights ×1 (two files + two measure-twins), intro Fig-1 ×2, intro contribution bullet ×1, intro key-observation ×1, conclusion recap ×1.
- **LEGITIMATE-DETAIL (keep secondary): 7** occurrence-groups (D1–D7) — method para, method figure ×2 spots, ablation row + para, sparsity arg, appendix ×3 spots.
- Net change of length must be ~0 (reorders + recasts, not deletions of paragraphs).

---

## (B) CONTRIBUTION WEAKNESS ANALYSIS

### B.1 Current 4 contribution bullets (verbatim, `sec1_intro.tex:66-84`)

1. **Contaminated semi-supervised setting and benchmark protocol.** "We formalize the *contaminated semi-supervised* setting, in which labeled anomalies coexist with unlabeled training windows, and introduce a benchmark protocol that incorporates the chronological prefix of each dataset's test stream into training..."
2. **Three-path label integration into masked autoencoder representation learning.** "CSMAD integrates labeled anomalies through three orthogonal mechanisms: (i) *anomaly-priority masking*...; (ii) *loss bifurcation*...; and (iii) *gradient-reversal suppression*..."
3. **Asymmetric Teacher–Student decoder architecture.** "A deeper Teacher decoder (3 layers)... while a capacity-limited Student decoder (2 layers) fails more severely on anomalous correlation patterns than on normal ones — a design intended to make the Teacher–Student output discrepancy a reliable anomaly signal..."
4. **Extensive empirical evaluation.** "Experiments on [N] multivariate datasets covering industrial control, IT infrastructure, and spacecraft telemetry demonstrate competitive performance against [N] baselines under five evaluation metrics, with the label sparsity sweep..."

### B.2 Abstract framing (verbatim, `main.tex:90-98`)
"We propose CSMAD... integrates labeled anomaly information directly into masked autoencoder representation learning through three orthogonal mechanisms: anomaly-priority masking, a Student imitation loss restricted to normal patches, and gradient-reversal suppression... CSMAD employs an asymmetric Teacher–Student decoder architecture in which the capacity-limited Student mimics the Teacher less faithfully on anomalous correlation patterns than on normal ones, amplifying the Teacher–Student discrepancy signal under contaminated training."

### B.3 Diagnosis — why the contributions do not stand out

1. **Bullet #2 is a list, not a claim.** "Three-path label integration" enumerates three mechanisms as co-equal, fronting the weakest (masking). A reader cannot tell which is novel. The genuinely-new idea (adversarial gradient-reversal suppression *inside* masked self-distillation) is buried as item (iii) and reads as one of three engineering knobs. KBS explicitly flags "incremental loss-term tweaks presented as the headline" as a *non*-reward (`KBS_AIMS_SCOPE.md` §2 "What KBS does NOT reward").
2. **No bullet owns the novelty.** The strongest "first" claim lives in prose (`sec1_intro.tex:62`, `sec2_related.tex:33`) but never in a contribution bullet. The bullets describe *what the system contains*, not *what is new about it*.
3. **The synergy is invisible at bullet level.** Bullets #2 and #3 are split (mechanisms vs architecture) and presented additively. The fact that the components *interact* to produce the result — the actual scientific story (intro lines 21–25 already argue it) — is never claimed as a contribution. Directive (4) is currently unmet at the contribution-list altitude.
4. **Practical value under-stated in the bullets.** The abstract has a strong practical hook ("rarely met in practice", "recorded fault events"), but contribution #1 reduces it to "formalize a setting + protocol" — a methods-paper framing. KBS's single highest-weight lever is *practical significance / deployment* (`KBS_AIMS_SCOPE.md` §2A). The "turn recorded incidents into a learning signal" angle never reaches a bullet.
5. **Bullet #4 is generic.** "Extensive empirical evaluation" is table-stakes phrasing; it does not differentiate. Cross-domain generality (industrial + IT + telemetry) and graceful degradation are real strengths that read as filler here.

### B.4 The GENUINELY-STRONG / novel claims (truth-grounded — these should stand out)

These are the claims to foreground. Each is verified against `271_CONFIG_TRUTH.md` r4 and is **not** over-claimed.

1. **[NEW MECHANISM — strongest hook] Adversarial gradient-reversal suppression integrated into masked self-distillation, for contaminated semi-supervised MTSAD.** Labels shape the *representation itself* via reversed-gradient suppression of anomaly information in the Student (config-271: `use_grl=True`, `grl_mode='classifier'`, GRL backward `−λ_rev·grad`, `model.py:129–144`; suppression direction confirmed, §VIII GRL Details). Prior semi-supervised TS work attaches labels to a generative/predictive loss (`xue2022fewpositive`, `huang2022slavae`); the closest deep work NRdetector (`wang2025nrdetector`) delegates representation learning to a *label-agnostic* backbone. So "labels never shape the representation" is the real, citeable contrast. **Scoped first-claim is defensible:** "to our knowledge, the first ... in the contaminated semi-supervised MTSAD setting" (already in `sec1_intro.tex:62`, `sec2_related.tex:33`). Over-claim risk: LOW *if* the qualifier is kept. **Flag:** never drop "to our knowledge / in the contaminated semi-supervised MTSAD setting" — unqualified "first" is an over-claim (`KBS_AIMS_SCOPE.md` §4.4).
2. **[INTERACTION / SYNERGY — directive 4] A co-designed system whose components interact to preserve and amplify the discrepancy signal.** Capacity-gap asymmetric decoder (T3/S2, `num_teacher_decoder_layers=3`, `num_student_decoder_layers=2`) makes Teacher–Student discrepancy the anomaly signal *only if* the Student does not secretly learn to reconstruct anomalies; loss bifurcation (`grl_disable_anomaly_loss=True`, OD on normal patches only, `loss.py:259-261`) stops the Student being *trained* toward anomalies but leaves an indirect representational pathway; gradient-reversal suppression closes that pathway. Remove any one and the discrepancy degrades. This is **emergent from interaction, not concatenation** — and it is the scientifically honest framing of the asymmetric-design + label-paths story. Over-claim risk: LOW (mechanistic argument already in `sec1_intro.tex:23-25` and `sec3_method.tex:220-232`; ablation supplies the evidence). **This should become a contribution bullet** (currently it is not).
3. **[CAPACITY-GAP ASYMMETRIC DESIGN — practical/architectural] The asymmetric Teacher(3)/Student(2) decoder converts a capacity gap into a reliable, label-amplified discrepancy signal that degrades gracefully to a purely unsupervised detector.** Verified: T3/S2, stop-gradient Student input (`h^S_in = stopgrad(h^enc)`), inference score = recon + scaled-disc/4 (§VIII Anomaly Score). The graceful-degradation property (intro line 81; §4.6 sparsity) is a genuine *deployment* strength — works at the label-availability upper bound and reverts to the unsupervised floor as labels vanish. Over-claim risk: LOW.
4. **[SETTING + PROTOCOL + PRACTICAL MOTIVATION] The contaminated semi-supervised MTSAD setting, a chronology-respecting benchmark protocol, and the fault-log practical motivation.** Verified protocol constants are real (50% test-prefix-into-training split, leakage-free temporal suffix eval, region-22 83.75% dual SWaT eval, train AR 0.52–6.20%, per-entity normalization). Practical hook: operational logs *do* record a few fault/attack labels that unsupervised methods discard — CSMAD turns them into a learning signal. This maps onto KBS's named scope items "prediction systems and warning systems" + "decision-support" (`KBS_AIMS_SCOPE.md` §2B). Over-claim risk: LOW. **Flag:** keep the protocol-precedent sentence (`sec4_experiments.tex:79-80`, NRdetector also re-splits) so the protocol does not read as an unfair self-serving construction.

### B.5 Recommended contribution re-weighting (4 bullets, page-budget-neutral — re-write, do not add length)

Per `KBS_AIMS_SCOPE.md` §4.3, restructure the four bullets (NOT add a fifth) so each *owns a claim*:
1. **Problem + setting (practical):** contaminated semi-supervised MTSAD + chronology-respecting benchmark protocol that exposes labeled anomalies absent from standard splits. (Lead with the deployment reality.)
2. **New mechanism (novelty):** first integration of sparse anomaly labels into masked self-distillation via adversarial gradient-reversal suppression of anomaly information in the representation — labels shape the representation, unlike loss-attached or backbone-delegated prior work. (Absorbs old bullet #2; masking demoted to an enabling step named once.)
3. **Synergistic, time-series-aware design (interaction):** asymmetric Teacher–Student decoder whose discrepancy is *preserved and amplified by the interaction* of loss bifurcation and gradient-reversal suppression (with anomaly-priority masking as the enabling step); co-designed around MTS structure (cross-sensor correlation, windowed patches, chronological eval). Components are co-designed, not concatenated. (Merges old #2-mechanisms-as-system + #3-architecture into one *interaction* claim — this is the directive-4 fix.)
4. **Cross-domain evaluation (generality + rigor):** [N] datasets across industrial / IT / telemetry; [N] baselines; five complementary metrics; graceful degradation to the unsupervised floor. (Recast old #4 to name generality + the degradation strength, not "extensive.")

**Register guardrails (all bullets):** "competitive with / comparable to," never "we beat all"; keep the scoped "to our knowledge, the first." No AI-tells. No fabricated numbers — keep `[N]`/`[X.XX]`.

---

## (C) TIME-SERIES CHARACTERISTIC + COMPONENT-INTERACTION GAPS

### C.1 Where the paper treats components as a concatenated LIST (interaction gap)

| # | file:line | Symptom | Fix (truth-grounded, length-neutral) |
|---|-----------|---------|--------------------------------------|
| I1 | `sec3_method.tex:87-97` (`\subsection{Overall Architecture}`) | "CSMAD comprises five functional blocks: a linear patch embedding, a shared encoder, a Teacher decoder, a Student decoder, and a training-only adversarial branch." This is a **parts list** — the canonical concatenation symptom. The interaction is not stated at the overview altitude. | **Add a 1–2 sentence synthesis** at the end of the subsection stating the *interaction*: the three label paths and the capacity gap are co-designed so that loss bifurcation + gradient-reversal suppression jointly keep the capacity-limited Student bad at anomalies, which is what preserves and amplifies the Teacher–Student discrepancy that the score reads. Grounded: §VIII (T3/S2, GRL suppression, OD-normal-only). Offset length by tightening the existing "five functional blocks" sentence. **This is the single most important interaction edit.** |
| I2 | `sec3_method.tex:170-175` (`\subsection{Label-Guided Training}` opener) | "Three loss components couple labeled anomaly information to the model at different levels." Reads as three independent levels. | **Recast opener** to frame the three as a dependency chain (bifurcation removes the training pull toward anomalies; FM keeps the Student honest internally; GRL closes the residual representational pathway). The "Why gradient reversal is necessary beyond loss bifurcation" paragraph (lines 220–232) already argues this — **promote a one-clause forward-reference** so the interaction is visible before the equations, not only after. |
| I3 | `main.tex:90-94` (abstract) | Abstract lists "three orthogonal mechanisms" then *separately* describes the asymmetric decoder. The word "orthogonal" actively signals "independent / non-interacting" — the opposite of directive (4). | **Replace "orthogonal" framing in the abstract** with an interaction phrasing (e.g. "three interacting label-guided mechanisms" / "co-designed"). The asymmetric-decoder sentence should connect *causally* to the mechanisms (the mechanisms are *why* the capacity gap yields a reliable signal), not sit as a separate fact. Sync the two measure-twins. NOTE: "orthogonal" also appears at `sec1_intro.tex:72`, `sec2_related.tex:32`, `sec5_conclusion.tex:13` — audit each; keep where it genuinely means "distinct entry points" but do not let it imply non-interaction at banner altitude. |
| I4 | `sec5_conclusion.tex:12-17` | Conclusion recaps "three orthogonal paths ... built on an asymmetric ... architecture that converts the capacity gap into a reliable discrepancy signal." Closer to interaction than the abstract, but still lists-then-architecture. | **Light edit:** make the recap state that the paths and the capacity gap *together* (not separately) produce the discrepancy signal. One-clause change. |

### C.2 Where the MULTIVARIATE / TEMPORAL rationale is thin (time-series-specific gap)

| # | file:line | Symptom | Fix (truth-grounded) |
|---|-----------|---------|----------------------|
| T1 | `sec3_method.tex:102-108` (`\paragraph{Linear patch embedding.}`) | Does state "Projecting an entire patch — $s$ timesteps across all $F$ channels — into a single token encodes cross-channel correlations directly in the token." This is GOOD and is the strongest existing TS-aware sentence. | **Keep and lean on it.** This is the anchor for "designed for MTS characteristics." Reference it from the new synthesis (I1) so the cross-sensor-correlation rationale is connected to the discrepancy signal. |
| T2 | `sec3_method.tex:143-152` (`\paragraph{Why the capacity gap matters.}`) | Says the Student "fails more severely on the correlation patterns characteristic of anomalies." Good, but the *why-MTS* link (anomalies = correlated multi-channel deviations, not single-channel spikes) is implicit. | **Add a half-clause** tying the capacity gap to the multivariate nature: the discrepancy is diagnostic precisely because MTS anomalies are *cross-sensor correlation breaks* that a shallow decoder cannot reproduce. Grounded in intro line 9 ("correlated deviations across multiple sensor dimensions") — pull that rationale into the method. |
| T3 | `sec3_method.tex:8-39` (`\subsection{Problem Formulation and Setting}`) | Lines 31–39 *do* state the multivariate motivation well ("fault and attack records that document anomalies as correlated deviations across multiple sensor channels"; "Recovering the normal multi-channel correlation structure ... is the central learning challenge"). | **Already strong — preserve.** This is the cleanest TS-characteristic statement in the paper. The fix is to make the method-overview (I1) and the contribution bullets (§B.5 #3) *echo* it, so the TS-aware rationale is not stranded in the problem formulation. |
| T4 | `sec4_experiments.tex:60-84` (protocol) + scoring `sec3_method.tex:247-285` | Chronological-order respect (test-prefix into training, evaluate temporal suffix, no lookahead) is the *temporal* TS-aware design choice, but it is framed purely as a data-partition mechanic, not as "designed for time-series order." Leave-one-out aggregation across overlapping windows (line 284, "ensemble effect") is a temporal-context design but not labeled as such. | **Light framing edit** (one clause): label the chronological split as a *leakage-free, time-order-respecting* protocol (the term already supported by "no temporal lookahead", line 72) and note the windowed/overlap aggregation exploits temporal context. No new claims; re-labeling existing true mechanics as TS-aware. |

### C.3 Missing SYNTHESIS paragraph (the core directive-4 deliverable)
There is **no single place** in the method where the reader is told: *these components are co-designed and interact; the result is emergent.* The intro has the seed (lines 21–25) and §3.5 has the late argument (lines 220–232), but the **Overall Architecture** subsection (I1) — the natural synthesis location — is a parts list. **Primary recommendation:** insert the synthesis at I1 (`sec3_method.tex:97`, end of Overall Architecture), 1–2 sentences, length offset by tightening the parts-list sentence. This is the highest-leverage single edit for directive (4).

### C.4 Truth cross-check (so no new interaction claim is fabricated)
Every interaction claim above is supported by `271_CONFIG_TRUTH.md` r4 and `RESEARCH_SYNTHESIS.md`:
- T3/S2 asymmetric decoder: §VIII Architecture (`num_teacher_decoder_layers=3`, `num_student_decoder_layers=2`). ✔
- Loss bifurcation = OD on normal patches only, anomaly side zeroed: §VI + `loss.py:259-261`, `grl_disable_anomaly_loss=True`. ✔
- GRL suppresses anomaly-identity feature (NOT generates discriminative features): §VIII GRL Details + 부록1 r1 correction (`model.py:129–144`). ✔ — **Flag:** any new prose must use "suppress anomaly information," never "learn discriminative features" (the corrected direction).
- Discrepancy is the inference signal (recon + scaled-disc/4): §VIII Anomaly Score. ✔
- Cross-sensor correlation captured in the patch token: consistent with linear patchify over $s×F$ (§VIII Architecture, patchify_mode='linear'). ✔
- Chronological/leakage-free protocol: §III split ratios + `RESEARCH_SYNTHESIS.md` (test stride / leave-one-out ensemble line). ✔
- **Do NOT claim:** real-time/latency, interpretability, on-device, security guarantees, or any numeric margin (`KBS_AIMS_SCOPE.md` §5 "Do NOT claim"). The leave-one-out ~50× inference cost is a real *limitation* and must stay disclosed (`sec3_method.tex:251`, `sec5_conclusion.tex:25-28`).

---

## (D) FLOW-IMPACT NOTE — edits that risk breaking narrative and must be smoothed

1. **Abstract reorder (H1, I3) is the riskiest sentence-level change.** The abstract's two sentences (mechanisms → asymmetric decoder) currently flow as "what it integrates" → "how the architecture exploits it." Reordering the mechanism list AND swapping "orthogonal"→"interacting" must keep that causal flow, or the abstract reads as a disjointed feature dump. **Smooth by:** making the asymmetric-decoder sentence the *consequence* of the mechanisms ("...so that the capacity-limited Student...amplifies the discrepancy"). Re-measure body after edit — the two measure-twins (`main_3p/5p_measure.tex`) must be edited identically or R6=8.997p drifts.
2. **Contribution-bullet restructure (§B.5) touches the intro spine.** Merging old #2+#3 into a single interaction bullet changes the count from "list of 4 features" to "claim-owning 4." Risk: the method section (§3) is organized by the old decomposition (masking / decoders / label-training), so the intro and §3 headings could de-sync. **Smooth by:** keeping §3 subsection structure as-is (it is the correct *mechanical* decomposition) but adding the synthesis paragraph (C.3/I1) so the *narrative* matches the new contribution framing. Intro promises synergy; §3 overview delivers it; §3.3–3.5 give the parts.
3. **Highlights recast (H2) is constrained.** ≤85 chars × 5, must stay 5 bullets (KBS GFA), and the interaction/novelty bullet must still be self-contained. Risk: an interaction-led bullet can become vague. **Smooth by:** keep the gradient-reversal term concrete in the bullet (it is the indexable novelty) and verify char count in BOTH `main.tex:118-125` and `highlights.txt`.
4. **"Why gradient reversal is necessary beyond loss bifurcation" forward-reference (I2)** must not duplicate the late paragraph (lines 220–232). Risk: saying it twice inflates length and reads repetitive. **Smooth by:** one *clause* forward-reference at the §3.5 opener, the full argument staying at lines 220–232.
5. **Masking demotion across 7 headline spots must stay globally consistent.** If the abstract leads with "self-distillation + gradient reversal" but the conclusion still fronts masking, the paper reads inconsistently. **Smooth by:** applying the same mechanism ordering (loss bifurcation / gradient-reversal suppression / anomaly-priority masking, or representation-pair-first) everywhere: abstract, Fig-1 ×2, intro bullet, conclusion, highlights. Single ordering, applied uniformly.
6. **Do not touch the ablation row label or the masking method paragraph beyond the linking clause.** These are the factual backbone (D1, D4); over-editing them to "demote" would risk the method-truth constraint. Demotion is a *prominence* operation on banners only.

---

## SUMMARY FOR ORCHESTRATOR

**Masking occurrence counts:** HEADLINE (demote/remove) = **7** (abstract, highlights bullet [+highlights.txt +2 measure-twins], Fig-1 placeholder, Fig-1 caption, intro contribution #2, intro key-observation, conclusion recap). LEGITIMATE-DETAIL (keep secondary) = **7** groups (method para, Fig-2 ×2, ablation row+para, sparsity arg, appendix config/pseudocode/Eq). Keyword line already clean (no masking term).

**3–4 genuinely-strong contribution claims to foreground:**
1. NEW MECHANISM: first adversarial gradient-reversal suppression of anomaly info integrated into masked self-distillation, for contaminated semi-supervised MTSAD (scoped "to our knowledge, the first ..." — keep qualifier).
2. INTERACTION/SYNERGY: co-designed system — asymmetric capacity-gap decoder + loss bifurcation + gradient-reversal suppression interact to preserve/amplify the Teacher–Student discrepancy (remove one → signal degrades; emergent, not concatenated).
3. CAPACITY-GAP ASYMMETRIC DESIGN + graceful degradation to the unsupervised floor (deployment strength).
4. SETTING + chronology-respecting PROTOCOL + practical fault-log motivation (KBS practical-significance / warning-system scope hit).

**Top interaction / TS-emphasis gap locations (most leverage first):**
- `sec3_method.tex:87-97` (Overall Architecture = parts list, NO synthesis) — insert the 1–2 sentence interaction synthesis here [I1/C.3]. Highest leverage.
- `main.tex:90-94` + measure-twins — abstract "three orthogonal mechanisms" signals non-interaction; recast to "interacting/co-designed" and connect causally to the asymmetric decoder [I3].
- `sec3_method.tex:143-152` ("Why the capacity gap matters") + `:102-108` (linear patch embedding) — strengthen the multivariate/cross-sensor-correlation rationale and link it to the discrepancy [T1/T2].
- `sec3_method.tex:170-175` (Label-Guided Training opener) — recast three losses from "different levels" to a dependency chain; forward-reference the §3.5 necessity argument [I2].

**Constraints honored:** all results stay placeholders (A8); method truth preserved (masking demoted in prominence only, never removed — it is genuinely 1 of 3 label-entry points); no fabricated numbers; KBS format (5 highlights ≤85ch, 6 keywords, 5 declarations) untouched by these edits; body stays 8.5–9.0p (all edits length-neutral/compressing; measure-twins must be edited in sync). paper_legacy/ not opened.
