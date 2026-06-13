---
phase: 5 (revision)
agent: manuscript-reviser
source_blueprint: paper/08_final_audit/REVISION_BLUEPRINT.md
method_truth: paper/01_research_understanding/271_CONFIG_TRUTH.md (r4)
last_modified: 2026-06-13
---

# CSMAD Revision Log r2 — Directives (2) demote masking, (3) strengthen contributions, (4) interaction / time-series-aware

All edits applied in place to the canonical `.tex` files under `paper/07_latex/`.
Frontmatter edits (abstract, highlights) applied **identically** to the three measurement
twins (`main.tex`, `main_3p_measure.tex`, `main_5p_measure.tex`). No performance number written —
all `[X.XX]`/`[N]` + `PH:NUM-xxx` comments preserved. No `\cite` key changed (48 unique body
keys identical to baseline). All section files build error-free; 0 undefined refs; 0 `??`.

**Body length (R6):** `main_5p_measure.tex` body (title…Conclusion end) measured **8.997p**
(same as baseline; Conclusion ends on printed p9 right column at yMax 762.8pt = 99.4% fill,
identical to the Phase-7 baseline endpoint). Within the 8.5–9.0p window. The directive-driven
additions (net before offset ≈ +164 words, concentrated in the §3.2 synthesis) were offset by
meaning-preserving compression of verbose pre-existing prose in §4 (setup/baselines/metrics/
sparsity/qualitative), restoring the baseline float arrangement and page footprint.

---

## 1. ABSTRACT — `main.tex` + both twins (lines 90–98) — directives (2),(4)

**Edit A1 + A2 (combined).**
- **Before:** "...integrates labeled anomaly information directly into masked autoencoder
  representation learning through **three orthogonal mechanisms: anomaly-priority masking**, a
  Student imitation loss restricted to normal patches, and gradient-reversal suppression... CSMAD
  employs an asymmetric Teacher–Student decoder architecture in which the capacity-limited Student
  mimics... amplifying the Teacher–Student discrepancy signal..."
- **After:** "...integrates labeled anomaly information directly into the masked-reconstruction
  objective: a Student imitation loss restricted to normal patches and, **as the central novelty,
  gradient-reversal suppression** that removes anomaly-specific information from the Student's
  representation, **with anomaly-priority masking exposing the labeled positions on which the two
  act**. **These label-guided mechanisms are effective because of** an asymmetric Teacher–Student
  decoder: the capacity-limited Student mimics the Teacher less faithfully on anomalous correlation
  patterns... so suppressing anomaly information keeps that discrepancy sharp and amplifies it..."
- **Rationale (2):** "three orthogonal mechanisms" removed; masking named **last** as the enabling
  step, not a headline. **(4):** "orthogonal" → "on which the two act" / "because of" makes the
  decoder a *consequence* of the mechanisms (interaction, not concatenation). Three mechanisms
  remain factually present (271 r4 §VI: all three are real label-entry points). Length-neutral.

## 2. HIGHLIGHTS — `main.tex` + both twins (bullet 2) + `highlights.txt` — directives (2),(3),(4)

**Edit H-1.**
- **Before:** "Three label paths: anomaly-priority masking, loss bifurcation, gradient reversal." (82)
- **After:** "Gradient reversal makes labels shape the representation, not just the loss." (**75 chars**)
- **Rationale (2):** masking **removed entirely** from the highlights banner (it is an
  implementation detail). **(3a):** replaced with the indexable novelty hook (gradient reversal
  shaping the representation). Verification comment updated to new char counts (79/75/84/84/83).
  Exactly 5 bullets; all ≤85 chars; `highlights.txt` synced; `PH:NUM-003` preserved.

## 3. KEYWORDS — `main.tex` — NO CHANGE (already clean: 6 keywords, no masking term).

## 4. INTRODUCTION — `sec1_intro.tex`

**Edit N-1 (key-observation, line 22) — directives (2),(4).**
- Before: "The design exploits all three **simultaneously** to amplify both the reconstruction
  error and the Teacher–Student discrepancy..."
- After: "The design exploits all three **together**: (a) merely surfaces the labeled positions so
  that (b) and (c) **can act on them**, amplifying both..."
- Rationale: "simultaneously" (reads as independent) → "together … act on them" (interaction);
  signal (a)=masking demoted to *enabling*. No new sentence.

**Edit F1 (Fig-1 placeholder body line 40–42 AND caption line 55–57) — directives (2),(4).**
- Before: "...through three paths — **anomaly-priority masking, loss bifurcation, and
  gradient-reversal suppression** — turning contamination into a learning signal." (masking first)
- After: "...through **co-designed** label paths — loss bifurcation and gradient-reversal
  suppression, **enabled by anomaly-priority masking** — turning contamination into a learning
  signal." (masking last, "co-designed"). Applied identically in both spots.

**Edit (contribution bullets, lines 66–84) — full restructure to C1–C4 — directives (2),(3),(4).**
- **C1** (kept + light touch, directive 3a/practical): added "Motivated by operational logs that
  record a few fault and attack events" opener so the bullet owns the deployment value.
- **C2 (replaces old #2 "Three-path label integration", directive 2+3a):** retitled
  "**Label-driven representation suppression via gradient reversal.**" Recast from a 3-item list
  (which fronted masking) into a single novelty claim; masking demoted to a **trailing enabling
  clause** ("anomaly-priority masking surfaces the labeled positions this acts on"). Scoped "to our
  knowledge, the first" retained.
- **C3 (NEW interaction bullet, absorbs old #3, directive 4):** "**Synergistic, time-series-aware
  design.**" States components *interact* (co-designed, not concatenated) and that the design
  targets MTS structure (cross-sensor correlation per patch token, chronological evaluation).
  This is the directive-4 contribution at bullet altitude.
- **C4 (recast old #4 "Extensive empirical evaluation", directive 3b):** "**Cross-domain
  evaluation and label-sparsity robustness.**" Dropped "Extensive"; names the differentiators
  (industrial/IT/telemetry, graceful degradation, deployability). `PH:NUM-004/005` re-homed here.

## 5. RELATED WORK — `sec2_related.tex` (line 32) — directive (4)

- "through **three orthogonal mechanisms** that shape what the model learns to represent" →
  "through **co-designed mechanisms** that shape...". Stops signalling non-interaction at this
  positioning-critical sentence. The C2 anchor (NRdetector contrast, "first end-to-end" scoped
  claim line 33) kept verbatim.

## 6. METHOD — `sec3_method.tex`

**Edit §6.2 SYNTHESIS / INTERACTION paragraph (end of §3.2 Overall Architecture) — PRIMARY directive (4) edit.**
- Tightened the parts-list sentence (dropped the adversarial-branch appositive re-description; it
  is defined in §3.5).
- **Added** a co-design synthesis paragraph: "These blocks are co-designed rather than stacked: the
  asymmetric capacity gap makes the Teacher–Student discrepancy informative only if the Student
  stays poor at anomalies, so loss bifurcation removes the training pull toward anomalous patches,
  gradient-reversal suppression closes the residual representational pathway by which the Student
  could still learn them, and anomaly-priority masking surfaces the labeled positions both act on.
  The discrepancy the score reads is thus preserved and amplified by the interaction of these
  components, not by any one alone." Grounded in 271 r4 (§VIII T3/S2; §VI OD-normal-only; §VIII GRL
  **suppression** direction; §VIII masking). Wording guard honored: "suppress anomaly information,"
  never "learn discriminative features."

**Edit §6.5 "Why the capacity gap matters" (line ~145) — directive (4) time-series.**
- Added one clause: "The discrepancy is diagnostic precisely because anomalies here are
  cross-sensor correlation breaks rather than single-channel spikes that a shallow decoder cannot
  reproduce; consequently, the output discrepancy carries a stronger anomaly signal..." (T2,
  unique MTS insight). Self-distillation citations kept.

**Edit §6.6 Label-Guided Training opener (line 173) — directive (4) interaction.**
- "Three loss components couple labeled anomaly information to the model **at different levels**" →
  "...as a **dependency chain** rather than independent terms: bifurcation removes the training
  pull toward anomalies, and gradient reversal then closes the residual representational pathway it
  leaves open (argued below)." One-clause forward-reference to the existing necessity paragraph
  (kept verbatim at lines 234+). λ_GRL/λ_rev dual-λ paragraph untouched (method-truth §VIII r4).

**§6.4 masking paragraph / §6.7 scoring:** the optional linking half-clauses (D1/T4) were drafted
then **reverted to their original baseline wording** because the §3.2 synthesis already carries the
synergy link prominently and the body-length budget required the offset; masking and scoring remain
factually accurate and unchanged from baseline. (Masking stays a real, accurately described
component — directive 2 changes prominence only.)

## 7. EXPERIMENTS — `sec4_experiments.tex`

**Edit §4.3 ablation framing (subsection opener, line 332) — directive (4).**
- Added topic sentence: "The ablations test whether the components interact as designed: each
  removal should degrade the Teacher–Student discrepancy the score depends on." Reframes the
  ablation as **interaction evidence** (C3 support). All rows/numbers untouched
  (`PH:NUM-020/021/022/023` intact; the "w/o anomaly-priority masking" Row 3 + paragraph kept as a
  *minor component* study — directive 2: secondary, not promoted).
- Removed the redundant "Extended variants" body paragraph (its content is already in the Table 3
  caption note) — length offset, no information lost.

**Length-offset compressions (meaning-preserving, no numbers/constants touched):** §4.1 protocol
("no temporal lookahead … no model sees evaluation labels" merged), SWaT dual-eval prose, baseline
comparison-conditions prose, metrics threshold prose, epoch-asymmetry disclosure, protocol-effect
analysis, sparsity "Design"/"expected"/"Results" prose, qualitative §4.4 body (de-duplicated
against the Fig-4 caption). All protocol constants (50% split, region-22 83.75%, train AR
0.52–6.20%, 26 baselines, five metrics, layer counts) preserved verbatim.

## 8. CONCLUSION — `sec5_conclusion.tex` (recap, lines 12–17) — directives (2),(4)

- **Before:** "...through **three orthogonal paths** — **anomaly-priority masking**, loss
  bifurcation..., and gradient-reversal suppression... — built on an asymmetric Teacher–Student
  decoder... that converts the capacity gap into a reliable discrepancy signal."
- **After:** "...through **co-designed, interacting paths** — loss bifurcation that restricts
  Student mimicry to normal patches and gradient-reversal suppression of anomaly-specific
  information, **with anomaly-priority masking surfacing the labeled positions they act on**.
  Together with the asymmetric Teacher–Student decoder (3-layer Teacher, 2-layer Student), these
  paths turn the capacity gap into a reliable, amplified discrepancy signal under contaminated
  training."
- Rationale (2): masking no longer fronted (now "surfacing the labeled positions"). (4):
  "orthogonal" → "co-designed, interacting"; paths + capacity gap *together* produce the signal.
  Practical framing, benchmark sentence, ~50× limitation, future-work, code link all kept. No
  "In conclusion" introduced.

---

## Masking-demotion status (7 HEADLINE spots)

| # | Location | Action | After |
|---|----------|--------|-------|
| H1 | abstract (main + 2 twins) | reorder + "orthogonal"→interaction | masking named **last**, enabling step |
| H2 | highlights (main + 2 twins) + highlights.txt | replaced bullet | **masking removed from highlights** |
| H3 | Fig-1 body `sec1_intro.tex` | reorder | masking last ("enabled by"), "co-designed" |
| H4 | Fig-1 caption `sec1_intro.tex` | reorder (match H3) | masking last |
| H5 | intro contribution bullet | retitle→C2, recast | masking = **trailing enabling clause** |
| H6 | intro key-observation | de-front (a) | masking = enabling signal ("act on them") |
| H7 | conclusion recap | reorder + "orthogonal"→interaction | masking "surfacing positions", not headline |

**Masking KEPT (factual, accurate — never removed):** §3.2 synthesis (as enabling step),
§3.2 method paragraph "Anomaly-priority masking" (`sec3_method.tex:118`, baseline wording),
ablation Row 3 + paragraph (`sec4_experiments.tex`), sparsity argument, appendix C config/eq.
All "orthogonal" component-uses removed (abstract, intro bullet, related:32, conclusion); the
legitimate metric-perspective "three orthogonal perspectives" at `sec4_experiments.tex:157` kept.

## Hard-constraint checklist (verified)

- [x] No performance number written; all `[X.XX]`/`[N]` + `PH:NUM-xxx` preserved (counts match baseline).
- [x] Masking factually present in method + ablation + appendix; demoted in prominence only.
- [x] GRL described as **suppressing anomaly information**, never "learning discriminative features."
- [x] λ_GRL / λ_rev dual-λ paragraph kept (not collapsed).
- [x] Score formula recon + scaled-disc/4 (c=4) untouched.
- [x] Scoped "to our knowledge, the first … contaminated semi-supervised MTSAD" kept on novelty mentions.
- [x] 5 highlights ≤85 chars (79/75/84/84/83) in both files; 6 keywords; 5 declarations; `journal{KBS}`; flat structure — untouched.
- [x] Citations only from refs.bib; 48 unique body cite keys identical to baseline; no near-paraphrase.
- [x] No AI-tells; no em-dash overuse (counts stable/reduced); paper_legacy/ never opened.
- [x] All three measure-twins edited identically for frontmatter; body re-measured **8.997p** (8.5–9.0p).
