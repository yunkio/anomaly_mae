---
review: p5r2 — independent plagiarism + academic-style review
reviewer: independent (fresh-eyes; blueprint/audit/prior-reviews NOT read)
scope: paper/07_latex canonical .tex (main_5p_measure + sec1-5 + appendix_A/B/C)
date: 2026-06-13
method_truth: 01_research_understanding/271_CONFIG_TRUTH.md (r4)
---

# VERDICT: PASS (no plagiarism hit; style clean with minor polish items)

**(a) Plagiarism: 0 hits.** No 6+ word distinctive verbatim overlap with any spot-checked
reference excerpt/abstract. Longest contiguous word-run overlap found anywhere = **3 words**,
all generic technical collocations. All paraphrases of close prior work (SDMAE, NRdetector, MAE,
DANN/Ganin, focal loss, Affiliation, liu2024) are reworded and correctly attributed via \cite{}.
All cite keys present in refs.bib.

**(b) Style: PASS.** Zero canonical AI-tells (delve/showcase/pivotal/seamless/leverage/"In
conclusion"/moreover/furthermore/etc. = 0 hits across all 8 files). Em-dash usage is normal
academic density (≈2-3 prose pairs per major section; the high raw `---` counts are LaTeX table
empty-cells and `% ---- ` comment rules, not prose). No formulaic "first/second/third" triads in
prose. Confident-but-not-overclaimed contribution framing. A few minor polish items below
(non-blocking).

Counts — Plagiarism hits: **0**. AI-tell hits: **0**. Style polish items: **5** (all minor).
Over-claim items: **1 minor** (repetition, not exaggeration). Compliance: highlights 5×≤85,
keywords 6, 5 declaration sections, journal{KBS}, flat layout — all OK. Body endpoint
(Conclusion → CRediT boundary) lands at the page-10/11 break in the 5p build, consistent with the
8.5-9.0p budget; nothing in this review enlarges the body.

---

## A. PLAGIARISM — detailed n-gram findings

Spot-checked against verbatim excerpts/abstracts in the library cards: wang2025nrdetector,
ristea2024sdmae, he2022mae, ganin2016dann, lin2017focal, liu2024elephant, huet2022affiliation,
zhang2022selfdistill. Longest-common-contiguous-word-run computed for each highest-risk
paraphrase. **None reaches the 6-word flag threshold.**

| Manuscript locus | Source excerpt | Longest exact run | Verdict |
|---|---|---|---|
| sec2_related.tex:38 "masking random patches and reconstructing the missing regions" | He MAE abstract "we mask random patches of the input image and reconstruct the missing pixels" | 2 words ("random patches") | OK — reworded ("pixels"→"regions", clause restructured), cited |
| sec2_related.tex:48 (footnote) "student decoder branches off from within the teacher decoder after its first transformer block" | SDMAE card "the student decoder branches out from the teacher after the first transformer block" | 3 words ("student decoder branches") | OK — closest case, but this is an unavoidable factual structural description of SDMAE, fully attributed, used to *contrast* our parallel-branch design. Not copied phrasing. |
| sec3_method.tex:212 "SDMAE suppresses anomaly reconstruction in the target/loss space (training the model to reconstruct anomaly-free targets)" | SDMAE card "we force our model to reconstruct the original training frames (without anomalies) to limit its ability to reconstruct anomalies" | 3 words ("model to reconstruct") | OK — paraphrased, cited; "anomaly-free targets" is the authors' own coinage, not copied |
| sec4_experiments.tex:148 "identified as the most reliable single measure for time-series anomaly detection by a large-scale study" | liu2024 abstract "we identify the most reliable and accurate measure, namely, VUS-PR for anomaly detection in time series" | 3 words ("the most reliable") | OK — "the most reliable [measure]" is the claim being attributed; reworded around it, cited |
| appendix_A.tex:191-193 Affiliation description | huet2022 abstract phrases | 2 words ("ground truth") | OK — independent phrasing ("per-event affinity scores within each event's affiliation zone"); none of the card's banned strings ("theoretically grounded, robust, parameter-free and interpretable") appear |
| sec3_method.tex:175 / appendix_C.tex:15 Ganin sigmoid schedule | Ganin §5.2 "λ_p = 2/(1+exp(−γ·p)) − 1" | 1 word | OK — the formula is a mathematical fact, reproduced as an equation with explicit \cite{ganin2016dann}; prose ("sigmoid schedule of Ganin et al.") is correct attribution |

Banned-string scan (card "복사 금지 표현" lists) returned **0 hits**: "flawed datasets / biased
evaluation / inconsistent benchmarking practices" (liu2024), "down-weights the loss assigned to
well-classified examples" / "easy negatives" (lin2017), "theoretically grounded, robust,
parameter-free and interpretable extension" (huet2022), "we are the first to introduce a variant
of self-distillation" (ristea2024) — none present.

**Factual-precedent claim check (not plagiarism, but verified for honesty):**
sec4:77 "NRdetector likewise re-splits standard benchmarks (at a 7:3 ratio)" — VERIFIED against
card excerpt "split the set of all segments by 7:3 ratio into training and test sets" (§5.1).
Claim is accurate.

---

## B. STYLE — AI-tells, em-dashes, prose quality

- **AI-tell vocabulary: 0 hits.** Grep over delve/delving, showcase, pivotal, seamless,
  crucial, paramount, realm, landscape, tapestry, underscore, leverage, harness, robustly,
  moreover, furthermore, "In conclusion/summary", "it is worth noting", "plays a key/vital role",
  cutting-edge, holistic, myriad, plethora, intricate, nuanced, meticulous, navigate, foster,
  unlock, unleash, elevate, profound. Clean across all section + appendix files.
- **Em-dash density: acceptable.** Real prose em-dash pairs: sec1 ≈3 (plus caption duplicates),
  sec2 0, sec3 3, sec4 ≈2, sec5 1. The raw counts (sec4:47, appx_A:31) are dominated by LaTeX
  empty table cells `& --- &` and `% ---- Table/Figure ----` comment rules, not prose. No
  overuse.
- **No formulaic triads in prose.** The one explicit enumeration (sec4:391 "Three structural
  properties bound this degradation. First… Second… Third…") is a legitimate technical list, not
  a rhetorical flourish.
- **Contribution framing is confident but not over-the-top.** The three "to our knowledge, the
  first…" claims are each tightly scoped to the *specific combination* (masked-reconstruction
  self-distillation + gradient reversal + contaminated semi-supervised MTSAD), never claiming GRL,
  MAE, or self-distillation as standalone inventions. Related Work (sec2) correctly credits Ganin,
  He, Zhang, Ristea, NRdetector as priors. This matches method-truth and avoids the GRL-card
  warning against "first to use GRL in anomaly detection." No false novelty.

### Minor polish items (non-blocking — author discretion)

1. **STYLE-1 (repetition, minor over-claim feel).** Three near-duplicate "first" sentences cluster
   at the paper front:
   - sec1_intro.tex:62 "CSMAD employs, to our knowledge, the first architecture combining
     masked-reconstruction self-distillation with gradient reversal…"
   - sec1_intro.tex:72 "CSMAD is, to our knowledge, the first framework to integrate sparse anomaly
     labels into masked self-distillation by adversarially suppressing…"
   - sec2_related.tex:33 "To our knowledge, CSMAD is the first end-to-end MTSAD model that
     integrates labeled anomalies into the gradient of a masked-reconstruction self-distillation
     objective through gradient reversal."
   These say the same thing three times within two pages; not exaggerated, but the repetition
   reads as over-insistence. *Suggested fix:* keep the contribution-list statement (sec1:72) and
   the Related-Work closing statement (sec2:33); soften or drop the abstract-paragraph one
   (sec1:62) to e.g. "CSMAD integrates masked-reconstruction self-distillation with
   gradient-reversal suppression of anomaly-specific information in a contaminated semi-supervised
   MTSAD setting." (removes the third redundant "first").

2. **STYLE-2 (mild awkwardness).** sec1_intro.tex:79 ends a contribution bullet with "— evidence
   of robustness and deployability." The dash-appended noun phrase reads as a tacked-on claim.
   *Suggested fix:* "…as labels become sparse, evidence of robust and deployable behavior." or
   simply drop "and deployability" (deployability is asserted, not shown).

3. **STYLE-3 (long sentence).** sec1_intro.tex:75 (contribution 3) is a 60+ word single sentence
   chaining "makes…; loss bifurcation and gradient-reversal suppression interact to keep…,
   preserving and amplifying…, so the components are co-designed rather than concatenated." Grammatical
   but dense. *Optional:* split at the second semicolon for readability. Non-blocking.

4. **STYLE-4 (word echo).** "co-designed rather than {stacked|concatenated}" appears 3× (sec1:75,
   sec3:98, and the figure caption sec1:40/55). The contrast is effective once; by the third it is
   a verbal tic. *Optional:* vary one instance.

5. **STYLE-5 (abstract phrasing, very minor).** main_5p_measure.tex:95-98 "These label-guided
   mechanisms are effective because of an asymmetric Teacher--Student decoder: the capacity-limited
   Student mimics the Teacher less faithfully on anomalous correlation patterns than on normal
   ones…" — "effective because of" is slightly loose causal phrasing for an abstract. *Optional:*
   "These label-guided mechanisms rely on an asymmetric Teacher--Student decoder…". Non-blocking.

No non-native/ESL grammar errors detected. Subject-verb agreement, article usage, and tense are
consistent and native-level throughout. Hyphenation ("label-aware", "gradient-reversal",
"anomaly-priority", "leave-one-out") is consistent.

---

## C. CONSTRAINT COMPLIANCE (verified, not exhaustive)

- **A8 no-fabricated-numbers:** All performance cells are `{[X.XX]}`/`[N]` placeholders with
  PH:NUM-xxx comments. Protocol constants present and real per method-truth: 50% midpoint split,
  5 metrics, region-22 83.75% (sec4:86 / appx_A:299), train AR 0.52%–6.20% (sec4:69), 113
  entities / 114 conditions, 26 baselines (22 unsup + 4 weak-sup). No performance number written.
  ✓
- **Method-truth fidelity:** GRL dual-λ (loss-weight λ_GRL grad-ratio×0.2 + Ganin sigmoid reversal
  coeff λ_rev, eq:rev_schedule e₀=250,e₁=500, ≈0.02→≈1) matches 271_CONFIG_TRUTH §VIII r4. GRL =
  *suppression* of anomaly information (not generation) — correct (sec3:227 "anomaly-invariant").
  2-layer MLP head (not 1-layer) — correct (sec3:215, appx_A:34, appx_C). Anomaly-priority masking
  retained as one of three label-entry points but appropriately demoted in prominence
  ("surfaces the labeled positions" / "merely surfaces") — factually present, not exaggerated. ✓
  Teacher-only warmup = student forward skipped (appx_C alg `if e>250`), framed as
  stability device not a contribution (sec3:166-167). ✓ Score = recon + scaled_disc/4, FM excluded
  at inference (sec3:266 "GRL classifier is not used at inference"; appx_A FM training-only). ✓
- **KBS format:** highlights 5 bullets measured 79/75/85→84(en-dash)/84/83 chars ≤85; keywords =
  6; declaration sections = 5 (CRediT, competing interest, generative AI, data availability,
  funding/ack); \journal{Knowledge-Based Systems}; flat \input layout (sec*/appendix* at top
  level). ✓
- **No paper_legacy / blueprint / audit / prior-99_reviews read.** ✓

---

## D. WHAT I COULD NOT FULLY VERIFY (honesty notes)

- Plagiarism spot-check covered the 8 closest/most-quoted references' cards, not all 49. Risk of
  an undetected overlap with a non-spot-checked source is low (the manuscript's distinctive
  technical vocabulary — CSMAD, anomaly-priority masking, loss bifurcation, dual-λ — is the
  authors' own), but not zero. The 8 checked are precisely the ones whose ideas the paper paraphrases
  most heavily.
- Body page measurement: I confirmed the body/declarations boundary lands at the page-10→11 break
  in main_5p_measure.pdf (Conclusion heading on p10, CRediT on p11), consistent with the stated
  ~9.0p body. I did not independently re-derive the 8.997p figure to two decimals; no change in
  this review affects body length.
