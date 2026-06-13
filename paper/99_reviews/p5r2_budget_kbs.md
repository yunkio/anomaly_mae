---
review: p5r2 — KBS-compliance & page-budget (independent)
reviewer: independent budget/KBS reviewer (fresh-eyes)
date: 2026-06-13
target: paper/07_latex (main_5p_measure.tex + shared section files)
verdict: PASS — no compile error, no KBS regression, body within R6 budget
---

# Independent KBS-Compliance & Page-Budget Review (p5r2)

Scope: judged the current `.tex` as it stands. Did NOT read REVISION_BLUEPRINT,
REVISION_AUDIT, prior 99_reviews, REVISION_LOG_r2, PROSE_DIFF_LOG, or pdf_qa/*
(prior-rationale + prior-review docs). Forbidden `paper_legacy/` not touched.
Method facts cross-checked against `271_CONFIG_TRUTH.md` (r4) only.

## 1. Compile status — PASS

```
latexmk -pdf -bibtex -interaction=nonstopmode -f main_5p_measure.tex
EXIT 0 — Output written on main_5p_measure.pdf (21 pages)
```

- Clean rebuild (after `latexmk -C`) succeeds end-to-end with bibtex.
- **No undefined references / citations** (`grep undefined main_5p_measure.log` → none).
- **No rerun-needed flag**; .aux/.out stable.
- **No `??` in compiled PDF** (`pdftotext` → 0 hits).
- bibtex .blg: only 9 benign `Warning--empty pages in <key>` (missing `pages=`
  field in some bib entries; not a compile error, no effect on numbered refs).
- All 48 `\cite` keys used in body+appendix resolve to refs.bib (49 keys defined).

## 2. Body page extent (R6: 8.5–9.0) — PASS, 8.99 pages

Layout note: in the 5p (final, twocolumn) build, **page 1 is the standalone KBS
Highlights page** (not body). The article body begins on **page 2** (title,
authors, abstract, keywords, then §1).

Measured by pdftotext -bbox coordinates (page height 841.89 pt; usable text
block ≈ 84.46→762.84 pt = 678.39 pt/page):

| Boundary | Location |
|----------|----------|
| Body start (title) | page 2, top (yMin ≈ 91.3) |
| §5 Conclusion (\label page) | 9 → renders across pp. 9–10 |
| §5 Conclusion text END | page 10, right column bottom (yMax ≈ 762.84 = full block bottom) |
| Declarations + References START | page 11 (excluded from body) |
| Appendix A START | page 12 (excluded) |

Computation: page-2 body fraction (title→bottom, capped) ≈ 0.990 + pages 3–9
full (7.000) + page-10 full to bottom (1.000) = **8.99 body pages**.

- Conclusion fills page 10 completely to the column bottom; no `\clearpage`/
  `\newpage` before §5, body flows continuously.
- Declarations and References correctly excluded (begin fresh on p. 11).
- 8.99 ∈ [8.5, 9.0]. Matches the stated 8.997p. **In budget.**

## 3. KBS format compliance — PASS (all preserved)

### Highlights — 5 bullets, all ≤85 chars
Counted precisely (rendered chars; LaTeX `--` renders as a single en-dash):

| # | rendered chars | bullet |
|---|------|--------|
| 1 | 79 | We formalize the contaminated semi-supervised MTSAD setting with sparse labels. |
| 2 | 75 | Gradient reversal makes labels shape the representation, not just the loss. |
| 3 | 84 | CSMAD's asymmetric Teacher–Student masked autoencoder amplifies anomaly discrepancy. |
| 4 | 84 | A contaminated benchmark adds test prefixes to training, exposing labeled anomalies. |
| 5 | 83 | Competitive on [N] datasets under five metrics; graceful decay with label sparsity. |

- `\begin{highlights}` env: exactly **5** `\item`s.
- Bullet 3 LaTeX source is 85 raw chars but contains `Teacher--Student`; the `--`
  renders as one en-dash → **84 rendered chars** (KBS counts rendered chars).
  Safe, but note it is the tightest bullet — any edit replacing `--` with two
  literal chars would push it to 85; keep the en-dash.
- **highlights.txt matches**: 5 bullets, single-dash form, counts 79/75/84/84/83
  (identical content to the LaTeX env). Consistent.

### Keywords — exactly 6
`Multivariate time series · Anomaly detection · Semi-supervised learning ·
Masked autoencoder · Asymmetric self-distillation · Gradient reversal`.
American spelling clean (no -ise/-isation; lone "optimistically" is a real word).

### Declaration sections — exactly 5 (correct order)
CRediT → Declaration of competing interest → Declaration of generative AI →
Data availability → Funding and acknowledgements. All present, between
Conclusion and References.

### Journal & structure
- `\journal{Knowledge-Based Systems}` set (main.tex:64, main_5p_measure.tex:64).
- Flat file layout (sec/appendix at single level).

## 4. Fact / fabrication / plagiarism checks — PASS

- **No fabricated performance numbers.** All results are placeholders: 424
  `[X.XX]` cells + 11 `[N]` counts across the body, each results table carrying
  PH:NUM-xxx comments. No concrete F1/AUC/precision value anywhere.
- Numeric values that DO appear are permitted protocol/dataset constants:
  dataset train/test lengths, dimensionality, Train/Test anomaly ratios
  (0.52%–6.20% train AR), label-fraction sweep {0.1,0.25,0.5,0.75,1.0}, the
  GRL λ_rev sigmoid ramp "≈0.02 → ≈1", and TikZ placeholder box heights
  (≈0.40p). The λ_rev ramp values match `271_CONFIG_TRUTH.md` §VIII exactly.
- **Anomaly-priority masking** is stated accurately and briefly (sec3 §3.3 +
  Label-Guided Training; intro): demoted in prominence (gradient reversal is the
  named central novelty) but factually preserved as one of the three label-entry
  points (`force_mask_anomaly=True`, anomalous patches masked first). Consistent
  with the truth doc; not invented, not removed.
- Method facts verified against truth doc: dual-λ GRL (loss-weight λ_GRL via
  grad-norm ratio + reversal coeff λ_rev sigmoid ramp), two-layer MLP head,
  window-level target, FM as training-only regularizer absent from inference
  score, L_OD on normal patches only, teacher-only warmup as stability device.
  All accurate.
- Citations are `\cite{key}` only; all keys in refs.bib. No copied/paraphrased
  reference text observed.

## 5. Academic style — PASS
- No AI-tells: zero hits for delve/showcase/pivotal/realm/seamless/
  "in conclusion"/"plays a crucial role"/cutting-edge/paradigm-shift.
- Em-dash usage normal: sec4's 47 `---` are mostly LaTeX comment rulers
  (`% ---- Table N ----`); genuine prose em-dashes ≈4–5, used for legitimate
  parenthetical enumeration. Not overuse.

## 6. Minor observations (non-blocking, no fix required)
- Highlights bullet 3 is at the 84-char ceiling (via en-dash). Fragile to edits;
  retain `Teacher--Student`.
- Dataset table entity counts (SMD ×28, SMAP ×54, MSL ×27) reflect the **full
  public benchmark protocol**, not config-271's run subset (SMD 22 / SMAP 5 /
  MSL 5 in the truth doc). These describe the benchmark, not the 271 run scope —
  no contradiction, but worth keeping straight when filling [N].

## Verdict
PASS. Compiles clean (exit 0, 21 pp), body 8.99 pp (within R6 8.5–9.0),
highlights 5×(79/75/84/84/83 ≤85) matching highlights.txt, keywords 6, five
declaration sections present, journal=KBS, no undefined refs / no `??`, no
fabricated numbers, no KBS regression.
