---
phase: 7
agent: latex-engineer
version: v2 (r2 — PDF QA r1 fixes: rotating dropped, adjustbox/placeins added, appendix counter resets)
last_modified: 2026-06-11
source: elsarticle-template-num.tex + elsdoc.pdf (bundle README + changelog inspection)
---

# TEMPLATE REQUIREMENTS CHECKLIST — Elsevier elsarticle (numbered citations)

## 1. Template selection rationale

**Template chosen**: `elsarticle-template-num.tex` (numbered style)
**Reason**: The manuscript uses `\cite{key}` numbered-style citations throughout (89 cite commands confirmed in v3 scan). The alternative `elsarticle-template-harv.tex` (author-year) would require converting every citation to `\citep`/`\citet` — unnecessary churn with no scientific benefit. The `elsarticle-num-names.bst` variant (surname+number) is not requested.

**documentclass option chosen**: `\documentclass[preprint,12pt]{elsarticle}`
**Reason and page-count relationship**:
- `preprint` is the standard single-column review/submission mode for Elsevier.
- `12pt` improves readability and is the template default.
- In preprint/1-column 12pt mode, a page holds approximately 550–620 words (body text only). With ~3,400 body words + equations + floats, the expected body page count is 11–14 pages in preprint mode.
- The D-010 budget gate requires 8.5–9p **body** count. This gate was defined for the **final journal layout** (1p/3p/5p options, typically ~700 words/page). In preprint mode the same content occupies ~1.5× more pages. The gate is measured in the compiled preprint output and reported in absolute terms; the submission-ready layout would use `\documentclass[final,3p,times]{elsarticle}` (Elsevier standard ~3-page journal format).
- For this LaTeX-engineer phase, `preprint,12pt` is used for readability/review; body page count is reported with the note that the final-layout equivalent would be approximately 0.6–0.65× the preprint page count.

## 2. Front matter requirements (elsarticle)

| Requirement | Command / environment | Status |
|---|---|---|
| Title | `\title{}` | Implemented |
| Author (placeholder) | `\author{[AUTHOR NAMES --- to be filled]}` | Implemented |
| Affiliation | `\affiliation{organization=...}` | Placeholder |
| Abstract | `\begin{abstract}...\end{abstract}` inside `\begin{frontmatter}` | Implemented |
| Graphical abstract | `\begin{graphicalabstract}` (optional; omitted) | Omitted (not required) |
| Highlights | `\begin{highlights}\item...\end{highlights}` | Implemented (5 items) |
| Keywords | `\begin{keyword}...\end{keyword}` with `\sep` separator | Implemented |
| Journal name | `\journal{...}` | Placeholder `[JOURNAL NAME]` |

## 3. Body structure requirements

| Requirement | Command | Notes |
|---|---|---|
| Sections | `\section`, `\subsection`, `\subsubsection` | No manual numbering |
| Equations | `\begin{equation}` with `\tag` override forbidden — use auto-numbering | Manual `\tag` from markdown converted to `\label`+`\eqref` where needed; appendix equations use `(C.x)` labels via custom counter |
| Footnotes | `\footnote{}` | Two footnotes in §3 (cs-fn, sd-fn) |
| Bold / emphasis | `\textbf{}`, `\emph{}` | Used for contribution terms |

## 4. Float requirements

| Float | Requirement | Implementation |
|---|---|---|
| Figures | `\begin{figure}[tbp]` + `\caption` + `\label` | TikZ framebox placeholders; no external files |
| Tables | `\begin{table}[t]`/`table*[tp]` + `\caption` + `\label` + booktabs | Real tabular skeletons with `[X.XX]` cells; wide tables are `table*` wrapped in `adjustbox{max width=\linewidth}` |
| Main results table (TAB-2) | Upright `table*` + `\scriptsize` + `\tabcolsep` 2pt (r2; sidewaystable abandoned — broke the twocolumn build, PDF QA r1 BLOCKER-2) | `rotating` package no longer used |
| Appendix figures/tables | Counter resets per appendix section: `\@addtoreset{table/figure/algocf}{section}` after `\appendix` + `\theH*` anchor disambiguation (elsarticle prefixes but does not reset) | A.1..., B.1..., C.1... numbering (PDF QA r1 MAJOR-5) |
| Algorithm | `algorithm2e` (`algorithm*[t]` two-column-spanning float) | ALG-C1 placeholder, numbered C.1 |
| Float drift control | `placeins` + `\BodyFloatBarrier` at the ends of §1/§3/§4 — active in one-column builds only (no-op under twocolumn to avoid column dead-space in the 5p gate build) | PDF QA r1 MAJOR-8 |

## 5. Bibliography requirements

| Requirement | Command | Notes |
|---|---|---|
| Style | `\bibliographystyle{elsarticle-num}` | Numbered [1], [2], ... |
| Database | `\bibliography{refs}` | `refs.bib` co-located with `main.tex` |
| `\citet` | **Not usable** with elsarticle-num (renders "(author?)"; PDF QA r1 MAJOR-3) | Replaced by literal "Ganin et al.~\cite{ganin2016dann}" (×2: §3.4, §C.1) |

## 6. Appendix requirements

| Requirement | Implementation |
|---|---|
| `\appendix` command before first appendix section | Done |
| Appendix sections as normal `\section` | Done (A, B, C groups via `\section`) |
| Appendix float numbering reset | `\setcounter` + `\renewcommand\the*` per group |
| Equation numbering in appendix | `\numberwithin{equation}{section}` + custom labels |

## 7. Packages used (beyond elsarticle defaults)

| Package | Purpose |
|---|---|
| `amsmath`, `amssymb` | Mathematics (template default) |
| `booktabs` | Professional tables (`\toprule`, `\midrule`, `\bottomrule`) |
| `tikz` | Figure placeholder boxes |
| `array` | Extended column specifications |
| `xcolor` | Gray fill for placeholder boxes |
| `adjustbox` | `max width=\linewidth` capping of wide tabulars (r2; loads `graphicx`) |
| `placeins` | `\FloatBarrier` for section-end float drift control (r2, one-column builds) |
| `algorithm2e` | ALG-C1 pseudocode placeholder (`algorithm*` two-column float) |
| `hyperref` | Cross-reference links (preprint mode) |
| `microtype` | Microtypography (optional, currently disabled) |

(r2 removal: `rotating` — sidewaystable abandoned per PDF QA r1 BLOCKER-2. Submission zip must bundle: adjustbox + collectbox (dependency), placeins, algorithm2e, plus the elsarticle bundle and `elsarticle-num.bst`.)

## 8. Known acceptable warnings

| Warning | Reason | Status |
|---|---|---|
| Overfull hbox, prose, <17pt (11× preprint / 4× 3p / 1× 5p) | Hyphenation limits in narrow measures | Acceptable; all table/algorithm overfulls >30pt eliminated in r2 |
| Underfull hbox | Float-heavy sections with placeholder text | Acceptable in draft |
| `natbib` redefinition | elsarticle loads natbib internally | Suppressed by not loading natbib explicitly |
| pdfTeX dest `page.1` duplicate (1× per build) | Unnumbered Highlights page + printed page 1 share the anchor | Benign elsarticle/hyperref artifact, pre-existing |
