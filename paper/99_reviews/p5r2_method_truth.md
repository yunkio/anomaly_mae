---
phase: 5 (round 2)
agent: independent method-truth & fabrication auditor
scope: current .tex manuscript under paper/07_latex (canonical) vs 271_CONFIG_TRUTH.md (r4) + EXPERIMENT_PROTOCOL_TRUTH.md (r4)
independence: judged AS-IS as a fresh reviewer; did NOT open REVISION_BLUEPRINT / REVISION_AUDIT / prior 99_reviews / paper_legacy
date: 2026-06-13
---

# P5R2 — Independent Method-Truth & Fabrication Audit

## Verdict

**CONDITIONAL PASS — 1 MAJOR must be fixed before submission.**

No fabricated performance numbers found. Every quantitative *result* in the
manuscript is a `[X.XX]`/`[N]` placeholder with its `PH:NUM-xxx` comment; all
real digits in prose are protocol constants or architecture/config constants,
and they verify against the truth docs — **with one exception** (SMD `4.16`,
MAJOR-1, a mislabeled dataset statistic, not a performance number).

The revised contribution/method claims (GRL as central novelty, demoted
anomaly-priority masking, the interaction/synergy framing) are factually true to
config 271. Masking is still stated to exist accurately in all required places.
Equations and notation that were touched are correct.

## Counts

- BLOCKER: 0
- MAJOR: 1
- MINOR: 3
- Fabricated performance numbers: **0**
- Truth violations (factual): **1** (MAJOR-1)

---

## MAJOR

### MAJOR-1 — SMD `4.16` is the original-benchmark full-test ratio, mislabeled as the contaminated re-split held-out Test AR
**Files/lines:**
- `sec4_experiments.tex:52` — Table 1 row `SMD ($\times$28) ... 4.16 (avg)` under the column header (caption, `:35-36`) "Test AR = anomaly ratio (\%) in the **held-out evaluation portion**."
- `appendix_A.tex:239` — Table A.5 (per-entity) `SMD ($\times$28) ... 4.16 (avg)`, same "held-out evaluation portion" semantics.

**Problem (truth violation):** `EXPERIMENT_PROTOCOL_TRUTH.md:49` shows `4.16%`
is the **official OmniAnomaly full-dataset** anomaly ratio
("28 machines × 38 features / train 708,405 / test 708,420 / anomaly 4.16% —
byte-level 일치"), i.e. the *original benchmark's whole test file*. The
contaminated protocol evaluates only the **back 50% of the original test file**,
whose per-machine Test AR varies and is explicitly "machine별 상이" in the truth
doc (example given: `SMD/machine-1-4 anomaly_ratio=0.0363` = 3.63%). There is
**no truth-confirmed re-split SMD-average of 4.16%**; the back-half ratio is a
different quantity from the full-test ratio. Presenting 4.16% as the
held-out-suffix Test AR is a factually incorrect protocol statistic (it is also
the one cell in the dataset tables that does not carry a `pending`/`[X.XX]`
marker although the protocol value is genuinely not yet computed).

**Fix (pick one):**
1. Replace `4.16 (avg)` in both tables with a placeholder consistent with the
   "SMD per-machine values pending" wording already used elsewhere
   (`sec4_experiments.tex:69`, `appendix_A.tex:228`), e.g. `[X.XX] (avg, pending)`
   with a `PH:NUM-xxx` comment — this is the cleanest A8-consistent option since
   the true re-split SMD test-AR average is not yet in the truth doc; **or**
2. If a full-benchmark reference figure is genuinely wanted, move it to a
   clearly-labeled column/footnote ("original full-test AR, Su et al. 2019") so
   it is not read as the re-split Test AR — but the held-out-suffix column must
   not display it.

Note this is a *dataset statistic*, not a model metric, so it is not an A8
"illegal result number"; it is a method/protocol-truth misstatement of MAJOR
severity because reviewers will compute it and catch the mismatch.

---

## MINOR

### MINOR-1 — Pseudocode reversal-schedule drops the `+1` epoch offset present in the equation and the truth
**File/line:** `appendix_C.tex:143` (Algorithm 1):
`τ ← clip((e − 250)/(500 − 250), 0, 1)`
vs Eq. (eq:rev_schedule) `appendix_C.tex:15-16`:
`τ = clip((e − e_0 + 1)/(e_1 − e_0), 0, 1)` and truth
`271_CONFIG_TRUTH.md:448`: `p = clip((epoch − 250 + 1)/250, 0, 1)`.
The pseudocode omits `+1`, so it is internally inconsistent with its own
equation and with config 271. **Fix:** `(e − 250 + 1)/(500 − 250)` in the
pseudocode (or drop the offset everywhere — but the truth has `+1`, so add it to
the pseudocode).

### MINOR-2 — `L_main = L_recon + L_OD` is exact for FM but approximate for the GRL adaptive weight
**File/line:** `appendix_C.tex:61-63` defines a single
`L_main = L_recon + L_OD` for the gradient-norm ratio of *both* FM and GRL.
Code (`trainer.py:639-652` FM block, then `:746-763` GRL block) adds the FM term
into the running `loss` **before** the GRL adaptive-lambda gradient is taken
(`_main_g = grad(loss.float())`, `:754`). So for GRL the effective "main"
gradient includes the already-added `λ_FM·L_FM` contribution, not just
`L_recon + L_OD`. For FM itself the statement is exact (FM not yet added at that
point). The truth doc only ever calls it `grad_main` without decomposition, so
the manuscript's explicit decomposition slightly overstates precision for GRL.
**Fix:** soften to "the gradient of the main reconstruction objective
(reconstruction plus output discrepancy)" without the literal `=` for the GRL
case, or add a one-clause note that the GRL ratio is taken against the running
loss including the FM term. Low severity (conceptual abstraction, not a
fabrication).

### MINOR-3 — NRdetector "7:3 ratio" re-split claim is unsupported by either truth doc
**File/line:** `sec4_experiments.tex:78`:
"NRdetector likewise re-splits standard benchmarks (at a 7:3 ratio)".
Neither `271_CONFIG_TRUTH.md` nor `EXPERIMENT_PROTOCOL_TRUTH.md` contains a
"7:3" figure for NRdetector; this is a claim about prior work
(`\cite{wang2025nrdetector}`) that lies outside the config-271 method-truth
scope and was not verifiable from the provided method-truth sources.
**Fix:** confirm the "7:3" against the NRdetector paper (a Phase-4 reference task)
or soften to "likewise re-splits standard benchmarks so that anomalous events
fall within the training stream" (the unquantified claim is safe; the specific
ratio is the only unverified token). Flagging for completeness; not a
config-271 truth violation.

---

## Items explicitly verified TRUE-to-truth (no action)

- **Masking factual existence preserved (directive a).** Anomaly-priority
  masking is stated and accurate in the abstract (`main.tex:92-94`), intro
  contribution 2 (`sec1_intro.tex:72`), method §3.3 (`sec3_method.tex:118-131`),
  Eq. (eq:masking_rule) (`appendix_C.tex:69-77`), and config Table
  (`appendix_A.tex:38`). It is demoted in *prominence* (described as the enabler
  that "surfaces/exposes the labeled positions") but never removed or misstated.
  This matches truth: it is genuinely one of the three label-entry points
  (`EXPERIMENT_PROTOCOL_TRUTH.md:224`).
- **Three label-guided pathways** = anomaly-priority masking + loss bifurcation
  (OD-exclusion) + GRL suppression — matches truth `§⑦:224`
  (force_mask_anomaly / GRL target / dynamic-margin-separation-via-point_labels),
  and the manuscript correctly does **not** claim the dynamic-margin/anomaly-loss
  path (DEAD in 271 — truth `§VII #1`): `L_total` (eq:ltotal) has no anomaly-loss
  term, and `sec3_method.tex:252` states the GRL term "contributes only when the
  batch contains a positive window."
- **Interaction/synergy claims** (`sec1_intro.tex:22,75`, `sec3_method.tex:98-105`,
  abstract `main.tex:95-98`, conclusion `sec5_conclusion.tex:12-17`) are framed
  as *mechanistic design rationale*; all empirical magnitudes are deferred to the
  ablation table (`[X.XX]`, PH:NUM-021/022/023) and "quantified in
  \ref{...}". No fabricated effect size. The GRL-keeps-Student-poor framing
  matches truth `§VIII` ("suppression … discrepancy 증폭").
- **Asymmetric decoder 3L/2L, encoder 4L, d_model 512, 8 heads, ff 2048,
  dropout 0.15** — `sec4_experiments.tex:97-98`, `appendix_A.tex:32-33`,
  `appendix_C` notation — all match truth `§VIII Architecture`.
- **GRL head "two-layer MLP"** (`sec3_method.tex:215`, `appendix_A.tex:34`:
  LN→Linear(512→256)→GELU→Dropout(0.1)→Linear(256→1)) — matches truth `§VIII
  GRL Details:441`; the forbidden "1-layer MLP" label is correctly avoided.
- **Dual-λ GRL** (`sec3_method.tex:169-177`, eqs rev_schedule/grl/adaptive_weight):
  λ_rev = 2/(1+exp(−10τ))−1, τ=clip((e−250+1)/250,0,1), ≈0.02→≈1; λ_GRL = adaptive
  grad-ratio clamp[0,10] × β_GRL=0.2; reaching gradient ∝ −λ_rev·λ_GRL — all match
  truth `§VIII:445-449` (including the `+1` offset and the multiplicative coupling).
- **Score formula** (eqs dscale/sigma): `d̃ = d·(r̄+ε)/(d̄+ε)`, ε=1e-4; `σ = r + d̃/c`,
  c=4; FM excluded; GRL not used at inference — matches truth `§VIII:480-494`.
- **Focal-BCE variant** (eq:lcls_app): `(1−e^{−ℓ})^γ ℓ`, p_t:=e^{−ℓ}, γ=2,
  pos-weight floored at 1e-3 — matches truth `§VIII` + `loss.py:337-340` and
  `§III-3b` (999.0 = (1−.001)/.001 floor).
- **Mean point aggregation** (eq:agg) and **anomaly-ratio (1−α)-quantile
  threshold** — match truth `EXPERIMENT_PROTOCOL_TRUTH.md §④-2 / §⑤`.
- **PA%K-AUC integral over K=0..100 step 1 (101-point grid)**
  (`sec4_experiments.tex:142`, `appendix_A.tex:173-175`) — matches truth `§VIII`
  "in steps of 1" and the r4 errata E-2 (101-point integration grid; the
  {0,5,…,100} grid is the per-K diagnostic grid, correctly NOT used here).
- **All dataset/protocol constants** (113 entities / 114 conditions; per-family
  train/test sizes, dims, train/test AR; region-22 [2869,38769) / 35,900 /
  83.75% / 15.96%; safe-cut 4/81, +166/−39/−39/+8, 252 aggregate; 26 baselines
  = 9+6+7+4; SWaT 45-dim derivation incl. the 6 constant cols; WaDi A2 123 =
  127−4 NaN; train/test stride 21/49; |M|=8, |V|=42; 500/250 epochs, bs 1024,
  seed 42, lr 1e-3) — verified against truth `§①/②/⑥/③` and `271_CONFIG_TRUTH §VIII`.
- **Test-set model selection disclosure** (`sec4_experiments.tex:119-123`,
  `appendix_A.tex:58-59`) and **epoch asymmetry 500/10/50, eval 5/1**
  (`sec4_experiments.tex:107-117`, `appendix_A.tex:71-73`) — both disclosed per
  truth `§④ M-3 / RB-1`; the prior stale "test stride=1" is correctly absent
  (stride 49 used).
- **No performance numbers anywhere.** Grep of all decimals/integers in prose
  and tables: every metric cell is `[X.XX]`; the only standalone decimals are
  the anomaly-ratio statistics, safe-cut shares, and sparsity-sweep p-values
  (all protocol/config constants). The "≈50×" inference cost
  (`sec5_conclusion.tex:26`) is the forward-pass count derived from N=50, with
  the *measured* wall-clock factor kept as `[X.XX]` (PH:NUM-031,
  `appendix_B.tex:89`) — legitimate.

## Fabricated-number / truth-violation hits

- Fabricated performance numbers: **NONE.**
- Truth violations: **1** — MAJOR-1 (SMD `4.16` is the OmniAnomaly full-test
  benchmark ratio mislabeled as the contaminated re-split held-out Test AR at
  `sec4_experiments.tex:52` and `appendix_A.tex:239`).
