---
phase: 5
agent: budget-surgeon + appendix-drafter (v2-r2); comprehensive-fixer (v2-r3)
version: v2-r3
directives: [T5, D-009, D-010, R3, R7]
last_modified: 2026-06-11
v2_r3_changes: |
  P5 fix round (target file now MANUSCRIPT_v2.md; fixlog: paper/99_reviews/p5_fixlog_r2.md):
  ① §7.3 Table A.2 source corrected — the former "batch 1024/512" attribution to
    EXPERIMENT_PROTOCOL_TRUTH §④ was erroneous (no baseline batch 512 exists in that document;
    method-truth B-6): baseline batches are model-specific per original presets
    (comparison/baseline_common.py MODEL_CONFIGS, 32–512 across models; weak 4 = 32).
  ② FIG-2 content spec: GRL position label made explicit (Student decoder final-layer hidden,
    before output projection — adversarial MINOR-PH-03); caption point-score symbol s_t → a_t.
  ③ FIG-B1 caption: masking-ratio symbol r_m → ρ (manuscript notation overhaul, method-truth M-5).
  ④ Notation sync note: manuscript v2 renames — masking ratio ρ, schedule progress τ (Eq. C.1),
    anomaly fraction α, point score a_t, window index u, patch embedding E, d_model;
    upright T/S branch superscripts. Captions produced in Phase 6/7 must follow Table C.2.
scope: |
  Canonical, exhaustive inventory of every placeholder in MANUSCRIPT_v2_draft.md.
  IDs are globally unique (section-local PH:NUM-### collisions from the four drafts
  were renumbered in document order; the orig→global map is in INTEGRATION_REPORT_v1.md §3).
  v2 changes (D-009 surgery; full map in SURGERY_REPORT_v2.md): TAB-1 respecified as a
  6-family summary; TXT-001 relocated to Appendix §A.1; new appendix placeholders
  TAB-A3/A6/A7/A8, TAB-B1..B4, FIG-B1, ALG-C1, NUM-031 (§7 below); realized appendix
  tables (real protocol constants, no placeholders) audited in §7.3.
  v2-r2 changes (D-010 targeted reduction; SURGERY_REPORT_v2.md §8): TAB-4 ABSORBED into
  TAB-2 as a bottom protocol-effect row-group (no body [TAB-4] marker remains; merged
  caption/spec in the TAB-2 entry, absorption audit in the TAB-4 entry); TAB-3 respecified
  to 4 confirmed rows — former rows 5/6/7 demoted to TAB-B4 with their prose, relocating
  NUM-024/NUM-025 to Appendix §B.5.
  Completeness guarantee: full regex scan of MANUSCRIPT_v2_draft.md (PH:NUM 31/31,
  PH:TXT 2, FIG 4+1, body TAB 3 + appendix TAB 8, ALG 1 — zero unmatched markers).
  Phase 6 (results fill) and Phase 7 (LaTeX) must resolve or explicitly carry over every entry.
conventions: |
  - In-text form: visible token "[X.XX]" / "[N]" / "[GPU model]" / "[URL]" / "[FIG-k]" /
    "[TAB-k]" / "[ALG-k]" plus an adjacent <!-- PH:ID | description --> comment.
  - "Expected value" states the protocol-constant resolution candidate where one exists;
    A8 forbids writing it into prose before experimental confirmation.
---

# PLACEHOLDER REGISTRY — MANUSCRIPT_v2_draft

## 1. Figures (FIG-1 … FIG-4) — blueprint numbering preserved

### FIG-1 — Setting-comparison diagram
- **Type**: figure | **Location**: §1, after Para 3 (observation paragraph), before the contribution paragraph
- **Caption (complete)**: "Figure 1. Three training paradigms for multivariate time series anomaly detection under a contaminated training stream. *Left (unsupervised)*: labeled anomalies are invisible to the model and act purely as contamination of the all-normal assumption. *Middle (label-aware filtering)*: labeled anomaly windows are excised before unsupervised training (Q3 condition) — contamination is removed but the label information is discarded. *Right (CSMAD)*: labeled anomalies are integrated into training through three paths — anomaly-priority masking, loss bifurcation, and gradient-reversal suppression — turning contamination into a learning signal."
- **Content spec**: 3-panel horizontal diagram; identical input stream strip (normal segments + red labeled-anomaly segments) on top of each panel; per-panel flow glyphs (model box + how labels flow: ignored / excised / three arrows into masking-loss-gradient). Terminology must match §1 bullet 2 (anomaly-priority masking, loss bifurcation, gradient reversal suppression).
- **Size assumption**: full-width, ~5 cm height ≈ **0.40p** (PAGE_BUDGET §3)

### FIG-2 — CSMAD architecture overview
- **Type**: figure | **Location**: §3.2 opening (marker precedes the architecture prose)
- **Caption (complete, from method.md spec)**: "Figure 2. CSMAD architecture overview. *Left panel (training)*: the input window is split into $N$ patches; anomaly-priority masking withholds $|M|$ patches (anomalous patches masked first). Visible patches enter the shared Transformer encoder; mask tokens are inserted before each decoder. The Teacher decoder (darker, deeper) produces reconstructions $\{o^T_i\}$; the Student decoder (lighter, shallower) produces $\{o^S_i\}$. An AnomalyClassifierHead with gradient reversal (dashed box, labeled **training only**) is applied to the Student's final hidden states. Loss connections: $L_{\mathrm{recon}}$ from Teacher outputs; $L_{\mathrm{OD}}$ and $L_{\mathrm{FM}}$ between Teacher and Student on normal masked patches; $L_{\mathrm{cls}}$ from classifier head to window label. The encoder receives no gradient from Student or GRL (stop-gradient indicated by $\perp$). *Right panel (inference)*: GRL branch inactive; leave-one-out masking patterns batch-parallelized; per-patch scores $\sigma_i$ averaged to point-level scores $a_t$."
- **Content spec**: five color regions — (1) Patch Embedding, (2) Transformer Encoder (shared), (3) Teacher Decoder, (4) Student Decoder, (5) GRL + AnomalyClassifierHead. GRL box requires explicit dashed "training only" annotation; stop-gradient symbol ($\perp$) on Student latent input mandatory; **the GRL attachment point must be labeled explicitly: the Student decoder's final-layer hidden states, before the output projection** (blueprint ADV BLK-002; v2-r3).
- **Size assumption**: full-width; blueprint baseline 6 cm = 0.50p, **integrator assumption 5 cm = 0.40p** (PAGE_BUDGET §3 drafter note allows the −0.1p reduction; Phase 7 confirms readability)
- **Note**: method.md section draft originally numbered this [FIG-1]/"Figure 1" — corrected to FIG-2 (blueprint numbering; see INTEGRATION_REPORT §4-1).

### FIG-3 — Label sparsity sweep
- **Type**: figure | **Location**: §4.4, after the "Results" lead sentence
- **Caption (complete, from experiments.md spec)**: "Figure 3. Label sparsity sweep. PA%K-AUC F1 as a function of the labeled anomaly fraction $p \in \{0.1, 0.25, 0.5, 0.75, 1.0\}$ for [N] representative datasets (one line per dataset). Dashed horizontal lines indicate the performance of the best unsupervised baseline (Q3, main protocol) on the corresponding dataset, providing the unsupervised floor. $p = 1.0$ corresponds to the main experimental setting; $p \to 0$ approximates the fully unsupervised limit."
- **Content spec**: X = labeled fraction $p$; Y = PA%K-AUC F1; one solid line per dataset (2–3 recommended) + one dashed reference line per dataset; all data points pending the label-sparsity sweep runs (EXPERIMENT_EXECUTION_TODO). Dataset count = NUM-026.
- **Size assumption**: half-width / full-width ~4 cm ≈ **0.33p** (PAGE_BUDGET §3)

### FIG-4 — Qualitative score decomposition
- **Type**: figure | **Location**: §4.5, after the lead sentence
- **Caption (complete, from experiments.md spec)**: "Figure 4. Qualitative score decomposition on representative anomaly events. Each column corresponds to one dataset ([Dataset-A], [Dataset-B]). Row 1: multivariate input (first feature shown) with ground-truth anomaly regions shaded in red. Row 2: Teacher reconstruction error per timestep. Row 3: Teacher–Student discrepancy per timestep (adaptively scaled). Row 4: combined anomaly score with the anomaly-ratio threshold (dashed horizontal line). The decomposition illustrates how the two score components respond differently to anomaly characteristics: reconstruction error captures deviations from the learned normal pattern regardless of anomaly label, while discrepancy captures structural divergence amplified by the capacity gap and label-guided training."
- **Content spec**: 2 columns × 4 rows of aligned panels, shared X (timestep) per column, per-trace normalized Y. Dataset selection: SWaT excl22 + one of WaDi A1 / PSM (visual distinctiveness after results; count = NUM-028). Event selection must represent diverse anomaly types; interpretation text in §4.5 revised post-results (RT MINOR-02). Gaussian smoothing must NOT be mentioned (R34).
- **Size assumption**: full-width, ~3.5–4 cm (compression ladder) ≈ **0.30p** (blueprint baseline 0.35p)

## 2. Tables (TAB-1 … TAB-4) — blueprint numbering preserved (v2-r2: TAB-4 absorbed into TAB-2; 3 body markers remain)

### TAB-1 — Dataset statistics (v2: 6-family summary — D-009)
- **Type**: table | **Location**: §4.1.1, after the Datasets paragraph
- **Caption (complete, v2)**: "Table 1. Dataset statistics under the contaminated benchmark protocol, summarized per family. Train/test sizes reflect the re-split described in §4.1.1. Train AR = anomaly ratio in the training portion (originating from the incorporated test prefix); Test AR = anomaly ratio in the held-out evaluation portion. The WaDi row aggregates the two independent entities A1/A2 (values given as A1 / A2); SMD, SMAP, and MSL values are per-entity averages or concatenated totals as indicated. SWaT is evaluated under both full and excl22 conditions; Table 2 uses excl22 (§4.1.1). Per-entity statistics are in Appendix §A.3 (Table A.4)."
- **Columns**: Dataset family | #Train pts | #Test pts | #Dimensions | Train AR (%) | Test AR (%) | Source
- **Row values (real, EXPERIMENT_PROTOCOL_TRUTH §①) — 6 family rows**:
  - SWaT (A1+A2) | 719,959 | 224,960 | 45 | 1.63 | 19.05 (full) / 3.68 (excl22) | \cite{goh2016swat}
  - WaDi (A1 / A2) | 1,296,001 / 870,972 | 86,401 / 86,402 | 123 | 0.52 / 0.76 | 3.82 / 3.87 | \cite{ahmed2017wadi}
  - PSM | 176,401 | 43,921 | 25 | 6.20 | 30.63 | \cite{abdulaal2021psm}
  - SMD (×28) | per-machine (§A.3) | per-machine (§A.3) | 29–36 | per-machine (§A.3) | 4.16 (avg) | \cite{su2019omnianomaly}
  - SMAP (×54) | 355,905 | 217,925 | 25 | 0.70 | 24.54 | \cite{hundman2018telemanom}
  - MSL (×27) | 95,271 | 36,775 | 55 | 1.70 | 16.72 | \cite{hundman2018telemanom}
- **Size assumption**: ~**0.25p** (6 data rows × 7 cols, booktabs)
- **Note (v2)**: per-entity detail (separate WaDi A1/A2 rows, SMD per-machine placeholders) moved to Appendix §A.3 Table A.4 per D-009; #Dimensions column remains the single in-§4.1.1 source (INTEGRATION_REPORT §4-12).

### TAB-2 — Main comparison results + protocol-effect block (v2-r2: TAB-4 absorbed — D-010 ①)
- **Type**: table | **Location**: §4.2, after the lead paragraph
- **Caption (complete, v2-r2 — TAB-4 caption merged)**: "Table 2. Main comparison results under the contaminated benchmark protocol (Q3 condition for unsupervised baselines; Q1 for weakly supervised baselines). Reported metrics: PA%K-AUC F1 and VUS-PR; the remaining three metrics are in Appendix §A.5. SWaT column uses the excl22 evaluation condition; full-condition results appear in Appendix §A.4. SMD, SMAP, and MSL values are macro-averages over all entities (per-entity results in Appendix §A.6). Bold = highest; underline = second-highest. *Bottom block (protocol effect, Section 4.2)*: CSMAD and [N] representative unsupervised baselines under a standard clean-train split (original training file only, no labeled anomalies), evaluated on the identical held-out evaluation suffix; standard-split CSMAD uses the identical configuration with all label-dependent paths self-deactivating in the absence of positive training windows. Cells are populated only for the representative protocol-effect dataset columns; the contaminated-protocol counterparts are the corresponding main-block rows."
- **Row structure (27 main rows in 7 groups + bottom block)**: Simple (5: random, sensor-range, PCA, L2-norm, NN-distance) / Neural (3: MLP, MLPMixer, Transformer) / GCN-LSTM (1) / SOTA legacy (6: Anomaly Transformer, TranAD, USAD, DAGMM-simplified, GDN, OmniAnomaly) / SOTA recent (7: TFMAE, NPSR, TimesNet, DCdetector, MEMTO, ModernTCN, CATCH) / Weakly supervised — Q1 only (4: DeepMIL, WETAS, TreeMIL, NRdetector) / **CSMAD (ours)** / **Protocol-effect block (standard clean-train split)**: {CSMAD, Baseline A, Baseline B, [Baseline C]} — baseline count = NUM-014; rows filled only for the protocol-effect datasets (other family columns "—")
- **Columns**: Method | {SWaT excl22, WaDi A1, WaDi A2, PSM, SMD avg, SMAP avg, MSL avg} × {PA%K-AUC F1, VUS-PR} — column set fixed per RT V3 (2 metrics; the other 3 in Appendix §A.5). All metric cells [X.XX] pending the experimental queue.
- **Size assumption**: landscape (sideways) full-width ≈ **0.55p** (0.50p + 0.05p bottom block; net −0.15p vs separate TAB-4 at 0.20p). ⚠️ **Phase 5→7 open flag (RT V1)**: elsarticle/journal sideways-table support unverified; fallback ladder (r3, updated): (a) \small + tabcolsep + dataset abbreviations → (b) ~~absorb Table 4 as a bottom row-group~~ **executed as D-010 ①** → (c) single-metric column only with orchestrator V3 re-decision.
- **Dependency (inherited from TAB-4)**: bottom block requires the standard-split run (EXPERIMENT_EXECUTION_TODO item 3).

### TAB-3 — Ablation study (v2-r2: 4 confirmed rows — D-010 ②)
- **Type**: table | **Location**: §4.3, after the lead sentence
- **Caption (complete, v2-r2)**: "Table 3. Ablation study. PA%K-AUC F1 for each model variant on [3–4 representative datasets]. Row 2 (w/o GRL) removes the GRL classifier and reversal but retains the anomaly-patch OD-loss exclusion, isolating the net effect of active adversarial suppression. Extended variants (feature matching, Teacher-only warmup, symmetric decoder) are in Appendix §B.5 (Table B.4)."
- **Rows (4, confirmed)**: 1 Full model (CSMAD) / 2 w/o GRL (OD-exclusion retained) / 3 w/o anomaly-priority masking / 4 w/o OD loss
- **Columns**: Variant | Dataset-A | Dataset-B | Dataset-C | [Dataset-D] | Avg. — all cells [X.XX]; dataset count = NUM-020
- **Size assumption**: half-width ≈ **0.20p** (was 0.25p at 7 rows)
- **Note (v2-r2)**: former rows 5/6/7 (w/o FM loss, w/o Teacher warmup 250→0, symmetric decoder Teacher 2L / Student 2L) demoted to **TAB-B4** per D-010 ② — no longer conditional on run completion for body inclusion; their §4.3 prose paragraphs moved to Appendix §B.5 (NUM-024/NUM-025 relocated). The symmetric-decoder quantification remains load-bearing for contribution bullet 3 — now supported from §B.5; if its run is unavailable at publication, bullet 3 is stated as a design principle (Phase 6 rule unchanged, landing spot already B.5). Row 3 label unified from internal identifier "w/o force_mask_anomaly" to "w/o anomaly-priority masking" (terminology pass; INTEGRATION_REPORT §4-10).

### TAB-4 — Protocol-effect analysis [ABSORBED — D-010 ①; audit entry, no body marker]
- **Status (v2-r2)**: absorbed into **TAB-2** as a bottom protocol-effect row-group (PAGE_BUDGET compression strategy 2 / TAB-2 fallback (b), executed). The `[TAB-4]` marker was removed from §4.2; the §4.2 protocol-effect narrative is retained with references updated to "the bottom block of Table 2" (also §4.1.4 and §4.4). Merged caption, row structure, and the standard-split-run dependency now live in the TAB-2 entry above.
- **Original spec (audit trail)**: "Table 4. Protocol-effect analysis. Performance of CSMAD and [N] representative unsupervised baselines (Q3 condition) under a standard clean-train split (condition i) and the contaminated protocol (condition ii). Both conditions are evaluated on the same held-out test set. ... Metric: PA%K-AUC F1." Rows {CSMAD, Baseline A, Baseline B, [Baseline C]} × {standard, contaminated}; half-width ≈ 0.20p; baseline count = NUM-014.
- **Net budget effect**: −0.15p (TAB-2 0.50p → 0.55p; standalone 0.20p removed).

## 3. Inline numeric placeholders (NUM-001 … NUM-030)

| ID | Location (§ / paragraph) | Token | Content to fill | Expected value / source | Sync group |
|----|--------------------------|-------|-----------------|------------------------|------------|
| NUM-001 | Abstract, sentence 6 | [N] | # benchmark dataset families in main experiments | "six" if all families complete (protocol constant; §4.1.1 states six as real) | A |
| NUM-002 | Abstract, sentence 6 | [N] | total # baselines | 26 (22 unsup + 4 weak) if WS runs complete; else "22 unsupervised" | B |
| NUM-003 | Highlights, bullet 5 | [N] | dataset-family count (was hard-coded "six" in draft — converted for consistency) | = NUM-001 | A |
| NUM-004 | §1 contribution bullet 4 | [N] | dataset-family count | = NUM-001 | A |
| NUM-005 | §1 contribution bullet 4 | [N] | total baseline count | = NUM-002 | B |
| NUM-006 | §4.2 ¶1 | [N]×2 | ranking summary: wins out of 6 families on PA%K-AUC F1 and on VUS-PR | Table 2 results | — |
| NUM-007 | §4.2 ¶1 | [X.XX]×2 | CSMAD averages (PA%K-AUC F1, VUS-PR) across families | Table 2 results | — |
| NUM-008 | §4.2 ¶1 | [X.XX] | avg margin over best unsupervised baseline, PA%K-AUC F1 | Table 2 results | — |
| NUM-009 | §4.2 ¶1 | [X.XX] | avg margin over best unsupervised baseline, VUS-PR | Table 2 results | — |
| NUM-010 | §4.2 ¶2 | [X.XX] | CSMAD PA%K-AUC F1 on PSM | Table 2 results | — |
| NUM-011 | §4.2 ¶2 | [X.XX] | best unsupervised baseline PA%K-AUC F1 on PSM | Table 2 results | — |
| NUM-012 | §4.2 ¶2 | [X.XX] | CSMAD PA%K-AUC F1 on SWaT excl22 | Table 2 results | — |
| NUM-013 | §4.2 ¶3 | [X.XX]×2 | CSMAD vs NRdetector margins (PA%K-AUC F1, VUS-PR, avg) | Table 2 results (Q1 NRdetector) | — |
| NUM-014 | §4.2 protocol-effect ¶ *(v2-r2: Table 2 bottom block — D-010 ①)* | [N] | # representative baselines in the protocol-effect block of Table 2 | design choice (2–3) + run | — |
| NUM-015 | §4.2 protocol-effect analysis ¶ | [X.XX] | CSMAD clean-train average (protocol-effect datasets) | standard-split run | — |
| NUM-016 | §4.2 protocol-effect analysis ¶ | [X.XX] | best unsupervised baseline clean-train average | standard-split run | — |
| NUM-017 | §4.2 protocol-effect analysis ¶ | [X.XX] | CSMAD contaminated-protocol average | Table 2 subset | — |
| NUM-018 | §4.2 protocol-effect analysis ¶ | [X.XX] | CSMAD gain, standard → contaminated | derived NUM-017 − NUM-015 | — |
| NUM-019 | §4.2 protocol-effect analysis ¶ | [X.XX] | best unsupervised baseline change across conditions | standard-split run | — |
| NUM-020 | §4.3 lead | [N] | # datasets in ablation table | design choice (3–4) + runs | — |
| NUM-021 | §4.3 Row-3 ¶ | [X.XX] | PA%K-AUC F1 drop, w/o anomaly-priority masking (avg) | ablation run | — |
| NUM-022 | §4.3 Row-4 ¶ | [X.XX] | PA%K-AUC F1 drop, w/o OD loss (avg) | ablation run | — |
| NUM-023 | §4.3 Row-2 ¶ | [X.XX] | PA%K-AUC F1 difference, row 2 vs row 1 (avg) | ablation run | — |
| NUM-024 | Appendix §B.5 symmetric-decoder ¶ *(v2-r2: moved from §4.3 — D-010 ②)* | [X.XX] | PA%K-AUC F1 drop, symmetric decoder | symmetric-decoder run (load-bearing, bullet 3) | — |
| NUM-025 | Appendix §B.5 FM-regularizer ¶ *(v2-r2: moved from §4.3 — D-010 ②)* | [X.XX] | PA%K-AUC F1 drop, w/o FM loss | FM ablation run | — |
| NUM-026 | §4.4 Results lead | [N] | # datasets in Fig. 3 | sparsity sweep design (2–3) | — |
| NUM-027 | §4.4 Results ¶ | [gradually / monotonically] | qualitative descriptor of degradation shape | Fig. 3 results | — |
| NUM-028 | §4.5 lead | [N] | # datasets in Fig. 4 | visualization design (2) | — |
| NUM-029 | §5, sentence 4 | [N] | dataset-family count | = NUM-001 | A |
| NUM-030 | §5, sentence 4 | [N] | total baseline count | = NUM-002 | B |
| NUM-031 *(v2 신설)* | Appendix §B.3 | [X.XX] | measured wall-clock overhead factor, leave-one-out vs single-mask inference | B.3 cost measurement; **sync condition**: if materially below 50, soften §5 "approximately 50×" (see §5 audit-trail row) | — |

**Sync groups** (must resolve to a single value each):
- **A (dataset count)**: NUM-001 = NUM-003 = NUM-004 = NUM-029, AND must match the hard-coded "six families / 113 learning units / 114 evaluation units" in §4.1.1 + "six dataset families" in §4.2. If any family is dropped at submission, §4.1.1 constants must be edited in the same pass.
- **B (baseline count)**: NUM-002 = NUM-005 = NUM-030, AND must match "26 baselines / 22 unsupervised / 4 weakly supervised" hard-coded in §4.1.2–§4.1.4 and Table 2 row structure. Weakly-supervised GPU runs incomplete as of 2026-06-11 — if still incomplete at submission, all of group B and the §4 constants fall back to "22 unsupervised baselines" and Table 2 loses group 6.

## 4. Inline text placeholders (TXT)

| ID | Occurrences | Content to fill |
|----|-------------|-----------------|
| TXT-001 | Appendix §A.1, Environment paragraph *(v2: relocated from §4.1.2 per D-009)* | GPU model used for all experiments (fill from experiment metadata; do not guess) |
| TXT-002 | Abstract (final sentence), Appendix §A.1 Environment *(v2: relocated from §4.1.2)*, §5 (final sentence) | Code repository URL (release upon acceptance) — three occurrences must be identical |

## 5. Resolved during integration (no longer placeholders — audit trail)

| Origin | Resolution | Basis |
|--------|------------|-------|
| front_intro_conclusion PH:NUM-007 (inference cost multiplier, §5) | Resolved to "approximately 50×" in §5 | Protocol constant N=50 patches (271_CONFIG_TRUTH §VIII); §3.6 states "approximately N"; FLOPs multiplier is derivable, not an experimental result. Wall-clock figure is now placeholder **NUM-031** (Appendix §B.3, v2) — if the measured factor deviates materially below 50, soften §5 wording to "up to 50×". |
| method.md [NUM-r_m] | Realized as real values in §4.1.2 ("masking ratio 0.15", "8 masked patches") | 271_CONFIG_TRUTH §VIII (`masking_ratio = 0.15`, $|M|=8$, $|V|=42$) |
| method.md [NUM-arch] | Realized as real values in §4.1.2 (L=500, s=10, N=50, d=512, 4/3/2 layers, 8 heads, ff 2048, dropout 0.15) | 271_CONFIG_TRUTH §VIII |
| method.md [NUM-c] | Realized as real value in §3.6 Eq. (11) ($c = 4$) | 271_CONFIG_TRUTH §VIII (`score_recon_disc_ratio = 4.0`) |

## 6. Completeness cross-check

- **v1 (integrator scan, 2026-06-11)**: MANUSCRIPT_v1.md regex scan: 30/30 NUM markers, 2 TXT IDs (4 occurrences), 4 FIG, 4 TAB — all present in this registry; no orphan `[X.XX]`/`[N]` token without a PH comment; no PH comment without a registry row.
- Section-draft placeholder blocks fully absorbed: front (7 entries → NUM-001..005, NUM-029..030 + 1 resolved), experiments (23 NUM + 4 TAB + 2 FIG), method (1 FIG + 3 resolved constants), related_work (0 numeric; its scoping-confirmation notes are tracked in INTEGRATION_REPORT §6 residual issues).
- **v2 (surgeon scan, 2026-06-11)**: MANUSCRIPT_v2_draft.md regex scan: 31/31 NUM markers (NUM-031 added), 2 TXT IDs (4 occurrences: Abstract, §A.1 ×2, §5), 4 body FIG + 1 appendix FIG (FIG-B1), 4 body TAB + 8 appendix TAB markers (TAB-A3/A6/A7/A8, TAB-B1..B4), 1 ALG (ALG-C1) — all present in this registry (§1–4, §7); realized appendix tables audited in §7.3.
- **v2-r2 (surgeon r2 scan, 2026-06-11, post D-010)**: 31/31 NUM markers (NUM-024/025 now in Appendix §B.5; NUM-014..019 reference the Table 2 bottom block), 2 TXT IDs (4 occurrences, unchanged), 4 body FIG + 1 appendix FIG, **3 body TAB** (TAB-1/2/3 — TAB-4 marker removed by absorption, audit entry retained in §2) + 8 appendix TAB, 1 ALG — zero unmatched markers.

## 7. Appendix floats and placeholders (v2 신설 — D-009 / R3 / blueprint §8)

### 7.1 Appendix placeholder floats (complete captions + content specs)

#### TAB-A3 — Per-baseline hyperparameters (Appendix §A.1)
- **Caption (complete)**: "Table A.3. Hyperparameters of all 26 baselines. Each method retains the settings of its original implementation or publication preset; deviations from the unified pipeline (window size, epochs, batch size) are listed explicitly. DAGMM follows the simplified TranAD-repository re-implementation (GMM energy term omitted)."
- **Content spec**: 26 rows (method tiers of §4.1.4); columns = window size, learning rate, batch size, epochs, key model-specific parameters. **Fill from the comparison-pipeline model configurations (single source); no values invented (A8).**
- **Size**: ~0.8–1.2p (appendix; outside the 9p body count).

#### TAB-A6 — SWaT dual-condition results (Appendix §A.4)
- **Caption (complete)**: "Table A.6. SWaT dual-condition results: all five metrics for CSMAD and all baselines under the full condition and the excl22 condition (Section 4.1.1). Same trained models and identical scores in both conditions; only the evaluation mask differs. The excl22 best epoch is selected independently under the shared criterion."
- **Content spec**: rows = 27 methods; column groups = {full, excl22} × 5 metrics. All cells [X.XX] pending queue completion. ~0.3p.

#### TAB-A7 — Full multi-metric results (Appendix §A.5)
- **Caption (complete)**: "Table A.7. Complete multi-metric results for all methods and dataset families: PA%K-AUC AUC-PR, VUS-ROC, Affiliation F1, and PA F1 (oracle threshold; reported for comparability only, never used for ranking — Section 4.1.3). PA%K-AUC F1 and VUS-PR appear in Table 2."
- **Content spec**: 27 methods × 7 dataset columns × 4 metrics; cells [X.XX] pending. ~0.6–0.8p.

#### TAB-A8 — Per-entity results (Appendix §A.6)
- **Caption (complete)**: "Table A.8. Per-entity results (PA%K-AUC F1 / VUS-PR) for SMD (28 machines), SMAP (54 channels), and MSL (27 channels). Macro-averages over entities equal the corresponding family columns of Table 2."
- **Content spec**: 109 entity rows (28+54+27) × 2 metrics; cells [X.XX] pending. ~0.6–1.0p.

#### TAB-B1 — Q1 condition comparison (Appendix §B.1)
- **Caption (complete)**: "Table B.1. Q1 (full contaminated training) condition results for all 22 unsupervised baselines. Each method trains on the identical contaminated training stream used by CSMAD (no anomaly excision; labels unused) and is evaluated on the identical held-out evaluation half. Metrics: PA%K-AUC F1 and VUS-PR per dataset family; Δ columns give the change relative to the Q3 condition of Table 2 (positive = Q1 better). The CSMAD row is repeated from Table 2 for reference, as CSMAD trains on the contaminated stream in both conditions."
- **Content spec**: 22 baseline rows + CSMAD reference row; columns = families × {PA%K-AUC F1, VUS-PR, Δ vs Q3}; cells [X.XX] pending (Q1 runs registered in the comparison queue). Supports the R31 volume-asymmetry acknowledgment (§4.1.4). ~0.5p.

#### TAB-B2 — Epoch-budget sensitivity (Appendix §B.2)
- **Caption (complete)**: "Table B.2. Epoch-budget sensitivity. PA%K-AUC F1 of [N] representative unsupervised baselines trained for 10 (main budget), 50, and 100 epochs, and of CSMAD trained for 500 (main budget) and a reduced budget, on [N] representative datasets; best-epoch selection identical to the main protocol (Section 4.1.2)."
- **Content spec**: defends the §4.1.2 epoch-asymmetry disclosure (ADV BLK-005); runs pending (EXPERIMENT_EXECUTION_TODO candidate — blueprint §6.3 r3 권고: 실측 격상 시 rebuttal 화력 증가). ~0.2p.

#### TAB-B3 — Computational cost (Appendix §B.3)
- **Caption (complete)**: "Table B.3. Computational cost of CSMAD inference: per-window forward FLOPs, end-to-end wall-clock evaluation time, and peak GPU memory for leave-one-out masking versus single-mask scoring, measured on [N] representative datasets (hardware of Appendix §A.1)."
- **Content spec**: measured values pending; the wall-clock overhead factor is NUM-031 (sync condition with §5 "approximately 50×"). ~0.2p.

#### FIG-B1 — Parameter sensitivity (Appendix §B.4)
- **Caption (complete)**: "Figure B.1. Parameter sensitivity. PA%K-AUC F1 as a function of (left) the score combination ratio c around its default 4 and (right) the masking ratio ρ around its default 0.15, on [N] representative datasets; all other settings fixed to the main configuration."
- **Content spec**: two panels, one line per dataset; sweep grids TBD at run design; data pending. ~0.3p.

#### TAB-B4 — Extended ablations (Appendix §B.5; v2-r2: definitive host of former Table 3 rows 5/6/7 — D-010 ②)
- **Caption (complete, v2-r2)**: "Table B.4. Extended ablations: the variants beyond the confirmed rows of Table 3 — w/o FM loss, w/o Teacher-only warmup (250→0), and a symmetric decoder (Teacher 2L / Student 2L) — and a Teacher-decoder depth sensitivity study (3/2/1 layers against the 2-layer Student). PA%K-AUC F1 on the ablation datasets of Table 3."
- **Content spec**: hosts the three demoted variants unconditionally (D-010 ② — demotion no longer tied to run completion) plus depth sensitivity; supporting prose in §B.5 carries NUM-024 (symmetric decoder; load-bearing for contribution bullet 3) and NUM-025 (FM loss). Runs pending. ~0.25p.

#### ALG-C1 — Training pseudocode (Appendix §C.3)
- **Content spec**: single algorithm block covering (1) preprocessing incl. SWaT constant-column removal (45 = 51 − 6); (2) anomaly-priority masking (Eq. C.5); (3) Teacher-only gating for epochs < 250; (4) loss assembly (Eq. 3) with adaptive weights (Eq. C.4) and reversal schedule (Eq. C.1); (5) per-epoch evaluation every 5 epochs with best-epoch tracking. Draft in Phase 6/7 from the canonical training loop; no behavioral invention. ~0.2p.

### 7.2 Appendix partial placeholders (inside otherwise real tables)
- **Table A.4 (per-entity dataset statistics, §A.3)**: SMD row cells "[per-machine]" (train/test lengths, train AR) pending per-machine extraction; all other rows are real values (EXPERIMENT_PROTOCOL_TRUTH §①).

### 7.3 Realized appendix floats (real protocol constants — audit, no placeholders)
| Float | Location | Source of values |
|---|---|---|
| Table A.1 (CSMAD configuration) | §A.1 | 271_CONFIG_TRUTH r4 §VIII (architecture, masking, training, loss, GRL, scoring). ⚠️ test stride 49 row carries a Phase 6 verification flag (v1 said "1"; see SURGERY_REPORT_v2 §5) |
| Table A.2 (training budgets) | §A.1 | EXPERIMENT_PROTOCOL_TRUTH r4 §④ (500/10/50 epochs; eval 5/1/1; criterion pak_auc_f1; CSMAD batch 1024 — [271c] metadata). Baseline batch column = "model-specific (original presets; Table A.3)" — per-model values live in `comparison/baseline_common.py` MODEL_CONFIGS (32–512 across models). ⚠️ v2-r3 correction: the former "batch 1024/512" source note was erroneous — no baseline batch 512 exists in EXPERIMENT_PROTOCOL_TRUTH (method-truth B-6) |
| Table A.5 (SMAP/MSL split shifts) | §A.3 | EXPERIMENT_PROTOCOL_TRUTH r3 §② measured table (D-16 +166 / M-1 −39 / M-2 −39 / S-2 +8; SMAP 0/54) |
| §A.4 region-22 statistics (prose) | §A.4 | EXPERIMENT_PROTOCOL_TRUTH r3 §⑥ ([2,869, 38,769), 35,900 pts, 83.75%, 15.96%, 19.05→3.68%) |
| Table C.1 (input dimensionality) | §C.2 | EXPERIMENT_PROTOCOL_TRUTH r3 §① / 271_CONFIG_TRUTH r4 §III (45/123/123/25/29–36/25/55; d_model=512 fixed) |
| Table C.2 (notation summary) | §C.4 | PAPER_BLUEPRINT §9.1 (symbols realized with manuscript equation numbers) |
