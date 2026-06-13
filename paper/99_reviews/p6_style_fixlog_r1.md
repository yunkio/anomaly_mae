---
phase: 6
agent: style-fixer
round: r1
last_modified: 2026-06-11
target: paper/05_manuscript/MANUSCRIPT_v3.md (created from MANUSCRIPT_v2.md; v2 preserved)
inputs:
  - paper/06_style_audit/AI_PHRASING_LEDGER.md (52 itemized entries; header claims 38 — see §0.3)
  - paper/06_style_audit/STYLE_AUDIT_A.md (88 itemized entries; summary table claims 91 — see §0.3)
  - paper/06_style_audit/STYLE_AUDIT_B.md (67 findings incl. 5 PASS)
  - paper/06_style_audit/TERMINOLOGY_AUDIT.md (18 issues; 9 actionable + flags)
also_updated: paper/05_manuscript/PLACEHOLDER_REGISTRY.md (v3-r1)
constraint: STYLE ONLY — zero technical-meaning changes; all placeholder markers/IDs intact
  (verified: PH:NUM 31/31, PH:TXT 4 occurrences, FIG 5, TAB 11, ALG 1, \cite 89 — identical to v2)
---

# P6 Style Fixlog r1

## 0. Q1/Q3 확정 매핑 (최우선 기록)

### 0.1 Verified semantics (ground truth wins)

| Code label | Code meaning (verified) | **Adopted paper term** |
|---|---|---|
| **Q3** (`variant="normalonly"`) | Labeled anomaly regions **excised** from the contaminated training stream; surviving normal segments concatenated segment-aware; train labels all 0. Main-comparison condition for the 22 unsupervised baselines. | **the anomaly-excised condition** |
| **Q1** (`variant="full"` / minmax full) | Training on the **full contaminated stream without excision** (anomalies present in training; labels unused by unsupervised methods). Condition for the 4 weakly supervised baselines and the Appendix B.1 complementary comparison. | **the contaminated-training condition** |

### 0.2 Verification trail (semantics are NOT inverted)

- `EXPERIMENT_PROTOCOL_TRUTH.md` r4 §③ ([N-COMP] §2.2 원문): "**Q1 (minmax full)**: train data 안에 anomaly 포함 (실제 운영 환경 가정) — 라벨 미사용 unsupervised 그대로. **Q3 (minmax normalonly)**: train data에서 anomaly region 제거 → segment-aware concat — 라벨을 '제거'라는 형태로 활용한 unsupervised의 최선 (R12)."
- `comparison/data/unified_loader.py:34-36`: "'normalonly': Remove anomaly regions from training data" (code read-only; inspected only).
- MANUSCRIPT_v2 §4.1.4: "the Q3 (normal-only) condition … labeled anomaly regions are excised"; §B.1: "Q1 … train on the full contaminated stream without excision". Internally consistent with the truth doc.
- **Adjudication of the tasking hypothesis**: the briefing suspected the terminology audit's mapping was meaning-inverted and proposed "Q3 → the contaminated condition / Q1 → the standard-split condition". Verification shows the **opposite**: TERMINOLOGY_AUDIT §2's direction (Q3 → anomaly-excised, Q1 → contaminated-training) is **correct**, and the briefing's example mapping would itself have been the inversion (it conflates Q1/Q3 with the §4.2 protocol-effect conditions (i) standard clean-train split / (ii) contaminated protocol, which are a *different* axis: original-train-file-only vs test-prefix-incorporated). The briefing's example was therefore **not adopted**; the truth-verified mapping above was applied.
- Both Q1 and Q3 operate **on top of** the contaminated re-split protocol; they differ only in whether labeled anomaly regions are excised from the training stream. The §4.2 protocol-effect "standard clean-train split" wording is untouched and remains distinct.

### 0.3 Application

All 11 Q3/Q1 occurrences listed in TERMINOLOGY_AUDIT §2 replaced (§4.1.4 ×4, §4.2 ×2, §A.3 ×1, §B.1 heading + ×2 + caption), plus the registry captions (FIG-1, FIG-3, TAB-2 caption + row structure, TAB-B1 title/caption/spec, NUM-013 note). Both terms are **bold-defined at first use in §4.1.4** ("Comparison conditions" paragraph; the weakly-supervised sentence forward-references "(defined below)"). §B.1 heading → "Contaminated-Training (No-Excision) Condition Results". Remaining Q1/Q3 strings exist only in YAML frontmatter history/notes (metadata, not body).

Audit-count note: AI ledger header says 38 detections but itemizes 52 (11 MUST + 29 SHOULD + 12 FLAG per entry headers; its own §IV table is internally inconsistent). STYLE_AUDIT_A summary says 91/25-MUST but itemizes 88 entries with 17 explicitly-marked MUST. **All itemized entries of all four audits were processed** (전수 처리); statistics below are over itemized entries.

---

## 1. AI_PHRASING_LEDGER — 52/52 processed

### 1.1 MUST-FIX (11/11 applied)

| ID | Disposition | Note |
|---|---|---|
| A-01 | APPLIED (merged A-003/B-001) | Abstract: "loss bifurcation between … paths" → "a Student imitation loss restricted to normal patches"; third item made parallel ("gradient-reversal suppression of …") |
| I-01 | APPLIED (merged S1-001) | Opening em-dash splice removed; "multi-channel" double epithet dropped |
| I-03 | APPLIED | "share an implicit assumption" → "treat the training data as drawn entirely from normal operations" (overrides B-007 PASS; see §5-C1) |
| I-04 | APPLIED (merged S1-004/B-008) | Colon-launch removed; "architectural pathway for leveraging the information carried by" → "mechanism for exploiting"; dash continuation split |
| I-06 | APPLIED (merged S1-005) | Triple-clause compound → 3 sentences; "data protocol" → "data partition"; stranded preposition removed |
| I-08 | APPLIED-MODIFIED | Reframed as design rationale; pointer "(Section 4.3)" used instead of ledger's "as Section 4.3 confirms" (results pending — A8; avoids asserting confirmation pre-fill) |
| A-01 dup of abstract — see above | | |
| RW-06 | APPLIED (merged S2-004; B-017 "include") | 60-word PU sentence split |
| RW-08 | APPLIED (covers B-019) | "deep representation learning informed by label signals remains rare" → "methods that incorporate anomaly labels into the representation learning objective itself are rare" |
| M-03 | APPLIED (merged S3-003/B-032) | "underpinning" removed; sentence split; "receive a stop-gradient copy" |
| E-05 | APPLIED | §4.2 four-clause result compound split into two sentences (placeholders untouched) |
| C-01 | APPLIED (merged S5-001/B-062) | "underexplored setting" → named setting; split; present tense; "unsupported" → "unaddressed" |

### 1.2 SHOULD (29: 24 applied, 3 partial, 2 kept-with-reason)

| ID | Disposition | Note |
|---|---|---|
| A-02 | APPLIED | Abstract robustness claim → graceful-degradation + floor anchor (merged B-005) |
| H-01 | APPLIED-MODIFIED | Rewritten within the ≤125-char Elsevier budget (ledger's revision was 150+ chars): "CSMAD is competitive with unsupervised baselines under five metrics on [N] datasets, degrading gradually with label sparsity." (125) |
| I-02 | APPLIED | Split per S1-002; "has been" retained (see §5-C3) |
| I-05 | APPLIED | emphatic "do" removed |
| I-09 | APPLIED | Split; "the adversarial suppression of (c) closes this pathway" (merged B-011) |
| I-10 | PARTIAL | "robust detection" → "detection degrades gradually toward the unsupervised floor" applied; **"comparable to or surpassing" upgrade REJECTED** — claim-strength change pending results (A8); body keeps "competitive" |
| I-11 | APPLIED | Boilerplate roadmap opener dropped (minimal form) |
| RW-01 | APPLIED | "have matured into" → "fall into" (overrides B-013 PASS; §5-C2) |
| RW-03 | APPLIED (merged S2-002) | "Despite this breadth, every family above" → "All of these families, however" |
| RW-04 | APPLIED (merged S2-003/B-016) | Dash pair → parenthetical "(the contaminated setting of Section 3.1, …)"; "consequently"; "noise that corrupts the learned normality model" |
| RW-07 | APPLIED | "these ideas" → "these techniques" |
| RW-09 | APPLIED (merged S2-006/B-021) | "labels here enter the gradient of the masked-reconstruction pretext task, shaping …" |
| RW-10 | APPLIED-MODIFIED | Dash interruption removed by moving "through gradient reversal" to clause end; "into the gradient of" **retained** (S2-010's weakening rejected — §5-C4); TSAD → MTSAD |
| RW-12 | APPLIED | "this architectural paradigm" → "this asymmetric teacher–student masked autoencoder design" (lowercase per new capitalization rule — prior-work referent); "actively" removed |
| M-01 | APPLIED-MODIFIED (merged S3-002) | Nested dashes → colon + new sentence; ledger's added clause "and distinguishing it from anomalous deviations" NOT adopted (new content) |
| M-04 | APPLIED (merged S3-005) | Subject made explicit; *around/through* metaphor → "gains little experience reconstructing anomalous correlation patterns" |
| M-05 | APPLIED (merged S3-007/B-035) | "faithfully learns" → "accurately captures"; split |
| M-06 | APPLIED | Dash dramatization → ", thereby reducing the discrepancy where it is most diagnostic" |
| E-01 | APPLIED (merged S4-002/B-047) | "genuinely" deleted; ratios sentence extracted |
| E-03 | APPLIED-MODIFIED | Split per ledger; original claim "prevents any single failure mode from going undetected" **retained** over ledger's "dominates the comparison" (claim change — §5-C9) |
| E-04 | KEPT (reason) | Ledger itself concludes "No mandatory fix"; standard table-reference sentence; only the missing "are" added (S4-015 MUST) |
| E-06 | APPLIED (merged S4-018/B-058) | "support robustness" colon-list → "bound this degradation. First/Second/Third" sentences |
| C-02 | APPLIED (merged S5-004/B-064) | "natural avenue" removed; specific direction added ("amortized inference with learned masking schedules or sparse patch selection") — speculative future-work direction, no numbers |
| C-03 | PARTIAL | "suggests extending" → "motivates a fully unsupervised variant … obtained by disabling the gradient-reversal pathway"; ledger's trailing "which may inherit …" REJECTED (adds new speculative content) |
| AP-A-01 | APPLIED (merged SA-002) | "disclosed" → "reported"; "states" → "summarizes" (manuscript-wide "disclosed" count now 0) |
| AP-A-02 | APPLIED (merged SA-003) | Split; "use a single run" |
| AP-A-04 | KEPT (reason) | "therefore" is logically accurate (construction → consequence); sentence retained for clarity; SA-007's "test-stream prefixes" applied instead |
| AP-B-01 | PARTIAL (merged SB-001/B-068) | "For completeness," removed; dash continuation flattened; ledger's deletion of "contextualizing the training-volume asymmetry" REJECTED (R31-linked content) |
| AP-B-03 | APPLIED (via SB-002) | Dash parenthetical → coordinated clause; "otherwise unchanged protocol" retained for precision over ledger's "same protocol" |

### 1.3 OK-FLAG (12: 6 applied as cheap improvements, 5 recorded-keep, 1 N/A)

| ID | Disposition |
|---|---|
| I-07 | APPLIED via S1-006 ("is threefold:" → "is that … three distinct learning signals:") |
| RW-02 | KEPT — colon-of-elaboration acceptable per ledger's own judgment |
| RW-05 | KEPT — clean positioning sentence |
| RW-11 | KEPT — "strong transferable representations" acceptable paraphrase |
| M-02 | KEPT — colon-enumeration of concrete components is corpus-attested |
| M-07 | APPLIED via S3-015/B-045 ("context-dependent reconstruction variance") |
| E-02 | KEPT — clean (excl22 label clause restructured for R24 anyway) |
| E-07 | APPLIED — dash → colon before the 4-trace list |
| AP-A-03 | APPLIED — "consume" → "are computed on" |
| AP-B-02 | APPLIED — "disclosed" → "reported" |
| AP-C-01 | N/A — placeholder comment, not publishable prose |
| AP-C-02 | APPLIED via SC-004/B-071 ("introduced as part of the present design rather than adopted from prior work") |

---

## 2. STYLE_AUDIT_A — 88/88 processed

### 2.1 MUST-FIX (17/17 applied; 1 applied-corrected)

| ID | Disposition | Note |
|---|---|---|
| A-004 | APPLIED-SYNTH | Possessive nominalisation removed while keeping the mimicry semantics (merged B-002): "the capacity-limited Student mimics the Teacher less faithfully on anomalous correlation patterns than on normal ones" — also resolves the B-012-class comparative ambiguity in the Abstract |
| S1-002 | APPLIED | Split; "has been" kept (§5-C3) |
| S1-005 | APPLIED (merged I-06) | 69-word fusion → 3 sentences |
| S1-009 | APPLIED-SYNTH (merged B-012) | "fails more severely on anomalous correlation patterns than on normal ones" — unambiguous comparative without the audit's clunkier "than it fails on" |
| S2-004 | APPLIED | PU sentence split |
| S3-006 | APPLIED (B-034) | Auxiliaries restored ("are added", "is passed") |
| S3-008 | APPLIED | 101-word dual-λ sentence → 3 sentences; commas instead of dashes; **B-036's "exponential moving average" NOT imported** (§5-C5) |
| S3-010 | APPLIED-SYNTH (merged B-038 + BCE expansion) | 90-word sentence → 2; "takes as input … and predicts"; "focal-style binary cross-entropy (BCE) variant" |
| S4-001 | **APPLIED-CORRECTED** | The audit's count-resolution ("WaDi A1 and A2" as two families) **contradicts Table 1/TAB-1 spec** (WaDi is one family with entities A1/A2; SMAP and MSL are separate families; 6 = SWaT, WaDi, PSM, SMD, SMAP, MSL; 113 = 1+2+1+28+54+27). Fixed by listing "… SMD, SMAP, and MSL" (six names); "(below)" → "(detailed below)"; "learning units" → "entities" (B-046) and "evaluation units" → "evaluation conditions" |
| S4-010 | APPLIED-MODIFIED | 88-word threshold sentence → 3; **kept "follows the … mechanism introduced by"** over the audit's "conceptually related to" (citation-strength preservation); redundant α re-definition removed; R30 disclosure intact |
| S4-011 | APPLIED-MODIFIED (merged B-050 i–iv) | 155-word metric sentence → 5 sentences (one per metric); "complementary under class imbalance" (dropped by the audit's revision) **re-added** — content preservation; "TSAD" spelled out in the liu2024elephant claim (scope preservation, §5-C8) |
| S4-014 | APPLIED | "are concatenated" restored (B-052); + R24 condition rename |
| S4-015 | APPLIED | "are" restored |
| S5-001 | APPLIED | Present tense; split (merged C-01/B-062) |
| S5-002 | APPLIED (merged B-063) | "We proposed" dropped → "CSMAD integrates …"; "on top of" → "built on" |
| SA-004 | APPLIED | "$K = 100$ recovers point-wise scoring" |
| SC-001 | APPLIED | "is" ×2 restored; "$\tau$ is its normalized progress" |

### 2.2 SHOULD (42 itemized: 38 applied/merged, 4 partial)

Applied (with merge partners noted in §1/§3 where shared): A-001, A-002 (alternative wording "this partial labeling" — lighter than the audit's "label–normal coexistence structure"), A-003, A-005, H-001 (within char budget), H-002 (shortened to 121 chars), S1-001, S1-003 (kept "auxiliary training objectives", not "pretraining" — scope), S1-004 (subsumed by I-04's stronger fix), S1-008, S2-003, S2-006, S2-008, S2-011 (tense), S2-012, S2-013 (tense), S2-015, S3-002, S3-003, S3-007, S3-011 (via "multiplies the gradient by $-\lambda_{\mathrm{rev}}$" — cleaner than the audit's "scales-and-negates"), S3-012, S3-013, S3-015, S4-002, S4-003 ("outward" removed — accuracy gain: Table A.5 shows shifts of both signs), S4-004, S4-005, S4-006 (commas instead of a second dash pair), S4-007, S4-008, S4-009, S4-013, S4-016 (parentheses; merged B-055), S4-017, S4-018, S4-019, S4-020, S4-021 (simplified vs the audit's "cause it to fail in tracking"; B-061's "replicate the Teacher's output" used), S5-004, SA-003, SA-008, SB-001, SB-002, SC-004.

Partial:
| ID | What was withheld | Reason |
|---|---|---|
| S2-010 | "into the representation learning of …" | Weakens the precise gradient-space novelty claim (§2.2/§3.5 axis); RW-10's restructure applied instead |
| S4-012 | dropping "only" | "only in Appendix §A.5 … never used for ranking" carries the R29 non-ranking emphasis; oracle-gloss expansion applied |
| SA-005 | "… and min–max normalization" | Original "after min–max normalization" preserves the actual computation order (normalize scores → compute VUS); "a tolerance window of 100 timesteps" applied |
| S1-007 | inline glossing of (b)/(c) | References sit one sentence after the lettered list; glossing would duplicate mechanism names — OK-FLAG-grade, kept |

(Note: S1-007 is OK-FLAG in the source; listed here because its optional fix was declined.)

### 2.3 OK-FLAG (24+ itemized: applied where cheap, otherwise recorded)

Applied: A-006 ("complementary"), S2-001 (superseded by RW-01), S2-002 (merged), S2-005 (merged B-020), S2-007, S2-009 ("by contrast" removed), S2-014 ("is adopted from"), S3-001 ("is segmented into"), S3-004 (fronted "With masking ratio ρ"), S3-005 (subject made explicit), S3-009 ("excluded from $L_{\mathrm{OD}}$"), S3-014 ("Here, … respectively"), SA-001 ("determined by"), SA-002, SA-006, SA-007, SB-003 ("degenerating under the competing objectives"), SC-002 ("multiplies the gradient by $-\lambda_{\mathrm{rev}}$"), SC-003 ("Let … be …"), SC-005 (merged B-072; "remain distinct quantities" retained — see §5-C10).
Recorded-keep: S1-006→applied actually (see I-07). No A-audit OK-FLAG was left unconsidered.

---

## 3. STYLE_AUDIT_B — 67/67 processed

### 3.1 Moderate (3/3 applied)

| ID | Disposition |
|---|---|
| B-012 | APPLIED — contribution 3 comparative disambiguated: "fails more severely on anomalous correlation patterns than on normal ones"; the audit's "making the … discrepancy a reliable anomaly signal" REJECTED in favor of original "a design intended to make" (hedged design-intent, results pending) |
| B-046 | APPLIED — "learning units" → "entities" (§4.1.1, §A.1); "evaluation units" → "evaluation conditions"; registry sync-group A updated |
| B-063 | APPLIED — §5 "loss bifurcation toward normal-only Student mimicry" → "loss bifurcation that restricts Student mimicry to normal patches" ("bifurcation toward" incoherence removed; term retained per §4 below); (ii) "converts the capacity gap" kept (acceptable per audit) |

### 3.2 Minor / Very minor (59: 45 applied or merged, 9 partial, 5 kept-with-reason)

Applied/merged: B-001 (abstract occurrence — via A-01; see §4), B-002, B-003, B-004, B-008, B-010, B-011, B-014, B-016, B-017, B-019 (via RW-08), B-020, B-021, B-023, B-024 ("produces"), B-025, B-026 ("revealing"), B-027 ("self-contained"), B-028, B-030, B-032, B-033 (via M-04 alternative — same effect), B-034, B-037 ("penalizes the discrepancy"), B-038, B-040 ("removes the requirement … match the Teacher's output at those locations"), B-041, B-042, B-044, B-045, B-047, B-049, B-050 (i–iv), B-051 ("simple or lightweight"), B-052, B-055 ("automatically inactive"), B-056 ("selective distillation signal"), B-057, B-058 ("takes its positive supervision … omit the term"), B-060 ("Figure 4 illustrates the decomposition"), B-061, B-062, B-064, B-065 (dangling "following" restructured: "nine detectors adopted from the protocol study of \cite{sarfraz2024quovadis} — …"), B-066 ("eliminating implementation-dependent metric discrepancies"), B-067, B-068, B-070 ("architectural inductive bias"), B-071, B-072.

Partial (meaning-guard — the meaning-affecting component was withheld):
| ID | Withheld component | Reason |
|---|---|---|
| B-005 | "demonstrating robustness under reduced label availability" | Unanchored "robustness" (AI R-10); graceful-degradation + floor anchor used instead; "upper-bound labeling scenario" → "fully labeled regime" applied |
| B-015 | "learned association patterns and the observed input" | The sentence covers DCdetector (contrastive multi-scale views) as well as Anomaly Transformer; the audit's AT-specific rewrite would misdescribe DCdetector. Minimal fix: "actual" → "observed" |
| B-022 | "do not address" | Strengthens the claim about prior work beyond the cited paper's hedge; "only partially address" used (preserves "limited support" strength, removes API-speak) |
| B-031 | "couples the Student encoder representation" | Factually wrong component — the GRL reads the **Student decoder's** final-layer hidden states (FIG-2 spec, §3.5); synthesized: "adversarial branch that couples the Student decoder's hidden states …" |
| B-036 | "exponential moving average from the previous epoch" | **Factually wrong** — the mechanism is the previous epoch's plain average (Eq. C.4; 271_CONFIG_TRUTH); "computed adaptively … evaluated per batch and applied as the previous epoch's average" applied |
| B-039 | "prevents the Student encoder from learning" | Same wrong-component issue as B-031; anthropomorphism removed via "penalizes anomaly-discriminative information in the Student's hidden states" |
| B-048 | full rewrite "Results are from a single run per entity; …" | Redundant with the immediately preceding sentence ("one seed-42 run per entity"); scope clause "for the main results" added via S4-006 instead |
| B-059 | "without dropping below the performance of the unsupervised baseline" | "unsupervised floor" retained as the manuscript-consistent term (used in Abstract/§1/FIG-3 caption); "This confirms that CSMAD reverts …" applied |
| B-058 | "relies on labeled windows as positive examples" | Lighter synthesis used ("takes its positive supervision exclusively from labeled windows") to avoid drifting from the precise batch-gating semantics |

Kept-with-reason: B-006 ("has been" → "is": present perfect retained — the dominance claim is historical-to-present and matches S1-002's accepted revision; §5-C3), B-018 ("through" acceptable; RW-07 fix applied to the same sentence), B-029 (borderline-acceptable per the audit itself), B-043 ("not used at inference" — concise and accurate), B-054 ("absolute points" retained — metrics are reported as [X.XX] decimals, not percentages, so "percentage points" would be wrong).

PASS entries (B-007, B-009, B-013, B-053, B-069): recorded; B-007 and B-013 were overridden by higher-priority AI-ledger findings on the same sentences (§5-C1/C2).

### 3.3 Cross-cutting "loss bifurcation" (지시된 4회 처리)

| Occurrence | Treatment |
|---|---|
| Abstract | REMOVED → operational phrasing "a Student imitation loss restricted to normal patches" (A-01 MUST) |
| §1 contribution 2 | **RETAINED as the italicized paper-specific definition** with its operational gloss ("*loss bifurcation*, which restricts the Student decoder's imitation objective to normal-patch outputs") — this is the defining use the B cross-cutting note allows |
| Highlights bullet 3 | RETAINED (named mechanism triad; definition lives in contribution 2; replacing it would exceed the 125-char budget and orphan the §3.5 heading) |
| §5 | RETAINED with restrictive gloss "loss bifurcation that restricts Student mimicry to normal patches" (B-063 fixed) |
| §4.3 "bifurcated signal" (B-056) | REMOVED → "selective distillation signal" |
| §3.5 heading "… beyond loss bifurcation" | RETAINED (uses the defined term) |

---

## 4. TERMINOLOGY_AUDIT — 18/18 processed

| # | Issue | Disposition |
|---|---|---|
| 1 (HIGH) | Q1/Q3 R24 violation (11 occurrences) | APPLIED — see §0; registry synced |
| 2 (MED) | excl22 code label | APPLIED — first use now defines it as an explicit condition label ("a condition denoted excl22"); §4.2 prose → "under SWaT's excl22 condition"; §A.4 heading → "SWaT Evaluation with Region 22 Excluded (excl22): …"; retained as compact label in tables/defined prose thereafter (12 occurrences, all post-definition) |
| 3 (MED) | TSAD undefined / mixed with MTSAD | APPLIED — consolidated to MTSAD at Highlights b1, §1 (×1), §2.1, §2.2 (final claim), §4.1.4, §A.1; general-field claims spelled out as "time-series anomaly detection" where narrowing to MTSAD would alter the cited claim's scope (liu2024elephant VUS-PR claim in §4.1.3; NRdetector-setting sentence in §2.2). 0 bare "TSAD" remain |
| 4 (MED) | `d_\text{model}` ×3 | APPLIED — all → `d_{\mathrm{model}}` (§4.1.2, Table A.1, §C.2); 0 remain |
| 5 (MED) | gradient-reversal hyphen | APPLIED — §1 contribution 2 "*gradient-reversal suppression*"; Highlights b2 reworded to the hyphenated compound; pre-nominal uses now uniformly hyphenated; bare noun phrase "gradient reversal" untouched |
| 6 (MED) | BCE first prose use | APPLIED — §3.5 "focal-style binary cross-entropy (BCE) variant" |
| 7 (LOW) | Table C.2 missing $\bar{r}$, $\bar{d}$, $c$, $\varepsilon$ | APPLIED — 4 rows added (with values $10^{-4}$ / $4$ and Eq. references) |
| 8 (LOW) | Teacher/Student capitalization undeclared | APPLIED — one-sentence rule added to the §2.3 footnote ("We capitalize Teacher and Student when referring to CSMAD's own decoder branches and use lowercase for the general teacher–student paradigm."); §2.3 lowercase prior-work usage now declared, not accidental |
| 9 (LOW) | Lab-notebook imperative prose (§A.1/§A.3) | PARTIAL — §A.1 SWaT note reordered to evidence-first ("Loading the raw CSV files … yields 51 features, so reproductions should verify …"); §A.3 boundary-aware text KEPT (audit: "no removal recommended") |
| §3.5 flag | NRdetector capitalization | KEPT "NRdetector" + Phase 7 verification flag — refs.bib title carries no method name; project truth docs ([N-COMP], EXPERIMENT_PROTOCOL_TRUTH) consistently use "NRdetector"; external PDF check deferred to Phase 7 |
| §3.3, 3.4, 3.6, 3.7 | contaminated semi-supervised / anomaly-priority masking / leave-one-out / loss-bifurcation descriptors | NO ACTION (audit: consistent); loss-bifurcation handling per §3.3 above |
| §4.2, 4.3 | italic $T$ / $\varepsilon$ convention | NO ACTION (audit: consistent/appropriate) |
| §4.4 flag | $P_n$ vs $\mathbf{P}_i$ visual closeness | RECORDED, NOT CHANGED — medium-effort notation refactor (Eqs. 1–2, §3.5, Table C.2) explicitly deferred to Phase 7 by the audit |
| §5.2 | "OD loss" informal | APPLIED at the §4.2 occurrence → "all-normal $L_{\mathrm{OD}}$"; §4.3 "OD-exclusion" retained (symbol-derived shorthand, audit-rated low) |
| §7.1 | "contaminated semi-supervised" coinage | NO ACTION (audit: correct handling via footnote); recorded |
| §7.4 flag | "PA%K-AUC AUC-PR" double-AUC | NOT CHANGED — audit: "do not change without author decision"; renaming would touch all table headers/specs |
| §13 flag | "DAGMM (simplified)" table-label consistency | RECORDED for Phase 6/7 results fill (registry TAB-A3/TAB-2 specs already use it) |
| §10 | $c$, $\varepsilon$ table absence | APPLIED (see #7) |

---

## 5. 충돌 조정 기록 (priority: 의미 정확성 > terminology 통일 > AI MUST > style-A 문법 > style-B 관용)

| # | Conflict | Resolution |
|---|---|---|
| C1 | I-03 (AI MUST: "share an implicit assumption" inflated) vs B-007 (PASS) | AI MUST wins → direct statement |
| C2 | RW-01 (AI SHOULD: "have matured") vs B-013 (PASS) vs S2-001 (FLAG) | AI applied ("fall into several well-defined families") |
| C3 | B-006 ("has been" → "is") vs S1-002/I-02 (keep "has been") | Meaning: the dominance claim is historical-to-present; present perfect retained; B-006 rejected |
| C4 | S2-010 ("into the representation learning of") vs original gradient-space claim | Meaning accuracy wins — the gradient-space integration **is** the novelty axis (§3.5 target/loss vs gradient space); RW-10 restructure only |
| C5 | B-036 ("exponential moving average") vs Eq. C.4 / 271_CONFIG_TRUTH (previous epoch's plain average) | Ground truth wins; B-036's EMA wording rejected |
| C6 | B-031/B-039 ("Student encoder") vs FIG-2 spec/§3.5 (GRL attaches to Student **decoder** final-layer hidden states) | Ground truth wins; synthesized wording names the Student decoder's hidden states |
| C7 | A-01 (drop "loss bifurcation") vs B cross-cutting (standardize or define) vs term-audit 3.7 (no issue) | Hybrid: defined-term retention at contribution 2/Highlights/§5 + operational phrasing in Abstract/§4.3 (§3.3 above) |
| C8 | Term-audit TSAD consolidation vs citation-claim scope (liu2024elephant, NRdetector) | Spelled-out "time-series anomaly detection" at the two general-scope claims; MTSAD elsewhere |
| C9 | E-03's "ensures that no single failure mode dominates" vs original claim | Original claim ("prevents … going undetected") retained inside the split — R29 complementarity logic unchanged |
| C10 | SC-005/B-072 "govern distinct aspects of the gradient flow" vs original "remain distinct quantities" | Original retained — the sentence's point is exactly the don't-conflate-the-two-quantities warning; "enter the gradient multiplicatively" applied |
| C11 | S4-001's family-count resolution (WaDi A1/A2 as families) vs Table 1/TAB-1 spec + entity arithmetic | Ground truth wins: families = SWaT, WaDi, PSM, SMD, SMAP, MSL (113 = 1+2+1+28+54+27); audit's mapping corrected |
| C12 | Briefing's Q1/Q3 example naming vs EXPERIMENT_PROTOCOL_TRUTH/manuscript definitions | Truth docs win (§0.2) — terminology audit's direction confirmed correct; briefing example not adopted |
| C13 | S4-010 "conceptually related to" vs original "following the … mechanism introduced by" | Citation-strength preservation: "follows the … mechanism introduced by" retained inside the split |
| C14 | Em-dash policy vs enumeration appositions (S4-001 dataset list, §4.3 further-ablations list, §5 three-paths list, §A.1 baseline tiers, §B.5) | Enumeration-apposition pairs retained (legitimate use); all clause-splicing dashes removed/split — per-section prose instances now ≤2 everywhere (Abstract 1, §1 2, §2 0, §3 2, §4 2, §5 1, App-A 2, App-B 1, App-C 0) |

## 6. 의미 보존 거부 (meaning-preservation rejections)

Full rejections: **2** (B-006, B-054).
Partial rejections (finding applied minus its meaning-changing component): **14** (I-08, I-10, C-03, M-01, AP-B-01, S2-010, S4-012, SA-005, B-005, B-015, B-022, B-031/B-039 (counted once as the encoder/decoder error, applied corrected), B-036, B-048; plus E-03/SC-005/S4-010/B-012/B-059 resolved as claim-preserving syntheses — see §5).
Style-grade declines with recorded reasons (no meaning issue): E-04, AP-A-04, S1-007, B-018, B-029, B-043, RW-02, RW-05, RW-11, M-02, E-02.

## 7. Verification summary

- Placeholder integrity: PH:NUM 31/31 (IDs unchanged), PH:TXT 4 occurrences, FIG 5, TAB 11, ALG 1, `\cite` 89 — **identical to v2**. No `[X.XX]`/`[N]` token touched.
- Q1/Q3: 0 occurrences in body/comments (frontmatter metadata only).
- TSAD (bare): 0; `d_\text{model}`: 0; "learning units": 0; "disclosed": 0.
- Highlights: 109/120/120/121/125 chars — all ≤125.
- Em-dash prose instances per section ≤2 (was 11 splices manuscript-wide per the ledger §III).
- Mandatory narratives spot-checked intact: R13 (§4.1.1 protocol motivation + NRdetector precedent), R28 (excl22 rationale + dual-condition), R29 (five-metric complementarity + PA-F1 oracle/non-ranking), R30 (threshold defense, "never used in training"), R31 (anomaly-excised fairness + volume-asymmetry + B.1), R32 (three-property degradation logic, now First/Second/Third), epoch-asymmetry & test-set-selection disclosures, dual-λ structure, Teacher-only warmup, focal-variant distinction, GRL-necessity argument (§3.5).
- Registry: PLACEHOLDER_REGISTRY v3-r1 — FIG-1/FIG-3/TAB-2/TAB-B1/NUM-013/sync-group-A synced to the renamed conditions and entity terminology.
