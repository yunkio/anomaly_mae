---
phase: 6
agent: terminology-normalizer
directives: [R5, R15, R24, R35]
last_modified: 2026-06-11
source: paper/05_manuscript/MANUSCRIPT_v2.md (full text, lines 1–777)
auxiliary:
  - paper/02_venue_study/SENTENCE_CORPUS.md Appendix A (collocation)
  - paper/01_research_understanding/EXPERIMENT_PROTOCOL_TRUTH.md §④ (metric canonical names)
status: complete — inspection-only; no manuscript edits made
---

# TERMINOLOGY_AUDIT — Phase 6 Terminology & Notation Inspection

---

## 1. Executive Summary

| Category | Issues found | Severity breakdown |
|---|---|---|
| Consistency fluctuations | 6 | 2 medium, 4 low |
| Notation inconsistencies (R5) | 2 | 1 medium, 1 low |
| Internal / code terminology (R24) | 3 | 1 high, 2 medium |
| Unnecessary abbreviations (R15) | 1 | 1 medium |
| Domain-standard check flags | 4 | 1 uncertain, 3 acceptable |
| Overly-granular prose (R35) | 2 | 2 low |

Total distinct issues: **18** (1 high, 6 medium, 9 low, 2 uncertain/flag-only).
Unification-required terms: **8** (see Section 7 Unified Term Table).

---

## 2. Q1 / Q3 Internal Code Condition Labels (R24) — VERDICT

**Verdict: Q1 and Q3 are internal condition identifiers that have leaked into the published body text. This is a direct R24 violation.**

### Evidence

The labels derive from the internal experiment configuration system:
- `comparison/data/unified_loader.py:34–36`: variant strings `"normalonly"` (→ Q3) and `"full"` (→ Q1)
- `[N-COMP] §2.2`: "Q1 (minmax full)" and "Q3 (minmax normalonly)" — internal notation in the Notion comparison dashboard
- EXPERIMENT_PROTOCOL_TRUTH.md §③: lists Q1/Q3 as Notion/code labels, not proposed paper terms

### Occurrences in manuscript body

| Line | Context | Verdict |
|---|---|---|
| §4.1.4 L389 | "evaluated under the Q1 condition only, since removing all labeled anomalies (Q3) would eliminate…" | R24 violation — in body prose |
| §4.1.4 L393 | "The main comparison uses the Q3 (normal-only) condition for all 22 unsupervised baselines" | R24 violation — in body prose |
| §4.1.4 L394 | "Q3 grants each unsupervised method this most favorable use of the labels" | R24 violation |
| §4.1.4 L395 | "Appendix §B.1 reports Q1 results for all unsupervised baselines" | R24 violation |
| §4.2 L403 | "outperforms the strongest unsupervised competitor (Q3) by…" | R24 violation |
| §4.2 L405 | "the closest weakly supervised comparison (Q1)" | R24 violation |
| §A.3 L612 | "guards the excision boundaries of the Q3 condition" | R24 violation — in appendix |
| §B.1 heading | "B.1. Q1 (Full Contaminated Training) Condition Results" | R24 violation — section heading |
| §B.1 L641 | "The Q3 condition of the main comparison" | R24 violation |
| §B.1 L642 | "the complementary Q1 condition" | R24 violation |
| §B.1 caption L646 | "change relative to the Q3 condition of Table 2" | R24 violation |

**Total: 11 occurrences across body and appendix.**

### Proposed academic replacements

| Code label | Proposed paper term | Rationale |
|---|---|---|
| Q3 (normalonly) | **anomaly-excised condition** or **clean-training condition** | Describes the operation precisely: labeled anomaly regions are excised from training; comparable to "cleaned training set" conventions in contamination-aware ML literature |
| Q1 (full) | **contaminated-training condition** or **full-contaminated condition** | Matches the paper's own "contaminated" vocabulary (already used throughout §3.1, §4.2) |

**Flag**: The parenthetical "normal-only" in L393 ("the Q3 (normal-only) condition") is already a step toward description but is not sufficient on its own — "normal-only" reads as a property of the data rather than a named experimental condition. A consistent two-word label (e.g., "anomaly-excised condition") used every time would satisfy R24. The fixer should choose one label and apply it everywhere Q3/Q1 appears.

---

## 3. Consistency Audit — Term-by-Term

### 3.1 Teacher / Student capitalization (MEDIUM)

**Observation**: The manuscript capitalizes Teacher and Student as proper names for the two decoder branches throughout the body and appendix. This is consistent within the manuscript itself.

**Issue**: Section 2.3 (Related Work), when describing prior work (SDMAE, Bergmann et al., Deng et al.), uses lowercase:
- L189: "a capacity-limited **student** decoder with a deeper **teacher** inside a masked autoencoder"
- L189: "**teacher–student** frameworks in which a **student** trained to match a pre-trained **teacher's** representations"
- L193 footnote: "whose **student** decoder branches off from within the **teacher** decoder"

The capitalization switches from lowercase (when discussing prior work) to uppercase (when discussing CSMAD's own components). This is a defensible and conventional disambiguation — capitalizing to mark the specific named components of the proposed model vs. the general paradigm. However it creates a visual inconsistency in the same paragraph and may confuse readers.

**Verdict**: Functionally defensible, but the rule must be made explicit once (e.g., "We capitalize Teacher and Student throughout to distinguish CSMAD's specific decoder branches from the general teacher–student paradigm"). Currently no such declaration exists. **Low severity — flag for fixer to decide whether to add a one-sentence disambiguation or accept silently.**

### 3.2 "gradient reversal suppression" vs. "gradient reversal layer" vs. "GRL" (MEDIUM)

Three surface forms are used for the same mechanism:

| Location | Form used |
|---|---|
| Abstract L87 | "a **gradient reversal layer** that adversarially suppresses…" |
| Highlights L116 | "**gradient-reversal suppression**" |
| §1 contribution 2 L155 | "**gradient reversal suppression**" (no hyphen) |
| §2.3 footnote L193 | "**The gradient reversal layer** that adversarially suppresses…" |
| §3.2 L221 | "…through **gradient reversal**" |
| §3.4 heading L257 | "**GRL** dual-λ structure" |
| §3.5 heading L280 | "**GRL anomaly suppression loss** $L_\mathrm{cls}$" |
| §5 conclusion L475 | "**gradient-reversal suppression** of anomaly-specific information" |

**Issue**: "gradient reversal suppression" (the mechanism/pathway name) and "gradient reversal layer" (the component name) are distinct and their alternation is logical but the hyphen usage is inconsistent: "gradient-reversal suppression" at L116/L475 vs. "gradient reversal suppression" at L155. The compound modifier "gradient-reversal" before a noun should take a hyphen (Chicago / APA house rule for pre-nominal compounds). **Verdict: apply hyphen consistently to "gradient-reversal suppression" and "gradient-reversal layer" as pre-nominal modifiers; leave "gradient reversal" alone as a noun phrase.**

### 3.3 "contaminated semi-supervised" compound (LOW)

Used in hyphenated, unhyphenated, and italicized forms:

| Location | Form |
|---|---|
| Abstract L87, §3.1 L208 | unhyphenated: "contaminated semi-supervised" |
| Highlights L114 | unhyphenated |
| §1 contribution 1 L153 | italicized + unhyphenated: "*contaminated semi-supervised*" |
| Keywords | unhyphenated |
| §2.3 footnote L215 | italicized: "*contaminated semi-supervised*" |

**Verdict**: No inconsistency in hyphenation (consistently unhyphenated, which is correct — "semi-supervised" already contains a hyphen and a further compound hyphen would be unusual). The italic at first definition L153 and in the footnote L215 is appropriate; plain text elsewhere is also appropriate. This is consistent and acceptable. **No action required.**

### 3.4 "anomaly-priority masking" (LOW)

Used consistently as "anomaly-priority masking" (hyphenated compound adjective) throughout: L87, L116, L155, L232, L237, L423, L445, and Table A.1 L505. **Consistent — no issue.**

### 3.5 NRdetector capitalization (LOW)

The paper consistently uses "NRdetector" (lowercase 'd') at every occurrence (L144, L183, L348, L388, L405, L533). The cite key is `wang2025nrdetector`. The paper's own title is "Noise-Resilient Point-wise Anomaly Detection in Time Series Using Weak Segment Labels" — the detector name used in that paper should be verified against the source. The internal comment at L390 uses "NRdetector" in the method-name list parenthetical. **Flag**: Verify the official capitalization from the Wang et al. (KDD 2025) paper. If the authors use "NR-detector" or "NRDetector", the manuscript should match. This is a low-severity external consistency issue. **Uncertain — fixer should verify against published paper.**

### 3.6 "leave-one-out masking" (LOW)

Used consistently: §3.6 heading "Leave-one-out masking" (capitalized as heading), prose "leave-one-out masking" (L237, L300, L375, Table A.1 L505). **Consistent — no issue.**

### 3.7 "loss bifurcation" descriptors (LOW)

In the abstract (L87) the third mechanism is described as "loss bifurcation between normal and anomalous reconstruction paths." In contribution 2 (L155) it is "loss bifurcation, which restricts the Student decoder's imitation objective to normal-patch outputs." In §5 (L475) it is "loss bifurcation toward normal-only Student mimicry." In §3.5 section heading (L266) it is "Output discrepancy loss $L_{\mathrm{OD}}$" — making it clear that the mechanism name is "loss bifurcation" and the loss component is $L_{\mathrm{OD}}$. The §4.1.4 L409 uses "OD loss" informally. **The variation in the descriptive phrase (the mechanism name is "loss bifurcation"; descriptions like "toward normal-only Student mimicry" or "between normal and anomalous reconstruction paths" are explanatory expansions, not separate terms.) No unification issue — low severity.**

---

## 4. Notation Audit (R5)

### 4.1 $d_{\mathrm{model}}$ vs. $d_{\text{model}}$ inconsistency (MEDIUM)

**Critical notation split found.** The same quantity uses two different LaTeX commands:

| Location | Form | Render |
|---|---|---|
| §3.3 Eq. embedding definition L229 | `d_{\mathrm{model}}` | upright (correct for multi-letter subscript) |
| §3.5 Eq. (2) L276 | `d_{\mathrm{model}}` | upright |
| Table C.2 notation table L750 | `d_{\mathrm{model}}` | upright |
| §4.1.2 prose L360 | `d_\text{model}` | upright (equivalent in many engines, but not \mathrm) |
| Table A.1 encoder row L500 | `d_\text{model}` | same |
| §C.2 prose L720 | `d_\text{model}` | same |

**Issue**: `\mathrm{model}` and `\text{model}` produce visually identical output in most LaTeX engines but are semantically different commands. Field convention for multi-letter subscripts naming a concept (not a variable) strongly prefers `\mathrm{}` (Roman/upright math) over `\text{}` (text mode), for the same reason that loss subscripts use `\mathrm{OD}`, `\mathrm{FM}`, etc. (which is done consistently elsewhere in this manuscript). **Verdict: normalize all instances to `d_{\mathrm{model}}`. Three occurrences at L360, L500, L720 use `\text{model}` and should be changed to `\mathrm{model}`.**

### 4.2 Notation table vs. body: $T$ italic for series length (LOW)

Table C.2 header (L740) notes: "Upright superscripts $\mathrm{T}$/$\mathrm{S}$ tag the Teacher/Student branches (italic $T$ denotes the series length)." The body uses $T$ for series length in §3.1 L205 ($\mathbf{X} \in \mathbb{R}^{T \times F}$) and Table C.2 L744. This is correct and consistent. **No issue — the disambiguation is already documented in the table caption.**

### 4.3 $\varepsilon$ in Eq. (4) — domain convention check (LOW)

Eq. (4) uses $\varepsilon = 10^{-4}$ as a smoothing constant. The symbol $\varepsilon$ for small positive stabilizer is standard in neural-network optimization (Adam uses $\epsilon$) and in numerical analysis. **Domain-appropriate. No issue.**

### 4.4 $P_n$ for normal-patch set — collision risk (LOW, UNCERTAIN)

§3.5 defines $P_n = \{i \in M : y^p_i = 0\}$ as masked patches labeled normal. The notation C.2 table (L757) lists it correctly. However $P$ is already used for patches $\mathbf{P}_i$ (bold), and $P_n$ (non-bold, subscript n) is the set. In the equations these are distinct: bold $\mathbf{P}_i$ is the patch tensor; plain $P_n$ is an index set. The distinction depends on bold vs. non-bold, which is clear in LaTeX output but might be easy to misread. **Flag**: This is not a standard notation conflict but is a potential readability issue. The fixer might consider $\mathcal{N}$ or $M_n$ for the normal-masked-patch index set to eliminate the visual closeness to $\mathbf{P}_i$. However, changing this would require updating Eqs. (1), (2), §3.5 body, and Table C.2 — this is a medium-effort refactor. **Severity: low, uncertain — flag for Phase 7 fixer.**

### 4.5 Table C.2 vs. body: $\bar{r}$, $\bar{d}$ defined only in Eq. (4) prose, not in notation table (LOW)

Eq. (4) introduces $\bar{r}$ and $\bar{d}$ as "the means of the patch-level reconstruction errors and discrepancies over all (window, patch) pairs." These are not in Table C.2 (L762 lists $r_i$, $d_i$, $\tilde{d}_i$ but not $\bar{r}$, $\bar{d}$). **Minor omission — the bar-means are local to one equation and adequately defined in the prose there, but Table C.2 should include them for completeness. Flag for fixer to add two rows.**

---

## 5. Unnecessary Abbreviation Audit (R15)

### Defined abbreviations — full inventory

| Abbreviation | Definition location | Assessment |
|---|---|---|
| MTSAD | §1 L131 ("multivariate time series anomaly detection (MTSAD)") | Justified: used 8+ times throughout |
| TSAD | Highlights L114, body L149, L173, L183, L388 (no formal definition found) | **Issue — see 5.1** |
| CSMAD | §1 L148 ("CSMAD (Contaminated Semi-supervised Masked Anomaly Detector)") | Justified: method name, used throughout |
| GRL | §3.4 L257 (heading uses GRL before prose definition) | Defined at §3.5 L280 heading; first prose occurrence §3.2 L221 uses full form. Acceptable for a standard term (Ganin et al. coined GRL). |
| MAE | §2.3 L187 (references "He et al. MAE") | Standard field abbreviation; acceptable |
| PU | §2.2 L179 ("Positive and Unlabeled (PU) learning") | Justified: used several times |
| PA | §A.2 L544 ("point adjustment (PA)") | Justified: used many times, standard field term |
| BCE | §3.5 L697 ($\mathrm{BCE}_{w_+}$) | Standard; no definition in prose, but used only in math context. Acceptable in equation. **Flag: prose at L282 says "focal-style BCE variant" — BCE should be spelled out at first prose use. Check if "binary cross-entropy (BCE)" appears earlier.** On inspection: L282 is the first prose use and does not expand BCE. **Low severity — add "(BCE)" once.** |
| GELU | Table A.1 L500 | Standard activation function name; no expansion needed |
| AdamW | Table A.1 L507 | Standard optimizer name; no expansion needed |
| TFMAE | §4.1.4 L388 | External model name; no expansion needed (introduced with citation) |
| NPSR | same | Same |
| TimesNet, DCdetector, MEMTO, ModernTCN, CATCH, DAGMM, USAD, TranAD, GDN, OmniAnomaly | throughout | All external model names with citations; no new abbreviation |
| SWaT, WaDi, PSM, SMD, SMAP, MSL | throughout | Standard dataset abbreviations from original papers |
| MLP, MLP-Mixer | §A.1 L531 | Standard ML abbreviations |
| GCN-LSTM | same | Standard |
| VUS | §4.1.3 and §A.2 | Defined at first use with citation; standard |
| AUC | throughout | Standard |
| MSE | §3.6 L303 | Standard; no expansion in prose (used in equation context only) |

### 5.1 TSAD used without definition (MEDIUM)

TSAD appears at: L114 (Highlights), L149 (§1), L173 (§2.1 heading area), L183 (§2.2), L388 (§4.1.4). MTSAD is defined at L131. TSAD is the unqualified (univariate + multivariate) version, but is used interchangeably with MTSAD in several contexts.

**Issue 1 (R15)**: TSAD is never formally defined. It appears first at L114 in Highlights before MTSAD is defined at L131.

**Issue 2 (consistency)**: At L173 the section uses "unsupervised MTSAD" (full scope), but the heading prior is "Multivariate Time Series Anomaly Detection." At L183, L388 the paper uses "TSAD" where context is clearly multivariate (the sentence is about CSMAD and multivariate methods). The paper should either (a) define TSAD once as a superset and use MTSAD where multivariate is intended, or (b) consolidate to MTSAD everywhere and drop TSAD.

**Verdict**: Since all substantive discussion in this paper is multivariate, consolidating to MTSAD (already defined) and removing TSAD would both satisfy R15 (no undefined abbreviation) and tighten scope. If the authors want TSAD as a broader category term (e.g., in a sentence about the general field), it must be defined on first use. **Severity: medium.**

### 5.2 "OD loss" — informal abbreviation in body (LOW)

§4.2 L409 uses "OD loss" as shorthand for $L_{\mathrm{OD}}$ (Output Discrepancy loss). This occurs inside the protocol-effect analysis prose. The abbreviated form "OD loss" is used only once in body prose (though the notation $L_{\mathrm{OD}}$ is used from §3.5 onward). This is a minor informalism but is not a new abbreviation — OD is the subscript of the loss symbol. **Low severity.**

---

## 6. Internal Terminology / Variable Names (R24)

### 6.1 Q1 / Q3 — already covered in Section 2 (HIGH)

See Section 2. This is the most severe R24 violation: 11 body occurrences of code-condition labels.

### 6.2 "excl22" — code variable name in body (MEDIUM)

The evaluation condition excluding SWaT attack region 22 is referred to as "excl22" throughout:
- §4.1.1 L354: "we therefore also report all metrics with region 22 masked out (excl22; anomaly ratio 3.68%…)"
- §4.1.1 L355: "Table 2 ranks under excl22"
- §A.4 section heading: "SWaT excl22: Region Definition and Dual-Condition Results"
- §A.4 L621, L622, L623, L624 (multiple uses)
- Table A.4 caption L578: "Test AR denotes… (full) / 3.68 (excl22)"
- Table A.1 L504 (in table row text)
- §B.1 caption L646: "repeated from Table 2 for reference, as CSMAD trains on the contaminated stream in both conditions" (no excl22 here, but surrounding text uses it)

"excl22" is the internal code variable name used in the codebase (`evaluator.py`, metadata keys `excl22_*`, worker spawn code). It has entered body text as though it were a paper term.

**Verdict**: R24 applies. The code name is `swat_eval_mode="excl22"`. A proper academic name would be "SWaT (excl. region 22)" or "SWaT-excl22" (as a label in tables) or "SWaT without region 22" in prose. The current usage is borderline — "excl22" functions as a compact table label/condition shorthand and the first use at L354 does explain what it means. However the section heading "SWaT excl22" is unapologetically the code variable. **Recommended fix: spell out as "SWaT (excl. R22)" or "SWaT (excl. region 22)" at first use in each section, and use it as a table-column label shorthand "excl22" only in tables (where space is limited and the label has already been defined). The §A.4 heading should change to "SWaT Evaluation with Region 22 Excluded" or similar. Severity: medium.**

### 6.3 "normal-only" as a condition description — ambiguous origin (MEDIUM)

§4.1.4 L393 uses "the Q3 (**normal-only**) condition." The parenthetical "normal-only" directly reflects the internal code variant name `"normalonly"` (unified_loader.py:34). While it is descriptive, it simultaneously serves as the code's internal variant name. If Q3 is renamed (see §2 above), the parenthetical "(normal-only)" should be replaced with the description of what the condition is (e.g., "anomaly-excised condition, in which labeled anomaly regions are removed from training").

This item is partially subsumed by the Q3 fix — resolving Q3 → "anomaly-excised condition" eliminates the need for the "(normal-only)" parenthetical since the condition name is already descriptive. **Severity: medium, resolved by Q3 fix.**

---

## 7. Domain Standard Term Verification

### 7.1 "contaminated semi-supervised" — domain standard check (UNCERTAIN)

The paper introduces "contaminated semi-supervised" as a named setting and provides a footnote (L215) distinguishing it from "contamination-resilient" and "contamination-resistant" settings. This is a newly coined setting name, not an established field term.

**Usage in literature**: The combination "contaminated semi-supervised" does not appear to be a standard term in existing TSAD or broader anomaly detection literature at the time of writing. "Semi-supervised anomaly detection" is standard (Ruff et al., DeepSAD). "Contaminated training" appears in outlier detection literature (e.g., robust PCA, Huber contamination model) but typically refers to unlabeled contamination. The authors' footnote correctly distinguishes their coinage from prior uses.

**Verdict**: This is an appropriate new term for a genuinely novel setting, introduced with careful footnoting. The term is not fashionable AI jargon (R5 concern) but a descriptive compound. The footnote at L215 is the right mechanism to establish it. **No action required, but flag: if a reviewer disputes the naming, the response should cite the footnote's distinctions. The paper should not call this a "new term" explicitly — the current handling (silent definition through the setting formalization + footnote disambiguation) is correct.**

### 7.2 "self-distillation" — domain standard check (ACCEPTABLE)

The paper uses "self-distillation" citing Zhang et al. (2022) and Ristea et al. (SDMAE 2024). The footnote at L193 correctly attributes the terminology. Zhang et al. coined the term for within-architecture distillation; Ristea et al. applied it in the video anomaly context. **Acceptable — properly sourced.**

### 7.3 "anomaly-ratio threshold" — partially standard (ACCEPTABLE)

The paper uses "anomaly-ratio threshold" to name the $(1-\alpha)$ quantile thresholding where $\alpha$ is the evaluation anomaly fraction. This mechanism was introduced in Anomaly Transformer (Xu et al. 2022), and the paper correctly credits it. The specific term "anomaly-ratio threshold" is descriptive rather than a direct quote from Xu et al., but is accurate. EXPERIMENT_PROTOCOL_TRUTH §⑤ confirms the code uses `ar_th` (anomaly ratio threshold). **Acceptable — the term accurately describes the mechanism and is credited.**

### 7.4 "PA%K-AUC F1" and "PA%K-AUC AUC-PR" — compound metric names (ACCEPTABLE)

These are constructed names for metrics that aggregate the PA%K protocol (Kim et al. AAAI 2022) over the tolerance spectrum. The canonical code keys are `pak_auc_f1` and `pak_auc_prc_auc`. The paper's rendered names "PA\%K-AUC F1" and "PA\%K-AUC AUC-PR" accurately reflect: (a) the PA%K protocol prefix, (b) the AUC integration, (c) the integrand (F1 vs. AUC-PR). EXPERIMENT_PROTOCOL_TRUTH §④ confirms these as the approved paper-facing names.

**One naming awkwardness**: "PA%K-AUC AUC-PR" contains "AUC" twice — once for the K-integration ("PA%K-AUC") and once for the integrand ("AUC-PR"). This is technically accurate but verbose. It follows from the structure: the AUC-PR at each K is the integrand, and the K-integration produces a second AUC. A shorter form like "PA%K-AUC PR" (where PR is the precision-recall AUC) might be cleaner, but changing this at Phase 6 would require updating all table headers, the §4.1.3 definition, §A.2, and Table A.7 — and risks confusion with existing shorthand. **Flag only — do not change without author decision. Severity: low.**

---

## 8. Overly Granular Prose (R35)

### 8.1 SWaT preprocessing footnote detail in §A.1 (LOW)

§A.1 L538–540: "Reproductions should verify this dimension explicitly, as loading the raw CSV files without the constant-column filter yields 51 features." This is sound reproducibility disclosure, but the sentence "Reproductions should verify this dimension explicitly" is directed at a reproduction engineer rather than a reader assessing the paper's contributions. **Low severity — this is appropriate in an appendix implementation section; however it reads like a lab notebook warning. A passive restatement ("the input dimension is 45 after removing 6 constant columns; the raw CSV yields 51 features") would be less imperative. Flag only.**

### 8.2 "boundary-aware windowing" detail at §A.3 L611 (LOW)

§A.3 L610–612: "The original training portion and the incorporated test prefix are not temporally adjacent; segment boundaries are registered so that no sliding window crosses non-contiguous data. The same mechanism guards the excision boundaries of the Q3 condition." This is correct protocol disclosure, but for a journal appendix it occupies two sentences on an implementation detail that adds no scientific content beyond what §4.1.1 already states. **Low severity — the disclosure is required for reproducibility; the prose is not excessively granular by journal appendix standards. Flag but no removal recommended.**

---

## 9. Unified Term Table

The following table gives the canonical paper-facing form for each term with observed variation. Fixer should apply these globally.

| Concept | Canonical form | Current variations | Notes |
|---|---|---|---|
| Internal condition Q3 | **anomaly-excised condition** (or "clean-training condition" — author choice) | "Q3", "Q3 (normal-only)", "normalonly" | R24 violation; 11 occurrences |
| Internal condition Q1 | **contaminated-training condition** | "Q1", "full contaminated training" | R24 violation; 5 occurrences in body+appendix |
| Code label "excl22" in headings/prose | **SWaT (excl. R22)** in prose; "excl22" acceptable as table-column shorthand after first definition | "excl22", "SWaT excl22" | R24, medium severity |
| d_model notation | `$d_{\mathrm{model}}$` | `$d_{\mathrm{model}}$` (correct) and `$d_\text{model}$` (3 occurrences) | R5; normalize `\text` → `\mathrm` |
| Gradient-reversal compound modifier | **gradient-reversal** (hyphenated before a noun) | "gradient reversal suppression" (L155, no hyphen) vs. "gradient-reversal suppression" (L116, L475, hyphenated) | Hyphen in pre-nominal position only |
| TSAD abbreviation | Either define at first use or consolidate to **MTSAD** | Undefined in body; used 5× | R15 |
| BCE first prose use | "binary cross-entropy (BCE)" at first prose occurrence | L282 uses "BCE" without prose expansion | R15, low |
| $\bar{r}$, $\bar{d}$ | Add to Table C.2 | Defined in Eq. (4) prose but absent from notation table | R5 completeness |

---

## 10. Notation Consistency: Table C.2 vs. Body — Full Check

The following symbols are used in the body and appear in Table C.2. Checked for consistency:

| Symbol | Table C.2 entry | Body use | Status |
|---|---|---|---|
| $\mathbf{X}$, $T$, $F$ | L744 | §3.1 L205 | Consistent |
| $\mathbf{W}$, $L$ | L745 | §3.1 L205 | Consistent |
| $\mathbf{P}_i$, $s$ | L746 | §3.1 L205 | Consistent |
| $N$ | L747 | §3.1 L205, §3.3 L233 | Consistent |
| $y^w$, $y^p_i$ | L748 | §3.1 L206 | Consistent |
| $\mathbf{E}$, $\mathbf{b}$ | L749 | §3.3 L229 | Consistent |
| $\mathbf{z}_i$, $d_{\mathrm{model}}$ | L750 | §3.3 L229 (uses $d_{\mathrm{model}}$) | Consistent in math; inconsistency in prose (`\text{model}`) covered in §4.1 |
| $\rho$, $M$, $V$ | L751–752 | §3.3 L233 | Consistent |
| $\pi_i$, $\eta_i$ | L753 | §3.3 L234 | Consistent |
| $n_e$, $n_{\mathrm{T}}$, $n_{\mathrm{S}}$ | L754 | §3.4 | Consistent |
| $h^{\mathrm{T}}_i$, $h^{\mathrm{S}}_i$ | L755 | §3.4–3.5 | Consistent |
| $o^{\mathrm{T}}_i$, $o^{\mathrm{S}}_i$ | L756 | §3.4–3.5 | Consistent |
| $P_n$ | L757 | §3.5 L267 | Consistent; visual closeness to $\mathbf{P}_i$ noted in §4.4 |
| $L_{\mathrm{recon}}$, $L_{\mathrm{OD}}$, $L_{\mathrm{FM}}$, $L_{\mathrm{cls}}$ | L758 | §3.5 | Consistent |
| $g_\phi$, $\hat{y}_i$, $\tilde{h}^{\mathrm{S}}_i$ | L759 | §3.5, §C.1 | Consistent |
| $w_+$, $\gamma$ | L760 | §C.1 Eq. C.3 | Consistent |
| $\lambda_{\mathrm{FM}}$, $\lambda_{\mathrm{GRL}}$ | L761 | §3.4–3.5 | Consistent |
| $\lambda_{\mathrm{rev}}$, $\tau$ | L762 | §3.4, §C.1 | Consistent |
| $r_i$, $d_i$, $\tilde{d}_i$ | L763 | §3.6 | Consistent; $\bar{r}$, $\bar{d}$ missing from table (§4.5) |
| $\sigma_i$, $a_t$ | L764 | §3.6 Eqs. 5–6 | Consistent |
| $\alpha$ | L765 | §4.1.2 L376 | Consistent |
| $c$ | Not in Table C.2 | §3.6 Eq. (5), §4.1.2 | **Flag: $c = 4$ (score combination ratio) is used in Eq. (5) and prose but is absent from Table C.2. Should be added.** |
| $\varepsilon$ | Not in Table C.2 | §3.6 Eq. (4) | **Flag: $\varepsilon = 10^{-4}$ is used in Eq. (4) but absent from Table C.2.** |

---

## 11. Training-time vs. Inference-time Wording Check

The manuscript uses three surface forms:
- "training-time mechanism" (L237) — hyphenated compound adjective, correct
- "at test time" (L237) — unhyphenated when used adverbially, correct
- "at inference time" (L143) — unhyphenated adverbial, correct
- "training-only" (FIG-2 caption placeholder, L219; L221) — hyphenated, correct

These are all grammatically correct and contextually appropriate forms. **No inconsistency found.**

The distinction between "test time" (inference on the test split) and "inference time" (general inference) is used without conflict:
- "at inference time" at L143 refers to the general inference scenario
- "at test time" at L237 refers specifically to the test-split scoring protocol
- "inference" in "leave-one-out inference" is consistent

**No action required.**

---

## 12. Point-wise vs. Point-level Terminology

The paper uses two forms:
- "point-level annotation" (L131), "point-level anomaly score" (Table C.2 L764), "point-level scores" (§A.2 L564, L571), "point-level detection" (§2.2 L183) — hyphenated compound adjective
- "pointwise scoring" (§A.2 L546: "K = 100 point-wise scoring") — also hyphenated

Both are hyphenated and used consistently. "Point-level" refers to granularity of labels/scores; "point-wise" refers to the scoring protocol (vs. segment-level adjustment). The distinction is used correctly and consistently. **No issue.**

---

## 13. Dataset Naming Check

| Name used in manuscript | Canonical name in source paper | Status |
|---|---|---|
| SWaT | Secure Water Treatment — Goh et al. 2016. Common abbreviation is SWaT | Consistent |
| WaDi | Water Distribution — Ahmed et al. 2017. Abbreviated as "WaDi" in source | Consistent |
| WaDi A1, WaDi A2 | Attack files are "A1" and "A2" in the original dataset | Consistent |
| PSM | Pooled Server Metrics — Abdulaal et al. 2021. Abbreviated PSM | Consistent |
| SMD | Server Machine Dataset — Su et al. 2019. Abbreviated SMD | Consistent |
| SMAP | Soil Moisture Active Passive — Hundman et al. 2018 | Consistent |
| MSL | Mars Science Laboratory — Hundman et al. 2018 | Consistent |
| DAGMM (simplified) | Paper uses "DAGMM (simplified)" noting the TranAD re-implementation | Consistent; flag: ensure table header uses exactly "DAGMM (simplified)" not just "DAGMM" |

**No dataset naming inconsistencies found.**

---

## 14. Metric Naming Check vs. EXPERIMENT_PROTOCOL_TRUTH §④

| Code key | Protocol Truth canonical name | Manuscript name | Status |
|---|---|---|---|
| `pak_auc_f1` | PA%K-AUC F1 | **PA\%K-AUC F1** (bold, first use §4.1.3 L380) | Consistent |
| `pak_auc_prc_auc` | PA%K-AUC AUC-PR | **PA\%K-AUC AUC-PR** | Consistent |
| `vus_pr` | VUS-PR | **VUS-PR** | Consistent |
| `vus_roc` | VUS-ROC | **VUS-ROC** | Consistent |
| `affiliation_f1` | Affiliation F1 | **Affiliation F1** | Consistent |
| `pa_0_f1` | Point-Adjusted F1 (PA F1) | **PA F1** (§4.1.3 L384) | Consistent |

The Protocol Truth §④ also flags that the Affiliation F1 reported in body uses the anomaly-ratio threshold variant (`affiliation_f1_ar`), but the paper refers to it simply as "Affiliation F1…computed at the anomaly-ratio threshold" (L380). This is correct — the name "Affiliation F1" is the metric; the threshold choice is the modifier. **Consistent.**

The body also refers to "PA\%K-AUC AUC-PR" in §4.1.3 as "PA\%K-AUC AUC-PR" and later abbreviates it as "PA\%K-AUC AUC-PR" (no shorter form introduced). Table A.7 placeholder refers to "PA%K-AUC AUC-PR". **Consistent.**

---

## 15. Baseline Naming Check

| Name in manuscript | Canonical / citation name | Status |
|---|---|---|
| Anomaly Transformer | Xu et al. 2022 "Anomaly Transformer" | Consistent |
| TranAD | Tuli et al. 2022 "TranAD" | Consistent |
| USAD | Audibert et al. 2020 "USAD" | Consistent |
| DAGMM | Zong et al. 2018 "DAGMM" | Consistent |
| GDN | Deng & Hooi 2021 "GDN" | Consistent |
| OmniAnomaly | Su et al. 2019 "OmniAnomaly" | Consistent |
| TFMAE | Fang et al. 2024 | Consistent |
| NPSR | Lai et al. 2023 | Consistent |
| TimesNet | Wu et al. 2023 "TimesNet" | Consistent |
| DCdetector | Yang et al. 2023 "DCdetector" | Consistent |
| MEMTO | Song et al. 2023 "MEMTO" | Consistent |
| ModernTCN | Luo et al. 2024 "ModernTCN" | Consistent |
| CATCH | Wu et al. 2025 "CATCH" | Consistent |
| NRdetector | Wang et al. 2025 | Consistent internally; verify external capitalization (§3.5) |
| WETAS | Lee et al. 2021 (cited as `lee2021wetas`) | Used once at §2.2 L183 as "the WETAS architecture" — consistent |
| DeepMIL | Sultani et al. 2018 (cited as `sultani2018deepmil`) | Not spelled out in body prose; only in comment L390 | Acceptable — method name lives in Table 2 rows per comment |
| TreeMIL | Liu et al. 2024 (cited as `liu2024treemil`) | Same as DeepMIL | Acceptable |
| DACAD | Darban et al. 2024 (cited as `darban2024dacad`) | §2.2 L181 "DACAD" | Consistent |
| SDMAE | Ristea et al. 2024 (cited as `ristea2024sdmae`) | §2.3 throughout | Consistent |

---

## 16. Summary of Actionable Findings for Fixer

Ordered by priority:

| # | Severity | Location | Issue | Action |
|---|---|---|---|---|
| 1 | HIGH | §4.1.4 L389, L393–395; §4.2 L403, L405; §A.3 L612; §B.1 heading, L641–642, caption | Q1/Q3 internal code labels in body prose | Replace Q3 → academic term (e.g., "anomaly-excised condition"); Q1 → "contaminated-training condition"; globally throughout |
| 2 | MEDIUM | §A.4 heading; §4.1.1 L354–355; §A.4 L617–624; Table A.4 caption | "excl22" code variable in body prose and section heading | Prose: "SWaT (excl. R22)" at first use per section; heading: "SWaT Evaluation with Region 22 Excluded"; tables: "excl22" as compact label is acceptable after prose definition |
| 3 | MEDIUM | L149, L173, L183, L388 (Highlights L114) | TSAD undefined in body; used before MTSAD defined | Either define "TSAD" at first use (L114) or replace all TSAD occurrences with MTSAD; preferred: consolidate to MTSAD |
| 4 | MEDIUM | §3.3 L360; Table A.1 L500; §C.2 L720 | `$d_\text{model}$` should be `$d_{\mathrm{model}}$` | Change `\text{model}` → `\mathrm{model}` at all 3 occurrences |
| 5 | MEDIUM | Highlights L116; §1 contribution 2 L155 | Hyphen inconsistency in "gradient reversal(-)suppression" | Normalize to "gradient-reversal suppression" (hyphenated) at L155; already hyphenated at L116 and L475 |
| 6 | MEDIUM | §3.5 L282 | "BCE" used in prose without first-use expansion | Add "(BCE)" → "binary cross-entropy (BCE)" at L282 first prose occurrence |
| 7 | LOW | Table C.2 | $\bar{r}$, $\bar{d}$, $c$, $\varepsilon$ absent from notation summary table | Add 4 rows to Table C.2 |
| 8 | LOW | §2.3 L189, footnote L193 | Lowercase teacher/student for prior work vs. uppercase for CSMAD — defensible but undeclared | Add one-sentence clarification in §2.3 footnote or accept silently |
| 9 | LOW | §§A.1, A.3 | Minor prose granularity (lab-notebook-style imperative wording) | Rephrase to passive scientific register |

---

*End of audit. No manuscript modifications made. All findings are inspection results for the fixer agent.*
