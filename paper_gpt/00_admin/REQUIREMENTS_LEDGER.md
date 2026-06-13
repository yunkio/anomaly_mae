# Requirements Ledger

Created: 2026-06-13 KST

This ledger converts the master prompt into stable requirement IDs. Each phase must update status and evidence. Status values: `Pending`, `In progress`, `Satisfied`, `Blocked`, `Superseded`.

## Governance and Process

| R-ID | Requirement | Phase(s) | Evidence artifact | Status | Notes |
|---|---|---|---|---|---|
| GOV-001 | All work must be created under `paper_gpt/`. | 0-9 | `MASTER_INDEX.md`; file tree | In progress | Directory tree created. |
| GOV-002 | Maintain `MASTER_INDEX.md` and update it for major artifacts. | 0-9 | `MASTER_INDEX.md` | In progress | Initial index created. |
| GOV-003 | Create and maintain `REQUIREMENTS_LEDGER.md` with stable IDs. | 0-9 | `REQUIREMENTS_LEDGER.md` | In progress | Initial ledger created. |
| GOV-004 | Create `PHASE_LOG.md`, `DECISION_LOG.md`, `OPEN_QUESTIONS.md`, `AGENT_REGISTRY.md`, and `REVIEW_LEDGER.md`. | 0 | Admin files | In progress | Initial files created in Phase 0. |
| GOV-005 | Do not begin substantive research/writing before Phase 0 governance exists. | 0 | Phase log | Satisfied | Phase 0 performed first. |
| GOV-006 | At every phase end, update phase log, master index, requirement ledger, and report outputs/decisions/open questions/next phase. | 0-9 | `PHASE_LOG.md`; `MASTER_INDEX.md`; ledger | Pending | Applies at each phase gate. |
| GOV-007 | Use producer-reviewer-contrarian-revision-acceptance loop for every major artifact. | 0-9 | `REVIEW_LEDGER.md` | Pending | Initial review loop defined. |
| GOV-008 | Do not mark final completion until requirement-by-requirement evidence proves completion. | 9 | `REQUIREMENT_COVERAGE_MATRIX.md` | Pending | Final audit requirement. |
| GOV-009 | Use `dc_vis` conda environment for all Python execution in this repository. | 0-9 | Commands in logs | Pending | Project `AGENTS.md` enforces this. |
| GOV-010 | Optimize for quality over time/token efficiency. | 0-9 | Phase outputs and review logs | Pending | Applies globally. |

## Source Boundaries and Inputs

| R-ID | Requirement | Phase(s) | Evidence artifact | Status | Notes |
|---|---|---|---|---|---|
| SRC-001 | Use current project scripts and markdown documents, respecting forbidden paths. | 1 | `PROJECT_SOURCE_INDEX.md`; `CODEBASE_UNDERSTANDING.md` | Pending | Phase 1. |
| SRC-002 | Fetch and use method overview Notion page. | 1 | `NOTION_EXTRACTION.md` | Pending | Requires Notion MCP. |
| SRC-003 | Fetch and use baseline comparison Notion page. | 1 | `NOTION_EXTRACTION.md` | Pending | Requires Notion MCP. |
| SRC-004 | Read allowed PDF `paper/윤기오_대한산업공학회_2026_춘계.pdf`. | 1 | `PDF_EXTRACTION.md` | Pending | Allowed exception. |
| SRC-005 | Read allowed template text `paper/elsevier template.txt`. | 8 | LaTeX notes/build log | Pending | Allowed exception. |
| SRC-006 | Locate Exp 271 result metadata and trace exact runtime behavior. | 1 | `EXP271_CONFIG_TRACE.md`; `METHOD_FACTS_LEDGER.md` | Pending | Critical source of truth. |
| SRC-007 | Do not use `paper_legacy/**`. | 0-9 | Source index; final audit | Pending | Forbidden. |
| SRC-008 | Do not use `paper/**` except the allowed PDF and template text. | 0-9 | Source index; final audit | Pending | Forbidden. |
| SRC-009 | Verify KBS target-journal identity and guidance from current official sources before using journal-fit claims. | 2-9 | `TARGET_JOURNAL_KBS_PROFILE.md` | Pending | Requires web verification. |

## Agent Team and Review

| R-ID | Requirement | Phase(s) | Evidence artifact | Status | Notes |
|---|---|---|---|---|---|
| AGT-001 | Register required core management agents. | 0 | `AGENT_REGISTRY.md` | In progress | Initial registry created. |
| AGT-002 | Register research understanding agents. | 0-1 | `AGENT_REGISTRY.md` | In progress | Includes code, Notion, PDF, Exp 271 tracing. |
| AGT-003 | Register literature/reference agents. | 0,2,4 | `AGENT_REGISTRY.md` | In progress | Includes KBS fit and reference verification. |
| AGT-004 | Register manuscript and writing agents. | 0,3,5 | `AGENT_REGISTRY.md` | In progress | Includes method interaction and contribution strategy. |
| AGT-005 | Register review/finalization agents. | 0,6-9 | `AGENT_REGISTRY.md` | In progress | Includes final publication reviewer. |
| REV-001 | Every major artifact must receive specialist and contrarian review before acceptance. | 0-9 | `REVIEW_LEDGER.md` | Pending | Applies after artifacts exist. |
| REV-002 | Reviewer logs must include artifact, reviewer, issues, required fixes, acceptance status, and evidence of fixes. | 0-9 | `REVIEW_LEDGER.md` | Pending | Template created. |

## Project Understanding

| R-ID | Requirement | Phase(s) | Evidence artifact | Status | Notes |
|---|---|---|---|---|---|
| INT-001 | Read all relevant current scripts and markdown documents to understand the research. | 1 | `CODEBASE_UNDERSTANDING.md` | Pending | Excluding forbidden sources. |
| INT-002 | Extract and synthesize required Notion pages. | 1 | `NOTION_EXTRACTION.md` | Pending | Must cite page URLs. |
| INT-003 | Extract useful method/background/figure/table logic from allowed PDF without copying expressions. | 1 | `PDF_EXTRACTION.md` | Pending | Avoid plagiarism. |
| INT-004 | Build `METHOD_FACTS_LEDGER.md` where every method claim has a source. | 1 | `METHOD_FACTS_LEDGER.md` | Pending | Required before method writing. |
| INT-005 | Distinguish active Exp 271 components from inactive optional code. | 1 | `EXP271_CONFIG_TRACE.md` | Pending | Critical. |
| INT-006 | Exclude dynamic margin if unused in Exp 271. | 1,5,9 | Exp trace; manuscript; final audit | Pending | Must verify active status first. |
| INT-007 | Exclude Gaussian smoothing. | 1,5,9 | Exp trace; manuscript; final audit | Pending | User explicitly removed it. |
| INT-008 | Exclude Simulation and Exathlon datasets. | 1,5,9 | Source index; manuscript; final audit | Pending | User explicitly removed them. |

## Literature, References, and Integrity

| R-ID | Requirement | Phase(s) | Evidence artifact | Status | Notes |
|---|---|---|---|---|---|
| LIT-001 | List top-tier AI/ML/deep-learning venues from 2023-2026 and study strong papers. | 2 | `TOP_TIER_AI_CONFERENCE_MAP_2023_2026.md` | Pending | Current year 2026. |
| LIT-002 | Include time-series anomaly detection papers. | 2,4 | `TIME_SERIES_AD_STRUCTURE_STUDY.md`; reference database | Pending | Required domain coverage. |
| LIT-003 | Include SSL/PU-learning papers and emphasize scarcity in time-series anomaly detection. | 2,4,5 | Literature study; manuscript | Pending | NRDetector important. |
| LIT-004 | Study NRDetector experiment structure and logic; emphasize differences more than similarities. | 2,4,5 | Literature study; manuscript | Pending | User-specified. |
| LIT-005 | Verify why self-distilled video MAE uses the term self-distilled. | 4,5 | Reference notes; manuscript | Pending | Defensive terminology. |
| REF-001 | Prefer top-tier or highly cited papers for references where possible. | 4 | `REFERENCE_SEARCH_PLAN.md`; database | Pending | Must be evidence-based. |
| REF-002 | Verify every reference from reliable primary/official sources; no guessing. | 4 | `REFERENCE_VERIFICATION_AUDIT.md` | Pending | Veto if uncertain. |
| REF-003 | Cross-check important metadata from at least two reliable sources when possible. | 4 | `REFERENCE_VERIFICATION_AUDIT.md` | Pending | Required for confidence. |
| REF-004 | Store short exact excerpts and usage notes; do not copy into manuscript. | 4,6 | `QUOTE_AND_USAGE_NOTES.md`; plagiarism review | Pending | Anti-plagiarism. |
| REF-005 | Produce IEEE-style reference list and BibTeX. | 4,8 | `IEEE_REFERENCE_LIST.md`; `references.bib` | Pending | Required deliverables. |
| REF-006 | Add citations for support-worthy claims; find references when citation is missing. | 4-6 | `CITATION_COVERAGE_REVIEW.md` | Pending | Required. |

## Paper Positioning and Contribution

| R-ID | Requirement | Phase(s) | Evidence artifact | Status | Notes |
|---|---|---|---|---|---|
| POS-001 | Focus on semi-supervised or positive-unlabeled multivariate time-series anomaly detection. | 3,5 | Positioning; manuscript | Pending | Core framing. |
| POS-002 | Emphasize rare known anomalies in mostly unlabeled training data. | 3,5 | Positioning; manuscript | Pending | Core problem. |
| POS-003 | Explain why unsupervised models underuse rare labeled anomalies. | 3,5 | Positioning; manuscript | Pending | Core gap. |
| POS-004 | Contribution emphasis is central; novelty must be strong but not exaggerated. | 3,5,9 | Contribution audit; manuscript; final audit | Pending | New feedback reinforced. |
| POS-005 | Align contribution framing with verified KBS scope and article style. | 2,3,5,9 | KBS profile; KBS review | Pending | New feedback. |
| POS-006 | Emphasize practical monitoring value and new attempt in sparse-known-anomaly setting. | 3,5 | Contribution plan; manuscript | Pending | New feedback. |
| POS-007 | Avoid unnecessary acronyms, except title/model name may use a strong acronym. | 3,5 | Section outline; manuscript | Pending | User-specified. |
| POS-008 | Do not overcenter self-distilled video MAE or list differences in a way that implies high similarity. | 3-5 | Positioning; manuscript | Pending | User-specified. |
| POS-009 | Patch/masking influence comes from vision MAE; do not imply inheritance from time-series patching works. | 4-6 | Reference notes; manuscript; review | Pending | User-specified. |
| POS-010 | Treat anomaly-priority masking as a weak contribution and only an auxiliary implementation detail if active. | 1,3,5,9 | Exp trace; contribution plan; manuscript; final audit | Pending | New feedback. |
| POS-011 | Explain components as interacting time-series-aware design, not stitched-together modules. | 3,5,9 | `METHOD_INTERACTION_MAP.md`; manuscript; final audit | Pending | New feedback. |

## Manuscript Content and Style

| R-ID | Requirement | Phase(s) | Evidence artifact | Status | Notes |
|---|---|---|---|---|---|
| MAN-001 | Produce complete English manuscript. | 5 | `DRAFT_MAIN_TEXT.md` | Pending | With placeholders. |
| MAN-002 | Main body target is 9 pages including figures/tables, excluding appendix/references. | 3,8,9 | Outline; PDF inspection; final audit | Pending | Requires LaTeX check. |
| MAN-003 | Use complete placeholders for figures/tables; no vague placeholders. | 5,7,8 | Placeholder plan; Notion specs; LaTeX | Pending | Required. |
| MAN-004 | Do not fabricate numeric results. | 5-9 | Manuscript; reviews | Pending | Placeholder-safe. |
| MAN-005 | Do not frame missing current figures/data as a limitation. | 5-9 | Manuscript; final audit | Pending | User-specified. |
| MAN-006 | Related work, contributions, and experiments must be MECE. | 3,5,6 | Section outline; manuscript; reviews | Pending | Required. |
| MAN-007 | Do not over-discuss every baseline in related work; cite performance-only baselines in experiments when enough. | 5 | Related work; experiments | Pending | User-specified. |
| MAN-008 | Use publication-level terminology, not internal variable names or rough lab terms. | 5,6 | Domain terminology review | Pending | Required. |
| MAN-009 | Sentence-level review for natural AI/deep-learning/time-series AD academic English. | 6 | `ACADEMIC_STYLE_REVIEW.md` | Pending | Required. |
| MAN-010 | Check plagiarism risk and close paraphrase against quote notes. | 6 | `PLAGIARISM_RISK_REVIEW.md` | Pending | Required. |
| MAN-011 | Use correct, conventional, understandable notation. | 5,6 | Method draft; terminology review | Pending | Required. |
| MAN-012 | Mention Git code release only if natural. | 5 | Manuscript | Pending | User-specified. |

## Experimental Protocol and Metrics

| R-ID | Requirement | Phase(s) | Evidence artifact | Status | Notes |
|---|---|---|---|---|---|
| EXP-001 | Main protocol: split original test data temporally, front 50% folded into train, later 50% test. | 3,5 | Section outline; experiments | Pending | User-specified. |
| EXP-002 | Apply split uniformly to all datasets without cherry-picking. | 3,5 | Experiments | Pending | Fairness. |
| EXP-003 | For unsupervised baselines, use known labels by removing known anomalies from training data. | 3,5 | Experiments | Pending | User-specified. |
| EXP-004 | Defend fairness given scarcity of SSL/PU time-series baselines. | 3,5 | Positioning; experiments | Pending | Reviewer risk. |
| EXP-005 | Include label-sparsity sweep experiments. | 3,5,7 | Experiment plan; placeholders | Pending | User-specified. |
| EXP-006 | Explain robustness when unlabeled anomalies are mixed into training data. | 3,5 | Method/experiments | Pending | User-specified. |
| EXP-007 | Explain SWaT anomaly region 22 exclusion and present separate metrics. | 3,5 | Experiment section | Pending | User-specified. |
| MET-001 | Use VUS-ROC, VUS-PR, PAK-AUC-F1, PAK-AUC-PR, affiliated-F1. | 3,5 | Experiment section | Pending | Required metrics. |
| MET-002 | Explain complementary perspectives of metrics. | 3,5 | Experiment section | Pending | Required. |
| MET-003 | Present PA-F1 only with caveats and not as primary reference point. | 5 | Experiment section | Pending | User-specified. |
| MET-004 | Use test anomaly ratio threshold protocol and defend it as evaluation protocol, complemented by threshold-independent metrics. | 3,5 | Experiment section | Pending | Reviewer risk. |

## Figure, Table, Notion, and LaTeX

| R-ID | Requirement | Phase(s) | Evidence artifact | Status | Notes |
|---|---|---|---|---|---|
| FIG-001 | Define figure/table placeholders with ID, title, caption, placement, size, intended content, source data, and narrative role. | 5 | `FIGURE_TABLE_PLACEHOLDERS.md` | Pending | Required. |
| FIG-002 | Create Korean Notion child page for figure/table specs. | 7 | `NOTION_CREATION_LOG.md` | Pending | Requires Notion MCP. |
| FIG-003 | Notion placeholder page must be highly readable using Markdown/Notion features beyond a numbered list. | 7,9 | `NOTION_PAGE_DESIGN.md`; Notion page | Pending | New feedback. |
| FIG-004 | Readability must not reduce specificity, completeness, caption quality, or content detail. | 7,9 | Specs; final audit | Pending | New feedback. |
| LAT-001 | Use Elsevier template information strictly. | 8 | LaTeX source; build log | Pending | Required. |
| LAT-002 | Compile LaTeX to PDF and inspect layout, pages, floats, tables, figures, appendix, references. | 8 | `build_log.md`; `pdf_inspection.md` | Pending | Required. |
| LAT-003 | Placeholder figures/tables must be appropriately sized and placed. | 8 | PDF inspection | Pending | Required. |

## Final Audit

| R-ID | Requirement | Phase(s) | Evidence artifact | Status | Notes |
|---|---|---|---|---|---|
| FIN-001 | Produce final publication-level audit with PASS / PASS WITH PLACEHOLDERS / NEEDS REVISION. | 9 | `FINAL_PUBLICATION_LEVEL_AUDIT.md` | Pending | Required. |
| FIN-002 | Produce requirement coverage matrix proving no instruction was omitted. | 9 | `REQUIREMENT_COVERAGE_MATRIX.md` | Pending | Required. |
| FIN-003 | Re-check source-boundary compliance. | 9 | Final audit | Pending | Required. |
| FIN-004 | Re-check Exp 271-only active component compliance. | 9 | Final audit | Pending | Required. |
| FIN-005 | Re-check no Gaussian smoothing, no Simulation/Exathlon, no dynamic margin if unused. | 9 | Final audit | Pending | Required. |
| FIN-006 | Re-check reference integrity and citation coverage. | 9 | Final audit | Pending | Required. |
| FIN-007 | Re-check contribution strength, KBS fit, anomaly-priority masking de-emphasis, time-series interaction framing, and Notion readability. | 9 | Final audit | Pending | New feedback integrated. |
