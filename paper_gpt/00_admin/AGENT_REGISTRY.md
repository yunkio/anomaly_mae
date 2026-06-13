# Agent Registry

Created: 2026-06-13 KST

Global forbidden sources for every agent unless a row states an allowed exception:

- Do not use `paper_legacy/**`.
- Do not use `paper/**`, except `paper/윤기오_대한산업공학회_2026_춘계.pdf` for the PDF Extractor and `paper/elsevier template.txt` for the LaTeX Production Agent.

## Core Management Agents

| Agent | Role | Inputs | Forbidden sources | Deliverables | Evidence requirements | Reviewer | Completion criteria |
|---|---|---|---|---|---|---|---|
| Project Steward | Maintain structure, indexes, phase logs, decisions, open questions, and source boundaries | Master prompt; user feedback; all phase outputs | `paper_legacy/**`; `paper/**` except allowed PDF/template references in boundary notes | `MASTER_INDEX.md`, `PHASE_LOG.md`, `DECISION_LOG.md`, `OPEN_QUESTIONS.md` | File paths, phase updates, boundary notes | Orchestrator QA Auditor | Governance files stay current at every phase gate |
| Orchestrator QA Auditor | Check skipped instructions and phase coverage | Requirements ledger; phase outputs | `paper_legacy/**`; `paper/**` except allowed exceptions already extracted into phase artifacts | QA entries in `REVIEW_LEDGER.md`; phase-gate findings | Requirement IDs with evidence artifacts | Contrarian Reviewer | No unchecked major requirement at phase gate |
| Contrarian Reviewer | Critique novelty, contribution framing, fairness, source boundaries, and reviewer objections | Major artifacts | `paper_legacy/**`; `paper/**` except allowed exceptions already extracted into phase artifacts | Rejection-risk memos in `REVIEW_LEDGER.md` and relevant phase files | Concrete objections and required fixes | Orchestrator | Major risks either fixed or logged |

## Research Understanding Agents

| Agent | Role | Inputs | Forbidden sources | Deliverables | Evidence requirements | Reviewer | Completion criteria |
|---|---|---|---|---|---|---|---|
| Codebase and Script Auditor | Read relevant current scripts and markdown files | Repository excluding forbidden sources | `paper_legacy/**`; `paper/**` except no direct access unless explicitly delegated to PDF/template agents | `PROJECT_SOURCE_INDEX.md`, `CODEBASE_UNDERSTANDING.md` | Paths, summaries, source-boundary notes | Contrarian Reviewer | Research implementation is accurately summarized |
| Notion Extractor | Fetch required Notion pages | Method overview and baseline comparison URLs | `paper_legacy/**`; `paper/**` | `NOTION_EXTRACTION.md` | Notion URLs, extracted facts, citations | Orchestrator QA Auditor | Required Notion facts captured |
| PDF Extractor | Read allowed Korean PDF | `paper/윤기오_대한산업공학회_2026_춘계.pdf` | `paper_legacy/**`; all `paper/**` except the named PDF | `PDF_EXTRACTION.md` | Page/section notes; no copied prose | Plagiarism and Citation Integrity Reviewer | Useful content extracted safely |
| Exp 271 Configuration Tracer | Trace exact Exp 271 components | Result metadata; code paths; config files outside forbidden paths | `paper_legacy/**`; `paper/**` | `EXP271_CONFIG_TRACE.md`, `METHOD_FACTS_LEDGER.md` | Metadata paths and code references | Method Writer; Contrarian Reviewer | Active/inactive components are clear |

## Literature and Reference Agents

| Agent | Role | Inputs | Forbidden sources | Deliverables | Evidence requirements | Reviewer | Completion criteria |
|---|---|---|---|---|---|---|---|
| Top-Tier Paper Structure Analyst | Study recent strong AI papers and structure | Current web/official sources | `paper_legacy/**`; `paper/**` | `TOP_TIER_AI_CONFERENCE_MAP_2023_2026.md`, `HIGH_QUALITY_PAPER_STRUCTURE_STUDY.md`, `FIGURE_TABLE_PATTERN_STUDY.md` | Source links and access dates | Time-Series AD Literature Agent | Practical structure principles produced |
| Time-Series Anomaly Detection Literature Agent | Study time-series AD papers, metrics, benchmark norms | Current literature | `paper_legacy/**`; `paper/**` | `TIME_SERIES_AD_STRUCTURE_STUDY.md`; domain portions of `VERIFIED_REFERENCE_DATABASE.md` | Verified references | Reference Verification Agent | Domain claims support blueprint |
| Semi-Supervised and PU-Learning Agent | Study SSL/PU framing and scarcity in time-series AD | Literature and NRDetector | `paper_legacy/**`; `paper/**` | SSL/PU sections in `TIME_SERIES_AD_STRUCTURE_STUDY.md`; candidate refs for `VERIFIED_REFERENCE_DATABASE.md` | Verified references | Contrarian Reviewer | Difference-focused NRDetector framing |
| Reference Verification Agent | Verify all references and metadata | Reference candidates; official sources | `paper_legacy/**`; `paper/**` except extracted allowed PDF notes if cited | `REFERENCE_SEARCH_PLAN.md`, `VERIFIED_REFERENCE_DATABASE.md`, `QUOTE_AND_USAGE_NOTES.md`, `IEEE_REFERENCE_LIST.md`, `REFERENCE_VERIFICATION_AUDIT.md` | Official/primary sources and cross-checks | Orchestrator QA Auditor | No unverified citation enters manuscript |
| Citation Coverage Agent | Find citation-needed claims and missing support | Drafts and claim map | `paper_legacy/**`; `paper/**` except phase artifacts | `CITATION_COVERAGE_REVIEW.md`; citation gaps in `OPEN_QUESTIONS.md` | Claim-to-reference mapping | Reference Verification Agent | Support-worthy claims are cited or logged |
| Target Journal Fit Agent | Verify KBS profile and article style | Current official KBS sources; recent KBS papers | `paper_legacy/**`; `paper/**` | `TARGET_JOURNAL_KBS_PROFILE.md`, `KBS_FIT_AND_CONTRIBUTION_REVIEW.md` | Official links, access dates, article examples | Contrarian Reviewer | KBS-fit guidance informs positioning |

## Manuscript Design and Writing Agents

| Agent | Role | Inputs | Forbidden sources | Deliverables | Evidence requirements | Reviewer | Completion criteria |
|---|---|---|---|---|---|---|---|
| Positioning and Novelty Strategist | Define novelty and contribution | Intake, literature, KBS profile | `paper_legacy/**`; `paper/**` except extracted phase artifacts | `PAPER_POSITIONING.md`, `NOVELTY_AND_CONTRIBUTION_AUDIT.md`, `CONTRIBUTION_REFRAMING_PLAN.md`, `METHOD_INTERACTION_MAP.md`, `CLAIM_EVIDENCE_MAP.md` | Claim-evidence links; contribution risks | Contrarian Reviewer | Contribution is visible and defensible without anomaly-priority masking |
| Outline Architect | Design section plan and page allocation | Positioning; KBS guidance | `paper_legacy/**`; `paper/**` except extracted phase artifacts | `SECTION_OUTLINE.md`; page allocation notes in `PAPER_POSITIONING.md` | Section goals and evidence needs | Orchestrator QA Auditor | Writer can draft without inventing structure |
| Method Writer | Write method using only Exp 271 active components | Method facts; interaction map | `paper_legacy/**`; `paper/**` except extracted phase artifacts | Method portion of `DRAFT_MAIN_TEXT.md`; method notes in `REVISION_HISTORY.md` | Source-backed components | Domain Terminology Reviewer | Accurate, concise, time-series-aware method |
| Experiment Writer | Design experiment narrative/placeholders | Protocol requirements; datasets; metrics | `paper_legacy/**`; `paper/**` except extracted phase artifacts | Experiment portion of `DRAFT_MAIN_TEXT.md`; experiment entries in `FIGURE_TABLE_PLACEHOLDERS.md` | Protocol and metric evidence | Contrarian Reviewer | Fair and persuasive experiment design |
| Related Work Writer | Write MECE related work | Verified references | `paper_legacy/**`; `paper/**` except extracted phase artifacts | Related-work portion of `DRAFT_MAIN_TEXT.md` | Citation coverage | Citation Coverage Agent | No over-discussion of performance-only baselines |
| Full Manuscript Writer | Integrate complete English manuscript | Blueprint, refs, method, experiments | `paper_legacy/**`; `paper/**` except extracted phase artifacts | `DRAFT_MAIN_TEXT.md`, `DRAFT_APPENDIX.md`, `FIGURE_TABLE_PLACEHOLDERS.md`, `REVISION_HISTORY.md` | Citations and placeholder IDs | Academic Style Reviewer | Complete paper-quality draft |

## Review and Finalization Agents

| Agent | Role | Inputs | Forbidden sources | Deliverables | Evidence requirements | Reviewer | Completion criteria |
|---|---|---|---|---|---|---|---|
| Academic Style Reviewer | Sentence-level academic English review | Manuscript draft | `paper_legacy/**`; `paper/**` except extracted phase artifacts | `ACADEMIC_STYLE_REVIEW.md` | Sentence-level findings | Orchestrator | Natural academic style |
| Domain Terminology Reviewer | Check AI/DL/time-series AD terminology | Manuscript draft; method facts | `paper_legacy/**`; `paper/**` except extracted phase artifacts | `DOMAIN_TERMINOLOGY_REVIEW.md` | Term replacements and rationale | Method Writer | No unsuitable internal terminology |
| Plagiarism and Citation Integrity Reviewer | Check close paraphrase and citation safety | Draft; quote notes | `paper_legacy/**`; `paper/**` except verified/extracted quote artifacts | `PLAGIARISM_RISK_REVIEW.md` | Source comparisons | Reference Verification Agent | No unsafe phrasing remains |
| Figure and Table Specification Agent | Create detailed readable specs and Notion page | Placeholder plan | `paper_legacy/**`; `paper/**` except extracted phase artifacts | `NOTION_PAGE_DESIGN.md`, `FIGURE_TABLE_SPEC_INDEX.md`, `FIGURE_SPECS_KO.md`, `TABLE_SPECS_KO.md`, `NOTION_CREATION_LOG.md` | Notion URL and structured spec | Experiment Writer | Future worker can produce every asset |
| LaTeX Production Agent | Produce Elsevier-style LaTeX and inspect PDF | Manuscript, template, refs | `paper_legacy/**`; all `paper/**` except `paper/elsevier template.txt` | `main.tex`, `references.bib`, section/table/figure files, `build_log.md`, `pdf_inspection.md` | Build output and inspection notes | Final Publication Reviewer | PDF compiles coherently |
| Final Publication Reviewer | Final publication-level audit | All artifacts | `paper_legacy/**`; `paper/**` except extracted phase artifacts and allowed template/PDF notes | `FINAL_PUBLICATION_LEVEL_AUDIT.md`, `REQUIREMENT_COVERAGE_MATRIX.md`, `FINAL_DELIVERABLE_SUMMARY.md` | Requirement-level evidence | Contrarian Reviewer | PASS/PASS WITH PLACEHOLDERS/NEEDS REVISION assigned correctly |
