# Paper GPT Master Index

Created: 2026-06-13 KST

This index tracks every major artifact produced for the end-to-end publication-level paper workflow. All work is scoped to `paper_gpt/` and follows `paper_gpt/orchestrator_master_prompt.md`.

## Governance

| Artifact | Purpose | Status |
|---|---|---|
| `paper_gpt/orchestrator_master_prompt.md` | Authoritative master workflow prompt | Created |
| `paper_gpt/00_admin/MASTER_INDEX.md` | Index of all major artifacts | Active |
| `paper_gpt/00_admin/REQUIREMENTS_LEDGER.md` | Stable requirement IDs and phase evidence | Active |
| `paper_gpt/00_admin/PHASE_LOG.md` | Phase-by-phase progress log | Active |
| `paper_gpt/00_admin/DECISION_LOG.md` | Decisions and rationale | Active |
| `paper_gpt/00_admin/OPEN_QUESTIONS.md` | Blockers, unknowns, and unresolved items | Active |
| `paper_gpt/00_admin/AGENT_REGISTRY.md` | Agent roles, deliverables, reviewers | Active |
| `paper_gpt/00_admin/REVIEW_LEDGER.md` | Producer-reviewer loop records | Active |

## Phase Artifact Map

| Phase | Directory | Expected major outputs | Status |
|---|---|---|---|
| Phase 0 | `00_admin/` | Governance, requirements, agent registry, review ledger | In progress |
| Phase 1 | `01_project_intake/` | Source index, codebase understanding, Notion extraction, PDF extraction, Exp 271 trace, method facts | Pending |
| Phase 2 | `02_literature_standards/` | KBS profile, top-tier venue map, paper-structure study, time-series AD structure, figure/table patterns | Pending |
| Phase 3 | `03_blueprint/` | Positioning, novelty/contribution audit, contribution reframing, method interaction map, section outline, claim-evidence map | Pending |
| Phase 4 | `04_references/` | Search plan, verified reference database, quote/usage notes, IEEE list, verification audit | Pending |
| Phase 5 | `05_manuscript/` | Main text draft, appendix draft, placeholder plan, revision history | Pending |
| Phase 6 | `06_language_and_integrity_reviews/` | Style, terminology, plagiarism, citation, KBS/contribution reviews, fix log | Pending |
| Phase 7 | `08_notion_figure_table_specs/` | Notion page design, figure/table specs in Korean, Notion creation log | Pending |
| Phase 8 | `07_latex/` | Elsevier-style LaTeX source, BibTeX, build log, PDF inspection | Pending |
| Phase 9 | `09_final_audit/` | Final audit, requirement coverage matrix, deliverable summary | Pending |

## Source Boundary

Allowed required sources:

- Current project scripts and markdown documents, excluding forbidden paths below.
- Notion method overview: `https://www.notion.so/0-MAE-31387856b20781cd8d4ed14df7f65470?source=copy_link`
- Notion baseline comparison: `https://www.notion.so/Baseline-Comparison-22-Active-Models-9-Datasets-2-Conditions-incl-SMAP-MSL-Pattern-A-B-32087856b2078112b500c81664181ee7?source=copy_link`
- Allowed PDF: `paper/윤기오_대한산업공학회_2026_춘계.pdf`
- Allowed template text: `paper/elsevier template.txt`
- Exp 271 result metadata and code paths needed to verify active components.
- Current official KBS target-journal sources.

Forbidden sources unless the user later explicitly authorizes them:

- `paper_legacy/**`
- `paper/**`, except the two allowed files above.

## Current Next Step

Complete Phase 0 review loop, then begin Phase 1 source inventory and extraction.
