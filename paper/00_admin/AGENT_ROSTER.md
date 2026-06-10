---
phase: 0
agent: orchestrator
directives: [M1, M2, M3]
last_modified: 2026-06-10
---

# Agent Roster — sub-agent 명세 (§5.2 기반 확정)

> **모든 dispatch 공통 규약 (D-001)**: ① 페르소나 정의의 과거 가정(IEEE, 12페이지 등)은 전부 무효 — **Elsevier elsarticle / 본문 9페이지가 진실**. ② 정의에 하드코딩된 파일 경로 전부 무효 — 프롬프트에 입출력 경로 전체 명시. ③ `./paper_legacy/` 접근 절대 금지(A4). ④ 코드베이스 read-only, 쓰기는 배정된 `paper/` 경로만(A9). ⑤ 관련 Directive는 §9 원문 그대로 발췌 포함(의역 금지). ⑥ 의문/요청은 산출물에 `REQUEST:`/`FEEDBACK:` 블록으로.
> **리뷰 분리 원칙**: 작업 agent ≠ 리뷰 agent. 심각도 rubric(BLOCKER/MAJOR/MINOR)과 waive 규칙(§5.3)을 모든 리뷰어 프롬프트에 포함.

| 역할 (마스터 §5.2) | 사용할 agent type | 주요 Phase | 입력 | 산출물 | 리뷰 상대 |
|--------------------|------------------|-----------|------|--------|----------|
| research-archaeologist | paper-codebase-archaeologist | 1 | `mae_anomaly/`, `scripts/`, `docs/`, 루트 .md | `01_research_understanding/CODEBASE_UNDERSTANDING.md` | adversarial-reviewer |
| config-forensics | paper-config-271-archaeologist | 1 | 271 결과 폴더(재귀 metadata 37+개), `mae_anomaly/` | `01_research_understanding/271_CONFIG_TRUTH.md` | 강화: 독립 리뷰어 2인 |
| notion-analyst | paper-notion-analyst (Notion MCP) | 1 | 방법론·비교실험 Notion 페이지 | `01_research_understanding/NOTION_DIGEST.md` | adversarial-reviewer |
| protocol-truth-writer | general-purpose | 1 | 코드/Notion/metadata | `01_research_understanding/EXPERIMENT_PROTOCOL_TRUTH.md` | adversarial-reviewer |
| pdf-digest | general-purpose (PDF Read) | 1 | `paper/윤기오_대한산업공학회_2026_춘계.pdf` | `01_research_understanding/CONFERENCE_PDF_DIGEST.md` | adversarial-reviewer |
| synthesis-writer | paper-methodology-synthesizer | 1 | Phase 1 전 산출물 | `01_research_understanding/RESEARCH_SYNTHESIS.md` | adversarial-reviewer |
| venue-scout | paper-literature-scout (+WebSearch/WebFetch) | 2 | 웹 | `02_venue_study/VENUE_AND_PAPER_LIST.md`, `STRUCTURE_AND_FIGURE_PATTERNS.md`, `SENTENCE_CORPUS.md` | adversarial-reviewer |
| anchor-paper-analyst | paper-core-anchor-analyst (SDMAE) / general-purpose (NRdetector) | 2 | 웹, 논문 원문 | `02_venue_study/ANCHOR_SDMAE_DOSSIER.md`, `NRDETECTOR_DOSSIER.md` | adversarial-reviewer |
| narrative-architect | paper-narrative-architect | 3 | Phase 1·2 산출물 | `03_blueprint/PAPER_BLUEPRINT.md`, `PAGE_BUDGET.md` | outline-red-teamer + adversarial-reviewer (2중) |
| outline-red-teamer | paper-outline-red-teamer | 3 | 블루프린트 | `99_reviews/p3_blueprint_r*.md` | — (리뷰어) |
| claim-citation-mapper | paper-claim-citation-mapper | 4 | 블루프린트 | `04_references/CLAIM_CITATION_MAP.md` | adversarial-reviewer |
| reference-scout | general-purpose (+WebSearch/WebFetch) | 4 | 수요 목록, Notion truth(R26) | 후보 목록 (MAP 갱신) | source-verifier 파이프라인 |
| excerpt-curator | paper-source-excerpt-curator | 4 | 후보 논문 원문 | `04_references/library/{key}.md` | source-verifier A/B |
| source-verifier A | paper-reference-verifier | 4, 5 | reference card | `04_references/VERIFICATION_LEDGER.md` (A 기록) | 게이트: 전수 재감사 리뷰어 |
| source-verifier B | paper-source-triangulator | 4, 5 | **card 비공개** — 공식 소스에서 BibTeX 신규 export | VERIFICATION_LEDGER.md (B 기록), `refs.bib` 원료 | 게이트: 전수 재감사 리뷰어 |
| section-drafter (섹션별) | paper-manuscript-drafter | 5 | 블루프린트·PAGE_BUDGET·reference card·진실 문서 | `05_manuscript/sections/*.md` | 검증 루프 4종 + adversarial-reviewer |
| method-truth-auditor | paper-manuscript-method-auditor | 5, 6(spot), 7(diff) | 본문 ↔ 진실 문서 | `99_reviews/p5_method_truth_r*.md` | — (리뷰어) |
| plagiarism-guardian | paper-plagiarism-guardian | 5, 6, 7(diff) | 본문 ↔ card verbatim/abstract/dossier | `99_reviews/p5_plagiarism_r*.md` | — (리뷰어, 강화) |
| claim-citation-auditor | paper-citation-integrity-auditor | 5 | 본문 ↔ CLAIM_CITATION_MAP·card | `99_reviews/p5_citation_r*.md` | — (리뷰어) |
| ai-phrasing-detector | paper-ai-phrasing-detector | 6, 7(diff) | 본문 + SENTENCE_CORPUS | `06_style_audit/AI_PHRASING_LEDGER.md` | — (리뷰어) |
| style-auditor ×2 | paper-sentence-style-auditor / paper-academic-style-editor | 6 | 본문 | `06_style_audit/STYLE_AUDIT_{A,B}.md` | — (리뷰어 2인 독립) |
| terminology-normalizer | paper-field-terminology-editor | 6 | 본문 | `06_style_audit/TERMINOLOGY_AUDIT.md` | — (리뷰어) |
| latex-engineer | paper-latex-engineer (**Elsevier override**) | 7 | MANUSCRIPT_v3, elsarticle 번들, PLACEHOLDER_REGISTRY | `07_latex/` 프로젝트 + `TEMPLATE_REQUIREMENTS.md` + zip | pdf-qa-reviewer |
| pdf-qa-reviewer | fresh-paper-pdf-visual-qa-reviewer | 7 | 컴파일된 PDF | `07_latex/pdf_qa/` | — (리뷰어) |
| adversarial-reviewer (범용) | paper-adversarial-reviewer | 전체 | 각 산출물 | `99_reviews/{phase}_{artifact}_{round}.md` | — (리뷰어) |
| final-peer-reviewer ×2 | paper-final-package-reviewer / fresh-paper-feedback-top-venue-reviewer | 8 | 최종 PDF/LaTeX | `08_final_audit/FINAL_AUDIT_REPORT.md` | — (리뷰어 2인, 학회 리뷰 양식) |
| coverage-auditor | general-purpose (독립) | 0, 각 게이트, 8 | §9, ORIGINAL_USER_DIRECTIVES, COVERAGE_MATRIX, 산출물 | `99_reviews/p{N}_coverage_r*.md` | — (감사자) |
| notion-publisher | notion-expert | 8 | 검수 통과한 NOTION_PLACEHOLDER_SPECS.md | Notion 하위 페이지 | 독립 리뷰어(발행 전 명세 검증) |

비고:
- 역할이 더 세분화될 필요가 생기면 이 표에 행을 추가하고 DECISION_LOG에 기록한다.
- `fresh-paper-*` 계열은 동종 역할의 대체 풀로 사용 가능 (독립 2인 검증 시 서로 다른 계열을 써서 관점 독립성 확보).
