---
phase: 0
agent: orchestrator
directives: [M5, M7, R14]
last_modified: 2026-06-10
---

# Task Board

> 태스크 단위 진행 상황. **dispatch 시점과 완료 시점에 즉시 갱신** (게이트 시점이 아님).
> 절대 엄격 구역의 2인 검증은 **검증자별 개별 행**으로 기록.
> 상태: PLANNED / DISPATCHED / DONE / BLOCKED / REWORK.

## Phase 0 — 셋업 & 지시사항 내재화

| Task | 담당 | 상태 | 비고 |
|------|------|------|------|
| P0-1 워크스페이스 구조 생성 (§4) | orchestrator | DONE | 2026-06-10 |
| P0-2 COVERAGE_MATRIX.md 작성 + 57행 기계 대조 | orchestrator | DONE | T7+R37+M13=57 확인 |
| P0-3a 감사 A — Registry 원문 충실성 | coverage-auditor-A (독립) | DONE | PASS (B0/M0/m2 — ERRATA E-002·E-003 기록). 웹 도구 sub-agent 가용 확인 포함 |
| P0-3b 감사 B — Matrix 완전성·매핑 정합 (r1) | coverage-auditor-B (독립) | DONE | 조건부 FAIL (B0/M1/m6) → Matrix 6행 보정 |
| P0-3c 수정분 재리뷰 (r2) | coverage-auditor-r2 (독립 신규) | DONE | PASS (B0/M0/m1 → R19 표기 즉시 보정) |
| P0-4 AGENT_ROSTER.md 확정 | orchestrator | DONE | |
| P0-5 pre-flight (a) elsarticle 번들 + cls 설치 | orchestrator | DONE | kpsewhich PASS |
| P0-5 pre-flight (b) Notion MCP 2페이지 접근 | orchestrator | DONE | 방법론 78,956자 / 비교실험 112,046자 수신 |
| P0-5 pre-flight (c) 학회 PDF 읽기 | orchestrator | DONE | p.1–2 정상 렌더 |
| P0-5 pre-flight (d) latexmk/pdflatex | orchestrator | DONE | latexmk 4.76 |
| P0-5 pre-flight (e) 271 metadata 전수 개수 | orchestrator | DONE | **37개** (2026-06-10 실측) |
| P0-5 pre-flight (f) WebSearch+WebFetch (DBLP/arXiv) | orchestrator | DONE | arXiv·DBLP fetch PASS; sub-agent 가용성은 P0-3 dispatch에서 확인 |
| P0-6 PHASE_LEDGER/TASK_BOARD Phase 1–8 계획 등재 | orchestrator | DONE | 본 파일 하단 |
| P0-7 phase0_report.md + git commit | orchestrator | DONE | 게이트 PASS — `Paper: Phase 0 setup complete (gate passed)` |

## Phase 1–8 계획 (개요 등재 — 상세 태스크는 각 Phase 시작 시 전개)

| Phase | 핵심 태스크 (마스터 §7) | 엄격 구역 |
|-------|------------------------|----------|
| 1 | CODEBASE_UNDERSTANDING / NOTION_DIGEST / 271_CONFIG_TRUTH / EXPERIMENT_PROTOCOL_TRUTH / CONFERENCE_PDF_DIGEST / RESEARCH_SYNTHESIS + 리뷰 루프 | 271_CONFIG_TRUTH (코드 근거 file:line, 리뷰어 2인) |
| 2 | VENUE_AND_PAPER_LIST / STRUCTURE_AND_FIGURE_PATTERNS / SENTENCE_CORPUS / ANCHOR_SDMAE_DOSSIER / NRDETECTOR_DOSSIER + 리뷰 | — |
| 3 | PAPER_BLUEPRINT / PAGE_BUDGET / 모델명·제목 후보 / red-team 2중 리뷰 | — |
| 4 | CLAIM_CITATION_MAP / reference 탐색·card / 2인 독립 서지 검증 / refs.bib / REFERENCES_IEEE | 서지 검증 전체 (할루시네이션 0) |
| 5 | 섹션별 드래프트 → v1 → 인용 보강(R36) → 검증 루프 4종 → v2 / PLACEHOLDER_REGISTRY | 표절·인용 정합·진실 정합·수치 창작 0 |
| 6 | AI_PHRASING_LEDGER / style 2인 / terminology / 회귀 검사 2종 → v3 | — |
| 7 | TEMPLATE_REQUIREMENTS / LaTeX 변환 / placeholder 삽입 / 컴파일·pdf-qa 루프 / overleaf_package.zip self-contained 검증 | — |
| 8 | FINAL_AUDIT(모의 피어리뷰 2인) / coverage 최종 전수(57행) / NOTION_PLACEHOLDER_SPECS → 검수 → 발행 / 핸드오프 | 최종 감사 |
