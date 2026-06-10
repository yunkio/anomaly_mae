---
phase: 0
agent: orchestrator
directives: [R14]
last_modified: 2026-06-10
---

# INDEX — 전 산출물 인덱스

> 매 Phase 게이트 통과 시 갱신. "X에 대한 내용이 어디 있지?"에 이 파일만 보고 답할 수 있어야 한다.

## 최상위 (수정 금지 입력물)

- `paper/MASTER_ORCHESTRATION_PROMPT.md` — 프로젝트 유일 최상위 지시서. §7 Phase 계획, §9 Directive Registry(헌법, 57개). 의심스러우면 여기.
- `paper/ORIGINAL_USER_DIRECTIVES.md` — 사용자 지시 원문(2026-06-10). §9의 원본, 감사 최종 기준.
- `paper/elsarticle/` — 공식 Elsevier elsarticle 번들(CTAN 2024-04). 템플릿 3종, .bst 3종, doc/elsdoc.pdf. Phase 7 기준. **수정 금지 — 작업 사본은 07_latex/에.**
- `paper/윤기오_대한산업공학회_2026_춘계.pdf` — 학회 발표자료(한국어). 연구 핵심 논리 요약. Phase 1에서 digest 생성.

## 00_admin/ — 관리

- `COVERAGE_MATRIX.md` — Directive 57행(T7+R37+M13) × Phase × 상태 × 충족 근거. 최종 게이트 100% 요구.
- `PHASE_LEDGER.md` — Phase별 시작/종료/게이트/반복/재진입. **세션 재개 시 여기부터.**
- `TASK_BOARD.md` — 태스크 단위 진행 (dispatch/완료 즉시 갱신). Phase 1–8 계획 개요 포함.
- `DECISION_LOG.md` — 중요 결정 + 근거 (D-001 페르소나 재사용 규약, D-002 2인 검증 구조, D-003 자율 진행 기본값).
- `ERRATA.md` — 마스터 문서 정오 (E-001 elsevier template.txt → elsarticle/ 대체).
- `AGENT_ROSTER.md` — 역할 ↔ 실제 agent type 매핑, 입출력, 리뷰 상대, dispatch 공통 규약.
- `REQUESTS_AND_FEEDBACK.md` — agent 간 요청·피드백 라우팅 테이블.
- `PHASE_REPORTS/` — phase0_report.md ~ phase8_report.md.

## 01_research_understanding/ — Phase 1 (예정)

(Phase 1 산출물: CODEBASE_UNDERSTANDING, NOTION_DIGEST, 271_CONFIG_TRUTH, EXPERIMENT_PROTOCOL_TRUTH, CONFERENCE_PDF_DIGEST, RESEARCH_SYNTHESIS)

## 02_venue_study/ — Phase 2 (예정)

## 03_blueprint/ — Phase 3 (예정)

## 04_references/ — Phase 4 (예정)

## 05_manuscript/ — Phase 5 (예정)

## 06_style_audit/ — Phase 6 (예정)

## 07_latex/ — Phase 7 (예정)

## 08_final_audit/ — Phase 8 (예정)

## 99_reviews/ — 모든 리뷰 산출물 `{phase}_{artifact}_{round}.md`

- `p0_registry_fidelity_A_r1.md` — 감사 A: §9 ↔ 사용자 원문 문자 단위 diff 전수 대조 (PASS, MINOR 2 → ERRATA E-002·E-003). sub-agent 웹 도구 가용성 확인 포함.
- `p0_matrix_completeness_B_r1.md` — 감사 B r1: Matrix 57행·요약 왜곡·3-way 매핑(§9.4↔§7↔Matrix) 감사 (MAJOR 1: R29 요약 탈락 → 보정).
- `p0_matrix_completeness_B_r2.md` — 수정분 재리뷰: 보정 6행 원문 대조 + 57행 무손상 확인 (PASS).
