---
phase: 0
agent: orchestrator
directives: [M1, M11]
last_modified: 2026-06-10
---

# Decision Log

> 모든 중요 결정 + 근거. (예: contribution 구조 채택/기각, 모델명, placeholder 위치, waive 사유)

| # | 일시 | Phase | 결정 | 근거 | 관련 Directive |
|---|------|-------|------|------|---------------|
| D-001 | 2026-06-10 | 0 | sub-agent는 환경에 정의된 기존 페르소나(`paper-*`, `fresh-paper-*`, `general-purpose` 등)를 역할 적합 시 재사용하되, 모든 dispatch 프롬프트에 ① 과거 가정 override(Elsevier 템플릿·9페이지가 진실), ② 페르소나 정의 내 하드코딩 경로 전부 무효 + 이번 태스크 입출력 경로 명시, ③ A4(`paper_legacy/` 접근 절대 금지) 경고, ④ A9(코드 read-only) 명시를 포함한다. 매핑은 AGENT_ROSTER.md에 기록. | 마스터 §5.1 — 재사용 허용 + override 의무. 과거 `paper/` 산출물이 `paper_legacy/`로 이동했으므로 정의 경로 탐색이 A4 위반의 전형 경로. | M1, R37, A4, A9 |
| D-002 | 2026-06-10 | 0 | 서지 검증 2인 독립 구조: verifier A = card 메타데이터를 공식 소스에서 검증(`paper-reference-verifier` 계열), verifier B = card를 보지 않고 DBLP/publisher에서 BibTeX 신규 export(`paper-source-triangulator` 계열). orchestrator가 필드 단위 기계 diff. 서로의 ledger 비공개. | 마스터 §7 Phase 4 절차 4의 구조적 독립성 요구. | T4, A1 |
| D-003 | 2026-06-10 | 0 | Phase 진행 중 사용자 확인 필요 사항은 Phase 보고 ⑤항에 모아 전달하고 보수적 기본값으로 작업 계속 (마스터 §8 자율 진행 규칙). 단 §8 (a)–(e) 블로커 조건에서는 정지. | 마스터 §8. | M9, M13 |
| D-004 | 2026-06-10 | 0 | `.gitignore:68`의 `paper/` ignore 규칙(이전 작업물 시절 규칙)이 마스터 §4의 "매 게이트 paper/ commit" 요구와 충돌 → 규칙을 `paper_legacy/` + LaTeX build 부산물 ignore로 교체. A9의 read-only 범위는 코드·실험 환경이며, 이 수정은 마스터 명시 요구 이행을 위한 최소 인프라 변경으로 판정. | 마스터 §4 commit 규칙 vs 구식 ignore 규칙 충돌. paper_legacy/를 ignore함으로써 A4(legacy 접근/오염 방지)도 강화. | M7, R37(보조), A9(해석) |
