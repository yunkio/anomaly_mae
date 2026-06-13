---
phase: 0
agent: orchestrator
directives: [M10]
last_modified: 2026-06-10
---

# Errata — MASTER_ORCHESTRATION_PROMPT.md 정오 기록

> 마스터 문서 자체는 수정하지 않고, 발견된 오류/모호/대체 사항을 여기에 기록한다 (§4 규칙).

| # | 일시 | 위치 | 내용 | 처리 |
|---|------|------|------|------|
| E-001 | 2026-06-10 | §9.1 T7 / §3 | 원문 T7이 지칭한 `paper/elsevier template.txt`는 존재하지 않음. 공식 Elsevier elsarticle 번들 `paper/elsarticle/`(CTAN 2024-04; 템플릿 3종 + .bst 3종 + elsdoc.pdf)로 대체됨 — 마스터 문서 §3에 이미 명시된 대체이며, Phase 0 pre-flight (a)에서 번들 존재 + `kpsewhich elsarticle.cls` 시스템 설치(`/usr/share/texlive/texmf-dist/tex/latex/elsarticle/elsarticle.cls`) 재확인 완료. | 기록 완료. Phase 7은 elsarticle 번들 기준으로 진행. |
| E-002 | 2026-06-10 | §9.3 M11 | M11 재서술에서 원문 "그에 따르는 phase와 **각 phase의 프롬프트로** 재구성" 중 '각 phase의 프롬프트' 자구가 탈락 (감사 A, MINOR). 의미는 §5.1(프롬프트 작성 규약)·§7(Phase별 프롬프트 지침)에 운영적으로 보존됨. | 기록 완료. M11 이행 시 "Phase별 sub-agent 프롬프트 품질" 자체가 요구사항임을 상시 인지 — dispatch 프롬프트 품질이 M11 충족 근거의 일부. |
| E-003 | 2026-06-10 | §9.4 표 | §9.4 매핑 표에서 M8 행이 M13 뒤 최하단에 배치 (ID 순서 이탈; 감사 A·B 공통, MINOR). 전 57 ID 존재 자체는 확인됨. | 기록 완료. COVERAGE_MATRIX.md는 ID 순서대로 정렬하여 전사함 (M8은 M7 다음). |
