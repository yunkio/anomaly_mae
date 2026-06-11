---
phase: 6
agent: orchestrator
directives: [M9]
last_modified: 2026-06-11
---

# Phase 6 보고 — 학술 문체 정밀 검증 (문장 단위)

## ① 수행 내용 요약

1. **검사 4종 (3인 독립 + terminology)**: ai-phrasing(corpus 기준 — 187문장, MUST 11: em-dash 절-접합 패턴이 최대 발견) / style-A(영어 산문 — 214문장, MUST 25: 과장문·분사 오류) / style-B(분야 관용 — 67건: "learning units"→entities 등) / terminology(Q1/Q3 내부 코드 11곳, TSAD 미정의, notation 동기화). ※ style-A는 사용자 중단 1회로 재dispatch.
2. **style-fixer → v3**: 전수 처리(적용 ~190건). **의미 보존 거부 16건** — 검사 제안 중 사실 오류 3건(EMA 아님·Student decoder 부착점·family 산법)을 정본 대조로 걸러냄. Q1/Q3 매핑은 orchestrator의 반전 의심이 틀렸고 terminology가 정확함을 정본으로 확정 (Q3=anomaly-excised/Q1=contaminated-training).
3. **재검사 + 회귀 2종**: 수정분 재검사(MUST 전건 실해소 + 신규 MAJOR 3건 적발) / 표절 회귀 **0건** (Phase 5 수정 4건 유지 확인) / method-truth spot **PASS** (47개 변경 지점, Q1/Q3 반전 0).
4. **touch-up**: 신규 MAJOR 3건+철자 1건을 orchestrator가 권고 문안 그대로 적용 (마커 무손상 기계 확인) → 게이트 확인 감사 PASS.

## ② 산출물

`06_style_audit/` 4종, `05_manuscript/MANUSCRIPT_v3.md` (**본문 정본**), `99_reviews/p6_*.md` 6건.

## ③ 게이트/리뷰 결과

3종 검사 잔존 0 (MINOR 4건 D-011 waive — Phase 7 polish 이월) + 회귀 2종 통과. Coverage: T6·R4·R5·R15·R24·R35 DONE (37/57행 DONE).

## ④ 주요 결정

- D-011: MINOR 4건 Phase 7 이월 waive.

## ⑤ 사용자 확인 필요 사항

- 없음 (이전 보고의 Notion 위치·코드 URL 질의는 유효).

## ⑥ 다음 Phase 예고

**Phase 7 — LaTeX 조판 (Elsevier) & PDF 시각 검증**: elsarticle 요구사항 체크리스트 → v3→LaTeX 변환 (이후 .tex이 정본, v3 동결) → placeholder 배치 → 컴파일·pdf-qa 루프 (9페이지 판정 — 분량 잔여 리스크 +0.42p 실측 해소) → overleaf_package.zip self-contained 검증.
