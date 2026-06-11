---
phase: 5
agent: orchestrator
directives: [M9]
last_modified: 2026-06-11
---

# Phase 5 보고 — 영어 본문 작성 (절대 엄격 구역: 표절 0·진실 정합·수치 창작 0)

## ① 수행 내용 요약

1. **섹션 drafter 4인 병렬**: front(abstract/keywords/highlights/intro/conclusion) / related work / method / experiments — 출판 수준 영어, D-007 제목·CSMAD, D-008 스코핑 준수.
2. **통합(v1)**: 인용 key 검증(무효 1건 정정), placeholder 전역 ID 재부여, 용어 통일, 모순 16건 처리. **분량 11.8p 적발** (예산 9.0p).
3. **분량 수술 2회 (D-009/D-010)**: Appendix A/B/C 신설·이관 (지표 정의·excl22 유도·구현 세부·보조 수식), Table 4→Table 2 흡수, conditional ablation 강등 — 의무 서술(R13/R28/R29/R30/R31/R32) 삭제 0건. 결과 10.42p (잔여 +0.42p는 Phase 7 LaTeX 실측 ±0.76p 오차범위 — 실측 판정 인계).
4. **검증 5종 병렬 (99건 발견)**: R36 인용 공백 15 (신규 reference 수요 0 — 전부 기존 49 key·재서술 해소) / method-truth BLOCKER 9 (PA%K 격자, 창작 LayerNorm, 수식 정밀도 3건, 허위 batch 수치, 추론 마스킹, GRL property, complementary masking) / 표절 MAJOR 4 (6-gram 1건 포함 — 고위험 목록은 전부 클린) / 인용 역방향 109 인스턴스 (UNSUPPORTED 6 — xue 오귀속 등) / adversarial BLOCKER 8 (GRL 필요성 논증의 구조적 결함 포함).
5. **종합 수정 (94건)**: GRL 논증을 encoder-문맥 오염 경로로 재구축, notation 6충돌군+미정의 5건 해소, D-008 정렬, 표절 해체 — placeholder 정책(R3/A8)과 충돌하는 지적 5건은 Directive 원문 사유로 기각 (기각 사유 게이트 정합 판정 통과).
6. **게이트**: 마감 99/99, 고위험 재추적 14/14 (코드 file:line), A8 수치 재스윕 미등재 0, 인용 무효 0, R9 차이-나열 0, Directive 32종 근거 확인. F-1 (fixlog 기록 누락 1문장) 정정 후 **PASS**.

## ② 산출물

`05_manuscript/MANUSCRIPT_v2.md` (정본), `PLACEHOLDER_REGISTRY.md` (49종), 통합·수술 보고 2건, `99_reviews/p5_*.md` 7건, 정본 errata (EXPERIMENT_PROTOCOL_TRUTH r4).

## ③ 게이트/리뷰 결과

리뷰 3라운드 (검증 5종 → 종합 수정 → 게이트+F-1). 최종 BLOCKER 0 / MAJOR 0 / 수치 창작 0 / 표절 0 / QUARANTINE 인용 0.

## ④ 주요 결정

- D-009/D-010: 분량 수술 (Appendix 이관 원칙 + 의무 서술 불가침 + 9.0p 유지, Phase 7 실측 판정).
- placeholder 정책 우선 기각 5건 (R3/A8 vs 리뷰어의 "실험 선행" 요구 — Directive 우선).

## ⑤ 사용자 확인 필요 사항 (작업은 계속)

- **R3 "[노션 페이지]" 해석 재확인 (마스터 §7 Phase 8 규정에 따른 정식 질의)**: placeholder 49종의 한국어 명세 하위 페이지를 **비교 실험 Notion 페이지 하위에 단일 페이지(figure/table별 섹션)**로 발행하는 것이 기본값입니다. 다른 위치/구조를 원하시면 Phase 8 전까지 알려주세요 — 응답 없으면 기본값으로 진행합니다 (D-003).
- 코드 공개 URL: 본문에 placeholder로 포함됨 — URL 확정 시 알려주시면 반영.

## ⑥ 다음 Phase 예고

**Phase 6 — 학술 문체 정밀 검증**: ai-phrasing-detector (SENTENCE_CORPUS 기준 문장 단위 전수) + style-auditor 2인 독립 + terminology-normalizer → 수정 → 재검사 → v3 + 회귀 검사 2종 (표절 재검 + method-truth spot).
