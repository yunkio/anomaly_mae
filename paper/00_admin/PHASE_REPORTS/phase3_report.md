---
phase: 3
agent: orchestrator
directives: [M9]
last_modified: 2026-06-11
---

# Phase 3 보고 — 논문 블루프린트

## ① 수행 내용 요약

1. **narrative-architect**: Phase 1 진실 + Phase 2 패턴으로 PAPER_BLUEPRINT + PAGE_BUDGET 작성 (Elsevier 요소 포함 전체 구조, contribution 설계, 결정 사안 6건 처리, 모델명·제목 후보).
2. **2중 적대 리뷰 (r1)**: red-team(B3/M10 — test-label 학습 사용 공격 방어 빈틈, warmup contribution 모순, 방법론-프로토콜 효과 미분리) + adversarial(B5/M12 — 분량 불일치, GRL 위치, SMD F, λ 공식, epoch 비대칭 방어 부재).
3. **개정 r2**: 49건 전수 처리 — §14 프로토콜 방어 5논거 재구축, warmup contribution 제외, Table 4 protocol-effect 보조분석 main text 격상, d_model=512 고정 발견 등.
4. **재리뷰 r2**: red-team 관점 — BLOCKER 3건 실질 해소 확인(코드 수준 검증 포함). adversarial 관점 — **신규 BLOCKER 2건 적발**: GRL λ는 이중 구조(손실 가중 grad-ratio×0.2 + 반전 계수 Ganin sigmoid ramp), warmup 중 student forward는 skip — 근본 원인이 Phase 1 정본 누락으로 판명.
5. **§6.3 회귀 프로토콜 발동**: Phase 1 정본 보강 (271_CONFIG_TRUTH r4, CODEBASE r4, SYNTHESIS r3 — 코드 file:line 확정) + 블루프린트 r3.
6. **R15 선정 (D-007)**: 모델명 **CSMAD**, 제목 **"Label-Aware Masked Autoencoding with Gradient Reversal for Multivariate Time Series Anomaly Detection"** (Self-Distilled 전면 배치 후보는 R9 위험으로 기각).
7. **게이트 감사**: spot 6/6 (코드 1차 소스), r2 발견 마감 11/11, Directive 17/17 — **PASS**.

## ② 산출물

`03_blueprint/PAPER_BLUEPRINT.md` (r3), `PAGE_BUDGET.md` (r3, 분량 단일 정본) + `99_reviews/p3_*.md` 7건 + Phase 1 정본 3종 r4/r3 보강 + DECISION_LOG D-005~D-007.

## ③ 게이트/리뷰 결과

리뷰 3라운드 (r1 2중 → r2 재리뷰 2중 → r3 game). 최종 BLOCKER 0 / MAJOR 0. Phase 1 재진입 1회 기록 (PHASE_LEDGER).

## ④ 주요 결정 (DECISION_LOG)

- D-005: contribution 4-bullet 재설계(C1–C4 수정/기각), setting="contaminated semi-supervised", excl22 기준=excl22 entity headline, Q3 main + Table 4 보조분석, SDMAE 옵션 C, 코드 공개 조건부.
- D-006: DAGMM simplified variant 표기, TS-SDMAE 후보 제외, PAGE_BUDGET 정본, d_model 512 고정, P1 정본 회귀 보강.
- D-007: CSMAD + 제목 후보 2 (R9 사유).

## ⑤ 사용자 확인 필요 사항 (작업은 계속)

- **코드 공개 URL (결정 ⑥)**: 본문에 "Code is available at [URL]" placeholder로 포함 예정 — 공개 repo URL 확정 시점에 알려주시면 반영합니다 (미확정이어도 placeholder로 진행).
- R3의 "[노션 페이지]" 해석: placeholder 명세는 **비교 실험 페이지 하위 단일 명세 페이지**로 발행하는 것이 기본값입니다 (Phase 8). 다른 위치를 원하시면 알려주세요.

## ⑥ 다음 Phase 예고

**Phase 4 — Reference 확보 & 절대 검증** (절대 엄격 구역: 할루시네이션 0): claim-citation-mapper → reference-scout → excerpt-curator(카드) → 2인 독립 서지 검증(A: 공식 소스 검증 / B: card 비공개 BibTeX 신규 export) → 기계 diff → REFERENCES_IEEE + refs.bib.
