---
phase: 2
agent: orchestrator
directives: [M9]
last_modified: 2026-06-11
---

# Phase 2 보고 — 탑티어 논문 구조 연구

## ① 수행 내용 요약

1. **venue-scout**: 2024–2026 탑티어 학회 리스트업 + Elsevier 저널 관례(타깃 포맷 반영) + 논문 14편(TSAD 11편) 선정·분석 — intro 4단 논증 패턴, contribution 제시 방식, related work 조직법, method 소절 구조, 실험 서술 순서, figure/table 유형 10종 + 9페이지 분량 배분안.
2. **corpus-collector**: 11편에서 섹션 유형 10종 × 5–10문장 = 92엔트리 verbatim corpus + 분야 collocation 7범주 + AI-티 금지 패턴 시드 (Phase 6 기준).
3. **anchor-paper-analyst (SDMAE)**: 'self-distilled' 명명 근거 원문 확보 — **용어 계보 발견: Zhang et al. (TPAMI 2022) → SDMAE (CVPR 2024) → 본 연구**, R21 방어가 2단 계보로 강화됨. 유사 12/차이 17 + 위험도 평가 + 포지셔닝 옵션 3종(권장: related work distillation 계보 내 자연 언급). anomaly-map 분기와 GRL의 개념적 평행에 대한 방어 3축 정리.
4. **nrdetector-analyst**: 실험 구성 전모(라벨 sweep 설계 선례), R19 인용 처리 선례(related work 내 baseline 모델명 0건 — grep 검증), R20 차이축 D1–D9 + "시계열 SSL/PU 거의 없음" 주장의 안전한 정밀 스코핑.
5. **리뷰 루프**: r1 리뷰 2인 (verbatim 36건 바이트 단위 대조 — 할루시네이션 0, BLOCKER 0, MAJOR 4) → fixer 26건 전수 처리 → 게이트 감사 spot 7/7 PASS.

※ 작업 중 사용자 중단 1회 발생 — 중단 시점 디스크 검증으로 NRdetector dossier 완결 확인(리뷰에서 추가 검증), 미생성분 2건만 재dispatch. 품질 영향 없음.

## ② 산출물

`02_venue_study/` 5종 (VENUE_AND_PAPER_LIST, STRUCTURE_AND_FIGURE_PATTERNS, SENTENCE_CORPUS, ANCHOR_SDMAE_DOSSIER, NRDETECTOR_DOSSIER) + `99_reviews/p2_*.md` 4종.

## ③ 게이트/리뷰 결과

리뷰 2라운드 (r1 → fix r2). 최종 BLOCKER 0 / MAJOR 0. 게이트 기준 4항목(실행 가능 패턴 / TSAD 포함 / corpus 확보 / dossier 완성도) 전부 충족. Directive 6종(T2 DONE, R9·R16·R19·R20·R21 P2분 충족) 근거 확인.

## ④ 주요 결정사항·발견

- **R21 방어 강화**: self-distillation 용어는 Zhang et al. TPAMI 2022가 원류이고 SDMAE가 AD에 적용한 선례 — 본 연구는 이 계보의 시계열 확장으로 서술 가능. 'SDMAE가 만든 용어' 식 서술은 사실 위반으로 금지 플래그.
- **R9 포지셔닝 권장안**: related work의 distillation 계보 흐름 속 1–2문장 자연 언급 (차이점 나열식 금지) — Phase 3에서 최종 채택.
- Phase 4 인용 후보 메모: TSB-AD "Elephant in the Room" (NeurIPS'24 D&B — VUS-PR 권고, R29 지표 정당화), Zhang TPAMI 2022 (R21).
- Phase 5 표절 검사 고위험 목록 등재 (RF-006): DCdetector·SDMAE 표면 유사 문장.

## ⑤ 사용자 확인 필요 사항

- 없음.

## ⑥ 다음 Phase 예고

**Phase 3 — 논문 블루프린트**: narrative-architect가 Phase 1 진실 + Phase 2 패턴으로 PAPER_BLUEPRINT/PAGE_BUDGET 작성 (MECE, contribution 설계 R8, PU 중심 환경 R11, R10 논증 구조, 모델명·제목 후보 R15) → outline-red-teamer + adversarial-reviewer 2중 리뷰.
