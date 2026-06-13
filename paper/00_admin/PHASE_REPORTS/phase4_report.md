---
phase: 4
agent: orchestrator
directives: [M9]
last_modified: 2026-06-11
---

# Phase 4 보고 — Reference 확보 & 절대 검증 (절대 엄격 구역: 할루시네이션 0)

## ① 수행 내용 요약

1. **claim-citation-mapper**: 블루프린트 전수 추출 — claim 85개 (인용 필요 72: 필수 52/권장 20), OPEN 31.
2. **reference-scout**: OPEN 전수 해소 (CANDIDATE 30 / NOT_FOUND 1). **중대 발견 2건**: ① 최초성 반증 후보 (Xue & Yan IJCNN 2022, SLA-VAE WWW 2022) → contribution 스코핑 축소 (D-008) ② "contaminated semi-supervised" 선사용 부재 → 신조어 정의 가능. R26 truth의 venue 오류 4건 교정 (WETAS→ICCV 2021, TreeMIL→ICASSP 2024 등).
3. **excerpt-curator ×3**: reference card 49편 (FULL 22 — verbatim 발췌+활용 맥락+abstract, LIGHT 27 — 서지+abstract+역할; A2 경고문 전 카드).
4. **2인 독립 검증 (구조적 독립 보장)**: A(card↔공식 소스 — 199+166 tool call) — **할루시네이션급 저자 오류 4건 적발·정정** (xu2018kpivae 24→13인, treemil·rosas·xue 저자명) + 발췌 13건 해소 (PA%K 정의, focal 수식, GRL λ schedule, AR-threshold 등 — R30 보류 해제). B(card 비공개, blind seed→공식 BibTeX export) — 49편 전건 공식 export. **B의 blind 구조가 실제로 작동**: seed 결함(제목 누락 11건) 중 1건(zhang)의 오매칭을 diff가 적발 → 재export로 해소.
5. **orchestrator 기계 diff**: 완전 일치 33 / 표기 관례 10 / 실질 충돌 6 전건 해소 (tiebreak: DBLP/Crossref/proceedings 직접 재질의). **QUARANTINE 0**.
6. **조립**: refs.bib (49, 공식 export만), REFERENCES_IEEE, REFERENCE_LIBRARY_INDEX, 통합 VERIFICATION_LEDGER, P4_DIFF_REPORT.
7. **강화 게이트**: 전수 재감사(49행 기계 대조 모순 0) + 무작위 16편 공식 소스 재검증 전건 일치. GB-1 (refs.bib 1항목 구문 결함 — orchestrator 병합 시 도입) 적발 → DBLP verbatim export 교체 + 49/49 파싱 검증 → **PASS**.

## ② 산출물

`04_references/` 전체 (refs.bib, CLAIM_CITATION_MAP r3, VERIFICATION_LEDGER + 4분할, P4_DIFF_REPORT, REFERENCES_IEEE, REFERENCE_LIBRARY_INDEX, library/ 49 cards) + `99_reviews/p4_coverage_gate_r1.md`.

## ③ 게이트/리뷰 결과

검증 라운드: 2채널 독립 + 기계 diff + 전수 재감사 + 무작위 재검증 = 사실상 4중. 최종 BLOCKER 0 / QUARANTINE 0 / 비검증 인용 경로 0. EXCERPT_UNVERIFIED 잔존 3건은 2단계 격리 (서지 인용 가능, verbatim 금지).

## ④ 주요 결정

- D-008: 최초성 주장을 "masked-reconstruction self-distillation + GRL adversarial 통합" 수준으로 한정, Xue&Yan·SLA-VAE는 related work 인용·차별화.
- 해소 6건: blazquez year=2022, DACAD=TKDE 2025 본판, lai 저자 "Jeffrey H. Lang"(A측 card 오류 — B 정당), zhang 재export, sultani DOI 추가, NRdetector pages=1551–1562.
- CSMAD 명칭 충돌 없음 확인 (D-008 후속).

## ⑤ 사용자 확인 필요 사항

- 없음.

## ⑥ 다음 Phase 예고

**Phase 5 — 영어 본문 작성** (절대 엄격 구역: 표절 0·진실 정합·수치 창작 0): 섹션별 drafter → MANUSCRIPT_v1 → 인용 보강(R36) → 검증 루프 4종 병렬 (method-truth/plagiarism/claim-citation 양방향/adversarial) → v2. PLACEHOLDER_REGISTRY 가동.
