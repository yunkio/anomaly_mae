---
phase: 0
agent: orchestrator
directives: [M3]
last_modified: 2026-06-10
---

# Requests & Feedback 라우팅 테이블

> agent 산출물 내 `REQUEST:` / `FEEDBACK:` 블록을 orchestrator가 여기에 등재하고 대상 agent에게 라우팅, 결과를 회신시킨다.
> 상태: OPEN / ROUTED / RESOLVED / WONTFIX(사유 필수).

| # | 일시 | 요청자 → 대상 | 유형 | 내용 요약 | 상태 | 해소 기록 |
|---|------|--------------|------|----------|------|----------|
| RF-001 | 2026-06-10 | orchestrator → reconciler | FEEDBACK | P1-1↔P1-3 모순 (patchify, encoder/decoder 층수, warmup, dynamic margin 등) | RESOLVED | reconciler 전수 판정 20건 (`99_reviews/p1_reconciliation_r1.md`): P1-3 승 16, P1-1 승 1(test stride 산식), 양쪽 부분 오류 2(masking 8/42, GRL=student decoder 대상 suppression), 정밀화 1. 원인: P1-1의 Set A/C 오인. 3개 문서 정정 완료 |
| RF-002 | 2026-06-10 | protocol-truth-writer → reconciler | REQUEST | affiliation-F1·PA-F1 threshold 확정 + `pak_auc_pr` 키 매핑 | RESOLVED | `pa_0_f1`=F1-최적 threshold(`evaluator.py:929-955`), `pa_0_f1_ar` 부재 확정. affiliation은 F1-최적/`_ar` 양립. "pak_auc_pr"=내부 키 `pak_auc_prc_auc` |
| RF-003 | 2026-06-10 | protocol-truth-writer → orchestrator | REQUEST | 비교표의 Q1/Q3 조건 확정 | RESOLVED | D-005 ④: main table = anomaly-excised(Q3) 단독 + standard-split 비교는 본문 보조 블록·Appendix B.1 (P3 확정, P5~P7 반영) |
| RF-004 | 2026-06-10 | protocol-truth-writer → orchestrator | FEEDBACK | 271canon 미완주 entity (SMD 22/28, SMAP 5/54, MSL 5/27), WaDi A2 feature 123 vs 127, RankAvg 재계산 등 6건 | RESOLVED(부분) | WaDi A2=123 확정 (all-NaN 4 sensor drop, 직접 재현). 미완주·RankAvg는 placeholder 정책(A8/R3)상 본문 차단 요소 아님 — Phase 8 Notion 명세·핸드오프에 반영 예정 |
| RF-005 | 2026-06-10 | reconciler → orchestrator | FEEDBACK | SWaT 모델 입력 45 features (= 51 − combined-constant 6) 삼중 확인되나, 현 machineA CSV+loader는 51 반환 — 재현성 플래그 | CARRIED(핸드오프) | 논문 서술은 271 metadata(45) 기준. 재현성 이슈는 Phase 8 핸드오프 보고에 등재 |
| RF-006 | 2026-06-11 | P2 리뷰어 A (C-005) → Phase 5 plagiarism-guardian | FEEDBACK | 표절 검사 최우선 고위험 문장 목록 | RESOLVED | P5/P6 plagiarism 검사에 포함 — 전부 클린 판정 |
| RF-007 | 2026-06-11 | P2 fixer (C-002) → orchestrator | REQUEST | SENTENCE_CORPUS RigorEval AAAI 2022 직접 소스 주석 | RESOLVED | orchestrator가 P1 검증된 AAAI OJS URL 주석 추가 (2026-06-11) |
| RF-008 | 2026-06-11 | P2 fixer → Phase 4 | FEEDBACK | TSB-AD·zhang TPAMI 인용 후보 | RESOLVED | P4에서 liu2024elephant·zhang2022selfdistill로 검증·인용 완료 |
