---
phase: 1
agent: orchestrator
directives: [M9]
last_modified: 2026-06-11
---

# Phase 1 보고 — 연구 완전 이해 (절대 엄격 구역: 271 config 진실)

## ① 수행 내용 요약

1. **병렬 작업 5건**: 코드베이스 정독(P1-1) / Notion 2페이지 완전 정독(P1-2, 하위 페이지 포함) / 271 config 포렌식(P1-3, metadata 37 전수 + 코드 추적) / 실험 프로토콜 진실(P1-4, 지표 정식 명칭 웹 검증 포함) / 학회 발표 PDF 34p 전수 정독(P1-5).
2. **모순 정정(reconciliation)**: P1-1과 P1-3의 모순 20건을 1차 소스(metadata·코드·checkpoint 실측)로 전수 판정 — P1-1이 exp271을 Set A로 오인한 것이 주 원인 (실제 Set C 기반+override). GRL 서술은 양쪽 모두 오류여서 코드로 확정 (student decoder 대상 suppression).
3. **종합(P1-6)**: RESEARCH_SYNTHESIS.md — R11 3단 프레이밍(설정 가정 / main 실험 = label 가용성 상한 / 희소화 sweep = 일반 케이스), R10 원재료 표, 논문 제외 목록, Phase 3 판단 사안 8건.
4. **리뷰 루프 3라운드**: r1 리뷰 5인(BLOCKER 15·MAJOR 21 적발 — 핵심 수치층은 전수 검증 통과, 대부분 근거 포인터·서술 정밀도) → 수정 4인(76건 전건 FIXED, 1차 소스 재검증 동반) → 재리뷰 r2 2인(잔존 BLOCKER 4건 정밀 적발) → fixer-5 r3(전건 마감, CSV 수치 일치까지 검증) → orchestrator MINOR 2건 직접 패치.
5. **게이트 감사**: coverage-auditor가 Phase 1 매핑 Directive 18종 전수 근거 확인 + r3 수정분 spot 재검증 4/4 — **PASS**.

## ② 산출물 목록

| 경로 | 내용 (rev) |
|------|-----------|
| `01_research_understanding/271_CONFIG_TRUTH.md` | 기술 사실 최종 정본 (r3) — canonical config, 사용/미사용 표, 제외 목록 26항 |
| `01_research_understanding/CODEBASE_UNDERSTANDING.md` | 코드베이스 전모 (r3) |
| `01_research_understanding/EXPERIMENT_PROTOCOL_TRUTH.md` | 실험 프로토콜 진실 (r3) — split·지표·threshold·excl22·sweep |
| `01_research_understanding/NOTION_DIGEST.md` | Notion 2페이지 digest (r2) — R26 truth 목록 포함 |
| `01_research_understanding/CONFERENCE_PDF_DIGEST.md` | 발표자료 digest (r2) |
| `01_research_understanding/RESEARCH_SYNTHESIS.md` | 전체 종합 (r2+) — 이후 Phase의 1차 입력 |
| `99_reviews/p1_*.md` (12건) | 리뷰·fixlog·재리뷰·게이트 감사 전 기록 |

## ③ 게이트/리뷰 결과

- 리뷰 라운드 3회 (r1 → r2 → r3). 최종 BLOCKER 0 / MAJOR 0 / MINOR 잔존 0 (전건 수정 — waive 없음).
- 엄격 구역(271_CONFIG_TRUTH): 검증자 2인(재추적/완전성 관점) + 전체 재검증 1라운드(재리뷰 α) + 게이트 spot 4/4 — 강화 프로토콜 이행 완료.
- coverage-auditor: Phase 1 매핑 Directive 18/18 근거 확인, T1·M12 DONE 전환, 16종 IN_PROGRESS(후속 Phase 분 잔존).

## ④ 주요 결정사항·확정 사실

- **exp271 정본 확정**: linear patchify(patch 10×50), d_model 512, encoder 4L, teacher 3L/student 2L decoder, 500ep(teacher-only warmup 250), masking 15%(anomaly-first, 8/42), score = recon + scaled_disc/4 (FM은 훈련 전용), GRL = student decoder hidden에 DANN-style suppression. dynamic margin은 도달 불가(코드 검증) — 논문에서 완전 제외.
- **지표 정식 명칭 확정**: VUS-ROC/VUS-PR(PVLDB'22), PA%K 기반 AUC-F1/AUC-PR(AAAI'22), Affiliation-F1(KDD'22), PA-F1(WWW'18, 비판 대상). threshold = test anomaly 비율 분위수.
- **데이터셋 확정**: SWaT(45 feat), WaDi A1/A2(123), PSM, SMD×28, SMAP×54, MSL×27 — Simulation/Exathlon 제외.
- Phase 3 이관 판단 사안: C1~C4 contribution 구조 채택 여부, excl22 수치 기준(0.62730 vs 0.62899), 비교표 Q1/Q3 조건(RF-003), B2 variant 포함 여부 등 8건 (RESEARCH_SYNTHESIS §⑥).

## ⑤ 사용자 확인이 필요하거나 요청할 사항

- 현재 차단 사항 없음. 참고로 알려드릴 사실(작업은 계속): ① 271canon 일부 entity 실행 진행 중(SMD 22/28, SMAP 5/54, MSL 5/27) — 논문은 placeholder 정책(A8/R3)이므로 차단 아님. ② 라벨 희소화 sweep(R32)은 미구현·미실행 — 본문은 '실험 잘 됨' 가정으로 서술하고 Notion placeholder 명세에 실험 설계를 구체 기술 예정. ③ SWaT 입력 feature 수(45)와 현 로더 상태(51) 간 재현성 플래그(RF-005) — 핸드오프 시 상세 보고.

## ⑥ 다음 Phase 예고

**Phase 2 — 탑티어 논문 구조 연구**: venue-scout(최근 3년 탑티어 + 시계열 이상탐지 논문 구조·figure 패턴·문장 corpus) + anchor-paper-analyst(SDMAE dossier — 'self-distilled' 명명 근거 확인 R21, NRdetector dossier — 실험 구성·차이점 재료 R16/R20).
