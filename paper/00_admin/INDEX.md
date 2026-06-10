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

## 01_research_understanding/ — Phase 1 (완료, 게이트 PASS 2026-06-11)

- `271_CONFIG_TRUTH.md` (r3) — **기술적 사실의 최종 정본.** exp271 = Set C 기반+override. metadata 37 전수 + canonical config(공통 114키) + 사용/미사용 component 표(file:line) + 논문 제외 목록 26항 (dynamic margin 도달불가, Gaussian smoothing q3 스크립트만, SCAD/discriminator/RevIN/EMA 등). 모델: linear patchify(10×50), d_model 512, encoder 4L, teacher 3L/student 2L, 500ep/250 warmup, masking 15%(8/42), score=recon+scaled_disc/4.
- `CODEBASE_UNDERSTANDING.md` (r3) — 코드베이스 전모: 아키텍처/loss(GRL 3경로 adaptive λ 구분)/scoring(leave-one-out batch 확장)/데이터 파이프라인/학습 루프/평가(~153 metrics)/post-mortem 3건.
- `EXPERIMENT_PROTOCOL_TRUTH.md` (r3) — split //2 전수 라인 + SMAP/MSL safe-cut 실측(무제한 outward), normalonly 비교군, 지표 정식명칭 매핑(VUS PVLDB'22, PA%K AAAI'22, Affiliation KDD'22, PA WWW'18), AR threshold, SWaT excl22(83.75% bit-exact), 희소화 sweep 미구현(placeholder 입력), 실행 프로토콜(seed 42, MAE 500ep/unsup 10ep/weak 50ep 비대칭, test-set best-epoch selection).
- `NOTION_DIGEST.md` (r2) — [Notion의 주장]/[검증된 사실] 분리. R26 truth: baseline 22모델+4 weak reference, 데이터셋 9종 reference. C1~C4 contribution은 Phase 3 판단 사안. WaDi A2=123 확정.
- `CONFERENCE_PDF_DIGEST.md` (r2) — 학회 발표 34p 전수: 문제 설정(PU), 방법(발표 notation 비계승), baseline 26종, 결론 7 bullet (Phase 3 판단 표시).
- `RESEARCH_SYNTHESIS.md` (r2+) — 전체 종합: R11 3단 프레이밍(설정/상한 구현/희소화 sweep), R10 원재료 표A, 제외 목록, Phase 3 판단 사안 8건, 정본 우선순위.

## 02_venue_study/ — Phase 2 (완료, 게이트 PASS 2026-06-11)

- `VENUE_AND_PAPER_LIST.md` (r2) — 2024–26 탑티어 학회 + Elsevier 저널 관례 + 분석 논문 14편 (TSAD 11편: Anomaly Transformer, DCdetector, NRdetector, CATCH, SARAD, TSB-AD 등). 서지는 Phase 4 재검증 전제.
- `STRUCTURE_AND_FIGURE_PATTERNS.md` (r2) — intro 4단 논증, contribution bullet 3–4, related work 조직법, method 소절 구조 권장안, figure/table 유형 10종 + 배치·크기, 9페이지 분량 배분안 — Phase 3 직접 입력.
- `SENTENCE_CORPUS.md` — 11편 92엔트리(verbatim+출처+패턴 해설) + collocation 7범주 + AI-티 금지 패턴 시드 (Phase 6 기준 corpus; 본문 복사 절대 금지 경고).
- `ANCHOR_SDMAE_DOSSIER.md` (r2) — R21 방어: self-distillation 용어 계보 (Zhang TPAMI 2022 → SDMAE CVPR 2024 → 본 연구), 'coining' 표현 금지 플래그. R9: 유사 12/차이 17 + 위험도 + 포지셔닝 옵션(권장 C: related work distillation 계보 내 1–2문장) + anomaly-map 분기 vs GRL 방어 3축.
- `NRDETECTOR_DOSSIER.md` (r2) — R16: 실험 구성 전모 (e₁ sweep, baseline 3계층, 11지표). R19: related work 내 baseline 모델명 0건 (grep 검증) → 운영 규칙 3조. R20: 차이축 D1–D9 + "거의 없음" 주장의 정밀 스코핑 (표현 학습-PU 통합 기준).

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
- `p1_reconciliation_r1.md` — P1-1↔P1-3 모순 20건 전수 판정표 (1차 소스 기준; Set A/C 오인이 주 원인).
- `p1_271truth_verifier{1,2}_r1.md` — 엄격 구역 2인 검증 (재추적 관점 / 완전성 관점).
- `p1_codebase_synthesis_r1.md`, `p1_digests_r1.md`, `p1_protocol_r1.md` — adversarial 리뷰 r1 3건.
- `p1_271truth_fixlog_r2.md`, `p1_codebase_synthesis_fixlog_r2.md`, `p1_digests_fixlog_r2.md`, `p1_protocol_fixlog_r2.md` — 수정 라운드 처리표 (76건 전건 FIXED).
- `p1_rereview_{alpha,beta}_r2.md` — 수정분 재리뷰 (잔존 BLOCKER 4건 적발).
- `p1_fixlog_r3.md` — 잔존 4건 + MINOR 3건 마감 (supersedes 일부 r2 fixlog 행).
- `p1_coverage_gate_r1.md` — Phase 1 게이트 감사: Directive 18/18 근거 확인 + r3 spot 4/4 VERIFIED.
- `p2_venue_corpus_r1.md`, `p2_dossiers_r1.md` — Phase 2 리뷰 r1 (할루시네이션 0, verbatim 36건 바이트 대조).
- `p2_fixlog_r2.md` — 26건 전수 처리 + C-005 표절 고위험 목록 (Phase 5 입력) + Phase 4 인용 후보 (TSB-AD, Zhang TPAMI 2022).
- `p2_coverage_gate_r1.md` — Phase 2 게이트: 6/6 Directive + spot 7/7 PASS.
