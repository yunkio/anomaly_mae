---
phase: 0
agent: orchestrator
directives: [R14]
last_modified: 2026-06-11
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

## 03_blueprint/ — Phase 3 (완료, 게이트 PASS 2026-06-11)

- `PAPER_BLUEPRINT.md` (r3) — **Phase 4·5의 1차 입력.** 전체 섹션 구조(Elsevier: abstract/keywords/highlights 포함), contribution 4-bullet (D-005①), setting="contaminated semi-supervised"(D-005②), R10 논증 배치 전수표(§12), related work 전략(§4 — R9 옵션 C·R19 클러스터 인용·R20 스코핑·R22 계보), 프로토콜 방어 5논거(§14)+reject 시나리오(§15), Table 4 protocol-effect 보조분석, ablation 계획(§6.7), 모델명·제목 확정(§10=D-007: CSMAD), 인용 수요 목록(Phase 4 입력).
- `PAGE_BUDGET.md` (r3) — **분량 단일 정본**: 1.6/1.1/2.7/3.3/0.3=9.0p + figure/table 크기 가정 + 단어수 환산 + 초과 시 fallback 사다리.

## 04_references/ — Phase 4 (완료, 게이트 PASS 2026-06-11)

- `refs.bib` — **인용 정본** (49항목, 전부 공식 BibTeX export 기반, bibtexparser 49/49 검증). 수정 이력은 항목 코멘트 + P4_DIFF_REPORT.
- `CLAIM_CITATION_MAP.md` (r3) — claim C-001~C-085 ↔ reference 매핑 (VERIFIED 78행). Phase 5 인용 배치의 정본.
- `VERIFICATION_LEDGER.md` — 통합 마스터 (49행, 2채널 판정) + 상세 4분할 ledger (A1/A2/B1/B2) + `P4_DIFF_REPORT.md` (diff·해소·정오 부기).
- `REFERENCES_IEEE.md` — IEEE 잠정 정리본 (최종 스타일은 Phase 7 elsarticle).
- `REFERENCE_LIBRARY_INDEX.md` — card 49 색인 (등급/검증/커버 claim/활용 위치).
- `library/` — reference card 49 (FULL 22: verbatim 발췌+활용 맥락 / LIGHT 27: 서지+abstract+역할). EXCERPT_UNVERIFIED 잔존 3 (zhang2022selfdistill·xu2018kpivae·ruff2020deepsad — 서지 인용 가능, verbatim 금지).
- `SCOUT_CANDIDATE_LIST.md`, `VERIFIER_B_SEED.md` — 중간 산출물 (탐색·blind seed).

## 05_manuscript/ — Phase 5 (완료, 게이트 PASS 2026-06-11)

- `MANUSCRIPT_v2.md` — **본문 정본** (Title/Abstract/Keywords/Highlights + §1–5 + Appendix A/B/C; 영어 완성본, placeholder 49종). v1·v2_draft는 이력 보존.
- `PLACEHOLDER_REGISTRY.md` (v2-r3) — placeholder 전수 (NUM 31, TXT 2, FIG 5, TAB 11, ALG 1; ID/위치/완성 캡션/내용 명세/크기) — Phase 7 배치·Phase 8 Notion 명세의 정본.
- `INTEGRATION_REPORT_v1.md`, `SURGERY_REPORT_v2.md` — 통합·분량 수술 기록 (분량 추정 10.42p, Phase 7 실측 판정 인계 — D-010 ⑤).
- `sections/` — 섹션별 초안 (이력).

## 06_style_audit/ — Phase 6 (완료, 게이트 PASS 2026-06-11)

- `AI_PHRASING_LEDGER.md` — corpus 기반 금지 패턴 + 검출 52건 (em-dash 패턴 등). `STYLE_AUDIT_A.md` (영어 산문 품질 88건) / `STYLE_AUDIT_B.md` (분야 관용 67건) / `TERMINOLOGY_AUDIT.md` (Q1/Q3 판정·약어 인벤토리·notation).
- 결과물은 `05_manuscript/MANUSCRIPT_v3.md` — **현 본문 정본** (문체 패스 + 회귀 검사 통과; MINOR 4건 Phase 7 polish 이월 D-011).

## 07_latex/ — Phase 7 (완료, 게이트 PASS 5/5 2026-06-11)

- `overleaf_package.zip` — **최종 인도물** (12파일: main.tex + sections 8 + refs.bib + elsarticle-num.bst + elsarticle.cls; 단독 컴파일 검증 2회 PASS). 그대로 Overleaf 업로드 가능.
- `main.tex` + `sections/*.tex` — **본문 정본 (.tex)** (MANUSCRIPT_v3.md는 동결). preprint,12pt 배포 모드.
- `main.pdf` (46p) — 검수용 PDF. `main_{3p,5p}_measure.*` — 분량 측정 빌드 (5p 본문 8.997p — D-012 기준 판형; zip 미포함).
- `TEMPLATE_REQUIREMENTS.md` (v2), `PROSE_DIFF_LOG.md` (산문 변경 통제 전 기록 — D-011 7건 + D-013 압축 + 측정 무결성 §5.6), `pdf_qa/` (QA r1 + FIX r2).

## 08_final_audit/ — Phase 8 (완료 2026-06-11)

- `FINAL_AUDIT_reviewer1.md` / `FINAL_AUDIT_reviewer2.md` — 신규 리뷰어 2인 모의 피어리뷰 (학회 양식; placeholder-비본질 약점은 D-014 triage로 처리).
- `NOTION_PLACEHOLDER_SPECS.md` (r2, 검수 통과) — placeholder 전수(FIG 5/TAB 12/ALG 1/NUM 31/TXT)의 한국어 실험·시각화 명세 + 신규 실행 11건 우선순위 + 재사용 판정 + R-PROBE 권고. **발행본**: https://www.notion.so/37c87856b207810e83e3d1b5f14766fc (비교 실험 페이지 하위, 렌더링 검증 완료).

## 최종 인도물 요약 ("무엇을 어디서 찾는가")

- **Overleaf 업로드용 zip**: `paper/07_latex/overleaf_package.zip` (압축 해제만으로 단독 컴파일 — 검증 3회)
- **검수용 PDF**: `paper/07_latex/main.pdf` (46p; 본문 8.997p @5p 판형)
- **placeholder를 채우는 법**: Notion 명세 페이지 (위 URL) 또는 `08_final_audit/NOTION_PLACEHOLDER_SPECS.md`
- **연구 사실의 정본**: `01_research_understanding/271_CONFIG_TRUTH.md` (r4)
- **인용 정본**: `04_references/refs.bib` + `VERIFICATION_LEDGER.md`
- **결정 이력**: `00_admin/DECISION_LOG.md` (D-001~D-014)

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
