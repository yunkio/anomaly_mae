---
phase: 0
agent: orchestrator
directives: [M10, A10]
last_modified: 2026-06-10
---

# Coverage Matrix — Directive 추적 (총 57행: T 7 + R 37 + M 13)

> 정본 출처: `paper/MASTER_ORCHESTRATION_PROMPT.md` §9 + `paper/ORIGINAL_USER_DIRECTIVES.md`.
> 상태: PENDING / IN_PROGRESS / DONE. 상태 전이는 반드시 "충족 근거" 포인터(파일+섹션)와 함께.
> 한 Directive가 여러 Phase에 걸치면 Phase별 부분 충족을 "근거" 열에 누적 기록, 전부 충족 시 DONE.
> Phase 8 최종 게이트: 57행 전부 DONE + 근거 유효성 재검증.

## 작업 지시 (T1–T7)

| ID | 요약 | 담당 Phase | 상태 | 충족 근거 |
|----|------|-----------|------|----------|
| T1 | 프로젝트 모든 스크립트·문서(md)·Notion 페이지 정독, 연구 완전 이해 | 1 | DONE | P1: CODEBASE_UNDERSTANDING §1–10 + NOTION_DIGEST §I–IV + CONFERENCE_PDF_DIGEST §①–⑧ + RESEARCH_SYNTHESIS §①–⑨ (r2/r3; 게이트 `99_reviews/p1_coverage_gate_r1.md` PASS) |
| T2 | 최근 3년 탑티어 AI 학회 리스트업 + 고평가 논문의 논리 흐름·구성·figure/table 패턴 파악 (시계열 이상탐지 필수 포함) | 2 | DONE | P2: VENUE_AND_PAPER_LIST §1·§3 (2024–26 학회 + 14편, TSAD 11편) + STRUCTURE_AND_FIGURE_PATTERNS §A–G + SENTENCE_CORPUS §1–10 (게이트 `99_reviews/p2_coverage_gate_r1.md` PASS) |
| T3 | 논문 전체 개요·틀: 전체 구성 → 섹션 구성 → 섹션별 내용 → 필요 근거 | 3 | PENDING | |
| T4 | reference 탐색(고퀄리티) + 원문 발췌·활용 맥락 정리 + 서지 철저 검증(할루시네이션 절대 0) + IEEE 스타일 정리 | 4 (+5 보강) | PENDING | |
| T5 | 영어 본문 완성 — 완벽하게 완성된 형태, figure/table은 어디에 어떤 형태로 삽입될지 포함 + [placeholder]로 두되 들어갈 내용 서술, 표절 절대 금지(여러 번 체크), T4 reference로 뒷받침 | 5 | PENDING | |
| T6 | 분야 학술 문체 검증 — AI 티·비관용 표현을 문장 단위로 철저 검사 | 6 | PENDING | |
| T7 | Elsevier 템플릿 준수 LaTeX(Overleaf) 조판 + placeholder 배치 + PDF 변환·직접 확인 루프 | 7 | PENDING | |

## 참고사항 (R1–R37)

| ID | 요약 | 담당 Phase | 상태 | 충족 근거 |
|----|------|-----------|------|----------|
| R1 | Related work·contribution·실험 MECE 구성 | 3, 5, 8 | PENDING | |
| R2 | 참고자료(특히 Notion) 논리·서술은 참고만 — 충분한 판단 후 활용 (contribution 구조 채택 여부 선판단) | 1, 3 | IN_PROGRESS | P1: NOTION_DIGEST 헤더 R2 경고 + [주장]/[사실] 등급 분리 전 섹션 + CONFERENCE_PDF_DIGEST 헤더·§⑦ + RESEARCH_SYNTHESIS §⑥ (Phase 3 판단 유보) |
| R3 | figure/실험 placeholder 틀 + '실험 잘 되었다' 가정 서술 + Notion 하위 페이지에 placeholder별 구체적 한국어 명세 + 실험 데이터 부족 지적 금지(한계 아님) + 캡션·설명은 placeholder여도 완성형 | 5, 7, 8 | PENDING | |
| R4 | 'AI가 작성한 티'·도메인 비관용·논문 비관용 표현 엄격 검증 | 5, 6 | PENDING | |
| R5 | notation 오류 없이 + 최대한 일반적·이해 쉬운 방식 (참고자료는 참고만) | 3, 5, 6 | PENDING | |
| R6 | 분량: appendix·reference 제외, table/figure 합쳐 9page. 크기 넉넉히 가정 | 3, 7 | PENDING | |
| R7 | Appendix 구성 주의 | 3, 7 | PENDING | |
| R8 | Contribution 강조 핵심 — novelty 충분 탐색·충분 강조 | 3, 5 | PENDING | |
| R9 | SDMAE는 핵심 인용이되 과도 유사 인상 금지 — 차이점 나열 방식 금지, 자연스럽게 언급하고 넘어가기 | 2, 3, 5 | IN_PROGRESS | P2: ANCHOR_SDMAE_DOSSIER §4 (유사12/차이17 + 위험도) + §7 포지셔닝 옵션 (권장 C) + §7-2 방어 3축 + §8 |
| R10 | 각 component마다 "왜 다변량 시계열에서 이렇게 해야만 하는가" 치열하게 고민·반영 | 1, 3, 5 | IN_PROGRESS | P1: RESEARCH_SYNTHESIS §③ 표A "R10 원재료" 열 + §⑨ REQUEST-F |
| R11 | semi-supervised/PU 환경 집중 — 대부분 unlabeled + 소수 핵심 이상 label, 기존 unsupervised는 labeled 활용 불가가 핵심 | 1, 3, 5 | IN_PROGRESS | P1: RESEARCH_SYNTHESIS §②-1~⑥ (3단 프레이밍) + CODEBASE_UNDERSTANDING §4.3 |
| R12 | unsupervised 비교군의 label 활용 최선 = 학습 데이터에서 알려진 이상 제거 (순도 높은 정상 학습) | 1, 5 | IN_PROGRESS | P1: EXPERIMENT_PROTOCOL_TRUTH §③ (normalonly 구현 file:line) + RESEARCH_SYNTHESIS §④ |
| R13 | main 실험: 기존 벤치마크는 train에 anomaly 부재가 대부분 → test에 포함된 anomaly를 학습 단계에 반영하기 위해 test를 길이 기준 반반 분할, 앞 50% train 포함. 이때 기존 unsupervised는 알려진 이상 제거로 순도 높은 정상 학습 데이터 구성(이상 포함 시 성능 하락). 시간적으로 뒤쪽 데이터를 test로 사용 + 공정성 위해 취사선택 없이 전 데이터셋 통일 적용 강조 | 1, 5 | IN_PROGRESS | P1: EXPERIMENT_PROTOCOL_TRUTH §② (//2 전수 라인 + safe-cut 실측) + §① train anomaly 실측 |
| R14 | 중간 산출물 철저 구조화 + 쉽게 찾을 수 있는 index 유지 | 0, 8 (전 Phase 상시) | IN_PROGRESS | P0: §4 워크스페이스 골격 + `00_admin/INDEX.md` 가동 (frontmatter 규칙 포함) |
| R15 | 불필요한 신규 축약어 금지 — 단 제목·모델명·모델 축약어는 novelty 부각 방향 | 3, 5, 6 | PENDING | |
| R16 | NRdetector의 실험 구성·논리 참고 (거의 유일한 시계열 semi-supervised) | 2, 3, 5 | IN_PROGRESS | P2: NRDETECTOR_DOSSIER §1–3 (2-stage PU 구조·정당화 논리·split·라벨 sweep·baseline 3계층·11지표) |
| R17 | 271 config만 사용 — 미사용 option(예: dynamic margin) 전부 무시, metadata+코드 추적으로 사용/미사용 명확 구분 | 1, 5 | IN_PROGRESS | P1: 271_CONFIG_TRUTH §I–VIII (r3; metadata 37 전수 + verifier 2인 + 재리뷰 α + 게이트 spot 4/4) |
| R18 | 완료 후 "정말 출판된 level의 완성 논문인가" 점검 (placeholder 허용) | 8 | PENDING | |
| R19 | baseline 전부 related work 언급 불필요 — 단순 비교 모델은 실험 섹션 인용으로 충분(NRdetector 논문 참고). 핵심 계승 요소가 있거나 직접 비교 대상이 되는 경우에만 설명 | 2(근거 수집), 3, 5 | IN_PROGRESS | P2: NRDETECTOR_DOSSIER §4 (related work 내 baseline 0건 grep 검증 + §4.3 운영 규칙 3조) |
| R20 | SSL/PU related work: 기존 방법론 목표 언급 + 시계열 부재 강조, NRdetector는 차이점 위주 | 2(준비), 3, 5 | IN_PROGRESS | P2: NRDETECTOR_DOSSIER §5 (차이축 D1–D9 + "거의 없음" 정밀 스코핑 + 차이-중심 전략) |
| R21 | self-distillation 용어 — SDMAE 선례를 방어논리로 (해당 논문의 명명 이유 확인) | 2, 3, 5 | IN_PROGRESS | P2: ANCHOR_SDMAE_DOSSIER §3.5·§5.1 (용어 계보 Zhang TPAMI 2022 → SDMAE → 본 연구; coining 금지 플래그 §9) |
| R22 | patch/masking 계보는 오직 vision MAE — 시계열 patch 연구와 계승 혼동 금지 | 3, 5 | PENDING | |
| R23 | hyperparameter 구체값은 꼭 필요한 것만, 주로 일반적 서술 | 5 | PENDING | |
| R24 | 연구 과정 내부 용어·변수명 그대로 사용 금지 — 출판 수준 표현 | 1(명칭 확인), 5, 6 | IN_PROGRESS | P1: EXPERIMENT_PROTOCOL_TRUTH §④ 정식 명칭 매핑표 + §⑧ REQUEST-2 RESOLVED |
| R25 | 코드 git 공개 예정 — 논문에 자연스러우면 넣고 아니면 생략 | 1, 5 | IN_PROGRESS | P1: RESEARCH_SYNTHESIS §⑦ (git 공개 예정 기록) |
| R26 | Notion 비교 실험 페이지의 비교 모델·데이터셋 reference는 엄격 검증된 truth로 활용 가능 | 1, 4 | IN_PROGRESS | P1: NOTION_DIGEST §I-10·II-2·II-3 [truth 등급 — R26] (Phase 4 공식 소스 재확인 단서 포함) |
| R27 | 구현 방식 과도하게 하나하나 나열 금지 — 필요·핵심 정보만 | 5 | PENDING | |
| R28 | SWaT '22번 이상 영역' 지배적 거대 → 제외 지표 별도 제시 + 충분한 설명 | 1, 5 | IN_PROGRESS | P1: EXPERIMENT_PROTOCOL_TRUTH §⑥ (region22 bit-exact 재현) + 271_CONFIG_TRUTH §IV + RESEARCH_SYNTHESIS §④ |
| R29 | 평가지표 vus_roc/vus_pr/pak_auc_f1/pak_auc_pr/affiliated-f1 — 각 지표가 평가하는 다른 관점·상호보완성 설명 + 전 지표에서 뛰어난 성능 강조 + PA-F1은 전체 관점에서 좋아 제시하되 challenge·문제점을 지적하여 **주 지표로 참고하지 않을 것임을 본문에 명시** | 1, 5 | IN_PROGRESS | P1: EXPERIMENT_PROTOCOL_TRUTH §④ (5지표 관점·상호보완성·PA-F1 문제점 + 웹 재검증) |
| R30 | threshold = test anomaly 비율 — threshold 무관 지표 병행 + 평가 protocol일 뿐임을 설득력 있게 (cherry-picking 인상 금지) | 1, 5 | IN_PROGRESS | P1: EXPERIMENT_PROTOCOL_TRUTH §⑤ (AR threshold file:line) + §⑧ REQUEST-1 RESOLVED |
| R31 | 공정성 방어: label 활용 가능한 기존 시계열 모델 부족 + unsupervised의 label 활용 방식 제공 — 설득력 필수 | 1, 5 | IN_PROGRESS | P1: EXPERIMENT_PROTOCOL_TRUTH §③ 방어 논리 재료 + RESEARCH_SYNTHESIS §②-5 |
| R32 | 라벨 희소화 sweep 실험 포함 + unlabeled anomaly 혼입 시 강건한 이유의 논리적 설명 | 1, 3, 5 | IN_PROGRESS | P1: EXPERIMENT_PROTOCOL_TRUTH §⑦ (미구현 사실 + placeholder 설계 입력) + RESEARCH_SYNTHESIS §②-3·§⑨ REQUEST-C |
| R33 | Simulation·Exathlon 데이터셋 논문 미포함 | 1, 5 | IN_PROGRESS | P1: EXPERIMENT_PROTOCOL_TRUTH §① R33 절 + RESEARCH_SYNTHESIS §⑤ 제외 목록 |
| R34 | Gaussian smoothing 내용 제외 (사용 안 함) | 1, 5 | IN_PROGRESS | P1: 271_CONFIG_TRUTH §VI·§VII#18·§IX (gauss() 실재·271 무참조) + RESEARCH_SYNTHESIS §⑤ (CG-1 패치 완료) |
| R35 | 너무 지엽적인 것 생략 | 5, 6 | PENDING | |
| R36 | 뒷받침 근거 필요한데 인용 없는 부분 → 인용 가능 논문 찾아 reference 부착 | 4, 5 | PENDING | |
| R37 | ./paper_legacy 작업물 절대 참고 금지 | 0 (전 Phase 상시) | IN_PROGRESS | P0: 전 dispatch 프롬프트에 금지 명문화 (감사 A/B/r2 프롬프트 확인 가능) + `AGENT_ROSTER.md` 공통 규약 ③ |

## 메타 지시 (M1–M13)

| ID | 요약 | 담당 Phase | 상태 | 충족 근거 |
|----|------|-----------|------|----------|
| M1 | orchestrator: 전체 계획 수립·관리 + sub-agent 정의·배정·관리·팀 작업 | 0 (전 Phase 상시) | IN_PROGRESS | P0: `AGENT_ROSTER.md` 확정 + `TASK_BOARD.md` Phase 1–8 계획 등재 + 감사 dispatch 운영 |
| M2 | 작업별 리뷰 전문 sub-agent 피드백 루프 = 성공의 핵심 | 0 (전 Phase 상시) | IN_PROGRESS | P0: §5.3 루프 실증 — 감사 r1(MAJOR 1) → 수정 → r2(PASS), `99_reviews/p0_*` 3건 |
| M3 | agent 간 작업 요청/피드백 가능 — orchestrator가 중간 조율 | 0 (전 Phase 상시) | IN_PROGRESS | P0: `REQUESTS_AND_FEEDBACK.md` 라우팅 테이블 가동 + 전 dispatch에 REQUEST:/FEEDBACK: 규약 포함 |
| M4 | 퀄리티 최우선 — 시간·토큰 효율 고려 금지, 모든 퀄리티 테크닉 동원 | 0 (전 Phase 상시) | IN_PROGRESS | P0: 독립 감사 2인 병렬 + 수정분 재리뷰 라운드 운영 |
| M5 | 한꺼번에 과다 지시 금지 — phase 분할, 차례대로 | 0 (전 Phase 상시) | IN_PROGRESS | P0: Phase 0–8 + 태스크 단위 분할 (`TASK_BOARD.md`), dispatch당 단일 역할 원칙 (`AGENT_ROSTER.md` 규약) |
| M6 | phase별 절대 엄격 파트 존재 (reference 표기, 표절 방지 등) | 0 (전 Phase 상시) | IN_PROGRESS | P0: 엄격 구역 명세 (`TASK_BOARD.md` Phase 계획 열 + `AGENT_ROSTER.md` 강화 프로토콜 행) |
| M7 | ./paper/ 디렉토리에서 작업 — 계획부터 결과까지 철저 구조화 | 0 (전 Phase 상시) | IN_PROGRESS | P0: §4 구조 생성 완료, 전 산출물 frontmatter 규칙 적용 |
| M8 | Notion 페이지 MCP 접근 | 0, 1 (전 Phase 상시) | IN_PROGRESS | P0: pre-flight (b) fetch 성공 + P1: NOTION_DIGEST 완전 정독 (75,820/108,461자 본문 + 하위 페이지) |
| M9 | 매 phase 내용 정리 보고 + 필요·요청사항 전달 | 매 Phase 종료 시 (0–8 전부; §7의 P0/P8 명시 배정 포섭) | IN_PROGRESS | P0: `PHASE_REPORTS/phase0_report.md` + 채팅 보고 |
| M10 | 지시·참고사항의 어떤 문장·단어도 누락 금지 — 전부 핵심 | 0, 8 (전 게이트) | IN_PROGRESS | P0: Matrix 57행 기계 대조 + 독립 감사 A(원문 충실성 PASS)·B(완전성 r2 PASS) — `99_reviews/p0_registry_fidelity_A_r1.md`, `p0_matrix_completeness_B_r1.md`, `p0_matrix_completeness_B_r2.md` |
| M11 | 지시 나열 순서가 아닌 효과적 process/phase로 재구성하되 전 내용 포함 + 각 phase 프롬프트 품질 (E-002) | 0 | IN_PROGRESS | P0: 마스터 §7 process 채택·가동 + Phase 0 구축. dispatch 프롬프트 품질은 전 Phase 지속 의무 (ERRATA E-002) |
| M12 | 입력 자료: 방법론 Notion + 비교 실험 Notion + 학회 발표 PDF | 0 | DONE | P0: 접근 확인 + P1: 전량 소화 (NOTION_DIGEST + CONFERENCE_PDF_DIGEST 34p 전수) |
| M13 | 단일 프롬프트로 받아 누락·생략·유실 없이 Phase 0→8 자율 완주 | 전 Phase 상시, 8 | IN_PROGRESS | P0: 마스터 프롬프트 전체 정독 + Phase 0 완료 + Phase 1 자율 진행 개시 |

---

## 기계적 행 수 대조 (Phase 0, §6.2)

- §9.1 전사: T1, T2, T3, T4, T5, T6, T7 → **7행** ✓
- §9.2 전사: R1–R37 순차 (R1, R2, …, R37 결번 없음) → **37행** ✓
- §9.3 전사: M1–M13 순차 (결번 없음) → **13행** ✓
- **합계 57행** = 문서 명시 57 ✓ (2026-06-10 orchestrator 대조; 독립 coverage-auditor 교차 검증은 별도 수행)
