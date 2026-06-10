---
phase: 1
agent: coverage-auditor
directives: [M10]
last_modified: 2026-06-10
round: r1 (Phase 1 게이트 감사)
inputs:
  - paper/MASTER_ORCHESTRATION_PROMPT.md §9 (Directive 원문 + §9.4 매핑표)
  - paper/00_admin/COVERAGE_MATRIX.md
  - paper/01_research_understanding/ 산출물 6종 (전부 r2/r3 수정본)
  - paper/99_reviews/p1_* 리뷰 11건 (r1 리뷰 5 + fixlog 3 + 재리뷰 2 + reconciliation + fixlog_r3)
sources_reverified:
  - "evaluator.py:1986-2021 (compute_detailed_losses — :2017 total_loss = recon + lambda_disc·disc, score-mode 무조건)"
  - "model.py:160-191 (default 분기 :177-186 — LayerNorm→Linear(512→256)→GELU→Dropout(0.1)→Linear(256→1) = Linear 2개; 주석 :178 '1-layer MLP' 실재)"
  - "trainer.py:740-774 (GRL inline λ — :751 게이트, :753 anchor, :760 (‖∇L_main‖/(‖∇L_grl‖+1e-4)).clamp(0,10), :762-763 prev-epoch 적용)"
  - "comparison/baseline_common.py ('epochs' 전수 grep: unsup 10 + '2026-06-06: unsupervised unified to 10' :272-525, weak 50 :333-384, stale 'epochs=50 user override' :256/:266 실재, eval_interval: int = 1 :943, early-stop grep 0건, n_runs=5 random :757)"
  - "scripts/run_base_experiments.py:772/:1908 (compute_detailed_losses 호출처 — :765-790 SNR 입력, :1917 total_loss → CSV); scripts/ablation/run_ablation.py:562+ (compute_loss_statistics — recon/disc/sample_types만 소비, total_loss 무참여)"
  - "compute_adaptive_lambda 호출처 grep 전수 — trainer.py:610 유일 (import :20 외)"
  - "[271c] SWaT A1A2_full/A1A2_excl22 metadata (num_epochs=500, eval_interval=5; full metrics.pak_auc_f1=0.94436, metrics_excl_region22.pak_auc_f1=0.62730; excl22 headline metrics.pak_auc_f1=0.62899)"
  - "[271c] PSM/best_model_detailed.csv 첫 행: 0.0018601296 + 2.0×1.7052759 = 3.412412 (기록값 일치)"
---

# Phase 1 Coverage Gate 감사 (r1) — Directive 충족 근거 + r3 spot 재검증

수행: ① Phase 1 매핑 Directive 18개의 충족 근거(산출물 §섹션) 전수 확인, ② r3 수정 4건(α-B1/B2/B3, RB-1) 코드 직접 spot 재검증, ③ orchestrator 직접 수정 2건(α-m2/α-m3) 확인, ④ 산출물 6종 frontmatter 적합성.
`paper_legacy/` 미접근. 코드·실험 환경 read-only(검증용 읽기만). 쓰기 산출물: 본 파일 1개.

---

## ① Directive별 판정표 (Phase 1 매핑분 18개)

> "제안 근거 문자열"은 COVERAGE_MATRIX "충족 근거" 열 갱신용. 각 Directive의 **Phase 1 요구분**만 판정 (최종 논증·본문 반영은 후속 Phase).

| ID | Phase 1 요구분 | 판정 | 제안 근거 문자열 (파일 §섹션) |
|----|----------------|------|------------------------------|
| T1 | 스크립트·md 문서·Notion 정독 → 연구 완전 이해 | **PASS** | P1: CODEBASE_UNDERSTANDING.md §1–10 (코드 전계층, r3) + NOTION_DIGEST.md §I–IV (두 페이지 완전 정독, r3) + CONFERENCE_PDF_DIGEST.md §①–⑧ (34p 전수, r2) + RESEARCH_SYNTHESIS.md §①–⑨ (종합, r2) |
| R2 | 참고자료(특히 Notion) 비절대 — 판단 유보 장치 | **PASS** | P1: NOTION_DIGEST.md 헤더 R2 경고 + 전 섹션 [Notion의 주장]/[검증된 사실 후보] 등급 분리 (§I-2·§III contribution C1–C4 Phase 3 유보) + CONFERENCE_PDF_DIGEST.md 헤더 [R2]·§⑦ + RESEARCH_SYNTHESIS.md §⑥ Phase 3 판단 사안 목록 |
| R10 | component별 "왜 다변량 시계열인가" 원재료 수집 | **PASS** | P1: RESEARCH_SYNTHESIS.md §③ 표A "R10 원재료" 열 (활성 component 10행 전수, 논리 강도 부족 항목 "보강 필요" 표기) + §⑨ REQUEST-F (warmup ablation 부재 CRITICAL RISK 등재) |
| R11 | semi-supervised/PU 환경 정의 문서화 | **PASS** | P1: RESEARCH_SYNTHESIS.md §②-1~⑥ (설정/구현/sweep 3단 구조 + label 3지점 + 비지도 구현 불가 논리 + PU 관계) + CODEBASE_UNDERSTANDING.md §4.3 (동일 프레이밍 정합) |
| R12 | unsupervised의 label 최선 활용 = 제거 — 사실·구현 확보 | **PASS** | P1: EXPERIMENT_PROTOCOL_TRUTH.md §③ (normalonly 구현 `unified_loader.py:392-485` + Q1/Q3 정의 + "최선 활용" 논리) + RESEARCH_SYNTHESIS.md §④ 비교군 label 정책 |
| R13 | test 반반 분할 main 실험 프로토콜의 진실 확보 | **PASS** | P1: EXPERIMENT_PROTOCOL_TRUTH.md §② (전 데이터셋 `//2` 분할 file:line 전수 + safe-cut 실측 이동량 표 + 시간순 보존 + 논문 서술 재료 4항) + §① 표 (train anomaly 실측 0.52–6.20%) |
| R17 | 271 config만 사용 — 사용/미사용 전수 구분 | **PASS** | P1: 271_CONFIG_TRUTH.md §I–VIII (r3: 37 entity metadata 전수 §II 114키 + §VI used/unused 판정표 + §VII 제외 목록 26항 + §VIII 활성 설정) — verifier 2인 + 재리뷰 α + 본 게이트 spot 재검증 통과 |
| R24 | 내부 용어·변수명 → 정식 명칭 확인 | **PASS** | P1: EXPERIMENT_PROTOCOL_TRUTH.md §④ 매핑표 (내부 키 ↔ 정식 학술 명칭 + 제안 논문, "우측 정식 표기 사용 — R24") + §⑧ REQUEST-2 RESOLVED (pak_auc_pr = pak_auc_prc_auc 확정) |
| R25 | 코드 git 공개 예정 — 사실 인지·기록 | **PASS** | P1: RESEARCH_SYNTHESIS.md §⑦ (R25 원문 FACT + repo 상태 + 공개 전 점검 checklist 4항) |
| R26 | Notion 비교 모델·데이터셋 reference = truth 인지 | **PASS** | P1: NOTION_DIGEST.md §I-10 (truth 범위를 R26 원문대로 한정 — 방법론 인용 5건 제외) + §II-2 [B1]–[B18] + §II-3 [D1]–[D8] [truth 등급 — R26] 명시 (Phase 4 공식 소스 재확인 단서 포함) |
| R28 | SWaT 22번 영역 — 정의·구현·설명 재료 | **PASS** | P1: EXPERIMENT_PROTOCOL_TRUTH.md §⑥ (정의 [2869,38769)·35,900 pts·83.75% 직접 계산 + dual-eval 구현 file:line + 서술 재료 3항) + 271_CONFIG_TRUTH.md §IV (metadata dual-condition + excl22 best-epoch override 주의) + RESEARCH_SYNTHESIS.md §④ excl22 (0.62730/0.62899 이원 수치 주석) |
| R29 | 평가지표 5+1종 — 명칭·관점·상호보완성 재료 | **PASS** | P1: EXPERIMENT_PROTOCOL_TRUTH.md §④ (매핑표 + 상호보완성 논리 재료 + PA F1 문제점 Kim et al. 원문 + 2026-06-10 웹 재검증) |
| R30 | AR threshold — 구현 사실·방어 논리 재료 | **PASS** | P1: EXPERIMENT_PROTOCOL_TRUTH.md §⑤ (구현 `evaluator.py:752-815` + 방어 논리 4항, 4항 문헌 근거는 Phase 4 이관 명시) + §⑧ REQUEST-1 RESOLVED (pa_0_f1_ar 부재 등 전제 사실 확정) |
| R31 | 공정성 방어 — 근거 재료 수집 | **PASS** | P1: EXPERIMENT_PROTOCOL_TRUTH.md §③ "R31 방어 논리 재료" (weak 4종 희소성 + 최선 활용 제공 + 동일 split·평가 단일 원천 + Q1 병행) + RESEARCH_SYNTHESIS.md §②-5 (비지도가 label 3지점을 구조적으로 구현 불가) |
| R32 | 라벨 희소화 sweep — 자산·설계 입력 수집 | **PASS** | P1: EXPERIMENT_PROTOCOL_TRUTH.md §⑦ (전용 sweep 부재 grep 실측 + 재사용 자산 `noisy.py`/`apply_normal50_noise` + placeholder 설계 입력 + 라벨 영향 경로) + RESEARCH_SYNTHESIS.md §②-3·§⑨ REQUEST-C |
| R33 | Simulation·Exathlon 미포함 — 제외 확정·영향 파악 | **PASS** | P1: EXPERIMENT_PROTOCOL_TRUTH.md §① "R33 명시" 절 + §⑧ FEEDBACK-3 (RankAvg 재계산 필수) + RESEARCH_SYNTHESIS.md §⑤ 제외 목록 |
| R34 | Gaussian smoothing 제외 — 사실관계 확정 | **PASS** | P1: 271_CONFIG_TRUTH.md §VI Gaussian 행 + §VII #18 + §IX FEEDBACK (코드 존재하나 271 파이프라인 무참조·전 점수 비평활 — r2 정정 후 정확) + RESEARCH_SYNTHESIS.md §⑤ (단 §⑤ 1행 stale — 잔존 발견 CG-1, 근거 효력은 271_CONFIG_TRUTH로 충족) |
| M8 | Notion MCP 접근 (Phase 1 실사용) | **PASS** | P1: NOTION_DIGEST.md 소스 페이지 절 + 말미 (Page 0 75,820 chars / Page B 108,461 chars 완전 정독) — P0 pre-flight(COVERAGE_MATRIX M8 행)와 합산 |

**18/18 PASS — 근거 부재 Directive 0건.** 모든 Directive가 frontmatter 담당 문서와 본문 섹션 양쪽에서 추적 가능.

---

## ② r3 수정 4건 spot 재검증 (절대 엄격 구역 마감) — 전건 일치

| ID | 문서 반영 확인 | 코드/데이터 직접 재확인 (본 감사, 2026-06-10) | 판정 |
|----|---------------|---------------------------------------------|------|
| α-B1 | 271_CONFIG_TRUTH §VI lambda_disc 행 + §VII #21 — "score-path dead + 진단 CSV 소비" 재서술 반영됨 | `evaluator.py:2017` 실측: `'total_loss': recon_loss + self.config.lambda_disc * disc_loss` — `compute_detailed_losses`(def :1986)에 score-mode 분기 없음 ✓; 호출처 `run_base_experiments.py:772`(SNR 입력)·`:1908`(저장, total_loss는 :1917 CSV 행) ✓; `compute_loss_statistics`(run_ablation.py:562+)는 recon/disc/sample_types만 소비 — total_loss 무참여 ✓; [271c] PSM `best_model_detailed.csv` 첫 행 0.0018601296+2.0×1.7052759=3.412412 기록값 일치 ✓ | **VERIFIED** |
| α-B2 | §VI GRL classifier 행 + §VIII Architecture — "2-layer MLP" 교체 반영됨 | `model.py:177-186` 실측: default 분기 = `LayerNorm(d_model) → Linear(d_model, d_model//2) → GELU → Dropout(0.1) → Linear(hidden, 1)` — **Linear 정확히 2개** ✓; 코드 주석 `:178` "Default: 1-layer MLP with LayerNorm" 실재 (hidden-층 수 기준 표현이라는 문서 부기 정확) ✓ | **VERIFIED** |
| α-B3 | §VIII Loss Components GRL 행 — "(VQGAN-style)" 삭제 → trainer inline grad-ratio 교체 반영됨 | `trainer.py:751-765` 실측: `:760` `(_main_g.norm() / (_grl_g.norm() + 1e-4)).clamp(0.0, 10.0)` + prev-epoch 적용 `:762-763` — 문서 공식·라인 정확 일치 ✓; `compute_adaptive_lambda` 호출처 grep 전수 = `trainer.py:610` 유일 (D 경로, 271 비활성) ✓ | **VERIFIED** |
| RB-1 | EXPERIMENT_PROTOCOL_TRUTH §④-실행 3항 — 패리티 삭제 → 3단 비대칭(500/10/50, eval 5 vs 1) 재작성 반영됨 | [271c] SWaT full/excl22 metadata `num_epochs=500`·`eval_interval=5` ✓; `baseline_common.py` unsup `'epochs': 10` + 주석 "2026-06-06: unsupervised unified to 10" (:272–:525) ✓; weak `'epochs': 50` "weak unified to 50" (:333–:384) ✓; stale "epochs=50 user override" `:256`(docstring)·`:266`(주석) 실재 — 실값 10과 모순(인용 금지 격하 타당) ✓; `eval_interval: int = 1` `:943` ✓; early-stop grep 0건 ✓; (RM-1 부속) `n_runs = 5 if model_name == 'random' else 1` `:757` ✓ | **VERIFIED** |
| α-m1 (부속) | §VI freeze 두 행 부기 ±1 교정 반영됨 | 재리뷰 α 실측(49-55/:55, 75-79/:76) 인용 — 본 감사 별도 재확인 생략 (라인 부기 한정, 재리뷰 α가 실측 완료) | 반영 확인 |

---

## ③ Orchestrator 직접 수정 2건 확인 — 전건 정확

1. **α-m2 — CODEBASE_UNDERSTANDING.md REQUEST 절 RESOLVED 주석 (line ~592)**: "lambda_disc=2.0은 score 경로(adaptive)에서 미사용이나, 진단용 `compute_detailed_losses`(`evaluator.py:2017`)가 mode 무관하게 `recon + 2.0·disc`를 계산해 `best_model_detailed.csv`에 기록 — score·지표·best-epoch 선정에는 무참여 (α-m2 정정 2026-06-10, 271_CONFIG_TRUTH §VI 정합)". → 본 감사 ②-α-B1 직접 재검증과 **전 항목 일치** (특히 "무참여" — `compute_loss_statistics`가 total_loss 키를 사용하지 않음을 코드로 확인). **정확.**
2. **α-m3 — RESEARCH_SYNTHESIS.md §④ excl22 출처 주석**: "0.62730은 `A1A2_full` metadata의 `metrics_excl_region22.pak_auc_f1`(full best epoch 기준); `A1A2_excl22` 자체 headline `metrics.pak_auc_f1`은 0.62899 (best epoch을 excl22_pak_auc_f1로 별도 선정); 두 값 모두 실존 — 논문 표 기준은 Phase 3 결정 (혼용 금지)". → metadata 직접 조회: full `metrics_excl_region22.pak_auc_f1=0.62730` ✓, excl22 `metrics.pak_auc_f1=0.62899` ✓; 별도 선정 서술은 271_CONFIG_TRUTH §IV r2 주석(timing.best_epoch_metric='excl22_pak_auc_f1')과 정합 ✓. **정확.**

---

## ④ 산출물 6종 frontmatter 적합성 — 전건 적합

| 문서 | phase | agent | directives | last_modified | 판정 |
|------|-------|-------|-----------|---------------|------|
| CODEBASE_UNDERSTANDING.md | 1 | research-archaeologist | [T1] | 2026-06-10 (r3) | 적합 |
| NOTION_DIGEST.md | 1 | notion-analyst | [T1, R2, R26, M8] | 2026-06-10 (r3) | 적합 |
| 271_CONFIG_TRUTH.md | 1 | config-forensics | [R17, R28, R34] | 2026-06-10 | 적합 (revision 필드 없음 — r3 이력은 헤더 노트·부록 3에 명기, 실해 없음) |
| EXPERIMENT_PROTOCOL_TRUTH.md | 1 | protocol-truth-writer | [R12, R13, R24, R28, R29, R30, R31, R32, R33] | 2026-06-10 (r3) | 적합 |
| CONFERENCE_PDF_DIGEST.md | 1 | pdf-digest | [T1, R2, R5] | 2026-06-10 (r2) | 적합 (R5는 P3/5/6 매핑이나 notation 비계승 명시는 적절한 선행 준비) |
| RESEARCH_SYNTHESIS.md | 1 | synthesis-writer | [T1, R10, R11, R25] | 2026-06-10 (r2) | 적합 |

frontmatter directives 합집합 = Phase 1 매핑 18개 전부 포섭 (R28은 2개 문서 분담).

---

## ⑤ 잔존 발견 (게이트 비차단 — 차기 patch 라운드 권고)

- **CG-1 (MINOR — RESEARCH_SYNTHESIS.md §⑤ Gaussian smoothing 행)**: "오직 simulation 코드에서만 사용; **codebase 다른 곳에 없음** (271_CONFIG_TRUTH §VI 명시)" — 인용 대상인 271_CONFIG_TRUTH §VI는 r2(V2-B3)에서 정확히 그 반대("코드 존재 — `q3_exploration/core/scoring.py:48-51` gauss() 등, 271 미사용")로 정정되었다. 즉 r2 이전의 허위 부재 진술이 SYNTHESIS에 stale 인용으로 잔존하며, 같은 §⑤의 "MAE 271 B2 (post-hoc sigma=10 Gaussian smoothing)" 행과도 내부 모순. R34의 근거 효력은 정본(271_CONFIG_TRUTH)이 담보하므로 게이트 비차단이나, SYNTHESIS는 authority 2순위 문서이므로 **1행 patch 권고** ("simulation 생성 내부 + q3_exploration 후처리 탐색 스크립트에 존재, 271 파이프라인 무참조"로 교체).
- **CG-2 (정보성)**: COVERAGE_MATRIX M8 행의 P0 pre-flight 문자수(78,956/112,046)와 NOTION_DIGEST 말미 디코딩 문자수(75,820/108,461) 상이 — 별개 fetch 이벤트/디코딩 차이로 추정, 양쪽 모두 "접근 성공" 근거로서는 유효. 실해 없음.
- **CG-3 (정보성, fixlog r3 비고 1 재확인)**: 구 fixlog 2건(p1_271truth_fixlog_r2 V2-B4 행, p1_protocol_fixlog_r2 M-4 행)의 오검증 기록은 미수정 상태이나, p1_fixlog_r3가 supersede를 양 대상 문서 정정 이력과 함께 명기 — 추가 조치 불요.

---

## ⑥ 종합 판정

**Phase 1 게이트: PASS.**

- Directive 충족 근거: **18/18 PASS** (근거 부재 0건; 전 Directive가 산출물 §섹션 단위로 추적 가능).
- 절대 엄격 구역(r3 수정 4건) spot 재검증: **4/4 코드·metadata·산출 CSV 실측 일치** (α-m1 부속 포함 5/5 반영).
- Orchestrator 직접 수정 2건(α-m2/α-m3): **2/2 정확**.
- frontmatter: **6/6 적합**.
- 잔존: MINOR 1건(CG-1 — SYNTHESIS §⑤ Gaussian 행 stale 인용, 1행 patch 권고) + 정보성 2건. R34 충족 자체는 정본 문서로 담보되어 게이트 비차단.

→ COVERAGE_MATRIX의 Phase 1 매핑 18행을 본 표 ①의 근거 문자열로 갱신 가능. CG-1은 Phase 2 dispatch 전 orchestrator 1행 patch 권고.
