---
phase: 1
agent: rereviewer-alpha
directives: [R17, T1, R10, R11]
last_modified: 2026-06-10
review_round: r2 (재리뷰 — 수정 라운드 검증)
inputs:
  - paper/01_research_understanding/271_CONFIG_TRUTH.md (r2)
  - paper/01_research_understanding/CODEBASE_UNDERSTANDING.md (r3)
  - paper/01_research_understanding/RESEARCH_SYNTHESIS.md (r2)
  - paper/99_reviews/p1_271truth_verifier1_r1.md / p1_271truth_verifier2_r1.md / p1_271truth_fixlog_r2.md
  - paper/99_reviews/p1_codebase_synthesis_r1.md / p1_codebase_synthesis_fixlog_r2.md
---

# Phase 1 재리뷰 α (round 2) — 수정본 3종 검증 보고서

## 판정 요약

| 문서 | 판정 | BLOCKER | MAJOR | MINOR |
|------|------|---------|-------|-------|
| `271_CONFIG_TRUTH.md` (r2) | **FAIL** | **3** (α-B1 신규 사실 오류 · α-B2 미해소 r1 발견 · α-B3 3자 모순) | 0 | 1 |
| `CODEBASE_UNDERSTANDING.md` (r3) | **PASS** | 0 | 0 | 1 |
| `RESEARCH_SYNTHESIS.md` (r2) | **PASS** | 0 | 0 | 1 |

- **발견 전수 마감**: fixer-1 23건(V1 9 + V2 14) — fixlog·문서 1:1 대조 **23/23 반영 확인**. 단 그중 1건(V2-B4)이 **신규 사실 오류를 도입** (α-B1). fixer-2 22건 — **22/22 반영 확인**, 신규 오류 0건. 단 r1 리뷰 **MAJ-004가 명시 지목한 271_CONFIG_TRUTH §VIII의 "1-layer MLP" 오기**는 fixer-2가 "담당 외"로 미루고 fixer-1에게 라우팅되지 않아 **미해소 잔존** (α-B2).
- **고위험 수정 재추적**: 271_CONFIG_TRUTH 수정 행 전수(코드 인용 ~60개소 + metadata 8계열)를 1차 소스로 재확인 — α-B1 1건 제외 전부 정확. CODEBASE/SYNTHESIS BLOCKER급 5건(adaptive λ 3경로, leave-one-out batch 확장, focal 식+pos_weight 논거, R11 3단 프레이밍) + MAJOR 표본 4건(MAJ-002/003/008/009) — **전부 코드·metadata와 일치**.
- **3자 정합**: masking 8/42 ✓, GRL 대상·방향(student decoder suppression) ✓, focal 식 ✓. **adaptive λ 서술에서 모순 1건 재발견** (α-B3), classifier 아키텍처 표기 모순 1건 (α-B2). 두 건 모두 오류 측이 271_CONFIG_TRUTH에 있음.
- R17/A3 관련(α-B1·α-B3은 R17 사용/미사용 판정 레이어, α-B2는 §VIII 아키텍처 표) — **waive 불가**.

---

## BLOCKER 상세 (모두 271_CONFIG_TRUTH.md)

### α-B1 — `lambda_disc` "유일 소비처 / 271 dead" — r2 수정(V2-B4)이 도입한 신규 사실 오류

**위치**: §VI "Default / ratio_weighted score modes (`lambda_disc`)" 행 + §VII #21.

**문서 주장**: "`lambda_disc=2.0`의 **유일한 런타임 소비처**는 `compute_default_score`(scoring.py:286-293)이며 … 271에서 **절대 실행되지 않는다** → 271 dead."

**실측 반증** (재리뷰에서 직접 추적):
- `evaluator.py:2017` — `compute_detailed_losses()`가 score-mode **무관하게** `'total_loss': recon_loss + self.config.lambda_disc * disc_loss`를 계산한다 (`evaluator.py:1986` 정의).
- 이 메서드는 271 실행 경로에서 **실제로 호출된다**: `run_base_experiments.py:772` (per-epoch fallback eval — disc_SNR/recon_SNR 산출 입력) 및 `run_base_experiments.py:1908` (최종 저장 — `best_model_detailed.csv`의 `total_loss` 칼럼으로 기록).
- 산출물 실존 확인: 271 entity 디렉토리(예: `PSM/`, `SWaT/A1A2_full/`)에 `best_model_detailed.csv` 존재 → **`lambda_disc=2.0`이 271 런타임에서 실제로 읽혀 저장 아티팩트에 반영되었다.**
- 추가 비게이트 소비처: `visualization/best_model_visualizer.py:1184` (`total = teacher + self.config.lambda_disc * disc`, sample-type 플롯 내부 — mode 분기 없음).
- 단, **anomaly score·전 평가지표에는 무영향** — `compute_loss_statistics`(run_ablation.py:562)는 `total_loss` 키를 사용하지 않으며(recon/disc만 소비, 직접 확인), score dispatch(scoring.py:326-333)가 adaptive로 분기하는 사실, `compute_default_score`/`compute_ratio_weighted_score` 미호출 사실은 모두 맞다.

**판정**: 결론("점수식은 adaptive뿐, `score = recon + 2·disc` 재구성 금지")은 유효하나, "유일 소비처"·"절대 실행되지 않는다"·"dead"는 **허위 코드 구조 진술**이다. V2-B3(Gaussian smoothing "코드베이스에 부재" — 결론 옳고 존재 진술 허위 → BLOCKER) 과 동일 클래스이며, r2에서 새로 도입되었으므로 rubric상 BLOCKER. fixlog의 "① 1차 소스 재확인: 유일 소비처 … 확인" 기록도 grep 누락에 의한 오검증이다.

**요구 조치**: §VI 행·§VII #21을 "**score-path에서 dead** (dispatch scoring.py:326-333; default/ratio_weighted 미호출) — 단 진단 경로 `evaluator.py:2017` `compute_detailed_losses`는 score-mode 무관하게 `recon + 2.0·disc`를 `best_model_detailed.csv`('total_loss' 칼럼)에 기록하며 271에서 실행됨(run_base_experiments.py:772, 1908). 이 칼럼은 점수·지표가 아니다"로 정밀화. (코드 측 정리는 별도 사안 — 문서는 사실대로만.)

### α-B2 — "1-layer MLP" — r1 MAJ-004의 271_CONFIG_TRUTH 부분 미해소 + 3자 모순

**위치**: §VI:295 "GRL classifier (DANN-style, **1-layer MLP**)" + §VIII GRL Details "Architecture | **1-layer MLP**: LayerNorm → Linear(512→256) → GELU → Dropout(0.1) → Linear(256, 1)".

**경위**: `p1_codebase_synthesis_r1.md` MAJ-004가 "271_CONFIG_TRUTH §VIII GRL Details 표에 동일 오류"라고 **명시 지목**했으나, fixer-2는 "본 fixer 담당 외"로 기록만 하고(fixlog MAJ-004 행) fixer-1에게 라우팅되지 않았다 → r1 발견 미해소.

**실측**: `model.py:177-186` — Linear **2개**(512→256, 256→1) + LayerNorm/GELU/Dropout. 코드 주석의 "Default: 1-layer MLP with LayerNorm"은 hidden-층 수 기준 표현. RESEARCH_SYNTHESIS 표A는 r2에서 "**2-layer MLP** 표기 확정 / '1-layer MLP' 표현 금지"로 수정 완료 → 현재 두 문서가 **정면 모순**이며, authority 최상위 문서(271_CONFIG_TRUTH)에 금지 표기가 남아 있다.

**요구 조치**: §VI:295·§VIII:439 두 곳을 "2-layer MLP" (또는 "MLP, hidden 1층 = Linear 2개")로 교체, SYNTHESIS 표A와 동일 표기로 통일.

### α-B3 — GRL adaptive lambda "(VQGAN-style)" — 3자 모순 (adaptive λ, 지정 점검 항목)

**위치**: §VIII Loss Components "GRL classifier loss" 행(:430): "… `grl_loss_weight=0.2`; **adaptive lambda (VQGAN-style)**; …"

**실측**: VQGAN-style 공식(`loss.py:683-728` `compute_adaptive_lambda`, 분자 = ‖∇L_normal‖+‖∇L_anom_forward‖)의 **유일 호출처는 discriminator 경로 `trainer.py:610`** (`use_discriminator=False`인 271에서 비활성) — grep 전수 재확인. GRL λ는 trainer inline `(_main_g.norm()/(_grl_g.norm()+1e-4)).clamp(0,10)` (`trainer.py:760`)로 **별개 공식**이다. CODEBASE §2.6(r3)·SYNTHESIS 표A(r2)는 BLK-001 수정으로 "VQGAN-style을 GRL/FM에 귀속 금지"를 명문화했는데, 정본 1순위인 271_CONFIG_TRUTH §VIII에 그 귀속 표기가 잔존 → **3자 모순** + 같은 문서 §VIII GRL Details의 올바른 공식 행과 **내부 모순**. SYNTHESIS authority 체계상 Phase 3 작성자가 §VIII을 우선 인용하면 BLK-001이 잡은 오류가 그대로 재유입된다.

**요구 조치**: :430의 "(VQGAN-style)" 삭제 → "adaptive lambda (trainer inline grad-ratio, `trainer.py:751-765` — VQGAN-style `compute_adaptive_lambda`와 별개, §VIII GRL Details 참조)"로 교체.

---

## MINOR

- **α-m1 (271_CONFIG_TRUTH)**: §VI freeze 두 행의 부기 라인 off-by-one — init override "trainer.py:49-54"(실측 49-55, 대입문 :55), ValueError guard "74-78"(실측 75-79, `if`문 :76). 본 근거 라인(1141-1142 / 1169-1170)은 정확 — 괄호 부기만 수정.
- **α-m2 (CODEBASE_UNDERSTANDING)**: REQUEST 절의 r1-era RESOLVED 주석 "`lambda_disc` … inference에서 미사용 — 판단 맞음" — α-B1과 같은 이유로 부정확(score 미사용은 맞으나 evaluator 진단 경로가 inference 시 소비). α-B1 정정 시 동일 문구로 동기화 권장.
- **α-m3 (RESEARCH_SYNTHESIS)**: §④ excl22 수치의 출처 이원성 미주석 — 인용된 0.6273은 `A1A2_full` metadata의 `metrics_excl_region22.pak_auc_f1`(full의 best epoch 기준, 실측 0.62730 일치)이고, `A1A2_excl22` entity 자체 headline `metrics.pak_auc_f1`은 **0.62899** (best epoch을 `excl22_pak_auc_f1`로 별도 선정 — 271_CONFIG_TRUTH §IV r2 주석과 정합). 두 값이 모두 실존하므로 어느 쪽이 논문 표 기준인지 1줄 주석 필요 (Phase 3 혼용 위험).

---

## 재추적 검증 기록 (PASS 확인 사항)

### 271_CONFIG_TRUTH r2 수정 행 전수 (α-B1 제외 전부 일치)
- **코드 인용**: trainer.py 485(getattr)·522-524·608/610(D 전용 λ)·639/647/652(FM λ)·746/751/760/762-763(GRL λ)·1141·1169·1201-1210(annealing 구현+미진입)·336-348·160-164(betas/fused)·189-190·1298-1306 / loss.py 259-261(zeroing)·262(elif 선점)·265,272,404(warmup_factor×anomaly_loss_weight — **유일 소비처 grep 재확인, B2 no-op 판정 정확**)·283-287·313·330-340(pos_weight+focal)·436/438·683 / model.py 406/419-423·444/457-461·492-505·514·541·580/624·986(round)·1028-1036/1119-1129·129-144·177-186·530-538·1150-1154 / scoring.py 237(fm_active=False)·239-256·286-304·326-333 / evaluator.py 1716·1737-1745 / dataset_sliding.py 935-998·:956 docstring("271 default: …, clip=True, clamp=None")·1019-1028·:1025(None,None) / run_base_experiments.py 94(EVAL_INTERVAL=5)·287(PSM 220,322)·1804(len(signals))·2584-2585(999.0 유도) / utils/experiment.py 16-39(양수 override→공식) / config.py 21·97·112·226-229·241-243·249·289·315-319 — **전부 실측 일치**.
- **R34 (Gaussian smoothing)**: `q3_exploration/core/scoring.py:48-51` `def gauss` 실재 ✓, `core/postprocess.py:51/:129` ✓, `experiments/exp_P14_boundary_refinement.py:147` `gauss(base_unsmoothed, 10)` ✓; evaluator/scoring/trainer/visualization/run_base_experiments에서 `q3_exploration` grep **0건** ✓ — "존재하나 271 미사용" 등재 형식 정확.
- **합성 전용 판정 (B1/B6)**: `sliding_window_total_length`·`anomaly_interval_scale` 소비처 grep 전수 — run_ablation/visualization 한정, run_base_experiments 무참조 ✓.
- **metadata**: 37 파일 / 공통 114키 byte-일치 / 가변 3키 ✓; SWaT excl22 `timing.best_epoch_metric='excl22_pak_auc_f1'` vs config `'pak_auc_f1'`, full timing=`'pak_auc_f1'`, wall_time 동일(4636.57s) ✓; `grl_pos_weight` min 3.1410(SMAP/T-1)·max 999.0(SMD/m-1-5)·SWaT 59.1814 ✓; train_ratio G-7 0.6167("0.617–0.626" 표기 정확)·PSM 0.8007·SWaT 0.7619·WaDi 0.9375/0.9098·SMD 0.74998–0.75 ✓; num_features MSL 55/SMAP 25/PSM 25/SWaT 45/WaDi 123/SMD 29–36 ✓.

### CODEBASE/SYNTHESIS BLOCKER급 5건 + MAJOR 표본 4건
1. **adaptive λ 3경로 분리 (BLK-001/003)**: `compute_adaptive_lambda` 유일 호출처 trainer.py:610 (D 경로, :608 게이트) ✓; GRL inline :760, FM inline :647, w = `student_decoder.parameters()[-1]` / D는 `student_output_projection.weight`(:609) ✓; prev-epoch 적용(:652, :762-763, 초기값 :189-190, epoch 말 갱신 :1297-1306) ✓.
2. **leave-one-out batch 확장 (BLK-002)**: def :1647, docstring :1650 ✓; `sequences.unsqueeze(1).expand(...)` :1807-1808, forward :1818 ✓; pb=2 HARD-LOCK "does NOT affect numerical results" :1703-1717 ✓.
3. **focal 식·pos_weight 논거 (BLK-004)**: `_p_t=exp(−_bce); _focal=(1−_p_t)²×_bce` (loss.py:337-340) ✓; pos_weight 내장(:330-336, `grl_balanced_sampling=False` else 분기) ✓; **수학 재검증**: pos_weight 부재 시 y=1→exp(−BCE)=p, y=0→1−p = 정확히 p_t 성립, pos_weight w 내장 시 positive에서 p^w ≠ p_t — fixer-2의 정밀화 논거 **수학적으로 정확**; SWaT 59.1814 metadata ✓.
4. **R11 3단 프레이밍 (BLK-005/MAJ-006)**: label 3지점(model.py:975-1002 / loss.py:244-261 / loss.py:282-350) ✓; train anomaly 실측 SWaT 1.63%·WaDi A1 0.52%·A2 0.76%·PSM 6.20%·SMAP 0.70%·MSL 1.70% — EXPERIMENT_PROTOCOL_TRUTH §① 표와 일치, "0.52–6.20%" 범위 정확 ✓; ②-5의 Q1/Q3 비교군 정책 서술은 §④ FACT와 정합 ✓.
5. **MAJOR 표본**: MAJ-002 coverage(비중첩 50×10, 500/49≈10.2, sentinel 조건 utils/experiment.py:35-39, 서로소 다양성 docstring) ✓ / MAJ-003 threshold(`find_f1_optimal_idx` :215-226, roc_curve :928, strict `>` :931) ✓ / MAJ-008 (PA_K_VALUES step5 :831, 사용처 :854/:950/:2111/:2139; AUC 적분 `np.arange(0,101)` :1034) ✓ / MAJ-009 (83.75% 파생 재계산: 35,900/(0.190541×224,960)=0.83753 ✓; 최종값 0.94436/0.62730 metadata 일치 — stale 아님 ✓; docstring "~84%" evaluator.py:2299-2311 ✓).
- 기타 신규 주장: MAJ-007 37 entity(2+2+1+22+5+5) 디렉토리 실측 ✓; MIN-001 시작 LR 1e-7(start_factor=1e-4, trainer.py:170-174) ✓; MIN-002 단일 `decoder_pos_encoder`(:344)+`register_buffer`(:275) 비학습 buffer ✓; MIN-003 패치별 독립 적용(:1153-1154 squeeze/transpose) ✓; NOTE-004 safe-cut(`_load_smap_msl_simple_single(..., safe_cut_margin=10)` loaders.py:2527, `_find_safe_cut_point` :1050, 적용 :2591-2596) ✓.

### 3자 정합 (지정 4항목)
| 항목 | 결과 |
|------|------|
| adaptive λ 서술 | **모순 1건** — α-B3 (271_CONFIG_TRUTH §VIII:430 "VQGAN-style" 잔존; CODEBASE·SYNTHESIS는 정확) |
| masking 8/42 | 3문서 일치 (`round(50×0.15)=8` / visible 42) ✓ |
| GRL 대상·방향 | 3문서 일치 — student decoder, anomaly-identity suppression, encoder는 detach 차단 ✓ |
| classifier 아키텍처 | **모순 1건** — α-B2 (271_CONFIG_TRUTH "1-layer" vs SYNTHESIS "2-layer 확정·1-layer 금지") |

---

## 요구 조치 요약 (재수정 라운드 입력)

1. [α-B1] 271_CONFIG_TRUTH §VI lambda_disc 행 + §VII #21 — "유일 소비처/dead" → "score-path dead + 진단 CSV 소비처(evaluator.py:2017, run_base:772/1908) 명시"로 교체.
2. [α-B2] 271_CONFIG_TRUTH §VI:295·§VIII:439 — "1-layer MLP" → "2-layer MLP" (SYNTHESIS 표기와 통일).
3. [α-B3] 271_CONFIG_TRUTH §VIII:430 — "(VQGAN-style)" 삭제, inline grad-ratio로 교체.
4. [α-m1~m3] 부기 라인 ±1 교정 / CODEBASE RESOLVED 주석 동기화 / SYNTHESIS excl22 이원 수치 주석.

`paper_legacy/` 미접근. 코드·실험 환경 read-only (검증용 읽기 전용 명령만 실행). 쓰기 산출물: 본 파일 1개.
