---
phase: 1
agent: fixer-5 (r3 잔존 BLOCKER 일괄 수정)
last_modified: 2026-06-10
round: r3
inputs:
  - paper/99_reviews/p1_rereview_alpha_r2.md (α-B1/B2/B3 + α-m1)
  - paper/99_reviews/p1_rereview_beta_r2.md (RB-1 + RM-1 + NMr-1)
targets_modified:
  - paper/01_research_understanding/271_CONFIG_TRUTH.md (r3)
  - paper/01_research_understanding/EXPERIMENT_PROTOCOL_TRUTH.md (r3)
  - paper/01_research_understanding/NOTION_DIGEST.md (r3 — NMr-1 1건; 발견의 물리적 위치가 이 문서라 지시 목록 ②를 여기서 수정. "대상 문서 2개" 집계 외 3번째 파일임을 명기)
sources_reverified:
  - "evaluator.py:1986/2017, run_base_experiments.py:772/856/1908, scoring.py:286-333, run_ablation.py:562-640(741/1299는 비-271), best_model_visualizer.py:1184, [271c] PSM/SWaT best_model_detailed.csv 실존+첫 행 수치 재계산"
  - "model.py:160-195 (default classifier else 분기 :177-186, Linear 2개), trainer.py:740-775 (GRL inline :746/:751/:760/:762), loss.py compute_adaptive_lambda 유일 호출처 trainer.py:610 (grep 전수)"
  - "trainer.py:45-58 (init override 49-55, 대입 :55), :72-82 (guard 75-79, if :76)"
  - "[271c] metadata PSM/SWaT full/WaDi A2: num_epochs=500, eval_interval=5 (json 직접 조회)"
  - "baseline_common.py epochs 전수 grep: unsup 10 (:272,279,286,297,300-302,308,314,323,397,405,412,429,452,494,525 — '2026-06-06: unsupervised unified to 10'), weak 50 (:333,337,355,367,384), stale ':256/:266 epochs=50 user override' 실재, eval_interval: int = 1 (:943 등), early stopping grep 0건, random n_runs=5 (:757) + mean/std 집계 (:786-796)"
  - "Page 0 덤프 디코딩(78,913 chars) 후 '127' 전수 regex: §1.2 표(@3013)·d_model 매핑(@13385)·num_features 표(@36627)·§5.2.1(@68794) = 4개소"
---

# Phase 1 fixer-5 처리표 (r3) — 재리뷰 r2 잔존 발견 일괄 수정

모든 건: 1차 소스 재확인 → 수정 → frontmatter/정정 이력 갱신 완료. `paper_legacy/` 미접근, 코드·실험 환경 read-only(검증용 읽기만).

## 처리 요약

| ID | 등급 | 대상 문서 | 상태 | 1차 소스 재검증 결과 |
|----|------|----------|------|---------------------|
| α-B1 | BLOCKER | 271_CONFIG_TRUTH §VI 행 + §VII #21 | **FIXED** | `evaluator.py:2017` mode 무관 `recon + 2.0·disc` 계산 확인; 호출처 `run_base_experiments.py:772, 1908` (271 경로) 확인; `best_model_detailed.csv` 실존 + 첫 행 `0.00186+2.0×1.70528=3.41241` 수치 일치; `visualization/best_model_visualizer.py:1184` 비게이트 소비처 확인; `run_base_experiments.py:856`은 non-adaptive else 분기라 271 미실행 확인; score dispatch(scoring.py:326-333) adaptive 분기·default/ratio_weighted 미호출 사실은 유지 |
| α-B2 | BLOCKER | 271_CONFIG_TRUTH §VI GRL classifier 행 + §VIII GRL Details Architecture | **FIXED** | `model.py:177-186` default 분기: `LayerNorm → Linear(512→256) → GELU → Dropout(0.1) → Linear(256→1)` = **Linear 2개** 실측; 코드 주석 "Default: 1-layer MLP"(:178)는 hidden-층 수 기준 표현 확인 |
| α-B3 | BLOCKER | 271_CONFIG_TRUTH §VIII Loss Components GRL 행 | **FIXED** | `compute_adaptive_lambda`(loss.py:683) 유일 호출처 = `trainer.py:610` (D 경로, 271 비활성) grep 전수 재확인; GRL λ는 inline `(‖∇L_main‖/(‖∇L_grl‖+1e-4)).clamp(0,10)` (`trainer.py:751-765`, 공식 :760) 별개 공식 확인 |
| α-m1 | MINOR | 271_CONFIG_TRUTH §VI freeze 두 행 부기 | **FIXED** | 실측: init override 블록 `trainer.py:49-55`(대입문 :55), ValueError guard `:75-79`(`if`문 :76) |
| RB-1 | BLOCKER | EXPERIMENT_PROTOCOL_TRUTH §④-실행 3항 | **FIXED** | [271c] metadata `num_epochs=500`·`eval_interval=5` (PSM/SWaT full/WaDi A2 json 직접 조회); `baseline_common.py` unsup 22종 `'epochs': 10` + "2026-06-06: unsupervised unified to 10" 주석 실측, weak 4종 50; `:256/:266` "epochs=50 user override"는 stale; `eval_interval: int = 1` 기본값(:943); early stopping grep 0건 — "양쪽 50ep 완주" 삭제, 비대칭+실제 공통점으로 재작성 (해석·정당화 미추가, 사실만) |
| RM-1 | MINOR | EXPERIMENT_PROTOCOL_TRUTH §④-실행 1항 | **FIXED** | `baseline_common.py:757` `n_runs = 5 if model_name == 'random' else 1`, `:786-796` mean(보고값)+std+per_run_metrics 집계 실측 — "모든 실험 단일 run" → MAE/deterministic baseline 한정 + random 5-run 예외 명시 |
| NMr-1 | MINOR | **NOTION_DIGEST.md** §IV-11 + I-7 † 주석 | **FIXED** (3번째 파일 — 발견 위치가 이 문서) | Page 0 덤프 디코딩 후 "127" 전수 regex: ①§1.2 지원 데이터셋 표(@3013) ②d_model 매핑 표(@13385) ③num_features 파라미터 표(@36627) ④§5.2.1(@68794) = **4개소** 재확정 — "3개소" 2개소(IV-11 본문 + I-7 † 주석) 모두 4개소로 정정 |

## 수정 전후 인용

### α-B1 — 271_CONFIG_TRUTH §VII #21 (§VI 행도 동일 취지)

- **전 (r2)**: "`lambda_disc=2.0`의 유일한 런타임 소비처는 `compute_default_score` (…)이며, dispatch(…)가 `anomaly_score_mode='adaptive'`에서 분기하므로 271에서 **절대 실행되지 않는다**."
- **후 (r3)**: "**score-path에서 dead** (…): dispatch(scoring.py:326-333)가 adaptive에서 분기하므로 default/ratio_weighted 미호출 — 은 맞다. **단 `lambda_disc=2.0`은 271 런타임에서 실제로 읽힌다**: 진단 경로 `evaluator.py:2017`(`compute_detailed_losses`)이 score-mode 무관하게 `'total_loss' = recon + 2.0·disc`를 계산하고, 271 경로 `run_base_experiments.py:772`·`:1908`에서 호출되어 `best_model_detailed.csv`의 `total_loss` 칼럼으로 기록된다 (…). **이 칼럼은 점수도 지표도 아니다** (…). 정밀 결론: **진단용 detailed losses CSV에는 `lambda_disc=2.0`이 쓰이나, 평가·선정에 쓰이는 anomaly score(adaptive 모드)·전 평가지표에는 무참여 — 논문의 score 식과 무관.**"
- §VI 행 Status: `**INACTIVE**` → `**INACTIVE in score-path** (진단 CSV에는 소비 — r3 정정)`.

### α-B2 — 271_CONFIG_TRUTH §VI:295 / §VIII:439

- **전**: "GRL classifier (DANN-style, 1-layer MLP)" / "Architecture | 1-layer MLP: `LayerNorm → Linear(…) → … → Linear(256, 1)`"
- **후**: "GRL classifier (DANN-style, **2-layer MLP** — r3 정정, SYNTHESIS 표A 표기 통일)" / "Architecture | **2-layer MLP** (Linear 2개 = hidden 1층): `LayerNorm → Linear(d_model, d_model//2=256) → GELU → Dropout(0.1) → Linear(256, 1)` (`model.py:177-186`; 코드 주석 "Default: 1-layer MLP with LayerNorm"(model.py:178)은 hidden-층 수 기준 표현 — "1-layer MLP" 표기 금지 …)"

### α-B3 — 271_CONFIG_TRUTH §VIII:430

- **전**: "… `grl_loss_weight=0.2`; adaptive lambda (VQGAN-style); …"
- **후**: "… `grl_loss_weight=0.2`; adaptive lambda (trainer inline grad-ratio, `trainer.py:751-765` — `(‖∇L_main‖/(‖∇L_grl‖+1e-4)).clamp(0,10)` :760; discriminator 전용 VQGAN-style `compute_adaptive_lambda`(loss.py:683, 유일 호출처 trainer.py:610, 271 비활성)와 **별개 공식** — 귀속 금지 …); …"

### α-m1 — 271_CONFIG_TRUTH §VI freeze 두 행 부기

- **전**: "(trainer.py:49-54는 별개의 init-시 config-validation override)" / "(trainer.py:74-78은 별개의 동시-flag ValueError guard)"
- **후**: "(trainer.py:49-55 … — 대입문 :55)" / "(trainer.py:75-79 … — `if`문 :76)"

### RB-1 — EXPERIMENT_PROTOCOL_TRUTH §④-실행 3항

- **전 (r2)**: "**Baseline 학습 패리티 (확인 가능 범위)**: ① epoch 수 — MAE `num_epochs=50` (`config.py:264`) = baseline 50 ("epochs=50 user override", `baseline_common.py:256, 266`; …) … ④ … → 양쪽 모두 50 epoch 완주. …"
- **후 (r3)**: "**Baseline 학습 설정 — epoch 수·eval 간격은 패리티가 아니라 비대칭 (r3 정정, RB-1 …)**: ① epoch 수 (3단 비대칭): MAE(271) = **500 epochs** ([271c] metadata …) / unsupervised 22종 = **10 epochs** (… "2026-06-06: unsupervised unified to 10") / weak 4종 = **50 epochs** (…). ⚠️ 인용 금지 2건: `config.py:264`는 dataclass default …, `baseline_common.py:256/:266`은 stale …. ② eval 간격 (비대칭): MAE 5-epoch 간격 vs baseline 매 epoch. ③ 실제 공통점: ⓐ best-epoch 기준 동일 pak_auc_f1 ⓑ 주기평가-후-best 구조 동일 ⓒ early stopping 양쪽 부재 → 각자 설정된 epoch 수 완주. … 논문 기재 사항 (사실만): 비대칭(500/10/50, 5 vs 1)을 명시 공개 …; 해석·정당화는 Phase 3/5."
- 근거 인덱스 행의 패리티 인용(`config.py:264`, `baseline_common.py:256, 266, …`)도 비대칭 실측 인용으로 교체.

### RM-1 — EXPERIMENT_PROTOCOL_TRUTH §④-실행 1항

- **전**: "모든 실험은 dataset entry당 **단일 run** (…)"
- **후**: "**MAE 실험(run_base_experiments) 및 random 제외 baseline**은 dataset entry당 **단일 run** (…). **예외: baseline `random`은 5회 독립 run → mean±std** (`baseline_common.py:757` `n_runs = 5 if model_name == 'random' else 1`, 집계 `:786-796` …)."

### NMr-1 — NOTION_DIGEST §IV-11 + I-7 † 주석

- **전**: "Page 0은 3개소(num_features 표·d_model 매핑·§5.2.1)에서 127" / "(d_model 표 · num_features 표 · §5.2.1, 3개소)"
- **후**: "Page 0은 **4개소**(§1.2 지원 데이터셋 표·num_features 표·d_model 매핑·§5.2.1)에서 127 (r3 정정 — 초판 '3개소'는 §1.2 표 누락 …)" / "(§1.2 지원 데이터셋 표 — I-7이 전사한 표 — · d_model 표 · num_features 표 · §5.2.1, **4개소**)"
- 결론(123 확정·주석 체계) 불변 — 카운트만 정정.

## 비고 (쓰기 범위 외 — 후속 라운드 메모)

1. **구 fixlog 2건의 오검증 기록은 미수정** (쓰기 허용 범위 외): `p1_271truth_fixlog_r2.md` V2-B4 행의 "유일 소비처 재확인" 및 `p1_protocol_fixlog_r2.md` M-4 행의 "epoch 50 통일 재검증"은 grep 누락/stale 주석에 의한 오검증 — **본 문서가 해당 기록을 대체(supersede)**. 두 대상 문서의 r3 정정 이력에 동일 취지 명기 완료.
2. **α-m2 (CODEBASE_UNDERSTANDING RESOLVED 주석 동기화)·α-m3 (RESEARCH_SYNTHESIS excl22 이원 수치 주석)**: 본 fixer 지시 범위 외 (지시 목록은 α-MINOR 1건 = α-m1만 271_CONFIG_TRUTH 대상) — 차기 patch 라운드 라우팅 필요.
3. NOTION_DIGEST.md 수정은 지시 목록 B-②("IV-11 '127 3개소' → 4개소")의 이행이며, 발견의 물리적 위치가 NOTION_DIGEST §IV-11이라 해당 문서에서 수정 — "대상 문서 2개" 집계와의 차이를 본 frontmatter `targets_modified`에 명시.
