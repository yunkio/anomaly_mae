---
phase: 1
agent: 271truth-verifier-2
directives: [R17, R34]
last_modified: 2026-06-10
---

# 271_CONFIG_TRUTH.md — 검증자 2 보고서 (코드 → 문서 완전성 / 누락 색출)

**검증 대상**: `paper/01_research_understanding/271_CONFIG_TRUTH.md` (reconciler 정정본, 2026-06-10)
**관점**: 검증자 1과 반대 방향 — "문서에 적힌 것이 맞는가"가 아니라 **"코드/메타데이터에 있는 것이 문서에 빠짐없이 판정되었는가"**. 검증자 1 산출물 미열람. `paper_legacy/` 미접촉.

---

## 판정 요약

**FAIL — BLOCKER 6건, MAJOR 3건, MINOR 5건.**

문서의 metadata 수치층(§I–§V)은 **전수 재검증에서 무결**했다 (37개 파일 직접 파싱, 114 공통키 byte-단위 일치, 가변키 3개뿐임을 기계적으로 확인). 그러나 R17의 핵심 요구 — "옵션으로 남아있으나 271에서 안 쓰이는 부분의 명확한 구분" — 에서 **판정 자체가 누락된 옵션 4계열**(`lambda_disc`, `minmax_clamp_*`, `anomaly_interval_scale`, `anomaly_loss_weight/direction`)과 **잘못된 활성 판정 2건**(§VIII "Total dataset length 275,000", §VIII "Anomaly loss warmup 50ep ramp"), 그리고 **R34 근거 진술 오류 1건**(Gaussian smoothing "코드베이스에 부재" — 실제로는 존재)이 발견되었다. 모두 §II의 값만 보고 논문에 잘못 옮겨 적을 수 있는 A3-위반 경로다.

| 심각도 | 건수 | 항목 |
|---|---|---|
| BLOCKER | 6 | B1 total_length 오판정 · B2 anomaly-loss ramp 오판정 · B3 Gaussian smoothing 허위 부재 진술(R34) · B4 lambda_disc 판정 누락 · B5 minmax_clamp 판정 누락 · B6 anomaly_interval_scale 판정 누락 |
| MAJOR | 3 | M1 anomaly_loss_weight/direction 제외목록 누락 · M2 bare `adaptive_lambda` 이름충돌 미경고 · M3 정규화 활성 세부(train-only fit + [0,1] tight clip) 미기재 |
| MINOR | 5 | m1–m5 (말미) |

---

## 검증 방법 (실측)

1. **전수 인벤토리**: `dataclasses.fields(Config)` 기계 추출 → **117 필드**. 문서 §II 표 키를 정규식 파싱 → 114개. 차집합 = 정확히 §III의 가변 3키 (`num_features`, `grl_pos_weight`, `sliding_window_train_ratio`). **§II 표에 dataclass 외 키·중복·누락 0건** ✅
2. **metadata 전수 대조**: 37개 `experiment_metadata.json` 전부 로드 → 키별 값 분포 diff. **가변 키는 정확히 3개뿐** (batch_size 등 그 외 114키는 37개 파일 모두 동일) ✅ — §V "0 blockers" 주장 자체는 사실. 대표 3개체(PSM, SWaT/excl22, SMD/m-1-5) §II 값 114/114 일치, 가변키 대표값(grl_pos_weight 3.141→3.14, 999.0, 59.181→59.18; train_ratio 0.8007/0.937; num_features 55/123) §III 일치 ✅. `swat_eval_mode`는 config가 아닌 **top-level metadata 키**로 존재(`None` vs `'excl22'`) — §IV 서술과 일치 ✅
3. **조건부 분기 전수 수집**: `model.py / loss.py / scoring.py / trainer.py / evaluator.py / dataset_sliding.py / datasets/ / utils/experiment.py`에서 config 필드 참조를 grep으로 전수 매핑 → 아래 대조표.
4. **launch provenance**: 실제 271 런은 `configs/queue_fullrerun_20260601_190603.json` 계열의 `exp271_271canon_baseline` override(lr=0.001, metadata와 일치; `queue_271_to_305.json`의 lr=0.0015는 **구버전 큐**로 미사용)이며, `summary.json`상 실행 키는 `SMD_simple_<machine>` 등 per-entity 데이터셋. `resume_dedup_v2b.py`의 "WaDi batch_size=512" 주석은 의심했으나 **metadata 전수 diff로 반증**(전 개체 1024) ✅

---

## BLOCKER 상세

### B1. §VIII "Total dataset length | 275,000 timesteps" — 잘못된 판정 (미사용 합성 전용 필드를 271 사실로 제시)

- `sliding_window_total_length=275000`은 **합성(simulation) 데이터 생성 전용** 필드다. 소비처는 `scripts/ablation/run_ablation.py`와 `mae_anomaly/visualization/*`(합성 재생성)뿐이며, 271의 실행 경로인 `scripts/run_base_experiments.py`는 이 필드를 **한 번도 읽지 않는다** (실데이터 길이는 `total_length = len(signals)`, run_base_experiments.py:1804로 실측).
- 271의 37개 개체는 모두 실데이터로 길이가 제각각이다 — PSM 220,322 (run_base_experiments.py:287 주석 및 `EXPERIMENT_PROTOCOL_TRUTH.md` §데이터표: 176,401+43,921), WaDi A1 ≈1,382,402, SMD machine당 수만 단위. **"275,000 timesteps"는 어느 개체에도 해당하지 않는다.**
- 위험: 논문 Dataset 절에 가짜 길이가 들어가는 직통 경로. 같은 문서 §VIII 표가 곧장 인용될 1순위 표이므로 waive 불가.
- 요구 조치: 해당 행 삭제(또는 "stale 합성 전용 필드, 271 미사용; 실 길이는 EXPERIMENT_PROTOCOL_TRUTH 참조"로 교체), `sliding_window_total_length`를 §VII 제외목록에 등재.

### B2. §VIII Training "Anomaly loss warmup | Ramp after warmup: `max(250//5, 2) = 50` epochs" — 271에서 no-op인 메커니즘을 활성 훈련 속성으로 제시

- `_compute_warmup_factor`(trainer.py:336-348)의 산식 자체는 맞다. 그러나 `warmup_factor`의 소비처는 **`anomaly_loss` 곱셈 3곳뿐**(loss.py:265, 272, 404)이고, 271에서는 `use_grl=True ∧ grl_disable_anomaly_loss=True`로 그 곱셈 라인에 도달하기 전에 `anomaly_loss = 0.0`으로 하드 제로된다(loss.py:259-261) — 문서 스스로 §VI에서 인정한 사실.
- GRL·FM 항은 ramp 없이 `not teacher_only` 게이트만으로 **epoch 251부터 즉시 full adaptive weight**로 투입된다 (trainer.py:639 FM, trainer.py:746 GRL — warmup_factor 미사용).
- 즉 §VIII Training 표의 이 행은 271에서 **아무 효과가 없는 메커니즘**이며, 논문에 "student 손실은 50 epoch에 걸쳐 ramp-up된다"는 허구의 서술이 들어갈 수 있다. R17 위반.
- 요구 조치: 행 삭제 + "GRL/FM은 ramp 없이 warmup 종료 직후 즉시 활성"을 명시.

### B3. Gaussian smoothing "Not present in codebase at all" (§VI 마지막 행, §IX FEEDBACK) — 허위 부재 진술 (R34)

- Gaussian smoothing 코드는 **존재한다**:
  - `mae_anomaly/scripts/q3_exploration/core/scoring.py:48-52` — `def gauss(score, sigma): return gaussian_filter1d(..., sigma=max(sigma,0.5), mode='reflect')`
  - `mae_anomaly/scripts/q3_exploration/core/postprocess.py:51` `savitzky_golay_smooth`, `:129` `double_gaussian`
  - 적용부: `exp_P1_tri_routing.py:104`, `exp_P10_stacking.py:131`, `exp_P14_boundary_refinement.py:147` 등 — 모두 `gauss(base_unsmoothed, 10)` 즉 **Notion에 언급된 'B2 variant(sigma=10 post-hoc smoothing)'가 코드로 실재**.
- 다행히 271 파이프라인(`evaluator.py`/`scoring.py`/`trainer.py`/`visualization/*`)은 q3_exploration을 일절 import하지 않음을 grep으로 확인 — **271의 모든 저장 점수·지표는 비평활(unsmoothed)** 이다. 따라서 "EXCLUDED (R34)" 판정 결론 자체는 옳다.
- 그러나 근거 진술("코드베이스에 전혀 없음")이 거짓이고, R34 관점에서 요구되는 등재 형식 — **"B2 variant 코드 존재(후처리 탐색 스크립트), 271 미사용, 논문 제외"** — 이 빠져 있다. 이대로면 Notion의 B2 결과와 이 문서를 대조하는 후속 에이전트가 모순에 빠지거나, 재현성 문구에 "스무딩 없음(코드에 부재)"이라는 검증 시 깨지는 주장이 실릴 수 있다.
- 요구 조치: §VI 행과 §IX를 "존재하나 271 미사용(증거: evaluator/scoring 무참조)·R34로 논문 제외"로 정정.

### B4. `lambda_disc = 2.0` — 판정 전면 누락 (비활성 옵션이 §II에 값만 노출)

- `lambda_disc`의 유일한 런타임 소비처는 `compute_default_score`(scoring.py:286-293, `recon + lambda_disc * disc`)이며, 이 분기는 `anomaly_score_mode='adaptive'`인 271에서 **절대 실행되지 않는다** (dispatch: scoring.py:326-333).
- 문서 어디에도(§VI 표, §VII 제외목록) `lambda_disc` 판정이 없다. §II에 `2.0`이라는 값만 있어, 점수식을 §II에서 재구성하는 작성자가 **"score = recon + 2·disc"** 라는 잘못된 식을 쓸 위험이 직접적이다 — 2026-05-28 FM-omission과 동급의 score-formula 사고 경로.
- 부수: 비활성 대안 score mode 2종(`'default'`, `'ratio_weighted'`)도 inactive component로 §VI/§VII에 미등재.
- 요구 조치: §VII에 "`lambda_disc=2.0` — adaptive 모드에서 dead; default/ratio_weighted 점수 분기 일체 미사용" 등재.

### B5. `minmax_clamp_min = -4.0` / `minmax_clamp_max = 4.0` — 판정 전면 누락 (비활성인데 값이 ±4로 노출)

- clamp는 `minmax_range='neg1_1'`(NPSR-style)일 때만 적용된다: `dataset_sliding.py:1019-1028`(`if minmax_range == 'neg1_1': ... cm_min, cm_max = minmax_clamp_min, minmax_clamp_max`; else 분기는 clamp=None) 및 `_minmax_per_feature` docstring 명문: **"271 default: feature_range=(0, 1), clip=True, clamp=None"** (dataset_sliding.py:956).
- 271은 `minmax_range='0_1'` → **±4 test-only clamp는 한 번도 적용되지 않는다.** 그러나 §II에 ±4.0이 노출되고 §VI/§VII/§VIII 어디에도 판정이 없어, 전처리 절에 "test 구간을 [-4,4]로 clamp"라는 허구가 들어갈 수 있다.
- 요구 조치: §VII 등재 + §VIII Normalization에 실제 활성 동작(M3) 기재.

### B6. `anomaly_interval_scale = 0.75` — 판정 전면 누락 (합성 전용 필드)

- 소비처는 합성 데이터 생성뿐: `run_ablation.py:944,1456` (`interval_scale=...`), `visualization/base.py:306`. 271 경로(`run_base_experiments.py`, `loaders.py`, `dataset_sliding.py` 실데이터 경로)에서는 **무참조**.
- §II에 0.75가 노출되나 판정 없음. 271은 합성 데이터셋을 포함하지 않으므로 명백히 미사용 — 오용 위험은 B4/B5보다 낮으나 rubric상 판정 누락은 BLOCKER.
- 요구 조치: §VII 등재 ("합성 simulation 전용, 271 미사용").

---

## MAJOR 상세

### M1. `anomaly_loss_weight=2.0`, `anomaly_loss_direction='maximize'` — §VII 제외목록 불완전

§VII item 1은 같은 dead-branch의 `margin=0.5`, `dynamic_margin_k=6`은 명시하면서 동일 분기에서만 쓰이는 `anomaly_loss_weight`(loss.py:265,272,404 — 모두 zeroing 이후 도달 불가)와 `anomaly_loss_direction`(loss.py:262 elif — GRL 분기에 선점됨)은 누락했다. §II의 "2.0"을 보고 "anomaly 패치 손실에 2× 가중"이라 쓸 위험. → item 1에 두 키 추가.

### M2. bare `adaptive_lambda = True` — 판정 누락 + 활성 동명 메커니즘과 이름 충돌

`adaptive_lambda`는 **discriminator 전용** adaptive lambda로 `use_discriminator=False`인 271에서 dead (trainer.py — D 경로에서만 소비). 문제는 §VIII이 GRL/FM의 "adaptive lambda"를 활성 메커니즘으로 상세 기술하고 있어, §II의 `adaptive_lambda=True`를 그 메커니즘의 스위치로 오독하기 매우 쉽다는 점이다. 실제 활성 스위치는 `grl_adaptive_lambda`/`fm_adaptive_lambda`. → §VII에 "bare `adaptive_lambda`는 discriminator 전용, 271 dead — GRL/FM adaptive lambda와 무관" 명시 등재. (같은 family인 `disc_lr_ratio`, `adv_loss_weight`, `disc_warmup_epochs`, `disc_channels`는 §VII item 3의 묵시 범위로 수용 가능하나 키 명시가 바람직.)

### M3. §VIII Normalization 활성 세부 누락: train-only fit + 전구간 [0,1] tight clip

271의 실제 정규화(`_minmax_per_feature`, dataset_sliding.py:934-998)는: ① scaler min/max를 **train 구간에서만 fit** ② 전체 신호 변환 ③ **`clip=True`로 train+test 전체를 [0,1]에 tight-clip** (train 범위 밖 test 값 포화) ④ clamp 없음. 문서 §VIII은 "Min-max, per-feature, range [0,1]"만 기술 — fit 범위와 clip은 재현성·전처리 절에 들어가야 할 load-bearing 세부다. docstring(956행)이 "271 default"를 명문화하고 있으므로 인용만 하면 된다.

---

## MINOR

- **m1** §I "Note on orchestrator prior count (37 vs prior report of 37)" — 동어반복 문장. 정리 요망.
- **m2** §III grl_pos_weight "999.0 (capped sentinel)" — 실제 메커니즘은 cap이 아니라 patch-ratio 하한: `_patch_ratio = max(_patch_ratio, 0.001)` → `(1-0.001)/0.001 = 999.0` (run_base_experiments.py:2584-2585). 결과값 동일하나 표현 부정확.
- **m3** `eval_interval=5`의 실제 구동원은 config가 아니라 스크립트 상수 `EVAL_INTERVAL = 5` (run_base_experiments.py:94) — 값은 일치하므로 무해하나, "스크립트 레벨 하드코딩" 사실은 각주 가치.
- **m4** `teacher_warmup_early_stop_metric='recon_snr'` — config.py 외 **코드 전체 무참조** (early-stop 활성 시에도 metric 필드는 읽히지 않음). early-stop family가 §VI에서 INACTIVE 판정되어 실위험은 없으나, "필드 자체가 dead"라는 사실은 §VII 19/20번(NOT YET IMPLEMENTED류)과 같은 급으로 등재 가능.
- **m5** `use_sliding_window_dataset`, `random_seed=42`, `device='cuda'` — 판정 행 없음(값만 §II). 동작 위험 없음.

**참고(본 문서 책임 아님, 교차 기록)**: 271 실측 개체는 SMD 22/28, SMAP 5/54, MSL 5/27로 **벤치마크 전체의 부분집합**이다 (`SMD_simple_*` per-entity 실행, summary.json 실측). `EXPERIMENT_PROTOCOL_TRUTH.md`는 "SMD 28 machines / SMAP 54 channels"로 기술하고 있어 개체수 정합은 protocol 문서·reconciler 라인에서 별도 확인 필요.

---

## 옵션 인벤토리 대조표 (전수 117 필드)

판정범례 — ✅=문서에 명시 판정·실측 일치 / ◐=family 부모 행으로 묵시 커버 / ❌=판정 누락 / ✖️=오판정

| # | Config 필드 | 271 값 | 코드 실측 (활성?) | 문서 판정 | 대조 |
|---|---|---|---|---|---|
| 1 | seq_length | 500 | ACTIVE | §VIII | ✅ |
| 2 | num_features | 개체별 | ACTIVE | §III | ✅ |
| 3 | use_sliding_window_dataset | True | ACTIVE(서술적) | 없음 | ❌(m5) |
| 4 | sliding_window_total_length | 275000 | **미사용**(합성 전용) | §VIII이 사실처럼 제시 | ✖️ **B1** |
| 5 | sliding_window_stride | 21 | ACTIVE | §VIII | ✅ |
| 6 | sliding_window_test_stride | -1 | ACTIVE(→49, `W//10-1`, experiment.py:16-39 실측) | §VIII 정정본 | ✅ |
| 7 | epoch_offset | True | ACTIVE([0,stride) 비복원 순회, dataset_sliding.py:1326-1341) | §VIII | ✅ |
| 8 | sliding_window_train_ratio | 개체별 | ACTIVE(실데이터에서 계산, run_base:2407) | §III | ✅ |
| 9 | normalize_mode | minmax | ACTIVE | §VIII | ✅(세부 M3) |
| 10 | minmax_range | 0_1 | ACTIVE | §VIII | ✅ |
| 11 | minmax_clamp_min | -4.0 | **미사용**(neg1_1 전용) | 없음 | ❌ **B5** |
| 12 | minmax_clamp_max | 4.0 | **미사용** | 없음 | ❌ **B5** |
| 13 | anomaly_interval_scale | 0.75 | **미사용**(합성 전용) | 없음 | ❌ **B6** |
| 14–15 | d_model / nhead | 512 / 8 | ACTIVE | §VIII | ✅ |
| 16 | num_encoder_layers | 4 | ACTIVE (model.py:359-362 실측) | §VI | ✅ |
| 17 | num_teacher_decoder_layers | 3 | ACTIVE (model.py:407-423) | §VI | ✅ |
| 18 | num_student_decoder_layers | 2 | ACTIVE (model.py:445-461) | §VI | ✅ |
| 19 | num_shared_decoder_layers | 0 | INACTIVE (model.py:367-368) | §VI | ✅ |
| 20–21 | dim_feedforward / dropout | 2048 / 0.15 | ACTIVE | §VIII | ✅ |
| 22 | masking_ratio | 0.15 | ACTIVE(`round(50×0.15)=8`, model.py:986 실측) | §VI/§VIII | ✅ |
| 23–24 | masking_ratio_min/max | -1/-1 | INACTIVE (trainer.py:522-524) | §VI | ✅ |
| 25–26 | num_patches / patch_size | 50 / 10 | ACTIVE | §VIII | ✅ |
| 27 | patchify_mode | linear | ACTIVE / CNN 분기 INACTIVE | §VI ×2행 | ✅ |
| 28 | mask_after_encoder | True | ACTIVE (model.py:1119-1129) | §VI | ✅ |
| 29 | shared_mask_token | False | INACTIVE(분리 토큰, model.py:499-505) | §VI | ✅ |
| 30 | use_transformer_encoder_decoder | True | ACTIVE | §VI | ✅ |
| 31 | use_flatten_linear_embedding | True | ACTIVE (model.py:605) | §VI | ✅ |
| 32–33 | cnn_channels / cnn_kernel_size | None / 3 | dead(CNN 분기) | §VI CNN행 | ◐ |
| 34 | use_revin | False | INACTIVE (model.py:312-314) | §VI | ✅ |
| 35–37 | revin_affine / revin_eps / revin_visible_only | — | dead | §VII#9 | ◐ |
| 38, 40, 41 | margin / margin_type / dynamic_margin_k | 0.5 / dynamic / 6 | dead — `_compute_patch_anomaly_loss` 도달 불가 확인(loss.py:259-272 실측) | §VII#1 | ✅ |
| 39 | lambda_disc | 2.0 | **dead**(default 점수모드 전용, scoring.py:286-293/326-333) | 없음 | ❌ **B4** |
| 42 | patch_level_loss | True | ACTIVE (loss.py:225-252) | §VI | ✅ |
| 43 | anomaly_loss_weight | 2.0 | dead(zeroed 이후 라인) | 부모 component만 | ❌ **M1** |
| 44 | anomaly_loss_direction | maximize | dead(GRL 분기에 선점) | 없음 | ❌ **M1** |
| 45 | normal_loss_weight | 1.0 | ACTIVE (loss.py:255-256) | §VIII | ✅ |
| 46 | student_recon_weight | 0.0 | NOT-IMPL(코드 전체 무참조 확인) | §VII#19 | ✅ |
| 47 | anomaly_score_mode | adaptive | ACTIVE(대안 2모드 dead — 미등재는 B4에 포함) | §VIII | ✅/❌ |
| 48 | grl_disable_anomaly_loss | True | ACTIVE (loss.py:259-261) | §VI | ✅ |
| 49 | use_grl | True | ACTIVE | §VI | ✅ |
| 50 | grl_cls_hidden | 0 | ACTIVE(default→`d_model//2=256`, model.py:178-187 실측 — §VIII 아키텍처 기술과 일치) | §VI/§VIII | ✅ |
| 51 | grl_loss_weight | 0.2 | ACTIVE(`_prev_epoch_grl_lambda*0.2`, trainer.py:762-764 실측) | §VIII | ✅ |
| 52 | grl_target_mode | window | ACTIVE (loss.py:285-287) | §VI | ✅ |
| 53 | grl_pos_weight | 개체별 | ACTIVE(ratio 하한 0.001→999.0, run_base:2584-2585) | §III | ✅(m2) |
| 54 | grl_balanced_sampling | False | INACTIVE (loss.py:313) | §VI | ✅ |
| 55 | grl_mode | classifier | ACTIVE / wdgrl 분기 INACTIVE (trainer.py:662) | §VI ×2행 | ✅ |
| 56 | grl_use_focal | True | ACTIVE (loss.py:337-340) | §VI | ✅ |
| 57 | grl_cls_lr_ratio | 0.1 | ACTIVE(→1e-4) | §VIII | ✅ |
| 58 | grl_cls_arch | default | ACTIVE('dann'/'2layer' 분기 dead) | §VI | ✅ |
| 59 | grl_adaptive_lambda | True | ACTIVE (trainer.py:751-765) | §VI | ✅ |
| 60–62 | wdgrl_k_critic / gp_weight / critic_lr | 5 / 10 / 1e-4 | dead | §VII#4 | ◐ |
| 63 | use_feature_matching | True | ACTIVE(train만; score 제외 = scoring.py:237 `fm_active=False` 실측) | §VI ×2행 | ✅ |
| 64 | fm_adaptive_lambda | True | ACTIVE(이중계상 없음 — loss.py:434-438에서 total 제외 후 trainer가 가산, 실측) | §VI | ✅ |
| 65 | fm_distance_metric | l2 | ACTIVE (loss.py:419-421) | §VI | ✅ |
| 66 | use_output_discrepancy | True | ACTIVE | §VI | ✅ |
| 67 | fm_loss_weight | 1.0 | ACTIVE | §VIII | ✅ |
| 68 | use_scad | False | INACTIVE (model.py:541, loss.py:355) | §VI | ✅ |
| 69–78 | scad_form/d_proj/temperature/margin/loss_weight/adaptive_lambda/ramp_up/patch_label_mode/use_memory_bank/memory_bank_size/proj_head_arch | — | dead | §VII#2 | ◐ |
| 79 | eval_disc_weight | -1.0 | ACTIVE(→1.0 fallback, scoring.py:99-102) | §VIII | ✅ |
| 80 | eval_fm_weight | -1.0 | 무관(FM 제외) | §VIII | ✅ |
| 81 | score_recon_disc_ratio | 4.0 | ACTIVE (scoring.py:247-250) | §VIII | ✅ |
| 82 | eval_complementary_masking | False | INACTIVE | §VI | ✅ |
| 83 | eval_complementary_k | 7 | dead | §VII#12 | ◐ |
| 84 | freeze_teacher_after_warmup | False | INACTIVE (trainer.py:50-55) | §VI | ✅ |
| 85 | freeze_encoder_only | False | INACTIVE (trainer.py:75-79) | §VI | ✅ |
| 86 | masking_ratio_anneal | False | INACTIVE | §VI | ✅ |
| 87 | use_discriminator | False | INACTIVE (trainer.py:236-237) | §VI | ✅ |
| 88 | d_grad_student_layers | all | NOT-IMPL | §VII#20 | ✅ |
| 89, 91–93 | disc_lr_ratio / adv_loss_weight / disc_warmup_epochs / disc_channels | — | dead(D 전용) | §VII#3 | ◐ |
| 90 | adaptive_lambda | True | **dead**(D 전용 — GRL/FM의 활성 adaptive lambda와 별개) | 없음 | ❌ **M2** |
| 94–97 | batch_size / num_epochs / learning_rate / weight_decay | 1024/500/0.001/0.001 | ACTIVE (AdamW fused, betas=(0.9,0.99) trainer.py:160-164 실측) | §VIII | ✅ |
| 98 | warmup_epochs | 10 | ACTIVE(LR LinearLR+CosineAnnealing, trainer.py:167-169) | §VIII | ✅ |
| 99 | teacher_only_warmup_epochs | 250 | ACTIVE | §VI/§VIII | ✅ (단 ramp 행은 ✖️ **B2**) |
| 100 | use_teacher_output_ema | False | INACTIVE (model.py:514) | §VI | ✅ |
| 101 | teacher_output_ema_momentum | 0.996 | dead | §VII#7 | ◐ |
| 102 | use_teacher_warmup_early_stop | False | INACTIVE (trainer.py:485) | §VI | ✅ |
| 103–104 | …patience / …min_epochs | 10 / 50 | dead | §VII#8 | ◐ |
| 105 | teacher_warmup_early_stop_metric | recon_snr | dead(**코드 전체 무참조**) | §VII#8 | ◐(m4) |
| 106 | best_epoch_metric | pak_auc_f1 | ACTIVE | §VIII | ✅ |
| 107 | eval_interval | 5 | ACTIVE(실구동은 스크립트 상수 `EVAL_INTERVAL=5`) | §VIII | ✅(m3) |
| 108–109 | use_amp / amp_dtype | True / bf16 | ACTIVE | §VIII | ✅ |
| 110 | use_discrepancy_loss | True | ACTIVE (loss.py:34) | §VI | ✅ |
| 111–114 | use_teacher / use_student / use_masking / force_mask_anomaly | True×4 | ACTIVE | §VI | ✅ |
| 115 | random_seed | 42 | ACTIVE(분할/노이즈 시드) | 없음(값만) | ❌(m5) |
| 116 | device | cuda | ACTIVE | 값만 | — |
| (비-config) | swat_eval_mode | None/'excl22' | top-level metadata 키 | §IV | ✅ |
| (비-config) | Gaussian smoothing | — | **존재**(q3_exploration; 271 미사용) | §VI "부재" | ✖️ **B3** |

**판정 통계**: 명시 ✅ 72 / 묵시 ◐ 28 / 누락 ❌ 9 (B4·B5×2·B6·M1×2·M2·m5×2) / 오판정 ✖️ 3 (B1·B2·B3).

---

## 추가 확인 사항 (문서 주장 중 본 검증에서 실측 통과한 것)

- §VI의 코드 라인 인용 22개소 전수 spot-check — **22/22 비-stale** (model.py:129-144 GRL 역전 `-lambda·grad` + "adversarial feature suppression" docstring 포함; reconciler 정정 방향 옳음).
- adaptive score 식(§VIII): scoring.py:239-253 실측 일치 — `scaled_disc/4.0`, FM 제외(`fm_active=False` 하드코딩 :237), eps=1e-4.
- §VII item 1(dynamic margin 도달 불가): loss.py:259-272 분기 구조로 확정 — `_compute_patch_anomaly_loss`는 GRL-비활성 else 분기에만 존재.
- batch_size 의혹(resume 스크립트의 WaDi 512 주석): 37개 metadata 전수 diff로 **반증** — 본 런은 전 개체 1024.

```
REQUEST: config-forensics 에이전트에 B1–B6 정정 + §VII에 lambda_disc/minmax_clamp/anomaly_interval_scale/
anomaly_loss_weight/anomaly_loss_direction/adaptive_lambda(disc)/sliding_window_total_length 등재 요청.
FEEDBACK: metadata 수치층은 전수 검증 통과 — 정정 범위는 '판정 레이어'(§VI–§VIII)에 국한되며 §I–§V는 재작업 불요.
```
