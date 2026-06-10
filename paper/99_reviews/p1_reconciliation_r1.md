---
phase: 1
agent: reconciler
directives: [R17]
last_modified: 2026-06-10
---

# Phase 1 Reconciliation Report (r1) — P1-1 (CODEBASE_UNDERSTANDING) vs P1-3 (271_CONFIG_TRUTH)

**판정 원칙 (R17)**: 271 config가 정본. 1차 소스 = ① 271 metadata
(`results/experiments/271_20260602_020545_271canon_baseline/**/experiment_metadata.json` — PSM·SWaT full/excl22·WaDi A2 직접 재추출, 2026-06-10)
② 코드 직접 추적 (`mae_anomaly/config.py`, `model.py`, `loss.py`, `scoring.py`, `evaluator.py`, `trainer.py`, `utils/experiment.py`,
`datasets/loaders.py`, `scripts/run_base_experiments.py`) ③ 실행 부수 산출물 (`summary.json`, `best_config.json`, checkpoint state_dict,
`configs/queue_fullrerun_20260601_190603.json`). P1-3도 무오류로 가정하지 않고 재검증함.

**핵심 판정**: exp271은 P1-1이 가정한 Set A(`CONFIG_PRESETS['A']`)가 아니라 **Set C 기반 + 대량 config override**다.
근거: `summary.json: "config_set": "C"` + `"description": "w500, p10, enc2, td4, sd1, dynamic d_model, linear embed"`;
`configs/queue_fullrerun_20260601_190603.json` exp271 entry `"set": "C"`, `config_override`(enc4/td3/sd2/d512/dff2048/linear/500ep/250warmup/bs1024/lr0.001);
전 37 entity metadata config 일치. P1-1의 Set-A 기반 수치들이 모순의 주 원인이었다.

---

## I. 모순 전수 목록 + 판정

| # | 항목 | P1-1 주장 | P1-3 주장 | 1차 소스 판정 + 근거 | 처리 |
|---|------|-----------|-----------|---------------------|------|
| 1 | exp271 preset 정체 | Set A archetype (`patch_cnn`) | (Set C 계열) linear 기반 | **P1-3 승** — `summary.json: config_set='C'`; queue entry `set: 'C'` + override | P1-1 헤더·§1·§8 정정 |
| 2 | patchify_mode | `patch_cnn` | `linear` | **P1-3 승** — metadata `patchify_mode='linear'` (37/37); `model.py:580` CNN 분기 미진입; checkpoint에 `patch_embed.weight` 존재(=linear 경로, `model.py:628`) | P1-1 §1·Patchify·§8 정정 |
| 3 | patch_size / num_patches | 5 / 100 | 10 / 50 | **P1-3 승** — metadata `patch_size=10`, `num_patches=50`; checkpoint `patch_embed=(512, 450)`=Linear(10×45→512) (SWaT) | P1-1 §1·§3.3·§4.4·§8 정정 |
| 4 | d_model / dim_feedforward | 128 / 512 | 512 / 2048 | **P1-3 승** — metadata `d_model=512`, `dim_feedforward=2048`; checkpoint 출력 차원 512 실측 | P1-1 §1·Encoder/Decoder·§8 정정 |
| 5 | num_encoder_layers | 2 | 4 | **P1-3 승** — metadata `num_encoder_layers=4`; code default 2는 `config.py:43` (override됨) | P1-1 정정 (default/271 병기) |
| 6 | teacher / student decoder 층수 | 4 / 1 | 3 / 2 | **P1-3 승** — metadata `num_teacher_decoder_layers=3`, `num_student_decoder_layers=2`; defaults 4/1은 `config.py:44-45` | P1-1 §1·Encoder/Decoder·§7.1·§8 정정 |
| 7 | num_epochs / teacher_only_warmup | 50 / 25(auto) | 500 / 250 | **P1-3 승** — metadata `num_epochs=500`, `teacher_only_warmup_epochs=250`(명시; auto 분기 `trainer.py:43-48`은 음수일 때만) | P1-1 §5.3·§8·REQUEST 해소 |
| 8 | batch_size | 512 (Set A preset) | 1024 | **P1-3 승** — metadata `batch_size=1024` (preset 512를 override) | P1-1 §8 정정 |
| 9 | dynamic_margin_k | 2.0 (code default) | 6 | **P1-3 승** — metadata `dynamic_margin_k=6`; default 2.0은 `config.py:99`. 단 #10에 의해 어차피 무효 | P1-1 §2.2·§8 정정 |
| 10 | dynamic margin 활성 여부 | §2.2에서 "exp271 default"로 동작 서술 + `L_OD = L_normal + L_anomaly` 표기 (§2.2 말미에 GRL-disable 언급은 있으나 margin 목록과 모순 방치) | 도달 불가(비활성) | **P1-3 승** — `loss.py:259-261`: `use_grl and grl_disable_anomaly_loss`(둘 다 True, metadata) → `anomaly_loss = torch.tensor(0.0, …)`; margin 분기(`_compute_patch_anomaly_loss`) 호출 자체가 없음. `margin=0.5`/`margin_type='dynamic'`/`dynamic_margin_k=6` 전부 학습에 무영향 | P1-1 §2.2 margin 목록 전체 [271 도달 불가] 마킹 + L_OD 식 정정 |
| 11 | masking | "round(100×0.15)=15 masked / 85 visible" | "15% fixed, anomaly-first; 7-8 patches" | **양쪽 부분 오류** — 비율(0.15)·anomaly-first(`force_mask_anomaly=True`)는 양쪽 일치·옳음. 개수: 실제 `round(50×0.15) = 8` masked / 42 visible (`model.py:986` `target_num_masked = round(current_seq_len * masking_ratio)`). P1-1의 15/85는 num_patches=100 가정 오류; P1-3의 "7-8"은 부정확(정확히 8) | P1-1 §1 정정; P1-3 §VIII 정정 |
| 12 | 입력 차원 (B, 500, 8) | 8 features | dataset별 25–123 | **P1-3 승** — 8-feature는 simulation(R33 논문 제외) 전용; metadata `num_features` 25–123 | P1-1 §1 정정 |
| 13 | test stride 산식 | `seq_length // 10 - 1 = 49` (`utils/experiment.py`) | "`num_patches - 1 = 49` at call site" | **P1-1 승 (유일한 P1-1-우위 항목)** — `resolve_test_stride` 구현은 `W // 10 - 1` (`utils/experiment.py:16-39`). P1-3은 `config.py:23-26`의 stale 주석("-1 = num_patches-1")을 따름. 271은 patch_size=10이라 두 식이 49로 우연히 일치(값 무영향) | P1-3 §VIII 정정 |
| 14 | GRL 작용 서술 | "making **the encoder** produce anomaly-uninformative representations" | "forces student to generate anomaly-**discriminative** features" | **양쪽 오류 (각각 다른 절반)** — 코드: `GradientReversalFunction` backward `-lambda × grad` (`model.py:129-140`); head docstring "GRL for adversarial feature **suppression**" (`model.py:143-144`); student path는 `latent_visible.detach()`로 encoder gradient 차단 (`model.py:1123-1126`). 즉 ①작용 대상 = **student decoder** (encoder 아님 — P1-1 오류), ②방향 = anomaly-identity 정보 **억제/uninformative** (discriminative 생성 아님 — P1-3 오류) | P1-1 GRL절 정정; P1-3 §VIII GRL행 정정 |
| 15 | anomaly-loss ramp 길이 | `warmup_epochs//5` (=10//5=2) | `max(250//5, 2)=50` | **P1-3 승** — `trainer.py:336-348` `_compute_warmup_factor`: `warmup_length = max(student_start // 5, 2)`(:342), `student_start = config.teacher_only_warmup_epochs`(=250). P1-1은 LR-warmup(`warmup_epochs=10`)과 변수 혼동 | P1-1 §2.5 정정 |
| 16 | WaDi 분할 서술 | "14days → train; attack → test" | 14days + attack 앞 50% → train; attack 뒤 50% → test | **P1-3 승** — `loaders.py:2201` `train_len = n_14d + n_atk // 2`; metadata `sliding_window_train_ratio` WaDi A1 0.937/A2 0.910 (attack 절반 포함 시에만 성립) | P1-1 §4.1 표 정정 |
| 17 | PSM 분할 서술 | "80% train, 20% test" | orig train 전체 + test 앞 50% → train (ratio 0.8007) | **P1-3 승** — `loaders.py:1686-1693` `// 2` 분할; 0.8007은 결과값이지 규칙이 아님 | P1-1 §4.1 표 정정 |
| 18 | SWaT loader | `load_swat_combined` (`loaders.py:23`) | `load_swat_a1a2_raw` (registry `SWaT_A1A2`) | **P1-3 승** — `DATASET_LOADERS['SWaT_A1A2'] = lambda: load_swat_a1a2_raw(swap=False)` (`loaders.py:2690`, 학습 시점 commit e5938eb에서도 동일 매핑); queue entry dataset 키 `SWaT_A1A2`. `load_swat_combined`은 legacy 키 `swat_A1A2` 전용 | P1-1 §4.1 표 정정 |
| 19 | WaDi loader | `load_wadi_14days_combined` (`loaders.py:278`) | `load_wadi_14days_raw` (registry `WaDi_14days_A1/_A2`) | **P1-3 승** — `loaders.py:2697-2698`; queue dataset 키 `WaDi_A1`/`WaDi_A2` | P1-1 §4.1 표 정정 |
| 20 | SMD loader / feature 수 | `load_smd(machine_id)` (`loaders.py:876`), 38 per machine | `SMD_simple_<machine>` → `load_smd_simple`, metadata 29–36 | **P1-3 승** — `summary.json` results key `SMD_simple_machine-1-4` 실측; metadata `num_features` 범위 29(machine-3-10)–36(machine-3-3), 22/28 entity | P1-1 §4.1 표 정정 |

**모순 전수: 20건** (P1-3 승 16, P1-1 승 1 [#13], 양쪽 부분 오류 2 [#11, #14], 양쪽 일치·수치만 정밀화 1 [#11의 비율 부분]).

추가 검증(모순 아님, 일치 확인): scoring 공식(`recon + scaled_disc/4`, FM 제외 `scoring.py:237` `fm_active = False`) — 양 문서 일치·코드 일치. AdamW fused/betas(0.9,0.99)(`trainer.py:160-164`) 일치. 평가 시 leave-one-out per-patch masking(P1-1 §3.1) — `evaluator.py:1648` docstring "masking each patch one at a time" 확인, `eval_complementary_masking=False`로 complementary 분기 비활성 — P1-1 옳음. PA%K AUC는 K=0..100 step1 적분 / per-K 보고 키는 0,5,…,100 — 양 문서 모순 없음.

---

## II. RF-002 판정 (EXPERIMENT_PROTOCOL_TRUTH의 REQUEST-1·2)

1. **`pa_0_f1`의 threshold**: **F1-최적 threshold** — `compute_full_metric_set`이 ROC sweep에서 F1-최적 threshold 1개를 선정(`evaluator.py:929-930`)하고, 모든 per-K PA%K(`pa_{k}_*`)가 그 threshold로 이진화한 예측에 PA 조정을 적용 (`evaluator.py:951-955`). anomaly-ratio threshold는 **적용되지 않음**.
2. **`pa_0_f1_ar` 부재**: 확정 — AR 변형 함수의 출력 키에 PA 계열 없음 (`evaluator.py:769-781`), [271c] PSM metrics 149키 실측에서 `_ar` 키 10개 중 `pa_*` 0건.
3. **affiliation의 threshold 의존성**: threshold-dependent — `affiliation_f1`=F1-최적 threshold (`evaluator.py:980-982` → `:730` `pred=(s>threshold)`), `affiliation_f1_ar`=AR threshold (`evaluator.py:790,794,811-813`). 두 키 모두 metadata에 실재.
4. **"pak_auc_pr" 내부 키**: **`pak_auc_prc_auc`** (`evaluator.py:840-841` `PAK_AUC_KEYS`; 산출 `evaluator.py:1271`; [271c] PSM metrics 키 실측). `pak_auc_pr`/`pak_auc_prc` 키는 코드·metadata 어디에도 없음. `vus_pr`은 별도 키.

→ `EXPERIMENT_PROTOCOL_TRUTH.md` REQUEST-1/REQUEST-2 아래 RESOLVED 블록으로 기록 완료. (보고 시 어느 threshold를 채택할지는 여전히 사용자/Phase 2+ 결정 — 사실관계만 확정.)

## III. WaDi A2 feature 수 (RF-004 일부) 판정

**123으로 확정.** ① [271c] `WaDi/A2` metadata `config.num_features=123`; ② 현 raw CSV 실측 124 cols = 123 + label; ③ 127의 정체: 원본 `WADI_attackdataLABLE.csv`의 sensor 127개(131 − 3 meta − 1 label) 중 **all-NaN 4개** (`2_LS_001_AL`, `2_LS_002_AL`, `2_P_001_STATUS`, `2_P_002_STATUS`)를 `prepare_raw_datasets.py` `handle_nan`이 drop → 123. 2026-06-10 직접 diff 재계산으로 4개 컬럼 식별. [N-METH]의 127은 drop 전 원본 수치. → `EXPERIMENT_PROTOCOL_TRUTH.md` FEEDBACK-2 RESOLVED + §① 표 정정 완료.

## IV. 부수 발견 (스코프 외이나 근거 확보되어 함께 정정)

1. **SWaT feature 수 51 → 45 (EXPERIMENT_PROTOCOL §① 표 오류)**: 학습 모델 입력 45 — metadata·`best_config.json`·checkpoint `patch_embed.weight=(512,450)` 삼중 확인. 45 = 51 − combined-constant 6 {P202, P401, P404, P502, P601, P603} (재계산 일치). ⚠️ 현 machineA CSV(51 feat) + 현행/학습-시점 loader(constant 제거 없음)는 51을 반환 → 학습 당시 source-machine CSV가 45-feature 버전이었던 것으로 추정. **재현성 미해결 플래그**로 FEEDBACK-7 신설 (재실행 시 feature 수 확인 필수).
2. **SMD feature 범위 "32–38" → "29–36"** (EXPERIMENT_PROTOCOL §① 표): metadata 실측(22/28) 범위로 정정.

## V. 산출물

- `paper/01_research_understanding/CODEBASE_UNDERSTANDING.md` — 13개 절 정정 + 헤더 노트 + 말미 정정 목록 + REQUEST 2건 RESOLVED.
- `paper/01_research_understanding/271_CONFIG_TRUTH.md` — 3건 정정 (test-stride 산식, masking 개수=8, GRL suppression 방향) + 말미 목록. metadata 수치(§II–IV)는 재검증 전건 일치.
- `paper/01_research_understanding/EXPERIMENT_PROTOCOL_TRUTH.md` — REQUEST-1/2 RESOLVED, FEEDBACK-2 RESOLVED(WaDi 123), §① 표 3셀 정정(SWaT 45·WaDi A2·SMD 29–36), FEEDBACK-7(SWaT 재현성 플래그) 신설.
