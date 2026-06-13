---
phase: 1
agent: fixer-1
directives: [R17, R34]
last_modified: 2026-06-10
---

# Fix Log r2 — 271_CONFIG_TRUTH.md (verifier-1/2 리뷰 반영)

**대상 문서**: `paper/01_research_understanding/271_CONFIG_TRUTH.md`
**리뷰 입력**: `p1_271truth_verifier1_r1.md` (V1), `p1_271truth_verifier2_r1.md` (V2)
**원칙**: 각 발견은 리뷰 주장 자체를 1차 소스(코드 file:line / metadata 필드)로 재확인한 뒤에만 반영. 추측 반영 0건.

## 처리 요약

| 구분 | 건수 |
|---|---|
| FIXED (BLOCKER/MAJOR/MINOR) | **23** (V1: 9, V2: 14) |
| REJECTED | **0** |
| NOTE급 선택 반영 | 2 (V1-N2, V1-N3) / 무조치 1 (V1-N1 — 리뷰어 스스로 "문제 아님" 판정) |

두 리뷰의 모든 BLOCKER/MAJOR/MINOR 주장이 1차 소스 재검증에서 **사실로 확인**되어 전건 FIXED. 반박(REJECTED) 사유 발생 없음. 단, 리뷰 자체의 경미한 표기 부정확 2건을 말미 "리뷰 정확성 메모"에 기록 (실체 판단에는 영향 없음 — 문서에는 정밀 라인으로 반영).

---

## Verifier-1 처리표

| ID | 심각도 | 리뷰 주장 | ① 1차 소스 재확인 | ② 처리 |
|---|---|---|---|---|
| B1 | BLOCKER | masking annealing 근거 오류 — trainer가 경로를 구현함 (trainer.py:1201) | **확인**: `trainer.py:1201` `if getattr(self.config, 'masking_ratio_anneal', False) and epoch >= teacher_warmup:` + 경로 본문 :1201-1210 실재 (직접 sed 실측). config.py:241-243은 flag 정의일 뿐 | **FIXED** — §VI annealing 행 근거를 trainer.py:1201(-1210) 조건 False 평가로 교체 + "trainer never triggers" 오서술 명시 정정; §VII #14 보강 |
| M2 | MAJOR | SWaT excl22의 `timing.best_epoch_metric='excl22_pak_auc_f1'` ≠ config 키 `'pak_auc_f1'` | **확인**: metadata 직접 파싱 — excl22: config=`pak_auc_f1`, timing=`excl22_pak_auc_f1`; full: timing=`pak_auc_f1`; wall_time 양쪽 동일(같은 모델) | **FIXED** — §IV에 운영 주의 블록 추가 (config=템플릿, 런타임이 excl22에서 metric 이름 override) |
| M3 | MAJOR | complementary masking 근거는 evaluator.py:1716이어야 | **확인**: `evaluator.py:1716` `_use_complementary = getattr(...)`, `:1737` `if _use_complementary:` (K-group 경로 :1737-1745) 실측 | **FIXED** — §VI 행 근거를 evaluator.py:1716 + :1737로 교체 (config.py:226-229는 flag 정의로 부기) |
| M4 | MAJOR | freeze_teacher_after_warmup 런타임 gate는 trainer.py:1141-1142 | **확인**: `:1141-1142` `if (getattr(...'freeze_teacher_after_warmup'...) and epoch == teacher_warmup ...)` 실측; 기존 인용부(49-54)는 init-시 warmup-길이 override | **FIXED** — §VI 행 근거를 trainer.py:1141-1142로 교체, init override(49-54)는 별개임을 부기 |
| M5 | MAJOR | freeze_encoder_only 런타임 gate는 trainer.py:1169-1170 | **확인**: `:1169-1170` 동일 패턴 실측; 기존 인용부(74-78)는 동시-flag ValueError guard | **FIXED** — §VI 행 근거를 trainer.py:1169-1170으로 교체, guard(74-78)는 별개임을 부기 |
| M1 | MINOR | SMAP/G-7 train_ratio는 0.617 (~0.625 아님) | **확인**: metadata 실측 G-7=0.6167064, P-1=0.6262, P-4=0.6255, T-1=0.6251, T-3=0.6255 | **FIXED** — §III-3c를 "0.617–0.626 (G-7: 0.617; 나머지 ~0.625)"로 교체 |
| Mi1 | MINOR | decoder 구성 라인: teacher 419-423, student 457-461 | **확인**: `model.py:419` `self.teacher_decoder = nn.TransformerEncoder(` (:421 num_layers), `:457` student 동형 (:459) 실측; 게이트는 :406/:444 | **FIXED** — §VI 두 행 라인 교체 (+게이트 라인 부기) |
| Mi2 | MINOR | linear embedding: :580 patch_cnn 분기, :624 linear 분기 | **확인**: `model.py:580` `if self.patchify_mode == 'patch_cnn':`, `:624` `elif self.patchify_mode == 'linear':` 실측 (577-578은 함수 def) | **FIXED** — §VI 행 근거를 :580 skip + :624 진입으로 교체 |
| Mi3 | MINOR | mask_after_encoder teacher branch(:1028-)도 인용 필요 | **확인**: `model.py:1028` `if self.mask_after_encoder:` (teacher, branch 본문 1028-1036), `:1119-1129` student branch 실측 | **FIXED** — §VI 행에 teacher branch model.py:1028-1036 병기 |
| N1 | NOTE | PSM 0.8007은 4자리 반올림 — 문제 아님 | 확인 (0.8006508…) | 무조치 (리뷰어 판정대로 허용 가능 반올림) |
| N2 | NOTE | `_es_on = False` 표현은 하드코딩처럼 오독 가능 | **확인**: `trainer.py:485` getattr 동적 평가 | 선택 **FIXED** — §VI 표현을 getattr 평가식으로 정밀화 |
| N3 | NOTE | score formula 인용 범위 239-253 → ~256 | **확인**: return dict의 `'score'` 조립이 :255-256 | 선택 **FIXED** — §VIII 인용 239-256으로 교체 |

## Verifier-2 처리표

| ID | 심각도 | 리뷰 주장 | ① 1차 소스 재확인 | ② 처리 |
|---|---|---|---|---|
| B1 | BLOCKER | §VIII "Total length 275,000"은 합성 전용 필드 오판정 | **확인**: `sliding_window_total_length` 소비처 grep 전수 — config.py:21(기본값), visualization/* 3곳, run_ablation.py:942/1454/420 + ablation configs뿐. `run_base_experiments.py` 무참조; 실길이는 `:1804` `total_length = len(signals)` (PSM 220,322 — :287 주석) | **FIXED** — §VIII 행을 "개체별 실데이터 길이" + stale 필드 경고로 교체; §VI INACTIVE 행 + §VII #24 신설 |
| B2 | BLOCKER | anomaly-loss 50ep ramp는 271에서 no-op; GRL/FM은 ramp 없이 즉시 투입 | **확인**: `warmup_factor` 소비처 grep 전수 = loss.py:265,272,404 (전부 anomaly_loss 곱셈)뿐; 271은 loss.py:259-261에서 anomaly_loss 하드 제로(zeroing이 :262 elif/:268 else에 선행). FM 가산 trainer.py:639(게이트 `not teacher_only`)/:652, GRL :746/:762-763 — warmup_factor 미사용 | **FIXED** — §VIII Training 행을 "Student-loss 활성화 시점: ramp 없음" 정정 서술로 교체; §VI에 no-op 판정 행 신설 |
| B3 | BLOCKER | Gaussian smoothing "코드베이스에 부재"는 허위; q3_exploration에 실재(B2 variant 포함), 271 무참조 | **확인**: `q3_exploration/core/scoring.py:48-51` `def gauss(...)` gaussian_filter1d, `core/postprocess.py:51` savitzky_golay_smooth / `:129` double_gaussian 실측; `gauss(base_unsmoothed, 10)` 적용부 다수 실측 (`experiments/exp_P14_boundary_refinement.py:147` 등). evaluator/scoring/trainer/visualization/run_base_experiments에서 `q3_exploration` grep 0건 → 271 무참조 | **FIXED** — §VI 행을 "EXCLUDED (R34) — 코드 존재, 271 미사용"으로 재작성; §VII #18 재작성; §IX FEEDBACK 정정 |
| B4 | BLOCKER | `lambda_disc=2.0` 판정 누락 — adaptive 모드에서 dead | **확인**: 유일 소비처 `compute_default_score` scoring.py:286-293 (`recon + lambda_disc * disc`); dispatch :326-333이 `mode=='adaptive'` 분기 → default/ratio_weighted(:296-304) 미호출 | **FIXED** — §VI INACTIVE 행 + §VII #21 신설 + §VIII Anomaly Score 절에 주의 블록 ("score = recon + 2·disc" 재구성 금지) |
| B5 | BLOCKER | `minmax_clamp_min/max=±4` 판정 누락 — neg1_1 전용이라 미적용 | **확인**: `dataset_sliding.py:1019-1028` — `'neg1_1'`일 때만 `cm_min, cm_max = minmax_clamp_*`; `'0_1'` 분기 :1023-1025는 `None, None`. docstring `:956` "271 default: feature_range=(0, 1), clip=True, clamp=None" 실측 | **FIXED** — §VI INACTIVE 행 + §VII #22 신설 + §VIII Normalization 세부(M3와 함께) 반영 |
| B6 | BLOCKER | `anomaly_interval_scale=0.75` 판정 누락 — 합성 전용 | **확인**: 소비처 grep 전수 — run_ablation.py:944/1456, visualization/base.py:306, training_visualizer.py:97, data_visualizer(고정값 1.5) 등 합성 생성 경로뿐; run_base_experiments 무참조 | **FIXED** — §VI INACTIVE 행 + §VII #23 신설 |
| M1 | MAJOR | `anomaly_loss_weight=2.0`/`anomaly_loss_direction='maximize'` §VII 누락 | **확인**: weight 소비처 loss.py:265,272,404 — 모두 :259-261 zeroing 이후 도달 불가; direction 판정 :262 `elif`는 :259 GRL 분기에 선점 | **FIXED** — §VII #1을 "Dynamic margin + anomaly-loss family"로 확장, 두 키 dead 판정 추가 |
| M2 | MAJOR | bare `adaptive_lambda=True`는 D 전용 dead + 이름 충돌 위험 | **확인**: bare 필드 소비처 grep — `trainer.py:608` `if self.config.adaptive_lambda:` 1곳 (D adversarial 경로 내부, `compute_student_adversarial_loss` 인접); `use_discriminator=False` → `self.discriminator = None`(trainer.py:236-237)으로 미도달 | **FIXED** — §VI INACTIVE 행 + §VII #25 신설 (grl/fm_adaptive_lambda와 별개임을 명시; D-family 4키 부기) |
| M3 | MAJOR | 정규화 활성 세부(train-only fit + [0,1] tight clip + clamp 없음) 미기재 | **확인**: `_minmax_per_feature` dataset_sliding.py:935-998 — `signals[:train_end]` fit, `clip=True` 전구간 tight-clip, docstring :943-957 "271 default" 명문 | **FIXED** — §VIII Normalization 행에 ①~④ 세부 + 근거 라인 기재 |
| m1 | MINOR | §I "37 vs prior report of 37" 동어반복 | 확인 (문서 원문) | **FIXED** — 한 문장으로 정리 |
| m2 | MINOR | 999.0은 cap이 아니라 patch-ratio 하한 유도값 | **확인**: `run_base_experiments.py:2584` `_patch_ratio = max(_patch_ratio, 0.001)`, `:2585` `grl_pos_weight = (1-_patch_ratio)/_patch_ratio` → 999.0 | **FIXED** — §III-3b 표현 교체 (근거 라인 포함) |
| m3 | MINOR | eval_interval 실구동은 스크립트 상수 | **확인**: `run_base_experiments.py:94` `EVAL_INTERVAL = 5` (grep -n 실측; 값은 config와 일치) | **FIXED** — §VIII Eval interval 행에 각주 추가 |
| m4 | MINOR | `teacher_warmup_early_stop_metric` 코드 전체 무참조 | **확인**: grep 전수 — `config.py:289` 정의 1건뿐 (mae_anomaly/ + scripts/ 전체) | **FIXED** — §VII #26 신설 (NOT-IMPL급 dead 필드) |
| m5 | MINOR | `use_sliding_window_dataset`/`random_seed`/`device` 판정 행 없음 | 확인 (§VI/§VII 부재) | **FIXED** — §VI에 운영 키 판정 행(ACTIVE, 서술적) 신설 |

---

## 리뷰 정확성 메모 (실체 영향 없음, 기록용)

1. **V2-B3 적용부 경로 표기**: 리뷰는 `exp_P1_tri_routing.py:104` 등으로 표기했으나 실제 파일은 `q3_exploration/experiments/` 서브디렉토리 하위. 라인·실체는 정확 (`experiments/exp_P14_boundary_refinement.py:147` `gauss(base_unsmoothed, 10)` 직접 실측 일치). 문서에는 정확한 경로로 반영.
2. **off-by-one 라인**: V1-M4/M5가 지칭한 문서 인용부의 실제 위치는 config-validation override = trainer.py:49-54 (문서 표기 50-55), ValueError guard = trainer.py:74-78 (문서 표기 75-79). 리뷰·문서 모두 ±1 어긋났으나 해당 코드 블록 식별은 동일 — 문서에는 실측 라인(49-54 / 74-78)으로 기재.

## 문서 메타 처리

- frontmatter `last_modified: 2026-06-10` (당일 재확인 — 유지).
- 문서 상단 정정 이력에 r2 블록 추가, 기존 부록을 "부록 1 (r1)"로 개칭, "부록 2: r2 정정 목록 (24항)" 신설.
- `paper_legacy/` 미접촉, 코드·실험 환경 무수정 (읽기 전용 검증만 수행).
