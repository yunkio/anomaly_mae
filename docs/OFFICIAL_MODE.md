# Official MAE-mode (`official=True`)

> 2026-06-22 신설. `official=True`는 **별도 코드 경로**로 MAE식 학습/평가 번들을 적용한다.
> `official=False`(기본)는 기존 코드와 **byte-identical**. 모든 신규 동작은 `if getattr(config,'official',False)` 가드 뒤에 있고, `apply_official_overrides()`는 `if not official: return config`로 단락한다.
>
> 핵심 검증(요약): 실제 271 경로(Set C preset + 271 `config_override`) vs official 경로(`CANON_271`)를 전 dataclass 필드 1:1 비교 → **150/155 필드 동일, 의도한 5필드만 차이**. 단일 `random_seed`로 model init·torch·numpy·random·DataLoader generator 전부 일괄 제어(실증).

---

## 1. 개요

`official=True`이면:

1. **271(canonical)을 기준(default)** 으로 깔고 → 사용자가 `config_override`로 명시한 key가 그 위를 덮고 → **official 번들이 최우선**으로 적용된다(아래 §3).
2. 학습/평가/시각화/저장/seed 동작 일부가 MAE식 별도 경로로 바뀐다.
3. 언급하지 않은 모든 config은 **정확히 271 값**으로 동작한다(§4, §5).

---

## 2. 사용법

### 2.1 실행
```bash
python scripts/run_base_experiments.py --set C --no-wait \
  --dataset <KEY> [<KEY> ...] \
  --output-base <DIR> \
  --config-override "official=True [<추가 override> ...]"
```
- `--set C` 는 필수(argparse). official이면 Set C preset 기하는 무시되고 271 기하가 적용된다(271이 자체 공급).
- `--dataset` 으로 대상 데이터셋 선택. 미지정 시 전체.

### 2.2 Override 가능한 항목
official 번들 중 **강제(forced)** 항목은 사용자가 명시해도 official 값이 이긴다. 단, 아래 둘은 **override 가능**:

| 항목 | official default | override |
|---|---|---|
| `num_epochs` | **30** | `num_epochs=N` 명시 시 N |
| `teacher_only_warmup_epochs` | **`num_epochs // 2`** (예: 30→15) | `teacher_only_warmup_epochs=M` 명시 시 M |

그 외 271 기준 config은 전부 기존 방식대로 개별 override 가능(예: `use_scad=True`, `batch_size=64` 등). 예:
```
# epoch만 40으로 (warmup 자동 20)
--config-override "official=True num_epochs=40"
# epoch 40 + warmup 10
--config-override "official=True num_epochs=40 teacher_only_warmup_epochs=10"
# 271에서 SCAD만 켜기
--config-override "official=True use_scad=True scad_form=C"
```

### 2.3 체크포인트 보존 옵션 (per-dataset, 전역 fallback)
| config | 기본 | 의미 |
|---|---|---|
| `official_keep_checkpoints` | `True` | 전역. True=`official_epochs/`(매 epoch) + best/last 체크포인트 보존 |
| `official_ckpt_overrides` | `''` | per-dataset 예외, `'key1:false,key2:true'` (미명시 dataset은 전역) |

- `False`(**저장 안함**): `official_epochs/` writes를 **skip**, **eval·시각화는 정상 수행**, 끝나면 `best_model/best_checkpoint/latest`를 **삭제**(save_weights·KEEP_CHECKPOINT_DATASETS 무관). 결과(`epoch_metrics.json`·`epoch_scores/*.npz`·시각화 PNG)만 남는다. 디스크: dataset당 **3.3GB → 13MB** (스모크 실측).
```
# 전역 저장안함
--config-override "official=True official_keep_checkpoints=false"
# 전역 저장안함 + BASE4만 보존
--config-override "official=True official_keep_checkpoints=false official_ckpt_overrides=SWaT_A1A2:true,WaDi_A1:true,WaDi_A2:true,PSM:true"
# 전역 저장 + 특정 dataset만 저장안함
--config-override "official=True official_ckpt_overrides=MSL_simple_T-13:false"
```
⚠️ `official_keep_checkpoints=True`(기본)에서 `official_epochs/`는 d_model=512 기준 **~3.3GB/dataset**(138MB×25). 대량 실행 시 디스크 주의.

### 2.4 Seed
- 기본 `random_seed = 42`.
- 변경: `--config-override "official=True random_seed=123"` (random_seed는 강제 항목이 아니라 그대로 적용).
- **단일 knob**: `random_seed` 하나로 model init·torch·numpy·random·DataLoader generator·마스킹·샘플 순서가 전부 일괄 바뀌고, 같은 값이면 완전 재현. `cudnn.benchmark=True` 유지·`use_deterministic_algorithms` 미호출(속도 보존).

---

## 3. `official=True`가 하는 일 (구현 상세)

### 3.1 강제 항목 — `apply_official_overrides()` ([config.py](../mae_anomaly/config.py))
official=True일 때 **마지막 writer**로 강제(사용자/preset/271 무관):

| config | 271 | official | 이유 |
|---|---|---|---|
| `epoch_offset` | True | **False** | train epoch-offset 제거 |
| `sliding_window_stride` | 21 | **1** | train stride=1 (run_base가 **로컬** train_stride도 1로 강제 — 데이터셋이 config 필드가 아닌 로컬 변수를 읽으므로 둘 다 필요) |
| `use_teacher_warmup_early_stop` | False | **False** | 고정 warmup 모드(런타임 단축 방지). 271도 False라 실질 동일 |

`num_epochs`(30)·`teacher_only_warmup_epochs`(num_epochs//2)는 **강제가 아니라** run_base의 official 빌드에서 **user merge 전 default**로 적용(§2.2 override 가능).

### 3.2 코드 경로 차이 (config 필드 아님 — `config.official` 가드로만 동작)
| # | 동작 | 위치 |
|---|---|---|
| LR | **per-iteration LR** (MAE `util/lr_sched.py`): 0→선형 warmup over `w=teacher_only_warmup_epochs`, 이후 half-cosine→`min_lr=0` over `[w, num_epochs)`, `e=epoch+batch/len(loader)`. group별 peak-LR 캡처 후 비율 스케일(GRL-cls lr 비율 보존). per-epoch `scheduler.step()` skip | `trainer.py _official_lr_now` + 배치 루프 |
| 저장 | **매 epoch model-only 체크포인트** → `official_epochs/epoch_NNN.pt` (별도 namespace, `official_keep_checkpoints`로 on/off) | `run_base post_epoch_save_callback` |
| eval | **eval_interval=1** (per-experiment 로컬; 전역 `EVAL_INTERVAL=5`는 미변경) | `run_base` 로컬 |
| score | **causal/online anomaly score** (단일출처 `scoring.py`): seed `R_tr=Σrecon_tr[정상]`,`D_tr=Σdisc_tr[정상]`; `s_t=(R_tr+cumsum recon)/(D_tr+cumsum disc+1e-8)`; `score_t=recon_t+0.25·disc_t·s_t`. prefix-only cumsum=미래·라벨 미사용. 매 epoch train-inference로 `R_tr/D_tr` 산출 → best-epoch을 이 점수의 `pak_auc_f1`로 선택. 최종 metric·VUS·excl22·viz **전부 causal로 일관**. npz에 `official_score` key 추가(`adaptive_score` 보존) | `scoring.compute_official_causal_score`, `run_base _evaluate_all_parallel`/`_official_train_seed` |
| viz | best epoch 기준 + 점수는 best-epoch `official_score`(full point)로 표시 | `run_base derive_pred_data 직후` |
| seed | 단일 전역 seed(§2.4) | `config.set_seed_official` + DataLoader |

### 3.3 메커니즘 — 271-base 레이어링 + 단일 깔때기
- `make_config()` ([utils/experiment.py](../mae_anomaly/utils/experiment.py)) 가 queue/CLI 양쪽이 거치는 **유일한** 학습 config 빌더. merge loop 직후·일관성 검증 직전에 `apply_official_overrides(config)` 호출 → official이 last-writer.
- run_base의 official 빌드: `overrides = CANON_271`(완전한 271 dict) → `num_epochs=30` default → **사용자 명시 key 덮기** → `teacher_only_warmup_epochs = num_epochs//2`(미명시 시) → `official=True`. 즉 **271-base < 사용자 명시 < official 강제**.
- `CANON_271` = exp271 `config_override` + Set C 기하(seq=500/p=10/np=50)를 그대로 인코딩한 dict (단일 출처; 271 변경 시 동기화 필요).

---

## 4. official=True 사용 시 적용되는 전체 config (ACTIVE)

> 아래는 **추가 override 없이** `official=True`만 줬을 때 실제 적용되는 값. (num_features/sliding_window_train_ratio/device는 dataset·환경 의존이라 제외.)

**[official 전용]**
| config | 값 | 비고 |
|---|---|---|
| `official` | `True` | 제어 플래그 |
| `num_epochs` | `30` | override 가능 |
| `teacher_only_warmup_epochs` | `15` | =num_epochs//2, override 가능 |
| `sliding_window_stride` | `1` | 강제 (+ 로컬 train_stride=1) |
| `epoch_offset` | `False` | 강제 |
| `use_teacher_warmup_early_stop` | `False` | 강제 |
| `min_lr` | `0.0` | per-iter LR cosine floor (active) |
| `official_keep_checkpoints` | `True` | active |
| `official_ckpt_overrides` | `''` | active |
| (effective) `eval_interval` | **`1`** | per-experiment 로컬(아래 §5의 config 필드 5는 무시) |

**[아키텍처 — 271 그대로, active]**
| config | 값 |
|---|---|
| `d_model` / `nhead` | 512 / 8 |
| `num_encoder_layers` / `num_teacher_decoder_layers` / `num_student_decoder_layers` | 4 / 3 / 2 |
| `dim_feedforward` / `dropout` | 2048 / 0.15 |
| `seq_length` / `patch_size` / `num_patches` | 500 / 10 / 50 |
| `patchify_mode` / `mask_after_encoder` | 'linear' / True |
| `use_transformer_encoder_decoder` / `use_flatten_linear_embedding` | True / True |
| `shared_mask_token` / `num_shared_decoder_layers` / `decoder_half_dim` | False / 0 / False |

**[데이터/마스킹 — active]**
| config | 값 |
|---|---|
| `normalize_mode` / `minmax_range` | 'minmax' / '0_1' |
| `minmax_clamp_min` / `minmax_clamp_max` | -4.0 / 4.0 |
| `masking_ratio` | 0.15 |
| `force_mask_anomaly` / `force_mask_all_anomaly` | True / False |
| `sliding_window_test_stride` | -1 → 49 (=W//10−1, 추론 stride) |
| `use_sliding_window_dataset` / `use_masking` | True / True |

**[손실 — active]**
| config | 값 | 비고 |
|---|---|---|
| `use_output_discrepancy` / `use_discrepancy_loss` | True / True | normal_loss(=OD normal) active |
| `normal_loss_weight` | 1.0 | active |
| `lambda_disc` | 2.0 | |
| `patch_level_loss` | True | |
| `use_teacher` / `use_student` | True / True | |

**[Feature Matching — active]**
| config | 값 | 비고 |
|---|---|---|
| `use_feature_matching` | True | |
| `fm_adaptive_lambda` | True | **legacy grad-norm-ratio λ** 경로(prev-epoch carry) |
| `fm_distance_metric` / `fm_loss_weight` | 'l2' / 1.0 | |
| `fm_balance_mode` | 'none' | =legacy (relobralo/famo/uwso 아님) |

**[GRL — active]**
| config | 값 | 비고 |
|---|---|---|
| `use_grl` / `grl_mode` | True / 'classifier' | |
| `grl_loss_weight` / `grl_target_mode` | 0.2 / 'window' | |
| `grl_pos_weight` / `grl_use_focal` | 19.0 / True | |
| `grl_balanced_sampling` | False | |
| `grl_cls_lr_ratio` / `grl_attach_layer` | 0.1 / 'last' | |
| `grl_adaptive_lambda` | True | prev-epoch carry λ |
| `grl_disable_anomaly_loss` | True | → OD anomaly_loss 비활성(아래 §5) |
| `loss_balance_mode` | 'adaptive_lambda_legacy' | =legacy(grad-norm-ratio) |
| `grl_cls_hidden` / `grl_cls_arch` | 0 / 'default' | hidden=0 → linear head |

**[학습 — active]**
| config | 값 | 비고 |
|---|---|---|
| `batch_size` | 1024 | override 가능 |
| `learning_rate` | 0.001 | per-iter LR peak |
| `weight_decay` | 0.001 | AdamW(betas 0.9/0.99) |
| `use_amp` / `amp_dtype` | True / 'bf16' | |
| `best_epoch_metric` | 'pak_auc_f1' | causal 점수에 적용 |
| `random_seed` | 42 | 단일 knob |
| `anomaly_score_mode` | 'adaptive' | **진단용** adaptive_score 계산에만(주 점수=causal). §5 참조 |

---

## 5. 무시되는 config (특정 코드경로 하위라 효과 없음)

> 아래는 official(=271 base) 실행 시 **해당 기능 플래그가 off이거나 다른 경로가 대체**해서 **효과가 없는** config들. 271에서 이미 inactive인 것 + official이 새로 대체하는 것 둘 다 포함.

### 5.1 official이 새로 대체/변경해서 무시되는 것
| config | 값(무시됨) | 무시 이유 / 대체 경로 |
|---|---|---|
| `eval_interval` (필드) | 5 | official은 **per-experiment 로컬 `eval_interval=1`** 사용. 전역/필드값은 게이트에서 안 읽음 → effective=1 |
| `warmup_epochs` | 10 | **LR warmup horizon(legacy)**. official은 per-iteration LR이 `teacher_only_warmup_epochs`를 w로 사용. legacy `LinearLR+CosineAnnealingLR+SequentialLR`은 생성되나 `scheduler.step()` skip + 매 배치 `pg['lr']` 덮어씀 → warmup_epochs 효과 0 |
| `score_recon_disc_ratio` | 4.0 | 주 점수=causal(w=0.25 **하드코딩**). 이 값은 진단용 `adaptive_score` npz 계산에만 영향, **보고 metric/best-epoch/viz엔 영향 없음** |
| `eval_disc_weight` / `eval_fm_weight` | -1.0 / -1.0 | 동일(진단 adaptive_score에만). causal 점수는 미사용 |
| `anomaly_score_mode='adaptive'` | adaptive | adaptive 점수는 계산·npz 저장(진단)되지만 **best-epoch·metric·viz는 causal `official_score`** 사용 |

### 5.2 epoch_offset off로 무시
| config/코드 | 무시 이유 |
|---|---|
| epoch-offset 증강 + 하드코딩 seed `np.random.RandomState(42+cycle)` ([trainer.py](../mae_anomaly/trainer.py) `_epoch_offset_for`) | `epoch_offset=False`라 호출 자체가 안 됨(stride=1) |

### 5.3 OD anomaly-loss 게이트로 무시 (`use_grl=True` + `grl_disable_anomaly_loss=True` → `disable_anomaly_loss=True`, [loss.py](../mae_anomaly/loss.py))
| config | 값 | 무시 이유 |
|---|---|---|
| `anomaly_loss_weight` | 2.0 | OD anomaly_loss(maximize)가 GRL로 대체되어 **0** → weight 효과 없음 |
| `anomaly_loss_direction` | 'maximize' | 동일(anomaly_loss 비활성) |
| `margin` / `margin_type` / `dynamic_margin_k` | 0.5 / 'dynamic' / 6 | `_compute_(patch_)anomaly_loss`가 호출 안 됨(disable_anomaly_loss) → 무시. (271도 동일) |

### 5.4 기능 플래그 off라 하위 config 전부 무시
| off 플래그 | 무시되는 config |
|---|---|
| `use_scad=False` | `scad_form, scad_d_proj, scad_temperature, scad_margin, scad_gamma, scad_one_sided, scad_loss_weight, scad_adaptive_lambda, scad_ramp_up, scad_patch_label_mode, scad_use_memory_bank, scad_memory_bank_size, scad_proj_head_arch, scad_apply_space` |
| `use_discriminator=False` (271은 GRL 사용, GAN disc 아님) | `d_grad_student_layers, disc_lr_ratio, adaptive_lambda, adv_loss_weight, disc_warmup_epochs, disc_channels` |
| `grl_mode='classifier'` (WDGRL 아님) | `wdgrl_k_critic, wdgrl_gp_weight, wdgrl_critic_lr` |
| `loss_balance_mode='adaptive_lambda_legacy'` (신규 밸런서 아님) | `mse_norm_ema_beta, mse_norm_eps, mse_norm_log_variant, dann_ramp_gamma, dann_ramp_horizon, relobralo_T, relobralo_alpha, relobralo_rho, relobralo_eps, relobralo_update_freq, famo_gamma, famo_w_lr, famo_max_norm, famo_reforward, uwso_temperature, uwso_loss_floor_mse, uwso_loss_floor_bce, uwso_ema_beta, fixed_grl_weight` |
| `fm_balance_mode='none'` (FM은 legacy λ) | `fm_uwso_temperature, fm_uwso_loss_floor, fm_uwso_ema_beta` |
| `use_teacher_output_ema=False` | `teacher_output_ema_momentum` |
| `use_teacher_warmup_early_stop=False` (official 강제) | `teacher_warmup_early_stop_patience, teacher_warmup_early_stop_min_epochs, teacher_warmup_early_stop_metric, teacher_warmup_es_check_interval, teacher_warmup_es_patience_checks, teacher_warmup_es_relative_threshold, teacher_warmup_es_min_epoch` |
| `use_revin=False` | `revin_affine, revin_eps, revin_visible_only` |
| `freeze_teacher_after_warmup=False`, `freeze_encoder_only=False` | (서브 config 없음 — 플래그 자체 inactive) |
| `patchify_mode='linear'` (CNN 아님) | `cnn_channels, cnn_kernel_size` |
| `masking_ratio_anneal=False`, `masking_ratio_min/max=-1` | masking ratio range/anneal 무시 → 고정 `masking_ratio=0.15` |
| `eval_complementary_masking=False` | `eval_complementary_k` |
| `student_recon_weight=0.0` | student recon 항이 0 가중 → 손실 기여 없음 |
| `force_mask_all_anomaly=False` | per-sample 전체-anomaly 마스킹 비활성(고정 budget) |

> 위 "무시" 항목들은 official 고유가 아니라 **271 base의 구성(GRL on, SCAD/disc/EMA/balancer off 등)에서 비롯**된 것이 대부분이다. official은 그중 `eval_interval`·`warmup_epochs`·adaptive-score 파라미터를 **새 경로(per-iter LR / eval=1 / causal score)로 추가 대체**한다.

---

## 6. 검증 요약 (2026-06-22)
- **official=False byte-identity**: 모든 신규 동작 가드/단락 + 삭제 라인 전수감사(의도된 교체만) + `make_config(official=False)==hand-built` 단위테스트.
- **271 재현 strict diff**: 실제 271 경로 vs official 경로 전 필드 비교 → **150/155 동일**, 의도한 5필드만 차이(`official`, `num_epochs 500→30`, `teacher_only_warmup_epochs 250→15`, `epoch_offset True→False`, `sliding_window_stride 21→1`).
- **override**: `num_epochs`/`teacher_only_warmup_epochs` 5케이스(30/15·40/20·40/10·30/5·100/50) 통과 + 강제항목 유지.
- **causal score**: 수동계산 일치 + **t=5 perturb 시 score[0..4] 불변=미래미사용 경험증명** + TypeError 가드 + NaN 무전파.
- **단일 seed**: 같은 seed=재현 / 다른 seed=전부 변경(model init·torch·numpy·random·generator).
- **per-iter LR**: MAE 공식 전(epoch,batch) 일치, group 비율 보존.
- **GPU e2e 스모크**(MSL T-13): 전 파이프라인 완주, official 산출물 확인, 동시 실행 실험 무영향.
- **저장안함 스모크**: official_epochs 미생성, eval+viz 유지, best_* 삭제, 3.3GB→13MB.

---

## 7. 코드 위치
| 요소 | 파일·심볼 |
|---|---|
| 플래그·CANON_271·강제·seed | `mae_anomaly/config.py`: `official`, `min_lr`, `official_keep_checkpoints`, `official_ckpt_overrides`, `CANON_271`, `apply_official_overrides`, `set_seed_official`, `official_worker_init_fn` |
| 깔때기 | `mae_anomaly/utils/experiment.py`: `make_config` (apply_official_overrides 호출) |
| causal score | `mae_anomaly/scoring.py`: `compute_train_normal_seed`, `compute_official_causal_score` (w=0.25, eps=1e-8 하드코딩) |
| per-iteration LR | `mae_anomaly/trainer.py`: `_official_lr_now`, 배치 루프 LR 적용, `scheduler.step()` 가드 |
| 271-base 빌드·로컬 stride·eval_interval·매-epoch 저장·per-epoch train-inference·causal 치환·npz·viz·seeding·저장안함 cleanup | `scripts/run_base_experiments.py`: parser user-keys, official overrides 빌드, `_official_keep_ckpt_for`, `_official_train_seed`, `_evaluate_all_parallel(official_seed=)`, `compute_epoch_test_eval`, `post_epoch_save_callback`, end-cleanup |

상세 변경 이력: [CHANGELOG.md](CHANGELOG.md) 2026-06-22 항목 / 아키텍처: [ARCHITECTURE.md](ARCHITECTURE.md) "Official MAE-mode".
