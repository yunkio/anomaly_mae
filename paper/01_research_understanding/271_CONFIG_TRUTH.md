---
phase: 1
agent: config-forensics
directives: [R17, R28, R34]
last_modified: 2026-06-11
---

# Config 271 — Ground Truth Report

> **2026-06-10 reconciler 정정 (r1): 3건 (test-stride 산식, masking 개수, GRL 방향 서술) — 수정 목록은 부록 1.**
> **2026-06-11 fixer 보강 (r4): Phase 3 재리뷰 escalation 2건 — ① GRL 이중 λ 구조 등재 (손실 가중치 λ_GRL grad-ratio×0.2 + 반전 계수 λ_rev Ganin sigmoid ramp — 둘 다 271 활성, §VIII GRL Details), ② teacher-only warmup 중 학습 경로 student decoder forward skip 등재 (§VIII Training). 수정 목록은 부록 4, 처리표는 `paper/99_reviews/p3_fixlog_r3.md`.**
> **2026-06-10 fixer-1 정정 (r2): verifier-1/2 리뷰 반영 23건 — 근거 오류 교정(masking annealing·complementary masking·freeze gate 라인), §VIII 오판정 2건 정정(total length 275K, anomaly-loss ramp no-op), Gaussian smoothing R34 등재 형식 교정, 판정 누락 옵션 등재(lambda_disc·minmax_clamp·anomaly_interval_scale·sliding_window_total_length·bare adaptive_lambda 외). 수정 목록은 부록 2, 처리표는 `paper/99_reviews/p1_271truth_fixlog_r2.md`.**
> **2026-06-10 fixer-5 정정 (r3): 재리뷰 α(r2) 발견 4건 — `lambda_disc` "유일 소비처/271 dead" 허위 구조 진술 정밀 재서술(α-B1: score-path dead + 진단 CSV 소비처 명시), GRL classifier "1-layer MLP"→"2-layer MLP" 2개소(α-B2), §VIII GRL adaptive lambda "(VQGAN-style)" 귀속 제거→trainer inline grad-ratio(α-B3), freeze 부기 라인 ±1 교정(α-m1). 수정 목록은 부록 3, 처리표는 `paper/99_reviews/p1_fixlog_r3.md`.**

Experiment directory: `271_20260602_020545_271canon_baseline`
All claims below are derived directly from code (`mae_anomaly/`) and metadata JSON files.
No inference from manuscript text.

---

## I. Metadata Collection Result

**Total `experiment_metadata.json` files found: 37**

Full path list (relative to experiment root):

```
MSL/C-1/experiment_metadata.json
MSL/C-2/experiment_metadata.json
MSL/F-7/experiment_metadata.json
MSL/P-11/experiment_metadata.json
MSL/T-13/experiment_metadata.json
PSM/experiment_metadata.json
SMAP/G-7/experiment_metadata.json
SMAP/P-1/experiment_metadata.json
SMAP/P-4/experiment_metadata.json
SMAP/T-1/experiment_metadata.json
SMAP/T-3/experiment_metadata.json
SMD/machine-1-2/experiment_metadata.json
SMD/machine-1-3/experiment_metadata.json
SMD/machine-1-4/experiment_metadata.json
SMD/machine-1-5/experiment_metadata.json
SMD/machine-1-6/experiment_metadata.json
SMD/machine-1-7/experiment_metadata.json
SMD/machine-1-8/experiment_metadata.json
SMD/machine-2-1/experiment_metadata.json
SMD/machine-2-3/experiment_metadata.json
SMD/machine-2-4/experiment_metadata.json
SMD/machine-2-5/experiment_metadata.json
SMD/machine-2-6/experiment_metadata.json
SMD/machine-2-7/experiment_metadata.json
SMD/machine-3-1/experiment_metadata.json
SMD/machine-3-10/experiment_metadata.json
SMD/machine-3-2/experiment_metadata.json
SMD/machine-3-3/experiment_metadata.json
SMD/machine-3-4/experiment_metadata.json
SMD/machine-3-5/experiment_metadata.json
SMD/machine-3-6/experiment_metadata.json
SMD/machine-3-7/experiment_metadata.json
SMD/machine-3-9/experiment_metadata.json
SWaT/A1A2_excl22/experiment_metadata.json
SWaT/A1A2_full/experiment_metadata.json
WaDi/A1/experiment_metadata.json
WaDi/A2/experiment_metadata.json
```

**Dataset structure note:**
- PSM: 1 entity (dataset root)
- SMAP: 5 channel entities
- MSL: 5 channel entities
- SMD: 22 machine entities (machine-1-*, machine-2-*, machine-3-*)
- SWaT: 2 evaluation conditions (A1A2_full, A1A2_excl22) — same trained model, different eval masks (R28)
- WaDi: 2 attack scenarios (A1, A2)

**Note on entity count:** Count confirmed at 37 (orchestrator prior estimate와 일치; 신규 개체 없음).

---

## II. Canonical Config (Common to All 37 Entities)

These 114 keys are **identical** across all 37 metadata files.

| Key | Value |
|-----|-------|
| `adaptive_lambda` | `True` |
| `adv_loss_weight` | `1.0` |
| `amp_dtype` | `'bf16'` |
| `anomaly_interval_scale` | `0.75` |
| `anomaly_loss_direction` | `'maximize'` |
| `anomaly_loss_weight` | `2.0` |
| `anomaly_score_mode` | `'adaptive'` |
| `batch_size` | `1024` |
| `best_epoch_metric` | `'pak_auc_f1'` |
| `cnn_channels` | `None` |
| `cnn_kernel_size` | `3` |
| `d_grad_student_layers` | `'all'` |
| `d_model` | `512` |
| `device` | `'cuda'` |
| `dim_feedforward` | `2048` |
| `disc_channels` | `[64, 32]` |
| `disc_lr_ratio` | `4.0` |
| `disc_warmup_epochs` | `10` |
| `dropout` | `0.15` |
| `dynamic_margin_k` | `6` |
| `epoch_offset` | `True` |
| `eval_complementary_k` | `7` |
| `eval_complementary_masking` | `False` |
| `eval_disc_weight` | `-1.0` |
| `eval_fm_weight` | `-1.0` |
| `eval_interval` | `5` |
| `fm_adaptive_lambda` | `True` |
| `fm_distance_metric` | `'l2'` |
| `fm_loss_weight` | `1.0` |
| `force_mask_anomaly` | `True` |
| `freeze_encoder_only` | `False` |
| `freeze_teacher_after_warmup` | `False` |
| `grl_adaptive_lambda` | `True` |
| `grl_balanced_sampling` | `False` |
| `grl_cls_arch` | `'default'` |
| `grl_cls_hidden` | `0` |
| `grl_cls_lr_ratio` | `0.1` |
| `grl_disable_anomaly_loss` | `True` |
| `grl_loss_weight` | `0.2` |
| `grl_mode` | `'classifier'` |
| `grl_use_focal` | `True` |
| `grl_target_mode` | `'window'` |
| `lambda_disc` | `2.0` |
| `learning_rate` | `0.001` |
| `margin` | `0.5` |
| `margin_type` | `'dynamic'` |
| `mask_after_encoder` | `True` |
| `masking_ratio` | `0.15` |
| `masking_ratio_anneal` | `False` |
| `masking_ratio_max` | `-1.0` |
| `masking_ratio_min` | `-1.0` |
| `minmax_clamp_max` | `4.0` |
| `minmax_clamp_min` | `-4.0` |
| `minmax_range` | `'0_1'` |
| `nhead` | `8` |
| `normal_loss_weight` | `1.0` |
| `normalize_mode` | `'minmax'` |
| `num_encoder_layers` | `4` |
| `num_epochs` | `500` |
| `num_patches` | `50` |
| `num_shared_decoder_layers` | `0` |
| `num_student_decoder_layers` | `2` |
| `num_teacher_decoder_layers` | `3` |
| `patch_level_loss` | `True` |
| `patch_size` | `10` |
| `patchify_mode` | `'linear'` |
| `random_seed` | `42` |
| `revin_affine` | `True` |
| `revin_eps` | `1e-05` |
| `revin_visible_only` | `False` |
| `scad_adaptive_lambda` | `True` |
| `scad_d_proj` | `128` |
| `scad_form` | `'A'` |
| `scad_loss_weight` | `0.5` |
| `scad_margin` | `0.3` |
| `scad_memory_bank_size` | `1024` |
| `scad_patch_label_mode` | `'patch'` |
| `scad_proj_head_arch` | `'default'` |
| `scad_ramp_up` | `'sigmoid'` |
| `scad_temperature` | `0.1` |
| `scad_use_memory_bank` | `False` |
| `score_recon_disc_ratio` | `4.0` |
| `seq_length` | `500` |
| `shared_mask_token` | `False` |
| `sliding_window_stride` | `21` |
| `sliding_window_test_stride` | `-1` |
| `sliding_window_total_length` | `275000` |
| `student_recon_weight` | `0.0` |
| `teacher_only_warmup_epochs` | `250` |
| `teacher_output_ema_momentum` | `0.996` |
| `teacher_warmup_early_stop_metric` | `'recon_snr'` |
| `teacher_warmup_early_stop_min_epochs` | `50` |
| `teacher_warmup_early_stop_patience` | `10` |
| `use_amp` | `True` |
| `use_discrepancy_loss` | `True` |
| `use_discriminator` | `False` |
| `use_feature_matching` | `True` |
| `use_flatten_linear_embedding` | `True` |
| `use_grl` | `True` |
| `use_masking` | `True` |
| `use_output_discrepancy` | `True` |
| `use_revin` | `False` |
| `use_scad` | `False` |
| `use_sliding_window_dataset` | `True` |
| `use_student` | `True` |
| `use_teacher` | `True` |
| `use_teacher_output_ema` | `False` |
| `use_teacher_warmup_early_stop` | `False` |
| `use_transformer_encoder_decoder` | `True` |
| `warmup_epochs` | `10` |
| `wdgrl_critic_lr` | `0.0001` |
| `wdgrl_gp_weight` | `10.0` |
| `wdgrl_k_critic` | `5` |
| `weight_decay` | `0.001` |

---

## III. Per-Dataset / Per-Entity Varying Keys

Exactly **3 keys** differ across entities. These are expected dataset-specific values, not config inconsistencies.

### 3a. `num_features` (dataset input dimensionality)
| Dataset | Value |
|---------|-------|
| MSL (all 5 entities) | 55 |
| SMAP (all 5 entities) | 25 |
| PSM | 25 |
| SMD/machine-1-2 | 33 |
| SMD/machine-1-3 | 34 |
| SMD/machine-1-4 | 34 |
| SMD/machine-1-5 | 30 |
| SMD/machine-1-6 | 34 |
| SMD/machine-1-7 | 34 |
| SMD/machine-1-8 | 30 |
| SMD/machine-2-1 | 33 |
| SMD/machine-2-3 | 32 |
| SMD/machine-2-4 | 30 |
| SMD/machine-2-5 | 32 |
| SMD/machine-2-6 | 30 |
| SMD/machine-2-7 | 31 |
| SMD/machine-3-1 | 31 |
| SMD/machine-3-10 | 29 |
| SMD/machine-3-2 | 32 |
| SMD/machine-3-3 | 36 |
| SMD/machine-3-4 | 30 |
| SMD/machine-3-5 | 30 |
| SMD/machine-3-6 | 31 |
| SMD/machine-3-7 | 33 |
| SMD/machine-3-9 | 31 |
| SWaT (both conditions) | 45 |
| WaDi/A1 | 123 |
| WaDi/A2 | 123 |

### 3b. `grl_pos_weight` (GRL focal-BCE class-imbalance weight)
Varies per entity, reflecting each entity's actual normal/anomaly ratio.
Computed automatically from actual dataset statistics at run time.
Range: 3.14 (SMAP/T-1) to 999.0 (SMD/machine-1-5).
999.0은 cap/sentinel이 아니라 **patch-ratio 하한에서 유도된 값**: `_patch_ratio = max(_patch_ratio, 0.001)` →
`(1 − 0.001) / 0.001 = 999.0` (`run_base_experiments.py:2584-2585`).
SWaT both conditions share: 59.18.

### 3c. `sliding_window_train_ratio` (fraction of total length used for training)
Varies per entity due to dataset-specific train/test split lengths.
SMD entities: ~0.75; PSM: 0.8007; SMAP entities: 0.617–0.626 (G-7: 0.617; P-1/P-4/T-1/T-3: ~0.625);
MSL entities: 0.635–0.764; SWaT (both): 0.762; WaDi/A1: 0.937; WaDi/A2: 0.910.

---

## IV. SWaT Dual-Condition Confirmation (R28)

- `SWaT/A1A2_full`: `swat_eval_mode = None` (full test set including anomaly region 22+)
- `SWaT/A1A2_excl22`: `swat_eval_mode = 'excl22'` (region 22+ excluded from evaluation)

Both share **identical** config (same `num_features=45`, same `grl_pos_weight=59.18`,
same `sliding_window_train_ratio=0.762`). They are the **same trained model** evaluated
under two different anomaly-mask conditions (timing.wall_time 동일 — metadata 실측). This confirms R28: the dominant anomaly
region 22+ makes the full-set metric incomparable, so the excl22 metric is reported
as a separate indicator.

> **운영 주의 (2026-06-10 r2 추가)**: config 키 `best_epoch_metric`은 두 condition 모두 `'pak_auc_f1'`로 저장되어 있으나,
> `SWaT/A1A2_excl22`의 **`timing.best_epoch_metric`은 `'excl22_pak_auc_f1'`** (A1A2_full은 `'pak_auc_f1'`) — metadata 실측.
> 즉 excl22 개체의 best-epoch 선택은 excl22-마스킹된 PA%K AUC F1로 수행되었다. config 키는 템플릿 값이고,
> 런타임 평가가 excl22 condition에서 metric 이름을 derived metric으로 override한다. 재현·서술 시 이 구분을 유지할 것.

---

## V. Blockers

**0 blockers.** All 114 common keys are identical across all 37 entities.
The 3 varying keys (`num_features`, `grl_pos_weight`, `sliding_window_train_ratio`)
are legitimately dataset/entity-specific and expected to differ.

---

## VI. Used vs. Unused Component Table

Each row: component | relevant config key(s) + value | active/inactive in 271 | code evidence

| Component | Config Key(s) & Value | Status | Code Evidence (file:line) |
|-----------|----------------------|--------|--------------------------|
| Teacher encoder (Transformer, 4 layers) | `use_teacher=True`, `num_encoder_layers=4` | **ACTIVE** | `model.py:359-362` — `nn.TransformerEncoder(encoder_layer, num_layers=config.num_encoder_layers)` |
| Teacher decoder (Transformer, 3 layers, self-attn) | `use_teacher=True`, `num_teacher_decoder_layers=3`, `use_transformer_encoder_decoder=True` | **ACTIVE** | `model.py:419-423` — `nn.TransformerEncoder(teacher_decoder_layer, num_layers=config.num_teacher_decoder_layers)` (게이트 `if config.use_teacher:` model.py:406) |
| Student decoder (Transformer, 2 layers, self-attn) | `use_student=True`, `num_student_decoder_layers=2`, `use_transformer_encoder_decoder=True` | **ACTIVE** | `model.py:457-461` — `nn.TransformerEncoder(student_decoder_layer, num_layers=config.num_student_decoder_layers)` (게이트 `if config.use_student:` model.py:444) |
| Linear patch embedding (`patchify_mode='linear'`) | `patchify_mode='linear'`, `use_flatten_linear_embedding=True` | **ACTIVE** | `model.py:580` — `if self.patchify_mode == 'patch_cnn':` branch skipped; `model.py:624` — `elif self.patchify_mode == 'linear':` path entered |
| CNN patch embedding | `patchify_mode='linear'` (not 'patch_cnn') | **INACTIVE** | `model.py:580` — `if self.patchify_mode == 'patch_cnn':` branch not entered |
| Patch masking (15%, anomaly-first) | `use_masking=True`, `masking_ratio=0.15`, `force_mask_anomaly=True` | **ACTIVE** | `config.py:315-319` flag; masking applied in `model.py` forward path |
| Mask-after-encoder (standard MAE layout) | `mask_after_encoder=True` | **ACTIVE** | `model.py:1028-1036` (teacher branch) + `model.py:1119-1129` (student branch) — `if self.mask_after_encoder:`; mask tokens inserted before each decoder, encoder sees only visible patches |
| Separate mask tokens (teacher + student) | `shared_mask_token=False` | **ACTIVE** | `model.py:499-505` — `else:` branch creates `teacher_mask_token` and `student_mask_token` separately |
| Shared decoder layers | `num_shared_decoder_layers=0` | **INACTIVE** | `model.py:367-368` — `if self.num_shared_decoder_layers > 0:` not entered; `self.shared_decoder = None` |
| Teacher reconstruction loss | `use_teacher=True`, `use_discrepancy_loss=True` | **ACTIVE** | `loss.py:172-179` — always computed in `forward()` |
| Output discrepancy loss (normal patches only, since GRL disables anomaly_loss) | `use_output_discrepancy=True`, `use_grl=True`, `grl_disable_anomaly_loss=True` | **ACTIVE (normal side only)** | `loss.py:254-261` — `normal_loss` computed; `anomaly_loss = 0.0` at line 261 when `use_grl and grl_disable_anomaly_loss` |
| Anomaly discrepancy loss (push anomaly disc up) | `use_grl=True`, `grl_disable_anomaly_loss=True` | **INACTIVE** | `loss.py:259-261` — `anomaly_loss = torch.tensor(0.0, ...)` when GRL is active |
| GRL classifier (DANN-style, **2-layer MLP** — r3 정정, SYNTHESIS 표A 표기 통일) | `use_grl=True`, `grl_mode='classifier'`, `grl_cls_arch='default'`, `grl_cls_hidden=0` | **ACTIVE** | `model.py:530-538` — `AnomalyClassifierHead` instantiated; `model.py:1150-1154` — called on student hidden; `trainer.py:746-771` — GRL cls loss added to total loss |
| GRL adaptive lambda | `grl_adaptive_lambda=True` | **ACTIVE** | `trainer.py:751-765` — gradient-norm ratio computed each batch; `_prev_epoch_grl_lambda` smoothed |
| GRL focal loss | `grl_use_focal=True` | **ACTIVE** | `loss.py:337-340` — `_p_t = torch.exp(-_bce); _focal = ((1-_p_t)**2.0)*_bce` |
| GRL window-level target | `grl_target_mode='window'` | **ACTIVE** | `loss.py:285-287` — `_window_label = has_anomaly_sample.unsqueeze(1).expand_as(patch_has_anomaly)` |
| GRL balanced sampling | `grl_balanced_sampling=False` | **INACTIVE** | `loss.py:313` — `if self.grl_balanced_sampling:` branch not entered; all patches used with pos_weight |
| GRL WDGRL mode | `grl_mode='classifier'` (not 'wdgrl') | **INACTIVE** | `trainer.py:662` — `if ... _grl_mode == 'wdgrl':` not entered |
| Feature matching loss (L2, adaptive lambda) | `use_feature_matching=True`, `fm_distance_metric='l2'`, `fm_adaptive_lambda=True` | **ACTIVE (training only)** | `loss.py:414-430` — FM computed on normal masked patches; `trainer.py:638-658` — added to loss with prev-epoch adaptive lambda |
| Feature matching in anomaly score | `use_feature_matching=True` (overridden by code) | **INACTIVE** | `scoring.py:237` — `fm_active = False` hardcoded; FM never enters inference score regardless of config |
| Patch-level loss | `patch_level_loss=True` | **ACTIVE** | `loss.py:225-252` — `if self.patch_level_loss:` branch; per-patch disc computed |
| Teacher-only warmup (250 epochs) | `teacher_only_warmup_epochs=250` | **ACTIVE** | `trainer.py:43-44` — warmup epochs set; student frozen during epochs 0–249 (학습 경로에서는 student forward 자체 skip — `model.py:1119`, §VIII Training r4) |
| Teacher warmup early stop | `use_teacher_warmup_early_stop=False` | **INACTIVE** | `trainer.py:485` — `_es_on = getattr(self.config, 'use_teacher_warmup_early_stop', False)` → config=False라 False로 평가; early-stop accumulation never triggered |
| Teacher output EMA | `use_teacher_output_ema=False` | **INACTIVE** | `model.py:514` — `self._has_teacher_output_ema = False`; EMA modules never created |
| RevIN | `use_revin=False` | **INACTIVE** | `model.py:312-314` — `if self.use_revin:` not entered; `self.revin = None` |
| Adversarial discriminator | `use_discriminator=False` | **INACTIVE** | `trainer.py:236-237` — `self.discriminator = None`; D optimizer never created |
| SCAD head | `use_scad=False` | **INACTIVE** | `model.py:541` — `if getattr(config, 'use_scad', False):` not entered; `scad_head` not instantiated; `loss.py:355` — `scad_z is not None and self.use_scad` is False |
| Random masking ratio range | `masking_ratio_min=-1.0`, `masking_ratio_max=-1.0` | **INACTIVE** | `trainer.py:522-524` — `if (_mr_min >= 0 and _mr_max >= 0)` is False; fixed ratio used |
| Masking ratio annealing | `masking_ratio_anneal=False` | **INACTIVE** | `trainer.py:1201` — `if getattr(self.config, 'masking_ratio_anneal', False) and epoch >= teacher_warmup:` 조건이 flag=False로 False 평가; annealing 경로는 trainer.py:1201-1210에 **구현되어 있으나** 미진입 (정정 r2: 초판의 "trainer never triggers annealing path"는 코드 구조 오서술) |
| Complementary masking at inference | `eval_complementary_masking=False` | **INACTIVE** | `evaluator.py:1716` — `_use_complementary = getattr(self.config, 'eval_complementary_masking', False)` → False; `evaluator.py:1737` `if _use_complementary:` branch (K-group 경로 :1737-1745) 미진입 (flag 정의는 config.py:226-229) |
| Shared mask token | `shared_mask_token=False` | **INACTIVE** | `model.py:492-505` — `else:` branch used; separate tokens |
| freeze_teacher_after_warmup | `freeze_teacher_after_warmup=False` | **INACTIVE** | `trainer.py:1141-1142` — 런타임 freeze gate `if (getattr(self.config, 'freeze_teacher_after_warmup', False) and epoch == teacher_warmup ...)` False 평가; 모듈 동결 없음 (trainer.py:49-55는 별개의 init-시 config-validation override — 대입문 :55; r3 ±1 교정) |
| freeze_encoder_only | `freeze_encoder_only=False` | **INACTIVE** | `trainer.py:1169-1170` — 런타임 freeze gate `if (getattr(self.config, 'freeze_encoder_only', False) and epoch == teacher_warmup ...)` False 평가; encoder 동결 없음 (trainer.py:75-79는 별개의 동시-flag ValueError guard — `if`문 :76; r3 ±1 교정) |
| Anomaly-loss warmup ramp (`_compute_warmup_factor`) | `teacher_only_warmup_epochs=250` (산식 `max(250//5,2)=50`) | **INACTIVE (no-op in 271)** | `warmup_factor` 소비처는 anomaly_loss 곱셈 3곳뿐(`loss.py:265,272,404`); 271은 `use_grl ∧ grl_disable_anomaly_loss`로 도달 전 `anomaly_loss=0.0` 하드 제로(loss.py:259-261). GRL/FM은 ramp 없이 `not teacher_only` 게이트만으로 warmup 종료 직후 즉시 adaptive-lambda 가중 투입(trainer.py:746,762-763 GRL; :639,652 FM) |
| Default / ratio_weighted score modes (`lambda_disc`) | `anomaly_score_mode='adaptive'`, `lambda_disc=2.0` | **INACTIVE in score-path** (진단 CSV에는 소비 — r3 정정) | **score-path dead**: `scoring.py:326-333` — dispatch가 `mode == 'adaptive'`에서 분기; `compute_default_score`(`recon + lambda_disc * disc`, scoring.py:286-293)와 `compute_ratio_weighted_score`(:296-304) 미호출. **단** 진단 경로 `evaluator.py:2017` `compute_detailed_losses`는 score-mode **무관하게** `'total_loss' = recon + 2.0·disc`를 계산하며 271 실행 경로에서 호출됨 — `run_base_experiments.py:772`(per-epoch disc_SNR/recon_SNR 산출 입력) 및 `:1908`(최종 저장 → `best_model_detailed.csv` 'total_loss' 칼럼; 271 entity 디렉토리에 실존). 비게이트 소비처 추가: `visualization/best_model_visualizer.py:1184`. 이 칼럼은 **점수도 지표도 아니다** — SNR 계산(`compute_loss_statistics`, run_ablation.py:562)은 recon/disc만 소비. §VII #21 참조 |
| NPSR-style test-only clamp | `minmax_clamp_min=-4.0`, `minmax_clamp_max=4.0` | **INACTIVE** | `dataset_sliding.py:1019-1028` — clamp는 `minmax_range == 'neg1_1'` 분기에서만 전달; 271은 `'0_1'` → `cm_min, cm_max = None, None` (:1025). docstring 명문 "271 default: feature_range=(0, 1), clip=True, clamp=None" (dataset_sliding.py:956) |
| Synthetic anomaly interval scale | `anomaly_interval_scale=0.75` | **INACTIVE (합성 전용)** | 소비처는 합성 데이터 생성뿐: `run_ablation.py:944,1456`, `visualization/base.py:306`, `visualization/training_visualizer.py:97`; 271 경로 `run_base_experiments.py`는 무참조 |
| Synthetic total length | `sliding_window_total_length=275000` | **INACTIVE (합성 전용 stale)** | 소비처: `run_ablation.py:942,1454`, `visualization/*` (합성 재생성)뿐; 271 실데이터 길이는 `total_length = len(signals)` (`run_base_experiments.py:1804`)로 실측 — config 필드 미사용 |
| Bare `adaptive_lambda` (discriminator 전용) | `adaptive_lambda=True` | **INACTIVE** | 유일 소비처 `trainer.py:608` — discriminator adversarial 경로 내부; `use_discriminator=False` → `self.discriminator = None`(trainer.py:236-237)이라 미도달. **활성 GRL/FM adaptive lambda(`grl_adaptive_lambda`/`fm_adaptive_lambda`)와는 별개 필드 — 이름 충돌 주의** |
| 운영 키 (판정 보강) | `use_sliding_window_dataset=True`, `random_seed=42`, `device='cuda'` | **ACTIVE** (서술적/운영) | SlidingWindowDataset 경로 사용, 시드 고정, CUDA 실행 — 모델 구조·손실에 영향 없는 운영 파라미터 |
| Gaussian smoothing | N/A (R34 exclusion) | **EXCLUDED (R34) — 코드 존재, 271 미사용** | 코드는 **존재**: `mae_anomaly/scripts/q3_exploration/core/scoring.py:48-51` `gauss()` (gaussian_filter1d), `core/postprocess.py:51` `savitzky_golay_smooth` / `:129` `double_gaussian`; 적용은 q3_exploration 후처리 탐색 스크립트 한정 (예: `experiments/exp_P14_boundary_refinement.py:147` `gauss(base_unsmoothed, 10)` = Notion B2 variant, sigma=10). 271 파이프라인(evaluator.py/scoring.py/trainer.py/visualization/run_base_experiments.py)은 q3_exploration **무참조**(grep 0건) → 271의 모든 저장 점수·지표는 비평활(unsmoothed). 논문 제외(R34) (정정 r2: 초판의 "Not present in codebase at all"은 허위 부재 진술) |

---

## VII. Paper Exclusion List

The following items are present as configuration options or code paths in exp271 but are **inactive** and must not appear in the paper description of config 271:

1. **Dynamic margin + anomaly-loss family** — `margin_type='dynamic'` is set in config, but the dynamic margin path in `loss.py` is **never reached** because `use_grl=True` and `grl_disable_anomaly_loss=True` force `anomaly_loss = 0.0` before the `_compute_patch_anomaly_loss` call (loss.py:259-261). The margin computation only runs on anomaly patches; since anomaly_loss is zeroed, `_compute_patch_anomaly_loss` is never invoked. The `margin=0.5` and `dynamic_margin_k=6` settings have no effect on training. **같은 dead branch에 속한 `anomaly_loss_weight=2.0`** (소비처 loss.py:265, 272, 404 — 모두 zeroing 이후 도달 불가) **와 `anomaly_loss_direction='maximize'`** (분기 판정 loss.py:262 `elif` — :259의 GRL 분기에 선점됨) **도 동일하게 무효** — "anomaly 패치 손실 2× 가중" 같은 서술이 논문에 들어가면 안 된다 (r2 추가).

2. **SCAD (Supervised Contrastive Anomaly Discrimination)** — `use_scad=False`. ScadProjectionHead never instantiated, loss never computed.

3. **Adversarial discriminator** — `use_discriminator=False`. PatchDiscriminator never created, adversarial loss path never entered.

4. **GRL WDGRL mode** — `grl_mode='classifier'`. Wasserstein critic path never entered.

5. **GRL balanced sampling** — `grl_balanced_sampling=False`. All patches used, no downsampling.

6. **Feature matching in anomaly score** — FM is a training loss only. `scoring.py:237` forces `fm_active = False` at scoring time; FM never contributes to inference anomaly scores.

7. **Teacher output EMA** — `use_teacher_output_ema=False`. EMA modules never created.

8. **Teacher warmup early stop** — `use_teacher_warmup_early_stop=False`. Early-stop logic never triggered.

9. **RevIN** — `use_revin=False`. Per-window normalization not applied.

10. **CNN patch embedding** — `patchify_mode='linear'`. CNN patchify path never entered.

11. **Shared decoder layers** — `num_shared_decoder_layers=0`. No shared decoder between teacher and student.

12. **Complementary masking at inference** — `eval_complementary_masking=False`. Standard single-pass inference only.

13. **Random masking ratio range** — `masking_ratio_min=-1.0`, `masking_ratio_max=-1.0`. Both negative; fixed ratio of 0.15 used throughout.

14. **Masking ratio annealing** — `masking_ratio_anneal=False`. Fixed ratio throughout. (annealing 경로 자체는 `trainer.py:1201-1210`에 구현되어 있음 — flag=False라 미진입; r2 정정.)

15. **freeze_teacher_after_warmup** — `False`.

16. **freeze_encoder_only** — `False`.

17. **Shared mask token** — `shared_mask_token=False` (not shared, separate tokens used — this IS active in the sense that separate tokens exist, but this is the default expected path, not an optional variant).

18. **Gaussian smoothing** — Excluded per R34. **코드는 존재하나 271 미사용** (r2 정정 — 초판의 "Not present in the codebase"는 오류): `mae_anomaly/scripts/q3_exploration/core/scoring.py:48-51` `gauss()` 및 `core/postprocess.py:51,129`에 구현, `q3_exploration/experiments/*.py`(예: `exp_P14_boundary_refinement.py:147`의 `gauss(base_unsmoothed, 10)` = B2 variant)에서만 적용. 271 파이프라인은 q3_exploration 무참조 → 271 저장 점수·지표 전부 비평활. 논문 본문에 "스무딩 없음"을 서술할 때는 "코드에 부재"가 아니라 "후처리 탐색 스크립트 한정, 271 평가 경로 미사용"으로 근거를 들 것.

19. **`student_recon_weight`** — `0.0`. Marked as `[NOT YET IMPLEMENTED]` in `config.py:112`.

20. **`d_grad_student_layers`** — `'all'`. Marked as `[NOT YET IMPLEMENTED]` in `config.py:249`.

21. **`lambda_disc` + 대안 score mode 2종** — **score-path에서 dead** (r3 정정 — r2의 "유일한 런타임 소비처"·"절대 실행되지 않는다"·"271 dead"는 허위 코드 구조 진술): dispatch(scoring.py:326-333)가 `anomaly_score_mode='adaptive'`에서 분기하므로 `compute_default_score`(`recon + lambda_disc * disc`, scoring.py:286-293)·`compute_ratio_weighted_score`(:296-304)는 271에서 미호출 — `'default'`/`'ratio_weighted'` score mode 분기 일체 미사용은 맞다. **단 `lambda_disc=2.0`은 271 런타임에서 실제로 읽힌다**: 진단 경로 `evaluator.py:2017`(`compute_detailed_losses`)이 score-mode 무관하게 `'total_loss' = recon + 2.0·disc`를 계산하고, 271 경로 `run_base_experiments.py:772`(per-epoch disc_SNR/recon_SNR 산출 입력)·`:1908`(최종 저장)에서 호출되어 `best_model_detailed.csv`의 `total_loss` 칼럼으로 기록된다 (entity 디렉토리에 실존 — 예: PSM 첫 행 `0.00186 + 2.0×1.70528 = 3.41241` 수치 재검증 일치). mode 비게이트 소비처 추가: `visualization/best_model_visualizer.py:1184`(`total = teacher + lambda_disc·disc`, sample-type 플롯 내부). **이 칼럼은 점수도 지표도 아니다** — SNR 계산(`compute_loss_statistics`, run_ablation.py:562)은 recon/disc만 소비하고 `total_loss` 키를 사용하지 않는다. 정밀 결론: **진단용 detailed losses CSV에는 `lambda_disc=2.0`이 쓰이나, 평가·선정에 쓰이는 anomaly score(adaptive 모드)·전 평가지표에는 무참여 — 논문의 score 식과 무관.** §II의 `2.0`만 보고 "score = recon + 2·disc"로 점수식을 재구성하면 안 된다 — 실제 271 점수식은 §VIII Anomaly Score 절(adaptive)뿐 (r2 추가, r3 정밀화).

22. **`minmax_clamp_min=-4.0` / `minmax_clamp_max=4.0`** — NPSR-style test-only clamp는 `minmax_range='neg1_1'`일 때만 전달된다 (dataset_sliding.py:1019-1028; `'0_1'` 분기는 clamp=None, :1025). 271은 `'0_1'` → **±4 clamp는 한 번도 적용되지 않음**. "test 구간을 [-4,4]로 clamp"라는 전처리 서술 금지 (r2 추가).

23. **`anomaly_interval_scale=0.75`** — 합성(simulation) 데이터 생성 전용 (소비처: run_ablation.py:944,1456; visualization/base.py:306 등). 271은 실데이터만 사용 → 미사용 (r2 추가).

24. **`sliding_window_total_length=275000`** — 합성 전용 stale 필드. 271 실행 경로(`run_base_experiments.py`)는 무참조; 실데이터 길이는 `total_length = len(signals)`(run_base_experiments.py:1804)로 실측되며 개체별로 다르다 (PSM 220,322 등 — `EXPERIMENT_PROTOCOL_TRUTH.md` 데이터표 참조). **"275,000 timesteps"는 271의 어느 개체에도 해당하지 않는다** (r2 추가).

25. **bare `adaptive_lambda=True`** — discriminator 전용 adaptive lambda (유일 소비처 trainer.py:608, D adversarial 경로 내부). `use_discriminator=False`인 271에서 dead. **활성 메커니즘인 `grl_adaptive_lambda`/`fm_adaptive_lambda`와 이름이 유사하나 완전히 별개의 필드** — §VIII의 "adaptive lambda" 서술의 스위치로 오독하지 말 것 (r2 추가). (같은 D-family인 `disc_lr_ratio=4.0`, `adv_loss_weight=1.0`, `disc_warmup_epochs=10`, `disc_channels=[64,32]`도 item 3의 범위로 모두 dead.)

26. **`teacher_warmup_early_stop_metric='recon_snr'`** — config.py:289 정의 외 **코드 전체 무참조** (early-stop family가 활성이어도 이 필드는 읽히지 않음). item 19/20과 같은 급의 dead 필드 (r2 추가).

---

## VIII. Used Component Settings — Config 271 Specifics

### Architecture

| Component | Setting |
|-----------|---------|
| Patchify mode | `linear` (flatten + linear projection; no CNN) |
| Encoder | Transformer, 4 layers, self-attention, Pre-Norm, GELU, `d_model=512`, `nhead=8`, `dim_feedforward=2048` |
| Teacher decoder | Transformer (self-attn only, MAE-style), 3 layers |
| Student decoder | Transformer (self-attn only, MAE-style), 2 layers |
| Shared decoder | None (`num_shared_decoder_layers=0`) |
| Mask tokens | Separate for teacher and student (`shared_mask_token=False`) |
| Masking layout | Mask-after-encoder (`mask_after_encoder=True`): encoder sees only visible patches; mask tokens inserted before decoder |
| Window/sequence | `seq_length=500`, `patch_size=10`, `num_patches=50` |
| Dropout | `0.15` |

### Masking

| Parameter | Value |
|-----------|-------|
| Masking ratio | `0.15` (fixed; 정확히 `round(50 × 0.15) = 8` patches per window — `model.py:986` `target_num_masked = round(current_seq_len * masking_ratio)`; visible 42) |
| Masking strategy | Anomaly-first (`force_mask_anomaly=True`): anomaly patches are masked first within the 15% budget |
| Masking ratio range | Fixed (not random; `masking_ratio_min=-1`, `masking_ratio_max=-1`) |
| Masking ratio annealing | Disabled |

### Training

| Parameter | Value |
|-----------|-------|
| Epochs | 500 |
| Teacher-only warmup | 250 epochs (0-based epoch 0–249: teacher decoder만 학습; student frozen). **학습 경로에서는 student decoder forward 자체가 skip된다** (r4 보강): trainer가 `teacher_only`를 model forward에 전파(`trainer.py:526–535`, 2026-05-29 변경 — 271 실행 2026-06-02 이전 반영)하고 model 게이트 `… and not teacher_only`(`model.py:1119`)가 student decoder·GRL classifier·SCAD head forward를 생략 → `student_output=None`(`loss.py:193` None 처리), 손실 게이트(`loss.py:213` `not teacher_only`)는 이중 방어. **student 학습은 0-based epoch 250(=251번째 epoch)부터 개시.** 평가·시각화 경로는 `teacher_only=False` 기본값이라 full forward 유지 |
| Warmup LR ramp | 10 epochs linear warmup (`warmup_epochs=10`) |
| Optimizer | AdamW, fused, `lr=0.001`, `weight_decay=0.001`, `betas=(0.9, 0.99)` |
| LR schedule | Linear warmup (10 ep) + CosineAnnealingLR |
| Batch size | 1024 |
| Mixed precision | AMP bf16 (no GradScaler) |
| Student-loss 활성화 시점 | **ramp 없음** (r2 정정): GRL/FM 항은 `not teacher_only` 게이트만으로 warmup 종료 직후 첫 student epoch(0-based epoch 250)부터 즉시 adaptive-lambda 가중으로 투입 (trainer.py:746,762-763 GRL; :639,652 FM). `_compute_warmup_factor`의 50-epoch ramp(trainer.py:336-348)는 anomaly_loss 전용(loss.py:265,272,404)인데 271은 anomaly_loss가 하드 제로(loss.py:259-261) → **no-op**. 초판의 "Anomaly loss warmup: 50 epochs ramp" 행은 271에 효과가 없는 메커니즘이라 삭제. **단 (r4): "ramp 없음"은 손실 항 투입에 한정** — GRL의 gradient **반전 계수 λ_rev**는 별도로 Ganin-style sigmoid ramp를 따라 epoch 250부터 0→≈1로 점진 증가한다 (§VIII GRL Details 참조; trainer.py:1201–1211) |

### Loss Components (All Active)

| Loss Term | Configuration |
|-----------|--------------|
| Teacher reconstruction loss | MSE on masked positions, averaged over features |
| Normal discrepancy loss (output-level) | Mean patch-level OD on normal masked patches; `normal_loss_weight=1.0` |
| Anomaly discrepancy loss | **Zeroed** (`grl_disable_anomaly_loss=True`); GRL handles this side |
| GRL classifier loss | Focal BCE on student hidden (window-level label); `grl_loss_weight=0.2`; adaptive lambda (trainer inline grad-ratio, `trainer.py:751-765` — `(‖∇L_main‖/(‖∇L_grl‖+1e-4)).clamp(0,10)` :760; discriminator 전용 VQGAN-style `compute_adaptive_lambda`(loss.py:683, 유일 호출처 trainer.py:610, 271 비활성)와 **별개 공식** — 귀속 금지, §VIII GRL Details 참조; r3 정정); GRL gradient reversal은 student decoder의 anomaly-identity feature를 **억제(suppression)** — 정정(2026-06-10 reconciler): 초판의 "anomaly-discriminative features 생성"은 방향 오류. `GradientReversalFunction`은 backward에서 `-lambda × grad` (`model.py:129–140`) — **이 lambda는 손실 가중치가 아니라 반전 계수 λ_rev**(Ganin-style sigmoid ramp, §VIII GRL Details; r4), head docstring "GRL for adversarial feature suppression" (`model.py:143–144`) — student가 anomaly 정보를 표현하지 못하게 만들어 anomaly에서 teacher와의 discrepancy를 증폭 |
| Feature matching loss (FM) | L2 distance between teacher and student hidden on normal masked patches; `fm_loss_weight=1.0`; added with adaptive lambda |
| Total loss | `reconstruction_loss + normal_loss + adaptive_grl_weight * grl_cls_loss + adaptive_fm_weight * fm_loss` |

### GRL Details

| Parameter | Value |
|-----------|-------|
| Mode | `classifier` (DANN-style GRL with binary BCE, not WDGRL) |
| Architecture | **2-layer MLP** (Linear 2개 = hidden 1층): `LayerNorm → Linear(d_model, d_model//2=256) → GELU → Dropout(0.1) → Linear(256, 1)` (`model.py:177-186`; 코드 주석 "Default: 1-layer MLP with LayerNorm"(model.py:178)은 hidden-층 수 기준 표현 — "1-layer MLP" 표기 금지, RESEARCH_SYNTHESIS 표A와 통일; r3 정정) |
| Target granularity | Window-level (`grl_target_mode='window'`): all patches in an anomaly window get target=1 |
| Loss | Focal BCE (`grl_use_focal=True`, gamma=2.0) |
| Class weight | `grl_pos_weight` — per-entity, computed from actual data ratio |
| Lambda balancing (**손실 가중치 λ_GRL** — r4 명칭 명확화) | Adaptive (`grl_adaptive_lambda=True`): `lambda = ||grad_main|| / (||grad_grl|| + 1e-4)`, clamped [0, 10], smoothed via prev-epoch average (`trainer.py:751–765`, 공식 :760; prev-epoch 갱신 :1317–1319, 초기값 1.0 :190) |
| Effective weight | `_prev_epoch_grl_lambda * 0.2 * grl_cls_loss` (`trainer.py:762–763`; `grl_loss_weight=0.2` :749 + metadata) — **×0.2 계수 실재** |
| **이중 λ 구조 (r4 신설)** | GRL에는 **서로 별개의 λ 2개**가 공존하며 271에서 **둘 다 활성**: ① 손실 가중치 λ_GRL(위 두 행 — grad-ratio adaptive × `grl_loss_weight` 0.2), ② gradient **반전 계수 λ_rev**(아래 행 — Ganin-style sigmoid ramp). 단일 λ로 합쳐 서술 금지 — P3 재리뷰 NEW-B1의 근본 원인은 본 정본의 λ_rev 미등재였음 |
| **Reversal coefficient λ_rev (Ganin-style sigmoid ramp; r4 신설)** | 매 epoch train_epoch **전에** `model._grl_lambda`로 설정 (`trainer.py:1201–1211`; 게이트는 `use_grl`뿐 — 271 활성): `p = clip((epoch − 250 + 1) / 250, 0, 1)` (250 = `teacher_only_warmup_epochs`; 분모 250 = `num_epochs − warmup` = student-phase 길이), `λ_rev = 2/(1 + exp(−10·p)) − 1`; warmup 중(epoch<250)은 0.0 고정(:1209). 0-based epoch 250에서 ≈0.020으로 시작, 마지막 epoch 499에서 ≈0.9999까지 단조 증가. 소비처: `model.py:1152–1153` `anomaly_classifier(student_hidden, lambda_grl)` → `GradientReversalFunction.backward`가 `−λ_rev × grad` 반환(`model.py:129–140`). `model._grl_lambda` 대입 지점은 trainer.py:1209/1211 **뿐**(grep 전수). **FM에는 대응 메커니즘 없음** — sigmoid ramp는 GRL 반전 계수 전용(FM은 손실 가중 단일 구조, `trainer.py:639–653`) |
| **Student hidden 도달 adversarial gradient (r4 신설)** | `−λ_rev × λ_GRL_eff × ∂L_cls/∂(GRL 출력)` — 손실 가중치(λ_GRL_eff = prev-epoch grad-ratio × 0.2)와 반전 계수(λ_rev sigmoid ramp)가 **곱으로 함께** 작용 |
| Classifier LR | `0.001 * 0.1 = 0.0001` (separate param group) |
| Balanced sampling | Off (`grl_balanced_sampling=False`) |

### Feature Matching Details

| Parameter | Value |
|-----------|-------|
| Distance metric | L2: `((teacher_hidden.detach() - student_hidden)**2).mean(dim=-1)` |
| Target patches | Normal masked patches only (`patch_is_normal * patch_has_masked`) |
| Lambda balancing | Adaptive (`fm_adaptive_lambda=True`): `_fm_lambda = (||grad_main|| / (||grad_fm|| + 1e-4))`, clamped [0, 10], smoothed |
| Effective weight | `_prev_epoch_fm_lambda * 1.0 * fm_loss_tensor` |
| At inference | **Not included in anomaly score** (`scoring.py:237` forces `fm_active = False`) |

### Data / Normalization

| Parameter | Value |
|-----------|-------|
| Normalization | Min-max, per-feature, range [0, 1] (`normalize_mode='minmax'`, `minmax_range='0_1'`). **세부 (r2 추가)**: ① scaler min/max를 **train 구간에서만 fit** (`signals[:train_end]`) ② 전체 신호 변환 ③ `clip=True`로 **train+test 전체를 [0,1]에 tight-clip** (train 범위 밖 test 값은 포화) ④ test-only clamp 없음 (`_minmax_per_feature`, dataset_sliding.py:935-998; docstring "271 default: feature_range=(0, 1), clip=True, clamp=None" :956) |
| Dataset length | **개체별 실데이터 길이** (`total_length = len(signals)`, run_base_experiments.py:1804; 예: PSM 220,322). config의 `sliding_window_total_length=275000`은 합성 전용 stale 필드로 271 미사용 — §VII #24 (r2 정정: 초판 "275,000 timesteps" 행은 어느 개체에도 해당하지 않는 오판정) |
| Train stride | 21 (overlapping windows) |
| Test stride | -1 sentinel → `resolve_test_stride` = **`seq_length // 10 - 1` = 49** (`mae_anomaly/utils/experiment.py:16–39`). 정정(2026-06-10 reconciler): 초판의 "`num_patches - 1`"은 `config.py:23–26` 주석을 따른 것이나 실제 구현 산식은 `W // 10 - 1` — 271은 `patch_size=10`이라 두 식이 49로 우연히 일치 (patch_size≠10이면 달라짐) |
| Epoch offset | `True` (random start offset per epoch, cycles through [0, stride)) |

### Anomaly Score (Inference)

Mode: `adaptive` (`anomaly_score_mode='adaptive'`)

> 주의 (r2): 대안 mode `'default'`(`recon + lambda_disc * disc`)와 `'ratio_weighted'`는 271에서 미사용 —
> dispatch가 `mode == 'adaptive'`에서 분기(scoring.py:326-333)하므로 `lambda_disc=2.0`은 점수식에 **아무 기여 없음** (§VII #21).

Formula (from `scoring.py:239-256`):
```
recon_mean = mean(recon) + 1e-4
disc_mean  = mean(disc) + 1e-4
scaled_disc = disc * (recon_mean / disc_mean)
student_error = scaled_disc / score_recon_disc_ratio   # = scaled_disc / 4.0
score = recon + student_error
```

Where:
- `recon` = teacher reconstruction error (MSE on masked positions, per timestep)
- `disc` = output-level student-teacher discrepancy (per timestep)
- FM is **NOT included** (hardcoded `fm_active = False` at `scoring.py:237`)
- `score_recon_disc_ratio = 4.0` → recon:disc contribution = 4:1 after scale normalization
- `eval_disc_weight = -1.0` → resolves to 1.0 (default); `eval_fm_weight = -1.0` → irrelevant (FM excluded)

### Threshold / Evaluation

| Parameter | Value |
|-----------|-------|
| Best epoch selection | `pak_auc_f1` (PA%K AUC of F1 with per-K threshold re-optimization) |
| Eval interval | Every 5 epochs (config `eval_interval=5`; 실구동은 스크립트 상수 `EVAL_INTERVAL = 5`, run_base_experiments.py:94 — 값 일치, r2 각주) |
| Threshold method | Optimal F1 threshold sweep (point-level) + anomaly-ratio threshold (_ar suffix metrics) |
| PA%K adjustment | PA%K sweep from k=0 to k=100 in steps of 1 |

---

## IX. REQUEST / FEEDBACK

```
REQUEST: None at this time. Full config forensics complete per R17.
FEEDBACK: R34 (Gaussian smoothing exclusion) confirmed — 코드는 q3_exploration 후처리 탐색
스크립트(core/scoring.py:48 gauss 등)에 존재하나 271 파이프라인은 무참조·미사용; 271 저장
점수·지표는 전부 비평활이며 논문에서 제외(R34). (r2 정정 — 초판의 "no such component exists
anywhere in the codebase"는 허위 부재 진술이었음.)
R28 (SWaT dual-condition) confirmed in metadata structure.
```

---

## 부록 1: 2026-06-10 reconciler 정정 목록 (r1)

1. **§VIII Data / Test stride**: "`num_patches - 1`" → 실제 구현은 `resolve_test_stride` = `seq_length // 10 - 1` (`utils/experiment.py:16–39`). 271에서는 patch_size=10이라 값(49)은 동일 — 산식만 정정.
2. **§VIII Masking**: "7-8 patches" → 정확히 8 (`round(50×0.15)=8`, `model.py:986`).
3. **§VIII GRL classifier loss**: "anomaly-discriminative features 생성" → **anomaly-identity feature 억제(suppression)** (`model.py:129–144`).

본 문서의 metadata 수치(§II–§IV)는 reconciler 재검증에서 **전건 일치** 확인 (PSM·SWaT full/excl22·WaDi A2 직접 재추출 + SWaT checkpoint `patch_embed.weight=(512,450)` 실측). 판정 근거 전체: `paper/99_reviews/p1_reconciliation_r1.md`.

---

## 부록 2: 2026-06-10 fixer-1 정정 목록 (r2)

리뷰 출처: `paper/99_reviews/p1_271truth_verifier1_r1.md` (V1), `p1_271truth_verifier2_r1.md` (V2).
전 건 1차 소스(코드 file:line / metadata 필드) 재검증 후 반영 — 처리표: `paper/99_reviews/p1_271truth_fixlog_r2.md`.

1. **V1-B1 — §VI Masking ratio annealing 근거**: "trainer never triggers annealing path"(config.py:241-243 인용) → annealing 경로는 `trainer.py:1201-1210`에 구현되어 있고, flag=False라 조건이 False 평가되어 미진입하는 것으로 정정. §VII #14에도 동일 보강.
2. **V1-M2 — §IV SWaT excl22**: `timing.best_epoch_metric='excl22_pak_auc_f1'`(config 키는 `'pak_auc_f1'`) 운영 차이 주석 추가 (metadata 실측).
3. **V1-M3 — §VI complementary masking 근거**: config.py:226-229 → `evaluator.py:1716` flag read + `:1737` branch 미진입으로 교체.
4. **V1-M4 — §VI freeze_teacher_after_warmup 근거**: trainer.py:50-55(init-시 config-validation override) → 런타임 gate `trainer.py:1141-1142`로 교체.
5. **V1-M5 — §VI freeze_encoder_only 근거**: trainer.py:75-79(ValueError guard) → 런타임 gate `trainer.py:1169-1170`로 교체.
6. **V1-M1 — §III-3c SMAP**: "~0.625" → "0.617–0.626 (G-7: 0.617)" (metadata 실측 0.6167).
7. **V1-Mi1 — §VI decoder 라인**: teacher `model.py:407-423`→`419-423`, student `445-461`→`457-461`.
8. **V1-Mi2 — §VI linear patch embedding 라인**: `577-580` → `:580`(patch_cnn skip) + `:624`(linear 진입).
9. **V1-Mi3 — §VI mask-after-encoder**: teacher branch `model.py:1028-1036` 추가 (기존은 student branch :1119-1129만 인용).
10. **V2-B1 — §VIII "Total dataset length 275,000"**: 오판정 정정 — `sliding_window_total_length`는 합성 전용 stale 필드(소비처 run_ablation.py:942,1454·visualization뿐), 271 실길이는 `len(signals)`(run_base_experiments.py:1804). §VII #24 신설.
11. **V2-B2 — §VIII "Anomaly loss warmup 50ep ramp"**: 271에서 no-op으로 정정 — warmup_factor 소비처(loss.py:265,272,404)는 전부 anomaly_loss 곱셈인데 271은 하드 제로(loss.py:259-261); GRL/FM은 ramp 없이 epoch 250(0-based)부터 즉시 투입(trainer.py:639,652,746,762-763). §VI에 no-op 판정 행 신설.
12. **V2-B3 — Gaussian smoothing (R34)**: "Not present in codebase at all" 허위 부재 진술 정정 — `q3_exploration/core/scoring.py:48-51` `gauss()` 등 실재, 적용은 탐색 스크립트 한정(`experiments/exp_P14_boundary_refinement.py:147` 등 B2 variant), 271 파이프라인 무참조(grep 0건)·논문 제외. §VI 행·§VII #18·§IX FEEDBACK 모두 교정.
13. **V2-B4 — `lambda_disc=2.0` 판정 등재**: adaptive dispatch(scoring.py:326-333)에서 default 분기(scoring.py:286-293) 미실행 → dead. §VI 행 + §VII #21 + §VIII score 절 주의 신설.
14. **V2-B5 — `minmax_clamp_min/max=±4.0` 판정 등재**: `'neg1_1'` 전용(dataset_sliding.py:1019-1028; '0_1'은 clamp=None :1025, docstring :956) → 271 미적용. §VI 행 + §VII #22 신설.
15. **V2-B6 — `anomaly_interval_scale=0.75` 판정 등재**: 합성 전용(run_ablation.py:944,1456; visualization/base.py:306) → 271 미사용. §VI 행 + §VII #23 신설.
16. **V2-M1 — `anomaly_loss_weight=2.0`/`anomaly_loss_direction='maximize'`**: §VII #1에 dead 판정 추가 (loss.py:259-272).
17. **V2-M2 — bare `adaptive_lambda=True`**: discriminator 전용(trainer.py:608) dead + `grl_adaptive_lambda`/`fm_adaptive_lambda`와의 이름 충돌 경고. §VI 행 + §VII #25 신설.
18. **V2-M3 — §VIII Normalization 세부**: train-only fit + 전구간 [0,1] tight clip(`clip=True`) + clamp 없음 명기 (`_minmax_per_feature` dataset_sliding.py:935-998, docstring :956).
19. **V2-m1 — §I 동어반복 문장 정리.**
20. **V2-m2 — §III-3b 999.0**: "capped sentinel" → patch-ratio 하한 유도값 (`max(_patch_ratio, 0.001)` → 999.0, run_base_experiments.py:2584-2585).
21. **V2-m3 — §VIII eval_interval**: 실구동은 스크립트 상수 `EVAL_INTERVAL=5`(run_base_experiments.py:94) 각주 추가.
22. **V2-m4 — `teacher_warmup_early_stop_metric`**: 코드 전체 무참조 dead 필드 — §VII #26 신설.
23. **V2-m5 — `use_sliding_window_dataset`/`random_seed`/`device`**: §VI에 운영 키 판정 행 신설.
24. *(NOTE급 선택 반영)* V1-N2: §VI early-stop 근거 표현을 getattr 평가로 정밀화. V1-N3: §VIII score formula 인용 범위 239-253 → 239-256.

---

## 부록 3: 2026-06-10 fixer-5 정정 목록 (r3)

리뷰 출처: `paper/99_reviews/p1_rereview_alpha_r2.md` (재리뷰 α, round 2).
전 건 1차 소스(코드 file:line / metadata / 산출 CSV) 재검증 후 반영 — 처리표: `paper/99_reviews/p1_fixlog_r3.md`.

1. **α-B1 — `lambda_disc` "유일 소비처 / 271 dead" 허위 구조 진술 (r2 V2-B4가 도입)**: §VI 행 + §VII #21 재서술 — score-path dead(dispatch scoring.py:326-333, default/ratio_weighted 미호출)는 유지하되, 진단 경로 `evaluator.py:2017` `compute_detailed_losses`가 score-mode 무관하게 `recon + 2.0·disc`를 계산하고 271 경로(`run_base_experiments.py:772, 1908`)에서 실행되어 `best_model_detailed.csv` `total_loss` 칼럼에 기록됨을 명시 (CSV 실존 + 첫 행 수치 재검증 일치). 정밀 결론: **진단용 detailed losses CSV에는 lambda_disc=2.0이 쓰이나, 평가·선정에 쓰이는 anomaly score(adaptive 모드)에는 무참여 — 논문의 score 식과 무관.** r2 fixlog(p1_271truth_fixlog_r2.md V2-B4 행)의 "유일 소비처 확인" 재검증 기록은 grep 누락에 의한 오검증 — 본 r3 fixlog가 대체 (구 fixlog는 본 phase 쓰기 범위 외라 미수정).
2. **α-B2 — "1-layer MLP" 오기 2개소 (r1 MAJ-004 잔존)**: §VI GRL classifier 행 + §VIII GRL Details Architecture 행 → **2-layer MLP** (Linear 2개 = hidden 1층, `model.py:177-186` 실측: Linear(512→256) + Linear(256→1)). 코드 주석 "Default: 1-layer MLP"(model.py:178)는 hidden-층 수 기준 표현임을 병기. RESEARCH_SYNTHESIS 표A("2-layer MLP 확정 / 1-layer 표기 금지")와 통일 — 3자 모순 해소.
3. **α-B3 — §VIII "adaptive lambda (VQGAN-style)" 귀속 오류 (3자 모순)**: VQGAN-style 공식(`compute_adaptive_lambda`, loss.py:683)의 유일 호출처는 discriminator 경로 `trainer.py:610`(271 비활성, grep 재확인). GRL λ는 trainer inline `(‖∇L_main‖/(‖∇L_grl‖+1e-4)).clamp(0,10)`(trainer.py:751-765, 공식 :760)로 별개 — "(VQGAN-style)" 삭제, inline grad-ratio + 별개 공식 명시로 교체. CODEBASE §2.6(r3)·SYNTHESIS 표A(r2)의 "VQGAN-style 귀속 금지"와 정합.
4. **α-m1 — §VI freeze 두 행 부기 라인 ±1**: init override "trainer.py:49-54" → **49-55**(대입문 :55), ValueError guard "74-78" → **75-79**(`if`문 :76) — 실측 교정. 본 근거 라인(1141-1142 / 1169-1170)은 변경 없음.

---

## 부록 4: 2026-06-11 fixer 정정 목록 (r4 — Phase 3 재리뷰 escalation)

리뷰 출처: `paper/99_reviews/p3_rereview_adversarial_r2.md` (NEW-B1/NEW-B2 — 두 건의 근본 원인이 **본 정본의 메커니즘 누락**으로 판정되어 §6.3 회귀 프로토콜에 따라 Phase 1 정본을 보강). 전 건 1차 소스(코드 file:line + 271 metadata) 재검증 후 반영 — 처리표: `paper/99_reviews/p3_fixlog_r3.md`. CODEBASE_UNDERSTANDING(r4)·RESEARCH_SYNTHESIS(r3) 동기화 동시 수행.

1. **GRL 이중 λ 구조 등재 (NEW-B1 escalation)**: §VIII GRL Details에 반전 계수 λ_rev 행 신설 — Ganin-style sigmoid ramp `λ_rev = 2/(1+exp(−10p))−1`, `p = clip((epoch−250+1)/250, 0, 1)`, warmup 중 0.0, epoch 250≈0.020 → epoch 499≈0.9999 (`trainer.py:1201–1211`; 소비처 `model.py:1152–1153` → `GradientReversalFunction.backward` `−λ_rev×grad`, `model.py:129–140`; 대입 지점 trainer.py:1209/1211 뿐 — grep 전수). 기존 등재된 손실 가중치 λ_GRL(grad-ratio clamp[0,10] × 0.2, `trainer.py:751–765`)과 **별개이며 271에서 둘 다 활성**. §VIII Training "ramp 없음" 서술을 **손실 항 투입 한정**으로 정밀화(λ_rev는 ramp됨), Loss Components GRL 행의 `-lambda × grad`의 lambda를 λ_rev로 명시, student hidden 도달 gradient = `−λ_rev × λ_GRL_eff × ∂L_cls/∂(GRL 출력)` 행 신설. FM에는 대응 ramp 없음(`trainer.py:639–653` 손실 가중 단일 구조 — sigmoid는 GRL 전용) 확인 병기. 구판(r2 V2-B2, r3까지)의 "GRL/FM ramp 없이 즉시 투입" 서술 자체는 손실 항에 한해 참 — 허위가 아니라 **누락**이었음.
2. **Warmup 중 student decoder forward skip 등재 (NEW-B2 escalation)**: §VIII Training warmup 행 보강 — **학습 경로에서 student decoder·GRL classifier·SCAD head forward 자체가 생략**(`teacher_only` 전파 `trainer.py:526–535` → 게이트 `model.py:1119`; `student_output=None` 처리 `loss.py:193`; 손실 게이트 `loss.py:213`은 이중 방어). 2026-05-29 변경으로 271 실행(2026-06-02) **이전** 반영 — student 학습은 0-based epoch 250(=251번째 epoch)부터 개시. 평가/시각화 경로는 `teacher_only=False` 기본값으로 full forward 유지. 기존 "student frozen" 축약 표현은 실동작에 부합했으나 forward 수행 여부 미등재가 P3 NEW-B2 오류(NOTION I-4 stale 서술 채택)의 근본 원인.
