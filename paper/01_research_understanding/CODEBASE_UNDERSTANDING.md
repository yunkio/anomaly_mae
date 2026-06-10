---
phase: 1
agent: research-archaeologist
directives: [T1]
last_modified: 2026-06-10
revision: r3 (reconciler 정정 + fixer-2 리뷰 반영 — paper/99_reviews/p1_codebase_synthesis_r1.md 전수 처리, fixlog: p1_codebase_synthesis_fixlog_r2.md)
---

# TSMAE Codebase Understanding

> **2026-06-10 reconciler 정정: 271_CONFIG_TRUTH 기준 정합화 (수정 목록은 말미).**
> **2026-06-10 fixer-2 정정(2차): adversarial review `paper/99_reviews/p1_codebase_synthesis_r1.md` 반영 — adaptive lambda 3경로 분리, leave-one-out 추론 방식, focal loss 비표준형 플래그, threshold/coverage/PA%K 서술 정밀화 (수정 목록은 말미 부록 2).**
> 초판은 exp271을 `CONFIG_PRESETS['A']`(Set A) 아키타입으로 가정했으나, 1차 소스 재추적 결과
> exp271(=`271_20260602_020545_271canon_baseline`)은 **Set C 기반 + 대량 config override**로 실행되었다
> (근거: 전 37 entity `experiment_metadata.json` config 블록; `summary.json: "config_set": "C"`;
> `configs/queue_fullrerun_20260601_190603.json` exp271 entry의 `"set": "C"` + `config_override`).
> 본 문서의 exp271 수치는 모두 metadata 실측값으로 정정되었다. code default와 271 실측값이 다른 곳은
> "code default는 X이나 **271 config에서는 Y**" 형식으로 병기한다.

## 1. Model Architecture

### Overview

The model is named `SelfDistilledMAEMultivariate` (file: `mae_anomaly/model.py:282`). It implements a self-distillation MAE (Masked Autoencoder) for multivariate time series anomaly detection. The core idea is a shared encoder, a deep teacher decoder, and a shallow student decoder; the teacher-student output discrepancy serves as the anomaly signal.

### Data Flow (Canonical Exp271 Config — Set C 기반 + overrides: `linear` patchify, `mask_after_encoder=True`)

> 정정(2026-06-10 reconciler): 초판의 "Set A — patch_cnn" 가정은 오류. 271 실측:
> `patchify_mode='linear'`, `patch_size=10`, `num_patches=50`, `d_model=512`,
> `dim_feedforward=2048`, encoder 4층, teacher decoder 3층, student decoder 2층
> (metadata 필드 `patchify_mode`/`patch_size`/`num_patches`/`d_model`/`dim_feedforward`/
> `num_encoder_layers`/`num_teacher_decoder_layers`/`num_student_decoder_layers`,
> 전 37 entity 동일; checkpoint 실측 `patch_embed.weight=(512, 450)`=Linear(10×F→512), SWaT F=45).

```
Input: (B, 500, F)        # F = dataset별 입력 차원 (metadata num_features: 25–123; 8이 아님 — 8은 R33 제외 대상인 simulation)
  │
  ├─ [Optional RevIN normalize — disabled in exp271]
  │
  ▼
Patchify (linear mode — exp271):
  reshape → (B×50, 10×F)            # 50 patches × patch_size=10 × F features (flatten)
  patch_embed: Linear(10×F → 512)   # model.py:628 (CNN 없음; patch_cnn 분기는 진입 안 함)
  reshape → (B, 50, 512)
  transpose → (50, B, 512)  [seq_first format]
  │
  ▼
force_mask_anomaly (training):
  anomaly patches get priority in masking budget (round(50 × 0.15)=8 patches masked; model.py:986)
  → patch_mask: (50, B), 0=masked, 1=visible
  │
  ▼
Encode visible only (_encode_visible_only):
  gather 42 visible patches → (42, B, 512)
  add original positional encodings (gathered by position, not sequential)
  TransformerEncoder (4 layers, 8 heads, d_ff=2048, Pre-Norm, GELU, eps=1e-6) → latent_visible: (42, B, 512)
  ids_restore: (50, B) for unshuffling
  │
  ├─────────────────────────────────────────┐
  ▼                                         ▼
Teacher path:                           Student path (detach encoder):
  Insert teacher_mask_token at          latent_visible.detach() → (42, B, 512)
  masked positions, unshuffle           Insert student_mask_token, unshuffle
  → (50, B, 512)                        → (50, B, 512)
  + decoder_pos_enc                     + decoder_pos_enc
  TransformerEncoder (3 layers) → teacher_hidden  TransformerEncoder (2 layers) → student_hidden
  teacher_output_projection(512→10×F)    student_output_projection(512→10×F)
  transpose + unpatchify → (B, 500, F)   transpose + unpatchify → (B, 500, F)
  [Optional RevIN denorm]               [Optional RevIN denorm]
  │
  ├── self._teacher_hidden = teacher_hidden  (for FM loss)
  ├── self._student_hidden = student_hidden  (for FM loss)
  ├── self._grl_cls_logits = anomaly_classifier(student_hidden, lambda_grl)  [if use_grl=True]
  └── self._ema_teacher_output [if use_teacher_output_ema=True, post-warmup only — exp271 off]

Returns: teacher_output, student_output, mask (all: B×500×F or B×500)
```

Evidence: `mae_anomaly/model.py:893–1219`

### Patchify Modes

**`linear` (exp271 actual)**: Flatten patch timesteps × features → Linear projection (`patch_embed: Linear(patch_size×num_features → d_model)`). No CNN. MAE original style. `mae_anomaly/model.py:624–631`. **271 config에서 활성** (metadata `patchify_mode='linear'`; checkpoint `patch_embed.weight=(512,450)`).

**`patch_cnn`**: Patchify first, then a 2-layer 1D-CNN per patch independently. CNN channels auto-scale as `(d_model//2, d_model)`. No cross-patch leakage during embedding. `mae_anomaly/model.py:580–633`. code default는 `patch_cnn`(`config.py:57`)이나 **271 config에서는 `linear` — CNN 분기 비활성** (`model.py:580` 분기 진입 안 함).

Set C uses `linear`; Sets A and B use `patch_cnn`. `scripts/run_base_experiments.py:178–244`. **exp271은 Set C 기반** (`summary.json: config_set='C'`).

### Encoder / Decoder Architecture

- **Encoder**: `nn.TransformerEncoder`, Pre-Norm (`norm_first=True`), GELU, `layer_norm_eps=1e-6`. `mae_anomaly/model.py:348–362`. code default는 `num_encoder_layers=2`, `d_model=128`, `dim_feedforward=512`(`config.py:43,41,49`)이나 **271 config에서는 4 layers, `d_model=512`, `nhead=8`, `dim_feedforward=2048`** (metadata `num_encoder_layers=4`, `d_model=512`, `dim_feedforward=2048`).
- **Teacher Decoder**: `nn.TransformerEncoder` (self-attention only, no cross-attention). `mae_anomaly/model.py:406–423`. code default는 4 layers(`config.py:44`)이나 **271 config에서는 3 layers** (metadata `num_teacher_decoder_layers=3`).
- **Student Decoder**: Same architecture. Shallow by design to amplify the discrepancy gap. `mae_anomaly/model.py:443–461`. code default는 1 layer(`config.py:45`)이나 **271 config에서는 2 layers** (metadata `num_student_decoder_layers=2`; teacher 3층 > student 2층의 비대칭은 유지).
- **`use_transformer_encoder_decoder=True`** (exp271 default): decoders use `nn.TransformerEncoder` (self-attention only). `False` would use `nn.TransformerDecoder` with cross-attention from encoder. `mae_anomaly/config.py:68–70`
- **Mask tokens**: Separate for teacher and student (`shared_mask_token=False`, exp271 default). `mae_anomaly/model.py:498–505`
- **Decoder positional encoding은 teacher/student 공유 단일 인스턴스**: `self.decoder_pos_encoder` 하나만 생성되어 두 decoder path가 공유한다 (`model.py:343–346`). 단, `PositionalEncoding`은 학습 파라미터가 없는 고정 sinusoidal buffer이므로(`model.py:263–279`, `register_buffer`) 공유/비공유가 수치에 영향을 주지 않는다 — 논문에서 "shared learned pos-enc"처럼 설계 결정으로 기술하지 말 것 (fixer-2, MIN-002).
- **`mask_after_encoder=True`** (exp271 default): standard MAE — encode visible patches only, insert mask tokens before decoder. Encoder is never exposed to masked positions. `mae_anomaly/model.py:1009–1015`
- **Encoder gradient isolation**: Student decoder receives `latent_visible.detach()`. Encoder is updated only via teacher path. `mae_anomaly/model.py:1123–1126`

### GRL (Gradient Reversal Layer)

When `use_grl=True` (exp271): an `AnomalyClassifierHead` is attached to the student decoder hidden states. Its gradient is reversed via `GradientReversalFunction` (DANN-style; identity forward / `-lambda × grad` backward, `model.py:129–140`), making **the student decoder** produce anomaly-uninformative (suppressed) representations — head docstring: "GRL for adversarial feature suppression" (`model.py:143–144`). 정정(2026-06-10 reconciler): 초판의 "making the encoder produce …"는 오류 — student path는 `latent_visible.detach()`로 encoder와 gradient가 차단되므로(`model.py:1123–1126`) 역전된 gradient는 student decoder(+ student mask token/decoder pos-enc)에만 도달하고 encoder에는 도달하지 않는다. Student가 anomaly-identity 정보를 표현하지 못하게 되어 anomaly를 정상처럼 복원 → teacher와의 discrepancy가 anomaly에서 증폭된다. `mae_anomaly/model.py:129–190`

- `grl_target_mode='window'` (exp271): all patches in anomaly window share target=1. `mae_anomaly/config.py:131`
- `grl_adaptive_lambda=True` (exp271): λ_GRL auto-balanced — trainer 내 **inline gradient-ratio 공식** `λ_GRL = clamp(‖∇_w L_total_main‖ / (‖∇_w L_GRL_cls‖ + 1e-4), 0, 10)` (w = student decoder 마지막 파라미터; `trainer.py:752–760`). 정정(fixer-2, BLK-001): `loss.py:683`의 VQGAN-style `compute_adaptive_lambda`와는 **별개 코드** — 그 함수는 discriminator 전용이며 GRL은 호출하지 않는다 (§2.6 참조).
- `grl_cls_lr_ratio=0.1` (exp271): classifier LR = 0.1 × main LR. `mae_anomaly/config.py:150`
- `grl_use_focal=True` (exp271): focal loss (γ=2) on classifier. `mae_anomaly/config.py:147`

### Feature Matching (FM)

When `use_feature_matching=True` (exp271 default): L2 distance between `teacher_hidden` and `student_hidden` on normal masked patches. FM is a **training loss only** — it is NOT part of the anomaly score as of 2026-06-01. `mae_anomaly/config.py:167–180`, `mae_anomaly/scoring.py:237`

### EMA Teacher Output (optional, default off)

`use_teacher_output_ema=False` (exp271). When enabled post-warmup, a weight-EMA copy of the teacher decoder modules provides a smoother discrepancy target. The EMA is applied only to `teacher_decoder + teacher_output_projection + (shared_decoder)`, not the encoder. `mae_anomaly/model.py:507–528`, `mae_anomaly/config.py:271–279`

### RevIN (optional, default off)

`use_revin=False` (exp271). Per-window instance normalization. Requires `normalize_mode='zscore'` (raises ValueError with 'minmax'). `mae_anomaly/model.py:311–334`

### SCAD (optional, default off)

`use_scad=False` (exp271). Supervised Contrastive Anomaly Discrimination — replaces GRL with contrastive loss on student projection head. Mutually exclusive with GRL. `mae_anomaly/config.py:183–213`, `mae_anomaly/loss.py:583–680`

### Adversarial Discriminator (optional, default off)

`use_discriminator=False` (exp271). 1D CNN PatchDiscriminator with spectral normalization distinguishes real vs student-generated patches. Mutually exclusive with GRL. `mae_anomaly/model.py:96–126`, `mae_anomaly/config.py:246–261`

### Weight Initialization

Xavier uniform for Linear layers, constant (0/1) for LayerNorm. CNN projection gets reduced gain (0.5) for stability. `mae_anomaly/model.py:567–576`

---

## 2. Loss Functions

All losses in `mae_anomaly/loss.py`, class `SelfDistillationLoss`.

### 2.1 Teacher Reconstruction Loss

MSE on masked positions only:

```
L_recon = mean_over_batch[ sum_{masked positions}( (teacher_output - original)^2 ) / (n_masked × F) ]
```

Where `F` is number of features, masked positions = where `mask==0`. This is the primary training signal during warmup. `mae_anomaly/loss.py:172–179`

### 2.2 Output Discrepancy Loss (patch-level, `patch_level_loss=True`, exp271)

Per-patch discrepancy:
```
patch_disc[b, p] = mean_{masked timesteps in patch p, features}( (teacher_output - student_output)^2 )
```

The discrepancy target is `teacher_output.detach()` (or EMA teacher output if `use_teacher_output_ema=True`). `mae_anomaly/loss.py:213–218`

**Normal patch loss** (minimize discrepancy): `L_normal = mean_{normal masked patches}(patch_disc) × normal_loss_weight`

**Anomaly patch loss** (maximize discrepancy with margin) — **271에서는 전체 분기 도달 불가(비활성)**:

`use_grl=True`이고 `grl_disable_anomaly_loss=True`(271 metadata 둘 다 True)이면 margin 분기에 진입하기 *전에* `anomaly_loss = torch.tensor(0.0, …)`으로 강제된다 (`loss.py:259–261`). 따라서 아래 margin 변형들은 271 학습에 **아무 영향이 없다** — `margin_type='dynamic'`, `margin=0.5`, `dynamic_margin_k=6`(271 metadata; code default는 2.0 `config.py:99`)은 metadata에 기록만 되어 있고 effect 없음. GRL classifier가 anomaly 측 분리를 전담한다.

- `margin_type='dynamic'`: dynamic margin `μ + k×σ` computed from normal patch discrepancies in the batch, then `relu(dynamic_margin - disc)`. **[271 도달 불가]**
- `margin_type='hinge'`: `relu(margin - disc)` with fixed `margin=0.5` **[271 도달 불가]**
- `margin_type='softplus'`: `log(1 + exp(margin - disc))` **[271 도달 불가]**
- `margin_type='none'`: `-disc` (unbounded push) **[271 도달 불가]**

**Total OD loss** (if `use_output_discrepancy=True`):
```
L_OD = L_normal + L_anomaly        # 271에서는 L_anomaly = 0 (loss.py:259–261) → L_OD = L_normal
       (가중치 anomaly_loss_weight=2.0은 271에서 무효; normal_loss_weight=1.0만 유효)
```

### 2.3 Feature Matching Loss

```
L_FM = mean_{normal masked patches}( ||teacher_hidden_detach - student_hidden||^2_2 / d_model )
```

When `fm_distance_metric='l2'` (exp271). FM은 두 경로 모두에서 학습에 포함된다 — 포함 위치만 다르다 (fixer-2, MAJ-001):

- `fm_adaptive_lambda=False`: loss.py 내부에서 `discrepancy_loss = normal_loss + anomaly_loss + fm_loss_weight × L_FM` (`loss.py:438`).
- `fm_adaptive_lambda=True` (**exp271**): FM은 `discrepancy_loss` 합산에서 제외되고(`loss.py:436`) trainer가 별도로 추가한다: `L_total += λ_FM_prev × fm_loss_weight × L_FM` (`trainer.py:652`). λ_FM은 trainer 내 **inline gradient-ratio 공식** `λ_FM = clamp(‖∇_w L_total_main‖ / (‖∇_w L_FM‖ + 1e-4), 0, 10)` (w = student decoder 마지막 파라미터, `trainer.py:639–653`)이며, 안정화를 위해 당 batch 계산값이 아니라 **직전 epoch 집계값**(`self._prev_epoch_fm_lambda`, `trainer.py:1301–1303` 갱신)이 적용된다. `loss.py:683`의 `compute_adaptive_lambda`(discriminator 전용)는 호출되지 않는다 (BLK-001/003).

`mae_anomaly/loss.py:412–438`

### 2.4 GRL Classifier Loss

Focal-style BCE on patch-level anomaly predictions from `AnomalyClassifierHead`. Applied to valid (masked) patches (`valid = patch_has_masked`, `loss.py:283–284`). When `grl_target_mode='window'`, all patches in an anomaly window get label=1. `mae_anomaly/loss.py:282–350`

Formula (코드 그대로, `loss.py:337–340`):
```python
_p_t = torch.exp(-_bce)              # _bce = BCEWithLogits(..., pos_weight=grl_pos_weight)
_focal = ((1 - _p_t) ** 2.0) * _bce  # γ=2
grl_cls_loss = _focal.mean()
```
즉 `focal_loss_i = (1 − exp(−BCE_i))² × BCE_i`.

⚠️ **표준 focal loss 아님 — 논문에서 "standard focal loss (Lin et al. 2017)"로 표기 금지** (fixer-2, BLK-004): 표준형은 `p_t = σ(logit)` 기반 `(1−p_t)^γ × CE`다. 여기서는 `p_t := exp(−BCE_i)`를 사용하는데, **`pos_weight`가 BCE 안에 곱해져 있으므로**(`loss.py:330–336`; exp271은 `grl_balanced_sampling=False`라 pos_weight 경로 활성, 값은 dataset별 자동 설정 — 예: SWaT metadata `grl_pos_weight=59.18`) positive 샘플에서 `exp(−BCE_w) = p_t^w ≠ p_t`가 되어 표준형과 갈라진다. 논문 method에는 이 식 그대로("focal-style modulation with `p_t = exp(−BCE)`") 기술하고 Lin et al.과의 차이를 명주할 것.

### 2.5 Total Loss (during student training phase)

정정(fixer-2, MAJ-001): GRL과 FM의 adaptive lambda는 **서로 다른 별개 값**(λ_GRL ≠ λ_FM)이며, FM은 `fm_adaptive_lambda` 값과 무관하게 항상 학습에 포함된다(포함 위치만 다름).

```
# exp271 (fm_adaptive_lambda=True, grl_adaptive_lambda=True):
L_total = L_recon + L_OD
          + λ_FM_prev  × fm_loss_weight(=1.0 default) × L_FM    (trainer.py:652)
          + λ_GRL_prev × grl_loss_weight(=0.2)        × L_GRL   (trainer.py:762–763)

# fm_adaptive_lambda=False 경로 (참고; 271 아님):
L_total = L_recon + (L_OD + fm_loss_weight × L_FM)              (loss.py:438)
          + λ_GRL_prev × grl_loss_weight × L_GRL                (trainer)
```

- `λ_GRL = clamp(‖∇_w L_total_main‖ / (‖∇_w L_GRL_cls‖ + 1e-4), 0, 10)` (`trainer.py:752–760`)
- `λ_FM  = clamp(‖∇_w L_total_main‖ / (‖∇_w L_FM‖ + 1e-4), 0, 10)` (`trainer.py:639–653`)
- 두 λ 모두 w = student decoder 마지막 파라미터 기준이고, 적용값은 직전 epoch 집계값(`_prev_epoch_grl_lambda`/`_prev_epoch_fm_lambda`, `trainer.py:1298–1306`).

During teacher-only warmup: `L_total = L_recon` only.

After warmup: anomaly_loss is ramped via warmup_factor (linear from 0→1 over `max(teacher_only_warmup_epochs // 5, 2)` epochs after warmup end — 271에서는 `max(250//5, 2) = 50` epochs). 정정(2026-06-10 reconciler): 초판의 "`warmup_epochs//5`"는 변수 혼동 — 기준은 LR warmup(`warmup_epochs=10`)이 아니라 teacher-only warmup(250)이다 (`trainer.py` `_compute_warmup_factor`: `warmup_length = max(student_start // 5, 2)`, `student_start = config.teacher_only_warmup_epochs`). `mae_anomaly/trainer.py:336–348`

### 2.6 Adaptive Lambda — 세 가지 별개 경로 (정정: fixer-2, BLK-001/BLK-003)

> 초판/1차 정정판의 "VQGAN-style 공식이 discriminator와 FM 모두에 사용된다"는 서술은 **오류**.
> adaptive lambda는 **서로 독립적인 3개 코드 경로**가 있고, 공식도 각각 다르다. 공유되는 것은
> "student decoder 마지막 파라미터에서의 gradient norm 비교 + clamp [0,10] + prev-epoch 적용" 패턴뿐이다.

**(1) Discriminator λ_adv — VQGAN-style, `loss.py:683–728` `compute_adaptive_lambda`. exp271 비활성.**
```
λ_adv = (||∇_w L_normal|| + ||∇_w L_anom_forward||) / (||∇_w L_adv|| + δ),  δ=1e-4, clamp [0,10]
```
w = `student_output_projection.weight`. **이 함수의 유일한 호출처는 discriminator 학습 경로뿐** (`trainer.py:608–615`; `use_discriminator=True` + `config.adaptive_lambda` 조건). exp271은 `use_discriminator=False`이므로 이 경로 전체가 미사용. GRL/FM은 이 함수를 import만 될 뿐 호출하지 않는다.

**(2) GRL λ_GRL — trainer inline, `trainer.py:746–765`. exp271 활성.**
```
λ_GRL = clamp(||∇_w L_total_main|| / (||∇_w L_GRL_cls|| + 1e-4), 0, 10)     (trainer.py:752–760)
적용: L_total += λ_GRL_prev × grl_loss_weight × L_GRL_cls                    (trainer.py:762–763)
```
w = `student_decoder.parameters()` 마지막 원소. 분자는 (1)처럼 normal/anomaly 분해 합이 아니라 **현재 누적 main loss 전체의 gradient norm 단일값**.

**(3) FM λ_FM — trainer inline, `trainer.py:639–655`. exp271 활성.**
```
λ_FM = clamp(||∇_w L_total_main|| / (||∇_w L_FM|| + 1e-4), 0, 10)           (trainer.py:643–647)
적용: L_total += λ_FM_prev × fm_loss_weight × L_FM                           (trainer.py:652)
```

공통: (2)(3)의 적용값은 당-batch 계산값이 아닌 **직전 epoch 집계값**(`_prev_epoch_grl_lambda` / `_prev_epoch_fm_lambda`, 초기값 1.0 `trainer.py:189–190`, epoch 말 갱신 `trainer.py:1298–1306`; batch 값은 모니터링용 로깅만). 논문 method에 λ를 기술할 때 (2)(3)의 공식을 사용해야 하며, (1)의 VQGAN-style 공식을 GRL/FM에 귀속시키면 안 된다.

---

## 3. Anomaly Scoring Pipeline

Single source of truth: `mae_anomaly/scoring.py`.

### 3.1 Forward Pass (Inference)

The evaluator uses a **leave-one-out masking** approach: 윈도우당 50개의 마스킹 패턴(패치 p만 마스킹, 나머지 49개 가시)에 대해 각 패치의 reconstruction/discrepancy를 계산한다. 구현: `Evaluator._compute_patch_scores_all_patches` (`mae_anomaly/evaluator.py:1647`).

정정(fixer-2, BLK-002): 이것은 **50회 순차 독립 forward가 아니다**. 마스킹 패턴들을 **batch 차원으로 확장**해 병렬 처리한다 — docstring: "Optimized: All patches processed in a single forward pass by expanding batch dimension" (`evaluator.py:1650`). 실제 코드는 `sequences.unsqueeze(1).expand(...)`로 `(batch_size × patch_batch_size)` 크기의 확장 배치를 만들어 forward한다 (`evaluator.py:1801–1818`). `patch_batch_size=2`로 분할 처리되는데, 이는 메모리 대역폭 관리용 순수 batching 파라미터로 **수치 결과에 영향이 없다** (2026-05-28 HARD-LOCK 주석, `evaluator.py:1703–1717`: "does NOT affect numerical results").

논문 기술 시 주의: (a) 추론 절차는 "50개 leave-one-out 마스킹 패턴을 batch 확장으로 병렬 평가"로 기술할 것. (b) 연산량(FLOPs) 관점에서는 윈도우당 forward 연산이 단일-pass 대비 ~50×인 사실 자체는 유효하다(발표 p13의 비용 한계 언급과 정합) — 다만 "순차 50회 호출" 식의 wall-clock 서술은 금지.

Outputs stored in `PatchScoresBundle` (file: `mae_anomaly/types.py:50`):
- `recon`: (n_windows, num_patches) — teacher reconstruction error per patch
- `disc`: (n_windows, num_patches) — output-level student-teacher discrepancy per patch
- `student_recon`: (n_windows, num_patches) — student reconstruction per patch
- `fm`: Optional (n_windows, num_patches) — hidden-level feature-matching distance per patch
- `labels`, `sample_types`, `anomaly_types`: (n_windows,) per-window metadata

### 3.2 Anomaly Score Formula (mode='adaptive', exp271)

As of 2026-06-01, FM is NOT included in the inference score (training loss only):

```python
# mae_anomaly/scoring.py:223–264
eps = 1e-4
recon_mean = recon.mean() + eps
disc_mean  = disc.mean()  + eps
scaled_disc = disc × (recon_mean / disc_mean)   # scale disc to recon magnitude
ratio = score_recon_disc_ratio  # default 4.0 (config.py:223)
student_error = scaled_disc / ratio             # down-weight disc: recon:disc = 4:1
score = recon + student_error
```

Pre-warmup gate: when `epoch <= teacher_only_warmup_epochs`, `student_error = 0`, `score = recon`. Determined by `is_prewarmup_epoch(config, epoch)` (`mae_anomaly/scoring.py:111–135`).

### 3.3 Patch → Point-Level Aggregation

Multiple windows cover each timestep (due to overlapping sliding windows with stride << seq_length). Each patch in each window maps to its corresponding timesteps via `_build_aggregation_map`. Default aggregation: **mean** over all patch scores covering a timestep. `mae_anomaly/evaluator.py:229–292`

```
point_score[t] = mean over all (window, patch) pairs covering timestep t of patch_score[w, p]
```

Coverage 유도 (정정: fixer-2, MAJ-002): seq_length=500, patch_size=10 (271 metadata; 초판의 5는 오류), test stride=49.

- 한 윈도우 안에서 타임스텝 t를 포함하는 패치는 **정확히 1개**다 (패치는 윈도우 내 비중첩 분할: 50 패치 × 10 스텝 = 500).
- 따라서 t당 score 수 = **t를 덮는 윈도우 수** ≈ seq_length / test_stride = 500/49 ≈ **10.2** → "~10회 평균". 초판의 "10 windows × patch-position coverage" 곱셈 인수는 오류(불필요).
- stride 49는 patch_size 10과 서로소이므로, t가 윈도우마다 다른 패치 위치 잔차(`(k·S) mod P`)에 놓여 **10개 score의 패치 맥락이 다양화**된다 — 이는 개수가 아니라 다양성 논리다 (`resolve_test_stride` docstring, `utils/experiment.py:16–34`).
- test stride 49의 적용 조건: `resolve_test_stride`는 `sliding_window_test_stride`가 **양수면 그 값을 그대로 사용**하고, 비양수(271 metadata는 sentinel `-1`)일 때만 `seq_length // 10 − 1 = 49` 공식을 적용한다 (`utils/experiment.py:35–39`).

### 3.4 Other Score Modes

- `mode='default'`: `recon + lambda_disc × disc` (direct weighted sum). `mae_anomaly/scoring.py:286–293`
- `mode='ratio_weighted'`: `recon × (1 + disc / median(disc))`. `mae_anomaly/scoring.py:296–304`

---

## 4. Data Pipeline

### 4.1 Dataset Taxonomy

**Active in paper** (not excluded):

정정(2026-06-10 reconciler): 초판 표의 loader명/split/feature 수를 271 실측·registry 기준으로 정정.
271이 실제 사용한 loader는 `DATASET_LOADERS` registry 키 기준 (`loaders.py:2688+`; queue entry dataset 키 + `summary.json` results key로 확인).

| Dataset | Loader (271 actual) | Format | Features (271 metadata `num_features`) | Split Strategy | Notes |
|---------|----------------|--------|----------|----------------|-------|
| SWaT A1A2 | `load_swat_a1a2_raw` (registry key `SWaT_A1A2`, `loaders.py:2690`) | CSV | **45** (학습 모델 입력 실측; §주1) | A1 all + front 50% A2 → train; back 50% A2 → test (`loaders.py:2018`) | 초판의 `load_swat_combined`(`loaders.py:23`)은 legacy 키 `swat_A1A2`용 — 271 미사용 |
| WaDi A1 | `load_wadi_14days_raw('A1')` (registry key `WaDi_14days_A1`, `loaders.py:2697`) | CSV | 123 | **14days 전체 + attack 앞 50% → train; attack 뒤 50% → test** (`loaders.py:2201`: `train_len = n_14d + n_atk // 2`) | 초판의 "14days → train; attack → test"는 오류 (attack 앞 절반도 train) |
| WaDi A2 | `load_wadi_14days_raw('A2')` (registry key `WaDi_14days_A2`, `loaders.py:2698`) | CSV | 123 (원본 127 sensor 중 all-NaN 4개 drop — `prepare_raw_datasets.py` `handle_nan`) | same | same |
| PSM | `load_psm` | CSV | 25 | **orig train 전체 + orig test 앞 50% → train; orig test 뒤 50% → test** (`loaders.py:1686–1693`, `// 2` 분할; train_ratio 0.8007은 결과값) | 초판의 "80% train, 20% test"는 분할 규칙 오기 (80%는 우연의 결과) |
| SMD (×28) | `load_smd_simple(machine)` (registry key `SMD_simple_<machine>`, `loaders.py:2810–2812`) — per-machine 독립 학습 | txt | **29–36 per machine** (metadata 실측 22/28; raw 38에서 machine별 constant-col 제거) | orig train 전체 + test 앞 50% → train; test 뒤 50% → test (`loaders.py:1153`) | `summary.json` key `SMD_simple_machine-1-4` 확인; 초판의 `load_smd`(`loaders.py:876`)·"38 per machine"은 271 미사용/오기 |
| SMAP (×54 channels) | `SMAP_simple_<ch>` registry → `_load_smap_msl_simple_single` (`datasets/loaders.py:2527`) — per-channel 독립 학습 | CSV | 25 | per-channel: orig train + test 앞 ~50%(safe-cut) → train; 뒤 ~50% → test | 271 결과 dir도 per-channel (`SMAP/G-7` 등). safe-cut 코드 확정(fixer-2, NOTE-004): `safe_cut_margin: int = 10` 기본값(`loaders.py:2527`) — 50% 지점이 anomaly region에서 10스텝 이내면 `_find_safe_cut_point`(`loaders.py:1050`)가 region 밖으로 cut을 밀어냄 (`loaders.py:2591–2596`) |
| MSL (×27 channels) | `MSL_simple_<ch>` registry → 동일 | CSV | 55 | same | same |

**§주1 (SWaT 45 features)**: 학습된 271 SWaT 모델의 입력 차원은 **45** — 근거 ① metadata `config.num_features=45` (full/excl22 동일), ② `best_config.json: num_features=45`, ③ checkpoint 실측 `patch_embed.weight=(512, 450)`=Linear(10×45→512). 45 = 원본 51 sensor − combined(A1+A2) 기준 constant 6개 {P202, P401, P404, P502, P601, P603} (2026-06-10 재계산으로 정확히 일치). ⚠️ 단, 현 machineA의 raw CSV(51 features) + 현행 `load_swat_a1a2_raw`(constant 제거 없음) 경로는 51을 반환 — 학습 당시 source-machine 데이터/경로와의 차이로, 재현 시 반드시 재확인 필요 (재현성 미해결 플래그).

**[논문 제외 대상]** Simulation dataset: synthetically generated 275K timestep × 8 feature server metrics with 9 anomaly types (6 value-based, 3 pattern-based). Gaussian smoothing is used internally in `_generate_phase_jitter` and `_apply_regime_transitions` for simulation data complexity. `dataset_sliding.py:86–93` [논문 제외 대상: R33, R34]

**[논문 제외 대상]** Exathlon dataset: `load_exathlon(app)`, 6 app-level traces. `loaders.py:1376` [논문 제외 대상: R33]

### 4.2 Normalization

**Exp271 canonical**: `normalize_mode='minmax'`, `minmax_range='0_1'` (min-max scale train to [0,1], tight-clip test to [0,1]). `mae_anomaly/config.py:29–37`

Normalization is fitted on the **train portion only** to prevent data leakage.

**Per-entity normalization** (concat multi-entity datasets — SMD/SMAP/MSL/Exathlon): each entity (machine/channel/app) is normalized independently using its own train segment. Introduced 2026-06-02 to fix whole-array normalization crushing small-scale entities. `CHANGELOG.md:107–120`

Per-entity fit uses `_standardize_per_feature(signals, train_end)` in float64 for numerical stability. `mae_anomaly/dataset_sliding.py:86–125`

`normalize_mode='zscore'` uses same function. `minmax_range='neg1_1'` + `minmax_clamp_min/max=±4.0` is an option for NPSR-style scaling (not exp271).

### 4.3 Label Handling — Semi-supervised / PU perspective

Labels (`point_labels`) are used during **training** as weak supervision:

1. **`force_mask_anomaly=True`** (exp271): Anomaly patches are prioritized in the masking budget, so the model is forced to reconstruct anomaly positions. Normal patches that happen to be in the budget are also masked. `mae_anomaly/model.py:975–1002`
2. **`loss.py` patch classification**: `patch_has_anomaly` and `patch_is_normal` separate the discrepancy loss gradient direction per patch. Anomaly patches push discrepancy up (or are handled by GRL); normal patches push it down. `mae_anomaly/loss.py:244–248`
3. **GRL classifier target**: `grl_target_mode='window'` — window-level anomaly flag (1 if any masked position has anomaly). `mae_anomaly/config.py:131`
4. **Labels are NOT used at inference**: The anomaly score is computed purely from reconstruction/discrepancy signals. The training-time label usage makes this a **semi-supervised** setup (labels available only for training data; test labels are held-out ground truth for evaluation only).

**설정 vs 구현의 구분** (정정: fixer-2, BLK-005 정합화 — RESEARCH_SYNTHESIS §②와 동일 3단 프레이밍):

1. **설정(가정, R11)**: "대부분 unlabeled(이상 여부 미상) + 소수 labeled anomaly" — 기존 unsupervised는 분포는 학습하지만 소수의 핵심 labeled anomaly를 활용하지 못한다는 것이 문제 의식.
2. **main 실험 구현(FACT, R13)**: 구현상 train 구간(원본 train + 원본 test 앞 50% 편입)의 **모든 샘플에 라벨이 존재**하며, train 내 anomaly(실측 0.52–6.20%, 데이터셋별 상이 — EXPERIMENT_PROTOCOL_TRUTH §①)는 전부 label이 제공된다. 즉 main 실험은 R11 설정의 **label 가용성 상한(upper-bound) 케이스**다. 라벨이 학습에 개입하는 지점은 위 1–3(`force_mask_anomaly` / loss 방향 분기 / GRL 타겟)이 전부.
3. **라벨 희소화 sweep(계획, R32)**: label 비율을 낮춰 일부 anomaly가 unlabeled 상태로 train에 잔류하는 **R11 설정의 일반 케이스**를 검증할 계획 (전용 실험 미실행 — RESEARCH_SYNTHESIS §④ 참조).

따라서 "엄밀한 PU setting(positive + unlabeled만 존재)"은 아니며, anomaly 라벨로 masking/loss 방향을 유도하는 방식은 standard PU learning보다 weakly/semi-supervised에 가깝다. 초판의 "minority, ~5%" 표기는 실측치(0.52–6.20%)로 대체.

### 4.4 Sliding Window Dataset

`mae_anomaly/dataset_sliding.py` — class `SlidingWindowDataset`.

- Window size (`seq_length`): 500 timesteps (exp271)
- Patch size: **10** (271 metadata `patch_size=10`; 초판의 5는 code default `config.py:56` — 271과 다름), giving **50** patches per window (`num_patches=50`)
- Train stride: 21 (exp271, metadata `sliding_window_stride=21`)
- Test stride: auto-resolved as `seq_length // 10 - 1 = 49` (sentinel -1 → `resolve_test_stride`). `mae_anomaly/utils/experiment.py:16–38`
- `epoch_offset=True`: each epoch uses a random offset in [0, stride) to shift the train window grid, providing data augmentation.
- Run boundaries: for multi-entity / multi-file datasets, windows are not allowed to cross entity boundaries. `loaders.py` emits `run_boundaries` and `SlidingWindowDataset` respects them.

Dataset batches return 5-tuples: `(sequences, window_labels, point_labels, sample_types, anomaly_types)`. `mae_anomaly/trainer.py:497`

---

## 5. Training Loop

File: `mae_anomaly/trainer.py`, class `Trainer`.

### 5.1 Optimizer

AdamW (fused, CUDA kernel), `betas=(0.9, 0.99)`, `lr=1e-3`, `weight_decay=1e-3`. Bias/LayerNorm/mask-token parameters have `weight_decay=0`. GRL classifier uses a separate lower LR (`grl_cls_lr_ratio=0.1 × main_lr`). `mae_anomaly/trainer.py:113–165`

### 5.2 Learning Rate Schedule

Linear warmup for `warmup_epochs=10`: `LinearLR(start_factor=1e-4)` → 시작 LR = 1e-3 × 1e-4 = **1e-7**, 10 epoch에 걸쳐 1e-3까지 선형 증가 (`trainer.py:168–174`). 이후 `CosineAnnealingLR(T_max = num_epochs − 10)` (`trainer.py:175–183`). 논문 hyper-parameter 표에는 "near-zero"가 아닌 정확한 시작 LR(1e-7)을 기재할 것 (fixer-2, MIN-001).

### 5.3 Warmup Phase (teacher-only)

For the first `teacher_only_warmup_epochs` epochs — code default는 `-1`(auto: `num_epochs // 2`, `trainer.py:43–48`)이나 **271 config에서는 명시적 250** (metadata `teacher_only_warmup_epochs=250`, `num_epochs=500`; 초판의 "25/50-epoch" 가정은 Set A 기준 오류):
- `model.forward(teacher_only=True)` — student decoder, GRL, SCAD head are all skipped.
- `loss.py` receives `teacher_only=True` → only reconstruction loss is computed, disc/FM/GRL = 0.
- The student decoder weights are at random init during this phase.

After warmup: `warmup_factor` ramps anomaly_loss from 0 to 1 over `max(teacher_only_warmup_epochs//5, 2)` post-warmup epochs. `mae_anomaly/trainer.py:336–348`

Optional: `use_teacher_warmup_early_stop=False` (exp271). When enabled, warmup ends early based on `recon_snr` plateau detection.

### 5.4 Best Epoch Selection

Metric: `best_epoch_metric='pak_auc_f1'` (exp271) — PA%K AUC of F1 with per-K threshold re-optimization, integrated over K=0..100. `mae_anomaly/config.py:291–295`

Evaluated every `eval_interval=5` epochs. Per-epoch scores saved as `epoch_NNN_scores.npz` containing `adaptive_score`, `teacher_recon_error`, `discrepancy_error`, `fm_error`. Best checkpoint: `best_checkpoint.pt`.

**Critical 2026-06-08 fix**: Final `experiment_metadata.json["metrics"]` is recomputed from `npz@best_epoch` (not from a second `evaluate()` call which can diverge). `docs/POST_MORTEMS/2026-06-08_finalize_wrong_epoch_metadata.md`

### 5.5 AMP

Mixed precision training (`use_amp=True`, `amp_dtype='bf16'`). Requires CUDA capability >= 8.0. No GradScaler needed for bf16. `mae_anomaly/trainer.py:203–228`

### 5.6 Threshold Selection

정정(fixer-2, MAJ-003): "F1-optimal point on the ROC curve"라는 표현은 부정확/혼동 유발(ROC는 F1-optimal point를 직접 제공하지 않음). 정확한 절차: test set point-level score에 대해 `roc_curve`로 threshold 격자(fpr, tpr, thresholds)를 얻고, 그 격자 위에서 `find_f1_optimal_idx`가 (class-count 기반으로 fpr/tpr에서 precision/recall을 유도해) **F1을 최대화하는 threshold를 선택**한다 (`evaluator.py:215–226` 함수 정의, `evaluator.py:928–930` 호출). 이진화는 strict `>` (`evaluator.py:931`).

⚠️ **Oracle threshold 경고** (fixer-2, MAJ-005): 이 threshold는 **test label을 알아야만 최적화 가능한 oracle(best-F1) threshold**다. 이 threshold에 의존하는 지표(`precision`, `recall`, `f1_score`, `f1_t`, `pa_{K}_f1`, affiliation@optimal 등)를 논문 테이블에 실을 때는 반드시 "best-F1 (oracle) threshold" 표기를 병기해야 한다. label-leak 없는 대안으로 anomaly-ratio threshold 변형(`_ar` suffix, `(1−anomaly_ratio)`-quantile, §6.2)이 병산된다.

---

## 6. Evaluation System

### 6.1 Evaluation Entry Point

`mae_anomaly/evaluator.py` — class `Evaluator`. Single source of truth for metric computation shared by MAE and baseline pipelines: `compute_full_metric_set` at `mae_anomaly/evaluator.py:864`.

### 6.2 Metric Set

`compute_full_metric_set` computes approximately 133 scalar metrics plus 4 diagnostic lists (`_per_k_*`). Key metrics:

**Threshold-based core** (on `point_scores[eval_mask]`):
- `roc_auc`: sklearn ROC AUC
- `prc_auc`: sklearn average precision (AP)
- `precision`, `recall`, `f1_score`: at F1-optimal threshold
- `optimal_threshold`: threshold value
- `f1_t`, `precision_t`, `recall_t`: time-series F1 (Tatbul et al. NeurIPS 2018) at optimal threshold

**Per-K PA%K** (보고 키: K = 0, 5, 10, ..., 100 — **step 5, 21개 값**; `PA_K_VALUES = list(range(0, 101, 5))`, `evaluator.py:831`, 사용처 `evaluator.py:2111, 2139`):
- `pa_{K}_f1`, `pa_{K}_precision`, `pa_{K}_recall`, `pa_{K}_roc_auc`, `pa_{K}_prc_auc`

⚠️ 해상도 구분 (fixer-2, MAJ-008): **Per-K 보고 키는 step=5(21점)**이지만, **PA%K AUC 적분은 K = 0..100 step=1(101점, `k_values = np.arange(0, 101)`, `evaluator.py:1034`) trapz**다. 두 해상도는 다르므로 논문에서 "pa_{K} 키들을 적분한 값이 pak_auc"라고 서술하면 안 된다.

PA%K: If ≥ K% of an anomaly segment is detected, the entire segment is credited as detected (Kim et al. AAAI 2022). Implementation: `mae_anomaly/evaluator.py:361–406`

**PA%K AUC integrated** (trapz over K=0..100, **step 1 — 101점**, `evaluator.py:1034`):
- `pak_auc_f1`: Per-K threshold re-optimization (tadpak method, Kim et al. AAAI 2022). Primary selection/ranking metric in exp271.
- `pak_auc_f1_raw`: Fixed threshold across K values
- `pak_auc_prc_auc`, `pak_auc_roc_auc`, `pak_auc_f1_t`, `pak_auc_precision`, `pak_auc_recall`

**VUS-PR / VUS-ROC** (Paparrizos et al. VLDB 2022):
- Threshold-free volume metrics. Skipped during per-epoch training eval (`lite=True`) due to ~40s/call cost. `mae_anomaly/evaluator.py:706–748`

**Affiliation metrics** (Huet et al. KDD 2022):
- `affiliation_precision`, `affiliation_recall`, `affiliation_f1`

**R-based F1** (Tatbul et al. NeurIPS 2018):
- `r_based_f1`

**Anomaly-ratio threshold variants** (`_ar` suffix):
- Same threshold-based metrics re-computed at the `(1 - anomaly_ratio)`-quantile threshold. `mae_anomaly/evaluator.py:752–815`

### 6.3 SWaT Exclusion Region 22

SWaT has a known ambiguous attack region (#22). The `excl22` evaluation path is run in parallel, restricting `eval_mask` to exclude region 22. Both full and excl22 metrics are saved. `mae_anomaly/evaluator.py:897` (eval_mask parameter).

### 6.4 Lite vs Full Mode

`lite=True` in `compute_full_metric_set`: skips only VUS-PR/ROC. Affiliation, R-F1, and AR variants are always computed (cheap). Used for per-epoch training eval. `mae_anomaly/evaluator.py:969–981`

---

## 7. Research Design Decisions (from docs + history)

### 7.1 Architecture Choices

- **Self-attention only decoders** (`use_transformer_encoder_decoder=True`): decoders use `nn.TransformerEncoder` (no cross-attention from encoder), matching MAE original style. This decision is baked into the default config. `mae_anomaly/config.py:68`
- **Separate mask tokens**: Teacher and student decoders have separate learned mask tokens (`shared_mask_token=False`). `mae_anomaly/config.py:63`
- **Shallow student decoder**: Intentionally shallow to amplify the discrepancy gap relative to the deeper teacher. This asymmetry is the core of the self-distillation design. code default는 student 1층/teacher 4층(`config.py:44–45`)이나 **271 config에서는 student 2층/teacher 3층** (metadata `num_student_decoder_layers=2`, `num_teacher_decoder_layers=3` — 비대칭(teacher>student)은 유지되나 층수는 초판 표기와 다름).
- **`force_mask_anomaly=True`**: Ensures the model is always challenged with anomaly reconstruction, preventing easy avoidance. `mae_anomaly/config.py:315–320`
- **GRL over adversarial discriminator**: GRL (`use_grl=True`) is the canonical exp271 anomaly-aware training mechanism. The discriminator (`use_discriminator`) is an alternative that is mutually exclusive with GRL.

### 7.2 Scoring Design

- **FM dropped from inference score** (2026-06-01): Feature matching proved to be a training stabilizer only. Including FM in the score introduced an extra numerics-dependent component that caused score-path divergences. Score simplification: `recon + scaled_disc / 4`. `mae_anomaly/scoring.py:232–265`
- **Pre-warmup gate** (`force_recon_only`): During teacher-only warmup, the student decoder is random-init. Including its disc/FM in the score would corrupt per-epoch best-epoch selection. The gate reduces score to teacher reconstruction only for epochs ≤ warmup. `mae_anomaly/scoring.py:111–135`

### 7.3 Normalization Decisions

- **Per-entity normalization** for concat datasets (2026-06-02): Whole-array normalization across entities was causing entity-identity bias instead of anomaly detection. `CHANGELOG.md:107–120`
- **Min-max not z-score** (exp271 canonical): `normalize_mode='minmax'` with `minmax_range='0_1'`. The `neg1_1` option with test-clamp was added for NPSR-style experiments.

### 7.4 Evaluation Rigor

- **PA%K with per-K threshold re-optimization** (tadpak): The `pak_auc_f1` metric allows each K value its own optimal threshold, which is more rigorous than applying a single threshold across all K. `mae_anomaly/evaluator.py:990–1086`
- **`>` vs `>=` threshold convention**: All binarization uses strict `>` following Kim et al. AAAI 2022 Eq. 1. Fixed in 2026-06-03 to unify across all code paths. `CHANGELOG.md:49–63`
- **K=0 guard** (2026-06-03): Removed a bug where K=0 PA%K auto-credited any zero-detection-count segment as detected (`ratio >= 0` was always true). `CHANGELOG.md:58`

### 7.5 Key Post-Mortems

| Date | Bug | Fix | File |
|------|-----|-----|------|
| 2026-05-29 | FM score omitted in per-epoch eval path (silent Optional=None) | `scoring.py` single source, `PatchScoresBundle` typed container, required-kw `force_recon_only` | `docs/POST_MORTEMS/2026-05-29_fm_score_omission.md` |
| 2026-06-01 | Pre-warmup student disc/FM leaked into score (eval had no warmup gate) | `is_prewarmup_epoch()` + `force_recon_only` required-kw | `docs/POST_MORTEMS/2026-06-01_prewarmup_student_score_leak.md` |
| 2026-06-08 | `experiment_metadata.json` diverged from best-epoch score (second evaluate() path) | Final metadata from `npz@best_epoch` only | `docs/POST_MORTEMS/2026-06-08_finalize_wrong_epoch_metadata.md` |
| 2026-06-03 | excl22 VUS/Aff/AR ignored eval_mask; K=0 guard missing; `>` vs `>=` inconsistency | eval_mask threading, K=0 fix, `>` unification | `CHANGELOG.md:49–63` |

---

## 8. Experiment Configuration — Exp271 Canonical

정정(2026-06-10 reconciler): 초판은 "Set A config (exp271 archetype)"로 code default/Set A preset 값을 나열했으나 오류.
**exp271 = Set C 기반 + config override** (`summary.json: config_set='C'`; `configs/queue_fullrerun_20260601_190603.json` exp271 entry).
아래 표의 Value는 전부 **271 metadata 실측값** (`experiment_metadata.json` config 블록, 전 37 entity 동일 — `paper/01_research_understanding/271_CONFIG_TRUTH.md` §II 전수표 참조). code default와 다른 항목은 명기.

| Parameter | Value (271 actual) | 비고 (code default와의 차이) |
|-----------|-------|--------|
| `seq_length` | 500 | = default (`config.py:16`) |
| `patch_size` | **10**, `num_patches`=**50** | default 5/100 (`config.py:55–56`)과 다름 |
| `patchify_mode` | **`'linear'`** | default `'patch_cnn'` (`config.py:57`)과 다름 |
| `mask_after_encoder` | `True` | = default |
| `d_model` | **512**, `nhead`=8, `dim_feedforward`=**2048** | default 128/512 (`config.py:41,49`)과 다름 |
| `num_encoder_layers` | **4** | default 2 (`config.py:43`)와 다름 |
| `num_teacher_decoder_layers` | **3** | default 4 (`config.py:44`)와 다름 |
| `num_student_decoder_layers` | **2** | default 1 (`config.py:45`)과 다름 |
| `masking_ratio` | 0.15 (50패치 중 round(50×0.15)=**8**개 마스킹) | = default (`config.py:51`; 개수 산식 `model.py:986`) |
| `normalize_mode` | `'minmax'`, `minmax_range='0_1'` | = default |
| `use_grl` | `True`, **`grl_disable_anomaly_loss=True`** | anomaly-side OD loss를 0으로 강제 (`loss.py:259–261`) |
| `grl_target_mode` | `'window'` | |
| `grl_loss_weight` | 0.2 | |
| `grl_adaptive_lambda` | `True` | |
| `grl_cls_lr_ratio` | 0.1 | |
| `use_feature_matching` | `True` | 학습 loss 전용; score 미포함 (`scoring.py:237`) |
| `fm_distance_metric` | `'l2'` | |
| `fm_adaptive_lambda` | `True` | |
| `use_output_discrepancy` | `True` | 271에서는 normal-side만 유효 |
| `margin_type` | `'dynamic'`, `dynamic_margin_k`=**6** | **271에서 도달 불가(무효)** — `grl_disable_anomaly_loss=True`로 anomaly_loss=0 (`loss.py:259–261`); k의 code default는 2.0 (`config.py:99`) |
| `anomaly_score_mode` | `'adaptive'` | |
| `score_recon_disc_ratio` | 4.0 | |
| `best_epoch_metric` | `'pak_auc_f1'` | `eval_interval=5` |
| `num_epochs` | **500** | default 50 (`config.py:264`)과 다름 |
| `teacher_only_warmup_epochs` | **250** (명시) | default -1=auto num_epochs//2 (`config.py:268`) |
| `warmup_epochs` (LR ramp) | 10 | |
| `batch_size` | **1024** | Set A/C preset 512와 다름 (override) |
| `learning_rate` | 1e-3 | `weight_decay`=1e-3 |
| `amp_dtype` | `'bf16'`, `use_amp=True` | |
| `use_revin` | `False` | |
| `use_teacher_output_ema` | `False` | |
| `use_teacher_warmup_early_stop` | `False` | |
| `sliding_window_stride` | 21 (train) | test stride: -1 → `resolve_test_stride`=`seq_length//10−1`=49 (`utils/experiment.py:16–39`) |
| `num_features` | dataset별 25–123 | §4.1 표 및 `271_CONFIG_TRUTH.md` §3a |

---

## 9. Scripts Summary

| Script | Purpose |
|--------|---------|
| `scripts/run_base_experiments.py` | Main training launcher. Supports Sets A/B/C, per-dataset or batch. GPU train + bg-worker CPU eval+viz pipeline. |
| `scripts/ablation/run_ablation.py` | Ablation experiment runner with config files. |
| `scripts/visualize_all.py` | Re-generate visualizations for an experiment directory. |
| `scripts/monitor_status.py` | Live training queue status monitor. |
| `scripts/resume_dedup.py`, `v2`, `v2b` | Resume/dedup experiment queues. |
| `scripts/reexp_phase*.py` | Re-experiment phases for metadata correction. |
| `scripts/backfill_prewarmup_recon_only.py` | Offline fix for pre-warmup score contamination (2026-06-01). |
| `scripts/reexp_comprehensive_audit.py` | 370-cell full consistency audit. |
| `scripts/reexp_auditB_forensic.py` | Weight-level forensic audit for 4 flip cells. |

---

## 10. Output Artifacts

Per-experiment directory structure (under `results/experiments/{timestamp}_{suffix}/{DatasetGroup}/{Scenario}/`):

- `best_model.pt`, `best_config.json`, `training_histories.json`
- `epoch_metrics.json` — per-epoch scalar metrics
- `experiment_metadata.json` — final metrics (from `npz@best_epoch`)
- `checkpoints/best_checkpoint.pt`, `latest_checkpoint.pt`
- `epoch_scores/epoch_{NNN}_scores.npz` — `adaptive_score`, `teacher_recon_error`, `discrepancy_error`, `fm_error` (point-level arrays)
- `visualization/best_model/` — 15+ PNG visualizations (ROC, PRC, confusion matrix, score distribution, reconstructions, score contribution charts)
- `visualization/epoch_metrics/` — training curve plots

---

## REQUEST / FEEDBACK

**REQUEST to config-forensics agent**:
The directives specify to only use exp271 config settings and ignore unused options. The canonical exp271 is Set A with the `CONFIG_PRESETS['A']` overrides on top of the defaults in `mae_anomaly/config.py`. Please confirm which specific experiment ID "271" corresponds to and whether all of the `config.py` defaults not overridden by Set A presets are canonical for that experiment. Key uncertainty: the `lambda_disc` field (`config.py:97`, default 2.0) is used in `anomaly_score_mode='default'` but exp271 uses `'adaptive'`, so `lambda_disc` is unused at inference. Similarly, `margin=0.5` (the static margin) is superseded by `margin_type='dynamic'`.

> **RESOLVED (reconciler, 2026-06-10)**: exp271 = `271_20260602_020545_271canon_baseline`. 전제("Set A archetype")가 오류 — 실제는 **Set C 기반 + config override** (`summary.json: config_set='C'`; `configs/queue_fullrerun_20260601_190603.json`). canonical 값은 `271_CONFIG_TRUTH.md` §II의 metadata 전수표가 정본. `lambda_disc=2.0`은 **anomaly score 경로(adaptive 모드)에서는 미사용이나, 진단용 `compute_detailed_losses`(`evaluator.py:2017`)가 mode 무관하게 `recon + 2.0·disc`를 계산해 `best_model_detailed.csv`에 기록** — score·지표·best-epoch 선정에는 무참여 (α-m2 정정 2026-06-10, 271_CONFIG_TRUTH §VI 정합). `margin`/`margin_type`/`dynamic_margin_k`는 superseded 정도가 아니라 **도달 불가** (`grl_disable_anomaly_loss=True` → `loss.py:259–261`에서 anomaly_loss=0; margin 계산 경로 자체가 호출 안 됨).

**REQUEST to config-forensics agent**:
Confirm `teacher_only_warmup_epochs` for exp271. The default is `-1` (auto: `num_epochs // 2 = 25` for 50 epochs). This means the first 25 epochs train teacher only, then epochs 26–50 train both.

> **RESOLVED (reconciler, 2026-06-10)**: 271 metadata `teacher_only_warmup_epochs=250` (명시), `num_epochs=500`. 즉 epoch 0–249 teacher-only, 250–499 양쪽 학습. "25/50" 가정은 Set A 기준 오류.

**FEEDBACK**:
The `grl_pos_weight=19.0` default (config.py:134) is noted as "automatically set from actual dataset anomaly ratio by run_base_experiments.py". The runtime override in `run_base_experiments.py` should be verified to ensure this is dataset-specific and not the fixed 19.0 in all cases.

> **부분 RESOLVED (fixer-2, 2026-06-10)**: 271 metadata 실측으로 dataset-specific 자동 설정 확인 — 예: SWaT `experiment_metadata.json` config `grl_pos_weight=59.1814...` (고정 19.0 아님; train anomaly ratio 기반 값). 전 entity 분포의 전수 확인은 271_CONFIG_TRUTH 담당 범위.

**FEEDBACK**:
The `eval_fm_weight` field (config.py:217) has comment "UNUSED in score since 2026-06-01". This is confirmed by `scoring.py:237` where `fm_active` is hardcoded to `False`. Any paper claim about FM in the inference score is incorrect as of the current codebase.

---

## 부록: 2026-06-10 reconciler 정정 목록 (271_CONFIG_TRUTH 기준 정합화)

판정 근거 전체는 `paper/99_reviews/p1_reconciliation_r1.md` 참조. 1차 소스 = 271 metadata (`results/experiments/271_20260602_020545_271canon_baseline/**/experiment_metadata.json`) + 코드 직접 추적.

1. **exp271 정체**: "Set A archetype" → **Set C 기반 + override** (헤더 노트, §1, §8).
2. **patchify_mode**: `patch_cnn` → **`linear`** (§1 diagram, Patchify Modes, §8).
3. **patch_size/num_patches**: 5/100 → **10/50** (§1, §3.3, §4.4, §8).
4. **d_model/dim_feedforward**: 128/512 → **512/2048** (§1, Encoder/Decoder, §8).
5. **num_encoder_layers**: 2 → **4**; **teacher/student decoder**: 4/1 → **3/2** (§1, Encoder/Decoder, §7.1, §8).
6. **num_epochs/teacher_only_warmup_epochs**: 50/25 → **500/250** (§5.3, §8, REQUEST 해소).
7. **batch_size**: 512 → **1024**; **dynamic_margin_k**: 2.0 → **6** (§8).
8. **dynamic margin/anomaly OD loss**: "exp271 default로 동작" 뉘앙스 → **도달 불가(무효)** — `grl_disable_anomaly_loss=True`로 `anomaly_loss=0` (`loss.py:259–261`) (§2.2, §8).
9. **masking 패치 수**: round(100×0.15)=15 masked/85 visible → **round(50×0.15)=8 masked/42 visible** (`model.py:986`) (§1).
10. **GRL 작용 대상/방향**: "encoder가 anomaly-uninformative 표현" → **student decoder**의 anomaly-identity feature **suppression** (encoder는 detach로 차단; `model.py:129–144, 1123–1126`) (§1 GRL).
11. **anomaly-loss ramp**: `warmup_epochs//5` → **`max(teacher_only_warmup_epochs//5, 2)` = 50 epochs** (`trainer.py` `_compute_warmup_factor`) (§2.5).
12. **데이터 차원 (B,500,8)**: 8-feature는 simulation(R33 제외) — **F=dataset별 25–123** (§1).
13. **§4.1 dataset 표**: loader명(`load_swat_a1a2_raw`/`load_wadi_14days_raw`/`SMD_simple_*` registry), WaDi·PSM 분할 규칙(`//2` front-half-to-train), SMD features 38→**29–36**, SWaT features→**45**(checkpoint 실측; 재현성 플래그 포함) 정정.

---

## 부록 2: 2026-06-10 fixer-2 정정 목록 (adversarial review p1_codebase_synthesis_r1 반영)

판정 근거 전체는 `paper/99_reviews/p1_codebase_synthesis_fixlog_r2.md` 참조. 모든 항목 코드 직접 재검증 완료.

1. **[BLK-001/003] Adaptive lambda 3경로 분리** (§2.6 전면 재작성, §1 GRL bullet, §2.3, §2.5): `compute_adaptive_lambda`(`loss.py:683–728`)는 **discriminator 전용**(유일 호출처 `trainer.py:608–615`, exp271 비활성). GRL λ는 `trainer.py:752–760`, FM λ는 `trainer.py:639–653`의 **각각 별도 inline grad-ratio 공식** `clamp(‖∇L_main‖/(‖∇L_x‖+1e-4), 0, 10)` + prev-epoch 적용(`trainer.py:1298–1306`). "VQGAN-style이 GRL/FM에도 사용" 서술 삭제.
2. **[BLK-002] 추론 leave-one-out** (§3.1): "각 패치를 순서대로 마스킹해 N회 forward" → **마스킹 패턴 batch-차원 확장 forward** (`evaluator.py:1650` docstring, `1801–1818` 구현; `patch_batch_size=2`는 수치 무영향 메모리 분할 `1703–1717`).
3. **[BLK-004] GRL focal loss 비표준형 플래그** (§2.4): 식 `(1−exp(−BCE))²×BCE` (`loss.py:337–340`)는 pos_weight 내장 BCE 기반이라 표준 focal loss(Lin et al. 2017, p_t=σ(logit))와 다름 — "standard focal loss" 표기 금지.
4. **[MAJ-001] Total loss λ 분리** (§2.5): λ_GRL ≠ λ_FM 별도 표기 + FM은 `fm_adaptive_lambda` 양 경로 모두 학습에 포함(위치만 다름, `loss.py:436/438`).
5. **[MAJ-002] coverage 유도** (§3.3): t당 score 수 = 덮는 윈도우 수 ≈ 500/49 ≈ 10.2 (윈도우 내 t의 패치는 1개); "× patch-position coverage" 인수 삭제; stride sentinel 조건(`utils/experiment.py:35–39`) 명시.
6. **[MAJ-003/005] threshold** (§5.6): "ROC curve의 threshold 격자에서 `find_f1_optimal_idx`로 F1 최대화 threshold 선택"으로 정밀화 + **oracle threshold 논문 표기 의무** 경고 추가.
7. **[MAJ-008] PA%K 해상도** (§6.2): Per-K 보고 키 step=5(21점, `evaluator.py:831`) vs AUC 적분 step=1(101점, `evaluator.py:1034`) 구분 명시.
8. **[MIN-001] LR warmup** (§5.2): 시작 LR = 1e-7 (`start_factor=1e-4`, `trainer.py:171`) 명시.
9. **[MIN-002] decoder pos-enc 공유** (§1): 단일 인스턴스 공유 사실 + 고정 sinusoidal buffer라 수치 무영향 명시 (`model.py:263–279, 343–346`).
10. **[NOTE-004] SMAP/MSL safe-cut 코드 확정** (§4.1): `safe_cut_margin=10` (`datasets/loaders.py:2527`), `_find_safe_cut_point` (`loaders.py:1050, 2591–2596`) — "±10" 근거 코드로 확인.
11. **[BLK-005 정합화] §4.3**: "PU-like / ~5%" 서술 → 3단 프레이밍(설정 R11 / main 구현 = label 상한 케이스 R13 / 희소화 sweep 계획 R32) + 실측 train anomaly 0.52–6.20%로 교체.
12. **FEEDBACK(grl_pos_weight)**: dataset-specific 자동 설정 metadata 실측(예: SWaT 59.18)으로 부분 해소.
