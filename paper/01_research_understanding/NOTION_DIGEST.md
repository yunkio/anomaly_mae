---
phase: 1
agent: notion-analyst
directives: [T1, R2, R26, M8]
last_modified: 2026-06-10
revision: r3 — r2(p1_digests_r1 리뷰 BLOCKER 2 + MAJOR 4 + MINOR 6 전수 반영, fixer-3) + r3(재리뷰 β r2 NMr-1: §IV-11 "127 3개소"→4개소, fixer-5)
fix_log: paper/99_reviews/p1_digests_fixlog_r2.md + paper/99_reviews/p1_fixlog_r3.md
---

# NOTION DIGEST — TSMAE Research Understanding

> **R2 핵심 경고**: 이 문서의 "[Notion의 주장]" 블록은 Notion이 그렇게 서술/주장한 것이다.
> 그 내용이 논문의 실제 contribution 구조나 서사로 그대로 채택될지는 **Phase 3에서 별도 판단**한다.
> "[검증된 사실 후보]"는 실험 구성·수치·모델 목록·데이터셋 목록이며, 이 중 baseline 모델 reference와
> 데이터셋 reference는 **R26에 따라 truth 등급**으로 취급한다.
> 단, 최종 서지 표기는 **Phase 4에서 공식 소스(DBLP/IEEE/ACM DL)로 재확인**한다.

---

## 소스 페이지

- **Page 0 MAE**: "0 MAE 프로젝트 개요" — `https://app.notion.com/p/31387856b20781cd8d4ed14df7f65470` (as of 2026-05-31)
- **Page B**: "Baseline Comparison: 22 Active Models + 4 Weakly-Supervised · 9 Datasets · 2 Conditions" — `https://app.notion.com/p/32087856b2078112b500c81664181ee7` (as of 2026-06-06)

---

# I. 방법론 페이지 Digest (Page 0 MAE)

## I-1. 모델 정체성 및 설계 철학

**[Notion의 주장]** 모델명: **Self-Distilled Masked Autoencoder for Multivariate Time Series Anomaly Detection**.

Teacher–Student 자기증류 메커니즘과 Masked Autoencoder를 결합한 다변량 시계열 이상탐지 프레임워크. Notion은 이 모델이 다음 세 가지 고질적 문제를 정면 해결하는 디자인이라 주장한다:
- (i) 정상 분포의 풍부한 표현 부족
- (ii) 이상 신호의 약함
- (iii) 학습 가능한 anomaly 라벨의 희소성

**[Notion의 주장]** 세 가지 핵심 설계 원칙:
1. Masked Reconstruction: 입력 시계열의 15%를 패치 단위로 마스킹 후 가시 패치만으로 복원 → 복원 오차가 1차 anomaly evidence
2. Self-Distillation: 동일 인코더 출력을 Teacher 디코더(3층)와 Student 디코더(2층)로 분기 → capacity gap이 anomaly에서 더 크게 벌어짐
3. Patch-based Processing: patch_size=10, 총 50 patches per window

**[검증된 사실 후보]** 본 baseline 고정값: `seq_length=500`, `patch_size=10`, `num_patches=50`, `masking_ratio=0.15` → **round(50×0.15) = 8개 패치 고정 마스킹** (batch 전체 균일 — Standard MAE 인코더의 uniform input length 제약; Page 0 §3.3.2. 상단 callout의 "약 8개"는 부정확한 축약 표현). `num_epochs=500`, `teacher_only_warmup_epochs=250`. 실험 ID: **exp271**.

---

## I-2. Four Core Contributions (Notion 원문 발췌)

> **Phase 3 판단 사안**: Notion이 C1~C4로 정리한 contribution 구조가 논문의 실제 contribution으로 그대로 사용될지, 재정의될지는 별도 판단이 필요하다.

**[Notion의 주장 — C1]**
"Context-Aware Time-Series Representation via Masking": 입력의 15% 패치를 마스킹, 나머지 85% 가시 패치만으로 bidirectional 맥락 기반 복원 학습. 일반적 "다음 시점 예측" 패러다임과 달리, MAE는 context-aware bidirectional reconstruction을 강제 → 더 풍부한 정상 분포 표현 학습 → 표현학습 강점을 이상탐지 성능으로 전이.

**[Notion의 주장 — C2]**
"Capacity-Gap Self-Distillation (비대칭 Teacher–Student)": 동일 인코더 출력을 Teacher 디코더(3층) / Student 디코더(2층)로 분기. Student 디코더는 인코더 latent의 `gradient detach` 사용 → 인코더가 Student의 conflicting objective에 오염되지 않음. 외부 teacher 없이 자기 자신만으로 distillation 구성. Notion은 이를 "self-distillation 디자인"으로 서술.

**[Notion의 주장 — C3]**
"Discrepancy as Primary Anomaly Signal": Teacher–Student 차이를 두 단계에서 측정:
- Output Discrepancy (OD): `||teacher_output - student_output||^2` (masked positions, patch-level), Dynamic margin `μ + 6σ`
- Feature Matching (FM, hidden-level, L2 + adaptive λ): `||teacher_hidden.detach - student_hidden||^2` (masked normal patches 한정)

Notion은 FM adaptive λ가 FM gradient와 main gradient의 자동 균형을 VQGAN-style로 보장한다고 서술. **단, 2026-06-01 이후 FM은 anomaly score 계산에서 제외되고 학습 손실(regularizer)로만 유지됨** — 이는 Notion 페이지에 명시된 내용이다.

**[Notion의 주장 — C4]**
"Semi-Supervised Anomaly Awareness via Gradient Reversal": 학습 셋에 소량 존재하는 anomaly 라벨을 GRL(Gradient Reversal Layer, Ganin et al. 2016)로 활용. Student hidden 위에 anomaly classifier를 얹고 GRL이 gradient를 반전 → student hidden에서 anomaly 정보 적극 제거. Notion은 이 모델을 "semi-supervised setting"으로 만드는 결정적 요소라 서술. GRL이 없으면 C3의 신호가 약해진다고 주장.

**[Notion의 주장 — 상호작용]**
C1(masking) = 학습 신호 문제 설정 / C2(self-distill) = 신호 증폭 메커니즘 / C3(discrepancy+FM) = 측정 도구 / C4(GRL) = anomaly 라벨 활용 채널. "네 요소는 분리될 수 없으며 전체가 함께 작동하는 단일 디자인"이라 주장.

---

## I-3. 아키텍처 상세

**[검증된 사실 후보 — Forward Flow]**

```
Input (batch, 500, F)
→ Patchify + Linear Embed (50, batch, d_model)
→ Random/Force-Mask-Anomaly 15% patches
→ Remove Masked Patches (visible only: ~42 patches)
→ + Positional Encoding (original positions)
→ Transformer Encoder (4 layers, Pre-Norm + GELU, dropout=0.15)

→ [Teacher path] Insert Teacher Mask Tokens + Decoder PE
→ Teacher Decoder (3 layers) → teacher_hidden
→ Teacher Output Projection → teacher_output

→ [Student path] latent.detach() + Insert Student Mask Tokens + Decoder PE
→ Student Decoder (2 layers) → student_hidden
→ Student Output Projection → student_output
→ GRL → Anomaly Classifier (window-label, pos_weight Focal-BCE)
```

**[검증된 사실 후보 — 텐서 형상 (simulation F=8, d_model=128 기준)]**

| 단계 | 형상 |
|------|------|
| 입력 | (batch, 500, 8) |
| Patchify | (batch, 50, 10, 8) |
| Linear Embed (flatten+Linear(80→128)+LN) | (50, batch, 128) |
| 마스킹 후 | (~42, batch, 128) |
| Encoder 출력 | (~42, batch, 128) |
| Teacher/Student 디코더 입력 | (50, batch, 128) |
| Output Projection | (batch, 50, 80) |
| Unpatchify | (batch, 500, 8) |
| GRL Classifier | cls_logits (batch, 50) |

**[검증된 사실 후보 — d_model Dynamic 매핑 (Set C)]**
공식: `d_model = min{d ∈ {128,192,256,384,512} : d ≥ 10×F}`, cap=512

| 데이터셋 | F | d_model | dim_feedforward |
|---------|---|---------|----------------|
| simulation | 8 | 128 | 512 |
| PSM | 25 | 256 | 1024 |
| SMD | 38 | 384 | 1536 |
| SWaT | 51 | 512 | 2048 |
| Exathlon | 19 | 192 | 768 |
| WaDi A1 | 123 | 512 | 2048 |
| WaDi A2 | 127 † | 512 | 2048 |

> † **원천 간 모순 (r2 주석, r3 개소수 정정)**: Page 0은 WaDi A2 F=**127**로 기재하나(§1.2 지원 데이터셋 표 — I-7이 전사한 표 — · d_model 표 · num_features 표 · §5.2.1, **4개소**), **Page B §2.1 표는 123이며 Page B 전문에 "127"은 단 한 번도 등장하지 않는다**. 코드 검증(`p1_reconciliation_r1.md` §III)으로 **123이 확정값**: exp271 metadata `config.num_features=123`, raw CSV 124 cols(=123+label); 127은 all-NaN 4개 컬럼(`2_LS_001_AL` 등) drop **이전**의 원본 sensor 수. d_model=512(cap) 결론에는 영향 없음. → §IV-11 참조.

**[검증된 사실 후보 — 마스킹 상세 (Page 0 §3.3.2)]**
50개 패치 중 round(50×0.15)=**8개를 항상 마스킹** (batch 균일 고정). 학습 시 `force_mask_anomaly=True` — anomaly 포함 패치 우선 마스킹. Priority 공식:
```
priority_p = 1[patch_p contains anomaly] × 1000 + η_p,  η_p ~ U(0,1)
masked patches = TopK_8(priority)
```
anomaly 패치가 budget(8) 이하이면 anomaly 전부 마스킹 + 나머지는 normal에서 random; budget 초과이면 **random subset 8개만 마스킹**되고 초과분은 visible로 인코더 컨텍스트에 포함.

**[검증된 사실 후보 — 디코더 구조]**
`use_transformer_encoder_decoder=True` → **두 디코더(Teacher/Student) 모두 TransformerEncoder (self-attention only, cross-attention 없음)**. Mask token이 이미 시퀀스에 삽입되어 있으므로 cross-attention 불필요 — Page 0 §3.3.4 명시. (아키텍처 그림/서술 작성 시 직접 영향.)

---

## I-4. 학습 파이프라인 상세

**[검증된 사실 후보 — 학습 단계 (총 500 epochs)]**

| Epoch | 단계 | 활성 손실 | GRL λ |
|-------|------|----------|-------|
| 0~9 | LR warmup | L_recon (Teacher만) | 0 |
| 10~249 | Teacher-only warmup | L_recon (Teacher만) | 0 |
| 250~499 | Student + Adversarial | L_recon + L_normal_OD + L_FM_L2 + L_GRL_cls | sigmoid 0→0.9999 |

**[검증된 사실 후보 — Optimizer]**
- AdamW: betas=(0.9, 0.99), lr=1e-3, weight_decay=1e-3
- Bias/LayerNorm/mask_token: weight_decay=0 (별도 param group)
- GRL classifier: lr = 1e-4 (main lr × 0.1, 별도 param group)
- batch_size=512 (Set C), warmup_epochs=10

**[검증된 사실 후보 — LR Schedule]**
`SequentialLR`: LinearLR (epoch 0~9: 1e-4→1e-3) + CosineAnnealingLR (epoch 10~500: 1e-3→0)

**[검증된 사실 후보 — GRL λ Sigmoid Ramp-up (Ganin et al. 2016 schedule)]**
`λ = 2/(1+exp(-10p)) - 1` where `p = max(0, (epoch-250)/(500-250))`

| Epoch | λ_GRL |
|-------|-------|
| 0~249 | 0.000 |
| 250 | 0.000 |
| 300 | 0.762 |
| 350 | 0.965 |
| 400 | 0.995 |
| 499 | ≈1.000 |

**[검증된 사실 후보 — Teacher-only Warmup 메커니즘 (Page 0 §3.5)]**
Epoch < 250 구간에서 **Student 디코더 forward는 수행되지만**, loss.py의 `teacher_only=True` 플래그로 모든 discrepancy/FM/GRL 손실 항이 비활성화된다. Total loss = L_recon 단독 → 인코더와 Teacher만 학습 (Student forward는 metric 계산용 유지, backward 영향 없음). — 논문 학습 절차 서술에 필요한 메커니즘.

**[검증된 사실 후보 — 학습 인프라 (Page 0 §4)]**
- **AMP**: `use_amp=True`, `amp_dtype='bf16'` (2026-05-27 fp16→bf16 변경 — fp32 exponent range로 SCAD logsumexp/focal exp(-bce)에 안전, GradScaler 불필요, CUDA capability ≥8.0 필요. 'fp16'은 pre-flip 재현용).
- **eval_interval=5**: 테스트 평가 epoch 간격 (epoch 5, 10, ..., 500).
- **random_seed=42**: set_seed로 torch/numpy/random 모두 설정.
- **Config validation 룰 (trainer.py 5종)**: ① `teacher_only_warmup_epochs<0` → `num_epochs//2` auto, ② `freeze_teacher_after_warmup=True` → warmup 강제 `num_epochs//2`, ③ `use_grl`+`use_discriminator` 동시 활성 금지, ④ `use_grl=True` 시 `patch_level_loss=True` 필수, ⑤ `shared_mask_token=True`+`freeze_teacher_after_warmup=True` 금지.

---

## I-5. 손실 함수

**[검증된 사실 후보 — 총 손실]**
```
L_total = L_recon + w_n × L_normal_OD + 0 × L_anomaly_OD + λ_FM_eff × L_FM_L2 + λ_GRL_eff × L_GRL_cls
```
여기서 `w_n=1.0`, `λ_FM_eff = λ_FM_adp × fm_loss_weight (=1.0)`, `λ_GRL_eff = λ_adp × 0.2`.
Anomaly OD term은 0 (`grl_disable_anomaly_loss=True`).

**[검증된 사실 후보 — Reconstruction Loss]**
Teacher output에 대한 masked timestep MSE만. Student reconstruction loss 없음.

**[검증된 사실 후보 — Output Discrepancy (Normal Loss)]**
마스킹된 정상 패치 집합 P_n에 대해 patch-level `||teacher_out.detach - student_out||^2`. Teacher output detach → gradient는 Student에만 흐름.

**[검증된 사실 후보 — Dynamic Margin (Anomaly OD, 본 baseline에서 disabled)]**
`m_dyn = μ_n + 6σ_n` (정상 패치 disc 평균 + 6σ). `anomaly_loss_weight=2.0`. **`use_grl=True`이므로 현재 비활성**이나 코드에 유지됨.

**[검증된 사실 후보 — Feature Matching (FM, L2)]**
정상 패치 한정: `(1/|P_n|) Σ (1/d) ||teacher_hidden.detach - student_hidden||^2`. FM Adaptive λ: `λ_FM_adp = clip(||∇_w L_main|| / (||∇_w L_FM|| + 1e-4), 0, 10)`.

**[검증된 사실 후보 — GRL Classifier Loss (Focal-BCE)]**
`L_GRL = (1/|P_mask|) Σ (1-p_t)^2 × BCE_{w+}(l_p, y_p)` where `p_t = exp(-BCE)`, `w+ ≈ 7.29`.
Window-mode: 윈도우 내 anomaly 1개라도 있으면 모든 마스킹 패치에 label=1.

**[검증된 사실 후보 — GRL Adaptive λ의 anchor]**
`λ_GRL_adp = clip(||∇_w L_main|| / (||∇_w L_GRL_cls|| + 1e-4), 0, 10)`, `λ_GRL_eff = λ_GRL_adp × 0.2` — 여기서 **w는 student decoder의 마지막 weight (두 gradient 경로의 공통 anchor)**. 매 step gradient magnitude 비교로 자동 균형(VQGAN-style); `grl_loss_weight=0.2`는 GRL 영향을 main의 약 20%로 제한하는 conservative scaling (Page 0 §3.3.5).

---

## I-6. 추론 파이프라인

**[검증된 사실 후보 — All-Patches Inference]**
각 윈도우의 50개 패치를 leave-one-out 방식으로 순차 마스킹 → **총 50회 forward pass**.
각 forward에서 마스킹 패치 1개에 대한 recon / OD / FM L2 수집.

**[검증된 사실 후보 — Adaptive Scoring Formula (2026-06-01 이후)]**
FM은 점수에서 **제외**, 학습 손실로만 유지.
```
scaled_disc = disc × (mean_recon / mean_disc)
student_error = scaled_disc / r,  r = 4  (recon:disc = 4:1)
score = recon + student_error
```
결과적으로 recon:disc = 4:1 합산. `config.score_recon_disc_ratio` (default 4.0)로 조정 가능.

**[검증된 사실 후보 — Point-level Aggregation]**
Patch-level score → 덮는 timestep 분배 → 같은 timestep 덮는 모든 (window, patch) score의 **평균**으로 timestep별 최종 anomaly score 산출.

**[검증된 사실 후보 — Best Epoch 선정 기준]**
`pak_auc_f1` = PA%K (Kim et al., AAAI 2022) AUC of F1 with per-K threshold re-optimization. K=0..100 sweep, 각 K에서 PA%K 조정 후 최적 threshold 재탐색, trapezoidal rule 적분.

---

## I-7. 데이터셋 (MAE 학습 범위)

**[검증된 사실 후보]**

| 데이터셋 | Features (F) | Train/Test 구성 | 비고 |
|---------|-------------|----------------|------|
| Simulation | 8 | 275K total (220K train + 55K test) | 9 anomaly types (6 value + 3 pattern), 7 normal complexity |
| SWaT (A1+A2) | 51 | A1 normal + A2 attack | Dual eval: full + excl_region22 |
| WaDi A1 | 123 | 14days normal + attack | — |
| WaDi A2 | 127 † | 14days normal + attack | † Page 0 원문 기재. **검증값은 123** (Page B + 코드 검증 — §IV-11) |
| SMD (28 machines) | 38 | Per-machine 50/50 split | machine-1-1 ~ machine-3-11 |
| PSM | 25 | 132K train + 87K test | — |
| Exathlon (6 apps) | 19 (FScustom) | Per-app, undisturbed→train | apps {1,2,4,5,6,9} |

**공통**: train/test stride=21, per-feature minmax (train-only fit), window=500, epoch_offset=True.

**[Notion의 주장]** SMAP/MSL은 baseline comparison pipeline에는 통합 완료(2026-05-26)되었으나 MAE 학습(run_base_experiments.py) 범위에는 아직 미등록 — "의도된 scope 분리". 출처: Hundman et al., KDD'18, DOI `10.1145/3219819.3219845`.

---

## I-8. 핵심 파라미터 목록 (config.py — exp271 baseline)

**[검증된 사실 후보]**

| 파라미터 | 값 |
|---------|---|
| seq_length | 500 |
| patch_size | 10 |
| num_patches | 50 |
| masking_ratio | 0.15 |
| mask_after_encoder | True |
| shared_mask_token | False |
| force_mask_anomaly | True |
| patchify_mode | 'linear' |
| num_encoder_layers | 4 |
| num_teacher_decoder_layers | 3 |
| num_student_decoder_layers | 2 |
| nhead | 8 |
| dropout | 0.15 |
| d_model | 'dynamic' |
| num_epochs | 500 |
| teacher_only_warmup_epochs | 250 |
| warmup_epochs | 10 |
| batch_size | 512 |
| learning_rate | 1e-3 |
| weight_decay | 1e-3 |
| use_grl | True |
| grl_mode | 'classifier' |
| grl_disable_anomaly_loss | True |
| grl_loss_weight | 0.2 |
| grl_target_mode | 'window' |
| grl_pos_weight | ≈7.29 |
| grl_use_focal | True |
| grl_cls_lr_ratio | 0.1 |
| grl_adaptive_lambda | True |
| dynamic_margin_k | 6 |
| fm_adaptive_lambda | True |
| anomaly_score_mode | 'adaptive' |
| score_recon_disc_ratio | 4.0 |
| normalize_mode | 'minmax' |
| sliding_window_stride | 21 |
| epoch_offset | True |

---

## I-9. 한계 및 향후 확장 (Notion 서술)

**[Notion의 주장 — 디자인 한계]**
1. **Train anomaly 라벨 의존성**: C4(GRL)는 학습 셋에 anomaly 라벨이 일부 존재한다는 가정. 완전 unsupervised setting에서는 GRL 비활성화 → C3에만 의존.
2. **Memory cost**: 윈도우당 50회 forward. `eval_complementary_masking=True`로 7회로 축소 가능 (정확도 약간 손실).
3. **Anomaly 라벨 noise**: label noise가 큰 환경에서 GRL 효과 약화될 수 있음.
4. **Dynamic d_model 한계**: WaDi (123/127 features)가 모두 d_model=512로 통합 → high-dim feature 처리 부족 가능성. *(127은 Page 0 원문 표기 — 검증값은 A2=123, §IV-11.)*

**[검증된 사실 후보 — 코드에 있으나 본 baseline 미사용 옵션]**
`patchify_mode='patch_cnn'` / `use_discriminator=True` / `grl_mode='wdgrl'` / `freeze_teacher_after_warmup=True` / `eval_complementary_masking=True` / `masking_ratio_anneal=True` / `use_scad=True` / `use_revin=True`

---

## I-10. Notion 정리 References (Page 0 — 출처 검증 완료라 명시)

> **(r2) 등급 분리**: R26의 truth 범위는 "**비교 대상 모델 reference + 데이터셋 reference**"다 (MASTER_ORCHESTRATION_PROMPT [R26]).
> 따라서 아래 12건 중 **baseline/데이터셋 reference인 [4], [6]–[10], [12]만 truth 등급**이며 (모두 Page B의 [B2]/[D1]–[D5]/[D8]과 동일 항목),
> **방법론 인용 [1] He(MAE), [2] Ganin(GRL), [3] Kim(PA%K), [5] Esser(VQGAN), [11] Lin(Focal)은 R26 적용 대상이 아니다** —
> 이 5건은 "[Notion의 주장 — 검증 완료 주장]" 등급 (Page 0이 "DBLP/IEEE Xplore/ACM DL/openaccess.thecvf.com 1차 출처에서 검증"이라 명시했으나, **Phase 4 공식 소스 재확인 필수**).

**[Notion의 주장 — 검증 완료 주장] 방법론 인용 5건 (R26 범위 외)**

| [N] | 참고 | 역할 |
|-----|------|------|
| [1] | He et al., "Masked autoencoders are scalable vision learners," CVPR 2022, doi:10.1109/CVPR52688.2022.01553 | MAE 원형 |
| [2] | Ganin et al., "Domain-adversarial training of neural networks," JMLR 2016, vol.17 no.59 | GRL + sigmoid λ ramp-up |
| [3] | Kim et al., "Towards a rigorous evaluation of time-series anomaly detection," AAAI 2022, doi:10.1609/aaai.v36i7.20680 | PA%K AUC F1 |
| [5] | Esser et al., "Taming transformers," CVPR 2021, doi:10.1109/CVPR46437.2021.01268 | Adaptive λ (VQGAN-style) |
| [11] | Lin et al., "Focal loss for dense object detection," ICCV 2017, doi:10.1109/ICCV.2017.324 | Focal loss |

**[truth 등급 — R26] baseline 모델/데이터셋 reference 7건**

| [N] | 참고 | 역할 |
|-----|------|------|
| [4] | Xu et al., "Anomaly transformer," ICLR 2022 | 핵심 baseline (= Page B [B2]) |
| [6] | Su et al., KDD 2019, doi:10.1145/3292500.3330672 | SMD 출처 (= [D3]) |
| [7] | Abdulaal et al., KDD 2021, doi:10.1145/3447548.3467174 | PSM 출처 (= [D4]) |
| [8] | Jacob et al., PVLDB 14(11) 2021, doi:10.14778/3476249.3476307 | Exathlon 출처 (= [D5]) |
| [9] | Goh et al., CRITIS 2016, doi:10.1007/978-3-319-71368-7_8 | SWaT 출처 (= [D1]) |
| [10] | Ahmed et al., CySWATER 2017, doi:10.1145/3055366.3055375 | WaDi 출처 (= [D2]) |
| [12] | Hundman et al., KDD 2018, doi:10.1145/3219819.3219845 | SMAP/MSL 출처 (= [D8]) |

---

# II. 비교 실험 페이지 Digest (Page B)

## II-1. 실험 개요

**[검증된 사실 후보]**
- 페이지 제목: "Baseline Comparison: 22 Active Models + 4 Weakly-Supervised (구현 완료) · 9 Datasets · 2 Conditions (incl. SMAP/MSL Pattern A+B)"
- Active 모델: **22개 (unsupervised)** + **4개 (weakly-supervised, 구현 완료·결과 보류)**
- 조건(Condition): **2개** (Q1: minmax full / Q3: minmax normalonly). Q2/Q4(zscore) 폐기.
- Dataset runs (Page B Total 문장 원문 산식): "Pattern A **39 base dataset runs + 2 new SMAP/MSL concat runs = 41 runs per condition × 2 conditions × 22 active models = 1,804** (model, dataset, condition) cells". Pattern B 추가: 162 per-channel entries.
  - *(r2 주석 — 원천 내부 불일치)*: Page B Snapshot callout은 "9 actual base datasets ... = **39 dataset runs (Pattern A)**"라 하고, Total 문장은 "39 base **+ 2 SMAP/MSL = 41**"이라 한다 (39에 SMAP/MSL 포함 여부 상호 모순). 본 digest는 Total 문장의 41 해석을 채택.
- 모든 unsupervised(neural/SOTA) 모델: **10 epoch** (2026-06-06 통일). weakly-supervised: **50 epoch**.
  - *(r2 주석 — 원천 내부 불일치)*: Snapshot callout 원문은 "weakly-supervised **5종** 50 epoch (2026-06-06 통일)"으로 표기 — 페이지 제목/§1.1/§6.4는 **4종**이며, `nrdetector_full` 변형이 별도 존재한다(Page B §6 callout·§9 변경 이력에 "nrdetector / nrdetector_full"로 등장). 본 digest는 4종(+`nrdetector_full` 변형) 해석을 채택.
- Best-epoch 기준: `pak_auc_f1`.

---

## II-2. 모델 22개 + Weakly-Supervised 4개 목록 및 References

### [truth 등급 — R26] Simple 5개 (QuoVadisTAD 출처)

| Key | Type | 출처 |
|-----|------|------|
| `random` | Simple | Sarfraz et al., ICML 2024 [B1] |
| `sensor_range` | Simple | Sarfraz et al., ICML 2024 [B1] |
| `pca_error` | Simple | Sarfraz et al., ICML 2024 [B1] |
| `l2_norm` | Simple | Sarfraz et al., ICML 2024 [B1] |
| `nn_distance` | Simple | Sarfraz et al., ICML 2024 [B1] |

**[B1]** M. S. Sarfraz, M.-Y. Chen, L. Layer, K. Peng, and M. Koulakis, "Position: Quo Vadis, unsupervised time series anomaly detection?" in Proc. 41st Int. Conf. Mach. Learn. (ICML), Vienna, Austria, Jul. 2024, vol. 235, pp. 43461–43476. Available: https://proceedings.mlr.press/v235/sarfraz24a.html

### [truth 등급 — R26] Neural (QuoVadisTAD) 3개

| Key | Paper | Venue |
|-----|-------|-------|
| `mlp` | QuoVadisTAD [B1] | ICML 2024 Position |
| `mlpmixer` | QuoVadisTAD [B1] | ICML 2024 Position |
| `transformer` | QuoVadisTAD [B1] | ICML 2024 Position |

### [truth 등급 — R26] SOTA Legacy 7개

| Key | 논문명 | Venue | Year | Repo |
|-----|-------|-------|------|------|
| `gcn_lstm` | QuoVadisTAD-introduced 1-Layer GCN-LSTM (별도 원논문 없음) | ICML Position | 2024 | ssarfraz/QuoVadisTAD |
| `anomaly_transformer` | Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy | ICLR (Spotlight) | 2022 | thuml/Anomaly-Transformer |
| `tranad` | TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data | VLDB (PVLDB vol.15 no.6) | 2022 | imperial-qore/TranAD |
| `usad` | USAD: UnSupervised Anomaly Detection on Multivariate Time Series | KDD | 2020 | manigalati/usad |
| `dagmm` | Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection | ICLR | 2018 | (구현: TranAD repo src/models.py::DAGMM) |
| `gdn` | Graph Neural Network-Based Anomaly Detection in Multivariate Time Series | AAAI | 2021 | d-ailin/GDN |
| `omnianomaly` | Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network | KDD | 2019 | NetManAIOps/OmniAnomaly |

**IEEE References [B1–B7]:**
- **[B1]** (위 동일)
- **[B2]** J. Xu, H. Wu, J. Wang, and M. Long, "Anomaly transformer: Time series anomaly detection with association discrepancy," in Proc. ICLR, Apr. 2022. https://openreview.net/forum?id=LzQQ89U1qm_
- **[B3]** S. Tuli, G. Casale, and N. R. Jennings, "TranAD: Deep transformer networks for anomaly detection in multivariate time series data," Proc. VLDB Endow., vol. 15, no. 6, pp. 1201–1214, Feb. 2022, doi: 10.14778/3514061.3514067.
- **[B4]** J. Audibert, P. Michiardi, F. Guyard, S. Marti, and M. A. Zuluaga, "USAD: UnSupervised anomaly detection on multivariate time series," in Proc. 26th ACM SIGKDD (KDD), Aug. 2020, pp. 3395–3404, doi: 10.1145/3394486.3403392.
- **[B4b]** *(r2 추가 — 구현 출처 reference)* F. Galati, J. Audibert, and M. A. Zuluaga, "usad — Unsupervised anomaly detection on multivariate time series (PyTorch implementation)," GitHub, 2020. https://github.com/manigalati/usad — official-affiliated PyTorch 구현 (원논문 저자 Audibert/Zuluaga가 README contributor; code attribution용 entry).
- **[B5]** B. Zong et al., "Deep autoencoding Gaussian mixture model for unsupervised anomaly detection," in Proc. ICLR, Apr. 2018. https://openreview.net/forum?id=BJJLHbb0-
- **[B5b]** *(r2 추가 — 구현 출처 reference)* S. Tuli, G. Casale, and N. R. Jennings, TranAD (PVLDB 15(6) 2022, doi: 10.14778/3514061.3514067) — code: `src/models.py::DAGMM` in https://github.com/imperial-qore/TranAD. TranAD 저자의 TS-AD DAGMM reimplementation; 2026-05-25부터 dagmm baseline 구현 reference로 사용 (§IV-6 직결).
- **[B6]** A. Deng and B. Hooi, "Graph neural network-based anomaly detection in multivariate time series," in Proc. AAAI, vol. 35, no. 5, May 2021, pp. 4027–4035, doi: 10.1609/aaai.v35i5.16523.
- **[B7]** Y. Su et al., "Robust anomaly detection for multivariate time series through stochastic recurrent neural network," in Proc. 25th ACM SIGKDD (KDD), Aug. 2019, pp. 2828–2837, doi: 10.1145/3292500.3330672.

### [truth 등급 — R26] SOTA New 7개 (2026-05-19 batch, 활성)

| Key | 논문명 | Venue | Year | Repo | License |
|-----|-------|-------|------|------|---------|
| `tfmae` | Temporal-Frequency Masked Autoencoders for Time Series Anomaly Detection | ICDE (IEEE) | 2024 | LMissher/TFMAE | MIT |
| `npsr` | Nominality Score Conditioned Time Series Anomaly Detection by Point/Sequential Reconstruction | NeurIPS | 2023 | andrewlai61616/NPSR | no LICENSE |
| `timesnet` | TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis | ICLR | 2023 | thuml/Time-Series-Library | MIT |
| `dcdetector` | DCdetector: Dual Attention Contrastive Representation Learning for Time Series Anomaly Detection | KDD | 2023 | DAMO-DI-ML/KDD2023-DCdetector | no LICENSE |
| `memto` | MEMTO: Memory-guided Transformer for Multivariate Time Series Anomaly Detection | NeurIPS | 2023 | gunny97/MEMTO | no LICENSE |
| `moderntcn` | ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis | ICLR (Spotlight) | 2024 | luodhhh/ModernTCN | MIT |
| `catch` | CATCH: Channel-Aware multivariate Time Series Anomaly Detection via Frequency Patching | ICLR | 2025 | decisionintelligence/CATCH | no LICENSE |

**IEEE References [B8–B14]:**
- **[B8]** Y. Fang et al., "Temporal-frequency masked autoencoders for time series anomaly detection," in Proc. 40th IEEE ICDE, Utrecht, May 2024, pp. 1228–1241, doi: 10.1109/ICDE60146.2024.00099.
- **[B9]** C.-Y. Lai et al., "Nominality score conditioned time series anomaly detection by point/sequential reconstruction," in Adv. NeurIPS, vol. 36, Dec. 2023. https://openreview.net/forum?id=ljgM3vNqfQ
- **[B10]** H. Wu et al., "TimesNet: Temporal 2D-variation modeling for general time series analysis," in Proc. ICLR, May 2023. https://openreview.net/forum?id=ju_Uqw384Oq
- **[B11]** Y. Yang et al., "DCdetector: Dual attention contrastive representation learning for time series anomaly detection," in Proc. 29th ACM SIGKDD (KDD), Aug. 2023, pp. 3033–3045, doi: 10.1145/3580305.3599295.
- **[B12]** J. Song et al., "MEMTO: Memory-guided transformer for multivariate time series anomaly detection," in Adv. NeurIPS, vol. 36, Dec. 2023. https://openreview.net/forum?id=UFW67uduJd
- **[B13]** D. Luo and X. Wang, "ModernTCN: A modern pure convolution structure for general time series analysis," in Proc. ICLR, May 2024. https://openreview.net/forum?id=vpJMJerXHU
- **[B14]** X. Wu et al., "CATCH: Channel-aware multivariate time series anomaly detection via frequency patching," in Proc. ICLR, Apr. 2025. https://openreview.net/forum?id=m08aK3xxdJ

### [truth 등급 — R26] Weakly Supervised 4개 (구현 완료, 결과 보류, Q1-only)

| Key | 논문명 | Venue | Year | Repo | License |
|-----|-------|-------|------|------|---------|
| `deepmil` | Real-World Anomaly Detection in Surveillance Videos | CVPR (IEEE/CVF) | 2018 | WaqasSultani/AnomalyDetectionCVPR2018 | no LICENSE |
| `wetas` | Weakly Supervised Temporal Anomaly Segmentation With Dynamic Time Warping | ICCV (IEEE/CVF) | 2021 | donalee/WETAS | GPL-3.0 |
| `treemil` | TreeMIL: A Multi-instance Learning Framework for Time Series Anomaly Detection with Inexact Supervision | ICASSP (IEEE) | 2024 | fly-orange/TreeMIL | GPL-3.0 |
| `nrdetector` | Noise-Resilient Point-wise Anomaly Detection in Time Series Using Weak Segment Labels | KDD (ACM SIGKDD) | 2025 | UCSC-REAL/NRdetector | MIT |

**IEEE References [B15–B18]:**
- **[B15]** W. Sultani, C. Chen, and M. Shah, "Real-world anomaly detection in surveillance videos," in Proc. IEEE/CVF CVPR, Salt Lake City, Jun. 2018, pp. 6479–6488, doi: 10.1109/CVPR.2018.00678.
- **[B16]** D. Lee, S. Yu, H. Ju, and H. Yu, "Weakly supervised temporal anomaly segmentation with dynamic time warping," in Proc. IEEE/CVF ICCV, Montreal, Oct. 2021, pp. 7335–7344, doi: 10.1109/ICCV48922.2021.00726.
- **[B17]** C. Liu, S. He, H. Liu, and S. Li, "TreeMIL: A multi-instance learning framework for time series anomaly detection with inexact supervision," in Proc. IEEE ICASSP, Seoul, Apr. 2024, pp. 7510–7514, doi: 10.1109/ICASSP48485.2024.10447536.
- **[B18]** Y. Wang et al., "Noise-resilient point-wise anomaly detection in time series using weak segment labels," in Proc. 31st ACM SIGKDD (KDD), Toronto, Aug. 2025, doi: 10.1145/3690624.3709257.

---

## II-2b. 실험 조건 상세 (r2 신설 — Page B §1.2 + 2026-06-02/06-04 정합성 패스)

> 논문 §Experiments의 fair-comparison 서술과 reproducibility에 직결되는 내용 (p1_digests_r1 NM-3 반영).

**[검증된 사실 후보 — 모델별 하이퍼파라미터 preset (Page B §1.2, "default preset, 전 데이터셋 동일")]**
- **Simple 5 (§1.2.1)**: `random` seed=None(2026-06-04: seed=42 고정 제거 → driver가 **5회 독립 run → mean±std 집계**); `sensor_range` `(0,1)` 고정 boolean max; `pca_error` `pca_dim='auto'`+`svd_solver='full'` (paper auto branch); `l2_norm` ord=2; `nn_distance` euclidean 1-NN.
- **Neural 3 (§1.2.2, paper yaml 그대로)**: `mlp` seq_len=5·embed=32·lr=1e-3; `mlpmixer` seq_len=5·embed=128·lr=2e-4; `transformer` seq_len=5·embed=128·heads=1·lr=1e-3. 전부 epochs=10(2026-06-06 통일)·bs=512(paper).
- **SOTA legacy 7 (§1.2.3)**: `gcn_lstm` seq_len=5·weight_decay=0(2026-06-04); `anomaly_transformer` **win_size=100·d_model=512**·n_heads=8·e_layers=3·bs=128; `tranad` seq_len=10·d_ff=16·**lr=1e-4**(constants.py lr 재현값; paper-text 0.01은 run-code 값 아님; 2026-06-05 config-layer)·activation **LeakyReLU**(2026-06-04, 이전 ReLU); `usad` seq_len=5·latent=40; `dagmm` n_window=5·lr=1e-4 (TranAD-reimpl, 2026-05-25); `gdn` seq_len=5·embed=64·head zero-pad(2026-06-04)·**batch=32**(run.sh 재현값; argparse default 128은 wrong config-layer; 2026-06-05); `omnianomaly` **seq_len=100**·hidden=500·z_dim=3.
- **SOTA new 7 (§1.2.4)**: `tfmae` win=100·temporal/freq mask 0.25/0.4; `npsr` win=100·z_dim=10 (+2026-06-04 channel-engineering: zero-std drop+pad-to-head+one-hot → default head 정상 동작); `timesnet` win=100·d_model=128·d_ff=128·e_layers=3·top_k=3; `dcdetector` win=105·patch=[3,5,7]; `memto` win=100·n_memory=10·**train_stride=100**(non-overlap default)·2-phase(Phase1 3ep→K-means→Phase2); `moderntcn` win=100·patch=8/4; `catch` **win_size=192**·patch=16/8·λ_freq=0.005.
- **Weak 4종 preset (§1.2.5)**: `comparison/baseline_common.py:324–369`에서 직접 추출, 각 파라미터에 **provenance 태그 4종 분류 체계** — `[fixed]`(원논문/원코드 고정 default) / `[normalization]`(원논문 정규화 *방법*) / `[runtime-estimated]`(런타임 동적 추정) / `[impl-invented]`(official 출처 없는 구현 선택, confound 명시). 예: `deepmil` bag window=128[fixed]·ranking loss=전체 n_Nor×n_Abn cross-product(Sultani 원식; 2026-06-04 이전 paired)·encoder=WETAS DiCNN(DERIVATIVE_CITED).

**[검증된 사실 후보 — weak label 정의 (Page B §1.2.5)]**
weakly-supervised 4종은 학습 시 **weak label = `max(point label over window)`** (window/segment/bag 단위)를 사용 — **train split 한정 → leak-free**. 전용 실행 경로 `run_weak_sota_baseline_with_epoch_eval`. **Q1-only (Q3 = N/A)**: 4모델 모두 positive(이상) bag이 필요해 Q3(정상-only) 입력에서는 wrapper가 `RuntimeError` — 구조적 부적합(placeholder가 아님).

**[검증된 사실 후보 — 2026-06-04 faithfulness pass v2 (12개 모델 upstream 충실도 재정리)]**
upstream 원논문/공식 repo 재검증 + red-team 후 12개 모델 수정 (11 model file + `baseline_common.py`; epochs 전 모델 불변). 핵심:
- `tranad`: activation LeakyReLU (lr은 2026-06-05 config-layer에서 1e-4로 확정 — 위 §1.2.3).
- `gdn`: head zero-pad (batch=32는 2026-06-05 run.sh 재현값).
- `timesnet`: HP를 SMAP anomaly-detection script 값(d_model=128/d_ff=128/e_layers=3/top_k=3)으로 채택 — argparse default 512/2048은 forecasting default(거대 모델)이고 anomaly-detection TASK는 dataset별 small config 사용; 512-variant 대비 ~25% 적은 params·~55% 적은 compute. all-position score.
- `memto`: Phase2 **FRESH re-init**(no warm-start)·train_stride=100.
- `random`: seed=None + 5-run mean±std.
- 정규화 leak 재정리: **`deepmil`만 train-fit/test-transform leak-free** (유일한 leak-free weak model); `nrdetector`/`treemil`/`wetas`(WETAS family)는 원코드대로 **fit-on-test** (label-free → upstream-충실, leak 아님; wetas official `timeseries.py:37-40` 2026-06-04 복원). 이전 "4건 leak 제거" 서술은 superseded.
- 불변(UNCHANGED): moderntcn, dagmm, l2_norm, nn_distance, sensor_range, mlp, mlpmixer, transformer, omnianomaly, dcdetector, anomaly_transformer, tfmae, pca_error.

**[검증된 사실 후보 — 2026-06-02 boundary-safe TEST windowing (21개 windowing baseline)]**
test 추론의 window가 entity 경계를 넘지 않도록 21개 windowing baseline 전부 수정 (harness는 train만 boundary-safe였고 test-side 메커니즘 전무 → entity 경계를 가로지르는 window가 두 entity를 섞어 경계 부근 score 오염). 공유 helper `per_entity_concat`가 entity slice별 windowing+inference 독립 수행 → raw score concat; score 후처리는 concat 후 전체 test 기준 유지(granularity 불변). **단일파일(PSM/SWaT/WaDi)은 bit-identical NO-OP, multi-entity(SMD/MSL/SMAP/Exathlon)는 재실행 시 corrected 수치** — multi-entity 재실행 사유. upstream 자체가 entity 개별 처리(npsr "only one entity should be input at a time", wetas/deepmil per-file chunking, omnianomaly single-entity)이므로 per-entity windowing이 곧 official 충실.

---

## II-3. 데이터셋 9개 목록 및 References

### [truth 등급 — R26] Dataset References

| Dataset | Features | Runs/Condition | 원논문 | DOI |
|---------|---------|---------------|--------|-----|
| Simulation | 8 | 1 | 내부 합성 | — |
| SWaT (A1+A2) | 51 | 1 (+excl22) | Goh et al., CRITIS 2016 | 10.1007/978-3-319-71368-7_8 |
| WaDi A1 | 123 | 1 | Ahmed et al., CySWATER 2017 | 10.1145/3055366.3055375 |
| WaDi A2 | **123** | 1 | Ahmed et al., CySWATER 2017 | 10.1145/3055366.3055375 |
| SMD (28 machines) | 32–38 | 28 | Su et al., KDD 2019 | 10.1145/3292500.3330672 |
| PSM (eBay) | 25 | 1 | Abdulaal et al., KDD 2021 | 10.1145/3447548.3467174 |
| SMAP (54 channels) | 25 | 1 + 54 (Pattern B) | Hundman et al., KDD 2018 | 10.1145/3219819.3219845 |
| MSL (27 channels) | 55 | 1 + 27 (Pattern B) | Hundman et al., KDD 2018 | 10.1145/3219819.3219845 |
| Exathlon (6 apps) | 19 (FScustom) | 6 | Jacob et al., PVLDB 2021 | 10.14778/3476249.3476307 |
| TEP (Tennessee Eastman, **참고 — 비교 실험 미사용**) | 52 (process variables) | — | Rieth et al., Harvard Dataverse 2017 [D6] / Downs & Vogel 1993 [D7] | 10.7910/DVN/6C3JR1 / 10.1016/0098-1354(93)80018-I |

> **(r2) WaDi A2 Features 정정**: r1에서 **127**로 오기 — 출처(Page B §2.1 표)는 **123**이며 Page B 전문에 "127"은 등장하지 않는다. 127은 Page 0의 값으로 두 원천이 상호 모순. 코드 검증(`p1_reconciliation_r1.md` §III: exp271 metadata `num_features=123`, raw CSV 124 cols=123+label, 127=all-NaN 4컬럼 drop 이전 원본 수)으로 **123 확정**. 발표 PDF p19도 123 dim. → §IV-11.
>
> **(r2) TEP 행 추가**: Page B §2.1 표에 "TEP (Tennessee Eastman, 참고)" 행 존재 — **보유·검증 완료(로컬 4 RData files 일치), 비교 실험 미사용(Runs/Condition "—", 참고용)**. [D6]/[D7]은 이 TEP의 reference. License: §2.1 표는 "Harvard Dataverse 공개", §6.4 citation 표는 **CC0 Public Domain Dedication** 명시.

**[검증된 사실 후보 — 데이터셋 상세]**
- SWaT: SUTD iTrust Labs 신청제. License: iTrust Terms of Use. 재배포 절대 금지. SWaT.A1 & A2_Dec 2015.
- WaDi: SUTD iTrust Labs 신청제. A1 (2017-10-09) / A2 (2019-11-19 — attack scenario 재정의). iTrust Terms of Use. 재배포 금지.
- SMD: github.com/NetManAIOps/OmniAnomaly. MIT License. 28 machines × 38 features / train 708,405 / test 708,420 / anomaly 4.16%. 로컬 byte-level 일치.
- PSM: github.com/eBay/RANSynCoders. CC BY 4.0. train 132,481 + test 87,841 × 25 features / anomaly 27.76%. 로컬 byte-level 일치.
- SMAP: telemanom (khundman), canonical URL 현재 HTTP 403 → Wayback Machine 2022-10-16 snapshot 사용. Code Apache 2.0, Data license 미명시(NASA-derived, public domain 관례 가정). Pattern A stats: total 573,830 / train 355,905 / test 217,925. P-2 채널 CSV 이중 등장 → UNION 처리.
- MSL: 동일 출처. Pattern A stats: total 132,046 / train 95,271 / test 36,775. Safe-cut moved 4 channels (D-16, M-1, M-2, S-2).
- Exathlon: CC BY-NC-SA 4.0 (Data) / Apache 2.0 (Code). 93 traces × 10 apps × 2,283 raw features. FScustom 19 features. Apps {1,2,4,5,6,9}. 비상업 조건.

**IEEE Dataset References [D1–D8]:**
- **[D1]** J. Goh, S. Adepu, K. N. Junejo, and A. Mathur, "A dataset to support research in the design of secure water treatment systems," in Proc. CRITIS, Paris, Oct. 2016, pp. 88–99, doi: 10.1007/978-3-319-71368-7_8.
- **[D2]** C. M. Ahmed, V. R. Palleti, and A. P. Mathur, "WADI: A water distribution testbed for research in the design of secure cyber physical systems," in Proc. CySWATER, Pittsburgh, Apr. 2017, pp. 25–28, doi: 10.1145/3055366.3055375.
- **[D3]** Y. Su, Y. Zhao, C. Niu, R. Liu, W. Sun, and D. Pei, "Robust anomaly detection for multivariate time series through stochastic recurrent neural network," in Proc. 25th ACM SIGKDD (KDD), Anchorage, Aug. 2019, pp. 2828–2837, doi: 10.1145/3292500.3330672.
- **[D4]** A. Abdulaal, Z. Liu, and T. Lancewicki, "Practical approach to asynchronous multivariate time series anomaly detection and localization," in Proc. 27th ACM SIGKDD (KDD), Singapore, Aug. 2021, pp. 2485–2494, doi: 10.1145/3447548.3467174.
- **[D5]** V. Jacob, F. Song, A. Stiegler, B. Rad, Y. Diao, and N. Tatbul, "Exathlon: A benchmark for explainable anomaly detection over time series," Proc. VLDB Endow., vol. 14, no. 11, pp. 2613–2626, Jul. 2021, doi: 10.14778/3476249.3476307.
- **[D6]** C. A. Rieth, B. D. Amsel, R. Tran, and M. B. Cook, "Additional Tennessee Eastman process simulation data for anomaly detection evaluation," Harvard Dataverse, V1, 2017, doi: 10.7910/DVN/6C3JR1.
- **[D7]** J. J. Downs and E. F. Vogel, "A plant-wide industrial process control problem," Comput. Chem. Eng., vol. 17, no. 3, pp. 245–255, Mar. 1993, doi: 10.1016/0098-1354(93)80018-I.
- **[D8]** K. Hundman, V. Constantinou, C. Laporte, I. Colwell, and T. Söderström, "Detecting spacecraft anomalies using LSTMs and nonparametric dynamic thresholding," in Proc. 24th ACM SIGKDD (KDD), London, Aug. 2018, pp. 387–395, doi: 10.1145/3219819.3219845.

---

## II-4. 두 Condition (Q1 vs Q3)의 의미

**[검증된 사실 후보]**
- **Q1 (minmax full)**: Min-max scaling (train fit, no clip — paper-faithful sklearn MinMaxScaler 동작), train 데이터 안에 anomaly 포함. 실제 운영 환경 가정.
- **Q3 (minmax normalonly)**: Min-max scaling (train fit, no clip), train에서 anomaly region 제거 → segment-aware concat. 이상 미지수 환경 가정.
- 실험 순서: Q3 → Q1.
- no clip 정책: 2026-05-25부터 `np.clip(0,1)` 강제 제거 → paper-faithful.

**[검증된 사실 후보 — Per-entity 정규화 (2026-06-01 수정)]**
Multi-entity concat (SMD 28 / SMAP 54 / MSL 27 / Exathlon 6) 데이터셋에서 whole-array 단일 scaler → entity별 독립 minmax (각 entity train slice에 fit → 자기 test slice transform, leak-free). 예외 (Page B 원문 전체): (a) **진짜 단일 entity 변형 — PSM/SWaT/WaDi/simulation/`*_simple`/단일 machine** → per-entity ≡ whole-array NO-OP(bit-identical); (b) 6 untouchable SOTA(timesnet/tfmae/memto/moderntcn/dcdetector/catch) → raw `none` branch + 내부 whole-array self-norm(upstream-faithful) 유지.

**[검증된 사실 후보]** Q2/Q4(zscore) 폐기 — Page B Snapshot("Q2/Q4 (zscore) 폐기") + §2("Q2/Q4 (zscore 변형) 는 폐기되었음") 명시. *(r2: II-1과 등급 통일 — 사실 후보.)*

**[Notion의 주장]** SWaT는 region 22 제외(`excl_region22`) 결과를 메인으로 사용.

---

## II-5. 평가 지표

**[검증된 사실 후보]**
- **PA%K AUC F1** (`pak_auc_f1`): best-epoch 기준. Kim et al. AAAI 2022의 per-K threshold re-optimization 방식.
- **PRC-AUC** (`prc_auc`): 보조 메트릭.
- **F1_T** (`f1_t`): TimeSeAD / QuoVadisTAD 방식의 time-series F1.
- Rank Avg: 6 dataset (SWaT excl22, WaDi A1, WaDi A2, SMD, PSM, Exathlon — simulation 제외) 등수 평균 (낮을수록 좋음).

---

## II-6. 실험 결과 요약 (Q3, 2026-05-22 기준 partial)

> **(r2) 수치 유효성 한정 — 아래 표 전체에 적용**:
> 1. **§3 status 기준 시점 = 2026-05-22** (Page B §3 status callout) — 표의 수치는 이 시점에 채워진 값이다.
> 2. **2026-05-25 paper-faithful 재실행으로 15개 모델 행 무효화 (swap-in 대기)** — QuoVadis 9종(Random/SensorRange/PCA/L2/NN/MLP/MLPmixer/Transformer/GCN-LSTM) line-by-line 수정(pca_error aggregation fix, mlp/mlpmixer/transformer reimpl, neural_base predict Pass 2 fix 등) + non-self_norm SOTA 6종(gcn_lstm/tranad/usad/dagmm/gdn/omnianomaly) clip 제거. Page B 원문: "결과 row 는 재실험 완료 후 swap-in 예정. **영향 모델: 15 종** (9 QuoVadis + 6 non-self_norm SOTA, gcn_lstm 중복)". 영향 entries(~70 dirs)는 백업 후 삭제. **5번 실험 폐기 → 6번 실험(`6_20260526_085028_..._segaware`) 재실행**.
> 3. **2026-06-04 faithfulness pass v2로 12개 모델 추가 수정** (tranad/gdn/timesnet/memto/npsr/random/wetas/deepmil/treemil/nrdetector/catch/gcn_lstm — §II-2b 참조).
> 4. per-entity 정규화(2026-06-01) 이전 산출이므로 **SMD/Exathlon column은 STALE**.
> → 즉 **simple/neural/legacy 행 다수가 fidelity-audit 이전 코드의 수치**다. Phase 3는 이 표의 수치를 결과 수치로 사용하지 말 것 — clean re-run 후 swap-in 값만 사용.

**[검증된 사실 후보 — 수치 (2026-05-22 시점 스냅샷; 위 r2 한정 블록 적용)]**

### PA%K AUC F1 (Q3, partial, MAE+legacy 완료 / SOTA new 7개 pending)

| Model | sim | SWaT(excl22) | WaDi A1 | WaDi A2 | SMD | PSM | Exathlon | Rank Avg |
|-------|-----|-------------|---------|---------|-----|-----|----------|----------|
| MAE 271 | 0.607 | 0.630 | 0.850 | 0.794 | 0.732 | 0.803 | 0.645 | 2.00 |
| MAE 271 B2 | 0.635 | 0.640 | 0.878 | 0.811 | 0.752 | 0.804 | 0.656 | **1.00** |
| random | 0.331 | 0.181 | 0.189 | 0.191 | 0.218 | 0.652 | 0.426 | 15.00 |
| sensor_range | 0.734 | 0.076 | 0.490 | 0.483 | 0.279 | 0.476 | 0.394 | 14.67 |
| pca_error | 0.875 | 0.265 | 0.492 | 0.467 | 0.564 | 0.752 | 0.489 | 10.67 |
| l2_norm | 0.539 | 0.081 | 0.252 | 0.247 | 0.449 | 0.572 | 0.494 | 14.83 |
| nn_distance | 0.877 | 0.434 | 0.528 | 0.544 | 0.559 | 0.743 | 0.498 | 9.17 |
| mlp | 0.800 | 0.462 | 0.586 | 0.554 | 0.588 | 0.755 | 0.482 | 7.83 |
| mlpmixer | 0.828 | 0.414 | 0.611 | 0.606 | 0.600 | 0.732 | 0.517 | 7.50 |
| transformer | 0.825 | 0.424 | 0.629 | 0.617 | 0.585 | 0.707 | 0.500 | 8.50 |
| gcn_lstm | 0.582 | 0.143 | 0.113 | 0.129 | 0.531 | 0.647 | 0.511 | 14.17 |
| anomaly_transformer | 0.744 | 0.502 | 0.692 | 0.687 | 0.606 | 0.734 | 0.506 | 5.33 |
| tranad | 0.861 | 0.458 | 0.651 | 0.663 | 0.659 | 0.728 | 0.572 | 5.00 |
| usad | 0.851 | 0.447 | 0.346 | 0.346 | 0.628 | 0.587 | 0.552 | 9.50 |
| dagmm | 0.756 | 0.179 | 0.475 | 0.467 | 0.580 | 0.553 | 0.519 | 11.67 |
| gdn | 0.821 | 0.414 | 0.449 | 0.418 | 0.593 | 0.737 | 0.520 | 9.33 |
| omnianomaly | 0.901 | 0.483 | 0.524 | 0.519 | 0.598 | 0.787 | 0.519 | 6.33 |
| tfmae, npsr, timesnet, dcdetector, memto, moderntcn, catch | — | — | — | — | — | — | — | — |
| deepmil, wetas, treemil, nrdetector | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

> **주의**: 위 수치 중 SMD(28 avg) 및 Exathlon(6 avg) column은 per-entity 정규화(2026-06-01) 이전에 산출된 STALE 값이다. SWaT excl22 / WaDi A1·A2 / PSM는 단일-entity이므로 정규화 변경 영향은 없으나, **simple/neural/legacy 행은 column 무관하게 2026-05-25/06-04 fidelity 수정 이전 산출 — clean re-run 후 swap-in 예정** (상단 r2 한정 블록).

**[Notion의 주장]** "모든 metric에서 MAE 271 B2가 RankAvg 1위" (pak_auc_f1: 1.00, prc_auc: 1.00, f1_t: 1.50). MAE 271 단독도 2위.

MAE 271 B2 = MAE 271 anomaly score에 σ=10 two-sided Gaussian smoothing (`scipy.ndimage.gaussian_filter1d(s, sigma=10, mode='reflect')`) post-hoc 적용. 학습 graph 영향 없음 — re-train 불필요.

### 정성적 경향 (Q3 partial 결과 기준)

**[Notion의 주장]** 상위 성능 모델 집단 (Rank Avg 기준): MAE 271 B2 (1.00) > MAE 271 (2.00) > tranad (5.00) > anomaly_transformer (5.33) > omnianomaly (6.33). 하위권에는 gcn_lstm, random, sensor_range, l2_norm이 위치. SOTA new 7개 및 weakly-supervised 4개의 Q3 결과는 아직 없음.

---

## III. Notion이 정리한 Contribution (원문 발췌)

> **Phase 3 판단 사안**: 아래 C1~C4 구조를 논문 contribution으로 그대로 가져갈지, 재구성할지는 Phase 3에서 독립 판단 필요.

**[Notion의 주장 — C1 원문]** "Context-Aware Time-Series Representation via Masking: 입력 시계열의 15% 패치를 마스킹하고, 나머지 85% 가시 패치만으로 마스킹된 영역을 양방향(bidirectional) 맥락 기반으로 복원하도록 학습한다. 정상 분포에 대해 학습된 모델은 이상 구간을 정확히 복원하지 못하므로 복원 오차가 1차 anomaly evidence로 작동한다. 핵심: 일반적인 시계열 이상탐지에서 흔히 쓰이는 '한 점만 예측' 혹은 '다음 시점 예측' 패러다임과 달리, MAE는 context-aware bidirectional reconstruction을 강제하여 더 풍부한 정상 분포 표현을 학습한다."

**[Notion의 주장 — C2 원문]** "Capacity-Gap Self-Distillation (비대칭 Teacher–Student): 동일한 인코더 출력을 깊은 Teacher 디코더(3층)와 얕은 Student 디코더(2층)에 분기시킨다. Student 디코더는 인코더에 대해 gradient detach되어, 인코더는 오직 Teacher의 정확한 복원 목적에 의해서만 학습된다. 두 디코더가 같은 latent에서 출발하지만 capacity gap으로 표현력이 달라지고, 이 gap이 이상 구간에서 더 크게 벌어진다. 외부 teacher 없이 자기 자신만으로 distillation을 구성하는 self-distillation 디자인."

**[Notion의 주장 — C3 원문]** "Discrepancy as Primary Anomaly Signal: Teacher–Student 차이를 Output Discrepancy(OD)와 Feature Matching(FM, hidden-level, L2 + adaptive λ)의 두 단계에서 측정. Output-level OD와 Hidden-level FM의 두 신호가 상보적으로 작동한다. FM adaptive λ가 두 loss 간 자동 스케일링을 보장하여 hidden-level 신호의 과도/과소 weighting 문제를 동적으로 해소. 추론 시 adaptive normalization으로 균형 잡힌 anomaly score를 산출한다."

**[Notion의 주장 — C4 원문]** "Semi-Supervised Anomaly Awareness via Gradient Reversal: 학습 셋에 소량 존재하는 anomaly 라벨을 적극 활용하는 디자인. Student hidden 위에 anomaly classifier를 얹고, GRL(Ganin et al. 2016)로 classifier의 gradient를 반전시켜 student hidden에서 anomaly 정보를 적극적으로 제거한다. 결과적으로 Student는 anomaly를 '진짜 몰라서' 재구성에 실패 → 큰 discrepancy 발생 → C3의 anomaly score가 강화되는 two-stage 메커니즘."

**[Notion의 주장 — Top 3 디자인 결정]**
1. GRL = 본 모델의 차별점 (타 MAE 기반 anomaly detector와 달리 semi-supervised)
2. 비대칭 디코더 + detach + FM adaptive λ (3:2 moderate gap, instability 회피)
3. Adaptive scoring + per-K threshold re-opt

---

## IV. 코드/실측과 대조가 필요한 의심 지점

1. **FM 제외 시점 (2026-06-01)**: Notion 페이지는 "FM은 2026-06-01 이후 anomaly score 계산에 사용하지 않음"이라 명시. 코드베이스(`mae_anomaly/scoring.py`, `evaluator.py`)에서 이 변경이 실제로 반영되어 있는지 확인 필요.

2. **Q3 결과 STALE 여부**: per-entity 정규화 변경(2026-06-01) 이후 SMD/Exathlon 결과가 STALE이라고 Notion이 명시. 최신 clean re-run 결과가 어느 파일에 저장되어 있는지 확인 필요.

3. **MAE 271 B2 post-hoc smoothing이 논문에서 정당한지**: `sigma=10` Gaussian smoothing을 post-hoc으로 적용하는 것이 논문 기여로 주장될 수 있는지, 아니면 단순 engineering trick인지 판단 필요 (Phase 3).

4. **SOTA new 7개 Q3 결과 미완**: tfmae/npsr/timesnet/dcdetector/memto/moderntcn/catch의 Q3 결과가 아직 없음. 이 수치 없이는 fair comparison table이 불완전.

5. **Weakly-supervised 4개 Q1 결과 미완**: deepmil/wetas/treemil/nrdetector의 GPU 실험 미실행. 논문에 포함시킬지, 그리고 어떤 조건으로 포함시킬지 결정 필요.

6. **dagmm 구현 provenance**: Notion에서 dagmm 구현을 TranAD repo의 reimplementation으로 교체했음을 명시 (구현 출처 reference = [B5b], r2 추가). GMM energy loss가 제거된 simplified variant. *(r2 정정 — 열린 질문이 아니라 Page B가 이미 **결정**해 둔 사안)*: Page B 원문 "조치 = **scoreboard에서 `dagmm_tranad`로 relabel** · energy-DAGMM paper target과 직접 비교 금지" (판정 "RELABEL only"). 논문 표기 시 이 결정(dagmm_tranad relabel + 직접 비교 금지)을 따를지 Phase 3에서 확인.

7. **GRL classifier loss에서 pos_weight ≈ 7.29**: "데이터셋의 anomaly ratio 추정치 기반 자동 계산"이라고 하는데, 각 데이터셋별 실제 pos_weight 값이 코드에서 어떻게 계산되는지 확인 필요.

8. **tranad lr=0.01 vs lr=1e-4**: Notion에서 "paper-text 0.01은 run-code 값 아님"이라며 constants.py의 lr=1e-4를 사용한다고 명시. 어느 값이 더 faithful한지 논문 작성 시 명확히 해야 함.

9. **deepmil encoder 출처**: WETAS DiCNN을 encoder로 사용 (Sultani CVPR 2018 원논문은 frozen C3D feature 기반). Notion은 "DERIVATIVE_CITED"라 표시하나, 이를 논문에서 어떻게 서술할지 판단 필요.

10. **SWaT excl_region22의 정당성**: region 22가 test의 ~16%를 차지한다고 하는데, 이 제외가 어떤 근거로 standard practice인지 명확한 인용이 필요할 수 있음.

11. **(r2 추가) WaDi A2 feature 수 — 원천 간 모순 (Page 0=127 vs Page B=123, 검증값 123)**: Page 0은 **4개소**(§1.2 지원 데이터셋 표·num_features 표·d_model 매핑·§5.2.1)에서 127 (r3 정정 — 초판 "3개소"는 §1.2 지원 데이터셋 표 누락; 본 digest I-7이 전사한 표가 바로 그 표라 자기 본문과도 불일치했음. Page 0 덤프 디코딩 후 "127" 전수 재카운트로 4개소 확정), Page B §2.1 표는 123 (Page B 전문에 "127" 0회). 발표 PDF p19도 123 dim. **코드 검증 완료 — 123 확정** (`p1_reconciliation_r1.md` §III: exp271 metadata `config.num_features=123`; raw CSV 실측 124 cols = 123+label; 127의 정체 = 원본 `WADI_attackdataLABLE.csv` sensor 127개 중 all-NaN 4개 컬럼을 `prepare_raw_datasets.py` `handle_nan`이 drop하기 **이전** 수치). Page 0의 127 기재는 drop 전 원본 수 — Notion Page 0 측 수정 필요 사항으로 기록.

12. **(r2 추가) Page 0 내부 모순 — 스코어링 비율 서술 (4:1 vs 1:1 stale text)**: §2.4 Adaptive Scoring(2026-06-01 이후 기준 서술)은 "recon:disc = **4:1** + FM 점수 제외"를 수식과 함께 명시하나, §3.6 evaluator("recon 평균을 anchor로 disc/FM을 정규화 후 **1:1 결합**"), §4.3.3 anomaly_score_mode 역할란("자동 normalize 후 1:1 결합"), §5.3 Top 3 callout("1:1 결합")에는 **구버전(1:1 + FM 포함) 서술이 잔존**. 본 digest는 §2.4의 4:1을 채택(올바른 선택 — 2026-06-01 명시 변경과 정합)했으나, 원천 내부 모순 자체를 기록해 둔다. 코드(`mae_anomaly/scoring.py`) 기준 최종 확정은 IV-1과 동일 라인에서 처리.

---

## V. REQUEST / FEEDBACK 블록

이 Notion 페이지들에서 별도의 REQUEST: 또는 FEEDBACK: 블록은 발견되지 않았다.

---

## 정정 이력

**r2 (2026-06-10, fixer-3 — `p1_digests_r1.md` 전수 반영, 상세: `p1_digests_fixlog_r2.md`)**
- **NB-1 (BLOCKER)**: II-3 truth 표 WaDi A2 Features 127→**123** (Page B 원문 + 코드 검증). Page 0의 127 기재(I-3/I-7/I-9)는 원문 전사 유지 + 모순 주석. §IV-11 신설.
- **NB-2 (BLOCKER)**: II-1 cells 산식에 누락된 "**× 22 active models**" 인자 복원 (원문 정확 전사) + Page B 내부 39/41 불일치 주석.
- **NM-1**: I-10 truth 라벨을 R26 범위([4],[6]–[10],[12])로 한정 — 방법론 인용 5건([1][2][3][5][11])은 [Notion의 주장 — 검증 완료 주장]으로 강등.
- **NM-2**: II-6에 수치 유효성 한정 블록 신설 (2026-05-22 시점 + 2026-05-25 재실행 15종 무효화/swap-in 대기 + 5번 폐기→6번 재실행 + 2026-06-04 v2 12종).
- **NM-3**: II-2b 신설 — §1.2 HP preset 전체, weak label 정의(leak-free), provenance 태그 4종, 2026-06-04 faithfulness pass v2, 2026-06-02 boundary-safe TEST windowing.
- **NM-4**: I-1 마스킹 8개 고정(round), I-3 priority 공식·디코더 구조(self-attn only), I-4 teacher_only 플래그 메커니즘·bf16·eval_interval·random_seed·config validation 5종, I-5 GRL adaptive λ anchor 추가.
- **Nm-1**: §IV-12 신설 (Page 0 내부 4:1 vs 1:1 stale text 모순).
- **Nm-2**: II-1에 "weakly-supervised 5종"(Snapshot 원문) vs 4종(제목/§1.1/§6.4) 불일치 주석.
- **Nm-3**: Q2/Q4 폐기 등급을 [검증된 사실 후보]로 통일 (II-4).
- **Nm-4**: II-4 per-entity 예외 목록 전체 복원 (simulation/`*_simple`/단일 machine 추가).
- **Nm-5**: II-3에 TEP 행 + 보유·검증 완료/비교 미사용/CC0 한 줄 추가.
- **Nm-6**: [B4b]/[B5b] 구현 출처 reference 추가, IV-6을 Page B의 기결정(dagmm_tranad relabel)으로 정정.

---

*Digest 작성: notion-analyst, 2026-06-10. r2 정정: fixer-3, 2026-06-10.*
*소스: Page 0 MAE (75,820 chars), Page B Baseline Comparison (108,461 chars), 완전 정독 완료. r2 정정 시 원천 덤프 python 슬라이스 재확인 + 코드 검증은 `p1_reconciliation_r1.md` 인용.*
