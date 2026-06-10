---
phase: 1
agent: synthesis-writer
directives: [T1, R10, R11, R25]
last_modified: 2026-06-10
revision: r2 (fixer-2 — adversarial review paper/99_reviews/p1_codebase_synthesis_r1.md 전수 반영; fixlog: p1_codebase_synthesis_fixlog_r2.md; 정정 이력은 말미 부록)
authority: |
  정본 우선순위: 271_CONFIG_TRUTH.md > 본 문서 > CODEBASE_UNDERSTANDING.md / EXPERIMENT_PROTOCOL_TRUTH.md / NOTION_DIGEST.md / CONFERENCE_PDF_DIGEST.md.
  본 문서의 수치·설정이 하위 문서와 충돌하면 271_CONFIG_TRUTH.md §II–VIII을 1차 소스로 삼아 재확인할 것.
---

# RESEARCH SYNTHESIS — TSMAE Phase 1 종합 문서

> **이 문서의 용도**: Phase 2 이후 모든 에이전트가 "연구의 진실"로 참조하는 단일 종합 문서다.
> 입력 5개 문서의 검증된 사실만을 종합했다. 아직 검증되지 않은 사항은 INFERENCE/UNKNOWN/RISK로 명기한다.
> 발명된 수식·명칭·주장은 일절 없다.

---

## ① 연구 한 문단 요약

본 연구는 **다변량 시계열(multivariate time series) 이상 탐지**를 위한 준지도(semi-supervised) 학습 프레임워크를 제안한다. 핵심 문제 설정(가정, R11)은 "학습 데이터의 대부분이 정상·이상 여부를 모르는 unlabeled 상태이지만, 그 중 일부 anomaly에는 실제 라벨이 붙어 있는 준지도 환경"이며, 기존 비지도 방법이 이 소량의 anomaly 라벨을 전혀 활용하지 못한다는 점을 주된 동기로 삼는다 (설정/구현 구분의 상세는 §② — main 실험 구현은 이 설정의 label 가용성 상한 케이스다). 제안 방법(`SelfDistilledMAEMultivariate`)은 Masked Autoencoder(MAE) 원리 위에 비대칭 Teacher–Student 자기증류(self-distillation)를 결합하고, Gradient Reversal Layer(GRL)를 통해 소량의 anomaly 라벨을 적극적으로 학습에 반영한다: 깊은 Teacher 디코더(3층)가 masked 패치를 복원하는 동안 얕은 Student 디코더(2층)는 알려진 anomaly 정보를 표현하지 못하도록 GRL이 억제함으로써, 정상 구간에서는 Teacher–Student 불일치(discrepancy)가 작고 anomaly 구간에서는 커지는 신호 구조를 형성한다. 추론 점수는 Teacher 복원 오차(recon)와 Teacher–Student 출력 불일치(scaled_disc/4)의 가중 합산으로 구성된다. 평가 대상은 SWaT·WaDi A1/A2·PSM·SMD(28)·SMAP(54)·MSL(27) 6계열이며 22개 비지도 baseline과 비교한다 — 단, 정정(fixer-2, MAJ-007): **현재 완료된 MAE 271 entity는 37개**(SWaT 2 = full/excl22 dual-eval 집계 · WaDi 2 · PSM 1 · **SMD 22/28 · SMAP 5/54 · MSL 5/27**; `results/experiments/271_20260602_020545_271canon_baseline/` 실측)로 SMD/SMAP/MSL 잔여 entity는 **실행 진행 중**이고, 논문 본문은 placeholder 정책(A8/R3)에 따라 실험 완료를 가정하고 수치 자리는 placeholder로 비워 작성한다. 핵심 지표로 PA%K-AUC F1, VUS-PR, VUS-ROC, Affiliation F1, PA%K-AUC PR을 사용한다.

---

## ② 문제 설정의 정의 (R11)

> 정정(fixer-2, BLK-005): 초판의 "오염된 unlabeled 다수" 서술은 바로 아래 "모든 학습 샘플에 라벨 존재" FACT와 모순되었다.
> 모순 제거를 위해 **설정(가정) / main 실험 구현(FACT) / 라벨 희소화 sweep(계획)**의 3단 구조로 재서술한다.

### ②-1. 설정(가정) — R11이 정의하는 문제 환경

**Directive 원문 (R11)**: "훈련 데이터셋이 대부분 이상인지 아닌지 모르는 unlabeled 상태지만, 그 중 일부는 실제 고장상황 발생 등으로 이상 label이 되어 있는 상황을 가정하고 있으며, 기존 unsupervised learning 기반의 이상탐지는 대량의 unlabel 데이터로부터 전체 데이터 분포는 학습할 수 있지만, 소수 존재하지만 매우 중요한 labeled 데이터를 활용하지 못하는 것이 매우 핵심임."

즉 **연구가 가정하는 환경**은: 대부분 unlabeled(이상 여부 미상) + 소수 labeled anomaly. 이것은 문제 설정(assumption)이지, 아래 ②-2의 구현 그 자체가 아니다.

### ②-2. main 실험 구현 (FACT) — 설정의 label 가용성 상한 케이스

**FACT**: main 실험(R13)의 훈련 데이터 구성은 다음과 같다.
- 원본 train 파일 전체 + **원본 test 파일의 앞 50%를 train에 편입** (시간순 분할; 분할 코드 근거는 §④).
- 구현상 **train 구간의 모든 샘플에 라벨이 존재**하며(정상=0, 이상=1), 편입된 test 앞 50% 안의 실제 anomaly에도 전부 라벨이 제공된다. 실측 train anomaly 비율: SWaT 1.63%, WaDi A1 0.52%, WaDi A2 0.76%, PSM 6.20%, SMAP concat 0.70%, MSL concat 1.70%. (근거: EXPERIMENT_PROTOCOL_TRUTH §① + [271c] metadata `anomaly_ratio` 실측.)

**프레이밍 (FACT 기반)**: 따라서 main 실험은 R11 설정에서 "일부 labeled anomaly"의 비율을 **가용한 최대치(train 구간 내 anomaly 전부 labeled)로 둔 상한(upper-bound) 케이스**다. 라벨이 학습에 개입하는 지점은 아래 3곳(②-3)에 국한되며, 그 외에는 라벨이 학습에 쓰이지 않는다.

### ②-3. 라벨 희소화 sweep (계획) — 설정의 일반 케이스 검증 (R32)

**Directive (R32)**: 라벨 희소화 sweep 실험 포함 + unlabeled anomaly 혼입 시 강건한 이유의 논리적 설명.

label 제공 비율을 낮추면 일부 anomaly가 **unlabeled 상태로 train에 잔류**한다 — 이것이 R11 가정(대부분 unlabeled + 소수 labeled)의 일반 케이스이며, sweep은 이 일반 케이스에서의 강건성을 검증하는 실험이다. 현재 전용 실험은 미실행(수치 없음, §④ 참조); 재사용 가능한 코드 메커니즘은 존재한다(`NoisyLabelSlidingWindowDataset` 등, §④). 논문에서는 placeholder 정책(A8/R3) 적용.

### ②-4. 코드에서 label이 실제로 쓰이는 지점들

아래 세 지점이 전부다. 이 외의 곳에서 라벨은 학습에 개입하지 않는다.

1. **`force_mask_anomaly=True`** (config, `model.py:975–1002`): masking 예산(round(50×0.15)=8 패치) 안에서 anomaly 패치를 우선순위로 마스킹한다. 모델이 anomaly 위치의 복원을 회피하지 못하도록 강제.
2. **출력 불일치 손실의 방향 분기** (`loss.py:244–261`): `patch_has_anomaly` / `patch_is_normal`로 패치를 구분해 정상 패치는 Teacher–Student discrepancy를 줄이는 방향으로, anomaly 패치는 GRL이 담당하므로 손실 계산에서 0으로 설정(`grl_disable_anomaly_loss=True` → `loss.py:259–261`에서 `anomaly_loss = torch.tensor(0.0, …)`).
3. **GRL 분류기 타겟** (`config.py:131`, `loss.py:282–350`): `grl_target_mode='window'` — 윈도우 안에 anomaly가 1개라도 있으면 마스킹된 모든 패치의 타겟=1. Student decoder **마지막 층 hidden 전체**(`(num_patches, batch, d_model)`)에 AnomalyClassifierHead가 **패치별 독립 적용**된다 — 풀링 없음; `model.py:1153–1154`에서 `squeeze(-1).transpose(0,1)` → `(batch, num_patches)` 로짓 (fixer-2, MIN-003). 손실은 **masked(valid) 패치에만** 계산된다(`valid = patch_has_masked`, `loss.py:283–284`). 이 타겟으로 focal-style BCE(표준 focal loss 아님 — 표A GRL 행 참조)를 계산하고, GRL이 gradient를 반전(-lambda × grad)시켜 student decoder가 anomaly 정보를 표현하지 못하도록 억제한다.

**추론 시**: 라벨은 일절 사용하지 않는다. test 라벨은 평가용 ground truth로만 보유.

### ②-5. 왜 기존 비지도 방법은 이 3지점을 구현할 수 없는가 (R11의 핵심 — fixer-2, MAJ-006 보강)

기존 unsupervised 방법(예: 본 비교군의 비지도 baseline들)은 **학습 시 라벨 입력 자체를 받지 않는다** — 본 프로젝트의 비교 조건 정의가 이를 그대로 반영한다: Q1은 라벨 미사용 학습, Q3에서도 라벨은 train 데이터에서 anomaly 구간을 절제하는 **데이터 정제 용도**로만 쓰이고 모델 학습 신호로는 들어가지 않는다 (§④ 비교군 label 정책). 따라서 위 3지점은 비지도 방법에서 구조적으로 정의 불가능하다:

1. **masking 우선순위** (`force_mask_anomaly`): anomaly 패치를 우선 마스킹하려면 패치별 anomaly 라벨(`point_labels` → patch label)이 입력으로 필요하다. 라벨 없는 학습에서는 이 우선순위 함수 자체가 정의되지 않는다.
2. **손실 방향 분기** (`patch_has_anomaly`/`patch_is_normal`): 정상/이상 패치를 손실 수준에서 다르게 취급하는 분기 조건이 라벨의 함수다. 비지도 손실은 모든 train 샘플을 동일하게(통상 "전부 정상" 가정으로) 취급할 수밖에 없다.
3. **GRL 타겟**: 분류기 타겟(window/patch anomaly flag)이 라벨 그 자체다. 라벨 없이는 adversarial suppression의 supervision 신호가 존재하지 않는다.

이것이 R11의 "기존 unsupervised는 소수의 핵심 labeled 데이터를 활용하지 못한다"의 코드-직결 논리다. 반대로 비지도 방법에 라벨을 줄 수 있는 최선은 Q3처럼 "오염원 제거"뿐이며(R12 논리, §④), 제안 방법은 동일 라벨을 **학습 신호(masking 우선순위 + 손실 분기 + negative supervision)**로 사용한다는 점이 차별점이다.

### ②-6. PU Learning과의 관계 (FACT + INFERENCE 혼재)

**FACT**: 발표자료 p7에서 Semi-supervised Learning과 Positive-Unlabeled Learning을 나란히 도식으로 비교했다. **②-1의 설정(가정)**은 "labeled anomaly(positive) 소수 + 대량 unlabeled" 구조로, PU learning과 유사한 분포를 가진다.

**INFERENCE**: 그러나 발표 본문에서 "PU learning"이라는 용어로 방법론을 정식화하지는 않았다 (CONFERENCE_PDF_DIGEST §② 명시). 또한 **main 실험 구현(②-2)은 train 구간 라벨이 전부 존재하는 상한 케이스**이므로 엄밀한 PU setting(positive + unlabeled만 존재)이 아니다 — PU-likeness가 실제로 성립하는 것은 라벨 희소화 sweep(②-3)의 일반 케이스 쪽이다. main 구현은 더 정확하게는 "contaminated semi-supervised" — 정상·이상 혼재 데이터에서 소수 anomaly 라벨을 약한 감독 신호로 활용하는 weakly-supervised 변형에 가깝다. (CODEBASE_UNDERSTANDING §4.3: "The use of anomaly labels during training to guide masking and loss direction is closer to weakly supervised than standard PU learning.")

**Phase 3 판단 사안**: 논문에서 본 설정을 "semi-supervised"로 표현할지, "PU learning의 변형"으로 표현할지, "contaminated semi-supervised"로 표현할지는 Phase 3에서 결정한다.

---

## ③ 방법론 Component 분해 + R10 원재료

**주의**: 아래 표의 모든 설정값은 **exp271 실측치** (271_CONFIG_TRUTH.md §II 전수표 기준, 전 37 entity metadata 동일). code default와 다른 경우 병기.

> "왜 다변량 시계열에서 이래야만 하는가" 논리 후보는 코드 설계 사실로부터 도출한 원재료다. 논리 강도가 낮은 항목은 "논리 보강 필요 — Phase 3"로 표기했다.

### 표 A: 활성 Component (271 canonical)

| Component | 271 설정값 | 기능적 역할 | 왜 다변량 시계열에서 이래야만 하는가 (R10 원재료) |
|-----------|-----------|------------|--------------------------------------------------|
| **Linear Patchify** | `patchify_mode='linear'`, `patch_size=10`, `num_patches=50`; `patch_embed: Linear(patch_size × F → d_model)` | 한 패치(10 타임스텝 × F 피처)를 flatten → linear projection으로 d_model=512 토큰화 | 다변량에서 한 패치 내 모든 피처의 시간축 + 채널 간 관계를 단일 벡터로 포착. CNN 없이 flatten이므로 피처 간 선형 결합이 직접 임베딩에 반영된다. 단, 비선형 채널 간 상호작용을 1층 선형으로만 포착한다는 한계가 있다 — 논리 보강 필요 (patch_cnn 비교 ablation이 있으면 근거 보강 가능). |
| **Patch masking (15%, anomaly-first)** | `masking_ratio=0.15`, `force_mask_anomaly=True`, `mask_after_encoder=True`; 8 패치 마스킹/42 패치 가시 (`model.py:986`) | 가시 패치만 encoder에 입력하고, mask token은 decoder 직전에 삽입 (MAE 원형). anomaly 패치가 마스킹 예산에서 우선 선택됨 | 다변량 시계열에서 masking은 특정 시간 구간의 피처 전체가 숨겨지는 맥락 복원 문제가 됨. 정상 패턴에서 변수 간 상관 구조(예: 센서 A와 B의 동기)를 학습해야 복원이 가능 — anomaly는 이 상관 구조에서 이탈하므로 복원 오차가 커진다. `force_mask_anomaly`는 모델이 anomaly 위치 복원을 회피하는 것을 방지함 (class imbalance 문제의 직접 대응). |
| **Transformer Encoder (4층, Pre-Norm, GELU)** | `num_encoder_layers=4`, `d_model=512`, `nhead=8`, `dim_feedforward=2048`, `dropout=0.15`; self-attention only | 가시 패치들의 전역 맥락 표현 학습. encoder는 teacher path gradient만으로 학습(student는 `latent_visible.detach()`) | 다변량 시계열의 패치 간 장거리 의존성(예: 센서 이상이 다른 센서에 지연 전파되는 패턴)을 self-attention으로 포착. positional encoding이 원래 위치를 보존하므로 masked 위치와 가시 위치의 공간 관계가 유지됨. Pre-Norm은 긴 시계열 학습의 안정성에 기여 (논리 보강 필요 — 시계열 도메인 고유 정당화 부족). |
| **비대칭 Teacher decoder (3층) / Student decoder (2층)** | `num_teacher_decoder_layers=3`, `num_student_decoder_layers=2`; 둘 다 self-attention only (`use_transformer_encoder_decoder=True`); 별도 mask token (`shared_mask_token=False`) | Teacher: 정확한 복원 기준 제공. Student: teacher보다 얕아 anomaly 복원 실패 확률이 높음 → discrepancy 신호 형성. Encoder gradient는 student로부터 차단(detach) | 다변량 시계열에서 teacher가 학습한 "정상 변수 간 상관 구조"를 student가 낮은 capacity로 모방하려 할 때, anomaly로 인한 비정상 상관 패턴에서 모방이 더 크게 실패함. 즉 capacity gap이 다변량 구조적 이탈 탐지로 자연스럽게 이어진다는 설계 논리. 단 teacher 3층 vs student 2층의 gap 크기 선정 근거가 코드에만 있고 이론적 정당화가 부족 — 논리 보강 필요 + ablation(층 수 조합) 필요. |
| **Teacher-only warmup 250 epochs** | `teacher_only_warmup_epochs=250`, `num_epochs=500`; warmup 중 student decoder, GRL, FM 전부 비활성; 이후 anomaly loss ramp max(250//5,2)=50 epochs | Teacher가 먼저 정상 복원 기준(stable reference)을 충분히 학습한 후 student가 합류하는 단계적 학습 | 다변량 시계열에서 teacher가 충분히 수렴하지 않은 상태에서 student가 합류하면, student가 모방할 "정상 기준"이 불안정해 discrepancy 신호가 noise가 됨. 길고 안정적인 warmup이 teacher 기준의 품질을 보장. 발표 p24 학습 곡선이 warmup 후 pak_auc_f1 +0.1 내외 상승을 보이며 설계 효과를 정성적으로 지지함. ⚠️ **CRITICAL RISK (fixer-2, NOTE-001 격상)**: warmup 효과를 입증하는 정식 ablation(예: warmup 0/50/250 비교)이 **존재하지 않는다**. 학습 곡선의 정성적 상승만으로는 reviewer 방어 불가 — 이 component의 motivation을 논문에서 주장하려면 Phase 2에서 해당 ablation 실험이 **필수** (§⑨ REQUEST-F 등재). |
| **Output Discrepancy Loss (정상 패치 측만 활성)** | `use_output_discrepancy=True`, `grl_disable_anomaly_loss=True` → 정상 masked 패치에 대해서만 `||teacher_out.detach - student_out||²` 최소화; `normal_loss_weight=1.0` | 정상 패치에서 teacher–student 불일치를 줄이는 방향으로만 student를 학습 → 정상에서 낮은 discrepancy를 유도 | 비지도 방법은 정상 분포를 배우지만 discrepancy가 anomaly에서만 커지도록 명시적으로 유도하지 못함. 정상 패치 OD loss는 student를 "정상에서만 teacher를 잘 따르는" 상태로 유도해 이상 탐지 신호의 대비(contrast)를 높인다. 다변량에서 피처 전체에 걸친 불일치를 patch 단위로 집계하므로 국소 이상(단일 피처)뿐만 아니라 다중 피처 동시 이탈도 포착. |
| **GRL (Gradient Reversal Layer, student decoder 대상)** | `use_grl=True`, `grl_mode='classifier'`, `grl_disable_anomaly_loss=True`; `AnomalyClassifierHead` = **2-layer MLP** (Linear 2개; `grl_cls_arch='default'`+`grl_cls_hidden=0`→hidden=d_model//2=256 자동, `model.py:177–186` — 코드 주석의 "1-layer MLP"는 hidden-층 수 기준 표현이므로 논문에는 "2-layer MLP"로 표기, fixer-2 MAJ-004): LayerNorm(512) → Linear(512→256) → GELU → Dropout(0.1) → Linear(256→1); student decoder 마지막 층 hidden에 **패치별 독립 적용**(풀링 없음, `model.py:1153–1154`; 손실은 masked 패치만 `loss.py:283–284`, fixer-2 MIN-003); backward: `-lambda × grad`; `grl_target_mode='window'`; `grl_use_focal=True` → **focal-style BCE 변형** `(1−exp(−BCE))²×BCE`, γ=2, `p_t:=exp(−BCE)` — pos_weight 내장 BCE 기반이므로 **표준 focal loss(Lin et al. 2017) 아님, "standard focal loss" 표기 금지** (`loss.py:330–340`, fixer-2 BLK-004); `grl_loss_weight=0.2`; `grl_adaptive_lambda=True` → trainer inline `λ_GRL = clamp(‖∇L_main‖/(‖∇L_GRL‖+1e-4), 0, 10)`, 직전 epoch 값 적용 (`trainer.py:752–763`; FM의 λ_FM과 별개 값, fixer-2 BLK-001); classifier LR = main LR × 0.1 | student decoder에서 anomaly-identity 정보를 능동적으로 억제(suppression). student가 알려진 anomaly 패턴을 복원 단서로 쓰지 못하게 하여 anomaly 구간에서 discrepancy를 증폭 | 다변량 시계열에서 anomaly는 종종 특정 피처 조합 패턴으로 나타난다. GRL이 없으면 student는 학습 중 anomaly 패턴을 기억해 잘 복원할 수 있고, 그러면 anomaly에서 discrepancy가 커지지 않는다. GRL은 학습 중 labeled anomaly 정보를 "negative supervision"으로 사용하여 이 문제를 해결 — 이것이 준지도 환경에서 labeled anomaly를 비지도 방법 대비 추가로 활용하는 핵심 메커니즘. Encoder는 `latent_visible.detach()`로 GRL gradient로부터 완전히 차단되므로, encoder는 teacher path의 복원 목적에만 집중. |
| **Feature Matching Loss (FM, 훈련 전용)** | `use_feature_matching=True`, `fm_distance_metric='l2'`, `fm_adaptive_lambda=True`; 정상 masked 패치에 대해 `(1/d_model)||teacher_hidden.detach - student_hidden||²`; adaptive lambda: **trainer inline** `λ_FM = clip(||grad_main||/(||grad_fm||+1e-4), 0, 10)` (`trainer.py:639–653`; 적용은 직전 epoch 값 `trainer.py:652, 1301–1303`) — GRL의 λ_GRL과 **별개 값**이고, discriminator 전용 `compute_adaptive_lambda`(`loss.py:683`)와도 무관 (fixer-2, BLK-001) | hidden 표현 수준에서도 student가 정상 패치에 대해 teacher를 따르도록 유도. 학습 regularizer 역할. **추론 점수에는 포함하지 않음** (`scoring.py:237` `fm_active=False` hardcoded, 2026-06-01 이후) | 출력 수준 OD 외에 hidden 수준에서도 student를 teacher 표현 공간에 앵커링하는 이중 압력. 다변량 피처의 joint 표현이 hidden 공간에서 collapse되는 것을 방지하는 regularization 효과 (논리 보강 필요 — FM 제외 ablation 결과 없음). Adaptive λ는 FM gradient가 주 손실 대비 지나치게 커지는 경우를 자동 균형조정. |
| **추론 점수 (adaptive mode)** | `anomaly_score_mode='adaptive'`, `score_recon_disc_ratio=4.0`; 공식: `scaled_disc = disc × (mean_recon + 1e-4)/(mean_disc + 1e-4)`, `score = recon + scaled_disc / 4`; FM 제외 | recon과 disc의 스케일 차이를 자동 보정 후 4:1 비율로 합산. 추론 시 라벨 무사용. | 다변량 시계열에서 recon 오차와 discrepancy의 절대 스케일은 데이터셋·피처 수·anomaly 유형에 따라 크게 달라진다. Adaptive scaling이 없으면 한 성분이 점수를 지배해 다변량 설정 간 일반화가 떨어진다. 4:1 비율은 config로 조정 가능하나 271 실험에서 고정값. |
| **Patch → Point 집계 (leave-one-out, mean 집계)** | `eval_complementary_masking=False` → 윈도우당 **50개 leave-one-out 마스킹 패턴**(패치 p만 마스킹)을 **batch 차원으로 확장해 병렬 forward** (`evaluator.py:1650` docstring "single forward pass by expanding batch dimension", 구현 `1801–1818`; `patch_batch_size=2` 분할은 메모리 관리용 — 수치 무영향, `evaluator.py:1703–1717`) — 정정(fixer-2, BLK-002): "순서대로 1개씩 × 50회 순차 forward" 아님; 각 타임스텝은 덮는 (window, patch) 쌍들의 score를 평균 — 윈도우 내에서 타임스텝이 속한 패치는 정확히 1개이므로 타임스텝당 score 수 = 덮는 윈도우 수 ≈ 500/49 ≈ 10 (test stride=49) | 모든 위치에 균등한 맥락 기반 anomaly 점수 부여. 여러 window의 서로 다른 맥락으로 같은 시점을 반복 평가하는 "앙상블 효과" | 다변량 시계열에서 이상은 특정 시점·피처 조합으로 국소화되는 경우가 많다. 한 번의 forward로 모든 패치를 한꺼번에 평가하는 대신 leave-one-out으로 각 패치를 독립적으로 평가하면, 한 패치의 이상 여부가 다른 패치 점수에 간섭하지 않는다. 여러 window의 평균은 단일 window의 score noise를 줄이는 효과. 단, 마스킹 패턴이 50개이므로 **forward 연산량(FLOPs)이 단일-pass 대비 ~50×**라는 비용 한계는 유효 (발표자료 p13에서 공개; batch 확장은 wall-clock 병렬화일 뿐 연산량을 줄이지 않음). |

### 표 B: 비활성 Component (271에서 꺼져 있음 — 논문에서 언급하지 않을 것)

CODEBASE_UNDERSTANDING §7.1 + 271_CONFIG_TRUTH §VII에 전수 목록이 있다. 대표 항목:

| Component | 비활성 근거 |
|-----------|------------|
| CNN patchify (`patch_cnn`) | `patchify_mode='linear'` (271 실측) |
| Anomaly OD margin (dynamic/hinge/softplus) | `grl_disable_anomaly_loss=True` → `loss.py:259–261`에서 anomaly_loss=0으로 강제; margin 계산 경로 자체 미진입 |
| Teacher output EMA | `use_teacher_output_ema=False` |
| RevIN | `use_revin=False` |
| SCAD | `use_scad=False` |
| Adversarial Discriminator | `use_discriminator=False` |
| Teacher warmup early stop | `use_teacher_warmup_early_stop=False` |
| Complementary masking (inference 7-pass) | `eval_complementary_masking=False` |
| FM in anomaly score | `scoring.py:237` hardcoded `fm_active=False` (2026-06-01 이후) |

---

## ④ 실험 프로토콜 요약

### 데이터셋 구성 (논문 포함 6계열)

**FACT**: Simulation 및 Exathlon은 코드에 존재하나 **논문 실험에서 제외** (R33 원문).

| 데이터셋 | Entity 수 | 모델 입력 피처 수 | Train 구성 | Test 구성 | Test anomaly 비율 | 비고 |
|---------|----------|-----------------|-----------|----------|------------------|------|
| SWaT (A1+A2) | 1 (학습 1회) | **45** | A1 전체(495,000) + A2 앞 50%(224,959) | A2 뒤 50%(224,960) | 19.05% (full) / 3.68% (excl22) | dual eval: full + excl22; ⚠️ 재현성 플래그 (아래 §⑧) |
| WaDi A1 | 1 | 123 | 14days(1,209,601) + attack 앞 50%(86,400) | attack 뒤 50%(86,401) | 3.82% | |
| WaDi A2 | 1 | 123 | 14days(784,571) + attack 앞 50%(86,401) | attack 뒤 50%(86,402) | 3.87% | 원본 127 피처 → NaN 4개 drop → 123 (확정) |
| PSM | 1 | 25 | orig train 전체(132,481) + orig test 앞 50%(43,920) | orig test 뒤 50%(43,921) | 30.63% | |
| SMD | 28 machines (독립 학습) | 29–36 per machine | per-machine orig train 전체 + test 앞 50% | per-machine test 뒤 50% | machine별 상이 | |
| SMAP | 54 channels (per-channel 독립) | 25 | per-channel: orig train + test 앞 ~50% (safe-cut ±10) | test 뒤 ~50% | concat 24.54% | Pattern B: per-channel 독립 평가 |
| MSL | 27 channels (per-channel 독립) | 55 | 동일 | 동일 | concat 16.72% | Pattern B: per-channel 독립 평가 |

**공통 분할 규칙 (코드 근거)**: "원본 test를 시간 순서 기준 앞 50%를 train에 편입, 뒤 50%를 test로 사용" — SWaT `loaders.py:2018`, WaDi `loaders.py:2201`, SMD `loaders.py:1153`, PSM `loaders.py:1686–1693`, SMAP/MSL `loaders.py:2592–2595`. SMAP/MSL의 ±10 safe-cut은 anomaly region 경계 보호 장치.

**정규화**: per-feature min-max, train 구간만으로 fit (leak-free). multi-entity concat(SMD/SMAP/MSL)는 entity별 독립 fit (`docs/DATASET.md:1159`, 2026-06-02 수정).

### 비교군 label 정책 (두 조건)

**FACT** (EXPERIMENT_PROTOCOL_TRUTH §③, NOTION_DIGEST §II-4):

- **Q1 (minmax full)**: train 데이터에 anomaly 그대로 포함 (라벨 미사용 비지도 모드; 실제 운영 환경 가정).
- **Q3 (minmax normalonly)**: train에서 anomaly region 절제 후 segment-aware concat. 이것이 "비지도 모델에게 라벨의 최선 활용을 제공"하는 조건 — 비지도 방법론에서 anomaly는 오염원이므로 라벨로 제거해주는 것이 최선 (R12 논리). `comparison/data/unified_loader.py:392–485`.
- weakly-supervised 4종(deepmil/wetas/treemil/nrdetector)은 Q1 전용 (Q3에서는 train 라벨이 전부 0이라 구조적으로 실행 불가).

**논문 메인 비교 조건**: Q3 (정당화: 비지도 baseline에게 가장 유리한 라벨 활용 제공). Q1은 보조 조건.

### 평가 지표 5+1종 (정식 명칭)

**FACT** (EXPERIMENT_PROTOCOL_TRUTH §④, `evaluator.py` 직접 근거):

| 내부 키 | 정식 학술 명칭 | 제안 논문 | threshold 의존성 |
|--------|-------------|---------|----------------|
| `pak_auc_f1` | **PA%K-AUC F1** (per-K threshold 재최적화, K=0..100 trapz 적분) | Kim et al., AAAI 2022, DOI 10.1609/aaai.v36i7.20680 | threshold-sweep(적분) |
| `pak_auc_prc_auc` | **PA%K-AUC AUC-PR** (PA%K 조정 후 threshold sweep AUC-PR, K 전구간 적분) | 〃 | threshold-sweep(적분) |
| `vus_pr` | **VUS-PR** (Volume Under the Precision-Recall Surface) | Paparrizos et al., PVLDB 15(11), 2022, DOI 10.14778/3551793.3551830 | threshold-free |
| `vus_roc` | **VUS-ROC** (Volume Under the ROC Surface) | 〃 | threshold-free |
| `affiliation_f1_ar` | **Affiliation F1** (AR threshold 기반) | Huet et al., KDD 2022, arXiv:2206.13167 | threshold-dependent (AR threshold) |
| `pa_0_f1` (참고) | **PA F1** (K=0 point adjustment, F1-최적 threshold) | Xu et al., WWW 2018, DOI 10.1145/3178876.3185996 | threshold-dependent; Kim et al. AAAI 2022가 과대평가 위험 지적 → 순위 판단에 사용하지 않고 비교 가능성 위해 제시만 함 |

**Best-epoch 선정 기준**: `pak_auc_f1` (config `best_epoch_metric`, `eval_interval=5`).

**단일 threshold 사용 시**: anomaly-ratio threshold (`ar_th = quantile(score, 1 - anomaly_ratio)`, strict `>`) — `evaluator.py:790, 793–794`. 이는 oracle F1-최적 threshold가 아님 (ground truth leak 없음).

⚠️ **Oracle threshold 표기 의무** (fixer-2, MAJ-005): 위 표의 `pa_0_f1`을 포함해 F1-최적 threshold 기반 지표(`precision`/`recall`/`f1_score`/`f1_t`/`pa_{K}_*` 등)는 **test label로 최적화된 oracle(best-F1) threshold**를 사용한다 (`roc_curve` threshold 격자에서 `find_f1_optimal_idx` 선택, `evaluator.py:215–226, 928–930`). 이 지표들이 논문 테이블에 들어갈 때는 반드시 "best-F1 (oracle) threshold" 임을 표기해야 한다 — 미표기 시 reviewer의 "unfair threshold" 지적이 확실시된다. AR-threshold 변형(`_ar` suffix)이 leak-free 대안으로 병산되어 있다.

**단일 진실 원천**: 모든 지표는 MAE·baseline 공통의 `mae_anomaly/evaluator.py:864` `compute_full_metric_set`에서 계산. baseline pipeline도 이 함수에 직접 위임 (`comparison/baseline_common.py:553`).

### SWaT excl22 프로토콜

**FACT**: SWaT test split 안의 anomaly region #22(시간순 첫 번째 region, [2869, 38769), 길이 35,900 pts)가 test anomaly 질량의 **83.75%**를 차지한다.

83.75%의 재현 가능한 산출 근거 (fixer-2, MAJ-009 — 코드 docstring은 "~84%" 근사치만 제공, `find_swat_region_22` `evaluator.py:2299–2310`):
- [271c] `SWaT/A1A2_full/experiment_metadata.json` 실측: `excl_region22_info.region_length = 35,900`, `excl_region22_info.test_length = 224,960`, `metrics.anomaly_ratio = 0.190541` → test anomaly 점 수 = 0.190541 × 224,960 = **42,864** → 35,900 / 42,864 = **0.83753 = 83.75%**.
- EXPERIMENT_PROTOCOL_TRUTH §⑥ 원 CSV 직접 계산(83.75%)과 일치.

full 조건에서는 이 단일 사건 탐지 여부가 recall 대부분을 결정해 모델 변별력이 낮아진다. 실측: full `pak_auc_f1` **0.9444** vs excl22 **0.6273** — 정정(fixer-2, MAJ-009): 초판의 0.944/0.629는 발표 시점 스냅샷이었으나, SWaT entity는 완주 상태이므로 본 수치는 최종 `experiment_metadata.json`(`metrics.pak_auc_f1=0.94436`, `metrics_excl_region22.pak_auc_f1=0.62730`) 실측값으로 갱신 — stale 아님. **(α-m3 출처 주석, 2026-06-10)** 0.62730은 `A1A2_full` metadata의 `metrics_excl_region22.pak_auc_f1`(full의 best epoch 기준)이며, `A1A2_excl22` entity 자체 headline `metrics.pak_auc_f1`은 **0.62899**(best epoch을 `excl22_pak_auc_f1`로 별도 선정 — 271_CONFIG_TRUTH §IV r2 주석 정합). 두 값 모두 실존 — 논문 표가 어느 쪽 기준인지는 Phase 3 결정 사안 (혼용 금지).

구현: 단일 학습 + dual evaluation (excl22는 eval_mask로 region 22 구간만 평가에서 제외; 학습·score 산출은 동일). 모든 baseline에 동일하게 적용. 논문 모델 변별은 excl22 기준 사용.

### 라벨 희소화 sweep (R32)

**FACT**: 전용 파라미터·스크립트는 현재 코드에 없다. 재사용 가능한 기존 메커니즘은 존재한다: `mae_anomaly/datasets/noisy.py:7–87` `NoisyLabelSlidingWindowDataset` + `scripts/run_base_experiments.py:397–416` `apply_normal50_noise` (train anomaly region의 50%를 무작위 0으로 재라벨, 현재 비활성 `normal50: False`).

**UNKNOWN**: 실험 수치가 없다. 논문 본문에서는 placeholder로만 표기.

---

## ⑤ 논문 제외 목록 종합

**FACT**: 아래 항목들은 코드/파이프라인에는 존재하나 논문 실험·설명에서 언급하지 않는다.

| 제외 항목 | 근거 |
|---------|------|
| **Simulation 데이터셋** (275K × 8 features, 9 anomaly types) | R33 원문 ("논문에 포함하지 않을 예정"). |
| **Exathlon 데이터셋** (6 apps, 19 features) | R33 원문. |
| **Gaussian smoothing** | R34 원문. 코드 실재 지점 2종 모두 271 파이프라인 무참조 — ① simulation 데이터 생성 내부(`_generate_phase_jitter` 등), ② q3_exploration 스크립트의 score post-hoc `gauss()` (`mae_anomaly/scripts/q3_exploration/core/scoring.py:48`, B2 variant) (271_CONFIG_TRUTH §VI r2 정정 정합 — CG-1 패치 2026-06-11). |
| **MAE 271 B2** (post-hoc sigma=10 Gaussian smoothing on anomaly score) | 논문 기여로 주장하기에 적절한지 미결 (NOTION_DIGEST §IV-3). Phase 3 판단 사안. |
| **Dynamic margin / hinge / softplus / none** (anomaly OD loss 변형) | `grl_disable_anomaly_loss=True`로 anomaly_loss=0 강제 (`loss.py:259–261`) → margin 계산 경로 자체 미진입. 271에서 무효. |
| **patch_cnn patchify** | `patchify_mode='linear'` (271 실측). |
| **RevIN, EMA teacher, SCAD, Discriminator, WDGRL, balanced sampling** | 전부 271 config에서 False/비활성. |
| **eval_complementary_masking** (7-pass inference) | `eval_complementary_masking=False`. |
| **Feature Matching in anomaly score** | `scoring.py:237` hardcoded `fm_active=False`. FM은 학습 손실로만 기능. |

---

## ⑥ Notion 주장 vs 검증 사실의 미해결 차이 + Phase 3 판단 사안

### 미해결 차이 (Notion 주장과 코드 사실 간)

| # | Notion 주장 | 코드/metadata 사실 | 상태 |
|---|------------|-------------------|------|
| N1 | SMAP/MSL이 MAE 학습 범위 미등록 (Notion [N-METH] 2026-05-31) | [271c]에 SMAP/MSL 결과 존재; `run_base_experiments.py:299–312, 379–390`에서 통합 완료 | **코드가 ground truth** — Notion 스냅샷 stale. |
| N2 | batch_size=512 (Set C preset) | 271 metadata `batch_size=1024` (override) | **metadata 우선**. |
| N3 | WaDi A2 features=127 | 271 metadata `num_features=123`; 재계산 123 확인 | **123 확정** (reconciler §III). |
| N4 | FM이 anomaly score에 포함 (C3 서술) | `scoring.py:237` `fm_active=False` hardcoded (2026-06-01) | **코드 우선**. FM은 훈련 loss 전용. Notion 페이지 자체도 이 변경을 명시함. |
| N5 | SMD features=38 per machine | 271 metadata 실측 29–36 | **29–36 (실측 22/28 machine 기준; 잔여 6 machine은 metadata 미존재로 측정 미완 — 완주 후 범위 재확인 필요)** (fixer-2, MIN-004 정밀화). |

### Phase 3로 넘길 판단 사안

- **Contribution 구조 (C1–C4)**: Notion이 정리한 C1(Masking)·C2(Self-distillation)·C3(Discrepancy)·C4(GRL semi-supervised) 4분할을 논문의 공식 contribution 구조로 그대로 채택할지, 재정의할지는 Phase 3에서 결정.
- **"비대칭 학습(asymmetric learning)"이라는 발표 제목 framing**: 용량 비대칭(3층 vs 2층)·학습 신호 비대칭(teacher 전 패치 vs student 정상 패치만)·구조 비대칭(GRL은 student에만)의 3축 비대칭 narrative를 논문 contribution 프레이밍으로 채택할지 Phase 3 판단.
- **"semi-supervised" vs "weakly supervised" vs "PU learning" 표현**: §② 참고. Phase 3에서 확정.
- **논문 메인 비교표 조건 확정**: Q3 단독 vs Q1+Q3 병기. EXPERIMENT_PROTOCOL_TRUTH REQUEST-3 미결. 사용자 결정 사항.
- **MAE 271 B2 (post-hoc smoothing) 포함 여부**: 논문 기여로 주장하기에 적절한지, 단순 engineering trick인지. Phase 3 판단.
- **Affiliation F1·PA F1의 threshold 표기 방식**: EXPERIMENT_PROTOCOL_TRUTH REQUEST-1 해소 후속. PA F1은 현재 F1-최적 threshold만 존재; AR-threshold PA F1은 미구현. 보고 방식 확정 필요 (사용자 결정).
- **DAGMM 구현 provenance**: TranAD repo reimplementation (GMM energy 제거 simplified variant). 논문에서 "DAGMM"으로 표기하는 것의 적절성. Phase 3 판단. ⚠️ 조기 결정 권고 (fixer-2, NOTE-003): GMM energy를 제거한 구현을 무수식 "DAGMM"으로 표기하면 **방법 재정의**에 해당해 reviewer의 "이 구현은 DAGMM이 아니다" reject 사유가 될 수 있다. 표기 후보: "DAGMM-simplified" / 각주로 구현 차이 명시. Phase 3 시작 시점에 최우선 확정할 것.
- **deepmil encoder 출처 서술**: WETAS DiCNN encoder 사용 (Sultani CVPR 2018 원논문은 frozen C3D feature 기반). 논문 서술 방식 확정 필요.
- **tranad LR**: Notion이 paper-text 0.01 대신 코드 `constants.py` lr=1e-4 사용한다고 명시. 어느 값이 논문 표에서 faithful한지 확정 필요.

---

## ⑦ 코드 공개 (R25)

**FACT**: 코드는 git으로 공개할 예정이다 (R25 원문: "코드는 git으로 공개할 예정임"). 논문에 코드 링크를 포함하는 것이 자연스럽다면 포함하고, 그렇지 않으면 포함하지 않는다.

**현재 repo 상태**: git repo, branch: machineA. public 공개 시점·링크는 미결.

**공개 전 점검 checklist (fixer-2, NOTE-002 — 전부 미결; 논문 제출 전 해소 필요)**:
- [ ] 공개 branch 결정: machineA 그대로 vs main 정리 후 공개 (현재 main이 default branch).
- [ ] 공개 범위 정리: `configs/` 내 실험 큐 JSON·`results/`·`temp/`·`paper*/` 등 비공개 대상 분리.
- [ ] secret/credential 스캔 (API key, 절대경로, 사용자 식별 정보).
- [ ] 재현 진입점 문서화 (환경 + 학습/평가 명령 — exp271 재현 기준; SWaT 45-feature 재현성 플래그 §⑧-6 포함).
- 미해소 시 reproducibility claim 전체가 흔들린다 — Phase 5(본문 코드 공개 서술) 전에 사용자 확인 필요.

---

## ⑧ 이후 Phase가 이 문서를 어떻게 써야 하는가

### 정본 우선순위

```
271_CONFIG_TRUTH.md §II (metadata 전수표)
  > RESEARCH_SYNTHESIS.md (본 문서 — 종합·해석·분류)
  > CODEBASE_UNDERSTANDING.md (코드 상세)
  > EXPERIMENT_PROTOCOL_TRUTH.md (실험 프로토콜 상세)
  > NOTION_DIGEST.md (Notion 주장 — Phase 3 contribution 판단 입력)
  > CONFERENCE_PDF_DIGEST.md (발표자료 — 논리 전개 참고용; 수치 사용 금지)
```

### 사용 지침

1. **수치 참조**: 모델 아키텍처·학습 하이퍼파라미터 수치는 반드시 271_CONFIG_TRUTH §II–VIII을 1차 소스로 확인. 본 문서 §③ 표와 일치하나, 충돌 시 271_CONFIG_TRUTH 우선.
2. **contribution 서술**: Phase 3 완료 전에는 Notion의 C1–C4를 "Notion의 주장"으로만 인용하고, 논문의 공식 contribution으로 단정하지 않는다.
3. **실험 결과 수치**: NOTION_DIGEST §II-6의 Q3 partial 수치(SMD/Exathlon 열)는 per-entity 정규화(2026-06-02) 이전의 stale 값이다. 논문 표 작성 시 실험 큐 완주 후 최신 수치만 사용한다.
4. **데이터셋 포함 여부**: Simulation·Exathlon은 항상 제외.
5. **발표 수치**: CONFERENCE_PDF_DIGEST의 성능 수치(p22, p29–33)는 발표 시점 스냅샷 — 논문에 직접 사용 금지.
6. **SWaT feature 수 재현성 플래그**: 현 machineA 환경에서 재실행 시 `patch_embed` 입력 차원이 45와 일치하는지 반드시 확인할 것. 불일치 시 checkpoint 로드 실패 가능 (EXPERIMENT_PROTOCOL_TRUTH FEEDBACK-7).

---

## ⑨ REQUEST / FEEDBACK 블록

### REQUEST (이후 phase에서 해소 필요)

**REQUEST-A (Phase 3 필수)**: 비교군 메인 조건을 Q3 단독으로 확정할지, Q1+Q3 병기로 할지 사용자 결정 필요. 현재 Q1 실험 결과 일부 pending 상태 (NOTION_DIGEST §II-1: "Q1: pending").

**REQUEST-B (Phase 3 판단)**: PA F1의 threshold 표기 방식 확정. 현재 `pa_0_f1`(F1-최적 threshold)만 존재; AR-threshold PA F1(`pa_0_f1_ar`)은 코드에 없음 (EXPERIMENT_PROTOCOL_TRUTH REQUEST-1 후속). 논문 표에서 어느 threshold를 사용할지 사용자 확인 필요.

**REQUEST-C (Phase 2 실험)**: 라벨 희소화 sweep (R32) 수치가 없다. 설계 입력은 EXPERIMENT_PROTOCOL_TRUTH §⑦에 있으나 실험 미실행. 논문에 포함하려면 별도 실험 실행 필요.

**REQUEST-D (Phase 2/3)**: baseline 쪽 per-entity 정규화(2026-06-02) 이후 SMD/SMAP/MSL/Exathlon 결과가 stale. 완전한 비교표를 위해 재실행 필요. 현재 271 entity도 SMD 22/28, SMAP 5/54, MSL 5/27만 완료.

**REQUEST-E (잔존 모순 — 사용자 판단 필요)**: NOTION_DIGEST §II-5의 "5+1종" 지표 목록 (`pak_auc_f1`, `prc_auc`, `f1_t`, Rank Avg)과 본 문서 §④의 지표 목록 사이에서 `f1_t`의 논문 포함 여부가 미확정이다. `f1_t`는 코드에 존재하나 EXPERIMENT_PROTOCOL_TRUTH의 "5종" 지표 정의에서는 보조 지표로 분류된다. 논문 메인 테이블에 포함할지 확정 필요.

**REQUEST-F (Phase 2 실험 — fixer-2 신규, NOTE-001 격상)**: Teacher-only warmup(250 epochs)의 효과를 입증하는 ablation이 부재하다 (예: `teacher_only_warmup_epochs ∈ {0, 50, 250}` 비교). 표A의 해당 component motivation은 현재 발표 학습 곡선의 정성적 근거뿐이라 reviewer 방어 불가. 논문에서 warmup을 설계 기여로 주장하려면 이 ablation이 필수.

### FEEDBACK

**FEEDBACK-1 (재현성)**: SWaT 모델 학습 당시 source-machine의 CSV가 45-feature 버전이었으나 현 machineA의 raw CSV와 loader(`load_swat_a1a2_raw`)는 51 features를 반환한다. 재실험 전 반드시 feature 수 확인 및 loader 수정 검토 필요. (EXPERIMENT_PROTOCOL_TRUTH FEEDBACK-7 상세)

**FEEDBACK-2 (weakly-supervised 4종 미실행)**: deepmil·wetas·treemil·nrdetector의 GPU 전체 실험이 미실행. 논문에 포함 여부 및 결과 없이 포함 시 서술 방식 결정 필요.

**FEEDBACK-3 (RankAvg 재정의)**: 기존 Notion의 RankAvg는 Exathlon 포함 6-dataset 기준. 논문용은 Simulation·Exathlon 제외 + SMAP/MSL 포함으로 재계산 필수 (EXPERIMENT_PROTOCOL_TRUTH FEEDBACK-3).

---

## 부록: 정정 이력

### 2026-06-10 fixer-2 (adversarial review `paper/99_reviews/p1_codebase_synthesis_r1.md` 전수 반영; fixlog: `p1_codebase_synthesis_fixlog_r2.md`)

1. **[BLK-005/MAJ-006] §② 전면 재구성**: "오염된 unlabeled 다수" ↔ "모든 샘플 라벨 존재" 모순 제거 — 3단 구조(②-1 설정/가정 R11 원문 → ②-2 main 실험 구현 = label 가용성 상한 케이스 R13 → ②-3 라벨 희소화 sweep 계획 R32)로 재서술. ②-5 신설: 비지도 방법이 label 사용 3지점을 구조적으로 구현 불가한 코드-직결 논리 (R11 충족).
2. **[BLK-001] 표A FM/GRL 행**: λ_FM(`trainer.py:639–653`)·λ_GRL(`trainer.py:752–763`)은 각각 별개의 trainer inline grad-ratio 공식 + 직전 epoch 값 적용 — discriminator 전용 `compute_adaptive_lambda`(`loss.py:683`)와 무관함을 명시.
3. **[BLK-002] 표A Patch→Point 행**: "50회 순차 forward" → 마스킹 패턴 batch-차원 확장 병렬 forward(`evaluator.py:1650, 1801–1818`; `patch_batch_size=2`는 수치 무영향)로 정정; 비용 한계는 FLOPs ~50× 관점으로 재서술; 타임스텝당 ~10회 평균의 유도(덮는 윈도우 수) 정정.
4. **[BLK-004] 표A GRL 행**: "focal γ=2" → focal-style BCE 변형 `(1−exp(−BCE))²×BCE` (`loss.py:330–340`), pos_weight 내장으로 표준 focal loss(Lin et al. 2017) 아님 — "standard focal loss" 표기 금지 플래그.
5. **[MAJ-004/MIN-003] 표A GRL 행**: AnomalyClassifierHead = 2-layer MLP(hidden=d_model//2 자동, `model.py:177–186`) 표기 확정("1-layer MLP" 표현 금지); 패치별 독립 적용 + masked 패치만 손실(`model.py:1153–1154`, `loss.py:283–284`) 세부 추가.
6. **[MAJ-005] §④**: oracle(best-F1) threshold 기반 지표의 논문 표기 의무 경고 추가.
7. **[MAJ-007] §①**: "총 112 entity" 완료 뉘앙스 제거 → 실측 완료 37 entity(SWaT 2 dual-eval·WaDi 2·PSM 1·SMD 22/28·SMAP 5/54·MSL 5/27), 잔여 실행 진행 중 + 논문은 placeholder 정책(A8/R3) 명시.
8. **[MAJ-009] §④ excl22**: 83.75%의 재현 가능 산출 근거(metadata `excl_region22_info` + `anomaly_ratio` 유도 = 35,900/42,864; 코드 docstring ~84%는 `evaluator.py:2299–2310`) 명시; 0.944/0.629는 최종 metadata 실측(0.9444/0.6273)으로 갱신(stale 아님 확인).
9. **[MIN-004] §⑥ N5**: SMD 29–36은 실측 22/28 machine 기준, 잔여 6 machine 미측정 명시.
10. **[NOTE-001] 표A warmup 행 + §⑨ REQUEST-F**: warmup ablation 부재를 CRITICAL RISK로 격상, Phase 2 필수 실험으로 등재.
11. **[NOTE-002] §⑦**: 코드 공개 전 점검 checklist(branch/범위/secret/재현 진입점) 추가.
12. **[NOTE-003] §⑥ DAGMM**: "DAGMM-simplified" 표기 후보 등 조기 결정 권고 추가.
