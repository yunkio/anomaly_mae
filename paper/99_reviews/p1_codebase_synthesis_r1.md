---
phase: 1
agent: adversarial-reviewer-A
directives: [T1, R10, R11, R17]
last_modified: 2026-06-10
---

# Adversarial Review: Phase 1 Artifacts — CODEBASE_UNDERSTANDING + RESEARCH_SYNTHESIS

## 판정 요약

| 문서 | 판정 | 잔존 BLOCKER | 잔존 MAJOR | MINOR / NOTE |
|------|------|-------------|-----------|--------------|
| CODEBASE_UNDERSTANDING.md | **FAIL** | 3 | 4 | 4 |
| RESEARCH_SYNTHESIS.md | **FAIL** | 2 | 5 | 4 |

**총 BLOCKER: 5건, MAJOR: 9건, MINOR/NOTE: 8건**

reconciler 정정 후에도 두 문서 모두 method 섹션 작성에 직접적 오류를 유발할 오기(BLOCKER)가 잔존한다. RESEARCH_SYNTHESIS는 추가로 창작성 주장과 수치 누락 위험이 있다.

---

## 발견사항 상세

---

### BLOCKER

---

**[BLK-001]** CODEBASE_UNDERSTANDING §2.2 / RESEARCH_SYNTHESIS ③표A

**위치**: CODEBASE §2.2 "Adaptive Lambda (VQGAN-style)" 공식; RESEARCH_SYNTHESIS 표A "Feature Matching Loss" 행 adaptive lambda 설명

**오류 주장**:
- CODEBASE §2.6: `λ = (||∇_w L_normal|| + ||∇_w L_anom_forward||) / (||∇_w L_adv|| + δ)`
  이것이 "GRL과 FM adaptive weighting 모두에 사용된다"고 서술
- RESEARCH_SYNTHESIS 표A FM 행: `clip(||grad_main||/(||grad_fm||+1e-4), 0, 10)`

**코드 증거**:
- `loss.py:683–728` `compute_adaptive_lambda` — 이 함수는 **Discriminator** adaptive lambda 전용 (`normal_loss + anomaly_disc_forward` vs `adv_loss`). GRL이나 FM에 사용되지 않는다.
- **GRL adaptive lambda** (`trainer.py:754–760`): `(_main_g.norm() / (_grl_g.norm() + 1e-4)).clamp(0.0, 10.0)` — 분자가 L_normal + L_anom_forward가 아니라 **현재 total loss 기준 main gradient norm** 단일값.
- **FM adaptive lambda** (`trainer.py:646–648`): `(_main_g_fm.norm() / (_fm_g.norm() + 1e-4)).clamp(0.0, 10.0)` — 마찬가지로 FM 전용 단순 ratio.
- 두 공식 모두 `loss.py:compute_adaptive_lambda`를 호출하지 않는다.

**영향**: 논문 method 섹션의 GRL/FM adaptive lambda 수식이 틀리게 기술될 확률 100%. VQGAN-style이라는 framing 자체도 discriminator path에만 해당한다 (trainer.py:604–628이 discriminator path에서만 `compute_adaptive_lambda`를 호출).

**수정안**: CODEBASE §2.6 및 §2.3/§2.4를 세 가지 adaptive lambda 경로로 분리 기술:
1. **Discriminator λ** (`loss.py:683–728`, VQGAN-style, exp271 비활성)
2. **GRL λ** (`trainer.py:754–760`, main_grad / grl_grad, prev-epoch 평활화)
3. **FM λ** (`trainer.py:646–648`, main_grad / fm_grad, prev-epoch 평활화)
RESEARCH_SYNTHESIS 표A도 동일하게 수정.

---

**[BLK-002]** CODEBASE_UNDERSTANDING §3.1 — 추론 절차 오기 (leave-one-out)

**위치**: CODEBASE §3.1 "Forward Pass (Inference)": "each patch is masked in turn, and reconstruction/discrepancy is computed for that patch."

**오류 주장**: 이 서술은 50회 순차적 forward pass를 암시하나, `evaluator.py:1647–1719` docstring/구현을 보면 **"Optimized: All patches processed in a single forward pass by expanding batch dimension"** — 50개 패치 마스킹을 batch dimension으로 확장해 단일 forward pass로 병렬 처리한다.

**코드 증거**: `evaluator.py:1647–1648` docstring: "Optimized: All patches processed in a single forward pass by expanding batch dimension." `patch_batch_size=2`로 나눠 처리하나 이는 메모리 분할이지 순차 masking이 아님 (`evaluator.py:1706–1715` 주석 명시).

**영향**: 논문에서 추론 복잡도를 O(N_patches × 1_forward)로 기술해야 할 것을 O(N_forward × N_patches)로 잘못 표기하거나, 복잡도를 실제보다 높게 주장할 수 있다.

**수정안**: "각 패치를 순서대로 마스킹해 50회 forward"가 아니라 "50개 마스킹 패턴을 batch 차원으로 확장한 단일(또는 patch_batch로 분할된) forward pass"로 정정. RESEARCH_SYNTHESIS 표A "Patch → Point 집계" 행도 동일하게 정정.

---

**[BLK-003]** CODEBASE_UNDERSTANDING §2.6 — compute_adaptive_lambda 적용 대상 오기

**위치**: CODEBASE §2.6 마지막 문장: "Used for both discriminator and FM adaptive weighting."

**오류**: `compute_adaptive_lambda` (`loss.py:683–728`)는 discriminator 전용이다. FM의 adaptive weighting은 `trainer.py:646–648`의 별도 inline 코드로 처리되며 해당 함수를 호출하지 않는다. GRL도 마찬가지 (`trainer.py:754–760`).

**코드 증거**: `trainer.py:638–658` FM adaptive lambda 블록은 `compute_adaptive_lambda`를 한 번도 import/호출하지 않는다. `trainer.py:746–771` GRL 블록도 동일.

**수정안**: "Used for discriminator adaptive weighting only (exp271 비활성). GRL/FM은 별도 inline gradient-ratio 공식 사용."

---

**[BLK-004]** CODEBASE_UNDERSTANDING §2.4 / RESEARCH_SYNTHESIS 표A — GRL focal loss 공식 오기

**위치**: CODEBASE §2.4: `focal_loss = ((1 - exp(-BCE))^2) × BCE`
RESEARCH_SYNTHESIS 표A GRL 행: `focal γ=2`

**오류**: CODEBASE의 focal_loss 공식은 틀렸다. 코드 실제:
```python
_p_t = torch.exp(-_bce)
_focal = ((1 - _p_t) ** 2.0) * _bce
```
이것은 `_p_t = exp(-BCE)`, `focal = (1 - exp(-BCE))^2 × BCE`이며, 이는 **표준 focal loss `(1-p_t)^2 × BCE_per_sample`의 변형이지만 표준적이지 않다**.
표준 focal loss는 `p_t = σ(logit)` (시그모이드 확률)을 사용하며, 여기서는 `exp(-BCE)` (BCE 값의 지수 변환)을 사용한다. 이 차이는 `exp(-BCE) ≠ σ(logit)` 이므로 공식이 학술 논문에서 표준 focal loss라고 주장되면 리뷰어가 즉시 문제를 제기할 것이다.

**코드 증거**: `loss.py:337–340` 직접 확인.

**수정안**: 논문에서 이 공식을 기술할 때 "standard focal loss (Lin et al. 2017)와 구조적으로 유사하나 p_t = exp(-BCE_i)를 사용한다"는 주석 필요. 또는 구현을 표준 focal loss (`p_t = sigmoid(logit)`)로 수정 후 공식 일치.

---

**[BLK-005]** RESEARCH_SYNTHESIS ② — "오염된 unlabeled 다수" 표현의 사실 오류

**위치**: RESEARCH_SYNTHESIS §② 첫 번째 FACT 항목: "오염된 unlabeled 다수: 원본 train 파일(대부분 정상, 일부 미라벨 이상 혼재 가능) + 원본 test 파일의 앞 50%를 train에 편입. 편입된 test 앞 50% 안에는 실제 이상이 섞여 있으며 그에 대한 라벨도 포함된다."

**오류**: 두 번째 FACT 항목이 모순적이다: "모든 학습 샘플에는 실제로 라벨이 부여되어 있다." 이를 종합하면 "labeled anomaly 소수 + labeled normal 다수"이지 "unlabeled 다수"가 아니다. 또한 원본 train 파일에도 anomaly 라벨이 존재한다. 즉 "오염된 unlabeled" 표현은 사실적으로 부정확하다.

**영향**: 논문의 PU setting 또는 semi-supervised 주장의 근거가 될 이 정의가 reviewer에게 즉시 "그럼 fully labeled 아닌가?"라는 반박을 초래한다. R11 directive의 핵심 구분이 충분히 논리화되지 않은 상태.

**수정안**: "오염된 unlabeled"를 "라벨이 있지만 운영 환경에서는 수집 당시 레이블 비용이 높고 anomaly 비율이 극히 낮아 실질적으로 PU-like 상황"으로 정밀화. 또는 아예 "contaminated semi-supervised" 프레이밍으로 통일하고 "unlabeled"라는 단어를 §②에서 제거.

---

### MAJOR

---

**[MAJ-001]** CODEBASE_UNDERSTANDING §2.5 — Total Loss 공식에 FM 이중 경로 오기

**위치**: CODEBASE §2.5: `L_total = L_recon + L_OD [+ adaptive_lambda × grl_loss_weight × L_GRL (added in trainer)] [+ adaptive_lambda × fm_loss_weight × L_FM (if fm_adaptive_lambda=True)]`

**오류**: 이 공식은 FM이 `fm_adaptive_lambda=True`일 때만 trainer에서 추가된다는 것을 맞게 서술하지만, `fm_adaptive_lambda=False` 경우 (`loss.py:438`: `discrepancy_loss = normal_loss + anomaly_loss + self.fm_loss_weight * fm_loss`)를 "비적용 케이스"처럼 처리해 마치 `fm_adaptive_lambda=True`일 때만 FM이 포함된다는 인상을 준다. 실제로는 두 경로 모두 FM을 포함하며 다만 추가 위치가 다르다.

**더 중요한 오류**: adaptive lambda 변수명이 혼재된다. 공식에서 `adaptive_lambda` 단일 기호를 사용하나, GRL과 FM의 adaptive lambda는 별개 값이다 (BLK-001에서 이미 지적). 이 공식이 논문에 그대로 들어가면 GRL λ = FM λ로 오해된다.

**수정안**: L_GRL 항에 `λ_GRL`, FM 항에 `λ_FM`으로 별도 표기하고, 두 lambda의 산출 공식을 각각 명시.

---

**[MAJ-002]** CODEBASE_UNDERSTANDING §3.3 — "약 10회 평균" 근거 계산 오류

**위치**: CODEBASE §3.3: "test stride=49 (auto: `resolve_test_stride` = `seq_length // 10 - 1` = 49 …): each timestep is covered by approximately 500/49 ≈ 10 windows × patch-position coverage, giving ~10 score averages per point."

**오류**: "약 10회 평균"의 계산 로직이 잘못되었다. 타임스텝 t를 덮는 (window, patch) 쌍의 수는 단순히 500/49가 아니다. 윈도우 수(sequence coverage) × 해당 타임스텝을 포함하는 patch 수의 곱이 아니라, 각 윈도우 내에서 t를 포함하는 patch는 정확히 1개(`patch_size=10`이므로 각 타임스텝은 특정 1개 patch에 속한다)이다. 따라서 coverage = t를 포함하는 윈도우의 수 ≈ seq_length / test_stride = 500/49 ≈ 10.2이며, 이것이 "~10 score averages per point"의 올바른 유도다 (`×patch-position coverage` 인수가 불필요하게 삽입됨).

**추가 오류**: `resolve_test_stride`는 `seq_length // 10 - 1`이지만 이 값은 **sentinel이 -1일 때만** 적용된다 (`utils/experiment.py:34`). exp271 metadata `sliding_window_test_stride=-1`로 sentinel 적용이 맞다.

**수정안**: "test stride=49 → 각 타임스텝을 포함하는 윈도우 수 ≈ 500/49 ≈ 10.2 → ~10회 평균. (각 윈도우 내에서 타임스텝 t를 포함하는 patch는 1개 뿐, patch_size=10)"으로 정정.

---

**[MAJ-003]** CODEBASE_UNDERSTANDING §5.6 / RESEARCH_SYNTHESIS ④ — threshold 설명 불완전

**위치**: CODEBASE §5.6: "Threshold is set at the F1-optimal point on the ROC curve computed on the test set point-level scores."
RESEARCH_SYNTHESIS §④: "Best-epoch 선정 기준: `pak_auc_f1`"

**문제**: ROC curve에서 F1-optimal threshold를 구하는 것은 일반적이지 않다 (ROC curve는 F1-optimal threshold를 직접 제공하지 않으며, 통상 별도 threshold sweep을 통해 F1이 최대화되는 점을 찾는다). 코드에서 `evaluator.py:928–930`은 `fpr, tpr, thresholds = roc_curve(…)` → `find_f1_optimal_idx`로 F1-최적 threshold를 ROC의 thresholds 배열에서 찾는다. 기술적으로는 "ROC curve의 thresholds에서 F1을 최대화하는 threshold 선택"이 더 정확하다. "F1-optimal point on the ROC curve"라는 표현은 precision-recall curve를 연상시켜 혼동을 준다.

**수정안**: "Threshold는 test set ROC curve의 threshold sweep에서 F1을 최대화하는 값으로 설정. `roc_curve`의 threshold 격자 중 `find_f1_optimal_idx`로 선택 (`evaluator.py:928–930`)."

---

**[MAJ-004]** RESEARCH_SYNTHESIS ③표A — AnomalyClassifierHead 아키텍처 오기

**위치**: RESEARCH_SYNTHESIS 표A GRL 행: `AnomalyClassifierHead: LayerNorm → Linear(512→256) → GELU → Dropout(0.1) → Linear(256→1)`

**오류**: `grl_cls_hidden=0`이면 `hidden_dim = d_model // 2 = 512 // 2 = 256`으로 자동 계산된다 (`model.py:179`). 따라서 `Linear(512→256)`는 맞다. 그러나 아키텍처 서술의 "1-layer MLP"는 오해를 준다 — 실제로는 **2-layer MLP** (Linear(512→256) + Linear(256→1))이며, LayerNorm + GELU + Dropout까지 포함하면 4-block 구조다.

**271_CONFIG_TRUTH §VIII GRL Details 표**: "1-layer MLP: LayerNorm → Linear(d_model, d_model//2=256) → GELU → Dropout(0.1) → Linear(256, 1)"라고 동일 오류가 있다.

**수정안**: "2-layer MLP: LayerNorm(d_model) → Linear(d_model→d_model//2) → GELU → Dropout(0.1) → Linear(d_model//2→1)"로 정정. "1-layer MLP"라는 표현 제거.

---

**[MAJ-005]** CODEBASE_UNDERSTANDING §5.6 — 추론 시 threshold가 test set 라벨 사용

**위치**: CODEBASE §5.6, RESEARCH_SYNTHESIS §④.

**문제**: "Threshold is set at the F1-optimal point … on the test set" — 이것은 oracle threshold (test ground-truth leak)다. 논문에서 이 threshold를 사용한 지표(precision, recall, f1_score, pa_0_f1 등)는 threshold selection에 test label을 사용한다는 점을 반드시 명기해야 한다. 그렇지 않으면 reviewer가 "비공정한 threshold"라고 지적한다.

RESEARCH_SYNTHESIS §④에서 이를 부분적으로 인지하여 AR threshold를 별도로 언급하고 있으나, **oracle threshold 지표들이 논문 테이블에 들어갈 때 반드시 "oracle" 또는 "best F1 threshold"임을 표기**해야 한다는 경고가 명시되지 않았다.

**수정안**: CODEBASE §5.6 및 RESEARCH_SYNTHESIS §④에 "이 threshold는 test label을 알아야 최적화 가능한 oracle threshold임을 논문 테이블에서 반드시 표기해야 함" 경고 추가.

---

**[MAJ-006]** RESEARCH_SYNTHESIS ② — PU Learning 정의 논리 충분성 미달 (R11)

**위치**: RESEARCH_SYNTHESIS §② "PU Learning과의 관계"

**문제**: R11 directive는 "소수 labeled 활용 불가가 기존 unsupervised의 핵심 한계"임을 정의하도록 요구한다. 현재 문서는 INFERENCE로 분류한 PU 유사성 논의를 Phase 3으로 미루고 있으나, **기존 비지도 방법이 왜 labeled anomaly를 활용하지 못하는지에 대한 코드-직결 논리**가 완전히 누락되어 있다.

예: 기존 비지도 방법(OCSVM, AnoGAN 등)이 anomaly label을 masking priority나 loss gradient 방향 설정에 사용할 수 없는 이유가 구체적으로 서술되지 않았다. "비지도 방법은 이 소량의 anomaly 라벨을 전혀 활용하지 못한다"(§① 요약)는 주장이 `force_mask_anomaly`, `patch_has_anomaly`, GRL target과 연결되어야 하는데, R10 원재료 표(표A)에서 각 component의 "왜 이래야만 하는가"는 서술되었으나 **이 component들이 왜 비지도 방법에서 구현 불가인지**를 R11 관점에서 정면 다루지 않았다.

**수정안**: §②에 "기존 비지도 방법 대비 제안 방법이 labeled anomaly를 활용하는 3가지 구체적 지점 — `force_mask_anomaly`, 손실 방향 분기, GRL 타겟 — 이 비지도 방법의 구조적으로 불가능한 이유 (비지도 모델은 training 시 label 자체가 없으므로)"를 추가하는 것이 R11의 충족 조건이다.

---

**[MAJ-007]** RESEARCH_SYNTHESIS ① 요약 — SMAP/MSL entity 수 불일치

**위치**: RESEARCH_SYNTHESIS §① 요약: "SWaT·WaDi A1/A2·PSM·SMD(28)·SMAP(54)·MSL(27) 6계열 총 112 entity"

**문제**: 271_CONFIG_TRUTH §I 실측에 따르면 실제로 완료된 entity는 **37개** (SMAP 5, MSL 5, SMD 22, SWaT 2, WaDi 2, PSM 1)이고, SMAP 54·MSL 27은 계획 수치다. RESEARCH_SYNTHESIS §④의 REQUEST-D에서도 "SMD 22/28, SMAP 5/54, MSL 5/27만 완료"라고 명시된다. 그러나 §① 요약에서 "SMAP(54)·MSL(27) … 총 112 entity"로 계획 수치를 사용해 독자에게 이미 실험이 완료된 것처럼 오인을 준다.

**수정안**: §① 요약에 "현재 완료된 entity 37개(계획: 112)" 또는 "(진행 중: SMAP 54, MSL 27, SMD 28 목표)"라는 명시 필요.

---

**[MAJ-008]** CODEBASE_UNDERSTANDING §6.2 — PA%K 보고 step 불일치

**위치**: CODEBASE §6.2: "`pa_{K}_f1` 등, K = 0, 5, 10, ..., 100"

**오류**: `compute_full_metric_set` (`evaluator.py:886`)의 docstring: "Per-K PA%K (k=0..100 step 5)"이나, `compute_pa_k_auc` (`evaluator.py:998–1016`)는 K=0,1,...,100 step=1로 적분한다. 즉 **보고 키는 step=5** (0,5,10,...,100)이지만 **AUC 적분은 step=1**이다. CODEBASE §6.2의 PA%K AUC 항목에 "K=0..100 step1 적분"이 서술되어 있어 이 부분은 맞지만, Per-K 섹션의 "K = 0, 5, 10, ..., 100" 표기와 합쳐지면 혼동을 준다.

**수정안**: "Per-K PA%K 보고 키: K = 0, 5, 10, ..., 100 (step=5, 21개 값). PA%K AUC 적분: K = 0..100 step=1 (101점 trapz). 두 해상도는 다름."으로 명확히 구분.

---

**[MAJ-009]** RESEARCH_SYNTHESIS §④ SWaT excl22 설명 — "83.75%" 수치 출처 불명

**위치**: RESEARCH_SYNTHESIS §④ SWaT excl22 프로토콜: "region #22 … test anomaly 질량의 83.75%를 차지한다 (`evaluator.py:2302–2306`)"

**문제**: `evaluator.py`의 행 번호 2302–2306을 직접 확인할 수 없으나, 이 수치(83.75%)와 line 참조가 NOTION_DIGEST 또는 EXPERIMENT_PROTOCOL_TRUTH에서 인용한 것인지, 코드에서 직접 계산한 것인지 불명확하다. 논문 reviewer는 이 수치의 재현 방법을 물을 것이다.

또한 "full `pak_auc_f1` 0.944 vs excl22 0.629"는 발표 시점 스냅샷 ([271c] 표기)이므로 최종 완주 후 stale 값이 될 수 있다.

**수정안**: 83.75% 계산 근거를 코드 직접 참조 또는 metadata 계산식으로 명시. 0.944/0.629 수치에 "[STALE: pre-completition snapshot]" 경고 추가.

---

### MINOR

---

**[MIN-001]** CODEBASE_UNDERSTANDING §5.2 — LR schedule 기술 부정확

**위치**: CODEBASE §5.2: "Linear warmup for `warmup_epochs=10` (from near-zero to 1e-3), then cosine annealing for remaining epochs."

**문제**: LinearLR의 `start_factor=1e-4` (`trainer.py:171`)이므로 시작 LR은 `1e-3 × 1e-4 = 1e-7`이지 "near-zero"의 모호한 표현이 아니다. 논문 hyper-parameter 표에서 정확한 시작 LR이 요구될 것이다.

**수정안**: "Linear warmup: 10 epochs, LR=1e-7 → 1e-3 (`start_factor=1e-4`). 이후 CosineAnnealingLR."

---

**[MIN-002]** CODEBASE_UNDERSTANDING §1 Data Flow — pos-enc 공유 여부

**위치**: CODEBASE §1 Data Flow 다이어그램: "teacher path"와 "student path" 모두 `+ decoder_pos_enc`로 표기됨.

**문제**: 코드에서 `self.decoder_pos_encoder`가 하나뿐이고 teacher/student가 공유한다 (`model.py:342–346`, 단일 `PositionalEncoding` 인스턴스). 이것이 의도된 설계인지(shared decoder pos-enc) 기술 누락인지 불명확하며, 논문에서 positional encoding 공유/비공유를 설계 결정으로 기술해야 할 수 있다.

---

**[MIN-003]** RESEARCH_SYNTHESIS 표A — "student decoder hidden state" 출처

**위치**: RESEARCH_SYNTHESIS 표A GRL 행: "Student decoder hidden state 위의 AnomalyClassifierHead"

**문제**: GRL classifier는 `student_hidden` = student decoder의 **마지막 층 출력** 전체 (num_patches × batch × d_model)에 적용된다 (`model.py:1153`). 이것이 모든 패치에 독립적으로 적용되는지, 풀링 후에 적용되는지에 대한 명시가 없다 (정답: 패치별 독립 적용, `squeeze(-1).transpose(0,1)` → `(batch, num_patches, 1)`, `loss.py:282–290`에서 valid mask 적용). 이 세부사항이 논문 method에 필요하다.

---

**[MIN-004]** RESEARCH_SYNTHESIS §⑥ Notion 미해결 차이 — N5 SMD feature 표기

**위치**: RESEARCH_SYNTHESIS §⑥ N5: "SMD features=38 per machine → 29–36 확정"

**문제**: 271_CONFIG_TRUTH §III.3a에서 SMD machine-3-3은 36, machine-3-10은 29로 범위가 29–36이 맞다. 그러나 22/28 entity만 완료되었으므로 "나머지 6 machine의 features 범위"는 아직 미측정이다. "29–36"이 전체 28 machine의 확정 범위인 것처럼 서술하면 오해가 생긴다.

**수정안**: "29–36 (실측 22/28 entity 기준; 잔여 6 machine은 측정 미완)."

---

**[NOTE-001]** RESEARCH_SYNTHESIS §③ 표A — Warmup 학습 곡선 근거의 리뷰 취약성

**위치**: 표A "Teacher-only warmup 250 epochs" 행: "발표 p24 학습 곡선이 warmup 후 pak_auc_f1 +0.1 내외 상승을 보이며 설계 효과를 정성적으로 지지함 (정식 ablation 수치는 미확인 — RISK)."

**NOTE**: "(정식 ablation 수치는 미확인)"으로 스스로 RISK 표기를 했으나, 이것은 reviewer-critical missing detail이다. Teacher-only warmup의 효과를 입증하는 ablation (warmup=0, warmup=50, warmup=250 비교)이 없으면 설계 motivation의 핵심 근거가 없다. Phase 2에서 이 ablation 실험이 반드시 필요함을 보다 강하게 강조해야 한다. 현재 RISK 표기만으로는 방어가 부족하다.

---

**[NOTE-002]** RESEARCH_SYNTHESIS §⑦ 코드 공개 서술 미흡

**위치**: §⑦: "현재 repo 상태: git repo, branch: machineA. public 공개 시점·링크는 미결."

**NOTE**: R25 directive ("코드는 git으로 공개할 예정")에 응하는 서술이나, branch: machineA가 공개될 것인지 (main branch 정리 필요 여부), 공개 전 코드 정리 범위 (예: configs/ 내 실험 결과 JSON 제거, API key 여부) 등의 checklist가 없다. 논문 제출 시 코드가 준비되지 않으면 reproducibility claim 전체가 흔들린다.

---

**[NOTE-003]** 3자 미해결 잔존 모순 — DAGMM 구현 provenance

**위치**: RESEARCH_SYNTHESIS §⑥ Phase 3 판단 사안: "DAGMM 구현 provenance: TranAD repo reimplementation (GMM energy 제거 simplified variant)."

**NOTE**: 이것이 Phase 3으로 미뤄진 것은 맞으나, 만약 최종 비교 테이블에 "DAGMM"으로 표기하면서 GMM energy를 제거했다면 이는 **방법 재정의**이며, 리뷰어가 "이 구현은 DAGMM이 아니다"라고 reject 근거로 쓸 수 있다. 조기에 baseline 구현 정당화 (또는 명칭 변경: "DAGMM-simplified")가 필요하다.

---

**[NOTE-004]** CODEBASE_UNDERSTANDING §4.1 — SMAP/MSL safe-cut "±10" 근거 코드 미확인

**위치**: CODEBASE §4.1 SMAP 행: "test 앞 ~50%(±10 safe-cut)"

**NOTE**: "±10 safe-cut"이 `loaders.py:2592–2595`에 실제로 구현된 것인지 코드 확인이 선행되어야 한다. 271_CONFIG_TRUTH에는 이 언급이 없고, SMAP/MSL entity 5/54밖에 완료되지 않아 실측 `sliding_window_train_ratio`로 검증 불가. Phase 2 SMAP/MSL 완주 시 재확인 필요.

---

## 검증 완료 사항 (PASS)

다음은 코드에서 직접 재확인하여 두 문서의 주장과 일치함을 확인했다:

- exp271 = Set C + override (summary.json `config_set='C'`) — 맞음
- patchify_mode='linear', patch_size=10, num_patches=50 — metadata 전수 일치
- d_model=512, nhead=8, dim_feedforward=2048 — 맞음
- encoder 4층, teacher 3층, student 2층 — 맞음
- mask_after_encoder=True, latent_visible.detach() for student — `model.py:1122–1126` 확인
- GRL 작용 대상 = student decoder (encoder gradient 차단 확인) — 맞음
- grl_disable_anomaly_loss=True → anomaly_loss=0.0 (`loss.py:259–261`) — 확인
- scoring formula: `recon + scaled_disc/4`, `fm_active=False` hardcoded (`scoring.py:237`) — 확인
- teacher_only_warmup_epochs=250, num_epochs=500 — metadata 일치
- resolve_test_stride = W//10-1 = 49 (`utils/experiment.py:16–38`) — 확인
- _compute_warmup_factor: `max(student_start//5, 2) = 50` (`trainer.py:336–348`) — 확인
- AdamW fused, betas=(0.9,0.99), lr=1e-3 (`trainer.py:160–164`) — 확인
- focal BCE: `_p_t = exp(-_bce); _focal = (1-_p_t)^2 × _bce` (`loss.py:337–340`) — 확인 (공식 자체는 확인, BLK-004는 표준 focal과의 차이 지적)
- FM L2 distance: `(teacher_hidden.detach() - student_hidden)^2 / d_model` (loss.py:420) — 확인 (`/ d_model`은 `.mean(dim=-1)`이 d_model 차원 평균이므로 사실상 동일)
- batch_size=1024, dropout=0.15 — metadata 일치
- SWaT num_features=45 (checkpoint `patch_embed.weight=(512,450)`) — reconciler 확인
- WaDi A2 num_features=123 (NaN 4개 drop) — reconciler 확인
