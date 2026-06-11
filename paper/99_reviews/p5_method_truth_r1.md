---
phase: 5
agent: method-truth-auditor
directives: [R17, R5, A8, R24, R27]
last_modified: 2026-06-11
target: paper/05_manuscript/MANUSCRIPT_v2_draft.md (본문 + Appendix 전체)
canon: 271_CONFIG_TRUTH.md (r4, 1순위) / EXPERIMENT_PROTOCOL_TRUTH.md (r3) / RESEARCH_SYNTHESIS.md (r3) / PLACEHOLDER_REGISTRY.md (v2-r2)
verdict: REJECT (BLOCKER 9 / MAJOR 6 / MINOR 8)
---

# P5 Method-Truth Audit — MANUSCRIPT_v2_draft r1

> 모든 판정은 정본 3종 + 코드 직접 재검증(file:line) 기반. 코드 read-only 검증 수행:
> `mae_anomaly/evaluator.py`, `scoring.py`, `loss.py`, `model.py`, `utils/experiment.py`,
> `scripts/run_base_experiments.py`, `comparison/baseline_common.py`.

## 0. 판정 요약

| 등급 | 건수 | 핵심 |
|---|---|---|
| **BLOCKER** | **9** | PA%K 격자 오기(플래그 b), patch-embed LayerNorm 창작, Eq(4) window-mean 오서술, Eq(1) 정규화 누락, w₊ window/patch 오기, baseline batch "512" 미등재·허위, test-time "uniform random masking" 오서술, §4.4 GRL 강건성 메커니즘 오서술, §5 complementary masking 언급(A3) |
| **MAJOR** | **6** | AR-threshold 문헌 관행 인용(Phase 4 검증 전 사용 금지 위반), 0.52–6.20% 범위 SMD-pending 무단서, §A.1 "only F varies" 과대 일반화, y^w GRL 타겟 의미 부정확, notation 충돌군(R5), 미정의 기호군(R5) |
| **MINOR** | **8** | 표기·표현 8건 (아래 §4) |

**선행 플래그 판정**: (a) test stride — **PASS** (본문 제거 확인, Table A.1에만 정본값 49). (b) PA%K 격자 — **본문이 틀렸고 271truth("steps of 1")가 옳다** (코드 실측). 271truth 정정 불요.

---

## 1. 선행 플래그 2건 판정

### 1-(a) Test stride 서술 — PASS

- 본문 전체 grep(`stride`): §4.1.2 본문에 test-stride 주장 **없음** (D-009 수술 주석으로만 흔적 존재). 정본값은 **Table A.1 한 곳**에만: "Train / test stride | 21 / 49" (MANUSCRIPT_v2_draft.md:484).
- 코드 재검증: `mae_anomaly/utils/experiment.py:20–43` `resolve_test_stride` — `sliding_window_test_stride=-1` sentinel → `W // 10 - 1` = 500//10−1 = **49**. 271_CONFIG_TRUTH r4 §VIII 산식·값과 일치. Train stride 21 = `sliding_window_stride=21` (271truth §II) 일치.
- ⚠️ 부수 발견 (정본 측 errata, §5 권고): EXPERIMENT_PROTOCOL_TRUTH r3 §④-실행 2항에 stale "**test stride=1이므로** 한 점을 다수 window가 덮는…" 문장이 잔존 — 1순위 정본(271truth r4 = 49)과 모순. 원고는 정본을 따랐으므로 원고 측 문제 없음.

### 1-(b) PA%K 격자 — 본문 {0, 5, …, 100} 은 오류, 271truth "steps of 1"이 정답 (BLOCKER B-1)

코드 실측 (`mae_anomaly/evaluator.py`):

- **두 개의 격자가 공존**한다:
  1. `PA_K_VALUES = list(range(0, 101, 5))` (`evaluator.py:831`) — **진단용 per-K 키**(`pa_{k}_f1` 등, 소비처 `:854, :950`)와 zero-fill 전용.
  2. `k_values = np.arange(0, 101)` (`evaluator.py:1034`, `compute_pa_k_auc` 내부; docstring `:998` "sweep **K=0,1,...,100** and integrate") — **보고 지표 `pak_auc_f1`/`pak_auc_prc_auc` 등 PA%K-AUC 적분의 실제 격자** (적분: `:1271–1282` `np.trapz(..., k_values)/100.0`).
- 따라서 **헤드라인 지표(PA%K-AUC F1 = 주 지표·선정 기준)의 적분 격자는 step 1 (101점)** 이다. 271_CONFIG_TRUTH r4 §VIII "PA%K sweep from k=0 to k=100 **in steps of 1**" = 정확. **정정 불요.**
- 원고 위반 지점 2곳:
  - §4.1.3 (:362): "integrates point-adjusted F1 over the tolerance spectrum $K \in \{0, 5, \ldots, 100\}$" → **틀림**.
  - §A.2 (:528–529): "For each $K \in \{0, 5, \ldots, 100\}$, the PA%K-adjusted F1 is computed…" (+ :533 "The same $K$-grid"가 오류를 상속) → **틀림**.
- 수정안: $K \in \{0, 1, \ldots, 100\}$ (101점, trapezoidal, [0,1] 정규화). per-K 재최적화·tadpak 준거·정규화 서술 자체는 코드와 일치(`:1003–1005`, 'best' 모드)하므로 격자만 교체.
- 혼동의 추정 근원: EXPERIMENT_PROTOCOL_TRUTH §④ 매핑표 PA F1 행의 "K 그리드 `PA_K_VALUES = 0,5,…,100` `evaluator.py:831`" — 이는 per-K **진단 키** 격자로서는 참이나 AUC 적분 격자로 오독되기 쉬움 → 정본 측 명확화 권고 (§5).

---

## 2. 임무 1 — 방법론 진실 정합 (본문·Appendix 전수 대조)

### 2.1 BLOCKER

**B-1. PA%K-AUC 적분 격자 오기** — §1-(b) 참조. (§4.1.3 :362, §A.2 :528–529·:533)

**B-2. Patch embedding "LayerNorm" 창작 — 비활성 경로 속성의 혼입** (§3.3 :216, Table A.1 :479)
- 원고: $\mathbf{z}_i = \mathrm{LayerNorm}(\mathbf{W}_{\mathrm{emb}}\,\mathrm{vec}(\mathbf{P}_i) + \mathbf{b}_{\mathrm{emb}})$ + Table A.1 "Linear (flatten + projection) **with LayerNorm**".
- 코드: 271 활성 경로(`patchify_mode='linear'`, use_patch=True)의 임베딩은 **bare** `nn.Linear(patch_size×F → d_model)` (`model.py:628` 정의, `:682` 적용) — **LayerNorm 없음**. 프로젝션 뒤 LayerNorm은 **비활성 `patch_cnn` flatten 경로 전용** (`model.py:610–614` `cnn_flatten_proj = Sequential(Linear, LayerNorm)`).
- 271truth r4 §VIII도 "linear (flatten + linear projection; no CNN)"만 기재 — LayerNorm 무근거. CODEBASE_UNDERSTANDING에도 해당 서술 없음(LayerNorm 언급은 init/optimizer 절뿐).
- 판정: 비활성 component 속성 혼입 + 수식-정본 불일치. 수정: 두 곳 모두 LayerNorm 제거.

**B-3. Eq (4) 적응 스케일링의 "window-level means" 오서술** (§3.6 :288–292)
- 원고: "$\bar{r} = (1/N)\sum_j r_j$, $\bar{d} = (1/N)\sum_j d_j$" — **윈도우 내 N패치 평균**으로 명시.
- 코드: 점수 합성은 **전체 평가셋의 (window, patch) patch-score 배열에 대해 단일 호출** — `evaluator.py:2175–2178` (`recon_patches` shape = (n_windows, num_patches) 전체) → `scoring.py:239–241` `recon_mean = float(recon.mean()) + 1e-4` = **개체(평가셋) 전역 평균**. 윈도우별 평균 아님.
- 271truth r4 §VIII 점수식(`recon_mean = mean(recon)+1e-4` …)도 전역 mean 의미.
- 판정: 수식-정본 불일치 (스케일 상수의 정의역이 다름 — 재현 시 다른 점수 산출). 수정: $\bar{r}, \bar{d}$를 "평가 시계열 전체의 patch-score 평균(per entity)"으로 재정의 (Eq 4 자체 형태와 ε=10⁻⁴, Eq 5의 σ=r+d̃/c, c=4는 코드와 일치 확인).

**B-4. Eq (1) $L_{\mathrm{OD}}$ 원소 정규화 1/(s·F) 누락** (§3.5 :255)
- 원고: $L_{\mathrm{OD}} = \frac{1}{|P_n|}\sum_{i\in P_n}\|o^T_i - o^S_i\|^2$.
- 코드: per-patch OD = masked 원소들에 대한 **per-element 평균** — `loss.py:240–244` (`patch_discrepancy_sum / (mask_count × F)`), 그 뒤 normal masked 패치 평균 `loss.py:260` → 실제 $L_{\mathrm{OD}} = \frac{1}{|P_n|}\sum \|o^T_i-o^S_i\|^2/(s\cdot F)$.
- 원고 자체 규약과도 모순: Eq (2)는 $1/(|P_n|\cdot d)$ 포함, §3.6은 $d_i=\|\cdot\|^2/(s\cdot F)$ 포함 — ‖·‖²을 raw 제곱합으로 쓰는 규약 하에서 Eq (1)만 누락. L_recon(per-element MSE, :279 "mean squared")과의 1:1 가중 균형 서술이 (s·F)≈450배 왜곡됨.
- 판정: 수식-정본 불일치. 수정: 분모를 $|P_n|\cdot s\cdot F$로. (teacher detach·정상 패치 한정·anomaly 패치 완전 제외 서술은 코드 일치 확인 — `loss.py:225–261`, `_disc_target = teacher_output.detach()` :220.)

**B-5. Eq C.3 $w_+$ = "normal-to-anomalous **window** ratio" 오기 — 실제는 patch ratio** (§C.1 :677)
- 원고: "$w_+$ the per-entity normal-to-anomalous **window** ratio".
- 코드: `run_base_experiments.py:2578–2586` — train 전 윈도우의 **패치 단위** anomaly 비율을 집계(`_patch_has_anomaly`, `_anomaly_patches/_total_patches`), `_patch_ratio = max(_patch_ratio, 0.001)` 후 `grl_pos_weight = (1 − _patch_ratio)/_patch_ratio`. 271truth §III-3b(r2)도 patch-ratio 하한 유도값으로 명기 (999.0 사례).
- 판정: 사실 오류. 수정: "per-entity normal-to-anomalous **patch** ratio (floor 10⁻³)". (Eq C.3 본체 — $(1-e^{-\ell})^\gamma\ell$, γ=2, masked 패치 평균, pos-weight 내장 BCE, 표준 focal 아님 구분 — 은 `loss.py:325–345` 실측 일치 확인.)

**B-6. Baseline batch size "512" — 미등재·허위 상수 (A8 ③ 해당)** (§4.1.2 :349, Table A.2 :503–505)
- 원고: "batch sizes follow original implementations (**512 for baselines**)" + Table A.2: unsupervised 512 / weak 512.
- 코드: baseline batch는 모델별 이질적 — `comparison/baseline_common.py:272–412`: simple류 512, GCN-LSTM 100, anomaly_transformer/tranad 128, usad 256, gdn 32, omnianomaly 50, tfmae 64, 기타 128…, **weak 4종은 32** (`:333–367`). "원 구현 충실" 원칙(protocol truth §④-실행 3-④)과 "일률 512"는 자기모순이기도 함.
- 정본 무근거: EXPERIMENT_PROTOCOL_TRUTH r3 어디에도 baseline batch 512 없음 (유일한 512-batch 언급은 RESEARCH_SYNTHESIS §⑥-N2의 **MAE Set-C preset** stale 값). ⚠️ PLACEHOLDER_REGISTRY §7.3 Table A.2 행의 "batch 1024/**512**" 출처 표기가 허위 — 레지스트리가 오류의 매개 (§5 errata).
- 판정: 미등재 수치 + 사실 오류 → BLOCKER. 수정: Table A.2 batch 열을 "model-specific (32–512, original presets; Table A.3)"로, §4.1.2 괄호 삭제 또는 "per-model original presets"로 교체. (TAB-A3가 per-baseline 표이므로 자연 수용처.)

**B-7. "Test 시 uniform random masking" 오서술 — 실제 추론은 결정적 leave-one-out** (§3.3 :223, Table A.1 :485)
- 원고: "at test time, masking reverts to uniform random selection" + Table A.1 "uniform random at test time".
- 코드: 추론·per-epoch 평가의 유일 점수 경로는 **leave-one-out** — evaluator가 패치별 명시 마스크를 생성해 `model(expanded, masking_ratio=0.0, mask=masks)`로 전달 (`evaluator.py:1805–1818`; complementary 분기는 비활성 `:1716`). 모델 내부 anomaly-priority/random 선택은 **training 게이트** (`model.py:975–977` `if (self.training and force_mask_anomaly and point_labels is not None)`)라 test에서 아예 미실행 — "uniform random으로 회귀"하는 경로가 실행되지 않는다.
- §3.6의 LOO 서술(정확)과 **내부 모순**이며, Table A.1은 canonical config 표라 재현 오도 위험이 큼.
- 판정: 사실 오류. 수정: "at test time anomaly-priority masking is not applied; windows are scored under the deterministic leave-one-out masking of Section 3.6" 류로 두 곳 모두 교체.

**B-8. §4.4 강건성 property (ii) — GRL 메커니즘 오서술** (:425)
- 원고: "(ii) GRL suppression **activates only for windows containing a labeled anomaly**, so unlabeled anomaly windows **contribute no destabilizing adversarial gradient**".
- 코드: GRL classifier+reversal은 배치 내 **모든 윈도우의 모든 masked 패치**에 적용된다 — 라벨 없는(=정상 취급) 윈도우도 **target=0 negative로 BCE에 참여**하고 reversed gradient를 받는다 (`loss.py:287–345`; balanced_sampling=False → "all patches with pos_weight" `:325–333`). 배치 단위 스킵은 **positive target이 0개일 때뿐** (`loss.py:294–302` `_pos_count == 0 → skip`). 즉 unlabeled anomaly 윈도우는 (배치에 labeled positive가 있는 한) adversarial gradient를 **받는다** — 심지어 잘못된 negative 라벨로.
- 참고: §3.5의 "the GRL term contributes only when the batch contains a positive window"(:279)는 코드와 **일치** (정확). 오류는 §4.4의 윈도우-단위 활성화 주장에 국한.
- 판정: 사실 오류(R32 mandated narrative의 load-bearing 메커니즘 주장). 수정안: "(ii) GRL의 양성 supervision은 labeled 윈도우에서만 발생하며, labeled positive가 전무한 배치에서는 GRL 항 자체가 생략된다; unlabeled anomaly 윈도우는 잘못된 양성 신호를 만들지 않는다(negative로 취급)" — 코드 사실에 맞는 형태로 재정식화 필요 (강건성 논리는 재구성 가능하나 현재 문장은 허위).

**B-9. §5 complementary masking 언급 — A3 미사용 component (271truth §VII #12 명시 제외 항목)** (:458)
- 원고: "an alternative **complementary-masking strategy (implemented but not used in the present experiments)** offers a potential avenue for cost reduction…".
- 정본: 271truth §VII #12 — `eval_complementary_masking=False`, "must not appear in the paper description of config 271". RESEARCH_SYNTHESIS 표B에도 비활성 명시. 코드 실재(`evaluator.py:1737–1745`)·미사용 확인.
- rubric상 미사용 component 언급 = BLOCKER. 추가로 "(implemented but not used)"는 코드 내부 상태 노출(R27). frontmatter·registry에 이 언급을 승인한 directive 근거 없음.
- 수정: 해당 절 삭제(권장) 또는 — future-work 한정 언급을 orchestrator가 명시 승인하는 경우 — "implemented" 등 코드 내부 노출 어구 제거 후 일반론("masking-pattern grouping 등")으로 대체. 판단은 Phase 6/orchestrator 의결 필요.

### 2.2 일치 확인 (대표 항목 — 전수 대조 결과 이상 없음)

- 아키텍처: 4L encoder(pre-LN, 8 heads, ff 2048, GELU, dropout 0.15)/3L Teacher/2L Student(self-attn only, 별도 mask token), d_model=512 고정, stop-gradient(`model.py:1124–1125` `latent_visible.detach()`), GRL head 구조(LayerNorm→512→256→GELU→Dropout 0.1→1, `model.py:181–186`), GRL training-only(`model.py:1150–1154`) — 전부 정본·코드 일치.
- Masking: round(N×r_m)=8/42, 우선순위 π=10³·y+Uniform(0,1) noise·top-k (`model.py:986–996` `anomaly_patches*1000+noise`), Eq C.5 argtopk 의미(초과 시 anomaly 중 균등 부분집합) 일치.
- 손실: Eq (2) FM 1/(|Pₙ|·d) 일치; Eq (3) 합성 = 271truth Total loss 행 일치; Eq C.1 λ_rev sigmoid ramp(p=clip((e−250+1)/250), ≈0.02→≈1) = `trainer.py:1201–1211` 일치; Eq C.2 −λ_rev·I 일치; Eq C.4 β·clip(ratio,0,10)+prev-epoch smoothing = trainer inline grad-ratio 일치(VQGAN 귀속 없음 ✓); 이중 λ 구분(λ_GRL vs λ_rev) 본문·C.1·C.4 모두 유지 ✓; teacher-only warmup 중 student forward skip 서술(§3.4) = r4 정본 일치.
- 점수: Eq (5) c=4, ε=10⁻⁴(`scoring.py:69`), FM 미포함, GRL head 추론 미사용, Eq (6) mean 집계(covering (window,patch) 쌍) = `evaluator.py:2186–2187` method='mean' 일치. LOO "batch 차원 병렬" 일치.
- 프로토콜: 6계열/113 학습·114 평가 단위, 재분할 //2 규칙, SMAP/MSL safe-cut(±10 clearance·무제한 outward·4/81·max+166·합계 252), SWaT dual-eval(83.75%/15.96%/35,900/[2,869,38,769)/19.05→3.68%/잔여 13 region), excl22 독립 best-epoch, Q3 normalonly·boundary-aware windowing, weak 4종 Q1 전용, epoch 비대칭 500/10/50·eval 5/1/1 공개, test-set model selection 공개, AR-threshold (1−r) quantile·strict >, random 5-run mean±std·그 외 단일 run seed 42, SWaT 45=51−6 상수 목록, Table A.4/A.5/C.1 전 수치 — 전부 EXPERIMENT_PROTOCOL_TRUTH r3·271truth r4와 일치.
- 미사용 component grep (dynamic margin/hinge/softplus/Gaussian smoothing/SCAD/discriminator/RevIN/EMA/WDGRL/balanced sampling/masking annealing/minmax clamp/275K/patch_cnn/Simulation/Exathlon/memory bank): **B-9(complementary masking) 외 전부 무검출** ✓. ("cosine annealing"=LR 스케줄 실재 ✓, "no early stopping"=부재 공시 ✓, "ensemble smoothing"은 §4 m-4 참조.)

### 2.3 MAJOR

**M-1. AR-threshold "convention of \cite{xu2022anomalytransformer}" 인용** (§4.1.2 :358)
- EXPERIMENT_PROTOCOL_TRUTH §⑤-4 (r2): 문헌 관행 방어는 코드베이스 내 근거 부재로 **Phase 4 검증 전 사용 금지** 명시. 원고는 검증 전 인용을 사용.
- 내용상 위험도 부가: Anomaly Transformer의 비율 threshold는 **사전 설정 하이퍼파라미터 비율**이고, 본 프로토콜의 r은 **평가셋 ground-truth anomaly 비율** — "동일 관행" 등치는 검증 필요. 원고가 "r derives from evaluation-set ground truth"를 공개한 점은 양호하나 인용 자체의 적합성은 Phase 4 확정 전 미보증. 조치: Phase 4 reference-verifier 큐 등재 또는 인용 보류.

**M-2. §4.1.1 "(ratios 0.52%–6.20%)"에 SMD-pending 단서 누락** (:328)
- §4.1.4(:377)는 "(0.52%–6.20%; **SMD pending**)"으로 단서를 달았으나 §4.1.1 동일 범위 주장에는 없음. SMD 28대의 per-machine train AR은 미측정(registry §7.2) — 측정 후 범위 이탈 시 두 곳 모두 갱신 필요. 동일 단서 병기 권고.

**M-3. §A.1 "all values are shared across the 113 learning units, with only the input dimensionality F varying"** (:473)
- 271truth §III: 가변 키는 **3개** — num_features, **grl_pos_weight**, **sliding_window_train_ratio**. Table A.1 자체에 "w₊ computed per entity"가 있어 내부 모순. "only F varying" → "with the input dimensionality F (Table C.1) and the data-derived w₊ varying per entity" 류로 교정.

**M-4. y^w(윈도우 라벨)와 GRL 타겟의 의미 차** (§3.1 :197, §3.5 :268)
- 원고 정의: y^w = 윈도우 내 **임의 timestep** anomaly. 코드 GRL 타겟: **masked 영역 내** anomaly (`loss.py:166–168` `masked_point_labels = point_labels*(1−mask)`; `:287` window broadcast).
- 271 학습 경로에서는 force_mask_anomaly가 anomaly 패치를 우선 마스킹하므로 두 정의가 **결과적으로 일치**하나, 일반 서술로는 부정확 (예: random masking 조건의 protocol-effect 분석에서는 불일치 가능). 각주 1줄("under anomaly-priority masking the two coincide") 권고.

**M-5 / M-6.** Notation 결함 — §3 (임무 2) 참조.

### 2.4 검증 보류 (Phase 4 이관 — 문헌 사실)

- "7:3 re-split of \cite{wang2025nrdetector}" (:331), "VUS-PR rated the most reliable single TSAD measure \cite{liu2024elephant}" (:362), NRdetector가 "WETAS architecture에서 파생된 pre-trained backbone" 사용 (§2.2 :174) — 정본 3종 범위 밖의 인용 내용 주장. Phase 4 reference-verifier 검증 필수 (현 시점 진위 미판정; m-5/m-6로 등재).

---

## 3. 임무 2 — Notation 검증 (R5)

**수식 (1)–(6) + Appendix C 대 정본 계산 대조**: Eq (2)(3)(5)(6), C.1, C.2, C.4, C.5 **일치**. Eq (1) — B-4 (1/(s·F) 누락). Eq (4) — B-3 (mean 정의역). Eq C.3 — 본체 일치, w₊ 서술 B-5. **λ_GRL/λ_rev 구분은 §3.4·C.1·C.4 전 구간에서 유지됨** (이중 λ 합산 서술 없음, adversarial gradient 곱 구조 −λ_rev·λ_GRL 명시 :269·:688 — r4 정본 정합) ✓. 식 번호 (1)–(6) 연속·중복 없음 ✓.

**M-5. 기호 충돌 (재사용/오버로드)** — 충돌군 6건:
1. $s$ (patch size, §3.1) vs $s_t$ (point score, Eq 6) — Table C.2에 둘 다 등재되어 충돌이 공식화됨.
2. $d$ (embedding dim, §3.3/Eq 2) vs $d_i,\tilde d_i,\bar d$ (discrepancy, §3.6) — 특히 $\bar d$(스칼라)와 $d$(차원)는 bare 충돌. 추가: 같은 양을 $d$ (§3/C.4)와 $d_\text{model}$ (§4.1.2/A.1/C.2)로 이중 표기.
3. $r$ 3중 오버로드: $r_m$ (masking ratio, §3.3), $r_i,\bar r$ (recon error, §3.6), $r$ (anomaly fraction, §4.1.2/§A.2).
4. $p$ 충돌: Eq C.1의 schedule progress vs §4.4/Fig 3의 labeled fraction — 양쪽 모두 공식 변수로 사용.
5. $T$ 충돌: $\mathbf{X}\in\mathbb{R}^{T\times F}$의 timestep 수 vs 위첨자 $T$ (Teacher: $o^T_i, h^T_i, n_T$).
6. $w$ 오버로드: $y^w$의 "window" 태그 vs Eq (6)의 window **index** $w$ ($\sigma^w_i, \mathbf{P}^w_i$ — 인덱스 용법 사전 정의 없음) vs $w_+$ (pos-weight). 부가: $\mathbf{W}$ (window, §3.1) vs $\mathbf{W}_{\mathrm{emb}}$ (§3.3).
- 권고: Phase 6/7에서 일괄 rename (예: masking ratio→$\rho$, labeled fraction→$\phi$ 또는 schedule progress→$\tau$, point score→$a_t$, $d\to d_{\text{model}}$ 통일).

**M-6. 미정의·미등재 기호** — 5건:
1. $L_{\mathrm{main}}$ (Eq C.4) — 정의 없음 (코드상 main loss = recon+OD; 명시 필요).
2. $\hat{y}_i$ (Eq C.3) — classifier 예측 기호 미정의 ($g_\phi$ 출력임을 명시 필요).
3. $\tilde{h}^S_i$ (Eq C.2) — GRL 출력으로 도입되나 정의문 없음.
4. $r_m$ — §3.3 첫 사용에서 "masking ratio"라는 명명 없이 등장; Table C.2 미등재 (§B.4에서야 "masking ratio r_m"으로 호명).
5. $n_e, n_T, n_S$ — §3.4 inline 정의는 있으나 Table C.2 미등재 (값 4/3/2는 §4.1.2에만).
- Table C.2 보강 + 첫 사용 정의 1줄씩.

**첫 사용 전 정의 검사 (전수)**: 위 5건 외 통과 — $y^p_i, y^w$(§3.1), $o/h$(§3.4, Eq 1·2 사용 전), $\varepsilon$(Eq 4 inline), $\pi_i,\eta_i$(§3.3), $\gamma, w_+$(C.3/Table A.1), $e, e_0, e_1$(C.1), $P_n$(§3.5) ✓.

---

## 4. 임무 3 — 수치 전수 색출 (A8)

산문·표의 전 구체 수치 grep 전수 분류 결과:

**① PLACEHOLDER_REGISTRY 등재 placeholder** — NUM-001..031 (31/31), TXT-001/002 (4 occurrences), FIG-1..4+B1, TAB-1/2/3+appendix 8종, ALG-C1, §4.4 "[gradually/monotonically]"(NUM-027), Table A.4 SMD "[per-machine]"(registry §7.2) — **registry v2-r2 스캔과 일치, 고아 토큰 0건** ✓. §5 "approximately 50×"는 registry §5 audit-trail의 해소 항목(N=50 프로토콜 상수 유도 + NUM-031 sync 조건) ✓.

**② 프로토콜 상수 (출처 확인 완료)** — 아래 전부 EXPERIMENT_PROTOCOL_TRUTH r3 / 271truth r4에서 실측 확인:
- 데이터: 113/114, //2 분할, 0.52/0.76/1.63/6.20/0.70/1.70 (train AR), 3.82/3.87/19.05/3.68/30.63/24.54/16.72/4.16 (test AR), 719,959/224,960/1,296,001/86,401/870,972/86,402/176,401/43,921/355,905/217,925/95,271/36,775, 차원 45/123/123/25/29–36/25/55 (51−6, 127−4, 38−const, 1+24, 1+54), safe-cut 표(2,191/1,095/1,261/+166/7.58% 등)·252, SWaT region22(2,869/38,769/35,900/83.75%/15.96%/13 잔여).
- 설정: 500/10/50 epochs, 250 warmup, eval 5/1/1, batch 1024(MAE), seed 42, lr 10⁻³/10⁻⁴, wd 10⁻³, β(0.9,0.99), bf16, L=500/s=10/N=50/8/42/0.15, 21/49 stride, 4/3/2/8/2048/512/0.15(dropout), γ=2, β_GRL=0.2/β_FM=1.0, clip[0,10], 10⁻⁴(ε 2종), 10³(priority), c=4, λ_rev ≈0.02→≈1, p∈{1.0,…,0.1}(R32 설계 입력 §⑦), VUS window 100, 26/22/9/6/7/4(baseline 구성), NRdetector win 100, random 5-run.

**③ 미등재 수치 — 1건 검출 (BLOCKER)**:
- **"512 for baselines"** (§4.1.2 :349 + Table A.2 :504–505) — 정본 3종 어디에도 없고 코드와 모순 (→ B-6). registry §7.3의 허위 출처 표기("batch 1024/512")가 매개.
- (실험 **결과** 수치의 창작은 0건 — 모든 결과 자리는 placeholder 유지 ✓. SWaT 0.944/0.629 등 실측 결과값의 본문 유입 없음 ✓.)
- 부가: "{0, 5, …, 100}" 격자(B-1)는 등재된 상수의 **오용**(진단 격자를 보고-지표 격자로 오기) — ③이 아닌 ② 오류로 분류하나 BLOCKER 동급.

---

## 5. 정본·레지스트리 측 errata 권고 (본 리뷰 쓰기 범위 외 — 후속 phase 처리)

1. **EXPERIMENT_PROTOCOL_TRUTH r3 §④-실행 2항**: "test stride=1이므로" → stale (271truth r4 = 49와 모순; 1순위 정본이 옳음). r4 정정 권고.
2. **EXPERIMENT_PROTOCOL_TRUTH r3 §④ 매핑표 PA F1 행**: "K 그리드 PA_K_VALUES = 0,5,…,100 (evaluator.py:831)" — per-K 진단 키 격자임을 명시하고, PA%K-AUC 적분 격자는 `np.arange(0,101)` step 1 (evaluator.py:1034)임을 병기 권고 (B-1 재발 방지).
3. **PLACEHOLDER_REGISTRY §7.3 Table A.2 행**: "EXPERIMENT_PROTOCOL_TRUTH r3 §④ (… batch 1024/512 …)" — 512는 출처 문서에 없음; B-6 수정과 함께 교정 필요.
4. **271_CONFIG_TRUTH r4**: 정정 불요 (플래그 b 판정 — "steps of 1" 코드 실측 정확).

---

## 6. REQUEST / FEEDBACK

```
REQUEST-1 (Phase 6 의결): B-9 complementary masking — future-work 언급의 허용 여부.
  A3/271truth §VII #12 엄격 적용 시 삭제; 유지하려면 orchestrator 명시 승인 +
  "implemented" 등 코드-내부 노출 어구 제거 필요.
REQUEST-2 (Phase 4): m-5/m-6/M-1의 문헌 사실 3건 (NRdetector 7:3 re-split, liu2024
  VUS-PR 평가, AT 비율-threshold 관행 등치) reference-verifier 검증 큐 등재.
FEEDBACK: 결과 수치 창작 0건·placeholder 체계 무결(A8 ①②) — 결과-측 규율은 양호.
  BLOCKER 9건 중 7건이 §3 수식·메커니즘 서술과 Table A.1/A.2 config 표에 집중 —
  Phase 6 수정 후 method-truth re-audit(r2) 필수.
```

---

## 부록: MINOR 목록

| # | 위치 | 내용 |
|---|---|---|
| m-1 | §A.3 :571 | "SMAP and MSL training labels are explicitly zero" — 라벨은 **코드가 부여한 zeros** (`loaders.py:2602–2604`); 원 배포 데이터 속성으로 오독 여지. "treated as normal (labels set to zero by the loader)" 권고 |
| m-2 | §A.2 :547–548 | PA F1 (oracle) "evaluated at its F1-optimal threshold" — 실제는 **pre-PA(비조정) F1-최적 threshold 선정 후 PA 조정 적용** (protocol truth REQUEST-1 RESOLVED-1, `evaluator.py:929–930, 951–955`); "its"가 PA-후 최적으로 읽힘. 1구 정밀화 |
| m-3 | Table A.1 :487 | "AdamW (fused)" — 구현 플래그 노출(R27-경계). 재현성 목적이면 유지 가능, 본문 톤 통일 차원에서 "fused" 삭제 고려 |
| m-4 | §3.6 :303 | "ensemble smoothing" — Gaussian smoothing(R34) 아님은 확인되나 'smoothing' 어휘 회피 권장 → "ensemble averaging" |
| m-5 | §4.1.1 :331 | NRdetector "7:3 re-split" — 문헌 사실, Phase 4 검증 필요 (§2.4) |
| m-6 | §4.1.3 :362 | liu2024elephant "most reliable single TSAD measure" — 문헌 사실, Phase 4 검증 필요 |
| m-7 | §3.2 :208 | "label-guided module **coupling the decoders** through gradient reversal" — GRL은 Student hidden↔classifier 결합; "decoders 결합"은 부정확. "coupling the Student branch to a window-level anomaly classifier" 권고 |
| m-8 | §3.4 :244 / Eq C.4 | λ 산출은 **배치마다** 계산·누적되고 적용은 직전 epoch 평균 (271truth §VI "computed each batch") — "set adaptively each epoch"는 정본 §VIII 요약과는 일치하나, 정밀 표현은 "applied as the previous epoch's average of per-batch gradient-norm ratios" |
