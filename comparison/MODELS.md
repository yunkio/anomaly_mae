# Baseline Models

> **2026-05-26 QuoVadis 9-baseline line-by-line re-fidelity audit 완료**:
>
> 이전 "paper-faithful" 라벨은 `random`/`sensor_range`에만 line-by-line 검증이 적용되어 있었고, 나머지 7개(`l2_norm`/`nn_distance`/`pca_error` + neural 4) 는 upstream `quovadis_tad/` 코드와 코드 라인 단위로 비교된 적이 없었음. 본 audit에서 docstring 라벨이 아닌 **실제 실행 코드** 기준으로 9개 모두 검증·수정 완료.
>
> - **Upstream commit**: `8e2de5a` (https://github.com/ssarfraz/QuoVadisTAD)
> - **작업 로그**: `temp/quovadis_9baselines_refidelity_0526/` (PLAN.md, REFERENCE_MAP.md, CURRENT_IMPLEMENTATION_AUDIT.md, LINE_BY_LINE_DIFF.md, PIPELINE_COMPAT_REVIEW.md, REVIEW_VERDICT.md, PASS2_REPORT.md, FINAL_REPORT.md)
> - **리뷰 verdict**: Pipeline Compatibility — COMPATIBLE / Independent Review Pass 1 + Pass 2 — ACCEPT
>
> | Baseline | Verdict | 변경 |
> |---|---|---|
> | `random` / `sensor_range` / `l2_norm` / `nn_distance` | 동등 | 변경 없음 |
> | `pca_error` | AGGREGATION FIX | `mean(axis=1)` → `normalise_scores(median-iqr).max(axis=1)` smooth=5 |
> | `mlp` | REIMPL | per-timestep `Linear(F→E)` → GAP → `Linear(E→F)`, no Flatten/ReLU/Dropout |
> | `mlpmixer` | REIMPL | shared LayerNorm + `mlp2` 5-dim bottleneck + 출력 LN 제거 |
> | `transformer` | REIMPL | FFN 단일 `Linear+ReLU`, positional encoding 제거, 출력 LN 제거 |
> | `gcn_lstm` | REMOVE EXTRA | NaN/Inf guard + `clip_grad_norm_` 제거 |
> | `neural_base.py` `predict` | Pass 2 fix | MSE-mean → `|abs|` + median-IQR + smooth(5) + max(axis=1) (`mlp`/`mlpmixer`/`transformer` 공통 영향) |
>
> **Intended exceptions** (paper-faithful 미달 아님): `nn_distance` `batch_size=1000` (메모리 안전 batching, 수학적 동치), `gcn_lstm` 첫 `seq_len` head forward-fill (upstream의 `labels[seq_len:]` truncate를 `(T_test,)` contract 강제 때문에 대체), neural 4종 (`mlp`/`mlpmixer`/`transformer`/`gcn_lstm`) `epochs=50` (Domain B 정책; paper yaml 100-200), `weight_decay` dead-key (upstream `model_def.py:get_model`이 참조 안 함 — upstream도 dead).
>
> 이전 5번 실험(`5_20260525_224237_baseline_minmax_normalonly_segaware`) → paper-faithful 미달로 폐기 (`.trash/0526/results_5_deleted_quovadis_audit/`). 새 6번 실험(`6_20260526_085028_baseline_minmax_normalonly_segaware`) → 본 audit 통과 baseline 코드로 재실행. 자세한 내용은 `GUIDE.md §19` 참조.

> **2026-05-25 paper-faithful 재정합 작업 — 변경 사항**:
> - **9 QuoVadis-paper baseline** (`random`, `sensor_range`, `pca_error`, `l2_norm`, `nn_distance`, `mlp`, `mlpmixer`, `transformer`, `gcn_lstm`) 알고리즘과 hyperparameter 를 ICML 2024 reference (`quovadis_tad/baselines/simple_baselines.py` + `model_configs/*.yaml`) 와 line-by-line 일치시킴.
>   - `random` → binary {0,1} (`np.random.randint(0, 2)`)
>   - `sensor_range` → `sensor_range=(0,1)` 고정 + boolean max
>   - `pca_error` → paper auto branch (`univariate→2, ≤50→10, else→30`) + `svd_solver='full'`
>   - Neural 4: paper yaml (seq_len=5, paper batch/lr/embed/dropout/weight_decay) + epochs=50 (paper 100-200 대신 사용자 변형)
> - **non-self_norm SOTA 6개** (`gcn_lstm`, `tranad`, `usad`, `dagmm`, `gdn`, `omnianomaly`) 의 min-max 후 `[0,1]` clip 제거 → paper-faithful sklearn `MinMaxScaler` 기본 동작. (Note: `dagmm`은 2026-05-25 TranAD-author reimpl 적용 후 **decoder에 Sigmoid** 가 추가되어 출력이 [0,1]로 강제됨. 입력 측 minmax(no-clip) 정책 자체는 동일하나 출력-입력 간 분포 매치 관점에서는 새 impl이 TranAD upstream과 정합. 자세한 내용은 §13 참조.)
> - **DAGMM 구현 reference 교체 (2026-05-25 추가)**: `danieltan07/dagmm` (community reproduction) → `imperial-qore/TranAD/src/models.py::DAGMM` (TranAD-author 변형, TS-AD benchmark de-facto 표준). Sliding window 5 + two-MSE 손실 + AdamW + StepLR(5, 0.9) + epochs=5. 원본 ICLR'18 paper citation은 보존. 영향: 1번/3번/4번 DAGMM 결과는 이전 모델 — 새 결과와 직접 비교 불가. 자세한 내용은 §13 참조.
> - 영향 entries `.trash/0525/results_backup/` 백업 후 재실행. 자세한 내용은 `GUIDE.md §18` 참조.

**22 active baseline models** (5 simple + 3 neural-simple + 7 legacy SOTA + 7 new SOTA 2023-2025) = **총 22개 디렉토리**. All neural models use unified epoch (Neural 4 = **50 epochs** paper-faithful 2026-05-25, SOTA = 10 epochs) with `pak_auc_f1` (PA%K AUC F1, per-K re-optimized; Kim et al. AAAI 2022) best-epoch selection. Single preset (`default`) across all datasets — see `MODEL_PRESETS` in `comparison/baseline_common.py`.

Sources: [QuoVadisTAD](https://arxiv.org/abs/2405.02678) (ICML 2024 Position Paper, 9 baselines — covers all simple/neural-simple + GCN-LSTM), **6 standalone legacy SOTA papers** (2018-2022), and **7 active new SOTA papers (2023-2025)** integrated 2026-05-19: TFMAE (ICDE'24), NPSR (NeurIPS'23), TimesNet (ICLR'23), DCdetector (KDD'23), MEMTO (NeurIPS'23), ModernTCN (ICLR'24 Spot), CATCH (ICLR'25). GCN-LSTM is grouped under **SOTA** in this guide because it uses an internal training loop with per-epoch callback (same execution interface as the other SOTA models), even though QuoVadisTAD provides its configuration.

Datasets covered (9): Simulation, SWaT A1+A2, WaDi A1, WaDi A2, SMD (28 machines), PSM, Exathlon (6 apps {1,2,4,5,6,9}), SMAP (54 channels), MSL (27 channels). (SMAP/MSL = NASA Telemanom, Hundman et al. KDD 2018, 2026-05-26 통합; Pattern A whole + Pattern B per-channel — `docs/DATASET.md` 참조.) All 22 active models are evaluated on each dataset under Q1 (minmax full) and Q3 (minmax normalonly) conditions.

> **Normalization (2026-06-02 per-entity unification):** the minmax/zscore label above denotes the
> *scaler identity*; the *fitting granularity* is now **per source file / entity** for the multi-entity
> concatenated datasets — **SMD (28 machines), SMAP (54 channels), MSL (27 channels), Exathlon (6 apps)**.
> Each entity's scaler is fit on its OWN train slice and transforms its OWN test slice (leak-free), then
> the concat layout is kept. THE PRINCIPLE: per-entity normalization is applied consistently **regardless
> of how a dataset is set up** (smd_concat / smd / smap / msl / exathlon_concat all → per-entity); genuinely
> single-entity datasets (PSM, SWaT A1+A2, WaDi A1/A2, Simulation, *_simple, single-machine SMD) are a
> bit-identical NO-OP (per-entity ≡ whole-array). The segmentation source is the SAME for the comparison
> harness (`comparison/data/unified_loader.get_file_norm_segments` + `comparison/baselines/_per_file_norm.py`)
> and the MAE pipeline (`mae_anomaly/dataset_sliding.SlidingWindowDataset._normalize_per_entity`), both
> reading `data_info['entity_norm_segments']`. **Two intentional exceptions are preserved:** (a) minmax
> uses **clip=False** (paper-faithful sklearn default); (b) per-model scaler identity (npsr =
> MinMaxScaler(−1,1)+clamp, others = StandardScaler, 14 harness-minmax models = minmax). **6 "untouchable"
> SOTA** — timesnet, tfmae, memto, moderntcn, dcdetector, catch — take the raw `none` branch and
> self-normalize **whole-array internally** (their official upstream ships a single pre-concatenated train
> array + one global scaler, often + per-window RevIN); per-entity is NOT applied to them, by design.

> **Results invalidation (2026-06-02 — re-run NOT yet performed):** prior numbers on the **multi-entity
> concat datasets — SMD / MSL / SMAP / Exathlon** are invalidated for all models EXCEPT the 6 untouchable
> (timesnet, tfmae, memto, moderntcn, dcdetector, catch), due to the per-entity normalization change.
> Additionally, **moderntcn & nrdetector** are invalidated on **ALL** datasets (score-formula fix; for
> nrdetector the 2026-06-03 (PM) correction-of-the-correction RESTORES the **classifier-gated** continuous
> actmap = `actmap × [seg_prob ≥ mean(seg_prob)]` — upstream `rank_test` ranks the actmap of
> classifier-flagged windows, so the gate is integral. The brief no-gate variant `5cff9da` was a regression
> that FROZE per-epoch scores (encoder is frozen after Stage 0; only the PU-classifier trains) and collapsed
> ranking (SWaT pak_auc_f1 0.86→0.44); it is REVERTED. Prior no-gate scores are STALE → must be
> re-scored), and
> **wetas / treemil / deepmil** on single-file datasets too (test-side leak fix). **Unaffected (no re-run
> needed):** the 6 untouchable on any dataset; single-file datasets (PSM / SWaT / WaDi / Simulation) for
> the 16 non-leak-fixed models.

---

## Simple Baselines

These models require no neural network training and serve as sanity checks.

### 1. Random Baseline

**File:** `baselines/random/model.py`

**Description:** Generates uniform random anomaly scores in [0, 1].

**Purpose:** Lower bound for any anomaly detection method. If a method doesn't beat random, it's not learning anything useful.

**Configuration:**
- `seed`: Random seed for reproducibility (default: 42)

**Usage:**
```python
from comparison.baselines import RandomBaseline
model = RandomBaseline(seed=42)
model.fit(train_data)  # No-op
scores = model.predict(test_data)
```

---

### 2. Sensor Range Deviation

**File:** `baselines/sensor_range/model.py`

**Description:** Measures deviation from the min/max range observed in training data.

**Method:**
1. Learn min/max for each feature from training data
2. For each test point, compute deviation from learned range
3. Anomaly score = sum or count of out-of-range features

**Configuration:**
- `count_sensors`: If True, count out-of-range sensors; if False, sum deviations

**From QuoVadisTAD:** "A simple baseline that checks whether sensor readings fall outside the expected range. Surprisingly effective on some datasets."

---

### 3. PCA Error (PCA Reconstruction Error)

**File:** `baselines/pca_error/model.py`

**Upstream reference:** `quovadis_tad/baselines/simple_baselines.py:119-143` (auto-dim branch + `transform` returning `np.abs(x - reconstructed)`) + `quovadis_tad/dataset_utils/data_utils.py:47-72` (`normalise_scores(norm='median-iqr', smooth=True, smooth_window=5)`) + `quovadis_tad/evaluation/evaluation_utils.py:21` (`.max(axis=1)` over sensors).

**Our implementation:** `comparison/baselines/pca_error/model.py:31-54` (`_median_iqr_smooth` helper, ports `data_utils.py:55-69`) + `:107-124` (`predict`).

**Architecture/algorithm:** PCA `svd_solver='full'`, auto dim `univariate→2, n_feat≤50→10, else→30` with cap `min(max(2, n//5), n)` (paper-faithful). Per-feature absolute reconstruction error `|x - x_recon|` (shape `(n, f)`).

**Aggregation (2026-05-26 fix):** per-feature absolute error → median-IQR per-sensor normalize (`epsilon=1e-2`) → 5-box smoothing (first 5 timesteps = 0) → `max(axis=1)` over sensors → `(T_test,) float32`. 이전 구현은 `mean(axis=1)`이었으나 upstream evaluation path와 다름 → 본 audit에서 paper-faithful로 교체.

**Intentional exception:** 없음. (단 `__init__`의 `int(pca_dim)` cast는 upstream에 없지만 preset의 `'auto'` 경로에서는 미발동 — 무의미한 textual divergence.)

---

### 4. L2-Norm

**File:** `baselines/l2_norm/model.py`

**Upstream reference:** `quovadis_tad/baselines/simple_baselines.py:51-68` (`LNorm` class with `transform = np.linalg.norm(x, ord=self.ord, axis=1)`).

**Our implementation:** `comparison/baselines/l2_norm/model.py:69` (`np.linalg.norm(test_data, ord=self.ord, axis=1)`).

**Architecture/algorithm:** Per-timestep L2 norm of feature vector. `ord=2` (preset), `axis=1` reduce → scalar score per timestep.

**Aggregation:** None (norm already returns scalar per timestep).

**Intentional exception:** 없음. (`normalize=False` 옵션 분기는 upstream에 없는 dead code이나 preset에서 미전달 → 미발동.)

---

### 5. 1-NN Distance (Nearest Neighbor Distance)

**File:** `baselines/nn_distance/model.py`

**Upstream reference:** `quovadis_tad/baselines/simple_baselines.py:71-95` (`NNDistance` class with `pairwise_distances(x, self.train_data, metric=self.distance).min(axis=1)`).

**Our implementation:** `comparison/baselines/nn_distance/model.py:89-101` (batched `pairwise_distances + min(axis=1)`).

**Architecture/algorithm:** Each test row의 nearest-neighbor 거리 = `pairwise_distances(x, train).min(axis=1)`. `metric='euclidean'`.

**Aggregation:** None (`.min(axis=1)`이 이미 scalar per timestep).

**Intentional exception:** `batch_size=1000` — 파이프라인 메모리 안전 batching. 각 test row의 NN 거리는 다른 row와 독립적이므로 batched와 unbatched가 수학적으로 정확히 동일. `nn_distance/model.py:64-77` docstring에 사유 명시.

---

## Neural Baselines (QuoVadisTAD)

These are minimal neural network architectures designed to be simple yet competitive.

### 6. 1-Layer MLP

**File:** `baselines/mlp/model.py`

**Upstream reference:** `quovadis_tad/model_utils/model_def.py:204-241` (`1_Layer_MLP` branch). Key lines: `:219` per-timestep `Dense(units=embedding_dim)`, `:227-228` `if model == "1_Layer_MLP": embedding = x` (skips Dropout-only blocks), `:234` `GlobalAveragePooling1D`, `:236` final `Dense(input_shape[1])`. Scoring path: `model_def.py:483` `np.abs(predictions - orig_target)` + `data_utils.py:47-72` `normalise_scores` + `evaluation_utils.py:21` `.max(1)`.

**Our implementation:** `comparison/baselines/mlp/model.py:38-51` (`MLPModel`) + scoring via `comparison/baselines/neural_base.py:215-347` (`NeuralBaselineBase.predict`).

**Architecture/algorithm (2026-05-26 REIMPL):**
```
Input: (B, seq_len, n_features)
  → embed = Linear(n_features → embedding_dim)   # per-timestep Dense
  → h = h.mean(dim=1)                             # GAP over time → (B, embedding_dim)
  → output = Linear(embedding_dim → n_features)   # (B, n_features)
```
이전 구현의 `Flatten`/`ReLU`/`Dropout`은 upstream `1_Layer_MLP` branch에 없음 → 본 audit에서 모두 제거.

**Configuration** (`MODEL_PRESETS['default']['mlp']`):
| Parameter | Value | Notes |
|-----------|-------|-------|
| seq_len | 5 | upstream yaml `MLP_1_layer_embedd_32_seq_5.yaml` |
| embedding_dim | 32 | upstream yaml |
| dropout | 0.0 | preset; `MLPModel`에서는 무시 (paper-faithful) |
| epochs | 50 | **intentional exception**: paper yaml=200, Domain B 정책 |
| lr / batch_size | 0.001 / 512 | upstream yaml |
| weight_decay | 1e-4 | preset; upstream code에서도 dead-key (`model_def.py:get_model`이 참조 안 함) |

**Anomaly Score (2026-05-26 Pass 2 fix):** per-window `|outputs - targets|` 절대 잔차 (`(n_windows, n_features)`) → median-IQR per-sensor normalize + 5-box smooth → `max(axis=-1)` over sensors → first `seq_len` timesteps forward-fill. 이전은 `np.mean((outputs - targets)**2, axis=1)` (MSE-mean) → upstream `model_def.py:483`과 일치하지 않음 → 본 audit Pass 2에서 paper-faithful로 교체.

**Intentional exception:** `epochs=50` (Domain B 정책), `weight_decay` dead-key (upstream도 dead), head forward-fill (pipeline `(T_test,)` contract 강제).

---

### 7. Single Block MLPMixer

**File:** `baselines/mlpmixer/model.py`

**Upstream reference:** `quovadis_tad/model_utils/model_def.py:77-114` (`MLPMixerLayer`) + `:181-241` (`build_sequence_embedder`). Key lines: `:95` `self.normalize = layers.LayerNormalization(epsilon=1e-6)` (단일 인스턴스), `:99` + `:109` (두 번 호출, γ/β 공유), `:81-86` `mlp1` Sequential (`Dense(num_patches, gelu) → Dense(num_patches) → Dropout`), `:88-93` `mlp2` Sequential (`Dense(num_patches, gelu) → Dense(hidden_units) → Dropout` — 5-dim bottleneck). `build_sequence_embedder` 출력 LN 없음.

**Our implementation:** `comparison/baselines/mlpmixer/model.py:55-91` (`MLPMixerBlock`) + `:94-116` (`MLPMixerModel`).

**Architecture/algorithm (2026-05-26 REIMPL):**
```
Input: (B, seq_len, n_features)
  → embedding = Linear(n_features → embedding_dim)         # (B, seq_len, E)
  → MLPMixerBlock:
      norm → transpose → mlp1 → transpose → skip
      norm → mlp2 → skip                                    # 동일 normalize 재호출
  → x.mean(dim=1)                                            # GAP over time
  → output = Linear(embedding_dim → n_features)
```
- `mlp1`: `Linear(seq_len, seq_len) → GELU → Linear(seq_len, seq_len) → Dropout` (no intermediate Dropout).
- `mlp2`: `Linear(embedding_dim, seq_len) → GELU → Linear(seq_len, embedding_dim) → Dropout` (★ 5-dim bottleneck).
- `LayerNorm(embedding_dim, eps=1e-6)` 단일 인스턴스 2회 호출.
- 출력측 추가 LayerNorm 없음 (upstream `build_sequence_embedder`도 없음).

이전 구현은 `norm1`/`norm2` 별도 LN + `mlp2` intermediate-dim `embedding_dim` (no bottleneck) + intermediate Dropout + 출력측 추가 LN → 모두 upstream과 불일치 → 본 audit에서 교체.

**Configuration** (`MODEL_PRESETS['default']['mlpmixer']`):
| Parameter | Value | Notes |
|-----------|-------|-------|
| seq_len | 5 | upstream yaml `MLPMixer_blocks_1_embedd_128_seq_5.yaml` |
| embedding_dim | 128 | upstream yaml |
| num_blocks | 1 | upstream yaml |
| dropout | 0.1 | upstream yaml |
| epochs | 50 | **intentional exception**: paper yaml=100, Domain B 정책 |
| lr / batch_size | 2e-4 / 512 | upstream yaml |
| weight_decay | 1e-4 | preset; upstream code에서도 dead-key |

**Anomaly Score:** MLP와 동일 (Pass 2 fix via `NeuralBaselineBase.predict`).

**Intentional exception:** `epochs=50` (Domain B 정책), `weight_decay` dead-key, head forward-fill.

---

### 8. Single Transformer Block

**File:** `baselines/transformer/model.py`

**Upstream reference:** `quovadis_tad/model_utils/model_def.py:117-133` (`TransformerBlock`) + `:181-241` (`build_sequence_embedder`, `Single_Transformer_block` branch). Key lines: `:121` `self.ffn = layers.Dense(ff_dim, activation="relu")` (단일 Dense with ReLU inside), `:122-123` `LayerNormalization(epsilon=1e-6)` (eps=1e-6), `:130 + :133` Post-Norm `LN(x + dropout(sublayer))`. `:195` `TransformerBlock(embedding_dim, MHA_blocks, embedding_dim, dropout_rate)` — `ff_dim = embedding_dim`. yaml `positional_encoding: False`.

**Our implementation:** `comparison/baselines/transformer/model.py:52-86` (`TransformerBlock`) + `:91-108` (`TransformerModel`).

**Architecture/algorithm (2026-05-26 REIMPL):**
```
Input: (B, seq_len, n_features)
  → embedding = Linear(n_features → embedding_dim)
  → TransformerBlock:
      MHA(num_heads, key_dim=embedding_dim, batch_first=True) → dropout1 → norm1(x + ·)   # Post-Norm
      ffn = Linear(embedding_dim → embedding_dim) + ReLU → dropout2 → norm2(out1 + ·)     # Post-Norm
  → x.mean(dim=1)                                            # GAP over time
  → output = Linear(embedding_dim → n_features)
```
- FFN: **단일 `Linear+ReLU`** (not 2-layer Linear→ReLU→Dropout→Linear).
- Positional encoding 없음.
- 출력측 LayerNorm 없음.
- LayerNorm `eps=1e-6`.

이전 구현은 2-layer FFN + learned positional embedding + 출력측 LayerNorm 추가 → 모두 upstream에 없음 → 본 audit에서 제거.

**Configuration** (`MODEL_PRESETS['default']['transformer']`):
| Parameter | Value | Notes |
|-----------|-------|-------|
| seq_len | 5 | upstream yaml `Transformer_blocks_1_1_embedd_128_seq_5.yaml` |
| embedding_dim | 128 | upstream yaml |
| num_heads | 1 | upstream yaml |
| num_blocks | 1 | upstream yaml |
| dropout | 0.1 | upstream yaml |
| epochs | 50 | **intentional exception**: paper yaml=100, Domain B 정책 |
| lr / batch_size | 1e-3 / 512 | upstream yaml |
| weight_decay | 1e-4 | preset; upstream code에서도 dead-key |

**Anomaly Score:** MLP와 동일 (Pass 2 fix via `NeuralBaselineBase.predict`).

**Intentional exception:** `epochs=50` (Domain B 정책), `weight_decay` dead-key, head forward-fill.

---

## SOTA Comparison

### 9. 1-Layer GCN-LSTM (from QuoVadisTAD)

**File:** `baselines/gcn_lstm/model.py`

**Upstream reference:** `quovadis_tad/model_utils/gnn.py:23-39` (FINCH 1-NN adj) + `:88-170` (`GraphConv` with `aggregation_type='max'` + `combination_type='concat'` + ReLU) + `:200` LSTM + `:202` Dense(output_seq_len, relu) + `quovadis_tad/model_utils/model_def.py:171` extra Dense(num_nodes) + `model_def.py:425-426` label truncation + `model_def.py:483` `np.abs(predictions - orig_target)` + `data_utils.py:47-72` median-IQR smooth + `evaluation_utils.py:21` max(axis=1). yaml `GCN_LSTM_block_1_embedd_10_seq_5.yaml`.

**Our implementation:** `comparison/baselines/gcn_lstm/model.py:49-85` (FINCH adj), `:108-229` (GraphConv), `:281-291` (LSTM + Dense), `:294` (extra node-axis Dense), `:586-622` (`_normalise_scores`), `:624-728` (`predict`).

**Architecture/algorithm:** FINCH 1-NN adjacency (cosine pairwise + 1-NN + identity + `A @ A.T → sign()` symmetric + diag-zero) → GraphConv (max aggregation via `scatter_reduce(reduce='amax', include_self=False)` + concat + Glorot uniform init + ReLU) → LSTM(64) → Dense(seq_len) + ReLU → extra Dense(n_features). 알고리즘 자체는 upstream과 line-by-line 일치.

**2026-05-26 변경 (REMOVE EXTRA):** `fit()` 학습 step에서 다음 두 paper-non-faithful 안정화 장치 제거:
- NaN/Inf guard `if not (torch.isnan(loss) or torch.isinf(loss)): ...` (이전 line 549).
- Gradient clipping `clip_grad_norm_(..., max_norm=1.0)` (이전 line 551).

이후 학습 loop는 표준 `loss.backward(); optimizer.step()` (upstream과 일치).

**Aggregation:** per-window per-sensor `|pred - target|` (`gcn_lstm/model.py:671`) → `_normalise_scores` (median-IQR, smooth=True, smooth_window=5, epsilon=1e-2; `:688-694`) → `max(axis=-1)` over sensors (`:697`) → first `seq_len` timesteps forward-fill (`:715`). 이미 paper-faithful (Pass 2 영향 없음).

**Configuration** (`MODEL_PRESETS['default']['gcn_lstm']`):
| Parameter | Value | Notes |
|-----------|-------|-------|
| seq_len | 5 | upstream yaml |
| gcn_out_dim | 10 | upstream yaml |
| lstm_units | 64 | upstream yaml |
| dropout | 0.1 | upstream yaml |
| epochs | 50 | **intentional exception**: paper yaml=100, Domain B 정책 |
| lr / batch_size | 1e-3 / 100 | upstream yaml |

**Intentional exception:**
- `epochs=50` (Domain B 정책).
- Head forward-fill — upstream `model_def.py:425-426`의 `gt_labels = labels[input_sequence_length:]` truncation을 pipeline `(T_test,)` contract 강제 때문에 forward-fill로 대체.

---

### 10. Anomaly Transformer (ICLR 2022)

**File:** `baselines/anomaly_transformer/`

**Paper:** [Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy](https://arxiv.org/abs/2110.02642)

**Description:** Uses association discrepancy to distinguish anomalies. Normal points associate with adjacent timestamps (high prior-association), while anomalies associate with distant points (low prior-association).

**Key Concept - Association Discrepancy:**
- **Prior-Association:** Expected association with adjacent points (learned Gaussian prior)
- **Series-Association:** Actual association learned by self-attention
- **Anomaly Score:** KL divergence between prior and series associations + reconstruction error

**Architecture:**
```
Input: (win_size × n_features)
  → Embedding
  → N Transformer Encoder Layers with:
      - Anomaly Attention (computes both prior and series associations)
      - Feed-Forward Network
  → Output projection
```

**Configuration** (actual `MODEL_PRESETS['default']` in `baseline_common.py`):
| Parameter | Value |
|-----------|-------|
| win_size | 100 |
| d_model | 512 |
| n_heads | 8 |
| e_layers | 3 |
| d_ff | 512 |
| dropout | 0.0 |
| epochs | 10 |
| batch_size | 128 |
| train_stride | 1 |
| k | 3.0 |
| temperature | 50.0 |

**Code attribution** (from `comparison/baselines/anomaly_transformer/model.py` docstring):
"Original code: https://github.com/thuml/Anomaly-Transformer (Apache 2.0 License per docstring; GitHub LICENSE file: MIT). Modified for device-agnostic operation and integration with comparison framework."

**Minimax Training Strategy:**
1. Minimize association discrepancy (make series-association match prior)
2. Maximize association discrepancy for anomaly detection
3. Alternates between these objectives during training

**From the paper:** "The association discrepancy provides an inherent distinguishable criterion for anomaly detection, which makes Anomaly Transformer achieve state-of-the-art performance."

---

### 11. TranAD (VLDB 2022)

**File:** `baselines/tranad/model.py`

**Paper:** [TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data](https://arxiv.org/abs/2201.07284)

**Description:** Transformer-based model with two-phase reconstruction and adversarial training. Phase 1 reconstructs without conditioning, Phase 2 uses Phase 1's reconstruction error as conditioning.

**Architecture:**
```
Input: (seq_len × n_features)
  → Positional Encoding
  → Transformer Encoder
  → Two Transformer Decoders:
      - Decoder 1: Standard reconstruction
      - Decoder 2: Conditioned on Decoder 1's error
  → FCN (Sigmoid)
```

**Key Innovation:**
- Two-phase adversarial training: Phase 2 learns to "focus" on errors from Phase 1
- Conditioning helps the model learn more robust representations

**Configuration** (actual `MODEL_PRESETS['default']`):
| Parameter | Value |
|-----------|-------|
| seq_len | 10 |
| d_ff | 16 |
| n_encoder_layers | 1 |
| n_decoder_layers | 1 |
| dropout | 0.1 |
| epochs | 10 |
| batch_size | 128 |
| train_stride | 1 |

**Reference (official code from paper authors):** https://github.com/imperial-qore/TranAD (BSD-3-Clause)

---

### 12. USAD (KDD 2020)

**File:** `baselines/usad/model.py`

**Paper:** [USAD: UnSupervised Anomaly Detection on Multivariate Time Series](https://dl.acm.org/doi/10.1145/3394486.3403392)

**Description:** UnSupervised Anomaly Detection using dual-decoder autoencoders with adversarial training.

**Architecture:**
```
Input: Flattened window (seq_len × n_features)
  → Shared Encoder → z
  → Decoder 1: D1(z) → w1
  → Decoder 2: D2(z) → w2
  → Decoder 2 on Encoder(w1): D2(E(w1)) → w3
```

**Training:**
- Loss1 = α||X - w1||² + β||X - w3||²
- Loss2 = α||X - w2||² - β||X - w3||²
- α increases, β decreases during training (adversarial schedule)

**Anomaly Score:** α||X - w1||² + β||X - w3||²

**Configuration** (actual `MODEL_PRESETS['default']`):
| Parameter | Value |
|-----------|-------|
| seq_len | 5 |
| hidden_dims | [input_dim//2, input_dim//4] |
| latent_dim | 32 |
| alpha | 0.5 |
| beta | 0.5 |
| epochs | 10 |
| batch_size | 256 |
| train_stride | 1 |

**Reference (semi-official):** https://github.com/manigalati/usad
README states: "Implementation by: Francesco Galati. Additional contributions: Julien Audibert, Maria A. Zuluaga [paper authors]. Copyright 2020 Eurecom. Released under BSD-3 license." → 원본 저자(EURECOM)의 기여 + 저작권 명시된 반공식 구현. 별도 조직 계정 [robustml-eurecom/usad](https://github.com/robustml-eurecom/usad)도 존재 (★21, 동일 소스).

---

### 13. DAGMM (ICLR 2018) — TranAD-author 구현 (2026-05-25 reimpl)

**File:** `baselines/dagmm/model.py`

**Paper (model citation, 유지):** Zong et al., [Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection](https://openreview.net/forum?id=BJJLHbb0-), ICLR 2018.

**Implementation reference (변경):** Tuli et al., TranAD (VLDB 2022) — [imperial-qore/TranAD](https://github.com/imperial-qore/TranAD/blob/main/src/models.py) (DAGMM class, lines 81-123) + `main.py::backprop` DAGMM branch (lines 79-104) + `main.py::convert_to_windows` (lines 18-24) + `main.py::load_model` (lines 60-61, optimizer/scheduler).

> **2026-05-25 변경 이력**: 기존 구현은 [danieltan07/dagmm](https://github.com/danieltan07/dagmm) (community reproduction)을 따른 row-by-row + GMM energy/cov 손실 형태였음. **DAGMM 원본 저자 (NEC Labs America)는 공식 PyTorch 코드를 공개하지 않음**. TS-AD benchmark 도메인에서는 TranAD 저자가 자체 reimplement한 변형 (시계열 sliding window + two-MSE 손실)이 사실상 표준으로 인용됨 (CARLA, MEMTO, DCdetector 등 후속 paper 다수 인용). 본 baseline의 구현 reference를 TranAD-author 변형으로 교체했음. 원본 paper citation은 보존.

**Description:**
원본 DAGMM은 KDD CUP'99 같은 비-시계열 record-level 이상탐지를 위한 모델이지만, TranAD 저자는 시계열용으로 두 가지 핵심 변형을 추가했음: (1) 5-step sliding window를 flatten하여 입력, (2) GMM energy/cov 손실을 제거하고 단순 two-MSE 손실 (`MSE(x_hat, x) + MSE(gamma, x)`) 로 학습.

**Architecture (TranAD-faithful):**
```
Input: data[i-5:i] flattened → (5F,)            # left-padded for i < 5
  → Encoder: Linear(5F, 16) → Tanh → Linear(16, 16) → Tanh → Linear(16, 8)
  → Decoder: Linear(8, 16) → Tanh → Linear(16, 16) → Tanh → Linear(16, 5F) → Sigmoid
  → Reconstruction features z_r = (relative_euclidean, cosine_similarity)
  → Full latent: z = cat([z_c (8), z_r (2)], dim=1) → (10,)
  → Estimate: Linear(10, 16) → Tanh → Dropout(0.5) → Linear(16, 5F) → Softmax(dim=1) → gamma (5F,)
```

**Anomaly Score (TranAD-faithful):** 각 timestep `t` 에서, window의 last row (=현재 timestep의 row) 에 대한 reconstruction error `(x_hat - x)^2` 의 feature-mean. **GMM energy 사용 안 함** — TranAD 저자가 의도적으로 단순화.

**Loss (TranAD-faithful):** `mean(MSE(x_hat, x)) + mean(MSE(gamma, x))`. **GMM energy / covariance regularization 없음** — TranAD `main.py:79-95`에 `ComputeLoss(...)`가 import만 되고 실제로는 사용되지 않음 (dead code). 본 구현도 동일하게 사용하지 않음.

**Configuration** (`MODEL_PRESETS['default']['dagmm']`):
| Parameter | Value | 출처 |
|-----------|-------|------|
| n_window | 5 | TranAD `src/models.py:90` |
| n_hidden | 16 | TranAD `src/models.py:88` |
| n_latent | 8 | TranAD `src/models.py:89` |
| lr | 1e-4 | TranAD `src/models.py:85` |
| weight_decay | 1e-5 | TranAD `main.py:60` (AdamW) |
| lr_step_size | 5 | TranAD `main.py:61` (StepLR) |
| lr_gamma | 0.9 | TranAD `main.py:61` |
| epochs | 5 | TranAD `main.py:310` (`num_epochs = 5`) |
| batch_size | 256 | pipeline default (TranAD 자체는 per-sample, but our pipeline은 batched) |
| n_gmm | derived (= n_feats × n_window) | TranAD `src/models.py:92` |

**Pipeline 호환성:** 기존 wrapper API (`fit(train_X, epoch_callback=None, train_segments=None) -> self`, `predict(test_X) -> (T,) float32`) 동일 유지. **현재 `MODEL_PRESETS['default']['dagmm']`은 새 TranAD-author 키만 사용** — legacy 키는 포함되어 있지 않음. 단 외부 호출자가 옛 키 (`seq_len`, `latent_dim`, `n_gmm`, `lambda_energy`, `lambda_cov`, `hidden_dims`) 로 `DAGMMBaseline`을 직접 인스턴스화하는 경우를 대비해 wrapper `__init__`에서 자동 흡수 (별도 import 경로 / 사용자 코드 호환성 목적). `seq_len`은 `n_window` alias로 해석됨.

**Train/Test 비교 가능성:**
- 1번 / 3번 / 4번 실험의 DAGMM 결과는 **이전 구현 (danieltan07 reimpl, energy-based row-by-row)** 결과 — 새 구현과 직접 비교 불가능.
- 6번 (segment-aware 적용 후) 부터 새 구현 사용. dagmm 결과는 dispatch 진행에 따라 자동 생성됨.

---

### 14. GDN (AAAI 2021)

**File:** `baselines/gdn/model.py`

**Paper:** [Graph Neural Network-Based Anomaly Detection in Multivariate Time Series](https://arxiv.org/abs/2106.06947)

**Description:** Graph Deviation Network that models inter-sensor dependencies as a learnable graph structure.

**Architecture:**
```
Input: (seq_len × n_features)
  → Node Embeddings (learnable per feature)
  → Input Projection (seq_len → embed_dim)
  → Graph Construction (cosine similarity + top-k)
  → Graph Attention Layer (GAT)
  → Batch Norm + ReLU
  → Concatenate with Node Embeddings
  → Output MLP
```

**Key Innovation:**
- Learnable graph structure based on node embedding similarity
- Uses top-k neighbors for sparse graph
- Captures inter-sensor dependencies

**Configuration** (actual `MODEL_PRESETS['default']`):
| Parameter | Value |
|-----------|-------|
| seq_len | 5 |
| embed_dim | 64 |
| gnn_out_dim | 64 |
| n_heads | 1 |
| top_k | 5 |
| dropout | 0.2 |
| epochs | 10 |
| batch_size | 256 |
| train_stride | 1 |

**Reference (official code, 제1저자):** https://github.com/d-ailin/GDN (MIT, ★602). `d-ailin` = Deng Ailin (NUS), GDN 논문 제1저자.

---

### 15. OmniAnomaly (KDD 2019)

**File:** `baselines/omnianomaly/model.py`

**Paper:** [Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network](https://dl.acm.org/doi/10.1145/3292500.3330672)

**Description:** Stochastic RNN-based VAE for time series anomaly detection with uncertainty estimation.

**Architecture:**
```
Input: (seq_len × n_features)
  → RNN Encoder (GRU/LSTM)
  → Latent distribution: q(z|x) = N(μ, σ²)
  → Reparameterization: z ~ q(z|x)
  → RNN Decoder (GRU/LSTM)
  → Reconstruction distribution: p(x|z) = N(μ_x, σ_x²)
```

**Loss:** ELBO = Recon_loss + β·KL(q(z|x) || p(z))

**Anomaly Score:** Negative log-likelihood (Monte Carlo estimation)

**Configuration** (actual `MODEL_PRESETS['default']`):
| Parameter | Value |
|-----------|-------|
| seq_len | 100 |
| hidden_dim | 100 |
| z_dim | 3 |
| n_layers | 1 |
| rnn_type | gru |
| beta | 0.01 |
| n_mc_samples | 10 |
| epochs | 10 |
| batch_size | 256 |
| train_stride | 1 |

**Reference (official code, 저자 소속 lab):** https://github.com/NetManAIOps/OmniAnomaly (MIT, ★923). `NetManAIOps` = 칭화대 NetMan AIOps 연구실 (Su Ya et al. 소속).

---

## New SOTA (2026-05-19 batch, 7-model active set)

The 7 new SOTA models below (sections 16–22) cover 2023–2025 frontier work in MTS anomaly detection. All share the same execution interface (`fit(train_X, epoch_callback) / predict(test_X)`) and are launched via `run_sota_baseline_with_epoch_eval`. Hyperparameter presets live in `MODEL_PRESETS['default']` (single preset across all datasets). Verified by smoke-test: each builds, accepts an `(N, win_size, M)` input on GPU, and returns a (N_test,) score.

| # | Model | Venue | File | Loss | Distinct Feature |
|---|-------|-------|------|------|-------------------|
| 16 | TimesNet | ICLR'23 | timesnet/ | MSE recon | FFT-top-k period → 2D Inception conv |
| 17 | TFMAE | ICDE'24 | tfmae/ | KL adv-con | Dual temporal+frequency MAE |
| 18 | DCdetector | KDD'23 | dcdetector/ | Symmetric KL | Patch + In-patch dual attention |
| 19 | MEMTO | NeurIPS'23 | memto/ | Recon+Gather+Entropy | Memory module w/ K-means init (2-phase) |
| 20 | ModernTCN | ICLR'24 Spot | moderntcn/ | MSE recon | Large-kernel DW conv + dual ConvFFN |
| 21 | CATCH | ICLR'25 | catch/ | Time+Freq+ChDisc | Channel masking + freq reconstruction |
| 22 | NPSR | NeurIPS'23 | npsr/ | Point+Induction MSE | Nominality-conditioned scoring |

Notes:
- NPSR: `performer-pytorch` (with `--no-deps` install, 2026-05-22) provides the canonical Performer attention path. Code still preserves the `nn.MultiheadAttention` fallback when the package is missing.
- CATCH: **upstream code vendored verbatim** (2026-05-22). Direct port of `ts_benchmark/baselines/catch/{layers,utils,models}/*`. Modifications: (a) device-agnostic operation (`register_buffer` / `x.device` instead of hard-coded `.cuda()`), (b) TAB-framework `from ts_benchmark...` imports replaced with in-file definitions.



### 16. TimesNet (ICLR 2023)

**File:** `baselines/timesnet/{model.py, wrapper.py}`

**Paper:** [TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis](https://openreview.net/pdf?id=ju_Uqw384Oq)

**Description:** General SOTA time-series model. Detects top-k dominant periods via FFT, reshapes 1D → 2D tensor per period, applies parameter-efficient 2D Inception conv, adaptive period aggregation, residual connection. For anomaly detection: reconstruction-based with RevIN-style normalization.

**Architecture:**
```
Input: (win_size × n_features)
  → Instance Normalization (mean/var detach)
  → DataEmbedding (TokenConv + Positional)
  → e_layers × TimesBlock:
      - FFT_for_Period: top_k periods
      - reshape 1D → 2D
      - 2-stage Inception_Block_V1 (num_kernels=6) + GELU
      - reshape 2D → 1D, period-weighted aggregation
      - residual
  → LayerNorm + Linear(d_model → c_in)
  → De-normalization
```

**Anomaly Score:** Per-timestep MSE between input and reconstruction, max-aggregated over overlapping windows.

**Configuration** (actual `MODEL_PRESETS['default']`):
| Parameter | Value |
|-----------|-------|
| win_size | 100 |
| d_model | 64 |
| d_ff | 64 |
| e_layers | 3 |
| top_k | 3 |
| num_kernels | 6 |
| dropout | 0.1 |
| lr | 1e-4 |
| train_stride | 1 |
| epochs | 10 |
| batch_size | 128 |

**Reference (official code, TSLib repo):** [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library) (MIT, ★12,285). Note: `thuml/TimesNet` is an older standalone repo — the actively maintained official implementation is in Time-Series-Library.

**Citation:** 2,047 (Semantic Scholar, 2026-05). 영향력 인용 260.

---

### 17. TFMAE (ICDE 2024)

**File:** `baselines/tfmae/{model.py, wrapper.py}`

**Paper:** [Temporal-Frequency Masked Autoencoders for Time Series Anomaly Detection](https://ieeexplore.ieee.org/document/10597757) (ICDE 2024)

**Description:** Dual masked autoencoder operating in both temporal and frequency domains. Temporal branch uses variance-score based masking; frequency branch uses amplitude-based masking in FFT domain. Trained with adversarial-contrastive KL discrepancy between the two views. **사용자 self-distilled MAE의 직접 경쟁모델**.

**Architecture:**
```
Input: (win_size × n_features)
  → DataEmbedding
  → TemEnc (Temporal MAE):
      - Variance-score based top-tr masking
      - Encoder (unmasked) → mask_token + pos_emb
      - Decoder → reconstruction
      - Returns: list of [B,T,T] attention maps + recon
  → FreEnc (Frequency MAE):
      - FFT → amplitude-based bottom-fr masking
      - Replace with learnable complex mask_token
      - iFFT → Encoder → reconstruction
      - Returns: list of [B,T,T] attention maps + recon
```

**Loss (TFMAE paper Eq.15)**: `loss = con_loss - adv_loss`
- `adv_loss = Σ_u KL(tem_u, sg(fre_norm_u)) + KL(sg(fre_norm_u), tem_u)`
- `con_loss = Σ_u KL(fre_norm_u, sg(tem_u)) + KL(sg(tem_u), fre_norm_u)`

**Anomaly Score:** `softmax(adv_loss + con_loss, dim=-1) × k_temperature` per window position, max-aggregated.

**Configuration** (actual `MODEL_PRESETS['default']`):
| Parameter | Value |
|-----------|-------|
| win_size | 100 |
| seq_size | 5 |
| d_model | 128 |
| e_layers | 3 |
| n_heads | 8 |
| temporal_mask_ratio (tr) | 0.5 |
| freq_mask_ratio (fr) | 0.4 |
| dropout | 0.05 |
| lr | 1e-4 |
| train_stride | 1 |
| epochs | 10 |
| batch_size | 64 |
| k_temperature | 50.0 |

**Reference (official code):** [LMissher/TFMAE](https://github.com/LMissher/TFMAE) (MIT, ★36). Vendored upstream `model/{MTFAE,attn,embed}.py` to single `model.py` with device-agnostic operation (removed global `device` variable).

**비교 가치**: TFMAE는 TimesNet/DCdetector/GPT4TS를 자체 Table III에서 직접 비교한 frontier 논문. dual MAE + contrastive + adversarial 구조가 사용자 teacher/student + recon/disc MAE와 가장 유사 카테고리.

---

### 18. DCdetector (KDD 2023)

**File:** `baselines/dcdetector/{model.py, wrapper.py}`

**Paper:** [DCdetector: Dual Attention Contrastive Representation Learning for Time Series Anomaly Detection](https://arxiv.org/abs/2306.10347)

**Description:** Dual-attention contrastive representation learning. No reconstruction loss — instead minimizes a KL discrepancy between patch-wise and in-patch attention representations. Trained as a min-max objective on the KL gap.

**Configuration:** win_size=105, patch_size=[3,5,7], d_model=256, n_heads=1, e_layers=3, k_temperature=50.0, lr=1e-4, batch_size=128, epochs=10.

**Reference:** [DAMO-DI-ML/KDD2023-DCdetector](https://github.com/DAMO-DI-ML/KDD2023-DCdetector) (no LICENSE — attribution only).

---

### 19. MEMTO (NeurIPS 2023)

**File:** `baselines/memto/{model.py, wrapper.py}`

**Paper:** [MEMTO: Memory-guided Transformer for Multivariate Time Series Anomaly Detection](https://arxiv.org/abs/2312.02530)

**Description:** Transformer with gated memory module that maintains M cluster prototypes. Two-phase training: Phase 1 trains with random memory init, then K-means on encoder outputs initializes memory for Phase 2. Loss = MSE recon + λ·gathering_loss + λ·entropy_loss.

**Configuration:** win_size=100, n_memory=10, d_model=512, n_heads=8, e_layers=3, λ_gather=0.1, λ_entropy=0.1, phase1_epochs=3, lr=1e-4, batch_size=128, epochs=10.

**Reference:** [gunny97/MEMTO](https://github.com/gunny97/MEMTO).

---

### 20. ModernTCN (ICLR 2024 Spotlight)

**File:** `baselines/moderntcn/{model.py, wrapper.py}`

**Paper:** [ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis](https://openreview.net/forum?id=vpJMJerXHU)

**Description:** Patch-based pure-conv structure with large-kernel depthwise convolution (+ small kernel branch) and dual ConvFFN mixers (D-mixer per-variate, M-mixer per-feature). RevIN-wrapped reconstruction autoencoder.

**Anomaly Score (2026-06-01 faithfulness fix — was MAJOR_DEVIATION → now faithful):** per-test-window **per-position** reconstruction MSE (mean over features, exactly upstream), **overlap-averaged across all stride=1 windows covering each timestep** → `(T_test,)`. The previous implementation scored only the **last window position** (`err[:, -1]`), discarding 99/100 of upstream's reconstruction signal; the fix restores upstream ModernTCN-detection's flatten-all-positions behavior (`comparison/baselines/moderntcn/wrapper.py:33-38, 263-274`). moderntcn is one of the **6 "untouchable"** models for normalization (raw `none` branch + internal whole-array RevIN; NOT per-entity'd).

**Configuration:** win_size=96, patch_size=8, patch_stride=4, dims=[32], num_blocks=[1], large_size=[13], small_size=[5], ffn_ratio=1, dropout=0.3, lr=1e-4, batch_size=128, epochs=10.

**Reference:** [luodhhh/ModernTCN](https://github.com/luodhhh/ModernTCN) (MIT).

---

### 21. CATCH (ICLR 2025)


**File:** `baselines/catch/{model.py, wrapper.py}`

**Paper:** "CATCH" (ICLR 2025) — Channel-Aware Tokenization with frequency reconstruction

**Description:** Per-channel patching, channel masking (drop random channels at training), cross-channel Transformer over channel embeddings. Joint loss = time-domain MSE + α·frequency-domain MSE (|FFT|) + β·channel-discovery BCE.

**Configuration:** win_size=96, patch_size=24, patch_stride=12, d_model=128, n_heads=8, e_layers=2, ch_mask_ratio=0.3, λ_freq=0.5, λ_ch_disc=0.5, lr=1e-4, batch_size=128, epochs=10.

**Reference:** [decisionintelligence/CATCH](https://github.com/decisionintelligence/CATCH) (TAB framework). Implementation: **upstream vendored verbatim** (2026-05-22). Seven upstream files (`CATCH.py`, `models/CATCH_model.py`, `layers/{RevIN,channel_mask,cross_channel_Transformer}.py`, `utils/{ch_discover_loss,fre_rec_loss}.py`) consolidated into `model.py`. Only modifications: (a) `from ts_benchmark...` imports removed, (b) `.to(x.device)` replaces hard-coded `.cuda()` in `channel_mask` + `ch_discover_loss`, (c) `device=` kw added to `torch.eye/zeros` in `frequency_criterion`. No architectural change. Wrapper reproduces upstream `detect_fit` / `detect_score` (mask generator stepped every `min(N/10, 100)` mini-batches). External dep: `einops` (already in dc_vis).

---

### 22. NPSR (NeurIPS 2023)

**File:** `baselines/npsr/{model.py, wrapper.py}`

**Paper:** [Nominality Score Conditioned Time Series Anomaly Detection by Point/Sequential Reconstruction](https://arxiv.org/abs/2310.15416)

**Description:** Two complementary autoencoders. M_pt does per-timestep reconstruction with short-range attention; M_seq does sequence-level reconstruction with random induction-position masking (longer-range attention, optionally Performer). Scoring formula: `score(t) = max(err_pt(t), err_seq(t) − N(x))` where N(x) is the θ_N-quantile of err_pt(t) (per-batch nominality threshold).

**Configuration:** win_size=100, induction_length=16, d_model=256, n_heads=4, e_layers=4, θ_N=0.985, lr=1e-4, batch_size=64, epochs=10.

**Reference:** [andrewlai61616/NPSR](https://github.com/andrewlai61616/NPSR). The optional `performer-pytorch` dependency is auto-detected; fallback to `nn.MultiheadAttention` if not installed.

---

## Reference-only Models (코드는 유지, 실험 queue에서 제외)

> **2026-05-30 정정 (final audit rework)**: 이 섹션은 더 이상 유효하지 않다. 과거 2026-05-19 batch의 reference-only 후보 3개 모델은 **2026-05-26에 코드/wrapper/디렉토리까지 완전히 제거**되었으며 (`comparison/baselines/` 에 해당 dir 없음), 활성 모델은 `BASELINE_MODELS` (22개) 가 단일 source of truth이다. 이전 본문이 참조하던 `NEW_SOTA_MODELS` 심볼은 현재 코드에 **존재하지 않는다**. (제거 사유: 단일 (dataset, model) 작업당 3-37시간 소요로 39 데이터셋 pipeline에 비현실적인 timing outlier.)

원본(stale) 기술: ~~다음 3개 모델은 모델 코드와 wrapper, `MODEL_PRESETS` 진입점, `create_model()` 분기까지 모두 유지되어 실험 queue에서만 제외~~ — 위 정정으로 대체됨.

---

## Our Model: Self-Distilled MAE

**Location:** `mae_anomaly/` (separate from comparison baselines)

**Description:** Masked Autoencoder with self-distillation for time series anomaly detection. The model produces multiple loss components:
- `reconstruction_loss`: Teacher network's reconstruction error
- `discrepancy_loss`: Teacher-student discrepancy (self-distillation signal)
- `total_loss`: Combined loss

**Running MAE:**
```bash
# Run ablation study
python scripts/ablation/run_ablation.py --config configs/{config}.py

# Results saved to:
# results/{Dataset}/{scenario}/
```

**Scoring Modes:**
- `default`: recon + λ × disc (fixed weight)
- `adaptive`: recon + (μ_recon/μ_disc) × disc (dynamic weight)
- `normalized`: z(recon) + z(disc) (z-score normalization)

See `docs/ABLATION_STUDIES.md` for configuration details.

---

## Adding a New Model

1. **Create directory:** `comparison/baselines/new_model/`

2. **Implement model** matching the unified wrapper interface (matches `comparison/baselines/anomaly_transformer/wrapper.py` and others):
```python
# comparison/baselines/new_model/model.py
import numpy as np
import torch.nn as nn
from pathlib import Path
from typing import Optional

class NewModelBaseline:
    """
    Wrapper interface compatible with comparison/baseline_common.py
    execution paths (run_simple_baseline / run_dl_baseline_with_epoch_eval /
    run_sota_baseline_with_epoch_eval).
    """

    def __init__(self, **hparams):
        self.name = "New Model"
        # Read hparams from MODEL_PRESETS['default']['new_model']

    def fit(self, train_X: np.ndarray, epoch_callback=None) -> 'NewModelBaseline':
        """
        Args:
            train_X: (N_train, n_features) — already z-score or min-max normalized
            epoch_callback(self, epoch): Optional. Called after each training epoch.
                                         Pipeline uses this to compute per-epoch
                                         metrics via predict() + save scores.npz.
        Returns:
            self
        """
        # ... training loop with epoch_callback(self, ep+1) per epoch
        return self

    def predict(self, test_X: np.ndarray) -> np.ndarray:
        """
        Args:
            test_X: (N_test, n_features)
        Returns:
            (N_test,) float32 — 1D anomaly score per timestep.
            Window→point aggregation handled internally (mean or max).
        """
        return scores

    # Optional: for DL models (enables model checkpointing)
    def save(self, save_dir: Path) -> None: ...
    def load(self, save_dir: Path) -> 'NewModelBaseline': ...
```

3. **Create `__init__.py`:**
```python
# comparison/baselines/new_model/__init__.py
from .model import NewModelBaseline
__all__ = ['NewModelBaseline']
```

4. **Register in `comparison/baselines/__init__.py`:**
```python
from .new_model import NewModelBaseline
# Add to __all__ list
```

5. **Register in `comparison/baseline_common.py`:**
   - Import wrapped in `try/except ImportError` (matches existing SOTA pattern):
   ```python
   try:
       from comparison.baselines import NewModelBaseline
       HAS_NEW_MODEL = True
   except ImportError:
       HAS_NEW_MODEL = False
   ```
   - Add to `BASELINE_MODELS` list, `SOTA_MODELS` (or `NEURAL_MODELS`), `SOTA_AVAILABILITY` dict.
   - Add hyperparameter entry to `_get_default_model_params()` (MODEL_PRESETS['default']).

6. **Add to `comparison/experiment_configs.py`:**
   ```python
   STANDARD_BASELINES = [
       ...existing models...,
       'new_model',
   ]
   ```

7. **Wire into `comparison/run_baseline.py`** dispatch (the runner auto-picks the execution path based on whether the model is in `SIMPLE_MODELS`, `NEURAL_MODELS`, or `SOTA_MODELS`). For SOTA path: model must accept `epoch_callback=None` in `fit()`.

---

## References

### Benchmark & Evaluation

1. **QuoVadisTAD:** Sarfraz et al., "Position: Quo Vadis, Unsupervised Time Series Anomaly Detection?", ICML 2024
   - Paper: https://arxiv.org/abs/2405.02678
   - Code: https://github.com/ssarfraz/QuoVadisTAD

2. **F1_T Metric:** Tatbul et al., "Precision and Recall for Time Series", NeurIPS 2018
   - Paper: https://arxiv.org/abs/1803.03639

### Deep Learning Methods

3. **Anomaly Transformer:** Xu et al., "Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy", ICLR 2022
   - Paper: https://arxiv.org/abs/2110.02642
   - Code: https://github.com/thuml/Anomaly-Transformer

4. **TranAD:** Tuli et al., "TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data", VLDB 2022
   - Paper: https://arxiv.org/abs/2201.07284
   - Code: https://github.com/imperial-qore/TranAD

5. **GDN:** Deng & Hooi, "Graph Neural Network-Based Anomaly Detection in Multivariate Time Series", AAAI 2021
   - Paper: https://arxiv.org/abs/2106.06947
   - Code: https://github.com/d-ailin/GDN

6. **USAD:** Audibert et al., "USAD: UnSupervised Anomaly Detection on Multivariate Time Series", KDD 2020
   - Paper: https://dl.acm.org/doi/10.1145/3394486.3403392
   - Code: https://github.com/manigalati/usad

7. **OmniAnomaly:** Su et al., "Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network", KDD 2019
   - Paper: https://dl.acm.org/doi/10.1145/3292500.3330672
   - Code: https://github.com/NetManAIOps/OmniAnomaly

8. **DAGMM:** Zong et al., "Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection", ICLR 2018
   - Paper: https://openreview.net/forum?id=BJJLHbb0-
   - Implementation reference (TS-AD adaptation, 2026-05-25): https://github.com/imperial-qore/TranAD `src/models.py::DAGMM`
   - Note: 원본 저자는 PyTorch official code 공개 안 함. TS-AD benchmark에서는 TranAD-author 변형이 사실상 표준.

### Other Methods

9. **MLPMixer:** Tolstikhin et al., "MLP-Mixer: An all-MLP Architecture for Vision", NeurIPS 2021
    - Paper: https://arxiv.org/abs/2105.01601

---

## Weakly Supervised Baselines (2026-05-30 통합 — Q1-only, GPU 미실행)

기존 22개와 달리 학습 시 `train_y` 사용. 약 label = `max(train_y over window)` (train split 한정, leak-free). 전용 실행 경로 `run_weak_sota_baseline_with_epoch_eval`. **Q1-only (Q3 = N/A — `train_y` 전부 0 → positive bag 없음 → `RuntimeError`).** 상태 = **구현 완료 · CPU dry-test 통과 · GPU 미실행** (결과표 weak 행 수치 0개). 독립 리뷰 verdict 병기. 상세 work-log: `temp/ssl_official_baseline_porting_0529/`.

**Normalization fidelity (2026-05-30 핵심 수정):** 4종 모두 이전엔 pipeline **global MinMax** 를 받아 원논문과 불일치(fidelity 결함)했으나, 현재 `run_baseline.py` `SELF_NORMALIZING_WEAK={wetas, treemil, nrdetector, deepmil}` 등록으로 **raw 데이터 수신 + 원논문 normalization 자체 적용**: WETAS/TreeMIL/DeepMIL = per-recording/per-file StandardScaler(z-score), NRdetector = per-split z-score StandardScaler. 4종 모두 2026-06-02 leak-free per-source-file rework 적용: fit() 이 각 source file 의 TRAIN slice 에 자신의 scaler 를 fit 후 캐시하고, predict() 는 PAIRED test file 을 그 캐시된 TRAIN scaler 로 `.transform` (절대 fit-on-test 안 함, 공유 kernel `comparison/baselines/_per_file_norm.py`). 이는 이전 4건의 fit-on-test leak (wetas/treemil/deepmil/nrdetector predict 경로) 을 제거한 것. 단일 파일 데이터셋에서는 per-file ≡ whole-array NO-OP. 모델별 상세는 각 §의 **Normalization** 항목.

**Provenance gate (G1–G5):** 본 4종은 `GUIDE.md §7.1` 의 baseline 포팅 재발방지 gate (G1 출처 라벨+source locus / G2 NON_OFFICIAL ≥5-round source-chain / G3 in-project sibling cross-check / G4 provenance≠comparability / G5 vendored VCS 가시성) 를 적용. 각 §의 **Provenance** 항목에 라벨 기록 — DeepMIL encoder(DERIVATIVE_CITED, G3 WETAS DiCNN 재사용)·NRdetector encoder schedule(NON_OFFICIAL/IMPL-INVENTED)이 대표 사례.

### 23. DeepMIL (CVPR 2018) — clean-room 재구현

**File:** `baselines/deepmil/{model.py, wrapper.py}`

**Paper:** Sultani, Chen, Shah, "Real-World Anomaly Detection in Surveillance Videos", CVPR 2018, pp. 6479–6488 (doi:10.1109/CVPR.2018.00678, [arXiv:1801.04264](https://arxiv.org/abs/1801.04264)).

**Reference (중요, two-part provenance):**
- **head+loss = FAITHFUL (clean-room):** 공식 repo [WaqasSultani/AnomalyDetectionCVPR2018](https://github.com/WaqasSultani/AnomalyDetectionCVPR2018) 은 **무 LICENSE · legacy Keras 1.1.0/Theano · video/C3D · 실행 불가**. vendoring 불가 → MIL ranking head+loss 를 paper 로부터 clean-room 재구현 (DAGMM §13-style reference substitution; 공식 repo 코드 미복사, ekosman MIT PyTorch port 는 교차검증용으로만 참조).
- **encoder = DERIVATIVE_CITED (NOT OFFICIAL):** Sultani 원논문은 frozen C3D fc6 4096-d feature 를 소비하는 **video** 방법이라 학습형 TS encoder 가 없다. DeepMIL-on-TS encoder 의 canonical 정의는 후속 WS-TSAD 논문 **WETAS (Lee et al., ICCV'21, CVF p.7360)** 의 verbatim 문장: *"DeepMIL employs the same model architecture with WETAS (i.e., DiCNN)"* + *"DeepMIL-4,8,16"*. 따라서 in-project vendored **WETAS `DilatedCNN`** (`comparison/baselines/wetas/model.py`) 을 encoder 로 재사용 (input=n_features, hidden=output=128, kernel=2, n_layers=7 → RF=2^7=128). **OFFICIAL Sultani encoder 아님** (공식 video 방법엔 TS encoder 부재). 이전 bespoke `TSSegmentEncoder` 은퇴.

**Normalization:** per-recording StandardScaler (z-score, WETAS-lineage; `timeseries.py`). wrapper 가 raw 데이터에 자체 적용 (`run_baseline.py` `SELF_NORMALIZING_WEAK`) — 이전 global-minmax 결함 수정. train = train recording fit/transform, predict = PAIRED test file 마다 캐시된 TRAIN scaler 로 `.transform` (per-source-file leak-free kernel `_per_file_norm.transform_test_per_file`; 이전 transductive whole-test leak 제거).

**Architecture:** encoder = WETAS DiCNN (위 DERIVATIVE_CITED, dense per-timestep feature map `(B,L,128)`) + MIL ranking head `D→512(ReLU,Dropout0.6)→32(Dropout0.6)→1(Sigmoid)`, xavier init, L2=0.001 (FAITHFUL). bag = window.

**Loss:** ranking hinge (margin 1.0) on max-over-timesteps(+bag vs −bag) + smoothness(λ=8e-5) + sparsity(λ=8e-5), positive bag 대상 (FAITHFUL; max 가 segment→timestep 로 dense 화된 것 외 불변).

**Configuration:** **optimizer = Adam lr=1e-4 (WETAS `train_classifier.py:234` 출처)** — Sultani 의 Adagrad lr=0.01 은 frozen-C3D shallow head 전용이라 deep DiCNN encoder 와 joint 학습 시 발산(logits→-200/-440, score collapse)하므로 encoder 출처 optimizer 사용 (preset `optimizer='adam'`/`lr=1e-4`; head/loss 는 Sultani 유지, optimizer 만 encoder 출처). 60 bags/batch (30 pos + 30 neg), seq_len(bag window)=128, encoder_dim=128, dropout=0.6, epochs=10, iters_per_epoch=50. `n_segments=32` 은 config 호환용 **vestigial** (dense per-timestep MIL — 32-seg 변형 미구현).

**Score (dense per-timestep, NON_OFFICIAL disclosed):** encoder dense feature → head 를 매 timestep 에 적용 → per-timestep sigmoid `(B,L)` → overlap mean aggregate → `(N_test,)` raw. (원 DeepMIL 의 32 video segment scoring 은 streaming TS 에 official counterpart 가 없어 dense 로 대체; NON_OFFICIAL disclosed.)

**Provenance (G1–G5, `GUIDE.md §7.1`):** encoder=DERIVATIVE_CITED (G3 sibling 재사용), head+loss=FAITHFUL, optimizer=encoder-sourced (G1, "design choice" 아님), dense scoring=NON_OFFICIAL (G1 disclosed).

**Phase 4 verdict:** head/loss/optimizer VERIFIED_SAME; encoder = literature-sanctioned DiCNN (DERIVATIVE_CITED, NEEDS_REVIEW → ACCEPT). F-1 LOW: paired vs all-pairs hinge (동일 objective). 또한 deep DiCNN encoder 와 Sultani 의 Adagrad(lr=0.01) joint 학습 시 **epoch-1 collapse** (logits→ −200/−440, score 붕괴) 관측 → encoder-출처 Adam(lr=1e-4) 로 대체하여 회피 (위 Configuration 참조).

### 24. WETAS (ICCV 2021) — 공식 vendoring

**File:** `baselines/wetas/{model.py, softdtw_cuda.py, wrapper.py}`

**Paper:** Lee, Yu, Ju, Yu, "Weakly Supervised Temporal Anomaly Segmentation With Dynamic Time Warping", ICCV 2021, pp. 7335–7344 (IEEE pagination; CVF 7355–7364) (doi:10.1109/ICCV48922.2021.00726, [arXiv:2108.06816](https://arxiv.org/abs/2108.06816)).

**Reference:** [donalee/WETAS](https://github.com/donalee/WETAS) (GPL-3.0; bundled soft-DTW MIT), commit `cb149dc`. `model.py`+`softdtw_cuda.py` verbatim vendoring (device-agnostic 편집만, license header 보존). numba 0.61.2 기존 설치 → install 불요.

**Normalization:** per-recording StandardScaler (z-score), `timeseries.py:35-40` (`_preprocess` 가 recording 파일마다 fresh `StandardScaler().fit_transform`). wrapper 가 raw 데이터에 자체 적용 (`SELF_NORMALIZING_WEAK`) — 이전 pipeline global-minmax 결함 수정. train = `train_segments` 별 fit/transform, predict = PAIRED test recording 마다 캐시된 TRAIN StandardScaler 로 `.transform` (per-source-file leak-free kernel; 2026-06-02 이전 transductive fit-on-test leak 제거).

**Architecture:** WaveNet-style dilated-CNN (n_layers=7, gated residual) + 공유 `fc(128,1)` head (weak pool head + dense per-timestep head).

**Loss:** `BCE(wscore, wlabel) + dtw_loss` (soft-DTW triplet hinge, beta=0.1, gamma=0.1), Adam lr=1e-4, batch_size=32, split_size=500.

**Score:** continuous dense `dscore` (binary `dpred` 아님) → non-overlap split flatten, front zero-pad drop → `(N_test,)`. dense `dscore = σ(fc(out))` 는 upstream 의 `dauc`/`dauprc` ranking-score 입력과 정확히 동일 ⇒ **본 실험의 ranking metric (pak_auc_f1/ROC/PRC) 에 대해 faithful**. WETAS 논문 headline 의 DTW-aligned point-F1/IoU (`get_dpred`) 은 **다른 segmentation metric 으로 본 실험 metric suite 에 없음 → 의도적으로 미산출** (누락 아님).

**Phase 4 verdict:** CLEAN (architecture byte-identical except device edits; 17/17 HP 일치).

### 25. TreeMIL (ICASSP 2024) — 공식 active core vendoring

**File:** `baselines/treemil/{model.py, wrapper.py}`

**Paper:** Liu, He, Liu, Li, "TreeMIL: A Multi-instance Learning Framework for Time Series Anomaly Detection with Inexact Supervision", ICASSP 2024, pp. 7510–7514 (doi:10.1109/ICASSP48485.2024.10447536, [arXiv:2401.11235](https://arxiv.org/abs/2401.11235)).

**Reference:** [fly-orange/TreeMIL](https://github.com/fly-orange/TreeMIL) (GPL-3.0), commit `16f166c`. N-ary-tree transformer core + window-BCE (`last_loss`) vendoring. **soft-DTW/alignment 은 학습 gradient 경로에 없는 dead code → 제외** (Phase 4 증명: `train.py:143` backward = BCE만). torch-only.

**Normalization (2026-05-30 — 이전 silent, deviation 명시):** 원논문 = per-file StandardScaler (z-score), `timeseries.py:53-55` (`_preprocess` 가 input 파일마다 전체 `(T,D)` 에 fit 후 train/valid/test slice 에 transductive 적용). wrapper 가 raw 데이터에 자체 적용 (`SELF_NORMALIZING_WEAK`) — 이전엔 pipeline global-minmax 를 받아 원논문과 불일치했고 이 deviation 이 문서에 기록되지 않았음(silent). 현재 train = `train_segments` 별 per-file z-score (windowing 前), predict = PAIRED test file 마다 캐시된 TRAIN scaler 로 `.transform` (per-source-file leak-free kernel `_per_file_norm.transform_test_per_file`; 이전 transductive whole-test fit_transform leak 제거).

**Architecture:** Conv embedding+positional → multi-scale tree nodes(=MIL instances) → masked MHA (parent/child/neighbor/self) → 공유 `Linear(d_model,1)+sigmoid`. window-score = max-pool, dense-score = gather-ancestors.

**Configuration:** split_size=500, ary_size=2, d_model=128, n_head=5, n_layer=2, lr=1e-4, batch_size=32, epochs=200. effective ctor 값 사용 (오해의 argparse default 아님).

**Score:** dense `dscore (B, split_size)` → contiguous non-overlap window, **tail-pad** + truncate to N_test (공식 left-pad 미복제 — misalignment 방지).

**Phase 4 verdict:** CLEAN.

### 26. NRdetector (KDD 2025) — 공식 vendoring + fidelity 수정

**File:** `baselines/nrdetector/{model.py, wrapper.py}`

**Paper:** Wang et al., "Noise-Resilient Point-wise Anomaly Detection in Time Series Using Weak Segment Labels", KDD 2025 (doi:10.1145/3690624.3709257, [arXiv:2501.11959](https://arxiv.org/abs/2501.11959)).

**Reference:** [UCSC-REAL/NRdetector](https://github.com/UCSC-REAL/NRdetector) (MIT, Yang Liu lab), commit `bd5592b`. encoder + PU-LP selector + PU classifier vendoring. CLI `Solver` → in-memory `fit/predict` adapter. **HOC(이진화기)·soft-DTW(공식 encoder 학습 코드 부재) 제외.**

**Normalization:** per-split z-score StandardScaler, `data_loader.py:50-55` (paper §5.2 "following Xu 2021"; `_preprocess` 가 split 마다 fresh `StandardScaler` fit/transform). wrapper 가 raw 데이터에 자체 적용 (`SELF_NORMALIZING_WEAK`) — 이전 global-minmax 결함 수정. 현재 normalization 은 공유 leak-free per-source-file kernel (`comparison/baselines/_per_file_norm.py`) 을 NRdetector 자신의 StandardScaler identity 로 통과시킴: `fit()` 이 각 source file 의 TRAIN slice 에 scaler fit 후 `self._scalers` 에 캐시, `predict()` 가 PAIRED test file 을 그 캐시된 TRAIN scaler 로 `.transform` (절대 fit-on-test 안 함). 이전 global-minmax + 일부 fit-on-test 경로 모두 제거.

**Architecture:** 2-stage PU — DilatedCNN encoder → PU-LP kNN-graph selector (`noisy_rate=0.4`) → PU classifier (`LabelDistributionLoss` + `constraint_loss`).

**Configuration:** win_size=100, hidden=64, output=64, classifier_hidden=128, lr=1e-5, batch_size=32, epochs=200. **encoder_epochs=50 / encoder_lr=1e-3 (2026-05-30 파라미터화)**. 분류 구분:
- **fixed-param:** win_size, hidden, classifier_hidden, lr, batch_size, epochs, knn_k=5, seed=0.
- **runtime-estimated:** `prior=None` → 런타임 동적 추정 (train wlabel rate, clip [0.05,0.5]). 데이터셋마다 anomaly ratio 가 다른 PU class prior 는 **intrinsic 속성이라 추정이 맞음** (공식 고정 0.25/0.31 우회는 의도적). 사용된 prior 는 로깅됨.
- **fixed knob (추정 대상 아님):** `noisy_rate=0.4` = experimenter-imposed **reveal fraction** — 양성 train segment 중 첫 40%만 labeled-P 로 공개, 나머지는 unlabeled 로 demote (`selector.py:31-39`). dataset 속성이 아닌 실험 knob 이라 고정.
- **IMPL-INVENTED (confound):** `encoder_epochs=50` / `encoder_lr=1e-3`. 공식엔 encoder **학습 recipe 가 없음** — pretrained `.pth` 를 로드만 한다:
  > `modules/extractor.py:65` — `model.load_state_dict(torch.load(".../"+dataset+"_model_4.pth"))`

  관련 paper §4.2.1 verbatim:
  > *"we utilize the basic architecture of dilated CNN (DiCNN) ... put it into the WETAS framework. Note that the extractor here can be replaced with another temporal feature extractor."*

  우리는 해당 `.pth` 미보유라 from-scratch 로 BCE-only 학습(`bce(wscore, wlabel)`, `extractor.py:127`; soft-DTW 는 install ban 으로 제외). 따라서 출처 없는 50/1e-3 는 feature-quality 에 영향을 줄 수 있는 **confound 로 문서화** (NON_OFFICIAL). 완전 일치는 공식 pretrained weight 필요(미가용 가능성).

**Score (2026-06-03 PM — classifier-gate RESTORED; see `NRDETECTOR_CORRECTION_v2.md`):** `predict()` emits the **CLASSIFIER-GATED continuous per-window min-max actmap**: `scores = actmap × [seg_prob ≥ mean(seg_prob)+anomaly_thre·(max−min)]`, `anomaly_thre=0` (`actmap=(h−min)/max`, upstream `extractor.py get_dpred`; `seg_prob` = PU-classifier per-window prob, `solver.test()` mean-gate). Non-flagged windows → 0; flagged windows keep their continuous actmap → `(N_test,)`. **The gate is INTEGRAL to the upstream DEFAULT path, not optional:** `--mode` default `'train'` → runs ONLY `solver.rank_test()` (`begin/train/test/pick_test` commented, main.py:44-48), which ranks `point_Score = self.interested_instance.reshape(-1)` (solver.py:219); `interested_instance` = `save_instance_files` keeping ONLY classifier-flagged windows (`instance_label[i]>0`, solver.py:198-203), `instance_label` = the classifier mean-gate (solver.py:164-171,185). So upstream ranks the **classifier-flagged-window actmap** (the gate selects the ranked pool). Our harness (ROC/PRC/pak_auc_f1) does its own operating-point selection, replacing upstream's `anomaly_ratio=0.65`+HOC single-label selectors. ⚠ **The no-gate "continuous ACTMAP" of commit `5cff9da` (2026-06-03 AM) was a DOUBLE REGRESSION — REVERTED:** (1) the encoder is frozen after Stage 0 (only the PU-classifier trains per-epoch), so an encoder-only actmap is IDENTICAL every epoch → per-epoch scores became **bit-identical** (best-epoch meaningless); (2) the ungated actmap floods the ranking with normal-window points → collapse. Measured (inference-only): SWaT pak_auc_f1 **0.858 (gated) → 0.440 (no-gate)**, roc_auc 0.883→0.365. The earlier "binary-gate = MAJOR_DEVIATION" adjudication was an **oversight** — the gate is faithful; removing it was the bug. Per-window min-max actmap + boundary-safe windowing + per-file leak-free normalization all preserved.

**Provenance (G1–G5, `GUIDE.md §7.1`):** encoder schedule = NON_OFFICIAL/IMPL-INVENTED (G1 confound 문서화 + G2 source-chain: official `.pth`-load 확인). `prior` = runtime-estimated (intrinsic). `noisy_rate` = fixed experiment knob.

**Phase 4 verdict:** FIXED → VERIFIED_SAME. 수정 3건: MM1(HIGH) 공식 `calc_lp` 포팅 → PU classifier가 RP=labeled-P / RN=label-propagation `lp_n` 로 학습; MM3 directed adjacency for W; MM4 per-window min-max actmap base score (+ raw-h features). 문서화된 제약: encoder BCE-only, `prior` 동적 추정, `encoder_epochs`/`encoder_lr` IMPL-INVENTED. ⚠ **2026-06-03 PM 정정:** MM4의 classifier-gated score(`actmap × [seg_prob ≥ mean]`)가 **faithful**이다 — upstream DEFAULT `rank_test`가 ranking하는 `interested_instance`는 classifier가 flag한 window의 actmap만 모은 것(`save_instance_files`, `instance_label[i]>0`, solver.py:198-203)이므로 gate가 ranked pool을 결정한다. 잠깐 gate를 제거했던 commit `5cff9da`(2026-06-03 AM, "binary-gate=MAJOR_DEVIATION" 오판)는 **회귀였고 revert됨** — encoder가 Stage 0 후 frozen이라 gate 없는 actmap은 epoch마다 불변(bit-identical)이 되고 ranking도 붕괴(SWaT pak 0.86→0.44). per-window min-max actmap base + classifier gate 모두 유지. (`NRDETECTOR_CORRECTION_v2.md`)
