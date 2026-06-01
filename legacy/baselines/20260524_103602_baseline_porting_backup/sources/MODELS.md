# Baseline Models

**22 active baseline models** (5 simple + 3 neural-simple + 7 legacy SOTA + 7 new SOTA 2023-2025) + **3 reference-only models** (코드 유지, 실험 제외; sections 23-25 참조) = **총 25개 디렉토리**. All neural models use unified **10 epochs** with `pak_auc_f1` (PA%K AUC F1, per-K re-optimized; Kim et al. AAAI 2022) best-epoch selection. Single preset (`default`) across all datasets — see `MODEL_PRESETS` in `comparison/baseline_common.py`.

Sources: [QuoVadisTAD](https://arxiv.org/abs/2405.02678) (ICML 2024 Position Paper, 9 baselines — covers all simple/neural-simple + GCN-LSTM), **6 standalone legacy SOTA papers** (2018-2022), and **7 active new SOTA papers (2023-2025)** integrated 2026-05-19: TFMAE (ICDE'24), NPSR (NeurIPS'23), TimesNet (ICLR'23), DCdetector (KDD'23), MEMTO (NeurIPS'23), ModernTCN (ICLR'24 Spot), CATCH (ICLR'25). AnomalyBERT (ICLR'23 WS), CAROTS (ICML'25), CrossAD (NeurIPS'25) 3개 모델은 timing outlier로 실험 제외 — 코드는 sections 23-25에 reference로 보존. GCN-LSTM is grouped under **SOTA** in this guide because it uses an internal training loop with per-epoch callback (same execution interface as the other SOTA models), even though QuoVadisTAD provides its configuration.

Datasets covered (7): Simulation, SWaT A1+A2, WaDi A1, WaDi A2, SMD (28 machines), PSM, Exathlon (6 apps {1,2,4,5,6,9}). All 22 active models are evaluated on each dataset under Q1 (minmax full) and Q3 (minmax normalonly) conditions.

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

**Description:** Uses PCA to learn a low-dimensional representation of normal data, then measures reconstruction error.

**Method:**
1. Fit PCA on training data
2. Project test data to low-dim space and reconstruct
3. Anomaly score = reconstruction error (MSE)

**Configuration:**
- `n_components`: Number of PCA components ('auto' for 95% variance)

**From QuoVadisTAD:** "PCA captures the main modes of variation in the data. Anomalies that don't fit these modes will have high reconstruction error."

---

### 4. L2-Norm

**File:** `baselines/l2_norm/model.py`

**Description:** Uses the L2 norm of feature vectors as anomaly score.

**Method:**
- Anomaly score = ||x||_2 (optionally normalized)

**Configuration:**
- `ord`: Norm order (default: 2)
- `normalize`: Whether to normalize by number of features

**From QuoVadisTAD:** "A trivial baseline that assumes anomalies have larger feature magnitudes. Works when anomalies manifest as extreme values."

---

### 5. 1-NN Distance (Nearest Neighbor Distance)

**File:** `baselines/nn_distance/model.py`

**Description:** Computes Euclidean distance to the nearest training sample.

**Method:**
1. Store (subsampled) training data
2. For each test point, find nearest neighbor in training set
3. Anomaly score = distance to nearest neighbor

**Configuration:**
- `distance`: Distance metric ('euclidean', 'manhattan')
- `subsample`: Number of training samples to store (for memory efficiency)

**From QuoVadisTAD:** "Points far from all training samples are likely anomalies. Simple but effective, especially when normal data is clustered."

**Note:** Memory-intensive for large datasets. Use `subsample` to reduce.

---

## Neural Baselines (QuoVadisTAD)

These are minimal neural network architectures designed to be simple yet competitive.

### 6. 1-Layer MLP

**File:** `baselines/mlp/model.py`

**Description:** Single hidden layer MLP for time series forecasting. Predicts the next timestamp from a sliding window.

**Architecture:**
```
Input: (seq_len × n_features) → Flatten → Linear(embedding_dim) → ReLU → Linear(n_features)
```

**Configuration (from QuoVadisTAD paper):**
| Parameter | Value |
|-----------|-------|
| seq_len | 5 |
| embedding_dim | 32 |
| epochs | 200 |
| learning_rate | 0.001 |
| batch_size | 512 |

**Anomaly Score:** MSE between predicted and actual next timestamp.

**From QuoVadisTAD:** "The simplest neural baseline. Despite its simplicity, it often outperforms more complex methods."

---

### 7. Single Block MLPMixer

**File:** `baselines/mlpmixer/model.py`

**Description:** MLPMixer architecture with a single block. Mixes information across both time (token) and feature (channel) dimensions.

**Architecture:**
```
Input: (seq_len × n_features)
  → Linear(embedding_dim)
  → MLPMixer Block:
      - Token Mixing: MLP across time dimension
      - Channel Mixing: MLP across feature dimension
  → LayerNorm
  → Global Average Pool
  → Linear(n_features)
```

**Configuration (from QuoVadisTAD paper):**
| Parameter | Value |
|-----------|-------|
| seq_len | 5 |
| embedding_dim | 128 |
| dropout | 0.1 |
| epochs | 20 |
| learning_rate | 0.0002 |
| batch_size | 512 |

**From QuoVadisTAD:** "MLPMixer provides an attention-free alternative to Transformers. The token mixing captures temporal patterns while channel mixing captures feature interactions."

---

### 8. Single Transformer Block

**File:** `baselines/transformer/model.py`

**Description:** Single Transformer encoder block with self-attention.

**Architecture:**
```
Input: (seq_len × n_features)
  → Linear(embedding_dim)
  → Positional Encoding (learned)
  → Transformer Block:
      - Multi-head Self-Attention (1 head)
      - Feed-Forward Network
  → LayerNorm
  → Global Average Pool
  → Linear(n_features)
```

**Configuration (from QuoVadisTAD paper):**
| Parameter | Value |
|-----------|-------|
| seq_len | 5 |
| embedding_dim | 128 |
| num_heads | 1 |
| dropout | 0.1 |
| epochs | 20 |
| learning_rate | 0.001 |
| batch_size | 512 |

**From QuoVadisTAD:** "Self-attention can capture long-range dependencies. Even with a single head and block, it's competitive with deeper models."

---

## SOTA Comparison

### 9. 1-Layer GCN-LSTM (from QuoVadisTAD)

**File:** `baselines/gcn_lstm/model.py`

**Description:** Combines Graph Convolutional Network (for sensor relationships) with LSTM (for temporal dynamics).

**Architecture:**
```
Input: (seq_len × n_features)
  → Transpose to (n_features × seq_len)
  → GCN Layer (aggregate across sensors using adjacency matrix)
  → LSTM (process each sensor's sequence)
  → Linear(n_features)
```

**Adjacency Matrix:** Built from top-k nearest sensors based on correlation in training data.

**Configuration (from QuoVadisTAD paper):**
| Parameter | Value |
|-----------|-------|
| seq_len | 5 |
| gcn_out_dim | 10 |
| lstm_units | 64 |
| adj_topk | 10 |
| dropout | 0.1 |
| epochs | 20 |
| learning_rate | 0.001 |
| batch_size | 100 |

**From QuoVadisTAD:** "GCN captures spatial (sensor-to-sensor) relationships while LSTM captures temporal dynamics. Useful when sensors have known or learnable relationships."

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

### 13. DAGMM (ICLR 2018)

**File:** `baselines/dagmm/model.py`

**Paper:** [Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection](https://openreview.net/forum?id=BJJLHbb0-)

**Description:** Combines deep autoencoder with Gaussian Mixture Model for density-based anomaly detection.

**Architecture:**
```
Input: Flattened window
  → Compression Network (Autoencoder):
      - Encoder: Input → z_c (latent)
      - Decoder: z_c → Reconstruction
  → Reconstruction Features z_r:
      - Relative Euclidean distance
      - Cosine similarity
  → Full latent: z = [z_c, z_r]
  → Estimation Network:
      - z → GMM membership γ
  → GMM density estimation
```

**Anomaly Score:** Energy (negative log probability under GMM)

**Loss:** Recon_loss + λ₁·Energy_loss + λ₂·Cov_regularization

**Configuration** (actual `MODEL_PRESETS['default']`):
| Parameter | Value |
|-----------|-------|
| seq_len | 5 |
| latent_dim | 1 |
| n_gmm | 2 |
| lambda_energy | 0.1 |
| lambda_cov | 0.005 |
| epochs | 10 |
| batch_size | 256 |
| train_stride | 1 |

**Reference (community reproduction; no official code from authors):** https://github.com/danieltan07/dagmm (★422, no LICENSE, PyTorch port). Zong et al. (NEC Labs America)는 공식 코드를 공개하지 않음. README 명시: "My attempt at reproducing the paper". KDD99에서만 부분 검증 (F1 0.9607 vs 논문 0.9369).
대체 reproduction:
- [mperezcarrasco/PyTorch-DAGMM](https://github.com/mperezcarrasco/PyTorch-DAGMM) (★66, KDD99 F1 0.9432)
- [tnakae/DAGMM](https://github.com/tnakae/DAGMM) (★181, MIT, TensorFlow)

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

## New SOTA (2026-05-19 batch, 10-model expansion — 활성 7개 + 참고용 3개)

The 7 new SOTA models below (sections 16–22) cover 2023–2025 frontier work in MTS anomaly detection. All share the same execution interface (`fit(train_X, epoch_callback) / predict(test_X)`) and are launched via `run_sota_baseline_with_epoch_eval`. Hyperparameter presets live in `MODEL_PRESETS['default']` (single preset across all datasets). Verified by smoke-test: each builds, accepts an `(N, win_size, M)` input on GPU, and returns a (N_test,) score.

**실험 제외 모델 (23-25, AnomalyBERT + CAROTS + CrossAD, 2026-05-22)**: 3개 모두 단일 (dataset, model) 작업당 3–37시간 소요 — 39 데이터셋 pipeline에 비현실적인 timing outlier. **코드/구현은 그대로 보존** (`comparison/baselines/{anomalybert,carots,crossad}/`), `MODEL_PRESETS`와 `create_model()` 분기도 유지하여 코드 호출 가능. 단 `BASELINE_MODELS` / `NEW_SOTA_MODELS` 활성 리스트에서만 빠져, 표준 실험 파이프라인은 자동으로 학습을 건너뜀. 상세 사유는 sections 23-25 참조.

| # | Model | Venue | File | Loss | Distinct Feature |
|---|-------|-------|------|------|-------------------|
| 16 | TimesNet | ICLR'23 | timesnet/ | MSE recon | FFT-top-k period → 2D Inception conv |
| 17 | TFMAE | ICDE'24 | tfmae/ | KL adv-con | Dual temporal+frequency MAE |
| 18 | DCdetector | KDD'23 | dcdetector/ | Symmetric KL | Patch + In-patch dual attention |
| 19 | MEMTO | NeurIPS'23 | memto/ | Recon+Gather+Entropy | Memory module w/ K-means init (2-phase) |
| 20 | ModernTCN | ICLR'24 Spot | moderntcn/ | MSE recon | Large-kernel DW conv + dual ConvFFN |
| 21 | CATCH | ICLR'25 | catch/ | Time+Freq+ChDisc | Channel masking + freq reconstruction |
| 22 | NPSR | NeurIPS'23 | npsr/ | Point+Induction MSE | Nominality-conditioned scoring |
| **참고용 (실험 제외)** | | | | | |
| 23 | AnomalyBERT | ICLR'23 ML4IoT WS | anomalybert/ | Cross-entropy on degradation | 4-type degradation + Pre-LN BERT (sim 1 task **4시간+** → 제외) |
| 24 | CAROTS | ICML'25 | carots/ | Contrastive | 3-stage (CUTS_Plus → contrastive → centroid) (sim 1 task **3시간+** → 제외) |
| 25 | CrossAD | NeurIPS'25 | crossad/ | MSE recon | Multi-scale + learnable query library, Channel Independence (SWaT 1 모델 **37시간** → 제외) |

Notes:
- NPSR: `performer-pytorch` (with `--no-deps` install, 2026-05-22) provides the canonical Performer attention path. Code still preserves the `nn.MultiheadAttention` fallback when the package is missing.
- CrossAD / CATCH: **upstream code vendored verbatim** (2026-05-22). Earlier (2026-05-19) structural reconstructions were discarded — the new files in `comparison/baselines/{crossad,catch}/model.py` are direct ports of upstream `models/CrossAD/*` and `ts_benchmark/baselines/catch/{layers,utils,models}/*` respectively. Modifications: (a) device-agnostic operation (`register_buffer` / `x.device` instead of hard-coded `.cuda()`), (b) TAB-framework `from ts_benchmark...` imports replaced with in-file definitions. Previous structural-reconstruction implementations are backed up under `.trash/0522/comparison/baselines_{crossad,catch}_*_v1_structural.py`. (CrossAD는 활성 7개에서는 빠지고 25번 참고용 섹션에 위치.)



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

다음 3개 모델은 모델 코드와 wrapper, `MODEL_PRESETS` 진입점, `create_model()` 분기까지 모두 유지되어 있다. **실험 queue (`BASELINE_MODELS` / `NEW_SOTA_MODELS`)에서만 제외되어** 표준 파이프라인은 자동으로 학습을 건너뛴다. 직접 import해서 호출하는 것은 그대로 가능.

공통 제외 사유: 단일 (dataset, model) 작업당 3-37시간 소요로 39 데이터셋 pipeline에 비현실적인 timing outlier.

---

### 23. AnomalyBERT (ICLR 2023 ML4IoT Workshop) — 실험 제외

**File:** `baselines/anomalybert/{model.py, wrapper.py}` (유지)

**Paper:** "AnomalyBERT: Self-supervised Transformer for Time Series Anomaly Detection using Data Degradation Scheme" (ICLR 2023 ML4IoT Workshop)

**Description:** Self-supervised Pre-LN Transformer. Training 시 정상 시퀀스에 4가지 type의 degradation (soft/hard noise injection, point/contextual mutation)을 적용하여 classification objective로 학습. Inference 시 degradation 없이 anomaly probability 추정. BERT 아키텍처(encoder-only Transformer)만 차용한 random-init 모델로 LLM이 아님 — 즉 LLM 제외 정책과는 무관.

**Configuration:** win_size=512, d_model=512, n_heads=8, e_layers=6, dropout=0.1, lr=1e-4, degradation_ratio=0.15, batch_size=64, epochs=10.

**Reference:** [Jhryu30/AnomalyBERT](https://github.com/Jhryu30/AnomalyBERT) (MIT).

**Exclusion**: sim 1 task (8 features, 가장 작은 데이터셋)에서도 4시간+ 소요. 다른 모델 대비 5-10× 시간 outlier. 결과 백업: `.trash/0522/comparison/results_anomalybert_q3/`.

---

### 24. CAROTS (ICML 2025) — 실험 제외

**File:** `baselines/carots/{model.py, wrapper.py}` (유지)

**Paper:** "CAROTS: Contrastive Anomaly detection for Robust Online Time Series" (ICML 2025)

**Description:** 3-stage training pipeline — (1) CUTS_Plus causal-graph discoverer (variational causal structure learning), (2) positive/negative augmenter 학습 + contrastive loss, (3) centroid distance scoring. 본질적으로 multi-objective 3-stage로 학습 비용이 크다.

**Configuration:** win_size=100, d_model=256, n_heads=8, e_layers=3, patch_size=10, patch_stride=10, pos_aug_strength=0.1, neg_aug_strength=0.3, contrastive_margin=1.0, lr=1e-4, batch_size=64, epochs=10.

**Reference:** Upstream code reconstruction (paper + 저자 후속 발표 기반).

**Exclusion**: sim 1 task에 3시간+ 소요. CUTS_Plus 3-stage가 long-tail 비용 — 다른 SOTA 모델 대비 4-7× 시간 outlier. 결과 백업: `.trash/0522/comparison/results_carots_q3/`.

---

### 25. CrossAD (NeurIPS 2025) — 실험 제외

**File:** `baselines/crossad/{model.py, wrapper.py}` (유지)

**Paper:** "CrossAD" (NeurIPS 2025) — Cross-scale anomaly detection with query library

**Description:** Cross-scale anomaly detection with a learnable query library (Q tokens). Multi-scale patch embeddings (n_scales different patch sizes) are cross-attended-to by the queries; the resulting representations are processed by a self-attention encoder and projected back to the input via a linear reconstruction head. Anomaly score = per-timestep recon MSE. **Channel Independence** 패턴 사용 — `_forward`에서 `(bs, t, c) → (bs*c, t, 1)`로 변환하여 채널별 독립 학습. 이로 인해 effective batch가 입력 batch_size × n_features로 폭발적으로 증가.

**Configuration:** win_size=100, d_model=256, n_heads=8, e_layers=3, n_scales=3, query_lib_size=64, base_patch=4, lr=1e-4, batch_size=32 (originally 64, reduced 2026-05-22 for OOM investigation), epochs=10.

**Reference:** [decisionintelligence/CrossAD](https://github.com/decisionintelligence/CrossAD). Implementation: **upstream vendored verbatim** (2026-05-22). Four upstream files concatenated into `model.py`; only modifications are `register_buffer` for scale masks and `x.device` placement of inference-time zeros tensor. Upstream repo has no LICENSE — used for research comparison with attribution.

**Exclusion (memory-driven timing outlier)**:
- Channel Independence(`model.py:558-559`)로 effective batch = `bs × c_features`
- SWaT (c=45): bs=64 → effective 2880, 12 GB GPU OOM swap thrashing → **12s/iter** (실측) → 단일 epoch **37시간** 예상
- bs=32로 축소 (effective 1440): OOM 해소되어 0.38s/iter 회복 (32배 빨라짐, 실측)했으나 여전히 SWaT 단일 모델당 23시간으로 39 데이터셋 pipeline에 비현실적
- WaDi (c=127)에서는 bs=32도 effective 4064로 더 큰 문제 예상

결과 백업: `.trash/0522/crossad_removal/results_crossad_q3/` (sim 결과 보존).

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
   - Code: https://github.com/danieltan07/dagmm

### Other Methods

9. **MLPMixer:** Tolstikhin et al., "MLP-Mixer: An all-MLP Architecture for Vision", NeurIPS 2021
    - Paper: https://arxiv.org/abs/2105.01601
