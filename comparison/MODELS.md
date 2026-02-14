# Baseline Models

This document describes all baseline models used for comparison, based on [QuoVadisTAD](https://arxiv.org/abs/2405.02678) (ICML 2024) and [Anomaly Transformer](https://arxiv.org/abs/2110.02642) (ICLR 2022).

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
| epochs | 100 |
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
| epochs | 100 |
| learning_rate | 0.001 |
| batch_size | 512 |

**From QuoVadisTAD:** "Self-attention can capture long-range dependencies. Even with a single head and block, it's competitive with deeper models."

---

### 9. 1-Layer GCN-LSTM

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
| epochs | 100 |
| learning_rate | 0.001 |
| batch_size | 100 |

**From QuoVadisTAD:** "GCN captures spatial (sensor-to-sensor) relationships while LSTM captures temporal dynamics. Useful when sensors have known or learnable relationships."

---

## SOTA Comparison

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

**Configuration:**
| Parameter | Value |
|-----------|-------|
| win_size | 100 |
| d_model | 512 |
| n_heads | 8 |
| e_layers | 3 |
| d_ff | 512 |
| dropout | 0.0 |
| epochs | 10 |
| batch_size | 32 |
| k | 3.0 |
| temperature | 50.0 |

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

**Configuration:**
| Parameter | Value |
|-----------|-------|
| seq_len | 10 |
| d_ff | 16 |
| n_encoder_layers | 1 |
| n_decoder_layers | 1 |
| dropout | 0.1 |
| epochs | 100 |
| batch_size | 128 |

**Reference:** https://github.com/imperial-qore/TranAD

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

**Configuration:**
| Parameter | Value |
|-----------|-------|
| seq_len | 5 |
| hidden_dims | [input_dim//2, input_dim//4] |
| latent_dim | 32 |
| alpha | 0.5 |
| beta | 0.5 |
| epochs | 100 |
| batch_size | 256 |

**Reference:** https://github.com/manigalati/usad

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

**Configuration:**
| Parameter | Value |
|-----------|-------|
| seq_len | 5 |
| latent_dim | 1 |
| n_gmm | 2 |
| lambda_energy | 0.1 |
| lambda_cov | 0.005 |
| epochs | 100 |
| batch_size | 256 |

**Reference:** https://github.com/danieltan07/dagmm

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

**Configuration:**
| Parameter | Value |
|-----------|-------|
| seq_len | 5 |
| embed_dim | 64 |
| gnn_out_dim | 64 |
| n_heads | 1 |
| top_k | 5 |
| dropout | 0.2 |
| epochs | 100 |
| batch_size | 256 |

**Reference:** https://github.com/d-ailin/GDN

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

**Configuration:**
| Parameter | Value |
|-----------|-------|
| seq_len | 100 |
| hidden_dim | 100 |
| z_dim | 3 |
| n_layers | 1 |
| rnn_type | gru |
| beta | 0.01 |
| n_mc_samples | 10 |
| epochs | 100 |
| batch_size | 256 |

**Reference:** https://github.com/NetManAIOps/OmniAnomaly

---

### 16. MERLIN (ICDM 2020)

**File:** `baselines/merlin/model.py`

**Paper:** MERLIN: Parameter-Free Discovery of Arbitrary Length Anomalies in Massive Time Series Archives

**Description:** Matrix Profile-based anomaly detection using discord discovery. Finds subsequences most dissimilar from all others.

**Method:**
1. Compute Matrix Profile for each feature
   - For each subsequence, find distance to nearest neighbor
2. Combine profiles across features (sum/max/mean)
3. High Matrix Profile value = anomaly (discord)

**Key Concepts:**
- **Matrix Profile:** Vector of distances to nearest neighbor for each subsequence
- **Discord:** Subsequence with highest Matrix Profile value (most dissimilar)
- **Exclusion Zone:** Prevents trivial self-matches

**Configuration:**
| Parameter | Value |
|-----------|-------|
| subsequence_length | 50 |
| sample_rate | 0.1 |
| use_fast | True |
| multivariate_mode | sum |

**Note:** This is a non-learning method. Uses sampling-based fast approximation.

**Reference:** Matrix Profile Foundation (https://github.com/matrix-profile-foundation)

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

2. **Implement model:**
```python
# comparison/baselines/new_model/model.py

class NewModelBaseline:
    def __init__(self, **params):
        self.name = "New Model"

    def fit(self, train_data: np.ndarray) -> 'NewModelBaseline':
        # Train model
        return self

    def predict(self, test_data: np.ndarray) -> np.ndarray:
        # Return anomaly scores (higher = more anomalous)
        return scores

    # Optional: for neural models
    def save(self, save_dir: Path) -> None:
        ...

    def load(self, save_dir: Path) -> 'NewModelBaseline':
        ...
```

3. **Create `__init__.py`:**
```python
# comparison/baselines/new_model/__init__.py
from .model import NewModelBaseline
__all__ = ['NewModelBaseline']
```

4. **Register in `baselines/__init__.py`:**
```python
from .new_model import NewModelBaseline
```

5. **Add to `run_comparison.py`:**
```python
all_results["new_model"] = run_baseline(
    NewModelBaseline(**params),
    train_X, test_X, test_y, segments, "new_model"
)
```

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

9. **MERLIN:** Nakamura et al., "MERLIN: Parameter-Free Discovery of Arbitrary Length Anomalies in Massive Time Series Archives", ICDM 2020
   - Related: Matrix Profile Foundation (https://github.com/matrix-profile-foundation)

10. **MLPMixer:** Tolstikhin et al., "MLP-Mixer: An all-MLP Architecture for Vision", NeurIPS 2021
    - Paper: https://arxiv.org/abs/2105.01601
