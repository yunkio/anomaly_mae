"""
Single Transformer Block Baseline for Time Series Anomaly Detection — paper-faithful (QuoVadisTAD).

Based on QuoVadisTAD (arXiv:2405.02678), upstream
`quovadis_tad/model_utils/model_def.py:117-133` (`TransformerBlock`) and
`model_def.py:181-241` (`build_sequence_embedder`, `Single_Transformer_block` branch).

Architecture (forecasting mode, reconstruction=False, num_seq=5):
  Input (B, seq_len, n_features)
    → Dense(embedding_dim) per timestep          # Linear over feature axis
    → TransformerBlock (single — num_blocks=1 per yaml)
    → GlobalAveragePooling1D over time          # (B, embedding_dim)
    → Dense(n_features)                          # 1-step-ahead forecast

Key upstream-faithful properties of TransformerBlock:
- FFN is a SINGLE Dense layer with ReLU inside (`Dense(ff_dim, activation='relu')`
  at `model_def.py:121`), NOT a 2-layer FFN.
- ff_dim = embedding_dim (upstream `build_sequence_embedder:195` passes
  `embedding_dim` twice to `TransformerBlock(embed_dim, mha_heads, ff_dim=embedding_dim, ...)`).
- Post-Norm: `x = LayerNorm(x + dropout(sublayer_out))` (`:130`, `:133`).
- LayerNorm epsilon = 1e-6 (`layers.LayerNormalization(epsilon=1e-6)`).
- NO positional encoding (yaml `positional_encoding: False` per QuoVadis configs).
- NO output LayerNorm before final GAP+Dense.

Config from paper:
- embedding_dim: 128
- num_heads: 1
- num_blocks: 1
- dropout: 0.1
- epochs: 100
- learning_rate: 0.001
"""

import torch
import torch.nn as nn
from ..neural_base import NeuralBaselineBase


class TransformerBlock(nn.Module):
    """Single Transformer encoder block — upstream-faithful.

    Port of `quovadis_tad/model_utils/model_def.py:117-133`.

    - Multi-head self-attention (`layers.MultiHeadAttention(num_heads, key_dim=embed_dim)`).
    - FFN = single `Dense(ff_dim, activation='relu')` (NOT a 2-layer FFN).
    - Post-Norm: LayerNorm(x + dropout(sublayer)).
    - LayerNorm eps=1e-6.
    """

    def __init__(self, embedding_dim: int, num_heads: int = 1, dropout: float = 0.1):
        super().__init__()
        # Upstream `model_def.py:120` MultiHeadAttention(num_heads, key_dim=embed_dim).
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        # `:124-125` dropout1, dropout2 with rate=dropout_rate.
        self.dropout1 = nn.Dropout(dropout)
        # `:122` layernorm1 with epsilon=1e-6.
        self.norm1 = nn.LayerNorm(embedding_dim, eps=1e-6)
        # `:121` ffn = layers.Dense(ff_dim, activation='relu') — a SINGLE Dense
        # with the ReLU activation embedded inside. `build_sequence_embedder:195`
        # passes ff_dim=embedding_dim.
        self.ffn = nn.Linear(embedding_dim, embedding_dim)
        self.ffn_activation = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        # `:123` layernorm2 with epsilon=1e-6.
        self.norm2 = nn.LayerNorm(embedding_dim, eps=1e-6)

    def forward(self, x):
        # x: (B, seq_len, embedding_dim)
        # Mirrors `TransformerBlock.call` at `model_def.py:127-133` step-by-step.
        attn_output, _ = self.attention(x, x, x)             # `:128`
        attn_output = self.dropout1(attn_output)             # `:129`
        out1 = self.norm1(x + attn_output)                   # `:130` Post-Norm
        ffn_output = self.ffn_activation(self.ffn(out1))     # `:131` Dense(...,relu)
        ffn_output = self.dropout2(ffn_output)               # `:132`
        return self.norm2(out1 + ffn_output)                 # `:133` Post-Norm


class TransformerModel(nn.Module):
    """Single Transformer Block forecaster — upstream-faithful.

    Mirrors `build_sequence_embedder` (`model_def.py:181-241`,
    `Single_Transformer_block` branch).

    - NO positional encoding (yaml `positional_encoding: False`).
    - NO output LayerNorm before the final GAP + Dense.
    """

    def __init__(self, seq_len: int, n_features: int, embedding_dim: int, num_heads: int = 1, dropout: float = 0.1):
        super().__init__()
        # Per-timestep Linear (upstream `Dense(units=embedding_dim)` at `:219`).
        self.embedding = nn.Linear(n_features, embedding_dim)
        # num_blocks=1 per `model_configs` yaml.
        self.transformer_block = TransformerBlock(embedding_dim, num_heads, dropout)
        # NO positional embedding (yaml positional_encoding: False).
        # NO output LayerNorm (upstream `model_def.py:181-241` has none before GAP).
        self.output = nn.Linear(embedding_dim, n_features)

    def forward(self, x):
        # x: (B, seq_len, n_features)
        x = self.embedding(x)                     # (B, seq_len, embedding_dim)
        x = self.transformer_block(x)
        x = x.mean(dim=1)                         # GAP over time → (B, embedding_dim)
        return self.output(x)                     # (B, n_features) — 1-step-ahead forecast


class TransformerBaseline(NeuralBaselineBase):
    """
    Single Transformer Block baseline for anomaly detection.

    Uses self-attention to capture temporal dependencies
    in the input sequence.
    """

    def __init__(
        self,
        seq_len: int = 5,
        embedding_dim: int = 128,
        num_heads: int = 1,
        num_blocks: int = 1,
        dropout: float = 0.1,
        lr: float = 0.001,
        weight_decay: float = 0.0001,
        batch_size: int = 512,
        epochs: int = 10,
        train_stride: int = 3,
        device: str = None,
        verbose: bool = True,
        **extra_kwargs,
    ):
        super().__init__(
            seq_len=seq_len,
            embedding_dim=embedding_dim,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            epochs=epochs,
            train_stride=train_stride,
            device=device,
            verbose=verbose,
            **extra_kwargs,
        )
        self.num_heads = num_heads
        # `num_blocks` paper preset 기록용 — 현재 TransformerModel 은 single block.
        # multi-block 확장 시 _build_model 에서 self._num_blocks_preset 사용 가능.
        self._num_blocks_preset = int(num_blocks)

    @property
    def name(self) -> str:
        return "Single Transformer Block"

    def _build_model(self) -> nn.Module:
        return TransformerModel(
            seq_len=self.seq_len,
            n_features=self.n_features,
            embedding_dim=self.embedding_dim,
            num_heads=self.num_heads,
            dropout=self.dropout
        )

    def __repr__(self) -> str:
        return f"TransformerBaseline(seq_len={self.seq_len}, embedding_dim={self.embedding_dim}, num_heads={self.num_heads})"


if __name__ == "__main__":
    # Test the baseline
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.wadi_loader import WaDiLoader

    data_path = Path(__file__).parent.parent.parent / "dataset/WaDi/WADI.A1_9 Oct 2017/WADI_attackdata_preprocessed.csv"
    loader = WaDiLoader(data_path=str(data_path), train_ratio=0.5)
    loader.load()

    train_X, train_y = loader.get_train_data()
    test_X, test_y = loader.get_test_data()

    # Test with fewer epochs for quick validation
    model = TransformerBaseline(
        seq_len=5,
        embedding_dim=128,
        num_heads=1,
        epochs=10,  # Reduced for testing
        verbose=True
    )
    model.fit(train_X)

    # Test on subset
    scores = model.predict(test_X[:10000])

    print(f"\nSingle Transformer Block Results (on 10000 test samples):")
    print(f"  Score range: [{scores.min():.6f}, {scores.max():.6f}]")
    print(f"  Score mean: {scores.mean():.6f}")
    print(f"  Score std: {scores.std():.6f}")
