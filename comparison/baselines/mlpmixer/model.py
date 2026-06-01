"""
Single Block MLPMixer Baseline for Time Series Anomaly Detection — paper-faithful (QuoVadisTAD).

Based on QuoVadisTAD (arXiv:2405.02678), upstream
`quovadis_tad/model_utils/model_def.py:77-114` (`MLPMixerLayer`) and
`model_def.py:181-241` (`build_sequence_embedder`, `Single_block_MLPMixer` branch).

Architecture (forecasting mode, reconstruction=False, num_seq=5):
  Input (B, seq_len, n_features)
    → Dense(embedding_dim) per timestep          # Linear over feature axis
    → MLPMixerBlock (single — num_blocks=1 per yaml)
    → GlobalAveragePooling1D over time          # (B, embedding_dim)
    → Dense(n_features)                          # 1-step-ahead forecast

Key upstream-faithful properties of MLPMixerLayer:
- Single `self.normalize` LayerNorm instance called TWICE (γ/β shared) — see
  `model_def.py:95` and call sites at `:99` and `:109`.
- `mlp1` token-mixing: `Dense(num_patches, gelu) → Dense(num_patches) → Dropout`.
  The first Dense embeds the activation inside (`activation='gelu'` kwarg) and
  there is no intermediate Dropout between the two Denses (single trailing Dropout).
- `mlp2` channel-mixing: `Dense(num_patches, gelu) → Dense(hidden_units) → Dropout`.
  ★ NOTE the intermediate dim is `num_patches=seq_len=5`, NOT `hidden_units` — this
  is a 5-dim BOTTLENECK on the channel-mixing path. Matches upstream `model_def.py:90`
  verbatim.
- No output LayerNorm before the final GAP+Dense (`build_sequence_embedder:181-241`
  has no extra LN there).
- LayerNorm epsilon = 1e-6 (upstream `layers.LayerNormalization(epsilon=1e-6)`).

Config from paper:
- embedding_dim: 128
- num_blocks: 1
- dropout: 0.1
- epochs: 100
- learning_rate: 0.0002
"""

import torch
import torch.nn as nn
from ..neural_base import NeuralBaselineBase


class MLPMixerBlock(nn.Module):
    """Single MLPMixer block — upstream-faithful.

    Port of `quovadis_tad/model_utils/model_def.py:77-114` (`MLPMixerLayer`).

    - Single LayerNorm instance reused for both norm positions (γ/β shared).
    - `mlp1` (token mixing): two Linears with GELU between, terminal Dropout.
      No intermediate Dropout (upstream `Sequential([Dense(gelu), Dense, Dropout])`).
    - `mlp2` (channel mixing): Linear(embed → seq_len) → GELU → Linear(seq_len → embed)
      → Dropout. ★ Intermediate dim = `seq_len` (paper bottleneck).
    """

    def __init__(self, seq_len: int, embedding_dim: int, dropout: float = 0.1):
        super().__init__()
        # Single LayerNorm instance (γ/β shared) — matches `model_def.py:95`.
        self.normalize = nn.LayerNorm(embedding_dim, eps=1e-6)

        # mlp1: token mixing. Operates on transposed (B, embedding_dim, seq_len).
        # Upstream: Sequential([Dense(num_patches, gelu), Dense(num_patches), Dropout])
        # → in PyTorch: Linear, GELU, Linear, Dropout (single trailing dropout, no mid-dropout).
        self.mlp1 = nn.Sequential(
            nn.Linear(seq_len, seq_len),
            nn.GELU(),
            nn.Linear(seq_len, seq_len),
            nn.Dropout(dropout),
        )

        # mlp2: channel mixing. Operates on (B, seq_len, embedding_dim).
        # Upstream `model_def.py:88-93`: Sequential([Dense(num_patches, gelu),
        #   Dense(hidden_units), Dropout]) — note Dense(num_patches) is the FIRST layer,
        #   creating a 5-dim BOTTLENECK before expanding back to hidden_units.
        self.mlp2 = nn.Sequential(
            nn.Linear(embedding_dim, seq_len),      # bottleneck to num_patches (=seq_len)
            nn.GELU(),
            nn.Linear(seq_len, embedding_dim),      # expand back to hidden_units
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B, seq_len, embedding_dim)
        # Mirrors `MLPMixerLayer.call` at `model_def.py:97-114` step-by-step.
        y = self.normalize(x)                        # `:99` x = self.normalize(inputs)
        y = y.transpose(1, 2)                        # `:101` transpose to (B, E, seq_len)
        y = self.mlp1(y)
        y = y.transpose(1, 2)                        # `:105` back to (B, seq_len, E)
        x = x + y                                    # `:107` skip connection
        y = self.normalize(x)                        # `:109` ★ SAME instance reused
        y = self.mlp2(y)
        x = x + y                                    # `:113` skip connection
        return x


class MLPMixerModel(nn.Module):
    """Single Block MLPMixer forecaster — upstream-faithful.

    Mirrors `build_sequence_embedder` (`model_def.py:181-241`,
    `Single_block_MLPMixer` branch). No positional encoding (yaml
    positional_encoding: False), no output LayerNorm.
    """

    def __init__(self, seq_len: int, n_features: int, embedding_dim: int, dropout: float = 0.1):
        super().__init__()
        # Per-timestep Linear (upstream `Dense(units=embedding_dim)` at `:219`).
        self.embedding = nn.Linear(n_features, embedding_dim)
        # num_blocks=1 per `model_configs` yaml.
        self.mixer_block = MLPMixerBlock(seq_len, embedding_dim, dropout)
        # NO output LayerNorm — upstream `model_def.py:181-241` has none before GAP.
        self.output = nn.Linear(embedding_dim, n_features)

    def forward(self, x):
        # x: (B, seq_len, n_features)
        x = self.embedding(x)            # (B, seq_len, embedding_dim)
        x = self.mixer_block(x)
        x = x.mean(dim=1)                # GAP over time → (B, embedding_dim)
        return self.output(x)            # (B, n_features) — 1-step-ahead forecast


class MLPMixerBaseline(NeuralBaselineBase):
    """
    Single Block MLPMixer baseline for anomaly detection.

    MLPMixer is an all-MLP architecture that mixes information
    across both token (time) and channel (feature) dimensions.
    """

    def __init__(
        self,
        seq_len: int = 5,
        embedding_dim: int = 128,
        num_blocks: int = 1,
        dropout: float = 0.1,
        lr: float = 0.0002,
        weight_decay: float = 0.0001,
        batch_size: int = 512,
        epochs: int = 10,
        train_stride: int = 3,
        device: str = None,
        verbose: bool = True,
        **extra_kwargs,
    ):
        # `num_blocks` 는 paper preset 기록용 — 현재 MLPMixerModel 은 single-block.
        # multi-block 확장 시 _build_model 에서 self.num_blocks 사용 가능.
        self._num_blocks_preset = int(num_blocks)
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

    @property
    def name(self) -> str:
        return "Single Block MLPMixer"

    def _build_model(self) -> nn.Module:
        return MLPMixerModel(
            seq_len=self.seq_len,
            n_features=self.n_features,
            embedding_dim=self.embedding_dim,
            dropout=self.dropout
        )


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
    model = MLPMixerBaseline(
        seq_len=5,
        embedding_dim=128,
        epochs=10,  # Reduced for testing
        verbose=True
    )
    model.fit(train_X)

    # Test on subset
    scores = model.predict(test_X[:10000])

    print(f"\nSingle Block MLPMixer Results (on 10000 test samples):")
    print(f"  Score range: [{scores.min():.6f}, {scores.max():.6f}]")
    print(f"  Score mean: {scores.mean():.6f}")
    print(f"  Score std: {scores.std():.6f}")
