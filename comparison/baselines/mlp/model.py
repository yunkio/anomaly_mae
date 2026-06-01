"""
1-Layer MLP Baseline for Time Series Anomaly Detection — paper-faithful (QuoVadisTAD).

Based on QuoVadisTAD (arXiv:2405.02678), upstream
`quovadis_tad/model_utils/model_def.py:204-241` with the `1_Layer_MLP` branch.

Architecture (forecasting mode, reconstruction=False, num_seq=5):
  Input (B, seq_len, n_features)
    → Dense(embedding_dim) per timestep          # Linear over feature axis
    → (no blocks, no activation, no dropout — upstream `Sequential([Dropout])`
       is built but `if model == "1_Layer_MLP": embedding = x` at model_def.py:227-228
       skips it entirely)
    → GlobalAveragePooling1D over time          # (B, embedding_dim)
    → Dense(n_features)                          # (B, n_features) — 1-step-ahead

Config from paper:
- embedding_dim: 32
- epochs: 200 (Domain B pipeline overrides via preset, intentional)
- learning_rate: 0.001
- input_sequence_length: 5
"""

import torch
import torch.nn as nn
from ..neural_base import NeuralBaselineBase


class MLPModel(nn.Module):
    """1-Layer MLP forecaster — upstream-faithful.

    Mirrors `model_def.py:204-241` (1_Layer_MLP branch). The upstream
    `blocks = Sequential([Dropout(rate=dropout_rate)])` is constructed but
    `model_def.py:227-228` (`if model == "1_Layer_MLP": embedding = x`) skips
    `blocks(x)` entirely — so there is NO Dropout, NO activation, NO Flatten
    on the forward path. Just two Dense layers around GAP over time.
    """

    def __init__(self, seq_len: int, n_features: int, embedding_dim: int, dropout: float = 0.0):
        super().__init__()
        # Per-timestep Dense (Linear applied independently over feature axis).
        self.embed = nn.Linear(n_features, embedding_dim)
        # Final Dense → 1-step-ahead forecast over n_features.
        self.output = nn.Linear(embedding_dim, n_features)
        # `dropout` retained in signature for backwards-compat with MODEL_PRESETS /
        # NeuralBaselineBase kwargs; intentionally NOT used (upstream-faithful).

    def forward(self, x):
        # x: (B, seq_len, n_features)
        h = self.embed(x)             # (B, seq_len, embedding_dim) — per-timestep Linear
        h = h.mean(dim=1)             # GAP over time → (B, embedding_dim)
        return self.output(h)         # (B, n_features) — 1-step-ahead forecast


class MLPBaseline(NeuralBaselineBase):
    """
    1-Layer MLP baseline for anomaly detection.

    This is the simplest neural baseline - a single hidden layer
    that maps the flattened input window to the next timestamp.
    """

    def __init__(
        self,
        seq_len: int = 5,
        embedding_dim: int = 32,
        dropout: float = 0.0,
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

    @property
    def name(self) -> str:
        return "1-Layer MLP"

    def _build_model(self) -> nn.Module:
        return MLPModel(
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
    model = MLPBaseline(
        seq_len=5,
        embedding_dim=32,
        epochs=10,  # Reduced for testing
        verbose=True
    )
    model.fit(train_X)

    # Test on subset
    scores = model.predict(test_X[:10000])

    print(f"\n1-Layer MLP Results (on 10000 test samples):")
    print(f"  Score range: [{scores.min():.6f}, {scores.max():.6f}]")
    print(f"  Score mean: {scores.mean():.6f}")
    print(f"  Score std: {scores.std():.6f}")
