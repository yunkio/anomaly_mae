"""
TranAD - Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data

Based on: "TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data"
Paper: VLDB 2022, https://arxiv.org/abs/2201.07284
Reference: https://github.com/imperial-qore/TranAD

Architecture:
- Transformer encoder-decoder with two-phase reconstruction
- Phase 1: Standard reconstruction without conditioning
- Phase 2: Reconstruction conditioned on Phase 1 reconstruction error
- Adversarial training between the two phases

Anomaly Score: Combines reconstruction errors from both phases
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple
from pathlib import Path
import json
import time
import sys


class PositionalEncoding(nn.Module):
    """Positional encoding with dropout."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (seq_len, batch, d_model)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TranADModel(nn.Module):
    """
    TranAD Model with two-phase transformer decoder.

    Architecture:
    1. Encoder processes input concatenated with conditioning (zeros or reconstruction error)
    2. Two separate decoders for the two phases
    3. Final FCN projects to original feature space
    """

    def __init__(
        self,
        n_features: int,
        seq_len: int = 10,
        d_model: int = None,
        n_heads: int = None,
        d_ff: int = 16,
        n_encoder_layers: int = 1,
        n_decoder_layers: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()

        self.n_features = n_features
        self.seq_len = seq_len

        # d_model should be 2 * n_features for concatenation
        self.d_model = d_model if d_model is not None else 2 * n_features
        # n_heads should divide d_model evenly
        self.n_heads = n_heads if n_heads is not None else max(1, n_features)

        # Ensure n_heads divides d_model
        while self.d_model % self.n_heads != 0:
            self.n_heads -= 1
            if self.n_heads < 1:
                self.n_heads = 1
                break

        self.pos_encoder = PositionalEncoding(self.d_model, dropout, seq_len)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)

        # Two transformer decoders for two phases
        decoder_layer1 = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_decoder1 = nn.TransformerDecoder(decoder_layer1, num_layers=n_decoder_layers)

        decoder_layer2 = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_decoder2 = nn.TransformerDecoder(decoder_layer2, num_layers=n_decoder_layers)

        # Input projection to d_model//2 (will be concatenated with conditioning)
        self.input_projection = nn.Linear(n_features, self.d_model // 2)

        # Final projection to original feature space
        self.fcn = nn.Sequential(
            nn.Linear(self.d_model, n_features),
            nn.Sigmoid()
        )

    def encode(self, src: torch.Tensor, condition: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode source with conditioning.

        Args:
            src: Input tensor (seq_len, batch, n_features)
            condition: Conditioning tensor (seq_len, batch, n_features) - zeros or reconstruction error
            tgt: Target tensor for decoder (seq_len, batch, n_features)

        Returns:
            tgt_embedded: Target with d_model dimension
            memory: Encoder output
        """
        # Project input to d_model//2
        src_proj = self.input_projection(src)  # (seq_len, batch, d_model//2)
        cond_proj = self.input_projection(condition)  # (seq_len, batch, d_model//2)

        # Concatenate source and condition
        src_combined = torch.cat([src_proj, cond_proj], dim=-1)  # (seq_len, batch, d_model)

        # Scale and add positional encoding
        src_combined = src_combined * math.sqrt(self.d_model // 2)
        src_combined = self.pos_encoder(src_combined)

        # Encode
        memory = self.transformer_encoder(src_combined)

        # Prepare target (repeat to d_model)
        tgt_proj = self.input_projection(tgt)
        tgt_embedded = torch.cat([tgt_proj, tgt_proj], dim=-1)  # (seq_len, batch, d_model)

        return tgt_embedded, memory

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with two-phase decoding.

        Args:
            src: Input tensor (batch, seq_len, n_features)
            tgt: Target tensor (batch, seq_len, n_features)

        Returns:
            x1: Phase 1 reconstruction
            x2: Phase 2 reconstruction (conditioned on phase 1 error)
        """
        # Transpose to (seq_len, batch, n_features) for transformer
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)

        # Phase 1: Reconstruct without conditioning (use zeros)
        condition1 = torch.zeros_like(src)
        tgt_embedded1, memory1 = self.encode(src, condition1, tgt)
        dec_out1 = self.transformer_decoder1(tgt_embedded1, memory1)
        x1 = self.fcn(dec_out1)  # (seq_len, batch, n_features)

        # Phase 2: Reconstruct with conditioning (reconstruction error)
        condition2 = (x1 - src) ** 2  # Squared reconstruction error
        tgt_embedded2, memory2 = self.encode(src, condition2, tgt)
        dec_out2 = self.transformer_decoder2(tgt_embedded2, memory2)
        x2 = self.fcn(dec_out2)  # (seq_len, batch, n_features)

        # Transpose back to (batch, seq_len, n_features)
        x1 = x1.transpose(0, 1)
        x2 = x2.transpose(0, 1)

        return x1, x2


class TranADBaseline:
    """
    TranAD baseline wrapper for time series anomaly detection.
    """

    def __init__(
        self,
        seq_len: int = 10,
        d_ff: int = 16,
        n_encoder_layers: int = 1,
        n_decoder_layers: int = 1,
        dropout: float = 0.1,
        lr: float = 0.001,
        batch_size: int = 128,
        epochs: int = 10,
        train_stride: int = 3,
        device: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Args:
            seq_len: Window size
            d_ff: Feed-forward dimension
            n_encoder_layers: Number of encoder layers
            n_decoder_layers: Number of decoder layers
            dropout: Dropout rate
            lr: Learning rate
            batch_size: Training batch size
            epochs: Number of training epochs
            train_stride: Stride for training windows (default 11)
            device: 'cuda' or 'cpu'
            verbose: Print training progress
        """
        self.seq_len = seq_len
        self.d_ff = d_ff
        self.n_encoder_layers = n_encoder_layers
        self.n_decoder_layers = n_decoder_layers
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_stride = train_stride
        self.verbose = verbose

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model: Optional[TranADModel] = None
        self.n_features: int = 0
        self.train_loss_history = []

    @property
    def name(self) -> str:
        return "TranAD"

    def _create_windows(self, data: np.ndarray, stride: int = 1) -> np.ndarray:
        """Create sliding windows."""
        n_samples = len(data)
        n_windows = (n_samples - self.seq_len) // stride + 1

        windows = np.zeros((n_windows, self.seq_len, data.shape[1]), dtype=np.float32)
        for idx, i in enumerate(range(0, n_samples - self.seq_len + 1, stride)):
            if idx >= n_windows:
                break
            windows[idx] = data[i:i + self.seq_len]

        return windows

    def fit(self, train_data: np.ndarray, epoch_callback=None) -> 'TranADBaseline':
        """
        Train TranAD on training data.

        Args:
            train_data: Training data (n_samples, n_features)
            epoch_callback: Optional callback(model, epoch) called after each epoch

        Returns:
            self
        """
        self.n_features = train_data.shape[1]

        # Build model
        self.model = TranADModel(
            n_features=self.n_features,
            seq_len=self.seq_len,
            d_ff=self.d_ff,
            n_encoder_layers=self.n_encoder_layers,
            n_decoder_layers=self.n_decoder_layers,
            dropout=self.dropout
        ).to(self.device)

        if self.verbose:
            print(f"{self.name} training:")
            print(f"  Device: {self.device}")
            print(f"  Input shape: ({self.seq_len}, {self.n_features})")
            print(f"  d_model: {self.model.d_model}, n_heads: {self.model.n_heads}")
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"  Total parameters: {total_params:,}")

        # Create windows with stride for training
        windows = self._create_windows(train_data, stride=self.train_stride)
        if self.verbose:
            print(f"  Training windows: {len(windows):,} (stride={self.train_stride})")

        # Create DataLoader
        dataset = torch.utils.data.TensorDataset(torch.FloatTensor(windows))
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )

        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self.train_loss_history = []
        self.model.train()

        start_time = time.time()

        for epoch in range(self.epochs):
            epoch_start = time.time()
            epoch_losses = []

            for batch_idx, (batch_x,) in enumerate(dataloader):
                batch_x = batch_x.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                x1, x2 = self.model(batch_x, batch_x)

                # TranAD loss: minimize both reconstruction errors
                # with focus parameter that shifts emphasis during training
                focus = epoch / self.epochs
                loss1 = criterion(x1, batch_x)
                loss2 = criterion(x2, batch_x)
                loss = (1 - focus) * loss1 + focus * loss2

                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())

            avg_loss = np.mean(epoch_losses)
            self.train_loss_history.append(avg_loss)

            epoch_time = time.time() - epoch_start
            elapsed = time.time() - start_time
            remaining = elapsed / (epoch + 1) * (self.epochs - epoch - 1)

            if self.verbose:
                progress = (epoch + 1) / self.epochs * 100
                print(f"\r  Training: [{epoch + 1}/{self.epochs}] {progress:5.1f}% | "
                      f"Loss: {avg_loss:.6f} | "
                      f"Time: {epoch_time:.1f}s/epoch | "
                      f"ETA: {remaining:.0f}s", end="")
                sys.stdout.flush()

            if epoch_callback is not None:
                epoch_callback(self, epoch + 1)
                self.model.train()

        if self.verbose:
            total_time = time.time() - start_time
            print(f"\n  Training complete: {total_time:.1f}s total, Final loss: {self.train_loss_history[-1]:.6f}")

        return self

    def predict(self, test_data: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores for test data.

        Args:
            test_data: Test data (n_samples, n_features)

        Returns:
            Anomaly scores for each timestamp
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        self.model.eval()

        # Create windows
        windows = self._create_windows(test_data)
        n_windows = len(windows)

        # Compute scores in batches
        all_scores = []
        n_batches = (n_windows + self.batch_size - 1) // self.batch_size
        start_time = time.time()

        if self.verbose:
            print(f"  Inference: {n_windows:,} windows in {n_batches} batches")

        with torch.no_grad():
            for batch_idx, i in enumerate(range(0, n_windows, self.batch_size)):
                batch_x = torch.FloatTensor(windows[i:i + self.batch_size]).to(self.device)

                x1, x2 = self.model(batch_x, batch_x)

                # Anomaly score: combine both reconstruction errors
                # Take max of the two errors per sample
                err1 = torch.mean((x1 - batch_x) ** 2, dim=(1, 2))  # (batch,)
                err2 = torch.mean((x2 - batch_x) ** 2, dim=(1, 2))  # (batch,)
                scores = (err1 + err2) / 2

                all_scores.append(scores.cpu().numpy())

                if self.verbose and (batch_idx + 1) % max(1, n_batches // 10) == 0:
                    progress = (batch_idx + 1) / n_batches * 100
                    elapsed = time.time() - start_time
                    print(f"\r  Inference: [{batch_idx + 1}/{n_batches}] {progress:5.1f}% | "
                          f"Elapsed: {elapsed:.1f}s", end="")
                    sys.stdout.flush()

        if self.verbose:
            total_time = time.time() - start_time
            print(f"\n  Inference complete: {total_time:.1f}s")

        window_scores = np.concatenate(all_scores)

        # Map window scores to point-level
        n_samples = len(test_data)
        scores = np.zeros(n_samples, dtype=np.float32)

        for i, score in enumerate(window_scores):
            target_idx = i + self.seq_len - 1
            if target_idx < n_samples:
                scores[target_idx] = score

        # Fill first timestamps
        if len(window_scores) > 0:
            for i in range(min(self.seq_len - 1, n_samples)):
                scores[i] = window_scores[0]

        return scores

    def save(self, save_dir: Path) -> None:
        """Save model weights and config."""
        if self.model is None:
            raise RuntimeError("No model to save.")

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), save_dir / "model.pt")

        config = {
            "seq_len": self.seq_len,
            "d_ff": self.d_ff,
            "n_encoder_layers": self.n_encoder_layers,
            "n_decoder_layers": self.n_decoder_layers,
            "n_features": self.n_features,
            "model_name": self.name,
        }
        with open(save_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

    def load(self, save_dir: Path) -> 'TranADBaseline':
        """Load model weights and config."""
        save_dir = Path(save_dir)

        with open(save_dir / "config.json", 'r') as f:
            config = json.load(f)

        self.seq_len = config["seq_len"]
        self.d_ff = config["d_ff"]
        self.n_encoder_layers = config["n_encoder_layers"]
        self.n_decoder_layers = config["n_decoder_layers"]
        self.n_features = config["n_features"]

        self.model = TranADModel(
            n_features=self.n_features,
            seq_len=self.seq_len,
            d_ff=self.d_ff,
            n_encoder_layers=self.n_encoder_layers,
            n_decoder_layers=self.n_decoder_layers,
            dropout=self.dropout
        ).to(self.device)
        self.model.load_state_dict(torch.load(save_dir / "model.pt", map_location=self.device))
        self.model.eval()

        return self

    def __repr__(self) -> str:
        return f"TranADBaseline(seq_len={self.seq_len}, d_ff={self.d_ff})"
