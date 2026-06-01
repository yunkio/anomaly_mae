"""
TranAD - Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data

Based on: "TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data"
Paper: VLDB 2022, https://arxiv.org/abs/2201.07284
Reference: https://github.com/imperial-qore/TranAD (BSD-3-Clause)

Architecture (reference-faithful, Phase 2 porting 2026-05-24):
- PositionalEncoding: stride=1 div_term with `pe += sin + cos` overlay (upstream dlutils.py)
- TranADModel.encode: raw `torch.cat((src, c), dim=2)` — NO learnable input projection
- TranADModel.forward: expects inputs as (seq, batch, feats); tgt is the *last timestep only*
  (caller supplies `elem = window[-1].unsqueeze(0)`)
- Two-phase: Phase 1 c=zeros, Phase 2 c=(x1-src)**2 (self-conditioning by squared error)
- Output: Sigmoid head (Path A — driver MinMax to [0, 1])

Training (reference-faithful):
- AdamW(lr, weight_decay=1e-5) + StepLR(step_size=5, gamma=0.9)
- Loss: n = epoch + 1; l1 = (1/n) * MSE(x1, elem) + (1 - 1/n) * MSE(x2, elem)
  Upstream main.py: fresh model has `epoch=-1` in load_model() → outer loop starts at `e=0`
  → backprop(e=0) sets `n = e + 1 = 1` → coeff(L1)=1.0, coeff(L2)=0.0 at first epoch
  Our 0-indexed Python loop (epoch=0..epochs-1) → `n = epoch + 1` matches upstream byte-for-byte.
  - epoch=0=>n=1 (1.0/0.0), epoch=1=>n=2 (0.5/0.5), epoch=2=>n=3 (1/3, 2/3), ..., epoch=9=>n=10 (0.1/0.9)
  - 1/n SHRINKS over epochs (NOT growing focus); L2 (Phase 2) emphasis INCREASES from 0 → 0.9

Scoring (reference-faithful):
- Use Phase 2 output only (`x2`)
- MSE on the last-timestep only, mean over features → per-window scalar
- Window i → timestamp i + seq_len - 1; first seq_len-1 forward-filled
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from comparison.segment_utils import compute_segment_safe_window_indices
import math
from typing import Optional, Tuple
from pathlib import Path
import json
import time
import sys


class PositionalEncoding(nn.Module):
    """Positional encoding — upstream `imperial-qore/TranAD` dlutils.py form.

    Uses stride-1 div_term and `pe += sin; pe += cos` overlay (non-standard,
    but matches upstream byte-for-byte to preserve training dynamics).
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Upstream stride=1 div_term — covers ALL dims (NOT just even).
        div_term = torch.exp(torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model))
        # Upstream overlay: pe accumulates both sin and cos into every dim.
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (seq_len, batch, d_model)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TranADModel(nn.Module):
    """
    TranAD Model — reference-faithful port.

    Architecture (matches upstream src/models.py:TranAD):
    1. encode(src, c, tgt):
         src = cat([src, c], dim=2)          # raw concat → (seq, batch, 2*feats)
         src = src * sqrt(n_feats)
         src = pos_encoder(src)
         memory = transformer_encoder(src)
         tgt = tgt.repeat(1, 1, 2)            # (seq, batch, 2*feats)
         returns (tgt, memory)
    2. Phase 1: c=zeros → decoder1 → fcn → x1
    3. Phase 2: c=(x1-src)**2 → decoder2 → fcn → x2
    4. fcn = Linear(2*feats, feats) + Sigmoid
    """

    def __init__(
        self,
        n_features: int,
        seq_len: int = 10,
        d_model: Optional[int] = None,
        n_heads: Optional[int] = None,
        d_ff: int = 16,
        n_encoder_layers: int = 1,
        n_decoder_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.n_features = n_features
        self.seq_len = seq_len

        # Upstream: d_model = 2 * feats, nhead = feats (always divides).
        # We expose d_model/n_heads overrides but default to upstream values.
        self.d_model = d_model if d_model is not None else 2 * n_features
        self.n_heads = n_heads if n_heads is not None else max(1, n_features)

        # Safety: ensure n_heads divides d_model (upstream's default always satisfies this).
        while self.d_model % self.n_heads != 0:
            self.n_heads -= 1
            if self.n_heads < 1:
                self.n_heads = 1
                break

        # PositionalEncoding sized to d_model = 2 * n_features.
        self.pos_encoder = PositionalEncoding(self.d_model, dropout, seq_len)

        # Transformer encoder (1 layer in upstream).
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)

        # Two separate decoders (1 layer each in upstream).
        decoder_layer1 = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=False,
        )
        self.transformer_decoder1 = nn.TransformerDecoder(decoder_layer1, num_layers=n_decoder_layers)

        decoder_layer2 = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=False,
        )
        self.transformer_decoder2 = nn.TransformerDecoder(decoder_layer2, num_layers=n_decoder_layers)

        # NOTE: NO `self.input_projection` — upstream uses raw concat (see encode()).

        # Final projection back to feature space + Sigmoid (assumes input in [0, 1]).
        self.fcn = nn.Sequential(
            nn.Linear(self.d_model, n_features),
            nn.Sigmoid(),
        )

    def encode(self, src: torch.Tensor, condition: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Upstream-faithful encode.

        Args:
            src: (seq_len, batch, n_features)
            condition: (seq_len, batch, n_features) — zeros (Phase 1) or squared error (Phase 2)
            tgt: (1, batch, n_features) — last-timestep only (caller's responsibility)

        Returns:
            tgt_embedded: (1, batch, d_model)
            memory: (seq_len, batch, d_model)
        """
        # Raw concat (no projection) → (seq_len, batch, 2 * n_features)
        src = torch.cat((src, condition), dim=2)
        # Upstream scales by sqrt(n_feats) (NOT sqrt(d_model)).
        src = src * math.sqrt(self.n_features)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        # Upstream: tgt.repeat along last dim → matches d_model = 2 * feats.
        tgt = tgt.repeat(1, 1, 2)
        return tgt, memory

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Two-phase forward (upstream-faithful).

        Args:
            src: (seq_len, batch, n_features)         — full window
            tgt: (1, batch, n_features)               — typically `src[-1:].clone()`

        Returns:
            x1: (1, batch, n_features) — Phase 1 reconstruction of last timestep
            x2: (1, batch, n_features) — Phase 2 reconstruction (self-conditioned)
        """
        # Phase 1 — c = zeros
        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
        # Phase 2 — c = squared reconstruction error
        c = (x1 - src) ** 2  # broadcast: x1 is (1, B, F), src is (seq, B, F) — produces (seq, B, F)
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
        return x1, x2


class TranADBaseline:
    """
    TranAD baseline wrapper — pipeline-compatible (`fit`/`predict`/`save`/`load`).

    Path A: expects driver-normalized [0, 1] input (Sigmoid head assumes this).
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
        train_stride: int = 1,
        device: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        Args:
            seq_len: Window size (default 10, matches upstream n_window).
            d_ff: Feed-forward dimension (default 16, matches upstream).
            n_encoder_layers: Number of encoder layers (default 1, matches upstream).
            n_decoder_layers: Number of decoder layers (default 1, matches upstream).
            dropout: Dropout rate (default 0.1, matches upstream).
            lr: Learning rate (default 0.001; upstream uses dataset-dependent lr).
            batch_size: Training batch size (default 128, matches upstream).
            epochs: Number of training epochs (default 10; pipeline contract — upstream paper uses 5).
            train_stride: Stride for training windows (default 1, matches upstream stride=1).
            device: 'cuda' or 'cpu' (auto-detect if None).
            verbose: Print training progress.
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
        """Create sliding windows (no left padding; first seq_len-1 timestamps handled in predict)."""
        n_samples = len(data)
        n_windows = (n_samples - self.seq_len) // stride + 1

        windows = np.zeros((n_windows, self.seq_len, data.shape[1]), dtype=np.float32)
        for idx, i in enumerate(range(0, n_samples - self.seq_len + 1, stride)):
            if idx >= n_windows:
                break
            windows[idx] = data[i:i + self.seq_len]

        return windows

    def fit(self, train_data: np.ndarray, epoch_callback=None, train_segments=None) -> 'TranADBaseline':
        """Train TranAD on training data.

        Loss formula (upstream-faithful, main.py:`n = epoch + 1`):
            Upstream's `epoch` arg passed to backprop() starts at 0 for fresh training
            (load_model sets epoch=-1 → outer loop `range(epoch+1, ...)` begins at 0).
            Our 0-indexed Python loop maps 1:1, so we also use `n = epoch + 1`.
            loss = (1/n) * MSE(x1, elem) + (1 - 1/n) * MSE(x2, elem)

        Optimizer/scheduler (upstream-faithful):
            AdamW(lr, weight_decay=1e-5) + StepLR(step_size=5, gamma=0.9, per-epoch step).
        """
        self.n_features = train_data.shape[1]

        # Build model (no input_projection — uses raw concat in encode).
        self.model = TranADModel(
            n_features=self.n_features,
            seq_len=self.seq_len,
            d_ff=self.d_ff,
            n_encoder_layers=self.n_encoder_layers,
            n_decoder_layers=self.n_decoder_layers,
            dropout=self.dropout,
        ).to(self.device)

        if self.verbose:
            print(f"{self.name} training:")
            print(f"  Device: {self.device}")
            print(f"  Input shape: ({self.seq_len}, {self.n_features})")
            print(f"  d_model: {self.model.d_model}, n_heads: {self.model.n_heads}")
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"  Total parameters: {total_params:,}")

        # Create windows with stride for training.
        windows = self._create_windows(train_data, stride=self.train_stride)
        if train_segments is not None:
            valid_idx = compute_segment_safe_window_indices(
                train_segments, self.seq_len, self.train_stride, len(windows),
            )
            if self.verbose:
                print(f"  Segment-aware: kept {len(valid_idx)}/{len(windows)} windows "
                      f"({len(valid_idx)/max(len(windows),1):.1%}) — dropped boundary-crossing")
            windows = windows[valid_idx]
        if self.verbose:
            print(f"  Training windows: {len(windows):,} (stride={self.train_stride})")

        # DataLoader.
        dataset = torch.utils.data.TensorDataset(torch.FloatTensor(windows))
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=True,
        )

        # Upstream optimizer/scheduler.
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
        criterion = nn.MSELoss(reduction='mean')

        self.train_loss_history = []
        self.model.train()

        start_time = time.time()

        for epoch in range(self.epochs):
            epoch_start = time.time()
            epoch_losses = []

            # Upstream loss formula: main.py:264 `n = epoch + 1`.
            # Upstream's `epoch` passed to backprop() starts at 0 (fresh model: load_model sets
            # epoch=-1, outer loop `range(epoch+1, ...)` begins at 0). Our 0-indexed Python loop
            # maps 1:1, so we also use `n = epoch + 1` (n ∈ {1, 2, ..., epochs}).
            # → coeff(L1) = 1/n SHRINKS (1.0, 0.5, 0.333, ..., 1/epochs)
            # → coeff(L2) = 1 - 1/n GROWS (0.0, 0.5, 0.667, ..., (epochs-1)/epochs)
            n = epoch + 1

            n_batches = len(dataloader)
            for batch_idx, (batch_x,) in enumerate(dataloader):
                batch_start = time.time()
                batch_x = batch_x.to(self.device)  # (B, seq, feats)

                # Upstream-faithful tensor layout: (seq, batch, feats).
                window = batch_x.permute(1, 0, 2).contiguous()           # (seq, B, feats)
                # `elem` = last timestep, kept as a length-1 sequence.
                elem = window[-1, :, :].unsqueeze(0)                     # (1, B, feats)

                optimizer.zero_grad()

                x1, x2 = self.model(window, elem)                        # (1, B, feats) each

                # Upstream weighting: (1/n) L(x1, elem) + (1 - 1/n) L(x2, elem)
                loss1 = criterion(x1, elem)
                loss2 = criterion(x2, elem)
                loss = (1.0 / n) * loss1 + (1.0 - 1.0 / n) * loss2

                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())
                print(f"[BATCH] model={self.name} epoch={epoch + 1}/{self.epochs} batch={batch_idx + 1}/{n_batches} time={time.time() - batch_start:.3f}s", flush=True)

            # Per-epoch lr scheduler step (upstream).
            scheduler.step()

            avg_loss = float(np.mean(epoch_losses)) if epoch_losses else float('nan')
            self.train_loss_history.append(avg_loss)

            epoch_time = time.time() - epoch_start
            elapsed = time.time() - start_time
            remaining = elapsed / (epoch + 1) * (self.epochs - epoch - 1)

            if self.verbose:
                progress = (epoch + 1) / self.epochs * 100
                print(f"\r  Training: [{epoch + 1}/{self.epochs}] {progress:5.1f}% | "
                      f"Loss: {avg_loss:.6f} | "
                      f"n={n} (coeff L1={1.0/n:.3f}, L2={1.0-1.0/n:.3f}) | "
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
        """Compute anomaly scores (upstream-faithful: Phase 2 only, last-timestep MSE).

        Args:
            test_data: (n_samples, n_features) float32.

        Returns:
            scores: (n_samples,) float32.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        self.model.eval()

        windows = self._create_windows(test_data, stride=1)
        n_windows = len(windows)

        all_scores = []
        n_batches = (n_windows + self.batch_size - 1) // self.batch_size
        start_time = time.time()

        if self.verbose:
            print(f"  Inference: {n_windows:,} windows in {n_batches} batches")

        with torch.no_grad():
            for batch_idx, i in enumerate(range(0, n_windows, self.batch_size)):
                batch_x = torch.FloatTensor(windows[i:i + self.batch_size]).to(self.device)  # (B, seq, F)

                # Upstream-faithful: (seq, B, F) + elem = last timestep.
                window = batch_x.permute(1, 0, 2).contiguous()           # (seq, B, F)
                elem = window[-1, :, :].unsqueeze(0)                     # (1, B, F)

                _, x2 = self.model(window, elem)                         # x2: (1, B, F)

                # Upstream score: MSE on last-timestep only (`(z - elem)**2`), per-feature → mean over features.
                err = ((x2 - elem) ** 2).mean(dim=2)                     # (1, B)
                err = err.squeeze(0)                                      # (B,)

                all_scores.append(err.cpu().numpy())

                if self.verbose and (batch_idx + 1) % max(1, n_batches // 10) == 0:
                    progress = (batch_idx + 1) / n_batches * 100
                    elapsed = time.time() - start_time
                    print(f"\r  Inference: [{batch_idx + 1}/{n_batches}] {progress:5.1f}% | "
                          f"Elapsed: {elapsed:.1f}s", end="")
                    sys.stdout.flush()

        if self.verbose:
            total_time = time.time() - start_time
            print(f"\n  Inference complete: {total_time:.1f}s")

        window_scores = np.concatenate(all_scores) if all_scores else np.zeros(0, dtype=np.float32)

        # Map window scores to point-level — last-position assignment.
        n_samples = len(test_data)
        scores = np.zeros(n_samples, dtype=np.float32)

        for i, score in enumerate(window_scores):
            target_idx = i + self.seq_len - 1
            if target_idx < n_samples:
                scores[target_idx] = score

        # Forward-fill first seq_len-1 timestamps with first window's score (boundary handling).
        if len(window_scores) > 0:
            first_score = float(window_scores[0])
            for i in range(min(self.seq_len - 1, n_samples)):
                scores[i] = first_score

        return scores.astype(np.float32)

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
            "dropout": self.dropout,
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
        self.dropout = config.get("dropout", self.dropout)
        self.n_features = config["n_features"]

        self.model = TranADModel(
            n_features=self.n_features,
            seq_len=self.seq_len,
            d_ff=self.d_ff,
            n_encoder_layers=self.n_encoder_layers,
            n_decoder_layers=self.n_decoder_layers,
            dropout=self.dropout,
        ).to(self.device)
        self.model.load_state_dict(torch.load(save_dir / "model.pt", map_location=self.device))
        self.model.eval()

        return self

    def __repr__(self) -> str:
        return f"TranADBaseline(seq_len={self.seq_len}, d_ff={self.d_ff})"
