"""
NPSR Baseline Wrapper

Joint training of point-level (M_pt) and sequence-level (M_seq) autoencoders with
random induction masking. Inference uses the NPSR scoring formula combining both
reconstruction errors via the per-batch nominality threshold N(x).

Interface: comparison.baseline_common.run_sota_baseline_with_epoch_eval()
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .model import NPSR


class _SlidingWindowDataset(Dataset):
    def __init__(self, data: np.ndarray, win_size: int, stride: int = 1):
        self.data = data
        self.win_size = win_size
        self.stride = stride
        self.n_windows = (len(data) - win_size) // stride + 1

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        start = idx * self.stride
        return torch.from_numpy(self.data[start : start + self.win_size].copy()).float()


class NPSRBaseline:
    """NPSR (NeurIPS 2023) wrapper for sliding-window MTS anomaly detection.

    Original code: https://github.com/andrewlai61616/NPSR.
    Hyperparameters from `MODEL_PRESETS['default']['npsr']`.
    """

    def __init__(
        self,
        win_size: int = 100,
        d_model: int = 256,
        n_heads: int = 4,
        e_layers: int = 4,
        induction_length: int = 16,
        theta_N: float = 0.985,
        dropout: float = 0.1,
        lr: float = 1e-4,
        batch_size: int = 64,
        epochs: int = 10,
        train_stride: int = 1,
        device: Optional[str] = None,
        verbose: bool = True,
    ):
        self.win_size = win_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.induction_length = induction_length
        self.theta_N = theta_N
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_stride = train_stride
        self.verbose = verbose

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model: Optional[NPSR] = None
        self.n_features: Optional[int] = None
        self.train_loss_history: list = []

    @property
    def name(self) -> str:
        return "NPSR"

    def _build_model(self) -> NPSR:
        return NPSR(
            win_size=self.win_size,
            enc_in=self.n_features,
            d_model=self.d_model,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            induction_length=self.induction_length,
            dropout=self.dropout,
        ).to(self.device)

    def fit(self, train_X: np.ndarray, epoch_callback=None) -> "NPSRBaseline":
        self.n_features = train_X.shape[1]
        self.model = self._build_model()

        dataset = _SlidingWindowDataset(train_X, self.win_size, stride=self.train_stride)
        if self.verbose:
            print(f"  Created {len(dataset)} training windows (stride={self.train_stride})")
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        mse = nn.MSELoss()

        self.model.train()
        self.train_loss_history = []
        for epoch in range(self.epochs):
            epoch_loss_sum = 0.0
            n_batches = 0
            iterator = (
                tqdm(loader, desc=f"Epoch {epoch + 1}/{self.epochs}", leave=False)
                if self.verbose
                else loader
            )
            for batch in iterator:
                input_x = batch.to(self.device)
                optimizer.zero_grad()
                out = self.model(input_x)
                loss_pt = mse(out["pt_recon"], input_x)
                # M_seq loss: only on induction positions
                mask = out["induction_mask"]
                if mask.any():
                    diff = (out["seq_recon"] - input_x) ** 2
                    loss_seq = (diff * mask.unsqueeze(-1)).sum() / (mask.sum() * input_x.size(-1) + 1e-8)
                else:
                    loss_seq = mse(out["seq_recon"], input_x)
                loss = loss_pt + loss_seq

                loss.backward()
                optimizer.step()
                epoch_loss_sum += loss.item()
                n_batches += 1

            avg_loss = epoch_loss_sum / max(n_batches, 1)
            self.train_loss_history.append(avg_loss)
            if self.verbose:
                print(f"  Epoch {epoch + 1}: loss = {avg_loss:.6f}")
            if epoch_callback is not None:
                epoch_callback(self, epoch + 1)
                self.model.train()
        return self

    def predict(self, test_X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        N_test, n_features = test_X.shape
        if N_test < self.win_size:
            raise ValueError(f"Test sequence length {N_test} shorter than win_size {self.win_size}")

        n_windows = N_test - self.win_size + 1
        point_scores = np.zeros(N_test, dtype=np.float32)
        self.model.eval()

        n_batches = (n_windows + self.batch_size - 1) // self.batch_size
        with torch.no_grad():
            for batch_idx in range(n_batches):
                batch_start = batch_idx * self.batch_size
                batch_end = min(batch_start + self.batch_size, n_windows)
                actual_bs = batch_end - batch_start

                batch_windows = np.zeros((actual_bs, self.win_size, n_features), dtype=np.float32)
                for j, w_idx in enumerate(range(batch_start, batch_end)):
                    batch_windows[j] = test_X[w_idx : w_idx + self.win_size]

                input_x = torch.from_numpy(batch_windows).to(self.device)
                scores = self.model.anomaly_score(input_x, theta_N=self.theta_N).cpu().numpy()  # (B, L)

                for j, w_idx in enumerate(range(batch_start, batch_end)):
                    for pos in range(self.win_size):
                        t = w_idx + pos
                        if t < N_test:
                            point_scores[t] = max(point_scores[t], float(scores[j, pos]))

                if self.verbose and (batch_idx + 1) % max(1, n_batches // 10) == 0:
                    progress = (batch_idx + 1) / n_batches * 100
                    print(f"\r  Inference: [{batch_idx + 1}/{n_batches}] {progress:5.1f}%", end="")

        if self.verbose:
            print()
        return point_scores

    def save(self, save_dir: Path) -> None:
        if self.model is None:
            raise RuntimeError("No model to save.")
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), save_dir / "model.pt")
        config = {
            "model_name": "NPSR",
            "win_size": self.win_size,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "e_layers": self.e_layers,
            "induction_length": self.induction_length,
            "theta_N": self.theta_N,
            "n_features": self.n_features,
        }
        with open(save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    def load(self, save_dir: Path) -> "NPSRBaseline":
        save_dir = Path(save_dir)
        with open(save_dir / "config.json", "r") as f:
            config = json.load(f)
        for k in ("win_size", "d_model", "n_heads", "e_layers", "induction_length",
                  "theta_N", "n_features"):
            setattr(self, k, config[k])
        self.model = self._build_model()
        self.model.load_state_dict(torch.load(save_dir / "model.pt", map_location=self.device))
        self.model.eval()
        return self
