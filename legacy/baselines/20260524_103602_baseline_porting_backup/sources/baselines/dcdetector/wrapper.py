"""
DCdetector Baseline Wrapper

Dual-attention contrastive representation learning for MTS anomaly detection.
No reconstruction loss — uses KL discrepancy between patch-wise and in-patch
attention representations.

Interface: comparison.baseline_common.run_sota_baseline_with_epoch_eval()
"""

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .model import DCdetector, my_kl_loss


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


class DCdetectorBaseline:
    """DCdetector (KDD 2023) wrapper for sliding-window MTS anomaly detection.

    Original code: https://github.com/DAMO-DI-ML/KDD2023-DCdetector (no LICENSE — attribution only).
    Hyperparameters from `MODEL_PRESETS['default']['dcdetector']`.
    """

    def __init__(
        self,
        win_size: int = 105,
        patch_size=(3, 5, 7),
        d_model: int = 256,
        n_heads: int = 1,
        e_layers: int = 3,
        dropout: float = 0.0,
        lr: float = 1e-4,
        batch_size: int = 128,
        epochs: int = 10,
        train_stride: int = 1,
        k_temperature: float = 50.0,
        device: Optional[str] = None,
        verbose: bool = True,
    ):
        self.win_size = win_size
        self.patch_size = list(patch_size)
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_stride = train_stride
        self.k_temperature = k_temperature
        self.verbose = verbose

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model: Optional[DCdetector] = None
        self.n_features: Optional[int] = None
        self.train_loss_history: list = []
        # Paper-faithful normalization: upstream DCdetector data_factory/data_loader uses
        # StandardScaler fit on train_data, then transform both train+test. Driver passes
        # raw data (normalize_mode='none' via run_baseline.py override).
        self.scaler: Optional[StandardScaler] = None

    @property
    def name(self) -> str:
        return "DCdetector"

    def _build_model(self) -> DCdetector:
        return DCdetector(
            win_size=self.win_size,
            enc_in=self.n_features,
            c_out=self.n_features,
            n_heads=self.n_heads,
            d_model=self.d_model,
            e_layers=self.e_layers,
            patch_size=self.patch_size,
            channel=self.n_features,
            dropout=self.dropout,
            output_attention=True,
        ).to(self.device)

    def _series_prior_losses(self, series, prior, *, scale=1.0):
        """Compute symmetric KL between series and normalized prior, per upstream solver.

        Returns (series_loss_sum, prior_loss_sum) — sums over encoder layers / multi-scale patches.
        """
        series_loss = 0.0
        prior_loss = 0.0
        for u in range(len(prior)):
            prior_norm = prior[u] / torch.unsqueeze(
                torch.sum(prior[u], dim=-1), dim=-1
            ).repeat(1, 1, 1, self.win_size)
            series_loss = series_loss + scale * (
                torch.mean(my_kl_loss(series[u], prior_norm.detach()))
                + torch.mean(my_kl_loss(prior_norm.detach(), series[u]))
            )
            prior_loss = prior_loss + scale * (
                torch.mean(my_kl_loss(prior_norm, series[u].detach()))
                + torch.mean(my_kl_loss(series[u].detach(), prior_norm))
            )
        return series_loss / max(len(prior), 1), prior_loss / max(len(prior), 1)

    def fit(self, train_X: np.ndarray, epoch_callback=None) -> "DCdetectorBaseline":
        """Train DCdetector via dual-attention contrastive KL.

        train_X is raw (driver passes normalize_mode='none' for self-normalizing SOTA).
        Wrapper applies StandardScaler — matches upstream DCdetector data_factory.

        Objective (matches upstream solver):
            loss = prior_loss - series_loss   (min)
        """
        # --- Paper-faithful normalization (upstream data_factory equivalent) ---
        self.scaler = StandardScaler()
        self.scaler.fit(train_X)
        train_X = self.scaler.transform(train_X).astype(np.float32)

        self.n_features = train_X.shape[1]
        self.model = self._build_model()

        dataset = _SlidingWindowDataset(train_X, self.win_size, stride=self.train_stride)
        if self.verbose:
            print(f"  Created {len(dataset)} training windows (stride={self.train_stride})")
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

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
                series, prior = self.model(input_x)
                series_loss, prior_loss = self._series_prior_losses(series, prior)
                loss = prior_loss - series_loss
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
        """1D anomaly score per timestep.

        Score per window position = softmax(-series_loss - prior_loss, dim=-1) × temperature.
        Aggregate across overlapping windows via max.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        if self.scaler is None:
            raise RuntimeError("Scaler not fit. Call fit() first.")

        # Apply paper-faithful StandardScaler.transform on test data (train-only fit).
        test_X = self.scaler.transform(test_X).astype(np.float32)

        N_test, n_features = test_X.shape
        if N_test < self.win_size:
            raise ValueError(
                f"Test sequence length {N_test} shorter than win_size {self.win_size}"
            )

        n_windows = N_test - self.win_size + 1
        T = self.k_temperature
        self.model.eval()
        point_scores = np.zeros(N_test, dtype=np.float32)
        n_batches = (n_windows + self.batch_size - 1) // self.batch_size

        with torch.no_grad():
            for batch_idx in range(n_batches):
                batch_start = batch_idx * self.batch_size
                batch_end = min(batch_start + self.batch_size, n_windows)
                actual_bs = batch_end - batch_start

                batch_windows = np.zeros(
                    (actual_bs, self.win_size, n_features), dtype=np.float32
                )
                for j, w_idx in enumerate(range(batch_start, batch_end)):
                    batch_windows[j] = test_X[w_idx : w_idx + self.win_size]

                input_x = torch.from_numpy(batch_windows).to(self.device)
                series, prior = self.model(input_x)

                # Test scoring (per upstream solver.test): per-batch scalar via my_kl_loss
                # series/prior elements are 4D [B, H, W, W]; my_kl_loss reduces last 2 dims to [B, W]
                series_loss = None
                prior_loss = None
                for u in range(len(prior)):
                    prior_norm = prior[u] / torch.unsqueeze(
                        torch.sum(prior[u], dim=-1), dim=-1
                    ).repeat(1, 1, 1, self.win_size)
                    s_u = my_kl_loss(series[u], prior_norm.detach()) * T
                    p_u = my_kl_loss(prior_norm, series[u].detach()) * T
                    if u == 0:
                        series_loss, prior_loss = s_u, p_u
                    else:
                        series_loss = series_loss + s_u
                        prior_loss = prior_loss + p_u

                # metric: [B, win_size] → softmax across W
                metric = torch.softmax((-series_loss - prior_loss), dim=-1)
                window_scores = metric.cpu().numpy()  # [B, W]

                for j, w_idx in enumerate(range(batch_start, batch_end)):
                    for pos in range(self.win_size):
                        t = w_idx + pos
                        if t < N_test:
                            point_scores[t] = max(point_scores[t], window_scores[j, pos])

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
        if self.scaler is not None:
            joblib.dump(self.scaler, save_dir / "scaler.pkl")
        config = {
            "model_name": "DCdetector",
            "win_size": self.win_size,
            "patch_size": list(self.patch_size),
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "e_layers": self.e_layers,
            "n_features": self.n_features,
        }
        with open(save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    def load(self, save_dir: Path) -> "DCdetectorBaseline":
        save_dir = Path(save_dir)
        with open(save_dir / "config.json", "r") as f:
            config = json.load(f)
        self.win_size = config["win_size"]
        self.patch_size = config["patch_size"]
        self.d_model = config["d_model"]
        self.n_heads = config["n_heads"]
        self.e_layers = config["e_layers"]
        self.n_features = config["n_features"]
        self.model = self._build_model()
        self.model.load_state_dict(torch.load(save_dir / "model.pt", map_location=self.device))
        self.model.eval()
        scaler_path = save_dir / "scaler.pkl"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
        return self
