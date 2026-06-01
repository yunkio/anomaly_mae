"""
AnomalyTransformer Baseline Wrapper

Minimax training strategy (matches upstream solver.py):
  - Phase 1 (minimize): loss = recon_loss + lambda * series_loss - lambda * prior_loss
    Updates encoder to suppress association discrepancy.
  - Phase 2 (maximize): loss = recon_loss - lambda * series_loss + lambda * prior_loss
    Increases association discrepancy to distinguish anomalies.

Inference score (matches upstream solver.test):
  per-timestep `score = softmax(-AssocDis * k_temperature) * recon_loss`
  max-aggregate across overlapping windows.

Interface: comparison.baseline_common.run_sota_baseline_with_epoch_eval()
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .model import AnomalyTransformer, my_kl_loss


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


class AnomalyTransformerBaseline:
    """AnomalyTransformer (ICLR 2022 Spotlight) wrapper for sliding-window MTS anomaly detection.

    Original code: https://github.com/thuml/Anomaly-Transformer (MIT License).
    Hyperparameters from `MODEL_PRESETS['default']['anomaly_transformer']`.
    """

    def __init__(
        self,
        win_size: int = 100,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 3,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        lr: float = 1e-4,
        batch_size: int = 128,
        epochs: int = 10,
        train_stride: int = 1,
        k_temperature: float = 50.0,
        kl_weight: float = 3.0,
        device: Optional[str] = None,
        verbose: bool = True,
    ):
        self.win_size = win_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_ff = d_ff if d_ff is not None else d_model
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_stride = train_stride
        self.k_temperature = k_temperature
        self.kl_weight = kl_weight
        self.verbose = verbose

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model: Optional[AnomalyTransformer] = None
        self.n_features: Optional[int] = None
        self.train_loss_history: list = []

    @property
    def name(self) -> str:
        return "AnomalyTransformer"

    def _build_model(self) -> AnomalyTransformer:
        return AnomalyTransformer(
            win_size=self.win_size,
            enc_in=self.n_features,
            c_out=self.n_features,
            d_model=self.d_model,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
            activation="gelu",
            output_attention=True,
        ).to(self.device)

    def _normalize_prior(self, prior_layer):
        """Normalize prior so it sums to 1 over last dim (upstream pattern)."""
        return prior_layer / torch.unsqueeze(
            torch.sum(prior_layer, dim=-1), dim=-1
        ).repeat(1, 1, 1, self.win_size)

    def _series_prior_losses(self, series_list, prior_list):
        """Per-batch (series_loss, prior_loss) sums over encoder layers (matches upstream)."""
        series_loss = 0.0
        prior_loss = 0.0
        n_layers = len(prior_list)
        for u in range(n_layers):
            prior_norm = self._normalize_prior(prior_list[u])
            series_loss = series_loss + (
                torch.mean(my_kl_loss(series_list[u], prior_norm.detach()))
                + torch.mean(my_kl_loss(prior_norm.detach(), series_list[u]))
            )
            prior_loss = prior_loss + (
                torch.mean(my_kl_loss(prior_norm, series_list[u].detach()))
                + torch.mean(my_kl_loss(series_list[u].detach(), prior_norm))
            )
        return series_loss / n_layers, prior_loss / n_layers

    def fit(self, train_X: np.ndarray, epoch_callback=None) -> "AnomalyTransformerBaseline":
        self.n_features = train_X.shape[1]
        self.model = self._build_model()

        dataset = _SlidingWindowDataset(train_X, self.win_size, stride=self.train_stride)
        if self.verbose:
            print(f"  Created {len(dataset)} training windows (stride={self.train_stride})")
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        mse = nn.MSELoss()
        k = self.kl_weight

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

                output, series_list, prior_list, _ = self.model(input_x)
                rec_loss = mse(output, input_x)
                series_loss, prior_loss = self._series_prior_losses(series_list, prior_list)

                # Phase 1: minimize series_loss
                loss1 = rec_loss - k * series_loss
                # Phase 2: maximize prior_loss (gradient ascent on prior_loss inside same step)
                loss2 = rec_loss + k * prior_loss

                # Upstream solver.py uses retain_graph=True on first backward; we sum the
                # two losses (sign accounts for min/max direction).
                total_loss = loss1 + loss2
                total_loss.backward()
                optimizer.step()

                epoch_loss_sum += rec_loss.item()
                n_batches += 1

            avg_loss = epoch_loss_sum / max(n_batches, 1)
            self.train_loss_history.append(avg_loss)
            if self.verbose:
                print(f"  Epoch {epoch + 1}: rec_loss = {avg_loss:.6f}")
            if epoch_callback is not None:
                epoch_callback(self, epoch + 1)
                self.model.train()
        return self

    def predict(self, test_X: np.ndarray) -> np.ndarray:
        """1D anomaly score per timestep.

        Per-window: `metric = softmax((-series_loss - prior_loss) * temp, dim=-1) * recon_mse`.
        Aggregate across overlapping windows via max.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        N_test, n_features = test_X.shape
        if N_test < self.win_size:
            raise ValueError(f"Test sequence length {N_test} shorter than win_size {self.win_size}")

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

                batch_windows = np.zeros((actual_bs, self.win_size, n_features), dtype=np.float32)
                for j, w_idx in enumerate(range(batch_start, batch_end)):
                    batch_windows[j] = test_X[w_idx : w_idx + self.win_size]

                input_x = torch.from_numpy(batch_windows).to(self.device)
                output, series_list, prior_list, _ = self.model(input_x)

                # Per-timestep recon MSE: mean over feature dim
                recon_mse = ((output - input_x) ** 2).mean(dim=-1)  # (B, L)

                # Per-timestep AssocDis (upstream solver.test pattern)
                series_loss = None
                prior_loss = None
                for u in range(len(prior_list)):
                    prior_norm = self._normalize_prior(prior_list[u])
                    s_u = my_kl_loss(series_list[u], prior_norm.detach()) * T
                    p_u = my_kl_loss(prior_norm, series_list[u].detach()) * T
                    if u == 0:
                        series_loss = s_u
                        prior_loss = p_u
                    else:
                        series_loss = series_loss + s_u
                        prior_loss = prior_loss + p_u

                metric = torch.softmax((-series_loss - prior_loss), dim=-1)  # (B, L)
                cri = metric * recon_mse  # (B, L)
                window_scores = cri.cpu().numpy()

                for j, w_idx in enumerate(range(batch_start, batch_end)):
                    for pos in range(self.win_size):
                        t = w_idx + pos
                        if t < N_test:
                            point_scores[t] = max(point_scores[t], float(window_scores[j, pos]))

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
            "model_name": "AnomalyTransformer",
            "win_size": self.win_size,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "e_layers": self.e_layers,
            "d_ff": self.d_ff,
            "dropout": self.dropout,
            "n_features": self.n_features,
        }
        with open(save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    def load(self, save_dir: Path) -> "AnomalyTransformerBaseline":
        save_dir = Path(save_dir)
        with open(save_dir / "config.json", "r") as f:
            config = json.load(f)
        for k in ("win_size", "d_model", "n_heads", "e_layers", "d_ff", "dropout", "n_features"):
            setattr(self, k, config[k])
        self.model = self._build_model()
        self.model.load_state_dict(torch.load(save_dir / "model.pt", map_location=self.device))
        self.model.eval()
        return self
