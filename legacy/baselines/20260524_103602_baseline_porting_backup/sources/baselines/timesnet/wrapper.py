"""
TimesNet Baseline Wrapper

Reconstruction-based MTS anomaly detection. Trained with MSE on sliding windows.
Anomaly score per timestep = MSE between input and reconstruction (aggregated
across overlapping windows via max).

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

from .model import TimesNet


class _SlidingWindowDataset(Dataset):
    """Lazy sliding window dataset (matches anomaly_transformer pattern)."""

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


class TimesNetBaseline:
    """TimesNet (ICLR 2023) wrapper for sliding-window MTS anomaly detection.

    Original code: https://github.com/thuml/Time-Series-Library (MIT).
    Hyperparameters from `MODEL_PRESETS['default']['timesnet']`.
    """

    def __init__(
        self,
        win_size: int = 100,
        d_model: int = 64,
        d_ff: int = 64,
        e_layers: int = 3,
        top_k: int = 3,
        num_kernels: int = 6,
        dropout: float = 0.1,
        lr: float = 1e-4,
        batch_size: int = 128,
        epochs: int = 10,
        train_stride: int = 1,
        device: Optional[str] = None,
        verbose: bool = True,
    ):
        self.win_size = win_size
        self.d_model = d_model
        self.d_ff = d_ff
        self.e_layers = e_layers
        self.top_k = top_k
        self.num_kernels = num_kernels
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

        self.model: Optional[TimesNet] = None
        self.n_features: Optional[int] = None
        self.train_loss_history: list = []
        # Paper-faithful normalization: upstream Time-Series-Library SWATSegLoader uses
        # StandardScaler fit on train_data, then transform both train+test. Driver passes
        # raw data (normalize_mode='none' via run_baseline.py override).
        self.scaler: Optional[StandardScaler] = None

    @property
    def name(self) -> str:
        return "TimesNet"

    def _build_model(self) -> TimesNet:
        return TimesNet(
            seq_len=self.win_size,
            c_in=self.n_features,
            d_model=self.d_model,
            d_ff=self.d_ff,
            e_layers=self.e_layers,
            top_k=self.top_k,
            num_kernels=self.num_kernels,
            dropout=self.dropout,
        ).to(self.device)

    def fit(self, train_X: np.ndarray, epoch_callback=None) -> "TimesNetBaseline":
        """Train TimesNet via reconstruction MSE.

        Args:
            train_X: (N_train, n_features) — raw (driver passes normalize_mode='none'
                for self-normalizing SOTA). Wrapper applies StandardScaler to match
                upstream Time-Series-Library data_provider.
            epoch_callback(self, ep+1): optional, invoked after each epoch
        """
        # --- Paper-faithful normalization (upstream data_provider equivalent) ---
        self.scaler = StandardScaler()
        self.scaler.fit(train_X)
        train_X = self.scaler.transform(train_X).astype(np.float32)

        self.n_features = train_X.shape[1]
        self.model = self._build_model()

        dataset = _SlidingWindowDataset(train_X, self.win_size, stride=self.train_stride)
        if self.verbose:
            print(f"  Created {len(dataset)} training windows (stride={self.train_stride})")

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        criterion = nn.MSELoss()
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
                output = self.model(input_x)
                loss = criterion(output, input_x)
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
        """Compute 1D anomaly score per timestep (reconstruction error per timestep, max-aggregated)."""
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
                output = self.model(input_x)
                # Per-timestep squared error, mean over features: [B, T]
                window_scores = ((output - input_x) ** 2).mean(dim=-1).cpu().numpy()

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
            "model_name": "TimesNet",
            "win_size": self.win_size,
            "d_model": self.d_model,
            "d_ff": self.d_ff,
            "e_layers": self.e_layers,
            "top_k": self.top_k,
            "num_kernels": self.num_kernels,
            "n_features": self.n_features,
        }
        with open(save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    def load(self, save_dir: Path) -> "TimesNetBaseline":
        save_dir = Path(save_dir)
        with open(save_dir / "config.json", "r") as f:
            config = json.load(f)
        self.win_size = config["win_size"]
        self.d_model = config["d_model"]
        self.d_ff = config["d_ff"]
        self.e_layers = config["e_layers"]
        self.top_k = config["top_k"]
        self.num_kernels = config["num_kernels"]
        self.n_features = config["n_features"]

        self.model = self._build_model()
        self.model.load_state_dict(torch.load(save_dir / "model.pt", map_location=self.device))
        self.model.eval()
        scaler_path = save_dir / "scaler.pkl"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
        return self

    def get_model_info(self) -> dict:
        return {
            "model": "TimesNet",
            "win_size": self.win_size,
            "d_model": self.d_model,
            "d_ff": self.d_ff,
            "e_layers": self.e_layers,
            "top_k": self.top_k,
            "num_kernels": self.num_kernels,
            "dropout": self.dropout,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "n_features": self.n_features,
        }
