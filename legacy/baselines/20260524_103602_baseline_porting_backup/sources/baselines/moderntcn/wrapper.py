"""
ModernTCN Baseline Wrapper

Modern TCN architecture with patch embedding, large-kernel depthwise conv, and dual
ConvFFN mixers (D-mixer + M-mixer). Trained as a reconstruction autoencoder.

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

from .model import ModernTCN


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


class ModernTCNBaseline:
    """ModernTCN (ICLR 2024 Spotlight) wrapper for sliding-window MTS anomaly detection.

    Original code: https://github.com/luodhhh/ModernTCN (MIT License).
    Hyperparameters from `MODEL_PRESETS['default']['moderntcn']`.
    """

    def __init__(
        self,
        # Paper SWaT.sh defaults (luodhhh/ModernTCN/ModernTCN-detection/scripts/SWaT.sh)
        win_size: int = 100,
        patch_size: int = 8,
        patch_stride: int = 4,
        dims=(128,),
        num_blocks=(3,),
        large_size=(51,),
        small_size=(5,),
        ffn_ratio: int = 1,
        dropout: float = 0.1,
        head_dropout: float = 0.0,
        use_revin: bool = True,
        affine: bool = False,
        subtract_last: bool = False,
        use_multi_scale: bool = False,    # paper SWaT
        small_kernel_merged: bool = False,
        stem_ratio: int = 6,              # upstream argparse default
        downsample_ratio: int = 2,        # upstream argparse default
        # Training
        lr: float = 3e-4,                 # paper SWaT 0.0003
        pct_start: float = 0.3,           # OneCycleLR pct_start
        batch_size: int = 128,
        epochs: int = 10,
        train_stride: int = 1,
        device: Optional[str] = None,
        verbose: bool = True,
    ):
        self.win_size = win_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.dims = list(dims)
        self.num_blocks = list(num_blocks)
        self.large_size = list(large_size)
        self.small_size = list(small_size)
        self.ffn_ratio = ffn_ratio
        self.dropout = dropout
        self.head_dropout = head_dropout
        self.use_revin = use_revin
        self.affine = affine
        self.subtract_last = subtract_last
        self.use_multi_scale = use_multi_scale
        self.small_kernel_merged = small_kernel_merged
        self.stem_ratio = stem_ratio
        self.downsample_ratio = downsample_ratio
        self.lr = lr
        self.pct_start = pct_start
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_stride = train_stride
        self.verbose = verbose

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model: Optional[ModernTCN] = None
        self.n_features: Optional[int] = None
        self.train_loss_history: list = []
        # Paper-faithful normalization: upstream ModernTCN-detection data_provider uses
        # StandardScaler fit on train_data, then transform both train+test. Driver passes
        # raw data (normalize_mode='none' via run_baseline.py override).
        self.scaler: Optional[StandardScaler] = None

    @property
    def name(self) -> str:
        return "ModernTCN"

    def _build_model(self) -> ModernTCN:
        return ModernTCN(
            seq_len=self.win_size,
            enc_in=self.n_features,
            c_out=self.n_features,
            patch_size=self.patch_size,
            patch_stride=self.patch_stride,
            dims=self.dims,
            num_blocks=self.num_blocks,
            large_size=self.large_size,
            small_size=self.small_size,
            ffn_ratio=self.ffn_ratio,
            dropout=self.dropout,
            head_dropout=self.head_dropout,
            use_revin=self.use_revin,
            affine=self.affine,
            subtract_last=self.subtract_last,
            use_multi_scale=self.use_multi_scale,
            small_kernel_merged=self.small_kernel_merged,
            stem_ratio=self.stem_ratio,
            downsample_ratio=self.downsample_ratio,
            task_name='anomaly_detection',
        ).to(self.device)

    def fit(self, train_X: np.ndarray, epoch_callback=None) -> "ModernTCNBaseline":
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

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        # OneCycleLR scheduler (upstream ModernTCN-detection/exp/exp_anomaly_detection.py):
        #   scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=train_steps,
        #                                       pct_start=pct_start, epochs=train_epochs)
        train_steps = len(loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=self.lr,
            steps_per_epoch=train_steps,
            epochs=self.epochs,
            pct_start=self.pct_start,
        )

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
                recon = self.model(input_x)
                loss = loss_fn(recon, input_x)
                loss.backward()
                optimizer.step()
                scheduler.step()  # OneCycleLR steps per batch
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
        if self.scaler is None:
            raise RuntimeError("Scaler not fit. Call fit() first.")

        # Apply paper-faithful StandardScaler.transform on test data (train-only fit).
        test_X = self.scaler.transform(test_X).astype(np.float32)

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
                recon = self.model(input_x)
                err = ((recon - input_x) ** 2).mean(dim=-1).cpu().numpy()  # (B, L)

                for j, w_idx in enumerate(range(batch_start, batch_end)):
                    for pos in range(self.win_size):
                        t = w_idx + pos
                        if t < N_test:
                            point_scores[t] = max(point_scores[t], float(err[j, pos]))

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
            "model_name": "ModernTCN",
            "win_size": self.win_size,
            "patch_size": self.patch_size,
            "patch_stride": self.patch_stride,
            "dims": self.dims,
            "num_blocks": self.num_blocks,
            "large_size": self.large_size,
            "small_size": self.small_size,
            "ffn_ratio": self.ffn_ratio,
            "affine": self.affine,
            "subtract_last": self.subtract_last,
            "n_features": self.n_features,
        }
        with open(save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    def load(self, save_dir: Path) -> "ModernTCNBaseline":
        save_dir = Path(save_dir)
        with open(save_dir / "config.json", "r") as f:
            config = json.load(f)
        for k in ("win_size", "patch_size", "patch_stride", "dims", "num_blocks",
                  "large_size", "small_size", "ffn_ratio", "n_features"):
            setattr(self, k, config[k])
        for k in ("affine", "subtract_last"):
            if k in config: setattr(self, k, config[k])
        self.model = self._build_model()
        self.model.load_state_dict(torch.load(save_dir / "model.pt", map_location=self.device))
        self.model.eval()
        scaler_path = save_dir / "scaler.pkl"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
        return self
