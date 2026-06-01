"""
TimesNet Baseline Wrapper

Reconstruction-based MTS anomaly detection. Trained with MSE on sliding windows.
Anomaly score per timestep = MSE between input and reconstruction (aggregated
across overlapping windows via max).

Training schedule: per-epoch learning rate halving (upstream `lradj='type1'`,
``lr = base_lr * 0.5^(epoch-1)`` where ``epoch`` is 1-indexed). Reproduces
``utils/tools.py:adjust_learning_rate`` of the Time-Series-Library, which the
upstream ``exp/exp_anomaly_detection.py`` calls at the end of every epoch with
``args.lradj`` defaulting to ``type1`` (run.py argparse).

Interface: comparison.baseline_common.run_sota_baseline_with_epoch_eval()
"""

import json
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Subset

from comparison.segment_utils import compute_segment_safe_window_indices
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

    def fit(self, train_X: np.ndarray, epoch_callback=None, train_segments=None) -> "TimesNetBaseline":
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
        if train_segments is not None:
            valid_idx = compute_segment_safe_window_indices(
                train_segments, self.win_size, self.train_stride, len(dataset),
            )
            if self.verbose:
                print(f"  Segment-aware: kept {len(valid_idx)}/{len(dataset)} windows "
                      f"({len(valid_idx)/max(len(dataset),1):.1%}) — dropped boundary-crossing")
            dataset = Subset(dataset, valid_idx.tolist())
        if self.verbose:
            print(f"  Created {len(dataset)} training windows (stride={self.train_stride})")

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        self.train_loss_history = []
        n_batches_total = len(loader)
        for epoch in range(self.epochs):
            epoch_loss_sum = 0.0
            n_batches = 0

            iterator = (
                tqdm(loader, desc=f"Epoch {epoch + 1}/{self.epochs}", leave=False)
                if self.verbose
                else loader
            )
            for batch_idx, batch in enumerate(iterator):
                batch_start = time.time()
                input_x = batch.to(self.device)
                optimizer.zero_grad()
                output = self.model(input_x)
                loss = criterion(output, input_x)
                loss.backward()
                optimizer.step()

                epoch_loss_sum += loss.item()
                n_batches += 1
                print(f"[BATCH] model={self.name} epoch={epoch + 1}/{self.epochs} batch={batch_idx + 1}/{n_batches_total} time={time.time() - batch_start:.3f}s", flush=True)

            avg_loss = epoch_loss_sum / max(n_batches, 1)
            self.train_loss_history.append(avg_loss)

            if self.verbose:
                print(f"  Epoch {epoch + 1}: loss = {avg_loss:.6f}")

            if epoch_callback is not None:
                epoch_callback(self, epoch + 1)
                self.model.train()

            # --- Upstream type1 LR schedule (utils/tools.py:adjust_learning_rate) ---
            # exp/exp_anomaly_detection.py calls
            #   adjust_learning_rate(model_optim, epoch + 1, self.args)
            # at the end of each epoch, where `epoch` is 0-indexed in upstream's
            # for-loop too. With `args.lradj='type1'` (run.py default) and
            # `args.learning_rate=self.lr`, that resolves to
            #   lr_new = self.lr * (0.5 ** ((epoch + 1 - 1) // 1))
            #          = self.lr * (0.5 ** epoch)
            # i.e. after epoch=0 lr stays at base, after epoch=1 lr → base/2, etc.
            next_lr = self.lr * (0.5 ** epoch)
            for pg in optimizer.param_groups:
                pg["lr"] = next_lr
            if self.verbose and epoch > 0:
                print(f"  Adjusted LR to {next_lr:.3e} (type1 schedule)")

        return self

    def predict(self, test_X: np.ndarray) -> np.ndarray:
        """Compute 1D anomaly score per timestep (reconstruction MSE, last-position scoring).

        Unified pipeline pattern (Phase 2.5.1, user direction 2026-05-24): use
        **stride=1 sliding windows + last-position per-window score + Option B
        forward-fill head**, the same canonical pattern as
        `omnianomaly`/`usad`/`tranad`/`gcn_lstm`/`gdn`. This supersedes the
        Phase 2.5 SMD-branch non-overlap simplification, which was a
        dataset-conditional choice unfaithful to TimesNet's intended behavior
        on the majority of datasets (upstream uses `step=1` for
        PSM/MSL/SMAP/SWaT; SMD is the outlier).

        Per-window score formula (faithful to upstream
        `exp/exp_anomaly_detection.py::test()`):
            window_scores[b, t] = mean_features((output[b, t] - input[b, t]) ** 2)
        Then reduce each window to a scalar via the LAST timestep:
            score_w = window_scores[:, -1]  # per-window scalar

        Window-to-timestep mapping (last-position pattern):
            score_w covers test_X timestep t = w + win_size - 1 (last position of
            window w). For w ∈ [0, n_windows), valid coverage is
            t ∈ [win_size - 1, N_test).

        Option B forward-fill head (canonical):
            The first (win_size - 1) timesteps lack a window whose last position
            falls on them; forward-fill them from valid_scores[0] (the first
            valid score). Documented in
            `model_work/timesnet/11_INFERENCE_AGGREGATION_CORRECTION.md`.
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

        # stride=1 sliding (canonical pattern from omnianomaly.predict()).
        test_stride = 1
        n_windows = (N_test - self.win_size) // test_stride + 1  # = N_test - W + 1
        self.model.eval()
        n_batches = (n_windows + self.batch_size - 1) // self.batch_size

        per_window_last: list = []  # each: (actual_bs,) — last-position scalar per window

        with torch.no_grad():
            for batch_idx in range(n_batches):
                batch_start = batch_idx * self.batch_size
                batch_end = min(batch_start + self.batch_size, n_windows)
                actual_bs = batch_end - batch_start

                batch_windows = np.zeros(
                    (actual_bs, self.win_size, n_features), dtype=np.float32
                )
                for j, w_idx in enumerate(range(batch_start, batch_end)):
                    start = w_idx * test_stride
                    batch_windows[j] = test_X[start : start + self.win_size]

                input_x = torch.from_numpy(batch_windows).to(self.device)
                output = self.model(input_x)
                # Per-window per-timestep MSE, mean over features → [B, T]
                window_scores = ((output - input_x) ** 2).mean(dim=-1)
                # Last-position reduction → [B] scalar per window
                last_pos = window_scores[:, -1].cpu().numpy().astype(np.float32)
                per_window_last.append(last_pos)

                if self.verbose and (batch_idx + 1) % max(1, n_batches // 10) == 0:
                    progress = (batch_idx + 1) / n_batches * 100
                    print(f"\r  Inference: [{batch_idx + 1}/{n_batches}] {progress:5.1f}%", end="")

        if self.verbose:
            print()

        # valid_scores: length = n_windows = N_test - W + 1.
        # Aligned to test_X timesteps via: valid_scores[w] → t = w + W - 1.
        valid_scores = np.concatenate(per_window_last, axis=0).astype(np.float32)
        valid_len = valid_scores.shape[0]
        assert valid_len == n_windows, (
            f"aggregation length mismatch: {valid_len} != {n_windows}"
        )

        # Map valid_scores into the output, plus Option B forward-fill head.
        # Head: t ∈ [0, W-1) inherits valid_scores[0] (first valid score).
        # Body: t ∈ [W-1, N_test) gets valid_scores[t - (W-1)] = valid_scores[0..valid_len-1].
        # Total: head_len + valid_len = (W-1) + (N_test - W + 1) = N_test.
        scores = np.empty(N_test, dtype=np.float32)
        head_len = self.win_size - 1
        if head_len > 0:
            scores[:head_len] = valid_scores[0]
        scores[head_len:head_len + valid_len] = valid_scores

        return scores

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
            "lradj": "type1",  # upstream default, halves lr each epoch end
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "n_features": self.n_features,
        }
