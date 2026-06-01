"""
Neural Baseline Base Class for Time Series Anomaly Detection

Based on QuoVadisTAD (arXiv:2405.02678):
- Shared training and inference logic for neural baselines
- Sliding window approach for time series
- Reconstruction-based anomaly scoring

Models using this base class:
- 1-Layer MLP
- Single Block MLPMixer
- Single Transformer Block
- 1-Layer GCN-LSTM
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import iqr
from typing import Optional, Tuple
from abc import ABC, abstractmethod
from pathlib import Path
import json
import time
import sys


class NeuralBaselineBase(ABC):
    """
    Abstract base class for neural network baselines.

    All neural baselines follow the same pattern:
    1. Create sliding windows from data
    2. Train model to predict/reconstruct
    3. Use reconstruction error as anomaly score
    """

    def __init__(
        self,
        seq_len: int = 5,
        embedding_dim: int = 128,
        dropout: float = 0.1,
        lr: float = 0.001,
        weight_decay: float = 0.0001,
        batch_size: int = 512,
        epochs: int = 10,
        train_stride: int = 1,
        device: Optional[str] = None,
        verbose: bool = True,
        **extra_kwargs,
    ):
        """
        Args:
            seq_len: Input sequence length (window size)
            embedding_dim: Hidden dimension for embeddings
            dropout: Dropout rate
            lr: Learning rate
            weight_decay: Adam weight_decay (paper default 0.0001)
            batch_size: Training batch size
            epochs: Number of training epochs
            train_stride: Stride for training windows (default 1 = paper)
            device: Device ('cuda' or 'cpu'). Auto-detect if None
            verbose: Print training progress
            **extra_kwargs: Swallow extra model-specific kwargs (e.g. num_heads, num_blocks).
        """
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_stride = train_stride
        self.verbose = verbose
        self._extra = extra_kwargs

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model: Optional[nn.Module] = None
        self.n_features: int = 0
        self.train_loss_history = []

    @abstractmethod
    def _build_model(self) -> nn.Module:
        """Build and return the neural network model."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Model name."""
        pass

    def _create_windows(self, data: np.ndarray, stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding windows for training/inference.

        Args:
            data: Input data (n_samples, n_features)
            stride: Stride between windows (default 1 for test, use train_stride for train)

        Returns:
            Tuple of (windows, targets)
            - windows: (n_windows, seq_len, n_features)
            - targets: (n_windows, n_features) - next timestamp after each window
        """
        n_samples = len(data)
        # Calculate number of windows with stride
        n_windows = (n_samples - self.seq_len - 1) // stride + 1

        windows = np.zeros((n_windows, self.seq_len, data.shape[1]), dtype=np.float32)
        targets = np.zeros((n_windows, data.shape[1]), dtype=np.float32)

        for idx, i in enumerate(range(0, n_samples - self.seq_len, stride)):
            if idx >= n_windows:
                break
            windows[idx] = data[i:i + self.seq_len]
            targets[idx] = data[i + self.seq_len]

        return windows, targets

    def fit(self, train_data: np.ndarray) -> 'NeuralBaselineBase':
        """
        Train the model on training data.

        Args:
            train_data: Training data (n_samples, n_features)

        Returns:
            self
        """
        self.n_features = train_data.shape[1]

        # Build model
        self.model = self._build_model().to(self.device)

        if self.verbose:
            print(f"{self.name} training:")
            print(f"  Device: {self.device}")
            print(f"  Input shape: ({self.seq_len}, {self.n_features})")
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"  Total parameters: {total_params:,}")

        # Create windows with stride for training
        windows, targets = self._create_windows(train_data, stride=self.train_stride)
        if self.verbose:
            print(f"  Training windows: {len(windows):,} (stride={self.train_stride})")

        # Create DataLoader
        dataset = TensorDataset(
            torch.FloatTensor(windows),
            torch.FloatTensor(targets)
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True
        )

        # Training — paper-faithful Adam(lr, weight_decay)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=getattr(self, 'weight_decay', 0.0),
        )
        criterion = nn.MSELoss()

        self.train_loss_history = []
        self.model.train()

        start_time = time.time()
        n_batches = len(dataloader)

        for epoch in range(self.epochs):
            epoch_start = time.time()
            epoch_losses = []

            for batch_idx, (batch_windows, batch_targets) in enumerate(dataloader):
                batch_windows = batch_windows.to(self.device)
                batch_targets = batch_targets.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_windows)
                loss = criterion(outputs, batch_targets)
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

        if self.verbose:
            total_time = time.time() - start_time
            print(f"\n  Training complete: {total_time:.1f}s total, Final loss: {self.train_loss_history[-1]:.6f}")

        return self

    @staticmethod
    def _paper_faithful_scoring(
        delta: np.ndarray,
        smooth_window: int = 5,
        epsilon: float = 1e-2,
    ) -> np.ndarray:
        """Port of QuoVadisTAD `quovadis_tad/dataset_utils/data_utils.py::normalise_scores`
        followed by `.max(axis=1)` (notebook `nn_baselines_models_train_test.ipynb` standard call).

        Mirrors `gcn_lstm.model.GCNLSTMBaseline._normalise_scores` for consistency.

        Args:
            delta: (n_windows, n_sensors) per-window per-sensor absolute residual
                   `|pred - target|`.
            smooth_window: box-smooth half-window (reference default = 5).
            epsilon: IQR floor (reference default = 1e-2).

        Returns:
            per_window_score: (n_windows,) float32 — median-IQR normalized,
                              5-box smoothed, sensor-MAX aggregated. First
                              `smooth_window` rows are 0 (matches reference).
        """
        delta = np.asarray(delta, dtype=np.float32)
        # median-IQR normalize per sensor (axis=0 over windows)
        n_err_mid = np.median(delta, axis=0)
        n_err_iqr = iqr(delta, axis=0)
        err_scores = (delta - n_err_mid) / (np.abs(n_err_iqr) + epsilon)

        # box smooth over the window axis (reference normalise_scores `smooth=True`)
        if len(err_scores) > smooth_window:
            smoothed = np.zeros_like(err_scores)
            for i in range(smooth_window, len(err_scores)):
                lo = i - smooth_window
                hi = i + smooth_window - 1  # exclusive; window length = 2*sw - 1
                hi = min(hi, len(err_scores))
                if hi > lo:
                    smoothed[i] = np.mean(err_scores[lo:hi], axis=0)
            err_scores = smoothed

        # MAX over sensors (notebook: `normalise_scores(score).max(1)`)
        return err_scores.max(axis=-1).astype(np.float32)

    def predict(self, test_data: np.ndarray) -> np.ndarray:
        """Compute paper-faithful anomaly scores for test data.

        Reference inference behavior (QuoVadisTAD `quovadis_tad/model_utils/model_def.py::test_embedder`
        line 483 + notebook `nn_baselines_models_train_test.ipynb` standard call):
            score = np.abs(predictions - orig_target)           # (n_windows, n_features)
            normalise_scores(score).max(1)                       # median-IQR + smooth(5) + MAX

        Reference EXPLICITLY TRUNCATES labels: `gt_labels = labels[input_sequence_length:]`,
        producing a score sequence of length `T_test - seq_len`. The `./comparison/` pipeline
        cannot truncate labels, so we apply Option B fallback — forward-fill the first
        `seq_len` head timesteps from the first valid per-timestep score (matches the
        GCN-LSTM port in `gcn_lstm/model.py`).

        Args:
            test_data: Test data (n_samples, n_features).

        Returns:
            (T_test,) float32 — higher = more anomalous.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        self.model.eval()

        # Create windows
        windows, targets = self._create_windows(test_data)
        n_samples = len(test_data)

        if len(windows) == 0:
            return np.zeros(n_samples, dtype=np.float32)

        # Predict in batches — accumulate per-feature absolute residuals
        all_residuals = []
        n_batches = (len(windows) + self.batch_size - 1) // self.batch_size
        start_time = time.time()

        if self.verbose:
            print(f"  Inference: {len(windows):,} windows in {n_batches} batches")

        with torch.no_grad():
            for batch_idx, i in enumerate(range(0, len(windows), self.batch_size)):
                batch_windows = torch.FloatTensor(windows[i:i + self.batch_size]).to(self.device)
                batch_targets = targets[i:i + self.batch_size]

                outputs = self.model(batch_windows).cpu().numpy()

                # per-sensor absolute residual (reference: np.abs(predictions - orig_target))
                residuals = np.abs(outputs - batch_targets)
                all_residuals.append(residuals)

                if self.verbose and (batch_idx + 1) % max(1, n_batches // 10) == 0:
                    progress = (batch_idx + 1) / n_batches * 100
                    elapsed = time.time() - start_time
                    print(f"\r  Inference: [{batch_idx + 1}/{n_batches}] {progress:5.1f}% | "
                          f"Elapsed: {elapsed:.1f}s", end="")
                    sys.stdout.flush()

        if self.verbose:
            total_time = time.time() - start_time
            print(f"\n  Inference complete: {total_time:.1f}s")

        # (n_windows, n_features) absolute residual
        delta = np.concatenate(all_residuals, axis=0).astype(np.float32)

        # median-IQR + 5-box smooth + sensor-MAX (paper-faithful)
        per_window_score = self._paper_faithful_scoring(delta)  # (n_windows,)

        # Map window scores to point-level scores.
        # Each window at position i predicts timestamp i + seq_len.
        scores = np.zeros(n_samples, dtype=np.float32)
        n_windows = len(per_window_score)
        valid_start = self.seq_len
        valid_end = min(valid_start + n_windows, n_samples)

        # Place valid scores
        if valid_end > valid_start:
            scores[valid_start:valid_end] = per_window_score[:valid_end - valid_start]

        # Option B head fallback: forward-fill first `seq_len` from first valid score.
        if valid_start > 0 and n_windows > 0:
            scores[:valid_start] = per_window_score[0]

        # Defensive tail fallback (nominal stride=1 leaves no gap)
        if valid_end < n_samples and n_windows > 0:
            scores[valid_end:] = per_window_score[-1]

        # Final NaN/inf guard
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return scores

    def save(self, save_dir: Path) -> None:
        """
        Save model weights and config to directory.

        Args:
            save_dir: Directory to save model
        """
        if self.model is None:
            raise RuntimeError("No model to save. Call fit() first.")

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save model weights
        model_path = save_dir / "model.pt"
        torch.save(self.model.state_dict(), model_path)

        # Save config
        config = {
            "seq_len": self.seq_len,
            "embedding_dim": self.embedding_dim,
            "dropout": self.dropout,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "n_features": self.n_features,
            "model_name": self.name,
            "train_loss_history": self.train_loss_history,
        }
        config_path = save_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        if self.verbose:
            print(f"  Saved model to {save_dir}")

    def load(self, save_dir: Path) -> 'NeuralBaselineBase':
        """
        Load model weights and config from directory.

        Args:
            save_dir: Directory containing saved model

        Returns:
            self
        """
        save_dir = Path(save_dir)

        # Load config
        config_path = save_dir / "config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)

        self.seq_len = config["seq_len"]
        self.embedding_dim = config["embedding_dim"]
        self.dropout = config["dropout"]
        self.n_features = config["n_features"]
        self.train_loss_history = config.get("train_loss_history", [])

        # Build and load model
        self.model = self._build_model().to(self.device)
        model_path = save_dir / "model.pt"
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        if self.verbose:
            print(f"  Loaded model from {save_dir}")

        return self

    def __repr__(self) -> str:
        return f"{self.name}(seq_len={self.seq_len}, embedding_dim={self.embedding_dim})"


class MLPBlock(nn.Module):
    """Simple MLP block with GELU activation."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
