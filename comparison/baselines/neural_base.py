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

# Boundary-safe TEST windowing (shared single source of truth). `entity_test_slices`
# returns per-entity (lo, hi) TEST-LOCAL half-open slices from the SAME source as
# per-entity normalization (loader.get_file_norm_segments() test side); None / single
# entity -> [(0, n_test)] (exact legacy whole-array behaviour). See module docstring.
#
# NOTE on `per_entity_concat` vs `entity_test_slices`: the shared `per_entity_concat`
# helper carries a 1D per-timestep `(Li,)` raw score per entity and tiles [0, n_test).
# QuoVadis neural scoring cannot use it directly because the boundary-sensitive raw
# quantity here is the per-window, PER-SENSOR forecast residual `(n_windows_i, F)` whose
# row count (Le - seq_len) does NOT equal the entity length Le, and whose whole-test
# post-processing (median-IQR per sensor over the full window axis) needs the 2D per-sensor
# delta to survive concatenation. We therefore use the SAME module's `entity_test_slices`
# (identical no-op guarantee + source of truth) to window each entity in isolation, concat
# the per-window per-sensor deltas, and run the whole-test post-proc ONCE — keeping
# post-proc granularity bit-identical to the legacy single-entity path.
from comparison.baselines._boundary_safe_window import entity_test_slices


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

    def _window_inference_delta(self, sub_X: np.ndarray) -> np.ndarray:
        """RAW per-window per-sensor forecast residual for ONE entity slice.

        Boundary-safe by construction: windows are built ONLY from `sub_X` (a single
        entity's test slice), so no model-input window can span an entity boundary.
        This is exactly the legacy windowing + batched inference + `|pred - target|`,
        scoped to `sub_X`.

        Args:
            sub_X: (Li, F) one entity's test slice (already normalized upstream).

        Returns:
            delta: (n_windows_i, F) float32 — per-window per-sensor absolute residual,
            where ``n_windows_i = max(Li - seq_len, 0)``. A slice shorter than the window
            (``Li <= seq_len``) yields ``(0, F)`` (no crash; caller head-fills it).
        """
        n_feat = sub_X.shape[1] if sub_X.ndim == 2 else self.n_features
        # Edge-safe: a slice no longer than the window yields zero next-step windows.
        # `_create_windows` computes n_windows = (Li - seq_len - 1)//stride + 1, which is
        # NEGATIVE for Li <= seq_len and would crash `np.zeros((n_windows, ...))`. Guard here.
        if len(sub_X) <= self.seq_len:
            return np.zeros((0, n_feat), dtype=np.float32)
        windows, targets = self._create_windows(sub_X)
        if len(windows) == 0:
            # Entity slice shorter than the window → zero windows. Return empty (0, F);
            # the per-timestep scatter below leaves this entity's points to head-fill.
            return np.zeros((0, n_feat), dtype=np.float32)

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

        return np.concatenate(all_residuals, axis=0).astype(np.float32)

    def predict(self, test_data: np.ndarray, test_segments=None) -> np.ndarray:
        """Compute paper-faithful anomaly scores for test data (boundary-safe windowing).

        Reference inference behavior (QuoVadisTAD `quovadis_tad/model_utils/model_def.py::test_embedder`
        line 483 + notebook `nn_baselines_models_train_test.ipynb` standard call):
            score = np.abs(predictions - orig_target)           # (n_windows, n_features)
            normalise_scores(score).max(1)                       # median-IQR + smooth(5) + MAX

        Reference EXPLICITLY TRUNCATES labels: `gt_labels = labels[input_sequence_length:]`,
        producing a score sequence of length `T_test - seq_len`. The `./comparison/` pipeline
        cannot truncate labels, so we apply Option B fallback — forward-fill the first
        `seq_len` head timesteps from the first valid per-timestep score (matches the
        GCN-LSTM port in `gcn_lstm/model.py`).

        Boundary-safe TEST windowing (multi-file datasets: SMD machines / SMAP-MSL channels /
        Exathlon apps): on multi-entity test arrays, each entity's test slice is windowed +
        inferred INDEPENDENTLY (via `entity_test_slices`), so no model-input window spans an
        entity boundary. The per-window per-sensor residuals are concatenated and the WHOLE-TEST
        post-processing (median-IQR per sensor over the full window axis + 5-box smooth +
        sensor-MAX, in `_paper_faithful_scoring`) is applied ONCE on the concatenation — exactly
        as before. Single-entity / None / non-tiling `test_segments` -> ONE windowing call over
        the whole array == bit-identical legacy behaviour.

        Args:
            test_data: Test data (n_samples, n_features).
            test_segments: per-entity (lo, hi) TEST-LOCAL half-open slices (from
                `loader.get_file_norm_segments()` test side). None / single entity ->
                whole-array (legacy no-op).

        Returns:
            (T_test,) float32 — higher = more anomalous.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        self.model.eval()

        n_samples = len(test_data)
        slices = entity_test_slices(n_samples, test_segments)

        # --- RAW producer (boundary-sensitive): per-entity windowing + inference + the
        #     per-window -> per-timestep target mapping. Concatenated per-window per-sensor
        #     residuals + their GLOBAL target timesteps survive to the whole-test post-proc. ---
        delta_parts = []           # list of (m_e, F) per-entity per-window residuals
        target_index_parts = []    # list of (m_e,) global target timestep for each window row

        for (lo, hi) in slices:
            sub_X = test_data[lo:hi]
            delta_e = self._window_inference_delta(sub_X)   # (m_e, F), m_e = max(hi-lo-seq_len, 0)
            m_e = delta_e.shape[0]
            if m_e > 0:
                # Window i (entity-local) predicts local timestep i + seq_len -> global lo+i+seq_len.
                local_targets = np.arange(m_e, dtype=np.int64) + self.seq_len
                target_index_parts.append(local_targets + lo)
                delta_parts.append(delta_e)
            # else: entity slice <= seq_len -> no valid windows. Its whole [lo, hi) span has no
            #       own score and is left at 0 (the head-fill loop below skips it). Unscorable
            #       region == legacy all-zero behaviour; never crashes.

        if not delta_parts:
            # No entity produced any window (every slice <= seq_len) -> all-zero, finite.
            return np.zeros(n_samples, dtype=np.float32)

        # (Σ m_e, F) per-window per-sensor residual, in entity order.
        delta = np.concatenate(delta_parts, axis=0).astype(np.float32)
        target_index = np.concatenate(target_index_parts, axis=0)  # (Σ m_e,) global timesteps

        # --- WHOLE-TEST post-processing (UNCHANGED granularity): median-IQR per sensor over
        #     the full concatenated window axis + 5-box smooth + sensor-MAX. Applied ONCE. For
        #     single-entity this `delta` IS the legacy whole-array per-window delta -> identical. ---
        per_window_score = self._paper_faithful_scoring(delta)  # (Σ m_e,)

        # --- Map per-window scores to per-timestep via recorded GLOBAL target indices. ---
        scores = np.zeros(n_samples, dtype=np.float32)
        scores[target_index] = per_window_score

        # --- Option B head fallback (per entity): forward-fill each entity's first `seq_len`
        #     head timesteps from that entity's first valid score. The per-window scores are in
        #     entity order, so the first row of each entity block is its first valid score. ---
        cursor = 0
        for (lo, hi) in slices:
            m_e = max((hi - lo) - self.seq_len, 0)
            if m_e > 0:
                first_valid = per_window_score[cursor]
                scores[lo:lo + self.seq_len] = first_valid
                cursor += m_e
            else:
                # Unscorable entity (<= seq_len): leave its span at 0 (already initialized).
                pass

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
