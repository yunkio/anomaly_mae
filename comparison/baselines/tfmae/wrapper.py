"""
TFMAE Baseline Wrapper

Adversarial contrastive training of dual Temporal/Frequency Masked Autoencoders.
Sliding-window inference with KL-discrepancy score per timestep.

Interface: comparison.baseline_common.run_sota_baseline_with_epoch_eval()
  - fit(train_X, epoch_callback=None)
  - predict(test_X) -> 1D anomaly score per timestep
"""

import json
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Subset

from comparison.segment_utils import compute_segment_safe_window_indices
from tqdm import tqdm

from .model import MTFA, my_kl_loss


class _SlidingWindowDataset(Dataset):
    """Lazy sliding window dataset matching anomaly_transformer pattern."""

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


class TFMAEBaseline:
    """TFMAE (ICDE 2024) wrapper for sliding-window MTS anomaly detection.

    Hyperparameters from `MODEL_PRESETS['default']['tfmae']` in baseline_common.py.
    """

    def __init__(
        self,
        win_size: int = 100,
        seq_size: int = 5,
        d_model: int = 128,
        e_layers: int = 3,
        n_heads: int = 8,  # unused (single-head attention); kept for preset compatibility
        temporal_mask_ratio: float = 0.5,  # tr
        freq_mask_ratio: float = 0.4,  # fr (lowest fraction of magnitudes to mask)
        dropout: float = 0.05,
        lr: float = 1e-4,
        batch_size: int = 64,
        epochs: int = 10,
        train_stride: int = 1,
        k_temperature: float = 50.0,  # temperature for inference score
        device: Optional[str] = None,
        verbose: bool = True,
    ):
        self.win_size = win_size
        self.seq_size = seq_size
        self.d_model = d_model
        self.e_layers = e_layers
        self.tr = temporal_mask_ratio
        self.fr = freq_mask_ratio
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

        self.model: Optional[MTFA] = None
        self.n_features: Optional[int] = None
        self.train_loss_history: list = []
        # Paper-faithful normalization: upstream TFMAE data_provider uses StandardScaler
        # fit on train_data, then transform both train and test. Driver passes raw data
        # (normalize_mode='none' via run_baseline.py override).
        self.scaler: Optional[StandardScaler] = None

    @property
    def name(self) -> str:
        return "TFMAE"

    def _build_model(self) -> MTFA:
        return MTFA(
            win_size=self.win_size,
            seq_size=self.seq_size,
            c_in=self.n_features,
            c_out=self.n_features,
            d_model=self.d_model,
            e_layers=self.e_layers,
            fr=self.fr,
            tr=self.tr,
        ).to(self.device)

    # ----------------------------------------------------------------------
    # Loss helpers (per-batch). Operates on `attlist`-style tensors [B, T, T].
    # ----------------------------------------------------------------------

    @staticmethod
    def _per_layer_kl(tematt_u, freatt_u, *, stopgrad_target: str):
        """Compute symmetric KL between attention maps of one encoder layer.

        Returns per-row KL summed over both directions; reduction over batch+T done by caller.
        """
        freatt_norm = freatt_u / torch.unsqueeze(torch.sum(freatt_u, dim=-1), dim=-1)
        if stopgrad_target == "fre":
            target = freatt_norm.detach()
            return my_kl_loss(tematt_u, target) + my_kl_loss(target, tematt_u)
        elif stopgrad_target == "tem":
            target = tematt_u.detach()
            return my_kl_loss(freatt_norm, target) + my_kl_loss(target, freatt_norm)
        raise ValueError(f"Unknown stopgrad_target: {stopgrad_target}")

    def fit(self, train_X: np.ndarray, epoch_callback=None, train_segments=None) -> "TFMAEBaseline":
        """Train TFMAE on `train_X` (N_train, n_features).

        train_X is raw (driver passes normalize_mode='none' for self-normalizing SOTA).
        Wrapper applies StandardScaler.fit(train_X).transform(...) — matches upstream
        TFMAE data_provider line-by-line (StandardScaler train-only fit).

        Adversarial contrastive objective per TFMAE paper Eq.15:
          loss = con_loss - adv_loss
            adv_loss = sum_u KL(tem_u, sg(fre_u)) + KL(sg(fre_u), tem_u)
            con_loss = sum_u KL(fre_u, sg(tem_u)) + KL(sg(tem_u), fre_u)
        Calls `epoch_callback(self, ep + 1)` after each epoch when provided.
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
                tematt, freatt = self.model(input_x)

                # Drop the final reconstruction tensor (`.pro(dx)`) appended by encoders;
                # only attention maps (last == reconstruction) are used here. The original
                # TFMAE solver iterates over freatt[u] using the same length as tematt[u].
                # Iterate over all layer attentions (last item is reconstruction projection
                # but original code also includes it as part of `attlist`).
                adv_loss = 0.0
                con_loss = 0.0
                for u in range(len(freatt)):
                    adv_loss = adv_loss + torch.mean(
                        self._per_layer_kl(tematt[u], freatt[u], stopgrad_target="fre")
                    )
                    con_loss = con_loss + torch.mean(
                        self._per_layer_kl(tematt[u], freatt[u], stopgrad_target="tem")
                    )

                adv_loss = adv_loss / len(freatt)
                con_loss = con_loss / len(freatt)

                # Combined objective: minimize (con - adv)
                loss = con_loss - adv_loss
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

        return self

    def predict(self, test_X: np.ndarray) -> np.ndarray:
        """Compute 1D anomaly score per timestep — paper-faithful inference.

        Reference: `LMissher/TFMAE @ main`
          - `data_factory/data_loader.py::SMDSegLoader` (mode='test'):
              non-overlap walk with stride = win_size
              (`__len__ = (T - W) // W + 1`,
               `__getitem__` returns `self.test[index*W : index*W + W]`
               since the `step` passed to `__init__` is hardcoded to 1).
          - `solver.py::Solver.test()`:
              per-window per-position score = `softmax(temperature * (adv + con), dim=-1)`,
              aggregated via `np.concatenate(attens_energy, axis=0).reshape(-1)` (flatten),
              i.e., one score per timestep within each non-overlap window, concatenated.
          - Tail handling in reference: IMPLICIT_NONE — the trailing `T mod W` timesteps
            are silently dropped because labels come from the same loader.

        Pipeline contract requires output length == T_test exactly. Per user-decided
        fallback (Option B), pad the tail by repeating the last valid score, and pad
        the head (if any) by forward-fill from the first valid score.
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

        W = self.win_size
        T = self.k_temperature

        # Reference-faithful non-overlap walk: stride = win_size.
        # n_windows matches reference `__len__ = (T - W) // W + 1`.
        n_windows = (N_test - W) // W + 1
        valid_len = n_windows * W  # = reference's effective output length

        self.model.eval()
        per_window_scores: list = []  # each entry: np.ndarray of shape (B, W)
        n_batches = (n_windows + self.batch_size - 1) // self.batch_size

        with torch.no_grad():
            for batch_idx in range(n_batches):
                batch_start = batch_idx * self.batch_size
                batch_end = min(batch_start + self.batch_size, n_windows)
                actual_bs = batch_end - batch_start

                # Build batch of non-overlapping windows (start = w_idx * W).
                batch_windows = np.zeros(
                    (actual_bs, W, n_features), dtype=np.float32
                )
                for j, w_idx in enumerate(range(batch_start, batch_end)):
                    s = w_idx * W
                    batch_windows[j] = test_X[s : s + W]

                input_x = torch.from_numpy(batch_windows).to(self.device)
                tematt, freatt = self.model(input_x)

                # Per-batch attention-based discrepancy (matches solver.test pattern,
                # unidirectional KL summed across layers).
                adv = None
                con = None
                for u in range(len(freatt)):
                    freatt_norm = freatt[u] / torch.unsqueeze(
                        torch.sum(freatt[u], dim=-1), dim=-1
                    )
                    adv_u = my_kl_loss(tematt[u], freatt_norm.detach()) * T
                    con_u = my_kl_loss(freatt_norm, tematt[u].detach()) * T
                    if u == 0:
                        adv, con = adv_u, con_u
                    else:
                        adv = adv + adv_u
                        con = con + con_u

                # metric: [B, W] — softmax across the window's time dimension.
                metric = torch.softmax((adv + con), dim=-1)
                per_window_scores.append(metric.cpu().numpy())  # [B, W]

                if self.verbose and (batch_idx + 1) % max(1, n_batches // 10) == 0:
                    progress = (batch_idx + 1) / n_batches * 100
                    print(f"\r  Inference: [{batch_idx + 1}/{n_batches}] {progress:5.1f}%", end="")

        if self.verbose:
            print()

        # Reference-faithful aggregation: np.concatenate(..., axis=0).reshape(-1)
        # → flat (n_windows * W,) sequence aligned to test_X[0 : valid_len].
        valid_scores = (
            np.concatenate(per_window_scores, axis=0).reshape(-1).astype(np.float32)
        )
        assert valid_scores.shape == (valid_len,), (
            f"aggregation length mismatch: got {valid_scores.shape}, expected {(valid_len,)}"
        )

        # ---- Option B fallback (user-decided, IMPLICIT_NONE policy) ----
        # Reference's output naturally aligns to test_X[0:valid_len] (head_len = 0,
        # tail_len = N_test - valid_len). Repeat-last for tail; forward-fill for head
        # (no-op when head_len = 0, which is the typical case).
        scores = np.empty(N_test, dtype=np.float32)
        head_len = 0  # reference starts windows at index 0
        tail_len = N_test - valid_len - head_len

        if head_len > 0:
            # Forward-fill head from first valid score
            scores[:head_len] = valid_scores[0]
        scores[head_len : head_len + valid_len] = valid_scores
        if tail_len > 0:
            # Repeat-last per-timestep score
            scores[head_len + valid_len:] = valid_scores[-1]

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
            "model_name": "TFMAE",
            "win_size": self.win_size,
            "seq_size": self.seq_size,
            "d_model": self.d_model,
            "e_layers": self.e_layers,
            "tr": self.tr,
            "fr": self.fr,
            "n_features": self.n_features,
        }
        with open(save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    def load(self, save_dir: Path) -> "TFMAEBaseline":
        save_dir = Path(save_dir)
        with open(save_dir / "config.json", "r") as f:
            config = json.load(f)
        self.win_size = config["win_size"]
        self.seq_size = config["seq_size"]
        self.d_model = config["d_model"]
        self.e_layers = config["e_layers"]
        self.tr = config["tr"]
        self.fr = config["fr"]
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
            "model": "TFMAE",
            "win_size": self.win_size,
            "seq_size": self.seq_size,
            "d_model": self.d_model,
            "e_layers": self.e_layers,
            "tr": self.tr,
            "fr": self.fr,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "k_temperature": self.k_temperature,
            "n_features": self.n_features,
        }
