"""
TFMAE Baseline Wrapper

Adversarial contrastive training of dual Temporal/Frequency Masked Autoencoders.
Sliding-window inference with KL-discrepancy score per timestep.

Interface: comparison.baseline_common.run_sota_baseline_with_epoch_eval()
  - fit(train_X, epoch_callback=None)
  - predict(test_X) -> 1D anomaly score per timestep
"""

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
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

    def fit(self, train_X: np.ndarray, epoch_callback=None) -> "TFMAEBaseline":
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

            avg_loss = epoch_loss_sum / max(n_batches, 1)
            self.train_loss_history.append(avg_loss)

            if self.verbose:
                print(f"  Epoch {epoch + 1}: loss = {avg_loss:.6f}")

            if epoch_callback is not None:
                epoch_callback(self, epoch + 1)
                self.model.train()

        return self

    def predict(self, test_X: np.ndarray) -> np.ndarray:
        """Compute 1D anomaly score per timestep.

        Score per window position = softmax(adv_loss + con_loss) * temperature
        Aggregate to point level via max across overlapping windows.
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

                # Build batch of windows
                batch_windows = np.zeros(
                    (actual_bs, self.win_size, n_features), dtype=np.float32
                )
                for j, w_idx in enumerate(range(batch_start, batch_end)):
                    batch_windows[j] = test_X[w_idx : w_idx + self.win_size]

                input_x = torch.from_numpy(batch_windows).to(self.device)
                tematt, freatt = self.model(input_x)

                # Per-batch attention-based discrepancy (matches solver.test pattern)
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

                # metric: [B, T] → softmax across T, normalize each window
                metric = torch.softmax((adv + con), dim=-1)
                window_scores = metric.cpu().numpy()  # [B, T]

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
