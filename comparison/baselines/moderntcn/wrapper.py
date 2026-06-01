"""
ModernTCN Baseline Wrapper

Modern TCN architecture with patch embedding, large-kernel depthwise conv, and dual
ConvFFN mixers (D-mixer + M-mixer). Trained as a reconstruction autoencoder.

Interface: comparison.baseline_common.run_sota_baseline_with_epoch_eval()

Paper-faithful notes (vs upstream `luodhhh/ModernTCN/ModernTCN-detection/`):
- LR schedule: upstream `exp_anomaly_detection.py` creates `OneCycleLR` but NEVER
  calls `scheduler.step()`; the effective schedule comes from
  `utils/tools.py:adjust_learning_rate(... lradj='type1')` per-epoch, i.e.
  ``lr_new = base_lr * 0.5 ** (epoch_idx_1based - 1)``. We mirror this here via
  ``lr_decay_per_epoch=0.5`` applied at the end of each epoch (skipping the very
  first epoch so the first epoch runs at base_lr, matching the upstream formula).
- Score aggregation (Phase 2.5.1 correction, 2026-05-24 — supersedes earlier
  Phase 2.5 non-overlap simplification per user direction): Upstream
  `exp_anomaly_detection.py::test()` (lines 31-40 of the test() body) iterates
  the test loader and concatenates per-window per-position MSE via
  ``np.concatenate(attens_energy, axis=0).reshape(-1)``. The loader's effective
  stride is dataset-conditional in upstream (``data_provider/data_loader.py``):
  **SMD `step=100`** (non-overlap if win=100) vs **PSM/MSL/SMAP/SWaT `step=1`**
  (full overlap). The majority of upstream's reference datasets use
  ``step=1``; SMD is the outlier. User directive (2026-05-24): unify to the
  **stride=1 sliding window + last-position score + Option B forward-fill
  head** pattern (the standard "stride=1 sliding, score = per-window scalar
  mapped to last-timestep" recipe used by omnianomaly / usad / tranad /
  gcn_lstm / gdn). For each stride=1 window we compute the per-window scalar
  ``score_w = mean(MSE(recon[:, -1, :], input[:, -1, :]))`` (i.e. the
  last-timestep position of the per-position MSE map, mean-over-features), and
  assign it to test timestep ``t = w + win_size - 1`` (last position of
  window w). This produces ``valid_scores`` of length
  ``T_test - win_size + 1``. The first ``win_size - 1`` timesteps (which lack
  a fully-contextualized window ending on them) are filled via Option B head
  forward-fill (`scores[:win_size-1] = valid_scores[0]`). No tail padding is
  required because the last valid window already ends at index ``T_test - 1``.
  Architecture and per-position MSE formula are unchanged from upstream.
  Canonical pattern reference: `comparison/baselines/omnianomaly/model.py`
  `predict()`.
- StandardScaler: upstream `data_provider/data_loader.py:SWATSegLoader` fits a
  StandardScaler on the train split and transforms both train and test. We do the
  same (Path B self-normalizing), driver passes raw data
  (`normalize_mode='none'`, see `run_baseline.py:SELF_NORMALIZING_SOTA`).
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
        lr: float = 3e-4,                       # paper SWaT 0.0003
        lr_decay_per_epoch: float = 0.5,        # upstream lradj='type1' = lr * 0.5^(epoch-1)
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
        self.lr_decay_per_epoch = lr_decay_per_epoch
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

    def fit(self, train_X: np.ndarray, epoch_callback=None, train_segments=None) -> "ModernTCNBaseline":
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
        # Upstream data_factory.py anomaly_detection branch uses drop_last=False.
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        # Paper-faithful LR schedule (upstream `exp_anomaly_detection.py` + `utils/tools.py`):
        # upstream constructs `OneCycleLR` but never calls `scheduler.step()`; instead, at the
        # end of each epoch it calls `adjust_learning_rate(optimizer, scheduler, epoch+1, args)`
        # with `args.lradj='type1'` (the argparse default), which sets
        #     lr_new = base_lr * 0.5 ** (epoch_1based - 1)
        # i.e. epoch_1based=1 → lr = base_lr (no change), epoch_1based=2 → lr = base_lr * 0.5,
        # epoch_1based=3 → lr = base_lr * 0.25, etc.
        # We replicate this exactly by applying `lr *= lr_decay_per_epoch` at the end of every
        # epoch starting from the SECOND epoch (skip after the first epoch so the formula matches).
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
                recon = self.model(input_x)
                loss = loss_fn(recon, input_x)
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
            # Apply upstream `adjust_learning_rate(optimizer, scheduler, epoch+1, args)` with
            # `lradj='type1'`: lr_new = base_lr * 0.5 ** ((epoch_1based - 1) // 1).
            # Trajectory: epoch idx 0 → call(1) → lr=base (unchanged); epoch idx 1 → call(2) →
            # lr=base*0.5; epoch idx 2 → call(3) → lr=base*0.25; ...
            # Multiplicatively equivalent: skip after epoch idx 0; halve after epoch idx >= 1.
            # Skip on the very last epoch too (no following epoch will use the new value).
            if epoch >= 1 and epoch < self.epochs - 1:
                for g in optimizer.param_groups:
                    g['lr'] = g['lr'] * self.lr_decay_per_epoch
                if self.verbose:
                    print(f"  Updating learning rate to {optimizer.param_groups[0]['lr']:.3e}")
        return self

    def predict(self, test_X: np.ndarray) -> np.ndarray:
        """Compute per-timestep anomaly score (length=T_test) using stride=1
        sliding-window inference + last-position scalar score + Option B
        forward-fill head (canonical pattern; see
        `comparison/baselines/omnianomaly/model.py::predict()`).

        Reference upstream test loop (per-window-per-position MSE flatten —
        unchanged for the per-window MSE *formula*; we map it to a length-T_test
        score via the standard sliding+last-position recipe instead of the
        dataset-conditional concat.reshape(-1) flatten):

            outputs = self.model(batch_x, None, None, None)
            score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)  # (B, W)

        Reduction-to-scalar-per-window: take the *last position* of the
        per-window MSE map ``score[:, -1]``; this is the per-window scalar
        ``score_w = mean(MSE(recon[:, -1, :], input[:, -1, :]))`` aligned with
        test timestep ``t = w + W - 1``. Stride=1 sliding produces
        ``valid_scores`` of length ``T_test - W + 1``. Option B head
        forward-fill assigns ``valid_scores[0]`` to the first ``W-1`` indices
        (which have no fully-contextualized window ending on them). Tail needs
        no padding: the last valid score is at index ``T_test - 1``.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        if self.scaler is None:
            raise RuntimeError("Scaler not fit. Call fit() first.")

        # Apply paper-faithful StandardScaler.transform on test data (train-only fit).
        test_X = self.scaler.transform(test_X).astype(np.float32)

        N_test, n_features = test_X.shape
        if N_test < self.win_size:
            raise ValueError(f"Test sequence length {N_test} shorter than win_size {self.win_size}")

        W = self.win_size
        # Stride=1 sliding window (canonical pattern: omnianomaly / usad / tranad / gcn_lstm / gdn).
        # n_windows = N_test - W + 1; window w starts at index w, ends at index w + W - 1.
        n_windows = N_test - W + 1
        valid_len = n_windows  # one scalar per window → length T_test - W + 1

        self.model.eval()
        valid_scores = np.empty(valid_len, dtype=np.float32)
        n_batches = (n_windows + self.batch_size - 1) // self.batch_size
        with torch.no_grad():
            for batch_idx in range(n_batches):
                batch_start = batch_idx * self.batch_size
                batch_end = min(batch_start + self.batch_size, n_windows)
                actual_bs = batch_end - batch_start

                batch_windows = np.zeros((actual_bs, W, n_features), dtype=np.float32)
                for j, w_idx in enumerate(range(batch_start, batch_end)):
                    batch_windows[j] = test_X[w_idx : w_idx + W]

                input_x = torch.from_numpy(batch_windows).to(self.device)
                recon = self.model(input_x)
                # Per-timestep, per-feature MSE → mean over features → (B, W).
                err = ((recon - input_x) ** 2).mean(dim=-1)  # (B, W)
                # Last-position scalar per window: err[:, -1] → (B,). Canonical
                # "stride=1 sliding + last-position score" pattern (omnianomaly).
                last_pos = err[:, -1].cpu().numpy().astype(np.float32)  # (B,)
                valid_scores[batch_start:batch_end] = last_pos

                if self.verbose and (batch_idx + 1) % max(1, n_batches // 10) == 0:
                    progress = (batch_idx + 1) / n_batches * 100
                    print(f"\r  Inference: [{batch_idx + 1}/{n_batches}] {progress:5.1f}%", end="")

        if self.verbose:
            print()

        # Map each window's last-position score to its corresponding test timestep:
        # window w (w = 0..n_windows-1) → t = w + W - 1.
        # So valid_scores[w] is the score for timestep (w + W - 1), i.e. valid_scores
        # populates indices [W-1, W, W+1, ..., T_test-1].
        # Option B head forward-fill: indices [0, W-2] copy valid_scores[0].
        scores = np.empty(N_test, dtype=np.float32)
        scores[W - 1 :] = valid_scores  # length N_test - (W-1) == valid_len ✓
        if W > 1:
            scores[: W - 1] = valid_scores[0]
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
