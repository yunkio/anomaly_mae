"""
DCdetector Baseline Wrapper

Dual-attention contrastive representation learning for MTS anomaly detection.
No reconstruction loss — uses KL discrepancy between patch-wise and in-patch
attention representations.

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

    def fit(self, train_X: np.ndarray, epoch_callback=None, train_segments=None) -> "DCdetectorBaseline":
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
        # Capture initial LR for per-epoch halving (matches upstream solver.adjust_learning_rate).
        # Upstream formula: lr_t = lr_0 * 0.5^((t-1)//1) where t is the 1-indexed just-finished epoch.
        # i.e., during epoch k (1-indexed): lr = lr_0 * 0.5^(k-1). Applied AFTER each epoch.
        self._initial_lr = self.lr

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
                series, prior = self.model(input_x)
                series_loss, prior_loss = self._series_prior_losses(series, prior)
                loss = prior_loss - series_loss
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

            # --- Per-epoch LR halving (paper-faithful, upstream solver.adjust_learning_rate) ---
            # After the (epoch+1)-th epoch (1-indexed), set lr for the NEXT epoch to lr_0 * 0.5^epoch.
            # Mirrors `adjust_learning_rate(optimizer, epoch+1, self.lr)` at end of upstream train loop.
            # Applied AFTER epoch_callback so the callback observes the lr that was actually used
            # during the just-finished epoch.
            new_lr = self._initial_lr * (0.5 ** epoch)
            for pg in optimizer.param_groups:
                pg["lr"] = new_lr
        return self

    def _raw_scores(self, sub_X: np.ndarray) -> np.ndarray:
        """RAW per-timestep score producer for ONE entity slice (no cross-entity dependence).

        Runs the upstream-faithful `mode='thre'` test path on `sub_X` ALONE:
          - Non-overlap windowing (stride = win_size; matches upstream
            `data_loader.py` thre `__getitem__`: `index*win_size : index*win_size+win_size`,
            `__len__ = (L - win_size)//win_size + 1`).
          - Per-window metric `softmax(-series_loss - prior_loss, dim=-1)` (temperature T),
            series/prior losses SUMMED over the `len(prior)` attention maps (upstream
            `solver.py::test`). This softmax is taken over the WITHIN-WINDOW axis, so it is
            intrinsically per-window — there is NO cross-window / whole-test post-processing.
          - Flatten (`np.concatenate(...).reshape(-1)`), then Option-B repeat-last tail pad to
            `len(sub_X)` (no head pad; upstream's first valid score maps to t=0).

        `sub_X` is ALREADY scaler-transformed (caller transforms the whole test once, before
        slicing; the train-fit StandardScaler is a fixed transform and is slice-invariant, and
        the model's internal RevIN is per-window — so per-entity slicing cannot change either
        normalization). This method does exactly what the legacy whole-array predict did, just
        on one entity's slice → no window can cross an entity boundary.

        Edge case (entity slice shorter than win_size): cannot form a single non-overlap
        window. Fall back to a uniform `1/L` constant score over the slice. This is the natural
        degenerate of the per-window softmax (every window's `win_size` scores sum to 1, so a
        length-`L` window would assign `1/L` each); it keeps the output finite, the correct
        length, and on the same scale as real windows. Documented, never crashes.
        """
        L, n_features = sub_X.shape
        if L < self.win_size:
            # Boundary-safe fallback for a short entity (no full window available).
            return np.full(L, 1.0 / max(L, 1), dtype=np.float32)

        T = self.k_temperature
        # Non-overlap windowing: matches upstream `mode='thre'` effective stride = win_size.
        n_windows = (L - self.win_size) // self.win_size + 1
        n_batches = (n_windows + self.batch_size - 1) // self.batch_size

        attens_energy = []  # list of [B, win_size] arrays, concat+flatten per upstream

        with torch.no_grad():
            for batch_idx in range(n_batches):
                batch_start = batch_idx * self.batch_size
                batch_end = min(batch_start + self.batch_size, n_windows)
                actual_bs = batch_end - batch_start

                batch_windows = np.zeros(
                    (actual_bs, self.win_size, n_features), dtype=np.float32
                )
                # Non-overlap walk: window j covers sub_X[j*win_size : (j+1)*win_size]
                for j, w_idx in enumerate(range(batch_start, batch_end)):
                    start = w_idx * self.win_size
                    batch_windows[j] = sub_X[start : start + self.win_size]

                input_x = torch.from_numpy(batch_windows).to(self.device)
                series, prior = self.model(input_x)

                # Test scoring (per upstream solver.test): per-window [B, win_size] metric
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

                # metric: [B, win_size] → softmax across W (matches upstream)
                metric = torch.softmax((-series_loss - prior_loss), dim=-1)
                cri = metric.detach().cpu().numpy()  # [B, win_size]
                attens_energy.append(cri)

        # Upstream-faithful aggregation: np.concatenate(..., axis=0).reshape(-1) flatten.
        valid_scores = np.concatenate(attens_energy, axis=0).reshape(-1).astype(np.float32)
        valid_len = valid_scores.shape[0]  # = n_windows * win_size = L - (L % win_size)

        # Option B fallback (user-directed 2026-05-24 Phase 2.5): repeat-last tail to satisfy
        # the per-entity length contract. No head padding (upstream starts at t=0).
        point_scores = np.empty(L, dtype=np.float32)
        point_scores[:valid_len] = valid_scores
        if valid_len < L:
            # Repeat last per-timestep valid score for tail [valid_len, L)
            point_scores[valid_len:] = valid_scores[-1]

        return point_scores

    def predict(self, test_X: np.ndarray, test_segments=None) -> np.ndarray:
        """1D anomaly score per timestep — paper-faithful upstream `mode='thre'` aggregation.

        Boundary-safe TEST windowing: on multi-entity datasets (SMD machines / SMAP-MSL
        channels / Exathlon apps) NO non-overlap window may span an entity boundary. The
        windowing + inference + per-window→per-timestep mapping is routed PER ENTITY through
        ``per_entity_concat`` (see ``_boundary_safe_window.py``). ``test_segments`` is the
        per-entity (lo,hi) TEST-LOCAL list from ``loader.get_file_norm_segments()`` (same
        source as normalization). ``None`` / single-entity / non-tiling segments → ONE call
        over the whole array == bit-identical legacy behaviour (helper-guaranteed NO-OP).

        Upstream (`solver.py::Solver.test()` + `data_factory/data_loader.py`, verified @main):
          - Test windowing uses `mode='thre'` loader → non-overlap stride = `win_size`
            (effective stride; `self.step` is overridden by `index // step * win_size`
            arithmetic in __getitem__).
          - `__len__` (thre) = `(L - win_size) // win_size + 1`
          - Aggregation: per-window `metric = softmax(-series_loss - prior_loss, dim=-1)` →
            `attens_energy.append(cri)` → `np.concatenate(attens_energy, axis=0).reshape(-1)`
            flatten. This per-window softmax is the ENTIRE score — there is NO whole-test score
            post-processing (no global re-normalization, no smoothing), so per-entity slicing of
            the windowing leaves the score formula and its granularity UNCHANGED.
          - Tail handling: IMPLICIT_NONE upstream — silently drops last `L mod win_size`
            timesteps. Our pipeline's `(T_test,)` contract forbids dropping; Option-B repeat-last
            tail pad fills it (per entity).

        Normalization (untouchable-norm class): WHOLE-ARRAY train-fit StandardScaler.transform
        is applied ONCE here, BEFORE per-entity slicing. The scaler is a fixed train-derived
        transform (slice-invariant), and the model's internal RevIN is per-window
        (slice-invariant) — so boundary-safe windowing does NOT alter normalization.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        if self.scaler is None:
            raise RuntimeError("Scaler not fit. Call fit() first.")

        from comparison.baselines._boundary_safe_window import per_entity_concat

        # Apply paper-faithful StandardScaler.transform on test data (train-only fit), ONCE
        # over the WHOLE test (whole-array norm preserved; slice-invariant fixed transform).
        test_X = self.scaler.transform(test_X).astype(np.float32)

        N_test, n_features = test_X.shape
        # Legacy guard preserved for the single-entity / None path: the WHOLE test shorter than
        # win_size is still an error (matches pre-edit behaviour). Multi-entity short slices are
        # handled gracefully inside _raw_scores (per-entity fallback) and never reach this.
        from comparison.baselines._boundary_safe_window import is_multi_entity
        if N_test < self.win_size and not is_multi_entity(N_test, test_segments):
            raise ValueError(
                f"Test sequence length {N_test} shorter than win_size {self.win_size}"
            )

        self.model.eval()

        # Route windowing + inference + per-window→per-timestep mapping (incl. tail pad) through
        # the shared per-entity helper. Each entity is windowed in isolation → no cross-boundary
        # window. Single-entity / None → exactly one call on the whole array (NO-OP).
        point_scores = per_entity_concat(test_X, test_segments, self._raw_scores)

        if self.verbose:
            print()

        return point_scores.astype(np.float32)

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
