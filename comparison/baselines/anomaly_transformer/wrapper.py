"""
AnomalyTransformer Baseline Wrapper

Minimax training strategy (matches upstream solver.py):
  - Phase 1 (minimize): loss = recon_loss + lambda * series_loss - lambda * prior_loss
    Updates encoder to suppress association discrepancy.
  - Phase 2 (maximize): loss = recon_loss - lambda * series_loss + lambda * prior_loss
    Increases association discrepancy to distinguish anomalies.

Inference score (matches upstream solver.test):
  per-timestep `score = softmax(-AssocDis * k_temperature) * recon_loss`
  non-overlap walk (stride=win_size) + flat concat (`reshape(-1)`),
  matching `data_factory/data_loader.py:SegLoader` mode='thre' behavior:
    __len__  = (T - win_size) // win_size + 1
    __getitem__ = test[idx // step * win_size : idx // step * win_size + win_size]
  Tail handling (Option B fallback): the trailing `T_test mod win_size` timesteps
  not covered by the non-overlap walk are filled by repeating the last valid
  per-timestep score. This preserves the `./comparison/` length contract
  (predict() -> (T_test,) float32) while matching reference's effective
  windowing+aggregation exactly.

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
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from .model import AnomalyTransformer, my_kl_loss
from comparison.segment_utils import compute_segment_safe_window_indices
from comparison.baselines._per_file_norm import (
    fit_transform_train_per_file,
    transform_test_per_file,
)
from comparison.baselines._boundary_safe_window import per_entity_concat


def _adjust_learning_rate_type1(optimizer: torch.optim.Optimizer, epoch_one_indexed: int, lr_base: float) -> float:
    """Per-epoch LR halving (matches upstream solver.adjust_learning_rate).

    Formula: lr = lr_base * 0.5 ** (epoch_one_indexed - 1)
    - epoch_one_indexed=1 -> lr_base (no change)
    - epoch_one_indexed=2 -> lr_base / 2
    - epoch_one_indexed=N -> lr_base / 2**(N-1)

    Reference: https://github.com/thuml/Anomaly-Transformer/blob/main/solver.py
    """
    new_lr = lr_base * (0.5 ** (epoch_one_indexed - 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return new_lr


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
        # Path B (SELF_NORMALIZING_SOTA): wrapper applies upstream's StandardScaler
        # internally. Driver passes RAW data when args.model is in SELF_NORMALIZING_SOTA
        # (see comparison/run_baseline.py:~209).
        self.scaler: Optional[StandardScaler] = None
        # Per-source-FILE leak-free norm (multi-file datasets only). One StandardScaler
        # per file (entity), fit on that file's TRAIN slice in fit(), cached here, then
        # applied (.transform) to that file's paired TEST slice in predict(). Single-file
        # => a single whole-array scaler (bit-identical to legacy). Scaler IDENTITY is
        # preserved (StandardScaler), matching upstream data_factory/data_loader.py.
        self._scalers: Optional[list] = None

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

    def fit(self, train_X: np.ndarray, epoch_callback=None, train_segments=None,
            norm_train_segs=None) -> "AnomalyTransformerBaseline":
        self.n_features = train_X.shape[1]
        self.model = self._build_model()

        # Path B: per-source-FILE StandardScaler, fit on each file's TRAIN slice only
        # (matches upstream data_factory/data_loader.py StandardScaler, but per-entity
        # for multi-file datasets — no global blend). `norm_train_segs` is a SEPARATE
        # per-file segment list (NORM only); it does NOT touch `train_segments`, which
        # remains the window-safety segmentation. Single-file / None => one whole-array
        # scaler (bit-identical to legacy global StandardScaler fit-on-train).
        train_X_scaled, self._scalers, _ = fit_transform_train_per_file(
            train_X, norm_train_segs, lambda: StandardScaler()
        )
        train_X_scaled = train_X_scaled.astype(np.float32, copy=False)
        # Preserve back-compat attribute: expose the first (single-file: the only) scaler.
        self.scaler = self._scalers[0] if self._scalers else None

        dataset = _SlidingWindowDataset(train_X_scaled, self.win_size, stride=self.train_stride)
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
        mse = nn.MSELoss()
        k = self.kl_weight

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
                print(f"[BATCH] model={self.name} epoch={epoch + 1}/{self.epochs} batch={batch_idx + 1}/{n_batches_total} time={time.time() - batch_start:.3f}s", flush=True)

            avg_loss = epoch_loss_sum / max(n_batches, 1)
            self.train_loss_history.append(avg_loss)
            if self.verbose:
                print(f"  Epoch {epoch + 1}: rec_loss = {avg_loss:.6f}")
            # Upstream per-epoch LR halving (solver.adjust_learning_rate called with
            # epoch+1 at end of each epoch, using base lr `self.lr`).
            # FIX 2026-05-25: was `(epoch + 1) + 1` (off-by-one — started halving 1
            # epoch early, ~50% smaller cumulative LR). Now matches upstream
            # `solver.py:205 adjust_learning_rate(opt, epoch + 1, lr_)` exactly.
            next_epoch_one_indexed = epoch + 1  # upstream's epoch_arg = 0-indexed epoch + 1
            new_lr = _adjust_learning_rate_type1(optimizer, next_epoch_one_indexed, self.lr)
            if self.verbose:
                print(f"  -> next-epoch lr = {new_lr:.6g}")
            if epoch_callback is not None:
                epoch_callback(self, epoch + 1)
                self.model.train()
        return self

    def predict(self, test_X: np.ndarray, test_segments=None) -> np.ndarray:
        """1D anomaly score per timestep — reference-faithful inference.

        Per-window: `metric = softmax((-series_loss - prior_loss) * temp, dim=-1) * recon_mse`.

        Windowing + aggregation (matches upstream `solver.py:Solver.test()` exactly):
          * Non-overlap walk: stride = win_size (upstream `mode='thre'`
            `SegLoader` uses `index // step * win_size` indexing).
          * `__len__` = (T - win_size) // win_size + 1 windows.
          * Aggregation: `np.concatenate(per_window_scores, axis=0).reshape(-1)`
            i.e. flat concat of per-timestep scores within each non-overlap window.

        Length contract (Option B fallback for tail):
          * Upstream effective output length = `((T - W) // W + 1) * W` ≤ T_test.
          * To satisfy `./comparison/` `(T_test,) float32` length contract, the
            trailing `T_test mod W` timesteps are filled by repeating the LAST
            per-timestep valid score (no head padding needed since non-overlap
            walk starts at index 0).

        BOUNDARY-SAFE TEST WINDOWING (multi-entity datasets — SMD machines / SMAP-MSL
        channels / Exathlon apps): the RAW score producer (non-overlap windowing + model
        inference + per-window->per-timestep concat + tail fill) is run INDEPENDENTLY on
        each entity's own (already-normalized) test slice via
        ``per_entity_concat(test_X, test_segments, raw_fn)``, so no non-overlap window
        ever spans an entity boundary (which would mix two entities and corrupt the
        boundary-region scores). Single-entity / None / non-tiling test_segments ->
        ONE call over the whole array == bit-identical legacy behaviour (the helper
        guarantees the no-op). AnomalyTransformer has NO whole-test score post-processing
        (the per-window softmax is local to each window; the percentile threshold +
        anomaly-state propagation in upstream solver.test live DOWNSTREAM of our
        ``(T_test,)`` continuous-score harness contract), so there is nothing to apply
        after concatenation — the concatenated per-entity raw scores ARE the final scores.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        # Path B: per-source-FILE leak-free test transform — the i-th test file is
        # transformed by the i-th cached TRAIN StandardScaler (.transform only, NEVER
        # fit-on-test), matching upstream data_factory/data_loader.py (scaler fit on
        # train, .transform applied to test). `test_segments` is the per-file TEST-LOCAL
        # segment list paired 1:1 with the train files. Single-file / None / mismatch =>
        # the single train scaler over the whole test array (bit-identical to legacy).
        # Normalization is done HERE, before windowing — unchanged by the boundary-safe
        # refactor (the per_entity_concat below re-windows the ALREADY-normalized array;
        # it does NOT re-normalize).
        if self._scalers:
            test_X = transform_test_per_file(
                test_X, test_segments, self._scalers
            ).astype(np.float32, copy=False)
        test_X = np.asarray(test_X, dtype=np.float32)

        _, n_features = test_X.shape
        T = self.k_temperature
        self.model.eval()

        def _raw_score_slice(sub_X: np.ndarray) -> np.ndarray:
            """RAW per-timestep scores for a SINGLE entity's test slice (no cross-entity
            dependence, no whole-test post-proc). Reproduces the exact legacy windowing +
            inference + per-window->per-timestep concat + tail fill on ``sub_X`` alone.

            Edge-safe: if this entity slice is shorter than ``win_size`` no non-overlap
            window can be formed; we edge-pad the slice up to ``win_size`` (repeat the last
            row), score that single window, and return only the first ``len(sub_X)``
            per-timestep scores. This keeps the score finite and length-correct without
            crashing (the legacy whole-array path raised ValueError; per-entity slicing
            can now legitimately yield a short slice, e.g. a tiny SMD test entity).
            """
            sub_X = np.asarray(sub_X, dtype=np.float32)
            L = sub_X.shape[0]

            if L < self.win_size:
                # Edge-pad up to one full window by repeating the last timestep.
                pad = self.win_size - L
                last_row = sub_X[-1:].repeat(pad, axis=0) if L > 0 else \
                    np.zeros((pad, n_features), dtype=np.float32)
                padded = np.concatenate([sub_X, last_row], axis=0)  # (win_size, F)
                win_X = padded[None, :, :]  # (1, win_size, F)
                input_x = torch.from_numpy(win_X).to(self.device)
                with torch.no_grad():
                    output, series_list, prior_list, _ = self.model(input_x)
                    recon_mse = ((output - input_x) ** 2).mean(dim=-1)  # (1, win_size)
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
                    metric = torch.softmax((-series_loss - prior_loss), dim=-1)
                    cri = (metric * recon_mse).cpu().numpy().astype(np.float32, copy=False)
                return cri.reshape(-1)[:L].copy()

            # Upstream `SegLoader(mode='thre')` window count + non-overlap walk:
            #   n_windows = (L - W) // W + 1 ; window i covers sub_X[i*W : i*W + W]
            n_windows = (L - self.win_size) // self.win_size + 1
            n_batches = (n_windows + self.batch_size - 1) // self.batch_size
            per_window_scores: list = []  # each entry: (B, win_size) np.float32

            with torch.no_grad():
                for batch_idx in range(n_batches):
                    batch_start = batch_idx * self.batch_size
                    batch_end = min(batch_start + self.batch_size, n_windows)
                    actual_bs = batch_end - batch_start

                    batch_windows = np.zeros((actual_bs, self.win_size, n_features), dtype=np.float32)
                    for j, w_idx in enumerate(range(batch_start, batch_end)):
                        start = w_idx * self.win_size  # non-overlap stride = win_size
                        batch_windows[j] = sub_X[start : start + self.win_size]

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
                    per_window_scores.append(cri.cpu().numpy().astype(np.float32, copy=False))

                    if self.verbose and (batch_idx + 1) % max(1, n_batches // 10) == 0:
                        progress = (batch_idx + 1) / n_batches * 100
                        print(f"\r  Inference: [{batch_idx + 1}/{n_batches}] {progress:5.1f}%", end="")

            # Upstream aggregation: `np.concatenate(attens_energy, axis=0).reshape(-1)`
            valid_scores = np.concatenate(per_window_scores, axis=0).reshape(-1)
            valid_len = valid_scores.shape[0]  # == n_windows * win_size

            # Option B fallback: pad trailing `L mod win_size` timesteps by
            # repeating the last valid per-timestep score (no head pad — non-overlap
            # walk starts at index 0).
            point_scores = np.empty(L, dtype=np.float32)
            point_scores[:valid_len] = valid_scores
            if valid_len < L:
                point_scores[valid_len:] = valid_scores[-1]
            return point_scores

        # Boundary-safe TEST windowing: run the RAW producer per-entity (no window spans
        # an entity boundary), concat to (n_test,). No whole-test post-proc to apply.
        point_scores = per_entity_concat(test_X, test_segments, _raw_score_slice)

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
        # Persist Path B scaler so predict() after load() retains train-fit stats.
        if self.scaler is not None:
            joblib.dump(self.scaler, save_dir / "scaler.pkl")
        # Persist the per-source-FILE train scaler LIST so a loaded model's predict()
        # can do leak-free per-file .transform (multi-file). Single-file => 1-elem list.
        if self._scalers:
            joblib.dump(self._scalers, save_dir / "scalers.pkl")

    def load(self, save_dir: Path) -> "AnomalyTransformerBaseline":
        save_dir = Path(save_dir)
        with open(save_dir / "config.json", "r") as f:
            config = json.load(f)
        for k in ("win_size", "d_model", "n_heads", "e_layers", "d_ff", "dropout", "n_features"):
            setattr(self, k, config[k])
        self.model = self._build_model()
        self.model.load_state_dict(torch.load(save_dir / "model.pt", map_location=self.device))
        self.model.eval()
        scaler_path = save_dir / "scaler.pkl"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
        # Restore the per-source-FILE train scaler list for leak-free per-file predict().
        scalers_path = save_dir / "scalers.pkl"
        if scalers_path.exists():
            self._scalers = joblib.load(scalers_path)
        elif self.scaler is not None:
            # Back-compat: older checkpoints only saved the single scaler.
            self._scalers = [self.scaler]
        return self
