"""
NPSR Baseline Wrapper — paper-faithful reimplementation.

Implements the full NPSR procedure (Lai et al., NeurIPS 2023):

1. **Normalization (Path B)**: ``MinMaxScaler(feature_range=(-1, 1))`` fit on
   train, transform both train/test, then clamp test only to ``[clamp_min,
   clamp_max]`` (default ±4). The wrapper is therefore registered in
   ``run_baseline.py``'s ``SELF_NORMALIZING_SOTA``.
2. **Two sub-models, two optimizers**: M_pt (NPSRPointAE) and M_seq
   (NPSRSeqReducer) are stepped *alternately within each batch* using two
   independent Adam optimizers — matches reference ``train.py`` behaviour.
3. **M_seq training data**: left+right halves concatenated into a
   ``pred_dl``-length input; the central ``delta``-length window is the target.
4. **Nominality score**: ``N(t) = mean((Δ_x0 − Δ_xp)²) / mean(Δ_x0²)``.
5. **θ_N**: ``np.sort(trn_Nt)[int(N * theta_N_ratio)]`` (training-set quantile,
   default ``0.9985``).
6. **Induced anomaly score**: gated cumulative-product sum with default
   ``d=16, gate='soft'`` (paper Section 3.4 best-performing default).
7. **Test windowing**: non-overlap stride ``= win_size`` (reference's
   ``tst_stride='no_rep'``). M_pt uses wrap-around padding so the
   non-overlap reshape is exact, then trims back to ``T_test``.
8. **Score length contract / tail-handling**: reference performs an
   EXPLICIT_OTHER "gamma-trim" — discards the first ``pred_dl//2`` and last
   ``pred_dl//2 + (T-pred_dl)%delta`` timesteps from the aligned errors
   **and** the labels (evaluation is on the trimmed length). Since this
   wrapper cannot truncate labels (length contract: ``predict() ->
   (T_test,)``), we fall back to **Option B** for those boundary timesteps:
   head → forward-fill from first valid score; tail → repeat-last valid
   score. See ``predict()`` for the precise comment block.

Interface contract: ``comparison.baseline_common.run_sota_baseline_with_epoch_eval()``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import torch
import torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from .model import NPSR
from comparison.segment_utils import compute_segment_safe_window_indices
from comparison.baselines._per_file_norm import (
    fit_transform_train_per_file,
    transform_test_per_file,
)
from comparison.baselines._boundary_safe_window import per_entity_concat


# ---------------------------------------------------------------------------
# Lightweight dataset helpers (CPU side).
# ---------------------------------------------------------------------------


class _MPtDataset(Dataset):
    """Sliding windows of length ``win_size`` with stride ``stride`` for M_pt.

    Returns ``(B, win_size, F)`` float32 tensors.
    """

    def __init__(self, data: np.ndarray, win_size: int, stride: int):
        self.data = data
        self.win_size = win_size
        self.stride = stride
        if len(data) < win_size:
            raise ValueError(f"len(data)={len(data)} < win_size={win_size}")
        self.n_windows = (len(data) - win_size) // stride + 1

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        start = idx * self.stride
        return torch.from_numpy(self.data[start : start + self.win_size].copy()).float()


class _MSeqDataset(Dataset):
    """Builds (x_cut, y_cut) pairs for M_seq following reference utils.

    For each start position ``si`` (stride = ``delta``):
        full_window = data[si : si + delta_dl]         # delta_dl = pred_dl + delta
        x_cut       = concat(full[:pred_dl//2], full[-pred_dl//2:], axis=0)  # (pred_dl, F)
        y_cut       = data[si + pred_dl//2 : si + pred_dl//2 + delta]        # (delta, F)
    """

    def __init__(self, data: np.ndarray, pred_dl: int, delta: int):
        self.data = data
        self.pred_dl = pred_dl
        self.delta = delta
        self.delta_dl = pred_dl + delta
        if len(data) < self.delta_dl:
            raise ValueError(
                f"len(data)={len(data)} < pred_dl+delta={self.delta_dl}"
            )
        # use stride = delta (reference: for si in range(0, len(trn)-delta_dl+1, delta))
        self.starts = list(range(0, len(data) - self.delta_dl + 1, delta))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        si = self.starts[idx]
        half = self.pred_dl // 2
        full = self.data[si : si + self.delta_dl]
        x_cut = np.concatenate((full[:half], full[-half:]), axis=0)
        y_cut = self.data[si + half : si + half + self.delta]
        return (
            torch.from_numpy(x_cut.copy()).float(),
            torch.from_numpy(y_cut.copy()).float(),
        )


# ---------------------------------------------------------------------------
# Nominality + induced anomaly score (paraphrased from utils/evaluation.py).
# ---------------------------------------------------------------------------


def _nominality_score(delta_xp: np.ndarray, delta_x0: np.ndarray) -> np.ndarray:
    """N(t) = mean((Δ_x0 − Δ_xp)²) / mean(Δ_x0²) along channel axis.

    Parameters
    ----------
    delta_xp : (T, D) — point-AE reconstruction error.
    delta_x0 : (T, D) — sequence-AE reconstruction error.

    Returns (T,) float64 ratio.
    """
    assert delta_xp.shape == delta_x0.shape
    delta_d = delta_x0 - delta_xp
    numer = (delta_d ** 2).mean(axis=-1)
    denom = (delta_x0 ** 2).mean(axis=-1)
    # protect against div-by-zero: anywhere denom is 0, the ratio is 0 (perfectly normal)
    safe_denom = np.where(denom > 0, denom, 1.0)
    out = numer / safe_denom
    out[denom <= 0] = 0.0
    return out


def _induced_anomaly_score(
    nominality_score: np.ndarray,
    anomaly_score: np.ndarray,
    theta_N: float,
    d: int,
    gate_func: str = "soft",
) -> np.ndarray:
    """Induced anomaly score over horizon d, gated by nominality.

    Faithful paraphrase of ``utils/evaluation.get_induced_anomaly_score``.
    Inputs are 1-D arrays of equal length T.
    """
    assert nominality_score.ndim == 1 and anomaly_score.ndim == 1
    assert nominality_score.shape == anomaly_score.shape
    if d < 1:
        return np.copy(anomaly_score)

    if gate_func == "soft":
        gN = 1.0 - nominality_score / theta_N
        gN = np.where(gN < 0, 0.0, gN)
    elif gate_func == "hard":
        gN = 1.0 - nominality_score / theta_N
        gN = np.where(gN < 0, 0.0, gN)
        gN = np.where(gN > 0, 1.0, gN)
    else:
        raise ValueError(f"gate_func must be 'soft' or 'hard', got {gate_func!r}")

    T = len(gN)
    induced = np.copy(anomaly_score).astype(np.float64)

    # denominator schedule (effective neighborhood size, edge-aware)
    denom = np.ones(T) * min(T, 2 * d + 1)
    if d < T - 1:
        denom[:d] = np.minimum(denom[:d], np.arange(d + 1, 2 * d + 1))
        denom[-1:-d - 1:-1] = np.minimum(
            denom[-1:-d - 1:-1], np.arange(d + 1, 2 * d + 1)
        )

    # cumulative-product over forward/backward sliding windows
    # Reference appends d-1 zeros (so first window has length len(gN) before cumprod
    # selects slices). For d=1 the appended length is 0 — handle specially.
    if d == 1:
        gN_forw_cp = np.zeros((0, T))
        gN_back_cp = np.zeros((0, T))
    else:
        gN_forw = sliding_window_view(np.concatenate((gN, np.zeros(d - 1))), T).copy()
        gN_back = np.flip(
            sliding_window_view(np.concatenate((np.zeros(d - 1), gN)), T).copy(),
            axis=0,
        )
        gN_forw_cp = np.cumprod(gN_forw[:, 1:], axis=0)
        gN_back_cp = np.cumprod(gN_back[:, :-1], axis=0)

    A_gN_forw_flip = np.flip(
        np.expand_dims(anomaly_score[:-1], axis=0) * gN_forw_cp, axis=-1
    ) if gN_forw_cp.size else np.zeros((0, T - 1))
    A_gN_back = (
        np.expand_dims(anomaly_score[1:], axis=0) * gN_back_cp
    ) if gN_back_cp.size else np.zeros((0, T - 1))

    # diagonal sums (reference's numer construction)
    if A_gN_forw_flip.size:
        diag_forw = [np.diagonal(A_gN_forw_flip, i).sum() for i in range(T - 1)]
        numer = np.insert(np.flip(diag_forw), 0, 0.0)
    else:
        numer = np.zeros(T)

    if A_gN_back.size:
        diag_back = np.array(
            [np.diagonal(A_gN_back, i).sum() for i in range(T - 1)]
        )
        numer[:-1] += diag_back

    induced += numer * 2 * d / denom
    return induced


# ---------------------------------------------------------------------------
# NPSRBaseline — public wrapper.
# ---------------------------------------------------------------------------


class NPSRBaseline:
    """NPSR (NeurIPS 2023) wrapper for sliding-window MTS anomaly detection.

    Original code: https://github.com/andrewlai61616/NPSR.
    Self-normalizing (Path B). Hyperparameters from
    ``baseline_common._get_default_model_params()['npsr']``.
    """

    def __init__(
        self,
        win_size: int = 100,
        pred_dl: int = 100,
        delta: int = 20,
        z_dim: int = 4,
        ff_mult: int = 4,
        enc_depth: int = 4,
        pred_depth: int = 4,
        n_heads: int = 4,
        dropout: float = 0.0,
        theta_N_ratio: float = 0.9985,
        induction_d: int = 16,
        gate_func: str = "soft",
        clamp_max: float = 4.0,
        clamp_min: float = -4.0,
        lr: float = 1e-4,
        batch_size: int = 64,
        epochs: int = 10,
        train_stride: int = 10,
        device: Optional[str] = None,
        verbose: bool = True,
    ):
        self.win_size = win_size
        self.pred_dl = pred_dl
        self.delta = delta
        self.z_dim = z_dim
        self.ff_mult = ff_mult
        self.enc_depth = enc_depth
        self.pred_depth = pred_depth
        self.n_heads = n_heads
        self.dropout = dropout
        self.theta_N_ratio = theta_N_ratio
        self.induction_d = induction_d
        self.gate_func = gate_func
        self.clamp_max = clamp_max
        self.clamp_min = clamp_min
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_stride = train_stride
        self.verbose = verbose

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model: Optional[NPSR] = None
        self.n_features: Optional[int] = None
        self.train_loss_history: list = []  # list of (mean_pt_loss, mean_seq_loss)

        # Path-B Self-normalization
        self.scaler: Optional[MinMaxScaler] = None
        # Per-source-FILE leak-free scalers (cached from fit; one MinMaxScaler(-1,1)
        # per entity). Multi-file -> N scalers; single-file -> 1 (bit-identical to
        # the legacy whole-array fit). predict() transforms test by these (no test fit).
        self._scalers: Optional[list] = None

        # Cached training data after fit() — used by predict() to recompute θ_N.
        self._train_X_norm: Optional[np.ndarray] = None
        self._theta_N: Optional[float] = None

        # ---- Channel-engineering state (upstream preprocess.py:38-42,78-81 +
        # preprocess_SMD.py:39-41), computed ONCE at fit() and replayed in
        # predict() so train/test channel layout is identical. -----------------
        #   _keep_chns : bool mask over RAW input channels (zero-std drop).
        #   _pad_width : #zero channels PREPENDED to reach a multiple of n_heads
        #                (so Performer's `dim % heads == 0` assert holds and the
        #                configured head count is honoured WITHOUT collapse).
        #   _n_entities: #entities engineered into a one-hot tail (combined path).
        self._keep_chns: Optional[np.ndarray] = None
        self._pad_width: int = 0
        self._n_entities: int = 1
        self._n_features_raw: Optional[int] = None  # RAW channel count seen at fit()

    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return "NPSR"

    # ------------------------------------------------------------------
    # Build / scaler helpers
    # ------------------------------------------------------------------

    def _build_model(self) -> NPSR:
        return NPSR(
            win_size=self.win_size,
            enc_in=self.n_features,
            pred_dl=self.pred_dl,
            delta=self.delta,
            n_heads=self.n_heads,
            enc_depth=self.enc_depth,
            pred_depth=self.pred_depth,
            z_dim=self.z_dim,
            ff_mult=self.ff_mult,
        ).to(self.device)

    # ------------------------------------------------------------------
    # Channel engineering (upstream-faithful) — replicates the upstream
    # data-path so the CONFIGURED head count works (Performer asserts
    # ``dim % heads == 0`` at performer_pytorch/__init__.py:395):
    #   1. zero-std channel drop  (preprocess.py:38-42, single-entity branch)
    #   2. pad feature dim to a multiple of n_heads with PREPENDED zero
    #      channels (the mechanism of preprocess_SMD.py:39-41, "38 sens + 2
    #      add = 40 = 8 heads * 5")
    #   3. entity one-hot tail for the COMBINED multi-entity path
    #      (preprocess.py:78-81, ``train_method == 'train_together'``)
    # ------------------------------------------------------------------

    @staticmethod
    def _seg_count(segs) -> int:
        """#entities implied by per-file segments (1 if None/empty/single)."""
        return len(segs) if segs else 1

    def _pad_to_head_multiple(self, X: np.ndarray) -> np.ndarray:
        """Prepend ``self._pad_width`` zero channels (upstream SMD pad mechanism).

        Zero channels contribute 0 to every Δ and to the channel-mean MSE in
        A(t)/N(t)'s numerator and denominator alike, so they do not change the
        anomaly ordering — their sole purpose is to make ``D`` divisible by the
        head count (preprocess_SMD.py:40-41 prepends zeros for exactly this).
        """
        if self._pad_width <= 0:
            return X
        pad = np.zeros((X.shape[0], self._pad_width), dtype=X.dtype)
        return np.concatenate((pad, X), axis=-1)

    def _append_entity_onehot(self, X: np.ndarray, segs, train_local: bool) -> np.ndarray:
        """Append a per-entity one-hot tail (combined method only).

        Mirrors preprocess.py:78-81: for ``entities > 1`` each entity's rows get
        ``np.eye(entities)[entity_id]`` tiled and concatenated on the channel
        axis. ``segs`` are per-file (lo, hi) in the array's own local coords. For
        a single entity (operative simple jobs) this is a NO-OP — exactly as
        upstream skips the one-hot when ``entities == 1``.
        """
        n_ent = self._n_entities
        if n_ent <= 1 or not segs or len(segs) != n_ent:
            return X
        eye = np.eye(n_ent, dtype=X.dtype)
        onehot = np.zeros((X.shape[0], n_ent), dtype=X.dtype)
        for ei, (lo, hi) in enumerate(segs):
            onehot[lo:hi] = eye[ei]
        return np.concatenate((X, onehot), axis=-1)

    def _normalize_train(self, train_X: np.ndarray, norm_train_segs=None) -> np.ndarray:
        train_X = np.asarray(train_X, dtype=np.float32)
        self._n_features_raw = train_X.shape[1]  # RAW input channel count (pre-engineering)

        # --- (1) Zero-std channel drop (upstream preprocess.py:38-42) ---------
        # Upstream computes keep_chns = (std_trn + std_tst) > 0; the model's input
        # dim is fixed at build time, so the mask MUST be decided here at fit()
        # (test std is unknown until predict()). We therefore use TRAIN std only —
        # leak-free and architecturally forced. A channel constant in TRAIN is a
        # dead channel (its MinMax range collapses to 1.0); dropping it removes
        # the dilution it would otherwise add to the channel-mean of A(t)/N(t).
        self._keep_chns = train_X.std(axis=0) > 0
        if not self._keep_chns.any():
            # degenerate (all-constant) — keep everything to avoid a 0-dim model.
            self._keep_chns = np.ones(train_X.shape[1], dtype=bool)
        train_X = train_X[:, self._keep_chns]

        # --- Per-source-FILE leak-free fit/transform (NPSR's OWN scaler identity:
        # MinMaxScaler(feature_range=(-1, 1)), fit on each file's TRAIN slice).
        # Multi-file -> one scaler per entity; single-file / None segs -> one
        # whole-array scaler (bit-identical to the legacy single-fit path).
        out, scalers, _ = fit_transform_train_per_file(
            train_X, norm_train_segs, lambda: MinMaxScaler(feature_range=(-1, 1))
        )
        self._scalers = scalers
        # Keep `self.scaler` set for save()/back-compat (last per-file scaler;
        # for single-file this IS the one-and-only scaler == legacy behavior).
        self.scaler = scalers[-1] if scalers else None
        out = out.astype(np.float32)

        # --- (3) Entity one-hot tail (combined method, preprocess.py:78-81) ----
        # Operative simple jobs are single-entity -> NO-OP. Append BEFORE padding
        # so the pad target accounts for the final channel count.
        self._n_entities = self._seg_count(norm_train_segs)
        out = self._append_entity_onehot(out, norm_train_segs, train_local=True)

        # --- (2) Pad feature dim to a multiple of n_heads (SMD pad mechanism) --
        D = out.shape[1]
        rem = D % self.n_heads
        self._pad_width = (self.n_heads - rem) if rem else 0
        out = self._pad_to_head_multiple(out)
        return out.astype(np.float32)

    def _normalize_test(self, test_X: np.ndarray, test_segments=None) -> np.ndarray:
        if not self._scalers:
            raise RuntimeError("scaler not fit. Call fit() first.")
        test_X = np.asarray(test_X, dtype=np.float32)
        # Replay the fit-time zero-std channel drop (same mask, same order).
        if self._keep_chns is not None:
            test_X = test_X[:, self._keep_chns]
        # LEAK-FREE: i-th test file transformed by i-th cached TRAIN scaler
        # (.transform only — never fit on test). Single-file / mismatch -> first
        # train scaler over the whole array (same as legacy transform-by-train).
        out = transform_test_per_file(test_X, test_segments, self._scalers).astype(np.float32)
        # reference: clamp test only (global constant; order-safe after per-file transform)
        if self.clamp_max is not None:
            out = np.where(out > self.clamp_max, self.clamp_max, out)
        if self.clamp_min is not None:
            out = np.where(out < self.clamp_min, self.clamp_min, out)
        out = out.astype(np.float32)
        # Replay entity one-hot tail then head-pad (same widths as train).
        out = self._append_entity_onehot(out, test_segments, train_local=False)
        out = self._pad_to_head_multiple(out)
        return out.astype(np.float32)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def fit(self, train_X: np.ndarray, epoch_callback=None, train_segments=None,
            norm_train_segs=None) -> "NPSRBaseline":
        # --- Path B normalization (per-source-FILE MinMax(-1,1) train-only fit) ---
        # `norm_train_segs` (per-file (lo,hi) in TRAIN-LOCAL coords) is SEPARATE from
        # `train_segments` (window-safety, unchanged below). None -> single whole-array file.
        train_X = self._normalize_train(train_X, norm_train_segs=norm_train_segs)
        self.n_features = train_X.shape[1]
        self._train_X_norm = train_X  # cached for predict-time θ_N

        # Build sub-models & per-sub-model Adam (two-optimizer alternating step)
        self.model = self._build_model()
        opt_pt = torch.optim.Adam(self.model.M_pt.parameters(), lr=self.lr)
        opt_seq = torch.optim.Adam(self.model.M_seq.parameters(), lr=self.lr)
        mse = nn.MSELoss()

        # Datasets
        pt_ds = _MPtDataset(train_X, win_size=self.win_size, stride=self.train_stride)
        seq_ds = _MSeqDataset(train_X, pred_dl=self.pred_dl, delta=self.delta)
        if train_segments is not None:
            # M_pt: standard (win_size, train_stride) sliding window — same filter
            # as any other Pattern-A wrapper.
            pt_valid = compute_segment_safe_window_indices(
                train_segments, self.win_size, self.train_stride, len(pt_ds),
            )
            # M_seq: per `_MSeqDataset`, starts = range(0, N - delta_dl + 1, delta),
            # so window length is (pred_dl + delta), stride is delta.
            delta_dl = self.pred_dl + self.delta
            seq_valid = compute_segment_safe_window_indices(
                train_segments, delta_dl, self.delta, len(seq_ds),
            )
            if self.verbose:
                print(f"  Segment-aware: M_pt kept {len(pt_valid)}/{len(pt_ds)} windows "
                      f"({len(pt_valid)/max(len(pt_ds),1):.1%}); "
                      f"M_seq kept {len(seq_valid)}/{len(seq_ds)} pairs "
                      f"({len(seq_valid)/max(len(seq_ds),1):.1%}) — dropped boundary-crossing")
            pt_ds = Subset(pt_ds, pt_valid.tolist())
            seq_ds = Subset(seq_ds, seq_valid.tolist())
        if self.verbose:
            print(f"  M_pt windows : {len(pt_ds)} (win={self.win_size}, stride={self.train_stride})")
            print(f"  M_seq pairs  : {len(seq_ds)} (pred_dl={self.pred_dl}, delta={self.delta})")

        pt_loader = DataLoader(pt_ds, batch_size=self.batch_size, shuffle=True, drop_last=True)
        seq_loader = DataLoader(seq_ds, batch_size=self.batch_size, shuffle=True, drop_last=True)

        self.model.train()
        self.train_loss_history = []
        pt_n_batches = len(pt_loader)
        seq_n_batches = len(seq_loader)

        def _log_batch(phase: str, ep: int, bi: int, tot: int, t0: float) -> None:
            print(f"[BATCH] model={self.name} phase={phase} epoch={ep}/{self.epochs} batch={bi}/{tot} time={time.time() - t0:.3f}s", flush=True)

        for epoch in range(self.epochs):
            # --- M_pt epoch (autoencoder reconstruction) ---
            pt_loss_sum, pt_n = 0.0, 0
            pt_iter = (
                tqdm(pt_loader, desc=f"E{epoch+1}/{self.epochs} M_pt", leave=False)
                if self.verbose else pt_loader
            )
            for batch_idx, batch in enumerate(pt_iter):
                batch_start = time.time()
                x = batch.to(self.device)
                opt_pt.zero_grad()
                pred = self.model.M_pt(x)
                loss = mse(pred, x)
                loss.backward()
                opt_pt.step()
                pt_loss_sum += float(loss.item())
                pt_n += 1
                _log_batch("M_pt", epoch + 1, batch_idx + 1, pt_n_batches, batch_start)

            # --- M_seq epoch (induction reconstruction) ---
            seq_loss_sum, seq_n = 0.0, 0
            seq_iter = (
                tqdm(seq_loader, desc=f"E{epoch+1}/{self.epochs} M_seq", leave=False)
                if self.verbose else seq_loader
            )
            for batch_idx, (x_cut, y_cut) in enumerate(seq_iter):
                batch_start = time.time()
                x_cut = x_cut.to(self.device)
                y_cut = y_cut.to(self.device)
                opt_seq.zero_grad()
                pred = self.model.M_seq(x_cut)
                loss = mse(pred, y_cut)
                loss.backward()
                opt_seq.step()
                seq_loss_sum += float(loss.item())
                seq_n += 1
                _log_batch("M_seq", epoch + 1, batch_idx + 1, seq_n_batches, batch_start)

            mean_pt = pt_loss_sum / max(pt_n, 1)
            mean_seq = seq_loss_sum / max(seq_n, 1)
            self.train_loss_history.append((mean_pt, mean_seq))
            if self.verbose:
                print(f"  Epoch {epoch + 1}: M_pt={mean_pt:.6f}  M_seq={mean_seq:.6f}")

            if epoch_callback is not None:
                epoch_callback(self, epoch + 1)
                self.model.train()
        return self

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_pt_error(self, data: np.ndarray, win_size: int) -> np.ndarray:
        """Run M_pt over non-overlap windows; return per-timestep error (T_eff, F).

        Uses wrap-around padding (reference ``x_trn_no_rep``-style) so the
        reshape is exact.
        """
        F = self.n_features
        if len(data) < win_size:
            raise ValueError(f"data too short: {len(data)} < {win_size}")
        # wrap-around padding so length is a multiple of win_size
        pad_len = (-len(data)) % win_size
        if pad_len:
            padded = np.concatenate((data, data[:pad_len]), axis=0)
        else:
            padded = data
        windows = padded.reshape(-1, win_size, F).astype(np.float32)

        bs = self.batch_size
        err_chunks = []
        self.model.eval()
        for i in range(0, len(windows), bs):
            batch = torch.from_numpy(windows[i : i + bs]).to(self.device)
            recon = self.model.M_pt(batch).cpu().numpy()
            err_chunks.append(recon - windows[i : i + bs])
        err = np.concatenate(err_chunks, axis=0).reshape(-1, F)
        # trim wrap-around tail
        if pad_len:
            err = err[: len(data)]
        return err

    @torch.no_grad()
    def _compute_seq_error(self, data: np.ndarray) -> np.ndarray:
        """Run M_seq over non-overlap (x_cut, y_cut) pairs; return error covering
        the central delta blocks. Output shape: (T_seq, F) where T_seq covers
        ``[pred_dl//2, pred_dl//2 + N_pairs * delta)``.
        """
        F = self.n_features
        half = self.pred_dl // 2
        delta = self.delta
        delta_dl = self.pred_dl + delta

        starts = list(range(0, len(data) - delta_dl + 1, delta))
        if len(starts) == 0:
            raise ValueError(f"data too short for M_seq: {len(data)} < {delta_dl}")

        x_cuts = np.empty((len(starts), self.pred_dl, F), dtype=np.float32)
        y_cuts = np.empty((len(starts), delta, F), dtype=np.float32)
        for j, si in enumerate(starts):
            full = data[si : si + delta_dl]
            x_cuts[j, :half] = full[:half]
            x_cuts[j, half:] = full[-half:]
            y_cuts[j] = data[si + half : si + half + delta]

        bs = self.batch_size
        err_chunks = []
        self.model.eval()
        for i in range(0, len(x_cuts), bs):
            batch = torch.from_numpy(x_cuts[i : i + bs]).to(self.device)
            recon = self.model.M_seq(batch).cpu().numpy()
            err_chunks.append(recon - y_cuts[i : i + bs])
        err = np.concatenate(err_chunks, axis=0).reshape(-1, F)  # (N_pairs * delta, F)
        return err

    def _align_pt_to_seq(self, pt_err: np.ndarray, seq_len: int) -> np.ndarray:
        """Slice point-error to the same temporal span the sequence error covers.

        Sequence covers timesteps ``[pred_dl//2, pred_dl//2 + seq_len)``.
        """
        half = self.pred_dl // 2
        return pt_err[half : half + seq_len]

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, test_X: np.ndarray, test_segments=None) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        F = test_X.shape[1]
        # Compare against the RAW (pre channel-engineering) feature count: the
        # zero-std drop / pad / one-hot are replayed inside `_normalize_test`, so
        # `test_X` here is still raw-width. (`self.n_features` is the engineered
        # width baked into the model.)
        expected_raw = self._n_features_raw if self._n_features_raw is not None else self.n_features
        if F != expected_raw:
            raise ValueError(f"feature mismatch: test has {F}, fit had {expected_raw} (raw)")

        # Path-B normalization on test (per-source-FILE transform-by-TRAIN-scaler,
        # leak-free, with clamp on test only). `test_segments` = per-file (lo,hi) in
        # TEST-LOCAL coords; None -> single whole-array file (legacy behavior).
        test_X_n = self._normalize_test(test_X, test_segments=test_segments)

        # --- training-set reference for θ_N (one-shot per predict; cached if epoch_callback drives it) ---
        train_X = self._train_X_norm
        if train_X is None:
            raise RuntimeError("Train data cache missing — wrapper must be fit() first.")

        # ---- θ_N : single global training-set threshold (computed ONCE, NOT
        # test windowing) -------------------------------------------------------
        # The reference computes θ_N from the (concatenated) TRAIN nominality and
        # uses ONE θ_N across all entities (utils.py). θ_N depends only on TRAIN
        # data — it is invariant to how the TEST series is sliced — so it is
        # computed here, before the per-entity test loop, and passed UNCHANGED
        # into every entity's induction. Routing θ_N per-entity would CHANGE the
        # score formula (different threshold per entity), which is forbidden.
        trn_err_pt = self._compute_pt_error(train_X, win_size=self.win_size)  # (T_trn, F)
        trn_err_seq = self._compute_seq_error(train_X)                          # (N*delta, F)
        trn_pt_aligned = self._align_pt_to_seq(trn_err_pt, len(trn_err_seq))
        trn_Nt = _nominality_score(delta_xp=trn_pt_aligned, delta_x0=trn_err_seq)
        if len(trn_Nt) == 0:
            theta_N = 1.0
        else:
            q_idx = int(len(trn_Nt) * self.theta_N_ratio)
            q_idx = min(max(q_idx, 0), len(trn_Nt) - 1)
            theta_N = float(np.sort(trn_Nt)[q_idx])
        # Guard against degenerate θ_N=0
        if not np.isfinite(theta_N) or theta_N <= 0:
            theta_N = max(float(np.nanmean(trn_Nt) if len(trn_Nt) else 1.0), 1e-8)
        self._theta_N = theta_N

        # ---- RAW per-entity producer ----------------------------------------
        # NPSR's official evaluation is *explicitly* per-entity:
        #   utils/evaluation.py:get_induced_anomaly_score carries the comment
        #   "# note that for multi-entity datasets, only one entity should be
        #    input at a time".
        # So the entire test-side chain — non-overlap (no_rep) windowing of M_pt
        # & M_seq, the (x_cut,y_cut) reshape, error alignment, nominality + base
        # anomaly score, the induced-anomaly-score sliding cumulative product,
        # and the gamma-trim head/tail fill — must run on ONE entity at a time.
        # `per_entity_concat` runs this `raw_fn` on each entity's own test slice
        # (TEST-LOCAL `test_segments` from `get_file_norm_segments()`, the SAME
        # source as the per-file normalization above), so no window can cross an
        # entity boundary, and concatenates the per-timestep raw scores.
        # Single-entity / None / non-tiling segments -> ONE call over the whole
        # `test_X_n` == bit-identical legacy behaviour (helper guarantees no-op).
        head_len = self.pred_dl // 2
        delta_dl = self.pred_dl + self.delta

        def raw_fn(sub_X: np.ndarray) -> np.ndarray:
            """Per-entity RAW NPSR scores for ``sub_X`` (Li, F) -> (Li,) float32.

            Edge-safe: an entity slice too short to window for M_pt
            (``Li < win_size``) or M_seq (``Li < pred_dl + delta``) cannot
            produce an induced score, so it falls back to an all-zeros
            (perfectly-nominal) score of the slice length — never crashes.
            """
            L = len(sub_X)
            out = np.empty(L, dtype=np.float32)
            # Short-slice fallback: not enough timesteps to window either branch.
            if L < self.win_size or L < delta_dl:
                out.fill(0.0)
                return out

            # M_pt / M_seq test errors (non-overlap windowing, this slice only).
            err_pt = self._compute_pt_error(sub_X, win_size=self.win_size)  # (L, F)
            err_seq = self._compute_seq_error(sub_X)                         # (N*delta, F)
            pt_aligned = self._align_pt_to_seq(err_pt, len(err_seq))
            Nt = _nominality_score(delta_xp=pt_aligned, delta_x0=err_seq)
            At = (pt_aligned ** 2).mean(axis=-1)  # reference tst_At

            # Induced anomaly score (default d=16, gate='soft') over THIS entity.
            d_eff = max(1, min(self.induction_d, len(At) - 1))
            induced = _induced_anomaly_score(
                nominality_score=Nt,
                anomaly_score=At,
                theta_N=theta_N,
                d=d_eff,
                gate_func=self.gate_func,
            )

            # ---- Reference inference behavior (re-verified 2026-06-02) ----
            # https://raw.githubusercontent.com/andrewlai61616/NPSR/main/utils/utils.py
            #   lab_c = lab[pred_dl//2 : len(lab) - (len(lab)-pred_dl)%delta - pred_dl//2]
            # Reference EXPLICIT_OTHER "gamma-trim": discards the first `pred_dl//2`
            # ("left gamma") and last `pred_dl//2 + (len-pred_dl)%delta` ("right
            # gamma + remainder") timesteps from BOTH the aligned errors AND the
            # labels. Our `induced` already covers exactly `lab_c`'s span
            # [pred_dl//2, pred_dl//2 + N_pairs*delta). `./comparison/` cannot
            # truncate labels (length contract: predict() -> (T_test,)), so per
            # dispatch we use Option B for the boundary timesteps:
            #   head (left gamma): forward-fill from first valid score
            #   tail (right gamma + remainder): repeat-last valid score
            valid_len = len(induced)
            if valid_len == 0:
                out.fill(0.0)
                return out
            valid_end = head_len + valid_len
            if valid_end > L:  # pathological short-input clamp
                valid_end = L
                valid_len_eff = valid_end - head_len
                out[head_len:valid_end] = induced[:valid_len_eff].astype(np.float32)
                valid_first = float(induced[0])
                valid_last = float(induced[valid_len_eff - 1])
            else:
                out[head_len:valid_end] = induced.astype(np.float32)
                valid_first = float(induced[0])
                valid_last = float(induced[-1])
            if head_len > 0:
                out[:head_len] = valid_first
            tail_len = L - valid_end
            if tail_len > 0:
                out[valid_end:] = valid_last
            return out

        # Boundary-safe: each entity windowed/scored independently, then concat.
        scores = per_entity_concat(test_X_n, test_segments, raw_fn)

        # ---- Whole-test post-proc: finite safety net (granularity UNCHANGED) --
        # Applied ONCE on the concatenated raw scores, exactly as before — a
        # global NaN/Inf scrub, not a re-aggregation across entities.
        if not np.isfinite(scores).all():
            finite_max = float(np.nanmax(scores[np.isfinite(scores)])) if np.isfinite(scores).any() else 0.0
            scores = np.where(np.isfinite(scores), scores, finite_max)
            scores = scores.astype(np.float32)

        return scores

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, save_dir: Path) -> None:
        if self.model is None:
            raise RuntimeError("No model to save.")
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), save_dir / "model.pt")
        if self.scaler is not None:
            joblib.dump(self.scaler, save_dir / "scaler.pkl")
        if self._train_X_norm is not None:
            # cache compressed for re-load (predict() needs it to recompute θ_N)
            np.savez_compressed(save_dir / "train_cache.npz", train_X_norm=self._train_X_norm)
        config = {
            "model_name": "NPSR",
            "win_size": self.win_size,
            "pred_dl": self.pred_dl,
            "delta": self.delta,
            "z_dim": self.z_dim,
            "ff_mult": self.ff_mult,
            "enc_depth": self.enc_depth,
            "pred_depth": self.pred_depth,
            "n_heads": self.n_heads,
            "dropout": self.dropout,
            "theta_N_ratio": self.theta_N_ratio,
            "induction_d": self.induction_d,
            "gate_func": self.gate_func,
            "clamp_max": self.clamp_max,
            "clamp_min": self.clamp_min,
            "n_features": self.n_features,
            # Channel-engineering state — required to replay drop/pad/one-hot in
            # predict() after a load() (model input dim is the engineered width).
            "n_features_raw": self._n_features_raw,
            "keep_chns": self._keep_chns.tolist() if self._keep_chns is not None else None,
            "pad_width": self._pad_width,
            "n_entities": self._n_entities,
        }
        with open(save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    def load(self, save_dir: Path) -> "NPSRBaseline":
        save_dir = Path(save_dir)
        with open(save_dir / "config.json", "r") as f:
            config = json.load(f)
        for k in (
            "win_size", "pred_dl", "delta", "z_dim", "ff_mult", "enc_depth",
            "pred_depth", "n_heads", "dropout", "theta_N_ratio", "induction_d",
            "gate_func", "clamp_max", "clamp_min", "n_features",
        ):
            setattr(self, k, config[k])
        # Restore channel-engineering state (back-compat: older saves lack it).
        self._n_features_raw = config.get("n_features_raw", config["n_features"])
        kc = config.get("keep_chns")
        self._keep_chns = np.asarray(kc, dtype=bool) if kc is not None else None
        self._pad_width = int(config.get("pad_width", 0))
        self._n_entities = int(config.get("n_entities", 1))
        self.model = self._build_model()
        self.model.load_state_dict(torch.load(save_dir / "model.pt", map_location=self.device))
        self.model.eval()
        scaler_path = save_dir / "scaler.pkl"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
            # Reconstruct the per-file scaler cache so predict() (which transforms
            # test via the cached TRAIN scalers, leak-free) works after load().
            # save() persists only the single (last) scaler, so the restored cache
            # is length-1 -> transform_test_per_file uses it whole-array (legacy).
            self._scalers = [self.scaler] if self.scaler is not None else None
        cache_path = save_dir / "train_cache.npz"
        if cache_path.exists():
            self._train_X_norm = np.load(cache_path)["train_X_norm"].astype(np.float32)
        return self
