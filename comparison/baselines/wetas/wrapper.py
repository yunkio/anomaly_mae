"""
WETAS — Weakly supervised Temporal Anomaly Segmentation (ICCV 2021, Dongha Lee et al.)

Pipeline adapter (thin) around the VENDORED official code:
  - `comparison/baselines/wetas/model.py`        = upstream `model.py`        (GPL-3.0, donalee/WETAS@cb149dc)
  - `comparison/baselines/wetas/softdtw_cuda.py` = upstream `softdtw_cuda.py` (MIT, Maghoumi)
Both vendored verbatim except device-agnostic edits (see each file's header).

This wrapper re-hosts ONLY the training/inference glue (not vendored): it builds windows,
derives weak labels, runs the official loss, exposes `epoch_callback`, and reconstructs a
per-timestep `(N_test,)` continuous score.

NORMALIZATION (2026-05-30 fidelity rework). The pipeline now classifies WETAS as
SELF_NORMALIZING_WEAK (run_baseline.py:240): it hands this wrapper RAW, un-normalized data
(normalize_mode='none'). The wrapper therefore reproduces the ORIGINAL-PAPER normalization
itself — **per-recording StandardScaler (z-score)**, fit and transformed independently for
each recording (donalee/WETAS `timeseries.py:35-40`, where `_preprocess` runs once per file
in the `_load_data` loop at `timeseries.py:21-24`). This replaces the previous (now-removed)
assumption that the pipeline pre-applied a global MinMax scaler.

It deliberately does NOT reuse:
  * upstream valid-F1 early stopping (best-epoch selection is the pipeline's job via epoch_callback);
  * upstream `utils.py` metric code (frozen pipeline metric layer replaces it).

CONTRACT (mirrors comparison/baselines/tranad/model.py:TranADBaseline):
  WETASBaseline(**hparams)
  .fit(train_X, train_y=None, epoch_callback=None, train_segments=None) -> self
  .predict(test_X) -> (N_test,) float32, higher = more anomalous
  .save(save_dir) / .load(save_dir)

References to the official code use `file:line` at commit cb149dc.
"""

import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler  # upstream normalization (timeseries.py:6,38)

from comparison.baselines.wetas.model import DilatedCNN
from comparison.baselines.wetas.softdtw_cuda import SoftDTW
from comparison.segment_utils import compute_segment_safe_window_indices


class WETASBaseline:
    """WETAS weakly-supervised anomaly-segmentation baseline (Q1 only; Q3 N/A).

    Defaults are upstream argparse defaults (train_classifier.py:218-242), except `epochs`
    which the pipeline governs (--sota-epochs), per 08_HYPERPARAMETER_TABLE.
    """

    def __init__(
        self,
        # --- architecture (train_classifier.py:218-221) ---
        hidden_size: int = 128,       # :218
        output_size: int = 128,       # :219  (must equal hidden_size — fc in-features quirk, see 04_ARCHITECTURE_AUDIT)
        kernel_size: int = 2,         # :220
        n_layers: int = 7,            # :221  -> receptive field 2^7 = 128
        # --- WETAS framework (train_classifier.py:225-242) ---
        pooling_type: str = 'avg',    # :225  (argparse default 'avg' overrides model.py ctor default 'max')
        local_threshold: float = 0.3, # :226  (pseudo-label binarize; NOT used to threshold our emitted score)
        granularity: int = 4,         # :227
        beta: float = 0.1,            # :228  alignment hinge margin (model.py:166)
        gamma: float = 0.1,           # :229  soft-DTW smoothing
        split_size: int = 500,        # :242  window length (non-overlapping chunk)
        # --- optimization (train_classifier.py:232-234) ---
        batch_size: int = 32,         # :232
        epochs: int = 10,             # :233 upstream default 200; pipeline sets via --sota-epochs
        lr: float = 1e-4,             # :234
        # --- adapter-only knobs ---
        train_stride: Optional[int] = None,  # None -> non-overlapping (== split_size), faithful to timeseries.py
        seq_len: Optional[int] = None,       # alias for split_size (pipeline naming); split_size wins if both set
        device: Optional[str] = None,
        verbose: bool = True,
    ):
        # `seq_len` is the pipeline's generic window name; map it onto split_size if split_size left default.
        if seq_len is not None:
            split_size = int(seq_len)

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.pooling_type = pooling_type
        self.local_threshold = local_threshold
        self.granularity = granularity
        self.beta = beta
        self.gamma = gamma
        self.split_size = int(split_size)
        self.seq_len = self.split_size  # exposed for the pipeline's segment-aware helpers
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        # Non-overlapping chunks by default (timeseries.py:46). train_stride lets the pipeline
        # densify training windows; predict ALWAYS uses non-overlapping + left-pad-drop (07_SCORING).
        self.train_stride = int(train_stride) if train_stride is not None else self.split_size
        self.verbose = verbose

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        # SoftDTW kernel: FORCED CPU @jit path (use_cuda=False) — 2026-06-01.
        # Rationale: this env's `numba.cuda` cannot init (`cuInit → CUDA_ERROR_NO_DEVICE 100`)
        # — verified via 3-case test matrix. numba is conda-managed and CLAUDE.md
        # forbids `pip install` in `dc_vis`. CPU `@jit` is the only viable kernel;
        # the encoder still trains on GPU via torch, only soft-DTW is CPU-bound.
        # Original (kept for reference): self.use_cuda = (self.device.type == 'cuda')
        self.use_cuda = False

        self.model: Optional[DilatedCNN] = None
        self.n_features: int = 0
        self.train_loss_history = []
        # NOTE wetas does not persist a fitted StandardScaler instance — it
        # re-runs ``_normalize_per_recording`` from RAW train_X at every fit()
        # entry (deterministic given identical inputs), so resume only needs
        # to restore model + optimizer + RNG, not scaler params.

        # Resume infrastructure (2026-05-31, B option — see _checkpoint.py).
        self._resume_checkpoint_path: Optional[Path] = None
        self._checkpoint_dir: Optional[Path] = None

    @property
    def name(self) -> str:
        return "WETAS"

    # ------------------------------------------------------------------ windowing
    def _windows_with_stride(self, data: np.ndarray, stride: int):
        """Sliding windows of length split_size with given stride (for TRAINING).

        Returns (windows (n, split_size, F), starts (n,)). No padding — windows that
        would run past the end are simply not emitted (mirrors a standard sliding dataset).
        """
        n = len(data)
        L = self.split_size
        if n < L:
            # single left-padded window (front zero-pad), like timeseries.py:44 for short series
            pad = L - n
            w = np.zeros((1, L, data.shape[1]), dtype=np.float32)
            w[0, pad:] = data
            return w, np.array([0], dtype=np.int64), pad
        starts = list(range(0, n - L + 1, stride))
        windows = np.stack([data[s:s + L] for s in starts]).astype(np.float32)
        return windows, np.array(starts, dtype=np.int64), 0

    def _nonoverlap_chunks_leftpad(self, data: np.ndarray):
        """Non-overlapping split_size chunks with FRONT zero-pad on the first chunk.

        Faithful to upstream `timeseries.py:44` (`f.pad(..., (split_size - N % split_size, 0))`).
        Returns (chunks (n_chunks, split_size, F), pad:int). The recovered flattened series
        equals `flatten[pad:]` -> exactly N timesteps (used by predict).
        """
        n = len(data)
        L = self.split_size
        rem = n % L
        pad = (L - rem) if rem != 0 else 0
        if pad > 0:
            padded = np.zeros((n + pad, data.shape[1]), dtype=np.float32)
            padded[pad:] = data
        else:
            padded = data.astype(np.float32)
        n_chunks = len(padded) // L
        chunks = padded.reshape(n_chunks, L, data.shape[1])
        return chunks, pad

    # ------------------------------------------------------------------ normalization (timeseries.py:35-40)
    @staticmethod
    def _normalize_per_recording(data: np.ndarray, segments) -> np.ndarray:
        """Per-recording StandardScaler(z-score), faithful to upstream `timeseries.py:35-40`.

        Upstream `_preprocess` (timeseries.py:35) is invoked once per recording file inside the
        `_load_data` loop (timeseries.py:21-24); each call does `scaler = StandardScaler();
        scaler.fit(data); data = scaler.transform(data)` on that recording ALONE. We replicate
        that by fitting/transforming one StandardScaler per `(start, end)` segment.

        `segments` is the pipeline's recording-boundary list in `data`-coordinates
        (run_baseline.py:434/455 -> loader.normal_segments / get_boundary_train_segments).
        When `segments` is None/empty the whole array is treated as ONE recording (single
        StandardScaler.fit on all of `data`), which is exactly upstream behaviour for a
        single-file split.
        """
        data = np.asarray(data, dtype=np.float32)
        out = np.empty_like(data, dtype=np.float32)
        ranges = None
        if segments:
            ranges = [(int(s), int(e)) for (s, e) in segments if int(e) > int(s)]
        if not ranges:
            ranges = [(0, len(data))]

        covered = np.zeros(len(data), dtype=bool)
        for (s, e) in ranges:
            s = max(0, s)
            e = min(len(data), e)
            if e <= s:
                continue
            scaler = StandardScaler()                 # timeseries.py:38
            out[s:e] = scaler.fit_transform(data[s:e]).astype(np.float32)  # timeseries.py:39-40 (per recording)
            covered[s:e] = True
        # Any gap not spanned by a segment (defensive; should not happen for contiguous
        # boundary lists) is z-scored against its own stats so no row is left un-normalized.
        if not covered.all():
            scaler = StandardScaler()
            out[~covered] = scaler.fit_transform(data[~covered]).astype(np.float32)
        return out

    @staticmethod
    def _normalize_test_transductive(data: np.ndarray) -> np.ndarray:
        """Test-time z-score. Upstream also fits a FRESH StandardScaler per *test* recording
        (test files run through the same `_preprocess`, timeseries.py:35-40). Our `predict`
        contract receives only `test_X` with NO per-recording boundaries, so per-recording test
        normalization is infeasible. The most faithful feasible substitute is a TRANSDUCTIVE
        z-score over the whole `test_X` (fit a fresh StandardScaler on the test array, then
        transform it). This preserves the upstream principle that test data is standardized by
        statistics derived from the TEST data itself (never train stats); the only residual gap
        is granularity (whole-test vs per-recording). See `normalization_approach` in
        CODE_REWORK_NOTES.md.
        """
        data = np.asarray(data, dtype=np.float32)
        if len(data) == 0:
            return data
        scaler = StandardScaler()
        return scaler.fit_transform(data).astype(np.float32)

    # ------------------------------------------------------------------ fit
    def fit(self, train_X: np.ndarray, train_y=None, epoch_callback=None, train_segments=None) -> 'WETASBaseline':
        """Train WETAS on RAW (un-normalized) `train_X`.

        Normalization (timeseries.py:35-40): the wrapper now receives raw data
        (run_baseline.py SELF_NORMALIZING_WEAK) and applies the ORIGINAL-PAPER per-recording
        StandardScaler(z-score) itself — one scaler fit/transformed per `train_segments`
        recording (single scaler over all of `train_X` when no segments are supplied).
        Weak labels (leak-free): wlabel = max(train_y over each window)  (timeseries.py:54).
        Loss (train_classifier.py:127-129): BCE(wscore, wlabel) + dtw_loss(output, wlabel).mean(0).
        Optimizer: Adam(lr) (train_classifier.py:113).  No scheduler (upstream has none).
        """
        train_X = np.asarray(train_X, dtype=np.float32)
        self.n_features = train_X.shape[1]

        if train_y is None:
            raise RuntimeError("WETAS requires labeled anomalies (Q1 only; Q3 N/A).")
        train_y = np.asarray(train_y).astype(np.float32).reshape(-1)
        if len(train_y) != len(train_X):
            raise RuntimeError(
                f"WETAS: train_y length {len(train_y)} != train_X length {len(train_X)}."
            )

        # Original-paper normalization: per-recording StandardScaler (timeseries.py:35-40).
        # Applied to RAW train_X BEFORE windowing so windows + weak labels are built on the
        # z-scored series exactly as upstream `_preprocess` feeds the model.
        train_X = self._normalize_per_recording(train_X, train_segments)
        if self.verbose:
            n_rec = len([1 for (s, e) in (train_segments or []) if int(e) > int(s)]) or 1
            print(f"  Normalization: per-recording StandardScaler (z-score) over {n_rec} recording(s) "
                  f"[timeseries.py:35-40]")

        # Build training windows (sliding with train_stride; default = non-overlapping).
        windows, starts, _ = self._windows_with_stride(train_X, self.train_stride)

        # Segment-aware: drop windows that cross a run/train boundary (leak-safe, 06_LABEL_SEMANTICS).
        if train_segments is not None:
            valid_idx = compute_segment_safe_window_indices(
                train_segments, self.split_size, self.train_stride, len(windows),
            )
            if len(valid_idx) > 0:
                windows = windows[valid_idx]
                starts = starts[valid_idx]
                if self.verbose:
                    print(f"  Segment-aware: kept {len(valid_idx)} boundary-safe windows")

        # Weak labels: max of point labels over each window (timeseries.py:54).
        wlabels = np.stack([
            float(train_y[s:s + self.split_size].max()) for s in starts
        ]).astype(np.float32)

        n_pos = int((wlabels >= 0.5).sum())
        n_neg = int((wlabels < 0.5).sum())
        if n_pos == 0:
            # Every window normal -> BCE collapses + pos/neg alignment split degenerates (05/06 audit).
            raise RuntimeError("WETAS requires labeled anomalies (Q1 only; Q3 N/A).")
        if self.verbose:
            print(f"{self.name} training:")
            print(f"  Device: {self.device} (use_cuda={self.use_cuda})")
            print(f"  Windows: {len(windows)} (split_size={self.split_size}, stride={self.train_stride})")
            print(f"  Weak labels: {n_pos} anomalous / {n_neg} normal")

        # Build the soft-DTW module + model (vendored; device-agnostic).
        dtw = SoftDTW(use_cuda=self.use_cuda, gamma=self.gamma, normalize=False)  # train_classifier.py:97
        self.model = DilatedCNN(
            input_size=self.n_features,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            kernel_size=self.kernel_size,
            n_layers=self.n_layers,
            pooling_type=self.pooling_type,
            local_threshold=self.local_threshold,
            granularity=self.granularity,
            beta=self.beta,
            split_size=self.split_size,
            dtw=dtw,
        ).to(self.device)

        bce = nn.BCELoss(reduction='mean')                                   # train_classifier.py:112
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)    # train_classifier.py:113

        win_t = torch.from_numpy(windows)            # (n, L, F)
        wlab_t = torch.from_numpy(wlabels)           # (n,)
        dataset = torch.utils.data.TensorDataset(win_t, wlab_t)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True,               # train_classifier.py:91 shuffle=True
        )

        self.train_loss_history = []

        # Attempt resume — restores model/optimizer/RNG/loss_history (B option).
        start_epoch = self._maybe_resume(optimizer)
        start_time = time.time()

        for epoch in range(start_epoch, self.epochs):
            self.model.train()
            epoch_losses = []
            for batch_x, batch_w in loader:
                batch_x = batch_x.to(self.device)   # (B, L, F)
                batch_w = batch_w.to(self.device)   # (B,)

                out = self.model.get_scores(batch_x)                          # model.py:118
                bce_loss = bce(out['wscore'], batch_w)                        # train_classifier.py:127
                dtw_loss = self.model.dtw_loss(out['output'], batch_w).mean(0)  # train_classifier.py:128
                loss = bce_loss + dtw_loss                                    # train_classifier.py:129

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            avg_loss = float(np.mean(epoch_losses)) if epoch_losses else float('nan')
            self.train_loss_history.append(avg_loss)
            if self.verbose:
                print(f"  Epoch [{epoch + 1}/{self.epochs}] loss={avg_loss:.6f} "
                      f"elapsed={time.time() - start_time:.1f}s", flush=True)

            # Save last.pt BEFORE epoch_callback (see deepmil rationale).
            self._save_last_checkpoint(epoch + 1, optimizer)

            if epoch_callback is not None:
                epoch_callback(self, epoch + 1)
                self.model.train()

        return self

    # ------------------------------------------------------------------ predict
    def predict(self, test_X: np.ndarray) -> np.ndarray:
        """Per-timestep continuous anomaly score, shape (N_test,), higher = more anomalous.

        Uses the continuous dense `dscore` (model.py:134), NOT binary `dpred`/`dscore>=thr`.
        Normalization: RAW `test_X` is z-scored with a transductive StandardScaler over the
        whole test array (most faithful feasible substitute for upstream's per-recording test
        StandardScaler, since the contract provides no test-segment boundaries — see
        `_normalize_test_transductive`).
        Reconstruction (07_SCORING_AGGREGATION):
          1. Split test_X into non-overlapping split_size chunks with FRONT zero-pad on the
             first chunk (timeseries.py:44).
          2. Encoder -> dscore (n_chunks, split_size); flatten to padded length.
          3. Drop the `pad` front entries -> exactly N_test timesteps (`flatten[pad:]`).
        np.nan_to_num applied. len(out) == len(test_X) guaranteed.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        test_X = np.asarray(test_X, dtype=np.float32)
        n = len(test_X)

        # Original-paper normalization at test time (timeseries.py:35-40, transductive variant).
        test_X = self._normalize_test_transductive(test_X)

        chunks, pad = self._nonoverlap_chunks_leftpad(test_X)   # (n_chunks, L, F), pad
        self.model.eval()

        all_dscore = []
        with torch.no_grad():
            for i in range(0, len(chunks), self.batch_size):
                batch = torch.from_numpy(chunks[i:i + self.batch_size]).to(self.device)  # (B, L, F)
                out = self.model.get_scores(batch)
                dscore = out['dscore']                  # (B, L) continuous, model.py:134
                all_dscore.append(dscore.detach().cpu().numpy())

        flat = np.concatenate(all_dscore, axis=0).reshape(-1).astype(np.float32)  # padded length
        # Drop the FRONT zero-pad to recover the original timeline (timeseries.py:44 pads front).
        scores = flat[pad:]

        # Length integrity safeguard (forward/tail-fill any residual mismatch).
        if len(scores) > n:
            scores = scores[:n]
        elif len(scores) < n:
            fill = scores[-1] if len(scores) > 0 else 0.0
            scores = np.concatenate([scores, np.full(n - len(scores), fill, dtype=np.float32)])

        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return scores

    # ------------------------------------------------------------------ resume (2026-05-31, B)
    def set_resume(self, ckpt_path) -> None:
        """Enable resume from a previously-saved last.pt (called pre-fit())."""
        self._resume_checkpoint_path = Path(ckpt_path) if ckpt_path else None

    def set_checkpoint_dir(self, ckpt_dir) -> None:
        """Enable per-epoch last.pt save (called pre-fit())."""
        self._checkpoint_dir = Path(ckpt_dir) if ckpt_dir else None

    def _maybe_resume(self, optimizer) -> int:
        if self._resume_checkpoint_path is None or not self._resume_checkpoint_path.exists():
            return 0
        from comparison.baselines._checkpoint import load_checkpoint, restore_rng_state
        ckpt = load_checkpoint(self._resume_checkpoint_path)
        ck_hp = ckpt.get('wrapper_hparams', {})
        # Architecture + dynamics knobs must match — silent partial restore on
        # a mismatched encoder shape would corrupt training subtly. Widened
        # 2026-05-31 (Bug 4) to also reject dynamics-only changes
        # (output_size, granularity, etc.) which silently re-route training
        # without raising load_state_dict shape errors.
        for key in ('split_size', 'hidden_size', 'output_size', 'n_layers',
                    'kernel_size', 'pooling_type', 'granularity', 'beta',
                    'gamma', 'local_threshold', 'n_features'):
            if ck_hp.get(key) != getattr(self, key):
                raise RuntimeError(
                    f"WETAS resume hparam mismatch on '{key}': "
                    f"checkpoint={ck_hp.get(key)} vs current={getattr(self, key)}."
                )
        self.model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.train_loss_history = list(ckpt.get('train_loss_history', []))
        restore_rng_state(ckpt.get('rng_state'))
        start_epoch = int(ckpt['epoch'])
        if self.verbose:
            print(f"  [RESUME] WETAS from epoch {start_epoch}/{self.epochs} "
                  f"({self._resume_checkpoint_path.name})", flush=True)
        return start_epoch

    def _save_last_checkpoint(self, current_epoch: int, optimizer) -> None:
        if self._checkpoint_dir is None:
            return
        from comparison.baselines._checkpoint import (
            atomic_save, capture_rng_state, CHECKPOINT_VERSION,
        )
        ckpt = {
            'checkpoint_version': CHECKPOINT_VERSION,
            'model_name': self.name,
            'epoch': current_epoch,
            'target_epochs': self.epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss_history': list(self.train_loss_history),
            'wrapper_hparams': {
                'split_size': self.split_size, 'hidden_size': self.hidden_size,
                'output_size': self.output_size, 'kernel_size': self.kernel_size,
                'n_layers': self.n_layers, 'pooling_type': self.pooling_type,
                'local_threshold': self.local_threshold,
                'granularity': self.granularity, 'beta': self.beta,
                'gamma': self.gamma, 'batch_size': self.batch_size,
                'lr': self.lr, 'train_stride': self.train_stride,
                'n_features': self.n_features,
            },
            'rng_state': capture_rng_state(),
        }
        atomic_save(ckpt, self._checkpoint_dir / 'last.pt')

    # ------------------------------------------------------------------ persistence
    def save(self, save_dir) -> None:
        if self.model is None:
            raise RuntimeError("No model to save.")
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), save_dir / "model.pt")
        config = {
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "kernel_size": self.kernel_size,
            "n_layers": self.n_layers,
            "pooling_type": self.pooling_type,
            "local_threshold": self.local_threshold,
            "granularity": self.granularity,
            "beta": self.beta,
            "gamma": self.gamma,
            "split_size": self.split_size,
            "n_features": self.n_features,
            "model_name": self.name,
        }
        with open(save_dir / "config.json", 'w') as fp:
            json.dump(config, fp, indent=2)

    def load(self, save_dir) -> 'WETASBaseline':
        save_dir = Path(save_dir)
        with open(save_dir / "config.json", 'r') as fp:
            config = json.load(fp)
        self.hidden_size = config["hidden_size"]
        self.output_size = config["output_size"]
        self.kernel_size = config["kernel_size"]
        self.n_layers = config["n_layers"]
        self.pooling_type = config["pooling_type"]
        self.local_threshold = config["local_threshold"]
        self.granularity = config["granularity"]
        self.beta = config["beta"]
        self.gamma = config["gamma"]
        self.split_size = int(config["split_size"])
        self.seq_len = self.split_size
        self.n_features = config["n_features"]

        dtw = SoftDTW(use_cuda=self.use_cuda, gamma=self.gamma, normalize=False)
        self.model = DilatedCNN(
            input_size=self.n_features,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            kernel_size=self.kernel_size,
            n_layers=self.n_layers,
            pooling_type=self.pooling_type,
            local_threshold=self.local_threshold,
            granularity=self.granularity,
            beta=self.beta,
            split_size=self.split_size,
            dtw=dtw,
        ).to(self.device)
        self.model.load_state_dict(torch.load(save_dir / "model.pt", map_location=self.device))
        self.model.eval()
        return self

    def __repr__(self) -> str:
        return (f"WETASBaseline(split_size={self.split_size}, hidden_size={self.hidden_size}, "
                f"beta={self.beta}, gamma={self.gamma})")
