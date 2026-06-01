"""
TreeMIL pipeline adapter (thin wrapper around the vendored Pyramid core).

Implements the comparison-pipeline contract: fit(train_X, train_y, ...) /
predict(test_X) -> (N_test,) RAW continuous anomaly scores.

TreeMIL is WEAKLY (timestamp) SUPERVISED — it consumes anomaly labels at train
time (window-level weak labels = max of point labels over each non-overlapping
window). It is therefore Q1-only; Q3 (normal-only) is N/A (every weak label
collapses to 0 -> degenerate BCE). See docs 05_/06_.

NORMALIZATION (original-paper fidelity, 2026-05-30 rework)
---------------------------------------------------------
Upstream TreeMIL normalizes EACH input FILE with a per-feature z-score
``StandardScaler`` *fit on the whole file* before windowing:

    fly-orange/TreeMIL  ``timeseries.py`` (commit 16f166c)
        line  6:  ``from sklearn.preprocessing import StandardScaler``
        line 52:  ``# normalize``
        line 53:  ``scaler = StandardScaler()``
        line 54:  ``scaler.fit(data)``         # data = full (T, D) file
        line 55:  ``data = scaler.transform(data)``
    (``_preprocess`` is invoked once per filename inside ``_load_data``'s loop,
     lines 21-24, so the scaler is per-file; train/valid/test are slices of the
     SAME normalized file, lines 32-40 — i.e. one scaler fit per file, applied
     transductively to all three splits.)

The comparison pipeline classifies TreeMIL as SELF_NORMALIZING_WEAK and now
hands this wrapper **RAW (un-normalized) data**, so the wrapper applies that
StandardScaler itself (NOT the external global-minmax). See `predict` for the
test-side feasibility note (no per-recording boundaries are passed to predict).

GPL-3.0 NOTICE: the vendored model core (model.py) is a GPL-3.0 derivative of
fly-orange/TreeMIL. This adapter file is original glue code; the package as a
whole is governed by GPL-3.0.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from .model import Pyramid, PyraMIL


# Upstream effective defaults (train.py construction call + argparse; see 08_).
_DEFAULTS = dict(
    split_size=500,   # window length == seq_size (non-overlapping)
    epochs=200,
    batch_size=32,
    lr=1e-4,
    ary_size=2,
    inner_size=3,
    d_model=128,
    d_k=128,
    d_v=128,
    d_inner_hid=32,
    n_head=5,
    n_layer=2,
    dropout=0.5,
    agg_type='max',
    pooling_type='max',
)


class TreeMILBaseline:
    """Weakly-supervised TreeMIL baseline adapter."""

    def __init__(self, **hparams):
        self.name = "TreeMIL"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        h = dict(_DEFAULTS)
        h.update({k: v for k, v in hparams.items() if v is not None})

        self.split_size = int(h['split_size'])
        self.seq_len = self.split_size          # contract alias: seq_len == window
        self.epochs = int(h['epochs'])
        self.batch_size = int(h['batch_size'])
        self.lr = float(h['lr'])
        self.ary_size = int(h['ary_size'])
        self.inner_size = int(h['inner_size'])
        self.d_model = int(h['d_model'])
        self.d_k = int(h['d_k'])
        self.d_v = int(h['d_v'])
        self.d_inner_hid = int(h['d_inner_hid'])
        self.n_head = int(h['n_head'])
        self.n_layer = int(h['n_layer'])
        self.dropout = float(h['dropout'])
        self.agg_type = str(h['agg_type'])
        self.pooling_type = str(h['pooling_type'])

        # Non-overlapping windows upstream; train_stride retained for contract
        # parity (default 1) but TreeMIL windowing is stride==split_size.
        self.train_stride = int(hparams.get('train_stride', 1))

        self.n_features = None
        self.model = None
        self.loss_fn = PyraMIL()
        self.train_loss_history = []  # populated by fit(); persisted in checkpoint.

        # Resume infrastructure (2026-05-31, B option — see _checkpoint.py).
        self._resume_checkpoint_path: Optional[Path] = None
        self._checkpoint_dir: Optional[Path] = None

    # ------------------------------------------------------------------ utils
    def _build_model(self, n_features):
        self.n_features = int(n_features)
        self.model = Pyramid(
            input_size=self.n_features,
            seq_size=self.split_size,
            ary_size=self.ary_size,
            inner_size=self.inner_size,
            d_model=self.d_model,
            n_layer=self.n_layer,
            n_head=self.n_head,
            d_k=self.d_k,
            d_v=self.d_v,
            d_inner_hid=self.d_inner_hid,
            agg_type=self.agg_type,
            dropout=self.dropout,
            pooling_type=self.pooling_type,
            granularity=1,
            local_threshold=0.5,
            global_threshold=0.5,
            beta=10,
            dtw=None,
        ).to(self.device)

    def _segment_ranges(self, n_total, train_segments):
        """Yield (start, end) ranges to window over.

        If train_segments provided, window WITHIN each segment (so no window
        crosses a normal-segment / run-boundary cut). Otherwise the whole array
        is one range.
        """
        if train_segments:
            ranges = [(int(s), int(e)) for (s, e) in train_segments if int(e) > int(s)]
            if ranges:
                return ranges
        return [(0, int(n_total))]

    @staticmethod
    def _zscore_per_file(X_seg):
        """Per-feature z-score on a RAW per-file (per-recording) slice.

        Faithful re-implementation of upstream `timeseries.py:53-55`:
            scaler = StandardScaler(); scaler.fit(data); data = scaler.transform(data)
        StandardScaler sets scale_=1.0 for zero-variance columns, so this never
        divides by zero (matches sklearn semantics exactly; we use sklearn here).
        """
        return StandardScaler().fit_transform(X_seg.astype(np.float64)).astype(np.float32)

    def _make_train_windows(self, X, y, train_segments):
        """Non-overlapping windows + weak labels. Matches timeseries.py:53-71.

        - PER-FILE z-score normalization first: each `train_segments` range is
          one upstream "file" (run-boundary-delimited recording), so a fresh
          StandardScaler is fit+applied to that RAW slice BEFORE windowing —
          exactly upstream `_preprocess` (one scaler per file, lines 53-55).
        - Windows of length split_size, stride == split_size (NON-overlapping).
        - A trailing partial window (len < split_size) is DROPPED at train time
          (upstream head-pads it; for training we simply skip incomplete
          windows to avoid feeding synthetic zeros as a labeled bag).
        - wlabel = max(point labels over the window)  [train split only].
        """
        n = X.shape[0]
        win_list, lab_list = [], []
        for (s, e) in self._segment_ranges(n, train_segments):
            seg_len = e - s
            n_full = seg_len // self.split_size
            if n_full == 0:
                continue
            X_norm = self._zscore_per_file(X[s:e])     # per-file StandardScaler (raw -> z)
            for w in range(n_full):
                a = w * self.split_size
                b = a + self.split_size
                win_list.append(X_norm[a:b])
                lab_list.append(float(y[s + a:s + b].max()))
        if not win_list:
            return None, None
        windows = np.stack(win_list, axis=0).astype(np.float32)      # (B, T, F)
        wlabels = np.asarray(lab_list, dtype=np.float32)             # (B,)
        return windows, wlabels

    # -------------------------------------------------------------------- fit
    def fit(self, train_X, train_y=None, epoch_callback=None, train_segments=None):
        train_X = np.asarray(train_X, dtype=np.float32)
        if train_X.ndim != 2:
            raise ValueError(f"TreeMIL expects train_X of shape (N, F); got {train_X.shape}")

        if train_y is None:
            raise RuntimeError("TreeMIL requires labeled anomalies (Q1 only; Q3 N/A)")
        train_y = np.asarray(train_y).astype(np.float32).reshape(-1)

        windows, wlabels = self._make_train_windows(train_X, train_y, train_segments)
        if windows is None:
            raise RuntimeError(
                "TreeMIL requires labeled anomalies (Q1 only; Q3 N/A): "
                f"no complete window of length split_size={self.split_size} could be built."
            )
        if wlabels.max() <= 0.0:
            raise RuntimeError(
                "TreeMIL requires labeled anomalies (Q1 only; Q3 N/A): "
                "no positive (anomalous) window in the train split."
            )

        self._build_model(train_X.shape[1])

        X_t = torch.from_numpy(windows)        # (B, T, F)
        y_t = torch.from_numpy(wlabels)        # (B,)
        n_win = X_t.shape[0]

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Attempt resume — restores model/optimizer/RNG/loss_history (B option).
        start_epoch = self._maybe_resume(optimizer)

        for epoch in range(start_epoch, self.epochs):
            self.model.train()
            perm = torch.randperm(n_win)
            epoch_losses = []
            for i in range(0, n_win, self.batch_size):
                idx = perm[i:i + self.batch_size]
                data = X_t[idx].to(self.device)
                wlabel = y_t[idx].to(self.device)

                out = self.model.get_scores(data)
                loss = self.loss_fn.last_loss(out['wscore'], wlabel)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))

            avg = float(np.mean(epoch_losses)) if epoch_losses else float('nan')
            self.train_loss_history.append(avg)

            # Save last.pt BEFORE epoch_callback (crash in eval -> still resumable).
            self._save_last_checkpoint(epoch + 1, optimizer)

            if epoch_callback is not None:
                epoch_callback(self, epoch + 1)

        return self

    # ---------------------------------------------------------------- predict
    @torch.no_grad()
    def predict(self, test_X):
        """Return (N_test,) float32 RAW continuous per-timestep anomaly scores.

        Normalization (original-paper fidelity + feasibility note)
        ----------------------------------------------------------
        Upstream applies a per-FILE z-score StandardScaler (timeseries.py:53-55),
        fit on the whole file that contains the test region. The pipeline contract
        gives `predict` only `test_X` (NO per-recording boundaries), so the most
        faithful feasible analog is a **transductive per-test-file z-score**: fit
        one StandardScaler on the ENTIRE `test_X` and transform it. This matches
        upstream's "scaler fit on the file, applied to its test split" — the only
        residual gap is that upstream fits per-file while here, lacking the test
        run-boundaries, we fit ONE scaler over the whole test array (if test_X
        concatenates several recordings, upstream would fit one scaler each).
        Train and test scalers are intentionally separate, exactly as upstream
        (each file gets its own fresh StandardScaler).

        Non-overlapping contiguous windows of length split_size; the LAST window
        is TAIL-padded with zeros to fill split_size (NEVER upstream left/head
        pad — that would shift every score). Dense scores are flattened in window
        order and TRUNCATED to len(test_X). See doc 06_.
        """
        if self.model is None:
            raise RuntimeError("TreeMIL.predict called before fit().")

        test_X = np.asarray(test_X, dtype=np.float32)
        if test_X.ndim != 2:
            raise ValueError(f"TreeMIL expects test_X of shape (N, F); got {test_X.shape}")
        n = test_X.shape[0]
        T = self.split_size

        # Per-test-file z-score (transductive; closest feasible to upstream
        # per-file StandardScaler.fit on the test region — see docstring).
        test_X = self._zscore_per_file(test_X)

        n_win = (n + T - 1) // T               # ceil
        padded_len = n_win * T
        if padded_len > n:
            pad = np.zeros((padded_len - n, test_X.shape[1]), dtype=np.float32)
            test_pad = np.concatenate([test_X, pad], axis=0)   # TAIL pad
        else:
            test_pad = test_X

        windows = test_pad.reshape(n_win, T, test_X.shape[1])   # (B, T, F)

        self.model.eval()
        scores = np.empty((n_win, T), dtype=np.float32)
        for i in range(0, n_win, self.batch_size):
            batch = torch.from_numpy(windows[i:i + self.batch_size]).to(self.device)
            out = self.model.get_scores(batch)
            scores[i:i + batch.shape[0]] = out['dscore'].detach().cpu().numpy()

        flat = scores.reshape(-1)[:n]          # flatten in window order, truncate tail pad
        flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return flat

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
        for key in ('split_size', 'ary_size', 'd_model', 'n_layer', 'n_head', 'n_features'):
            if ck_hp.get(key) != getattr(self, key):
                raise RuntimeError(
                    f"TreeMIL resume hparam mismatch on '{key}': "
                    f"checkpoint={ck_hp.get(key)} vs current={getattr(self, key)}."
                )
        self.model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.train_loss_history = list(ckpt.get('train_loss_history', []))
        restore_rng_state(ckpt.get('rng_state'))
        start_epoch = int(ckpt['epoch'])
        print(f"  [RESUME] TreeMIL from epoch {start_epoch}/{self.epochs} "
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
            'model_name': 'TreeMIL',
            'epoch': current_epoch,
            'target_epochs': self.epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss_history': list(self.train_loss_history),
            'wrapper_hparams': {
                'split_size': self.split_size, 'ary_size': self.ary_size,
                'inner_size': self.inner_size, 'd_model': self.d_model,
                'd_k': self.d_k, 'd_v': self.d_v,
                'd_inner_hid': self.d_inner_hid, 'n_head': self.n_head,
                'n_layer': self.n_layer, 'dropout': self.dropout,
                'agg_type': self.agg_type, 'pooling_type': self.pooling_type,
                'lr': self.lr, 'batch_size': self.batch_size,
                'n_features': self.n_features,
            },
            'rng_state': capture_rng_state(),
        }
        atomic_save(ckpt, self._checkpoint_dir / 'last.pt')

    # -------------------------------------------------------------- save/load
    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        if self.model is not None:
            torch.save(self.model.state_dict(), os.path.join(save_dir, "treemil_model.pt"))
        meta = dict(
            n_features=self.n_features, split_size=self.split_size,
            ary_size=self.ary_size, inner_size=self.inner_size,
            d_model=self.d_model, d_k=self.d_k, d_v=self.d_v,
            d_inner_hid=self.d_inner_hid, n_head=self.n_head,
            n_layer=self.n_layer, dropout=self.dropout,
            agg_type=self.agg_type, pooling_type=self.pooling_type,
        )
        np.save(os.path.join(save_dir, "treemil_meta.npy"), meta, allow_pickle=True)

    def load(self, save_dir):
        meta = np.load(os.path.join(save_dir, "treemil_meta.npy"), allow_pickle=True).item()
        # Restore architecture hparams so the rebuilt model matches the checkpoint.
        self.split_size = int(meta['split_size']); self.seq_len = self.split_size
        self.ary_size = int(meta['ary_size']); self.inner_size = int(meta['inner_size'])
        self.d_model = int(meta['d_model']); self.d_k = int(meta['d_k']); self.d_v = int(meta['d_v'])
        self.d_inner_hid = int(meta['d_inner_hid']); self.n_head = int(meta['n_head'])
        self.n_layer = int(meta['n_layer']); self.dropout = float(meta['dropout'])
        self.agg_type = str(meta['agg_type']); self.pooling_type = str(meta['pooling_type'])
        self._build_model(meta['n_features'])
        state = torch.load(os.path.join(save_dir, "treemil_model.pt"), map_location=self.device)
        self.model.load_state_dict(state)
        return self
