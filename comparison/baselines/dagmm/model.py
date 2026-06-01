"""DAGMM Baseline — TranAD-author implementation (VLDB 2022).

Implementation reference (line-by-line):
    https://github.com/imperial-qore/TranAD
    - Model class: ``src/models.py::DAGMM`` (lines 81-123)
    - Windowing: ``main.py::convert_to_windows`` (lines 18-24)
    - Training/inference: ``main.py::backprop`` DAGMM branch (lines 79-104)
    - Optimizer/scheduler: ``main.py::load_model`` (lines 60-61)
    - Epochs: ``main.py:310`` (``num_epochs = 5``)

Paper / model citation (preserved):
    Zong et al., "Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly
    Detection", ICLR 2018. https://openreview.net/forum?id=BJJLHbb0-

Source-of-truth boundary
------------------------
* Method-internal behaviour (windowing, architecture, loss, scoring, hyperparameters):
  TranAD-author code above.
* Everything else (driver min-max normalization, train/test split, threshold,
  PA@K/AUC/F1, visualization, result directories) is owned by the ``./comparison``
  pipeline and intentionally left untouched.

Pipeline adaptations vs upstream TranAD (documented to avoid hidden divergence)
-------------------------------------------------------------------------------
The following three behaviours deviate from TranAD's literal ``main.py`` flow
in ways that match standard ``./comparison`` SOTA-wrapper practice. None changes
the loss / optimizer / scoring math — only the iteration order over training
samples. All other SOTA wrappers (anomaly_transformer, tranad, usad, gdn,
omnianomaly, gcn_lstm, catch, ...) use the same pattern.

* **Batched training (``batch_size=256``)** — TranAD's ``main.py:84-92``
  iterates per-sample (``for d in data: ...``). We use a ``DataLoader`` with
  ``batch_size=256`` (pipeline default, overridable via ``MODEL_PRESETS`` or
  ``--sota-epochs``). Loss is identical (``mean(MSE) + mean(MSE)``); only the
  per-step gradient estimate is batch-averaged rather than per-sample. Same
  optimizer (AdamW 1e-4, wd=1e-5) and same scheduler (StepLR 5/0.9) per epoch.
* **``shuffle=True``** — TranAD iterates training samples in input order; we
  shuffle each epoch (standard best-practice for SGD optimization, applied
  uniformly across all our SOTA wrappers). The objective and final model are
  the same in expectation.
* **``drop_last=True``** — final partial batch dropped each epoch (≤ B-1
  samples = ≤0.1% per epoch for our datasets). Avoids the irregular last-batch
  gradient that PyTorch would otherwise produce. Negligible effect; consistent
  across SOTA wrappers.

TranAD's DAGMM is **not** the original ICLR'18 energy-based model: the upstream
authors replaced the GMM energy + cov regularizer with a simple two-MSE objective
(``MSE(x_hat, x) + MSE(gamma, x)``) and score the LAST row of each window's
reconstruction — this wrapper mirrors that exactly. The original DAGMM paper
citation is preserved as the *model* citation; the *implementation* citation is
TranAD's repo (paper-author code, widely cited in TS-AD benchmarks).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from comparison.baselines._boundary_safe_window import per_entity_concat
from comparison.segment_utils import compute_segment_safe_window_indices


# ---------------------------------------------------------------------------
# Windowing (TranAD main.py::convert_to_windows, lines 18-24)
# ---------------------------------------------------------------------------


def _convert_to_windows(data: np.ndarray, n_window: int) -> np.ndarray:
    """Left-padded, length-preserving sliding window flatten for DAGMM.

    Mirrors ``main.py::convert_to_windows`` for the non-TranAD/non-Attention branch:
    each timestep ``i`` in ``[0, T)`` is replaced by the previous ``n_window``
    timesteps; for ``i < n_window`` the prefix is repeated copies of ``data[0]``.
    The resulting window is then flattened.

    Args:
        data: ``(T, F)`` float32.
        n_window: window length (TranAD default 5).

    Returns:
        ``(T, n_window * F)`` float32 — length preserved.
    """
    T, F_ = data.shape
    span = n_window * F_
    windows = np.empty((T, span), dtype=np.float32)
    first_row = data[0:1]  # (1, F)
    for i in range(T):
        if i >= n_window:
            w = data[i - n_window:i]
        else:
            pad = np.repeat(first_row, n_window - i, axis=0)
            w = np.concatenate([pad, data[0:i]], axis=0)
        windows[i] = w.reshape(-1)
    return windows


# ---------------------------------------------------------------------------
# Model (TranAD src/models.py::DAGMM, lines 81-123)
# ---------------------------------------------------------------------------


class DAGMMModel(nn.Module):
    """DAGMM as implemented by the TranAD authors.

    Architecture (vector-batched; numerically identical to the upstream per-sample
    ``x.view(1, -1)`` formulation but ~100× faster):
      - Encoder: ``Linear(5F, 16) → Tanh → Linear(16, 16) → Tanh → Linear(16, 8)``
      - Decoder: ``Linear(8, 16) → Tanh → Linear(16, 16) → Tanh → Linear(16, 5F) → Sigmoid``
      - Estimate: ``Linear(10, 16) → Tanh → Dropout(0.5) → Linear(16, 5F) → Softmax(dim=1)``

    ``n_gmm = n_feats * n_window`` (upstream choice — gamma has the same shape as the
    flattened input and is trained as a second reconstruction head).
    """

    def __init__(
        self,
        n_feats: int,
        n_window: int = 5,
        n_hidden: int = 16,
        n_latent: int = 8,
    ):
        super().__init__()
        self.n_feats = n_feats
        self.n_window = n_window
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.n = n_feats * n_window          # flattened window dim
        self.n_gmm = n_feats * n_window      # estimate output dim (upstream choice)

        self.encoder = nn.Sequential(
            nn.Linear(self.n, self.n_hidden), nn.Tanh(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.Tanh(),
            nn.Linear(self.n_hidden, self.n_latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.n_latent, self.n_hidden), nn.Tanh(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.Tanh(),
            nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
        )
        self.estimate = nn.Sequential(
            nn.Linear(self.n_latent + 2, self.n_hidden), nn.Tanh(), nn.Dropout(0.5),
            nn.Linear(self.n_hidden, self.n_gmm), nn.Softmax(dim=1),
        )

    def compute_reconstruction(self, x: torch.Tensor, x_hat: torch.Tensor):
        """Reconstruction features (TranAD models.py::compute_reconstruction).

        Returns per-row scalars ``(rel_eucl, cos_sim)`` matching upstream:
          * ``(x - x_hat).norm(2, dim=1) / x.norm(2, dim=1)``
          * ``F.cosine_similarity(x, x_hat, dim=1)``

        Numerical guard: ``x.norm(2, dim=1) + 1e-12``. Upstream has no guard;
        we add ``1e-12`` strictly to avoid a division-by-zero NaN when a
        single training row is entirely zero (a degenerate min-max edge case).
        ``1e-12`` is sub-ULP relative to the typical post-Sigmoid / post-minmax
        magnitudes, so it does not perturb the reconstruction feature values
        for any non-degenerate row — purely a safety net.
        """
        diff_norm = (x - x_hat).norm(p=2, dim=1)
        x_norm = x.norm(p=2, dim=1) + 1e-12
        rel_eucl = diff_norm / x_norm
        cos_sim = F.cosine_similarity(x, x_hat, dim=1)
        return rel_eucl, cos_sim

    def forward(self, x: torch.Tensor):
        """Batched forward over flattened windows.

        Args:
            x: ``(B, 5F)`` flattened windows.

        Returns:
            z_c:   ``(B, 8)`` latent
            x_hat: ``(B, 5F)`` reconstruction (Sigmoid output, values in [0, 1])
            z:     ``(B, 10)`` latent + 2 reconstruction features
            gamma: ``(B, 5F)`` softmax estimate (trained as second reconstruction head)
        """
        z_c = self.encoder(x)
        x_hat = self.decoder(z_c)
        rec_1, rec_2 = self.compute_reconstruction(x, x_hat)
        z = torch.cat([z_c, rec_1.unsqueeze(-1), rec_2.unsqueeze(-1)], dim=1)
        gamma = self.estimate(z)
        return z_c, x_hat, z, gamma


# ---------------------------------------------------------------------------
# Wrapper (Domain B contract — fit / predict / epoch_callback)
# ---------------------------------------------------------------------------


class DAGMMBaseline:
    """DAGMM baseline wrapper for the ``./comparison`` pipeline.

    Internal behaviour mirrors TranAD's DAGMM (see module docstring); the
    interface follows the SOTA wrapper contract used by
    ``comparison.baseline_common.run_sota_baseline_with_epoch_eval``:

      * ``fit(train_X, epoch_callback=None, train_segments=None) -> self``
      * ``predict(test_X) -> np.ndarray (T_test,) float32``

    All hyperparameter defaults match upstream (``lr=1e-4, wd=1e-5,
    StepLR(5, 0.9), n_window=5, n_hidden=16, n_latent=8, epochs=5``). The wrapper
    silently accepts the legacy preset keys (``seq_len``, ``latent_dim``,
    ``n_gmm``, ``lambda_energy``, ``lambda_cov``, ``hidden_dims``) so existing
    ``MODEL_PRESETS`` entries keep importing — they map onto the TranAD defaults
    instead (TranAD doesn't expose those knobs).
    """

    def __init__(
        self,
        n_window: int = 5,
        n_hidden: int = 16,
        n_latent: int = 8,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        lr_step_size: int = 5,
        lr_gamma: float = 0.9,
        epochs: int = 5,
        batch_size: int = 256,
        train_stride: int = 1,
        device: Optional[str] = None,
        verbose: bool = True,
        **legacy_kwargs,  # absorb seq_len/latent_dim/n_gmm/lambda_*/hidden_dims
    ):
        # Honour `seq_len` as an alias for `n_window` if explicitly provided —
        # earlier presets used `seq_len=5` to mean the same thing.
        if 'seq_len' in legacy_kwargs:
            try:
                seq_len_val = int(legacy_kwargs.pop('seq_len'))
                if seq_len_val > 0:
                    n_window = seq_len_val
            except (TypeError, ValueError):
                pass

        # Drop other legacy keys silently (no longer meaningful under TranAD impl).
        for _legacy_key in ('latent_dim', 'n_gmm', 'lambda_energy', 'lambda_cov', 'hidden_dims'):
            legacy_kwargs.pop(_legacy_key, None)

        if legacy_kwargs:
            # Unknown extra kwargs — surface them so we don't silently ignore typos.
            unknown = ', '.join(sorted(legacy_kwargs))
            raise TypeError(f"DAGMMBaseline received unexpected kwargs: {unknown}")

        self.n_window = int(n_window)
        self.n_hidden = int(n_hidden)
        self.n_latent = int(n_latent)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.lr_step_size = int(lr_step_size)
        self.lr_gamma = float(lr_gamma)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.train_stride = int(train_stride)
        self.verbose = bool(verbose)
        # seq_len attr kept for preset/CLI introspection paths
        self.seq_len = self.n_window

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model: Optional[DAGMMModel] = None
        self.n_features: int = 0
        self.train_loss_history: list = []

    @property
    def name(self) -> str:
        return "DAGMM"

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        train_X: np.ndarray,
        epoch_callback=None,
        train_segments=None,
    ) -> 'DAGMMBaseline':
        """Train DAGMM following TranAD's ``backprop`` DAGMM branch (training=True).

        Loss per batch (upstream main.py:84-92):
            l1 = MSE(x_hat, x_window)    # element-wise (B, 5F)
            l2 = MSE(gamma, x_window)    # element-wise (B, 5F)
            loss = mean(l1) + mean(l2)

        Optimizer: ``AdamW(lr=1e-4, weight_decay=1e-5)`` (main.py:60).
        Scheduler: ``StepLR(step_size=5, gamma=0.9)`` (main.py:61) — step per epoch.

        Args:
            train_X: ``(T_train, n_features)`` float32, already normalized by driver.
            epoch_callback: optional ``callable(self, epoch_one_indexed)`` invoked
                at the end of each epoch (used by pipeline for per-epoch eval).
            train_segments: optional list of ``(start, end)`` tuples from
                ``UnifiedLoader.normal_segments``. When provided (normalonly variant)
                we drop windows whose left edge would fall before its right edge's
                segment — preserving paper-faithful behaviour for the standard
                (full-train) variant while keeping segment safety for normalonly.
        """
        self.n_features = int(train_X.shape[1])
        self.model = DAGMMModel(
            n_feats=self.n_features,
            n_window=self.n_window,
            n_hidden=self.n_hidden,
            n_latent=self.n_latent,
        ).to(self.device)

        if self.verbose:
            print(f"{self.name} training (TranAD-author impl):")
            print(f"  Device: {self.device}")
            print(f"  n_window={self.n_window}, n_hidden={self.n_hidden}, n_latent={self.n_latent}")
            print(f"  n_feats={self.n_features}, flatten dim={self.n_features * self.n_window}")
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"  Total parameters: {total_params:,}")

        # 1) Left-padded, length-preserving windowing (T windows out of T timesteps)
        windows = _convert_to_windows(train_X.astype(np.float32), self.n_window)
        T_total = len(windows)

        # 2) Segment-aware filtering for normalonly variant.
        #    TranAD-style window at idx i covers data[i-n_window:i] for i >= n_window
        #    and a left-pad-of-data[0] + data[0:i] for i < n_window. Under normalonly
        #    the left-padded windows are semantically unreliable (they repeat data[0]
        #    of the concat'd train), so we drop them here. For i >= n_window, the
        #    pipeline helper checks whether [i - n_window, i] fits inside any single
        #    normal segment.
        if train_segments is not None:
            # Helper enumerates starts s = i * stride for i in [0, n_total). Here we
            # want s = i - n_window for i in [n_window, T) — equivalent to stride=1,
            # n_total = T - n_window, then re-map idx back to T-space.
            n_starts = max(T_total - self.n_window, 0)
            valid_starts = compute_segment_safe_window_indices(
                train_segments, self.n_window, 1, n_starts
            )
            valid_t_idx = (valid_starts + self.n_window).astype(np.int64)
            kept = len(valid_t_idx)
            if self.verbose:
                print(
                    f"  Segment-aware: kept {kept}/{T_total} windows "
                    f"({kept / max(T_total, 1):.1%}) — dropped boundary-crossing + "
                    f"left-padded (i < {self.n_window})"
                )
            windows = windows[valid_t_idx]

        if len(windows) == 0:
            if self.verbose:
                print("  WARNING: no training windows after windowing/filtering — fit aborted.")
            return self

        dataset = TensorDataset(torch.from_numpy(windows))
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma
        )
        criterion = nn.MSELoss(reduction='none')

        self.model.train()
        n_batches_total = len(loader)
        start_time = time.time()
        for epoch in range(self.epochs):
            epoch_l1, epoch_l2 = 0.0, 0.0
            n_batches = 0
            for batch_idx, (batch_x,) in enumerate(loader):
                batch_start = time.time()
                batch_x = batch_x.to(self.device)             # (B, 5F)
                _, x_hat, _, gamma = self.model(batch_x)      # both (B, 5F)
                l1 = criterion(x_hat, batch_x)
                l2 = criterion(gamma, batch_x)
                loss = torch.mean(l1) + torch.mean(l2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_l1 += float(torch.mean(l1).item())
                epoch_l2 += float(torch.mean(l2).item())
                n_batches += 1
                # [BATCH] line for the monitoring/snapshot infrastructure
                print(
                    f"[BATCH] model={self.name} epoch={epoch + 1}/{self.epochs} "
                    f"batch={batch_idx + 1}/{n_batches_total} "
                    f"time={time.time() - batch_start:.3f}s",
                    flush=True,
                )
            scheduler.step()

            avg_l1 = epoch_l1 / max(n_batches, 1)
            avg_l2 = epoch_l2 / max(n_batches, 1)
            self.train_loss_history.append(avg_l1 + avg_l2)

            if self.verbose:
                elapsed = time.time() - start_time
                remaining = elapsed / (epoch + 1) * (self.epochs - epoch - 1)
                print(
                    f"  Epoch {epoch + 1}/{self.epochs}: "
                    f"L1={avg_l1:.6f}, L2={avg_l2:.6f}, "
                    f"lr={optimizer.param_groups[0]['lr']:.3e} | "
                    f"ETA: {remaining:.0f}s"
                )

            if epoch_callback is not None:
                epoch_callback(self, epoch + 1)
                self.model.train()

        if self.verbose and self.train_loss_history:
            total_time = time.time() - start_time
            print(
                f"\n  Training complete: {total_time:.1f}s total, "
                f"Final loss: {self.train_loss_history[-1]:.6f}"
            )
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, test_X: np.ndarray, test_segments=None) -> np.ndarray:
        """Per-timestep anomaly score (TranAD ``backprop`` training=False branch).

        Score at timestep ``t`` = mean over features of ``(x_hat - x)^2`` taken on
        the LAST row of the window that ends at ``t`` (the current-step row,
        ``data.shape[1] - feats : data.shape[1]`` slice in upstream).

        Boundary-safe TEST windowing
        ----------------------------
        ``_convert_to_windows`` builds a length-preserving window for every timestep
        ``i`` from ``data[i - n_window:i]`` (history) with a head-fill (repeat
        ``data[0]``) for ``i < n_window``. On a multi-entity test array (SMD machines,
        SMAP-MSL channels, Exathlon apps), the history slice of a window near an entity
        boundary would otherwise pull rows from the PREVIOUS entity, corrupting the
        boundary-region scores. We therefore route the ENTIRE raw-score producer —
        windowing + inference + last-row squared-error per-timestep mapping (incl. the
        head-fill) — through ``per_entity_concat``, which runs ``_raw`` INDEPENDENTLY on
        each entity's own test slice (so no window can span a boundary) and concatenates.

        ``test_segments``: per-entity ``(lo, hi)`` TEST-LOCAL list from the SAME unified
        source as per-entity normalization (``get_file_norm_segments()`` test side).
        ``None`` / single-entity / non-tiling -> exactly ONE call over the whole array
        == bit-identical legacy behaviour (the helper guarantees the no-op).

        There is NO whole-test score post-processing in DAGMM (the per-timestep raw
        squared-error IS the final score), so nothing is applied after concatenation —
        the score granularity is identical to before for the single-entity path.

        Edge case: an entity slice shorter than ``n_window`` does NOT crash —
        ``_convert_to_windows`` only ever indexes ``data[0:i]`` / ``data[i-n_window:i]``
        with ``i < len(slice)``, and head-fills the remainder from that entity's own
        ``data[0]`` (faithful to upstream's first-row repeat, now entity-local).

        Returns:
            ``(T_test,)`` float32. Domain B contract — no trimming, no padding.
        """
        if self.model is None:
            raise RuntimeError("DAGMMBaseline.predict called before fit()")

        self.model.eval()
        test_X = np.asarray(test_X, dtype=np.float32)
        n_feats = self.n_features

        def _raw(sub_X: np.ndarray) -> np.ndarray:
            """RAW per-timestep score for ONE entity's test slice (no cross-entity dep).

            Mirrors the upstream per-slice logic exactly: length-preserving windowing,
            batched inference, last-row squared error mean over features.
            """
            sub_X = np.asarray(sub_X, dtype=np.float32)
            windows = _convert_to_windows(sub_X, self.n_window)
            T = len(windows)
            scores = np.zeros(T, dtype=np.float32)
            with torch.no_grad():
                for start in range(0, T, self.batch_size):
                    stop = min(start + self.batch_size, T)
                    batch_x_np = windows[start:stop]
                    batch_x = torch.from_numpy(batch_x_np).to(self.device)
                    _, x_hat, _, _ = self.model(batch_x)
                    last_x = batch_x[:, -n_feats:]      # (B, F)
                    last_hat = x_hat[:, -n_feats:]      # (B, F)
                    sq = (last_hat - last_x) ** 2       # (B, F)
                    scores[start:stop] = sq.mean(dim=1).detach().cpu().numpy().astype(np.float32)
            return scores

        # Boundary-safe: run _raw per-entity (no window spans a boundary), then concat.
        # No whole-test post-processing in DAGMM -> the concatenated raw IS the score.
        return per_entity_concat(test_X, test_segments, _raw)

    # ------------------------------------------------------------------
    # Persistence (optional — pipeline-owned best-epoch handling supersedes this)
    # ------------------------------------------------------------------

    def save(self, output_dir: Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.model is not None:
            torch.save(self.model.state_dict(), output_dir / 'model_state.pt')
        config = {
            'name': self.name,
            'impl_ref': 'imperial-qore/TranAD',
            'paper': 'Zong et al., ICLR 2018',
            'n_window': self.n_window,
            'n_hidden': self.n_hidden,
            'n_latent': self.n_latent,
            'lr': self.lr,
            'weight_decay': self.weight_decay,
            'lr_step_size': self.lr_step_size,
            'lr_gamma': self.lr_gamma,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'train_stride': self.train_stride,
            'n_features': self.n_features,
        }
        with open(output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
