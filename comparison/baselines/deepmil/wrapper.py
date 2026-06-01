"""
DeepMIL pipeline wrapper — weak-supervised (Q1-only) baseline, DiCNN encoder.

Wraps the DeepMIL-on-TS model (model.py = WETAS DiCNN encoder + Sultani MIL ranking
head/loss) into the fit/predict pipeline contract. See model.py for the two-part
provenance (encoder = DERIVATIVE_CITED WETAS DiCNN, p.7360; head+loss = FAITHFUL
Sultani CVPR'18, arXiv 1801.04264) and the dense-scoring (C2) rationale.

Normalization (Change 3 — RAW input, wrapper self-normalizes):
- The driver now classifies DeepMIL as SELF_NORMALIZING_WEAK (run_baseline.py:240),
  so `train_X`/`test_X` arrive RAW (normalize_mode='none'). The wrapper applies the
  WETAS-lineage normalization itself:  per-recording `StandardScaler` (zero-mean /
  unit-variance), the DeepMIL-on-TS normalization reference.
  Source: OFFICIAL WETAS `donalee/WETAS@cb149dc` `timeseries.py::_preprocess`:
      from sklearn.preprocessing import StandardScaler
      scaler = StandardScaler(); scaler.fit(data); data = scaler.transform(data)
  i.e. a fresh StandardScaler fit on EACH recording's own data (per-recording /
  transductive), not a single global train-only scaler.
- fit(): StandardScaler fit on train_X, transform train_X (the train recording).
- predict(): predict() receives ONLY test_X (no per-recording segment boundaries), so
  per-recording-per-file scaling is infeasible. Most-faithful feasible choice — match
  WETAS's "fit a fresh scaler on THIS recording's own data" by fitting a transductive
  StandardScaler on test_X itself (zero-mean/unit-variance over the whole test array)
  and transforming it. Residual scope gap vs the upstream reference: upstream fits one
  scaler PER recording/file; here test_X is the concatenation of all test recordings,
  so a single test-wide scaler is used (the boundaries are not visible to predict).
  This is strictly closer to the per-recording reference than applying the TRAIN scaler
  to test (which would be cross-recording). Documented in CODE_REWORK_NOTES.md.

Bag/label rule (FAITHFUL MIL semantics; window length is NON_OFFICIAL / TS-specific):
- A BAG = one window of length ``seq_len`` over the (StandardScaler-normalized) TS.
- The DiCNN encoder scores the window DENSELY (one sigmoid score per timestep); the
  MIL "instances" of the bag are therefore its ``seq_len`` timesteps (C2 — replaces
  the previous 32-segment split). bag_label = max(train_y over the window) ==
  any(anomaly in window), computed on the TRAIN SPLIT ONLY (leak-free).
- The ranking hinge needs BOTH a positive (label 1) and a negative (label 0) bag; if
  either is absent (e.g. Q3 normal-only) fit() raises RuntimeError. (Q1 only.)

Scoring rule (NON_OFFICIAL — dense per-timestep, no official streaming counterpart):
- Slide bag-windows over test_X (stride 1 by default), encode each window with DiCNN +
  MIL head to get one continuous score PER TIMESTEP (no segment broadcast), then
  overlap-aggregate across windows (default: mean) into one score per timestep.
  Head/tail filled, nan_to_num, length == len(test_X).
"""

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from comparison.segment_utils import compute_segment_safe_window_indices
from comparison.baselines.deepmil.model import DeepMILModel, mil_ranking_loss
from comparison.baselines._per_file_norm import (
    fit_transform_train_per_file,
    transform_test_per_file,
)


class DeepMILBaseline:
    """DeepMIL weak-supervised baseline (Q1 only; Q3 N/A) — DiCNN encoder, dense MIL."""

    def __init__(self, **hparams):
        self.name = "DeepMIL"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # --- FAITHFUL (code-sourced) head/loss/optim defaults (Sultani CVPR'18) ---
        self.dropout = hparams.get('dropout', 0.6)               # [C]
        self.ranking_margin = hparams.get('ranking_margin', 1.0)  # [C]
        self.lambda_smooth = hparams.get('lambda_smooth', 8e-5)  # [C]
        self.lambda_sparse = hparams.get('lambda_sparse', 8e-5)  # [C]
        # OPTIMIZER — encoder-swap forced deviation (documented in CODE_REWORK_NOTES.md).
        # Sultani's ORIGINAL optimizer is Adagrad lr=0.01 — but that was tuned for a SHALLOW
        # 3-layer head on FROZEN C3D features. With the now-learnable DEEP WETAS DiCNN encoder
        # (7 dilated layers, jointly trained), Adagrad lr=0.01 DIVERGES: the per-timestep logits
        # blow up to ~ -200/-440 and the sigmoid scores collapse to an identical 0.0 (verified —
        # nunique=1). Since the encoder is wholesale the WETAS DiCNN, we train it with the
        # encoder's OWN paper-sourced optimizer — Adam lr=1e-4 (OFFICIAL WETAS
        # donalee/WETAS@cb149dc train_classifier.py:113,234; mirrored in baselines/wetas) — which
        # is stable (pos_max=1.0 >> neg_max->0, MIL ranking learns; verified 300 iters). The MIL
        # ranking HEAD + LOSS remain Sultani-FAITHFUL; only the optimizer follows the encoder's
        # source to keep the deep encoder trainable. (Was: Adagrad lr=0.01.)
        self.optimizer_name = hparams.get('optimizer', 'adam')   # [WETAS-sourced] encoder stability
        self.lr = hparams.get('lr', 1e-4)                        # [WETAS train_classifier.py:234]
        self.bags_per_batch = hparams.get('bags_per_batch', 60)  # [C] 30 pos + 30 neg
        self.l2_reg = hparams.get('l2_reg', 0.001)               # [C] Keras W_regularizer

        # n_segments: kept ONLY for config back-compat. After the DiCNN encoder swap (C2)
        # scoring is dense per-timestep, so bags are no longer split into 32 segments and
        # this value no longer affects the model/loss. Retained so old config.json loads.
        self.n_segments = hparams.get('n_segments', 32)

        # DiCNN output dim — DERIVATIVE_CITED (sourced encoder parameter), not a free choice.
        self.encoder_dim = hparams.get('encoder_dim', 128)       # [DERIVATIVE_CITED] DiCNN out d=128 (WETAS p.7360)
        # --- NON_OFFICIAL (TS-specific streaming choices, disclosed; no official counterpart — G1) ---
        self.seq_len = hparams.get('seq_len', 128)               # [NON_OFFICIAL] bag window length (== DiCNN RF 2^7)
        self.test_stride = hparams.get('test_stride', 1)         # [NON_OFFICIAL] overlap aggregation stride
        self.aggregation = hparams.get('aggregation', 'mean')    # [NON_OFFICIAL] mean|max over overlaps

        # --- pipeline contract ---
        self.epochs = hparams.get('epochs', 10)                  # [TS] epoch-based pipeline
        self.iters_per_epoch = hparams.get('iters_per_epoch', 50)  # [TS] sampled MIL iters/epoch
        self.batch_size = self.bags_per_batch                    # alias for contract
        self.train_stride = hparams.get('train_stride', 1)
        self.verbose = hparams.get('verbose', True)

        # Per-batch positive/negative split (half/half, official 30+30).
        self.pos_per_batch = self.bags_per_batch // 2
        self.neg_per_batch = self.bags_per_batch - self.pos_per_batch

        self.model: Optional[DeepMILModel] = None
        self.n_features: int = 0
        # WETAS-lineage per-recording StandardScaler, fit on the TRAIN recording (Change 3).
        # `self.scaler` keeps the single-file (or first-file) StandardScaler for save/load +
        # resume back-compat (scaler identity unchanged). `self._scalers` is the per-source-FILE
        # cache of TRAIN-fit StandardScalers (leak-free per-file norm, multi-file datasets);
        # the i-th cached train scaler transforms the i-th test file in predict().
        self.scaler: Optional[StandardScaler] = None
        self._scalers: Optional[list] = None
        self.train_loss_history = []

        # Resume infrastructure (2026-05-31, B option — epoch-level dynamics).
        # Both populated by run_baseline.py via set_resume()/set_checkpoint_dir()
        # before fit() is called. If unset, fit() runs from scratch and never
        # writes a checkpoint (i.e. behaviour for non-resume code paths is
        # unchanged).
        self._resume_checkpoint_path: Optional[Path] = None
        self._checkpoint_dir: Optional[Path] = None
        # NumPy bag-sampling RNG; promoted from a fit()-local Generator so its
        # state can be persisted across resume (without it, resume at ep51
        # would replay ep1's bag samples instead of ep51's — breaking
        # epoch-level learning dynamics).
        self._bag_rng: Optional[np.random.Generator] = None

    # ------------------------------------------------------------------ #
    # bag construction (dense — a bag is a whole window, no segment split)
    # ------------------------------------------------------------------ #
    def _build_bag_windows(self, data: np.ndarray, stride: int,
                           segments=None) -> np.ndarray:
        """Return window starts (np.int64) of length seq_len, segment-boundary safe."""
        n = len(data)
        if n < self.seq_len:
            return np.array([], dtype=np.int64)
        n_windows = (n - self.seq_len) // stride + 1
        starts = np.arange(n_windows, dtype=np.int64) * stride
        if segments is not None:
            valid = compute_segment_safe_window_indices(
                segments, self.seq_len, stride, n_windows,
            )
            starts = starts[valid]
        return starts

    def fit(self, train_X, train_y=None, epoch_callback=None, train_segments=None,
            norm_train_segs=None) -> 'DeepMILBaseline':
        train_X = np.asarray(train_X, dtype=np.float32)
        self.n_features = train_X.shape[1]

        # --- Change 3 (+ per-file leak-free norm): WETAS-lineage StandardScaler.
        # (donalee/WETAS@cb149dc timeseries.py::_preprocess: fresh StandardScaler fit on the
        #  recording's own data.) RAW input now arrives (run_baseline SELF_NORMALIZING_WEAK).
        # `norm_train_segs` = per-source-FILE (lo,hi) TRAIN slices (multi-file datasets only);
        # None/single-seg -> one whole-train StandardScaler == legacy behaviour (bit-identical).
        # FIT on each file's TRAIN slice ONLY (leak-free); cache the per-file train scalers on
        # self so predict() can transform each test file by its paired train scaler.
        train_X, self._scalers, _ = fit_transform_train_per_file(
            train_X, norm_train_segs, StandardScaler)
        train_X = train_X.astype(np.float32)
        # Preserve scaler identity for save/load + resume: keep the FIRST (single-file = only)
        # StandardScaler as self.scaler.
        self.scaler = self._scalers[0]

        if train_y is None:
            raise RuntimeError(
                "DeepMIL requires labeled normal+anomalous bags (Q1 only; Q3 N/A)")
        train_y = np.asarray(train_y).astype(np.int64).reshape(-1)

        starts = self._build_bag_windows(train_X, self.train_stride, train_segments)
        if len(starts) == 0:
            raise RuntimeError(
                "DeepMIL: no valid bag windows (train too short or all boundary-crossing)")

        # bag_label = max(train_y in window) — leak-free, train split only.
        bag_labels = np.array(
            [int(train_y[s:s + self.seq_len].max()) for s in starts], dtype=np.int64)
        pos_idx = starts[bag_labels == 1]
        neg_idx = starts[bag_labels == 0]

        if len(pos_idx) == 0 or len(neg_idx) == 0:
            raise RuntimeError(
                "DeepMIL requires labeled normal+anomalous bags (Q1 only; Q3 N/A)")

        self.model = DeepMILModel(
            n_features=self.n_features,
            seq_len=self.seq_len,
            encoder_dim=self.encoder_dim,
            dropout=self.dropout,
        ).to(self.device)

        if self.verbose:
            print(f"{self.name} training:")
            print(f"  Device: {self.device}")
            print(f"  encoder=WETAS DiCNN (DERIVATIVE_CITED p.7360) hidden=out={self.encoder_dim} "
                  f"k=2 n_layers=7 RF=128")
            print(f"  bag(seq_len)={self.seq_len}  dense per-timestep MIL (instances=timesteps)")
            print(f"  features={self.n_features}  normalization=per-recording StandardScaler (WETAS-lineage)")
            print(f"  positive bags={len(pos_idx):,}  negative bags={len(neg_idx):,}")
            print(f"  optimizer={self.optimizer_name} lr={self.lr} (WETAS-sourced; "
                  f"Adagrad lr=0.01 diverges with deep DiCNN encoder)  "
                  f"margin={self.ranking_margin}  lambda_smooth=sparse={self.lambda_smooth}")

        # Adam lr=1e-4 (WETAS train_classifier.py:234) — the WETAS DiCNN encoder's own optimizer,
        # required for stable joint training of the deep encoder (Sultani's Adagrad lr=0.01 collapses
        # the sigmoid scores to a constant 0). L2 added explicitly via head.l2_weight_penalty.
        if self.optimizer_name.lower() == 'adagrad':
            optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.lr, eps=1e-8)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Bag-sampling RNG: promoted to instance attribute so its bit-generator
        # state can be persisted across resume (B option, see _checkpoint.py).
        # seed=42 unchanged from the original from-scratch implementation.
        self._bag_rng = np.random.default_rng(42)
        self.train_loss_history = []
        self.model.train()

        # Attempt resume from a previously-saved checkpoint. Done AFTER model +
        # optimizer + scaler are built so load_state_dict has somewhere to land.
        start_epoch = self._maybe_resume(optimizer)
        start_time = time.time()

        def _sample_windows(idx_pool, k):
            replace = len(idx_pool) < k
            chosen = self._bag_rng.choice(idx_pool, size=k, replace=replace)
            # Each bag is a whole (seq_len, F) window — dense scoring, no segment split.
            arr = np.stack([train_X[s:s + self.seq_len] for s in chosen])
            return torch.from_numpy(arr.astype(np.float32)).to(self.device)

        for epoch in range(start_epoch, self.epochs):
            epoch_losses = []
            for it in range(self.iters_per_epoch):
                # GUARANTEE >=1 positive and >=1 negative bag per batch (resample w/ replacement
                # for low-anomaly train splits).
                pos_windows = _sample_windows(pos_idx, self.pos_per_batch)
                neg_windows = _sample_windows(neg_idx, self.neg_per_batch)

                pos_scores = self.model(pos_windows)   # (P, L)  dense per-timestep
                neg_scores = self.model(neg_windows)   # (Nn, L) dense per-timestep

                loss, _ = mil_ranking_loss(
                    pos_scores, neg_scores,
                    margin=self.ranking_margin,
                    lambda_smooth=self.lambda_smooth,
                    lambda_sparse=self.lambda_sparse,
                )
                loss = loss + self.l2_reg * self.model.head.l2_weight_penalty()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))

            avg = float(np.mean(epoch_losses)) if epoch_losses else float('nan')
            self.train_loss_history.append(avg)
            if self.verbose:
                print(f"  [DeepMIL] epoch={epoch + 1}/{self.epochs} "
                      f"loss={avg:.6f} elapsed={time.time() - start_time:.1f}s", flush=True)

            # Save last.pt BEFORE epoch_callback so a crash in eval (the most
            # likely failure point: long PA%K sweep, slow VUS, OOM on test_X)
            # still leaves a resumable checkpoint for THIS finished epoch.
            self._save_last_checkpoint(epoch + 1, optimizer)

            if epoch_callback is not None:
                epoch_callback(self, epoch + 1)
                self.model.train()

        if self.verbose:
            print(f"  Training complete: {time.time() - start_time:.1f}s")
        return self

    def predict(self, test_X, test_segments=None) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        test_X = np.asarray(test_X, dtype=np.float32)

        # --- Per-file LEAK-FREE test normalization (replaces the previous transductive
        # fit-on-test). `test_segments` = per-source-FILE (lo,hi) TEST slices (multi-file
        # datasets only). The i-th test file is z-scored by the i-th cached TRAIN scaler via
        # .transform ONLY (never fit on test). Single-file / None -> the single cached train
        # scaler transforms the whole test array (leak-free; matches the other self-norm models'
        # regime — no test-statistic leakage).
        test_X = transform_test_per_file(test_X, test_segments, self._scalers).astype(np.float32)

        n = len(test_X)
        self.model.eval()

        sum_scores = np.zeros(n, dtype=np.float64)
        cnt_scores = np.zeros(n, dtype=np.float64)
        max_scores = np.full(n, -np.inf, dtype=np.float64)

        starts = self._build_bag_windows(test_X, self.test_stride, segments=None)

        with torch.no_grad():
            for s in starts:
                window = test_X[s:s + self.seq_len]                    # (L, F)
                bag = torch.from_numpy(window[None].astype(np.float32)).to(self.device)
                # Dense per-timestep score for the window (C2) — no segment broadcast.
                point = self.model(bag).squeeze(0).cpu().numpy().astype(np.float64)  # (L,)

                e = min(s + self.seq_len, n)
                span = e - s
                sum_scores[s:e] += point[:span]
                cnt_scores[s:e] += 1.0
                np.maximum(max_scores[s:e], point[:span], out=max_scores[s:e])

        if self.aggregation == 'max':
            scores = np.where(cnt_scores > 0, max_scores, np.nan)
        else:  # mean overlap aggregation (default, documented)
            scores = np.where(cnt_scores > 0, sum_scores / np.maximum(cnt_scores, 1.0), np.nan)

        # Head/tail fill: timesteps never covered by any window (only possible when
        # n < seq_len) get the nearest covered value via forward/back fill.
        scores = self._fill_uncovered(scores)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        return scores.astype(np.float32)

    @staticmethod
    def _fill_uncovered(scores: np.ndarray) -> np.ndarray:
        if np.all(np.isnan(scores)):
            return np.zeros_like(scores)
        # forward fill then back fill
        idx = np.where(~np.isnan(scores))[0]
        first, last = idx[0], idx[-1]
        scores[:first] = scores[first]
        scores[last + 1:] = scores[last]
        # interior NaNs (rare): forward fill
        for i in range(first + 1, last + 1):
            if np.isnan(scores[i]):
                scores[i] = scores[i - 1]
        return scores

    # ------------------------------------------------------------------ #
    # Resume infrastructure (2026-05-31, B option — see _checkpoint.py)
    # ------------------------------------------------------------------ #
    def set_resume(self, ckpt_path) -> None:
        """Called by run_baseline.py BEFORE fit() to enable resume.

        If set and the file exists, fit() will load model/optimizer/scaler/RNG
        state from it and continue from the saved epoch number. If unset or
        the file is missing, fit() runs from scratch — there is no implicit
        resume from a stale last.pt.
        """
        self._resume_checkpoint_path = Path(ckpt_path) if ckpt_path else None

    def set_checkpoint_dir(self, ckpt_dir) -> None:
        """Called by run_baseline.py BEFORE fit() to enable per-epoch saving.

        If unset, fit() does NOT write last.pt (preserves byte-identical
        behaviour for any caller that doesn't opt into the resume path).
        """
        self._checkpoint_dir = Path(ckpt_dir) if ckpt_dir else None

    def _maybe_resume(self, optimizer) -> int:
        """Restore model/optimizer/scaler/RNG from self._resume_checkpoint_path.

        Returns the next epoch index (0 = from scratch, N = continue at ep N+1).
        Refuses to resume if the hparam fingerprint disagrees (e.g. someone
        changed seq_len between runs) — silent partial restore would corrupt
        training more subtly than a hard error.
        """
        if self._resume_checkpoint_path is None or not self._resume_checkpoint_path.exists():
            return 0
        from comparison.baselines._checkpoint import (
            load_checkpoint, restore_rng_state, dict_to_standard_scaler,
        )
        ckpt = load_checkpoint(self._resume_checkpoint_path)
        # Sanity: target epochs may legitimately differ (user can request
        # extra epochs on resume), so we only enforce architecture-level
        # hparams here.
        ck_hp = ckpt.get('wrapper_hparams', {})
        # Architecture + training-dynamics knobs: any mismatch silently
        # changes results, so reject. dropout/bags_per_batch/iters_per_epoch
        # added per code-review 2026-05-31 (Bug 4 — dynamics-only knobs that
        # do not affect tensor shape are still silently divergent on resume).
        for key in ('seq_len', 'encoder_dim', 'n_features', 'dropout',
                    'bags_per_batch', 'iters_per_epoch'):
            if ck_hp.get(key) != getattr(self, key):
                raise RuntimeError(
                    f"DeepMIL resume hparam mismatch on '{key}': "
                    f"checkpoint={ck_hp.get(key)} vs current={getattr(self, key)}. "
                    f"Re-run from scratch (--force) or load the matching checkpoint."
                )
        self.model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.train_loss_history = list(ckpt.get('train_loss_history', []))
        # Scaler: typically already fit on train_X earlier in fit() to the
        # SAME values (deterministic on inputs), but restoring from checkpoint
        # is cheaper and guards against any future drift in StandardScaler.
        restored_scaler = dict_to_standard_scaler(ckpt.get('scaler_state'))
        if restored_scaler is not None:
            self.scaler = restored_scaler
            # Keep the per-file scaler cache consistent on resume. The checkpoint
            # persists a SINGLE scaler_state (the single-file / first-file scaler),
            # so only the single-file cache can be safely synced from it; multi-file
            # per-file scalers were just re-fit (deterministically) above in fit().
            if self._scalers is not None and len(self._scalers) == 1:
                self._scalers = [restored_scaler]
        # Wrapper-local NumPy Generator (bag sampling) — torch-global capture
        # does not see it, so we persisted bit_generator.state separately.
        wrapper_state = ckpt.get('rng_wrapper_specific', {}) or {}
        if wrapper_state.get('bag_rng_state') is not None:
            self._bag_rng.bit_generator.state = wrapper_state['bag_rng_state']
        restore_rng_state(ckpt.get('rng_state'))
        start_epoch = int(ckpt['epoch'])
        if self.verbose:
            print(f"  [RESUME] DeepMIL from epoch {start_epoch}/{self.epochs} "
                  f"(checkpoint: {self._resume_checkpoint_path.name})", flush=True)
        return start_epoch

    def _save_last_checkpoint(self, current_epoch: int, optimizer) -> None:
        """Atomic write of checkpoints/last.pt after the current epoch.

        No-op if set_checkpoint_dir() was not called — keeps non-resume code
        paths byte-identical to the pre-2026-05-31 behaviour.
        """
        if self._checkpoint_dir is None:
            return
        from comparison.baselines._checkpoint import (
            atomic_save, capture_rng_state, standard_scaler_to_dict,
            CHECKPOINT_VERSION,
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
                'seq_len': self.seq_len, 'encoder_dim': self.encoder_dim,
                'dropout': self.dropout, 'n_features': self.n_features,
                'lr': self.lr, 'optimizer': self.optimizer_name,
                'ranking_margin': self.ranking_margin,
                'lambda_smooth': self.lambda_smooth,
                'lambda_sparse': self.lambda_sparse,
                'bags_per_batch': self.bags_per_batch,
                'iters_per_epoch': self.iters_per_epoch,
            },
            'scaler_state': standard_scaler_to_dict(self.scaler),
            'rng_state': capture_rng_state(),
            'rng_wrapper_specific': {
                'bag_rng_state': (
                    self._bag_rng.bit_generator.state if self._bag_rng is not None else None
                ),
            },
        }
        atomic_save(ckpt, self._checkpoint_dir / 'last.pt')

    # ------------------------------------------------------------------ #
    def save(self, save_dir) -> None:
        if self.model is None:
            raise RuntimeError("No model to save.")
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), save_dir / "model.pt")
        # Persist the WETAS-lineage train StandardScaler (mean/scale) for reproducibility.
        scaler_blob = None
        if self.scaler is not None and hasattr(self.scaler, 'mean_'):
            scaler_blob = {
                "mean": self.scaler.mean_.tolist(),
                "scale": self.scaler.scale_.tolist(),
                "var": self.scaler.var_.tolist(),
                "n_samples_seen": int(self.scaler.n_samples_seen_),
            }
        config = {
            "n_segments": self.n_segments, "dropout": self.dropout,
            "ranking_margin": self.ranking_margin,
            "lambda_smooth": self.lambda_smooth, "lambda_sparse": self.lambda_sparse,
            "lr": self.lr, "optimizer": self.optimizer_name,
            "bags_per_batch": self.bags_per_batch,
            "seq_len": self.seq_len, "encoder_dim": self.encoder_dim,
            "test_stride": self.test_stride, "aggregation": self.aggregation,
            "n_features": self.n_features,
            "train_scaler": scaler_blob,
            "model_name": self.name,
        }
        with open(save_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

    def load(self, save_dir) -> 'DeepMILBaseline':
        save_dir = Path(save_dir)
        with open(save_dir / "config.json", 'r') as f:
            config = json.load(f)
        self.n_segments = config.get("n_segments", 32)
        self.dropout = config["dropout"]
        self.seq_len = config["seq_len"]
        self.encoder_dim = config["encoder_dim"]
        self.test_stride = config.get("test_stride", 1)
        self.aggregation = config.get("aggregation", "mean")
        self.n_features = config["n_features"]
        # Restore the WETAS-lineage train StandardScaler if present.
        scaler_blob = config.get("train_scaler")
        if scaler_blob is not None:
            self.scaler = StandardScaler()
            self.scaler.mean_ = np.asarray(scaler_blob["mean"], dtype=np.float64)
            self.scaler.scale_ = np.asarray(scaler_blob["scale"], dtype=np.float64)
            self.scaler.var_ = np.asarray(scaler_blob["var"], dtype=np.float64)
            self.scaler.n_samples_seen_ = scaler_blob["n_samples_seen"]
            self.scaler.n_features_in_ = len(scaler_blob["mean"])
        self.model = DeepMILModel(
            n_features=self.n_features, seq_len=self.seq_len,
            encoder_dim=self.encoder_dim, dropout=self.dropout,
        ).to(self.device)
        self.model.load_state_dict(torch.load(save_dir / "model.pt", map_location=self.device))
        self.model.eval()
        return self

    def __repr__(self) -> str:
        return (f"DeepMILBaseline(seq_len={self.seq_len}, encoder=DiCNN(RF=128), "
                f"encoder_dim={self.encoder_dim}, lr={self.lr})")
