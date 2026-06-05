"""
CATCH Baseline Wrapper

Channel-aware reconstruction with frequency-domain head (ICLR 2025).
Vendored from https://github.com/decisionintelligence/CATCH (TAB framework).

Training objective (matches upstream CATCH.detect_fit):
    L = MSE(recon, x) + dc_lambda * dcloss + auxi_lambda * freq_loss(complex_z, x_norm)

Inference (matches upstream CATCH.detect_score):
    score = mean_chan(MSE(recon, x)) + score_lambda * mean_chan(freq_criterion(recon, x))
Aggregated to point-level via upstream `mode='thre'` non-overlap windowing
(stride=win_size) + `np.concatenate(scores, axis=0).reshape(-1)` flatten, as
verified 2026-05-24 against `ts_benchmark/baselines/catch/CATCH.py::detect_score`
and `ts_benchmark/baselines/utils.py::SegLoader` (mode='thre'). The reference
silently drops the `T_test mod win_size` tail; this wrapper applies Option B
fallback (repeat-last per-timestep score for the tail) to satisfy the
`./comparison/` pipeline length contract `scores.shape == (T_test,)`.

Interface: comparison.baseline_common.run_sota_baseline_with_epoch_eval()
"""

import json
import os
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Subset

# Batch-progress logging: stdout goes through conda-run buffer (invisible until
# subprocess exits). Mirror every [BATCH] line to a file so live monitoring
# (tail -f) sees per-batch heartbeats. Path overridable via env var.
_BATCH_LOG_PATH = os.environ.get(
    'CATCH_BATCH_LOG',
    '/home/ykio/notebooks/claude/temp/baseline_experiment_run/catch_batch_progress.log',
)

from comparison.segment_utils import compute_segment_safe_window_indices
from comparison.baselines._boundary_safe_window import per_entity_concat
from tqdm import tqdm

from .model import (
    CATCHModel,
    TransformerConfig,
    frequency_loss,
    frequency_criterion,
)


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
        return torch.from_numpy(self.data[start: start + self.win_size].copy()).float()


def _adjust_learning_rate_type1(optimizer: torch.optim.Optimizer,
                                epoch_one_indexed: int,
                                lr_base: float) -> None:
    """Upstream CATCH adjust_learning_rate('type1') exact replica.

    Mirrors `ts_benchmark/baselines/catch/utils/tools.py:adjust_learning_rate` for
    lradj='type1' (upstream default, verified 2026-05-24 via WebFetch on master branch):

        lr = lr_base * (0.5 ** ((epoch - 1) // 1))

    where `epoch` is 1-indexed. Epoch 1 keeps `lr_base`; each subsequent epoch halves it.
    """
    lr = lr_base * (0.5 ** ((epoch_one_indexed - 1) // 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class CATCHBaseline:
    """CATCH (ICLR 2025) wrapper for sliding-window MTS anomaly detection.

    Architecture and losses are bit-identical to upstream
    `ts_benchmark/baselines/catch/` (vendored verbatim into `model.py`).
    Hyperparameters from `MODEL_PRESETS['default']['catch']`.

    Training schedule follows upstream `CATCH.detect_fit` exactly:
      * Two Adam optimizers (main + mask_generator with `Mlr = lr × mlr_ratio`).
      * Per-epoch learning-rate halving via `_adjust_learning_rate_type1` (upstream
        `adjust_learning_rate('type1')`). OneCycleLR object in upstream is dead code
        (never `.step()`-ed); we mirror upstream by not constructing it.
      * `optimizerM.step()` every `min(len(loader)/10, 100)` minibatches.
    """

    def __init__(
        self,
        # Paper defaults — upstream DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS
        win_size: int = 192,
        patch_size: int = 16,
        patch_stride: int = 8,
        d_model: int = 128,
        n_heads: int = 2,                       # paper default (was 8 in our prev preset)
        e_layers: int = 3,                      # paper default (was 2 in our prev preset)
        lambda_ch_discover: float = 0.005,      # paper dc_lambda (was 0.5 in our prev preset)
        lambda_freq: float = 0.005,             # paper auxi_lambda (was 0.5 in our prev preset)
        dropout: float = 0.2,                   # paper default (was 0.1)
        lr: float = 1e-4,
        train_stride: int = 1,
        epochs: int = 10,
        batch_size: int = 128,
        # ---- additional upstream knobs (default to upstream defaults) ----
        head_dim: int = 64,
        cf_dim: int = 64,
        d_ff: int = 256,
        individual: int = 0,
        head_dropout: float = 0.1,
        auxi_loss: str = "MAE",
        auxi_type: str = "complex",
        auxi_mode: str = "fft",
        score_lambda: float = 0.05,
        regular_lambda: float = 0.5,
        temperature: float = 0.07,
        inference_patch_size: int = 32,
        inference_patch_stride: int = 1,
        module_first: bool = True,
        mask: bool = False,
        use_revin: bool = True,
        affine: int = 0,
        subtract_last: int = 0,
        # Paper-faithful training (upstream uses type1 per-epoch LR halving)
        mlr_ratio: float = 0.1,                 # Mlr = lr × mlr_ratio (paper Mlr=1e-5, lr=1e-4)
        pct_start: float = 0.3,                 # preset compatibility; unused (upstream OneCycleLR is dead code)
        # ---- runtime --------------------------------------------------
        device: Optional[str] = None,
        verbose: bool = True,
    ):
        # Preset-facing hyperparameters
        self.win_size = win_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.lambda_freq = lambda_freq
        self.lambda_ch_discover = lambda_ch_discover
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_stride = train_stride

        # Upstream-only knobs
        self.head_dim = head_dim
        self.cf_dim = cf_dim
        self.d_ff = d_ff
        self.individual = individual
        self.head_dropout = head_dropout
        self.auxi_loss = auxi_loss
        self.auxi_type = auxi_type
        self.auxi_mode = auxi_mode
        self.score_lambda = score_lambda
        self.regular_lambda = regular_lambda
        self.temperature = temperature
        # inference_patch_size must be ≤ win_size; clamp defensively if win_size small
        self.inference_patch_size = min(inference_patch_size, win_size)
        self.inference_patch_stride = inference_patch_stride
        self.module_first = module_first
        self.mask = mask
        self.use_revin = use_revin
        self.affine = affine
        self.subtract_last = subtract_last
        self.mlr_ratio = mlr_ratio
        self.pct_start = pct_start

        self.verbose = verbose
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

        self.model: Optional[CATCHModel] = None
        self.n_features: Optional[int] = None
        self.train_loss_history: list = []
        # Paper-faithful normalization: upstream CATCH uses StandardScaler.fit(train_data).
        # Driver passes raw data (normalize_mode='none' via run_baseline.py override).
        self.scaler: Optional[StandardScaler] = None

    @property
    def name(self) -> str:
        return "CATCH"

    # --------------------------------------------------------------
    # Model construction
    # --------------------------------------------------------------

    def _make_config(self) -> TransformerConfig:
        return TransformerConfig(
            seq_len=self.win_size,
            patch_size=self.patch_size,
            patch_stride=self.patch_stride,
            d_model=self.d_model,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            cf_dim=self.cf_dim,
            d_ff=self.d_ff,
            head_dim=self.head_dim,
            individual=self.individual,
            dropout=self.dropout,
            head_dropout=self.head_dropout,
            auxi_loss=self.auxi_loss,
            auxi_type=self.auxi_type,
            auxi_mode=self.auxi_mode,
            auxi_lambda=self.lambda_freq,
            dc_lambda=self.lambda_ch_discover,
            score_lambda=self.score_lambda,
            regular_lambda=self.regular_lambda,
            temperature=self.temperature,
            inference_patch_size=self.inference_patch_size,
            inference_patch_stride=self.inference_patch_stride,
            module_first=self.module_first,
            mask=self.mask,
            affine=self.affine,
            subtract_last=self.subtract_last,
            lr=self.lr,
            batch_size=self.batch_size,
            num_epochs=self.epochs,
            c_in=self.n_features,
        )

    def _build_model(self) -> CATCHModel:
        if self.n_features is None:
            raise RuntimeError("n_features not set — call fit() first or set self.n_features manually.")
        config = self._make_config()
        model = CATCHModel(config).to(self.device)
        # Cache config for losses
        self._config = config
        return model

    # --------------------------------------------------------------
    # Training (mirrors upstream CATCH.detect_fit core loop)
    # --------------------------------------------------------------

    def fit(self, train_X: np.ndarray, epoch_callback=None, train_segments=None) -> "CATCHBaseline":
        # --- Paper-faithful normalization (upstream CATCH StandardScaler equivalent) ---
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
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        # Upstream uses two optimizers: main + mask_generator. Mirror that.
        main_params = [p for n, p in self.model.named_parameters() if 'mask_generator' not in n]
        mask_params = [p for n, p in self.model.named_parameters() if 'mask_generator' in n]
        optimizer = torch.optim.Adam(main_params, lr=self.lr)
        mask_lr_base = self.lr * self.mlr_ratio
        optimizerM = torch.optim.Adam(mask_params, lr=mask_lr_base)  # upstream Mlr = lr × mlr_ratio

        # Learning-rate schedule — upstream CATCH.detect_fit constructs OneCycleLR but
        # NEVER calls scheduler.step(); instead it calls adjust_learning_rate('type1')
        # at the end of each epoch (verified 2026-05-24 via WebFetch on master branch).
        # We mirror upstream's effective behavior: per-epoch geometric halving of lr for
        # both optimizers (Mlr halves at the same rate so Mlr/lr ratio stays constant).

        criterion = nn.MSELoss()
        auxi_loss_fn = frequency_loss(self._config)

        self.model.train()
        self.train_loss_history = []
        for epoch in range(self.epochs):
            epoch_loss_sum = 0.0
            n_batches = 0
            iterator = (
                tqdm(loader, desc=f"Epoch {epoch + 1}/{self.epochs}", leave=False)
                if self.verbose else loader
            )
            # upstream: optimizerM steps every `step` minibatches (min(N/10,100))
            step_for_M = max(1, min(int(len(loader) / 10), 100))

            n_batches_total = len(loader)
            _epoch_start = time.time()
            _ep_msg = (f"[EPOCH_BEGIN] model={self.name} epoch={epoch + 1}/{self.epochs} "
                       f"n_batches={n_batches_total} batch_size={self.batch_size}")
            print(_ep_msg, flush=True)
            try:
                with open(_BATCH_LOG_PATH, 'a') as _bl:
                    _bl.write(_ep_msg + '\n')
            except Exception:
                pass
            for i, batch in enumerate(iterator):
                batch_start = time.time()
                input_x = batch.float().to(self.device)
                optimizer.zero_grad()

                output, output_complex, dcloss = self.model(input_x)

                rec_loss = criterion(output, input_x)
                norm_input = self.model.revin_layer(input_x, 'transform')
                auxi_loss_val = auxi_loss_fn(output_complex, norm_input)

                loss = rec_loss + self._config.dc_lambda * dcloss + self._config.auxi_lambda * auxi_loss_val
                loss.backward()
                optimizer.step()

                # Mask-generator update (every `step_for_M` minibatches, mirrors upstream)
                if (i + 1) % step_for_M == 0:
                    optimizerM.step()
                    optimizerM.zero_grad()

                epoch_loss_sum += loss.item()
                n_batches += 1
                _msg = (f"[BATCH] model={self.name} epoch={epoch + 1}/{self.epochs} "
                        f"batch={i + 1}/{n_batches_total} loss={loss.item():.4f} "
                        f"time={time.time() - batch_start:.3f}s")
                print(_msg, flush=True)
                try:
                    with open(_BATCH_LOG_PATH, 'a') as _bl:
                        _bl.write(_msg + '\n')
                except Exception:
                    pass

            # End-of-epoch LR halving (upstream adjust_learning_rate('type1')).
            # Upstream call: adjust_learning_rate(optimizer, scheduler, epoch + 1, config)
            # where `epoch` is the 0-indexed Python loop variable. Same applied to optimizerM.
            _adjust_learning_rate_type1(optimizer, epoch + 1, self.lr)
            _adjust_learning_rate_type1(optimizerM, epoch + 1, mask_lr_base)

            avg_loss = epoch_loss_sum / max(n_batches, 1)
            self.train_loss_history.append(avg_loss)
            _ep_end_msg = (f"[EPOCH_END] model={self.name} epoch={epoch + 1}/{self.epochs} "
                           f"avg_loss={avg_loss:.6f} elapsed={time.time() - _epoch_start:.1f}s")
            print(_ep_end_msg, flush=True)
            try:
                with open(_BATCH_LOG_PATH, 'a') as _bl:
                    _bl.write(_ep_end_msg + '\n')
            except Exception:
                pass
            if self.verbose:
                print(f"  Epoch {epoch + 1}: loss = {avg_loss:.6f}")
            if epoch_callback is not None:
                epoch_callback(self, epoch + 1)
                self.model.train()
        return self

    # --------------------------------------------------------------
    # Inference (mirrors upstream CATCH.detect_score)
    # --------------------------------------------------------------

    def predict(self, test_X: np.ndarray, test_segments=None) -> np.ndarray:
        """Reference-faithful CATCH inference (upstream `detect_score`, mode='thre').

        Windowing: non-overlap stride=`win_size` (upstream `SegLoader(mode='thre')`
        uses `index // step * win_size` with `step=1`, effective stride=`win_size`,
        `__len__ = (T - W) // W + 1`).

        Aggregation: per-window [B, W] scores → `np.concatenate(...).reshape(-1)`
        flatten (upstream `attens_energy = np.concatenate(..., axis=0).reshape(-1)`).
        Effective valid length: `((T_test - W) // W + 1) * W`.

        Tail handling: reference IMPLICIT_NONE (silently drops `T_test mod W` tail).
        We apply Option B fallback — repeat-last per-timestep score — to satisfy
        the `./comparison/` pipeline contract `scores.shape == (T_test,)`.

        Boundary-safe TEST windowing (2026-06-02): on multi-entity datasets (SMD
        machines / SMAP-MSL channels / Exathlon apps) the non-overlap windowing +
        inference + per-window→per-timestep mapping (incl. Option B tail fill) is run
        INDEPENDENTLY on each entity's own test slice via `per_entity_concat`, so no
        non-overlap window ever spans an entity boundary. CATCH has NO whole-test score
        post-processing in upstream `detect_score` (verified 2026-06-02: it returns the
        concatenated/reshaped window scores directly — no median/IQR/smoothing/
        normalization), so the per-entity raw producer IS the full score path and there
        is nothing to re-apply globally afterwards.

        Normalization (StandardScaler, fit on TRAIN only) is applied to the whole test
        array BEFORE per-entity windowing. The scaler is a pointwise affine transform
        fit on train, so slicing the already-normalized test is bit-identical to
        slicing-then-transforming — per-entity windowing does NOT change normalization
        (the "untouchable-norm" guarantee holds). Single-entity / None / non-tiling
        `test_segments` -> ONE call over the whole array == exact legacy behaviour.

        Re-verified 2026-05-24 / 2026-06-02 via WebFetch:
          - https://raw.githubusercontent.com/decisionintelligence/CATCH/master/ts_benchmark/baselines/catch/CATCH.py
          - https://raw.githubusercontent.com/decisionintelligence/CATCH/master/ts_benchmark/baselines/utils.py
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        if self.scaler is None:
            raise RuntimeError("Scaler not fit. Call fit() first.")

        # Apply paper-faithful StandardScaler.transform on test data (train-only fit).
        # Done on the WHOLE test array BEFORE per-entity windowing: the scaler is
        # train-fit and pointwise, so this is slicing-invariant (normalization UNCHANGED).
        test_X = self.scaler.transform(test_X).astype(np.float32)

        N_test, n_features = test_X.shape

        self.model.eval()
        temp_criterion = nn.MSELoss(reduction='none')
        freq_criterion_fn = frequency_criterion(self._config)
        score_lambda = self._config.score_lambda

        def _raw_score(sub_X: np.ndarray) -> np.ndarray:
            """RAW per-timestep CATCH scores for a SINGLE entity's test slice.

            Exactly the legacy non-overlap windowing + model inference + upstream
            concat/reshape + Option B repeat-last tail fill, but confined to `sub_X`
            (so no window crosses an entity boundary). Returns shape `(len(sub_X),)`.
            """
            sub_X = np.asarray(sub_X, dtype=np.float32)
            n_sub = sub_X.shape[0]
            # Edge case: an entity shorter than win_size has no non-overlap window.
            # Do NOT crash (multi-entity datasets may contain a tiny entity); fall back
            # to a neutral constant score for that entity. Documented as a sensible
            # fallback for the degenerate short-slice case (upstream would error here).
            if n_sub < self.win_size:
                return np.zeros(n_sub, dtype=np.float32)

            # Upstream SegLoader(mode='thre') effective behavior: non-overlap stride=win_size.
            n_windows = (n_sub - self.win_size) // self.win_size + 1

            per_batch_scores: list = []  # list of [B, W] numpy arrays (upstream attens_energy)
            n_batches = (n_windows + self.batch_size - 1) // self.batch_size
            with torch.no_grad():
                for batch_idx in range(n_batches):
                    batch_start = batch_idx * self.batch_size
                    batch_end = min(batch_start + self.batch_size, n_windows)
                    actual_bs = batch_end - batch_start

                    batch_windows = np.zeros((actual_bs, self.win_size, n_features), dtype=np.float32)
                    for j, w_idx in enumerate(range(batch_start, batch_end)):
                        start = w_idx * self.win_size  # non-overlap stride
                        batch_windows[j] = sub_X[start: start + self.win_size]

                    input_x = torch.from_numpy(batch_windows).to(self.device)
                    outputs, _, _ = self.model(input_x)

                    # Per upstream detect_score: temp_score [B,L], freq_score [B,L]
                    temp_score = torch.mean(temp_criterion(input_x, outputs), dim=-1)  # [B, L]
                    freq_score = torch.mean(freq_criterion_fn(input_x, outputs), dim=-1)  # [B, L]
                    window_scores = (temp_score + score_lambda * freq_score).cpu().numpy()  # [B, L]
                    per_batch_scores.append(window_scores.astype(np.float32))

                    if self.verbose and (batch_idx + 1) % max(1, n_batches // 10) == 0:
                        progress = (batch_idx + 1) / n_batches * 100
                        print(f"\r  Inference: [{batch_idx + 1}/{n_batches}] {progress:5.1f}%", end="")

            if self.verbose:
                print()

            # Upstream aggregation: np.concatenate(attens_energy, axis=0).reshape(-1)
            valid_scores = np.concatenate(per_batch_scores, axis=0).reshape(-1).astype(np.float32)
            valid_len = valid_scores.shape[0]  # == n_windows * win_size

            # Option B fallback: scores aligned to start; pad tail by repeating last score.
            sub_scores = np.empty(n_sub, dtype=np.float32)
            sub_scores[:valid_len] = valid_scores
            if valid_len < n_sub:
                sub_scores[valid_len:] = valid_scores[-1]  # Option B repeat-last
            return sub_scores

        # Route windowing + inference + per-window→per-timestep mapping (incl. tail fill)
        # through per_entity_concat so no non-overlap window spans an entity boundary.
        # No whole-test post-processing follows (upstream detect_score returns raw scores).
        point_scores = per_entity_concat(test_X, test_segments, _raw_score)
        return point_scores.astype(np.float32)

    # --------------------------------------------------------------
    # Save / load
    # --------------------------------------------------------------

    def save(self, save_dir: Path) -> None:
        if self.model is None:
            raise RuntimeError("No model to save.")
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), save_dir / "model.pt")
        if self.scaler is not None:
            joblib.dump(self.scaler, save_dir / "scaler.pkl")
        config = {
            "model_name": "CATCH",
            "win_size": self.win_size,
            "patch_size": self.patch_size,
            "patch_stride": self.patch_stride,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "e_layers": self.e_layers,
            "lambda_freq": self.lambda_freq,
            "lambda_ch_discover": self.lambda_ch_discover,
            "dropout": self.dropout,
            "head_dim": self.head_dim,
            "cf_dim": self.cf_dim,
            "d_ff": self.d_ff,
            "individual": self.individual,
            "head_dropout": self.head_dropout,
            "auxi_loss": self.auxi_loss,
            "auxi_type": self.auxi_type,
            "auxi_mode": self.auxi_mode,
            "score_lambda": self.score_lambda,
            "regular_lambda": self.regular_lambda,
            "temperature": self.temperature,
            "inference_patch_size": self.inference_patch_size,
            "inference_patch_stride": self.inference_patch_stride,
            "module_first": self.module_first,
            "mask": self.mask,
            "affine": self.affine,
            "subtract_last": self.subtract_last,
            "n_features": self.n_features,
        }
        with open(save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    def load(self, save_dir: Path) -> "CATCHBaseline":
        save_dir = Path(save_dir)
        with open(save_dir / "config.json", "r") as f:
            config = json.load(f)
        for k, v in config.items():
            if k == "model_name":
                continue
            setattr(self, k, v)
        self.model = self._build_model()
        self.model.load_state_dict(torch.load(save_dir / "model.pt", map_location=self.device))
        self.model.eval()
        scaler_path = save_dir / "scaler.pkl"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
        return self
