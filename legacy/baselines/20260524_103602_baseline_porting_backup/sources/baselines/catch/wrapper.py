"""
CATCH Baseline Wrapper

Channel-aware reconstruction with frequency-domain head (ICLR 2025).
Vendored from https://github.com/decisionintelligence/CATCH (TAB framework).

Training objective (matches upstream CATCH.detect_fit):
    L = MSE(recon, x) + dc_lambda * dcloss + auxi_lambda * freq_loss(complex_z, x_norm)

Inference (matches upstream CATCH.detect_score):
    score = mean_chan(MSE(recon, x)) + score_lambda * mean_chan(freq_criterion(recon, x))
Aggregated to point-level via max over overlapping windows.

Interface: comparison.baseline_common.run_sota_baseline_with_epoch_eval()
"""

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
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


class CATCHBaseline:
    """CATCH (ICLR 2025) wrapper for sliding-window MTS anomaly detection.

    Architecture and losses are bit-identical to upstream
    `ts_benchmark/baselines/catch/` (vendored verbatim into `model.py`).
    Hyperparameters from `MODEL_PRESETS['default']['catch']`.

    All preset keys are accepted by `__init__`. Keys that have no analog in the
    upstream model (`ch_mask_ratio` — upstream generator is parameter-less and
    uses Gumbel-Sigmoid resampling instead of a fixed ratio) are stored but not
    forwarded, with a documented note.
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
        ch_mask_ratio: float = 0.3,             # not used by upstream model; accepted for preset compatibility
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
        # OneCycleLR + paper-faithful training
        mlr_ratio: float = 0.1,                 # Mlr = lr × mlr_ratio (paper Mlr=1e-5, lr=1e-4)
        pct_start: float = 0.3,                 # OneCycleLR pct_start
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
        self.ch_mask_ratio = ch_mask_ratio  # stored but unused (see class docstring)
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

    def fit(self, train_X: np.ndarray, epoch_callback=None) -> "CATCHBaseline":
        # --- Paper-faithful normalization (upstream CATCH StandardScaler equivalent) ---
        self.scaler = StandardScaler()
        self.scaler.fit(train_X)
        train_X = self.scaler.transform(train_X).astype(np.float32)

        self.n_features = train_X.shape[1]
        self.model = self._build_model()

        dataset = _SlidingWindowDataset(train_X, self.win_size, stride=self.train_stride)
        if self.verbose:
            print(f"  Created {len(dataset)} training windows (stride={self.train_stride})")
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        # Upstream uses two optimizers: main + mask_generator. Mirror that.
        main_params = [p for n, p in self.model.named_parameters() if 'mask_generator' not in n]
        mask_params = [p for n, p in self.model.named_parameters() if 'mask_generator' in n]
        optimizer = torch.optim.Adam(main_params, lr=self.lr)
        optimizerM = torch.optim.Adam(mask_params, lr=self.lr * self.mlr_ratio)  # upstream Mlr = lr × mlr_ratio

        # OneCycleLR scheduler — upstream CATCH.detect_fit uses lr_scheduler.OneCycleLR(max_lr, pct_start)
        train_steps = len(loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=self.lr,
            steps_per_epoch=train_steps,
            epochs=self.epochs,
            pct_start=self.pct_start,
        )

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

            for i, batch in enumerate(iterator):
                input_x = batch.float().to(self.device)
                optimizer.zero_grad()

                output, output_complex, dcloss = self.model(input_x)

                rec_loss = criterion(output, input_x)
                norm_input = self.model.revin_layer(input_x, 'transform')
                auxi_loss_val = auxi_loss_fn(output_complex, norm_input)

                loss = rec_loss + self._config.dc_lambda * dcloss + self._config.auxi_lambda * auxi_loss_val
                loss.backward()
                optimizer.step()
                scheduler.step()  # OneCycleLR per batch

                # Mask-generator update (every `step_for_M` minibatches, mirrors upstream)
                if (i + 1) % step_for_M == 0:
                    optimizerM.step()
                    optimizerM.zero_grad()

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

    # --------------------------------------------------------------
    # Inference (mirrors upstream CATCH.detect_score)
    # --------------------------------------------------------------

    def predict(self, test_X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        if self.scaler is None:
            raise RuntimeError("Scaler not fit. Call fit() first.")

        # Apply paper-faithful StandardScaler.transform on test data (train-only fit).
        test_X = self.scaler.transform(test_X).astype(np.float32)

        N_test, n_features = test_X.shape
        if N_test < self.win_size:
            raise ValueError(f"Test sequence length {N_test} shorter than win_size {self.win_size}")

        n_windows = N_test - self.win_size + 1
        point_scores = np.full(N_test, -np.inf, dtype=np.float32)
        self.model.eval()

        temp_criterion = nn.MSELoss(reduction='none')
        freq_criterion_fn = frequency_criterion(self._config)
        score_lambda = self._config.score_lambda

        n_batches = (n_windows + self.batch_size - 1) // self.batch_size
        with torch.no_grad():
            for batch_idx in range(n_batches):
                batch_start = batch_idx * self.batch_size
                batch_end = min(batch_start + self.batch_size, n_windows)
                actual_bs = batch_end - batch_start

                batch_windows = np.zeros((actual_bs, self.win_size, n_features), dtype=np.float32)
                for j, w_idx in enumerate(range(batch_start, batch_end)):
                    batch_windows[j] = test_X[w_idx: w_idx + self.win_size]

                input_x = torch.from_numpy(batch_windows).to(self.device)
                outputs, _, _ = self.model(input_x)

                # Per upstream detect_score: temp_score [B,L], freq_score [B,L]
                temp_score = torch.mean(temp_criterion(input_x, outputs), dim=-1)  # [B, L]
                freq_score = torch.mean(freq_criterion_fn(input_x, outputs), dim=-1)  # [B, L]
                window_scores = (temp_score + score_lambda * freq_score).cpu().numpy()  # [B, L]

                # Max-aggregate over overlapping windows
                for j, w_idx in enumerate(range(batch_start, batch_end)):
                    for pos in range(self.win_size):
                        t = w_idx + pos
                        if t < N_test:
                            s = float(window_scores[j, pos])
                            if s > point_scores[t]:
                                point_scores[t] = s

                if self.verbose and (batch_idx + 1) % max(1, n_batches // 10) == 0:
                    progress = (batch_idx + 1) / n_batches * 100
                    print(f"\r  Inference: [{batch_idx + 1}/{n_batches}] {progress:5.1f}%", end="")

        if self.verbose:
            print()
        # Replace any uninitialized positions (shouldn't happen if N>=win) with 0
        point_scores[point_scores == -np.inf] = 0.0
        return point_scores

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
            "ch_mask_ratio": self.ch_mask_ratio,
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
