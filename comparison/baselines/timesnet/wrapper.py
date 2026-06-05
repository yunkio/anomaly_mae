"""
TimesNet Baseline Wrapper

Reconstruction-based MTS anomaly detection. Trained with MSE on sliding windows.
Anomaly score per timestep = MSE between input and reconstruction. Faithful to
upstream exp_anomaly_detection.py:150,165,170-171, which emits EVERY window
position via reshape(-1); here realized as the length-N_test overlap-average
(each timestep = mean of its per-position MSEs over all covering windows).

Training schedule: per-epoch learning rate halving (upstream `lradj='type1'`,
``lr = base_lr * 0.5^(epoch-1)`` where ``epoch`` is 1-indexed). Reproduces
``utils/tools.py:adjust_learning_rate`` of the Time-Series-Library, which the
upstream ``exp/exp_anomaly_detection.py`` calls at the end of every epoch with
``args.lradj`` defaulting to ``type1`` (run.py argparse).

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
from comparison.baselines._boundary_safe_window import per_entity_concat, is_multi_entity
from tqdm import tqdm

from .model import TimesNet


class _SlidingWindowDataset(Dataset):
    """Lazy sliding window dataset (matches anomaly_transformer pattern)."""

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


class TimesNetBaseline:
    """TimesNet (ICLR 2023) wrapper for sliding-window MTS anomaly detection.

    Original code: https://github.com/thuml/Time-Series-Library (MIT).
    Hyperparameters from `MODEL_PRESETS['default']['timesnet']`.
    """

    def __init__(
        self,
        win_size: int = 100,
        d_model: int = 64,
        d_ff: int = 64,
        e_layers: int = 3,
        top_k: int = 3,
        num_kernels: int = 6,
        dropout: float = 0.1,
        lr: float = 1e-4,
        batch_size: int = 128,
        epochs: int = 10,
        train_stride: int = 1,
        device: Optional[str] = None,
        verbose: bool = True,
    ):
        self.win_size = win_size
        self.d_model = d_model
        self.d_ff = d_ff
        self.e_layers = e_layers
        self.top_k = top_k
        self.num_kernels = num_kernels
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_stride = train_stride
        self.verbose = verbose

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model: Optional[TimesNet] = None
        self.n_features: Optional[int] = None
        self.train_loss_history: list = []
        # Paper-faithful normalization: upstream Time-Series-Library SWATSegLoader uses
        # StandardScaler fit on train_data, then transform both train+test. Driver passes
        # raw data (normalize_mode='none' via run_baseline.py override).
        self.scaler: Optional[StandardScaler] = None

    @property
    def name(self) -> str:
        return "TimesNet"

    def _build_model(self) -> TimesNet:
        return TimesNet(
            seq_len=self.win_size,
            c_in=self.n_features,
            d_model=self.d_model,
            d_ff=self.d_ff,
            e_layers=self.e_layers,
            top_k=self.top_k,
            num_kernels=self.num_kernels,
            dropout=self.dropout,
        ).to(self.device)

    def fit(self, train_X: np.ndarray, epoch_callback=None, train_segments=None) -> "TimesNetBaseline":
        """Train TimesNet via reconstruction MSE.

        Args:
            train_X: (N_train, n_features) — raw (driver passes normalize_mode='none'
                for self-normalizing SOTA). Wrapper applies StandardScaler to match
                upstream Time-Series-Library data_provider.
            epoch_callback(self, ep+1): optional, invoked after each epoch
        """
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

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

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
                output = self.model(input_x)
                loss = criterion(output, input_x)
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

            # --- Upstream type1 LR schedule (utils/tools.py:adjust_learning_rate) ---
            # exp/exp_anomaly_detection.py calls
            #   adjust_learning_rate(model_optim, epoch + 1, self.args)
            # at the end of each epoch, where `epoch` is 0-indexed in upstream's
            # for-loop too. With `args.lradj='type1'` (run.py default) and
            # `args.learning_rate=self.lr`, that resolves to
            #   lr_new = self.lr * (0.5 ** ((epoch + 1 - 1) // 1))
            #          = self.lr * (0.5 ** epoch)
            # i.e. after epoch=0 lr stays at base, after epoch=1 lr → base/2, etc.
            next_lr = self.lr * (0.5 ** epoch)
            for pg in optimizer.param_groups:
                pg["lr"] = next_lr
            if self.verbose and epoch > 0:
                print(f"  Adjusted LR to {next_lr:.3e} (type1 schedule)")

        return self

    def predict(self, test_X: np.ndarray, test_segments=None) -> np.ndarray:
        """Compute 1D anomaly score per timestep (reconstruction MSE, all-position).

        Faithful to upstream ``exp/exp_anomaly_detection.py::test()``
        (thuml/Time-Series-Library, re-verified live 2026-06-04). Upstream emits
        EVERY window position, not a single per-window scalar:

            score = torch.mean(MSELoss(reduce=False)(batch_x, outputs), dim=-1)
            #   -> shape [B, win] : per-window, per-position MSE (mean over features)
            attens_energy = np.concatenate(...).reshape(-1)
            #   -> length n_win*win : ALL positions of ALL step-1 windows kept

        (exp_anomaly_detection.py:150, 165, 170-171). Over step-1 test windows
        (PSM/SWaT loaders use ``step=1`` -> data_loader.py:413,475) every true
        timestep t is covered by ``win`` overlapping windows and contributes a
        score from each. Upstream then thresholds/evaluates over that duplicated
        length-``n_win*win`` array (labels duplicated identically).

        Our harness owns a length-``N_test`` per-timestep label vector (``test_y``)
        and ``compute_all_metrics`` requires ``len(scores) == len(test_y)``
        (baseline_common.py:565-576). A literal ``reshape(-1)`` would need the
        harness to duplicate labels identically (a SHARED change we do not own).
        The length-``N_test`` realization that still uses ALL window positions is
        **overlap-averaging**: each timestep t receives the MEAN of the
        per-position MSEs from every window that covers it. This keeps every
        window position (matching exp:150,165) and is the in-harness equivalent
        of upstream's all-position emission, superseding the previous
        last-position + forward-fill convention (which used only ONE window
        context per timestep and was unfaithful to upstream's keep-all-positions
        ``reshape(-1)``).

        Per-window score formula (faithful to upstream):
            window_scores[b, t] = mean_features((output[b, t] - input[b, t]) ** 2)

        Window-to-timestep mapping (overlap-average):
            window w (start s = w*stride, stride=1) contributes window_scores[w, j]
            to timestep t = s + j for j in [0, win_size). Each timestep
            accumulates contributions from all covering windows; the final score
            is the mean. Coverage spans ALL t in [0, N_test) (the leading
            positions of early windows cover the head, so NO forward-fill is
            needed).

        Boundary-safe TEST windowing (multi-entity datasets):
            ``test_segments`` is the per-entity (lo, hi) TEST-LOCAL list from
            ``UnifiedLoader.get_file_norm_segments()`` (test side) — the SAME
            source as per-entity normalization. On multi-file datasets (SMD
            machines / SMAP-MSL channels / Exathlon apps) the sliding windower +
            model inference + per-window→per-timestep overlap-average is run
            INDEPENDENTLY on each entity's test slice via ``per_entity_concat``
            so NO window straddles an entity boundary.
            Single-entity / None / non-tiling segments -> ONE call over the whole
            array == bit-identical legacy behaviour (helper guarantees the no-op).

            NORMALIZATION IS UNCHANGED: the train-fit ``StandardScaler.transform``
            is applied to the WHOLE test array BEFORE windowing (below). Because it
            is a pure ``transform`` (scaler fit on train only, NOT re-fit on test),
            slicing the test array per-entity afterwards cannot change any
            normalized value — so this stays in the "untouchable-norm" class:
            only the WINDOWING is made boundary-safe, never the normalization.
            Upstream confirms train-only fit + transform on test
            (Time-Series-Library/data_provider/data_loader.py SMDSegLoader:
            ``self.scaler.fit(train_data); test_data = self.scaler.transform(test_data)``).
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        if self.scaler is None:
            raise RuntimeError("Scaler not fit. Call fit() first.")

        # Apply paper-faithful StandardScaler.transform on test data (train-only fit).
        # WHOLE-TEST normalization, done BEFORE windowing — untouchable-norm class.
        test_X = self.scaler.transform(test_X).astype(np.float32)

        N_test, n_features = test_X.shape
        # Legacy contract: the WHOLE test must be at least one window long. A
        # genuine per-entity sub-slice shorter than win_size is handled inside
        # raw_fn (edge-safe fallback) and must NOT reach this guard, so we only
        # raise for the whole-array case (preserves bit-identical legacy error).
        if N_test < self.win_size and not is_multi_entity(N_test, test_segments):
            raise ValueError(
                f"Test sequence length {N_test} shorter than win_size {self.win_size}"
            )

        self.model.eval()

        def raw_fn(sub_X: np.ndarray) -> np.ndarray:
            """RAW per-timestep score producer for ONE entity's test slice.

            Runs the canonical stride=1 sliding windower + model inference +
            per-window ALL-POSITION MSE + overlap-average over ``sub_X`` IN
            ISOLATION (no cross-entity dependence => boundary-safe). Returns
            ``(len(sub_X),)`` finite scores. There is NO whole-test
            post-processing in TimesNet, so the full raw score IS the final
            score; ``per_entity_concat`` simply stitches the per-entity raw
            arrays back together with no extra reduction needed afterwards.

            All-position emission (faithful to upstream
            exp_anomaly_detection.py:150,165,170-171): each window contributes
            ``win_size`` per-position MSE values (one per timestep it spans), not
            a single scalar. Over step-1 windows every timestep is covered by up
            to ``win_size`` windows; its final score is the MEAN of all those
            contributions (overlap-average). This is the length-``N_test``
            in-harness equivalent of upstream's keep-all-positions ``reshape(-1)``.

            Edge case (per-entity slice shorter than win_size): the TimesNet
            model is built with a FIXED ``seq_len=win_size`` and its internal
            period-folding reshape (``TimesBlock.forward``) assumes that exact
            length, so it CANNOT consume a shorter window. We therefore LEFT-PAD
            the slice up to ``win_size`` (edge-replicate the first row), score the
            single full-length window, and assign each real timestep its OWN
            per-position MSE (the window's all-position scores over the true rows,
            i.e. the tail ``L`` positions). This never crosses a boundary, keeps
            the model's input shape valid, and returns a finite ``(L,)`` score. It
            cannot occur on the legacy single-entity path for any real dataset
            (guarded above); it only protects pathologically short entity slices.
            """
            sub_X = np.asarray(sub_X, dtype=np.float32)
            L = sub_X.shape[0]

            # --- short-slice fallback: pad to win_size, score one window, take tail ---
            if L < self.win_size:
                pad = self.win_size - L
                # edge-replicate the FIRST row at the head so the real data sits at
                # the window TAIL; the real timesteps then read their own positions.
                padded = np.concatenate(
                    [np.repeat(sub_X[:1], pad, axis=0), sub_X], axis=0
                ).astype(np.float32)  # (win_size, F)
                with torch.no_grad():
                    ix = torch.from_numpy(padded[None]).to(self.device)  # (1, W, F)
                    out = self.model(ix)
                    ws = ((out - ix) ** 2).mean(dim=-1)            # (1, W)
                    tail = ws[0, pad:].cpu().numpy().astype(np.float32)  # (L,)
                return tail

            test_stride = 1
            n_windows = (L - self.win_size) // test_stride + 1  # = L - W + 1

            # Overlap-average accumulators (length L): sum of per-position MSEs and
            # the coverage count per timestep. Final score = sum / count.
            score_sum = np.zeros(L, dtype=np.float64)
            score_cnt = np.zeros(L, dtype=np.float64)

            n_batches = (n_windows + self.batch_size - 1) // self.batch_size
            with torch.no_grad():
                for batch_idx in range(n_batches):
                    batch_start = batch_idx * self.batch_size
                    batch_end = min(batch_start + self.batch_size, n_windows)
                    actual_bs = batch_end - batch_start

                    batch_windows = np.zeros(
                        (actual_bs, self.win_size, n_features), dtype=np.float32
                    )
                    for j, w_idx in enumerate(range(batch_start, batch_end)):
                        start = w_idx * test_stride
                        batch_windows[j] = sub_X[start : start + self.win_size]

                    input_x = torch.from_numpy(batch_windows).to(self.device)
                    output = self.model(input_x)
                    # Per-window per-position MSE, mean over features → [B, win]
                    window_scores = ((output - input_x) ** 2).mean(dim=-1)
                    ws_np = window_scores.cpu().numpy().astype(np.float64)  # [B, win]

                    # Scatter every window position into its timestep accumulator.
                    for j in range(actual_bs):
                        start = (batch_start + j) * test_stride
                        score_sum[start : start + self.win_size] += ws_np[j]
                        score_cnt[start : start + self.win_size] += 1.0

                    if self.verbose and (batch_idx + 1) % max(1, n_batches // 10) == 0:
                        progress = (batch_idx + 1) / n_batches * 100
                        print(f"\r  Inference: [{batch_idx + 1}/{n_batches}] {progress:5.1f}%", end="")

            if self.verbose:
                print()

            # Every timestep in [0, L) is covered by >=1 window (n_windows>=1 here
            # because L>=win_size), so count is strictly positive — no head fill.
            assert (score_cnt > 0).all(), "uncovered timestep in overlap-average"
            scores = (score_sum / score_cnt).astype(np.float32)
            return scores

        # Boundary-safe: per-entity windowing+inference, then concat. Single-entity
        # / None / non-tiling -> ONE raw_fn(whole test) == bit-identical legacy path.
        scores = per_entity_concat(test_X, test_segments, raw_fn)
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
            "model_name": "TimesNet",
            "win_size": self.win_size,
            "d_model": self.d_model,
            "d_ff": self.d_ff,
            "e_layers": self.e_layers,
            "top_k": self.top_k,
            "num_kernels": self.num_kernels,
            "n_features": self.n_features,
        }
        with open(save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    def load(self, save_dir: Path) -> "TimesNetBaseline":
        save_dir = Path(save_dir)
        with open(save_dir / "config.json", "r") as f:
            config = json.load(f)
        self.win_size = config["win_size"]
        self.d_model = config["d_model"]
        self.d_ff = config["d_ff"]
        self.e_layers = config["e_layers"]
        self.top_k = config["top_k"]
        self.num_kernels = config["num_kernels"]
        self.n_features = config["n_features"]

        self.model = self._build_model()
        self.model.load_state_dict(torch.load(save_dir / "model.pt", map_location=self.device))
        self.model.eval()
        scaler_path = save_dir / "scaler.pkl"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
        return self

    def get_model_info(self) -> dict:
        return {
            "model": "TimesNet",
            "win_size": self.win_size,
            "d_model": self.d_model,
            "d_ff": self.d_ff,
            "e_layers": self.e_layers,
            "top_k": self.top_k,
            "num_kernels": self.num_kernels,
            "dropout": self.dropout,
            "lr": self.lr,
            "lradj": "type1",  # upstream default, halves lr each epoch end
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "n_features": self.n_features,
        }
