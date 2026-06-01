"""
MEMTO Baseline Wrapper (paper-faithful)

Memory-guided Transformer with 2-phase training:
  Phase 1: random memory init → train encoder + memory gate (collect encoder outputs)
  K-means: cluster collected outputs into M cluster centers
  Phase 2: rebuild model with K-means cluster centers as memory init → continue
           training with lr=phase2_lr (5e-5)

Training loss (upstream `solver.py:195-198`):
    loss = recon_mse + lambda_entropy * entropy(attn)        # lambd=0.01, NO gather term

Inference anomaly score (upstream `solver.py:258-261`):
    rec_loss     = mean( (x - recon)**2, dim=-1)             # (B, L)
    latent_score = softmax(gathering_loss / temperature, dim=-1)  # τ=0.1 over L
    score        = latent_score * rec_loss                   # (B, L)

Normalization (Path B): wrapper applies StandardScaler internally
  (driver passes raw data via SELF_NORMALIZING_SOTA membership in run_baseline.py).

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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Subset

from comparison.segment_utils import compute_segment_safe_window_indices
from comparison.baselines._boundary_safe_window import per_entity_concat
from tqdm import tqdm

from .model import EntropyLoss, GatheringLoss, TransformerVar


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


class MEMTOBaseline:
    """MEMTO (NeurIPS 2023) wrapper for sliding-window MTS anomaly detection.

    Original code: https://github.com/gunny97/MEMTO (no LICENSE — attribution only).
    Hyperparameters from `MODEL_PRESETS['default']['memto']`.

    2-phase training: epochs split into `phase1_epochs` + (epochs - phase1_epochs) phase-2 epochs.
    Memory is re-initialized via K-means on collected encoder outputs at boundary.
    Phase 2 uses a lower learning rate (`phase2_lr=5e-5`) per upstream `test.sh`.
    """

    def __init__(
        self,
        win_size: int = 100,
        n_memory: int = 10,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.0,
        shrink_thres: float = 0.0,
        lambda_entropy: float = 0.01,
        temperature: float = 0.1,
        lr: float = 1e-4,
        phase2_lr: float = 5e-5,
        batch_size: int = 128,
        epochs: int = 10,
        phase1_epochs: int = 3,
        train_stride: int = 1,
        kmeans_max_samples: int = 50_000,
        kmeans_random_state: int = 0,
        device: Optional[str] = None,
        verbose: bool = True,
    ):
        self.win_size = win_size
        self.n_memory = n_memory
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.shrink_thres = shrink_thres
        self.lambda_entropy = lambda_entropy
        self.temperature = temperature
        self.lr = lr
        self.phase2_lr = phase2_lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.phase1_epochs = min(phase1_epochs, max(epochs - 1, 1))
        self.train_stride = train_stride
        self.kmeans_max_samples = kmeans_max_samples
        self.kmeans_random_state = kmeans_random_state
        self.verbose = verbose

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model: Optional[TransformerVar] = None
        self.n_features: Optional[int] = None
        self.train_loss_history: list = []
        # Path B: wrapper-internal StandardScaler (upstream data_loader.py:SMDSegLoader).
        # Driver passes raw data (memto ∈ SELF_NORMALIZING_SOTA in run_baseline.py).
        self.scaler: Optional[StandardScaler] = None

    @property
    def name(self) -> str:
        return "MEMTO"

    def _build_model(self, memory_init_embedding=None, phase_type: str = "first_train") -> TransformerVar:
        return TransformerVar(
            win_size=self.win_size,
            enc_in=self.n_features,
            c_out=self.n_features,
            n_memory=self.n_memory,
            shrink_thres=self.shrink_thres,
            d_model=self.d_model,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
            memory_init_embedding=memory_init_embedding,
            memory_initial=False,
            phase_type=phase_type,
        ).to(self.device)

    def _train_one_epoch(self, loader, optimizer, entropy_fn, recon_fn, desc, phase=None, epoch_1idx=None):
        """One training epoch.

        Loss = recon_mse + lambda_entropy * entropy(attn)
        Matches upstream solver.py:195-198 (lambd=0.01, NO gather term in training).
        """
        self.model.train()
        epoch_loss_sum = 0.0
        n_batches = 0
        n_batches_total = len(loader)
        iterator = tqdm(loader, desc=desc, leave=False) if self.verbose else loader
        for batch_idx, batch in enumerate(iterator):
            batch_start = time.time()
            input_x = batch.to(self.device)
            optimizer.zero_grad()
            outputs = self.model(input_x)
            recon = outputs["out"]
            attn = outputs.get("attn")

            recon_loss = recon_fn(recon, input_x)
            entropy_loss = entropy_fn(attn) if attn is not None else torch.tensor(0.0, device=self.device)
            loss = recon_loss + self.lambda_entropy * entropy_loss

            loss.backward()
            optimizer.step()
            epoch_loss_sum += loss.item()
            n_batches += 1
            phase_str = f" phase={phase}" if phase is not None else ""
            ep = epoch_1idx if epoch_1idx is not None else 0
            print(f"[BATCH] model={self.name}{phase_str} epoch={ep}/{self.epochs} batch={batch_idx + 1}/{n_batches_total} time={time.time() - batch_start:.3f}s", flush=True)
        return epoch_loss_sum / max(n_batches, 1)

    def _collect_encoder_outputs(self, loader) -> np.ndarray:
        """Run encoder on all windows; return concatenated (T_total, d_model) on CPU."""
        self.model.eval()
        feats = []
        n_collected = 0
        with torch.no_grad():
            for batch in loader:
                input_x = batch.to(self.device)
                # Manually run embedding + encoder to skip memory module
                emb = self.model.embedding(input_x)
                enc_out = self.model.encoder(emb)  # (N, L, d_model)
                flat = enc_out.contiguous().view(-1, enc_out.size(-1)).cpu().numpy()
                feats.append(flat)
                n_collected += flat.shape[0]
                if n_collected >= self.kmeans_max_samples:
                    break
        feats = np.concatenate(feats, axis=0)
        if feats.shape[0] > self.kmeans_max_samples:
            idx = np.random.RandomState(self.kmeans_random_state).choice(
                feats.shape[0], self.kmeans_max_samples, replace=False
            )
            feats = feats[idx]
        return feats

    def fit(self, train_X: np.ndarray, epoch_callback=None, train_segments=None) -> "MEMTOBaseline":
        """Train MEMTO with 2-phase schedule.

        Phase 1 (epochs [1..phase1_epochs]): random memory init, end-to-end training (lr).
        K-means on collected encoder outputs.
        Phase 2 (epochs [phase1_epochs+1..epochs]): rebuilt model with K-means centers,
            optimizer reset with lr=phase2_lr.

        Path B normalization: StandardScaler.fit_transform(train_X) applied before
        any windowing (upstream data_loader.py:SMDSegLoader pattern).
        """
        # ---------- Path B: paper-faithful StandardScaler (train-only fit) ----------
        self.scaler = StandardScaler()
        self.scaler.fit(train_X)
        train_X = self.scaler.transform(train_X).astype(np.float32)

        self.n_features = train_X.shape[1]
        self.model = self._build_model(memory_init_embedding=None, phase_type="first_train")

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
        loader_train = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        loader_collect = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        recon_fn = nn.MSELoss()
        entropy_fn = EntropyLoss()

        self.train_loss_history = []

        # ---------- Phase 1 ----------
        for epoch in range(self.phase1_epochs):
            avg = self._train_one_epoch(
                loader_train, optimizer, entropy_fn, recon_fn,
                desc=f"P1 {epoch + 1}/{self.phase1_epochs}",
                phase=1, epoch_1idx=epoch + 1,
            )
            self.train_loss_history.append(avg)
            if self.verbose:
                print(f"  [Phase 1] Epoch {epoch + 1}: loss = {avg:.6f}")
            if epoch_callback is not None:
                epoch_callback(self, epoch + 1)

        # ---------- K-means ----------
        if self.verbose:
            print(f"  Running K-means (k={self.n_memory}) on encoder outputs ...")
        feats = self._collect_encoder_outputs(loader_collect)
        if self.verbose:
            print(f"    Encoder feature samples: {feats.shape}")
        kmeans = KMeans(n_clusters=self.n_memory, random_state=self.kmeans_random_state, n_init=10)
        kmeans.fit(feats)
        centers = torch.from_numpy(kmeans.cluster_centers_).float()  # (M, d_model)

        # ---------- Phase 2: rebuild model with cluster centers as memory init ----------
        # Save encoder + decoder weights, rebuild model with new memory, reload weights
        old_state = self.model.state_dict()
        self.model = self._build_model(memory_init_embedding=centers, phase_type="second_train")
        # Restore non-memory weights (memory buffer comes from cluster centers in fresh init)
        new_state = self.model.state_dict()
        for k, v in old_state.items():
            if k.startswith("mem_module.mem"):
                continue
            if k in new_state and new_state[k].shape == v.shape:
                new_state[k] = v
        self.model.load_state_dict(new_state)

        # Phase 2 lr per upstream test.sh (--lr 5e-5 for mode=memory_initial)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.phase2_lr)
        if self.verbose:
            print(f"  [Phase 2] Optimizer rebuilt with phase2_lr={self.phase2_lr}")

        remaining_epochs = self.epochs - self.phase1_epochs
        for epoch in range(remaining_epochs):
            global_epoch = self.phase1_epochs + epoch + 1
            avg = self._train_one_epoch(
                loader_train, optimizer, entropy_fn, recon_fn,
                desc=f"P2 {epoch + 1}/{remaining_epochs}",
                phase=2, epoch_1idx=global_epoch,
            )
            self.train_loss_history.append(avg)
            if self.verbose:
                print(f"  [Phase 2] Epoch {global_epoch}: loss = {avg:.6f}")
            if epoch_callback is not None:
                epoch_callback(self, global_epoch)

        # Mark for test-time behavior
        self.model.mem_module.phase_type = "test"
        return self

    def predict(self, test_X: np.ndarray, test_segments=None) -> np.ndarray:
        """1D anomaly score per timestep.

        Per-window score formula (upstream solver.py:258-261):
            rec_loss     = mean((x - recon)**2, dim=-1)                       # (B, L)
            latent_score = softmax(gathering_loss / temperature, dim=-1)      # τ=0.1 over L
            score_bl     = latent_score * rec_loss                            # (B, L)

        Reference-faithful aggregation (upstream solver.py:275, data_loader.py:
        SMDSegLoader/__getitem__ with default step=100):
          * Test windowing is non-overlap, stride = win_size (=100).
          * `np.concatenate(per_window_scores, axis=0).reshape(-1)` flattens
            (n_windows, win_size) → (n_windows * win_size,).
        Phase 2.5 correction (2026-05-24): switched from stride=1 + max-over-overlap
        to non-overlap stride=win_size + concat.reshape(-1), matching upstream exactly.

        Tail handling: upstream is IMPLICIT_NONE (output length =
        `(N_test - W) // W + 1) * W` ≤ N_test; tail `T mod W` is discarded).
        User-directed Option B fallback applied to satisfy the wrapper's
        T_test length contract:
          * Tail: repeat-last per-timestep score for `scores[valid_len:T_test]`.
          * Head: forward-fill from first valid score if any leading gap.

        BOUNDARY-SAFE TEST WINDOWING (2026-06-02):
          MEMTO's anomaly score is fully PER-WINDOW (the softmax-over-L weighting and
          the recon MSE are computed inside each non-overlap window; upstream then just
          `np.concatenate(...).reshape(-1)`). There is NO whole-test score
          post-processing (no median-IQR, no smoothing, no max-over-features). So the
          ENTIRE windowing + inference + per-window→per-timestep mapping (incl. the
          head/tail fill that satisfies the (T_test,) contract) is the RAW score
          producer; nothing runs after it.

          On multi-entity datasets (SMD machines / SMAP-MSL channels / Exathlon apps)
          ``test_segments`` are the per-entity (lo,hi) TEST-LOCAL slices from the SAME
          source as normalization (``loader.get_file_norm_segments()`` test side). We
          route the raw producer through ``per_entity_concat`` so every non-overlap
          window lives inside ONE entity — no window spans an entity boundary.
          ``per_entity_concat`` runs the raw producer independently per slice and
          concatenates the (Li,) outputs.

          Normalization is UNCHANGED: the train-fit ``StandardScaler`` is applied to
          the WHOLE test array BEFORE per-entity slicing. The scaler is fit on TRAIN
          only (upstream data_loader.py:L20-L26), so slicing the test array cannot
          change it — the per-slice raw scores are bit-identical to the whole-array
          legacy scores for any single entity.

          Single-entity / None / non-tiling ``test_segments`` → ``per_entity_concat``
          makes ONE call over the whole (normalized) array == bit-identical legacy NO-OP.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        if self.scaler is None:
            raise RuntimeError("Scaler not fit. Call fit() before predict().")

        # Apply paper-faithful StandardScaler.transform on test data (train-only fit).
        # Done on the WHOLE test array BEFORE per-entity windowing — the scaler is
        # train-fit, hence slice-invariant; normalization granularity is unchanged.
        test_X = self.scaler.transform(test_X).astype(np.float32)

        # Ensure memory module is in 'test' mode (no buffer update)
        prev_phase = self.model.mem_module.phase_type
        self.model.mem_module.phase_type = "test"
        self.model.eval()

        n_features = test_X.shape[1]
        gather_pointwise = GatheringLoss(reduce=False)

        def _raw_fn(sub_X: np.ndarray) -> np.ndarray:
            """Raw per-timestep MEMTO score for ONE entity slice (len(sub_X),).

            Exactly the legacy non-overlap windowing + inference + concat→reshape(-1)
            + head/tail fill, restricted to this slice. Edge-safe: a slice shorter than
            ``win_size`` yields zero windows — return all-zeros (neutral, finite) rather
            than crashing. Such a tiny entity contributes no model signal; zeros keep the
            (T_test,) contract intact and never inject a spurious high/low rank.
            """
            n_test = sub_X.shape[0]

            # ------------------------------------------------------------
            # Reference-faithful non-overlap windowing (stride = win_size).
            # n_windows = (n_test - W) // W + 1   (upstream SMDSegLoader.__len__)
            # ------------------------------------------------------------
            stride = self.win_size
            if n_test < self.win_size:
                # Edge case (zero-tolerance): entity slice shorter than one window.
                # No window can be formed; emit a neutral all-zero score (finite).
                return np.zeros(n_test, dtype=np.float32)

            n_windows = (n_test - self.win_size) // stride + 1
            valid_len = n_windows * self.win_size  # ≤ n_test

            per_window_scores = []  # list of (B, L) np arrays → concat → reshape(-1)
            n_batches = (n_windows + self.batch_size - 1) // self.batch_size

            with torch.no_grad():
                for batch_idx in range(n_batches):
                    batch_start = batch_idx * self.batch_size
                    batch_end = min(batch_start + self.batch_size, n_windows)
                    actual_bs = batch_end - batch_start

                    batch_windows = np.zeros((actual_bs, self.win_size, n_features), dtype=np.float32)
                    for j, w_idx in enumerate(range(batch_start, batch_end)):
                        start = w_idx * stride
                        batch_windows[j] = sub_X[start : start + self.win_size]

                    input_x = torch.from_numpy(batch_windows).to(self.device)
                    outputs = self.model(input_x)
                    recon = outputs["out"]
                    queries = outputs["queries"]
                    mem = outputs["mem"]

                    # Per-timestep recon MSE: mean over feature dim → (B, L)
                    recon_mse = ((recon - input_x) ** 2).mean(dim=-1)  # (B, L)
                    # Per-timestep gather distance: (B, L)
                    gather_pt = gather_pointwise(queries, mem)  # (B, L)
                    # Paper-faithful: softmax over sequence dim, scaled by temperature τ=0.1
                    latent_score = torch.softmax(gather_pt / self.temperature, dim=-1)  # (B, L)
                    score_bl = (latent_score * recon_mse).cpu().numpy()  # (B, L)
                    per_window_scores.append(score_bl)

                    if self.verbose and (batch_idx + 1) % max(1, n_batches // 10) == 0:
                        progress = (batch_idx + 1) / n_batches * 100
                        print(f"\r  Inference: [{batch_idx + 1}/{n_batches}] {progress:5.1f}%", end="")

            # --------------------------------------------------------
            # Reference-faithful aggregation: concat → reshape(-1)
            # (upstream solver.py:275 `np.concatenate(...).reshape(-1)`)
            # --------------------------------------------------------
            valid_scores = np.concatenate(per_window_scores, axis=0).reshape(-1).astype(np.float32)
            assert valid_scores.shape[0] == valid_len, (
                f"valid_scores len {valid_scores.shape[0]} != expected {valid_len}"
            )

            # --------------------------------------------------------
            # Option B fallback (user-directed; reference is IMPLICIT_NONE):
            #   head: forward-fill from first valid score (head_len=0 — non-overlap
            #         windowing starts at index 0)
            #   tail: repeat-last per-timestep score for [valid_len, n_test)
            # --------------------------------------------------------
            scores = np.empty(n_test, dtype=np.float32)
            head_len = 0  # non-overlap windowing emits scores aligned to start
            scores[head_len : head_len + valid_len] = valid_scores
            tail_len = n_test - valid_len - head_len
            if head_len > 0:
                scores[:head_len] = valid_scores[0]
            if tail_len > 0:
                scores[head_len + valid_len:] = valid_scores[-1]
            return scores

        # Boundary-safe: window+infer per entity (no window spans a boundary).
        # Single-entity / None → ONE call over the whole array (bit-identical no-op).
        scores = per_entity_concat(test_X, test_segments, _raw_fn)

        if self.verbose:
            print()
        self.model.mem_module.phase_type = prev_phase
        return scores

    def save(self, save_dir: Path) -> None:
        if self.model is None:
            raise RuntimeError("No model to save.")
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), save_dir / "model.pt")
        config = {
            "model_name": "MEMTO",
            "win_size": self.win_size,
            "n_memory": self.n_memory,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "e_layers": self.e_layers,
            "d_ff": self.d_ff,
            "dropout": self.dropout,
            "shrink_thres": self.shrink_thres,
            "lambda_entropy": self.lambda_entropy,
            "temperature": self.temperature,
            "lr": self.lr,
            "phase2_lr": self.phase2_lr,
            "n_features": self.n_features,
        }
        with open(save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        # Persist Path B scaler so predict() after load() can transform test data.
        if self.scaler is not None:
            joblib.dump(self.scaler, save_dir / "scaler.pkl")

    def load(self, save_dir: Path) -> "MEMTOBaseline":
        save_dir = Path(save_dir)
        with open(save_dir / "config.json", "r") as f:
            config = json.load(f)
        for k in ("win_size", "n_memory", "d_model", "n_heads", "e_layers", "d_ff",
                  "dropout", "shrink_thres", "n_features"):
            setattr(self, k, config[k])
        # Newly-added keys; tolerate older configs via .get
        self.lambda_entropy = config.get("lambda_entropy", self.lambda_entropy)
        self.temperature = config.get("temperature", self.temperature)
        self.lr = config.get("lr", self.lr)
        self.phase2_lr = config.get("phase2_lr", self.phase2_lr)
        # Rebuild model with random memory init; state_dict overwrites memory buffer
        self.model = self._build_model(memory_init_embedding=None, phase_type="test")
        self.model.load_state_dict(torch.load(save_dir / "model.pt", map_location=self.device))
        self.model.eval()
        # Restore Path B scaler
        scaler_path = save_dir / "scaler.pkl"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
        return self
