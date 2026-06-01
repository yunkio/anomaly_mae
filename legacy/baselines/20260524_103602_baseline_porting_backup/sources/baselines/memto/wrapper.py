"""
MEMTO Baseline Wrapper

Memory-guided Transformer with 2-phase training:
  Phase 1: random memory init → train encoder + memory gate (collect encoder outputs)
  K-means: cluster collected outputs into M cluster centers
  Phase 2: rebuild model with K-means cluster centers as memory init → continue training

Inference: anomaly score = recon_loss(t) * gathering_loss(t)  (per-timestep).

Interface: comparison.baseline_common.run_sota_baseline_with_epoch_eval()
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Dataset
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
        lambda_gather: float = 0.1,
        lambda_entropy: float = 0.1,
        lr: float = 1e-4,
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
        self.lambda_gather = lambda_gather
        self.lambda_entropy = lambda_entropy
        self.lr = lr
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

    def _train_one_epoch(self, loader, optimizer, gather_fn, entropy_fn, recon_fn, desc):
        self.model.train()
        epoch_loss_sum = 0.0
        n_batches = 0
        iterator = tqdm(loader, desc=desc, leave=False) if self.verbose else loader
        for batch in iterator:
            input_x = batch.to(self.device)
            optimizer.zero_grad()
            outputs = self.model(input_x)
            recon = outputs["out"]
            queries = outputs["queries"]  # (N, L, d_model)
            mem = outputs["mem"]
            attn = outputs.get("attn")

            recon_loss = recon_fn(recon, input_x)
            gather_loss = gather_fn(queries, mem)
            entropy_loss = entropy_fn(attn) if attn is not None else torch.tensor(0.0, device=self.device)
            loss = recon_loss + self.lambda_gather * gather_loss + self.lambda_entropy * entropy_loss

            loss.backward()
            optimizer.step()
            epoch_loss_sum += loss.item()
            n_batches += 1
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

    def fit(self, train_X: np.ndarray, epoch_callback=None) -> "MEMTOBaseline":
        """Train MEMTO with 2-phase schedule.

        Phase 1 (epochs [1..phase1_epochs]): random memory init, end-to-end training.
        K-means on collected encoder outputs.
        Phase 2 (epochs [phase1_epochs+1..epochs]): rebuilt model with K-means centers,
            optimizer reset.
        """
        self.n_features = train_X.shape[1]
        self.model = self._build_model(memory_init_embedding=None, phase_type="first_train")

        dataset = _SlidingWindowDataset(train_X, self.win_size, stride=self.train_stride)
        if self.verbose:
            print(f"  Created {len(dataset)} training windows (stride={self.train_stride})")
        loader_train = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        loader_collect = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        recon_fn = nn.MSELoss()
        gather_fn = GatheringLoss(reduce=True)
        entropy_fn = EntropyLoss()

        self.train_loss_history = []

        # ---------- Phase 1 ----------
        for epoch in range(self.phase1_epochs):
            avg = self._train_one_epoch(
                loader_train, optimizer, gather_fn, entropy_fn, recon_fn,
                desc=f"P1 {epoch + 1}/{self.phase1_epochs}",
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

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        remaining_epochs = self.epochs - self.phase1_epochs
        for epoch in range(remaining_epochs):
            avg = self._train_one_epoch(
                loader_train, optimizer, gather_fn, entropy_fn, recon_fn,
                desc=f"P2 {epoch + 1}/{remaining_epochs}",
            )
            self.train_loss_history.append(avg)
            global_epoch = self.phase1_epochs + epoch + 1
            if self.verbose:
                print(f"  [Phase 2] Epoch {global_epoch}: loss = {avg:.6f}")
            if epoch_callback is not None:
                epoch_callback(self, global_epoch)

        # Mark for test-time behavior
        self.model.mem_module.phase_type = "test"
        return self

    def predict(self, test_X: np.ndarray) -> np.ndarray:
        """1D anomaly score per timestep.

        Per-timestep score = recon_mse(t) * gather_loss(t)  (latent × memory distance).
        Aggregate across overlapping windows via max.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        # Ensure memory module is in 'test' mode (no buffer update)
        prev_phase = self.model.mem_module.phase_type
        self.model.mem_module.phase_type = "test"
        self.model.eval()

        N_test, n_features = test_X.shape
        if N_test < self.win_size:
            raise ValueError(f"Test sequence length {N_test} shorter than win_size {self.win_size}")

        n_windows = N_test - self.win_size + 1
        point_scores = np.zeros(N_test, dtype=np.float32)
        gather_pointwise = GatheringLoss(reduce=False)

        n_batches = (n_windows + self.batch_size - 1) // self.batch_size

        with torch.no_grad():
            for batch_idx in range(n_batches):
                batch_start = batch_idx * self.batch_size
                batch_end = min(batch_start + self.batch_size, n_windows)
                actual_bs = batch_end - batch_start

                batch_windows = np.zeros((actual_bs, self.win_size, n_features), dtype=np.float32)
                for j, w_idx in enumerate(range(batch_start, batch_end)):
                    batch_windows[j] = test_X[w_idx : w_idx + self.win_size]

                input_x = torch.from_numpy(batch_windows).to(self.device)
                outputs = self.model(input_x)
                recon = outputs["out"]
                queries = outputs["queries"]
                mem = outputs["mem"]

                # Per-timestep recon MSE: mean over feature dim → (B, L)
                recon_mse = ((recon - input_x) ** 2).mean(dim=-1)  # (B, L)
                # Per-timestep gather distance: (B, L)
                gather_pt = gather_pointwise(queries, mem)  # (B, L)
                # Combined score per upstream test.py: latent_score = softmax(-gather, dim=-1) → scale recon
                # Use multiplicative combination per paper
                score_bl = (recon_mse * gather_pt).cpu().numpy()  # (B, L)

                for j, w_idx in enumerate(range(batch_start, batch_end)):
                    for pos in range(self.win_size):
                        t = w_idx + pos
                        if t < N_test:
                            point_scores[t] = max(point_scores[t], float(score_bl[j, pos]))

                if self.verbose and (batch_idx + 1) % max(1, n_batches // 10) == 0:
                    progress = (batch_idx + 1) / n_batches * 100
                    print(f"\r  Inference: [{batch_idx + 1}/{n_batches}] {progress:5.1f}%", end="")

        if self.verbose:
            print()
        self.model.mem_module.phase_type = prev_phase
        return point_scores

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
            "n_features": self.n_features,
        }
        with open(save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    def load(self, save_dir: Path) -> "MEMTOBaseline":
        save_dir = Path(save_dir)
        with open(save_dir / "config.json", "r") as f:
            config = json.load(f)
        for k in ("win_size", "n_memory", "d_model", "n_heads", "e_layers", "d_ff",
                  "dropout", "shrink_thres", "n_features"):
            setattr(self, k, config[k])
        # Rebuild model with random memory init; state_dict overwrites memory buffer
        self.model = self._build_model(memory_init_embedding=None, phase_type="test")
        self.model.load_state_dict(torch.load(save_dir / "model.pt", map_location=self.device))
        self.model.eval()
        return self
