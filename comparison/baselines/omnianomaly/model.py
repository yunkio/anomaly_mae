"""
OmniAnomaly - Stochastic Recurrent Neural Network for Multivariate Time Series Anomaly Detection

Paper: Su et al., "Robust Anomaly Detection for Multivariate Time Series through
       Stochastic Recurrent Neural Network", KDD 2019
       https://dl.acm.org/doi/10.1145/3292500.3330672
Reference repo: https://github.com/NetManAIOps/OmniAnomaly (master branch, TF1 codebase)
License: MIT (vendoring not required; PyTorch reimplementation here)

Paper-faithful PyTorch reimplementation (CLEAN_REIMPL 2026-05-24).

Three core contributions, all implemented:
    1. Linear Gaussian State Space Model (LGSSM) prior — random-walk identity transition
       with unit-variance Gaussian noise, closed-form per-step log-prob.
    2. RecurrentDistribution posterior — autoregressive q(z_t | z_{t-1}, h_t) over time.
    3. 20-layer Planar Normalizing Flow on top of the posterior, with log-det Jacobian
       correction in the ELBO.

Architecture (reference defaults):
    - Encoder: GRU(rnn_num_hidden=500) -> 2x Linear(dense_dim=500)
    - Decoder: GRU(rnn_num_hidden=500) -> 2x Linear(dense_dim=500) -> (mean, std)
    - Heads use softplus_std(h) = softplus(Linear(h)) + std_epsilon (epsilon=1e-4)
    - z_dim = 3, window_length = 100, nf_layers = 20

Training: full SGVB ELBO,
    L = E_q[ -log p(x|z_K) ] + E_q[ log q(z_0 | x) - sum_log_det - log p_LGSSM(z_K) ]
where z_K = NF(z_0). KL weighting `beta` defaults to 1.0 (paper-faithful).

Scoring: per-window reconstruction NLL averaged over `n_mc_samples` MC samples of z;
window-to-timestamp via last-position assignment (`scores[:, -1]`); first (seq_len-1)
positions forward-filled with the first window's score so the output length equals T_test
(Domain B contract).
"""

import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from comparison.segment_utils import compute_segment_safe_window_indices
from comparison.baselines._boundary_safe_window import per_entity_concat


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def _softplus_std(raw: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Match upstream wrapper.softplus_std: softplus(x) + epsilon (additive)."""
    return F.softplus(raw) + epsilon


class PlanarFlow(nn.Module):
    """Single Planar Normalizing Flow layer (Rezende & Mohamed 2015).

    f(z) = z + u_hat * tanh(w . z + b)
    where u_hat = u + (softplus(w . u) - 1 - w . u) * w / ||w||^2  (invertibility constraint)
    log|det df/dz| = log|1 + u_hat . psi|, psi = (1 - tanh^2(w.z + b)) * w
    """

    def __init__(self, dim: int):
        super().__init__()
        # Small init so the flow starts close to identity.
        self.u = nn.Parameter(torch.randn(dim) * 0.01)
        self.w = nn.Parameter(torch.randn(dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(1))

    def _u_hat(self) -> torch.Tensor:
        # Invertibility: u_hat = u + (m(w.u) - w.u) * w / (||w||^2), m(x) = -1 + softplus(x)
        wu = torch.dot(self.w, self.u)  # scalar
        m = -1.0 + F.softplus(wu)
        w_norm_sq = torch.dot(self.w, self.w) + 1e-8
        return self.u + (m - wu) * self.w / w_norm_sq

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: (..., dim)
        Returns:
            z_new: (..., dim)
            log_det: (...) — log|det df/dz| per sample/time-step
        """
        u_hat = self._u_hat()  # (dim,)
        # w . z over the last dim → broadcast w to z's shape via matmul on the last axis.
        wz = (z * self.w).sum(dim=-1, keepdim=True) + self.b  # (..., 1)
        h = torch.tanh(wz)  # (..., 1)
        z_new = z + h * u_hat  # broadcasting (..., dim)
        # psi = (1 - tanh^2) * w  → (..., dim)
        psi = (1.0 - h * h) * self.w  # (..., dim)
        # u_hat . psi → (...,)
        u_psi = (psi * u_hat).sum(dim=-1)
        log_det = torch.log(torch.abs(1.0 + u_psi) + 1e-8)
        return z_new, log_det


class PlanarFlowStack(nn.Module):
    """Sequential stack of PlanarFlow layers."""

    def __init__(self, dim: int, n_layers: int = 20):
        super().__init__()
        self.layers = nn.ModuleList([PlanarFlow(dim) for _ in range(n_layers)])

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: (..., dim)
        Returns:
            z_K: (..., dim)
            sum_log_det: (...) — summed over flow layers
        """
        sum_log_det = torch.zeros(z.shape[:-1], device=z.device, dtype=z.dtype)
        z_k = z
        for layer in self.layers:
            z_k, log_det = layer(z_k)
            sum_log_det = sum_log_det + log_det
        return z_k, sum_log_det


class RecurrentPosterior(nn.Module):
    """Autoregressive posterior q(z_t | z_{t-1}, h_t).

    Mirrors omni_anomaly/recurrent_distribution.py:
      input_q = concat(z_previous, h_t)
      mu_q    = Linear_mu(input_q)
      std_q   = softplus_std(Linear_std(input_q), epsilon)
      z_t     = mu_q + std_q * noise_t
    """

    def __init__(self, h_dim: int, z_dim: int, std_epsilon: float = 1e-4):
        super().__init__()
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.std_epsilon = std_epsilon
        self.mu_mlp = nn.Linear(h_dim + z_dim, z_dim)
        self.std_mlp = nn.Linear(h_dim + z_dim, z_dim)

    def sample(self, h: torch.Tensor):
        """
        Args:
            h: (B, T, h_dim) — encoder output features
        Returns:
            z:   (B, T, z_dim) — sampled latents (autoregressive)
            mu:  (B, T, z_dim) — per-step posterior mean
            std: (B, T, z_dim) — per-step posterior std
        """
        B, T, _ = h.shape
        device = h.device
        dtype = h.dtype
        z_prev = torch.zeros(B, self.z_dim, device=device, dtype=dtype)
        mus, stds, zs = [], [], []
        for t in range(T):
            inp = torch.cat([h[:, t, :], z_prev], dim=-1)  # (B, h+z)
            mu_t = self.mu_mlp(inp)  # (B, z)
            std_t = _softplus_std(self.std_mlp(inp), self.std_epsilon)
            eps = torch.randn_like(mu_t)
            z_t = mu_t + std_t * eps
            mus.append(mu_t)
            stds.append(std_t)
            zs.append(z_t)
            z_prev = z_t
        mu = torch.stack(mus, dim=1)
        std = torch.stack(stds, dim=1)
        z = torch.stack(zs, dim=1)
        return z, mu, std


def _gaussian_log_prob(x: torch.Tensor, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Per-element Gaussian log-density. Returns same shape as x."""
    log_2pi = 1.8378770664093453  # log(2 * pi)
    var = std * std
    return -0.5 * (log_2pi + 2.0 * torch.log(std + 1e-8) + (x - mu) ** 2 / (var + 1e-8))


class LGSSMPrior:
    """Random-walk LGSSM prior with identity transition + unit Gaussian noise.

    Reference (omni_anomaly/model.py:30-40, when use_connected_z_p=True):
        transition_matrix = I, transition_noise ~ N(0, I)
        observation_matrix = I, observation_noise ~ N(0, I)
        initial_state_prior ~ N(0, I)

    For analysis we use the *latent* random-walk prior directly:
        z_1 ~ N(0, I)
        z_t | z_{t-1} ~ N(z_{t-1}, I)  for t >= 2

    log p(z_{1:T}) = log N(z_1 | 0, I) + sum_{t>=2} log N(z_t | z_{t-1}, I)
    (Per-step per-dim form — caller sums over (T, z_dim) as needed.)
    """

    @staticmethod
    def log_prob(z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, T, z_dim)
        Returns:
            log_p: (B, T, z_dim) — per-step per-dim Gaussian log-density
        """
        z_prev = torch.zeros_like(z[:, :1, :])  # (B, 1, z) initial state prior mean = 0
        # Reference mean for each t: z_{t-1} (t=1 uses zero)
        mean_seq = torch.cat([z_prev, z[:, :-1, :]], dim=1)  # (B, T, z)
        std_seq = torch.ones_like(z)  # unit variance
        return _gaussian_log_prob(z, mean_seq, std_seq)


# ---------------------------------------------------------------------------
# Encoder / Decoder (RNN + 2-layer dense MLP)
# ---------------------------------------------------------------------------


class Encoder(nn.Module):
    """RNN + 2-layer dense MLP feature extractor (no head — RecurrentPosterior consumes h)."""

    def __init__(self, input_dim: int, rnn_hidden: int, dense_dim: int,
                 n_layers: int = 1, rnn_type: str = 'gru'):
        super().__init__()
        self.rnn_hidden = rnn_hidden
        self.dense_dim = dense_dim
        if rnn_type == 'gru':
            self.rnn = nn.GRU(input_dim, rnn_hidden, n_layers, batch_first=True)
        else:
            self.rnn = nn.LSTM(input_dim, rnn_hidden, n_layers, batch_first=True)
        # Reference uses 2 dense layers with default linear activation (no nonlinearity)
        self.dense1 = nn.Linear(rnn_hidden, dense_dim)
        self.dense2 = nn.Linear(dense_dim, dense_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(x)
        h = self.dense1(h)
        h = self.dense2(h)
        return h


class Decoder(nn.Module):
    """RNN + 2-layer dense MLP + (mean, softplus_std) heads on x."""

    def __init__(self, z_dim: int, rnn_hidden: int, dense_dim: int, output_dim: int,
                 std_epsilon: float = 1e-4, n_layers: int = 1, rnn_type: str = 'gru'):
        super().__init__()
        self.rnn_hidden = rnn_hidden
        self.dense_dim = dense_dim
        self.std_epsilon = std_epsilon
        if rnn_type == 'gru':
            self.rnn = nn.GRU(z_dim, rnn_hidden, n_layers, batch_first=True)
        else:
            self.rnn = nn.LSTM(z_dim, rnn_hidden, n_layers, batch_first=True)
        self.dense1 = nn.Linear(rnn_hidden, dense_dim)
        self.dense2 = nn.Linear(dense_dim, dense_dim)
        self.mean_head = nn.Linear(dense_dim, output_dim)
        self.std_head = nn.Linear(dense_dim, output_dim)

    def forward(self, z: torch.Tensor):
        h, _ = self.rnn(z)
        h = self.dense1(h)
        h = self.dense2(h)
        x_mu = self.mean_head(h)
        x_std = _softplus_std(self.std_head(h), self.std_epsilon)
        return x_mu, x_std


# ---------------------------------------------------------------------------
# OmniAnomaly orchestrator
# ---------------------------------------------------------------------------


class OmniAnomalyModel(nn.Module):
    """Full OmniAnomaly model: Encoder -> RecurrentPosterior -> PlanarNF -> Decoder."""

    def __init__(
        self,
        input_dim: int,
        rnn_hidden: int = 500,
        dense_dim: int = 500,
        z_dim: int = 3,
        nf_layers: int = 20,
        std_epsilon: float = 1e-4,
        n_layers: int = 1,
        rnn_type: str = 'gru',
    ):
        super().__init__()
        self.input_dim = input_dim
        self.rnn_hidden = rnn_hidden
        self.dense_dim = dense_dim
        self.z_dim = z_dim
        self.nf_layers = nf_layers
        self.std_epsilon = std_epsilon

        self.encoder = Encoder(input_dim, rnn_hidden, dense_dim, n_layers, rnn_type)
        self.posterior = RecurrentPosterior(dense_dim, z_dim, std_epsilon)
        if nf_layers > 0:
            self.nf = PlanarFlowStack(z_dim, n_layers=nf_layers)
        else:
            self.nf = None
        self.decoder = Decoder(z_dim, rnn_hidden, dense_dim, input_dim,
                               std_epsilon=std_epsilon, n_layers=n_layers, rnn_type=rnn_type)

    def _sample_once(self, x: torch.Tensor):
        """One forward pass: sample z_0 -> NF -> decode."""
        h = self.encoder(x)                          # (B, T, dense_dim)
        z0, mu_q, std_q = self.posterior.sample(h)   # (B, T, z_dim)
        if self.nf is not None:
            zK, sum_log_det = self.nf(z0)
        else:
            zK = z0
            sum_log_det = torch.zeros(z0.shape[:-1], device=z0.device, dtype=z0.dtype)
        x_mu, x_std = self.decoder(zK)               # (B, T, F)
        return {
            'h': h, 'z0': z0, 'mu_q': mu_q, 'std_q': std_q,
            'zK': zK, 'sum_log_det': sum_log_det,
            'x_mu': x_mu, 'x_std': x_std,
        }

    def forward(self, x: torch.Tensor):
        """Convenience forward returning x_mu, x_std plus posterior trace."""
        out = self._sample_once(x)
        return out['x_mu'], out['x_std'], out

    def compute_loss(self, x: torch.Tensor, beta: float = 1.0, n_z: int = 1):
        """Full SGVB ELBO with NF log-det correction and LGSSM prior.

        L = -E_q[ log p(x|z_K) ] + beta * E_q[ log q(z_0|x) - sum_log_det - log p_LGSSM(z_K) ]
        """
        # MC-average over n_z latent samples (n_z=1 by default, matching reference)
        recon_terms = []
        kl_terms = []
        last_info = {}
        for _ in range(max(1, int(n_z))):
            out = self._sample_once(x)
            # Reconstruction log-likelihood per element, summed over feature dim
            x_log_prob = _gaussian_log_prob(x, out['x_mu'], out['x_std'])  # (B, T, F)
            recon_per_t = x_log_prob.sum(dim=-1)  # (B, T)
            recon_terms.append(recon_per_t)

            # log q(z_0 | x): per-step per-dim Gaussian under (mu_q, std_q)
            log_q0 = _gaussian_log_prob(out['z0'], out['mu_q'], out['std_q'])  # (B, T, z)
            log_q0_per_t = log_q0.sum(dim=-1)  # (B, T)
            # NF change-of-variable: log q(z_K) = log q(z_0) - sum_log_det
            log_qK_per_t = log_q0_per_t - out['sum_log_det']  # (B, T)
            # log p_LGSSM(z_K)
            log_p_per_t = LGSSMPrior.log_prob(out['zK']).sum(dim=-1)  # (B, T)
            kl_per_t = log_qK_per_t - log_p_per_t  # (B, T) — MC estimate of KL(q||p)
            kl_terms.append(kl_per_t)
            last_info = out

        recon_per_t = torch.stack(recon_terms, dim=0).mean(dim=0)  # (B, T)
        kl_per_t = torch.stack(kl_terms, dim=0).mean(dim=0)        # (B, T)

        # Negative ELBO (loss to minimize), averaged per (window, step)
        recon_nll = -recon_per_t.mean()
        kl = kl_per_t.mean()
        loss = recon_nll + beta * kl
        info = {
            'recon_nll': float(recon_nll.detach().cpu().item()),
            'kl': float(kl.detach().cpu().item()),
        }
        return loss, info

    @torch.no_grad()
    def get_anomaly_score(self, x: torch.Tensor, n_mc_samples: int = 1) -> torch.Tensor:
        """Per-window per-timestep anomaly score (higher = more anomalous).

        score_t = -E_q[ log p(x_t | z_K) ] summed over feature dim, MC-averaged.
        """
        scores_list = []
        for _ in range(max(1, int(n_mc_samples))):
            out = self._sample_once(x)
            log_p_x = _gaussian_log_prob(x, out['x_mu'], out['x_std']).sum(dim=-1)  # (B, T)
            scores_list.append(-log_p_x)
        return torch.stack(scores_list, dim=0).mean(dim=0)  # (B, T)


# ---------------------------------------------------------------------------
# Dataset (lazy sliding window)
# ---------------------------------------------------------------------------


class SlidingWindowDataset(torch.utils.data.Dataset):
    """Lazy sliding window dataset — windows generated on-the-fly."""

    def __init__(self, data: np.ndarray, seq_len: int, stride: int = 1):
        self.data = data
        self.seq_len = seq_len
        self.stride = stride
        self.n_windows = max(0, (len(data) - seq_len) // stride + 1)

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        start = idx * self.stride
        return torch.from_numpy(
            self.data[start:start + self.seq_len].copy()
        ).float()


# ---------------------------------------------------------------------------
# Wrapper (Path A — driver normalizes input)
# ---------------------------------------------------------------------------


class OmniAnomalyBaseline:
    """OmniAnomaly baseline wrapper compatible with ./comparison/ pipeline.

    Public interface (Domain B contract):
      - fit(train_X: (T, F), epoch_callback=None) -> self
      - predict(test_X: (T, F)) -> (T,) float32
      - save(dir), load(dir)
      - name property
    """

    def __init__(
        self,
        seq_len: int = 100,
        hidden_dim: int = 500,
        dense_dim: int = 500,
        z_dim: int = 3,
        nf_layers: int = 20,
        std_epsilon: float = 1e-4,
        n_layers: int = 1,
        rnn_type: str = 'gru',
        beta: float = 1.0,
        n_mc_samples: int = 1,
        lr: float = 0.001,
        l2_reg: float = 1e-4,
        batch_size: int = 50,
        epochs: int = 10,
        train_stride: int = 1,
        device: Optional[str] = None,
        verbose: bool = True,
        # Back-compat: accept and ignore legacy kwargs from older presets / saved configs
        use_connected_z_q: Optional[bool] = None,
        use_connected_z_p: Optional[bool] = None,
        **_legacy_ignored,
    ):
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.dense_dim = dense_dim
        self.z_dim = z_dim
        self.nf_layers = nf_layers
        self.std_epsilon = std_epsilon
        self.n_layers = n_layers
        self.rnn_type = rnn_type
        self.beta = beta
        self.n_mc_samples = n_mc_samples
        self.lr = lr
        self.l2_reg = l2_reg
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_stride = train_stride
        self.verbose = verbose
        # Kept for save/load round-trip; reference defaults are both True.
        self.use_connected_z_q = True if use_connected_z_q is None else use_connected_z_q
        self.use_connected_z_p = True if use_connected_z_p is None else use_connected_z_p

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model: Optional[OmniAnomalyModel] = None
        self.n_features: int = 0
        self.train_loss_history = []

    @property
    def name(self) -> str:
        return "OmniAnomaly"

    def _build_model(self, n_features: int) -> OmniAnomalyModel:
        return OmniAnomalyModel(
            input_dim=n_features,
            rnn_hidden=self.hidden_dim,
            dense_dim=self.dense_dim,
            z_dim=self.z_dim,
            nf_layers=self.nf_layers,
            std_epsilon=self.std_epsilon,
            n_layers=self.n_layers,
            rnn_type=self.rnn_type,
        ).to(self.device)

    def fit(self, train_data: np.ndarray, epoch_callback=None, train_segments=None) -> 'OmniAnomalyBaseline':
        """Train OmniAnomaly on training data following the SOTA epoch_callback contract."""
        self.n_features = train_data.shape[1]
        self.model = self._build_model(self.n_features)

        if self.verbose:
            print(f"{self.name} training:")
            print(f"  Device: {self.device}")
            print(f"  Input shape: ({self.seq_len}, {self.n_features})")
            print(f"  rnn_hidden={self.hidden_dim}, dense_dim={self.dense_dim}, "
                  f"z_dim={self.z_dim}, nf_layers={self.nf_layers}")
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"  Total parameters: {total_params:,}")

        dataset = SlidingWindowDataset(train_data, self.seq_len, stride=self.train_stride)
        if train_segments is not None:
            valid_idx = compute_segment_safe_window_indices(
                train_segments, self.seq_len, self.train_stride, len(dataset),
            )
            if self.verbose:
                print(f"  Segment-aware: kept {len(valid_idx)}/{len(dataset)} windows "
                      f"({len(valid_idx)/max(len(dataset),1):.1%}) — dropped boundary-crossing")
            dataset = torch.utils.data.Subset(dataset, valid_idx.tolist())
        if self.verbose:
            print(f"  Training windows: {len(dataset):,} (stride={self.train_stride})")

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.l2_reg
        )

        self.train_loss_history = []
        self.model.train()
        start_time = time.time()

        n_batches = len(dataloader)
        for epoch in range(self.epochs):
            epoch_start = time.time()
            epoch_losses = []
            for batch_idx, batch_x in enumerate(dataloader):
                batch_start = time.time()
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                loss, _ = self.model.compute_loss(
                    batch_x, beta=self.beta, n_z=1
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                optimizer.step()
                epoch_losses.append(loss.item())
                print(f"[BATCH] model={self.name} epoch={epoch + 1}/{self.epochs} batch={batch_idx + 1}/{n_batches} time={time.time() - batch_start:.3f}s", flush=True)

            avg_loss = float(np.mean(epoch_losses)) if epoch_losses else float('nan')
            self.train_loss_history.append(avg_loss)

            epoch_time = time.time() - epoch_start
            elapsed = time.time() - start_time
            remaining = elapsed / (epoch + 1) * (self.epochs - epoch - 1)

            if self.verbose:
                progress = (epoch + 1) / self.epochs * 100
                print(f"\r  Training: [{epoch + 1}/{self.epochs}] {progress:5.1f}% | "
                      f"Loss: {avg_loss:.6f} | "
                      f"Time: {epoch_time:.1f}s/epoch | "
                      f"ETA: {remaining:.0f}s", end="")
                sys.stdout.flush()

            if epoch_callback is not None:
                epoch_callback(self, epoch + 1)
                self.model.train()

        if self.verbose and self.train_loss_history:
            total_time = time.time() - start_time
            print(f"\n  Training complete: {total_time:.1f}s total, "
                  f"Final loss: {self.train_loss_history[-1]:.6f}")

        return self

    def _raw_scores_one_entity(self, sub_data: np.ndarray) -> np.ndarray:
        """RAW per-timestep scores for ONE entity's test slice, no cross-entity coupling.

        This is the model's complete RAW score producer: sliding-window (stride=1) over
        ``sub_data`` ONLY, model inference, last-position-in-window assignment, and head fill.
        Called in isolation per entity by ``per_entity_concat`` so no window can span an
        entity boundary. Returns ``(len(sub_data),)`` float32.

        Edge-safe: if the slice is shorter than ``seq_len`` (n_windows == 0), no window can
        be formed for this entity, so we return all-zeros for the slice (finite, non-crashing
        — identical to the legacy whole-array behaviour when the entire test set is too short).
        OmniAnomaly applies NO whole-test post-processing, so the concatenation of these raw
        per-entity scores IS the final score (post-proc granularity unchanged == none).
        """
        sub_data = np.asarray(sub_data, dtype=np.float32)
        n_samples, n_features = sub_data.shape
        n_windows = max(0, n_samples - self.seq_len + 1)

        all_scores = []
        n_batches = (n_windows + self.batch_size - 1) // self.batch_size

        with torch.no_grad():
            for batch_idx in range(n_batches):
                batch_start = batch_idx * self.batch_size
                batch_end = min(batch_start + self.batch_size, n_windows)
                actual_bs = batch_end - batch_start
                batch_windows = np.zeros(
                    (actual_bs, self.seq_len, n_features), dtype=np.float32
                )
                for j, w_idx in enumerate(range(batch_start, batch_end)):
                    batch_windows[j] = sub_data[w_idx:w_idx + self.seq_len]

                batch_x = torch.from_numpy(batch_windows).to(self.device)
                window_scores = self.model.get_anomaly_score(
                    batch_x, n_mc_samples=self.n_mc_samples
                )  # (B, T)
                last_pos = window_scores[:, -1].cpu().numpy()
                all_scores.append(last_pos)

        scores = np.zeros(n_samples, dtype=np.float32)
        if all_scores:
            window_scores_all = np.concatenate(all_scores).astype(np.float32)
            for i, score in enumerate(window_scores_all):
                target_idx = i + self.seq_len - 1
                if target_idx < n_samples:
                    scores[target_idx] = score
            # Forward-fill first (seq_len-1) positions with first window's score so the
            # output length contract is preserved for this entity slice.
            for i in range(min(self.seq_len - 1, n_samples)):
                scores[i] = window_scores_all[0]

        return scores

    def predict(self, test_data: np.ndarray, test_segments=None) -> np.ndarray:
        """Compute (T_test,) float32 anomaly scores via last-position-in-window scoring.

        Boundary-safe windowing (multi-file datasets): the RAW score producer
        (sliding-window + model inference + last-position mapping + head fill) is run
        INDEPENDENTLY on each entity's own test slice via ``per_entity_concat``, so no
        window straddles an entity boundary (e.g. SMD machine-k tail + machine-(k+1) head).
        ``test_segments`` are the per-entity TEST-LOCAL (lo, hi) slices from the SAME source
        as per-entity normalization (``loader.get_file_norm_segments()`` test side).
        ``None`` / single-entity / non-tiling -> ONE call over the whole array == EXACT
        legacy behaviour (bit-identical NO-OP). OmniAnomaly applies no whole-test
        post-processing, so concatenated raw scores are the final output unchanged.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        self.model.eval()
        test_data = np.asarray(test_data, dtype=np.float32)

        if self.verbose:
            n_windows_total = max(0, len(test_data) - self.seq_len + 1)
            print(f"  Inference: {n_windows_total:,} windows (boundary-safe per-entity)")
        start_time = time.time()

        scores = per_entity_concat(test_data, test_segments, self._raw_scores_one_entity)

        if self.verbose:
            total_time = time.time() - start_time
            print(f"\n  Inference complete: {total_time:.1f}s")

        return scores.astype(np.float32)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _config_dict(self) -> dict:
        return {
            "seq_len": self.seq_len,
            "hidden_dim": self.hidden_dim,
            "dense_dim": self.dense_dim,
            "z_dim": self.z_dim,
            "nf_layers": self.nf_layers,
            "std_epsilon": self.std_epsilon,
            "n_layers": self.n_layers,
            "rnn_type": self.rnn_type,
            "beta": self.beta,
            "n_mc_samples": self.n_mc_samples,
            "lr": self.lr,
            "l2_reg": self.l2_reg,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "train_stride": self.train_stride,
            "n_features": self.n_features,
            "use_connected_z_q": self.use_connected_z_q,
            "use_connected_z_p": self.use_connected_z_p,
            "model_name": self.name,
        }

    def save(self, save_dir) -> None:
        if self.model is None:
            raise RuntimeError("No model to save.")
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), save_dir / "model.pt")
        with open(save_dir / "config.json", 'w') as f:
            json.dump(self._config_dict(), f, indent=2)

    def load(self, save_dir) -> 'OmniAnomalyBaseline':
        save_dir = Path(save_dir)
        with open(save_dir / "config.json", 'r') as f:
            config = json.load(f)
        for key in ("seq_len", "hidden_dim", "dense_dim", "z_dim", "nf_layers",
                    "std_epsilon", "n_layers", "rnn_type", "beta", "n_mc_samples",
                    "lr", "l2_reg", "batch_size", "epochs", "train_stride",
                    "n_features"):
            if key in config:
                setattr(self, key, config[key])
        self.use_connected_z_q = config.get("use_connected_z_q", True)
        self.use_connected_z_p = config.get("use_connected_z_p", True)
        self.model = self._build_model(self.n_features)
        self.model.load_state_dict(
            torch.load(save_dir / "model.pt", map_location=self.device)
        )
        self.model.eval()
        return self

    def __repr__(self) -> str:
        return (f"OmniAnomalyBaseline(seq_len={self.seq_len}, hidden_dim={self.hidden_dim}, "
                f"dense_dim={self.dense_dim}, z_dim={self.z_dim}, nf_layers={self.nf_layers})")
