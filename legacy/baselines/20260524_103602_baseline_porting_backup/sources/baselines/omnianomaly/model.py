"""
OmniAnomaly - Stochastic Recurrent Neural Network for Multivariate Time Series Anomaly Detection

Based on: "Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network"
Paper: KDD 2019, https://dl.acm.org/doi/10.1145/3292500.3330672
Reference: https://github.com/NetManAIOps/OmniAnomaly

Architecture:
1. RNN Encoder: GRU/LSTM that encodes input sequence
2. Variational Inference: Learn posterior q(z|x)
3. RNN Decoder: Generate reconstruction from latent z
4. Anomaly Score: Negative log-likelihood (reconstruction probability)

Key Features:
- Stochastic latent variables for uncertainty
- Connected z over time (optional)
- Uses reconstruction probability as anomaly score
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
from pathlib import Path
import json
import time
import sys


class Encoder(nn.Module):
    """RNN Encoder for OmniAnomaly."""

    def __init__(self, input_dim: int, hidden_dim: int, z_dim: int,
                 n_layers: int = 1, rnn_type: str = 'gru'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        if rnn_type == 'gru':
            self.rnn = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        else:
            self.rnn = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        hidden, _ = self.rnn(x)
        return self.fc_mu(hidden), self.fc_logvar(hidden), hidden


class Decoder(nn.Module):
    """RNN Decoder for OmniAnomaly."""

    def __init__(self, z_dim: int, hidden_dim: int, output_dim: int,
                 n_layers: int = 1, rnn_type: str = 'gru'):
        super().__init__()
        self.hidden_dim = hidden_dim
        if rnn_type == 'gru':
            self.rnn = nn.GRU(z_dim, hidden_dim, n_layers, batch_first=True)
        else:
            self.rnn = nn.LSTM(z_dim, hidden_dim, n_layers, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, output_dim)
        self.fc_logvar = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        hidden, _ = self.rnn(z)
        return self.fc_mu(hidden), self.fc_logvar(hidden)


class OmniAnomalyModel(nn.Module):
    """OmniAnomaly VAE Model."""

    def __init__(self, input_dim: int, hidden_dim: int = 100, z_dim: int = 3,
                 n_layers: int = 1, rnn_type: str = 'gru'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.encoder = Encoder(input_dim, hidden_dim, z_dim, n_layers, rnn_type)
        self.decoder = Decoder(z_dim, hidden_dim, input_dim, n_layers, rnn_type)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        z_mu, z_logvar, _ = self.encoder(x)
        z = self.reparameterize(z_mu, z_logvar)
        x_mu, x_logvar = self.decoder(z)
        return x_mu, x_logvar, z_mu, z_logvar, z

    def compute_loss(self, x, x_mu, x_logvar, z_mu, z_logvar, beta=1.0):
        recon_loss = 0.5 * torch.mean(
            x_logvar + (x - x_mu) ** 2 / (torch.exp(x_logvar) + 1e-8)
        )
        kl_loss = -0.5 * torch.mean(1 + z_logvar - z_mu ** 2 - torch.exp(z_logvar))
        loss = recon_loss + beta * kl_loss
        return loss, {'recon_loss': recon_loss.item(), 'kl_loss': kl_loss.item()}

    def get_anomaly_score(self, x, n_samples=1):
        scores_list = []
        for _ in range(n_samples):
            x_mu, x_logvar, _, _, _ = self.forward(x)
            nll = 0.5 * (x_logvar + (x - x_mu) ** 2 / (torch.exp(x_logvar) + 1e-8))
            nll = torch.sum(nll, dim=-1)
            scores_list.append(nll)
        return torch.stack(scores_list, dim=0).mean(dim=0)


class SlidingWindowDataset(torch.utils.data.Dataset):
    """Lazy sliding window dataset — creates windows on-the-fly, no bulk allocation."""

    def __init__(self, data: np.ndarray, seq_len: int, stride: int = 1):
        self.data = data
        self.seq_len = seq_len
        self.stride = stride
        self.n_windows = (len(data) - seq_len) // stride + 1

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        start = idx * self.stride
        return torch.from_numpy(
            self.data[start:start + self.seq_len].copy()
        ).float()


class OmniAnomalyBaseline:
    """OmniAnomaly baseline wrapper for time series anomaly detection."""

    def __init__(
        self,
        seq_len: int = 100,
        hidden_dim: int = 100,
        z_dim: int = 3,
        n_layers: int = 1,
        rnn_type: str = 'gru',
        beta: float = 0.01,
        n_mc_samples: int = 10,
        lr: float = 0.001,
        batch_size: int = 256,
        epochs: int = 10,
        train_stride: int = 3,
        device: Optional[str] = None,
        verbose: bool = True
    ):
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.rnn_type = rnn_type
        self.beta = beta
        self.n_mc_samples = n_mc_samples
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_stride = train_stride
        self.verbose = verbose

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

    def fit(self, train_data: np.ndarray, epoch_callback=None) -> 'OmniAnomalyBaseline':
        """Train OmniAnomaly on training data."""
        self.n_features = train_data.shape[1]

        self.model = OmniAnomalyModel(
            input_dim=self.n_features,
            hidden_dim=self.hidden_dim,
            z_dim=self.z_dim,
            n_layers=self.n_layers,
            rnn_type=self.rnn_type
        ).to(self.device)

        if self.verbose:
            print(f"{self.name} training:")
            print(f"  Device: {self.device}")
            print(f"  Input shape: ({self.seq_len}, {self.n_features})")
            print(f"  Hidden dim: {self.hidden_dim}, z_dim: {self.z_dim}")
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"  Total parameters: {total_params:,}")

        # Lazy sliding window dataset
        dataset = SlidingWindowDataset(train_data, self.seq_len, stride=self.train_stride)
        if self.verbose:
            print(f"  Training windows: {len(dataset):,} (stride={self.train_stride})")

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.train_loss_history = []
        self.model.train()
        start_time = time.time()

        for epoch in range(self.epochs):
            epoch_start = time.time()
            epoch_losses = []

            for batch_idx, batch_x in enumerate(dataloader):
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()

                x_mu, x_logvar, z_mu, z_logvar, _ = self.model(batch_x)
                loss, loss_dict = self.model.compute_loss(
                    batch_x, x_mu, x_logvar, z_mu, z_logvar, self.beta
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                optimizer.step()
                epoch_losses.append(loss.item())

            avg_loss = np.mean(epoch_losses)
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

        if self.verbose:
            total_time = time.time() - start_time
            print(f"\n  Training complete: {total_time:.1f}s total, Final loss: {self.train_loss_history[-1]:.6f}")

        return self

    def predict(self, test_data: np.ndarray) -> np.ndarray:
        """Compute anomaly scores with batched on-the-fly windowing."""
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        self.model.eval()

        n_samples, n_features = test_data.shape
        n_windows = n_samples - self.seq_len + 1

        all_scores = []
        n_batches = (n_windows + self.batch_size - 1) // self.batch_size
        start_time = time.time()

        if self.verbose:
            print(f"  Inference: {n_windows:,} windows in {n_batches} batches")

        with torch.no_grad():
            for batch_idx in range(n_batches):
                batch_start = batch_idx * self.batch_size
                batch_end = min(batch_start + self.batch_size, n_windows)
                actual_bs = batch_end - batch_start

                batch_windows = np.zeros(
                    (actual_bs, self.seq_len, n_features), dtype=np.float32
                )
                for j, w_idx in enumerate(range(batch_start, batch_end)):
                    batch_windows[j] = test_data[w_idx:w_idx + self.seq_len]

                batch_x = torch.FloatTensor(batch_windows).to(self.device)
                window_scores = self.model.get_anomaly_score(batch_x, self.n_mc_samples)
                scores = window_scores[:, -1].cpu().numpy()
                all_scores.append(scores)

                if self.verbose and (batch_idx + 1) % max(1, n_batches // 10) == 0:
                    progress = (batch_idx + 1) / n_batches * 100
                    elapsed = time.time() - start_time
                    print(f"\r  Inference: [{batch_idx + 1}/{n_batches}] {progress:5.1f}% | "
                          f"Elapsed: {elapsed:.1f}s", end="")
                    sys.stdout.flush()

        if self.verbose:
            total_time = time.time() - start_time
            print(f"\n  Inference complete: {total_time:.1f}s")

        window_scores_all = np.concatenate(all_scores)

        scores = np.zeros(n_samples, dtype=np.float32)
        for i, score in enumerate(window_scores_all):
            target_idx = i + self.seq_len - 1
            if target_idx < n_samples:
                scores[target_idx] = score

        if len(window_scores_all) > 0:
            for i in range(min(self.seq_len - 1, n_samples)):
                scores[i] = window_scores_all[0]

        return scores

    def save(self, save_dir: Path) -> None:
        if self.model is None:
            raise RuntimeError("No model to save.")
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), save_dir / "model.pt")
        config = {
            "seq_len": self.seq_len, "hidden_dim": self.hidden_dim,
            "z_dim": self.z_dim, "n_layers": self.n_layers,
            "rnn_type": self.rnn_type, "n_features": self.n_features,
            "model_name": self.name,
        }
        with open(save_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

    def load(self, save_dir: Path) -> 'OmniAnomalyBaseline':
        save_dir = Path(save_dir)
        with open(save_dir / "config.json", 'r') as f:
            config = json.load(f)
        self.seq_len = config["seq_len"]
        self.hidden_dim = config["hidden_dim"]
        self.z_dim = config["z_dim"]
        self.n_layers = config["n_layers"]
        self.rnn_type = config["rnn_type"]
        self.n_features = config["n_features"]
        self.model = OmniAnomalyModel(
            input_dim=self.n_features, hidden_dim=self.hidden_dim,
            z_dim=self.z_dim, n_layers=self.n_layers, rnn_type=self.rnn_type
        ).to(self.device)
        self.model.load_state_dict(torch.load(save_dir / "model.pt", map_location=self.device))
        self.model.eval()
        return self

    def __repr__(self) -> str:
        return f"OmniAnomalyBaseline(seq_len={self.seq_len}, hidden_dim={self.hidden_dim}, z_dim={self.z_dim})"
