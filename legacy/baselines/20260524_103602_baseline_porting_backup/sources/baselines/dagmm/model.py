"""
DAGMM - Deep Autoencoding Gaussian Mixture Model

Based on: "Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection"
Paper: ICLR 2018, https://openreview.net/forum?id=BJJLHbb0-
Reference: https://github.com/danieltan07/dagmm

Architecture:
1. Compression Network: Autoencoder that learns compressed representation
2. Estimation Network: Predicts GMM membership from latent + reconstruction features
3. GMM: Gaussian Mixture Model for density estimation

Anomaly Score: Negative log probability under the GMM (energy)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple
from pathlib import Path
import json
import time
import sys


class CompressionNetwork(nn.Module):
    """
    Autoencoder for dimensionality reduction.

    Returns:
    - Compressed latent representation (z_c)
    - Reconstruction
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
        super().__init__()

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.Tanh(),
            ])
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.Tanh(),
            ])
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_c = self.encoder(x)
        x_hat = self.decoder(z_c)
        return z_c, x_hat


class EstimationNetwork(nn.Module):
    """
    Network to estimate GMM component membership probabilities (gamma).

    Input: concatenated [z_c, z_r] where z_r includes reconstruction features
    Output: soft membership probabilities
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], n_gmm: int):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.Tanh(),
                nn.Dropout(0.5),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, n_gmm))
        layers.append(nn.Softmax(dim=-1))

        self.network = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.network(z)


class DAGMMModel(nn.Module):
    """
    DAGMM Model combining compression network, estimation network, and GMM.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        latent_dim: int = 1,
        n_gmm: int = 2,
        lambda_energy: float = 0.1,
        lambda_cov: float = 0.005
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [60, 30, 10]

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_gmm = n_gmm
        self.lambda_energy = lambda_energy
        self.lambda_cov = lambda_cov

        # Compression network (autoencoder)
        self.compression_net = CompressionNetwork(input_dim, hidden_dims, latent_dim)

        # z dimension = latent_dim + 2 (reconstruction features: relative euclidean, cosine similarity)
        z_dim = latent_dim + 2

        # Estimation network
        self.estimation_net = EstimationNetwork(z_dim, [10], n_gmm)

        # GMM parameters (will be computed from data)
        self.register_buffer('phi', torch.zeros(n_gmm))
        self.register_buffer('mu', torch.zeros(n_gmm, z_dim))
        self.register_buffer('cov', torch.zeros(n_gmm, z_dim, z_dim))

    def compute_reconstruction_features(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction features z_r:
        - Relative Euclidean distance
        - Cosine similarity
        """
        # Relative Euclidean distance
        euclidean = torch.sqrt(torch.sum((x - x_hat) ** 2, dim=-1, keepdim=True))
        norm_x = torch.sqrt(torch.sum(x ** 2, dim=-1, keepdim=True)) + 1e-10
        relative_euclidean = euclidean / norm_x

        # Cosine similarity
        cos_sim = F.cosine_similarity(x, x_hat, dim=-1).unsqueeze(-1)

        return torch.cat([relative_euclidean, cos_sim], dim=-1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            z_c: Compressed latent
            x_hat: Reconstruction
            z: Full latent (z_c + reconstruction features)
            gamma: GMM membership probabilities
        """
        # Compression
        z_c, x_hat = self.compression_net(x)

        # Reconstruction features
        z_r = self.compute_reconstruction_features(x, x_hat)

        # Full latent representation
        z = torch.cat([z_c, z_r], dim=-1)

        # Estimate GMM membership
        gamma = self.estimation_net(z)

        return z_c, x_hat, z, gamma

    def compute_gmm_params(self, z: torch.Tensor, gamma: torch.Tensor) -> None:
        """
        Compute GMM parameters (phi, mu, cov) from data.
        """
        n_samples = z.size(0)

        # Mixture weights
        sum_gamma = torch.sum(gamma, dim=0)
        self.phi = sum_gamma / n_samples

        # Means
        self.mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / (sum_gamma.unsqueeze(-1) + 1e-10)

        # Covariances
        z_centered = z.unsqueeze(1) - self.mu.unsqueeze(0)  # (N, K, D)
        self.cov = torch.sum(
            gamma.unsqueeze(-1).unsqueeze(-1) * z_centered.unsqueeze(-1) * z_centered.unsqueeze(-2),
            dim=0
        ) / (sum_gamma.unsqueeze(-1).unsqueeze(-1) + 1e-10)

        # Add small diagonal for numerical stability
        self.cov = self.cov + 1e-6 * torch.eye(z.size(-1), device=z.device).unsqueeze(0)

    def compute_energy(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute sample energy (negative log probability under GMM).
        Lower energy = higher probability = more normal.
        """
        z_dim = z.size(-1)

        # Compute energy for each component
        energies = []
        for k in range(self.n_gmm):
            # Centered samples
            z_centered = z - self.mu[k]

            # Inverse covariance
            try:
                cov_inv = torch.inverse(self.cov[k])
            except RuntimeError:
                # Add more regularization if singular
                cov_reg = self.cov[k] + 0.01 * torch.eye(z_dim, device=z.device)
                cov_inv = torch.inverse(cov_reg)

            # Log determinant
            try:
                log_det = torch.logdet(self.cov[k])
            except RuntimeError:
                log_det = torch.tensor(0.0, device=z.device)

            # Mahalanobis distance
            mahal = torch.sum(z_centered @ cov_inv * z_centered, dim=-1)

            # Component energy
            energy_k = 0.5 * (z_dim * np.log(2 * np.pi) + log_det + mahal)
            energies.append(energy_k)

        energies = torch.stack(energies, dim=-1)  # (N, K)

        # Weighted energy (using mixture weights)
        weighted_energies = energies - torch.log(self.phi + 1e-10).unsqueeze(0)

        # Log-sum-exp for numerical stability
        min_energy = torch.min(weighted_energies, dim=-1, keepdim=True)[0]
        energy = -torch.log(torch.sum(torch.exp(-weighted_energies + min_energy), dim=-1) + 1e-10) + min_energy.squeeze(-1)

        return energy

    def loss_function(self, x: torch.Tensor, x_hat: torch.Tensor, z: torch.Tensor, gamma: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute DAGMM loss.

        Returns:
            loss: Total loss
            losses: Dictionary of individual losses
        """
        # Reconstruction loss
        recon_loss = torch.mean(torch.sum((x - x_hat) ** 2, dim=-1))

        # Update GMM parameters
        self.compute_gmm_params(z, gamma)

        # Energy loss
        energy = self.compute_energy(z)
        energy_loss = torch.mean(energy)

        # Covariance regularization (penalty for small diagonal elements)
        cov_diag = torch.diagonal(self.cov, dim1=-2, dim2=-1)
        cov_loss = torch.mean(1.0 / (cov_diag + 1e-10))

        # Total loss
        loss = recon_loss + self.lambda_energy * energy_loss + self.lambda_cov * cov_loss

        return loss, {
            'recon_loss': recon_loss.item(),
            'energy_loss': energy_loss.item(),
            'cov_loss': cov_loss.item(),
        }


class DAGMMBaseline:
    """
    DAGMM baseline wrapper for time series anomaly detection.
    """

    def __init__(
        self,
        seq_len: int = 5,
        hidden_dims: List[int] = None,
        latent_dim: int = 1,
        n_gmm: int = 2,
        lambda_energy: float = 0.1,
        lambda_cov: float = 0.005,
        lr: float = 0.0001,
        batch_size: int = 256,
        epochs: int = 10,
        train_stride: int = 3,
        device: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Args:
            seq_len: Window size
            hidden_dims: Hidden layer dimensions for autoencoder
            latent_dim: Dimension of compressed latent space
            n_gmm: Number of GMM components
            lambda_energy: Weight for energy loss
            lambda_cov: Weight for covariance regularization
            lr: Learning rate
            batch_size: Training batch size
            epochs: Number of training epochs
            train_stride: Stride for training windows (default 11)
            device: 'cuda' or 'cpu'
            verbose: Print training progress
        """
        self.seq_len = seq_len
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.n_gmm = n_gmm
        self.lambda_energy = lambda_energy
        self.lambda_cov = lambda_cov
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_stride = train_stride
        self.verbose = verbose

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model: Optional[DAGMMModel] = None
        self.n_features: int = 0
        self.input_dim: int = 0
        self.train_loss_history = []

    @property
    def name(self) -> str:
        return "DAGMM"

    def _create_windows(self, data: np.ndarray, stride: int = 1) -> np.ndarray:
        """Create sliding windows and flatten."""
        n_samples = len(data)
        n_windows = (n_samples - self.seq_len) // stride + 1

        windows = np.zeros((n_windows, self.seq_len * data.shape[1]), dtype=np.float32)
        for idx, i in enumerate(range(0, n_samples - self.seq_len + 1, stride)):
            if idx >= n_windows:
                break
            windows[idx] = data[i:i + self.seq_len].flatten()

        return windows

    def _update_gmm_params(self, windows: np.ndarray):
        """Compute GMM parameters from training windows (needed before predict)."""
        import torch
        self.model.eval()
        with torch.no_grad():
            all_z = []
            all_gamma = []
            for i in range(0, len(windows), self.batch_size):
                batch_x = torch.FloatTensor(windows[i:i + self.batch_size]).to(self.device)
                _, _, z, gamma = self.model(batch_x)
                all_z.append(z)
                all_gamma.append(gamma)
            all_z = torch.cat(all_z, dim=0)
            all_gamma = torch.cat(all_gamma, dim=0)
            self.model.compute_gmm_params(all_z, all_gamma)

    def fit(self, train_data: np.ndarray, epoch_callback=None) -> 'DAGMMBaseline':
        """
        Train DAGMM on training data.

        Args:
            train_data: Training data (n_samples, n_features)
            epoch_callback: Optional callback(model, epoch) called after each epoch

        Returns:
            self
        """
        self.n_features = train_data.shape[1]
        self.input_dim = self.seq_len * self.n_features

        # Set hidden dims if not specified
        if self.hidden_dims is None:
            self.hidden_dims = [self.input_dim // 2, self.input_dim // 4, 10]

        # Build model
        self.model = DAGMMModel(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            latent_dim=self.latent_dim,
            n_gmm=self.n_gmm,
            lambda_energy=self.lambda_energy,
            lambda_cov=self.lambda_cov
        ).to(self.device)

        if self.verbose:
            print(f"{self.name} training:")
            print(f"  Device: {self.device}")
            print(f"  Input dim: {self.input_dim} (seq_len={self.seq_len} × n_features={self.n_features})")
            print(f"  Latent dim: {self.latent_dim}, GMM components: {self.n_gmm}")
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"  Total parameters: {total_params:,}")

        # Create windows with stride for training
        windows = self._create_windows(train_data, stride=self.train_stride)
        if self.verbose:
            print(f"  Training windows: {len(windows):,} (stride={self.train_stride})")

        # Create DataLoader
        dataset = torch.utils.data.TensorDataset(torch.FloatTensor(windows))
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )

        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.train_loss_history = []
        self.model.train()

        start_time = time.time()

        for epoch in range(self.epochs):
            epoch_start = time.time()
            epoch_losses = []

            for batch_idx, (batch_x,) in enumerate(dataloader):
                batch_x = batch_x.to(self.device)

                optimizer.zero_grad()

                z_c, x_hat, z, gamma = self.model(batch_x)
                loss, loss_dict = self.model.loss_function(batch_x, x_hat, z, gamma)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
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
                # DAGMM needs GMM params before predict can work
                self._update_gmm_params(windows)
                epoch_callback(self, epoch + 1)
                self.model.train()

        if self.verbose:
            total_time = time.time() - start_time
            print(f"\n  Training complete: {total_time:.1f}s total, Final loss: {self.train_loss_history[-1]:.6f}")

        # Final GMM parameter estimation on full training data
        self._update_gmm_params(windows)

        return self

    def predict(self, test_data: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores for test data.

        Args:
            test_data: Test data (n_samples, n_features)

        Returns:
            Anomaly scores for each timestamp (energy = negative log probability)
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        self.model.eval()

        # Create windows
        windows = self._create_windows(test_data)
        n_windows = len(windows)

        # Compute scores in batches
        all_scores = []
        n_batches = (n_windows + self.batch_size - 1) // self.batch_size
        start_time = time.time()

        if self.verbose:
            print(f"  Inference: {n_windows:,} windows in {n_batches} batches")

        with torch.no_grad():
            for batch_idx, i in enumerate(range(0, n_windows, self.batch_size)):
                batch_x = torch.FloatTensor(windows[i:i + self.batch_size]).to(self.device)

                _, _, z, _ = self.model(batch_x)
                energy = self.model.compute_energy(z)

                all_scores.append(energy.cpu().numpy())

                if self.verbose and (batch_idx + 1) % max(1, n_batches // 10) == 0:
                    progress = (batch_idx + 1) / n_batches * 100
                    elapsed = time.time() - start_time
                    print(f"\r  Inference: [{batch_idx + 1}/{n_batches}] {progress:5.1f}% | "
                          f"Elapsed: {elapsed:.1f}s", end="")
                    sys.stdout.flush()

        if self.verbose:
            total_time = time.time() - start_time
            print(f"\n  Inference complete: {total_time:.1f}s")

        window_scores = np.concatenate(all_scores)

        # Map window scores to point-level
        n_samples = len(test_data)
        scores = np.zeros(n_samples, dtype=np.float32)

        for i, score in enumerate(window_scores):
            target_idx = i + self.seq_len - 1
            if target_idx < n_samples:
                scores[target_idx] = score

        # Fill first timestamps
        if len(window_scores) > 0:
            for i in range(min(self.seq_len - 1, n_samples)):
                scores[i] = window_scores[0]

        return scores

    def save(self, save_dir: Path) -> None:
        """Save model weights and config."""
        if self.model is None:
            raise RuntimeError("No model to save.")

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), save_dir / "model.pt")

        config = {
            "seq_len": self.seq_len,
            "hidden_dims": self.hidden_dims,
            "latent_dim": self.latent_dim,
            "n_gmm": self.n_gmm,
            "n_features": self.n_features,
            "input_dim": self.input_dim,
            "model_name": self.name,
        }
        with open(save_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

    def load(self, save_dir: Path) -> 'DAGMMBaseline':
        """Load model weights and config."""
        save_dir = Path(save_dir)

        with open(save_dir / "config.json", 'r') as f:
            config = json.load(f)

        self.seq_len = config["seq_len"]
        self.hidden_dims = config["hidden_dims"]
        self.latent_dim = config["latent_dim"]
        self.n_gmm = config["n_gmm"]
        self.n_features = config["n_features"]
        self.input_dim = config["input_dim"]

        self.model = DAGMMModel(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            latent_dim=self.latent_dim,
            n_gmm=self.n_gmm,
        ).to(self.device)
        self.model.load_state_dict(torch.load(save_dir / "model.pt", map_location=self.device))
        self.model.eval()

        return self

    def __repr__(self) -> str:
        return f"DAGMMBaseline(seq_len={self.seq_len}, latent_dim={self.latent_dim}, n_gmm={self.n_gmm})"
