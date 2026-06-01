"""
USAD - UnSupervised Anomaly Detection using Autoencoders
Based on: "USAD: UnSupervised Anomaly Detection on Multivariate Time Series"
Paper: KDD 2020, https://dl.acm.org/doi/10.1145/3394486.3403392
Reference: https://github.com/manigalati/usad

Architecture:
- Shared Encoder
- Two parallel Decoders
- Adversarial training: D2 tries to distinguish E(X) from E(D1(E(X)))
- Anomaly score: α * ||X - D1(E(X))||² + β * ||X - D2(E(D1(E(X))))||²
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List
from pathlib import Path
import json
import time
import sys


class Encoder(nn.Module):
    """Encoder network that compresses input to latent space."""

    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, latent_dim))
        layers.append(nn.ReLU(inplace=True))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class Decoder(nn.Module):
    """Decoder network that reconstructs from latent space."""

    def __init__(self, latent_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        # Use Tanh instead of Sigmoid - outputs in [-1, 1] range
        # Better suited for normalized data (StandardScaler gives mean~0, std~1)
        # Sigmoid was causing bias toward 0.3-0.5 range, harming anomaly detection
        layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class USADModel(nn.Module):
    """
    USAD Model with shared encoder and two decoders.

    Training alternates between:
    1. Minimizing AE loss for D1: ||X - D1(E(X))||²
    2. Minimizing adversarial loss for D2: ||X - D2(E(D1(E(X))))||²
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder1 = Decoder(latent_dim, hidden_dims, input_dim)
        self.decoder2 = Decoder(latent_dim, hidden_dims, input_dim)

    def forward(self, x):
        z = self.encoder(x)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        return w1, w2, w3

    def compute_loss(self, x, n_epoch: int, n_epochs: int):
        """
        Compute USAD loss.

        Args:
            x: Input batch
            n_epoch: Current epoch (1-indexed)
            n_epochs: Total epochs

        Returns:
            loss1, loss2: Losses for the two decoder paths
        """
        w1, w2, w3 = self.forward(x)

        # Loss weighting based on epoch
        alpha = 1 / n_epochs * n_epoch
        beta = 1 - alpha

        # AE1 loss: reconstruct from z
        loss1 = alpha * torch.mean((x - w1) ** 2) + beta * torch.mean((x - w3) ** 2)

        # AE2 loss: adversarial - minimize reconstruction, maximize D2(E(w1)) error
        loss2 = alpha * torch.mean((x - w2) ** 2) - beta * torch.mean((x - w3) ** 2)

        return loss1, loss2

    def get_anomaly_score(self, x, alpha: float = 0.5, beta: float = 0.5):
        """
        Compute anomaly score.

        Args:
            x: Input data
            alpha: Weight for D1 reconstruction error
            beta: Weight for D2(E(D1(E(x)))) reconstruction error

        Returns:
            Anomaly scores
        """
        w1, w2, w3 = self.forward(x)
        score = alpha * torch.mean((x - w1) ** 2, dim=1) + beta * torch.mean((x - w3) ** 2, dim=1)
        return score


class USADBaseline:
    """
    USAD baseline wrapper for time series anomaly detection.

    Uses sliding windows to create training samples, then computes
    reconstruction-based anomaly scores.
    """

    def __init__(
        self,
        seq_len: int = 5,
        hidden_dims: List[int] = None,
        latent_dim: int = 32,
        lr: float = 0.001,
        batch_size: int = 256,
        epochs: int = 10,
        train_stride: int = 3,
        alpha: float = 0.5,
        beta: float = 0.5,
        device: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Args:
            seq_len: Window size for input
            hidden_dims: Hidden layer dimensions (default: [seq_len*n_features//2])
            latent_dim: Latent space dimension
            lr: Learning rate
            batch_size: Training batch size
            epochs: Number of training epochs
            train_stride: Stride for training windows (default 11)
            alpha: Weight for D1 reconstruction in anomaly score
            beta: Weight for D2 reconstruction in anomaly score
            device: 'cuda' or 'cpu'
            verbose: Print training progress
        """
        self.seq_len = seq_len
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_stride = train_stride
        self.alpha = alpha
        self.beta = beta
        self.verbose = verbose

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model: Optional[USADModel] = None
        self.n_features: int = 0
        self.input_dim: int = 0
        self.train_loss_history = []

    @property
    def name(self) -> str:
        return "USAD"

    def _create_windows(self, data: np.ndarray, stride: int = 1) -> np.ndarray:
        """Create sliding windows and flatten for USAD input."""
        n_samples = len(data)
        n_windows = (n_samples - self.seq_len) // stride + 1

        windows = np.zeros((n_windows, self.seq_len * data.shape[1]), dtype=np.float32)
        for idx, i in enumerate(range(0, n_samples - self.seq_len + 1, stride)):
            if idx >= n_windows:
                break
            windows[idx] = data[i:i + self.seq_len].flatten()

        return windows

    def fit(self, train_data: np.ndarray, epoch_callback=None) -> 'USADBaseline':
        """
        Train USAD on training data.

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
            self.hidden_dims = [self.input_dim // 2, self.input_dim // 4]

        # Build model
        self.model = USADModel(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            latent_dim=self.latent_dim
        ).to(self.device)

        if self.verbose:
            print(f"{self.name} training:")
            print(f"  Device: {self.device}")
            print(f"  Input dim: {self.input_dim} (seq_len={self.seq_len} × n_features={self.n_features})")
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

        # Optimizers for each decoder path
        optimizer1 = torch.optim.Adam(
            list(self.model.encoder.parameters()) + list(self.model.decoder1.parameters()),
            lr=self.lr
        )
        optimizer2 = torch.optim.Adam(
            list(self.model.encoder.parameters()) + list(self.model.decoder2.parameters()),
            lr=self.lr
        )

        self.train_loss_history = []
        self.model.train()

        start_time = time.time()

        for epoch in range(self.epochs):
            epoch_start = time.time()
            epoch_losses = []

            for batch_idx, (batch_x,) in enumerate(dataloader):
                batch_x = batch_x.to(self.device)

                # Train path 1
                optimizer1.zero_grad()
                loss1, loss2 = self.model.compute_loss(batch_x, epoch + 1, self.epochs)
                loss1.backward(retain_graph=True)
                optimizer1.step()

                # Train path 2
                optimizer2.zero_grad()
                loss1, loss2 = self.model.compute_loss(batch_x, epoch + 1, self.epochs)
                loss2.backward()
                optimizer2.step()

                epoch_losses.append((loss1.item() + loss2.item()) / 2)

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
        """
        Compute anomaly scores for test data.

        Args:
            test_data: Test data (n_samples, n_features)

        Returns:
            Anomaly scores for each timestamp
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
                scores = self.model.get_anomaly_score(batch_x, self.alpha, self.beta)
                all_scores.append(scores.cpu().numpy())

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
        # Each window at position i covers timestamps [i, i+seq_len)
        # Assign score to the last timestamp of the window
        n_samples = len(test_data)
        scores = np.zeros(n_samples, dtype=np.float32)

        for i, score in enumerate(window_scores):
            target_idx = i + self.seq_len - 1
            if target_idx < n_samples:
                scores[target_idx] = score

        # Fill first timestamps with first available score
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
            "n_features": self.n_features,
            "input_dim": self.input_dim,
            "alpha": self.alpha,
            "beta": self.beta,
            "model_name": self.name,
        }
        with open(save_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

    def load(self, save_dir: Path) -> 'USADBaseline':
        """Load model weights and config."""
        save_dir = Path(save_dir)

        with open(save_dir / "config.json", 'r') as f:
            config = json.load(f)

        self.seq_len = config["seq_len"]
        self.hidden_dims = config["hidden_dims"]
        self.latent_dim = config["latent_dim"]
        self.n_features = config["n_features"]
        self.input_dim = config["input_dim"]
        self.alpha = config.get("alpha", 0.5)
        self.beta = config.get("beta", 0.5)

        self.model = USADModel(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            latent_dim=self.latent_dim
        ).to(self.device)
        self.model.load_state_dict(torch.load(save_dir / "model.pt", map_location=self.device))
        self.model.eval()

        return self

    def __repr__(self) -> str:
        return f"USADBaseline(seq_len={self.seq_len}, latent_dim={self.latent_dim})"
