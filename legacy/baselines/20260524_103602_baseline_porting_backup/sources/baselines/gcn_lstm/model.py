"""
1-Layer GCN-LSTM Baseline for Time Series Anomaly Detection

Based on QuoVadisTAD (arXiv:2405.02678):
- Graph Convolutional Network to capture sensor relationships
- LSTM to capture temporal dynamics
- Anomaly score = reconstruction error

The adjacency matrix is built from sensor correlations in training data.

Config from paper:
- lstm_units: 64
- embedding_dim: 10
- epochs: 100
- learning_rate: 0.001
"""

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Tuple
from pathlib import Path
from sklearn.metrics import pairwise_distances
import time
import sys


class GraphConvLayer(nn.Module):
    """
    Simple Graph Convolutional Layer.

    Aggregates neighbor features using adjacency matrix.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        """
        Args:
            x: Node features (batch, n_nodes, in_features)
            adj: Adjacency matrix (n_nodes, n_nodes)

        Returns:
            Updated node features (batch, n_nodes, out_features)
        """
        # Normalize adjacency matrix (add self-loop)
        adj = adj + torch.eye(adj.size(0), device=adj.device)
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1)
        adj_norm = adj / deg

        # Aggregate neighbor features
        x = torch.matmul(adj_norm, x)  # (batch, n_nodes, in_features)

        # Linear transformation
        return self.linear(x)


class GCNLSTM(nn.Module):
    """
    GCN-LSTM model for time series forecasting.

    Architecture (following QuoVadisTAD):
    1. Apply GCN at each timestep to capture spatial (sensor) relationships
    2. LSTM processes the full temporal sequence for each sensor
    3. Output projection to predict next timestamp

    Key insight: GCN operates on spatial dimension at each timestep,
    LSTM then processes the temporal sequence with spatially-enriched features.
    """

    def __init__(
        self,
        n_features: int,
        seq_len: int,
        gcn_out_dim: int = 10,
        lstm_units: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()

        self.n_features = n_features
        self.seq_len = seq_len
        self.gcn_out_dim = gcn_out_dim

        # GCN layer: input is 1 (scalar value per sensor per timestep)
        # Output is gcn_out_dim features per sensor
        self.gcn = GraphConvLayer(1, gcn_out_dim)
        self.gcn_act = nn.ReLU()
        self.gcn_dropout = nn.Dropout(dropout)

        # LSTM: processes seq_len timesteps, each with gcn_out_dim features
        self.lstm = nn.LSTM(
            input_size=gcn_out_dim,
            hidden_size=lstm_units,
            num_layers=1,
            batch_first=True,
            dropout=0
        )

        # Output projection: predict 1 value per sensor
        self.output = nn.Linear(lstm_units, 1)

    def forward(self, x, adj):
        """
        Args:
            x: Input (batch, seq_len, n_features)
            adj: Adjacency matrix (n_features, n_features)

        Returns:
            Predictions (batch, n_features)
        """
        batch_size = x.size(0)

        # Apply GCN at each timestep
        # x shape: (batch, seq_len, n_features)
        # We need to process each timestep through GCN

        gcn_outputs = []
        for t in range(self.seq_len):
            # Get timestep t: (batch, n_features)
            x_t = x[:, t, :]  # (batch, n_features)

            # Reshape for GCN: (batch, n_features, 1) - each sensor has 1 feature
            x_t = x_t.unsqueeze(-1)  # (batch, n_features, 1)

            # GCN aggregates across sensors
            gcn_out = self.gcn(x_t, adj)  # (batch, n_features, gcn_out_dim)
            gcn_out = self.gcn_act(gcn_out)
            gcn_out = self.gcn_dropout(gcn_out)

            gcn_outputs.append(gcn_out)

        # Stack: (batch, seq_len, n_features, gcn_out_dim)
        gcn_outputs = torch.stack(gcn_outputs, dim=1)

        # Reshape for LSTM: treat each sensor independently
        # (batch, seq_len, n_features, gcn_out_dim) -> (batch * n_features, seq_len, gcn_out_dim)
        gcn_outputs = gcn_outputs.permute(0, 2, 1, 3)  # (batch, n_features, seq_len, gcn_out_dim)
        gcn_outputs = gcn_outputs.reshape(batch_size * self.n_features, self.seq_len, self.gcn_out_dim)

        # LSTM processes the full temporal sequence for each sensor
        lstm_out, _ = self.lstm(gcn_outputs)  # (batch*n_features, seq_len, lstm_units)

        # Take the last timestep output
        lstm_out = lstm_out[:, -1, :]  # (batch*n_features, lstm_units)

        # Output projection
        out = self.output(lstm_out)  # (batch*n_features, 1)

        # Reshape back to (batch, n_features)
        out = out.reshape(batch_size, self.n_features)

        return out


class GCNLSTMBaseline:
    """
    1-Layer GCN-LSTM baseline for anomaly detection.

    Combines Graph Convolutional Networks to model sensor relationships
    with LSTM to capture temporal dynamics.
    """

    def __init__(
        self,
        seq_len: int = 5,
        gcn_out_dim: int = 10,
        lstm_units: int = 64,
        dropout: float = 0.1,
        lr: float = 0.001,
        batch_size: int = 100,
        epochs: int = 10,
        train_stride: int = 3,
        adj_topk: int = 10,
        device: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Args:
            seq_len: Input sequence length
            gcn_out_dim: GCN output dimension
            lstm_units: LSTM hidden units
            dropout: Dropout rate
            lr: Learning rate
            batch_size: Training batch size
            epochs: Number of training epochs
            train_stride: Stride for training windows (default 11)
            adj_topk: Number of top-k neighbors for adjacency matrix
            device: Device ('cuda' or 'cpu')
            verbose: Print training progress
        """
        self.seq_len = seq_len
        self.gcn_out_dim = gcn_out_dim
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_stride = train_stride
        self.adj_topk = adj_topk
        self.verbose = verbose

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model: Optional[GCNLSTM] = None
        self.adj: Optional[torch.Tensor] = None
        self.n_features: int = 0
        self.train_loss_history = []
        self.name = "1-Layer GCN-LSTM"

    def _compute_adjacency(self, data: np.ndarray) -> np.ndarray:
        """
        Compute adjacency matrix from sensor correlations.

        Uses top-k nearest neighbors based on Euclidean distance
        between sensor time series (transposed data).
        """
        # Transpose: (n_samples, n_features) -> (n_features, n_samples)
        # Each row is a sensor's time series
        sensor_series = data.T

        # Compute pairwise distances between sensors
        distances = pairwise_distances(sensor_series, metric='euclidean')

        # Create adjacency matrix using top-k neighbors
        n_features = len(distances)
        adj = np.zeros((n_features, n_features), dtype=np.float32)

        for i in range(n_features):
            # Get top-k closest sensors (excluding self)
            top_k_idx = np.argsort(distances[i])[1:self.adj_topk + 1]
            adj[i, top_k_idx] = 1

        # Make symmetric
        adj = np.maximum(adj, adj.T)

        # NaN/Inf guard
        adj = np.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=0.0)

        return adj

    def _create_windows(self, data: np.ndarray, stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Create sliding windows for training/inference."""
        n_samples = len(data)
        n_windows = (n_samples - self.seq_len - 1) // stride + 1

        windows = np.zeros((n_windows, self.seq_len, data.shape[1]), dtype=np.float32)
        targets = np.zeros((n_windows, data.shape[1]), dtype=np.float32)

        for idx, i in enumerate(range(0, n_samples - self.seq_len, stride)):
            if idx >= n_windows:
                break
            windows[idx] = data[i:i + self.seq_len]
            targets[idx] = data[i + self.seq_len]

        return windows, targets

    def fit(self, train_data: np.ndarray, epoch_callback=None) -> 'GCNLSTMBaseline':
        """
        Train the model on training data.

        Args:
            train_data: Training data (n_samples, n_features)
            epoch_callback: Optional callback(model, epoch) called after each epoch

        Returns:
            self
        """
        self.n_features = train_data.shape[1]

        # Compute adjacency matrix
        if self.verbose:
            print(f"{self.name} computing adjacency matrix...")
        adj_np = self._compute_adjacency(train_data[:10000])  # Use subset for efficiency
        self.adj = torch.FloatTensor(adj_np).to(self.device)
        if self.verbose:
            n_edges = (adj_np > 0).sum()
            print(f"  Adjacency: {self.n_features} nodes, {n_edges} edges")

        # Build model
        self.model = GCNLSTM(
            n_features=self.n_features,
            seq_len=self.seq_len,
            gcn_out_dim=self.gcn_out_dim,
            lstm_units=self.lstm_units,
            dropout=self.dropout
        ).to(self.device)

        if self.verbose:
            print(f"{self.name} training:")
            print(f"  Device: {self.device}")
            print(f"  Input shape: ({self.seq_len}, {self.n_features})")
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"  Total parameters: {total_params:,}")

        # Create windows with stride for training
        windows, targets = self._create_windows(train_data, stride=self.train_stride)
        if self.verbose:
            print(f"  Training windows: {len(windows):,} (stride={self.train_stride})")

        # Create DataLoader
        dataset = TensorDataset(
            torch.FloatTensor(windows),
            torch.FloatTensor(targets)
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True
        )

        # Training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self.train_loss_history = []
        self.model.train()

        start_time = time.time()

        for epoch in range(self.epochs):
            epoch_start = time.time()
            epoch_losses = []

            for batch_windows, batch_targets in dataloader:
                batch_windows = batch_windows.to(self.device)
                batch_targets = batch_targets.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_windows, self.adj)
                loss = criterion(outputs, batch_targets)

                if torch.isnan(loss) or torch.isinf(loss):
                    continue  # skip NaN/Inf batches

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
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
        windows, targets = self._create_windows(test_data)

        # Predict in batches
        all_errors = []
        n_batches = (len(windows) + self.batch_size - 1) // self.batch_size
        start_time = time.time()

        if self.verbose:
            print(f"  Inference: {len(windows):,} windows in {n_batches} batches")

        with torch.no_grad():
            for batch_idx, i in enumerate(range(0, len(windows), self.batch_size)):
                batch_windows = torch.FloatTensor(windows[i:i + self.batch_size]).to(self.device)
                batch_targets = targets[i:i + self.batch_size]

                outputs = self.model(batch_windows, self.adj).cpu().numpy()

                # Compute MSE error for each sample
                errors = np.mean((outputs - batch_targets) ** 2, axis=1)
                all_errors.append(errors)

                if self.verbose and (batch_idx + 1) % max(1, n_batches // 10) == 0:
                    progress = (batch_idx + 1) / n_batches * 100
                    elapsed = time.time() - start_time
                    print(f"\r  Inference: [{batch_idx + 1}/{n_batches}] {progress:5.1f}% | "
                          f"Elapsed: {elapsed:.1f}s", end="")
                    sys.stdout.flush()

        if self.verbose:
            total_time = time.time() - start_time
            print(f"\n  Inference complete: {total_time:.1f}s")

        window_errors = np.concatenate(all_errors)

        # Map window errors to point-level scores
        n_samples = len(test_data)
        scores = np.zeros(n_samples, dtype=np.float32)

        for i, error in enumerate(window_errors):
            target_idx = i + self.seq_len
            if target_idx < n_samples:
                scores[target_idx] = error

        # Forward-fill for first seq_len timestamps
        if len(window_errors) > 0:
            first_error = window_errors[0]
            for i in range(min(self.seq_len, n_samples)):
                scores[i] = first_error

        return scores

    def __repr__(self) -> str:
        return f"GCNLSTMBaseline(seq_len={self.seq_len}, lstm_units={self.lstm_units})"

    def save(self, save_dir: Path) -> None:
        """Save model weights, adjacency matrix, and config."""
        if self.model is None:
            raise RuntimeError("No model to save. Call fit() first.")

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save model weights
        torch.save(self.model.state_dict(), save_dir / "model.pt")

        # Save adjacency matrix
        np.save(save_dir / "adjacency.npy", self.adj.cpu().numpy())

        # Save config
        config = {
            "seq_len": self.seq_len,
            "gcn_out_dim": self.gcn_out_dim,
            "lstm_units": self.lstm_units,
            "dropout": self.dropout,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "adj_topk": self.adj_topk,
            "n_features": self.n_features,
            "train_loss_history": self.train_loss_history,
        }
        with open(save_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

        if self.verbose:
            print(f"  Saved model to {save_dir}")

    def load(self, save_dir: Path) -> 'GCNLSTMBaseline':
        """Load model weights, adjacency matrix, and config."""
        save_dir = Path(save_dir)

        # Load config
        with open(save_dir / "config.json", 'r') as f:
            config = json.load(f)

        self.seq_len = config["seq_len"]
        self.gcn_out_dim = config["gcn_out_dim"]
        self.lstm_units = config["lstm_units"]
        self.dropout = config["dropout"]
        self.n_features = config["n_features"]
        self.train_loss_history = config.get("train_loss_history", [])

        # Load adjacency matrix
        adj_np = np.load(save_dir / "adjacency.npy")
        self.adj = torch.FloatTensor(adj_np).to(self.device)

        # Build and load model
        self.model = GCNLSTM(
            n_features=self.n_features,
            seq_len=self.seq_len,
            gcn_out_dim=self.gcn_out_dim,
            lstm_units=self.lstm_units,
            dropout=self.dropout
        ).to(self.device)
        self.model.load_state_dict(torch.load(save_dir / "model.pt", map_location=self.device))
        self.model.eval()

        if self.verbose:
            print(f"  Loaded model from {save_dir}")

        return self


if __name__ == "__main__":
    # Test the baseline
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.wadi_loader import WaDiLoader

    data_path = Path(__file__).parent.parent.parent / "dataset/WaDi/WADI.A1_9 Oct 2017/WADI_attackdata_preprocessed.csv"
    loader = WaDiLoader(data_path=str(data_path), train_ratio=0.5)
    loader.load()

    train_X, train_y = loader.get_train_data()
    test_X, test_y = loader.get_test_data()

    # Test with fewer epochs for quick validation
    model = GCNLSTMBaseline(
        seq_len=5,
        gcn_out_dim=10,
        lstm_units=64,
        epochs=10,  # Reduced for testing
        verbose=True
    )
    model.fit(train_X)

    # Test on subset
    scores = model.predict(test_X[:10000])

    print(f"\n1-Layer GCN-LSTM Results (on 10000 test samples):")
    print(f"  Score range: [{scores.min():.6f}, {scores.max():.6f}]")
    print(f"  Score mean: {scores.mean():.6f}")
    print(f"  Score std: {scores.std():.6f}")
