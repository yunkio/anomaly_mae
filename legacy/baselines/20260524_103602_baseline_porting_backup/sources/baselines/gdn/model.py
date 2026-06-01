"""
GDN - Graph Deviation Network

Based on: "Graph Neural Network-Based Anomaly Detection in Multivariate Time Series"
Paper: AAAI 2021, https://arxiv.org/abs/2106.06947
Reference: https://github.com/d-ailin/GDN

Architecture:
1. Learn embeddings for each sensor/feature
2. Build a graph structure based on embedding similarity
3. Use Graph Attention Network (GAT) to propagate information
4. Predict next values and use deviation as anomaly score

Key Insight: Models inter-sensor dependencies as a graph where edges
represent learned relationships between sensors.
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


class GraphLayer(nn.Module):
    """
    Graph Attention Layer for message passing between nodes.

    Uses attention mechanism to weight contributions from neighbors.
    """

    def __init__(self, in_features: int, out_features: int, n_heads: int = 1, dropout: float = 0.0):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads

        # Linear transformations for attention
        self.W = nn.Linear(in_features, out_features * n_heads, bias=False)
        self.a_src = nn.Parameter(torch.Tensor(n_heads, out_features))
        self.a_dst = nn.Parameter(torch.Tensor(n_heads, out_features))

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features (batch, n_nodes, in_features)
            edge_index: Edge indices (2, n_edges)

        Returns:
            Updated node features (batch, n_nodes, out_features * n_heads)
        """
        batch_size, n_nodes, _ = x.shape

        # Linear transformation
        h = self.W(x)  # (batch, n_nodes, out_features * n_heads)
        h = h.view(batch_size, n_nodes, self.n_heads, self.out_features)

        # Compute attention scores
        src_idx, dst_idx = edge_index[0], edge_index[1]

        # Get source and destination node features for each edge
        h_src = h[:, src_idx]  # (batch, n_edges, n_heads, out_features)
        h_dst = h[:, dst_idx]  # (batch, n_edges, n_heads, out_features)

        # Attention coefficients
        e_src = (h_src * self.a_src).sum(dim=-1)  # (batch, n_edges, n_heads)
        e_dst = (h_dst * self.a_dst).sum(dim=-1)  # (batch, n_edges, n_heads)
        e = self.leaky_relu(e_src + e_dst)

        # Softmax over incoming edges for each node
        # Create sparse attention matrix and normalize
        # attention shape: (batch, n_nodes, n_nodes, n_heads)
        # e shape: (batch, n_edges, n_heads)
        # Using advanced indexing: attention[:, dst_idx, src_idx] selects (batch, n_edges, n_heads)
        attention = torch.zeros(batch_size, n_nodes, n_nodes, self.n_heads, device=x.device)
        attention[:, dst_idx, src_idx, :] = e

        attention = F.softmax(attention, dim=2)  # Normalize over source nodes
        attention = self.dropout(attention)

        # Aggregate messages
        h_out = torch.einsum('bijk,bjkf->bikf', attention, h)  # (batch, n_nodes, n_heads, out_features)
        h_out = h_out.reshape(batch_size, n_nodes, -1)

        return h_out


class GDNModel(nn.Module):
    """
    Graph Deviation Network model.

    Architecture:
    1. Node embeddings (learnable)
    2. Graph construction via embedding similarity
    3. GAT layers for message passing
    4. Output layer for prediction
    """

    def __init__(
        self,
        n_features: int,
        seq_len: int = 5,
        embed_dim: int = 64,
        gnn_out_dim: int = 64,
        n_heads: int = 1,
        top_k: int = 5,
        dropout: float = 0.2
    ):
        super().__init__()

        self.n_features = n_features
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.top_k = min(top_k, n_features - 1)

        # Learnable node embeddings
        self.node_embedding = nn.Embedding(n_features, embed_dim)

        # Input projection (from seq_len to embed_dim)
        self.input_proj = nn.Linear(seq_len, embed_dim)

        # Graph attention layer
        self.gat = GraphLayer(embed_dim, gnn_out_dim, n_heads=n_heads, dropout=dropout)
        # BatchNorm after GAT: input shape (batch, gnn_out_dim*n_heads, n_features)
        self.bn = nn.BatchNorm1d(gnn_out_dim * n_heads)

        # Output layer
        self.output = nn.Sequential(
            nn.Linear(gnn_out_dim * n_heads + embed_dim, gnn_out_dim),
            nn.ReLU(),
            nn.Linear(gnn_out_dim, 1)
        )

        # Edge index (will be computed in forward)
        self.edge_index = None

    def _compute_graph(self) -> torch.Tensor:
        """
        Compute graph structure based on node embedding similarity.
        Returns edge_index tensor of shape (2, n_edges).
        """
        # Get all node embeddings
        node_ids = torch.arange(self.n_features, device=self.node_embedding.weight.device)
        embeddings = self.node_embedding(node_ids)  # (n_features, embed_dim)

        # Compute cosine similarity
        embeddings_norm = F.normalize(embeddings, dim=-1)
        similarity = torch.mm(embeddings_norm, embeddings_norm.t())  # (n_features, n_features)

        # Set self-similarity to -inf to exclude self-loops
        similarity.fill_diagonal_(-float('inf'))

        # Get top-k neighbors for each node
        _, topk_indices = torch.topk(similarity, self.top_k, dim=1)

        # Create edge index
        src_nodes = torch.arange(self.n_features, device=similarity.device).unsqueeze(1).expand(-1, self.top_k)
        src_nodes = src_nodes.reshape(-1)
        dst_nodes = topk_indices.reshape(-1)

        edge_index = torch.stack([src_nodes, dst_nodes], dim=0)

        return edge_index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq_len, n_features)

        Returns:
            Predictions for next timestamp (batch, n_features)
        """
        batch_size = x.shape[0]

        # Transpose to (batch, n_features, seq_len)
        x = x.transpose(1, 2)

        # Project input sequence to embedding dimension
        h = self.input_proj(x)  # (batch, n_features, embed_dim)

        # Get node embeddings
        node_ids = torch.arange(self.n_features, device=x.device)
        node_emb = self.node_embedding(node_ids)  # (n_features, embed_dim)
        node_emb = node_emb.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, n_features, embed_dim)

        # Combine with node embeddings
        h = h * node_emb

        # Compute graph structure
        edge_index = self._compute_graph()

        # Apply graph attention
        h_gat = self.gat(h, edge_index)  # (batch, n_features, gnn_out_dim * n_heads)

        # Batch normalization
        h_gat = self.bn(h_gat.transpose(1, 2)).transpose(1, 2)
        h_gat = F.relu(h_gat)

        # Concatenate with node embeddings
        h_cat = torch.cat([h_gat, node_emb], dim=-1)

        # Output prediction
        out = self.output(h_cat).squeeze(-1)  # (batch, n_features)

        return out


class GDNBaseline:
    """
    GDN baseline wrapper for time series anomaly detection.
    """

    def __init__(
        self,
        seq_len: int = 5,
        embed_dim: int = 64,
        gnn_out_dim: int = 64,
        n_heads: int = 1,
        top_k: int = 5,
        dropout: float = 0.2,
        lr: float = 0.001,
        batch_size: int = 256,
        epochs: int = 10,
        train_stride: int = 3,
        device: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Args:
            seq_len: Window size for input
            embed_dim: Node embedding dimension
            gnn_out_dim: GNN output dimension
            n_heads: Number of attention heads
            top_k: Number of neighbors in graph
            dropout: Dropout rate
            lr: Learning rate
            batch_size: Training batch size
            epochs: Number of training epochs
            train_stride: Stride for training windows (default 11)
            device: 'cuda' or 'cpu'
            verbose: Print training progress
        """
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.gnn_out_dim = gnn_out_dim
        self.n_heads = n_heads
        self.top_k = top_k
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_stride = train_stride
        self.verbose = verbose

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model: Optional[GDNModel] = None
        self.n_features: int = 0
        self.train_loss_history = []

    @property
    def name(self) -> str:
        return "GDN"

    def _create_windows(self, data: np.ndarray, stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Create sliding windows with targets."""
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

    def fit(self, train_data: np.ndarray, epoch_callback=None) -> 'GDNBaseline':
        """
        Train GDN on training data.

        Args:
            train_data: Training data (n_samples, n_features)
            epoch_callback: Optional callback(model, epoch) called after each epoch

        Returns:
            self
        """
        self.n_features = train_data.shape[1]

        # Build model
        self.model = GDNModel(
            n_features=self.n_features,
            seq_len=self.seq_len,
            embed_dim=self.embed_dim,
            gnn_out_dim=self.gnn_out_dim,
            n_heads=self.n_heads,
            top_k=self.top_k,
            dropout=self.dropout
        ).to(self.device)

        if self.verbose:
            print(f"{self.name} training:")
            print(f"  Device: {self.device}")
            print(f"  Input shape: ({self.seq_len}, {self.n_features})")
            print(f"  Graph: {self.n_features} nodes, top-{self.top_k} neighbors")
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"  Total parameters: {total_params:,}")

        # Create windows with stride for training
        windows, targets = self._create_windows(train_data, stride=self.train_stride)
        if self.verbose:
            print(f"  Training windows: {len(windows):,} (stride={self.train_stride})")

        # Create DataLoader
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(windows),
            torch.FloatTensor(targets)
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )

        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self.train_loss_history = []
        self.model.train()

        start_time = time.time()

        for epoch in range(self.epochs):
            epoch_start = time.time()
            epoch_losses = []

            for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()

                pred = self.model(batch_x)
                loss = criterion(pred, batch_y)

                loss.backward()
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
            Anomaly scores for each timestamp (prediction error)
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        self.model.eval()

        # Create windows
        windows, targets = self._create_windows(test_data)
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
                batch_y = targets[i:i + self.batch_size]

                pred = self.model(batch_x).cpu().numpy()

                # Compute MSE per sample
                errors = np.mean((pred - batch_y) ** 2, axis=1)
                all_scores.append(errors)

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
            target_idx = i + self.seq_len
            if target_idx < n_samples:
                scores[target_idx] = score

        # Fill first timestamps
        if len(window_scores) > 0:
            for i in range(min(self.seq_len, n_samples)):
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
            "embed_dim": self.embed_dim,
            "gnn_out_dim": self.gnn_out_dim,
            "n_heads": self.n_heads,
            "top_k": self.top_k,
            "n_features": self.n_features,
            "model_name": self.name,
        }
        with open(save_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

    def load(self, save_dir: Path) -> 'GDNBaseline':
        """Load model weights and config."""
        save_dir = Path(save_dir)

        with open(save_dir / "config.json", 'r') as f:
            config = json.load(f)

        self.seq_len = config["seq_len"]
        self.embed_dim = config["embed_dim"]
        self.gnn_out_dim = config["gnn_out_dim"]
        self.n_heads = config["n_heads"]
        self.top_k = config["top_k"]
        self.n_features = config["n_features"]

        self.model = GDNModel(
            n_features=self.n_features,
            seq_len=self.seq_len,
            embed_dim=self.embed_dim,
            gnn_out_dim=self.gnn_out_dim,
            n_heads=self.n_heads,
            top_k=self.top_k,
        ).to(self.device)
        self.model.load_state_dict(torch.load(save_dir / "model.pt", map_location=self.device))
        self.model.eval()

        return self

    def __repr__(self) -> str:
        return f"GDNBaseline(seq_len={self.seq_len}, embed_dim={self.embed_dim}, top_k={self.top_k})"
