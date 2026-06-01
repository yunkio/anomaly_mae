"""
1-Layer GCN-LSTM Baseline for Time Series Anomaly Detection
(QuoVadisTAD reproduction, ICML 2024 Position Paper — Sarfraz et al., arXiv:2405.02678)

Reference implementation: https://github.com/ssarfraz/QuoVadisTAD (MIT)
- Architecture: 1-layer GCN over sensor graph + 1-layer LSTM forecaster
- Graph: FINCH cosine 1-NN adjacency (symmetric A @ A.T → sign())
- GCN: shared weight W; aggregation_type="max" (segment-max over neighbors);
  combination_type="concat" (self + neighbors → 2*out_feat); activation="relu"
- LSTM: tanh activation, 1 layer, no dropout
- Output: Dense(output_seq_len=1, activation='relu') + extra Linear(num_nodes, num_nodes)
- Loss: MSE on next-step forecast
- Score: per-sensor abs(pred - target) → median-IQR normalize → smooth (window=5)
        → MAX over sensors → 1D scalar per timestamp; first seq_len → forward-fill
        from first valid score (Phase 2.5 Option B substitution for reference's
        EXPLICIT_TRUNCATE; pipeline cannot truncate labels)

Notes vs reference:
- epochs = 50 (./comparison/ Domain B policy; reference is 100 + EarlyStopping).
  This is an intentional pipeline policy — Domain B uniformity across baselines
  with best-epoch tracked via pak_auc_f1.
- batch_size = 256 (Domain B uniformity; reference is 100).
- train/val split: removed (Domain B handles best-epoch via pak_auc_f1).
- Training loop is otherwise paper-faithful (no NaN/Inf guard, no gradient clipping).

License: MIT (compatible with QuoVadisTAD upstream).
"""

import json
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from comparison.segment_utils import compute_segment_safe_window_indices
from comparison.baselines._boundary_safe_window import entity_test_slices
import torch
import torch.nn as nn
from scipy.stats import iqr
from sklearn.metrics import pairwise_distances
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# Adjacency: FINCH cosine 1-NN (port of compute_finch_adjacency_matrix)
# ============================================================

def _compute_finch_adjacency(train_array: np.ndarray, symmetric: bool = True) -> np.ndarray:
    """FINCH 1-NN adjacency over cosine pairwise distances of sensor time series.

    Port of `quovadis_tad/model_utils/gnn.py::compute_finch_adjacency_matrix`.

    Args:
        train_array: shape (T, F) — float training data (already normalized upstream).
        symmetric: if True, apply A @ A.T → sign() (1-NN ∪ shared-1-NN consensus).

    Returns:
        Dense binary adjacency (F, F), float32, diag = 0.
    """
    arr = train_array.T  # (F, T) — each row is one sensor's time series
    s = arr.shape[0]

    # cosine pairwise; sklearn returns dense (F, F)
    orig_dist = pairwise_distances(arr, arr, metric='cosine')
    np.fill_diagonal(orig_dist, 1e12)

    # 1-NN: nearest non-self neighbor per row
    initial_rank = np.argmin(orig_dist, axis=1)

    # Dense binary adjacency (F, F)
    A = np.zeros((s, s), dtype=np.float32)
    A[np.arange(s), initial_rank] = 1.0
    # Add identity (self-loops, just like reference `A + sp.eye(s)`)
    np.fill_diagonal(A, 1.0)

    if symmetric:
        # A @ A.T → binary sign (high-order proximity: shared 1-NN)
        A = A @ A.T
        A = np.sign(A).astype(np.float32)

    # setdiag(0) — remove self-loops from edge list (reference's final lil.setdiag(0))
    np.fill_diagonal(A, 0.0)

    return A


def _adj_to_edge_index(adj: np.ndarray) -> torch.Tensor:
    """Convert dense binary adjacency to edge_index (2, E) as in TF reference.

    Reference (gnn.py::graph_info): edges = (target_nodes, source_nodes) where
    edges[0] is the destination of each message and edges[1] is the source.
    We build edges such that edge[0]=i, edge[1]=j iff adj[i, j] == 1.
    """
    src_dst = np.argwhere(adj > 0)  # shape (E, 2): col 0 = i (dest), col 1 = j (src)
    if src_dst.size == 0:
        # Degenerate graph (e.g. tiny test data). Return empty.
        return torch.zeros((2, 0), dtype=torch.long)
    # Transpose to (2, E)
    edges = torch.from_numpy(src_dst.T.astype(np.int64))
    return edges


# ============================================================
# GraphConv (port of gnn.py::GraphConv)
# ============================================================

class GraphConvLayer(nn.Module):
    """FINCH-style GraphConv.

    Reference (gnn.py L84-L165):
      - shared weight `W: (in_feat, out_feat)`
      - self repr  = features @ W           # (..., out_feat)
      - aggregated = segment_<agg>(gather(features, edges[1]), edges[0], num_nodes) @ W
      - combine    = concat([self, aggregated], dim=-1)  → (..., 2*out_feat)
      - output     = activation(combine)
    """

    def __init__(
        self,
        in_feat: int,
        out_feat: int,
        aggregation_type: str = "max",
        combination_type: str = "concat",
        activation: str = "relu",
    ):
        super().__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.aggregation_type = aggregation_type
        self.combination_type = combination_type

        # shared weight (Glorot uniform — matches keras default)
        self.weight = nn.Parameter(torch.empty(in_feat, out_feat))
        nn.init.xavier_uniform_(self.weight)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation is None or activation == "linear":
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # edges and num_nodes are set externally by GCNLSTM (after FINCH compute)
        self._edges: Optional[torch.Tensor] = None  # shape (2, E)
        self._num_nodes: int = 0

    def set_graph(self, edges: torch.Tensor, num_nodes: int) -> None:
        """Register graph edges + node count. Called once after adjacency build."""
        # Save edges as buffer-like (non-trainable) state
        self._edges = edges
        self._num_nodes = num_nodes

    def _segment_aggregate(self, messages: torch.Tensor, dest: torch.Tensor) -> torch.Tensor:
        """Aggregate messages by destination node index.

        Args:
            messages: (E, batch, seq, out_feat) — one message per edge
            dest:     (E,) — destination node index for each edge

        Returns:
            (num_nodes, batch, seq, out_feat)
        """
        E = messages.shape[0]
        tail_shape = messages.shape[1:]  # (batch, seq, out_feat)
        out = messages.new_zeros((self._num_nodes,) + tuple(tail_shape))

        if E == 0:
            return out

        # Build index of same shape as messages, broadcasting dest along tail dims
        index = dest.view(E, *([1] * len(tail_shape))).expand_as(messages)

        if self.aggregation_type == "max":
            out = out.scatter_reduce(
                dim=0, index=index, src=messages, reduce="amax", include_self=False
            )
        elif self.aggregation_type == "mean":
            out = out.scatter_reduce(
                dim=0, index=index, src=messages, reduce="mean", include_self=False
            )
        elif self.aggregation_type == "sum":
            out = out.scatter_reduce(
                dim=0, index=index, src=messages, reduce="sum", include_self=False
            )
        else:
            raise ValueError(f"Invalid aggregation type: {self.aggregation_type}")
        return out

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            features: (num_nodes, batch, seq, in_feat)

        Returns:
            (num_nodes, batch, seq, 2*out_feat)   if combination_type=="concat"
            (num_nodes, batch, seq, out_feat)     if "add"
        """
        if self._edges is None:
            raise RuntimeError("GraphConvLayer: edges not set. Call set_graph() first.")

        # Move edges to the same device as features (cheap; cached)
        edges = self._edges.to(features.device)
        dest = edges[0]   # (E,)
        src = edges[1]    # (E,)

        # Self representation: features @ W → (num_nodes, batch, seq, out_feat)
        nodes_repr = torch.matmul(features, self.weight)

        # Neighbor messages: gather features at src indices → (E, batch, seq, in_feat)
        if src.numel() > 0:
            gathered = features.index_select(dim=0, index=src)
            # Aggregate by dest → (num_nodes, batch, seq, in_feat)
            aggregated_in = self._segment_aggregate(gathered, dest)
            # Transform: (num_nodes, batch, seq, out_feat)
            aggregated = torch.matmul(aggregated_in, self.weight)
        else:
            aggregated = torch.zeros_like(nodes_repr)

        # Combine
        if self.combination_type == "concat":
            h = torch.cat([nodes_repr, aggregated], dim=-1)  # (..., 2*out_feat)
        elif self.combination_type == "add":
            h = nodes_repr + aggregated
        else:
            raise ValueError(f"Invalid combination type: {self.combination_type}")

        return self.activation(h)


# ============================================================
# GCNLSTM (port of gnn.py::LSTMGC + model_def.py::gcn_sequense_sensor_error_model_v1)
# ============================================================

class GCNLSTM(nn.Module):
    """GCNLSTM forecaster.

    Forward:
      inputs (batch, seq_len, num_nodes)
        → reshape (batch, seq_len, num_nodes, 1)
        → transpose (num_nodes, batch, seq_len, 1)
        → GraphConv → (num_nodes, batch, seq_len, 2*gcn_out_dim)
        → reshape (batch*num_nodes, seq_len, 2*gcn_out_dim)
        → LSTM → (batch*num_nodes, lstm_units)   # last hidden
        → Dense(output_seq_len, relu) → (batch*num_nodes, output_seq_len)
        → reshape (num_nodes, batch, output_seq_len)
        → transpose (batch, output_seq_len, num_nodes)
        → extra Dense(num_nodes) over node axis → (batch, output_seq_len, num_nodes)
        → squeeze output_seq_len=1 → (batch, num_nodes)
    """

    def __init__(
        self,
        n_features: int,
        gcn_out_dim: int = 10,
        lstm_units: int = 64,
        output_seq_len: int = 1,
        aggregation_type: str = "max",
        combination_type: str = "concat",
        graph_conv_activation: str = "relu",
    ):
        super().__init__()

        self.n_features = n_features
        self.gcn_out_dim = gcn_out_dim
        self.lstm_units = lstm_units
        self.output_seq_len = output_seq_len
        self.combination_type = combination_type

        self.graph_conv = GraphConvLayer(
            in_feat=1,
            out_feat=gcn_out_dim,
            aggregation_type=aggregation_type,
            combination_type=combination_type,
            activation=graph_conv_activation,
        )

        # LSTM input dim = 2*gcn_out_dim if concat, else gcn_out_dim
        lstm_in_dim = 2 * gcn_out_dim if combination_type == "concat" else gcn_out_dim
        self.lstm = nn.LSTM(
            input_size=lstm_in_dim,
            hidden_size=lstm_units,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )

        # Per-sensor dense projection with ReLU (matches reference Dense(1, activation='relu'))
        self.dense = nn.Linear(lstm_units, output_seq_len)
        self.dense_act = nn.ReLU()

        # Extra Linear(num_nodes, num_nodes) over node axis (reference top-level Dense(num_nodes))
        self.extra_node_dense = nn.Linear(n_features, n_features)

    def set_graph(self, edges: torch.Tensor, num_nodes: int) -> None:
        self.graph_conv.set_graph(edges, num_nodes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, num_nodes)

        Returns:
            predictions (batch, num_nodes) — squeeze output_seq_len if 1
        """
        batch, seq_len, num_nodes = x.shape

        # (batch, seq, nodes) → (batch, seq, nodes, 1)
        x4 = x.unsqueeze(-1)
        # → (nodes, batch, seq, 1)
        x4 = x4.permute(2, 0, 1, 3).contiguous()

        # GraphConv → (nodes, batch, seq, 2*gcn_out_dim or gcn_out_dim)
        gcn_out = self.graph_conv(x4)
        out_feat = gcn_out.shape[-1]

        # LSTM input: (nodes*batch, seq, out_feat). Reference reshape order is
        # `tf.reshape(gcn_out, (batch*num_nodes, seq, out_feat))` after the
        # tensor has shape (num_nodes, batch_size, seq, out_feat) — i.e. the
        # natural row-major flatten of (num_nodes, batch, ...) groups
        # `num_nodes` slowest. We mirror that.
        gcn_flat = gcn_out.reshape(num_nodes * batch, seq_len, out_feat)

        lstm_out, _ = self.lstm(gcn_flat)
        last = lstm_out[:, -1, :]  # (nodes*batch, lstm_units)

        # Dense + ReLU
        dense_out = self.dense_act(self.dense(last))  # (nodes*batch, output_seq_len)

        # → (nodes, batch, output_seq_len)
        dense_out = dense_out.reshape(num_nodes, batch, self.output_seq_len)
        # → (batch, output_seq_len, nodes)
        dense_out = dense_out.permute(1, 2, 0).contiguous()

        # Extra node-axis Dense (reference top-level `Dense(num_nodes)`)
        # Apply linear over last dim (which is `nodes`)
        out = self.extra_node_dense(dense_out)  # (batch, output_seq_len, nodes)

        if self.output_seq_len == 1:
            out = out.squeeze(1)  # (batch, nodes)

        return out


# ============================================================
# Wrapper (preserves fit/predict/save/load contract for run_sota_baseline_with_epoch_eval)
# ============================================================

class GCNLSTMBaseline:
    """1-Layer GCN-LSTM baseline for anomaly detection (QuoVadisTAD reproduction).

    Compatible with `run_sota_baseline_with_epoch_eval`: `fit(train_X, epoch_callback=...)`.
    """

    def __init__(
        self,
        seq_len: int = 5,
        gcn_out_dim: int = 10,
        lstm_units: int = 64,
        output_seq_len: int = 1,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 256,
        epochs: int = 10,
        train_stride: int = 1,
        aggregation_type: str = "max",
        combination_type: str = "concat",
        graph_conv_activation: str = "relu",
        adj_symmetric: bool = True,
        score_norm: str = "median-iqr",
        score_smooth_window: int = 5,
        score_iqr_epsilon: float = 1e-2,
        device: Optional[str] = None,
        verbose: bool = True,
        # legacy params (ignored but accepted for backwards compat)
        dropout: float = 0.0,
        adj_topk: Optional[int] = None,
        **extra_kwargs,
    ):
        self.seq_len = seq_len
        self.gcn_out_dim = gcn_out_dim
        self.lstm_units = lstm_units
        self.output_seq_len = output_seq_len
        self.lr = lr
        self.weight_decay = float(weight_decay)
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_stride = train_stride
        self.aggregation_type = aggregation_type
        self.combination_type = combination_type
        self.graph_conv_activation = graph_conv_activation
        self.adj_symmetric = adj_symmetric
        self.score_norm = score_norm
        self.score_smooth_window = score_smooth_window
        self.score_iqr_epsilon = score_iqr_epsilon
        self.verbose = verbose
        # `extra_kwargs` 흡수 — preset 에 future keys 추가 시 silent fail 방지.
        # 사용되지 않는 키는 그대로 무시되며 wrapper 동작에 영향 없음.

        # Legacy (no-op): reference GCN-LSTM has no dropout, no top-k adjacency.
        # Kept in signature so older configs / persisted checkpoints don't break.
        self.dropout = dropout
        self.adj_topk = adj_topk

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model: Optional[GCNLSTM] = None
        self.edges: Optional[torch.Tensor] = None  # (2, E)
        self.n_features: int = 0
        self.train_loss_history: list = []
        self.name = "1-Layer GCN-LSTM"

    # ------------------------------------------------------------------
    # Graph + windowing helpers
    # ------------------------------------------------------------------

    def _compute_adjacency(self, data: np.ndarray) -> np.ndarray:
        """FINCH cosine 1-NN adjacency. See module-level `_compute_finch_adjacency`."""
        return _compute_finch_adjacency(data, symmetric=self.adj_symmetric)

    def _create_windows(self, data: np.ndarray, stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Next-step forecasting windows.

        Reference (tf_data_loader.py::create_tf_dataset, forecast_horizon=1):
          - For i in [0, N - seq_len - 1] (inclusive end via < N - seq_len + 1):
            window = data[i : i + seq_len], target = data[i + seq_len]
          - i.e. number of windows = N - seq_len  (stride=1)
        """
        n_samples = len(data)
        n_features = data.shape[1]
        last_valid_start = n_samples - self.seq_len  # window i requires data[i + seq_len]
        if last_valid_start <= 0:
            return (np.zeros((0, self.seq_len, n_features), dtype=np.float32),
                    np.zeros((0, n_features), dtype=np.float32))

        # stride=1 → exactly `last_valid_start` windows (i in [0, last_valid_start - 1])
        n_windows = (last_valid_start + stride - 1) // stride
        windows = np.zeros((n_windows, self.seq_len, n_features), dtype=np.float32)
        targets = np.zeros((n_windows, n_features), dtype=np.float32)

        for idx in range(n_windows):
            i = idx * stride
            if i >= last_valid_start:
                break
            windows[idx] = data[i:i + self.seq_len]
            targets[idx] = data[i + self.seq_len]

        return windows, targets

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, train_data: np.ndarray, epoch_callback=None, train_segments=None) -> 'GCNLSTMBaseline':
        """Train next-step forecaster.

        Args:
            train_data: (T_train, n_features) float32. Already normalized by driver (Path A).
            epoch_callback: optional callable(self, ep) — called after each epoch end
                            (ep ∈ [1, self.epochs]).
        """
        self.n_features = train_data.shape[1]

        # 1) FINCH adjacency over the full training series (sensors ≤ 130 → cheap)
        if self.verbose:
            print(f"{self.name} computing FINCH cosine 1-NN adjacency...")
        adj_np = self._compute_adjacency(train_data)
        edges = _adj_to_edge_index(adj_np)
        self.edges = edges
        if self.verbose:
            n_edges = int(edges.shape[1])
            print(f"  Adjacency: {self.n_features} nodes, {n_edges} edges "
                  f"(symmetric={self.adj_symmetric})")

        # 2) Build model + register graph
        self.model = GCNLSTM(
            n_features=self.n_features,
            gcn_out_dim=self.gcn_out_dim,
            lstm_units=self.lstm_units,
            output_seq_len=self.output_seq_len,
            aggregation_type=self.aggregation_type,
            combination_type=self.combination_type,
            graph_conv_activation=self.graph_conv_activation,
        ).to(self.device)
        self.model.set_graph(self.edges, self.n_features)

        if self.verbose:
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"{self.name} training:")
            print(f"  Device: {self.device}")
            print(f"  Input shape: ({self.seq_len}, {self.n_features})")
            print(f"  Total parameters: {total_params:,}")

        # 3) Build sliding windows (train stride from preset)
        windows, targets = self._create_windows(train_data, stride=self.train_stride)
        if train_segments is not None:
            # GCN-LSTM predicts data[s + seq_len] from data[s : s + seq_len], so
            # the target step must also remain inside the segment → extra_horizon=1.
            valid_idx = compute_segment_safe_window_indices(
                train_segments, self.seq_len, self.train_stride, len(windows),
                extra_horizon=1,
            )
            if self.verbose:
                print(f"  Segment-aware: kept {len(valid_idx)}/{len(windows)} windows "
                      f"({len(valid_idx)/max(len(windows),1):.1%}) — dropped boundary-crossing")
            windows = windows[valid_idx]
            targets = targets[valid_idx]
        if self.verbose:
            print(f"  Training windows: {len(windows):,} (stride={self.train_stride})")

        if len(windows) == 0:
            if self.verbose:
                print("  WARNING: zero training windows (T_train < seq_len + 1).")
            return self

        dataset = TensorDataset(
            torch.from_numpy(windows), torch.from_numpy(targets)
        )
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=False
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr,
            weight_decay=getattr(self, 'weight_decay', 0.0),
        )
        criterion = nn.MSELoss()

        self.train_loss_history = []
        self.model.train()
        start_time = time.time()

        n_batches = len(dataloader)
        for epoch in range(self.epochs):
            epoch_start = time.time()
            epoch_losses = []

            for batch_idx, (batch_windows, batch_targets) in enumerate(dataloader):
                batch_start = time.time()
                batch_windows = batch_windows.to(self.device)
                batch_targets = batch_targets.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_windows)
                loss = criterion(outputs, batch_targets)

                loss.backward()
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

        if self.verbose:
            total_time = time.time() - start_time
            final = self.train_loss_history[-1] if self.train_loss_history else float('nan')
            print(f"\n  Training complete: {total_time:.1f}s total, Final loss: {final:.6f}")

        return self

    # ------------------------------------------------------------------
    # Scoring (FINCH-style: |pred-target| → median-IQR → smooth → max-over-sensors)
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_scores(
        test_delta: np.ndarray,
        norm: str = "median-iqr",
        smooth: bool = True,
        smooth_window: int = 5,
        epsilon: float = 1e-2,
    ) -> np.ndarray:
        """Port of `quovadis_tad/dataset_utils/data_utils.py::normalise_scores`.

        Args:
            test_delta: (n_windows, n_sensors) — per-window per-sensor residual.

        Returns:
            smoothed scores (n_windows, n_sensors). First `smooth_window` rows = 0
            (matches reference behavior).
        """
        if norm == "median-iqr":
            n_err_mid = np.median(test_delta, axis=0)
            n_err_iqr = iqr(test_delta, axis=0)
            err_scores = (test_delta - n_err_mid) / (np.abs(n_err_iqr) + epsilon)
        elif norm is None or norm == "none":
            err_scores = test_delta
        else:
            raise ValueError(f"Unsupported norm: {norm}")

        if smooth and len(err_scores) > smooth_window:
            smoothed = np.zeros_like(err_scores)
            for i in range(smooth_window, len(err_scores)):
                # Reference: mean over [i - sw, i + sw - 1) — length 2*sw - 1
                lo = i - smooth_window
                hi = i + smooth_window - 1
                hi = min(hi, len(err_scores))
                if hi > lo:
                    smoothed[i] = np.mean(err_scores[lo:hi], axis=0)
            return smoothed
        return err_scores

    def _infer_window_residuals(self, sub_data: np.ndarray) -> np.ndarray:
        """RAW per-entity producer: stride-1 windowing + model inference → per-window residual.

        Boundary-safe by construction — operates on ONE entity's test slice only, so no window
        spans an entity boundary. Returns the per-sensor absolute residual matrix
        ``(n_windows, n_sensors)`` (NOT collapsed / NOT post-processed). Reference residual:
        ``np.abs(predictions - orig_target)`` (model_def.py::test_embedder L426-435).

        Edge-safe: a slice with ``len(sub_data) <= seq_len`` yields 0 windows; ``_create_windows``
        already returns empty arrays for that case, so an empty ``(0, n_sensors)`` is returned and
        the caller fills that entity's timesteps via the documented fallback.
        """
        windows, targets = self._create_windows(sub_data, stride=1)
        if len(windows) == 0:
            return np.zeros((0, self.n_features), dtype=np.float32)

        residuals: list = []  # each (B, n_features)
        with torch.no_grad():
            for i in range(0, len(windows), self.batch_size):
                bw = torch.from_numpy(windows[i:i + self.batch_size]).to(self.device)
                bt = targets[i:i + self.batch_size]  # numpy
                pred = self.model(bw).cpu().numpy()
                # per-sensor absolute residual (reference: np.abs(predictions - orig_target))
                residuals.append(np.abs(pred - bt))

        return np.concatenate(residuals, axis=0).astype(np.float32)

    def predict(self, test_data: np.ndarray, test_segments=None) -> np.ndarray:
        """Compute paper-faithful anomaly scores (boundary-safe TEST windowing).

        Args:
            test_data: (T_test, n_features) float32. Already normalized by driver (Path A).
            test_segments: per-entity (lo, hi) TEST-LOCAL slices (from the SAME source as
                per-entity normalization, ``loader.get_file_norm_segments()`` test side). None /
                single-entity / non-tiling → ONE pass over the whole array == legacy behaviour.

        Returns:
            (T_test,) float32 — higher = more anomalous.

        Reference inference behavior (QuoVadisTAD `quovadis_tad/model_utils/model_def.py::test_embedder`,
        L423-436 + 407-409): stride=1 sliding windows → per-window per-sensor
        `abs(pred - target)` → median-IQR normalize + 5-box smooth → MAX-over-sensors.
        Reference EXPLICITLY TRUNCATES labels: `gt_labels = labels[input_sequence_length:]`,
        producing a score sequence of length `T_test - seq_len` aligned to t ∈ [seq_len, T_test - 1].

        Per Phase 2.5 user direction: `./comparison/` pipeline cannot truncate labels, so we
        apply Option B fallback — forward-fill the first `seq_len` head timesteps from the
        first valid per-timestep score (applied PER ENTITY). There is no tail gap (next-step
        forecasting with stride=1 covers t ∈ [seq_len, Li - 1] of each entity), so tail
        repeat-last is N/A in the nominal case.

        Boundary-safety (multi-file datasets: SMD / SMAP-MSL / Exathlon):
            RAW PRODUCER (windowing + inference → per-window per-sensor residual) runs INDEPENDENTLY
            on each entity's test slice via the shared single-source-of-truth helper
            (`_boundary_safe_window.entity_test_slices`), so NO window spans an entity boundary.
            WHOLE-TEST POST-PROC (median-IQR over axis=0, box-smooth, MAX-over-sensors) is applied
            ONCE on the concatenated window-residual — EXACTLY as before, granularity UNCHANGED
            (`normalise_scores` uses np.median/iqr over the full window axis: data_utils.py L34-35).
            Note: per_entity_concat returns (n_test,) 1-D and cannot carry the per-sensor dim that
            the whole-test median-IQR needs, so we use its sibling `entity_test_slices` (same module,
            same boundary-safe tiling) and concat the per-window residual MATRIX on axis 0.
            Single-entity / None → ONE entity slice → bit-identical to the legacy path.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        self.model.eval()

        n_samples = len(test_data)
        test_data = np.asarray(test_data, dtype=np.float32)
        slices = entity_test_slices(n_samples, test_segments)

        # --- RAW PRODUCER (per-entity, boundary-safe): windowing + inference → residual ---
        # Per entity, record its (lo, hi) and how many windows it contributed, so we can scatter
        # the post-processed per-window scores back to the right timesteps with per-entity head-fill.
        start_time = time.time()
        if self.verbose:
            tag = "1 entity (whole array)" if len(slices) == 1 else f"{len(slices)} entities (boundary-safe)"
            print(f"  Inference (per-entity windowing): {tag}")

        delta_blocks: list = []          # each (n_windows_i, n_sensors)
        entity_meta: list = []           # (lo, hi, n_windows_i)
        for (lo, hi) in slices:
            delta_i = self._infer_window_residuals(test_data[lo:hi])
            delta_blocks.append(delta_i)
            entity_meta.append((lo, hi, len(delta_i)))

        if self.verbose:
            total_windows = int(sum(m[2] for m in entity_meta))
            print(f"\n  Inference complete: {time.time() - start_time:.1f}s "
                  f"({total_windows:,} windows over {len(slices)} entity slice(s))")

        # Concatenate the per-window per-sensor residuals across entities (axis 0 = window axis).
        # Single-entity → this is exactly the legacy `delta`.
        delta = (np.concatenate(delta_blocks, axis=0).astype(np.float32)
                 if delta_blocks else np.zeros((0, self.n_features), dtype=np.float32))

        if len(delta) == 0:
            # No valid windows anywhere (every entity shorter than seq_len, or empty test).
            return np.zeros(n_samples, dtype=np.float32)

        # --- WHOLE-TEST POST-PROC (applied ONCE, granularity UNCHANGED) ---
        # per-sensor median-IQR normalize + box smooth (reference normalise_scores)
        smoothed = self._normalise_scores(
            delta,
            norm=self.score_norm,
            smooth=True,
            smooth_window=self.score_smooth_window,
            epsilon=self.score_iqr_epsilon,
        )
        # Aggregate over sensors: MAX  (reference notebook cells 13/16/19: `.max(1)`)
        per_window_score = smoothed.max(axis=-1).astype(np.float32)  # (total_windows,)

        # --- Map per-window scores back to (T_test,), PER ENTITY (head-fill per entity) ---
        # For entity (lo, hi) with n_windows_i windows: window j → timestep lo + seq_len + j,
        # covering t ∈ [lo + seq_len, hi - 1]; the first `seq_len` head positions [lo, lo+seq_len)
        # forward-fill from that entity's first valid score (Option B, applied per entity).
        scores = np.empty(n_samples, dtype=np.float32)
        cursor = 0
        for (lo, hi, n_windows_i) in entity_meta:
            if n_windows_i == 0:
                # Degenerate entity (Li <= seq_len): no own valid score. Fill 0.0 (neutral) —
                # NEVER borrow another entity's score (would be cross-entity leakage). In practice
                # SMD/SMAP/Exathlon test entities are >> seq_len=5, so this never triggers.
                scores[lo:hi] = 0.0
                continue
            block = per_window_score[cursor:cursor + n_windows_i]
            cursor += n_windows_i
            valid_start = lo + self.seq_len            # first index with a valid per-timestep score
            valid_end = valid_start + n_windows_i      # exclusive; nominally == hi
            scores[valid_start:valid_end] = block
            # Option B head fallback (per entity): forward-fill the head seq_len from first valid.
            if valid_start > lo:
                scores[lo:valid_start] = block[0]
            # Defensive tail fallback (never executes in nominal stride=1 next-step forecasting,
            # where valid_end == hi). Kept in case _create_windows returns fewer windows.
            if valid_end < hi:
                scores[valid_end:hi] = block[-1]

        # Final guard
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return scores

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (f"GCNLSTMBaseline(seq_len={self.seq_len}, lstm_units={self.lstm_units}, "
                f"gcn_out_dim={self.gcn_out_dim}, agg={self.aggregation_type}, "
                f"comb={self.combination_type})")

    def save(self, save_dir: Path) -> None:
        if self.model is None:
            raise RuntimeError("No model to save. Call fit() first.")

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), save_dir / "model.pt")

        # Save edges (sparse representation rather than dense adjacency)
        if self.edges is not None and self.edges.numel() > 0:
            np.save(save_dir / "edges.npy", self.edges.cpu().numpy())
        else:
            np.save(save_dir / "edges.npy", np.zeros((2, 0), dtype=np.int64))

        config = {
            "seq_len": self.seq_len,
            "gcn_out_dim": self.gcn_out_dim,
            "lstm_units": self.lstm_units,
            "output_seq_len": self.output_seq_len,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "train_stride": self.train_stride,
            "aggregation_type": self.aggregation_type,
            "combination_type": self.combination_type,
            "graph_conv_activation": self.graph_conv_activation,
            "adj_symmetric": self.adj_symmetric,
            "score_norm": self.score_norm,
            "score_smooth_window": self.score_smooth_window,
            "score_iqr_epsilon": self.score_iqr_epsilon,
            "n_features": self.n_features,
            "train_loss_history": self.train_loss_history,
        }
        with open(save_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

        if self.verbose:
            print(f"  Saved model to {save_dir}")

    def load(self, save_dir: Path) -> 'GCNLSTMBaseline':
        save_dir = Path(save_dir)

        with open(save_dir / "config.json", 'r') as f:
            config = json.load(f)

        self.seq_len = config["seq_len"]
        self.gcn_out_dim = config["gcn_out_dim"]
        self.lstm_units = config["lstm_units"]
        self.output_seq_len = config.get("output_seq_len", 1)
        self.aggregation_type = config.get("aggregation_type", "max")
        self.combination_type = config.get("combination_type", "concat")
        self.graph_conv_activation = config.get("graph_conv_activation", "relu")
        self.adj_symmetric = config.get("adj_symmetric", True)
        self.score_norm = config.get("score_norm", "median-iqr")
        self.score_smooth_window = config.get("score_smooth_window", 5)
        self.score_iqr_epsilon = config.get("score_iqr_epsilon", 1e-2)
        self.n_features = config["n_features"]
        self.train_loss_history = config.get("train_loss_history", [])

        edges_np = np.load(save_dir / "edges.npy")
        self.edges = torch.from_numpy(edges_np.astype(np.int64))

        self.model = GCNLSTM(
            n_features=self.n_features,
            gcn_out_dim=self.gcn_out_dim,
            lstm_units=self.lstm_units,
            output_seq_len=self.output_seq_len,
            aggregation_type=self.aggregation_type,
            combination_type=self.combination_type,
            graph_conv_activation=self.graph_conv_activation,
        ).to(self.device)
        self.model.set_graph(self.edges, self.n_features)
        self.model.load_state_dict(torch.load(save_dir / "model.pt", map_location=self.device))
        self.model.eval()

        if self.verbose:
            print(f"  Loaded model from {save_dir}")
        return self
