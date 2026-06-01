"""
GDN - Graph Deviation Network

Adapted from https://github.com/d-ailin/GDN (MIT License, branch: main, fetched 2026-05-24).
Paper: Deng & Hooi, "Graph Neural Network-Based Anomaly Detection in Multivariate
Time Series" (AAAI 2021).

Architecture (paper-faithful, audit 12_FINAL_MODEL_REPORT.md §9 준수):
  1. Learnable per-sensor embedding (nn.Embedding)
  2. Cosine-similarity + top-k over node embeddings → dynamic adjacency
     (gradient detached during graph construction, gradient flows through
     GraphLayer's `message` path via embedding injection)
  3. GraphLayer (paper-faithful re-implementation of the upstream
     graph_layer.py): embedding-conditioned multi-head attention with
     att_em_i / att_em_j parameters, scatter softmax over destination nodes,
     remove_self_loops + add_self_loops. Edge direction: neighbor → self.
  4. Output head: elementwise multiply with embedding → bn_outlayer_in (BN+ReLU)
     → Dropout(0.2) → OutLayer(layer_num=1) = single Linear(embed_dim, 1).
  5. Score formula (predict()):  per-sensor |pred - gt| → robust normalize
     (delta - median) / (|IQR| + 1e-2) → 4-window moving average → MAX over
     sensor axis → timestamp-level scalar (audit §6.1+§6.2+§6.4).

Implementation note: upstream uses torch_geometric. `dc_vis` conda env does NOT
have torch_geometric installed, and policy `feedback_env_install_caution`
forbids installation. We implement native scatter softmax / scatter add via
PyTorch primitives (~30 LOC, see _scatter_softmax / _scatter_add helpers).
"""

import math
import time
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from comparison.segment_utils import compute_segment_safe_window_indices
from comparison.baselines._boundary_safe_window import entity_test_slices
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Native PyG-equivalent helpers (avoid torch_geometric dependency)
# ----------------------------------------------------------------------------

def _remove_self_loops(edge_index: torch.Tensor) -> torch.Tensor:
    """Drop edges where source == destination. Matches torch_geometric.utils.remove_self_loops."""
    mask = edge_index[0] != edge_index[1]
    return edge_index[:, mask]


def _add_self_loops(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Append (i,i) for i in [0, num_nodes). Matches torch_geometric.utils.add_self_loops."""
    device = edge_index.device
    self_idx = torch.arange(num_nodes, device=device, dtype=edge_index.dtype)
    loop = torch.stack([self_idx, self_idx], dim=0)  # (2, N)
    return torch.cat([edge_index, loop], dim=1)


def _scatter_softmax(src: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    softmax over `src` (shape (E, H)) grouped by `index` (shape (E,)).
    For each group (destination node), normalises the slice of src so that
    sum-over-edges-with-that-destination equals 1.

    Equivalent to torch_geometric.utils.softmax(src, index, num_nodes=num_nodes)
    for the case where index is the destination index of each edge.

    Implementation note: uses out-of-place ops so autograd works correctly
    (in-place scatter_reduce_ on the autograd graph caused version-mismatch
    errors during backward).
    """
    # Numerical stability: subtract per-group max (computed without grad — only
    # used for shifting, gradient flows through src in src_shift)
    src_detached = src.detach()
    src_max = torch.full((num_nodes, src.size(1)), -1e30,
                        dtype=src.dtype, device=src.device)
    src_max.scatter_reduce_(0, index.unsqueeze(1).expand_as(src_detached),
                            src_detached, reduce='amax', include_self=True)
    # Avoid -inf for groups with no edges (shouldn't happen given self-loops, but be safe)
    src_max = torch.where(src_max <= -1e29, torch.zeros_like(src_max), src_max)
    src_shift = src - src_max[index]
    src_exp = src_shift.exp()
    # Sum per group (out-of-place new zero tensor avoids in-place issues)
    denom = torch.zeros((num_nodes, src.size(1)), dtype=src.dtype, device=src.device)
    denom = denom.scatter_add(0, index.unsqueeze(1).expand_as(src_exp), src_exp)
    return src_exp / (denom[index] + 1e-16)


def _scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int, dim_size: int) -> torch.Tensor:
    """
    scatter-add src into a new tensor with shape[dim] = dim_size.
    Other dims preserved. `index` shape == src.shape[dim].
    """
    out_shape = list(src.shape)
    out_shape[dim] = dim_size
    out = src.new_zeros(out_shape)
    # broadcast index to src shape
    idx_shape = [1] * src.dim()
    idx_shape[dim] = -1
    index_b = index.view(*idx_shape).expand_as(src)
    out.scatter_add_(dim, index_b, src)
    return out


def _get_batch_edge_index(org_edge_index: torch.Tensor,
                          batch_num: int,
                          node_num: int) -> torch.Tensor:
    """
    Replicate edge_index `batch_num` times, offsetting node indices by
    `i*node_num` for batch i. Matches upstream models/GDN.py:get_batch_edge_index.
    """
    edge_num = org_edge_index.shape[1]
    out = org_edge_index.repeat(1, batch_num).contiguous().clone()
    for i in range(batch_num):
        out[:, i*edge_num:(i+1)*edge_num] += i * node_num
    return out.long()


# ----------------------------------------------------------------------------
# GraphLayer (paper-faithful re-implementation of upstream graph_layer.py)
# ----------------------------------------------------------------------------

class GraphLayer(nn.Module):
    """
    Embedding-conditioned multi-head graph attention layer (GDN, AAAI 2021).

    Forward signature mirrors upstream models/graph_layer.py:
        forward(x, edge_index, embedding)
    where edge_index row 0 = source (neighbor), row 1 = destination (self).
    """

    def __init__(self, in_channels: int, out_channels: int, heads: int = 1,
                 concat: bool = False, negative_slope: float = 0.2,
                 dropout: float = 0.0, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        # Linear projection of node features
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)

        # Attention parameters — split into "feature" and "embedding" halves,
        # concatenated at message-time to form a 2*out_channels-wide attention vector.
        self.att_i = nn.Parameter(torch.empty(1, heads, out_channels))
        self.att_j = nn.Parameter(torch.empty(1, heads, out_channels))
        self.att_em_i = nn.Parameter(torch.empty(1, heads, out_channels))
        self.att_em_j = nn.Parameter(torch.empty(1, heads, out_channels))

        if bias and concat:
            self.bias = nn.Parameter(torch.empty(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # glorot init equivalent (torch_geometric.nn.inits.glorot)
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_i)
        nn.init.xavier_uniform_(self.att_j)
        # upstream initialises att_em_i / att_em_j to zeros
        nn.init.zeros_(self.att_em_i)
        nn.init.zeros_(self.att_em_j)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, in_channels) flat node features (batch * node, slide_win)
            edge_index: (2, E) — row 0 = source (neighbor j), row 1 = dest (self i)
            embedding: (N, embed_dim) flat per-node embedding (batch-replicated)

        Returns:
            (N, out_channels) if concat=False (mean over heads),
            (N, heads * out_channels) if concat=True.
        """
        N = x.size(0)
        x = self.lin(x)  # (N, heads * out_channels)

        # remove + add self loops (canonical upstream behaviour)
        edge_index = _remove_self_loops(edge_index)
        edge_index = _add_self_loops(edge_index, num_nodes=N)

        src, dst = edge_index[0], edge_index[1]  # src = j (neighbor), dst = i (self)

        # gather per-edge features
        x_i = x[dst].view(-1, self.heads, self.out_channels)  # target/self features
        x_j = x[src].view(-1, self.heads, self.out_channels)  # source/neighbor features

        # gather per-edge embeddings (mirroring upstream message())
        # upstream: embedding_i = embedding[edge_index_i], embedding_j = embedding[edges[0]]
        emb_i = embedding[dst].unsqueeze(1).repeat(1, self.heads, 1)  # (E, heads, embed_dim)
        emb_j = embedding[src].unsqueeze(1).repeat(1, self.heads, 1)

        # embedding-conditioned key
        key_i = torch.cat([x_i, emb_i], dim=-1)  # (E, heads, 2*out_channels)  (assumes embed_dim == out_channels)
        key_j = torch.cat([x_j, emb_j], dim=-1)

        cat_att_i = torch.cat([self.att_i, self.att_em_i], dim=-1)  # (1, heads, 2*out_channels)
        cat_att_j = torch.cat([self.att_j, self.att_em_j], dim=-1)

        # attention logit
        alpha = (key_i * cat_att_i).sum(-1) + (key_j * cat_att_j).sum(-1)  # (E, heads)
        alpha = F.leaky_relu(alpha, self.negative_slope)

        # scatter softmax: normalise per destination node (sum over incoming edges = 1)
        alpha = _scatter_softmax(alpha, dst, num_nodes=N)  # (E, heads)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # weighted message: x_j * alpha
        msg = x_j * alpha.unsqueeze(-1)  # (E, heads, out_channels)

        # aggregate (sum) per destination
        out = _scatter_add(msg, dst, dim=0, dim_size=N)  # (N, heads, out_channels)

        if self.concat:
            out = out.view(N, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)  # (N, out_channels)

        if self.bias is not None:
            out = out + self.bias

        return out


# ----------------------------------------------------------------------------
# GNNLayer + OutLayer wrappers (mirror upstream models/GDN.py)
# ----------------------------------------------------------------------------

class _GNNLayer(nn.Module):
    """Wraps GraphLayer + BN + ReLU (mirrors models/GDN.py:GNNLayer)."""

    def __init__(self, in_channel: int, out_channel: int, heads: int = 1):
        super().__init__()
        # upstream uses concat=False so output is (N, out_channel) regardless of heads
        self.gnn = GraphLayer(in_channel, out_channel, heads=heads, concat=False)
        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                embedding: torch.Tensor) -> torch.Tensor:
        out = self.gnn(x, edge_index, embedding)  # (N, out_channel)
        out = self.bn(out)
        return self.relu(out)


class _OutLayer(nn.Module):
    """
    Mirrors models/GDN.py:OutLayer.
    With layer_num=1: a single Linear(in_num, 1).
    With layer_num>1: stack of Linear→BN→ReLU blocks (last → Linear(*, 1)).
    BatchNorm applied along the embed_dim (permute to (B, dim, N)).
    """

    def __init__(self, in_num: int, layer_num: int, inter_num: int = 128):
        super().__init__()
        modules = []
        for i in range(layer_num):
            if i == layer_num - 1:
                modules.append(nn.Linear(in_num if layer_num == 1 else inter_num, 1))
            else:
                layer_in_num = in_num if i == 0 else inter_num
                modules.append(nn.Linear(layer_in_num, inter_num))
                modules.append(nn.BatchNorm1d(inter_num))
                modules.append(nn.ReLU())
        self.mlp = nn.ModuleList(modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, in_num)
        out = x
        for mod in self.mlp:
            if isinstance(mod, nn.BatchNorm1d):
                out = out.permute(0, 2, 1)
                out = mod(out)
                out = out.permute(0, 2, 1)
            else:
                out = mod(out)
        return out


# ----------------------------------------------------------------------------
# GDNModel (paper-faithful)
# ----------------------------------------------------------------------------

class GDNModel(nn.Module):
    """
    Graph Deviation Network model (paper-faithful).

    Mirrors upstream models/GDN.py:GDN with edge_set_num = 1 (single fully-
    connected prior graph collapsed to the dynamic cosine-topk graph computed
    inside forward()).
    """

    def __init__(
        self,
        n_features: int,
        seq_len: int = 5,
        embed_dim: int = 64,
        out_layer_inter_dim: int = 128,
        out_layer_num: int = 1,
        n_heads: int = 1,
        top_k: int = 5,
        output_dropout: float = 0.2,
    ):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len     # = slide_win in upstream
        self.embed_dim = embed_dim
        self.top_k = min(top_k, n_features - 1)

        # Per-sensor learnable embedding
        self.embedding = nn.Embedding(n_features, embed_dim)
        self.bn_outlayer_in = nn.BatchNorm1d(embed_dim)

        # Single GNN layer (edge_set_num = 1).
        # Upstream: GNNLayer(input_dim=slide_win, dim, inter_dim=dim+embed_dim, heads=1).
        # `inter_dim` is unused in upstream GraphLayer beyond informational naming
        # (the actual concat dim is out_channels + embed_dim).
        self.gnn = _GNNLayer(seq_len, embed_dim, heads=n_heads)

        # Output head: OutLayer takes (B, N, embed_dim) → (B, N, 1)
        self.out_layer = _OutLayer(embed_dim, out_layer_num, inter_num=out_layer_inter_dim)

        # Hard-coded 0.2 dropout in upstream (models/GDN.py:101)
        self.dp = nn.Dropout(output_dropout)

        # Init embedding identical to upstream (kaiming_uniform with a=sqrt(5))
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)  — driver convention.

        Returns:
            (batch, n_features)  — 1-step forecast per sensor.
        """
        # Upstream expects (batch, n_features, slide_win)
        x = x.transpose(1, 2).contiguous()  # → (batch, n_features, seq_len)
        batch_num, node_num, all_feature = x.shape
        x_flat = x.view(-1, all_feature).contiguous()  # (batch * N, slide_win)

        device = x.device

        # ---- dynamic graph: cosine sim + topk over learnable node embeddings
        all_embeddings = self.embedding(torch.arange(node_num, device=device))
        weights = all_embeddings.detach().clone()  # detach: graph construction does not
                                                   # propagate gradient (matches upstream)
        cos_ji = weights @ weights.T
        norms = weights.norm(dim=-1)
        norm_outer = norms.view(-1, 1) @ norms.view(1, -1)
        cos_ji = cos_ji / (norm_outer + 1e-12)
        topk_idx = torch.topk(cos_ji, self.top_k, dim=-1)[1]  # (N, k)

        # neighbor→self edge direction (audit §3.3 — paper-faithful)
        gated_i = torch.arange(node_num, device=device).unsqueeze(1) \
                       .repeat(1, self.top_k).flatten().unsqueeze(0)
        gated_j = topk_idx.flatten().unsqueeze(0)
        gated_edge_index = torch.cat([gated_j, gated_i], dim=0)  # (2, N*k)  src=j, dst=i

        # batch-replicate edge_index with offsets
        batch_edge_index = _get_batch_edge_index(gated_edge_index, batch_num, node_num).to(device)
        batch_embedding = all_embeddings.repeat(batch_num, 1)  # (batch*N, embed_dim)

        # ---- message passing
        h = self.gnn(x_flat, batch_edge_index, embedding=batch_embedding)  # (batch*N, embed_dim)
        h = h.view(batch_num, node_num, -1)  # (batch, N, embed_dim)

        # ---- output head: multiplicative embedding gate → BN + ReLU → Dropout → OutLayer
        indexes = torch.arange(node_num, device=device)
        out = torch.mul(h, self.embedding(indexes))  # (batch, N, embed_dim)
        out = out.permute(0, 2, 1).contiguous()      # (batch, embed_dim, N) for BN1d on dim axis
        out = F.relu(self.bn_outlayer_in(out))
        out = out.permute(0, 2, 1).contiguous()      # (batch, N, embed_dim)
        out = self.dp(out)
        out = self.out_layer(out)                    # (batch, N, 1)
        out = out.view(-1, node_num)                 # (batch, N)
        return out


# ----------------------------------------------------------------------------
# Wrapper
# ----------------------------------------------------------------------------

class GDNBaseline:
    """
    GDN baseline wrapper for time series anomaly detection.

    fit(train_X, epoch_callback=None):  Adam(lr=0.001), MSE loss, 1-step forecast.
    predict(test_X): per-sensor |pred-gt| → robust normalize → 4-window MA → MAX.
    """

    def __init__(
        self,
        seq_len: int = 5,
        embed_dim: int = 64,
        gnn_out_dim: int = 64,   # kept for backwards compatibility; in upstream
                                 # GNNLayer out_channel == embed_dim (dim).
                                 # When != embed_dim, embed_dim is used for the head.
        n_heads: int = 1,
        top_k: int = 5,
        out_layer_inter_dim: int = 128,
        out_layer_num: int = 1,
        output_dropout: float = 0.2,
        dropout: float = 0.0,   # NOTE: legacy kwarg, mapped to output_dropout if explicit
        lr: float = 0.001,
        batch_size: int = 256,
        epochs: int = 10,
        train_stride: int = 1,
        device: Optional[str] = None,
        verbose: bool = True,
        **kwargs,  # accept future hyperparameter additions
    ):
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        # In paper-faithful form gnn_out_dim must equal embed_dim (out_channel = dim).
        # We honour an explicit non-default gnn_out_dim by warning but using embed_dim.
        if gnn_out_dim != embed_dim and verbose:
            print(f"  [gdn] warning: gnn_out_dim ({gnn_out_dim}) != embed_dim ({embed_dim}); "
                  f"upstream ties them equal — using embed_dim for paper-faithful behaviour.")
        self.gnn_out_dim = embed_dim
        self.n_heads = n_heads
        self.top_k = top_k
        self.out_layer_inter_dim = out_layer_inter_dim
        self.out_layer_num = out_layer_num
        # `dropout` was the legacy kwarg name; honour it if user passed it explicitly.
        # Default behaviour: output_dropout=0.2 (upstream hard-coded).
        self.output_dropout = output_dropout if output_dropout != 0.2 or dropout == 0.0 \
                              else dropout
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
        self.train_loss_history: list = []
        # Path A: no internal scaler (driver normalises).
        self.scaler = None

    @property
    def name(self) -> str:
        return "GDN"

    # ------------------------------------------------------------------
    # Window construction (matches upstream datasets/TimeDataset.py)
    # ------------------------------------------------------------------

    def _create_windows(self, data: np.ndarray, stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forecasting windows: history = data[i:i+seq_len], target = data[i+seq_len].
        Mirrors upstream:  range(slide_win, total_time_len, slide_stride)
                           ft = data[:, i-slide_win:i], tar = data[:, i].
        """
        n_samples, n_features = data.shape
        n_windows = (n_samples - self.seq_len - 1) // stride + 1
        if n_windows <= 0:
            raise ValueError(f"data too short for seq_len={self.seq_len} (got T={n_samples})")

        windows = np.zeros((n_windows, self.seq_len, n_features), dtype=np.float32)
        targets = np.zeros((n_windows, n_features), dtype=np.float32)

        for idx, i in enumerate(range(0, n_samples - self.seq_len, stride)):
            if idx >= n_windows:
                break
            windows[idx] = data[i:i + self.seq_len]
            targets[idx] = data[i + self.seq_len]

        return windows, targets

    # ------------------------------------------------------------------
    # fit / predict
    # ------------------------------------------------------------------

    def fit(self, train_data: np.ndarray, epoch_callback=None, train_segments=None) -> 'GDNBaseline':
        """
        Train GDN on driver-normalised training data.

        Args:
            train_data: (T_train, n_features) float32, already minmax/zscore-normalised by driver.
            epoch_callback: optional callable(self, ep_idx) invoked after each epoch.
        """
        self.n_features = train_data.shape[1]

        self.model = GDNModel(
            n_features=self.n_features,
            seq_len=self.seq_len,
            embed_dim=self.embed_dim,
            out_layer_inter_dim=self.out_layer_inter_dim,
            out_layer_num=self.out_layer_num,
            n_heads=self.n_heads,
            top_k=self.top_k,
            output_dropout=self.output_dropout,
        ).to(self.device)

        if self.verbose:
            print(f"{self.name} training:")
            print(f"  Device: {self.device}")
            print(f"  Input shape: ({self.seq_len}, {self.n_features})")
            print(f"  Graph: {self.n_features} nodes, top-{self.model.top_k} neighbors (neighbor→self)")
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"  Total parameters: {total_params:,}")

        windows, targets = self._create_windows(train_data, stride=self.train_stride)
        if train_segments is not None:
            # GDN predicts data[s + seq_len] from data[s : s + seq_len], so the
            # target step must also remain inside the segment → extra_horizon=1.
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

        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(windows), torch.from_numpy(targets),
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=True,
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()  # reduction='mean' — matches upstream train.py:21

        self.train_loss_history = []
        self.model.train()
        start_time = time.time()

        n_batches = len(dataloader)
        for epoch in range(self.epochs):
            epoch_start = time.time()
            epoch_losses = []
            for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
                batch_start = time.time()
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = criterion(pred, batch_y)
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
                      f"Loss: {avg_loss:.6f} | Time: {epoch_time:.1f}s/epoch | "
                      f"ETA: {remaining:.0f}s", end="")
                sys.stdout.flush()

            if epoch_callback is not None:
                epoch_callback(self, epoch + 1)
                self.model.train()

        if self.verbose:
            total_time = time.time() - start_time
            print(f"\n  Training complete: {total_time:.1f}s total, "
                  f"Final loss: {self.train_loss_history[-1]:.6f}")

        return self

    # ------------------------------------------------------------------
    # RAW per-entity window producer (boundary-safe).
    #
    # This is the *raw score producer* in the sense of _boundary_safe_window:
    # windowing (stride=1) + model inference + per-window |pred-gt| for ONE
    # entity's test slice in ISOLATION, so no window's seq_len rows can ever span
    # an entity boundary. It returns the per-WINDOW (n_win, F) per-sensor absolute
    # error AND the GLOBAL target timestep of each window (i+seq_len, offset by the
    # entity's lo). It does NOT apply any whole-test post-processing — the per-sensor
    # median/IQR, the 4-window MA and the MAX over sensors are applied ONCE on the
    # CONCATENATED per-window delta in predict(), so post-proc granularity is UNCHANGED.
    # ------------------------------------------------------------------
    def _raw_window_delta(self, sub_X: np.ndarray, lo: int):
        """Per-entity raw producer.

        Returns (win_delta, tgt_idx, head_target):
          win_delta  : (n_win, F) per-window per-sensor |pred-gt| (float64), window order.
          tgt_idx    : (n_win,)   GLOBAL target timestep of each window = lo + i + seq_len.
          head_target: GLOBAL index range end for the entity's head fill = lo + seq_len
                       (head timesteps [lo, lo+seq_len) lack a forecast → filled from this
                       entity's first window score downstream; bounded to the entity).
        Edge-safe: an entity slice shorter than seq_len+1 (no full window) yields an EMPTY
        per-window delta (n_win=0); its head fill is handled downstream (zeros → finite).
        """
        sub_X = np.asarray(sub_X, dtype=np.float32)
        Li = sub_X.shape[0]
        head_target = lo + min(self.seq_len, Li)

        # Edge case: slice too short for even one forecasting window → no windows.
        if Li < self.seq_len + 1:
            empty = np.zeros((0, sub_X.shape[1]), dtype=np.float64)
            return empty, np.zeros((0,), dtype=np.int64), head_target

        windows, targets = self._create_windows(sub_X, stride=1)  # boundary-safe: only this entity
        n_windows = len(windows)

        pred_list = []
        for i in range(0, n_windows, self.batch_size):
            batch_x = torch.from_numpy(windows[i:i + self.batch_size]).to(self.device)
            pred = self.model(batch_x).cpu().numpy()  # (B, F)
            pred_list.append(pred)
        preds = np.concatenate(pred_list, axis=0).astype(np.float64)   # (n_windows, F)
        gts = targets.astype(np.float64)                               # (n_windows, F)
        win_delta = np.abs(preds - gts)                                # (n_windows, F)

        # GLOBAL target timestep of window i within this entity = lo + i + seq_len.
        tgt_idx = lo + self.seq_len + np.arange(n_windows, dtype=np.int64)
        return win_delta, tgt_idx, head_target

    @torch.no_grad()
    def predict(self, test_data: np.ndarray, test_segments=None) -> np.ndarray:
        """
        Compute per-timestamp anomaly scores using upstream evaluate.py formula
        (d-ailin/GDN main, fetched 2026-05-24):
            (1) per-sensor absolute error delta = |pred - gt|        (N_windows, F)
            (2) per-sensor robust normalisation (evaluate.py:47-48,
                util/data.py:get_err_median_and_iqr)
                err[:, f] = (delta[:, f] - median_f) / (|iqr_f| + 1e-2)
            (3) per-sensor 4-window moving average (evaluate.py:50-54)
                smoothed[i, f] = mean(err[i-before_num:i+1, f])  for i >= before_num
                smoothed[i<before_num, f] = 0                    (reference boundary: ZEROS,
                                                                  np.zeros(err_scores.shape))
            (4) MAX across sensor axis (main.py: np.max(full_scores, axis=0)) → (N_windows,)
            (5) Map window i → timestamp (i + seq_len), then OPTION-B head fill.

        Boundary-safe TEST windowing (test_segments):
            On multi-entity test arrays (SMD machines / SMAP-MSL channels / Exathlon apps,
            concatenated by the harness), a single sliding window's seq_len history rows must
            NEVER span two entities. We therefore run ONLY the RAW producer — windowing +
            inference + per-window |pred-gt| — INDEPENDENTLY on each entity's test slice
            (_raw_window_delta), then CONCATENATE the per-window deltas (in window order).
            The WHOLE-TEST post-processing — per-sensor median/IQR robust normalisation,
            4-window MA, MAX over sensors — is applied ONCE on that concatenated per-window
            delta, EXACTLY as before, so its granularity is UNCHANGED (median/IQR/MA/MAX over
            the full concatenated per-window error series). Each window's score is then scattered
            to its GLOBAL target timestep, with the per-entity OPTION-B head fill bounded to the
            entity. ``test_segments`` are TEST-LOCAL (lo,hi) per-entity ranges from the same
            source as per-entity normalisation; None / single-entity → ONE entity slice covering
            the whole array == bit-identical legacy behaviour (helper guarantees the no-op).

        Returns:
            scores: np.ndarray shape (T_test,), dtype float32.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        self.model.eval()
        test_data = np.asarray(test_data, dtype=np.float32)
        n_samples = test_data.shape[0]
        n_feat = test_data.shape[1]

        slices = entity_test_slices(n_samples, test_segments)

        # ---- RAW producer (boundary-safe): per-entity windowing + inference + |pred-gt|.
        # Each entity is windowed in isolation, then per-window deltas are concatenated in
        # window order — identical to the legacy single windowing pass when there is one
        # entity (the no-op), and free of cross-entity windows when there are many.
        start_time = time.time()
        win_delta_parts = []      # list of (n_win_i, F) per-window deltas
        tgt_idx_parts = []        # list of (n_win_i,) GLOBAL target timesteps
        head_targets = []         # per-entity (lo, head_end)
        for (lo, hi) in slices:
            wd, tgt, head_end = self._raw_window_delta(test_data[lo:hi], lo)
            head_targets.append((lo, head_end))
            win_delta_parts.append(wd)
            tgt_idx_parts.append(tgt)

        if win_delta_parts and sum(len(p) for p in win_delta_parts) > 0:
            delta = np.concatenate(win_delta_parts, axis=0)            # (N_windows_total, F)
            tgt_idx = np.concatenate(tgt_idx_parts, axis=0)            # (N_windows_total,)
        else:
            # No entity produced any window (every slice shorter than seq_len+1). Return a
            # finite all-zero score of the required length (sensible "no evidence" fallback).
            if self.verbose:
                print("  Inference: no forecastable windows; returning zero scores")
            return np.zeros(n_samples, dtype=np.float32)

        n_windows = len(delta)
        if self.verbose:
            multi = f" across {len(slices)} entities (boundary-safe)" if len(slices) > 1 else ""
            print(f"  Inference complete: {n_windows:,} windows in "
                  f"{time.time() - start_time:.1f}s{multi}")

        # ============================================================
        # WHOLE-TEST POST-PROCESSING (UNCHANGED granularity — applied ONCE on the
        # concatenated per-window delta, NOT per entity). Per-sensor median/IQR over the
        # ENTIRE test-error distribution, MA along the full window axis, MAX over sensors —
        # identical to legacy (when single-entity, this IS the legacy computation verbatim).
        # ============================================================

        # ---- per-sensor robust normalisation over the entire test-error distribution
        # (mirrors upstream util/data.py:get_err_median_and_iqr called per sensor in
        # evaluate.py:get_err_scores)
        median_per_sensor = np.median(delta, axis=0)                    # (F,)
        iqr_per_sensor = scipy.stats.iqr(delta, axis=0)                 # (F,)
        epsilon = 1e-2
        err = (delta - median_per_sensor[None, :]) / (np.abs(iqr_per_sensor)[None, :] + epsilon)  # (N_windows, F)

        # ---- 4-window moving average (before_num = 3)
        # Reference (evaluate.py:50-54) initialises smoothed = np.zeros(...) and only
        # assigns indices i >= before_num. Indices [0, before_num) therefore remain 0.
        before_num = 3
        smoothed = np.zeros_like(err)
        # vectorised moving average over axis 0:
        # cumsum trick — sum of err[i-3:i+1] = cumsum[i+1] - cumsum[i-3]
        cs = np.concatenate([np.zeros((1, n_feat)), np.cumsum(err, axis=0)], axis=0)  # (N+1, F)
        for i in range(before_num, err.shape[0]):
            smoothed[i] = (cs[i + 1] - cs[i - before_num]) / (before_num + 1)

        # ---- MAX across sensor axis → (N_windows,)
        aggregated = np.max(smoothed, axis=1)                           # (N_windows,)

        # ---- map each window → its GLOBAL target timestamp (vectorised scatter).
        # Window k (concatenated order) targets global index tgt_idx[k]; covers each
        # entity's [lo+seq_len, hi-1]. Head indices [lo, lo+seq_len) per entity lack a
        # forecast and are filled below (OPTION B), bounded to the entity (no leak).
        scores = np.zeros(n_samples, dtype=np.float32)
        valid = tgt_idx < n_samples
        scores[tgt_idx[valid]] = aggregated[valid].astype(np.float32)

        # ---- OPTION-B head fill, per entity (bounded — never reaches across a boundary).
        # This mirrors the legacy head policy VERBATIM, applied to each entity's own global
        # window-slot range [w0, w0+n_win_i) and head start `lo` (head_end = lo+seq_len):
        #   legacy, n_windows>before_num : scores[0 : seq_len+before_num] = aggregated[before_num]
        #   legacy, 0<n_windows<=before_num: scores[0 : seq_len]          = aggregated[0]
        # For a single entity (lo=0, head_end=seq_len, w0=0) this is bit-identical to legacy.
        cursor = 0
        for part_i, (lo, head_end) in enumerate(head_targets):
            n_win_i = len(win_delta_parts[part_i])
            if n_win_i == 0:
                # Entity produced no window; its whole slice [lo, hi) stays 0 (finite).
                continue
            w0 = cursor                      # this entity's first global window slot
            cursor += n_win_i
            if n_win_i > before_num:
                first_valid = aggregated[w0 + before_num]
                fill_end = min(head_end + before_num, n_samples)
            else:
                first_valid = aggregated[w0]
                fill_end = min(head_end, n_samples)
            scores[lo:fill_end] = np.float32(first_valid)

        return scores

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------

    def save(self, save_dir) -> None:
        if self.model is None:
            raise RuntimeError("No model to save.")
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), save_dir / "model.pt")
        config = {
            "seq_len": self.seq_len,
            "embed_dim": self.embed_dim,
            "n_heads": self.n_heads,
            "top_k": self.top_k,
            "out_layer_inter_dim": self.out_layer_inter_dim,
            "out_layer_num": self.out_layer_num,
            "output_dropout": self.output_dropout,
            "n_features": self.n_features,
            "model_name": self.name,
        }
        with open(save_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

    def load(self, save_dir) -> 'GDNBaseline':
        save_dir = Path(save_dir)
        with open(save_dir / "config.json", 'r') as f:
            config = json.load(f)
        self.seq_len = config["seq_len"]
        self.embed_dim = config["embed_dim"]
        self.n_heads = config["n_heads"]
        self.top_k = config["top_k"]
        self.out_layer_inter_dim = config.get("out_layer_inter_dim", 128)
        self.out_layer_num = config.get("out_layer_num", 1)
        self.output_dropout = config.get("output_dropout", 0.2)
        self.n_features = config["n_features"]
        self.model = GDNModel(
            n_features=self.n_features,
            seq_len=self.seq_len,
            embed_dim=self.embed_dim,
            out_layer_inter_dim=self.out_layer_inter_dim,
            out_layer_num=self.out_layer_num,
            n_heads=self.n_heads,
            top_k=self.top_k,
            output_dropout=self.output_dropout,
        ).to(self.device)
        self.model.load_state_dict(torch.load(save_dir / "model.pt", map_location=self.device))
        self.model.eval()
        return self

    def __repr__(self) -> str:
        return (f"GDNBaseline(seq_len={self.seq_len}, embed_dim={self.embed_dim}, "
                f"top_k={self.top_k}, out_layer_num={self.out_layer_num})")
