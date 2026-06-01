"""
MEMTO Model — Memory-guided Transformer for Multivariate Time Series Anomaly Detection

Based on: "MEMTO: Memory-guided Transformer for Multivariate Time Series Anomaly Detection"
Paper: NeurIPS 2023, https://arxiv.org/abs/2312.02530
Original code: https://github.com/gunny97/MEMTO (no LICENSE — attribution only)

Modified for:
- Device-agnostic operation (removed hardcoded `.cuda()` calls in embedding/memory module)
- Single-file vendoring (merged model/{Transformer,attn_layer,embedding,ours_memory_module,loss_functions}.py)
- K-means init uses sklearn.cluster.KMeans (CPU-based) instead of kmeans-pytorch (avoids extra dep)
- 2-phase training orchestrated by wrapper
"""

import math
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# Embedding (vendored from model/embedding.py)
# ===========================================================================


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros((max_len, d_model), dtype=torch.float)
        pe.requires_grad = False
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        _2i = torch.arange(0, d_model, step=2).float()
        pe[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        pe[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, in_dim, d_model):
        super(TokenEmbedding, self).__init__()
        pad = 1 if torch.__version__ >= "1.5.0" else 2
        self.conv = nn.Conv1d(
            in_channels=in_dim,
            out_channels=d_model,
            kernel_size=3,
            padding=pad,
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x):
        x = self.conv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class InputEmbedding(nn.Module):
    def __init__(self, in_dim, d_model, dropout=0.0):
        super(InputEmbedding, self).__init__()
        self.token_embedding = TokenEmbedding(in_dim=in_dim, d_model=d_model)
        self.pos_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # PositionalEmbedding output is on the buffer's device (same as model), no .cuda() needed
        return self.dropout(self.token_embedding(x) + self.pos_embedding(x))


# ===========================================================================
# Attention (vendored from model/attn_layer.py)
# ===========================================================================


class Attention(nn.Module):
    def __init__(self, window_size, mask_flag=False, scale=None, dropout=0.0):
        super(Attention, self).__init__()
        self.window_size = window_size
        self.mask_flag = mask_flag
        self.scale = scale
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, queries, keys, values, attn_mask=None):
        N, L, Head, C = queries.shape
        scale = self.scale if self.scale is not None else 1.0 / sqrt(C)
        attn_scores = torch.einsum("nlhd,nshd->nhls", queries, keys)
        attn_weights = self.dropout(torch.softmax(scale * attn_scores, dim=-1))
        updated_values = torch.einsum("nhls,nshd->nlhd", attn_weights, values)
        return updated_values.contiguous()


class AttentionLayer(nn.Module):
    def __init__(
        self,
        window_size,
        d_model,
        n_heads,
        d_keys=None,
        d_values=None,
        mask_flag=False,
        scale=None,
        dropout=0.0,
    ):
        super(AttentionLayer, self).__init__()
        self.d_keys = d_keys if d_keys is not None else (d_model // n_heads)
        self.d_values = d_values if d_values is not None else (d_model // n_heads)
        self.n_heads = n_heads
        self.d_model = d_model

        self.W_Q = nn.Linear(self.d_model, self.n_heads * self.d_keys)
        self.W_K = nn.Linear(self.d_model, self.n_heads * self.d_keys)
        self.W_V = nn.Linear(self.d_model, self.n_heads * self.d_values)
        self.out_proj = nn.Linear(self.n_heads * self.d_values, self.d_model)

        self.attn = Attention(window_size=window_size, mask_flag=mask_flag, scale=scale, dropout=dropout)

    def forward(self, input):
        N, L, _ = input.shape
        Q = self.W_Q(input).contiguous().view(N, L, self.n_heads, -1)
        K = self.W_K(input).contiguous().view(N, L, self.n_heads, -1)
        V = self.W_V(input).contiguous().view(N, L, self.n_heads, -1)
        updated_V = self.attn(Q, K, V)
        out = updated_V.view(N, L, -1)
        return self.out_proj(out)


# ===========================================================================
# Transformer Encoder / Decoder (vendored from model/Transformer.py)
# ===========================================================================


class EncoderLayer(nn.Module):
    def __init__(self, attn, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff if d_ff is not None else 4 * d_model
        self.attn_layer = attn
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        out = self.attn_layer(x)
        x = x + self.dropout(out)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x):
        for attn_layer in self.attn_layers:
            x = attn_layer(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


class Decoder(nn.Module):
    """Linear decoder (weak_decoder in upstream — d_model*2 → c_out)."""

    def __init__(self, d_model, c_out, d_ff=None, activation="relu", dropout=0.1):
        super(Decoder, self).__init__()
        self.out_linear = nn.Linear(d_model, c_out)

    def forward(self, x):
        # x: (N, L, d_model)  -> (N, L, c_out)
        return self.out_linear(x)


# ===========================================================================
# Memory Module (vendored from model/ours_memory_module.py, device-agnostic)
# ===========================================================================


class MemoryModule(nn.Module):
    """Gated memory module that maintains M memory items (cluster prototypes).

    Memory items are stored as a buffer so they move with `.to(device)`.
    Init options:
      - `memory_init_embedding=None`, `phase_type='first_train'`: random unit-norm init
      - `memory_init_embedding=<Tensor>`, `phase_type='second_train'`: use provided centers
    """

    def __init__(
        self,
        n_memory,
        fea_dim,
        shrink_thres=0.0025,
        memory_init_embedding=None,
        phase_type=None,
    ):
        super(MemoryModule, self).__init__()
        self.n_memory = n_memory
        self.fea_dim = fea_dim
        self.shrink_thres = shrink_thres
        self.phase_type = phase_type

        self.U = nn.Linear(fea_dim, fea_dim)
        self.W = nn.Linear(fea_dim, fea_dim)

        if memory_init_embedding is None:
            # First phase or test (random init)
            mem_init = F.normalize(
                torch.rand((self.n_memory, self.fea_dim), dtype=torch.float), dim=1
            )
        else:
            # Second phase (cluster centers)
            mem_init = memory_init_embedding.float()
        # Register as buffer (not Parameter — memory items are not directly optimized via backprop;
        # they are updated via the gate in the forward pass)
        self.register_buffer("mem", mem_init)

    def hard_shrink_relu(self, input, lambd=0.0025, epsilon=1e-12):
        output = (F.relu(input - lambd) * input) / (torch.abs(input - lambd) + epsilon)
        return output

    def get_attn_score(self, query, key):
        attn = torch.matmul(query, key.t())  # TxC × CxM = TxM
        attn = F.softmax(attn, dim=-1)
        if self.shrink_thres > 0:
            attn = self.hard_shrink_relu(attn, self.shrink_thres)
            attn = F.normalize(attn, p=1, dim=1)
        return attn

    def read(self, query):
        attn = self.get_attn_score(query, self.mem.detach())  # T x M
        add_memory = torch.matmul(attn, self.mem.detach())  # T x C
        read_query = torch.cat((query, add_memory), dim=1)  # T x 2C
        return {"output": read_query, "attn": attn}

    def update(self, query):
        attn = self.get_attn_score(self.mem, query.detach())  # M x T
        add_mem = torch.matmul(attn, query.detach())  # M x C
        update_gate = torch.sigmoid(self.U(self.mem) + self.W(add_mem))  # M x C
        # Update buffer in-place
        new_mem = (1 - update_gate) * self.mem + update_gate * add_mem
        self.mem.data = new_mem.detach()

    def forward(self, query):
        s = query.data.shape
        l = len(s)
        query = query.contiguous().view(-1, s[-1])  # T x C

        if self.phase_type != "test":
            self.update(query)

        outs = self.read(query)
        read_query, attn = outs["output"], outs["attn"]

        if l == 2:
            pass
        elif l == 3:
            read_query = read_query.view(s[0], s[1], 2 * s[2])
            attn = attn.view(s[0], s[1], self.n_memory)
        else:
            raise TypeError("Wrong input dimension")

        return {"output": read_query, "attn": attn, "memory_init_embedding": self.mem}


# ===========================================================================
# Loss helpers (vendored from model/loss_functions.py)
# ===========================================================================


class GatheringLoss(nn.Module):
    """MSE between query and its nearest memory item."""

    def __init__(self, reduce=True):
        super(GatheringLoss, self).__init__()
        self.reduce = reduce

    @staticmethod
    def get_score(query, key):
        score = torch.matmul(query, key.t())  # TxM
        score = F.softmax(score, dim=1)
        return score

    def forward(self, queries, items):
        batch_size = queries.size(0)
        d_model = queries.size(-1)
        loss_mse = nn.MSELoss(reduction="mean" if self.reduce else "none")

        queries = queries.contiguous().view(-1, d_model)  # T x C
        score = self.get_score(queries, items)
        _, indices = torch.topk(score, 1, dim=1)
        gathering_loss = loss_mse(queries, items[indices].squeeze(1))

        if self.reduce:
            return gathering_loss
        gathering_loss = torch.sum(gathering_loss, dim=-1)
        gathering_loss = gathering_loss.contiguous().view(batch_size, -1)
        return gathering_loss


class EntropyLoss(nn.Module):
    def __init__(self, eps=1e-12):
        super(EntropyLoss, self).__init__()
        self.eps = eps

    def forward(self, x):
        loss = -1 * x * torch.log(x + self.eps)
        loss = torch.sum(loss, dim=-1)
        return torch.mean(loss)


# ===========================================================================
# TransformerVar main model
# ===========================================================================


class TransformerVar(nn.Module):
    """MEMTO Transformer with gated memory module.

    Args:
        memory_initial: If True, return encoder output only (first-phase / K-means collection mode).
                        If False (default), apply memory + decoder for reconstruction.
        memory_init_embedding: Tensor of shape (n_memory, d_model) — cluster centers for second phase.
                               None → random init (first phase).
        phase_type: 'first_train' | 'second_train' | 'test'
    """

    def __init__(
        self,
        win_size,
        enc_in,
        c_out,
        n_memory,
        shrink_thres=0.0,
        d_model=512,
        n_heads=8,
        e_layers=3,
        d_ff=512,
        dropout=0.0,
        activation="gelu",
        memory_init_embedding=None,
        memory_initial=False,
        phase_type=None,
    ):
        super(TransformerVar, self).__init__()
        self.memory_initial = memory_initial

        self.embedding = InputEmbedding(in_dim=enc_in, d_model=d_model, dropout=dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(win_size, d_model, n_heads, dropout=dropout),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )

        self.mem_module = MemoryModule(
            n_memory=n_memory,
            fea_dim=d_model,
            shrink_thres=shrink_thres,
            memory_init_embedding=memory_init_embedding,
            phase_type=phase_type,
        )

        # weak_decoder: 2*d_model → c_out (concat(query, mem-read))
        self.weak_decoder = Decoder(2 * d_model, c_out, d_ff=d_ff, activation="gelu", dropout=0.1)

    def forward(self, x):
        x = self.embedding(x)
        queries = out = self.encoder(x)  # (N, L, d_model)

        outputs = self.mem_module(out)
        out, attn, memory_item_embedding = (
            outputs["output"],
            outputs["attn"],
            outputs["memory_init_embedding"],
        )

        mem = self.mem_module.mem

        if self.memory_initial:
            return {
                "out": out,
                "memory_item_embedding": None,
                "queries": queries,
                "mem": mem,
            }
        else:
            out = self.weak_decoder(out)  # (N, L, c_out) — reconstruction
            return {
                "out": out,
                "memory_item_embedding": memory_item_embedding,
                "queries": queries,
                "mem": mem,
                "attn": attn,
            }
