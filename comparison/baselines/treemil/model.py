"""
TreeMIL - Timestamp-level Anomaly Detection via Multiple Instance Learning
on an N-ary tree of multi-scale subsequences.

Based on: "Timestamp-Supervised Multivariate Time Series Anomaly Detection
Through Multiple Instance Learning" (TreeMIL), Chen Liu et al.
Reference: https://github.com/fly-orange/TreeMIL  (commit 16f166c, owner fly-orange = Chen Liu)

LICENSE NOTICE (IMPORTANT)
--------------------------
The upstream TreeMIL repository is licensed under the GNU General Public
License v3.0 (GPL-3.0). The code in THIS file is a vendored, lightly-adapted
derivative of the upstream `model/embed.py`, `model/layers.py`,
`model/Pyramid.py`, and `utils.py` (the `PyraMIL.last_loss` method).
As a derivative work it inherits the GPL-3.0 terms. The full license text is
preserved in the upstream `LICENSE.md` (reference repo) and must accompany any
redistribution.

WHAT WAS VENDORED (verbatim, device-agnostic edits only)
--------------------------------------------------------
- embed.py: TokenEmbedding, PositionalEmbedding, DataEmbedding              (verbatim)
- layers.py: get_mask, refer_points, ScaledDotProductAttention,
  MultiHeadAttention, PositionwiseFeedForward, EncoderLayer, ConvLayer,
  Bottleneck_Construct, MaxPooling_Construct, AvgPooling_Construct          (verbatim)
- Pyramid.py: Pyramid.__init__ / forward / get_scores                       (vendored)
- utils.py: PyraMIL.last_loss == nn.BCELoss (the ACTIVE training objective) (vendored)

WHAT WAS INTENTIONALLY DROPPED (DEAD CODE — confirmed by Phase 1 audit)
----------------------------------------------------------------------
The Soft-DTW / alignment machinery is EXCLUDED because Phase 1 confirmed it is
NOT on the default training gradient path:
  * upstream `train.py:134` uses `pyra.last_loss(out['wscore'], wlabel)` = plain BCE.
  * `Pyramid.get_dpred / get_seqlabel / get_alignment` are only called from the
    eval-only `evaluation()` (train.py:45) for a predefined-threshold dense-F1
    metric — never to compute a training gradient.
  * `softdtw_cuda.py` (numba/CUDA Soft-DTW) is therefore unused at train time.
Dropping these makes the port torch-only (no numba, no CUDA Soft-DTW kernels):
  - DROPPED methods: Pyramid.get_seqlabel, Pyramid.get_alignment, Pyramid.get_dpred
  - DROPPED ctor arg: `dtw` (kept signature-compatible default None; unused)
  - DROPPED file: softdtw_cuda.py (not vendored)
  - DROPPED file: model/PreNet.py (only used for the optional `--fine_tune` flag)
  - DROPPED: PyraMIL.total_loss / weak_loss / ranking_loss / all metric fns
    (the comparison pipeline owns metrics; only `last_loss` is the active loss).

DEVICE-AGNOSTIC EDITS (the only behavioral edits)
-------------------------------------------------
Upstream hard-codes CUDA in `get_scores`:
  - Pyramid.py:113  `.type(torch.cuda.FloatTensor)`  (wpred)
  - Pyramid.py:126  `.type(torch.cuda.FloatTensor)`  (dpred)
These produce ONLY the predefined-threshold preds `wpred`/`dpred`, which the
comparison pipeline does not consume (it thresholds the RAW continuous score
itself). They are removed from `get_scores` here. All tensors created in
forward/get_scores derive their device from the input `x` (`.to(x.device)`),
so the model runs on CPU or GPU unchanged. NO architecture / loss /
hyperparameter / label-semantics change.

Default hyperparameters (from upstream `train.py` construction call, lines 101-103,
and argparse defaults): d_model=128, n_head=5, n_layer=2, ary_size=2,
inner_size=3, agg_type='max', pooling_type='max', split_size(seq_size)=500,
d_k=128, d_v=128, d_inner_hid=32, dropout=0.5.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# embed.py  (vendored verbatim from upstream model/embed.py)
# =============================================================================
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()

        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=1, padding=0, padding_mode='circular', bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)  # (T, 1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class DataEmbedding(nn.Module):
    def __init__(self, input_size, d_model, seq_len, dropout=0.0):
        super().__init__()

        self.input_size = input_size

        self.value_embedding = TokenEmbedding(input_size, d_model)
        self.position_embedding = PositionalEmbedding(d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):  # x (B, T, d)
        token = self.value_embedding(x)  # (B, T, D)
        position = self.position_embedding(x)
        embedding = token + position
        return self.dropout(embedding)


# =============================================================================
# layers.py  (vendored verbatim from upstream model/layers.py)
# =============================================================================
def get_mask(seq_size, ary_size, inner_size):
    """Get the attention mask of the N-ary tree (PAM)."""
    all_size = []
    effe_size = []

    s = math.ceil(math.log(seq_size, ary_size))
    all_size.append(int(math.pow(ary_size, s)))
    effe_size.append(seq_size)

    for i in range(s):
        layer_size = all_size[i] // ary_size
        all_size.append(layer_size)
        effe_size.append(math.ceil(effe_size[i] / ary_size))

    seq_length = sum(all_size)
    mask = torch.zeros(seq_length, seq_length)

    # intra-scale (neighbor) mask
    inner_window = inner_size // 2
    for layer_idx in range(len(all_size)):
        start = sum(all_size[:layer_idx])
        for i in range(start, start + effe_size[layer_idx]):
            left_side = max(i - inner_window, start)
            right_side = min(i + inner_window + 1, start + effe_size[layer_idx])
            mask[i, left_side:right_side] = 1

    # inter-scale (parent <-> child) mask
    for layer_idx in range(1, len(all_size)):
        start = sum(all_size[:layer_idx])
        for i in range(start, start + effe_size[layer_idx]):
            left_side = (start - all_size[layer_idx - 1]) + (i - start) * ary_size
            right_side = min((start - all_size[layer_idx - 1]) + (i - start + 1) * ary_size,
                             ((start - all_size[layer_idx - 1]) + effe_size[layer_idx - 1]))
            mask[i, left_side:right_side] = 1
            mask[left_side:right_side, i] = 1

    mask = (1 - mask).bool()

    return mask, all_size, effe_size


def refer_points(effe_sizes, ary_size):
    """Gather features from the pyramid sequences (leaf -> ancestor chain indices)."""
    input_size = effe_sizes[0]
    indexes = torch.zeros(input_size, len(effe_sizes))

    for i in range(input_size):
        indexes[i][0] = i
        former_index = i
        for j in range(1, len(effe_sizes)):
            start = sum(effe_sizes[:j])
            inner_layer_idx = former_index - (start - effe_sizes[j - 1])
            former_index = start + min(inner_layer_idx // ary_size, effe_sizes[j] - 1)
            indexes[i][j] = former_index

    return indexes.long()


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention."""

    def __init__(self, temperature, attn_dropout=0.2):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module."""

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.xavier_uniform_(self.w_qs.weight)
        nn.init.xavier_uniform_(self.w_ks.weight)
        nn.init.xavier_uniform_(self.w_vs.weight)

        self.fc = nn.Linear(d_v * n_head, d_model)
        nn.init.xavier_uniform_(self.fc.weight)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=dropout)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        if self.normalize_before:
            q = self.layer_norm(q)

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            if len(mask.size()) == 3:
                mask = mask.unsqueeze(1)  # For head axis broadcasting.

        output, attn = self.attention(q, k, v, mask=mask)

        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output += residual

        if not self.normalize_before:
            output = self.layer_norm(output)
        return output, attn


class PositionwiseFeedForward(nn.Module):
    """Two-layer position-wise feed-forward network."""

    def __init__(self, d_in, d_hid, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before

        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)

        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        if self.normalize_before:
            x = self.layer_norm(x)

        x = F.gelu(self.w_1(x))
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        x = x + residual

        if not self.normalize_before:
            x = self.layer_norm(x)
        return x


class EncoderLayer(nn.Module):
    """Compose with two layers."""

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, normalize_before=True,
                 q_k_mask=None, k_q_mask=None):
        super(EncoderLayer, self).__init__()

        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout,
                                           normalize_before=normalize_before)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout,
                                               normalize_before=normalize_before)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class ConvLayer(nn.Module):
    def __init__(self, c_in, window_size):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in, out_channels=c_in,
                                  kernel_size=window_size, stride=window_size)
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.downConv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class Bottleneck_Construct(nn.Module):
    """Bottleneck convolution CSCM."""

    def __init__(self, d_model, depth, ary_size, d_inner):
        super(Bottleneck_Construct, self).__init__()

        self.conv_layers = []
        for i in range(depth - 1):
            self.conv_layers.append(ConvLayer(d_inner, ary_size))
        self.conv_layers = nn.ModuleList(self.conv_layers)

        self.up = nn.Linear(d_inner, d_model)
        self.down = nn.Linear(d_model, d_inner)
        self.norm = nn.LayerNorm(d_model)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, enc_input):
        temp_input = self.down(enc_input).permute(0, 2, 1)  # (B, d, T)
        all_inputs = []
        for i in range(len(self.conv_layers)):
            temp_input = self.conv_layers[i](temp_input)  # (B, d, T/C)
            all_inputs.append(temp_input)

        all_inputs = torch.cat(all_inputs, dim=2).transpose(1, 2)
        all_inputs = self.up(all_inputs)
        all_inputs = torch.cat([enc_input, all_inputs], dim=1)
        all_inputs = self.norm(all_inputs)
        return all_inputs


class MaxPooling_Construct(nn.Module):
    """Max pooling CSCM (default `agg_type='max'`)."""

    def __init__(self, d_model, depth, ary_size):
        super(MaxPooling_Construct, self).__init__()

        self.pooling_layers = []
        for i in range(depth - 1):
            self.pooling_layers.append(nn.MaxPool1d(kernel_size=ary_size))
        self.pooling_layers = nn.ModuleList(self.pooling_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_input):
        all_inputs = []
        enc_input = enc_input.transpose(1, 2).contiguous()
        all_inputs.append(enc_input)

        for layer in self.pooling_layers:
            enc_input = layer(enc_input)
            all_inputs.append(enc_input)

        all_inputs = torch.cat(all_inputs, dim=2).transpose(1, 2)
        all_inputs = self.norm(all_inputs)
        return all_inputs


class AvgPooling_Construct(nn.Module):
    """Average pooling CSCM."""

    def __init__(self, d_model, depth, ary_size):
        super(AvgPooling_Construct, self).__init__()

        self.pooling_layers = []
        for i in range(depth - 1):
            self.pooling_layers.append(nn.AvgPool1d(kernel_size=ary_size))
        self.pooling_layers = nn.ModuleList(self.pooling_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_input):
        all_inputs = []
        enc_input = enc_input.transpose(1, 2).contiguous()
        all_inputs.append(enc_input)

        for layer in self.pooling_layers:
            enc_input = layer(enc_input)
            all_inputs.append(enc_input)

        all_inputs = torch.cat(all_inputs, dim=2).transpose(1, 2)
        all_inputs = self.norm(all_inputs)
        return all_inputs


# =============================================================================
# Pyramid.py  (vendored; DTW path dropped; device-agnostic)
# =============================================================================
class Pyramid(nn.Module):
    """N-ary-tree Transformer core. Vendored from upstream model/Pyramid.py.

    Compared to upstream:
      - `get_seqlabel` / `get_alignment` / `get_dpred` removed (DTW dead path).
      - The `dtw` ctor arg is retained (default None) for signature parity but
        is never used.
      - `get_scores` no longer builds `wpred`/`dpred` (the only `.cuda()` casts);
        it returns only `wscore` (B,) and `dscore` (B, seq_size) — exactly the
        tensors the active BCE loss and the comparison scorer need.
    """

    def __init__(self, input_size, seq_size, ary_size, inner_size, d_model, n_layer, n_head,
                 d_k, d_v, d_inner_hid, agg_type, dropout,
                 pooling_type='max', granularity=1, local_threshold=0.5, global_threshold=0.5,
                 beta=10, dtw=None):
        super(Pyramid, self).__init__()

        self.input_size = input_size
        self.seq_size = seq_size
        self.d_model = d_model
        self.ary_size = ary_size
        self.num_heads = n_head

        self.mask, self.all_size, self.effe_size = get_mask(seq_size, ary_size, inner_size)

        self.depth = len(self.all_size)
        # `indexes` is a plain (non-parameter) tensor; register as buffer so it
        # moves with .to(device)/.cuda(). Upstream stored it as an attribute and
        # relied on `.to(out.device)` at call time — we keep the same call-time
        # `.to(out.device)` for safety AND register it so state_dict/.to() work.
        self.register_buffer('indexes', refer_points(self.effe_size, ary_size))

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout, normalize_before=False)
            for i in range(n_layer)
        ])

        self.embedding = DataEmbedding(input_size, d_model, seq_size)

        if agg_type == 'conv':
            self.conv_layers = Bottleneck_Construct(d_model, self.depth, ary_size, d_k)
        elif agg_type == 'max':
            self.conv_layers = MaxPooling_Construct(d_model, self.depth, ary_size)
        elif agg_type == 'avg':
            self.conv_layers = AvgPooling_Construct(d_model, self.depth, ary_size)

        self.scorenet = nn.Linear(d_model, 1)

        self.rf_size = self.all_size[0]  # receptive-field size
        self.pooling_type = pooling_type

        self.local_threshold = local_threshold
        self.global_threshold = global_threshold
        self.granularity = granularity
        self.beta = beta

        self.dtw = dtw  # retained for signature parity; UNUSED (DTW path dropped)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):  # (B, T, D)
        seq_enc = self.embedding(x).transpose(1, 2)  # (B, D, T)

        mask = self.mask.repeat(len(seq_enc), self.num_heads, 1, 1).to(x.device)

        padding_size = self.rf_size - seq_enc.shape[2]
        if padding_size > 0:
            seq_enc = F.pad(seq_enc, (0, padding_size), "constant", 0).transpose(1, 2)  # (B, T, D)

        # Coarse-to-fine tree construction (CSCM)
        seq_enc = self.conv_layers(seq_enc)  # (B, T+T/C+..., D)

        for i in range(len(self.layers)):
            seq_enc, seq_attn = self.layers[i](seq_enc, mask)  # (B, N, D)

        out = [seq_enc[:, sum(self.all_size[:i]):sum(self.all_size[:i]) + self.effe_size[i]]
               for i in range(len(self.all_size))]
        out = torch.cat(out, dim=1)  # (B, L1+L2+...+Ls, D)

        return out

    def get_scores(self, x):
        ret = {}
        out = self.forward(x)  # (B, L1+L2+...+Ls, D)

        ret['output'] = out

        if self.pooling_type == 'avg':
            _out = torch.mean(out, dim=1)
        elif self.pooling_type == 'max':
            _out = torch.max(out, dim=1)[0]

        # Window (bag/weak) score — the BCE training target.
        ret['wscore'] = torch.sigmoid(self.scorenet(_out).squeeze(dim=1))  # (B,)

        # Dense (per-timestep) score — the per-point output used for scoring.
        indexes = self.indexes.unsqueeze(0).unsqueeze(-1).repeat(
            out.size(0), 1, 1, out.size(2)).to(out.device)  # (B, T, depth, D)
        indexes = indexes.view(out.size(0), -1, out.size(2))  # (B, T*depth, D)
        all_enc = torch.gather(out, 1, indexes)
        all_enc = all_enc.view(out.size(0), self.effe_size[0], -1, out.size(2))  # (B, T, depth, D)
        h = torch.mean(all_enc, dim=2)  # (B, T, D)

        d_score = torch.sigmoid(self.scorenet(h)).squeeze(dim=2)  # (B, T)  T = seq_size
        ret['dscore'] = d_score

        return ret


# =============================================================================
# utils.py  (vendored: PyraMIL.last_loss == the ACTIVE training objective)
# =============================================================================
class PyraMIL:
    """Holds the active TreeMIL loss. Upstream `utils.py:PyraMIL`.

    Only `last_loss` (plain window-level BCE) is vendored — it is the sole loss
    on the default training gradient path (train.py:134). The other methods
    (total_loss / weak_loss / ranking_loss) and all metric helpers are omitted.
    """

    def __init__(self):
        self.bce = nn.BCELoss(reduction='mean')

    def last_loss(self, scores, wlabel):
        loss = self.bce(scores, wlabel)
        return loss
