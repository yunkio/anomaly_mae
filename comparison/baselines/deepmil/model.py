"""
DeepMIL (time-series) — MIL ranking head + loss (Sultani CVPR'18) on a DiCNN encoder.

Two-part provenance (read this before editing):

1. ENCODER = WETAS DiCNN — DERIVATIVE_CITED (NOT OFFICIAL DeepMIL).
   Sultani et al. CVPR'18 (arXiv 1801.04264) is a VIDEO method: it consumes frozen
   C3D fc6 4096-d features and has NO learnable time-series encoder. The canonical
   DeepMIL-on-time-series encoder is defined by a downstream WS-TSAD paper:
     DERIVATIVE_CITED: WETAS (Lee et al., ICCV'21, CVF PDF p.7360) states verbatim
     "DeepMIL employs the same model architecture with WETAS (i.e., DiCNN)" and
     "DeepMIL-4,8,16". WETAS Impl-Details: DiCNN = 7 dilated-conv layers (MTS),
     filter size 2 → receptive field 2^7=128, hidden=output dim d=128.
   This file therefore imports the in-project vendored WETAS `DilatedCNN`
   (`comparison/baselines/wetas/model.py`, GPL-3.0, donalee/WETAS@cb149dc) as the
   encoder, configured exactly as WETAS p.7360 prescribes for its DeepMIL baseline
   (input_size=n_features, hidden_size=128, output_size=128, kernel_size=2,
   n_layers=7). This is a CITED-DERIVATIVE encoder — it is **NOT** the official
   Sultani encoder (the official method has none) and is **NOT** claimed to be a
   verbatim DeepMIL artifact; it is the literature-sanctioned DeepMIL-on-TS encoder.
   (Supersedes the previous bespoke `TSSegmentEncoder`, which A6 classified
   NON_OFFICIAL_ADAPTATION — neither official nor the cited DiCNN.)

2. RANKING HEAD + LOSS = Sultani CVPR'18 — FAITHFUL (clean-room).
   Official repo unlicensed+non-runnable (video/C3D; Keras 1.1.0/Theano, Python 2);
   the head+loss are clean-room reimplemented from the paper's documented form per
   Phase-1 reference-substitution. NO CODE COPIED from the unlicensed official repo;
   the ekosman MIT PyTorch port was consulted as a CROSS-CHECK only.
   - MIL ranking head: Linear(D->512)+ReLU+Dropout(0.6) -> Linear(512->32)
     (NO activation) +Dropout(0.6) -> Linear(32->1)+Sigmoid.  D = DiCNN out = 128.
   - glorot_normal (Xavier-normal) init on all three Linear layers.
   - L2 weight regularization 0.001 on all three Linear layers (Keras `l2(0.001)`).
   - Loss = MIL ranking HINGE (margin 1.0) on the max-instance score of a positive
     bag vs a negative bag  +  temporal SMOOTHNESS (lambda=8e-5)  +  SPARSITY
     (lambda=8e-5) — both regularizers on POSITIVE (anomalous) bags only.

DENSE scoring (C2 — NON_OFFICIAL, disclosed; forced by the DiCNN encoder, NOT paper-faithful):
   The original DeepMIL scores 32 fixed video segments per bag. The DiCNN encoder is
   a whole-window, per-timestep feature extractor (it produces a length-L feature
   map), mirroring WETAS `get_scores` whose `dscore = sigmoid(fc(out))` is dense
   per-timestep. We therefore score DENSELY: the Sultani MIL head is applied at every
   timestep, giving per-timestep instance scores (B, L). The MIL "instances" of a bag
   are now its L timesteps (not 32 segments), and the ranking max is taken over
   timesteps (max-over-timesteps): positive-bag max > negative-bag max. This keeps the
   MIL ranking essence (positive>negative hinge on the max instance) while being
   structurally consistent with the dense DiCNN encoder. The window length over the
   continuous TS stream and the test-time overlap aggregation are NON_OFFICIAL
   (TS-specific, disclosed) choices (handled in wrapper.py).
"""

from typing import Tuple

import torch
import torch.nn as nn

# DERIVATIVE_CITED encoder: in-project vendored WETAS DiCNN (GPL-3.0, donalee/WETAS).
# WETAS(ICCV'21 p.7360) defines the DeepMIL-on-TS encoder = DiCNN. NOT the OFFICIAL
# Sultani encoder (the official video method has no learnable TS encoder).
from comparison.baselines.wetas.model import DilatedCNN


class MILRankingHead(nn.Module):
    """MIL ranking head — FAITHFUL to Sultani et al. CVPR'18 / official code.

    D -> 512 (ReLU, Dropout 0.6) -> 32 (linear, Dropout 0.6) -> 1 (Sigmoid).
    glorot_normal (Xavier-normal) init on all three Linear layers.
    Operates per instance; bag-level max reduction lives in the loss. With the DiCNN
    encoder an "instance" is a single timestep, so the head is applied densely over
    the length-L DiCNN feature map (see DeepMILModel.forward / C2 dense scoring).
    """

    def __init__(self, input_dim: int, dropout: float = 0.6):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, 32)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

        # glorot_normal == Xavier-normal (paper/official `init='glorot_normal'`).
        for fc in (self.fc1, self.fc2, self.fc3):
            nn.init.xavier_normal_(fc.weight)
            nn.init.zeros_(fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., input_dim) -> (..., 1) sigmoid score per instance.
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.fc2(x))             # NO activation (official model.json: linear)
        x = self.sigmoid(self.fc3(x))
        return x

    def l2_weight_penalty(self) -> torch.Tensor:
        """Sum of squared weights of all three Linear layers.

        Multiply by 0.001 in the loss to emulate the Keras `W_regularizer=l2(0.001)`.
        """
        return sum((fc.weight ** 2).sum() for fc in (self.fc1, self.fc2, self.fc3))


class DeepMILModel(nn.Module):
    """Full DeepMIL-on-TS model = WETAS DiCNN encoder + Sultani MIL ranking head.

    Encoder: WETAS DiCNN (DERIVATIVE_CITED, WETAS p.7360) — input_size=n_features,
      hidden_size=output_size=128, kernel_size=2, n_layers=7 (RF=2^7=128). Maps a
      window (B, L, F) -> dense per-timestep features (B, L, 128).
    Head: Sultani MIL ranking head (FAITHFUL) applied DENSELY at every timestep ->
      per-timestep sigmoid scores (B, L).

    forward(windows) where windows is (B, L, n_features) returns per-TIMESTEP sigmoid
    scores (B, L). (C2: dense per-timestep scoring; the previous per-segment (B, S)
    output is superseded.)
    """

    # WETAS p.7360 / Impl-Details DiCNN config for the DeepMIL-on-TS baseline.
    _DICNN_HIDDEN = 128       # WETAS Impl-Details: d = 128
    _DICNN_KERNEL = 2         # WETAS Impl-Details: filter size 2
    _DICNN_N_LAYERS = 7       # WETAS Impl-Details: 7 layers for MTS -> RF = 2^7 = 128

    def __init__(
        self,
        n_features: int,
        seq_len: int,
        encoder_dim: int = 128,
        dropout: float = 0.6,
    ):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.encoder_dim = encoder_dim

        # DERIVATIVE_CITED DiCNN encoder (WETAS p.7360). dtw/pooling_type/thresholds are
        # WETAS-framework knobs used only by WETAS-specific methods (dtw_loss/get_seqlabel);
        # DeepMIL uses ONLY DilatedCNN.forward (dense feature map), so dtw=None is safe and
        # split_size is set to the bag window length for the encoder's RF left-pad logic.
        self.encoder = DilatedCNN(
            input_size=n_features,
            hidden_size=self._DICNN_HIDDEN,
            output_size=encoder_dim,
            kernel_size=self._DICNN_KERNEL,
            n_layers=self._DICNN_N_LAYERS,
            split_size=seq_len,
            dtw=None,
        )
        # Sultani MIL ranking head (FAITHFUL), input dim = DiCNN output_size (128).
        self.head = MILRankingHead(encoder_dim, dropout=dropout)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        # windows: (B, L, F) -> DiCNN dense features (B, L, D) -> head -> (B, L).
        feat = self.encoder(windows)            # (B, L, D)  (DiCNN.forward transposes internally)
        scores = self.head(feat).squeeze(-1)    # (B, L)  per-timestep sigmoid score
        return scores


def mil_ranking_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    margin: float = 1.0,
    lambda_smooth: float = 8e-5,
    lambda_sparse: float = 8e-5,
) -> Tuple[torch.Tensor, dict]:
    """DeepMIL loss — FAITHFUL to Sultani et al. CVPR'18 (Eq. in §3.1).

    With the DiCNN encoder the MIL "instances" of a bag are its L timesteps, so the
    score tensors are per-TIMESTEP (B, L) instead of per-segment (B, S). The loss is
    otherwise unchanged — the bag-level reduction is max-over-instances, here
    max-over-timesteps (C2).

    Args:
        pos_scores: (P, L) per-timestep sigmoid scores of POSITIVE (anomalous) bags.
        neg_scores: (Nn, L) per-timestep sigmoid scores of NEGATIVE (normal) bags.
        margin: ranking hinge margin (1.0, code-sourced).
        lambda_smooth: temporal-smoothness weight (8e-5, code-sourced).
        lambda_sparse: sparsity weight (8e-5, code-sourced).

    Returns:
        (loss, components_dict)

    Ranking term: FULL n_Nor x n_Abn cross-product, FAITHFUL to the official Keras
    custom_objective (TrainingAnomalyDetector_public.py@L265-271, fetched live
    2026-06-04). For each NORMAL bag max (Sub_Nor[ii], L266), the official code forms
    sub_z = max(1 - Sub_Abn + Sub_Nor[ii], 0) — a vector over ALL abnormal-bag maxes —
    then T.sum(sub_z) sums the hinge over every abnormal bag (L267-268); finally
    T.mean over the n_Nor normal-bag sums (L271). (NB: the official variable names are
    label-swapped — indx_nor = F_labels==32 = NORMAL videos, so Sub_Nor = normal-bag
    maxes and Sub_Abn = abnormal-bag maxes.) In our naming pos_scores = anomalous
    (abnormal) bags -> pos_max == Sub_Abn; neg_scores = normal bags -> neg_max ==
    Sub_Nor. The max is over timesteps (max-over-timesteps), the dense analog of the
    original max-over-segments.
    Smoothness + sparsity are applied to POSITIVE bags only (official slices to the
    abnormal half; the cross-check uses anomalous segments only). On dense per-timestep
    scores these become the standard temporal-smoothness / sparsity over the timeline.
    """
    pos_max = pos_scores.max(dim=1).values   # (P,)  max-over-timesteps == Sub_Abn
    neg_max = neg_scores.max(dim=1).values   # (Nn,) max-over-timesteps == Sub_Nor

    # FULL cross-product ranking hinge (official custom_objective L265-271):
    # for each normal-bag max neg_max[j], SUM over ALL abnormal-bag maxes of
    # max(0, margin - pos_max + neg_max[j]); then MEAN over normal bags.
    # pairwise (Nn, P): margin - pos_max[None, :] + neg_max[:, None]
    pairwise = torch.clamp(
        margin - pos_max.unsqueeze(0) + neg_max.unsqueeze(1), min=0.0
    )                                        # (Nn, P)
    hinge = pairwise.sum(dim=1).mean()       # sum over abnormal bags, mean over normal bags

    # Temporal smoothness: squared diff of consecutive per-timestep scores (positive bags).
    smooth = ((pos_scores[:, 1:] - pos_scores[:, :-1]) ** 2).sum(dim=1).mean()

    # Sparsity: sum of per-timestep scores (positive bags).
    sparse = pos_scores.sum(dim=1).mean()

    loss = hinge + lambda_smooth * smooth + lambda_sparse * sparse
    comps = {
        "hinge": float(hinge.detach().cpu()),
        "smooth": float(smooth.detach().cpu()),
        "sparse": float(sparse.detach().cpu()),
    }
    return loss, comps
