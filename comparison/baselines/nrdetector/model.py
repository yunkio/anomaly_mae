"""
NRdetector — vendored OFFICIAL method modules (device-agnostic, torch-only).

Paper : "Noise-Resilient Point-wise Anomaly Detection in Time Series Using
         Weak Segment Labels" (Wang, Cheng, Xiong, Wen, Jia, Song, Zhang, Zhu,
         Liu), ACM SIGKDD (KDD) 2025. arXiv:2501.11959, DOI 10.1145/3690624.3709257.
Source : https://github.com/UCSC-REAL/NRdetector  (commit bd5592b, 2025-01-16)
Owner  : UCSC-REAL (Yang Liu lab, UC Santa Cruz)
License: MIT — Copyright (c) 2025 Pretty Dora (see upstream LICENSE.txt).

This file is a faithful, device-agnostic re-host of the OFFICIAL upstream method
for the in-memory `fit/predict` pipeline adapter (see `wrapper.py`). NOTHING in
the architecture / PU-loss / label-distribution semantics / hyperparameters has
been altered; only the CLI/disk plumbing was removed.

VENDORED (1:1 ported):
  - `ResidualBlock`, `DilatedCNN` (WETAS-style encoder)  ← extractor.py:154-358
  - `Classifier_six_layer`                                ← models.py:42-74
  - `LabelDistributionLoss`, `create_loss`                ← models.py:77-155
  - `smooth`, `constraint_loss`                           ← models.py:158-217
  - `CLASS_PRIOR`                                         ← models.py:6-13

DROPPED (documented reference-substitutions, decided in Phase-1 §10/§12):
  - HOC threshold (modules/hoc.py — pretrained resnet18 + torchvision):
    threshold-only stage. The comparison pipeline performs its OWN thresholding
    (PA%K sweep / ROC); HOC is unused here, which also removes the torchvision
    dependency entirely (Phase-1 G4).

RESTORED (2026-06-01 SOFTDTW_RESTORE — documentation only; see SOFTDTW_RESTORE_IMPL_NOTES.md):
  - soft-DTW `dtw_loss` alignment regularizer (extractor.py:313-331) is now
    part of the encoder objective. The encoder loss is the OFFICIAL combination
    `bce(wscore, wlabel) + dtw_loss(output, wlabel).mean(0)` (extractor.py:127-129,
    EQUAL-WEIGHT sum). `dtw_loss`/`get_seqlabel`/`get_alignment` are vendored here
    (1:1 with the already-vendored WETAS model.py:157-198, which is byte-identical
    to NRdetector extractor.py:301-342). The soft-DTW kernel is REUSED via import
    from `comparison.baselines.wetas.softdtw_cuda` (MIT, Maghoumi) — the CPU numba
    `@jit` path (use_cuda=False) requires NO `pip install` / NO CUDA. `get_dpred`
    (extractor.py:344-358, eval/threshold-only) remains omitted — not on the
    training path.

DEVICE EDITS (Phase-1 G10):
  - Removed import-time `os.environ["CUDA_VISIBLE_DEVICES"]="7"` (extractor.py:7).
  - Replaced all `.cuda()` / `cuda:7` / `cuda:1` literals with caller-provided
    `device` (the upstream tensors created via `torch.zeros(...).cuda()` now use
    the input tensor's device — extractor.py:255).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Per-dataset class prior for the PU LabelDistributionLoss (models.py:6-13).
# Upstream values; pipeline datasets fall back to the argparse default (0.25,
# main.py:98) or an estimate from the train wlabel rate (see wrapper.py).
# ---------------------------------------------------------------------------
CLASS_PRIOR = {
    'EMG': 0.4,
    'SMD': 0.31,
    'MSL': 0.20,
    'SMAP': 0.174,
    'PSM': 0.31,
}


# ===========================================================================
# Stage-0 encoder — WETAS-style DilatedCNN  (extractor.py:154-299)
# ===========================================================================
class ResidualBlock(nn.Module):
    """Gated (tanh*sigmoid) dilated residual block — extractor.py:154-181 (1:1)."""

    def __init__(self, input_size, output_size, kernel_size, dilation):
        super(ResidualBlock, self).__init__()
        self.dilation = dilation
        self.diconv = nn.Conv1d(in_channels=input_size, out_channels=output_size,
                                kernel_size=kernel_size, dilation=dilation)
        self.conv1by1_skip = nn.Conv1d(in_channels=output_size, out_channels=output_size,
                                       kernel_size=1, dilation=1)
        self.conv1by1_out = nn.Conv1d(in_channels=output_size, out_channels=output_size,
                                      kernel_size=1, dilation=1)

    def forward(self, x):
        x = F.pad(x, (self.dilation, 0), "constant", 0)
        z = self.diconv(x)
        z = torch.tanh(z) * torch.sigmoid(z)
        s = self.conv1by1_skip(z)
        z = self.conv1by1_out(z) + x[:, :, -z.shape[2]:]
        return z, s


class DilatedCNN(nn.Module):
    """WETAS-style dilated causal CNN encoder — extractor.py:184-299.

    Faithful port of the architecture + `forward` + `get_scores`. The soft-DTW
    alignment methods (`dtw_loss`, `get_seqlabel`, `get_alignment`) are RESTORED
    (2026-06-01, see module docstring) and are 1:1 with the vendored WETAS
    model.py:157-198 (== NRdetector extractor.py:301-342). The training objective
    is the OFFICIAL combination `bce(wscore, wlabel) + dtw_loss(output, wlabel)`
    (extractor.py:127-129). `get_dpred` (eval/threshold-only) remains omitted.
    The soft-DTW kernel is injected via the `dtw=` ctor kwarg (a
    `SoftDTW(use_cuda, gamma, normalize=False)` instance, extractor.py:42/204).

    Per-window outputs of `get_scores` (extractor.py:275-299):
      - 'output' (B, T, hidden) dense features
      - 'h'      (B, hidden)    pooled (avg/max) feature
      - 'wscore' (B,)           segment anomaly prob = sigmoid(fc(pooled))   :288
      - 'dscore' (B, T)         per-point anomaly prob = sigmoid(fc(output)) :296
      - 'dh'     (B, T)         raw dense logit fc(output) BEFORE sigmoid;
                                source of the official ACTMAP (per-window min-max
                                of `h=fc(out).squeeze`, extractor.py:344-358).
    """

    def __init__(self, input_size, hidden_size, output_size, kernel_size, n_layers,
                 pooling_type='avg', granularity=1, local_threshold=0.5,
                 global_threshold=0.5, beta=10, split_size=100, dtw=None):
        super(DilatedCNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.rf_size = self.kernel_size ** (self.n_layers - 1)  # receptive field (extractor.py:195)
        self.pooling_type = pooling_type
        self.local_threshold = local_threshold
        self.global_threshold = global_threshold
        self.granularity = granularity
        self.beta = beta
        self.split_size = split_size
        self.dtw = dtw  # SoftDTW(use_cuda, gamma, normalize=False) instance (extractor.py:204)
        self.build_model(input_size, hidden_size, output_size, kernel_size, n_layers)

    def build_model(self, input_size, hidden_size, output_size, kernel_size, n_layers):
        self.causal_conv = nn.Conv1d(in_channels=input_size, out_channels=hidden_size,
                                     kernel_size=kernel_size, stride=1, dilation=1)
        self.diconv_layers = nn.ModuleList()
        for i in range(n_layers):
            self.diconv_layers.append(
                ResidualBlock(input_size=hidden_size, output_size=hidden_size,
                              kernel_size=kernel_size, dilation=kernel_size ** i))
        self.conv1by1_skip1 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size,
                                        kernel_size=1, dilation=1)
        self.conv1by1_skip2 = nn.Conv1d(in_channels=hidden_size, out_channels=output_size,
                                        kernel_size=1, dilation=1)
        self.fc = nn.Linear(hidden_size, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                init.normal_(m.bias.data)
            if isinstance(m, nn.Conv1d):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    init.normal_(m.bias.data)

    def forward(self, x):
        # x: (B, T, F) — extractor.py:245-273 (1:1, device-agnostic `out` init)
        x = x.transpose(1, 2)
        padding_size = self.rf_size - x.shape[2]
        if padding_size > 0:
            x = F.pad(x, (padding_size, 0), "constant", 0)
        x = F.pad(x, (1, 0), "constant", 0)
        z = self.causal_conv(x)
        # Upstream: `torch.zeros(z.shape).cuda()` — made device-agnostic (Phase-1 G10).
        out = torch.zeros(z.shape, device=z.device, dtype=z.dtype)
        for diconv in self.diconv_layers:
            z, s = diconv(z)
            out += s
        out = F.relu(out)
        out = self.conv1by1_skip1(out)
        out = F.relu(out)
        out_plus = out.transpose(1, 2)
        out = self.conv1by1_skip2(out).transpose(1, 2)
        return out, out_plus

    def get_scores(self, x):
        # extractor.py:275-299 — `wpred`/`dpred` (DTW/threshold-only) dropped.
        ret = {}
        out, out_plus = self.forward(x)
        ret['output'] = out
        ret['p'] = out_plus
        if self.pooling_type == 'avg':
            _out = torch.mean(out, dim=1)
        elif self.pooling_type == 'max':
            _out = torch.max(out, dim=1)[0]
        ret['h'] = _out
        ret['wscore'] = torch.sigmoid(self.fc(_out).squeeze(dim=1))   # (B,)   :288
        h = self.fc(out).squeeze(dim=2)
        ret['dh'] = h                                                  # (B, T) raw logit (actmap source) :294/319/350
        ret['dscore'] = torch.sigmoid(h)                               # (B, T) :296
        return ret

    # -------------------------------------------------------------------
    # soft-DTW alignment regularizer — RESTORED 2026-06-01.
    # 1:1 with WETAS model.py:157-198 (== NRdetector extractor.py:301-342),
    # this repo's standard device-agnostic edits (.float()/device=) applied.
    # USE the existing `F` alias (official uses `as f`); the dead cuda:1 line
    # (extractor.py:307) and `get_dpred` (eval-only) are intentionally dropped.
    # -------------------------------------------------------------------
    def get_seqlabel(self, actmap, wlabel):   # extractor.py:301-311
        # non-inplace (spec): safer under no_grad, numerically identical (actmap discarded).
        actmap = actmap * wlabel.unsqueeze(dim=1).repeat(1, actmap.shape[1])
        seqlabel = (actmap >= self.local_threshold).float()
        seqlabel = F.pad(seqlabel, (self.granularity - seqlabel.shape[1] % self.granularity, 0), 'constant', 0)
        seqlabel = torch.reshape(seqlabel, (seqlabel.shape[0], -1, int(seqlabel.shape[1] / self.granularity)))
        seqlabel = torch.max(seqlabel, dim=2)[0]
        zcol = torch.zeros(seqlabel.shape[0], 1, device=seqlabel.device, dtype=seqlabel.dtype)
        seqlabel = torch.cat([zcol, seqlabel, zcol], dim=1)
        return seqlabel

    def dtw_loss(self, out, wlabel):   # extractor.py:313-331 (verbatim numerics)
        h = self.fc(out).squeeze(dim=2)
        dscore = torch.sigmoid(self.fc(out).squeeze(dim=2))
        with torch.no_grad():
            actmap = h
            actmin = torch.min(actmap, dim=1)[0]
            actmap = actmap - actmin.unsqueeze(dim=1)
            actmax = torch.max(actmap, dim=1)[0]
            actmap = actmap / actmax.unsqueeze(dim=1)
            pos_seqlabel = self.get_seqlabel(actmap, wlabel)
            neg_seqlabel = self.get_seqlabel(actmap, 1 - wlabel)
        pos_dist = self.dtw(pos_seqlabel.unsqueeze(dim=1), dscore.unsqueeze(dim=1)) / self.split_size
        neg_dist = self.dtw(neg_seqlabel.unsqueeze(dim=1), dscore.unsqueeze(dim=1)) / self.split_size
        loss = F.relu(self.beta + pos_dist - neg_dist)   # hinge margin; returns (B,)
        return loss

    def get_alignment(self, label, score):   # extractor.py:333-342
        assert label.shape[0] == score.shape[0]
        assert len(label.shape) == 2
        assert len(score.shape) == 2
        A = self.dtw.align(label.unsqueeze(dim=1), score.unsqueeze(dim=1))
        indices = torch.max(A, dim=1)[1]
        return torch.gather(label, 1, indices)


# ===========================================================================
# Stage-2 PU classifier  (models.py:42-74)
# ===========================================================================
class Classifier_six_layer(nn.Module):
    """PU segment classifier — models.py:42-74 (1:1).

    forward returns (x, dscores):
      - x       (B, num_classes) segment anomaly prob (sigmoid)
      - dscores (B, 100)         penultimate per-point activation (constraint_loss)
    """

    def __init__(self, input_size, hidden_size, num_classes):
        super(Classifier_six_layer, self).__init__()
        self.pooling_type = 'avg'
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 100),
            nn.ReLU(),
        )
        self.fc1 = nn.Linear(100, 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        dscores = self.net(x)            # (B, 100)
        x = self.fc1(dscores)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x, dscores


# ===========================================================================
# PU LabelDistributionLoss  (models.py:77-155)
# ===========================================================================
class LabelDistributionLoss(torch.nn.Module):
    """Distribution-matching PU loss (dist-PU style) — models.py:77-148 (1:1)."""

    def __init__(self, prior, device, num_bins=1, proxy='polar', dist='L1'):
        super(LabelDistributionLoss, self).__init__()
        self.prior = prior
        self.frac_prior = 1.0 / (2 * prior)
        self.step = 1 / num_bins
        self.device = device
        self.t = torch.arange(0, 1 + self.step, self.step).view(1, -1).requires_grad_(False)
        self.t_size = num_bins + 1

        self.dist = None
        if dist == 'L1':
            self.dist = F.l1_loss
        else:
            raise NotImplementedError("The distance: {} is not defined!".format(dist))

        proxy_p, proxy_n = None, None
        if proxy == 'polar':
            proxy_p = np.zeros(self.t_size, dtype=float)
            proxy_n = np.zeros_like(proxy_p)
            proxy_p[-1] = 1
            proxy_n[0] = 1
        else:
            raise NotImplementedError("The proxy: {} is not defined!".format(proxy))

        proxy_mix = prior * proxy_p + (1 - prior) * proxy_n
        self.proxy_p = torch.from_numpy(proxy_p).requires_grad_(False).float()
        self.proxy_mix = torch.from_numpy(proxy_mix).requires_grad_(False).float()

        self.t = self.t.to(self.device)
        self.proxy_p = self.proxy_p.to(self.device)
        self.proxy_mix = self.proxy_mix.to(self.device)

    def histogram(self, scores):
        scores_rep = scores.repeat(1, self.t_size)
        hist = torch.abs(scores_rep - self.t)
        inds = (hist > self.step)
        hist = self.step - hist
        hist[inds] = 0
        return hist.sum(dim=0) / (len(scores) * self.step)

    def forward(self, outputs, labels):
        scores = torch.sigmoid(outputs)
        labels = labels.view(-1, 1)
        scores = scores.view_as(labels)
        s_p = scores[labels == 1].view(-1, 1)
        s_u = scores[labels == 0].view(-1, 1)

        l_p = 0
        l_u = 0
        if s_p.numel() > 0:
            hist_p = self.histogram(s_p)
            l_p = self.dist(hist_p, self.proxy_p, reduction='mean')
        if s_u.numel() > 0:
            hist_u = self.histogram(s_u)
            l_u = self.dist(hist_u, self.proxy_mix, reduction='mean')

        return l_p + self.frac_prior * l_u


def create_loss(dataset_prior, device):
    """models.py:151-155 (1:1)."""
    return LabelDistributionLoss(prior=dataset_prior, device=device)


# ===========================================================================
# constraint_loss (smoothness + separability)  (models.py:158-217)
# ===========================================================================
def smooth(arr, lamda1):
    """models.py:158-163 (1:1)."""
    arr2 = torch.zeros_like(arr)
    arr2[:-1] = arr[1:]
    arr2[-1] = arr[-1]
    loss = torch.sum((arr2 - arr) ** 2)
    return lamda1 * loss


def constraint_loss(dscores, wlabel, batch_size):
    """Paper's novel loss: smoothness of consecutive points + separability
    across labels — models.py:183-217 (1:1).

    Splits the batch dense scores into anomaly (wlabel>0) vs normal instances;
    separability pushes the anomaly mean above the normal mean; smoothness
    penalizes large consecutive jumps. (Unused upstream helpers `sparsity`,
    `ranking` are not vendored — not called in train().)
    """
    loss = torch.tensor(0., requires_grad=True)
    a_instance = []
    n_instance = []
    for i in range(batch_size):
        if wlabel[i] > 0:
            a_instance.append(dscores[i])
        else:
            n_instance.append(dscores[i])

    if a_instance and n_instance:
        L2 = loss
        L1 = loss
        for anomaly in a_instance:
            L2 = L2 + max(torch.mean(torch.stack(n_instance)) - torch.mean(anomaly), 0.0)
            L2 = L2 + max(smooth(anomaly, 8e-5), 0.0)
        for normal in n_instance:
            L1 = L1 + max(smooth(normal, 8e-5), 0.0)
        return (L1 + L2) / batch_size
    else:
        if a_instance is None:
            L1 = loss
            L1 = L1 + max(smooth(torch.stack(n_instance), 8e-5), 0.0)
            return L1 / batch_size
        elif n_instance is None:
            L3 = loss
            L3 = L3 + max(torch.max(torch.stack(a_instance)) - torch.mean(torch.stack(a_instance)), 0.0)
            L3 = L3 + max(smooth(torch.stack(a_instance), 8e-5), 0.0)
            return L3 / batch_size

    return loss / batch_size
