"""
NRdetector — in-memory `fit/predict` pipeline adapter (Phase-2 porting).

Wraps the OFFICIAL vendored method (`model.py`) into the comparison pipeline
contract. The upstream code is a CLI `Solver` (solver.py) driven by argparse +
disk `.npy`/`.graphml`/`.csv` artifacts with `DEBUG=True` reloads and hardcoded
`cuda:7`/`cuda:1`; here the SAME computation is re-hosted entirely in memory.

Provenance: https://github.com/UCSC-REAL/NRdetector (commit bd5592b, MIT,
UCSC-REAL / Yang Liu lab). Paper: KDD'25, arXiv:2501.11959.

Two-stage method (paper: coarse PU learning → fine point detection), realized
upstream as 4 sub-stages orchestrated by `Solver`. Mapping CLI → in-memory:

  Stage 0  encoder train + dense/pooled scores   ← solver.begin()/get_h_scores
  Stage 1  PU-LP reliable-positive/negative sel. ← selector.PU_LP (in-memory)
  Stage 2  PU classifier train (per-epoch cb)    ← solver.train()
  Stage 3a rank_test binarization                DROPPED (pipeline thresholds)
  Stage 3b HOC (resnet18) threshold              DROPPED (Phase-1 G4)

Key fidelity decisions (Phase-1 §06/§07/§10/§12; cite repo file:line):
  - Encoder trained per-dataset (NO general checkpoint exists upstream; the repo
    *loads* a pretrained `.pth` (extractor.py:64) whose producing recipe (WETAS)
    is external/absent, and the only shipped checkpoint is EMG-8feat-specific).
    We re-init per dataset to match `n_features`. Encoder loss = the OFFICIAL
    combination `bce(wscore, wlabel) + dtw_loss(output, wlabel).mean(0)`
    (extractor.py:127-129, EQUAL-WEIGHT sum). The soft-DTW alignment loss
    (`dtw_loss`, extractor.py:128/313-331) was RESTORED 2026-06-01 (MM2 closed):
    the soft-DTW kernel is REUSED via import from
    `comparison.baselines.wetas.softdtw_cuda` (MIT, Maghoumi), whose CPU numba
    `@jit` path (use_cuda=False) needs NO pip install / NO CUDA. See
    SOFTDTW_RESTORE_IMPL_NOTES.md. `encoder_epochs=50` / encoder `lr=1e-3` have
    NO official recipe (the repo never trains in-repo); flagged UNKNOWN for a human.
  - `win_size=100` non-overlapping windows (main.py:96; KEPT, not pipeline 500).
  - `wlabel = max(point labels over window)` (data_loader.py:69).
  - PU `noisy_rate=0.4` keeps the first 40% of POSITIVE train segments as
    labeled-P, demotes the rest to unlabeled (selector.py:31-38; main.py:102).
    Applied on the TRAIN split ONLY → leak-free.
  - Stage-2 training set EXACTLY matches official `selector.train_()`
    (selector.py:62-75): `RP = labeled positives (D+) only`, `RN = lp_n` from the
    in-memory label-propagation port of `calc_lp` (selector.py:139-240). The
    calc_pu-expanded RP is DISCARDED by `train_()` and is NOT fed to the
    classifier — see `_pulp_select` (MM1 fix).
  - `prior=0.25` FIXED — the official runtime constant (`main.py:98 --prior
    default=0.25`, consumed verbatim at `solver.py:116 create_loss(args.prior)`;
    the per-dataset `CLASS_PRIOR` map `models.py:6-13` is DEAD/never read at
    runtime). Faithfulness fix 2026-06-04: the prior is the executed upstream
    constant 0.25 for every dataset (preset `baseline_common.py:351/363`); the
    earlier train-wlabel-rate estimation (clip[0.05,0.5]) is NOT applied.
  - Per-split z-score StandardScaler (data_loader.py:50-55,102-104, paper §5.2):
    the official `_preprocess` fits a FRESH StandardScaler on EACH split's own
    data (`scaler.fit(data); scaler.transform(data)`) and is called separately
    for train/valid/test (data_loader.py:102-104) — train fits on train, test
    FITS ON TEST. This is TRANSDUCTIVE fit-on-test (label-free, NOT a leak) and
    is the only concrete norm spec upstream gives (paper §5.2 only says "following
    Xu 2021", no fit-scope). As of the SELF_NORMALIZING_WEAK reclassification
    (run_baseline) this wrapper receives RAW data and applies the official
    normalization itself: fit() fits StandardScaler(s) on raw train_X; predict()
    fits a FRESH StandardScaler per ENTITY slice on that slice's OWN test data and
    transforms it (data_loader.py:53-55). Per-entity slices come from
    `entity_test_slices` (same source as windowing); single-entity / None → one
    fresh fit over the whole test array. The train scaler is NOT reused on test
    (upstream never does). Faithfulness fix 2026-06-04: predict() previously
    reused the cached TRAIN scaler (train-fit `.transform`); restored to the
    upstream fit-on-test per-split scheme.
  - predict() point score = the official ACTMAP = per-window min-max of
    `h = fc(out)` (extractor.py get_dpred `actmap=(h-min)/max`; assigned to
    `self.dense_scores`, solver.py:72 "Actually is the act map"), GATED by the
    PU-classifier's official BINARY segment decision
    `seg_prob >= mean(seg_prob) + anomaly_thre*(max-min)`, `anomaly_thre=0`
    (solver.test:L164-171; main.py default 0.0), computed over the whole test
    split. Non-flagged windows score 0; flagged windows keep their continuous
    per-window actmap. THIS GATE IS INTEGRAL TO UPSTREAM, NOT optional:
      default run `python main.py` (--mode default 'train') runs ONLY
      `solver.rank_test()` (main.py:44-48; begin/train/test/pick_test all
      commented). rank_test ranks `point_Score = self.interested_instance`
      (solver.py:219), and `interested_instance` is built by `save_instance_files`
      keeping ONLY windows where `instance_label[i] > 0` (solver.py:198-203),
      where `instance_label` is the classifier's per-window mean-gate decision
      (solver.py:164-171,185). So upstream ranks the classifier-flagged-window
      actmap — the gate selects which points exist in the ranked pool.
    ⚠ DO NOT REMOVE THIS GATE. Removing it (commit 5cff9da, 2026-06-03, reverted)
      was a DOUBLE regression: (1) the encoder is trained ONCE in Stage 0 then
      FROZEN; only the PU-classifier trains in the per-epoch loop, so an
      encoder-only (ungated) actmap is IDENTICAL at every epoch → per-epoch
      scores became bit-identical (best-epoch selection meaningless); (2) the
      raw ungated actmap floods the ranking with normal-window points → ranking
      collapses. Measured (inference-only, saved weights): SWaT pak_auc_f1
      0.858→0.440, roc_auc 0.883→0.365 when the gate is removed. HOC top-k
      binarization is DROPPED (the harness selects the operating point). Length
      == len(test_X). (Classifier-gate is the faithful object; the brief
      no-gate variant 5cff9da was the bug. Earlier `× sigmoid(seg_prob)` soft
      relaxation also gated by the classifier but is less faithful than the
      upstream hard mean-gate used here.)
  - device-agnostic (no cuda:N, no import-time CUDA_VISIBLE_DEVICES).
"""

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from comparison.segment_utils import compute_segment_safe_window_indices
# Boundary-safe TEST windowing helper (shared single source of truth): runs the RAW
# score producer per-entity so no window spans an entity boundary on multi-file datasets
# (SMD machines / SMAP-MSL channels / Exathlon apps). Single-entity / None -> bit-identical no-op.
from comparison.baselines._boundary_safe_window import per_entity_concat, entity_test_slices
# Per-source-FILE leak-free normalization kernel (shared single source of truth).
# nrdetector keeps its OWN scaler identity (StandardScaler) — the helper only owns
# the per-file segment looping + leak-free fit-train / transform-test pairing.
from comparison.baselines._per_file_norm import (
    fit_transform_train_per_file,
)
# soft-DTW kernel REUSED from the already-vendored WETAS copy (MIT, Maghoumi).
# CPU numba @jit path (use_cuda=False) — NO pip install, NO second vendored copy.
# Evidence (byte diff vs official NRdetector modules/softdtw_cuda.py): same kernel;
# the CPU _SoftDTW/_SoftAlign @jit path our pipeline uses is BYTE-IDENTICAL. See
# SOFTDTW_RESTORE_IMPL_NOTES.md.
from comparison.baselines.wetas.softdtw_cuda import SoftDTW
from .model import (
    DilatedCNN,
    Classifier_six_layer,
    create_loss,
    constraint_loss,
    CLASS_PRIOR,
)


class NRdetectorBaseline:
    """NRdetector weak-supervised baseline (Q1 only; Q3 normalonly = N/A)."""

    def __init__(self, **hparams):
        self.name = "NRdetector"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # SoftDTW kernel: FORCED CPU @jit path (use_cuda=False) — 2026-06-01.
        # Rationale: this env's `numba.cuda` cannot init (`cuInit → CUDA_ERROR_NO_DEVICE 100`)
        # even on an idle GPU and with no torch context — verified via 3-case test
        # matrix (numba alone / torch-first / numba-first). numba is a conda-managed
        # dep and CLAUDE.md forbids `pip install` in `dc_vis`. The CPU `@jit` path
        # is the only viable kernel; performance overhead is contained because the
        # encoder/classifier still run on GPU via torch — only the soft-DTW alignment
        # loss is CPU-bound (batch_size=32 small).
        # Original (kept for reference): self.use_cuda = (self.device.type == 'cuda')
        self.use_cuda = False

        # ---- data / windowing (main.py:96-97) ----
        self.win_size = hparams.get('win_size', 100)
        self.seq_len = self.win_size                 # pipeline alias
        self.train_stride = hparams.get('train_stride', 1)  # unused: NRdetector windows are non-overlapping

        # ---- encoder (DilatedCNN, main.py:62-72) ----
        self.hidden_size = hparams.get('hidden_size', 64)
        self.output_size = hparams.get('output_size', 64)
        self.kernel_size = hparams.get('kernel_size', 2)
        self.n_layers = hparams.get('n_layers', 7)
        self.d_model = hparams.get('d_model', 64)
        self.pooling_type = hparams.get('pooling_type', 'avg')
        self.local_threshold = hparams.get('local_threshold', 0.3)
        self.granularity = hparams.get('granularity', 4)
        self.beta = hparams.get('beta', 0.1)
        self.gamma = hparams.get('gamma', 0.1)  # soft-DTW smoothing (main.py:74)
        self.encoder_epochs = hparams.get('encoder_epochs', 50)  # Stage-0 cap (recipe not in repo; see docs)
        self.encoder_lr = hparams.get('encoder_lr', 1e-4)        # Stage-0 Adam lr = WETAS DiCNN recipe (donalee/WETAS train_classifier.py:113,232-234); 1e-3 over-fit BCE->0, 1e-5 was classifier lr (wrong component)
        self.encoder_bce_min = hparams.get('encoder_bce_min', 0.05)  # Stage-0 BCE early-stop: halt when epoch-mean BCE <= this (prevents actmap-collapsing memorization; see docs)

        # ---- classifier / optimization (main.py:78-80, solver.py:109) ----
        self.classifier_hidden = hparams.get('classifier_hidden', 128)
        self.num_classes = hparams.get('num_classes', 1)
        self.batch_size = hparams.get('batch_size', 32)
        self.epochs = hparams.get('epochs', hparams.get('n_epochs', 200))  # classifier epochs
        self.lr = hparams.get('lr', hparams.get('learning_rate', 1e-5))

        # ---- PU / label (main.py:98-102, selector.py:16-17) ----
        self.noisy_rate = hparams.get('noisy_rate', 0.4)
        self.prior = hparams.get('prior', 0.25)      # FIXED upstream default (main.py:98); not estimated
        self.pulp_m = hparams.get('pulp_m', 4)
        self.pulp_lmbda = hparams.get('pulp_lmbda', 0.32)
        self.knn_k = hparams.get('knn_k', 5)
        self.pulp_a = hparams.get('pulp_a', 0.0005)  # utils.MatricesCalc.arg_a

        self.seed = hparams.get('seed', 0)
        self.verbose = hparams.get('verbose', True)

        self.encoder: Optional[DilatedCNN] = None
        self.classifier: Optional[Classifier_six_layer] = None
        self.scaler = None                           # per-split StandardScaler (data_loader.py:53-55)
        self._scalers = None                         # per-source-FILE TRAIN scalers (leak-free test transform)
        self.n_features = 0
        self._h_dim = self.output_size
        self.train_loss_history = []  # classifier-stage epoch-mean BCE+const loss (Stage 2 only).

        # Resume infrastructure (2026-05-31, B option).
        # NRdetector has 3 stages: (0) encoder train, (1) PU-LP select, (2) classifier train.
        # The visible "epoch" passed to epoch_callback is the Stage-2 classifier
        # epoch. Checkpoints save the FULL state needed to skip Stages 0+1 on
        # resume (encoder weights + RP/RN indices + scaler) so resume jumps
        # straight to the Stage-2 classifier loop.
        self._resume_checkpoint_path: Optional[Path] = None
        self._checkpoint_dir: Optional[Path] = None
        # Stage-0/1 outputs populated during fit() — captured into checkpoints
        # so resume doesn't need to re-run encoder training or PU-LP selection.
        self._h_scores_cache: Optional[np.ndarray] = None
        self._RP_cache: Optional[np.ndarray] = None
        self._RN_cache: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Windowing — data_loader.py:_preprocess (front-pad, non-overlapping)
    # ------------------------------------------------------------------
    def _window(self, data: np.ndarray, label: Optional[np.ndarray]):
        """Front-pad + split into non-overlapping win_size windows.

        Mirrors data_loader.py:59-69 (`f.pad(..., split_size - n%split_size, 0)`
        pads at the START). Returns (windows (Nw,T,F), dlabel (Nw,T) or None,
        wlabel (Nw,) or None, pad) — pad = #front-padded timesteps.
        """
        x = torch.as_tensor(data, dtype=torch.float32)
        n = x.shape[0]
        rem = n % self.win_size
        pad = (self.win_size - rem) if rem != 0 else 0
        x = torch.nn.functional.pad(x, (0, 0, pad, 0), 'constant', 0)
        x = x.unsqueeze(0)
        x = torch.cat(torch.split(x, self.win_size, dim=1), dim=0)   # (Nw, T, F)

        dlabel = wlabel = None
        if label is not None:
            y = torch.as_tensor(label, dtype=torch.float32)
            y = torch.nn.functional.pad(y, (pad, 0), 'constant', 0)
            y = y.unsqueeze(0)
            y = torch.cat(torch.split(y, self.win_size, dim=1), dim=0)  # (Nw, T)
            dlabel = y
            wlabel = torch.max(y, dim=1)[0]                             # data_loader.py:69
        return x, dlabel, wlabel, pad

    # ------------------------------------------------------------------
    # Stage 0 — train the WETAS-style encoder (OFFICIAL BCE + soft-DTW loss)
    # ------------------------------------------------------------------
    def _train_encoder(self, windows: torch.Tensor, wlabel: torch.Tensor):
        # soft-DTW kernel (CPU @jit when use_cuda=False) — extractor.py:42.
        dtw = SoftDTW(use_cuda=self.use_cuda, gamma=self.gamma, normalize=False)
        self.encoder = DilatedCNN(
            input_size=self.n_features, hidden_size=self.hidden_size,
            output_size=self.output_size, kernel_size=self.kernel_size,
            n_layers=self.n_layers, pooling_type=self.pooling_type,
            granularity=self.granularity, local_threshold=self.local_threshold,
            beta=self.beta, split_size=self.win_size, dtw=dtw,
        ).to(self.device)
        opt = torch.optim.Adam(self.encoder.parameters(), lr=self.encoder_lr)
        bce = nn.BCELoss(reduction='mean')   # upstream weak-segment loss (extractor.py:68,127)

        ds = torch.utils.data.TensorDataset(windows, wlabel)
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=False)
        self.encoder.train()
        for ep in range(self.encoder_epochs):
            ep_bce_sum, ep_dtw_sum, ep_n = 0.0, 0.0, 0
            for bx, bw in dl:
                bx = bx.to(self.device); bw = bw.to(self.device)
                opt.zero_grad()
                out = self.encoder.get_scores(bx)
                # OFFICIAL combination (extractor.py:127-129, EQUAL-WEIGHT sum):
                bce_loss = bce(out['wscore'], bw)                          # extractor.py:127
                dtw_loss = self.encoder.dtw_loss(out['output'], bw).mean(0)  # extractor.py:128 (B,)->scalar
                loss = bce_loss + dtw_loss                                  # extractor.py:129
                loss.backward()
                opt.step()
                nb = bx.size(0)
                ep_bce_sum += bce_loss.item() * nb   # sample-weighted -> epoch-mean (stable; not last-batch)
                ep_dtw_sum += dtw_loss.item() * nb
                ep_n += nb
            ep_bce = ep_bce_sum / max(ep_n, 1)
            ep_dtw = ep_dtw_sum / max(ep_n, 1)
            if self.verbose and (ep + 1) % 10 == 0:
                print(f"  [NRdetector enc] epoch {ep+1}/{self.encoder_epochs} "
                      f"bce={ep_bce:.4f} dtw={ep_dtw:.4f}", flush=True)
            # BCE early-stop (2026-06-13): halt BEFORE the encoder memorizes the weak
            # segment labels. BCE->0 saturates the per-window dense logit and collapses
            # the min-max actmap that IS the point-level anomaly score (wrapper.py:809,878).
            # Threshold self.encoder_bce_min (default 0.05) is a sweepable knob.
            if ep_bce <= self.encoder_bce_min:
                if self.verbose:
                    print(f"  [NRdetector enc] early-stop at epoch {ep+1}/{self.encoder_epochs} "
                          f"(epoch-mean bce={ep_bce:.4f} <= {self.encoder_bce_min})", flush=True)
                break

    @torch.no_grad()
    def _encode(self, windows: torch.Tensor):
        """Return pooled h (Nw,hidden) and dense dscore (Nw,T), batched."""
        self.encoder.eval()
        H, D = [], []
        for i in range(0, len(windows), 256):
            bx = windows[i:i + 256].to(self.device)
            out = self.encoder.get_scores(bx)
            H.append(out['h'].cpu()); D.append(out['dscore'].cpu())
        return torch.cat(H, 0).numpy(), torch.cat(D, 0).numpy()

    @torch.no_grad()
    def _encode_test_for_graph(self, test_X, test_segments) -> Optional[np.ndarray]:
        """Encode TEST windows into pooled embeddings `h_test` for the
        TRANSDUCTIVE PU-LP graph (FIX #1=B). Returns ``None`` when ``test_X`` is
        not supplied (graceful no-op → train-only graph).

        Faithfully mirrors how upstream feeds the TEST split into the graph:
          - TEST is z-scored fit-on-test per-split (data_loader._preprocess
            :50-55 fits a FRESH StandardScaler on the test data itself, called
            for the 'test' split :104). Here that is done per ENTITY slice from
            `test_segments` (SAME source/scheme predict() uses via
            entity_test_slices); single-entity / None → one fresh scaler over the
            whole test array.
          - windowed with the SAME front-pad, non-overlapping win_size split as
            train (data_loader._preprocess :59-61; `_window`).
          - encoded by the SAME frozen Stage-0 encoder (`_encode` → pooled `h`),
            exactly the embedding the graph is built on (get_h_scores → scores).
        Only the pooled `h` is needed (the graph uses embeddings, not dense
        scores); test windows are graph-structure-only and never enter RP/RN
        selection (restricted to the train range in `_pulp_select`).
        """
        if test_X is None:
            return None
        from sklearn.preprocessing import StandardScaler
        test_X = np.asarray(test_X, dtype=np.float32).copy()
        n_test = len(test_X)
        if n_test == 0:
            return None
        # BOUNDARY-SAFE per-entity processing — upstream windows EACH file
        # independently (data_loader._load_data loop + _preprocess :59-61 front-pad
        # + split PER split/file). On a multi-entity concat array we therefore
        # normalize (fit-on-test, data_loader.py:50-55,104) AND window each entity
        # slice in ISOLATION so no graph window straddles an entity boundary, then
        # concatenate the per-WINDOW embeddings. Single-entity / None → one slice
        # over the whole array == the simple front-pad+split path.
        H = []
        for (lo, hi) in entity_test_slices(n_test, test_segments):
            seg = test_X[lo:hi]
            if len(seg) == 0:
                continue
            sc = StandardScaler()                                   # FRESH per-entity, data_loader.py:53
            seg = np.asarray(sc.fit_transform(seg), dtype=np.float32)  # fit+transform on TEST itself :54-55
            twindows, _, _, _ = self._window(seg, None)             # front-pad + non-overlapping win_size
            h_e, _ = self._encode(twindows)                         # frozen Stage-0 encoder → pooled h
            H.append(np.asarray(h_e, dtype=np.float32))
        if not H:
            return None
        return np.concatenate(H, axis=0)

    # ------------------------------------------------------------------
    # Stage 1 — PU-LP selector (in-memory kNN graph + W; selector.py)
    # ------------------------------------------------------------------
    def _pulp_select(self, h_scores: np.ndarray, wlabel: np.ndarray,
                     h_test: Optional[np.ndarray] = None):
        """Reliable-positive (RP) / reliable-negative (RN) selection.

        Faithful in-memory re-host of utils.MatricesCalc (cosine kNN → directed A
        → W via pinv(I-aA) - I) + selector.PU_LP.calc_pu (m-iteration ranking) +
        selector.PU_LP.calc_lp (kNN-graph label propagation). Replaces the
        upstream disk `.graphml`/`.csv` path and the EMG-specific `data_len=4353`
        (selector.py:58). 1-based indexing in the upstream `.loc[vi][vj-1]` is
        mapped to 0-based here.

        TRANSDUCTIVE GRAPH (FIX #1=B, 2026-06-04). Upstream builds the cosine-kNN
        adjacency A and Katz `W = pinv(I - aA) - I` over the WHOLE concatenated
        embedding set train+valid+TEST (data_loader.get_segment :106 ConcatDataset
        + :135 cat(train,valid,test); MatricesCalc fed the whole `scores`
        solver.py:43,49; selector.data_len = whole :58), while RP/RN SELECTION is
        restricted to the TRAIN+valid node-id range (solver.begin calls only
        selector.train_() :59, whose candidate pool P=train_DP / U=train_DU spans
        train+valid only; selector.py:62-75). The graph STRUCTURE (A, W, undirected
        kNN nbr for calc_lp) still spans the whole concat because calc_lp walks
        `self.G` (the whole-graph neighbours) and W indexes whole-graph columns.
        Here, when `h_test` is given the kNN graph + W are built over
        `concat(h_scores, h_test)` (train nodes [0,n_train), test nodes appended),
        but `pos_idx`/`neg_idx` (hence DP/DU and every RP/RN candidate) are mined
        ONLY from the train range — test nodes are graph-structure-only, never
        selectable. `h_test=None` → bit-identical train-only behaviour (graph ==
        train embeddings; the no-op path used before the harness wires test_X).

        MATCHES official `train_()` (selector.py:62-75):
            P, RP, RN = calc_pu(P, U)
            results, labels, lp_i, lp_n = calc_lp(P, RN, whole_U, "train")
            return P, lp_n
        i.e. the classifier trains on RP = `P` (labeled positives only) and
        RN = `lp_n` (label-propagation negatives). The calc_pu-expanded RP is
        DISCARDED by `train_()` (MM1 fix — calc_lp DOES drive the RN pool).

        - directed A (utils.py:120-129): `A[i, knn[i]]=1`, exactly k ones per row,
          NO symmetrization; W=pinv(I-aA)-I uses this directed A (MM3 fix).
        - undirected neighbor sets (`nbr`) reproduce the `nx.Graph G`
          (utils.py:101/125, add_edge symmetric) used by calc_lp.

        noisy_rate sub-sampling (selector.py:31-38): keep first
        int(noisy_rate*P) positives as labeled-P (D+); rest demoted to unlabeled.
        """
        from scipy.spatial.distance import cdist

        # n_train = number of TRAIN(+valid) candidate nodes. The RP/RN candidate
        # pool is restricted to [0, n_train); test nodes (appended below) are
        # graph-structure-only and never enter pos_idx/neg_idx (FIX #1=B).
        n_train = len(h_scores)
        pos_idx = [i for i in range(n_train) if wlabel[i] > 0]
        neg_idx = [i for i in range(n_train) if wlabel[i] <= 0]
        if len(pos_idx) == 0:
            raise RuntimeError("NRdetector requires labeled anomalies (Q1 only; Q3 N/A)")

        # --- TRANSDUCTIVE embedding set: graph over concat(train, test) ---
        # Upstream: MatricesCalc(scores) where scores = h over train+valid+TEST
        # (solver.py:43,49 + data_loader.py:106,135). When h_test is provided the
        # graph spans the whole concat; otherwise it is train-only (no-op path).
        if h_test is not None and len(h_test) > 0:
            graph_h = np.concatenate([h_scores, np.asarray(h_test, dtype=h_scores.dtype)], axis=0)
        else:
            graph_h = h_scores
        n = len(graph_h)   # whole-graph node count (train + test)

        # --- noisy_rate sub-sample of positives (TRAIN ONLY) — selector.py:31-38 ---
        known = int(self.noisy_rate * len(pos_idx))
        DP = list(pos_idx[:known])           # labeled positives (first-N by order, upstream caveat)
        DU = list(pos_idx[known:]) + list(neg_idx)  # demoted positives + all negatives
        # NB: DP/DU draw exclusively from pos_idx/neg_idx ⊂ [0, n_train) → test
        # node-ids [n_train, n) are NEVER candidates (selector.train_() :62-75).

        # --- cosine kNN: directed A (W path) + undirected nbr (calc_lp graph) ---
        # adjacency = cosine distance; smaller = more similar. Built over the WHOLE
        # graph (train+test when transductive) — MatricesCalc.calc_MatrizAdj
        # (utils.py:74-80) operates on the whole `scores`.
        Y = cdist(graph_h, graph_h, metric='cosine')
        np.fill_diagonal(Y, np.inf)          # exclude self (PU_LP_knn: index_i != name_j)
        knn = np.argsort(Y, axis=1)[:, :self.knn_k]
        # Directed adjacency — utils.py:123 `A.loc[index_i][vizinho]=1` (k ones/row,
        # NOT symmetrized). Feeds the W matrix only (MM3).
        A = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            A[i, knn[i]] = 1.0
        # Undirected neighbor sets — utils.py:125 `G.add_edge(i, vizinho)`
        # (nx.Graph is symmetric). Feeds calc_lp only (MM1). weight=1 for all edges.
        nbr = [set() for _ in range(n)]
        for i in range(n):
            for j in knn[i]:
                nbr[i].add(int(j)); nbr[int(j)].add(i)
        # W = (I - a*A)^-1 - I  (calc_knn_w_matrices: pinv(I - A.mul(arg_a)) - I)
        I = np.eye(n, dtype=np.float32)
        W = np.linalg.pinv(I - self.pulp_a * A).astype(np.float32) - I

        # --- calc_pu: iterative ranking to expand RP, then mine RN (selector.py:91-137) ---
        P = list(DP)
        U = list(DU)
        RP = []
        if len(P) > 0:
            top = int((self.pulp_lmbda / self.pulp_m) * len(P))
            for _ in range(self.pulp_m):
                if top <= 0 or len(U) == 0:
                    break
                # mean similarity of each unlabeled vi to current positive set P
                Svi = W[np.ix_(U, P)].mean(axis=1)
                order = np.argsort(Svi)[::-1][:top]   # highest-similarity unlabeled → new RP
                chosen = [U[k] for k in order]
                for c in chosen:
                    P.append(c); RP.append(c); U.remove(c)
        # RN = bottom (|RP|+|P|) unlabeled by mean similarity to (P+RP) — selector.py:119-134
        bottom = len(RP) + len(DP)
        if len(U) > 0 and bottom > 0:
            ref = list(set(DP + RP))
            Svi = W[np.ix_(U, ref)].mean(axis=1)
            rn_order = np.argsort(Svi)[:bottom]       # lowest similarity → reliable negatives
            RN = [U[k] for k in rn_order]
        else:
            RN = list(neg_idx)                        # fallback: all-negative segments

        # --- calc_lp: label propagation → lp_n (selector.py:139-240) -------------
        # Official train_() call: calc_lp(P=DP, CN=RN, whole_U). CP = labeled
        # positives (P here is DP — the expanded RP was appended in-place but
        # train_() passes the ORIGINAL `P` arg, which == DP since calc_pu mutates
        # only a copy for ranking; the returned P is DP unchanged). lp_n = nodes
        # whose argmax class = negative (class index 1).
        lp_n = self._calc_lp(CP=set(DP), CN=set(RN), U=list(DU), nbr=nbr)

        # --- official return: RP = P (= DP), RN = lp_n (MM1) ---
        RP_final = list(DP)
        if len(RP_final) == 0:
            RP_final = list(pos_idx)                  # safety (Q1 guarantees pos_idx≠∅)
        if len(lp_n) == 0:
            lp_n = list(RN) if RN else (list(neg_idx) if neg_idx else [int(np.argmin(wlabel))])
        return RP_final, list(lp_n)

    # ------------------------------------------------------------------
    # calc_lp — in-memory label propagation (selector.py:139-273), pure numpy.
    # ------------------------------------------------------------------
    def _calc_lp(self, CP: set, CN: set, U: list, nbr: list):
        """Port of `PU_LP.calc_lp` (selector.py:139-240) + `getCMN`
        (selector.py:242-258) + `argMaxClass`/`probClasse` (selector.py:260-293).

        For every node `i` in U:
          - i ∈ CP → class vector [1, 0]            (selector.py:144-145)
          - i ∈ CN → class vector [0, 1]            (selector.py:147-148)
          - else   → 1-hop (+2-hop) class-vote accumulation over the UNDIRECTED
                     kNN graph from CP/CN, with edge weight w=1 (selector.py:150-181).
        Class index 0 = positive, 1 = negative. `lp_n` = nodes whose
        argMaxClass == 1 (selector.py:230-238).

        Faithful reproduction of the official accumulation (including its 2-hop
        loop semantics, where each neighbor `key` of `i` contributes `temp`, and
        every neighbor `m` of `key` adds `temo`; non-CP/CN neighbors contribute
        [0,0]). cmn = per-class column sum over ALL nodes' raw vectors
        (selector.py:242-258), used by argMaxClass's `f[i]/cmn[i]` weighting.
        """
        n_nodes = len(U)
        vecs = np.zeros((n_nodes, 2), dtype=np.float64)
        pos_set = set(int(x) for x in CP)
        neg_set = set(int(x) for x in CN)
        U = [int(x) for x in U]

        for row, i in enumerate(U):
            if i in pos_set:
                vecs[row] = (1.0, 0.0)            # selector.py:144-145
                continue
            if i in neg_set:
                vecs[row] = (0.0, 1.0)            # selector.py:147-148
                continue
            # unlabeled node: 1-hop + 2-hop accumulation (selector.py:150-181)
            label = np.zeros(2, dtype=np.float64)
            for key in nbr[i]:                   # i_adj = G[i]
                w = 1.0                          # G.add_edge(weight=1)
                temp = np.zeros(2, dtype=np.float64)
                if key in pos_set:
                    temp = np.array([w, 0.0])    # selector.py:159-161
                elif key in neg_set:
                    temp = np.array([0.0, w])    # selector.py:162-163
                # 2-hop: neighbors of `key` (selector.py:166-177)
                for m in nbr[key]:
                    if m in pos_set:
                        temp = temp + np.array([w, 0.0])
                    elif m in neg_set:
                        temp = temp + np.array([0.0, w])
                label = label + temp             # selector.py:178-179
            vecs[row] = label

        # cmn = per-class column sum (selector.py:242-258)
        cmn = vecs.sum(axis=0)

        lp_n = []
        for row, i in enumerate(U):
            arg = self._arg_max_class(vecs[row], cmn)
            if arg == 1:                         # negative class → lp_n (selector.py:230-238)
                lp_n.append(i)
        return lp_n

    @staticmethod
    def _arg_max_class(f, cmn):
        """argMaxClass (selector.py:260-273) + probClasse (selector.py:282-293)."""
        s = float(f[0] + f[1])
        if s == 0.0:
            prob = (0.0, 0.0)
        else:
            prob = (f[0] / s, f[1] / s)
        if (prob[0] + prob[1]) == 0.0:
            return 1                             # isolated/no-vote node → negative (selector.py:264-265)
        arg_max = -1
        max_value = 2.2250738585072014e-308      # selector.py:267
        for idx in range(2):
            denom = cmn[idx] if cmn[idx] != 0 else 1e-12   # guard /0 (cmn[idx]==0 ⇒ f[idx]==0)
            value = prob[idx] * (f[idx] / denom)
            if value > max_value:
                arg_max = idx
                max_value = value
        return arg_max

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(self, train_X, train_y=None, epoch_callback=None, train_segments=None,
            norm_train_segs=None, test_X=None, test_segments=None) -> 'NRdetectorBaseline':
        """Fit NRdetector (Stages 0/1/2).

        TRANSDUCTIVE PU-LP graph (FIX #1=B, 2026-06-04). Upstream constructs the
        cosine-kNN adjacency + Katz W over the WHOLE concatenated embedding set
        train+valid+TEST (data_loader.get_segment :106/:135, MatricesCalc fed the
        whole scores solver.py:43,49, selector.data_len=whole :58), while RP/RN
        SELECTION is restricted to the TRAIN(+valid) node range (only
        selector.train_() runs, solver.py:59). When the harness supplies `test_X`
        (OPTIONAL), the test windows are normalized fit-on-test per-entity (same
        scheme as predict(), data_loader.py:50-55,102-104) and encoded with the
        SAME frozen Stage-0 encoder, then `_pulp_select` builds the graph over
        concat(h_train, h_test) but mines RP/RN from the train range only.
        `test_X=None` → graceful no-op = the prior train-only graph (so nothing
        breaks before the harness is wired; see SHARED_EDITS_v2.md). `test_segments`
        delineates per-entity test slices for the fit-on-test normalization (same
        source the harness passes to predict()); None → one fresh scaler over the
        whole test array."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        train_X = np.asarray(train_X, dtype=np.float32)
        self.n_features = train_X.shape[1]

        if train_y is None:
            raise RuntimeError("NRdetector requires labeled anomalies (Q1 only; Q3 N/A)")
        train_y = np.asarray(train_y).astype(np.float32).reshape(-1)
        if train_y.max() <= 0:
            raise RuntimeError("NRdetector requires labeled anomalies (Q1 only; Q3 N/A)")

        # ---- per-split z-score (data_loader.py:50-55): fit on RAW train, store scaler.
        #      Mirrors official `_preprocess`: scaler operates on the un-split (N,F)
        #      array BEFORE windowing. Wrapper now receives RAW data
        #      (run_baseline SELF_NORMALIZING_WEAK) → re-activated here.
        #      PER-SOURCE-FILE leak-free fix: fit one StandardScaler per source file
        #      on that file's TRAIN slice (multi-file: SMD machine / SMAP-MSL channel
        #      / Exathlon app). Single-file / norm_train_segs=None → one whole-array
        #      scaler == legacy behavior. Cache the per-file TRAIN scalers so predict()
        #      can transform each test file by its OWN train scaler (never fit on test).
        #      nrdetector's scaler identity = StandardScaler (data_loader.py:53). ----
        from sklearn.preprocessing import StandardScaler
        train_X, self._scalers, _ = fit_transform_train_per_file(
            train_X, norm_train_segs, lambda: StandardScaler())
        train_X = train_X.astype(np.float32)
        # Keep `self.scaler` = the FIRST per-file train scaler (single-file = the only
        # one) for the checkpoint/save fallback path (data_loader.py:50-55 fallback).
        self.scaler = self._scalers[0] if self._scalers else None

        # ---- window train_X (front-pad, win=100) + wlabel=max ----
        windows, dlabel, wlabel, _ = self._window(train_X, train_y)

        # ---- segment-safe drop (normalonly variant tolerance) ----
        if train_segments is not None:
            valid = compute_segment_safe_window_indices(
                train_segments, self.win_size, self.win_size, len(windows))
            if len(valid) > 0:
                windows = windows[valid]; dlabel = dlabel[valid]; wlabel = wlabel[valid]

        # ---- prior: per-dataset default or estimate from train wlabel rate ----
        if self.prior is None:
            rate = float((wlabel > 0).float().mean().item())
            self.prior = float(np.clip(rate, 0.05, 0.5)) if rate > 0 else 0.25
        if self.verbose:
            print(f"  [NRdetector] windows={len(windows)} feat={self.n_features} "
                  f"prior={self.prior:.3f} noisy_rate={self.noisy_rate}", flush=True)

        # Resume short-circuit: if a checkpoint exists, we restore the FULL
        # Stage-0/1 state from it (encoder weights + h_scores + RP/RN) so we
        # can skip the multi-minute encoder training + PU-LP selection and
        # jump straight to the Stage-2 classifier loop. Otherwise run all 3
        # stages from scratch.
        resume_state = self._try_load_resume_state()
        if resume_state is not None:
            self._restore_stage01_from_checkpoint(resume_state)
            h_scores = self._h_scores_cache
            RP = self._RP_cache
            RN = self._RN_cache
        else:
            # ---- Stage 0: encoder ----
            self._train_encoder(windows, wlabel)
            h_scores, _ = self._encode(windows)

            # ---- TRANSDUCTIVE: encode TEST windows for the PU-LP graph (FIX #1=B).
            #      Upstream's graph spans train+valid+TEST embeddings (get_segment
            #      :106/:135 → MatricesCalc solver.py:43,49). When the harness passes
            #      test_X, normalize it fit-on-test per-entity (SAME scheme as
            #      predict(), data_loader.py:50-55,102-104), window (front-pad,
            #      win=100, NO labels) and encode with the SAME frozen Stage-0
            #      encoder to get test embeddings `h_test`. test_X=None → h_test=None
            #      → train-only graph (no-op). ----
            h_test = self._encode_test_for_graph(test_X, test_segments)

            # ---- Stage 1: PU-LP select RP / RN (graph over concat(train, test)) ----
            RP, RN = self._pulp_select(h_scores, wlabel.numpy(), h_test=h_test)
            if self.verbose:
                print(f"  [NRdetector] PU-LP: |RP|={len(RP)} |RN|={len(RN)}", flush=True)
            # Cache Stage-0/1 outputs so subsequent per-epoch checkpoints can
            # include them — resume only needs Stage 2 to restart.
            # _pulp_select returns Python lists; coerce to ndarray here so
            # the cache has a stable dtype for serialization (.tolist() works,
            # checkpoint round-trips ndarray<->list-via-tolist cleanly).
            self._h_scores_cache = np.asarray(h_scores, dtype=np.float32)
            self._RP_cache = np.asarray(RP, dtype=np.int64)
            self._RN_cache = np.asarray(RN, dtype=np.int64)

        # ---- Stage 2: PU classifier (solver.get_dataset/train) ----
        pos = torch.tensor(h_scores[RP], dtype=torch.float32)
        neg = torch.tensor(h_scores[RN], dtype=torch.float32)
        inp = torch.cat([pos, neg], dim=0)
        lab = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))], dim=0)  # solver.py:94-95
        ds = torch.utils.data.TensorDataset(inp, lab)
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, drop_last=True, shuffle=True)

        self.classifier = Classifier_six_layer(
            self.d_model, hidden_size=self.classifier_hidden, num_classes=self.num_classes
        ).to(self.device)
        criterion = create_loss(self.prior, self.device)
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self.lr)  # solver.py:117

        # Stage-2 resume: restore classifier weights / optimizer / RNG / loss
        # history if we loaded a Stage-0/1 cache above; else start fresh at 0.
        start_epoch = self._restore_stage2_state(resume_state, optimizer) if resume_state else 0

        for epoch in range(start_epoch, self.epochs):
            # FAITHFULNESS (solver.py:124-141): upstream calls `optimizer.zero_grad()`
            # ONCE per epoch (solver.py:126, BEFORE the batch loop) while `optimizer.step()`
            # runs per batch (solver.py:141). Grads therefore ACCUMULATE across all batches
            # within an epoch (each step() walks on the running grad sum since the last
            # epoch's zero). We previously zero'd per batch (standard SGD); restored to the
            # upstream once-per-epoch zero so the optimization dynamics match.
            optimizer.zero_grad()                                            # solver.py:126 (once/epoch)
            self.classifier.train()
            epoch_losses = []
            for inputs, labels in dl:
                inputs = inputs.to(self.device); labels = labels.to(self.device)
                outputs, out = self.classifier(inputs)
                pu_loss = criterion(outputs, labels.unsqueeze(-1))          # solver.py:136
                const_loss = constraint_loss(out, labels, self.batch_size)  # solver.py:137
                loss = pu_loss + const_loss                                  # lamda=lamda_1=1, solver.py:138
                loss.backward()                                             # solver.py:140 (no zero here)
                optimizer.step()                                            # solver.py:141 (per batch)
                epoch_losses.append(float(loss.detach().cpu()))

            avg = float(np.mean(epoch_losses)) if epoch_losses else float('nan')
            self.train_loss_history.append(avg)

            # Save last.pt BEFORE epoch_callback (crash-in-eval safety).
            self._save_last_checkpoint(epoch + 1, optimizer)

            # bridge upstream loss-based early-stop → pipeline epoch_callback / pak_auc_f1
            if epoch_callback is not None:
                epoch_callback(self, epoch + 1)
                self.classifier.train()
        return self

    # ------------------------------------------------------------------
    # predict — RAW continuous gated score → (N_test,) float32
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, test_X, test_segments=None) -> np.ndarray:
        if self.encoder is None or self.classifier is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        test_X = np.asarray(test_X, dtype=np.float32)
        n_test = len(test_X)

        # ---- per-source-FILE FIT-ON-TEST z-score (data_loader.py:50-55,102-104): the
        #      official `_preprocess` fits a FRESH `StandardScaler` on EACH split's OWN
        #      data and is called for the 'test' split (data_loader.py:104), so the test
        #      split is z-scored by a scaler fit on the TEST data itself — per source FILE.
        #      This is TRANSDUCTIVE fit-on-test (label-free, NOT a label leak) and is the
        #      only concrete normalization spec upstream provides (paper §5.2 only says
        #      "following Xu 2021", no fit-scope). Here predict() receives RAW test_X (no
        #      per-recording boundaries beyond `test_segments`), so we fit a FRESH
        #      StandardScaler per ENTITY slice on that slice's own test data and transform
        #      it in place. Single-entity / test_segments=None → ONE fresh scaler over the
        #      whole test array (still fit-on-test). The cached TRAIN scalers (`self._scalers`)
        #      are NOT used for the test transform (upstream never reuses the train scaler
        #      on test). Done BEFORE windowing so the per-entity raw producer below receives
        #      an already-normalized array (NO re-normalization). ----
        from sklearn.preprocessing import StandardScaler
        test_X = test_X.copy()
        for (lo, hi) in entity_test_slices(n_test, test_segments):
            sc = StandardScaler()                              # FRESH per-entity, data_loader.py:53
            test_X[lo:hi] = np.asarray(
                sc.fit_transform(test_X[lo:hi]), dtype=np.float32   # fit+transform on TEST itself :54-55
            )
        test_X = test_X.astype(np.float32)

        self.encoder.eval(); self.classifier.eval()

        # ==============================================================================
        # BOUNDARY-SAFE TEST WINDOWING (2026-06-02)
        # ------------------------------------------------------------------------------
        # Upstream windows EACH recording FILE independently: `data_loader._load_data`
        # loads one file's data.npy per split and `_preprocess` front-pads + splits THAT
        # file into non-overlapping win_size windows, then the per-file windowed segments
        # are concatenated (re-verified live: data_loader.py `_load_data` loop +
        # `_preprocess` :52 `f.pad(..., split_size - n%split_size, 0)` :54 `cat(split(...))`).
        # On a multi-entity concatenated test array, our previous code front-padded ONCE
        # and chunked over the WHOLE array, so a single non-overlapping window could
        # straddle two entities (machine-k tail + machine-(k+1) head). That corrupts both
        # the per-window min-max ACTMAP and the classifier seg-prob for boundary windows.
        # Fix: run the RAW score producer (windowing + encoder/classifier inference +
        # per-window->per-timestep ACTMAP mapping incl. front-pad drop) INDEPENDENTLY on
        # each entity's own test slice via `per_entity_concat`, so no window crosses a
        # boundary. This is STRICTLY MORE faithful to upstream's per-file chunking.
        #
        # RAW-PRODUCER vs WHOLE-TEST POST-PROC split (kept identical in granularity):
        #   RAW (per-entity, boundary-safe): per-window min-max ACTMAP -> per-timestep
        #     (front-pad dropped), plus the window-level classifier seg-prob.
        #   POST-PROC (whole-test, UNCHANGED): the official BINARY segment gate
        #     `threshold = mean(seg) + anomaly_thre*(max-min)` is computed over the WHOLE
        #     concatenated test split's WINDOW-level seg scores (solver.test :158-160
        #     thresholds over `test_outputs[s_index:]` = all concatenated test windows;
        #     re-verified live). We therefore collect the window-level seg scores across
        #     ALL entities and apply the global threshold ONCE, AFTER per-entity windowing
        #     — preserving upstream's transductive whole-test gate granularity exactly.
        # Single-entity / None / non-tiling test_segments -> ONE call over the whole array
        # == bit-identical legacy behaviour (per_entity_concat guarantees the no-op; the
        # post-proc reduces to the original global mean/max/min over all windows).
        # ==============================================================================
        anomaly_thre = float(getattr(self, "anomaly_thre", 0.0))

        # Side-channel: window-level seg scores + per-timestep window-membership map,
        # captured per entity (in entity order) by raw_fn so the WHOLE-TEST gate can be
        # applied after concatenation. `_seg_blocks[k]` = (seg_e (Nw_e,), win_idx_e (Li,))
        # where win_idx_e[t] = the GLOBAL window index of real timestep t.
        _seg_blocks = []
        _win_offset = [0]   # running global window counter (mutable closure cell)

        def raw_fn(sub_X: np.ndarray) -> np.ndarray:
            """RAW per-timestep ACTMAP for ONE entity slice (boundary-safe windowing+inference).

            Reproduces upstream's per-FILE chunking (data_loader `_preprocess`:52-54):
            front-pad to a whole number of win_size windows, encoder -> per-window min-max
            ACTMAP (extractor.py get_dpred), classifier -> per-window seg-prob, flatten the
            ACTMAP and drop the FRONT pad -> exactly len(sub_X) per-timestep scores.
            The window-level seg-prob + a global per-timestep window index are stashed in
            `_seg_blocks` for the WHOLE-TEST gate (applied after concat, NOT here). Edge-safe:
            a slice shorter than win_size front-pads to ONE full window (`_window` pads to
            win_size), so short entities never crash. `sub_X` is already normalized."""
            sub_X = np.asarray(sub_X, dtype=np.float32)
            li = len(sub_X)
            if li == 0:
                _seg_blocks.append((np.zeros(0, np.float32), np.zeros(0, np.int64)))
                return np.zeros(0, dtype=np.float32)

            windows, _, _, pad = self._window(sub_X, None)   # non-overlapping, front-padded
            act_chunks = []
            seg_chunks = []
            for i in range(0, len(windows), 256):
                bx = windows[i:i + 256].to(self.device)
                out = self.encoder.get_scores(bx)
                h = out['dh']                                  # (B, T) raw logit (model.py get_scores)
                actmin = torch.min(h, dim=1, keepdim=True)[0]  # extractor.py get_dpred
                act = h - actmin
                actmax = torch.max(act, dim=1, keepdim=True)[0]
                act = act / (actmax + 1e-12)                   # per-window min-max → [0,1]
                # classifier features = raw `h` (= out['h']), matching weak_scores
                #   (extractor.py `wscores.append(out['h'])`, solver.py).
                seg_prob, _ = self.classifier(out['h'])        # (B, 1) classifier segment prob
                act_chunks.append(act.cpu().numpy())           # (B, T) actmap
                seg_chunks.append(seg_prob.view(-1).cpu().numpy())  # (B,) seg prob
            act_e = np.concatenate(act_chunks, axis=0)         # (Nw_e, T)
            seg_e = np.concatenate(seg_chunks, axis=0)         # (Nw_e,)
            nw_e = act_e.shape[0]

            # per-timestep ACTMAP for this entity (front-pad dropped) — RAW producer output.
            act_flat = act_e.reshape(-1)[pad:pad + li].astype(np.float32)
            if len(act_flat) < li:                             # safety (shouldn't trigger)
                fill = act_flat[-1] if len(act_flat) else 0.0
                act_flat = np.concatenate([act_flat, np.full(li - len(act_flat), fill, np.float32)])
            act_flat = act_flat[:li]

            # GLOBAL per-timestep window index (front-pad dropped) for the whole-test gate.
            base = _win_offset[0]
            win_idx_full = np.repeat(np.arange(base, base + nw_e, dtype=np.int64), self.win_size)
            win_idx_e = win_idx_full[pad:pad + li]
            if len(win_idx_e) < li:                            # safety mirror of act fill
                last = win_idx_e[-1] if len(win_idx_e) else base
                win_idx_e = np.concatenate([win_idx_e, np.full(li - len(win_idx_e), last, np.int64)])
            win_idx_e = win_idx_e[:li]
            _win_offset[0] = base + nw_e
            _seg_blocks.append((seg_e.astype(np.float32), win_idx_e))
            return act_flat

        # Boundary-safe: per-entity windowing+inference of the ACTMAP, concatenated to
        # (N_test,). Single-entity / None / non-tiling test_segments -> ONE raw_fn(test_X)
        # == legacy behaviour (helper no-op; _seg_blocks holds a single block).
        act_ts = per_entity_concat(test_X, test_segments, raw_fn)   # (N_test,) per-timestep actmap

        # ---- WHOLE-TEST POST-PROC: official BINARY segment gate (solver.test:158-160).
        #        threshold = mean(seg) + anomaly_thre*(max-min)   (anomaly_thre=0, main.py)
        #        flagged_window = seg_prob >= threshold
        #      Computed over the WHOLE concatenated test split's WINDOW-level seg scores,
        #      EXACTLY as upstream (transductive whole-test gate). Only flagged windows
        #      contribute candidate points; the per-window min-max actmap ranks points
        #      WITHIN flagged windows. Non-flagged windows score 0 (= least anomalous),
        #      the contract-safe continuous analog of upstream's binary pred=0. ----
        seg_all = (np.concatenate([blk[0] for blk in _seg_blocks])
                   if _seg_blocks else np.zeros(0, np.float32))   # (Nw_total,) window-level
        if len(seg_all) > 0:
            seg_thr = seg_all.mean() + anomaly_thre * (seg_all.max() - seg_all.min())
            gate_win = (seg_all >= seg_thr).astype(np.float32)    # (Nw_total,) binary, window-level
        else:
            gate_win = np.zeros(0, np.float32)

        # Broadcast the global binary window gate back to each entity's timesteps via the
        # stashed GLOBAL window index, then multiply with the concatenated actmap. This
        # reproduces the original `act_all * gate` -> reshape -> pad-drop mapping exactly,
        # now boundary-safe (per-entity windowing) with an UNCHANGED whole-test gate.
        gate_ts = np.empty(n_test, dtype=np.float32)
        pos = 0
        for (_seg_e, win_idx_e) in _seg_blocks:
            li = len(win_idx_e)
            if li == 0:
                continue
            gate_ts[pos:pos + li] = gate_win[win_idx_e] if len(gate_win) else 0.0
            pos += li
        if pos != n_test:                                   # safety (shouldn't trigger)
            gate_ts = np.resize(gate_ts, n_test)

        scores = act_ts * gate_ts
        scores = np.nan_to_num(scores[:n_test], nan=0.0, posinf=0.0, neginf=0.0)
        return scores.astype(np.float32)

    # ------------------------------------------------------------------ resume (2026-05-31, B)
    def set_resume(self, ckpt_path) -> None:
        """Enable resume (called pre-fit())."""
        self._resume_checkpoint_path = Path(ckpt_path) if ckpt_path else None

    def set_checkpoint_dir(self, ckpt_dir) -> None:
        """Enable per-epoch save (called pre-fit())."""
        self._checkpoint_dir = Path(ckpt_dir) if ckpt_dir else None

    def _try_load_resume_state(self) -> Optional[dict]:
        """Load checkpoint dict if a valid last.pt exists, else None.

        Loaded ONCE up-front in fit() so we know whether to skip Stages 0/1
        and how to populate h_scores/RP/RN. Returning None means "from scratch".
        """
        if self._resume_checkpoint_path is None or not self._resume_checkpoint_path.exists():
            return None
        from comparison.baselines._checkpoint import load_checkpoint
        ckpt = load_checkpoint(self._resume_checkpoint_path)
        ck_hp = ckpt.get('wrapper_hparams', {})
        # Widened 2026-05-31 (Bug 4) — kernel_size/output_size affect
        # encoder shape; classifier_hidden affects classifier shape;
        # noisy_rate affects PU-LP selection (already cached, but flag
        # mismatch as a sanity guard).
        for key in ('win_size', 'hidden_size', 'output_size', 'kernel_size',
                    'd_model', 'n_layers', 'classifier_hidden', 'noisy_rate',
                    'gamma', 'n_features'):
            if ck_hp.get(key) != getattr(self, key):
                raise RuntimeError(
                    f"NRdetector resume hparam mismatch on '{key}': "
                    f"checkpoint={ck_hp.get(key)} vs current={getattr(self, key)}."
                )
        return ckpt

    def _restore_stage01_from_checkpoint(self, ckpt: dict) -> None:
        """Rebuild encoder + restore h_scores/RP/RN from a resume checkpoint.

        After this returns, ``self.encoder`` is on device with the saved
        weights and ``self._{h_scores,RP,RN}_cache`` are populated, so the
        Stage-2 classifier loop can run as if Stages 0/1 had just finished.
        """
        # Rebuild encoder with the same architecture as the saved one.
        # Attach a soft-DTW kernel (inference-inert here, but prevents a crash
        # if dtw_loss is ever called post-resume) — SOFTDTW_RESTORE.
        dtw = SoftDTW(use_cuda=self.use_cuda, gamma=self.gamma, normalize=False)
        self.encoder = DilatedCNN(
            input_size=self.n_features, hidden_size=self.hidden_size,
            output_size=self.output_size, kernel_size=self.kernel_size,
            n_layers=self.n_layers, pooling_type=self.pooling_type,
            granularity=self.granularity, local_threshold=self.local_threshold,
            beta=self.beta, split_size=self.win_size, dtw=dtw,
        ).to(self.device)
        self.encoder.load_state_dict(ckpt['encoder_state_dict'])
        self.encoder.eval()
        # Scaler / prior were already re-fit deterministically in fit(), but
        # restoring them explicitly guards against any future drift in the
        # upstream preprocessing.
        from comparison.baselines._checkpoint import dict_to_standard_scaler
        restored_scaler = dict_to_standard_scaler(ckpt.get('scaler_state'))
        if restored_scaler is not None:
            self.scaler = restored_scaler
        if 'prior' in ckpt:
            self.prior = float(ckpt['prior'])
        # Stage-0/1 outputs.
        self._h_scores_cache = np.asarray(ckpt['h_scores'], dtype=np.float32)
        self._RP_cache = np.asarray(ckpt['RP'], dtype=np.int64)
        self._RN_cache = np.asarray(ckpt['RN'], dtype=np.int64)
        if self.verbose:
            print(f"  [RESUME] NRdetector restored Stage-0/1 from {self._resume_checkpoint_path.name} "
                  f"(|RP|={len(self._RP_cache)} |RN|={len(self._RN_cache)})", flush=True)

    def _restore_stage2_state(self, ckpt: dict, optimizer) -> int:
        """Restore classifier weights / optimizer / RNG / loss history.

        Returns the next classifier epoch to train (= ``ckpt['epoch']``).
        """
        from comparison.baselines._checkpoint import restore_rng_state
        self.classifier.load_state_dict(ckpt['classifier_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.train_loss_history = list(ckpt.get('train_loss_history', []))
        restore_rng_state(ckpt.get('rng_state'))
        start_epoch = int(ckpt['epoch'])
        if self.verbose:
            print(f"  [RESUME] NRdetector classifier from epoch {start_epoch}/{self.epochs}",
                  flush=True)
        return start_epoch

    def _save_last_checkpoint(self, current_epoch: int, optimizer) -> None:
        """Persist FULL state needed to skip Stages 0/1 on resume.

        Includes encoder weights + h_scores + RP/RN + classifier weights +
        optimizer + RNG + scaler + prior. h_scores and RP/RN add a few MB to
        the file but eliminate the encoder-retrain + PU-LP-rerun overhead on
        each resume (otherwise ~minutes wasted before Stage-2 can start).
        """
        if self._checkpoint_dir is None:
            return
        from comparison.baselines._checkpoint import (
            atomic_save, capture_rng_state, standard_scaler_to_dict,
            CHECKPOINT_VERSION,
        )
        ckpt = {
            'checkpoint_version': CHECKPOINT_VERSION,
            'model_name': self.name,
            'epoch': current_epoch,
            'target_epochs': self.epochs,
            'encoder_state_dict': self.encoder.state_dict() if self.encoder is not None else None,
            'classifier_state_dict': self.classifier.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss_history': list(self.train_loss_history),
            'h_scores': (
                self._h_scores_cache.tolist() if self._h_scores_cache is not None else None
            ),
            'RP': self._RP_cache.tolist() if self._RP_cache is not None else None,
            'RN': self._RN_cache.tolist() if self._RN_cache is not None else None,
            'prior': float(self.prior) if self.prior is not None else None,
            'scaler_state': standard_scaler_to_dict(self.scaler),
            'wrapper_hparams': {
                'win_size': self.win_size, 'hidden_size': self.hidden_size,
                'output_size': self.output_size, 'kernel_size': self.kernel_size,
                'n_layers': self.n_layers, 'd_model': self.d_model,
                'classifier_hidden': self.classifier_hidden,
                'num_classes': self.num_classes,
                'noisy_rate': self.noisy_rate, 'pulp_m': self.pulp_m,
                'pulp_lmbda': self.pulp_lmbda, 'knn_k': self.knn_k,
                'pulp_a': self.pulp_a, 'gamma': self.gamma,
                'encoder_epochs': self.encoder_epochs,
                'encoder_lr': self.encoder_lr,
                'encoder_bce_min': self.encoder_bce_min,
                'lr': self.lr, 'batch_size': self.batch_size,
                'seed': self.seed,
                'n_features': self.n_features,
            },
            'rng_state': capture_rng_state(),
        }
        atomic_save(ckpt, self._checkpoint_dir / 'last.pt')

    # ------------------------------------------------------------------
    # save / load (optional)
    # ------------------------------------------------------------------
    def save(self, save_dir):
        save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
        if self.encoder is not None:
            torch.save(self.encoder.state_dict(), save_dir / "encoder.pt")
        if self.classifier is not None:
            torch.save(self.classifier.state_dict(), save_dir / "classifier.pt")
        cfg = {
            'win_size': self.win_size, 'n_features': self.n_features,
            'hidden_size': self.hidden_size, 'output_size': self.output_size,
            'kernel_size': self.kernel_size, 'n_layers': self.n_layers,
            'd_model': self.d_model, 'classifier_hidden': self.classifier_hidden,
            'num_classes': self.num_classes, 'prior': self.prior,
            'noisy_rate': self.noisy_rate, 'encoder_lr': self.encoder_lr,
            'model_name': self.name,
        }
        # persist the train-fit StandardScaler (predict() prefers a fresh test-fit;
        # this is the documented degenerate-case fallback — data_loader.py:50-55)
        if self.scaler is not None:
            cfg['scaler_mean'] = self.scaler.mean_.tolist()
            cfg['scaler_scale'] = self.scaler.scale_.tolist()
        with open(save_dir / "config.json", 'w') as f:
            json.dump(cfg, f, indent=2)

    def load(self, save_dir):
        save_dir = Path(save_dir)
        with open(save_dir / "config.json") as f:
            cfg = json.load(f)
        self.win_size = cfg['win_size']; self.n_features = cfg['n_features']
        self.hidden_size = cfg['hidden_size']; self.output_size = cfg['output_size']
        self.kernel_size = cfg['kernel_size']; self.n_layers = cfg['n_layers']
        self.d_model = cfg['d_model']; self.classifier_hidden = cfg['classifier_hidden']
        self.num_classes = cfg['num_classes']; self.prior = cfg['prior']
        self.encoder_lr = cfg.get('encoder_lr', self.encoder_lr)
        if 'scaler_mean' in cfg:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            self.scaler.mean_ = np.asarray(cfg['scaler_mean'], dtype=np.float64)
            self.scaler.scale_ = np.asarray(cfg['scaler_scale'], dtype=np.float64)
            self.scaler.n_features_in_ = len(self.scaler.mean_)
        self.encoder = DilatedCNN(
            input_size=self.n_features, hidden_size=self.hidden_size,
            output_size=self.output_size, kernel_size=self.kernel_size,
            n_layers=self.n_layers, pooling_type=self.pooling_type,
            granularity=self.granularity, local_threshold=self.local_threshold,
            beta=self.beta, split_size=self.win_size,
        ).to(self.device)
        self.encoder.load_state_dict(torch.load(save_dir / "encoder.pt", map_location=self.device))
        self.classifier = Classifier_six_layer(
            self.d_model, hidden_size=self.classifier_hidden, num_classes=self.num_classes
        ).to(self.device)
        self.classifier.load_state_dict(torch.load(save_dir / "classifier.pt", map_location=self.device))
        return self

    def __repr__(self):
        return f"NRdetectorBaseline(win_size={self.win_size}, noisy_rate={self.noisy_rate}, prior={self.prior})"
