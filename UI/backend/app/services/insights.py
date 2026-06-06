"""Insight services (FB-NEW B1/B2/B3/B5) — additive, read-cheap/lazy/tolerant.

Each insight is grounded in TSMAE mechanics and cites its data source:
  * B1 — score-component decomposition from the best-epoch NPZ's OWN
    ``adaptive_score``/``teacher_recon_error``/``discrepancy_error`` arrays + a real
    consistency gate. We do NOT re-derive the score formula in UI/ (CLAUDE.md rule 3 /
    R2-08): the disc contribution is the EMPIRICAL remainder
    ``adaptive_score - teacher_recon_error`` read off the NPZ, never a re-implemented
    ``scaled_disc = disc * mean(recon)/mean(disc) / ratio``. The consistency gate
    (RV2-01) is the genuine anti-FM-omission guard: it checks the empirical remainder
    against the NPZ's OWN ``discrepancy_error`` array (``disc_raw``) — two INDEPENDENT
    arrays. The remainder is what the disc path actually adds to the score; it must
    TRACK a positive scaling of ``disc_raw`` (the un-scaled discrepancy). We report the
    Pearson correlation + the scaled relative-error of a 1-param fit
    ``remainder ≈ scale · disc_raw`` and badge ``ok=False`` when an ACTIVE remainder
    diverges from ``disc_raw`` (formula drift / wrong arrays). When the remainder is
    ~0 (best epoch PRE-warmup → adaptive ≈ teacher-recon-only, ``disc_raw`` non-zero
    but un-used), the gate reports ``disc_active=False`` — that is the headline "disc
    not carrying the detector" finding, NOT a drift failure. (This replaces the old
    tautological ``adaptive - (recon + (adaptive-recon)) ≡ 0`` self-identity, which
    could never fire.)
  * B2 — teacher↔student divergence trajectory: per-eval-epoch adaptive vs teacher
    recon + the 4 best-epoch score-variant scalars, masked pre-warmup.
  * B3 — pre↔post-warmup best delta: best metric in ep≤warmup vs ep>warmup.
  * B5 — normal-vs-anomaly score distribution: histogram from the NPZ ``point_labels``
    split (reuses ``npz_reader.histogram``).

Safety: NPZ access is lazy + ``CorruptOrWriting``-tolerant + downsampled; never opens
``*.pt``; read-only over results. Every function returns ``available: False`` rather
than raising when data is absent (graceful degradation).
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Optional

from ..dataaccess.io_safe import CorruptOrWriting
from ..dataaccess.npz_reader import histogram as npz_histogram
from ..dataaccess.npz_reader import list_arrays as npz_list_arrays
from ..dataaccess.npz_reader import read_arrays
from ..dataaccess.repository import LeafBundle, Repository

try:  # numpy is in UI/.venv (torch is NOT); sklearn/scipy are intentionally ABSENT.
    import numpy as np  # type: ignore

    _HAVE_NUMPY = True
except Exception:  # pragma: no cover - partial venv only
    np = None  # type: ignore
    _HAVE_NUMPY = False

# B1 consistency gate (RV2-01 / R2-08, the anti-FM-omission guard).
#   * ``_DISC_ACTIVE_EPS`` — below this the empirical disc remainder
#     (``adaptive - teacher_recon``) is ~constant ⇒ the disc path is NOT contributing
#     to the score (best epoch pre-warmup, recon-only). That is the headline finding,
#     not a drift failure → ``disc_active=False`` and the gate stays ``ok=True``.
#   * ``_DISC_CORR_MIN`` / ``_DISC_RELERR_MAX`` — when the remainder IS active, it must
#     TRACK a positive scaling of the NPZ's OWN ``discrepancy_error`` array (two
#     independent arrays). If correlation drops below the floor OR the scaled
#     relative-error of the 1-param fit ``remainder ≈ scale·disc_raw`` exceeds the cap,
#     the stored components are inconsistent (formula drift / wrong arrays) → ok=False.
_DISC_ACTIVE_EPS = 1e-9
_DISC_CORR_MIN = 0.98
_DISC_RELERR_MAX = 5e-2
# B1 NPZ downsample target — bounds memory/wire; one best-epoch NPZ, one pass.
_DECOMP_DOWNSAMPLE = 6000


def _std(xs: list[float]) -> float:
    m = _mean(xs)
    if m is None or len(xs) < 2:
        return 0.0
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _disc_consistency(remainder: list, disc_raw: list) -> dict[str, Any]:
    """RV2-01 / R2-08 — INDEPENDENT consistency check: does the empirical disc
    remainder (``adaptive - teacher_recon``, what the disc path adds to the score)
    track the NPZ's OWN ``discrepancy_error`` array (the un-scaled discrepancy)?

    Two INDEPENDENT arrays (the remainder is NOT derived from ``disc_raw``), so this is
    a genuine anti-FM-omission guard — it WOULD fire on a deliberate mismatch (wrong
    array / formula drift), unlike the old self-identity ``adaptive-(recon+remainder)``.
    Returns the Pearson correlation, the least-squares positive scale of
    ``remainder ≈ scale·disc_raw``, the scaled relative error, and the ``disc_active``/
    ``ok`` flags. A ~constant remainder (pre-warmup, disc not carrying) ⇒
    ``disc_active=False``, ``ok=True`` (expected, the headline finding)."""
    if not disc_raw:
        return {"ok": True, "disc_active": False, "correlation": None, "scale": None,
                "scaled_rel_error": None,
                "note": "no discrepancy_error array in NPZ to cross-check against"}
    n = min(len(remainder), len(disc_raw))
    rem = [remainder[i] for i in range(n)
           if isinstance(remainder[i], (int, float)) and math.isfinite(remainder[i])
           and isinstance(disc_raw[i], (int, float)) and math.isfinite(disc_raw[i])]
    dr = [disc_raw[i] for i in range(n)
          if isinstance(remainder[i], (int, float)) and math.isfinite(remainder[i])
          and isinstance(disc_raw[i], (int, float)) and math.isfinite(disc_raw[i])]
    if len(rem) < 2:
        return {"ok": True, "disc_active": False, "correlation": None, "scale": None,
                "scaled_rel_error": None, "note": "too few finite paired points"}
    rem_std = _std(rem)
    # disc NOT carrying the score: the remainder is ~constant (≈0) even though disc_raw
    # is non-zero. Expected pre-warmup — the headline finding, NOT a drift failure.
    if rem_std <= _DISC_ACTIVE_EPS:
        return {
            "ok": True, "disc_active": False, "correlation": None, "scale": 0.0,
            "scaled_rel_error": None,
            "note": ("disc path NOT carrying the score (remainder ≈ constant) while the "
                     "raw discrepancy_error is non-zero — disc essentially unused at "
                     "this epoch"),
        }
    rem_m, dr_m = _mean(rem), _mean(dr)
    cov = sum((rem[i] - rem_m) * (dr[i] - dr_m) for i in range(len(rem))) / len(rem)
    dr_std = _std(dr)
    correlation = (cov / (rem_std * dr_std)) if (rem_std > 0 and dr_std > 0) else None
    # 1-param least-squares scale of remainder ≈ scale·disc_raw (no intercept — the
    # disc CONTRIBUTION to the score is a positive scaling of the un-scaled discrepancy).
    denom = sum(d * d for d in dr)
    scale = (sum(rem[i] * dr[i] for i in range(len(rem))) / denom) if denom > 0 else None
    rel_err = None
    if scale is not None:
        mean_abs_rem = _mean([abs(x) for x in rem]) or 0.0
        if mean_abs_rem > 0:
            resid = _mean([abs(rem[i] - scale * dr[i]) for i in range(len(rem))]) or 0.0
            rel_err = resid / mean_abs_rem
    ok = (
        correlation is not None and correlation >= _DISC_CORR_MIN
        and (scale is not None and scale > 0)
        and (rel_err is not None and rel_err <= _DISC_RELERR_MAX)
    )
    return {
        "ok": bool(ok),
        "disc_active": True,
        "correlation": correlation,
        "scale": scale,
        "scaled_rel_error": rel_err,
        "corr_floor": _DISC_CORR_MIN,
        "rel_error_cap": _DISC_RELERR_MAX,
        "note": ("empirical disc remainder TRACKS the NPZ's own discrepancy_error "
                 "(consistent)" if ok else
                 "FORMULA DRIFT: the disc contribution to the score does NOT track the "
                 "raw discrepancy_error array (wrong arrays / scoring.py changed)"),
    }


def _best_epoch(bundle: LeafBundle) -> Optional[int]:
    if bundle.metadata is None:
        return None
    be = bundle.metadata.timing.get("best_epoch")
    return int(be) if isinstance(be, (int, float)) else None


def _npz_path(leaf_path: Path, epoch: int) -> Path:
    return leaf_path / "epoch_scores" / f"epoch_{epoch:03d}_scores.npz"


def _nearest_score_epoch(leaf_path: Path, target: Optional[int]) -> Optional[int]:
    """The available NPZ epoch nearest ``target`` (NPZ grid may not contain best_epoch
    exactly). Returns None if no epoch_scores NPZ exists."""
    d = leaf_path / "epoch_scores"
    if not d.is_dir():
        return None
    eps: list[int] = []
    for f in sorted(d.glob("epoch_*_scores.npz")):
        try:
            eps.append(int(f.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    if not eps:
        return None
    if target is None:
        return eps[-1]
    return min(eps, key=lambda e: abs(e - target))


def _mean(xs: list[float]) -> Optional[float]:
    vals = [x for x in xs if isinstance(x, (int, float)) and math.isfinite(x)]
    return (sum(vals) / len(vals)) if vals else None


# ── B1 — score-component decomposition (best-epoch NPZ) ───────────────────────
def score_decomposition(repo: Repository, bundle: LeafBundle) -> dict[str, Any]:
    """Decompose the FINAL per-timestep adaptive score into its recon vs disc
    contribution from the best-epoch NPZ's OWN arrays (B1).

    ``recon_component = teacher_recon_error`` (the NPZ array); ``disc_component =
    adaptive_score - teacher_recon_error`` (the EMPIRICAL remainder — what the
    discrepancy path actually adds to the score, NOT a re-derived scaled_disc). The
    consistency gate (RV2-01 / R2-08) cross-checks the remainder against the NPZ's OWN
    ``discrepancy_error`` array (``disc_raw``) — two INDEPENDENT arrays: an active
    remainder must TRACK a positive scaling of ``disc_raw`` (correlation + scaled
    relative-error), and ``ok=False`` flags formula drift / wrong arrays. A ~constant
    remainder ⇒ ``disc_active=False`` (disc not carrying — the headline finding, not a
    failure). We also report the mean contribution of each component to the
    normal-class vs anomaly-class score and a "% of separation from disc" headline."""
    leaf_path = bundle.leaf.leaf_path
    best = _best_epoch(bundle)
    warmup = bundle.leaf.warmup_epochs
    epoch = _nearest_score_epoch(leaf_path, best)
    if epoch is None:
        return {"available": False, "reason": "no epoch_scores NPZ"}
    npz = _npz_path(leaf_path, epoch)
    if not npz.exists():
        return {"available": False, "reason": "best-epoch NPZ absent",
                "epoch": epoch}
    try:
        res = read_arrays(
            npz,
            ["adaptive_score", "teacher_recon_error", "discrepancy_error",
             "point_labels"],
            downsample=_DECOMP_DOWNSAMPLE,
        )
    except CorruptOrWriting:
        return {"available": False, "reason": "corrupt_or_writing", "epoch": epoch}

    a = res["arrays"]
    adaptive = a.get("adaptive_score")
    recon = a.get("teacher_recon_error")
    disc_raw = a.get("discrepancy_error")
    labels = a.get("point_labels")
    if not adaptive or not recon:
        return {"available": False, "reason": "missing adaptive/recon arrays",
                "epoch": epoch, "available_arrays": res.get("available_arrays")}

    n = min(len(adaptive), len(recon))
    disc_remainder = [
        (adaptive[i] - recon[i])
        if isinstance(adaptive[i], (int, float)) and isinstance(recon[i], (int, float))
        else None
        for i in range(n)
    ]

    # Consistency gate (RV2-01 / R2-08, anti-FM-omission): cross-check the EMPIRICAL
    # disc remainder against the NPZ's OWN ``discrepancy_error`` array (disc_raw) — two
    # INDEPENDENT arrays. An active remainder must TRACK a positive scaling of disc_raw;
    # a ~constant remainder ⇒ disc not carrying (the headline finding). This WOULD fire
    # on a deliberate mismatch, unlike the old vacuous self-identity gate.
    disc_gate = _disc_consistency(disc_remainder, disc_raw or [])

    # mean component contributions, split by point_labels (normal=0 vs anomaly=1).
    def _split(arr, want_label):
        if not labels:
            return [x for x in arr if isinstance(x, (int, float))]
        return [arr[i] for i in range(min(len(arr), len(labels)))
                if labels[i] == want_label and isinstance(arr[i], (int, float))]

    recon_norm = _mean(_split(recon, 0))
    recon_anom = _mean(_split(recon, 1))
    disc_norm = _mean(_split(disc_remainder, 0))
    disc_anom = _mean(_split(disc_remainder, 1))

    # "% of separation from disc": how much of the normal→anomaly score gap each
    # component supplies (the central recon-vs-disc question for this self-distilled MAE).
    recon_sep = ((recon_anom - recon_norm)
                 if recon_norm is not None and recon_anom is not None else None)
    disc_sep = ((disc_anom - disc_norm)
                if disc_norm is not None and disc_anom is not None else None)
    pct_from_disc = None
    if recon_sep is not None and disc_sep is not None:
        total = abs(recon_sep) + abs(disc_sep)
        pct_from_disc = (abs(disc_sep) / total) if total > 0 else None

    pre_warmup_best = (best is not None and warmup is not None and best <= warmup)
    return {
        "available": True,
        "epoch": epoch,
        "best_epoch": best,
        "warmup_epochs": warmup,
        "best_epoch_pre_warmup": pre_warmup_best,
        "n_points": n,
        "downsampled": res.get("downsampled"),
        "data_source": f"epoch_scores/epoch_{epoch:03d}_scores.npz "
                       "(adaptive_score, teacher_recon_error, discrepancy_error, "
                       "point_labels)",
        # RV2-01: the gate now exercises the INDEPENDENT NPZ ``discrepancy_error`` array
        # (disc_raw), not a tautology. ``residual_gate`` is kept as the consumer-facing
        # key (additive fields) so existing readers pick up the real check.
        "residual_gate": {
            "ok": disc_gate["ok"],
            "disc_active": disc_gate["disc_active"],
            "correlation": disc_gate["correlation"],
            "scale": disc_gate["scale"],
            "scaled_rel_error": disc_gate["scaled_rel_error"],
            "corr_floor": disc_gate.get("corr_floor", _DISC_CORR_MIN),
            "rel_error_cap": disc_gate.get("rel_error_cap", _DISC_RELERR_MAX),
            "checks": "empirical (adaptive - teacher_recon) vs NPZ discrepancy_error",
            "note": disc_gate["note"],
        },
        "disc_consistency_gate": disc_gate,
        "components": {
            "recon": {"mean_normal": recon_norm, "mean_anomaly": recon_anom,
                      "separation": recon_sep,
                      "source": "teacher_recon_error (NPZ)"},
            "disc": {"mean_normal": disc_norm, "mean_anomaly": disc_anom,
                     "separation": disc_sep,
                     "source": "adaptive_score - teacher_recon_error (empirical "
                               "remainder; NOT a re-derived scaled_disc)"},
        },
        "pct_separation_from_disc": pct_from_disc,
        "raw_discrepancy_present": disc_raw is not None,
        "headline": (
            "best epoch is PRE-warmup → adaptive ≈ teacher-recon-only; the discrepancy "
            "term barely contributes" if pre_warmup_best else
            ("disc carries a large share of the separation"
             if (pct_from_disc is not None and pct_from_disc > 0.4)
             else "recon carries most of the separation")
        ),
    }


# ── B2 — teacher↔student divergence trajectory ────────────────────────────────
def divergence_trajectory(repo: Repository, bundle: LeafBundle, metric: str = "pak_auc_f1"
                          ) -> dict[str, Any]:
    """Per-eval-epoch adaptive vs teacher recon (for ``metric``) + the 4 best-epoch
    score-variant scalars (B2). The adaptive−teacher gap over epochs = the discrepancy
    contribution; the masked pre-warmup span is flagged. P3-01: only adaptive(metrics)
    + teacher_* are genuine per-epoch SERIES; student/disc are best-epoch SCALARS."""
    epoch = bundle.epoch
    md = bundle.metadata
    if epoch is None:
        return {"available": False, "reason": "no epoch_metrics"}
    adaptive_key = metric
    teacher_key = f"teacher_{metric}"
    out_series: dict[str, Any] = {}
    for label, k in (("adaptive", adaptive_key), ("teacher", teacher_key)):
        if k in epoch.series:
            m = repo.registry.resolve(k, namespace="epoch_metrics").model_dump()
            out_series[label] = {"key": k, "values": epoch.series[k],
                                 "display_name": m["display_name"]}
    # adaptive − teacher discrepancy-contribution band (only where BOTH present).
    contrib = None
    if "adaptive" in out_series and "teacher" in out_series:
        av = out_series["adaptive"]["values"]
        tv = out_series["teacher"]["values"]
        contrib = [
            (av[i] - tv[i])
            if i < len(av) and i < len(tv)
            and isinstance(av[i], (int, float)) and isinstance(tv[i], (int, float))
            else None
            for i in range(max(len(av), len(tv)))
        ]
    # the 4 best-epoch score-variant scalars (P3-01 markers).
    markers: dict[str, Optional[float]] = {}
    if md is not None:
        for block in ("metrics", "teacher_recon_metrics", "student_recon_metrics",
                      "disc_metrics"):
            markers[block] = (md.score_variants.get(block, {}) or {}).get(metric)
    return {
        "available": bool(out_series),
        "metric": metric,
        "epochs": epoch.epochs,
        "warmup_epochs": epoch.warmup_epochs,
        "series": out_series,
        "discrepancy_contribution": contrib,
        "best_epoch_scalars": markers,
        "data_source": "epoch_metrics.json (adaptive/teacher_* series) + "
                       "experiment_metadata score-variant blocks (best-epoch scalars)",
    }


# ── B3 — pre↔post-warmup best delta ───────────────────────────────────────────
def warmup_delta(repo: Repository, bundle: LeafBundle, metric: str = "pak_auc_f1"
                 ) -> dict[str, Any]:
    """Best metric in ep≤warmup (pre, recon-only-gated) vs ep>warmup (post, student
    joined); ``delta = best_post − best_pre`` answers "did letting the student in
    help?" (B3). Direction-aware (argmax ↑ / argmin ↓). On 271, best_epoch=100 <
    warmup=250 → pre-warmup won → delta ≤ 0 (a notable negative result)."""
    epoch = bundle.epoch
    if epoch is None or metric not in epoch.series:
        return {"available": False, "reason": "metric not a per-epoch series"}
    warmup = epoch.warmup_epochs
    meta = repo.registry.resolve(metric, namespace="epoch_metrics").model_dump()
    direction = meta.get("direction", "neutral")
    if direction not in ("up", "down"):
        return {"available": False, "reason": "neutral metric has no 'best'",
                "metric": metric, "direction": direction}
    epochs = epoch.epochs
    values = epoch.series[metric]

    def _best(pairs):
        finite = [(e, v) for e, v in pairs
                  if isinstance(v, (int, float)) and math.isfinite(v)]
        if not finite:
            return None, None
        return (min(finite, key=lambda p: p[1]) if direction == "down"
                else max(finite, key=lambda p: p[1]))

    if warmup is None:
        return {"available": False, "reason": "no warmup boundary on this leaf",
                "metric": metric}
    pre = [(e, v) for e, v in zip(epochs, values) if e <= warmup]
    post = [(e, v) for e, v in zip(epochs, values) if e > warmup]
    pre_ep, pre_best = _best(pre)
    post_ep, post_best = _best(post)
    delta = (post_best - pre_best
             if pre_best is not None and post_best is not None else None)
    if delta is None:
        helped = None
    elif direction == "down":
        helped = delta < 0  # lower is better post-warmup ⇒ student helped
    else:
        helped = delta > 0
    return {
        "available": True,
        "metric": metric,
        "direction": direction,
        "warmup_epochs": warmup,
        "pre_warmup": {"best_epoch": pre_ep, "best_value": pre_best},
        "post_warmup": {"best_epoch": post_ep, "best_value": post_best},
        "delta": delta,
        "student_helped": helped,
        "caption": (None if delta is None else
                    f"student {'helped' if helped else 'hurt'} {delta:+.4f}"),
        "data_source": "epoch_metrics.json split at config.teacher_only_warmup_epochs",
    }


# ── B5 — normal-vs-anomaly score distribution (NPZ) ───────────────────────────
def score_distribution(repo: Repository, bundle: LeafBundle, *,
                       epoch: Optional[int] = None, array: str = "adaptive_score",
                       bins: int = 80) -> dict[str, Any]:
    """Two histograms of the per-timestep score split by ``point_labels`` (normal vs
    anomaly) from the best-epoch (or requested) NPZ (B5). Reuses
    ``npz_reader.histogram`` (lazy, BadZipFile-tolerant). The visual overlap IS the
    detector's real difficulty."""
    leaf_path = bundle.leaf.leaf_path
    target = epoch if epoch is not None else _best_epoch(bundle)
    ep = _nearest_score_epoch(leaf_path, target)
    if ep is None:
        return {"available": False, "reason": "no epoch_scores NPZ"}
    npz = _npz_path(leaf_path, ep)
    if not npz.exists():
        return {"available": False, "reason": "NPZ absent", "epoch": ep}
    try:
        res = npz_histogram(npz, array, bins=bins)
    except CorruptOrWriting:
        return {"available": False, "reason": "corrupt_or_writing", "epoch": ep}
    res["epoch"] = ep
    res["best_epoch"] = _best_epoch(bundle)
    res["array"] = array
    res["data_source"] = (f"epoch_scores/epoch_{ep:03d}_scores.npz "
                          f"({array} split by point_labels)")
    return res


# ══════════════════════════════════════════════════════════════════════════════
# E-3 (P0-5 / I1+I2) — per-epoch COMPONENT DETECTION TRAJECTORIES + efficacy
#                       + E-4 (I3) — score-vs-time COMPONENT OVERLAY
#
# THE flagship self-distillation-efficacy insight. For each eval epoch that has an
# NPZ, we RECOMPUTE detection (ROC-AUC, PR-AUC/AP, and a torch-free PA%K-best-F1)
# from the RAW saved arrays vs ``point_labels`` for three score channels:
#   * adaptive_score      — the production score. SELF-VALIDATED: at the best epoch
#                           its recompute must ≈ the STORED metrics/{roc_auc,prc_auc}
#                           (asserted + reported; small float32-NPZ quantization gap
#                           tolerated). This proves the recompute pipeline is faithful.
#   * teacher_recon_error — recon-only ABLATION (teacher path alone).
#   * discrepancy_error   — disc-only ABLATION (the discrepancy path alone).
#
# P3-01 honesty: teacher-only / disc-only are RECOMPUTED ABLATIONS off the raw saved
# arrays — a legitimate channel ablation, NOT a fabricated per-epoch student/disc
# metric series. Where no NPZ exists for an epoch we emit a GAP (None), never a
# synthesized value. We DO NOT re-derive the score formula (CLAUDE.md rule 3) and DO
# NOT import mae_anomaly/* (needs torch) — every metric is numpy-only here.
#
# R-01 (binding): numpy-only — NO sklearn/scipy. R-06 (binding): caches key on
# ``_INSIGHT_VERSION`` (distinct from the GIF ``_RENDER_VERSION``) so GIF-only bumps
# never invalidate insight caches.
# ══════════════════════════════════════════════════════════════════════════════

# R-06: insight-compute cache version — DISTINCT from gif/service.py::_RENDER_VERSION.
# Bump when the trajectory math / payload shape changes (NOT on GIF render changes).
_INSIGHT_VERSION = 1

# Self-validation tolerance: the recomputed adaptive ROC/AP at the best epoch must
# match the STORED metrics/{roc_auc,prc_auc}. The stored scalars were computed at
# training time on the full-precision in-memory score; the NPZ persists float32, so a
# few×1e-3 quantization gap is expected and OK. We assert within this cap AND report
# the exact recomputed vs stored numbers + the delta so the check is transparent.
_SELFVAL_TOL = 5e-3

# Per-epoch ROC/AP recompute downsample cap. ROC/AP are RANK statistics — a uniform
# stride preserves them to <1e-3 while bounding the per-epoch sort cost across a long
# trajectory. Self-validation (which must match stored tightly) uses FULL resolution.
_TRAJ_DOWNSAMPLE = 20000

# In-process trajectory cache (a trajectory spans MANY NPZ files, so MtimeCache's
# single-file signature doesn't fit). Keyed on (leaf_path, _INSIGHT_VERSION,
# n_npz, max_mtime_ns) — a live append bumps n_npz/mtime ⇒ auto-invalidate. Mirrors
# the gif service's multi-source key; NEVER keyed on _RENDER_VERSION (R-06).
_TRAJ_CACHE: dict[tuple, dict[str, Any]] = {}


def _np_roc_auc(scores, labels) -> Optional[float]:
    """numpy-only ROC-AUC via the rank statistic (Mann–Whitney U). Tie-correct
    (average ranks). NO sklearn. Returns None if a class is empty / degenerate."""
    if not _HAVE_NUMPY:
        return None
    s = np.asarray(scores, dtype="float64")
    y = np.asarray(labels)
    m = np.isfinite(s)
    s = s[m]
    y = (y[m] == 1)
    P = int(y.sum())
    N = int((~y).sum())
    if P == 0 or N == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    # average (tie-corrected) ranks, 1-based
    ranks_sorted = np.empty(s_sorted.shape[0], dtype="float64")
    n = s_sorted.shape[0]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks_sorted[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    ranks = np.empty(n, dtype="float64")
    ranks[order] = ranks_sorted
    sum_pos = float(ranks[y].sum())
    return (sum_pos - P * (P + 1) / 2.0) / (P * N)


def _np_average_precision(scores, labels) -> Optional[float]:
    """numpy-only Average-Precision (PR-AUC, sklearn ``average_precision_score``
    convention: AP = Σ (Rₙ − Rₙ₋₁)·Pₙ). NO sklearn. None if no positives."""
    if not _HAVE_NUMPY:
        return None
    s = np.asarray(scores, dtype="float64")
    y = np.asarray(labels)
    m = np.isfinite(s)
    s = s[m]
    y = (y[m] == 1).astype("int64")
    P = int(y.sum())
    if P == 0:
        return None
    order = np.argsort(-s, kind="mergesort")  # descending score
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / P
    recall_prev = np.concatenate(([0.0], recall[:-1]))
    return float(np.sum((recall - recall_prev) * precision))


def _anomaly_segments(y):
    """Contiguous (start, end_exclusive) spans where ``y==1``. Vectorized (one diff).
    Returns parallel ``starts``/``ends`` int arrays + segment lengths."""
    yb = (y == 1).astype("int8")
    if yb.size == 0:
        return np.empty(0, "int64"), np.empty(0, "int64")
    pad = np.concatenate(([0], yb, [0]))
    d = np.diff(pad)
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return starts.astype("int64"), ends.astype("int64")


def _np_pak_best_f1(scores, labels, *, n_thresholds: int = 80) -> Optional[float]:
    """torch-free numpy PA%K(=0) best-F1: sweep thresholds over the score quantiles,
    point-adjust each prediction (any true-anomaly SEGMENT with ≥1 predicted point
    inside is counted fully detected — the standard TS-AD PA adjust), take the max F1.
    NO sklearn/scipy.

    VECTORIZED: the anomaly segments are computed ONCE; per threshold we use a prefix
    sum + ``np.add.reduceat`` to test, in O(n + #segments), whether each segment has any
    predicted point — no Python loop over points or segments. Bounded threshold sweep
    (``n_thresholds`` quantiles). Returns None on a degenerate split."""
    if not _HAVE_NUMPY:
        return None
    s = np.asarray(scores, dtype="float64")
    y = np.asarray(labels)
    m = np.isfinite(s)
    s = s[m]
    y = (y[m] == 1).astype("int8")
    n = y.shape[0]
    P = int(y.sum())
    if P == 0 or P == n:
        return None
    starts, ends = _anomaly_segments(y)
    seg_len = ends - starts
    thr = np.unique(np.quantile(s, np.linspace(0.0, 1.0, n_thresholds)))
    y_pos = (y == 1)
    y_neg = ~y_pos
    best = 0.0
    for t in thr:
        pred = (s >= t)
        # how many predicted points fall inside each anomaly segment (prefix-sum).
        if starts.size:
            csum = np.concatenate(([0], np.cumsum(pred.astype("int64"))))
            inside = csum[ends] - csum[starts]          # per-segment predicted count
            detected = inside > 0                         # segment fully detected ⇒ +len
            tp = int(seg_len[detected].sum())            # PA: detected segments → all TP
            fn = int(seg_len[~detected].sum())           # missed segments → all FN
        else:
            tp = fn = 0
        # FP are predicted NORMAL points (PA never moves a normal point).
        fp = int((pred & y_neg).sum())
        denom = 2 * tp + fp + fn
        if denom > 0:
            f1 = 2.0 * tp / denom
            if f1 > best:
                best = f1
    return float(best)


def _epoch_npz_grid(leaf_path: Path) -> list[tuple[int, Path]]:
    """All (epoch, npz_path) under epoch_scores/, ascending. Cheap dir scan."""
    d = leaf_path / "epoch_scores"
    if not d.is_dir():
        return []
    out: list[tuple[int, Path]] = []
    for f in sorted(d.glob("epoch_*_scores.npz")):
        mo = re.search(r"epoch_(\d+)_scores", f.name)
        if mo:
            out.append((int(mo.group(1)), f))
    out.sort(key=lambda t: t[0])
    return out


def _read_channels_np(npz_path: Path, names: list[str], *, downsample: int = 0):
    """Lazily read selected arrays as numpy float64/int (one NPZ, mmap, tolerant).
    Returns a dict name->ndarray (only present arrays). Raises CorruptOrWriting on a
    half-written/corrupt file (caller emits a GAP for that epoch). numpy-only."""
    if not _HAVE_NUMPY:
        raise RuntimeError("numpy unavailable in this venv")
    import zipfile
    present = set(npz_list_arrays(npz_path))  # raises CorruptOrWriting
    want = [a for a in names if a in present]
    out: dict[str, Any] = {}
    try:
        with np.load(npz_path, mmap_mode="r", allow_pickle=False) as z:
            length = None
            idx = None
            for nm in want:
                arr = z[nm]
                length = int(arr.shape[0]) if arr.ndim >= 1 else 1
                if downsample and length > downsample:
                    if idx is None:
                        stride = max(1, length // downsample)
                        idx = np.arange(0, length, stride)
                    out[nm] = np.asarray(arr[idx])
                else:
                    out[nm] = np.asarray(arr)
    except (zipfile.BadZipFile, EOFError, OSError) as exc:
        raise CorruptOrWriting(f"NPZ corrupt/writing: {npz_path}: {exc}") from exc
    return out


_CHANNELS = {
    "adaptive_score": "adaptive",
    "teacher_recon_error": "teacher_only",
    "discrepancy_error": "disc_only",
}


def _detection_for_epoch(npz_path: Path, *, full_res: bool, want_pak: bool
                         ) -> Optional[dict[str, Any]]:
    """Recompute ROC/AP(+PA%K) for the 3 channels from ONE NPZ. None ⇒ GAP (no
    usable arrays / corrupt). numpy-only; never opens *.pt."""
    try:
        arrs = _read_channels_np(
            npz_path,
            ["point_labels", *list(_CHANNELS.keys())],
            downsample=0 if full_res else _TRAJ_DOWNSAMPLE,
        )
    except CorruptOrWriting:
        return None
    labels = arrs.get("point_labels")
    if labels is None:
        return None
    labels = np.asarray(labels)
    if (labels == 1).sum() == 0 or (labels == 0).sum() == 0:
        return None
    row: dict[str, Any] = {}
    for npz_name, chan in _CHANNELS.items():
        s = arrs.get(npz_name)
        if s is None:
            row[chan] = {"roc": None, "ap": None, "pak_f1": None}
            continue
        row[chan] = {
            "roc": _np_roc_auc(s, labels),
            "ap": _np_average_precision(s, labels),
            "pak_f1": (_np_pak_best_f1(s, labels) if want_pak else None),
        }
    return row


def _traj_cache_key(leaf_path: Path) -> Optional[tuple]:
    grid = _epoch_npz_grid(leaf_path)
    if not grid:
        return None
    max_mtime = 0
    for _ep, p in grid:
        try:
            max_mtime = max(max_mtime, p.stat().st_mtime_ns)
        except OSError:
            continue
    return (str(leaf_path), _INSIGHT_VERSION, len(grid), max_mtime)


# ── E-3 — component detection trajectories + I2 efficacy verdict ──────────────
def component_trajectories(repo: Repository, bundle: LeafBundle, *,
                           metric: str = "roc", want_pak: bool = True
                           ) -> dict[str, Any]:
    """E-3 (P0-5 / I1+I2). Per-eval-epoch RECOMPUTED detection (ROC-AUC, PR-AUC/AP, and
    a torch-free PA%K best-F1) for three score channels — adaptive (production),
    teacher-only (recon ablation), disc-only (discrepancy ablation) — vs point_labels,
    read from the saved NPZs (numpy-only, lazy, BadZipFile-tolerant).

    Returns the 3 trajectories (gaps where no usable NPZ — P3-01: a real raw-array
    ablation, never a fabricated adaptive series), the warmup boundary, a SELF-
    VALIDATION of the adaptive recompute against the STORED metrics/{roc_auc,prc_auc},
    and the I2 EFFICACY VERDICT: does disc-only detection rise meaningfully AFTER warmup?

    numpy-only (R-01: NO sklearn/scipy); cached on ``_INSIGHT_VERSION`` (R-06)."""
    if not _HAVE_NUMPY:
        return {"available": False, "reason": "numpy unavailable"}
    leaf_path = bundle.leaf.leaf_path
    ck = _traj_cache_key(leaf_path)
    if ck is None:
        return {"available": False, "reason": "no epoch_scores NPZ"}
    cached = _TRAJ_CACHE.get(ck)
    if cached is not None:
        return cached

    grid = _epoch_npz_grid(leaf_path)
    best = _best_epoch(bundle)
    warmup = bundle.leaf.warmup_epochs

    epochs: list[int] = []
    series: dict[str, dict[str, list]] = {
        "adaptive": {"roc": [], "ap": [], "pak_f1": []},
        "teacher_only": {"roc": [], "ap": [], "pak_f1": []},
        "disc_only": {"roc": [], "ap": [], "pak_f1": []},
    }
    n_npz_used = 0
    n_gap = 0
    for ep, p in grid:
        epochs.append(ep)
        row = _detection_for_epoch(p, full_res=False, want_pak=want_pak)
        if row is None:
            n_gap += 1
            for chan in series:
                for k in ("roc", "ap", "pak_f1"):
                    series[chan][k].append(None)
            continue
        n_npz_used += 1
        for chan in series:
            for k in ("roc", "ap", "pak_f1"):
                series[chan][k].append(row[chan][k])

    # ── SELF-VALIDATION: recompute adaptive at the best epoch (FULL resolution) and
    # compare to the STORED metrics/{roc_auc,prc_auc}. Assert within tolerance + report.
    selfval = _self_validate_adaptive(bundle, grid, best)

    # ── I2 EFFICACY VERDICT: does disc-only detection rise meaningfully post-warmup?
    efficacy = _disc_efficacy_verdict(epochs, series["disc_only"][metric], warmup,
                                      metric=metric)

    result: dict[str, Any] = {
        "available": n_npz_used > 0,
        "label": "RECOMPUTED ABLATION (numpy ROC/AP/PA%K from saved NPZ arrays)",
        "insight_version": _INSIGHT_VERSION,
        "metric_for_verdict": metric,
        "epochs": epochs,
        "warmup_epochs": warmup,
        "best_epoch": best,
        "n_npz": len(grid),
        "n_npz_used": n_npz_used,
        "n_gap": n_gap,
        "channels": {
            "adaptive": {
                "label": "adaptive_score (production score)",
                "is_ablation": False,
                "trajectory": series["adaptive"],
            },
            "teacher_only": {
                "label": "teacher_recon_error ONLY (recon-path ablation, recomputed)",
                "is_ablation": True,
                "trajectory": series["teacher_only"],
            },
            "disc_only": {
                "label": "discrepancy_error ONLY (disc-path ablation, recomputed)",
                "is_ablation": True,
                "trajectory": series["disc_only"],
            },
        },
        "self_validation": selfval,
        "efficacy_verdict": efficacy,
        "metrics_available": ["roc", "ap"] + (["pak_f1"] if want_pak else []),
        "data_source": ("epoch_scores/epoch_NNN_scores.npz per eval epoch — RAW "
                        "adaptive_score / teacher_recon_error / discrepancy_error vs "
                        "point_labels; numpy-only ROC-AUC + Average-Precision"
                        + (" + PA%K(=0) best-F1" if want_pak else "")),
        "p3_01_note": ("teacher-only / disc-only are RECOMPUTED channel ablations from "
                       "the saved raw arrays (gaps where no NPZ); NOT fabricated "
                       "per-epoch student/disc metric trajectories."),
    }
    _TRAJ_CACHE[ck] = result
    return result


def _self_validate_adaptive(bundle: LeafBundle, grid: list[tuple[int, Path]],
                            best: Optional[int]) -> dict[str, Any]:
    """Recompute adaptive ROC/AP at the (nearest available) best epoch FULL-resolution
    and compare to the STORED metrics/{roc_auc,prc_auc}. Asserts within _SELFVAL_TOL
    and reports the exact recomputed vs stored values + the delta (transparent)."""
    if not grid:
        return {"ok": None, "reason": "no NPZ"}
    eps = [e for e, _ in grid]
    target = best if best is not None else eps[-1]
    ep = min(eps, key=lambda e: abs(e - target))
    npz = dict(grid)[ep]
    try:
        arrs = _read_channels_np(npz, ["point_labels", "adaptive_score"], downsample=0)
    except CorruptOrWriting:
        return {"ok": None, "reason": "best-epoch NPZ corrupt", "epoch": ep}
    labels = arrs.get("point_labels")
    adaptive = arrs.get("adaptive_score")
    if labels is None or adaptive is None:
        return {"ok": None, "reason": "missing arrays", "epoch": ep}
    roc_rc = _np_roc_auc(adaptive, labels)
    ap_rc = _np_average_precision(adaptive, labels)
    stored = (bundle.metadata.score_variants.get("metrics", {}) or {}
              ) if bundle.metadata is not None else {}
    roc_st = stored.get("roc_auc")
    ap_st = stored.get("prc_auc")
    roc_d = (abs(roc_rc - roc_st)
             if roc_rc is not None and isinstance(roc_st, (int, float)) else None)
    ap_d = (abs(ap_rc - ap_st)
            if ap_rc is not None and isinstance(ap_st, (int, float)) else None)
    ok = ((roc_d is not None and roc_d <= _SELFVAL_TOL)
          and (ap_d is not None and ap_d <= _SELFVAL_TOL))
    return {
        "ok": bool(ok) if (roc_d is not None and ap_d is not None) else None,
        "epoch": ep,
        "best_epoch": best,
        "tolerance": _SELFVAL_TOL,
        "recomputed": {"roc": roc_rc, "ap": ap_rc},
        "stored": {"roc": roc_st, "ap": ap_st},
        "abs_delta": {"roc": roc_d, "ap": ap_d},
        "note": (
            "numpy recompute of adaptive_score MATCHES the stored metrics/"
            "{roc_auc,prc_auc} within tolerance (recompute pipeline validated; the "
            "small gap is float32-NPZ quantization vs the full-precision stored scalar)"
            if ok else
            "recompute vs stored adaptive metrics differ by more than the float32 "
            "tolerance — inspect (stored metric may use a different alignment)"
            if (roc_d is not None and ap_d is not None) else
            "no stored adaptive metric to validate against"),
    }


def _disc_efficacy_verdict(epochs: list[int], disc_series: list, warmup: Optional[int],
                           *, metric: str) -> dict[str, Any]:
    """I2 EFFICACY: does disc-only detection rise MEANINGFULLY after the warmup
    boundary? Compares the best pre-warmup disc-only detection to the best post-warmup,
    finds the onset epoch where post-warmup first exceeds the pre-warmup ceiling, and
    returns a delta + verdict text. ``warmup`` None ⇒ no boundary → unverdictable."""
    pairs = [(e, v) for e, v in zip(epochs, disc_series)
             if isinstance(v, (int, float)) and math.isfinite(v)]
    if warmup is None:
        return {"verdict": "no_warmup_boundary",
                "text": "leaf has no teacher_only_warmup_epochs — cannot split pre/post",
                "delta": None, "onset_epoch": None}
    pre = [(e, v) for e, v in pairs if e <= warmup]
    post = [(e, v) for e, v in pairs if e > warmup]
    if not pre or not post:
        side = "post-warmup" if not post else "pre-warmup"
        return {"verdict": "insufficient_epochs",
                "text": f"no usable disc-only {side} epochs to compare across warmup",
                "delta": None, "onset_epoch": None,
                "warmup_epochs": warmup,
                "pre_best": (max(v for _, v in pre) if pre else None),
                "post_best": (max(v for _, v in post) if post else None)}
    pre_best = max(v for _, v in pre)
    pre_mean = sum(v for _, v in pre) / len(pre)
    post_best = max(v for _, v in post)
    post_mean = sum(v for _, v in post) / len(post)
    delta_best = post_best - pre_best
    delta_mean = post_mean - pre_mean
    # onset: first post-warmup epoch whose disc-only detection exceeds the pre ceiling.
    onset = next((e for e, v in post if v > pre_best), None)
    # "meaningful" = the post-warmup disc-only ceiling clears the pre ceiling by a
    # non-trivial margin (chance ROC is 0.5; we use a modest absolute lift threshold).
    meaningful_lift = 0.02
    rises = delta_best > meaningful_lift
    verdict = "disc_efficacy_confirmed" if rises else "disc_efficacy_weak"
    if rises:
        text = (f"disc-only {metric.upper()} RISES after warmup: best pre={pre_best:.4f} "
                f"→ post={post_best:.4f} (Δ={delta_best:+.4f}); first exceeds the "
                f"pre-warmup ceiling at epoch {onset}. Letting the student/discrepancy "
                f"path in adds genuine detection signal post-warmup.")
    else:
        text = (f"disc-only {metric.upper()} does NOT rise meaningfully after warmup: "
                f"best pre={pre_best:.4f} → post={post_best:.4f} (Δ={delta_best:+.4f} "
                f"≤ {meaningful_lift}). The discrepancy path does not add standalone "
                f"detection here — the detector leans on the teacher/recon channel.")
    return {
        "verdict": verdict,
        "text": text,
        "metric": metric,
        "warmup_epochs": warmup,
        "pre_best": pre_best,
        "post_best": post_best,
        "delta_best": delta_best,
        "delta_mean": delta_mean,
        "onset_epoch": onset,
        "meaningful_lift_threshold": meaningful_lift,
    }


# ── E-4 (I3) — score-vs-time component overlay (one epoch, TEST axis) ─────────
def component_overlay(repo: Repository, bundle: LeafBundle, *,
                      epoch: Optional[int] = None, downsample: int = 4000
                      ) -> dict[str, Any]:
    """E-4 (I3). On the TEST time axis at a chosen epoch, overlay the three score
    channels — teacher_recon_error, discrepancy_error, adaptive_score — plus
    point_labels and the derived anomaly_regions (contiguous label==1 spans) so the
    frontend can show WHERE each channel fires. Reuses the lazy ``read_arrays`` with
    downsample (read-cheap, one NPZ). Tolerant; never opens *.pt."""
    leaf_path = bundle.leaf.leaf_path
    target = epoch if epoch is not None else _best_epoch(bundle)
    ep = _nearest_score_epoch(leaf_path, target)
    if ep is None:
        return {"available": False, "reason": "no epoch_scores NPZ"}
    npz = _npz_path(leaf_path, ep)
    if not npz.exists():
        return {"available": False, "reason": "NPZ absent", "epoch": ep}
    try:
        res = read_arrays(
            npz,
            ["adaptive_score", "teacher_recon_error", "discrepancy_error",
             "point_labels"],
            downsample=downsample,
        )
    except CorruptOrWriting:
        return {"available": False, "reason": "corrupt_or_writing", "epoch": ep}

    a = res["arrays"]
    labels = a.get("point_labels") or []
    # contiguous anomaly regions (on the DOWNSAMPLED index space the arrays live in —
    # the frontend plots channels on the same index axis, so regions align).
    regions: list[dict[str, int]] = []
    i = 0
    nlab = len(labels)
    while i < nlab:
        if labels[i] == 1:
            j = i
            while j < nlab and labels[j] == 1:
                j += 1
            regions.append({"start": i, "end": j - 1})
            i = j
        else:
            i += 1
    return {
        "available": True,
        "epoch": ep,
        "best_epoch": _best_epoch(bundle),
        "requested_epoch": epoch,
        "length": res.get("length"),
        "downsampled": res.get("downsampled"),
        "n_points": len(a.get("adaptive_score") or labels),
        "channels": {
            "adaptive_score": a.get("adaptive_score"),
            "teacher_recon_error": a.get("teacher_recon_error"),
            "discrepancy_error": a.get("discrepancy_error"),
        },
        "point_labels": labels,
        "anomaly_regions": regions,
        "available_arrays": res.get("available_arrays"),
        "data_source": (f"epoch_scores/epoch_{ep:03d}_scores.npz "
                        "(adaptive_score, teacher_recon_error, discrepancy_error, "
                        "point_labels) on the test time axis"),
        "label": ("teacher/disc/adaptive channels on the TEST axis at the chosen epoch "
                  "(read directly from the saved NPZ — no recompute, no *.pt)"),
    }
