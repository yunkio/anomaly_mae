"""Performance basis resolution — epoch × threshold selectable across perf tables.

PERF_BASIS_SPEC (2026-06-11, authoritative). Two orthogonal bases:

* **Epoch basis** ∈ {``best``, ``last``, or a numeric epoch like ``"300"``/``"400"``}:
  ``best`` = the best-epoch snapshot (``experiment_metadata.json`` top-level score-variant
  blocks, the epoch maximizing ``pak_auc_f1``); ``last`` = the final eval epoch from the
  per-epoch ``epoch_metrics`` series; a NUMERIC epoch N = the value at eval epoch N (nearest
  eval ≤ N; ``None`` if the run never reached epoch N). Every non-``best`` mode reads the
  per-epoch ``epoch_metrics`` series (adaptive = the ``metrics`` variant only).
* **Threshold basis** ∈ {``optimal``, ``anomaly_ratio``}: ``optimal`` = the F1-optimal
  threshold (current base keys); ``anomaly_ratio`` = the leakage-free ``_ar`` sibling,
  swapped in ONLY for threshold-dependent metrics (i.e. when the ``_ar`` sibling exists).
  Threshold-free metrics (``pak_auc_f1``, ``prc_auc``, …) ignore ``threshold_basis``.

DEFAULT = (best, optimal) = current behaviour, byte-identical.
"""
from __future__ import annotations

from typing import Optional

EPOCH_BASES = ("best", "best_post", "es", "last")  # plus any numeric epoch, e.g. "300"
# "best_post" = best epoch RESTRICTED to post-warm-up (epoch > teacher_only_warmup_epochs).
# "es" = the epoch an EARLY-STOPPING rule on G_e would select (see _es_selected_epoch).
THRESHOLD_BASES = ("optimal", "anomaly_ratio")

# ── Early-stopping (ES) selection on the leakage-free game-health scalar G_e ──────────
# G_e (raw relgap) = (train_student_recon_anomaly - train_normal_loss)
#       / (|train_student_recon_anomaly| + |train_normal_loss| + eps)
# evaluated per TRAINING epoch (training_histories), restricted to POST-warm-up.
#
# EVAL-GRID decision (PERF_STORE_SPEC 2026-06-16): the EMA (alpha 0.1) is updated over
# EVERY post-warm-up TRAINING epoch in order — building ``ema_by_epoch`` — but the STOP
# DECISION (relative-improvement gate >1% / patience 2 / rollback to best seen BEFORE the
# trigger) runs ONLY at the post-warm-up EVAL epochs that exist in ``epoch_metrics``. The
# selected epoch is therefore ALWAYS an eval epoch, so the metric value is read EXACTLY at
# it (no nearest-epoch mapping). Uses ONLY training-side quantities → no test-label leak.
ES_EMA_ALPHA = 0.1
ES_PATIENCE = 2
ES_MIN_REL_IMPROVE = 0.01
ES_EPS = 1e-8


def _es_selected_epoch(history, eval_epochs, warmup: int) -> Optional[int]:
    """Replay the EVAL-GRID ES rule; return the selected (rolled-back best-EMA) EVAL
    epoch, or None if it can't be computed.

    The EMA over G_e is updated at EVERY post-warm-up TRAINING epoch (``ema_by_epoch``);
    the improve/patience/rollback STOP decision runs ONLY at the post-warm-up EVAL epochs
    (``eval_epochs``, from ``epoch_metrics``). ``selected`` is one of ``eval_epochs``.
    """
    sa = (history.series.get("train_student_recon_anomaly") if history else None) or []
    nl = (history.series.get("train_normal_loss") if history else None) or []
    train_epochs = (history.train_epochs if history else None) or list(range(1, len(sa) + 1))
    n = min(len(sa), len(nl), len(train_epochs))

    # ── EMA over EVERY post-warm-up TRAINING epoch (skip e ≤ warmup entirely) ──
    ema = None
    ema_by_epoch: dict[int, float] = {}
    for i in range(n):
        e = train_epochs[i]
        if e <= warmup:            # teacher-only warm-up → not judged (spec step 1)
            continue
        sra, nrm = sa[i], nl[i]
        if sra is None or nrm is None:
            continue
        raw = (sra - nrm) / (abs(sra) + abs(nrm) + ES_EPS)
        ema = raw if ema is None else ES_EMA_ALPHA * raw + (1.0 - ES_EMA_ALPHA) * ema
        ema_by_epoch[e] = ema

    # ── STOP decision ONLY at post-warm-up EVAL epochs ────────────────────────
    best_metric = None
    best_epoch = None
    wait = 0
    for ev in (eval_epochs or []):
        if ev <= warmup:           # pre/at warm-up eval points are not judged
            continue
        cur = ema_by_epoch.get(ev)
        if cur is None:            # no EMA available at this eval epoch → skip
            continue
        if best_epoch is None:
            improved = True
        else:
            rel = (cur - best_metric) / max(abs(best_metric), ES_EPS)
            improved = rel > ES_MIN_REL_IMPROVE
        if improved:
            best_metric = cur
            best_epoch = ev
            wait = 0
        else:
            wait += 1
        if wait >= ES_PATIENCE:    # trigger → stop; selected = best seen BEFORE trigger
            break
    return best_epoch


def _resolve_es(repo, exp, ds, leaf, base_key, threshold_basis) -> Optional[float]:
    """Value of ``base_key`` at the ES-selected EVAL epoch (read EXACTLY at it — the
    eval-grid decision guarantees ``selected`` is itself an eval epoch)."""
    b = repo.load_bundle(exp, ds, leaf, want={"epoch", "history"})
    if not b.epoch or not b.history:
        return None
    warmup = b.epoch.warmup_epochs if isinstance(b.epoch.warmup_epochs, int) else 0
    eval_epochs = b.epoch.epochs or []
    ser = b.epoch.series or {}
    selected = _es_selected_epoch(b.history, eval_epochs, warmup)
    if selected is None or not eval_epochs:
        return None
    ek = base_key
    if threshold_basis == "anomaly_ratio" and ar_key(base_key) in ser:
        ek = ar_key(base_key)
    vals = ser.get(ek) or []
    try:
        idx = eval_epochs.index(selected)       # EXACT eval epoch (no nearest mapping)
    except ValueError:
        return None
    if idx >= len(vals):
        return None
    v = vals[idx]
    return float(v) if v is not None else None


def ar_key(base: str) -> str:
    return {"f1_score": "f1_ar"}.get(base, base + "_ar")


def effective_key(block: dict, base: str, threshold_basis: str) -> str:
    if threshold_basis == "anomaly_ratio":
        ak = ar_key(base)
        if ak in block:
            return ak          # swap only if the AR sibling exists
    return base                # threshold-free / optimal → base key


def resolve_value(repo, exp, ds, leaf, base_key, *, epoch_basis, threshold_basis,
                  score_variant="metrics") -> Optional[float]:
    """Thin value-only API — delegates to the precomputed perf store (PERF_STORE_SPEC)."""
    return resolve_value_epoch(
        repo, exp, ds, leaf, base_key,
        epoch_basis=epoch_basis, threshold_basis=threshold_basis,
        score_variant=score_variant)[0]


def resolve_value_epoch(repo, exp, ds, leaf, base_key, *, epoch_basis, threshold_basis,
                        score_variant="metrics"):
    """``(value, epoch)`` for ``base_key`` under (epoch_basis, threshold_basis).

    Served from the precomputed perf store (``services/perf_store.get``), which builds
    one per-basis snapshot per leaf ONCE and serves from memory/disk thereafter. The
    direct per-call path below (``_resolve_value_direct``) is the SAME computation the
    store performs and is kept as the canonical reference + fallback if the store path
    is ever unavailable. (Lazy import → perf_store may import perf_basis.)"""
    from . import perf_store
    return perf_store.get(
        repo, exp, ds, leaf, base_key,
        epoch_basis=epoch_basis, threshold_basis=threshold_basis,
        score_variant=score_variant)


def _resolve_value_direct(repo, exp, ds, leaf, base_key, *, epoch_basis, threshold_basis,
                          score_variant="metrics") -> Optional[float]:
    """The original heavy per-call resolver (reference implementation the store mirrors)."""
    if epoch_basis == "best":
        b = repo.load_bundle(exp, ds, leaf, want={"metadata"})
        block = (b.metadata.score_variants.get(score_variant, {}) if b.metadata else {}) or {}
        return block.get(effective_key(block, base_key, threshold_basis))
    # non-best ("es", "best_post", "last", or a numeric epoch): read the per-epoch series
    # (adaptive = "metrics" only; other score variants have no per-epoch series → None).
    if score_variant != "metrics":
        return None
    if epoch_basis == "es":
        return _resolve_es(repo, exp, ds, leaf, base_key, threshold_basis)
    b = repo.load_bundle(exp, ds, leaf, want={"epoch"})
    ser = (b.epoch.series if b.epoch else {}) or {}
    epochs = (b.epoch.epochs if b.epoch else []) or []
    # effective key against the SERIES keyset (so AR swap still applies on this path)
    ek = base_key
    if threshold_basis == "anomaly_ratio" and ar_key(base_key) in ser:
        ek = ar_key(base_key)
    vals = ser.get(ek) or []
    if not vals:
        return None

    if epoch_basis == "best_post":
        # "best epoch, restricted to AFTER warm-up": the epoch maximizing pak_auc_f1 among
        # post-warmup epochs (epoch > teacher_only_warmup_epochs), then THIS metric's value
        # there. Differs from "best" (which is argmax over ALL epochs, incl. teacher-only
        # warmup) when a pre-warmup epoch happens to top pak_auc_f1.
        warmup = b.epoch.warmup_epochs if (b.epoch and isinstance(b.epoch.warmup_epochs, int)) else 0
        pak = ser.get("pak_auc_f1") or []
        best_i = None
        best_pv = None
        for i, ep in enumerate(epochs):
            if ep > warmup and i < len(pak) and pak[i] is not None:
                if best_pv is None or pak[i] > best_pv:
                    best_pv = pak[i]
                    best_i = i
        if best_i is None or best_i >= len(vals):
            return None
        v = vals[best_i]
        return float(v) if v is not None else None

    if epoch_basis == "last":
        for v in reversed(vals):            # last NON-null value at/near the final epoch
            if v is not None:
                return float(v)
        return None

    # numeric epoch N: the value at eval epoch N (nearest eval ≤ N). Honest gap (None) if
    # the run never reached epoch N, so a short run is never silently relabelled as epoch-N.
    try:
        target = int(epoch_basis)
    except (TypeError, ValueError):
        return None
    if not epochs or epochs[-1] < target:
        return None                         # run never reached epoch N → "—"
    idx = None
    for i, ep in enumerate(epochs):         # epochs is ascending; take the largest ≤ target
        if ep <= target:
            idx = i
        else:
            break
    if idx is None or idx >= len(vals):
        return None
    v = vals[idx]
    return float(v) if v is not None else None
