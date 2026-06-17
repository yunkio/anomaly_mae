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

EPOCH_BASES = ("best", "last")  # plus any numeric epoch string, e.g. "300" / "400"
THRESHOLD_BASES = ("optimal", "anomaly_ratio")


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
    if epoch_basis == "best":
        b = repo.load_bundle(exp, ds, leaf, want={"metadata"})
        block = (b.metadata.score_variants.get(score_variant, {}) if b.metadata else {}) or {}
        return block.get(effective_key(block, base_key, threshold_basis))
    # non-best ("last" or a numeric epoch): read the per-epoch series (adaptive = "metrics"
    # only; other score variants have no per-epoch series → None).
    if score_variant != "metrics":
        return None
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
