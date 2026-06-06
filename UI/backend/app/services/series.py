"""Series + stats services — backend-design.md §5.3 / §5.4.

Builds the batched multi-key epoch-series (meta once-per-key), the separate
training-history axis, and direction-aware summary statistics. All direction logic
reads the resolver (registry), never a hard-coded rule.
"""
from __future__ import annotations

import math
from typing import Any, Optional

from ..dataaccess.parsers import anomaly_type_display_names
from ..dataaccess.repository import LeafBundle, Repository


def epoch_series(
    repo: Repository, bundle: LeafBundle, keys: list[str], *, include_meta: bool = True
) -> dict[str, Any]:
    """Batched multi-key per-EVAL-epoch series (FE-FB-1). meta once-per-key."""
    epoch = bundle.epoch
    if epoch is None:
        return {"axis": "eval-epoch", "epochs": [], "series": {}, "meta": {},
                "warmup_epochs": bundle.leaf.warmup_epochs, "errors": bundle.errors,
                "available": False}
    avail = set(epoch.series.keys())
    sel = [k for k in keys if k in avail] if keys else sorted(avail)
    series = {k: epoch.series[k] for k in sel}
    meta = ({k: repo.registry.resolve(k, namespace="epoch_metrics").model_dump() for k in sel}
            if include_meta else {})
    return {
        "axis": "eval-epoch",
        "epochs": epoch.epochs,
        "series": series,
        "meta": meta,
        "warmup_epochs": epoch.warmup_epochs,
        "requested_missing": [k for k in keys if k not in avail],
        "errors": bundle.errors,
    }


def training_history(
    repo: Repository, bundle: LeafBundle, keys: list[str], *, include_meta: bool = True
) -> dict[str, Any]:
    """Per-TRAINING-epoch axis (P2-01). Separate axis; pivoted nested families."""
    h = bundle.history
    if h is None:
        return {"axis": "training-epoch", "train_epochs": [], "series": {}, "meta": {},
                "nested": {}, "errors": bundle.errors, "available": False}
    avail = set(h.series.keys())
    sel = [k for k in keys if k in avail] if keys else sorted(avail)
    series = {k: h.series[k] for k in sel}
    meta = ({k: repo.registry.resolve(k, namespace="history").model_dump() for k in sel}
            if include_meta else {})
    nested_out: dict[str, Any] = {}
    for nk, piv in h.nested.items():
        sub_meta = {sm: repo.registry.resolve(sm, namespace="history").model_dump()
                    for sm in piv.sub_metrics} if include_meta else {}
        nested_out[nk] = {
            "types": piv.types,
            "sub_metrics": piv.sub_metrics,
            "pivot": piv.pivot,
            "sub_metric_meta": sub_meta,
            # FB-7 / R2-06: scoped per-type display names over the pivot's discovered
            # type vocabulary ({normal, spike} -> {Normal, Anomaly} single-bucket;
            # simulation 9-type kept). Frontend reads ``type_display_names[type]``.
            "type_display_names": anomaly_type_display_names(piv.types),
        }
    return {
        "axis": "training-epoch",
        "train_epochs": h.train_epochs,
        "series": series,
        "meta": meta,
        "nested": nested_out,
        "empty_keys": h.empty_keys,
        "errors": bundle.errors,
    }


def _finite(xs: list[Optional[float]]) -> list[float]:
    return [x for x in xs if isinstance(x, (int, float)) and math.isfinite(x)]


def stats_for_keys(
    repo: Repository, bundle: LeafBundle, keys: list[str]
) -> dict[str, Any]:
    """Direction-aware per-key summary stats (§5.4). best = max for up / min for down."""
    epoch = bundle.epoch
    out: dict[str, Any] = {}
    if epoch is None:
        return out
    sel = [k for k in keys if k in epoch.series] if keys else sorted(epoch.series)
    epochs = epoch.epochs
    for k in sel:
        vals = epoch.series[k]
        meta = repo.registry.resolve(k, namespace="epoch_metrics")
        pairs = [(epochs[i], v) for i, v in enumerate(vals)
                 if isinstance(v, (int, float)) and math.isfinite(v)]
        if not pairs:
            out[k] = {"direction": meta.direction, "available": False}
            continue
        fv = [v for _, v in pairs]
        if meta.direction == "down":
            best_ep, best = min(pairs, key=lambda p: p[1])
        else:  # up or neutral -> report max (neutral has no ranking sense)
            best_ep, best = max(pairs, key=lambda p: p[1])
        first, last = fv[0], fv[-1]
        # trailing-window slope (last 20% of points) for a plateau flag
        tail = fv[max(0, int(len(fv) * 0.8)):]
        slope = (tail[-1] - tail[0]) / max(1, len(tail) - 1) if len(tail) > 1 else 0.0
        out[k] = {
            "direction": meta.direction,
            "rankable": meta.direction in ("up", "down"),
            "best": best, "argbest_epoch": best_ep,
            "mean": sum(fv) / len(fv), "first": first, "last": last,
            "slope": slope, "plateau": abs(slope) < 1e-4,
            "delta_best_last": best - last, "delta_best_first": best - first,
            "n_points": len(fv),
        }
    return out
