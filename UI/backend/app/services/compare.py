"""Multi-experiment comparison + config-diff services — backend-design.md §5.2 / §5.6.

All endpoints take the SHARED ``LeafSel`` shape ``{exp_id, dataset_key, variant}``
(FE-FB-4) and echo ``leaves[]`` in request order with a stable
``leaf_id = "{exp_id}|{dataset_key}|{variant or '_'}"`` so the frontend can zip
config-axes against metric-axes by ``leaf_id``.

P3-05: cross-run "distribution" = N best-epoch SCALARS (one per leaf) — a strip/dot
dataset, never a synthetic spread. ``compare/matrix`` returns one scalar per
(leaf, metric); box/violin is reserved for the NPZ histogram endpoint.

Missing values are ``null`` (rendered "—"), NEVER a fabricated 0 (CMP-1).
"""
from __future__ import annotations

from typing import Any, Optional

from ..dataaccess.repository import Repository
from . import perf_basis


def leaf_id(exp_id: str, dataset_key: str, variant: Optional[str]) -> str:
    return f"{exp_id}|{dataset_key}|{variant or '_'}"


def _resolve_each(
    repo: Repository, leaves: list[dict]
) -> list[tuple[dict, Any, Any, Any]]:
    """Resolve each LeafSel to (echo, exp, ds, leaf). Bad selectors keep echo + None."""
    out: list[tuple[dict, Any, Any, Any]] = []
    for sel in leaves:
        exp_id = sel.get("exp_id", "")
        dataset_key = sel.get("dataset_key", "")
        variant = sel.get("variant")
        exp, ds, leaf = repo.resolve_leaf(exp_id, dataset_key, variant)
        echo = {
            "exp_id": exp_id,
            "dataset_key": dataset_key,
            "variant": (leaf.variant if leaf else variant),
            "leaf_id": leaf_id(exp_id, dataset_key, leaf.variant if leaf else variant),
        }
        out.append((echo, exp, ds, leaf))
    return out


def compare_matrix(
    repo: Repository,
    leaves: list[dict],
    metrics: list[str],
    *,
    score_variant: str = "metrics",
    epoch_basis: str = "best",
    threshold_basis: str = "optimal",
) -> dict[str, Any]:
    """Best-epoch metric matrix across N leaves × M metrics (≥1 metric per call)."""
    resolved = _resolve_each(repo, leaves)
    meta = {
        k: repo.registry.resolve(k, namespace="epoch_metrics").model_dump()
        for k in metrics
    }
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for echo, exp, ds, leaf in resolved:
        cells: dict[str, Any] = {}
        complete = leaf is not None and leaf.state == "complete"
        if complete:
            b = repo.load_bundle(exp, ds, leaf, want={"metadata"})
            errors.extend(b.errors)
        for k in metrics:
            val = (perf_basis.resolve_value(
                repo, exp, ds, leaf, k,
                epoch_basis=epoch_basis, threshold_basis=threshold_basis,
                score_variant=score_variant) if complete else None)
            cells[k] = {
                "value": val,                                  # null => "—", never 0
                "direction": meta[k]["direction"],
                "state": (leaf.state if leaf else "missing"),
            }
        rows.append({**echo, "cells": cells})
    return {
        "leaves": [echo for echo, *_ in resolved],
        "metrics": metrics,
        "score_variant": score_variant,
        "meta": meta,
        "rows": rows,
        "errors": errors,
    }


def _argbest(values: list[Optional[float]], direction: str) -> tuple[Optional[int], Optional[float]]:
    """Direction-aware best-epoch index/value over a per-epoch series (FB-4).

    argmax for ↑, argmin for ↓; ``None``/non-finite are SKIPPED (never treated as 0).
    Neutral metrics have no inherent "best" → returns ``(None, None)``. Returns the
    INDEX into the series (the frontend maps it to the epoch number) + the value."""
    import math
    if direction not in ("up", "down"):
        return None, None
    best_i: Optional[int] = None
    best_v: Optional[float] = None
    for i, v in enumerate(values):
        if not (isinstance(v, (int, float)) and math.isfinite(v)):
            continue
        if best_v is None or (v < best_v if direction == "down" else v > best_v):
            best_v, best_i = float(v), i
    return best_i, best_v


def compare_series(
    repo: Repository, leaves: list[dict], key: str
) -> dict[str, Any]:
    """Per-eval-epoch overlay of one metric across N leaves; align by epoch NUMBER.

    FB-4: each series carries its direction-aware ``best_epoch_index``/``best_epoch``/
    ``best_value`` so the frontend can mark the best-epoch point (additive — existing
    consumers ignore the new fields).

    FB-3 / R2-05 / P3-01: the per-epoch series is served ONLY from ``epoch.series``
    (the genuine per-eval-epoch ``epoch_metrics``). A metric that exists only as a
    best-epoch SCALAR (student/disc detection variants are NOT in ``epoch.series``)
    yields ``epochs=[]``/``values=[]`` (empty/gaps) and ``available=False`` for that
    leaf — NEVER a fabricated per-epoch trajectory."""
    resolved = _resolve_each(repo, leaves)
    meta = repo.registry.resolve(key, namespace="epoch_metrics").model_dump()
    direction = meta.get("direction", "neutral")
    series: list[dict[str, Any]] = []
    errors: list[str] = []
    for echo, exp, ds, leaf in resolved:
        epochs: list[int] = []
        values: list[Optional[float]] = []
        warmup = None
        available = False
        if leaf is not None:
            b = repo.load_bundle(exp, ds, leaf, want={"epoch", "metadata"})
            errors.extend(b.errors)
            if b.epoch is not None and key in b.epoch.series:
                epochs = b.epoch.epochs
                values = b.epoch.series[key]
                warmup = b.epoch.warmup_epochs
                available = True
        best_i, best_v = _argbest(values, direction)
        best_epoch = (epochs[best_i] if best_i is not None and best_i < len(epochs)
                      else None)
        series.append({**echo, "epochs": epochs, "values": values,
                       "warmup_epochs": warmup, "available": available,
                       "best_epoch_index": best_i, "best_epoch": best_epoch,
                       "best_value": best_v})
    return {
        "leaves": [echo for echo, *_ in resolved],
        "key": key,
        "meta": meta,
        "direction": direction,
        "series": series,
        "align": "epoch_number",
        "errors": errors,
    }


def config_diff(repo: Repository, leaves: list[dict]) -> dict[str, Any]:
    """UNION of config keys across N leaves; per-key {values, differs, added/removed}.

    An ablation flag present in one run but absent in another shows as
    ``added_in``/``removed_in`` (CFG-2). Values are compared by equality.
    """
    resolved = _resolve_each(repo, leaves)
    configs: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for echo, exp, ds, leaf in resolved:
        cfg: dict[str, Any] = {}
        if leaf is not None:
            b = repo.load_bundle(exp, ds, leaf, want={"config", "metadata"})
            cfg = b.config or {}
            errors.extend(b.errors)
        configs[echo["leaf_id"]] = cfg

    all_keys: set[str] = set()
    for cfg in configs.values():
        all_keys |= set(cfg.keys())

    out_keys: list[dict[str, Any]] = []
    for k in sorted(all_keys):
        values: dict[str, Any] = {}
        present_in: list[str] = []
        absent_in: list[str] = []
        for lid, cfg in configs.items():
            if k in cfg:
                values[lid] = cfg[k]
                present_in.append(lid)
            else:
                absent_in.append(lid)
        distinct = {repr(v) for v in values.values()}
        differs = len(distinct) > 1 or bool(absent_in)
        out_keys.append({
            "key": k,
            "values": values,
            "differs": differs,
            "added_in": present_in if absent_in else [],
            "removed_in": absent_in,
        })
    return {
        "leaves": [echo for echo, *_ in resolved],
        "keys": out_keys,
        "n_keys": len(out_keys),
        "errors": errors,
    }


# ── config grouping (§5.2 grouped view) ──────────────────────────────────────
_CONFIG_GROUPS: dict[str, tuple[str, ...]] = {
    "architecture": ("patchify", "patch", "cnn", "linear", "hidden", "latent",
                     "embed", "encoder", "decoder", "num_features", "window",
                     "d_model", "n_head", "layers", "dim"),
    "masking": ("mask", "patch_mask", "mask_ratio"),
    "loss_margin": ("margin", "hinge", "softplus", "dynamic_margin", "loss"),
    "grl": ("grl", "domain", "adversar"),
    "fm": ("fm_", "feature_match", "_fm"),
    "scad": ("scad", "z_separation"),
    "score": ("score_", "recon_disc_ratio", "scoring", "lambda"),
    "schedule": ("warmup", "epoch", "lr", "scheduler", "batch", "optim", "weight_decay"),
}


def group_config(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Group ~110 dynamic config keys into the §5.2 buckets; unknown -> 'other'.

    This is a DISPLAY grouping only (substring heuristics) — every key still flows
    through (F5); a key matching no bucket lands in 'other', never dropped.
    """
    groups: dict[str, dict[str, Any]] = {g: {} for g in _CONFIG_GROUPS}
    groups["other"] = {}
    for k, v in config.items():
        kl = k.lower()
        placed = False
        for g, subs in _CONFIG_GROUPS.items():
            if any(s in kl for s in subs):
                groups[g][k] = v
                placed = True
                break
        if not placed:
            groups["other"][k] = v
    return {g: kv for g, kv in groups.items() if kv}
