"""Array-valued epoch-metric services — per-feature heatmap, PA%K curve.

backend-design.md §5.3 (per-feature, pak-curve). These read the ARRAY-valued keys in
``epoch_metrics.json::epochs[]`` (recorded in ``EpochSeries.array_keys`` but not
flattened into the scalar series map). Re-read the raw file lazily for the requested
array field so the scalar path stays cheap.

Phase masking is honored: ``_train_feature_disc_*`` is null pre-warmup (the student/
disc engine has not joined), so the heatmap greys that span — the renderer / frontend
reads ``mask_prewarmup`` + ``warmup_epochs`` from the response.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ..dataaccess.io_safe import read_json
from ..dataaccess.repository import LeafBundle, Repository

# array fields that carry the per-feature engine that only exists post-warmup.
_POST_WARMUP_ARRAY_PREFIXES = ("_train_feature_disc_", "_infer_feature_disc_")


def _read_epoch_entries(leaf_path: Path) -> list[dict]:
    doc = read_json(leaf_path / "epoch_metrics.json")
    if not isinstance(doc, dict):
        return []
    return [e for e in (doc.get("epochs") or []) if isinstance(e, dict)]


def per_feature(
    repo: Repository, bundle: LeafBundle, field: str
) -> dict[str, Any]:
    """Build a {epochs, features, matrix[[..]]} for one array-valued field (§5.3, B11).

    ``matrix[i][j]`` = feature j's value at eval-epoch i (null where absent). The field
    must be one of the leaf's ``array_keys`` (len = num_features) — never hard-coded.
    """
    epoch = bundle.epoch
    if epoch is None or field not in epoch.array_keys:
        return {"available": False, "field": field,
                "array_keys": (epoch.array_keys if epoch else []),
                "errors": bundle.errors}

    entries = _read_epoch_entries(bundle.path)
    epochs: list[int] = []
    rows: list[list[Optional[float]]] = []
    nfeat = 0
    for e in entries:
        ep = e.get("epoch")
        epochs.append(int(ep) if isinstance(ep, (int, float)) else len(epochs) + 1)
        vec = e.get(field)
        if isinstance(vec, list):
            row = [float(x) if isinstance(x, (int, float)) else None for x in vec]
            nfeat = max(nfeat, len(row))
            rows.append(row)
        else:
            rows.append([])
    # right-pad every row to nfeat with None so the matrix is rectangular.
    matrix = [r + [None] * (nfeat - len(r)) for r in rows]
    mask = field.startswith(_POST_WARMUP_ARRAY_PREFIXES) and field.startswith("_train_feature_disc_")
    meta = repo.registry.resolve(field, namespace="epoch_metrics").model_dump()
    return {
        "available": True,
        "field": field,
        "meta": meta,
        "epochs": epochs,
        "features": list(range(nfeat)),
        "matrix": matrix,
        "mask_prewarmup": mask or meta.get("phase_validity") == "post-warmup",
        "warmup_epochs": epoch.warmup_epochs,
        "errors": bundle.errors,
    }


def pak_curve(
    repo: Repository, bundle: LeafBundle, epoch_num: Optional[int] = None
) -> dict[str, Any]:
    """Return the PA%K sweep (`_per_k_{f1,prc_auc,roc_auc}` vs `_per_k_values`) at one
    eval-epoch (default the LAST). One curve set, not 105 flattened scalars (§5.3, B3).
    """
    epoch = bundle.epoch
    needed = {"_per_k_values", "_per_k_f1", "_per_k_prc_auc", "_per_k_roc_auc"}
    if epoch is None or not (needed & set(epoch.array_keys)):
        return {"available": False, "errors": bundle.errors}
    entries = _read_epoch_entries(bundle.path)
    if not entries:
        return {"available": False, "errors": bundle.errors}
    target = None
    if epoch_num is not None:
        target = next((e for e in entries if e.get("epoch") == epoch_num), None)
    target = target or entries[-1]

    def _vec(name: str) -> list[Optional[float]]:
        v = target.get(name)
        return [float(x) if isinstance(x, (int, float)) else None for x in v] if isinstance(v, list) else []

    return {
        "available": True,
        "epoch": target.get("epoch"),
        "k_values": _vec("_per_k_values"),
        "curves": {
            "f1": _vec("_per_k_f1"),
            "prc_auc": _vec("_per_k_prc_auc"),
            "roc_auc": _vec("_per_k_roc_auc"),
        },
        "errors": bundle.errors,
    }


def separation(repo: Repository, bundle: LeafBundle) -> dict[str, Any]:
    """Separation magnitudes (neutral pairs) + ratios/SNR (↑) from loss_stats (§5.7, B7).

    P1-02: raw magnitudes are NEUTRAL (no inherent better); only the ``_ratio`` /
    ``_SNR`` summaries carry an up-direction. Direction comes from the resolver.
    """
    md = bundle.metadata
    if md is None or not md.loss_stats:
        return {"available": False, "errors": bundle.errors}
    pairs: list[dict[str, Any]] = []
    ratios: list[dict[str, Any]] = []
    for k, v in md.loss_stats.items():
        meta = repo.registry.resolve(k, namespace="loss_stats")
        entry = {"key": k, "value": v, "direction": meta.direction,
                 "family": meta.family, "inferred": meta.inferred}
        if meta.direction == "up" and ("ratio" in k.lower() or "snr" in k.lower()):
            ratios.append(entry)
        else:
            pairs.append(entry)
    return {
        "available": True,
        "pairs": pairs,            # neutral magnitudes
        "ratios_snr": ratios,      # up-direction summaries
        "errors": bundle.errors,
    }
