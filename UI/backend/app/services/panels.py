"""EXT-6 computed-panel plugin registry — backend-design.md §8.

A new derived view is a CPU compute function over the already-parsed ``LeafBundle``
— no core change. A panel declares ``requires`` (file-kinds the loader provides),
``cost``, and a pure ``compute(leaf, params)``. It inherits the read-only / lazy /
tolerant access by CONSTRUCTION: it only ever receives the normalized ``LeafBundle``
(no writable fs handle), so it cannot open a ``.pt`` or write under results/.

Register a panel with ``@register_panel("name")``; it is then exposed at
``GET /api/experiments/{id}/datasets/{key}/panels/{name}``.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from ..dataaccess.repository import LeafBundle, Repository

_REGISTRY: dict[str, "ComputePanel"] = {}


class ComputePanel:
    name: str = ""
    requires: list[str] = []          # parsed kinds the loader must provide
    cost: str = "read-cheap"          # "read-cheap" | "needs-compute"
    description: str = ""

    def compute(self, repo: Repository, leaf: LeafBundle, params: dict) -> dict:
        raise NotImplementedError


def register_panel(name: str, *, requires: Optional[list[str]] = None,
                   cost: str = "read-cheap", description: str = "") -> Callable:
    def deco(fn: Callable[[Repository, LeafBundle, dict], dict]) -> Callable:
        panel = ComputePanel()
        panel.name = name
        panel.requires = requires or []
        panel.cost = cost
        panel.description = description
        panel.compute = lambda repo, leaf, params, _fn=fn: _fn(repo, leaf, params)  # type: ignore
        _REGISTRY[name] = panel
        return fn
    return deco


def list_panels() -> list[dict[str, Any]]:
    return [{"name": p.name, "requires": p.requires, "cost": p.cost,
             "description": p.description} for p in _REGISTRY.values()]


def get_panel(name: str) -> Optional[ComputePanel]:
    return _REGISTRY.get(name)


def required_kinds(name: str) -> set[str]:
    p = _REGISTRY.get(name)
    return set(p.requires) if p else {"metadata", "epoch", "history"}


# ── built-in example panel (contribution-share trend over training) ───────────
@register_panel("separation_trend", requires=["history"], cost="read-cheap",
                description="Per-training-epoch recon & disc CONTRIBUTION-SHARE trend "
                            "(component/(recon+disc) per sample type; context — NPZ-free "
                            "diagnostic). FB-R4-03: this is the contribution share, NOT an "
                            "anomaly/normal (a/n) ratio.")
def _separation_trend(repo: Repository, leaf: LeafBundle, params: dict) -> dict:
    h = leaf.history
    if h is None:
        return {"available": False}
    keys = [k for k in h.series if "ratio" in k.lower() and k.startswith("epoch_")]
    series = {}
    for k in keys:
        meta = repo.registry.resolve(k, namespace="history")
        series[k] = {"values": h.series[k], "direction": meta.direction,
                     "family": meta.family}
    return {"available": True, "axis": "training-epoch",
            "train_epochs": h.train_epochs, "series": series}
