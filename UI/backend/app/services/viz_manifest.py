"""Visualization manifest — backend-design.md §5.10.

Enumerate the PNGs that ACTUALLY exist under a leaf's ``visualization/`` tree
(conditional FM/GRL/SCAD images are enumerated, never assumed), flag staleness from
``STALE_VIZ.txt`` (or the metadata backfill caveat), and surface ``gif_available`` +
the matching cached-GIF descriptors from the GIF cache index (FE-FB-5) so the gallery
cross-links without re-deriving the sha1 cache key.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ..dataaccess.models import LeafState
from .compare import leaf_id


def _rel(leaf_path: Path, p: Path) -> str:
    try:
        return str(p.relative_to(leaf_path))
    except ValueError:
        return p.name


def viz_manifest(
    leaf: LeafState,
    *,
    exp_id: str,
    dataset_key: str,
    gif_index: Optional[list[dict]] = None,
) -> dict[str, Any]:
    """Enumerate present PNGs + stale flags + cached-GIF cross-links for one leaf."""
    leaf_path = leaf.leaf_path
    viz = leaf_path / "visualization"
    categories: dict[str, list[str]] = {"epoch_metrics": [], "best_model": []}
    abs_paths: dict[str, str] = {}
    if viz.is_dir():
        for cat in categories:
            sub = viz / cat
            if sub.is_dir():
                for png in sorted(sub.glob("*.png")):
                    rel = _rel(leaf_path, png)
                    categories[cat].append(rel)
                    abs_paths[rel] = str(png.resolve())
        # any other categories present under visualization/ (forward-compatible).
        for other in sorted(viz.iterdir()):
            if other.is_dir() and other.name not in categories:
                pngs = [_rel(leaf_path, p) for p in sorted(other.glob("*.png"))]
                if pngs:
                    categories[other.name] = pngs
                    for png in other.glob("*.png"):
                        abs_paths[_rel(leaf_path, png)] = str(png.resolve())

    # staleness: STALE_VIZ.txt anywhere under the leaf, or the backfill caveat marker.
    stale: list[str] = []
    stale_reason: Optional[str] = None
    for marker in leaf_path.rglob("STALE_VIZ.txt"):
        stale.append(_rel(leaf_path, marker.parent))
        if stale_reason is None:
            try:
                stale_reason = marker.read_text("utf-8", errors="replace")[:200].strip()
            except OSError:
                stale_reason = "stale viz marker present"

    # GIF cross-link (FE-FB-5): match cached GIFs whose leaves include this leaf.
    lid = leaf_id(exp_id, dataset_key, leaf.variant)
    gifs: list[dict] = []
    if gif_index:
        for g in gif_index:
            leaf_ids = {f"{s.get('exp_id')}|{s.get('dataset_key')}|{s.get('variant') or '_'}"
                        for s in g.get("experiment_set", [])}
            # P4B-03: require an EXACT leaf_id membership. The old ``or not leaf_ids``
            # clause advertised an empty-experiment_set GIF on every leaf (false positive).
            if lid in leaf_ids:
                gifs.append({"cache_key": g.get("cache_key"), "story_id": g.get("story_id"),
                             "metric_keys": g.get("metric_keys")})

    return {
        "exp_id": exp_id, "dataset_key": dataset_key, "variant": leaf.variant,
        "categories": categories,
        "abs_paths": abs_paths,                 # for /api/files/png?path=
        "stale": stale,
        "stale_reason": stale_reason,
        "gif_available": bool(gifs),
        "gifs": gifs,
    }
