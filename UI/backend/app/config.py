"""Runtime configuration — paths, source roots, cache location, tunables.

Everything is overridable via environment variables so the verifier can point at
fixtures without editing code. NO metric/key lists live here (F5: nothing about
metrics is hard-coded anywhere). Defaults are derived from the project layout.

backend-design.md §1.3 (process model) + §7 (cache under ./UI/.cache).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_path(name: str, default: Path) -> Path:
    v = os.environ.get(name)
    return Path(v).expanduser().resolve() if v else default


# ── project anchors ─────────────────────────────────────────────────────────
# This file lives at  <project_root>/UI/backend/app/config.py
_THIS = Path(__file__).resolve()
UI_DIR = _THIS.parents[2]                     # .../UI
PROJECT_ROOT = UI_DIR.parent                  # project root (/home/ykio/notebooks/TSMAE)


@dataclass(frozen=True)
class Settings:
    # ── source roots (READ-ONLY; the data-access layer opens these 'rb' only) ──
    # The live, in-progress run lives under results/experiments/; the rich
    # read-only fixtures live under .trash/0601/pre_fullrerun/ (DECISION_LOG #6).
    live_root: Path = field(
        default_factory=lambda: _env_path(
            "TSMAE_LIVE_ROOT", PROJECT_ROOT / "results" / "experiments"
        )
    )
    fixture_root: Path = field(
        default_factory=lambda: _env_path(
            "TSMAE_FIXTURE_ROOT", PROJECT_ROOT / ".trash" / "0601" / "pre_fullrerun"
        )
    )
    # Third source (FB-R4d-01): archived prior runs (new_/old_ prefixed dirs with
    # rich SMD/SMAP/MSL multi-entity leaves). READ-ONLY like the others.
    old_root: Path = field(
        default_factory=lambda: _env_path(
            "TSMAE_OLD_ROOT", PROJECT_ROOT / "temp" / "old_experiments"
        )
    )

    # ── registry (authoritative metric semantics) ─────────────────────────────
    registry_path: Path = field(
        default_factory=lambda: _env_path(
            "TSMAE_REGISTRY", UI_DIR / "registry" / "metric-semantics-registry.json"
        )
    )

    # ── cache (WRITES land here only; never under results/ or .trash/) ─────────
    cache_root: Path = field(
        default_factory=lambda: _env_path("TSMAE_CACHE", UI_DIR / ".cache")
    )

    # ── frontend static build (served by the same uvicorn; may be absent) ──────
    frontend_dist: Path = field(
        default_factory=lambda: _env_path(
            "TSMAE_FRONTEND_DIST", UI_DIR / "frontend" / "dist"
        )
    )

    # ── tunables ─────────────────────────────────────────────────────────────
    max_scan_depth: int = 2            # leaf dirs at depth 1 OR 2 (never assume; §2.1)
    gif_workers: int = 2              # bounded ThreadPool for heavy compute (§1.3)
    default_downsample: int = 4000     # NPZ slab downsample target (§5.8)
    host: str = os.environ.get("TSMAE_HOST", "127.0.0.1")
    port: int = int(os.environ.get("TSMAE_PORT", "8000"))

    # ── completion / discovery markers (file presence only; §2.1/§2.2) ─────────
    # These are FILE-KIND names, not metric keys — the stable part of the contract.
    leaf_marker_files: tuple[str, ...] = ("epoch_metrics.json", "experiment_metadata.json")
    completion_marker: str = "experiment_metadata.json"

    @property
    def source_roots(self) -> dict[str, Path]:
        # 2026-06-04: show ONLY ./results/experiments (the live root) by default, per
        # user request. The read-only fixture (.trash/0601/pre_fullrerun) and archived
        # old_experiments roots are now OPT-IN via env flags — set TSMAE_INCLUDE_FIXTURE=1
        # / TSMAE_INCLUDE_OLD=1 to restore them. The field definitions + env path
        # overrides are kept so re-enabling is a one-line flip, not a code resurrection.
        roots: dict[str, Path] = {"live": self.live_root}
        if os.environ.get("TSMAE_INCLUDE_FIXTURE", "").strip().lower() in ("1", "true", "yes", "on"):
            roots["fixture"] = self.fixture_root
        if os.environ.get("TSMAE_INCLUDE_OLD", "").strip().lower() in ("1", "true", "yes", "on"):
            roots["old"] = self.old_root
        return roots

    def cache_dir(self, sub: str) -> Path:
        p = self.cache_root / sub
        p.mkdir(parents=True, exist_ok=True)
        return p


SETTINGS = Settings()
