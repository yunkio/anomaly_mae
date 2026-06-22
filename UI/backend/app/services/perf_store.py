"""Precomputed PERF STORE — fast (value, epoch) serving for the ranking surfaces.

PERF_STORE_SPEC (2026-06-16, authoritative). The ranking / leaderboard / compare pages
must NOT re-parse every leaf's (large) ``epoch_metrics`` + ``training_histories`` on each
click. This module precomputes — ONCE per leaf — a per-EPOCH-BASIS snapshot
``{epoch, metrics{metric_key: value}}`` for the bases

    ("best", "best_post", "es", "last", "300", "350", "400", "450")

and serves ``(value, epoch)`` from memory thereafter. Each leaf entry is persisted to a
disk sidecar under ``UI/.cache/perf_store/<hash>.json`` keyed by the source files'
signatures (mtime/size of ``epoch_metrics`` + ``experiment_metadata`` +
``training_histories``); a live run rewriting any of those bumps the signature → that
leaf is rebuilt (mtime discipline, like ``MtimeCache``). Read-only over ``results/``;
writes land only under ``UI/.cache``; never opens a ``.pt``.

The ``best`` basis copies ``metadata.score_variants["metrics"]`` VERBATIM so the default
(best, optimal) stays BYTE-IDENTICAL to the legacy path. Every other (non-best) basis
snapshots the FULL ``epoch_metrics`` entry at the selected eval epoch's index — ALL
scalar keys, including the ``*_ar`` (anomaly-ratio threshold) siblings and the SWaT
``excl22_*`` keys — so ``effective_key`` (the AR swap) and the SWaT excl22 fallback both
keep working unchanged off the store.

``get`` REPLACES the per-call heavy resolve path: ``perf_basis.resolve_value`` /
``resolve_value_epoch`` delegate here.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

from ..config import SETTINGS
from . import perf_basis

# The epoch bases the store precomputes (PERF_STORE_SPEC §2). "best"/"best_post"/"es"/
# "last" + the numeric milestones the Performance Basis selector offers.
PERF_BASES: tuple[str, ...] = ("best", "best_post", "es", "last", "300", "350", "400", "450")

# the leaf files whose signatures gate a rebuild (mtime/size discipline).
_SIG_FILES = ("epoch_metrics.json", "experiment_metadata.json", "training_histories.json")

# in-process cache: leaf abspath -> {"sig": {...}, "bases": {basis: {epoch, metrics}}}.
_MEM: dict[str, dict[str, Any]] = {}


# ── signatures / paths ───────────────────────────────────────────────────────
def _leaf_signature(leaf_path: Path) -> dict[str, Optional[list]]:
    """``{filename: [st_mtime_ns, st_size]}`` over the gating files (None if absent)."""
    sig: dict[str, Optional[list]] = {}
    for name in _SIG_FILES:
        try:
            st = (leaf_path / name).stat()
            sig[name] = [st.st_mtime_ns, st.st_size]
        except OSError:
            sig[name] = None
    return sig


def _sidecar_path(leaf_path: Path) -> Path:
    h = hashlib.sha1(str(leaf_path.resolve()).encode("utf-8")).hexdigest()[:16]
    return SETTINGS.cache_dir("perf_store") / f"{h}.json"


def _atomic_write_json(target: Path, value: Any) -> None:
    """Atomic JSON write under UI/.cache (temp file + os.replace). Never raises out."""
    try:
        payload = json.dumps(value).encode("utf-8")
    except (TypeError, ValueError):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
        os.replace(tmp, target)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass


# ── per-basis epoch selection over a leaf's eval grid ────────────────────────
def _nearest_eval_le(eval_epochs: list[int], target: int) -> Optional[int]:
    """Index of the largest eval epoch ≤ ``target`` (None if the run never reached it).

    Honest gap (None) so a short run is never silently relabelled as epoch-N — matches
    ``perf_basis`` numeric-epoch semantics exactly."""
    if not eval_epochs or eval_epochs[-1] < target:
        return None
    idx = None
    for i, ep in enumerate(eval_epochs):     # ascending → largest ≤ target
        if ep <= target:
            idx = i
        else:
            break
    return idx


def _basis_eval_index(basis: str, eval_epochs: list[int], series: dict[str, list],
                      history, warmup: int) -> Optional[int]:
    """The eval-series INDEX a non-best basis selects (None if unavailable)."""
    if not eval_epochs:
        return None
    if basis == "last":
        return len(eval_epochs) - 1          # max eval epoch (PERF_STORE_SPEC §2)
    if basis == "best_post":
        # argmax pak_auc_f1 over POST-warm-up eval epochs (epoch > warmup).
        pak = series.get("pak_auc_f1") or []
        best_i = None
        best_pv = None
        for i, ep in enumerate(eval_epochs):
            if ep > warmup and i < len(pak) and pak[i] is not None:
                if best_pv is None or pak[i] > best_pv:
                    best_pv = pak[i]
                    best_i = i
        return best_i
    if basis == "es":
        selected = perf_basis._es_selected_epoch(history, eval_epochs, warmup)
        if selected is None:
            return None
        try:
            return eval_epochs.index(selected)   # EXACT eval epoch (eval-grid decision)
        except ValueError:
            return None
    # numeric milestone N → nearest eval ≤ N.
    try:
        target = int(basis)
    except (TypeError, ValueError):
        return None
    return _nearest_eval_le(eval_epochs, target)


def _entry_at_index(series: dict[str, list], idx: int) -> dict[str, Optional[float]]:
    """The full epoch_metrics SCALAR entry at eval index ``idx`` — ALL keys (incl.
    ``*_ar`` and ``excl22_*``). Array-valued keys are not in ``series`` by construction."""
    out: dict[str, Optional[float]] = {}
    for k, vals in series.items():
        v = vals[idx] if idx < len(vals) else None
        out[k] = (float(v) if isinstance(v, (int, float)) else None)
    return out


# ── build one leaf's per-basis snapshots ─────────────────────────────────────
def build_leaf_entry(repo, exp, ds, leaf) -> dict[str, Any]:
    """Load the leaf's metadata + epoch + history ONCE; build a per-basis snapshot.

    Returns ``{"sig": <signatures>, "bases": {basis: {"epoch": int|None,
    "metrics": {metric_key: value}}}}``. ``best`` copies the metadata ``metrics`` block
    verbatim (byte-identical default); every other basis snapshots the full epoch_metrics
    entry at its selected eval epoch.
    """
    # metadata via the cache (small; the aggregate gating reuses it, so keep it warm).
    b = repo.load_bundle(exp, ds, leaf, want={"metadata"})
    warmup = leaf.warmup_epochs if isinstance(leaf.warmup_epochs, int) else 0
    if (not isinstance(warmup, int) or warmup == 0) and b.metadata:
        w = (b.metadata.config or {}).get("teacher_only_warmup_epochs")
        warmup = w if isinstance(w, int) else warmup
    # epoch + history parsed DIRECTLY (TRANSIENT — NOT through the MtimeCache). Caching the
    # big per-epoch series for every leaf at build time used to FILL the bounded cache and
    # EVICT the metadata entries the aggregate gating relies on → 20 s re-parses + multi-GB
    # RSS (2026-06-16). Here they are parsed, used to snapshot the selected epochs, and then
    # GC'd; only the small metadata stays cached.
    from ..dataaccess.parsers import parse_epoch_metrics, parse_training_histories
    lp = leaf.leaf_path
    try:
        _epoch = parse_epoch_metrics(lp / "epoch_metrics.json", warmup if isinstance(warmup, int) else None)
    except Exception:  # noqa: BLE001 — a corrupt/absent series just yields empty snapshots
        _epoch = None
    try:
        _history = parse_training_histories(lp / "training_histories.json")
    except Exception:  # noqa: BLE001
        _history = None
    eval_epochs = (_epoch.epochs if _epoch else []) or []
    series = (_epoch.series if _epoch else {}) or {}

    bases: dict[str, dict[str, Any]] = {}

    # best → metadata.score_variants["metrics"] VERBATIM; epoch = metadata.timing.best_epoch.
    best_block = (b.metadata.score_variants.get("metrics", {}) if b.metadata else {}) or {}
    best_epoch = b.metadata.timing.get("best_epoch") if b.metadata else None
    bases["best"] = {
        "epoch": int(best_epoch) if isinstance(best_epoch, (int, float)) else None,
        "metrics": {k: (float(v) if isinstance(v, (int, float)) else None)
                    for k, v in best_block.items()},
    }

    # every other basis → full epoch_metrics entry at the selected eval index.
    for basis in PERF_BASES:
        if basis == "best":
            continue
        idx = _basis_eval_index(basis, eval_epochs, series, _history, warmup)
        if idx is None:
            bases[basis] = {"epoch": None, "metrics": {}}
            continue
        bases[basis] = {
            "epoch": int(eval_epochs[idx]),
            "metrics": _entry_at_index(series, idx),
        }

    return {"sig": _leaf_signature(leaf.leaf_path), "bases": bases}


# ── load / lazy-build one leaf entry (mem → disk sidecar → build) ────────────
def _load_entry(repo, exp, ds, leaf) -> dict[str, Any]:
    leaf_path = leaf.leaf_path
    key = str(leaf_path.resolve())
    cur_sig = _leaf_signature(leaf_path)

    mem = _MEM.get(key)
    if mem is not None and mem.get("sig") == cur_sig:
        return mem

    sidecar = _sidecar_path(leaf_path)
    if sidecar.exists():
        try:
            disk = json.loads(sidecar.read_text("utf-8"))
            if isinstance(disk, dict) and disk.get("sig") == cur_sig:
                _MEM[key] = disk
                return disk
        except (OSError, json.JSONDecodeError):
            pass

    entry = build_leaf_entry(repo, exp, ds, leaf)
    _MEM[key] = entry
    _atomic_write_json(sidecar, entry)
    return entry


# ── public API ────────────────────────────────────────────────────────────────
def get(repo, exp, ds, leaf, base_key, *, epoch_basis, threshold_basis,
        score_variant="metrics"):
    """``(value: Optional[float], epoch: Optional[int])`` for ``base_key`` under
    (``epoch_basis``, ``threshold_basis``), served from the precomputed store.

    Mirrors the legacy ``perf_basis`` rules exactly:
      * ``score_variant != "metrics"`` on a NON-best basis → ``(None, None)`` (no per-epoch
        series for the student/disc/teacher variants).
      * ``score_variant != "metrics"`` on ``best`` → read that metadata variant block.
      * AR swap via ``perf_basis.effective_key`` against the BASIS metrics block.
    """
    if leaf is None or getattr(leaf, "state", None) != "complete":
        return None, None

    # best basis honours an alternate score_variant by reading the metadata block directly
    # (the store snapshots only the "metrics" variant per epoch; best is the only basis
    # with the other variant blocks available — verbatim from metadata).
    if epoch_basis == "best" and score_variant != "metrics":
        b = repo.load_bundle(exp, ds, leaf, want={"metadata"})
        block = (b.metadata.score_variants.get(score_variant, {}) if b.metadata else {}) or {}
        ek = perf_basis.effective_key(block, base_key, threshold_basis)
        v = block.get(ek)
        ep = b.metadata.timing.get("best_epoch") if b.metadata else None
        return ((float(v) if isinstance(v, (int, float)) else None),
                (int(ep) if isinstance(ep, (int, float)) else None))

    # non-best + non-"metrics" variant → no per-epoch series (unchanged rule).
    if epoch_basis != "best" and score_variant != "metrics":
        return None, None

    entry = _load_entry(repo, exp, ds, leaf)
    block = entry["bases"].get(epoch_basis)
    if block is None:
        return None, None
    metrics = block.get("metrics") or {}
    epoch = block.get("epoch")
    ek = perf_basis.effective_key(metrics, base_key, threshold_basis)
    v = metrics.get(ek)
    return ((float(v) if isinstance(v, (int, float)) else None),
            (int(epoch) if isinstance(epoch, (int, float)) else None))


def build_all(repo) -> dict[str, Any]:
    """Iterate every COMPLETE leaf once, build + persist its entry. Idempotent."""
    built = 0
    leaves = 0
    errors: list[str] = []
    for e in repo.experiments():
        for d in e.datasets:
            for leaf in d.variants.values():
                if leaf is None or getattr(leaf, "state", None) != "complete":
                    continue
                leaves += 1
                try:
                    _load_entry(repo, e, d, leaf)
                    built += 1
                except Exception as exc:  # noqa: BLE001 — isolation; one bad leaf never aborts
                    errors.append(f"{leaf.leaf_path}: {exc}")
    return {"leaves": leaves, "built": built, "errors": errors}
