"""Repository — the single read-only facade over the data-access layer.

Ties together: registry (+ self-test gate), discovery scan (cached), lazy per-leaf
parsing (cached by file signature), dynamic catalog building, and the score-formula
resolver. Every API endpoint goes through this object.

backend-design.md §2 / §3 / §4 / §7.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Optional

from ..config import SETTINGS
from .cache import MtimeCache
from .catalog import CatalogEntry, build_catalog, global_catalog
from .discovery import scan
from .io_safe import CorruptOrWriting
from .models import (
    BestSnapshot,
    DatasetRef,
    EpochSeries,
    ExperimentRef,
    LeafState,
    TrainingHistory,
)
from .parsers import (
    parse_anomaly_type_metrics,
    parse_best_config,
    parse_epoch_metrics,
    parse_experiment_metadata,
    parse_training_histories,
)
from .registry import MetricRegistry

# Scan-cache TTL backstop (seconds). The stat-signature fast-path catches new
# experiments / dataset groups instantly; this TTL bounds how long an in-leaf
# state transition (in_progress→complete, which doesn't bump a parent mtime) can
# stay stale. Small enough to feel live, large enough that the full scan never
# runs more than once per this interval under rapid frontend polling.
_SCAN_TTL_SECONDS = 10.0


class LeafBundle:
    """Parsed, normalized payloads for one leaf (the unit a panel/endpoint consumes).

    Inherits read-only/lazy/tolerant access by construction: it holds no writable
    filesystem handle and has no code path that opens a ``.pt`` (EXT-6 guarantee).
    """

    def __init__(self, exp: ExperimentRef, dataset: DatasetRef, leaf: LeafState):
        self.exp = exp
        self.dataset = dataset
        self.leaf = leaf
        self.path: Path = leaf.leaf_path
        self.errors: list[str] = list(leaf.errors)
        self.metadata: Optional[BestSnapshot] = None
        self.epoch: Optional[EpochSeries] = None
        self.history: Optional[TrainingHistory] = None
        self.anomaly_types: Optional[dict[str, dict]] = None
        self.config: dict[str, Any] = {}


class Repository:
    def __init__(self):
        self.registry = MetricRegistry.load(SETTINGS.registry_path)
        self._selftest = self.registry.run_self_test()
        self._scan_cache: Optional[list[ExperimentRef]] = None
        self._scan_sig: tuple = ()
        self._scan_built_at: float = 0.0
        self._parsed = MtimeCache("parsed")

    # ── startup gate ─────────────────────────────────────────────────────────
    @property
    def selftest(self) -> dict[str, Any]:
        return self._selftest

    def rerun_selftest(self) -> dict[str, Any]:
        self._selftest = self.registry.run_self_test()
        return self._selftest

    # ── discovery ────────────────────────────────────────────────────────────
    def experiments(self, *, force: bool = False) -> list[ExperimentRef]:
        # Auto-invalidation (2026-06-03 "288 invisible" fix). A live training run
        # writes new experiment dirs / dataset leaves into the source roots AFTER
        # the server starts; the scan cache must not freeze for the process
        # lifetime. Rescan when (a) a cheap stat-signature of the source roots +
        # their experiment children changes — new experiment or new dataset group
        # — or (b) a short TTL lapses, which catches in-leaf state transitions
        # (in_progress→complete) that don't bump a parent dir's mtime. The
        # signature is stat-only; no file bodies are read.
        now = time.monotonic()
        if not force and self._scan_cache is not None:
            fresh = (now - self._scan_built_at) < _SCAN_TTL_SECONDS
            if fresh and self._source_signature() == self._scan_sig:
                return self._scan_cache
        self._scan_cache = scan(SETTINGS.source_roots)
        self._scan_built_at = now
        self._scan_sig = self._source_signature()
        return self._scan_cache

    def _source_signature(self) -> tuple:
        """Cheap stat-only fingerprint of the source roots + their immediate
        experiment-dir children. Changes when an experiment dir or a dataset group
        is added/removed (the containing dir's mtime bumps), so a live run is
        detected without a full rescan on every request. ~one scandir + a stat per
        experiment dir across the 3 roots (tens of syscalls) vs. thousands for a
        full scan — so this is the fast path; scan() only runs on a real change."""
        sig: list[tuple[str, int]] = []
        for root in SETTINGS.source_roots.values():
            try:
                sig.append((str(root), root.stat().st_mtime_ns))
            except OSError:
                sig.append((str(root), -1))
                continue
            try:
                for e in os.scandir(root):
                    if e.is_dir():
                        try:
                            sig.append((e.path, e.stat().st_mtime_ns))
                        except OSError:
                            sig.append((e.path, -1))
            except OSError:
                pass
        return tuple(sig)

    def rescan(self) -> dict[str, Any]:
        before = {e.exp_id for e in (self._scan_cache or [])}
        self._scan_cache = scan(SETTINGS.source_roots)
        self._scan_built_at = time.monotonic()
        self._scan_sig = self._source_signature()
        after = {e.exp_id for e in self._scan_cache}
        self._parsed.invalidate_all()
        return {
            "experiments_found": len(self._scan_cache),
            "added": sorted(after - before),
            "removed": sorted(before - after),
        }

    def find_experiment(self, exp_id: str) -> Optional[ExperimentRef]:
        for e in self.experiments():
            if e.exp_id == exp_id:
                return e
        return None

    def find_dataset(self, exp_id: str, dataset_key: str) -> tuple[Optional[ExperimentRef], Optional[DatasetRef]]:
        exp = self.find_experiment(exp_id)
        if exp is None:
            return None, None
        # dataset keys may contain "/" (e.g. "WaDi/A1"); the API encodes those as "~"
        # to keep URL path segments unambiguous. Match either form.
        wanted = {dataset_key, dataset_key.replace("~", "/")}
        for d in exp.datasets:
            if d.dataset_key in wanted or d.dataset_key.replace("/", "~") in wanted:
                return exp, d
        return exp, None

    def resolve_leaf(
        self, exp_id: str, dataset_key: str, variant: Optional[str]
    ) -> tuple[Optional[ExperimentRef], Optional[DatasetRef], Optional[LeafState]]:
        exp, ds = self.find_dataset(exp_id, dataset_key)
        if ds is None:
            return exp, ds, None
        if variant and variant in ds.variants:
            return exp, ds, ds.variants[variant]
        # default: prefer a complete variant, else any, preferring "full"/"_".
        for pref in ("full", "_"):
            if pref in ds.variants:
                return exp, ds, ds.variants[pref]
        # first available
        first = next(iter(ds.variants.values()), None)
        return exp, ds, first

    # ── lazy per-leaf parsing (cached by file signature) ─────────────────────
    def _cached_parse(self, path: Path, fn) -> Any:
        if not path.exists():
            return None
        try:
            return self._parsed.get_or_compute(path, fn)
        except CorruptOrWriting:
            raise

    def load_bundle(
        self, exp: ExperimentRef, dataset: DatasetRef, leaf: LeafState,
        *, want: Optional[set[str]] = None,
    ) -> LeafBundle:
        """Parse the requested file-kinds for a leaf, isolating per-file failures.

        ``want`` limits which kinds are parsed (cheap when an endpoint needs one).
        A corrupt/half-written file attaches to ``bundle.errors`` and the field stays
        None — never raises out of here.
        """
        want = want or {"metadata", "epoch", "history", "anomaly_types", "config"}
        b = LeafBundle(exp, dataset, leaf)
        p = leaf.leaf_path

        warmup = None
        if "metadata" in want or "config" in want or "epoch" in want:
            try:
                meta = self._cached_parse(p / "experiment_metadata.json", parse_experiment_metadata)
                if meta is not None:
                    b.metadata = meta
                    b.config = meta.config
                    warmup = meta.config.get("teacher_only_warmup_epochs")
                    leaf.warmup_epochs = warmup if isinstance(warmup, int) else leaf.warmup_epochs
                    nf = meta.config.get("num_features")
                    leaf.num_features = nf if isinstance(nf, int) else leaf.num_features
            except CorruptOrWriting as exc:
                b.errors.append(f"experiment_metadata.json: {exc}")
        if "config" in want and not b.config:
            try:
                cfg = self._cached_parse(p / "best_config.json", parse_best_config)
                if cfg:
                    b.config = cfg
                    warmup = cfg.get("teacher_only_warmup_epochs", warmup)
            except CorruptOrWriting as exc:
                b.errors.append(f"best_config.json: {exc}")
        if "epoch" in want:
            try:
                epoch = self._cached_parse(
                    p / "epoch_metrics.json",
                    lambda pp: parse_epoch_metrics(pp, warmup),
                )
                if epoch is not None:
                    epoch.warmup_epochs = warmup if isinstance(warmup, int) else epoch.warmup_epochs
                    leaf.eval_interval = epoch.eval_interval or leaf.eval_interval
                    b.epoch = epoch
            except CorruptOrWriting as exc:
                b.errors.append(f"epoch_metrics.json: {exc}")
        if "history" in want:
            try:
                b.history = self._cached_parse(p / "training_histories.json", parse_training_histories)
            except CorruptOrWriting as exc:
                b.errors.append(f"training_histories.json: {exc}")
        if "anomaly_types" in want:
            try:
                b.anomaly_types = self._cached_parse(p / "anomaly_type_metrics.json", parse_anomaly_type_metrics)
            except CorruptOrWriting as exc:
                b.errors.append(f"anomaly_type_metrics.json: {exc}")
        return b

    # ── catalog ──────────────────────────────────────────────────────────────
    @staticmethod
    def _leaf_variant_scope(leaf: LeafState) -> str:
        """Map a leaf's variant to the M-1 catalog scope (full/excl22/default).

        The leaf variant comes from discovery (``A1A2_full`` -> "full",
        ``A1A2_excl22`` -> "excl22", everything else -> "_"). Only "full"/"excl22"
        carry a SWaT eval scope; "_" (non-SWaT, or unscoped) maps to "default" so the
        legacy behaviour is byte-identical for non-SWaT leaves."""
        v = getattr(leaf, "variant", "_")
        return v if v in ("full", "excl22") else "default"

    def leaf_catalog(self, bundle: LeafBundle) -> list[CatalogEntry]:
        return build_catalog(
            self.registry,
            epoch=bundle.epoch,
            metadata=bundle.metadata,
            history=bundle.history,
            anomaly_types=bundle.anomaly_types,
            leaf_variant_scope=self._leaf_variant_scope(bundle.leaf),
        )

    def global_catalog(self) -> list[CatalogEntry]:
        cats: list[list[CatalogEntry]] = []
        for exp in self.experiments():
            for ds in exp.datasets:
                # Iterate ALL variants (M-1/FB-6): on SWaT the full leaf carries the
                # excl22-prefixed keys (tagged variant_scope="excl22") AND its plain
                # keys ("full"), and the excl22 leaf carries its own plain keys
                # ("excl22"). Visiting every variant ensures the global union surfaces
                # the proper variant-scoped entries (ADDITIVE — never drops anything).
                for leaf in ds.variants.values():
                    if leaf is None or leaf.state == "missing":
                        continue
                    try:
                        b = self.load_bundle(exp, ds, leaf)
                        cats.append(self.leaf_catalog(b))
                    except Exception:  # noqa: BLE001 - isolation
                        continue
        return global_catalog(cats)

    # ── health ───────────────────────────────────────────────────────────────
    def health(self) -> dict[str, Any]:
        return {
            "status": "ok" if self._selftest["status"] == "PASS" else "degraded",
            "resolver_selftest": self._selftest["status"],
            "selftest_detail": self._selftest,
            "registry_version": self.registry.version,
            "cache": self._parsed.stats(),
            "roots": {k: str(v) for k, v in SETTINGS.source_roots.items()},
        }


# module-level singleton (lazy)
_REPO: Optional[Repository] = None


def get_repository() -> Repository:
    global _REPO
    if _REPO is None:
        _REPO = Repository()
    return _REPO
