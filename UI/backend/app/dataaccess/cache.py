"""mtime/content-aware cache under ./UI/.cache/ — backend-design.md §7.

Key = ``(absolute_path, st_mtime_ns, st_size)``. A live append changes mtime/size
=> the entry is recomputed (no TTL — mtime is truth). Writes go to a temp file +
atomic ``os.replace`` so a half-written cache file is never read. NEVER writes under
``results/`` or ``.trash/`` — the cache root is always inside ``./UI/``.

This is a lightweight in-process memo keyed on the file signature, with an optional
on-disk JSON sidecar for cross-restart reuse of cheap parsed bodies. Heavy columnar
caching (parquet) is layered on by callers that need it; the contract here is the
signature/invalidate discipline.
"""
from __future__ import annotations

import json
import os
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Optional

from ..config import SETTINGS


def file_signature(path: Path) -> Optional[tuple[str, int, int]]:
    """(abspath, st_mtime_ns, st_size) — None if the file is gone."""
    try:
        st = path.stat()
    except OSError:
        return None
    return (str(path.resolve()), st.st_mtime_ns, st.st_size)


class MtimeCache:
    """In-process signature-keyed memo with an optional on-disk JSON sidecar.

    Memory discipline (2026-06-16 leak fix): the in-process memo is BOUNDED.
    * **Per-path stale eviction** — a path is keyed by signature ``(abspath, mtime, size)``;
      when the same path is re-read with a NEW signature (a live training run rewriting the
      file), the prior entry for that path is dropped instead of lingering forever. This was
      the dominant leak with active training + the perf-basis last/epoch modes that read
      every leaf's (large) ``epoch_metrics`` series.
    * **LRU cap** — at most ``_MAX_ENTRIES`` entries; the least-recently-used is evicted.
    """

    # Bound on retained parsed bodies. Large enough to hold a full ranking pass over every
    # leaf (epoch series included) without thrashing, small enough to cap RSS.
    _MAX_ENTRIES = 4000

    def __init__(self, namespace: str):
        self._mem: "OrderedDict[tuple[str, int, int], Any]" = OrderedDict()
        self._by_path: dict[str, tuple[str, int, int]] = {}  # abspath -> current signature
        self._dir = SETTINGS.cache_dir(namespace)

    def _sidecar(self, sig: tuple[str, int, int]) -> Path:
        # filename = hash of (path, mtime, size); JSON sidecar.
        import hashlib

        h = hashlib.sha1(repr(sig).encode("utf-8")).hexdigest()[:16]
        return self._dir / f"{h}.json"

    def get_or_compute(
        self,
        path: Path,
        compute: Callable[[Path], Any],
        *,
        persist: bool = False,
    ) -> Any:
        """Return cached value for ``path`` (by signature) or compute + cache it.

        ``persist`` additionally mirrors the value to an on-disk JSON sidecar so a
        cheap parsed body survives a restart. Non-JSON-serializable values are kept
        in-process only.
        """
        sig = file_signature(path)
        if sig is None:
            return compute(path)  # file vanished mid-read — recompute (will raise/None)

        # per-path stale eviction: drop any prior (different-signature) entry for this path
        # so a rewritten file doesn't leave its old parse lingering forever.
        prev = self._by_path.get(sig[0])
        if prev is not None and prev != sig:
            self._mem.pop(prev, None)

        if sig in self._mem:
            self._mem.move_to_end(sig)            # LRU bump
            return self._mem[sig]
        if persist:
            sc = self._sidecar(sig)
            if sc.exists():
                try:
                    val = json.loads(sc.read_text("utf-8"))
                    self._store(sig, val)
                    return val
                except (OSError, json.JSONDecodeError):
                    pass
        val = compute(path)
        self._store(sig, val)
        if persist:
            self._atomic_write_json(self._sidecar(sig), val)
        return val

    def _store(self, sig: tuple[str, int, int], val: Any) -> None:
        """Insert + record the per-path signature, then evict LRU beyond the cap."""
        self._mem[sig] = val
        self._mem.move_to_end(sig)
        self._by_path[sig[0]] = sig
        while len(self._mem) > self._MAX_ENTRIES:
            old_sig, _ = self._mem.popitem(last=False)   # least-recently-used
            if self._by_path.get(old_sig[0]) == old_sig:
                self._by_path.pop(old_sig[0], None)

    @staticmethod
    def _atomic_write_json(target: Path, value: Any) -> None:
        try:
            payload = json.dumps(value, default=str).encode("utf-8")
        except (TypeError, ValueError):
            return  # non-serializable -> skip the sidecar, keep the in-memory entry
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(payload)
            os.replace(tmp, target)  # atomic
        except OSError:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def invalidate_all(self) -> None:
        self._mem.clear()
        self._by_path.clear()

    def stats(self) -> dict[str, Any]:
        return {"namespace": self._dir.name, "entries": len(self._mem), "cap": self._MAX_ENTRIES}
