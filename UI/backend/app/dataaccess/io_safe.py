"""Tolerant, read-only IO primitives.

backend-design.md §9 (robustness): tolerant JSON (NaN/Infinity -> null), JSONL
line-by-line skipping a truncated last line, path confinement, never opening *.pt.

Every reader here opens the file READ-ONLY. Nothing in this module writes under a
source root. ``orjson`` is preferred (fast + rejects NaN by erroring, which we
catch); we always fall back to the stdlib ``json`` with a sanitizing parser so a
``NaN``/``Infinity`` literal the writer's ``default=float`` can emit becomes JSON
``null`` rather than crashing the API.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

try:  # orjson is in requirements; degrade gracefully if absent (e.g. partial venv)
    import orjson  # type: ignore

    _HAVE_ORJSON = True
except Exception:  # pragma: no cover - exercised only without the dep
    orjson = None  # type: ignore
    _HAVE_ORJSON = False


class CorruptOrWriting(Exception):
    """A file appears half-written / corrupt (BadZipFile, truncated JSON, etc.)."""


def sanitize(obj: Any) -> Any:
    """Recursively replace NaN/Inf floats with None so the wire payload is valid JSON.

    Applied on the way OUT of the parser and again before serialization. Lists and
    dicts are walked; everything else is returned as-is.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    return obj


def read_json(path: Path) -> Any:
    """Parse a JSON file tolerantly. NaN/Infinity -> None. Raises CorruptOrWriting
    on a structurally broken / truncated body (so a per-leaf wrapper can isolate it).
    """
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        raise
    except OSError as exc:  # pragma: no cover - rare fs race
        raise CorruptOrWriting(f"read failed: {path}: {exc}") from exc

    if _HAVE_ORJSON:
        try:
            return sanitize(orjson.loads(raw))
        except orjson.JSONDecodeError:
            # orjson rejects NaN/Infinity; retry with the stdlib tolerant parser.
            pass
    try:
        # stdlib json accepts NaN/Infinity by default; sanitize() then nulls them.
        return sanitize(json.loads(raw.decode("utf-8")))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CorruptOrWriting(f"invalid/truncated JSON: {path}: {exc}") from exc


def read_jsonl(path: Path) -> list[dict]:
    """Parse a JSONL file line-by-line, UTF-8, SKIPPING a truncated final line.

    monitoring_log.jsonl is appended live; the last line may be half-written.
    Korean (UTF-8) verdict strings are preserved. Never raises on a bad line.
    """
    out: list[dict] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        raise
    except OSError as exc:  # pragma: no cover
        raise CorruptOrWriting(f"read failed: {path}: {exc}") from exc
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            # truncated / half-written trailing line — skip, don't crash.
            continue
        if isinstance(obj, dict):
            out.append(sanitize(obj))
    return out


def confine(path: Path, *roots: Path) -> Path:
    """Resolve ``path`` and assert it lives under one of ``roots`` (no `..` escape).

    Used by the static-file serve so a crafted relpath cannot read outside the
    source roots. Raises ValueError on an escape attempt.
    """
    rp = path.resolve()
    for root in roots:
        try:
            rp.relative_to(root.resolve())
            return rp
        except ValueError:
            continue
    raise ValueError(f"path {rp} escapes all allowed roots {[str(r) for r in roots]}")
