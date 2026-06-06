"""Lazy NPZ score-slab reader — backend-design.md §5.8 / §6.4.

Discipline:
  * ``np.load(mmap_mode='r')`` — never eager-load whole arrays into RAM.
  * key off ``z.files`` — never assume an array (``fm_error`` is conditional).
  * one NPZ open at a time (the caller iterates; we never hold >1 handle).
  * BadZipFile / EOFError / OSError on a half-written live file -> CorruptOrWriting
    (the caller turns it into ``{"error": "corrupt_or_writing"}`` and skips).
  * NEVER opens *.pt — there is no code path here that touches a .pt body.
"""
from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Optional

try:
    import numpy as np  # type: ignore

    _HAVE_NUMPY = True
except Exception:  # pragma: no cover
    np = None  # type: ignore
    _HAVE_NUMPY = False

from .io_safe import CorruptOrWriting


def list_arrays(npz_path: Path) -> list[str]:
    """Return the array names present in an NPZ WITHOUT loading any array body.

    Reads only the zip central directory (``z.namelist``), strips the ``.npy``
    suffix. Tolerates a half-written file by raising CorruptOrWriting.
    """
    try:
        with zipfile.ZipFile(npz_path, "r") as z:
            return [n[:-4] if n.endswith(".npy") else n for n in z.namelist()]
    except (zipfile.BadZipFile, EOFError, OSError) as exc:
        raise CorruptOrWriting(f"NPZ corrupt/writing: {npz_path}: {exc}") from exc


def _downsample_idx(length: int, target: int) -> "Optional[np.ndarray]":
    """Stride-based downsample index to <= target points (None => keep all)."""
    if target <= 0 or length <= target:
        return None
    stride = max(1, length // target)
    return np.arange(0, length, stride)


def read_arrays(
    npz_path: Path,
    arrays: Optional[list[str]] = None,
    *,
    downsample: int = 0,
    page: Optional[int] = None,
    page_size: Optional[int] = None,
) -> dict:
    """Lazily read selected arrays from one NPZ. Returns a JSON-ready dict.

    Only ``arrays`` that actually exist (per ``z.files``) are read. Downsampling /
    paging bound the wire payload. Memory peak ~ one array at a time.
    """
    if not _HAVE_NUMPY:  # pragma: no cover - partial venv only
        raise RuntimeError("numpy unavailable in this venv")

    present = set(list_arrays(npz_path))  # may raise CorruptOrWriting
    want = [a for a in (arrays or list(present)) if a in present]

    try:
        with np.load(npz_path, mmap_mode="r", allow_pickle=False) as z:
            out_arrays: dict[str, list] = {}
            length = 0
            downsampled = False
            idx = None
            for name in want:
                arr = z[name]               # mmap view, not a copy
                length = int(arr.shape[0]) if arr.ndim >= 1 else 1
                if page is not None and page_size:
                    start = page * page_size
                    sl = arr[start: start + page_size]
                    out_arrays[name] = np.asarray(sl).tolist()
                    continue
                if downsample:
                    if idx is None:
                        idx = _downsample_idx(length, downsample)
                    if idx is not None:
                        out_arrays[name] = np.asarray(arr[idx]).tolist()
                        downsampled = True
                        continue
                out_arrays[name] = np.asarray(arr).tolist()
    except (zipfile.BadZipFile, EOFError, OSError) as exc:
        raise CorruptOrWriting(f"NPZ corrupt/writing: {npz_path}: {exc}") from exc

    result = {
        "length": length,
        "arrays": out_arrays,
        "available_arrays": sorted(present),
        "downsampled": downsampled,
    }
    # P4B-04 / VERIF-FB1: echo the effective paging back so <NpzScrubber> can render
    # page controls without re-deriving its paging state. ``page``/``page_size`` are the
    # requested values; when paging is active also surface the effective slice bounds.
    result["page"] = page
    result["page_size"] = page_size
    if page is not None and page_size:
        start = page * page_size
        result["page_start"] = start
        result["page_end"] = min(start + page_size, length)
    return result


def histogram(
    npz_path: Path,
    array: str = "adaptive_score",
    *,
    bins: int = 80,
    label_array: str = "point_labels",
) -> dict:
    """Compute a normal-vs-anomaly histogram of one array, split by point_labels.

    Loads one NPZ, one array at a time; never holds the full set. Tolerant.
    """
    if not _HAVE_NUMPY:  # pragma: no cover
        raise RuntimeError("numpy unavailable in this venv")
    present = set(list_arrays(npz_path))
    if array not in present:
        return {"available": False, "reason": f"array {array!r} absent",
                "available_arrays": sorted(present)}
    try:
        with np.load(npz_path, mmap_mode="r", allow_pickle=False) as z:
            scores = np.asarray(z[array], dtype="float64")
            labels = (np.asarray(z[label_array]) if label_array in present
                      else np.zeros_like(scores, dtype="int8"))
    except (zipfile.BadZipFile, EOFError, OSError) as exc:
        raise CorruptOrWriting(f"NPZ corrupt/writing: {npz_path}: {exc}") from exc

    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return {"available": False, "reason": "no finite values"}
    lo, hi = float(finite.min()), float(finite.max())
    if lo == hi:
        hi = lo + 1e-9
    edges = np.linspace(lo, hi, bins + 1)
    normal = scores[(labels == 0) & np.isfinite(scores)]
    anomaly = scores[(labels == 1) & np.isfinite(scores)]
    n_hist, _ = np.histogram(normal, bins=edges)
    a_hist, _ = np.histogram(anomaly, bins=edges)

    snr = None
    if normal.size and anomaly.size:
        sd = normal.std() + anomaly.std()
        if sd > 0:
            snr = float((anomaly.mean() - normal.mean()) / sd)
    return {
        "available": True,
        "bins": edges.tolist(),
        "normal_hist": n_hist.tolist(),
        "anomaly_hist": a_hist.tolist(),
        "snr": snr,
        "n_normal": int(normal.size),
        "n_anomaly": int(anomaly.size),
    }
