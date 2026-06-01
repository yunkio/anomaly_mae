"""Per-source-FILE, leak-free normalization kernel (shared, single source of truth).

For multi-FILE datasets (SMAP/MSL channels, SMD machines, Exathlon apps) the official
upstreams normalize EACH source file (entity) independently — fit a scaler on that file's
TRAIN portion, transform that file's TEST portion. Concatenating first and fitting ONE global
scaler (whole-array) blends entities of different scale and is a faithfulness deviation on
multi-file data. This module provides the ONE leak-free per-file fit/transform logic reused by
BOTH the harness normalization path (Path A) and the self-normalizing wrappers (Path B), so the
"fit on TRAIN slice → transform paired TEST slice" discipline is written exactly once and cannot
drift.

Single-file datasets (PSM/SWaT/WaDi/simulation) pass a single segment → the per-file loop reduces
to whole-array behavior, which is bit-identical to the legacy
``mae_anomaly.dataset_sliding._standardize_per_feature`` / ``_minmax_per_feature(clip=False)``
(verified by the unit self-test). So the change is a strict NO-OP on single-file data.

NOTHING imports this module until the harness/wrappers are wired to it; importing it has no side
effects.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

__all__ = [
    "fit_file_scaler",
    "apply_file_scaler",
    "normalize_concat_per_file",
    "fit_transform_train_per_file",
    "transform_test_per_file",
]


# --------------------------------------------------------------------------- #
# Path A primitives — array-level, used by the harness minmax/zscore branch.
# Deliberately reproduce the EXACT arithmetic of
#   mae_anomaly.dataset_sliding._standardize_per_feature            (zscore)
#   mae_anomaly.dataset_sliding._minmax_per_feature(clip=False, feature_range=(0,1))  (minmax)
# so a single-file run is bit-identical (NO-OP). See unit self-test.
# --------------------------------------------------------------------------- #
def fit_file_scaler(train_slice: np.ndarray, mode: str) -> dict:
    """Fit ONE scaler on a single file's TRAIN slice only (leak-free)."""
    # Stats in float64 (matches origin/main _standardize_per_feature: a float32
    # axis-0 reduction over near-constant features leaves the normalized train
    # mean off-0 by ~0.02; float64 removes that artifact). Stored as float32.
    if mode == "zscore":
        ts = train_slice.astype(np.float64)
        mean = ts.mean(axis=0)
        std = ts.std(axis=0)
        std = std.copy()
        std[std < 1e-8] = 1.0
        return {"mode": "zscore", "mean": mean.astype(np.float32), "std": std.astype(np.float32)}
    elif mode == "minmax":
        ts = train_slice.astype(np.float64)
        mn = ts.min(axis=0)
        mx = ts.max(axis=0)
        rng = mx - mn
        rng = rng.copy()
        rng[rng < 1e-8] = 1.0
        return {"mode": "minmax", "min": mn.astype(np.float32), "range": rng.astype(np.float32)}
    else:
        raise ValueError(f"unknown mode {mode!r} (expected 'zscore' or 'minmax')")


def apply_file_scaler(x: np.ndarray, st: dict) -> np.ndarray:
    """Transform with a fitted scaler dict. (minmax => feature_range (0,1), clip=False)."""
    if st["mode"] == "zscore":
        return ((x - st["mean"]) / st["std"]).astype(np.float32)
    return ((x - st["min"]) / st["range"]).astype(np.float32)


def normalize_concat_per_file(
    features_raw: np.ndarray,
    train_end: int,
    file_train_segs: List[Tuple[int, int]],
    file_test_segs: List[Tuple[int, int]],
    mode: str,
) -> np.ndarray:
    """Per-file leak-free normalization of a ``[ALL-train | ALL-test]`` concat array (Path A).

    Args:
        features_raw: (total_length, F) raw concat array.
        train_end: index separating the train block [0, train_end) from test [train_end, N).
        file_train_segs: per-file (lo, hi) in TRAIN coords (subset of [0, train_end)).
        file_test_segs:  per-file (lo, hi) in TEST-LOCAL coords (subset of [0, N - train_end)).
                         The i-th train file pairs with the i-th test file.
        mode: 'zscore' | 'minmax'.

    Returns a NEW float32 array (copy); each file's train slice fits its own scaler, and that
    scaler transforms that file's train AND paired test slice. Test never influences any stat.
    """
    if len(file_train_segs) != len(file_test_segs):
        raise ValueError(
            f"file segment count mismatch: {len(file_train_segs)} train vs "
            f"{len(file_test_segs)} test (entity order must pair 1:1)"
        )
    out = features_raw.astype(np.float32).copy()
    for (tlo, thi), (slo, shi) in zip(file_train_segs, file_test_segs):
        st = fit_file_scaler(out[tlo:thi], mode)  # FIT on train slice ONLY
        out[tlo:thi] = apply_file_scaler(out[tlo:thi], st)  # transform train
        out[train_end + slo:train_end + shi] = apply_file_scaler(
            out[train_end + slo:train_end + shi], st
        )  # transform paired test
    return out


# --------------------------------------------------------------------------- #
# Path B helpers — scaler-pluggable, used by the self-normalizing wrappers.
# Each wrapper keeps its OWN scaler type (its scaler identity is part of its faithfulness):
#   npsr -> MinMaxScaler((-1,1)) (+clamp after); AT/wetas/treemil/deepmil/nrdetector -> StandardScaler.
# These helpers only own the per-file segment looping + leak-free train/test pairing.
# --------------------------------------------------------------------------- #
def fit_transform_train_per_file(train_X: np.ndarray, train_segs, make_scaler):
    """Fit one ``make_scaler()`` per train file; return (train_norm, scalers list, segs used).

    ``train_segs``: per-file (lo, hi) in TRAIN-LOCAL [0, len(train_X)) coords.
    None / [] -> a single whole-array "file" (single-file behavior).
    ``make_scaler``: zero-arg callable returning a fresh sklearn-like scaler
    (has ``fit_transform`` and ``transform``).
    """
    segs = list(train_segs) if train_segs else [(0, len(train_X))]
    train_X = np.asarray(train_X, dtype=np.float32)
    out = np.empty_like(train_X, dtype=np.float32)
    scalers = []
    for (lo, hi) in segs:
        sc = make_scaler()
        out[lo:hi] = np.asarray(sc.fit_transform(train_X[lo:hi]), dtype=np.float32)
        scalers.append(sc)
    return out, scalers, segs


def transform_test_per_file(test_X: np.ndarray, test_segs, scalers) -> np.ndarray:
    """LEAK-FREE test transform: i-th test file transformed by i-th cached TRAIN scaler.

    ``test_segs``: per-file (lo, hi) in TEST-LOCAL coords. None / [] -> single-file path.
    If the segment count does not match the cached train scalers (single-file, unsegmented, or
    any mismatch), fall back to the FIRST train scaler over the whole test array — still
    leak-free (never fits on test). NEVER calls .fit / .fit_transform here.
    """
    test_X = np.asarray(test_X, dtype=np.float32)
    if not scalers:
        raise ValueError("transform_test_per_file: no train scalers cached (call fit first)")
    segs = list(test_segs) if test_segs else [(0, len(test_X))]
    if len(segs) != len(scalers):
        return np.asarray(scalers[0].transform(test_X), dtype=np.float32)
    out = np.empty_like(test_X, dtype=np.float32)
    for (lo, hi), sc in zip(segs, scalers):
        out[lo:hi] = np.asarray(sc.transform(test_X[lo:hi]), dtype=np.float32)
    return out
