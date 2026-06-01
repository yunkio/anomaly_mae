"""Boundary-safe TEST windowing helper (shared, single source of truth).

Problem: at test time every windowing baseline slides/chunks over the WHOLE concatenated
test array, so windows AT an entity boundary mix two entities (e.g. SMD machine-k tail +
machine-(k+1) head). Those cross-entity windows produce corrupt scores for the boundary-region
timesteps. The harness already prevents this for TRAINING (create_train_windows_boundary_safe /
train_segments); this module extends the same guarantee to TEST.

Mechanism (side-effect-minimal): run each model's windowing + inference INDEPENDENTLY on every
entity's own test slice, so no window can cross an entity boundary, and concatenate the resulting
per-timestep RAW scores. The model's SCORE POST-PROCESSING (e.g. QuoVadis median-IQR over the whole
test series, smoothing, max-over-features) is applied AFTER concatenation by the caller, exactly as
before — so the post-processing granularity is NOT changed by this fix.

Entity segments come from the SAME unified source as per-entity normalization
(`UnifiedLoader.get_file_norm_segments()` -> `data_info['entity_norm_segments']`), so windowing and
normalization stay consistent and dataset-setting-agnostic. Single-entity / single-file / None ->
ONE call over the whole array == EXACT legacy behaviour (bit-identical NO-OP).
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np

__all__ = ["entity_test_slices", "per_entity_concat", "is_multi_entity"]


def entity_test_slices(n_test: int, test_segments) -> List[Tuple[int, int]]:
    """Per-entity (lo, hi) half-open test-local slices.

    ``test_segments``: list of (lo, hi) in TEST-LOCAL coords (the test side of
    ``get_file_norm_segments()``). None / empty / a single entity -> ``[(0, n_test)]`` (whole array).
    """
    if not test_segments or len(test_segments) <= 1:
        return [(0, int(n_test))]
    slices = [(int(lo), int(hi)) for (lo, hi) in test_segments]
    # Defensive: if the segments do not tile [0, n_test) exactly, fall back to whole-array
    # (never silently mis-slice). Covers any loader inconsistency.
    if slices[0][0] != 0 or slices[-1][1] != int(n_test):
        return [(0, int(n_test))]
    for (a_lo, a_hi), (b_lo, b_hi) in zip(slices, slices[1:]):
        if a_hi != b_lo or a_hi <= a_lo:
            return [(0, int(n_test))]
    return slices


def is_multi_entity(n_test: int, test_segments) -> bool:
    """True iff test_segments describe >1 entity tiling [0, n_test) (i.e. per-entity is active)."""
    return len(entity_test_slices(n_test, test_segments)) > 1


def per_entity_concat(
    test_X: np.ndarray,
    test_segments,
    raw_score_fn: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Run ``raw_score_fn`` on each entity's test slice independently; concat to ``(n_test,)``.

    ``raw_score_fn(sub_X: (Li, F)) -> (Li,)`` must return the model's RAW per-timestep scores for a
    SINGLE entity (its own windowing + inference, NO cross-entity dependence, NO whole-test post-proc
    — apply that to the returned full array). Boundary-safe by construction: each entity is windowed
    in isolation, so no window spans a boundary.

    Single-entity / None -> exactly ``raw_score_fn(test_X)`` (bit-identical NO-OP).
    """
    test_X = np.asarray(test_X, dtype=np.float32)
    n = len(test_X)
    slices = entity_test_slices(n, test_segments)
    if len(slices) == 1:
        out = np.asarray(raw_score_fn(test_X), dtype=np.float32)
        if out.shape[0] != n:
            raise ValueError(f"raw_score_fn returned len {out.shape[0]} != n_test {n}")
        return out
    out = np.empty(n, dtype=np.float32)
    for (lo, hi) in slices:
        seg = np.asarray(raw_score_fn(test_X[lo:hi]), dtype=np.float32)
        if seg.shape[0] != (hi - lo):
            raise ValueError(
                f"raw_score_fn on entity slice [{lo}:{hi}] returned len {seg.shape[0]} != {hi - lo}"
            )
        out[lo:hi] = seg
    return out
