"""Segment-aware windowing utilities (NormalOnly variant, 2026-05-25).

Standalone module to avoid the circular import that arises from
``comparison/baseline_common.py``: ``baseline_common`` already imports every
baseline wrapper (via ``comparison.baselines``), so any wrapper that needs the
helper cannot in turn import it back from ``baseline_common``.

Each SOTA wrapper imports ``compute_segment_safe_window_indices`` from this
module to drop sliding windows that would cross a normal-segment boundary
(either an anomaly-removal cut or a ``run_boundaries`` recording-join cut).
"""

from __future__ import annotations

import numpy as np


def compute_segment_safe_window_indices(
    segments,
    win_size: int,
    stride: int,
    n_total: int,
    extra_horizon: int = 0,
):
    """Return original-window indices that do not cross a segment boundary.

    The standard sliding-window dataset enumerates windows at positions
    ``s = i * stride`` for ``i in [0, n_total)``. This helper returns the
    subset of indices ``i`` whose covered range ``[s, s + win_size + extra_horizon]``
    fits entirely within a single segment.

    Args:
        segments: Iterable of ``(start, end)`` tuples (end exclusive), in the
            concatenated ``train_X`` coordinate (matches
            ``UnifiedLoader.normal_segments`` after ``_apply_normalonly``).
            Must describe segments of the SAME array the dataset is built on.
        win_size: Sliding-window length used by the dataset.
        stride: Stride used by the dataset (must match the
            ``(i * stride)`` → start mapping).
        n_total: Total windows in the original dataset (= ``len(dataset)``).
        extra_horizon: Additional steps past ``win_size`` that must stay within
            the same segment (e.g. ``1`` when the model also reads
            ``data[s + win_size]`` as a target — GDN, GCN-LSTM).

    Returns:
        ``np.ndarray[int64]`` of indices into ``[0, n_total)``, sorted ascending.
        Returns an empty array when ``segments`` is empty or no window fits.
    """
    if not segments or n_total <= 0:
        return np.array([], dtype=np.int64)

    span = win_size + extra_horizon
    valid = []
    for seg_start, seg_end in segments:
        if seg_end - seg_start < span:
            continue
        first_s = ((seg_start + stride - 1) // stride) * stride
        last_s = seg_end - span
        if first_s > last_s:
            continue
        for s in range(first_s, last_s + 1, stride):
            i = s // stride
            if 0 <= i < n_total:
                valid.append(i)
    return np.asarray(valid, dtype=np.int64)
