"""Typed data containers for the inference/evaluation pipeline.

This module exists to prevent the class of bug that produced the
2026-05-28 FM-omission incident: a key (``fm_patches``) silently
dropped while a patch-score dict was passed between modules. A typed
container with explicit fields removes the possibility of a missing
key going undetected.

PatchScoresBundle
=================
The single carrier of patch-level inference outputs. It is constructed
once after the GPU forward pass and passed unchanged through the
per-epoch eval path, the bg-worker final eval path, and the
ablation evaluation path. Every consumer reads the same fields,
including ``fm`` which is an explicit ``Optional[np.ndarray]`` field —
the previous untyped dict allowed callers to drop the key entirely.

Pickling
========
The bundle is sent across process boundaries by
``multiprocessing.spawn`` when the bg-worker is spawned. ``frozen=True``
does not affect pickle support, and ``numpy.ndarray`` fields pickle
without copying as long as the receiving process imports ``numpy``
before unpickling (which our bg-worker entrypoint guarantees). The
bundle therefore has zero IPC overhead beyond the raw array bytes,
which are the same bytes the previous dict-based path already moved.

Backward-compatible constructors
================================
Two classmethods are provided so the (already-correct) data flows in
``compute_epoch_test_inference`` and ``_cpu_eval_viz_worker`` can adopt
the bundle without restructuring upstream code:

- ``PatchScoresBundle.from_eval_data(eval_data: dict)``: builds from the
  dict returned by ``compute_epoch_test_inference``.
- ``PatchScoresBundle.from_patch_scores_dict(ps: dict)``: builds from
  the dict consumed by ``_bg_worker_body``. The ``fm`` field is the
  ``'fm'`` key of the dict (or ``None``); the patch_scores dict
  producer **must** include the key so the consumer cannot drop it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class PatchScoresBundle:
    """Container for patch-level inference outputs.

    Required fields:
    - ``recon`` : (n_windows, num_patches) teacher reconstruction error
    - ``disc`` : (n_windows, num_patches) output-level discrepancy
    - ``student_recon`` : (n_windows, num_patches) student reconstruction
    - ``fm`` : Optional. Hidden-level feature-matching distance with the
      same shape, or ``None`` when ``use_feature_matching`` was False (or
      the model did not produce it). This field is ``Optional[ndarray]``
      and **must be supplied explicitly** at construction; the producer
      cannot let it drop to the default by accident, which was the root
      cause of the 2026-05-28 FM-omission bug.
    - ``labels``, ``sample_types``, ``anomaly_types`` : (n_windows,)
      per-window metadata.

    Optional fields:
    - ``disc_per_feature`` : per-feature discrepancy (for diagnostic
      plots; not on the critical score path).
    - ``inference_time`` : wall-clock GPU forward pass time, for logging.
    """

    recon: np.ndarray
    disc: np.ndarray
    student_recon: np.ndarray
    fm: Optional[np.ndarray]
    labels: np.ndarray
    sample_types: np.ndarray
    anomaly_types: np.ndarray
    disc_per_feature: Optional[np.ndarray] = None
    inference_time: Optional[float] = None

    @classmethod
    def from_eval_data(cls, eval_data: Mapping[str, Any]) -> "PatchScoresBundle":
        """Build from the dict returned by ``compute_epoch_test_inference``.

        The producer already populates ``fm_patches`` (it reads
        ``getattr(evaluator, 'fm_patches', None)`` after the GPU forward
        pass), so this constructor is safe to use as the canonical
        ingestion point.
        """
        return cls(
            recon=eval_data['recon_patches'],
            disc=eval_data['disc_patches'],
            student_recon=eval_data['student_recon_patches'],
            fm=eval_data.get('fm_patches'),
            labels=eval_data['labels'],
            sample_types=eval_data['sample_types'],
            anomaly_types=eval_data['anomaly_types'],
            disc_per_feature=eval_data.get('disc_per_feature'),
            inference_time=eval_data.get('inference_time'),
        )

    @classmethod
    def from_patch_scores_dict(cls, ps: Mapping[str, Any]) -> "PatchScoresBundle":
        """Build from the dict consumed by ``_bg_worker_body``.

        The producer of this dict (``compute_epoch_test_metrics`` and the
        final-eval pipeline) **must** include the ``'fm'`` key (set to
        ``evaluator.fm_patches``); the consumer then reads it back here.
        """
        return cls(
            recon=ps['recon'],
            disc=ps['disc'],
            student_recon=ps['student_recon'],
            fm=ps.get('fm'),
            labels=ps['labels'],
            sample_types=ps['sample_types'],
            anomaly_types=ps['anomaly_types'],
            disc_per_feature=ps.get('disc_per_feature'),
        )
