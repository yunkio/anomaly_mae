"""Single source of truth for anomaly score computation.

This module contains the canonical implementations of all anomaly score
formulas used across the codebase (training, evaluation, NPZ save,
visualization). All other modules MUST call these functions instead of
re-implementing the formulas inline.

Background
==========
Prior to 2026-05-29 the adaptive scoring formula was duplicated inline at
9 locations (evaluator, NPZ save, contribution chart, train-data scoring,
collect_predictions, derive_pred_data, three best_model_visualizer
helpers). Each location used slightly different inputs or epsilon
(1e-4 vs 1e-8), and the visualization paths silently dropped the
feature-matching (FM) term entirely. This caused the per-epoch saved
metrics and the NPZ-saved adaptive score to disagree, which in turn
caused the wrong best_epoch to be selected for `use_feature_matching=True`
experiments (root cause of the 2026-05-28 FM-omission bug).

Design
======
- ``ADAPTIVE_SCORE_EPSILON`` is the single shared numerical safeguard
  for the adaptive formula (1e-4, sufficient for typical disc/fm mean
  values of order 1e-3 to 1e-2; chosen over 1e-8 as the more conservative
  default used by the canonical evaluator path).
- ``resolve_score_weights`` is the single source for converting config
  fields (``eval_disc_weight``, ``eval_fm_weight``, ``fm_loss_weight``,
  ``use_output_discrepancy``, ``use_feature_matching``) into the
  effective scalar weights ``w_disc`` and ``w_fm`` and the FM-active
  flag. Accepts either a Config dataclass instance or a plain ``dict``
  (after ``dataclasses.asdict``) so bg-worker subprocesses that
  reconstruct config from a pickled dict can call the same function.
- ``compute_adaptive_components`` returns the unscaled and scaled
  components plus the combined score. Use this when the caller needs
  separate components (e.g., score contribution chart).
- ``compute_adaptive_point_score`` is a thin wrapper returning only the
  combined score.
- ``compute_default_score`` covers the ``anomaly_score_mode='default'``
  branch (``recon + lambda_disc * disc``).
- ``compute_ratio_weighted_score`` covers the ``ratio_weighted`` branch.
- ``compute_score`` dispatches to the correct branch based on
  ``config.anomaly_score_mode``.

Compatibility note
==================
The numerical output of the adaptive formula is identical (within
floating-point tolerance) to the previous canonical implementation in
``Evaluator._apply_scoring_formula`` for the FM-included path. For the
viz paths that previously dropped FM, the output now correctly includes
FM when ``use_feature_matching=True``, matching the NPZ-saved score.

Callers must not re-implement these formulas inline. If a new score
component is added (e.g., a third student signal), extend this module
only; the type signature and dispatch logic intentionally make it hard
to silently drop a component again.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Mapping, Optional

import numpy as np


# Single shared epsilon for adaptive score numerical safety.
# Chosen as 1e-4 (the value used in the canonical evaluator path) over
# the 1e-8 used in two viz paths. The difference is negligible for
# typical disc.mean() / fm.mean() values in the 1e-3 - 1e-2 range.
ADAPTIVE_SCORE_EPSILON: float = 1e-4


def _cfg_get(config: Any, key: str, default):
    """Read a config field from either a Config dataclass or a dict.

    bg-worker subprocesses receive ``asdict(config)`` and reconstruct a
    Config-like dict in some paths. This helper handles both.
    """
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def resolve_score_weights(config: Any) -> tuple[float, float, bool]:
    """Return effective ``(w_disc, w_fm, fm_active)`` for the adaptive
    formula.

    Rules (matching the canonical evaluator implementation):
    - ``w_disc = config.eval_disc_weight`` (negative → fallback to 1.0).
    - ``w_fm = config.eval_fm_weight`` (negative → fallback to
      ``config.fm_loss_weight`` (default 1.0)).
    - If ``config.use_output_discrepancy is False``, ``w_disc`` is forced
      to 0 (the discriminator never trained).
    - ``fm_active`` is ``True`` iff ``config.use_feature_matching`` is
      True. This flag is informational; the caller must additionally
      ensure ``fm`` array is non-None when including FM in the score.
    """
    w_disc = float(_cfg_get(config, 'eval_disc_weight', -1.0))
    w_fm = float(_cfg_get(config, 'eval_fm_weight', -1.0))
    if w_disc < 0:
        w_disc = 1.0
    if w_fm < 0:
        w_fm = float(_cfg_get(config, 'fm_loss_weight', 1.0))
    if not bool(_cfg_get(config, 'use_output_discrepancy', True)):
        w_disc = 0.0
    fm_active = bool(_cfg_get(config, 'use_feature_matching', False))
    return w_disc, w_fm, fm_active


def is_prewarmup_epoch(config: Any, epoch: Optional[int]) -> bool:
    """Return True iff ``epoch`` falls inside the teacher-only warmup window.

    Single source for the pre-warmup gate predicate. During the teacher-only
    warmup (``0 < epoch <= config.teacher_only_warmup_epochs``) the student
    decoder is frozen / random-initialised, so its output-discrepancy and
    feature-matching signals are noise that must NOT enter the anomaly score.
    Callers feed the boolean returned here into ``force_recon_only`` of the
    scoring functions so the score reduces to teacher reconstruction only.

    ``epoch`` is **1-indexed** (``ep = trained_epoch + 1``) — the same index
    used for ``epoch_metrics`` rows and the ``epoch_NNN_scores.npz`` filename.

    ``epoch=None`` (post-hoc / final viz with no defined epoch) → ``False``
    (legacy full scoring). A non-positive / missing ``teacher_only_warmup_epochs``
    (no warmup configured) → ``False`` for every epoch.

    This predicate is intentionally policy-only (no score math) so it can be
    shared by the evaluator, the npz-save path, the contribution chart, and
    the train-data scoring without re-implementing the window check.
    """
    if epoch is None:
        return False
    warmup = int(_cfg_get(config, 'teacher_only_warmup_epochs', -1))
    return warmup > 0 and int(epoch) <= warmup


def compute_adaptive_components(
    recon: np.ndarray,
    disc: np.ndarray,
    fm: Optional[np.ndarray],
    config: Any,
    *,
    force_recon_only: bool,
) -> dict:
    """Compute the adaptive score and its individual components.

    force_recon_only : bool (keyword-only, REQUIRED)
        When True, the student terms (output discrepancy + feature matching)
        are dropped so ``student_error == 0`` and ``score == recon`` (teacher
        reconstruction only). This is the pre-warmup gate: during the
        teacher-only warmup window (eval epoch <= teacher_only_warmup_epochs)
        the student decoder is frozen/random-init, so its discrepancy/FM are
        noise that must NOT enter the anomaly score. The warmup *policy*
        (which epochs are pre-warmup) is decided by the caller and passed as
        this boolean — the score core stays policy-free. Required keyword-only
        (not Optional=None) per the API-change checklist so a missed caller
        raises TypeError immediately rather than silently changing scores
        (2026-05-29 FM-omission class). force_recon_only=False == legacy
        behavior (post-warmup, byte-for-byte unchanged).

    Parameters
    ----------
    recon : np.ndarray
        Teacher reconstruction error. Any shape; the function uses only
        the array mean to compute the scaling factor and treats the
        array element-wise for the additive composition.
    disc : np.ndarray
        Output-level student-teacher discrepancy. Same shape as
        ``recon``.
    fm : np.ndarray or None
        Hidden-level student-teacher distance ("feature matching").
        Same shape as ``recon`` when provided. Pass ``None`` when FM was
        not computed for this run (i.e., the model did not produce it).
    config : Config dataclass or dict
        Source of ``eval_disc_weight``, ``eval_fm_weight``,
        ``fm_loss_weight``, ``use_output_discrepancy``,
        ``use_feature_matching``.

    Returns
    -------
    dict
        Keys:
        - ``score``: ``recon + student_error`` (same shape as ``recon``)
        - ``scaled_disc``: discrepancy term after recon-scale matching
        - ``scaled_fm``: FM term after recon-scale matching, or ``None``
        - ``student_error``: combined student contribution
        - ``recon_mean``: scalar used for scale matching
        - ``w_disc``, ``w_fm``, ``fm_active``: effective parameters

    Examples
    --------
    >>> import numpy as np
    >>> from mae_anomaly.scoring import compute_adaptive_components
    >>> class _Cfg:
    ...     eval_disc_weight = -1.0
    ...     eval_fm_weight = -1.0
    ...     fm_loss_weight = 1.0
    ...     use_output_discrepancy = True
    ...     use_feature_matching = True
    >>> recon = np.array([0.01, 0.02, 0.03])
    >>> disc  = np.array([0.001, 0.002, 0.003])
    >>> fm    = np.array([0.0005, 0.001, 0.0015])
    >>> out = compute_adaptive_components(recon, disc, fm, _Cfg(), force_recon_only=False)
    >>> bool(np.allclose(out['recon_mean'], recon.mean() + 1e-4))
    True
    >>> # force_recon_only=True drops disc+fm → score == recon exactly.
    >>> out_ro = compute_adaptive_components(recon, disc, fm, _Cfg(), force_recon_only=True)
    >>> bool(np.array_equal(out_ro['score'], recon))
    True
    >>> # scaled_disc preserves disc's variation but rescaled to recon's scale.
    >>> # Exact mean depends on disc.mean()+eps vs recon.mean()+eps ratio,
    >>> # so we only check shape and that scaling is bounded.
    >>> out['scaled_disc'].shape == disc.shape
    True
    >>> # score = recon + 0.5 * scaled_disc + 0.5 * scaled_fm  (w_disc=w_fm=1)
    >>> bool(np.allclose(out['score'],
    ...     recon + 0.5 * out['scaled_disc'] + 0.5 * out['scaled_fm']))
    True
    """
    eps = ADAPTIVE_SCORE_EPSILON
    w_disc, w_fm, fm_active = resolve_score_weights(config)

    # Pre-warmup gate: when the caller signals the eval epoch is within the
    # teacher-only warmup window, drop BOTH the discriminator and FM terms so
    # student_error == 0 and score == teacher recon (see the no-FM / w_disc<=0
    # branch below). Zeroing w_disc alone is insufficient — the FM branch would
    # still leak scaled_fm (2026-05-29 FM-omission class). Both must be off.
    if force_recon_only:
        w_disc = 0.0
        fm_active = False

    recon_mean = float(recon.mean()) + eps
    disc_mean = float(disc.mean()) + eps
    scaled_disc = disc * (recon_mean / disc_mean)

    # Include FM only when (a) the config enables FM AND (b) the caller
    # actually supplied an FM array. Either condition false → FM dropped
    # and student_error reduces to scaled_disc (or zero if w_disc==0).
    if fm_active and fm is not None:
        fm_mean = float(fm.mean()) + eps
        scaled_fm = fm * (recon_mean / fm_mean)
        denom = w_disc + w_fm
        if denom > 0:
            student_error = (w_disc * scaled_disc + w_fm * scaled_fm) / denom
        else:
            student_error = np.zeros_like(recon)
    else:
        scaled_fm = None
        if w_disc > 0:
            student_error = scaled_disc
        else:
            # Both OD disabled and FM unavailable → teacher recon only.
            student_error = np.zeros_like(recon)

    return {
        'score': recon + student_error,
        'scaled_disc': scaled_disc,
        'scaled_fm': scaled_fm,
        'student_error': student_error,
        'recon_mean': recon_mean,
        'w_disc': w_disc,
        'w_fm': w_fm,
        'fm_active': fm_active,
    }


def compute_adaptive_point_score(
    recon: np.ndarray,
    disc: np.ndarray,
    fm: Optional[np.ndarray],
    config: Any,
    *,
    force_recon_only: bool,
) -> np.ndarray:
    """Return only the combined adaptive score (thin wrapper).

    ``force_recon_only`` (keyword-only, REQUIRED): see
    ``compute_adaptive_components``. Forwarded verbatim.
    """
    return compute_adaptive_components(
        recon, disc, fm, config, force_recon_only=force_recon_only
    )['score']


def compute_default_score(
    recon: np.ndarray,
    disc: np.ndarray,
    config: Any,
) -> np.ndarray:
    """Default score: ``recon + lambda_disc * disc``."""
    lambda_disc = float(_cfg_get(config, 'lambda_disc', 0.5))
    return recon + lambda_disc * disc


def compute_ratio_weighted_score(
    recon: np.ndarray,
    disc: np.ndarray,
    config: Any,  # accepted for symmetry; unused
) -> np.ndarray:
    """Ratio-weighted score: ``recon * (1 + disc / disc_median)``."""
    eps = ADAPTIVE_SCORE_EPSILON
    disc_median = float(np.median(disc)) + eps
    return recon * (1 + disc / disc_median)


def compute_score(
    recon: np.ndarray,
    disc: np.ndarray,
    fm: Optional[np.ndarray],
    config: Any,
    *,
    force_recon_only: bool,
) -> np.ndarray:
    """Dispatch by ``config.anomaly_score_mode``.

    Modes:
    - ``'adaptive'``: full adaptive formula with optional FM.
    - ``'ratio_weighted'``: ``recon * (1 + disc / median(disc))``.
    - anything else: ``recon + lambda_disc * disc``.

    ``force_recon_only`` (keyword-only, REQUIRED): the pre-warmup gate; only
    the ``'adaptive'`` mode honors it (the other modes are not the
    student-augmented path this fix targets and ignore it).
    """
    mode = _cfg_get(config, 'anomaly_score_mode', 'default')
    if mode == 'adaptive':
        return compute_adaptive_point_score(
            recon, disc, fm, config, force_recon_only=force_recon_only
        )
    if mode == 'ratio_weighted':
        return compute_ratio_weighted_score(recon, disc, config)
    return compute_default_score(recon, disc, config)
