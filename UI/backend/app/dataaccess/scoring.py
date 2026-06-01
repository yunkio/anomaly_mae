"""Score-formula resolver — backend-design.md §4 (the §A6 order; never blank).

The applied recon:disc ratio (and the FM-in-score flag) is resolved in ONE place
(CLAUDE.md rule 3 spirit). Reading ``config.score_recon_disc_ratio`` alone would
render blank on every backfilled run (the field is absent/null there); this resolver
falls through to ``_lambda_recompute.ratio`` and finally the 4.0 code default.
"""
from __future__ import annotations

from typing import Any, Optional

from .models import ScoreFormula

_CODE_DEFAULT_RATIO = 4.0


def _parse_ratio_string(s: Any) -> Optional[float]:
    """Parse a provenance ratio like ``"4:1"`` -> 4.0. Returns None if unparseable."""
    if isinstance(s, (int, float)):
        return float(s)
    if not isinstance(s, str):
        return None
    txt = s.strip()
    if ":" in txt:
        head = txt.split(":", 1)[0].strip()
        try:
            return float(head)
        except ValueError:
            return None
    try:
        return float(txt)
    except ValueError:
        return None


def resolve_score_formula(metadata: dict[str, Any]) -> ScoreFormula:
    """Resolve the score formula from a parsed experiment_metadata dict (never blank).

    Order (registry ``_meta.score_ratio_resolution`` / data-schema §6):
      1. ``_lambda_recompute.ratio`` (parse "R:1" -> R) + ``.fm_in_score``
      2. else ``config.score_recon_disc_ratio`` when non-null
      3. else the 4.0 code default
    """
    config = metadata.get("config") or {}

    # 1 — provenance block (authoritative on backfilled runs).
    lam = metadata.get("_lambda_recompute")
    if isinstance(lam, dict):
        ratio = _parse_ratio_string(lam.get("ratio"))
        if ratio is not None:
            return ScoreFormula(
                ratio=ratio,
                fm_in_score=bool(lam.get("fm_in_score", False)),
                source="_lambda_recompute",
            )

    # 2 — config field, only when present AND non-null.
    cfg_ratio = config.get("score_recon_disc_ratio", None)
    if cfg_ratio is not None:
        try:
            return ScoreFormula(ratio=float(cfg_ratio), fm_in_score=False, source="config")
        except (TypeError, ValueError):
            pass

    # 3 — code default; still never blank.
    return ScoreFormula(ratio=_CODE_DEFAULT_RATIO, fm_in_score=False,
                        source="code_default(4.0)")
