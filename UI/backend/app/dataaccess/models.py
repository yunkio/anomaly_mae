"""Normalized internal data model (pydantic v2) — backend-design.md §2.3.

Every consumer sees these models, never raw dicts. ALL metric-bearing fields are
OPEN maps (``dict[str, float | None]`` / ``dict[str, list]``) so a renamed / added /
removed metric key flows through with ZERO code change (F5). Missing files become
``None`` fields, never exceptions (G4 graceful degradation).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

# Direction / family meta carried per metric key (resolver output, §3.2).
Direction = Literal["up", "down", "neutral"]


class MetricMeta(BaseModel):
    """Resolved semantics for one metric key (registry-entry-first, else fallback)."""

    key: str
    display_name: str
    family: str
    direction: Direction
    phase_validity: Literal["all", "post-warmup"] = "all"
    unit: str = "mixed"
    description: str = ""
    viz_hint: dict[str, Any] = Field(default_factory=dict)
    namespace: str = "epoch_metrics"
    inferred: bool = False
    inferred_direction: bool = False
    rule_id: Optional[str] = None  # which fallback rule fired (None if explicit)
    # M-1 (R2-01/FB-6): the SWaT eval-variant this catalog entry belongs to. ADDITIVE,
    # default "default" preserves behaviour (CLAUDE.md API-rule 2 — defaulted-but-
    # explicit; no caller branches on its absence). Only the catalog build sets it to
    # "full"/"excl22"; everything else keeps "default". Carries the variant dimension
    # for the SWaT excl22 strip so full vs excl22 of the SAME metric never collide
    # under one (display_label, variant_scope) identity. The raw ``.key`` is preserved
    # (F5) — variant_scope is a tag, not a rename.
    variant_scope: Literal["full", "excl22", "default"] = "default"


class PresenceFlags(BaseModel):
    """Which file-kinds exist in a leaf (bools). Drives missing-file degradation."""

    epoch_metrics: bool = False
    experiment_metadata: bool = False
    best_config: bool = False
    training_histories: bool = False
    anomaly_type_metrics: bool = False
    batch_profiling: bool = False
    best_model_detailed: bool = False
    best_epoch_train_scores: bool = False
    epoch_scores_dir: bool = False
    checkpoints_dir: bool = False
    visualization_dir: bool = False
    summary: bool = False
    monitoring_log: bool = False
    telemetry: bool = False


class ScoreFormula(BaseModel):
    """Resolved anomaly-score formula (§4 — never blank)."""

    ratio: float
    fm_in_score: bool
    source: str  # "_lambda_recompute" | "config" | "code_default(4.0)"
    formula_text: str = "score = teacher_recon + scaled_disc / ratio"


class LeafState(BaseModel):
    """A single eval leaf (one dataset, one variant)."""

    variant: str                       # "full" | "excl22" | "_"
    leaf_path: Path
    depth: int
    present: PresenceFlags
    state: Literal["complete", "in_progress", "missing"] = "missing"
    warmup_epochs: Optional[int] = None     # config.teacher_only_warmup_epochs
    num_features: Optional[int] = None
    eval_interval: Optional[int] = None
    errors: list[str] = Field(default_factory=list)


class DatasetRef(BaseModel):
    """Logical dataset (collapses SWaT full/excl22 into one with a variant selector)."""

    dataset_key: str                   # "PSM", "SWaT_A1A2", "SMD/concat", ...
    variants: dict[str, LeafState]     # {"full":.., "excl22":..} or {"_":..}
    state: Literal["complete", "in_progress", "missing"] = "missing"


class SummaryRollup(BaseModel):
    total_time: Optional[float] = None
    config_set: Optional[str] = None
    description: Optional[str] = None
    timestamp: Optional[str] = None
    results: list[dict[str, Any]] = Field(default_factory=list)  # dir resolved §3.9


class ExperimentRef(BaseModel):
    """Top-level experiment (one ``<num>_<TS>_<suffix>`` dir)."""

    exp_id: str
    exp_num: Optional[int] = None
    timestamp: Optional[str] = None
    suffix: Optional[str] = None
    root: Path
    source: Literal["live", "fixture", "old"]  # "old": archived runs (FB-R4d-01)
    state: Literal["complete", "in_progress", "early_abort"] = "early_abort"
    datasets: list[DatasetRef] = Field(default_factory=list)
    summary: Optional[SummaryRollup] = None
    errors: list[str] = Field(default_factory=list)


# ── per-file parsed payloads (open maps) ─────────────────────────────────────
class EpochSeries(BaseModel):
    """epoch_metrics.json — per-EVAL-epoch axis. UNION over all entries' keys."""

    axis: Literal["eval-epoch"] = "eval-epoch"
    eval_interval: Optional[int] = None
    epochs: list[int] = Field(default_factory=list)
    series: dict[str, list[Optional[float]]] = Field(default_factory=dict)  # key->aligned
    array_keys: list[str] = Field(default_factory=list)   # keys whose values are arrays
    warmup_epochs: Optional[int] = None


class NestedPivot(BaseModel):
    """Pivoted, null-padded nested history (e.g. epoch_anomaly_type_scores). §5.3.2."""

    types: list[str] = Field(default_factory=list)
    sub_metrics: list[str] = Field(default_factory=list)
    pivot: dict[str, dict[str, list[Optional[float]]]] = Field(default_factory=dict)


class TrainingHistory(BaseModel):
    """training_histories['0'] — per-TRAINING-epoch axis (P2-01). SEPARATE axis."""

    axis: Literal["training-epoch"] = "training-epoch"
    train_epochs: list[int] = Field(default_factory=list)
    series: dict[str, list[Optional[float]]] = Field(default_factory=dict)  # 61 flat lists union
    nested: dict[str, NestedPivot] = Field(default_factory=dict)
    empty_keys: list[str] = Field(default_factory=list)  # disabled components (len 0)


class BestSnapshot(BaseModel):
    """experiment_metadata.json — best-epoch scalars + config + provenance."""

    experiment_name: Optional[str] = None
    scoring_mode: Optional[str] = None
    swat_eval_mode: Optional[str] = None
    # the 4 score-variant blocks are BEST-EPOCH SCALARS (P3-01) — never per-epoch.
    score_variants: dict[str, dict[str, Optional[float]]] = Field(default_factory=dict)
    loss_stats: dict[str, Optional[float]] = Field(default_factory=dict)
    timing: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    score_formula: Optional[ScoreFormula] = None
    caveats: list[str] = Field(default_factory=list)
