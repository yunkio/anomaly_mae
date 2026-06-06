"""Per-file-kind parsers — backend-design.md §2.4 / §3.

Each parser is read-only and tolerant: a missing file returns ``None``; a corrupt
body raises ``CorruptOrWriting`` which the leaf-level wrapper isolates into
``errors[]`` (never a 500). All metric-bearing outputs are OPEN maps (F5).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .io_safe import CorruptOrWriting, read_json
from .models import (
    BestSnapshot,
    EpochSeries,
    NestedPivot,
    TrainingHistory,
)
from .scoring import resolve_score_formula

# The four best-epoch score-variant blocks (BEST-EPOCH SCALARS only, P3-01).
SCORE_VARIANT_BLOCKS = (
    "metrics",
    "teacher_recon_metrics",
    "student_recon_metrics",
    "disc_metrics",
)
# conditional SWaT dual-eval variants
SCORE_VARIANT_CONDITIONAL = (
    "metrics_excl_region22",
    "teacher_recon_metrics_excl_region22",
    "metrics_full",
    "teacher_recon_metrics_full",
)


def _is_scalar(v: Any) -> bool:
    return isinstance(v, (int, float)) or v is None


# FB-7 / R2-06 (display-layer only; result files untouched). The anomaly-type seed
# names (data-schema.json::anomaly_type_vocabulary.canonical_full_vocabulary.
# display_name_seeds). The scoped relabel rule (``anomaly_type_display_names`` below)
# decides spike->"Anomaly" ONLY for the single-bucket real-world case; for the
# multi-type simulation vocabulary spike keeps its own seed "Spike / DDoS". The type
# vocabulary is ALWAYS discovered at runtime — this is a seeded display map, NEVER a
# hard-coded type list (F5-analog / DS-1).
_ANOMALY_TYPE_DISPLAY_SEEDS: dict[str, str] = {
    "normal": "Normal",
    "spike": "Spike / DDoS",
    "memory_leak": "Memory Leak",
    "cpu_saturation": "CPU Saturation",
    "network_congestion": "Network Congestion",
    "cascading_failure": "Cascading Failure",
    "resource_contention": "Resource Contention",
    "correlation_inversion": "Correlation Inversion (pattern)",
    "temporal_flatline": "Temporal Flatline (pattern)",
    "frequency_shift": "Frequency Shift (pattern)",
    "attack": "Attack",
    "anomaly": "Anomaly",
}


def anomaly_type_display_names(type_keys: list[str] | set[str]) -> dict[str, str]:
    """Backend-supplied per-anomaly-type display map (FB-7, R2-06) over a discovered
    type vocabulary. SCOPED rule:

      * ``normal`` -> "Normal" ALWAYS.
      * if the anomaly types (``set(keys) - {"normal"}``) are EXACTLY ``{"spike"}``
        (the single-bucket real-world case — PSM/SWaT/WaDi route vocabulary is just
        ``{spike}``; the training-history pivot is ``{normal, spike}``; subtracting an
        absent ``normal`` is harmless so the predicate fires on BOTH surfaces) ->
        relabel ``spike`` -> "Anomaly".
      * otherwise (the simulation multi-type vocabulary) keep each type's own seed
        (``spike`` stays "Spike / DDoS") so the 9-type vocabulary is intact.
      * any unknown/new type (e.g. ``fault_12``) -> seed if present, else title-cased
        from the key — NEVER dropped, never collides with a hard-coded list (F5).

    Display-layer only; the underlying TYPE KEY in every payload stays the raw key.
    """
    keys = list(type_keys)
    anomaly_types = {k for k in keys if k != "normal"}
    single_bucket = anomaly_types == {"spike"}
    out: dict[str, str] = {}
    for k in keys:
        if k == "normal":
            out[k] = "Normal"
        elif k == "spike" and single_bucket:
            out[k] = "Anomaly"
        else:
            out[k] = _ANOMALY_TYPE_DISPLAY_SEEDS.get(k, k.replace("_", " ").title())
    return out


def parse_epoch_metrics(path: Path, warmup_epochs: Optional[int] = None) -> EpochSeries:
    """epoch_metrics.json -> EpochSeries by UNION over all entries' keys (§3.1).

    Scalar keys become aligned ``series`` lists (null where a key is absent in some
    epoch — never a fabricated 0). Array-valued keys (``_per_k_*``, ``_*_feature_*``)
    are recorded in ``array_keys`` but not flattened into the scalar series map.
    """
    doc = read_json(path)
    if not isinstance(doc, dict):
        raise CorruptOrWriting(f"epoch_metrics not an object: {path}")
    entries = doc.get("epochs", []) or []
    eval_interval = doc.get("eval_interval")

    epochs: list[int] = []
    scalar_keys: set[str] = set()
    array_keys: set[str] = set()
    for e in entries:
        if not isinstance(e, dict):
            continue
        ep = e.get("epoch")
        epochs.append(int(ep) if isinstance(ep, (int, float)) else len(epochs) + 1)
        for k, v in e.items():
            if k == "epoch":
                continue
            if isinstance(v, list):
                array_keys.add(k)
            elif _is_scalar(v):
                scalar_keys.add(k)
            else:
                array_keys.add(k)  # nested/object -> treat as non-scalar

    series: dict[str, list[Optional[float]]] = {k: [] for k in scalar_keys}
    for e in entries:
        if not isinstance(e, dict):
            continue
        for k in scalar_keys:
            v = e.get(k, None)
            series[k].append(float(v) if isinstance(v, (int, float)) else None)

    return EpochSeries(
        eval_interval=int(eval_interval) if isinstance(eval_interval, (int, float)) else None,
        epochs=epochs,
        series=series,
        array_keys=sorted(array_keys),
        warmup_epochs=warmup_epochs,
    )


def parse_experiment_metadata(path: Path) -> BestSnapshot:
    """experiment_metadata.json -> BestSnapshot (best-epoch scalars + config + formula)."""
    doc = read_json(path)
    if not isinstance(doc, dict):
        raise CorruptOrWriting(f"experiment_metadata not an object: {path}")

    score_variants: dict[str, dict[str, Optional[float]]] = {}
    for block in (*SCORE_VARIANT_BLOCKS, *SCORE_VARIANT_CONDITIONAL):
        b = doc.get(block)
        if isinstance(b, dict):
            score_variants[block] = {
                k: (float(v) if isinstance(v, (int, float)) else None)
                for k, v in b.items()
            }

    loss_stats_raw = doc.get("loss_stats") or {}
    loss_stats = {
        k: (float(v) if isinstance(v, (int, float)) else None)
        for k, v in loss_stats_raw.items()
    } if isinstance(loss_stats_raw, dict) else {}

    caveats: list[str] = []
    if isinstance(doc.get("_prewarmup_backfill"), dict):
        pb = doc["_prewarmup_backfill"]
        if pb.get("stale_best_model_viz"):
            caveats.append("stale_best_model_viz")
        if pb.get("applied"):
            caveats.append("prewarmup_backfill_applied")
    formula = resolve_score_formula(doc)
    if formula.fm_in_score:
        caveats.append("fm_in_score")

    return BestSnapshot(
        experiment_name=doc.get("experiment_name"),
        scoring_mode=doc.get("scoring_mode"),
        swat_eval_mode=doc.get("swat_eval_mode"),
        score_variants=score_variants,
        loss_stats=loss_stats,
        timing=doc.get("timing") or {},
        config=doc.get("config") or {},
        score_formula=formula,
        caveats=caveats,
    )


def parse_best_config(path: Path) -> dict[str, Any]:
    doc = read_json(path)
    return doc if isinstance(doc, dict) else {}


def _is_type_submetric_list(seq: list) -> bool:
    """True only for the anomaly-type signature ``list[dict[type, dict[sub, scalar]]]``.

    Discriminates ``epoch_anomaly_type_scores`` (every value of every populated entry
    is itself a dict of SCALARS) from look-alikes such as ``batch_profiling`` (whose
    entry dicts mix scalar fields with a nested ``layer_timing`` dict) and
    ``epoch_timings`` (entry values are scalars, not dicts). Empty entries (``{}``)
    are ignored. Requires at least one populated entry to qualify.
    """
    saw_populated = False
    for entry in seq:
        if not isinstance(entry, dict) or not entry:
            continue
        for _t, smap in entry.items():
            if not isinstance(smap, dict):
                return False  # a non-dict value => not the type->sub-metric shape
            for _sm, val in smap.items():
                if not (isinstance(val, (int, float)) or val is None):
                    return False  # sub-metric value must be a scalar
        saw_populated = True
    return saw_populated


def _pivot_anomaly_type_scores(seq: list) -> NestedPivot:
    """list[dict[type, dict[sub_metric, val]]] -> {type:{sub:[len-N, null-padded]}}.

    backend-design.md §5.3.2. Ragged input (early entries ``{}``) is null-padded so
    every series stays aligned to the training-epoch grid (never a fabricated 0).
    Types + sub_metrics are the UNION across all entries (dynamic; never hard-coded).
    """
    n = len(seq)
    types: list[str] = []
    sub_metrics: list[str] = []
    type_seen: set[str] = set()
    sub_seen: set[str] = set()
    for entry in seq:
        if not isinstance(entry, dict):
            continue
        for t, smap in entry.items():
            if t not in type_seen:
                type_seen.add(t)
                types.append(t)
            if isinstance(smap, dict):
                for sm in smap:
                    if sm not in sub_seen:
                        sub_seen.add(sm)
                        sub_metrics.append(sm)

    pivot: dict[str, dict[str, list[Optional[float]]]] = {
        t: {sm: [None] * n for sm in sub_metrics} for t in types
    }
    for i, entry in enumerate(seq):
        if not isinstance(entry, dict):
            continue
        for t, smap in entry.items():
            if not isinstance(smap, dict):
                continue
            for sm, val in smap.items():
                if t in pivot and sm in pivot[t]:
                    pivot[t][sm][i] = float(val) if isinstance(val, (int, float)) else None
    return NestedPivot(types=types, sub_metrics=sub_metrics, pivot=pivot)


def parse_training_histories(path: Path) -> TrainingHistory:
    """training_histories.json -> TrainingHistory (per-TRAINING-epoch axis, P2-01).

    UNION over all flat per-epoch lists; nested keys (``epoch_anomaly_type_scores``)
    pivoted + null-padded. Empty lists (disabled components) are recorded separately.
    """
    doc = read_json(path)
    if not isinstance(doc, dict):
        raise CorruptOrWriting(f"training_histories not an object: {path}")
    # top-level keyed by a string repeat-index, usually "0".
    inner_key = "0" if "0" in doc else (next(iter(doc), None))
    h = doc.get(inner_key) if inner_key is not None else None
    if not isinstance(h, dict):
        return TrainingHistory()

    train_epochs = [int(x) for x in h.get("epoch", []) if isinstance(x, (int, float))]
    series: dict[str, list[Optional[float]]] = {}
    nested: dict[str, NestedPivot] = {}
    empty_keys: list[str] = []

    for k, v in h.items():
        if k == "epoch":
            continue
        if not isinstance(v, list):
            continue
        if len(v) == 0:
            empty_keys.append(k)
            series[k] = []
            continue
        head = v[0]
        if isinstance(head, (int, float)) or head is None:
            series[k] = [float(x) if isinstance(x, (int, float)) else None for x in v]
        elif isinstance(head, dict):
            # dict-valued nested list -> pivot ONLY for the anomaly-type signature
            # (dict[type, dict[sub_metric, scalar]]). batch_profiling / epoch_timings
            # are look-alikes (mixed/nested or scalar values) and are intentionally
            # NOT pivoted — they are exposed via the profiling/timing endpoints.
            if _is_type_submetric_list(v):
                nested[k] = _pivot_anomaly_type_scores(v)
        # list-of-list (per-feature) handled by a dedicated per-feature endpoint.

    return TrainingHistory(
        train_epochs=train_epochs,
        series=series,
        nested=nested,
        empty_keys=sorted(empty_keys),
    )


def parse_anomaly_type_metrics(path: Path) -> dict[str, dict[str, Any]]:
    doc = read_json(path)
    return doc if isinstance(doc, dict) else {}


def parse_summary(path: Path, experiment_root: Path) -> dict[str, Any]:
    """summary.json -> rollup with each ``results[].dir`` resolved RELATIVE to the
    experiment dir (DISC-4 — the captured absolute path is stale after a move)."""
    doc = read_json(path)
    if not isinstance(doc, dict):
        return {}
    results = doc.get("results") or []
    fixed: list[dict] = []
    for r in results:
        if not isinstance(r, dict):
            continue
        r2 = dict(r)
        d = r.get("dir")
        if isinstance(d, str) and d:
            # stale absolute path -> resolve by tail relative to the experiment dir.
            tail = Path(d).name
            parent = Path(d).parent.name
            candidate = experiment_root / parent / tail
            if not candidate.exists():
                candidate = experiment_root / tail
            r2["dir_resolved"] = str(candidate) if candidate.exists() else None
        fixed.append(r2)
    return {
        "total_time": doc.get("total_time"),
        "config_set": doc.get("config_set"),
        "description": doc.get("description"),
        "timestamp": doc.get("timestamp"),
        "results": fixed,
    }
