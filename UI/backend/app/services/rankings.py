"""Leaderboard / rankings service — backend-design.md §5.5.

Default metric ``pak_auc_f1`` (registry primary). Sort by the registry DIRECTION
(↓ ⇒ "lower wins"); neutral metrics are excluded from the default ranking but still
returnable with a "no inherent better" note. SWaT ``excl22`` selector picks the right
paired variant. ``filter``/``search`` operate on exp name / suffix / description /
dataset key / status.

P3-05: the rows ARE the N best-epoch scalars (one per leaf) — a strip/dot dataset for
the cross-run distribution view, never a synthetic spread.
"""
from __future__ import annotations

import math
from typing import Any, Optional

# E-1 (P0-1): sentinel for "auto" min_coverage so the route can leave it unspecified and
# the service derives ``max(3, ceil(0.5·coverage_total))``. An explicit integer (incl. 0)
# is honored verbatim; ``min_coverage=0`` reproduces today's coverage-blind aggregate.
_AUTO_MIN_COVERAGE = -1

from ..dataaccess.repository import Repository


def _matches(text_fields: list[Optional[str]], needle: str) -> bool:
    n = needle.lower()
    return any(t and n in t.lower() for t in text_fields)


def leaderboard(
    repo: Repository,
    *,
    metric: str = "pak_auc_f1",
    score_variant: str = "metrics",
    scope: str = "all",
    group_by: str = "dataset",
    swat: str = "full",
    source: str = "all",
    sort: str = "auto",
    search: str = "",
    state: Optional[str] = None,
) -> dict[str, Any]:
    meta = repo.registry.resolve(metric, namespace="epoch_metrics")
    rankable = meta.direction in ("up", "down")

    # RV2-03: ``/leaderboard`` models ONE SWaT variant per table. An unrecognized
    # ``swat`` (e.g. ``both``) used to match NO filter branch and silently drop every
    # SWaT row. The route already rejects ``both`` with a 422; defensively coerce any
    # OTHER unrecognized value to ``full`` here so a direct service caller never gets a
    # silently-empty SWaT set. (``both`` is handled by ``rankings_aggregate``.)
    if swat not in ("full", "excl22"):
        swat = "full"

    # scope parsing: "all" | "exp:{id}" | "dataset:{key}"
    scope_exp = scope_ds = None
    if scope.startswith("exp:"):
        scope_exp = scope[4:]
    elif scope.startswith("dataset:"):
        scope_ds = scope[8:].replace("~", "/")

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for e in repo.experiments():
        if source != "all" and e.source != source:
            continue
        if scope_exp and e.exp_id != scope_exp:
            continue
        if search and not _matches(
            [e.exp_id, e.suffix,
             (e.summary.description if e.summary else None),
             (e.summary.config_set if e.summary else None)], search):
            # still allow a search hit on dataset key below — defer the skip.
            exp_hit = False
        else:
            exp_hit = True
        for d in e.datasets:
            if scope_ds and d.dataset_key != scope_ds:
                continue
            if not exp_hit and search and search.lower() not in d.dataset_key.lower():
                continue
            for variant, leaf in d.variants.items():
                if state and leaf.state != state:
                    continue
                if leaf.state != "complete":
                    continue
                # SWaT paired dataset: pick only the requested variant.
                if set(d.variants) >= {"full", "excl22"} and variant != swat:
                    continue
                b = repo.load_bundle(e, d, leaf, want={"metadata"})
                errors.extend(b.errors)
                if not b.metadata:
                    continue
                block = b.metadata.score_variants.get(score_variant, {}) or {}
                val = block.get(metric)
                if val is None:
                    continue
                rows.append({
                    "exp_id": e.exp_id, "exp_num": e.exp_num,
                    "dataset_key": d.dataset_key,
                    "dataset_key_url": d.dataset_key.replace("/", "~"),
                    "variant": variant, "value": val,
                    "best_epoch": b.metadata.timing.get("best_epoch"),
                    "state": leaf.state, "source": e.source,
                    "description": (e.summary.description if e.summary else None),
                })

    if rankable:
        reverse = meta.direction != "down" if sort == "auto" else (sort == "desc")
        rows.sort(key=lambda r: (r["value"] is None, r["value"]), reverse=reverse)
        for i, r in enumerate(rows):
            r["rank"] = i + 1

    # group_by grouping (a thin index the frontend can pivot on).
    group_index: dict[str, list[int]] = {}
    if group_by in ("dataset", "experiment", "score_variant"):
        keyfn = {
            "dataset": lambda r: r["dataset_key"],
            "experiment": lambda r: r["exp_id"],
            "score_variant": lambda _r: score_variant,
        }[group_by]
        for i, r in enumerate(rows):
            group_index.setdefault(keyfn(r), []).append(i)

    return {
        "metric": metric,
        "score_variant": score_variant,
        "direction": meta.direction,
        "rankable": rankable,
        "neutral_note": (None if rankable else "no inherent better (neutral metric)"),
        "swat": swat,
        "group_by": group_by,
        "group_index": group_index,
        "rows": rows,
        "errors": errors,
    }


# ── FB-R4c: multi-entity (SMD/SMAP/MSL/Exathlon) averaging + canonical/granular cols ─
# A multi-entity dataset GROUP is identified by its dataset_key PREFIX. Its per-entity
# leaves (``<GROUP>/<entity>``) collapse into ONE averaged canonical column
# "<GROUP> (avg)"; the ``<GROUP>/concat`` leaf is a DISTINCT training/eval condition and
# is EXCLUDED from the avg (it remains a separate granular column). Detection is by the
# dataset_key shape (prefix + "/" + entity), NOT a hard-coded machine/channel/app list —
# a new machine/channel/app collapses in automatically (F5 spirit). PSM / WaDi single-leaf
# datasets are untouched. (2026-06-04: Exathlon added — its per-app leaves Exathlon/app1..
# app9 now collapse to "Exathlon (avg)" like SMD/SMAP/MSL, per user feedback.)
_ENTITY_GROUPS = ("SMD", "SMAP", "MSL", "Exathlon")


def _entity_group_of(dataset_key: str) -> Optional[str]:
    """Return the multi-entity GROUP for a per-entity leaf (``SMD/machine-1-2`` -> "SMD",
    ``SMAP/G-7`` -> "SMAP"), or ``None`` for a non-entity / concat / single-leaf key.

    The ``<GROUP>/concat`` leaf is intentionally NOT a group member (it is a distinct
    eval condition, excluded from the avg per FB-R4c-01)."""
    for g in _ENTITY_GROUPS:
        prefix = g + "/"
        if dataset_key.startswith(prefix):
            entity = dataset_key[len(prefix):]
            if entity and entity != "concat":
                return g
    return None


# ── FB-12/FB-13 (R2-03) + FB-R4c: per-dataset ranking → cross-dataset average-rank ─
def rankings_aggregate(
    repo,
    *,
    metric: str = "pak_auc_f1",
    score_variant: str = "metrics",
    datasets: Optional[str] = None,
    swat: str = "excl22",
    source: str = "all",
    search: str = "",
    state: Optional[str] = None,
    min_coverage: int = _AUTO_MIN_COVERAGE,
) -> dict[str, Any]:
    """Per-dataset rankings + a cross-dataset AVERAGE-RANK aggregate (FB-12/FB-13/FB-R4c).

    COLUMN MODEL (FB-R4c):
      * Multi-entity datasets (SMD/SMAP/MSL) COLLAPSE into ONE averaged canonical column
        each — "SMD (avg)" / "SMAP (avg)" / "MSL (avg)" — whose value for an experiment is
        the mean over that exp's per-machine/channel leaves of ``metric`` (EXCLUDING the
        ``*/concat`` leaf). Each counts ONCE in the average-rank; no per-machine column
        inflates it (FB-R4c-01). The per-machine/channel columns + each ``*/concat`` stay
        available as GRANULAR selectable columns.
      * SWaT: the canonical column is ``SWaT_A1A2 · excl22`` (FB-R4c-02); ``SWaT_A1A2 ·
        full`` is a granular column (selectable, not default).
      * PSM / WaDi / other single-leaf datasets: one column each (unchanged).

    SELECTABLE UNIVERSE (FB-R4c-04): the response advertises ``selectable_columns`` =
    every canonical column + every granular column (each machine/channel, each concat,
    SWaT full), and ``default_columns`` = the canonical set. ``datasets=`` is a comma-list
    of column keys chosen FROM that universe; default (None) = the canonical set. The
    average-rank recomputes over whatever set is chosen. ``metric`` may be ANY discovered
    key (F5); direction-aware (F4); neutral metrics returned un-ranked.

    A model's FINAL rank = mean of its per-column ranks over the active columns where it is
    PRESENT; ``coverage = present/total``. Aggregate sorts ascending by ``(avg_rank,
    -coverage, -mean_value)``. R2-03 SWaT-leaf reads preserved. ADDITIVE; ``/leaderboard``
    untouched (FB-12 guard). best-epoch scalars only (P3-01).

    E-1 (P0-1) COVERAGE QUALIFICATION: an experiment scored on too FEW of the active
    columns earns a meaningless avg-rank (e.g. one easy dataset @ rank #1 → avg_rank 1.0
    @ coverage 1/14). Rows whose ``coverage_present < min_coverage`` are moved OUT of
    ``aggregate`` into a SEPARATE ``low_coverage`` array (NOT interleaved) so the headline
    ranking reflects only models evaluated broadly enough to compare. ``min_coverage``
    default = ``max(3, ceil(0.5·coverage_total))`` (auto, derived from the ACTIVE column
    count). An explicit integer is honored verbatim; ``min_coverage=0`` disables the gate
    and reproduces TODAY's coverage-blind aggregate (every row stays in ``aggregate``,
    ``low_coverage=[]``). Both arrays keep the same row shape + their own dense rank index
    (``aggregate_rank`` / ``low_coverage_rank``). The split is applied AFTER the global
    sort, so within each bucket the order is unchanged."""
    meta = repo.registry.resolve(metric, namespace="epoch_metrics")
    rankable = meta.direction in ("up", "down")
    lower_wins = meta.direction == "down"

    # the optional column filter (FB-2/FB-R4c-04). None/"" => the canonical DEFAULT set.
    ds_filter: Optional[set[str]] = None
    if datasets:
        ds_filter = {c.strip() for c in datasets.split(",") if c.strip()}

    errors: list[str] = []
    # raw GRANULAR columns: col_key -> list of {exp_id, exp_num, value, source, description}.
    # SMD/SMAP/MSL per-entity leaves land here too (as granular columns) before collapse.
    granular: dict[str, list[dict[str, Any]]] = {}
    # per-experiment entity-group values (for the canonical "(avg)" columns).
    # group_vals[group][exp_id] -> list of per-entity metric values.
    group_vals: dict[str, dict[str, list[float]]] = {g: {} for g in _ENTITY_GROUPS}
    # exp metadata captured once (for the canonical avg rows whose exp may not appear in
    # any RETAINED granular column once the per-machine columns are filtered out).
    exp_meta_all: dict[str, dict[str, Any]] = {}

    def _emit_granular(col_key: str, e, value: Optional[float]) -> None:
        if value is None:
            return
        granular.setdefault(col_key, []).append({
            "exp_id": e.exp_id, "exp_num": e.exp_num, "value": value,
            "source": e.source,
            "description": (e.summary.description if e.summary else None),
        })

    def _remember_exp(e) -> None:
        exp_meta_all.setdefault(e.exp_id, {
            "exp_id": e.exp_id, "exp_num": e.exp_num, "source": e.source,
            "description": (e.summary.description if e.summary else None)})

    for e in repo.experiments():
        if source != "all" and e.source != source:
            continue
        if search and not _matches(
            [e.exp_id, e.suffix,
             (e.summary.description if e.summary else None),
             (e.summary.config_set if e.summary else None)], search):
            exp_hit_global = False
        else:
            exp_hit_global = True
        for d in e.datasets:
            variants = d.variants
            is_swat_family = bool({"full", "excl22"} & set(variants))
            excl22_present = "excl22" in variants
            group = _entity_group_of(d.dataset_key)
            if not exp_hit_global and search and search.lower() not in d.dataset_key.lower():
                continue
            for variant, leaf in variants.items():
                if state and leaf.state != state:
                    continue
                if leaf.state != "complete":
                    continue
                b = repo.load_bundle(e, d, leaf, want={"metadata"})
                errors.extend(b.errors)
                if not b.metadata:
                    continue
                # R2-03: read EACH leaf's OWN score_variant block — the excl22 leaf's
                # plain ``metrics`` IS the excl22 eval.
                block = b.metadata.score_variants.get(score_variant, {}) or {}
                val = block.get(metric)
                if is_swat_family:
                    col_key = f"{d.dataset_key} · {variant}"
                else:
                    col_key = d.dataset_key
                _emit_granular(col_key, e, val)
                _remember_exp(e)
                # FB-R4c-01: feed a per-entity SMD/SMAP/MSL leaf into its group average
                # (concat excluded by ``_entity_group_of``).
                if group is not None and isinstance(val, (int, float)):
                    group_vals[group].setdefault(e.exp_id, []).append(val)

            # R2-03 fallback: SWaT excl22 LEAF absent → the excl22 eval lives in the FULL
            # leaf's dual-eval blocks. Read ``metrics_excl_region22[metric]`` else
            # ``metrics[excl22_<metric>]`` as the granular ``· excl22`` column.
            if is_swat_family and not excl22_present:
                full_leaf = variants.get("full")
                if full_leaf is not None and full_leaf.state == "complete":
                    fb = repo.load_bundle(e, d, full_leaf, want={"metadata"})
                    errors.extend(fb.errors)
                    if fb.metadata:
                        excl_block = (fb.metadata.score_variants
                                      .get("metrics_excl_region22", {}) or {})
                        excl_val = excl_block.get(metric)
                        if excl_val is None:
                            excl_val = (fb.metadata.score_variants
                                        .get(score_variant, {}) or {}).get(
                                            f"excl22_{metric}")
                        _emit_granular(f"{d.dataset_key} · excl22", e, excl_val)

    # ── build the canonical "(avg)" columns from the per-group per-experiment values ──
    canonical_cols: set[str] = set()
    # group_coverage[col_key][exp_id] -> n entities averaged (FB-R4c-01 report coverage).
    group_coverage: dict[str, dict[str, int]] = {}
    for g in _ENTITY_GROUPS:
        per_exp = group_vals[g]
        if not per_exp:
            continue
        col_key = f"{g} (avg)"
        canonical_cols.add(col_key)
        group_coverage[col_key] = {}
        for exp_id, vals in per_exp.items():
            if not vals:
                continue
            em = exp_meta_all.get(exp_id, {"exp_id": exp_id})
            granular.setdefault(col_key, []).append({
                "exp_id": exp_id, "exp_num": em.get("exp_num"),
                "value": sum(vals) / len(vals),          # mean over the exp's entities
                "source": em.get("source"), "description": em.get("description"),
                "n_entities": len(vals),
            })
            group_coverage[col_key][exp_id] = len(vals)

    # ── classify every discovered column → canonical vs granular, build the universe ──
    # A granular ENTITY-LEAF column (SMD/machine-*, SMAP/<ch>, MSL/<ch>) is collapsed into
    # its "(avg)" canonical column for the DEFAULT set, but stays SELECTABLE.
    all_cols = set(granular.keys())
    swat_full_cols = {c for c in all_cols if c.endswith(" · full")}
    swat_excl_cols = {c for c in all_cols if c.endswith(" · excl22")}
    entity_leaf_cols = {c for c in all_cols
                        if c not in canonical_cols and _entity_group_of(c) is not None}
    concat_cols = {c for c in all_cols if c.endswith("/concat")}

    # the canonical DEFAULT column set (FB-R4c-04 default):
    #   - "(avg)" for each multi-entity group present
    #   - "· excl22" for each SWaT family present (FB-R4c-02); NOT "· full"
    #   - every other single-leaf column (PSM, WaDi/A1, WaDi/A2, …) that is neither an
    #     entity leaf, nor a concat leaf, nor a SWaT variant column.
    default_cols: set[str] = set(canonical_cols) | set(swat_excl_cols)
    for c in all_cols:
        if (c in canonical_cols or c in swat_full_cols or c in swat_excl_cols
                or c in entity_leaf_cols or c in concat_cols):
            continue
        default_cols.add(c)

    # the SELECTABLE universe = canonical columns + every granular column (entity leaves,
    # concat leaves, SWaT full + excl22) + every single-leaf default column.
    selectable_cols = set(all_cols) | set(canonical_cols)

    # ── pick the ACTIVE columns the aggregate ranks over ──────────────────────────
    if ds_filter is None:
        active_cols = set(default_cols)
    else:
        # honor exactly what the caller chose (intersected with what actually exists).
        active_cols = ds_filter & selectable_cols

    # ── rank within each ACTIVE column (direction-aware); build per-column tables ──
    per_dataset: dict[str, list[dict[str, Any]]] = {}
    rank_lookup: dict[str, dict[str, int]] = {}   # exp_id -> {col_key: rank}
    value_lookup: dict[str, dict[str, float]] = {}  # exp_id -> {col_key: value}
    for col_key in sorted(active_cols):
        entries = granular.get(col_key, [])
        if rankable:
            entries = sorted(entries, key=lambda r: (r["value"] is None, r["value"]),
                             reverse=not lower_wins)
        for i, r in enumerate(entries):
            r2 = dict(r)
            r2["rank"] = (i + 1) if rankable else None
            per_dataset.setdefault(col_key, []).append(r2)
            if rankable:
                rank_lookup.setdefault(r["exp_id"], {})[col_key] = i + 1
            value_lookup.setdefault(r["exp_id"], {})[col_key] = r["value"]

    dataset_cols = sorted(per_dataset.keys())
    total_cols = len(dataset_cols)

    # ── cross-column average-rank aggregate over the ACTIVE columns ───────────────
    aggregate: list[dict[str, Any]] = []
    exp_meta: dict[str, dict[str, Any]] = {}
    for col_entries in per_dataset.values():
        for r in col_entries:
            exp_meta.setdefault(r["exp_id"], {
                "exp_id": r["exp_id"], "exp_num": r["exp_num"],
                "source": r["source"], "description": r["description"]})
    for exp_id, em in exp_meta.items():
        ranks = rank_lookup.get(exp_id, {})
        col_vals = value_lookup.get(exp_id, {})
        present = len(col_vals)
        per_col_rank = {c: ranks.get(c) for c in dataset_cols}
        avg_rank = (sum(ranks.values()) / len(ranks)) if ranks else None
        vals = list(col_vals.values())
        mean_value = (sum(vals) / len(vals)) if vals else None
        aggregate.append({
            **em,
            "avg_rank": avg_rank,
            "coverage_present": present,
            "coverage_total": total_cols,
            "mean_value": mean_value,
            "per_dataset_rank": per_col_rank,
        })

    if rankable:
        aggregate.sort(key=lambda a: (
            a["avg_rank"] if a["avg_rank"] is not None else float("inf"),
            -(a["coverage_present"] or 0),
            -(a["mean_value"] if a["mean_value"] is not None else float("-inf")),
        ))

    # ── E-1 (P0-1): resolve the coverage gate, then SPLIT into qualified vs low ────
    # ``min_coverage`` default (auto) = max(3, ceil(0.5·active-column-count)); explicit
    # 0 disables the gate (today's behaviour). Clamp a too-large request to ``total_cols``
    # so the gate can never empty the qualified set for an honest reason.
    if min_coverage == _AUTO_MIN_COVERAGE:
        effective_min_coverage = max(3, math.ceil(0.5 * total_cols)) if total_cols else 0
    else:
        effective_min_coverage = max(0, int(min_coverage))
    effective_min_coverage = min(effective_min_coverage, total_cols)

    qualified: list[dict[str, Any]] = []
    low_coverage: list[dict[str, Any]] = []
    for a in aggregate:
        # NOTE: the global sort above already ordered both buckets; partitioning here
        # preserves that order WITHIN each bucket (a stable split, not a re-sort).
        if (a["coverage_present"] or 0) >= effective_min_coverage:
            qualified.append(a)
        else:
            low_coverage.append(a)

    if rankable:
        for i, a in enumerate(qualified):
            a["aggregate_rank"] = i + 1
        for i, a in enumerate(low_coverage):
            a["low_coverage_rank"] = i + 1

    # column-classification metadata (FB-R4c-04) so the frontend can group the selector.
    def _col_kind(c: str) -> str:
        if c in canonical_cols:
            return "canonical_avg"
        if c in swat_excl_cols:
            return "swat_excl22"
        if c in swat_full_cols:
            return "swat_full"
        if c in entity_leaf_cols:
            return "entity_leaf"
        if c in concat_cols:
            return "concat"
        return "single_leaf"

    selectable_sorted = sorted(selectable_cols)
    column_meta = {c: {
        "kind": _col_kind(c),
        "group": (_entity_group_of(c)
                  if _entity_group_of(c) is not None
                  else (c[:-len(" (avg)")] if c in canonical_cols else None)),
        "is_default": c in default_cols,
    } for c in selectable_sorted}

    return {
        "metric": metric,
        "score_variant": score_variant,
        "direction": meta.direction,
        "rankable": rankable,
        "neutral_note": (None if rankable else "no inherent better (neutral metric)"),
        "swat": swat,
        # the ACTIVE columns the aggregate currently ranks over.
        "datasets": dataset_cols,
        "dataset_labels": {c: c for c in dataset_cols},
        "coverage_total": total_cols,
        # FB-R4c-04: the fully-configurable selector universe + the canonical default set.
        "selectable_columns": selectable_sorted,
        "default_columns": sorted(default_cols),
        "canonical_columns": sorted(canonical_cols),
        "column_meta": column_meta,
        # FB-R4c-01: how many entities each "(avg)" column averaged, per experiment.
        "group_coverage": group_coverage,
        "per_dataset": per_dataset,
        # E-1 (P0-1): ``aggregate`` is the QUALIFIED headline set (coverage_present >=
        # min_coverage); under-covered rows live in ``low_coverage`` (NOT interleaved).
        "aggregate": qualified,
        "low_coverage": low_coverage,
        # the effective coverage threshold actually applied (auto-derived or explicit), so
        # the frontend can label the "insufficient coverage (n/N)" section.
        "min_coverage": effective_min_coverage,
        "min_coverage_auto": (min_coverage == _AUTO_MIN_COVERAGE),
        "missing_dataset_rule": ("avg_rank = mean(per-column rank over active columns "
                                 "where present); coverage = present/total. SMD/SMAP/MSL "
                                 "(avg) = mean over per-entity leaves (concat excluded), "
                                 "counted once. SWaT default = excl22 (full selectable). "
                                 "E-1: rows with coverage_present < min_coverage move to "
                                 "low_coverage (min_coverage=0 reproduces the old set)."),
        "errors": errors,
    }
