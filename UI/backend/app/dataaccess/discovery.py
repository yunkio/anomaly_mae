"""Discovery — presence-driven leaf detection + completion classification.

backend-design.md §2.1 / §2.2. NEVER assumes a name or a fixed depth: leaves are
found by the PRESENCE of ``epoch_metrics.json`` OR ``experiment_metadata.json`` at
depth 1 OR 2 under an experiment dir. Non-experiment siblings (``queue_summary.json``,
``telemetry/``, …) are skipped without error. SWaT ``A1A2_full``/``A1A2_excl22``
collapse to one logical dataset with a variant selector.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

from ..config import SETTINGS
from .io_safe import CorruptOrWriting
from .models import (
    DatasetRef,
    ExperimentRef,
    LeafState,
    PresenceFlags,
    SummaryRollup,
)
from .parsers import parse_summary

# [<alpha-prefix>_]<expNum>_<YYYYMMDD_HHMMSS>[_<suffix>]
# The optional alpha prefix (FB-R4d-01) lets archived dirs `new_271_…`/`old_271_…`
# match while keeping bare-numeric `271_…` valid. exp_id stays the FULL dir name
# (so new_/old_/bare are DISTINCT experiments); exp_num is the numeric `num` group.
# The trailing suffix is OPTIONAL (2026-06-04) so a `legacy_271_20260508_094241`-style
# name (prefix + num + timestamp, no descriptor) still parses exp_num/timestamp.
_EXP_NAME_RE = re.compile(
    r"^(?:(?P<prefix>[A-Za-z]+)_)?(?P<num>\d+)_(?P<ts>\d{8}_\d{6})(?:_(?P<suffix>.+))?$"
)
# SWaT sibling leaves: "<DatasetGroup>/A1A2_full" + "A1A2_excl22"
_SWAT_VARIANT_RE = re.compile(r"^(?P<base>.+)_(?P<variant>full|excl22)$")


def _presence(leaf: Path) -> PresenceFlags:
    """Stat-only presence flags for a leaf dir (no file body is read)."""
    def has(name: str) -> bool:
        return (leaf / name).exists()
    return PresenceFlags(
        epoch_metrics=has("epoch_metrics.json"),
        experiment_metadata=has("experiment_metadata.json"),
        best_config=has("best_config.json"),
        training_histories=has("training_histories.json"),
        anomaly_type_metrics=has("anomaly_type_metrics.json"),
        batch_profiling=has("batch_profiling.json"),
        best_model_detailed=has("best_model_detailed.csv"),
        best_epoch_train_scores=has("best_epoch_train_scores.npz"),
        epoch_scores_dir=(leaf / "epoch_scores").is_dir(),
        checkpoints_dir=(leaf / "checkpoints").is_dir(),
        visualization_dir=(leaf / "visualization").is_dir(),
    )


def _classify_leaf(present: PresenceFlags) -> str:
    """complete | in_progress | missing — from file presence only (§2.2)."""
    if present.experiment_metadata:
        return "complete"
    if present.epoch_metrics or present.epoch_scores_dir:
        return "in_progress"
    return "missing"


def _is_leaf_dir(d: Path) -> bool:
    """A dir is a leaf if it carries either marker file (epoch_metrics/metadata) OR
    is plainly an in-progress leaf (epoch_scores/ only, no marker yet)."""
    if any((d / m).exists() for m in SETTINGS.leaf_marker_files):
        return True
    # in-progress: epoch_scores/ present but no marker file yet (live run).
    if (d / "epoch_scores").is_dir() or (d / "checkpoints").is_dir():
        # but only count as a leaf if it is NOT itself a parent of leaves
        return True
    return False


def _scan_leaves(exp_root: Path) -> list[tuple[str, str, Path, int]]:
    """Return (dataset_group, variant, leaf_path, depth) tuples.

    Walks depth 1 then depth 2. A depth-1 dir that itself contains leaf-dirs is a
    GROUP (e.g. ``SWaT/``), not a leaf; its children are the leaves at depth 2.
    """
    found: list[tuple[str, str, Path, int]] = []
    try:
        level1 = [Path(e.path) for e in os.scandir(exp_root) if e.is_dir()]
    except OSError:
        return found

    for d1 in level1:
        name1 = d1.name
        if name1 in ("telemetry",):
            continue
        # depth-2 children that are themselves leaves => d1 is a group.
        try:
            level2 = [Path(e.path) for e in os.scandir(d1) if e.is_dir()]
        except OSError:
            level2 = []
        child_leaves = [d2 for d2 in level2 if _is_leaf_dir(d2)]

        if _is_leaf_dir(d1) and not child_leaves:
            # depth-1 leaf (e.g. PSM). variant "_" unless a full/excl22 suffix.
            m = _SWAT_VARIANT_RE.match(name1)
            if m:
                found.append((m.group("base"), m.group("variant"), d1, 1))
            else:
                found.append((name1, "_", d1, 1))
        elif child_leaves:
            # depth-2 leaves under a group dir (e.g. SWaT/A1A2_full).
            for d2 in child_leaves:
                m = _SWAT_VARIANT_RE.match(d2.name)
                if m:
                    # logical dataset key = group + base, e.g. SWaT_A1A2
                    ds_key = f"{name1}_{m.group('base')}"
                    found.append((ds_key, m.group("variant"), d2, 2))
                else:
                    ds_key = f"{name1}/{d2.name}"
                    found.append((ds_key, "_", d2, 2))
    return found


def _group_into_datasets(
    leaves: list[tuple[str, str, Path, int]]
) -> list[DatasetRef]:
    """Collapse (group, variant, path, depth) tuples into DatasetRefs (SWaT pairing)."""
    by_key: dict[str, dict[str, LeafState]] = {}
    for ds_key, variant, leaf_path, depth in leaves:
        present = _presence(leaf_path)
        state = _classify_leaf(present)
        ls = LeafState(
            variant=variant,
            leaf_path=leaf_path,
            depth=depth,
            present=present,
            state=state,
        )
        by_key.setdefault(ds_key, {})[variant] = ls

    refs: list[DatasetRef] = []
    for ds_key, variants in sorted(by_key.items()):
        states = {v.state for v in variants.values()}
        if "complete" in states:
            ds_state = "complete"
        elif "in_progress" in states:
            ds_state = "in_progress"
        else:
            ds_state = "missing"
        refs.append(DatasetRef(dataset_key=ds_key, variants=variants, state=ds_state))
    return refs


def _classify_experiment(datasets: list[DatasetRef]) -> str:
    """complete (any leaf complete) | in_progress | early_abort (no leaves)."""
    if not datasets:
        return "early_abort"
    states = {d.state for d in datasets}
    if "complete" in states:
        return "complete"
    if "in_progress" in states:
        return "in_progress"
    return "early_abort"


def discover_experiment(exp_root: Path, source: str) -> Optional[ExperimentRef]:
    """Build an ExperimentRef from one experiment dir, or None if the name doesn't
    match the experiment pattern (sibling files/dirs are skipped, not errored)."""
    m = _EXP_NAME_RE.match(exp_root.name)
    exp_num = timestamp = suffix = None
    if m:
        # exp_id is the FULL dir name (set on the ExperimentRef below); exp_num is
        # the numeric group only — the optional alpha prefix (new_/old_) is captured
        # but intentionally NOT folded into exp_num, so labels/sorting stay numeric
        # while new_271/old_271/bare-271 remain distinct experiments via exp_id.
        exp_num = int(m.group("num"))
        timestamp = m.group("ts")
        suffix = m.group("suffix")

    errors: list[str] = []
    leaves = _scan_leaves(exp_root)
    datasets = _group_into_datasets(leaves)

    summary: Optional[SummaryRollup] = None
    summ_path = exp_root / "summary.json"
    if summ_path.exists():
        try:
            summary = SummaryRollup(**parse_summary(summ_path, exp_root))
        except CorruptOrWriting as exc:
            errors.append(f"summary.json: {exc}")

    return ExperimentRef(
        exp_id=exp_root.name,
        exp_num=exp_num,
        timestamp=timestamp,
        suffix=suffix,
        root=exp_root,
        source=source,  # type: ignore[arg-type]
        state=_classify_experiment(datasets),  # type: ignore[arg-type]
        datasets=datasets,
        summary=summary,
        errors=errors,
    )


def scan(roots: Optional[dict[str, Path]] = None) -> list[ExperimentRef]:
    """Scan all source roots for experiments. Sibling NON-experiment files
    (``queue_summary.json`` etc.) at the root are skipped silently.

    An experiment dir is one whose name matches ``<num>_<TS>_<suffix>``; everything
    else at that level is ignored. Per-experiment failures isolate to ``errors[]``.
    """
    roots = roots or SETTINGS.source_roots
    out: list[ExperimentRef] = []
    for source, root in roots.items():
        if not root.exists():
            continue
        try:
            children = sorted(os.scandir(root), key=lambda e: e.name)
        except OSError:
            continue
        for entry in children:
            if not entry.is_dir():
                continue  # queue_summary.json and friends: skipped, never errored.
            d = Path(entry.path)
            if not _EXP_NAME_RE.match(d.name):
                continue  # non-experiment dir: skipped.
            if any(d.name.startswith(p) for p in SETTINGS.exclude_prefixes):
                continue  # hidden by config (e.g. legacy_*): on disk but not shown.
            try:
                ref = discover_experiment(d, source)
            except Exception as exc:  # noqa: BLE001 - isolation by design
                ref = ExperimentRef(
                    exp_id=d.name, root=d, source=source,  # type: ignore[arg-type]
                    state="early_abort", errors=[f"discover failed: {exc}"],
                )
            if ref is not None:
                out.append(ref)
    return out
