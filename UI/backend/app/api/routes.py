"""Core API routes — backend-design.md §5 (data-access subset for the backend core).

Implements the must-have, read-cheap endpoints that bind directly to the data layer:
discovery, experiment/dataset detail, config, dynamic metrics-catalog, batched
epoch-series (eval-epoch), training-history (training-epoch), stats, leaderboard,
anomaly-types, NPZ score slabs + histogram, monitoring, health, rescan.

GIF rendering (§6), compare matrices (§5.6), and computed-panel plugins (§8) are
later stages; this module keeps ./UI/ runnable and validated on real fixtures now.
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import PlainTextResponse

from ..config import SETTINGS
from ..dataaccess.io_safe import CorruptOrWriting, read_json
from ..dataaccess.npz_reader import histogram as npz_histogram
from ..dataaccess.npz_reader import read_arrays
from ..dataaccess.parsers import (
    anomaly_type_display_names,
    parse_anomaly_type_metrics,
)
from ..dataaccess.repository import get_repository
from ..services import arrays as arr_svc
from ..services import compare as cmp_svc
from ..services import export as export_svc
from ..services import insights as insight_svc
from ..services import panels as panel_svc
from ..services.rankings import _AUTO_MIN_COVERAGE as _AGG_AUTO_MIN_COVERAGE
from ..services.rankings import leaderboard as leaderboard_svc
from ..services.rankings import rankings_aggregate as rankings_aggregate_svc
from ..services.series import epoch_series, stats_for_keys, training_history
from ..services.viz_manifest import viz_manifest as viz_manifest_svc

router = APIRouter(prefix="/api")


def _split_keys(keys: Optional[str]) -> list[str]:
    return [k for k in (keys or "").split(",") if k] if keys else []


# ── discovery / experiment / dataset ─────────────────────────────────────────
@router.get("/health")
def health():
    return get_repository().health()


@router.post("/rescan")
def rescan():
    return get_repository().rescan()


@router.get("/experiments")
def list_experiments(source: str = Query("all"), state: Optional[str] = None):
    repo = get_repository()
    out = []
    for e in repo.experiments():
        if source != "all" and e.source != source:
            continue
        if state and e.state != state:
            continue
        out.append({
            "exp_id": e.exp_id, "exp_num": e.exp_num, "timestamp": e.timestamp,
            "suffix": e.suffix, "source": e.source, "state": e.state,
            "n_datasets": len(e.datasets),
            "datasets": [{"dataset_key": d.dataset_key,
                          "dataset_key_url": d.dataset_key.replace("/", "~"),
                          "state": d.state,
                          "variants": list(d.variants.keys())} for d in e.datasets],
            "errors": e.errors,
        })
    return out


@router.get("/experiments/{exp_id}")
def experiment_detail(exp_id: str):
    repo = get_repository()
    exp = repo.find_experiment(exp_id)
    if exp is None:
        raise HTTPException(404, f"experiment {exp_id} not found")
    datasets = []
    for d in exp.datasets:
        best = {}
        leaf = next(iter(d.variants.values()), None)
        if leaf and leaf.state == "complete":
            b = repo.load_bundle(exp, d, leaf, want={"metadata"})
            if b.metadata:
                best = {
                    "score_variants": b.metadata.score_variants,
                    "score_formula": b.metadata.score_formula.model_dump()
                    if b.metadata.score_formula else None,
                    "timing": b.metadata.timing,
                    "caveats": b.metadata.caveats,
                }
        datasets.append({"dataset_key": d.dataset_key,
                         "dataset_key_url": d.dataset_key.replace("/", "~"),
                         "state": d.state,
                         "variants": list(d.variants.keys()), "best": best})
    return {
        "exp_id": exp.exp_id, "source": exp.source, "state": exp.state,
        "summary": exp.summary.model_dump() if exp.summary else None,
        "datasets": datasets, "errors": exp.errors,
    }


@router.get("/experiments/{exp_id}/datasets/{dataset_key}")
def dataset_detail(exp_id: str, dataset_key: str, variant: Optional[str] = None):
    repo = get_repository()
    exp, ds, leaf = repo.resolve_leaf(exp_id, dataset_key, variant)
    if exp is None or ds is None or leaf is None:
        raise HTTPException(404, f"{exp_id}/{dataset_key} (variant={variant}) not found")
    b = repo.load_bundle(exp, ds, leaf, want={"metadata", "config"})
    md = b.metadata
    return {
        "exp_id": exp_id, "dataset_key": dataset_key,
        "variant": leaf.variant, "state": leaf.state,
        "present": leaf.present.model_dump(),
        "warmup_epochs": leaf.warmup_epochs, "num_features": leaf.num_features,
        "eval_interval": leaf.eval_interval,
        "score_variants": md.score_variants if md else {},
        "loss_stats": md.loss_stats if md else {},
        "timing": md.timing if md else {},
        "score_formula": md.score_formula.model_dump() if md and md.score_formula else None,
        "caveats": md.caveats if md else [],
        "errors": b.errors,
    }


@router.get("/experiments/{exp_id}/datasets/{dataset_key}/config")
def dataset_config(exp_id: str, dataset_key: str, variant: Optional[str] = None):
    repo = get_repository()
    exp, ds, leaf = repo.resolve_leaf(exp_id, dataset_key, variant)
    if leaf is None:
        raise HTTPException(404, "leaf not found")
    b = repo.load_bundle(exp, ds, leaf, want={"config", "metadata"})
    return {
        "config": b.config,
        "score_formula": b.metadata.score_formula.model_dump()
        if b.metadata and b.metadata.score_formula else None,
        "errors": b.errors,
    }


# ── dynamic metric catalog (F5) ──────────────────────────────────────────────
@router.get("/metrics-catalog")
def global_metrics_catalog():
    """GLOBAL UNION catalog across every discovered leaf (F5). Powers the
    cross-cutting pickers (Rankings IA-8, Compare IA-7) which are not scoped to a
    single leaf — so a renamed/new metric is rankable/comparable automatically and
    NO hard-coded metric list lives in the frontend (review-P4-frontend F-01)."""
    repo = get_repository()
    cat = repo.global_catalog()
    return {"count": len(cat), "entries": [c.model_dump() for c in cat]}


@router.get("/experiments/{exp_id}/datasets/{dataset_key}/metrics-catalog")
def metrics_catalog(exp_id: str, dataset_key: str, variant: Optional[str] = None):
    repo = get_repository()
    exp, ds, leaf = repo.resolve_leaf(exp_id, dataset_key, variant)
    if leaf is None:
        raise HTTPException(404, "leaf not found")
    b = repo.load_bundle(exp, ds, leaf)
    cat = repo.leaf_catalog(b)
    return {"count": len(cat), "entries": [c.model_dump() for c in cat],
            "errors": b.errors}


# ── epoch-series (eval-epoch) + training-history (training-epoch) ─────────────
@router.get("/experiments/{exp_id}/datasets/{dataset_key}/epoch-series")
def get_epoch_series(exp_id: str, dataset_key: str,
                     keys: Optional[str] = None, variant: Optional[str] = None,
                     include_meta: bool = True):
    repo = get_repository()
    exp, ds, leaf = repo.resolve_leaf(exp_id, dataset_key, variant)
    if leaf is None:
        raise HTTPException(404, "leaf not found")
    b = repo.load_bundle(exp, ds, leaf, want={"epoch", "metadata"})
    return epoch_series(repo, b, _split_keys(keys), include_meta=include_meta)


@router.get("/experiments/{exp_id}/datasets/{dataset_key}/training-history")
def get_training_history(exp_id: str, dataset_key: str,
                         keys: Optional[str] = None, variant: Optional[str] = None,
                         include_meta: bool = True):
    repo = get_repository()
    exp, ds, leaf = repo.resolve_leaf(exp_id, dataset_key, variant)
    if leaf is None:
        raise HTTPException(404, "leaf not found")
    b = repo.load_bundle(exp, ds, leaf, want={"history"})
    return training_history(repo, b, _split_keys(keys), include_meta=include_meta)


@router.get("/experiments/{exp_id}/datasets/{dataset_key}/stats")
def get_stats(exp_id: str, dataset_key: str,
              keys: Optional[str] = None, variant: Optional[str] = None):
    repo = get_repository()
    exp, ds, leaf = repo.resolve_leaf(exp_id, dataset_key, variant)
    if leaf is None:
        raise HTTPException(404, "leaf not found")
    b = repo.load_bundle(exp, ds, leaf, want={"epoch"})
    return {"stats": stats_for_keys(repo, b, _split_keys(keys)), "errors": b.errors}


# ── leaderboard / rankings ───────────────────────────────────────────────────
@router.get("/leaderboard")
def leaderboard(metric: str = "pak_auc_f1", score_variant: str = "metrics",
                scope: str = "all", group_by: str = "dataset",
                swat: str = "full", source: str = "all", sort: str = "auto",
                search: str = "", state: Optional[str] = None):
    # RV2-03: ``/leaderboard`` only models ONE SWaT variant per call (full | excl22).
    # ``swat=both`` previously matched no filter branch and SILENTLY dropped every SWaT
    # row. Reject it with a 422 that points the caller to the both-columns aggregate,
    # so ``both`` can never be wired here expecting both rows. (FB-11's both-variants
    # view lives in ``/rankings/aggregate``.)
    if swat == "both":
        raise HTTPException(
            status_code=422,
            detail=("/leaderboard supports swat=full|excl22 only (one variant per "
                    "table); use /rankings/aggregate?swat=both for both SWaT variants "
                    "as columns (FB-11)."))
    return leaderboard_svc(
        get_repository(), metric=metric, score_variant=score_variant, scope=scope,
        group_by=group_by, swat=swat, source=source, sort=sort, search=search,
        state=state)


@router.get("/rankings/aggregate")
def rankings_aggregate(metric: str = "pak_auc_f1", score_variant: str = "metrics",
                       datasets: Optional[str] = None, swat: str = "excl22",
                       source: str = "all", search: str = "",
                       state: Optional[str] = None,
                       min_coverage: int = _AGG_AUTO_MIN_COVERAGE):
    """FB-12/FB-13/FB-R4c: per-dataset rankings + a cross-dataset average-rank aggregate.

    ADDITIVE endpoint — the existing ``/leaderboard`` shape is UNCHANGED (Overview +
    Rankings still share it; FB-12 guard). FB-R4c: SMD/SMAP/MSL collapse into ONE
    "(avg)" column each (mean over per-entity leaves, concat excluded, counted once);
    SWaT default column = excl22 (full stays SELECTABLE). The response advertises the full
    ``selectable_columns`` universe (canonical units + every granular machine/channel,
    concat, SWaT full) + the canonical ``default_columns``; ``datasets=`` picks any subset
    (default = canonical). ``metric`` may be any discovered key (F5).

    E-1 (P0-1): ``min_coverage`` qualifies the headline ``aggregate`` — rows scored on
    fewer than ``min_coverage`` active columns are returned in a SEPARATE ``low_coverage``
    array instead. Omitted ⇒ auto threshold ``max(3, ceil(0.5·active_columns))``;
    ``min_coverage=0`` disables the gate (reproduces the previous coverage-blind set)."""
    return rankings_aggregate_svc(
        get_repository(), metric=metric, score_variant=score_variant,
        datasets=datasets, swat=swat, source=source, search=search, state=state,
        min_coverage=min_coverage)


# ── anomaly types ────────────────────────────────────────────────────────────
@router.get("/experiments/{exp_id}/datasets/{dataset_key}/anomaly-types")
def anomaly_types(exp_id: str, dataset_key: str, variant: Optional[str] = None):
    repo = get_repository()
    exp, ds, leaf = repo.resolve_leaf(exp_id, dataset_key, variant)
    if leaf is None:
        raise HTTPException(404, "leaf not found")
    p = leaf.leaf_path / "anomaly_type_metrics.json"
    if not p.exists():
        return {"available": False, "types": []}
    try:
        atm = parse_anomaly_type_metrics(p)
    except CorruptOrWriting as exc:
        return {"available": False, "error": str(exc), "types": []}
    # FB-7 / R2-06: backend-supplied scoped per-type display_name over the WHOLE
    # discovered type vocabulary (so spike->"Anomaly" fires only single-bucket).
    display = anomaly_type_display_names(list(atm.keys()))
    types = []
    for name, metrics in atm.items():
        resolved = {k: repo.registry.resolve(k, namespace="anomaly_type").model_dump()
                    for k in metrics} if isinstance(metrics, dict) else {}
        types.append({"name": name, "display_name": display.get(name, name),
                      "metrics": metrics, "meta": resolved})
    return {"available": True, "types": types}


# ── NPZ score slabs (lazy / tolerant) ────────────────────────────────────────
@router.get("/experiments/{exp_id}/datasets/{dataset_key}/score-epochs")
def score_epochs(exp_id: str, dataset_key: str, variant: Optional[str] = None):
    repo = get_repository()
    _, _, leaf = repo.resolve_leaf(exp_id, dataset_key, variant)
    if leaf is None:
        raise HTTPException(404, "leaf not found")
    d = leaf.leaf_path / "epoch_scores"
    if not d.is_dir():
        return {"epochs": [], "available": False}
    eps = []
    for f in sorted(d.glob("epoch_*_scores.npz")):
        try:
            eps.append(int(f.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return {"epochs": sorted(eps), "available": bool(eps)}


@router.get("/experiments/{exp_id}/datasets/{dataset_key}/scores/{epoch}")
def scores_epoch(exp_id: str, dataset_key: str, epoch: int,
                 arrays: Optional[str] = None, downsample: int = SETTINGS.default_downsample,
                 page: Optional[int] = None, page_size: Optional[int] = None,
                 variant: Optional[str] = None):
    repo = get_repository()
    _, _, leaf = repo.resolve_leaf(exp_id, dataset_key, variant)
    if leaf is None:
        raise HTTPException(404, "leaf not found")
    npz = leaf.leaf_path / "epoch_scores" / f"epoch_{epoch:03d}_scores.npz"
    if not npz.exists():
        return {"epoch": epoch, "available": False, "reason": "npz_absent"}
    try:
        res = read_arrays(npz, _split_keys(arrays) or None,
                          downsample=downsample, page=page, page_size=page_size)
    except CorruptOrWriting:
        return {"epoch": epoch, "error": "corrupt_or_writing"}  # 200, not 500
    res["epoch"] = epoch
    res["available"] = True
    return res


@router.get("/experiments/{exp_id}/datasets/{dataset_key}/scores/{epoch}/histogram")
def scores_histogram(exp_id: str, dataset_key: str, epoch: int,
                     array: str = "adaptive_score", bins: int = 80,
                     variant: Optional[str] = None):
    repo = get_repository()
    _, _, leaf = repo.resolve_leaf(exp_id, dataset_key, variant)
    if leaf is None:
        raise HTTPException(404, "leaf not found")
    npz = leaf.leaf_path / "epoch_scores" / f"epoch_{epoch:03d}_scores.npz"
    if not npz.exists():
        return {"epoch": epoch, "available": False}
    try:
        res = npz_histogram(npz, array, bins=bins)
    except CorruptOrWriting:
        return {"epoch": epoch, "error": "corrupt_or_writing"}
    res["epoch"] = epoch
    return res


# ── monitoring (tail + per-eval series) ──────────────────────────────────────
@router.get("/experiments/{exp_id}/monitoring")
def monitoring(exp_id: str, as_: str = Query("tail", alias="as"), n: int = 40):
    repo = get_repository()
    exp = repo.find_experiment(exp_id)
    if exp is None:
        raise HTTPException(404, "experiment not found")
    from ..dataaccess.io_safe import read_jsonl

    p = exp.root / "monitoring_log.jsonl"
    if not p.exists():
        return {"available": False, "rows": []}
    try:
        rows = read_jsonl(p)
    except CorruptOrWriting:
        return {"available": False, "error": "corrupt_or_writing"}
    if as_ == "series":
        fields = ["runtime", "latest_pak_f1", "best_pak_f1", "nan_inf_count",
                  "gpu_util", "gpu_mem_mb", "cpu_load1", "ram_used_mb", "judgment_verdict"]
        out = {f: [r.get(f) for r in rows] for f in fields}
        out["available"] = True
        out["n_rows"] = len(rows)
        return out
    return {"available": True, "rows": rows[-n:], "n_rows": len(rows)}


# ── config grouped view + multi-config diff (§5.2) ───────────────────────────
@router.get("/experiments/{exp_id}/datasets/{dataset_key}/config/grouped")
def dataset_config_grouped(exp_id: str, dataset_key: str, variant: Optional[str] = None):
    repo = get_repository()
    exp, ds, leaf = repo.resolve_leaf(exp_id, dataset_key, variant)
    if leaf is None:
        raise HTTPException(404, "leaf not found")
    b = repo.load_bundle(exp, ds, leaf, want={"config", "metadata"})
    return {
        "groups": cmp_svc.group_config(b.config),
        "score_formula": b.metadata.score_formula.model_dump()
        if b.metadata and b.metadata.score_formula else None,
        "errors": b.errors,
    }


@router.post("/config/diff")
def config_diff(body: dict = Body(...)):
    leaves = body.get("leaves", [])
    if not leaves:
        raise HTTPException(400, "body.leaves[] is required (LeafSel list)")
    return cmp_svc.config_diff(get_repository(), leaves)


# ── multi-experiment comparison (§5.6) ───────────────────────────────────────
@router.post("/compare/matrix")
def compare_matrix(body: dict = Body(...)):
    leaves = body.get("leaves", [])
    metrics = body.get("metrics", [])
    if not leaves or not metrics:
        raise HTTPException(400, "body.leaves[] and body.metrics[] are required")
    return cmp_svc.compare_matrix(get_repository(), leaves, metrics,
                                  score_variant=body.get("score_variant", "metrics"))


@router.post("/compare/series")
def compare_series(body: dict = Body(...)):
    leaves = body.get("leaves", [])
    key = body.get("key")
    if not leaves or not key:
        raise HTTPException(400, "body.leaves[] and body.key are required")
    return cmp_svc.compare_series(get_repository(), leaves, key)


# ── per-feature heatmap / PA%K curve / separation (§5.3 / §5.7) ──────────────
@router.get("/experiments/{exp_id}/datasets/{dataset_key}/per-feature")
def per_feature(exp_id: str, dataset_key: str, field: str,
                variant: Optional[str] = None):
    repo = get_repository()
    exp, ds, leaf = repo.resolve_leaf(exp_id, dataset_key, variant)
    if leaf is None:
        raise HTTPException(404, "leaf not found")
    b = repo.load_bundle(exp, ds, leaf, want={"epoch", "metadata"})
    return arr_svc.per_feature(repo, b, field)


@router.get("/experiments/{exp_id}/datasets/{dataset_key}/pak-curve")
def pak_curve(exp_id: str, dataset_key: str, epoch: Optional[int] = None,
              variant: Optional[str] = None):
    repo = get_repository()
    exp, ds, leaf = repo.resolve_leaf(exp_id, dataset_key, variant)
    if leaf is None:
        raise HTTPException(404, "leaf not found")
    b = repo.load_bundle(exp, ds, leaf, want={"epoch"})
    return arr_svc.pak_curve(repo, b, epoch)


@router.get("/experiments/{exp_id}/datasets/{dataset_key}/separation")
def separation(exp_id: str, dataset_key: str, variant: Optional[str] = None):
    repo = get_repository()
    exp, ds, leaf = repo.resolve_leaf(exp_id, dataset_key, variant)
    if leaf is None:
        raise HTTPException(404, "leaf not found")
    b = repo.load_bundle(exp, ds, leaf, want={"metadata"})
    return arr_svc.separation(repo, b)


# ── train-split NPZ scores (optional; absent on excl22) (§5.8) ────────────────
@router.get("/experiments/{exp_id}/datasets/{dataset_key}/train-scores")
def train_scores(exp_id: str, dataset_key: str,
                 arrays: Optional[str] = None,
                 downsample: int = SETTINGS.default_downsample,
                 variant: Optional[str] = None):
    repo = get_repository()
    _, _, leaf = repo.resolve_leaf(exp_id, dataset_key, variant)
    if leaf is None:
        raise HTTPException(404, "leaf not found")
    npz = leaf.leaf_path / "best_epoch_train_scores.npz"
    if not npz.exists():
        return {"available": False}              # excl22 shares no train NPZ (G4)
    try:
        res = read_arrays(npz, _split_keys(arrays) or None, downsample=downsample)
    except CorruptOrWriting:
        return {"error": "corrupt_or_writing"}
    res["available"] = True
    return res


# ── viz manifest + telemetry + profiling + detailed-windows (§5.9 / §5.10) ────
@router.get("/experiments/{exp_id}/datasets/{dataset_key}/viz-manifest")
def viz_manifest(exp_id: str, dataset_key: str, variant: Optional[str] = None):
    repo = get_repository()
    exp, ds, leaf = repo.resolve_leaf(exp_id, dataset_key, variant)
    if leaf is None:
        raise HTTPException(404, "leaf not found")
    from ..gif.service import get_gif_service

    gif_idx = get_gif_service().index()
    return viz_manifest_svc(leaf, exp_id=exp_id, dataset_key=dataset_key,
                            gif_index=gif_idx)


@router.get("/experiments/{exp_id}/telemetry")
def telemetry(exp_id: str):
    repo = get_repository()
    exp = repo.find_experiment(exp_id)
    if exp is None:
        raise HTTPException(404, "experiment not found")
    p = exp.root / "telemetry" / "snapshots.csv"
    if not p.is_file():
        return {"available": False}
    try:
        import pandas as pd

        df = pd.read_csv(p)
    except Exception as exc:  # noqa: BLE001 - tolerate a half-written/odd CSV
        return {"available": False, "error": str(exc)}
    df = df.where(pd.notnull(df), None)
    return {"available": True, "columns": list(df.columns),
            "series": {c: df[c].tolist() for c in df.columns}, "n_rows": len(df)}


@router.get("/experiments/{exp_id}/profiling/{dataset_key}")
def profiling(exp_id: str, dataset_key: str, variant: Optional[str] = None):
    repo = get_repository()
    exp, ds, leaf = repo.resolve_leaf(exp_id, dataset_key, variant)
    if leaf is None:
        raise HTTPException(404, "leaf not found")
    out: dict[str, Any] = {"available": False}
    bp = leaf.leaf_path / "batch_profiling.json"
    if bp.is_file():
        try:
            out = {"available": True, "batch_profiling": read_json(bp)}
        except CorruptOrWriting:
            out = {"available": False, "error": "corrupt_or_writing"}
    b = repo.load_bundle(exp, ds, leaf, want={"metadata"})
    if b.metadata:
        out["timing"] = b.metadata.timing
    return out


@router.get("/experiments/{exp_id}/datasets/{dataset_key}/detailed-windows")
def detailed_windows(exp_id: str, dataset_key: str, top: int = 50,
                     sort: Optional[str] = None, variant: Optional[str] = None):
    repo = get_repository()
    exp, ds, leaf = repo.resolve_leaf(exp_id, dataset_key, variant)
    if leaf is None:
        raise HTTPException(404, "leaf not found")
    p = leaf.leaf_path / "best_model_detailed.csv"
    if not p.is_file():
        return {"available": False}
    try:
        import pandas as pd

        df = pd.read_csv(p)
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": str(exc)}
    sample_size = len(df)
    if sort and sort in df.columns:
        df = df.sort_values(sort, ascending=False)
    df = df.head(top).where(lambda x: x.notnull(), None)
    return {"available": True, "is_sample": True, "sample_size": sample_size,
            "columns": list(df.columns), "rows": df.to_dict(orient="records")}


# ── FB-NEW insight endpoints (B1/B2/B3/B5 — additive, lazy, tolerant) ─────────
@router.get("/experiments/{exp_id}/datasets/{dataset_key}/insights/score-decomposition")
def insight_score_decomposition(exp_id: str, dataset_key: str,
                                variant: Optional[str] = None):
    """B1: score-component decomposition (recon vs disc remainder) from the best-epoch
    NPZ's own arrays + a real disc-consistency gate (the empirical remainder
    cross-checked against the NPZ's own discrepancy_error — RV2-01). Read-cheap/lazy/
    tolerant; never *.pt."""
    repo = get_repository()
    exp, ds, leaf = repo.resolve_leaf(exp_id, dataset_key, variant)
    if leaf is None:
        raise HTTPException(404, "leaf not found")
    b = repo.load_bundle(exp, ds, leaf, want={"metadata"})
    return insight_svc.score_decomposition(repo, b)


@router.get("/experiments/{exp_id}/datasets/{dataset_key}/insights/divergence")
def insight_divergence(exp_id: str, dataset_key: str, metric: str = "pak_auc_f1",
                       variant: Optional[str] = None):
    """B2: teacher↔student divergence trajectory (adaptive vs teacher per-epoch +
    4 best-epoch score-variant scalars; masked pre-warmup)."""
    repo = get_repository()
    exp, ds, leaf = repo.resolve_leaf(exp_id, dataset_key, variant)
    if leaf is None:
        raise HTTPException(404, "leaf not found")
    b = repo.load_bundle(exp, ds, leaf, want={"epoch", "metadata"})
    return insight_svc.divergence_trajectory(repo, b, metric)


@router.get("/experiments/{exp_id}/datasets/{dataset_key}/insights/warmup-delta")
def insight_warmup_delta(exp_id: str, dataset_key: str, metric: str = "pak_auc_f1",
                         variant: Optional[str] = None):
    """B3: pre↔post-warmup best delta (best pak_auc_f1 in ep≤warmup vs ep>warmup)."""
    repo = get_repository()
    exp, ds, leaf = repo.resolve_leaf(exp_id, dataset_key, variant)
    if leaf is None:
        raise HTTPException(404, "leaf not found")
    b = repo.load_bundle(exp, ds, leaf, want={"epoch", "metadata"})
    return insight_svc.warmup_delta(repo, b, metric)


@router.get("/experiments/{exp_id}/datasets/{dataset_key}/insights/score-distribution")
def insight_score_distribution(exp_id: str, dataset_key: str,
                               epoch: Optional[int] = None,
                               array: str = "adaptive_score", bins: int = 80,
                               variant: Optional[str] = None):
    """B5: normal-vs-anomaly score distribution (histogram from NPZ point_labels
    split at the best epoch by default). Lazy NPZ, tolerant; never *.pt."""
    repo = get_repository()
    exp, ds, leaf = repo.resolve_leaf(exp_id, dataset_key, variant)
    if leaf is None:
        raise HTTPException(404, "leaf not found")
    b = repo.load_bundle(exp, ds, leaf, want={"metadata"})
    return insight_svc.score_distribution(repo, b, epoch=epoch, array=array, bins=bins)


@router.get("/experiments/{exp_id}/datasets/{dataset_key}/insights/component-trajectories")
def insight_component_trajectories(exp_id: str, dataset_key: str,
                                   metric: str = "roc", pak: bool = True,
                                   variant: Optional[str] = None):
    """E-3 (P0-5 / I1+I2): per-eval-epoch RECOMPUTED detection trajectories (numpy-only
    ROC-AUC + Average-Precision + a torch-free PA%K best-F1) for three channels —
    adaptive (production), teacher-only (recon ablation), disc-only (discrepancy
    ablation) — vs point_labels, read from the saved NPZs. Returns the 3 trajectories
    (gaps where no NPZ; P3-01 raw-array ablation, NOT a fabricated adaptive series), the
    warmup boundary, a self-validation of the adaptive recompute vs the stored
    metrics/{roc_auc,prc_auc}, and the I2 disc-efficacy verdict. numpy-only (no
    sklearn/scipy, no mae_anomaly import, no *.pt); cached on _INSIGHT_VERSION (R-06)."""
    repo = get_repository()
    exp, ds, leaf = repo.resolve_leaf(exp_id, dataset_key, variant)
    if leaf is None:
        raise HTTPException(404, "leaf not found")
    b = repo.load_bundle(exp, ds, leaf, want={"metadata"})
    return insight_svc.component_trajectories(repo, b, metric=metric, want_pak=pak)


@router.get("/experiments/{exp_id}/datasets/{dataset_key}/insights/component-overlay")
def insight_component_overlay(exp_id: str, dataset_key: str,
                              epoch: Optional[int] = None, downsample: int = 4000,
                              variant: Optional[str] = None):
    """E-4 (I3): score-vs-time component overlay — teacher_recon_error vs
    discrepancy_error vs adaptive_score on the TEST time axis at a chosen epoch (best
    epoch by default), plus point_labels + derived anomaly_regions. Reuses the lazy
    read_arrays with downsample (read-cheap, one NPZ). Tolerant; never *.pt."""
    repo = get_repository()
    exp, ds, leaf = repo.resolve_leaf(exp_id, dataset_key, variant)
    if leaf is None:
        raise HTTPException(404, "leaf not found")
    b = repo.load_bundle(exp, ds, leaf, want={"metadata"})
    return insight_svc.component_overlay(repo, b, epoch=epoch, downsample=downsample)


# ── EXT-6 computed-panel plugins (§8) ────────────────────────────────────────
@router.get("/panels")
def panels():
    return {"panels": panel_svc.list_panels()}


@router.get("/experiments/{exp_id}/datasets/{dataset_key}/panels/{name}")
def run_panel(exp_id: str, dataset_key: str, name: str,
              variant: Optional[str] = None):
    repo = get_repository()
    panel = panel_svc.get_panel(name)
    if panel is None:
        raise HTTPException(404, f"panel {name!r} not registered")
    exp, ds, leaf = repo.resolve_leaf(exp_id, dataset_key, variant)
    if leaf is None:
        raise HTTPException(404, "leaf not found")
    b = repo.load_bundle(exp, ds, leaf, want=panel_svc.required_kinds(name))
    try:
        result = panel.compute(repo, b, {})
    except Exception as exc:  # noqa: BLE001 - isolate panel failures
        return {"available": False, "error": str(exc)}
    return {"panel": name, "result": result, "errors": b.errors}


# ── CSV export (§5.11) ───────────────────────────────────────────────────────
@router.get("/export/series.csv", response_class=PlainTextResponse)
def export_series_csv(exp_id: str, dataset_key: str,
                      keys: Optional[str] = None, variant: Optional[str] = None):
    repo = get_repository()
    exp, ds, leaf = repo.resolve_leaf(exp_id, dataset_key, variant)
    if leaf is None:
        raise HTTPException(404, "leaf not found")
    b = repo.load_bundle(exp, ds, leaf, want={"epoch", "metadata"})
    res = epoch_series(repo, b, _split_keys(keys), include_meta=False)
    csv_text = export_svc.series_to_csv(res.get("epochs", []), res.get("series", {}))
    return PlainTextResponse(csv_text, media_type="text/csv")


@router.post("/export/matrix.csv", response_class=PlainTextResponse)
def export_matrix_csv(body: dict = Body(...)):
    leaves = body.get("leaves", [])
    metrics = body.get("metrics", [])
    if not leaves or not metrics:
        raise HTTPException(400, "body.leaves[] and body.metrics[] are required")
    res = cmp_svc.compare_matrix(get_repository(), leaves, metrics,
                                 score_variant=body.get("score_variant", "metrics"))
    return PlainTextResponse(export_svc.matrix_to_csv(res["rows"], metrics),
                             media_type="text/csv")
