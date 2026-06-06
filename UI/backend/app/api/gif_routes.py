"""GIF rendering endpoints — backend-design.md §6.1.

  GET  /api/gif/stories            the STORY_REGISTRY manifest (FE-FB-3, P2-04)
  POST /api/gif/render             render a GifSpec -> {job_id, cache_key, cached}
  GET  /api/gif/status/{job_id}    async render progress
  GET  /api/gif/{cache_key}.gif    serve the cached GIF (read-only)
  GET  /api/gif/list               cached-GIF index (FE-FB-5)

CPU-only (matplotlib Agg + imageio, no ffmpeg). Cache + sidecars live under
``UI/.cache/gif/`` — never under results/.
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..gif.service import get_gif_service
from ..gif.stories import STORY_IDS

router = APIRouter(prefix="/api/gif")


class LeafSel(BaseModel):
    exp_id: str
    dataset_key: str
    variant: Optional[str] = None


class GifSpec(BaseModel):
    story: str
    metric_keys: list[str] = Field(default_factory=list)
    experiment_set: list[LeafSel] = Field(default_factory=list)
    variant: Optional[str] = None
    max_epoch: Optional[int] = None
    params: dict[str, Any] = Field(default_factory=dict)


@router.get("/stories")
def gif_stories():
    return get_gif_service().stories()


@router.get("/loss-drift-metrics")
def gif_loss_drift_metrics(exp_id: str, dataset_key: str,
                           variant: Optional[str] = None):
    """FB-14: the full grouped inventory of per-training-epoch ``history`` series for a
    leaf, so the GIF-3 (loss_drift) picker can surface ALL collected training metrics
    (teacher/student recon, discrepancy, SNR/ratios, per-type) grouped by family. The
    raw catalog/history endpoints are unchanged; this is an ADDITIVE convenience."""
    return get_gif_service().loss_drift_metrics(
        {"exp_id": exp_id, "dataset_key": dataset_key, "variant": variant})


@router.get("/list")
def gif_list(leaf: Optional[str] = None):
    return {"gifs": get_gif_service().index(leaf)}


@router.post("/render")
def gif_render(spec: GifSpec = Body(...)):
    if spec.story not in STORY_IDS:
        raise HTTPException(400, f"unknown story {spec.story!r}; "
                                 f"valid: {sorted(STORY_IDS)}")
    svc = get_gif_service()
    # normalize the spec to a plain dict the service consumes.
    payload = {
        "story": spec.story,
        "metric_keys": spec.metric_keys,
        "experiment_set": [s.model_dump() for s in spec.experiment_set],
        "variant": spec.variant,
        "max_epoch": spec.max_epoch,
        "params": spec.params,
    }
    return svc.render_spec(payload)


@router.get("/status/{job_id}")
def gif_status(job_id: str):
    st = get_gif_service().status(job_id)
    if st is None:
        raise HTTPException(404, "job not found (it may have completed and been GC'd)")
    return st


@router.get("/sidecar/{cache_key}")
def gif_sidecar(cache_key: str):
    """R3-SR-03: the rendered GIF's sidecar — the per-frame eval-epoch schedule
    (``frame_epochs``: frame index → eval epoch) + the peak-hold tail length
    (``hold_len``) + ``n_frames``. Lets the canvas player label/seek by epoch and
    collapse the trailing hold frames to the final epoch. Pure read; used for the
    instant cache-hit path (POST /render returned ``cached:true`` with no job to poll).
    404 if the GIF/sidecar is absent (render first)."""
    sc = get_gif_service().read_sidecar(cache_key)
    if sc is None:
        raise HTTPException(404, "sidecar not found (render the GIF first)")
    return sc


@router.get("/{cache_key}.gif")
def gif_file(cache_key: str):
    p = get_gif_service().gif_path(cache_key)
    if not p.is_file():
        raise HTTPException(404, "gif not rendered (render first via POST /api/gif/render)")
    return FileResponse(p, media_type="image/gif")
