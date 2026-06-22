"""FastAPI application entrypoint — backend-design.md §1.3.

Single uvicorn process: serves ``/api/*`` AND (when built) the frontend ``dist/``
with an SPA history-fallback so React-Router deep links resolve. The registry
resolver self-test runs at startup as a loud gate (``/api/health`` goes degraded on
FAIL); the server still serves data so the verifier can inspect the failure.

Run (from the project root, in the isolated venv):
    UI/.venv/bin/uvicorn app.main:app --app-dir UI/backend --host 127.0.0.1 --port 8000
or:
    UI/.venv/bin/python -m uvicorn app.main:app --app-dir UI/backend
"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .api.gif_routes import router as gif_router
from .api.routes import router as api_router
from .config import SETTINGS
from .dataaccess.io_safe import confine
from .dataaccess.repository import get_repository

log = logging.getLogger("tsmae.dashboard")

app = FastAPI(
    title="TSMAE Experiment Dashboard — backend",
    version="0.2.0",
    description="Read-only, CPU-only data/compute/GIF API over results/experiments/ "
                "(+ .trash fixtures). Never touches dc_vis, the GPU, or *.pt.",
)
app.include_router(api_router)
app.include_router(gif_router)

# Hashed frontend bundles under dist/assets/ are served by a dedicated mount so the
# SPA history-fallback (below) only ever returns index.html for non-asset deep links.
_ASSETS = SETTINGS.frontend_dist / "assets"
if _ASSETS.is_dir():
    app.mount("/assets", StaticFiles(directory=str(_ASSETS)), name="assets")


@app.on_event("startup")
def _startup_gate() -> None:
    """Run the registry resolver self-test as a startup gate (§3.3)."""
    repo = get_repository()
    st = repo.selftest
    if st["status"] != "PASS":
        log.error("RESOLVER SELF-TEST FAILED — /api/health is degraded: %s", st["failures"])
    else:
        log.info("resolver self-test PASS (registry v%s); discovered %d experiments",
                 repo.registry.version, len(repo.experiments()))
    # No eager perf-store warm: the precomputed sidecars + the transient (non-cached)
    # epoch/history parse keep even a COLD aggregate at ~370 ms and the steady state at
    # ~280 MB RSS. An eager background warm running CONCURRENTLY with the page's first
    # burst of requests drove a transient multi-GB allocation peak that Python never
    # returned to the OS (2026-06-16). Lazy load on first request is both fast and lean.


# ── static PNG serve (path-confined to the source roots; §9) ─────────────────
@app.get("/api/files/png")
def serve_png(path: str = Query(...)):
    try:
        confined = confine(Path(path), *SETTINGS.source_roots.values())
    except ValueError:
        raise HTTPException(403, "path escapes the allowed roots")
    if not confined.is_file() or confined.suffix.lower() != ".png":
        raise HTTPException(404, "not a PNG under the allowed roots")
    return FileResponse(confined, media_type="image/png")


# ── SPA history-fallback (§1.3) — registered LAST, after /api/* ───────────────
_DIST = SETTINGS.frontend_dist


@app.get("/{full_path:path}")
def spa_fallback(full_path: str):
    """Serve the built frontend; deep links fall back to index.html.

    404s anything under /api (the API router owns those). If the frontend has not
    been built yet, returns a friendly placeholder so the backend is usable alone.
    """
    if full_path.startswith("api/") or full_path == "api":
        raise HTTPException(404, "unknown API route")
    if _DIST.is_dir():
        candidate = _DIST / full_path
        if full_path and candidate.is_file():
            try:
                confine(candidate, _DIST)
                return FileResponse(candidate)
            except ValueError:
                raise HTTPException(403, "path escape")
        index = _DIST / "index.html"
        if index.is_file():
            return FileResponse(index)
    return Response(
        "<html><body style='font-family:sans-serif;padding:2rem'>"
        "<h2>TSMAE Dashboard backend is running.</h2>"
        "<p>The frontend build (<code>UI/frontend/dist/</code>) is not present yet. "
        "The API is live at <a href='/api/health'>/api/health</a> and "
        "<a href='/docs'>/docs</a>.</p></body></html>",
        media_type="text/html",
    )
