"""GIF service orchestration — cache key, data collection, async jobs, cache index.

backend-design.md §6.1/§6.4/§6.5. Renders the 6 stories from the STORY_REGISTRY,
honoring the P3 constraints:
  * P3-01 — GIF-6 (``what_carries``) plots adaptive(``metrics``)+``teacher_*`` as
    per-epoch lines and the 4 score variants as best-epoch markers; NO fabricated
    per-epoch student/disc lines.
  * P3-02 — GIF-5 (``signal_lives``) ``anom_type_over_training`` binds the 2-level
    ``pivot[type][sub_metric]`` (default ``recon_ratio``).

Cache key = sha1(story_id, sorted metric_keys, canonical experiment_set, max_epoch,
canonical params, max source mtime over the involved files). A live append changes the
source mtime ⇒ the GIF is invalidated. Files live under ``UI/.cache/gif/`` (never
under results/); each carries a ``<cache_key>.json`` sidecar for the ``/gif/list``
index + ``viz-manifest`` cross-link (FE-FB-5).

Rendering runs in a bounded ThreadPool (§1.3) so a heavy render never blocks the
read-cheap endpoints.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

from ..config import SETTINGS
from ..dataaccess.io_safe import CorruptOrWriting
from ..dataaccess.npz_reader import histogram as npz_histogram
from ..dataaccess.repository import LeafBundle, Repository
from . import render
from .stories import STORY_REGISTRY, get_story

# RV2-02: render-code version baked into the GIF cache key. The cache key otherwise
# carries only source mtimes, so a render-CODE change (FB-9 loop=0→1, FB-15 DPI/size)
# does NOT change the key for an unchanged source file ⇒ a stale GIF (still looping /
# low-DPI) would keep being served. Bump this whenever the GIF rendering OUTPUT changes
# so all stale cached GIFs are deterministically invalidated.
#   v2 — FB-9 loop=1 (non-looping) + FB-15 DPI 90→140 / size (7.2,4.3)→(8.0,4.8).
#   v3 — FB-R3-09: per-eval-epoch stories frame ONE frame per saved eval epoch (5-step),
#        no implicit 2× stride (was every 10 epochs). Bumping invalidates the stale
#        10-step GIFs so the 5-step grid is actually re-rendered + served.
#   v4 — FB-R4-01: the HIGH-FRAME-COUNT per-eval-epoch line/race/grid stories now render
#        at the lighter PLAYBACK profile (96 DPI / 7.0×4.2in ≈ 672×403 px) so a ~110-frame
#        GIF renders ~2× faster on the server AND is ~2.8× lighter to decode/composite on
#        the client — frame COUNT is unchanged (every distinct eval epoch is still its own
#        frame, sidecar frame_epochs + 5-step seek granularity preserved). The GIF BYTES
#        change ⇒ bump so the stale 140-DPI per-epoch GIFs are deterministically
#        invalidated and the lighter ones are re-rendered + served.
#   v5 — FB-R4b + R4 carried nits. GIF-7 (what_carries) markers change: the adaptive +
#        teacher ceilings are now max(per-epoch series) drawn as SOLID rules (a TRUE
#        ceiling the line never exceeds), student/disc are best-epoch SCALAR markers
#        labeled "<variant> best-epoch" (dashed + diamond), and ``params.show_ceilings``
#        toggles which markers are drawn. R4-F-01: loss_drift (npz_free) now renders at the
#        lighter playback profile (96 DPI). R4-F-02: the loss_drift contribution-share axis
#        label is data-driven (recon/total vs disc/total). All change GIF BYTES ⇒ bump.
#   v6 — FB-R5-01: the signal_lives feature_heatmap now applies a configurable color NORM
#        (params.heatmap_scale ∈ {linear, sqrt, log}; default sqrt = PowerNorm γ=0.5) so a
#        heavy-tailed per-feature recon distribution no longer washes out (the ≈0.001 bulk
#        renders as a clearly non-black color distinct from the floor, not "no influence").
#        The p2/p98 clip + magma cmap are unchanged; only value→color changes. heatmap_scale
#        is in params ⇒ each scale caches independently (params already key the cache); the
#        DEFAULT sqrt path changes the bytes vs the old linear render ⇒ bump so the stale
#        linear-washout GIFs are deterministically invalidated and re-rendered.
#   v8 — E-2 (P0-4): GIF-7 (what_carries) near-coincident ceiling/best-epoch LABELS are
#        now vertically STAGGERED (deterministic stack + faint leader + a light text bbox)
#        so they never overlap into garbled text; and metric_diff (GIF-8) renders an
#        explicit explanatory NOTE when the A−B diff is all-gaps (best-epoch-only B) instead
#        of an empty/blank GIF, with a SIGNED "Δ … (A−B)" y-axis that straddles zero (never
#        the [0,1] score clamp). Both change GIF BYTES ⇒ bump so the stale v7 GIFs are
#        deterministically invalidated and the corrected renders are served.
_RENDER_VERSION = 8   # E-2: GIF-7 staggered labels + metric_diff all-gaps note / signed axis

# FB-R3-09: the per-eval-epoch stories (climb_plateau, warmup_join, what_carries,
# line_race, bar_race, compare_grid, and the loss_drift NPZ-FREE training-epoch path)
# must place ONE frame per saved eval epoch (eval_interval=5 → epochs 5,10,15,…), NOT
# stride to every 10. We achieve this by giving these stories a frame budget that is
# UNCAPPED (every point is a frame) up to a generous safety ceiling, so a pathological
# very-long run still cannot render thousands of frames. The NPZ hist-drift
# (``_frames_npz_drift``) and feature-heatmap paths KEEP their own stride caps — they
# bound NPZ loads / per-frame memory and must not be relaxed.
_PER_EPOCH_FRAME_CEILING = 150
_PER_EPOCH_STORIES = frozenset({
    "climb_plateau", "warmup_join", "what_carries",
    "line_race", "bar_race", "compare_grid",
    # FB-R4-02: the metric-diff story is also a per-eval-epoch line (one frame per saved
    # eval epoch) ⇒ uncapped frame budget + the lighter playback DPI profile.
    "metric_diff",
})

# per-story peak-hold tail length (must match the renderer ``hold=`` defaults in
# render.py so the sidecar's frame→epoch schedule is bit-accurate, R3-SR-03).
_STORY_HOLD = {
    "climb_plateau": 10, "warmup_join": 10, "what_carries": 10,
    "loss_drift": 10, "signal_lives": 8,
    "line_race": 12, "bar_race": 12, "compare_grid": 10,
    "metric_diff": 10,
}


def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


def _max_source_mtime(paths: list[Path]) -> int:
    m = 0
    for p in paths:
        try:
            m = max(m, p.stat().st_mtime_ns)
        except OSError:
            continue
    return m


class GifJob:
    __slots__ = ("job_id", "cache_key", "status", "frame", "n_frames", "error", "spec")

    def __init__(self, job_id: str, cache_key: str, spec: dict):
        self.job_id = job_id
        self.cache_key = cache_key
        self.status = "pending"          # pending|rendering|done|error
        self.frame = 0
        self.n_frames = 0
        self.error: Optional[str] = None
        self.spec = spec

    def as_dict(self) -> dict:
        return {"job_id": self.job_id, "cache_key": self.cache_key,
                "status": self.status, "frame": self.frame,
                "n_frames": self.n_frames, "error": self.error}


class GifService:
    def __init__(self, repo: Repository):
        self.repo = repo
        self.cache_dir = SETTINGS.cache_dir("gif")
        self._pool = ThreadPoolExecutor(max_workers=SETTINGS.gif_workers,
                                        thread_name_prefix="gif")
        self._jobs: dict[str, GifJob] = {}
        self._lock = threading.Lock()

    # ── stories manifest ──────────────────────────────────────────────────────
    def stories(self) -> dict[str, Any]:
        # FB-R4b-01: advertise the togglable GIF-7 ceiling/marker variants so the picker
        # can render one checkbox per variant (adaptive/teacher carry a TRUE ceiling;
        # student/disc are best-epoch scalars only). The render param is
        # ``params.show_ceilings`` (list of variant keys; default = all).
        return {
            "stories": STORY_REGISTRY,
            "what_carries_ceilings": self.what_carries_ceiling_variants(),
            # FB-R5-01: advertise the signal_lives feature_heatmap color scales so the
            # GIF-Studio selector is never a hard-coded literal. ``default`` is the
            # faithful sqrt (PowerNorm γ=0.5); the param is ``params.heatmap_scale``.
            "signal_lives_heatmap_scales": {
                "scales": list(render.HEATMAP_SCALES),
                "default": render.DEFAULT_HEATMAP_SCALE,
                "sub_mode": "feature_heatmap",
                "param": "heatmap_scale",
            },
        }

    # ── FB-14: GIF-3 / loss_drift selectable history families (grouped) ─────────
    def loss_drift_metrics(self, sel: dict) -> dict[str, Any]:
        """List EVERY populated per-training-epoch ``history`` series for one leaf,
        grouped by registry family, so the GIF-3 picker can surface ALL collected
        training metrics (teacher/student recon, discrepancy, SNR/ratios, per-type),
        not just the default separation pair (FB-14).

        Pure read + resolver; an empty/disabled series (SCAD/discriminator off) is
        reported under ``disabled`` with its family so the picker shows a
        "component off" chip rather than crashing. The list is a runtime UNION of
        ``history.series`` (F5) — never a hard-coded enumeration."""
        exp, ds, leaf = self.repo.resolve_leaf(
            sel.get("exp_id", ""), sel.get("dataset_key", ""), sel.get("variant"))
        if leaf is None:
            return {"available": False, "families": {}, "disabled": [],
                    "default_metric_keys": get_story("loss_drift")["default_metric_keys"]}
        b = self.repo.load_bundle(exp, ds, leaf, want={"history"})
        if b.history is None:
            return {"available": False, "families": {}, "disabled": [],
                    "default_metric_keys": get_story("loss_drift")["default_metric_keys"]}
        families: dict[str, list[dict[str, Any]]] = {}
        for k in sorted(b.history.series.keys()):
            meta = self.repo.registry.resolve(k, namespace="history").model_dump()
            families.setdefault(meta["family"], []).append({
                "key": k, "display_name": meta["display_name"],
                "direction": meta["direction"], "phase_validity": meta["phase_validity"],
                "inferred": meta["inferred"],
                "populated": bool(b.history.series.get(k)),
            })
        disabled = []
        for k in sorted(b.history.empty_keys):
            meta = self.repo.registry.resolve(k, namespace="history").model_dump()
            disabled.append({"key": k, "display_name": meta["display_name"],
                             "family": meta["family"], "reason": "component_disabled"})
        return {
            "available": True,
            "families": families,
            "disabled": disabled,
            "default_metric_keys": get_story("loss_drift")["default_metric_keys"],
            "nested": sorted(b.history.nested.keys()),
        }

    # ── cache key + index ─────────────────────────────────────────────────────
    def _source_paths(self, spec: dict) -> list[Path]:
        paths: list[Path] = []
        for sel in spec.get("experiment_set", []):
            exp, ds, leaf = self.repo.resolve_leaf(
                sel.get("exp_id", ""), sel.get("dataset_key", ""), sel.get("variant"))
            if leaf is None:
                continue
            for name in ("epoch_metrics.json", "experiment_metadata.json",
                         "training_histories.json"):
                p = leaf.leaf_path / name
                if p.exists():
                    paths.append(p)
            esd = leaf.leaf_path / "epoch_scores"
            if esd.is_dir():
                paths.append(esd)
        return paths

    def cache_key(self, spec: dict) -> str:
        payload = _canon({
            "story": spec.get("story"),
            "metric_keys": sorted(spec.get("metric_keys", [])),
            "experiment_set": spec.get("experiment_set", []),
            "max_epoch": spec.get("max_epoch"),
            "params": spec.get("params", {}),
            "src_mtime": _max_source_mtime(self._source_paths(spec)),
            # RV2-02: invalidate stale GIFs when the render OUTPUT changes (loop/DPI/size)
            # even though the source files did not — see ``_RENDER_VERSION``.
            "render_version": _RENDER_VERSION,
        })
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]

    def gif_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.gif"

    def _sidecar_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.json"

    def _write_sidecar(self, cache_key: str, spec: dict, n_frames: int, *,
                       frame_epochs: Optional[list[int]] = None,
                       hold_len: int = 0) -> None:
        sc = {
            "cache_key": cache_key,
            "story": spec.get("story"),
            "story_id": get_story(spec["story"])["story_id"],
            "metric_keys": spec.get("metric_keys", []),
            "experiment_set": spec.get("experiment_set", []),
            "variant": spec.get("variant"),
            "max_epoch": spec.get("max_epoch"),
            "n_frames": n_frames,
            # R3-SR-03: the eval-epoch shown on each rendered frame (frame index → epoch
            # for the canvas player's seek/label) + the peak-hold tail length so the
            # frontend can collapse the trailing duplicate-final-epoch frames to one stop.
            # Empty for npz/heatmap stories (they label epochs on-frame).
            "frame_epochs": frame_epochs or [],
            "hold_len": int(hold_len),
            "mtime": time.time(),
        }
        self._atomic_write(self._sidecar_path(cache_key),
                           json.dumps(sc).encode("utf-8"))

    @staticmethod
    def _atomic_write(target: Path, payload: bytes) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
        os.replace(tmp, target)

    def read_sidecar(self, cache_key: str) -> Optional[dict]:
        """Read one cached GIF's sidecar (frame→epoch schedule, hold tail, n_frames, …)
        by cache_key. Pure read; ``None`` if the GIF/sidecar is absent. Powers the
        canvas player's seek-by-epoch (R3-SR-03) for BOTH the async-render and the
        instant cache-hit paths."""
        sc = self._sidecar_path(cache_key)
        if not sc.is_file() or not self.gif_path(cache_key).is_file():
            return None
        try:
            return json.loads(sc.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def index(self, leaf_filter: Optional[str] = None) -> list[dict]:
        """Read the cached-GIF index (the ``<cache_key>.json`` sidecars). Pure read."""
        out: list[dict] = []
        for sc in sorted(self.cache_dir.glob("*.json")):
            try:
                d = json.loads(sc.read_text("utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not self.gif_path(d.get("cache_key", "")).exists():
                continue
            if leaf_filter:
                # P4B-02: exact leaf_id membership (mirrors viz_manifest cross-link),
                # NOT a loose startswith/contains that can over-match. The canonical
                # leaf_id is ``exp_id|dataset_key|variant``; accept the ``:``-separated
                # and ``~``-encoded forms too and normalise them all to leaf_id.
                want = leaf_filter.replace("~", "/").replace(":", "|")
                leaf_ids = {
                    f"{s.get('exp_id')}|{s.get('dataset_key')}|{s.get('variant') or '_'}"
                    for s in d.get("experiment_set", [])
                }
                if want not in leaf_ids:
                    continue
            out.append(d)
        return out

    # ── render entrypoint ─────────────────────────────────────────────────────
    def render_spec(self, spec: dict) -> dict:
        """Return {job_id, cache_key, cached}. If cached, served instantly; else a job
        is queued in the ThreadPool and progress is polled via ``status``."""
        ck = self.cache_key(spec)
        if self.gif_path(ck).exists():
            return {"job_id": None, "cache_key": ck, "cached": True}
        job_id = hashlib.sha1(f"{ck}{time.time()}".encode()).hexdigest()[:12]
        job = GifJob(job_id, ck, spec)
        with self._lock:
            self._jobs[job_id] = job
        self._pool.submit(self._run_job, job)
        return {"job_id": job_id, "cache_key": ck, "cached": False}

    def render_sync(self, spec: dict) -> dict:
        """Render in the calling thread (used by the smoke test). Returns the result."""
        ck = self.cache_key(spec)
        if self.gif_path(ck).exists():
            return {"cache_key": ck, "cached": True, "path": str(self.gif_path(ck))}
        job = GifJob("sync", ck, spec)
        self._render(job)
        return {"cache_key": ck, "cached": False, "path": str(self.gif_path(ck)),
                "n_frames": job.n_frames, "status": job.status, "error": job.error}

    def status(self, job_id: str) -> Optional[dict]:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            return None
        out = job.as_dict()
        # R3-SR-03: once done, surface the frame→epoch schedule + peak-hold tail so the
        # canvas player can label/seek by eval epoch without a second round-trip.
        if job.status == "done":
            sc = self.read_sidecar(job.cache_key)
            if sc is not None:
                out["frame_epochs"] = sc.get("frame_epochs", [])
                out["hold_len"] = sc.get("hold_len", 0)
        return out

    def _run_job(self, job: GifJob) -> None:
        try:
            self._render(job)
        except Exception as exc:  # noqa: BLE001 - isolate render failures
            job.status = "error"
            job.error = str(exc)

    # ── per-story rendering ───────────────────────────────────────────────────
    def _story_max_frames(self, story: str, spec: dict) -> Optional[int]:
        """FB-R3-09: the effective frame budget for a story.

        Per-eval-epoch line/race/compare stories (``_PER_EPOCH_STORIES``) — and the
        loss_drift NPZ-FREE training-epoch path — get an UNCAPPED budget (the generous
        ``_PER_EPOCH_FRAME_CEILING``) so EVERY saved eval epoch is its own frame (5-step,
        not every 10). Everything else keeps the default 60 cap. The NPZ hist-drift +
        feature-heatmap paths read this default and KEEP their own stride caps (bounded
        NPZ loads / per-frame memory) — those are NOT in ``_PER_EPOCH_STORIES``.
        """
        params = spec.get("params", {})
        default = params.get("max_frames", 60)
        if story in _PER_EPOCH_STORIES:
            return _PER_EPOCH_FRAME_CEILING
        if story == "loss_drift" and params.get("sub_mode", "npz_free") != "npz_hist":
            # NPZ-free separation drift over the training-epoch axis is a per-saved-epoch
            # line story too; the npz_hist sub-mode keeps the default (its NPZ list is
            # strided in ``_frames_npz_drift`` to bound file loads).
            return _PER_EPOCH_FRAME_CEILING
        return default

    def _playback_kwargs(self, story: str, spec: Optional[dict] = None) -> dict:
        """FB-R4-01: the lighter (dpi, figsize) for the HIGH-FRAME-COUNT per-eval-epoch
        line/race/grid stories (``_PER_EPOCH_STORIES``) so a ~110-frame GIF renders fast
        on the server AND stays light to decode/composite on the client — WITHOUT dropping
        any frame (frame count = one per distinct eval epoch is unchanged, so the sidecar
        ``frame_epochs`` still maps every 5-step epoch and seek granularity is preserved).
        The low-frame-count static stories (signal_lives heatmap, note) get ``{}`` ⇒ the
        crisp 140-DPI default. Returns plain kwargs the renderers accept.

        R4-F-01: the loss_drift NPZ-FREE path is ALSO a per-saved-training-epoch line with
        the same uncapped (~135-frame) budget as the per-eval-epoch stories — so it must
        get the lighter playback profile too (previously it kept the heavy 140-DPI default
        despite the uncapped frame count). The loss_drift ``npz_hist`` sub-mode keeps the
        crisp default (its NPZ list is strided to a bounded frame count, so it is not
        frame-bound). The decision mirrors ``_story_max_frames`` exactly.
        """
        if story in _PER_EPOCH_STORIES:
            dpi, figsize = render.playback_profile()
            return {"dpi": dpi, "figsize": figsize}
        if story == "loss_drift":
            params = (spec or {}).get("params", {}) if spec else {}
            if params.get("sub_mode", "npz_free") != "npz_hist":
                dpi, figsize = render.playback_profile()
                return {"dpi": dpi, "figsize": figsize}
        return {}

    def _render(self, job: GifJob) -> None:
        spec = job.spec
        story = spec["story"]
        job.status = "rendering"
        params = spec.get("params", {})
        max_epoch = spec.get("max_epoch")
        max_frames = self._story_max_frames(story, spec)
        fps = float(params.get("fps", 8.0))

        if story in ("climb_plateau", "warmup_join"):
            frames = self._frames_climb(spec, max_frames=max_frames)
        elif story == "what_carries":
            frames = self._frames_what_carries(spec, max_frames=max_frames)
        elif story == "compare_grid":
            frames = self._frames_compare_grid(spec, max_frames=max_frames)
        elif story == "line_race":
            frames = self._frames_line_race(spec, max_frames=max_frames)
        elif story == "metric_diff":
            frames = self._frames_metric_diff(spec, max_frames=max_frames)
        elif story == "bar_race":
            frames = self._frames_bar_race(spec, max_frames=max_frames)
        elif story == "loss_drift":
            frames = self._frames_loss_drift(spec, max_frames=max_frames)
        elif story == "signal_lives":
            frames = self._frames_signal_lives(spec, max_frames=max_frames)
        else:
            raise ValueError(f"unknown story {story!r}")

        job.n_frames = len(frames)
        if not frames:
            job.status = "error"
            job.error = "no_data_for_story"
            return
        gif_bytes = render.encode_gif(frames, fps=fps)
        self._atomic_write(self.gif_path(job.cache_key), gif_bytes)
        # R3-SR-03: the per-frame eval-epoch schedule (so the canvas player can seek by
        # epoch + collapse the peak-hold tail). Computed from the SAME schedule the
        # renderers build; harmless [] for non-line stories (hist/heatmap have their own
        # epoch labels drawn on-frame).
        frame_epochs, hold_len = self._frame_epoch_schedule(spec, story, max_frames)
        self._write_sidecar(job.cache_key, spec, len(frames),
                            frame_epochs=frame_epochs, hold_len=hold_len)
        job.frame = len(frames)
        job.status = "done"

    def _frame_epoch_schedule(
        self, spec: dict, story: str, max_frames: Optional[int]
    ) -> tuple[list[int], int]:
        """R3-SR-03: the eval-epoch shown on each rendered frame, for the canvas
        seek/label. Returns ``([], 0)`` for stories whose frame axis is not a simple
        eval/training-epoch reveal (npz hist-drift, feature-heatmap) — those carry their
        epoch label drawn on-frame, and the client need not seek-by-epoch for them.
        """
        epochs = self._epochs_for_story(spec, story)
        if not epochs:
            return [], 0
        hold = _STORY_HOLD.get(story, 10)
        return render.frame_epoch_schedule(epochs, max_frames=max_frames, hold=hold)

    def _epochs_for_story(self, spec: dict, story: str) -> list[int]:
        """Reconstruct the epoch axis a line/race/compare story animates over — the SAME
        axis the matching ``_frames_*`` builder uses — for the frame→epoch sidecar.
        Returns [] for npz/heatmap/anom-type stories (no simple eval-epoch reveal axis).
        """
        if story in ("climb_plateau", "warmup_join", "what_carries", "metric_diff"):
            # FB-R4-02: metric_diff animates over the first leaf's own eval-epoch axis
            # (both A and B share it), exactly like climb/what_carries.
            res = self._first_leaf_bundle(spec, {"epoch"})
            if res is None or res[0].epoch is None:
                return []
            ep = res[0].epoch.epochs
            cap = self._cap_epochs(ep, spec.get("max_epoch"))
            return list(ep[:cap])
        if story in ("line_race", "bar_race", "compare_grid"):
            return self._union_epochs_for_set(spec)
        if story == "loss_drift":
            params = spec.get("params", {})
            if params.get("sub_mode", "npz_free") == "npz_hist":
                return []  # npz path labels epochs on-frame from the NPZ filenames
            res = self._first_leaf_bundle(spec, {"history"})
            if res is None or res[0].history is None:
                return []
            return list(res[0].history.train_epochs)
        return []  # signal_lives (heatmap / anom-type) labels epochs on-frame

    def _union_epochs_for_set(self, spec: dict) -> list[int]:
        """The sorted UNION eval-epoch axis across the experiment_set for the metric —
        mirrors the alignment in ``_frames_line_race``/``_bar_race``/``_compare_grid``
        (align by epoch NUMBER, max_epoch clamp applied)."""
        metric = (spec.get("metric_keys") or ["pak_auc_f1"])[0]
        union: set[int] = set()
        for sel in spec.get("experiment_set", []):
            exp, ds, leaf = self.repo.resolve_leaf(
                sel.get("exp_id", ""), sel.get("dataset_key", ""), sel.get("variant"))
            if leaf is None:
                continue
            b = self.repo.load_bundle(exp, ds, leaf, want={"epoch"})
            if b.epoch is None or metric not in b.epoch.series:
                continue
            union |= set(b.epoch.epochs)
        epochs = sorted(union)
        if spec.get("max_epoch") is not None:
            epochs = [e for e in epochs if e <= spec["max_epoch"]]
        return epochs

    def _first_leaf_bundle(self, spec: dict, want: set[str]) -> Optional[tuple[LeafBundle, Any]]:
        sel = (spec.get("experiment_set") or [{}])[0]
        exp, ds, leaf = self.repo.resolve_leaf(
            sel.get("exp_id", ""), sel.get("dataset_key", ""), sel.get("variant"))
        if leaf is None:
            return None
        return self.repo.load_bundle(exp, ds, leaf, want=want), leaf

    def _cap_epochs(self, epochs: list[int], max_epoch: Optional[int]) -> int:
        if max_epoch is None:
            return len(epochs)
        return sum(1 for e in epochs if e <= max_epoch) or len(epochs)

    # story 1/2 — detection / warmup-join lines
    def _frames_climb(self, spec: dict, *, max_frames: int) -> list:
        res = self._first_leaf_bundle(spec, {"epoch", "metadata"})
        if res is None:
            return []
        b, leaf = res
        if b.epoch is None:
            return []
        epochs = b.epoch.epochs
        cap = self._cap_epochs(epochs, spec.get("max_epoch"))
        epochs = epochs[:cap]
        story = get_story(spec["story"])
        # FB-10: the user's explicit ``metric_keys`` OVERRIDE the story default. The
        # story's overlay_with/pair_with only SEED the picker when NO metric was
        # selected (empty list); a non-empty selection is rendered EXACTLY as given (no
        # auto-added "Teacher PA%K-AUC F1"/"F1 (strict)").
        explicit = spec.get("metric_keys")
        if explicit:
            keys = list(explicit)
        else:
            keys = list(story["default_metric_keys"])
            keys += [k for k in story.get("overlay_with", []) if k not in keys]
            keys += [k for k in story.get("pair_with", []) if k not in keys]
        lines = []
        for k in keys:
            if k not in b.epoch.series:
                continue
            meta = self.repo.registry.resolve(k, namespace="epoch_metrics").model_dump()
            lines.append({"label": meta["display_name"],
                          "values": b.epoch.series[k][:cap], "meta": meta})
        best_epoch = b.metadata.timing.get("best_epoch") if b.metadata else None
        title = story["label"]
        return render.render_lines(
            title=title, epochs=epochs, lines=lines,
            warmup_epochs=leaf.warmup_epochs, best_epoch=best_epoch,
            y_unit=(lines[0]["meta"].get("unit") if lines else None),
            max_frames=max_frames, **self._playback_kwargs(spec["story"]))

    # the GIF-7 score-variant markers, declared ONCE so the renderer + the /gif/stories
    # manifest never diverge. Each entry: (variant_key, metadata block, has-per-epoch-
    # series?, the per-epoch series key for that variant given the chosen <metric>).
    #   * metrics(adaptive)  ↔ the <metric> series        (HAS a per-epoch line)
    #   * teacher            ↔ teacher_<metric> series     (HAS a per-epoch line)
    #   * student / disc     ↔ best-epoch SCALAR only      (NO per-epoch series, P3-01)
    _WHAT_CARRIES_VARIANTS = (
        ("metrics", "metrics", True),
        ("teacher", "teacher_recon_metrics", True),
        ("student", "student_recon_metrics", False),
        ("disc", "disc_metrics", False),
    )

    @staticmethod
    def _what_carries_series_key(variant: str, metric: str) -> Optional[str]:
        """The per-epoch ``epoch_metrics`` series key that backs a GIF-7 variant's line
        (``None`` for student/disc — best-epoch scalars only, P3-01)."""
        if variant == "metrics":
            return metric
        if variant == "teacher":
            return f"teacher_{metric}"
        return None

    @classmethod
    def what_carries_ceiling_variants(cls) -> list[dict[str, Any]]:
        """FB-R4b-01: the togglable GIF-7 markers + WHICH carry a TRUE ceiling.

        ``has_series`` variants (adaptive/teacher) expose a ceiling = max(per-epoch
        series); the others (student/disc) expose a best-epoch scalar (NOT a ceiling,
        P3-01). Surfaced in ``/gif/stories`` so the picker can list the checkboxes."""
        out = []
        for variant, block, has_series in cls._WHAT_CARRIES_VARIANTS:
            out.append({
                "variant": variant, "metadata_block": block,
                "has_series": has_series,
                "marker_kind": "ceiling" if has_series else "best_epoch",
            })
        return out

    # story 6 — what carries the detector (P3-01)
    def _frames_what_carries(self, spec: dict, *, max_frames: int) -> list:
        res = self._first_leaf_bundle(spec, {"epoch", "metadata"})
        if res is None:
            return []
        b, leaf = res
        if b.epoch is None or b.metadata is None:
            return []
        epochs = b.epoch.epochs
        cap = self._cap_epochs(epochs, spec.get("max_epoch"))
        epochs = epochs[:cap]
        metric = (spec.get("metric_keys") or ["pak_auc_f1"])[0]
        lines = []
        # P3-01: per-epoch lines ONLY for adaptive(metrics) + teacher_<metric>.
        adaptive_key = metric                      # the adaptive score's metric
        teacher_key = f"teacher_{metric}"
        series_max: dict[str, float] = {}          # variant -> max(per-epoch series)
        for variant, k, lbl in (("metrics", adaptive_key, "adaptive (metrics)"),
                                 ("teacher", teacher_key, "teacher")):
            if k in b.epoch.series:
                meta = self.repo.registry.resolve(k, namespace="epoch_metrics").model_dump()
                vals = b.epoch.series[k][:cap]
                lines.append({"label": lbl, "values": vals, "meta": meta})
                finite = [v for v in vals if isinstance(v, (int, float))]
                if finite:
                    series_max[variant] = max(finite)
        # FB-R4b-01: only the SELECTED ceiling/marker variants are drawn. ``show_ceilings``
        # is a list of variant keys (default = all four); threaded into the cache key via
        # ``params`` so each combination caches independently.
        params = spec.get("params", {}) or {}
        all_keys = [v for v, _, _ in self._WHAT_CARRIES_VARIANTS]
        show = params.get("show_ceilings")
        selected = set(show) if isinstance(show, list) else set(all_keys)
        # FB-R4b-02: build the markers as TRUE ceilings (max-of-series) where a per-epoch
        # series exists, and best-epoch SCALARS (labeled, NOT a ceiling) for student/disc.
        markers = []
        bm = self.repo.registry.resolve(metric, namespace="epoch_metrics").model_dump()
        for variant, block, has_series in self._WHAT_CARRIES_VARIANTS:
            if variant not in selected:
                continue
            if has_series:
                # a TRUE ceiling: the max of the variant's OWN drawn per-epoch line within
                # the rendered range — the line can never exceed it. If the line is absent
                # on this leaf, there is no ceiling to draw (skip).
                ceil = series_max.get(variant)
                if not isinstance(ceil, (int, float)):
                    continue
                label = ("adaptive max (series)" if variant == "metrics"
                         else "teacher max (series)")
                # FB-R4b-02 (optional): annotate the small metadata-vs-series provenance
                # gap (the 2026-06-01 lambda recompute left the metadata best-epoch scalar
                # slightly below the per-epoch series max for the SAME metric/epoch). Shown
                # only when the metadata best differs from the series max, so the
                # discrepancy is EXPLAINED rather than hidden — and never claimed as the
                # (exceeded) ceiling.
                meta_best = (b.metadata.score_variants.get(block, {}) or {}).get(metric)
                if (isinstance(meta_best, (int, float))
                        and abs(meta_best - ceil) > 1e-6):
                    label += f"  (meta best {meta_best:.4f})"
                markers.append({"label": label, "value": ceil, "kind": "ceiling",
                                "meta": bm})
            else:
                # student/disc — best-epoch SCALAR only (P3-01); NOT a ceiling.
                blk = b.metadata.score_variants.get(block, {}) or {}
                val = blk.get(metric)
                if isinstance(val, (int, float)):
                    markers.append({"label": f"{variant} best-epoch", "value": val,
                                    "kind": "best_epoch", "meta": bm})
        best_epoch = b.metadata.timing.get("best_epoch")
        return render.render_lines(
            title=f"What carries the detector — {metric}",
            epochs=epochs, lines=lines, warmup_epochs=leaf.warmup_epochs,
            best_epoch=best_epoch, best_markers=markers,
            y_unit="score [0,1]", max_frames=max_frames,
            **self._playback_kwargs(spec["story"]))

    # story 4 — bar-chart race across the experiment_set
    def _frames_bar_race(self, spec: dict, *, max_frames: int) -> list:
        metric = (spec.get("metric_keys") or ["pak_auc_f1"])[0]
        meta = self.repo.registry.resolve(metric, namespace="epoch_metrics")
        palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
                   "#56B4E9", "#F0E442", "#999999"]
        bars = []
        union_epochs: set[int] = set()
        collected = []
        for i, sel in enumerate(spec.get("experiment_set", [])):
            exp, ds, leaf = self.repo.resolve_leaf(
                sel.get("exp_id", ""), sel.get("dataset_key", ""), sel.get("variant"))
            if leaf is None:
                continue
            b = self.repo.load_bundle(exp, ds, leaf, want={"epoch"})
            if b.epoch is None or metric not in b.epoch.series:
                continue
            union_epochs |= set(b.epoch.epochs)
            collected.append((f"{sel.get('exp_id', '')[:14]}/{sel.get('dataset_key', '')}",
                              dict(zip(b.epoch.epochs, b.epoch.series[metric])),
                              palette[i % len(palette)]))
        if not collected:
            return []
        epochs = sorted(union_epochs)
        if spec.get("max_epoch") is not None:
            epochs = [e for e in epochs if e <= spec["max_epoch"]]
        for label, byep, color in collected:
            # align by epoch NUMBER; gap (None) where a run lacks that epoch — never 0.
            vals = [byep.get(e) for e in epochs]
            bars.append({"label": label, "values": vals, "color": color})
        return render.render_bar_race(
            title=f"Ranking race — {meta.display_name}", epochs=epochs, bars=bars,
            direction=meta.direction, metric_display=meta.display_name,
            max_frames=max_frames, **self._playback_kwargs(spec["story"]))

    # story 8 — line race (FB-3): one shared axis, N leaf lines, per-epoch overtaking
    def _frames_line_race(self, spec: dict, *, max_frames: int) -> list:
        """One LINE per pinned leaf over the UNION eval-epoch axis (aligned by epoch
        number; gaps stay gaps, never 0). The value is the real per-eval-epoch
        ``epoch_metrics`` series value AT the animated epoch — NOT best-epoch-fixed (the
        FB-3 fix to the old single-bar best-epoch race). 5-epoch resolution is the
        NATIVE grid (eval_interval=5 → epochs are 5,10,…). P3-01-safe: only the
        adaptive ``epoch_metrics`` per-epoch series is used; a best-epoch-only-scalar
        metric (student/disc) is simply absent from ``epoch.series`` → that leaf yields
        no line (empty/gap), never a fabricated per-epoch trajectory."""
        metric = (spec.get("metric_keys") or ["pak_auc_f1"])[0]
        meta = self.repo.registry.resolve(metric, namespace="epoch_metrics").model_dump()
        union_epochs: set[int] = set()
        collected: list[tuple[str, dict[int, Any]]] = []
        for sel in spec.get("experiment_set", []):
            exp, ds, leaf = self.repo.resolve_leaf(
                sel.get("exp_id", ""), sel.get("dataset_key", ""), sel.get("variant"))
            if leaf is None:
                continue
            b = self.repo.load_bundle(exp, ds, leaf, want={"epoch"})
            if b.epoch is None or metric not in b.epoch.series:
                continue  # best-epoch-only-scalar metric => no per-epoch line (P3-01)
            union_epochs |= set(b.epoch.epochs)
            label = f"{sel.get('exp_id', '')[:14]} · {sel.get('dataset_key', '')}"
            collected.append((label, dict(zip(b.epoch.epochs, b.epoch.series[metric]))))
        if not collected:
            return []
        epochs = sorted(union_epochs)
        if spec.get("max_epoch") is not None:
            epochs = [e for e in epochs if e <= spec["max_epoch"]]
        lines = []
        for label, byep in collected:
            vals = [byep.get(e) for e in epochs]   # align by epoch NUMBER; gap = None
            lines.append({"label": label, "values": vals})
        return render.render_line_race(
            title=f"Line race — {meta['display_name']}", epochs=epochs, lines=lines,
            metric_display=meta["display_name"], direction=meta.get("direction", "up"),
            y_unit=meta.get("unit"), max_frames=max_frames,
            **self._playback_kwargs(spec["story"]))

    @staticmethod
    def _diff_pair(spec: dict) -> tuple[str, str]:
        """FB-R4-02: resolve the (metric_a, metric_b) pair for the metric-diff story.

        Precedence (F5 — any discovered key is acceptable; never a hard-coded list):
          1. explicit ``params.metric_a`` / ``params.metric_b``;
          2. else ``metric_keys[0]`` / ``metric_keys[1]`` if two were selected;
          3. else ``metric_keys[0]`` as A and ``teacher_<A>`` as B (the canonical
             best-vs-teacher discrepancy-contribution diff);
          4. else the story default pair (pak_auc_f1 − teacher_pak_auc_f1).
        """
        params = spec.get("params", {}) or {}
        a = params.get("metric_a")
        b = params.get("metric_b")
        if a and b:
            return str(a), str(b)
        keys = spec.get("metric_keys") or []
        if len(keys) >= 2:
            return str(keys[0]), str(keys[1])
        if len(keys) == 1:
            base = str(keys[0])
            return base, (b or f"teacher_{base}")
        return (a or "pak_auc_f1"), (b or "teacher_pak_auc_f1")

    # story 9 — metric difference (A − B) over eval epochs (FB-R4-02)
    def _frames_metric_diff(self, spec: dict, *, max_frames: int) -> list:
        """Animate the per-eval-epoch DIFFERENCE A − B of two metrics on the FIRST leaf.

        BOTH series come from the leaf's per-eval-epoch ``epoch_metrics`` (aligned by
        epoch index, since both share the leaf's own eval-epoch axis). A − B is computed
        only where BOTH have a finite value AT that epoch; otherwise the diff is a GAP
        (None) — so a best-epoch-only-scalar metric (student/disc, with no per-epoch
        series in ``epoch.series``) yields a fully-gapped diff line, NEVER a fabricated
        per-epoch value (P3-01). A zero reference marker line flags A==B."""
        res = self._first_leaf_bundle(spec, {"epoch", "metadata"})
        if res is None:
            return []
        b, leaf = res
        if b.epoch is None:
            return []
        epochs = b.epoch.epochs
        cap = self._cap_epochs(epochs, spec.get("max_epoch"))
        epochs = epochs[:cap]
        key_a, key_b = self._diff_pair(spec)
        meta_a = self.repo.registry.resolve(key_a, namespace="epoch_metrics").model_dump()
        meta_b = self.repo.registry.resolve(key_b, namespace="epoch_metrics").model_dump()
        ser_a = b.epoch.series.get(key_a)   # None ⇒ best-epoch-only/absent (P3-01 gap)
        ser_b = b.epoch.series.get(key_b)
        if ser_a is None and ser_b is None:
            # neither metric has a per-epoch series on this leaf — explicit note, not empty.
            return render.render_note(
                title=f"Metric difference — {key_a} − {key_b}",
                message=(f"Neither '{key_a}' nor '{key_b}' has a per-eval-epoch series on "
                         f"this run (both are best-epoch-only or absent).\n"
                         "The per-epoch difference is undefined → all gaps (P3-01)."))
        diff_vals: list[Optional[float]] = []
        for i in range(cap):
            va = ser_a[i] if (ser_a is not None and i < len(ser_a)) else None
            vb = ser_b[i] if (ser_b is not None and i < len(ser_b)) else None
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                diff_vals.append(float(va) - float(vb))
            else:
                diff_vals.append(None)   # P3-01: gap where either side is missing
        # E-2 / GIF-8 (A-4): if NO epoch has BOTH sides finite the diff is ALL gaps — a
        # best-epoch-only B (or a non-overlapping A/B axis). render_lines would yield an
        # EMPTY GIF (blank chart). Render an explicit note naming the missing side instead,
        # so the surface is never a blank/no_data frame.
        if not any(v is not None for v in diff_vals):
            a_has = ser_a is not None and any(isinstance(v, (int, float)) for v in ser_a)
            b_has = ser_b is not None and any(isinstance(v, (int, float)) for v in ser_b)
            if a_has and not b_has:
                miss = (f"'{key_b}' has no per-eval-epoch series on this run "
                        "(best-epoch-only or absent), so A − B is undefined at every "
                        "epoch → all gaps (P3-01).")
            elif b_has and not a_has:
                miss = (f"'{key_a}' has no per-eval-epoch series on this run "
                        "(best-epoch-only or absent), so A − B is undefined at every "
                        "epoch → all gaps (P3-01).")
            else:
                miss = ("A and B never share a finite eval epoch on this run, so the "
                        "per-epoch difference is undefined → all gaps (P3-01).")
            return render.render_note(
                title=f"Metric difference — {key_a} − {key_b}", message=miss)
        # the diff is a NEUTRAL signed quantity (can be ±); render as one line + a zero ref.
        # E-2 / GIF-8: the axis UNIT is a signed delta of the base metric, NOT a [0,1]
        # "score" range — keep "Δ" in the unit string so render_lines' score-clamp is never
        # triggered (and pass signed_axis=True so the range straddles zero).
        base_unit = meta_a.get("unit") or meta_b.get("unit") or "mixed"
        diff_meta = {
            "display_name": f"{meta_a['display_name']} − {meta_b['display_name']}",
            "direction": "neutral", "family": "diagnostic.diff",
            "unit": f"Δ {base_unit} (A−B)",
            "inferred": False, "viz_hint": {},
        }
        lines = [{"label": f"{key_a} − {key_b}", "values": diff_vals, "meta": diff_meta}]
        best_epoch = b.metadata.timing.get("best_epoch") if b.metadata else None
        return render.render_lines(
            title=f"Metric difference — {key_a} − {key_b}",
            epochs=epochs, lines=lines, warmup_epochs=leaf.warmup_epochs,
            best_epoch=best_epoch,
            best_markers=[{"label": "A = B (zero)", "value": 0.0,
                           "meta": {"direction": "neutral"}}],
            y_unit=f"Δ {base_unit}  (signed, A−B)", max_frames=max_frames,
            signed_axis=True,
            **self._playback_kwargs(spec["story"]))

    # synchronized side-by-side compare — ONE combined GIF, lockstep playhead (F-03)
    def _frames_compare_grid(self, spec: dict, *, max_frames: int) -> list:
        """A single GIF with one sub-panel per leaf in the experiment_set, revealed in
        lockstep over the UNION epoch axis (aligned by epoch number; gaps stay gaps,
        never 0). The shared frame index synchronizes every panel by construction —
        the synchronized-playhead contract (frontend-design §6.3)."""
        metric = (spec.get("metric_keys") or ["pak_auc_f1"])[0]
        meta = self.repo.registry.resolve(metric, namespace="epoch_metrics").model_dump()
        union_epochs: set[int] = set()
        collected: list[tuple[str, dict[int, Any], Optional[int]]] = []
        for sel in spec.get("experiment_set", []):
            exp, ds, leaf = self.repo.resolve_leaf(
                sel.get("exp_id", ""), sel.get("dataset_key", ""), sel.get("variant"))
            if leaf is None:
                continue
            b = self.repo.load_bundle(exp, ds, leaf, want={"epoch"})
            if b.epoch is None or metric not in b.epoch.series:
                continue
            union_epochs |= set(b.epoch.epochs)
            label = f"{sel.get('exp_id', '')[:14]} · {sel.get('dataset_key', '')}"
            collected.append((label, dict(zip(b.epoch.epochs, b.epoch.series[metric])),
                              leaf.warmup_epochs))
        if not collected:
            return []
        epochs = sorted(union_epochs)
        if spec.get("max_epoch") is not None:
            epochs = [e for e in epochs if e <= spec["max_epoch"]]
        panels = []
        for label, byep, warmup in collected:
            vals = [byep.get(e) for e in epochs]   # align by epoch NUMBER; gap = None
            panels.append({
                "panel_title": label,
                "warmup_epochs": warmup,
                "lines": [{"label": meta["display_name"], "values": vals, "meta": meta}],
            })
        return render.render_compare_grid(
            title=f"Synchronized compare — {meta['display_name']}",
            epochs=epochs, panels=panels, y_unit=meta.get("unit"),
            max_frames=max_frames, **self._playback_kwargs(spec["story"]))

    # story 3 — loss descent / separation drift (NPZ-free default, NPZ sub-mode)
    def _frames_loss_drift(self, spec: dict, *, max_frames: int) -> list:
        params = spec.get("params", {})
        sub_mode = params.get("sub_mode", "npz_free")
        res = self._first_leaf_bundle(spec, {"epoch", "history", "metadata"})
        if res is None:
            return []
        b, leaf = res
        if sub_mode == "npz_hist":
            return self._frames_npz_drift(spec, leaf, max_frames=max_frames)
        # NPZ-free: animate loss descent + separation ratios over training-epoch.
        if b.history is None:
            # fall back to eval-epoch loss line
            return self._frames_climb({**spec, "story": "loss_drift",
                                       "metric_keys": spec.get("metric_keys") or
                                       ["reconstruction_loss"]}, max_frames=max_frames)
        h = b.history
        default_keys = ["epoch_recon_ratio_anomaly", "epoch_disc_ratio_anomaly"]
        keys = spec.get("metric_keys") or default_keys
        epochs = h.train_epochs
        lines = self._loss_drift_lines(h, keys)
        if not lines:
            # RF-1 (D-1): the requested key(s) have no training-history series on this
            # leaf. Rather than render an empty/no_data GIF, gracefully fall back to a
            # compatible AVAILABLE separation series so the first-click default always
            # produces a frame.
            fallback = self._compatible_separation_keys(h, exclude=set(keys))
            lines = self._loss_drift_lines(h, fallback)
        if not lines:
            # No compatible separation series at all on this leaf — return an explicit,
            # informative single frame instead of an empty/no_data GIF.
            req = ", ".join(keys) if keys else "(default)"
            return render.render_note(
                title="Separation drift over training (NPZ-free)",
                message=(f"No separation series for {req} on this run.\n"
                         "This leaf has no recon/disc ratio training-history series yet."))
        # FB-R4-03: the loss_drift DEFAULT series is the per-sample-type CONTRIBUTION
        # SHARE family epoch_{recon,disc}_ratio_{type} = component/(recon+disc) ∈[0,100]
        # (run_base_experiments.py:843-855) — NOT an anomaly/normal (a/n) ratio. The
        # axis label is now driven by the FIRST line's (corrected) registry family: the
        # share family resolves to diagnostic.context ⇒ "contribution share"; a genuine
        # separation series (recon_SNR/cohens_d/*_separation) resolves to
        # diagnostic.separation ⇒ "separation"; anything else defers to its own unit.
        # R4-F-02: when the family is diagnostic.context the label is DATA-DRIVEN — a
        # disc share (epoch_disc_ratio_*) reads "contribution share (disc/total)"; a recon
        # share (epoch_recon_ratio_*) reads "contribution share (recon/total)"; anything
        # else falls back to the generic "contribution share [%]". (Previously this was
        # hard-coded "disc/total" even when the first line was a recon share.)
        first_key = lines[0]["meta"].get("key", "") if lines else ""
        first_fam = lines[0]["meta"].get("family", "") if lines else ""
        if first_fam == "diagnostic.context":
            y_unit = self._contribution_share_label(first_key)
        elif first_fam == "diagnostic.separation":
            y_unit = "separation"
        else:
            y_unit = lines[0]["meta"].get("unit") or "value"
        return render.render_lines(
            title="Separation drift over training (NPZ-free)",
            epochs=epochs, lines=lines, warmup_epochs=leaf.warmup_epochs,
            y_unit=y_unit, max_frames=max_frames,
            **self._playback_kwargs("loss_drift", spec))

    @staticmethod
    def _contribution_share_label(key: str) -> str:
        """R4-F-02: DATA-DRIVEN y-axis label for the loss_drift contribution-share family.

        The plotted series is ``epoch_{recon,disc}_ratio_{type}`` = component/(recon+disc)
        ∈[0,100] (run_base_experiments.py:843-855). Reflect whether the FIRST plotted line
        is a disc share or a recon share; otherwise a generic share label — never the
        hard-coded "disc/total" when the line is actually a recon share."""
        k = (key or "").lower()
        if "disc" in k:
            return "contribution share (disc/total) [%]"
        if "recon" in k:
            return "contribution share (recon/total) [%]"
        return "contribution share [%]"

    def _loss_drift_lines(self, history, keys) -> list:
        """Build line dicts for the loss_drift NPZ-free renderer from history series keys
        that actually exist (skips absent keys — the F5 graceful-missing contract)."""
        lines = []
        for k in keys:
            if k in history.series:
                meta = self.repo.registry.resolve(k, namespace="history").model_dump()
                lines.append({"label": meta["display_name"],
                              "values": history.series[k], "meta": meta})
        return lines

    @staticmethod
    def _compatible_separation_keys(history, *, exclude: set) -> list:
        """Pick available training-history series compatible with the separation-drift
        story: prefer recon/disc *ratio* series, then any recon/disc separation series.
        Pure discovery over ``history.series`` — no hard-coded metric list required to
        exist; if none match, returns []."""
        avail = [k for k in history.series.keys() if k not in exclude]
        ratio = [k for k in avail
                 if "ratio" in k and ("recon" in k or "disc" in k)]
        if ratio:
            return ratio[:2]
        sep = [k for k in avail if ("recon" in k or "disc" in k)]
        return sep[:2]

    def _frames_npz_drift(self, spec: dict, leaf, *, max_frames: int) -> list:
        esd = leaf.leaf_path / "epoch_scores"
        if not esd.is_dir():
            return []
        npz_files = sorted(esd.glob("epoch_*_scores.npz"))
        if spec.get("max_epoch") is not None:
            npz_files = [f for f in npz_files
                         if int(f.stem.split("_")[1]) <= spec["max_epoch"]]
        # cap frame count by striding the NPZ list
        if len(npz_files) > max_frames:
            stride = len(npz_files) // max_frames + 1
            npz_files = npz_files[::stride]
        bins = spec.get("params", {}).get("bins", 60)

        def provider(i: int):
            if i >= len(npz_files):
                return None
            f = npz_files[i]
            try:
                hist = npz_histogram(f, "adaptive_score", bins=bins)
            except CorruptOrWriting:
                return None  # half-written NPZ -> skip frame (GIF-INV-4)
            if not hist.get("available"):
                return None
            try:
                hist["epoch"] = int(f.stem.split("_")[1])
            except (IndexError, ValueError):
                pass
            return hist

        return render.render_hist_drift(
            title="Score distribution drift (normal vs anomaly)",
            frame_provider=provider, n_frames=len(npz_files))

    # story 5 — where the signal lives (feature heatmap OR anomaly-type over training)
    def _frames_signal_lives(self, spec: dict, *, max_frames: int) -> list:
        params = spec.get("params", {})
        sub_mode = params.get("sub_mode", "feature_heatmap")
        if sub_mode == "anom_type_over_training":
            return self._frames_anom_type(spec, max_frames=max_frames)
        # feature heatmap
        res = self._first_leaf_bundle(spec, {"epoch"})
        if res is None:
            return []
        b, leaf = res
        if b.epoch is None:
            return []
        field = (spec.get("metric_keys") or ["_train_feature_recon_mean"])[0]
        from ..services.arrays import per_feature
        pf = per_feature(self.repo, b, field)
        if not pf.get("available"):
            return []
        # FB-R5-01: the configurable color scale for the heavy-tailed per-feature recon
        # distribution (default sqrt = PowerNorm γ=0.5; faithful, not washed out). Already
        # inside ``params`` ⇒ inside the cache key, so each scale caches independently.
        scale = params.get("heatmap_scale", render.DEFAULT_HEATMAP_SCALE)
        return render.render_feature_heatmap(
            title=f"Where the signal lives — {field}",
            epochs=pf["epochs"], matrix=pf["matrix"],
            warmup_epochs=pf["warmup_epochs"], mask_prewarmup=pf["mask_prewarmup"],
            max_frames=max_frames, heatmap_scale=scale)

    def _frames_anom_type(self, spec: dict, *, max_frames: int) -> list:
        """P3-02: per-anomaly-type evolution over training from the 2-level pivot."""
        res = self._first_leaf_bundle(spec, {"history"})
        if res is None:
            return []
        b, leaf = res
        if b.history is None or "epoch_anomaly_type_scores" not in b.history.nested:
            return []
        piv = b.history.nested["epoch_anomaly_type_scores"]
        sub_metric = spec.get("params", {}).get("sub_metric", "recon_ratio")
        if sub_metric not in piv.sub_metrics:
            sub_metric = piv.sub_metrics[0] if piv.sub_metrics else None
        if sub_metric is None:
            return []
        meta = self.repo.registry.resolve(sub_metric, namespace="history").model_dump()
        epochs = b.history.train_epochs
        # FB-7 / R2-06: scoped per-type display labels (spike->Anomaly single-bucket).
        from ..dataaccess.parsers import anomaly_type_display_names
        display = anomaly_type_display_names(piv.types)
        lines = []
        for t in piv.types:
            vals = piv.pivot.get(t, {}).get(sub_metric, [])
            lines.append({"label": display.get(t, t), "values": vals, "meta": meta})
        if not lines:
            return []
        return render.render_lines(
            title=f"Per-anomaly-type {sub_metric} over training",
            epochs=epochs, lines=lines, warmup_epochs=leaf.warmup_epochs,
            y_unit=sub_metric, max_frames=max_frames)


# module-level singleton bound to the repository
_GIF: Optional[GifService] = None


def get_gif_service() -> GifService:
    global _GIF
    if _GIF is None:
        from ..dataaccess.repository import get_repository
        _GIF = GifService(get_repository())
    return _GIF
