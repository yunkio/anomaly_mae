# TSMAE Experiment Dashboard — `./UI/`

A local-only, read-only dashboard over `results/experiments/` (+ the read-only
`.trash/0601/pre_fullrerun/` fixtures). It is a **complete two-part app**:

- a **FastAPI backend** (data layer, full HTTP API, CPU-only compute services, the
  flagship **GIF rendering service**) running in an isolated `UI/.venv`, and
- a **React 18 + Vite + TypeScript + Plotly.js SPA** (`UI/frontend/`) built to
  `UI/frontend/dist/` and **served by the same uvicorn** via a `StaticFiles` mount +
  SPA history-fallback — so the whole product launches from **one** command, same
  origin, no CORS, no second server.

> **TL;DR — run it:** build the SPA once (`cd UI/frontend && npm install && npm run
> build`), then from the project root launch the single command:
> `UI/.venv/bin/python -m uvicorn app.main:app --app-dir UI/backend --host 127.0.0.1 --port 8000`
> and open `http://127.0.0.1:8000/`. Full steps below.

> **Hard safety (enforced in code):** the backend opens result files **read-only**,
> writes **only** under `UI/.cache/`, is **CPU-only** (no torch, no GPU), never
> opens `*.pt`, and runs in an **isolated venv** that never touches `dc_vis`.
> Node/npm are a **build-time** dependency only — they never enter `UI/.venv`.

## Layout

```
UI/
├── requirements.txt          # pinned backend deps (NO torch) — installed ONLY into UI/.venv
├── .venv/                    # isolated venv (base anaconda py3.12, NOT dc_vis)
├── .cache/                   # mtime-aware cache (the ONLY place the backend writes)
├── registry/
│   ├── metric-semantics-registry.json   # authoritative metric direction/family (v1.1)
│   └── data-schema.json                 # machine-readable schema sidecar (reference)
└── backend/
    ├── smoke_backend.py      # end-to-end data-layer smoke test (CPU, read-only)
    ├── tests/                # resolver self-test + score-formula CI gate
    └── app/
        ├── config.py         # paths / source roots / tunables (env-overridable; no metric lists)
        ├── main.py           # FastAPI app: /api/* + /api/gif/* + /assets mount + SPA fallback + self-test gate
        ├── api/
        │   ├── routes.py      # core + compute endpoints (discovery…NPZ…compare…panels…export)
        │   └── gif_routes.py  # /api/gif/* (stories, render, status, serve, list)
        ├── services/         # CPU-only compute (pandas/numpy)
        │   ├── series.py      # batched epoch-series, training-history, direction-aware stats
        │   ├── compare.py     # compare/matrix, compare/series, config/diff, config grouping (shared LeafSel)
        │   ├── rankings.py    # leaderboard (default pak_auc_f1, SWaT excl22, neutral excluded, group_by)
        │   ├── arrays.py      # per-feature heatmap, PA%K curve, separation (from array/loss_stats keys)
        │   ├── viz_manifest.py# PNG enumeration + stale flags + cached-GIF cross-link
        │   ├── panels.py      # EXT-6 @register_panel plugin registry (+ a built-in separation_trend)
        │   └── export.py      # CSV export (series, matrix)
        ├── gif/              # the FLAGSHIP GIF service (matplotlib Agg + imageio, NO ffmpeg)
        │   ├── stories.py     # the single STORY_REGISTRY (7 stories, explicit story→metric, P3-01/P3-02)
        │   ├── render.py      # direction-aware, phase-masked Agg frame renderers + GIF encode
        │   └── service.py     # cache key (source-mtime), async jobs, per-story data collection, cache index
        └── dataaccess/       # the read-only data layer (the heart of the backend)
            ├── io_safe.py     # tolerant JSON/JSONL, NaN/Inf->null, path confinement
            ├── models.py      # pydantic normalized model (OPEN metric maps — F5)
            ├── discovery.py   # presence-driven leaf scan (depth 1|2), SWaT pairing, completion
            ├── parsers.py     # per-file-kind parsers (UNION discovery, anomaly-type pivot)
            ├── registry.py    # registry loader + resolve() + resolver_self_test (F5 gate)
            ├── scoring.py     # score-formula resolver (§A6 order; never blank)
            ├── catalog.py     # dynamic UNION metric catalog (F5; no hard-coded lists)
            ├── npz_reader.py  # lazy mmap NPZ reader (downsample, one-at-a-time, BadZipFile-tolerant)
            ├── cache.py       # mtime/size-keyed cache under UI/.cache (atomic writes)
            └── repository.py  # the single read-only facade every endpoint goes through
└── frontend/                # the React 18 + Vite + TS + Plotly.js SPA (built to dist/)
    ├── package.json / package-lock.json   # build-time deps (npm ci-capable; NOT in UI/.venv)
    ├── vite.config.ts       # build -> dist/; dev-server proxies /api -> 127.0.0.1:8000
    ├── index.html           # Vite entry (loads /assets/index-<hash>.js|css)
    ├── dist/                # the BUILT SPA the backend serves (index.html + assets/<hash>.{js,css})
    └── src/
        ├── main.tsx / App.tsx          # QueryClient + Router bootstrap + route map
        ├── store.ts                    # Zustand pin-set + theme + live-refresh (IndexedDB; never writes results)
        ├── styles/{tokens.css,global.css}   # design system: concrete hex/HSL tokens, light+dark, WCAG/CB-safe
        ├── api/{types.ts,client.ts,queries.ts}   # typed fetch + TanStack Query hooks (catalog cached once)
        ├── viz/{useMetricStyle.ts,plotlyTemplate.ts,Plot.ts}   # the registry-driven style chokepoint + one Plotly template
        ├── components/
        │   ├── charts/MetricChart.tsx  # THE F4/F5 chart primitive (phase-mask/direction/inferred ALL from MetricMeta)
        │   ├── MetricFamilySelector.tsx# the dynamic metric picker (runtime catalog; never a fixed list)
        │   ├── AsyncPanel.tsx          # the 4 canonical states (loading/empty/partial/error) from one wrapper
        │   ├── panels/                 # EpochTrend, ScoreVariant (P3-01), AnomalyType, Separation,
        │   │                           #   TrainingHistory (P2-01 + 2-level pivot P3-02), NpzScrubber, PngGallery
        │   └── gif/GifViewer.tsx       # the flagship GIF player (render -> poll -> serve)
        └── views/                      # the 8 views + GifStudio (see "Frontend views" below)
```

## Setup (isolated venv — never `dc_vis`)

```bash
# from the project root. The base python is NON-dc_vis (anaconda py3.12.4).
/home/ykio/anaconda3/bin/python3 -m venv UI/.venv
UI/.venv/bin/pip install -U pip
UI/.venv/bin/pip install -r UI/requirements.txt
# isolation check (must print .../UI/.venv):
UI/.venv/bin/python -c "import sys; print(sys.prefix)"
```

## Build the frontend (one-time; build-time only — does NOT touch `UI/.venv`)

The SPA is React 18 + Vite + TS + Plotly.js. Node v20.16.0 / npm 10.8.1 (nvm) build it
to `UI/frontend/dist/`, which the backend serves via its `StaticFiles` mount. Node/npm
are **build-time only** — never installed into the python venv.

```bash
cd UI/frontend
npm ci               # deterministic install from package-lock.json (or `npm install`)
npm run build        # tsc -b && vite build -> emits UI/frontend/dist/
```

This emits `UI/frontend/dist/index.html` + `dist/assets/index-<hash>.{js,css}` — the
hashed layout the backend's `/assets` mount expects. `dist/` (and `node_modules/`) are
**gitignored**, so on a fresh clone run the two commands above once before launching;
in this workspace `dist/` is already built, so the run command below works as-is.

### Two run modes (the single run command is robust to both)

1. **Served-SPA (default, production).** When `UI/frontend/dist/` exists, the backend
   serves it at `/` with an SPA history-fallback (deep links resolve to `index.html`).
   This is the single-command product — nothing else to start.
2. **Dev-server fallback (no `dist/`, or live frontend editing).** If the npm registry
   is ever unreachable and `dist/` cannot be (re)built, or you want HMR, run the Vite
   dev server in a second terminal: `cd UI/frontend && npm run dev` (serves on
   `:5173` and **proxies `/api` -> `127.0.0.1:8000`**, configured in `vite.config.ts`).
   Open `http://127.0.0.1:5173/`. The backend command is unchanged.

If neither a `dist/` nor the dev server is present, the **backend still runs**: `/`
returns a friendly placeholder linking to `/api/health` and `/docs`, and every API
route is live — so the app is never un-launchable.

## Run (single command)

```bash
# from the project root, in the isolated venv
UI/.venv/bin/python -m uvicorn app.main:app --app-dir UI/backend --host 127.0.0.1 --port 8000
```

Then open:
- `http://127.0.0.1:8000/` — the **full SPA** (8 views + the flagship GIF Studio)
- `http://127.0.0.1:8000/api/health` — resolver self-test status + cache + roots
- `http://127.0.0.1:8000/docs` — auto OpenAPI for every endpoint

Point at different data / change the port with env vars (no code edit):
`TSMAE_LIVE_ROOT`, `TSMAE_FIXTURE_ROOT`, `TSMAE_REGISTRY`, `TSMAE_CACHE`,
`TSMAE_FRONTEND_DIST`, `TSMAE_HOST`, `TSMAE_PORT`.

## Feature tour (every view + the GIF feature)

The left rail navigates 8 views; the top context bar carries the breadcrumb, the
**pin tray** (the global multi-select of experiments/datasets, each in a stable
color-blind-safe color), the **command palette** (press **Cmd/Ctrl-K** — searches
experiments, datasets/leaves, and the runtime metric catalog; ↵ navigates, and a leaf
can be pinned from it), the `/health` status dot, a Rescan button, a live-refresh
toggle, and the light/dark theme switch.

| View | Route | What it shows | Key backend routes |
|---|---|---|---|
| **Overview / Leaderboard** | `/` | summary-stat cards (totals, complete/in-progress/aborted, best `pak_auc_f1`) + the leaderboard (rank by **any** discovered metric; SWaT full/excl22; neutral metrics excluded by default) | `/api/experiments`, `/api/leaderboard`, `/api/health` |
| **Experiment Detail** | `/exp/:id` | header (best metric, state chip, wall-time, warmup boundary, **resolved score-formula** — never blank), caveat badges, dataset sub-tabs, run-health trend (P2-02) | `/api/experiments/{id}`, `…/monitoring?as=series`, `…/config` |
| **Dataset Detail** (analytic core) | `/exp/:id/ds/:key` | an 8-tab workbench: **Trends** · **Score variants** (P3-01: adaptive+teacher per-epoch lines + 4 best-epoch markers) · **Anomaly types** (dynamic vocabulary) · **Separation/SNR** · **Per-feature** heatmap · **NPZ scrubber** · **Training-epoch** (P2-01 separate axis + the 2-level `pivot[type][sub_metric]` P3-02) · **Animations** · this leaf's **Gallery**. SWaT full/excl22 toggle. | `…/datasets/{key}`, `…/epoch-series`, `…/training-history`, `…/anomaly-types`, `…/separation`, `…/per-feature`, `…/score-epochs`+`…/scores/{epoch}`, `…/viz-manifest` |
| **Rankings** | `/rankings` | rank by **any** discovered metric — the metric picker is the **global runtime catalog** (`/api/metrics-catalog`, no hard-coded list) so a renamed/new/inferred metric is rankable automatically — + the flagship **bar-chart-race** GIF (GIF-4). `?metric=` is deep-linkable (the command palette jumps here). | `/api/metrics-catalog`, `/api/leaderboard`, `POST /api/gif/render` |
| **Comparison Workbench** | `/compare` | pin N leaves → a direction-colored metric **matrix** (missing = "—", never 0; add columns from the **runtime catalog**) + **overlay trends** synced by epoch number (metric from the catalog) + per-row sparklines; cross-link to config diff | `/api/metrics-catalog`, `POST /api/compare/matrix`, `POST /api/compare/series`, `POST /api/config/diff`, `/api/export/matrix.csv` |
| **Metric / Epoch Explorer** | `/explorer` | the dynamic metric picker → single/overlay/small-multiples `<MetricChart>` (eval-epoch **and** the separate training-epoch axis, never co-plotted) + direction-aware stat strip + CSV export | `…/metrics-catalog`, `…/epoch-series`, `…/training-history`, `…/stats`, `POST /api/compare/series` |
| **Config Inspector + Diff** | `/config` | grouped config accordion + the resolved score-formula card + multi-config **diff** over the UNION of keys (added/removed shown, not dropped) | `…/config`, `POST /api/config/diff` |
| **Visualization Gallery** | `/gallery` | enumerate present stored PNGs (`epoch_metrics`/`best_model`) with a stale-viz badge, Radix lightbox, two-up compare, and an "open animated version" cross-link when a cached GIF exists | `…/viz-manifest`, `/api/files/png`, `/api/gif/list` |
| **GIF Studio** (flagship F1) | `/gif` | pick a **story** (from `/api/gif/stories` — single source, never prose) → pick metric(s) from the **runtime catalog ∩ the story's families** → pick the experiment set from the pin tray → render / poll / play / scrub / export; **side-by-side synchronized compare** of two animations | `/api/gif/stories`, `POST /api/gif/render`, `…/status/{job}`, `…/{key}.gif` |

**The GIF feature (flagship).** Seven CPU-rendered stories (`climb_plateau`,
`warmup_join`, `loss_drift`, `bar_race`, `signal_lives`, `what_carries`, and the
synchronized `compare_grid`) — every one is **direction-aware** (good-hue + race sort
from the registry; a negative SNR shows a real warn state, never clamped) and
**phase-aware** (greys the pre-warmup span using `config.teacher_only_warmup_epochs` of
*that* leaf; never draws a post-warmup metric as a real line at 0). They are cached by a
source-mtime-aware key, so a live training append invalidates the GIF and a repeat is an
instant hit. `<GifViewer>` composes the `GifSpec`, polls `/api/gif/status`, plays the
looping GIF, and exports it to file.
**P3-01:** GIF-6 (`what_carries`) draws the 4 score variants as best-epoch markers —
it never fabricates a per-epoch student/disc trajectory. **P3-02:** GIF-5's
`anom_type_over_training` sub-mode binds the 2-level `pivot[type][sub_metric]` (its
sub-metric options come from the story default + the live pivot, never a literal list).
**Synchronized compare:** the "side-by-side compare" toggle renders **GIF-7
`compare_grid`** — a single combined GIF with one sub-panel per pinned leaf revealed in
**lockstep over the union epoch axis**, so the playhead is synchronized *by
construction* (no client multi-GIF drift).
**`loss_drift`** (GIF-3) defaults to the `npz_free` separation-drift sub-mode over the
training-epoch axis; its declared default metrics are the real separation series
`epoch_recon_ratio_anomaly` + `epoch_disc_ratio_anomaly`, and `_frames_loss_drift`
gracefully falls back to a compatible available separation series (or an explicit
informative single frame) if a requested key has no data — so the first click always
renders a frame, never an empty GIF.

## Dynamic-metric behavior (F4/F5 — no hard-coded metric list anywhere)

Metrics are discovered at runtime and rendered registry-driven, end to end:

- The metric set is a **UNION** over `epoch_metrics`, the 4 score-variant blocks,
  `loss_stats`, the training-history lists, and the anomaly-type sub-metrics — built
  per leaf, never enumerated in code (`…/metrics-catalog`).
- A single backend `resolve()` returns a `MetricMeta` for **every** key (registry
  entry first, else an ordered fallback, else a default) — so a discovered key always
  resolves and a renamed/added/removed key flows through with **zero code change**.
- `<MetricFamilySelector>` is built from that runtime catalog (grouped by family, with
  a namespace facet) — **never a fixed list**. `<MetricChart>` reads phase-masking,
  direction (good-hue / fill-on-descent), sparsity, the `inferred` badge, and the
  negative-SNR warn state **entirely from `MetricMeta`** — no chart hard-codes a color,
  a direction, or a metric name. A new/unknown metric appears **dotted + "inferred"**.
- A bad future registry edit is caught: the resolver **self-test** runs at startup and
  in CI; on failure `/api/health` goes degraded and the SPA shows an amber health dot.

## Endpoints (backend core)

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | resolver self-test status, cache stats, source roots |
| POST | `/api/rescan` | re-scan source roots, invalidate cache |
| GET | `/api/experiments` | discovered experiments (`?source=live\|fixture\|all`, `?state=`) |
| GET | `/api/experiments/{id}` | experiment detail (datasets + best snapshots + score formula) |
| GET | `/api/experiments/{id}/datasets/{key}` | dataset detail (`?variant=full\|excl22`) |
| GET | `…/datasets/{key}/config` | grouped config + score formula |
| GET | `/api/metrics-catalog` | the **GLOBAL** dynamic F5 metric catalog (UNION over every leaf) — powers the cross-cutting Rankings / Compare pickers (no hard-coded list) |
| GET | `…/datasets/{key}/metrics-catalog` | the per-leaf **dynamic F5** metric catalog (UNION + resolved) |
| GET | `…/datasets/{key}/epoch-series` | batched eval-epoch series (`?keys=`, meta once-per-key) |
| GET | `…/datasets/{key}/training-history` | the **separate** training-epoch axis + pivoted nested |
| GET | `…/datasets/{key}/stats` | direction-aware per-key summary stats |
| GET | `/api/leaderboard` | rank by best-epoch metric (`?metric=`, `?swat=full\|excl22`) |
| GET | `…/datasets/{key}/anomaly-types` | per-anomaly-type metrics (dynamic vocabulary) |
| GET | `…/datasets/{key}/score-epochs` | available NPZ epoch grid |
| GET | `…/datasets/{key}/scores/{epoch}` | lazy NPZ slab (downsampled / paged; BadZipFile-tolerant) |
| GET | `…/datasets/{key}/scores/{epoch}/histogram` | normal-vs-anomaly score histogram (+ SNR) |
| GET | `…/{id}/monitoring` | `?as=tail` or `?as=series` (per-eval health trend, P2-02) |
| GET | `…/datasets/{key}/config/grouped` | config grouped into the §5.2 buckets |
| POST | `/api/config/diff` | multi-config diff over N leaves (UNION keys; added/removed) |
| POST | `/api/compare/matrix` | best-epoch metric matrix (N leaves × M metrics; shared `LeafSel`) |
| POST | `/api/compare/series` | per-metric overlay across N leaves (align by epoch number) |
| GET | `…/datasets/{key}/per-feature` | feature×epoch matrix for an array field (`?field=`) |
| GET | `…/datasets/{key}/pak-curve` | PA%K sweep at one eval-epoch (`?epoch=`) |
| GET | `…/datasets/{key}/separation` | loss_stats separation pairs (neutral) + ratios/SNR (↑) |
| GET | `…/datasets/{key}/train-scores` | best-epoch train-split NPZ (optional; excl22 → `{available:false}`) |
| GET | `…/datasets/{key}/viz-manifest` | enumerate present PNGs + stale flag + `gif_available` |
| GET | `…/{id}/telemetry` | hardware telemetry CSV (optional) |
| GET | `…/{id}/profiling/{key}` | batch profiling + timing |
| GET | `…/datasets/{key}/detailed-windows` | sampled window table (`is_sample:true`) |
| GET | `/api/panels` · `…/datasets/{key}/panels/{name}` | EXT-6 computed panels |
| GET | `/api/export/series.csv` · POST `/api/export/matrix.csv` | CSV export |
| GET | `/api/files/png` | path-confined static PNG serve |

### GIF service (flagship, CPU-only — no ffmpeg)

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/gif/stories` | the `STORY_REGISTRY` manifest (7 stories, explicit story→metric) |
| POST | `/api/gif/render` | render a `GifSpec` → `{job_id, cache_key, cached}` (async, ThreadPool) |
| GET | `/api/gif/status/{job_id}` | render progress (`pending`/`rendering`/`done`/`error`) |
| GET | `/api/gif/{cache_key}.gif` | serve the cached GIF (read-only) |
| GET | `/api/gif/list` | cached-GIF index (`?leaf=` filter; for the gallery cross-link) |

`GifSpec` = `{story, metric_keys[], experiment_set:[LeafSel], variant?, max_epoch?,
params:{max_frames,fps,bins,sub_mode,sub_metric,…}}`. Every animation reads the
warmup boundary from `config.teacher_only_warmup_epochs` of **that leaf** (greys the
pre-warmup span), takes its good-hue + race sort from the registry `direction`, and
caches by `sha1(story_id, metric_keys, experiment_set, max_epoch, params, source-mtime)`
under `UI/.cache/gif/`. **P3-01:** GIF-6 (`what_carries`) draws adaptive(`metrics`)
+ `teacher_*` as per-epoch lines and the 4 score variants as best-epoch markers —
it never fabricates per-epoch student/disc trajectories. **P3-02:** GIF-5's
`anom_type_over_training` sub-mode binds the 2-level `pivot[type][sub_metric]`.

**Dataset keys with a `/`** (e.g. `WaDi/A1`, `SMD/concat`) are URL-encoded with `~`
in the path segment (`WaDi~A1`); the listing responses expose `dataset_key_url` for
the client. The canonical key (with `/`) is preserved everywhere else.

## Extension points (so new metrics/datasets/viz/pages/panels drop in without core changes)

The whole app is registry-/schema-driven on both sides — most extensions are a data
edit, not a code change.

- **Add a metric (new or renamed)** → **zero code**. The catalog UNION-discovers it; the
  backend resolver resolves it (registry entry first, else ordered `fallback.rules`,
  else default), so it appears in `<MetricFamilySelector>`, charts via `<MetricChart>`
  (dotted + `inferred` if unknown), and ranks/compares — with no edit anywhere.
  *To give it real semantics:* add an `entries[<key>]` to
  `UI/registry/metric-semantics-registry.json` (direction/family/phase/viz_hint); the
  resolver **self-test** guards regressions on startup/CI. No frontend change.
- **Add a dataset / anomaly-type** → **zero code**. Datasets come from presence-driven
  discovery (`/api/experiments`); the anomaly-type vocabulary is read at runtime. A new
  dataset dir or a new anomaly-type name simply appears (slash-bearing keys are exposed
  as `dataset_key_url`, e.g. `WaDi/A1` → `WaDi~A1`, and the frontend binds that form).
- **Add a visualization (interactive chart)** → for a metric-driven chart, reuse
  `<MetricChart>` (it inherits direction/phase/inferred from `MetricMeta` for free). For
  a genuinely new client chart kind, add a Plotly component under
  `UI/frontend/src/components/` and one render branch keyed on `viz_hint.chart`
  (`animated_line`/`bar_chart_race`/`animated_hist_drift`/`feature_heatmap`/…); existing
  metrics opt in via their registry `viz_hint.chart`.
- **Add a GIF story (animation)** → add a row to
  `UI/backend/app/gif/stories.py::STORY_REGISTRY` (served as data via
  `/api/gif/stories`, so the GIF-Studio picker binds it with no client edit) + a
  renderer keyed on the story name in `gif/service.py`/`render.py`.
- **Add a page / view** → drop a component in `UI/frontend/src/views/`, register its
  route in `src/App.tsx`, and add a rail entry in `src/components/AppShell.tsx`. The
  drill-down spine (Overview → Experiment → Dataset) and the Compare/Explorer/GIF
  studios are the anchors; the pin-set store (`src/store.ts`) gives the new view the
  shared selection for free.
- **Add a computed panel** → write a `@register_panel("name")` function in
  `UI/backend/app/services/panels.py` that receives the **read-only** `LeafBundle`
  (cannot open `.pt` or write results by construction). It is auto-listed at
  `/api/panels` and served at `…/datasets/{key}/panels/{name}`; the frontend's generic
  panel renderer surfaces it with no bespoke component. Ships a built-in
  `separation_trend` as the worked example.

## Status & verify

The app is **complete and runnable from the single command above** — backend + the
built SPA served from one uvicorn, plus the flagship GIF service. Quick checks:

```bash
# backend CI gates (CPU, read-only, exit 0 = PASS)
UI/.venv/bin/python UI/backend/smoke_backend.py
UI/.venv/bin/python UI/backend/tests/test_resolver_selftest.py

# served-SPA smoke (start the backend, then in another shell):
curl -s http://127.0.0.1:8000/api/health | python -m json.tool    # resolver_selftest: PASS
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/             # 200 (SPA index)
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/exp/x/ds/PSM # 200 (deep link -> index, SPA fallback)
```

The backend opens result files read-only, writes only under `UI/.cache/`, is CPU-only,
never opens `*.pt`, and runs in a venv isolated from `dc_vis`.
