/* Thin typed fetch client over the backend /api contract. All paths are
 * same-origin (the SPA is served by the same uvicorn, no CORS). Per-dataset routes
 * use the blessed dataset_key_url ('/'->'~') encoding (DECISION_LOG #19). */
import type {
  AnomalyTypes,
  CatalogEntry,
  CompareMatrix,
  CompareSeries,
  ComponentTrajectories,
  ComponentOverlay,
  ConfigDiff,
  ConfigGroups,
  DatasetDetail,
  DetailedWindows,
  DivergenceTrajectory,
  EpochSeries,
  ExperimentDetail,
  ExperimentRef,
  GifListItem,
  GifRenderResp,
  GifSidecar,
  GifSpec,
  GifStatus,
  GifStory,
  GifStoriesResp,
  Health,
  Leaderboard,
  LossDriftMetrics,
  RankingsAggregate,
  LeafSel,
  MonitoringSeries,
  PanelList,
  ScoreDecomposition,
  ScoreDistribution,
  Separation,
  ScoresEpoch,
  ScoresHistogram,
  StatRow,
  TrainingHistory,
  VizManifest,
  WarmupDelta,
} from "./types";

const BASE = "/api";

async function jget<T>(path: string, params?: Record<string, any>): Promise<T> {
  const url = new URL(BASE + path, window.location.origin);
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      if (v === undefined || v === null || v === "") continue;
      url.searchParams.set(k, String(v));
    }
  }
  const r = await fetch(url.toString());
  if (!r.ok) throw new Error(`${r.status} ${r.statusText} for ${path}`);
  return (await r.json()) as T;
}

async function jpost<T>(path: string, body: any): Promise<T> {
  const r = await fetch(BASE + path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${r.status} ${r.statusText} for ${path}`);
  return (await r.json()) as T;
}

/** Encode a dataset key for a path segment per the blessed scheme. */
export function dsUrl(datasetKey: string): string {
  return datasetKey.replace(/\//g, "~");
}

const dpath = (exp: string, dsKey: string, suffix = "") =>
  `/experiments/${encodeURIComponent(exp)}/datasets/${dsUrl(dsKey)}${suffix}`;

export const api = {
  // ── discovery / health ──────────────────────────────────────────────────
  health: () => jget<Health>("/health"),
  rescan: () => jpost<{ experiments_found: number; changed: string[] }>("/rescan", {}),
  experiments: (p?: { source?: string; state?: string }) =>
    jget<ExperimentRef[]>("/experiments", p),
  experiment: (expId: string) => jget<ExperimentDetail>(`/experiments/${encodeURIComponent(expId)}`),
  dataset: (exp: string, dsKey: string, variant?: string | null) =>
    jget<DatasetDetail>(dpath(exp, dsKey), { variant }),

  // ── config + diff ─────────────────────────────────────────────────────────
  config: (exp: string, dsKey: string, variant?: string | null) =>
    jget<ConfigGroups>(dpath(exp, dsKey, "/config"), { variant }),
  configDiff: (leaves: LeafSel[]) => jpost<ConfigDiff>("/config/diff", { leaves }),

  // ── metric catalog + series (the two axes are SEPARATE endpoints) ─────────
  /** GLOBAL UNION catalog across every leaf — for the cross-cutting Rankings /
   * Compare pickers that are not scoped to one leaf (no hard-coded metric list). */
  globalMetricsCatalog: () =>
    jget<{ entries?: CatalogEntry[] } | CatalogEntry[]>("/metrics-catalog").then((r) =>
      Array.isArray(r) ? r : r.entries ?? []
    ),
  metricsCatalog: (exp: string, dsKey: string, variant?: string | null) =>
    jget<{ entries?: CatalogEntry[] } | CatalogEntry[]>(dpath(exp, dsKey, "/metrics-catalog"), {
      variant,
    }).then((r) => (Array.isArray(r) ? r : r.entries ?? [])),
  epochSeries: (exp: string, dsKey: string, keys: string[], variant?: string | null) =>
    jget<EpochSeries>(dpath(exp, dsKey, "/epoch-series"), {
      keys: keys.join(","),
      variant,
      include_meta: true,
    }),
  trainingHistory: (exp: string, dsKey: string, keys: string[], variant?: string | null) =>
    jget<TrainingHistory>(dpath(exp, dsKey, "/training-history"), {
      keys: keys.join(","),
      variant,
      include_meta: true,
    }),
  /** FB-R3-02: the backend `/stats` returns `{ stats: { <metric>: {...} } }` — a DICT
   * keyed by metric, where each entry has NO own `key` field (the metric IS the key).
   * The panel needs `StatRow[]`. Normalize here (one place, backend unchanged): inject
   * `key` from the dict key. A future backend list (`rows`/bare array) still works. */
  stats: (exp: string, dsKey: string, keys: string[], variant?: string | null) =>
    jget<
      { stats?: Record<string, Omit<StatRow, "key">> | StatRow[]; rows?: StatRow[] } | StatRow[]
    >(dpath(exp, dsKey, "/stats"), {
      keys: keys.join(","),
      variant,
    }).then((r): StatRow[] => {
      if (Array.isArray(r)) return r;
      if (Array.isArray(r.rows)) return r.rows;
      const s = r.stats;
      if (Array.isArray(s)) return s;
      if (s && typeof s === "object")
        return Object.entries(s).map(([key, v]) => ({ key, ...(v as Omit<StatRow, "key">) }));
      return [];
    }),
  perFeature: (exp: string, dsKey: string, field: string, variant?: string | null) =>
    jget<any>(dpath(exp, dsKey, "/per-feature"), { field, variant }),
  pakCurve: (exp: string, dsKey: string, epoch?: number, variant?: string | null) =>
    jget<any>(dpath(exp, dsKey, "/pak-curve"), { epoch, variant }),

  // ── rankings + comparison ─────────────────────────────────────────────────
  leaderboard: (p: {
    metric?: string;
    scope?: string;
    group_by?: string;
    swat?: string;
    sort?: string;
    filter?: string;
    search?: string;
    state?: string;
  }) => jget<Leaderboard>("/leaderboard", p),
  /** FB-12/FB-13: per-dataset rank tables + cross-dataset average-rank aggregate.
   * `datasets` is a comma-list of dataset COLUMN keys (FB-2); `swat=both` shows full
   * AND excl22 as separate columns (FB-11). ADDITIVE — /leaderboard is untouched. */
  rankingsAggregate: (p: {
    metric?: string;
    score_variant?: string;
    datasets?: string;
    swat?: string;
    source?: string;
    search?: string;
    state?: string;
  }) => jget<RankingsAggregate>("/rankings/aggregate", p),
  compareMatrix: (leaves: LeafSel[], metrics: string[], score_variant?: string) =>
    jpost<CompareMatrix>("/compare/matrix", { leaves, metrics, score_variant }),
  compareSeries: (leaves: LeafSel[], key: string, variant?: string | null) =>
    jpost<CompareSeries>("/compare/series", { leaves, key, variant }),

  // ── FB-NEW insight endpoints (B1/B2/B3/B5 — additive, lazy, tolerant) ──────
  insightScoreDecomposition: (exp: string, dsKey: string, variant?: string | null) =>
    jget<ScoreDecomposition>(dpath(exp, dsKey, "/insights/score-decomposition"), { variant }),
  insightDivergence: (exp: string, dsKey: string, metric = "pak_auc_f1", variant?: string | null) =>
    jget<DivergenceTrajectory>(dpath(exp, dsKey, "/insights/divergence"), { metric, variant }),
  insightWarmupDelta: (exp: string, dsKey: string, metric = "pak_auc_f1", variant?: string | null) =>
    jget<WarmupDelta>(dpath(exp, dsKey, "/insights/warmup-delta"), { metric, variant }),
  insightScoreDistribution: (
    exp: string,
    dsKey: string,
    p?: { epoch?: number; array?: string; bins?: number; variant?: string | null }
  ) => jget<ScoreDistribution>(dpath(exp, dsKey, "/insights/score-distribution"), p),
  /** E-3 (F-5): the 3 RECOMPUTED-ABLATION component detection trajectories + the
   * self-validation + I2 efficacy verdict (numpy-only backend). `metric` keys the
   * efficacy verdict (roc per EV2A-N2); `pak` toggles the PA%K-F1 channel. */
  insightComponentTrajectories: (
    exp: string,
    dsKey: string,
    p?: { metric?: string; pak?: boolean; variant?: string | null }
  ) => jget<ComponentTrajectories>(dpath(exp, dsKey, "/insights/component-trajectories"), p),
  /** E-4 (F-6): the score-vs-time component overlay at a chosen epoch (test axis). */
  insightComponentOverlay: (
    exp: string,
    dsKey: string,
    p?: { epoch?: number; downsample?: number; variant?: string | null }
  ) => jget<ComponentOverlay>(dpath(exp, dsKey, "/insights/component-overlay"), p),

  // ── per-dataset breakdowns ────────────────────────────────────────────────
  anomalyTypes: (exp: string, dsKey: string, variant?: string | null) =>
    jget<AnomalyTypes>(dpath(exp, dsKey, "/anomaly-types"), { variant }),
  separation: (exp: string, dsKey: string, variant?: string | null) =>
    jget<Separation>(dpath(exp, dsKey, "/separation"), { variant }),

  // ── NPZ per-timestep ──────────────────────────────────────────────────────
  scoreEpochs: (exp: string, dsKey: string, variant?: string | null) =>
    jget<{ epochs: number[]; available: boolean }>(dpath(exp, dsKey, "/score-epochs"), { variant }),
  scores: (
    exp: string,
    dsKey: string,
    epoch: number,
    p?: { arrays?: string; downsample?: number; page?: number; page_size?: number; variant?: string | null }
  ) => jget<ScoresEpoch>(dpath(exp, dsKey, `/scores/${epoch}`), p),
  scoresHistogram: (exp: string, dsKey: string, epoch: number, array = "adaptive_score", bins = 80, variant?: string | null) =>
    jget<ScoresHistogram>(dpath(exp, dsKey, `/scores/${epoch}/histogram`), { array, bins, variant }),
  detailedWindows: (exp: string, dsKey: string, p?: { sort?: string; top?: number; variant?: string | null }) =>
    jget<DetailedWindows>(dpath(exp, dsKey, "/detailed-windows"), p),

  // ── health / telemetry ────────────────────────────────────────────────────
  monitoring: (expId: string, as: "tail" | "series" = "series") =>
    jget<MonitoringSeries>(`/experiments/${encodeURIComponent(expId)}/monitoring`, { as }),
  telemetry: (expId: string) => jget<any>(`/experiments/${encodeURIComponent(expId)}/telemetry`),
  profiling: (expId: string, dsKey: string, variant?: string | null) =>
    jget<any>(`/experiments/${encodeURIComponent(expId)}/profiling/${dsUrl(dsKey)}`, { variant }),

  // ── viz / panels / export ─────────────────────────────────────────────────
  vizManifest: (exp: string, dsKey: string, variant?: string | null) =>
    jget<VizManifest>(dpath(exp, dsKey, "/viz-manifest"), { variant }),
  pngUrl: (path: string) => `${BASE}/files/png?path=${encodeURIComponent(path)}`,
  panels: () => jget<PanelList>("/panels"),
  panel: (exp: string, dsKey: string, name: string, variant?: string | null) =>
    jget<any>(dpath(exp, dsKey, `/panels/${encodeURIComponent(name)}`), { variant }),
  exportSeriesUrl: (exp: string, dsKey: string, keys: string[], variant?: string | null) => {
    const u = new URL(BASE + "/export/series.csv", window.location.origin);
    u.searchParams.set("exp_id", exp);
    u.searchParams.set("dataset_key", dsKey);
    u.searchParams.set("keys", keys.join(","));
    if (variant) u.searchParams.set("variant", variant);
    return u.toString();
  },

  // ── GIF (flagship) ─────────────────────────────────────────────────────────
  /** FB-R4b-01: the response also carries `what_carries_ceilings` (the GIF-7 togglable
   * ceiling/marker variants) — consumed by GIF Studio for the per-ceiling checkboxes. */
  gifStories: () => jget<GifStoriesResp>("/gif/stories"),
  /** FB-14: the full grouped inventory of per-training-epoch `history` series for a
   * leaf, so the GIF-3 / training-metric picker can surface ALL collected metrics. */
  gifLossDriftMetrics: (exp: string, dsKey: string, variant?: string | null) =>
    jget<LossDriftMetrics>("/gif/loss-drift-metrics", { exp_id: exp, dataset_key: dsKey, variant }),
  gifRender: (spec: GifSpec) => jpost<GifRenderResp>("/gif/render", spec),
  gifStatus: (jobId: string) => jget<GifStatus>(`/gif/status/${jobId}`),
  /** R3-SR-03: full sidecar for an already-cached GIF (frame_epochs + hold_len) — the
   * instant cache-hit path where there is no job to poll. */
  gifSidecar: (cacheKey: string) => jget<GifSidecar>(`/gif/sidecar/${cacheKey}`),
  gifUrl: (cacheKey: string) => `${BASE}/gif/${cacheKey}.gif`,
  gifList: (leaf?: string) => jget<{ gifs: GifListItem[] }>("/gif/list", { leaf }),
};

export { jget, jpost };
