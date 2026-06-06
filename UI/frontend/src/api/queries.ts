/* TanStack Query hooks. Keys mirror backend cache keys; the catalog is fetched
 * once-and-cached (FE-FB-1) with a long staleTime so a re-pick of an already-fetched
 * metric is instant (AC-F3.3). The live-refresh toggle flips refetchInterval. */
import { useQuery } from "@tanstack/react-query";
import { api } from "./client";
import { useStore } from "../store";
import type { LeafSel } from "./types";

const FIVE_MIN = 5 * 60 * 1000;

/** Function form so TanStack Query re-reads the live-refresh flag on every tick —
 * toggling "live refresh" takes effect reactively, not only on the next render
 * of a component using the hook (review-P4-frontend F-07). */
const refetch = (): number | false => (useStore.getState().liveRefresh ? 10000 : false);

export const useHealth = () =>
  useQuery({ queryKey: ["health"], queryFn: api.health, refetchInterval: 30000 });

export const useExperiments = (source?: string, state?: string) =>
  useQuery({
    queryKey: ["experiments", source ?? "all", state ?? "all"],
    queryFn: () => api.experiments({ source, state }),
    refetchInterval: refetch,
  });

export const useExperiment = (expId?: string) =>
  useQuery({
    queryKey: ["experiment", expId],
    queryFn: () => api.experiment(expId!),
    enabled: !!expId,
    refetchInterval: refetch,
  });

export const useDataset = (exp?: string, dsKey?: string, variant?: string | null) =>
  useQuery({
    queryKey: ["dataset", exp, dsKey, variant ?? "_"],
    queryFn: () => api.dataset(exp!, dsKey!, variant),
    enabled: !!exp && !!dsKey,
    refetchInterval: refetch,
  });

export const useMetricsCatalog = (exp?: string, dsKey?: string, variant?: string | null) =>
  useQuery({
    queryKey: ["catalog", exp, dsKey, variant ?? "_"],
    queryFn: () => api.metricsCatalog(exp!, dsKey!, variant),
    enabled: !!exp && !!dsKey,
    staleTime: FIVE_MIN,
  });

/** GLOBAL UNION catalog (every leaf) — for the cross-cutting Rankings / Compare /
 * ScoreVariant pickers (F5; no hard-coded metric list, review-P4-frontend F-01). */
export const useGlobalMetricsCatalog = () =>
  useQuery({
    queryKey: ["catalog-global"],
    queryFn: api.globalMetricsCatalog,
    staleTime: FIVE_MIN,
  });

export const useEpochSeries = (exp?: string, dsKey?: string, keys?: string[], variant?: string | null) =>
  useQuery({
    queryKey: ["epoch-series", exp, dsKey, variant ?? "_", (keys ?? []).join(",")],
    queryFn: () => api.epochSeries(exp!, dsKey!, keys!, variant),
    enabled: !!exp && !!dsKey && !!keys && keys.length > 0,
    refetchInterval: refetch,
  });

export const useTrainingHistory = (exp?: string, dsKey?: string, keys?: string[], variant?: string | null) =>
  useQuery({
    queryKey: ["training-history", exp, dsKey, variant ?? "_", (keys ?? []).join(",")],
    queryFn: () => api.trainingHistory(exp!, dsKey!, keys!, variant),
    enabled: !!exp && !!dsKey && !!keys && keys.length > 0,
  });

export const useStats = (exp?: string, dsKey?: string, keys?: string[], variant?: string | null) =>
  useQuery({
    queryKey: ["stats", exp, dsKey, variant ?? "_", (keys ?? []).join(",")],
    queryFn: () => api.stats(exp!, dsKey!, keys!, variant),
    enabled: !!exp && !!dsKey && !!keys && keys.length > 0,
  });

export const useLeaderboard = (p: {
  metric?: string;
  scope?: string;
  group_by?: string;
  swat?: string;
  state?: string;
}) =>
  useQuery({
    queryKey: ["leaderboard", p],
    queryFn: () => api.leaderboard(p),
    refetchInterval: refetch,
  });

/** FB-12/FB-13: per-dataset rank tables + cross-dataset average-rank aggregate.
 * `datasets` is a comma-list of dataset COLUMN keys (FB-2); `swat=both` shows both
 * SWaT variants as columns (FB-11). ADDITIVE — does not touch the /leaderboard hook. */
export const useRankingsAggregate = (p: {
  metric?: string;
  datasets?: string;
  swat?: string;
  source?: string;
  search?: string;
  state?: string;
}) =>
  useQuery({
    queryKey: ["rankings-aggregate", p],
    queryFn: () => api.rankingsAggregate(p),
    refetchInterval: refetch,
  });

export const useConfig = (exp?: string, dsKey?: string, variant?: string | null) =>
  useQuery({
    queryKey: ["config", exp, dsKey, variant ?? "_"],
    queryFn: () => api.config(exp!, dsKey!, variant),
    enabled: !!exp && !!dsKey,
  });

// ── FB-NEW insight hooks (B1/B2/B3/B5). Lazy NPZ-backed (B1/B5) → no live refetch. ──
export const useInsightScoreDecomposition = (exp?: string, dsKey?: string, variant?: string | null) =>
  useQuery({
    queryKey: ["insight-decomp", exp, dsKey, variant ?? "_"],
    queryFn: () => api.insightScoreDecomposition(exp!, dsKey!, variant),
    enabled: !!exp && !!dsKey,
    staleTime: FIVE_MIN,
  });

export const useInsightDivergence = (exp?: string, dsKey?: string, metric = "pak_auc_f1", variant?: string | null) =>
  useQuery({
    queryKey: ["insight-divergence", exp, dsKey, metric, variant ?? "_"],
    queryFn: () => api.insightDivergence(exp!, dsKey!, metric, variant),
    enabled: !!exp && !!dsKey,
    refetchInterval: refetch,
  });

export const useInsightWarmupDelta = (exp?: string, dsKey?: string, metric = "pak_auc_f1", variant?: string | null) =>
  useQuery({
    queryKey: ["insight-warmup-delta", exp, dsKey, metric, variant ?? "_"],
    queryFn: () => api.insightWarmupDelta(exp!, dsKey!, metric, variant),
    enabled: !!exp && !!dsKey,
    refetchInterval: refetch,
  });

export const useInsightScoreDistribution = (
  exp?: string,
  dsKey?: string,
  p?: { epoch?: number; array?: string; bins?: number; variant?: string | null }
) =>
  useQuery({
    queryKey: ["insight-score-dist", exp, dsKey, p?.epoch ?? "_", p?.array ?? "adaptive_score", p?.variant ?? "_"],
    queryFn: () => api.insightScoreDistribution(exp!, dsKey!, p),
    enabled: !!exp && !!dsKey,
    staleTime: FIVE_MIN,
  });

/* E-3 (F-5): the 3 RECOMPUTED-ABLATION component detection trajectories + the adaptive
 * self-validation + the I2 efficacy verdict. Per-epoch numpy recompute is heavy ⇒ no
 * live refetch (the leaf's NPZ grid only grows; the backend caches on _INSIGHT_VERSION).
 * `metric` here is the EFFICACY-VERDICT metric (roc), NOT the displayed trajectory metric
 * (all of roc/ap/pak_f1 ship in one response, toggled client-side). */
export const useInsightComponentTrajectories = (
  exp?: string,
  dsKey?: string,
  variant?: string | null,
  metric = "roc"
) =>
  useQuery({
    queryKey: ["insight-traj", exp, dsKey, variant ?? "_", metric],
    queryFn: () => api.insightComponentTrajectories(exp!, dsKey!, { metric, variant }),
    enabled: !!exp && !!dsKey,
    staleTime: FIVE_MIN,
  });

/* E-4 (F-6): the score-vs-time component overlay at a chosen epoch (best epoch by
 * default). One downsampled NPZ read; cheap, but epoch-keyed so changing the leaf+epoch
 * selector refetches. */
export const useInsightComponentOverlay = (
  exp?: string,
  dsKey?: string,
  variant?: string | null,
  epoch?: number
) =>
  useQuery({
    queryKey: ["insight-overlay", exp, dsKey, variant ?? "_", epoch ?? "_"],
    queryFn: () => api.insightComponentOverlay(exp!, dsKey!, { epoch, variant }),
    enabled: !!exp && !!dsKey,
    staleTime: FIVE_MIN,
  });

/* E-5 (F-8): the hardest (worst-scoring) windows. Sorted server-side by `sort`; the
 * response carries is_sample/sample_size (the frontend MUST badge "~N-window SAMPLE"). */
export const useDetailedWindows = (
  exp?: string,
  dsKey?: string,
  variant?: string | null,
  p?: { sort?: string; top?: number }
) =>
  useQuery({
    queryKey: ["detailed-windows", exp, dsKey, variant ?? "_", p?.sort ?? "_", p?.top ?? 50],
    queryFn: () => api.detailedWindows(exp!, dsKey!, { ...p, variant }),
    enabled: !!exp && !!dsKey,
    staleTime: FIVE_MIN,
  });

/** FB-14: the grouped training-metric inventory for the GIF-3 / loss-drift picker. */
export const useLossDriftMetrics = (exp?: string, dsKey?: string, variant?: string | null) =>
  useQuery({
    queryKey: ["loss-drift-metrics", exp, dsKey, variant ?? "_"],
    queryFn: () => api.gifLossDriftMetrics(exp!, dsKey!, variant),
    enabled: !!exp && !!dsKey,
    staleTime: FIVE_MIN,
  });

export const useAnomalyTypes = (exp?: string, dsKey?: string, variant?: string | null) =>
  useQuery({
    queryKey: ["anomaly-types", exp, dsKey, variant ?? "_"],
    queryFn: () => api.anomalyTypes(exp!, dsKey!, variant),
    enabled: !!exp && !!dsKey,
  });

export const useSeparation = (exp?: string, dsKey?: string, variant?: string | null) =>
  useQuery({
    queryKey: ["separation", exp, dsKey, variant ?? "_"],
    queryFn: () => api.separation(exp!, dsKey!, variant),
    enabled: !!exp && !!dsKey,
  });

export const useScoreEpochs = (exp?: string, dsKey?: string, variant?: string | null) =>
  useQuery({
    queryKey: ["score-epochs", exp, dsKey, variant ?? "_"],
    queryFn: () => api.scoreEpochs(exp!, dsKey!, variant),
    enabled: !!exp && !!dsKey,
  });

/* F-2 (P0-2): the chosen score channel is threaded into BOTH the per-timestep slab
 * (so the score-over-time trace plots the selected array) AND the histogram. The
 * default is adaptive_score (the inference score). `array` is always fetched ALONGSIDE
 * point_labels so the histogram split / available_arrays list stay correct. */
export const useScores = (
  exp?: string,
  dsKey?: string,
  epoch?: number,
  variant?: string | null,
  array = "adaptive_score"
) =>
  useQuery({
    queryKey: ["scores", exp, dsKey, epoch, variant ?? "_", array],
    queryFn: () =>
      api.scores(exp!, dsKey!, epoch!, {
        // always include point_labels so the panel can split; include the chosen channel.
        arrays: array === "point_labels" ? "point_labels" : `${array},point_labels`,
        downsample: 4000,
        variant,
      }),
    enabled: !!exp && !!dsKey && epoch !== undefined,
  });

/* F-2 (P0-2 / V-1 / V-9): the histogram array is now a threaded arg (default
 * adaptive_score), not hard-coded — so the NpzScrubber channel selector drives it. */
export const useScoresHistogram = (
  exp?: string,
  dsKey?: string,
  epoch?: number,
  variant?: string | null,
  array = "adaptive_score"
) =>
  useQuery({
    queryKey: ["scores-hist", exp, dsKey, epoch, variant ?? "_", array],
    queryFn: () => api.scoresHistogram(exp!, dsKey!, epoch!, array, 80, variant),
    enabled: !!exp && !!dsKey && epoch !== undefined,
  });

export const useMonitoring = (expId?: string) =>
  useQuery({
    queryKey: ["monitoring", expId],
    queryFn: () => api.monitoring(expId!, "series"),
    enabled: !!expId,
    refetchInterval: refetch,
  });

export const useVizManifest = (exp?: string, dsKey?: string, variant?: string | null) =>
  useQuery({
    queryKey: ["viz-manifest", exp, dsKey, variant ?? "_"],
    queryFn: () => api.vizManifest(exp!, dsKey!, variant),
    enabled: !!exp && !!dsKey,
  });

export const useGifStories = () =>
  useQuery({ queryKey: ["gif-stories"], queryFn: api.gifStories, staleTime: FIVE_MIN });

export const usePanels = () =>
  useQuery({ queryKey: ["panels"], queryFn: api.panels, staleTime: FIVE_MIN });

export const useCompareMatrix = (leaves: LeafSel[], metrics: string[], scoreVariant?: string) =>
  useQuery({
    queryKey: ["compare-matrix", leaves, metrics, scoreVariant ?? "_"],
    queryFn: () => api.compareMatrix(leaves, metrics, scoreVariant),
    enabled: leaves.length > 0 && metrics.length > 0,
  });

export const useCompareSeries = (leaves: LeafSel[], key?: string) =>
  useQuery({
    queryKey: ["compare-series", leaves, key],
    queryFn: () => api.compareSeries(leaves, key!),
    enabled: leaves.length > 0 && !!key,
  });

export const useConfigDiff = (leaves: LeafSel[]) =>
  useQuery({
    queryKey: ["config-diff", leaves],
    queryFn: () => api.configDiff(leaves),
    enabled: leaves.length >= 1,
  });
