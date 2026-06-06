/* TypeScript mirrors of the backend contract (backend-design.md §2.3/§3.2/§5/§6).
 * MetricMeta is the F4/F5 registry type that flows into every chart prop. */

export type Direction = "up" | "down" | "neutral";
export type PhaseValidity = "all" | "post-warmup";
export type LeafState = "complete" | "in_progress" | "early_abort" | "missing";
export type SourceKind = "live" | "fixture";
export type Axis = "eval-epoch" | "training-epoch";
/** SWaT eval variant the catalog entry belongs to (M-1, additive backend tag). The
 * frontend CONSUMES this tag for FB-1/FB-6 identity — it never re-derives provenance. */
export type VariantScope = "full" | "excl22" | "default";

export interface MetricMeta {
  key: string;
  display_name: string;
  family: string;
  direction: Direction;
  phase_validity: PhaseValidity;
  unit: string;
  description?: string;
  viz_hint: Record<string, any>;
  namespace?: string;
  inferred: boolean;
  inferred_direction?: boolean;
  rule_id?: string | null;
  /** M-1 additive tag (full | excl22 | default). Absent on legacy responses → treated
   * as "default" by metricCatalogView(). */
  variant_scope?: VariantScope;
}

export interface DatasetRef {
  dataset_key: string;
  dataset_key_url: string; // blessed '/'->'~' encoding (DECISION_LOG #19)
  variants: string[];
  state: LeafState;
  depth?: number;
}

export interface ExperimentRef {
  exp_id: string;
  exp_num?: number;
  timestamp?: string;
  suffix?: string;
  state: LeafState;
  source: SourceKind;
  datasets: DatasetRef[];
  summary?: Record<string, any> | null;
  best_pak_auc_f1?: number | null;
}

export interface ScoreFormula {
  ratio: number;
  fm_in_score: boolean;
  source: string;
  formula_text: string;
}

export interface DatasetDetail {
  exp_id: string;
  dataset_key: string;
  dataset_key_url: string;
  variant: string | null;
  state: LeafState;
  warmup_epochs: number | null;
  num_features?: number | null;
  score_variants: Record<string, Record<string, number | null>>;
  loss_stats?: Record<string, number | null>;
  timing?: Record<string, number | null>;
  score_formula?: ScoreFormula;
  present?: Record<string, boolean>;
  caveats?: string[];
  errors?: string[];
}

export interface ExperimentDetail {
  exp_id: string;
  source: SourceKind;
  state: LeafState;
  summary?: {
    total_time?: number | null;
    config_set?: string | null;
    description?: string | null;
    timestamp?: string | null;
    results?: any[];
    [k: string]: any;
  } | null;
  datasets: (DatasetRef & { best?: Record<string, number | null> | null })[];
  caveats?: string[];
  score_formula?: ScoreFormula;
  errors?: string[];
}

export interface EpochSeries {
  axis: Axis;
  epochs: number[];
  series: Record<string, (number | null)[]>;
  meta: Record<string, MetricMeta>;
  warmup_epochs: number | null;
  requested_missing?: string[];
  errors?: string[];
}

export interface NestedPivot {
  types: string[];
  sub_metrics: string[];
  pivot: Record<string, Record<string, (number | null)[]>>;
  sub_metric_meta: Record<string, MetricMeta>;
  /** FB-7: backend-supplied per-type display labels (spike→"Anomaly" when sole bucket,
   * normal→"Normal"; simulation 9-type keeps distinct names). Frontend just reads it. */
  type_display_names?: Record<string, string>;
}

export interface TrainingHistory {
  axis: "training-epoch";
  train_epochs: number[];
  series: Record<string, (number | null)[]>;
  meta: Record<string, MetricMeta>;
  empty_keys?: string[];
  nested: Record<string, NestedPivot>;
  warmup_epochs?: number | null;
  errors?: string[];
}

export interface CatalogEntry extends MetricMeta {}

export interface StatRow {
  key: string;
  best: number | null;
  argbest_epoch: number | null;
  mean: number | null;
  last: number | null;
  first: number | null;
  slope: number | null;
  plateau: boolean;
  delta_best_last: number | null;
  delta_best_first: number | null;
  direction: Direction;
  rankable?: boolean;
}

export interface LeaderboardRow {
  exp_id: string;
  dataset_key: string;
  dataset_key_url?: string;
  variant: string | null;
  value: number | null;
  best_epoch?: number | null;
  state: LeafState;
  source?: SourceKind;
}

export interface Leaderboard {
  metric: string;
  direction: Direction;
  meta?: MetricMeta;
  rankable?: boolean;
  neutral_note?: string | null;
  rows: LeaderboardRow[];
}

/* FB-12/FB-13: per-dataset rank tables + cross-dataset average-rank aggregate
 * (GET /api/rankings/aggregate). SWaT full & excl22 are SEPARATE dataset columns
 * (FB-11); the default is per-dataset, never a cross-dataset value mix (FB-12). */
export interface PerDatasetRankRow {
  exp_id: string;
  exp_num?: number | null;
  value: number | null;
  rank: number | null;
  source?: SourceKind;
  description?: string | null;
}
export interface AggregateRow {
  exp_id: string;
  exp_num?: number | null;
  source?: SourceKind;
  description?: string | null;
  avg_rank: number | null;
  aggregate_rank?: number | null;
  /** E-1 (P0-1): dense rank index WITHIN the low_coverage bucket (present only on rows
   * returned in `low_coverage`, not in the qualified `aggregate`). */
  low_coverage_rank?: number | null;
  coverage_present: number;
  coverage_total: number;
  mean_value: number | null;
  per_dataset_rank: Record<string, number | null>;
}
/** FB-R4c-04: per-column classification advertised by the backend so the dataset-set
 * selector can GROUP the universe (canonical units vs granular machines/channels/concat/
 * SWaT-full) and mark which are in the canonical default. */
export type RankColumnKind =
  | "canonical_avg"
  | "swat_excl22"
  | "swat_full"
  | "entity_leaf"
  | "concat"
  | "single_leaf";
export interface RankColumnMeta {
  kind: RankColumnKind;
  /** the parent group for an entity leaf / canonical avg (e.g. "SMD"); else null. */
  group?: string | null;
  is_default: boolean;
}
export interface RankingsAggregate {
  metric: string;
  score_variant?: string;
  direction: Direction;
  rankable: boolean;
  neutral_note?: string | null;
  swat?: string;
  /** the ACTIVE dataset COLUMN keys the aggregate currently ranks over (the chosen set;
   * SWaT split into `SWaT_A1A2 · full` / `· excl22`; SMD/SMAP/MSL as `<G> (avg)`). */
  datasets: string[];
  dataset_labels?: Record<string, string>;
  coverage_total: number;
  /** FB-R4c-04: the full selectable universe (canonical units + every granular
   * machine/channel/concat + SWaT full), the canonical DEFAULT set, the canonical
   * "(avg)" columns, and per-column classification metadata. */
  selectable_columns?: string[];
  default_columns?: string[];
  canonical_columns?: string[];
  column_meta?: Record<string, RankColumnMeta>;
  /** FB-R4c-01: per "(avg)" column → {exp_id: n_entities averaged}. */
  group_coverage?: Record<string, Record<string, number>>;
  per_dataset: Record<string, PerDatasetRankRow[]>;
  /** E-1 (P0-1): the QUALIFIED headline set — rows scored on >= min_coverage active
   * columns (sorted; rank 1 = best). The headline / top-N render ONLY this. */
  aggregate: AggregateRow[];
  /** E-1 (P0-1): rows BELOW the coverage threshold (NOT interleaved into `aggregate`).
   * Surfaced under a collapsed "insufficient coverage (n/N)" section. */
  low_coverage?: AggregateRow[];
  /** E-1 (P0-1): the effective coverage threshold actually applied (auto-derived or the
   * explicit value), and whether it was auto-derived. */
  min_coverage?: number;
  min_coverage_auto?: boolean;
  missing_dataset_rule?: string;
  errors?: string[];
}

export interface LeafSel {
  exp_id: string;
  dataset_key: string;
  variant: string | null;
}

export interface CompareCell {
  value: number | null;
  direction: Direction;
  state?: LeafState;
}
export interface CompareMatrix {
  leaves: LeafSel[];
  metrics: string[];
  meta: Record<string, MetricMeta>;
  rows: { leaf_id: string; cells: Record<string, CompareCell> }[];
}
export interface CompareSeriesItem {
  leaf_id: string;
  epochs: number[];
  values: (number | null)[];
  warmup_epochs?: number | null;
  available?: boolean;
  /* FB-4: direction-aware best-epoch marker (additive). A best-epoch-only-scalar /
   * neutral metric yields null here — never a fabricated per-epoch point (P3-01). */
  best_epoch_index?: number | null;
  best_epoch?: number | null;
  best_value?: number | null;
}
export interface CompareSeries {
  leaves: LeafSel[];
  meta: Record<string, MetricMeta>;
  align: string;
  direction?: Direction;
  key?: string;
  series: CompareSeriesItem[];
  warmup?: Record<string, number | null>;
  errors?: string[];
}

/* FB-NEW insight payloads (GET /api/insights/* — B1/B2/B3/B5). All additive, lazy,
 * tolerant; `available:false` (never a throw) when the data is absent. The frontend
 * READS these — it never re-derives the score split client-side (CLAUDE.md rule 3). */
export interface DiscConsistencyGate {
  ok: boolean;
  disc_active: boolean;
  correlation?: number | null;
  scale?: number | null;
  scaled_rel_error?: number | null;
  corr_floor?: number;
  rel_error_cap?: number;
  checks?: string;
  note?: string;
}
export interface ScoreDecomposition {
  available: boolean;
  reason?: string;
  epoch?: number;
  best_epoch?: number | null;
  warmup_epochs?: number | null;
  best_epoch_pre_warmup?: boolean;
  n_points?: number;
  downsampled?: boolean;
  data_source?: string;
  residual_gate?: DiscConsistencyGate;
  disc_consistency_gate?: DiscConsistencyGate;
  components?: {
    recon: { mean_normal: number | null; mean_anomaly: number | null; separation: number | null; source: string };
    disc: { mean_normal: number | null; mean_anomaly: number | null; separation: number | null; source: string };
  };
  pct_separation_from_disc?: number | null;
  raw_discrepancy_present?: boolean;
  headline?: string;
}
export interface DivergenceTrajectory {
  available: boolean;
  reason?: string;
  metric?: string;
  epochs?: number[];
  warmup_epochs?: number | null;
  series?: Record<string, { key: string; values: (number | null)[]; display_name: string }>;
  discrepancy_contribution?: (number | null)[] | null;
  best_epoch_scalars?: Record<string, number | null>;
  data_source?: string;
}
export interface WarmupDelta {
  available: boolean;
  reason?: string;
  metric?: string;
  direction?: Direction;
  warmup_epochs?: number | null;
  pre_warmup?: { best_epoch: number | null; best_value: number | null };
  post_warmup?: { best_epoch: number | null; best_value: number | null };
  delta?: number | null;
  student_helped?: boolean | null;
  caption?: string | null;
  data_source?: string;
}
export interface ScoreDistribution {
  available: boolean;
  reason?: string;
  epoch?: number;
  best_epoch?: number | null;
  array?: string;
  bins?: number[];
  normal_hist?: number[];
  anomaly_hist?: number[];
  snr?: number | null;
  data_source?: string;
  error?: string;
}

/* E-3 (F-5) — GET …/insights/component-trajectories. The 3 RECOMPUTED-ABLATION
 * detection trajectories (adaptive / teacher-only / disc-only) over eval epochs, the
 * self-validation of the adaptive recompute vs the STORED metrics, and the I2 efficacy
 * verdict. EV2A-N1: each channel's series lives at channels.<name>.trajectory.{roc,ap,
 * pak_f1} (NOT a flat `series`); EV2A-N2: the verdict keys on roc / a 0.02 lift. */
export interface TrajSeries {
  roc: (number | null)[];
  ap: (number | null)[];
  pak_f1: (number | null)[];
}
export interface TrajChannel {
  label: string;
  is_ablation: boolean;
  trajectory: TrajSeries;
}
export interface TrajSelfValidation {
  ok: boolean | null;
  reason?: string;
  epoch?: number;
  best_epoch?: number | null;
  tolerance?: number;
  recomputed?: { roc: number | null; ap: number | null };
  stored?: { roc: number | null; ap: number | null };
  abs_delta?: { roc: number | null; ap: number | null };
  note?: string;
}
export interface DiscEfficacyVerdict {
  verdict: string;
  text: string;
  metric?: string;
  warmup_epochs?: number | null;
  pre_best?: number | null;
  post_best?: number | null;
  delta_best?: number | null;
  delta_mean?: number | null;
  onset_epoch?: number | null;
  meaningful_lift_threshold?: number;
}
export interface ComponentTrajectories {
  available: boolean;
  reason?: string;
  label?: string;
  insight_version?: number;
  metric_for_verdict?: string;
  epochs?: number[];
  warmup_epochs?: number | null;
  best_epoch?: number | null;
  n_npz?: number;
  n_npz_used?: number;
  n_gap?: number;
  channels?: {
    adaptive: TrajChannel;
    teacher_only: TrajChannel;
    disc_only: TrajChannel;
  };
  self_validation?: TrajSelfValidation;
  efficacy_verdict?: DiscEfficacyVerdict;
  metrics_available?: string[];
  data_source?: string;
  p3_01_note?: string;
}

/* E-4 (F-6) — GET …/insights/component-overlay. At a chosen eval epoch, the three score
 * channels on the TEST time axis + derived anomaly_regions (on the downsampled index). */
export interface ComponentOverlay {
  available: boolean;
  reason?: string;
  epoch?: number;
  best_epoch?: number | null;
  requested_epoch?: number | null;
  length?: number | null;
  downsampled?: boolean;
  n_points?: number;
  channels?: {
    adaptive_score?: (number | null)[] | null;
    teacher_recon_error?: (number | null)[] | null;
    discrepancy_error?: (number | null)[] | null;
  };
  point_labels?: number[];
  anomaly_regions?: { start: number; end: number }[];
  available_arrays?: string[];
  data_source?: string;
  label?: string;
}

/* E-5 (F-8) — GET …/detailed-windows. The worst-scoring windows; is_sample/sample_size
 * MUST badge that this is a SAMPLE, not the full test length. */
export interface DetailedWindows {
  available: boolean;
  is_sample?: boolean;
  sample_size?: number;
  columns?: string[];
  rows?: Record<string, any>[];
  error?: string;
}

/* FB-14: GET /api/gif/loss-drift-metrics — the full grouped inventory of per-training-
 * epoch `history` series for a leaf (the GIF-3 / training-metric picker source). */
export interface LossDriftMetricItem {
  key: string;
  display_name: string;
  direction: Direction;
  phase_validity: PhaseValidity;
  inferred: boolean;
  populated: boolean;
}
export interface LossDriftDisabledItem {
  key: string;
  display_name: string;
  family: string;
  reason: string;
}
export interface LossDriftMetrics {
  available: boolean;
  families: Record<string, LossDriftMetricItem[]>;
  disabled: LossDriftDisabledItem[];
  default_metric_keys: string[];
  nested?: string[];
}

export interface ConfigGroups {
  /** backend returns a FLAT config dict under `config` (+ optional grouped `groups`). */
  config?: Record<string, any>;
  groups?: Record<string, Record<string, any>>;
  score_formula?: ScoreFormula;
}
export interface ConfigDiff {
  leaves: LeafSel[];
  keys: {
    key: string;
    values: Record<string, any>;
    differs: boolean;
    added_in?: string[];
    removed_in?: string[];
  }[];
}

export interface AnomalyTypes {
  available?: boolean;
  types: {
    name: string;
    display_name?: string;
    /** backend returns a flat {metric: value} map + a sibling {metric: meta} map */
    metrics: Record<string, number | null>;
    meta?: Record<string, MetricMeta>;
  }[];
}

export interface Separation {
  available?: boolean;
  pairs: { key: string; value: number | null; direction: Direction; family?: string; inferred?: boolean }[];
  ratios_snr: { key: string; value: number | null; direction: Direction; family?: string; inferred?: boolean }[];
}

export interface ScoresEpoch {
  epoch: number;
  available?: boolean;
  length?: number;
  arrays?: Record<string, number[]>;
  available_arrays?: string[];
  downsampled?: boolean;
  page?: number | null;
  page_size?: number | null;
  page_start?: number;
  page_end?: number;
  prewarmup?: boolean;
  error?: string;
  reason?: string;
}
export interface ScoresHistogram {
  epoch: number;
  available?: boolean;
  bins?: number[];
  normal_hist?: number[];
  anomaly_hist?: number[];
  snr?: number | null;
  error?: string;
}

export interface MonitoringSeries {
  runtime?: number[];
  latest_pak_f1?: (number | null)[];
  best_pak_f1?: (number | null)[];
  nan_inf_count?: (number | null)[];
  gpu_util?: (number | null)[];
  gpu_mem_mb?: (number | null)[];
  cpu_load1?: (number | null)[];
  ram_used_mb?: (number | null)[];
  verdicts?: (string | null)[];
  [k: string]: any;
}

export interface VizManifest {
  exp_id: string;
  dataset_key: string;
  variant: string | null;
  categories: Record<string, string[]>;
  abs_paths?: Record<string, string>;
  stale: string[];
  stale_reason?: string | null;
  gif_available: boolean;
  gifs: { cache_key: string; story_id: number; metric_keys: string[] }[];
}

export interface GifStorySubMode {
  id: string;
  label: string;
  source?: string;
  data_axis?: string;
  default_sub_metric?: string;
}
export interface GifStory {
  story_id: number;
  story: string;
  label: string;
  allowed_metric_families: string[];
  default_metric_keys: string[];
  pair_with?: string[];
  overlay_with?: string[];
  chart_kinds: string[];
  data_axes: string[];
  priority: string;
  sub_modes: GifStorySubMode[];
  /** FB-R3-10/11: the VISIBLE display position (1..N; null = hidden from the primary
   * list, e.g. compare_grid), the rendered label (e.g. "GIF-1"), and a grounded
   * one-line description. `story_id` stays the STABLE internal id. */
  display_order?: number | null;
  display_label?: string | null;
  description?: string | null;
  /** FB-R4-02: present ONLY on the metric_diff (A − B) story — the default A/B metric
   * pair the renderer diffs (`params.metric_a` / `params.metric_b`). The picker seeds
   * its A/B from here and writes the user's choice back into `GifSpec.params`. */
  diff_params?: { metric_a?: string; metric_b?: string };
}

/** FB-R4b-01: the GIF-7 (what_carries) togglable ceiling/marker variants advertised by
 * GET /gif/stories.what_carries_ceilings. adaptive/teacher carry a TRUE ceiling
 * (max-of-series); student/disc are best-epoch SCALARS only (NOT ceilings, P3-01). The
 * GIF Studio renders one checkbox per variant; the checked set → `params.show_ceilings`. */
export interface WhatCarriesCeiling {
  variant: string; // "metrics" | "teacher" | "student" | "disc"
  metadata_block: string;
  has_series: boolean;
  marker_kind: "ceiling" | "best_epoch";
}
/** FB-R5-01: the signal_lives feature_heatmap color scales advertised by GET
 * /gif/stories.signal_lives_heatmap_scales. sqrt (PowerNorm γ=0.5) is the faithful
 * default that lifts small-but-nonzero features off the near-zero floor; linear shows the
 * original wash-out; log reveals the low end most strongly. The GIF Studio renders one
 * button per scale; the choice → `params.heatmap_scale` (∈ cache key → re-render). */
export interface SignalLivesHeatmapScales {
  scales: string[]; // ["linear","sqrt","log"]
  default: string; // "sqrt"
  sub_mode: string; // "feature_heatmap"
  param: string; // "heatmap_scale"
}
export interface GifStoriesResp {
  stories: GifStory[];
  what_carries_ceilings?: WhatCarriesCeiling[];
  signal_lives_heatmap_scales?: SignalLivesHeatmapScales;
}

export interface GifSpec {
  story: string;
  metric_keys: string[];
  experiment_set: LeafSel[];
  variant?: string | null;
  max_epoch?: number | null;
  params?: Record<string, any>;
}
export interface GifRenderResp {
  job_id: string | null;
  cache_key: string;
  cached: boolean;
}
export interface GifStatus {
  status: "pending" | "rendering" | "done" | "error";
  frame?: number;
  total?: number;
  cache_key?: string;
  error?: string;
  /** R3-SR-03: present when status==="done" — the eval epoch shown on EACH rendered
   * frame (frame index → epoch) + the trailing peak-hold duplicate-final-epoch count.
   * Empty `[]`/0 for the NPZ hist-drift + feature-heatmap stories (epoch drawn on-frame). */
  frame_epochs?: number[];
  hold_len?: number;
}

/** FB-R3-04/05: a decoded GIF frame from gifuct-js `decompressFrames(gif, true)` — the
 * RGBA `patch` is composed onto a persistent full-frame canvas by <GifViewer>. */
export interface ParsedGifFrame {
  dims: { width: number; height: number; top: number; left: number };
  delay: number;
  disposalType: number;
  patch: Uint8ClampedArray;
}

/** R3-SR-03: the full GIF sidecar — the instant cache-hit path (POST /render returned
 * cached:true, no job to poll) reads this from GET /api/gif/sidecar/{cache_key}. */
export interface GifSidecar {
  cache_key?: string;
  story_id?: number;
  n_frames?: number;
  frame_epochs?: number[];
  hold_len?: number;
  [k: string]: any;
}
export interface GifListItem {
  cache_key: string;
  story_id: number;
  metric_keys: string[];
  experiment_set?: LeafSel[];
  variant?: string | null;
  max_epoch?: number | null;
  mtime?: number;
}

export interface Health {
  status: string;
  resolver_selftest?: string | { status: string; [k: string]: any };
  registry?: string;
  cache?: any;
  roots?: any;
}

export interface PanelList {
  panels: { name: string; cost?: string; requires?: string[] }[];
}
