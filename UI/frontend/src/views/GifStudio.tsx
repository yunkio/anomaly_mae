/* GIF Studio (flagship F1, frontend-design §6) — first-class showcase route.
 *
 * • Story picker bound to GET /gif/stories (single backend source, P2-04) — never
 *   re-derives metrics from prose.
 * • Metric picker = runtime catalog intersected with the story's allowed families
 *   (F5); default = the story's default_metric_keys.
 * • GIF-6 (what_carries) constrains to a SINGLE per-variant metric (P3-01).
 * • GIF-5 (signal_lives) exposes the 2-level pivot sub-metric selector (P3-02) and
 *   the sub_modes[] (wired to params.sub_mode + params.sub_metric, IMPL-FB-gifparams).
 * • Side-by-side synchronized animated compare across the pin set (<GifCompare>). */
import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { useGifStories, useLossDriftMetrics, useMetricsCatalog, useTrainingHistory } from "../api/queries";
import GifViewer from "../components/gif/GifViewer";
import AsyncPanel from "../components/AsyncPanel";
import MetricPicker from "../components/MetricPicker";
import { Card } from "../components/common";
import { metricCatalogView } from "../api/metricView";
import { useStore } from "../store";
import { useScopeDataset, shortModel } from "../scope";
import type { CatalogEntry, GifSpec, GifStory, LeafSel } from "../api/types";

const NESTED_KEY = "epoch_anomaly_type_scores";

export default function GifStudio() {
  const storiesQ = useGifStories();
  const pins = useStore((s) => s.pins);
  const setScope = useStore((s) => s.setScope);
  // F-9 (A-5): a deep link from the Gallery carries ?leaf=<leafId>&story=<story>. Seed the
  // story + the selected leaf from those params ONCE (lazy useState initializers), so the
  // Studio opens already focused on THE leaf the user came from — not a generic default.
  const [urlParams] = useSearchParams();
  const [storyId, setStoryId] = useState<string>(() => urlParams.get("story") || "climb_plateau");
  const [metricKeys, setMetricKeys] = useState<string[]>(["pak_auc_f1"]);
  const [subMode, setSubMode] = useState<string>("");
  const [subMetric, setSubMetric] = useState<string>("recon_ratio");
  const [compare, setCompare] = useState(false);
  // FB-R5-01: the signal_lives feature_heatmap COLOR scale (linear | sqrt | log; default
  // sqrt = PowerNorm γ=0.5, faithful — small-but-nonzero features no longer wash out).
  // null = the backend-advertised default; the chosen value → params.heatmap_scale (∈
  // cache key → re-render). Reset when the story changes (below).
  const [heatmapScale, setHeatmapScale] = useState<string | null>(null);
  // FB-R4-02: the metric-diff (A − B) story's A/B pair. Seeded from the story's
  // diff_params (backend manifest); chosen via the registry-driven <MetricPicker> (F5).
  const [diffA, setDiffA] = useState<string>("pak_auc_f1");
  const [diffB, setDiffB] = useState<string>("teacher_pak_auc_f1");
  // FB-R4b-01: GIF-7 (what_carries) per-ceiling toggles. The CHECKED set of variant keys
  // (adaptive=`metrics` / `teacher` / `student` / `disc`) drives `params.show_ceilings`,
  // re-rendering the GIF (each combination caches independently — show_ceilings is inside
  // the spec params → inside the cache key). `null` = all variants checked (default).
  const [shownCeilings, setShownCeilings] = useState<string[] | null>(null);
  // Which pinned MODEL(s) to render. Single-leaf stories → one chosen model (default
  // pins[0]); race/compare stories → a checked SUBSET (default all). The DATASET is the
  // shared scope (picked below), so a "leaf" = model × scope dataset.
  // F-9: seed the model + dataset scope from the ?leaf=exp|ds|variant deep-link.
  const [selModel, setSelModel] = useState<string | null>(() => {
    const lid = urlParams.get("leaf");
    return lid ? lid.split("|")[0] : null;
  });
  const [checkedModels, setCheckedModels] = useState<string[] | null>(null); // null = all pins

  const story: GifStory | undefined = useMemo(
    () => storiesQ.data?.stories.find((s) => s.story === storyId),
    [storiesQ.data, storyId]
  );

  // GIF-6/7 (what_carries) = single metric (P3-01); bar_race + line_race = across the pin set.
  const isSingleMetric = story?.story === "what_carries";
  // FB-R4b-01: the togglable ceiling/marker variants (adaptive/teacher carry a TRUE
  // ceiling = max-of-series; student/disc are best-epoch SCALARS, P3-01) — advertised by
  // the backend so the checkbox list is never a hard-coded enumeration.
  const ceilingVariants = storiesQ.data?.what_carries_ceilings ?? [];
  // the set of variant keys CURRENTLY drawn (default = all advertised variants).
  const shownCeilingSet = useMemo(
    () => (shownCeilings == null ? ceilingVariants.map((v) => v.variant) : shownCeilings),
    [shownCeilings, ceilingVariants]
  );
  const isBarRace = story?.story === "bar_race";
  const isLineRace = story?.story === "line_race";
  const isMultiLeaf = isBarRace || isLineRace; // stories that animate the whole pin set
  const isLossDrift = story?.story === "loss_drift"; // GIF-3 — training-metric inventory
  const isMetricDiff = story?.story === "metric_diff"; // FB-R4-02 — (A − B) over epochs

  // the chosen single MODEL (default pins[0]); falls back to pins[0] if the selected model
  // was unpinned. All single-leaf queries key off THIS model × the scope dataset.
  const firstModel = useMemo(
    () => (selModel && pins.includes(selModel) ? selModel : pins[0]),
    [pins, selModel]
  );
  // the multi-model subset (default = all pins) for the race/compare stories.
  const multiModels = useMemo(() => {
    if (checkedModels == null) return pins;
    const sub = pins.filter((p) => checkedModels.includes(p));
    return sub.length ? sub : pins;
  }, [pins, checkedModels]);

  // the shared dataset scope: single-leaf → the chosen model's datasets; race → the raced
  // models' COMMON datasets. A "leaf" = model × (scope dataset, scope variant).
  const scopeModels = isMultiLeaf ? multiModels : firstModel ? [firstModel] : [];
  const scope = useScopeDataset(scopeModels);

  // F-9: once, seed the dataset scope from the ?leaf= deep-link (exp|ds|variant).
  useEffect(() => {
    const lid = urlParams.get("leaf");
    if (!lid) return;
    const [, ds, variant] = lid.split("|");
    if (ds) setScope(ds, variant && variant !== "_" ? variant : null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const firstLeaf: LeafSel | undefined =
    firstModel && scope.datasetKey
      ? { exp_id: firstModel, dataset_key: scope.datasetKey, variant: scope.variant }
      : undefined;
  const multiSet: LeafSel[] = scope.datasetKey
    ? multiModels.map((m) => ({ exp_id: m, dataset_key: scope.datasetKey as string, variant: scope.variant }))
    : [];

  // metric catalog from the SELECTED pinned leaf (any discovered metric animates, F5).
  const cat = useMetricsCatalog(firstLeaf?.exp_id, firstLeaf?.dataset_key, firstLeaf?.variant);
  // FB-14: for GIF-3 (loss_drift), surface ALL collected per-training-epoch metrics from
  // the leaf's grouped inventory (history namespace), not just the epoch-metrics catalog.
  const lossDrift = useLossDriftMetrics(
    isLossDrift ? firstLeaf?.exp_id : undefined,
    isLossDrift ? firstLeaf?.dataset_key : undefined,
    firstLeaf?.variant
  );

  // F-05: the GIF-5 sub-metric options come from the story's default_sub_metric +
  // the LIVE pivot's sub_metrics (mirrors <TrainingHistoryPanel>), never a literal list.
  const isSignalLives = story?.story === "signal_lives";
  // FB-R5-01: the advertised feature_heatmap color scales (never a hard-coded literal —
  // sourced from GET /gif/stories.signal_lives_heatmap_scales). The active scale defaults
  // to the backend default (sqrt) when the user has not overridden it.
  const heatmapScalesAdv = storiesQ.data?.signal_lives_heatmap_scales;
  const heatmapScaleDefault = heatmapScalesAdv?.default ?? "sqrt";
  const activeHeatmapScale = heatmapScale ?? heatmapScaleDefault;
  // the selector shows ONLY for the signal_lives feature_heatmap sub_mode (not the
  // anom_type_over_training sub_mode, which is a line story with no heatmap norm).
  const isFeatureHeatmap =
    isSignalLives && (subMode === (heatmapScalesAdv?.sub_mode ?? "feature_heatmap"));
  const pivotQ = useTrainingHistory(
    firstLeaf?.exp_id,
    firstLeaf?.dataset_key,
    isSignalLives ? [NESTED_KEY] : [],
    firstLeaf?.variant
  );
  const subMetricOptions = useMemo(() => {
    const opts = new Set<string>();
    // seed from every sub_mode's declared default_sub_metric
    story?.sub_modes?.forEach((sm) => sm.default_sub_metric && opts.add(sm.default_sub_metric));
    // add the live pivot's discovered sub_metrics
    (pivotQ.data?.nested?.[NESTED_KEY]?.sub_metrics ?? []).forEach((m) => opts.add(m));
    if (subMetric) opts.add(subMetric);
    return Array.from(opts);
  }, [story?.story, pivotQ.data, subMetric]);

  useEffect(() => {
    if (story) {
      setMetricKeys(story.default_metric_keys.length ? story.default_metric_keys.slice(0, isSingleMetric ? 1 : undefined) : ["pak_auc_f1"]);
      const sm = story.sub_modes?.[0];
      setSubMode(sm?.id ?? "");
      setSubMetric(sm?.default_sub_metric ?? "recon_ratio");
      // FB-R4-02: seed the diff A/B from the story's diff_params (manifest default),
      // falling back to its default_metric_keys[0]/[1] — never a hard-coded literal.
      const dp = story.diff_params;
      setDiffA(dp?.metric_a ?? story.default_metric_keys[0] ?? "pak_auc_f1");
      setDiffB(dp?.metric_b ?? story.default_metric_keys[1] ?? "teacher_pak_auc_f1");
      // FB-R4b-01: reset the GIF-7 ceiling toggles to the default (all on) when the story
      // changes — so a prior subset never silently carries into another story.
      setShownCeilings(null);
      // FB-R5-01: reset the feature_heatmap color scale to the backend default (sqrt) when
      // the story changes — a prior linear/log choice never silently carries over.
      setHeatmapScale(null);
    }
  }, [storyId, story?.story]);

  const allowedFamilies = story?.allowed_metric_families ?? [];
  const familyMetrics = useMemo(() => {
    // FB-14: GIF-3 (loss_drift) draws from the FULL per-training-epoch history inventory
    // (teacher/student recon, discrepancy, SNR/ratios, per-type) grouped by registry
    // family — so every collected training metric is selectable, not just a couple. The
    // list is the runtime UNION (F5); disabled components are listed separately below.
    if (isLossDrift) {
      const fams = lossDrift.data?.families ?? {};
      const out: any[] = [];
      Object.keys(fams)
        .sort()
        .forEach((fam) =>
          fams[fam].forEach((it) =>
            out.push({ key: it.key, display_name: it.display_name, direction: it.direction, inferred: it.inferred, family: fam })
          )
        );
      return out;
    }
    // Route through the SHARED metricCatalogView() chokepoint so the GIF metric picker
    // gets the same FB-1 de-dup + FB-5 pa-keep + FB-6 excl22 policy as every other
    // picker (no duplicate excl22 chips, no pa_* flood). Scope to epoch_metrics (the
    // per-eval-epoch animation source) + the story's allowed families.
    const base = (cat.data ?? []).filter((c) => c.namespace === "epoch_metrics");
    return metricCatalogView(base, { leafVariant: firstLeaf?.variant }).filter((c) =>
      allowedFamilies.some((f) => (c.family || "").startsWith(f))
    );
  }, [cat.data, allowedFamilies.join(","), isLossDrift, lossDrift.data, firstLeaf?.variant]);

  // FB-R4-02: the A/B pickers draw from the SELECTED leaf's RAW runtime catalog (F5);
  // <MetricPicker> scopes to namespace=epoch_metrics — the SAME per-eval-epoch source the
  // backend `_frames_metric_diff` diffs — and applies the shared dedup/pa-keep/excl22 view
  // (metricCatalogView) internally. Any discovered metric (incl. teacher_* per-epoch
  // series) is pickable; a best-epoch-only metric simply yields P3-01 gaps in the diff.
  const diffCatalog: CatalogEntry[] = cat.data ?? [];

  // FB-R3-07: render the USER-CHOSEN pinned experiment(s).
  const expSet: LeafSel[] = isMultiLeaf ? multiSet : firstLeaf ? [firstLeaf] : [];

  const params: Record<string, any> = {};
  if (subMode) params.sub_mode = subMode;
  if (story?.story === "signal_lives") params.sub_metric = subMetric;
  // FB-R5-01: thread the feature_heatmap color scale ONLY for that sub_mode, and ONLY when
  // it differs from the backend default — so the common (default sqrt) render matches a
  // server call with no heatmap_scale and keeps that cache key (the default sqrt path is
  // what render_feature_heatmap uses when heatmap_scale is absent).
  if (isFeatureHeatmap && activeHeatmapScale !== heatmapScaleDefault) {
    params.heatmap_scale = activeHeatmapScale;
  }
  // FB-R4-02: the metric-diff story is driven by the A/B pair, not the multi-metric set.
  if (isMetricDiff) {
    params.metric_a = diffA;
    params.metric_b = diffB;
  }
  // FB-R4b-01: the GIF-7 ceiling toggles. Only emit when the variant set is a PROPER
  // subset of all advertised variants (so the default ALL-on render matches a server
  // call with no show_ceilings, keeping the existing cache key for the common case).
  if (isSingleMetric && ceilingVariants.length > 0 && shownCeilingSet.length !== ceilingVariants.length) {
    params.show_ceilings = shownCeilingSet;
  }

  const spec: GifSpec | null =
    story && expSet.length > 0
      ? {
          story: story.story,
          metric_keys: metricKeys.length ? metricKeys : ["pak_auc_f1"],
          experiment_set: expSet,
          variant: firstLeaf?.variant ?? null,
          max_epoch: null,
          params,
        }
      : null;

  return (
    <div className="col" style={{ gap: 24 }}>
      <div>
        <h1 className="view-title">GIF Studio</h1>
        <p className="view-sub">
          Flagship epoch-progression animations — CPU-rendered, cached, phase-masked, direction-aware. Pick a story,
          choose metrics from the live catalog, and the pinned run set.
        </p>
      </div>

      <AsyncPanel query={storiesQ} isEmpty={(d) => !d.stories?.length} height={120}>
        {(d) => (
          <div className="row wrap" style={{ gap: 24, alignItems: "flex-start" }}>
            <div style={{ flex: "1 1 320px", minWidth: 300 }}>
              <Card title="Story">
                <div className="col" style={{ gap: 8 }}>
                  {/* FB-R3-10: compare_grid (display_order=null) is HIDDEN from the
                      primary list — reached via the "side-by-side compare" toggle below.
                      The visible stories are sorted by display_order and rendered with
                      their display_label (1..N contiguous; line_race is no longer "GIF-8"
                      sitting between 5 and 6). */}
                  {d.stories
                    .filter((s) => s.story !== "compare_grid" && s.display_order != null)
                    .slice()
                    .sort((a, b) => (a.display_order ?? 999) - (b.display_order ?? 999))
                    .map((s) => (
                      <button
                        key={s.story}
                        className={`btn ${storyId === s.story ? "primary" : ""}`}
                        style={{ textAlign: "left" }}
                        onClick={() => setStoryId(s.story)}
                        title={s.description ?? undefined}
                      >
                        <div style={{ fontWeight: 600 }}>
                          {s.display_label ?? `GIF-${s.display_order}`} · {s.label}
                        </div>
                        {s.description && (
                          <div className="subtle" style={{ fontSize: "var(--fs-small)", whiteSpace: "normal" }}>
                            {s.description}
                          </div>
                        )}
                        <div className="subtle" style={{ fontSize: "var(--fs-small)" }}>
                          {s.data_axes.join(" / ")} · {s.priority}
                        </div>
                      </button>
                    ))}
                </div>
              </Card>

              <div style={{ height: 16 }} />

              <Card title="Render set">
                <div className="col" style={{ gap: 10 }}>
                  <div className="subtle" style={{ fontSize: "var(--fs-small)" }}>
                    {pins.length === 0
                      ? "Pin at least one model (Overview / Rankings)."
                      : scope.noCommon
                      ? "The pinned models share no dataset — pick models that overlap."
                      : isMultiLeaf
                      ? `${isLineRace ? "line race" : "bar-chart race"} across ${multiModels.length} model(s) on ${scope.datasetKey ?? "—"}`
                      : firstModel
                      ? `${shortModel(firstModel)} on ${scope.datasetKey ?? "—"}${scope.variant && scope.variant !== "_" ? " · " + scope.variant : ""}`
                      : "pick a model"}
                  </div>

                  {/* dataset scope: every "leaf" = model × this dataset. */}
                  {pins.length > 0 && scope.options.length > 0 && (
                    <label className="row wrap" style={{ gap: 6, alignItems: "center" }}>
                      <span style={{ fontSize: "var(--fs-small)", fontWeight: 600 }}>dataset</span>
                      <select
                        className="select sm"
                        value={scope.datasetKey ?? ""}
                        onChange={(e) => scope.setDatasetKey(e.target.value)}
                      >
                        {scope.options.map((o) => (
                          <option key={o.dataset_key} value={o.dataset_key}>
                            {o.dataset_key}
                          </option>
                        ))}
                      </select>
                      {scope.current && scope.current.variants.filter((v) => v && v !== "_").length > 1 && (
                        <span className="seg" role="group" aria-label="variant">
                          {scope.current.variants
                            .filter((v) => v && v !== "_")
                            .map((v) => (
                              <button
                                key={v}
                                className={`seg-btn ${scope.variant === v ? "active" : ""}`}
                                onClick={() => scope.setVariant(v)}
                              >
                                {v}
                              </button>
                            ))}
                        </span>
                      )}
                    </label>
                  )}

                  {/* choose WHICH pinned MODEL(s) to render. Single-leaf stories → a <select>
                      over the pinned models; race/compare → a checkbox multi-select. */}
                  {pins.length > 0 && (
                    isMultiLeaf ? (
                      <div className="col" style={{ gap: 4 }}>
                        <div className="subtle" style={{ fontSize: "var(--fs-small)", fontWeight: 600 }}>
                          models to race ({multiModels.length}/{pins.length})
                        </div>
                        <div className="row wrap" style={{ gap: 6 }}>
                          <button
                            className={`btn sm ${checkedModels == null ? "primary" : ""}`}
                            onClick={() => setCheckedModels(null)}
                            title="race all pinned models"
                          >
                            all
                          </button>
                          {pins.map((m) => {
                            const on = checkedModels == null ? true : checkedModels.includes(m);
                            return (
                              <button
                                key={m}
                                className={`btn sm ${on ? "primary" : ""}`}
                                title={m}
                                onClick={() =>
                                  setCheckedModels((cur) => {
                                    const base = cur == null ? [...pins] : cur;
                                    return base.includes(m) ? base.filter((x) => x !== m) : [...base, m];
                                  })
                                }
                              >
                                {shortModel(m)}
                              </button>
                            );
                          })}
                        </div>
                      </div>
                    ) : (
                      <label className="row" style={{ gap: 6 }}>
                        model
                        <select className="select" value={firstModel ?? ""} onChange={(e) => setSelModel(e.target.value)}>
                          {pins.map((m) => (
                            <option key={m} value={m}>
                              {shortModel(m)}
                            </option>
                          ))}
                        </select>
                      </label>
                    )
                  )}

                  {/* FB-R4-02: the metric-diff story is driven by an A / B metric PAIR
                      (registry-driven <MetricPicker>, F5) — not the multi-metric set. The
                      rendered GIF is the per-eval-epoch (A − B) curve; if either metric is
                      best-epoch-only (no per-epoch series) the diff line shows P3-01 gaps. */}
                  {story && isMetricDiff && (
                    <div className="col" style={{ gap: 6 }}>
                      <div className="subtle" style={{ fontSize: "var(--fs-small)", fontWeight: 600 }}>
                        metric difference — animates (A − B) per eval epoch (P3-01 gaps where
                        either metric has no per-epoch value)
                      </div>
                      <MetricPicker
                        label="A"
                        catalog={diffCatalog}
                        value={diffA}
                        onChange={setDiffA}
                        namespaceFilter={["epoch_metrics"]}
                        loading={cat.isLoading}
                        disabled={!firstLeaf}
                        leafVariant={firstLeaf?.variant}
                      />
                      <MetricPicker
                        label="B"
                        catalog={diffCatalog}
                        value={diffB}
                        onChange={setDiffB}
                        namespaceFilter={["epoch_metrics"]}
                        loading={cat.isLoading}
                        disabled={!firstLeaf}
                        leafVariant={firstLeaf?.variant}
                      />
                      <div className="subtle mono" style={{ fontSize: "var(--fs-small)" }}>
                        rendering: <strong>{diffA}</strong> − <strong>{diffB}</strong>
                      </div>
                    </div>
                  )}

                  {/* metric picker (single for GIF-6; chip grid for the rest) */}
                  {story && !isMetricDiff && (
                    <div className="col" style={{ gap: 6 }}>
                      <div className="subtle" style={{ fontSize: "var(--fs-small)", fontWeight: 600 }}>
                        {isSingleMetric
                          ? "metric (single — score variants of one metric, P3-01)"
                          : isLossDrift
                          ? "training metrics (ALL collected, grouped by family — FB-14)"
                          : "metrics (default from story)"}
                      </div>
                      <div className="row wrap" style={{ gap: 6 }}>
                        {(familyMetrics.length ? familyMetrics : story.default_metric_keys.map((k) => ({ key: k, display_name: k, direction: "up", inferred: false } as any))).map((c: any) => {
                          const on = metricKeys.includes(c.key);
                          return (
                            <button
                              key={c.key}
                              className={`btn sm ${on ? "primary" : ""}`}
                              onClick={() =>
                                isSingleMetric
                                  ? setMetricKeys([c.key])
                                  : setMetricKeys(on ? metricKeys.filter((k) => k !== c.key) : [...metricKeys, c.key])
                              }
                            >
                              {c.display_name || c.key}
                              {c.inferred ? " · inf" : ""}
                            </button>
                          );
                        })}
                      </div>
                      {isLossDrift && (lossDrift.data?.disabled?.length ?? 0) > 0 && (
                        <div className="subtle" style={{ fontSize: "var(--fs-small)" }}>
                          {lossDrift.data!.disabled.length} component(s) off for this run (e.g.{" "}
                          {lossDrift.data!.disabled.slice(0, 3).map((d) => d.key).join(", ")}…) — not
                          selectable until enabled.
                        </div>
                      )}
                    </div>
                  )}

                  {/* FB-R4b-01/02: GIF-7 (what_carries) per-ceiling toggles. One checkbox
                      per advertised score variant — adaptive/teacher draw a TRUE ceiling
                      (max-of-series, the line can never exceed it); student/disc are
                      best-epoch SCALARS only (P3-01, NOT ceilings, shown as a labeled
                      observed point). Toggling re-renders the GIF (show_ceilings ∈ params
                      → cache key, so each combination caches independently). */}
                  {isSingleMetric && ceilingVariants.length > 0 && (
                    <div className="col" style={{ gap: 6 }}>
                      <div className="subtle" style={{ fontSize: "var(--fs-small)", fontWeight: 600 }}>
                        ceilings / markers (toggle each independently)
                      </div>
                      <div className="row wrap" style={{ gap: 10 }}>
                        {ceilingVariants.map((cv) => {
                          const label =
                            cv.variant === "metrics"
                              ? "adaptive max (series)"
                              : cv.variant === "teacher"
                              ? "teacher max (series)"
                              : cv.variant === "student"
                              ? "student best-epoch"
                              : cv.variant === "disc"
                              ? "disc best-epoch"
                              : cv.variant;
                          const on = shownCeilingSet.includes(cv.variant);
                          return (
                            <label
                              key={cv.variant}
                              className="row"
                              style={{ gap: 4 }}
                              title={
                                cv.marker_kind === "ceiling"
                                  ? "TRUE ceiling = max of this variant's own per-epoch series (the line never exceeds it)"
                                  : "best-epoch scalar — one observed point, not a ceiling (P3-01)"
                              }
                            >
                              <input
                                type="checkbox"
                                checked={on}
                                onChange={() =>
                                  setShownCeilings((cur) => {
                                    const base = cur == null ? ceilingVariants.map((v) => v.variant) : cur;
                                    return base.includes(cv.variant)
                                      ? base.filter((x) => x !== cv.variant)
                                      : [...base, cv.variant];
                                  })
                                }
                              />
                              {label}
                              <span className="subtle" style={{ fontSize: "var(--fs-small)" }}>
                                {cv.marker_kind === "ceiling" ? " (ceiling)" : " (best-epoch)"}
                              </span>
                            </label>
                          );
                        })}
                      </div>
                      <div className="subtle" style={{ fontSize: "var(--fs-small)" }}>
                        adaptive / teacher = max-of-series ceilings (the drawn line never exceeds
                        them); student / disc = best-epoch observed points (P3-01, not ceilings). A
                        small metadata-vs-series gap from the λ-recompute is annotated on the
                        ceiling label rather than hidden.
                      </div>
                    </div>
                  )}

                  {/* sub_modes -> params.sub_mode (IMPL-FB-gifparams) */}
                  {story && story.sub_modes?.length > 0 && (
                    <label className="row" style={{ gap: 6 }}>
                      sub-mode
                      <select className="select" value={subMode} onChange={(e) => setSubMode(e.target.value)}>
                        {story.sub_modes.map((sm) => (
                          <option key={sm.id} value={sm.id}>
                            {sm.label}
                          </option>
                        ))}
                      </select>
                    </label>
                  )}

                  {/* GIF-5 2-level pivot sub-metric selector (P3-02). Options come from
                      the story default_sub_metric + the LIVE pivot sub_metrics (F-05). */}
                  {isSignalLives && (
                    <label className="row" style={{ gap: 6 }}>
                      sub-metric (per-type pivot)
                      <select className="select" value={subMetric} onChange={(e) => setSubMetric(e.target.value)}>
                        {(subMetricOptions.length ? subMetricOptions : [subMetric]).map((m) => (
                          <option key={m} value={m}>
                            {m}
                          </option>
                        ))}
                      </select>
                    </label>
                  )}

                  {/* FB-R5-01: GIF-6 (signal_lives) feature_heatmap COLOR scale. A
                      heavy-tailed per-feature recon distribution washes out on a LINEAR
                      norm (the ≈0.001 bulk maps to ~17% color → reads as "no influence");
                      sqrt (PowerNorm γ=0.5, default) lifts it to a clearly non-black color
                      WITHOUT inverting the ordering; log reveals the low end most strongly.
                      The choice → params.heatmap_scale (∈ cache key → re-render); the
                      colorbar keeps REAL magnitudes. Shown ONLY for the feature_heatmap
                      sub_mode (not the anom_type line sub_mode). */}
                  {isFeatureHeatmap && (heatmapScalesAdv?.scales?.length ?? 0) > 0 && (
                    <div className="col" style={{ gap: 4 }}>
                      <div className="subtle" style={{ fontSize: "var(--fs-small)", fontWeight: 600 }}>
                        heatmap color scale (faithful low-end reveal)
                      </div>
                      <div className="row wrap" style={{ gap: 6 }}>
                        {heatmapScalesAdv!.scales.map((s) => (
                          <button
                            key={s}
                            className={`btn sm ${activeHeatmapScale === s ? "primary" : ""}`}
                            onClick={() => setHeatmapScale(s)}
                            title={
                              s === "linear"
                                ? "linear norm — the original (heavy-tailed data washes the low-mid features toward black)"
                                : s === "sqrt"
                                ? "sqrt (PowerNorm γ=0.5) — DEFAULT; lifts small-but-nonzero features off the floor, ordering unchanged"
                                : s === "log"
                                ? "log (LogNorm) — strongest low-end reveal; vmin floored >0"
                                : s
                            }
                          >
                            {s}
                            {s === heatmapScaleDefault ? " · default" : ""}
                          </button>
                        ))}
                      </div>
                      <div className="subtle" style={{ fontSize: "var(--fs-small)" }}>
                        sqrt/log brighten the low-mid features (small-but-nonzero ≠ zero); the
                        colorbar still shows the REAL magnitudes and a "scale:" tag is drawn
                        on the GIF. Switch to linear to see the original.
                      </div>
                    </div>
                  )}

                  <label className="row" style={{ gap: 6 }}>
                    <input type="checkbox" checked={compare} onChange={(e) => setCompare(e.target.checked)} />
                    side-by-side compare across pins
                  </label>
                </div>
              </Card>
            </div>

            <div style={{ flex: "2 1 520px", minWidth: 360 }}>
              {/* FB-R3-10: the header uses the DISPLAY label (not the internal story_id). */}
              <Card title={story ? `${story.display_label ?? `GIF-${story.display_order ?? story.story_id}`} · ${story.label}` : "Preview"}>
                {/* FB-R3-11: the selected story's grounded description. */}
                {story?.description && (
                  <div className="subtle" style={{ fontSize: "var(--fs-small)", marginBottom: 8, whiteSpace: "normal" }}>
                    {story.description}
                  </div>
                )}
                {compare && !isMultiLeaf ? (
                  <GifCompare baseSpec={spec} pins={multiSet} story={story} />
                ) : (
                  <GifViewer spec={spec} height={400} />
                )}
                {story && (
                  <div className="subtle" style={{ fontSize: "var(--fs-small)", marginTop: 8 }}>
                    families: {story.allowed_metric_families.join(", ")}
                  </div>
                )}
              </Card>
            </div>
          </div>
        )}
      </AsyncPanel>
    </div>
  );
}

/* <GifCompare> — GENUINELY synchronized side-by-side animated compare (F-03).
 * Instead of N independent native-GIF <img> players (which drift), this binds a SINGLE
 * <GifViewer> to the backend `compare_grid` story: one combined GIF with one sub-panel
 * per pinned leaf, revealed in LOCKSTEP over the union epoch axis — the playhead is
 * synchronized by construction (frontend-design §6.3, DECISION_LOG #22). */
function GifCompare({ baseSpec, pins, story }: { baseSpec: GifSpec | null; pins: LeafSel[]; story?: GifStory }) {
  if (!baseSpec || pins.length === 0) return <div className="subtle">Pin ≥1 run to compare.</div>;
  const panels = pins.slice(0, 4);
  const gridSpec: GifSpec = {
    story: "compare_grid",
    metric_keys: baseSpec.metric_keys,
    experiment_set: panels,
    variant: null,
    max_epoch: baseSpec.max_epoch ?? null,
    params: {},
  };
  return (
    <div className="col" style={{ gap: 8 }}>
      <GifViewer
        spec={gridSpec}
        height={420}
        title={`Synchronized compare across ${panels.length} run(s)${story ? ` · ${story.label}` : ""}`}
      />
      <div className="subtle" style={{ fontSize: "var(--fs-small)" }}>
        One combined GIF — all panels share a single frame index (synchronized playhead),
        aligned by epoch number; gaps are real gaps.
      </div>
    </div>
  );
}
