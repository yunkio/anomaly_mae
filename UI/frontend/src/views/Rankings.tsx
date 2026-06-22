/* Rankings (IA-8) — REBUILT for R2 (FB-2/FB-11/FB-12/FB-13), redesigned for R4c
 * (FB-R4c-01..04 aggregate-ranking redesign).
 *
 * The TRUE model ranking (FB-13): within EACH dataset COLUMN rank the experiments by the
 * selection metric (default pak_auc_f1, direction from the registry), then a model's
 * FINAL rank = the AVERAGE of its per-column ranks. Bound to GET /api/rankings/aggregate.
 *
 * R4c column model (backend-driven):
 *  • SMD/SMAP/MSL COLLAPSE into ONE "(avg)" column each = mean over the exp's per-
 *    machine/channel leaves (concat excluded), counted ONCE (FB-R4c-01);
 *  • SWaT default column = `· excl22`; `· full` is selectable but not default (FB-R4c-02);
 *  • the Dataset filter row includes SMD (avg) / SMAP (avg) / MSL (avg) chips alongside
 *    the existing canonical chips (FB-R4c-03);
 *  • a registry-driven ranking METRIC selector (F5) AND a fully-configurable DATASET-SET
 *    selection over the backend's advertised universe (canonical units + every granular
 *    machine/channel/concat + SWaT full), DEFAULT = the canonical set (FB-R4c-04).
 *
 * The metric OPTION LIST is the runtime UNION catalog (F-01/F5), never a hard-coded array.
 * pin-from-ranking (FB-R3-06) still works with the new (avg) rows (see representativeLeaf). */
import { useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { Card, fmt } from "../components/common";
import MetricPicker from "../components/MetricPicker";
import MetricQuickPicks from "../components/MetricQuickPicks";
import PerformanceBasis from "../components/PerformanceBasis";
import PerDatasetRankTable from "../components/PerDatasetRankTable";
import GifViewer from "../components/gif/GifViewer";
import LineRace from "../components/LineRace";
import AsyncPanel from "../components/AsyncPanel";
import { useGlobalMetricsCatalog, useRankingsAggregate } from "../api/queries";
import { useStore, catColor } from "../store";
import { useScopeDataset } from "../scope";
import type { GifSpec, LeafSel, RankColumnMeta } from "../api/types";

/** leaf_id is "exp|dataset|variant"; the model is the first segment. */
const expOf = (leafId: string): string => leafId.split("|")[0];

/* A dataset COLUMN key is the bare dataset_key, OR `SWaT_A1A2 · full`/`· excl22` for the
 * SWaT split (FB-11), OR `<G> (avg)` for a collapsed SMD/SMAP/MSL group (FB-R4c-01).
 * Recover a (dataset_key, variant) LEAF to pin from a column. */
function columnToLeaf(col: string, expId: string, columnMeta?: Record<string, RankColumnMeta>): LeafSel {
  const swat = col.match(/^(.*) · (full|excl22)$/);
  if (swat) return { exp_id: expId, dataset_key: swat[1], variant: swat[2] };
  // FB-R4c-01 / R3-SR-04 pin rule for an "(avg)" column: an aggregate "(avg)" row spans
  // many machine/channel leaves, so it is NOT itself a pinnable leaf. We pin a
  // REPRESENTATIVE granular leaf — the FIRST per-entity member of the group by sorted
  // dataset_key (e.g. SMD (avg) → SMD/machine-1-2). The exact leaf can be refined from the
  // per-dataset granular tables. (Documented in the help text below the aggregate table.)
  const avg = col.match(/^(.*) \(avg\)$/);
  if (avg && columnMeta) {
    const group = avg[1];
    const member = Object.entries(columnMeta)
      .filter(([, m]) => m.kind === "entity_leaf" && m.group === group)
      .map(([c]) => c)
      .sort()[0];
    if (member) return { exp_id: expId, dataset_key: member, variant: null };
  }
  return { exp_id: expId, dataset_key: col, variant: null };
}

/* a friendly section title for the dataset-set selector groups (FB-R4c-04). */
const KIND_LABEL: Record<string, string> = {
  canonical_avg: "Multi-entity (avg)",
  swat_excl22: "SWaT (excl22)",
  single_leaf: "Single-leaf datasets",
  swat_full: "SWaT (full)",
  entity_leaf: "Individual machines / channels",
  concat: "Concatenated leaves",
};
/* the order canonical (default) groups are shown first, then the granular add-ons. */
const KIND_ORDER = ["canonical_avg", "swat_excl22", "single_leaf", "swat_full", "entity_leaf", "concat"];

export default function Rankings() {
  const [params] = useSearchParams();
  const [metric, setMetric] = useState(params.get("metric") || "pak_auc_f1");
  // FB-R4c-04: the chosen dataset-set. `null` = "use the backend canonical default"
  // (resolved once the universe loads); an explicit array once the user composes a set.
  const [selDatasets, setSelDatasets] = useState<string[] | null>(null);
  const pins = useStore((s) => s.pins);
  const togglePin = useStore((s) => s.togglePinModel);
  const isModelPinned = useStore((s) => s.isModelPinned);
  const modelIndex = useStore((s) => s.modelIndex);
  // PERF_BASIS_SPEC: the chosen (epoch × threshold) basis re-resolves + re-ranks every
  // aggregate/per-dataset value. Read at the top of the body (NOT inside an AsyncPanel
  // render-prop) and thread into BOTH the universe + the active-set aggregate queries.
  const epochBasis = useStore((s) => s.epochBasis);
  const thresholdBasis = useStore((s) => s.thresholdBasis);
  const catalog = useGlobalMetricsCatalog();

  // FB-3 race over the pinned MODELS — needs a dataset (the shared scope). raceLeaves =
  // pinned models × the chosen dataset; a compact dataset picker sits in the race card.
  const raceScope = useScopeDataset(pins);
  const raceLeaves = useMemo<LeafSel[]>(
    () =>
      raceScope.datasetKey
        ? pins.map((m) => ({ exp_id: m, dataset_key: raceScope.datasetKey as string, variant: raceScope.variant }))
        : [],
    [pins, raceScope.datasetKey, raceScope.variant]
  );
  const raceColor = (lid: string) => catColor(Math.max(0, modelIndex(expOf(lid))));

  // the full column universe (for the FB-2/FB-R4c-04 selector) — discovered from the
  // UNFILTERED aggregate so the selector never hides a column that exists (F5-analog). The
  // metric drives the universe (a metric only present on some leaves still lists its cols).
  const fullAggQ = useRankingsAggregate({ metric, epoch_basis: epochBasis, threshold_basis: thresholdBasis });
  const universe = useMemo(() => fullAggQ.data?.selectable_columns ?? [], [fullAggQ.data]);
  const defaultCols = useMemo(() => fullAggQ.data?.default_columns ?? [], [fullAggQ.data]);
  const columnMeta = useMemo(() => fullAggQ.data?.column_meta ?? {}, [fullAggQ.data]);
  const groupCoverage = useMemo(() => fullAggQ.data?.group_coverage ?? {}, [fullAggQ.data]);

  // the effective active set = the explicit selection, else the canonical default.
  const activeSet = useMemo(
    () => (selDatasets == null ? defaultCols : selDatasets),
    [selDatasets, defaultCols]
  );

  // FB-R3-06 / R3-SR-04: an aggregate row spans many columns, so pin ONE representative
  // leaf — iterate the active columns IN ORDER and take the FIRST where this experiment
  // has a non-null per-column rank, then recover a pinnable leaf via columnToLeaf (an
  // "(avg)" column resolves to its first per-entity member; see columnToLeaf).
  function representativeLeaf(a: { exp_id: string; per_dataset_rank: Record<string, number | null> }, cols: string[]): LeafSel | null {
    for (const col of cols) {
      if (a.per_dataset_rank[col] != null) return columnToLeaf(col, a.exp_id, columnMeta);
    }
    return null;
  }

  // FB-R4c-04: pass the chosen metric + dataset SET to the aggregate. `datasets` is a
  // comma-list of column keys; when it equals the canonical default we still send it
  // explicitly so the active set is unambiguous (the backend re-ranks over exactly it).
  const aggQ = useRankingsAggregate({
    metric,
    datasets: activeSet.length ? activeSet.join(",") : undefined,
    epoch_basis: epochBasis,
    threshold_basis: thresholdBasis,
  });

  // FB-3: the primary pinned animation is the interactive client <LineRace> (per-epoch,
  // overtaking, play/pause/replay/seek). The server `line_race` GIF (story 8) and the
  // `bar_race` GIF are kept as secondary EXPORT artifacts (a re-pick of the export story).
  const [exportStory, setExportStory] = useState<"line_race" | "bar_race">("line_race");
  const [showExport, setShowExport] = useState(false);
  const raceSpec: GifSpec | null =
    raceLeaves.length > 0
      ? {
          story: exportStory,
          metric_keys: [metric],
          experiment_set: raceLeaves,
          variant: null,
          max_epoch: null,
          params: {},
        }
      : null;

  function toggleDataset(col: string) {
    setSelDatasets((cur) => {
      const base = cur == null ? defaultCols : cur;
      return base.includes(col) ? base.filter((c) => c !== col) : [...base, col];
    });
  }

  // group the universe by column KIND for the grouped selector (canonical first).
  const grouped = useMemo(() => {
    const byKind: Record<string, string[]> = {};
    for (const col of universe) {
      const kind = columnMeta[col]?.kind ?? "single_leaf";
      (byKind[kind] ??= []).push(col);
    }
    return KIND_ORDER.filter((k) => byKind[k]?.length).map((k) => [k, byKind[k].slice().sort()] as const);
  }, [universe, columnMeta]);

  const isCanonicalDefault = selDatasets == null;

  return (
    <div className="col" style={{ gap: 24 }}>
      <div>
        <h1 className="view-title">Rankings</h1>
        <p className="view-sub">
          True model ranking: experiments are ranked <strong>within each dataset column</strong> (sort direction from
          the registry), then a model's final rank is the <strong>average of its per-column ranks</strong> (FB-13).
          SMD / SMAP / MSL collapse into ONE <strong>(avg)</strong> column each (mean over their machine/channel leaves,
          counted once); SWaT defaults to <strong>excl22</strong> (full is selectable). Neutral metrics are returned
          un-ranked.
        </p>
      </div>

      <div className="toolbar">
        {/* FB-R4c-04: the ranking METRIC selector (registry-driven runtime catalog, F5,
            direction-aware). Changing it re-ranks the aggregate + per-dataset tables. */}
        <MetricPicker catalog={catalog.data ?? []} value={metric} onChange={setMetric} loading={catalog.isLoading} />
        {/* one-click shortcuts for the most-used metrics (full list stays in the picker). */}
        <MetricQuickPicks value={metric} onChange={setMetric} />
        <span className="grow" style={{ flex: 1 }} />
        <PerformanceBasis />
      </div>

      {/* FB-2 / FB-R4c-03 / FB-R4c-04 — fully-configurable dataset-set selection. The
          canonical units (incl. SMD/SMAP/MSL (avg)) are the DEFAULT; the user can add any
          granular column (individual machines/channels, concat, SWaT full). */}
      {universe.length > 0 && (
        <Card title="Dataset set (FB-R4c-04) — choose which columns the aggregate ranks over">
          <div className="col" style={{ gap: 10 }}>
            <div className="row wrap" style={{ gap: 6, alignItems: "center" }}>
              <button
                className={`btn sm ${isCanonicalDefault ? "primary" : ""}`}
                onClick={() => setSelDatasets(null)}
                title="reset to the canonical set: PSM, SWaT excl22, WaDi/A1, WaDi/A2, SMD (avg), SMAP (avg), MSL (avg)"
              >
                canonical default
              </button>
              <button
                className="btn sm"
                onClick={() => setSelDatasets(universe.slice())}
                title="select every selectable column (canonical + all granular)"
              >
                select all
              </button>
              <span className="subtle" style={{ fontSize: "var(--fs-small)" }}>
                {activeSet.length} / {universe.length} columns selected
                {isCanonicalDefault ? " (canonical default)" : ""}
              </span>
            </div>
            {grouped.map(([kind, cols]) => (
              <div key={kind} className="col" style={{ gap: 4 }}>
                <div className="subtle" style={{ fontSize: "var(--fs-small)", fontWeight: 600 }}>
                  {KIND_LABEL[kind] ?? kind}
                  {(kind === "canonical_avg" || kind === "swat_excl22" || kind === "single_leaf") && (
                    <span style={{ fontWeight: 400 }}> · in the canonical default</span>
                  )}
                </div>
                <div className="row wrap" style={{ gap: 6 }}>
                  {cols.map((col) => {
                    const on = activeSet.includes(col);
                    const cov = groupCoverage[col];
                    const covN = cov ? Object.keys(cov).length : 0;
                    return (
                      <button
                        key={col}
                        className={`btn sm ${on ? "primary" : ""}`}
                        onClick={() => toggleDataset(col)}
                        title={
                          kind === "canonical_avg"
                            ? `${col} = mean over each experiment's per-entity leaves (concat excluded), counted once${
                                covN ? ` · averaged for ${covN} experiment(s)` : ""
                              }`
                            : col
                        }
                      >
                        {col.replace("SWaT_A1A2 · ", "SWaT·")}
                      </button>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* FB-13 — the headline cross-dataset AVERAGE-RANK aggregate (with coverage). */}
      <Card title="Average-rank aggregate (FB-13) — final model ranking across the chosen dataset set">
        <AsyncPanel query={aggQ} isEmpty={(d) => !d.aggregate || d.aggregate.length === 0} height={260}>
          {(d) => (
            <div className="col" style={{ gap: 8 }}>
              <div className="subtle" style={{ fontSize: "var(--fs-small)" }}>
                metric <strong>{d.metric}</strong> ({d.direction === "down" ? "lower wins" : d.direction === "up" ? "higher wins" : "neutral"})
                {d.rankable ? "" : " — neutral metric, not ranked"} · coverage = present / {d.coverage_total} dataset
                columns · {d.missing_dataset_rule} · per-dataset cells show <strong>rank ({d.metric} value)</strong>; an{" "}
                <strong>(avg)</strong> column's value is the mean over its entities
              </div>
              <div className="tbl-wrap" style={{ maxHeight: 480 }}>
                <table className="tbl tabular" style={{ fontSize: "0.82rem" }}>
                  <thead>
                    <tr>
                      <th>rank</th>
                      <th>experiment</th>
                      <th>pin</th>
                      <th className="num">avg rank</th>
                      <th className="num">coverage</th>
                      <th className="num">mean {d.metric}</th>
                      {d.datasets.map((col) => (
                        <th key={col} className="num" title={col}>
                          {col.replace("SWaT_A1A2 · ", "SWaT·")}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {d.aggregate.map((a) => (
                      <tr key={a.exp_id}>
                        <td>{a.aggregate_rank ?? "—"}</td>
                        <td title={a.description ?? undefined}>{a.exp_id}</td>
                        <td>
                          {(() => {
                            const pinned = isModelPinned(a.exp_id);
                            return (
                              <button
                                className={`btn sm ${pinned ? "primary" : ""}`}
                                onClick={() => togglePin(a.exp_id)}
                                title={pinned ? "unpin this model" : "pin this model (pick a dataset to compare on in Compare/GIF/Config)"}
                              >
                                {pinned ? "pinned ✓" : "+ pin"}
                              </button>
                            );
                          })()}
                        </td>
                        <td className="num">
                          <strong>{a.avg_rank != null ? a.avg_rank.toFixed(2) : "—"}</strong>
                        </td>
                        <td className="num">
                          <span
                            className={`chip ${a.coverage_present < a.coverage_total ? "warn" : ""}`}
                            title="datasets where this experiment is present"
                          >
                            {a.coverage_present}/{a.coverage_total}
                          </span>
                        </td>
                        <td className="num">{fmt(a.mean_value)}</td>
                        {d.datasets.map((col) => {
                          const r = a.per_dataset_rank[col];
                          const v = a.per_dataset_value?.[col];
                          // PERF_STORE_SPEC (3): the SELECTED epoch alongside the value — the
                          // backend may not serve per_dataset_epoch yet, so read defensively.
                          const ep = a.per_dataset_epoch?.[col];
                          // P3-01/coverage: a missing per-dataset rank is a real GAP — show
                          // a dash, never a fabricated rank. The metric VALUE (+ epoch when
                          // known) is shown next to the rank ("rank (value, Nep)"); for an
                          // "(avg)" column the value is the mean and the epoch the rounded mean.
                          return (
                            <td key={col} className="num" style={{ opacity: r == null ? 0.4 : 1 }}>
                              {r == null ? (
                                "—"
                              ) : (
                                <>
                                  {r}
                                  {typeof v === "number" && (
                                    <span className="subtle" style={{ fontSize: "0.82em" }}>
                                      {" "}
                                      ({v.toFixed(4)}
                                      {typeof ep === "number" ? `, ${ep}ep` : ""})
                                    </span>
                                  )}
                                </>
                              )}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* F-1 (P0-1): rows BELOW the coverage threshold are kept OUT of the
                  headline ranking and surfaced under a COLLAPSED "insufficient coverage"
                  section — so a 1/14-coverage experiment can't ride near the top. The
                  threshold (auto-derived or explicit) and n/N come from the response. */}
              {(d.low_coverage?.length ?? 0) > 0 && (
                <details className="low-coverage">
                  <summary className="subtle" style={{ cursor: "pointer", fontSize: "var(--fs-small)" }}>
                    insufficient coverage ({d.low_coverage!.length}/{d.aggregate.length + d.low_coverage!.length}) — scored
                    on fewer than {d.min_coverage} of {d.coverage_total} dataset columns
                    {d.min_coverage_auto ? " (auto threshold)" : " (explicit threshold)"} · excluded from the headline ranking
                  </summary>
                  <div className="tbl-wrap" style={{ maxHeight: 320, marginTop: 8 }}>
                    <table className="tbl tabular" style={{ fontSize: "0.82rem" }}>
                      <thead>
                        <tr>
                          <th>#</th>
                          <th>experiment</th>
                          <th className="num">avg rank</th>
                          <th className="num">coverage</th>
                          <th className="num">mean {d.metric}</th>
                        </tr>
                      </thead>
                      <tbody>
                        {d.low_coverage!.map((a, i) => (
                          <tr key={a.exp_id} style={{ opacity: 0.75 }}>
                            <td>{a.low_coverage_rank ?? i + 1}</td>
                            <td title={a.description ?? undefined}>{a.exp_id}</td>
                            <td className="num">{a.avg_rank != null ? a.avg_rank.toFixed(2) : "—"}</td>
                            <td className="num">
                              <span className="chip warn" title="datasets where this experiment is present">
                                {a.coverage_present}/{a.coverage_total}
                              </span>
                            </td>
                            <td className="num">{fmt(a.mean_value)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <div className="subtle" style={{ fontSize: "var(--fs-small)", marginTop: 4 }}>
                    These are NOT comparable to the headline models — too few shared datasets to rank fairly.
                  </div>
                </details>
              )}
            </div>
          )}
        </AsyncPanel>
      </Card>

      {/* FB-12 — per-dataset ranking tables (one per active column; SWaT split, SMD/SMAP/
          MSL as collapsed (avg) tables). */}
      <Card title="Per-dataset rankings (FB-12) — ranked within each column, no cross-dataset mixing">
        <AsyncPanel query={aggQ} isEmpty={(d) => !d.datasets || d.datasets.length === 0} height={260}>
          {(d) => (
            <div className="card-grid">
              {d.datasets.map((col) => (
                <PerDatasetRankTable
                  key={col}
                  label={d.dataset_labels?.[col] ?? col}
                  rows={d.per_dataset[col] ?? []}
                  metricLabel={d.metric}
                  rankable={d.rankable}
                  showPin
                />
              ))}
            </div>
          )}
        </AsyncPanel>
      </Card>

      {/* FB-3 — interactive LINE race over the pinned MODELS on one dataset (the primary
          animation); bar_race / line_race GIFs are the secondary export artifacts. */}
      <Card title="Line race (FB-3) — pinned models on one dataset, per-epoch, overtaking · play / pause / replay / seek">
        {pins.length === 0 ? (
          <div className="subtle">Pin models (the + pin buttons above) to race them across epochs.</div>
        ) : raceScope.noCommon ? (
          <div className="subtle">The pinned models share no dataset to race on — pin models that overlap.</div>
        ) : raceLeaves.length > 0 ? (
          <div className="col" style={{ gap: 12 }}>
            <div className="row wrap" style={{ gap: 8, alignItems: "center" }}>
              <label className="row" style={{ gap: 6, fontSize: "var(--fs-small)" }}>
                dataset
                <select
                  className="select sm"
                  value={raceScope.datasetKey ?? ""}
                  onChange={(e) => raceScope.setDatasetKey(e.target.value)}
                  title="race the pinned models on this dataset"
                >
                  {raceScope.options.map((o) => (
                    <option key={o.dataset_key} value={o.dataset_key}>
                      {o.dataset_key}
                    </option>
                  ))}
                </select>
              </label>
              {raceScope.current && raceScope.current.variants.filter((v) => v && v !== "_").length > 1 && (
                <span className="seg" role="group" aria-label="variant">
                  {raceScope.current.variants
                    .filter((v) => v && v !== "_")
                    .map((v) => (
                      <button
                        key={v}
                        className={`seg-btn ${raceScope.variant === v ? "active" : ""}`}
                        onClick={() => raceScope.setVariant(v)}
                      >
                        {v}
                      </button>
                    ))}
                </span>
              )}
              <span className="subtle" style={{ fontSize: "var(--fs-small)" }}>
                {pins.length} model{pins.length === 1 ? "" : "s"} · {raceScope.options.length} dataset
                {raceScope.options.length === 1 ? "" : "s"} in common
              </span>
            </div>
            <LineRace leaves={raceLeaves} metricKey={metric} colorOf={raceColor} />
            <div className="toolbar">
              <label className="row" style={{ gap: 6 }}>
                <input type="checkbox" checked={showExport} onChange={(e) => setShowExport(e.target.checked)} /> show
                export race GIF
              </label>
              {showExport && (
                <div className="pill-toggle" title="export race story">
                  <button className={exportStory === "line_race" ? "on" : ""} onClick={() => setExportStory("line_race")}>
                    line_race (per-epoch)
                  </button>
                  <button className={exportStory === "bar_race" ? "on" : ""} onClick={() => setExportStory("bar_race")}>
                    bar_race (best-so-far)
                  </button>
                </div>
              )}
            </div>
            {showExport && raceSpec && (
              <div>
                <div className="subtle" style={{ fontSize: "var(--fs-small)", marginBottom: 6 }}>
                  Server-rendered <span className="mono">{exportStory}</span> GIF — non-looping, higher-DPI,
                  shareable export.
                </div>
                <GifViewer spec={raceSpec} height={380} autoRender={false} />
              </div>
            )}
          </div>
        ) : (
          <div className="subtle">Pin runs from a per-dataset table to race them across epochs.</div>
        )}
      </Card>
    </div>
  );
}
