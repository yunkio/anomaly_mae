/* Comparison Workbench (IA-7, 2026-06-04 pin refactor) — pick MODELS, pick a
 * DATASET, compare many metrics at once. Models come from the pinned set (editable
 * inline via <CompareControls>); the dataset is the shared scope. Everything below
 * is fed by `sel.leaves` (= selected models × the scope dataset).
 * • <ComparisonMatrix>: rows=model, cols=metrics; cells DIRECTION-COLORED (good-hue
 *   scaled within a column by direction; neutral=neutral; missing="—" never 0).
 * • <OverlayTrends>: N series of one metric on synced axes, aligned by EPOCH NUMBER
 *   (gaps are real gaps), stable per-model categorical color. */
import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import ChartFrame from "../components/charts/ChartFrame";
import { useCompareMatrix, useCompareSeries, useGlobalMetricsCatalog } from "../api/queries";
import { Card, fmt } from "../components/common";
import AsyncPanel from "../components/AsyncPanel";
import MetricPicker from "../components/MetricPicker";
import LineRace from "../components/LineRace";
import GifViewer from "../components/gif/GifViewer";
import CompareControls from "../components/CompareControls";
import { baseLayout, PLOTLY_CONFIG } from "../viz/plotlyTemplate";
import { useStore, catColor } from "../store";
import { shortModel, useCompareSelection } from "../scope";
import type { CompareMatrix, GifSpec, LeafSel } from "../api/types";

const tok = (n: string) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
// the INITIAL selected set only (never the option list — that is the runtime catalog).
const DEFAULT_METRICS = ["pak_auc_f1", "prc_auc", "roc_auc", "f1_score"];

/** leaf_id is "exp|dataset|variant"; the model is the first segment. */
const expOf = (leafId: string): string => leafId.split("|")[0];

export default function CompareWorkbench() {
  const nav = useNavigate();
  const sel = useCompareSelection();
  const modelIndex = useStore((s) => s.modelIndex);
  const [metrics, setMetrics] = useState<string[]>(DEFAULT_METRICS);
  const [addKey, setAddKey] = useState("pak_auc_f1");
  const [overlayKey, setOverlayKey] = useState("pak_auc_f1");
  const [sharedScale, setSharedScale] = useState(true);
  const catalog = useGlobalMetricsCatalog();

  const leaves = sel.leaves;
  const ready = leaves.length >= 1 && !!sel.scope.datasetKey;
  const matrixQ = useCompareMatrix(ready ? leaves : [], metrics);
  const seriesQ = useCompareSeries(ready ? leaves : [], overlayKey);

  // F-02: hooks computed from query.data at the TOP LEVEL with a null-guard — never
  // inside the <AsyncPanel> render-prop (which only runs on the success render and so
  // changes the hook count on a loading/error/empty transition → Rules-of-Hooks crash).
  const colStats = useMemo(() => columnStats(matrixQ.data), [matrixQ.data]);
  const colorOf = (leafId: string) => catColor(Math.max(0, modelIndex(expOf(leafId))));

  return (
    <div className="col" style={{ gap: 24 }}>
      <div>
        <h1 className="view-title">Comparison Workbench</h1>
        <p className="view-sub">
          Pick the models to compare, then one dataset. Cells are direction-colored; missing values render "—", never 0;
          trends align by epoch number.
        </p>
        <CompareControls sel={sel} />
        {sel.scope.datasetKey && (
          <div className="row wrap" style={{ gap: 8 }}>
            <button className="btn sm" onClick={() => nav("/config")} title="diff the configs of these models on this dataset">
              ⚙ diff configs of these →
            </button>
            <span className="subtle" style={{ fontSize: "var(--fs-small)", alignSelf: "center" }}>
              comparing on <span className="mono">{sel.scope.datasetKey}</span>
              {sel.scope.variant && sel.scope.variant !== "_" ? ` · ${sel.scope.variant}` : ""}
            </span>
          </div>
        )}
      </div>

      {!ready ? (
        <div className="async-state" style={{ minHeight: 160 }}>
          <span className="glyph">⇄</span>
          <div>
            {sel.candidates.length === 0
              ? "Pin one or more models (Overview / Rankings), or add a model above — then choose a dataset."
              : "Choose a dataset above to compare the selected models."}
          </div>
        </div>
      ) : (
        <>
          <Card title="Comparison matrix (CMP-1)">
            <div className="toolbar">
              <MetricPicker
                catalog={catalog.data ?? []}
                value={addKey}
                onChange={setAddKey}
                label="add metric"
                loading={catalog.isLoading}
              />
              <button className="btn sm" onClick={() => addKey && setMetrics(Array.from(new Set([...metrics, addKey])))}>
                + add
              </button>
              <div className="row wrap" style={{ gap: 6 }}>
                {metrics.map((m) => (
                  <span className="chip" key={m}>
                    {m}
                    <button
                      style={{ border: 0, background: "transparent", cursor: "pointer", marginLeft: 4 }}
                      onClick={() => setMetrics(metrics.filter((k) => k !== m))}
                    >
                      ×
                    </button>
                  </span>
                ))}
              </div>
            </div>
            <AsyncPanel query={matrixQ} isEmpty={(d) => !d.rows?.length} height={200}>
              {(d) => (
                <div className="tbl-wrap">
                  <table className="tbl tabular">
                    <thead>
                      <tr>
                        <th>model</th>
                        {d.metrics.map((m) => (
                          <th key={m} className="num" title={d.meta[m]?.description}>
                            {d.meta[m]?.display_name ?? m}
                            {d.meta[m]?.direction === "up" ? " ↑" : d.meta[m]?.direction === "down" ? " ↓" : " ◦"}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {d.rows.map((row) => (
                        <tr key={row.leaf_id}>
                          <td>
                            <span className="row" style={{ gap: 6 }}>
                              <span style={{ width: 10, height: 10, borderRadius: 3, background: colorOf(row.leaf_id) }} />
                              <span className="mono" style={{ fontSize: "0.74rem" }} title={row.leaf_id}>
                                {shortModel(expOf(row.leaf_id))}
                              </span>
                            </span>
                          </td>
                          {d.metrics.map((m) => {
                            const cell = row.cells[m];
                            const v = cell?.value;
                            const bg = cellBg(v, colStats[m], cell?.direction ?? d.meta[m]?.direction);
                            return (
                              <td key={m} className="num" style={{ background: bg }}>
                                {fmt(v)}
                              </td>
                            );
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </AsyncPanel>
          </Card>

          <Card title="Overlay trends (TREND-4) — synced by epoch number">
            <div className="toolbar">
              <MetricPicker catalog={catalog.data ?? []} value={overlayKey} onChange={setOverlayKey} loading={catalog.isLoading} />
              <label className="row" style={{ gap: 6 }}>
                <input type="checkbox" checked={sharedScale} onChange={(e) => setSharedScale(e.target.checked)} /> shared [0,1]
                scale
              </label>
            </div>
            <AsyncPanel query={seriesQ} isEmpty={(d) => !d.series?.length} height={320}>
              {(d) => {
                const traces: any[] = [];
                for (const s of d.series) {
                  const color = colorOf(s.leaf_id);
                  traces.push({
                    x: s.epochs,
                    y: s.values,
                    name: shortModel(expOf(s.leaf_id)),
                    type: "scattergl",
                    mode: "lines+markers",
                    connectgaps: false,
                    line: { color, width: 2 },
                    marker: { size: 3 },
                  });
                  // FB-4: mark each line's BEST-EPOCH point using the backend's direction-aware
                  // best_epoch / best_value (argmax for ↑, argmin for ↓; None/gaps skipped, never 0).
                  if (s.best_epoch != null && s.best_value != null) {
                    traces.push({
                      x: [s.best_epoch],
                      y: [s.best_value],
                      name: `${shortModel(expOf(s.leaf_id))} · best`,
                      type: "scatter",
                      mode: "markers",
                      marker: { color, size: 13, symbol: "star", line: { width: 1, color: "var(--surface)" } },
                      showlegend: false,
                      hovertemplate: `${shortModel(expOf(s.leaf_id))} best<br>epoch %{x}: %{y:.4f}<extra></extra>`,
                    });
                  }
                }
                const meta = d.meta?.[overlayKey];
                const score = (meta?.unit || "").includes("[0,1]");
                const layout = baseLayout({
                  height: 340,
                  yaxis: { ...baseLayout().yaxis, range: sharedScale && score ? [0, 1] : undefined, title: { text: overlayKey } },
                  xaxis: { ...baseLayout().xaxis, title: { text: "eval epoch (aligned by number)" } },
                });
                return <ChartFrame name={`compare_overlay_${overlayKey}`} data={traces} layout={layout} config={PLOTLY_CONFIG} style={{ width: "100%", height: 340 }} />;
              }}
            </AsyncPanel>
            <div className="subtle" style={{ fontSize: "var(--fs-small)", marginTop: 6 }}>
              ★ marks each line's best epoch (direction-aware from the registry — argmax for ↑, argmin for ↓); a
              gap/scalar-only metric has no marker (FB-4 / P3-01).
            </div>
          </Card>

          {/* FB-3 — the headline LINE RACE for the selected models on the scope dataset. */}
          <RaceCard leaves={leaves} colorOf={colorOf} />
        </>
      )}
    </div>
  );
}

/* FB-3 race card: the interactive client <LineRace> (primary) + the server `line_race`
 * GIF (story 8) as the shareable EXPORT artifact. */
function RaceCard({ leaves, colorOf }: { leaves: LeafSel[]; colorOf: (leafId: string) => string }) {
  const [raceKey, setRaceKey] = useState("pak_auc_f1");
  const [showExport, setShowExport] = useState(false);
  const catalog = useGlobalMetricsCatalog();
  const exportSpec: GifSpec | null =
    leaves.length > 0
      ? { story: "line_race", metric_keys: [raceKey], experiment_set: leaves, variant: null, max_epoch: null, params: {} }
      : null;
  return (
    <Card title="Line race (FB-3) — per-epoch, lines overtake; play / pause / replay / seek">
      <div className="toolbar">
        <MetricPicker catalog={catalog.data ?? []} value={raceKey} onChange={setRaceKey} label="race metric" loading={catalog.isLoading} />
        <label className="row" style={{ gap: 6 }}>
          <input type="checkbox" checked={showExport} onChange={(e) => setShowExport(e.target.checked)} /> show export GIF
          (server line_race)
        </label>
      </div>
      <LineRace leaves={leaves} metricKey={raceKey} colorOf={colorOf} />
      {showExport && (
        <div style={{ marginTop: 12 }}>
          <div className="subtle" style={{ fontSize: "var(--fs-small)", marginBottom: 6 }}>
            Server-rendered <span className="mono">line_race</span> GIF (story 8) — a non-looping, higher-DPI shareable
            export of the same race.
          </div>
          {exportSpec && <GifViewer spec={exportSpec} height={360} autoRender={false} />}
        </div>
      )}
    </Card>
  );
}

/** Per-column min/max for the direction-aware cell shading. Null-guarded so it can be
 * computed at the top level (hoisted out of the AsyncPanel render-prop, F-02). */
function columnStats(d: CompareMatrix | undefined): Record<string, { min: number; max: number }> {
  const out: Record<string, { min: number; max: number }> = {};
  if (!d?.metrics || !d?.rows) return out;
  for (const m of d.metrics) {
    let mn = Infinity,
      mx = -Infinity;
    for (const row of d.rows) {
      const v = row.cells[m]?.value;
      if (typeof v === "number") {
        mn = Math.min(mn, v);
        mx = Math.max(mx, v);
      }
    }
    out[m] = { min: mn, max: mx };
  }
  return out;
}

/** direction-aware cell background: good-hue scaled within the column. neutral = grey. */
function cellBg(
  v: number | null | undefined,
  stat: { min: number; max: number } | undefined,
  direction?: string
): string {
  if (typeof v !== "number" || !stat || !isFinite(stat.min) || stat.max === stat.min) return "transparent";
  if (direction === "neutral" || !direction) return "transparent";
  let t = (v - stat.min) / (stat.max - stat.min); // 0..1, higher=larger
  if (direction === "down") t = 1 - t; // lower is better -> invert
  const alpha = 0.08 + 0.42 * t;
  const good = tok("--sem-good") || "#128a6e";
  return hexA(good, alpha);
}

function hexA(hex: string, a: number): string {
  const h = hex.replace("#", "");
  if (h.length < 6) return `rgba(18,138,110,${a})`;
  const r = parseInt(h.slice(0, 2), 16);
  const g = parseInt(h.slice(2, 4), 16);
  const b = parseInt(h.slice(4, 6), 16);
  return `rgba(${r},${g},${b},${a})`;
}
