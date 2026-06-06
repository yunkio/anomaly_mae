/* <TrainingHistoryPanel> (P2-01) — its OWN training-epoch axis, NEVER co-plotted
 * with the eval-epoch trends (AC-F4.4). Reads the UNION over training_histories['0']
 * via /training-history. Two modes:
 *   • flat series (train_*, the 18 epoch_{recon,disc}_{ratio,score,raw}_* families)
 *     over train_epochs (len 500), aligned by epoch NUMBER.
 *   • the 2-level nested pivot[type][sub_metric] for epoch_anomaly_type_scores
 *     (P3-02): default sub_metric=recon_ratio (↑), direction from sub_metric_meta,
 *     one trace per type. */
import { useMemo, useState } from "react";
import { useLossDriftMetrics, useMetricsCatalog, useTrainingHistory } from "../../api/queries";
import MetricFamilySelector from "../MetricFamilySelector";
import MetricChart, { MetricSeriesInput } from "../charts/MetricChart";
import AsyncPanel from "../AsyncPanel";
import ChartFrame from "../charts/ChartFrame";
import { baseLayout, PLOTLY_CONFIG, warmupShape, warmupLine } from "../../viz/plotlyTemplate";
import { catColor } from "../../store";

const NESTED_KEY = "epoch_anomaly_type_scores";

/* FB-14 — illustrative "quick pick" groups so the SIMPLEST examples the user called out
 * (teacher reconstruction, discrepancy, contribution share) are reachable in ≤1 click,
 * without rebuilding the existing UNION selector. Each group lists candidate keys; we
 * intersect with the leaf's populated `loss-drift-metrics` inventory so a disabled key is
 * never offered (F5 — the inventory is a runtime union, not a hard-coded enumeration). */
const QUICK_PICKS: { label: string; title?: string; keys: string[] }[] = [
  { label: "teacher reconstruction", keys: ["train_teacher_recon_normal", "train_teacher_recon_anomaly"] },
  { label: "discrepancy", keys: ["train_disc_loss", "train_mean_discrepancy"] },
  { label: "losses", keys: ["train_loss", "train_rec_loss", "train_normal_loss", "train_anomaly_loss"] },
  // FB-R4-03: `epoch_{recon,disc}_ratio_{type}` is the per-type CONTRIBUTION SHARE
  // `component/(recon+disc)` of the score (context/neutral, run_base_experiments.py:843-855)
  // — NOT an anomaly/normal "separation ratio". The chip label reads as a contribution
  // share; direction stays neutral (the backend resolves diagnostic.context, no ↑/↓).
  {
    label: "contribution share",
    title: "per-sample-type recon/disc CONTRIBUTION SHARE of the score (component/(recon+disc)); context, not an a/n ratio",
    keys: ["epoch_recon_ratio_anomaly", "epoch_disc_ratio_anomaly"],
  },
  { label: "student reconstruction", keys: ["train_student_recon_normal", "train_student_recon_anomaly"] },
];

interface Props {
  exp: string;
  dsKey: string;
  variant: string | null;
  warmup: number | null;
}

export default function TrainingHistoryPanel({ exp, dsKey, variant, warmup }: Props) {
  const cat = useMetricsCatalog(exp, dsKey, variant);
  // FB-14: the leaf's full grouped training-metric inventory (populated vs disabled).
  const inv = useLossDriftMetrics(exp, dsKey, variant);
  const [mode, setMode] = useState<"flat" | "anom_type">("flat");
  // FB-14: a more ILLUSTRATIVE default than a single `train_loss` — teacher reconstruction
  // (always populated) + discrepancy, so the panel opens on the components a researcher
  // actually wants. Falls back gracefully if a key is absent on this leaf.
  const [flatKeys, setFlatKeys] = useState<string[]>(["train_teacher_recon_normal", "train_disc_loss"]);
  const [subMetric, setSubMetric] = useState("recon_ratio");

  const historyCatalog = useMemo(
    () => (cat.data ?? []).filter((c) => c.namespace === "history" && !c.key.startsWith(NESTED_KEY)),
    [cat.data]
  );

  // populated key set (from the inventory) so quick-picks only offer reachable series.
  const populatedKeys = useMemo(() => {
    const s = new Set<string>();
    Object.values(inv.data?.families ?? {}).forEach((items) =>
      items.forEach((it) => it.populated && s.add(it.key))
    );
    return s;
  }, [inv.data]);

  const flatQ = useTrainingHistory(exp, dsKey, mode === "flat" ? flatKeys : [], variant);
  const pivotQ = useTrainingHistory(exp, dsKey, mode === "anom_type" ? [NESTED_KEY] : [], variant);

  return (
    <div className="col" style={{ gap: 12 }}>
      <div className="toolbar">
        <div className="pill-toggle">
          <button className={mode === "flat" ? "on" : ""} onClick={() => setMode("flat")}>
            flat training curves
          </button>
          <button className={mode === "anom_type" ? "on" : ""} onClick={() => setMode("anom_type")}>
            per-anomaly-type (2-level pivot)
          </button>
        </div>
        <span className="subtle" style={{ fontSize: "var(--fs-small)" }}>
          separate <strong>training-epoch</strong> axis (len ≈ num_epochs) — aligned by epoch number, never index
        </span>
      </div>

      {mode === "flat" && (
        <div className="row wrap" style={{ gap: 12, alignItems: "flex-start" }}>
          <div style={{ flex: "1 1 320px", minWidth: 260 }}>
            {/* FB-14 quick picks — the simplest insightful examples in ≤1 click. Only the
                populated keys (from the leaf inventory) render; a disabled component never
                appears here (the full UNION selector below still lists everything, F5). */}
            <div className="row wrap" style={{ gap: 6, marginBottom: 8 }}>
              {QUICK_PICKS.map((g) => {
                const keys = g.keys.filter((k) => populatedKeys.size === 0 || populatedKeys.has(k));
                if (keys.length === 0) return null;
                return (
                  <button
                    key={g.label}
                    className="btn sm"
                    title={g.title ? `${g.title} — ${keys.join(", ")}` : keys.join(", ")}
                    onClick={() => setFlatKeys(keys)}
                  >
                    {g.label}
                  </button>
                );
              })}
            </div>
            <AsyncPanel query={cat} title="History metrics (UNION · F5)" height={140}>
              {() => (
                <MetricFamilySelector catalog={historyCatalog} selected={flatKeys} onChange={setFlatKeys} multi />
              )}
            </AsyncPanel>
            {(inv.data?.disabled?.length ?? 0) > 0 && (
              <div className="subtle" style={{ fontSize: "var(--fs-small)", marginTop: 6 }}>
                {inv.data!.disabled.length} component(s) off for this run (e.g.{" "}
                {inv.data!.disabled.slice(0, 3).map((d) => d.key).join(", ")}…) — hidden until enabled.
              </div>
            )}
          </div>
          <div style={{ flex: "2 1 520px", minWidth: 320 }}>
            <AsyncPanel
              query={flatQ}
              isEmpty={(d) => !d.train_epochs || d.train_epochs.length === 0}
              height={300}
            >
              {(d) => {
                const inputs: MetricSeriesInput[] = flatKeys
                  .filter((k) => d.series[k])
                  .map((k) => ({ key: k, x: d.train_epochs, y: d.series[k], meta: d.meta[k] }));
                if (inputs.length === 0)
                  return <div className="subtle">Selected keys are empty/disabled for this run.</div>;
                return (
                  <MetricChart series={inputs} axis="training-epoch" warmupEpochs={d.warmup_epochs ?? warmup} />
                );
              }}
            </AsyncPanel>
          </div>
        </div>
      )}

      {mode === "anom_type" && (
        <AsyncPanel
          query={pivotQ}
          isEmpty={(d) => !d.nested?.[NESTED_KEY]}
          emptyLabel="No epoch_anomaly_type_scores for this run."
          height={320}
        >
          {(d) => {
            const piv = d.nested[NESTED_KEY];
            const smMeta = piv.sub_metric_meta?.[subMetric];
            const isUp = smMeta?.direction === "up";
            // FB-7: render the backend-supplied per-type display_name (spike→"Anomaly"
            // when sole bucket; normal→"Normal"); falls back to the raw key (F5).
            const traces: any[] = piv.types.map((t, i) => ({
              x: d.train_epochs,
              y: piv.pivot[t]?.[subMetric] ?? [],
              name: piv.type_display_names?.[t] || t,
              type: "scatter",
              mode: "lines",
              line: { color: catColor(i), width: 2 },
              connectgaps: false,
            }));
            const shapes = [...warmupShape(d.warmup_epochs ?? warmup), ...warmupLine(d.warmup_epochs ?? warmup)];
            const layout = baseLayout({
              height: 320,
              shapes,
              xaxis: { ...baseLayout().xaxis, title: { text: "training epoch" } },
              yaxis: { ...baseLayout().yaxis, title: { text: `${subMetric} (${isUp ? "↑" : smMeta?.direction ?? ""})` } },
            });
            return (
              <div className="col" style={{ gap: 8 }}>
                <div className="toolbar">
                  <label className="row" style={{ gap: 6 }}>
                    sub-metric
                    <select className="select" value={subMetric} onChange={(e) => setSubMetric(e.target.value)}>
                      {piv.sub_metrics.map((m) => (
                        <option key={m} value={m}>
                          {m}
                          {piv.sub_metric_meta?.[m]?.direction === "up" ? " ↑" : ""}
                        </option>
                      ))}
                    </select>
                  </label>
                  <span className="subtle" style={{ fontSize: "var(--fs-small)" }}>
                    direction from the SUB-METRIC, not the type (P3-02) · {piv.types.length} types
                  </span>
                </div>
                <ChartFrame name={`training_history_pivot_${subMetric}`} data={traces} layout={layout} config={PLOTLY_CONFIG} style={{ width: "100%", height: 320 }} />
              </div>
            );
          }}
        </AsyncPanel>
      )}
    </div>
  );
}
