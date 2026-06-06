/* <AnomalyTypePanel> (DS-1) — grouped bars of detection_rate/f1_score (↑) per
 * dynamically-discovered anomaly type (never hard-codes spike/attack). count /
 * mean_score / std_score shown as neutral context. */
import { useMemo, useState } from "react";
import ChartFrame from "../charts/ChartFrame";
import { useAnomalyTypes } from "../../api/queries";
import AsyncPanel from "../AsyncPanel";
import { baseLayout, PLOTLY_CONFIG } from "../../viz/plotlyTemplate";

const tok = (n: string) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

export default function AnomalyTypePanel({ exp, dsKey, variant }: { exp: string; dsKey: string; variant: string | null }) {
  const q = useAnomalyTypes(exp, dsKey, variant);
  const [submetric, setSubmetric] = useState("f1_score");

  // F-02: derived from query.data at the TOP LEVEL (null-guarded), never as a hook
  // inside the AsyncPanel render-prop (whose hook count changes across query states).
  const subMetrics = useMemo(() => {
    const s = new Set<string>();
    (q.data?.types ?? []).forEach((t) => Object.keys(t.metrics).forEach((k) => s.add(k)));
    return Array.from(s).sort();
  }, [q.data]);

  return (
    <AsyncPanel
      query={q}
      isEmpty={(d) => !d.types || d.types.length === 0}
      emptyLabel="No anomaly_type_metrics.json for this leaf."
      height={320}
    >
      {(d) => {
        const meta = d.types[0]?.meta?.[submetric];
        const isUp = meta?.direction === "up";
        const color = isUp ? tok("--sem-good") : tok("--sem-neutral");
        const trace: any = {
          x: d.types.map((t) => t.display_name || t.name),
          y: d.types.map((t) => t.metrics[submetric] ?? null),
          type: "bar",
          marker: { color },
          hovertemplate: "%{x}: %{y:.4f}<extra></extra>",
        };
        const layout = baseLayout({
          height: 320,
          showlegend: false,
          yaxis: { ...baseLayout().yaxis, title: { text: submetric }, range: isUp ? [0, 1] : undefined },
          xaxis: { ...baseLayout().xaxis, title: { text: "anomaly type (dynamic vocabulary)" } },
        });
        return (
          <div className="col" style={{ gap: 8 }}>
            <div className="toolbar">
              <label className="row" style={{ gap: 6 }}>
                sub-metric
                <select className="select" value={submetric} onChange={(e) => setSubmetric(e.target.value)}>
                  {subMetrics.map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
              </label>
              {meta && (
                <span className="subtle" style={{ fontSize: "var(--fs-small)" }}>
                  {meta.direction === "up" ? "higher better ↑" : meta.direction === "down" ? "lower better ↓" : "context ◦"}
                </span>
              )}
            </div>
            <ChartFrame name={`anomaly_types_${submetric}`} data={[trace]} layout={layout} config={PLOTLY_CONFIG} style={{ width: "100%", height: 320 }} />
          </div>
        );
      }}
    </AsyncPanel>
  );
}
