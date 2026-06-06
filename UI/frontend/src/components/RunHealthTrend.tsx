/* <RunHealthTrend> (HLTH-1, P2-02) — the always-present per-eval health/perf trend
 * from monitoring_log.jsonl (GET /monitoring?as=series): pak_f1 climb, nan/inf
 * accrual, GPU/CPU/RAM over runtime, Korean/UTF-8 verdicts as hover annotations. */
import ChartFrame from "./charts/ChartFrame";
import { useMonitoring } from "../api/queries";
import AsyncPanel from "./AsyncPanel";
import { baseLayout, PLOTLY_CONFIG } from "../viz/plotlyTemplate";
import type { MonitoringSeries } from "../api/types";

const tok = (n: string) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

export default function RunHealthTrend({ expId }: { expId: string }) {
  const q = useMonitoring(expId);
  return (
    <AsyncPanel
      query={q}
      isEmpty={(d) => !d.runtime || d.runtime.length === 0}
      emptyLabel="No monitoring_log.jsonl health series for this run."
      height={300}
    >
      {(d: MonitoringSeries) => {
        const x = d.runtime ?? d.runtime ?? [];
        const traces: any[] = [];
        const push = (name: string, y?: (number | null)[], color?: string, yaxis = "y") => {
          if (y && y.some((v) => v !== null && v !== undefined))
            traces.push({
              x,
              y,
              name,
              type: "scatter",
              mode: "lines",
              line: { color, width: 2 },
              yaxis,
              connectgaps: false,
            });
        };
        push("latest pak_f1", d.latest_pak_f1, tok("--cat-1"));
        push("best pak_f1", d.best_pak_f1, tok("--sem-good"));
        push("nan/inf count", d.nan_inf_count, tok("--sem-bad"), "y2");
        push("gpu util %", d.gpu_util, tok("--cat-2"), "y2");
        push("cpu load1", d.cpu_load1, tok("--cat-5"), "y2");

        // verdict annotations (UTF-8 Korean preserved)
        const annotations = (d.verdicts ?? [])
          .map((v, i) => ({ v, i }))
          .filter(({ v }) => v)
          .slice(-6)
          .map(({ v, i }) => ({
            x: x[i],
            y: d.latest_pak_f1?.[i] ?? 0,
            text: String(v),
            showarrow: false,
            font: { size: 9, color: tok("--text-subtle") },
            yshift: 12,
          }));

        const layout = baseLayout({
          height: 300,
          annotations,
          xaxis: { ...baseLayout().xaxis, title: { text: "runtime (s)", font: { size: 11 } } },
          yaxis: { ...baseLayout().yaxis, title: { text: "pak_f1 [0,1]" }, range: [0, 1] },
          yaxis2: {
            overlaying: "y",
            side: "right",
            showgrid: false,
            title: { text: "util / nan-inf / load", font: { size: 11 } },
          },
        });
        return <ChartFrame name={`run_health_${expId}`} data={traces} layout={layout} config={PLOTLY_CONFIG} style={{ width: "100%", height: 300 }} />;
      }}
    </AsyncPanel>
  );
}
