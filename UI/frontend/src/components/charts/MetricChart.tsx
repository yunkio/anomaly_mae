/* <MetricChart> — the single registry-driven time-series primitive (the F4/F5
 * chokepoint, frontend-design §4.2). Phase-mask / direction / sparsity / inferred /
 * negative-SNR ALL read from MetricMeta — never per-chart hard-coding, never a fixed
 * metric list. The `axis` is fixed per chart; the two axes are NEVER co-plotted
 * (a caller renders eval-epoch and training-epoch in separate <MetricChart>s). */
import { useMemo, useRef } from "react";
import Plot from "../../viz/Plot";
import type { Axis, MetricMeta } from "../../api/types";
import { metricStyle, isSparse } from "../../viz/useMetricStyle";
import { DownloadPngButton } from "./ChartFrame";
import {
  baseLayout,
  PLOTLY_CONFIG,
  warmupShape,
  warmupLine,
  warmupAnnotation,
} from "../../viz/plotlyTemplate";

export interface MetricSeriesInput {
  key: string;
  x: number[];
  y: (number | null)[];
  meta: MetricMeta;
  /** Optional explicit color override (e.g. stable per-run categorical color). */
  color?: string;
  /** Optional label override (e.g. the leaf label in an overlay). */
  label?: string;
}

interface Props {
  series: MetricSeriesInput[];
  axis: Axis;
  warmupEpochs?: number | null;
  height?: number;
  title?: string;
  /** force a shared [0,1] y-range (score overlays). */
  forceUnit?: boolean;
}

const tok = (n: string) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

/** A post-warmup metric must never be a real line at 0 in the warmup span: we null
 * out values where epoch <= warmup so the line truly breaks (the correctness
 * invariant), and the grey band + boundary marker explain why. */
function maskValues(x: number[], y: (number | null)[], warmup: number | null): (number | null)[] {
  if (!warmup || warmup <= 0) return y;
  return y.map((v, i) => (x[i] <= warmup ? null : v));
}

export default function MetricChart({
  series,
  axis,
  warmupEpochs,
  height = 320,
  title,
  forceUnit,
}: Props) {
  const gdRef = useRef<any>(null);
  const { traces, layout, hasInferred, hasWarn } = useMemo(() => {
    let anyScore = forceUnit ?? false;
    let inferred = false;
    let warn = false;
    const traces: any[] = series.map((s) => {
      const st = metricStyle(s.meta, lastNum(s.y));
      if (st.badge === "inferred") inferred = true;
      if (st.badge === "warn") warn = true;
      if (st.yRange) anyScore = true;
      const masked = st.maskPrewarmup ? maskValues(s.x, s.y, warmupEpochs ?? null) : s.y;
      const sparse = isSparse(s.meta);
      return {
        x: s.x,
        y: masked,
        name: s.label ?? s.meta.display_name ?? s.key,
        type: "scattergl",
        mode: sparse ? "markers" : "lines+markers",
        connectgaps: false, // gaps are real gaps, never interpolated (AC-F4.5)
        line: { color: s.color ?? st.color, dash: st.dash, width: 2, shape: "linear" },
        marker: { size: sparse ? 6 : 3, color: s.color ?? st.color },
        hovertemplate:
          `<b>${s.label ?? s.meta.display_name}</b><br>` +
          `${axis === "eval-epoch" ? "eval-epoch" : "training-epoch"} %{x}<br>` +
          `%{y:.4g} <i>(${st.directionLabel})</i><extra></extra>`,
      };
    });

    const shapes = [...warmupShape(warmupEpochs), ...warmupLine(warmupEpochs)];
    const annotations = warmupAnnotation(warmupEpochs);
    const layout = baseLayout({
      height,
      title: title ? { text: title, font: { size: 13 }, x: 0, xanchor: "left" } : undefined,
      shapes,
      annotations,
      xaxis: {
        ...baseLayout().xaxis,
        title: { text: axis === "eval-epoch" ? "eval epoch" : "training epoch", font: { size: 11 } },
      },
      yaxis: {
        ...baseLayout().yaxis,
        range: anyScore ? [0, 1] : undefined,
        title: { text: series.length === 1 ? unitLabel(series[0].meta) : "", font: { size: 11 } },
      },
      showlegend: series.length > 1,
    });
    return { traces, layout, hasInferred: inferred, hasWarn: warn };
  }, [series, axis, warmupEpochs, height, title, forceUnit]);

  return (
    <div>
      <Plot
        data={traces}
        layout={layout}
        config={PLOTLY_CONFIG}
        style={{ width: "100%", height }}
        useResizeHandler
        onInitialized={(_fig: any, gd: any) => {
          gdRef.current = gd;
        }}
        onUpdate={(_fig: any, gd: any) => {
          gdRef.current = gd;
        }}
      />
      <div className="row wrap" style={{ gap: 8, marginTop: 4, alignItems: "center" }}>
        {hasInferred && <span className="chip inferred">inferred semantics</span>}
        {hasWarn && <span className="chip warn">negative SNR · anti-correlated</span>}
        {!!warmupEpochs && warmupEpochs > 0 && (
          <span className="chip" style={{ color: tok("--plot-mask-line") }}>
            pre-warmup span masked
          </span>
        )}
        {/* FB-R3-03: download this chart as a high-res PNG (callers inherit for free). */}
        <span style={{ flex: 1 }} />
        <DownloadPngButton
          gdRef={gdRef}
          name={title || series.map((s) => s.key).join("_") || "metric_chart"}
        />
      </div>
    </div>
  );
}

function lastNum(y: (number | null)[]): number | null {
  for (let i = y.length - 1; i >= 0; i--) if (typeof y[i] === "number") return y[i] as number;
  return null;
}
function unitLabel(meta: MetricMeta): string {
  return meta.unit || meta.family || "";
}
