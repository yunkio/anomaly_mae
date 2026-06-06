/* <ScoreVariantPanel> (CMP-4 "what carries the detector") — HONORS P3-01.
 *
 * The 4 score variants (metrics=adaptive, teacher_recon, student_recon, disc) are
 * BEST-EPOCH SCALARS (from DatasetDetail.score_variants). Only adaptive (epoch
 * `metrics`) + teacher_* are per-eval-epoch SERIES — so we draw:
 *   • adaptive + teacher_<metric> as per-epoch LINES (from /epoch-series), and
 *   • all 4 variants' best-epoch values as horizontal MARKER lines + a strip/dot row
 *     (P3-05 cross-run distribution = strip/dot, never a fabricated 4-line trajectory).
 * student / disc are NEVER drawn as per-epoch lines (the backend returns them in
 * requested_missing). This is GIF-6's interactive twin. */
import { useMemo, useState } from "react";
import ChartFrame from "../charts/ChartFrame";
import { useDataset, useEpochSeries, useMetricsCatalog } from "../../api/queries";
import AsyncPanel from "../AsyncPanel";
import MetricPicker from "../MetricPicker";
import { baseLayout, PLOTLY_CONFIG, warmupShape, warmupLine } from "../../viz/plotlyTemplate";
import { fmt } from "../common";
import type { DatasetDetail } from "../../api/types";

const tok = (n: string) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

const VARIANT_BLOCKS = ["metrics", "teacher_recon_metrics", "student_recon_metrics", "disc_metrics"];
const VARIANT_COLORS: Record<string, string> = {
  metrics: "--cat-1",
  teacher_recon_metrics: "--cat-3",
  student_recon_metrics: "--cat-2",
  disc_metrics: "--cat-5",
};
const VARIANT_LABEL: Record<string, string> = {
  metrics: "adaptive (score)",
  teacher_recon_metrics: "teacher-recon",
  student_recon_metrics: "student-recon",
  disc_metrics: "discrepancy",
};

interface Props {
  exp: string;
  dsKey: string;
  variant: string | null;
  warmup: number | null;
}

export default function ScoreVariantPanel({ exp, dsKey, variant, warmup }: Props) {
  const [metric, setMetric] = useState("pak_auc_f1");
  const ds = useDataset(exp, dsKey, variant);
  const cat = useMetricsCatalog(exp, dsKey, variant);
  // per-epoch lines: adaptive (`metric`) + the teacher version (`teacher_<metric>`).
  const teacherKey = `teacher_${metric}`;
  const series = useEpochSeries(exp, dsKey, [metric, teacherKey], variant);

  // F-01/F5: the option list is the RUNTIME catalog (no hard-coded array). Restrict to
  // eval-epoch detection metrics that HAVE a `teacher_<m>` counterpart in the catalog,
  // since this panel pairs adaptive vs teacher per epoch.
  const eligibleCatalog = useMemo(() => {
    const all = cat.data ?? [];
    const teacherSet = new Set(all.filter((c) => c.key.startsWith("teacher_")).map((c) => c.key));
    return all.filter((c) => c.namespace === "epoch_metrics" && teacherSet.has(`teacher_${c.key}`));
  }, [cat.data]);

  // F-02: scalars computed at the TOP LEVEL from query.data (null-guarded), never as a
  // hook inside the AsyncPanel render-prop (whose hook count changes across states).
  const scalars = useMemo(() => variantScalars(ds.data, metric), [ds.data, metric]);

  return (
    <div className="col" style={{ gap: 8 }}>
      <div className="toolbar">
        <MetricPicker
          catalog={eligibleCatalog}
          value={metric}
          onChange={setMetric}
          namespaceFilter={["epoch_metrics"]}
          loading={cat.isLoading}
        />
        <span className="subtle" style={{ fontSize: "var(--fs-small)" }}>
          adaptive + teacher = per-epoch lines · all 4 variants = best-epoch markers (P3-01)
        </span>
      </div>

      <AsyncPanel query={ds} height={300}>
        {(_d) => {
          const lineTraces: any[] = [];
          if (series.data?.epochs?.length) {
            for (const k of [metric, teacherKey]) {
              const y = series.data.series[k];
              if (y && y.some((v) => v !== null))
                lineTraces.push({
                  x: series.data.epochs,
                  y,
                  name: k,
                  type: "scatter",
                  mode: "lines",
                  line: { color: tok(k === metric ? "--cat-1" : "--cat-3"), width: 2 },
                });
            }
          }
          // best-epoch scalar markers (horizontal lines spanning the x range)
          const xmax = series.data?.epochs?.length ? series.data.epochs[series.data.epochs.length - 1] : 100;
          const shapes = [
            ...warmupShape(_d.warmup_epochs ?? warmup),
            ...warmupLine(_d.warmup_epochs ?? warmup),
            ...scalars
              .filter((s) => s.value !== null)
              .map((s) => ({
                type: "line" as const,
                x0: 0,
                x1: xmax,
                y0: s.value as number,
                y1: s.value as number,
                line: { color: tok(VARIANT_COLORS[s.block]), width: 1.4, dash: "dash" as const },
              })),
          ];
          const annotations = scalars
            .filter((s) => s.value !== null)
            .map((s) => ({
              x: xmax,
              y: s.value as number,
              xanchor: "right" as const,
              text: `${VARIANT_LABEL[s.block]} = ${fmt(s.value)}`,
              showarrow: false,
              font: { size: 9, color: tok(VARIANT_COLORS[s.block]) },
            }));

          const layout = baseLayout({
            height: 300,
            shapes,
            annotations,
            yaxis: { ...baseLayout().yaxis, range: [0, 1], title: { text: metric } },
            xaxis: { ...baseLayout().xaxis, title: { text: "eval epoch" } },
          });

          return (
            <div className="col" style={{ gap: 8 }}>
              <ChartFrame name={`score_variants_${metric}`} data={lineTraces} layout={layout} config={PLOTLY_CONFIG} style={{ width: "100%", height: 300 }} />
              {/* strip/dot best-epoch distribution (P3-05) */}
              <StripDot scalars={scalars} />
              <div className="subtle" style={{ fontSize: "var(--fs-small)" }}>
                Pre-warmup, <span className="mono">metrics ≡ teacher_recon</span> by construction; the post-warmup gap
                between adaptive and teacher is the discrepancy contribution.
              </div>
            </div>
          );
        }}
      </AsyncPanel>
    </div>
  );
}

/** The 4 score-variant best-epoch scalars for `metric` (null-guarded so it can be
 * computed at the top level, hoisted out of the AsyncPanel render-prop — F-02). */
function variantScalars(d: DatasetDetail | undefined, metric: string): { block: string; value: number | null }[] {
  return VARIANT_BLOCKS.map((b) => {
    const v = d?.score_variants?.[b]?.[metric];
    return { block: b, value: typeof v === "number" ? v : null };
  });
}

function StripDot({ scalars }: { scalars: { block: string; value: number | null }[] }) {
  const x = scalars.map((s) => s.value);
  const labels = scalars.map((s) => VARIANT_LABEL[s.block]);
  const colors = scalars.map((s) => tok(VARIANT_COLORS[s.block]));
  const trace: any = {
    x,
    y: labels,
    type: "scatter",
    mode: "markers",
    marker: { size: 13, color: colors, line: { width: 1, color: "var(--surface)" } },
    hovertemplate: "%{y}: %{x:.4f}<extra></extra>",
  };
  const layout = baseLayout({
    height: 150,
    showlegend: false,
    margin: { l: 110, r: 16, t: 8, b: 32 },
    xaxis: { ...baseLayout().xaxis, range: [0, 1], title: { text: "best-epoch value (strip/dot)" } },
    yaxis: { ...baseLayout().yaxis, automargin: true },
  });
  return <ChartFrame name="score_variants_best_epoch_strip" data={[trace]} layout={layout} config={PLOTLY_CONFIG} style={{ width: "100%", height: 150 }} />;
}
