/* <ComponentOverlayPanel> (F-6 · I3 — score-vs-time component overlay).
 *
 * At a chosen eval epoch, overlay the three score channels on the TEST time axis —
 * teacher_recon_error vs discrepancy_error vs adaptive_score — with the anomaly_regions
 * (from point_labels) shaded, so the user sees WHERE recon vs disc fires. The series are
 * downsampled by the backend (read-cheap, one NPZ). An EPOCH selector (the leaf is fixed
 * by the route) drives which NPZ is overlaid; the backend defaults to the best epoch.
 *
 * Reads GET …/insights/component-overlay. Region shading uses the backend's
 * anomaly_regions (contiguous label==1 spans on the SAME downsampled index space the
 * channels live in, so they align). Hooks live in this component body (F-02). */
import { useMemo, useState } from "react";
import ChartFrame from "../charts/ChartFrame";
import AsyncPanel from "../AsyncPanel";
import { Card } from "../common";
import { useScoreEpochs, useInsightComponentOverlay } from "../../api/queries";
import { baseLayout, PLOTLY_CONFIG } from "../../viz/plotlyTemplate";
import type { ComponentOverlay } from "../../api/types";

const tok = (n: string) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

interface Props {
  exp: string;
  dsKey: string;
  variant: string | null;
}

export default function ComponentOverlayPanel({ exp, dsKey, variant }: Props) {
  const epochsQ = useScoreEpochs(exp, dsKey, variant);
  const epochs = epochsQ.data?.epochs ?? [];
  // undefined ⇒ the backend picks the best epoch; an explicit pick overrides it.
  const [epoch, setEpoch] = useState<number | undefined>(undefined);
  const q = useInsightComponentOverlay(exp, dsKey, variant, epoch);

  const selector = (
    <label className="row" style={{ gap: 6 }} title="eval epoch to overlay (defaults to the best epoch)">
      epoch
      <select
        className="select"
        value={epoch ?? ""}
        onChange={(e) => setEpoch(e.target.value === "" ? undefined : Number(e.target.value))}
        style={{ width: 150 }}
        disabled={epochs.length === 0}
      >
        <option value="">best (default)</option>
        {epochs.map((ep) => (
          <option key={ep} value={ep}>
            epoch {ep}
          </option>
        ))}
      </select>
    </label>
  );

  return (
    <Card
      title="F-6 · Score vs time (components) — where recon vs disc fires on the test axis"
      actions={selector}
    >
      <AsyncPanel
        query={q}
        softError={(d: ComponentOverlay) => (!d.available && d.reason ? `overlay unavailable: ${d.reason.replace(/_/g, " ")}` : null)}
        isEmpty={(d: ComponentOverlay) => !d.available || !d.channels}
        emptyLabel="No epoch_scores NPZ to overlay for this leaf."
        height={360}
      >
        {(d) => <OverlayBody d={d} />}
      </AsyncPanel>
    </Card>
  );
}

function OverlayBody({ d }: { d: ComponentOverlay }) {
  const ch = d.channels ?? {};
  const xLen = useMemo(() => {
    const lens = [ch.adaptive_score, ch.teacher_recon_error, ch.discrepancy_error]
      .map((a) => (a ? a.length : 0))
      .concat((d.point_labels ?? []).length);
    return Math.max(0, ...lens);
  }, [ch, d.point_labels]);
  const x = useMemo(() => Array.from({ length: xLen }, (_, i) => i), [xLen]);

  const traces: any[] = [];
  const mk = (y: (number | null)[] | null | undefined, name: string, color: string, dash?: string) => {
    if (!y || y.length === 0) return;
    traces.push({
      x,
      y,
      name,
      type: "scattergl",
      mode: "lines",
      line: { color, width: 1.3, dash },
      connectgaps: false,
    });
  };
  // adaptive last so it sits on top; teacher (recon) and disc each own color.
  mk(ch.teacher_recon_error, "teacher_recon_error", tok("--cat-3") || "#0e9f6e", "dot");
  mk(ch.discrepancy_error, "discrepancy_error", tok("--cat-5") || "#9b59b6", "dash");
  mk(ch.adaptive_score, "adaptive_score", tok("--cat-1") || "#2563eb");

  // anomaly regions shaded (on the same downsampled index space as the channels).
  const regionColor = tok("--sem-bad") || "#c0392b";
  const shapes = (d.anomaly_regions ?? []).map((r) => ({
    type: "rect",
    xref: "x",
    yref: "paper",
    x0: r.start,
    x1: r.end,
    y0: 0,
    y1: 1,
    fillcolor: regionColor,
    opacity: 0.12,
    line: { width: 0 },
    layer: "below",
  }));

  const layout = baseLayout({
    height: 360,
    shapes,
    yaxis: { ...baseLayout().yaxis, title: { text: "per-timestep error / score" } },
    xaxis: { ...baseLayout().xaxis, title: { text: "test timestep index" + (d.downsampled ? " (downsampled)" : "") } },
  });

  return (
    <div className="col" style={{ gap: 10 }}>
      <div className="row wrap" style={{ gap: 8, alignItems: "center" }}>
        <span className="chip mono" title="the NPZ epoch overlaid">
          epoch {d.epoch}
          {d.best_epoch != null ? ` (best ${d.best_epoch})` : ""}
        </span>
        <span className="chip" title="anomaly spans (contiguous point_labels==1) shaded behind the channels">
          {(d.anomaly_regions ?? []).length} anomaly region{(d.anomaly_regions ?? []).length === 1 ? "" : "s"}
        </span>
        {d.length != null && (
          <span className="chip" title="full test length before downsampling">
            test length {d.length}
            {d.downsampled ? ` → ${d.n_points ?? xLen} plotted` : ""}
          </span>
        )}
      </div>
      <ChartFrame
        name={`insight_F6_component_overlay_epoch_${d.epoch}`}
        data={traces}
        layout={layout}
        config={PLOTLY_CONFIG}
        style={{ width: "100%", height: 360 }}
      />
      <div className="subtle" style={{ fontSize: "var(--fs-small)", whiteSpace: "normal", lineHeight: 1.45 }}>
        {d.label ?? "teacher/disc/adaptive channels on the test axis, read directly from the saved NPZ (no recompute)."}{" "}
        The shaded spans are the ground-truth anomaly regions — read off where{" "}
        <span className="mono">teacher_recon_error</span> spikes vs where <span className="mono">discrepancy_error</span> does
        to see which component is actually firing inside each anomaly. {d.data_source}
      </div>
    </div>
  );
}
