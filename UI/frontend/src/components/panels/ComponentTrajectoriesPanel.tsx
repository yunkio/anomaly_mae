/* <ComponentTrajectoriesPanel> (F-5 · E-3 / I1+I2 — THE FLAGSHIP insight).
 *
 * The 3 RECOMPUTED-ABLATION detection trajectories — adaptive (production), teacher-only
 * (recon-path ablation), disc-only (discrepancy-path ablation) — over eval epochs, vs
 * point_labels, recomputed numpy-only from the saved NPZ arrays by the backend. A metric
 * toggle (roc / ap / pak_f1) drives the displayed series. The warmup boundary is marked;
 * epochs with no usable NPZ render as HONEST GAPS (connectgaps:false — P3-01: a real
 * raw-array ablation, never a fabricated adaptive series).
 *
 * EV2A-N1: binds the NESTED payload shape — channels.<name>.trajectory.<metric> aligned
 * to `epochs`, the channel `is_ablation` flag surfaced, and `self_validation` shown
 * verbatim (recomputed roc vs stored roc + Δ). EV2A-N2: the I2 efficacy verdict CARD
 * shows metric_for_verdict + meaningful_lift_threshold + delta_best/delta_mean + onset so
 * the weak/confirmed call is AUDITABLE, not a bare badge.
 *
 * disc-only is labeled honestly — on 271/PSM it is weak post-warmup; that is a real
 * negative result and is shown as such, not papered over.
 *
 * Hooks live in this component's body, never inside an <AsyncPanel> render-prop (F-02). */
import { useState } from "react";
import ChartFrame from "../charts/ChartFrame";
import AsyncPanel from "../AsyncPanel";
import { Card, fmt } from "../common";
import { useInsightComponentTrajectories } from "../../api/queries";
import { baseLayout, PLOTLY_CONFIG, warmupShape, warmupLine, warmupAnnotation } from "../../viz/plotlyTemplate";
import type { ComponentTrajectories, TrajChannel } from "../../api/types";

const tok = (n: string) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

type TrajMetric = "roc" | "ap" | "pak_f1";
const METRIC_LABEL: Record<TrajMetric, string> = {
  roc: "ROC-AUC",
  ap: "Average Precision",
  pak_f1: "PA%K best-F1",
};

interface Props {
  exp: string;
  dsKey: string;
  variant: string | null;
  warmup: number | null;
}

export default function ComponentTrajectoriesPanel({ exp, dsKey, variant, warmup }: Props) {
  // the displayed-trajectory metric (client toggle). The efficacy verdict ALWAYS keys on
  // roc (per EV2A-N2 / the backend default) — kept separate so the toggle never moves the
  // verdict goalposts.
  const [metric, setMetric] = useState<TrajMetric>("roc");
  // the verdict metric stays "roc" so the response includes a roc-keyed verdict regardless
  // of the displayed trajectory; one request serves all three trajectory metrics.
  const q = useInsightComponentTrajectories(exp, dsKey, variant, "roc");

  const toggle = (
    <div className="pill-toggle" title="which recomputed detection metric to plot (one fetch carries all three)">
      {(["roc", "ap", "pak_f1"] as TrajMetric[]).map((m) => (
        <button key={m} className={metric === m ? "on" : ""} onClick={() => setMetric(m)} title={METRIC_LABEL[m]}>
          {m}
        </button>
      ))}
    </div>
  );

  return (
    <Card
      title="F-5 · Component detection trajectories — what carries detection, epoch by epoch (RECOMPUTED ABLATION)"
      actions={toggle}
    >
      <AsyncPanel
        query={q}
        isEmpty={(d: ComponentTrajectories) => !d.available || !d.channels}
        emptyLabel="No epoch_scores NPZ to recompute component trajectories for this leaf."
        height={360}
      >
        {(d) => <TrajBody d={d} metric={metric} warmup={warmup} />}
      </AsyncPanel>
    </Card>
  );
}

function TrajBody({ d, metric, warmup }: { d: ComponentTrajectories; metric: TrajMetric; warmup: number | null }) {
  const epochs = d.epochs ?? [];
  const wm = d.warmup_epochs ?? warmup;
  const ch = d.channels!;
  const avail = new Set(d.metrics_available ?? ["roc", "ap", "pak_f1"]);
  // if the toggle landed on a metric the backend did not compute (e.g. pak disabled),
  // fall back to roc so the chart is never blank.
  const m: TrajMetric = avail.has(metric) ? metric : "roc";

  const mk = (chan: TrajChannel, name: string, color: string, dash?: string) => ({
    x: epochs,
    y: chan.trajectory[m] ?? [],
    name: name + (chan.is_ablation ? " (ablation)" : ""),
    type: "scatter" as const,
    mode: "lines+markers" as const,
    line: { color, width: 2, dash },
    marker: { size: 4 },
    connectgaps: false, // P3-01: honest GAPS where an epoch has no NPZ — never bridged.
  });

  const traces: any[] = [
    mk(ch.adaptive, "adaptive (production)", tok("--cat-1") || "#2563eb"),
    mk(ch.teacher_only, "teacher-only", tok("--cat-3") || "#0e9f6e", "dot"),
    mk(ch.disc_only, "disc-only", tok("--cat-5") || "#9b59b6", "dash"),
  ];

  const yIsBounded = m === "roc" || m === "pak_f1"; // roc & PA%K-F1 live in [0,1]; AP too
  const layout = baseLayout({
    height: 360,
    shapes: [...warmupShape(wm), ...warmupLine(wm)],
    annotations: warmupAnnotation(wm),
    yaxis: {
      ...baseLayout().yaxis,
      title: { text: `recomputed ${METRIC_LABEL[m]} (vs point_labels)` },
      range: yIsBounded ? [0, 1] : undefined,
    },
    xaxis: { ...baseLayout().xaxis, title: { text: "eval epoch" } },
  });

  const sv = d.self_validation;
  const ev = d.efficacy_verdict;

  return (
    <div className="col" style={{ gap: 12 }}>
      {/* honesty header — this is a recomputed RAW-ARRAY ablation, not the live score. */}
      <div className="row wrap" style={{ gap: 8, alignItems: "center" }}>
        <span className="chip mono" title="recomputed numpy-only from the saved NPZ arrays — a real channel ablation, not the live inference score">
          {d.label ?? "RECOMPUTED ABLATION"}
        </span>
        {wm != null && <span className="chip">warmup = {wm}</span>}
        {d.best_epoch != null && <span className="chip">best epoch {d.best_epoch}</span>}
        <span className="chip" title="eval epochs with a usable NPZ vs honest gaps (no NPZ)">
          {d.n_npz_used ?? 0}/{d.n_npz ?? 0} epochs · {d.n_gap ?? 0} gap{(d.n_gap ?? 0) === 1 ? "" : "s"}
        </span>
      </div>

      <ChartFrame
        name={`insight_F5_component_trajectories_${m}`}
        data={traces}
        layout={layout}
        config={PLOTLY_CONFIG}
        style={{ width: "100%", height: 360 }}
      />

      {/* I2 EFFICACY VERDICT CARD (EV2A-N2: auditable — metric + threshold + both deltas + onset). */}
      {ev && <EfficacyCard ev={ev} />}

      {/* self-validation provenance note (EV2A-N1: recomputed vs stored + Δ, verbatim). */}
      {sv && <SelfValidationNote sv={sv} />}

      <div className="subtle" style={{ fontSize: "var(--fs-small)", whiteSpace: "normal", lineHeight: 1.45 }}>
        {d.p3_01_note ??
          "teacher-only / disc-only are RECOMPUTED channel ablations from the saved raw arrays (gaps where no NPZ); NOT fabricated per-epoch metric trajectories."}{" "}
        <strong>disc-only</strong> is shown honestly — where it stays flat/weak post-warmup, the discrepancy
        path is not carrying standalone detection (a real result, not a defect). {d.data_source}
      </div>
    </div>
  );
}

/* I2 — the disc-self-distillation efficacy verdict, fully auditable. */
function EfficacyCard({ ev }: { ev: NonNullable<ComponentTrajectories["efficacy_verdict"]> }) {
  const confirmed = ev.verdict === "disc_efficacy_confirmed";
  const indeterminate = ev.verdict !== "disc_efficacy_confirmed" && ev.verdict !== "disc_efficacy_weak";
  const cls = confirmed ? "good" : indeterminate ? "" : "warn";
  return (
    <div className={`callout ${cls}`} style={{ padding: 12, borderRadius: 8, border: "1px solid var(--border)", background: "var(--surface-2, var(--surface))" }}>
      <div className="row wrap" style={{ gap: 8, alignItems: "center", marginBottom: 6 }}>
        <span className="card-title" style={{ margin: 0 }}>I2 · Self-distillation efficacy</span>
        <span className={`chip ${cls}`} title="does disc-only detection rise meaningfully AFTER warmup?">
          {confirmed ? "disc efficacy confirmed ✓" : indeterminate ? ev.verdict.replace(/_/g, " ") : "disc efficacy weak ✗"}
        </span>
        {ev.metric && (
          <span className="chip mono" title="the trajectory metric the verdict keys on (EV2A-N2: roc)">
            keys on {ev.metric} · lift ≥ {ev.meaningful_lift_threshold ?? 0.02}
          </span>
        )}
      </div>
      <div style={{ fontSize: "var(--fs-small)", whiteSpace: "normal", lineHeight: 1.45, marginBottom: 8 }}>{ev.text}</div>
      <div className="row wrap" style={{ gap: 18, fontSize: "var(--fs-small)" }}>
        <span title="best disc-only detection at epochs ≤ warmup">pre-warmup best: <strong>{fmt(ev.pre_best)}</strong></span>
        <span title="best disc-only detection at epochs > warmup">post-warmup best: <strong>{fmt(ev.post_best)}</strong></span>
        <span title="post-best − pre-best (the lift the threshold judges)">Δ best: <strong>{fmt(ev.delta_best)}</strong></span>
        <span title="post-mean − pre-mean">Δ mean: <strong>{fmt(ev.delta_mean)}</strong></span>
        <span title="first post-warmup epoch whose disc-only detection exceeds the pre ceiling">
          onset: <strong>{ev.onset_epoch != null ? `epoch ${ev.onset_epoch}` : "—"}</strong>
        </span>
      </div>
    </div>
  );
}

/* the faithfulness provenance — adaptive recompute ≈ stored. */
function SelfValidationNote({ sv }: { sv: NonNullable<ComponentTrajectories["self_validation"]> }) {
  const rc = sv.recomputed?.roc;
  const st = sv.stored?.roc;
  const dRoc = sv.abs_delta?.roc;
  const okChip =
    sv.ok === true ? <span className="chip good" title={sv.note}>self-validated ✓</span>
    : sv.ok === false ? <span className="chip warn" title={sv.note}>recompute drift ⚠</span>
    : <span className="chip" title={sv.note ?? sv.reason}>no stored metric to validate</span>;
  return (
    <div className="row wrap" style={{ gap: 8, alignItems: "center", fontSize: "var(--fs-small)" }}>
      {okChip}
      <span className="subtle" style={{ whiteSpace: "normal", lineHeight: 1.4 }}>
        recomputed ablation from saved NPZ arrays (numpy); self-validation: adaptive recompute ≈ stored
        {rc != null && st != null ? (
          <>
            {" "}(roc <strong className="mono">{fmt(rc)}</strong> vs <strong className="mono">{fmt(st)}</strong>
            {dRoc != null ? <>, Δ<strong className="mono">{fmt(dRoc)}</strong></> : null})
          </>
        ) : null}
        {sv.epoch != null ? ` — validated at epoch ${sv.epoch}` : ""}
        {sv.tolerance != null ? ` (tol ${sv.tolerance})` : ""}.
      </span>
    </div>
  );
}
