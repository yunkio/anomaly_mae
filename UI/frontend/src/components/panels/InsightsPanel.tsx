/* <InsightsPanel> (FB-NEW B1/B2/B3/B5) — the model-grounded analytical views.
 *
 * Four panels bound to GET /api/insights/* (additive, lazy, tolerant). The dashboard
 * READS the backend's reported components/verdicts — it NEVER re-derives the score split
 * client-side (CLAUDE.md rule 3 / R2-08): B1 shows the backend's own decomposition +
 * residual/consistency gate badge; B3 shows the backend's helped/hurt verdict.
 *
 *  • B1 — score-component decomposition: teacher-recon vs disc contribution at the best
 *    epoch + the residual-gate badge + the "best epoch pre-warmup" headline.
 *  • B2 — teacher↔student divergence: adaptive vs teacher per-epoch trajectory + the 4
 *    best-epoch score-variant scalars (P3-01 markers) + the contribution band; phase-masked.
 *  • B3 — pre↔post-warmup best delta: ☆ best-pre / ♦ best-post + helped/hurt verdict.
 *  • B5 — normal-vs-anomaly score distribution: two histograms split by point_labels + SNR.
 *
 * Registry-direction-aware (the backend resolves direction) + phase-masked (warmup band).
 * Hooks live in each sub-component's body, never inside an <AsyncPanel> render-prop (F-02). */
import { useMemo } from "react";
import ChartFrame from "../charts/ChartFrame";
import {
  useInsightDivergence,
  useInsightScoreDecomposition,
  useInsightScoreDistribution,
  useInsightWarmupDelta,
} from "../../api/queries";
import AsyncPanel from "../AsyncPanel";
import { Card, fmt } from "../common";
import { baseLayout, PLOTLY_CONFIG, warmupShape, warmupLine, warmupAnnotation } from "../../viz/plotlyTemplate";

const tok = (n: string) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

interface Props {
  exp: string;
  dsKey: string;
  variant: string | null;
  warmup: number | null;
  metric?: string;
}

export default function InsightsPanel({ exp, dsKey, variant, warmup, metric = "pak_auc_f1" }: Props) {
  return (
    <div className="col" style={{ gap: 20 }}>
      <Card title="B1 · Score-component decomposition — what carries the detector">
        <ScoreDecompositionView exp={exp} dsKey={dsKey} variant={variant} />
      </Card>
      <Card title="B2 · Teacher↔student divergence — does the disc path add signal?">
        <DivergenceView exp={exp} dsKey={dsKey} variant={variant} warmup={warmup} metric={metric} />
      </Card>
      <Card title="B3 · Pre↔post-warmup best delta — did letting the student in help?">
        <WarmupDeltaView exp={exp} dsKey={dsKey} variant={variant} metric={metric} />
      </Card>
      <Card title="B5 · Normal-vs-anomaly score distribution — the ground-truth separability">
        <ScoreDistributionView exp={exp} dsKey={dsKey} variant={variant} />
      </Card>
    </div>
  );
}

/* ── B1 — score-component decomposition ─────────────────────────────────────── */
function ScoreDecompositionView({ exp, dsKey, variant }: { exp: string; dsKey: string; variant: string | null }) {
  const q = useInsightScoreDecomposition(exp, dsKey, variant);
  return (
    <AsyncPanel query={q} isEmpty={(d) => !d.available} emptyLabel="No best-epoch NPZ to decompose." height={220}>
      {(d) => {
        const c = d.components;
        const gate = d.residual_gate;
        const reconSep = c?.recon.separation ?? null;
        const discSep = c?.disc.separation ?? null;
        const traces: any[] = [
          {
            x: ["normal", "anomaly"],
            y: [c?.recon.mean_normal ?? null, c?.recon.mean_anomaly ?? null],
            name: "teacher-recon",
            type: "bar",
            marker: { color: tok("--cat-3") },
          },
          {
            x: ["normal", "anomaly"],
            y: [c?.disc.mean_normal ?? null, c?.disc.mean_anomaly ?? null],
            name: "disc contribution",
            type: "bar",
            marker: { color: tok("--cat-5") },
          },
        ];
        const layout = baseLayout({
          height: 280,
          barmode: "group",
          yaxis: { ...baseLayout().yaxis, title: { text: "mean per-timestep contribution" } },
          xaxis: { ...baseLayout().xaxis, title: { text: "point class (from point_labels)" } },
        });
        const pct = d.pct_separation_from_disc;
        return (
          <div className="col" style={{ gap: 10 }}>
            <div className="row wrap" style={{ gap: 8, alignItems: "center" }}>
              {/* the residual / disc-consistency gate badge (RV2-01 anti-FM-omission guard). */}
              <GateBadge gate={gate} />
              {d.best_epoch_pre_warmup && (
                <span className="chip warn" title="best_epoch ≤ warmup → adaptive ≈ teacher-recon-only">
                  best epoch pre-warmup
                </span>
              )}
              <span className="chip mono" title="the NPZ epoch decomposed">
                epoch {d.epoch}
                {d.best_epoch != null ? ` (best ${d.best_epoch})` : ""}
              </span>
              {pct != null && (
                <span className="chip" title="share of the normal→anomaly score gap supplied by the disc path">
                  {(pct * 100).toFixed(1)}% of separation from disc
                </span>
              )}
            </div>
            <div className="headline" style={{ fontWeight: 600 }}>
              {d.headline}
            </div>
            <ChartFrame name="insight_B1_score_decomposition" data={traces} layout={layout} config={PLOTLY_CONFIG} style={{ width: "100%", height: 280 }} />
            <div className="row wrap" style={{ gap: 16, fontSize: "var(--fs-small)" }}>
              <span>
                recon separation (anom−norm): <strong>{fmt(reconSep)}</strong>
              </span>
              <span>
                disc separation (anom−norm): <strong>{fmt(discSep)}</strong>
              </span>
            </div>
            <div className="subtle" style={{ fontSize: "var(--fs-small)" }}>
              {d.data_source}. The disc contribution is the EMPIRICAL remainder{" "}
              <span className="mono">adaptive − teacher_recon</span> read off the NPZ (not a
              re-derived scaled_disc); the gate cross-checks it against the NPZ's own{" "}
              <span className="mono">discrepancy_error</span> array.
            </div>
          </div>
        );
      }}
    </AsyncPanel>
  );
}

function GateBadge({ gate }: { gate: any }) {
  if (!gate) return null;
  if (gate.disc_active === false) {
    return (
      <span className="chip" title={gate.note}>
        disc inactive (consistency n/a)
      </span>
    );
  }
  const ok = gate.ok;
  return (
    <span className={`chip ${ok ? "" : "warn"}`} title={gate.note}>
      consistency {ok ? "✓ ok" : "⚠ drift"}
      {gate.correlation != null ? ` · r=${gate.correlation.toFixed(3)}` : ""}
    </span>
  );
}

/* ── B2 — teacher↔student divergence trajectory ─────────────────────────────── */
function DivergenceView({
  exp,
  dsKey,
  variant,
  warmup,
  metric,
}: {
  exp: string;
  dsKey: string;
  variant: string | null;
  warmup: number | null;
  metric: string;
}) {
  const q = useInsightDivergence(exp, dsKey, metric, variant);
  return (
    <AsyncPanel query={q} isEmpty={(d) => !d.available} emptyLabel="No per-epoch adaptive/teacher series." height={300}>
      {(d) => {
        const epochs = d.epochs ?? [];
        const wm = d.warmup_epochs ?? warmup;
        const traces: any[] = [];
        const adaptive = d.series?.adaptive;
        const teacher = d.series?.teacher;
        if (adaptive)
          traces.push({
            x: epochs,
            y: adaptive.values,
            name: adaptive.display_name || "adaptive",
            type: "scatter",
            mode: "lines",
            line: { color: tok("--cat-1"), width: 2 },
            connectgaps: false,
          });
        if (teacher)
          traces.push({
            x: epochs,
            y: teacher.values,
            name: teacher.display_name || "teacher",
            type: "scatter",
            mode: "lines",
            line: { color: tok("--cat-3"), width: 2, dash: "dot" },
            connectgaps: false,
          });
        // the discrepancy-contribution band (adaptive − teacher) on a secondary trace.
        if (d.discrepancy_contribution)
          traces.push({
            x: epochs,
            y: d.discrepancy_contribution,
            name: "disc contribution (adaptive − teacher)",
            type: "scatter",
            mode: "lines",
            line: { color: tok("--cat-5"), width: 1.5 },
            fill: "tozeroy",
            fillcolor: "rgba(150,90,160,0.10)",
            connectgaps: false,
            yaxis: "y2",
          });
        const shapes = [...warmupShape(wm), ...warmupLine(wm)];
        const layout = baseLayout({
          height: 320,
          shapes,
          annotations: warmupAnnotation(wm),
          yaxis: { ...baseLayout().yaxis, range: [0, 1], title: { text: metric } },
          yaxis2: {
            overlaying: "y",
            side: "right",
            title: { text: "disc contribution" },
            showgrid: false,
            zeroline: true,
          },
          xaxis: { ...baseLayout().xaxis, title: { text: "eval epoch" } },
        });
        const sc = d.best_epoch_scalars ?? {};
        return (
          <div className="col" style={{ gap: 8 }}>
            <ChartFrame name={`insight_B2_divergence_${metric}`} data={traces} layout={layout} config={PLOTLY_CONFIG} style={{ width: "100%", height: 320 }} />
            <div className="row wrap" style={{ gap: 14, fontSize: "var(--fs-small)" }}>
              {/* the 4 best-epoch score-variant SCALARS (P3-01 markers — never per-epoch lines). */}
              <span title="adaptive score best-epoch">adaptive: <strong>{fmt(sc.metrics)}</strong></span>
              <span title="teacher-recon best-epoch">teacher: <strong>{fmt(sc.teacher_recon_metrics)}</strong></span>
              <span title="student-recon best-epoch (scalar only)">student: <strong>{fmt(sc.student_recon_metrics)}</strong></span>
              <span title="discrepancy best-epoch (scalar only)">disc: <strong>{fmt(sc.disc_metrics)}</strong></span>
            </div>
            <div className="subtle" style={{ fontSize: "var(--fs-small)" }}>
              Pre-warmup, <span className="mono">adaptive ≡ teacher_recon</span> by construction; the
              post-warmup gap (shaded) is the discrepancy contribution. student/disc are best-epoch
              SCALARS (P3-01), shown as values, never fabricated per-epoch lines.
            </div>
          </div>
        );
      }}
    </AsyncPanel>
  );
}

/* ── B3 — pre↔post-warmup best delta ────────────────────────────────────────── */
function WarmupDeltaView({
  exp,
  dsKey,
  variant,
  metric,
}: {
  exp: string;
  dsKey: string;
  variant: string | null;
  metric: string;
}) {
  const q = useInsightWarmupDelta(exp, dsKey, metric, variant);
  return (
    <AsyncPanel query={q} isEmpty={(d) => !d.available} emptyLabel="No warmup boundary / per-epoch series." height={160}>
      {(d) => {
        const helped = d.student_helped;
        const pre = d.pre_warmup;
        const post = d.post_warmup;
        return (
          <div className="col" style={{ gap: 10 }}>
            <div className="row wrap" style={{ gap: 8, alignItems: "center" }}>
              <span className={`chip ${helped ? "" : "warn"}`} title="verdict from the backend (direction-aware)">
                {helped == null ? "no verdict" : helped ? "student helped ✓" : "student hurt ✗"}
              </span>
              {d.caption && <span className="chip mono">{d.caption}</span>}
              <span className="chip" title="the teacher-only warmup boundary for this leaf">
                warmup = {d.warmup_epochs}
              </span>
              <span className="chip mono">
                {metric} ({d.direction === "down" ? "↓ lower wins" : "↑ higher wins"})
              </span>
            </div>
            <div className="row wrap" style={{ gap: 24, fontSize: "var(--fs-small)" }}>
              <span>
                ☆ best PRE-warmup: <strong>{fmt(pre?.best_value)}</strong>
                {pre?.best_epoch != null ? ` @ epoch ${pre.best_epoch}` : ""}
              </span>
              <span>
                ♦ best POST-warmup: <strong>{fmt(post?.best_value)}</strong>
                {post?.best_epoch != null ? ` @ epoch ${post.best_epoch}` : ""}
              </span>
              <span>
                Δ (post − pre): <strong>{fmt(d.delta)}</strong>
              </span>
            </div>
            <div className="subtle" style={{ fontSize: "var(--fs-small)" }}>
              {d.data_source}. The teacher-only warmup is THE architecture phase boundary; a negative
              Δ means the best detector was reached BEFORE the student joined.
            </div>
          </div>
        );
      }}
    </AsyncPanel>
  );
}

/* ── B5 — normal-vs-anomaly score distribution ──────────────────────────────── */
function ScoreDistributionView({ exp, dsKey, variant }: { exp: string; dsKey: string; variant: string | null }) {
  const q = useInsightScoreDistribution(exp, dsKey, { variant });
  const centers = useMemo(() => {
    const bins = q.data?.bins ?? [];
    if (bins.length < 2) return bins;
    const out: number[] = [];
    for (let i = 0; i < bins.length - 1; i++) out.push((bins[i] + bins[i + 1]) / 2);
    return out;
  }, [q.data]);
  return (
    <AsyncPanel query={q} isEmpty={(d) => !d.available || !(d.normal_hist || d.anomaly_hist)} emptyLabel="No best-epoch NPZ for the score distribution." height={300}>
      {(d) => {
        const traces: any[] = [
          {
            x: centers.length ? centers : d.bins,
            y: d.normal_hist ?? [],
            name: "normal",
            type: "bar",
            marker: { color: tok("--cat-2"), opacity: 0.65 },
          },
          {
            x: centers.length ? centers : d.bins,
            y: d.anomaly_hist ?? [],
            name: "anomaly",
            type: "bar",
            marker: { color: tok("--sem-bad") || "#c0392b", opacity: 0.65 },
          },
        ];
        const layout = baseLayout({
          height: 300,
          barmode: "overlay",
          yaxis: { ...baseLayout().yaxis, title: { text: "count" } },
          xaxis: { ...baseLayout().xaxis, title: { text: `${d.array} (per timestep)` } },
        });
        return (
          <div className="col" style={{ gap: 8 }}>
            <div className="row wrap" style={{ gap: 8, alignItems: "center" }}>
              <span className="chip mono" title="the NPZ epoch">
                epoch {d.epoch}
                {d.best_epoch != null ? ` (best ${d.best_epoch})` : ""}
              </span>
              {d.snr != null && (
                <span
                  className={`chip ${d.snr < 0 ? "warn" : ""}`}
                  title="normal-vs-anomaly separation (negative = anti-correlated, a red flag)"
                >
                  SNR = {fmt(d.snr)}
                </span>
              )}
            </div>
            <ChartFrame name={`insight_B5_score_distribution_epoch_${d.epoch}`} data={traces} layout={layout} config={PLOTLY_CONFIG} style={{ width: "100%", height: 300 }} />
            <div className="subtle" style={{ fontSize: "var(--fs-small)" }}>
              {d.data_source}. The overlap of the two histograms IS the detector's real difficulty;
              the more they pull apart, the easier the separation.
            </div>
          </div>
        );
      }}
    </AsyncPanel>
  );
}
