/* PerformanceBasis (PERF_BASIS_SPEC) — the ONE compact control that drives which
 * epoch snapshot + which threshold every perf TABLE (rankings / per-dataset /
 * Overview top-models / leaderboard / compare matrix) resolves its values under.
 * Reads/writes the global store so a change anywhere reflects everywhere. Default
 * (Best, Optimal) = current behaviour, byte-identical.
 *
 *   Epoch [ Best | 300 | 350 | 400 | 450 | Last ]   Threshold [ Optimal | Anomaly-ratio ]  ⓘ
 */
import { useStore } from "../store";
import type { EpochBasis } from "../store";

const INFO =
  "Best = epoch with the best pak_auc_f1 (all epochs); Best≥WU = best pak_auc_f1 restricted " +
  "to AFTER warm-up; ES = epoch chosen by a leakage-free early-stopping rule on the G_e " +
  "game-health signal (EMA α=0.1, patience 2, ≥1% rel. improvement, post-warm-up only, " +
  "rolled back to the best-G_e epoch); Last = final epoch; 300 / 350 / 400 / 450 = value at " +
  "that eval epoch (— if the run never reached it). " +
  "Threshold applies to F1 / Affiliation-F1 / etc.; pak_auc_f1 is threshold-free (unchanged).";

/* Epoch-basis options in display order — selection criteria (Best, post-WU, ES) first,
 * then fixed epochs, then Last. */
const EPOCH_OPTIONS: { value: EpochBasis; label: string; title: string }[] = [
  { value: "best", label: "Best", title: "epoch maximizing pak_auc_f1 over ALL epochs (current)" },
  { value: "best_post", label: "Best≥WU", title: "best pak_auc_f1 restricted to post-warm-up epochs (epoch > teacher_only_warmup_epochs)" },
  { value: "es", label: "ES", title: "early-stopping selection on G_e (leakage-free: EMA α=0.1, patience 2, ≥1% rel. improvement, post-warm-up only, rolled back to best-G_e epoch)" },
  { value: "300", label: "300", title: "value at eval epoch 300 (— if the run never reached it)" },
  { value: "350", label: "350", title: "value at eval epoch 350 (— if the run never reached it)" },
  { value: "400", label: "400", title: "value at eval epoch 400 (— if the run never reached it)" },
  { value: "450", label: "450", title: "value at eval epoch 450 (— if the run never reached it)" },
  { value: "last", label: "Last", title: "final eval epoch" },
];

export default function PerformanceBasis() {
  const epochBasis = useStore((s) => s.epochBasis);
  const thresholdBasis = useStore((s) => s.thresholdBasis);
  const setEpochBasis = useStore((s) => s.setEpochBasis);
  const setThresholdBasis = useStore((s) => s.setThresholdBasis);

  return (
    <div className="row" style={{ gap: 12, alignItems: "center", flexWrap: "wrap" }}>
      <span className="row" style={{ gap: 6, alignItems: "center" }}>
        <span className="subtle" style={{ fontSize: "var(--fs-small)" }}>
          Epoch
        </span>
        <span className="seg" role="group" aria-label="epoch basis">
          {EPOCH_OPTIONS.map((o) => (
            <button
              key={o.value}
              className={`seg-btn ${epochBasis === o.value ? "active" : ""}`}
              onClick={() => setEpochBasis(o.value)}
              title={o.title}
            >
              {o.label}
            </button>
          ))}
        </span>
      </span>

      <span className="row" style={{ gap: 6, alignItems: "center" }}>
        <span className="subtle" style={{ fontSize: "var(--fs-small)" }}>
          Threshold
        </span>
        <span className="seg" role="group" aria-label="threshold basis">
          <button
            className={`seg-btn ${thresholdBasis === "optimal" ? "active" : ""}`}
            onClick={() => setThresholdBasis("optimal")}
            title="F1-optimal threshold (current)"
          >
            Optimal
          </button>
          <button
            className={`seg-btn ${thresholdBasis === "anomaly_ratio" ? "active" : ""}`}
            onClick={() => setThresholdBasis("anomaly_ratio")}
            title="threshold = test anomaly ratio (leakage-free _ar variant)"
          >
            Anomaly-ratio
          </button>
        </span>
      </span>

      <span
        className="subtle"
        style={{ fontSize: "var(--fs-small)", cursor: "help" }}
        title={INFO}
        aria-label={INFO}
      >
        ⓘ
      </span>
    </div>
  );
}
