/* <NpzScrubber> (NPZ-1/2/3, needs-compute) — epoch slider scrubs the per-timestep
 * score slab. Shows the chosen score-channel-vs-time trace (scattergl) + the normal-vs-
 * anomaly two-histogram (split by point_labels). Uses the echoed page/page_size
 * (P4B-04). A corrupt/half-written NPZ -> error chip for that epoch, never a crash.
 *
 * F-2 (P0-2 / V-1 / V-9): a small CHANNEL selector populated from the slab response
 * `available_arrays` (NEVER a literal; point_labels excluded — score channels only).
 * The choice threads into BOTH the score-over-time trace AND the histogram. When the
 * selected array is `fm_error` AND its histogram SNR < 0, a teaching caption explains
 * why feature-matching is excluded from the inference score (SNR is read from the
 * histogram response — never re-derived here). */
import { useMemo, useState } from "react";
import ChartFrame from "../charts/ChartFrame";
import { useScoreEpochs, useScores, useScoresHistogram } from "../../api/queries";
import AsyncPanel from "../AsyncPanel";
import { baseLayout, PLOTLY_CONFIG } from "../../viz/plotlyTemplate";
import { fmt } from "../common";

const tok = (n: string) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

interface Props {
  exp: string;
  dsKey: string;
  variant: string | null;
  warmup: number | null;
}

const DEFAULT_ARRAY = "adaptive_score";

export default function NpzScrubber({ exp, dsKey, variant, warmup }: Props) {
  const epochsQ = useScoreEpochs(exp, dsKey, variant);
  const [idx, setIdx] = useState(0);
  // F-2: the chosen score channel (default the inference score). Drives slab + histogram.
  const [array, setArray] = useState<string>(DEFAULT_ARRAY);
  const epochs = epochsQ.data?.epochs ?? [];
  const epoch = epochs[Math.min(idx, Math.max(0, epochs.length - 1))];

  const slab = useScores(exp, dsKey, epoch, variant, array);
  const hist = useScoresHistogram(exp, dsKey, epoch, variant, array);
  const prewarmup = useMemo(() => (warmup != null && epoch != null ? epoch <= warmup : false), [warmup, epoch]);

  // F-2: the channel option list is the RUNTIME `available_arrays` from the slab response
  // (NEVER a hard-coded literal), EXCLUDING point_labels (a label mask, not a score
  // channel). The currently-selected array is always kept present so a transient empty
  // response never drops the active option from the picker.
  const channels = useMemo(() => {
    const present = (slab.data?.available_arrays ?? []).filter((a) => a !== "point_labels");
    const set = new Set<string>(present);
    set.add(array);
    if (!set.has(DEFAULT_ARRAY)) set.add(DEFAULT_ARRAY);
    return Array.from(set).sort();
  }, [slab.data?.available_arrays, array]);

  // F-2 FM teaching caption: only for fm_error and only when its histogram SNR is
  // negative — pulled straight from the histogram response (snr), never re-derived.
  const fmSnr = hist.data?.snr ?? null;
  const showFmCaption = array === "fm_error" && fmSnr != null && fmSnr < 0;

  return (
    <AsyncPanel
      query={epochsQ}
      isEmpty={(d) => !d.epochs || d.epochs.length === 0}
      emptyLabel="No epoch_scores/*.npz for this leaf (best-epoch distribution unavailable)."
      height={120}
    >
      {() => (
        <div className="col" style={{ gap: 12 }}>
          <div className="toolbar">
            <label className="row" style={{ gap: 8, flex: 1 }}>
              epoch <strong className="tabular">{epoch ?? "—"}</strong>
              <input
                type="range"
                min={0}
                max={Math.max(0, epochs.length - 1)}
                value={idx}
                onChange={(e) => setIdx(Number(e.target.value))}
                style={{ flex: 1 }}
              />
              <span className="subtle">
                {idx + 1}/{epochs.length}
              </span>
            </label>
            {/* F-2: the score-channel selector — options are the runtime available_arrays
                (point_labels excluded); the choice drives the trace AND the histogram. */}
            <label className="row" style={{ gap: 6 }} title="score channel to scrub (from this NPZ's available arrays; point_labels excluded)">
              channel
              <select className="select" value={array} onChange={(e) => setArray(e.target.value)} style={{ width: 180 }}>
                {channels.map((c) => (
                  <option key={c} value={c}>
                    {c}
                    {c === DEFAULT_ARRAY ? " (inference score)" : ""}
                  </option>
                ))}
              </select>
            </label>
            {prewarmup && <span className="chip warn">pre-warmup · recon-only-gated score</span>}
          </div>

          {/* F-2: FM-negative-SNR teaching caption — why feature-matching is excluded
              from the inference score (SNR from the histogram response). */}
          {showFmCaption && (
            <div className="chip warn" style={{ whiteSpace: "normal", lineHeight: 1.4 }} title="feature-matching separability">
              feature-matching separability is negative here (SNR {fmt(fmSnr, 3)}) — empirical reason FM is excluded from
              the inference score: anomalies do not separate from normal on this channel.
            </div>
          )}

          <div className="row wrap" style={{ gap: 16, alignItems: "flex-start" }}>
            <div style={{ flex: "1 1 380px", minWidth: 320 }}>
              <AsyncPanel
                query={slab}
                softError={(d) => (d.error ? `epoch ${d.epoch}: ${d.error.replace(/_/g, " ")}` : null)}
                isEmpty={(d) => d.available === false || !d.arrays?.[array]}
                emptyLabel="NPZ absent for this epoch."
                height={280}
              >
                {(d) => {
                  const score = d.arrays?.[array] ?? [];
                  const trace: any = {
                    x: score.map((_, i) => i),
                    y: score,
                    type: "scattergl",
                    mode: "lines",
                    line: { color: tok("--cat-1"), width: 1 },
                    name: array,
                  };
                  const layout = baseLayout({
                    height: 280,
                    showlegend: false,
                    title: { text: `${array} over time` + (d.downsampled ? " (downsampled)" : ""), font: { size: 12 }, x: 0 },
                    xaxis: { ...baseLayout().xaxis, title: { text: "timestep index" } },
                  });
                  return (
                    <div>
                      <ChartFrame name={`npz_${array}_over_time_epoch_${d.epoch ?? epoch}`} data={[trace]} layout={layout} config={PLOTLY_CONFIG} style={{ width: "100%", height: 280 }} />
                      <div className="subtle" style={{ fontSize: "var(--fs-small)" }}>
                        length {d.length ?? "—"}
                        {d.page != null && d.page_size != null && (
                          <>
                            {" "}· page {d.page} (size {d.page_size}
                            {d.page_start != null ? `, [${d.page_start}..${d.page_end}]` : ""})
                          </>
                        )}
                      </div>
                    </div>
                  );
                }}
              </AsyncPanel>
            </div>

            <div style={{ flex: "1 1 380px", minWidth: 320 }}>
              <AsyncPanel
                query={hist}
                softError={(d) => (d.error ? `epoch ${d.epoch}: ${d.error.replace(/_/g, " ")}` : null)}
                isEmpty={(d) => !d.bins}
                emptyLabel="No histogram for this epoch."
                height={280}
              >
                {(d) => {
                  const x = d.bins ?? [];
                  const traces: any[] = [
                    { x, y: d.normal_hist ?? [], name: "normal", type: "bar", marker: { color: tok("--cat-11") }, opacity: 0.75 },
                    { x, y: d.anomaly_hist ?? [], name: "anomaly", type: "bar", marker: { color: tok("--cat-2") }, opacity: 0.75 },
                  ];
                  const layout = baseLayout({
                    height: 280,
                    barmode: "overlay",
                    title: { text: `${array} distribution${d.snr != null ? ` · SNR ${fmt(d.snr, 3)}` : ""}`, font: { size: 12 }, x: 0 },
                    xaxis: { ...baseLayout().xaxis, title: { text: array } },
                  });
                  return <ChartFrame name={`npz_${array}_histogram_epoch_${d.epoch ?? epoch}`} data={traces} layout={layout} config={PLOTLY_CONFIG} style={{ width: "100%", height: 280 }} />;
                }}
              </AsyncPanel>
            </div>
          </div>
        </div>
      )}
    </AsyncPanel>
  );
}
