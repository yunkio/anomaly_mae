/* <EpochTrendPanel> (TREND-1/2/3) — pick metrics from the runtime catalog and plot
 * over the eval-epoch axis via <MetricChart>. Overlay vs small-multiples (group by
 * family, AC-F4.3) + the direction-aware summary-stat strip. */
import { useMemo, useState } from "react";
import { useMetricsCatalog, useEpochSeries, useStats } from "../../api/queries";
import MetricFamilySelector from "../MetricFamilySelector";
import MetricChart, { MetricSeriesInput } from "../charts/MetricChart";
import AsyncPanel from "../AsyncPanel";
import { fmt } from "../common";
import { api } from "../../api/client";
import type { StatRow } from "../../api/types";

interface Props {
  exp: string;
  dsKey: string;
  variant: string | null;
  warmup: number | null;
}

export default function EpochTrendPanel({ exp, dsKey, variant, warmup }: Props) {
  const cat = useMetricsCatalog(exp, dsKey, variant);
  const [keys, setKeys] = useState<string[]>(["pak_auc_f1"]);
  const [layout, setLayout] = useState<"overlay" | "multiples">("multiples");
  const series = useEpochSeries(exp, dsKey, keys, variant);
  const stats = useStats(exp, dsKey, keys, variant);

  // only eval-epoch namespace keys are valid here; training-history is a separate tab.
  const epochCatalog = useMemo(
    () => (cat.data ?? []).filter((c) => c.namespace === "epoch_metrics"),
    [cat.data]
  );

  return (
    <div className="col" style={{ gap: 16 }}>
      <div className="row wrap" style={{ gap: 12, alignItems: "flex-start" }}>
        <div style={{ flex: "1 1 360px", minWidth: 280 }}>
          <AsyncPanel query={cat} title="Metrics (runtime catalog · F5)" height={160}>
            {() => (
              <MetricFamilySelector catalog={epochCatalog} selected={keys} onChange={setKeys} multi leafVariant={variant} />
            )}
          </AsyncPanel>
        </div>
        <div style={{ flex: "2 1 520px", minWidth: 320 }}>
          <div className="toolbar">
            <div className="pill-toggle">
              <button className={layout === "overlay" ? "on" : ""} onClick={() => setLayout("overlay")}>
                overlay
              </button>
              <button className={layout === "multiples" ? "on" : ""} onClick={() => setLayout("multiples")}>
                small multiples
              </button>
            </div>
            <a className="btn sm" href={api.exportSeriesUrl(exp, dsKey, keys, variant)}>
              ↓ CSV
            </a>
          </div>
          <AsyncPanel
            query={series}
            isEmpty={(d) => !d.epochs || d.epochs.length === 0}
            isPartial={() => false}
            height={320}
          >
            {(d) => {
              const inputs: MetricSeriesInput[] = keys
                .filter((k) => d.series[k])
                .map((k) => ({ key: k, x: d.epochs, y: d.series[k], meta: d.meta[k] }));
              if (layout === "overlay") {
                // overlay only if same family (avoid mixing a loss↓ with a detection↑)
                return <MetricChart series={inputs} axis="eval-epoch" warmupEpochs={d.warmup_epochs ?? warmup} />;
              }
              // small-multiples grouped by family
              const byFam = new Map<string, MetricSeriesInput[]>();
              for (const s of inputs) {
                const f = s.meta.family || "uncategorized";
                if (!byFam.has(f)) byFam.set(f, []);
                byFam.get(f)!.push(s);
              }
              return (
                <div className="col" style={{ gap: 12 }}>
                  {Array.from(byFam.entries()).map(([fam, ss]) => (
                    <div key={fam}>
                      <div className="subtle" style={{ fontSize: "var(--fs-small)", fontWeight: 600 }}>
                        {fam}
                      </div>
                      <MetricChart series={ss} axis="eval-epoch" warmupEpochs={d.warmup_epochs ?? warmup} height={240} />
                    </div>
                  ))}
                  {d.requested_missing && d.requested_missing.length > 0 && (
                    <div className="subtle" style={{ fontSize: "var(--fs-small)" }}>
                      not on the eval-epoch axis (best-epoch scalars only):{" "}
                      <span className="mono">{d.requested_missing.join(", ")}</span>
                    </div>
                  )}
                </div>
              );
            }}
          </AsyncPanel>
        </div>
      </div>

      <AsyncPanel query={stats} title="Summary statistics (direction-aware)" height={120}>
        {(raw) => {
          // FB-R3-02: the client adapter now always returns a StatRow[] (dict→array,
          // `key` injected). Consume it directly — the brittle `(raw as any).rows` path
          // that left the grid empty is gone.
          const rows: StatRow[] = Array.isArray(raw) ? raw : [];
          return (
            <div className="stat-grid">
              {rows.map((r) => (
                <div className="stat-card" key={r.key}>
                  <div className="label">{r.key}</div>
                  <div className={`value tabular ${r.direction === "neutral" ? "neutral" : ""}`}>{fmt(r.best)}</div>
                  <div className="subtle" style={{ fontSize: "var(--fs-small)" }}>
                    {r.direction === "neutral" ? "no best (context)" : `@epoch ${r.argbest_epoch ?? "—"}`} ·
                    mean {fmt(r.mean, 3)} · last {fmt(r.last, 3)}
                    {r.plateau ? " · plateau" : ""}
                  </div>
                </div>
              ))}
            </div>
          );
        }}
      </AsyncPanel>
    </div>
  );
}
