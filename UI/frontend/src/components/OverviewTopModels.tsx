/* <OverviewTopModels> — RV3-01 fix (FB-12/FB-13).
 *
 * The Overview's "winning models" summary must NOT be a cross-dataset VALUE mix (the
 * FB-12 anti-pattern). Instead it shows the TRUE FB-13 model ranking: each model's rank
 * is the MEAN of its per-dataset ranks (a model is ranked within each dataset, then the
 * per-dataset ranks are averaged). Sourced from GET /api/rankings/aggregate — the SAME
 * endpoint the full Rankings view uses, so the two surfaces agree by construction.
 *
 * This is a compact top-N table: aggregate rank · experiment · avg rank · coverage ·
 * mean value. The aggregate is ALREADY sorted ascending by avg_rank (rank 1 = best)
 * with ties broken by (−coverage, −mean_value) on the backend, so we just slice the
 * head. Coverage = datasets present / total dataset columns (SWaT full & excl22 count
 * as two columns, FB-11). A missing per-dataset entry is a real gap; the aggregate's
 * own coverage chip conveys it — no per-dataset value column is shown here (Overview
 * must never present a pooled cross-dataset value column as "the ranking").
 *
 * F-1 (P0-1): the headline top-N renders the QUALIFIED `aggregate` ONLY (rows scored on
 * >= min_coverage dataset columns); under-covered runs live in `low_coverage` and are
 * surfaced under a COLLAPSED "insufficient coverage (n/N)" section, never in the
 * headline (so a 1/14-coverage exp can't ride near the top).
 *
 * ADDITIVE: the shared <Leaderboard> is untouched (Rankings + any scoped usage keep
 * working); this is a new, self-contained widget. */
import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useRankingsAggregate, useGlobalMetricsCatalog } from "../api/queries";
import AsyncPanel from "./AsyncPanel";
import MetricPicker from "./MetricPicker";
import PerformanceBasis from "./PerformanceBasis";
import DatasetSetSelector from "./DatasetSetSelector";
import { useStore } from "../store";
import { fmt } from "./common";

interface Props {
  /** initial selection metric (default the primary pak_auc_f1). */
  metric?: string;
  /** how many top models to show in the compact summary. */
  topN?: number;
}

export default function OverviewTopModels({ metric: metric0 = "pak_auc_f1", topN = 8 }: Props) {
  // (3) 2026-06-04: like the Rankings screen, let the user choose the ranking METRIC and
  // WHICH dataset columns the average-rank is computed over. `selDatasets=null` = the
  // backend canonical default. A separate universe query (no dataset filter) advertises
  // the selectable columns + default set; the main query re-ranks over the active set.
  const [metric, setMetric] = useState(metric0);
  const [selDatasets, setSelDatasets] = useState<string[] | null>(null);
  // PERF_BASIS_SPEC: thread the global (epoch × threshold) basis into both aggregate
  // queries so the Overview top-models table re-ranks consistently with Rankings.
  const epochBasis = useStore((s) => s.epochBasis);
  const thresholdBasis = useStore((s) => s.thresholdBasis);
  const catalog = useGlobalMetricsCatalog();
  const universeQ = useRankingsAggregate({ metric, swat: "both", epoch_basis: epochBasis, threshold_basis: thresholdBasis });
  const universe = useMemo(() => universeQ.data?.selectable_columns ?? [], [universeQ.data]);
  const defaultCols = useMemo(() => universeQ.data?.default_columns ?? [], [universeQ.data]);
  const columnMeta = useMemo(() => universeQ.data?.column_meta ?? {}, [universeQ.data]);
  const activeSet = selDatasets == null ? defaultCols : selDatasets;
  const q = useRankingsAggregate({
    metric,
    swat: "both",
    datasets: activeSet.length ? activeSet.join(",") : undefined,
    epoch_basis: epochBasis,
    threshold_basis: thresholdBasis,
  });

  return (
    <div className="col" style={{ gap: 12 }}>
      <div className="toolbar" style={{ marginBottom: 0 }}>
        <MetricPicker catalog={catalog.data ?? []} value={metric} onChange={setMetric} loading={catalog.isLoading} />
        <span className="grow" style={{ flex: 1 }} />
        <PerformanceBasis />
      </div>
      <DatasetSetSelector
        universe={universe}
        defaultCols={defaultCols}
        columnMeta={columnMeta}
        value={selDatasets}
        onChange={setSelDatasets}
      />
      <AsyncPanel
      query={q}
      isEmpty={(d) => !d.aggregate || d.aggregate.length === 0}
      emptyLabel="No ranked models yet (need ≥1 complete leaf for this metric)."
      height={220}
    >
      {(d) => {
        const top = d.aggregate.slice(0, topN);
        const ranked = d.rankable;
        return (
          <div className="col" style={{ gap: 8 }}>
            <div className="subtle" style={{ fontSize: "var(--fs-small)" }}>
              True model ranking (FB-13): a model's rank = <strong>mean of its per-dataset ranks</strong> over the{" "}
              <strong>{d.coverage_total}</strong> dataset column{d.coverage_total === 1 ? "" : "s"} where it is present
              (SWaT full &amp; excl22 count separately, FB-11) — never a cross-dataset value mix.
              {!ranked && " · neutral metric — not ranked."}
            </div>
            <div className="tbl-wrap" style={{ maxHeight: 320 }}>
              <table className="tbl tabular">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>experiment</th>
                    <th className="num">avg rank</th>
                    <th className="num">coverage</th>
                    <th className="num">mean {d.metric}</th>
                  </tr>
                </thead>
                <tbody>
                  {top.map((a, i) => (
                    <tr key={a.exp_id}>
                      <td>{a.aggregate_rank ?? i + 1}</td>
                      <td title={a.description ?? undefined}>
                        <Link to={`/exp/${encodeURIComponent(a.exp_id)}`}>{a.exp_id}</Link>
                      </td>
                      <td className="num">
                        <strong>{a.avg_rank != null ? a.avg_rank.toFixed(2) : "—"}</strong>
                      </td>
                      <td className="num">
                        <span
                          className={`chip ${a.coverage_present < a.coverage_total ? "warn" : ""}`}
                          title="dataset columns where this experiment is present"
                        >
                          {a.coverage_present}/{a.coverage_total}
                        </span>
                      </td>
                      <td className="num">{fmt(a.mean_value)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="subtle" style={{ fontSize: "var(--fs-small)" }}>
              Showing top {Math.min(topN, d.aggregate.length)} of {d.aggregate.length} qualified. See{" "}
              <Link to="/rankings">Rankings</Link> for per-dataset tables, the full aggregate, dataset filters, and the
              line race.
            </div>
            {/* F-1 (P0-1): under-covered runs are OUT of the headline, surfaced collapsed. */}
            {(d.low_coverage?.length ?? 0) > 0 && (
              <details>
                <summary className="subtle" style={{ cursor: "pointer", fontSize: "var(--fs-small)" }}>
                  insufficient coverage ({d.low_coverage!.length}/{d.aggregate.length + d.low_coverage!.length}) — scored on
                  fewer than {d.min_coverage} of {d.coverage_total} dataset columns{d.min_coverage_auto ? " (auto)" : ""} ·
                  excluded from the headline
                </summary>
                <div className="tbl-wrap" style={{ maxHeight: 220, marginTop: 6 }}>
                  <table className="tbl tabular">
                    <tbody>
                      {d.low_coverage!.map((a) => (
                        <tr key={a.exp_id} style={{ opacity: 0.7 }}>
                          <td>
                            <Link to={`/exp/${encodeURIComponent(a.exp_id)}`}>{a.exp_id}</Link>
                          </td>
                          <td className="num">{a.avg_rank != null ? a.avg_rank.toFixed(2) : "—"}</td>
                          <td className="num">
                            <span className="chip warn">
                              {a.coverage_present}/{a.coverage_total}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </details>
            )}
          </div>
        );
      }}
    </AsyncPanel>
    </div>
  );
}
