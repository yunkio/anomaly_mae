/* Overview (IA-1) — "what do I have and what is winning."
 * Summary-stat cards + the experiment table + the TRUE FB-13 average-rank model summary
 * (<OverviewTopModels>, sourced from /api/rankings/aggregate). RV3-01: the Overview no
 * longer renders the shared <Leaderboard> compact table, which pooled values across
 * datasets into one ranked column (the FB-12 cross-dataset-mix anti-pattern). The
 * average-rank summary ranks each model WITHIN each dataset, then averages the ranks, so
 * no single column ever mixes datasets. Empty -> onboarding card. */
import { useNavigate } from "react-router-dom";
import { useExperiments } from "../api/queries";
import AsyncPanel from "../components/AsyncPanel";
import { Card, StateChip, fmt, EmptyOnboarding } from "../components/common";
import OverviewTopModels from "../components/OverviewTopModels";
import { useStore } from "../store";
import type { ExperimentRef } from "../api/types";

export default function Overview() {
  const nav = useNavigate();
  const q = useExperiments();
  const togglePin = useStore((s) => s.togglePinModel);
  const isModelPinned = useStore((s) => s.isModelPinned);
  useStore((s) => s.pins); // re-render on pin changes

  return (
    <div className="col" style={{ gap: 24 }}>
      <div>
        <h1 className="view-title">Overview</h1>
        <p className="view-sub">
          Auto-discovered experiments and leaf datasets from <span className="mono">results/experiments/</span>{" "}
          (+ read-only fixtures). Ranked by the primary metric <span className="mono">pak_auc_f1</span>.
        </p>
      </div>

      <AsyncPanel query={q} isEmpty={(d) => d.length === 0} emptyLabel="No experiments discovered.">
        {(exps: ExperimentRef[]) => {
          if (exps.length === 0) return <EmptyOnboarding message="No experiments found." />;
          // pure function call — NOT a hook (F-02: no hooks inside a render-prop / after
          // an early return; this is cheap and runs only on the success render).
          const roll = summarize(exps);
          return (
            <div className="col" style={{ gap: 24 }}>
              <div className="stat-grid">
                <StatCard label="Experiments" value={String(roll.total)} />
                <StatCard label="Complete" value={String(roll.complete)} />
                <StatCard label="In progress" value={String(roll.inProgress)} />
                <StatCard label="Early-abort" value={String(roll.abort)} />
                <StatCard label="Datasets (leaves)" value={String(roll.datasets)} />
                <StatCard label="Best pak_auc_f1" value={fmt(roll.bestPak)} />
              </div>

              <Card title="Experiments">
                <div className="tbl-wrap" style={{ maxHeight: 360 }}>
                  <table className="tbl tabular">
                    <thead>
                      <tr>
                        <th>experiment</th>
                        <th>state</th>
                        <th className="num">datasets</th>
                        <th className="num">best pak_auc_f1</th>
                        <th>source</th>
                        <th>pin</th>
                      </tr>
                    </thead>
                    <tbody>
                      {exps.map((e) => {
                        const pinned = isModelPinned(e.exp_id);
                        return (
                        <tr key={e.exp_id}>
                          <td>
                            <a style={{ cursor: "pointer" }} onClick={() => nav(`/exp/${encodeURIComponent(e.exp_id)}`)}>
                              {e.exp_id}
                            </a>
                          </td>
                          <td>
                            <StateChip state={e.state} source={e.source} />
                          </td>
                          <td className="num">{e.datasets?.length ?? 0}</td>
                          <td className="num">{fmt(e.best_pak_auc_f1 ?? bestOf(e))}</td>
                          <td>
                            <span className={`chip ${e.source === "fixture" ? "fixture" : ""}`}>{e.source}</span>
                          </td>
                          <td>
                            <button
                              className={`btn sm ${pinned ? "primary" : ""}`}
                              onClick={() => togglePin(e.exp_id)}
                              title={pinned ? "unpin this model" : "pin this model (pick a dataset to compare on in Compare)"}
                            >
                              {pinned ? "pinned ✓" : "+ pin"}
                            </button>
                          </td>
                        </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </Card>

              {/* RV3-01 / FB-12: the Overview summary is the TRUE FB-13 average-rank model
                  ranking (per-dataset rank → averaged), NOT a cross-dataset value mix. */}
              <Card title="Top models (avg rank across datasets)">
                <OverviewTopModels metric="pak_auc_f1" topN={8} />
              </Card>
            </div>
          );
        }}
      </AsyncPanel>
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="stat-card">
      <div className="label">{label}</div>
      <div className="value tabular">{value}</div>
    </div>
  );
}

function bestOf(e: ExperimentRef): number | null {
  return (e.summary as any)?.best_pak_auc_f1 ?? null;
}

function summarize(exps: ExperimentRef[]) {
  let complete = 0,
    inProgress = 0,
    abort = 0,
    datasets = 0,
    bestPak: number | null = null;
  for (const e of exps) {
    if (e.state === "complete") complete++;
    else if (e.state === "in_progress") inProgress++;
    else if (e.state === "early_abort") abort++;
    datasets += e.datasets?.length ?? 0;
    const b = e.best_pak_auc_f1 ?? bestOf(e);
    if (typeof b === "number" && (bestPak === null || b > bestPak)) bestPak = b;
  }
  return { total: exps.length, complete, inProgress, abort, datasets, bestPak };
}
