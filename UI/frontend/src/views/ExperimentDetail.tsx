/* Experiment Detail (IA-2) — header card (best pak_auc_f1, state, score-formula,
 * caveats), dataset sub-tabs -> Dataset detail, and the run-health panel (P2-02).
 * early_abort -> header + health only; in_progress -> live tail. */
import { useNavigate, useParams } from "react-router-dom";
import { useExperiment } from "../api/queries";
import AsyncPanel from "../components/AsyncPanel";
import { Card, StateChip, fmt } from "../components/common";
import ScoreFormulaCard from "../components/ScoreFormulaCard";
import RunHealthTrend from "../components/RunHealthTrend";
import { dsUrl } from "../api/client";
import { useStore } from "../store";
import type { ExperimentDetail as ExpDetail } from "../api/types";

export default function ExperimentDetail() {
  const { expId } = useParams();
  const nav = useNavigate();
  const q = useExperiment(expId);
  const togglePin = useStore((s) => s.togglePinModel);
  const isModelPinned = useStore((s) => s.isModelPinned);
  useStore((s) => s.pins); // re-render on pin changes

  return (
    <div className="col" style={{ gap: 24 }}>
      <AsyncPanel query={q} emptyLabel="Experiment not found.">
        {(e: ExpDetail) => (
          <div className="col" style={{ gap: 24 }}>
            <div>
              <div className="row wrap" style={{ gap: 12, alignItems: "center" }}>
                <h1 className="view-title" style={{ margin: 0 }}>{e.exp_id}</h1>
                <button
                  className={`btn sm ${isModelPinned(e.exp_id) ? "primary" : ""}`}
                  onClick={() => togglePin(e.exp_id)}
                  title={isModelPinned(e.exp_id) ? "unpin this model" : "pin this model (compare it on any dataset in Compare)"}
                >
                  {isModelPinned(e.exp_id) ? "model pinned ✓" : "+ pin model"}
                </button>
              </div>
              <div className="row wrap" style={{ gap: 10, marginTop: 8 }}>
                <StateChip state={e.state} source={e.source} />
                {e.summary?.config_set && <span className="chip">{e.summary.config_set}</span>}
                {e.summary?.total_time != null && <span className="chip">wall {fmt(e.summary.total_time, 1)}s</span>}
                {(e.caveats ?? []).map((c) => (
                  <span className="chip warn" key={c}>
                    {c}
                  </span>
                ))}
              </div>
              {e.summary?.description && <p className="view-sub" style={{ marginTop: 8 }}>{e.summary.description}</p>}
            </div>

            {e.score_formula && (
              <Card title="Resolved score formula (CFG-4)">
                <ScoreFormulaCard formula={e.score_formula} />
              </Card>
            )}

            {e.state !== "early_abort" && (
              <Card title="Datasets">
                {(e.datasets ?? []).length === 0 ? (
                  <div className="subtle">No leaf datasets.</div>
                ) : (
                  <div className="card-grid">
                    {e.datasets.map((ds) => (
                      <div className="stat-card" key={ds.dataset_key}>
                        <div className="row" style={{ justifyContent: "space-between" }}>
                          <strong>{ds.dataset_key}</strong>
                          <StateChip state={ds.state} />
                        </div>
                        <div className="row wrap" style={{ gap: 6, marginTop: 6 }}>
                          {ds.variants.map((v) => (
                            <span className="chip" key={v}>
                              {v}
                            </span>
                          ))}
                        </div>
                        {ds.best?.pak_auc_f1 != null && (
                          <div className="subtle" style={{ fontSize: "var(--fs-small)", marginTop: 6 }}>
                            best pak_auc_f1 <strong className="tabular">{fmt(ds.best.pak_auc_f1)}</strong>
                          </div>
                        )}
                        <div className="row" style={{ gap: 8, marginTop: 10 }}>
                          <button
                            className="btn sm primary"
                            onClick={() => nav(`/exp/${encodeURIComponent(e.exp_id)}/ds/${dsUrl(ds.dataset_key)}`)}
                          >
                            open →
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </Card>
            )}

            <Card title="Run health (P2-02 · monitoring_log.jsonl)">
              <RunHealthTrend expId={e.exp_id} />
            </Card>
          </div>
        )}
      </AsyncPanel>
    </div>
  );
}
