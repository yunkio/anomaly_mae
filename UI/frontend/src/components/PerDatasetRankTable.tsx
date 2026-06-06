/* <PerDatasetRankTable> (FB-12/FB-13) — one ranking table for ONE dataset column.
 * Experiments are ranked WITHIN this dataset by the selection metric (direction from
 * the backend). SWaT full and excl22 arrive as SEPARATE columns (FB-11), so each gets
 * its own table — there is no cross-dataset value mixing here (FB-12). Rows pin/click. */
import { useNavigate } from "react-router-dom";
import { fmt } from "./common";
import { useStore } from "../store";
import type { PerDatasetRankRow } from "../api/types";

interface Props {
  label: string;
  rows: PerDatasetRankRow[];
  metricLabel: string;
  rankable: boolean;
  /** show the pin column (pins the row's MODEL — dataset is chosen later in Compare). */
  showPin?: boolean;
}

export default function PerDatasetRankTable({ label, rows, metricLabel, rankable, showPin }: Props) {
  const nav = useNavigate();
  // Pins are MODELS now: the pin toggle adds/removes the row's experiment from the shared
  // set; you pick which dataset to compare on later (Compare/GIF/Config).
  const togglePin = useStore((s) => s.togglePinModel);
  const isModelPinned = useStore((s) => s.isModelPinned);
  // subscribe to pins so the toggle re-renders when the store changes.
  useStore((s) => s.pins);

  return (
    <div className="stat-card" style={{ overflow: "hidden" }}>
      <div className="card-title" style={{ fontSize: "var(--fs-body)", marginBottom: 6 }}>
        {label} <span className="subtle">({rows.length})</span>
      </div>
      <div className="tbl-wrap" style={{ maxHeight: 320 }}>
        <table className="tbl tabular" style={{ fontSize: "0.82rem" }}>
          <thead>
            <tr>
              <th>#</th>
              <th>experiment</th>
              <th className="num">{metricLabel}</th>
              {showPin && <th />}
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={r.exp_id}>
                <td>{rankable ? r.rank ?? i + 1 : "—"}</td>
                <td>
                  <a
                    onClick={() => nav(`/exp/${encodeURIComponent(r.exp_id)}`)}
                    style={{ cursor: "pointer" }}
                    title={r.description ?? undefined}
                  >
                    {r.exp_id}
                  </a>
                </td>
                <td className="num">
                  <strong>{fmt(r.value)}</strong>
                </td>
                {showPin && (() => {
                  const pinned = isModelPinned(r.exp_id);
                  return (
                    <td>
                      <button
                        className={`btn sm ${pinned ? "primary" : ""}`}
                        onClick={() => togglePin(r.exp_id)}
                        title={pinned ? "unpin this model" : "pin this model (pick a dataset to compare on in Compare/GIF/Config)"}
                      >
                        {pinned ? "pinned ✓" : "+ pin"}
                      </button>
                    </td>
                  );
                })()}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
