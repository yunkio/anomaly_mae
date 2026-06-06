/* <Leaderboard> — rank by ANY metric (RANK-1/2/3/4). Default pak_auc_f1; sort
 * direction from the backend's registry `direction` (↓ ⇒ "lower wins"); neutral
 * metrics excluded from the default rank with a "no inherent better" note. Pin/click
 * rows. Client-side sort/filter/search (<200 ms).
 *
 * R2-02: this is the SHARED compact table — used by BOTH Rankings AND Overview. FB-11
 * removed the internal SWaT pill-TOGGLE; the `swat` variant is now an EXPLICIT PROP
 * (default "full") so the parent decides the single variant. Rankings' both-columns
 * view lives in the NEW /rankings/aggregate surface, not here — so Overview's compact
 * table is NOT broken (no mode toggle, no cross-dataset both-rows mutation). */
import { useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useLeaderboard } from "../api/queries";
import { dsUrl } from "../api/client";
import AsyncPanel from "./AsyncPanel";
import { StateChip, fmt, DirectionBadge } from "./common";
import { useStore } from "../store";
import type { LeaderboardRow } from "../api/types";

interface Props {
  metric?: string;
  scope?: string;
  groupBy?: string;
  compact?: boolean;
  /** R2-02: the single SWaT eval variant this table shows (default "full"). The
   * both-variants view is /rankings/aggregate (FB-11), never an in-table toggle. */
  swat?: "full" | "excl22";
}

export default function Leaderboard({ metric = "pak_auc_f1", scope, groupBy, compact, swat = "full" }: Props) {
  const nav = useNavigate();
  const [search, setSearch] = useState("");
  const q = useLeaderboard({ metric, scope, group_by: groupBy, swat });
  const togglePin = useStore((s) => s.togglePinModel);
  const isModelPinned = useStore((s) => s.isModelPinned);
  useStore((s) => s.pins); // re-render on pin changes

  // F-02: filtered/sorted rows computed at the TOP LEVEL from query.data (null-guarded),
  // never as a hook inside the AsyncPanel render-prop (whose hook count changes across
  // loading/error/empty/success states → Rules-of-Hooks crash, AC-F3.1).
  const rows = useMemo(() => {
    const d = q.data;
    if (!d?.rows) return [] as LeaderboardRow[];
    const s = search.toLowerCase();
    const r = d.rows.filter(
      (row) =>
        !s || row.exp_id.toLowerCase().includes(s) || row.dataset_key.toLowerCase().includes(s)
    );
    const asc = d.direction === "down";
    return [...r].sort((a, b) => {
      const av = a.value ?? (asc ? Infinity : -Infinity);
      const bv = b.value ?? (asc ? Infinity : -Infinity);
      return asc ? av - bv : bv - av;
    });
  }, [q.data, search]);

  return (
    <AsyncPanel
      query={q}
      isEmpty={(d) => !d.rows || d.rows.length === 0}
      emptyLabel="No ranked rows for this metric/scope."
      height={260}
    >
      {(d) => {
        const dir = d.direction;
        return (
          <div>
            {/* F-1 (V-3): this is a FLAT per-leaf table — a single metric value pooled
                across heterogeneous datasets. Those values are NOT directly comparable
                across datasets; the fair cross-dataset ranking is the average-rank
                aggregate. Caveat chip + a jump link to it. */}
            <div className="chip warn" style={{ whiteSpace: "normal", lineHeight: 1.4, marginBottom: 8 }}>
              values across different datasets are not directly comparable → use the{" "}
              <Link to="/rankings" style={{ textDecoration: "underline" }}>
                average-rank aggregate
              </Link>{" "}
              for a fair cross-dataset model ranking.
            </div>
            <div className="toolbar">
              <span className="row" style={{ gap: 6 }}>
                <strong>{d.meta?.display_name ?? d.metric}</strong>
                <DirectionBadge direction={dir} />
                <span className="subtle">{dir === "down" ? "lower wins" : dir === "up" ? "higher wins" : "no inherent better"}</span>
              </span>
              {d.neutral_note && <span className="chip warn">{d.neutral_note}</span>}
              <span className="subtle" style={{ fontSize: "var(--fs-small)" }} title="SWaT eval variant for this table (both-columns view is in Rankings)">
                SWaT: {swat}
              </span>
              <span className="grow" style={{ flex: 1 }} />
              <input
                className="input"
                placeholder="search…"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                style={{ width: 160 }}
              />
            </div>
            <div className="tbl-wrap" style={{ maxHeight: compact ? 360 : 600 }}>
              <table className="tbl tabular">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>experiment</th>
                    <th>dataset</th>
                    <th className="num">{d.meta?.display_name ?? d.metric}</th>
                    <th className="num">best epoch</th>
                    <th>state</th>
                    <th>pin</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row: LeaderboardRow, i) => {
                    const pinned = isModelPinned(row.exp_id);
                    return (
                      <tr key={row.exp_id + "|" + row.dataset_key + i}>
                        <td>{i + 1}</td>
                        <td>
                          <a onClick={() => nav(`/exp/${encodeURIComponent(row.exp_id)}`)} style={{ cursor: "pointer" }}>
                            {row.exp_id}
                          </a>
                        </td>
                        <td>
                          <a
                            onClick={() =>
                              nav(`/exp/${encodeURIComponent(row.exp_id)}/ds/${dsUrl(row.dataset_key)}`)
                            }
                            style={{ cursor: "pointer" }}
                          >
                            {row.dataset_key}
                            {row.variant ? ` · ${row.variant}` : ""}
                          </a>
                        </td>
                        <td className="num">
                          <strong>{fmt(row.value)}</strong>
                        </td>
                        <td className="num">{row.best_epoch ?? "—"}</td>
                        <td>
                          <StateChip state={row.state} source={row.source} />
                        </td>
                        <td>
                          <button
                            className={`btn sm ${pinned ? "primary" : ""}`}
                            onClick={() => togglePin(row.exp_id)}
                            title={pinned ? "unpin this model" : "pin this model (pick a dataset in Compare)"}
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
          </div>
        );
      }}
    </AsyncPanel>
  );
}
