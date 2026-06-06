/* <HardestWindowsPanel> (F-8 · V-5 — "where it fails").
 *
 * The worst-scoring windows from GET …/detailed-windows (best_model_detailed.csv), sorted
 * DESCENDING by a chosen numeric score column (default a recon/total-loss-like column when
 * present). The backend returns is_sample / sample_size — this table BADGES "~N-window
 * SAMPLE" so the user never reads it as the full test length (the F-8 acceptance criterion).
 *
 * registry-direction-aware in spirit: "hardest" = the highest anomaly SCORE rows; the sort
 * column is user-selectable from the numeric columns the CSV actually carries (no hard-coded
 * column name — F5). Hooks live in this component body (F-02). */
import { useMemo, useState } from "react";
import AsyncPanel from "../AsyncPanel";
import { Card, fmt } from "../common";
import { useDetailedWindows } from "../../api/queries";
import type { DetailedWindows } from "../../api/types";

interface Props {
  exp: string;
  dsKey: string;
  variant: string | null;
}

const TOP = 25;

// columns that look like a per-window anomaly SCORE (higher = harder), preferred for the
// default "hardest" sort. Pure-string heuristics over the runtime columns — never a
// hard-coded full list.
const SCORE_HINTS = ["total_loss", "adaptive", "score", "discrepancy_loss", "reconstruction_loss", "loss"];

export default function HardestWindowsPanel({ exp, dsKey, variant }: Props) {
  // probe (unsorted) first to learn the columns, then drive a sort selector.
  const probe = useDetailedWindows(exp, dsKey, variant, { top: 1 });
  const cols = probe.data?.columns ?? [];
  // numeric columns from the probe row (exclude obvious label/category columns).
  const numericCols = useMemo(() => {
    const row = probe.data?.rows?.[0];
    if (!row) return cols;
    return cols.filter((c) => {
      const v = row[c];
      return typeof v === "number" && Number.isFinite(v) && !/^label$|anomaly_type$/.test(c);
    });
  }, [probe.data, cols]);
  const defaultSort = useMemo(() => {
    for (const h of SCORE_HINTS) {
      const hit = numericCols.find((c) => c.toLowerCase().includes(h));
      if (hit) return hit;
    }
    return numericCols[0];
  }, [numericCols]);
  const [sort, setSort] = useState<string | undefined>(undefined);
  const activeSort = sort ?? defaultSort;

  const q = useDetailedWindows(exp, dsKey, variant, { sort: activeSort, top: TOP });

  const selector = numericCols.length > 0 && (
    <label className="row" style={{ gap: 6 }} title="sort the windows DESCENDING by this score column (hardest = highest)">
      sort by
      <select className="select" value={activeSort ?? ""} onChange={(e) => setSort(e.target.value)} style={{ width: 200 }}>
        {numericCols.map((c) => (
          <option key={c} value={c}>
            {c}
          </option>
        ))}
      </select>
    </label>
  );

  return (
    <Card title="F-8 · Hardest windows (where it fails)" actions={selector}>
      <AsyncPanel
        query={q}
        softError={(d: DetailedWindows) => (!d.available && d.error ? `detailed windows error: ${d.error}` : null)}
        isEmpty={(d: DetailedWindows) => !d.available || !(d.rows && d.rows.length)}
        emptyLabel="No best_model_detailed.csv for this leaf (per-window detail unavailable)."
        height={220}
      >
        {(d) => <WindowsTable d={d} sort={activeSort} />}
      </AsyncPanel>
    </Card>
  );
}

function WindowsTable({ d, sort }: { d: DetailedWindows; sort?: string }) {
  const cols = d.columns ?? [];
  const rows = d.rows ?? [];
  return (
    <div className="col" style={{ gap: 10 }}>
      <div className="row wrap" style={{ gap: 8, alignItems: "center" }}>
        {/* the SAMPLE badge — the F-8 acceptance criterion: do NOT imply full test length. */}
        {d.is_sample && (
          <span
            className="chip warn"
            title="these are a SAMPLE of windows from best_model_detailed.csv — NOT the full test length; sorted to surface the hardest rows"
          >
            ~{(d.sample_size ?? rows.length).toLocaleString()}-window SAMPLE
          </span>
        )}
        {sort && (
          <span className="chip mono" title="worst-first by this column">
            hardest by {sort} ↓
          </span>
        )}
        <span className="chip" title="rows shown">
          top {rows.length}
        </span>
      </div>

      <div className="tbl-wrap" style={{ maxHeight: 420 }}>
        <table className="tbl tabular" style={{ fontSize: "0.82rem" }}>
          <thead>
            <tr>
              <th className="num">#</th>
              {cols.map((c) => (
                <th key={c}>
                  {c}
                  {c === sort ? " ↓" : ""}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i}>
                <td className="num subtle">{i + 1}</td>
                {cols.map((c) => {
                  const v = r[c];
                  const isNum = typeof v === "number";
                  return (
                    <td key={c} className={isNum ? "num" : ""}>
                      {isNum ? fmt(v as number) : v == null ? "—" : String(v)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="subtle" style={{ fontSize: "var(--fs-small)", whiteSpace: "normal", lineHeight: 1.45 }}>
        The worst-scoring windows from <span className="mono">best_model_detailed.csv</span>, sorted descending by{" "}
        <span className="mono">{sort ?? "the chosen column"}</span>. These rows are the model's hardest cases — scan{" "}
        <span className="mono">label</span> / <span className="mono">anomaly_type</span> to see what it gets wrong. This is a{" "}
        <strong>sample</strong>, not the full test stream.
      </div>
    </div>
  );
}
