/* <DatasetSetSelector> (2026-06-04) — choose WHICH dataset columns an average-rank
 * aggregate is computed over, mirroring the Rankings "Dataset set" control. Backend-
 * driven: the universe + default set + per-column KIND come from /api/rankings/aggregate
 * (selectable_columns / default_columns / column_meta). `value=null` means "the canonical
 * default". Rendered as a compact collapsible so it fits the Overview without dominating. */
import { useMemo } from "react";

const KIND_LABEL: Record<string, string> = {
  canonical_avg: "Multi-entity (avg)",
  swat_excl22: "SWaT (excl22)",
  single_leaf: "Single-leaf datasets",
  swat_full: "SWaT (full)",
  entity_leaf: "Individual machines / channels / apps",
  concat: "Concatenated leaves",
};
const KIND_ORDER = ["canonical_avg", "swat_excl22", "single_leaf", "swat_full", "entity_leaf", "concat"];

interface Props {
  universe: string[];
  defaultCols: string[];
  columnMeta: Record<string, { kind?: string }>;
  /** null = canonical default; an explicit list once the user composes a set. */
  value: string[] | null;
  onChange: (next: string[] | null) => void;
  /** start expanded (default false → a collapsed summary line). */
  defaultOpen?: boolean;
}

export default function DatasetSetSelector({ universe, defaultCols, columnMeta, value, onChange, defaultOpen }: Props) {
  const activeSet = value == null ? defaultCols : value;
  const isCanonicalDefault = value == null;

  const grouped = useMemo(() => {
    const byKind: Record<string, string[]> = {};
    for (const col of universe) {
      const kind = columnMeta[col]?.kind ?? "single_leaf";
      (byKind[kind] ??= []).push(col);
    }
    return KIND_ORDER.filter((k) => byKind[k]?.length).map((k) => [k, byKind[k].slice().sort()] as const);
  }, [universe, columnMeta]);

  function toggle(col: string) {
    const base = value == null ? defaultCols : value;
    onChange(base.includes(col) ? base.filter((c) => c !== col) : [...base, col]);
  }

  if (universe.length === 0) return null;

  return (
    <details open={defaultOpen} className="dataset-set-selector">
      <summary className="subtle" style={{ cursor: "pointer", fontSize: "var(--fs-small)" }}>
        averaging over <strong>{activeSet.length}</strong> of {universe.length} dataset columns
        {isCanonicalDefault ? " (canonical default)" : " (custom set)"} — click to choose
      </summary>
      <div className="col" style={{ gap: 10, marginTop: 8 }}>
        <div className="row wrap" style={{ gap: 6, alignItems: "center" }}>
          <button
            className={`btn sm ${isCanonicalDefault ? "primary" : ""}`}
            onClick={() => onChange(null)}
            title="reset to the canonical set (PSM, SWaT excl22, WaDi A1/A2, SMD/SMAP/MSL/Exathlon (avg))"
          >
            canonical default
          </button>
          <button className="btn sm" onClick={() => onChange(universe.slice())} title="select every selectable column">
            select all
          </button>
        </div>
        {grouped.map(([kind, cols]) => (
          <div key={kind} className="col" style={{ gap: 4 }}>
            <div className="subtle" style={{ fontSize: "var(--fs-small)", fontWeight: 600 }}>
              {KIND_LABEL[kind] ?? kind}
              {(kind === "canonical_avg" || kind === "swat_excl22" || kind === "single_leaf") && (
                <span style={{ fontWeight: 400 }}> · in the canonical default</span>
              )}
            </div>
            <div className="row wrap" style={{ gap: 6 }}>
              {cols.map((col) => {
                const on = activeSet.includes(col);
                return (
                  <button key={col} className={`btn sm ${on ? "primary" : ""}`} onClick={() => toggle(col)} title={col}>
                    {col.replace("SWaT_A1A2 · ", "SWaT·")}
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </details>
  );
}
