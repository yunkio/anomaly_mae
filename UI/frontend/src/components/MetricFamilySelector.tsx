/* <MetricFamilySelector> — the dynamic metric picker (TREND-2, F5). Options come
 * from the runtime UNION catalog (GET /metrics-catalog) — NEVER a fixed list.
 *
 * FB-1 / FB-5 / FB-6 (R2): the chip list is the SINGLE catalog VIEW `metricCatalogView()`
 * (api/metricView.ts), which composes excl22-strip → pa-keep → variant-suffix → de-dup
 * by identity=(display_label, variant_scope) in one place. So:
 *   • the 4 score-variant namespaces become a `· teacher`/`· student`/`· disc` SUFFIX
 *     facet, not N identical chips (FB-1);
 *   • only the 8 KEEP_PA `pa_*` chips show, the other ~100 sweep keys hidden (FB-5);
 *   • excl22 is conveyed by variant_scope, no full/excl22 collision (FB-6);
 *   • the family-header (N) counts the DE-DUPED set;
 *   • RV2-05: a click emits the NON-prefixed canonical key.
 * The raw catalog is NEVER dropped — a "show raw keys" toggle re-exposes every entry
 * (F5: an unknown metric still gets a chip). */
import { useMemo, useState } from "react";
import type { CatalogEntry } from "../api/types";
import { metricCatalogView, groupByFamily, type MetricChip } from "../api/metricView";

interface Props {
  catalog: CatalogEntry[];
  selected: string[];
  onChange: (keys: string[]) => void;
  multi?: boolean;
  /** restrict to these namespaces (default: all). Applied BEFORE the de-dup view so a
   * leaf-scoped picker (e.g. epoch_metrics only) still de-dups within that scope. */
  namespaceFilter?: string[];
  /** restrict to these families (e.g. a GIF story's allowed families) */
  familyFilter?: string[];
  /** the SWaT variant of the single leaf this catalog came from ("full"/"excl22"); when
   * set, the OTHER variant's metrics are hidden (2026-06-04: SWaT full no longer shows
   * ·excl22 chips). Leave undefined for catalog-wide pickers. */
  leafVariant?: string | null;
}

export default function MetricFamilySelector({
  catalog,
  selected,
  onChange,
  multi = true,
  namespaceFilter,
  familyFilter,
  leafVariant,
}: Props) {
  const [q, setQ] = useState("");
  const [ns, setNs] = useState<string>("__all__");
  const [showRaw, setShowRaw] = useState(false);

  const namespaces = useMemo(() => {
    const s = new Set<string>();
    catalog.forEach((c) => c.namespace && s.add(c.namespace));
    return Array.from(s).sort();
  }, [catalog]);

  // Apply namespace scoping FIRST (so the de-dup view collapses within the requested
  // scope), then run the single metricCatalogView() chokepoint (FB-1/5/6 + RV2-05).
  const chips = useMemo(() => {
    let base = catalog;
    if (namespaceFilter) base = base.filter((c) => c.namespace && namespaceFilter.includes(c.namespace));
    if (ns !== "__all__") base = base.filter((c) => c.namespace === ns);
    return metricCatalogView(base, { showRaw, leafVariant });
  }, [catalog, namespaceFilter, ns, showRaw, leafVariant]);

  const grouped = useMemo(() => {
    const ql = q.toLowerCase();
    const filtered = chips.filter((c) => {
      if (familyFilter && !familyFilter.some((f) => (c.family || "").startsWith(f))) return false;
      if (
        ql &&
        !(
          c.key.toLowerCase().includes(ql) ||
          c.display_label.toLowerCase().includes(ql) ||
          (c.display_name || "").toLowerCase().includes(ql)
        )
      )
        return false;
      return true;
    });
    return groupByFamily(filtered);
  }, [chips, q, familyFilter]);

  function toggle(key: string) {
    if (!multi) {
      onChange([key]);
      return;
    }
    onChange(selected.includes(key) ? selected.filter((k) => k !== key) : [...selected, key]);
  }

  function chipLabel(c: MetricChip): string {
    let label = c.display_label || c.key;
    // a chip that collapsed >1 score-variant facet shows the available facets (FB-1).
    if (showRaw && c.namespace) label += ` [${c.namespace}]`;
    return label;
  }

  return (
    <div className="col" style={{ gap: 8 }}>
      <div className="row wrap" style={{ gap: 8 }}>
        <input
          className="input"
          placeholder="Filter metrics…"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          style={{ flex: 1, minWidth: 160 }}
        />
        {namespaces.length > 1 && !namespaceFilter && (
          <select className="select" value={ns} onChange={(e) => setNs(e.target.value)}>
            <option value="__all__">all namespaces</option>
            {namespaces.map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
        )}
        {/* F5 debug escape — re-expose every raw (key,namespace,scope) entry. */}
        <label className="row subtle" style={{ gap: 4, fontSize: "var(--fs-small)" }} title="Show every raw catalog key (no de-dup / pa-keep / suffix) — F5 escape">
          <input type="checkbox" checked={showRaw} onChange={(e) => setShowRaw(e.target.checked)} /> raw
        </label>
      </div>
      <div
        style={{
          maxHeight: 320,
          overflowY: "auto",
          border: "1px solid var(--border)",
          borderRadius: "var(--radius-md)",
          padding: "var(--space-2)",
          background: "var(--surface)",
        }}
      >
        {grouped.length === 0 && <div className="subtle" style={{ padding: 8 }}>No metrics match.</div>}
        {grouped.map(([fam, items]) => (
          <div key={fam} style={{ marginBottom: 8 }}>
            <div className="subtle" style={{ fontSize: "var(--fs-small)", margin: "4px 2px", fontWeight: 600 }}>
              {fam} <span className="subtle">({items.length})</span>
            </div>
            <div className="row wrap" style={{ gap: 6 }}>
              {items.map((c) => {
                const on = selected.includes(c.key);
                const facets =
                  c.score_facets.length > 1
                    ? ` (${c.score_facets.join("/")})`
                    : "";
                return (
                  <button
                    key={`${c.variant_scope ?? "default"}:${c.namespace}:${c.key}`}
                    className={`btn sm ${on ? "primary" : ""}`}
                    title={`${c.display_label}${facets} · ${c.direction}${
                      c.variant_scope && c.variant_scope !== "default" ? ` · ${c.variant_scope}` : ""
                    }${c.collapsed > 1 ? ` · ${c.collapsed} variants collapsed` : ""}${
                      c.description ? " — " + c.description : ""
                    }`}
                    onClick={() => toggle(c.key)}
                  >
                    {chipLabel(c)}
                    {c.direction === "up" ? " ↑" : c.direction === "down" ? " ↓" : ""}
                    {c.variant_scope === "excl22" && !showRaw ? " ·excl22" : ""}
                    {c.inferred && <span style={{ marginLeft: 4, opacity: 0.8 }}>· inf</span>}
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
