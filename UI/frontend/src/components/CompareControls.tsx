/* CompareControls (2026-06-04 pin refactor) — the ONE control surface every
 * comparison view shares: pick MODELS, then pick a DATASET. Kept deliberately
 * small: two labelled rows, no jargon. Backs Compare / GIF Studio / Config diff so
 * the same models+dataset stay selected as you move between them.
 *
 *   Models:  [● 288·no_focal ✓ ×] [● 287·unmask ✓ ×]   + add model ▾
 *   Dataset: [ SWaT_A1A2 ▾ ]  ( full | excl22 )      3 datasets in common
 */
import { useMemo } from "react";
import { useExperiments } from "../api/queries";
import { useStore, catColor } from "../store";
import { hasRealVariants, shortModel, type CompareSelection } from "../scope";

export default function CompareControls({
  sel,
  minModels = 2,
}: {
  sel: CompareSelection;
  minModels?: number;
}) {
  const expsQ = useExperiments();
  const modelIndex = useStore((s) => s.modelIndex);
  const pinModel = useStore((s) => s.pinModel);
  const unpinModel = useStore((s) => s.unpinModel);

  const addable = useMemo(() => {
    const have = new Set(sel.candidates);
    return (expsQ.data ?? [])
      .filter((e) => !have.has(e.exp_id))
      .sort((a, b) => (b.exp_num ?? 0) - (a.exp_num ?? 0));
  }, [expsQ.data, sel.candidates]);

  const { options, datasetKey, variant, current, noCommon } = sel.scope;

  return (
    <div className="cmp-controls">
      {/* ── models ─────────────────────────────────────────── */}
      <div className="cmp-row">
        <span className="cmp-label">Models</span>
        <div className="row wrap" style={{ gap: 8, flex: 1 }}>
          {sel.candidates.length === 0 && (
            <span className="subtle" style={{ fontSize: "var(--fs-small)" }}>
              No pinned models — pin from Rankings / Overview, or add one →
            </span>
          )}
          {sel.candidates.map((id) => {
            const on = sel.isIncluded(id);
            return (
              <span
                key={id}
                className={`pin-chip cmp-model ${on ? "on" : "off"}`}
                title={on ? `${id}\n(click to exclude from this comparison)` : `${id}\n(click to include)`}
              >
                <button
                  className="cmp-toggle"
                  onClick={() => sel.toggleInclude(id)}
                  aria-pressed={on}
                  title={on ? "included — click to exclude" : "excluded — click to include"}
                >
                  <span className="swatch" style={{ background: on ? catColor(modelIndex(id)) : "var(--border)" }} />
                  <span className="mono" style={{ fontSize: "0.72rem", opacity: on ? 1 : 0.5 }}>
                    {shortModel(id)}
                  </span>
                  <span className="cmp-check">{on ? "✓" : "+"}</span>
                </button>
                <button className="cmp-x" onClick={() => unpinModel(id)} aria-label="unpin model" title="unpin">
                  ×
                </button>
              </span>
            );
          })}
          <select
            className="select sm"
            value=""
            onChange={(e) => {
              if (e.target.value) pinModel(e.target.value);
            }}
            title="add another model to compare"
          >
            <option value="">+ add model…</option>
            {addable.map((e) => (
              <option key={e.exp_id} value={e.exp_id}>
                {shortModel(e.exp_id)}
                {e.state === "in_progress" ? " (running)" : ""}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* ── dataset ────────────────────────────────────────── */}
      <div className="cmp-row">
        <span className="cmp-label">Dataset</span>
        <div className="row wrap" style={{ gap: 8, flex: 1, alignItems: "center" }}>
          {noCommon ? (
            <span className="chip warn" title="the selected models have no dataset in common">
              selected models share no dataset
            </span>
          ) : options.length === 0 ? (
            <span className="subtle" style={{ fontSize: "var(--fs-small)" }}>
              pick at least one model first
            </span>
          ) : (
            <>
              <select
                className="select sm"
                value={datasetKey ?? ""}
                onChange={(e) => sel.scope.setDatasetKey(e.target.value)}
                title="dataset to compare the selected models on"
              >
                {options.map((o) => (
                  <option key={o.dataset_key} value={o.dataset_key}>
                    {o.dataset_key}
                  </option>
                ))}
              </select>
              {current && hasRealVariants(current.variants) && (
                <span className="seg" role="group" aria-label="variant">
                  {current.variants
                    .filter((v) => v && v !== "_")
                    .map((v) => (
                      <button
                        key={v}
                        className={`seg-btn ${variant === v ? "active" : ""}`}
                        onClick={() => sel.scope.setVariant(v)}
                        title={v === "excl22" ? "exclude region-22 (canonical)" : v}
                      >
                        {v}
                      </button>
                    ))}
                </span>
              )}
              <span className="subtle" style={{ fontSize: "var(--fs-small)", marginLeft: "auto" }}>
                {options.length} dataset{options.length === 1 ? "" : "s"} in common · comparing{" "}
                <b>{sel.models.length}</b> model{sel.models.length === 1 ? "" : "s"}
              </span>
            </>
          )}
        </div>
      </div>

      {sel.models.length < minModels && sel.candidates.length > 0 && (
        <div className="cmp-hint subtle">
          Select at least {minModels} models to compare (click a chip to include it).
        </div>
      )}
    </div>
  );
}
