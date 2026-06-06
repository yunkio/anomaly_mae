/* Config Inspector + Cross-Experiment Diff (IA-5, 2026-06-04 pin refactor).
 * • <ConfigInspector>: grouped, searchable config of one MODEL (on the scope dataset) +
 *   the resolved score-formula card (never blank, CFG-4).
 * • <ConfigDiff>: aligned diff over the UNION of keys across two pinned MODELS on the
 *   shared dataset; a key in one run but absent in another shows as added/removed. */
import { useMemo, useState } from "react";
import { useConfig, useConfigDiff } from "../api/queries";
import { Card } from "../components/common";
import AsyncPanel from "../components/AsyncPanel";
import ScoreFormulaCard from "../components/ScoreFormulaCard";
import { computeConfigActivity } from "../api/configActivity";
import { useStore, catColor } from "../store";
import { useScopeDataset, shortModel } from "../scope";
import type { LeafSel } from "../api/types";

export default function ConfigView() {
  const pins = useStore((s) => s.pins);
  const scope = useScopeDataset(pins);
  const [sel, setSel] = useState(0);
  const model = pins[Math.min(sel, Math.max(0, pins.length - 1))];
  const cfg = useConfig(model, scope.datasetKey ?? undefined, scope.variant);
  // explicit LEFT + RIGHT MODEL selectors (from the pinned set) drive the two diff
  // columns. Defaults: 0 and min(1,n-1). The dataset is the shared scope.
  const [leftIdx, setLeftIdx] = useState(0);
  const [rightIdx, setRightIdx] = useState(1);
  const leftSel = Math.min(leftIdx, Math.max(0, pins.length - 1));
  const rightSel = Math.min(rightIdx, Math.max(0, pins.length - 1));
  const leaves = useMemo<LeafSel[]>(() => {
    if (pins.length === 0 || !scope.datasetKey) return [];
    const toLeaf = (m: string): LeafSel => ({ exp_id: m, dataset_key: scope.datasetKey as string, variant: scope.variant });
    const l = pins[leftSel];
    const r = pins[rightSel];
    if (leftSel === rightSel) return [toLeaf(l)];
    return [toLeaf(l), toLeaf(r)];
  }, [pins, leftSel, rightSel, scope.datasetKey, scope.variant]);
  const diff = useConfigDiff(leaves);
  const [search, setSearch] = useState("");
  const [onlyDiff, setOnlyDiff] = useState(true);
  // FB-8: show only ACTIVE config by default; the toggle reveals inactive keys.
  const [showInactive, setShowInactive] = useState(false);

  // FB-8: compute the inactive-key set at the TOP LEVEL from query.data (null-guarded),
  // never as a hook inside the AsyncPanel render-prop (hook count must be state-invariant
  // → Rules-of-Hooks, AC-F3.1). Grounded in the flag→dependent-key map (config.py).
  const activity = useMemo(() => computeConfigActivity(cfg.data?.config), [cfg.data]);

  // F-02: filtered diff keys computed at the TOP LEVEL from query.data (null-guarded).
  const diffKeys = useMemo(() => {
    const ks = diff.data?.keys ?? [];
    return onlyDiff ? ks.filter((k) => k.differs) : ks;
  }, [diff.data, onlyDiff]);

  if (pins.length === 0)
    return (
      <div>
        <h1 className="view-title">Config Inspector + Diff</h1>
        <p className="view-sub">Pin models to inspect and diff their configs.</p>
        <div className="async-state" style={{ minHeight: 200 }}>
          <span className="glyph">⚙</span>
          <div>No pinned models.</div>
        </div>
      </div>
    );

  const DatasetPicker = scope.options.length > 0 && (
    <>
      <label className="row" style={{ gap: 6 }}>
        dataset
        <select className="select" value={scope.datasetKey ?? ""} onChange={(e) => scope.setDatasetKey(e.target.value)}>
          {scope.options.map((o) => (
            <option key={o.dataset_key} value={o.dataset_key}>
              {o.dataset_key}
            </option>
          ))}
        </select>
      </label>
      {scope.current && scope.current.variants.filter((v) => v && v !== "_").length > 1 && (
        <span className="seg" role="group" aria-label="variant">
          {scope.current.variants
            .filter((v) => v && v !== "_")
            .map((v) => (
              <button key={v} className={`seg-btn ${scope.variant === v ? "active" : ""}`} onClick={() => scope.setVariant(v)}>
                {v}
              </button>
            ))}
        </span>
      )}
    </>
  );

  return (
    <div className="col" style={{ gap: 24 }}>
      <div>
        <h1 className="view-title">Config Inspector + Diff</h1>
        <p className="view-sub">
          {pins.length} pinned model(s){scope.noCommon ? " — but they share no dataset" : scope.datasetKey ? ` · configs on ${scope.datasetKey}` : ""}. Diff
          is a UNION over all keys — an ablation flag present in one run shows as added/removed.
        </p>
        <div className="toolbar">
          <label className="row" style={{ gap: 6 }}>
            inspect model
            <select className="select" value={sel} onChange={(e) => setSel(Number(e.target.value))}>
              {pins.map((m, i) => (
                <option key={m} value={i}>
                  {shortModel(m)}
                </option>
              ))}
            </select>
          </label>
          {DatasetPicker}
        </div>
      </div>

      <Card title="Config inspector (CFG-1)">
        <AsyncPanel query={cfg} height={240}>
          {(d) => (
            <div className="col" style={{ gap: 12 }}>
              <ScoreFormulaCard formula={d.score_formula} />
              <div className="toolbar">
                <input className="input" placeholder="search keys…" value={search} onChange={(e) => setSearch(e.target.value)} style={{ maxWidth: 300 }} />
                {/* FB-8: active config only by default; reveal inactive (flag-off) keys. */}
                <label className="row" style={{ gap: 6 }} title="Reveal keys whose deciding feature flag is OFF (e.g. scad_* when use_scad=false). Nothing is irreversibly hidden.">
                  <input type="checkbox" checked={showInactive} onChange={(e) => setShowInactive(e.target.checked)} />
                  show inactive ({activity.inactive.size})
                </label>
              </div>
              <div className="card-grid">
                {Object.entries(groupConfig(d)).map(([g, kv]) => {
                  const rows = Object.entries(kv).filter(
                    ([k]) =>
                      (!search || k.toLowerCase().includes(search.toLowerCase())) &&
                      // FB-8: hide inactive keys unless the toggle is on (payload keeps them).
                      (showInactive || !activity.inactive.has(k))
                  );
                  if (rows.length === 0) return null;
                  return (
                    <div className="stat-card" key={g}>
                      <div className="card-title" style={{ fontSize: "var(--fs-body)" }}>
                        {g}
                      </div>
                      {/* FB-R3-01: `cfg-kv` scopes a value-WRAP fix to the inspector
                          table only — the value `<td>` wraps so nothing is clipped by
                          the card edge; the KEY cell stays compact (ellipsis + title). */}
                      <table className="tbl mono cfg-kv" style={{ fontSize: "0.78rem" }}>
                        <tbody>
                          {rows.map(([k, v]) => {
                            const isInactive = activity.inactive.has(k);
                            return (
                              <tr key={k} style={{ opacity: isInactive ? 0.5 : 1 }}>
                                <td className="subtle cfg-key" title={k}>
                                  {k}
                                  {isInactive && (
                                    <span className="chip" style={{ marginLeft: 6, fontSize: "0.7rem" }} title={activity.reason[k]}>
                                      inactive
                                    </span>
                                  )}
                                </td>
                                <td className="cfg-val">{renderVal(v)}</td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </AsyncPanel>
      </Card>

      <Card title="Cross-experiment config diff (CFG-2)">
        <div className="toolbar">
          {/* left/right MODEL selectors (from the pinned set) drive the two diff columns. */}
          <label className="row" style={{ gap: 6 }}>
            left
            <select className="select" value={leftSel} onChange={(e) => setLeftIdx(Number(e.target.value))}>
              {pins.map((m, i) => (
                <option key={m} value={i}>
                  {shortModel(m)}
                </option>
              ))}
            </select>
          </label>
          <label className="row" style={{ gap: 6 }}>
            right
            <select className="select" value={rightSel} onChange={(e) => setRightIdx(Number(e.target.value))}>
              {pins.map((m, i) => (
                <option key={m} value={i}>
                  {shortModel(m)}
                </option>
              ))}
            </select>
          </label>
          <span className="grow" style={{ flex: 1 }} />
          <label className="row" style={{ gap: 6 }}>
            <input type="checkbox" checked={onlyDiff} onChange={(e) => setOnlyDiff(e.target.checked)} /> only differing keys
          </label>
        </div>
        <AsyncPanel query={diff} isEmpty={(d) => !d.keys?.length} height={240}>
          {(d) => {
            const keys = diffKeys; // hoisted (F-02); never a hook inside this render-prop
            return (
              <div className="tbl-wrap">
                <table className="tbl mono" style={{ fontSize: "0.78rem" }}>
                  <thead>
                    <tr>
                      <th>key</th>
                      {d.leaves.map((l, i) => (
                        <th key={i}>
                          <span className="row" style={{ gap: 6 }}>
                            <span style={{ width: 9, height: 9, borderRadius: 3, background: catColor(i) }} />
                            {shortModel(l.exp_id)}
                          </span>
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {keys.map((row) => (
                      <tr key={row.key}>
                        <td className="subtle">{row.key}</td>
                        {d.leaves.map((l) => {
                          const id = `${l.exp_id}|${l.dataset_key}|${l.variant ?? "_"}`;
                          const present = row.values && id in row.values;
                          return (
                            <td key={id} style={{ background: row.differs ? "var(--sem-warn-soft)" : "transparent" }}>
                              {present ? renderVal(row.values[id]) : <span className="subtle">— absent</span>}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            );
          }}
        </AsyncPanel>
      </Card>
    </div>
  );
}

function renderVal(v: any): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

/* Display-only grouping of the flat config dict by a key-prefix heuristic (F5: a
 * no-match key lands in "other", never dropped). Prefers a server-provided `groups`. */
function groupConfig(d: { config?: Record<string, any>; groups?: Record<string, Record<string, any>> }): Record<string, Record<string, any>> {
  if (d.groups && Object.keys(d.groups).length > 0) return d.groups;
  const flat = d.config ?? {};
  const buckets: Record<string, Record<string, any>> = {
    architecture: {},
    masking: {},
    loss_margin: {},
    grl: {},
    fm: {},
    scad: {},
    score: {},
    schedule: {},
    other: {},
  };
  for (const [k, v] of Object.entries(flat)) {
    const lk = k.toLowerCase();
    let g = "other";
    if (/(grl)/.test(lk)) g = "grl";
    else if (/(^fm_|_fm_|feature_match)/.test(lk)) g = "fm";
    else if (/(scad)/.test(lk)) g = "scad";
    else if (/(mask|patch)/.test(lk)) g = "masking";
    else if (/(margin|hinge|softplus|loss|lambda|discrepancy)/.test(lk)) g = "loss_margin";
    else if (/(score|ratio)/.test(lk)) g = "score";
    else if (/(warmup|epoch|schedule|lr|optimizer|batch|ema|early)/.test(lk)) g = "schedule";
    else if (/(seq|feature|hidden|layer|dim|head|encoder|decoder|model|cnn|linear)/.test(lk)) g = "architecture";
    buckets[g][k] = v;
  }
  // drop empty buckets
  return Object.fromEntries(Object.entries(buckets).filter(([, kv]) => Object.keys(kv).length > 0));
}
