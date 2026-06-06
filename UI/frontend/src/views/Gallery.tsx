/* Visualization Gallery (IA-4) — browse the pre-rendered PNGs (+ cached GIF
 * cross-links) for a chosen MODEL on a chosen dataset (2026-06-04 pin refactor). */
import { useState } from "react";
import { Card } from "../components/common";
import PngGallery from "../components/panels/PngGallery";
import { useStore } from "../store";
import { useScopeDataset, shortModel } from "../scope";

export default function Gallery() {
  const pins = useStore((s) => s.pins);
  const [sel, setSel] = useState(0);
  const model = pins[Math.min(sel, Math.max(0, pins.length - 1))];
  const scope = useScopeDataset(model ? [model] : []);

  if (pins.length === 0)
    return (
      <div>
        <h1 className="view-title">Visualization Gallery</h1>
        <p className="view-sub">Pin a model (Overview / Rankings) to browse its visualizations.</p>
        <div className="async-state" style={{ minHeight: 200 }}>
          <span className="glyph">▤</span>
          <div>No pinned models.</div>
        </div>
      </div>
    );

  return (
    <div className="col" style={{ gap: 24 }}>
      <div>
        <h1 className="view-title">Visualization Gallery</h1>
        <p className="view-sub">Enumerates exactly the PNGs present; stale viz is badged, not trusted.</p>
        <div className="toolbar">
          <label className="row" style={{ gap: 6 }}>
            model
            <select className="select" value={sel} onChange={(e) => setSel(Number(e.target.value))}>
              {pins.map((m, i) => (
                <option key={m} value={i}>
                  {shortModel(m)}
                </option>
              ))}
            </select>
          </label>
          {scope.options.length > 0 && (
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
          )}
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
        </div>
      </div>
      <Card title={model ? `${shortModel(model)} · ${scope.datasetKey ?? "—"}` : ""}>
        {model && scope.datasetKey && (
          <PngGallery exp={model} dsKey={scope.datasetKey} variant={scope.variant} />
        )}
      </Card>
    </div>
  );
}
