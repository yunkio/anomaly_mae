/* The resolved score-formula card (CFG-4) — never blank, from the §A6 resolver. */
import type { ScoreFormula } from "../api/types";

export default function ScoreFormulaCard({ formula }: { formula?: ScoreFormula }) {
  if (!formula) return <span className="subtle">score formula unavailable</span>;
  return (
    <div className="row wrap" style={{ gap: 8 }}>
      <span className="chip" title="resolved recon:disc ratio">
        recon:disc = {formula.ratio}:1
      </span>
      <span className="chip" title="feature-matching in the inference score?">
        FM {formula.fm_in_score ? "included" : "excluded"}
      </span>
      <span className="chip" title="provenance of the resolved ratio">
        via {formula.source}
      </span>
      <span className="mono subtle" style={{ fontSize: "0.78rem" }}>
        {formula.formula_text}
      </span>
    </div>
  );
}
