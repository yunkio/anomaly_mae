/* Small shared presentational helpers. */
import type { ReactNode } from "react";
import type { Direction, LeafState, MetricMeta } from "../api/types";

export function StateChip({ state, source }: { state: LeafState; source?: string }) {
  return (
    <span className="row" style={{ gap: 6 }}>
      <span className={`chip ${state}`}>{state.replace("_", " ")}</span>
      {source === "fixture" && <span className="chip fixture">fixture</span>}
    </span>
  );
}

export function DirectionBadge({ direction }: { direction: Direction }) {
  const sym = direction === "up" ? "↑" : direction === "down" ? "↓" : "◦";
  const label =
    direction === "up" ? "higher better" : direction === "down" ? "lower better" : "context";
  return (
    <span className={`badge-dir ${direction}`} title={label}>
      {sym}
    </span>
  );
}

export function MetaBadges({ meta }: { meta: MetricMeta }) {
  return (
    <span className="row" style={{ gap: 6 }}>
      <DirectionBadge direction={meta.direction} />
      {meta.inferred && <span className="chip inferred">inferred</span>}
      {meta.phase_validity === "post-warmup" && <span className="chip">post-warmup</span>}
    </span>
  );
}

export function fmt(v: number | null | undefined, digits = 4): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  if (Math.abs(v) >= 1000 || (Math.abs(v) < 0.001 && v !== 0)) return v.toExponential(2);
  return Number(v.toFixed(digits)).toString();
}

export function Card({ title, children, actions }: { title?: ReactNode; children: ReactNode; actions?: ReactNode }) {
  return (
    <div className="card">
      {(title || actions) && (
        <div className="card-title" style={{ justifyContent: "space-between" }}>
          <span>{title}</span>
          {actions}
        </div>
      )}
      {children}
    </div>
  );
}

export function EmptyOnboarding({ message }: { message: string }) {
  return (
    <div className="async-state" style={{ minHeight: 220 }}>
      <span className="glyph">◔</span>
      <div style={{ fontWeight: 600 }}>{message}</div>
      <div className="subtle">
        Point the backend at <span className="mono">results/experiments/</span> (or the read-only{" "}
        <span className="mono">.trash/0601/pre_fullrerun/</span> fixtures) and rescan.
      </div>
    </div>
  );
}
