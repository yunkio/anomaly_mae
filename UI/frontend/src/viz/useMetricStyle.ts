/* The single registry-driven style chokepoint (frontend-design §2.5). Maps a
 * MetricMeta -> {lineColor, dash, yRange, badge, masked} so NO chart hard-codes a
 * color/direction. A renamed/added/removed metric is handled with zero chart code. */
import type { MetricMeta } from "../api/types";

function tok(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export interface MetricStyle {
  color: string; // good-hue / neutral / inferred
  dash: "solid" | "dot" | "dash";
  yRange?: [number, number];
  badge?: "inferred" | "warn";
  maskPrewarmup: boolean;
  directionLabel: string;
  rankable: boolean;
}

/** Score-[0,1] units get a fixed y-range so plateaus read honestly (viz Story 1). */
function isScoreUnit(meta: MetricMeta): boolean {
  return (meta.unit || "").includes("[0,1]");
}

export function metricStyle(meta: MetricMeta, value?: number | null): MetricStyle {
  const inferred = !!meta.inferred;
  const negativeSnr =
    (meta.family || "").startsWith("diagnostic.separation") &&
    typeof value === "number" &&
    value < 0;

  let color: string;
  if (inferred) color = tok("--sem-inferred") || "#7b5bd6";
  else if (negativeSnr) color = tok("--sem-warn") || "#c47b12";
  else if (meta.direction === "neutral") color = tok("--sem-neutral") || "#5d6b85";
  else color = tok("--sem-good") || "#128a6e";

  const maskPrewarmup =
    meta.phase_validity === "post-warmup" || !!meta.viz_hint?.mask_prewarmup;

  return {
    color,
    dash: inferred ? "dot" : "solid",
    yRange: isScoreUnit(meta) ? [0, 1] : undefined,
    badge: inferred ? "inferred" : negativeSnr ? "warn" : undefined,
    maskPrewarmup,
    directionLabel:
      meta.direction === "up" ? "higher better ↑" : meta.direction === "down" ? "lower better ↓" : "context ◦",
    rankable: meta.direction !== "neutral",
  };
}

/** Whether this series should be drawn as markers+gaps, not an interpolated line
 * (AC-F4.5 sparsity honesty). */
export function isSparse(meta: MetricMeta): boolean {
  return meta.viz_hint?.sparse_axis === "epoch" || !!meta.viz_hint?.verify_density;
}
