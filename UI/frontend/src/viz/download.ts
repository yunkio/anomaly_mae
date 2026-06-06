/* FB-R3-03 — reusable high-resolution PNG export for any Plotly chart.
 *
 * Uses `Plotly.toImage` on the live graph div at scale 4 (the same FB-15 resolution the
 * modebar camera already uses, `plotlyTemplate.ts PLOTLY_CONFIG.toImageButtonOptions`),
 * then triggers a same-origin `<a download>` save. Fully client-side; no backend render.
 * The graph div `gd` is the element react-plotly.js passes to `onInitialized(fig, gd)`. */
// @ts-ignore - dist-min has no types (declared in vite-env.d.ts)
import Plotly from "plotly.js-dist-min";

/** Default export scale — mirrors PLOTLY_CONFIG.toImageButtonOptions.scale (FB-15). */
export const PNG_EXPORT_SCALE = 4;

function safeName(name: string): string {
  const base = (name || "chart").replace(/[^a-z0-9._-]+/gi, "_").replace(/^_+|_+$/g, "");
  return (base || "chart") + (base.toLowerCase().endsWith(".png") ? "" : ".png");
}

/** Export the chart in `gd` as a crisp (scale 4) PNG and save it as `filename`. */
export async function downloadChartPng(gd: any, filename: string): Promise<void> {
  if (!gd) return;
  // size from the rendered layout so the PNG matches what is on screen (then ×scale).
  const w = gd._fullLayout?.width || gd.clientWidth || 900;
  const h = gd._fullLayout?.height || gd.clientHeight || 480;
  const url: string = await Plotly.toImage(gd, {
    format: "png",
    scale: PNG_EXPORT_SCALE,
    width: w,
    height: h,
  });
  const a = document.createElement("a");
  a.href = url;
  a.download = safeName(filename);
  document.body.appendChild(a);
  a.click();
  a.remove();
}
