/* One Plotly template (light+dark) read from the active CSS theme tokens, so every
 * chart is visually consistent (AC-F3.2). Layout helpers compose on top of it. */

function tok(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export function baseLayout(extra: Partial<any> = {}): any {
  return {
    paper_bgcolor: tok("--plot-bg") || "#fff",
    plot_bgcolor: tok("--plot-bg") || "#fff",
    font: { family: tok("--font-sans") || "system-ui", size: 12, color: tok("--text") || "#1c2330" },
    margin: { l: 52, r: 16, t: 16, b: 40 },
    xaxis: {
      gridcolor: tok("--plot-grid") || "#e3e8f0",
      zerolinecolor: tok("--plot-zeroline") || "#c7cfdc",
      linecolor: tok("--plot-axis") || "#5a6677",
      tickfont: { size: 11 },
      automargin: true,
    },
    yaxis: {
      gridcolor: tok("--plot-grid") || "#e3e8f0",
      zerolinecolor: tok("--plot-zeroline") || "#c7cfdc",
      linecolor: tok("--plot-axis") || "#5a6677",
      tickfont: { size: 11 },
      automargin: true,
    },
    hovermode: "closest",
    hoverlabel: { bgcolor: tok("--surface") || "#fff", bordercolor: tok("--border") || "#ddd" },
    legend: { font: { size: 11 }, orientation: "h", y: -0.18 },
    showlegend: true,
    ...extra,
  };
}

export const PLOTLY_CONFIG: any = {
  displaylogo: false,
  responsive: true,
  modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"],
  // FB-15: raise the client-side export resolution so a downloaded PNG is crisp on a
  // high-DPI screen (scale 2→4 ≈ 4× the pixels at the on-screen size). The server GIFs
  // already render at higher DPI (render.py _FIG_DPI=140); this matches that bar for the
  // interactive Plotly charts' image export.
  toImageButtonOptions: { format: "png", scale: 4 },
};

/** A pre-warmup grey/hatch span shape for the masking band. */
export function warmupShape(warmup: number | null | undefined): any[] {
  if (!warmup || warmup <= 0) return [];
  return [
    {
      type: "rect",
      xref: "x",
      yref: "paper",
      x0: 0,
      x1: warmup,
      y0: 0,
      y1: 1,
      fillcolor: tok("--plot-mask") || "rgba(120,132,152,0.13)",
      line: { width: 0 },
      layer: "below",
    },
  ];
}

export function warmupAnnotation(warmup: number | null | undefined): any[] {
  if (!warmup || warmup <= 0) return [];
  return [
    {
      x: warmup,
      yref: "paper",
      y: 1,
      text: `student joins · warmup=${warmup}`,
      showarrow: false,
      font: { size: 10, color: tok("--plot-mask-line") || "#8893a4" },
      xanchor: "left",
      yanchor: "top",
      bgcolor: "transparent",
    },
  ];
}

export function warmupLine(warmup: number | null | undefined): any[] {
  if (!warmup || warmup <= 0) return [];
  return [
    {
      type: "line",
      xref: "x",
      yref: "paper",
      x0: warmup,
      x1: warmup,
      y0: 0,
      y1: 1,
      line: { color: tok("--plot-mask-line") || "#8893a4", width: 1, dash: "dot" },
      layer: "below",
    },
  ];
}
