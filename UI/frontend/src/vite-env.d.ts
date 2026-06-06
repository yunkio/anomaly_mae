/// <reference types="vite/client" />

declare module "react-plotly.js" {
  import * as React from "react";
  interface PlotParams {
    data: any[];
    layout?: any;
    config?: any;
    style?: React.CSSProperties;
    className?: string;
    useResizeHandler?: boolean;
    onInitialized?: (figure: any, graphDiv: HTMLElement) => void;
    onUpdate?: (figure: any, graphDiv: HTMLElement) => void;
    [key: string]: any;
  }
  const Plot: React.ComponentType<PlotParams>;
  export default Plot;
}

declare module "plotly.js-dist-min" {
  const Plotly: any;
  export default Plotly;
}

declare module "react-plotly.js/factory" {
  import * as React from "react";
  import type { PlotParams } from "react-plotly.js";
  export default function createPlotlyComponent(plotly: any): React.ComponentType<PlotParams>;
}

/* FB-R3-04/05 — gifuct-js (client-side GIF decode for the canvas player). The package
 * ships its own index.d.ts, but this minimal fallback keeps the build self-contained. */
declare module "gifuct-js" {
  export function parseGIF(arrayBuffer: ArrayBuffer): any;
  export function decompressFrames(parsedGif: any, buildImagePatches: boolean): any[];
  export function decompressFrame(frame: any, gct: any, buildImagePatches: boolean): any;
}
