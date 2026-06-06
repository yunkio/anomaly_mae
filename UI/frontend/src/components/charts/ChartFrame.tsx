/* FB-R3-03 — <ChartFrame> + <DownloadPngButton>: the reusable "↓ PNG" download control
 * for every Plotly chart surface.
 *
 * <ChartFrame> renders a <Plot> (capturing its graph div via onInitialized) and a
 * header-row "↓ PNG" button wired to downloadChartPng (scale 4, FB-15). Raw-<Plot>
 * panels swap `<Plot .../>` for `<ChartFrame .../>` and inherit the button. The
 * MetricChart primitive captures its own gd and mounts <DownloadPngButton> directly, so
 * its callers inherit the button for free.
 *
 * These are real components (hooks live in their bodies), so using them as JSX inside an
 * AsyncPanel render-prop is safe — no hook is added to the parent's render-prop. */
import { useRef } from "react";
import Plot from "../../viz/Plot";
import { downloadChartPng } from "../../viz/download";

/** A standalone "↓ PNG" button bound to an already-captured graph div ref. */
export function DownloadPngButton({
  gdRef,
  name,
  className = "btn sm",
  label = "↓ PNG",
}: {
  gdRef: { current: any };
  name: string;
  className?: string;
  label?: string;
}) {
  return (
    <button
      type="button"
      className={className}
      title="download this chart as a high-resolution PNG"
      onClick={() => void downloadChartPng(gdRef.current, name)}
    >
      {label}
    </button>
  );
}

interface ChartFrameProps {
  /** Plotly traces. */
  data: any[];
  layout: any;
  config?: any;
  style?: React.CSSProperties;
  useResizeHandler?: boolean;
  /** filename stem for the downloaded PNG. */
  name: string;
  /** extra controls rendered in the header row, left of the download button. */
  actions?: React.ReactNode;
  /** optional caption above the chart (right-aligned download button always shown). */
  title?: React.ReactNode;
}

/** A <Plot> wrapped with a header-row "↓ PNG" download button. */
export default function ChartFrame({
  data,
  layout,
  config,
  style,
  useResizeHandler = true,
  name,
  actions,
  title,
}: ChartFrameProps) {
  const gdRef = useRef<any>(null);
  return (
    <div>
      <div className="row wrap" style={{ gap: 8, alignItems: "center", justifyContent: "space-between" }}>
        <div className="row wrap" style={{ gap: 8, alignItems: "center" }}>
          {title}
          {actions}
        </div>
        <DownloadPngButton gdRef={gdRef} name={name} />
      </div>
      <Plot
        data={data}
        layout={layout}
        config={config}
        style={style}
        useResizeHandler={useResizeHandler}
        onInitialized={(_fig: any, gd: any) => {
          gdRef.current = gd;
        }}
        onUpdate={(_fig: any, gd: any) => {
          gdRef.current = gd;
        }}
      />
    </div>
  );
}
