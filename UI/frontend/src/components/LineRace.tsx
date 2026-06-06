/* <LineRace> (FB-3, the headline) — a CLIENT-SIDE interactive per-epoch LINE race for
 * the pinned compare set, built on Plotly animation frames.
 *
 *  • each pinned leaf = one distinct-colored LINE (catColor by pin order, same color the
 *    leaf has in every other view);
 *  • x = eval epoch, y = the selected metric's value AT that epoch — the REAL per-eval-
 *    epoch series from POST /api/compare/series (native 5-epoch grid, eval_interval=5);
 *  • points race / overtake as the animation advances (a leaf higher at epoch 200 but
 *    lower at 350 visibly overtakes);
 *  • ► PLAY animates to the final epoch then PAUSES (no infinite loop — `mode:"immediate"`,
 *    a single forward sweep, then we stop on the last frame);
 *  • ↻ REPLAY restarts from frame 0;
 *  • a SLIDER / "stop at epoch" select clamps the sweep to a target epoch and freezes
 *    there; a current-epoch readout shows the animated epoch.
 *
 * P3-01 / R2-05: `compare/series` serves per-epoch ONLY from `epoch.series` (genuine
 * `epoch_metrics`). A best-epoch-only-scalar metric (student/disc detection variants are
 * NOT in `epoch.series`) yields an EMPTY series for that leaf → its line is absent (a
 * GAP), NEVER a fabricated per-epoch trajectory. We surface that as a "scalar-only (no
 * per-epoch line)" note per affected leaf.
 *
 * Hooks live in the component body (NOT inside an <AsyncPanel> render-prop) — the
 * imperative `Plotly.animate` is driven off the graph-div ref captured in onInitialized. */
import { useEffect, useMemo, useRef, useState } from "react";
// @ts-ignore - dist-min has no types (see vite-env.d.ts)
import Plotly from "plotly.js-dist-min";
import Plot from "../viz/Plot";
import { DownloadPngButton } from "./charts/ChartFrame";
import { useCompareSeries } from "../api/queries";
import { baseLayout, PLOTLY_CONFIG } from "../viz/plotlyTemplate";
import { catColor } from "../store";
import { shortModel } from "../scope";
import type { CompareSeriesItem, LeafSel } from "../api/types";

interface Props {
  leaves: LeafSel[];
  metricKey: string;
  /** when true clamp y to [0,1] (score metrics). */
  scoreScale?: boolean;
  height?: number;
  /** model-stable color for a leaf_id (defaults to race order); keeps a model the same
   * color it has in every other view. */
  colorOf?: (leafId: string) => string;
}

/** leaf_id is "exp|dataset|variant"; the model (compared dimension) is the first segment. */
const expOf = (leafId: string): string => leafId.split("|")[0];

const FRAME_MS = 110; // per-frame duration for a brisk-but-readable race

export default function LineRace({ leaves, metricKey, scoreScale = true, height = 380, colorOf }: Props) {
  const seriesQ = useCompareSeries(leaves, metricKey);
  const gdRef = useRef<any>(null); // the Plotly graph div (captured in onInitialized)
  const [playing, setPlaying] = useState(false);
  const [frameIdx, setFrameIdx] = useState(0); // index into the union epoch grid
  const [stopAt, setStopAt] = useState<number | null>(null); // target epoch to freeze at

  // colour each line by its model (stable across views), else fall back to race order.
  const pinColor = (lid: string, fallback: number) => (colorOf ? colorOf(lid) : catColor(fallback));
  const lineLabel = (lid: string) => shortModel(expOf(lid));

  // The UNION epoch grid (aligned by epoch NUMBER across leaves; gaps stay gaps). 5-epoch
  // resolution is native (eval_interval=5 → the series IS 5,10,…,500). value(leaf,epoch)
  // looks up by epoch number so a leaf missing an epoch contributes a gap (null), never 0.
  const { epochGrid, available, scalarOnly } = useMemo(() => {
    const data = seriesQ.data;
    if (!data?.series?.length) return { epochGrid: [] as number[], available: [] as CompareSeriesItem[], scalarOnly: [] as string[] };
    const avail = data.series.filter((s) => (s.epochs?.length ?? 0) > 0);
    const scalar = data.series.filter((s) => (s.epochs?.length ?? 0) === 0).map((s) => s.leaf_id);
    const grid = Array.from(new Set(avail.flatMap((s) => s.epochs))).sort((a, b) => a - b);
    return { epochGrid: grid, available: avail, scalarOnly: scalar };
  }, [seriesQ.data]);

  // per-leaf {epoch -> value} maps (lookup by epoch number; missing → gap).
  const valueMaps = useMemo(() => {
    const m: Record<string, Map<number, number | null>> = {};
    for (const s of available) {
      const mm = new Map<number, number | null>();
      for (let i = 0; i < s.epochs.length; i++) {
        const v = s.values[i];
        mm.set(s.epochs[i], typeof v === "number" ? v : null);
      }
      m[s.leaf_id] = mm;
    }
    return m;
  }, [available]);

  const lastFrame = epochGrid.length - 1;
  // the target frame the sweep stops at (the "stop at epoch" clamp, else the final epoch).
  const stopFrame = useMemo(() => {
    if (stopAt == null) return lastFrame;
    const i = epochGrid.findIndex((e) => e >= stopAt);
    return i < 0 ? lastFrame : i;
  }, [stopAt, epochGrid, lastFrame]);

  // Build the Plotly figure: one base trace per leaf (the line UP TO the current frame)
  // + a head marker at the current epoch. We update traces imperatively via animate frames
  // for the race, and via this initial `data` for the first paint.
  const buildTraces = (uptoIdx: number) => {
    const upto = epochGrid.slice(0, uptoIdx + 1);
    const traces: any[] = [];
    for (let li = 0; li < available.length; li++) {
      const s = available[li];
      const color = pinColor(s.leaf_id, li);
      const ys = upto.map((e) => {
        const v = valueMaps[s.leaf_id]?.get(e);
        return v === undefined ? null : v;
      });
      // the line up to `upto`
      traces.push({
        x: upto,
        y: ys,
        name: lineLabel(s.leaf_id),
        type: "scatter",
        mode: "lines",
        line: { color, width: 2.2 },
        connectgaps: false,
        hovertemplate: `${lineLabel(s.leaf_id)}<br>epoch %{x}: %{y:.4f}<extra></extra>`,
      });
      // the racing HEAD marker at the current epoch
      const headEpoch = upto[upto.length - 1];
      const headVal = headEpoch === undefined ? null : valueMaps[s.leaf_id]?.get(headEpoch) ?? null;
      traces.push({
        x: headVal == null ? [] : [headEpoch],
        y: headVal == null ? [] : [headVal],
        type: "scatter",
        mode: "markers",
        marker: { color, size: 11, line: { width: 1.5, color: "var(--surface)" } },
        showlegend: false,
        hoverinfo: "skip",
      });
    }
    return traces;
  };

  const initialTraces = useMemo(() => buildTraces(frameIdx), [available, valueMaps, epochGrid, frameIdx]);

  const direction = seriesQ.data?.direction;
  const dispName = seriesQ.data?.meta?.[metricKey]?.display_name ?? metricKey;
  const layout = useMemo(
    () =>
      baseLayout({
        height,
        yaxis: {
          ...baseLayout().yaxis,
          range: scoreScale ? [0, 1] : undefined,
          title: { text: `${dispName}${direction === "down" ? " ↓" : direction === "up" ? " ↑" : ""}` },
        },
        xaxis: {
          ...baseLayout().xaxis,
          range: epochGrid.length ? [epochGrid[0], epochGrid[lastFrame]] : undefined,
          title: { text: "eval epoch (per-epoch value, 5-epoch resolution)" },
        },
      }),
    [height, scoreScale, dispName, direction, epochGrid, lastFrame]
  );

  // Imperative single-sweep animation: step frameIdx forward on a timer to `stopFrame`,
  // then STOP (no loop). Each tick restyles the gd's traces to the current epoch.
  const timerRef = useRef<number | null>(null);
  useEffect(() => {
    if (!playing) {
      if (timerRef.current) window.clearInterval(timerRef.current);
      timerRef.current = null;
      return;
    }
    timerRef.current = window.setInterval(() => {
      setFrameIdx((cur) => {
        const next = cur + 1;
        if (next >= stopFrame) {
          setPlaying(false); // reached the target epoch → PAUSE (no infinite loop)
          return stopFrame;
        }
        return next;
      });
    }, FRAME_MS);
    return () => {
      if (timerRef.current) window.clearInterval(timerRef.current);
      timerRef.current = null;
    };
  }, [playing, stopFrame]);

  // push the current frame into the live graph div (restyle is cheaper than re-render).
  useEffect(() => {
    const gd = gdRef.current;
    if (!gd || !epochGrid.length) return;
    const traces = buildTraces(frameIdx);
    try {
      Plotly.react(gd, traces, layout, PLOTLY_CONFIG);
    } catch {
      /* gd not ready yet — the initial <Plot data> already painted frame 0 */
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [frameIdx, layout]);

  function play() {
    if (!epochGrid.length) return;
    // if we are already at the stop frame, a play restarts from 0 (acts as replay+play).
    if (frameIdx >= stopFrame) setFrameIdx(0);
    setPlaying(true);
  }
  function pause() {
    setPlaying(false);
  }
  function replay() {
    setPlaying(false);
    setFrameIdx(0);
    // next tick start playing from the beginning
    window.setTimeout(() => setPlaying(true), 0);
  }

  if (seriesQ.isLoading) return <div className="async-skel" style={{ height }} aria-busy="true" />;
  if (seriesQ.isError)
    return (
      <div className="async-state error" style={{ minHeight: 140 }}>
        <span className="glyph">⚠</span>
        <div>Race series request failed</div>
      </div>
    );
  if (!available.length)
    return (
      <div className="async-state" style={{ minHeight: 160 }}>
        <span className="glyph">∅</span>
        <div>
          No per-epoch series for <span className="mono">{metricKey}</span> on the selected models.
          {scalarOnly.length > 0 && (
            <div className="subtle" style={{ marginTop: 6, fontSize: "var(--fs-small)" }}>
              {scalarOnly.length} leaf(s) have this metric only as a best-epoch SCALAR (e.g. a
              student/disc detection variant) — a scalar has no per-epoch line, so it is shown as a
              gap, never a fabricated trajectory (P3-01).
            </div>
          )}
        </div>
      </div>
    );

  const curEpoch = epochGrid[frameIdx] ?? epochGrid[lastFrame];

  return (
    <div className="col" style={{ gap: 10 }}>
      <div className="row wrap" style={{ gap: 8, alignItems: "center" }}>
        {!playing ? (
          <button className="btn primary sm" onClick={play} title="play to the target epoch then pause">
            ► play
          </button>
        ) : (
          <button className="btn sm" onClick={pause} title="pause the race">
            ❚❚ pause
          </button>
        )}
        <button className="btn sm" onClick={replay} title="restart from epoch 0">
          ↻ replay
        </button>
        <label className="row" style={{ gap: 6, fontSize: "var(--fs-small)" }}>
          stop at epoch
          <select
            className="select"
            value={stopAt == null ? "" : String(stopAt)}
            onChange={(e) => setStopAt(e.target.value === "" ? null : Number(e.target.value))}
          >
            <option value="">final ({epochGrid[lastFrame]})</option>
            {epochGrid.map((e) => (
              <option key={e} value={e}>
                {e}
              </option>
            ))}
          </select>
        </label>
        <span className="chip mono" title="the epoch the race is currently showing">
          epoch {curEpoch} / {epochGrid[lastFrame]}
        </span>
        <span style={{ flex: 1 }} />
        {/* FB-R3-03: download the current race frame as a high-res PNG (the gd captured
            in onInitialized). */}
        <DownloadPngButton gdRef={gdRef} name={`line_race_${metricKey}_epoch_${curEpoch}`} />
      </div>

      {/* the seek SLIDER — drag to any frame; freezes the race there (pauses). */}
      <input
        type="range"
        min={0}
        max={lastFrame}
        step={1}
        value={frameIdx}
        onChange={(e) => {
          setPlaying(false);
          setFrameIdx(Number(e.target.value));
        }}
        style={{ width: "100%" }}
        aria-label="seek epoch"
      />

      <Plot
        data={initialTraces}
        layout={layout}
        config={PLOTLY_CONFIG}
        style={{ width: "100%", height }}
        useResizeHandler
        onInitialized={(_fig: any, gd: any) => {
          gdRef.current = gd;
        }}
      />

      {scalarOnly.length > 0 && (
        <div className="subtle" style={{ fontSize: "var(--fs-small)" }}>
          {scalarOnly.length} model(s) have <span className="mono">{metricKey}</span> only as a
          best-epoch scalar → shown as a gap (no fabricated per-epoch line, P3-01).
        </div>
      )}
      <div className="subtle" style={{ fontSize: "var(--fs-small)" }}>
        Real per-epoch values aligned by epoch number; lines overtake as the race advances. Play
        stops at the target epoch (no infinite loop); replay restarts; the slider seeks/freezes.
      </div>
    </div>
  );
}
