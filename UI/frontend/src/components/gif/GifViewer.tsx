/* <GifViewer> — the flagship showcase player, a CANVAS-BASED FRAME PLAYER
 * (FB-R3-04 / FB-R3-05; FB-R4-01 perf pass).
 *
 * It still composes a GifSpec, POSTs /gif/render, polls /gif/status, then — instead of a
 * native <img> (which cannot be seeked, paused, or downloaded per-frame) — it:
 *   1. fetches the cached GIF bytes and DECODES them client-side with gifuct-js
 *      (pure JS, CPU-only — no backend render, preserves the safety envelope);
 *   2. composes each (possibly PARTIAL) frame's patch onto a PERSISTENT full-frame
 *      canvas honoring the per-frame disposal method — ONCE — caching the fully-composited
 *      ImageData per frame index (so seek-to-frame never shows holes AND never re-decodes
 *      or re-composites from frame 0). The one-time decode runs OFF the React render path
 *      after a paint yield so the "decoding…" state is actually visible (FB-R4-01);
 *   3. PLAYS exactly TWICE then FREEZES on the last frame — the loop count is enforced in
 *      JS (the embedded GIF loop flag is ignored). Playback is driven by ONE
 *      requestAnimationFrame loop that accumulates real elapsed time against each frame's
 *      own `delay` and advances frame index in O(1) (no per-tick decode/composite,
 *      no setTimeout drift, auto-throttles when the tab is hidden) (FB-R3-05a / FB-R4-01);
 *   4. a SEEK SLIDER holds any frame in O(1) (just paints the cached ImageData for that
 *      index); the frame→epoch readout uses the backend `frame_epochs` (from /gif/status
 *      or /gif/sidecar) and COLLAPSES the trailing peak-hold tail to the final epoch so
 *      the last ~hold_len duplicate frames map to one seek stop (R3-SR-03 / R3-SR-05)
 *      (FB-R3-05b);
 *   5. a "↓ PNG (this epoch)" button saves the currently-painted frame via canvas.toBlob
 *      — the SAME canvas the slider holds, so download-at-epoch and seek share one frame
 *      model by construction (FB-R3-04).
 *
 * The whole-GIF "↓ export GIF" + "copy cache key" controls are kept. */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
// @ts-ignore - gifuct-js ships its own .d.ts but resolution via dist may not surface it
import { parseGIF, decompressFrames } from "gifuct-js";
import { api } from "../../api/client";
import type { GifSpec, GifStatus, ParsedGifFrame } from "../../api/types";

interface Props {
  spec: GifSpec | null;
  height?: number;
  title?: string;
  autoRender?: boolean;
}

type Phase = "idle" | "rendering" | "ready" | "error";

const LOOP_COUNT = 2; // FB-R3-05a: play through exactly twice, then freeze.

export default function GifViewer({ spec, height = 360, title, autoRender = true }: Props) {
  const [phase, setPhase] = useState<Phase>("idle");
  const [cacheKey, setCacheKey] = useState<string | null>(null);
  const [progress, setProgress] = useState<string>("");
  const [err, setErr] = useState<string>("");
  const pollRef = useRef<number | null>(null);

  // ── decoded frame state ──────────────────────────────────────────────────
  // frame DELAYS only (centiseconds); the heavy decoded patches + composited ImageData
  // live in refs (never in React state — they must not trigger re-render or be cloned).
  const [frameDelays, setFrameDelays] = useState<number[]>([]);
  const [dims, setDims] = useState<{ width: number; height: number }>({ width: 0, height: 0 });
  const [frameEpochs, setFrameEpochs] = useState<number[]>([]);
  const [holdLen, setHoldLen] = useState<number>(0);
  const [frameIdx, setFrameIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [decoding, setDecoding] = useState(false); // FB-R4-01: visible one-time-decode state
  const [decodeMs, setDecodeMs] = useState<number | null>(null); // measured decode cost

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  // pre-composed full-frame ImageData per frame index (so seek is instant + hole-free).
  const composedRef = useRef<ImageData[]>([]);
  const delaysRef = useRef<number[]>([]); // frame delays in centiseconds (rAF reads this)
  const rafRef = useRef<number | null>(null);
  const loopCountRef = useRef(0);
  const frameIdxRef = useRef(0); // mirror of frameIdx for the rAF loop (no stale closure)
  const playingRef = useRef(false);

  const nFrames = frameDelays.length;

  // ── render request → cache key + frame_epochs ────────────────────────────
  const stopRaf = useCallback(() => {
    if (rafRef.current != null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
  }, []);

  const resetPlayback = useCallback(() => {
    stopRaf();
    loopCountRef.current = 0;
    frameIdxRef.current = 0;
    playingRef.current = false;
    setPlaying(false);
    setFrameIdx(0);
    setFrameDelays([]);
    composedRef.current = [];
    delaysRef.current = [];
    setFrameEpochs([]);
    setHoldLen(0);
    setDecoding(false);
    setDecodeMs(null);
  }, [stopRaf]);

  async function render() {
    if (!spec || spec.experiment_set.length === 0) return;
    setPhase("rendering");
    setProgress("queuing…");
    setErr("");
    resetPlayback();
    try {
      const r = await api.gifRender(spec);
      if (r.cached || !r.job_id) {
        setCacheKey(r.cache_key);
        // instant cache hit — there is no job to poll; read the sidecar for frame_epochs.
        try {
          const sc = await api.gifSidecar(r.cache_key);
          setFrameEpochs(sc.frame_epochs ?? []);
          setHoldLen(sc.hold_len ?? 0);
        } catch {
          /* sidecar optional — seek still works by frame index */
        }
        setPhase("ready");
        return;
      }
      poll(r.job_id, r.cache_key);
    } catch (e) {
      setErr(String((e as Error).message));
      setPhase("error");
    }
  }

  function poll(jobId: string, ck: string) {
    if (pollRef.current) window.clearInterval(pollRef.current);
    pollRef.current = window.setInterval(async () => {
      try {
        const s: GifStatus = await api.gifStatus(jobId);
        if (s.status === "rendering") setProgress(`rendering frame ${s.frame ?? "?"}/${s.total ?? "?"}`);
        else if (s.status === "pending") setProgress("pending…");
        else if (s.status === "done") {
          window.clearInterval(pollRef.current!);
          setCacheKey(s.cache_key ?? ck);
          setFrameEpochs(s.frame_epochs ?? []);
          setHoldLen(s.hold_len ?? 0);
          setPhase("ready");
        } else if (s.status === "error") {
          window.clearInterval(pollRef.current!);
          setErr(s.error ?? "render error");
          setPhase("error");
        }
      } catch (e) {
        window.clearInterval(pollRef.current!);
        setErr(String((e as Error).message));
        setPhase("error");
      }
    }, 700);
  }

  // ── re-render when the spec changes ──────────────────────────────────────
  useEffect(() => {
    setCacheKey(null);
    setPhase("idle");
    resetPlayback();
    if (autoRender) void render();
    return () => {
      if (pollRef.current) window.clearInterval(pollRef.current);
      stopRaf();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify(spec)]);

  // ── decode the cached GIF ONCE (FB-R4-01) ────────────────────────────────
  // The decode + full-frame compose is the only heavy step; it runs a SINGLE time per
  // cache key. We first flip `decoding=true` and yield a paint (double rAF) so the
  // "decoding…" state is actually visible, THEN do the (synchronous) decode/compose,
  // cache the per-frame ImageData in a ref, and start playback. Seek/play after this are
  // pure O(1) paints of cached ImageData — never re-decode or re-composite from frame 0.
  useEffect(() => {
    if (phase !== "ready" || !cacheKey) return;
    let cancelled = false;
    setDecoding(true);
    (async () => {
      try {
        const buf = await fetch(api.gifUrl(cacheKey)).then((r) => r.arrayBuffer());
        if (cancelled) return;
        // yield two animation frames so React paints the "decoding…" state before the
        // main thread is occupied by the synchronous decode/compose loop.
        await new Promise<void>((res) => requestAnimationFrame(() => requestAnimationFrame(() => res())));
        if (cancelled) return;
        const t0 = performance.now();
        const gif = parseGIF(buf);
        const decoded = decompressFrames(gif, true) as ParsedGifFrame[];
        const w = gif.lsd.width;
        const h = gif.lsd.height;
        // compose every (possibly partial) frame onto a persistent full-frame canvas,
        // honoring disposal (2 = restore-to-background → clear the patched rect). Each
        // composited frame is cached as ImageData so seek/play are O(1) thereafter.
        const off = document.createElement("canvas");
        off.width = w;
        off.height = h;
        const octx = off.getContext("2d")!;
        const patchCanvas = document.createElement("canvas");
        const pctx = patchCanvas.getContext("2d")!;
        const composed: ImageData[] = [];
        const delays: number[] = [];
        let prevDisposal = 0;
        let prevDims: ParsedGifFrame["dims"] | null = null;
        for (const fr of decoded) {
          if (prevDisposal === 2 && prevDims) {
            octx.clearRect(prevDims.left, prevDims.top, prevDims.width, prevDims.height);
          }
          patchCanvas.width = fr.dims.width;
          patchCanvas.height = fr.dims.height;
          // allocate a fresh ImageData and copy the decoded RGBA patch in (avoids a
          // Uint8ClampedArray<ArrayBufferLike> vs ImageDataArray typing mismatch).
          const patchImg = pctx.createImageData(fr.dims.width, fr.dims.height);
          patchImg.data.set(fr.patch);
          pctx.putImageData(patchImg, 0, 0);
          octx.drawImage(patchCanvas, fr.dims.left, fr.dims.top);
          composed.push(octx.getImageData(0, 0, w, h));
          delays.push(fr.delay ?? 10);
          prevDisposal = fr.disposalType;
          prevDims = fr.dims;
        }
        if (cancelled) return;
        const ms = Math.round(performance.now() - t0);
        composedRef.current = composed;
        delaysRef.current = delays;
        setDims({ width: w, height: h });
        setFrameDelays(delays);
        setDecodeMs(ms);
        setDecoding(false);
        // start the first pass (FB-R3-05a counts 2 passes, then freezes).
        frameIdxRef.current = 0;
        loopCountRef.current = 0;
        setFrameIdx(0);
        playingRef.current = true;
        setPlaying(true);
        // eslint-disable-next-line no-console
        console.debug(`[GifViewer] decoded ${composed.length} frames in ${ms}ms (${w}x${h})`);
      } catch (e) {
        if (!cancelled) {
          setDecoding(false);
          setErr("decode failed: " + String((e as Error).message));
        }
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase, cacheKey]);

  // ── paint a frame onto the visible canvas (O(1) — just the cached ImageData) ─────
  const paintFrame = useCallback(
    (i: number) => {
      const cv = canvasRef.current;
      const img = composedRef.current[i];
      if (!cv || !img) return;
      if (cv.width !== dims.width || cv.height !== dims.height) {
        cv.width = dims.width;
        cv.height = dims.height;
      }
      cv.getContext("2d")!.putImageData(img, 0, 0);
    },
    [dims]
  );

  // repaint whenever the held frame changes (seek + the rAF loop both go through here).
  useEffect(() => {
    paintFrame(frameIdx);
  }, [frameIdx, paintFrame]);

  // ── requestAnimationFrame playback (FB-R4-01) ────────────────────────────
  // ONE rAF loop. It accumulates real elapsed time and advances the frame index only when
  // the current frame's own delay has elapsed — O(1) per tick, no decode/composite, no
  // setTimeout drift. Stops after exactly LOOP_COUNT passes (freeze on the last frame).
  useEffect(() => {
    playingRef.current = playing;
    if (!playing || nFrames === 0) {
      stopRaf();
      return;
    }
    let last = performance.now();
    let acc = 0; // ms accumulated toward the current frame's delay
    const tick = (now: number) => {
      if (!playingRef.current) return;
      acc += now - last;
      last = now;
      const delays = delaysRef.current;
      // advance as many frames as the elapsed time covers (handles tab-throttle catch-up
      // gracefully but still in O(frames-advanced), which is tiny per tick).
      let idx = frameIdxRef.current;
      let guard = 0;
      while (guard++ < delays.length + 2) {
        const delayMs = Math.max(20, (delays[idx] ?? 10) * 10); // gif delay is 1/100 s
        if (acc < delayMs) break;
        acc -= delayMs;
        const next = idx + 1;
        if (next >= delays.length) {
          loopCountRef.current += 1;
          if (loopCountRef.current >= LOOP_COUNT) {
            // FB-R3-05a: after exactly 2 passes, FREEZE on the last frame.
            idx = delays.length - 1;
            frameIdxRef.current = idx;
            setFrameIdx(idx);
            playingRef.current = false;
            setPlaying(false);
            return;
          }
          idx = 0; // start the next pass
        } else {
          idx = next;
        }
      }
      if (idx !== frameIdxRef.current) {
        frameIdxRef.current = idx;
        setFrameIdx(idx);
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => stopRaf();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, nFrames]);

  // ── frame → epoch mapping (collapse the peak-hold tail to the final epoch) ──
  const lastDistinctFrame = useMemo(
    () => Math.max(0, nFrames - 1 - Math.max(0, holdLen - 1)),
    [nFrames, holdLen]
  );
  function epochForFrame(i: number): number | null {
    if (frameEpochs.length === 0) return null;
    const idx = Math.min(i, frameEpochs.length - 1);
    return frameEpochs[idx] ?? null;
  }
  const curEpoch = epochForFrame(frameIdx);
  const finalEpoch = frameEpochs.length ? frameEpochs[frameEpochs.length - 1] : null;

  function play() {
    if (nFrames === 0) return;
    if (frameIdxRef.current >= nFrames - 1) {
      loopCountRef.current = 0;
      frameIdxRef.current = 0;
      setFrameIdx(0);
    }
    playingRef.current = true;
    setPlaying(true);
  }
  function pause() {
    playingRef.current = false;
    setPlaying(false);
  }
  function replay() {
    loopCountRef.current = 0;
    frameIdxRef.current = 0;
    setFrameIdx(0);
    playingRef.current = true;
    setPlaying(true);
  }
  function seek(i: number) {
    playingRef.current = false;
    setPlaying(false);
    frameIdxRef.current = i;
    setFrameIdx(i); // the paint effect repaints from the cached ImageData (O(1))
  }

  // FB-R3-04: download the currently-held frame as a PNG (the same canvas the slider holds).
  function downloadFrame() {
    const cv = canvasRef.current;
    if (!cv) return;
    cv.toBlob((blob) => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      const ep = curEpoch != null ? `epoch_${curEpoch}` : `frame_${frameIdx}`;
      a.href = url;
      a.download = `${spec?.story ?? "gif"}_${ep}.png`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    }, "image/png");
  }

  const seekable = nFrames > 0;

  return (
    <div className="col" style={{ gap: 8 }}>
      {title && <div className="card-title">{title}</div>}
      <div
        style={{
          height,
          background: "var(--surface-2)",
          border: "1px solid var(--border)",
          borderRadius: "var(--radius-md)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          overflow: "hidden",
        }}
      >
        {phase === "idle" && !spec && <span className="subtle">Pick a story + run set, then render.</span>}
        {phase === "idle" && spec && !autoRender && (
          <button className="btn primary sm" onClick={() => void render()}>
            ► render
          </button>
        )}
        {phase === "rendering" && (
          <div className="async-state" style={{ border: 0, background: "transparent" }}>
            <span className="glyph">◌</span>
            <div>{progress || "rendering…"}</div>
            <div className="async-skel" style={{ width: 200, height: 6 }} />
          </div>
        )}
        {phase === "error" && (
          <div className="async-state error" style={{ border: 0 }}>
            <span className="glyph">⚠</span>
            <div>{err}</div>
          </div>
        )}
        {phase === "ready" && (
          <canvas
            ref={canvasRef}
            style={{ maxWidth: "100%", maxHeight: height, display: seekable ? "block" : "none" }}
          />
        )}
        {phase === "ready" && decoding && (
          <div className="async-state" style={{ border: 0, background: "transparent" }}>
            <span className="glyph">◌</span>
            <div>decoding… (one-time)</div>
            <div className="async-skel" style={{ width: 200, height: 6 }} />
          </div>
        )}
      </div>

      {/* seek slider — drag to hold any frame; the readout maps to the eval epoch and
          collapses the peak-hold tail to the final epoch (R3-SR-03/05). Each seek is an
          O(1) paint of the cached ImageData — no re-decode, no re-composite. */}
      {seekable && (
        <div className="col" style={{ gap: 4 }}>
          <input
            type="range"
            min={0}
            max={nFrames - 1}
            step={1}
            value={frameIdx}
            onChange={(e) => seek(Number(e.target.value))}
            style={{ width: "100%" }}
            aria-label="seek frame"
          />
          <div className="subtle" style={{ fontSize: "var(--fs-small)" }}>
            frame {frameIdx + 1}/{nFrames}
            {curEpoch != null && (
              <>
                {" · "}
                {frameIdx > lastDistinctFrame && finalEpoch != null ? (
                  <span title="peak-hold tail — frozen on the final epoch">epoch {finalEpoch} (hold)</span>
                ) : (
                  <span>epoch {curEpoch}</span>
                )}
              </>
            )}
            {decodeMs != null && (
              <span title="one-time client decode + frame cache; play/seek are O(1) thereafter">
                {" · "}
                decoded once in {decodeMs} ms
              </span>
            )}
          </div>
        </div>
      )}

      <div className="row wrap" style={{ gap: 8 }}>
        {!playing ? (
          <button className="btn primary sm" disabled={!seekable} onClick={play} title="play (2 passes then freeze)">
            ► play
          </button>
        ) : (
          <button className="btn sm" onClick={pause} title="pause">
            ❚❚ pause
          </button>
        )}
        <button className="btn sm" disabled={!seekable} onClick={replay} title="restart from frame 0 (2 passes)">
          ↻ replay
        </button>
        <button className="btn sm" onClick={() => void render()} disabled={!spec}>
          ⟳ re-render
        </button>
        {/* FB-R3-04: download the held frame's PNG (same canvas as the slider). */}
        <button className="btn sm" disabled={!seekable} onClick={downloadFrame} title="download the currently held frame as PNG">
          ↓ PNG (this epoch)
        </button>
        {cacheKey && (
          <a className="btn sm" href={api.gifUrl(cacheKey)} download>
            ↓ export GIF
          </a>
        )}
        {cacheKey && (
          <button className="btn sm" onClick={() => navigator.clipboard?.writeText(cacheKey)} title={cacheKey}>
            ⧉ copy cache key
          </button>
        )}
      </div>
    </div>
  );
}
