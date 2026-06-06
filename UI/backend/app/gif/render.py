"""GIF frame renderers (matplotlib Agg -> RGB frames -> imageio GIF).

backend-design.md §6.3/§6.4. CPU-only, headless (``Agg``), no ffmpeg. Every renderer:
  * reads the warmup boundary from ``config.teacher_only_warmup_epochs`` of THAT leaf
    and greys/hatches the pre-warmup epoch span for post-warmup metrics (GIF-INV-1);
  * takes the "good" hue + race sort order from the registry ``direction`` (GIF-INV-2);
  * draws markers+gaps for sparse metrics rather than interpolating (GIF-INV-6);
  * animates an inferred metric dotted + badged (GIF-INV-5);
  * holds peak frames before looping.

Frames are produced one at a time; the figure is closed each frame so peak memory is
bounded (one figure + the source series). NPZ-backed drift loads ONE NPZ per frame via
the lazy reader and discards it (GIF-INV-3); a BadZipFile frame is skipped (GIF-INV-4).
"""
from __future__ import annotations

import io
import math
from typing import Any, Callable, Optional

import matplotlib

matplotlib.use("Agg")  # headless, no display, no GPU
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# FB-R5-01: the color-norm modes the feature heatmap (story 5, signal_lives) supports.
# A heavy-tailed per-feature recon distribution (most features ≈0.001, a few ≈0.006)
# washes out on a LINEAR norm — the ≈0.001 bulk maps to ~17% color (near-black) and
# reads as "no influence" (over-exaggerated contrast). ``sqrt`` (PowerNorm γ=0.5) lifts
# the low-mid features to a clearly non-black color (0.001/0.006 → ~41% vs ~17% linear)
# WITHOUT inverting the ordering; ``log`` (LogNorm) is offered for an even stronger
# low-end reveal. Default is ``sqrt`` (faithful, not over-corrected). The single source
# of truth so render + the /gif/stories manifest never diverge.
HEATMAP_SCALES = ("linear", "sqrt", "log")
DEFAULT_HEATMAP_SCALE = "sqrt"

# direction -> a colorblind-safe "good" hue (Okabe-Ito derived); neutral = slate.
_GOOD_HUE = {"up": "#0072B2", "down": "#009E73", "neutral": "#6c7a89"}
_WARN_HUE = "#D55E00"
_GREY = "#B8BEC6"
_BG = "#FFFFFF"
# FB-15: bumped, bounded DPI + pixel size for crisp output on high-DPI screens.
# 140 DPI × (8.0, 4.8) in ≈ 1120×672 px/frame (~3× the old 648×387) — still
# GIF-reasonable. The per-frame figure-close memory bound + the frame cap keep total
# bytes/time bounded; compare_grid scales these per-panel below.
_FIG_DPI = 140
_FIG_SIZE = (8.0, 4.8)

# FB-R4-01: a LIGHTER playback profile for the HIGH-FRAME-COUNT per-eval-epoch line/
# race/grid stories. A 100-eval-epoch leaf renders ~110 frames; at 140 DPI / 1120×672
# that is ~9 s server render (≈half matplotlib draw, ≈half imageio palette-quantize) and
# ~330 MB of client ImageData (100 frames × 1120×672 × 4 B) — the FB-R4-01 slowness.
# 96 DPI × (7.0, 4.2) in ≈ 672×403 px/frame cuts the server render to ~5 s and the client
# ImageData to ~120 MB while KEEPING every distinct eval epoch as its own frame (frame
# count unchanged ⇒ the sidecar frame_epochs still maps every 5-step epoch; the 5-epoch
# seek granularity is preserved). The low-frame-count static stories (hist drift, feature
# heatmap, bar race, note) KEEP the crisp 140 DPI — they are not frame-bound. The DPI
# override flows through ``_new_fig``/the renderers as a per-render argument (no global
# mutation, so concurrent renders at different profiles never race).
_PLAYBACK_DPI = 96
_PLAYBACK_SIZE = (7.0, 4.2)

# FB-3 (line_race) distinct categorical palette (Okabe-Ito + extras) — also reused for
# FB-10 multi-line GIF-1 colors so each selected metric/leaf line is legibly distinct.
_CATEGORICAL = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
                "#56B4E9", "#F0E442", "#999999", "#117733", "#882255"]


def _new_fig(*, dpi: Optional[int] = None, figsize: Optional[tuple] = None):
    fig = plt.figure(figsize=figsize or _FIG_SIZE, dpi=dpi or _FIG_DPI, facecolor=_BG)
    ax = fig.add_subplot(111)
    ax.set_facecolor(_BG)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(True, alpha=0.18, linewidth=0.6)
    return fig, ax


def playback_profile() -> tuple[int, tuple]:
    """FB-R4-01: the lighter (dpi, figsize) used for the high-frame-count per-eval-epoch
    line/race/grid stories so a ~110-frame GIF renders fast + stays light to decode."""
    return _PLAYBACK_DPI, _PLAYBACK_SIZE


def _fig_to_rgb(fig) -> np.ndarray:
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return rgba[..., :3].copy()


def _direction_hue(meta: dict, *, negative_value: Optional[float] = None) -> str:
    d = meta.get("direction", "neutral")
    if d == "up" and negative_value is not None and negative_value < 0:
        return _WARN_HUE  # e.g. negative SNR — real signal, warn-hue
    return _GOOD_HUE.get(d, _GOOD_HUE["neutral"])


def _is_post_warmup(meta: dict) -> bool:
    return (meta.get("phase_validity") == "post-warmup"
            or bool(meta.get("viz_hint", {}).get("mask_prewarmup")))


def _frame_indices(n_points: int, max_frames: Optional[int]) -> list[int]:
    """The cumulative reveal frame schedule (1..n), capped to ~max_frames, + peak hold.

    Each entry is an ``upto`` count: the frame reveals the first ``upto`` epochs (so the
    on-frame "current" epoch is ``epochs[upto-1]``). When ``max_frames`` is ``None`` (or
    ``>= n_points``), EVERY point is a frame (no implicit stride) — the per-eval-epoch
    grid (FB-R3-09): one frame per saved eval epoch (5,10,15,…), not every 2nd.
    """
    if n_points <= 0:
        return []
    cap = max_frames or n_points
    if n_points <= cap:
        upto = list(range(1, n_points + 1))
    else:
        stride = math.ceil(n_points / cap)
        upto = sorted(set(list(range(stride, n_points + 1, stride)) + [n_points]))
    return upto


def _stagger_marker_labels(
    markers: list[dict[str, Any]], *, ylo: float, yhi: float
) -> dict[int, float]:
    """E-2 / GIF-7 (A-3): vertical LABEL positions for horizontal ``best_markers`` so
    near-coincident ceiling/best-epoch labels never collide into garbled text.

    Markers anchored at the SAME (or near-same) y-value would draw their text on top of
    each other (the old ``ax.text(epochs[-1], m["value"], …)`` placed every label exactly
    on its rule). We sort the markers by value, walk upward, and whenever the next label
    would sit within ``min_gap`` of the previous one we push it up by ``min_gap`` — a
    deterministic stack — so the LABEL y is offset from the RULE while the rule itself
    stays exactly on the value. Returns ``{marker_index: label_y}`` (the rule is still
    drawn at ``m["value"]``; only the text moves). Computed once from the fixed y-range,
    so every animation frame places labels identically (no per-frame jitter).
    """
    span = (yhi - ylo) or 1.0
    # a label band ≈ one 7-pt text line in DATA units; tuned so 2–3 stacked labels fit.
    min_gap = span * 0.052
    valued = [(i, m) for i, m in enumerate(markers)
              if isinstance(m.get("value"), (int, float))]
    valued.sort(key=lambda t: t[1]["value"])
    placed: dict[int, float] = {}
    prev_y: Optional[float] = None
    for idx, m in valued:
        y = float(m["value"])
        if prev_y is not None and y - prev_y < min_gap:
            y = prev_y + min_gap        # push this label up off the previous one
        # keep the label inside the axes (clamp to just under the top so it stays legible)
        y = min(y, yhi - span * 0.01)
        placed[idx] = y
        prev_y = y
    return placed


def frame_epoch_schedule(
    epochs: list[int], *, max_frames: Optional[int], hold: int
) -> tuple[list[int], int]:
    """Return ``(per_frame_epochs, hold_len)`` for a draw-on line/race/grid story.

    ``per_frame_epochs[i]`` is the eval-epoch SHOWN on rendered frame ``i`` — the
    renderer draws up to ``epochs[upto-1]`` for each scheduled ``upto`` (the reveal
    schedule from :func:`_frame_indices`), then appends a peak-hold of ``hold`` extra
    copies of the FINAL epoch (the encoder's freeze dwell). This mirrors EXACTLY the
    schedule the renderers build (``_frame_indices(n, max_frames) + [n]*hold``), so the
    frontend canvas player can map frame index → epoch and collapse the trailing
    ``hold_len`` frames to the final epoch (R3-SR-03 / R3-SR-05).

    The "distinct seekable epochs" are the reveal frames; the hold tail repeats the last
    epoch and should collapse to a single seek stop.
    """
    n = len(epochs)
    if n == 0:
        return [], 0
    schedule = _frame_indices(n, max_frames)
    per_frame = [epochs[min(upto, n) - 1] for upto in schedule]
    hold_len = max(0, int(hold))
    per_frame += [epochs[-1]] * hold_len
    return per_frame, hold_len


# ── story 1/2/6 : animated line (draw-on) ────────────────────────────────────
def render_lines(
    *,
    title: str,
    epochs: list[int],
    lines: list[dict[str, Any]],       # {label, values[], meta, style?}
    warmup_epochs: Optional[int],
    best_epoch: Optional[int] = None,
    best_markers: Optional[list[dict[str, Any]]] = None,  # P3-01 score-variant scalars
    y_unit: Optional[str] = None,
    max_frames: Optional[int] = None,
    hold: int = 10,
    dpi: Optional[int] = None,
    figsize: Optional[tuple] = None,
    signed_axis: bool = False,
) -> list[np.ndarray]:
    """Animated multi-line draw-on. Pre-warmup span greyed for post-warmup lines.

    ``best_markers`` (story 6 / P3-01): a list of {label, value, meta} drawn as a fixed
    horizontal marker at the best epoch — these are BEST-EPOCH SCALARS (student/disc),
    NEVER per-epoch trajectories.

    ``signed_axis`` (E-2 / GIF-8): the values are a SIGNED Δ (A − B) that may be ±; the
    y-range must straddle zero and NEVER be clamped to a [0,1] "score" range. When set, the
    range is computed symmetrically around 0 from the data extent so the sign is legible.
    """
    n = len(epochs)
    if n == 0 or not lines:
        return []
    # global y-range
    allv = [v for ln in lines for v in ln["values"] if isinstance(v, (int, float))]
    for m in (best_markers or []):
        if isinstance(m.get("value"), (int, float)):
            allv.append(m["value"])
    if not allv:
        return []
    ylo, yhi = min(allv), max(allv)
    if signed_axis:
        # E-2 / GIF-8: a signed Δ axis — straddle zero, label it as a signed range, and
        # NEVER apply the [0,1] score clamp (a diff of two scores is not itself a score).
        mag = max(abs(ylo), abs(yhi), 1e-6)
        ylo, yhi = -mag, mag
    else:
        score_like = all((ln["meta"].get("unit") or "").startswith("score") or
                         "score" in (ln["meta"].get("family") or "") for ln in lines)
        if score_like and 0.0 <= ylo and yhi <= 1.0:
            ylo, yhi = 0.0, 1.0
    pad = (yhi - ylo) * 0.08 or 0.05
    ylo, yhi = ylo - pad, yhi + pad

    frames: list[np.ndarray] = []
    schedule = _frame_indices(n, max_frames)
    schedule += [n] * hold  # peak hold

    for upto in schedule:
        fig, ax = _new_fig(dpi=dpi, figsize=figsize)
        ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
        ax.set_xlim(epochs[0], epochs[-1])
        ax.set_ylim(ylo, yhi)
        ax.set_xlabel("eval epoch")
        if y_unit:
            ax.set_ylabel(y_unit)

        if warmup_epochs and warmup_epochs > 0:
            ax.axvspan(epochs[0], min(warmup_epochs, epochs[-1]), color=_GREY,
                       alpha=0.16, hatch="///", zorder=0)
            if epochs[0] <= warmup_epochs <= epochs[-1]:
                ax.axvline(warmup_epochs, color="#555", ls="--", lw=1.0, zorder=1)
                ax.text(warmup_epochs, yhi, " student joins", fontsize=8,
                        va="top", ha="left", color="#555")

        multi = len(lines) > 1
        for li, ln in enumerate(lines):
            vals = ln["values"][:upto]
            xs = epochs[:upto]
            meta = ln["meta"]
            neg = next((v for v in vals if isinstance(v, (int, float)) and v < 0), None)
            # FB-10: when MULTIPLE lines are drawn, give each a DISTINCT categorical
            # color so they are legible (previously all shared one direction hue →
            # indistinguishable blue). A SINGLE line keeps the registry direction hue
            # (F4); negative values still warn-hue; inferred still dotted.
            if multi:
                hue = _WARN_HUE if (neg is not None) else _CATEGORICAL[li % len(_CATEGORICAL)]
            else:
                hue = _direction_hue(meta, negative_value=neg)
            ls = ":" if (meta.get("inferred") or ln.get("style") == "dotted") else "-"
            # phase mask: split into pre/post warmup; pre-warmup of a post-warmup
            # metric is greyed, NOT drawn as a real value at 0.
            xs_a = np.asarray(xs, dtype=float)
            ys_a = np.asarray([v if isinstance(v, (int, float)) else np.nan
                               for v in vals], dtype=float)
            if _is_post_warmup(meta) and warmup_epochs:
                pre = xs_a <= warmup_epochs
                ax.plot(xs_a[pre], ys_a[pre], color=_GREY, ls=ls, lw=1.3, alpha=0.5)
                ax.plot(xs_a[~pre], ys_a[~pre], color=hue, ls=ls, lw=2.0,
                        label=ln["label"])
            else:
                ax.plot(xs_a, ys_a, color=hue, ls=ls, lw=2.0, label=ln["label"])
            # current-epoch marker
            if len(xs) and isinstance(vals[-1], (int, float)):
                ax.scatter([xs[-1]], [vals[-1]], color=hue, s=22, zorder=5)

        # E-2 / GIF-7 (A-3): precompute STAGGERED label y-positions so near-coincident
        # ceiling/best-epoch labels don't overlap into garbled text. The rule stays on the
        # value; only the TEXT y is offset (with a faint leader when it moves).
        label_y = _stagger_marker_labels(best_markers or [], ylo=ylo, yhi=yhi)
        for mi, m in enumerate(best_markers or []):
            if not isinstance(m.get("value"), (int, float)):
                continue
            # FB-R4b-02: distinguish a TRUE ceiling (max of a per-epoch series the line
            # can never exceed) from a single best-epoch SCALAR observation. A
            # ``kind == "ceiling"`` marker is the max of the variant's OWN per-epoch line
            # → drawn as a SOLID horizontal rule (the line touches but never crosses it).
            # A ``kind == "best_epoch"`` marker is a single best-epoch scalar (student/disc
            # — no per-epoch series, P3-01) → drawn as a DASHED rule + an end dot so it
            # reads as one observed point, NOT a bound. Default (no kind) keeps the legacy
            # dash-dot rule. Toggling which variants are passed is done upstream
            # (``show_ceilings``); the renderer just draws whatever it is given.
            kind = m.get("kind")
            hue = _GOOD_HUE.get(m["meta"].get("direction", "neutral"), "#888")
            if kind == "ceiling":
                ax.axhline(m["value"], color=hue, ls="-", lw=1.2, alpha=0.75)
            elif kind == "best_epoch":
                ax.axhline(m["value"], color=hue, ls=":", lw=1.0, alpha=0.5)
                ax.scatter([epochs[-1]], [m["value"]], color=hue, s=20,
                           marker="D", zorder=6, alpha=0.8)
            else:
                ax.axhline(m["value"], color=hue, ls="-.", lw=1.0, alpha=0.55)
            ly = label_y.get(mi, m["value"])
            # a faint leader from the (offset) label back to its rule, so a staggered
            # label is unambiguously tied to its value.
            if abs(ly - m["value"]) > (yhi - ylo) * 1e-3:
                ax.plot([epochs[-1], epochs[-1]], [m["value"], ly], color=hue,
                        lw=0.6, alpha=0.4, zorder=4)
            ax.text(epochs[-1], ly, f' {m["label"]}', fontsize=7,
                    va="center", ha="right", color="#444", zorder=7,
                    bbox=dict(facecolor=_BG, alpha=0.6, edgecolor="none", pad=0.5))

        if best_epoch and epochs[0] <= best_epoch <= epochs[-1] and upto >= n:
            ax.scatter([best_epoch], [yhi - pad], marker="*", s=160,
                       color="#E1A100", zorder=6)

        ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
        frames.append(_fig_to_rgb(fig))
    return frames


# ── story 8 : line race (FB-3) — per-epoch overtaking, distinct per-leaf colors ─
def render_line_race(
    *,
    title: str,
    epochs: list[int],
    lines: list[dict[str, Any]],       # {label, values[]}  (aligned to ``epochs``)
    metric_display: str,
    direction: str = "up",
    y_unit: Optional[str] = None,
    max_frames: Optional[int] = None,
    hold: int = 12,
    dpi: Optional[int] = None,
    figsize: Optional[tuple] = None,
) -> list[np.ndarray]:
    """Animate N leaf lines climbing over the eval-epoch axis (value AT each epoch).

    Each frame reveals every line UP TO the animated epoch; the marker sits on the
    value AT that epoch (real per-eval-epoch series — never best-epoch-fixed). Distinct
    categorical colors per leaf (overtaking is legible). Alignment is by epoch NUMBER;
    a gap (``None``) breaks the line, never a fabricated 0. Non-looping is applied at
    encode time (M-3); a peak-hold dwells on the final epoch."""
    n = len(epochs)
    if n == 0 or not lines:
        return []
    allv = [v for ln in lines for v in ln["values"] if isinstance(v, (int, float))]
    if not allv:
        return []
    ylo, yhi = min(allv), max(allv)
    if 0.0 <= ylo and yhi <= 1.0:
        ylo, yhi = 0.0, 1.0
    pad = (yhi - ylo) * 0.08 or 0.05
    ylo, yhi = ylo - pad, yhi + pad

    colors = [_CATEGORICAL[i % len(_CATEGORICAL)] for i in range(len(lines))]
    frames: list[np.ndarray] = []
    schedule = _frame_indices(n, max_frames) + [n] * hold
    for upto in schedule:
        fig, ax = _new_fig(dpi=dpi, figsize=figsize)
        ax.set_title(f"{title}  ·  epoch {epochs[min(upto, n) - 1]}", fontsize=12,
                     fontweight="bold", loc="left")
        ax.set_xlim(epochs[0], epochs[-1])
        ax.set_ylim(ylo, yhi)
        ax.set_xlabel("eval epoch")
        ax.set_ylabel(metric_display + (f"  [{y_unit}]" if y_unit else ""))
        for li, ln in enumerate(lines):
            xs = np.asarray(epochs[:upto], dtype=float)
            ys = np.asarray([v if isinstance(v, (int, float)) else np.nan
                             for v in ln["values"][:upto]], dtype=float)
            ax.plot(xs, ys, color=colors[li], lw=2.0, label=ln["label"])
            # marker on the value AT the animated epoch (last finite ≤ current).
            if ys.size and np.isfinite(ys[-1]):
                ax.scatter([xs[-1]], [ys[-1]], color=colors[li], s=24, zorder=5)
        order = "lower wins" if direction == "down" else "higher wins"
        ax.legend(loc="lower right", fontsize=8, framealpha=0.85, title=order)
        frames.append(_fig_to_rgb(fig))
    return frames


# ── synchronized side-by-side compare (one combined GIF, lockstep playhead) ───
def render_compare_grid(
    *,
    title: str,
    epochs: list[int],
    panels: list[dict[str, Any]],     # {panel_title, warmup_epochs, lines:[{label,values,meta}]}
    y_unit: Optional[str] = None,
    max_frames: Optional[int] = None,
    hold: int = 10,
    dpi: Optional[int] = None,
    figsize: Optional[tuple] = None,
) -> list[np.ndarray]:
    """Render N panels as ONE combined GIF revealed in lockstep — a single shared frame
    index drives every panel, so the playhead is synchronized BY CONSTRUCTION (no client
    multi-GIF drift; frontend-design §6.3, review-P4-frontend F-03).

    Each panel keeps its own warmup span + the registry direction hue; all panels share
    the global y-range and the same per-frame reveal count.
    """
    n = len(epochs)
    if n == 0 or not panels:
        return []
    allv = [v for p in panels for ln in p["lines"]
            for v in ln["values"] if isinstance(v, (int, float))]
    if not allv:
        return []
    ylo, yhi = min(allv), max(allv)
    score_like = all((ln["meta"].get("unit") or "").startswith("score") or
                     "score" in (ln["meta"].get("family") or "")
                     for p in panels for ln in p["lines"] if p["lines"])
    if score_like and 0.0 <= ylo and yhi <= 1.0:
        ylo, yhi = 0.0, 1.0
    pad = (yhi - ylo) * 0.08 or 0.05
    ylo, yhi = ylo - pad, yhi + pad

    ncol = min(len(panels), 2)
    nrow = math.ceil(len(panels) / ncol)
    frames: list[np.ndarray] = []
    schedule = _frame_indices(n, max_frames) + [n] * hold

    # FB-15: compare_grid combines N panels into ONE figure — cap its DPI slightly
    # below the single-panel target so a many-panel grid stays bounded in px/bytes
    # while still visibly crisper than the old 90 DPI. FB-R4-01: when a lighter playback
    # ``dpi``/``figsize`` is passed (high-frame-count per-eval-epoch grid), honor it so
    # the multi-panel combined GIF also renders fast + stays light to decode.
    base_dpi = dpi or _FIG_DPI
    base_size = figsize or _FIG_SIZE
    grid_dpi = min(base_dpi, 120)
    for upto in schedule:
        fig = plt.figure(figsize=(base_size[0] * ncol * 0.78,
                                  base_size[1] * nrow * 0.78),
                         dpi=grid_dpi, facecolor=_BG)
        fig.suptitle(f"{title}  ·  epoch {epochs[min(upto, n) - 1]}",
                     fontsize=12, fontweight="bold", x=0.02, ha="left")
        for pi, panel in enumerate(panels):
            ax = fig.add_subplot(nrow, ncol, pi + 1)
            ax.set_facecolor(_BG)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            ax.grid(True, alpha=0.18, linewidth=0.6)
            ax.set_title(panel.get("panel_title", ""), fontsize=9, loc="left")
            ax.set_xlim(epochs[0], epochs[-1])
            ax.set_ylim(ylo, yhi)
            if y_unit:
                ax.set_ylabel(y_unit, fontsize=8)
            wu = panel.get("warmup_epochs")
            if wu and wu > 0:
                ax.axvspan(epochs[0], min(wu, epochs[-1]), color=_GREY,
                           alpha=0.16, hatch="///", zorder=0)
                if epochs[0] <= wu <= epochs[-1]:
                    ax.axvline(wu, color="#555", ls="--", lw=0.9, zorder=1)
            for ln in panel["lines"]:
                vals = ln["values"][:upto]
                xs = epochs[:upto]
                meta = ln["meta"]
                neg = next((v for v in vals if isinstance(v, (int, float)) and v < 0), None)
                hue = _direction_hue(meta, negative_value=neg)
                ls = ":" if meta.get("inferred") else "-"
                xs_a = np.asarray(xs, dtype=float)
                ys_a = np.asarray([v if isinstance(v, (int, float)) else np.nan
                                   for v in vals], dtype=float)
                if _is_post_warmup(meta) and wu:
                    pre = xs_a <= wu
                    ax.plot(xs_a[pre], ys_a[pre], color=_GREY, ls=ls, lw=1.2, alpha=0.5)
                    ax.plot(xs_a[~pre], ys_a[~pre], color=hue, ls=ls, lw=1.8,
                            label=ln["label"])
                else:
                    ax.plot(xs_a, ys_a, color=hue, ls=ls, lw=1.8, label=ln["label"])
                if len(xs) and isinstance(vals[-1], (int, float)):
                    ax.scatter([xs[-1]], [vals[-1]], color=hue, s=18, zorder=5)
            ax.legend(loc="lower right", fontsize=7, framealpha=0.85)
            ax.tick_params(labelsize=7)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        frames.append(_fig_to_rgb(fig))
    return frames


# ── story 4 : bar-chart race ──────────────────────────────────────────────────
def render_bar_race(
    *,
    title: str,
    epochs: list[int],
    bars: list[dict[str, Any]],        # {label, values[], color}
    direction: str,
    metric_display: str,
    max_frames: Optional[int] = None,
    hold: int = 12,
    dpi: Optional[int] = None,
    figsize: Optional[tuple] = None,
) -> list[np.ndarray]:
    """Bars reorder each epoch by best-so-far; ↓-metrics sort 'lower wins' (legend)."""
    n = len(epochs)
    if n == 0 or not bars:
        return []
    lower_wins = direction == "down"
    # precompute best-so-far per bar
    running: list[list[Optional[float]]] = []
    for b in bars:
        cur = None
        out: list[Optional[float]] = []
        for v in b["values"]:
            if isinstance(v, (int, float)):
                if cur is None:
                    cur = v
                else:
                    cur = min(cur, v) if lower_wins else max(cur, v)
            out.append(cur)
        running.append(out)
    allv = [v for r in running for v in r if isinstance(v, (int, float))]
    if not allv:
        return []
    vmax = max(allv) * 1.05
    vmin = min(0.0, min(allv))

    frames: list[np.ndarray] = []
    schedule = _frame_indices(n, max_frames) + [n] * hold
    for upto in schedule:
        i = upto - 1
        vals = [(bars[j]["label"], running[j][i], bars[j]["color"])
                for j in range(len(bars))]
        vals = [(lbl, (v if isinstance(v, (int, float)) else vmin), c) for lbl, v, c in vals]
        vals.sort(key=lambda t: t[1], reverse=not lower_wins)
        fig, ax = _new_fig(dpi=dpi, figsize=figsize)
        ax.set_title(f"{title}  ·  epoch {epochs[i]}", fontsize=12,
                     fontweight="bold", loc="left")
        ypos = list(range(len(vals)))[::-1]
        ax.barh(ypos, [v for _, v, _ in vals],
                color=[c for _, _, c in vals], alpha=0.9)
        ax.set_yticks(ypos)
        ax.set_yticklabels([lbl for lbl, _, _ in vals], fontsize=8)
        ax.set_xlim(vmin, vmax)
        order = "lower wins" if lower_wins else "higher wins"
        ax.set_xlabel(f"{metric_display}  ({order})")
        for y, (_, v, _) in zip(ypos, vals):
            ax.text(v, y, f" {v:.3f}", va="center", fontsize=7)
        frames.append(_fig_to_rgb(fig))
    return frames


# ── story 3 : histogram drift (NPZ-backed OR NPZ-free ratio drift) ────────────
def render_hist_drift(
    *,
    title: str,
    frame_provider: Callable[[int], Optional[dict]],  # i -> {normal_hist, anomaly_hist, bins, snr, epoch} or None
    n_frames: int,
    hold: int = 10,
) -> list[np.ndarray]:
    """Animate two histograms (normal vs anomaly) pulling apart. ``frame_provider``
    returns the per-frame histogram (loading ONE NPZ at a time); None => skip frame.
    """
    frames: list[np.ndarray] = []
    order = list(range(n_frames))
    if order:
        order += [order[-1]] * hold
    last_good = None
    for i in order:
        data = frame_provider(i)
        if data is None:
            if last_good is None:
                continue
            data = last_good
        last_good = data
        bins = np.asarray(data["bins"], dtype=float)
        centers = (bins[:-1] + bins[1:]) / 2 if len(bins) > 1 else bins
        nh = np.asarray(data["normal_hist"], dtype=float)
        ah = np.asarray(data["anomaly_hist"], dtype=float)
        snr = data.get("snr")
        fig, ax = _new_fig()
        ep = data.get("epoch")
        ax.set_title(f"{title}" + (f"  ·  epoch {ep}" if ep is not None else ""),
                     fontsize=12, fontweight="bold", loc="left")
        w = (centers[1] - centers[0]) if len(centers) > 1 else 1.0
        ax.bar(centers, nh, width=w, color=_GOOD_HUE["down"], alpha=0.55, label="normal")
        ax.bar(centers, ah, width=w, color=_WARN_HUE, alpha=0.55, label="anomaly")
        ax.set_xlabel("score"); ax.set_ylabel("count")
        if snr is not None:
            hue = _WARN_HUE if snr < 0 else _GOOD_HUE["up"]
            badge = "anti-correlated" if snr < 0 else "separating"
            ax.text(0.98, 0.95, f"SNR={snr:.3f}  ({badge})", transform=ax.transAxes,
                    ha="right", va="top", fontsize=9, color=hue, fontweight="bold")
        ax.legend(loc="upper left", fontsize=8)
        frames.append(_fig_to_rgb(fig))
    return frames


# ── story 5 : per-feature heatmap reveal ──────────────────────────────────────
def _heatmap_norm(scale: str, vmin: float, vmax: float, *, floor: float):
    """FB-R5-01: build the color NORM for the feature heatmap from ``scale``.

    Returns a matplotlib ``Normalize`` subclass that maps the REAL data range
    [vmin, vmax] (the p2/p98 robust clip) onto [0,1] color, NON-LINEARLY for sqrt/log so
    small-but-nonzero features are visibly distinct from the near-zero floor. The colorbar
    stays labelled in REAL data values because the norm is passed to ``imshow`` (matplotlib
    places ticks in data space and only the COLOR mapping is transformed). Guards:
      * sqrt  → PowerNorm(γ=0.5) — gentle, faithful (default); the [0,1] reveal is sqrt.
      * log   → LogNorm; vmin is floored strictly >0 (LogNorm requires vmin>0). ``floor``
                is a tiny positive epsilon derived from the finite data so an all-≤0 or
                degenerate range degrades to a Normalize rather than raising.
      * linear→ the original Normalize (the unchanged behaviour).
    Any unknown scale falls back to linear (defensive; the API validates upstream).
    """
    if vmax <= vmin:
        # degenerate range (constant matrix slice) — a plain Normalize avoids div-by-zero
        # in PowerNorm/LogNorm; the heatmap reads as a single uniform color (correct).
        return mcolors.Normalize(vmin=vmin, vmax=vmax)
    if scale == "sqrt":
        return mcolors.PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)
    if scale == "log":
        lo = vmin if vmin > 0 else floor
        if not (lo > 0) or vmax <= lo:
            # cannot form a valid positive log range → fall back to linear (faithful, no crash)
            return mcolors.Normalize(vmin=vmin, vmax=vmax)
        return mcolors.LogNorm(vmin=lo, vmax=vmax)
    return mcolors.Normalize(vmin=vmin, vmax=vmax)  # linear (and any unknown scale)


def render_feature_heatmap(
    *,
    title: str,
    epochs: list[int],
    matrix: list[list[Optional[float]]],   # [epoch][feature]
    warmup_epochs: Optional[int],
    mask_prewarmup: bool,
    max_frames: Optional[int] = None,
    hold: int = 8,
    heatmap_scale: str = DEFAULT_HEATMAP_SCALE,
) -> list[np.ndarray]:
    """Reveal a feature×epoch heatmap column-by-column over training.

    FB-R5-01: ``heatmap_scale`` ∈ {linear, sqrt, log} selects the COLOR norm so a
    heavy-tailed per-feature recon distribution does not wash out (sqrt is the faithful
    default — see :data:`HEATMAP_SCALES`). The p2/p98 robust clip + cmap=magma are kept;
    only the value→color transform changes, and the colorbar keeps REAL data values.
    """
    n = len(epochs)
    if n == 0 or not matrix:
        return []
    arr = np.array([[v if isinstance(v, (int, float)) else np.nan for v in row]
                    for row in matrix], dtype=float)  # [epoch, feat]
    if arr.size == 0:
        return []
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return []
    scale = heatmap_scale if heatmap_scale in HEATMAP_SCALES else DEFAULT_HEATMAP_SCALE
    vmin, vmax = float(np.nanpercentile(finite, 2)), float(np.nanpercentile(finite, 98))
    # a tiny strictly-positive floor for LogNorm when p2 ≤ 0 (heavy-tailed data can have a
    # zero/near-zero floor): the smallest positive finite value, else a tiny epsilon.
    pos = finite[finite > 0]
    floor = float(pos.min()) if pos.size else 1e-12
    norm = _heatmap_norm(scale, vmin, vmax, floor=floor)
    nfeat = arr.shape[1]
    frames: list[np.ndarray] = []
    schedule = _frame_indices(n, max_frames) + [n] * hold
    for upto in schedule:
        fig, ax = _new_fig()
        ax.set_title(f"{title}  ·  epoch {epochs[upto - 1]}", fontsize=12,
                     fontweight="bold", loc="left")
        shown = arr[:upto].T  # [feat, epoch]
        # FB-R5-01: pass the NORM (not vmin/vmax) so the color mapping is non-linear while
        # the colorbar ticks stay in REAL data values.
        im = ax.imshow(shown, aspect="auto", origin="lower", cmap="magma",
                       norm=norm,
                       extent=[epochs[0], epochs[upto - 1], 0, nfeat])
        if mask_prewarmup and warmup_epochs and epochs[0] <= warmup_epochs <= epochs[-1]:
            ax.axvline(warmup_epochs, color="#fff", ls="--", lw=1.0)
        ax.set_xlabel("eval epoch"); ax.set_ylabel("feature")
        # FB-R5-01: make the transform EXPLICIT (not hidden) — a small scale annotation.
        # R5-N1: moved INSIDE the plot (bottom-right, semi-transparent backing) so it no
        # longer overlaps the right end of the "· epoch NNN" title.
        ax.text(0.985, 0.025, f"scale: {scale}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color="#fff", fontstyle="italic",
                bbox=dict(facecolor="#000", alpha=0.45, edgecolor="none", pad=1.5))
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        frames.append(_fig_to_rgb(fig))
    return frames


# ── informative single-frame note (graceful no-data fallback) ─────────────────
def render_note(*, title: str, message: str) -> list[np.ndarray]:
    """A single explicit, readable frame stating WHY there is no animation, instead of
    an empty/no_data GIF. Used as the last-resort fallback (e.g. RF-1 loss_drift when no
    compatible separation series exists on the leaf). Always returns one valid frame."""
    fig, ax = _new_fig()
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
    ax.axis("off")
    ax.text(0.5, 0.55, message, transform=ax.transAxes, ha="center", va="center",
            fontsize=11, color="#444", wrap=True)
    ax.text(0.5, 0.30, "Pick a compatible metric, or switch the sub-mode.",
            transform=ax.transAxes, ha="center", va="center", fontsize=9,
            color="#888")
    return [_fig_to_rgb(fig)]


# ── encode frames -> GIF bytes ────────────────────────────────────────────────
def encode_gif(frames: list[np.ndarray], *, fps: float = 8.0) -> bytes:
    """Assemble RGB frames into an animated GIF (imageio/pillow, no ffmpeg)."""
    import imageio.v2 as imageio

    if not frames:
        raise ValueError("no frames to encode")
    buf = io.BytesIO()
    duration = max(0.04, 1.0 / max(1.0, fps))
    # FB-9: play ONCE and freeze on the last frame (``loop=1``) instead of looping
    # forever. The renderers append a peak-hold of the final frame, so the freeze
    # dwells on the end state; the viewer adds an explicit replay (re-fetch). Only the
    # loop count changes — GIF89a output, frame identity, cache keying are unchanged.
    imageio.mimsave(buf, frames, format="GIF", duration=duration, loop=1)
    return buf.getvalue()
