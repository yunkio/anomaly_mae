#!/usr/bin/env python
"""Telemetry plotter. One call generates 4 category plots:
  1) {prefix}_main.png         — 4 panels: GPU mem%, GPU util%, RAM%, CPU%
  2) {prefix}_gpu_details.png  — GPU power, temperature, clocks (sm/mem), pstate
  3) {prefix}_memory.png       — RAM breakdown (used/free/buffcache/avail) + Swap
  4) {prefix}_cpu_system.png   — CPU breakdown (us/sy/id/wa/st), load avg, ctx switches, IO

Usage:
    python plot_gpu_telemetry.py --csv path.csv --out-dir ./temp --prefix 285_wadi_a2 \
        [--title-suffix "exp285 WaDi_A2"] [--start-unix N --end-unix N]
"""
import argparse
import csv
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def load_csv(path: Path, start_unix: int | None, end_unix: int | None):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                unix = int(row["unix"])
            except (KeyError, ValueError):
                continue
            if start_unix is not None and unix < start_unix:
                continue
            if end_unix is not None and unix > end_unix:
                continue
            rows.append(row)
    return rows


def col(rows, name, cast=float, default=0.0):
    out = []
    for r in rows:
        try:
            out.append(cast(r[name]))
        except (KeyError, ValueError, TypeError):
            out.append(default)
    return out


def _setup_xaxis(fig, ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()


def _save(fig, out, title, n, dur_min):
    fig.suptitle(f"{title}  ({n} samples / {dur_min:.1f} min, ~10s sampling)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_main(rows, times, out: Path, title: str):
    """4-panel main plot: GPU mem%, GPU util%, RAM%, CPU%."""
    gpu_mem_used = col(rows, "gpu_mem_used")
    gpu_mem_total = col(rows, "gpu_mem_total")
    gpu_util = col(rows, "gpu_util")
    ram_used = col(rows, "ram_used")
    ram_total = col(rows, "ram_total")
    cpu_id = col(rows, "cpu_id")

    gpu_mem_pct = [u / t * 100 if t > 0 else 0 for u, t in zip(gpu_mem_used, gpu_mem_total)]
    ram_pct = [u / t * 100 if t > 0 else 0 for u, t in zip(ram_used, ram_total)]
    cpu_busy = [100 - i for i in cpu_id]

    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)

    axes[0].plot(times, gpu_mem_pct, color="tab:blue", linewidth=0.9)
    axes[0].set_ylabel("GPU memory (%)")

    axes[1].plot(times, gpu_util, color="tab:green", linewidth=0.9)
    axes[1].set_ylabel("GPU util (%)")

    axes[2].plot(times, ram_pct, color="tab:purple", linewidth=0.9)
    axes[2].set_ylabel("System RAM (%)")

    axes[3].plot(times, cpu_busy, color="tab:red", linewidth=0.9)
    axes[3].set_ylabel("CPU busy (%)")
    axes[3].set_xlabel("Time")

    _setup_xaxis(fig, axes[3])
    n = len(times)
    dur = (times[-1] - times[0]).total_seconds() / 60 if n > 1 else 0
    _save(fig, out, title, n, dur)


def plot_gpu_details(rows, times, out: Path, title: str):
    """GPU power, temperature, clock, pstate."""
    gpu_power = col(rows, "gpu_power")
    gpu_power_limit = col(rows, "gpu_power_limit")
    gpu_temp = col(rows, "gpu_temp")
    gpu_clock_sm = col(rows, "gpu_clock_sm")
    gpu_clock_mem = col(rows, "gpu_clock_mem")
    pstate_raw = [r.get("gpu_pstate", "") for r in rows]
    # Encode P-state: 'P0'=0, 'P2'=2, ... 'P8'=8
    pstate = [int(p[1:]) if p.startswith("P") and p[1:].isdigit() else -1 for p in pstate_raw]

    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    for ax in axes:
        ax.grid(True, alpha=0.3)

    axes[0].plot(times, gpu_power, color="tab:red", linewidth=0.9, label="draw")
    if gpu_power_limit and max(gpu_power_limit) > 0:
        axes[0].axhline(gpu_power_limit[0], color="gray", linestyle="--",
                        linewidth=0.6, label=f"limit {gpu_power_limit[0]:.0f}W")
    axes[0].set_ylabel("GPU power (W)")
    axes[0].legend(loc="upper left", fontsize=9)

    axes[1].plot(times, gpu_temp, color="tab:orange", linewidth=0.9)
    axes[1].set_ylabel("GPU temp (°C)")
    axes[1].set_ylim(0, max(85, max(gpu_temp) + 5 if gpu_temp else 85))

    axes[2].plot(times, gpu_clock_sm, color="tab:green", linewidth=0.9, label="SM")
    axes[2].plot(times, gpu_clock_mem, color="tab:cyan", linewidth=0.6, alpha=0.7, label="memory")
    axes[2].set_ylabel("Clock (MHz)")
    axes[2].legend(loc="upper left", fontsize=9)

    axes[3].plot(times, pstate, color="tab:purple", linewidth=0.9, marker=".", markersize=3)
    axes[3].set_ylabel("Power state (P0=full, P8=idle)")
    axes[3].invert_yaxis()  # P0 at top
    axes[3].set_xlabel("Time")

    _setup_xaxis(fig, axes[3])
    n = len(times)
    dur = (times[-1] - times[0]).total_seconds() / 60 if n > 1 else 0
    _save(fig, out, title, n, dur)


def plot_memory(rows, times, out: Path, title: str):
    """RAM breakdown + swap."""
    ram_used = col(rows, "ram_used")
    ram_free = col(rows, "ram_free")
    ram_buffcache = col(rows, "ram_buffcache")
    ram_available = col(rows, "ram_available")
    ram_total = col(rows, "ram_total")
    swap_used = col(rows, "swap_used")
    swap_total = col(rows, "swap_total")

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    for ax in axes:
        ax.grid(True, alpha=0.3)

    # Panel 1: RAM components in MiB
    axes[0].plot(times, ram_used, color="tab:purple", linewidth=0.9, label="used")
    axes[0].plot(times, ram_buffcache, color="tab:olive", linewidth=0.6, alpha=0.8, label="buff/cache")
    axes[0].plot(times, ram_free, color="tab:cyan", linewidth=0.6, alpha=0.8, label="free")
    axes[0].set_ylabel("RAM (MiB)")
    axes[0].legend(loc="upper left", fontsize=9)
    if ram_total:
        axes[0].axhline(ram_total[0], color="gray", linestyle="--",
                        linewidth=0.6, label=f"total {ram_total[0]} MiB")

    # Panel 2: Available memory (more useful than 'free')
    axes[1].plot(times, ram_available, color="tab:green", linewidth=0.9)
    axes[1].set_ylabel("RAM available (MiB)")

    # Panel 3: Swap
    axes[2].plot(times, swap_used, color="tab:brown", linewidth=0.9, label="used")
    if swap_total:
        axes[2].axhline(swap_total[0], color="gray", linestyle="--",
                        linewidth=0.6, label=f"total {swap_total[0]} MiB")
    axes[2].set_ylabel("Swap (MiB)")
    axes[2].set_xlabel("Time")
    axes[2].legend(loc="upper left", fontsize=9)

    _setup_xaxis(fig, axes[2])
    n = len(times)
    dur = (times[-1] - times[0]).total_seconds() / 60 if n > 1 else 0
    _save(fig, out, title, n, dur)


def plot_cpu_system(rows, times, out: Path, title: str):
    """CPU breakdown + load avg + system activity."""
    cpu_us = col(rows, "cpu_us")
    cpu_sy = col(rows, "cpu_sy")
    cpu_id = col(rows, "cpu_id")
    cpu_wa = col(rows, "cpu_wa")
    cpu_st = col(rows, "cpu_st")
    load_1m = col(rows, "load_1m")
    load_5m = col(rows, "load_5m")
    load_15m = col(rows, "load_15m")
    sys_cs = col(rows, "sys_cs")
    sys_in = col(rows, "sys_in")
    io_bi = col(rows, "io_bi")
    io_bo = col(rows, "io_bo")
    procs_r = col(rows, "procs_r")
    procs_b = col(rows, "procs_b")

    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    for ax in axes:
        ax.grid(True, alpha=0.3)

    # Panel 1: CPU breakdown
    axes[0].plot(times, cpu_us, color="tab:red", linewidth=0.9, label="us (user)")
    axes[0].plot(times, cpu_sy, color="darkorange", linewidth=0.7, label="sy (sys)")
    axes[0].plot(times, cpu_wa, color="black", linewidth=0.5, alpha=0.7, label="wa (iowait)")
    axes[0].plot(times, cpu_st, color="tab:gray", linewidth=0.5, alpha=0.7, label="st (steal)")
    axes[0].set_ylabel("CPU (%)")
    axes[0].set_ylim(0, 105)
    axes[0].legend(loc="upper left", fontsize=8, ncol=4)

    # Panel 2: Load average
    axes[1].plot(times, load_1m, color="tab:red", linewidth=0.9, label="1m")
    axes[1].plot(times, load_5m, color="tab:orange", linewidth=0.6, alpha=0.7, label="5m")
    axes[1].plot(times, load_15m, color="tab:olive", linewidth=0.5, alpha=0.6, label="15m")
    axes[1].set_ylabel("Load average")
    axes[1].legend(loc="upper left", fontsize=9)

    # Panel 3: Process state
    axes[2].plot(times, procs_r, color="tab:green", linewidth=0.9, label="runnable (r)")
    axes[2].plot(times, procs_b, color="tab:red", linewidth=0.7, label="blocked (b)")
    axes[2].set_ylabel("Process count")
    axes[2].legend(loc="upper left", fontsize=9)

    # Panel 4: System activity
    ax = axes[3]
    ax.plot(times, sys_cs, color="tab:blue", linewidth=0.7, label="ctx switches/s")
    ax.set_ylabel("Context switches/s", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax.twinx()
    ax2.plot(times, io_bi, color="tab:olive", linewidth=0.5, alpha=0.7, label="block in")
    ax2.plot(times, io_bo, color="tab:brown", linewidth=0.5, alpha=0.7, label="block out")
    ax2.set_ylabel("IO blocks/s", color="tab:brown")
    ax2.tick_params(axis="y", labelcolor="tab:brown")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    ax.set_xlabel("Time")

    _setup_xaxis(fig, axes[3])
    n = len(times)
    dur = (times[-1] - times[0]).total_seconds() / 60 if n > 1 else 0
    _save(fig, out, title, n, dur)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--prefix", required=True, help="Output filename prefix (e.g., 285_wadi_a2)")
    ap.add_argument("--title-suffix", default="", help="Appended to per-plot titles")
    ap.add_argument("--start-unix", type=int, default=None)
    ap.add_argument("--end-unix", type=int, default=None)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_csv(args.csv, args.start_unix, args.end_unix)
    if not rows:
        print(f"No data in {args.csv}")
        return
    times = [datetime.fromtimestamp(int(r["unix"])) for r in rows]
    sfx = f" — {args.title_suffix}" if args.title_suffix else ""

    plot_main(rows, times, args.out_dir / f"{args.prefix}_main.png",
              f"Main metrics{sfx}")
    plot_gpu_details(rows, times, args.out_dir / f"{args.prefix}_gpu_details.png",
                     f"GPU details{sfx}")
    plot_memory(rows, times, args.out_dir / f"{args.prefix}_memory.png",
                f"Memory (RAM + Swap){sfx}")
    plot_cpu_system(rows, times, args.out_dir / f"{args.prefix}_cpu_system.png",
                    f"CPU & System{sfx}")


if __name__ == "__main__":
    main()
