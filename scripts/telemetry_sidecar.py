#!/usr/bin/env python
"""Telemetry sidecar — polls hardware/process stats and writes CSV + plot.

Usage:
    python scripts/telemetry_sidecar.py --exp-dir <path> [--worker-pid <pid>] [--interval 5]

Writes:
    <exp_dir>/telemetry/snapshots.csv
    <exp_dir>/telemetry/plot.png

Designed to run as a background process alongside an experiment. Stops on
SIGTERM/SIGINT, regenerates plot every 60 samples.
"""
import argparse
import csv
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

CSV_FIELDS = [
    'ts_iso', 'elapsed_s',
    'gpu_util_pct', 'gpu_mem_used_mb', 'gpu_mem_total_mb',
    'gpu_mem_bw_pct', 'gpu_power_w', 'gpu_temp_c', 'gpu_fan_pct',
    'gpu_pstate', 'gpu_clk_gr_mhz', 'gpu_clk_mem_mhz',
    'cpu_load1', 'cpu_load5', 'cpu_load15',
    'ram_used_mb', 'ram_free_mb', 'ram_total_mb', 'swap_used_mb',
    'worker_pid', 'worker_cpu_pct', 'worker_mem_pct', 'worker_rss_mb', 'worker_threads',
    'disk_used_pct',
]

_stop = False


def _sigterm(_signum, _frame):
    global _stop
    _stop = True


def collect_gpu():
    """nvidia-smi single-shot, return dict with fields or None."""
    try:
        out = subprocess.run(
            ['nvidia-smi',
             '--query-gpu=utilization.gpu,memory.used,memory.total,utilization.memory,power.draw,temperature.gpu,fan.speed,pstate,clocks.current.graphics,clocks.current.memory',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return None
        parts = [p.strip() for p in out.stdout.strip().split(',')]
        return {
            'gpu_util_pct': parts[0],
            'gpu_mem_used_mb': parts[1],
            'gpu_mem_total_mb': parts[2],
            'gpu_mem_bw_pct': parts[3],
            'gpu_power_w': parts[4],
            'gpu_temp_c': parts[5],
            'gpu_fan_pct': parts[6],
            'gpu_pstate': parts[7],
            'gpu_clk_gr_mhz': parts[8],
            'gpu_clk_mem_mhz': parts[9],
        }
    except Exception:
        return None


def collect_cpu_ram():
    """/proc-based CPU load and RAM."""
    res = {}
    try:
        with open('/proc/loadavg') as f:
            la = f.read().split()
            res['cpu_load1'] = la[0]
            res['cpu_load5'] = la[1]
            res['cpu_load15'] = la[2]
    except Exception:
        pass
    try:
        meminfo = {}
        with open('/proc/meminfo') as f:
            for line in f:
                k, _, v = line.partition(':')
                meminfo[k.strip()] = int(v.strip().split()[0])  # kB
        res['ram_used_mb'] = (meminfo.get('MemTotal', 0) - meminfo.get('MemAvailable', 0)) // 1024
        res['ram_free_mb'] = meminfo.get('MemAvailable', 0) // 1024
        res['ram_total_mb'] = meminfo.get('MemTotal', 0) // 1024
        res['swap_used_mb'] = (meminfo.get('SwapTotal', 0) - meminfo.get('SwapFree', 0)) // 1024
    except Exception:
        pass
    return res


def collect_worker(pid):
    """Per-process stats via /proc."""
    if not pid:
        return {}
    res = {'worker_pid': pid}
    try:
        with open(f'/proc/{pid}/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    res['worker_rss_mb'] = int(line.split()[1]) // 1024
                elif line.startswith('Threads:'):
                    res['worker_threads'] = int(line.split()[1])
    except Exception:
        return res
    try:
        # %CPU = (utime+stime delta) / (now - start) over uptime; use top-style snapshot:
        out = subprocess.run(
            ['top', '-b', '-n', '1', '-p', str(pid)],
            capture_output=True, text=True, timeout=5,
        )
        for line in out.stdout.splitlines():
            parts = line.split()
            if parts and parts[0] == str(pid):
                # %CPU is at index 8, %MEM at index 9 in top default format
                res['worker_cpu_pct'] = parts[8]
                res['worker_mem_pct'] = parts[9]
                break
    except Exception:
        pass
    return res


def collect_disk(path):
    try:
        out = subprocess.run(
            ['df', '-BM', str(path)],
            capture_output=True, text=True, timeout=5,
        )
        lines = out.stdout.strip().splitlines()
        if len(lines) >= 2:
            parts = lines[1].split()
            # Use%
            return {'disk_used_pct': parts[4].rstrip('%')}
    except Exception:
        pass
    return {}


def find_worker_pid(exp_dir):
    """Heuristic: find the python process whose cwd contains this exp dir."""
    try:
        out = subprocess.run(
            ['pgrep', '-f', 'run_base_experiments.py|run_queue.py'],
            capture_output=True, text=True, timeout=5,
        )
        pids = out.stdout.strip().split()
        # Pick the one with highest CPU (most likely the active training worker)
        candidates = []
        for pid in pids:
            try:
                with open(f'/proc/{pid}/status') as f:
                    state = f.read()
                if 'VmRSS' in state:
                    rss_line = [l for l in state.splitlines() if l.startswith('VmRSS:')][0]
                    rss_kb = int(rss_line.split()[1])
                    candidates.append((int(pid), rss_kb))
            except Exception:
                continue
        if candidates:
            # highest RSS = main training proc
            candidates.sort(key=lambda x: -x[1])
            return candidates[0][0]
    except Exception:
        pass
    return None


def make_plot(csv_path, png_path):
    """Multi-panel time series plot from CSV."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import csv as _csv

        rows = []
        with open(csv_path) as f:
            reader = _csv.DictReader(f)
            for r in reader:
                rows.append(r)
        if not rows:
            return

        def col(name, cast=float):
            out = []
            for r in rows:
                v = r.get(name, '')
                try:
                    out.append(cast(v))
                except (ValueError, TypeError):
                    out.append(float('nan'))
            return out

        ts = [float(r['elapsed_s']) / 60 for r in rows]  # minutes

        fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=True)
        fig.suptitle(f'Telemetry — {Path(csv_path).parent.parent.name}', fontsize=12)

        # GPU util, mem-bw
        ax = axes[0, 0]
        ax.plot(ts, col('gpu_util_pct'), label='GPU util %', color='tab:blue', linewidth=0.8)
        ax.plot(ts, col('gpu_mem_bw_pct'), label='GPU mem-bw %', color='tab:cyan', linewidth=0.8, alpha=0.6)
        ax.set_ylabel('%')
        ax.set_ylim(0, 105)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_title('GPU utilization')

        # GPU mem MB
        ax = axes[0, 1]
        used = col('gpu_mem_used_mb')
        total = col('gpu_mem_total_mb')
        ax.plot(ts, used, color='tab:green', linewidth=0.8)
        if total:
            ax.axhline(total[0], color='gray', linestyle='--', alpha=0.5, label=f'total {int(total[0])} MiB')
        ax.set_ylabel('MiB')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_title('GPU memory used')

        # GPU power, temp
        ax = axes[1, 0]
        ax.plot(ts, col('gpu_power_w'), color='tab:orange', linewidth=0.8)
        ax.set_ylabel('W')
        ax.grid(alpha=0.3)
        ax.set_title('GPU power')

        ax = axes[1, 1]
        ax.plot(ts, col('gpu_temp_c'), label='temp °C', color='tab:red', linewidth=0.8)
        ax.set_ylabel('°C', color='tab:red')
        ax.grid(alpha=0.3)
        ax2 = ax.twinx()
        ax2.plot(ts, col('gpu_fan_pct'), label='fan %', color='tab:purple', linewidth=0.8, alpha=0.6)
        ax2.set_ylabel('fan %', color='tab:purple')
        ax.set_title('GPU temp + fan')

        # CPU load
        ax = axes[2, 0]
        ax.plot(ts, col('cpu_load1'), label='load1', linewidth=0.8)
        ax.plot(ts, col('cpu_load5'), label='load5', linewidth=0.8, alpha=0.7)
        ax.plot(ts, col('cpu_load15'), label='load15', linewidth=0.8, alpha=0.5)
        ax.axhline(16, color='gray', linestyle='--', alpha=0.4, label='16 cores')
        ax.set_ylabel('load')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_title('CPU load average')

        # Worker %CPU
        ax = axes[2, 1]
        ax.plot(ts, col('worker_cpu_pct'), color='tab:brown', linewidth=0.8)
        ax.set_ylabel('%')
        ax.grid(alpha=0.3)
        ax.set_title('Worker process %CPU (1 core = 100%)')

        # RAM
        ax = axes[3, 0]
        ax.plot(ts, col('ram_used_mb'), color='tab:olive', linewidth=0.8)
        if rows:
            try:
                ram_total = float(rows[0].get('ram_total_mb', 0))
                if ram_total:
                    ax.axhline(ram_total, color='gray', linestyle='--', alpha=0.5, label=f'total {int(ram_total)} MiB')
            except (ValueError, TypeError):
                pass
        ax.set_ylabel('MiB')
        ax.set_xlabel('elapsed time (min)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_title('RAM used')

        # Worker RSS + threads
        ax = axes[3, 1]
        ax.plot(ts, col('worker_rss_mb'), label='RSS MiB', color='tab:gray', linewidth=0.8)
        ax.set_ylabel('RSS MiB', color='tab:gray')
        ax.set_xlabel('elapsed time (min)')
        ax.grid(alpha=0.3)
        ax2 = ax.twinx()
        ax2.plot(ts, col('worker_threads'), label='threads', color='tab:pink', linewidth=0.8, alpha=0.6)
        ax2.set_ylabel('threads', color='tab:pink')
        ax.set_title('Worker RSS + threads')

        plt.tight_layout()
        plt.savefig(png_path, dpi=80, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        # Don't crash sidecar on plot error
        sys.stderr.write(f'[telemetry] plot error: {e}\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp-dir', required=True, help='Experiment directory (will create telemetry/ subdir)')
    ap.add_argument('--worker-pid', type=int, default=None, help='PID of training worker (auto-detected if omitted)')
    ap.add_argument('--interval', type=float, default=5.0, help='Sampling interval in seconds')
    ap.add_argument('--plot-every', type=int, default=60, help='Regenerate plot every N samples')
    ap.add_argument('--stop-when-file', default='summary.json',
                    help='Sentinel filename inside exp_dir that signals experiment completion '
                         '(default: summary.json, written when all datasets done). Set empty to disable.')
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)

    exp_dir = Path(args.exp_dir)
    out_dir = exp_dir / 'telemetry'
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'snapshots.csv'
    png_path = out_dir / 'plot.png'

    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    f = open(csv_path, 'a', buffering=1)
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction='ignore')
    if write_header:
        writer.writeheader()

    start = time.time()
    n_samples = 0
    worker_pid = args.worker_pid

    sentinel = (exp_dir / args.stop_when_file) if args.stop_when_file else None
    sys.stderr.write(f'[telemetry] sidecar started for {exp_dir.name} '
                     f'(interval={args.interval}s, plot_every={args.plot_every}, '
                     f'stop_when={sentinel.name if sentinel else "manual"})\n')

    while not _stop:
        if sentinel is not None and sentinel.exists():
            sys.stderr.write(f'[telemetry] {sentinel.name} detected — experiment complete, exiting\n')
            break
        if worker_pid is None or not Path(f'/proc/{worker_pid}').exists():
            worker_pid = find_worker_pid(exp_dir)

        row = {
            'ts_iso': datetime.now().isoformat(timespec='seconds'),
            'elapsed_s': round(time.time() - start, 1),
        }
        gpu = collect_gpu()
        if gpu:
            row.update(gpu)
        row.update(collect_cpu_ram())
        row.update(collect_worker(worker_pid))
        row.update(collect_disk(exp_dir))

        writer.writerow(row)
        n_samples += 1

        if n_samples % args.plot_every == 0:
            make_plot(csv_path, png_path)

        # Sleep loop that respects _stop
        slept = 0.0
        while slept < args.interval and not _stop:
            time.sleep(min(0.5, args.interval - slept))
            slept += 0.5

    # Final plot
    make_plot(csv_path, png_path)
    f.close()
    sys.stderr.write(f'[telemetry] sidecar stopped after {n_samples} samples ({(time.time() - start) / 60:.1f} min)\n')


if __name__ == '__main__':
    main()
