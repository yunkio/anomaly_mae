#!/usr/bin/env python3
"""
Experiment Progress Monitor

Monitors running comparison experiments and displays:
- System resources (RAM, GPU, CPU)
- Experiment progress table with metrics
- Process health status

Usage:
    python comparison/monitor_experiment.py
    python comparison/monitor_experiment.py --exp WaDi_A1_14days
    python comparison/monitor_experiment.py --watch  # Continuous monitoring
"""

import json
import os
import subprocess
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "comparison" / "results"

# Model configurations
SIMPLE_MODELS = ["random", "sensor_range", "pca_error", "l2_norm", "nn_distance"]
NEURAL_MODELS = ["mlp", "mlpmixer", "transformer", "gcn_lstm"]
SOTA_MODELS = ["tranad", "usad", "dagmm", "gdn", "omnianomaly"]
SKIPPED_MODELS = ["anomaly_transformer"]

ALL_MODELS = SIMPLE_MODELS + NEURAL_MODELS + SOTA_MODELS


def get_system_resources():
    """Get system resource usage."""
    resources = {}

    # RAM
    try:
        result = subprocess.run(['free', '-h'], capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        if len(lines) >= 2:
            parts = lines[1].split()
            resources['ram_total'] = parts[1]
            resources['ram_used'] = parts[2]
            resources['ram_free'] = parts[3]
    except:
        resources['ram'] = 'N/A'

    # GPU
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(', ')
            resources['gpu_name'] = parts[0]
            resources['gpu_mem_used'] = parts[1]
            resources['gpu_mem_total'] = parts[2]
            resources['gpu_util'] = parts[3]
        else:
            resources['gpu'] = 'N/A'
    except:
        resources['gpu'] = 'N/A'

    return resources


def get_running_processes():
    """Get information about running experiment processes."""
    processes = []
    try:
        result = subprocess.run(
            ['ps', 'aux'], capture_output=True, text=True
        )
        for line in result.stdout.split('\n'):
            if 'run_wadi_14days' in line and 'python' in line and 'grep' not in line:
                parts = line.split()
                if len(parts) >= 11:
                    processes.append({
                        'pid': parts[1],
                        'cpu': parts[2],
                        'mem': parts[3],
                        'time': parts[9],
                        'cmd': ' '.join(parts[10:])
                    })
    except:
        pass
    return processes


def get_model_status(exp_dir, model_name):
    """Get model status and metrics."""
    model_dir = Path(exp_dir) / model_name
    metadata_path = model_dir / "metadata.json"
    scores_path = model_dir / "scores.npy"
    model_subdir = model_dir / "model"

    if not model_dir.exists():
        return "⏳ 대기", None

    if metadata_path.exists() and os.path.getsize(metadata_path) > 0:
        try:
            with open(metadata_path) as f:
                data = json.load(f)
            metrics = data.get("metrics", {})
            return "✅", metrics
        except:
            return "🔄 진행중", None

    if scores_path.exists() or model_subdir.exists():
        return "🔄 진행중", None

    # Directory exists but empty - model is running
    if model_dir.exists():
        return "🔄 진행중", None

    return "⏳ 대기", None


def extract_metrics(metrics):
    """Extract specific metrics for display."""
    if not metrics:
        return ["-"] * 8

    pl = metrics.get("point_level", {})
    f1t = metrics.get("f1_t", {})
    pak = metrics.get("pa_k", {})

    roc = pl.get("roc_auc", 0)
    prc = pl.get("prc_auc", 0)
    f1 = pl.get("f1_score", 0)
    f1_t_val = f1t.get("f1_t", 0)

    pa20 = pak.get("pa_20", {})
    pa80 = pak.get("pa_80", {})

    pa20_f1 = pa20.get("f1_score", 0)
    pa20_prc = pa20.get("prc_auc", 0)
    pa80_f1 = pa80.get("f1_score", 0)
    pa80_prc = pa80.get("prc_auc", 0)

    return [f"{roc:.3f}", f"{prc:.3f}", f"{f1:.3f}", f"{f1_t_val:.3f}",
            f"{pa20_f1:.3f}", f"{pa20_prc:.3f}", f"{pa80_f1:.3f}", f"{pa80_prc:.3f}"]


def print_experiment_status(exp_name, exp_dir):
    """Print status table for an experiment."""
    print(f"\n{'='*110}")
    print(f"📊 {exp_name}")
    print(f"{'='*110}")

    header = f"{'#':<3} {'Model':<20} {'Status':<12} {'ROC':>7} {'PRC':>7} {'F1':>7} {'F1_T':>7} {'PA20_F1':>8} {'PA20_PRC':>9} {'PA80_F1':>8} {'PA80_PRC':>9}"
    print(header)
    print("-" * 110)

    completed = 0
    running = 0
    waiting = 0

    for i, model in enumerate(ALL_MODELS, 1):
        status, metrics = get_model_status(exp_dir, model)
        values = extract_metrics(metrics)

        status_short = status.split()[0] if status else "-"
        print(f"{i:<3} {model:<20} {status_short:<12} {values[0]:>7} {values[1]:>7} {values[2]:>7} {values[3]:>7} {values[4]:>8} {values[5]:>9} {values[6]:>8} {values[7]:>9}")

        if "✅" in status:
            completed += 1
        elif "🔄" in status:
            running += 1
        else:
            waiting += 1

    # Show skipped models
    for model in SKIPPED_MODELS:
        print(f"-   {model:<20} {'⏭️스킵':<12} {'-':>7} {'-':>7} {'-':>7} {'-':>7} {'-':>8} {'-':>9} {'-':>8} {'-':>9}")

    print("-" * 110)
    total = len(ALL_MODELS)
    pct = (completed / total) * 100
    print(f"진행상황: ✅ 완료 {completed}/{total} ({pct:.1f}%) | 🔄 진행중 {running} | ⏳ 대기 {waiting}")

    return completed, running, waiting


def monitor(experiments=None, watch=False, interval=60):
    """Main monitoring function."""
    if experiments is None:
        # Auto-detect experiments with 14days in name
        experiments = [d.name for d in RESULTS_DIR.iterdir()
                      if d.is_dir() and '14days' in d.name]

    while True:
        os.system('clear' if os.name == 'posix' else 'cls')

        print("=" * 110)
        print(f"🖥️  실험 모니터링 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 110)

        # System resources
        res = get_system_resources()
        print(f"\n📈 시스템 리소스:")
        if 'ram_total' in res:
            print(f"   RAM: {res['ram_used']} / {res['ram_total']} (free: {res['ram_free']})")
        if 'gpu_name' in res:
            print(f"   GPU: {res['gpu_name']} - {res['gpu_mem_used']} / {res['gpu_mem_total']} ({res['gpu_util']})")

        # Running processes
        procs = get_running_processes()
        if procs:
            print(f"\n🔄 실행 중인 프로세스:")
            for p in procs:
                print(f"   PID {p['pid']}: CPU {p['cpu']}%, MEM {p['mem']}%, TIME {p['time']}")
        else:
            print(f"\n⚠️  실행 중인 실험 프로세스 없음")

        # Experiment status
        all_done = True
        for exp_name in experiments:
            exp_dir = RESULTS_DIR / exp_name
            if exp_dir.exists():
                completed, running, waiting = print_experiment_status(exp_name, exp_dir)
                if running > 0 or waiting > 0:
                    all_done = False

        if not watch:
            break

        if all_done:
            print("\n✅ 모든 실험 완료!")
            break

        print(f"\n⏳ {interval}초 후 갱신... (Ctrl+C로 종료)")
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description='Monitor experiment progress')
    parser.add_argument('--exp', type=str, nargs='*', help='Experiment names to monitor')
    parser.add_argument('--watch', '-w', action='store_true', help='Continuous monitoring')
    parser.add_argument('--interval', '-i', type=int, default=60, help='Refresh interval in seconds')
    args = parser.parse_args()

    try:
        monitor(experiments=args.exp, watch=args.watch, interval=args.interval)
    except KeyboardInterrupt:
        print("\n\n모니터링 종료")


if __name__ == "__main__":
    main()
