#!/usr/bin/env python
"""
Experiment Queue Orchestrator

Manages sequential GPU experiments with parallel CPU background work.
Pure orchestrator — delegates ALL ML work to run_base_experiments.py via subprocess.
Zero ML imports. Zero direct model/data access.

GPU idle time is eliminated by using --no-wait: each experiment subprocess exits
after GPU work completes, and the orchestrator immediately starts the next one.
Background CPU processes (eval+viz) continue independently.

Usage:
    conda activate dc_vis

    # Run from JSON queue file
    python scripts/run_queue.py --queue configs/queue_example.json

    # Run inline experiments
    python scripts/run_queue.py \\
        --exp "set=C" \\
        --exp "set=C config_override='use_discriminator=True'"

    # Dry run (show commands without executing)
    python scripts/run_queue.py --queue configs/queue.json --dry-run

    # Monitor resources only (no experiments)
    python scripts/run_queue.py --monitor

    # Limit background CPU workers
    python scripts/run_queue.py --queue configs/queue.json --max-bg-workers 10

    # Custom output base
    python scripts/run_queue.py --exp "set=C" --output-base results/experiments
"""

import sys
import os
import json
import time
import signal
import argparse
import subprocess
import threading
import re
from pathlib import Path
from datetime import datetime, timedelta


# =============================================================================
# Resource Monitor
# =============================================================================

class ResourceMonitor:
    """Monitors GPU/CPU/RAM usage. No ML imports."""

    @staticmethod
    def get_gpu_stats():
        """Query nvidia-smi for GPU stats."""
        try:
            out = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5,
            )
            if out.returncode == 0:
                parts = out.stdout.strip().split(', ')
                return {
                    'gpu_util': int(parts[0]),
                    'mem_used_mb': int(parts[1]),
                    'mem_total_mb': int(parts[2]),
                }
        except Exception:
            pass
        return None

    @staticmethod
    def get_cpu_ram_stats():
        """Get CPU and RAM usage via psutil."""
        try:
            import psutil
            vm = psutil.virtual_memory()
            return {
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'ram_used_gb': vm.used / (1024 ** 3),
                'ram_total_gb': vm.total / (1024 ** 3),
            }
        except ImportError:
            return None

    @staticmethod
    def format_stats():
        """One-line resource summary."""
        parts = []
        gpu = ResourceMonitor.get_gpu_stats()
        if gpu:
            parts.append(
                f"GPU {gpu['gpu_util']}% "
                f"{gpu['mem_used_mb']/1024:.1f}/{gpu['mem_total_mb']/1024:.1f}GB"
            )
        cpu = ResourceMonitor.get_cpu_ram_stats()
        if cpu:
            parts.append(f"CPU {cpu['cpu_percent']:.0f}%")
            parts.append(f"RAM {cpu['ram_used_gb']:.1f}/{cpu['ram_total_gb']:.1f}GB")
        return ' | '.join(parts) if parts else 'N/A'


# =============================================================================
# Background Process Tracker
# =============================================================================

class BackgroundTracker:
    """Tracks background CPU processes spawned by experiments."""

    def __init__(self, max_workers=10):
        self.max_workers = max_workers
        self._entries = []  # [(exp_name, pid, dataset_key, start_time)]
        self._lock = threading.Lock()

    def register(self, exp_name, pids, dataset_keys=None):
        """Register PIDs from a completed experiment."""
        with self._lock:
            for i, pid in enumerate(pids):
                key = dataset_keys[i] if dataset_keys and i < len(dataset_keys) else f'worker_{i}'
                self._entries.append((exp_name, pid, key, time.time()))

    def alive_count(self):
        """Count currently alive background processes."""
        self._cleanup()
        with self._lock:
            return len(self._entries)

    def alive_entries(self):
        """Get list of alive background entries."""
        self._cleanup()
        with self._lock:
            return list(self._entries)

    def _cleanup(self):
        """Remove entries for dead processes."""
        with self._lock:
            alive = []
            for entry in self._entries:
                _, pid, _, _ = entry
                try:
                    os.kill(pid, 0)  # Check if process exists
                    alive.append(entry)
                except OSError:
                    pass
            self._entries = alive

    def wait_until_below(self, threshold=None):
        """Block until alive count is below threshold."""
        if threshold is None:
            threshold = self.max_workers
        while self.alive_count() >= threshold:
            time.sleep(5)

    def wait_all(self, timeout=600):
        """Wait for all background processes to finish."""
        deadline = time.time() + timeout
        while self.alive_count() > 0 and time.time() < deadline:
            time.sleep(5)
        remaining = self.alive_count()
        if remaining > 0:
            print(f"  Warning: {remaining} background processes still running after {timeout}s timeout")


# =============================================================================
# Experiment Runner
# =============================================================================

class ExperimentRunner:
    """Runs a single experiment via subprocess."""

    # Patterns for parsing run_base_experiments.py stdout
    RE_DATASET_START = re.compile(r'#\s*\[(\d+)/(\d+)\]\s+(\S+)')
    RE_EPOCH = re.compile(r'Epoch\s+(\d+)/(\d+)')
    RE_EVAL = re.compile(r'\[Epoch\s+(\d+)\]\s+PRC=([\d.]+)')
    RE_BG_SPAWN = re.compile(r'Spawning background eval\+viz for\s+(\S+)')
    RE_BG_PIDS = re.compile(r'Background PIDs:\s*\[([\d,\s]*)\]')
    RE_COMPLETE = re.compile(r'ALL EXPERIMENTS COMPLETE')

    def __init__(self, exp_config, idx, total, output_base=None):
        self.config = exp_config
        self.idx = idx
        self.total = total
        self.output_base = output_base

        self.name = exp_config.get('name', f'exp_{idx}')
        self.status = 'pending'  # pending → running → done / error
        self.current_dataset = None
        self.current_dataset_idx = 0
        self.total_datasets = 0
        self.current_epoch = 0
        self.total_epochs = 0
        self.spawned_bg_keys = []
        self.bg_pids = []
        self.gpu_time = 0
        self.start_time = None
        self.end_time = None
        self.result_dir = None

    def build_command(self):
        """Build the subprocess command."""
        cmd = [
            sys.executable, 'scripts/run_base_experiments.py',
            '--set', self.config['set'],
            '--no-wait',
        ]
        if 'dataset' in self.config:
            datasets = self.config['dataset']
            if isinstance(datasets, str):
                datasets = [datasets]
            cmd += ['--dataset'] + datasets
        if 'config_override' in self.config:
            overrides = self.config['config_override']
            if isinstance(overrides, str):
                overrides = [overrides]
            cmd += ['--config-override'] + overrides
        if 'start_from' in self.config:
            cmd += ['--start-from', str(self.config['start_from'])]
        if self.output_base:
            cmd += ['--output-base', self.output_base]
        return cmd

    def run(self, on_status_change=None):
        """Run the experiment subprocess, parsing stdout in real-time."""
        cmd = self.build_command()
        self.status = 'running'
        self.start_time = time.time()
        self.spawned_bg_keys = []
        self.bg_pids = []

        if on_status_change:
            on_status_change(self)

        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, cwd=str(Path(__file__).parent.parent),
            )

            for line in proc.stdout:
                line = line.rstrip('\n')
                self._parse_line(line)
                # Print with experiment prefix
                print(f"  [{self.name}] {line}")

            proc.wait()
            self.end_time = time.time()
            self.gpu_time = self.end_time - self.start_time

            if proc.returncode == 0:
                self.status = 'done'
            else:
                self.status = 'error'
                print(f"  [{self.name}] EXIT CODE: {proc.returncode}")

        except Exception as e:
            self.end_time = time.time()
            self.status = 'error'
            print(f"  [{self.name}] EXCEPTION: {e}")

        if on_status_change:
            on_status_change(self)

        return self.bg_pids

    def _parse_line(self, line):
        """Best-effort parsing of subprocess output for progress tracking."""
        m = self.RE_DATASET_START.search(line)
        if m:
            self.current_dataset_idx = int(m.group(1))
            self.total_datasets = int(m.group(2))
            self.current_dataset = m.group(3)
            self.current_epoch = 0
            return

        m = self.RE_EPOCH.search(line)
        if m:
            self.current_epoch = int(m.group(1))
            self.total_epochs = int(m.group(2))
            return

        m = self.RE_BG_SPAWN.search(line)
        if m:
            self.spawned_bg_keys.append(m.group(1))
            return

        m = self.RE_BG_PIDS.search(line)
        if m:
            pid_str = m.group(1).strip()
            if pid_str:
                self.bg_pids = [int(p.strip()) for p in pid_str.split(',') if p.strip()]
            return

    def status_line(self):
        """Short status string for display."""
        if self.status == 'pending':
            return f"[{self.idx+1}/{self.total}] {self.name}: pending"
        elif self.status == 'running':
            ds = self.current_dataset or '...'
            ep = f"Epoch {self.current_epoch}/{self.total_epochs}" if self.total_epochs else "starting"
            elapsed = timedelta(seconds=int(time.time() - self.start_time))
            return (f"[{self.idx+1}/{self.total}] {self.name}: "
                    f"{ds} [{self.current_dataset_idx}/{self.total_datasets}] {ep} ({elapsed})")
        elif self.status == 'done':
            elapsed = timedelta(seconds=int(self.gpu_time))
            return f"[{self.idx+1}/{self.total}] {self.name}: done ({elapsed})"
        else:
            return f"[{self.idx+1}/{self.total}] {self.name}: ERROR"


# =============================================================================
# Queue Orchestrator
# =============================================================================

class QueueOrchestrator:
    """Manages experiment queue execution."""

    def __init__(self, experiments, max_bg_workers=10, output_base=None):
        self.experiments = experiments
        self.bg_tracker = BackgroundTracker(max_workers=max_bg_workers)
        self.output_base = output_base
        self.runners = []
        self.start_time = None
        self._interrupted = False

        signal.signal(signal.SIGINT, self._handle_interrupt)

    def _handle_interrupt(self, signum, frame):
        if self._interrupted:
            print("\nForce quit.")
            sys.exit(1)
        self._interrupted = True
        print("\n\nInterrupted. Finishing current experiment... (Ctrl+C again to force quit)")
        print(f"  Background processes ({self.bg_tracker.alive_count()}) will continue independently.")

    def run(self):
        """Execute the full experiment queue."""
        self.start_time = time.time()
        n = len(self.experiments)

        self._print_header()

        for i, exp_config in enumerate(self.experiments):
            if self._interrupted:
                print(f"\nSkipping remaining {n - i} experiments.")
                break

            runner = ExperimentRunner(exp_config, i, n, self.output_base)
            self.runners.append(runner)

            # Wait if too many background workers
            if self.bg_tracker.alive_count() >= self.bg_tracker.max_workers:
                print(f"\n  Waiting for background workers "
                      f"({self.bg_tracker.alive_count()}/{self.bg_tracker.max_workers})...")
                self.bg_tracker.wait_until_below()

            # Print experiment separator
            self._print_experiment_start(runner)

            # Run experiment (blocks until GPU phase done)
            bg_pids = runner.run(on_status_change=self._on_status_change)

            # Register background PIDs
            if bg_pids:
                self.bg_tracker.register(runner.name, bg_pids, runner.spawned_bg_keys)
                print(f"  [{runner.name}] Background: {len(bg_pids)} processes registered")

            # Print inter-experiment status
            self._print_transition(i, n)

        # Final wait for background processes
        self._finish()

    def _print_header(self):
        n = len(self.experiments)
        print(f"\n{'='*70}")
        print(f"  Experiment Queue: {n} experiments")
        print(f"  Max background workers: {self.bg_tracker.max_workers}")
        print(f"  Resources: {ResourceMonitor.format_stats()}")
        print(f"{'='*70}")
        for i, exp in enumerate(self.experiments):
            name = exp.get('name', f'exp_{i}')
            set_name = exp['set']
            override = exp.get('config_override', '')
            datasets = exp.get('dataset', 'all')
            print(f"  [{i+1}] {name}: set={set_name} datasets={datasets} {override}")
        print(f"{'='*70}\n")

    def _print_experiment_start(self, runner):
        print(f"\n{'#'*70}")
        print(f"# QUEUE [{runner.idx+1}/{runner.total}] {runner.name}")
        print(f"# Command: {' '.join(runner.build_command())}")
        print(f"# Background workers: {self.bg_tracker.alive_count()}"
              f"/{self.bg_tracker.max_workers}")
        print(f"# Resources: {ResourceMonitor.format_stats()}")
        print(f"{'#'*70}")

    def _print_transition(self, current_idx, total):
        elapsed = timedelta(seconds=int(time.time() - self.start_time))
        done = current_idx + 1
        bg = self.bg_tracker.alive_count()

        print(f"\n{'─'*70}")
        print(f"  Queue progress: {done}/{total} GPU done | "
              f"{bg} background workers | Elapsed: {elapsed}")
        print(f"  Resources: {ResourceMonitor.format_stats()}")

        # Show active background workers
        entries = self.bg_tracker.alive_entries()
        if entries:
            print(f"  Background workers:")
            for exp_name, pid, key, t0 in entries:
                age = timedelta(seconds=int(time.time() - t0))
                print(f"    [{exp_name}] {key} (PID {pid}, {age})")
        print(f"{'─'*70}")

    def _on_status_change(self, runner):
        pass  # Could add logging here

    def _finish(self):
        total_time = time.time() - self.start_time
        bg_count = self.bg_tracker.alive_count()

        print(f"\n{'='*70}")
        print(f"  ALL GPU WORK COMPLETE ({timedelta(seconds=int(total_time))})")
        print(f"{'='*70}")

        # Summary table
        if self.runners:
            print(f"\n  {'#':<4} {'Name':<30} {'Status':<8} {'GPU Time':>10}")
            print(f"  {'-'*56}")
            total_gpu = 0
            for r in self.runners:
                gpu_str = str(timedelta(seconds=int(r.gpu_time))) if r.gpu_time else '-'
                total_gpu += r.gpu_time or 0
                print(f"  {r.idx+1:<4} {r.name:<30} {r.status:<8} {gpu_str:>10}")
            print(f"  {'-'*56}")
            print(f"  {'Total GPU':>34} {'':<8} {str(timedelta(seconds=int(total_gpu))):>10}")

        # Wait for background
        if bg_count > 0:
            print(f"\n  Waiting for {bg_count} background workers...")
            entries = self.bg_tracker.alive_entries()
            for exp_name, pid, key, _ in entries:
                print(f"    [{exp_name}] {key} (PID {pid})")
            self.bg_tracker.wait_all(timeout=1200)
            remaining = self.bg_tracker.alive_count()
            if remaining == 0:
                print(f"  All background workers completed.")
            else:
                print(f"  {remaining} background workers still running (will complete independently).")
        else:
            print(f"\n  No background workers remaining.")

        # Save queue summary
        self._save_summary(total_time)
        print(f"\n  Resources: {ResourceMonitor.format_stats()}")
        print(f"\nDone!")

    def _save_summary(self, total_time):
        """Save queue execution summary."""
        summary = {
            'total_time': total_time,
            'timestamp': datetime.now().isoformat(),
            'experiments': [],
        }
        for r in self.runners:
            summary['experiments'].append({
                'name': r.name,
                'config': r.config,
                'status': r.status,
                'gpu_time': r.gpu_time,
                'bg_pids': r.bg_pids,
                'spawned_bg_keys': r.spawned_bg_keys,
                'command': r.build_command(),
            })

        summary_path = Path('results/experiments/queue_summary.json')
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  Queue summary saved: {summary_path}")


# =============================================================================
# CLI
# =============================================================================

def parse_inline_exp(exp_str):
    """Parse inline experiment string: 'set=C config_override=...' """
    parts = {}
    # Split by space, but respect quoted values
    tokens = re.findall(r"(\w+)='([^']*)'|(\w+)=(\S+)", exp_str)
    for match in tokens:
        if match[0]:  # key='value with spaces'
            parts[match[0]] = match[1]
        elif match[2]:  # key=value
            parts[match[2]] = match[3]
    return parts


def monitor_mode():
    """Show resource usage periodically."""
    print("Resource Monitor (Ctrl+C to exit)")
    print(f"{'='*60}")
    try:
        while True:
            ts = datetime.now().strftime('%H:%M:%S')
            stats = ResourceMonitor.format_stats()
            # Count python processes (rough background worker count)
            try:
                import psutil
                py_procs = len([p for p in psutil.process_iter(['name'])
                               if 'python' in (p.info['name'] or '').lower()])
            except Exception:
                py_procs = '?'
            print(f"  [{ts}] {stats} | Python procs: {py_procs}")
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nDone.")


def main():
    parser = argparse.ArgumentParser(
        description='Experiment Queue Orchestrator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # JSON queue file
  python scripts/run_queue.py --queue configs/queue.json

  # Inline experiments
  python scripts/run_queue.py \\
      --exp "set=C name=baseline" \\
      --exp "set=C name=with_disc config_override='use_discriminator=True'"

  # Dry run
  python scripts/run_queue.py --queue configs/queue.json --dry-run

  # Monitor only
  python scripts/run_queue.py --monitor
""",
    )

    parser.add_argument('--queue', '-q', type=str,
                        help='JSON file with experiment queue')
    parser.add_argument('--exp', '-e', type=str, action='append', default=[],
                        help='Inline experiment (e.g., "set=C name=exp1")')
    parser.add_argument('--max-bg-workers', type=int, default=10,
                        help='Max background CPU workers (default: 10)')
    parser.add_argument('--output-base', type=str, default=None,
                        help='Override output base directory')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show commands without executing')
    parser.add_argument('--monitor', action='store_true',
                        help='Resource monitor mode (no experiments)')

    args = parser.parse_args()

    if args.monitor:
        monitor_mode()
        return

    # Build experiment list
    experiments = []

    if args.queue:
        with open(args.queue) as f:
            queue_data = json.load(f)
        experiments = queue_data.get('experiments', queue_data)
        if isinstance(experiments, dict):
            experiments = [experiments]

    for exp_str in args.exp:
        experiments.append(parse_inline_exp(exp_str))

    if not experiments:
        parser.print_help()
        print("\nError: provide --queue or --exp")
        return

    # Validate experiments
    for i, exp in enumerate(experiments):
        if 'set' not in exp:
            print(f"Error: experiment {i} missing 'set' field: {exp}")
            return
        if 'name' not in exp:
            exp['name'] = f"exp_{i}"

    # Dry run
    if args.dry_run:
        print(f"\nDry run: {len(experiments)} experiments\n")
        for i, exp in enumerate(experiments):
            runner = ExperimentRunner(exp, i, len(experiments), args.output_base)
            cmd = runner.build_command()
            print(f"  [{i+1}] {exp.get('name', '?')}")
            print(f"      {' '.join(cmd)}")
        print()
        return

    # Run queue
    orchestrator = QueueOrchestrator(
        experiments,
        max_bg_workers=args.max_bg_workers,
        output_base=args.output_base,
    )
    orchestrator.run()


if __name__ == '__main__':
    main()
