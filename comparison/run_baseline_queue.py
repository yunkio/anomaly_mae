#!/usr/bin/env python
"""
Baseline Experiment Queue Orchestrator

Manages sequential baseline experiments with auto-numbered output directories.
Delegates ALL work to run_baseline.py via subprocess.

Output structure mirrors MAE experiments:
    comparison/results/experiments/{N}_{timestamp}_{desc}/{dataset}/{model}/

Usage:
    conda activate dc_vis

    # Run from JSON queue file
    python comparison/run_baseline_queue.py --queue configs/baseline_queue_example.json

    # Dry run (show commands without executing)
    python comparison/run_baseline_queue.py --queue configs/baseline_queue.json --dry-run

    # Custom description for numbered directory
    python comparison/run_baseline_queue.py --queue configs/baseline_queue.json --desc "10ep_full"

    # Manual output base (skip auto-numbering)
    python comparison/run_baseline_queue.py --queue configs/baseline_queue.json \\
        --output-base comparison/results/experiments/1_20260311_test

    # Monitor resources only
    python comparison/run_baseline_queue.py --monitor

Queue JSON format:
    {
        "description": "Baseline 10 epoch experiments",
        "experiments": [
            {
                "name": "sim_10ep",
                "experiment": "simulation",
                "model": "all",
                "neural_epochs": 10,
                "eval_interval": 1
            },
            {
                "name": "swat_10ep",
                "experiment": "swat_a1a2",
                "model": "all",
                "neural_epochs": 10,
                "skip_sota": true
            }
        ]
    }
"""

import sys
import os
import json
import time
import signal
import argparse
import subprocess
import re
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Resource Monitor (shared with MAE queue)
# =============================================================================

class ResourceMonitor:
    """Monitors GPU/CPU/RAM usage. No ML imports."""

    @staticmethod
    def get_gpu_stats():
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
# Baseline Experiment Runner
# =============================================================================

class BaselineExperimentRunner:
    """Runs a single baseline experiment via subprocess."""

    # Patterns for parsing run_baseline.py stdout
    RE_MODEL_START = re.compile(r'Running (\S+)')
    RE_EPOCH_EVAL = re.compile(r'\[Epoch\s+(\d+)\]\s+PRC=([\d.]+)\s+PAK_F1=([\d.]+)')
    RE_METRICS_LINE = re.compile(r'PAK_F1:\s+([\d.]+)')
    RE_SKIP = re.compile(r'\[SKIP\]\s+(\S+)')
    RE_ERROR = re.compile(r'\[ERROR\]\s+(\S+)')
    RE_STATUS_OK = re.compile(r'^\s*\d+\s+(\S+)\s+OK\s+([\d.]+)')

    def __init__(self, exp_config, idx, total, output_base):
        self.config = exp_config
        self.idx = idx
        self.total = total
        self.output_base = output_base

        self.name = exp_config.get('name', f'baseline_{idx}')
        self.experiment = exp_config['experiment']
        self.status = 'pending'
        self.current_model = None
        self.current_epoch = 0
        self.models_done = 0
        self.models_total = 0
        self.gpu_time = 0
        self.start_time = None
        self.end_time = None

    def build_command(self):
        """Build the subprocess command."""
        cmd = [
            sys.executable, 'comparison/run_baseline.py',
            '--experiment', self.experiment,
            '--model', self.config.get('model', 'all'),
            '--output-base', str(self.output_base),
        ]

        # Epoch overrides
        if 'neural_epochs' in self.config:
            cmd += ['--neural-epochs', str(self.config['neural_epochs'])]
        if 'sota_epochs' in self.config:
            cmd += ['--sota-epochs', str(self.config['sota_epochs'])]
        if 'at_epochs' in self.config:
            cmd += ['--at-epochs', str(self.config['at_epochs'])]

        # Eval control
        if 'eval_interval' in self.config:
            cmd += ['--eval-interval', str(self.config['eval_interval'])]
        if 'max_eval_workers' in self.config:
            cmd += ['--max-eval-workers', str(self.config['max_eval_workers'])]
        if self.config.get('no_epoch_eval'):
            cmd += ['--no-epoch-eval']

        # Model filters
        if self.config.get('skip_sota'):
            cmd += ['--skip-sota']
        if self.config.get('skip_at'):
            cmd += ['--skip-at']
        if self.config.get('only_simple'):
            cmd += ['--only-simple']
        if self.config.get('only_neural'):
            cmd += ['--only-neural']

        # Normalization
        if 'normalize_mode' in self.config:
            cmd += ['--normalize-mode', self.config['normalize_mode']]

        # Other
        if self.config.get('force'):
            cmd += ['--force']
        if 'nn_subsample' in self.config:
            cmd += ['--nn-subsample', str(self.config['nn_subsample'])]

        return cmd

    def run(self, on_status_change=None):
        """Run the experiment subprocess, parsing stdout in real-time."""
        cmd = self.build_command()
        self.status = 'running'
        self.start_time = time.time()
        self.exit_code = None
        # Capture last few lines for post-mortem on nonzero exit
        self._tail_buffer = []
        self._tail_max = 30

        if on_status_change:
            on_status_change(self)

        try:
            # `python -u` + PYTHONUNBUFFERED=1 forces the child to flush stdout immediately
            # so a crash mid-batch surfaces in the orchestrator log instead of vanishing
            # into a flushed-at-exit buffer. (Silent-fail prevention.)
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, cwd=str(PROJECT_ROOT), env=env,
            )

            for line in proc.stdout:
                line = line.rstrip('\n')
                self._parse_line(line)
                self._tail_buffer.append(line)
                if len(self._tail_buffer) > self._tail_max:
                    self._tail_buffer.pop(0)
                print(f"  [{self.name}] {line}", flush=True)

            proc.wait()
            self.end_time = time.time()
            self.gpu_time = self.end_time - self.start_time
            self.exit_code = proc.returncode

            if proc.returncode == 0:
                self.status = 'done'
            else:
                self.status = 'error'
                print(f"  [{self.name}] EXIT CODE: {proc.returncode}", flush=True)
                # exit code 3 = run_baseline.py reported model failure (silent-fail guard).
                # exit code 2 = preflight aborted. other codes = unexpected.
                if proc.returncode == 3:
                    print(f"  [{self.name}] (exit 3 = run_baseline.py FAILED MODELS detected)", flush=True)

        except Exception as e:
            self.end_time = time.time()
            self.status = 'error'
            self.exit_code = -1
            print(f"  [{self.name}] EXCEPTION: {e}", flush=True)

        if on_status_change:
            on_status_change(self)

    def _parse_line(self, line):
        """Best-effort parsing of subprocess output for progress tracking."""
        m = self.RE_MODEL_START.search(line)
        if m and '(' in line:
            self.current_model = m.group(1)
            self.current_epoch = 0
            return

        m = self.RE_EPOCH_EVAL.search(line)
        if m:
            self.current_epoch = int(m.group(1))
            return

        m = self.RE_STATUS_OK.search(line)
        if m:
            self.models_done += 1
            return

        m = self.RE_SKIP.search(line)
        if m:
            self.models_done += 1
            return

    def status_line(self):
        """Short status string for display."""
        if self.status == 'pending':
            return f"[{self.idx+1}/{self.total}] {self.name}: pending"
        elif self.status == 'running':
            model = self.current_model or '...'
            ep = f"Epoch {self.current_epoch}" if self.current_epoch else "starting"
            elapsed = timedelta(seconds=int(time.time() - self.start_time))
            return f"[{self.idx+1}/{self.total}] {self.name}: {model} {ep} ({elapsed})"
        elif self.status == 'done':
            elapsed = timedelta(seconds=int(self.gpu_time))
            return f"[{self.idx+1}/{self.total}] {self.name}: done ({elapsed})"
        else:
            return f"[{self.idx+1}/{self.total}] {self.name}: ERROR"


# =============================================================================
# Queue Orchestrator
# =============================================================================

class BaselineQueueOrchestrator:
    """Manages baseline experiment queue execution."""

    def __init__(self, experiments, output_base):
        self.experiments = experiments
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

    def _status_json_path(self):
        """Per-dispatch status JSON path. Lives at the experiment output base so a
        monitoring layer can read it without knowing the dispatch log location."""
        return Path(self.output_base) / 'queue_status.json'

    def _write_status_json(self):
        """Atomic write of the live queue status. Read by monitoring tooling to
        detect silent-fail (status='error' without epoch_metrics.json)."""
        try:
            path = self._status_json_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                'timestamp': time.time(),
                'queue_total': len(self.experiments),
                'queue_done': sum(1 for r in self.runners if r.status == 'done'),
                'queue_error': sum(1 for r in self.runners if r.status == 'error'),
                'entries': [
                    {
                        'idx': r.idx,
                        'name': r.name,
                        'experiment': r.experiment,
                        'model': r.config.get('model'),
                        'status': r.status,
                        'exit_code': getattr(r, 'exit_code', None),
                        'start_time': r.start_time,
                        'end_time': r.end_time,
                        'gpu_time': getattr(r, 'gpu_time', None),
                        'tail': getattr(r, '_tail_buffer', [])[-10:] if r.status == 'error' else [],
                    }
                    for r in self.runners
                ],
            }
            tmp_path = path.with_suffix('.json.tmp')
            with open(tmp_path, 'w') as f:
                json.dump(payload, f, indent=2)
            tmp_path.replace(path)
        except Exception as e:
            print(f"  [STATUS_JSON] write failed: {e}", flush=True)

    def run(self):
        """Execute the full experiment queue."""
        self.start_time = time.time()
        n = len(self.experiments)

        self._print_header()

        for i, exp_config in enumerate(self.experiments):
            if self._interrupted:
                print(f"\nSkipping remaining {n - i} experiments.")
                break

            runner = BaselineExperimentRunner(exp_config, i, n, self.output_base)
            self.runners.append(runner)

            self._print_experiment_start(runner)
            runner.run(on_status_change=self._on_status_change)
            self._print_transition(i, n)
            self._write_status_json()

        self._finish()
        self._write_status_json()

    def _print_header(self):
        n = len(self.experiments)
        print(f"\n{'='*70}")
        print(f"  Baseline Queue: {n} experiments")
        print(f"  Output: {self.output_base}")
        print(f"  Resources: {ResourceMonitor.format_stats()}")
        print(f"{'='*70}")
        for i, exp in enumerate(self.experiments):
            name = exp.get('name', f'baseline_{i}')
            experiment = exp['experiment']
            model = exp.get('model', 'all')
            epochs = exp.get('neural_epochs', 'default')
            flags = []
            if exp.get('skip_sota'):
                flags.append('skip-sota')
            if exp.get('only_simple'):
                flags.append('only-simple')
            flags_str = f" [{', '.join(flags)}]" if flags else ""
            print(f"  [{i+1}] {name}: experiment={experiment} model={model} "
                  f"epochs={epochs}{flags_str}")
        print(f"{'='*70}\n")

    def _print_experiment_start(self, runner):
        print(f"\n{'#'*70}")
        print(f"# QUEUE [{runner.idx+1}/{runner.total}] {runner.name}")
        print(f"# Experiment: {runner.experiment}")
        print(f"# Command: {' '.join(runner.build_command())}")
        print(f"# Resources: {ResourceMonitor.format_stats()}")
        print(f"{'#'*70}")

    def _print_transition(self, current_idx, total):
        elapsed = timedelta(seconds=int(time.time() - self.start_time))
        done = current_idx + 1

        print(f"\n{'─'*70}")
        print(f"  Queue progress: {done}/{total} done | Elapsed: {elapsed}")
        print(f"  Resources: {ResourceMonitor.format_stats()}")
        print(f"{'─'*70}")

    def _on_status_change(self, runner):
        pass

    def _finish(self):
        total_time = time.time() - self.start_time

        print(f"\n{'='*70}")
        print(f"  ALL EXPERIMENTS COMPLETE ({timedelta(seconds=int(total_time))})")
        print(f"{'='*70}")

        # Summary table
        if self.runners:
            print(f"\n  {'#':<4} {'Name':<30} {'Experiment':<20} {'Status':<8} {'Time':>10}")
            print(f"  {'-'*76}")
            total_gpu = 0
            for r in self.runners:
                t_str = str(timedelta(seconds=int(r.gpu_time))) if r.gpu_time else '-'
                total_gpu += r.gpu_time or 0
                print(f"  {r.idx+1:<4} {r.name:<30} {r.experiment:<20} "
                      f"{r.status:<8} {t_str:>10}")
            print(f"  {'-'*76}")
            print(f"  {'Total':>54} {'':<8} "
                  f"{str(timedelta(seconds=int(total_gpu))):>10}")

        # Save summary
        self._save_summary(total_time)
        print(f"\n  Resources: {ResourceMonitor.format_stats()}")
        print(f"\nDone!")

    def _save_summary(self, total_time):
        """Save queue execution summary."""
        summary = {
            'total_time': total_time,
            'timestamp': datetime.now().isoformat(),
            'output_base': str(self.output_base),
            'experiments': [],
        }
        for r in self.runners:
            summary['experiments'].append({
                'name': r.name,
                'experiment': r.experiment,
                'config': r.config,
                'status': r.status,
                'time': r.gpu_time,
                'command': r.build_command(),
            })

        summary_path = self.output_base / 'queue_summary.json'
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  Queue summary saved: {summary_path}")


# =============================================================================
# CLI
# =============================================================================

def monitor_mode():
    """Show resource usage periodically."""
    print("Resource Monitor (Ctrl+C to exit)")
    print(f"{'='*60}")
    try:
        while True:
            ts = datetime.now().strftime('%H:%M:%S')
            stats = ResourceMonitor.format_stats()
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
        description='Baseline Experiment Queue Orchestrator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run queue
  python comparison/run_baseline_queue.py --queue configs/baseline_queue.json

  # Dry run
  python comparison/run_baseline_queue.py --queue configs/baseline_queue.json --dry-run

  # Custom description
  python comparison/run_baseline_queue.py --queue configs/baseline_queue.json --desc "10ep_full"

  # Manual output base (skip auto-numbering)
  python comparison/run_baseline_queue.py --queue configs/baseline_queue.json \\
      --output-base comparison/results/experiments/1_20260311_test

  # Monitor only
  python comparison/run_baseline_queue.py --monitor
""",
    )

    parser.add_argument('--queue', '-q', type=str,
                        help='JSON file with baseline experiment queue')
    parser.add_argument('--desc', '-d', type=str, default='',
                        help='Description for auto-numbered directory name')
    parser.add_argument('--output-base', type=str, default=None,
                        help='Manual output base (skips auto-numbering)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show commands without executing')
    parser.add_argument('--monitor', action='store_true',
                        help='Resource monitor mode (no experiments)')

    args = parser.parse_args()

    if args.monitor:
        monitor_mode()
        return

    if not args.queue:
        parser.print_help()
        print("\nError: provide --queue")
        return

    # Load queue
    with open(args.queue) as f:
        queue_data = json.load(f)

    experiments = queue_data.get('experiments', queue_data)
    if isinstance(experiments, dict):
        experiments = [experiments]

    if not experiments:
        print("Error: no experiments in queue file")
        return

    # Validate experiments
    for i, exp in enumerate(experiments):
        if 'experiment' not in exp:
            print(f"Error: experiment {i} missing 'experiment' field: {exp}")
            return
        if 'name' not in exp:
            exp['name'] = f"{exp['experiment']}_{i}"

    # Determine output base
    if args.output_base:
        output_base = Path(args.output_base)
    else:
        from comparison.baseline_common import make_numbered_baseline_dir
        desc = args.desc or queue_data.get('description', '').replace(' ', '_')[:50]
        output_base = make_numbered_baseline_dir(desc)

    # Dry run
    if args.dry_run:
        print(f"\nDry run: {len(experiments)} experiments")
        print(f"Output: {output_base}\n")
        for i, exp in enumerate(experiments):
            runner = BaselineExperimentRunner(exp, i, len(experiments), output_base)
            cmd = runner.build_command()
            print(f"  [{i+1}] {exp.get('name', '?')}")
            print(f"      {' '.join(cmd)}")
        print()
        return

    # Run queue
    print(f"Output directory: {output_base}")
    orchestrator = BaselineQueueOrchestrator(experiments, output_base)
    orchestrator.run()


if __name__ == '__main__':
    main()
