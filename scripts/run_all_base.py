#!/usr/bin/env python
"""
Run all base experiments for a given config set.

Thin wrapper around run_base_experiments.py that produces hierarchical directory
structure matching old_base_50ep reference:
    results/experiments/{timestamp}_{suffix}/{DatasetGroup}/{Scenario}/

Usage:
    conda activate dc_vis
    python scripts/run_all_base.py --set A                      # All 15 datasets
    python scripts/run_all_base.py --set B                      # Set B params
    python scripts/run_all_base.py --set A --only SWaT_A1A2     # Specific dataset
    python scripts/run_all_base.py --set A --start-from 5       # Resume from index
    python scripts/run_all_base.py --set A --list               # List datasets
"""

import sys
import subprocess
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def main():
    parser = argparse.ArgumentParser(description='Run all base experiments')
    parser.add_argument('--set', type=str, required=True, choices=['A', 'B'],
                        help='Config preset (A=d128/p5/k3, B=d256/p20/k5)')
    parser.add_argument('--only', type=str, default=None,
                        help='Run only specific dataset (e.g., SWaT_A1A2)')
    parser.add_argument('--start-from', type=int, default=0,
                        help='Start from dataset index (0-based)')
    parser.add_argument('--list', action='store_true',
                        help='List datasets only')
    parser.add_argument('--output-base', type=str, default=None,
                        help='Override output base directory')
    args = parser.parse_args()

    # Build command for run_base_experiments.py
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / 'scripts' / 'run_base_experiments.py'),
        '--set', args.set,
    ]

    if args.only:
        cmd.extend(['--dataset', args.only])
    if args.start_from > 0:
        cmd.extend(['--start-from', str(args.start_from)])
    if args.list:
        cmd.append('--list')
    if args.output_base:
        cmd.extend(['--output-base', args.output_base])

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
