#!/usr/bin/env python3
"""Patch existing excl22 epoch_metrics.json with teacher_pak_auc_* metrics.

Uses saved epoch_scores (teacher_recon_error) and the evaluator's
compute_metrics_with_exclusion to compute proper teacher-only PA%K AUC
metrics for SWaT/A1A2_excl22.

This is a one-time patch script for experiments 40-115 that were run
before the fix in run_base_experiments.py (line 1229-1233).

Usage:
    python scripts/patch_excl22_teacher_metrics.py
"""

import json
import glob
import sys
import os
import numpy as np
from pathlib import Path
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mae_anomaly.evaluator import compute_metrics_with_exclusion
from mae_anomaly.datasets.loaders import load_swat_a1a2_raw

# Module-level namedtuple for pickling compatibility with multiprocessing
Region = namedtuple('Region', ['start', 'end'])


def load_swat_test_data():
    """Load SWaT test labels and anomaly regions."""
    raw = load_swat_a1a2_raw()
    full_data, full_labels = raw[0], raw[1]
    train_ratio = raw[4]

    n_train = int(len(full_data) * train_ratio)
    test_labels = full_labels[n_train:]

    regions = []
    in_anom = False
    start = 0
    for i in range(len(test_labels)):
        if test_labels[i] == 1 and not in_anom:
            start = i
            in_anom = True
        elif test_labels[i] == 0 and in_anom:
            regions.append(Region(start, i))
            in_anom = False
    if in_anom:
        regions.append(Region(start, len(test_labels)))

    return test_labels, regions


def find_largest_region(regions):
    """Find the largest anomaly region (region 22 in SWaT)."""
    return max(regions, key=lambda r: r.end - r.start)


def patch_experiment(mae_dir, test_labels, anomaly_regions, largest_region):
    """Patch a single experiment's excl22 epoch_metrics with teacher_pak_auc_*."""

    excl22_dir = mae_dir / 'SWaT' / 'A1A2_excl22'
    em_file = excl22_dir / 'epoch_metrics.json'
    score_dir = excl22_dir / 'epoch_scores'

    if not em_file.exists() or not score_dir.exists():
        return 'skip_no_files'

    data = json.load(open(em_file))

    # Already patched?
    if data['epochs'] and data['epochs'][0].get('teacher_pak_auc_f1') is not None:
        return 'already_patched'

    score_files = sorted(glob.glob(str(score_dir / 'epoch_*_scores.npz')))
    if not score_files:
        return 'skip_no_scores'

    # Build epoch -> score file mapping
    epoch_scores = {}
    for sf in score_files:
        ep_num = int(os.path.basename(sf).split('_')[1])
        epoch_scores[ep_num] = sf

    patched_count = 0
    for ep in data['epochs']:
        epoch_num = ep['epoch']
        if epoch_num not in epoch_scores:
            continue

        scores = np.load(epoch_scores[epoch_num])
        if 'teacher_recon_error' not in scores:
            continue

        teacher_scores = scores['teacher_recon_error']
        ml = min(len(teacher_scores), len(test_labels))

        teacher_em = compute_metrics_with_exclusion(
            teacher_scores[:ml], test_labels[:ml],
            anomaly_regions, largest_region
        )

        if not teacher_em:
            continue

        # Store teacher metrics with teacher_pak_auc_* prefix
        # (same logic as run_base_experiments.py line 515-522 and 1229-1233)
        ep['teacher_prc_auc'] = teacher_em.get('prc_auc', 0)
        ep['teacher_f1_t'] = teacher_em.get('f1_t', 0)
        for m in ['prc_auc', 'roc_auc', 'f1', 'f1_t', 'precision', 'recall',
                  'f1_raw', 'f1_t_raw', 'precision_raw', 'recall_raw']:
            ep[f'teacher_pak_auc_{m}'] = teacher_em.get(f'pak_auc_{m}', 0)
        ep['teacher_pak_auc_prc_auc'] = teacher_em.get('pak_auc_prc_auc', 0)
        patched_count += 1

    if patched_count > 0:
        with open(em_file, 'w') as f:
            json.dump(data, f, indent=2)
        return f'patched_{patched_count}_epochs'

    return 'skip_no_patches'


def _worker(mae_dir_str, test_labels, anomaly_regions, largest_region):
    """Worker function for multiprocessing."""
    mae_dir = Path(mae_dir_str)
    mae_id = int(mae_dir.name.split('_')[0])
    result = patch_experiment(mae_dir, test_labels, anomaly_regions, largest_region)
    return mae_id, result


def main():
    import multiprocessing as mp
    n_workers = min(mp.cpu_count(), 10)

    print("Loading SWaT test data...")
    test_labels, anomaly_regions = load_swat_test_data()
    largest_region = find_largest_region(anomaly_regions)
    print(f"  Test: {len(test_labels)} points, {len(anomaly_regions)} regions")
    print(f"  Largest region: [{largest_region.start}, {largest_region.end})")

    mae_base = Path(__file__).resolve().parent.parent / 'results' / 'experiments'
    mae_dirs = sorted([d for d in mae_base.iterdir() if d.is_dir() and d.name[0].isdigit()],
                       key=lambda x: int(x.name.split('_')[0]))
    mae_dirs = [d for d in mae_dirs if 40 <= int(d.name.split('_')[0]) <= 115]

    # Filter to unpatched only
    unpatched = []
    for d in mae_dirs:
        em = d / 'SWaT' / 'A1A2_excl22' / 'epoch_metrics.json'
        if not em.exists():
            continue
        data = json.load(open(em))
        if data['epochs'] and data['epochs'][0].get('teacher_pak_auc_f1') is not None:
            continue
        unpatched.append(d)

    print(f"\nPatching {len(unpatched)}/{len(mae_dirs)} unpatched experiments with {n_workers} workers...")
    sys.stdout.flush()

    results = {}
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_worker, str(d), test_labels, anomaly_regions, largest_region): d
            for d in unpatched
        }
        for future in as_completed(futures):
            mae_id, result = future.result()
            results[mae_id] = result
            status = 'OK' if 'patched' in result else result
            print(f"  exp {mae_id:>3}: {status}", flush=True)

    patched = sum(1 for v in results.values() if 'patched' in v)
    skipped = sum(1 for v in results.values() if 'skip' in v)
    already = sum(1 for v in results.values() if 'already' in v)
    print(f"\nDone: {patched} patched, {already} already done, {skipped} skipped")


if __name__ == '__main__':
    main()
