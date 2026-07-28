#!/usr/bin/env python
"""dcdetector 'neg' convention cache — permanent recompute / extend script.

dcdetector follows the paper protocol with the SIGN-INVERTED score
(neg = -canonical anomaly score; user decision, sign-inversion audit 2026-07-19).
Per-epoch neg metrics live in

    comparison/results/experiments/_dcdetector_neg_cache.json

keyed ``{seed}|{entity}|ep{epoch}`` (e.g. ``"44|PSM|ep1"``) and are consumed
read-only by ``comparison/build_results_md.py`` (``_dc_neg``).

The original cache-building script lived in /tmp and was lost to a reboot;
this file is its permanent replacement. Methodology (verified to reproduce the
existing 20 cache entries bit-exactly, see --verify):

  * score source  : ``<exp_dir>/<entity>/dcdetector/epoch_scores/epoch_XXX_scores.npz``
                    (``anomaly_score`` array, float32) — mtime recorded as
                    ``_score_mtime`` in each cache entry.
  * neg transform : ``neg = -scores.astype(np.float64)`` — float64 REQUIRED:
                    the original cache was built on float64-negated scores; with
                    float32 the PA%K threshold grid (float32 linspace under
                    NumPy 1.26 value-based promotion) shifts by ~1e-8 and
                    ``pa_100_roc_auc`` on WaDi/A1 deviates by 4.4e-08
                    (verified 2026-07-22; float64 reproduces all fields exactly).
  * full entities : ``comparison.baseline_common.compute_all_metrics(neg, test_y,
                    anomaly_regions, lite=False)`` — the EXACT function the
                    baseline pipeline uses (unified with MAE
                    ``compute_full_metric_set``; identical threshold convention,
                    PA%K AUC, VUS, affiliation, AR variants).
  * SWaT excl22   : derived from the SWaT/A1A2_full scores via
                    ``mae_anomaly.evaluator.compute_metrics_with_exclusion(neg,
                    test_y, anomaly_regions, excl_region, lite=False)`` — the
                    same function behind the pipeline's ``excl22_*`` columns —
                    with ``_``-prefixed diagnostic keys dropped and the SAME
                    ``_score_mtime`` as the full entity (matches original cache).

CPU-only: no model is loaded or trained; CUDA is masked before any import.

Usage:
  # methodology reproduction check against existing cache keys (NO write):
  python comparison/dcdetector_neg_recompute.py --seeds 42 --verify

  # compute + merge new seed(s) into the cache (existing keys preserved):
  python comparison/dcdetector_neg_recompute.py --seeds 44
"""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')   # CPU-only — never touch the GPU chain

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EXP = ROOT / 'comparison' / 'results' / 'experiments'
NEG_CACHE = EXP / '_dcdetector_neg_cache.json'

SUFFIX = '20260606_175756'
SEEDS = [42, 43, 40, 41, 44]                                   # k=1..5 (8-k dirs)
SEED_DIR = {s: EXP / f'8-{k}_{SUFFIX}_baseline' for k, s in enumerate(SEEDS, 1)}

# (experiment config name, entity string used in cache keys / result dirs)
DATASETS = [
    ('psm_normalonly',            'PSM'),
    ('swat_a1a2_normalonly',      'SWaT/A1A2_full'),           # + derived SWaT/A1A2_excl22
    ('wadi_14days_A1_normalonly', 'WaDi/A1'),
    ('wadi_14days_A2_normalonly', 'WaDi/A2'),
]


_LOADER_CACHE = {}


def _get_eval_data(cfg_name: str, cfg: dict):
    """Load (test_y, regions, excl_region) once per dataset (cached across seeds)."""
    if cfg_name not in _LOADER_CACHE:
        from comparison.baseline_common import load_data_from_config
        loader = load_data_from_config(cfg, normalize_mode='none')
        _, test_y = loader.get_test_data()
        regions = loader.get_test_anomaly_regions()
        excl_region = loader.excl_region if cfg.get('has_excl22') else None
        _LOADER_CACHE[cfg_name] = (test_y, regions, excl_region)
    return _LOADER_CACHE[cfg_name]


def _json_safe(v):
    if isinstance(v, (np.floating, np.integer)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def compute_seed_entries(seed: int) -> dict:
    """Compute all neg-convention cache entries for one seed (all epochs found)."""
    from comparison.baseline_common import compute_all_metrics
    from comparison.experiment_configs import EXPERIMENT_CONFIGS
    from mae_anomaly.evaluator import compute_metrics_with_exclusion

    exp_dir = SEED_DIR[seed]
    entries = {}
    for cfg_name, entity in DATASETS:
        cfg = EXPERIMENT_CONFIGS[cfg_name]
        cell = exp_dir / entity / 'dcdetector'
        ep_dir = cell / 'epoch_scores'
        ep_files = sorted(ep_dir.glob('epoch_*_scores.npz')) if ep_dir.exists() else []
        if not ep_files:
            print(f'  [skip] {seed}|{entity}: no epoch_scores found under {cell}')
            continue

        # dcdetector is SELF_NORMALIZING (normalize_mode='none' in the runs);
        # labels/regions are normalization-independent, but mirror the run anyway.
        test_y, regions, excl_region = _get_eval_data(cfg_name, cfg)

        for f in ep_files:
            ep = int(f.stem.split('_')[1])                     # epoch_001_scores -> 1
            scores = np.load(f)['anomaly_score']
            if len(scores) != len(test_y):
                raise RuntimeError(
                    f'{seed}|{entity}|ep{ep}: score len {len(scores)} != test len {len(test_y)}')
            neg = -scores.astype(np.float64)   # float64 REQUIRED (see module docstring)
            mtime = os.path.getmtime(f)

            m = compute_all_metrics(neg, test_y, regions, lite=False)
            entry = {k: _json_safe(v) for k, v in m.items()}
            entry['_score_mtime'] = mtime
            entries[f'{seed}|{entity}|ep{ep}'] = entry
            print(f'  [done] {seed}|{entity}|ep{ep}  pak_auc_f1={entry["pak_auc_f1"]:.6f}')

            if excl_region is not None:
                ex = compute_metrics_with_exclusion(neg, test_y, regions, excl_region,
                                                    lite=False)
                ex_entry = {k: _json_safe(v) for k, v in ex.items()
                            if not k.startswith('_')}
                ex_entry['_score_mtime'] = mtime
                ex_key = f'{seed}|{entity.replace("_full", "_excl22")}|ep{ep}'
                entries[ex_key] = ex_entry
                print(f'  [done] {ex_key}  pak_auc_f1={ex_entry["pak_auc_f1"]:.6f}')
    return entries


def verify_against_cache(new_entries: dict, cache: dict, tol: float = 1e-9) -> bool:
    """Field-by-field diff of recomputed entries vs existing cache. True = all match."""
    ok = True
    checked = 0
    for key, new in sorted(new_entries.items()):
        if key not in cache:
            print(f'  [verify] {key}: not in cache (new key) — skipped')
            continue
        old = cache[key]
        checked += 1
        mismatches = []
        for f in sorted(set(old) | set(new)):
            a, b = old.get(f, '<absent>'), new.get(f, '<absent>')
            if isinstance(a, float) and isinstance(b, float):
                if abs(a - b) > tol:
                    mismatches.append((f, a, b, abs(a - b)))
            elif a != b:
                mismatches.append((f, a, b, None))
        if mismatches:
            ok = False
            print(f'  [verify] {key}: {len(mismatches)} MISMATCH(ES)')
            for f, a, b, d in mismatches:
                print(f'      {f}: cache={a!r} recomputed={b!r}'
                      + (f' |diff|={d:.3e}' if d is not None else ''))
        else:
            print(f'  [verify] {key}: all {len(old)} fields match (tol {tol:g})')
    print(f'  [verify] {checked} cached key(s) checked -> {"PASS" if ok else "FAIL"}')
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--seeds', type=int, nargs='+', required=True,
                    help=f'seeds to (re)compute (dir map: {{s: f"8-k" for k,s}} over {SEEDS})')
    ap.add_argument('--verify', action='store_true',
                    help='compare recomputed values against existing cache; NEVER writes')
    ap.add_argument('--overwrite', action='store_true',
                    help='allow replacing existing cache keys (default: preserve)')
    args = ap.parse_args()

    cache = json.load(open(NEG_CACHE)) if NEG_CACHE.exists() else {}
    print(f'cache: {NEG_CACHE} ({len(cache)} keys)')

    new_entries = {}
    for seed in args.seeds:
        if seed not in SEED_DIR:
            raise SystemExit(f'unknown seed {seed}; known: {SEEDS}')
        print(f'== seed {seed} -> {SEED_DIR[seed].name}')
        new_entries.update(compute_seed_entries(seed))

    if args.verify:
        ok = verify_against_cache(new_entries, cache)
        sys.exit(0 if ok else 1)

    added = skipped = 0
    for k, v in new_entries.items():
        if k in cache and not args.overwrite:
            skipped += 1
            continue
        cache[k] = v
        added += 1
    tmp = NEG_CACHE.with_suffix('.json.tmp')
    with open(tmp, 'w') as fh:
        json.dump(cache, fh, indent=1)
    os.replace(tmp, NEG_CACHE)
    print(f'merged: +{added} keys ({skipped} existing preserved) -> {len(cache)} total')


if __name__ == '__main__':
    main()
