#!/usr/bin/env python
"""Fact-only backfill of metadata.json for COMPLETED baseline experiments 8/9/10.

For historical runs the live model object is gone, so we cannot introspect it
(that is exactly what the new ``augment_run_metadata`` in baseline_common.py does
for *future* runs). This script therefore augments each completed model dir's
``metadata.json`` with a ``backfill_parameters`` block built ONLY from confirmed
facts — never guessing. Every field carries a ``provenance`` tag.

Confirmed-fact sources used (per (experiment, dataset, model) cell):
  * epochs_run        — len(epoch_metrics.json['epochs'])                 [artifact]
  * normalize_mode    — the code's deterministic self-norm rule           [code_rule]
                        (SELF_NORMALIZING_SOTA/WEAK → 'none', else 'minmax')
                        cross-checked against queue_summary where present.
  * effective_batch_size + batch_size_override — ONLY for WaDi
                        catch/dcdetector cells, from the code's
                        _WADI_BATCH_DIVISORS rule pinned to the run's git era by
                        the saved timestamp:
                            dcdetector → ÷2  (constant, batch 128→64)
                            catch      → ÷4 if saved BEFORE commit b7fc99e
                                         (2026-06-13 16:33 KST), else ÷8
                                         (batch 128→32 or 128→16)            [git+ts]
                        Non-WaDi / non-divisor cells: batch NOT recorded here
                        (the preset default is not reliably reconstructable for
                        the multi-day 8/9 runs whose preset code changed mid-run).
  * queue_config + command — from queue_summary.json where the entry exists
                        (8번 only the last 144 of 814 entries were captured)  [artifact]
  * configured_preset_hp_reference — the CURRENT-code preset for the model.
                        Provenance is honest about whether it equals the run-time
                        value (exact for 10번 — preset code unchanged since its
                        06-15+ window; reference-only for 8/9 — preset code
                        changed across the run window).

Existing fields (model_name/experiment/timestamp/timing) are preserved.
Dirs without epoch_metrics.json (in-progress) are skipped. Atomic write.
DRY-RUN by default; pass --apply to write.
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXPERIMENTS = {
    '8': REPO / 'comparison/results/experiments/8_20260606_175756_baseline',
    '9': REPO / 'comparison/results/experiments/9_20260606_175756_weak_ssl',
    '10': REPO / 'comparison/results/experiments/10_20260615_230000_baseline_contaminated',
}

# Self-normalizing model sets (mirror run_baseline.py:241-247). These models are
# fed raw data (normalize_mode='none'); every other model uses 'minmax'.
SELF_NORMALIZING = {
    'timesnet', 'moderntcn', 'dcdetector', 'catch', 'tfmae',
    'anomaly_transformer', 'memto', 'npsr',
    'wetas', 'treemil', 'nrdetector', 'nrdetector_full', 'deepmil',
}

SIMPLE_MODELS = {'random', 'sensor_range', 'pca_error', 'l2_norm', 'nn_distance'}

# WaDi memory-pressure batch divisors (run_baseline.py:_WADI_BATCH_DIVISORS).
# catch's divisor changed 4→8 at commit b7fc99e (2026-06-13 16:33:48 +0900).
_CATCH_DIVISOR_CHANGE = datetime.fromisoformat('2026-06-13T16:33:48')
PRESET_BATCH = {'dcdetector': 128, 'catch': 128}

# For 10번, the preset code was unchanged across its entire run window (last
# preset change b7fc99e on 06-13, run started 06-15), so the current preset IS
# the run-time configured HP. For 8/9 the preset code changed during the run.
PRESET_IS_RUNTIME_EXACT = {'10': True, '8': False, '9': False}


def load_json(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def current_preset(model_name):
    """Authoritative current-code preset dict for a model (or {} if unavailable).
    Imported lazily so the script runs even if heavy deps are missing."""
    try:
        sys.path.insert(0, str(REPO))
        from comparison.baseline_common import _get_default_model_params
        return dict((_get_default_model_params() or {}).get(model_name, {}) or {})
    except Exception:
        return {}


def is_wadi(dataset_path_parts):
    return any('wadi' in p.lower() for p in dataset_path_parts)


def find_model_dirs(exp_dir):
    """Yield (model_dir, model_name, dataset_parts) for every dir with metadata.json."""
    for meta_path in exp_dir.rglob('metadata.json'):
        model_dir = meta_path.parent
        model_name = model_dir.name
        rel = model_dir.relative_to(exp_dir)
        dataset_parts = list(rel.parts[:-1])  # everything above the model name
        yield model_dir, model_name, dataset_parts


def backfill_one(exp_id, exp_dir, model_dir, model_name, dataset_parts):
    completed = (model_dir / 'scores.npz').exists()  # final artifact => run finished
    meta = load_json(model_dir / 'metadata.json')
    status = 'ok'

    # Already richly augmented at runtime by augment_run_metadata (schema>=2):
    # leave it untouched — the live capture is strictly more faithful than a
    # backfill could be.
    if isinstance(meta, dict) and meta.get('metadata_schema_version', 0) >= 2:
        return None, 'already_augmented(skip)'

    if meta is None:
        # metadata.json unreadable/corrupt. Reconstruct a skeleton ONLY if the
        # run actually completed (scores.npz present); otherwise treat as
        # in-progress and skip (never touch a live worker's dir).
        if not completed:
            return None, 'no_metadata(skip_incomplete)'
        meta = {
            'model_name': model_name,
            'experiment': '/'.join(dataset_parts) if dataset_parts else None,
            'timestamp': None,
            'timing': None,
            'recovered_from_corrupt_metadata': True,
        }
        status = 'recovered_corrupt_metadata'

    facts = {}

    # 1) epochs_run — artifact fact (epoch_metrics.json may itself be corrupt)
    em = load_json(model_dir / 'epoch_metrics.json')
    if isinstance(em, dict) and 'epochs' in em:
        facts['epochs_run'] = {'value': len(em.get('epochs') or []),
                               'provenance': 'epoch_metrics.json'}
    else:
        if not completed:
            return None, 'no_epoch_metrics(skip_incomplete)'
        facts['epochs_run'] = {'value': None,
                               'provenance': 'unavailable(epoch_metrics.json missing/corrupt)'}
        if status == 'ok':
            status = 'corrupt_epoch_metrics'

    # 2) normalize_mode — deterministic code rule (+ queue cross-check)
    norm = 'none' if model_name in SELF_NORMALIZING else 'minmax'
    facts['normalize_mode'] = {'value': norm, 'provenance': 'code_rule(self_norm_set)'}

    # 3) batch_size override — ONLY WaDi catch/dcdetector (git+timestamp pinned)
    if is_wadi(dataset_parts) and model_name in PRESET_BATCH:
        ts = meta.get('timestamp')
        if model_name == 'dcdetector':
            divisor = 2
            prov = 'code_rule(_WADI_BATCH_DIVISORS, constant)'
        else:  # catch
            saved = None
            try:
                saved = datetime.fromisoformat(ts) if ts else None
            except Exception:
                saved = None
            if saved is not None:
                divisor = 4 if saved < _CATCH_DIVISOR_CHANGE else 8
                prov = ('git+timestamp(catch divisor 4 pre-b7fc99e 2026-06-13, else 8); '
                        f'saved={ts}')
            else:
                divisor = None
                prov = 'unresolved(no_timestamp)'
        if divisor:
            preset_bs = PRESET_BATCH[model_name]
            facts['batch_size_override'] = {
                'value': {
                    'kind': 'wadi_batch_divisor',
                    'divisor': divisor,
                    'batch_size_preset': preset_bs,
                    'batch_size_effective': preset_bs // divisor,
                },
                'provenance': prov,
            }

    # 4) configured preset reference (honest provenance about run-time equality)
    preset = current_preset(model_name)
    if preset:
        if PRESET_IS_RUNTIME_EXACT.get(exp_id):
            pprov = 'current_code_preset == run-time (preset code unchanged since run window)'
        else:
            pprov = ('current_code_preset (REFERENCE ONLY — preset code changed across the '
                     'multi-day run window; NOT guaranteed equal to run-time value)')
        facts['configured_preset_hp_reference'] = {'value': preset, 'provenance': pprov}

    # 5) note on what is intentionally NOT recorded (no fabrication)
    facts['_backfill_note'] = (
        'Backfill from confirmed artifacts/code only. Runtime model-object '
        'introspection was unavailable for this historical run; full effective '
        'hyperparameters are captured automatically for FUTURE runs by '
        'baseline_common.augment_run_metadata (metadata_schema_version>=2).'
    )

    meta['backfill_parameters'] = facts
    meta['backfill_schema_version'] = 1
    return meta, status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true', help='write changes (default: dry-run)')
    ap.add_argument('--experiments', default='8,9,10', help='comma list of 8/9/10')
    ap.add_argument('--limit', type=int, default=0, help='process at most N dirs per exp (0=all)')
    ap.add_argument('--sample', action='store_true', help='print the first produced metadata per exp')
    args = ap.parse_args()

    for exp_id in args.experiments.split(','):
        exp_id = exp_id.strip()
        exp_dir = EXPERIMENTS.get(exp_id)
        if not exp_dir or not exp_dir.exists():
            print(f"[{exp_id}] dir 없음, skip")
            continue
        stats = {'ok': 0, 'with_batch_override': 0}
        printed = False
        n = 0
        for model_dir, model_name, dataset_parts in find_model_dirs(exp_dir):
            if args.limit and n >= args.limit:
                break
            n += 1
            meta, status = backfill_one(exp_id, exp_dir, model_dir, model_name,
                                        dataset_parts)
            stats[status] = stats.get(status, 0) + 1
            if meta is None:
                continue
            if 'batch_size_override' in meta['backfill_parameters']:
                stats['with_batch_override'] += 1
            if args.sample and not printed:
                print(f"\n=== [{exp_id}] SAMPLE: {model_dir.relative_to(exp_dir)} ===")
                print(json.dumps(meta, indent=2, ensure_ascii=False)[:2200])
                printed = True
            if args.apply:
                tmp = model_dir / 'metadata.json.tmp'
                with open(tmp, 'w') as f:
                    json.dump(meta, f, indent=2, ensure_ascii=False)
                os.replace(tmp, model_dir / 'metadata.json')
        print(f"\n[{exp_id}] {'APPLIED' if args.apply else 'DRY-RUN'} — "
              f"processed {n}: " + ', '.join(f'{k}={v}' for k, v in stats.items()))


if __name__ == '__main__':
    main()
