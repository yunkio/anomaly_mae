#!/usr/bin/env python
"""Build the retrain queue for the 4:1 / no-FM lambda change.

Re-runs ONLY the cells whose best epoch FLIPPED under the new score, restricted
to the canonical experiments {271,274,285,286,287}. lr / lr2 variants and 288
are excluded by user instruction (lr/lr2 = recompute-only; 288 = discarded).

Reads:  /tmp/lambda_bestepoch_classification.json  (flip determination, nt=200)
        temp/0529/queue_runtime_20260529_100418.json (original canonical configs)
Writes: configs/queue_retrain_lambda_<stamp>.json   (stamp passed via argv[1])

A flipped SWaT cell (full OR excl22) maps to the single 'SWaT_A1A2' dataset
(one training drives both eval modes). Datasets per experiment = union of its
flipped cells' dataset specs.
"""
import json, sys, os

CANON = {'271', '274', '285', '286', '287'}
DS_MAP = {'PSM': 'PSM', 'WaDi/A1': 'WaDi_A1', 'WaDi/A2': 'WaDi_A2',
          'SWaT/A1A2_full': 'SWaT_A1A2', 'SWaT/A1A2_excl22': 'SWaT_A1A2'}
CLASS = '/tmp/lambda_bestepoch_classification.json'
ORIG_QUEUE = 'temp/0529/queue_runtime_20260529_100418.json'


def base_of(run):
    # '271_20260529_...' -> '271' ; '271_lr_...' -> '271_lr'
    parts = run.split('_')
    return parts[0] if parts[1].startswith('2026') else f"{parts[0]}_{parts[1]}"


def main():
    stamp = sys.argv[1] if len(sys.argv) > 1 else 'STAMP'
    cls = json.load(open(CLASS))
    orig = json.load(open(ORIG_QUEUE))
    orig_ents = orig.get('experiments', orig)
    # canonical original entries keyed by base exp number
    cfg_by_base = {}
    for e in orig_ents:
        nm = e.get('name', '')           # e.g. exp271_271canon_baseline
        num = nm.replace('exp', '').split('_')[0]
        if num in CANON and 'lr' not in nm.lower():
            cfg_by_base[num] = e

    # collect flipped canonical cells
    flips = [c for c in cls if c['flip']]
    per_exp = {}   # num -> {'run':..., 'ds':set(), 'cells':[]}
    skipped = []
    for c in flips:
        b = base_of(c['run'])
        if b not in CANON:
            skipped.append((c['run'], c['ds'], 'non-canonical (lr/lr2) -> recompute-only'))
            continue
        d = per_exp.setdefault(b, {'run': c['run'], 'ds': set(), 'cells': []})
        d['ds'].add(DS_MAP[c['ds']])
        d['cells'].append({'ds': c['ds'], 'old_best': c['old_best'], 'new_best': c['new_best']})

    experiments = []
    DS_ORDER = ['SWaT_A1A2', 'WaDi_A1', 'WaDi_A2', 'PSM']
    for num in sorted(per_exp):
        info = per_exp[num]
        oe = cfg_by_base.get(num)
        if oe is None:
            print(f"!! no original config for {num} — SKIP"); continue
        ds = [d for d in DS_ORDER if d in info['ds']]
        experiments.append({
            'name': oe['name'],
            'set': oe.get('set', 'C'),   # canonical exps use set C (run_base --set ∈ {A,B,C})
            'orig_run_dir': info['run'],
            'dataset': ds,
            'flipped_cells': info['cells'],   # documentation (old->new best epoch)
            'config_override': oe['config_override'],
        })

    out = {
        'note': ('Retrain queue for 4:1/no-FM lambda change. Re-runs ONLY flipped '
                 'canonical cells (271/274/285/286/287); lr/lr2 + 288 excluded. '
                 'Requires scoring.py 4:1 change active so training selects + '
                 'checkpoints the NEW best epoch. dataset list = flipped datasets only.'),
        'classification_source': CLASS,
        'n_flipped_canonical_experiments': len(experiments),
        'recompute_only_flips_excluded': skipped,
        'experiments': experiments,
    }
    outp = f'configs/queue_retrain_lambda_{stamp}.json'
    json.dump(out, open(outp, 'w'), indent=2, ensure_ascii=False)
    print(f"WROTE {outp}  ({len(experiments)} experiments)")
    for e in experiments:
        cells = ', '.join(f"{c['ds']}({c['old_best']}->{c['new_best']})" for c in e['flipped_cells'])
        print(f"  {e['name']:<26} ds={e['dataset']}")
        print(f"      flipped: {cells}")
    if skipped:
        print(f"\n  recompute-only (lr/lr2 flips, NOT retrained): {len(skipped)}")
        for r, d, why in skipped:
            print(f"    {r.split('_2026')[0]}/{d}")


if __name__ == '__main__':
    main()
