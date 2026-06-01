#!/usr/bin/env python
"""Build the full re-run queue: 271(canon) -> 274(canon) -> 285..313, on 7 datasets,
amp=bf16, lr=0.001. Each entry carries its EXACT experiment number so the launcher
can pin the output dir number (no auto-numbering tangle).

Ground truth = the original queue config_overrides:
  - 271/274 canonical: temp/0529/queue_runtime_20260529_100418.json
  - 285-313:           temp/0529/queue_runtime_lr_plus_lr2_20260529_225351.json
Modifications applied to every override: learning_rate -> 0.001 (was 0.0015),
amp_dtype kept bf16 (already present). Dataset list overridden to the 7 targets.

Usage: python scripts/build_fullrerun_queue.py <stamp>   # writes configs/queue_fullrerun_<stamp>.json
"""
import json, re, sys

DATASETS = ['PSM', 'SWaT_A1A2', 'WaDi_A1', 'WaDi_A2', 'SMD_concat', 'SMAP_concat', 'MSL_concat']
Q_CANON = 'temp/0529/queue_runtime_20260529_100418.json'
Q_2 = 'temp/0529/queue_runtime_lr_plus_lr2_20260529_225351.json'


def set_lr_bf16(ov: str) -> str:
    """Force learning_rate=0.001 and amp_dtype=bf16 in a space-separated override string."""
    toks = ov.split()
    out, seen_lr, seen_amp = [], False, False
    for t in toks:
        if t.startswith('learning_rate='):
            out.append('learning_rate=0.001'); seen_lr = True
        elif t.startswith('amp_dtype='):
            out.append('amp_dtype=bf16'); seen_amp = True
        else:
            out.append(t)
    if not seen_lr:
        out.append('learning_rate=0.001')
    if not seen_amp:
        out.append('amp_dtype=bf16')
    return ' '.join(out)


def expnum(name):
    m = re.match(r'exp(\d+)', name)
    return int(m.group(1)) if m else None


def main():
    stamp = sys.argv[1] if len(sys.argv) > 1 else 'STAMP'
    canon = {x['name']: x for x in json.load(open(Q_CANON)).get('experiments', [])}
    q2 = {x['name']: x for x in json.load(open(Q_2)).get('experiments', [])}

    entries = []
    # 271 canon, 274 canon
    for nm in ('exp271_271canon_baseline', 'exp274_274canon_balsamp'):
        e = canon[nm]
        entries.append((expnum(nm), nm, e['config_override']))
    # 285-313 from q2 (canonical singles; skip _lr/_lr2)
    for nm, e in q2.items():
        n = expnum(nm)
        if n is not None and 285 <= n <= 313 and '_lr' not in nm.lower().split('_', 1)[-1][:3]:
            # exclude 271_lr/274_lr already out of range; 285-313 have no _lr variants
            entries.append((n, nm, e['config_override']))
    entries.sort(key=lambda x: x[0])

    experiments = []
    for num, nm, ov in entries:
        # dir suffix: drop the 'exp<NUM>_' prefix → readable name
        suffix = re.sub(r'^exp\d+_', '', nm)
        experiments.append({
            'name': nm,
            'exp_num': num,
            'dir_suffix': suffix,
            'set': 'C',
            'dataset': list(DATASETS),
            'config_override': set_lr_bf16(ov),
        })

    out = {
        'note': ('Full re-run 271(canon)->274(canon)->285..313 on 7 datasets '
                 '(PSM, SWaT_A1A2, WaDi_A1, WaDi_A2, SMD_concat, SMAP_concat, MSL_concat). '
                 'amp=bf16, lr=0.001. exp_num pins the output dir number (no auto-numbering). '
                 'config_override = queue ground truth with learning_rate forced 0.001.'),
        'datasets': DATASETS,
        'n_experiments': len(experiments),
        'experiments': experiments,
    }
    outp = f'configs/queue_fullrerun_{stamp}.json'
    json.dump(out, open(outp, 'w'), indent=2, ensure_ascii=False)
    print(f"WROTE {outp}  ({len(experiments)} experiments × {len(DATASETS)} datasets)")
    print("nums:", [e['exp_num'] for e in experiments])
    # sanity: every override has lr=0.001 + bf16
    bad = [e['exp_num'] for e in experiments
           if 'learning_rate=0.001' not in e['config_override'] or 'amp_dtype=bf16' not in e['config_override']]
    print("lr/amp sanity bad:", bad if bad else "OK (all lr=0.001, bf16)")


if __name__ == '__main__':
    main()
