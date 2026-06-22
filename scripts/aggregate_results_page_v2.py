"""Recompute the 4-metric Results page (PAK_F1 / PAK_PRC / Affiliation_F1 / VUS_PR)
from completed experiment_metadata.json across the active subset 271·274·285-337.

Per-DS cell = metric (rank, best_ep). SMD/SMAP/MSL = per-entity group average.
Starred(*) DS = SWaT(excl22), WaDi A1, WaDi A2, PSM -> Avg(4*) / RankAvg(4*) / medal.

Run: python scripts/aggregate_results_page_v2.py [--verify]
Outputs temp/results_page_v2.json (per-exp per-metric values + ranks + tables).
"""
import json, glob, os, sys
from collections import defaultdict

ROOT = '/home/ykio/notebooks/TSMAE/results/experiments'
SUBSET = [271, 274] + list(range(285, 338))
METRICS = ['pak_auc_f1', 'pak_auc_prc_auc', 'affiliation_f1', 'vus_pr']
SMD = ['machine-1-2','machine-1-3','machine-1-4','machine-1-5','machine-1-6','machine-1-7','machine-1-8',
       'machine-2-1','machine-2-3','machine-2-4','machine-2-5','machine-2-6','machine-2-7',
       'machine-3-1','machine-3-2','machine-3-3','machine-3-4','machine-3-5','machine-3-6','machine-3-7','machine-3-9','machine-3-10']
SMAP = ['G-7','P-1','P-4','T-1','T-3']
MSL = ['C-1','C-2','F-7','P-11','T-13']
# DS columns (key -> relative path under exp dir); group cols handled separately
SINGLE = {
    'swat_full':  'SWaT/A1A2_full',
    'swat_excl22':'SWaT/A1A2_excl22',
    'wadi_A1':    'WaDi/A1',
    'wadi_A2':    'WaDi/A2',
    'psm':        'PSM',
}
GROUP = {'smd': ('SMD', SMD), 'smap': ('SMAP', SMAP), 'msl': ('MSL', MSL)}
ALL_COLS = list(SINGLE) + list(GROUP)
STAR = ['swat_excl22', 'wadi_A1', 'wadi_A2', 'psm']   # 4* columns

def load(p):
    try: return json.load(open(p))
    except Exception: return None

def cell_from_meta(meta_path):
    d = load(meta_path)
    if not d: return None
    m = d.get('metrics', {}); t = d.get('timing', {})
    out = {k: m.get(k) for k in METRICS}
    out['best_ep'] = t.get('best_epoch')
    if out['pak_auc_f1'] is None: return None
    return out

def exp_dir(n):
    ds = sorted(glob.glob(os.path.join(ROOT, f'{n}_*')))
    return ds[-1] if ds else None

def aggregate_exp(n):
    base = exp_dir(n)
    if not base: return None
    res = {}
    for col, rel in SINGLE.items():
        c = cell_from_meta(os.path.join(base, rel, 'experiment_metadata.json'))
        if c: res[col] = c
    for col, (sub, ents) in GROUP.items():
        vals = defaultdict(list); eps = []
        for e in ents:
            c = cell_from_meta(os.path.join(base, sub, e, 'experiment_metadata.json'))
            if not c: continue
            for mk in METRICS:
                if c.get(mk) is not None: vals[mk].append(c[mk])
            if c.get('best_ep') is not None: eps.append(c['best_ep'])
        if vals.get('pak_auc_f1'):
            res[col] = {mk: (sum(vals[mk])/len(vals[mk]) if vals.get(mk) else None) for mk in METRICS}
            res[col]['best_ep'] = round(sum(eps)/len(eps)) if eps else None
            res[col]['_n'] = len(vals['pak_auc_f1'])
    return res or None

def main():
    data = {n: aggregate_exp(n) for n in SUBSET}
    data = {n: v for n, v in data.items() if v}
    present = sorted(data)
    # ranks: per metric, per col, descending (1=best)
    ranks = defaultdict(lambda: defaultdict(dict))   # ranks[metric][col][exp]=rank
    for mk in METRICS:
        for col in ALL_COLS:
            scored = [(n, data[n][col][mk]) for n in present if col in data[n] and data[n][col].get(mk) is not None]
            scored.sort(key=lambda x: -x[1])
            for r, (n, _) in enumerate(scored, 1):
                ranks[mk][col][n] = r
    # Avg(4*) / RankAvg(4*) per metric
    summary = defaultdict(dict)
    for mk in METRICS:
        for n in present:
            vals = [data[n][c][mk] for c in STAR if c in data[n] and data[n][c].get(mk) is not None]
            rks  = [ranks[mk][c][n] for c in STAR if c in ranks[mk] and n in ranks[mk][c]]
            if len(vals) == 4:
                summary[mk][n] = {'avg': sum(vals)/4, 'rankavg': sum(rks)/len(rks) if rks else None}
        # medal = overall rank by rankavg asc
        order = sorted([n for n in summary[mk]], key=lambda n: summary[mk][n]['rankavg'])
        for pos, n in enumerate(order, 1):
            summary[mk][n]['medal'] = pos
    out = {'present': present, 'data': data, 'ranks': {mk:{c:ranks[mk][c] for c in ranks[mk]} for mk in METRICS},
           'summary': {mk: summary[mk] for mk in METRICS}}
    os.makedirs('/home/ykio/notebooks/TSMAE/temp', exist_ok=True)
    json.dump(out, open('/home/ykio/notebooks/TSMAE/temp/results_page_v2.json','w'))
    print(f'completed exps: {len(present)} -> {present[:5]}...{present[-5:]}')
    # VERIFY against known page values (271)
    print('\n=== VERIFY 271 vs page (expected: PSM pak_f1=0.8333 prc=0.8533 aff=0.7931 vus=0.7792) ===')
    for mk in METRICS:
        v = data[271]['psm'][mk]; print(f'  271 PSM {mk:18s} = {v:.4f}  rank={ranks[mk]["psm"][271]}')
    print('  271 PAK_F1: SWaT_full=%.4f excl22=%.4f WaDiA1=%.4f WaDiA2=%.4f PSM=%.4f SMD=%.4f SMAP=%.4f MSL=%.4f' % (
        data[271]['swat_full']['pak_auc_f1'], data[271]['swat_excl22']['pak_auc_f1'],
        data[271]['wadi_A1']['pak_auc_f1'], data[271]['wadi_A2']['pak_auc_f1'], data[271]['psm']['pak_auc_f1'],
        data[271]['smd']['pak_auc_f1'], data[271]['smap']['pak_auc_f1'], data[271]['msl']['pak_auc_f1']))
    print('  271 PAK_F1 Avg(4*)=%.4f RankAvg(4*)=%.2f medal=%d' % (
        summary['pak_auc_f1'][271]['avg'], summary['pak_auc_f1'][271]['rankavg'], summary['pak_auc_f1'][271]['medal']))
    # page-known 271 PAK_F1 row: SWaT_full 0.9444(5) excl22 0.6290(6) A1 0.8431(2) A2 0.8835(3) PSM 0.8333(2) Avg 0.7972 RankAvg 3.25
    print('  [page expects] excl22=0.6290(r6) A1=0.8431(r2) A2=0.8835(r3) PSM=0.8333(r2) Avg=0.7972 RankAvg=3.25')

if __name__ == '__main__':
    main()
