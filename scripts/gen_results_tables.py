"""Generate the 4 Notion result tables from temp/results_page_v2.json (aggregate_results_page_v2.py).
Format matches the existing page: Exp(+medal) | Name | SWaT(full) | SWaT(excl22)* | WaDi A1* | WaDi A2* |
PSM* | SMD | SMAP | MSL | Avg(4*) | RankAvg(4*). Cell = `val (rank, ep)`.
"""
import json
A = json.load(open('/home/ykio/notebooks/TSMAE/temp/results_page_v2.json'))
Q = json.load(open('/home/ykio/notebooks/TSMAE/configs/queue_dedup_renumbered_v6.json'))['experiments']
NAME = {e['exp_num']: e['name'].split('_', 1)[1] for e in Q}
# page names for pre-queue rows / overrides to match existing page exactly
NAME.setdefault(271, '271canon_baseline'); NAME.setdefault(274, '274canon_balsamp')

SUBSET = [271, 274] + list(range(285, 338))
METRICS = [('pak_auc_f1', 'PAK_AUC_F1'), ('pak_auc_prc_auc', 'PAK_AUC_PRC'),
           ('affiliation_f1', 'Affiliation_F1'), ('vus_pr', 'VUS_PR')]
COLS = ['swat_full', 'swat_excl22', 'wadi_A1', 'wadi_A2', 'psm', 'smd', 'smap', 'msl']
GROUPCOLS = {'smd', 'smap', 'msl'}
MEDAL = {1:'🥇',2:'🥈',3:'🥉',4:'4️⃣',5:'5️⃣',6:'6️⃣',7:'7️⃣',8:'8️⃣',9:'9️⃣',10:'🔟'}
data = {int(k): v for k, v in A['data'].items()}
ranks = A['ranks']; summary = A['summary']
present = set(A['present'])

def cell(n, mk, col):
    if n not in present or col not in data[n] or data[n][col].get(mk) is None:
        return '—'
    val = data[n][col][mk]
    rk = ranks[mk].get(col, {}).get(str(n)) or ranks[mk].get(col, {}).get(n)
    ep = data[n][col].get('best_ep')
    eptxt = (f'ep~{ep}' if col in GROUPCOLS else f'ep{ep}') if ep is not None else 'ep?'
    return f'{val:.4f} ({rk}, {eptxt})'

def table(mk, title):
    L = []
    L.append(f'## Section: {title} 기준 결과 (Exp 271·274·285–337)')
    L.append('<table header-row="true">')
    L.append('<tr>\n<td>Exp</td>\n<td>Name</td>\n<td>SWaT(full)</td>\n<td>SWaT(excl22)\\*</td>'
             '\n<td>WaDi A1\\*</td>\n<td>WaDi A2\\*</td>\n<td>PSM\\*</td>\n<td>SMD</td>\n<td>SMAP</td>'
             '\n<td>MSL</td>\n<td>Avg(4\\*)</td>\n<td>RankAvg(4\\*)</td>\n</tr>')
    for n in SUBSET:
        sm = summary[mk].get(str(n)) or summary[mk].get(n)
        medal = MEDAL.get(sm['medal']) if sm and sm.get('medal') in MEDAL else ''
        exp = f'{n} {medal}'.strip()
        nm = NAME.get(n, '?')
        cells = [cell(n, mk, c) for c in COLS]
        avg = f"{sm['avg']:.4f}" if sm else '—'
        ravg = f"{sm['rankavg']:.2f}" if sm and sm.get('rankavg') is not None else '—'
        row = [exp, f'`{nm}`' if False else nm] + cells + [avg, ravg]
        L.append('<tr>\n' + '\n'.join(f'<td>{c}</td>' for c in row) + '\n</tr>')
    L.append('</table>')
    return '\n'.join(L)

import sys
which = sys.argv[1] if len(sys.argv) > 1 else 'all'
if which == 'verify':
    # print 271-296 PAK_F1 rows only for eyeball
    print(table('pak_auc_f1', 'PAK_AUC_F1').split('</tr>\n<tr>\n<td>297')[0][:3000])
else:
    out = '\n'.join(table(mk, t) for mk, t in METRICS)
    open('/home/ykio/notebooks/TSMAE/temp/results_tables.md', 'w').write(out)
    print('wrote temp/results_tables.md  (%d chars)' % len(out))
    print('--- PAK_F1 first rows (271,274,285,286,287,288) ---')
    rows = table('pak_auc_f1','PAK_AUC_F1').split('\n</tr>\n<tr>')
    for r in rows[1:7]:
        cells=[c.replace('<td>','').replace('</td>','').strip() for c in r.split('\n') if '<td>' in c]
        print('  ', ' | '.join(cells[:7]), '|... Avg', cells[-2], 'RAvg', cells[-1])
