"""Build Section 3 (5-DS) and Section 4 (6-DS = +Exathlon) leaderboard tables.

Section 3 columns: Exp, Name, excl22*, A1*, A2*, SMD(full), SMD(15)*, PSM*, Avg(5*), RA(5*)
Section 4 columns: Exp, Name, excl22*, A1*, A2*, SMD(full), SMD(15)*, PSM*, Exathlon*, Avg(6*), RA(6*)

* = used for Avg/RA computation. SMD(full) is display only.

RA universe = 60 model + TBD placeholders. TBD rows show — for all cells.
Cell format:
  - Per-dataset (single best_epoch): "0.587 (160)"
  - SMD aggregate / Exathlon aggregate: "0.730" (no best_ep)
  - Avg/RA: "0.722" / "25.50"
"""
import json
from pathlib import Path

DATA = json.load(open('/home/ykio/notebooks/claude/temp/notion_exp_results.json'))

# 60-exp ordered list (from current Section 4)
EXP_LIST = [
    '140', '150', '153', '155', '157', '159', '160', '161', '165', '166',
    '169', '172', '173', '179', '184', '187', '190', '191', '198', '203',
    '208', '209', '211', '212', '214', '217', '221', '222', '223', '224',
    '226', '228', '229', '230', '231', '234', '236', '245', '247', '248',
    '249', '254', '256', '264', '265', '266', '269', '270', '271', '272',
    '273', '274', '275', '276', '277', '278', '279', '282', '283', '284',
]
TBD_EXPS = [str(i) for i in range(285, 303)]  # 285-302 (Group P, 271 base)

# Display names (from current Section 4 text)
NAMES = {
    '140': '140 ep200', '150': 'sd1+offset', '153': 'ep200+td4', '155': 'ep300+td4',
    '157': 'ep300+sd1', '159': 'ep300', '160': 'sd1+offset_v2', '161': 'sd1+offset_v3',
    '165': 'fm_adaptive', '166': '247+wider_cls', '169': '247+focal', '172': '170+win',
    '173': 'GRL_w0.2', '179': 'GRL_lr0.5', '184': 'sd2+GRL slow_cls', '187': 'slow_cls+window',
    '190': 'slow+win+anomloss', '191': '190+focal', '198': '187+fm_adaptive', '203': '198+ep500',
    '208': '187+fm_l2', '209': '187+fm_l2+fm_adp', '211': '208+ep500', '212': '209+ep300',
    '214': 'sd1+L2+fm_adp', '217': 'sd2+fm_l2+fm_adp', '221': 'ep300+td4+fm_adp',
    '222': 'ep300+td4+fm_l2', '223': 'ep300+td4+L2_adp', '224': 'ep300+td4+sd2',
    '226': 'sd2+td4_no_fm', '228': 'sd2+td4+fm_cos', '229': 'sd2+td4+fm_l2',
    '230': 'sd2+td4+fm_l2+fm_adp', '231': 'sd2+td4+fm_l2+focal', '234': 'sd2+td3+fm_l2_adp',
    '236': 'sd2+td3+fm_l2+focal', '245': 'GRL_nofocal+bal+slow', '247': 'GRL_ep300+bal+slow',
    '248': 'GRL_w005+bal+slow', '249': 'GRL_w05+bal+slow', '254': '247_w0.5', '256': '247_freeze',
    '264': '247_ep400', '265': '247+ep500', '266': '247+wider_cls_v2', '269': '190_ep500',
    '270': '208_ep500', '271': '212_ep500', '272': '247+slow_cls', '273': '265_no_balanced',
    '274': '265_fm_l2', '275': '245_ep500', '276': '254_ep500', '277': '265_wider_cls',
    '278': '274_no_balanced_ep300', '279': '274_fm_adaptive_ep300', '282': '274_normal_w3_ep300',
    '283': '274_no_focal_ep300', '284': '274_ep300_anchor',
    # Group P (271 base, 285-302) — sequential renumbering after collapses removed
    '285': '271_unmask',
    '286': '271_no_focal',
    '287': '271_no_focal_wider_cls',
    '288': '271_no_focal_grl_w05',
    '289': '271_w300_p10', '290': '271_w200_p10', '291': '271_w100_p10',
    '292': '271_w500_p5', '293': '271_w200_p5', '294': '271_w100_p5',
    '295': '271_target_patch', '296': '271_grl_w1',
    '297': '271_td4', '298': '271_sd1',
    '299': '271_cls_lr1', '300': '271_anomloss_on',
    '301': '271_cls_hidden128', '302': '271_td4_sd1',
}


def fmt_cell_with_ep(exp_id: str, ds_key: str, metric: str):
    """Format cell as 'score (best_ep)' for single-dataset metrics."""
    rec = DATA.get(exp_id)
    if not rec:
        return '—'
    sub = rec.get(ds_key)
    if not isinstance(sub, dict):
        return '—'
    score = sub.get(metric)
    if score is None:
        return '—'
    ep = sub.get('best_ep')
    if ep is not None:
        return f'{score:.3f} ({int(ep)})'
    return f'{score:.3f}'


def fmt_cell_agg(exp_id: str, ds_key: str, metric: str):
    """Format aggregate cell as '0.730 (mean_best_ep)'."""
    rec = DATA.get(exp_id)
    if not rec:
        return '—'
    sub = rec.get(ds_key)
    if not isinstance(sub, dict):
        return '—'
    v = sub.get(metric)
    if v is None:
        return '—'
    ep = sub.get('best_ep')  # already computed as mean for smd_full/smd_15/exathlon
    if ep is not None:
        return f'{v:.3f} ({int(round(ep))})'
    return f'{v:.3f}'


def get_score(exp_id: str, ds_key: str, metric: str):
    rec = DATA.get(exp_id)
    if not rec:
        return None
    sub = rec.get(ds_key)
    if not isinstance(sub, dict):
        return None
    return sub.get(metric)


def compute_ranks_within_universe(metric: str, ds_keys: list, universe: list):
    """For each DS in ds_keys, rank experiments within universe (1=best)."""
    ranks = {}  # {ds: {exp_id: rank}}
    for ds in ds_keys:
        scored = []
        for e in universe:
            s = get_score(e, ds, metric)
            if s is not None:
                scored.append((e, s))
        scored.sort(key=lambda x: -x[1])
        r = {e: i+1 for i, (e, _) in enumerate(scored)}
        ranks[ds] = r
    return ranks


MEDALS = {1: '🥇', 2: '🥈', 3: '🥉',
          4: '4️⃣', 5: '5️⃣', 6: '6️⃣', 7: '7️⃣', 8: '8️⃣', 9: '9️⃣', 10: '🔟'}


def compute_table_ranks(metric: str, star_ds: list, exp_list: list, universe: list):
    """Compute RA for each exp in exp_list using ranks within universe.
    Returns sorted list [(exp, ra), ...] ascending by RA (best first).
    Only exps with full data on all star_ds get a rank.
    """
    ranks = compute_ranks_within_universe(metric, star_ds, universe)
    ra_list = []
    for e in exp_list:
        rks = [ranks[ds][e] for ds in star_ds if e in ranks[ds]]
        if len(rks) == len(star_ds):
            ra_list.append((e, sum(rks)/len(rks)))
    ra_list.sort(key=lambda x: x[1])
    return ra_list, ranks


def fmt_exp_label(exp_id: str, rank_position: dict) -> str:
    """Add medal/emoji if exp is in top 10 by RA."""
    pos = rank_position.get(exp_id)
    if pos is None:
        return exp_id
    medal = MEDALS.get(pos)
    if medal is None:
        return exp_id
    if pos <= 3:
        return f'**{exp_id}** {medal}'
    return f'{exp_id} {medal}'


def render_section3(metric: str, title: str, callout: str) -> str:
    """5-DS leaderboard. Avg/RA over excl22, A1, A2, smd_15, psm."""
    star_ds = ['swat_excl22', 'wadi_A1', 'wadi_A2', 'smd_15', 'psm']
    universe = EXP_LIST + TBD_EXPS
    ra_sorted, ranks = compute_table_ranks(metric, star_ds, EXP_LIST, universe)
    # Top 10 mapping: exp -> rank (1-10)
    rank_position = {e: i+1 for i, (e, _) in enumerate(ra_sorted[:10])}

    out = [f'## {title}',
           '<callout>',
           f'\t{callout}',
           '</callout>',
           '<table>',
           '<tr>',
           '<td>Exp</td><td>Name</td>',
           '<td>excl22*</td><td>A1*</td><td>A2*</td>',
           '<td>SMD(full)</td><td>SMD(15)*</td><td>PSM*</td>',
           '<td>Avg(5*)</td><td>RA(5*)</td>',
           '</tr>',
           ]
    # convert flat header to multi-line
    out = [f'## {title}', '<callout>', f'\t{callout}', '</callout>', '<table>']
    out.append('<tr>')
    for h in ['Exp', 'Name', 'excl22*', 'A1*', 'A2*', 'SMD(full)', 'SMD(15)*', 'PSM*', 'Avg(5*)', 'RA(5*)']:
        out.append(f'<td>{h}</td>')
    out.append('</tr>')

    for e in EXP_LIST:
        out.append('<tr>')
        out.append(f'<td>{fmt_exp_label(e, rank_position)}</td>')
        out.append(f'<td>{NAMES.get(e, e)}</td>')
        out.append(f'<td>{fmt_cell_with_ep(e, "swat_excl22", metric)}</td>')
        out.append(f'<td>{fmt_cell_with_ep(e, "wadi_A1", metric)}</td>')
        out.append(f'<td>{fmt_cell_with_ep(e, "wadi_A2", metric)}</td>')
        out.append(f'<td>{fmt_cell_agg(e, "smd_full", metric)}</td>')
        out.append(f'<td>{fmt_cell_agg(e, "smd_15", metric)}</td>')
        out.append(f'<td>{fmt_cell_with_ep(e, "psm", metric)}</td>')
        vals = []
        rks = []
        for ds in star_ds:
            s = get_score(e, ds, metric)
            if s is not None:
                vals.append(s)
                if e in ranks[ds]:
                    rks.append(ranks[ds][e])
        avg_s = f'{sum(vals)/len(vals):.3f}' if vals else '—'
        ra_s = f'{sum(rks)/len(rks):.2f}' if len(rks) == len(star_ds) else '—'
        out.append(f'<td>{avg_s}</td>')
        out.append(f'<td>{ra_s}</td>')
        out.append('</tr>')

    # TBD placeholders
    for e in TBD_EXPS:
        out.append('<tr>')
        out.append(f'<td>{e}</td>')
        out.append(f'<td>{NAMES.get(e, e)}</td>')
        for _ in range(8):
            out.append('<td>—</td>')
        out.append('</tr>')

    out.append('</table>')
    return '\n'.join(out)


def render_section4(metric: str, title: str, callout: str) -> str:
    """6-DS leaderboard = Section 3 + Exathlon."""
    star_ds = ['swat_excl22', 'wadi_A1', 'wadi_A2', 'smd_15', 'psm', 'exathlon']
    universe = EXP_LIST + TBD_EXPS
    ra_sorted, ranks = compute_table_ranks(metric, star_ds, EXP_LIST, universe)
    rank_position = {e: i+1 for i, (e, _) in enumerate(ra_sorted[:10])}

    out = [f'## {title}', '<callout>', f'\t{callout}', '</callout>', '<table>']
    out.append('<tr>')
    for h in ['Exp', 'Name', 'excl22*', 'A1*', 'A2*', 'SMD(full)', 'SMD(15)*', 'PSM*', 'Exathlon*', 'Avg(6*)', 'RA(6*)']:
        out.append(f'<td>{h}</td>')
    out.append('</tr>')

    for e in EXP_LIST:
        out.append('<tr>')
        out.append(f'<td>{fmt_exp_label(e, rank_position)}</td>')
        out.append(f'<td>{NAMES.get(e, e)}</td>')
        out.append(f'<td>{fmt_cell_with_ep(e, "swat_excl22", metric)}</td>')
        out.append(f'<td>{fmt_cell_with_ep(e, "wadi_A1", metric)}</td>')
        out.append(f'<td>{fmt_cell_with_ep(e, "wadi_A2", metric)}</td>')
        out.append(f'<td>{fmt_cell_agg(e, "smd_full", metric)}</td>')
        out.append(f'<td>{fmt_cell_agg(e, "smd_15", metric)}</td>')
        out.append(f'<td>{fmt_cell_with_ep(e, "psm", metric)}</td>')
        out.append(f'<td>{fmt_cell_agg(e, "exathlon", metric)}</td>')
        vals, rks = [], []
        for ds in star_ds:
            s = get_score(e, ds, metric)
            if s is not None:
                vals.append(s)
                if e in ranks[ds]:
                    rks.append(ranks[ds][e])
        avg_s = f'{sum(vals)/len(vals):.3f}' if vals else '—'
        ra_s = f'{sum(rks)/len(rks):.2f}' if len(rks) == len(star_ds) else '—'
        out.append(f'<td>{avg_s}</td>')
        out.append(f'<td>{ra_s}</td>')
        out.append('</tr>')

    for e in TBD_EXPS:
        out.append('<tr>')
        out.append(f'<td>{e}</td>')
        out.append(f'<td>{NAMES.get(e, e)}</td>')
        for _ in range(9):
            out.append('<td>—</td>')
        out.append('</tr>')

    out.append('</table>')
    return '\n'.join(out)


def main():
    callout_s3_intro = (
        '**Section 3: 60-model 5-DS Leaderboard (PSM + SMD(15) 포함)**\n'
        '\t**컬럼 정의**: 별표(\\*) = Avg/RA 계산 포함 (Swat excl22, Wadi A1/A2, SMD(15), PSM). SMD(full)은 표시만.\n'
        '\t셀 표시: `pak_score (best_ep)` for per-DS, aggregate (SMD/Avg/RA)는 숫자만.\n'
        '\tRA(5\\*) = 60-model + TBD placeholders 내부 등수 평균 (낮을수록 좋음). best_ep는 PAK_AUC_F1 기준.'
    )
    callout_s4_intro = (
        '**Section 4: 60-model 6-DS Leaderboard (Exathlon 포함)**\n'
        '\tSection 3와 동일 모델 set + Exathlon column (6 apps: app1/2/4/5/6/9 평균, 각 app 독립 best_epoch).\n'
        '\tAvg(6\\*) / RA(6\\*) = excl22 + A1 + A2 + SMD(15) + PSM + Exathlon.\n'
        '\tSection 3 모든 컬럼 유지 + Exathlon\\* 추가.'
    )

    s31 = render_section3('pak_f1', '3.1 PAK_AUC_F1 (60-model 5-DS)', callout_s3_intro)
    s32 = render_section3('pak_prc', '3.2 PAK_AUC_PRC (60-model 5-DS)', callout_s3_intro)
    s41 = render_section4('pak_f1', '4.1 PAK_AUC_F1 (60-model 6-DS, +Exathlon)', callout_s4_intro)
    s42 = render_section4('pak_prc', '4.2 PAK_AUC_PRC (60-model 6-DS, +Exathlon)', callout_s4_intro)

    out_dir = Path('/home/ykio/notebooks/claude/temp')
    (out_dir / 'sec3_pak_f1.txt').write_text(s31)
    (out_dir / 'sec3_pak_prc.txt').write_text(s32)
    (out_dir / 'sec4_pak_f1.txt').write_text(s41)
    (out_dir / 'sec4_pak_prc.txt').write_text(s42)
    print(f'Wrote sec3_pak_f1/prc + sec4_pak_f1/prc to {out_dir}')
    print(f'Sizes: 3.1={len(s31)}, 3.2={len(s32)}, 4.1={len(s41)}, 4.2={len(s42)}')


if __name__ == '__main__':
    main()
