"""V2: smd_15 제거, cell W/L + RA(순위 평균) W/L 모두 계산."""
import json
import math
from pathlib import Path
from collections import defaultdict
from itertools import combinations

ROOT = Path('/home/ykio/notebooks/claude/results/experiments')
RESULTS = json.load(open('/home/ykio/notebooks/claude/temp/notion_exp_results.json'))

# Datasets (smd_15 제거, smd_full만)
DS_KEYS = ['swat_excl22', 'wadi_A1', 'wadi_A2', 'psm', 'smd_full']

TEST_KEYS = [
    'num_epochs', 'teacher_only_warmup_epochs',
    'num_encoder_layers', 'num_teacher_decoder_layers', 'num_student_decoder_layers',
    'num_shared_decoder_layers',
    'patchify_mode', 'normalize_mode', 'masking_ratio', 'patch_size', 'num_patches',
    'seq_length',
    'dynamic_margin_k', 'margin_type',
    'use_grl', 'grl_loss_weight', 'grl_target_mode', 'grl_balanced_sampling',
    'grl_use_focal', 'grl_cls_lr_ratio', 'grl_cls_arch', 'grl_cls_hidden',
    'grl_adaptive_lambda', 'grl_disable_anomaly_loss',
    'use_feature_matching', 'fm_distance_metric', 'fm_adaptive_lambda',
    'use_output_discrepancy', 'anomaly_score_mode',
    'force_mask_anomaly', 'normal_loss_weight', 'anomaly_loss_weight',
    'anomaly_loss_direction',
    'freeze_teacher_after_warmup', 'freeze_encoder_only',
    'patch_level_loss', 'epoch_offset',
]


def load_configs():
    configs = {}
    for d in sorted(ROOT.iterdir()):
        if not d.is_dir():
            continue
        cfg_p = d / 'simulation/simulation/best_config.json'
        if not cfg_p.exists():
            continue
        try:
            configs[int(d.name.split('_')[0])] = json.loads(cfg_p.read_text())
        except Exception:
            continue
    return configs


def get_pak_f1(exp_id, ds_key):
    sub = RESULTS.get(str(exp_id), {}).get(ds_key)
    if not isinstance(sub, dict):
        return None
    return sub.get('pak_f1')


def compute_global_ranks(configs):
    """For each dataset, rank ALL experiments by pak_f1 (1 = highest)."""
    ranks = {}
    for ds in DS_KEYS:
        scored = [(eid, get_pak_f1(eid, ds)) for eid in configs]
        scored = [(e, s) for e, s in scored if s is not None]
        scored.sort(key=lambda x: -x[1])  # descending
        r = {}
        for i, (e, _) in enumerate(scored):
            r[e] = i + 1  # rank 1 = best
        ranks[ds] = r
    return ranks


def get_rank_avg(exp_id, ranks):
    rs = [ranks[ds][exp_id] for ds in DS_KEYS if exp_id in ranks[ds]]
    if not rs:
        return None
    return sum(rs) / len(rs)


def serialize(v):
    return tuple(v) if isinstance(v, list) else v


def analyze_param(param, configs, ranks):
    other_keys = sorted(set().union(*[c.keys() for c in configs.values()]) - {param})
    groups = defaultdict(list)
    for eid, cfg in configs.items():
        key = tuple((k, serialize(cfg.get(k))) for k in other_keys)
        val = serialize(cfg.get(param))
        groups[key].append((eid, val))

    # Cell-level wins & RA-level wins per value
    cell_wins = defaultdict(int)
    cell_losses = defaultdict(int)
    ra_wins = defaultdict(int)
    ra_losses = defaultdict(int)
    pair_count = 0
    distinct = set()

    for key, members in groups.items():
        vals = set(v for _, v in members)
        if len(vals) < 2:
            continue
        by_val = defaultdict(list)
        for eid, v in members:
            by_val[v].append(eid)
        values = sorted(by_val.keys(), key=str)
        for va, vb in combinations(values, 2):
            for ea in by_val[va]:
                for eb in by_val[vb]:
                    pair_count += 1
                    distinct.add(va); distinct.add(vb)
                    # Cell-level
                    for ds in DS_KEYS:
                        sa = get_pak_f1(ea, ds)
                        sb = get_pak_f1(eb, ds)
                        if sa is None or sb is None:
                            continue
                        if sa > sb:
                            cell_wins[va] += 1
                            cell_losses[vb] += 1
                        elif sb > sa:
                            cell_wins[vb] += 1
                            cell_losses[va] += 1
                    # RA-level (lower = better)
                    ra_a = get_rank_avg(ea, ranks)
                    ra_b = get_rank_avg(eb, ranks)
                    if ra_a is not None and ra_b is not None:
                        if ra_a < ra_b:
                            ra_wins[va] += 1
                            ra_losses[vb] += 1
                        elif ra_b < ra_a:
                            ra_wins[vb] += 1
                            ra_losses[va] += 1

    return {
        'pair_count': pair_count,
        'distinct_values': list(distinct),
        'cell_wins': dict(cell_wins),
        'cell_losses': dict(cell_losses),
        'ra_wins': dict(ra_wins),
        'ra_losses': dict(ra_losses),
    }


def zscore(wins, losses):
    n = wins + losses
    if n == 0:
        return 0.0
    return (wins - n/2) / math.sqrt(n/4)


def main():
    configs = load_configs()
    print(f'Loaded {len(configs)} experiments')
    ranks = compute_global_ranks(configs)
    print(f'Ranks computed for datasets: {DS_KEYS}')
    for ds in DS_KEYS:
        print(f'  {ds}: {len(ranks[ds])} ranked exp')

    rows = []
    for p in TEST_KEYS:
        r = analyze_param(p, configs, ranks)
        if r['pair_count'] == 0:
            continue
        # Find best value by cell wins
        vals = set(list(r['cell_wins'].keys()) + list(r['cell_losses'].keys()))
        # Cell-level summary: for each value, (wins, losses)
        cell_summary = {v: (r['cell_wins'].get(v, 0), r['cell_losses'].get(v, 0)) for v in vals}
        ra_summary = {v: (r['ra_wins'].get(v, 0), r['ra_losses'].get(v, 0)) for v in vals}
        # Best cell winner
        if not cell_summary:
            continue
        best_cell = max(cell_summary.items(), key=lambda x: (x[1][0] - x[1][1], x[1][0]))
        best_ra = max(ra_summary.items(), key=lambda x: (x[1][0] - x[1][1], x[1][0])) if ra_summary else best_cell

        cell_z = zscore(best_cell[1][0], best_cell[1][1])
        ra_z = zscore(best_ra[1][0], best_ra[1][1])
        rows.append({
            'param': p,
            'pair_count': r['pair_count'],
            'cell_best_value': best_cell[0],
            'cell_W': best_cell[1][0],
            'cell_L': best_cell[1][1],
            'cell_z': cell_z,
            'ra_best_value': best_ra[0],
            'ra_W': best_ra[1][0],
            'ra_L': best_ra[1][1],
            'ra_z': ra_z,
            'cell_summary': cell_summary,
            'ra_summary': ra_summary,
        })

    # Sort by max(|cell_z|, |ra_z|)
    rows.sort(key=lambda r: max(abs(r['cell_z']), abs(r['ra_z'])), reverse=True)

    # Print table
    print(f'\n{"Param":<32s} {"#pair":<5s} | {"Cell Best":<12s} {"W/L":<10s} {"z":<6s} | {"RA Best":<12s} {"W/L":<10s} {"z":<6s}')
    print('-' * 110)
    for r in rows:
        cell_v = str(r['cell_best_value'])[:12]
        ra_v = str(r['ra_best_value'])[:12]
        cell_wl = f'{r["cell_W"]}/{r["cell_L"]}'
        ra_wl = f'{r["ra_W"]}/{r["ra_L"]}'
        print(f'{r["param"]:<32s} {r["pair_count"]:<5d} | {cell_v:<12s} {cell_wl:<10s} {r["cell_z"]:<6.2f} | {ra_v:<12s} {ra_wl:<10s} {r["ra_z"]:<6.2f}')

    Path('/home/ykio/notebooks/claude/temp/param_wins_v2.json').write_text(
        json.dumps(rows, indent=2, default=str)
    )
    print(f'\nSaved to /home/ykio/notebooks/claude/temp/param_wins_v2.json')


if __name__ == '__main__':
    main()
