"""For each config parameter, find experiment pairs that differ ONLY in that parameter,
and tally per-dataset wins. Sort parameters by significance.

Datasets used for win-counting: swat_excl22, wadi_A1, wadi_A2, psm, smd_full, smd_15.
(simulation and swat_full excluded per user instruction)
"""
import json
from pathlib import Path
from collections import defaultdict
from itertools import combinations

ROOT = Path('/home/ykio/notebooks/claude/results/experiments')
RESULTS = json.load(open('/home/ykio/notebooks/claude/temp/notion_exp_results.json'))

# Datasets used for win counting (excl simulation + swat_full)
DS_KEYS = ['swat_excl22', 'wadi_A1', 'wadi_A2', 'psm', 'smd_full', 'smd_15']

# All config keys to test
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
    'patch_level_loss', 'epoch_offset', 'random_seed',
]


def load_all_configs():
    configs = {}
    for d in sorted(ROOT.iterdir()):
        if not d.is_dir():
            continue
        cfg_p = d / 'simulation/simulation/best_config.json'
        if not cfg_p.exists():
            continue
        try:
            cfg = json.loads(cfg_p.read_text())
        except Exception:
            continue
        exp_id = int(d.name.split('_')[0])
        configs[exp_id] = cfg
    return configs


def get_score(exp_id: int, ds_key: str, metric: str = 'pak_f1'):
    """Return None if not available."""
    rec = RESULTS.get(str(exp_id))
    if not rec:
        return None
    sub = rec.get(ds_key)
    if not isinstance(sub, dict):
        return None
    return sub.get(metric)


def serialize(v):
    if isinstance(v, list):
        return tuple(v)
    return v


def analyze_param(param: str, configs: dict):
    """Find all pairs that differ ONLY in `param`. Return per-value win counts."""
    # Group by all-other-keys
    other_keys = sorted(set().union(*[c.keys() for c in configs.values()]) - {param})

    groups = defaultdict(list)
    for exp_id, cfg in configs.items():
        key = tuple((k, serialize(cfg.get(k))) for k in other_keys)
        val = serialize(cfg.get(param))
        groups[key].append((exp_id, val))

    # Find groups with at least 2 distinct param values
    pairs = []
    for key, members in groups.items():
        vals = set(v for _, v in members)
        if len(vals) >= 2:
            # All distinct-value pairs within this group
            by_val = defaultdict(list)
            for eid, v in members:
                by_val[v].append(eid)
            pairs.append((dict(by_val), members))

    # Tally wins per value across pairs / datasets
    # For each pair (val_a, val_b), compare per dataset
    value_wins = defaultdict(lambda: defaultdict(int))  # value_wins[winner_val][ds]
    value_losses = defaultdict(lambda: defaultdict(int))
    value_ties = defaultdict(lambda: defaultdict(int))
    pair_count = 0
    distinct_values = set()

    for by_val, _members in pairs:
        values = sorted(by_val.keys(), key=str)
        for va, vb in combinations(values, 2):
            for ea in by_val[va]:
                for eb in by_val[vb]:
                    pair_count += 1
                    for ds in DS_KEYS:
                        sa = get_score(ea, ds)
                        sb = get_score(eb, ds)
                        if sa is None or sb is None:
                            continue
                        distinct_values.add(va)
                        distinct_values.add(vb)
                        if sa > sb:
                            value_wins[va][ds] += 1
                            value_losses[vb][ds] += 1
                        elif sb > sa:
                            value_wins[vb][ds] += 1
                            value_losses[va][ds] += 1
                        else:
                            value_ties[va][ds] += 1
                            value_ties[vb][ds] += 1

    return {
        'param': param,
        'pair_count': pair_count,
        'distinct_values': list(distinct_values),
        'value_wins': {str(v): dict(d) for v, d in value_wins.items()},
        'value_losses': {str(v): dict(d) for v, d in value_losses.items()},
        'value_ties': {str(v): dict(d) for v, d in value_ties.items()},
    }


def main():
    configs = load_all_configs()
    print(f'Loaded {len(configs)} experiments')

    all_results = {}
    for p in TEST_KEYS:
        r = analyze_param(p, configs)
        if r['pair_count'] > 0:
            all_results[p] = r

    # Sort by significance: total decisive comparisons (wins + losses), with ties broken by max-margin
    def significance(r):
        total_decisive = sum(sum(d.values()) for d in r['value_wins'].values())
        # max win - loss margin across values
        max_margin = 0
        for v in r['value_wins']:
            wins_v = sum(r['value_wins'].get(v, {}).values())
            losses_v = sum(r['value_losses'].get(v, {}).values())
            max_margin = max(max_margin, abs(wins_v - losses_v))
        return (total_decisive, max_margin)

    sorted_results = sorted(all_results.items(), key=lambda x: significance(x[1]), reverse=True)

    # Output
    print()
    print('=== Per-parameter isolated win/loss ===')
    print(f'Datasets used: {DS_KEYS}\n')
    for param, r in sorted_results:
        if r['pair_count'] == 0:
            continue
        # Compute aggregate wins per value
        agg = {}
        for v in set(list(r['value_wins'].keys()) + list(r['value_losses'].keys())):
            wins = sum(r['value_wins'].get(v, {}).values())
            losses = sum(r['value_losses'].get(v, {}).values())
            ties = sum(r['value_ties'].get(v, {}).values())
            agg[v] = (wins, losses, ties)
        print(f'--- {param} ({r["pair_count"]} cross-pairs) ---')
        for v, (w, l, t) in sorted(agg.items(), key=lambda x: -x[1][0]):
            print(f'  value={v:>15s}: {w:>3d} wins / {l:>3d} losses / {t:>3d} ties')
        print()

    # Save full results
    Path('/home/ykio/notebooks/claude/temp/param_isolated_wins.json').write_text(json.dumps(all_results, indent=2, default=str))
    print('Saved to /home/ykio/notebooks/claude/temp/param_isolated_wins.json')


if __name__ == '__main__':
    main()
