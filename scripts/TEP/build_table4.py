"""build_table4.py — legacy PAK diagnostic aggregator for TEP type-disjoint runs.

The current paper (LASAD v22) reports TEP as Table 3 with VUS-PR.  This script
only aggregates pak_auc_f1 into historically named table4_data*.json files;
those files are diagnostic and must not be used as the paper Table 3 source.
MAE conditions (A/B/B0/D) = train mean-fixed pak from pak_fill.json.
Simple baselines (Random/PCA/NN/Sensor/L2) = #12 per_fault_metrics pak_auc_f1, fold-matched,
cross-checked against partition_eval. Discriminants: ΔU(A−B)+Γ̂_A(vs B), ΔU(A−PCA)+Γ̂(vs PCA).
Output: table4_data.json + table4_values.txt (legacy filenames retained for compatibility).

Multi-seed (2026-07-22): --seed N (default 42).
  seed 42  → unchanged legacy behavior (byte-compatible table4_data.json / table4_values.txt).
  seed N   → MAE values from results/experiments/TEP_phase2_win100_ep30_s{N}/pak_fill.json
             (A/B/D only — B0 not run for extra seeds), Random row from the simple dir's
             per_fault_by_seed.json seed-N raw values (empty if not yet re-run), the 4
             deterministic baselines reuse the seed-42 values (seed-independent — footnoted
             in meta), output table4_data_s{N}.json + table4_values_s{N}.txt in the BASE
             root (results_baseline.md generator convention).

Data-seed axis (2026-07-24): --dataseed N (dataset allocation itself re-drawn with seed N;
  streams from scripts/TEP/data_dataseed{N}). MAE (B/A/D) from
  results/experiments/TEP_phase2_win100_ep30_dataseed{N}/pak_fill.json; simple 5 rows from
  scripts/TEP/results/simple_dataseed{N}/ (ALL data-dependent here — no seed-42 reuse;
  missing dirs → null cells until run_tep_simple_dataseed runs); Random row = that dir's
  per_fault_by_seed.json seed-N raw values. Output table4_data_ds{N}.json +
  table4_values_ds{N}.txt in the BASE root. NOTE: the dataseed test stream is layout-
  identical to canonical (same sizes/order rules), so y/run_table from load_test() apply.
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, 'scripts/TEP'); sys.path.insert(0, '.')
from tep_common import seen_faults, unseen_faults, USABLE_FAULTS
from run_tep_simple import partition_eval, load_test

BASE_ROOT = 'results/experiments/TEP_phase2_win100_ep30'
SIMPLE = 'scripts/TEP/results/12_20260610_211815_tep_typegen_simple'
FOLDS = ['f_step', 'f_rand', 'f_ds', 'f_unk']
FOLDLBL = {'f_step': 'F-STEP', 'f_rand': 'F-RAND', 'f_ds': 'F-DS', 'f_unk': 'F-UNK'}
BASELINES = [('Random', 'random'), ('PCA recon.', 'pca_error'), ('NN-distance', 'nn_distance'),
             ('Sensor range', 'sensor_range'), ('L2-norm', 'l2_norm')]

ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, default=42,
                help='42 (default) = legacy table4_data.json; N = table4_data_s{N}.json')
ap.add_argument('--dataseed', type=int, default=None,
                help='data-seed axis: table4_data_ds{N}.json (dataset-allocation seed)')
args = ap.parse_args()
SEED = args.seed
DATASEED = args.dataseed
if DATASEED is not None and SEED != 42:
    sys.exit('--dataseed and --seed N are mutually exclusive (one axis at a time)')
if DATASEED == 42:
    sys.exit('data-seed 42 = canonical dataset — represented by the legacy table4_data.json')
if DATASEED is not None:
    MAE_ROOT = f'{BASE_ROOT}_dataseed{DATASEED}'
    MAE_CONDS = ['B', 'A', 'D']          # B0 not run for the data-seed axis
    SFX = f'_ds{DATASEED}'
    SIMPLE_SRC = f'scripts/TEP/results/simple_dataseed{DATASEED}'
else:
    MAE_ROOT = BASE_ROOT if SEED == 42 else f'{BASE_ROOT}_s{SEED}'
    MAE_CONDS = ['B0', 'B', 'A', 'D'] if SEED == 42 else ['B', 'A', 'D']
    SFX = '' if SEED == 42 else f'_s{SEED}'
    SIMPLE_SRC = SIMPLE

# ---- MAE per-mode S/U from pak_fill.json (tmf) ----
pf = json.load(open(f'{MAE_ROOT}/pak_fill.json'))
FK = {'f_step': 'fstep', 'f_rand': 'frand', 'f_ds': 'fds', 'f_unk': 'funk'}
def mae_SU(cond, fk):
    """cond in {A,B,B0,D}; fk in FOLDS."""
    key = ('B0_tmf' if cond == 'B0' else f'{cond}_tmf_{FK[fk]}' if cond in ('A', 'B')
           else f'D_recon_{FK[fk]}')
    S = np.mean([pf[key][str(f)] for f in seen_faults(fk)])
    U = np.mean([pf[key][str(f)] for f in unseen_faults(fk)])
    return float(S), float(U)

# ---- simple baselines: load per_fault_metrics, cross-check one via partition_eval ----
# (data-seed axis: the dataseed test stream is layout-identical to canonical — same
#  y/run_table — so partition_eval over load_test() stays valid for its scores.)
_chk_scores = f'{SIMPLE_SRC}/f_step/pca_error/scores.npz'
if os.path.exists(_chk_scores):
    print('=== cross-check simple baseline pak (partition_eval vs per_fault_metrics) ===', flush=True)
    _, y, _, _, run_table = load_test()
    z = np.load(_chk_scores)['anomaly_score'].astype(np.float64)
    chk = partition_eval(z, y, run_table, {1}, lite=True)[0].get('pak_auc_f1', 0.0)
    ref = json.load(open(f'{SIMPLE_SRC}/f_step/pca_error/per_fault_metrics.json'))['1']['pak_auc_f1']
    print(f'  pca/f_step/IDV1: partition_eval={chk:.4f} per_fault_metrics={ref:.4f} diff={abs(chk-ref):.4f}')
    assert abs(chk - ref) < 0.02, 'MISMATCH — per_fault_metrics not from same protocol!'
    print('  OK → using per_fault_metrics.json\n', flush=True)
else:
    assert DATASEED is not None, f'missing simple results: {_chk_scores}'
    print(f'=== cross-check skipped — {SIMPLE_SRC} not yet run (null simple cells) ===', flush=True)

def simple_SU(model_dir, fk):
    p = f'{SIMPLE_SRC}/{fk}/{model_dir}/per_fault_metrics.json'
    if DATASEED is not None and not os.path.exists(p):
        return None, None, {}            # data-seed axis: simple not yet run → null
    m = json.load(open(p))
    S = np.mean([m[str(f)]['pak_auc_f1'] for f in seen_faults(fk)])
    U = np.mean([m[str(f)]['pak_auc_f1'] for f in unseen_faults(fk)])
    return float(S), float(U), {str(f): m[str(f)]['pak_auc_f1'] for f in USABLE_FAULTS}

def random_SU_seed(fk, seed):
    """seed-N raw values from per_fault_by_seed.json ({seed:{fault:{metric:v}}}).
    Missing file/seed → empty (None,None,{}) — allowed until the random baseline
    is re-run with the seeded protocol."""
    p = f'{SIMPLE_SRC}/{fk}/random/per_fault_by_seed.json'
    if not os.path.exists(p):
        return None, None, {}
    m = json.load(open(p)).get(str(seed))
    if not m:
        return None, None, {}
    S = np.mean([m[str(f)]['pak_auc_f1'] for f in seen_faults(fk)])
    U = np.mean([m[str(f)]['pak_auc_f1'] for f in unseen_faults(fk)])
    return float(S), float(U), {str(f): m[str(f)]['pak_auc_f1'] for f in USABLE_FAULTS}

# ---- assemble ----
data = {'folds': FOLDS, 'simple': {}, 'mae': {}, 'discriminant': {}}
for lbl, mdir in BASELINES:
    if DATASEED is not None and lbl == 'Random':
        data['simple'][lbl] = {fk: dict(zip(['S', 'U', 'paks'], random_SU_seed(fk, DATASEED)))
                               for fk in FOLDS}
    elif DATASEED is None and SEED != 42 and lbl == 'Random':
        data['simple'][lbl] = {fk: dict(zip(['S', 'U', 'paks'], random_SU_seed(fk, SEED)))
                               for fk in FOLDS}
    else:
        data['simple'][lbl] = {fk: dict(zip(['S', 'U', 'paks'], simple_SU(mdir, fk))) for fk in FOLDS}
for cond in MAE_CONDS:
    data['mae'][cond] = {fk: dict(zip(['S', 'U'], mae_SU(cond, fk))) for fk in FOLDS}

# discriminants per fold (PCA terms None-safe: data-seed axis may lack simple results)
for fk in FOLDS:
    SA, UA = data['mae']['A'][fk]['S'], data['mae']['A'][fk]['U']
    SB, UB = data['mae']['B'][fk]['S'], data['mae']['B'][fk]['U']
    SP, UP = data['simple']['PCA recon.'][fk]['S'], data['simple']['PCA recon.'][fk]['U']
    data['discriminant'][fk] = {
        'dU_AB': UA - UB, 'ghat_A': (SA - UA) - (SB - UB),
        'dU_APCA': (UA - UP) if UP is not None else None,
        'ghat_APCA': ((SA - UA) - (SP - UP)) if (SP is not None and UP is not None) else None,
        'SA': SA, 'UA': UA, 'SB': SB, 'UB': UB, 'SP': SP, 'UP': UP}
if DATASEED is not None:
    data['meta'] = {
        'dataseed': DATASEED,
        'mae_root': MAE_ROOT,
        'mae_conds': MAE_CONDS,          # B0 absent: not run for the data-seed axis
        'simple_root': SIMPLE_SRC,
        'random_source': f'{SIMPLE_SRC}/<fold>/random/per_fault_by_seed.json[{DATASEED}]',
        'deterministic_note': ('data-seed axis: dataset allocation re-drawn with seed '
                               f'{DATASEED} — ALL simple baselines are data-dependent '
                               '(no seed-42 reuse); missing simple results → null cells')}
elif SEED != 42:
    data['meta'] = {
        'seed': SEED,
        'mae_root': MAE_ROOT,
        'mae_conds': MAE_CONDS,          # B0 absent: not run for extra seeds (Table 4 밖)
        'random_source': f'{SIMPLE}/<fold>/random/per_fault_by_seed.json[{SEED}]',
        'deterministic_note': ('PCA recon./NN-distance/Sensor range/L2-norm reuse the '
                               'seed-42 values (deterministic, seed-independent)')}
json.dump(data, open(f'{BASE_ROOT}/table4_data{SFX}.json', 'w'), indent=1)

# ---- fill-ready text (paper table order) ----
def _f(v):  # None-safe cell (Random row may be empty for extra seeds pre-re-run)
    return f'{v:.4f}' if v is not None else '  --  '
L = []
def row(name, fn):
    L.append(f'{name:<16} ' + '  '.join(f'{_f(fn(fk)[0])} {_f(fn(fk)[1])}' for fk in FOLDS))
L.append('Condition            F-STEP S/U      F-RAND S/U      F-DS S/U        F-UNK S/U')
L.append('-- Simple baselines (no labels) --')
for lbl, _ in BASELINES:
    row(lbl, lambda fk, l=lbl: (data['simple'][l][fk]['S'], data['simple'][l][fk]['U']))
L.append('-- MAE / LASAD conditions (A/B/B0/D) --')
for cond, nm in [('B0', 'B0 clean ref.'), ('B', 'B label-blind'), ('A', 'A LASAD (ours)'), ('D', 'D recon.-only')]:
    if cond not in data['mae']:
        continue
    row(nm, lambda fk, c=cond: (data['mae'][c][fk]['S'], data['mae'][c][fk]['U']))
def _fs(v):  # None-safe signed cell (PCA terms may be null on the data-seed axis)
    return f'{v:+.4f}' if v is not None else '  --  '
L.append('-- Discriminant (within-fold matched; + supports generalization) --')
L.append('ΔU (A−B)        ' + '  '.join(f'[{data["discriminant"][fk]["dU_AB"]:+.4f}]   ' for fk in FOLDS))
L.append('Γ̂_A (vs B)      ' + '  '.join(f'[{data["discriminant"][fk]["ghat_A"]:+.4f}]   ' for fk in FOLDS))
L.append('ΔU (A−PCA)      ' + '  '.join('[' + _fs(data["discriminant"][fk]["dU_APCA"]) + ']   ' for fk in FOLDS))
L.append('Γ̂ (vs PCA)      ' + '  '.join('[' + _fs(data["discriminant"][fk]["ghat_APCA"]) + ']   ' for fk in FOLDS))
txt = '\n'.join(L)
open(f'{BASE_ROOT}/table4_values{SFX}.txt', 'w').write(txt)
print(txt)
print(f'\n저장: table4_data{SFX}.json, table4_values{SFX}.txt')
