"""build_mechanism.py — causal chain: homogeneity → unlabeled-faulty absorbed into normal set →
inflates train-normal disc (D_tr) → erratic R_tr/D_tr → U-valley.
Identify unlabeled-faulty = (TRUE y==1) & (model-seen label==0). Compare their disc to genuine FF.
Prediction: homogeneous family (step) → unlabeled-faulty form a tight high-signal cluster → larger
D_tr inflation than heterogeneous family (rand)."""
import json, glob, sys
import numpy as np
sys.path.insert(0, 'scripts/TEP'); sys.path.insert(0, '.')

R = 'results/experiments/TEP_phase2_win100_ep30'
DATA = 'scripts/TEP/data'
FAM_FILE = {'fstep': 'train_f_step.npz', 'frand': 'train_f_rand.npz', 'fds': 'train_f_ds.npz', 'funk': 'train_f_unk.npz'}

def analyze(fold, lab):
    d = f'{R}/noisy/TEP/typegen_{fold}_{lab}'
    f = glob.glob(f'{d}/best_epoch_train_scores.npz')
    if not f: return None
    z = np.load(f[0])
    rec = np.nan_to_num(z['teacher_recon_error']).astype(np.float64)
    dis = np.nan_to_num(z['discrepancy_error']).astype(np.float64)
    seen = z['point_labels'].astype(int)                       # model-seen labels (zeroed)
    true_y = np.load(f'{DATA}/{FAM_FILE[fold]}')['y'].astype(int)  # TRUE labels (pre-zeroing)
    if len(true_y) != len(seen): return None                   # alignment guard
    unl = (true_y == 1) & (seen == 0)        # unlabeled-faulty: anomaly masquerading as normal
    ff = (true_y == 0)                        # genuine normal (FF + normal regions)
    lbl = (seen == 1)                         # labeled-faulty (force_mask target)
    seen_normal = (seen == 0)                 # what the model treats as normal (FF + unlabeled-faulty)
    D_tr = dis[seen_normal].mean()            # the actual train-normal disc mean (drives R_tr/D_tr)
    D_ff = dis[ff].mean()                     # genuine-normal disc mean (no contamination)
    return {
        'n_unl': int(unl.sum()), 'n_lbl': int(lbl.sum()), 'n_ff': int(ff.sum()),
        'disc_ff': float(dis[ff].mean()), 'disc_unl': float(dis[unl].mean()) if unl.any() else 0.0,
        'disc_lbl': float(dis[lbl].mean()) if lbl.any() else 0.0,
        'recon_ff': float(rec[ff].mean()), 'recon_unl': float(rec[unl].mean()) if unl.any() else 0.0,
        'D_tr': float(D_tr), 'D_ff_only': float(D_ff),
        'D_tr_inflation': float(D_tr / (D_ff + 1e-12)),         # >1 = unlabeled-faulty inflate normal disc
        'unl_vs_ff_disc': float(dis[unl].mean() / (dis[ff].mean() + 1e-12)) if unl.any() else 0.0,
    }

hom = json.load(open(f'{R}/homogeneity.json'))['per_family_hom']
FOLD_FAM = {'fstep': 'step', 'frand': 'rand', 'fds': 'ds', 'funk': 'unk'}
print('=== mechanism: unlabeled-faulty absorption inflates train-normal disc (D_tr) ===')
print(f"{'fold':>6} {'lab':>6} {'hom':>6} | {'disc_ff':>8} {'disc_unl':>9} {'unl/ff':>7} | {'D_tr/D_ff_only':>15} {'recon_unl/ff':>13}")
rows = []
for fold in ['fstep', 'frand', 'fds', 'funk']:
    for lab in ['lab80', 'lab50', 'lab25']:
        a = analyze(fold, lab)
        if not a: continue
        rec_ratio = a['recon_unl'] / (a['recon_ff'] + 1e-12)
        print(f"  {fold:>6} {lab:>6} {hom[FOLD_FAM[fold]]:>6.3f} | {a['disc_ff']:>8.4f} {a['disc_unl']:>9.4f} {a['unl_vs_ff_disc']:>7.2f} | {a['D_tr_inflation']:>15.3f} {rec_ratio:>13.3f}")
        rows.append({'fold': fold, 'lab': lab, 'hom': hom[FOLD_FAM[fold]], **a})

# correlate D_tr inflation with homogeneity (lab50, the mid sweep)
import numpy as np
mid = [r for r in rows if r['lab'] == 'lab50']
if len(mid) >= 3:
    x = np.array([r['hom'] for r in mid]); y = np.array([r['D_tr_inflation'] for r in mid])
    print(f"\ncorr(homogeneity, D_tr_inflation @lab50) = {np.corrcoef(x, y)[0,1]:+.3f}")
    print('  → 균일 family일수록 unlabeled-faulty가 normal disc를 더 크게 inflate (가설 메커니즘)')
json.dump(rows, open(f'{R}/mechanism.json', 'w'), indent=1)
print('\n저장: mechanism.json')
