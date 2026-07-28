"""build_table_d1.py — per-mode S/U pak_auc_f1 aggregation for MAE TEP type-gen runs.

Reuses the SIMPLE-baseline machinery (partition_eval, compute_all_metrics) so MAE
numbers are computed under the IDENTICAL per-mode definition (design §3):
  per-fault = that fault's 20 runs + shared 40 FF runs (pos rate ~27.8%);
  per-mode S/U = MACRO mean of per-fault pak_auc_f1 over seen/unseen USABLE faults
  (IDV 3/9/15 excluded-hard already dropped by USABLE_FAULTS).
Score = best post-warmup epoch's official_score (causal), matching best-epoch selection.
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from tep_common import seen_faults, unseen_faults, USABLE_FAULTS, ALL_FAULTS, FAMILY, EXCLUDED_HARD
from run_tep_simple import partition_eval, load_test

FOLD_KEYS = {'fstep': 'f_step', 'frand': 'f_rand', 'fds': 'f_ds', 'funk': 'f_unk'}
FAM_OF = {f: fam for fam, fs in FAMILY.items() for f in fs}
ROOT = 'results/experiments/TEP_phase2_win100_ep30'
WARMUP = 15


def best_postwarmup_epoch(em_path):
    rows = json.load(open(em_path)).get('epochs', [])
    pw = [r for r in rows if r.get('epoch', 0) > WARMUP] or rows
    return max(pw, key=lambda r: r.get('pak_auc_f1', 0)).get('epoch')


def all_fault_paks(score, y, run_table):
    """pak_auc_f1 for each fault's partition (fault runs + FF). ALL 20 incl EXCL-HARD."""
    out = {}
    for f in ALL_FAULTS:
        m, _, _ = partition_eval(score, y, run_table, {f}, lite=True)
        out[f] = m.get('pak_auc_f1', 0.0)
    return out


def macro(paks, faults):
    return float(np.mean([paks[f] for f in faults]))


def load_score(cond_sub, fold_short, key='official_score'):
    d = f'{ROOT}/{cond_sub}/TEP/typegen_{fold_short}'
    em = f'{d}/epoch_metrics.json'
    if not os.path.exists(em):
        return None, None
    be = best_postwarmup_epoch(em)
    npz = f'{d}/epoch_scores/epoch_{be:03d}_scores.npz'
    return np.load(npz)[key], be


def main():
    _, y, _, _, run_table = load_test()
    res = {}

    # A / B : one run per fold
    for cond, sub in [('A', 'phase2_A'), ('B', 'phase2_B')]:
        for fs, fk in FOLD_KEYS.items():
            score, be = load_score(sub, fs)
            if score is None:
                continue
            paks = all_fault_paks(score, y, run_table)
            res[(cond, fs)] = dict(ep=be,
                                   S=macro(paks, seen_faults(fk)),
                                   U=macro(paks, unseen_faults(fk)),
                                   paks=paks)

    # B0 : ffonly scores are fold-independent → compute paks ONCE, macro per fold
    score0, be0 = load_score('phase2_B0', 'ffonly')
    if score0 is not None:
        paks0 = all_fault_paks(score0, y, run_table)
        for fs, fk in FOLD_KEYS.items():
            res[('B0', fs)] = dict(ep=be0,
                                   S=macro(paks0, seen_faults(fk)),
                                   U=macro(paks0, unseen_faults(fk)),
                                   paks=paks0)

    # D : recon-only = teacher_recon_error from the A run (teacher detached, free)
    for fs, fk in FOLD_KEYS.items():
        score, be = load_score('phase2_A', fs, key='teacher_recon_error')
        if score is None:
            continue
        paks = all_fault_paks(score, y, run_table)
        res[('D', fs)] = dict(ep=be, S=macro(paks, seen_faults(fk)),
                              U=macro(paks, unseen_faults(fk)), paks=paks)

    # ---- print table ----
    print(f"\n{'Cond':5} {'fold':6} {'ep':>3} {'S':>8} {'U':>8}")
    for cond in ['A', 'B', 'B0', 'D']:
        for fs in FOLD_KEYS:
            r = res.get((cond, fs))
            if r:
                print(f"{cond:5} {fs:6} {r['ep']:>3} {r['S']:>8.4f} {r['U']:>8.4f}")

    print("\n=== 판정량 ===")
    print(f"{'fold':6} {'A_U':>8} {'B_U':>8} {'D_U':>8} {'Δ_uns(A-B)':>11} {'A_S':>8} {'B_S':>8} {'C_dmg(B0_S-A_S)':>16}")
    for fs in FOLD_KEYS:
        a, b, b0, d = (res.get((c, fs)) for c in ['A', 'B', 'B0', 'D'])
        if a and b:
            du = a['U'] - b['U']
            cdmg = (b0['S'] - a['S']) if b0 else float('nan')
            print(f"{fs:6} {a['U']:>8.4f} {b['U']:>8.4f} {(d['U'] if d else 0):>8.4f} "
                  f"{du:>+11.4f} {a['S']:>8.4f} {b['S']:>8.4f} {cdmg:>16.4f}")

    # macro over folds
    def fold_macro(cond, key):
        v = [res[(cond, fs)][key] for fs in FOLD_KEYS if (cond, fs) in res]
        return float(np.mean(v)) if v else float('nan')
    print("\n=== fold-macro 요약 ===")
    for cond in ['A', 'B', 'B0', 'D']:
        if any((cond, fs) in res for fs in FOLD_KEYS):
            print(f"{cond:3} : S={fold_macro(cond,'S'):.4f}  U={fold_macro(cond,'U'):.4f}")
    print(f"Δ_unseen (A_U−B_U) fold-macro = {fold_macro('A','U')-fold_macro('B','U'):+.4f}")

    # ---- per-fault matrix (부록 B style; ★ = seen in that fold) ----
    paks0 = res.get(('B0', 'fstep'), {}).get('paks', {})
    def star(f, fs):
        return '*' if f in seen_faults(FOLD_KEYS[fs]) else ' '
    def c(cond, fs, f):
        r = res.get((cond, fs))
        return f"{r['paks'][f]:.3f}{star(f, fs)}" if r else '  --  '
    print("\n=== per-fault pak_auc_f1  (*=seen·labeled in that fold) ===")
    print(f"{'IDV':>4} {'fam':>5} {'B0':>6} | " +
          " ".join(f"A@{fs:<5}" for fs in FOLD_KEYS) + "| " +
          " ".join(f"B@{fs:<5}" for fs in FOLD_KEYS))
    for f in ALL_FAULTS:
        fam = FAM_OF.get(f, 'EXCL')
        b0 = paks0.get(f, 0)
        print(f"{f:>4} {fam:>5} {b0:>6.3f} | " +
              " ".join(f"{c('A',fs,f):>7}" for fs in FOLD_KEYS) + "| " +
              " ".join(f"{c('B',fs,f):>7}" for fs in FOLD_KEYS))

    # ---- seen faults: does labeling (A) recover the contaminated fault vs B / B0? ----
    print("\n=== SEEN(labeled) fault별 회복: B0(clean) → A(labeled) → B(blind) ===")
    for fs, fk in FOLD_KEYS.items():
        print(f"-- {fs} --")
        for f in seen_faults(fk):
            a = res[('A', fs)]['paks'][f]; b = res[('B', fs)]['paks'][f]; b0 = paks0.get(f, 0)
            print(f"  IDV{f:<2}({FAM_OF.get(f,''):>5}): B0={b0:.3f}  A={a:.3f}  B={b:.3f}  "
                  f"A-B={a-b:+.3f}  (A vs ceiling)={a-b0:+.3f}")

    json.dump({f"{c_}_{fs}": r for (c_, fs), r in res.items()},
              open(f'{ROOT}/table_d1_permode.json', 'w'), indent=2)
    print(f"\n저장: {ROOT}/table_d1_permode.json")


if __name__ == '__main__':
    main()
