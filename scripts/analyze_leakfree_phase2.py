"""Phase 2: from temp/leakfree_signals.json, (a) DIAGNOSE which leak-free signal's
channel-difference predicts the better channel, (b) build discrete SELECTORS that choose
per entity among {recon, disc, sum4, sum1, max} using only train/test-unlabeled signals,
(c) evaluate mean PAK vs fixed-sum4 baseline and oracle. No new PAK calls (lookup precomputed).
"""
import json, numpy as np

R = json.load(open('/home/ykio/notebooks/TSMAE/temp/leakfree_signals.json'))
SIGS = ['trAUC', 'trNcv', 'kurt', 'skew', 'p99z', 'tailmass', 'coh']
STRICT = {'trAUC', 'trNcv'}  # train-only (fully deployable/causal); rest are transductive test

def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ra = a.argsort().argsort(); rb = b.argsort().argsort()
    return float(np.corrcoef(ra, rb)[0, 1]) if ra.std() and rb.std() else float('nan')

# truth: per-channel quality gap
pdisc = np.array([r['paks']['disc'] for r in R])
precon = np.array([r['paks']['recon'] for r in R])
psum4 = np.array([r['paks']['sum4'] for r in R])
gap = pdisc - precon  # >0 => disc is the better single channel

print("=== DIAGNOSTIC: does (signal_disc - signal_recon) predict (PAK_disc - PAK_recon)? ===")
sig_diff = {}
for s in SIGS:
    dd = np.array([r['sig']['disc'][s] - r['sig']['recon'][s] for r in R])
    sig_diff[s] = dd
    tag = 'STRICT' if s in STRICT else 'transd'
    print(f"  {s:9s} [{tag}]  spearman(diff, PAK_disc-PAK_recon) = {spearman(dd, gap):+.3f}")

# === SELECTORS: choose a candidate per entity from leak-free signals ===
# action set guided by a "disc-dominance" signal d_s = sig_disc - sig_recon
def make_selector(sig_name, hi, lo):
    """disc-dominant -> 'max' (robust) ; recon-dominant -> 'recon' ; else 'sum4'."""
    def sel(r):
        d = r['sig']['disc'][sig_name] - r['sig']['recon'][sig_name]
        if d >= hi: return 'max'
        if d <= lo: return 'recon'
        return 'sum4'
    return sel

def evaluate(sel, name):
    chosen = [sel(r) for r in R]
    paks = np.array([R[i]['paks'][chosen[i]] for i in range(len(R))])
    base = psum4
    mslc1 = next((R[i]['paks'][chosen[i]] for i, r in enumerate(R) if r['sub'] == 'MSL/C-1'), None)
    win = int((paks > base + 1e-4).sum()); loss = int((paks < base - 1e-4).sum())
    print(f"  {name:28s} mean={paks.mean():.4f} (Δ{paks.mean()-base.mean():+.4f}) "
          f"win/loss {win}/{loss} | MSL/C-1={mslc1:.4f}")
    return paks.mean()

print("\n=== baselines ===")
print(f"  {'fixed sum4 (current)':28s} mean={psum4.mean():.4f} | MSL/C-1={dict((r['sub'],r['paks']['sum4']) for r in R)['MSL/C-1']:.4f}")
oracle = np.array([max(r['paks'].values()) for r in R])
print(f"  {'ORACLE (best per entity)':28s} mean={oracle.mean():.4f} (Δ{oracle.mean()-psum4.mean():+.4f}) | "
      f"MSL/C-1={max(dict((r['sub'],r['paks']) for r in R)['MSL/C-1'].values()):.4f}")

print("\n=== SELECTORS (threshold rule per signal; tuned coarse, reported honestly) ===")
# pick thresholds from each signal-diff distribution (robust quantiles)
for s in SIGS:
    dd = sig_diff[s]
    hi = float(np.quantile(dd, 0.80)); lo = float(np.quantile(dd, 0.20))
    evaluate(make_selector(s, hi, lo), f"S[{s}] (hi={hi:.2f},lo={lo:.2f})")

# combined vote: average z-scored disc-dominance over transductive tail signals
print("\n=== combined selectors ===")
def znorm(a): a = np.asarray(a, float); return (a - a.mean()) / (a.std() + 1e-12)
for combo_name, combo in [('vote[p99z,tailmass,kurt]', ['p99z', 'tailmass', 'kurt']),
                          ('vote[trAUC,p99z]', ['trAUC', 'p99z']),
                          ('STRICT vote[trAUC,trNcv]', ['trAUC', 'trNcv'])]:
    v = np.mean([znorm(sig_diff[s]) for s in combo], axis=0)
    hi = float(np.quantile(v, 0.80)); lo = float(np.quantile(v, 0.20))
    def sel(r, _v=v, _R=R):
        i = _R.index(r);
        return 'max' if _v[i] >= hi else ('recon' if _v[i] <= lo else 'sum4')
    evaluate(sel, combo_name)

# show entities where disc>>recon (the targets) + whether STRICT trAUC catches them
print("\n=== target entities (disc beats recon by >0.05) + signal diffs ===")
print(f"{'entity':20s} recon  disc   sum4   gap   | trAUCΔ p99zΔ tailmΔ kurtΔ")
for r in sorted(R, key=lambda x: x['paks']['recon'] - x['paks']['disc']):
    g = r['paks']['disc'] - r['paks']['recon']
    if g > 0.05:
        sd = lambda s: r['sig']['disc'][s] - r['sig']['recon'][s]
        print(f"{r['sub']:20s} {r['paks']['recon']:.3f} {r['paks']['disc']:.3f} {r['paks']['sum4']:.3f} "
              f"{g:+.3f} | {sd('trAUC'):+.2f}  {sd('p99z'):+.2f}  {sd('tailmass'):+.3f} {sd('kurt'):+.1f}")
