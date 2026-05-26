"""
F2 — Cross-Channel Interaction Scoring

현재 274 모델의 anomaly_score = recon + scaled_disc + scaled_fm. Additive only.
본 실험은 higher-order interaction terms (multiplicative, non-linear)을 검증.

Methods:
- F2_1: r×d (recon × disc geometric mean)
- F2_2: r-s (teacher recon - student recon)
- F2_3: r + sqrt(r*d)
- F2_4: r * sigmoid(d)
- F2_5: r * (1 + d/d.max())
- F2_6: max(r, d, f) per-point (channel-wise max)
- F2_7: harmonic mean of (r, d, f)
"""
import sys
from pathlib import Path
import numpy as np
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from mae_anomaly.scripts.q3_exploration.core.data import (
    DatasetScores, iter_dataset_aliases, median_anomaly_segment_length, get_per_group
)
from mae_anomaly.scripts.q3_exploration.core.scoring import (
    per_channel_points, adaptive_combine, gauss, zscore
)
from mae_anomaly.scripts.q3_exploration.core.evaluation import (
    pak_auc_f1, wilcoxon_test, per_group_summary
)


def normalize_positive(x):
    """Min-shift to ensure positive."""
    return x - x.min() + 1e-10


def f2_v1_geom_rd(pt_r, pt_d):
    """F2_1: recon + 0.5 * sqrt(recon * disc)."""
    r = normalize_positive(pt_r)
    d = normalize_positive(pt_d)
    return r + 0.5 * np.sqrt(r * d)


def f2_v2_teacher_student_diff(pt_r, pt_s):
    """F2_2: teacher recon - student recon (positive = teacher worse → maybe anomaly)."""
    return pt_r - pt_s


def f2_v3_r_plus_sqrt_rdf(pt_r, pt_d, pt_f):
    """F2_3: recon + sqrt(recon * (disc + fm))."""
    r = normalize_positive(pt_r)
    d = normalize_positive(pt_d)
    f = normalize_positive(pt_f)
    return r + 0.5 * np.sqrt(r * (d + f))


def f2_v4_r_sigmoid_d(pt_r, pt_d):
    """F2_4: recon * (1 + sigmoid_z(d))."""
    d_z = zscore(pt_d)
    sig = 1.0 / (1.0 + np.exp(-d_z))
    return pt_r * (1 + sig)


def f2_v5_max_channel(pt_r, pt_d, pt_f):
    """F2_5: channel-wise z-norm then max."""
    return np.maximum.reduce([zscore(pt_r), zscore(pt_d), zscore(pt_f)])


def f2_v6_harmonic(pt_r, pt_d, pt_f):
    """F2_6: harmonic mean of z-normed channels (positive shifted)."""
    eps = 1e-3
    zr = zscore(pt_r) + 5  # shift positive
    zd = zscore(pt_d) + 5
    zf = zscore(pt_f) + 5
    return 3.0 / (1.0/(zr+eps) + 1.0/(zd+eps) + 1.0/(zf+eps))


def f2_v7_robust_weighted(pt_r, pt_d, pt_f):
    """F2_7: adaptive weighting based on per-channel signal-to-noise.
    Each channel weighted by inverse of its IQR (more weight to less noisy)."""
    iqrs = []
    for ch in [pt_r, pt_d, pt_f]:
        iqr = np.percentile(ch, 75) - np.percentile(ch, 25)
        iqrs.append(iqr + 1e-9)
    weights = 1.0 / np.array(iqrs)
    weights /= weights.sum()
    return weights[0] * zscore(pt_r) + weights[1] * zscore(pt_d) + weights[2] * zscore(pt_f)


def process_one(alias, swat_excl22):
    ds = DatasetScores.load(alias, swat_excl22)
    if ds is None:
        return None

    pt_r, pt_d, pt_s, pt_f = per_channel_points(ds, stride=21)

    # Baseline: adaptive_combine + gauss10
    base = adaptive_combine(pt_r, pt_d, pt_f, use_fm=True)
    baseline = gauss(base, 10)
    baseline_pak = pak_auc_f1(baseline, ds.point_labels, ds.regions, ds.eval_mask)

    # F2 variants
    f2_1 = gauss(f2_v1_geom_rd(pt_r, pt_d), 10)
    f2_2 = gauss(f2_v2_teacher_student_diff(pt_r, pt_s), 10)
    f2_3 = gauss(f2_v3_r_plus_sqrt_rdf(pt_r, pt_d, pt_f), 10)
    f2_4 = gauss(f2_v4_r_sigmoid_d(pt_r, pt_d), 10)
    f2_5 = gauss(f2_v5_max_channel(pt_r, pt_d, pt_f), 10)
    f2_6 = gauss(f2_v6_harmonic(pt_r, pt_d, pt_f), 10)
    f2_7 = gauss(f2_v7_robust_weighted(pt_r, pt_d, pt_f), 10)

    return {
        'alias': alias, 'baseline_pak': baseline_pak,
        'f2_1_pak': pak_auc_f1(f2_1, ds.point_labels, ds.regions, ds.eval_mask),
        'f2_2_pak': pak_auc_f1(f2_2, ds.point_labels, ds.regions, ds.eval_mask),
        'f2_3_pak': pak_auc_f1(f2_3, ds.point_labels, ds.regions, ds.eval_mask),
        'f2_4_pak': pak_auc_f1(f2_4, ds.point_labels, ds.regions, ds.eval_mask),
        'f2_5_pak': pak_auc_f1(f2_5, ds.point_labels, ds.regions, ds.eval_mask),
        'f2_6_pak': pak_auc_f1(f2_6, ds.point_labels, ds.regions, ds.eval_mask),
        'f2_7_pak': pak_auc_f1(f2_7, ds.point_labels, ds.regions, ds.eval_mask),
    }


def main():
    targets = iter_dataset_aliases()
    print(f"F2 — Cross-Channel Interaction, {len(targets)} datasets")

    results = {}
    for i, (alias, swat) in enumerate(targets, 1):
        try:
            r = process_one(alias, swat)
            if r is None: continue
            results[alias] = r
            if i % 10 == 0 or i == len(targets):
                print(f"[{i:2d}/{len(targets)}] processed", flush=True)
        except Exception as e:
            print(f"FAILED {alias}: {e}")

    out = Path(__file__).parent.parent / 'results' / 'F2_cross_channel.json'
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")

    print("\n=== Aggregate Δ vs baseline ===")
    for method in ['f2_1', 'f2_2', 'f2_3', 'f2_4', 'f2_5', 'f2_6', 'f2_7']:
        deltas = {a: r[f'{method}_pak'] - r['baseline_pak'] for a, r in results.items()}
        baseline_s = [r['baseline_pak'] for r in results.values()]
        method_s = [r[f'{method}_pak'] for r in results.values()]
        mean_d = np.mean(list(deltas.values()))
        wins = sum(1 for d in deltas.values() if d > 0)
        losses = sum(1 for d in deltas.values() if d < 0)
        cata = sum(1 for d in deltas.values() if d < -0.05)
        p = wilcoxon_test(method_s, baseline_s, alternative='greater')
        desc = {
            'f2_1': 'r+sqrt(r*d)', 'f2_2': 'r-s', 'f2_3': 'r+sqrt(r*(d+f))',
            'f2_4': 'r*(1+sig(d))', 'f2_5': 'max(zr,zd,zf)',
            'f2_6': 'harm(zr,zd,zf)', 'f2_7': 'IQR-weighted'
        }
        print(f"  {method:<5s} [{desc[method]:<20s}]: meanΔ={mean_d:+.4f}  W/L={wins:2d}/{losses:2d}  cata={cata}  p(>)={p:.3f}")


if __name__ == "__main__":
    main()
