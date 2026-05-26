"""
F5 — Dataset Clustering by Anomaly Characteristics

본 2분기의 가장 강한 finding: per-dataset variability가 mean effect보다 10배 큼.
이는 dataset마다 다른 best method를 시사. Explicit clustering으로 검증.

Strategy:
1. 각 dataset의 anomaly characteristic signature 추출
2. KMeans clustering (K=3, 4, 5 sweep)
3. 각 cluster의 best method 발견 (E9, B1, B2 등 후보)
4. Cluster-conditional routing의 effect 측정 vs uniform best method
"""
import sys
from pathlib import Path
import numpy as np
import json
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, kurtosis

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from mae_anomaly.scripts.q3_exploration.core.data import (
    DatasetScores, iter_dataset_aliases, median_anomaly_segment_length, get_per_group
)
from mae_anomaly.scripts.q3_exploration.core.scoring import (
    point_score_from_loo, gauss, zscore, per_channel_points, adaptive_combine
)
from mae_anomaly.scripts.q3_exploration.core.evaluation import (
    pak_auc_f1, wilcoxon_test, per_group_summary
)


def extract_signature(ds):
    """각 dataset의 7-feature signature."""
    seg_lens = [r.end - r.start for r in ds.regions]
    if not seg_lens:
        return None

    pt_r, pt_d, pt_s, pt_f = per_channel_points(ds, stride=21)
    base = adaptive_combine(pt_r, pt_d, pt_f, use_fm=True)
    score_smoothed = gauss(base, 10)

    return {
        'median_seg': float(np.median(seg_lens)),
        'max_seg': float(max(seg_lens)),
        'std_seg': float(np.std(seg_lens) if len(seg_lens) > 1 else 0),
        'n_regions': len(seg_lens),
        'anomaly_ratio': float(ds.point_labels.mean()),
        'baseline_pak': pak_auc_f1(score_smoothed, ds.point_labels, ds.regions, ds.eval_mask),
        'recon_disc_ratio': float(pt_r.mean() / (pt_d.mean() + 1e-9)),
        'score_skewness': float(skew(score_smoothed)),
        'score_kurtosis': float(kurtosis(score_smoothed)),
        'score_iqr': float(np.percentile(score_smoothed, 75) - np.percentile(score_smoothed, 25)),
    }


def nlm_sigmoid(score, T_factor=2.0):
    centered = score - score.mean()
    T = T_factor * (score.std() + 1e-9)
    return 1.0 / (1.0 + np.exp(-np.clip(centered / T, -30, 30)))


def z5_pyramid(score):
    scales = [5, 25, 100, 400, 1600]
    s = np.maximum(score, 1e-10)
    log_s = np.log(s)
    smoothed = [gauss(log_s, max(W / 3.0, 0.5)) for W in scales]
    return np.exp(np.mean(np.stack(smoothed, axis=0), axis=0))


def compute_all_method_scores(ds):
    """각 dataset에 모든 candidate method 적용."""
    pt_r, pt_d, pt_s, pt_f = per_channel_points(ds, stride=21)
    base_unsmoothed = adaptive_combine(pt_r, pt_d, pt_f, use_fm=True)
    median_seg = median_anomaly_segment_length(ds.regions)
    sigma_adapt = median_seg / 3.0

    # Methods to evaluate per dataset
    methods = {
        'baseline_gauss10': gauss(base_unsmoothed, 10),
        'e9_adapt': gauss(base_unsmoothed, sigma_adapt),
        'b1_e9_nlm': nlm_sigmoid(gauss(base_unsmoothed, sigma_adapt), T_factor=2.0),
        'b2_conditional': gauss(base_unsmoothed, min(sigma_adapt, 50) if median_seg > 300 else sigma_adapt),
        'z5_pyramid': z5_pyramid(base_unsmoothed),
        'gauss5': gauss(base_unsmoothed, 5),
        'gauss30': gauss(base_unsmoothed, 30),
        'gauss100': gauss(base_unsmoothed, 100),
    }
    return {name: pak_auc_f1(s, ds.point_labels, ds.regions, ds.eval_mask)
            for name, s in methods.items()}


def main():
    targets = iter_dataset_aliases()
    print(f"F5 — Dataset Clustering, {len(targets)} datasets")

    signatures = {}
    all_method_scores = {}

    for i, (alias, swat) in enumerate(targets, 1):
        ds = DatasetScores.load(alias, swat)
        if ds is None:
            continue
        sig = extract_signature(ds)
        if sig is None:
            continue
        signatures[alias] = sig
        all_method_scores[alias] = compute_all_method_scores(ds)
        if i % 10 == 0 or i == len(targets):
            print(f"[{i:2d}/{len(targets)}] processed", flush=True)

    # Feature matrix
    aliases = list(signatures.keys())
    feat_keys = ['median_seg', 'max_seg', 'std_seg', 'n_regions', 'anomaly_ratio',
                 'baseline_pak', 'recon_disc_ratio', 'score_skewness', 'score_kurtosis', 'score_iqr']
    X = np.array([[signatures[a][k] for k in feat_keys] for a in aliases])

    # Log-scale segment features (heavy tail)
    X[:, 0] = np.log10(X[:, 0] + 1)  # median_seg
    X[:, 1] = np.log10(X[:, 1] + 1)  # max_seg
    X[:, 2] = np.log10(X[:, 2] + 1)  # std_seg
    X[:, 3] = np.log10(X[:, 3] + 1)  # n_regions

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cluster sweep K=3, 4, 5
    print("\n=== Cluster sweep ===")
    cluster_results = {}
    for K in [3, 4, 5]:
        km = KMeans(n_clusters=K, random_state=42, n_init=10)
        cluster_ids = km.fit_predict(X_scaled)
        cluster_results[K] = dict(zip(aliases, cluster_ids.tolist()))

        # Per-cluster best method
        print(f"\nK={K}:")
        for c in range(K):
            cluster_aliases = [a for a, ci in zip(aliases, cluster_ids) if ci == c]
            if not cluster_aliases:
                continue
            print(f"  Cluster {c} (n={len(cluster_aliases)}): {cluster_aliases[:4]}{' ...' if len(cluster_aliases) > 4 else ''}")
            # Cluster의 method별 평균 Δ vs baseline
            print(f"    method effects within cluster:")
            best_method, best_delta = None, -np.inf
            for method in ['e9_adapt', 'b1_e9_nlm', 'b2_conditional', 'z5_pyramid', 'gauss5', 'gauss30', 'gauss100']:
                deltas = [all_method_scores[a][method] - all_method_scores[a]['baseline_gauss10']
                          for a in cluster_aliases]
                md = np.mean(deltas)
                if md > best_delta:
                    best_delta, best_method = md, method
                wins = sum(1 for d in deltas if d > 0)
                if abs(md) > 0.005:
                    print(f"      {method:<18s}: meanΔ={md:+.4f}  W/L={wins}/{len(deltas)-wins}")
            print(f"    => best: {best_method} (meanΔ={best_delta:+.4f})")
            cluster_results.setdefault(f'K{K}_best', {})[c] = best_method

    # Best clustering: K=4 (heuristic)
    K_best = 4
    cluster_to_method = cluster_results[f'K{K_best}_best']
    print(f"\n=== Cluster-conditional routing (K={K_best}) ===")
    routed_scores = []
    baseline_scores = []
    cluster_ids_best = list(cluster_results[K_best].values())
    for a in aliases:
        ci = cluster_results[K_best][a]
        best_m = cluster_to_method[ci]
        routed_score = all_method_scores[a][best_m]
        baseline_score = all_method_scores[a]['baseline_gauss10']
        routed_scores.append(routed_score)
        baseline_scores.append(baseline_score)

    deltas = np.array(routed_scores) - np.array(baseline_scores)
    mean_d = deltas.mean()
    wins = (deltas > 0).sum()
    losses = (deltas < 0).sum()
    cata = (deltas < -0.05).sum()
    p = wilcoxon_test(routed_scores, baseline_scores, alternative='greater')
    print(f"  Cluster-routed: meanΔ={mean_d:+.4f}  W/L={wins}/{losses}  cata={cata}  p(>)={p:.3f}")

    # Per-group breakdown
    print("\n=== Per-group breakdown ===")
    routed_deltas_dict = dict(zip(aliases, deltas))
    summary = per_group_summary(routed_deltas_dict, get_per_group)
    for g, s in summary.items():
        print(f"  {g:<12s} n={s['n']:2d}  meanΔ={s['mean_delta']:+.4f}  W/L={s['wins']}/{s['losses']}  cata={s['catastrophic']}")

    # Save
    out = Path(__file__).parent.parent / 'results' / 'F5_dataset_clustering.json'
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w') as f:
        json.dump({
            'signatures': signatures,
            'method_scores': all_method_scores,
            'cluster_K3': cluster_results[3],
            'cluster_K4': cluster_results[4],
            'cluster_K5': cluster_results[5],
            'K4_best_methods': cluster_results['K4_best'],
            'routed_summary': {
                'mean_delta': float(mean_d), 'wins': int(wins),
                'losses': int(losses), 'cata': int(cata), 'p_value': p,
            },
        }, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
