"""
P1 — Combined Tri-Routing: Cluster × Method × σ

본 실험은 Q3 작업의 핵심 winners를 결합:
- F5 dataset clustering (per-dataset signature 기반)
- B1 hybrid (E9 × NLM-T2)
- σ multiplier sweep (median_seg / k for various k)

Strategy:
1. 각 dataset의 supervised signature 추출
2. K=4 clustering으로 4개 group
3. Per-cluster grid search over (method × σ_multiplier) → 최적 조합 발견
4. Cluster-routed final scores 계산

Expected: F5 alone (+0.0202)과 B1 alone (+0.0171)의 union이 +0.025 ~ +0.030 가능
"""
import sys
from pathlib import Path
import numpy as np
import json
from collections import defaultdict

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
from mae_anomaly.scripts.q3_exploration.core.clustering import (
    DatasetSignature, extract_signature_supervised, extract_signature_unsupervised,
    run_kmeans_clustering, get_cluster_best_methods, apply_cluster_routing,
)
from mae_anomaly.scripts.q3_exploration.core.postprocess import (
    nlm_sigmoid_transform, z5_pyramid, double_gaussian, savitzky_golay_smooth,
    median_filter1d,
)


# Method library — 다양한 candidates를 풍부하게
def build_method_candidates(base_unsmoothed, median_seg):
    """20개 method candidate을 생성 → cluster routing에서 best 선택."""
    candidates = {}

    # σ multiplier variants of E9
    for divisor in [2.5, 3.0, 4.0, 5.0, 6.0, 8.0]:
        sigma = max(median_seg / divisor, 0.5)
        candidates[f'e9_div{divisor:.1f}'] = gauss(base_unsmoothed, sigma)

    # Fixed σ baselines
    for sigma in [5, 10, 20, 30, 50, 100]:
        candidates[f'gauss{sigma}'] = gauss(base_unsmoothed, sigma)

    # Z5 pyramid
    candidates['z5_pyramid'] = z5_pyramid(base_unsmoothed)

    # Hybrids: E9 + NLM-T at different T
    e9_smoothed = gauss(base_unsmoothed, max(median_seg / 3.0, 0.5))
    for T in [1.0, 1.5, 2.0, 3.0]:
        candidates[f'e9_nlm_T{T}'] = nlm_sigmoid_transform(e9_smoothed, T_factor=T)

    # E9 with σ=median_seg/5 + NLM (combining best findings)
    e9_div5_smoothed = gauss(base_unsmoothed, max(median_seg / 5.0, 0.5))
    candidates['e9_div5_nlm_T2'] = nlm_sigmoid_transform(e9_div5_smoothed, T_factor=2.0)
    candidates['e9_div5_nlm_T3'] = nlm_sigmoid_transform(e9_div5_smoothed, T_factor=3.0)

    # Double gaussian (short + long)
    candidates['dbl_gauss_5_30'] = double_gaussian(base_unsmoothed, 5, 30, 0.7)
    candidates['dbl_gauss_10_100'] = double_gaussian(base_unsmoothed, 10, 100, 0.6)

    # Savitzky-Golay
    candidates['savgol_21_3'] = savitzky_golay_smooth(base_unsmoothed, 21, 3)

    return candidates


def evaluate_all_candidates(ds, candidates):
    """각 candidate에 대해 pak_auc_f1 계산."""
    results = {}
    for name, score in candidates.items():
        results[name] = pak_auc_f1(score, ds.point_labels, ds.regions, ds.eval_mask)
    return results


def main():
    targets = iter_dataset_aliases()
    print(f"P1 — Tri-Routing (Cluster × Method × σ), {len(targets)} datasets")

    signatures = {}
    all_method_scores = {}

    print("\n--- Stage 1: Method candidate evaluation per dataset ---")
    for i, (alias, swat) in enumerate(targets, 1):
        ds = DatasetScores.load(alias, swat)
        if ds is None:
            continue

        pt_r, pt_d, pt_s, pt_f = per_channel_points(ds, stride=21)
        base_unsmoothed = adaptive_combine(pt_r, pt_d, pt_f, use_fm=True)
        median_seg = median_anomaly_segment_length(ds.regions)
        base_smoothed_for_sig = gauss(base_unsmoothed, 10)

        # Baseline
        baseline_pak = pak_auc_f1(base_smoothed_for_sig,
                                   ds.point_labels, ds.regions, ds.eval_mask)

        # Build candidates
        candidates = build_method_candidates(base_unsmoothed, median_seg)
        candidates['baseline_gauss10'] = base_smoothed_for_sig

        scores = evaluate_all_candidates(ds, candidates)
        all_method_scores[alias] = scores

        # Signature (combined unsupervised + supervised features)
        unsup = extract_signature_unsupervised(base_smoothed_for_sig, pt_r, pt_d, pt_f)
        sup = extract_signature_supervised(ds.regions, ds.point_labels, baseline_pak)

        sig = DatasetSignature(alias=alias, features={**unsup, **sup})
        signatures[alias] = sig

        if i % 10 == 0 or i == len(targets):
            print(f"  [{i:2d}/{len(targets)}] processed, n_candidates={len(candidates)}", flush=True)

    print(f"\nN candidates per dataset: {len(all_method_scores[list(all_method_scores.keys())[0]])}")

    # --- Stage 2: Cluster sweep with two signature variants ---
    feature_sets = {
        'supervised': ['median_seg_log', 'max_seg_log', 'std_seg_log', 'n_regions_log',
                       'anomaly_ratio', 'baseline_pak', 'skewness', 'kurtosis',
                       'iqr', 'autocorr_lag1', 'recon_disc_ratio_log', 'disc_fm_ratio_log'],
        'unsupervised': ['skewness', 'kurtosis', 'iqr', 'std_to_mean', 'autocorr_lag1',
                         'top5_median_ratio', 'recon_disc_ratio_log', 'disc_fm_ratio_log',
                         'seq_len_log', 'peak_density'],
    }

    print("\n--- Stage 2: Cluster sweep ---")

    all_results = {}
    for sig_name, feature_keys in feature_sets.items():
        print(f"\n=== Signature: {sig_name} ({len(feature_keys)} features) ===")
        for K in [3, 4, 5, 6]:
            cluster_ids, _, _ = run_kmeans_clustering(signatures, feature_keys, K)
            cluster_to_method = get_cluster_best_methods(
                cluster_ids, all_method_scores, baseline_method='baseline_gauss10'
            )

            # Apply routing
            routed = apply_cluster_routing(cluster_ids, cluster_to_method, all_method_scores)
            baselines = {a: all_method_scores[a]['baseline_gauss10'] for a in routed}

            routed_scores = list(routed.values())
            baseline_scores = [baselines[a] for a in routed]
            deltas = np.array(routed_scores) - np.array(baseline_scores)

            mean_d = float(deltas.mean())
            wins = int((deltas > 0).sum())
            losses = int((deltas < 0).sum())
            cata = int((deltas < -0.05).sum())
            p = wilcoxon_test(routed_scores, baseline_scores, alternative='greater')

            print(f"  K={K}: meanΔ={mean_d:+.4f}  W/L={wins:2d}/{losses:2d}  cata={cata}  p={p:.4f}")
            print(f"      cluster→method: {dict((c, m) for c, m in cluster_to_method.items())}")

            cluster_sizes = defaultdict(int)
            for cid in cluster_ids.values():
                cluster_sizes[cid] += 1
            print(f"      cluster sizes:  {dict(cluster_sizes)}")

            key = f'{sig_name}_K{K}'
            all_results[key] = {
                'mean_delta': mean_d,
                'wins': wins, 'losses': losses, 'cata': cata,
                'p_value': p,
                'cluster_to_method': cluster_to_method,
                'cluster_sizes': dict(cluster_sizes),
                'cluster_ids': cluster_ids,
                'routed_scores': dict(zip(list(routed.keys()), routed_scores)),
            }

    # --- Stage 3: Per-group breakdown for best routing ---
    best_key = max(all_results, key=lambda k: all_results[k]['mean_delta'])
    print(f"\n\n=== BEST: {best_key} ===")
    print(f"  mean Δ: {all_results[best_key]['mean_delta']:+.4f}")
    print(f"  W/L:    {all_results[best_key]['wins']}/{all_results[best_key]['losses']}")
    print(f"  cata:   {all_results[best_key]['cata']}")
    print(f"  p:      {all_results[best_key]['p_value']:.4f}")
    print(f"  routing: {all_results[best_key]['cluster_to_method']}")

    # Per-group
    best_routed = all_results[best_key]['routed_scores']
    deltas_dict = {a: best_routed[a] - all_method_scores[a]['baseline_gauss10']
                    for a in best_routed}
    summary = per_group_summary(deltas_dict, get_per_group)
    print(f"\n  Per-group:")
    for g, s in summary.items():
        print(f"    {g:<12s} n={s['n']:2d}  meanΔ={s['mean_delta']:+.4f}  "
              f"W/L={s['wins']}/{s['losses']}  cata={s['catastrophic']}")

    # --- Stage 4: Top-5 method usage analysis ---
    print(f"\n=== Method usage analysis (best routing) ===")
    method_usage = defaultdict(int)
    cluster_ids_best = all_results[best_key]['cluster_ids']
    cluster_to_method_best = all_results[best_key]['cluster_to_method']
    for alias, cid in cluster_ids_best.items():
        method_usage[cluster_to_method_best[cid]] += 1
    for method, count in sorted(method_usage.items(), key=lambda x: -x[1]):
        print(f"    {method:<25s}: {count} datasets")

    # --- Save ---
    out = Path(__file__).parent.parent / 'results' / 'P1_tri_routing.json'
    out.parent.mkdir(exist_ok=True)
    save_data = {
        'all_routing_results': {k: {**v, 'cluster_ids': v['cluster_ids']}
                                 for k, v in all_results.items()},
        'best_key': best_key,
        'best_routing': all_results[best_key]['cluster_to_method'],
        'signatures': {a: s.features for a, s in signatures.items()},
        'all_method_scores': all_method_scores,
    }
    with open(out, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
