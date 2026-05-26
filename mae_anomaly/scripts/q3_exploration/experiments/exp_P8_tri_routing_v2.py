"""
P8 — Tri-Routing v2 (Expanded method library + ensemble)

P1 tri-routing은 23 candidates였고 cluster routing이 +0.0266 달성.
본 P8은:

1. 더 풍부한 method library (P2 grid 결과의 top 변형 + multi-stride 결합):
   - σ ∈ {2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7} (9 σ)
   - T ∈ {None, 1.0, 1.5, 2.0, 2.5} (5 T)
   - Stride ∈ {7, 14, 21} (3 strides)
   → 9 × 5 × 3 = 135 candidates

2. K=4, 5, 6, 7, 8 cluster routing sweep
3. Per-cluster best method discovery
4. Final winner identification across all 39 datasets
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
    per_channel_points, adaptive_combine, gauss, zscore, aggregate_K50, stride_subsample
)
from mae_anomaly.scripts.q3_exploration.core.evaluation import (
    pak_auc_f1, wilcoxon_test, per_group_summary
)
from mae_anomaly.scripts.q3_exploration.core.clustering import (
    DatasetSignature, extract_signature_supervised, extract_signature_unsupervised,
    run_kmeans_clustering, get_cluster_best_methods, apply_cluster_routing,
)
from mae_anomaly.scripts.q3_exploration.core.postprocess import nlm_sigmoid_transform


def per_channel_at_stride(ds, stride):
    r_s, ws_s = stride_subsample(ds.recon, ds.window_start_indices, stride)
    d_s, _ = stride_subsample(ds.disc, ds.window_start_indices, stride)
    f_s, _ = stride_subsample(ds.fm, ds.window_start_indices, stride)
    pt_r = aggregate_K50(r_s, ws_s, ds.num_patches, ds.patch_size, ds.total_length)
    pt_d = aggregate_K50(d_s, ws_s, ds.num_patches, ds.patch_size, ds.total_length)
    pt_f = aggregate_K50(f_s, ws_s, ds.num_patches, ds.patch_size, ds.total_length)
    return pt_r, pt_d, pt_f


def build_method_candidates(ds, median_seg):
    """Build 135+ candidate methods per dataset."""
    candidates = {}

    sigma_divisors = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0]
    t_factors = [None, 1.0, 1.5, 2.0, 2.5]
    strides = [7, 14, 21]

    for stride in strides:
        pt_r, pt_d, pt_f = per_channel_at_stride(ds, stride)
        base_unsmoothed = adaptive_combine(pt_r, pt_d, pt_f, use_fm=True)

        for div in sigma_divisors:
            sigma = max(median_seg / div, 0.5)
            smoothed = gauss(base_unsmoothed, sigma)

            for t in t_factors:
                if t is None:
                    score = smoothed
                    key = f'div{div:.1f}_noNLM_s{stride}'
                else:
                    score = nlm_sigmoid_transform(smoothed, T_factor=t)
                    key = f'div{div:.1f}_T{t}_s{stride}'

                candidates[key] = score

    # Baseline
    pt_r, pt_d, pt_f = per_channel_at_stride(ds, 21)
    base = adaptive_combine(pt_r, pt_d, pt_f, use_fm=True)
    candidates['baseline_gauss10'] = gauss(base, 10)

    return candidates


def main():
    targets = iter_dataset_aliases()
    print(f"P8 — Tri-Routing v2 (expanded library), {len(targets)} datasets")

    signatures = {}
    all_method_scores = {}

    print("\n--- Stage 1: Build candidate library + evaluate ---")
    for i, (alias, swat) in enumerate(targets, 1):
        ds = DatasetScores.load(alias, swat)
        if ds is None:
            continue

        median_seg = median_anomaly_segment_length(ds.regions)
        candidates = build_method_candidates(ds, median_seg)

        # Evaluate all
        scores_dict = {}
        for name, score in candidates.items():
            scores_dict[name] = pak_auc_f1(score, ds.point_labels, ds.regions, ds.eval_mask)
        all_method_scores[alias] = scores_dict

        # Signature
        pt_r, pt_d, pt_f = per_channel_at_stride(ds, 21)
        base_for_sig = gauss(adaptive_combine(pt_r, pt_d, pt_f, use_fm=True), 10)
        baseline_pak = scores_dict['baseline_gauss10']
        unsup = extract_signature_unsupervised(base_for_sig, pt_r, pt_d, pt_f)
        sup = extract_signature_supervised(ds.regions, ds.point_labels, baseline_pak)
        signatures[alias] = DatasetSignature(alias=alias, features={**unsup, **sup})

        if i % 10 == 0 or i == len(targets):
            print(f"  [{i:2d}/{len(targets)}] processed (n_candidates={len(candidates)})", flush=True)

    print(f"\nTotal candidates per dataset: {len(all_method_scores[list(all_method_scores.keys())[0]])}")

    # Stage 2: Single-best method analysis (no clustering)
    print("\n--- Stage 2: Single-best method across all 39 datasets ---")
    baseline_paks = [r['baseline_gauss10'] for r in all_method_scores.values()]

    method_summaries = {}
    for method in list(all_method_scores[list(all_method_scores.keys())[0]].keys()):
        if method == 'baseline_gauss10':
            continue
        scores = [r[method] for r in all_method_scores.values()]
        deltas = np.array(scores) - np.array(baseline_paks)
        mean_d = float(deltas.mean())
        wins = int((deltas > 0).sum())
        losses = int((deltas < 0).sum())
        cata = int((deltas < -0.05).sum())
        p = wilcoxon_test(scores, baseline_paks, alternative='greater')
        method_summaries[method] = {
            'mean_delta': mean_d, 'wins': wins, 'losses': losses,
            'cata': cata, 'p_value': p,
        }

    # Top-10 single methods
    print(f"\nTop-15 single methods (no clustering):")
    sorted_methods = sorted(method_summaries.items(), key=lambda x: -x[1]['mean_delta'])
    for name, m in sorted_methods[:15]:
        print(f"  {name:<28s}: meanΔ={m['mean_delta']:+.4f}  W/L={m['wins']:2d}/{m['losses']:2d}  cata={m['cata']}  p={m['p_value']:.4f}")

    # Stage 3: Clustering sweep
    print("\n--- Stage 3: Cluster routing sweep ---")

    feature_sets = {
        'supervised': ['median_seg_log', 'max_seg_log', 'std_seg_log', 'n_regions_log',
                       'anomaly_ratio', 'baseline_pak', 'skewness', 'kurtosis',
                       'iqr', 'autocorr_lag1', 'recon_disc_ratio_log',
                       'top5_median_ratio', 'peak_density'],
        'unsupervised': ['skewness', 'kurtosis', 'iqr', 'std_to_mean', 'autocorr_lag1',
                         'top5_median_ratio', 'recon_disc_ratio_log', 'disc_fm_ratio_log',
                         'seq_len_log', 'peak_density'],
    }

    routing_results = {}
    for sig_name, feature_keys in feature_sets.items():
        print(f"\n=== Signature: {sig_name} ===")
        for K in [3, 4, 5, 6, 7, 8]:
            cluster_ids, _, _ = run_kmeans_clustering(signatures, feature_keys, K)
            cluster_to_method = get_cluster_best_methods(
                cluster_ids, all_method_scores, baseline_method='baseline_gauss10'
            )
            routed = apply_cluster_routing(cluster_ids, cluster_to_method, all_method_scores)
            routed_scores = [routed[a] for a in cluster_ids.keys()]
            baseline_scores_routed = [all_method_scores[a]['baseline_gauss10']
                                       for a in cluster_ids.keys()]
            deltas = np.array(routed_scores) - np.array(baseline_scores_routed)
            mean_d = float(deltas.mean())
            wins = int((deltas > 0).sum())
            losses = int((deltas < 0).sum())
            cata = int((deltas < -0.05).sum())
            p = wilcoxon_test(routed_scores, baseline_scores_routed, alternative='greater')

            cluster_sizes = defaultdict(int)
            for cid in cluster_ids.values():
                cluster_sizes[cid] += 1

            print(f"  K={K}: meanΔ={mean_d:+.4f}  W/L={wins:2d}/{losses:2d}  cata={cata}  p={p:.4f}")
            print(f"      routing: {dict(cluster_to_method)}")
            print(f"      sizes:   {dict(cluster_sizes)}")

            routing_results[f'{sig_name}_K{K}'] = {
                'mean_delta': mean_d, 'wins': wins, 'losses': losses, 'cata': cata, 'p_value': p,
                'cluster_to_method': cluster_to_method,
                'cluster_sizes': dict(cluster_sizes),
                'cluster_ids': cluster_ids,
            }

    # Stage 4: Best routing analysis
    best_routing_key = max(routing_results, key=lambda k: routing_results[k]['mean_delta'])
    print(f"\n=== BEST ROUTING: {best_routing_key} ===")
    print(f"  mean Δ: {routing_results[best_routing_key]['mean_delta']:+.4f}")
    print(f"  W/L:    {routing_results[best_routing_key]['wins']}/{routing_results[best_routing_key]['losses']}")
    print(f"  cata:   {routing_results[best_routing_key]['cata']}")
    print(f"  p:      {routing_results[best_routing_key]['p_value']:.4f}")

    # Per-group
    cluster_ids = routing_results[best_routing_key]['cluster_ids']
    cluster_to_method = routing_results[best_routing_key]['cluster_to_method']
    deltas_dict = {}
    for alias, cid in cluster_ids.items():
        method = cluster_to_method[cid]
        delta = all_method_scores[alias][method] - all_method_scores[alias]['baseline_gauss10']
        deltas_dict[alias] = delta

    summary = per_group_summary(deltas_dict, get_per_group)
    print(f"\n  Per-group:")
    for g, s in summary.items():
        print(f"    {g:<12s} n={s['n']:2d}  meanΔ={s['mean_delta']:+.4f}  "
              f"W/L={s['wins']}/{s['losses']}  cata={s['catastrophic']}")

    # Method usage
    method_usage = defaultdict(int)
    for alias, cid in cluster_ids.items():
        method_usage[cluster_to_method[cid]] += 1
    print(f"\n=== Method usage ===")
    for method, count in sorted(method_usage.items(), key=lambda x: -x[1]):
        print(f"    {method:<28s}: {count} datasets")

    # Save
    out = Path(__file__).parent.parent / 'results' / 'P8_tri_routing_v2.json'
    out.parent.mkdir(exist_ok=True)
    save_data = {
        'best_key': best_routing_key,
        'routing_results': {k: {**v, 'cluster_ids': v['cluster_ids']}
                             for k, v in routing_results.items()},
        'method_summaries': method_summaries,
        'top_15_single_methods': [(m[0], m[1]) for m in sorted_methods[:15]],
    }
    with open(out, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
