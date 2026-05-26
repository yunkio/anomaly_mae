"""
P11 — Per-Cluster Continuous σ Predictor

기존 cluster routing (P1, P8) = discrete (cluster → method category).
본 P11 = hybrid: cluster routing + within-cluster continuous σ regression.

Strategy:
1. K=4 cluster routing → cluster assignment (with supervised signature)
2. Per-cluster, train log(σ) ~ features regression
3. LOO: test dataset의 prediction은 cluster의 OTHER datasets로 학습
4. Within-cluster regression이 cluster-level method보다 better인지 검증

Expected outcome:
- Cluster routing alone: +0.0276 (P8 K=8)
- + within-cluster σ refinement: +0.028 ~ +0.032 가능
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
    DatasetSignature, extract_signature_unsupervised, extract_signature_supervised,
    run_kmeans_clustering,
)
from mae_anomaly.scripts.q3_exploration.core.postprocess import nlm_sigmoid_transform

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut


def find_oracle_sigma_with_nlm(ds, base_unsmoothed, sigma_candidates, T=1.5):
    best_sigma, best_pak = None, -np.inf
    for sigma in sigma_candidates:
        smoothed = gauss(base_unsmoothed, sigma)
        if T is not None:
            final = nlm_sigmoid_transform(smoothed, T_factor=T)
        else:
            final = smoothed
        pak = pak_auc_f1(final, ds.point_labels, ds.regions, ds.eval_mask)
        if pak > best_pak:
            best_pak, best_sigma = pak, sigma
    return best_sigma, best_pak


def per_cluster_loo_predict(X, y, cluster_ids, n_clusters,
                              model_factory=lambda: Ridge(alpha=1.0)):
    """LOO prediction with within-cluster regression.

    For each test idx:
    - Find its cluster
    - Train regression on OTHER datasets in same cluster
    - Predict
    - If cluster has only 1 dataset, fall back to global heuristic
    """
    n = len(X)
    preds = np.zeros(n)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cluster_to_indices = defaultdict(list)
    for i, c in enumerate(cluster_ids):
        cluster_to_indices[c].append(i)

    for test_idx in range(n):
        test_cluster = cluster_ids[test_idx]
        cluster_members = [i for i in cluster_to_indices[test_cluster] if i != test_idx]

        if len(cluster_members) < 3:
            # Fallback: heuristic mean of cluster's y values
            if cluster_members:
                preds[test_idx] = np.mean(y[cluster_members])
            else:
                preds[test_idx] = y.mean()
            continue

        # Train model on cluster members
        X_train = X_scaled[cluster_members]
        y_train = y[cluster_members]
        try:
            model = model_factory()
            model.fit(X_train, y_train)
            preds[test_idx] = model.predict(X_scaled[test_idx:test_idx+1])[0]
        except Exception:
            preds[test_idx] = np.mean(y[cluster_members])

    return preds


def main():
    targets = iter_dataset_aliases()
    print(f"P11 — Per-Cluster Continuous σ Predictor, {len(targets)} datasets")

    sigma_candidates = [0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300]

    # Stage 1: Data collection
    print("\n--- Stage 1: Extract features + oracle σ ---")
    aliases = []
    X_sup_full = []
    X_unsup = []
    y_oracle = []
    base_unsmoothed_dict = {}
    signatures = {}
    baseline_paks = []
    oracle_paks = []
    median_segs = []

    for i, (alias, swat) in enumerate(targets, 1):
        ds = DatasetScores.load(alias, swat)
        if ds is None:
            continue
        pt_r, pt_d, pt_s, pt_f = per_channel_points(ds, stride=21)
        base_unsmoothed = adaptive_combine(pt_r, pt_d, pt_f, use_fm=True)
        median_seg = median_anomaly_segment_length(ds.regions)

        base_smoothed_for_sig = gauss(base_unsmoothed, 10)
        baseline_pak = pak_auc_f1(base_smoothed_for_sig,
                                   ds.point_labels, ds.regions, ds.eval_mask)
        oracle_sigma, oracle_pak = find_oracle_sigma_with_nlm(
            ds, base_unsmoothed, sigma_candidates, T=1.5)

        unsup = extract_signature_unsupervised(base_smoothed_for_sig, pt_r, pt_d, pt_f)
        sup = extract_signature_supervised(ds.regions, ds.point_labels, baseline_pak)

        aliases.append(alias)
        sig = DatasetSignature(alias=alias, features={**unsup, **sup})
        signatures[alias] = sig

        # Feature vectors
        unsup_keys = sorted(unsup.keys())
        sup_keys = sorted(sup.keys())
        X_unsup.append([unsup[k] for k in unsup_keys])
        X_sup_full.append([unsup[k] for k in unsup_keys] + [sup[k] for k in sup_keys])
        y_oracle.append(np.log(oracle_sigma + 1e-9))
        baseline_paks.append(baseline_pak)
        oracle_paks.append(oracle_pak)
        median_segs.append(median_seg)
        base_unsmoothed_dict[alias] = (base_unsmoothed, median_seg, ds)

        if i % 10 == 0 or i == len(targets):
            print(f"  [{i:2d}/{len(targets)}] processed", flush=True)

    X_unsup = np.array(X_unsup)
    X_sup_full = np.array(X_sup_full)
    y = np.array(y_oracle)
    baseline_paks = np.array(baseline_paks)
    oracle_paks = np.array(oracle_paks)
    median_segs = np.array(median_segs)

    # Stage 2: Cluster assignment (using full supervised features)
    print("\n--- Stage 2: K=4 cluster routing baseline ---")
    feature_keys_sup = sorted(extract_signature_supervised([], np.zeros(10), 0.0).keys()) + \
                       sorted(extract_signature_unsupervised(np.array([0.1,0.2,0.3,0.4,0.5]),
                                                                np.zeros(5), np.zeros(5), np.zeros(5)).keys())

    cluster_ids_dict, _, _ = run_kmeans_clustering(signatures, feature_keys_sup, 4)
    cluster_ids = np.array([cluster_ids_dict[a] for a in aliases])

    cluster_sizes = defaultdict(int)
    for cid in cluster_ids:
        cluster_sizes[int(cid)] += 1
    print(f"Cluster sizes (K=4): {dict(cluster_sizes)}")

    # Stage 3: Baseline — global heuristic div=5
    print("\n--- Stage 3: Baselines ---")
    global_div5_preds = np.log(np.maximum(median_segs / 5, 0.5))
    sigmas_div5 = np.maximum(np.exp(global_div5_preds), 0.5)

    eval_paks_div5 = []
    for j, alias in enumerate(aliases):
        base_unsmoothed, _, ds = base_unsmoothed_dict[alias]
        smoothed = gauss(base_unsmoothed, sigmas_div5[j])
        final = nlm_sigmoid_transform(smoothed, T_factor=1.5)
        eval_paks_div5.append(pak_auc_f1(final, ds.point_labels, ds.regions, ds.eval_mask))
    eval_paks_div5 = np.array(eval_paks_div5)
    deltas_div5 = eval_paks_div5 - baseline_paks
    print(f"Global heuristic div=5 + NLM-T1.5:    meanΔ={deltas_div5.mean():+.4f}  "
          f"W/L={(deltas_div5 > 0).sum()}/{(deltas_div5 < 0).sum()}  "
          f"p={wilcoxon_test(eval_paks_div5.tolist(), baseline_paks.tolist(), alternative='greater'):.4f}")

    # Per-cluster σ predictions
    print("\n--- Stage 4: Per-cluster continuous σ predictor ---")

    predictor_strategies = {
        'per_cluster_ridge_a1': lambda: Ridge(alpha=1.0),
        'per_cluster_ridge_a10': lambda: Ridge(alpha=10.0),
        'per_cluster_rf_50': lambda: RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42),
        'per_cluster_rf_100': lambda: RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
    }

    for strategy_name, model_factory in predictor_strategies.items():
        preds = per_cluster_loo_predict(X_sup_full, y, cluster_ids, 4, model_factory)
        predicted_sigmas = np.maximum(np.exp(preds) - 1e-9, 0.5)

        eval_paks = []
        for j, alias in enumerate(aliases):
            base_unsmoothed, _, ds = base_unsmoothed_dict[alias]
            smoothed = gauss(base_unsmoothed, predicted_sigmas[j])
            final = nlm_sigmoid_transform(smoothed, T_factor=1.5)
            eval_paks.append(pak_auc_f1(final, ds.point_labels, ds.regions, ds.eval_mask))
        eval_paks = np.array(eval_paks)
        deltas = eval_paks - baseline_paks
        mean_d = float(deltas.mean())
        wins = int((deltas > 0).sum())
        losses = int((deltas < 0).sum())
        cata = int((deltas < -0.05).sum())
        p = wilcoxon_test(eval_paks.tolist(), baseline_paks.tolist(), alternative='greater')
        oracle_capture = mean_d / (oracle_paks - baseline_paks).mean() * 100

        print(f"{strategy_name:<28s}: meanΔ={mean_d:>+.4f}  W/L={wins}/{losses}  "
              f"cata={cata}  p={p:.4f}  capture={oracle_capture:.1f}%")

    # Stage 5: Per-cluster oracle (cluster mean of oracle σ) — upper bound
    print("\n--- Stage 5: Per-cluster oracle σ (upper bound) ---")
    per_cluster_oracle_sigma = {}
    for c in range(4):
        mask = cluster_ids == c
        if mask.sum() > 0:
            per_cluster_oracle_sigma[c] = float(np.median(np.exp(y[mask])))

    print(f"Per-cluster oracle σ (median): {per_cluster_oracle_sigma}")

    # Eval using per-cluster oracle σ
    eval_paks_orc = []
    for j, alias in enumerate(aliases):
        cid = int(cluster_ids[j])
        sigma = per_cluster_oracle_sigma.get(cid, 5.0)
        base_unsmoothed, _, ds = base_unsmoothed_dict[alias]
        smoothed = gauss(base_unsmoothed, sigma)
        final = nlm_sigmoid_transform(smoothed, T_factor=1.5)
        eval_paks_orc.append(pak_auc_f1(final, ds.point_labels, ds.regions, ds.eval_mask))
    eval_paks_orc = np.array(eval_paks_orc)
    deltas_orc = eval_paks_orc - baseline_paks
    print(f"Per-cluster oracle σ apply:           meanΔ={deltas_orc.mean():+.4f}  "
          f"W/L={(deltas_orc > 0).sum()}/{(deltas_orc < 0).sum()}  "
          f"p={wilcoxon_test(eval_paks_orc.tolist(), baseline_paks.tolist(), alternative='greater'):.4f}")

    # Stage 6: Hybrid — per-cluster σ as prior, global model as residual
    print("\n--- Stage 6: Hybrid (cluster prior + global residual) ---")
    per_cluster_means = {}
    for c in range(4):
        mask = cluster_ids == c
        if mask.sum() > 0:
            per_cluster_means[c] = float(y[mask].mean())

    # Compute residuals: y - per_cluster_mean
    cluster_mean_priors = np.array([per_cluster_means.get(int(c), y.mean()) for c in cluster_ids])
    residuals = y - cluster_mean_priors

    # LOO ridge on residuals using full features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sup_full)
    loo = LeaveOneOut()
    pred_residuals = np.zeros(len(aliases))
    for train_idx, test_idx in loo.split(X_scaled):
        m = Ridge(alpha=1.0)
        m.fit(X_scaled[train_idx], residuals[train_idx])
        pred_residuals[test_idx[0]] = m.predict(X_scaled[test_idx])[0]

    # Final predictions
    final_log_sigmas = cluster_mean_priors + pred_residuals
    predicted_sigmas = np.maximum(np.exp(final_log_sigmas) - 1e-9, 0.5)

    eval_paks_hybrid = []
    for j, alias in enumerate(aliases):
        base_unsmoothed, _, ds = base_unsmoothed_dict[alias]
        smoothed = gauss(base_unsmoothed, predicted_sigmas[j])
        final = nlm_sigmoid_transform(smoothed, T_factor=1.5)
        eval_paks_hybrid.append(pak_auc_f1(final, ds.point_labels, ds.regions, ds.eval_mask))
    eval_paks_hybrid = np.array(eval_paks_hybrid)
    deltas_hybrid = eval_paks_hybrid - baseline_paks
    print(f"Hybrid (cluster prior + ridge residual): meanΔ={deltas_hybrid.mean():+.4f}  "
          f"W/L={(deltas_hybrid > 0).sum()}/{(deltas_hybrid < 0).sum()}  "
          f"p={wilcoxon_test(eval_paks_hybrid.tolist(), baseline_paks.tolist(), alternative='greater'):.4f}")

    # Save
    out = Path(__file__).parent.parent / 'results' / 'P11_per_cluster_continuous_sigma.json'
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w') as f:
        json.dump({
            'cluster_assignments': cluster_ids_dict,
            'cluster_sizes': dict(cluster_sizes),
            'baseline_paks': baseline_paks.tolist(),
            'oracle_paks': oracle_paks.tolist(),
            'aliases': aliases,
        }, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
