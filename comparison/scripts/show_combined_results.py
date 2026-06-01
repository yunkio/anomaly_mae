"""
Generate combined results table for simulation and simulation_normalonly experiments.
"""

import json
from pathlib import Path

def load_results(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_metrics(model_data):
    """Extract key metrics from model data."""
    pl = model_data.get('point_level', {})
    f1t = model_data.get('f1_t', {})
    pak = model_data.get('pa_k', {})

    return {
        'roc': pl.get('roc_auc', 0),
        'prc': pl.get('prc_auc', 0),
        'f1': pl.get('f1_score', 0),
        'precision': pl.get('precision', 0),
        'recall': pl.get('recall', 0),
        'f1_t': f1t.get('f1_t', 0),
        'pa20_roc': pak.get('pa_20', {}).get('roc_auc', 0),
        'pa20_f1': pak.get('pa_20', {}).get('f1_score', 0),
    }

def main():
    simulation_path = Path("/home/ykio/notebooks/claude/comparison/results/simulation/results.json")
    normalonly_path = Path("/home/ykio/notebooks/claude/comparison/results/simulation_normalonly/results.json")

    sim_data = load_results(simulation_path)
    norm_data = load_results(normalonly_path)

    # Define model order
    models = [
        'mae_adaptive',
        'mae_teacheronly',
        'random',
        'sensor_range',
        'pca_error',
        'l2_norm',
        'nn_distance',
        'mlp',
        'mlpmixer',
        'transformer',
        'gcn_lstm',
        'anomaly_transformer',
    ]

    # Print header
    print("=" * 140)
    print("Simulation vs Normal-Only Training Results Comparison")
    print("=" * 140)
    print()
    print(f"{'Model':<22} {'Training':<14} {'ROC':>7} {'PRC':>7} {'F1':>7} {'Prec':>7} {'Rec':>7} {'F1_T':>7} {'PA20_ROC':>9} {'PA20_F1':>8}")
    print("-" * 140)

    # Print mae_adaptive only once (same for both experiments)
    mae_adaptive = extract_metrics(sim_data['models']['mae_adaptive'])
    print(f"{'mae_adaptive':<22} {'(baseline)':<14} {mae_adaptive['roc']:>7.3f} {mae_adaptive['prc']:>7.3f} {mae_adaptive['f1']:>7.3f} {mae_adaptive['precision']:>7.3f} {mae_adaptive['recall']:>7.3f} {mae_adaptive['f1_t']:>7.3f} {mae_adaptive['pa20_roc']:>9.3f} {mae_adaptive['pa20_f1']:>8.3f}")

    # Print mae_teacheronly only once (same for both - doesn't use training)
    mae_teacher = extract_metrics(sim_data['models']['mae_teacheronly'])
    print(f"{'mae_teacheronly':<22} {'(baseline)':<14} {mae_teacher['roc']:>7.3f} {mae_teacher['prc']:>7.3f} {mae_teacher['f1']:>7.3f} {mae_teacher['precision']:>7.3f} {mae_teacher['recall']:>7.3f} {mae_teacher['f1_t']:>7.3f} {mae_teacher['pa20_roc']:>9.3f} {mae_teacher['pa20_f1']:>8.3f}")

    print("-" * 140)

    # Print other models with both experiments
    for model in models[2:]:  # Skip mae_adaptive and mae_teacheronly (already printed)
        # Simulation (full training)
        if model in sim_data['models']:
            m = extract_metrics(sim_data['models'][model])
            print(f"{model:<22} {'full':<14} {m['roc']:>7.3f} {m['prc']:>7.3f} {m['f1']:>7.3f} {m['precision']:>7.3f} {m['recall']:>7.3f} {m['f1_t']:>7.3f} {m['pa20_roc']:>9.3f} {m['pa20_f1']:>8.3f}")

        # Normal-only training
        if model in norm_data['models']:
            m = extract_metrics(norm_data['models'][model])
            print(f"{'':<22} {'normal-only':<14} {m['roc']:>7.3f} {m['prc']:>7.3f} {m['f1']:>7.3f} {m['precision']:>7.3f} {m['recall']:>7.3f} {m['f1_t']:>7.3f} {m['pa20_roc']:>9.3f} {m['pa20_f1']:>8.3f}")

        print("-" * 140)

    # Key findings
    print()
    print("=" * 140)
    print("Key Findings:")
    print("-" * 140)

    # Compare improvement
    improvements = []
    for model in models[2:]:
        if model in sim_data['models'] and model in norm_data['models']:
            if model in ['random', 'l2_norm']:  # These don't use training data
                continue
            sim_roc = sim_data['models'][model]['point_level']['roc_auc']
            norm_roc = norm_data['models'][model]['point_level']['roc_auc']
            delta = norm_roc - sim_roc
            improvements.append((model, delta, sim_roc, norm_roc))

    improvements.sort(key=lambda x: x[1], reverse=True)

    print("Models improved by normal-only training (ROC-AUC):")
    for model, delta, sim_roc, norm_roc in improvements:
        if delta > 0.01:
            print(f"  {model:<22}: {sim_roc:.3f} → {norm_roc:.3f} (+{delta:.3f})")

    print()
    print("Models degraded by normal-only training:")
    for model, delta, sim_roc, norm_roc in improvements:
        if delta < -0.01:
            print(f"  {model:<22}: {sim_roc:.3f} → {norm_roc:.3f} ({delta:.3f})")


if __name__ == "__main__":
    main()
