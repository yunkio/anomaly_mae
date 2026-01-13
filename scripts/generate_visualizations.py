"""
Generate Visualizations from Experiment Results

This script reads the experiment results JSON and generates all visualizations
with proper matplotlib backend configuration.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150


def load_results(filepath='experiment_results/experiment_results_complete.json'):
    """Load experiment results from JSON"""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_hyperparameter_comparison(results, output_dir='experiment_results'):
    """Plot hyperparameter tuning results"""
    print("\nGenerating hyperparameter comparison...")

    hyperparameter_results = [r for r in results if not r['experiment_name'].startswith('Ablation')]

    metrics_to_plot = ['roc_auc', 'f1_score', 'precision', 'recall']
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics_to_plot):
        names = [r['experiment_name'] for r in hyperparameter_results]
        values = [r['metrics'][metric] for r in hyperparameter_results]

        colors = sns.color_palette("husl", len(names))
        axes[idx].barh(names, values, color=colors)
        axes[idx].set_xlabel(metric.upper().replace('_', ' '), fontsize=12)
        axes[idx].set_title(f'Hyperparameter Comparison: {metric.upper()}', fontsize=14, fontweight='bold')
        axes[idx].set_xlim([0, 1])
        axes[idx].grid(True, alpha=0.3, axis='x')

        # Add value labels
        for i, v in enumerate(values):
            axes[idx].text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'hyperparameter_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved {save_path}")


def plot_ablation_comparison(results, output_dir='experiment_results'):
    """Plot ablation study results"""
    print("\nGenerating ablation study comparison...")

    ablation_results = [r for r in results if r['experiment_name'].startswith('Ablation') or r['experiment_name'] == 'Baseline']

    metrics_to_plot = ['roc_auc', 'f1_score', 'precision', 'recall']
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics_to_plot):
        names = [r['experiment_name'] for r in ablation_results]
        values = [r['metrics'][metric] for r in ablation_results]

        colors = ['#2ecc71' if 'Baseline' in n else '#e74c3c' for n in names]
        axes[idx].barh(names, values, color=colors, edgecolor='black', linewidth=1.5)
        axes[idx].set_xlabel(metric.upper().replace('_', ' '), fontsize=12)
        axes[idx].set_title(f'Ablation Study: {metric.upper()}', fontsize=14, fontweight='bold')
        axes[idx].set_xlim([0, 1])
        axes[idx].grid(True, alpha=0.3, axis='x')

        # Add value labels
        for i, v in enumerate(values):
            axes[idx].text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'ablation_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved {save_path}")


def plot_training_curves(results, output_dir='experiment_results'):
    """Plot training curves for all experiments"""
    print("\nGenerating training curves...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Use distinct colors for each experiment
    colors = sns.color_palette("tab10", len(results[:10]))

    for idx, result in enumerate(results[:10]):  # Plot first 10 to avoid clutter
        name = result['experiment_name']
        history = result['history']
        color = colors[idx]

        # Total loss
        axes[0].plot(history['epoch'], history['train_loss'], label=name, alpha=0.8, linewidth=2, color=color)

        # Reconstruction loss
        axes[1].plot(history['epoch'], history['train_rec_loss'], label=name, alpha=0.8, linewidth=2, color=color)

        # Discrepancy loss
        axes[2].plot(history['epoch'], history['train_disc_loss'], label=name, alpha=0.8, linewidth=2, color=color)

    # Configure axes
    for ax, title in zip(axes, ['Total Loss', 'Reconstruction Loss', 'Discrepancy Loss']):
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved {save_path}")


def plot_performance_heatmap(results, output_dir='experiment_results'):
    """Plot performance heatmap"""
    print("\nGenerating performance heatmap...")

    # Create dataframe
    data = []
    for result in results:
        row = {
            'Experiment': result['experiment_name'][:25],  # Truncate name
            'ROC-AUC': result['metrics']['roc_auc'],
            'F1': result['metrics']['f1_score'],
            'Precision': result['metrics']['precision'],
            'Recall': result['metrics']['recall']
        }
        data.append(row)

    df = pd.DataFrame(data)
    df = df.set_index('Experiment')

    plt.figure(figsize=(10, max(8, len(data) * 0.5)))
    sns.heatmap(df, annot=True, fmt='.3f', cmap='YlGnBu',
                cbar_kws={'label': 'Score'}, linewidths=0.5,
                vmin=0, vmax=1, annot_kws={'fontsize': 9})
    plt.title('Performance Heatmap Across All Experiments', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Experiments', fontsize=12)
    plt.tight_layout()

    save_path = os.path.join(output_dir, 'performance_heatmap.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved {save_path}")


def plot_metric_comparison_bar(results, output_dir='experiment_results'):
    """Plot side-by-side comparison of all metrics"""
    print("\nGenerating comprehensive metric comparison...")

    # Separate hyperparameter and ablation experiments
    hyper_results = [r for r in results if not r['experiment_name'].startswith('Ablation')]
    ablation_results = [r for r in results if r['experiment_name'].startswith('Ablation') or r['experiment_name'] == 'Baseline']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Hyperparameter experiments
    x_labels_hyper = [r['experiment_name'] for r in hyper_results]
    roc_hyper = [r['metrics']['roc_auc'] for r in hyper_results]
    f1_hyper = [r['metrics']['f1_score'] for r in hyper_results]

    x_pos = np.arange(len(x_labels_hyper))
    width = 0.35

    ax1.bar(x_pos - width/2, roc_hyper, width, label='ROC-AUC', color='#3498db', edgecolor='black')
    ax1.bar(x_pos + width/2, f1_hyper, width, label='F1-Score', color='#e74c3c', edgecolor='black')
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Hyperparameter Tuning: ROC-AUC vs F1-Score', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels_hyper, rotation=45, ha='right', fontsize=10)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 1])

    # Add value labels on bars
    for i, (r, f) in enumerate(zip(roc_hyper, f1_hyper)):
        ax1.text(i - width/2, r + 0.02, f'{r:.3f}', ha='center', va='bottom', fontsize=8)
        ax1.text(i + width/2, f + 0.02, f'{f:.3f}', ha='center', va='bottom', fontsize=8)

    # Ablation experiments
    x_labels_abl = [r['experiment_name'] for r in ablation_results]
    roc_abl = [r['metrics']['roc_auc'] for r in ablation_results]
    f1_abl = [r['metrics']['f1_score'] for r in ablation_results]

    x_pos_abl = np.arange(len(x_labels_abl))

    ax2.bar(x_pos_abl - width/2, roc_abl, width, label='ROC-AUC', color='#2ecc71', edgecolor='black')
    ax2.bar(x_pos_abl + width/2, f1_abl, width, label='F1-Score', color='#f39c12', edgecolor='black')
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Ablation Study: ROC-AUC vs F1-Score', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos_abl)
    ax2.set_xticklabels(x_labels_abl, rotation=45, ha='right', fontsize=10)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 1])

    # Add value labels on bars
    for i, (r, f) in enumerate(zip(roc_abl, f1_abl)):
        ax2.text(i - width/2, r + 0.02, f'{r:.3f}', ha='center', va='bottom', fontsize=8)
        ax2.text(i + width/2, f + 0.02, f'{f:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'comprehensive_metric_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved {save_path}")


def main():
    """Generate all visualizations"""
    print("="*80)
    print("GENERATING VISUALIZATIONS FROM EXPERIMENT RESULTS")
    print("="*80)

    # Load results
    results_file = 'experiment_results/experiment_results_complete.json'
    if not os.path.exists(results_file):
        print(f"\n❌ Error: Results file not found: {results_file}")
        print("Please run experiments first: python multivariate_mae_experiments.py")
        return

    results = load_results(results_file)
    print(f"\n✓ Loaded {len(results)} experiment results")

    output_dir = 'experiment_results'
    os.makedirs(output_dir, exist_ok=True)

    # Generate all visualizations
    try:
        plot_hyperparameter_comparison(results, output_dir)
        plot_ablation_comparison(results, output_dir)
        plot_training_curves(results, output_dir)
        plot_performance_heatmap(results, output_dir)
        plot_metric_comparison_bar(results, output_dir)

        print("\n" + "="*80)
        print("✓ ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
        print("="*80)
        print(f"\nGenerated files in '{output_dir}':")
        print("  1. hyperparameter_comparison.png")
        print("  2. ablation_comparison.png")
        print("  3. training_curves.png")
        print("  4. performance_heatmap.png")
        print("  5. comprehensive_metric_comparison.png")
        print("="*80)

    except Exception as e:
        print(f"\n❌ Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
