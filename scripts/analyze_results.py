"""
Analyze experiment results from the latest run
"""

import json
import os
from pathlib import Path

# Find the latest experiment folder
results_dir = Path("experiment_results")
latest_folder = max(results_dir.glob("*"), key=os.path.getmtime)
json_file = latest_folder / "experiment_results.json"

print("="*80)
print(f"분석 대상: {latest_folder.name}")
print("="*80)

# Load results
with open(json_file, 'r') as f:
    results = json.load(f)

print(f"\n총 실험 개수: {len(results)}")
print(f"저장된 파일:")
for file in sorted(latest_folder.glob("*.png")):
    print(f"  - {file.name}")

print("\n" + "="*80)
print("실험 결과 요약")
print("="*80)

# Group experiments
hyperparameter_exp = []
ablation_exp = []
masking_exp = []

for exp in results:
    name = exp['experiment_name']
    if name.startswith('Ablation'):
        ablation_exp.append(exp)
    elif name.startswith('Masking'):
        masking_exp.append(exp)
    else:
        hyperparameter_exp.append(exp)

print(f"\n1. Hyperparameter Tuning: {len(hyperparameter_exp)}개 실험")
print(f"2. Ablation Studies: {len(ablation_exp)}개 실험")
print(f"3. Masking Strategies: {len(masking_exp)}개 실험")

# Find best configurations
print("\n" + "="*80)
print("최고 성능 실험")
print("="*80)

best_combined = max(results, key=lambda x: x['metrics']['combined']['f1_score'])
best_sequence = max(results, key=lambda x: x['metrics']['sequence']['f1_score'])
best_point = max(results, key=lambda x: x['metrics']['point']['f1_score'])

print(f"\n🏆 Best Combined F1-Score: {best_combined['experiment_name']}")
print(f"   Combined F1: {best_combined['metrics']['combined']['f1_score']:.4f}")
print(f"   - Sequence F1: {best_combined['metrics']['sequence']['f1_score']:.4f}")
print(f"   - Point F1: {best_combined['metrics']['point']['f1_score']:.4f}")
print(f"   - ROC-AUC: {best_combined['metrics']['combined']['roc_auc']:.4f}")

print(f"\n🎯 Best Sequence-Level F1: {best_sequence['experiment_name']}")
print(f"   Sequence F1: {best_sequence['metrics']['sequence']['f1_score']:.4f}")
print(f"   ROC-AUC: {best_sequence['metrics']['sequence']['roc_auc']:.4f}")

print(f"\n📍 Best Point-Level F1: {best_point['experiment_name']}")
print(f"   Point F1: {best_point['metrics']['point']['f1_score']:.4f}")
print(f"   ROC-AUC: {best_point['metrics']['point']['roc_auc']:.4f}")

# Detailed results table
print("\n" + "="*80)
print("전체 실험 결과 (Combined Metrics 기준)")
print("="*80)

# Sort by combined F1
sorted_results = sorted(results, key=lambda x: x['metrics']['combined']['f1_score'], reverse=True)

print(f"\n{'Experiment':<30} {'ROC-AUC':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
print("-"*80)

for exp in sorted_results:
    name = exp['experiment_name'][:29]
    metrics = exp['metrics']['combined']
    print(f"{name:<30} {metrics['roc_auc']:>10.4f} {metrics['precision']:>10.4f} "
          f"{metrics['recall']:>10.4f} {metrics['f1_score']:>10.4f}")

# Ablation study analysis
if ablation_exp:
    print("\n" + "="*80)
    print("Ablation Study 분석")
    print("="*80)

    baseline = next((exp for exp in results if exp['experiment_name'] == 'Baseline'), None)

    if baseline:
        baseline_f1 = baseline['metrics']['combined']['f1_score']
        print(f"\nBaseline F1-Score: {baseline_f1:.4f}")
        print(f"\nAblation 실험별 성능 변화:")
        print(f"{'Ablation':<30} {'F1-Score':>10} {'Change':>10}")
        print("-"*50)

        for exp in ablation_exp:
            f1 = exp['metrics']['combined']['f1_score']
            change = f1 - baseline_f1
            sign = "+" if change >= 0 else ""
            name = exp['experiment_name'].replace('Ablation: ', '')
            print(f"{name:<30} {f1:>10.4f} {sign}{change:>9.4f}")

# Masking strategy comparison
if masking_exp:
    print("\n" + "="*80)
    print("Masking Strategy 비교")
    print("="*80)

    print(f"\n{'Strategy':<20} {'Combined F1':>12} {'Seq F1':>10} {'Point F1':>10}")
    print("-"*60)

    for exp in sorted(masking_exp, key=lambda x: x['metrics']['combined']['f1_score'], reverse=True):
        name = exp['experiment_name'].replace('Masking_', '')
        combined_f1 = exp['metrics']['combined']['f1_score']
        seq_f1 = exp['metrics']['sequence']['f1_score']
        point_f1 = exp['metrics']['point']['f1_score']
        print(f"{name:<20} {combined_f1:>12.4f} {seq_f1:>10.4f} {point_f1:>10.4f}")

# Hyperparameter analysis
if hyperparameter_exp:
    print("\n" + "="*80)
    print("Hyperparameter Tuning 분석")
    print("="*80)

    print(f"\n{'Configuration':<30} {'Combined F1':>12} {'ROC-AUC':>10}")
    print("-"*55)

    for exp in sorted(hyperparameter_exp, key=lambda x: x['metrics']['combined']['f1_score'], reverse=True):
        name = exp['experiment_name']
        f1 = exp['metrics']['combined']['f1_score']
        auc = exp['metrics']['combined']['roc_auc']
        print(f"{name:<30} {f1:>12.4f} {auc:>10.4f}")

print("\n" + "="*80)
print("분석 완료!")
print("="*80)
print(f"\n시각화 파일 위치: {latest_folder}")
print(f"JSON 결과 파일: {json_file}")
