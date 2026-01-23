"""
TEMP Ablation Study: Architecture and Loss Parameter Experiments

This script runs 42 experiments (21 configurations x 2 mask_after_encoder settings)
to analyze the impact of various architecture and loss parameters.

Base configuration:
- force_mask_anomaly: True
- margin_type: dynamic
- masking_ratio: 0.5
- masking_strategy: patch
- num_patches: 10 (patch_size=10)
- patch_level_loss: True
- patchify_mode: patch_cnn
- shared_mask_token: False
- d_model: 32
- encoder: 2 layers
- teacher decoder: 4 layers
- student decoder: 1 layer
- epochs: 10
- normal_complexity: False
"""

import sys
sys.path.insert(0, '/home/ykio/notebooks/claude')

import os
import json
import time
import subprocess
from datetime import datetime
from copy import deepcopy
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mae_anomaly import (
    Config, set_seed,
    SlidingWindowTimeSeriesGenerator, SlidingWindowDataset,
    NormalDataComplexity,
    SelfDistilledMAEMultivariate, SelfDistillationLoss,
    Trainer, Evaluator, SLIDING_ANOMALY_TYPE_NAMES
)


# =============================================================================
# Experiment Configurations
# =============================================================================

def get_base_config() -> Dict:
    """Base configuration"""
    return {
        'force_mask_anomaly': True,
        'margin_type': 'dynamic',
        'mask_after_encoder': False,  # Will be overridden per experiment round
        'masking_ratio': 0.5,
        'masking_strategy': 'patch',
        'seq_length': 100,  # window size
        'num_patches': 10,
        'patch_size': 10,
        'patch_level_loss': True,
        'patchify_mode': 'patch_cnn',
        'shared_mask_token': False,
        'd_model': 32,
        'nhead': 4,
        'num_encoder_layers': 2,
        'num_teacher_decoder_layers': 4,
        'num_student_decoder_layers': 1,
        'num_shared_decoder_layers': 0,
        'dim_feedforward': 128,  # 4 * d_model
        'cnn_channels': None,  # Default: (d_model//2, d_model) = (16, 32)
        'use_student_reconstruction_loss': False,
        'anomaly_loss_weight': 1.0,
        'num_epochs': 10,
        'mask_last_n': 10,  # patch_size
    }


def get_experiment_configs() -> List[Dict]:
    """Define 21 experiment configurations"""
    base = get_base_config()
    experiments = []

    # 1. Default (base)
    exp1 = deepcopy(base)
    exp1['name'] = '01_default'
    experiments.append(exp1)

    # 2. Shared decoder: 1 shared + 3 teacher + 1 student
    exp2 = deepcopy(base)
    exp2['name'] = '02_shared_decoder'
    exp2['num_shared_decoder_layers'] = 1
    exp2['num_teacher_decoder_layers'] = 3
    exp2['num_student_decoder_layers'] = 1
    experiments.append(exp2)

    # 3. Window size 200 (patch_size=10, num_patches=20)
    exp3 = deepcopy(base)
    exp3['name'] = '03_window_200'
    exp3['seq_length'] = 200
    exp3['num_patches'] = 20
    exp3['patch_size'] = 10
    exp3['mask_last_n'] = 20  # 2 patches
    experiments.append(exp3)

    # 4. Window size 500 (patch_size=10, num_patches=50)
    exp4 = deepcopy(base)
    exp4['name'] = '04_window_500'
    exp4['seq_length'] = 500
    exp4['num_patches'] = 50
    exp4['patch_size'] = 10
    exp4['mask_last_n'] = 50  # 5 patches
    experiments.append(exp4)

    # 5. Encoder 1 layer
    exp5 = deepcopy(base)
    exp5['name'] = '05_encoder_1'
    exp5['num_encoder_layers'] = 1
    experiments.append(exp5)

    # 6. Encoder 3 layers
    exp6 = deepcopy(base)
    exp6['name'] = '06_encoder_3'
    exp6['num_encoder_layers'] = 3
    experiments.append(exp6)

    # 7. Decoder t4s2
    exp7 = deepcopy(base)
    exp7['name'] = '07_decoder_t4s2'
    exp7['num_teacher_decoder_layers'] = 4
    exp7['num_student_decoder_layers'] = 2
    experiments.append(exp7)

    # 8. Decoder t3s2
    exp8 = deepcopy(base)
    exp8['name'] = '08_decoder_t3s2'
    exp8['num_teacher_decoder_layers'] = 3
    exp8['num_student_decoder_layers'] = 2
    experiments.append(exp8)

    # 9. Decoder t3s1
    exp9 = deepcopy(base)
    exp9['name'] = '09_decoder_t3s1'
    exp9['num_teacher_decoder_layers'] = 3
    exp9['num_student_decoder_layers'] = 1
    experiments.append(exp9)

    # 10. Decoder t2s1
    exp10 = deepcopy(base)
    exp10['name'] = '10_decoder_t2s1'
    exp10['num_teacher_decoder_layers'] = 2
    exp10['num_student_decoder_layers'] = 1
    experiments.append(exp10)

    # 11. d_model 16
    exp11 = deepcopy(base)
    exp11['name'] = '11_d_model_16'
    exp11['d_model'] = 16
    exp11['dim_feedforward'] = 64
    exp11['cnn_channels'] = (8, 16)
    experiments.append(exp11)

    # 12. d_model 8
    exp12 = deepcopy(base)
    exp12['name'] = '12_d_model_8'
    exp12['d_model'] = 8
    exp12['nhead'] = 2
    exp12['dim_feedforward'] = 32
    exp12['cnn_channels'] = (4, 8)
    experiments.append(exp12)

    # 13. d_model 64
    exp13 = deepcopy(base)
    exp13['name'] = '13_d_model_64'
    exp13['d_model'] = 64
    exp13['dim_feedforward'] = 256
    exp13['cnn_channels'] = (32, 64)
    experiments.append(exp13)

    # 14. d_model 4
    exp14 = deepcopy(base)
    exp14['name'] = '14_d_model_4'
    exp14['d_model'] = 4
    exp14['nhead'] = 2
    exp14['dim_feedforward'] = 16
    exp14['cnn_channels'] = (4, 4)
    experiments.append(exp14)

    # 15. CNN small (much smaller than default)
    exp15 = deepcopy(base)
    exp15['name'] = '15_cnn_small'
    exp15['cnn_channels'] = (4, 8)  # Default is (16, 32)
    experiments.append(exp15)

    # 16. CNN large (much larger than default)
    exp16 = deepcopy(base)
    exp16['name'] = '16_cnn_large'
    exp16['cnn_channels'] = (64, 128)  # Default is (16, 32)
    experiments.append(exp16)

    # 17. patch_level_loss = False (window level loss)
    exp17 = deepcopy(base)
    exp17['name'] = '17_window_level_loss'
    exp17['patch_level_loss'] = False
    experiments.append(exp17)

    # 18. Student reconstruction loss
    exp18 = deepcopy(base)
    exp18['name'] = '18_student_recon_loss'
    exp18['use_student_reconstruction_loss'] = True
    experiments.append(exp18)

    # 19. Anomaly loss weight 2x
    exp19 = deepcopy(base)
    exp19['name'] = '19_anomaly_weight_2x'
    exp19['anomaly_loss_weight'] = 2.0
    experiments.append(exp19)

    # 20. Anomaly loss weight 3x
    exp20 = deepcopy(base)
    exp20['name'] = '20_anomaly_weight_3x'
    exp20['anomaly_loss_weight'] = 3.0
    experiments.append(exp20)

    # 21. Anomaly loss weight 5x
    exp21 = deepcopy(base)
    exp21['name'] = '21_anomaly_weight_5x'
    exp21['anomaly_loss_weight'] = 5.0
    experiments.append(exp21)

    return experiments


# =============================================================================
# Single Experiment Runner
# =============================================================================

class SingleExperimentRunner:
    """Run a single experiment configuration"""

    def __init__(
        self,
        exp_config: Dict,
        output_dir: str,
        signals: np.ndarray,
        point_labels: np.ndarray,
        anomaly_regions: list,
        train_ratio: float = 0.5
    ):
        self.exp_config = exp_config
        self.output_dir = output_dir
        self.signals = signals
        self.point_labels = point_labels
        self.anomaly_regions = anomaly_regions
        self.train_ratio = train_ratio

        os.makedirs(output_dir, exist_ok=True)

    def _create_config(self) -> Config:
        """Create Config object from experiment configuration"""
        config = Config()

        # Apply experiment parameters
        for key, value in self.exp_config.items():
            if key == 'name':
                continue
            if hasattr(config, key):
                setattr(config, key, value)

        return config

    def run(self) -> Dict:
        """Run the experiment and return metrics"""
        config = self._create_config()
        set_seed(config.random_seed)

        # Create datasets
        # Calculate target counts from ratios (Stage 2 settings)
        total_test = 2000
        target_counts = {
            'pure_normal': int(total_test * config.test_ratio_pure_normal),
            'disturbing_normal': int(total_test * config.test_ratio_disturbing_normal),
            'anomaly': int(total_test * config.test_ratio_anomaly)
        }

        train_dataset = SlidingWindowDataset(
            signals=self.signals,
            point_labels=self.point_labels,
            anomaly_regions=self.anomaly_regions,
            window_size=config.seq_length,
            stride=config.sliding_window_stride,
            mask_last_n=config.mask_last_n,
            split='train',
            train_ratio=self.train_ratio,
            seed=config.random_seed
        )

        test_dataset = SlidingWindowDataset(
            signals=self.signals,
            point_labels=self.point_labels,
            anomaly_regions=self.anomaly_regions,
            window_size=config.seq_length,
            stride=config.sliding_window_stride,
            mask_last_n=config.mask_last_n,
            split='test',
            train_ratio=self.train_ratio,
            target_counts=target_counts,
            seed=config.random_seed
        )

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

        # Create and train model
        model = SelfDistilledMAEMultivariate(config)
        trainer = Trainer(model, config, train_loader, test_loader, verbose=False)
        trainer.train()

        # Evaluate
        evaluator = Evaluator(model, config, test_loader)
        metrics = evaluator.evaluate()

        # Get reconstruction losses from training history
        history = trainer.history
        final_recon_loss = history['train_rec_loss'][-1] if history['train_rec_loss'] else 0.0

        # Add reconstruction loss to metrics
        metrics['final_reconstruction_loss'] = final_recon_loss

        # Save model and config
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': asdict(config),
            'metrics': metrics
        }, os.path.join(self.output_dir, 'best_model.pt'))

        # Save config
        config_dict = {k: v for k, v in self.exp_config.items()}
        with open(os.path.join(self.output_dir, 'best_config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)

        # Compute teacher/student reconstruction losses on test set
        teacher_recon_losses = []
        student_recon_losses = []
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                if len(batch) == 5:
                    sequences, last_patch_labels, point_labels_b, sample_types_b, anomaly_types_b = batch
                else:
                    sequences, last_patch_labels, point_labels_b = batch[:3]
                sequences = sequences.to(config.device)
                batch_size_b, seq_length, num_features = sequences.shape
                mask = torch.ones(batch_size_b, seq_length, device=config.device)
                mask[:, -config.mask_last_n:] = 0
                teacher_output, student_output, _ = model(sequences, masking_ratio=0.0, mask=mask)
                masked_pos = (mask == 0)
                # Teacher reconstruction loss
                t_recon = ((teacher_output - sequences) ** 2).mean(dim=2)
                t_loss = (t_recon * masked_pos).sum(dim=1) / (masked_pos.sum(dim=1) + 1e-8)
                teacher_recon_losses.append(t_loss.cpu().numpy())
                # Student reconstruction loss
                s_recon = ((student_output - sequences) ** 2).mean(dim=2)
                s_loss = (s_recon * masked_pos).sum(dim=1) / (masked_pos.sum(dim=1) + 1e-8)
                student_recon_losses.append(s_loss.cpu().numpy())

        teacher_recon_loss = np.concatenate(teacher_recon_losses).mean()
        student_recon_loss = np.concatenate(student_recon_losses).mean()
        metrics['teacher_recon_loss'] = float(teacher_recon_loss)
        metrics['student_recon_loss'] = float(student_recon_loss)

        # Save detailed results
        detailed_losses = evaluator.compute_detailed_losses()
        detailed_df = pd.DataFrame({
            'reconstruction_loss': detailed_losses['reconstruction_loss'],
            'discrepancy_loss': detailed_losses['discrepancy_loss'],
            'total_loss': detailed_losses['total_loss'],
            'label': detailed_losses['labels'],
            'sample_type': detailed_losses['sample_types'],
            'anomaly_type': detailed_losses['anomaly_types'],
            'anomaly_type_name': [SLIDING_ANOMALY_TYPE_NAMES[int(at)] for at in detailed_losses['anomaly_types']]
        })
        detailed_df.to_csv(os.path.join(self.output_dir, 'best_model_detailed.csv'), index=False)

        # Get anomaly type metrics
        anomaly_type_metrics = evaluator.get_performance_by_anomaly_type()
        with open(os.path.join(self.output_dir, 'anomaly_type_metrics.json'), 'w') as f:
            json.dump(anomaly_type_metrics, f, indent=2)

        # Save training history
        histories_serializable = {
            k: [float(v) if isinstance(v, (int, float, np.floating)) else v for v in vals]
            for k, vals in history.items()
        }
        with open(os.path.join(self.output_dir, 'training_histories.json'), 'w') as f:
            json.dump({'0': histories_serializable}, f, indent=2)

        # Save metadata
        metadata = {
            'experiment_name': self.exp_config['name'],
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        with open(os.path.join(self.output_dir, 'experiment_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        # Cleanup
        del model, trainer, evaluator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return metrics


# =============================================================================
# Main Experiment Runner
# =============================================================================

def run_all_experiments(output_base_dir: str = 'results/experiments/temp'):
    """Run all 42 experiments (21 configs x 2 mask_after_encoder settings)"""
    import warnings
    warnings.filterwarnings('ignore')

    print("\n" + "="*80, flush=True)
    print(" " * 15 + "TEMP ABLATION STUDY EXPERIMENTS", flush=True)
    print("="*80, flush=True)

    # Create output directory
    os.makedirs(output_base_dir, exist_ok=True)

    # Generate datasets for different window sizes
    print("\nGenerating datasets (normal_complexity=False)...", flush=True)

    base_config = Config()
    complexity = NormalDataComplexity(enable_complexity=False)

    generator = SlidingWindowTimeSeriesGenerator(
        total_length=base_config.sliding_window_total_length,
        num_features=base_config.num_features,
        interval_scale=base_config.anomaly_interval_scale,
        complexity=complexity,
        seed=base_config.random_seed
    )
    signals, point_labels, anomaly_regions = generator.generate()
    print(f"Dataset: {len(signals):,} timesteps, {len(anomaly_regions)} anomaly regions", flush=True)

    # Get experiment configurations
    experiments = get_experiment_configs()
    total_experiments = len(experiments) * 2
    print(f"\nTotal experiments: {total_experiments} (21 configs x 2 mask_after_encoder settings)", flush=True)

    # Results storage
    all_results = []
    experiment_count = 0

    # Run experiments for both mask_after_encoder settings
    for mask_after_encoder in [False, True]:
        suffix = 'mask_before' if not mask_after_encoder else 'mask_after'
        print(f"\n{'='*80}", flush=True)
        print(f" ROUND: mask_after_encoder = {mask_after_encoder} ({suffix})", flush=True)
        print(f"{'='*80}", flush=True)

        for i, exp_config in enumerate(experiments):
            exp_config = deepcopy(exp_config)
            exp_config['mask_after_encoder'] = mask_after_encoder
            experiment_count += 1

            exp_name = f"{exp_config['name']}_{suffix}"
            exp_dir = os.path.join(output_base_dir, exp_name)

            print(f"\n[{experiment_count}/{total_experiments}] Running: {exp_name}", flush=True)
            print(f"  Output: {exp_dir}", flush=True)

            start_time = time.time()

            runner = SingleExperimentRunner(
                exp_config=exp_config,
                output_dir=exp_dir,
                signals=signals,
                point_labels=point_labels,
                anomaly_regions=anomaly_regions,
                train_ratio=0.5  # Stage 2 setting
            )

            metrics = runner.run()
            elapsed = time.time() - start_time

            print(f"  ROC-AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1_score']:.4f}", flush=True)
            print(f"  Teacher Recon: {metrics['teacher_recon_loss']:.4f}, Student Recon: {metrics['student_recon_loss']:.4f}", flush=True)
            if 'disturbing_roc_auc' in metrics and metrics['disturbing_roc_auc'] is not None:
                print(f"  Disturbing ROC-AUC: {metrics['disturbing_roc_auc']:.4f}", flush=True)
            print(f"  Time: {elapsed:.1f}s", flush=True)

            # Run visualization
            print(f"  Running visualization...", flush=True)
            try:
                viz_cmd = f"cd /home/ykio/notebooks/claude && /home/ykio/anaconda3/envs/dc_vis/bin/python scripts/visualize_all.py --experiment-dir {exp_dir} --skip-data --skip-architecture"
                subprocess.run(viz_cmd, shell=True, capture_output=True, timeout=180)
                print(f"  Visualization complete", flush=True)
            except Exception as e:
                print(f"  Visualization failed: {e}", flush=True)

            # Store result
            result = {
                'experiment': exp_name,
                'mask_after_encoder': mask_after_encoder,
                **{k: v for k, v in exp_config.items() if k != 'name'},
                **metrics
            }
            all_results.append(result)

            # Save intermediate results
            summary_df = pd.DataFrame(all_results)
            summary_path = os.path.join(output_base_dir, 'summary_results.csv')
            summary_df.to_csv(summary_path, index=False)

    # Print final summary table
    print(f"\n{'='*80}", flush=True)
    print(" " * 20 + "RESULTS SUMMARY", flush=True)
    print(f"{'='*80}", flush=True)

    summary_cols = ['experiment', 'roc_auc', 'f1_score', 'teacher_recon_loss', 'student_recon_loss']

    print("\n--- mask_after_encoder = False ---", flush=True)
    df_false = summary_df[summary_df['mask_after_encoder'] == False]
    print(df_false[summary_cols].to_string(index=False), flush=True)

    print("\n--- mask_after_encoder = True ---", flush=True)
    df_true = summary_df[summary_df['mask_after_encoder'] == True]
    print(df_true[summary_cols].to_string(index=False), flush=True)

    print(f"\n{'='*80}", flush=True)
    print(" " * 15 + "ALL EXPERIMENTS COMPLETE!", flush=True)
    print(f"{'='*80}", flush=True)

    return summary_df


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run TEMP ablation study experiments')
    parser.add_argument('--output-dir', type=str, default='results/experiments/temp',
                       help='Output directory for results')
    args = parser.parse_args()

    run_all_experiments(output_base_dir=args.output_dir)
