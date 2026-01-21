"""
Comprehensive Experiments for Self-Distilled MAE on Multivariate Time Series

Features:
- Grid search over parameter combinations
- Two-stage search: Quick screening → Full training on top combinations
- Results saving (no visualization - use visualize_all.py for that)
- Best model saving and analysis
"""

import sys
sys.path.insert(0, '/home/ykio/notebooks/claude')

import os
import json
import itertools
from datetime import datetime
from copy import deepcopy
from dataclasses import asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import from mae_anomaly package
from mae_anomaly import (
    Config, set_seed, MultivariateTimeSeriesDataset,
    SelfDistilledMAEMultivariate, SelfDistillationLoss,
    Trainer, Evaluator
)
from mae_anomaly.dataset import ANOMALY_TYPE_NAMES


# =============================================================================
# Default Parameter Grid
# =============================================================================

DEFAULT_PARAM_GRID = {
    # Masking (high impact)
    'masking_ratio': [0.4, 0.7],
    'masking_strategy': ['patch', 'feature_wise'],  # Masking strategy
    'num_patches': [10, 25, 50],

    # Loss parameters
    'margin': [0.25, 0.5, 1.0],
    'lambda_disc': [0.3, 0.5, 0.7],

    # Discrepancy loss strategy
    'margin_type': ['hinge', 'softplus', 'dynamic'],

    # Anomaly masking strategy
    'force_mask_anomaly': [False, True],

    # Loss granularity
    'patch_level_loss': [True, False],

    # Patchify mode (CNN vs Linear embedding)
    'patchify_mode': ['cnn_first', 'patch_cnn', 'linear'],
}
# Total combinations: 2*2*3*3*3*3*2*2*3 = 3888


# =============================================================================
# Experiment Runner (Grid Search)
# =============================================================================

class ExperimentRunner:
    """
    Run comprehensive experiments with parameter grid search

    Two-stage approach:
    1. Quick search: Fast screening with reduced epochs/data
    2. Full search: Complete training on top combinations

    Results are saved to output_dir. Use visualize_all.py for visualization.
    """

    def __init__(
        self,
        param_grid: Dict[str, List] = None,
        base_config: Config = None,
        output_dir: str = 'results/experiments'
    ):
        self.param_grid = param_grid or DEFAULT_PARAM_GRID
        self.base_config = base_config or Config()

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = os.path.join(output_dir, timestamp)
        os.makedirs(self.output_dir, exist_ok=True)

        self.results = []
        self.full_results = []
        self.histories = {}
        self.best_model = None
        self.best_config = None
        self.best_metrics = None

        print(f"\nExperiment results will be saved to: {self.output_dir}")

    def generate_combinations(self) -> List[Dict]:
        """Generate all parameter combinations"""
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())

        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))

        return combinations

    def _create_config(self, params: Dict, quick: bool = False) -> Config:
        """Create config with given parameters"""
        config = deepcopy(self.base_config)

        for key, value in params.items():
            if hasattr(config, key):
                # Ensure integer types for parameters that require it
                int_params = [
                    'num_patches', 'num_epochs', 'batch_size', 'num_train_samples', 'num_test_samples',
                    'd_model', 'nhead', 'num_encoder_layers', 'num_teacher_decoder_layers',
                    'num_student_decoder_layers', 'dim_feedforward'
                ]
                if key in int_params:
                    value = int(value)
                setattr(config, key, value)

        if 'num_patches' in params:
            config.num_patches = int(config.num_patches)
            config.patch_size = int(config.seq_length // config.num_patches)
            config.mask_last_n = int(config.patch_size)

        if quick:
            config.num_epochs = 15
            config.num_train_samples = 500
            config.num_test_samples = 200

        return config

    def _run_single_experiment(
        self,
        config: Config,
        experiment_id: int,
        save_history: bool = False
    ) -> Dict:
        """Run a single experiment"""
        set_seed(config.random_seed)

        train_dataset = MultivariateTimeSeriesDataset(
            num_samples=config.num_train_samples,
            seq_length=config.seq_length,
            num_features=config.num_features,
            anomaly_ratio=config.train_anomaly_ratio,
            mask_last_n=config.mask_last_n,
            seed=config.random_seed
        )

        test_dataset = MultivariateTimeSeriesDataset(
            num_samples=config.num_test_samples,
            seq_length=config.seq_length,
            num_features=config.num_features,
            anomaly_ratio=config.test_anomaly_ratio,
            mask_last_n=config.mask_last_n,
            seed=config.random_seed + 1
        )

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

        model = SelfDistilledMAEMultivariate(config)
        trainer = Trainer(model, config, train_loader, test_loader, verbose=False)
        trainer.train()

        evaluator = Evaluator(model, config, test_loader)
        metrics = evaluator.evaluate()

        if save_history:
            self.histories[experiment_id] = trainer.history

        # Check if this is the best model
        if self.best_metrics is None or metrics['roc_auc'] > self.best_metrics['roc_auc']:
            self.best_model = model
            self.best_config = config
            self.best_metrics = metrics

        # Clean up
        del trainer
        if not save_history:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return metrics

    def run_grid_search(
        self,
        quick_epochs: int = 15,
        quick_train: int = 1000,
        quick_test: int = 400,
        full_epochs: int = 100,
        full_train: int = 2000,
        full_test: int = 500,
        top_k: int = 50,
        two_stage: bool = True
    ) -> pd.DataFrame:
        """
        Run grid search over parameter combinations

        Args:
            quick_epochs: Epochs for quick search
            quick_train: Training samples for quick search
            quick_test: Test samples for quick search
            full_epochs: Epochs for full search
            full_train: Training samples for full search
            full_test: Test samples for full search
            top_k: Number of top combinations for full search
            two_stage: If True, run quick search first then full search on top-k
        """
        combinations = self.generate_combinations()
        n_combinations = len(combinations)

        print("\n" + "="*80)
        print(" " * 20 + "GRID SEARCH EXPERIMENT")
        print("="*80)
        print(f"Parameter grid: {self.param_grid}")
        print(f"Total combinations: {n_combinations}")
        print(f"Two-stage search: {two_stage}")

        # Stage 1: Quick Search
        print("\n" + "-"*80)
        print("STAGE 1: Quick Search")
        print(f"Epochs: {quick_epochs}, Train: {quick_train}, Test: {quick_test}")
        print("-"*80)

        self.base_config.num_epochs = quick_epochs
        self.base_config.num_train_samples = quick_train
        self.base_config.num_test_samples = quick_test

        self.results = []

        for i, params in enumerate(tqdm(combinations, desc="Quick Search")):
            config = self._create_config(params, quick=True)
            metrics = self._run_single_experiment(config, experiment_id=i)

            result = {
                'combination_id': i,
                **params,
                **metrics
            }
            self.results.append(result)

            # Print every trial's performance
            params_str = ", ".join(f"{k}={v}" for k, v in params.items())
            tqdm.write(f"  [{i+1}/{n_combinations}] {params_str}")
            tqdm.write(f"    -> ROC-AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1_score']:.4f}")
            if 'disturbing_roc_auc' in metrics:
                tqdm.write(f"    -> Disturbing Normal - ROC-AUC: {metrics['disturbing_roc_auc']:.4f}, F1: {metrics['disturbing_f1']:.4f}")

            # Print best model every 20 trials and at the end
            if (i + 1) % 20 == 0 or (i + 1) == n_combinations:
                best_idx = max(range(len(self.results)), key=lambda x: self.results[x]['roc_auc'])
                best_r = self.results[best_idx]
                # Get best model's parameters
                best_params = {k: best_r[k] for k in self.param_grid.keys() if k in best_r}
                best_params_str = ", ".join(f"{k}={v}" for k, v in best_params.items())
                tqdm.write(f"  *** BEST so far (#{best_idx+1}): ROC-AUC={best_r['roc_auc']:.4f}, F1={best_r['f1_score']:.4f} ***")
                tqdm.write(f"      Params: {best_params_str}")

        quick_df = pd.DataFrame(self.results)
        quick_df = quick_df.sort_values('roc_auc', ascending=False).reset_index(drop=True)

        print(f"\nQuick Search Complete!")
        print(f"Best ROC-AUC: {quick_df['roc_auc'].max():.4f}")

        # Save quick search results
        quick_df.to_csv(os.path.join(self.output_dir, 'quick_search_results.csv'), index=False)

        # Stage 2: Full Search (if two_stage)
        if two_stage and top_k > 0:
            print("\n" + "-"*80)
            print(f"STAGE 2: Full Search (Diverse Selection of {top_k} combinations)")
            print(f"Epochs: {full_epochs}, Train: {full_train}, Test: {full_test}")
            print("-"*80)

            # Use diverse selection method
            print("\nSelecting diverse candidates for Stage 2:")
            top_combinations = self._select_stage2_candidates(quick_df, total_k=top_k)
            actual_k = len(top_combinations)
            param_keys = list(self.param_grid.keys())

            self.full_results = []

            for i, (_, data) in enumerate(tqdm(top_combinations.iterrows(), total=actual_k, desc="Full Search")):
                params = {k: data[k] for k in param_keys}

                config = self._create_config(params, quick=False)
                config.num_epochs = full_epochs
                config.num_train_samples = full_train
                config.num_test_samples = full_test

                metrics = self._run_single_experiment(config, experiment_id=int(data['combination_id']),
                                                       save_history=True)

                # Enhanced result with all metrics (matching Stage 1 format)
                result = {
                    'stage2_rank': i + 1,
                    'combination_id': int(data['combination_id']),
                    'selection_criterion': data.get('selection_criterion', 'unknown'),
                    **params,
                    # Main metrics
                    'roc_auc': metrics['roc_auc'],
                    'f1_score': metrics['f1_score'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    # Disturbing normal metrics (if available)
                    'disturbing_roc_auc': metrics.get('disturbing_roc_auc', None),
                    'disturbing_f1': metrics.get('disturbing_f1', None),
                    'disturbing_precision': metrics.get('disturbing_precision', None),
                    'disturbing_recall': metrics.get('disturbing_recall', None),
                    # Quick search reference
                    'quick_roc_auc': data['roc_auc'],
                    'quick_f1': data.get('f1_score', None),
                    'quick_disturbing_roc_auc': data.get('disturbing_roc_auc', None),
                    # Improvement from quick to full
                    'roc_auc_improvement': metrics['roc_auc'] - data['roc_auc'],
                }
                self.full_results.append(result)

                # Enhanced per-trial output
                params_str = ", ".join(f"{k}={v}" for k, v in params.items())
                tqdm.write(f"  [{i+1}/{actual_k}] criterion={data.get('selection_criterion', 'N/A')}")
                tqdm.write(f"    Params: {params_str}")
                tqdm.write(f"    -> ROC-AUC: {metrics['roc_auc']:.4f} (quick: {data['roc_auc']:.4f}, Δ={metrics['roc_auc'] - data['roc_auc']:+.4f})")
                tqdm.write(f"    -> F1: {metrics['f1_score']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
                if 'disturbing_roc_auc' in metrics and metrics['disturbing_roc_auc'] is not None:
                    tqdm.write(f"    -> Disturbing Normal - ROC-AUC: {metrics['disturbing_roc_auc']:.4f}, F1: {metrics['disturbing_f1']:.4f}")

            full_df = pd.DataFrame(self.full_results)
            full_df = full_df.sort_values('roc_auc', ascending=False).reset_index(drop=True)

            print(f"\nFull Search Complete!")
            print(f"Best ROC-AUC: {full_df['roc_auc'].max():.4f}")
            print(f"Best F1-Score: {full_df['f1_score'].max():.4f}")
            if 'disturbing_roc_auc' in full_df.columns and full_df['disturbing_roc_auc'].notna().any():
                print(f"Best Disturbing ROC-AUC: {full_df['disturbing_roc_auc'].max():.4f}")

            # Print top 10 results summary
            print("\n--- Top 10 Stage 2 Results ---")
            for i, row in full_df.head(10).iterrows():
                print(f"  #{i+1}: ROC-AUC={row['roc_auc']:.4f}, F1={row['f1_score']:.4f}, "
                      f"criterion={row['selection_criterion']}")

            # Save full search results
            full_df.to_csv(os.path.join(self.output_dir, 'full_search_results.csv'), index=False)

            final_df = full_df
        else:
            final_df = quick_df

        # Save all results
        self._save_all_results()

        return final_df

    def _select_stage2_candidates(
        self,
        quick_df: pd.DataFrame,
        total_k: int = 150
    ) -> pd.DataFrame:
        """
        Select diverse candidates for Stage 2 training.

        Selection is proportionally scaled based on total_k.
        For total_k=150, base allocation is:
        - 30 by overall ROC-AUC
        - 20 by disturbing_roc_auc
        - Others get ~10 each

        For smaller total_k, allocations are scaled down proportionally.
        Returns unique candidates (no duplicates), capped at total_k.
        """
        selected_ids = set()
        selection_info = []

        def add_candidates(df_subset, criterion_name, max_count):
            """Add candidates from subset, skip duplicates. Respects total_k limit."""
            added = 0
            for _, row in df_subset.iterrows():
                if len(selected_ids) >= total_k:
                    break
                cid = int(row['combination_id'])
                if cid not in selected_ids:
                    selected_ids.add(cid)
                    selection_info.append({
                        'combination_id': cid,
                        'selection_criterion': criterion_name,
                        'criterion_rank': added + 1
                    })
                    added += 1
                    if added >= max_count:
                        break
            return added

        # Scale allocations based on total_k (base is 150)
        scale = total_k / 150.0

        # 1. Top by overall ROC-AUC (scaled from 30)
        df_by_roc = quick_df.sort_values('roc_auc', ascending=False)
        n_roc = max(1, int(30 * scale))
        added = add_candidates(df_by_roc, 'overall_roc_auc', n_roc)
        print(f"  Added {added} models by overall ROC-AUC")

        if len(selected_ids) >= total_k:
            pass  # Skip remaining criteria
        else:
            # 2. Top by disturbing_roc_auc (scaled from 20)
            if 'disturbing_roc_auc' in quick_df.columns:
                df_by_disturbing = quick_df.sort_values('disturbing_roc_auc', ascending=False)
                n_disturb = max(1, int(20 * scale))
                added = add_candidates(df_by_disturbing, 'disturbing_roc_auc', n_disturb)
                print(f"  Added {added} models by disturbing ROC-AUC")

        # Scale for category-specific selections (base is 10 each)
        n_category = max(1, int(10 * scale))

        if len(selected_ids) < total_k and 'force_mask_anomaly' in quick_df.columns:
            df_force = quick_df[quick_df['force_mask_anomaly'] == True].sort_values('roc_auc', ascending=False)
            added = add_candidates(df_force, 'force_mask_anomaly=True', n_category)
            print(f"  Added {added} models with force_mask_anomaly=True")

        if len(selected_ids) < total_k and 'patch_level_loss' in quick_df.columns:
            df_patch = quick_df[quick_df['patch_level_loss'] == True].sort_values('roc_auc', ascending=False)
            added = add_candidates(df_patch, 'patch_level_loss=True', n_category)
            print(f"  Added {added} models with patch_level_loss=True")

        if len(selected_ids) < total_k and 'margin_type' in quick_df.columns:
            for mt in ['dynamic', 'softplus', 'hinge']:
                if len(selected_ids) >= total_k:
                    break
                df_mt = quick_df[quick_df['margin_type'] == mt].sort_values('roc_auc', ascending=False)
                added = add_candidates(df_mt, f'margin_type={mt}', n_category)
                print(f"  Added {added} models with margin_type={mt}")

        if len(selected_ids) < total_k and 'patchify_mode' in quick_df.columns:
            for pm in ['cnn_first', 'patch_cnn', 'linear']:
                if len(selected_ids) >= total_k:
                    break
                df_pm = quick_df[quick_df['patchify_mode'] == pm].sort_values('roc_auc', ascending=False)
                added = add_candidates(df_pm, f'patchify_mode={pm}', n_category)
                print(f"  Added {added} models with patchify_mode={pm}")

        if len(selected_ids) < total_k and 'masking_strategy' in quick_df.columns:
            for ms in ['patch', 'feature_wise']:
                if len(selected_ids) >= total_k:
                    break
                df_ms = quick_df[quick_df['masking_strategy'] == ms].sort_values('roc_auc', ascending=False)
                added = add_candidates(df_ms, f'masking_strategy={ms}', n_category)
                print(f"  Added {added} models with masking_strategy={ms}")

        # Create the selection dataframe
        selection_df = pd.DataFrame(selection_info)
        selected_df = quick_df[quick_df['combination_id'].isin(selected_ids)].copy()
        selected_df = selected_df.merge(selection_df, on='combination_id', how='left')

        # Sort by ROC-AUC but keep selection info
        selected_df = selected_df.sort_values('roc_auc', ascending=False).reset_index(drop=True)

        print(f"\n  Total unique models selected: {len(selected_df)}")

        return selected_df

    def _save_all_results(self):
        """Save all experiment results"""
        # Save best config
        if self.best_config is not None:
            best_params = {}
            for key in self.param_grid.keys():
                best_params[key] = getattr(self.best_config, key)

            best_params['patch_size'] = self.best_config.patch_size
            best_params['mask_last_n'] = self.best_config.mask_last_n

            with open(os.path.join(self.output_dir, 'best_config.json'), 'w') as f:
                json.dump(best_params, f, indent=2)

        # Save best model
        if self.best_model is not None:
            torch.save({
                'model_state_dict': self.best_model.state_dict(),
                'config': asdict(self.best_config),
                'metrics': self.best_metrics
            }, os.path.join(self.output_dir, 'best_model.pt'))

            # Save detailed results for best model (anomaly type analysis)
            self._save_best_model_detailed_results()

        # Save training histories
        if self.histories:
            # Convert numpy arrays to lists for JSON serialization
            histories_serializable = {}
            for exp_id, history in self.histories.items():
                histories_serializable[str(exp_id)] = {
                    k: [float(v) if isinstance(v, (int, float, np.floating)) else v for v in vals]
                    for k, vals in history.items()
                }
            with open(os.path.join(self.output_dir, 'training_histories.json'), 'w') as f:
                json.dump(histories_serializable, f, indent=2)

        # Save experiment metadata
        metadata = {
            'param_grid': {k: [str(v) if isinstance(v, bool) else v for v in vals]
                          for k, vals in self.param_grid.items()},
            'total_combinations': len(self.generate_combinations()),
            'stage1_results_count': len(self.results),
            'stage2_results_count': len(self.full_results),
            'best_metrics': self.best_metrics,
            'timestamp': datetime.now().isoformat()
        }
        with open(os.path.join(self.output_dir, 'experiment_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        print(f"\nAll results saved to {self.output_dir}")
        print(f"  - quick_search_results.csv")
        if self.full_results:
            print(f"  - full_search_results.csv")
        print(f"  - best_model.pt")
        print(f"  - best_config.json")
        print(f"  - best_model_detailed.csv")
        print(f"  - anomaly_type_metrics.json")
        if self.histories:
            print(f"  - training_histories.json")
        print(f"  - experiment_metadata.json")

    def _save_best_model_detailed_results(self):
        """Save detailed results for best model including anomaly type analysis"""
        if self.best_model is None or self.best_config is None:
            return

        print("\nGenerating detailed results for best model...")

        # Create test dataset with same parameters used for final evaluation
        test_dataset = MultivariateTimeSeriesDataset(
            num_samples=self.best_config.num_test_samples,
            seq_length=self.best_config.seq_length,
            num_features=self.best_config.num_features,
            anomaly_ratio=self.best_config.test_anomaly_ratio,
            mask_last_n=self.best_config.mask_last_n,
            seed=self.best_config.random_seed + 1
        )
        test_loader = DataLoader(test_dataset, batch_size=self.best_config.batch_size, shuffle=False)

        # Create evaluator
        evaluator = Evaluator(self.best_model, self.best_config, test_loader)

        # Get detailed losses
        detailed_losses = evaluator.compute_detailed_losses()

        # Save detailed results as CSV
        detailed_df = pd.DataFrame({
            'reconstruction_loss': detailed_losses['reconstruction_loss'],
            'discrepancy_loss': detailed_losses['discrepancy_loss'],
            'total_loss': detailed_losses['total_loss'],
            'label': detailed_losses['labels'],
            'sample_type': detailed_losses['sample_types'],
            'anomaly_type': detailed_losses['anomaly_types'],
            'anomaly_type_name': [ANOMALY_TYPE_NAMES[int(at)] for at in detailed_losses['anomaly_types']]
        })
        detailed_df.to_csv(os.path.join(self.output_dir, 'best_model_detailed.csv'), index=False)

        # Get performance by anomaly type
        anomaly_type_metrics = evaluator.get_performance_by_anomaly_type()

        # Add summary statistics per anomaly type
        for atype_name in anomaly_type_metrics.keys():
            atype_mask = detailed_df['anomaly_type_name'] == atype_name
            if atype_mask.sum() > 0:
                anomaly_type_metrics[atype_name]['mean_reconstruction_loss'] = float(
                    detailed_df.loc[atype_mask, 'reconstruction_loss'].mean()
                )
                anomaly_type_metrics[atype_name]['std_reconstruction_loss'] = float(
                    detailed_df.loc[atype_mask, 'reconstruction_loss'].std()
                )
                anomaly_type_metrics[atype_name]['mean_discrepancy_loss'] = float(
                    detailed_df.loc[atype_mask, 'discrepancy_loss'].mean()
                )
                anomaly_type_metrics[atype_name]['std_discrepancy_loss'] = float(
                    detailed_df.loc[atype_mask, 'discrepancy_loss'].std()
                )

        # Save anomaly type metrics as JSON
        with open(os.path.join(self.output_dir, 'anomaly_type_metrics.json'), 'w') as f:
            json.dump(anomaly_type_metrics, f, indent=2)

        print(f"  - Saved best_model_detailed.csv ({len(detailed_df)} samples)")
        print(f"  - Saved anomaly_type_metrics.json ({len(anomaly_type_metrics)} types)")


# =============================================================================
# Main Function
# =============================================================================

def run_experiments(
    param_grid: Dict[str, List] = None,
    quick_epochs: int = 15,
    quick_train: int = 1000,
    quick_test: int = 400,
    full_epochs: int = 100,
    full_train: int = 2000,
    full_test: int = 500,
    top_k: int = 50,
    two_stage: bool = True,
    output_dir: str = 'results/experiments'
) -> Tuple[pd.DataFrame, str]:
    """
    Run complete experiment pipeline (without visualization)

    Results are saved to output_dir/YYYYMMDD_HHMMSS/
    Use visualize_all.py to generate visualizations from saved results.

    Args:
        param_grid: Parameter grid to search
        quick_epochs: Epochs for quick search
        quick_train: Training samples for quick search
        quick_test: Test samples for quick search
        full_epochs: Epochs for full search
        full_train: Training samples for full search
        full_test: Test samples for full search
        top_k: Number of top combinations for full search
        two_stage: If True, run quick then full search
        output_dir: Directory to save results

    Returns:
        (results_df, output_dir_path)
    """
    print("\n" + "="*80)
    print(" " * 15 + "SELF-DISTILLED MAE EXPERIMENTS")
    print("="*80)

    # Initialize runner
    runner = ExperimentRunner(
        param_grid=param_grid,
        output_dir=output_dir
    )

    # Run grid search
    results_df = runner.run_grid_search(
        quick_epochs=quick_epochs,
        quick_train=quick_train,
        quick_test=quick_test,
        full_epochs=full_epochs,
        full_train=full_train,
        full_test=full_test,
        top_k=top_k,
        two_stage=two_stage
    )

    # Print best configuration
    print("\n" + "="*80)
    print(" " * 20 + "BEST CONFIGURATION")
    print("="*80)

    best_row = results_df.iloc[0]
    print("\nParameters:")
    for key in runner.param_grid.keys():
        print(f"  {key}: {best_row[key]}")

    if 'num_patches' in runner.param_grid:
        print(f"  patch_size: {100 // int(best_row['num_patches'])}")

    print("\nMetrics:")
    print(f"  ROC-AUC: {best_row['roc_auc']:.4f}")
    print(f"  F1-Score: {best_row['f1_score']:.4f}")
    print(f"  Precision: {best_row['precision']:.4f}")
    print(f"  Recall: {best_row['recall']:.4f}")

    print("\n" + "="*80)
    print(" " * 15 + "EXPERIMENTS COMPLETE!")
    print(f"Results saved to: {runner.output_dir}")
    print(f"\nTo generate visualizations, run:")
    print(f"  python scripts/visualize_all.py --experiment-dir {runner.output_dir}")
    print("="*80)

    return results_df, runner.output_dir


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run Self-Distilled MAE experiments')
    parser.add_argument('--quick-epochs', type=int, default=15, help='Epochs for quick search')
    parser.add_argument('--quick-train', type=int, default=1000, help='Training samples for quick search')
    parser.add_argument('--quick-test', type=int, default=400, help='Test samples for quick search')
    parser.add_argument('--full-epochs', type=int, default=100, help='Epochs for full search')
    parser.add_argument('--full-train', type=int, default=2000, help='Training samples for full search')
    parser.add_argument('--full-test', type=int, default=500, help='Test samples for full search')
    parser.add_argument('--top-k', type=int, default=130, help='Number of top combinations for full search')
    parser.add_argument('--no-two-stage', action='store_true', help='Disable two-stage search')
    parser.add_argument('--output-dir', type=str, default='results/experiments', help='Output directory')
    args = parser.parse_args()

    run_experiments(
        quick_epochs=args.quick_epochs,
        quick_train=args.quick_train,
        quick_test=args.quick_test,
        full_epochs=args.full_epochs,
        full_train=args.full_train,
        full_test=args.full_test,
        top_k=args.top_k,
        two_stage=not args.no_two_stage,
        output_dir=args.output_dir
    )
