"""
Phase 2 Ablation Study Configuration
=====================================
Advanced Experiments Based on Phase 1 Insights

PRIMARY GOAL: Maximize disc_cohens_d_normal_vs_anomaly AND recon_cohens_d_normal_vs_anomaly
           while achieving high ROC-AUC and PA%80 performance

Created: 2026-01-27
Total Experiments: 150

KEY INSIGHTS FROM PHASE 1:
- Best models balance high disc_d (0.7-1.2) AND high recon_d (2.0-3.8)
- all_patches inference mode superior to last_patch (0.836 vs 0.789 avg ROC-AUC)
- default scoring mode best for deployment
- High t_ratio (5.5-6.6) correlates with good performance
- w500_p20 configuration shows strong baseline
- d_model=128 shows good balance between capacity and generalization
- disc_cohens_d_disturbing_vs_anomaly critical for robust detection

EXPERIMENT GROUPS:
  GROUP 1 (001-030): Balanced High Disc+Recon Optimization
  GROUP 2 (031-055): Window Size & Capacity Exploration
  GROUP 3 (056-075): Disturbing Normal Separation Focus
  GROUP 4 (076-095): PA%80 Optimization
  GROUP 5 (096-110): Teacher-Student Ratio Exploration
  GROUP 6 (111-125): Masking Strategy Fine-tuning
  GROUP 7 (126-140): Architecture Depth Optimization
  GROUP 8 (141-150): Lambda Discrepancy & Loss Weighting
"""

from copy import deepcopy
from typing import Dict, List

# =============================================================================
# Phase Metadata
# =============================================================================

PHASE_NAME = "phase2"
PHASE_DESCRIPTION = "Advanced Ablation Study - Maximize Disc+Recon Balance"
CREATED_AT = "2026-01-27"

# =============================================================================
# Scoring and Inference Modes
# =============================================================================

# Focus primarily on best-performing modes from Phase 1
SCORING_MODES = ['default']  # Primary focus
INFERENCE_MODES = ['all_patches']  # Primary focus

# Optional: Test specific variations with adaptive/normalized scoring
SCORING_MODES_EXTENDED = ['default', 'adaptive', 'normalized']
INFERENCE_MODES_EXTENDED = ['all_patches', 'last_patch']

# =============================================================================
# Base Configuration
# =============================================================================

# Optimized base configuration from Phase 1 top performers
BASE_CONFIG = {
    # Core settings
    'force_mask_anomaly': True,
    'margin_type': 'dynamic',
    'mask_after_encoder': False,  # Best performers used mask_after=False
    'masking_ratio': 0.75,  # Phase 1 default
    'masking_strategy': 'patch',

    # Window and patch (start from w500_p20 winner)
    'seq_length': 500,
    'patch_size': 20,
    'num_patches': 25,

    # Architecture (balanced configuration)
    'patch_level_loss': True,
    'patchify_mode': 'patch_cnn',
    'shared_mask_token': False,
    'd_model': 256,  # Good balance from Phase 1
    'nhead': 8,
    'num_encoder_layers': 6,
    'num_teacher_decoder_layers': 4,
    'num_student_decoder_layers': 1,
    'num_shared_decoder_layers': 0,
    'dim_feedforward': 2048,
    'dropout': 0.1,
    'cnn_channels': (32, 64),

    # Loss and training
    'anomaly_loss_weight': 1.0,
    'num_epochs': 50,
    'mask_last_n': 20,
    'margin': 0.5,
    'lambda_disc': 2.0,  # Phase 1 default
    'dynamic_margin_k': 2.0,  # Phase 1 default
    'learning_rate': 2e-3,
    'weight_decay': 1e-5,
    'teacher_only_warmup_epochs': 3,
    'warmup_epochs': 10,
}

# =============================================================================
# Experiment Configurations
# =============================================================================

def get_experiments() -> List[Dict]:
    """Define 150 experiment configurations for Phase 2."""
    experiments = []
    base = deepcopy(BASE_CONFIG)

    # =========================================================================
    # GROUP 1 (001-030): Balanced High Disc+Recon Optimization
    # =========================================================================
    # Build on 009_w500_p20, 063_w500_p20_d128, 028_d128_nhead_16
    # Goal: Maximize both disc_cohens_d AND recon_cohens_d

    # Subgroup 1a: d_model=128 variations (best t_ratio from Phase 1)
    for i, (mr, ld) in enumerate([
        (0.70, 2.0), (0.75, 2.0), (0.80, 2.0),
        (0.70, 1.5), (0.75, 1.5), (0.80, 1.5),
        (0.70, 2.5), (0.75, 2.5), (0.80, 2.5),
        (0.65, 2.0), (0.85, 2.0),
    ], 1):
        exp = deepcopy(base)
        exp['name'] = f'{i:03d}_optimal_d128_mr{mr}_ld{ld}'
        exp['d_model'] = 128
        exp['nhead'] = 8 if i % 2 == 0 else 16
        exp['masking_ratio'] = mr
        exp['lambda_disc'] = ld
        exp['num_teacher_decoder_layers'] = 4
        exp['num_student_decoder_layers'] = 1
        experiments.append(exp)

    # Subgroup 1b: nhead=16 variations (028_d128_nhead_16 was top in high disc+recon)
    for i, (dd, mr) in enumerate([
        (3, 0.75), (4, 0.75), (5, 0.75),
        (3, 0.70), (4, 0.70), (5, 0.70),
        (3, 0.80), (4, 0.80), (5, 0.80),
    ], 12):
        exp = deepcopy(base)
        exp['name'] = f'{i:03d}_optimal_d128_nhead16_dd{dd}_mr{mr}'
        exp['d_model'] = 128
        exp['nhead'] = 16
        exp['num_teacher_decoder_layers'] = dd
        exp['masking_ratio'] = mr
        exp['lambda_disc'] = 2.0
        experiments.append(exp)

    # Subgroup 1c: Patch size variations with optimal settings
    for i, (ps, np, ml) in enumerate([
        (10, 50, 10),
        (15, 33, 15),
        (25, 20, 25),
        (30, 16, 30),
    ], 21):
        exp = deepcopy(base)
        exp['name'] = f'{i:03d}_optimal_patch{ps}'
        exp['d_model'] = 128
        exp['patch_size'] = ps
        exp['num_patches'] = np
        exp['mask_last_n'] = ml
        exp['lambda_disc'] = 2.0
        experiments.append(exp)

    # Subgroup 1d: FFN dimension variations
    for i, ffn in enumerate([512, 1024, 1536, 2048, 3072], 25):
        exp = deepcopy(base)
        exp['name'] = f'{i:03d}_optimal_d128_ffn{ffn}'
        exp['d_model'] = 128
        exp['dim_feedforward'] = ffn
        exp['lambda_disc'] = 2.0
        experiments.append(exp)

    # Final experiment for group 1
    exp = deepcopy(base)
    exp['name'] = '030_optimal_combined_best'
    exp['d_model'] = 128
    exp['nhead'] = 16
    exp['num_teacher_decoder_layers'] = 4
    exp['masking_ratio'] = 0.75
    exp['lambda_disc'] = 2.0
    exp['dim_feedforward'] = 1536
    experiments.append(exp)

    # =========================================================================
    # GROUP 2 (031-055): Window Size & Capacity Exploration
    # =========================================================================
    # Goal: Understand optimal model capacity for different window sizes

    # Subgroup 2a: w500 with various capacities
    for i, (dm, nh, dd) in enumerate([
        (96, 6, 3), (128, 8, 3), (192, 8, 4),
        (256, 8, 4), (320, 8, 5),
        (96, 8, 4), (128, 16, 3), (192, 12, 4),
    ], 31):
        exp = deepcopy(base)
        exp['name'] = f'{i:03d}_w500_d{dm}_nh{nh}_dd{dd}'
        exp['seq_length'] = 500
        exp['patch_size'] = 20
        exp['num_patches'] = 25
        exp['d_model'] = dm
        exp['nhead'] = nh
        exp['num_teacher_decoder_layers'] = dd
        exp['lambda_disc'] = 2.0
        experiments.append(exp)

    # Subgroup 2b: w1000 exploration
    for i, (dm, ps, np, dd) in enumerate([
        (128, 20, 50, 3), (192, 20, 50, 4), (256, 20, 50, 4),
        (128, 25, 40, 4), (192, 25, 40, 4), (256, 25, 40, 5),
        (128, 40, 25, 3), (192, 40, 25, 4),
    ], 39):
        exp = deepcopy(base)
        exp['name'] = f'{i:03d}_w1000_d{dm}_p{ps}_dd{dd}'
        exp['seq_length'] = 1000
        exp['patch_size'] = ps
        exp['num_patches'] = np
        exp['mask_last_n'] = ps
        exp['d_model'] = dm
        exp['num_teacher_decoder_layers'] = dd
        exp['lambda_disc'] = 2.0
        experiments.append(exp)

    # Subgroup 2c: w100 with reduced capacity (baseline)
    for i, (dm, nh, dd) in enumerate([
        (64, 4, 2), (96, 6, 3), (128, 8, 3),
        (64, 8, 3), (96, 8, 4),
    ], 47):
        exp = deepcopy(base)
        exp['name'] = f'{i:03d}_w100_d{dm}_nh{nh}_dd{dd}'
        exp['seq_length'] = 100
        exp['patch_size'] = 10
        exp['num_patches'] = 10
        exp['mask_last_n'] = 10
        exp['d_model'] = dm
        exp['nhead'] = nh
        exp['num_teacher_decoder_layers'] = dd
        experiments.append(exp)

    # Subgroup 2d: w500 with different patch strategies
    for i, (ps, mr) in enumerate([
        (10, 0.70), (15, 0.75), (30, 0.80),
    ], 52):
        exp = deepcopy(base)
        exp['name'] = f'{i:03d}_w500_patch{ps}_mr{mr}'
        exp['seq_length'] = 500
        exp['patch_size'] = ps
        exp['num_patches'] = 500 // ps
        exp['mask_last_n'] = ps
        exp['d_model'] = 128
        exp['masking_ratio'] = mr
        experiments.append(exp)

    # Final experiment for group 2
    exp = deepcopy(base)
    exp['name'] = '055_w1000_optimal'
    exp['seq_length'] = 1000
    exp['patch_size'] = 25
    exp['num_patches'] = 40
    exp['mask_last_n'] = 25
    exp['d_model'] = 192
    exp['nhead'] = 12
    exp['num_teacher_decoder_layers'] = 4
    exp['lambda_disc'] = 2.0
    experiments.append(exp)

    # =========================================================================
    # GROUP 3 (056-075): Disturbing Normal Separation Focus
    # =========================================================================
    # Goal: Maximize disc_cohens_d_disturbing_vs_anomaly
    # Build on 009_w500_p20 (top disturbing separator)

    # Subgroup 3a: k value variations (dynamic margin)
    for i, k in enumerate([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], 56):
        exp = deepcopy(base)
        exp['name'] = f'{i:03d}_disturb_k{k}'
        exp['d_model'] = 128
        exp['dynamic_margin_k'] = k
        exp['lambda_disc'] = 2.0
        experiments.append(exp)

    # Subgroup 3b: lambda_disc variations for disturbing separation
    for i, ld in enumerate([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], 65):
        exp = deepcopy(base)
        exp['name'] = f'{i:03d}_disturb_lambda{ld}'
        exp['d_model'] = 128
        exp['lambda_disc'] = ld
        exp['dynamic_margin_k'] = 2.5
        experiments.append(exp)

    # Subgroup 3c: Anomaly weight variations
    for i, aw in enumerate([0.5, 1.0, 1.5, 2.0], 71):
        exp = deepcopy(base)
        exp['name'] = f'{i:03d}_disturb_anomalyweight{aw}'
        exp['d_model'] = 128
        exp['anomaly_loss_weight'] = aw
        exp['dynamic_margin_k'] = 2.5
        experiments.append(exp)

    # Final experiment for group 3
    exp = deepcopy(base)
    exp['name'] = '075_disturb_optimal'
    exp['d_model'] = 128
    exp['dynamic_margin_k'] = 3.0
    exp['lambda_disc'] = 2.5
    exp['anomaly_loss_weight'] = 1.5
    experiments.append(exp)

    # =========================================================================
    # GROUP 4 (076-095): PA%80 Optimization
    # =========================================================================
    # Goal: Maximize PA%80 performance for deployment

    # Subgroup 4a: High disc_ratio + scoring mode combinations
    # Test extended scoring modes on best configs
    scoring_tests = []
    for scoring in SCORING_MODES_EXTENDED:
        for inference in INFERENCE_MODES_EXTENDED:
            exp = deepcopy(base)
            exp['d_model'] = 128
            exp['lambda_disc'] = 2.0
            exp['num_teacher_decoder_layers'] = 4
            scoring_tests.append((scoring, inference))

    for i, (scoring, inference) in enumerate(scoring_tests, 76):
        exp = deepcopy(base)
        exp['name'] = f'{i:03d}_pa80_{scoring}_{inference}'
        exp['d_model'] = 128
        exp['lambda_disc'] = 2.0
        exp['num_teacher_decoder_layers'] = 4
        # Note: scoring and inference mode will be applied at runtime
        experiments.append(exp)

    # Subgroup 4b: Larger window for PA%80
    for i, (ws, ps) in enumerate([
        (500, 20), (1000, 25), (1000, 40),
    ], 82):
        exp = deepcopy(base)
        exp['name'] = f'{i:03d}_pa80_w{ws}_p{ps}'
        exp['seq_length'] = ws
        exp['patch_size'] = ps
        exp['num_patches'] = ws // ps
        exp['mask_last_n'] = ps
        exp['d_model'] = 128
        exp['lambda_disc'] = 2.5
        experiments.append(exp)

    # Subgroup 4c: High capacity for better PA%80
    for i, (dm, dd) in enumerate([
        (192, 4), (256, 4), (192, 5), (256, 5),
        (192, 6), (256, 6),
    ], 85):
        exp = deepcopy(base)
        exp['name'] = f'{i:03d}_pa80_d{dm}_dd{dd}'
        exp['d_model'] = dm
        exp['num_teacher_decoder_layers'] = dd
        exp['lambda_disc'] = 2.0
        experiments.append(exp)

    # Subgroup 4d: Combined optimizations
    for i, (ws, dm, dd, ld) in enumerate([
        (500, 192, 4, 2.0), (1000, 192, 4, 2.5),
        (500, 256, 5, 2.0), (1000, 256, 5, 2.5),
    ], 91):
        exp = deepcopy(base)
        exp['name'] = f'{i:03d}_pa80_combined_w{ws}_d{dm}'
        exp['seq_length'] = ws
        exp['patch_size'] = 25 if ws == 1000 else 20
        exp['num_patches'] = ws // (25 if ws == 1000 else 20)
        exp['mask_last_n'] = 25 if ws == 1000 else 20
        exp['d_model'] = dm
        exp['num_teacher_decoder_layers'] = dd
        exp['lambda_disc'] = ld
        experiments.append(exp)

    # Final experiment for group 4
    exp = deepcopy(base)
    exp['name'] = '095_pa80_ultimate'
    exp['seq_length'] = 1000
    exp['patch_size'] = 25
    exp['num_patches'] = 40
    exp['mask_last_n'] = 25
    exp['d_model'] = 256
    exp['nhead'] = 8
    exp['num_teacher_decoder_layers'] = 5
    exp['lambda_disc'] = 2.5
    experiments.append(exp)

    # =========================================================================
    # GROUP 5 (096-110): Teacher-Student Ratio Exploration
    # =========================================================================
    # Goal: Find optimal teacher-student loss balance

    # Subgroup 5a: Various T:S ratios
    for i, (td, sd) in enumerate([
        (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1),
        (2, 2), (3, 2), (4, 2), (5, 2),
        (3, 3), (4, 3), (5, 3),
        (4, 4), (5, 5),
    ], 96):
        exp = deepcopy(base)
        exp['name'] = f'{i:03d}_ts_t{td}s{sd}'
        exp['d_model'] = 128
        exp['num_teacher_decoder_layers'] = td
        exp['num_student_decoder_layers'] = sd
        exp['lambda_disc'] = 2.0
        experiments.append(exp)

    # =========================================================================
    # GROUP 6 (111-125): Masking Strategy Fine-tuning
    # =========================================================================
    # Goal: Find optimal masking ratio for different d_models

    # Subgroup 6a: d_model=128 masking
    for i, mr in enumerate([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35], 111):
        exp = deepcopy(base)
        exp['name'] = f'{i:03d}_d128_mask{mr}'
        exp['d_model'] = 128
        exp['masking_ratio'] = mr
        exp['lambda_disc'] = 2.0
        experiments.append(exp)

    # Subgroup 6b: d_model=256 masking
    for i, mr in enumerate([0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90], 118):
        exp = deepcopy(base)
        exp['name'] = f'{i:03d}_d256_mask{mr}'
        exp['d_model'] = 256
        exp['masking_ratio'] = mr
        exp['lambda_disc'] = 2.0
        experiments.append(exp)

    # Final experiment for group 6
    exp = deepcopy(base)
    exp['name'] = '125_mask_optimal'
    exp['d_model'] = 128
    exp['masking_ratio'] = 0.15
    exp['lambda_disc'] = 2.0
    experiments.append(exp)

    # =========================================================================
    # GROUP 7 (126-140): Architecture Depth Optimization
    # =========================================================================
    # Goal: Find optimal encoder/decoder depth balance

    # Subgroup 7a: Decoder depth variations
    for i, (ed, dd) in enumerate([
        (4, 2), (6, 2), (8, 2),
        (4, 3), (6, 3), (8, 3),
        (4, 4), (6, 4), (8, 4),
        (4, 5), (6, 5), (8, 5),
        (4, 6), (6, 6), (8, 6),
    ], 126):
        exp = deepcopy(base)
        exp['name'] = f'{i:03d}_depth_e{ed}d{dd}'
        exp['d_model'] = 128
        exp['num_encoder_layers'] = ed
        exp['num_teacher_decoder_layers'] = dd
        exp['lambda_disc'] = 2.0
        experiments.append(exp)

    # =========================================================================
    # GROUP 8 (141-150): Lambda Discrepancy & Loss Weighting
    # =========================================================================
    # Goal: Optimize loss weighting for maximum performance

    # Subgroup 8a: Fine-grained lambda_disc
    for i, ld in enumerate([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0], 141):
        exp = deepcopy(base)
        exp['name'] = f'{i:03d}_lambda{ld}'
        exp['d_model'] = 128
        exp['num_teacher_decoder_layers'] = 4
        exp['lambda_disc'] = ld
        exp['masking_ratio'] = 0.75
        experiments.append(exp)

    # Verify we have exactly 150 experiments
    assert len(experiments) == 150, f"Expected 150 experiments, got {len(experiments)}"

    return experiments


# =============================================================================
# Helper Functions
# =============================================================================

def print_experiment_summary():
    """Print summary of all experiments."""
    experiments = get_experiments()

    print(f"Phase 2 Ablation Study: {len(experiments)} experiments")
    print("=" * 80)

    groups = [
        ("GROUP 1: Balanced High Disc+Recon", 1, 30),
        ("GROUP 2: Window Size & Capacity", 31, 55),
        ("GROUP 3: Disturbing Normal Separation", 56, 75),
        ("GROUP 4: PA%80 Optimization", 76, 95),
        ("GROUP 5: Teacher-Student Ratios", 96, 110),
        ("GROUP 6: Masking Strategy", 111, 125),
        ("GROUP 7: Architecture Depth", 126, 140),
        ("GROUP 8: Lambda Discrepancy", 141, 150),
    ]

    for group_name, start, end in groups:
        count = end - start + 1
        print(f"{group_name}: {count} experiments ({start:03d}-{end:03d})")

    print("=" * 80)


if __name__ == "__main__":
    print_experiment_summary()
    experiments = get_experiments()
    print(f"\nGenerated {len(experiments)} experiments successfully!")

    # Print first and last few
    print("\nFirst 3 experiments:")
    for exp in experiments[:3]:
        print(f"  - {exp['name']}")

    print("\nLast 3 experiments:")
    for exp in experiments[-3:]:
        print(f"  - {exp['name']}")
