"""
Unified Phase 1 Ablation Study Configuration
=============================================
Combined Architecture Exploration + Focused Optimization

PRIMARY GOAL: Comprehensive ablation study with all experiments

Created: 2026-01-27
Total Experiments: 170 (70 from original Phase 1 + 100 from Phase 2)

ORIGINAL PHASE 1 (001-070): Architecture and Loss Parameter Experiments
  GROUP 1 (001-010): Window Size & Patch Variations
  GROUP 2 (011-020): Encoder/Decoder Depth Variations
  GROUP 3 (021-030): Model Capacity (d_model, nhead, FFN)
  GROUP 4 (031-040): Masking Ratio Variations
  GROUP 5 (041-050): Loss Parameters
  GROUP 6 (051-060): Training Parameters
  GROUP 7 (061-070): Combined Optimal Configurations

ORIGINAL PHASE 2 (071-170): Focused Experiments Based on Phase 1 Insights
  GROUP 8 (071-085): Maximize disc_ratio with d_model=128
  GROUP 9 (086-100): Maximize disc_ratio AND t_ratio simultaneously
  GROUP 10 (101-120): Best Model Variations (Scoring & Window)
  GROUP 11 (121-140): Window Size vs Model Depth Relationship
  GROUP 12 (141-155): PA%80 Optimization Experiments
  GROUP 13 (156-170): Additional Explorations
"""

from copy import deepcopy
from typing import Dict, List

# =============================================================================
# Phase Metadata
# =============================================================================

PHASE_NAME = "phase1"
PHASE_DESCRIPTION = "Unified Ablation Study (Architecture + Focused Optimization)"
CREATED_AT = "2026-01-27 05:22:20"

# =============================================================================
# Scoring and Inference Modes
# =============================================================================

SCORING_MODES = ['default', 'adaptive', 'normalized']
INFERENCE_MODES = ['last_patch', 'all_patches']

# =============================================================================
# Base Configurations
# =============================================================================

# Original Phase 1 base config (d_model=64)
BASE_CONFIG_PHASE1 = {
    'force_mask_anomaly': True,
    'margin_type': 'dynamic',
    'mask_after_encoder': False,
    'masking_ratio': 0.2,
    'masking_strategy': 'patch',
    'seq_length': 100,
    'num_patches': 10,
    'patch_size': 10,
    'patch_level_loss': True,
    'patchify_mode': 'patch_cnn',
    'shared_mask_token': False,
    'd_model': 64,
    'nhead': 2,
    'num_encoder_layers': 1,
    'num_teacher_decoder_layers': 2,
    'num_student_decoder_layers': 1,
    'num_shared_decoder_layers': 0,
    'dim_feedforward': 256,
    'dropout': 0.1,
    'cnn_channels': (32, 64),
    'anomaly_loss_weight': 1.0,
    'num_epochs': 50,
    'mask_last_n': 10,
    'margin': 0.5,
    'lambda_disc': 0.5,
    'dynamic_margin_k': 1.5,
    'learning_rate': 2e-3,
    'weight_decay': 1e-5,
    'teacher_only_warmup_epochs': 3,
    'warmup_epochs': 10,
}

# Original Phase 2 base config (now using same defaults as Phase 1)
BASE_CONFIG_PHASE2 = {
    'force_mask_anomaly': True,
    'margin_type': 'dynamic',
    'mask_after_encoder': False,
    'masking_ratio': 0.2,
    'masking_strategy': 'patch',
    'seq_length': 100,
    'num_patches': 10,
    'patch_size': 10,
    'patch_level_loss': True,
    'patchify_mode': 'patch_cnn',
    'shared_mask_token': False,
    'd_model': 64,
    'nhead': 2,
    'num_encoder_layers': 1,
    'num_teacher_decoder_layers': 2,
    'num_student_decoder_layers': 1,
    'num_shared_decoder_layers': 0,
    'dim_feedforward': 256,
    'dropout': 0.1,
    'cnn_channels': (32, 64),
    'anomaly_loss_weight': 1.0,
    'num_epochs': 50,
    'mask_last_n': 10,
    'margin': 0.5,
    'lambda_disc': 0.5,
    'dynamic_margin_k': 1.5,
    'learning_rate': 2e-3,
    'weight_decay': 1e-5,
    'teacher_only_warmup_epochs': 3,
    'warmup_epochs': 10,
}

# =============================================================================
# Experiment Configurations
# =============================================================================

def get_experiments() -> List[Dict]:
    """Define 170 experiment configurations for unified ablation study."""
    experiments = []

    # =========================================================================
    # ORIGINAL PHASE 1 EXPERIMENTS (001-070)
    # Using BASE_CONFIG_PHASE1 (d_model=64)
    # =========================================================================

    base1 = deepcopy(BASE_CONFIG_PHASE1)

    # -------------------------------------------------------------------------
    # GROUP 1 (001-010): Window Size & Patch Variations
    # -------------------------------------------------------------------------

    exp = deepcopy(base1)
    exp['name'] = '001_default'
    experiments.append(exp)

    exp = deepcopy(base1)
    exp['name'] = '002_window_200'
    exp['seq_length'] = 200
    exp['num_patches'] = 20
    exp['mask_last_n'] = 10
    experiments.append(exp)

    exp = deepcopy(base1)
    exp['name'] = '003_window_500'
    exp['seq_length'] = 500
    exp['num_patches'] = 50
    exp['mask_last_n'] = 10
    experiments.append(exp)

    exp = deepcopy(base1)
    exp['name'] = '004_window_1000_p20'
    exp['seq_length'] = 1000
    exp['patch_size'] = 20
    exp['num_patches'] = 50
    exp['mask_last_n'] = 20
    experiments.append(exp)

    exp = deepcopy(base1)
    exp['name'] = '005_window_1000_p10'
    exp['seq_length'] = 1000
    exp['num_patches'] = 100
    exp['mask_last_n'] = 10
    experiments.append(exp)

    exp = deepcopy(base1)
    exp['name'] = '006_patch_5'
    exp['patch_size'] = 5
    exp['num_patches'] = 20
    exp['mask_last_n'] = 5
    experiments.append(exp)

    exp = deepcopy(base1)
    exp['name'] = '007_patch_20'
    exp['patch_size'] = 20
    exp['num_patches'] = 5
    exp['mask_last_n'] = 20
    experiments.append(exp)

    exp = deepcopy(base1)
    exp['name'] = '008_w500_p5'
    exp['seq_length'] = 500
    exp['patch_size'] = 5
    exp['num_patches'] = 100
    exp['mask_last_n'] = 5
    experiments.append(exp)

    exp = deepcopy(base1)
    exp['name'] = '009_w500_p20'
    exp['seq_length'] = 500
    exp['patch_size'] = 20
    exp['num_patches'] = 25
    exp['mask_last_n'] = 20
    experiments.append(exp)

    exp = deepcopy(base1)
    exp['name'] = '010_window_2000_p20'
    exp['seq_length'] = 2000
    exp['patch_size'] = 20
    exp['num_patches'] = 100
    exp['mask_last_n'] = 20
    experiments.append(exp)

    # -------------------------------------------------------------------------
    # GROUP 2 (011-020): Encoder/Decoder Depth Variations
    # -------------------------------------------------------------------------

    exp = deepcopy(base1)
    exp['name'] = '011_encoder_2'
    exp['num_encoder_layers'] = 2
    experiments.append(exp)

    exp = deepcopy(base1)
    exp['name'] = '012_encoder_3'
    exp['num_encoder_layers'] = 3
    experiments.append(exp)

    exp = deepcopy(base1)
    exp['name'] = '013_encoder_4'
    exp['num_encoder_layers'] = 4
    experiments.append(exp)

    exp = deepcopy(base1)
    exp['name'] = '014_w500_p20_enc2'
    exp['seq_length'] = 500
    exp['patch_size'] = 20
    exp['num_patches'] = 25
    exp['mask_last_n'] = 20
    exp['num_encoder_layers'] = 2
    experiments.append(exp)

    exp = deepcopy(base1)
    exp['name'] = '015_decoder_t3s1'
    exp['num_teacher_decoder_layers'] = 3
    exp['num_student_decoder_layers'] = 1
    experiments.append(exp)

    exp = deepcopy(base1)
    exp['name'] = '016_decoder_t4s1'
    exp['num_teacher_decoder_layers'] = 4
    exp['num_student_decoder_layers'] = 1
    experiments.append(exp)

    exp = deepcopy(base1)
    exp['name'] = '017_decoder_t2s2'
    exp['num_teacher_decoder_layers'] = 2
    exp['num_student_decoder_layers'] = 2
    experiments.append(exp)

    exp = deepcopy(base1)
    exp['name'] = '018_decoder_t3s2'
    exp['num_teacher_decoder_layers'] = 3
    exp['num_student_decoder_layers'] = 2
    experiments.append(exp)

    exp = deepcopy(base1)
    exp['name'] = '019_decoder_t4s2'
    exp['num_teacher_decoder_layers'] = 4
    exp['num_student_decoder_layers'] = 2
    experiments.append(exp)

    exp = deepcopy(base1)
    exp['name'] = '020_shared_decoder'
    exp['num_shared_decoder_layers'] = 1
    exp['num_teacher_decoder_layers'] = 1
    exp['num_student_decoder_layers'] = 1
    experiments.append(exp)

    # -------------------------------------------------------------------------
    # GROUP 3 (021-030): Model Capacity
    # -------------------------------------------------------------------------

    exp = deepcopy(base1)
    exp['name'] = '021_d_model_32'
    exp['d_model'] = 32
    exp['dim_feedforward'] = 128
    exp['cnn_channels'] = (16, 32)
    experiments.append(exp)

    exp = deepcopy(base1)
    exp['name'] = '022_d_model_128'
    exp['d_model'] = 128
    exp['nhead'] = 4
    exp['dim_feedforward'] = 512
    exp['cnn_channels'] = (64, 128)
    experiments.append(exp)

    exp = deepcopy(base1)
    exp['name'] = '023_d_model_256'
    exp['d_model'] = 256
    exp['nhead'] = 8
    exp['dim_feedforward'] = 1024
    exp['cnn_channels'] = (128, 256)
    experiments.append(exp)

    exp = deepcopy(base1)
    exp['name'] = '024_d_model_16'
    exp['d_model'] = 16
    exp['nhead'] = 2
    exp['dim_feedforward'] = 64
    exp['cnn_channels'] = (8, 16)
    experiments.append(exp)

    for i, nh in enumerate([1, 4, 8]):
        exp = deepcopy(base1)
        exp['name'] = f'{25+i:03d}_nhead_{nh}'
        exp['nhead'] = nh
        experiments.append(exp)

    exp = deepcopy(base1)
    exp['name'] = '028_d128_nhead_16'
    exp['d_model'] = 128
    exp['nhead'] = 16
    exp['dim_feedforward'] = 512
    exp['cnn_channels'] = (64, 128)
    experiments.append(exp)

    exp = deepcopy(base1)
    exp['name'] = '029_ffn_128'
    exp['dim_feedforward'] = 128
    experiments.append(exp)

    exp = deepcopy(base1)
    exp['name'] = '030_ffn_512'
    exp['dim_feedforward'] = 512
    experiments.append(exp)

    # -------------------------------------------------------------------------
    # GROUP 4 (031-040): Masking Ratio Variations
    # -------------------------------------------------------------------------

    for i, mr in enumerate([0.1, 0.15, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]):
        exp = deepcopy(base1)
        exp['name'] = f'{31+i:03d}_mask_{mr:.2f}'
        exp['masking_ratio'] = mr
        experiments.append(exp)

    exp = deepcopy(base1)
    exp['name'] = '040_feature_wise_mask'
    exp['masking_strategy'] = 'feature_wise'
    experiments.append(exp)

    # -------------------------------------------------------------------------
    # GROUP 5 (041-050): Loss Parameters
    # -------------------------------------------------------------------------

    for i, ld in enumerate([0.1, 0.25, 1.0, 2.0, 3.0]):
        exp = deepcopy(base1)
        exp['name'] = f'{41+i:03d}_lambda_{ld}'
        exp['lambda_disc'] = ld
        experiments.append(exp)

    for i, k in enumerate([1.0, 2.0, 2.5, 3.0, 4.0]):
        exp = deepcopy(base1)
        exp['name'] = f'{46+i:03d}_k_{k}'
        exp['dynamic_margin_k'] = k
        experiments.append(exp)

    # -------------------------------------------------------------------------
    # GROUP 6 (051-060): Training Parameters
    # -------------------------------------------------------------------------

    for i, lr in enumerate([5e-4, 1e-3, 3e-3, 5e-3]):
        exp = deepcopy(base1)
        exp['name'] = f'{51+i:03d}_lr_{lr}'
        exp['learning_rate'] = lr
        experiments.append(exp)

    for i, dp in enumerate([0.0, 0.2, 0.3]):
        exp = deepcopy(base1)
        exp['name'] = f'{55+i:03d}_dropout_{dp}'
        exp['dropout'] = dp
        experiments.append(exp)

    exp = deepcopy(base1)
    exp['name'] = '058_wd_0'
    exp['weight_decay'] = 0
    experiments.append(exp)

    exp = deepcopy(base1)
    exp['name'] = '059_wd_1e-4'
    exp['weight_decay'] = 1e-4
    experiments.append(exp)

    exp = deepcopy(base1)
    exp['name'] = '060_anomaly_weight_2.0'
    exp['anomaly_loss_weight'] = 2.0
    experiments.append(exp)

    # -------------------------------------------------------------------------
    # GROUP 7 (061-070): Combined Optimal Configurations
    # -------------------------------------------------------------------------

    exp = deepcopy(base1)
    exp['name'] = '061_w500_d128'
    exp['seq_length'] = 500
    exp['num_patches'] = 50
    exp['d_model'] = 128
    exp['nhead'] = 4
    exp['dim_feedforward'] = 512
    exp['cnn_channels'] = (64, 128)
    experiments.append(exp)

    exp = deepcopy(base1)
    exp['name'] = '062_w500_d128_t3s1'
    exp['seq_length'] = 500
    exp['num_patches'] = 50
    exp['d_model'] = 128
    exp['nhead'] = 4
    exp['dim_feedforward'] = 512
    exp['cnn_channels'] = (64, 128)
    exp['num_teacher_decoder_layers'] = 3
    experiments.append(exp)

    exp = deepcopy(base1)
    exp['name'] = '063_w500_p20_d128'
    exp['seq_length'] = 500
    exp['patch_size'] = 20
    exp['num_patches'] = 25
    exp['mask_last_n'] = 20
    exp['d_model'] = 128
    exp['nhead'] = 4
    exp['dim_feedforward'] = 512
    exp['cnn_channels'] = (64, 128)
    experiments.append(exp)

    exp = deepcopy(base1)
    exp['name'] = '064_w1000_d128_t3s1'
    exp['seq_length'] = 1000
    exp['patch_size'] = 20
    exp['num_patches'] = 50
    exp['mask_last_n'] = 20
    exp['d_model'] = 128
    exp['nhead'] = 4
    exp['dim_feedforward'] = 512
    exp['cnn_channels'] = (64, 128)
    exp['num_teacher_decoder_layers'] = 3
    experiments.append(exp)

    exp = deepcopy(base1)
    exp['name'] = '065_optimal_v1'
    exp['d_model'] = 128
    exp['nhead'] = 4
    exp['dim_feedforward'] = 512
    exp['cnn_channels'] = (64, 128)
    exp['masking_ratio'] = 0.15
    experiments.append(exp)

    exp = deepcopy(base1)
    exp['name'] = '066_optimal_v2'
    exp['seq_length'] = 500
    exp['num_patches'] = 50
    exp['d_model'] = 128
    exp['nhead'] = 4
    exp['dim_feedforward'] = 512
    exp['cnn_channels'] = (64, 128)
    exp['masking_ratio'] = 0.15
    experiments.append(exp)

    exp = deepcopy(base1)
    exp['name'] = '067_optimal_v3'
    exp['seq_length'] = 500
    exp['patch_size'] = 20
    exp['num_patches'] = 25
    exp['mask_last_n'] = 20
    exp['d_model'] = 128
    exp['nhead'] = 4
    exp['dim_feedforward'] = 512
    exp['cnn_channels'] = (64, 128)
    exp['masking_ratio'] = 0.15
    experiments.append(exp)

    exp = deepcopy(base1)
    exp['name'] = '068_optimal_v4'
    exp['d_model'] = 128
    exp['nhead'] = 4
    exp['dim_feedforward'] = 512
    exp['cnn_channels'] = (64, 128)
    exp['num_teacher_decoder_layers'] = 3
    exp['masking_ratio'] = 0.15
    experiments.append(exp)

    exp = deepcopy(base1)
    exp['name'] = '069_optimal_v5'
    exp['seq_length'] = 500
    exp['num_patches'] = 50
    exp['d_model'] = 128
    exp['nhead'] = 4
    exp['dim_feedforward'] = 512
    exp['cnn_channels'] = (64, 128)
    exp['num_teacher_decoder_layers'] = 3
    exp['masking_ratio'] = 0.15
    experiments.append(exp)

    exp = deepcopy(base1)
    exp['name'] = '070_optimal_final'
    exp['seq_length'] = 500
    exp['patch_size'] = 20
    exp['num_patches'] = 25
    exp['mask_last_n'] = 20
    exp['d_model'] = 128
    exp['nhead'] = 4
    exp['dim_feedforward'] = 512
    exp['cnn_channels'] = (64, 128)
    exp['num_teacher_decoder_layers'] = 3
    exp['masking_ratio'] = 0.15
    experiments.append(exp)

    # =========================================================================
    # ORIGINAL PHASE 2 EXPERIMENTS (071-170)
    # Using BASE_CONFIG_PHASE2 (d_model=128)
    # =========================================================================

    base2 = deepcopy(BASE_CONFIG_PHASE2)

    # -------------------------------------------------------------------------
    # GROUP 8 (071-085): Maximize disc_ratio with d_model=128
    # -------------------------------------------------------------------------

    exp = deepcopy(base2)
    exp['name'] = '071_d128_baseline'
    experiments.append(exp)

    for i, mr in enumerate([0.05, 0.08, 0.10, 0.12, 0.18, 0.20]):
        exp = deepcopy(base2)
        exp['name'] = f'{72+i:03d}_d128_mask_{mr:.2f}'
        exp['masking_ratio'] = mr
        experiments.append(exp)

    for i, nh in enumerate([1, 2, 8, 16, 32]):
        exp = deepcopy(base2)
        exp['name'] = f'{78+i:03d}_d128_nhead_{nh}'
        exp['nhead'] = nh
        experiments.append(exp)

    exp = deepcopy(base2)
    exp['name'] = '083_d128_nhead1_mask0.10'
    exp['nhead'] = 1
    exp['masking_ratio'] = 0.10
    experiments.append(exp)

    exp = deepcopy(base2)
    exp['name'] = '084_d128_nhead16_mask0.10'
    exp['nhead'] = 16
    exp['masking_ratio'] = 0.10
    experiments.append(exp)

    exp = deepcopy(base2)
    exp['name'] = '085_d128_nhead4_mask0.05'
    exp['masking_ratio'] = 0.05
    experiments.append(exp)

    # -------------------------------------------------------------------------
    # GROUP 9 (086-100): Maximize disc_ratio AND t_ratio simultaneously
    # -------------------------------------------------------------------------

    for i, (td, sd) in enumerate([(3, 1), (4, 1), (3, 2), (4, 2)]):
        exp = deepcopy(base2)
        exp['name'] = f'{86+i:03d}_d128_t{td}s{sd}'
        exp['num_teacher_decoder_layers'] = td
        exp['num_student_decoder_layers'] = sd
        experiments.append(exp)

    exp = deepcopy(base2)
    exp['name'] = '090_combo_w500_mr0.10'
    exp['seq_length'] = 500
    exp['num_patches'] = 50
    exp['masking_ratio'] = 0.10
    experiments.append(exp)

    exp = deepcopy(base2)
    exp['name'] = '091_combo_w500_nhead16'
    exp['seq_length'] = 500
    exp['num_patches'] = 50
    exp['nhead'] = 16
    experiments.append(exp)

    exp = deepcopy(base2)
    exp['name'] = '092_combo_w500_t3s1_mr0.10'
    exp['seq_length'] = 500
    exp['num_patches'] = 50
    exp['num_teacher_decoder_layers'] = 3
    exp['masking_ratio'] = 0.10
    experiments.append(exp)

    exp = deepcopy(base2)
    exp['name'] = '093_combo_d128_nhead16_t3s1'
    exp['nhead'] = 16
    exp['num_teacher_decoder_layers'] = 3
    experiments.append(exp)

    exp = deepcopy(base2)
    exp['name'] = '094_combo_d128_nhead4_t3s1_mr0.10'
    exp['num_teacher_decoder_layers'] = 3
    exp['masking_ratio'] = 0.10
    experiments.append(exp)

    exp = deepcopy(base2)
    exp['name'] = '095_d128_t3s1_mask0.10'
    exp['num_teacher_decoder_layers'] = 3
    exp['masking_ratio'] = 0.10
    experiments.append(exp)

    exp = deepcopy(base2)
    exp['name'] = '096_d128_t3s1_nhead1'
    exp['num_teacher_decoder_layers'] = 3
    exp['nhead'] = 1
    experiments.append(exp)

    exp = deepcopy(base2)
    exp['name'] = '097_d128_t3s1_nhead16'
    exp['num_teacher_decoder_layers'] = 3
    exp['nhead'] = 16
    experiments.append(exp)

    exp = deepcopy(base2)
    exp['name'] = '098_d128_enc2_t3s1'
    exp['num_encoder_layers'] = 2
    exp['num_teacher_decoder_layers'] = 3
    experiments.append(exp)

    exp = deepcopy(base2)
    exp['name'] = '099_d128_enc2_t3s1_mask0.10'
    exp['num_encoder_layers'] = 2
    exp['num_teacher_decoder_layers'] = 3
    exp['masking_ratio'] = 0.10
    experiments.append(exp)

    exp = deepcopy(base2)
    exp['name'] = '100_optimal_combo_1'
    exp['nhead'] = 1
    exp['num_teacher_decoder_layers'] = 3
    exp['masking_ratio'] = 0.10
    experiments.append(exp)

    # -------------------------------------------------------------------------
    # GROUP 10 (101-120): Best Model Variations
    # -------------------------------------------------------------------------

    for i, ws in enumerate([200, 500]):
        exp = deepcopy(base2)
        exp['name'] = f'{101+i:03d}_d128_w{ws}'
        exp['seq_length'] = ws
        exp['num_patches'] = ws // 10
        experiments.append(exp)

    exp = deepcopy(base2)
    exp['name'] = '103_combo_w500_p20_mr0.10'
    exp['seq_length'] = 500
    exp['patch_size'] = 20
    exp['num_patches'] = 25
    exp['mask_last_n'] = 20
    exp['masking_ratio'] = 0.10
    experiments.append(exp)

    for i, config_var in enumerate(['baseline', 'nhead1', 'nhead16']):
        exp = deepcopy(base2)
        exp['name'] = f'{104+i:03d}_w500_p20_{config_var}'
        exp['seq_length'] = 500
        exp['patch_size'] = 20
        exp['num_patches'] = 25
        exp['mask_last_n'] = 20
        if config_var == 'nhead1':
            exp['nhead'] = 1
        elif config_var == 'nhead16':
            exp['nhead'] = 16
        experiments.append(exp)

    for i, (td, sd) in enumerate([(2, 1), (3, 1), (3, 2)]):
        exp = deepcopy(base2)
        exp['name'] = f'{107+i:03d}_w500_p20_d128_t{td}s{sd}'
        exp['seq_length'] = 500
        exp['patch_size'] = 20
        exp['num_patches'] = 25
        exp['mask_last_n'] = 20
        exp['num_teacher_decoder_layers'] = td
        exp['num_student_decoder_layers'] = sd
        experiments.append(exp)

    exp = deepcopy(base2)
    exp['name'] = '110_combo_w500_p20_nhead16_mr0.10'
    exp['seq_length'] = 500
    exp['patch_size'] = 20
    exp['num_patches'] = 25
    exp['mask_last_n'] = 20
    exp['nhead'] = 16
    exp['masking_ratio'] = 0.10
    experiments.append(exp)

    exp = deepcopy(base2)
    exp['name'] = '111_combo_d128_nhead16_t3s1_mr0.10'
    exp['nhead'] = 16
    exp['num_teacher_decoder_layers'] = 3
    exp['masking_ratio'] = 0.10
    experiments.append(exp)

    exp = deepcopy(base2)
    exp['name'] = '112_pa80_d128_mr0.10_t4s1'
    exp['masking_ratio'] = 0.10
    exp['num_teacher_decoder_layers'] = 4
    experiments.append(exp)

    for i, ws in enumerate([100, 500]):
        exp = deepcopy(base2)
        exp['name'] = f'{113+i:03d}_nhead1_w{ws}'
        exp['nhead'] = 1
        exp['seq_length'] = ws
        exp['num_patches'] = ws // 10
        experiments.append(exp)

    exp = deepcopy(base2)
    exp['name'] = '115_pa80_d128_nhead16_mr0.10'
    exp['nhead'] = 16
    exp['masking_ratio'] = 0.10
    experiments.append(exp)

    for i, ps in enumerate([5, 20, 25]):
        exp = deepcopy(base2)
        exp['name'] = f'{116+i:03d}_d128_patch{ps}'
        exp['patch_size'] = ps
        exp['num_patches'] = 100 // ps
        exp['mask_last_n'] = ps
        experiments.append(exp)

    exp = deepcopy(base2)
    exp['name'] = '119_pa80_d128_nhead16_t4s1'
    exp['nhead'] = 16
    exp['num_teacher_decoder_layers'] = 4
    experiments.append(exp)

    exp = deepcopy(base2)
    exp['name'] = '120_pa80_d128_nhead16_t4s1_mr0.10'
    exp['nhead'] = 16
    exp['num_teacher_decoder_layers'] = 4
    exp['masking_ratio'] = 0.10
    experiments.append(exp)

    # -------------------------------------------------------------------------
    # GROUP 11 (121-140): Window Size vs Model Depth Relationship
    # -------------------------------------------------------------------------

    for i, (td, sd) in enumerate([(2, 1), (3, 1), (4, 1), (2, 2), (3, 2), (4, 2)]):
        exp = deepcopy(base2)
        exp['name'] = f'{121+i:03d}_w100_t{td}s{sd}'
        exp['num_teacher_decoder_layers'] = td
        exp['num_student_decoder_layers'] = sd
        experiments.append(exp)

    for i, (td, sd) in enumerate([(2, 1), (3, 1), (4, 1), (3, 2)]):
        exp = deepcopy(base2)
        exp['name'] = f'{127+i:03d}_w500_t{td}s{sd}'
        exp['seq_length'] = 500
        exp['num_patches'] = 50
        exp['num_teacher_decoder_layers'] = td
        exp['num_student_decoder_layers'] = sd
        experiments.append(exp)

    for i, dm in enumerate([64, 96, 128, 192]):
        exp = deepcopy(base2)
        exp['name'] = f'{131+i:03d}_w500_d{dm}'
        exp['seq_length'] = 500
        exp['num_patches'] = 50
        exp['d_model'] = dm
        exp['nhead'] = 4 if dm >= 64 else 2
        exp['dim_feedforward'] = dm * 4
        exp['cnn_channels'] = (dm // 2, dm)
        experiments.append(exp)

    for i, mr in enumerate([0.05, 0.10, 0.15, 0.20]):
        exp = deepcopy(base2)
        exp['name'] = f'{135+i:03d}_w500_mask{mr:.2f}'
        exp['seq_length'] = 500
        exp['num_patches'] = 50
        exp['masking_ratio'] = mr
        experiments.append(exp)

    exp = deepcopy(base2)
    exp['name'] = '139_w500_t3s1_mask0.10'
    exp['seq_length'] = 500
    exp['num_patches'] = 50
    exp['num_teacher_decoder_layers'] = 3
    exp['masking_ratio'] = 0.10
    experiments.append(exp)

    exp = deepcopy(base2)
    exp['name'] = '140_w500_t4s2_mask0.10'
    exp['seq_length'] = 500
    exp['num_patches'] = 50
    exp['num_teacher_decoder_layers'] = 4
    exp['num_student_decoder_layers'] = 2
    exp['masking_ratio'] = 0.10
    experiments.append(exp)

    # -------------------------------------------------------------------------
    # GROUP 12 (141-155): PA%80 Optimization Experiments
    # -------------------------------------------------------------------------

    for i, nh in enumerate([1, 8, 16]):
        exp = deepcopy(base2)
        exp['name'] = f'{141+i:03d}_pa80_mr0.10_nhead{nh}'
        exp['masking_ratio'] = 0.10
        exp['nhead'] = nh
        experiments.append(exp)

    for i, (td, sd) in enumerate([(3, 1), (4, 2), (5, 1)]):
        exp = deepcopy(base2)
        exp['name'] = f'{144+i:03d}_pa80_mr0.10_t{td}s{sd}'
        exp['masking_ratio'] = 0.10
        exp['num_teacher_decoder_layers'] = td
        exp['num_student_decoder_layers'] = sd
        experiments.append(exp)

    for i, mr in enumerate([0.08, 0.12, 0.15]):
        exp = deepcopy(base2)
        exp['name'] = f'{147+i:03d}_pa80_nhead16_mr{mr:.2f}'
        exp['nhead'] = 16
        exp['masking_ratio'] = mr
        experiments.append(exp)

    for i, nh in enumerate([1, 4, 8]):
        exp = deepcopy(base2)
        exp['name'] = f'{150+i:03d}_pa80_t4s1_nhead{nh}'
        exp['num_teacher_decoder_layers'] = 4
        exp['nhead'] = nh
        experiments.append(exp)

    exp = deepcopy(base2)
    exp['name'] = '153_pa80_t4s1_nhead16_mr0.08'
    exp['num_teacher_decoder_layers'] = 4
    exp['nhead'] = 16
    exp['masking_ratio'] = 0.08
    experiments.append(exp)

    exp = deepcopy(base2)
    exp['name'] = '154_pa80_t4s2_nhead16_mr0.10'
    exp['num_teacher_decoder_layers'] = 4
    exp['num_student_decoder_layers'] = 2
    exp['nhead'] = 16
    exp['masking_ratio'] = 0.10
    experiments.append(exp)

    exp = deepcopy(base2)
    exp['name'] = '155_pa80_enc2_t4s1_nhead16_mr0.10'
    exp['num_encoder_layers'] = 2
    exp['num_teacher_decoder_layers'] = 4
    exp['nhead'] = 16
    exp['masking_ratio'] = 0.10
    experiments.append(exp)

    # -------------------------------------------------------------------------
    # GROUP 13 (156-170): Additional Explorations
    # -------------------------------------------------------------------------

    exp = deepcopy(base2)
    exp['name'] = '156_d128_lr0.003'
    exp['learning_rate'] = 0.003
    experiments.append(exp)

    exp = deepcopy(base2)
    exp['name'] = '157_combo_d128_nhead4_t3s1_mr0.08'
    exp['num_teacher_decoder_layers'] = 3
    exp['masking_ratio'] = 0.08
    experiments.append(exp)

    exp = deepcopy(base2)
    exp['name'] = '158_d128_lr0.005'
    exp['learning_rate'] = 0.005
    experiments.append(exp)

    exp = deepcopy(base2)
    exp['name'] = '159_combo_d128_nhead16_t4s1_mr0.15'
    exp['nhead'] = 16
    exp['num_teacher_decoder_layers'] = 4
    experiments.append(exp)

    for i, dp in enumerate([0.0, 0.2, 0.3]):
        exp = deepcopy(base2)
        exp['name'] = f'{160+i:03d}_d128_dropout{dp}'
        exp['dropout'] = dp
        experiments.append(exp)

    for i, wd in enumerate([0, 1e-4]):
        exp = deepcopy(base2)
        exp['name'] = f'{163+i:03d}_d128_wd{wd}'
        exp['weight_decay'] = wd
        experiments.append(exp)

    exp = deepcopy(base2)
    exp['name'] = '165_optimal_combo_2'
    exp['nhead'] = 16
    exp['num_teacher_decoder_layers'] = 3
    exp['masking_ratio'] = 0.10
    exp['learning_rate'] = 0.003
    experiments.append(exp)

    exp = deepcopy(base2)
    exp['name'] = '166_optimal_combo_3'
    exp['nhead'] = 1
    exp['num_teacher_decoder_layers'] = 3
    exp['masking_ratio'] = 0.08
    exp['dropout'] = 0.2
    experiments.append(exp)

    exp = deepcopy(base2)
    exp['name'] = '167_optimal_combo_4'
    exp['d_model'] = 192
    exp['nhead'] = 6
    exp['dim_feedforward'] = 768
    exp['cnn_channels'] = (96, 192)
    exp['num_teacher_decoder_layers'] = 3
    exp['masking_ratio'] = 0.10
    experiments.append(exp)

    exp = deepcopy(base2)
    exp['name'] = '168_pa80_final_opt1'
    exp['nhead'] = 16
    exp['num_teacher_decoder_layers'] = 4
    exp['masking_ratio'] = 0.10
    exp['learning_rate'] = 0.003
    experiments.append(exp)

    exp = deepcopy(base2)
    exp['name'] = '169_pa80_final_opt2'
    exp['nhead'] = 16
    exp['num_encoder_layers'] = 2
    exp['num_teacher_decoder_layers'] = 4
    exp['masking_ratio'] = 0.10
    exp['dropout'] = 0.05
    experiments.append(exp)

    exp = deepcopy(base2)
    exp['name'] = '170_pa80_final_optimal'
    exp['d_model'] = 128
    exp['nhead'] = 16
    exp['num_encoder_layers'] = 2
    exp['num_teacher_decoder_layers'] = 4
    exp['num_student_decoder_layers'] = 1
    exp['masking_ratio'] = 0.10
    exp['learning_rate'] = 0.003
    exp['dropout'] = 0.10
    experiments.append(exp)

    return experiments


# Generate experiments list
EXPERIMENTS = get_experiments()
