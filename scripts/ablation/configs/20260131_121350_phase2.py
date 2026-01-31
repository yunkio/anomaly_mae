"""
Phase 2 Ablation Study Configuration
=====================================
Systematic validation of Phase 1 insights with updated defaults.

Created: 2026-01-31
Total Experiments: 150

Based on Phase 1 analysis (1,014 evaluations, 169 base configs).
See docs/ablation/phase2/PHASE2_PLAN.md for full rationale per experiment.

GROUP 1 (001-005): Baseline & Reference
GROUP 2 (006-020): Window Size × Model Capacity
GROUP 3 (021-040): Encoder-Decoder Depth Ratio
GROUP 4 (041-060): Discrepancy Loss Optimization
GROUP 5 (061-070): Margin Type (Softplus vs Dynamic)
GROUP 6 (071-086): Patch Size Interactions
GROUP 7 (087-098): disc_SNR Optimization
GROUP 8 (099-108): Attention Heads & Model Width
GROUP 9 (109-117): Masking Ratio
GROUP 10 (118-127): Student Decoder Depth
GROUP 11 (128-133): Learning Rate
GROUP 12 (134-141): d_model Full Sweep
GROUP 13 (142-150): Combined Optimal Configurations
"""

from copy import deepcopy
from typing import Dict, List

# =============================================================================
# Phase Metadata
# =============================================================================

PHASE_NAME = "phase2"
PHASE_DESCRIPTION = "Phase 2 Ablation: Systematic validation of Phase 1 insights (150 configs)"
CREATED_AT = "2026-01-31 12:13:50"

# =============================================================================
# Scoring and Inference Modes
# =============================================================================

SCORING_MODES = ['default', 'normalized']
MASK_SETTINGS = [True, False]  # mask_after_encoder=True and False

# =============================================================================
# Base Configuration (New Defaults from Phase 1 Analysis)
# =============================================================================

BASE_CONFIG = {
    'force_mask_anomaly': True,
    'margin_type': 'dynamic',
    'mask_after_encoder': False,
    'masking_ratio': 0.15,
    'seq_length': 500,
    'num_patches': 25,
    'patch_size': 20,
    'patch_level_loss': True,
    'patchify_mode': 'patch_cnn',
    'shared_mask_token': False,
    'd_model': 128,
    'nhead': 8,
    'num_encoder_layers': 1,
    'num_teacher_decoder_layers': 4,
    'num_student_decoder_layers': 1,
    'num_shared_decoder_layers': 0,
    'dim_feedforward': 512,
    'dropout': 0.15,
    'cnn_channels': (64, 128),
    'anomaly_loss_weight': 2.0,
    'num_epochs': 50,
    'margin': 0.5,
    'lambda_disc': 2.0,
    'dynamic_margin_k': 2.0,
    'learning_rate': 2e-3,
    'weight_decay': 1e-5,
    'teacher_only_warmup_epochs': 3,
    'warmup_epochs': 10,
}


# =============================================================================
# Helper: set window params consistently
# =============================================================================

def _set_window(exp, w, p):
    """Set window size and patch size, computing num_patches."""
    exp['seq_length'] = w
    exp['patch_size'] = p
    exp['num_patches'] = w // p


def _set_dmodel(exp, d):
    """Set d_model and dim_feedforward = d*4, and cnn_channels."""
    exp['d_model'] = d
    exp['dim_feedforward'] = d * 4
    exp['cnn_channels'] = (d // 2, d)


# =============================================================================
# Experiment Configurations
# =============================================================================

def get_experiments() -> List[Dict]:
    """Define 150 experiment configurations for Phase 2 ablation study."""
    experiments = []
    base = deepcopy(BASE_CONFIG)

    # =========================================================================
    # GROUP 1 (001-005): Baseline & Reference
    # =========================================================================

    # 001: New default baseline
    exp = deepcopy(base)
    exp['name'] = '001_default_baseline'
    experiments.append(exp)

    # 002: Phase 1 old default adapted (enc=2, alw=2, dropout=0.15)
    exp = deepcopy(base)
    exp['name'] = '002_phase1_old_default'
    _set_window(exp, 100, 10)
    _set_dmodel(exp, 64)
    exp['nhead'] = 2
    exp['num_teacher_decoder_layers'] = 2
    exp['learning_rate'] = 0.002
    exp['masking_ratio'] = 0.2
    exp['lambda_disc'] = 0.5
    exp['dynamic_margin_k'] = 1.5
    experiments.append(exp)

    # 003: Phase 1 best roc_auc adapted (151_pa80_t4s1_nhead4)
    exp = deepcopy(base)
    exp['name'] = '003_p1_best_roc'
    _set_window(exp, 100, 10)
    _set_dmodel(exp, 64)
    exp['nhead'] = 4
    exp['num_teacher_decoder_layers'] = 4
    exp['masking_ratio'] = 0.2
    exp['learning_rate'] = 0.002
    exp['lambda_disc'] = 0.5
    exp['dynamic_margin_k'] = 1.5
    experiments.append(exp)

    # 004: Phase 1 best PA80 adapted (166_optimal_combo_3)
    exp = deepcopy(base)
    exp['name'] = '004_p1_best_pa80'
    _set_window(exp, 100, 10)
    _set_dmodel(exp, 64)
    exp['nhead'] = 4
    exp['num_teacher_decoder_layers'] = 3
    exp['masking_ratio'] = 0.1
    exp['learning_rate'] = 0.002
    exp['lambda_disc'] = 0.5
    exp['dynamic_margin_k'] = 1.5
    experiments.append(exp)

    # 005: Default architecture at w100
    exp = deepcopy(base)
    exp['name'] = '005_default_w100'
    _set_window(exp, 100, 20)
    experiments.append(exp)

    # =========================================================================
    # GROUP 2 (006-020): Window Size × Model Capacity
    # =========================================================================

    # 006: w100, p=10
    exp = deepcopy(base)
    exp['name'] = '006_w100_p10'
    _set_window(exp, 100, 10)
    experiments.append(exp)

    # 007: w100, p=5
    exp = deepcopy(base)
    exp['name'] = '007_w100_p5'
    _set_window(exp, 100, 5)
    experiments.append(exp)

    # 008: w100, p=25
    exp = deepcopy(base)
    exp['name'] = '008_w100_p25'
    _set_window(exp, 100, 25)
    experiments.append(exp)

    # 009: w200, p=20
    exp = deepcopy(base)
    exp['name'] = '009_w200_p20'
    _set_window(exp, 200, 20)
    experiments.append(exp)

    # 010: w200, p=10
    exp = deepcopy(base)
    exp['name'] = '010_w200_p10'
    _set_window(exp, 200, 10)
    experiments.append(exp)

    # 011: w200, p=25
    exp = deepcopy(base)
    exp['name'] = '011_w200_p25'
    _set_window(exp, 200, 25)
    experiments.append(exp)

    # 012: w500, d=64
    exp = deepcopy(base)
    exp['name'] = '012_w500_d64'
    _set_dmodel(exp, 64)
    experiments.append(exp)

    # 013: w500, d=256
    exp = deepcopy(base)
    exp['name'] = '013_w500_d256'
    _set_dmodel(exp, 256)
    experiments.append(exp)

    # 014: w500, d=512
    exp = deepcopy(base)
    exp['name'] = '014_w500_d512'
    _set_dmodel(exp, 512)
    experiments.append(exp)

    # 015: w100, d=64
    exp = deepcopy(base)
    exp['name'] = '015_w100_d64'
    _set_window(exp, 100, 20)
    _set_dmodel(exp, 64)
    experiments.append(exp)

    # 016: w100, d=256
    exp = deepcopy(base)
    exp['name'] = '016_w100_d256'
    _set_window(exp, 100, 20)
    _set_dmodel(exp, 256)
    experiments.append(exp)

    # 017: w100, d=512
    exp = deepcopy(base)
    exp['name'] = '017_w100_d512'
    _set_window(exp, 100, 20)
    _set_dmodel(exp, 512)
    experiments.append(exp)

    # 018: w200, d=64
    exp = deepcopy(base)
    exp['name'] = '018_w200_d64'
    _set_window(exp, 200, 20)
    _set_dmodel(exp, 64)
    experiments.append(exp)

    # 019: w200, d=256
    exp = deepcopy(base)
    exp['name'] = '019_w200_d256'
    _set_window(exp, 200, 20)
    _set_dmodel(exp, 256)
    experiments.append(exp)

    # 020: w500, d=256, nh=16
    exp = deepcopy(base)
    exp['name'] = '020_w500_d256_nh16'
    _set_dmodel(exp, 256)
    exp['nhead'] = 16
    experiments.append(exp)

    # =========================================================================
    # GROUP 3 (021-040): Encoder-Decoder Depth Ratio
    # =========================================================================

    # enc=2 series (td varies, enc=2 is default)
    # 021: enc=2, td=2
    exp = deepcopy(base)
    exp['name'] = '021_enc2_td2'
    exp['num_teacher_decoder_layers'] = 2
    experiments.append(exp)

    # 022: enc=2, td=3
    exp = deepcopy(base)
    exp['name'] = '022_enc2_td3'
    exp['num_teacher_decoder_layers'] = 3
    experiments.append(exp)

    # 023: enc=2, td=5
    exp = deepcopy(base)
    exp['name'] = '023_enc2_td5'
    exp['num_teacher_decoder_layers'] = 5
    experiments.append(exp)

    # 024: enc=2, td=6
    exp = deepcopy(base)
    exp['name'] = '024_enc2_td6'
    exp['num_teacher_decoder_layers'] = 6
    experiments.append(exp)

    # enc=3 series
    # 025: enc=3, td=2
    exp = deepcopy(base)
    exp['name'] = '025_enc3_td2'
    exp['num_encoder_layers'] = 3
    exp['num_teacher_decoder_layers'] = 2
    experiments.append(exp)

    # 026: enc=3, td=3
    exp = deepcopy(base)
    exp['name'] = '026_enc3_td3'
    exp['num_encoder_layers'] = 3
    exp['num_teacher_decoder_layers'] = 3
    experiments.append(exp)

    # 027: enc=3, td=4
    exp = deepcopy(base)
    exp['name'] = '027_enc3_td4'
    exp['num_encoder_layers'] = 3
    experiments.append(exp)

    # 028: enc=3, td=5
    exp = deepcopy(base)
    exp['name'] = '028_enc3_td5'
    exp['num_encoder_layers'] = 3
    exp['num_teacher_decoder_layers'] = 5
    experiments.append(exp)

    # 029: enc=3, td=6
    exp = deepcopy(base)
    exp['name'] = '029_enc3_td6'
    exp['num_encoder_layers'] = 3
    exp['num_teacher_decoder_layers'] = 6
    experiments.append(exp)

    # enc=4 series
    # 030: enc=4, td=2
    exp = deepcopy(base)
    exp['name'] = '030_enc4_td2'
    exp['num_encoder_layers'] = 4
    exp['num_teacher_decoder_layers'] = 2
    experiments.append(exp)

    # 031: enc=4, td=3
    exp = deepcopy(base)
    exp['name'] = '031_enc4_td3'
    exp['num_encoder_layers'] = 4
    exp['num_teacher_decoder_layers'] = 3
    experiments.append(exp)

    # 032: enc=4, td=4
    exp = deepcopy(base)
    exp['name'] = '032_enc4_td4'
    exp['num_encoder_layers'] = 4
    experiments.append(exp)

    # 033: enc=4, td=5
    exp = deepcopy(base)
    exp['name'] = '033_enc4_td5'
    exp['num_encoder_layers'] = 4
    exp['num_teacher_decoder_layers'] = 5
    experiments.append(exp)

    # 034: enc=4, td=6
    exp = deepcopy(base)
    exp['name'] = '034_enc4_td6'
    exp['num_encoder_layers'] = 4
    exp['num_teacher_decoder_layers'] = 6
    experiments.append(exp)

    # Depth at w100 (cross-window validation of H6)
    # 035: enc=2, td=2, w=100
    exp = deepcopy(base)
    exp['name'] = '035_w100_enc2_td2'
    _set_window(exp, 100, 20)
    exp['num_teacher_decoder_layers'] = 2
    experiments.append(exp)

    # 036: enc=2, td=4, w=100
    exp = deepcopy(base)
    exp['name'] = '036_w100_enc2_td4'
    _set_window(exp, 100, 20)
    experiments.append(exp)

    # 037: enc=2, td=6, w=100
    exp = deepcopy(base)
    exp['name'] = '037_w100_enc2_td6'
    _set_window(exp, 100, 20)
    exp['num_teacher_decoder_layers'] = 6
    experiments.append(exp)

    # 038: enc=3, td=4, w=100
    exp = deepcopy(base)
    exp['name'] = '038_w100_enc3_td4'
    _set_window(exp, 100, 20)
    exp['num_encoder_layers'] = 3
    experiments.append(exp)

    # 039: enc=4, td=4, w=100
    exp = deepcopy(base)
    exp['name'] = '039_w100_enc4_td4'
    _set_window(exp, 100, 20)
    exp['num_encoder_layers'] = 4
    experiments.append(exp)

    # 040: enc=4, td=2, w=100
    exp = deepcopy(base)
    exp['name'] = '040_w100_enc4_td2'
    _set_window(exp, 100, 20)
    exp['num_encoder_layers'] = 4
    exp['num_teacher_decoder_layers'] = 2
    experiments.append(exp)

    # =========================================================================
    # GROUP 4 (041-060): Discrepancy Loss Optimization
    # =========================================================================

    # Lambda disc sweep
    # 041: λ=0.5
    exp = deepcopy(base)
    exp['name'] = '041_lambda_0.5'
    exp['lambda_disc'] = 0.5
    experiments.append(exp)

    # 042: λ=1.0
    exp = deepcopy(base)
    exp['name'] = '042_lambda_1.0'
    exp['lambda_disc'] = 1.0
    experiments.append(exp)

    # 043: λ=3.0
    exp = deepcopy(base)
    exp['name'] = '043_lambda_3.0'
    exp['lambda_disc'] = 3.0
    experiments.append(exp)

    # 044: λ=4.0
    exp = deepcopy(base)
    exp['name'] = '044_lambda_4.0'
    exp['lambda_disc'] = 4.0
    experiments.append(exp)

    # 045: λ=5.0
    exp = deepcopy(base)
    exp['name'] = '045_lambda_5.0'
    exp['lambda_disc'] = 5.0
    experiments.append(exp)

    # Dynamic margin k sweep
    # 046: k=1.0
    exp = deepcopy(base)
    exp['name'] = '046_k_1.0'
    exp['dynamic_margin_k'] = 1.0
    experiments.append(exp)

    # 047: k=1.5
    exp = deepcopy(base)
    exp['name'] = '047_k_1.5'
    exp['dynamic_margin_k'] = 1.5
    experiments.append(exp)

    # 048: k=3.0
    exp = deepcopy(base)
    exp['name'] = '048_k_3.0'
    exp['dynamic_margin_k'] = 3.0
    experiments.append(exp)

    # 049: k=4.0
    exp = deepcopy(base)
    exp['name'] = '049_k_4.0'
    exp['dynamic_margin_k'] = 4.0
    experiments.append(exp)

    # 050: k=5.0
    exp = deepcopy(base)
    exp['name'] = '050_k_5.0'
    exp['dynamic_margin_k'] = 5.0
    experiments.append(exp)

    # Combined λ × k
    # 051: λ=3.0, k=3.0
    exp = deepcopy(base)
    exp['name'] = '051_lambda3_k3'
    exp['lambda_disc'] = 3.0
    exp['dynamic_margin_k'] = 3.0
    experiments.append(exp)

    # 052: λ=4.0, k=3.0
    exp = deepcopy(base)
    exp['name'] = '052_lambda4_k3'
    exp['lambda_disc'] = 4.0
    exp['dynamic_margin_k'] = 3.0
    experiments.append(exp)

    # 053: λ=3.0, k=4.0
    exp = deepcopy(base)
    exp['name'] = '053_lambda3_k4'
    exp['lambda_disc'] = 3.0
    exp['dynamic_margin_k'] = 4.0
    experiments.append(exp)

    # 054: λ=1.0, k=1.0 (minimal disc)
    exp = deepcopy(base)
    exp['name'] = '054_lambda1_k1_minimal'
    exp['lambda_disc'] = 1.0
    exp['dynamic_margin_k'] = 1.0
    experiments.append(exp)

    # Anomaly loss weight sweep
    # 055: alw=3
    exp = deepcopy(base)
    exp['name'] = '055_alw3'
    exp['anomaly_loss_weight'] = 3.0
    experiments.append(exp)

    # 056: alw=4
    exp = deepcopy(base)
    exp['name'] = '056_alw4'
    exp['anomaly_loss_weight'] = 4.0
    experiments.append(exp)

    # 057: alw=5
    exp = deepcopy(base)
    exp['name'] = '057_alw5'
    exp['anomaly_loss_weight'] = 5.0
    experiments.append(exp)

    # 058: λ=3.0, k=3.0, alw=3 (all boosted)
    exp = deepcopy(base)
    exp['name'] = '058_all_disc_boosted'
    exp['lambda_disc'] = 3.0
    exp['dynamic_margin_k'] = 3.0
    exp['anomaly_loss_weight'] = 3.0
    experiments.append(exp)

    # 059: λ=0.5, k=1.0 (disc-minimal at w500)
    exp = deepcopy(base)
    exp['name'] = '059_disc_minimal'
    exp['lambda_disc'] = 0.5
    exp['dynamic_margin_k'] = 1.0
    experiments.append(exp)

    # 060: λ=3.0, k=3.0, w=100 (strong disc at small window)
    exp = deepcopy(base)
    exp['name'] = '060_w100_disc_strong'
    _set_window(exp, 100, 20)
    exp['lambda_disc'] = 3.0
    exp['dynamic_margin_k'] = 3.0
    experiments.append(exp)

    # =========================================================================
    # GROUP 5 (061-070): Margin Type — Softplus vs Dynamic
    # =========================================================================

    # 061: softplus default
    exp = deepcopy(base)
    exp['name'] = '061_softplus'
    exp['margin_type'] = 'softplus'
    experiments.append(exp)

    # 062: softplus w100
    exp = deepcopy(base)
    exp['name'] = '062_softplus_w100'
    exp['margin_type'] = 'softplus'
    _set_window(exp, 100, 20)
    experiments.append(exp)

    # 063: softplus w200
    exp = deepcopy(base)
    exp['name'] = '063_softplus_w200'
    exp['margin_type'] = 'softplus'
    _set_window(exp, 200, 20)
    experiments.append(exp)

    # 064: softplus td=2
    exp = deepcopy(base)
    exp['name'] = '064_softplus_td2'
    exp['margin_type'] = 'softplus'
    exp['num_teacher_decoder_layers'] = 2
    experiments.append(exp)

    # 065: softplus td=6
    exp = deepcopy(base)
    exp['name'] = '065_softplus_td6'
    exp['margin_type'] = 'softplus'
    exp['num_teacher_decoder_layers'] = 6
    experiments.append(exp)

    # 066: softplus d=64
    exp = deepcopy(base)
    exp['name'] = '066_softplus_d64'
    exp['margin_type'] = 'softplus'
    _set_dmodel(exp, 64)
    experiments.append(exp)

    # 067: softplus d=256
    exp = deepcopy(base)
    exp['name'] = '067_softplus_d256'
    exp['margin_type'] = 'softplus'
    _set_dmodel(exp, 256)
    experiments.append(exp)

    # 068: softplus λ=0.5
    exp = deepcopy(base)
    exp['name'] = '068_softplus_lambda0.5'
    exp['margin_type'] = 'softplus'
    exp['lambda_disc'] = 0.5
    experiments.append(exp)

    # 069: softplus λ=4.0
    exp = deepcopy(base)
    exp['name'] = '069_softplus_lambda4'
    exp['margin_type'] = 'softplus'
    exp['lambda_disc'] = 4.0
    experiments.append(exp)

    # 070: softplus enc=3, td=5
    exp = deepcopy(base)
    exp['name'] = '070_softplus_enc3_td5'
    exp['margin_type'] = 'softplus'
    exp['num_encoder_layers'] = 3
    exp['num_teacher_decoder_layers'] = 5
    experiments.append(exp)

    # =========================================================================
    # GROUP 6 (071-086): Patch Size Interactions
    # =========================================================================

    # Patch × Window at w500
    # 071: p=5, w=500 (100 patches)
    exp = deepcopy(base)
    exp['name'] = '071_w500_p5'
    _set_window(exp, 500, 5)
    experiments.append(exp)

    # 072: p=10, w=500 (50 patches)
    exp = deepcopy(base)
    exp['name'] = '072_w500_p10'
    _set_window(exp, 500, 10)
    experiments.append(exp)

    # 073: p=25, w=500 (20 patches)
    exp = deepcopy(base)
    exp['name'] = '073_w500_p25'
    _set_window(exp, 500, 25)
    experiments.append(exp)

    # Patch × Window at w100
    # 074: p=5, w=100 (20 patches)
    exp = deepcopy(base)
    exp['name'] = '074_w100_p5'
    _set_window(exp, 100, 5)
    experiments.append(exp)

    # 075: p=10, w=100 (10 patches)
    exp = deepcopy(base)
    exp['name'] = '075_w100_p10'
    _set_window(exp, 100, 10)
    experiments.append(exp)

    # 076: p=25, w=100 (4 patches)
    exp = deepcopy(base)
    exp['name'] = '076_w100_p25'
    _set_window(exp, 100, 25)
    experiments.append(exp)

    # Patch × Window at w200
    # 077: p=5, w=200 (40 patches)
    exp = deepcopy(base)
    exp['name'] = '077_w200_p5'
    _set_window(exp, 200, 5)
    experiments.append(exp)

    # 078: p=10, w=200 (20 patches)
    exp = deepcopy(base)
    exp['name'] = '078_w200_p10'
    _set_window(exp, 200, 10)
    experiments.append(exp)

    # 079: p=5, d=256 (many patches + large model)
    exp = deepcopy(base)
    exp['name'] = '079_w500_p5_d256'
    _set_window(exp, 500, 5)
    _set_dmodel(exp, 256)
    experiments.append(exp)

    # 080: p=10, d=256
    exp = deepcopy(base)
    exp['name'] = '080_w500_p10_d256'
    _set_window(exp, 500, 10)
    _set_dmodel(exp, 256)
    experiments.append(exp)

    # 081: p=5, d=64 (many patches + small model)
    exp = deepcopy(base)
    exp['name'] = '081_w500_p5_d64'
    _set_window(exp, 500, 5)
    _set_dmodel(exp, 64)
    experiments.append(exp)

    # 082: p=25, d=64
    exp = deepcopy(base)
    exp['name'] = '082_w500_p25_d64'
    _set_window(exp, 500, 25)
    _set_dmodel(exp, 64)
    experiments.append(exp)

    # Patch × Decoder depth
    # 083: p=5, td=2
    exp = deepcopy(base)
    exp['name'] = '083_w500_p5_td2'
    _set_window(exp, 500, 5)
    exp['num_teacher_decoder_layers'] = 2
    experiments.append(exp)

    # 084: p=5, td=6
    exp = deepcopy(base)
    exp['name'] = '084_w500_p5_td6'
    _set_window(exp, 500, 5)
    exp['num_teacher_decoder_layers'] = 6
    experiments.append(exp)

    # 085: p=25, td=2
    exp = deepcopy(base)
    exp['name'] = '085_w500_p25_td2'
    _set_window(exp, 500, 25)
    exp['num_teacher_decoder_layers'] = 2
    experiments.append(exp)

    # 086: p=25, td=6
    exp = deepcopy(base)
    exp['name'] = '086_w500_p25_td6'
    _set_window(exp, 500, 25)
    exp['num_teacher_decoder_layers'] = 6
    experiments.append(exp)

    # =========================================================================
    # GROUP 7 (087-098): disc_SNR Optimization
    # =========================================================================

    # 087: mr=0.1, nh=4
    exp = deepcopy(base)
    exp['name'] = '087_snr_mr01_nh4'
    exp['masking_ratio'] = 0.1
    exp['nhead'] = 4
    experiments.append(exp)

    # 088: mr=0.1, nh=16
    exp = deepcopy(base)
    exp['name'] = '088_snr_mr01_nh16'
    exp['masking_ratio'] = 0.1
    exp['nhead'] = 16
    experiments.append(exp)

    # 089: mr=0.1, d=256
    exp = deepcopy(base)
    exp['name'] = '089_snr_mr01_d256'
    exp['masking_ratio'] = 0.1
    _set_dmodel(exp, 256)
    experiments.append(exp)

    # 090: mr=0.1, d=256, nh=16
    exp = deepcopy(base)
    exp['name'] = '090_snr_mr01_d256_nh16'
    exp['masking_ratio'] = 0.1
    _set_dmodel(exp, 256)
    exp['nhead'] = 16
    experiments.append(exp)

    # 091: mr=0.1, td=5
    exp = deepcopy(base)
    exp['name'] = '091_snr_mr01_td5'
    exp['masking_ratio'] = 0.1
    exp['num_teacher_decoder_layers'] = 5
    experiments.append(exp)

    # 092: mr=0.1, td=6
    exp = deepcopy(base)
    exp['name'] = '092_snr_mr01_td6'
    exp['masking_ratio'] = 0.1
    exp['num_teacher_decoder_layers'] = 6
    experiments.append(exp)

    # 093: mr=0.1, enc=3, td=5
    exp = deepcopy(base)
    exp['name'] = '093_snr_mr01_enc3_td5'
    exp['masking_ratio'] = 0.1
    exp['num_encoder_layers'] = 3
    exp['num_teacher_decoder_layers'] = 5
    experiments.append(exp)

    # 094: mr=0.1, d=256, td=5
    exp = deepcopy(base)
    exp['name'] = '094_snr_mr01_d256_td5'
    exp['masking_ratio'] = 0.1
    _set_dmodel(exp, 256)
    exp['num_teacher_decoder_layers'] = 5
    experiments.append(exp)

    # 095: d=256, nh=16 (default mr=0.15)
    exp = deepcopy(base)
    exp['name'] = '095_snr_d256_nh16'
    _set_dmodel(exp, 256)
    exp['nhead'] = 16
    experiments.append(exp)

    # 096: mr=0.1, w=100
    exp = deepcopy(base)
    exp['name'] = '096_snr_mr01_w100'
    exp['masking_ratio'] = 0.1
    _set_window(exp, 100, 20)
    experiments.append(exp)

    # 097: mr=0.1, w=200
    exp = deepcopy(base)
    exp['name'] = '097_snr_mr01_w200'
    exp['masking_ratio'] = 0.1
    _set_window(exp, 200, 20)
    experiments.append(exp)

    # 098: lr=0.003, mr=0.1
    exp = deepcopy(base)
    exp['name'] = '098_snr_lr003_mr01'
    exp['masking_ratio'] = 0.1
    exp['learning_rate'] = 0.003
    experiments.append(exp)

    # =========================================================================
    # GROUP 8 (099-108): Attention Heads & Model Width
    # =========================================================================

    # 099: nh=4
    exp = deepcopy(base)
    exp['name'] = '099_nh4'
    exp['nhead'] = 4
    experiments.append(exp)

    # 100: nh=16
    exp = deepcopy(base)
    exp['name'] = '100_nh16'
    exp['nhead'] = 16
    experiments.append(exp)

    # 101: nh=4, d=64
    exp = deepcopy(base)
    exp['name'] = '101_nh4_d64'
    exp['nhead'] = 4
    _set_dmodel(exp, 64)
    experiments.append(exp)

    # 102: nh=4, d=256
    exp = deepcopy(base)
    exp['name'] = '102_nh4_d256'
    exp['nhead'] = 4
    _set_dmodel(exp, 256)
    experiments.append(exp)

    # 103: nh=16, d=64
    exp = deepcopy(base)
    exp['name'] = '103_nh16_d64'
    exp['nhead'] = 16
    _set_dmodel(exp, 64)
    experiments.append(exp)

    # 104: nh=16, d=256
    exp = deepcopy(base)
    exp['name'] = '104_nh16_d256'
    exp['nhead'] = 16
    _set_dmodel(exp, 256)
    experiments.append(exp)

    # 105: nh=4, d=512
    exp = deepcopy(base)
    exp['name'] = '105_nh4_d512'
    exp['nhead'] = 4
    _set_dmodel(exp, 512)
    experiments.append(exp)

    # 106: nh=16, d=512
    exp = deepcopy(base)
    exp['name'] = '106_nh16_d512'
    exp['nhead'] = 16
    _set_dmodel(exp, 512)
    experiments.append(exp)

    # 107: nh=8, w=100 (same as 005, cross-ref)
    exp = deepcopy(base)
    exp['name'] = '107_nh8_w100'
    _set_window(exp, 100, 20)
    experiments.append(exp)

    # 108: nh=4, w=100, p=10
    exp = deepcopy(base)
    exp['name'] = '108_nh4_w100_p10'
    exp['nhead'] = 4
    _set_window(exp, 100, 10)
    experiments.append(exp)

    # =========================================================================
    # GROUP 9 (109-117): Masking Ratio
    # =========================================================================

    # 109: mr=0.1
    exp = deepcopy(base)
    exp['name'] = '109_mr01'
    exp['masking_ratio'] = 0.1
    experiments.append(exp)

    # 110: mr=0.2
    exp = deepcopy(base)
    exp['name'] = '110_mr02'
    exp['masking_ratio'] = 0.2
    experiments.append(exp)

    # 111: mr=0.1, w=100
    exp = deepcopy(base)
    exp['name'] = '111_mr01_w100'
    exp['masking_ratio'] = 0.1
    _set_window(exp, 100, 20)
    experiments.append(exp)

    # 112: mr=0.2, w=100
    exp = deepcopy(base)
    exp['name'] = '112_mr02_w100'
    exp['masking_ratio'] = 0.2
    _set_window(exp, 100, 20)
    experiments.append(exp)

    # 113: mr=0.1, w=200
    exp = deepcopy(base)
    exp['name'] = '113_mr01_w200'
    exp['masking_ratio'] = 0.1
    _set_window(exp, 200, 20)
    experiments.append(exp)

    # 114: mr=0.2, w=200
    exp = deepcopy(base)
    exp['name'] = '114_mr02_w200'
    exp['masking_ratio'] = 0.2
    _set_window(exp, 200, 20)
    experiments.append(exp)

    # 115: mr=0.1, td=3
    exp = deepcopy(base)
    exp['name'] = '115_mr01_td3'
    exp['masking_ratio'] = 0.1
    exp['num_teacher_decoder_layers'] = 3
    experiments.append(exp)

    # 116: mr=0.2, td=3
    exp = deepcopy(base)
    exp['name'] = '116_mr02_td3'
    exp['masking_ratio'] = 0.2
    exp['num_teacher_decoder_layers'] = 3
    experiments.append(exp)

    # 117: mr=0.1, td=5
    exp = deepcopy(base)
    exp['name'] = '117_mr01_td5'
    exp['masking_ratio'] = 0.1
    exp['num_teacher_decoder_layers'] = 5
    experiments.append(exp)

    # =========================================================================
    # GROUP 10 (118-127): Student Decoder Depth
    # =========================================================================

    # 118: sd=2
    exp = deepcopy(base)
    exp['name'] = '118_sd2'
    exp['num_student_decoder_layers'] = 2
    experiments.append(exp)

    # 119: sd=3
    exp = deepcopy(base)
    exp['name'] = '119_sd3'
    exp['num_student_decoder_layers'] = 3
    experiments.append(exp)

    # 120: sd=4
    exp = deepcopy(base)
    exp['name'] = '120_sd4'
    exp['num_student_decoder_layers'] = 4
    experiments.append(exp)

    # 121: sd=2, td=5
    exp = deepcopy(base)
    exp['name'] = '121_sd2_td5'
    exp['num_student_decoder_layers'] = 2
    exp['num_teacher_decoder_layers'] = 5
    experiments.append(exp)

    # 122: sd=3, td=5
    exp = deepcopy(base)
    exp['name'] = '122_sd3_td5'
    exp['num_student_decoder_layers'] = 3
    exp['num_teacher_decoder_layers'] = 5
    experiments.append(exp)

    # 123: sd=4, td=5
    exp = deepcopy(base)
    exp['name'] = '123_sd4_td5'
    exp['num_student_decoder_layers'] = 4
    exp['num_teacher_decoder_layers'] = 5
    experiments.append(exp)

    # 124: sd=2, td=6
    exp = deepcopy(base)
    exp['name'] = '124_sd2_td6'
    exp['num_student_decoder_layers'] = 2
    exp['num_teacher_decoder_layers'] = 6
    experiments.append(exp)

    # 125: sd=4, td=6
    exp = deepcopy(base)
    exp['name'] = '125_sd4_td6'
    exp['num_student_decoder_layers'] = 4
    exp['num_teacher_decoder_layers'] = 6
    experiments.append(exp)

    # 126: sd=2, td=2 (equal depth)
    exp = deepcopy(base)
    exp['name'] = '126_sd2_td2'
    exp['num_student_decoder_layers'] = 2
    exp['num_teacher_decoder_layers'] = 2
    experiments.append(exp)

    # 127: sd=2, w=100
    exp = deepcopy(base)
    exp['name'] = '127_sd2_w100'
    exp['num_student_decoder_layers'] = 2
    _set_window(exp, 100, 20)
    experiments.append(exp)

    # =========================================================================
    # GROUP 11 (128-133): Learning Rate
    # =========================================================================

    # 128: lr=0.003
    exp = deepcopy(base)
    exp['name'] = '128_lr003'
    exp['learning_rate'] = 0.003
    experiments.append(exp)

    # 129: lr=0.002
    exp = deepcopy(base)
    exp['name'] = '129_lr002'
    exp['learning_rate'] = 0.002
    experiments.append(exp)

    # 130: lr=0.001
    exp = deepcopy(base)
    exp['name'] = '130_lr001'
    exp['learning_rate'] = 0.001
    experiments.append(exp)

    # 131: lr=0.008
    exp = deepcopy(base)
    exp['name'] = '131_lr008'
    exp['learning_rate'] = 0.008
    experiments.append(exp)

    # 132: lr=0.01
    exp = deepcopy(base)
    exp['name'] = '132_lr010'
    exp['learning_rate'] = 0.01
    experiments.append(exp)

    # 133: lr=0.003, w=100
    exp = deepcopy(base)
    exp['name'] = '133_lr003_w100'
    exp['learning_rate'] = 0.003
    _set_window(exp, 100, 20)
    experiments.append(exp)

    # =========================================================================
    # GROUP 12 (134-141): d_model Full Sweep
    # =========================================================================

    # 134: d=64, nh=8
    exp = deepcopy(base)
    exp['name'] = '134_d64_nh8'
    _set_dmodel(exp, 64)
    experiments.append(exp)

    # 135: d=64, nh=4
    exp = deepcopy(base)
    exp['name'] = '135_d64_nh4'
    _set_dmodel(exp, 64)
    exp['nhead'] = 4
    experiments.append(exp)

    # 136: d=64, nh=16
    exp = deepcopy(base)
    exp['name'] = '136_d64_nh16'
    _set_dmodel(exp, 64)
    exp['nhead'] = 16
    experiments.append(exp)

    # 137: d=256, nh=8
    exp = deepcopy(base)
    exp['name'] = '137_d256_nh8'
    _set_dmodel(exp, 256)
    experiments.append(exp)

    # 138: d=512, nh=8
    exp = deepcopy(base)
    exp['name'] = '138_d512_nh8'
    _set_dmodel(exp, 512)
    experiments.append(exp)

    # 139: d=512, nh=16
    exp = deepcopy(base)
    exp['name'] = '139_d512_nh16'
    _set_dmodel(exp, 512)
    exp['nhead'] = 16
    experiments.append(exp)

    # 140: d=512, nh=4
    exp = deepcopy(base)
    exp['name'] = '140_d512_nh4'
    _set_dmodel(exp, 512)
    exp['nhead'] = 4
    experiments.append(exp)

    # 141: d=256, nh=4
    exp = deepcopy(base)
    exp['name'] = '141_d256_nh4'
    _set_dmodel(exp, 256)
    exp['nhead'] = 4
    experiments.append(exp)

    # =========================================================================
    # GROUP 13 (142-150): Combined Optimal Configurations
    # =========================================================================

    # 142: d=256, td=5, mr=0.1
    exp = deepcopy(base)
    exp['name'] = '142_combo_d256_td5_mr01'
    _set_dmodel(exp, 256)
    exp['num_teacher_decoder_layers'] = 5
    exp['masking_ratio'] = 0.1
    experiments.append(exp)

    # 143: enc=3, td=3, mr=0.1
    exp = deepcopy(base)
    exp['name'] = '143_combo_enc3_td3_mr01'
    exp['num_encoder_layers'] = 3
    exp['num_teacher_decoder_layers'] = 3
    exp['masking_ratio'] = 0.1
    experiments.append(exp)

    # 144: w500 disturbing specialist: d=256, nh=16
    exp = deepcopy(base)
    exp['name'] = '144_combo_disturb_d256_nh16'
    _set_dmodel(exp, 256)
    exp['nhead'] = 16
    experiments.append(exp)

    # 145: disc-recon balance: λ=1.0, sd=2
    exp = deepcopy(base)
    exp['name'] = '145_combo_balance_lambda1_sd2'
    exp['lambda_disc'] = 1.0
    exp['num_student_decoder_layers'] = 2
    experiments.append(exp)

    # 146: maximum depth: enc=4, td=6, sd=4
    exp = deepcopy(base)
    exp['name'] = '146_combo_max_depth'
    exp['num_encoder_layers'] = 4
    exp['num_teacher_decoder_layers'] = 6
    exp['num_student_decoder_layers'] = 4
    experiments.append(exp)

    # 147: minimum config: w=100, d=64, nh=4, td=2
    exp = deepcopy(base)
    exp['name'] = '147_combo_minimum'
    _set_window(exp, 100, 20)
    _set_dmodel(exp, 64)
    exp['nhead'] = 4
    exp['num_teacher_decoder_layers'] = 2
    experiments.append(exp)

    # 148: w100 optimized: p=25, nh=4, td=3, mr=0.1
    exp = deepcopy(base)
    exp['name'] = '148_combo_w100_opt'
    _set_window(exp, 100, 25)
    exp['nhead'] = 4
    exp['num_teacher_decoder_layers'] = 3
    exp['masking_ratio'] = 0.1
    experiments.append(exp)

    # 149: w200 optimized: p=25, td=3, mr=0.1
    exp = deepcopy(base)
    exp['name'] = '149_combo_w200_opt'
    _set_window(exp, 200, 25)
    exp['num_teacher_decoder_layers'] = 3
    exp['masking_ratio'] = 0.1
    experiments.append(exp)

    # 150: high-capacity SNR target: d=512, nh=16, enc=3, td=5, mr=0.1
    exp = deepcopy(base)
    exp['name'] = '150_combo_max_snr'
    _set_dmodel(exp, 512)
    exp['nhead'] = 16
    exp['num_encoder_layers'] = 3
    exp['num_teacher_decoder_layers'] = 5
    exp['masking_ratio'] = 0.1
    experiments.append(exp)

    # =========================================================================
    # Validation
    # =========================================================================
    assert len(experiments) == 150, f"Expected 150 experiments, got {len(experiments)}"

    # Validate names are unique
    names = [e['name'] for e in experiments]
    assert len(names) == len(set(names)), f"Duplicate experiment names found!"

    return experiments


# =============================================================================
# Required interface for run_ablation.py
# =============================================================================

EXPERIMENTS = get_experiments()
