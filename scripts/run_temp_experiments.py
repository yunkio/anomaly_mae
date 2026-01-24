"""
TEMP Ablation Study: Architecture and Loss Parameter Experiments

This script runs experiments to analyze the impact of various architecture and loss
parameters on normal vs anomaly discrepancy.

PRIMARY GOAL: Maximize anomaly/normal discrepancy ratio

Experiments 1-21: Basic architecture and loss variations (user-specified)
- 01: Default (base)
- 02: Shared decoder (1 shared + 3 teacher + 1 student)
- 03: Window size 200, num_patches 20
- 04: Window size 500, num_patches 50
- 05-06: Encoder 1/3 layers
- 07-10: Decoder variations (t4s2, t3s2, t3s1, t2s1)
- 11-14: d_model variations (16, 8, 64, 4)
- 15-16: CNN variations (small, large)
- 17: Window level loss (patch_level_loss=False)
- 18: Student reconstruction loss
- 19-21: Anomaly loss weight 2x/3x/5x

Experiments 22-27: Masking Ratio (0.1 intervals)
- 22-27: masking_ratio 0.2/0.3/0.4/0.6/0.7/0.8

Experiments 28-30: Lambda_disc (reduced)
- 28-30: lambda_disc 0.1/1.0/2.0

Experiments 31-35: Dynamic Margin k
- 31-35: dynamic_margin_k 1.0/1.5/2.0/4.0/5.0

Experiment 36: Feature-wise Masking (single)

Experiments 37-40: Attention Heads
- 37-40: nhead 1/2/8, d64_nhead8

Experiments 41-44: FFN Variations
- 41-44: dim_feedforward 64/256/512/1024

Experiments 45-48: Learning Parameters
- 45-48: lr 5e-4/2e-3, dropout 0.0/0.2

Experiments 49-51: Margin Types
- 49-51: hinge/softplus/dynamic_k4

Experiments 52-53: Larger d_model
- 52-53: d_model 128/256

Experiments 54-55: Patch Size Variations
- 54: patch_size=5, num_patches=20
- 55: patch_size=20, num_patches=5

Experiments 56-57: Large Windows
- 56-57: window 1000/2000

Experiments 58-63: Window + Patch Combinations
- Various combinations of window size and patch size

Experiments 64-65: Capacity + Context Combinations
- 64: d_model=128 + window=500
- 65: patch_size=5 + window=500

Experiments 66-68: Model Depth Variations
- 66: encoder=4
- 67: d_model=128 + nhead=8
- 68: shallow_all (enc=1, teacher=2, student=1)

Experiments 69-70: Training Strategy
- 69: warmup_epochs=5
- 70: lr=1e-4, weight_decay=1e-4

Experiments 71-110: PHASE 2 - NEW COMPREHENSIVE OPTIMIZATION
Completely redesigned based on 12-step sequential thinking analysis of Phase 1.

Key Phase 1 findings:
- masking_ratio=0.2 is optimal (ROC-AUC=0.978, disc_ratio=2.11, t_ratio=8.50)
- d_model=64 is sweet spot (ROC-AUC=0.970)
- t2s1 decoder outperforms deep decoder (t_ratio=7.16 vs 6.70)
- t_ratio has 0.773 correlation with ROC-AUC (strongest predictor)
- num_patches ≤ 50 rule is critical
- dynamic_margin_k=1.5 is optimal

GROUP A (71-78): Ultra-Low Masking Exploration - 8 experiments
- masking=0.1, 0.15, 0.25 with d_model=64 and t2s1

GROUP B (79-86): Triple Optimal Combinations - 8 experiments
- Combine masking + d_model + decoder depth + margin

GROUP C (87-96): Large Window + Low Masking [USER PRIORITY] - 10 experiments
- window=500/1000 with patch_size=10 (user request)
- Also includes safe patch_size=20 variants

GROUP D (97-102): Scoring Modes - 6 experiments
- Test normalized, adaptive scoring modes

GROUP E (103-108): Full Optimal Combinations - 6 experiments
- Combine ALL optimal factors (EXPECTED TOP performers)

GROUP F (109-110): Edge Cases & Validation - 2 experiments
- d_model=48, t1s1 decoder

Base configuration:
- epochs: 50, warmup: 3
- d_model: 32, nhead: 4
- window: 100, patches: 10, patch_size: 10
- margin_type: dynamic, dynamic_margin_k: 3.0
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
        'dropout': 0.1,
        'cnn_channels': None,  # Default: (d_model//2, d_model) = (16, 32)
        'use_student_reconstruction_loss': False,
        'anomaly_loss_weight': 1.0,
        'num_epochs': 30,
        'mask_last_n': 10,  # patch_size
        # Loss parameters
        'margin': 0.5,
        'lambda_disc': 0.5,
        'dynamic_margin_k': 3.0,
        # Training parameters
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'teacher_only_warmup_epochs': 3,
        'warmup_epochs': 10,
    }


def get_experiment_configs() -> List[Dict]:
    """Define 100 experiment configurations (1-70 base, 71-100 lambda_disc scaling)"""
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
    exp3['mask_last_n'] = 10  # Always 1 patch (patch_size)
    experiments.append(exp3)

    # 4. Window size 500 (patch_size=10, num_patches=50)
    exp4 = deepcopy(base)
    exp4['name'] = '04_window_500'
    exp4['seq_length'] = 500
    exp4['num_patches'] = 50
    exp4['patch_size'] = 10
    exp4['mask_last_n'] = 10  # Always 1 patch (patch_size)
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

    # =========================================================================
    # Experiments 22-27: Masking Ratio Variations (0.1 intervals, include lower)
    # Goal: Test impact of masking difficulty on discrepancy
    # =========================================================================

    # 22. Masking ratio 0.2 (very low - easy task)
    exp22 = deepcopy(base)
    exp22['name'] = '22_masking_ratio_0.2'
    exp22['masking_ratio'] = 0.2
    experiments.append(exp22)

    # 23. Masking ratio 0.3
    exp23 = deepcopy(base)
    exp23['name'] = '23_masking_ratio_0.3'
    exp23['masking_ratio'] = 0.3
    experiments.append(exp23)

    # 24. Masking ratio 0.4
    exp24 = deepcopy(base)
    exp24['name'] = '24_masking_ratio_0.4'
    exp24['masking_ratio'] = 0.4
    experiments.append(exp24)

    # 25. Masking ratio 0.6
    exp25 = deepcopy(base)
    exp25['name'] = '25_masking_ratio_0.6'
    exp25['masking_ratio'] = 0.6
    experiments.append(exp25)

    # 26. Masking ratio 0.7
    exp26 = deepcopy(base)
    exp26['name'] = '26_masking_ratio_0.7'
    exp26['masking_ratio'] = 0.7
    experiments.append(exp26)

    # 27. Masking ratio 0.8
    exp27 = deepcopy(base)
    exp27['name'] = '27_masking_ratio_0.8'
    exp27['masking_ratio'] = 0.8
    experiments.append(exp27)

    # =========================================================================
    # Experiments 28-30: Lambda_disc Variations (reduced)
    # Goal: Find optimal discrepancy loss weight
    # =========================================================================

    # 28. Lambda_disc 0.1 (very weak)
    exp28 = deepcopy(base)
    exp28['name'] = '28_lambda_disc_0.1'
    exp28['lambda_disc'] = 0.1
    experiments.append(exp28)

    # 29. Lambda_disc 1.0 (equal weight)
    exp29 = deepcopy(base)
    exp29['name'] = '29_lambda_disc_1.0'
    exp29['lambda_disc'] = 1.0
    experiments.append(exp29)

    # 30. Lambda_disc 2.0 (strong)
    exp30 = deepcopy(base)
    exp30['name'] = '30_lambda_disc_2.0'
    exp30['lambda_disc'] = 2.0
    experiments.append(exp30)

    # =========================================================================
    # Experiments 31-35: Dynamic Margin k Variations
    # Goal: Find optimal k for dynamic margin (mu + k*sigma)
    # =========================================================================

    # 31. Dynamic k=1.0 (very tight)
    exp31 = deepcopy(base)
    exp31['name'] = '31_dynamic_k_1.0'
    exp31['dynamic_margin_k'] = 1.0
    experiments.append(exp31)

    # 32. Dynamic k=1.5
    exp32 = deepcopy(base)
    exp32['name'] = '32_dynamic_k_1.5'
    exp32['dynamic_margin_k'] = 1.5
    experiments.append(exp32)

    # 33. Dynamic k=2.0
    exp33 = deepcopy(base)
    exp33['name'] = '33_dynamic_k_2.0'
    exp33['dynamic_margin_k'] = 2.0
    experiments.append(exp33)

    # 34. Dynamic k=4.0
    exp34 = deepcopy(base)
    exp34['name'] = '34_dynamic_k_4.0'
    exp34['dynamic_margin_k'] = 4.0
    experiments.append(exp34)

    # 35. Dynamic k=5.0
    exp35 = deepcopy(base)
    exp35['name'] = '35_dynamic_k_5.0'
    exp35['dynamic_margin_k'] = 5.0
    experiments.append(exp35)

    # =========================================================================
    # Experiment 36: Feature-wise Masking (single)
    # =========================================================================

    # 36. Feature-wise masking
    exp36 = deepcopy(base)
    exp36['name'] = '36_feature_wise_mask'
    exp36['masking_strategy'] = 'feature_wise'
    experiments.append(exp36)

    # =========================================================================
    # Experiments 37-40: Attention Head Variations
    # =========================================================================

    # 37. nhead=1
    exp37 = deepcopy(base)
    exp37['name'] = '37_nhead_1'
    exp37['nhead'] = 1
    experiments.append(exp37)

    # 38. nhead=2
    exp38 = deepcopy(base)
    exp38['name'] = '38_nhead_2'
    exp38['nhead'] = 2
    experiments.append(exp38)

    # 39. nhead=8
    exp39 = deepcopy(base)
    exp39['name'] = '39_nhead_8'
    exp39['nhead'] = 8
    experiments.append(exp39)

    # 40. d_model=64 + nhead=8
    exp40 = deepcopy(base)
    exp40['name'] = '40_d64_nhead8'
    exp40['d_model'] = 64
    exp40['nhead'] = 8
    exp40['dim_feedforward'] = 256
    exp40['cnn_channels'] = (32, 64)
    experiments.append(exp40)

    # =========================================================================
    # Experiments 41-44: FFN Variations
    # =========================================================================

    # 41. FFN 64 (smaller)
    exp41 = deepcopy(base)
    exp41['name'] = '41_ffn_64'
    exp41['dim_feedforward'] = 64
    experiments.append(exp41)

    # 42. FFN 256 (larger)
    exp42 = deepcopy(base)
    exp42['name'] = '42_ffn_256'
    exp42['dim_feedforward'] = 256
    experiments.append(exp42)

    # 43. FFN 512
    exp43 = deepcopy(base)
    exp43['name'] = '43_ffn_512'
    exp43['dim_feedforward'] = 512
    experiments.append(exp43)

    # 44. FFN 1024
    exp44 = deepcopy(base)
    exp44['name'] = '44_ffn_1024'
    exp44['dim_feedforward'] = 1024
    experiments.append(exp44)

    # =========================================================================
    # Experiments 45-48: Learning Parameters
    # =========================================================================

    # 45. Learning rate 5e-4 (lower)
    exp45 = deepcopy(base)
    exp45['name'] = '45_lr_5e-4'
    exp45['learning_rate'] = 5e-4
    experiments.append(exp45)

    # 46. Learning rate 2e-3 (higher)
    exp46 = deepcopy(base)
    exp46['name'] = '46_lr_2e-3'
    exp46['learning_rate'] = 2e-3
    experiments.append(exp46)

    # 47. Dropout 0.0 (no dropout)
    exp47 = deepcopy(base)
    exp47['name'] = '47_dropout_0.0'
    exp47['dropout'] = 0.0
    experiments.append(exp47)

    # 48. Dropout 0.2 (stronger)
    exp48 = deepcopy(base)
    exp48['name'] = '48_dropout_0.2'
    exp48['dropout'] = 0.2
    experiments.append(exp48)

    # =========================================================================
    # Experiments 49-51: Margin Type Variations
    # =========================================================================

    # 49. Margin type: hinge
    exp49 = deepcopy(base)
    exp49['name'] = '49_margin_type_hinge'
    exp49['margin_type'] = 'hinge'
    experiments.append(exp49)

    # 50. Margin type: softplus
    exp50 = deepcopy(base)
    exp50['name'] = '50_margin_type_softplus'
    exp50['margin_type'] = 'softplus'
    experiments.append(exp50)

    # 51. Dynamic margin with k=4.0
    exp51 = deepcopy(base)
    exp51['name'] = '51_dynamic_margin_k4'
    exp51['margin_type'] = 'dynamic'
    exp51['dynamic_margin_k'] = 4.0
    experiments.append(exp51)

    # =========================================================================
    # Experiments 52-53: Larger d_model (USER REQUESTED)
    # Goal: More model capacity for better normal pattern learning
    # =========================================================================

    # 52. d_model=128
    exp52 = deepcopy(base)
    exp52['name'] = '52_d_model_128'
    exp52['d_model'] = 128
    exp52['nhead'] = 8
    exp52['dim_feedforward'] = 512
    exp52['cnn_channels'] = (64, 128)
    experiments.append(exp52)

    # 53. d_model=256
    exp53 = deepcopy(base)
    exp53['name'] = '53_d_model_256'
    exp53['d_model'] = 256
    exp53['nhead'] = 8
    exp53['dim_feedforward'] = 1024
    exp53['cnn_channels'] = (128, 256)
    experiments.append(exp53)

    # =========================================================================
    # Experiments 54-55: Patch Size Variations (USER REQUESTED)
    # Goal: Test granularity impact on anomaly detection
    # =========================================================================

    # 54. patch_size=5, num_patches=20 (finer granularity)
    exp54 = deepcopy(base)
    exp54['name'] = '54_patch_size_5'
    exp54['patch_size'] = 5
    exp54['num_patches'] = 20
    exp54['mask_last_n'] = 5
    experiments.append(exp54)

    # 55. patch_size=20, num_patches=5 (coarser granularity)
    exp55 = deepcopy(base)
    exp55['name'] = '55_patch_size_20'
    exp55['patch_size'] = 20
    exp55['num_patches'] = 5
    exp55['mask_last_n'] = 20
    experiments.append(exp55)

    # =========================================================================
    # Experiments 56-57: Large Window Sizes (USER REQUESTED)
    # Goal: Longer temporal context for pattern learning
    # =========================================================================

    # 56. window=1000, num_patches=100
    exp56 = deepcopy(base)
    exp56['name'] = '56_window_1000'
    exp56['seq_length'] = 1000
    exp56['num_patches'] = 100
    exp56['patch_size'] = 10
    exp56['mask_last_n'] = 10
    experiments.append(exp56)

    # 57. window=2000, num_patches=200
    exp57 = deepcopy(base)
    exp57['name'] = '57_window_2000'
    exp57['seq_length'] = 2000
    exp57['num_patches'] = 200
    exp57['patch_size'] = 10
    exp57['mask_last_n'] = 10
    experiments.append(exp57)

    # =========================================================================
    # Experiments 58-63: Window + Patch Combinations (USER REQUESTED)
    # Goal: Find optimal window/patch combinations for discrepancy
    # =========================================================================

    # 58. window=1000, patch_size=5, num_patches=200
    exp58 = deepcopy(base)
    exp58['name'] = '58_w1000_p5'
    exp58['seq_length'] = 1000
    exp58['patch_size'] = 5
    exp58['num_patches'] = 200
    exp58['mask_last_n'] = 5
    experiments.append(exp58)

    # 59. window=1000, patch_size=20, num_patches=50
    exp59 = deepcopy(base)
    exp59['name'] = '59_w1000_p20'
    exp59['seq_length'] = 1000
    exp59['patch_size'] = 20
    exp59['num_patches'] = 50
    exp59['mask_last_n'] = 20
    experiments.append(exp59)

    # 60. window=2000, patch_size=20, num_patches=100
    exp60 = deepcopy(base)
    exp60['name'] = '60_w2000_p20'
    exp60['seq_length'] = 2000
    exp60['patch_size'] = 20
    exp60['num_patches'] = 100
    exp60['mask_last_n'] = 20
    experiments.append(exp60)

    # 61. window=500, patch_size=5, num_patches=100
    exp61 = deepcopy(base)
    exp61['name'] = '61_w500_p5'
    exp61['seq_length'] = 500
    exp61['patch_size'] = 5
    exp61['num_patches'] = 100
    exp61['mask_last_n'] = 5
    experiments.append(exp61)

    # 62. window=200, patch_size=5, num_patches=40
    exp62 = deepcopy(base)
    exp62['name'] = '62_w200_p5'
    exp62['seq_length'] = 200
    exp62['patch_size'] = 5
    exp62['num_patches'] = 40
    exp62['mask_last_n'] = 5
    experiments.append(exp62)

    # 63. window=100, patch_size=20, num_patches=5 (very coarse)
    exp63 = deepcopy(base)
    exp63['name'] = '63_w100_p20'
    exp63['seq_length'] = 100
    exp63['patch_size'] = 20
    exp63['num_patches'] = 5
    exp63['mask_last_n'] = 20
    experiments.append(exp63)

    # =========================================================================
    # Experiments 64-65: Capacity + Context Combinations
    # Goal: Test synergy between model capacity and temporal context
    # =========================================================================

    # 64. d_model=128 + window=500 (large model + long context)
    exp64 = deepcopy(base)
    exp64['name'] = '64_d128_w500'
    exp64['d_model'] = 128
    exp64['nhead'] = 8
    exp64['dim_feedforward'] = 512
    exp64['cnn_channels'] = (64, 128)
    exp64['seq_length'] = 500
    exp64['num_patches'] = 50
    exp64['patch_size'] = 10
    exp64['mask_last_n'] = 10
    experiments.append(exp64)

    # 65. patch_size=5 + window=500 (fine-grained + long context)
    exp65 = deepcopy(base)
    exp65['name'] = '65_p5_w500'
    exp65['seq_length'] = 500
    exp65['patch_size'] = 5
    exp65['num_patches'] = 100
    exp65['mask_last_n'] = 5
    experiments.append(exp65)

    # =========================================================================
    # Experiments 66-68: Model Depth Variations
    # Goal: Test encoder/decoder depth impact on discrepancy
    # =========================================================================

    # 66. Encoder 4 layers (deeper encoder)
    exp66 = deepcopy(base)
    exp66['name'] = '66_encoder_4'
    exp66['num_encoder_layers'] = 4
    experiments.append(exp66)

    # 67. d_model=128 + nhead=8 (large model with many heads)
    exp67 = deepcopy(base)
    exp67['name'] = '67_d128_nhead8'
    exp67['d_model'] = 128
    exp67['nhead'] = 8
    exp67['dim_feedforward'] = 512
    exp67['cnn_channels'] = (64, 128)
    experiments.append(exp67)

    # 68. Shallow all (encoder=1, teacher=2, student=1)
    exp68 = deepcopy(base)
    exp68['name'] = '68_shallow_all'
    exp68['num_encoder_layers'] = 1
    exp68['num_teacher_decoder_layers'] = 2
    exp68['num_student_decoder_layers'] = 1
    experiments.append(exp68)

    # =========================================================================
    # Experiments 69-70: Training Strategy Variations
    # Goal: Test training hyperparameters impact
    # =========================================================================

    # 69. Longer warmup (teacher_only_warmup_epochs=5)
    exp69 = deepcopy(base)
    exp69['name'] = '69_warmup_5'
    exp69['teacher_only_warmup_epochs'] = 5
    experiments.append(exp69)

    # 70. Slower learning + stronger regularization
    exp70 = deepcopy(base)
    exp70['name'] = '70_slow_regularized'
    exp70['learning_rate'] = 1e-4
    exp70['weight_decay'] = 1e-4
    experiments.append(exp70)

    # =========================================================================
    # Experiments 71-110: PHASE 2 - NEW COMPREHENSIVE OPTIMIZATION
    # Completely redesigned based on deep Phase 1 analysis (12-step sequential thinking)
    #
    # Key findings from Phase 1:
    # - masking_ratio=0.2 is optimal (ROC-AUC=0.978, disc_ratio=2.11, t_ratio=8.50)
    # - d_model=64 is sweet spot (ROC-AUC=0.970)
    # - Shallow decoder (t2s1) outperforms deep decoder (ROC=0.962, t_ratio=7.16)
    # - t_ratio has 0.773 correlation with ROC-AUC (strongest predictor)
    # - num_patches ≤ 50 rule is critical (>50 causes performance collapse)
    # - FFN=256 (4x d_model) is optimal
    # - nhead=1 showed highest t_ratio (7.616)
    # - dynamic_margin_k=1.5 is optimal (disc_ratio=1.372)
    #
    # Groups:
    # A (71-78): Ultra-Low Masking Exploration - 8 experiments
    # B (79-86): Triple Optimal Combinations - 8 experiments
    # C (87-96): Large Window + Low Masking [USER PRIORITY] - 10 experiments
    # D (97-102): Scoring Modes - 6 experiments
    # E (103-108): Full Optimal Combinations - 6 experiments
    # F (109-110): Edge Cases & Validation - 2 experiments
    # =========================================================================

    # =========================================================================
    # GROUP A: ULTRA-LOW MASKING EXPLORATION (71-78)
    # Goal: Test if masking < 0.2 improves performance further
    # =========================================================================

    # 71. masking=0.10 (ultra-low baseline)
    exp71 = deepcopy(base)
    exp71['name'] = '71_mask0.10'
    exp71['masking_ratio'] = 0.1
    experiments.append(exp71)

    # 72. masking=0.15 (between ultra and best)
    exp72 = deepcopy(base)
    exp72['name'] = '72_mask0.15'
    exp72['masking_ratio'] = 0.15
    experiments.append(exp72)

    # 73. masking=0.25 (precision between 0.2 and 0.3)
    exp73 = deepcopy(base)
    exp73['name'] = '73_mask0.25'
    exp73['masking_ratio'] = 0.25
    experiments.append(exp73)

    # 74. masking=0.10 + d_model=64
    exp74 = deepcopy(base)
    exp74['name'] = '74_mask0.10_d64'
    exp74['masking_ratio'] = 0.1
    exp74['d_model'] = 64
    exp74['nhead'] = 4
    exp74['dim_feedforward'] = 256
    exp74['cnn_channels'] = (32, 64)
    experiments.append(exp74)

    # 75. masking=0.15 + d_model=64
    exp75 = deepcopy(base)
    exp75['name'] = '75_mask0.15_d64'
    exp75['masking_ratio'] = 0.15
    exp75['d_model'] = 64
    exp75['nhead'] = 4
    exp75['dim_feedforward'] = 256
    exp75['cnn_channels'] = (32, 64)
    experiments.append(exp75)

    # 76. masking=0.20 + d_model=64 (EXPECTED TOP - combines two best factors)
    exp76 = deepcopy(base)
    exp76['name'] = '76_mask0.20_d64'
    exp76['masking_ratio'] = 0.2
    exp76['d_model'] = 64
    exp76['nhead'] = 4
    exp76['dim_feedforward'] = 256
    exp76['cnn_channels'] = (32, 64)
    experiments.append(exp76)

    # 77. masking=0.10 + t2s1 (ultra-low + shallow decoder)
    exp77 = deepcopy(base)
    exp77['name'] = '77_mask0.10_t2s1'
    exp77['masking_ratio'] = 0.1
    exp77['num_teacher_decoder_layers'] = 2
    exp77['num_student_decoder_layers'] = 1
    experiments.append(exp77)

    # 78. masking=0.20 + t2s1 (best mask + shallow decoder)
    exp78 = deepcopy(base)
    exp78['name'] = '78_mask0.20_t2s1'
    exp78['masking_ratio'] = 0.2
    exp78['num_teacher_decoder_layers'] = 2
    exp78['num_student_decoder_layers'] = 1
    experiments.append(exp78)

    # =========================================================================
    # GROUP B: TRIPLE OPTIMAL COMBINATIONS (79-86)
    # Goal: Combine masking + d_model + decoder depth
    # =========================================================================

    # 79. TRIPLE OPTIMAL: masking=0.2 + d64 + t2s1 (EXPECTED TOP)
    exp79 = deepcopy(base)
    exp79['name'] = '79_triple_opt'
    exp79['masking_ratio'] = 0.2
    exp79['d_model'] = 64
    exp79['nhead'] = 4
    exp79['dim_feedforward'] = 256
    exp79['cnn_channels'] = (32, 64)
    exp79['num_teacher_decoder_layers'] = 2
    exp79['num_student_decoder_layers'] = 1
    experiments.append(exp79)

    # 80. Triple with masking=0.15
    exp80 = deepcopy(base)
    exp80['name'] = '80_triple_mask0.15'
    exp80['masking_ratio'] = 0.15
    exp80['d_model'] = 64
    exp80['nhead'] = 4
    exp80['dim_feedforward'] = 256
    exp80['cnn_channels'] = (32, 64)
    exp80['num_teacher_decoder_layers'] = 2
    exp80['num_student_decoder_layers'] = 1
    experiments.append(exp80)

    # 81. Triple with masking=0.10
    exp81 = deepcopy(base)
    exp81['name'] = '81_triple_mask0.10'
    exp81['masking_ratio'] = 0.1
    exp81['d_model'] = 64
    exp81['nhead'] = 4
    exp81['dim_feedforward'] = 256
    exp81['cnn_channels'] = (32, 64)
    exp81['num_teacher_decoder_layers'] = 2
    exp81['num_student_decoder_layers'] = 1
    experiments.append(exp81)

    # 82. Shallow all: masking=0.2 + d64 + enc=1 + t2s1
    exp82 = deepcopy(base)
    exp82['name'] = '82_shallow_mask0.2'
    exp82['masking_ratio'] = 0.2
    exp82['d_model'] = 64
    exp82['nhead'] = 4
    exp82['dim_feedforward'] = 256
    exp82['cnn_channels'] = (32, 64)
    exp82['num_encoder_layers'] = 1
    exp82['num_teacher_decoder_layers'] = 2
    exp82['num_student_decoder_layers'] = 1
    experiments.append(exp82)

    # 83. Shallow all with masking=0.15
    exp83 = deepcopy(base)
    exp83['name'] = '83_shallow_mask0.15'
    exp83['masking_ratio'] = 0.15
    exp83['d_model'] = 64
    exp83['nhead'] = 4
    exp83['dim_feedforward'] = 256
    exp83['cnn_channels'] = (32, 64)
    exp83['num_encoder_layers'] = 1
    exp83['num_teacher_decoder_layers'] = 2
    exp83['num_student_decoder_layers'] = 1
    experiments.append(exp83)

    # 84. d64 + FFN=512 (expanded FFN)
    exp84 = deepcopy(base)
    exp84['name'] = '84_d64_ffn512'
    exp84['masking_ratio'] = 0.2
    exp84['d_model'] = 64
    exp84['nhead'] = 4
    exp84['dim_feedforward'] = 512
    exp84['cnn_channels'] = (32, 64)
    experiments.append(exp84)

    # 85. d64 + nhead=1 (single attention head - showed high t_ratio)
    exp85 = deepcopy(base)
    exp85['name'] = '85_d64_nhead1'
    exp85['masking_ratio'] = 0.2
    exp85['d_model'] = 64
    exp85['nhead'] = 1
    exp85['dim_feedforward'] = 256
    exp85['cnn_channels'] = (32, 64)
    experiments.append(exp85)

    # 86. d64 + dynamic_margin_k=1.5 (optimal margin)
    exp86 = deepcopy(base)
    exp86['name'] = '86_d64_k1.5'
    exp86['masking_ratio'] = 0.2
    exp86['d_model'] = 64
    exp86['nhead'] = 4
    exp86['dim_feedforward'] = 256
    exp86['cnn_channels'] = (32, 64)
    exp86['dynamic_margin_k'] = 1.5
    experiments.append(exp86)

    # =========================================================================
    # GROUP C: LARGE WINDOW + LOW MASKING (87-96) [USER PRIORITY]
    # Goal: Test window 500/1000 with patch_size=10 (as requested)
    # Note: num_patches > 50 is risky but included per user request
    # =========================================================================

    # 87. window=500 + masking=0.2 (50 patches - safe)
    exp87 = deepcopy(base)
    exp87['name'] = '87_w500_mask0.2'
    exp87['masking_ratio'] = 0.2
    exp87['seq_length'] = 500
    exp87['num_patches'] = 50
    exp87['patch_size'] = 10
    exp87['mask_last_n'] = 10
    experiments.append(exp87)

    # 88. window=500 + masking=0.15
    exp88 = deepcopy(base)
    exp88['name'] = '88_w500_mask0.15'
    exp88['masking_ratio'] = 0.15
    exp88['seq_length'] = 500
    exp88['num_patches'] = 50
    exp88['patch_size'] = 10
    exp88['mask_last_n'] = 10
    experiments.append(exp88)

    # 89. window=500 + masking=0.2 + d_model=64
    exp89 = deepcopy(base)
    exp89['name'] = '89_w500_mask0.2_d64'
    exp89['masking_ratio'] = 0.2
    exp89['seq_length'] = 500
    exp89['num_patches'] = 50
    exp89['patch_size'] = 10
    exp89['mask_last_n'] = 10
    exp89['d_model'] = 64
    exp89['nhead'] = 4
    exp89['dim_feedforward'] = 256
    exp89['cnn_channels'] = (32, 64)
    experiments.append(exp89)

    # 90. window=1000 + masking=0.2 (100 patches - RISKY but user requested)
    exp90 = deepcopy(base)
    exp90['name'] = '90_w1000_mask0.2'
    exp90['masking_ratio'] = 0.2
    exp90['seq_length'] = 1000
    exp90['num_patches'] = 100
    exp90['patch_size'] = 10
    exp90['mask_last_n'] = 10
    experiments.append(exp90)

    # 91. window=1000 + masking=0.15 (100 patches - RISKY)
    exp91 = deepcopy(base)
    exp91['name'] = '91_w1000_mask0.15'
    exp91['masking_ratio'] = 0.15
    exp91['seq_length'] = 1000
    exp91['num_patches'] = 100
    exp91['patch_size'] = 10
    exp91['mask_last_n'] = 10
    experiments.append(exp91)

    # 92. window=1000 + masking=0.2 + d_model=64 (RISKY)
    exp92 = deepcopy(base)
    exp92['name'] = '92_w1000_mask0.2_d64'
    exp92['masking_ratio'] = 0.2
    exp92['seq_length'] = 1000
    exp92['num_patches'] = 100
    exp92['patch_size'] = 10
    exp92['mask_last_n'] = 10
    exp92['d_model'] = 64
    exp92['nhead'] = 4
    exp92['dim_feedforward'] = 256
    exp92['cnn_channels'] = (32, 64)
    experiments.append(exp92)

    # 93. window=1000 + patch=20 + masking=0.2 (50 patches - SAFE combo)
    exp93 = deepcopy(base)
    exp93['name'] = '93_w1000_p20_mask0.2'
    exp93['masking_ratio'] = 0.2
    exp93['seq_length'] = 1000
    exp93['num_patches'] = 50
    exp93['patch_size'] = 20
    exp93['mask_last_n'] = 20
    experiments.append(exp93)

    # 94. window=1000 + patch=20 + d_model=64 (safe + capacity)
    exp94 = deepcopy(base)
    exp94['name'] = '94_w1000_p20_d64'
    exp94['masking_ratio'] = 0.2
    exp94['seq_length'] = 1000
    exp94['num_patches'] = 50
    exp94['patch_size'] = 20
    exp94['mask_last_n'] = 20
    exp94['d_model'] = 64
    exp94['nhead'] = 4
    exp94['dim_feedforward'] = 256
    exp94['cnn_channels'] = (32, 64)
    experiments.append(exp94)

    # 95. window=500 + d64 + t2s1 (triple + window)
    exp95 = deepcopy(base)
    exp95['name'] = '95_w500_d64_t2s1'
    exp95['masking_ratio'] = 0.2
    exp95['seq_length'] = 500
    exp95['num_patches'] = 50
    exp95['patch_size'] = 10
    exp95['mask_last_n'] = 10
    exp95['d_model'] = 64
    exp95['nhead'] = 4
    exp95['dim_feedforward'] = 256
    exp95['cnn_channels'] = (32, 64)
    exp95['num_teacher_decoder_layers'] = 2
    exp95['num_student_decoder_layers'] = 1
    experiments.append(exp95)

    # 96. window=1000 + patch=20 + t2s1 (safe + shallow)
    exp96 = deepcopy(base)
    exp96['name'] = '96_w1000_p20_t2s1'
    exp96['masking_ratio'] = 0.2
    exp96['seq_length'] = 1000
    exp96['num_patches'] = 50
    exp96['patch_size'] = 20
    exp96['mask_last_n'] = 20
    exp96['num_teacher_decoder_layers'] = 2
    exp96['num_student_decoder_layers'] = 1
    experiments.append(exp96)

    # =========================================================================
    # GROUP D: SCORING MODES (97-102)
    # Goal: Test scoring modes to better utilize discrepancy for detection
    # =========================================================================

    # 97. normalized scoring (baseline)
    exp97 = deepcopy(base)
    exp97['name'] = '97_score_normalized'
    exp97['anomaly_score_mode'] = 'normalized'
    experiments.append(exp97)

    # 98. adaptive scoring (baseline)
    exp98 = deepcopy(base)
    exp98['name'] = '98_score_adaptive'
    exp98['anomaly_score_mode'] = 'adaptive'
    experiments.append(exp98)

    # 99. normalized + masking=0.2
    exp99 = deepcopy(base)
    exp99['name'] = '99_norm_mask0.2'
    exp99['anomaly_score_mode'] = 'normalized'
    exp99['masking_ratio'] = 0.2
    experiments.append(exp99)

    # 100. adaptive + masking=0.2
    exp100 = deepcopy(base)
    exp100['name'] = '100_adap_mask0.2'
    exp100['anomaly_score_mode'] = 'adaptive'
    exp100['masking_ratio'] = 0.2
    experiments.append(exp100)

    # 101. normalized + d_model=64 + masking=0.2
    exp101 = deepcopy(base)
    exp101['name'] = '101_norm_d64_mask0.2'
    exp101['anomaly_score_mode'] = 'normalized'
    exp101['masking_ratio'] = 0.2
    exp101['d_model'] = 64
    exp101['nhead'] = 4
    exp101['dim_feedforward'] = 256
    exp101['cnn_channels'] = (32, 64)
    experiments.append(exp101)

    # 102. adaptive + d_model=64 + masking=0.2
    exp102 = deepcopy(base)
    exp102['name'] = '102_adap_d64_mask0.2'
    exp102['anomaly_score_mode'] = 'adaptive'
    exp102['masking_ratio'] = 0.2
    exp102['d_model'] = 64
    exp102['nhead'] = 4
    exp102['dim_feedforward'] = 256
    exp102['cnn_channels'] = (32, 64)
    experiments.append(exp102)

    # =========================================================================
    # GROUP E: FULL OPTIMAL COMBINATIONS (103-108)
    # Goal: Combine ALL optimal factors discovered
    # =========================================================================

    # 103. FULL OPTIMAL: mask=0.2 + d64 + t2s1 + k=1.5 + normalized (EXPECTED TOP)
    exp103 = deepcopy(base)
    exp103['name'] = '103_full_optimal'
    exp103['anomaly_score_mode'] = 'normalized'
    exp103['masking_ratio'] = 0.2
    exp103['d_model'] = 64
    exp103['nhead'] = 4
    exp103['dim_feedforward'] = 256
    exp103['cnn_channels'] = (32, 64)
    exp103['num_teacher_decoder_layers'] = 2
    exp103['num_student_decoder_layers'] = 1
    exp103['dynamic_margin_k'] = 1.5
    experiments.append(exp103)

    # 104. Full optimal with adaptive scoring
    exp104 = deepcopy(base)
    exp104['name'] = '104_full_opt_adap'
    exp104['anomaly_score_mode'] = 'adaptive'
    exp104['masking_ratio'] = 0.2
    exp104['d_model'] = 64
    exp104['nhead'] = 4
    exp104['dim_feedforward'] = 256
    exp104['cnn_channels'] = (32, 64)
    exp104['num_teacher_decoder_layers'] = 2
    exp104['num_student_decoder_layers'] = 1
    exp104['dynamic_margin_k'] = 1.5
    experiments.append(exp104)

    # 105. Full optimal + window=500
    exp105 = deepcopy(base)
    exp105['name'] = '105_full_w500'
    exp105['anomaly_score_mode'] = 'normalized'
    exp105['masking_ratio'] = 0.2
    exp105['seq_length'] = 500
    exp105['num_patches'] = 50
    exp105['patch_size'] = 10
    exp105['mask_last_n'] = 10
    exp105['d_model'] = 64
    exp105['nhead'] = 4
    exp105['dim_feedforward'] = 256
    exp105['cnn_channels'] = (32, 64)
    exp105['num_teacher_decoder_layers'] = 2
    exp105['num_student_decoder_layers'] = 1
    experiments.append(exp105)

    # 106. Full optimal + window=1000 + patch=20 (safe long window)
    exp106 = deepcopy(base)
    exp106['name'] = '106_full_w1000_p20'
    exp106['anomaly_score_mode'] = 'normalized'
    exp106['masking_ratio'] = 0.2
    exp106['seq_length'] = 1000
    exp106['num_patches'] = 50
    exp106['patch_size'] = 20
    exp106['mask_last_n'] = 20
    exp106['d_model'] = 64
    exp106['nhead'] = 4
    exp106['dim_feedforward'] = 256
    exp106['cnn_channels'] = (32, 64)
    exp106['num_teacher_decoder_layers'] = 2
    exp106['num_student_decoder_layers'] = 1
    experiments.append(exp106)

    # 107. Full optimal with masking=0.15
    exp107 = deepcopy(base)
    exp107['name'] = '107_full_mask0.15'
    exp107['anomaly_score_mode'] = 'normalized'
    exp107['masking_ratio'] = 0.15
    exp107['d_model'] = 64
    exp107['nhead'] = 4
    exp107['dim_feedforward'] = 256
    exp107['cnn_channels'] = (32, 64)
    exp107['num_teacher_decoder_layers'] = 2
    exp107['num_student_decoder_layers'] = 1
    exp107['dynamic_margin_k'] = 1.5
    experiments.append(exp107)

    # 108. Full optimal + shallow encoder (enc=1)
    exp108 = deepcopy(base)
    exp108['name'] = '108_full_shallow'
    exp108['anomaly_score_mode'] = 'normalized'
    exp108['masking_ratio'] = 0.2
    exp108['d_model'] = 64
    exp108['nhead'] = 4
    exp108['dim_feedforward'] = 256
    exp108['cnn_channels'] = (32, 64)
    exp108['num_encoder_layers'] = 1
    exp108['num_teacher_decoder_layers'] = 2
    exp108['num_student_decoder_layers'] = 1
    exp108['dynamic_margin_k'] = 1.5
    experiments.append(exp108)

    # =========================================================================
    # GROUP F: EDGE CASES & VALIDATION (109-110)
    # Goal: Test edge configurations
    # =========================================================================

    # 109. d_model=48 (between 32 and 64)
    exp109 = deepcopy(base)
    exp109['name'] = '109_d48_mask0.2'
    exp109['masking_ratio'] = 0.2
    exp109['d_model'] = 48
    exp109['nhead'] = 4
    exp109['dim_feedforward'] = 192
    exp109['cnn_channels'] = (24, 48)
    experiments.append(exp109)

    # 110. Teacher=1, Student=1 (minimal decoder)
    exp110 = deepcopy(base)
    exp110['name'] = '110_t1s1_mask0.2'
    exp110['masking_ratio'] = 0.2
    exp110['num_teacher_decoder_layers'] = 1
    exp110['num_student_decoder_layers'] = 1
    experiments.append(exp110)

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

        # Evaluate with test_dataset for point-level PA%K
        evaluator = Evaluator(model, config, test_loader, test_dataset=test_dataset)
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

        # Compute detailed metrics on test set (by normal/anomaly)
        teacher_recon_losses = []
        student_recon_losses = []
        discrepancy_losses = []
        all_labels_test = []
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
                # Discrepancy loss
                disc = ((teacher_output - student_output) ** 2).mean(dim=2)
                d_loss = (disc * masked_pos).sum(dim=1) / (masked_pos.sum(dim=1) + 1e-8)
                discrepancy_losses.append(d_loss.cpu().numpy())
                # Labels
                all_labels_test.append(last_patch_labels.numpy())

        teacher_recon_all = np.concatenate(teacher_recon_losses)
        student_recon_all = np.concatenate(student_recon_losses)
        disc_all = np.concatenate(discrepancy_losses)
        labels_all = np.concatenate(all_labels_test)

        # Compute by normal/anomaly
        normal_mask = (labels_all == 0)
        anomaly_mask = (labels_all == 1)

        metrics['teacher_recon_loss'] = float(teacher_recon_all.mean())
        metrics['student_recon_loss'] = float(student_recon_all.mean())
        metrics['discrepancy_loss'] = float(disc_all.mean())

        # By normal/anomaly
        metrics['teacher_recon_normal'] = float(teacher_recon_all[normal_mask].mean()) if normal_mask.sum() > 0 else 0.0
        metrics['teacher_recon_anomaly'] = float(teacher_recon_all[anomaly_mask].mean()) if anomaly_mask.sum() > 0 else 0.0
        metrics['student_recon_normal'] = float(student_recon_all[normal_mask].mean()) if normal_mask.sum() > 0 else 0.0
        metrics['student_recon_anomaly'] = float(student_recon_all[anomaly_mask].mean()) if anomaly_mask.sum() > 0 else 0.0
        metrics['discrepancy_normal'] = float(disc_all[normal_mask].mean()) if normal_mask.sum() > 0 else 0.0
        metrics['discrepancy_anomaly'] = float(disc_all[anomaly_mask].mean()) if anomaly_mask.sum() > 0 else 0.0

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

def run_all_experiments(
    output_base_dir: str = 'results/experiments/temp',
    start_exp: int = 1,
    end_exp: int = None,
    only_mask_before: bool = False,
    skip_existing: bool = True
):
    """Run experiments with optional filtering

    Args:
        output_base_dir: Output directory for results
        start_exp: Start from this experiment number (1-indexed)
        end_exp: End at this experiment number (inclusive, 1-indexed)
        only_mask_before: Only run mask_before experiments (skip mask_after)
        skip_existing: Skip experiments that already have results
    """
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

    # Apply start/end filtering
    if end_exp is None:
        end_exp = len(experiments)
    filtered_experiments = [(i+1, exp) for i, exp in enumerate(experiments) if start_exp <= i+1 <= end_exp]

    # Determine mask_after_encoder settings to run
    mask_settings = [False] if only_mask_before else [False, True]
    total_experiments = len(filtered_experiments) * len(mask_settings)

    print(f"\nRunning experiments {start_exp}-{end_exp} ({len(filtered_experiments)} configs)", flush=True)
    print(f"Mask settings: {'mask_before only' if only_mask_before else 'both mask_before and mask_after'}", flush=True)
    print(f"Total experiments: {total_experiments}", flush=True)
    print(f"Skip existing: {skip_existing}", flush=True)

    # Results storage - load existing results if available
    summary_path = os.path.join(output_base_dir, 'summary_results.csv')
    if skip_existing and os.path.exists(summary_path):
        existing_df = pd.read_csv(summary_path)
        all_results = existing_df.to_dict('records')
        print(f"Loaded {len(all_results)} existing results from {summary_path}", flush=True)
    else:
        all_results = []

    experiment_count = 0
    skipped_count = 0

    # Run experiments for selected mask_after_encoder settings
    for mask_after_encoder in mask_settings:
        suffix = 'mask_before' if not mask_after_encoder else 'mask_after'
        print(f"\n{'='*80}", flush=True)
        print(f" ROUND: mask_after_encoder = {mask_after_encoder} ({suffix})", flush=True)
        print(f"{'='*80}", flush=True)

        for exp_num, exp_config in filtered_experiments:
            exp_config = deepcopy(exp_config)
            exp_config['mask_after_encoder'] = mask_after_encoder
            experiment_count += 1

            exp_name = f"{exp_config['name']}_{suffix}"
            exp_dir = os.path.join(output_base_dir, exp_name)

            # Check if experiment already exists
            if skip_existing and os.path.exists(os.path.join(exp_dir, 'best_model.pt')):
                print(f"\n[{experiment_count}/{total_experiments}] SKIPPING (exists): {exp_name}", flush=True)
                skipped_count += 1
                continue

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

            # Compute ratios
            disc_normal = metrics.get('discrepancy_normal', 0)
            disc_anomaly = metrics.get('discrepancy_anomaly', 0)
            disc_ratio = disc_anomaly / (disc_normal + 1e-8)

            t_recon_normal = metrics.get('teacher_recon_normal', 0)
            t_recon_anomaly = metrics.get('teacher_recon_anomaly', 0)
            t_recon_ratio = t_recon_anomaly / (t_recon_normal + 1e-8)

            s_recon_normal = metrics.get('student_recon_normal', 0)
            s_recon_anomaly = metrics.get('student_recon_anomaly', 0)
            s_recon_ratio = s_recon_anomaly / (s_recon_normal + 1e-8)

            # Print results table with all PA%K metrics
            pa_10_f1 = metrics.get('pa_10_f1', 0.0)
            pa_20_f1 = metrics.get('pa_20_f1', 0.0)
            pa_50_f1 = metrics.get('pa_50_f1', 0.0)
            pa_80_f1 = metrics.get('pa_80_f1', 0.0)
            pa_10_auc = metrics.get('pa_10_roc_auc', 0.0)
            pa_20_auc = metrics.get('pa_20_roc_auc', 0.0)
            pa_50_auc = metrics.get('pa_50_roc_auc', 0.0)
            pa_80_auc = metrics.get('pa_80_roc_auc', 0.0)

            print(f"\n  ┌────────────────────────────────────────────────────────────────────────────────────────┐", flush=True)
            print(f"  │ ROC-AUC: {metrics['roc_auc']:.4f}  │  F1: {metrics['f1_score']:.4f}  │  Time: {elapsed:.1f}s                         │", flush=True)
            print(f"  ├────────────────────────────────────────────────────────────────────────────────────────┤", flush=True)
            print(f"  │ PA%K         │  PA%10   │  PA%20   │  PA%50   │  PA%80   │", flush=True)
            print(f"  │ F1 Score     │  {pa_10_f1:.4f}  │  {pa_20_f1:.4f}  │  {pa_50_f1:.4f}  │  {pa_80_f1:.4f}  │", flush=True)
            print(f"  │ ROC-AUC      │  {pa_10_auc:.4f}  │  {pa_20_auc:.4f}  │  {pa_50_auc:.4f}  │  {pa_80_auc:.4f}  │", flush=True)
            print(f"  ├────────────────────────────────────────────────────────────────────────────────────────┤", flush=True)
            print(f"  │ Metric              │ Normal     │ Anomaly    │ Ratio (Anom/Norm)  │", flush=True)
            print(f"  ├─────────────────────────────────────────────────────────────────────┤", flush=True)
            print(f"  │ Discrepancy Loss    │ {disc_normal:10.6f} │ {disc_anomaly:10.6f} │ {disc_ratio:18.4f} │", flush=True)
            print(f"  │ Teacher Recon Loss  │ {t_recon_normal:10.6f} │ {t_recon_anomaly:10.6f} │ {t_recon_ratio:18.4f} │", flush=True)
            print(f"  │ Student Recon Loss  │ {s_recon_normal:10.6f} │ {s_recon_anomaly:10.6f} │ {s_recon_ratio:18.4f} │", flush=True)
            print(f"  └─────────────────────────────────────────────────────────────────────┘", flush=True)

            if 'disturbing_roc_auc' in metrics and metrics['disturbing_roc_auc'] is not None:
                print(f"  Disturbing Normal - ROC-AUC: {metrics['disturbing_roc_auc']:.4f}", flush=True)

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
                'time_elapsed': elapsed,
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

    # Add ratio columns
    summary_df['disc_ratio'] = summary_df['discrepancy_anomaly'] / (summary_df['discrepancy_normal'] + 1e-8)
    summary_df['teacher_recon_ratio'] = summary_df['teacher_recon_anomaly'] / (summary_df['teacher_recon_normal'] + 1e-8)
    summary_df['student_recon_ratio'] = summary_df['student_recon_anomaly'] / (summary_df['student_recon_normal'] + 1e-8)

    summary_cols = [
        'experiment', 'roc_auc', 'f1_score',
        'pa_10_f1', 'pa_20_f1', 'pa_50_f1', 'pa_80_f1',
        'pa_10_roc_auc', 'pa_20_roc_auc', 'pa_50_roc_auc', 'pa_80_roc_auc',
        'discrepancy_normal', 'discrepancy_anomaly', 'disc_ratio',
    ]

    print("\n--- mask_after_encoder = False ---", flush=True)
    df_false = summary_df[summary_df['mask_after_encoder'] == False].copy()
    # Format for readability
    for col in summary_cols:
        if col in df_false.columns and col != 'experiment':
            df_false[col] = df_false[col].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else x)
    print(df_false[summary_cols].to_string(index=False), flush=True)

    print("\n--- mask_after_encoder = True ---", flush=True)
    df_true = summary_df[summary_df['mask_after_encoder'] == True].copy()
    for col in summary_cols:
        if col in df_true.columns and col != 'experiment':
            df_true[col] = df_true[col].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else x)
    print(df_true[summary_cols].to_string(index=False), flush=True)

    # Print insights
    print(f"\n{'='*80}", flush=True)
    print(" " * 20 + "KEY INSIGHTS", flush=True)
    print(f"{'='*80}", flush=True)

    # Best models by ROC-AUC
    best_false = summary_df[summary_df['mask_after_encoder'] == False].nlargest(5, 'roc_auc')
    best_true = summary_df[summary_df['mask_after_encoder'] == True].nlargest(5, 'roc_auc')

    print("\n[Top 5 by ROC-AUC - mask_after_encoder=False]", flush=True)
    for _, row in best_false.iterrows():
        print(f"  {row['experiment']}: ROC-AUC={row['roc_auc']:.4f}, F1={row['f1_score']:.4f}, disc_ratio={row['disc_ratio']:.4f}", flush=True)

    print("\n[Top 5 by ROC-AUC - mask_after_encoder=True]", flush=True)
    for _, row in best_true.iterrows():
        print(f"  {row['experiment']}: ROC-AUC={row['roc_auc']:.4f}, F1={row['f1_score']:.4f}, disc_ratio={row['disc_ratio']:.4f}", flush=True)

    # Best models by discrepancy ratio
    best_disc_false = summary_df[summary_df['mask_after_encoder'] == False].nlargest(5, 'disc_ratio')
    best_disc_true = summary_df[summary_df['mask_after_encoder'] == True].nlargest(5, 'disc_ratio')

    print("\n[Top 5 by Discrepancy Ratio - mask_after_encoder=False]", flush=True)
    for _, row in best_disc_false.iterrows():
        print(f"  {row['experiment']}: disc_ratio={row['disc_ratio']:.4f}, ROC-AUC={row['roc_auc']:.4f}", flush=True)

    print("\n[Top 5 by Discrepancy Ratio - mask_after_encoder=True]", flush=True)
    for _, row in best_disc_true.iterrows():
        print(f"  {row['experiment']}: disc_ratio={row['disc_ratio']:.4f}, ROC-AUC={row['roc_auc']:.4f}", flush=True)

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
    parser.add_argument('--start-exp', type=int, default=1,
                       help='Start from this experiment number (1-indexed)')
    parser.add_argument('--end-exp', type=int, default=None,
                       help='End at this experiment number (inclusive, 1-indexed)')
    parser.add_argument('--only-mask-before', action='store_true',
                       help='Only run mask_before experiments (skip mask_after)')
    parser.add_argument('--no-skip', action='store_true',
                       help='Do not skip existing experiments (re-run all)')
    args = parser.parse_args()

    run_all_experiments(
        output_base_dir=args.output_dir,
        start_exp=args.start_exp,
        end_exp=args.end_exp,
        only_mask_before=args.only_mask_before,
        skip_existing=not args.no_skip
    )
