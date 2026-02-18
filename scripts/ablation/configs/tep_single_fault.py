"""
TEP Single Fault Full Study Config
=====================================

특정 1개 fault type에 대한 전체 탐지 성능 분석 (50 epochs).
DATASET_TYPE을 변경하여 원하는 fault type 선택.

Usage:
    python scripts/ablation/run_ablation.py --config scripts/ablation/configs/tep_single_fault.py

Data scale (n_train_runs=50, n_test_runs=50):
    Train: 50 fault-free runs × 960 = 48,000 samples (all normal)
    Test:  1 fault × 50 runs × 960 = 48,000 samples
    Total: ~96,000 samples
    train_ratio: ~0.50 (balanced)

Available DATASET_TYPE values:
    'tep_fault1'   - A/C Feed Step (stream 4)          : 탐지 쉬움
    'tep_fault2'   - B Composition Step (stream 4)      : 탐지 쉬움
    'tep_fault3'   - D Feed Temperature Step            : 탐지 어려움
    'tep_fault4'   - Reactor Cooling Water Input Step   : 탐지 쉬움
    'tep_fault5'   - Condenser Cooling Water Input Step : 탐지 쉬움
    'tep_fault6'   - A Feed Loss                        : 탐지 쉬움
    'tep_fault7'   - C Header Pressure Loss             : 탐지 쉬움
    'tep_fault8'   - A, B, C Feed Composition Random    : 보통
    'tep_fault9'   - D Feed Temperature Random Variation: 탐지 어려움
    'tep_fault10'  - C Feed Temperature Random          : 보통
    'tep_fault11'  - Reactor Cooling Water Input Random : 보통
    'tep_fault12'  - Condenser Cooling Water Input Rand : 보통
    'tep_fault13'  - Reaction Kinetics Slow Drift       : 보통
    'tep_fault14'  - Reactor Cooling Water Valve Sticking: 탐지 쉬움
    'tep_fault15'  - Condenser Cooling Water Valve Stck : 탐지 어려움
    'tep_fault16'  - Unknown                            : 보통
    'tep_fault17'  - Unknown                            : 보통
    'tep_fault18'  - Unknown                            : 보통
    'tep_fault19'  - Unknown                            : 보통
    'tep_fault20'  - Unknown                            : 탐지 불가 수준

참조: docs/TEP_EXPERIMENT_GUIDE.md
"""

# Dataset configuration - 변경하여 원하는 fault 선택
DATASET_TYPE = 'tep_fault1'

# Phase metadata
PHASE_NAME = "tep_fault1_full"
PHASE_DESCRIPTION = "TEP fault1 full study (50 epochs)"

# Base configuration
BASE_CONFIG = {
    # Model architecture (Phase 2 best model 기반)
    'seq_length': 160,       # fault onset 기간과 대응
    'patch_size': 8,
    'num_patches': 20,       # seq_length / patch_size
    'd_model': 128,
    'nhead': 8,
    'num_encoder_layers': 2,
    'num_teacher_decoder_layers': 4,
    'num_student_decoder_layers': 1,
    'num_shared_decoder_layers': 0,
    'dim_feedforward': 512,
    'dropout': 0.15,
    'masking_ratio': 0.15,
    'patchify_mode': 'patch_cnn',
    'mask_after_encoder': True,
    'shared_mask_token': False,

    # Loss
    'margin': 0.5,
    'lambda_disc': 2.0,
    'margin_type': 'dynamic',
    'dynamic_margin_k': 2.0,
    'patch_level_loss': True,
    'anomaly_loss_weight': 2.0,
    'anomaly_score_mode': 'adaptive',

    # Training
    'num_epochs': 50,
    'learning_rate': 2e-3,
    'batch_size': 256,
    'weight_decay': 1e-5,
    'warmup_epochs': 10,
    'teacher_only_warmup_epochs': 3,
    'use_amp': True,
    'eval_interval': 5,
    'point_aggregation_method': 'voting',

    # Ablation flags
    'use_discrepancy_loss': True,
    'use_teacher': True,
    'use_student': True,
    'use_masking': True,
    'force_mask_anomaly': True,

    # Sliding window
    'use_sliding_window_dataset': True,
    'sliding_window_stride': 5,       # Train stride
    'sliding_window_test_stride': 1,  # Test stride=1 for proper PA%K

    # num_features, train_ratio: 자동 설정됨
    'random_seed': 42,
}

# Single experiment
EXPERIMENTS = [
    {
        'name': 'tep_fault1_base',
        'config': {},  # Use BASE_CONFIG
    },
]

# Scoring modes
SCORING_MODES = ['default', 'adaptive']
