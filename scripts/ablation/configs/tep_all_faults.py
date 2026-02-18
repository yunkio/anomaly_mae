"""
TEP All 20 Faults Config
==========================

전체 20개 fault type 포함 TEP 실험 (50 epochs).
모든 fault에 대한 전반적인 탐지 성능 평가.

Usage:
    python scripts/ablation/run_ablation.py --config scripts/ablation/configs/tep_all_faults.py

Data scale (n_train_runs=50, n_test_runs=50):
    Train: 50 fault-free runs × 960 = 48,000 samples (all normal)
    Test:  20 faults × 50 runs × 960 = 960,000 samples
    Total: ~1,008,000 samples
    train_ratio: ~0.0476 (train이 매우 작음 - 정상)
    Memory: ~210 MB (float32)

주의:
- Test 데이터가 방대하므로 evaluation에 시간이 걸림 (test stride=1 시 800K+ windows)
- GPU 메모리 4-8 GB 필요 (batch_size=256 기준)
- save_dataset_info는 run_ablation.py의 버그 수정 후 정상 동작 (수정 완료)

참조: docs/TEP_EXPERIMENT_GUIDE.md
"""

# Dataset configuration
DATASET_TYPE = 'tep'  # 전체 20 fault types

# Phase metadata
PHASE_NAME = "tep_all_faults"
PHASE_DESCRIPTION = "TEP all 20 faults full study (50 epochs)"

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
        'name': 'tep_all_faults_base',
        'config': {},  # Use BASE_CONFIG
    },
]

# Scoring modes
SCORING_MODES = ['default', 'adaptive']
