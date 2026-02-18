"""
Quick Smoke Test Config for TEP Dataset
=========================================

파이프라인 동작 확인용 최소 설정.
- 1 fault type, 1 epoch
- 소규모 실행으로 코드 검증

Usage:
    python scripts/ablation/run_ablation.py --config scripts/ablation/configs/tep_quick_test.py

Data scale:
    Train: 50 fault-free runs × 960 = 48,000 samples (all normal)
    Test:  fault1, 10 runs × 960 = 9,600 samples (fault onset at sample 160)
    Total: ~57,600 samples

참조: docs/TEP_EXPERIMENT_GUIDE.md
"""

# Dataset configuration
DATASET_TYPE = 'tep_fault1'  # fault type 1: A/C Feed Step

# Phase metadata
PHASE_NAME = "tep_quick_test"
PHASE_DESCRIPTION = "TEP fault1 quick smoke test (1 epoch)"

# Base configuration
BASE_CONFIG = {
    # Model architecture (Phase 2 best model 기반)
    'seq_length': 160,       # fault onset 기간과 대응 (samples 0-159)
    'patch_size': 8,
    'num_patches': 20,       # seq_length / patch_size = 160 / 8
    'd_model': 128,
    'nhead': 8,
    'num_encoder_layers': 2,
    'num_teacher_decoder_layers': 4,
    'num_student_decoder_layers': 1,
    'patchify_mode': 'patch_cnn',

    # Training (minimal for smoke test)
    'num_epochs': 1,
    'learning_rate': 2e-3,
    'batch_size': 256,
    'warmup_epochs': 0,
    'teacher_only_warmup_epochs': 0,
    'use_amp': True,

    # Sliding window
    'use_sliding_window_dataset': True,
    'sliding_window_stride': 5,       # Train: (960-160)/5+1 ≈ 161 windows/run
    'sliding_window_test_stride': 11, # Test: stride=11 for speed (use 1 for full PA%K)

    # num_features는 자동 설정됨 (loaders.py에서 상수 열 제거 후 결정)
    # train_ratio도 자동 설정됨 (data_info['train_ratio'] 사용)
}

# Single experiment
EXPERIMENTS = [
    {
        'name': 'tep_fault1_quick',
        'config': {},  # Use BASE_CONFIG
    },
]

# Scoring modes
SCORING_MODES = ['default']
