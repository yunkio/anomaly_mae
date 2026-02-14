"""
Test config for SWaT A1+A2 dataset using unified run_ablation.py

Usage:
    python scripts/ablation/run_ablation.py --config scripts/ablation/configs/swat_A1A2_test.py
"""

# Dataset configuration
DATASET_TYPE = 'swat_A1A2'  # Use SWaT A1 (normal) + A2 (attack) combined

# Phase metadata
PHASE_NAME = "swat_A1A2_test"
PHASE_DESCRIPTION = "SWaT A1+A2 dataset test with minimal epochs"

# Base configuration
BASE_CONFIG = {
    # Model architecture
    'seq_length': 500,
    'patch_size': 5,
    'num_patches': 100,
    'd_model': 128,
    'nhead': 8,
    'num_encoder_layers': 2,
    'num_teacher_decoder_layers': 4,
    'num_student_decoder_layers': 1,

    # Training
    'num_epochs': 1,  # Minimal for testing
    'learning_rate': 2e-3,
    'batch_size': 256,
    'warmup_epochs': 0,  # No warmup for quick test

    # Dataset (for SlidingWindowDataset)
    'use_sliding_window_dataset': True,
    'sliding_window_stride': 11,  # Train stride
    'sliding_window_test_stride': 11,  # Also use stride=11 for quick test
}

# Single experiment for testing
EXPERIMENTS = [
    {
        'name': 'swat_test',
        'config': {},  # Use BASE_CONFIG
    },
]

# Scoring modes
SCORING_MODES = ['default']
