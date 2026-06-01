"""
Test config for Simulation dataset (backward compatibility test)

This config tests that the default 'simulation' dataset type still works.

Usage:
    python scripts/ablation/run_ablation.py --config scripts/ablation/configs/simulation_test.py
"""

# Dataset configuration (optional - defaults to 'simulation')
DATASET_TYPE = 'simulation'

# Phase metadata
PHASE_NAME = "simulation_test"
PHASE_DESCRIPTION = "Simulation dataset test (backward compatibility)"

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

    # Simulation dataset params
    'use_sliding_window_dataset': True,
    'sliding_window_total_length': 27500,  # 1/10 of normal (275000)
    'sliding_window_stride': 11,
    'sliding_window_test_stride': 11,
}

# Single experiment for testing
EXPERIMENTS = [
    {
        'name': 'sim_test',
        'config': {},  # Use BASE_CONFIG
    },
]

# Scoring modes
SCORING_MODES = ['default']
