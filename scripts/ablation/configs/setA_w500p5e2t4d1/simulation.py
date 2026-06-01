"""Set A: Simulation (complexity=False)"""

DATASET_TYPE = 'simulation'
PHASE_NAME = "simulation"
PHASE_DESCRIPTION = "Simulation (complexity=False)"

BASE_CONFIG = {
    'seq_length': 500,
    'patch_size': 5,
    'num_patches': 100,
    'd_model': 128,
    'nhead': 8,
    'dim_feedforward': 512,
    'num_encoder_layers': 2,
    'num_teacher_decoder_layers': 4,
    'num_student_decoder_layers': 1,
    'num_epochs': 50,
    'learning_rate': 0.001,
    'batch_size': 256,
    'eval_interval': 5,
    'cnn_kernel_size': 3,
    'patchify_mode': 'patch_cnn',
    'mask_after_encoder': True,
    'anomaly_score_mode': 'adaptive',
    'sliding_window_stride': 3,
    'sliding_window_test_stride': 1,
}

EXPERIMENTS = [
    {'name': 'default', 'config': {}},
]

SCORING_MODES = ['adaptive']
MASK_SETTINGS = [True]
