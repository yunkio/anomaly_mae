"""Set B: WaDi 14days + A1 with 50% label noise (p20, d128, k5)"""

DATASET_TYPE = 'wadi_14days_A1'
PHASE_NAME = "wadi_14days_A1_normal50"
PHASE_DESCRIPTION = "WaDi 14days + A1 with 50% label noise (p20, d128, k5)"

LABEL_NOISE_RATIO = 0.5

BASE_CONFIG = {
    'seq_length': 500,
    'patch_size': 20,
    'num_patches': 25,
    'd_model': 128,
    'nhead': 8,
    'dim_feedforward': 512,
    'num_encoder_layers': 2,
    'num_teacher_decoder_layers': 4,
    'num_student_decoder_layers': 1,
    'num_epochs': 50,
    'learning_rate': 0.002,
    'batch_size': 512,
    'eval_interval': 5,
    'cnn_kernel_size': 5,
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
