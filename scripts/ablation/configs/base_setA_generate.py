#!/usr/bin/env python
"""Generate Set A config files for all datasets (w500_p5_e2_t4_d1, d_model=128, cnn_k=3)."""

import os

CONFIGS_DIR = os.path.dirname(os.path.abspath(__file__))
SET_DIR = os.path.join(CONFIGS_DIR, 'setA_w500p5e2t4d1')
os.makedirs(SET_DIR, exist_ok=True)

# Common base config for Set A
BASE = {
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
    'learning_rate': 1e-3,
    'batch_size': 256,
    'eval_interval': 5,
    'cnn_kernel_size': 3,
    'sliding_window_test_stride': 1,
    'patchify_mode': "'patch_cnn'",
    'mask_after_encoder': True,
    'anomaly_score_mode': "'adaptive'",
}

# Dataset definitions: (filename, DATASET_TYPE, stride, noise_ratio, description)
DATASETS = [
    # SWaT
    ('swat_A1A2',              'swat_A1A2',        11, 0.0, 'SWaT A1+A2 combined'),
    ('swat_A1A2_normal50',     'swat_A1A2',        11, 0.5, 'SWaT A1+A2 with 50% label noise'),
    ('swat_A1A2_swap',         'swat_A1A2_swap',   11, 0.0, 'SWaT A1+A2 with swapped A2 halves'),
    # WaDi 50:50
    ('wadi_A1',                'wadi_A1',           3, 0.0, 'WaDi A1 attack 50:50 split'),
    ('wadi_A1_swap',           'wadi_A1_swap',      3, 0.0, 'WaDi A1 attack swapped 50:50'),
    ('wadi_A2',                'wadi_A2',           3, 0.0, 'WaDi A2 attack 50:50 split'),
    ('wadi_A2_swap',           'wadi_A2_swap',      3, 0.0, 'WaDi A2 attack swapped 50:50'),
    # WaDi 14days
    ('wadi_14days_A1',         'wadi_14days_A1',   11, 0.0, 'WaDi 14days + A1'),
    ('wadi_14days_A1_normal50','wadi_14days_A1',   11, 0.5, 'WaDi 14days + A1 with 50% label noise'),
    ('wadi_14days_A2',         'wadi_14days_A2',   11, 0.0, 'WaDi 14days + A2'),
    ('wadi_14days_A2_normal50','wadi_14days_A2',   11, 0.5, 'WaDi 14days + A2 with 50% label noise'),
    # Simulation
    ('simulation',             'simulation',        3, 0.0, 'Simulation (complexity=False)'),
    ('simulation_normal50',    'simulation',        3, 0.5, 'Simulation with 50% label noise'),
    ('simulation_complex',     'simulation_complex', 3, 0.0, 'Simulation (complexity=True)'),
    ('simulation_complex_normal50', 'simulation_complex', 3, 0.5, 'Simulation complex with 50% label noise'),
]


def generate_config(name, dataset_type, stride, noise_ratio, description):
    lines = [
        f'"""Set A: {description}"""',
        f'',
        f"DATASET_TYPE = '{dataset_type}'",
        f'PHASE_NAME = "{name}"',
        f'PHASE_DESCRIPTION = "{description}"',
        f'',
    ]

    if noise_ratio > 0:
        lines.append(f'LABEL_NOISE_RATIO = {noise_ratio}')
        lines.append('')

    lines.append('BASE_CONFIG = {')
    for key, val in BASE.items():
        if key == 'sliding_window_test_stride':
            continue  # will add with stride
        lines.append(f"    '{key}': {val},")
    lines.append(f"    'sliding_window_stride': {stride},")
    lines.append(f"    'sliding_window_test_stride': 1,")
    lines.append('}')
    lines.append('')
    lines.append('EXPERIMENTS = [')
    lines.append("    {'name': 'default', 'config': {}},")
    lines.append(']')
    lines.append('')
    lines.append("SCORING_MODES = ['adaptive']")
    lines.append("MASK_SETTINGS = [True]")
    lines.append('')

    return '\n'.join(lines)


if __name__ == '__main__':
    for name, dataset_type, stride, noise_ratio, description in DATASETS:
        content = generate_config(name, dataset_type, stride, noise_ratio, description)
        filepath = os.path.join(SET_DIR, f'{name}.py')
        with open(filepath, 'w') as f:
            f.write(content)
        print(f'  Created: {filepath}')
    print(f'\nGenerated {len(DATASETS)} config files in {SET_DIR}')
