"""
MAE Anomaly Detection Visualization Module

This module provides visualization tools for:
- Data visualizations (dataset, anomaly types, features)
- Architecture visualizations (model pipeline, masking)
- Experiment visualizations (Stage 1/2 results)
- Best model analysis
- Training progress visualization

Usage:
    from mae_anomaly.visualization import (
        DataVisualizer,
        ArchitectureVisualizer,
        ExperimentVisualizer,
        Stage2Visualizer,
        BestModelVisualizer,
        TrainingProgressVisualizer,
    )
"""

from .base import (
    setup_style,
    find_latest_experiment,
    load_experiment_data,
    load_best_model,
    collect_predictions,
    collect_detailed_data,
    get_anomaly_colors,
    get_feature_colors,
    get_anomaly_type_info,
    SAMPLE_TYPE_NAMES,
    SAMPLE_TYPE_COLORS,
    VIS_COLORS,
    VIS_MARKERS,
    VIS_LINESTYLES,
)

from .data_visualizer import DataVisualizer
from .architecture_visualizer import ArchitectureVisualizer
from .experiment_visualizer import ExperimentVisualizer
from .stage2_visualizer import Stage2Visualizer
from .best_model_visualizer import BestModelVisualizer
from .training_visualizer import TrainingProgressVisualizer

__all__ = [
    # Base utilities
    'setup_style',
    'find_latest_experiment',
    'load_experiment_data',
    'load_best_model',
    'collect_predictions',
    'collect_detailed_data',
    'get_anomaly_colors',
    'get_feature_colors',
    'get_anomaly_type_info',
    'SAMPLE_TYPE_NAMES',
    'SAMPLE_TYPE_COLORS',
    'VIS_COLORS',
    'VIS_MARKERS',
    'VIS_LINESTYLES',
    # Visualizers
    'DataVisualizer',
    'ArchitectureVisualizer',
    'ExperimentVisualizer',
    'Stage2Visualizer',
    'BestModelVisualizer',
    'TrainingProgressVisualizer',
]
