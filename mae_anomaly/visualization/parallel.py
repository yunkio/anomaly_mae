"""
Parallel Visualization Module

Provides multiprocessing-based parallel plot generation for faster visualization.
Uses file-based data passing to avoid IPC overhead.
"""

import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Callable, Any, Optional
import numpy as np


def _plot_worker(args):
    """Worker function that runs in a separate process.

    Args:
        args: Tuple of (plot_func_name, data_path, output_dir, config_dict, extra_args)

    Returns:
        Tuple of (plot_func_name, success, elapsed_time, error_msg)
    """
    plot_func_name, data_path, output_dir, config_dict, extra_args = args

    # Import inside worker to avoid multiprocessing issues
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend required for multiprocessing
    import matplotlib.pyplot as plt

    start_time = time.time()

    try:
        # Load data from file
        data = np.load(data_path, allow_pickle=True)
        pred_data = data['pred_data'].item()
        detailed_data = data['detailed_data'].item()

        # Reconstruct config
        from mae_anomaly import Config
        config = Config()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)

        # Create a minimal visualizer for this plot
        from mae_anomaly.visualization.best_model_visualizer import BestModelVisualizer

        # Create visualizer with pre-computed data (no GPU needed)
        visualizer = BestModelVisualizer(
            model=None,  # Not needed for most plots with pre-computed data
            config=config,
            test_loader=None,  # Not needed
            output_dir=output_dir,
            pred_data=pred_data,
            detailed_data=detailed_data
        )

        # Call the specific plot function
        plot_func = getattr(visualizer, plot_func_name)
        if extra_args:
            plot_func(**extra_args)
        else:
            plot_func()

        plt.close('all')
        elapsed = time.time() - start_time
        return (plot_func_name, True, elapsed, None)

    except Exception as e:
        import traceback
        elapsed = time.time() - start_time
        return (plot_func_name, False, elapsed, str(e))


class ParallelVisualizer:
    """Execute visualization functions in parallel using multiprocessing."""

    def __init__(self, max_workers: int = 4):
        """Initialize ParallelVisualizer.

        Args:
            max_workers: Maximum number of parallel worker processes
        """
        self.max_workers = max_workers

    def generate_parallel(
        self,
        pred_data: Dict,
        detailed_data: Dict,
        config,
        output_dir: str,
        plot_functions: List[str],
        extra_args: Dict[str, Dict] = None
    ) -> Dict[str, float]:
        """Generate multiple plots in parallel.

        Args:
            pred_data: Pre-computed prediction data
            detailed_data: Pre-computed detailed data
            config: Model configuration
            output_dir: Output directory for plots
            plot_functions: List of plot function names to execute
            extra_args: Optional dict mapping function names to extra kwargs

        Returns:
            Dict mapping function name to elapsed time
        """
        extra_args = extra_args or {}
        timings = {}

        # Save data to temporary file for workers
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            data_path = f.name
            np.savez(f,
                     pred_data=pred_data,
                     detailed_data=detailed_data)

        # Prepare config dict for serialization
        config_dict = {}
        for attr in dir(config):
            if not attr.startswith('_'):
                val = getattr(config, attr)
                if isinstance(val, (int, float, str, bool, list, tuple)):
                    config_dict[attr] = val

        try:
            # Prepare task arguments
            tasks = []
            for func_name in plot_functions:
                func_extra = extra_args.get(func_name, None)
                tasks.append((func_name, data_path, output_dir, config_dict, func_extra))

            # Execute in parallel
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(_plot_worker, task): task[0]
                          for task in tasks}

                for future in as_completed(futures):
                    func_name = futures[future]
                    try:
                        result = future.result()
                        name, success, elapsed, error = result
                        timings[name] = elapsed
                        if success:
                            print(f"  [Parallel] {name}: {elapsed:.2f}s")
                        else:
                            print(f"  [Parallel] {name}: ERROR - {error}")
                    except Exception as e:
                        print(f"  [Parallel] {func_name}: FAILED - {e}")
                        timings[func_name] = -1

        finally:
            # Cleanup temp file
            if os.path.exists(data_path):
                os.unlink(data_path)

        return timings


def generate_plots_parallel(
    visualizer,
    plot_functions: List[str],
    max_workers: int = 4,
    experiment_dir: str = None
) -> Dict[str, float]:
    """Helper function to generate plots from an existing BestModelVisualizer in parallel.

    Args:
        visualizer: BestModelVisualizer instance
        plot_functions: List of plot function names
        max_workers: Number of parallel workers
        experiment_dir: Experiment directory for plots that need it

    Returns:
        Dict mapping function name to elapsed time
    """
    # Prepare extra args for functions that need experiment_dir
    extra_args = {}
    funcs_needing_exp_dir = [
        'plot_score_contribution_analysis',
        'plot_case_study_gallery',
        'plot_anomaly_type_case_studies',
        'plot_performance_by_anomaly_type',
        'plot_score_distribution_by_type',
    ]
    for func in funcs_needing_exp_dir:
        if func in plot_functions:
            extra_args[func] = {'experiment_dir': experiment_dir}

    parallel_viz = ParallelVisualizer(max_workers=max_workers)
    return parallel_viz.generate_parallel(
        pred_data=visualizer.pred_data,
        detailed_data=visualizer.detailed_data,
        config=visualizer.config,
        output_dir=visualizer.output_dir,
        plot_functions=plot_functions,
        extra_args=extra_args
    )
