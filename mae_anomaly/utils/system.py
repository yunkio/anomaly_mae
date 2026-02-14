"""System utilities for GPU and memory management."""

import gc
import torch
import psutil


def free_gpu():
    """Aggressively free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def mem_status():
    """Get memory status string."""
    import psutil
    m = psutil.virtual_memory()
    s = f"RAM:{m.used/1e9:.1f}/{m.total/1e9:.1f}G"
    if torch.cuda.is_available():
        u = torch.cuda.memory_allocated() / 1e6
        t = torch.cuda.get_device_properties(0).total_memory / 1e6
        s += f" GPU:{u:.0f}/{t:.0f}M"
    return s


# =============================================================================
# Data Loading for 14days + Attack
# =============================================================================


