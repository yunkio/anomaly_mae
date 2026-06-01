"""Resume checkpoint utilities for weak SSL baselines (2026-05-31).

Design (B option, user decision 2026-05-31):
  - Epoch-level dynamics consistency, NOT byte-level identical reproducibility.
  - We capture/restore RNG state so resume continues with the same learning
    trajectory the from-scratch run would have, but we DO NOT force
    `cudnn.deterministic` / `use_deterministic_algorithms` (would add 30–100%
    training time per PyTorch docs reproducibility notes and risks unsupported
    ops in wetas SoftDTW custom CUDA kernel).
  - Two files per (dataset, model):
      checkpoints/last.pt  — overwritten after every visible epoch
      checkpoints/best.pt  — copied from last.pt by baseline_common when
                             pak_auc_f1 improves (file-level copy, no extra
                             save logic in wrappers)
  - Atomic write via tmp + rename so a crash mid-save cannot leave a corrupt
    last.pt that would block resume.
  - Schema version stamp so future format changes can refuse incompatible
    checkpoints instead of silently corrupting a long run.

Scope: SSL 5 (deepmil, wetas, treemil, nrdetector, nrdetector_full). Other
baselines (catch/npsr/dagmm/dcdetector/etc.) are NOT modified — they don't
import this module.
"""

import os
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


CHECKPOINT_VERSION = 1


def capture_rng_state() -> dict:
    """Capture global RNG state across torch / torch.cuda / numpy / python.

    **Pre-eval snapshot semantics** (intentional, do NOT change without
    auditing all wrappers): every wrapper's fit() loop calls
    ``self._save_last_checkpoint(epoch + 1, optimizer)`` BEFORE calling
    ``epoch_callback`` (which runs the long predict + metrics eval). So a
    checkpoint captured here contains the RNG state AFTER ep N's training
    finishes but BEFORE ep N's evaluation consumes any RNG. This is a
    crash-safety choice — if the eval crashes (OOM, VUS exception, etc.) we
    keep a resumable checkpoint for the just-finished training epoch.
    Resume-after-ep-N-train + eval-on-resume therefore starts ep N+1 with
    RNG state slightly different from from-scratch-ep-N-train + eval +
    ep-N+1, because the from-scratch path consumed RNG in ep N's eval
    (predict-time scaler fits, torch.no_grad inference, dropout=False so no
    masks, but CUDA non-determinism still touches state). This is an
    inherent B-option trade-off; for byte-level reproducibility see the
    rejected A-option design (cudnn.deterministic etc., not implemented).

    Each wrapper that holds an *additional* RNG (e.g. deepmil's
    ``np.random.default_rng(42)`` bag-sampling generator) MUST persist that
    state itself in its checkpoint dict alongside the global state from this
    helper — torch-global capture does not see wrapper-local Generators.
    """
    state = {
        'torch_cpu': torch.get_rng_state(),
        'numpy_global': np.random.get_state(),
        'python': random.getstate(),
    }
    if torch.cuda.is_available():
        state['torch_cuda_all'] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Optional[dict]) -> None:
    """Restore global RNG state from a dict captured by ``capture_rng_state``.

    Missing keys are silently ignored (forward compatibility — older
    checkpoints without a CUDA RNG, or saved on a CPU-only host that is now
    resuming on GPU, must not break resume).
    """
    if not state:
        return
    if 'torch_cpu' in state:
        torch.set_rng_state(state['torch_cpu'])
    if 'torch_cuda_all' in state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state['torch_cuda_all'])
        except Exception:
            # CUDA device count or driver mismatch — non-fatal under B.
            pass
    if 'numpy_global' in state:
        np.random.set_state(state['numpy_global'])
    if 'python' in state:
        random.setstate(state['python'])


def atomic_save(obj: Any, path: Path) -> None:
    """torch.save to a sibling .tmp then os.replace -> atomic rename.

    Avoids leaving a half-written ``last.pt`` if the process is killed during
    the write — partial files are the typical resume-poison failure mode.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path) -> dict:
    """Load + version-check a checkpoint produced by ``atomic_save``.

    Raises RuntimeError on a version mismatch rather than silently loading —
    a partial state restore would break learning more subtly than a crash.
    """
    ckpt = torch.load(path, map_location='cpu')
    version = ckpt.get('checkpoint_version', 0)
    if version != CHECKPOINT_VERSION:
        raise RuntimeError(
            f"Checkpoint version mismatch at {path}: file={version} "
            f"expected={CHECKPOINT_VERSION}. Refuse to resume; either re-run "
            f"from scratch with --force, or upgrade/downgrade the wrapper."
        )
    return ckpt


def standard_scaler_to_dict(scaler) -> Optional[dict]:
    """Serialize a (possibly None) sklearn StandardScaler to a JSON-safe dict.

    Returns None if the scaler is None or has not been fit yet. Mirrors the
    existing ``train_scaler`` schema used by deepmil/wetas/nrdetector
    ``save()/load()`` methods so a checkpoint and a final ``model/config.json``
    use the same serialization shape.
    """
    if scaler is None or not hasattr(scaler, 'mean_'):
        return None
    return {
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist(),
        'var': scaler.var_.tolist(),
        'n_samples_seen': int(scaler.n_samples_seen_),
    }


def dict_to_standard_scaler(blob: Optional[dict]):
    """Reconstruct a StandardScaler from ``standard_scaler_to_dict`` output."""
    if not blob:
        return None
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(blob['mean'], dtype=np.float64)
    scaler.scale_ = np.asarray(blob['scale'], dtype=np.float64)
    scaler.var_ = np.asarray(blob['var'], dtype=np.float64)
    scaler.n_samples_seen_ = blob['n_samples_seen']
    scaler.n_features_in_ = len(blob['mean'])
    return scaler
