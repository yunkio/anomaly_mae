"""
Synthetic Anomaly Generation + Model Response Measurement.

본 module은:
- Synthetic anomaly inject into normal signals
- Measure 274 model's response (saved scores)에서 추정 — 274 model 자체는 retraining 없음
- 본 측정으로 model이 어떤 anomaly type에 strong/weak detection 보이는지 정량
"""
import numpy as np
from typing import Dict, List, Tuple, Callable


def inject_spike_anomaly(signal, position, length, magnitude=3.0):
    """Spike anomaly: magnitude * std로 multiplicative."""
    sig = signal.copy()
    region = sig[position:position+length]
    std = signal.std(axis=0)
    sig[position:position+length] = region + magnitude * std
    return sig


def inject_level_shift(signal, position, length, magnitude=2.0):
    """Level shift: constant offset."""
    sig = signal.copy()
    std = signal.std(axis=0)
    sig[position:position+length] += magnitude * std
    return sig


def inject_noise_burst(signal, position, length, multiplier=3.0):
    """Increase variance temporarily."""
    sig = signal.copy()
    n_features = signal.shape[1]
    noise = np.random.randn(length, n_features) * signal.std(axis=0) * multiplier
    sig[position:position+length] += noise
    return sig


def inject_shuffle(signal, position, length):
    """Permute the segment temporally."""
    sig = signal.copy()
    region = sig[position:position+length].copy()
    np.random.shuffle(region)
    sig[position:position+length] = region
    return sig


def inject_drift(signal, position, length, end_magnitude=2.0):
    """Gradual linear drift."""
    sig = signal.copy()
    std = signal.std(axis=0)
    drift = np.linspace(0, end_magnitude, length).reshape(-1, 1) * std
    sig[position:position+length] += drift
    return sig


def inject_frequency_shift(signal, position, length, freq_multiplier=2.0):
    """Modulate with different frequency content (sinusoidal addition)."""
    sig = signal.copy()
    n_features = signal.shape[1]
    t = np.arange(length).reshape(-1, 1)
    base_freq = 1.0 / 50.0
    modulation = 0.5 * signal.std(axis=0) * np.sin(2 * np.pi * base_freq * freq_multiplier * t)
    sig[position:position+length] += modulation
    return sig


def generate_synthetic_anomaly_set(clean_signal, n_anomalies=20, anomaly_length_range=(5, 50),
                                     types=('spike', 'level_shift', 'noise', 'shuffle',
                                            'drift', 'freq_shift'),
                                     random_state=42):
    """Generate clean_signal with N synthetic anomalies injected at random positions.

    Returns: (modified_signal, labels, anomaly_metadata)
    """
    rng = np.random.RandomState(random_state)
    sig = clean_signal.copy()
    total_length = len(sig)
    labels = np.zeros(total_length)
    metadata = []

    for _ in range(n_anomalies):
        anomaly_type = rng.choice(types)
        length = rng.randint(anomaly_length_range[0], anomaly_length_range[1] + 1)
        # Avoid overlapping with previously injected
        attempt = 0
        while attempt < 20:
            position = rng.randint(0, total_length - length)
            if labels[position:position+length].sum() == 0:
                break
            attempt += 1
        else:
            continue

        if anomaly_type == 'spike':
            magnitude = rng.uniform(2.0, 5.0)
            sig = inject_spike_anomaly(sig, position, length, magnitude)
            params = {'magnitude': magnitude}
        elif anomaly_type == 'level_shift':
            magnitude = rng.uniform(-3, 3)
            sig = inject_level_shift(sig, position, length, magnitude)
            params = {'magnitude': magnitude}
        elif anomaly_type == 'noise':
            mult = rng.uniform(2.0, 5.0)
            sig = inject_noise_burst(sig, position, length, mult)
            params = {'multiplier': mult}
        elif anomaly_type == 'shuffle':
            sig = inject_shuffle(sig, position, length)
            params = {}
        elif anomaly_type == 'drift':
            end_mag = rng.uniform(1.0, 4.0)
            sig = inject_drift(sig, position, length, end_mag)
            params = {'end_magnitude': end_mag}
        elif anomaly_type == 'freq_shift':
            mult = rng.uniform(1.5, 4.0)
            sig = inject_frequency_shift(sig, position, length, mult)
            params = {'freq_multiplier': mult}
        else:
            continue

        labels[position:position+length] = 1
        metadata.append({
            'type': anomaly_type,
            'position': position,
            'length': length,
            'params': params,
        })

    return sig, labels, metadata


def measure_anomaly_response(reconstruct_func, original_signal, modified_signal,
                                anomaly_metadata):
    """Per-injected-anomaly: model's response (recon error in/out of anomaly).

    Args:
        reconstruct_func: function (signal) → reconstructed_signal
        original_signal: clean signal
        modified_signal: with injected anomalies
        anomaly_metadata: from generate_synthetic_anomaly_set

    Returns: per-anomaly response metrics.
    """
    recon_modified = reconstruct_func(modified_signal)
    recon_error = np.abs(modified_signal - recon_modified).mean(axis=1)

    responses = []
    for meta in anomaly_metadata:
        pos = meta['position']
        length = meta['length']
        in_recon = recon_error[pos:pos+length].mean()
        # Context (200 timesteps before+after)
        ctx_start = max(0, pos - 200)
        ctx_end = min(len(recon_error), pos + length + 200)
        ctx_indices = np.concatenate([
            np.arange(ctx_start, pos),
            np.arange(pos + length, ctx_end),
        ])
        ctx_recon = recon_error[ctx_indices].mean() if len(ctx_indices) > 0 else 0
        ctx_std = recon_error[ctx_indices].std() if len(ctx_indices) > 5 else 1.0

        responses.append({
            **meta,
            'in_recon_error': float(in_recon),
            'ctx_recon_error': float(ctx_recon),
            'response_ratio': float(in_recon / (ctx_recon + 1e-9)),
            'normalized_response': float((in_recon - ctx_recon) / (ctx_std + 1e-9)),
            'detected_well': bool(in_recon > ctx_recon),
        })
    return responses
