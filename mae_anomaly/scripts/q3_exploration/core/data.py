"""
Data loading utilities for Q3 exploration.

Provides:
- DatasetScores: encapsulates saved LoO scores + labels + regions
- load_dataset: lazy load from saved npz
- iter_all_datasets: iterator over 39 datasets with proper swat_excl22 handling
"""
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np

LOO_DIR = Path('/home/ykio/notebooks/claude/temp/loo_tail_scores_s1')
HID_DIR = Path('/home/ykio/notebooks/claude/temp/tier1_hiddens')
GA1_DIR = Path('/home/ykio/notebooks/claude/temp/E18_GA1_scores')


@dataclass
class AnomalyRegion:
    start: int
    end: int
    anomaly_type: int = 1


def regions_from_labels(labels):
    """0/1 label array → list of AnomalyRegion."""
    regions = []
    in_e, st = False, None
    for i, v in enumerate(labels):
        if v == 1 and not in_e:
            st, in_e = i, True
        elif v == 0 and in_e:
            regions.append(AnomalyRegion(start=st, end=i))
            in_e = False
    if in_e:
        regions.append(AnomalyRegion(start=st, end=len(labels)))
    return regions


def find_swat_largest_region(regions):
    """SwAT excl22: largest region 식별."""
    if not regions:
        return None
    return max(regions, key=lambda r: r.end - r.start)


@dataclass
class DatasetScores:
    """LoO saved scores + metadata 한 dataset."""
    alias: str
    recon: np.ndarray            # (n_w, num_patches)
    disc: np.ndarray
    student: np.ndarray
    fm: np.ndarray
    window_start_indices: np.ndarray
    point_labels: np.ndarray
    patch_size: int
    num_patches: int
    swat_excl22: bool = False
    regions: list = field(default_factory=list)
    eval_mask: np.ndarray = None  # for swat_excl22
    total_length: int = 0

    @classmethod
    def load(cls, alias, swat_excl22=False):
        """Load LoO scores from saved npz."""
        npz_name = 'swat_full' if swat_excl22 else alias
        f = LOO_DIR / f'{npz_name}.npz'
        if not f.exists():
            return None
        d = np.load(f)
        labels = d['point_labels'].astype(np.int64)
        total_length = len(labels)
        regions = regions_from_labels(labels)
        eval_mask = None
        if swat_excl22:
            largest = find_swat_largest_region(regions)
            if largest is not None:
                eval_mask = np.ones(total_length, dtype=bool)
                eval_mask[largest.start:largest.end] = False
                regions = [r for r in regions
                           if not (r.start == largest.start and r.end == largest.end)]
        return cls(
            alias=alias,
            recon=d['recon'], disc=d['disc'],
            student=d.get('student', np.zeros_like(d['recon'])),
            fm=d['fm'],
            window_start_indices=d['window_start_indices'],
            point_labels=labels,
            patch_size=int(d['patch_size']),
            num_patches=int(d['num_patches']),
            swat_excl22=swat_excl22,
            regions=regions, eval_mask=eval_mask,
            total_length=total_length,
        )


def iter_dataset_aliases():
    """39 datasets (swat_excl22 포함)."""
    available = {f.stem for f in LOO_DIR.glob('*.npz')}
    smd = sorted([a for a in available if a.startswith('smd_')])
    exa = sorted([a for a in available if a.startswith('exathlon_')])
    standalone = [a for a in ['swat_full', 'psm', 'wadi_A1', 'wadi_A2', 'simulation']
                  if a in available]
    targets = []
    if 'swat_full' in available:
        targets.append(('swat_excl22', True))
    for a in standalone:
        if a != 'swat_full':
            targets.append((a, False))
    targets += [(a, False) for a in smd]
    targets += [(a, False) for a in exa]
    return targets


def median_anomaly_segment_length(regions):
    """median anomaly segment length (E9 σ를 위한)."""
    if not regions:
        return 10.0
    seg_lens = [r.end - r.start for r in regions]
    return float(np.median(seg_lens))


def get_per_group(alias):
    """SMD / Exathlon / Standalone 분류."""
    if alias.startswith('smd_'):
        return 'SMD'
    if alias.startswith('exathlon_'):
        return 'Exathlon'
    return 'Standalone'
