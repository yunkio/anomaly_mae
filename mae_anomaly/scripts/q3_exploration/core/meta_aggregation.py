"""
Meta-analysis result aggregation utilities.

본 module은 Q3 v1-v4 + 2분기 보고서의 모든 result JSON을 single matrix로 통합:

- ResultMatrix: alias × method_name → mean_Δ (float)
- 다양한 JSON schema에 대한 parser
- Method metadata (origin, family, semi-supervised flag)
- 통합 후 export to pandas DataFrame
"""
import json
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class MethodEntry:
    """Single method's result across datasets."""
    name: str
    family: str
    source: str  # 'Q3v1', 'Q3v2', 'Q3v3', 'Q3v4', '2분기'
    semi_supervised: bool = False
    requires_clustering: bool = False
    description: str = ""
    per_dataset_pak: Dict[str, float] = field(default_factory=dict)
    per_dataset_baseline: Dict[str, float] = field(default_factory=dict)

    def delta_for(self, alias):
        if alias not in self.per_dataset_pak or alias not in self.per_dataset_baseline:
            return None
        return self.per_dataset_pak[alias] - self.per_dataset_baseline[alias]

    @property
    def aliases(self):
        return list(self.per_dataset_pak.keys())


@dataclass
class MetaResultMatrix:
    """All methods × datasets results."""
    methods: Dict[str, MethodEntry] = field(default_factory=dict)
    all_aliases: List[str] = field(default_factory=list)

    def add_method(self, method: MethodEntry):
        self.methods[method.name] = method
        for alias in method.aliases:
            if alias not in self.all_aliases:
                self.all_aliases.append(alias)

    def to_matrix(self):
        """Return (aliases, method_names, delta_matrix).
        delta_matrix[i,j] = Δ_pak for (alias_i, method_j). NaN if missing.
        """
        aliases = sorted(self.all_aliases)
        method_names = sorted(self.methods.keys())
        matrix = np.full((len(aliases), len(method_names)), np.nan)
        for j, mname in enumerate(method_names):
            m = self.methods[mname]
            for i, alias in enumerate(aliases):
                d = m.delta_for(alias)
                if d is not None:
                    matrix[i, j] = d
        return aliases, method_names, matrix

    def to_pandas(self):
        """Export as DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            return None
        aliases, methods, matrix = self.to_matrix()
        df = pd.DataFrame(matrix, index=aliases, columns=methods)
        return df

    def summary(self):
        """Print summary statistics."""
        n_methods = len(self.methods)
        n_aliases = len(self.all_aliases)
        family_counts = {}
        for m in self.methods.values():
            family_counts[m.family] = family_counts.get(m.family, 0) + 1
        source_counts = {}
        for m in self.methods.values():
            source_counts[m.source] = source_counts.get(m.source, 0) + 1
        return {
            'n_methods': n_methods,
            'n_datasets': n_aliases,
            'family_distribution': family_counts,
            'source_distribution': source_counts,
        }


# ================== Parsers for various JSON formats ==================

def _safe_load_json(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


def parse_q3v1_phaseA(path, matrix: MetaResultMatrix):
    """phaseA_unsupervised_sigma.json"""
    d = _safe_load_json(path)
    if d is None: return
    methods = ['e9', 'a1', 'a2', 'a3', 'a4']
    method_meta = {
        'e9': ('E9_adapt_single', 'adaptive_sigma', True),
        'a1': ('A1_peak_run', 'unsup_sigma_estimation', False),
        'a2': ('A2_multi_sigma_agree', 'unsup_sigma_estimation', False),
        'a3': ('A3_kde_fwhm', 'unsup_sigma_estimation', False),
        'a4': ('A4_smoothed_peak', 'unsup_sigma_estimation', False),
    }
    for short_name, (full_name, family, is_supervised) in method_meta.items():
        entry = MethodEntry(name=full_name, family=family, source='Q3v1',
                            semi_supervised=is_supervised,
                            description=f"Phase A {short_name}")
        for alias, r in d.items():
            entry.per_dataset_baseline[alias] = r.get('baseline_pak', 0)
            entry.per_dataset_pak[alias] = r.get(f'{short_name}_pak', r.get('baseline_pak', 0))
        matrix.add_method(entry)


def parse_q3v1_phaseB(path, matrix: MetaResultMatrix):
    """phaseB_hybrid.json"""
    d = _safe_load_json(path)
    if d is None: return
    methods = {
        'b1_pak': ('B1_E9_NLM_T2', 'hybrid'),
        'b2_pak': ('B2_conditional_cap', 'hybrid_conditional'),
        'b3_pak': ('B3_3method_ensemble', 'ensemble'),
        'b4_pak': ('B4_unsup_A3_NLM', 'hybrid'),
        'b5_pak': ('B5_Z5_E9_routing', 'hybrid_conditional'),
    }
    for short_name, (full_name, family) in methods.items():
        entry = MethodEntry(name=full_name, family=family, source='Q3v1',
                            semi_supervised=True)
        for alias, r in d.items():
            entry.per_dataset_baseline[alias] = r.get('baseline_pak', 0)
            entry.per_dataset_pak[alias] = r.get(short_name, r.get('baseline_pak', 0))
        matrix.add_method(entry)


def parse_q3v1_F2(path, matrix: MetaResultMatrix):
    """F2_cross_channel.json"""
    d = _safe_load_json(path)
    if d is None: return
    methods = ['f2_1_pak', 'f2_2_pak', 'f2_3_pak', 'f2_4_pak',
               'f2_5_pak', 'f2_6_pak', 'f2_7_pak']
    desc = {
        'f2_1_pak': 'F2_geom_rd', 'f2_2_pak': 'F2_teacher_student_diff',
        'f2_3_pak': 'F2_sqrt_rdf', 'f2_4_pak': 'F2_r_sigmoid_d',
        'f2_5_pak': 'F2_max_channel', 'f2_6_pak': 'F2_harmonic',
        'f2_7_pak': 'F2_iqr_weighted',
    }
    for sn in methods:
        entry = MethodEntry(name=desc[sn], family='cross_channel', source='Q3v1')
        for alias, r in d.items():
            entry.per_dataset_baseline[alias] = r.get('baseline_pak', 0)
            entry.per_dataset_pak[alias] = r.get(sn, r.get('baseline_pak', 0))
        matrix.add_method(entry)


def parse_q3v1_F5(path, matrix: MetaResultMatrix):
    """F5_dataset_clustering.json"""
    d = _safe_load_json(path)
    if d is None: return
    if 'method_scores' not in d: return
    method_scores = d['method_scores']
    # 모든 method name 추출
    sample = next(iter(method_scores.values()))
    method_names = list(sample.keys())
    for mn in method_names:
        family = 'multi_scale' if 'z5' in mn else 'sigma_smoothing'
        if 'gauss' in mn: family = 'sigma_smoothing'
        if 'b1' in mn or 'b2' in mn: family = 'hybrid'
        entry = MethodEntry(name=f'F5_{mn}', family=family, source='Q3v1')
        for alias, ms in method_scores.items():
            entry.per_dataset_baseline[alias] = ms.get('baseline_gauss10', 0)
            entry.per_dataset_pak[alias] = ms.get(mn, ms.get('baseline_gauss10', 0))
        matrix.add_method(entry)


def parse_q3v2_P2(path, matrix: MetaResultMatrix):
    """P2_fine_sigma_sweep.json — large grid"""
    d = _safe_load_json(path)
    if d is None: return
    # Sample to get keys
    sample_alias = next(iter(d.keys()))
    grid_keys = list(d[sample_alias]['grid'].keys())
    for gk in grid_keys:
        family = 'sigma_NLM_grid'
        entry = MethodEntry(name=f'P2_{gk}', family=family, source='Q3v2',
                            semi_supervised=True)
        for alias, r in d.items():
            entry.per_dataset_baseline[alias] = r.get('baseline_pak', 0)
            entry.per_dataset_pak[alias] = r['grid'][gk].get('pak', r.get('baseline_pak', 0))
        matrix.add_method(entry)


def parse_q3v2_P4(path, matrix: MetaResultMatrix):
    """P4_threshold_optimization.json"""
    d = _safe_load_json(path)
    if d is None: return
    sample = next(iter(d.values()))
    method_names = list(sample['methods'].keys())
    for mn in method_names:
        family = 'threshold_opt'
        entry = MethodEntry(name=f'P4_{mn}_auc', family=family, source='Q3v2')
        entry_bestF1 = MethodEntry(name=f'P4_{mn}_bestF1', family='threshold_opt_bestF1', source='Q3v2')
        for alias, r in d.items():
            # baseline은 baseline_gauss10의 AUC F1
            base = r['methods'].get('baseline_gauss10', {}).get('auc_f1', 0)
            entry.per_dataset_baseline[alias] = base
            entry.per_dataset_pak[alias] = r['methods'][mn].get('auc_f1', base)

            base_bestF1 = r['methods'].get('baseline_gauss10', {}).get('best_f1', 0)
            entry_bestF1.per_dataset_baseline[alias] = base_bestF1
            entry_bestF1.per_dataset_pak[alias] = r['methods'][mn].get('best_f1', base_bestF1)
        matrix.add_method(entry)
        matrix.add_method(entry_bestF1)


def parse_q3v2_P6(path, matrix: MetaResultMatrix):
    """P6_multi_stride.json"""
    d = _safe_load_json(path)
    if d is None: return
    strides = [1, 7, 14, 21, 42, 63]
    ensembles = ['e1_mean_pak', 'e2_weighted_pak', 'e3_max_pak',
                 'e4_median_pak', 'e5_trim_pak', 'e6_small_mean_pak', 'e7_close_pak']
    # Per-stride
    for s in strides:
        entry = MethodEntry(name=f'P6_stride_{s}', family='multi_stride', source='Q3v2')
        for alias, r in d.items():
            entry.per_dataset_baseline[alias] = r.get('baseline_pak', 0)
            entry.per_dataset_pak[alias] = r['per_stride_paks'].get(str(s),
                                                                     r['per_stride_paks'].get(s, r.get('baseline_pak', 0)))
        matrix.add_method(entry)
    # Ensembles
    for e in ensembles:
        entry = MethodEntry(name=f'P6_{e}', family='multi_stride_ensemble', source='Q3v2')
        for alias, r in d.items():
            entry.per_dataset_baseline[alias] = r.get('baseline_pak', 0)
            entry.per_dataset_pak[alias] = r.get(e, r.get('baseline_pak', 0))
        matrix.add_method(entry)


def parse_q3v2_P8(path, matrix: MetaResultMatrix):
    """P8_tri_routing_v2.json — has both routing results and individual method summaries"""
    d = _safe_load_json(path)
    if d is None: return
    # Single method summaries
    if 'method_summaries' in d:
        for mn, ms in d['method_summaries'].items():
            entry = MethodEntry(name=f'P8_{mn}', family='sigma_NLM_stride_grid',
                                source='Q3v2', semi_supervised=True)
            # mean_delta는 단일 값. Per-dataset 정보 없음 → skip in matrix
            # Use mean_delta as flat value
            # (Per-dataset이 available한 경우 다른 path 사용 필요)
        # Skip single methods from P8 since per-dataset not directly available


def parse_q3v3_P9(path, matrix: MetaResultMatrix):
    """P9_unsupervised_seg.json"""
    d = _safe_load_json(path)
    if d is None: return
    sample = next(iter(d.values()))
    est_names = list(sample['estimators'].keys())
    for est in est_names:
        entry = MethodEntry(name=f'P9_{est}', family='unsup_sigma_estimation',
                            source='Q3v3', semi_supervised=False)
        for alias, r in d.items():
            entry.per_dataset_baseline[alias] = r.get('baseline_pak', 0)
            entry.per_dataset_pak[alias] = r['estimators'][est].get('pak',
                                                                       r.get('baseline_pak', 0))
        matrix.add_method(entry)


def parse_q3v3_P12(path, matrix: MetaResultMatrix):
    """P12_anomaly_type.json"""
    d = _safe_load_json(path)
    if d is None: return
    method_keys = ['discrete_type_pak', 'blend_type_pak', 'ref_pak_div5_T15']
    families = {'discrete_type_pak': 'type_routing', 'blend_type_pak': 'type_blend',
                'ref_pak_div5_T15': 'reference'}
    for mk in method_keys:
        family = families[mk]
        entry = MethodEntry(name=f'P12_{mk}', family=family, source='Q3v3',
                            semi_supervised=True)
        for alias, r in d.items():
            entry.per_dataset_baseline[alias] = r.get('baseline_pak', 0)
            entry.per_dataset_pak[alias] = r.get(mk, r.get('baseline_pak', 0))
        matrix.add_method(entry)


def parse_q3v3_P13(path, matrix: MetaResultMatrix):
    """P13_iterative_refinement.json"""
    d = _safe_load_json(path)
    if d is None: return
    method_keys = ['v1_per_region_pak', 'v2_multi_sigma_pak', 'v3_self_consistency_pak',
                   'ref_pak_div5_T15']
    families = {'v1_per_region_pak': 'iterative_refinement', 'v2_multi_sigma_pak': 'iterative_refinement',
                'v3_self_consistency_pak': 'iterative_refinement', 'ref_pak_div5_T15': 'reference'}
    for mk in method_keys:
        entry = MethodEntry(name=f'P13_{mk}', family=families[mk], source='Q3v3',
                            semi_supervised=True)
        for alias, r in d.items():
            entry.per_dataset_baseline[alias] = r.get('baseline_pak', 0)
            entry.per_dataset_pak[alias] = r.get(mk, r.get('baseline_pak', 0))
        matrix.add_method(entry)


def parse_q3v3_P14(path, matrix: MetaResultMatrix):
    """P14_boundary_refinement.json"""
    d = _safe_load_json(path)
    if d is None: return
    method_keys = ['ref_pak', 'v1_gradient_pak', 'v2_local_thr_pak',
                   'v3_dilate3_pak', 'v3_dilate5_pak']
    for mk in method_keys:
        family = 'boundary_refinement' if 'v' in mk else 'reference'
        entry = MethodEntry(name=f'P14_{mk}', family=family, source='Q3v3')
        for alias, r in d.items():
            entry.per_dataset_baseline[alias] = r.get('baseline_pak', 0)
            entry.per_dataset_pak[alias] = r.get(mk, r.get('baseline_pak', 0))
        matrix.add_method(entry)


def parse_q3v4_P16(path, matrix: MetaResultMatrix):
    """P16_evt_tail.json"""
    d = _safe_load_json(path)
    if d is None: return
    sample = next(iter(d.values()))
    variant_keys = list(sample['variants'].keys())
    for vk in variant_keys:
        for sub_v in ['pot_alone', 'hybrid', 'pot_nlm']:
            family = 'EVT_POT'
            entry = MethodEntry(name=f'P16_{vk}_{sub_v}', family=family, source='Q3v4')
            for alias, r in d.items():
                entry.per_dataset_baseline[alias] = r.get('baseline_pak', 0)
                entry.per_dataset_pak[alias] = r['variants'][vk].get(sub_v,
                                                                       r.get('baseline_pak', 0))
            matrix.add_method(entry)


def parse_q3v4_P17(path, matrix: MetaResultMatrix):
    """P17_gmm.json"""
    d = _safe_load_json(path)
    if d is None: return
    sample = next(iter(d.values()))
    variant_keys = list(sample['variants'].keys())
    for vk in variant_keys:
        family = 'GMM_distribution'
        entry = MethodEntry(name=f'P17_{vk}', family=family, source='Q3v4')
        for alias, r in d.items():
            entry.per_dataset_baseline[alias] = r.get('baseline_pak', 0)
            entry.per_dataset_pak[alias] = r['variants'].get(vk, r.get('baseline_pak', 0))
        matrix.add_method(entry)


def parse_q3v4_P18(path, matrix: MetaResultMatrix):
    """P18_ar_conformal.json"""
    d = _safe_load_json(path)
    if d is None: return
    sample = next(iter(d.values()))
    variant_keys = list(sample['variants'].keys())
    family_map = {
        'ar': 'AR_residual', 'conf': 'Conformal',
        'hmm': 'HMM_state', 'pers': 'state_persistence',
        'spectral': 'spectral_subtract', 'super': 'super_ensemble',
    }
    for vk in variant_keys:
        family = 'misc'
        for prefix, fam in family_map.items():
            if vk.startswith(prefix) or prefix in vk:
                family = fam
                break
        entry = MethodEntry(name=f'P18_{vk}', family=family, source='Q3v4')
        for alias, r in d.items():
            entry.per_dataset_baseline[alias] = r.get('baseline_pak', 0)
            entry.per_dataset_pak[alias] = r['variants'].get(vk, r.get('baseline_pak', 0))
        matrix.add_method(entry)


def load_all_results():
    """All Q3 v1-v4 results를 단일 MetaResultMatrix로 통합."""
    matrix = MetaResultMatrix()
    results_dir = Path('/home/ykio/notebooks/claude/mae_anomaly/scripts/q3_exploration/results')

    parsers = [
        # Q3 v1
        ('phaseA_unsupervised_sigma.json', parse_q3v1_phaseA),
        ('phaseB_hybrid.json', parse_q3v1_phaseB),
        ('F2_cross_channel.json', parse_q3v1_F2),
        ('F5_dataset_clustering.json', parse_q3v1_F5),
        # Q3 v2
        ('P2_fine_sigma_sweep.json', parse_q3v2_P2),
        ('P4_threshold_optimization.json', parse_q3v2_P4),
        ('P6_multi_stride.json', parse_q3v2_P6),
        # Q3 v3
        ('P9_unsupervised_seg.json', parse_q3v3_P9),
        ('P12_anomaly_type.json', parse_q3v3_P12),
        ('P13_iterative_refinement.json', parse_q3v3_P13),
        ('P14_boundary_refinement.json', parse_q3v3_P14),
        # Q3 v4
        ('P16_evt_tail.json', parse_q3v4_P16),
        ('P17_gmm.json', parse_q3v4_P17),
        ('P18_ar_conformal.json', parse_q3v4_P18),
    ]

    for fname, parser in parsers:
        path = results_dir / fname
        if path.exists():
            try:
                parser(path, matrix)
            except Exception as e:
                print(f"Error parsing {fname}: {e}")

    return matrix
