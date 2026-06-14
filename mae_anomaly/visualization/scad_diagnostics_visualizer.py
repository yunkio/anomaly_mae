"""SCAD-C (Form C) one-sided repulsion diagnostics.

This module produces a *focused, deeply-interpretable* set of figures answering a
single question:

    "Is SCAD-C's one-sided thresholded repulsion actually working - and, if so,
     does it translate into better anomaly detection?"

SCAD-C objective (mae_anomaly/loss.py, form == 'C'):

    L = mean over (anchor a, negative u) of  relu( cos(z_a, sg[z_u]) - gamma )^2

i.e. push every anomaly-anchor patch embedding z_a so its cosine similarity with
the (detached, one-sided) normal/background embedding z_u drops to <= gamma
(default gamma = 0 => decorrelate / make orthogonal). Only the anomaly anchors
receive gradient (U is stop-gradient when one_sided=True).

Six Form-C diagnostic series are logged every epoch (trainer.py), all read-only:

  train_scad_c_mean_sim         mean cos over ALL anchor x U pairs   (primary progress signal)
  train_scad_c_active_pair_frac frac of pairs still violating cos > gamma (loss "area")
  train_scad_c_active_sim_mean  conditional mean cos over violating pairs (residual severity)
  train_scad_c_gamma            gamma threshold echo (constant; reference line)
  train_scad_c_n_anchor         # anomaly anchors / batch (signal availability)
  train_scad_c_n_u              # U negatives / batch (negative pool size)

We deliberately cross-check these against the *general* SCAD geometry series
(train_scad_z_separation / _z_anom_var / _z_norm_var) and the optimization series
(train_scad_grad_norm / _main_grad_norm / _effective_weight) because angular
repulsion can be "faked" by representation collapse - the single most important
failure mode this module is designed to expose.

IMPORTANT - guarded no-op for non-Form-C runs:
  For Form A/B (or GRL / plain) runs the C-series are constant 0.0, so
  generate_all() writes NOTHING and returns []. The visualizer only emits files
  when real Form-C data is present (config.scad_form == 'C', or, if no config is
  supplied, when active_pair_frac/mean_sim carry non-zero signal). This keeps the
  existing pipeline byte-for-byte unchanged for every other experiment.

On-figure text is English (the matplotlib environment has no CJK font); the
machine-readable scad_c_diagnostics_summary.json carries the same verdict.

Data sources (all read-only, no model required):
  - history dict (training_histories.json[idx]) : per-epoch train stats - passed in
  - epoch_metrics.json (loaded if exp_dir given) : per-eval detection metrics

Output: <output_dir>/*.png  +  scad_c_diagnostics_summary.json
"""

import os
import json
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


# Consistent palette (hex, matches the rest of the visualization package style)
_C = {
    'sim':      '#1565C0',  # mean similarity (blue)
    'active':   '#D32F2F',  # active-pair severity (red)
    'frac':     '#EF6C00',  # active fraction (orange)
    'loss':     '#6A1B9A',  # scad loss (purple)
    'sep':      '#2E7D32',  # cluster separation (green)
    'anom_var': '#C62828',  # anomaly variance (red)
    'norm_var': '#1565C0',  # normal variance (blue)
    'grad_s':   '#AD1457',  # scad grad (magenta)
    'grad_m':   '#00838F',  # main grad (teal)
    'weight':   '#7B1FA2',  # effective weight
    'det':      '#2E7D32',  # detection metric (green)
    'gamma':    '#555555',  # gamma reference line (grey)
    'warmup':   '#9E9E9E',  # warmup boundary
    'best':     '#000000',  # best-epoch marker
}


class ScadDiagnosticsVisualizer:
    """Produce SCAD-C repulsion diagnostics from a training history dict.

    Parameters
    ----------
    history : dict
        One experiment's training history (training_histories.json[idx]); must
        contain the ``train_scad_c_*`` series for output to be produced.
    output_dir : str
        Directory to write PNGs + summary JSON into (created on demand, only when
        Form-C data is present).
    exp_dir : str, optional
        Dataset-level experiment dir; if given, epoch_metrics.json is loaded from
        here to build the detection-coupling figure.
    config : object, optional
        Config-like object; ``scad_form`` / ``scad_gamma`` /
        ``teacher_only_warmup_epochs`` / ``num_epochs`` are read if present.
    """

    def __init__(self, history: Dict, output_dir: str,
                 exp_dir: Optional[str] = None, config=None):
        self.history = history or {}
        self.output_dir = output_dir
        self.exp_dir = exp_dir
        self.config = config

        # --- core Form-C series ---
        self.ms = self._arr('train_scad_c_mean_sim')
        self.apf = self._arr('train_scad_c_active_pair_frac')
        self.asm = self._arr('train_scad_c_active_sim_mean')
        self.n_anchor = self._arr('train_scad_c_n_anchor')
        self.n_u = self._arr('train_scad_c_n_u')

        # --- general SCAD geometry / optimization series (cross-checks) ---
        self.scad_loss = self._arr('train_scad_loss')
        self.z_sep = self._arr('train_scad_z_separation')
        self.z_anom_var = self._arr('train_scad_z_anom_var')
        self.z_norm_var = self._arr('train_scad_z_norm_var')
        self.grad_scad = self._arr('train_scad_grad_norm')
        self.grad_main = self._arr('train_scad_main_grad_norm')
        self.eff_w = self._arr('train_scad_effective_weight')
        self.adapt_lam = self._arr('train_scad_adaptive_lambda')
        self.ramp = self._arr('train_scad_ramp')

        n = self.ms.size
        ep = self._arr('epoch')
        self.epochs = ep if ep.size == n and n > 0 else np.arange(1, n + 1)

        self.gamma = self._resolve_gamma()
        self.warmup = self._resolve_warmup(n)
        # post-warmup epochs where the C objective is actually active
        self.active = (self.epochs > self.warmup) if n else np.zeros(0, bool)

    # ------------------------------------------------------------------ utils
    def _arr(self, key: str) -> np.ndarray:
        v = self.history.get(key, [])
        try:
            return np.asarray(v, dtype=float)
        except (TypeError, ValueError):
            return np.asarray([], dtype=float)

    def _resolve_gamma(self) -> float:
        g = self._arr('train_scad_c_gamma')
        if g.size and np.isfinite(g[-1]):
            return float(g[-1])
        if self.config is not None and getattr(self.config, 'scad_gamma', None) is not None:
            return float(self.config.scad_gamma)
        return 0.0

    def _resolve_warmup(self, n: int) -> float:
        w = getattr(self.config, 'teacher_only_warmup_epochs', None) if self.config else None
        if w is not None and w > 0:
            return float(w)
        ne = getattr(self.config, 'num_epochs', None) if self.config else None
        if ne and ne > 0:
            return float(ne) / 2.0
        return float(n) / 2.0 if n else 0.0

    def has_scad_c(self) -> bool:
        """True only when genuine Form-C data is present.

        For Form A/B the C-series are constant 0.0 (the C branch never runs), so
        this returns False and the whole module is a no-op.  ``scad_c_n_anchor``
        is NOT used as the signal because loss.py populates it for every form.
        """
        if self.ms.size == 0:
            return False
        if self.config is not None:
            return str(getattr(self.config, 'scad_form', '')) == 'C'
        # No config: infer from data - active_pair_frac/mean_sim carry C signal
        return bool(np.any(self.apf != 0) or np.any(self.ms != 0))

    def _post(self, a: np.ndarray) -> np.ndarray:
        """Slice an array to the post-warmup (active) region."""
        if a.size == self.active.size and self.active.size:
            return a[self.active]
        return a

    # ----------------------------------------------------------- annotations
    def _mark_phases(self, ax, best_ep=None, gamma_line=False):
        if self.warmup and self.epochs.size:
            ax.axvline(self.warmup, color=_C['warmup'], ls='--', lw=1.2, alpha=0.7,
                       label=f'warmup end (ep{int(self.warmup)})')
        if gamma_line:
            ax.axhline(self.gamma, color=_C['gamma'], ls=':', lw=1.4, alpha=0.9,
                       label=f'gamma = {self.gamma:g}')
        if best_ep is not None:
            ax.axvline(best_ep, color=_C['best'], ls='-', lw=1.0, alpha=0.5,
                       label=f'best eval (ep{int(best_ep)})')

    # -------------------------------------------------------- detection coupling
    def _load_epoch_metrics(self):
        """Return (eval_epochs, {metric: values}) from epoch_metrics.json or (None, None)."""
        if not self.exp_dir:
            return None, None
        p = os.path.join(self.exp_dir, 'epoch_metrics.json')
        if not os.path.exists(p):
            return None, None
        try:
            with open(p) as f:
                d = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None, None
        rows = d.get('epochs') if isinstance(d, dict) else (d if isinstance(d, list) else None)
        if not rows:
            return None, None
        ev = np.array([r.get('epoch', i) for i, r in enumerate(rows)], dtype=float)
        out = {}
        for k in ('pak_auc_f1', 'prc_auc', 'f1_t', 'affiliation_f1'):
            vals = np.array([r.get(k, np.nan) for r in rows], dtype=float)
            if np.any(np.isfinite(vals)):
                out[k] = vals
        return ev, out

    def _ms_at(self, eval_epochs: np.ndarray) -> np.ndarray:
        """mean_sim sampled at the eval epochs (nearest training epoch)."""
        if self.epochs.size == 0:
            return np.full(eval_epochs.shape, np.nan)
        idx = np.searchsorted(self.epochs, eval_epochs)
        idx = np.clip(idx, 0, self.epochs.size - 1)
        return self.ms[idx]

    # ====================================================================== API
    def generate_all(self) -> List[str]:
        if not self.has_scad_c():
            print("  - [scad_diagnostics] skipped (no Form-C data: scad_form != 'C')")
            return []
        os.makedirs(self.output_dir, exist_ok=True)
        written = []
        for fn in (self._fig_repulsion_progress,
                   self._fig_collapse_guard,
                   self._fig_optimization_signal,
                   self._fig_detection_coupling,
                   self._fig_summary):
            try:
                path = fn()
                if path:
                    written.append(path)
                    print(f"  - {os.path.relpath(path, self.output_dir)}")
            except Exception as e:  # never let a viz failure break the run
                print(f"  - [scad_diagnostics] {fn.__name__} failed: {e}")
            finally:
                plt.close('all')
        summary_path = self._write_summary()
        if summary_path:
            written.append(summary_path)
        return written

    # ----------------------------------------------------------- verdict logic
    def compute_verdict(self) -> Dict:
        """Encode the deep interpretation of SCAD-C success/failure into numbers."""
        post_ms = self._post(self.ms)
        post_apf = self._post(self.apf)
        post_av = self._post(self.z_anom_var)
        post_sep = self._post(self.z_sep)

        def first_last(a):
            a = a[np.isfinite(a)]
            if a.size == 0:
                return (np.nan, np.nan)
            return (float(a[0]), float(a[-1]))

        ms0, msf = first_last(post_ms)
        msmin = float(np.nanmin(post_ms)) if post_ms.size else np.nan
        apf0, apff = first_last(post_apf)
        av0, avf = first_last(post_av)
        sep0, sepf = first_last(post_sep)

        # (1) repulsion: did mean_sim fall and reach/cross gamma?
        repulsion_drop = (ms0 - msf) if np.isfinite(ms0) and np.isfinite(msf) else np.nan
        crossed = bool(np.isfinite(msf) and msf <= self.gamma + 1e-4)
        # (2) saturation: active fraction emptied
        saturated = bool(np.isfinite(apff) and apff < 0.05)
        # (3) collapse: anomaly cluster variance shrank toward zero (BAD - fake separation)
        collapse_ratio = (avf / av0) if (np.isfinite(av0) and av0 > 1e-12) else np.nan
        collapse = bool(np.isfinite(avf) and (avf < 1e-4 or (np.isfinite(collapse_ratio) and collapse_ratio < 0.2)))
        # (4) separation trend (should hold or grow if repulsion is genuine)
        sep_grew = bool(np.isfinite(sep0) and np.isfinite(sepf) and sepf >= sep0)
        # (5) optimization dominance: does SCAD overpower reconstruction grads?
        m = self.active if self.active.size == self.grad_scad.size else np.ones(self.grad_scad.size, bool)
        gs = self.grad_scad[m] if self.grad_scad.size else np.array([])
        gm = self.grad_main[m] if self.grad_main.size else np.array([])
        valid = (gm > 1e-12) & np.isfinite(gs) & np.isfinite(gm) if gs.size else np.array([], bool)
        grad_dom = float(np.median(gs[valid] / gm[valid])) if np.any(valid) else np.nan
        # (6) does repulsion buy detection? corr(mean_sim, pak_f1) - expect NEGATIVE
        det_corr, det_metric = np.nan, None
        ev, mets = self._load_epoch_metrics()
        if ev is not None and mets:
            for k in ('pak_auc_f1', 'prc_auc'):
                if k in mets:
                    pak = mets[k]
                    msa = self._ms_at(ev)
                    keep = (ev > self.warmup) & np.isfinite(pak) & np.isfinite(msa)
                    if np.count_nonzero(keep) >= 3 and np.std(msa[keep]) > 1e-9 and np.std(pak[keep]) > 1e-9:
                        det_corr = float(np.corrcoef(msa[keep], pak[keep])[0, 1])
                        det_metric = k
                        break

        # ---- assemble a human verdict string (English; mirrored in summary JSON) ----
        parts = []
        if np.isfinite(repulsion_drop):
            if repulsion_drop > 0.02 or crossed:
                parts.append(f"repulsion active (mean_sim {ms0:.3f}->{msf:.3f}, "
                             f"{'gamma crossed' if crossed else 'gamma not reached'})")
            else:
                parts.append(f"repulsion weak (mean_sim {ms0:.3f}->{msf:.3f})")
        if saturated:
            parts.append(f"loss saturated (active_frac {apff:.2f}<0.05, gradient exhausted)")
        if collapse:
            parts.append("WARNING collapse suspected (anom_var->0, separation is fake)")
        elif np.isfinite(avf):
            parts.append(f"no collapse (anom_var {av0:.3g}->{avf:.3g}, sep {'up' if sep_grew else 'down'})")
        if np.isfinite(grad_dom):
            parts.append(f"grad SCAD/main={grad_dom:.2f}" + (" (SCAD dominates)" if grad_dom > 1.0 else ""))
        if np.isfinite(det_corr):
            sign = "negative (lower sim -> higher detection: helps)" if det_corr < -0.1 else \
                   ("positive (repulsion opposes detection)" if det_corr > 0.1 else "uncorrelated")
            parts.append(f"detection corr(mean_sim,{det_metric})={det_corr:.2f} {sign}")

        # overall flag
        success = bool((crossed or (np.isfinite(repulsion_drop) and repulsion_drop > 0.02))
                       and not collapse)
        return {
            'scad_form': str(getattr(self.config, 'scad_form', 'C')) if self.config else 'C',
            'gamma': self.gamma,
            'warmup_epoch': self.warmup,
            'mean_sim_start': ms0, 'mean_sim_final': msf, 'mean_sim_min': msmin,
            'repulsion_drop': repulsion_drop, 'crossed_gamma': crossed,
            'active_frac_start': apf0, 'active_frac_final': apff, 'saturated': saturated,
            'anom_var_start': av0, 'anom_var_final': avf, 'collapse_ratio': collapse_ratio,
            'collapse_suspected': collapse,
            'separation_start': sep0, 'separation_final': sepf, 'separation_grew': sep_grew,
            'grad_dominance_scad_over_main': grad_dom,
            'detection_corr': det_corr, 'detection_metric': det_metric,
            'repulsion_success': success,
            'verdict': "; ".join(parts) if parts else "insufficient data",
        }

    def _write_summary(self) -> Optional[str]:
        try:
            v = self.compute_verdict()
            p = os.path.join(self.output_dir, 'scad_c_diagnostics_summary.json')
            with open(p, 'w') as f:
                json.dump(v, f, indent=2,
                          default=lambda o: None if (isinstance(o, float) and not np.isfinite(o)) else float(o))
            print("  - scad_c_diagnostics_summary.json")
            return p
        except Exception as e:
            print(f"  - [scad_diagnostics] summary failed: {e}")
            return None

    # =============================================================== figures
    def _best_eval_epoch(self) -> Optional[float]:
        ev, mets = self._load_epoch_metrics()
        if ev is None or not mets:
            return None
        key = 'pak_auc_f1' if 'pak_auc_f1' in mets else next(iter(mets))
        vals = mets[key]
        keep = np.isfinite(vals)
        if not np.any(keep):
            return None
        return float(ev[keep][int(np.nanargmax(vals[keep]))])

    def _fig_repulsion_progress(self) -> Optional[str]:
        """Is the C objective being optimized? mean_sim / active_frac / severity / loss."""
        n = self.ms.size
        if n == 0:
            return None
        ep = self.epochs
        best = self._best_eval_epoch()
        fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))

        # (A) mean similarity + active-pair severity vs gamma
        ax = axes[0]
        ax.plot(ep, self.ms, color=_C['sim'], lw=2.2, label='mean cos(z_a, z_u)  (all pairs)')
        ax.plot(ep, self.asm, color=_C['active'], lw=1.6, ls='--', alpha=0.85,
                label='mean cos | active  (violating pairs)')
        ax.axhspan(self.gamma, max(1.0, np.nanmax(self.ms) if n else 1.0),
                   color=_C['active'], alpha=0.05)
        self._mark_phases(ax, best_ep=best, gamma_line=True)
        ax.set_xlabel('Epoch'); ax.set_ylabel('cosine similarity')
        ax.set_title('(A) Anomaly-vs-background alignment down (target: below gamma)', fontweight='bold')
        ax.legend(fontsize=7, loc='best'); ax.grid(True, alpha=0.3)

        # (B) active-pair fraction (loss "area")
        ax = axes[1]
        ax.plot(ep, self.apf, color=_C['frac'], lw=2.2, label='active pair fraction (cos>gamma)')
        ax.fill_between(ep, 0, self.apf, color=_C['frac'], alpha=0.12)
        self._mark_phases(ax, best_ep=best)
        ax.axhline(0.05, color='#888', ls=':', lw=1, alpha=0.7, label='saturation (0.05)')
        ax.set_xlabel('Epoch'); ax.set_ylabel('fraction of pairs')
        ax.set_ylim(-0.02, 1.02)
        ax.set_title('(B) Violating-pair fraction down (loss surface emptying)', fontweight='bold')
        ax.legend(fontsize=7, loc='best'); ax.grid(True, alpha=0.3)

        # (C) scad loss (log)
        ax = axes[2]
        sl = self.scad_loss if self.scad_loss.size == n else np.full(n, np.nan)
        pos = sl > 0
        if np.any(pos):
            ax.semilogy(ep[pos], sl[pos], color=_C['loss'], lw=2.0, label='SCAD loss')
        else:
            ax.plot(ep, sl, color=_C['loss'], lw=2.0, label='SCAD loss')
        self._mark_phases(ax, best_ep=best)
        ax.set_xlabel('Epoch'); ax.set_ylabel('SCAD loss (log)')
        ax.set_title('(C) SCAD loss convergence', fontweight='bold')
        ax.legend(fontsize=7, loc='best'); ax.grid(True, alpha=0.3, which='both')

        fig.suptitle('SCAD-C - Repulsion Progress  (one-sided thresholded decorrelation)',
                     fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        path = os.path.join(self.output_dir, 'scad_c_repulsion_progress.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        return path

    def _fig_collapse_guard(self) -> Optional[str]:
        """THE key failure-mode check: genuine separation vs representation collapse."""
        n = self.ms.size
        if n == 0:
            return None
        ep = self.epochs
        best = self._best_eval_epoch()
        fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))

        # (A) cluster separation (L2)
        ax = axes[0]
        if self.z_sep.size == n:
            ax.plot(ep, self.z_sep, color=_C['sep'], lw=2.2, label='||z_anom - z_norm|| (mean)')
        self._mark_phases(ax, best_ep=best)
        ax.set_xlabel('Epoch'); ax.set_ylabel('L2 separation')
        ax.set_title('(A) Cluster separation up (higher is better)', fontweight='bold')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        # (B) intra-cluster variance - collapse alarm
        ax = axes[1]
        if self.z_anom_var.size == n:
            av = np.where(self.z_anom_var > 0, self.z_anom_var, np.nan)
            ax.semilogy(ep, av, color=_C['anom_var'], lw=2.0, ls='-', label='anomaly var')
        if self.z_norm_var.size == n:
            nv = np.where(self.z_norm_var > 0, self.z_norm_var, np.nan)
            ax.semilogy(ep, nv, color=_C['norm_var'], lw=2.0, ls='--', label='normal var')
        self._mark_phases(ax, best_ep=best)
        ax.set_xlabel('Epoch'); ax.set_ylabel('intra-cluster variance (log)')
        ax.set_title('(B) Cluster variance (-> 0 means collapse alarm)', fontweight='bold')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3, which='both')

        # (C) phase plot: separation(down sim) vs collapse(down var), trajectory colored by epoch
        ax = axes[2]
        post = self.active if self.active.size == n else np.ones(n, bool)
        x = self.ms[post]
        y = self.z_anom_var[post] if self.z_anom_var.size == n else np.full(x.shape, np.nan)
        ce = ep[post]
        good = np.isfinite(x) & np.isfinite(y)
        if np.count_nonzero(good) >= 2:
            x, y, ce = x[good], y[good], ce[good]
            pts = np.array([x, y]).T.reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            lc = LineCollection(segs, cmap='viridis', linewidth=2.0)
            lc.set_array(ce[:-1])
            ax.add_collection(lc)
            ax.scatter(x[0], y[0], c='#2E7D32', s=60, marker='o', zorder=5, label='start (post-warmup)')
            ax.scatter(x[-1], y[-1], c='#C62828', s=70, marker='*', zorder=5, label='final')
            ax.axvline(self.gamma, color=_C['gamma'], ls=':', lw=1.2, alpha=0.8)
            ax.set_xlim(min(x.min(), self.gamma) - 0.02, x.max() + 0.02)
            ax.set_ylim(bottom=max(0, np.nanmin(y) * 0.5))
            cb = fig.colorbar(lc, ax=ax, pad=0.01); cb.set_label('epoch', fontsize=8)
            ax.annotate('genuine separation\n(down sim, var held)', xy=(0.03, 0.92),
                        xycoords='axes fraction', fontsize=7, color='#2E7D32', va='top')
            ax.annotate('collapse\n(down sim, down var)', xy=(0.03, 0.12),
                        xycoords='axes fraction', fontsize=7, color='#C62828', va='top')
        ax.set_xlabel('mean cos similarity  (<- target direction)'); ax.set_ylabel('anomaly cluster var')
        ax.set_title('(C) Separation vs collapse phase trajectory', fontweight='bold')
        ax.legend(fontsize=7, loc='upper right'); ax.grid(True, alpha=0.3)

        fig.suptitle('SCAD-C - Geometry Health  (is the separation real, or collapse?)',
                     fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        path = os.path.join(self.output_dir, 'scad_c_collapse_guard.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        return path

    def _fig_optimization_signal(self) -> Optional[str]:
        """Is SCAD fighting reconstruction? And is the anchor signal reliable?"""
        n = self.ms.size
        if n == 0:
            return None
        ep = self.epochs
        best = self._best_eval_epoch()
        fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))

        # (A) gradient balance: scad vs main
        ax = axes[0]
        has_grad = self.grad_scad.size == n and self.grad_main.size == n and np.any(self.grad_scad > 0)
        if has_grad:
            ax.plot(ep, self.grad_scad, color=_C['grad_s'], lw=1.8, label='SCAD grad norm')
            ax.plot(ep, self.grad_main, color=_C['grad_m'], lw=1.8, ls='--', label='main grad norm')
            ax.set_yscale('log')
            ax.set_ylabel('grad norm (log)')
            ax2 = ax.twinx()
            ratio = np.where(self.grad_main > 1e-12, self.grad_scad / (self.grad_main + 1e-12), np.nan)
            ax2.plot(ep, ratio, color='#455A64', lw=1.0, alpha=0.5)
            ax2.axhline(1.0, color='#455A64', ls=':', lw=1, alpha=0.6)
            ax2.set_ylabel('SCAD / main ratio', fontsize=8, color='#455A64')
        else:
            ax.text(0.5, 0.5, 'grad norms not logged', ha='center', va='center',
                    transform=ax.transAxes, fontsize=9, color='gray')
        self._mark_phases(ax, best_ep=best)
        ax.set_xlabel('Epoch')
        ax.set_title('(A) Gradient balance (SCAD vs reconstruction)', fontweight='bold')
        ax.legend(fontsize=7, loc='best'); ax.grid(True, alpha=0.3, which='both')

        # (B) effective weight schedule
        ax = axes[1]
        if self.eff_w.size == n:
            ax.plot(ep, self.eff_w, color=_C['weight'], lw=2.0, label='effective weight')
        if self.adapt_lam.size == n:
            ax.plot(ep, self.adapt_lam, color='#00695C', lw=1.4, ls=':', alpha=0.8, label='adaptive lambda')
        if self.ramp.size == n:
            ax.plot(ep, self.ramp, color='#F57C00', lw=1.4, ls='-.', alpha=0.8, label='ramp')
        self._mark_phases(ax, best_ep=best)
        ax.set_xlabel('Epoch'); ax.set_ylabel('weight')
        ax.set_title('(B) SCAD effective weight schedule', fontweight='bold')
        ax.legend(fontsize=7, loc='best'); ax.grid(True, alpha=0.3)

        # (C) signal availability: n_anchor / n_u
        ax = axes[2]
        if self.n_anchor.size == n:
            ax.plot(ep, self.n_anchor, color='#C62828', lw=1.8, label='# anomaly anchors / batch')
        if self.n_u.size == n:
            ax.plot(ep, self.n_u, color='#1565C0', lw=1.8, ls='--', label='# U negatives / batch')
        if self.n_anchor.size == n:
            thr = max(2.0, np.nanmedian(self.n_anchor[self.active]) * 0.25) if np.any(self.active) else 2.0
            low = self.n_anchor < thr
            if np.any(low):
                ax.fill_between(ep, 0, self.n_anchor, where=low, color='#C62828', alpha=0.08,
                                label=f'low-anchor (<{thr:.0f}, high variance)')
        self._mark_phases(ax, best_ep=best)
        ax.set_xlabel('Epoch'); ax.set_ylabel('count / batch')
        ax.set_title('(C) Anchor / negative sample counts (reliability)', fontweight='bold')
        ax.legend(fontsize=7, loc='best'); ax.grid(True, alpha=0.3)

        fig.suptitle('SCAD-C - Optimization Dynamics & Signal Reliability',
                     fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        path = os.path.join(self.output_dir, 'scad_c_optimization_signal.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        return path

    def _fig_detection_coupling(self) -> Optional[str]:
        """The payoff: does down(mean_sim) actually buy up(detection)?"""
        ev, mets = self._load_epoch_metrics()
        if ev is None or not mets:
            print("  - scad_c_detection_coupling.png skipped (no epoch_metrics.json)")
            return None
        det_key = 'pak_auc_f1' if 'pak_auc_f1' in mets else ('prc_auc' if 'prc_auc' in mets else next(iter(mets)))
        det = mets[det_key]
        msa = self._ms_at(ev)
        keep = (ev > self.warmup) & np.isfinite(det) & np.isfinite(msa)
        if np.count_nonzero(keep) < 3:
            print("  - scad_c_detection_coupling.png skipped (insufficient post-warmup evals)")
            return None
        evk, detk, msk = ev[keep], det[keep], msa[keep]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))

        # (A) twin-axis over epochs
        ax = axes[0]
        l1, = ax.plot(evk, msk, color=_C['sim'], lw=2.2, marker='o', ms=3, label='mean cos sim (target down)')
        ax.axhline(self.gamma, color=_C['gamma'], ls=':', lw=1.2, alpha=0.8)
        ax.set_xlabel('Epoch'); ax.set_ylabel('mean cos similarity', color=_C['sim'])
        ax.tick_params(axis='y', labelcolor=_C['sim'])
        ax2 = ax.twinx()
        l2, = ax2.plot(evk, detk, color=_C['det'], lw=2.2, marker='s', ms=3, label=f'{det_key} (target up)')
        ax2.set_ylabel(det_key, color=_C['det'])
        ax2.tick_params(axis='y', labelcolor=_C['det'])
        if self.warmup:
            ax.axvline(self.warmup, color=_C['warmup'], ls='--', lw=1.2, alpha=0.6)
        ax.set_title('(A) Repulsion vs detection (post-warmup)', fontweight='bold')
        ax.legend(handles=[l1, l2], fontsize=8, loc='best'); ax.grid(True, alpha=0.3)

        # (B) scatter mean_sim vs detection, colored by epoch + correlation
        ax = axes[1]
        sc = ax.scatter(msk, detk, c=evk, cmap='viridis', s=45, edgecolor='k', linewidth=0.3)
        cb = fig.colorbar(sc, ax=ax, pad=0.01); cb.set_label('epoch', fontsize=8)
        corr = np.nan
        if np.std(msk) > 1e-9 and np.std(detk) > 1e-9:
            corr = float(np.corrcoef(msk, detk)[0, 1])
            b, a = np.polyfit(msk, detk, 1)
            xs = np.linspace(msk.min(), msk.max(), 50)
            ax.plot(xs, b * xs + a, color='#C62828', lw=1.4, ls='--', alpha=0.8)
        ax.axvline(self.gamma, color=_C['gamma'], ls=':', lw=1.2, alpha=0.8, label=f'gamma={self.gamma:g}')
        interp = ('negative corr -> lower anomaly sim drives higher detection (SCAD-C effective)'
                  if (np.isfinite(corr) and corr < -0.1)
                  else ('positive corr -> repulsion opposes detection'
                        if (np.isfinite(corr) and corr > 0.1) else 'no clear correlation'))
        ax.set_xlabel('mean cos similarity'); ax.set_ylabel(det_key)
        ax.set_title((f'(B) corr = {corr:.2f}\n{interp}') if np.isfinite(corr) else '(B) correlation scatter',
                     fontweight='bold')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        fig.suptitle('SCAD-C - Detection Coupling  (does repulsion help real detection?)',
                     fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        path = os.path.join(self.output_dir, 'scad_c_detection_coupling.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        return path

    def _fig_summary(self) -> Optional[str]:
        """One-page executive dashboard with auto-interpretation text."""
        n = self.ms.size
        if n == 0:
            return None
        v = self.compute_verdict()
        ep = self.epochs
        fig = plt.figure(figsize=(16, 9))
        gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.28)

        ax = fig.add_subplot(gs[0, 0])
        ax.plot(ep, self.ms, color=_C['sim'], lw=2)
        ax.axhline(self.gamma, color=_C['gamma'], ls=':', lw=1.2)
        if self.warmup:
            ax.axvline(self.warmup, color=_C['warmup'], ls='--', lw=1, alpha=0.6)
        ax.set_title('mean cos sim (down)', fontsize=10, fontweight='bold'); ax.grid(True, alpha=0.3)
        ax.set_xlabel('epoch', fontsize=8)

        ax = fig.add_subplot(gs[0, 1])
        ax.plot(ep, self.apf, color=_C['frac'], lw=2)
        if self.warmup:
            ax.axvline(self.warmup, color=_C['warmup'], ls='--', lw=1, alpha=0.6)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title('active pair frac (down)', fontsize=10, fontweight='bold'); ax.grid(True, alpha=0.3)
        ax.set_xlabel('epoch', fontsize=8)

        ax = fig.add_subplot(gs[0, 2])
        if self.z_anom_var.size == n:
            ax.semilogy(ep, np.where(self.z_anom_var > 0, self.z_anom_var, np.nan),
                        color=_C['anom_var'], lw=2, label='anom var')
        if self.z_norm_var.size == n:
            ax.semilogy(ep, np.where(self.z_norm_var > 0, self.z_norm_var, np.nan),
                        color=_C['norm_var'], lw=2, ls='--', label='norm var')
        if self.warmup:
            ax.axvline(self.warmup, color=_C['warmup'], ls='--', lw=1, alpha=0.6)
        ax.set_title('cluster var (collapse guard)', fontsize=10, fontweight='bold')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3, which='both'); ax.set_xlabel('epoch', fontsize=8)

        # detection coupling mini
        ax = fig.add_subplot(gs[1, 0])
        ev, mets = self._load_epoch_metrics()
        if ev is not None and mets:
            det_key = 'pak_auc_f1' if 'pak_auc_f1' in mets else next(iter(mets))
            msa = self._ms_at(ev)
            keep = (ev > self.warmup) & np.isfinite(mets[det_key]) & np.isfinite(msa)
            if np.count_nonzero(keep) >= 2:
                ax.scatter(msa[keep], mets[det_key][keep], c=ev[keep], cmap='viridis',
                           s=30, edgecolor='k', linewidth=0.3)
                ax.set_xlabel('mean cos sim', fontsize=8); ax.set_ylabel(det_key, fontsize=8)
        ax.set_title('sim vs detection', fontsize=10, fontweight='bold'); ax.grid(True, alpha=0.3)

        # verdict text panel (spans 2 cells)
        ax = fig.add_subplot(gs[1, 1:])
        ax.axis('off')
        flag = 'SUCCESS' if v['repulsion_success'] else 'CHECK'
        col = '#2E7D32' if v['repulsion_success'] else '#C62828'

        def g(x, f='{:.3f}'):
            return f.format(x) if isinstance(x, (int, float)) and np.isfinite(x) else 'n/a'

        lines = [
            f"SCAD-C diagnostics summary   [{flag}]",
            "",
            f"- mean_sim : {g(v['mean_sim_start'])} -> {g(v['mean_sim_final'])}  "
            f"(min {g(v['mean_sim_min'])}, gamma={v['gamma']:g}, "
            f"{'crossed' if v['crossed_gamma'] else 'not reached'})",
            f"- active frac : {g(v['active_frac_start'],'{:.2f}')} -> {g(v['active_frac_final'],'{:.2f}')}  "
            f"({'saturated' if v['saturated'] else 'ongoing'})",
            f"- anom var : {g(v['anom_var_start'],'{:.3g}')} -> {g(v['anom_var_final'],'{:.3g}')}  "
            f"({'COLLAPSE' if v['collapse_suspected'] else 'no collapse'})",
            f"- separation : {g(v['separation_start'],'{:.3g}')} -> {g(v['separation_final'],'{:.3g}')}  "
            f"({'up' if v['separation_grew'] else 'down'})",
            f"- grad SCAD/main : {g(v['grad_dominance_scad_over_main'],'{:.2f}')}",
            f"- detection corr(mean_sim, {v['detection_metric']}) : {g(v['detection_corr'],'{:.2f}')}",
            "",
            "verdict: " + v['verdict'],
        ]
        ax.text(0.0, 0.98, "\n".join(lines), transform=ax.transAxes, va='top', ha='left',
                fontsize=9.5, family='monospace',
                bbox=dict(boxstyle='round', facecolor='#F5F5F5', edgecolor=col, lw=1.6))

        fig.suptitle('SCAD-C Diagnostics - Summary', fontsize=15, fontweight='bold')
        path = os.path.join(self.output_dir, 'scad_c_summary.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        return path
