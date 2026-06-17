"""GRL (Gradient-Reversal adversarial classifier) effect diagnostics.

Mirror of ``scad_diagnostics_visualizer`` but for the GRL path. It answers one
question with a focused, deeply-interpretable figure set:

    "Is the GRL adversarial game actually working - i.e. is the student being
     pushed to anomaly-INVARIANT hidden features - and does that translate into
     better anomaly detection?"

GRL mechanism (mae_anomaly/model.py + loss.py, classifier mode):
    student hidden  --GradientReversal(ramp_lambda)-->  anomaly_classifier --> BCE
The classifier (forward) tries to predict anomaly/normal from the student
hidden; the gradient reversal makes the student learn features that FOOL it
(anomaly-invariant). Invariant student features -> the student can't specialise
on anomalies -> larger teacher/student output discrepancy on anomalies -> score.

WHY GRL needs its OWN diagnostic logic (it is NOT SCAD):
    GRL is an adversarial minimax. "Success" is COUNTER-INTUITIVE - the classifier
    getting WORSE (balanced_acc -> 0.5) is the goal (student achieved invariance).
    But balanced_acc -> 0.5 is ALSO what you see if (a) the classifier never
    trained (loss-weight STARVED), or (b) the classifier collapsed to one class.
    So the central job of this module is to DISAMBIGUATE genuine invariance from
    a dead game (starvation / degeneracy). The #1 observed failure mode here is
    adaptive-lambda STARVATION (grl_effective_weight -> 0).

Per-epoch GRL series read (all logged in trainer.py, read-only, no model needed):
  train_grl_cls_loss          classifier BCE (what the classifier minimises)
  train_grl_balanced_acc      (TPR+TNR)/2 - discrimination (-> 0.5 == invariance OR dead)
  train_grl_anomaly_acc       TPR  (anomaly patches called anomaly)
  train_grl_normal_acc        TNR  (normal patches called normal)
  train_grl_acc_gap           |TNR-TPR| - degeneracy alarm (large == class-collapse)
  train_grl_lambda            adaptive grad-norm ratio ||grad_main||/||grad_adv|| (clamp 0..10)
  train_grl_effective_weight  prev_epoch_lambda * grl_loss_weight (actual loss multiplier)
  train_grl_ramp_lambda       Ganin gradient-reversal strength (2/(1+e^-10p)-1)        [2026-06-17]
  train_grl_pos_logit_mean    mean classifier logit on anomaly patches                  [2026-06-17]
  train_grl_neg_logit_mean    mean classifier logit on normal patches                   [2026-06-17]
  train_grl_logit_margin      mean |logit| - classifier confidence                      [2026-06-17]
  train_grl_main_grad_norm    ||grad of main loss w.r.t. last student weight||           [2026-06-17]
  train_grl_adv_grad_norm     ||grad of grl_cls_loss w.r.t. last student weight|| (ramp baked in) [2026-06-17]

The continuous invariance signal is the convergence of pos/neg logit means
(|pos_logit_mean - neg_logit_mean| shrinking), which is far more sensitive than
the thresholded balanced_acc.

Transfer signal: ``disc_snr`` from epoch_metrics.json - the anomaly-vs-normal
output-discrepancy separation that the anomaly SCORE is built on. GRL "transfers"
iff invariance growth tracks disc_snr growth.

IMPORTANT - guarded no-op for non-GRL runs:
  has_grl() is True only when config.use_grl and grl_mode=='classifier' (or, with
  no config, when GRL series carry non-zero signal). For SCAD/plain/WDGRL runs
  generate_all() writes NOTHING and returns []. The existing pipeline is byte-for-
  byte unchanged for every other experiment.

Relocation: at the end of generate_all() this module MOVES the pre-existing GRL
figures (best_model/GRL_contribution_trend.png, epoch_metrics/epoch_grl.png) into
the grl_diagnostics/ directory so all GRL visuals live in one place.

On-figure text is English (no CJK font); grl_diagnostics_summary.json carries the
machine-readable verdict.

Output: <output_dir>/*.png  +  grl_diagnostics_summary.json
"""

import os
import json
import shutil
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


_C = {
    'bacc':     '#1565C0',  # balanced accuracy (blue)
    'tpr':      '#C62828',  # anomaly acc / TPR (red)
    'tnr':      '#2E7D32',  # normal acc / TNR (green)
    'gap':      '#EF6C00',  # degeneracy gap (orange)
    'loss':     '#6A1B9A',  # classifier BCE (purple)
    'pos':      '#C62828',  # anomaly logit mean (red)
    'neg':      '#1565C0',  # normal logit mean (blue)
    'margin':   '#00838F',  # logit margin (teal)
    'grad_a':   '#AD1457',  # adversarial grad norm (magenta)
    'grad_m':   '#00838F',  # main grad norm (teal)
    'weight':   '#7B1FA2',  # effective weight
    'ramp':     '#455A64',  # ramp lambda (slate)
    'pressure': '#D32F2F',  # actual adversarial pressure (red)
    'det':      '#2E7D32',  # detection metric (green)
    'disc':     '#6A1B9A',  # disc_snr transfer (purple)
    'chance':   '#555555',  # 0.5 chance line (grey)
    'warmup':   '#9E9E9E',  # warmup boundary
    'best':     '#000000',  # best-epoch marker
}

_EPS = 1e-3  # effective-weight starvation threshold (essentially zero)


class GrlDiagnosticsVisualizer:
    """Produce GRL adversarial-game diagnostics from a training history dict.

    Parameters
    ----------
    history : dict
        One experiment's training history (training_histories.json[idx]); must
        contain the ``train_grl_*`` series for output to be produced.
    output_dir : str
        Directory to write PNGs + summary JSON into (created on demand, only when
        GRL data is present). Conventionally <dataset>/visualization/grl_diagnostics.
    exp_dir : str, optional
        Dataset-level experiment dir; if given, epoch_metrics.json is loaded for
        the detection-coupling and transfer figures.
    config : object, optional
        Config-like object; ``use_grl`` / ``grl_mode`` / ``grl_loss_weight`` /
        ``teacher_only_warmup_epochs`` / ``num_epochs`` are read if present.
    """

    def __init__(self, history: Dict, output_dir: str,
                 exp_dir: Optional[str] = None, config=None):
        self.history = history or {}
        self.output_dir = output_dir
        self.exp_dir = exp_dir
        self.config = config

        # --- core GRL classifier series ---
        self.bacc = self._arr('train_grl_balanced_acc')
        self.tpr = self._arr('train_grl_anomaly_acc')
        self.tnr = self._arr('train_grl_normal_acc')
        self.gap = self._arr('train_grl_acc_gap')
        self.cls_loss = self._arr('train_grl_cls_loss')

        # --- optimization / weighting series ---
        self.adapt_lam = self._arr('train_grl_lambda')
        self.eff_w = self._arr('train_grl_effective_weight')
        self.ramp = self._arr('train_grl_ramp_lambda')
        self.grad_main = self._arr('train_grl_main_grad_norm')
        self.grad_adv = self._arr('train_grl_adv_grad_norm')

        # --- continuous invariance (logit-level) series ---
        self.pos_logit = self._arr('train_grl_pos_logit_mean')
        self.neg_logit = self._arr('train_grl_neg_logit_mean')
        self.margin = self._arr('train_grl_logit_margin')

        n = self.bacc.size
        ep = self._arr('epoch')
        self.epochs = ep if ep.size == n and n > 0 else np.arange(1, n + 1)

        self.grl_w = float(getattr(self.config, 'grl_loss_weight', 1.0)) if self.config else 1.0
        self.warmup = self._resolve_warmup(n)
        self.active = (self.epochs > self.warmup) if n else np.zeros(0, bool)

    # ------------------------------------------------------------------ utils
    def _arr(self, key: str) -> np.ndarray:
        v = self.history.get(key, [])
        try:
            return np.asarray(v, dtype=float)
        except (TypeError, ValueError):
            return np.asarray([], dtype=float)

    def _resolve_warmup(self, n: int) -> float:
        w = getattr(self.config, 'teacher_only_warmup_epochs', None) if self.config else None
        if w is not None and w > 0:
            return float(w)
        ne = getattr(self.config, 'num_epochs', None) if self.config else None
        if ne and ne > 0:
            return float(ne) / 2.0
        return float(n) / 2.0 if n else 0.0

    def has_grl(self) -> bool:
        """True only when genuine classifier-GRL data is present.

        For SCAD / plain / WDGRL runs the classifier series are absent or constant
        so this returns False and the whole module is a no-op.
        """
        if self.bacc.size == 0:
            return False
        if self.config is not None:
            if not getattr(self.config, 'use_grl', False):
                return False
            if str(getattr(self.config, 'grl_mode', 'classifier')) != 'classifier':
                return False
            return True
        # No config: infer from data - a non-trivial classifier left a varying signal.
        return bool(np.any(self.cls_loss != 0) and np.nanstd(self.bacc) > 1e-6)

    def _post(self, a: np.ndarray) -> np.ndarray:
        """Slice an array to the post-warmup (GRL-active) region."""
        if a.size == self.active.size and self.active.size:
            return a[self.active]
        return a

    def _has(self, a: np.ndarray) -> bool:
        """True if the series carries real (non-all-zero, finite) signal."""
        return bool(a.size and np.any(np.isfinite(a)) and np.nanmax(np.abs(a)) > 0)

    @staticmethod
    def _corr(x: np.ndarray, y: np.ndarray) -> float:
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 3:
            return np.nan
        xs, ys = x[m], y[m]
        if np.std(xs) < 1e-12 or np.std(ys) < 1e-12:
            return np.nan
        return float(np.corrcoef(xs, ys)[0, 1])

    # ----------------------------------------------------------- annotations
    def _mark_phases(self, ax, best_ep=None, chance_line=False):
        if self.warmup and self.epochs.size:
            ax.axvline(self.warmup, color=_C['warmup'], ls='--', lw=1.2, alpha=0.7,
                       label=f'warmup end (ep{int(self.warmup)})')
        if chance_line:
            ax.axhline(0.5, color=_C['chance'], ls=':', lw=1.4, alpha=0.9, label='chance (0.5)')
        if best_ep is not None:
            ax.axvline(best_ep, color=_C['best'], ls='-', lw=1.0, alpha=0.5,
                       label=f'best eval (ep{int(best_ep)})')

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
        for k in ('pak_auc_f1', 'prc_auc', 'f1_t', 'affiliation_f1', 'disc_snr', 'recon_snr'):
            vals = np.array([r.get(k, np.nan) for r in rows], dtype=float)
            if np.any(np.isfinite(vals)):
                out[k] = vals
        return ev, out

    def _series_at(self, src: np.ndarray, eval_epochs: np.ndarray) -> np.ndarray:
        """Sample a per-epoch training series at the eval epochs (nearest train epoch)."""
        if self.epochs.size == 0 or src.size != self.epochs.size:
            return np.full(eval_epochs.shape, np.nan)
        idx = np.searchsorted(self.epochs, eval_epochs)
        idx = np.clip(idx, 0, self.epochs.size - 1)
        return src[idx]

    def _postwarm_mask(self, eval_epochs: np.ndarray) -> np.ndarray:
        """Mask selecting eval epochs in the GRL-active (post-warmup) region.

        Correlations MUST be restricted to post-warmup: GRL only acts there, and
        the pre-warmup signals are degenerate-zero (e.g. logit means == 0), which
        would otherwise inject a spurious sign into the correlation.
        """
        if self.warmup and eval_epochs is not None and eval_epochs.size:
            m = eval_epochs > self.warmup
            # < 3 post-warmup points: return all-False so _corr sees too-few points and
            # yields NaN ("insufficient data") rather than falling back to ALL points and
            # re-injecting the degenerate pre-warmup zeros (which would flip the corr sign).
            return m if m.sum() >= 3 else np.zeros_like(eval_epochs, bool)
        return np.ones_like(eval_epochs, bool) if eval_epochs is not None else None

    def _invariance_cont(self) -> np.ndarray:
        """Continuous invariance signal in [0, +): higher == more invariant.

        Prefers logit convergence (-|pos-neg logit mean|, shifted to be >=0); falls
        back to (1 - balanced_acc) when the logit series is unavailable (old runs).
        """
        if self._has(self.pos_logit) or self._has(self.neg_logit):
            sep = np.abs(self.pos_logit - self.neg_logit)
            return -sep  # larger (less negative) == means converged == more invariant
        return 1.0 - self.bacc

    def _best_eval_epoch(self) -> Optional[float]:
        ev, mets = self._load_epoch_metrics()
        if ev is None or 'pak_auc_f1' not in mets:
            return None
        v = mets['pak_auc_f1']
        if not np.any(np.isfinite(v)):
            return None
        return float(ev[int(np.nanargmax(v))])

    # ====================================================================== API
    def generate_all(self) -> List[str]:
        if not self.has_grl():
            print("  - [grl_diagnostics] skipped (no classifier-GRL data)")
            return []
        os.makedirs(self.output_dir, exist_ok=True)
        written = []
        for fn in (self._fig_adversarial_progress,
                   self._fig_game_health,
                   self._fig_optimization_signal,
                   self._fig_detection_coupling,
                   self._fig_transfer,
                   self._fig_summary):
            try:
                path = fn()
                if path:
                    written.append(path)
                    print(f"  - {os.path.relpath(path, self.output_dir)}")
            except Exception as e:  # never let a viz failure break the run
                print(f"  - [grl_diagnostics] {fn.__name__} failed: {e}")
            finally:
                plt.close('all')
        summary_path = self._write_summary()
        if summary_path:
            written.append(summary_path)
        written += self._relocate_existing()
        return written

    # ------------------------------------------------- relocate legacy GRL viz
    def _relocate_existing(self) -> List[str]:
        """Move pre-existing GRL figures into grl_diagnostics/ (single home).

        best_model/GRL_contribution_trend.png and epoch_metrics/epoch_grl.png are
        produced by other modules at their own paths; we move them here AFTER they
        are generated. No-op if they don't exist. Never raises.
        """
        moved = []
        viz_dir = os.path.dirname(self.output_dir.rstrip('/')) or '.'
        candidates = [
            os.path.join(viz_dir, 'best_model', 'GRL_contribution_trend.png'),
            os.path.join(viz_dir, 'epoch_metrics', 'epoch_grl.png'),
        ]
        for src in candidates:
            try:
                if os.path.exists(src):
                    dst = os.path.join(self.output_dir, os.path.basename(src))
                    shutil.move(src, dst)
                    moved.append(dst)
                    print(f"  - moved {os.path.basename(src)} -> grl_diagnostics/")
            except OSError as e:
                print(f"  - [grl_diagnostics] could not move {os.path.basename(src)}: {e}")
        return moved

    # ----------------------------------------------------------- verdict logic
    def compute_verdict(self) -> Dict:
        bacc_p = self._post(self.bacc)
        gap_p = self._post(self.gap)
        effw_p = self._post(self.eff_w)
        ramp_p = self._post(self.ramp)

        def fin(a):
            return a[np.isfinite(a)] if a.size else a

        # ---- (1) is the game ALIVE (not starved)? ----
        effw_f = fin(effw_p)
        starvation_fraction = float(np.mean(effw_f < _EPS)) if effw_f.size else np.nan
        eff_mean = float(np.nanmean(effw_p)) if effw_p.size else np.nan
        grl_active = bool(np.isfinite(eff_mean) and eff_mean > _EPS
                          and np.isfinite(starvation_fraction) and starvation_fraction < 0.5)

        # ---- (2) genuine invariance? classifier learned then was fooled ----
        bacc_f = fin(bacc_p)
        bacc_max = float(np.nanmax(bacc_f)) if bacc_f.size else np.nan
        bacc_final = float(bacc_f[-1]) if bacc_f.size else np.nan
        learned = bool(np.isfinite(bacc_max) and bacc_max > 0.6)   # classifier could discriminate
        dropped = bool(np.isfinite(bacc_final) and np.isfinite(bacc_max)
                       and bacc_final < bacc_max - 0.03 and bacc_final < 0.7)
        # logit convergence (continuous): |pos-neg| shrank
        sep = np.abs(self.pos_logit - self.neg_logit) if self._has(self.pos_logit) else np.array([])
        sep_p = fin(self._post(sep)) if sep.size else sep
        logit_converged = bool(sep_p.size > 1 and sep_p[-1] < sep_p[0] - 1e-6)
        invariance_achieved = bool(learned and (dropped or logit_converged))

        # ---- (3) degeneracy guard: balanced_acc=0.5 via class-collapse? ----
        gap_f = fin(gap_p)
        gap_max = float(np.nanmax(gap_f)) if gap_f.size else np.nan
        gap_final = float(gap_f[-1]) if gap_f.size else np.nan
        # SUSTAINED collapse (not a startup transient): late-region median gap high,
        # or balanced_acc driven below chance (anti-discrimination / one-class collapse).
        late = gap_f[max(0, int(len(gap_f) * 0.75)):] if gap_f.size else gap_f
        gap_late_med = float(np.nanmedian(late)) if late.size else np.nan
        class_collapse = bool((np.isfinite(gap_late_med) and gap_late_med > 0.4)
                              or (np.isfinite(bacc_final) and bacc_final < 0.45))

        # ---- (4) detection coupling (post-warmup eval epochs only) ----
        det_corr, det_metric = np.nan, None
        ev, mets = self._load_epoch_metrics()
        if ev is not None and 'pak_auc_f1' in mets:
            pm = self._postwarm_mask(ev)
            inv_at = self._series_at(1.0 - self.bacc, ev)
            det_corr = self._corr(inv_at[pm], mets['pak_auc_f1'][pm])
            det_metric = 'pak_auc_f1'
        detection_coupled = bool(np.isfinite(det_corr) and det_corr > 0.1)

        # ---- (5) invariance -> output discrepancy transfer (post-warmup only) ----
        transfer_corr = np.nan
        disc_first = disc_last = np.nan
        if ev is not None and 'disc_snr' in mets:
            pm = self._postwarm_mask(ev)
            inv_cont_at = self._series_at(self._invariance_cont(), ev)
            transfer_corr = self._corr(inv_cont_at[pm], mets['disc_snr'][pm])
            dv = fin(mets['disc_snr'][pm])
            if dv.size:
                disc_first, disc_last = float(dv[0]), float(dv[-1])
        transferred = bool(np.isfinite(transfer_corr) and transfer_corr > 0.1)

        # ---- assemble human verdict (priority order) ----
        if not np.isfinite(eff_mean):
            head = "insufficient GRL data"
        elif not grl_active:
            head = (f"GRL STARVED - effective_weight ~ 0 in {starvation_fraction*100:.0f}% of "
                    f"post-warmup epochs (mean {eff_mean:.3g}); adversarial signal inactive")
        elif class_collapse:
            if np.isfinite(gap_late_med) and gap_late_med > 0.4:
                head = (f"GRL DEGENERATE - sustained one-class collapse (late acc_gap median "
                        f"{gap_late_med:.2f}); balanced_acc~0.5 is fake, not invariance")
            else:
                head = (f"GRL DEGENERATE - classifier driven BELOW chance (balanced_acc final "
                        f"{bacc_final:.2f}); adversarial game broke down (over-fooled / anti-discrimination)")
        elif not invariance_achieved:
            head = (f"GRL active but NO invariance - classifier keeps discriminating "
                    f"(balanced_acc {bacc_max:.2f}->{bacc_final:.2f}); student not fooling it")
        elif not transferred:
            head = (f"invariance achieved but NO transfer - disc separation flat "
                    f"(transfer corr {transfer_corr:.2f}); adversarial pressure stuck in hidden")
        else:
            head = (f"GRL EFFECTIVE - active, invariance achieved (balanced_acc "
                    f"{bacc_max:.2f}->{bacc_final:.2f}), no collapse, transfers to disc "
                    f"(corr {transfer_corr:.2f})")

        return {
            'grl_active': grl_active,
            'starvation_fraction': starvation_fraction,
            'effective_weight_mean': eff_mean,
            'balanced_acc_max': bacc_max, 'balanced_acc_final': bacc_final,
            'classifier_learned': learned, 'balanced_acc_dropped': dropped,
            'logit_converged': logit_converged,
            'invariance_achieved': invariance_achieved,
            'acc_gap_max': gap_max, 'acc_gap_final': gap_final, 'acc_gap_late_median': gap_late_med,
            'class_collapse_suspected': class_collapse,
            'detection_corr': det_corr, 'detection_metric': det_metric,
            'detection_coupled': detection_coupled,
            'transfer_corr_invariance_disc': transfer_corr,
            'disc_snr_start': disc_first, 'disc_snr_final': disc_last,
            'transferred': transferred,
            'warmup': self.warmup,
            'verdict': head,
        }

    def _write_summary(self) -> Optional[str]:
        try:
            v = self.compute_verdict()
        except Exception as e:
            print(f"  - [grl_diagnostics] verdict failed: {e}")
            return None
        path = os.path.join(self.output_dir, 'grl_diagnostics_summary.json')
        with open(path, 'w') as f:
            json.dump(v, f, indent=2, default=lambda o: None
                      if isinstance(o, float) and not np.isfinite(o) else o)
        print(f"  - grl_diagnostics_summary.json")
        return path

    # ===================================================================== figures
    def _fig_adversarial_progress(self) -> Optional[str]:
        if self.bacc.size == 0:
            return None
        best_ep = self._best_eval_epoch()
        fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))

        # (A) balanced_acc -> 0.5 == invariance (or dead game)
        ax = axes[0]
        ax.plot(self.epochs, self.bacc, color=_C['bacc'], lw=2, marker='o', ms=2, label='balanced_acc')
        self._mark_phases(ax, best_ep=best_ep, chance_line=True)
        ax.set_ylim(0.35, 1.02)
        ax.set_xlabel('epoch'); ax.set_ylabel('balanced accuracy')
        ax.set_title('(A) Classifier discrimination\n(high->0.5 = student achieving invariance)', fontweight='bold')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # (B) classifier BCE (+ confidence)
        ax = axes[1]
        ax.plot(self.epochs, self.cls_loss, color=_C['loss'], lw=2, marker='s', ms=2, label='grl_cls_loss (BCE)')
        self._mark_phases(ax)
        ax.set_xlabel('epoch'); ax.set_ylabel('classifier BCE', color=_C['loss'])
        ax.tick_params(axis='y', labelcolor=_C['loss'])
        if self._has(self.margin):
            ax2 = ax.twinx()
            ax2.plot(self.epochs, self.margin, color=_C['margin'], lw=1.4, ls='--', label='mean|logit| (confidence)')
            ax2.set_ylabel('mean |logit|', color=_C['margin']); ax2.tick_params(axis='y', labelcolor=_C['margin'])
            l1, lb1 = ax.get_legend_handles_labels(); l2, lb2 = ax2.get_legend_handles_labels()
            ax.legend(l1 + l2, lb1 + lb2, fontsize=8, loc='best')
        else:
            ax.legend(fontsize=8)
        ax.set_title('(B) Classifier loss & confidence\n(decrease then plateau/rise = healthy fight)', fontweight='bold')
        ax.grid(alpha=0.3)

        # (C) actual adversarial pressure reaching the student
        ax = axes[2]
        ax.plot(self.epochs, self.eff_w, color=_C['weight'], lw=2, marker='^', ms=2, label='effective_weight')
        if self._has(self.ramp):
            ax.plot(self.epochs, self.ramp, color=_C['ramp'], lw=1.4, ls='--', label='ramp_lambda (reversal)')
        if self._has(self.grad_adv):
            pressure = self.eff_w * self.grad_adv
            ax2 = ax.twinx()
            ax2.plot(self.epochs, pressure, color=_C['pressure'], lw=1.6, label='eff_w x ||grad_adv||')
            ax2.set_ylabel('actual adv. gradient', color=_C['pressure']); ax2.tick_params(axis='y', labelcolor=_C['pressure'])
            l1, lb1 = ax.get_legend_handles_labels(); l2, lb2 = ax2.get_legend_handles_labels()
            ax.legend(l1 + l2, lb1 + lb2, fontsize=8, loc='best')
        else:
            ax.legend(fontsize=8)
        self._mark_phases(ax)
        ax.set_xlabel('epoch'); ax.set_ylabel('effective weight / ramp')
        ax.set_title('(C) Adversarial pressure on student\n(~0 = STARVED: GRL inactive)', fontweight='bold')
        ax.grid(alpha=0.3)

        fig.suptitle('GRL Diagnostics - Adversarial Progress  (is the game being played?)',
                     fontsize=14, fontweight='bold')
        path = os.path.join(self.output_dir, 'grl_adversarial_progress.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        return path

    def _fig_game_health(self) -> Optional[str]:
        """THE key failure-mode check: genuine invariance vs classifier class-collapse."""
        if self.bacc.size == 0:
            return None
        best_ep = self._best_eval_epoch()
        fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))

        # (A) TPR & TNR co-trajectory
        ax = axes[0]
        ax.plot(self.epochs, self.tpr, color=_C['tpr'], lw=2, label='anomaly_acc (TPR)')
        ax.plot(self.epochs, self.tnr, color=_C['tnr'], lw=2, label='normal_acc (TNR)')
        self._mark_phases(ax, best_ep=best_ep, chance_line=True)
        ax.set_ylim(-0.02, 1.02); ax.set_xlabel('epoch'); ax.set_ylabel('accuracy')
        ax.set_title('(A) TPR vs TNR\n(both drop together = real; one pinned at 0/1 = collapse)', fontweight='bold')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # (B) degeneracy gap
        ax = axes[1]
        ax.plot(self.epochs, self.gap, color=_C['gap'], lw=2, marker='o', ms=2, label='|TNR-TPR| acc_gap')
        self._mark_phases(ax)
        ax.axhline(0.5, color=_C['chance'], ls=':', lw=1.2, alpha=0.8, label='collapse alarm (0.5)')
        ax.set_ylim(-0.02, 1.02); ax.set_xlabel('epoch'); ax.set_ylabel('degeneracy gap')
        ax.set_title('(B) Degeneracy gap\n(large = one-class collapse; balanced_acc 0.5 is fake)', fontweight='bold')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # (C) phase plot: balanced_acc vs acc_gap, epoch-coloured
        ax = axes[2]
        bacc_p, gap_p, ep_p = self._post(self.bacc), self._post(self.gap), self._post(self.epochs)
        m = np.isfinite(bacc_p) & np.isfinite(gap_p)
        if m.sum() >= 2:
            pts = np.array([bacc_p[m], gap_p[m]]).T.reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            lc = LineCollection(segs, cmap='viridis', linewidth=2)
            lc.set_array(ep_p[m][:-1]); ax.add_collection(lc)
            ax.scatter(bacc_p[m][0], gap_p[m][0], c='green', s=70, zorder=5, label='start', edgecolor='k')
            ax.scatter(bacc_p[m][-1], gap_p[m][-1], c='red', s=70, marker='*', zorder=5, label='end', edgecolor='k')
            cb = fig.colorbar(lc, ax=ax); cb.set_label('epoch', fontsize=8)
        ax.axvline(0.5, color=_C['chance'], ls=':', lw=1.2, alpha=0.7)
        ax.set_xlim(0.45, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.annotate('genuine\ninvariance', xy=(0.5, 0.05), fontsize=8, color='green', ha='center')
        ax.annotate('class-collapse\n(fake 0.5)', xy=(0.52, 0.85), fontsize=8, color='red')
        ax.set_xlabel('balanced_acc'); ax.set_ylabel('acc_gap')
        ax.set_title('(C) Invariance vs collapse phase trajectory', fontweight='bold')
        ax.legend(fontsize=8, loc='upper right'); ax.grid(alpha=0.3)

        fig.suptitle('GRL Diagnostics - Game Health  (is balanced_acc->0.5 real invariance, or collapse?)',
                     fontsize=14, fontweight='bold')
        path = os.path.join(self.output_dir, 'grl_game_health.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        return path

    def _fig_optimization_signal(self) -> Optional[str]:
        if self.bacc.size == 0:
            return None
        fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))

        # (A) adaptive lambda ratio + clamp bounds
        ax = axes[0]
        ax.plot(self.epochs, self.adapt_lam, color=_C['weight'], lw=2, marker='o', ms=2, label='grl_lambda = ||g_main||/||g_adv||')
        ax.axhline(10.0, color=_C['chance'], ls=':', lw=1.0, alpha=0.7, label='clamp [0,10]')
        ax.axhline(0.0, color=_C['chance'], ls=':', lw=1.0, alpha=0.7)
        self._mark_phases(ax)
        ax.set_xlabel('epoch'); ax.set_ylabel('adaptive lambda')
        ax.set_title('(A) Adaptive lambda ratio\n(0 = adv starved, 10 = saturated)', fontweight='bold')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # (B) absolute grad norms: main vs adversarial
        ax = axes[1]
        if self._has(self.grad_main) or self._has(self.grad_adv):
            ax.plot(self.epochs, self.grad_main, color=_C['grad_m'], lw=2, label='||grad_main||')
            ax.plot(self.epochs, self.grad_adv, color=_C['grad_a'], lw=2, label='||grad_adv|| (ramp baked in)')
            ax.set_yscale('log')
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, 'grad-norm series unavailable\n(non-legacy GRL balance mode\nor pre-2026-06-17 run)',
                    ha='center', va='center', transform=ax.transAxes, fontsize=10, color='gray')
        self._mark_phases(ax)
        ax.set_xlabel('epoch'); ax.set_ylabel('gradient norm (log)')
        ax.set_title('(B) Absolute adversarial pressure\n(adv << main = starved in absolute terms)', fontweight='bold')
        ax.grid(alpha=0.3, which='both')

        # (C) effective weight + starvation fraction text
        ax = axes[2]
        ax.plot(self.epochs, self.eff_w, color=_C['weight'], lw=2, marker='^', ms=2, label='effective_weight')
        if self._has(self.ramp):
            ax.plot(self.epochs, self.ramp, color=_C['ramp'], lw=1.4, ls='--', label='ramp_lambda')
        self._mark_phases(ax)
        effw_p = self._post(self.eff_w); effw_f = effw_p[np.isfinite(effw_p)]
        sfrac = float(np.mean(effw_f < _EPS)) if effw_f.size else np.nan
        emean = float(np.nanmean(effw_p)) if effw_p.size else np.nan
        txt = (f"post-warmup starvation:\n  eff_w < {_EPS:g} in {sfrac*100:.0f}% epochs\n"
               f"  mean eff_w = {emean:.3g}\n  grl_loss_weight = {self.grl_w:g}")
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round', fc='#FFF3E0', ec='#EF6C00', alpha=0.9))
        ax.set_xlabel('epoch'); ax.set_ylabel('effective weight / ramp')
        ax.set_title('(C) Loss weighting & starvation', fontweight='bold')
        ax.legend(fontsize=8, loc='upper right'); ax.grid(alpha=0.3)

        fig.suptitle('GRL Diagnostics - Optimization Dynamics & Starvation', fontsize=14, fontweight='bold')
        path = os.path.join(self.output_dir, 'grl_optimization_signal.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        return path

    def _fig_detection_coupling(self) -> Optional[str]:
        ev, mets = self._load_epoch_metrics()
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
        inv = 1.0 - self.bacc  # invariance proxy

        # (A) invariance vs detection over epochs
        ax = axes[0]
        ax.plot(self.epochs, inv, color=_C['bacc'], lw=2, label='invariance (1 - balanced_acc)')
        ax.set_xlabel('epoch'); ax.set_ylabel('invariance', color=_C['bacc'])
        ax.tick_params(axis='y', labelcolor=_C['bacc'])
        self._mark_phases(ax)
        det_corr = np.nan
        if ev is not None and 'pak_auc_f1' in mets:
            ax2 = ax.twinx()
            ax2.plot(ev, mets['pak_auc_f1'], color=_C['det'], lw=2, marker='o', ms=3, label='pak_auc_f1')
            ax2.set_ylabel('pak_auc_f1', color=_C['det']); ax2.tick_params(axis='y', labelcolor=_C['det'])
            l1, lb1 = ax.get_legend_handles_labels(); l2, lb2 = ax2.get_legend_handles_labels()
            ax.legend(l1 + l2, lb1 + lb2, fontsize=8, loc='best')
            _pm = self._postwarm_mask(ev)
            det_corr = self._corr(self._series_at(inv, ev)[_pm], mets['pak_auc_f1'][_pm])
        else:
            ax.legend(fontsize=8)
            ax.text(0.5, 0.5, 'epoch_metrics.json unavailable', ha='center', va='center',
                    transform=ax.transAxes, fontsize=10, color='gray')
        ax.set_title('(A) Invariance vs detection over training', fontweight='bold')
        ax.grid(alpha=0.3)

        # (B) scatter invariance vs detection
        ax = axes[1]
        if ev is not None and 'pak_auc_f1' in mets:
            inv_at = self._series_at(inv, ev)
            sc = ax.scatter(inv_at, mets['pak_auc_f1'], c=ev, cmap='viridis', s=40, edgecolor='k', lw=0.3)
            cb = fig.colorbar(sc, ax=ax); cb.set_label('epoch', fontsize=8)
            ax.set_xlabel('invariance (1 - balanced_acc)'); ax.set_ylabel('pak_auc_f1')
            ax.set_title(f'(B) Coupling  (corr = {det_corr:.2f}, post-warmup)', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'no detection metrics', ha='center', va='center',
                    transform=ax.transAxes, fontsize=10, color='gray')
            ax.set_title('(B) Coupling', fontweight='bold')
        ax.grid(alpha=0.3)

        fig.suptitle('GRL Diagnostics - Detection Coupling  (does invariance help real detection?)',
                     fontsize=14, fontweight='bold')
        path = os.path.join(self.output_dir, 'grl_detection_coupling.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        return path

    def _fig_transfer(self) -> Optional[str]:
        ev, mets = self._load_epoch_metrics()
        fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))
        inv_cont = self._invariance_cont()

        # (A) disc_snr (output separation) + logit separation
        ax = axes[0]
        has_disc = ev is not None and 'disc_snr' in mets
        if has_disc:
            ax.plot(ev, mets['disc_snr'], color=_C['disc'], lw=2, marker='o', ms=3, label='disc_snr (output sep.)')
            ax.set_ylabel('disc_snr', color=_C['disc']); ax.tick_params(axis='y', labelcolor=_C['disc'])
        else:
            ax.text(0.5, 0.5, 'disc_snr unavailable', ha='center', va='center',
                    transform=ax.transAxes, fontsize=10, color='gray')
        if self._has(self.pos_logit):
            ax2 = ax.twinx()
            ax2.plot(self.epochs, np.abs(self.pos_logit - self.neg_logit), color=_C['margin'], lw=1.5, ls='--',
                     label='|pos-neg logit| (hidden sep.)')
            ax2.set_ylabel('logit separation', color=_C['margin']); ax2.tick_params(axis='y', labelcolor=_C['margin'])
            l1, lb1 = ax.get_legend_handles_labels(); l2, lb2 = ax2.get_legend_handles_labels()
            ax.legend(l1 + l2, lb1 + lb2, fontsize=8, loc='best')
        else:
            ax.legend(fontsize=8)
        self._mark_phases(ax)
        ax.set_xlabel('epoch')
        ax.set_title('(A) Hidden invariance vs output separation', fontweight='bold')
        ax.grid(alpha=0.3)

        # (B) invariance vs disc_snr (twin) over epochs
        ax = axes[1]
        ax.plot(self.epochs, inv_cont, color=_C['bacc'], lw=2, label='invariance (-(|pos-neg logit|))')
        ax.set_xlabel('epoch'); ax.set_ylabel('invariance (continuous)', color=_C['bacc'])
        ax.tick_params(axis='y', labelcolor=_C['bacc']); self._mark_phases(ax)
        if has_disc:
            ax2 = ax.twinx()
            ax2.plot(ev, mets['disc_snr'], color=_C['disc'], lw=2, marker='o', ms=3, label='disc_snr')
            ax2.set_ylabel('disc_snr', color=_C['disc']); ax2.tick_params(axis='y', labelcolor=_C['disc'])
            l1, lb1 = ax.get_legend_handles_labels(); l2, lb2 = ax2.get_legend_handles_labels()
            ax.legend(l1 + l2, lb1 + lb2, fontsize=8, loc='best')
        else:
            ax.legend(fontsize=8)
        ax.set_title('(B) Invariance & output disc. together\n(rise together = transfer)', fontweight='bold')
        ax.grid(alpha=0.3)

        # (C) scatter + corr
        ax = axes[2]
        tcorr = np.nan
        if has_disc:
            inv_at = self._series_at(inv_cont, ev)
            sc = ax.scatter(inv_at, mets['disc_snr'], c=ev, cmap='plasma', s=40, edgecolor='k', lw=0.3)
            cb = fig.colorbar(sc, ax=ax); cb.set_label('epoch', fontsize=8)
            _pm = self._postwarm_mask(ev)
            tcorr = self._corr(inv_at[_pm], mets['disc_snr'][_pm])
            ax.set_xlabel('invariance (continuous)'); ax.set_ylabel('disc_snr')
            ax.set_title(f'(C) Transfer  (corr = {tcorr:.2f}, post-warmup)\n(flat/negative = pressure stuck in hidden)', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'disc_snr unavailable', ha='center', va='center',
                    transform=ax.transAxes, fontsize=10, color='gray')
            ax.set_title('(C) Transfer', fontweight='bold')
        ax.grid(alpha=0.3)

        fig.suptitle('GRL Diagnostics - Invariance -> Discrepancy Transfer  (does invariance reach the score?)',
                     fontsize=14, fontweight='bold')
        path = os.path.join(self.output_dir, 'grl_transfer.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        return path

    def _fig_summary(self) -> Optional[str]:
        v = self.compute_verdict()
        fig = plt.figure(figsize=(16, 9))
        gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.28)

        # balanced_acc
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(self.epochs, self.bacc, color=_C['bacc'], lw=2)
        ax.axhline(0.5, color=_C['chance'], ls=':', lw=1.2)
        if self.warmup:
            ax.axvline(self.warmup, color=_C['warmup'], ls='--', lw=1.0)
        ax.set_title('balanced_acc (->0.5 = invariance)', fontsize=10, fontweight='bold')
        ax.set_ylim(0.35, 1.02); ax.grid(alpha=0.3)

        # acc_gap
        ax = fig.add_subplot(gs[0, 1])
        ax.plot(self.epochs, self.gap, color=_C['gap'], lw=2)
        ax.axhline(0.5, color=_C['chance'], ls=':', lw=1.2)
        if self.warmup:
            ax.axvline(self.warmup, color=_C['warmup'], ls='--', lw=1.0)
        ax.set_title('acc_gap (collapse guard)', fontsize=10, fontweight='bold')
        ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3)

        # effective weight (starvation)
        ax = fig.add_subplot(gs[0, 2])
        ax.plot(self.epochs, self.eff_w, color=_C['weight'], lw=2)
        if self._has(self.ramp):
            ax.plot(self.epochs, self.ramp, color=_C['ramp'], lw=1.2, ls='--')
        if self.warmup:
            ax.axvline(self.warmup, color=_C['warmup'], ls='--', lw=1.0)
        ax.set_title('effective_weight (starvation)', fontsize=10, fontweight='bold')
        ax.grid(alpha=0.3)

        # transfer scatter (or invariance)
        ax = fig.add_subplot(gs[1, 0])
        ev, mets = self._load_epoch_metrics()
        if ev is not None and 'disc_snr' in mets:
            inv_at = self._series_at(self._invariance_cont(), ev)
            ax.scatter(inv_at, mets['disc_snr'], c=ev, cmap='plasma', s=25, edgecolor='k', lw=0.2)
            ax.set_xlabel('invariance'); ax.set_ylabel('disc_snr')
            ax.set_title('invariance -> disc transfer', fontsize=10, fontweight='bold')
        else:
            ax.plot(self.epochs, 1.0 - self.bacc, color=_C['bacc'], lw=2)
            ax.set_title('invariance (1-balanced_acc)', fontsize=10, fontweight='bold')
        ax.grid(alpha=0.3)

        # verdict text panel
        ax = fig.add_subplot(gs[1, 1:]); ax.axis('off')

        def g(x, f='{:.3f}'):
            try:
                return f.format(x) if (x is not None and np.isfinite(x)) else 'n/a'
            except (TypeError, ValueError):
                return 'n/a'

        lines = [
            "GRL adversarial-game verdict",
            "",
            f"- active (not starved) : {v['grl_active']}  "
            f"(starvation {g(v['starvation_fraction']*100 if np.isfinite(v['starvation_fraction']) else np.nan,'{:.0f}')}% , "
            f"mean eff_w {g(v['effective_weight_mean'])})",
            f"- invariance achieved  : {v['invariance_achieved']}  "
            f"(balanced_acc {g(v['balanced_acc_max'],'{:.2f}')}->{g(v['balanced_acc_final'],'{:.2f}')}, "
            f"logit converged {v['logit_converged']})",
            f"- class-collapse       : {v['class_collapse_suspected']}  "
            f"(acc_gap max {g(v['acc_gap_max'],'{:.2f}')})",
            f"- detection coupled    : {v['detection_coupled']}  "
            f"(corr {g(v['detection_corr'],'{:.2f}')})",
            f"- transfers to disc    : {v['transferred']}  "
            f"(corr {g(v['transfer_corr_invariance_disc'],'{:.2f}')}, "
            f"disc_snr {g(v['disc_snr_start'],'{:.2f}')}->{g(v['disc_snr_final'],'{:.2f}')})",
            "",
            "verdict: " + v['verdict'],
        ]
        ax.text(0.0, 0.98, "\n".join(lines), transform=ax.transAxes, fontsize=10.5,
                va='top', family='monospace',
                bbox=dict(boxstyle='round', fc='#F5F5F5', ec='#9E9E9E', alpha=0.95))

        fig.suptitle('GRL Diagnostics - Summary', fontsize=15, fontweight='bold')
        path = os.path.join(self.output_dir, 'grl_diagnostics_summary.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        return path
