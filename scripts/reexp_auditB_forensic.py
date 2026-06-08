#!/usr/bin/env python
"""Audit B — empirical generalization of best_checkpoint == model@best_epoch.

For each target simple flip cell:
  1. Build the EXACT --config-override from its saved best_config.json (so the
     re-run reproduces the original bit-for-bit, independent of Set-C preset drift).
  2. Re-run via run_base into a temp dir with:
       TSMAE_SAVE_EPOCH_MODELS=1  -> saves a CPU clone of model.state_dict() at
                                      every eval epoch under TSMAE_EPOCH_MODEL_DIR
       KEEP_BEST_CKPT=1           -> keeps best_checkpoint.pt
  3. Forensic check: best_checkpoint.pt['model_state_dict'] is bit-identical to
     the per-epoch clone @ best_epoch  -> best_checkpoint really IS model@best.
  4. Determinism check: re-run npz@best == original npz@best (bit).

Run AFTER temporarily re-adding the env-guarded epoch-model hook to run_base.
Usage: python scripts/reexp_auditB_forensic.py
"""
import json, os, subprocess, sys, shutil
import numpy as np
import torch

PROJECT = '/home/ykio/notebooks/TSMAE'
EXP = os.path.join(PROJECT, 'results/experiments/271_20260602_020545_271canon_baseline')
CELLS = [('MSL/C-2', 'MSL_simple_C-2'),
         ('SMD/machine-1-4', 'SMD_simple_machine-1-4'),
         ('SMAP/T-3', 'SMAP_simple_T-3')]
SKIP_KEYS = {'num_features', 'device'}


def fmt(v):
    if isinstance(v, bool):
        return 'True' if v else 'False'
    if v is None:
        return 'none'
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        s = repr(v)
        return s if '.' in s or 'e' in s else s + '.0'
    if isinstance(v, (tuple, list)):
        if not v:
            return None
        return '(' + ','.join(fmt(x) for x in v) + ')'
    if isinstance(v, str):
        return v if ' ' not in v else None
    return None


def build_override(cell_dir):
    bc = json.load(open(os.path.join(cell_dir, 'best_config.json')))
    ov = []
    for k, v in bc.items():
        if k in SKIP_KEYS:
            continue
        f = fmt(v)
        if f is None:
            continue
        ov.append(f'{k}={f}')
    return ov, bc


def main():
    results = []
    for cellrel, arg in CELLS:
        cell_dir = os.path.join(EXP, cellrel)
        md = json.load(open(os.path.join(cell_dir, 'experiment_metadata.json')))
        be = int(md['timing']['best_epoch'])
        ov, bc = build_override(cell_dir)
        tag = arg
        out_base = os.path.join(PROJECT, f'temp/auditB_{tag}')
        em_dir = f'/tmp/em_{tag}'
        shutil.rmtree(out_base, ignore_errors=True)
        shutil.rmtree(em_dir, ignore_errors=True)
        env = dict(os.environ, TSMAE_SAVE_EPOCH_MODELS='1',
                   TSMAE_EPOCH_MODEL_DIR=em_dir, KEEP_BEST_CKPT='1')
        cmd = [sys.executable, 'scripts/run_base_experiments.py', '--set', 'C', '--no-wait',
               '--output-base', out_base, '--dataset', arg, '--config-override'] + ov
        print(f'\n##### Audit B: {cellrel} (best_epoch={be}) — re-run #####', flush=True)
        rc = subprocess.run(cmd, cwd=PROJECT, env=env,
                            stdout=open(f'/tmp/auditB_{tag}.log', 'w'), stderr=subprocess.STDOUT).returncode

        # --- forensic: best_checkpoint == clone@best_epoch ? ---
        bcp = os.path.join(out_base, cellrel, 'checkpoints', 'best_checkpoint.pt')
        clone_p = os.path.join(em_dir, f'ep{be:03d}.pt')
        verdict = {}
        if os.path.exists(bcp) and os.path.exists(clone_p):
            bcw = torch.load(bcp, map_location='cpu', weights_only=False)['model_state_dict']
            clw = torch.load(clone_p, map_location='cpu', weights_only=False)
            label = torch.load(bcp, map_location='cpu', weights_only=False).get('epoch')
            maxd = max((bcw[k] - clw[k]).abs().max().item()
                       for k in bcw if torch.is_tensor(bcw[k]) and k in clw)
            # which epoch does best_checkpoint actually match?
            match_eps = []
            for f in sorted(os.listdir(em_dir)):
                ep = int(f[2:5]); m = torch.load(os.path.join(em_dir, f), map_location='cpu', weights_only=False)
                d = max((bcw[k] - m[k]).abs().max().item() for k in bcw if torch.is_tensor(bcw[k]) and k in m)
                if d == 0:
                    match_eps.append(ep)
            verdict = {'label_epoch': label, 'maxd_vs_clone@best': maxd,
                       'weight_identical_epochs': match_eps,
                       'is_model_at_best': (maxd == 0 and match_eps == [be])}
        else:
            verdict = {'error': f'missing bcp={os.path.exists(bcp)} clone={os.path.exists(clone_p)}'}

        # --- determinism: re-run npz@best == original npz@best ? ---
        new_npz = os.path.join(out_base, cellrel, 'epoch_scores', f'epoch_{be:03d}_scores.npz')
        old_npz = os.path.join(cell_dir, 'epoch_scores', f'epoch_{be:03d}_scores.npz')
        det = None
        if os.path.exists(new_npz) and os.path.exists(old_npz):
            a = np.load(new_npz); b = np.load(old_npz)
            det = max(np.abs(a[k].astype(np.float64) - b[k].astype(np.float64)).max()
                      for k in ['adaptive_score', 'teacher_recon_error', 'discrepancy_error'])
        results.append((cellrel, be, rc, verdict, det))
        print(f'  -> {cellrel}: {verdict} | npz@best max|Δ|vs original={det}', flush=True)

    print('\n=== AUDIT B SUMMARY ===')
    allok = True
    for cellrel, be, rc, v, det in results:
        ok = v.get('is_model_at_best') and (det is not None and det == 0)
        allok = allok and ok
        print(f'  {cellrel:18s} best={be:3d}: best_checkpoint==model@best? {v.get("is_model_at_best")} '
              f'(match_eps={v.get("weight_identical_epochs")}) | npz bit-identical? {det==0 if det is not None else "?"}')
    print('\n  >>> ' + ('ALL: best_checkpoint == model@best AND deterministic ✓'
                        if allok else 'SOME FAILED ✗ — review above'))


if __name__ == '__main__':
    main()
