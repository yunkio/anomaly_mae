#!/usr/bin/env python
"""Phase 2 — re-experiment flipped cells (2026-06-06).

Reads temp/reexp_manifest.json. For each of the 10 completed experiments:
  1. DELETE the reexp_cells dirs (flip cells + both SWaT full/excl22).
  2. run_base --set C --output-base <exp_dir> --dataset <flip_args>
     --config-override <override>   (re-runs ONLY flip_args datasets into the
     SAME dir; --dataset filter leaves the 160 non-flip cells untouched).

SWaT_A1A2 is one arg → native dual-eval regenerates full+excl22 together.
Config = Set C preset + queue override == original (verified). Deterministic
(set_seed + per-epoch manual_seed) → reproduces flip-list new_best.

Each exp's flip-cell best epochs are verified separately (scripts/reexp_verify.py).

Usage: python scripts/reexp_phase2.py [--only EXP] [--start-exp EXP]
"""
import json, os, subprocess, sys, shutil, datetime, argparse

PROJECT = '/home/ykio/notebooks/TSMAE'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', type=int, default=None, help='run only this exp')
    ap.add_argument('--start-exp', type=int, default=0, help='start from this exp num')
    ap.add_argument('--resume', action='store_true',
                    help='preserve cells already re-run this session (metadata lacks '
                         '_evalrevert_recompute) so run_base skips them; only re-run the rest')
    args = ap.parse_args()

    m = json.load(open(os.path.join(PROJECT, 'temp/reexp_manifest.json')))
    exps = [e for e in m['exps'] if (args.only is None or e['exp'] == args.only) and e['exp'] >= args.start_exp]
    print(f"PHASE2 START {datetime.datetime.now():%Y%m%d_%H%M%S} — {len(exps)} exps, "
          f"{sum(len(e['reexp_cells']) for e in exps)} reexp cells", flush=True)

    for e in exps:
        d = e['dir']
        assert os.path.isdir(d), f"exp dir missing: {d}"
        # 1. delete reexp cells (idempotent). --resume: keep cells already re-run this
        #    session (fresh metadata has NO '_evalrevert_recompute' field; the original
        #    recompute-touched cells DO) so run_base's dataset-skip marker skips them.
        kept = 0
        for cell in e['reexp_cells']:
            p = os.path.join(d, cell)
            mdp = os.path.join(p, 'experiment_metadata.json')
            if args.resume and os.path.exists(mdp):
                try:
                    md = json.load(open(mdp))
                    if '_evalrevert_recompute' not in md:
                        kept += 1
                        continue  # freshly re-run — preserve (run_base will skip via marker)
                except Exception:
                    pass
            if os.path.isdir(p):
                shutil.rmtree(p)
        if args.resume and kept:
            print(f"  [resume] preserved {kept} already-re-run cells (run_base will skip them)", flush=True)
        # sanity: non-flip cells untouched (count check)
        remaining = sum(os.path.exists(os.path.join(d, c, 'experiment_metadata.json')) for c in e['reviz_cells'])
        print(f"\n##### exp{e['exp']} ({os.path.basename(d)}): deleted {len(e['reexp_cells'])} reexp cells; "
              f"{remaining}/{len(e['reviz_cells'])} reviz cells intact #####", flush=True)
        assert remaining == len(e['reviz_cells']), "reviz cell got deleted!"
        # 2. re-run flip_args
        cmd = [sys.executable, 'scripts/run_base_experiments.py', '--set', e['set'], '--no-wait',
               '--output-base', d, '--dataset'] + e['flip_args'] + ['--config-override', e['override']]
        print(f"  CMD: run_base --set {e['set']} --dataset ({len(e['flip_args'])} args) -> {os.path.basename(d)}", flush=True)
        t0 = datetime.datetime.now()
        rc = subprocess.run(cmd, cwd=PROJECT).returncode
        dt = (datetime.datetime.now() - t0).total_seconds() / 60
        print(f"##### exp{e['exp']} re-run done rc={rc} in {dt:.1f}min #####", flush=True)

    print("\n=== PHASE2 DONE ===", flush=True)


if __name__ == '__main__':
    main()
