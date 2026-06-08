#!/usr/bin/env python
"""Phase 3 (L2) viz regen for non-flip cells — mirrors reviz_noflip_lambda.py.

For each of the 160 non-flip (reviz) cells:
  1. epoch_metrics trend PNGs  <- plot_epoch_metrics(float32 epoch_metrics.json)   [CPU, all]
  2. anomaly_threshold.png     <- BestModelVisualizer.plot_anomaly_threshold(npz)  [CPU, all]
  3. (base cells w/ best_model.pt only, --gpu) full best_model figures via
     reviz_one_best_model.py (score-overlay / reconstruction). Simple cells have
     no checkpoint -> their reconstruction figures are deterministic & unchanged
     for a no-flip cell, so left as-is (per reviz_noflip_lambda precedent).

Backup is done separately (temp/reexp_phase3_backup_20260608) BEFORE running this.
Usage:
  python scripts/reexp_phase3_viz.py --spot-check        # 1 simple + 1 base
  python scripts/reexp_phase3_viz.py --cpu               # CPU regen, all 160
  python scripts/reexp_phase3_viz.py --gpu               # base-21 full best_model (GPU)
"""
import os, sys, json, glob, argparse, subprocess
sys.argv_backup = list(sys.argv)
sys.argv = ['reviz']  # neutralize argparse in imported run_base
sys.path.insert(0, '/home/ykio/notebooks/TSMAE')
from scripts.run_base_experiments import plot_epoch_metrics  # noqa
sys.argv = sys.argv_backup

BASE_KW = ('SWaT', 'WaDi', 'PSM')


def is_base(cell_name):
    return any(k in cell_name for k in BASE_KW)


def cpu_regen(cell_dir):
    out = []
    epoch_viz = os.path.join(cell_dir, 'visualization', 'epoch_metrics')
    best_viz = os.path.join(cell_dir, 'visualization', 'best_model')
    em = os.path.join(cell_dir, 'epoch_metrics.json')
    if os.path.exists(em):
        rows = json.load(open(em))['epochs']
        os.makedirs(epoch_viz, exist_ok=True)
        plot_epoch_metrics(rows, epoch_viz)
        out.append(f'epoch_metrics({len(rows)})')
    try:
        from mae_anomaly.visualization import BestModelVisualizer
        from mae_anomaly import Config
        bcp = os.path.join(cell_dir, 'best_config.json')
        bc = json.load(open(bcp)) if os.path.exists(bcp) else {}
        cfg = Config()
        for k, v in bc.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        viz = BestModelVisualizer.__new__(BestModelVisualizer)
        viz.output_dir = best_viz
        viz.config = cfg
        viz.test_loader = None
        os.makedirs(best_viz, exist_ok=True)
        viz.plot_anomaly_threshold(experiment_dir=cell_dir)
        out.append('anomaly_threshold')
    except Exception as e:
        out.append(f'anomaly_threshold FAIL: {type(e).__name__}: {e}')
    return out


def iter_cells():
    m = json.load(open('temp/reexp_manifest.json'))
    for e in m['exps']:
        for c in e['reviz_cells']:
            yield e['exp'], os.path.join(e['dir'], c), c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spot-check', action='store_true')
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--gpu', action='store_true')
    args = ap.parse_args()
    cells = list(iter_cells())

    if args.spot_check:
        sel = [c for c in cells if 'MSL/F-7' in c[1]][:1] + [c for c in cells if c[1].endswith('/PSM')][:1]
        for exp, cdir, cname in sel:
            r = cpu_regen(cdir)
            print(f'  exp{exp} {cname} (base={is_base(cname)}): {r}')
        return

    if args.cpu:
        ok = fail = 0
        for exp, cdir, cname in cells:
            r = cpu_regen(cdir)
            bad = any('FAIL' in x for x in r)
            ok += (0 if bad else 1); fail += (1 if bad else 0)
            if bad:
                print(f'  FAIL exp{exp} {cname}: {r}')
        print(f'\n=== Phase3 CPU viz: {ok} ok, {fail} fail / {len(cells)} ===')
        return

    if args.gpu:
        base_cells = [(e, d, n) for e, d, n in cells if is_base(n) and os.path.exists(os.path.join(d, 'best_model.pt'))]
        print(f'GPU best_model regen: {len(base_cells)} base cells')
        ok = fail = 0
        for exp, cdir, cname in base_cells:
            p = subprocess.run([sys.executable, 'scripts/reviz_one_best_model.py', cdir],
                               capture_output=True, text=True)
            good = 'BEST_MODEL_VIZ_OK' in p.stdout
            ok += (1 if good else 0); fail += (0 if good else 1)
            print(f'  {"OK" if good else "FAIL"} exp{exp} {cname}' + ('' if good else f'\n     {p.stdout[-200:]}{p.stderr[-200:]}'))
        print(f'\n=== Phase3 GPU best_model: {ok} ok, {fail} fail / {len(base_cells)} ===')
        return

    ap.print_help()


if __name__ == '__main__':
    main()
