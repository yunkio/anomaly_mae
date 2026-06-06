#!/usr/bin/env python
"""Verify re-experimented cells' best epochs match the expected (flip-list) values.

After Phase 2 re-runs the flip cells (with the 2026-06-06 float32-parity fix), each
re-run cell's timing['best_epoch'] MUST equal the flip-list new_best
(temp/reexp_expected_best.json). Any mismatch => STOP (determinism / fix problem).

Usage: python scripts/reexp_verify.py [--only EXP]
"""
import json, os, glob, sys, argparse


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--only', type=int, default=None)
    args = ap.parse_args()
    m = json.load(open('temp/reexp_manifest.json'))
    expected = json.load(open('temp/reexp_expected_best.json'))
    ok = bad = missing = 0
    for e in m['exps']:
        if args.only and e['exp'] != args.only:
            continue
        for cell in e['reexp_cells']:
            mdp = os.path.join(e['dir'], cell, 'experiment_metadata.json')
            key = f"{e['exp']}|{cell}"
            exp_be = expected.get(key, {}).get('expected')
            if not os.path.exists(mdp):
                print(f"  MISSING  exp{e['exp']} {cell} (not re-run yet)"); missing += 1; continue
            md = json.load(open(mdp))
            be = md.get('timing', {}).get('best_epoch')
            if be == exp_be:
                ok += 1
            else:
                print(f"  MISMATCH exp{e['exp']} {cell}: got best_epoch={be}, expected {exp_be}"); bad += 1
    print(f"\n=== reexp verify: {ok} OK, {bad} MISMATCH, {missing} not-yet-run ===")
    print("  >>> " + ("ALL MATCH ✓" if bad == 0 and missing == 0 else
                       f"{'MISMATCH — STOP & investigate' if bad else 'incomplete (still running)'}"))
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
