#!/usr/bin/env python3
"""Backend data-layer smoke test (CPU-only, read-only).

Exercises the whole data-access stack against the read-only fixtures + live results:
  1. discover experiments under .trash/0601/pre_fullrerun/ + results/experiments/
  2. parse the 271 fixture leaves
  3. build the DYNAMIC F5 metric catalog and report the number of keys
  4. run the registry resolver_self_test (MUST PASS — startup gate)
  5. print the resolved score formula for 271/PSM (MUST be 4:1, never blank)

Run:
    UI/.venv/bin/python UI/backend/smoke_backend.py
(or any python with numpy/pydantic/orjson available; degrades if a dep is missing).

Exit code 0 = all critical checks PASS; non-zero = a critical check FAILED.
"""
from __future__ import annotations

import sys
from pathlib import Path

# make the package importable when run as a script
HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))  # UI/backend  (so `import app...` works)

from app.config import SETTINGS  # noqa: E402
from app.dataaccess.repository import get_repository  # noqa: E402


FIXTURE_271 = "271_20260529_100418_271canon_baseline"


def _hr(title: str) -> None:
    print(f"\n{'=' * 4} {title} {'=' * (60 - len(title))}")


def main() -> int:
    failures: list[str] = []
    repo = get_repository()

    _hr("source roots")
    for k, v in SETTINGS.source_roots.items():
        print(f"  {k:8s} {v}  exists={v.exists()}")
    print(f"  registry  {SETTINGS.registry_path}  exists={SETTINGS.registry_path.exists()}")
    print(f"  cache     {SETTINGS.cache_root}")

    # 1 ── discovery ──────────────────────────────────────────────────────────
    _hr("1. discovery")
    exps = repo.experiments()
    print(f"  experiments discovered: {len(exps)}")
    for e in exps:
        ds_summary = ", ".join(f"{d.dataset_key}[{'/'.join(d.variants)}]={d.state}"
                               for d in e.datasets) or "(no leaves)"
        print(f"  - {e.exp_id}  src={e.source} state={e.state}")
        print(f"      datasets: {ds_summary}")
    if not exps:
        failures.append("discovery found ZERO experiments")

    # locate the 271 fixture
    exp271 = next((e for e in exps if e.exp_id == FIXTURE_271), None)
    if exp271 is None:
        # fall back to any 271canon fixture
        exp271 = next((e for e in exps if e.source == "fixture" and "271canon" in e.exp_id), None)
    if exp271 is None:
        failures.append("could not locate the 271 fixture experiment")
        print("  !! 271 fixture not found — aborting downstream checks")
        return _finish(failures)

    # in-progress detection sanity (live run should have an in_progress leaf)
    live_inprog = [e for e in exps if e.source == "live" and e.state == "in_progress"]
    print(f"  live in_progress experiments: {[e.exp_id for e in live_inprog]}")

    # 2 ── parse the 271 fixture ──────────────────────────────────────────────
    _hr("2. parse 271 fixture")
    psm = next((d for d in exp271.datasets if d.dataset_key == "PSM"), None)
    if psm is None:
        failures.append("271 fixture has no PSM dataset")
        return _finish(failures)
    _, _, leaf = repo.resolve_leaf(exp271.exp_id, "PSM", None)
    bundle = repo.load_bundle(exp271, psm, leaf)
    print(f"  PSM leaf: {bundle.path}")
    print(f"  state={leaf.state} warmup_epochs={leaf.warmup_epochs} "
          f"num_features={leaf.num_features} eval_interval={leaf.eval_interval}")
    if bundle.epoch:
        print(f"  epoch_metrics: {len(bundle.epoch.epochs)} eval-epochs, "
              f"{len(bundle.epoch.series)} scalar series, {len(bundle.epoch.array_keys)} array keys")
    else:
        failures.append("epoch_metrics did not parse for 271/PSM")
    if bundle.metadata:
        nblocks = len(bundle.metadata.score_variants)
        print(f"  metadata: {nblocks} score-variant blocks, "
              f"{len(bundle.metadata.loss_stats)} loss_stats, {len(bundle.metadata.config)} config keys")
    else:
        failures.append("experiment_metadata did not parse for 271/PSM")
    if bundle.history:
        print(f"  training_history: {len(bundle.history.series)} flat lists "
              f"({len(bundle.history.empty_keys)} empty), nested={list(bundle.history.nested)}")
        eats = bundle.history.nested.get("epoch_anomaly_type_scores")
        if eats:
            print(f"      anomaly-type pivot: types={eats.types} sub_metrics={eats.sub_metrics}")
    if bundle.anomaly_types:
        print(f"  anomaly_type_metrics: types={list(bundle.anomaly_types)}")
    if bundle.errors:
        print(f"  bundle.errors: {bundle.errors}")

    # 3 ── dynamic catalog ────────────────────────────────────────────────────
    _hr("3. dynamic F5 metric catalog")
    catalog = repo.leaf_catalog(bundle)
    by_ns: dict[str, int] = {}
    inferred = 0
    for c in catalog:
        by_ns[c.namespace] = by_ns.get(c.namespace, 0) + 1
        if c.inferred:
            inferred += 1
    print(f"  TOTAL catalog keys: {len(catalog)}  (inferred via fallback: {inferred})")
    for ns, n in sorted(by_ns.items()):
        print(f"    {ns:28s} {n}")
    # spot-check the primary metric resolves explicitly
    primary = next((c for c in catalog if c.key == "pak_auc_f1" and c.namespace == "epoch_metrics"), None)
    if primary:
        print(f"  pak_auc_f1 -> family={primary.family} dir={primary.direction} inferred={primary.inferred}")
        if primary.inferred or primary.direction != "up":
            failures.append("pak_auc_f1 did not resolve to explicit up-direction")
    else:
        failures.append("pak_auc_f1 absent from the catalog")
    if len(catalog) < 100:
        failures.append(f"catalog suspiciously small ({len(catalog)} keys)")

    # 4 ── resolver self-test gate ────────────────────────────────────────────
    _hr("4. resolver_self_test (startup gate)")
    st = repo.selftest
    print(f"  STATUS: {st['status']}  (registry v{st['registry_version']})")
    for g, r in st["groups"].items():
        flag = "OK" if not r["fail"] else "FAIL"
        print(f"    [{flag}] {g}: {r['n']} keys, {len(r['fail'])} failures")
        for f in r["fail"]:
            print(f"        - {f}")
    if st["status"] != "PASS":
        failures.append(f"resolver_self_test FAILED: {st['failures']}")

    # 5 ── score formula for 271/PSM ──────────────────────────────────────────
    _hr("5. score formula 271/PSM (must be 4:1, never blank)")
    sf = bundle.metadata.score_formula if bundle.metadata else None
    if sf is None:
        failures.append("score formula is None for 271/PSM")
    else:
        print(f"  ratio={sf.ratio}  fm_in_score={sf.fm_in_score}  source={sf.source}")
        print(f"  formula_text: {sf.formula_text}")
        if sf.ratio != 4.0:
            failures.append(f"score ratio expected 4.0, got {sf.ratio}")
        if sf.source != "_lambda_recompute":
            failures.append(f"score source expected _lambda_recompute, got {sf.source}")

    # 6 ── GIF render (flagship; CPU-only, no ffmpeg) ──────────────────────────
    _hr("6. GIF render (flagship — story climb_plateau)")
    try:
        from app.gif.service import get_gif_service  # noqa: E402

        gif = get_gif_service()
        spec = {
            "story": "climb_plateau",
            "metric_keys": ["pak_auc_f1"],
            "experiment_set": [{"exp_id": exp271.exp_id, "dataset_key": "PSM",
                                "variant": None}],
            "max_epoch": None,
            "params": {"max_frames": 20, "fps": 9},
        }
        res = gif.render_sync(spec)
        gif_path = Path(res["path"])
        if gif_path.is_file():
            data = gif_path.read_bytes()
            header = data[:6]
            ok = header in (b"GIF89a", b"GIF87a") and len(data) > 0
            print(f"  cache_key={res['cache_key']} frames={res.get('n_frames')} "
                  f"bytes={len(data)} header={header!r} valid={ok}")
            print(f"  path={gif_path}")
            if not ok:
                failures.append("rendered GIF is corrupt (bad header / empty)")
            if str(SETTINGS.cache_root) not in str(gif_path):
                failures.append("GIF written OUTSIDE UI/.cache (safety violation)")
        else:
            failures.append("GIF render produced no file")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"GIF render raised: {exc}")

    # ── health rollup ─────────────────────────────────────────────────────────
    _hr("health()")
    h = repo.health()
    print(f"  status={h['status']} resolver_selftest={h['resolver_selftest']} "
          f"registry={h['registry_version']}")

    return _finish(failures)


def _finish(failures: list[str]) -> int:
    print("\n" + "=" * 66)
    if failures:
        print(f"SMOKE RESULT: FAIL ({len(failures)} critical check(s) failed)")
        for f in failures:
            print(f"  ✗ {f}")
        return 1
    print("SMOKE RESULT: PASS — all critical checks green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
