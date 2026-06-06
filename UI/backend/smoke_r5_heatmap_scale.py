#!/usr/bin/env python3
"""FB-R5-01 smoke — signal_lives feature_heatmap faithful color scale (CPU-only, RO).

Verifies the FB-R5-01 fix WITHOUT a long-running server:
  1. GET /gif/stories advertises ``signal_lives_heatmap_scales`` (linear/sqrt/log, default
     sqrt) — never a hard-coded literal.
  2. render_sync the signal_lives feature_heatmap at scale=sqrt and scale=linear; BOTH are
     valid GIF89a, BOTH land under UI/.cache (safety), and the BYTES DIFFER (sqrt
     re-maps the heavy-tailed low-mid features → different pixels than linear).
  3. the sqrt frame is BRIGHTER on average in the low-mid range than linear (the wash-out
     fix): we render ONE frame from the same matrix via the renderer's own norm and assert
     the mean luminance of the sqrt frame > linear frame (the low-end is lifted).
  4. render at scale=log too (LogNorm vmin floored >0) — valid GIF89a (no crash on a
     near-zero floor).

Run:  UI/.venv/bin/python UI/backend/smoke_r5_heatmap_scale.py
Exit 0 = PASS.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))  # UI/backend

import numpy as np  # noqa: E402

from app.config import SETTINGS  # noqa: E402
from app.dataaccess.repository import get_repository  # noqa: E402
from app.gif import render  # noqa: E402
from app.gif.service import get_gif_service  # noqa: E402


def _hr(t: str) -> None:
    print(f"\n==== {t} {'=' * max(0, 56 - len(t))}")


def _valid_gif(b: bytes) -> bool:
    return len(b) > 0 and b[:6] in (b"GIF89a", b"GIF87a")


def _find_heatmap_leaf(repo):
    """Find a leaf whose epoch_metrics carry the per-feature recon array key so the
    feature_heatmap actually has data (any source; prefer the 271 fixture)."""
    for exp in sorted(repo.experiments(), key=lambda e: (e.source != "fixture", e.exp_id)):
        for ds in exp.datasets:
            _, _, leaf = repo.resolve_leaf(exp.exp_id, ds.dataset_key, None)
            if leaf is None:
                continue
            try:
                b = repo.load_bundle(exp, ds, leaf, want={"epoch"})
            except Exception:
                continue
            if b.epoch is not None and "_train_feature_recon_mean" in b.epoch.array_keys:
                return exp, ds
    return None, None


def main() -> int:
    fail: list[str] = []
    repo = get_repository()
    gif = get_gif_service()

    # 1 ── stories advertises the scales ──────────────────────────────────────
    _hr("1. /gif/stories advertises signal_lives heatmap scales")
    st = gif.stories()
    adv = st.get("signal_lives_heatmap_scales")
    print(f"  signal_lives_heatmap_scales = {adv}")
    if not adv:
        fail.append("stories() omits signal_lives_heatmap_scales")
    else:
        if set(adv.get("scales", [])) != {"linear", "sqrt", "log"}:
            fail.append(f"unexpected scales advertised: {adv.get('scales')}")
        if adv.get("default") != "sqrt":
            fail.append(f"default scale expected sqrt, got {adv.get('default')}")
        if adv.get("sub_mode") != "feature_heatmap":
            fail.append("advertised sub_mode is not feature_heatmap")
        if adv.get("param") != "heatmap_scale":
            fail.append("advertised param is not heatmap_scale")

    # locate a leaf with the per-feature recon array
    _hr("2. locate a feature_heatmap-capable leaf")
    exp, ds = _find_heatmap_leaf(repo)
    if exp is None:
        fail.append("no leaf with _train_feature_recon_mean array key found")
        return _finish(fail)
    print(f"  leaf: {exp.exp_id} / {ds.dataset_key}  (src={exp.source})")

    def _spec(scale: str | None) -> dict:
        params = {"sub_mode": "feature_heatmap", "max_frames": 12, "fps": 8}
        if scale is not None:
            params["heatmap_scale"] = scale
        return {
            "story": "signal_lives",
            "metric_keys": ["_train_feature_recon_mean"],
            "experiment_set": [{"exp_id": exp.exp_id, "dataset_key": ds.dataset_key,
                                "variant": None}],
            "max_epoch": None,
            "params": params,
        }

    # 3 ── render sqrt vs linear vs log; assert valid + differ + cache-safe ────
    _hr("3. render sqrt / linear / log (valid GIF89a, sqrt≠linear, cache-safe)")
    bytes_by_scale: dict[str, bytes] = {}
    keys_by_scale: dict[str, str] = {}
    for scale in ("sqrt", "linear", "log"):
        res = gif.render_sync(_spec(scale))
        p = Path(res["path"])
        if not p.is_file():
            fail.append(f"scale={scale}: no GIF file produced ({res})")
            continue
        data = p.read_bytes()
        bytes_by_scale[scale] = data
        keys_by_scale[scale] = res["cache_key"]
        ok = _valid_gif(data)
        in_cache = str(SETTINGS.cache_root) in str(p)
        print(f"  scale={scale:6s} key={res['cache_key']} frames={res.get('n_frames')} "
              f"bytes={len(data)} valid_gif={ok} in_cache={in_cache}")
        if not ok:
            fail.append(f"scale={scale}: rendered GIF is not valid GIF89a")
        if not in_cache:
            fail.append(f"scale={scale}: GIF written OUTSIDE UI/.cache (safety violation)")

    # each scale must cache under a DISTINCT key (params ∈ cache key)
    if len(set(keys_by_scale.values())) != len(keys_by_scale):
        fail.append(f"scales did NOT produce distinct cache keys: {keys_by_scale}")
    # sqrt vs linear bytes MUST differ (the whole point of FB-R5-01)
    if ("sqrt" in bytes_by_scale and "linear" in bytes_by_scale
            and bytes_by_scale["sqrt"] == bytes_by_scale["linear"]):
        fail.append("sqrt and linear GIFs are byte-identical (norm not applied)")
    else:
        print("  sqrt vs linear bytes DIFFER ✓ (the non-linear norm changed the pixels)")

    # 4 ── direct renderer luminance check (sqrt brightens the low-mid features) ─
    _hr("4. sqrt brightens the low-mid features vs linear (renderer luminance)")
    _, _, leaf = repo.resolve_leaf(exp.exp_id, ds.dataset_key, None)
    b = repo.load_bundle(exp, ds, leaf, want={"epoch"})
    from app.services.arrays import per_feature
    pf = per_feature(repo, b, "_train_feature_recon_mean")
    if not pf.get("available"):
        fail.append("per_feature unavailable for the chosen leaf")
        return _finish(fail)

    def _last_frame(scale: str) -> np.ndarray:
        frames = render.render_feature_heatmap(
            title="t", epochs=pf["epochs"], matrix=pf["matrix"],
            warmup_epochs=pf["warmup_epochs"], mask_prewarmup=pf["mask_prewarmup"],
            max_frames=12, hold=0, heatmap_scale=scale)
        return frames[-1].astype(float)

    fr_lin = _last_frame("linear")
    fr_sqrt = _last_frame("sqrt")
    # mean luminance over the whole frame (incl. axes/colorbar — those are scale-invariant,
    # so any delta comes from the heatmap pixels). sqrt should be >= linear (low-end lifted).
    lum_lin = fr_lin.mean()
    lum_sqrt = fr_sqrt.mean()
    print(f"  mean luminance: linear={lum_lin:.2f}  sqrt={lum_sqrt:.2f}  Δ={lum_sqrt - lum_lin:+.2f}")
    if not (lum_sqrt > lum_lin):
        fail.append(f"sqrt did not brighten the heatmap (linear={lum_lin:.2f} sqrt={lum_sqrt:.2f})")
    else:
        print("  sqrt mean luminance > linear ✓ (the low-mid features are lifted off the floor)")

    return _finish(fail)


def _finish(fail: list[str]) -> int:
    print("\n" + "=" * 60)
    if fail:
        print(f"R5 SMOKE: FAIL ({len(fail)} check(s))")
        for f in fail:
            print(f"  ✗ {f}")
        return 1
    print("R5 SMOKE: PASS — feature_heatmap faithful color scale verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
