"""CI gate for the F5 resolver — the registry's resolver_self_test MUST pass.

Run:  UI/.venv/bin/python -m pytest UI/backend/tests -q
(or directly:  UI/.venv/bin/python UI/backend/tests/test_resolver_selftest.py)

These mirror the backend-design.md §3.3 startup gate as a regression test, plus the
§4 score-formula "never blank" invariant on the canonical 271/PSM fixture.
"""
from __future__ import annotations

import sys
from pathlib import Path

# make `import app...` work whether run via pytest or directly.
_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.config import SETTINGS  # noqa: E402
from app.dataaccess.registry import MetricRegistry  # noqa: E402
from app.dataaccess.scoring import resolve_score_formula  # noqa: E402
from app.dataaccess.io_safe import read_json  # noqa: E402


def _registry() -> MetricRegistry:
    return MetricRegistry.load(SETTINGS.registry_path)


def test_resolver_self_test_passes():
    reg = _registry()
    result = reg.run_self_test()
    assert result["status"] == "PASS", result["failures"]


def test_explicit_entries_win_over_fallback():
    reg = _registry()
    m = reg.resolve("pak_auc_f1", namespace="epoch_metrics")
    assert m.direction == "up" and not m.inferred
    # a bare recon-quality name must NOT be force-labeled a loss (P1-06).
    rid = reg.fallback_rule_id("recon_quality_score")
    assert rid == "default_entry"


def test_unknown_key_is_inferred_not_dropped():
    reg = _registry()
    m = reg.resolve("totally_new_metric_xyz", namespace="epoch_metrics")
    assert m.inferred is True  # shown, badged inferred — never dropped (F5)


def _find_271_psm_metadata() -> Path | None:
    root = SETTINGS.fixture_root
    if not root.exists():
        return None
    for d in sorted(root.iterdir()):
        if d.is_dir() and d.name.startswith("271_") and "271canon" in d.name:
            p = d / "PSM" / "experiment_metadata.json"
            if p.exists():
                return p
    return None


def test_score_formula_271_psm_is_four_to_one_never_blank():
    p = _find_271_psm_metadata()
    if p is None:
        # fixture not present in this environment; skip rather than fail.
        return
    md = read_json(p)
    sf = resolve_score_formula(md)
    assert sf.ratio == 4.0, sf
    assert sf.source == "_lambda_recompute", sf
    assert sf.fm_in_score is False, sf


if __name__ == "__main__":
    test_resolver_self_test_passes()
    test_explicit_entries_win_over_fallback()
    test_unknown_key_is_inferred_not_dropped()
    test_score_formula_271_psm_is_four_to_one_never_blank()
    print("all tests PASS")
