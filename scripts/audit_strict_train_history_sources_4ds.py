"""Audit train-history metric sources for the 4 early-stopping datasets."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


BASE_SCRIPT = Path(__file__).with_name("early_stopping_train_metric_sweep_4ds.py")
spec = importlib.util.spec_from_file_location("early_stopping_base", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
assert spec.loader is not None
spec.loader.exec_module(base)

OUT_DIR = Path("temp/early_stopping_strict_train_scalar_4ds")
OUT_PATH = OUT_DIR / "history_source_audit.json"


def classify_key(key: str) -> tuple[str, str]:
    if key in {"epoch", "batch_profiling", "epoch_timings"}:
        return "exclude", "bookkeeping/timing"
    if key.startswith("epoch_"):
        return "exclude", "eval/test callback history, not train-loop input"
    if key.startswith("train_feature_"):
        return "exclude", "feature-level metric excluded by user"
    if key == "train_fm_loss" or key.startswith("train_fm_"):
        return "exclude", "feature-matching metric excluded by user"
    if key.startswith("train_grl_"):
        return "exclude", "GRL metric excluded by user"
    if key.startswith("train_scad_"):
        return "exclude", "SCAD metric excluded by user"
    if key.startswith("train_"):
        return "include", "train-loop scalar candidate"
    return "exclude", "not a train metric"


def main() -> None:
    cells = base.load_cells(Path("results/experiments"))
    by_key: dict[str, dict] = {}
    by_dataset = defaultdict(lambda: defaultdict(lambda: {"present": 0, "scalar": 0, "nonconstant": 0}))

    for cell in cells:
        history = cell["history"]
        n_epochs = len(history.get("epoch", []))
        for key, values in history.items():
            decision, reason = classify_key(key)
            rec = by_key.setdefault(
                key,
                {
                    "decision": decision,
                    "reason": reason,
                    "present_cells": 0,
                    "epoch_length_cells": 0,
                    "scalar_cells": 0,
                    "nonconstant_cells": 0,
                    "empty_list_cells": 0,
                    "non_scalar_cells": 0,
                    "example_cell": {
                        "dataset": cell["dataset"],
                        "history_path": str(cell["history_path"]),
                    },
                },
            )
            rec["present_cells"] += 1
            by_dataset[key][cell["dataset"]]["present"] += 1

            if isinstance(values, list) and len(values) == 0:
                rec["empty_list_cells"] += 1
                continue
            if not isinstance(values, list) or len(values) != n_epochs:
                rec["non_scalar_cells"] += 1
                continue

            rec["epoch_length_cells"] += 1
            series = base.to_scalar_series(values, n_epochs)
            if series is None:
                rec["non_scalar_cells"] += 1
                continue

            rec["scalar_cells"] += 1
            by_dataset[key][cell["dataset"]]["scalar"] += 1
            if np.nanstd(series) > 1e-12:
                rec["nonconstant_cells"] += 1
                by_dataset[key][cell["dataset"]]["nonconstant"] += 1

    included = sorted(
        key
        for key, rec in by_key.items()
        if rec["decision"] == "include" and rec["scalar_cells"] > 0
    )
    full_coverage_nonconstant = sorted(
        key
        for key, rec in by_key.items()
        if rec["decision"] == "include" and rec["nonconstant_cells"] == len(cells)
    )
    partial_coverage_nonconstant = sorted(
        key
        for key, rec in by_key.items()
        if rec["decision"] == "include" and 0 < rec["nonconstant_cells"] < len(cells)
    )

    payload = {
        "n_cells": len(cells),
        "datasets": list(base.DATASETS),
        "included_train_scalar_keys": included,
        "full_coverage_nonconstant_train_scalar_keys": full_coverage_nonconstant,
        "partial_coverage_nonconstant_train_scalar_keys": partial_coverage_nonconstant,
        "keys": dict(sorted(by_key.items())),
        "by_dataset": {
            key: dict(sorted(ds_map.items()))
            for key, ds_map in sorted(by_dataset.items())
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "n_cells": payload["n_cells"],
                "included_train_scalar_keys": included,
                "full_coverage_nonconstant_train_scalar_keys": full_coverage_nonconstant,
                "partial_coverage_nonconstant_train_scalar_keys": partial_coverage_nonconstant,
                "out": str(OUT_PATH),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
