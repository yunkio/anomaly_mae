"""CSV export helpers — backend-design.md §5.11. Pure formatting over computed data."""
from __future__ import annotations

import csv
import io
from typing import Any


def series_to_csv(epochs: list[int], series: dict[str, list]) -> str:
    """Long-axis series -> CSV: epoch,<key1>,<key2>,..."""
    out = io.StringIO()
    w = csv.writer(out)
    keys = list(series.keys())
    w.writerow(["epoch", *keys])
    for i, ep in enumerate(epochs):
        row = [ep]
        for k in keys:
            vals = series[k]
            row.append(vals[i] if i < len(vals) else "")
        w.writerow(row)
    return out.getvalue()


def matrix_to_csv(rows: list[dict[str, Any]], metrics: list[str]) -> str:
    """compare/matrix rows -> CSV: leaf_id,<metric1>,<metric2>,..."""
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["leaf_id", "exp_id", "dataset_key", "variant", *metrics])
    for r in rows:
        cells = r.get("cells", {})
        w.writerow([r.get("leaf_id"), r.get("exp_id"), r.get("dataset_key"),
                    r.get("variant"),
                    *[cells.get(m, {}).get("value", "") for m in metrics]])
    return out.getvalue()
