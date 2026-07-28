"""Build the two requested TEP VUS-PR seed axes from stored score artifacts.

Axes
----
1. ``fixed_model_seed``: canonical dataset allocation is fixed; only model /
   Random-baseline seed varies over {42, 43, 40, 41, 44}.
2. ``data_and_model_seed``: dataset-allocation seed and model seed are varied
   together over {42, 40, 41, 43, 44}.

The output uses the paper's Table-3 aggregation: compute VUS-PR separately for
each of the 17 usable fault modes, then take unweighted Seen/Unseen means inside
each fold.  It deliberately does not reuse PAK values from table4_data*.json.

Historical provenance
---------------------
These score artifacts come from the existing 30/15-epoch runs and use their
historical test-PAK epoch selection after epoch 15.  They are useful for the
requested seed-axis comparison, but paper v22's 10/5-epoch protocol still
requires a separate rerun.  Historical output preserves the original
concatenated-run VUS convention; ``--selection final`` defaults to the paper's
run-boundary-reset convention.

The w/o-GRL condition is required at every seed on both axes.  The companion
``run_tep_table3_nogrl_multiseed.sh`` fills the historical 30/15-epoch roots;
this builder refuses to validate an axis while any of those cells is missing.

Usage (required repository environment):
  conda run -n dc_vis python scripts/TEP/build_vus_seed_axes.py --workers 16
  conda run -n dc_vis python scripts/TEP/build_vus_seed_axes.py --workers 16 \
    --base-root results/experiments/TEP_table3_win100_ep10_warm5 --selection final
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Avoid multiplying BLAS threads inside the process pool.
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HIST_BASE = ROOT / "results" / "experiments" / "TEP_phase2_win100_ep30"
BASE = HIST_BASE
SIMPLE_CANONICAL = ROOT / "scripts" / "TEP" / "results" / "12_20260610_211815_tep_typegen_simple"
DATA_CANONICAL = ROOT / "scripts" / "TEP" / "data"

FIXED_SEEDS = [42, 43, 40, 41, 44]
DATA_SEEDS = [42, 40, 41, 43, 44]
FOLDS = ["f_step", "f_rand", "f_ds", "f_unk"]
FOLD_SHORT = {"f_step": "fstep", "f_rand": "frand", "f_ds": "fds", "f_unk": "funk"}
SEEN = {
    "f_step": [1, 2, 4, 5, 6, 7],
    "f_rand": [8, 10, 11, 12],
    "f_ds": [13, 14],
    "f_unk": [16, 17, 18, 19, 20],
}
FAULTS = sorted({f for values in SEEN.values() for f in values})
SIMPLE_KEYS = {
    "Random": "Random",
    "Sensor range": "Sensor",
    "PCA recon.": "PCA",
    "L2-norm": "L2",
    "NN-distance": "NN",
}
SIMPLE_DIRS = {
    "Sensor range": "sensor_range",
    "PCA recon.": "pca_error",
    "L2-norm": "l2_norm",
    "NN-distance": "nn_distance",
}
MAE_KEYS = {"Label-blind": "B", "Teacher-only": "D", "LASAD": "A", "w/o GRL": "nogrl"}

OUT_FIXED = BASE / "table3_vus_fixed_seed.json"
OUT_DATA = BASE / "table3_vus_data_seed.json"
SELECTION_MODE = "historical-pak"
BOUNDARY_MODE = "legacy_concat"

_Y = None
_FAULT_INDICES = None
_FAULT_BOUNDARIES = None


def _load_json(path: Path, default):
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as fp:
        return json.load(fp)


def _atomic_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fp:
        json.dump(value, fp, indent=2, ensure_ascii=False)
        fp.write("\n")
    os.replace(tmp, path)


def _new_output(axis: str, seeds: list[int]) -> dict:
    if SELECTION_MODE == "final":
        selection = "fixed final epoch 10; no test-side epoch selection"
        warning = None
    else:
        selection = "historical 30/15-epoch artifacts; test-PAK-best epoch among epoch>15"
        warning = ("paper v22 specifies 10 total epochs / 5 Teacher-only epochs "
                   "and VUS tolerance reset at every recorded run boundary")
    return {
        "metric": "vus_pr",
        "aggregation": "per-fault-mode VUS-PR, then unweighted Seen/Unseen fold mean",
        "run_boundary_handling": BOUNDARY_MODE,
        "axis": axis,
        "seeds": seeds,
        "selection": selection,
        "paper_protocol_warning": warning,
        "w_o_grl": "independently trained for every seed on both axes",
        "runs": {},
    }


def _init_worker(data_dir: str) -> None:
    global _Y, _FAULT_INDICES, _FAULT_BOUNDARIES
    d = np.load(Path(data_dir) / "test_stream.npz")
    _Y = d["y"].astype(np.int8, copy=False)
    with (Path(data_dir) / "test_run_table.json").open(encoding="utf-8") as fp:
        run_table = json.load(fp)
    _FAULT_INDICES = {}
    _FAULT_BOUNDARIES = {}
    for fault in FAULTS:
        selected_runs = [r for r in run_table if r["fault"] in (0, fault)]
        pieces = [np.arange(r["start"], r["end"], dtype=np.int64)
                  for r in selected_runs]
        _FAULT_INDICES[fault] = np.concatenate(pieces)
        offset = 0
        boundaries = []
        for piece in pieces:
            boundaries.append((offset, offset + len(piece)))
            offset += len(piece)
        _FAULT_BOUNDARIES[fault] = boundaries


def _positive_ranges(labels: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive positive ranges without joining adjacent run records."""
    padded = np.pad(labels.astype(np.int8, copy=False), (1, 1))
    edges = np.flatnonzero(np.diff(padded))
    return [(int(start), int(end - 1)) for start, end in edges.reshape(-1, 2)]


def _vus_pr_run_reset(
    labels: np.ndarray,
    score: np.ndarray,
    run_boundaries: list[tuple[int, int]],
    *,
    window_size: int = 100,
    thresholds: int = 250,
) -> float:
    """Compute pooled VUS-PR while clipping tolerance at every run boundary.

    This is algebraically equivalent to ``vus.metricor.RangeAUC_volume_opt``
    for a single run.  It keeps that implementation's global score thresholds
    and pooled precision/recall, but constructs fuzzy labels and existence
    events independently inside each recorded TEP run.
    """
    labels = labels.astype(np.int8, copy=False)
    score = score.astype(np.float64, copy=False)
    order = np.argsort(-score, kind="stable")
    sorted_score = score[order]
    sample_indices = np.linspace(0, len(score) - 1, thresholds).astype(int)
    selected_thresholds = sorted_score[sample_indices]
    # Include every tie exactly as the reference implementation's >= mask.
    n_pred = np.searchsorted(-sorted_score, -selected_thresholds, side="right")

    original_events: list[tuple[int, int]] = []
    for run_start, run_end in run_boundaries:
        for local_start, local_end in _positive_ranges(labels[run_start:run_end]):
            original_events.append((run_start + local_start, run_start + local_end))
    if not original_events:
        return 0.0

    positives = float(labels.sum())
    average_precision = []
    for window in range(window_size + 1):
        half = window // 2
        weights = labels.astype(np.float64, copy=True)
        expanded_events: list[tuple[int, int]] = []

        for run_start, run_end in run_boundaries:
            for local_start, local_end in _positive_ranges(labels[run_start:run_end]):
                start = run_start + local_start
                end = run_start + local_end
                expanded_start = max(start - half, run_start)
                expanded_end = min(end + half, run_end - 1)
                expanded_events.append((expanded_start, expanded_end))

                if window:
                    left = np.arange(expanded_start, start)
                    if len(left):
                        weights[left] = np.maximum(
                            weights[left], np.sqrt(1.0 - (start - left) / window)
                        )
                    right = np.arange(end + 1, expanded_end + 1)
                    if len(right):
                        weights[right] = np.maximum(
                            weights[right], np.sqrt(1.0 - (right - end) / window)
                        )

        sorted_weights = weights[order]
        cumulative_tp = np.cumsum(sorted_weights)
        fuzzy_weights = weights.copy()
        fuzzy_weights[labels != 0] = 0.0
        cumulative_fuzzy = np.cumsum(fuzzy_weights[order])
        tp = cumulative_tp[n_pred - 1]
        fuzzy_tp = cumulative_fuzzy[n_pred - 1]

        event_max = np.asarray([score[start:end + 1].max()
                                for start, end in expanded_events])
        existence = np.asarray([(event_max >= threshold).sum()
                                for threshold in selected_thresholds], dtype=np.float64)
        existence_ratio = existence / len(expanded_events)
        p_new = positives + fuzzy_tp / 2.0
        recall = np.minimum(tp / p_new, 1.0)
        tpr = recall * existence_ratio
        precision = tp / n_pred

        tpr_with_origin = np.concatenate(([0.0], tpr))
        average_precision.append(float(np.dot(np.diff(tpr_with_origin), precision)))

    return float(np.mean(average_precision))


def _best_epoch(run_dir: Path) -> int:
    rows = _load_json(run_dir / "epoch_metrics.json", {}).get("epochs", [])
    if SELECTION_MODE == "final":
        if not rows:
            raise RuntimeError(f"no epoch metrics: {run_dir}")
        return int(rows[-1]["epoch"])
    eligible = [row for row in rows if row.get("epoch", 0) > 15] or rows
    if not eligible:
        raise RuntimeError(f"no epoch metrics: {run_dir}")
    return int(max(eligible, key=lambda row: row.get("pak_auc_f1", 0.0))["epoch"])


def _mae_score(run_dir: Path, scoring: str) -> np.ndarray:
    epoch = _best_epoch(run_dir)
    z = np.load(run_dir / "epoch_scores" / f"epoch_{epoch:03d}_scores.npz")
    recon = np.nan_to_num(z["teacher_recon_error"]).astype(np.float64)
    if scoring == "recon":
        return recon
    if SELECTION_MODE == "final" and "official_score" in z:
        return np.nan_to_num(z["official_score"]).astype(np.float64)
    discrepancy = np.nan_to_num(z["discrepancy_error"]).astype(np.float64)
    tz = np.load(run_dir / "best_epoch_train_scores.npz")
    train_recon = np.nan_to_num(tz["teacher_recon_error"]).astype(np.float64)
    train_discrepancy = np.nan_to_num(tz["discrepancy_error"]).astype(np.float64)
    normal = tz["point_labels"] == 0
    denom = float(train_discrepancy[normal].mean())
    ratio = float(train_recon[normal].mean() / denom) if denom != 0 else 0.0
    return recon + ratio * discrepancy


def _score_for_task(task: dict) -> np.ndarray:
    kind = task["kind"]
    if kind == "npz":
        return np.load(task["path"])["anomaly_score"].astype(np.float64)
    if kind == "random":
        rng = np.random.RandomState(int(task["seed"]))
        return rng.randint(0, 2, size=len(_Y)).astype(np.float64)
    if kind == "mae":
        return _mae_score(Path(task["path"]), task["scoring"])
    raise ValueError(kind)


def _vus_per_fault(task: dict) -> tuple[str, dict[str, float]]:
    from vus.metrics import get_metrics

    score = _score_for_task(task)
    if len(score) != len(_Y) or not np.isfinite(score).all():
        raise RuntimeError(f"invalid score for {task['id']}: len={len(score)}, expected={len(_Y)}")
    values = {}
    for fault in FAULTS:
        idx = _FAULT_INDICES[fault]
        s = score[idx]
        y = _Y[idx]
        lo, hi = float(s.min()), float(s.max())
        s = (s - lo) / (hi - lo + 1e-12)
        if BOUNDARY_MODE == "run_reset":
            values[str(fault)] = _vus_pr_run_reset(
                y, s, _FAULT_BOUNDARIES[fault], window_size=100
            )
        else:
            metrics = get_metrics(s, y, metric="vus", slidingWindow=100)
            values[str(fault)] = float(metrics.get("VUS_PR", 0.0))
    return task["id"], values


def _fold_values(per_fault: dict[str, float]) -> dict:
    out = {}
    for fold in FOLDS:
        seen = SEEN[fold]
        unseen = [f for f in FAULTS if f not in set(seen)]
        out[fold] = {
            "S": float(np.mean([per_fault[str(f)] for f in seen])),
            "U": float(np.mean([per_fault[str(f)] for f in unseen])),
            "per_fault": {str(f): float(per_fault[str(f)]) for f in FAULTS},
        }
    return out


def _canonical_per_fault(vus: dict, section: str, key: str, fold: str) -> dict[str, float]:
    short = FOLD_SHORT[fold]
    source_key = f"simple_{key}_{short}" if section == "simple" else f"{key}_{short}"
    return {str(f): float(vus[source_key][str(f)]) for f in FAULTS}


def _put(run: dict, section: str, label: str, per_fault: dict[str, float]) -> None:
    run.setdefault(section, {})[label] = _fold_values(per_fault)


def _add_discriminants(run: dict) -> None:
    a = run.get("mae", {}).get("LASAD")
    b = run.get("mae", {}).get("Label-blind")
    if not a or not b:
        return
    run["discriminant"] = {}
    for fold in FOLDS:
        d_seen = a[fold]["S"] - b[fold]["S"]
        d_unseen = a[fold]["U"] - b[fold]["U"]
        run["discriminant"][fold] = {
            "delta_seen": float(d_seen),
            "delta_unseen": float(d_unseen),
            "delta_gap": float(d_seen - d_unseen),
        }


def _seed_run(output: dict, seed: int) -> dict:
    return output["runs"].setdefault(str(seed), {"simple": {}, "mae": {}})


def _complete_entry(run: dict, section: str, label: str) -> bool:
    entry = run.get(section, {}).get(label, {})
    return all(len(entry.get(fold, {}).get("per_fault", {})) == len(FAULTS) for fold in FOLDS)


def _bootstrap_existing(fixed: dict, data: dict, canonical_vus: dict, *, include_mae: bool) -> None:
    # Deterministic canonical simple values use the already audited VUS result;
    # they are epoch-independent. Random is recalculated with an explicit seed.
    for output in (fixed, data):
        run = _seed_run(output, 42)
        for label, key in SIMPLE_KEYS.items():
            if label == "Random":
                continue
            # Fold-specific simple scores yield fold-specific per-fault values.
            run["simple"][label] = {
                fold: _fold_values(_canonical_per_fault(canonical_vus, "simple", key, fold))[fold]
                for fold in FOLDS
            }
        if include_mae:
            for label, key in MAE_KEYS.items():
                run["mae"][label] = {
                    fold: _fold_values(_canonical_per_fault(canonical_vus, "mae", key, fold))[fold]
                    for fold in FOLDS
                }
        _add_discriminants(run)

    # With the dataset fixed, deterministic simple baselines are model-seed
    # invariant.  Replicate their audited canonical results, not the random row.
    source = fixed["runs"]["42"]["simple"]
    for seed in FIXED_SEEDS:
        run = _seed_run(fixed, seed)
        for label in SIMPLE_DIRS:
            run["simple"][label] = json.loads(json.dumps(source[label]))


def _build_tasks(fixed: dict, data: dict) -> tuple[list[dict], dict[str, tuple[dict, str, str]]]:
    tasks = []
    targets = {}

    def add(task: dict, output: dict, section: str, label: str):
        run = _seed_run(output, int(task["seed"]))
        if _complete_entry(run, section, label):
            return
        tasks.append(task)
        targets[task["id"]] = (output, section, label)

    # Random is score-seed dependent but data-value independent because every
    # TEP test stream has the same labels/layout. Compute once per seed and use
    # it in both axes.
    for seed in sorted(set(FIXED_SEEDS + DATA_SEEDS)):
        task = {"id": f"random:{seed}", "kind": "random", "seed": seed}
        # Store into fixed first; data reuse is applied after task completion.
        add(task, fixed, "simple", "Random")

    # Fixed dataset, extra model seeds: A/B/w-o-GRL plus D derived from A reconstruction.
    for seed in FIXED_SEEDS:
        root = BASE if seed == 42 else Path(f"{BASE}_s{seed}")
        for fold in FOLDS:
            short = FOLD_SHORT[fold]
            for label, phase, scoring in (
                ("LASAD", "phase2_A", "tmf"),
                ("Label-blind", "phase2_B", "tmf"),
                ("Teacher-only", "phase2_A", "recon"),
                ("w/o GRL", "phase2_nogrl", "tmf"),
            ):
                task = {
                    "id": f"fixed:{seed}:{label}:{fold}",
                    "kind": "mae",
                    "seed": seed,
                    "path": str(root / phase / "TEP" / f"typegen_{short}"),
                    "scoring": scoring,
                    "fold": fold,
                }
                # A task is one fold, so use a fold-level completion check below.
                entry = _seed_run(fixed, seed).get("mae", {}).get(label, {}).get(fold, {})
                if len(entry.get("per_fault", {})) != len(FAULTS):
                    tasks.append(task)
                    targets[task["id"]] = (fixed, "mae", label)

    # Data+model seed axis, extra seeds: all four deterministic simple models.
    for seed in DATA_SEEDS:
        if seed != 42:
            simple_root = ROOT / "scripts" / "TEP" / "results" / f"simple_dataseed{seed}"
            for fold in FOLDS:
                for label, model_dir in SIMPLE_DIRS.items():
                    task = {
                        "id": f"data:{seed}:{label}:{fold}",
                        "kind": "npz",
                        "seed": seed,
                        "path": str(simple_root / fold / model_dir / "scores.npz"),
                        "fold": fold,
                    }
                    entry = _seed_run(data, seed).get("simple", {}).get(label, {}).get(fold, {})
                    if len(entry.get("per_fault", {})) != len(FAULTS):
                        tasks.append(task)
                        targets[task["id"]] = (data, "simple", label)

        mae_root = BASE if seed == 42 else Path(f"{BASE}_dataseed{seed}")
        for fold in FOLDS:
            short = FOLD_SHORT[fold]
            for label, phase, scoring in (
                ("LASAD", "phase2_A", "tmf"),
                ("Label-blind", "phase2_B", "tmf"),
                ("Teacher-only", "phase2_A", "recon"),
                ("w/o GRL", "phase2_nogrl", "tmf"),
            ):
                task = {
                    "id": f"data:{seed}:{label}:{fold}",
                    "kind": "mae",
                    "seed": seed,
                    "path": str(mae_root / phase / "TEP" / f"typegen_{short}"),
                    "scoring": scoring,
                    "fold": fold,
                }
                entry = _seed_run(data, seed).get("mae", {}).get(label, {}).get(fold, {})
                if len(entry.get("per_fault", {})) != len(FAULTS):
                    tasks.append(task)
                    targets[task["id"]] = (data, "mae", label)
    return tasks, targets


def _store_task_result(task_id: str, per_fault: dict[str, float], targets, fixed: dict, data: dict) -> None:
    output, section, label = targets[task_id]
    parts = task_id.split(":")
    if parts[0] == "random":
        seed = int(parts[1])
        folds = _fold_values(per_fault)
        _seed_run(fixed, seed)["simple"]["Random"] = folds
        _seed_run(data, seed)["simple"]["Random"] = json.loads(json.dumps(folds))
        return
    _, seed_text, _, fold = parts
    seed = int(seed_text)
    run = _seed_run(output, seed)
    run.setdefault(section, {}).setdefault(label, {})[fold] = _fold_values(per_fault)[fold]


def _validate(output: dict) -> list[str]:
    issues = []
    for seed in output["seeds"]:
        run = output["runs"].get(str(seed), {})
        expected_simple = list(SIMPLE_KEYS)
        expected_mae = ["Label-blind", "Teacher-only", "w/o GRL", "LASAD"]
        for section, labels in (("simple", expected_simple), ("mae", expected_mae)):
            for label in labels:
                entry = run.get(section, {}).get(label, {})
                for fold in FOLDS:
                    vals = entry.get(fold, {})
                    if not (isinstance(vals.get("S"), (int, float)) and
                            isinstance(vals.get("U"), (int, float)) and
                            len(vals.get("per_fault", {})) == len(FAULTS)):
                        issues.append(f"{output['axis']} seed={seed} {section}/{label}/{fold}")
        _add_discriminants(run)
    return issues


def main() -> None:
    global BASE, OUT_FIXED, OUT_DATA, SELECTION_MODE, BOUNDARY_MODE
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    ap.add_argument("--base-root", type=Path, default=HIST_BASE,
                    help="canonical experiment root; seed suffixes are inferred")
    ap.add_argument("--selection", choices=("historical-pak", "final"),
                    default="historical-pak")
    ap.add_argument(
        "--boundary-mode",
        choices=("legacy_concat", "run_reset"),
        default=None,
        help="default: legacy_concat for historical-pak, run_reset for final",
    )
    args = ap.parse_args()

    BASE = args.base_root if args.base_root.is_absolute() else ROOT / args.base_root
    OUT_FIXED = BASE / "table3_vus_fixed_seed.json"
    OUT_DATA = BASE / "table3_vus_data_seed.json"
    SELECTION_MODE = args.selection
    BOUNDARY_MODE = args.boundary_mode or (
        "run_reset" if SELECTION_MODE == "final" else "legacy_concat"
    )

    fresh_fixed = _new_output("fixed_model_seed", FIXED_SEEDS)
    fresh_data = _new_output("data_and_model_seed", DATA_SEEDS)
    fixed = _load_json(OUT_FIXED, fresh_fixed)
    data = _load_json(OUT_DATA, fresh_data)
    # Never mix cached cells produced with a different boundary convention.
    if fixed.get("run_boundary_handling") not in (None, BOUNDARY_MODE):
        fixed = fresh_fixed
    if data.get("run_boundary_handling") not in (None, BOUNDARY_MODE):
        data = fresh_data
    for output, fresh in ((fixed, fresh_fixed), (data, fresh_data)):
        for key in ("metric", "aggregation", "run_boundary_handling",
                    "axis", "seeds", "selection",
                    "paper_protocol_warning", "w_o_grl"):
            output[key] = fresh[key]
    canonical_vus = _load_json(HIST_BASE / "vus_results.json", {})
    _bootstrap_existing(fixed, data, canonical_vus,
                        include_mae=(SELECTION_MODE == "historical-pak"))
    tasks, targets = _build_tasks(fixed, data)
    _atomic_json(OUT_FIXED, fixed)
    _atomic_json(OUT_DATA, data)
    print(f"pending score jobs: {len(tasks)} | workers: {args.workers}", flush=True)

    if tasks:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=_init_worker,
                                 initargs=(str(DATA_CANONICAL),)) as pool:
            future_to_task = {pool.submit(_vus_per_fault, task): task for task in tasks}
            done = 0
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                task_id, per_fault = future.result()
                _store_task_result(task_id, per_fault, targets, fixed, data)
                done += 1
                if done % 5 == 0 or done == len(tasks):
                    _atomic_json(OUT_FIXED, fixed)
                    _atomic_json(OUT_DATA, data)
                    print(f"completed {done}/{len(tasks)}", flush=True)

    fixed_issues = _validate(fixed)
    data_issues = _validate(data)
    _atomic_json(OUT_FIXED, fixed)
    _atomic_json(OUT_DATA, data)
    if fixed_issues or data_issues:
        raise RuntimeError("incomplete VUS axes:\n" + "\n".join(fixed_issues + data_issues))
    print(f"WROTE {OUT_FIXED}", flush=True)
    print(f"WROTE {OUT_DATA}", flush=True)
    print("VUS seed-axis validation: complete", flush=True)


if __name__ == "__main__":
    main()
