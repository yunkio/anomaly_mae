#!/usr/bin/env python
"""Render the protocol-separated TEP Table 3 report for 10/5-epoch runs.

Input JSON files are produced by ``scripts/TEP/build_vus_seed_axes.py`` with
``--selection final``.  The report deliberately does not read the historical
30/15-epoch Table-3 JSON or any PAK aggregation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = ROOT / "results" / "experiments" / "TEP_table3_win100_ep10_warm5"
FIXED_JSON = EXP_ROOT / "table3_vus_fixed_seed.json"
DATA_JSON = EXP_ROOT / "table3_vus_data_seed.json"
OUT = ROOT / "comparison" / "results" / "experiments" / "results_tep_10_5.md"

FOLDS = ["f_step", "f_rand", "f_ds", "f_unk"]
FOLD_LABELS = ["F-STEP", "F-RAND", "F-DS", "F-UNK"]
ROWS = [
    ("Random", "simple", "Random"),
    ("Sensor range", "simple", "Sensor range"),
    ("PCA recon.", "simple", "PCA recon."),
    ("L2-norm", "simple", "L2-norm"),
    ("NN-distance", "simple", "NN-distance"),
    ("Label-blind", "mae", "Label-blind"),
    ("w/o GRL", "mae", "w/o GRL"),
    ("Teacher-only", "mae", "Teacher-only"),
    ("LASAD (ours)", "mae", "LASAD"),
]


def load_axis(path: Path, expected_axis: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"missing VUS aggregation: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("axis") != expected_axis:
        raise RuntimeError(f"axis mismatch in {path}: {value.get('axis')}")
    if value.get("selection") != "fixed final epoch 10; no test-side epoch selection":
        raise RuntimeError(f"selection mismatch in {path}: {value.get('selection')}")
    if value.get("run_boundary_handling") != "run_reset":
        raise RuntimeError(
            f"TEP VUS must reset tolerance at run boundaries in {path}: "
            f"{value.get('run_boundary_handling')}"
        )
    validate_axis(value)
    return value


def validate_axis(axis: dict) -> None:
    issues = []
    for seed in axis.get("seeds", []):
        run = axis.get("runs", {}).get(str(seed), {})
        for _, section, key in ROWS:
            for fold in FOLDS:
                cell = run.get(section, {}).get(key, {}).get(fold, {})
                values = [cell.get("S"), cell.get("U")]
                if not all(isinstance(x, (int, float)) and np.isfinite(x) for x in values):
                    issues.append(f"seed={seed} {section}/{key}/{fold}")
                pf = cell.get("per_fault", {})
                if len(pf) != 17 or not all(np.isfinite(float(x)) for x in pf.values()):
                    issues.append(f"seed={seed} {section}/{key}/{fold}/per_fault")
    if issues:
        raise RuntimeError("incomplete 10/5 VUS axis:\n" + "\n".join(issues))


def values(axis: dict, section: str, key: str, fold: str, side: str) -> list[float]:
    return [
        float(axis["runs"][str(seed)][section][key][fold][side])
        for seed in axis["seeds"]
    ]


def average_values(axis: dict, section: str, key: str, side: str) -> list[float]:
    return [
        float(np.mean([
            axis["runs"][str(seed)][section][key][fold][side]
            for fold in FOLDS
        ]))
        for seed in axis["seeds"]
    ]


def stats(xs: list[float]) -> tuple[float, float, float, float]:
    a = np.asarray(xs, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)), float(a.min()), float(a.max())


def f4(x: float) -> str:
    return f"{x:.4f}"


def stat_cell(xs: list[float]) -> str:
    return " / ".join(f4(x) for x in stats(xs))


def header() -> tuple[str, str]:
    cols = []
    for fold in FOLD_LABELS:
        cols.extend([f"{fold} Seen", f"{fold} Unseen"])
    cols.extend(["Avg Seen", "Avg Unseen"])
    return "| Method | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)


def row_seed(axis: dict, section: str, key: str, seed: int) -> list[float]:
    run = axis["runs"][str(seed)][section][key]
    cells = []
    for fold in FOLDS:
        cells.extend([float(run[fold]["S"]), float(run[fold]["U"])])
    cells.extend([
        float(np.mean([run[fold]["S"] for fold in FOLDS])),
        float(np.mean([run[fold]["U"] for fold in FOLDS])),
    ])
    return cells


def row_mean(axis: dict, section: str, key: str) -> list[float]:
    cols = []
    for fold in FOLDS:
        cols.extend([
            float(np.mean(values(axis, section, key, fold, "S"))),
            float(np.mean(values(axis, section, key, fold, "U"))),
        ])
    cols.extend([
        float(np.mean(average_values(axis, section, key, "S"))),
        float(np.mean(average_values(axis, section, key, "U"))),
    ])
    return cols


def row_stats(axis: dict, section: str, key: str) -> list[list[float]]:
    cols = []
    for fold in FOLDS:
        cols.extend([
            values(axis, section, key, fold, "S"),
            values(axis, section, key, fold, "U"),
        ])
    cols.extend([
        average_values(axis, section, key, "S"),
        average_values(axis, section, key, "U"),
    ])
    return cols


def result_paths(axis_kind: str, seed: int) -> list[str]:
    if axis_kind == "fixed":
        root = ("results/experiments/TEP_table3_win100_ep10_warm5" if seed == 42 else
                f"results/experiments/TEP_table3_win100_ep10_warm5_s{seed}")
        return [root, "scripts/TEP/results/12_20260610_211815_tep_typegen_simple"]
    root = ("results/experiments/TEP_table3_win100_ep10_warm5" if seed == 42 else
            f"results/experiments/TEP_table3_win100_ep10_warm5_dataseed{seed}")
    data = "scripts/TEP/data" if seed == 42 else f"scripts/TEP/data_dataseed{seed}"
    simple = ("scripts/TEP/results/12_20260610_211815_tep_typegen_simple" if seed == 42 else
              f"scripts/TEP/results/simple_dataseed{seed}")
    return [root, data, simple]


def add_table(lines: list[str], axis: dict, axis_kind: str, prefix: str, title: str) -> None:
    h, sep = header()
    seeds = axis["seeds"]
    json_path = FIXED_JSON if axis_kind == "fixed" else DATA_JSON
    rel_json = json_path.relative_to(ROOT).as_posix()
    selection = ("canonical 데이터 고정 + model/random seed" if axis_kind == "fixed" else
                 "data-allocation seed + model/random seed 동시 변경")
    paper_role = ("**논문 Table 3 기입 후보**" if axis_kind == "fixed" else
                  "논문 Table 3 데이터-allocation robustness 보조표")

    lines.extend([
        f"## {title}", "",
        f"> **표 ID `{prefix}-MEAN`** | {paper_role} | {selection} 5-seed 평균 | **10/5-epoch 결과**",
        "> 선택 기준: 고정 10 epoch의 final checkpoint. test metric에 의한 epoch 선택 없음.",
        f"> 결과 디렉터리: `{rel_json}`", "", h, sep,
    ])
    for display, section, key in ROWS:
        name = f"**{display}**" if key == "LASAD" else display
        lines.append("| " + name + " | " + " | ".join(f4(x) for x in row_mean(axis, section, key)) + " |")

    lines.extend([
        "", "### mean / sample std / min / max", "",
        f"> **표 ID `{prefix}-STATS`** | `{prefix}-MEAN`의 동일 method/fold/side seed 셀을 집계한 기술통계",
        "> 셀 순서: mean / sample std / min / max. 통계표이므로 원자료 경로는 참조 결과표를 따른다.",
        "", h, sep,
    ])
    for display, section, key in ROWS:
        name = f"**{display}**" if key == "LASAD" else display
        lines.append("| " + name + " | " + " | ".join(stat_cell(x) for x in row_stats(axis, section, key)) + " |")

    lines.extend([
        "", "### LASAD - Label-blind transfer discriminants", "",
        f"> **표 ID `{prefix}-DELTA`** | 논문 Table 3의 Δunseen 및 Δgap 5-seed 통계",
        f"> 결과 디렉터리: `{rel_json}`", "",
        "| Discriminant | F-STEP | F-RAND | F-DS | F-UNK | Four-fold Avg |",
        "|---|---|---|---|---|---|",
    ])
    delta_u_by_fold = []
    delta_g_by_fold = []
    for fold in FOLDS:
        du, dg = [], []
        for seed in seeds:
            run = axis["runs"][str(seed)]
            a = run["mae"]["LASAD"][fold]
            b = run["mae"]["Label-blind"][fold]
            du.append(float(a["U"] - b["U"]))
            dg.append(float((a["S"] - b["S"]) - (a["U"] - b["U"])))
        delta_u_by_fold.append(du)
        delta_g_by_fold.append(dg)
    for label, by_fold in (("Δunseen", delta_u_by_fold), ("Δgap", delta_g_by_fold)):
        per_seed_avg = [float(np.mean([by_fold[i][j] for i in range(4)])) for j in range(len(seeds))]
        lines.append("| " + label + " | " + " | ".join(
            [stat_cell(xs) for xs in by_fold] + [stat_cell(per_seed_avg)]) + " |")

    lines.extend(["", "### Per-seed 원값", ""])
    for seed in seeds:
        paths = [rel_json] + result_paths(axis_kind, seed)
        lines.extend([
            f"> **표 ID `{prefix}-SEED-{seed}`** | 논문 Table 3의 seed {seed} 원값",
            "> 결과 디렉터리: " + " ; ".join(f"`{p}`" for p in paths),
            "", h, sep,
        ])
        for display, section, key in ROWS:
            name = f"**{display}**" if key == "LASAD" else display
            lines.append("| " + name + " | " + " | ".join(
                f4(x) for x in row_seed(axis, section, key, seed)) + " |")
        lines.append("")


def main() -> None:
    fixed = load_axis(FIXED_JSON, "fixed_model_seed")
    data = load_axis(DATA_JSON, "data_and_model_seed")
    lines = [
        "# LASAD TEP Table 3 - 10/5-epoch results", "",
        "> **문서 역할**: 논문 v22 Appendix A.3의 TEP 예외 규약(총 10 epoch, Teacher-only 5 epoch)으로 별도 실행한 Table 3 VUS-PR 결과 정본.",
        "> **30/15 결과와 분리**: `results_baseline.md`의 역사적 30/15-epoch 결과를 복사하거나 혼합하지 않는다.",
        "> **학습 조건**: LASAD, Label-blind, w/o GRL은 seed/data allocation마다 독립 학습했다. Teacher-only는 같은 LASAD final checkpoint의 teacher reconstruction-only 점수다.",
        "> **Epoch 선택**: 모든 learned condition은 epoch 10 final score를 사용하며 test-side PAK best 선택을 하지 않는다. 단순모델은 epoch가 없어 기존 완결 score를 재사용한다.",
        "> **지표**: 각 17개 fault mode의 VUS-PR를 먼저 계산하고 fold의 Seen/Unseen fault 집합에서 비가중 평균한다. VUS tolerance neighborhood는 각 960-sample run 경계에서 reset한다.",
        "", "## 문서 구성과 논문 표 인덱스", "",
        "| 논문 표/역할 | 이 문서의 표 | 상태 |",
        "|---|---|---|",
        "| Table 3 본표 | `T10-T3-VUS-FIXED-MEAN` | **논문 기입 후보**: canonical data allocation 고정, model seed 5개 평균 |",
        "| Table 3 통계 근거 | `T10-T3-VUS-FIXED-STATS`, `T10-T3-VUS-FIXED-SEED-*` | mean/std/min/max 및 seed 원값 |",
        "| Data-allocation robustness | `T10-T3-VUS-DATASEED-*` | data seed와 model seed를 함께 바꾼 보조축 |",
        "| Table A.3/A.4 | TEP 데이터 정의 절 | 논문 기입 대상 데이터·fault taxonomy |",
        "", "## 실험 완결성", "",
        "| 항목 | 값 |", "|---|---|",
        "| Unique learned runs | 108 = 9 seed/data allocations x 3 conditions x 4 folds |",
        "| Fixed-data axis | seeds {42,43,40,41,44}; LASAD/Label-blind/w/o GRL 각 4 folds |",
        "| Data+model-seed axis | seeds {42,40,41,43,44}; LASAD/Label-blind/w/o GRL 각 4 folds |",
        "| VUS JSON validation | 두 축 모두 9 methods x 5 seeds x 4 folds = 180 fold entries; non-finite 0 |",
        "", "## TEP 데이터 정의 - Table A.3/A.4", "",
        "> 결과 디렉터리: `scripts/TEP/data` ; `scripts/TEP/data_dataseed{40,41,43,44}`", "",
        "| 항목 | 값 |", "|---|---|",
        "| Variables | 52 (41 measured + 11 manipulated) |",
        "| Faults | 20 total; IDV 3/9/15 excluded; 17 used |",
        "| Folds | F-STEP, F-RAND, F-DS, F-UNK |",
        "| Train per fold | 240 fault-free + 60 seen-family faulty runs |",
        "| Test per fault | 20 faulty + 40 common fault-free runs |",
        "| Training horizon | 10 epochs total; first 5 Teacher-only |",
        "| Checkpoint used | fixed final epoch 10; no test-side selection |",
        "| VUS boundary rule | reset independently at every recorded run boundary |",
        "",
    ]
    add_table(lines, fixed, "fixed", "T10-T3-VUS-FIXED", "데이터 분할 고정 + 모델 seed 5개")
    add_table(lines, data, "data", "T10-T3-VUS-DATASEED", "데이터 분할 seed + 모델 seed 동시 변경 5개")
    OUT.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"WROTE {OUT} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
