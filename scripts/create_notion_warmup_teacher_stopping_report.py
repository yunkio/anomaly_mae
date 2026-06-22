"""Create a Notion report for dynamic teacher-only warm-up stopping."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from statistics import mean


sys.path.insert(0, str(Path(__file__).resolve().parent))
from refresh_notion_early_stopping_report_v2 import (  # noqa: E402
    PARENT_PAGE_ID,
    append_blocks,
    bullet,
    callout,
    clear_page,
    code_block,
    divider,
    heading,
    notion_request,
    paragraph,
    table,
    toc,
    toggle,
)


OUT = Path("temp/warmup_teacher_only_dynamic_sweep")
PAGE_ID = "38487856-b207-8141-8c53-e8cd2ead0a1b"
TITLE = "Dynamic Teacher Warm-up Stopping: train-only LASAD audit"
SCOPES = ["main4", "SMD", "MSL", "SMAP"]
SCOPE_LABELS = {
    "main4": "SWaT(excl22) / PSM / WaDi A1 / WaDi A2",
    "SMD": "SMD",
    "MSL": "MSL",
    "SMAP": "SMAP",
}
MAIN4_LABELS = {
    "SWaT_excl22": "SWaT(excl22)",
    "PSM": "PSM",
    "WaDi_A1": "WaDi A1",
    "WaDi_A2": "WaDi A2",
}
ROLLBACK = "best_seen_before_stop"

COMPARISON_VARIANTS = [
    {
        "label": "User proposal · gap-ratio peak reversal",
        "short": "Gap-ratio",
        "metric": "pair_ratio__train_teacher_recon_anomaly__train_teacher_recon_normal",
        "transform": "ema01",
        "direction_mode": "force_max",
        "rule": "peak_reversal",
        "patience": 8,
        "threshold_type": "rel",
        "threshold_value": 0.01,
        "start_policy": "epoch20",
        "description": "normal/anomaly를 함께 보고, anomaly/normal teacher recon ratio가 peak 이후 하락하면 gap이 좁혀지기 시작한 것으로 본다.",
    },
    {
        "label": "Normal-only plateau",
        "short": "Normal-only",
        "metric": "train_teacher_recon_normal",
        "transform": "ema03",
        "direction_mode": "auto",
        "rule": "standard",
        "patience": 2,
        "threshold_type": "rel",
        "threshold_value": 0.01,
        "start_policy": "epoch20",
        "description": "label-normal subset의 teacher recon loss 수렴만 본다. 성능은 강하지만 anomaly label 정보를 직접 쓰지 않는다.",
    },
    {
        "label": "Anomaly-only plateau",
        "short": "Anomaly-only",
        "metric": "train_teacher_recon_anomaly",
        "transform": "ema01",
        "direction_mode": "force_min",
        "rule": "standard",
        "patience": 3,
        "threshold_type": "abs",
        "threshold_value": 0.0,
        "start_policy": "epoch20",
        "description": "label-anomaly subset의 teacher recon loss plateau를 본다. anomaly label은 쓰지만 normal과의 상호작용은 쓰지 않는다.",
    },
    {
        "label": "Drop-strong · train_loss",
        "short": "train_loss",
        "metric": "train_loss",
        "transform": "raw",
        "direction_mode": "auto",
        "rule": "peak_reversal",
        "patience": 2,
        "threshold_type": "rel",
        "threshold_value": 0.01,
        "start_policy": "epoch20",
        "description": "전체 train loss 기준. mean drop은 강하지만 label-aware mechanism은 아니다.",
    },
    {
        "label": "Drop-strong · train_rec_loss",
        "short": "train_rec_loss",
        "metric": "train_rec_loss",
        "transform": "raw",
        "direction_mode": "auto",
        "rule": "peak_reversal",
        "patience": 2,
        "threshold_type": "rel",
        "threshold_value": 0.01,
        "start_policy": "epoch20",
        "description": "teacher reconstruction loss 기준. train_loss와 거의 같은 역할을 하며 label-aware mechanism은 아니다.",
    },
    {
        "label": "Drop-strong · train/rec ratio",
        "short": "train/rec ratio",
        "metric": "pair_ratio__train_loss__train_rec_loss",
        "transform": "ema05",
        "direction_mode": "force_min",
        "rule": "standard",
        "patience": 2,
        "threshold_type": "abs",
        "threshold_value": 0.0,
        "start_policy": "epoch20",
        "description": "drop 기준 상위권의 train-loss interaction baseline. label을 쓰지 않으므로 비교용으로 둔다.",
    },
]


def fmt(value: object, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def fmt_epoch(value: object) -> str:
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return "-"


def metric_name(row: dict) -> str:
    return str(row.get("metric_pretty") or row.get("metric") or "-")


def rule_text(row: dict) -> str:
    return (
        f"{row['transform']} / {row['direction_mode']} / {row['rule']} / "
        f"P={int(float(row['patience']))} / {row['threshold_type']}={float(row['threshold_value']):g} / "
        f"{row['start_policy']} / rollback={row.get('rollback', 'best_seen_before_stop')}"
    )


def load_csv(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            for key, value in list(row.items()):
                if key in {
                    "key",
                    "scope",
                    "metric",
                    "metric_pretty",
                    "transform",
                    "direction_mode",
                    "rule",
                    "threshold_type",
                    "rollback",
                    "start_policy",
                    "full_coverage",
                }:
                    continue
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    pass
            rows.append(row)
    return rows


def variant_key(spec: dict) -> str:
    return "|".join(
        [
            spec["metric"],
            spec["transform"],
            spec["direction_mode"],
            spec["rule"],
            str(int(spec["patience"])),
            spec["threshold_type"],
            str(float(spec["threshold_value"])),
            ROLLBACK,
            spec["start_policy"],
        ]
    )


def pretty_variant_metric(metric: str) -> str:
    if metric.startswith("pair_ratio__"):
        a, b = metric.removeprefix("pair_ratio__").split("__", 1)
        return f"ratio({a}, {b})"
    if metric.startswith("pair_relgap__"):
        a, b = metric.removeprefix("pair_relgap__").split("__", 1)
        return f"relgap({a}, {b})"
    if metric.startswith("pair_diff__"):
        a, b = metric.removeprefix("pair_diff__").split("__", 1)
        return f"diff({a}, {b})"
    if metric.startswith("pair_absdiff__"):
        a, b = metric.removeprefix("pair_absdiff__").split("__", 1)
        return f"absdiff({a}, {b})"
    return metric


def find_variant_row(leaderboards: dict[str, list[dict]], scope: str, spec: dict) -> dict:
    key = variant_key(spec)
    for row in leaderboards[scope]:
        if row["key"] == key:
            return row
    raise KeyError(f"{scope}: {key}")


def variant_rule_text(spec: dict) -> str:
    return (
        f"{spec['transform']} / {spec['direction_mode']} / {spec['rule']} / "
        f"P={int(spec['patience'])} / {spec['threshold_type']}={float(spec['threshold_value']):g} / "
        f"{spec['start_policy']}"
    )


def criterion_rows(rows: list[dict], n: int = 5, by: str = "rank") -> list[list[str]]:
    if by == "drop":
        selected = sorted(rows, key=lambda r: (float(r["mean_drop"]), -float(r["mean_score"])))[:n]
    else:
        selected = sorted(rows, key=lambda r: (float(r.get("avg_rank", 10**9)), -float(r["mean_score"])))[:n]
    out = [["#", "Criterion", "Rule", "Mean score", "Drop", "Selected epoch", "Avg rank"]]
    for idx, row in enumerate(selected, start=1):
        out.append(
            [
                str(idx),
                metric_name(row),
                rule_text(row),
                fmt(row["mean_score"]),
                fmt(row["mean_drop"]),
                fmt_epoch(row["mean_selected_epoch"]),
                fmt(row.get("avg_rank"), 1),
            ]
        )
    return out


def fixed_rows(summary: dict, scope: str) -> list[list[str]]:
    wanted = ["epoch50", "epoch100", "epoch150", "epoch200", "epoch250", "config_warmup"]
    labels = {
        "epoch50": "fixed 50",
        "epoch100": "fixed 100",
        "epoch150": "fixed 150",
        "epoch200": "fixed 200",
        "epoch250": "fixed 250",
        "config_warmup": "configured warm-up",
    }
    rows = [["Policy", "Coverage", "Mean score", "Drop vs teacher oracle"]]
    for row in summary["fixed_by_scope"][scope]:
        if row["policy"] not in wanted:
            continue
        rows.append(
            [
                labels[row["policy"]],
                str(int(row["total_n"])),
                fmt(row["mean_score"]),
                fmt(row["mean_drop"]),
            ]
        )
    return rows


def pf_scope_rows(summary: dict) -> list[list[str]]:
    rows = [["Scope", "Cells", "PF score", "Drop", "Selected epoch", "Avg rank"]]
    for scope in SCOPES:
        row = summary["paper_friendly_rows"][scope]
        rows.append(
            [
                SCOPE_LABELS[scope],
                str(int(summary["n_cells_by_scope"][scope])),
                fmt(row["mean_score"]),
                fmt(row["mean_drop"]),
                fmt_epoch(row["mean_selected_epoch"]),
                fmt(row["avg_rank"], 1),
            ]
        )
    return rows


def variant_scope_rows(summary: dict, leaderboards: dict[str, list[dict]], spec: dict) -> list[list[str]]:
    rows = [["Scope", "Cells", "Score", "Drop", "Selected epoch", "Avg rank"]]
    for scope in SCOPES:
        row = find_variant_row(leaderboards, scope, spec)
        rows.append(
            [
                SCOPE_LABELS[scope],
                str(int(summary["n_cells_by_scope"][scope])),
                fmt(row["mean_score"]),
                fmt(row["mean_drop"]),
                fmt_epoch(row["mean_selected_epoch"]),
                fmt(row["avg_rank"], 1),
            ]
        )
    return rows


def variant_summary_rows(leaderboards: dict[str, list[dict]]) -> list[list[str]]:
    rows = [["Variant", "Metric", "Rule", "Mean drop", "Mean selected epoch", "Interpretation"]]
    for spec in COMPARISON_VARIANTS:
        scope_rows = [find_variant_row(leaderboards, scope, spec) for scope in SCOPES]
        rows.append(
            [
                spec["short"],
                pretty_variant_metric(spec["metric"]),
                variant_rule_text(spec),
                fmt(mean(float(row["mean_drop"]) for row in scope_rows)),
                fmt_epoch(mean(float(row["mean_selected_epoch"]) for row in scope_rows)),
                spec["description"],
            ]
        )
    return rows


def source_rows(summary: dict) -> list[list[str]]:
    rows = [["Scope", "Cells", "Metric bank", "Teacher score rows", "Train scalar sources"]]
    for scope in SCOPES:
        audit = summary["source_audit_by_scope"][scope]
        keys = [
            key
            for key, rec in audit["keys"].items()
            if rec.get("nonconstant_warmup", 0)
        ]
        rows.append(
            [
                SCOPE_LABELS[scope],
                str(audit["n_cells"]),
                f"{audit['metric_bank_min']} / {audit['metric_bank_mean']:.1f} / {audit['metric_bank_max']}",
                f"{audit['teacher_pak_auc_f1_rows']} teacher_pak_auc_f1",
                ", ".join(keys),
            ]
        )
    return rows


def oracle_rows(summary: dict, scope: str) -> list[list[str]]:
    rows = [["Dataset/entity", "N", "Oracle score", "Oracle epoch", "Configured score", "Configured drop", "drop>0.03"]]
    by_group = summary["oracle_by_scope"][scope]["by_group"]
    for group, rec in sorted(by_group.items()):
        label = MAIN4_LABELS.get(group, group)
        rows.append(
            [
                label,
                str(int(rec["n"])),
                fmt(rec["mean_oracle_score"]),
                fmt_epoch(rec["mean_oracle_epoch"]),
                fmt(rec["mean_fixed_warmup_score"]),
                fmt(rec["mean_fixed_drop"]),
                str(int(rec["overfit_drop_gt_0_03"])),
            ]
        )
    return rows


def entity_pf_rows(summary: dict, scope: str, limit: int | None = None) -> list[list[str]]:
    pf = summary["paper_friendly_rows"][scope]
    groups = []
    for key in pf:
        if key.endswith("_mean_score"):
            group = key.removesuffix("_mean_score")
            if scope == "main4" and group not in MAIN4_LABELS:
                continue
            if scope != "main4" and not group.startswith(scope + "/"):
                continue
            groups.append(group)
    groups = sorted(groups)
    rows = [["Dataset/entity", "PF score", "Drop", "Selected epoch"]]
    for group in groups[:limit]:
        rows.append(
            [
                MAIN4_LABELS.get(group, group),
                fmt(pf[f"{group}_mean_score"]),
                fmt(pf[f"{group}_mean_drop"]),
                fmt_epoch(pf[f"{group}_mean_selected_epoch"]),
            ]
        )
    return rows


def overfit_rows(summary: dict, scope: str, limit: int = 10) -> list[list[str]]:
    rows = [["Group", "Experiment", "Oracle", "Oracle epoch", "Configured", "Configured epoch", "Drop"]]
    for item in summary["oracle_by_scope"][scope]["top_overfit_examples"][:limit]:
        rows.append(
            [
                MAIN4_LABELS.get(item["group"], item["group"]),
                item["exp"].split("_", 1)[0],
                fmt(item["oracle_score"]),
                str(int(item["oracle_epoch"])),
                fmt(item["fixed_score"]),
                str(int(item["fixed_epoch"])),
                fmt(item["drop"]),
            ]
        )
    return rows


def perf_cell(payload: dict | None) -> str:
    if not payload:
        return "-"
    return f"{payload['score']:.4f} (#{payload['rank']}, e{payload['selected_epoch']})"


def model_rank_rows(table_payload: dict, max_group_cols: int = 8) -> list[list[str]]:
    groups = table_payload["groups"][:max_group_cols]
    header = ["Avg rank", "Model", "Cov."] + [MAIN4_LABELS.get(g, g.replace("SMD/", "").replace("MSL/", "").replace("SMAP/", "")) for g in groups]
    rows = [header]
    for row in table_payload["rows"][:25]:
        avg = "-" if row["avg_rank"] is None else f"{row['avg_rank']:.2f}"
        rows.append(
            [
                avg,
                row["model"].split("_", 1)[0],
                str(row["coverage"]),
                *[perf_cell(row["groups"].get(group)) for group in groups],
            ]
        )
    return rows


def pf_pseudocode() -> str:
    return """# Dynamic teacher warm-up stopping (recommended PF)
# Inputs are train-split metrics only.
upper_bound = configured_teacher_only_warmup_epochs
min_epoch = 20
alpha = 0.1
patience_checks = 8
relative_drop_threshold = 0.01
check_interval = 5  # retrospective analysis used epoch_metrics cadence

ema = None
peak_metric = -inf
peak_epoch = None
best_state = None
num_drop_checks = 0

for epoch in range(1, upper_bound + 1):
    train_one_teacher_only_epoch()
    normal = mean_teacher_reconstruction_loss_on_labeled_normal_train_windows()
    anomaly = mean_teacher_reconstruction_loss_on_labeled_anomaly_train_windows()
    gap_ratio = anomaly / (abs(normal) + 1e-8)
    ema = gap_ratio if ema is None else alpha * gap_ratio + (1 - alpha) * ema

    if epoch < min_epoch or epoch % check_interval != 0:
        continue

    if peak_epoch is None or ema > peak_metric:
        peak_metric = ema
        peak_epoch = epoch
        best_state = snapshot(model, optimizer, scheduler)
        num_drop_checks = 0
    else:
        rel_drop = (peak_metric - ema) / max(abs(peak_metric), 1e-8)
        if rel_drop > relative_drop_threshold:
            num_drop_checks += 1
        else:
            num_drop_checks = 0

    if num_drop_checks >= patience_checks:
        restore(best_state)
        warmup_end_epoch = epoch
        selected_teacher_epoch = peak_epoch
        break
else:
    warmup_end_epoch = upper_bound
    selected_teacher_epoch = peak_epoch or upper_bound
"""


def build_blocks() -> list[dict]:
    summary = json.loads((OUT / "summary.json").read_text())
    top_tables = json.loads((OUT / "top_tables.json").read_text())
    pf_tables = json.loads((OUT / "paper_friendly_tables.json").read_text())
    leaderboards = {scope: load_csv(OUT / f"leaderboard_{scope}.csv") for scope in SCOPES}

    pf = summary["paper_friendly_spec"]
    blocks: list[dict] = [
        heading(1, TITLE),
        callout(
            [
                "결론: warm-up 종료 기준은 ",
                ("ratio(train_teacher_recon_anomaly, train_teacher_recon_normal)", "code"),
                "의 peak-reversal을 권고한다. 이 기준은 normal/anomaly label을 동시에 사용하고, 두 집단의 teacher reconstruction gap이 더 이상 벌어지지 않고 좁혀지기 직전의 teacher 상태를 선택한다.",
            ],
            "green_background",
        ),
        paragraph(
            "이 문서는 전체 early stopping과 별도로, Student가 합류하기 전 teacher-only warm-up을 동적으로 끝내기 위한 기준을 재분석한 보고서다. 성능 평가는 사후적으로 teacher-only score를 사용했지만, stopping 입력은 모두 훈련 데이터에서 수집 가능한 history metric으로 제한했다."
        ),
        toc(),
        divider(),
        heading(2, "1. 분석 범위와 입력 source"),
        table(source_rows(summary)),
        callout(
            [
                "성능 평가는 ",
                ("teacher_pak_auc_f1", "code"),
                "를 사용했다. 이것은 criterion 입력이 아니라, 선택된 warm-up epoch를 사후 평가하기 위한 teacher-only oracle 비교용 값이다.",
            ],
            "blue_background",
        ),
        bullet("사용한 train metric: train_loss, train_rec_loss, train_teacher_recon_normal, train_teacher_recon_anomaly 및 이들 사이의 ratio/diff/absdiff/relgap."),
        bullet("사용하지 않은 입력: test/eval loader에서 계산된 PRC/AUC/F1/PAK, epoch_* contribution, GRL/SCAD, feature-level metric."),
        bullet("Retrospective 계산은 epoch_metrics가 존재하는 eval point에서만 성능을 조회했다. 따라서 P=2는 현재 저장 구조에서는 대개 2개 eval point, 즉 약 10 epoch에 해당한다."),
        divider(),
        heading(2, "2. 핵심 결론"),
        table(pf_scope_rows(summary)),
        heading(3, "비교 기준별 scope table"),
        callout(
            "아래 표들은 동일한 형식으로 계산한 비교군이다. 성능은 teacher-only 사후 평가이며, stopping 입력은 모두 train history metric이다.",
            "blue_background",
        ),
        table(variant_summary_rows(leaderboards)),
        *[
            toggle(
                variant["label"],
                [
                    paragraph(
                        [
                            "Metric: ",
                            (pretty_variant_metric(variant["metric"]), "code"),
                            " / Rule: ",
                            (variant_rule_text(variant), "code"),
                        ]
                    ),
                    paragraph(variant["description"]),
                    table(variant_scope_rows(summary, leaderboards, variant)),
                ],
            )
            for variant in COMPARISON_VARIANTS
        ],
        paragraph(
            [
                "최종 권고 PF는 ",
                (summary["paper_friendly_spec"]["metric_pretty"], "code"),
                f" / {pf['transform']} / {pf['rule']} / P={pf['patience']} / {pf['threshold_type']}={pf['threshold_value']} / {pf['start_policy']}",
                " 이다. 방향은 anomaly/normal reconstruction ratio를 키우는 방향이며, peak 이후 상대 감소가 누적되면 중단한다.",
            ]
        ),
        bullet("normal-only 또는 anomaly-only plateau는 성능상 더 강할 수 있지만, normal과 anomaly의 상호작용을 쓰지 않으므로 최종 PF에서는 제외한다."),
        bullet("train_loss/train_rec_loss plateau는 전체 drop 기준으로 아주 강하지만 label을 쓰지 않기 때문에 논문형 기준으로는 보조 baseline으로만 둔다."),
        bullet("absdiff/diff/relgap도 같은 목적의 후보지만, 계산된 sweep에서는 ratio(anomaly, normal)의 peak-reversal이 strict gap-narrowing 후보 중 mean drop이 가장 작았다."),
        divider(),
        heading(2, "3. Fixed warm-up baseline 비교"),
        table(
            [["Scope", "Configured warm-up score/drop", "Best fixed among checked", "PF score/drop"]]
            + [
                [
                    SCOPE_LABELS[scope],
                    f"{fmt(next(r for r in summary['fixed_by_scope'][scope] if r['policy']=='config_warmup')['mean_score'])} / {fmt(next(r for r in summary['fixed_by_scope'][scope] if r['policy']=='config_warmup')['mean_drop'])}",
                    min(
                        [
                            r
                            for r in summary["fixed_by_scope"][scope]
                            if r["policy"] in {"epoch50", "epoch100", "epoch150", "epoch200", "epoch250"}
                        ],
                        key=lambda r: r["mean_drop"],
                    )["policy"]
                    + " · "
                    + fmt(
                        min(
                            [
                                r
                                for r in summary["fixed_by_scope"][scope]
                                if r["policy"] in {"epoch50", "epoch100", "epoch150", "epoch200", "epoch250"}
                            ],
                            key=lambda r: r["mean_drop"],
                        )["mean_drop"]
                    ),
                    f"{fmt(summary['paper_friendly_rows'][scope]['mean_score'])} / {fmt(summary['paper_friendly_rows'][scope]['mean_drop'])}",
                ]
                for scope in SCOPES
            ]
        ),
        callout(
            "SMD와 SMAP에서는 특정 fixed epoch가 평균적으로 더 강한 경우가 있다. 따라서 이 기준은 '모든 family에서 fixed를 이기는 만능 기준'이 아니라, train-only로 warm-up을 동적으로 줄이되 성능 희생을 작게 유지하는 paper-friendly 기준으로 해석해야 한다.",
            "yellow_background",
        ),
        divider(),
        heading(2, "4. Criterion sweep 결과"),
    ]

    for scope in SCOPES:
        rows = [r for r in leaderboards[scope] if str(r["full_coverage"]).lower() == "true"]
        blocks.extend(
            [
                heading(3, SCOPE_LABELS[scope]),
                toggle("평균 순위 기준 상위 5개", [table(criterion_rows(rows, by="rank"))]),
                toggle("mean drop 기준 상위 5개", [table(criterion_rows(rows, by="drop"))]),
                toggle("fixed baseline 상세", [table(fixed_rows(summary, scope))]),
                toggle("configured warm-up overfit 상위 사례", [table(overfit_rows(summary, scope))]),
                toggle("PF entity별 요약", [table(entity_pf_rows(summary, scope, limit=None))]),
            ]
        )
        if scope in pf_tables:
            blocks.append(
                toggle(
                    "PF model-level rank table",
                    [
                        callout("각 셀은 teacher-only 성능 (#scope 내 rank, selected epoch) 형식이다. epoch는 trigger가 아니라 rollback 후 선택 epoch다.", "blue_background"),
                        table(model_rank_rows(pf_tables[scope], max_group_cols=8)),
                    ],
                )
            )
        if scope in top_tables and top_tables[scope]["tables"]:
            first = top_tables[scope]["tables"][0]
            blocks.append(
                toggle(
                    "Top criterion model-level rank table",
                    [
                        callout(f"Criterion: {metric_name(first['criterion'])} / {rule_text(first['criterion'])}", "gray_background"),
                        table(model_rank_rows(first, max_group_cols=8)),
                    ],
                )
            )

    blocks.extend(
        [
            divider(),
            heading(2, "5. PF 구현 방법"),
            paragraph("아래 pseudo code는 현재 저장된 실험 데이터를 재현하는 형태다. 매 epoch train metric은 계산되지만, 이번 retrospective 평가는 5-epoch eval grid에서 성능을 조회했으므로 check_interval=5를 둔다."),
            code_block(pf_pseudocode(), "python"),
            bullet("patience_checks=8은 현재 분석 기준으로 8개 check point다. check_interval=5이면 약 40 epoch patience다."),
            bullet("trigger epoch와 selected epoch는 다르다. trigger는 patience가 찬 시점이고, selected epoch는 그 전까지 EMA metric이 가장 좋았던 checkpoint다."),
            bullet("실제 구현에서 매 epoch마다 검사하려면 동일한 시간 폭을 유지하기 위해 patience_epochs=40으로 두거나, 논문에 check_interval=5를 명시한다."),
            divider(),
            heading(2, "6. 판단"),
            callout(
                [
                    "Paper preference를 ",
                    ("ratio(train_teacher_recon_anomaly, train_teacher_recon_normal) peak-reversal", "code"),
                    "로 변경한다. 이 기준은 normal과 anomaly를 함께 보며, teacher-only warm-up이 anomaly까지 과도하게 재구성하기 시작해 분리도가 꺾이는 시점을 직접 겨냥한다.",
                ],
                "green_background",
            ),
        ]
    )
    return blocks


def create_page() -> str:
    payload = {
        "parent": {"page_id": PARENT_PAGE_ID},
        "properties": {
            "title": {
                "title": [
                    {
                        "type": "text",
                        "text": {"content": TITLE},
                    }
                ]
            }
        },
    }
    data = notion_request("POST", "/pages", payload)
    return data["id"]


def main() -> None:
    notion_request("GET", f"/pages/{PARENT_PAGE_ID}")
    page_id = PAGE_ID
    notion_request("GET", f"/pages/{page_id}")
    clear_page(page_id)
    blocks = build_blocks()
    append_blocks(page_id, blocks)
    data = notion_request("GET", f"/pages/{page_id}")
    title = data["properties"]["title"]["title"][0]["plain_text"]
    print(json.dumps({"page_id": page_id, "title": title, "blocks": len(blocks)}, indent=2))


if __name__ == "__main__":
    main()
