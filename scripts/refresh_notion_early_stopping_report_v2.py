"""Replace the Notion early-stopping report with a richer v2 report."""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable


PAGE_ID = "38487856-b207-816d-ac6b-c835eb0126cc"
PARENT_PAGE_ID = "31687856b20780e29fbcd961d69773ea"
NOTION_VERSION = "2022-06-28"
TOKEN_ENV = Path("/home/ykio/.codex/notion-api-token.env")
NONFEATURE_DIR = Path("temp/early_stopping_nonfeature_interaction_4ds")
FULL_SWEEP_DIR = Path("temp/early_stopping_train_metrics_4ds")
TOP5_TABLES = NONFEATURE_DIR / "top5_model_rank_tables.json"
LEADERBOARD = NONFEATURE_DIR / "leaderboard_post_warmup_full_coverage.csv"


TEXT_COLS = {
    "key",
    "metric",
    "transform",
    "direction_mode",
    "rule",
    "threshold_type",
    "rollback",
    "start_policy",
    "full_coverage",
}


def load_token() -> str:
    token = os.environ.get("NOTION_TOKEN")
    if token:
        return token
    if TOKEN_ENV.exists():
        for line in TOKEN_ENV.read_text().splitlines():
            if line.startswith("NOTION_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("NOTION_TOKEN is not set")


def notion_request(method: str, path: str, payload: dict | None = None) -> dict:
    token = load_token()
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"https://api.notion.com/v1{path}",
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Notion-Version": NOTION_VERSION,
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Notion API {method} {path} failed: {exc.code} {body}") from exc


def retrieve_children(block_id: str) -> list[dict]:
    children = []
    cursor = None
    while True:
        query = "?page_size=100"
        if cursor:
            query += "&start_cursor=" + urllib.parse.quote(cursor)
        data = notion_request("GET", f"/blocks/{block_id}/children{query}")
        children.extend(data.get("results", []))
        if not data.get("has_more"):
            return children
        cursor = data.get("next_cursor")


def clear_page(page_id: str) -> None:
    for child in retrieve_children(page_id):
        notion_request("DELETE", f"/blocks/{child['id']}")
        time.sleep(0.03)


def rt(content: str, *, bold: bool = False, code: bool = False, color: str = "default") -> dict:
    return {
        "type": "text",
        "text": {"content": content},
        "annotations": {
            "bold": bold,
            "italic": False,
            "strikethrough": False,
            "underline": False,
            "code": code,
            "color": color,
        },
    }


def rich(parts: Iterable[str | tuple[str, str]]) -> list[dict]:
    out = []
    for part in parts:
        if isinstance(part, tuple):
            text, style = part
            out.append(rt(text, bold=style == "bold", code=style == "code"))
        else:
            out.append(rt(part))
    return out


def paragraph(parts: Iterable[str | tuple[str, str]] | str) -> dict:
    if isinstance(parts, str):
        parts = [parts]
    return {"type": "paragraph", "paragraph": {"rich_text": rich(parts), "color": "default"}}


def heading(level: int, text: str) -> dict:
    key = f"heading_{level}"
    return {"type": key, key: {"rich_text": [rt(text)], "color": "default", "is_toggleable": False}}


def bullet(parts: Iterable[str | tuple[str, str]] | str) -> dict:
    if isinstance(parts, str):
        parts = [parts]
    return {"type": "bulleted_list_item", "bulleted_list_item": {"rich_text": rich(parts), "color": "default"}}


def numbered(parts: Iterable[str | tuple[str, str]] | str) -> dict:
    if isinstance(parts, str):
        parts = [parts]
    return {"type": "numbered_list_item", "numbered_list_item": {"rich_text": rich(parts), "color": "default"}}


def divider() -> dict:
    return {"type": "divider", "divider": {}}


def toc() -> dict:
    return {"type": "table_of_contents", "table_of_contents": {"color": "default"}}


def callout(parts: Iterable[str | tuple[str, str]] | str, color: str = "gray_background") -> dict:
    if isinstance(parts, str):
        parts = [parts]
    return {"type": "callout", "callout": {"rich_text": rich(parts), "color": color}}


def code_block(text: str, language: str = "bash") -> dict:
    return {"type": "code", "code": {"rich_text": [rt(text)], "language": language}}


def toggle(title: str, children: list[dict]) -> dict:
    return {
        "type": "toggle",
        "toggle": {
            "rich_text": [rt(title, bold=True)],
            "color": "default",
            "children": children,
        },
    }


def table(rows: list[list[str]], *, header: bool = True) -> dict:
    width = len(rows[0])
    return {
        "type": "table",
        "table": {
            "table_width": width,
            "has_column_header": header,
            "has_row_header": False,
            "children": [
                {
                    "type": "table_row",
                    "table_row": {
                        "cells": [
                            [rt(cell, bold=header and row_idx == 0)]
                            for cell in row
                        ]
                    },
                }
                for row_idx, row in enumerate(rows)
            ],
        },
    }


def load_csv(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            for key, value in list(row.items()):
                if key not in TEXT_COLS:
                    try:
                        row[key] = float(value)
                    except (TypeError, ValueError):
                        pass
            rows.append(row)
    return rows


def metric_matches(metric: str, mode: str, pattern: str) -> bool:
    if mode == "contains":
        return pattern in metric
    if mode == "prefix":
        return metric.startswith(pattern)
    raise ValueError(mode)


def metric_coverage_audit(full_rows: list[dict]) -> list[list[str]]:
    specs = [
        ("전체 discrepancy", "contains", "train_disc_loss"),
        ("anomaly discrepancy", "contains", "train_anomaly_loss"),
        ("teacher recon normal/anomaly", "contains", "train_teacher_recon_"),
        ("student recon normal/anomaly", "contains", "train_student_recon_"),
        ("disc_snr alias", "prefix", "disc_snr__"),
        ("disc_total_normal_snr alias", "prefix", "disc_total_normal_snr"),
        ("teacher_recon_snr alias", "prefix", "teacher_recon_snr"),
        ("student_recon_snr alias", "prefix", "student_recon_snr"),
    ]
    stats = {
        label: {"all": 0, "full": 0, "max_n": 0, "best_full": None, "best_all": None}
        for label, _, _ in specs
    }

    for label, mode, pattern in specs:
        matches = [row for row in full_rows if metric_matches(str(row["metric"]), mode, pattern)]
        stats[label]["full"] = len(matches)
        if matches:
            stats[label]["best_full"] = max(matches, key=lambda r: float(r["mean_score_4ds"]))

    all_path = NONFEATURE_DIR / "leaderboard_all_coverage.csv"
    with all_path.open(newline="") as f:
        for row in csv.DictReader(f):
            metric = row["metric"]
            for label, mode, pattern in specs:
                if not metric_matches(metric, mode, pattern):
                    continue
                st = stats[label]
                st["all"] += 1
                try:
                    n = int(float(row.get("total_n", 0) or 0))
                except ValueError:
                    n = 0
                st["max_n"] = max(st["max_n"], n)
                if st["best_all"] is None or float(row["mean_score_4ds"]) > float(st["best_all"]["mean_score_4ds"]):
                    st["best_all"] = row

    out = [["Requested family", "All-coverage candidates", "Strict full-coverage candidates", "Max cells", "Best strict-full mean"]]
    for label, _, _ in specs:
        st = stats[label]
        best = st["best_full"]
        if best is None:
            best_text = "none"
        else:
            best_text = f"{fmt(best['mean_score_4ds'])} · {best['transform']} · {best['rule']} · P={int(best['patience'])}"
        out.append([label, str(st["all"]), str(st["full"]), f"{st['max_n']}/177", best_text])
    return out


def source_availability_audit() -> tuple[list[list[str]], list[list[str]], dict]:
    base_script = Path(__file__).with_name("early_stopping_train_metric_sweep_4ds.py")
    spec = importlib.util.spec_from_file_location("early_stopping_base_for_report", base_script)
    base = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = base
    assert spec.loader is not None
    spec.loader.exec_module(base)

    cells = base.load_cells(Path("results/experiments"))
    counts = {
        "cells": len(cells),
        "train_histories": 0,
        "epoch_metrics_disc_snr": 0,
        "epoch_metrics_recon_snr": 0,
        "best_epoch_train_scores": 0,
        "epoch_scores_dirs": 0,
        "train_epoch_scores_dirs": 0,
        "train_anomaly_loss_nonconstant": 0,
    }
    for cell in cells:
        hist_path = Path(cell["history_path"])
        cell_dir = hist_path.parent
        if hist_path.exists():
            counts["train_histories"] += 1
            try:
                raw = json.loads(hist_path.read_text())
                hist = raw[next(iter(raw))]
                vals = hist.get("train_anomaly_loss")
                if isinstance(vals, list):
                    finite = [float(v) for v in vals if isinstance(v, (int, float))]
                    if finite:
                        mu = sum(finite) / len(finite)
                        var = sum((v - mu) ** 2 for v in finite) / len(finite)
                        if var ** 0.5 > 1e-12:
                            counts["train_anomaly_loss_nonconstant"] += 1
            except Exception:
                pass
        score_path = Path(cell["score_path"])
        try:
            rows = json.loads(score_path.read_text()).get("epochs", [])
        except Exception:
            rows = []
        if any("disc_snr" in row for row in rows):
            counts["epoch_metrics_disc_snr"] += 1
        if any("recon_snr" in row for row in rows):
            counts["epoch_metrics_recon_snr"] += 1
        if (cell_dir / "best_epoch_train_scores.npz").exists():
            counts["best_epoch_train_scores"] += 1
        if (cell_dir / "epoch_scores").is_dir():
            counts["epoch_scores_dirs"] += 1
        if (cell_dir / "train_epoch_scores").is_dir():
            counts["train_epoch_scores_dirs"] += 1

    n = counts["cells"]
    availability = [
        ["Artifact / source", "Exists in 4-dataset cells", "What it actually contains", "Use for final ES?"],
        [
            "training_histories.json",
            f"{counts['train_histories']}/{n}",
            "train loop에서 누적된 scalar train metrics",
            "yes, already used",
        ],
        [
            "epoch_metrics.json: disc_snr/recon_snr",
            f"{counts['epoch_metrics_disc_snr']}/{n}, {counts['epoch_metrics_recon_snr']}/{n}",
            "현재 저장값은 test/eval loader 기준 effect-size diagnostic",
            "not directly; recompute on train data",
        ],
        [
            "epoch_scores/*.npz",
            f"{counts['epoch_scores_dirs']}/{n}",
            "현재 저장값은 test/eval point score arrays",
            "not directly; train equivalent needed",
        ],
        [
            "best_epoch_train_scores.npz",
            f"{counts['best_epoch_train_scores']}/{n}",
            "train point scores, but best epoch only",
            "not enough for early stopping sweep",
        ],
        [
            "train_epoch_scores/*.npz",
            f"{counts['train_epoch_scores_dirs']}/{n}",
            "per-eval epoch train point scores",
            "required but currently absent",
        ],
        [
            "train_anomaly_loss",
            f"{counts['train_anomaly_loss_nonconstant']}/{n} non-constant",
            "OD anomaly loss; GRL/SCAD runs usually disable it",
            "usable where non-constant, not full coverage",
        ],
    ]

    train_computable = [
        ["Metric family", "Source formula", "Train-computable?", "Current status"],
        [
            "disc_snr",
            "(disc_anomaly - disc_normal) / (std_anomaly + std_normal + eps)",
            "yes: train labels + train discrepancy scores",
            "stored only for eval split; must collect train_epoch_metrics",
        ],
        [
            "recon_snr",
            "(recon_anomaly - recon_normal) / (std_anomaly + std_normal + eps)",
            "yes: train labels + train teacher recon scores",
            "stored only for eval split; must collect train_epoch_metrics",
        ],
        [
            "score distribution diagnostics",
            "mean/std/quantile/SNR/relgap/ratio of adaptive/recon/disc score grouped by train label",
            "yes",
            "not fully collected per epoch",
        ],
        [
            "PRC/AUC/F1/PAK/Aff/RF1/VUS",
            "ranking/threshold performance metrics",
            "excluded by design",
            "do not use as early-stopping inputs, even on train split",
        ],
        [
            "train loss/recon/disc scalar history",
            "trainer history fields",
            "yes",
            "already swept as partial evidence",
        ],
    ]
    return availability, train_computable, counts


def find_row(rows: list[dict], **criteria: object) -> dict:
    for row in rows:
        if all(row.get(key) == value for key, value in criteria.items()):
            return row
    raise KeyError(criteria)


def fmt(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def fmt_epoch(value: float) -> str:
    return f"{float(value):.1f}"


def pretty_metric(metric: str) -> str:
    if metric.startswith("pair_ratio__"):
        a, b = metric.removeprefix("pair_ratio__").split("__", 1)
        return f"ratio({a}, {b})"
    if metric.startswith("pair_relgap__"):
        a, b = metric.removeprefix("pair_relgap__").split("__", 1)
        return f"relgap({a}, {b})"
    if metric.startswith("pair_diff__"):
        a, b = metric.removeprefix("pair_diff__").split("__", 1)
        return f"diff({a}, {b})"
    return metric


def rule_text(c: dict) -> str:
    return (
        f"{c['transform']} / {c['direction_mode']} / {c['rule']} / "
        f"P={int(c['patience'])} / {c['threshold_type']}={float(c['threshold_value']):g} / "
        f"{c['start_policy']}"
    )


def perf_cell(value: dict | None) -> str:
    if not value:
        return "-"
    return f"{value['score']:.4f} (#{value['rank']}, e{value['epoch']})"


def model_rank_table_rows(table_payload: dict) -> list[list[str]]:
    out = [["Avg rank", "Model", "Cov.", "SWaT(excl22)", "PSM", "WaDi A1", "WaDi A2"]]
    for row in table_payload["rows"]:
        avg_rank = "-" if row["avg_rank"] is None else f"{row['avg_rank']:.2f}"
        ds = row["datasets"]
        out.append(
            [
                avg_rank,
                row["model"],
                str(row["coverage"]),
                perf_cell(ds.get("SWaT_excl22")),
                perf_cell(ds.get("PSM")),
                perf_cell(ds.get("WaDi_A1")),
                perf_cell(ds.get("WaDi_A2")),
            ]
        )
    return out


def criterion_summary_row(index: int | str, c: dict) -> list[str]:
    label = f"C{index}" if isinstance(index, int) else str(index)
    return [
        label,
        pretty_metric(c["metric"]),
        rule_text(c),
        fmt(c["mean_score_4ds"]),
        fmt(c["mean_drop_4ds"]),
        fmt_epoch(c["mean_stop_epoch_4ds"]),
    ]


def build_blocks() -> list[dict]:
    top5 = json.loads(TOP5_TABLES.read_text())
    nonfeature_summary = json.loads((NONFEATURE_DIR / "summary.json").read_text())
    full_summary = json.loads((FULL_SWEEP_DIR / "summary.json").read_text())
    analysis_summary = json.loads((FULL_SWEEP_DIR / "analysis_summary.json").read_text())
    leaderboard = load_csv(LEADERBOARD)
    metric_audit = metric_coverage_audit(leaderboard)
    source_availability, train_computable, source_counts = source_availability_audit()

    primary_p3 = find_row(
        leaderboard,
        metric="pair_ratio__train_loss__train_normal_loss",
        transform="ema03",
        direction_mode="force_min",
        rule="standard",
        patience=3.0,
        threshold_type="rel",
        threshold_value=0.005,
        start_policy="post_warmup",
    )
    primary_reversal = find_row(
        leaderboard,
        metric="pair_ratio__train_loss__train_normal_loss",
        transform="ema03",
        direction_mode="force_min",
        rule="peak_reversal",
        patience=3.0,
        threshold_type="rel",
        threshold_value=0.005,
        start_policy="post_warmup",
    )
    fixed = {row["policy"]: row for row in analysis_summary["fixed_epoch_baselines"]}
    oracle = full_summary["mean_oracle_by_dataset"]
    cells = nonfeature_summary["n_cells_by_dataset"]
    warmups = nonfeature_summary["warmup_distribution"]

    top5_summary = [
        ["Rank", "Criterion", "Rule", "Mean score", "Drop", "Mean stop"]
    ] + [
        criterion_summary_row(i, criterion)
        for i, criterion in enumerate(top5["criteria"], start=1)
    ]
    top5_summary.append(criterion_summary_row("PF", top5["paper_friendly_criterion"]))

    performance_toggles = []
    for table_payload in top5["tables"]:
        c = table_payload["criterion"]
        title = (
            f"C{table_payload['index']} · {pretty_metric(c['metric'])} · "
            f"mean={fmt(c['mean_score_4ds'])}, drop={fmt(c['mean_drop_4ds'])}"
        )
        children = [
            callout(
                [
                    "Rule: ",
                    (rule_text(c), "code"),
                    ". 각 셀은 ",
                    ("성능 (#순위, eepoch)", "code"),
                    " 형식이며, rank는 같은 데이터셋 내 numeric experiment 사이의 내림차순 성능 순위다.",
                ],
                "blue_background",
            ),
            table(model_rank_table_rows(table_payload)),
        ]
        performance_toggles.append(toggle(title, children))
    pf_payload = top5["paper_friendly_table"]
    pf_c = pf_payload["criterion"]
    pf_ema_tables = top5.get("paper_friendly_ema_tables", [])
    pf_ablation_tables = top5.get("paper_friendly_ablation_tables", [])
    pf_ema_summary = [
        ["Transform", "Alpha", "Mean score", "Drop", "Mean stop", "Top model"],
    ]
    for item in pf_ema_tables:
        c = item["criterion"]
        transform = c["transform"]
        alpha = {
            "raw": "none",
            "ema01": "0.1",
            "ema02": "0.2",
            "ema03": "0.3",
            "ema05": "0.5",
            "ema07": "0.7",
        }.get(transform, transform)
        pf_ema_summary.append(
            [
                transform,
                alpha,
                fmt(c["mean_score_4ds"]),
                fmt(c["mean_drop_4ds"]),
                fmt_epoch(c["mean_stop_epoch_4ds"]),
                item["rows"][0]["model"],
            ]
        )
    pf_ablation_summary = [
        ["Variant", "EMA", "Patience", "Mean score", "Drop", "Mean stop", "Top model"],
        [
            "PF",
            "on",
            f"P={pf_c['patience']}",
            fmt(pf_c["mean_score_4ds"]),
            fmt(pf_c["mean_drop_4ds"]),
            fmt_epoch(pf_c["mean_stop_epoch_4ds"]),
            pf_payload["rows"][0]["model"],
        ],
    ]
    for ablation in pf_ablation_tables:
        c = ablation["criterion"]
        pf_ablation_summary.append(
            [
                str(ablation["index"]),
                "off" if c["transform"] == "raw" else "on",
                "off; immediate first non-improvement" if int(c["patience"]) == 1 else f"P={c['patience']}",
                fmt(c["mean_score_4ds"]),
                fmt(c["mean_drop_4ds"]),
                fmt_epoch(c["mean_stop_epoch_4ds"]),
                ablation["rows"][0]["model"],
            ]
        )
    fixed_policy_order = [
        "epoch50",
        "epoch100",
        "epoch150",
        "epoch200",
        "epoch250",
        "epoch300",
        "epoch350",
        "epoch400",
        "epoch450",
        "epoch500",
        "warmup",
    ]
    fixed_rows = [["Policy", "4-dataset mean score", "Mean drop vs oracle"]]
    for policy in fixed_policy_order:
        if policy in fixed:
            label = "fixed warmup" if policy == "warmup" else policy
            fixed_rows.append(
                [
                    label,
                    fmt(fixed[policy]["mean_score_4ds"]),
                    fmt(fixed[policy]["mean_drop_4ds"]),
                ]
            )
    fixed_rows.extend(
        [
            [
                "Top C1",
                fmt(top5["criteria"][0]["mean_score_4ds"]),
                fmt(top5["criteria"][0]["mean_drop_4ds"]),
            ],
            [
                "Paper-friendly baseline",
                fmt(primary_p3["mean_score_4ds"]),
                fmt(primary_p3["mean_drop_4ds"]),
            ],
        ]
    )
    performance_toggles.append(
        toggle(
            (
                f"PF · paper-friendly baseline · {pretty_metric(pf_c['metric'])} · "
                f"mean={fmt(pf_c['mean_score_4ds'])}, drop={fmt(pf_c['mean_drop_4ds'])}"
            ),
            [
                callout(
                    [
                        "Rule: ",
                        (rule_text(pf_c), "code"),
                        ". 이 테이블은 Top-5 성능 후보가 아니라, 논문 본문에서 설명하기 쉬운 baseline criterion을 같은 기준으로 적용한 결과다. ",
                        "각 셀은 ",
                        ("성능 (#순위, eepoch)", "code"),
                        " 형식이다.",
                    ],
                    "yellow_background",
                ),
                table(model_rank_table_rows(pf_payload)),
            ],
        )
    )
    for ablation in pf_ablation_tables:
        c = ablation["criterion"]
        performance_toggles.append(
            toggle(
                (
                    f"{ablation['index']} · PF ablation · {pretty_metric(c['metric'])} · "
                    f"mean={fmt(c['mean_score_4ds'])}, drop={fmt(c['mean_drop_4ds'])}"
                ),
                [
                    callout(
                        [
                            "Rule: ",
                            (rule_text(c), "code"),
                            ". ",
                            ablation.get("description", ""),
                            " 각 셀은 ",
                            ("성능 (#순위, eepoch)", "code"),
                            " 형식이다.",
                        ],
                        "yellow_background",
                    ),
                    table(model_rank_table_rows(ablation)),
                ],
            )
        )

    blocks = [
        callout(
            [
                "이 페이지는 2026-06-20 KST 기준 재작성본이다. 기존 초안의 단순 요약 구조를 버리고, ",
                "metric 선정 기준을 source-aware하게 다시 세웠다. 핵심 기준은 현재 저장 파일명이 아니라 ",
                ("train data + train label만으로 train-time에 계산 가능한가", "bold"),
                "이다.",
            ],
            "gray_background",
        ),
        toc(),
        divider(),
        heading(1, "핵심 결론"),
        callout(
            [
                "이전 판단에는 중요한 오류가 있었다. ",
                ("epoch_metrics.json", "code"),
                "에 저장되어 있다는 이유만으로 지표를 early stopping 후보에서 배제하면 안 된다. ",
                "현재 저장된 값이 eval/test split에서 계산된 값이면 그대로 쓰면 안 되지만, 같은 수식을 train split과 train label로 재계산할 수 있다면 ",
                "그 지표는 train-time metric으로 추가해서 후보에 포함해야 한다.",
            ],
            "red_background",
        ),
        bullet(
            [
                "따라서 최종 early stopping 분석의 metric universe에는 ",
                ("disc_snr", "code"),
                ", ",
                ("recon_snr", "code"),
                ", train label로 정상/이상 그룹을 나눈 score distribution separation 같은 ",
                "train-computable diagnostic이 포함되어야 한다.",
            ]
        ),
        bullet(
            [
                "현재 존재하는 산출물만 보면 4개 기준 데이터셋의 ",
                ("epoch_metrics.json", "code"),
                "에는 ",
                ("disc_snr/recon_snr", "code"),
                f"가 {source_counts['epoch_metrics_disc_snr']}/{source_counts['cells']}, "
                f"{source_counts['epoch_metrics_recon_snr']}/{source_counts['cells']} cell에 있지만, ",
                "그 값은 eval/test loader 기준이다. 반대로 per-epoch train score 파일인 ",
                ("train_epoch_scores", "code"),
                f"는 {source_counts['train_epoch_scores_dirs']}/{source_counts['cells']} cell이다.",
            ]
        ),
        bullet(
            [
                "따라서 아래 Top-5/PF/성능표는 최종 결론이 아니라, 현재 저장된 ",
                ("training_histories.json", "code"),
                "와 그 안에서 만들 수 있는 non-feature interaction에 대한 ",
                ("partial sweep evidence", "code"),
                "다. 최종 sweep은 train_epoch_metrics를 추가 수집한 뒤 다시 해야 한다.",
            ]
        ),
        bullet(
            [
                "사용자 선호에 맞춰 stopping 후보는 가급적 warmup 이후에서 선택되도록 ",
                ("post_warmup", "code"),
                " 시작 정책을 기본으로 보았고, 방향성/변곡점은 ",
                ("force_min/force_max/auto", "code"),
                " 및 ",
                ("peak_reversal", "code"),
                "로 평가했다.",
            ]
        ),
        divider(),
        heading(1, "분석 범위"),
        paragraph(
            "분석 대상은 results/experiments 아래 numeric experiment directory다. legacy_* 실험은 제외했고, "
            "SWaT은 train history는 A1A2_full, score는 A1A2_excl22를 사용했다."
        ),
        table(
            [["Dataset", "Cells", "Mean oracle pak_auc_f1"]]
            + [[ds, str(cells[ds]), fmt(oracle[ds])] for ds in ("SWaT_excl22", "PSM", "WaDi_A1", "WaDi_A2")]
        ),
        table([["Warmup epoch", "Cells"]] + [[str(k), str(v)] for k, v in warmups.items()]),
        heading(1, "Early Stopping 후보 설계"),
        numbered(
            [
                "후보 자격의 기준은 storage path가 아니라 source다. ",
                "train split 입력, train label, train-time model output만으로 계산 가능하면 eligible metric이다.",
            ]
        ),
        numbered(
            [
                ("train_feature_*", "code"),
                " aggregate는 제외했다. 실험 분석에는 쓸 수 있지만, 논문 본문에서 일반 early stopping mechanism으로 설명하기에는 직관성이 낮다.",
            ]
        ),
        numbered(
            [
                "현재 저장된 값이 eval/test split에서 계산된 경우에는 그 값을 그대로 쓰지 않는다. 대신 같은 계산식을 train split에서 다시 실행해 ",
                ("train_epoch_metrics", "code"),
                "로 수집해야 한다.",
            ]
        ),
        numbered(
            [
                "현재 저장된 ",
                ("training_histories.json", "code"),
                "만으로 이미 sweep한 metric은 loss/reconstruction 계열이다: ",
                ("train_loss", "code"),
                ", ",
                ("train_rec_loss", "code"),
                ", ",
                ("train_disc_loss", "code"),
                ", ",
                ("train_normal_loss", "code"),
                ", ",
                ("train_anomaly_loss", "code"),
                ", ",
                ("train_mean_discrepancy", "code"),
                ", teacher/student reconstruction normal/anomaly.",
            ]
        ),
        numbered(
            [
                "상호작용은 ",
                ("ratio", "code"),
                ", ",
                ("relgap", "code"),
                ", ",
                ("diff", "code"),
                ", ",
                ("absdiff", "code"),
                "로 만들고, patience는 ",
                ("2/3/5/8/10", "code"),
                "을 평가했다. EMA는 ",
                ("raw/ema01/ema02/ema03/ema05/ema07", "code"),
                "을 모두 비교했다.",
            ]
        ),
        heading(2, "Source-aware eligibility audit"),
        paragraph(
            "아래 표는 4개 기준 데이터셋 cell에서 현재 실제로 존재하는 산출물과, 그것을 최종 early stopping 분석에 어떻게 써야 하는지를 구분한 것이다. "
            "핵심은 현재 저장된 eval/test 값은 leak 때문에 그대로 쓰면 안 되지만, train으로 재계산 가능한 수식은 후보에서 배제하면 안 된다는 점이다."
        ),
        table(source_availability),
        heading(2, "Train-computable metric universe"),
        paragraph(
            "최종 sweep에 포함해야 하는 metric family는 아래와 같다. feature aggregate는 사용자가 부적절하다고 판단했으므로 paper-level mechanism에서는 제외한다."
        ),
        table(train_computable),
        callout(
            [
                "정정: PRC/AUC/F1/PAK/Aff/RF1/VUS 같은 성능지표는 early stopping 기준으로 쓰지 않는다. ",
                "train label은 성능 측정용 ranking/threshold metric 계산이 아니라, 정상/이상 그룹의 loss/score 분리 통계(SNR, gap, ratio, relgap)를 만들 때만 사용한다.",
            ],
            "red_background",
        ),
        callout(
            [
                "재계산 여부: 아직 새 train-time metric을 다시 계산하지 않았다. 현재 페이지의 수치는 기존 ",
                ("training_histories.json", "code"),
                " 기반 partial sweep을 다시 해석한 결과와 source availability audit이다. ",
                ("train_epoch_metrics", "code"),
                " 또는 ",
                ("train_epoch_scores", "code"),
                "가 없기 때문에 disc_snr/recon_snr 및 score distribution diagnostic을 train split 기준으로 전수 재계산한 상태가 아니다.",
            ],
            "yellow_background",
        ),
        callout(
            [
                "판단: 현재 데이터만으로는 최종 universe 전체를 전수 조사했다고 말할 수 없다. ",
                ("training_histories", "code"),
                " 기반 결과는 유지하되, ",
                ("disc_snr/recon_snr", "code"),
                "와 train-label 기반 score/loss 분리 진단을 train split에서 per-epoch로 수집한 뒤 sweep을 다시 수행해야 한다.",
            ],
            "yellow_background",
        ),
        heading(2, "Stored-history partial sweep coverage audit"),
        paragraph(
            "아래 표는 이번 요청에서 명시된 discrepancy/anomaly/reconstruction/SNR 계열이 실제 sweep 후보에 얼마나 들어갔는지 확인한 것이다. "
            "단, 이 표는 현재 저장된 training history로 만든 partial sweep의 coverage일 뿐이며, 최종 train-computable universe의 coverage가 아니다."
        ),
        table(metric_audit),
        callout(
            [
                "주의: 여기서 ",
                ("disc_snr alias", "code"),
                "는 ",
                ("training_histories", "code"),
                " 안의 scalar loss들로 만든 proxy family다. 실제 evaluator의 variance-normalized ",
                ("disc_snr/recon_snr", "code"),
                "는 ",
                ("epoch_metrics.json", "code"),
                "에 존재하지만 현재 저장값은 eval/test split 기준이다. 따라서 해당 지표는 배제 대상이 아니라 ",
                ("train_epoch_metrics", "code"),
                "로 재수집해야 하는 대상이다.",
            ],
            "yellow_background",
        ),
        heading(1, "train_loss와 train_normal_loss의 정확한 의미"),
        callout(
            [
                "중요: ",
                ("train_loss", "code"),
                "는 이름만 보면 전체 optimizer loss처럼 보이지만, 실제 history에는 criterion이 반환한 ",
                ("loss_dict['total_loss']", "code"),
                "가 저장된다. trainer 단계에서 나중에 더해지는 adaptive FM, GRL, SCAD 보조항까지 모두 합친 최종 backprop loss와는 다를 수 있다.",
            ],
            "red_background",
        ),
        table(
            [
                ["Metric", "Actual source", "Meaning"],
                [
                    "train_loss",
                    "epoch mean of loss_dict['total_loss']",
                    "warmup 중에는 teacher reconstruction loss만. post-warmup에는 reconstruction_loss + discrepancy_loss.",
                ],
                [
                    "train_rec_loss",
                    "loss_dict['reconstruction_loss']",
                    "masked position 기준 teacher reconstruction MSE.",
                ],
                [
                    "train_disc_loss",
                    "loss_dict['discrepancy_loss']",
                    "normal_loss + anomaly_loss. fm_adaptive_lambda=False일 때만 fixed-weight FM이 여기에 포함된다.",
                ],
                [
                    "train_normal_loss",
                    "loss_dict['normal_loss']",
                    "정상 patch/window에 대한 student-teacher output discrepancy loss. reconstruction이나 FM/GRL/SCAD를 포함하지 않는다.",
                ],
                [
                    "train_fm_loss",
                    "loss_dict['fm_loss']",
                    "teacher/student hidden feature distance diagnostic. adaptive FM이면 train_loss에는 포함되지 않고 trainer에서 별도 가중치로 backprop에 더해진다.",
                ],
            ]
        ),
        toggle(
            "warmup 전후 train_loss 의미가 바뀌는가?",
            [
                paragraph(
                    [
                        "그렇다. trainer는 ",
                        ("teacher_only = epoch < teacher_only_warmup_epochs", "code"),
                        "로 warmup을 판단한다. 이때 student decoder, discrepancy, FM/GRL/SCAD 계열은 loss 계산에서 빠지고 ",
                        ("total_loss = reconstruction_loss", "code"),
                        "가 된다.",
                    ]
                ),
                paragraph(
                    [
                        "post-warmup에서는 ",
                        ("total_loss = reconstruction_loss + discrepancy_loss", "code"),
                        "가 되며, ",
                        ("discrepancy_loss", "code"),
                        "는 정상/이상 output discrepancy 항을 포함한다. 다만 ",
                        ("fm_adaptive_lambda=True", "code"),
                        "인 현 canonical 계열에서는 adaptive FM 항이 ",
                        ("loss_dict['total_loss']", "code"),
                        "에 들어가지 않고 trainer에서 실제 backprop loss에 별도로 더해진다.",
                    ]
                ),
                paragraph(
                    [
                        "따라서 ",
                        ("train_loss / train_normal_loss", "code"),
                        " 같은 ratio는 warmup 경계 전후를 하나의 동질적인 시계열로 해석하면 안 된다. ",
                        ("train_normal_loss", "code"),
                        "는 teacher-only warmup 동안 0 sentinel에 가깝기 때문에 PF 계열은 원칙적으로 strictly post-warmup 구간에서 평가해야 한다.",
                    ]
                ),
            ],
        ),
        callout(
            [
                "갱신 전 sweep의 ",
                ("start_policy='warmup'", "code"),
                "는 코드상 ",
                ("e >= teacher_only_warmup_epochs", "code"),
                "인 evaluation point부터 허용한다. 하지만 history epoch는 ",
                ("epoch + 1", "code"),
                "로 저장되므로, logged epoch == warmup 값은 아직 teacher-only epoch다. 따라서 갱신된 Top-5/PF 성능표는 ",
                ("post_warmup", "code"),
                " 시작, 즉 logged epoch > warmup만 허용하는 strict variant를 기준으로 작성했다.",
            ],
            "yellow_background",
        ),
        heading(1, "용어와 수식 정의"),
        callout(
            [
                "이 섹션이 핵심이다. 아래 정의는 보고서용 재해석이 아니라 실제 sweep 코드에서 사용한 계산식 그대로다. ",
                "모든 식의 ",
                ("eps", "code"),
                "는 ",
                ("1e-8", "code"),
                "이다.",
            ],
            "red_background",
        ),
        table(
            [
                ["Term", "Exact definition", "Interpretation"],
                [
                    "ratio(A, B)",
                    "A_e / (|B_e| + eps)",
                    "A가 B에 비해 얼마나 큰지 보는 방향성 있는 비율. ratio(A,B)와 ratio(B,A)는 다른 후보로 취급했다.",
                ],
                [
                    "relgap(A, B)",
                    "(A_e - B_e) / (|A_e| + |B_e| + eps)",
                    "두 지표의 상대적 차이. loss처럼 양수 지표에서는 A>B이면 양수, A≈B이면 0, A<B이면 음수에 가깝다.",
                ],
                [
                    "diff(A, B)",
                    "A_e - B_e",
                    "scale normalization 없이 두 지표의 차이를 본다.",
                ],
                [
                    "absdiff(A, B)",
                    "|A_e - B_e|",
                    "두 지표의 절대 거리. 어느 쪽이 큰지는 버린다.",
                ],
                [
                    "EMA_0.3(x)",
                    "z_e = x_e for first finite e; otherwise z_e = 0.3 x_e + 0.7 z_{e-1}",
                    "epoch별 지표 변동을 완화한다. 결측이면 직전 EMA 값을 유지했다.",
                ],
            ]
        ),
        toggle(
            "EMA ratio는 정확히 무엇인가?",
            [
                paragraph(
                    [
                        ("EMA ratio(A, B)", "code"),
                        "는 ",
                        ("EMA(A) / EMA(B)", "code"),
                        "가 아니다. 먼저 각 epoch에서 ",
                        ("x_e = A_e / (|B_e| + eps)", "code"),
                        "를 계산하고, 그 ratio series에 EMA를 적용했다.",
                    ]
                ),
                code_block(
                    "x_e = A_e / (abs(B_e) + 1e-8)\n"
                    "z_e = x_e                                  # first finite epoch\n"
                    "z_e = alpha * x_e + (1 - alpha) * z_{e-1}  # later epochs\n"
                    "\n"
                    "# Example: EMA_0.3 ratio(train_loss, train_normal_loss)\n"
                    "x_e = train_loss_e / (abs(train_normal_loss_e) + 1e-8)\n"
                    "z_e = EMA_0.3(x_e)",
                    "plain text",
                ),
                paragraph(
                    "따라서 EMA ratio는 두 loss를 각각 smoothing한 뒤 나누는 방식보다, 순간적인 상대 비율의 변화를 더 직접적으로 보되 "
                    "epoch 단위 노이즈만 줄이는 방식이다."
                ),
            ],
        ),
        toggle(
            "EMA alpha는 어디까지 비교했나?",
            [
                paragraph(
                    [
                        "갱신된 sweep에서는 transform 후보로 ",
                        ("raw", "code"),
                        ", ",
                        ("ema01", "code"),
                        ", ",
                        ("ema02", "code"),
                        ", ",
                        ("ema03", "code"),
                        ", ",
                        ("ema05", "code"),
                        ", ",
                        ("ema07", "code"),
                        "을 모두 평가했다. 숫자는 각각 alpha=0.1/0.2/0.3/0.5/0.7을 뜻한다.",
                    ]
                ),
                paragraph(
                    "이 비교는 train metric series에 대한 post-process EMA alpha sweep이다. 모델 재학습을 다시 하는 것이 아니라, 동일한 history 위에서 early-stopping decision rule을 다시 평가한 것이다."
                ),
            ],
        ),
        toggle(
            "disc_snr와 recon_snr는 확인했나?",
            [
                paragraph(
                    [
                        "확인했다. 실제 evaluator SNR은 ",
                        ("scripts/run_base_experiments.py::compute_epoch_test_eval", "code"),
                        "에서 ",
                        ("evaluator.compute_detailed_losses()", "code"),
                        "와 ",
                        ("compute_loss_statistics", "code"),
                        "를 거쳐 계산된다.",
                    ]
                ),
                paragraph(
                    [
                        ("disc_snr", "code"),
                        "는 ",
                        ("(disc_anomaly - disc_normal) / (std_disc_anomaly + std_disc_normal + eps)", "code"),
                        "이고, ",
                        ("recon_snr", "code"),
                        "는 ",
                        ("(recon_anomaly - recon_normal) / (std_recon_anomaly + std_recon_normal + eps)", "code"),
                        "이다. 둘 다 train label을 사용하면 train split에서도 계산 가능하다.",
                    ]
                ),
                paragraph(
                    [
                        "다만 현재 ",
                        ("epoch_metrics.json", "code"),
                        "에 저장된 값은 ",
                        ("test_loader", "code"),
                        " 기반 eval/test callback에서 계산된 값이다. 그러므로 early stopping 입력으로 그대로 쓰면 leak이다. 올바른 처리는 같은 evaluator logic을 ",
                        ("train_infer_loader", "code"),
                        "에 적용해 ",
                        ("train_disc_snr/train_recon_snr", "code"),
                        "를 per-eval epoch로 새로 저장하는 것이다.",
                    ]
                ),
                paragraph(
                    [
                        "이번 partial sweep에 추가했던 ",
                        ("disc_snr__train_anomaly_loss__train_normal_loss", "code"),
                        ", ",
                        ("teacher_recon_snr__anomaly__normal", "code"),
                        " 같은 이름은 실제 evaluator SNR이 아니라 scalar history 기반 proxy다. 최종 보고에서는 proxy와 실제 SNR을 분리해서 다뤄야 한다.",
                    ]
                ),
            ],
        ),
        toggle(
            "relgap은 왜 ratio와 다른가?",
            [
                paragraph(
                    [
                        ("ratio(A,B)", "code"),
                        "는 B가 작아질 때 값이 크게 흔들릴 수 있다. 반면 ",
                        ("relgap(A,B)", "code"),
                        "은 분모에 ",
                        ("|A| + |B|", "code"),
                        "를 쓰므로 두 값의 scale을 같이 반영한다.",
                    ]
                ),
                paragraph(
                    [
                        "C1의 ",
                        ("relgap(train_loss, train_student_recon_anomaly)", "code"),
                        "는 ",
                        ("(train_loss - train_student_recon_anomaly) / (|train_loss| + |train_student_recon_anomaly| + eps)", "code"),
                        "다. 이 후보는 ",
                        ("force_min", "code"),
                        "으로 평가했으므로, 이 상대 gap이 낮아지는 방향을 좋은 방향으로 본다.",
                    ]
                ),
                paragraph(
                    "해석은 조심해야 한다. 이 값이 낮다는 것은 train_loss가 anomaly-side student reconstruction diagnostic에 비해 상대적으로 작아진 상태를 뜻한다. "
                    "그 자체가 이론적 optimality를 증명하는 것은 아니며, 현재 누적 실험에서 성능 선택 신호로 강했다는 경험적 결과다."
                ),
            ],
        ),
        heading(1, "Direction, Patience, Rollback의 실제 동작"),
        paragraph(
            "모든 stopping rule은 내부적으로 하나의 값을 최대화하는 형태로 통일해서 평가했다. 원래 지표를 줄이는 것이 좋은 경우에는 "
            "부호를 뒤집어 내부 score를 만들었다."
        ),
        table(
            [
                ["Setting", "Internal value", "Meaning"],
                ["force_min", "v_e = -z_e", "원래 지표 z_e가 낮아질수록 improvement"],
                ["force_max", "v_e = z_e", "원래 지표 z_e가 높아질수록 improvement"],
                ["auto", "metric name 기반 heuristic", "loss/error/recon 계열은 min, ratio/gap 계열은 max로 초기 추정"],
            ]
        ),
        callout(
            [
                "중요: 이 보고서의 성능 테이블에 적힌 ",
                ("epoch", "code"),
                "는 patience가 만료된 trigger epoch가 아니다. 모든 보고된 sweep은 ",
                ("rollback=best_seen_before_stop", "code"),
                "이므로, 표의 epoch는 patience trigger 전까지 train metric이 가장 좋았던 checkpoint epoch다.",
            ],
            "red_background",
        ),
        toggle(
            "standard patience algorithm",
            [
                paragraph(
                    "standard rule은 warmup 이후 첫 evaluation epoch를 best로 두고 시작한다. 이후 지표가 threshold 이상 개선되면 best를 갱신하고 "
                    "counter를 0으로 되돌린다. 개선이 없으면 counter를 1씩 올리고, counter가 patience에 도달하면 early-stop trigger가 발생한다."
                ),
                code_block(
                    "best_epoch = first_eval_epoch_after_warmup\n"
                    "best_value = v[best_epoch]\n"
                    "bad_count = 0\n"
                    "\n"
                    "for e in later_eval_epochs:\n"
                    "    if improvement(v[e], best_value, threshold):\n"
                    "        best_epoch = e\n"
                    "        best_value = v[e]\n"
                    "        bad_count = 0\n"
                    "    else:\n"
                    "        bad_count += 1\n"
                    "        if bad_count >= patience:\n"
                    "            trigger_epoch = e\n"
                    "            selected_epoch = best_epoch   # rollback=best_seen_before_stop\n"
                    "            break",
                    "python",
                ),
                paragraph(
                    [
                        "예를 들어 ",
                        ("P=3", "code"),
                        "이면 개선 없는 evaluation point가 3번 연속 나오면 그 시점에서 stop trigger가 발생한다. 하지만 성능 계산은 ",
                        ("selected_epoch = best_epoch", "code"),
                        "으로 되돌아간 checkpoint에서 했다.",
                    ]
                ),
            ],
        ),
        toggle(
            "relative threshold 0.5%는 어떻게 적용했나?",
            [
                paragraph(
                    [
                        ("threshold_type='rel', threshold_value=0.005", "code"),
                        "는 내부 value 기준으로 ",
                        ("(new - old) / max(|old|, eps) > 0.005", "code"),
                        "일 때만 improvement로 인정한다는 뜻이다.",
                    ]
                ),
                paragraph(
                    [
                        ("force_min", "code"),
                        "에서는 내부 value가 ",
                        ("v=-z", "code"),
                        "이므로, 원래 지표 ",
                        ("z", "code"),
                        "가 충분히 낮아져야 improvement가 된다. 즉 0.5% threshold는 원래 지표 값이 아니라, 방향 변환 후의 내부 value 개선폭에 적용된다.",
                    ]
                ),
            ],
        ),
        toggle(
            "peak_reversal은 감소 후 상승 변곡점을 어떻게 잡나?",
            [
                paragraph(
                    [
                        ("peak_reversal", "code"),
                        "도 내부 value ",
                        ("v", "code"),
                        "를 기준으로 동작한다. ",
                        ("force_min", "code"),
                        "인 경우 ",
                        ("v=-z", "code"),
                        "이므로 내부 peak는 원래 지표 ",
                        ("z", "code"),
                        "의 valley, 즉 최소점에 해당한다.",
                    ]
                ),
                code_block(
                    "peak_epoch = first_eval_epoch_after_warmup\n"
                    "peak_value = v[peak_epoch]\n"
                    "drop_count = 0\n"
                    "\n"
                    "for e in later_eval_epochs:\n"
                    "    if v[e] > peak_value:\n"
                    "        peak_epoch = e\n"
                    "        peak_value = v[e]\n"
                    "        drop_count = 0\n"
                    "    elif significant_drop(peak_value, v[e], threshold):\n"
                    "        drop_count += 1\n"
                    "        if drop_count >= patience:\n"
                    "            trigger_epoch = e\n"
                    "            selected_epoch = peak_epoch   # rollback\n"
                    "            break\n"
                    "    else:\n"
                    "        drop_count = 0",
                    "python",
                ),
                paragraph(
                    "따라서 사용자가 말한 '감소하다가 상승하는 변곡점'은 force_min + peak_reversal 조합에서 명시적으로 평가했다. "
                    "다만 현재 누적 실험에서는 paper-friendly baseline에 대해 standard patience가 peak_reversal보다 약간 더 나았다."
                ),
            ],
        ),
        heading(1, "paper-friendly의 정확한 의미"),
        paragraph(
            "여기서 paper-friendly는 성능이 가장 높다는 뜻이 아니다. 논문 본문에서 독자가 납득하기 쉬운 stopping mechanism이라는 뜻으로 썼다. "
            "즉 계산식이 단순하고, train data와 train label만으로 계산 가능하며, feature별 p90/max 같은 구현 의존적 aggregate를 쓰지 않고, 특정 ablation에 과하게 맞춘 느낌이 적은 기준이다."
        ),
        table(
            [
                ["Criterion type", "Example", "Strength", "Weakness"],
                [
                    "Top-score criterion",
                    "C1: relgap(train_loss, train_student_recon_anomaly), EMA, P=10",
                    "현재 stored-history partial sweep에서 평균 score가 가장 높음",
                    "train-computable score-separation diagnostic을 아직 포함하지 않았으므로 최종 default로 단정할 수 없음",
                ],
                [
                    "paper-friendly baseline",
                    "EMA ratio(train_loss, train_normal_loss), P=3",
                    "전체 train objective와 normal split loss의 상대 진행도라는 설명이 직관적",
                    "Top-score C1보다 평균 score는 낮음",
                ],
            ]
        ),
        callout(
            "따라서 paper-friendly baseline은 '최종 성능 최강 후보'가 아니라, 논문 본문에서 기본 메커니즘으로 설명하기 쉬운 기준이다. "
            "성능 우선 후보와 paper-friendly 후보를 분리해서 보고하는 것이 가장 정직하다.",
            "yellow_background",
        ),
        heading(2, "PF ablation: EMA와 patience를 제거하면?"),
        paragraph(
            "PF 기준에서 EMA smoothing과 patience를 각각 제거한 경우를 추가로 계산했다. 여기서 patience를 쓰지 않는다는 것은 "
            "무한히 기다리거나 마지막 epoch를 고르는 뜻이 아니라, 개선이 없는 첫 평가 지점에서 바로 stop trigger를 발생시키는 설정이다. "
            "단, 기존 PF와 동일하게 rollback은 적용하므로 실제 선택 epoch는 trigger epoch가 아니라 trigger 전 best checkpoint다."
        ),
        table(pf_ablation_summary),
        callout(
            "요약하면, EMA 제거는 평균 성능을 크게 낮췄고, patience 제거는 PF와 거의 비슷했다. 따라서 PF에서 중요한 안정화 요소는 patience보다 EMA 쪽으로 보인다.",
            "green_background",
        ),
        heading(2, "PF EMA alpha sweep"),
        paragraph(
            "PF 기준의 transform만 바꾸고 나머지 rule은 동일하게 둔 비교다. raw는 EMA를 쓰지 않는 경우이며, ema01/02/03/05/07은 각각 alpha=0.1/0.2/0.3/0.5/0.7이다."
        ),
        table(pf_ema_summary),
        code_block(
            "conda run -n dc_vis python -u scripts/summarize_early_stopping_train_metric_sweep_4ds.py\n"
            "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 conda run -n dc_vis python -u scripts/early_stopping_nonfeature_interaction_sweep_4ds.py --workers 3\n"
            "conda run -n dc_vis python -u scripts/build_early_stopping_rank_tables.py"
        ),
        heading(1, "Stored-History Top-5 + Paper-Friendly 후보 요약"),
        callout(
            [
                "아래 표의 C1-C5는 현재 저장된 ",
                ("training_histories.json", "code"),
                " 기반 partial sweep에서의 mean score 기준 Top-5이고, PF는 같은 partial sweep에서의 paper-friendly baseline이다. ",
                "train-computable score-separation diagnostic까지 포함한 최종 순위가 아니다.",
            ],
            "yellow_background",
        ),
        table(top5_summary),
        heading(1, "Fixed Baseline 대비"),
        table(fixed_rows),
        heading(1, "성능 테이블"),
        callout(
            [
                "각 테이블은 해당 stored-history stopping criterion을 실제로 적용한 결과다. C1-C5는 partial sweep의 mean score 기준 Top-5이고, PF 및 PF ablation은 paper-friendly 계열 비교다. 행은 numeric experiment, 열은 데이터셋이다. ",
                "셀 형식은 ",
                ("성능 (#순위, eepoch)", "code"),
                "이며, 평균 순위가 낮은 모델부터 정렬했다. Coverage가 4보다 작으면 해당 데이터셋 산출물이 없거나 criterion 적용이 불가능했던 경우다.",
            ],
            "blue_background",
        ),
        *performance_toggles,
        divider(),
        heading(1, "방향성과 변곡점"),
        paragraph(
            "방향성은 early stopping에서 핵심이다. 같은 지표라도 커지는 것이 좋은지, 작아지는 것이 좋은지에 따라 stop epoch가 달라진다. "
            "현재 partial sweep에서는 auto/force_min/force_max를 모두 평가했고, 변곡점은 peak_reversal로 별도 계산했다. 최종 train_epoch_metrics sweep에서도 동일한 rule space를 유지해야 한다."
        ),
        table(
            [
                ["Criterion", "Rule", "Mean score", "Drop", "Mean stop", "Interpretation"],
                [
                    "EMA ratio(train_loss, train_normal_loss)",
                    "standard, P=3",
                    fmt(primary_p3["mean_score_4ds"]),
                    fmt(primary_p3["mean_drop_4ds"]),
                    fmt_epoch(primary_p3["mean_stop_epoch_4ds"]),
                    "paper-friendly baseline",
                ],
                [
                    "EMA ratio(train_loss, train_normal_loss)",
                    "peak_reversal, P=3",
                    fmt(primary_reversal["mean_score_4ds"]),
                    fmt(primary_reversal["mean_drop_4ds"]),
                    fmt_epoch(primary_reversal["mean_stop_epoch_4ds"]),
                    "감소 후 상승 변곡점 후보이나 baseline보다 약간 열세",
                ],
            ]
        ),
        heading(1, "판단"),
        paragraph(
            "정정된 판단은 명확하다. 현재 저장 위치가 epoch_metrics인지 training_histories인지가 기준이 아니다. "
            "train split과 train label만으로 train-time에 계산 가능한 지표라면 train-time metric으로 추가해서 early stopping 후보에 포함해야 한다."
        ),
        paragraph(
            [
                "따라서 현재 C1-C5/PF 성능표는 ",
                ("training_histories", "code"),
                " 기반 부분 증거로만 해석해야 한다. 이 결과는 fixed baseline 대비 어느 정도 유효한 후보가 있음을 보여주지만, ",
                ("disc_snr/recon_snr", "code"),
                " 및 train-label 기반 score/loss 분리 진단을 포함한 최종 전수조사를 대체하지 못한다.",
            ]
        ),
        paragraph(
            [
                "재분석을 위해 필요한 구현은 간단하다. 기존 eval callback에서 test/eval loader로 수행하던 inference/evaluation과 같은 로직을 ",
                ("train_infer_loader", "code"),
                "에도 적용해 per-eval epoch마다 ",
                ("train_epoch_metrics.json", "code"),
                "과 필요 시 ",
                ("train_epoch_scores/epoch_NNN_scores.npz", "code"),
                "를 저장한다. 여기에 train 기준 ",
                ("disc_snr/recon_snr", "code"),
                ", adaptive/recon/disc score의 정상/이상 mean/std/quantile gap, ratio, relgap을 넣고, PRC/AUC/F1/PAK/Aff/RF1/VUS 및 feature aggregate는 제외한 뒤 동일한 EMA/patience/direction/post_warmup sweep을 다시 실행해야 한다.",
            ]
        ),
        callout(
            "즉, 지금 보고서의 결론은 '최종 early stopping 기준 확정'이 아니라 '기존 stored-history 분석을 바로잡고, 누락된 train-computable metric universe를 명시한 재분석 계획 + 부분 증거'다.",
            "red_background",
        ),
    ]
    return blocks


def append_blocks(page_id: str, blocks: list[dict]) -> None:
    for block in blocks:
        notion_request("PATCH", f"/blocks/{page_id}/children", {"children": [block]})
        time.sleep(0.08)


def update_title(page_id: str) -> None:
    notion_request(
        "PATCH",
        f"/pages/{page_id}",
        {
            "properties": {
                "title": {
                    "title": [
                        {
                            "type": "text",
                            "text": {"content": "Early Stopping 기준 재검토: train-computable metric source audit"},
                        }
                    ]
                }
            }
        },
    )


def main() -> None:
    notion_request("GET", f"/pages/{PARENT_PAGE_ID}")
    notion_request("GET", f"/pages/{PAGE_ID}")
    update_title(PAGE_ID)
    clear_page(PAGE_ID)
    append_blocks(PAGE_ID, build_blocks())
    page = notion_request("GET", f"/pages/{PAGE_ID}")
    children = retrieve_children(PAGE_ID)
    print(json.dumps({"id": PAGE_ID, "url": page.get("url"), "blocks": len(children)}, indent=2))


if __name__ == "__main__":
    main()
