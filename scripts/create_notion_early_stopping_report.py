"""Create the TSMAE early-stopping report as a Notion child page."""

from __future__ import annotations

import csv
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable


PARENT_PAGE_ID = "31687856b20780e29fbcd961d69773ea"
NOTION_VERSION = "2022-06-28"
TOKEN_ENV = Path("/home/ykio/.codex/notion-api-token.env")
NONFEATURE_DIR = Path("temp/early_stopping_nonfeature_interaction_4ds")
FULL_SWEEP_DIR = Path("temp/early_stopping_train_metrics_4ds")


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
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Notion API {method} {path} failed: {exc.code} {body}") from exc


def load_csv(path: Path) -> list[dict]:
    rows = []
    text_cols = {
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
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            for key, value in list(row.items()):
                if key not in text_cols:
                    try:
                        row[key] = float(value)
                    except (TypeError, ValueError):
                        pass
            rows.append(row)
    return rows


def find_row(rows: list[dict], **criteria: object) -> dict:
    for row in rows:
        ok = True
        for key, value in criteria.items():
            if row.get(key) != value:
                ok = False
                break
        if ok:
            return row
    raise KeyError(criteria)


def fmt(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def fmt_epoch(value: float) -> str:
    return f"{float(value):.1f}"


def rich_text(content: str, *, bold: bool = False, code: bool = False) -> list[dict]:
    return [
        {
            "type": "text",
            "text": {"content": content},
            "annotations": {
                "bold": bold,
                "italic": False,
                "strikethrough": False,
                "underline": False,
                "code": code,
                "color": "default",
            },
        }
    ]


def mixed(parts: Iterable[str | tuple[str, str]]) -> list[dict]:
    out = []
    for part in parts:
        if isinstance(part, tuple):
            content, style = part
            out.extend(rich_text(content, bold=style == "bold", code=style == "code"))
        else:
            out.extend(rich_text(part))
    return out


def paragraph(parts: Iterable[str | tuple[str, str]] | str) -> dict:
    if isinstance(parts, str):
        parts = [parts]
    return {"type": "paragraph", "paragraph": {"rich_text": mixed(parts)}}


def heading(level: int, text: str) -> dict:
    key = f"heading_{level}"
    return {"type": key, key: {"rich_text": rich_text(text)}}


def bullet(parts: Iterable[str | tuple[str, str]] | str) -> dict:
    if isinstance(parts, str):
        parts = [parts]
    return {"type": "bulleted_list_item", "bulleted_list_item": {"rich_text": mixed(parts)}}


def quote(text: str) -> dict:
    return {"type": "quote", "quote": {"rich_text": rich_text(text)}}


def divider() -> dict:
    return {"type": "divider", "divider": {}}


def code_block(text: str, language: str = "bash") -> dict:
    return {"type": "code", "code": {"rich_text": rich_text(text), "language": language}}


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
                        "cells": [rich_text(cell, bold=header and i == 0) for cell in row]
                    },
                }
                for i, row in enumerate(rows)
            ],
        },
    }


def metric_label(metric: str) -> str:
    return metric.replace("pair_ratio__", "ratio(").replace("pair_relgap__", "relgap(").replace("__", ", ") + ")"


def candidate_summary(row: dict, name: str, judgment: str) -> list[str]:
    threshold = f"{row['threshold_type']} {row['threshold_value']}"
    rule = (
        f"{row['transform']} / {row['direction_mode']} / {row['rule']} / "
        f"P={int(row['patience'])} / {threshold}"
    )
    return [
        name,
        metric_label(row["metric"]),
        rule,
        fmt(row["mean_score_4ds"]),
        fmt(row["mean_drop_4ds"]),
        fmt_epoch(row["mean_stop_epoch_4ds"]),
        f"{fmt(row['mean_after_warmup_pct_4ds'], 1)}%",
        judgment,
    ]


def dataset_rows(row: dict) -> list[list[str]]:
    out = [["Dataset", "Mean score", "Drop vs oracle", "Mean stop epoch", "After warmup"]]
    for ds in ("SWaT_excl22", "PSM", "WaDi_A1", "WaDi_A2"):
        out.append(
            [
                ds,
                fmt(row[f"{ds}_mean_score"]),
                fmt(row[f"{ds}_mean_drop"]),
                fmt_epoch(row[f"{ds}_mean_stop_epoch"]),
                f"{fmt(row[f'{ds}_after_warmup_pct'], 1)}%",
            ]
        )
    return out


def build_blocks() -> list[dict]:
    nonfeature_rows = load_csv(NONFEATURE_DIR / "leaderboard_warmup_full_coverage.csv")
    nonfeature_summary = json.loads((NONFEATURE_DIR / "summary.json").read_text())
    full_summary = json.loads((FULL_SWEEP_DIR / "summary.json").read_text())

    primary_p2 = find_row(
        nonfeature_rows,
        metric="pair_ratio__train_loss__train_normal_loss",
        transform="ema03",
        direction_mode="force_min",
        rule="standard",
        patience=2.0,
        threshold_type="rel",
        threshold_value=0.005,
        start_policy="warmup",
    )
    primary_p3 = find_row(
        nonfeature_rows,
        metric="pair_ratio__train_loss__train_normal_loss",
        transform="ema03",
        direction_mode="force_min",
        rule="standard",
        patience=3.0,
        threshold_type="rel",
        threshold_value=0.005,
        start_policy="warmup",
    )
    primary_p3_reversal = find_row(
        nonfeature_rows,
        metric="pair_ratio__train_loss__train_normal_loss",
        transform="ema03",
        direction_mode="force_min",
        rule="peak_reversal",
        patience=3.0,
        threshold_type="rel",
        threshold_value=0.005,
        start_policy="warmup",
    )
    score_oriented = find_row(
        nonfeature_rows,
        metric="pair_relgap__train_loss__train_student_recon_anomaly",
        transform="ema03",
        direction_mode="force_min",
        rule="standard",
        patience=10.0,
        threshold_type="rel",
        threshold_value=0.005,
        start_policy="warmup",
    )
    formal_rank = sorted(nonfeature_rows, key=lambda r: (r["avg_rank_4ds"], -r["mean_score_4ds"]))[0]

    oracle = full_summary["mean_oracle_by_dataset"]
    cells = nonfeature_summary["n_cells_by_dataset"]
    warmups = nonfeature_summary["warmup_distribution"]
    fixed = {
        row["policy"]: row
        for row in json.loads((FULL_SWEEP_DIR / "analysis_summary.json").read_text())[
            "fixed_epoch_baselines"
        ]
    }

    blocks = [
        heading(1, "요약"),
        quote(
            "결론: 논문 수준의 early stopping 기준으로는 feature aggregate를 배제하고, warmup 이후에만 "
            "EMA(train_loss / train_normal_loss)를 최소화하는 patience 기반 규칙을 쓰는 것이 가장 설득력 있다."
        ),
        paragraph(
            [
                "이번 재분석은 사용자가 지정한 제약을 반영해 ",
                ("train_feature_*", "code"),
                " 계열을 제외하고, 훈련 단계에서 수집되는 loss 및 label-split diagnostic 사이의 상호작용만 다시 살핀 결과다. ",
                "핵심은 단순한 단일 지표가 아니라, 전체 학습 목적과 정상 구간 손실 사이의 상대적 진행도를 보는 것이다.",
            ]
        ),
        bullet(
            [
                "추천 기준: ",
                ("EMA_0.3(train_loss / train_normal_loss)", "code"),
                "를 warmup 이후부터 최소화하고, relative improvement 0.5% 기준의 patience 3을 적용한다.",
            ]
        ),
        bullet(
            [
                "성능 기준으로는 patience 2가 4개 데이터셋 평균 ",
                (fmt(primary_p2["mean_score_4ds"]), "bold"),
                "로 아주 근소하게 높지만, patience 3은 평균 ",
                (fmt(primary_p3["mean_score_4ds"]), "bold"),
                "로 거의 동일하면서 평가 노이즈에 덜 민감하다.",
            ]
        ),
        bullet(
            [
                "가장 높은 평균 점수 후보는 ",
                ("relgap(train_loss, train_student_recon_anomaly)", "code"),
                " 계열이지만, PSM 및 WaDi rank 균형이 나빠 보조 후보로만 보는 편이 타당하다.",
            ]
        ),
        bullet(
            [
                "warmup 분포는 177개 셀 중 161개가 250 epoch다. 따라서 stopping rule은 ",
                ("start_policy=warmup", "code"),
                "으로 정의해야 대부분의 실험에서 student 합류 이후의 변화를 기준으로 삼는다.",
            ]
        ),
        divider(),
        heading(2, "1. 분석 범위와 원칙"),
        paragraph(
            "분석 대상은 현재 results/experiments 아래의 numeric experiment directory이며, legacy 실험은 제외했다. "
            "평가 데이터셋은 SWaT(excl22), PSM, WaDi A1, WaDi A2다. 각 셀의 oracle은 해당 셀 epoch_metrics.json에서 "
            "pak_auc_f1이 최대인 epoch로 정의했다."
        ),
        table(
            [["Dataset", "Cells", "Mean oracle pak_auc_f1"]]
            + [[ds, str(cells[ds]), fmt(oracle[ds])] for ds in ("SWaT_excl22", "PSM", "WaDi_A1", "WaDi_A2")]
        ),
        paragraph(
            [
                "Early stopping 입력으로는 ",
                ("training_histories.json", "code"),
                "에서 훈련 중 계산되는 지표만 사용했다. Label은 훈련 단계에서 이미 알고 있는 정보이므로, ",
                ("train_normal_loss", "code"),
                ", ",
                ("train_anomaly_loss", "code"),
                ", ",
                ("train_student_recon_normal/anomaly", "code"),
                "처럼 label split으로 집계된 train diagnostic은 허용했다.",
            ]
        ),
        table([["Warmup epoch", "Cells"]] + [[str(k), str(v)] for k, v in warmups.items()]),
        heading(2, "2. 제외한 지표와 이유"),
        bullet(
            [
                ("epoch_*", "code"),
                " score 및 contribution history는 test-evaluation callback 산물이므로 early stopping 입력에서 제외했다.",
            ]
        ),
        bullet(
            [
                ("train_feature_*", "code"),
                " aggregate는 이번 재분석에서 제외했다. feature별 max, p90, range는 실험 분석에는 유용하지만, "
                "논문 본문에서 일반 early stopping mechanism으로 설명하기에는 직관성이 낮다.",
            ]
        ),
        bullet("batch profiling, epoch timing, GRL/SCAD ramp lambda처럼 데이터 상태보다 schedule을 반영하는 항목도 stopping 근거로 쓰지 않았다."),
        paragraph(
            "남긴 지표는 train_loss, train_rec_loss, train_disc_loss, train_normal_loss, train_anomaly_loss, "
            "train_mean_discrepancy, teacher/student reconstruction normal/anomaly diagnostic이다. 이들은 모두 손실 또는 "
            "재구성 품질이라는 동일한 의미 체계 안에서 설명할 수 있다."
        ),
        heading(2, "3. Sweep 설계"),
        paragraph(
            [
                "단일 지표뿐 아니라 지표 간 상호작용을 보기 위해 ordered ",
                ("ratio", "code"),
                ", ordered ",
                ("relgap", "code"),
                ", ",
                ("diff", "code"),
                ", ",
                ("absdiff", "code"),
                "를 구성했다. ordered ratio와 ordered relgap은 A/B와 B/A가 다른 의미를 갖기 때문에 양방향을 모두 평가했다.",
            ]
        ),
        bullet("변환: raw, EMA_0.3"),
        bullet("방향: auto, force_min, force_max. 즉 지표가 커지는 것이 좋은 경우와 작아지는 것이 좋은 경우를 모두 평가했다."),
        bullet(
            "규칙: standard patience와 peak reversal. force_min 지표에서 peak reversal은 원래 지표가 감소하다가 다시 상승하는 valley reversal을 포착한다."
        ),
        bullet("Patience: 2, 3, 5, 8, 10"),
        bullet("Threshold: absolute 0.0, relative 0.5%, relative 1.0%"),
        bullet("Checkpoint rollback: stopping 시점이 아니라 stop 전까지의 train-metric best checkpoint를 선택"),
        bullet("Start policy: epoch100과 warmup을 모두 계산했지만, 최종 추천은 warmup 시작 후보만 대상으로 판단"),
        table(
            [
                ["Sweep", "Full coverage criteria", "Warmup-start full coverage criteria"],
                [
                    "Non-feature interaction sweep",
                    str(nonfeature_summary["n_criteria_full_coverage"]),
                    str(nonfeature_summary["n_criteria_warmup_full_coverage"]),
                ],
            ]
        ),
        code_block(
            "conda run -n dc_vis python -u scripts/early_stopping_nonfeature_interaction_sweep_4ds.py\n"
            "conda run -n dc_vis python -u scripts/early_stopping_train_metric_sweep_4ds.py"
        ),
        heading(2, "4. 핵심 결과"),
        paragraph(
            "고정 epoch baseline과 비교하면 feature-free train-metric 기준은 뚜렷하게 낫다. 특히 warmup 시작 규칙은 "
            "student가 실제로 합류한 뒤의 학습 동역학을 보므로, 사용자가 우려한 pre-warmup 선택 문제를 구조적으로 피한다."
        ),
        table(
            [
                ["Policy", "4-dataset mean score", "Mean drop vs oracle"],
                ["epoch50", fmt(fixed["epoch50"]["mean_score_4ds"]), fmt(fixed["epoch50"]["mean_drop_4ds"])],
                ["epoch100", fmt(fixed["epoch100"]["mean_score_4ds"]), fmt(fixed["epoch100"]["mean_drop_4ds"])],
                ["epoch150", fmt(fixed["epoch150"]["mean_score_4ds"]), fmt(fixed["epoch150"]["mean_drop_4ds"])],
                ["warmup fixed", fmt(fixed["warmup"]["mean_score_4ds"]), fmt(fixed["warmup"]["mean_drop_4ds"])],
                [
                    "recommended P=3",
                    fmt(primary_p3["mean_score_4ds"]),
                    fmt(primary_p3["mean_drop_4ds"]),
                ],
            ]
        ),
        table(
            [
                [
                    "Candidate",
                    "Interaction",
                    "Rule",
                    "Mean score",
                    "Drop",
                    "Mean stop",
                    "After warmup",
                    "Judgment",
                ],
                candidate_summary(primary_p2, "Balanced P=2", "점수는 가장 높지만 다소 민감"),
                candidate_summary(primary_p3, "Recommended P=3", "논문 기본값으로 가장 적절"),
                candidate_summary(primary_p3_reversal, "Reversal P=3", "감소 후 상승 변곡점 후보, 기본값보다는 열세"),
                candidate_summary(score_oriented, "Score-oriented", "평균 점수는 높지만 dataset rank 불균형"),
                candidate_summary(formal_rank, "Formal avg-rank winner", "간접적이고 평균 점수가 낮아 비추천"),
            ]
        ),
        heading(3, "추천 후보의 데이터셋별 동작"),
        paragraph(
            [
                "아래는 ",
                ("EMA(train_loss / train_normal_loss)", "code"),
                ", patience 3, relative threshold 0.5%, warmup 시작 기준의 데이터셋별 결과다.",
            ]
        ),
        table(dataset_rows(primary_p3)),
        heading(2, "5. Patience 해석"),
        paragraph(
            "Patience를 키우면 항상 좋아지지 않았다. P=2와 P=3은 동일한 지표 family에서 가장 안정적이고, P=5 이상에서는 "
            "정지 시점이 늦어지면서 일부 데이터셋에서 점수는 유지되지만 PSM 및 WaDi rank 균형이 흔들렸다."
        ),
        table(
            [
                ["Patience", "Representative top family", "Mean score", "Mean stop", "Interpretation"],
                ["2", "EMA ratio(train_loss, train_normal_loss)", fmt(primary_p2["mean_score_4ds"]), fmt_epoch(primary_p2["mean_stop_epoch_4ds"]), "민감하지만 성능 최고"],
                ["3", "EMA ratio(train_loss, train_normal_loss)", fmt(primary_p3["mean_score_4ds"]), fmt_epoch(primary_p3["mean_stop_epoch_4ds"]), "성능 거의 동일, 운영 안정성 우위"],
                ["5", "EMA ratio(train_disc_loss, train_student_recon_normal)", "0.7200", "262.7", "균형은 가능하나 해석이 더 복잡"],
                ["8", "EMA ratio(train_student_recon_normal, train_mean_discrepancy)", "0.7153", "257.1", "평균 점수 하락"],
                ["10", "EMA relgap(train_loss, train_student_recon_anomaly)", fmt(score_oriented["mean_score_4ds"]), fmt_epoch(score_oriented["mean_stop_epoch_4ds"]), "평균 점수형 보조 후보"],
            ]
        ),
        paragraph(
            "따라서 기본값은 patience 3이 적절하다. P=2는 실험용 ablation으로 남길 수 있지만, 논문 또는 default 구현에서는 "
            "연속 평가 3회 동안 의미 있는 개선이 없는 경우 stop하는 설명이 더 자연스럽다."
        ),
        heading(2, "6. 방향성과 변곡점 기준"),
        paragraph(
            "사용자가 지적한 것처럼 early stopping은 단순히 값의 크기만 볼 것이 아니라 방향과 변곡점도 고려해야 한다. "
            "이번 sweep에서는 이를 두 층으로 반영했다. 첫째, 각 지표에 대해 작아지는 것이 좋은지, 커지는 것이 좋은지를 "
            "auto/force_min/force_max로 모두 비교했다. 둘째, 지표가 개선되다가 되돌아서는 지점을 잡기 위해 peak reversal "
            "규칙을 별도로 평가했다."
        ),
        paragraph(
            [
                "특히 추천 지표 ",
                ("EMA(train_loss / train_normal_loss)", "code"),
                "는 force_min 방향이므로, peak reversal은 비율이 감소한 뒤 다시 상승하는 지점, 즉 valley 이후의 반등을 잡는 규칙이다. "
                "따라서 사용자가 예로 든 '감소하다가 상승하는 변곡점'은 이미 이 후보군에 포함되어 있다.",
            ]
        ),
        table(
            [
                ["Rule", "Mean score", "Drop", "Mean stop", "After warmup", "Judgment"],
                [
                    "standard, P=3",
                    fmt(primary_p3["mean_score_4ds"]),
                    fmt(primary_p3["mean_drop_4ds"]),
                    fmt_epoch(primary_p3["mean_stop_epoch_4ds"]),
                    f"{fmt(primary_p3['mean_after_warmup_pct_4ds'], 1)}%",
                    "기본 추천",
                ],
                [
                    "peak reversal, P=3",
                    fmt(primary_p3_reversal["mean_score_4ds"]),
                    fmt(primary_p3_reversal["mean_drop_4ds"]),
                    fmt_epoch(primary_p3_reversal["mean_stop_epoch_4ds"]),
                    f"{fmt(primary_p3_reversal['mean_after_warmup_pct_4ds'], 1)}%",
                    "변곡점 명시 후보이나 평균 rank와 점수가 약간 낮음",
                ],
            ]
        ),
        paragraph(
            "따라서 방향성과 변곡점은 분석에서 배제하지 않았다. 다만 현재 누적 실험에서는 변곡점 규칙이 기본 patience 규칙을 "
            "일관되게 넘어서지는 못했다. 논문 본문에서는 standard patience를 기본 메커니즘으로 두고, appendix나 ablation에서 "
            "reversal variant를 비교하는 구성이 가장 자연스럽다."
        ),
        heading(2, "7. 제안하는 Early Stopping 메커니즘"),
        paragraph(
            [
                "각 평가 epoch t에서 ",
                ("r_t = EMA_0.3(train_loss_t / (train_normal_loss_t + eps))", "code"),
                "를 계산한다. t가 ",
                ("teacher_only_warmup_epochs", "code"),
                "보다 작으면 early stopping 판단을 하지 않는다.",
            ]
        ),
        bullet("목표 방향: r_t 최소화"),
        bullet("개선 판정: 기존 best보다 0.5% 이상 상대 개선되면 patience counter를 reset"),
        bullet("Patience: 3"),
        bullet("Stop 시 checkpoint: stop epoch 자체가 아니라 stop 이전까지 r_t가 가장 좋았던 checkpoint"),
        paragraph(
            "이 기준의 직관은 명확하다. warmup 이후 전체 학습 목적이 정상 구간 손실에 비해 더 이상 효율적으로 개선되지 않는다면, "
            "student가 합류한 뒤의 추가 학습이 anomaly-detection 성능으로 전이될 가능성이 낮다고 본다. 동시에 feature별 통계나 "
            "test-set score를 쓰지 않으므로 paper-level stopping rule로 설명하기 쉽다."
        ),
        heading(2, "8. 해석상 주의점과 다음 단계"),
        bullet(
            "이번 결과는 현재 누적된 ablation 실험 분포에 대해 최적화되어 있다. 최종 paper/default rule로 고정하기 전에는 experiment ID 기준 hold-out 검증이 필요하다."
        ),
        bullet(
            "score-oriented 후보는 평균 점수만 보면 유리하지만, anomaly-labeled student reconstruction에 더 직접적으로 의존한다. 논문 본문에서는 보조 분석 또는 ablation으로 두는 편이 안전하다."
        ),
        bullet(
            "2차 refined sweep은 ratio(train_loss, train_normal_loss), ratio(train_loss, train_student_recon_anomaly), relgap(train_loss, train_student_recon_anomaly) 세 family에 한정해 patience 2/3/5와 threshold 0.25%/0.5%/1.0%를 더 조밀하게 보는 것이 효율적이다."
        ),
        heading(2, "최종 판단"),
        paragraph(
            [
                "현재 데이터만 놓고 보면, 가장 설득력 있는 early stopping 기준은 ",
                ("warmup 이후 EMA(train_loss / train_normal_loss) 최소화, patience 3, relative threshold 0.5%", "bold"),
                "다. 이 규칙은 feature aggregate를 쓰지 않고, test label score를 참조하지 않으며, student 합류 이후의 train dynamics만으로 결정된다. "
                "평균 성능도 fixed warmup baseline 대비 명확히 높다.",
            ]
        ),
    ]
    return blocks


def append_blocks(page_id: str, blocks: list[dict]) -> None:
    for i in range(0, len(blocks), 80):
        notion_request("PATCH", f"/blocks/{page_id}/children", {"children": blocks[i : i + 80]})


def main() -> None:
    notion_request("GET", f"/pages/{PARENT_PAGE_ID}")
    title = "Early Stopping 기준 재검토: train-metric interaction sweep"
    page = notion_request(
        "POST",
        "/pages",
        {
            "parent": {"page_id": PARENT_PAGE_ID},
            "properties": {
                "title": {
                    "title": [
                        {
                            "type": "text",
                            "text": {"content": title},
                        }
                    ]
                }
            },
        },
    )
    append_blocks(page["id"], build_blocks())
    print(json.dumps({"id": page["id"], "url": page["url"], "title": title}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
