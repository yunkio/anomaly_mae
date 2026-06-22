"""Publish the strict train-history early-stopping report to Notion."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
from refresh_notion_early_stopping_report_v2 import (  # noqa: E402
    PAGE_ID,
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


OUT = Path("temp/early_stopping_strict_train_scalar_4ds")
DATASETS = ["SWaT_excl22", "PSM", "WaDi_A1", "WaDi_A2"]
DATASET_LABELS = {
    "SWaT_excl22": "SWaT(excl22)",
    "PSM": "PSM",
    "WaDi_A1": "WaDi A1",
    "WaDi_A2": "WaDi A2",
}


def fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def fmt_epoch(value: float | int | None) -> str:
    if value is None:
        return "-"
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
    if metric.startswith("pair_absdiff__"):
        a, b = metric.removeprefix("pair_absdiff__").split("__", 1)
        return f"absdiff({a}, {b})"
    return metric


def rule_text(c: dict) -> str:
    return (
        f"{c['transform']} / {c['direction_mode']} / {c['rule']} / "
        f"P={int(c['patience'])} / {c['threshold_type']}={float(c['threshold_value']):g} / "
        f"{c['start_policy']} / rollback={c['rollback']}"
    )


def perf_cell(payload: dict | None) -> str:
    if not payload:
        return "-"
    return f"{payload['score']:.4f} (#{payload['rank']}, e{payload['epoch']})"


def model_rank_rows(table_payload: dict) -> list[list[str]]:
    rows = [["Avg rank", "Model", "Cov.", "SWaT(excl22)", "PSM", "WaDi A1", "WaDi A2"]]
    for row in table_payload["rows"]:
        avg_rank = "-" if row["avg_rank"] is None else f"{row['avg_rank']:.2f}"
        ds = row["datasets"]
        rows.append(
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
    return rows


def criterion_summary_row(label: str, c: dict) -> list[str]:
    return [
        label,
        pretty_metric(c["metric"]),
        rule_text(c),
        fmt(c["mean_score_4ds"]),
        fmt(c["mean_drop_4ds"]),
        fmt_epoch(c["mean_stop_epoch_4ds"]),
    ]


def source_rows(audit: dict) -> list[list[str]]:
    rows = [["Key", "Source status in 177 cells", "Non-constant", "Decision"]]
    for key in audit["included_train_scalar_keys"]:
        rec = audit["keys"][key]
        rows.append(
            [
                key,
                f"{rec['scalar_cells']}/{audit['n_cells']} scalar epoch series",
                f"{rec['nonconstant_cells']}/{audit['n_cells']}",
                "include",
            ]
        )
    for key in ("train_d_loss", "train_d_real_acc", "train_d_fake_acc", "train_adv_loss", "train_adaptive_lambda"):
        rec = audit["keys"].get(key)
        if not rec:
            continue
        rows.append(
            [
                key,
                f"key present, empty list {rec['empty_list_cells']}/{audit['n_cells']}",
                "0/177",
                "not usable in these cells",
            ]
        )
    return rows


def fixed_rows(fixed: list[dict]) -> list[list[str]]:
    rows = [["Policy", "Mean score", "Mean drop", "SWaT", "PSM", "WaDi A1", "WaDi A2"]]
    for row in fixed:
        rows.append(
            [
                row["policy"],
                fmt(row["mean_score_4ds"]),
                fmt(row["mean_drop_4ds"]),
                fmt(row["SWaT_excl22_mean_score"]),
                fmt(row["PSM_mean_score"]),
                fmt(row["WaDi_A1_mean_score"]),
                fmt(row["WaDi_A2_mean_score"]),
            ]
        )
    return rows


def pf_variant_rows(variants: list[dict]) -> list[list[str]]:
    rows = [["Variant", "Metric", "Transform", "Patience", "Mean score", "Drop", "Mean stop"]]
    for row in variants:
        rows.append(
            [
                row["label"],
                pretty_metric(row["metric"]),
                row["transform"],
                str(int(row["patience"])),
                fmt(row["mean_score_4ds"]),
                fmt(row["mean_drop_4ds"]),
                fmt_epoch(row["mean_stop_epoch_4ds"]),
            ]
        )
    return rows


def build_blocks() -> list[dict]:
    analysis = json.loads((OUT / "analysis_summary.json").read_text())
    summary = analysis["summary"]
    source_audit = analysis["history_source_audit"]
    rank_payload = json.loads((OUT / "top5_model_rank_tables.json").read_text())

    top_summary = [["ID", "Criterion", "Rule", "Mean score", "Drop", "Mean stop"]]
    for idx, c in enumerate(rank_payload["criteria"], start=1):
        top_summary.append(criterion_summary_row(f"C{idx}", c))
    pf_v2 = rank_payload["criteria"][4]
    pf_current = rank_payload["paper_friendly_criterion"]
    pf_lasad = rank_payload["paper_friendly_lasad_criterion"]
    top_summary.append(criterion_summary_row("PF-LASAD", pf_lasad))
    top_summary.append(criterion_summary_row("PF-current", pf_current))

    performance_toggles = []
    for table_payload in rank_payload["tables"]:
        c = table_payload["criterion"]
        title = (
            f"C{table_payload['index']} · {pretty_metric(c['metric'])} · "
            f"mean={fmt(c['mean_score_4ds'])}, drop={fmt(c['mean_drop_4ds'])}"
        )
        performance_toggles.append(
            toggle(
                title,
                [
                    callout(
                        [
                            "각 셀은 ",
                            ("성능 (#데이터셋 내 순위, eepoch)", "code"),
                            " 형식이다. 성능은 사후 비교용이며, early stopping criterion 입력에는 사용하지 않았다.",
                        ],
                        "blue_background",
                    ),
                    table(model_rank_rows(table_payload)),
                ],
            )
        )

    performance_toggles.append(
        toggle(
            "PF-LASAD · relgap(train_student_recon_anomaly, train_normal_loss) · 논문형 권고 성능표",
            [
                callout(
                    [
                        "LASAD 논문 구조에 맞춘 label-conditioned suppression gap이다. ",
                        ("train_normal_loss", "code"),
                        "는 label-normal patch의 Teacher-Student normal imitation 안정성, ",
                        ("train_student_recon_anomaly", "code"),
                        "는 label-anomaly patch에서 Student가 anomaly를 재구성하지 못하는 suppression proxy로 해석한다.",
                    ],
                    "green_background",
                ),
                table(model_rank_rows(rank_payload["paper_friendly_lasad_table"])),
            ],
        )
    )

    performance_toggles.append(
        toggle(
            "PF-current · ratio(train_loss, train_normal_loss) · 기존 후보 성능표",
            [
                callout(
                    "기존 PF 후보는 비교용으로 유지한다. 최종 권고는 이 기준을 PF-LASAD로 교체하는 것이다.",
                    "yellow_background",
                ),
                table(model_rank_rows(rank_payload["paper_friendly_table"])),
            ],
        )
    )

    blocks = [
        heading(1, "Early Stopping 기준 재분석: LASAD-aligned PF & train-history audit"),
        callout(
            [
                "핵심 정정: ",
                ("train_anomaly_loss", "code"),
                "는 누락 지표가 아니다. ",
                ("loss.py → trainer.py history → training_histories.json → learning_curve.png", "code"),
                " 경로로 per-epoch 수집된다. 이번 재계산에는 해당 지표를 포함했다.",
            ],
            "red_background",
        ),
        paragraph(
            "이 문서는 SWaT(excl22), PSM, WaDi A1, WaDi A2의 271번 이후 실험군을 대상으로, 훈련 데이터에서 수집 가능한 scalar metric만 사용해 early stopping 기준을 다시 전수 조사한 결과다."
        ),
        toc(),
        divider(),
        heading(2, "결론"),
        bullet(
            [
                "최고 평균 성능 기준은 ",
                ("relgap(train_mean_discrepancy, train_teacher_recon_anomaly)", "code"),
                " + EMA 0.1 + peak reversal + P=2다. 4-dataset 평균 성능은 ",
                (fmt(rank_payload["criteria"][0]["mean_score_4ds"]), "code"),
                ", best epoch 대비 평균 drop은 ",
                (fmt(rank_payload["criteria"][0]["mean_drop_4ds"]), "code"),
                "다.",
            ]
        ),
        bullet(
            [
                "LASAD 논문 기준의 권고 PF는 ",
                ("PF-LASAD: relgap(train_student_recon_anomaly, train_normal_loss)", "code"),
                " + EMA 0.1 + standard + P=2다. 평균 성능 ",
                (fmt(pf_lasad["mean_score_4ds"]), "code"),
                ", drop ",
                (fmt(pf_lasad["mean_drop_4ds"]), "code"),
                ", 평균 stop epoch ",
                (fmt_epoch(pf_lasad["mean_stop_epoch_4ds"]), "code"),
                "이며, 최고 성능 기준 대비 희생폭은 작고 기존 PF-current보다 우세하다.",
            ]
        ),
        bullet(
            [
                ("train_anomaly_loss", "code"),
                "는 177/177 cell에 존재하지만 non-constant는 8/177이다. LASAD는 anomaly patch를 OD imitation loss에서 제외하는 구조이므로, 이 값이 0/constant인 것은 누락이 아니라 설계 결과에 가깝다. 따라서 audit에는 포함하되 PF의 핵심 신호로 쓰지 않는다.",
            ]
        ),
        bullet(
            "PRC/AUC/F1/PAK/VUS/Affiliation 등 성능 지표는 criterion 입력에서 제외했다. 표의 성능값은 선택된 epoch를 사후 평가하기 위한 비교값이다.",
        ),
        heading(2, "Source Audit: 누락 여부를 판단한 기준"),
        paragraph(
            "이번부터는 파생 파일 존재 여부가 아니라 원천 수집 경로를 기준으로 판단했다. 원천은 train loop가 기록한 training_histories.json이며, 여기서 train-only scalar인지, epoch 길이와 맞는지, 실제 변화가 있는지를 분리해 확인했다."
        ),
        table(source_rows(source_audit)),
        callout(
            [
                ("learning_curve.png", "code"),
                "는 ",
                ("BestModelVisualizer.plot_learning_curve(history)", "code"),
                "가 ",
                ("training_histories.json", "code"),
                "의 per-epoch history를 받아 그린다. Discrepancy panel은 ",
                ("train_normal_loss", "code"),
                ", ",
                ("train_anomaly_loss", "code"),
                ", ",
                ("train_disc_loss", "code"),
                "를 사용한다.",
            ],
            "blue_background",
        ),
        toggle(
            "코드 경로 확인",
            [
                bullet([("mae_anomaly/loss.py", "code")]),
                paragraph(
                    [
                        "loss_dict에 ",
                        ("anomaly_loss", "code"),
                        ", ",
                        ("normal_loss", "code"),
                        ", ",
                        ("mean_discrepancy", "code"),
                        ", teacher/student recon split metric을 넣는다.",
                    ]
                ),
                bullet([("mae_anomaly/trainer.py", "code")]),
                paragraph(
                    [
                        "history 초기화 후 매 epoch ",
                        ("train_anomaly_loss.append(epoch_losses['anomaly_loss'])", "code"),
                        "를 수행한다.",
                    ]
                ),
                bullet([("mae_anomaly/visualization/best_model_visualizer.py", "code")]),
                paragraph(
                    [
                        ("plot_learning_curve(history)", "code"),
                        "가 history의 ",
                        ("train_anomaly_loss", "code"),
                        "를 직접 plot한다.",
                    ]
                ),
            ],
        ),
        heading(2, "포함/배제 규칙"),
        table(
            [
                ["Category", "Decision", "Reason"],
                ["train_* scalar history", "include", "train loop에서 수집되고 label 사용 조건을 만족"],
                ["train_anomaly_loss", "include", "177/177 존재. 단 non-constant 8/177이라 해석 시 별도 주의"],
                ["train_d_* / train_adv_loss", "not used in this sweep", "key는 있으나 177개 기준 cell에서 empty list라 epoch series가 아님"],
                ["epoch_*", "exclude", "trainer 주석 및 callback 경로상 test/eval sample-type contribution history"],
                ["train_feature_* / train_fm_*", "exclude", "feature/FM 관련 지표는 사용자 지시에 따라 paper-level criterion에서 배제"],
                ["train_grl_* / train_scad_*", "exclude", "사용자 지시에 따라 GRL/SCAD metric 배제"],
                ["PRC/AUC/F1/PAK/VUS/Affiliation", "exclude as input", "성능/threshold metric이며 train-time stopping rule로 부적절"],
            ]
        ),
        heading(2, "Metric 정의와 stopping semantics"),
        table(
            [
                ["Term", "Definition"],
                ["ratio(A,B)", "A / (abs(B) + eps)"],
                ["relgap(A,B)", "(A - B) / (abs(A) + abs(B) + eps)"],
                ["diff(A,B)", "A - B"],
                ["absdiff(A,B)", "abs(A - B)"],
                ["EMA alpha", "y_t = alpha*x_t + (1-alpha)*y_{t-1}; raw, 0.1, 0.2, 0.3, 0.5, 0.7 전수 조사"],
                ["direction=auto", "ratio/relgap/gap/separation은 max, loss/recon/disc/discrepancy는 min으로 해석"],
                ["standard", "best 값이 threshold 이상 개선되지 않는 eval point가 P회 누적되면 trigger"],
                ["peak_reversal", "peak 이후 threshold 이상 하락이 P회 누적되면 변곡점으로 판단"],
                ["rollback", "trigger epoch에서 성능을 읽지 않고, trigger 전까지의 best criterion epoch로 되돌려 평가"],
                ["post_warmup", "warmup+1 epoch부터 stopping candidate를 허용"],
            ]
        ),
        callout(
            [
                ("train_loss", "code"),
                "는 history상 ",
                ("loss_dict['total_loss']", "code"),
                " 평균이다. warmup에서는 reconstruction only이고, post-warmup에서는 reconstruction + discrepancy 계열로 의미가 바뀐다. 이 때문에 논문형 기준에서는 ",
                ("train_loss", "code"),
                "보다 label-normal discrepancy와 label-anomaly Student reconstruction의 상호작용을 선호한다.",
            ],
            "yellow_background",
        ),
        heading(2, "LASAD 논문 기준 PF 재판단"),
        paragraph(
            "LASAD 논문은 label-guided adversarial suppression을 핵심으로 두고, Teacher-Student discrepancy는 그 결과가 읽히는 inherited substrate로 둔다. 따라서 paper-friendly early stopping은 discrepancy 자체를 headline으로 세우기보다, label이 만든 normal/anomaly 조건부 학습 상태를 읽어야 한다."
        ),
        table(
            [
                ["Candidate", "Criterion", "Mean score", "Drop", "Mean stop", "Judgment"],
                [
                    "PF-LASAD",
                    pretty_metric(pf_lasad["metric"]),
                    fmt(pf_lasad["mean_score_4ds"]),
                    fmt(pf_lasad["mean_drop_4ds"]),
                    fmt_epoch(pf_lasad["mean_stop_epoch_4ds"]),
                    "권고: label-normal imitation 안정성과 label-anomaly suppression proxy를 함께 사용",
                ],
                [
                    "Top performance",
                    pretty_metric(rank_payload["criteria"][0]["metric"]),
                    fmt(rank_payload["criteria"][0]["mean_score_4ds"]),
                    fmt(rank_payload["criteria"][0]["mean_drop_4ds"]),
                    fmt_epoch(rank_payload["criteria"][0]["mean_stop_epoch_4ds"]),
                    "성능은 최고지만 discrepancy substrate가 전면에 서고 논문 설명성이 약함",
                ],
                [
                    "PF-v2 previous",
                    pretty_metric(pf_v2["metric"]),
                    fmt(pf_v2["mean_score_4ds"]),
                    fmt(pf_v2["mean_drop_4ds"]),
                    fmt_epoch(pf_v2["mean_stop_epoch_4ds"]),
                    "성능은 좋지만 anomaly label 정보를 직접 살리지 못함",
                ],
                [
                    "PF-current",
                    pretty_metric(pf_current["metric"]),
                    fmt(pf_current["mean_score_4ds"]),
                    fmt(pf_current["mean_drop_4ds"]),
                    fmt_epoch(pf_current["mean_stop_epoch_4ds"]),
                    "train_loss 의미 변화 때문에 배제",
                ],
            ]
        ),
        callout(
            [
                "PF-LASAD 수식: ",
                ("G_e = relgap(train_student_recon_anomaly, train_normal_loss)", "code"),
                ". 직관적으로는 label-normal patch에서 Student가 Teacher를 충분히 따라가는 정상 기준선을 유지하면서, label-anomaly patch에서는 Student reconstruction이 normal imitation 기준선과 충분히 분리되는지를 본다.",
            ],
            "green_background",
        ),
        heading(2, "Strict sweep 설정"),
        table(
            [
                ["Item", "Value"],
                ["Datasets", "SWaT(excl22), PSM, WaDi A1, WaDi A2"],
                ["Cells", str(summary["n_cells_by_dataset"])],
                ["Warmup distribution", str(summary["warmup_distribution"])],
                ["Input keys", ", ".join(summary["included_train_scalar_candidate_keys"])],
                ["Criteria total", str(summary["n_criteria_total"])],
                ["Full-coverage criteria", str(summary["n_criteria_full_coverage"])],
                ["Post-warmup full-coverage criteria", str(summary["n_criteria_post_warmup_full_coverage"])],
            ]
        ),
        heading(2, "Top Criteria Summary"),
        table(top_summary),
        heading(2, "Fixed Baselines"),
        paragraph("고정 epoch baseline은 50, 100, 150, 200, 250, 300, 350, 400, 450, 500 및 warmup epoch를 모두 다시 계산했다."),
        table(fixed_rows(analysis["fixed_epoch_baselines"])),
        heading(2, "PF-current ablation"),
        paragraph(
            "아래 표는 기존 PF-current에 대해 EMA와 patience를 제거했을 때의 비교다. 결론적으로 raw/no-patience 조합은 안정적으로 개선되지 않았고, PF-current 자체도 PF-LASAD보다 낮다."
        ),
        table(pf_variant_rows(analysis["paper_friendly_variants"])),
        heading(2, "모델별 성능표"),
        paragraph(
            "아래 toggle들은 상위 5개 criterion, PF-LASAD, 기존 PF-current에 대해 각 numeric experiment가 네 데이터셋에서 얻는 선택 epoch 성능과 순위를 정리한다."
        ),
        *performance_toggles,
        heading(2, "최종 판단"),
        callout(
            [
                "paper preference는 PF-LASAD로 변경하는 것이 낫다. 기존 PF-current는 ",
                ("train_loss", "code"),
                "의 warmup/post-warmup 의미 변화 때문에 약하고, 이전 PF-v2는 anomaly label 정보를 직접 살리지 못한다. PF-LASAD는 LASAD의 label-guided suppression 논리와 맞고 성능 희생도 작다.",
            ],
            "green_background",
        ),
        code_block(
            "Repro commands:\n"
            "conda run -n dc_vis python -u scripts/audit_strict_train_history_sources_4ds.py\n"
            "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 conda run -n dc_vis python -u scripts/early_stopping_strict_train_scalar_sweep_4ds.py --workers 3\n"
            "conda run -n dc_vis python -u scripts/build_early_stopping_strict_rank_tables.py\n"
            "conda run -n dc_vis python -u scripts/summarize_early_stopping_strict_train_scalar_4ds.py",
            "bash",
        ),
    ]
    return blocks


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
                            "text": {
                                "content": "Early Stopping 기준 재분석: LASAD-aligned PF & train-history audit"
                            },
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
    blocks = build_blocks()
    append_blocks(PAGE_ID, blocks)
    time.sleep(0.5)
    page = notion_request("GET", f"/pages/{PAGE_ID}")
    children = notion_request("GET", f"/blocks/{PAGE_ID}/children?page_size=100")
    print(
        json.dumps(
            {
                "id": PAGE_ID,
                "url": page.get("url"),
                "top_level_blocks": len(children.get("results", [])),
                "title": "Early Stopping 기준 재분석: LASAD-aligned PF & train-history audit",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
