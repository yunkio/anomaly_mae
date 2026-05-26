"""
Build Notion content blocks for the Exp 119-290 page update.

Outputs (as text files in temp/):
- temp/notion_group_n_section.txt — replacement for Group N section
- temp/notion_section_21.txt — replacement for Section 2.1
- temp/notion_section_22.txt — replacement for Section 2.2
- temp/notion_section_4_table_replacements.txt — column splits for Section 4

Uses: temp/notion_exp_results.json (aggregated).
"""
import json
from pathlib import Path

RESULTS = json.loads(Path("/home/ykio/notebooks/claude/temp/notion_exp_results.json").read_text())

# Order of groups
GROUPS = [
    ("A", "Mechanism Ablation (119-133, 50ep, offset=False)", "base : 119", 119, 133),
    ("B", "Depth/Epoch on FM+OD (134-139, 50ep)", "base : 125", 134, 139),
    ("C", "Epoch Scaling + GRL×Depth (140-148, 100ep)", "base : 134", 140, 148),
    ("D", "Offset=True 보정 (149-152)", None, 149, 152),
    ("E", "최적 조합 탐색 (153-164, 200ep+, offset=True)", "base : 150", 153, 164),
    ("F", "GRL 검증 (165-172, 200ep, offset=True)", None, 165, 172),
    ("G", "sd=1 GRL + 추가 검증 (173-184)", None, 173, 184),
    ("H", "GRL 최적 조합 (185-217, 200ep, offset=True)", "base : 150", 185, 217),
    ("I", "sd=1 Loss/Arch 조합 (218-239)", "base : 150", 218, 239),
    ("J", "GRL Performance Optimization (240-257, 200-300ep)", None, 240, 257),
    ("K", "Inference & Freeze Ablation (258-263, 200ep)", None, 258, 263),
    ("L", "exp247 Extension + GRL-focused v3 (264-277)", None, 264, 277),
    ("N", "Post-Group M 274-focused Single-axis (278-284, 모두 완료)",
     "fair baseline : exp284 (274 base + ep=300/w=150)", 278, 284),
    ("P", "271-focused Single-axis (285-302, 모두 TBD)",
     "baseline : 271 — ep=500/w=250 유지", 285, 302),
]

# Top-N medals (rank → marker)
MEDALS = {1: '🥇', 2: '🥈', 3: '🥉',
          4: '4️⃣', 5: '5️⃣', 6: '6️⃣', 7: '7️⃣', 8: '8️⃣', 9: '9️⃣', 10: '🔟'}

# TBD exps: skip data fetch, render as —
TBD_EXPS_SET = set(str(i) for i in range(285, 303))

# Annotations for top performers
def f1_marker(exp):
    r = RESULTS.get(exp)
    if not r: return ""
    ra = r.get("rank_avg", {}).get("pak_f1")
    if ra is None: return ""
    return ""

def fmt(v, prec=3):
    if v is None or v == "":
        return "—"
    if isinstance(v, (int, float)):
        if isinstance(v, float):
            return f"{v:.{prec}f}"
        return str(v)
    return str(v)

def fmt_score_ep(node):
    """Return 'pak_f1 (ep)' format."""
    if node is None: return "—"
    f1 = node.get("pak_f1")
    ep = node.get("best_ep")
    if f1 is None: return "—"
    if ep is None:
        return f"{f1:.3f}"
    return f"{f1:.3f} ({ep})"

def fmt_prc_ep(node):
    if node is None: return "—"
    p = node.get("pak_prc")
    ep = node.get("best_ep")
    if p is None: return "—"
    if ep is None:
        return f"{p:.3f}"
    return f"{p:.3f} ({ep})"


# ============================================================
# Group N spec table (K-style 6 col)
# ============================================================
group_n_specs = [
    # exp, name, base, override_vs_base, baseline, 검증 요소
    ("278", "274_no_balanced", "274",
     "`grl_balanced_sampling=False`, ep=300/w=150",
     "284",
     "Balanced sampling 제거 효과 (Group M Top-3 후속, Group N ep 환경)"),
    ("279", "274_fm_adaptive", "274",
     "`fm_adaptive_lambda=True`, ep=300/w=150",
     "284",
     "FM adaptive lambda 도입 효과"),
    ("280", "274_cls_lr_025", "274",
     "`grl_cls_lr_ratio=0.25` (0.1→0.25), ep=300/w=150",
     "284",
     "GRL classifier LR 증가 효과 (sensitivity 측정)"),
    ("281", "274_cls_2layer", "274",
     "`grl_cls_arch=2layer` (1→2 layer), ep=300/w=150",
     "284",
     "GRL classifier 깊이 효과 (capacity 측정)"),
    ("282", "274_normal_w3", "274",
     "`normal_loss_weight=3.0` (1.0→3.0), ep=300/w=150",
     "284",
     "Normal reconstruction weight 증가 효과 (recon emphasis)"),
    ("283", "274_no_focal", "274",
     "`grl_use_focal=False`, ep=300/w=150",
     "284",
     "Focal loss 제거 효과 (BCE classifier)"),
    ("284", "274_ep300_anchor", "274",
     "ep=300/w=150 (다른 변경 없음)",
     "274 (원본 ep=500)",
     "Group N fair-ep anchor / 274의 ep=500 vs ep=300 효과 측정"),
    ("285", "274_fm_adaptive_pure", "274",
     "`fm_adaptive_lambda=True` (ep=500 유지)",
     "274",
     "**(TBD, 미실행)** Pure single-axis FM adaptive — 274 원본 ep에서 fm_adaptive 단독 효과 측정 (279와 ep 차이로 비교)"),
]

def build_group_n_section():
    lines = []
    lines.append("## Group N: Post-Group M GRL-focused Single-axis (Exp 278-285, **ep=300/w=150 통일**, 285 placeholder)")
    lines.append("<callout>")
    lines.append("\t**Base**: 모든 항목 Exp 274 ablation (Group M Top-3 — 274 🥇 / 271 🥈 / 273 🥉).")
    lines.append("\t**목적**: 274 champion 주변 single-axis isolation. 각 항목은 274의 하나의 hyperparameter만 변경.")
    lines.append("\t**ep=300, warmup=150** 통일 (Group N convention). 274 원본은 ep=500/w=250이므로 직접 비교는 ep confounding 발생 → **fair baseline = exp284** (274 + ep=300/w=150).")
    lines.append("\t**예외**: **Exp 284**는 anchor 역할 (baseline=274 원본). **Exp 285**는 ep=500 유지하여 274의 원본 환경에서 fm_adaptive_lambda 단독 효과 측정.")
    lines.append("\t**공통 GRL settings**: `use_grl=True`, `use_feature_matching=True`, `normalize_mode='minmax'`, `anomaly_interval_scale=0.75`, `fm_distance_metric='l2'`, `dynamic_margin_k=6`.")
    lines.append("</callout>")
    lines.append("<table>")
    lines.append("<tr>")
    for h in ["Exp", "Name", "Base", "Override (vs Base)", "Baseline (fair)", "검증 요소"]:
        lines.append(f"<td>{h}</td>")
    lines.append("</tr>")
    for spec in group_n_specs:
        exp, name, base, override, baseline, purpose = spec
        lines.append("<tr>")
        lines.append(f"<td>**{exp}**</td>")
        lines.append(f"<td>{name}</td>")
        lines.append(f"<td>{base}</td>")
        lines.append(f"<td>{override}</td>")
        lines.append(f"<td>{baseline}</td>")
        lines.append(f"<td>{purpose}</td>")
        lines.append("</tr>")
    lines.append("</table>")
    lines.append("---")
    return "\n".join(lines) + "\n"


# ============================================================
# Section 2.1 (PAK_AUC_F1) — full table reconstruction
# ============================================================
RANK_DATASETS = ["swat_excl22", "wadi_A1", "wadi_A2", "smd_15"]

def build_results_table(metric_name, section_title, callout_text):
    """metric_name = 'pak_f1' or 'pak_prc'."""
    lines = []
    lines.append(f"## {section_title}")
    lines.append("<callout>")
    for line in callout_text.split("\n"):
        lines.append(f"\t{line}")
    lines.append("</callout>")
    lines.append("<table>")
    # Header
    header = ["Exp", "Simulation", "Swat(full)", "Swat(excl22)*",
              "Wadi A1*", "Wadi A2*", "SMD(full)", "SMD(15)*",
              "Avg(4*)", "RankAvg(4*)"]
    lines.append("<tr>")
    for h in header:
        lines.append(f"<td>{h}</td>")
    lines.append("</tr>")

    # Helper to get value for row
    def row_cell(exp, ds, metric_name):
        if exp in TBD_EXPS_SET:
            return "—"
        r = RESULTS.get(exp)
        if r is None: return "—"
        node = r.get(ds)
        if node is None: return "—"
        val = node.get(metric_name)
        ep = node.get("best_ep")
        if val is None: return "—"
        if ep is None:
            return f"{val:.3f}"
        # ep is int for per-DS, float (mean) for smd aggregates
        ep_str = f"{int(round(ep))}"
        return f"{val:.3f} ({ep_str})"

    def avg_cell(exp, metric_name):
        if exp in TBD_EXPS_SET: return "—"
        r = RESULTS.get(exp)
        if r is None: return "—"
        v = r.get("avg", {}).get(metric_name)
        return f"{v:.3f}" if v is not None else "—"

    def rank_cell(exp, metric_name):
        if exp in TBD_EXPS_SET: return "—"
        r = RESULTS.get(exp)
        if r is None: return "—"
        v = r.get("rank_avg", {}).get(metric_name)
        return f"{v:.2f}" if v is not None else "—"

    # Top-10 by RA — medals for the metric
    valid_ras = [(e, r["rank_avg"][metric_name]) for e, r in RESULTS.items()
                 if e not in TBD_EXPS_SET and r.get("rank_avg", {}).get(metric_name) is not None]
    valid_ras.sort(key=lambda x: x[1])
    top10 = {valid_ras[i][0]: MEDALS[i+1] for i in range(min(10, len(valid_ras)))}

    # Build rows
    for letter, label, basenote, lo, hi in GROUPS:
        # Group separator row
        sep_text = f"**Group {letter}: {label}"
        if basenote:
            sep_text += f"<br>{basenote}"
        sep_text += "**"
        lines.append("<tr>")
        # Span the separator across all cols by repeating empty cells with the bold text in col1
        lines.append(f"<td>{sep_text}</td>")
        for _ in range(len(header) - 1):
            lines.append("<td></td>")
        lines.append("</tr>")

        for n in range(lo, hi + 1):
            exp = str(n)
            if exp not in RESULTS and exp not in TBD_EXPS_SET:
                continue  # Skip missing (e.g., never-run)
            # Exp cell — add medal if top10
            if exp in top10:
                medal = top10[exp]
                rank_num = list(top10).index(exp) + 1
                if rank_num <= 3:
                    exp_cell = f"**{exp}** {medal}"
                else:
                    exp_cell = f"{exp} {medal}"
            else:
                exp_cell = exp
            lines.append("<tr>")
            lines.append(f"<td>{exp_cell}</td>")
            for ds in ["simulation", "swat_full", "swat_excl22",
                       "wadi_A1", "wadi_A2", "smd_full", "smd_15"]:
                lines.append(f"<td>{row_cell(exp, ds, metric_name)}</td>")
            lines.append(f"<td>{avg_cell(exp, metric_name)}</td>")
            lines.append(f"<td>{rank_cell(exp, metric_name)}</td>")
            lines.append("</tr>")

    lines.append("</table>")
    return "\n".join(lines) + "\n"


# ============================================================
# Section 4 (PSM Leaderboard) — add SMD(15) column to existing
# ============================================================
# This will be done as a regex-style targeted edit. We don't rebuild the full table — we'll generate the SMD(15) value to add to each row.

def get_smd15_for_section4():
    """Return dict: exp_str → '0.NNN' for SMD(15) pak_f1 and pak_prc."""
    out_f1, out_prc = {}, {}
    for exp, r in RESULTS.items():
        s15 = r.get("smd_15")
        if s15:
            if s15.get("pak_f1") is not None:
                out_f1[exp] = f"{s15['pak_f1']:.3f}"
            if s15.get("pak_prc") is not None:
                out_prc[exp] = f"{s15['pak_prc']:.3f}"
    return out_f1, out_prc


# ============================================================
# Main
# ============================================================
def main():
    out_dir = Path("/home/ykio/notebooks/claude/temp")
    out_dir.mkdir(exist_ok=True)

    # Group N section
    group_n = build_group_n_section()
    (out_dir / "notion_group_n_section.txt").write_text(group_n)
    print(f"Group N section: {len(group_n)} chars")

    # Section 2.1
    callout_21 = (
        "**PAK_AUC_F1 기준 전체 결과** (Exp 119-285, 285 미실행).\n"
        "**컬럼 정의**:\n"
        "- 셀 표시: `pak_f1 (best_ep)`. SMD aggregates는 epoch 평균 처리 불가 — pak_f1만 표시.\n"
        "- **별표(*) 표시 컬럼**: Swat(excl22), Wadi A1, Wadi A2, SMD(15) — **RankAvg 4-DS 계산에 사용**.\n"
        "- **Avg(4*)** = 4 별표 DS의 pak_f1 평균.\n"
        "- **RankAvg(4*)** = 4 별표 DS 각각의 등수의 평균 (낮을수록 좋음). 165 valid exp 기준.\n"
        "- **SMD(15)**: TimeSeAD (Wagner et al., 2023) 권장 15-machine subset (excludes machine-1-1/1-3/1-4/1-5/1-6/1-8/2-5/2-8/3-4/3-5/3-7/3-10/3-11)."
    )
    section_21 = build_results_table("pak_f1", "2.1 PAK_AUC_F1 기준 결과", callout_21)
    (out_dir / "notion_section_21.txt").write_text(section_21)
    print(f"Section 2.1: {len(section_21)} chars, {section_21.count('<tr>')} rows")

    # Section 2.2
    callout_22 = (
        "**PAK_AUC_PRC 기준 전체 결과** (Exp 119-285, 285 미실행).\n"
        "**컬럼 정의**: 2.1과 동일. PRC = Area under Precision-Recall curve.\n"
        "- 별표(*) 표시 컬럼은 RankAvg 계산에 사용 (Swat excl22, Wadi A1/A2, SMD(15)).\n"
        "- best_ep는 PAK_AUC_F1 기준 (training_histories.json의 best_epoch). PRC는 그 epoch에서의 값."
    )
    section_22 = build_results_table("pak_prc", "2.2 PAK_AUC_PRC 기준 결과", callout_22)
    section_22 = section_22 + "---\n"  # 2.2 ends with --- separator before # 4.
    (out_dir / "notion_section_22.txt").write_text(section_22)
    print(f"Section 2.2: {len(section_22)} chars, {section_22.count('<tr>')} rows")

    # SMD(15) extracts for Section 4 — for adding new column
    smd15_f1, smd15_prc = get_smd15_for_section4()
    sec4_data = {
        "smd_15_pak_f1": smd15_f1,
        "smd_15_pak_prc": smd15_prc,
    }
    (out_dir / "notion_section_4_smd15.json").write_text(json.dumps(sec4_data, indent=2))
    print(f"SMD(15) extracted for {len(smd15_f1)} experiments")

    print("\nAll content built. Files:")
    for f in sorted(out_dir.glob("notion_*")):
        print(f"  {f}")


if __name__ == "__main__":
    main()
