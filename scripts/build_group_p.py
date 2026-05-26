"""Build Group P section (271-based ablations) for Spec page.

Group P = 271-based single/multi-axis ablations.
IDs preserved from original 274-based plan (285-304) MINUS 285 and 288 (which collapse on 271 base).
"""
from pathlib import Path

# 18 entries (rebased to 271), sequential IDs 285-302 (no gaps)
GROUP_P_SPECS = [
    # (exp, name, override_vs_271, 검증요소)
    ("285", "271_unmask",
     "`force_mask_anomaly=False` (ep=500 유지)",
     "**(TBD, 미실행)** Pure single-axis unmask — 271 base에서 force_mask_anomaly 단독 효과 측정"),
    ("286", "271_no_focal",
     "`grl_use_focal=False` (ep=500 유지)",
     "**(TBD, 미실행)** Focal off — 271은 balanced=False라 focal+balanced 중복 없이 순수 focal 효과 측정"),
    ("287", "271_no_focal_wider_cls",
     "`grl_use_focal=False`, `grl_cls_hidden=128` (ep=500 유지)",
     "**(TBD, 미실행)** 2-axis: focal=F + wider classifier on 271 base"),
    ("288", "271_no_focal_grl_w05",
     "`grl_use_focal=False`, `grl_loss_weight=0.5` (ep=500 유지)",
     "**(TBD, 미실행)** 2-axis: focal=F + GRL weight up on 271 base"),
    ("289", "271_w300_p10",
     "`seq_length=300`, `patch_size=10`, `num_patches=30`",
     "**(TBD, 미실행)** Window=300, patch=10, patches=30 — 짧은 context (patch 해상도 유지)"),
    ("290", "271_w200_p10",
     "`seq_length=200`, `patch_size=10`, `num_patches=20`",
     "**(TBD, 미실행)** Window=200, patch=10, patches=20"),
    ("291", "271_w100_p10",
     "`seq_length=100`, `patch_size=10`, `num_patches=10`",
     "**(TBD, 미실행)** Window=100, patch=10, patches=10 — 극단적 짧은 context"),
    ("292", "271_w500_p5",
     "`patch_size=5`, `num_patches=100` (seq_length=500 유지)",
     "**(TBD, 미실행)** Patch 해상도 증가 (10→5), patches 2배 (50→100). Fine-grain temporal modeling"),
    ("293", "271_w200_p5",
     "`seq_length=200`, `patch_size=5`, `num_patches=40`",
     "**(TBD, 미실행)** 짧은 window + fine patch"),
    ("294", "271_w100_p5",
     "`seq_length=100`, `patch_size=5`, `num_patches=20`",
     "**(TBD, 미실행)** 극단 짧은 window + fine patch"),
    ("295", "271_target_patch",
     "`grl_target_mode='patch'` (ep=500 유지)",
     "**(TBD, 미실행)** ★★★ GRL target mode patch — isolated win analysis에서 patch가 window 우세 (Cell 11/6, RA 3/1). 271 default window 정당성 검증"),
    ("296", "271_grl_w1",
     "`grl_loss_weight=1.0` (ep=500 유지)",
     "**(TBD, 미실행)** GRL weight up — RA W/L 4/2로 w=1.0이 0.2보다 우세"),
    ("297", "271_td4",
     "`num_teacher_decoder_layers=4` (td=3→4, ep=500 유지)",
     "**(TBD, 미실행)** Teacher decoder 심화 — td=4가 여러 base에서 excl22 강점. 271 base 단독 미측정"),
    ("298", "271_sd1",
     "`num_student_decoder_layers=1` (sd=2→1, ep=500 유지)",
     "**(TBD, 미실행)** Student decoder 축소 — Cell sd=2 우세나 RA tied. 271 base 검증"),
    ("299", "271_cls_lr1",
     "`grl_cls_lr_ratio=1.0` (slow_cls 0.1→1.0, ep=500 유지)",
     "**(TBD, 미실행)** Slow classifier 정당성 검증 — RA 0.1 우세 (5/1), Cell 1.0 우세 (14/11)"),
    ("300", "271_anomloss_on",
     "`grl_disable_anomaly_loss=False` (ep=500 유지)",
     "**(TBD, 미실행)** GRL과 anomaly_loss 동시 활성 — 269(disable=F + bal=F + cos)에서 confound 분리"),
    ("301", "271_cls_hidden128",
     "`grl_cls_hidden=128` (default 자동 → 128, ep=500 유지)",
     "**(TBD, 미실행)** Wider GRL classifier — 277(cosine + hidden=128) RA=13.8"),
    ("302", "271_td4_sd1",
     "`num_teacher_decoder_layers=4`, `num_student_decoder_layers=1` (td=3→4, sd=2→1, ep=500 유지)",
     "**(TBD, 미실행)** Asymmetric depth flip — teacher 심화 + student 축소. 297×298 곱이 아닌 2-axis cross effect"),
    ("303", "271_revin",
     "`use_revin=True`, `normalize_mode='zscore'` (minmax→zscore 동반), `revin_affine=True`, `revin_visible_only=False` (ep=500 유지)",
     "**(TBD, 미실행)** RevIN 표준 적용 (ModernTCN AD 패턴) — per-window 정규화로 distribution shift 보정. 271은 minmax이므로 normalize_mode=zscore로 동반 변경 필수 (use_revin=True + minmax는 ValueError). Affine γ/β learnable."),
    ("304", "271_revin_no_affine",
     "`use_revin=True`, `normalize_mode='zscore'`, `revin_affine=False` (CATCH AD 패턴)",
     "**(TBD, 미실행)** RevIN affine OFF — CATCH AD wrapper의 default (affine=0). γ, β 학습 없이 순수 per-window mean/std 정규화만. anomaly signal이 affine으로 흡수되는 효과 차단. 303 (affine=True) 대비"),
    ("305", "271_revin_visible_only",
     "`use_revin=True`, `normalize_mode='zscore'`, `revin_affine=True`, `revin_visible_only=True`",
     "**(TBD, 미실행)** RevIN visible-only stats — force_mask_anomaly 와 자연 통합. anomaly 위치 (point_labels==1)의 timestep을 stats 계산에서 제외 → 'normal 분포 기준' 정규화. 학습 시 RevIN이 anomaly outlier에 흔들리지 않음. Inference 시 point_labels=None이라 full window fallback. Masked MAE 특화 변형 (4 baseline에 없음)"),
]


def build_group_p_section() -> str:
    lines = []
    lines.append("## Group P: Post-Group N 271-focused Single-axis (Exp 285-305, **ep=500 유지**, 모두 TBD placeholder)")
    lines.append("<callout>")
    lines.append("\t**Base**: 모든 항목 Exp **271** (Section 3.1/4.1 RA F1 #1 🥇) ablation.")
    lines.append("\t**목적**: 271 champion 주변 single-axis isolation. 각 항목은 271의 hyperparameter 1-2개만 변경.")
    lines.append("\t**ep=500/w=250 통일** (271 원본 환경 보존). ep 변경 없음.")
    lines.append("\t**참고**: 원안에는 274 base의 fm_adaptive_lambda=True 단독(=271 자체), balanced=F+focal=F 2-axis(=focal=F 단독)가 포함되었으나 271 base 재기반 시 collapse → 제외. 총 18개 entry.")
    lines.append("\t**공통 settings (271 base)**: `use_grl=True`, `use_feature_matching=True`, `normalize_mode='minmax'`, `fm_distance_metric='l2'`, `fm_adaptive_lambda=True`, `grl_balanced_sampling=False`, `grl_use_focal=True`, `grl_cls_lr_ratio=0.1`, `dynamic_margin_k=6`, `grl_loss_weight=0.2`, `grl_target_mode='window'`.")
    lines.append("\t**vs Group N**: Group N은 274 base (RA #2), Group P는 271 base (RA #1). 두 그룹은 다른 anchor라 직접 비교 불가, 각자 독립적 ablation family.")
    lines.append("</callout>")
    lines.append("<table>")
    lines.append("<tr>")
    for h in ["Exp", "Name", "Base", "Override (vs Base)", "Baseline (fair)", "검증 요소"]:
        lines.append(f"<td>{h}</td>")
    lines.append("</tr>")
    for exp, name, override, purpose in GROUP_P_SPECS:
        lines.append("<tr>")
        lines.append(f"<td>**{exp}**</td>")
        lines.append(f"<td>{name}</td>")
        lines.append("<td>271</td>")
        lines.append(f"<td>{override}</td>")
        lines.append("<td>271</td>")
        lines.append(f"<td>{purpose}</td>")
        lines.append("</tr>")
    lines.append("</table>")
    return "\n".join(lines)


if __name__ == "__main__":
    out = build_group_p_section()
    out_path = Path("/home/ykio/notebooks/claude/temp/group_p_section.txt")
    out_path.write_text(out)
    print(f"Wrote {out_path} ({len(out)} chars)")
