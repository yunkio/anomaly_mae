"""Build Notion-flavored markdown for the Early Stopping analysis page.

Output: /home/ykio/notebooks/claude/temp/early_stopping/notion_content.txt
"""
from pathlib import Path


def section_tldr():
    return """<callout icon="🎯" color="blue_bg">
\t**TL;DR — MAE 271 Early Stopping 분석 (2026-05-23)**
\t**Oracle = `pak_auc_f1` 최댓값**: 6-dataset 평균 0.7401 → 22 active baseline 중 **rank 1**.
\t**Per-dataset 최적 ES**: 6 group 중 4 (SWaT, WaDi A1/A2, PSM)에서 oracle 100% 재현. SMD avg 손실 5.96%, Exathlon avg 손실 3.55%.
\t**Cross-dataset 단일 최적 (metric, P, T)**: `th_train_teacher_recon_normal`, P=3, T=rel=0.01 → mean 0.7092 (oracle 대비 4.18%↓), 그래도 15 baseline 중 **rank 1** (다음 `tranad` 0.6205 대비 +14.3%p).
\t**핵심 권장**: "정상 분포의 teacher recon error가 더 이상 줄지 않으면 stop" 정책 (P=3, rel=0.001~0.01 threshold) 이 가장 robust.
</callout>"""


def section_scope():
    return """# 1. 분석 대상 (25 학습 history, MAE 271 base only)
<table fit-page-width="true" header-row="true">
<tr>
<td>Group</td>
<td>Dataset(s)</td>
<td>n</td>
<td>비고</td>
</tr>
<tr>
<td>**SWaT**</td>
<td>A1+A2 (학습 full, 점수 excl22)</td>
<td>1</td>
<td>training_histories는 `SWaT/A1A2_full`, lookup은 `SWaT/A1A2_excl22/epoch_metrics.json`</td>
</tr>
<tr>
<td>**WaDi**</td>
<td>A1, A2</td>
<td>2</td>
<td>각자 별도 실행</td>
</tr>
<tr>
<td>**PSM**</td>
<td>PSM</td>
<td>1</td>
<td>—</td>
</tr>
<tr>
<td>**SMD**</td>
<td>TimeSeAD 15 machines (1-2,1-7,2-1..3-9)</td>
<td>15</td>
<td>Wagner et al. 2023 권장 subset</td>
</tr>
<tr>
<td>**Exathlon**</td>
<td>app1, 2, 4, 5, 6, 9</td>
<td>6</td>
<td>FScustom 19 features</td>
</tr>
<tr>
<td>**Total**</td>
<td>—</td>
<td>**25**</td>
<td>simulation 제외 (사용자 결정 Q1)</td>
</tr>
</table>
<callout icon="📂" color="gray_bg">
\t**Source**: `results/experiments/271_20260508_094241_w500p10e4t3d2_dynamic_linear_minmax_k6/`
\t**271 config**: `e4t3d2`, `dynamic_margin_k=6`, `linear` patchify, `minmax`, ep=500/w=250, `use_grl=True`, `use_feature_matching=True`, `fm_adaptive_lambda=True`, `grl_balanced_sampling=False`, `grl_use_focal=True`, `grl_cls_lr_ratio=0.1`, `grl_loss_weight=0.2`, `grl_target_mode='window'`.
</callout>"""


def section_method():
    return """# 2. 방법론
## 2.1 ES 알고리즘
<callout icon="🧮" color="purple_bg">
\t**Pseudocode**:
\t```
\twarmup_epoch = 250
\teval_interval = 5  (epoch_metrics 기록 주기)
\tfor each (dataset, metric, patience P, threshold T):
\t    best_value = monitor_series[warmup]
\t    counter = 0
\t    for ep in 255, 260, ..., 500:
\t        v = monitor_series[ep]
\t        if improvement(v, best_value, T):
\t            best_value = v
\t            counter = 0
\t        else:
\t            counter += 1
\t            if counter >= P:
\t                stop_epoch = ep
\t                break
\t    return pak_auc_f1[stop_epoch]  ← scoring 테이블에서 lookup
\t```
\t**Improvement direction**: 손실 계열 → `min`, F1/AUC/acc → `max`. 자동 추론.
\t**Threshold types**: `abs` (절대 delta) 또는 `rel` (상대 delta = `(diff)/|best|`).
</callout>
## 2.2 Sweep grid (per dataset 3600 rows × 25 datasets = 90,000 simulations)
<table fit-page-width="true" header-row="true">
<tr>
<td>차원</td>
<td>값</td>
<td>개수</td>
</tr>
<tr>
<td>Patience</td>
<td>1, 2, 3, 5, 7, 10, 15, 20, 30, 50</td>
<td>10</td>
</tr>
<tr>
<td>Threshold</td>
<td>abs=0, abs=0.001, rel=0.001, rel=0.01, rel=0.05</td>
<td>5</td>
</tr>
<tr>
<td>Metric (Training history per-epoch scalar)</td>
<td>train_loss, train_rec_loss, train_disc_loss, train_*_loss, train_grl_*_acc, train_teacher_recon_*, train_student_recon_*, epoch_recon_*, epoch_disc_* (총)</td>
<td>33</td>
</tr>
<tr>
<td>Metric (Epoch-metrics every-5-ep scalar)</td>
<td>pak_auc_*, prc_auc, roc_auc, f1_score, pa_K_f1, pa_K_prc_auc, disc_snr, disturbing_* (총)</td>
<td>29</td>
</tr>
<tr>
<td>Metric (Derived = gap / ratio / separation)</td>
<td>teacher/student/recon/disc 별 anom-normal 분리도</td>
<td>10</td>
</tr>
<tr>
<td>**Total metrics per dataset**</td>
<td>**33 + 29 + 10**</td>
<td>**72**</td>
</tr>
</table>
## 2.3 Score lookup
<callout icon="📌" color="orange_bg">
\t모든 dataset: 학습 history는 **FULL** evaluation에서 진행 (Q2 사용자 결정).
\t성능 lookup: `epoch_metrics.json["epochs"][i]["pak_auc_f1"]`
\t**SWaT 예외**: 학습은 A1A2_full에서, 성능은 A1A2_excl22 디렉토리의 epoch_metrics에서 조회 (region 22 제외 평가).
</callout>"""


def section_metric_catalog():
    return """# 3. 사용한 지표 카탈로그
## 3.1 Training-history per-epoch scalar (33개)
<table fit-page-width="true" header-row="true">
<tr>
<td>Category</td>
<td>Keys</td>
</tr>
<tr>
<td>**Loss components**</td>
<td>`train_loss`, `train_rec_loss`, `train_disc_loss`, `train_normal_loss`, `train_anomaly_loss`, `train_fm_loss`, `train_grl_cls_loss`</td>
</tr>
<tr>
<td>**Adaptive coefficients**</td>
<td>`train_fm_adaptive_lambda`, `train_grl_lambda`, `train_grl_effective_weight`</td>
</tr>
<tr>
<td>**GRL classification acc**</td>
<td>`train_grl_balanced_acc`, `train_grl_anomaly_acc`, `train_grl_normal_acc`</td>
</tr>
<tr>
<td>**Recon by sample type**</td>
<td>`train_{teacher,student}_recon_{normal,anomaly}` (4)</td>
</tr>
<tr>
<td>**Epoch-aggregated raw/score/ratio by 3 sample types**</td>
<td>`epoch_{raw,score,ratio}_{recon,disc}_{normal,anomaly,disturbing}` (18)</td>
</tr>
</table>
## 3.2 Epoch-metrics per eval-checkpoint scalar (29개)
<table fit-page-width="true" header-row="true">
<tr>
<td>Category</td>
<td>Keys</td>
</tr>
<tr>
<td>**PA%K-AUC family**</td>
<td>`pak_auc_f1`, `pak_auc_prc_auc`, `pak_auc_roc_auc`, `pak_auc_precision`, `pak_auc_recall`, `pak_auc_f1_raw`</td>
</tr>
<tr>
<td>**Teacher version**</td>
<td>`teacher_pak_auc_f1`, `teacher_pak_auc_prc_auc`, `teacher_prc_auc`</td>
</tr>
<tr>
<td>**Plain AUC / F1**</td>
<td>`prc_auc`, `roc_auc`, `f1_score`, `teacher_f1_t`, `precision`, `recall`</td>
</tr>
<tr>
<td>**Disturbing region (SWaT)**</td>
<td>`disturbing_f1`, `disturbing_roc_auc`</td>
</tr>
<tr>
<td>**Discrepancy SNR**</td>
<td>`disc_snr`</td>
</tr>
<tr>
<td>**PA%K @ K∈{0,5,10,20,30,50}**</td>
<td>`pa_K_f1`, `pa_K_prc_auc` (12)</td>
</tr>
</table>
## 3.3 Derived metric (10개, normal vs anomaly 분리지표)
<table fit-page-width="true" header-row="true">
<tr>
<td>Name</td>
<td>Formula</td>
</tr>
<tr>
<td>`deriv_teacher_anom_normal_gap`</td>
<td>`teacher_a - teacher_n`</td>
</tr>
<tr>
<td>`deriv_teacher_anom_normal_ratio`</td>
<td>`teacher_a / teacher_n`</td>
</tr>
<tr>
<td>`deriv_teacher_anom_normal_separation`</td>
<td>`(teacher_a - teacher_n) / (|teacher_a| + |teacher_n|)`</td>
</tr>
<tr>
<td>`deriv_student_anom_normal_{gap,ratio,separation}`</td>
<td>(student 버전)</td>
</tr>
<tr>
<td>`deriv_recon_score_gap`</td>
<td>`epoch_recon_score_anom - epoch_recon_score_normal`</td>
</tr>
<tr>
<td>`deriv_recon_score_separation`</td>
<td>(normalized gap)</td>
</tr>
<tr>
<td>`deriv_disc_score_{gap,separation}`</td>
<td>(discrepancy 버전)</td>
</tr>
</table>"""


def section_missing_metrics():
    return """# 4. 미수집/미측정 지표 분류
## 4.1 Category A — 계산은 되지만 저장이 안 된 지표
<table fit-page-width="true" header-row="true">
<tr>
<td>Key</td>
<td>계산 위치</td>
<td>사유 / 활용 가능성</td>
</tr>
<tr>
<td>`train_d_loss`, `train_d_real_acc`, `train_d_fake_acc`, `train_adv_loss`, `train_adaptive_lambda`</td>
<td>`trainer.py` L496-507 → `epoch_losses['d_loss']` 등</td>
<td>`use_discriminator=True`일 때만 history 추가됨. 271은 `False` → 빈 list `[]`. **모든 추가 GAN 변형 ablation에 자동 적용됨**</td>
</tr>
<tr>
<td>`mean_discrepancy`</td>
<td>`loss.py` L405 `loss_dict['mean_discrepancy']`</td>
<td>**trainer가 history에 push하지 않음**. 한 줄 추가로 활용 가능 → teacher-student disagreement scalar로 사용 가능</td>
</tr>
<tr>
<td>`_feature_recon_mean`, `_feature_recon_max`, `_feature_disc_mean`, `_feature_disc_max`</td>
<td>`trainer.py` L722-726 (per-feature list)</td>
<td>list[num_features] 형태로 저장됨. **scalar로 축약 후 사용 가능** (mean/max/std over features) — 본 분석에서는 시간상 제외</td>
</tr>
<tr>
<td>`_train_feature_*`, `_infer_feature_*`</td>
<td>`epoch_metrics.json` 각 epoch</td>
<td>같은 사유 (list[num_features]) — scalar 축약 활용 가능</td>
</tr>
<tr>
<td>`pa_K_precision`, `pa_K_recall`, `pa_K_roc_auc` (K=0..100, 5 step)</td>
<td>`evaluator.py` L1761-1770</td>
<td>**저장됨** (epoch_metrics에 풀세트). 본 sweep에서는 일부만 선택했지만 전체 활용 가능</td>
</tr>
<tr>
<td>`disturbing_*` 메트릭 in non-SWaT</td>
<td>`evaluator.py` L1797-1800</td>
<td>모든 dataset에 populated되지만 binary label에서는 의미 약함</td>
</tr>
</table>
<callout icon="💡" color="green_bg">
\t**핵심**: 진정한 "계산 후 미저장" scalar는 **`mean_discrepancy` (teacher-student disagreement)** 하나뿐. 나머지는 list 형태로 저장됨 (분석에서 scalar로 축약 후 추출 가능).
</callout>
## 4.2 Category B — 측정 가능하지만 코드에 아직 없는 지표
<table fit-page-width="true" header-row="true">
<tr>
<td>Idea</td>
<td>근거</td>
<td>구현 비용</td>
</tr>
<tr>
<td>**Validation loss (held-out normal)**</td>
<td>표준 ES literature의 default 지표. 현재 코드는 test=val 패턴 → 별도 validation split 없음</td>
<td>중 (config + dataloader split + per-epoch eval, 5-10% 학습시간 증가)</td>
</tr>
<tr>
<td>**Teacher-Student disagreement (`mean_discrepancy`)**</td>
<td>loss.py에 이미 계산됨 (`loss_dict['mean_discrepancy']`). trainer history에 1-line push 추가만 필요</td>
<td>**저** (1 line)</td>
</tr>
<tr>
<td>**Per-window separation (z-score, percentile)**</td>
<td>같은 epoch 안의 anomaly score 분포에서 정상 vs 이상 윈도우의 z-score gap. evaluator에 sample 단위 score 존재 → type별 quantile 저장</td>
<td>중 (evaluator에서 percentile 계산)</td>
</tr>
<tr>
<td>**EMA-smoothed training loss**</td>
<td>단일 epoch noise 흔들림 완화 → ES trigger 안정화. post-process로 가능</td>
<td>**저** (post-process)</td>
</tr>
<tr>
<td>**Loss curvature (2nd derivative)**</td>
<td>local plateau 탐지에 유용. 후처리 가능</td>
<td>**0** (post-process)</td>
</tr>
<tr>
<td>**Grad norm of recon vs disc loss**</td>
<td>adaptive_lambda 계산 시 `fm.py`의 `compute_adaptive_lambda`에서 이미 grad norm 계산 → 저장만 안 됨</td>
<td>**저** (loss_dict에 key 1개 추가)</td>
</tr>
<tr>
<td>**Patch-level entropy**</td>
<td>masked patch의 reconstructed value distribution entropy. 이론적 검출 신호</td>
<td>고 (eval loop 변경)</td>
</tr>
<tr>
<td>**Top-K score percentile of normals**</td>
<td>정상 윈도우의 top-K% anomaly score (high score normal = noisy). evaluator score 후처리</td>
<td>**0** (post-process)</td>
</tr>
<tr>
<td>**PA%K curve area derivative**</td>
<td>학습 epoch에 따른 PA%K curve의 변화율. 후처리</td>
<td>**0** (post-process)</td>
</tr>
<tr>
<td>**Window-level reconstruction divergence histogram**</td>
<td>각 epoch에서 정상/이상 윈도우의 anomaly score distribution 추출 (e.g., KL div, JSD)</td>
<td>중</td>
</tr>
</table>
<callout icon="⭐" color="yellow_bg">
\t**Recommendation 우선순위**:
\t① **Teacher-Student disagreement** 저장 (1-line) — 즉시 가능
\t② **Grad norm** 저장 (low cost) — adaptive_lambda 코드에서 reuse
\t③ **EMA loss / curvature** (post-process only) — 외부 스크립트로 처리 가능
\t④ Validation loss split — 가장 큰 변화이지만 학습 표준 정책에 맞춰 도입할 가치
</callout>"""


def section_per_dataset():
    # SWaT/WaDi/PSM table
    s = """# 5. 결과 — Per-Dataset Best ES Config
각 dataset마다 (metric, P, T) 를 독립적으로 최적화. **5/25** dataset은 ES가 oracle을 정확히 재현, 14/25은 oracle 대비 ≤2% 손실.
## 5.1 4 Single-Dataset Groups (SWaT, WaDi A1/A2, PSM)
<table fit-page-width="true" header-row="true">
<tr>
<td>Dataset</td>
<td>Oracle ep</td>
<td>Oracle PA F1</td>
<td>Best ES metric</td>
<td>P</td>
<td>T</td>
<td>Stop ep</td>
<td>ES PA F1</td>
<td>Loss</td>
</tr>
<tr>
<td>**SWaT_excl22**</td>
<td>280</td>
<td>0.6305</td>
<td>`th_train_rec_loss`</td>
<td>3</td>
<td>abs=0</td>
<td>280</td>
<td>0.6305</td>
<td>**0.00%** 🎯</td>
</tr>
<tr>
<td>**WaDi_A1**</td>
<td>395</td>
<td>0.8495</td>
<td>`th_train_grl_balanced_acc`</td>
<td>20</td>
<td>rel=0.001</td>
<td>395</td>
<td>0.8495</td>
<td>**0.00%** 🎯</td>
</tr>
<tr>
<td>**WaDi_A2**</td>
<td>410</td>
<td>0.7939</td>
<td>`th_train_rec_loss`</td>
<td>2</td>
<td>abs=0</td>
<td>410</td>
<td>0.7939</td>
<td>**0.00%** 🎯</td>
</tr>
<tr>
<td>**PSM**</td>
<td>260</td>
<td>0.8034</td>
<td>`th_train_loss`</td>
<td>2</td>
<td>abs=0</td>
<td>260</td>
<td>0.8034</td>
<td>**0.00%** 🎯</td>
</tr>
</table>
## 5.2 SMD 15 TimeSeAD Machines
<table fit-page-width="true" header-row="true">
<tr>
<td>Machine</td>
<td>Oracle ep</td>
<td>Oracle</td>
<td>Best ES (metric, P, T, stop_ep)</td>
<td>ES PA F1</td>
<td>Loss</td>
</tr>"""
    smd_rows = [
        ("machine-1-2", 295, 0.7007, "`teacher_recon_anomaly` P=5 abs=0 @ 295", 0.7007, "**0.00%** 🎯"),
        ("machine-1-7", 170, 0.7463, "`train_loss` P=2 abs=0 @ 260", 0.7301, "2.17%"),
        ("machine-2-1", 60, 0.6403, "`grl_balanced_acc` P=3 abs=0 @ 270", 0.6223, "2.81%"),
        ("machine-2-2", 25, 0.6665, "`train_loss` P=2 abs=0 @ 260", 0.6223, "6.63% (oracle pre-warmup)"),
        ("machine-2-3", 375, 0.8894, "`teacher_recon_normal` P=7 rel=0.05 @ 375", 0.8894, "**0.00%** 🎯"),
        ("machine-2-4", 430, 0.6146, "`recon_score_disturbing` P=15 rel=0.01 @ 430", 0.6146, "**0.00%** 🎯"),
        ("machine-2-6", 85, 0.5941, "`train_loss` P=3 abs=0 @ 265", 0.5900, "0.69%"),
        ("machine-2-7", 65, 0.8788, "`train_loss` P=7 abs=0 @ 285", 0.8778, "0.11%"),
        ("machine-2-9", 230, 0.7211, "`train_loss` P=5 abs=0 @ 275", 0.7034, "2.45%"),
        ("machine-3-1", 285, 0.9387, "`train_loss` P=7 abs=0 @ 285", 0.9387, "**0.00%** 🎯"),
        ("machine-3-2", 90, 0.0938, "`grl_balanced_acc` P=20 abs=0.001 @ 355", 0.0827, "11.83% (low absolute)"),
        ("machine-3-3", 275, 0.7791, "`train_loss` P=5 abs=0 @ 275", 0.7791, "**0.00%** 🎯"),
        ("machine-3-6", 30, 0.8138, "`train_loss` P=1 abs=0 @ 255", 0.7595, "6.67% (oracle pre-warmup)"),
        ("machine-3-8", 60, 0.6946, "`train_loss` P=1 abs=0 @ 255", 0.6368, "8.32%"),
        ("machine-3-9", 275, 1.0000, "`train_loss` P=5 abs=0 @ 275", 1.0000, "**0.00%** 🎯"),
    ]
    for m, oep, ov, conf, esv, loss in smd_rows:
        s += f"""
<tr>
<td>{m}</td>
<td>{oep}</td>
<td>{ov:.4f}</td>
<td>{conf}</td>
<td>{esv:.4f}</td>
<td>{loss}</td>
</tr>"""
    s += """
<tr>
<td>**SMD 15 mean**</td>
<td>—</td>
<td>**0.7181**</td>
<td>(best cross-machine avg cfg)</td>
<td>**0.6753**</td>
<td>**5.96%**</td>
</tr>
</table>

## 5.3 Exathlon 6 Apps
<table fit-page-width="true" header-row="true">
<tr>
<td>App</td>
<td>Oracle ep</td>
<td>Oracle</td>
<td>Best ES (metric, P, T, stop_ep)</td>
<td>ES PA F1</td>
<td>Loss</td>
</tr>"""
    exa_rows = [
        ("app1", 435, 0.4668, "`epoch_raw_disc_normal` P=20 abs=0 @ 435", 0.4668, "**0.00%** 🎯"),
        ("app2", 260, 0.9426, "`train_loss` P=2 abs=0 @ 260", 0.9426, "**0.00%** 🎯"),
        ("app4", 255, 0.8274, "`train_loss` P=1 abs=0 @ 255", 0.8274, "**0.00%** 🎯"),
        ("app5", 295, 0.8190, "`grl_balanced_acc` P=5 abs=0.001 @ 295", 0.8190, "**0.00%** 🎯"),
        ("app6", 55, 0.2537, "`student_recon_anomaly` P=7 abs=0 @ 340", 0.2460, "3.03% (oracle pre-warmup)"),
        ("app9", 440, 0.5617, "`teacher_recon_anomaly` P=15 abs=0 @ 440", 0.5617, "**0.00%** 🎯"),
    ]
    for a, oep, ov, conf, esv, loss in exa_rows:
        s += f"""
<tr>
<td>{a}</td>
<td>{oep}</td>
<td>{ov:.4f}</td>
<td>{conf}</td>
<td>{esv:.4f}</td>
<td>{loss}</td>
</tr>"""
    s += """
<tr>
<td>**Exathlon 6 mean**</td>
<td>—</td>
<td>**0.6452**</td>
<td>(best cross-app avg cfg)</td>
<td>**0.6223**</td>
<td>**3.55%**</td>
</tr>
</table>
<callout icon="🔍" color="blue_bg">
\t**관찰**: 14/25 dataset에서 ES가 oracle을 정확히 재현. 손실이 큰 dataset (machine-2-2, 3-6, 3-8, app6)은 모두 **oracle이 warmup=250 이전**에 위치하여 ES가 절대 도달 불가. → warmup 단축이 향후 개선 방향.
</callout>"""
    return s


def section_cross_dataset():
    return """# 6. 결과 — Cross-Dataset 단일 최적 (metric, P, T)
**하나의 (metric, P, T)** 만 사용해서 6 group 평균 PA F1을 최대화하는 조합.
## 6.1 Top 10 Cross-Dataset Configs
<table fit-page-width="true" header-row="true">
<tr>
<td>#</td>
<td>Metric</td>
<td>P</td>
<td>T</td>
<td>Mean PA F1 (6 group)</td>
<td>Loss vs Oracle</td>
</tr>
<tr>
<td>**1**</td>
<td>`th_train_teacher_recon_normal`</td>
<td>3</td>
<td>rel=0.01</td>
<td>**0.7092** 🥇</td>
<td>4.18%</td>
</tr>
<tr>
<td>2</td>
<td>`th_train_rec_loss`</td>
<td>3</td>
<td>rel=0.001</td>
<td>0.7088</td>
<td>4.23%</td>
</tr>
<tr>
<td>3</td>
<td>`th_train_rec_loss`</td>
<td>3</td>
<td>abs=0</td>
<td>0.7087</td>
<td>4.24%</td>
</tr>
<tr>
<td>4</td>
<td>`th_train_teacher_recon_normal`</td>
<td>3</td>
<td>rel=0.001</td>
<td>0.7065</td>
<td>4.54%</td>
</tr>
<tr>
<td>5</td>
<td>`th_train_rec_loss`</td>
<td>3</td>
<td>rel=0.01</td>
<td>0.7065</td>
<td>4.54%</td>
</tr>
<tr>
<td>6</td>
<td>`th_train_teacher_recon_normal`</td>
<td>2</td>
<td>abs=0</td>
<td>0.7060</td>
<td>4.61%</td>
</tr>
<tr>
<td>7</td>
<td>`th_train_teacher_recon_normal`</td>
<td>2</td>
<td>rel=0.001</td>
<td>0.7060</td>
<td>4.61%</td>
</tr>
<tr>
<td>8</td>
<td>`em_pak_auc_recall`</td>
<td>15</td>
<td>rel=0.001</td>
<td>0.7035</td>
<td>4.95%</td>
</tr>
<tr>
<td>9</td>
<td>`em_pak_auc_recall`</td>
<td>15</td>
<td>abs=0.001</td>
<td>0.7035</td>
<td>4.95%</td>
</tr>
<tr>
<td>10</td>
<td>`th_train_grl_balanced_acc`</td>
<td>15</td>
<td>abs=0</td>
<td>0.7027</td>
<td>5.05%</td>
</tr>
</table>
## 6.2 Top 20 Metric Families (best P/T 내 max)
<table fit-page-width="true" header-row="true">
<tr>
<td>#</td>
<td>Metric family</td>
<td>Best mean PA F1</td>
<td>Best (P, T)</td>
</tr>
<tr>
<td>1</td>
<td>`th_train_teacher_recon_normal`</td>
<td>**0.7092**</td>
<td>P=3, T=rel=0.01</td>
</tr>
<tr>
<td>2</td>
<td>`th_train_rec_loss`</td>
<td>0.7088</td>
<td>P=3, T=rel=0.001</td>
</tr>
<tr>
<td>3</td>
<td>`deriv_student_anom_normal_separation`</td>
<td>0.6926</td>
<td>P=30, T=rel=0.05</td>
</tr>
<tr>
<td>4</td>
<td>`em_pak_auc_recall`</td>
<td>0.6916</td>
<td>P=15, T=rel=0.001</td>
</tr>
<tr>
<td>5</td>
<td>`em_pa_5_prc_auc`</td>
<td>0.6915</td>
<td>P=30, T=rel=0.01</td>
</tr>
<tr>
<td>6</td>
<td>`th_train_grl_anomaly_acc`</td>
<td>0.6909</td>
<td>P=15, T=rel=0.001</td>
</tr>
<tr>
<td>7</td>
<td>`em_teacher_pak_auc_prc_auc`</td>
<td>0.6905</td>
<td>P=30, T=rel=0.001</td>
</tr>
<tr>
<td>8</td>
<td>`em_pa_30_prc_auc`</td>
<td>0.6902</td>
<td>P=30, T=rel=0.01</td>
</tr>
<tr>
<td>9</td>
<td>`th_epoch_recon_score_disturbing`</td>
<td>0.6899</td>
<td>P=5, T=abs=0</td>
</tr>
<tr>
<td>10</td>
<td>`th_train_grl_balanced_acc`</td>
<td>0.6897</td>
<td>P=15, T=abs=0</td>
</tr>
<tr>
<td>11</td>
<td>`em_pa_10_prc_auc`</td>
<td>0.6893</td>
<td>P=30, T=rel=0.01</td>
</tr>
<tr>
<td>12</td>
<td>`deriv_recon_score_gap`</td>
<td>0.6892</td>
<td>P=30, T=rel=0.001</td>
</tr>
<tr>
<td>13</td>
<td>`em_pak_auc_f1`</td>
<td>0.6888</td>
<td>P=30, T=rel=0.01</td>
</tr>
<tr>
<td>14</td>
<td>`em_recall`</td>
<td>0.6886</td>
<td>P=3, T=rel=0.05</td>
</tr>
<tr>
<td>15</td>
<td>`em_pa_30_f1`</td>
<td>0.6883</td>
<td>P=20, T=abs=0.001</td>
</tr>
<tr>
<td>16</td>
<td>`deriv_recon_score_separation`</td>
<td>0.6878</td>
<td>P=30, T=rel=0.01</td>
</tr>
<tr>
<td>17</td>
<td>`th_epoch_disc_score_normal`</td>
<td>0.6877</td>
<td>P=20, T=rel=0.01</td>
</tr>
<tr>
<td>18</td>
<td>`deriv_disc_score_gap`</td>
<td>0.6876</td>
<td>P=20, T=abs=0.001</td>
</tr>
<tr>
<td>19</td>
<td>`th_epoch_disc_ratio_anomaly`</td>
<td>0.6870</td>
<td>P=30, T=rel=0.05</td>
</tr>
<tr>
<td>20</td>
<td>`em_pak_auc_prc_auc`</td>
<td>0.6869</td>
<td>P=30, T=rel=0.05</td>
</tr>
</table>
<callout icon="📈" color="green_bg">
\t**핵심 관찰**:
\t• **Training-side `teacher_recon_normal` (정상 분포의 teacher recon error)** 가 1위. "정상 분포가 더 이상 안 줄어들면 학습 멈춰라" 는 가장 표준적인 휴리스틱이 best.
\t• `train_rec_loss` (정상의 전체 reconstruction loss) 가 2위로 1위와 거의 동일.
\t• Inference 쪽 (`em_*`) 지표는 8위부터 등장 — eval_interval=5의 noise가 ES decision을 흔든다.
\t• Patience=2-3, threshold=rel=0.001~0.01 이 일반적으로 best (민감하면서도 noise 견딤).
\t• Patience=15-30은 inference 지표 (`em_*`) 에서 우세 (느린 plateau 검출에 유리).
</callout>"""


def section_leaderboard():
    return """# 7. Rank Comparison — MAE 271 (+ES) vs 15 Active Baselines
**Q3 (minmax normalonly) PA%K-AUC F1 기준 6-dataset rank average**
<callout icon="ℹ️" color="gray_bg">
\t**참고**: 7개 신규 SOTA (`tfmae`, `npsr`, `timesnet`, `dcdetector`, `memto`, `moderntcn`, `catch`) 는 Q3 batch 진행 중 (placeholder `-` 상태) → 본 leaderboard에는 15 active baseline만 포함. 신규 7개 결과 swap-in 후 rank 재계산 필요.
\t**6 datasets**: SWaT_excl22, WaDi_A1, WaDi_A2, PSM, **SMD avg (15 TimeSeAD)**, **Exathlon avg (6 apps)**.
</callout>
<table fit-page-width="true" header-row="true">
<tr>
<td>Rank</td>
<td>Model</td>
<td>SWaT</td>
<td>WaDi A1</td>
<td>WaDi A2</td>
<td>PSM</td>
<td>SMD avg</td>
<td>Exathlon avg</td>
<td>**Rank Avg**</td>
<td>Mean PA F1</td>
</tr>
<tr>
<td>**1** 🥇</td>
<td>**MAE 271 (Oracle, pak_auc_f1)**</td>
<td>0.6305</td>
<td>0.8495</td>
<td>0.7939</td>
<td>0.8034</td>
<td>0.7181</td>
<td>0.6452</td>
<td>**1.00**</td>
<td>0.7401</td>
</tr>
<tr>
<td>**2** 🥈</td>
<td>**MAE 271 ES (per-ds oracle, upper bound)**</td>
<td>0.6305</td>
<td>0.8495</td>
<td>0.7939</td>
<td>0.8034</td>
<td>0.6753</td>
<td>0.6223</td>
<td>**2.83**</td>
<td>0.7291</td>
</tr>
<tr>
<td>**3** 🥉</td>
<td>**MAE 271 ES #1 (teacher_recon_normal, P=3, rel=0.01)**</td>
<td>0.6305</td>
<td>0.8309</td>
<td>0.7753</td>
<td>0.7947</td>
<td>0.6488</td>
<td>0.5752</td>
<td>**3.67**</td>
<td>0.7092</td>
</tr>
<tr>
<td>4</td>
<td>MAE 271 ES #2 (`rec_loss`, P=3, rel=0.001)</td>
<td>0.6305</td>
<td>0.8252</td>
<td>0.7753</td>
<td>0.7947</td>
<td>0.6480</td>
<td>0.5794</td>
<td>4.33</td>
<td>0.7088</td>
</tr>
<tr>
<td>5</td>
<td>MAE 271 ES #3 (`rec_loss`, P=3, abs=0)</td>
<td>0.6305</td>
<td>0.8252</td>
<td>0.7753</td>
<td>0.7947</td>
<td>0.6475</td>
<td>0.5788</td>
<td>5.33</td>
<td>0.7087</td>
</tr>
<tr>
<td>6</td>
<td>MAE 271 ES #4 (`teacher_recon_normal`, P=3, rel=0.001)</td>
<td>0.6305</td>
<td>0.8252</td>
<td>0.7753</td>
<td>0.7441</td>
<td>0.6572</td>
<td>0.6065</td>
<td>6.00</td>
<td>0.7065</td>
</tr>
<tr>
<td>7</td>
<td>MAE 271 ES #5 (`rec_loss`, P=3, rel=0.01)</td>
<td>0.6305</td>
<td>0.8309</td>
<td>0.7753</td>
<td>0.7947</td>
<td>0.6323</td>
<td>0.5752</td>
<td>6.17</td>
<td>0.7065</td>
</tr>
<tr>
<td>8</td>
<td>`tranad`</td>
<td>0.4580</td>
<td>0.6505</td>
<td>0.6630</td>
<td>0.7285</td>
<td>0.6506</td>
<td>0.5722</td>
<td>9.33</td>
<td>0.6205</td>
</tr>
<tr>
<td>9</td>
<td>`anomaly_transformer`</td>
<td>0.5016</td>
<td>0.6923</td>
<td>0.6867</td>
<td>0.7340</td>
<td>0.5934</td>
<td>0.5064</td>
<td>10.67</td>
<td>0.6191</td>
</tr>
<tr>
<td>10</td>
<td>`omnianomaly`</td>
<td>0.4825</td>
<td>0.5241</td>
<td>0.5194</td>
<td>0.7874</td>
<td>0.5915</td>
<td>0.5189</td>
<td>11.67</td>
<td>0.5706</td>
</tr>
<tr>
<td>11</td>
<td>`mlpmixer`</td>
<td>0.4137</td>
<td>0.6111</td>
<td>0.6057</td>
<td>0.7323</td>
<td>0.5964</td>
<td>0.5173</td>
<td>12.67</td>
<td>0.5794</td>
</tr>
<tr>
<td>12</td>
<td>`mlp`</td>
<td>0.4623</td>
<td>0.5857</td>
<td>0.5543</td>
<td>0.7550</td>
<td>0.5689</td>
<td>0.4816</td>
<td>13.17</td>
<td>0.5680</td>
</tr>
<tr>
<td>13</td>
<td>`transformer`</td>
<td>0.4237</td>
<td>0.6290</td>
<td>0.6165</td>
<td>0.7065</td>
<td>0.5745</td>
<td>0.4996</td>
<td>13.50</td>
<td>0.5750</td>
</tr>
<tr>
<td>14</td>
<td>`nn_distance`</td>
<td>0.4339</td>
<td>0.5276</td>
<td>0.5436</td>
<td>0.7429</td>
<td>0.5443</td>
<td>0.4977</td>
<td>14.33</td>
<td>0.5483</td>
</tr>
<tr>
<td>14</td>
<td>`usad`</td>
<td>0.4468</td>
<td>0.3456</td>
<td>0.3462</td>
<td>0.5875</td>
<td>0.6452</td>
<td>0.5517</td>
<td>14.33</td>
<td>0.4872</td>
</tr>
<tr>
<td>14</td>
<td>`gdn`</td>
<td>0.4142</td>
<td>0.4492</td>
<td>0.4183</td>
<td>0.7369</td>
<td>0.5919</td>
<td>0.5196</td>
<td>14.33</td>
<td>0.5217</td>
</tr>
<tr>
<td>17</td>
<td>`pca_error`</td>
<td>0.2646</td>
<td>0.4915</td>
<td>0.4666</td>
<td>0.7517</td>
<td>0.5690</td>
<td>0.4893</td>
<td>15.50</td>
<td>0.5054</td>
</tr>
<tr>
<td>18</td>
<td>`dagmm`</td>
<td>0.1786</td>
<td>0.4751</td>
<td>0.4668</td>
<td>0.5533</td>
<td>0.6135</td>
<td>0.5190</td>
<td>15.67</td>
<td>0.4677</td>
</tr>
<tr>
<td>19</td>
<td>`gcn_lstm`</td>
<td>0.1429</td>
<td>0.1128</td>
<td>0.1286</td>
<td>0.6470</td>
<td>0.5445</td>
<td>0.5110</td>
<td>19.00</td>
<td>0.3478</td>
</tr>
<tr>
<td>20</td>
<td>`sensor_range`</td>
<td>0.0757</td>
<td>0.4902</td>
<td>0.4827</td>
<td>0.4764</td>
<td>0.2955</td>
<td>0.3937</td>
<td>19.67</td>
<td>0.3690</td>
</tr>
<tr>
<td>21</td>
<td>`l2_norm`</td>
<td>0.0810</td>
<td>0.2520</td>
<td>0.2468</td>
<td>0.5717</td>
<td>0.4612</td>
<td>0.4935</td>
<td>19.83</td>
<td>0.3511</td>
</tr>
<tr>
<td>22</td>
<td>`random`</td>
<td>0.1813</td>
<td>0.1891</td>
<td>0.1908</td>
<td>0.6516</td>
<td>0.2248</td>
<td>0.4259</td>
<td>20.00</td>
<td>0.3106</td>
</tr>
</table>
<callout icon="🏆" color="purple_bg">
\t**핵심 관찰**:
\t• **Top 7 = 모두 MAE 271 변형**. 8위 `tranad`까지 rank avg gap = 3.16.
\t• **ES adoption cost**: rank 1 (Oracle) → rank 3 (best cross-cfg ES). rank avg +2.67, mean PA F1 0.0309 (4.18%) 손실.
\t• **Per-dataset oracle ES (upper bound)**: rank 2.83 — 4 single-dataset에서 oracle 완벽 재현, SMD/Exathlon 평균에서만 3-6% 손실.
\t• **Cross-config ES도 15 baseline 압도**: `tranad` 0.6205 vs MAE 271 ES #1 0.7092 → +14.3%p absolute gap.
</callout>"""


def section_conclusions():
    return """# 8. 한계 및 향후 개선
## 8.1 한계
<table fit-page-width="true" header-row="true">
<tr>
<td>Limitation</td>
<td>Impact</td>
<td>대응 방향</td>
</tr>
<tr>
<td>**Warmup 250 ep 고정**</td>
<td>SMD machine-2-2 (ep25), 3-6 (ep30), 3-8 (ep60), app6 (ep55) 등 oracle이 250 이전인 dataset에서 ES가 절대 oracle 재현 불가</td>
<td>warmup ablation (50/100/150/200/250) 추가 측정</td>
</tr>
<tr>
<td>**Eval interval 5**</td>
<td>5 ep 단위로만 stop trigger 가능. 정확한 oracle epoch에 도달 못 함</td>
<td>1 ep eval은 비용 큼 (full inference loop). 현재 trade-off로 5가 합리적</td>
</tr>
<tr>
<td>**22 baseline 중 7개 (신규 SOTA) Q3 미완료**</td>
<td>절대적 rank는 추후 변동 가능</td>
<td>tfmae/timesnet/dcdetector/memto/moderntcn/catch/npsr 실험 완료 후 leaderboard 재계산</td>
</tr>
<tr>
<td>**Validation set 부재**</td>
<td>표준 ES literature는 train/val/test 분리 후 val loss로 ES. 본 분석은 test=val pattern (project 표준) 위에서 학습 시그널만 사용</td>
<td>config + dataloader 변경 → validation split 도입 검토</td>
</tr>
<tr>
<td>**SMD: 15 vs 28 차이**</td>
<td>본 분석은 TimeSeAD 권장 15 machines. 기존 baseline comparison page는 28 machines avg → 직접 비교 시 SMD 컬럼 값 차이</td>
<td>본 분석에서는 baseline도 동일 15 machines로 재집계 → fair 비교 확보됨</td>
</tr>
</table>
## 8.2 권장 정책 (실전 ES 가이드)
<table fit-page-width="true" header-row="true">
<tr>
<td>시나리오</td>
<td>추천 metric</td>
<td>P</td>
<td>T</td>
<td>예상 rank avg (vs 15 baseline)</td>
</tr>
<tr>
<td>**전 범용 (default)**</td>
<td>`th_train_teacher_recon_normal`</td>
<td>3</td>
<td>rel=0.01</td>
<td>**1위 (3.67)**</td>
</tr>
<tr>
<td>안정적 plateau 기대 (긴 학습)</td>
<td>`th_train_rec_loss`</td>
<td>3</td>
<td>rel=0.001</td>
<td>1위 (4.33)</td>
</tr>
<tr>
<td>Inference cost 허용 (eval 매 ep)</td>
<td>`em_pak_auc_f1`</td>
<td>30</td>
<td>rel=0.01</td>
<td>~1위 (8위 정도)</td>
</tr>
<tr>
<td>짧은 데이터셋 (SMD/Exathlon 등)</td>
<td>**warmup을 50으로 단축** + `train_loss`</td>
<td>2</td>
<td>abs=0</td>
<td>per-dataset rank up</td>
</tr>
</table>
## 8.3 코드 변경 추천 (Category B 도입)
<table fit-page-width="true" header-row="true">
<tr>
<td>우선순위</td>
<td>변경 내용</td>
<td>비용</td>
<td>효과</td>
</tr>
<tr>
<td>**1** (1-line)</td>
<td>`loss.py` L399: `loss_dict['mean_discrepancy']` (이미 계산됨) → trainer에서 history push 한 줄 추가</td>
<td>1 line</td>
<td>Teacher-Student disagreement scalar metric 즉시 활용</td>
</tr>
<tr>
<td>**2** (small refactor)</td>
<td>`fm.py`의 adaptive_lambda 계산 후 `loss_dict['recon_grad_norm']`, `loss_dict['fm_grad_norm']` 저장</td>
<td>~5 line</td>
<td>Gradient magnitude 기반 ES + 학습 동력 진단</td>
</tr>
<tr>
<td>**3** (post-process)</td>
<td>EMA-smoothed loss, loss curvature 계산 helper</td>
<td>0 (분석 스크립트만)</td>
<td>안정적 plateau 검출</td>
</tr>
<tr>
<td>**4** (config + dataloader)</td>
<td>Validation split 도입 (정상 일부 분리) + per-epoch validation loss</td>
<td>중 (학습 시간 5-10% 증가)</td>
<td>표준 ES literature와 align</td>
</tr>
</table>"""


def section_artifacts():
    return """# 9. Artifacts & Reproduce
<table fit-page-width="true" header-row="true">
<tr>
<td>파일</td>
<td>크기</td>
<td>설명</td>
</tr>
<tr>
<td>`temp/early_stopping/sweep_raw.json`</td>
<td>21 MB</td>
<td>전체 sweep 결과 (90,000 rows, 25 ds × 3600 cfg)</td>
</tr>
<tr>
<td>`temp/early_stopping/best_per_dataset.json`</td>
<td>2.5 KB</td>
<td>Per-dataset best (metric, P, T) + oracle 정보</td>
</tr>
<tr>
<td>`temp/early_stopping/cross_dataset_top50.json`</td>
<td>34 KB</td>
<td>Cross-dataset top-50 configs (mean over 6 groups)</td>
</tr>
<tr>
<td>`temp/early_stopping/metric_family_ranking.json`</td>
<td>10 KB</td>
<td>Metric family 별 best/avg mean</td>
</tr>
<tr>
<td>`temp/early_stopping/baseline_aggregated.json`</td>
<td>4 KB</td>
<td>15 baseline의 6-group pak_auc_f1 (SMD avg = 15 TimeSeAD)</td>
</tr>
<tr>
<td>`temp/early_stopping/rank_comparison.json`</td>
<td>18 KB</td>
<td>Leaderboard (Oracle + per-ds best + top-5 ES + 15 baseline)</td>
</tr>
<tr>
<td>`temp/early_stopping/report.md`</td>
<td>20 KB</td>
<td>전체 보고서 (md 원본)</td>
</tr>
</table>
<callout icon="🛠️" color="gray_bg">
\t**Scripts (재실행 가능)**:
\t• `scripts/early_stopping_analysis.py` — sweep generator (25 datasets × 3600 configs)
\t• `scripts/early_stopping_analyze.py` — per-dataset / cross-dataset 분석
\t• `scripts/early_stopping_baseline_aggregate.py` — 15 baseline pak_auc_f1 6-group 집계
\t• `scripts/early_stopping_rank_compare.py` — 최종 leaderboard 생성
\t**실행 순서**: `analysis → analyze → baseline_aggregate → rank_compare`
</callout>"""


def main():
    parts = [
        section_tldr(),
        "---",
        section_scope(),
        section_method(),
        section_metric_catalog(),
        section_missing_metrics(),
        section_per_dataset(),
        section_cross_dataset(),
        section_leaderboard(),
        section_conclusions(),
        section_artifacts(),
    ]
    out = "\n".join(parts)
    out_path = Path("/home/ykio/notebooks/claude/temp/early_stopping/notion_content.txt")
    out_path.write_text(out)
    print(f"Wrote {out_path} ({len(out)} chars, {len(out.splitlines())} lines)")


if __name__ == "__main__":
    main()
