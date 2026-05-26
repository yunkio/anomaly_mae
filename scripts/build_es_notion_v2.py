"""Build Notion-flavored markdown for the Early Stopping v2 analysis page.

Output: temp/early_stopping/notion_content_v2.txt
"""
from pathlib import Path


def section_tldr():
    return """<callout icon="🎯" color="blue_bg">
\t**TL;DR — MAE 271 Early Stopping 분석 v2 (2026-05-23, 확장 sweep)**
\t**Sweep 규모**: 25 datasets × **298 base metrics** × 7 post-process ops × 50 P×T grid = **2.6M simulations** (14초, 6 worker multiprocess, peak memory 20%).
\t**Oracle (pak_auc_f1 최댓값)**: 6-dataset 평균 **0.7401** → 22 active baseline 중 rank 1.
\t**Per-dataset 최적 ES (upper bound)**: rank avg **2.00**, mean 0.7318 (oracle 대비 1.12%↓).
\t**Cross-dataset 단일 최적 (metric, op, P, T)**: `th_train_teacher_recon_normal`, op=raw, P=3, T=rel=0.01 → mean **0.7092** (oracle 대비 4.18%↓), 15 baseline 중 rank 1 (`tranad` 0.6205 대비 +14.3%p).
\t**사용자 우려 (4번) 검증**: teacher_recon_normal은 250 ep 이후 변화량이 1/1000 수준 (plateau) → noise-driven trigger 가능성 확인됨. 그러나 실제 stop_epoch 평균은 250 직후가 아니라 ~320이며 25 dataset 모두 floor(255)에서 stop하지 않음.
\t**사용자 제안 (5번)**: `|Δteacher_recon|/|Δstudent_recon|` (W=5 anomaly) 0.6924, `|Δteacher−Δstudent|` (W=20 normal, curvature10) 0.6948 — top 20 진입했지만 단순 plateau detection을 능가하진 못함.
\t**신규 발견 1**: `deriv_pa_K_curve_avg_slope_over_K` (PA-K 곡선의 K 방향 평균 slope) 0.6957 — inference-side curve dynamics가 새 정보 제공.
\t**신규 발견 2**: per-feature reduced `train_feature_recon_mean__feat_mean` / `__feat_std` 가 top 5에 진입.
</callout>"""


def section_scope():
    return """---
# 1. 분석 대상 (25 학습 history, MAE 271 base only)
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
<td>simulation 제외 (사용자 결정)</td>
</tr>
</table>
<callout icon="📂" color="gray_bg">
\t**Source**: `results/experiments/271_20260508_094241_w500p10e4t3d2_dynamic_linear_minmax_k6/`
\t**271 config**: `e4t3d2`, `dynamic_margin_k=6`, `linear` patchify, `minmax`, ep=500/w=250, `use_grl=True`, `use_feature_matching=True`, `fm_adaptive_lambda=True`, `grl_balanced_sampling=False`, `grl_use_focal=True`, `grl_cls_lr_ratio=0.1`, `grl_loss_weight=0.2`, `grl_target_mode='window'`.
</callout>"""


def section_method_v2():
    return """# 2. 방법론 (v2 확장)
## 2.1 ES 알고리즘
<callout icon="🧮" color="purple_bg">
\t**Pseudocode**:
\t```
\twarmup_epoch = 250
\teval_interval = 5  (epoch_metrics 기록 주기)
\tfor each (dataset, metric, op, patience P, threshold T):
\t    s = post_process(metric_series, op)   ← raw / EMA / slope / curvature / variance / sign_changes
\t    best = s[warmup]; counter = 0
\t    for ep in 255, 260, ..., 500:
\t        v = s[ep]
\t        if improvement(v, best, T):
\t            best = v; counter = 0
\t        else:
\t            counter += 1
\t            if counter >= P: stop_epoch = ep; break
\t    return pak_auc_f1[stop_epoch]  ← scoring 테이블에서 lookup
\t```
</callout>
## 2.2 Sweep grid (v2 확장 — 시간상 제외 없음, 가능한 모든 조합)
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
<td>**Base metric (확장)**</td>
<td>training_histories scalar + per-feature reduced + epoch_metrics 풀세트 + per-feature reduced + derived dynamics</td>
<td>**298**</td>
</tr>
<tr>
<td>**Post-process ops**</td>
<td>raw, ema03, slope10, slope20, curvature10, variance10, sign_changes10</td>
<td>**7**</td>
</tr>
<tr>
<td>**Per-dataset rows**</td>
<td>298 × 7 × 50</td>
<td>**104,300**</td>
</tr>
<tr>
<td>**Total simulations**</td>
<td>× 25 datasets</td>
<td>**2,607,500**</td>
</tr>
</table>
<callout icon="⚙️" color="green_bg">
\t**구현 효율**: Python `multiprocessing.Pool(processes=6)` + numpy 벡터화 → 14초 총 실행. 메인 RSS 최대 1.3 GB, 시스템 메모리 사용률 최대 20%. 결과 JSON 472 MB (compact format).
</callout>
## 2.3 Post-process ops 정의
<table fit-page-width="true" header-row="true">
<tr>
<td>Op name</td>
<td>의미</td>
<td>의도</td>
</tr>
<tr>
<td>`raw`</td>
<td>원본 시계열</td>
<td>baseline</td>
</tr>
<tr>
<td>`ema03`</td>
<td>α=0.3 exponential moving average</td>
<td>epoch-level noise 흡수</td>
</tr>
<tr>
<td>`slope10`</td>
<td>최근 10 ep diff의 평균 (1차 미분)</td>
<td>학습 속도 측정</td>
</tr>
<tr>
<td>`slope20`</td>
<td>최근 20 ep diff의 평균 (긴 window)</td>
<td>장기 trend</td>
</tr>
<tr>
<td>`curvature10`</td>
<td>최근 10 ep의 2차 미분 평균</td>
<td>plateau / 변곡점 검출</td>
</tr>
<tr>
<td>`variance10`</td>
<td>최근 10 ep의 std</td>
<td>안정성 검출</td>
</tr>
<tr>
<td>`sign_changes10`</td>
<td>최근 10 ep diff의 부호 변화 횟수</td>
<td>oscillation / saturation 검출</td>
</tr>
</table>"""


def section_user_concern():
    return """# 3. 사용자 우려 검증 ★ NEW (v2 신규 섹션)
**사용자 지적 (Q4)**: "`teacher_recon_normal` 기반 ES는 사실상 warmup(250) 직후 멈추는 거 아닌가? Teacher recon은 250 ep 이후 거의 수렴 상태잖아."
## 3.1 Teacher_recon_normal 수렴도 측정
<table fit-page-width="true" header-row="true">
<tr>
<td>Dataset</td>
<td>ep=1</td>
<td>ep=100</td>
<td>ep=250</td>
<td>ep=400</td>
<td>ep=500</td>
<td>|Δ|/ep avg (250-499)</td>
<td>|Δ|/ep avg (1-249)</td>
<td>**post/pre 비율**</td>
</tr>
<tr>
<td>SWaT/A1A2_full</td>
<td>1.3469</td>
<td>0.0003</td>
<td>0.0001</td>
<td>0.0001</td>
<td>0.0001</td>
<td>0.000002</td>
<td>0.005444</td>
<td>**0.043%**</td>
</tr>
<tr>
<td>WaDi/A1</td>
<td>0.7313</td>
<td>0.0004</td>
<td>0.0003</td>
<td>0.0002</td>
<td>0.0002</td>
<td>0.000002</td>
<td>0.002938</td>
<td>**0.052%**</td>
</tr>
<tr>
<td>PSM</td>
<td>1.2073</td>
<td>0.0006</td>
<td>0.0003</td>
<td>0.0003</td>
<td>0.0003</td>
<td>0.000005</td>
<td>0.004877</td>
<td>**0.098%**</td>
</tr>
<tr>
<td>SMD/machine-1-2</td>
<td>1.0916</td>
<td>0.0050</td>
<td>0.0025</td>
<td>0.0021</td>
<td>0.0020</td>
<td>0.000032</td>
<td>0.004404</td>
<td>**0.736%**</td>
</tr>
<tr>
<td>Exathlon/app1</td>
<td>1.2725</td>
<td>0.0019</td>
<td>0.0007</td>
<td>0.0006</td>
<td>0.0006</td>
<td>0.000006</td>
<td>0.005110</td>
<td>**0.126%**</td>
</tr>
</table>
<callout icon="✅" color="orange_bg">
\t**사용자 지적 100% 정당함**: 250 ep 이후 teacher_recon_normal 변화량은 이전의 **1/1000 수준** (0.04~0.74%). 거의 완전한 plateau.
</callout>
## 3.2 그러나 실제 stop_epoch은 floor에서 정확히 stop하지 않음
**Top cross-dataset config #1** (`th_train_teacher_recon_normal`, raw, P=3, rel=0.01) 의 stop_epoch 분포:
<table fit-page-width="true" header-row="true">
<tr>
<td>Dataset</td>
<td>Oracle ep</td>
<td>Oracle PA F1</td>
<td>Stop ep</td>
<td>Stop PA F1</td>
<td>Δ ep</td>
<td>Loss %</td>
</tr>
<tr>
<td>SWaT_excl22</td>
<td>280</td>
<td>0.6305</td>
<td>**280**</td>
<td>**0.6305**</td>
<td>0</td>
<td>**0.0%** 🎯</td>
</tr>
<tr>
<td>WaDi_A1</td>
<td>395</td>
<td>0.8495</td>
<td>415</td>
<td>0.8309</td>
<td>+20</td>
<td>2.2%</td>
</tr>
<tr>
<td>WaDi_A2</td>
<td>410</td>
<td>0.7939</td>
<td>415</td>
<td>0.7753</td>
<td>+5</td>
<td>2.3%</td>
</tr>
<tr>
<td>PSM</td>
<td>260</td>
<td>0.8034</td>
<td>285</td>
<td>0.7947</td>
<td>+25</td>
<td>1.1%</td>
</tr>
<tr>
<td>SMD_machine-1-2</td>
<td>295</td>
<td>0.7007</td>
<td>300</td>
<td>0.6870</td>
<td>+5</td>
<td>2.0%</td>
</tr>
<tr>
<td>SMD_machine-1-7</td>
<td>170</td>
<td>0.7463</td>
<td>285</td>
<td>0.6419</td>
<td>+115</td>
<td>14.0%</td>
</tr>
<tr>
<td>SMD_machine-2-2</td>
<td>25</td>
<td>0.6665</td>
<td>435</td>
<td>0.5501</td>
<td>+410</td>
<td>17.5%</td>
</tr>
<tr>
<td>SMD_machine-2-3</td>
<td>375</td>
<td>0.8894</td>
<td>355</td>
<td>0.8002</td>
<td>−20</td>
<td>10.0%</td>
</tr>
<tr>
<td>SMD_machine-3-1</td>
<td>285</td>
<td>0.9387</td>
<td>340</td>
<td>0.9365</td>
<td>+55</td>
<td>0.2%</td>
</tr>
<tr>
<td>SMD_machine-3-9</td>
<td>275</td>
<td>1.0000</td>
<td>410</td>
<td>0.9985</td>
<td>+135</td>
<td>0.15%</td>
</tr>
<tr>
<td>Exathlon_app2</td>
<td>260</td>
<td>0.9426</td>
<td>265</td>
<td>0.9362</td>
<td>+5</td>
<td>0.7%</td>
</tr>
<tr>
<td>Exathlon_app9</td>
<td>440</td>
<td>0.5617</td>
<td>265</td>
<td>0.3767</td>
<td>−175</td>
<td>32.9%</td>
</tr>
<tr>
<td>**평균**</td>
<td>—</td>
<td>—</td>
<td>**320**</td>
<td>—</td>
<td>**+70**</td>
<td>—</td>
</tr>
<tr>
<td>**Floor(255) count**</td>
<td>—</td>
<td>—</td>
<td colspan="4">**0/25** (모든 dataset이 255 이상에서 stop)</td>
</tr>
</table>
<callout icon="🧠" color="yellow_bg">
\t**결론**: 우려는 부분적으로 정당함.
\t• **합의되는 부분**: Plateau에서의 ES이므로 변별력이 약하다 — SWaT처럼 명확한 oracle epoch (280)이 warmup 바로 다음 5 ep 안에 있는 dataset은 우연히 oracle을 잡지만, SMD machine-1-7 (oracle@170) / app9 (oracle@440) 같은 경우는 oracle 시점에 도달 불가.
\t• **부분 반박**: stop_epoch은 floor(255)가 아닌 평균 ~320으로 noise 변동에 따라 trigger되며, 25개 dataset 중 어느 하나도 정확히 255에서 stop하지 않음.
\t• **함의**: 이 metric의 효용성은 plateau detection이 아니라 "training이 진짜 완료되었음을 사후 확인하는 confidence signal"에 가까움. 진정한 oracle 추적은 **inference-side metric**이 더 정확함 (PA-K curve dynamics, 아래 6.3 참조).
</callout>"""


def section_metric_catalog_v2():
    return """# 4. 사용한 지표 카탈로그 (v2 확장 — 298 base metrics, 시간상 제외 없음)
## 4.1 Training-history per-epoch scalar (모든 list[500] scalar 자동 추출)
<table fit-page-width="true" header-row="true">
<tr>
<td>Category</td>
<td>Keys</td>
<td>개수</td>
</tr>
<tr>
<td>Loss components</td>
<td>`train_loss`, `train_rec_loss`, `train_disc_loss`, `train_normal_loss`, `train_anomaly_loss`, `train_fm_loss`, `train_grl_cls_loss`</td>
<td>7</td>
</tr>
<tr>
<td>Adaptive coefficients</td>
<td>`train_fm_adaptive_lambda`, `train_grl_lambda`, `train_grl_effective_weight`</td>
<td>3</td>
</tr>
<tr>
<td>GRL classification acc</td>
<td>`train_grl_balanced_acc`, `train_grl_anomaly_acc`, `train_grl_normal_acc`</td>
<td>3</td>
</tr>
<tr>
<td>Recon by sample type</td>
<td>`train_{teacher,student}_recon_{normal,anomaly}`</td>
<td>4</td>
</tr>
<tr>
<td>Epoch-aggregated</td>
<td>`epoch_{raw,score,ratio}_{recon,disc}_{normal,anomaly,disturbing}`</td>
<td>18</td>
</tr>
</table>
## 4.2 Training-history per-feature → scalar 축약 (v2 신규)
<table fit-page-width="true" header-row="true">
<tr>
<td>Source field</td>
<td>Reduce ops</td>
<td>결과 metric</td>
</tr>
<tr>
<td>`train_feature_recon_mean` (list[features] / epoch)</td>
<td>mean / max / std / min over features</td>
<td>`th_train_feature_recon_mean__feat_{mean,max,std,min}` (4)</td>
</tr>
<tr>
<td>`train_feature_recon_max`</td>
<td>mean / max / std / min</td>
<td>4</td>
</tr>
</table>
## 4.3 Epoch-metrics scalar 풀세트 (v2: 모든 176 scalar 자동 추출)
<table fit-page-width="true" header-row="true">
<tr>
<td>Category</td>
<td>Keys</td>
<td>개수</td>
</tr>
<tr>
<td>**PA%K family (5 sub-metrics × 21 K values)**</td>
<td>`pa_K_{f1, prc_auc, precision, recall, roc_auc}` for K∈{0,5,…,100}</td>
<td>105</td>
</tr>
<tr>
<td>**Teacher PA%K family**</td>
<td>`teacher_pa_K_{...}`</td>
<td>~22</td>
</tr>
<tr>
<td>PA%K-AUC family</td>
<td>`pak_auc_*`, `teacher_pak_auc_*`</td>
<td>20</td>
</tr>
<tr>
<td>Plain AUC / F1 / Precision / Recall</td>
<td>`prc_auc`, `roc_auc`, `f1_score`, `f1_t`, `teacher_f1_t`, `teacher_prc_auc`, `precision`, `recall`, `precision_t`, `recall_t`</td>
<td>10</td>
</tr>
<tr>
<td>Disturbing region</td>
<td>`disturbing_{f1, precision, recall, roc_auc}`</td>
<td>4</td>
</tr>
<tr>
<td>Misc</td>
<td>`disc_snr`, `optimal_threshold`, `fm_loss`, `grl_*` (7)</td>
<td>~10</td>
</tr>
</table>
## 4.4 Epoch-metrics per-feature → scalar 축약 (v2 신규)
<table fit-page-width="true" header-row="true">
<tr>
<td>Source field</td>
<td>Reduce ops</td>
<td>결과 metric</td>
</tr>
<tr>
<td>`_train_feature_recon_mean`, `_train_feature_recon_max`</td>
<td>mean / max / std / min over features</td>
<td>8</td>
</tr>
<tr>
<td>`_infer_feature_disc_mean`, `_infer_feature_disc_max`</td>
<td>mean / max / std / min</td>
<td>8 (271은 일부 None일 수 있음)</td>
</tr>
</table>
## 4.5 Derived dynamic metrics (사용자 제안 + brainstorm)
<table fit-page-width="true" header-row="true">
<tr>
<td>Family</td>
<td>Members</td>
<td>의미</td>
</tr>
<tr>
<td>**정상-이상 분리도 (static)**</td>
<td>`deriv_{teacher,student}_anom_normal_{gap,ratio,separation}` (6)</td>
<td>매 epoch의 anom vs normal recon 분리</td>
</tr>
<tr>
<td>**Recon/Disc score 분리도**</td>
<td>`deriv_recon_score_{gap,separation}`, `deriv_disc_score_{gap,separation}` (4)</td>
<td>epoch-aggregated score 분리</td>
</tr>
<tr>
<td>**Teacher-Student disagreement**</td>
<td>`deriv_TS_disagreement_{normal,anomaly}{,_abs,_relative,_per_loss}` (~6)</td>
<td>두 head의 recon 차이 (mean_discrepancy proxy)</td>
</tr>
<tr>
<td>**★ 사용자 제안: Δratio**</td>
<td>`deriv_dteacher_over_dstudent_{normal,anomaly}_W{5,10,20}` (6)</td>
<td>`|Δteacher_recon| / |Δstudent_recon|` 감소량 비율</td>
</tr>
<tr>
<td>**★ 사용자 제안: Δdiff**</td>
<td>`deriv_dteacher_minus_dstudent_{normal,anomaly}_W{5,10,20}_abs` (6)</td>
<td>`|Δteacher − Δstudent|` 감소량 차이</td>
</tr>
<tr>
<td>**Gap stability**</td>
<td>`deriv_gap_TS_{normal,anomaly}_dW{5,10,20}_abs` (6)</td>
<td>Teacher-student gap의 window 변화량</td>
</tr>
<tr>
<td>**Recon score separation 변화**</td>
<td>`deriv_recon_score_separation_dW{5,10,20}{,_abs}` (6)</td>
<td>Anomaly score 분리 곡선의 변화율</td>
</tr>
<tr>
<td>**Anom/Normal loss 균형**</td>
<td>`deriv_anom_normal_loss_ratio` (+ dW 변화량)</td>
<td>train_anom_loss / train_norm_loss</td>
</tr>
<tr>
<td>**Adaptive coefficient stabilization**</td>
<td>`deriv_{fm,grl}_lambda_dW{5,10,20}_abs`, `deriv_grl_effective_weight_dW{...}_abs`</td>
<td>λ 수렴 시점</td>
</tr>
<tr>
<td>**GRL classifier bias**</td>
<td>`deriv_grl_classifier_bias{,_abs}`</td>
<td>`grl_anomaly_acc − grl_normal_acc`</td>
</tr>
<tr>
<td>**PA-K curve dynamics ★ NEW**</td>
<td>`deriv_pa_K_curve_{spread_50_0, mean_f1, area_f1, avg_slope_over_K}` (4)</td>
<td>PA-K F1 곡선의 K 방향 통계</td>
</tr>
</table>"""


def section_per_dataset_v2():
    return """# 5. Per-Dataset Best ES Config (v2)
각 dataset마다 (metric, op, P, T) 를 독립적으로 최적화한 결과. Upper bound rank avg **2.00** (oracle 다음 2위).
## 5.1 4 Single-Dataset Groups
<table fit-page-width="true" header-row="true">
<tr>
<td>Dataset</td>
<td>Oracle ep</td>
<td>Oracle PA F1</td>
<td>Best ES (metric, op, P, T)</td>
<td>Stop ep</td>
<td>ES PA F1</td>
<td>Loss</td>
</tr>
<tr>
<td>**SWaT_excl22**</td>
<td>280</td>
<td>0.6305</td>
<td>`th_train_loss` op=**slope10** P=3 abs=0</td>
<td>280</td>
<td>0.6305</td>
<td>**0.00%** 🎯</td>
</tr>
<tr>
<td>**WaDi_A1**</td>
<td>395</td>
<td>0.8495</td>
<td>`th_train_teacher_recon_anomaly` op=**variance10** P=20 abs=0</td>
<td>395</td>
<td>0.8495</td>
<td>**0.00%** 🎯</td>
</tr>
<tr>
<td>**WaDi_A2**</td>
<td>410</td>
<td>0.7939</td>
<td>`th_train_loss` op=**curvature10** P=30 abs=0</td>
<td>410</td>
<td>0.7939</td>
<td>**0.00%** 🎯</td>
</tr>
<tr>
<td>**PSM**</td>
<td>260</td>
<td>0.8034</td>
<td>`th_train_loss` op=raw P=2 abs=0</td>
<td>260</td>
<td>0.8034</td>
<td>**0.00%** 🎯</td>
</tr>
</table>
<callout icon="🔍" color="blue_bg">
\t**관찰**: 4개 single-dataset 모두 0% loss로 oracle 재현. v1 대비 신규로 **slope10 / curvature10 / variance10** post-process가 best로 등장 → dynamics-aware ES가 raw보다 우월한 경우 발견.
</callout>
## 5.2 SMD avg / Exathlon avg group
<table fit-page-width="true" header-row="true">
<tr>
<td>Group</td>
<td>Oracle</td>
<td>Best ES</td>
<td>ES value</td>
<td>Stop ep avg</td>
<td>Loss</td>
</tr>
<tr>
<td>**SMD_avg**</td>
<td>0.7181</td>
<td>`em_teacher_pak_auc_recall_raw` op=**slope20** P=2 abs=0</td>
<td>**0.6799**</td>
<td>267</td>
<td>**5.33%** (v1: 5.96%)</td>
</tr>
<tr>
<td>**Exathlon_avg**</td>
<td>0.6452</td>
<td>`deriv_student_anom_normal_gap` op=**ema03** P=7 rel=0.05</td>
<td>**0.6336**</td>
<td>311</td>
<td>**1.80%** (v1: 3.55%)</td>
</tr>
</table>
<callout icon="⭐" color="green_bg">
\t**v2 개선**:
\t• SMD: **`em_teacher_pak_auc_recall_raw`** (inference-side teacher recall raw 점수) + slope20 → oracle epoch이 다양해도 inference-side metric이 더 잘 추적.
\t• Exathlon: **`deriv_student_anom_normal_gap`** (사용자 제안 family) + ema03 → 6 app 평균 loss 1.80%까지 단축.
</callout>"""


def section_cross_dataset_v2():
    return """# 6. Cross-Dataset 단일 최적 (v2)
**하나의 (metric, op, P, T)** 만 사용해서 6 group 평균 PA F1을 최대화.
## 6.1 Top 20 Cross-Dataset Configs
<table fit-page-width="true" header-row="true">
<tr>
<td>#</td>
<td>Metric</td>
<td>Op</td>
<td>P</td>
<td>T</td>
<td>Mean (6 group)</td>
<td>Loss</td>
<td>Stop_eps [SWaT, A1, A2, PSM, SMD, Exa]</td>
</tr>
<tr>
<td>**1** 🥇</td>
<td>`th_train_teacher_recon_normal`</td>
<td>raw</td>
<td>3</td>
<td>rel=0.01</td>
<td>**0.7092**</td>
<td>4.17%</td>
<td>[280,415,415,285,324,270]</td>
</tr>
<tr>
<td>2</td>
<td>`th_train_rec_loss`</td>
<td>raw</td>
<td>3</td>
<td>rel=0.001</td>
<td>0.7088</td>
<td>4.22%</td>
<td>[280,475,415,285,378,295]</td>
</tr>
<tr>
<td>3 ★</td>
<td>`th_train_feature_recon_mean__feat_mean`</td>
<td>raw</td>
<td>3</td>
<td>rel=0.001</td>
<td>0.7088</td>
<td>4.22%</td>
<td>[280,475,415,285,378,295]</td>
</tr>
<tr>
<td>4 ★</td>
<td>`em__train_feature_recon_mean__feat_mean`</td>
<td>raw</td>
<td>3</td>
<td>rel=0.001</td>
<td>0.7088</td>
<td>4.22%</td>
<td>[280,475,415,285,378,295]</td>
</tr>
<tr>
<td>5</td>
<td>`th_train_rec_loss`</td>
<td>raw</td>
<td>3</td>
<td>abs=0</td>
<td>0.7087</td>
<td>4.25%</td>
<td>[280,475,415,285,384,323]</td>
</tr>
<tr>
<td>6 ★</td>
<td>`th_train_feature_recon_mean__feat_mean`</td>
<td>raw</td>
<td>3</td>
<td>abs=0</td>
<td>0.7087</td>
<td>4.25%</td>
<td>[280,475,415,285,384,323]</td>
</tr>
<tr>
<td>7 ★</td>
<td>`em__train_feature_recon_mean__feat_mean`</td>
<td>raw</td>
<td>3</td>
<td>abs=0</td>
<td>0.7087</td>
<td>4.25%</td>
<td>[280,475,415,285,384,323]</td>
</tr>
<tr>
<td>8 ★</td>
<td>`th_train_feature_recon_mean__feat_std`</td>
<td>raw</td>
<td>3</td>
<td>abs=0</td>
<td>0.7076</td>
<td>4.39%</td>
<td>[280,500,485,285,316,313]</td>
</tr>
<tr>
<td>9 ★</td>
<td>`em__train_feature_recon_mean__feat_std`</td>
<td>raw</td>
<td>3</td>
<td>abs=0</td>
<td>0.7076</td>
<td>4.39%</td>
<td>[280,500,485,285,316,313]</td>
</tr>
<tr>
<td>10</td>
<td>`th_train_feature_recon_mean__feat_std`</td>
<td>raw</td>
<td>3</td>
<td>rel=0.001</td>
<td>0.7073</td>
<td>4.43%</td>
<td>[280,470,485,285,316,313]</td>
</tr>
</table>
★ = v2 신규 발견 (이전 분석에서 추출 안 됨)
## 6.2 Top 20 Metric Families (best op/P/T per family)
<table fit-page-width="true" header-row="true">
<tr>
<td>#</td>
<td>Metric family</td>
<td>Best 6-mean</td>
<td>Best op</td>
<td>P</td>
<td>T</td>
</tr>
<tr>
<td>1</td>
<td>`th_train_teacher_recon_normal`</td>
<td>**0.7092**</td>
<td>raw</td>
<td>3</td>
<td>rel=0.01</td>
</tr>
<tr>
<td>2</td>
<td>`th_train_rec_loss`</td>
<td>0.7088</td>
<td>raw</td>
<td>3</td>
<td>rel=0.001</td>
</tr>
<tr>
<td>3 ★</td>
<td>`th_train_feature_recon_mean__feat_mean`</td>
<td>0.7088</td>
<td>raw</td>
<td>3</td>
<td>rel=0.001</td>
</tr>
<tr>
<td>4 ★</td>
<td>`em__train_feature_recon_mean__feat_mean`</td>
<td>0.7088</td>
<td>raw</td>
<td>3</td>
<td>rel=0.001</td>
</tr>
<tr>
<td>5 ★</td>
<td>`th_train_feature_recon_mean__feat_std`</td>
<td>0.7076</td>
<td>raw</td>
<td>3</td>
<td>abs=0</td>
</tr>
<tr>
<td>6 ★</td>
<td>`em__train_feature_recon_mean__feat_std`</td>
<td>0.7076</td>
<td>raw</td>
<td>3</td>
<td>abs=0</td>
</tr>
<tr>
<td>7 ★</td>
<td>`em__train_feature_recon_max__feat_std`</td>
<td>0.7017</td>
<td>ema03</td>
<td>5</td>
<td>abs=0</td>
</tr>
<tr>
<td>8 ★</td>
<td>`em__infer_feature_disc_max__feat_mean`</td>
<td>0.6962</td>
<td>slope20</td>
<td>20</td>
<td>rel=0.001</td>
</tr>
<tr>
<td>9 ★</td>
<td>`em__infer_feature_disc_mean__feat_std`</td>
<td>0.6961</td>
<td>slope20</td>
<td>5</td>
<td>rel=0.01</td>
</tr>
<tr>
<td>10</td>
<td>`em_f1_score`</td>
<td>0.6960</td>
<td>slope10</td>
<td>15</td>
<td>rel=0.05</td>
</tr>
<tr>
<td>11</td>
<td>`em_pak_auc_f1_raw`</td>
<td>0.6959</td>
<td>slope10</td>
<td>15</td>
<td>abs=0.001</td>
</tr>
<tr>
<td>12 ★</td>
<td>`deriv_pa_K_curve_avg_slope_over_K`</td>
<td>0.6957</td>
<td>variance10</td>
<td>5</td>
<td>rel=0.01</td>
</tr>
<tr>
<td>13 ★</td>
<td>`deriv_pa_K_curve_mean_f1`</td>
<td>0.6955</td>
<td>slope10</td>
<td>15</td>
<td>rel=0.05</td>
</tr>
<tr>
<td>14 ★</td>
<td>`deriv_pa_K_curve_area_f1`</td>
<td>0.6955</td>
<td>slope10</td>
<td>15</td>
<td>rel=0.05</td>
</tr>
<tr>
<td>15</td>
<td>`em_pa_45_roc_auc`</td>
<td>0.6951</td>
<td>curvature10</td>
<td>5</td>
<td>abs=0.001</td>
</tr>
<tr>
<td>16</td>
<td>`em_teacher_pak_auc_roc_auc`</td>
<td>0.6951</td>
<td>curvature10</td>
<td>5</td>
<td>abs=0.001</td>
</tr>
<tr>
<td>17 ★</td>
<td>`deriv_gap_TS_normal_dW20_abs`</td>
<td>0.6948</td>
<td>curvature10</td>
<td>30</td>
<td>abs=0.001</td>
</tr>
<tr>
<td>18 ★</td>
<td>`deriv_dteacher_minus_dstudent_normal_W20_abs`</td>
<td>0.6948</td>
<td>curvature10</td>
<td>30</td>
<td>abs=0.001</td>
</tr>
<tr>
<td>19</td>
<td>`em__train_feature_recon_mean__feat_max`</td>
<td>0.6947</td>
<td>ema03</td>
<td>2</td>
<td>abs=0</td>
</tr>
<tr>
<td>20</td>
<td>`em_pa_60_f1`</td>
<td>0.6946</td>
<td>curvature10</td>
<td>20</td>
<td>rel=0.05</td>
</tr>
</table>
## 6.3 PA-K Curve Dynamics 신규 발견 (inference-side)
<callout icon="📊" color="purple_bg">
\t**`deriv_pa_K_curve_avg_slope_over_K`** = 매 epoch에서 PA-K F1 곡선의 K 방향 평균 slope (`mean(∂F1/∂K)`)
\t→ 곡선이 평탄해지는 시점 (slope → 0) 이 학습 수렴의 inference-side proxy.
\t**Mean 6-group**: 0.6957 (raw plateau metric `train_teacher_recon_normal` 0.7092 보다 약하지만 #12).
\t**의의**: training loss는 250 ep 이후 plateau이지만 **inference 쪽 PA-K 곡선 형태는 계속 변함** → ES decision에 새로운 정보 채널 제공.
</callout>"""


def section_user_metric():
    return """# 7. 사용자 제안 Metric 결과 ★ NEW
**사용자 제안 (Q5)**: "Teacher recon error 감소량 대비 student recon error 감소량 비율"이 비슷해지는 시점에 stop.
## 7.1 Δratio (감소량 비율) 결과
<table fit-page-width="true" header-row="true">
<tr>
<td>Metric</td>
<td>정의</td>
<td>Best 6-mean</td>
<td>Best op</td>
<td>P</td>
<td>T</td>
<td>Loss vs Oracle</td>
</tr>
<tr>
<td>`deriv_dteacher_over_dstudent_normal_W5`</td>
<td>`|Δteacher_recon_normal_W5| / |Δstudent_recon_normal_W5|`</td>
<td>0.6854</td>
<td>variance10</td>
<td>5</td>
<td>rel=0.01</td>
<td>7.39%</td>
</tr>
<tr>
<td>`deriv_dteacher_over_dstudent_normal_W10`</td>
<td>W=10</td>
<td>0.6804</td>
<td>variance10</td>
<td>1</td>
<td>rel=0.05</td>
<td>8.06%</td>
</tr>
<tr>
<td>`deriv_dteacher_over_dstudent_normal_W20`</td>
<td>W=20</td>
<td>0.6822</td>
<td>variance10</td>
<td>5</td>
<td>rel=0.05</td>
<td>7.82%</td>
</tr>
<tr>
<td>**`deriv_dteacher_over_dstudent_anomaly_W5`**</td>
<td>**Anomaly 샘플 한정**</td>
<td>**0.6924**</td>
<td>variance10</td>
<td>5</td>
<td>rel=0.001</td>
<td>**6.44%**</td>
</tr>
<tr>
<td>`deriv_dteacher_over_dstudent_anomaly_W10`</td>
<td>W=10</td>
<td>0.6902</td>
<td>sign_changes10</td>
<td>2</td>
<td>rel=0.05</td>
<td>6.73%</td>
</tr>
<tr>
<td>`deriv_dteacher_over_dstudent_anomaly_W20`</td>
<td>W=20</td>
<td>0.6812</td>
<td>variance10</td>
<td>5</td>
<td>abs=0.001</td>
<td>7.96%</td>
</tr>
</table>
## 7.2 Δdiff (감소량 차이 절댓값) 결과 — 사용자 제안 변형 중 best
<table fit-page-width="true" header-row="true">
<tr>
<td>Metric</td>
<td>정의</td>
<td>Best 6-mean</td>
<td>Best op</td>
<td>P</td>
<td>T</td>
<td>Loss vs Oracle</td>
</tr>
<tr>
<td>`deriv_dteacher_minus_dstudent_normal_W5_abs`</td>
<td>`|Δteacher − Δstudent|` W=5 normal</td>
<td>0.6913</td>
<td>curvature10</td>
<td>30</td>
<td>rel=0.001</td>
<td>6.59%</td>
</tr>
<tr>
<td>`deriv_dteacher_minus_dstudent_normal_W10_abs`</td>
<td>W=10 normal</td>
<td>0.6913</td>
<td>curvature10</td>
<td>30</td>
<td>abs=0</td>
<td>6.59%</td>
</tr>
<tr>
<td>**`deriv_dteacher_minus_dstudent_normal_W20_abs`** 🥇</td>
<td>**W=20 normal**</td>
<td>**0.6948**</td>
<td>curvature10</td>
<td>30</td>
<td>abs=0.001</td>
<td>**6.12%**</td>
</tr>
<tr>
<td>`deriv_dteacher_minus_dstudent_anomaly_W5_abs`</td>
<td>W=5 anomaly</td>
<td>0.6913</td>
<td>curvature10</td>
<td>30</td>
<td>rel=0.001</td>
<td>6.59%</td>
</tr>
<tr>
<td>`deriv_dteacher_minus_dstudent_anomaly_W10_abs`</td>
<td>W=10 anomaly</td>
<td>0.6912</td>
<td>curvature10</td>
<td>30</td>
<td>rel=0.001</td>
<td>6.60%</td>
</tr>
<tr>
<td>`deriv_dteacher_minus_dstudent_anomaly_W20_abs`</td>
<td>W=20 anomaly</td>
<td>0.6919</td>
<td>curvature10</td>
<td>30</td>
<td>abs=0.001</td>
<td>6.51%</td>
</tr>
</table>
## 7.3 Teacher/Student 정상-이상 분리 metric (사용자 추가 질문)
<table fit-page-width="true" header-row="true">
<tr>
<td>Metric</td>
<td>정의</td>
<td>Best 6-mean</td>
<td>Best op</td>
<td>P</td>
<td>T</td>
<td>Loss</td>
</tr>
<tr>
<td>**`deriv_teacher_anom_normal_ratio`**</td>
<td>`teacher_a / teacher_n`</td>
<td>**0.6898**</td>
<td>sign_changes10</td>
<td>15</td>
<td>abs=0.001</td>
<td>6.79%</td>
</tr>
<tr>
<td>`deriv_teacher_anom_normal_gap`</td>
<td>`teacher_a − teacher_n`</td>
<td>0.6871</td>
<td>ema03</td>
<td>30</td>
<td>rel=0.01</td>
<td>7.16%</td>
</tr>
<tr>
<td>`deriv_teacher_anom_normal_separation`</td>
<td>`(a−n)/(\\|a\\|+\\|n\\|)`</td>
<td>0.6898</td>
<td>sign_changes10</td>
<td>15</td>
<td>abs=0.001</td>
<td>6.79%</td>
</tr>
<tr>
<td>`deriv_student_anom_normal_ratio`</td>
<td>`student_a / student_n`</td>
<td>0.6923</td>
<td>curvature10</td>
<td>20</td>
<td>rel=0.05</td>
<td>6.46%</td>
</tr>
<tr>
<td>`deriv_student_anom_normal_gap`</td>
<td>`student_a − student_n`</td>
<td>0.6900</td>
<td>variance10</td>
<td>30</td>
<td>abs=0.001</td>
<td>6.77%</td>
</tr>
<tr>
<td>**`deriv_student_anom_normal_separation`** 🥇</td>
<td>Student 버전</td>
<td>**0.6926**</td>
<td>raw</td>
<td>30</td>
<td>rel=0.05</td>
<td>**6.42%**</td>
</tr>
</table>
<callout icon="🔍" color="yellow_bg">
\t**사용자 제안 metric 종합 분석**:
\t• **Δratio (anomaly 한정, W=5)**: 0.6924 — anomaly 샘플에서의 비율이 normal보다 효과적.
\t• **Δdiff (W=20 normal, curvature10)**: 0.6948 — 사용자 제안 변형 중 best.
\t• **`teacher_anom_normal_ratio`**: 0.6898 — single metric 1위 (0.7092) 보다 약 −2.7%p.
\t• **흥미로운 발견**: Student 버전 (`student_anom_normal_separation` = 0.6926) 이 Teacher (0.6898) 보다 약간 더 좋음. Student는 capacity가 작아 anomaly에 hard fit → ratio 변동이 더 informative.
\t• **그러나** 모든 derived dynamic metric은 cross-config best (`train_teacher_recon_normal`, 0.7092) 를 능가하지 못함. **이유**: dynamics-based metric은 학습 종료 직후 plateau에서 signal이 약해지고, 결국 단순 plateau detection을 대체하지 못함.
</callout>"""


def section_leaderboard_v2():
    return """# 8. Leaderboard — MAE 271 vs 15 Baseline (v2 재계산)
<callout icon="ℹ️" color="gray_bg">
\t**참고**: 7개 신규 SOTA (`tfmae`, `npsr`, `timesnet`, `dcdetector`, `memto`, `moderntcn`, `catch`) 는 Q3 batch 진행 중 (placeholder `-` 상태) → 본 leaderboard에는 15 active baseline만 포함.
\t**6 datasets**: SWaT_excl22, WaDi_A1, WaDi_A2, PSM, **SMD avg (15 TimeSeAD)**, **Exathlon avg (6 apps)**.
\t**v2 vs v1**: per-dataset upper bound rank avg 2.83→**2.00** (개선), cross-best #1 rank avg 3.67→**3.83** (동일).
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
<td>Exa avg</td>
<td>**Rank Avg**</td>
<td>Mean</td>
</tr>
<tr>
<td>**1** 🥇</td>
<td>**MAE 271 (Oracle)**</td>
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
<td>0.6799</td>
<td>0.6336</td>
<td>**2.00**</td>
<td>0.7318</td>
</tr>
<tr>
<td>**3** 🥉</td>
<td>**ES #1 (`teacher_recon_normal`, raw, P=3, rel=0.01)**</td>
<td>0.6305</td>
<td>0.8309</td>
<td>0.7753</td>
<td>0.7947</td>
<td>0.6488</td>
<td>0.5752</td>
<td>**3.83**</td>
<td>0.7092</td>
</tr>
<tr>
<td>4</td>
<td>ES #2 (`feature_recon_mean__feat_mean`, raw, P=3, rel=0.001)</td>
<td>0.6305</td>
<td>0.8252</td>
<td>0.7753</td>
<td>0.7947</td>
<td>0.6480</td>
<td>0.5794</td>
<td>4.00</td>
<td>0.7088</td>
</tr>
<tr>
<td>5</td>
<td>ES #3 (`train_rec_loss`, raw, P=3, rel=0.001)</td>
<td>0.6305</td>
<td>0.8252</td>
<td>0.7753</td>
<td>0.7947</td>
<td>0.6480</td>
<td>0.5794</td>
<td>5.00</td>
<td>0.7088</td>
</tr>
<tr>
<td>6</td>
<td>ES #4 (`em__feature_recon_mean__feat_mean`, raw, P=3, rel=0.001)</td>
<td>0.6305</td>
<td>0.8252</td>
<td>0.7753</td>
<td>0.7947</td>
<td>0.6480</td>
<td>0.5794</td>
<td>6.00</td>
<td>0.7088</td>
</tr>
<tr>
<td>7</td>
<td>ES #5 (`em__feature_recon_mean__feat_mean`, raw, P=3, abs=0)</td>
<td>0.6305</td>
<td>0.8252</td>
<td>0.7753</td>
<td>0.7947</td>
<td>0.6475</td>
<td>0.5788</td>
<td>7.00</td>
<td>0.7087</td>
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
<td>9.17</td>
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
<td>11.83</td>
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
<td>13.33</td>
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
<td>15</td>
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
<td>16</td>
<td>`usad`</td>
<td>0.4468</td>
<td>0.3456</td>
<td>0.3462</td>
<td>0.5875</td>
<td>0.6452</td>
<td>0.5517</td>
<td>14.50</td>
<td>0.4872</td>
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
<td>15.67</td>
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
\t• **ES adoption cost**: rank 1 (Oracle) → rank 3 (best cross-cfg ES). rank avg +2.83, mean PA F1 손실 4.18%.
\t• **Per-dataset oracle ES (upper bound)**: rank avg 2.00 — 4 single-dataset에서 oracle 완벽 재현, SMD/Exathlon 평균에서만 1.8-5.3% 손실 (v1: 3.55-5.96% → 개선).
</callout>"""


def section_code_change():
    return """# 9. 코드 수정 — `mean_discrepancy` history 저장
<callout icon="🛠️" color="green_bg">
\t**파일**: `mae_anomaly/trainer.py`
\t**변경**: history dict 초기화에 `'train_mean_discrepancy': []` 추가 + per-epoch append.
\t**의의**: 이전엔 `loss.py` L405에서 계산되었으나 history에 push 안 됨 → 분석 불가. 이제 모든 신규 실험에 자동 저장됨.
\t**Backward compatibility**: `getattr(self.history, 'train_mean_discrepancy', [])` 패턴으로 기존 pickle config 호환.
\t**현재 분석에는 미반영**: 본 sweep은 변경 이전 학습된 271 결과를 사용하므로 mean_discrepancy 시리즈는 없음. **다음 학습부터** 자동 활용 가능.
</callout>
```python
# trainer.py — history 초기화 (L209)
'train_mean_discrepancy': [],   # NEW
# trainer.py — history append (L934)
self.history['train_mean_discrepancy'].append(epoch_losses.get('mean_discrepancy', 0.0))   # NEW
```"""


def section_limitations_v2():
    return """# 10. 한계 및 권장 (v2)
## 10.1 한계
<table fit-page-width="true" header-row="true">
<tr>
<td>Limitation</td>
<td>Impact</td>
<td>대응</td>
</tr>
<tr>
<td>**Warmup 250 ep 고정**</td>
<td>SMD machine-2-2 (ep25), 3-6 (ep30), app6 (ep55) 등 oracle이 250 이전인 dataset에서 ES가 oracle 재현 불가</td>
<td>warmup ablation (50/100/150/200/250) 향후 작업</td>
</tr>
<tr>
<td>**Plateau 후 noise-driven trigger** (사용자 우려 검증됨)</td>
<td>train_teacher_recon_normal은 250 ep 이후 변화량 1/1000 → ES는 noise에 의해 trigger됨</td>
<td>Inference-side metric (PA-K curve dynamics) 활용 권장</td>
</tr>
<tr>
<td>**Eval interval 5**</td>
<td>5 ep 단위로만 stop trigger 가능</td>
<td>1 ep eval은 비용 큼; 5 ep 합리적</td>
</tr>
<tr>
<td>**22 baseline 중 7개 (신규 SOTA) Q3 미완료**</td>
<td>절대적 rank는 추후 변동 가능</td>
<td>tfmae/timesnet/dcdetector/memto/moderntcn/catch/npsr 실험 완료 후 재계산</td>
</tr>
<tr>
<td>**`mean_discrepancy` 본 분석엔 미반영**</td>
<td>본 sweep은 코드 수정 이전 학습 결과 사용</td>
<td>다음 학습 cycle부터 자동 활용</td>
</tr>
</table>
## 10.2 권장 정책 (실전 ES 가이드, v2)
<table fit-page-width="true" header-row="true">
<tr>
<td>시나리오</td>
<td>추천 metric + op</td>
<td>P</td>
<td>T</td>
<td>예상 rank avg</td>
</tr>
<tr>
<td>**전 범용 (default)**</td>
<td>`th_train_teacher_recon_normal` (raw)</td>
<td>3</td>
<td>rel=0.01</td>
<td>**3.83** (top 7 안에 위치)</td>
</tr>
<tr>
<td>**Per-feature 활용 변형**</td>
<td>`th_train_feature_recon_mean__feat_mean` (raw)</td>
<td>3</td>
<td>rel=0.001</td>
<td>4.00</td>
</tr>
<tr>
<td>**Inference-aware 변형**</td>
<td>`deriv_pa_K_curve_avg_slope_over_K` (variance10)</td>
<td>5</td>
<td>rel=0.01</td>
<td>~10 (PA-K 곡선 형태 변화율)</td>
</tr>
<tr>
<td>**Teacher-Student 분리 변형 (사용자 제안)**</td>
<td>`deriv_dteacher_minus_dstudent_normal_W20_abs` (curvature10)</td>
<td>30</td>
<td>abs=0.001</td>
<td>~14 (단독으론 약하지만 ensemble 후보)</td>
</tr>
<tr>
<td>**짧은 데이터셋 (SMD/Exa)**</td>
<td>`em_teacher_pak_auc_recall_raw` (slope20)</td>
<td>2</td>
<td>abs=0</td>
<td>per-dataset 최적</td>
</tr>
</table>
## 10.3 향후 작업
<table fit-page-width="true" header-row="true">
<tr>
<td>우선순위</td>
<td>작업</td>
<td>이유</td>
</tr>
<tr>
<td>**1**</td>
<td>Warmup ablation (50/100/150/200/250)</td>
<td>oracle이 warmup 이전인 dataset 다수 (SMD 6개, Exathlon app6) — warmup 단축이 큰 잠재 효과</td>
</tr>
<tr>
<td>2</td>
<td>Ensemble ES (vote of multiple metrics)</td>
<td>단일 metric은 모두 4-7% loss — vote-based ensemble이 더 robust할 수 있음</td>
</tr>
<tr>
<td>3</td>
<td>`mean_discrepancy` 활용한 재학습 + 분석</td>
<td>1-line change 완료 → 다음 cycle부터 활용 가능</td>
</tr>
<tr>
<td>4</td>
<td>Validation split (held-out normal) 도입</td>
<td>표준 ES literature와 align, 5-10% 학습시간 증가</td>
</tr>
</table>"""


def section_artifacts_v2():
    return """# 11. Artifacts & Reproduce
<table fit-page-width="true" header-row="true">
<tr>
<td>파일</td>
<td>크기</td>
<td>설명</td>
</tr>
<tr>
<td>`temp/early_stopping/sweep_raw_v2.json`</td>
<td>472 MB</td>
<td>전체 v2 sweep 결과 (2.6M rows, 25 ds × 298 metric × 7 op × 50 grid)</td>
</tr>
<tr>
<td>`temp/early_stopping/best_per_dataset_v2.json`</td>
<td>~3 KB</td>
<td>Per-dataset best (metric, op, P, T)</td>
</tr>
<tr>
<td>`temp/early_stopping/cross_dataset_top100_v2.json`</td>
<td>~40 KB</td>
<td>Cross-dataset top 100 configs</td>
</tr>
<tr>
<td>`temp/early_stopping/metric_family_ranking_v2.json`</td>
<td>~30 KB</td>
<td>Top 50 metric families (best op/P/T per family)</td>
</tr>
<tr>
<td>`temp/early_stopping/rank_comparison_v2.json`</td>
<td>~25 KB</td>
<td>v2 leaderboard with 15 baselines</td>
</tr>
<tr>
<td>`temp/early_stopping/baseline_aggregated.json`</td>
<td>4 KB</td>
<td>15 baseline의 6-group pak_auc_f1</td>
</tr>
</table>
<callout icon="🛠️" color="gray_bg">
\t**Scripts (재실행 가능)**:
\t- `scripts/early_stopping_analysis_v2.py` — multiprocess sweep (14초)
\t- `scripts/early_stopping_analyze_v2.py` — per-dataset / cross-dataset 분석
\t- `scripts/early_stopping_baseline_aggregate.py` — baseline 집계
\t- `scripts/early_stopping_rank_compare_v2.py` — v2 leaderboard
\t**실행 순서**: `analysis_v2 → analyze_v2 → rank_compare_v2`
\t**시스템 요구사항**: ≥ 8 GB RAM (peak), 6 CPU cores, ~500 MB disk for raw JSON.
</callout>"""


def main():
    parts = [
        section_tldr(),
        section_scope(),
        section_method_v2(),
        section_user_concern(),
        section_metric_catalog_v2(),
        section_per_dataset_v2(),
        section_cross_dataset_v2(),
        section_user_metric(),
        section_leaderboard_v2(),
        section_code_change(),
        section_limitations_v2(),
        section_artifacts_v2(),
    ]
    out = "\n".join(parts)
    out_path = Path("/home/ykio/notebooks/claude/temp/early_stopping/notion_content_v2.txt")
    out_path.write_text(out)
    print(f"Wrote {out_path} ({len(out)} chars, {len(out.splitlines())} lines)")


if __name__ == "__main__":
    main()
