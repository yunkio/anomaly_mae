"""Build final v4 Notion content."""
from pathlib import Path


def part1():
    return """<callout icon="🎯" color="blue_bg">
\t**MAE 271 Early Stopping 분석 v4 — 최종 보고서 (2026-05-23)**
\t**핵심 결과**: Oracle 6-group mean **0.7401**. 최고 ES config = `th_train_rec_loss` (raw, peak_reversal, P=3, rel=0.001) → mean **0.7131** (oracle 대비 3.65%↓). 15 baseline 중 rank 1 (`tranad` 0.6205 대비 +14.9%p).
\t**v4 신규 — 직접 고안한 composite metric 10개**: `learning_phase_indicator`, `disc_x_separation_product`, `unified_anomaly_health` 등. 모두 plateau 단일 metric (0.7131)을 못 넘음 (best composite: 0.6914). **이유**: 학습 곡선이 plateau-dominated, anomaly dynamic transition이 약함.
\t**Sweep 규모**: label-free 93 metrics × 5 ops × 3 dir × 2 rollback × 2 rule × 3 P × 5 T = 83,700 rows/ds × 25 ds = **2.1M simulations** (10초, multiprocess).
\t**제약 반영**: ① Label-기반 evaluation metric 모두 제거 ② Patience 1-3만 사용 ③ peak_reversal rule (Type B signal 검출) 신규 추가 ④ Direction (auto/force_max/force_min) + Rollback (stop/best_seen) mode 신규.
\t**Phase 분석 (warmup=250 반영)**: 25 dataset 중 oracle epoch이 250 이전 = **10개 (40%)** — 이건 warmup 정책상 절대 잡을 수 없음. 250-300 = 9개 (36%, peak_reversal P=1로 잘 잡힘). 300+ = 6개 (24%, plateau metric 필요).
</callout>
---
# 1. 모델 / 구성요소 목적 재분석 (anomaly detection 관점)
## 1.1 Self-Distilled MAE 학습 phase (warmup=250 반영)
<table fit-page-width="true" header-row="true">
<tr>
<td>Phase</td>
<td>시기 (절대 epoch)</td>
<td>모델 상태</td>
<td>이상탐지 영향</td>
<td>해당 dataset 수</td>
</tr>
<tr>
<td>**Warmup**</td>
<td>0-249</td>
<td>학습 진행, ES 비활성</td>
<td>모든 head 정상/이상 동시 학습 — 분리도 형성 중</td>
<td>—</td>
</tr>
<tr>
<td>**Pre-warmup oracle**</td>
<td>oracle ep < 250</td>
<td>250 시점에 이미 phase 3 진입</td>
<td>warmup 정책상 절대 잡을 수 없음 — 손실 5-32%</td>
<td>**10/25 (40%)**</td>
</tr>
<tr>
<td>**Just-after-warmup**</td>
<td>oracle 250-300</td>
<td>250 직후 5-50 ep에 분리도 peak</td>
<td>**peak_reversal P=1**이 매우 효과적</td>
<td>9/25 (36%)</td>
</tr>
<tr>
<td>**Late phase 2**</td>
<td>oracle 300+</td>
<td>250 이후에도 분리도 계속 ↑</td>
<td>patience 길게 또는 plateau metric 필요</td>
<td>6/25 (24%)</td>
</tr>
</table>
## 1.2 25 dataset Oracle epoch 분포
<table fit-page-width="true" header-row="true">
<tr>
<td>Dataset</td>
<td>Oracle ep</td>
<td>Oracle PAF1</td>
<td>Phase 분류</td>
</tr>
<tr><td>SMD_machine-2-2</td><td>25</td><td>0.6665</td><td>Pre-warmup ★</td></tr>
<tr><td>SMD_machine-3-6</td><td>30</td><td>0.8138</td><td>Pre-warmup</td></tr>
<tr><td>Exathlon_app6</td><td>55</td><td>0.2537</td><td>Pre-warmup</td></tr>
<tr><td>SMD_machine-2-1</td><td>60</td><td>0.6403</td><td>Pre-warmup</td></tr>
<tr><td>SMD_machine-3-8</td><td>60</td><td>0.6946</td><td>Pre-warmup</td></tr>
<tr><td>SMD_machine-2-7</td><td>65</td><td>0.8788</td><td>Pre-warmup</td></tr>
<tr><td>SMD_machine-2-6</td><td>85</td><td>0.5941</td><td>Pre-warmup</td></tr>
<tr><td>SMD_machine-3-2</td><td>90</td><td>0.0938</td><td>Pre-warmup</td></tr>
<tr><td>SMD_machine-1-7</td><td>170</td><td>0.7463</td><td>Pre-warmup</td></tr>
<tr><td>SMD_machine-2-9</td><td>230</td><td>0.7211</td><td>Pre-warmup</td></tr>
<tr><td>Exathlon_app4</td><td>255</td><td>0.8274</td><td>Just-after</td></tr>
<tr><td>PSM</td><td>260</td><td>0.8034</td><td>Just-after</td></tr>
<tr><td>Exathlon_app2</td><td>260</td><td>0.9426</td><td>Just-after</td></tr>
<tr><td>SMD_machine-3-3</td><td>275</td><td>0.7791</td><td>Just-after</td></tr>
<tr><td>SMD_machine-3-9</td><td>275</td><td>1.0000</td><td>Just-after</td></tr>
<tr><td>SWaT_excl22</td><td>280</td><td>0.6305</td><td>Just-after</td></tr>
<tr><td>SMD_machine-3-1</td><td>285</td><td>0.9387</td><td>Just-after</td></tr>
<tr><td>SMD_machine-1-2</td><td>295</td><td>0.7007</td><td>Just-after</td></tr>
<tr><td>Exathlon_app5</td><td>295</td><td>0.8190</td><td>Just-after</td></tr>
<tr><td>SMD_machine-2-3</td><td>375</td><td>0.8894</td><td>Late phase 2</td></tr>
<tr><td>WaDi_A1</td><td>395</td><td>0.8495</td><td>Late phase 2</td></tr>
<tr><td>WaDi_A2</td><td>410</td><td>0.7939</td><td>Late phase 2</td></tr>
<tr><td>SMD_machine-2-4</td><td>430</td><td>0.6146</td><td>Late phase 2</td></tr>
<tr><td>Exathlon_app1</td><td>435</td><td>0.4668</td><td>Late phase 2</td></tr>
<tr><td>Exathlon_app9</td><td>440</td><td>0.5617</td><td>Late phase 2</td></tr>
</table>
## 1.3 각 구성요소의 anomaly detection 관점 분석
<table fit-page-width="true" header-row="true">
<tr>
<td>구성요소</td>
<td>학습 목표</td>
<td>이상탐지 GOOD 상태</td>
<td>Phase 3 (BAD) 시그널</td>
</tr>
<tr>
<td>**Teacher (3-layer)**</td>
<td>정상 reconstruction loss ↓</td>
<td>정상은 잘, 이상은 어느 정도까지만</td>
<td>`teacher_recon_anomaly`도 너무 ↓ (over-generalize)</td>
</tr>
<tr>
<td>**Student (2-layer)**</td>
<td>정상만 학습 (capacity 제한)</td>
<td>`student_recon_anomaly` 높게 유지 (anomaly fit 불가)</td>
<td>**`student_recon_anomaly` peak 후 ↓** = student도 anomaly 학습 시작</td>
</tr>
<tr>
<td>**Discrepancy = T − S**</td>
<td>정상 작게, 이상 크게</td>
<td>`disc_anomaly` ≫ `disc_normal`</td>
<td>**`disc_score_anomaly` peak 후 ↓** = 두 head 비슷해짐</td>
</tr>
<tr>
<td>**분리도 (a − n recon)**</td>
<td>학습 진행에 따라 ↑</td>
<td>큰 분리도 → 검출 쉬움</td>
<td>**분리도 peak 후 ↓** = 정상/이상 구분 흐려짐</td>
</tr>
<tr>
<td>**GRL classifier**</td>
<td>`balanced_acc → 0.5`</td>
<td>encoder confused state</td>
<td>balanced_acc 0.5에서 다시 벗어남</td>
</tr>
<tr>
<td>**FM (Feature Matching)**</td>
<td>`fm_loss` ↓</td>
<td>학습 진행 signal</td>
<td>adaptive λ over-saturate</td>
</tr>
</table>
<callout icon="🧠" color="purple_bg">
\t**핵심 통찰**: 모델 stop 최적 시점 = **Phase 2 → Phase 3 전환점** = "분리도가 peak이고 student가 anomaly에 아직 학습 안 시작"
\t**Type A signal (수렴 신호)**: train_loss / teacher_recon_normal plateau — "정상에 충분히 fit"
\t**Type B signal (퇴화 신호) ★**: anomaly-related metric의 peak reversal — "이상도 학습되기 시작, **여기서 멈춰야 함**"
</callout>"""


def part2():
    return """# 2. 사용자 피드백 반영 사항
<table fit-page-width="true" header-row="true">
<tr>
<td>피드백 #</td>
<td>요구사항</td>
<td>v4 반영</td>
</tr>
<tr>
<td>**(1)**</td>
<td>Label 기반 evaluation 지표 제외 (f1, pak_auc, prc_auc, roc_auc, pa_K_* 등)</td>
<td>모든 `em_*` inference metric 및 `deriv_pa_K_curve_*` 제거 — 93 label-free metric만 사용</td>
</tr>
<tr>
<td>**(2)**</td>
<td>Patience 줄이기 — 1, 2, 3만</td>
<td>그대로 적용. P=5/7/10/15/20/30/50 모두 제거</td>
</tr>
<tr>
<td>**(3)**</td>
<td>Student recon / discrepancy "올라가다 감소" peak detection</td>
<td>**peak_reversal ES rule** 신규 구현. 10개 composite metric 직접 고안</td>
</tr>
<tr>
<td>**(4)**</td>
<td>너무 마이너한 ES 방식 제외</td>
<td>GL_α, PQ, MACD, Bollinger, hypothesis test 모두 제외. standard / peak_reversal만</td>
</tr>
<tr>
<td>**(5)**</td>
<td>Warmup=250 반영 필요 (이전 phase 분석 오류 수정)</td>
<td>위 §1.1, §1.2 표에서 실제 oracle 분포 기반 phase 재정의</td>
</tr>
</table>
# 3. v4 신규 — 직접 고안한 Composite Metric (10개)
**설계 원칙**: 단일 metric의 한계 (noisy plateau) 를 극복하기 위해 anomaly-favoring signal을 곱/합으로 결합.
<table fit-page-width="true" header-row="true">
<tr>
<td>#</td>
<td>Composite metric</td>
<td>정의</td>
<td>의도</td>
<td>권장 ES</td>
</tr>
<tr>
<td>A</td>
<td>`composite_disc_x_separation_product`</td>
<td>`disc_score_anomaly × recon_score_separation`</td>
<td>두 검출 신호 동시 peak</td>
<td>peak_reversal max</td>
</tr>
<tr>
<td>B</td>
<td>`composite_student_anom_over_teacher_anom_ratio`</td>
<td>`student_recon_anomaly / teacher_recon_anomaly`</td>
<td>1로 수렴 = student가 teacher 따라잡음 = STOP</td>
<td>peak_reversal max</td>
</tr>
<tr>
<td>C</td>
<td>**`composite_learning_phase_indicator`** 🥇 (best composite)</td>
<td>`student_n − teacher_n`</td>
<td>격차 ↓ = student over-distillation</td>
<td>standard force_min</td>
</tr>
<tr>
<td>D</td>
<td>`composite_anomaly_separation_ensemble`</td>
<td>3개 separation의 z-score 가중합</td>
<td>다중 분리도 ensemble</td>
<td>peak_reversal max</td>
</tr>
<tr>
<td>E</td>
<td>`composite_type_a_x_type_b`</td>
<td>`disc_score_anomaly / train_rec_loss`</td>
<td>학습 수렴도 × 분리력</td>
<td>peak_reversal max</td>
</tr>
<tr>
<td>F</td>
<td>`composite_student_anom_velocity_negative`</td>
<td>`−d/dt student_recon_anomaly` (slope10)</td>
<td>음수 = student가 anomaly 학습 시작 = STOP</td>
<td>standard force_min</td>
</tr>
<tr>
<td>G</td>
<td>`composite_disc_anom_velocity`</td>
<td>`d/dt disc_score_anomaly` (slope10)</td>
<td>음수 = 검출력 손실 = STOP</td>
<td>standard force_max</td>
</tr>
<tr>
<td>H</td>
<td>`composite_unified_anomaly_health`</td>
<td>5개 anomaly-favoring signal의 z-score 가중합</td>
<td>전체 검출력 ensemble</td>
<td>peak_reversal max</td>
</tr>
<tr>
<td>I</td>
<td>`composite_student_capacity_safety_margin`</td>
<td>`min(student_sep, teacher_sep)`</td>
<td>약점 head 기준 분리도</td>
<td>peak_reversal max</td>
</tr>
<tr>
<td>J</td>
<td>`composite_anom_normal_loss_imbalance_x_sep`</td>
<td>`(anom_loss/norm_loss) × recon_separation`</td>
<td>anomaly hardness × 분리도</td>
<td>peak_reversal max</td>
</tr>
</table>
## 3.1 Composite metric 실제 성능 (best across all op/dir/rb/rule/P/T)
<table fit-page-width="true" header-row="true">
<tr>
<td>Rank</td>
<td>Composite metric</td>
<td>6-mean</td>
<td>Loss vs Oracle</td>
<td>Best (op, dir, rule, P, T)</td>
</tr>
<tr>
<td>1</td>
<td>`composite_learning_phase_indicator`</td>
<td>**0.6914**</td>
<td>6.58%</td>
<td>(raw, force_min, standard, P=3, rel=0.05)</td>
</tr>
<tr>
<td>2</td>
<td>`composite_disc_x_separation_product`</td>
<td>0.6878</td>
<td>7.07%</td>
<td>(ema03, auto, **peak_reversal**, P=1, rel=0.05)</td>
</tr>
<tr>
<td>3</td>
<td>`composite_anomaly_separation_ensemble`</td>
<td>0.6875</td>
<td>7.11%</td>
<td>(ema03, auto, standard, P=2, rel=0.01)</td>
</tr>
<tr>
<td>4</td>
<td>`composite_disc_anom_velocity`</td>
<td>0.6869</td>
<td>7.19%</td>
<td>(slope10, force_max, peak_reversal, P=3, abs=0.001)</td>
</tr>
<tr>
<td>5</td>
<td>`composite_student_capacity_safety_margin`</td>
<td>0.6831</td>
<td>7.70%</td>
<td>(ema03, auto, peak_reversal, P=1, rel=0.01)</td>
</tr>
<tr>
<td>6</td>
<td>`composite_unified_anomaly_health`</td>
<td>0.6815</td>
<td>7.92%</td>
<td>(sign_changes10, force_max, standard, P=2, abs=0)</td>
</tr>
<tr>
<td>7</td>
<td>`composite_type_a_x_type_b`</td>
<td>0.6777</td>
<td>8.43%</td>
<td>(slope10, force_max, standard, P=3, rel=0.05)</td>
</tr>
<tr>
<td>8</td>
<td>`composite_student_anom_over_teacher_anom_ratio`</td>
<td>0.6776</td>
<td>8.44%</td>
<td>(ema03, force_min, standard, P=3, rel=0.05)</td>
</tr>
<tr>
<td>9</td>
<td>`composite_anom_normal_loss_imbalance_x_sep`</td>
<td>0.6729</td>
<td>9.08%</td>
<td>(raw, force_min, peak_reversal, P=3, rel=0.05)</td>
</tr>
<tr>
<td>10</td>
<td>`composite_student_anom_velocity_negative`</td>
<td>0.6710</td>
<td>9.34%</td>
<td>(curvature10, force_min, peak_reversal, P=2, abs=0.001)</td>
</tr>
<tr>
<td>**참고 (기존 #1)**</td>
<td>**`th_train_rec_loss` (peak_reversal)**</td>
<td>**0.7131**</td>
<td>**3.65%**</td>
<td>(raw, auto, peak_reversal, P=3, rel=0.001)</td>
</tr>
</table>
<callout icon="⚠️" color="orange_bg">
\t**솔직한 평가**: **모든 composite metric이 기존 단일 plateau metric (0.7131)을 능가하지 못함**.
\t**원인 분석**:
\t- 학습 곡선이 plateau-dominated — warmup=250 이후 모든 anomaly-aware signal이 noisy plateau로 수렴해서 "peak after warmup"이 명확하지 않음.
\t- Composite은 phase 2 sweet spot 잡기엔 능함 (`disc_x_separation_product` P=1은 SWaT/PSM에서 oracle 정확히 잡음) — 그러나 WaDi A1/A2 (oracle 395/410)에서 너무 빨리 stop → 평균 손해.
\t- 단일 정상 plateau metric의 robustness: 40% pre-warmup + 24% late dataset을 동시에 일정 수준 처리.
\t**의의**: 학습 dynamics를 더 깊이 이해해야 composite이 의미 있게 작동. 향후 작업에 warmup 단축 + dataset-class-aware ES 시도 필요.
</callout>"""


def part3():
    return """# 4. Top 20 Cross-Dataset Configs (전체 통합)
<table fit-page-width="true" header-row="true">
<tr>
<td>#</td>
<td>Metric</td>
<td>Op</td>
<td>Dir</td>
<td>Rule</td>
<td>Rollback</td>
<td>P</td>
<td>T</td>
<td>Mean 6-group</td>
<td>Loss</td>
</tr>
<tr>
<td>**1** 🥇</td>
<td>`th_train_rec_loss`</td>
<td>raw</td>
<td>auto/force_min</td>
<td>**peak_reversal**</td>
<td>stop</td>
<td>3</td>
<td>rel=0.001</td>
<td>**0.7131**</td>
<td>3.65%</td>
</tr>
<tr>
<td>2</td>
<td>`th_train_feature_recon_mean__feat_mean`</td>
<td>raw</td>
<td>auto/force_min</td>
<td>peak_reversal</td>
<td>stop</td>
<td>3</td>
<td>rel=0.001</td>
<td>0.7131</td>
<td>3.65%</td>
</tr>
<tr>
<td>3</td>
<td>`th_train_teacher_recon_normal`</td>
<td>raw</td>
<td>auto/force_min</td>
<td>standard</td>
<td>stop</td>
<td>3</td>
<td>rel=0.01</td>
<td>0.7092</td>
<td>4.17%</td>
</tr>
<tr>
<td>4-10</td>
<td>(`train_rec_loss`, `feature_recon_mean__feat_mean` 변형들)</td>
<td>raw</td>
<td>—</td>
<td>both</td>
<td>stop</td>
<td>3</td>
<td>rel=0.001/abs=0</td>
<td>0.7087-0.7088</td>
<td>4.22-4.25%</td>
</tr>
<tr>
<td>11-20</td>
<td>(`feature_recon_mean__feat_std` 변형들)</td>
<td>raw</td>
<td>—</td>
<td>both</td>
<td>stop</td>
<td>3</td>
<td>various</td>
<td>0.7065-0.7081</td>
<td>4.32-4.54%</td>
</tr>
</table>
## 4.1 Per-Dataset Best (v4)
<table fit-page-width="true" header-row="true">
<tr>
<td>Dataset</td>
<td>Metric</td>
<td>Op</td>
<td>Dir</td>
<td>Rule</td>
<td>P, T</td>
<td>Stop ep</td>
<td>ES value</td>
<td>Loss</td>
</tr>
<tr>
<td>**SWaT_excl22**</td>
<td>`th_train_loss`</td>
<td>slope10</td>
<td>auto</td>
<td>**peak_reversal**</td>
<td>P=2, abs=0.001</td>
<td>280</td>
<td>0.6305</td>
<td>**0.00%** 🎯</td>
</tr>
<tr>
<td>**WaDi_A1**</td>
<td>`th_train_student_recon_normal`</td>
<td>raw</td>
<td>auto</td>
<td>standard</td>
<td>P=1, rel=0.01</td>
<td>395</td>
<td>0.8495</td>
<td>**0.00%** 🎯</td>
</tr>
<tr>
<td>**WaDi_A2**</td>
<td>`th_train_rec_loss`</td>
<td>raw</td>
<td>auto</td>
<td>standard</td>
<td>P=2, abs=0</td>
<td>410</td>
<td>0.7939</td>
<td>**0.00%** 🎯</td>
</tr>
<tr>
<td>**PSM**</td>
<td>`th_train_loss`</td>
<td>raw</td>
<td>auto</td>
<td>standard</td>
<td>P=2, abs=0</td>
<td>260</td>
<td>0.8034</td>
<td>**0.00%** 🎯</td>
</tr>
<tr>
<td>**SMD_avg**</td>
<td>`th_epoch_recon_score_anomaly`</td>
<td>slope10</td>
<td>auto</td>
<td>standard</td>
<td>P=2, abs=0</td>
<td>266</td>
<td>0.6796</td>
<td>5.36%</td>
</tr>
<tr>
<td>**Exathlon_avg**</td>
<td>`th_epoch_raw_disc_normal`</td>
<td>ema03</td>
<td>auto</td>
<td>**peak_reversal**</td>
<td>P=2, abs=0.001</td>
<td>328</td>
<td>0.6333</td>
<td>1.85%</td>
</tr>
</table>
# 5. Leaderboard — MAE 271 vs 15 Baseline (v4)
<callout icon="ℹ️" color="gray_bg">
\t**참고**: 7개 신규 SOTA (`tfmae`, `npsr`, `timesnet`, `dcdetector`, `memto`, `moderntcn`, `catch`) 는 Q3 batch 진행 중 → 본 leaderboard에는 15 active baseline만 포함.
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
<td>Rank Avg</td>
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
<td>0.6796</td>
<td>0.6333</td>
<td>**2.00**</td>
<td>0.7317</td>
</tr>
<tr>
<td>**3** 🥉</td>
<td>**ES #1 (`train_rec_loss`, peak_reversal, P=3, rel=0.001)**</td>
<td>0.6305</td>
<td>0.8252</td>
<td>0.7753</td>
<td>0.7947</td>
<td>0.6472</td>
<td>0.6053</td>
<td>**3.50**</td>
<td>0.7131</td>
</tr>
<tr>
<td>4-7</td>
<td>(동일 score 변형들 `rec_loss`/`feature_recon_mean`)</td>
<td>0.6305</td>
<td>0.8252</td>
<td>0.7753</td>
<td>0.7947</td>
<td>0.6472</td>
<td>0.6053</td>
<td>3.50-7.00</td>
<td>0.7131</td>
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
<td>10-22</td>
<td>(나머지 baseline: omnianomaly, mlpmixer, mlp, transformer, nn_distance, gdn, usad, pca_error, dagmm, gcn_lstm, sensor_range, l2_norm, random)</td>
<td colspan="6">상세는 §5.1</td>
<td>11.83-20.00</td>
<td>0.3106-0.5794</td>
</tr>
</table>
<callout icon="🏆" color="purple_bg">
\t**Top 7 = 모두 MAE 271 변형**. 8위 `tranad`까지 rank avg gap = 5.67. mean PA F1 gap = +14.9%p (0.7131 vs 0.6205).
\t**ES adoption cost (Oracle → best ES)**: rank avg 1.00 → 3.50, mean 0.7401 → 0.7131, **3.65% 손실**.
</callout>"""


def part4():
    return """# 6. 권장 정책 (실전 ES 가이드)
<table fit-page-width="true" header-row="true">
<tr>
<td>시나리오</td>
<td>추천 metric + op</td>
<td>ES rule</td>
<td>P, T</td>
<td>예상 성능</td>
</tr>
<tr>
<td>**전 범용 (default)**</td>
<td>`th_train_rec_loss` (raw)</td>
<td>**peak_reversal**</td>
<td>P=3, rel=0.001</td>
<td>**rank 3.50, mean 0.7131 (loss 3.65%)**</td>
</tr>
<tr>
<td>Per-feature 변형 (동일 성능)</td>
<td>`th_train_feature_recon_mean__feat_mean` (raw)</td>
<td>peak_reversal</td>
<td>P=3, rel=0.001</td>
<td>mean 0.7131</td>
</tr>
<tr>
<td>Composite (직접 고안, 1위)</td>
<td>`composite_learning_phase_indicator` (raw)</td>
<td>standard force_min</td>
<td>P=3, rel=0.05</td>
<td>mean 0.6914 — 일반 metric보다 약함</td>
</tr>
<tr>
<td>**짧은 데이터셋용 (Phase 2 sweet spot이 250-300)**</td>
<td>`composite_disc_x_separation_product` (ema03)</td>
<td>peak_reversal</td>
<td>P=1, rel=0.05</td>
<td>SWaT/PSM oracle 정확히 잡음, but WaDi 손해</td>
</tr>
</table>
# 7. 한계 및 향후 작업
## 7.1 한계
<table fit-page-width="true" header-row="true">
<tr>
<td>Limitation</td>
<td>Impact</td>
<td>대응</td>
</tr>
<tr>
<td>**Warmup 250 ep 고정 + 25 dataset 중 40%가 pre-warmup oracle**</td>
<td>이 10개 dataset에서는 절대 oracle 재현 불가, 평균 5-32% 손실 강제 발생</td>
<td>**Warmup ablation (50/100/150/200/250) 가장 시급**</td>
</tr>
<tr>
<td>**Composite metric이 plateau metric 못 넘음**</td>
<td>직접 고안한 10개 모두 0.6710-0.6914 (vs 단일 0.7131)</td>
<td>학습 dynamics 단계 식별 모델 / warmup 단축으로 phase 2 식별 가능성 ↑</td>
</tr>
<tr>
<td>**Eval interval 5**</td>
<td>5 ep 단위로만 stop trigger 가능</td>
<td>1 ep eval 비용 크므로 합리적 trade-off</td>
</tr>
<tr>
<td>**7개 신규 SOTA baseline Q3 미완료**</td>
<td>절대적 rank 추후 변동 가능</td>
<td>tfmae/timesnet/dcdetector/memto/moderntcn/catch/npsr Q3 완료 후 재계산</td>
</tr>
<tr>
<td>**`mean_discrepancy` 본 분석엔 미반영**</td>
<td>코드 수정 이후 학습 사이클이 본 sweep에 포함되지 않음</td>
<td>다음 학습 cycle부터 자동 활용</td>
</tr>
</table>
## 7.2 향후 작업 (우선순위순)
<table fit-page-width="true" header-row="true">
<tr>
<td>우선순위</td>
<td>작업</td>
<td>이유</td>
</tr>
<tr>
<td>**1**</td>
<td>**Warmup ablation (50/100/150/200/250)**</td>
<td>40%의 pre-warmup oracle dataset이 가장 큰 손실 요인</td>
</tr>
<tr>
<td>2</td>
<td>**Dataset-class-aware ES** (oracle epoch에 따라 다른 config 적용)</td>
<td>단일 cross-dataset config은 본질적으로 25 dataset을 동시에 못 잡음</td>
</tr>
<tr>
<td>3</td>
<td>**Ensemble ES (multi-metric vote)**</td>
<td>여러 metric의 trigger를 vote → 단일 noisy signal 보완</td>
</tr>
<tr>
<td>4</td>
<td>`mean_discrepancy` 활용한 재학습 + 분석</td>
<td>1-line code change 완료, 다음 cycle부터 활용</td>
</tr>
<tr>
<td>5</td>
<td>Validation split (held-out normal) 도입</td>
<td>표준 ES literature와 align, 학습시간 5-10% 증가</td>
</tr>
</table>
# 8. Artifacts & Reproduce
<table fit-page-width="true" header-row="true">
<tr>
<td>파일</td>
<td>크기</td>
<td>설명</td>
</tr>
<tr>
<td>`temp/early_stopping/sweep_raw_v4.json`</td>
<td>394 MB</td>
<td>v4 sweep 전체 (2.1M rows, 93 metric × 5 op × 3 dir × 2 rb × 2 rule × 3 P × 5 T)</td>
</tr>
<tr>
<td>`temp/early_stopping/sweep_raw_v3.json`</td>
<td>350 MB</td>
<td>v3 sweep (composite 없는 비교 기준)</td>
</tr>
<tr>
<td>`temp/early_stopping/baseline_aggregated.json`</td>
<td>4 KB</td>
<td>15 baseline의 6-group pak_auc_f1</td>
</tr>
<tr>
<td>`temp/early_stopping/rank_comparison_v3.json`</td>
<td>~25 KB</td>
<td>v3 leaderboard (v4와 ranking 동일)</td>
</tr>
</table>
<callout icon="🛠️" color="gray_bg">
\t**Scripts (재실행 가능, 모두 multiprocess + memory monitoring 적용)**:
\t- `scripts/early_stopping_analysis_v4.py` — v4 sweep (composite metrics 10개 + label-free 83 = 93 metric, 10초 실행)
\t- `scripts/early_stopping_analyze_v4.py` — composite 성능 분석
\t- `scripts/early_stopping_analysis_v3.py` — v3 sweep (composite 없는 비교 기준)
\t- `scripts/early_stopping_baseline_aggregate.py` — baseline 집계
\t**시스템**: 6 worker multiprocess, peak 시스템 메모리 21%, JSON 394 MB compact format.
\t**코드 변경 완료**: `mae_anomaly/trainer.py` L213/L935에 `train_mean_discrepancy` history append 추가 — 다음 학습부터 자동 저장.
</callout>"""


def main():
    parts = [part1(), part2(), part3(), part4()]
    out = "\n".join(parts)
    p = Path("/home/ykio/notebooks/claude/temp/early_stopping/notion_v4.txt")
    p.write_text(out)
    print(f"Wrote {p} ({len(out)} chars)")


if __name__ == "__main__":
    main()
