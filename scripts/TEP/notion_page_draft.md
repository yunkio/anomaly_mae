# TEP Type-Generalization #12 — Simple Baseline 사전 실험

:::callout {icon="🎯" color="blue_bg"}
**한 줄 요약** — MAE(exp271)의 핵심 주장 *"라벨은 anomaly 유형을 외우는 데 쓰이지 않고, 정상 모델을 정화(purify)하는 데만 쓰인다"* 를 검증하기 위해 설계한 TEP fault-type-disjoint 실험의 **simple baseline 사전 실행**입니다. GPU를 쓰지 않는 단순 계산 방법론 5종(random, sensor_range, pca_error, l2_norm, nn_distance)을 label-blind 기준선으로 돌려, (1) 파이프라인 전체를 검증하고 (2) MAE 결과를 해석할 좌표계(floor·ceiling·보정 기준선)를 확보했습니다. **핵심 발견: seen/unseen 성능 격차는 fault의 '난이도'가 아니라 train 오염(contamination)이 만들며, 라벨을 '인스턴스 제거'로만 쓰는 접근은 부분 라벨 환경에서 거의 무력합니다 — 정확히 MAE의 GRL purging이 차별화될 수 있는 지점입니다.**
:::

- **실험 번호**: #12 · **결과**: `temp/0610/TEP/results/12_20260610_211815_tep_typegen_simple/`
- **사전 등록 설계**: `temp/tep_design/80_experiment_design_final.md` (2026-06-10 동결) · **스크립트**: `temp/0610/TEP/` (기존 코드 무수정, import만)
- **평가**: 기존 baseline 비교 실험과 동일한 단일 코드 경로 (`compute_full_metric_set` — PA%K AUC, VUS, Affiliation 등 전체 metric set)

---

## 1. 진행한 실험

### 1.1 왜 TEP인가

현재 모든 benchmark(SWaT·WaDi·PSM·SMD 등)의 split은 *train 라벨에 등장한 anomaly 유형이 test에도 그대로 재등장*하는 구조입니다. 따라서 "학습 때 본 적 없는 유형의 이상도 잡는가"라는 질문 — MAE 메커니즘 주장의 핵심 — 은 한 번도 측정된 적이 없습니다. TEP(Tennessee Eastman Process)는 **도메인이 공인한 fault 분류 체계**(Downs & Vogel: Step / Random variation / Slow drift / Sticking / Unknown, IDV 1–20)를 가진 유일한 표준 벤치마크라, fault family 단위로 train과 test의 이상 유형을 완전히 분리(type-disjoint)할 수 있습니다.

### 1.2 데이터 구성 (사전 등록, run 번호까지 결정론적 고정)

<table fit-page-width="true" header-row="true">
	<tr>
		<td>구성</td>
		<td>내용</td>
		<td>규모</td>
	</tr>
	<tr>
		<td>Train (fold별 contaminated)</td>
		<td>FaultFree runs 1–240 + seen-family faulty **60 runs** (라벨 양 등화: F-STEP 6종×10 / F-RAND 4종×15 / F-DS 2종×30 / F-UNK 5종×12)</td>
		<td>288,000 samples · anomaly 16.67%</td>
	</tr>
	<tr>
		<td>Train (ffonly, clean 참조)</td>
		<td>FaultFree runs 1–240만 — 설계의 B0(clean-normal reference) 대응</td>
		<td>230,400 samples</td>
	</tr>
	<tr>
		<td>Test (4 fold 공유 고정)</td>
		<td>**fault 20종 전부** × runs 441–460 + FaultFree runs 461–500. fold가 바뀌면 seen/unseen **분류 라벨만** 바뀜 → fold 간 직접 비교 가능</td>
		<td>440 runs = 422,400 samples</td>
	</tr>
	<tr>
		<td>Label</td>
		<td>각 faulty run에서 sample 161부터 anomaly (run당 정상 160 + 이상 800; 기존 loader의 onset=160 off-by-one을 실측 기반 161로 정정)</td>
		<td>region 400개 (test)</td>
	</tr>
	<tr>
		<td>Run boundary</td>
		<td>모든 연산이 run 경계를 가로지르지 않음 — pca_error의 5-box smoothing을 run별로 재시작(나머지 4종은 pointwise라 무관), 검증 게이트에서 440개 run 전부 확인</td>
		<td>439 internal seams</td>
	</tr>
</table>

**4-fold 회전** — 각 fault family를 차례로 유일한 seen family로 두는 hard 설정:

<table fit-page-width="true" header-row="true">
	<tr>
		<td>Fold</td>
		<td>Seen (train에 오염으로 주입)</td>
		<td>Unseen (test에서만 등장)</td>
	</tr>
	<tr>
		<td>F-STEP</td>
		<td>Step: IDV 1,2,4,5,6,7</td>
		<td>나머지 11종</td>
	</tr>
	<tr>
		<td>F-RAND</td>
		<td>Random variation: IDV 8,10,11,12</td>
		<td>나머지 13종</td>
	</tr>
	<tr>
		<td>F-DS</td>
		<td>Drift+Sticking: IDV 13,14</td>
		<td>나머지 15종</td>
	</tr>
	<tr>
		<td>F-UNK</td>
		<td>Unknown: IDV 16–20</td>
		<td>나머지 12종</td>
	</tr>
</table>

IDV 3/9/15는 폐루프 제어가 외란을 보상하는 난제 fault로, 사전 등록된 정량 규칙(post-onset 평균 max|z| < 2×정상 변동)에 의해 headline 집계에서 제외하되 test에는 포함하여 **excluded-hard partition**으로 별도 보고합니다.

### 1.3 실험 매트릭스

<table fit-page-width="true" header-row="true">
	<tr>
		<td>블록</td>
		<td>조건</td>
		<td>목적</td>
	</tr>
	<tr>
		<td>① 메인 (25 runs)</td>
		<td>simple 5종 × {contaminated 4 folds + ffonly} — random은 규약대로 5 independent draws의 mean±std</td>
		<td>label-blind 모델의 seen/unseen 격차(G)와 오염 피해(C_dmg) 측정</td>
	</tr>
	<tr>
		<td>② Noisy-label sweep (20 runs)</td>
		<td>pca_error·sensor_range × {F-STEP, F-DS} × labeled ∈ {0, 20, 50, 80, 100}% — "라벨된 n%는 이상적으로 제거(oracle cleaning), (100−n)%는 unlabeled 오염으로 잔류"</td>
		<td>부분 라벨 환경에서 잔류 오염에 대한 내성 곡선 (MAE label sweep의 floor)</td>
	</tr>
	<tr>
		<td>③ IDV 3/9/15 심층 검증</td>
		<td>L1 point 수준(per-feature AUC) / L2 run-집계 수준(800-sample mean·std) / L3 모델 score 수준의 3단 분리도 검사</td>
		<td>"구분 불가" 주장의 수준별 검증 — window 모델의 구조적 기회 탐색</td>
	</tr>
</table>

:::callout {icon="✅" color="green_bg"}
**검증 게이트 전 항목 PASS** — stream 크기·label 산술(train 오염 16.67% 설계값 정확 일치)·partition 분리(seen∩unseen=∅, 3/9/15 제외)·run 경계(per-run smoothing 증거: 440개 run 선두 5pt=0)·score 무결성(길이/유한성/이진성)·sweep 끝점 일관성(labeled 100% ≡ ffonly, 수치 동일 재현). 상세: `analysis_report.md` §1.
:::

---

## 2. 실험 의도와 목적

이 실험은 그 자체가 목적이 아니라, **MAE 본 실험(조건 A: ours / B: label-blind control / B0: clean reference / C: supervised skyline / D: weak baselines)의 해석 좌표계를 미리 구축**하는 사전 작업입니다. 세 가지 질문에 답하도록 설계했습니다.

**Q1. seen/unseen 성능 격차는 무엇으로 만들어지는가?** — MAE가 unseen 유형에서 성능이 떨어졌을 때, 그것이 "유형 일반화 실패"인지 "그 fault들이 원래 어려운 것"인지 "train 오염의 부작용"인지 구분하려면, **라벨을 전혀 쓰지 않는 모델의 격차(G)를 먼저 알아야** 합니다. Simple 5종은 모두 label-blind이므로, 이들의 G는 난이도+오염 효과의 순수한 합입니다.

**Q2. 부분 라벨(noisy-label) 환경에서 "라벨 = 오염 제거"의 한계는 어디인가?** — 실무에서는 이상의 일부만 라벨링됩니다. 라벨된 인스턴스를 제거하는 것이 label-consuming 방법이 할 수 있는 가장 단순한 행동(oracle cleaning)인데, 이 전략의 성능 곡선이 MAE GRL purging이 넘어야 할 기준선이 됩니다. **MAE의 존재 이유와 직결**: GRL은 라벨된 인스턴스에서 fault signature를 학습해 *unlabeled 같은-유형 인스턴스까지* 정화할 수 있어야 합니다(within-type generalization).

**Q3. "구분 불가" fault는 정말 모든 수준에서 구분 불가인가?** — IDV 3/9/15의 제외 규칙은 point 수준 통계(|z|)로 정의했습니다. 시간 맥락(W=500)을 쓰는 MAE에게도 불가능한지는 별도 질문이며, 만약 시간-집계 수준에서 분리 가능하다면 그것은 **point-wise 방법 전체가 실패하는 지점에서 window 모델이 이기는 구조적 기회**가 됩니다.

---

## 3. 결과 분석

### 3.1 메인 결과 — seen/unseen 격차 G (pak_auc_f1)

G = seen − unseen. **음수 = seen이 오히려 나쁨**. 모든 모델이 label-blind이므로 라벨 효과는 0이고, G는 난이도와 오염 효과의 합만 반영합니다.

<table fit-page-width="true" header-row="true">
	<tr>
		<td>모델</td>
		<td>F-STEP</td>
		<td>F-RAND</td>
		<td>F-DS</td>
		<td>F-UNK</td>
		<td>ffonly (clean) 4-fold G 범위</td>
	</tr>
	<tr>
		<td>pca_error</td>
		<td color="orange">−0.051</td>
		<td color="orange">−0.025</td>
		<td color="red">−0.204</td>
		<td color="red">−0.116</td>
		<td color="green">−0.006 ~ +0.004 (≈0)</td>
	</tr>
	<tr>
		<td>l2_norm</td>
		<td color="orange">−0.022</td>
		<td color="orange">−0.068</td>
		<td color="red">−0.121</td>
		<td color="orange">−0.073</td>
		<td>−0.098 ~ +0.079</td>
	</tr>
	<tr>
		<td>nn_distance</td>
		<td color="orange">−0.051</td>
		<td color="orange">−0.035</td>
		<td color="green">**+0.030**</td>
		<td color="orange">−0.035</td>
		<td>−0.040 ~ +0.040</td>
	</tr>
	<tr>
		<td>sensor_range</td>
		<td color="red">−0.432</td>
		<td color="red">−0.445</td>
		<td color="red">−0.573</td>
		<td color="red">−0.261</td>
		<td>+0.019 ~ +0.130</td>
	</tr>
	<tr>
		<td>random (metric artifact)</td>
		<td>−0.033</td>
		<td>−0.070</td>
		<td>−0.161</td>
		<td>−0.049</td>
		<td>(동일 — train 무관)</td>
	</tr>
</table>

:::callout {icon="🔬" color="blue_bg"}
**판독 1 — G의 부호가 전부 음수(nn_distance/F-DS 단 1건 제외)**: "처음 보는 유형이 더 어렵다"는 직관과 정반대입니다. seen 유형은 train에 **unlabeled 오염으로 들어가 있기 때문에**, label-blind 검출기는 그 패턴을 정상으로 학습해 버리고 정확히 seen 쪽 탐지가 무너집니다. clean 학습(ffonly)에서는 pca_error의 G가 사실상 0(±0.006)이라는 것이 결정적 증거 — **TEP usable 17종의 '순수 난이도' 차이는 거의 없고, G는 오염이 만듭니다.**
:::

:::callout {icon="📐" color="purple_bg"}
**판독 2 — random 행은 detection이 아니라 metric artifact의 자**: random은 train을 보지 않으므로 G가 0이어야 할 것 같지만 −0.03~−0.16이 나옵니다. partition마다 region 수·길이·positive rate가 달라 PA%K의 관대함이 다르게 작동하기 때문입니다(F-DS seen은 2 faults×20 runs=40 regions뿐). **raw G를 partition 간에 직접 비교하면 안 되고, 반드시 같은 fold 안에서 matched control과의 차이(Ĝ = G − G_ctrl)로 읽어야 한다**는 본 설계 통계 원칙의 실증입니다.
:::

### 3.2 오염 피해 정량화 — C_dmg = ffonly − contaminated (seen partition, pak_auc_f1)

<table fit-page-width="true" header-row="true">
	<tr>
		<td>모델</td>
		<td>F-STEP</td>
		<td>F-RAND</td>
		<td>F-DS</td>
		<td>F-UNK</td>
		<td>해석</td>
	</tr>
	<tr>
		<td>pca_error</td>
		<td color="orange">0.126</td>
		<td color="orange">0.080</td>
		<td color="red">0.252</td>
		<td color="orange">0.132</td>
		<td>fault 방향을 PCA 부분공간이 **전역적으로 흡수** — seen 탐지력의 8~25%p가 오염에 잠식</td>
	</tr>
	<tr>
		<td>sensor_range</td>
		<td color="red">0.762</td>
		<td color="red">0.889</td>
		<td color="red">0.989</td>
		<td color="red">0.592</td>
		<td>faulty run이 train min/max 범위를 넓혀 **탐지 메커니즘 자체가 붕괴** (0.998→0.009)</td>
	</tr>
	<tr>
		<td>nn_distance</td>
		<td>0.081</td>
		<td>0.069</td>
		<td color="green">**0.037**</td>
		<td>0.041</td>
		<td>오염 run과 같은 상태공간 영역을 재방문하는 fault만 가려짐 — **국소적 흡수**라 피해 최소</td>
	</tr>
</table>

:::callout {icon="🧭" color="blue_bg"}
**판독 3 — 오염 피해는 detector 기하에 따라 전역적/국소적으로 갈립니다.** 부분공간 방법(PCA)은 오염의 *방향*을 정상 모델에 흡수해 같은 family 전체 탐지가 약화되고(심지어 같은 물리 변수의 unseen fault까지 — F-STEP에서 unseen IDV11이 0.996→0.722로 동반 하락하는 near-variable spillover 확인), 인스턴스 방법(1-NN)은 오염 run이 실제로 점유한 좌표 근방만 가려집니다. drift처럼 run마다 궤적이 다른 fault(F-DS)에서는 1-NN이 거의 무손상(+G 유일 사례)인 이유입니다. **MAE 같은 표현학습 모델은 전자(전역 흡수)에 가깝다고 예상되므로, GRL purging이 회복해야 할 피해의 형태도 전역적일 것입니다.**
:::

### 3.3 Noisy-label sweep — "라벨 = 인스턴스 제거"의 한계 곡선

라벨된 n%의 오염 run을 학습에서 제거(oracle cleaning)하고 (100−n)%를 unlabeled로 남긴 조건. **seen partition의 pak_auc_f1**:

<table fit-page-width="true" header-row="true">
	<tr>
		<td>labeled %  (잔류 오염 runs)</td>
		<td>pca_error @ F-STEP</td>
		<td>pca_error @ F-DS</td>
		<td>sensor_range @ F-STEP</td>
		<td>sensor_range @ F-DS</td>
	</tr>
	<tr>
		<td>0%  (60)</td>
		<td>0.874</td>
		<td>0.747</td>
		<td color="red">0.184</td>
		<td color="red">0.009</td>
	</tr>
	<tr>
		<td>20%  (48)</td>
		<td>0.879</td>
		<td>0.748</td>
		<td color="red">0.200</td>
		<td color="red">0.009</td>
	</tr>
	<tr>
		<td>50%  (30)</td>
		<td>0.943</td>
		<td>0.757</td>
		<td color="red">0.212</td>
		<td color="red">0.010</td>
	</tr>
	<tr>
		<td>80%  (12)</td>
		<td>0.975</td>
		<td color="orange">0.836</td>
		<td color="red">0.304</td>
		<td color="red">0.021</td>
	</tr>
	<tr>
		<td>100%  (0, clean)</td>
		<td color="green">**0.9997**</td>
		<td color="green">**0.999**</td>
		<td color="green">**0.946**</td>
		<td color="green">**0.998**</td>
	</tr>
</table>

:::callout {icon="⚠️" color="orange_bg"}
**판독 4 — 곡선이 심하게 볼록(convex)합니다.** 오염의 80%를 이상적으로 제거해도 F-DS의 PCA는 0.836(clean 0.999 대비 −16%p), sensor_range는 100% 직전까지 사실상 0입니다. **잔류 오염 소수가 피해의 대부분을 만듭니다** — PCA는 12개의 fault run만으로도 그 방향을 부분공간에 담아버리고, sensor_range는 단 하나의 unlabeled fault run이 범위를 넓혀 무력화됩니다. 결론: *부분 라벨 환경에서 "라벨된 것을 제거"하는 전략은 거의 무가치하며, unlabeled 잔류 오염에 대한 내성이 진짜 병목입니다.*
:::

### 3.4 IDV 3/9/15 — "구분 불가" fault의 3단 검증

<table fit-page-width="true" header-row="true">
	<tr>
		<td>Fault</td>
		<td>L1: point 수준 best 단일-feature AUC</td>
		<td>L2: run-집계(800-sample) best AUC</td>
		<td>L3: 모델 score (pca@ffonly roc)</td>
		<td>판정</td>
	</tr>
	<tr>
		<td>IDV3 (D feed 온도 step)</td>
		<td color="red">0.568</td>
		<td color="green">**1.000** (xmeas_21 run-mean)</td>
		<td color="red">0.510</td>
		<td>**point 비식별 / 시간-집계 분리 가능**</td>
	</tr>
	<tr>
		<td>IDV9 (D feed 온도 random)</td>
		<td color="red">0.514</td>
		<td color="red">0.740 (≈ 우연 수준*)</td>
		<td color="red">0.513</td>
		<td>**완전 비식별**</td>
	</tr>
	<tr>
		<td>IDV15 (응축기 밸브 sticking)</td>
		<td color="red">0.515</td>
		<td color="green">**0.967** (xmeas_22 run-std)</td>
		<td color="red">0.512</td>
		<td>**point 비식별 / 시간-집계 분리 가능**</td>
	</tr>
	<tr>
		<td>참조: IDV1 (쉬운 step)</td>
		<td color="green">0.996</td>
		<td color="green">1.000</td>
		<td color="green">1.000</td>
		<td>전 수준 식별</td>
	</tr>
	<tr>
		<td>참조: IDV16/19 (subtle)</td>
		<td color="red">0.514 / 0.511</td>
		<td color="green">1.000 / 1.000</td>
		<td color="green">0.966 / 0.992</td>
		<td>단일 feature point로는 비식별, **다변량 상관구조**(PCA residual)로 식별</td>
	</tr>
</table>

\* L2는 52 features × 2 집계 = 104회 비교의 최대값이라 selection bias가 있으며, n=20 vs 40 runs에서 우연 최대치가 ~0.75 수준 — IDV9의 0.740은 잡음과 구분 불가, IDV3/15의 1.000/0.967은 실제 신호입니다.

:::callout {icon="💡" color="yellow_bg"}
**판독 5 — fault 난이도는 4계층으로 정밀화됩니다.** ① 단일 feature point로 식별(IDV1형) → ② 다변량 상관구조 point로 식별(IDV16/19형: 단일 feature AUC 0.51인데 PCA가 0.97+) → ③ **시간-집계로만 식별(IDV3/15형: 모든 point-wise simple 모델이 roc≈0.51로 전멸, 그러나 수백 sample의 mean/std에는 미세한 지속 편이가 남음)** → ④ 완전 비식별(IDV9형). 기존 설계의 "IDV3/9/15는 어떤 방법으로도 비식별" 문구는 **"point-wise 비식별"로 한정**해야 정확합니다.
:::

### 3.5 평가 지표 calibration — random floor

random baseline(이진 {0,1} score, 5 draws)의 full-test **pak_auc_f1 = 0.764** (prc_auc 0.758 ≈ positive rate 75.8%). 이 test stream은 positive rate가 높고 region이 길어(800 samples) PA%K가 매우 관대하게 작동합니다. **pak_auc_f1 절대값 0.76 이하는 random 이하**이며, excluded-hard partition에서 pca의 0.79도 prc_auc(0.51 ≈ 50% positive rate)와 함께 읽어야 "사실상 무신호"임이 드러납니다. → MAE 결과표에는 random·composition 보정 행을 반드시 포함해야 합니다.

---

## 4. 해석

이번 결과를 한 문장으로 압축하면: **"label-blind 세계에서 type-generalization 격차란 존재하지 않고, 존재하는 것은 오염 격차다."**

1. **난이도 가설의 기각** — clean 학습(ffonly)에서 pca_error는 17개 usable fault 전부를 사실상 만점(0.99+)으로 잡습니다. 즉 TEP의 usable fault들은 깨끗한 정상 모델만 있으면 trivial하게 검출되며, fold 간 seen/unseen 구성 차이가 만드는 '순수 난이도' 격차는 거의 0입니다.
2. **오염 가설의 채택** — 같은 모델이 contaminated train에서는 seen partition만 선택적으로 무너집니다(G 전 fold 음수, C_dmg 최대 25%p). 오염이 들어간 유형 = 정확히 seen 유형이기 때문입니다. 이것은 MAE 본 실험에서 조건 B(label-blind MAE)가 보일 기준 행동의 예고편입니다.
3. **오염 흡수의 기하학** — 같은 오염이라도 detector의 표현 방식에 따라 피해가 전역적(PCA: 방향 흡수 → family 전체+near-variable spillover)이거나 국소적(1-NN: 점유 좌표 근방만)입니다. 표현학습 모델(MAE)은 전역형에 가까울 것이므로, **GRL purging의 가치는 "전역적으로 흡수될 뻔한 fault 방향을 라벨로 도려내는 것"**으로 정식화할 수 있습니다.
4. **부분 라벨의 본질적 어려움** — oracle cleaning조차 80% 라벨에서 피해의 절반도 회복하지 못합니다(F-DS 0.836 vs 0.999). 라벨의 가치를 "그 인스턴스를 제거하는 것"으로 한정하면 부분 라벨 환경에서는 패배가 구조적으로 확정됩니다. 라벨이 의미를 가지려면 **라벨된 소수에서 학습한 fault signature가 unlabeled 다수에 일반화**되어야 합니다.
5. **window 모델의 고유 영토** — IDV3/15는 point-wise 방법 전체가 구조적으로 닿을 수 없지만 시간-집계 통계에는 신호가 남아 있습니다. W=500 모델이 여기서 신호를 잡는다면 단순한 성능 우위가 아니라 **방법론 계층 자체가 다르다는 증거**가 됩니다.

---

## 5. 인사이트

:::callout {icon="1️⃣" color="blue_bg"}
**Raw seen/unseen 격차는 type-generalization의 증거가 아니다.** label-blind 모델조차 G ≠ 0이며(오염 + metric artifact), 그 방향은 오히려 seen이 나쁜 쪽입니다. MAE 결과는 반드시 같은 fold의 matched control(조건 B)과의 차이(Ĝ)로만 해석해야 하며, 이는 사전 등록 설계의 통계 원칙(§4.4)을 데이터로 실증한 것입니다.
:::

:::callout {icon="2️⃣" color="purple_bg"}
**라벨의 가치는 '제거'가 아니라 '일반화 정화'에 있다.** oracle-removal 곡선의 극단적 볼록성은, GRL이 라벨된 인스턴스의 *signature*를 학습해 unlabeled 동일-유형 인스턴스까지 정화할 때만 부분 라벨 환경에서 우위가 생긴다는 것을 보여줍니다. 이것이 MAE noisy-label 실험(P0-4)의 정확한 평가 축입니다.
:::

:::callout {icon="3️⃣" color="orange_bg"}
**PA%K 절대값의 함정 — 이 test stream의 random floor는 0.764다.** positive rate 75.8% + 긴 region에서 PA%K는 관대해집니다. 모든 결과표에 random 행과 prc_auc 병기가 없으면 무의미한 수치 비교가 됩니다(excluded-hard에서 pca 0.79가 '무신호'인 사례).
:::

:::callout {icon="4️⃣" color="green_bg"}
**IDV3/15는 point-wise 방법론 전체의 공백 지대다.** 폐루프 제어가 point 분포를 완전히 가리지만 시간-집계(run-mean/run-std)에는 신호가 남습니다(AUC 1.000/0.967). 이 두 fault는 MAE가 "window 맥락이 본질적으로 필요한 이상"을 잡을 수 있는지 확인하는 천연 진단 케이스입니다. IDV9만이 진정한 불가능 케이스(negative control)입니다.
:::

:::callout {icon="5️⃣" color="yellow_bg"}
**TEP의 분별력은 clean 체제가 아니라 contaminated 체제에 있다.** clean train에서는 PCA도 만점이므로 "MAE가 잘 잡는다"는 자랑이 무의미합니다. 의미 있는 비교는 (i) 오염 하에서 누가 덜 무너지나, (ii) 라벨로 얼마나 회복하나, (iii) 부분 라벨에서 회복이 얼마나 유지되나 — 세 축이며, supervised skyline의 식별력 게이트(Gate 0) 확인이 더욱 중요해졌습니다.
:::

:::callout {icon="6️⃣" color="blue_bg"}
**Subtle-fault set이 데이터 통계만으로 동결되었다** — post-onset 평균 max|z| 하위 5 = **IDV {16, 19, 10, 5, 20}**. 설계 §2.2가 요구한 "모델 결과 관측 전 동결" 조건을 충족하므로, MAE 본 실험의 discriminative sub-analysis에 그대로 사용합니다.
:::

---

## 6. MAE 실험에 어떻게 적용할 것인가

1. **동일 stream·동일 평가의 anchor** — 본 실험의 train/test stream(run 번호), partition 정의, 평가 코드가 MAE 조건 A/B/B0와 완전히 공유됩니다. MAE 결과표의 simple-baseline 행은 이 결과를 그대로 옮기면 됩니다.
2. **Ĝ 보정의 기준선 확정** — 조건 B(label-blind MAE)의 G와 함께, 본 실험의 pca/nn 행이 "표현 기하가 다른 label-blind 검출기들의 G 분포"를 제공해 MAE-B의 G가 비정상인지 판별할 외부 참조가 됩니다. random 행은 partition-composition artifact의 자(ruler)로 기능합니다.
3. **Noisy-label 프로토콜 합류** — MAE label sweep(P0-4)에 본 실험의 oracle-removal floor 곡선을 같은 그래프에 병치합니다. 비교 셋: MAE-A(n% labels) vs oracle-removal simple vs MAE-B(0% labels) vs MAE-B0(clean). labeled 선택 규칙(각 fault의 앞쪽 k runs)을 MAE 쪽도 동일하게 사용해 point-to-point 비교를 보장합니다.
4. **Excluded-hard의 재정의** — "어떤 방법으로도 비식별" → **"point-wise 비식별 (IDV9는 시간-집계로도 비식별)"**로 문구를 정정하고, IDV3/15를 excluded-hard에서 분리해 **diagnostic 행**으로 별도 보고합니다(헤드라인 집계 제외는 유지 — 제외 규칙 자체는 point 통계 기준으로 사전 등록된 그대로).
5. **Subtle-set [16,19,10,5,20] 동결 적용** — discriminative sub-analysis의 부분집합으로 사용.
6. **보고 규율** — 모든 표에 random floor·prc_auc 병기, partition 간 raw 비교 금지(within-fold matched만), per-run boundary 정책 명시.

---

## 7. MAE 실험에서 기대되는 결과

본 실험이 만든 좌표계 위에서, MAE 본 실험의 가설별 예상 시나리오는 다음과 같습니다 (수치는 pak_auc_f1, seen partition 기준의 예상 범위):

<table fit-page-width="true" header-row="true">
	<tr>
		<td>조건</td>
		<td>H1 (purging 성립) 예상</td>
		<td>H2 (implicit classifier) 예상</td>
		<td>판별 근거</td>
	</tr>
	<tr>
		<td>MAE-B (label-blind, contaminated)</td>
		<td colspan="2">simple 모델처럼 seen 잠식 (PCA의 C_dmg 8~25%p와 유사 또는 그 이상 — 표현학습은 전역 흡수형)</td>
		<td>공통 기준선 — 여기서 갈리지 않음</td>
	</tr>
	<tr>
		<td>MAE-A (full labels)</td>
		<td>seen에서 B0(clean ceiling)에 근접 — **C_dmg의 대부분을 라벨로 회복**, unseen은 B 대비 무손상(Δ_unseen ≥ 0)</td>
		<td>seen 회복은 되지만 **unseen이 B보다 하락**(negative transfer) 또는 supervised skyline과 같은 폭으로 unseen 붕괴</td>
		<td>Ĝ_ours ≈ 0 (4/4 folds) vs Ĝ_ours ≈ Ĝ_sup</td>
	</tr>
	<tr>
		<td>MAE-A (n% labels, sweep)</td>
		<td>**oracle-removal floor 곡선을 상회** — 50% 라벨에서 이미 ceiling 근접 (within-type 일반화 purging: classifier가 라벨된 50%에서 학습한 signature로 unlabeled 50%도 정화)</td>
		<td>oracle-removal 곡선과 유사하거나 그 이하 (라벨된 인스턴스만 영향)</td>
		<td>**가장 선명한 판별 축** — 볼록한 floor 곡선 위로 얼마나 뜨는가</td>
	</tr>
	<tr>
		<td>IDV3/15 (diagnostic)</td>
		<td colspan="2">가설과 독립적인 보너스 축: point-wise 전멸 지점(roc≈0.51)에서 MAE roc이 유의하게 0.5를 넘으면 **window-맥락 모델의 구조적 우위** 입증 (단, run-집계 신호가 존재함이 확인된 IDV3/15에 한정; IDV9는 negative control로 0.5 근방이어야 정상)</td>
		<td>L2 분석이 상한 존재를 보증</td>
	</tr>
</table>

:::callout {icon="🚦" color="red_bg"}
**사전 경고 두 가지.** (1) **Gate 0 리스크**: clean 체제에서 simple도 만점이므로, supervised skyline이 unseen에서 무너지지 않으면(Ĝ_sup < δ) 이 벤치마크는 H1/H2를 분별하지 못합니다 — skyline 결과를 가장 먼저 확인해야 합니다. (2) **MAE-B의 결과가 simple과 다를 수 있음**: 표현학습의 오염 흡수가 PCA보다 약하거나 강할 수 있고, 그 자체가 중요한 발견입니다. 본 실험의 floor들은 예측이 아니라 *좌표*입니다 — MAE 결과가 어디에 떨어지든 해석 가능하게 만드는 것이 이 사전 실험의 역할입니다.
:::

---

*실험·분석 산출물: `analysis_report.md` (검증 게이트 + 전체 표), `idv_hard_report.md` (3단 검증), `sweep/` (20 conditions), per-fault 분해 (`per_fault_metrics.json` × 25). 사전 등록 설계: `temp/tep_design/80_experiment_design_final.md`. 모든 수치는 기존 baseline 파이프라인과 동일한 평가 코드(`compute_full_metric_set`)로 산출.*
