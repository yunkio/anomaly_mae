:::callout {icon="🎯" color="blue_bg"}
**TEP type-disjoint 실험의 simple baseline 5종(GPU 불필요, label-blind) 사전 실행 결과입니다.** 목적: ① 파이프라인 검증 ② MAE 본 실험 해석용 좌표계(floor·ceiling) 확보.

- **발견 1 — seen/unseen 격차는 fault 난이도가 아니라 train 오염이 만듭니다.** clean 학습에서는 격차가 사라집니다.
- **발견 2 — "라벨 = 인스턴스 제거" 전략은 부분 라벨 환경에서 거의 무력합니다.** 잔류 오염 소수가 피해의 대부분을 만듭니다.
- 두 발견 모두 **MAE의 GRL purging이 차별화될 지점**입니다. (GRL: 라벨된 이상이 정상 표현 학습에 스며들지 못하게 gradient를 반전시키는 정화 메커니즘)
:::

- **실험 번호** #12 · **결과** `temp/0610/TEP/results/12_20260610_211815_tep_typegen_simple/`
- **사전 등록 설계** `temp/tep_design/80_experiment_design_final.md` (2026-06-10 동결) · **스크립트** `temp/0610/TEP/` (기존 코드는 수정 없이 그대로 사용)
- **평가 코드** 기존 baseline 실험과 동일한 단일 경로 `compute_full_metric_set`
- **모델 5종** random · sensor_range · pca_error · l2_norm · nn_distance

---

## 1. 진행한 실험

### 1.1 왜 TEP인가

- **기존 벤치마크의 공통 한계** — SWaT·WaDi·PSM·SMD는 train 라벨의 anomaly 유형이 test에 그대로 다시 나타납니다. → "본 적 없는 유형도 잡는가"라는 MAE 핵심 질문을 측정할 수 없습니다.
- **TEP만 가능한 이유** — Downs & Vogel이 공인한 fault 분류 체계(IDV 1~20 = Step / Random variation / Slow drift / Sticking / Unknown)를 갖춘 유일한 표준 벤치마크입니다. → family 단위로 train과 test의 유형을 완전히 분리(type-disjoint)할 수 있습니다.

### 1.2 데이터 구성

사전 등록 설계에 따라 run 번호까지 결정론적으로 고정했습니다.

<table fit-page-width="true" header-row="true">
	<tr>
		<td>구성</td>
		<td>내용</td>
		<td>규모</td>
	</tr>
	<tr>
		<td>Train (fold별 contaminated)</td>
		<td>FaultFree runs 1~240 + seen family faulty 60 runs. fold 간 라벨 총량 통일: F-STEP 6종×10 / F-RAND 4종×15 / F-DS 2종×30 / F-UNK 5종×12</td>
		<td>288,000 samples · anomaly 16.67%</td>
	</tr>
	<tr>
		<td>Train (ffonly, clean 참조)</td>
		<td>FaultFree runs 1~240만 사용. MAE 본 실험의 clean 참조 조건 B0에 대응</td>
		<td>230,400 samples</td>
	</tr>
	<tr>
		<td>Test (4 fold 공유 고정)</td>
		<td>fault 20종 전부 × runs 441~460 + FaultFree runs 461~500. fold가 바뀌어도 stream은 동일하고 seen/unseen 분류 라벨만 변경</td>
		<td>440 runs = 422,400 samples</td>
	</tr>
	<tr>
		<td>라벨·경계 무결성</td>
		<td>각 faulty run의 sample 161부터 anomaly. 모든 연산은 run 경계를 넘지 않음</td>
		<td>region 400개 · internal seam 439개</td>
	</tr>
</table>

각 fold는 **하나의 family만 seen**으로 두는 hard 설정입니다. test stream은 4 fold가 공유하므로 fold 간 직접 비교가 가능합니다.

<table fit-page-width="true" header-row="true">
	<tr>
		<td>Fold</td>
		<td>Seen (train에 오염으로 주입)</td>
		<td>Unseen (test에서만 등장)</td>
	</tr>
	<tr>
		<td>F-STEP</td>
		<td>Step: IDV 1, 2, 4, 5, 6, 7</td>
		<td>나머지 11종</td>
	</tr>
	<tr>
		<td>F-RAND</td>
		<td>Random variation: IDV 8, 10, 11, 12</td>
		<td>나머지 13종</td>
	</tr>
	<tr>
		<td>F-DS</td>
		<td>Drift + Sticking: IDV 13, 14</td>
		<td>나머지 15종</td>
	</tr>
	<tr>
		<td>F-UNK</td>
		<td>Unknown: IDV 16~20</td>
		<td>나머지 12종</td>
	</tr>
</table>

- **IDV 3, 9, 15 = 고난도 fault** — 폐루프 제어가 외란을 보상해 탐지가 본질적으로 어렵습니다.
	- 사전 등록한 정량 규칙(post-onset 평균 max|z| < 정상 변동의 2배)에 따라 **headline 집계에서 제외**
	- test stream에는 그대로 두고 **excluded-hard partition**으로 별도 보고
	- 나머지 17종은 **usable**이라 부릅니다

### 1.3 실험 매트릭스

<table fit-page-width="true" header-row="true">
	<tr>
		<td>블록</td>
		<td>조건</td>
		<td>목적</td>
	</tr>
	<tr>
		<td>① 메인 (25 runs)</td>
		<td>simple 5종 × {contaminated 4 folds + ffonly}. random은 규약대로 5회 독립 추출의 mean±std</td>
		<td>label-blind 모델의 seen/unseen 격차와 오염 피해 측정</td>
	</tr>
	<tr>
		<td>② Noisy-label sweep (20 runs)</td>
		<td>pca_error·sensor_range × {F-STEP, F-DS} × labeled {0, 20, 50, 80, 100}%. 라벨된 비율의 오염 run은 이상적으로 제거, 나머지는 unlabeled 오염으로 잔류</td>
		<td>부분 라벨 환경에서 "라벨 = 인스턴스 제거" 전략의 한계 곡선</td>
	</tr>
	<tr>
		<td>③ IDV 3/9/15 심층 검증</td>
		<td>L1 point 수준 / L2 run 집계 수준 / L3 모델 score 수준의 3단계 구분 가능성 검사</td>
		<td>"구분 불가" 주장의 수준별 검증과 window 모델의 구조적 기회 탐색</td>
	</tr>
</table>

:::callout {icon="✅" color="green_bg"}
**검증 게이트 전 항목 PASS**
- stream 크기 · 라벨 개수 · partition 분리 · score 무결성 · sweep 끝점 일관성 — 전부 확인
- train 오염 비율 = 설계값 **16.67%와 정확히 일치**
- 기존 loader의 onset off-by-one 오류 정정 (정정된 onset = sample 161)
- pca_error의 smoothing은 run마다 재시작 — **440개 run 전부 경계 무결성 확인**
- 상세: `analysis_report.md` §1
:::

---

## 2. 실험 의도와 목적

**MAE 본 실험의 결과를 해석할 좌표계를 미리 만드는 사전 작업입니다.** MAE 본 실험의 다섯 조건:

- **A** 제안 방법 · **B** label-blind 대조군 · **B0** clean 정상만 학습하는 참조 · **C** supervised skyline (라벨 전부 사용, 성능 상한 참조) · **D** weak baseline

이 좌표계가 답해야 할 질문 세 가지:

- **Q1. 격차의 원인 분리** — unseen 성능 하락이 일반화 실패인지, 원래 어려운 fault인지, 오염의 부작용인지 구분하려면 **라벨을 안 쓰는 모델의 격차부터** 알아야 합니다. simple 5종은 모두 label-blind → 격차에 난이도와 오염 효과만 남습니다.
- **Q2. "라벨 = 오염 제거" 전략의 한계** — 라벨된 인스턴스를 학습에서 제거하는 이상적 행동(**oracle cleaning**)의 성능 곡선이 GRL purging이 넘어야 할 기준선입니다. GRL이 의미 있으려면 라벨된 인스턴스에서 배운 fault signature로 **unlabeled 동일 유형까지** 정화해야 합니다.
- **Q3. "구분 불가" 판정의 유효 범위** — IDV 3/9/15의 제외 규칙은 point 통계 기준입니다. window 길이 500의 시간 맥락으로도 불가능한지는 별개 질문 → 시간 집계 수준에서 구분된다면 **point-wise 전체가 실패하는 자리에서 window 모델만 이기는 구조적 기회**가 됩니다.

---

## 3. 결과 분석

### 3.1 주 결과 — macro per-fault G

**측정 방법 요약:**

- **지표** — pak_auc_f1: PA%K(region 안에서 K% 이상 탐지하면 region 전체 인정) 프로토콜에서 K를 0~100% 훑은 F1 곡선의 AUC
- **격차** — G = seen − unseen. **음수 = 학습 때 접한 seen 유형을 오히려 더 못 잡음**
- **문제: micro 평가의 구성 artifact** — partition별 fault 수가 달라 positive rate가 다름 (seen 41.7~62.5% vs unseen 70.5~73.5%) → F1 계열 지표가 왜곡됨
- **해결: macro per-fault** — fault별 점수(각 fault 20 runs + 동일 FF 40 runs, **positive rate 29.4% 고정**)를 그룹 평균 → seen/unseen이 완전히 같은 조건
- **검증 기준: random 행** — train을 안 보므로, 비교 조건이 정말 같다면 **G = 0이어야 함**

색 기준: |G| ≤ 0.02 초록 / ≤ 0.10 주황 / > 0.10 빨강, 양수 G는 굵게. 마지막 열 = clean 학습(ffonly)의 G 범위.

<table fit-page-width="true" header-row="true" header-column="true">
	<tr>
		<td>모델</td>
		<td>F-STEP</td>
		<td>F-RAND</td>
		<td>F-DS</td>
		<td>F-UNK</td>
		<td>ffonly (clean) macro G 범위</td>
	</tr>
	<tr>
		<td>random (검증 기준 행)</td>
		<td color="green">−0.000</td>
		<td color="green">+0.001</td>
		<td color="green">−0.002</td>
		<td color="green">+0.000</td>
		<td color="green">−0.002 ~ +0.002</td>
	</tr>
	<tr>
		<td>pca_error</td>
		<td color="orange">−0.093</td>
		<td color="orange">−0.025</td>
		<td color="red">−0.161</td>
		<td color="red">−0.127</td>
		<td color="green">−0.009 ~ +0.008 (≈0)</td>
	</tr>
	<tr>
		<td>l2_norm</td>
		<td color="orange">**+0.045**</td>
		<td color="orange">−0.070</td>
		<td color="orange">−0.030</td>
		<td color="orange">−0.099</td>
		<td>−0.153 ~ +0.138</td>
	</tr>
	<tr>
		<td>nn_distance</td>
		<td color="orange">−0.034</td>
		<td color="green">−0.018</td>
		<td color="orange">**+0.093**</td>
		<td color="orange">−0.096</td>
		<td>−0.076 ~ +0.066</td>
	</tr>
	<tr>
		<td>sensor_range</td>
		<td color="red">−0.371</td>
		<td color="red">−0.322</td>
		<td color="red">−0.428</td>
		<td color="red">−0.179</td>
		<td>−0.286 ~ +0.190</td>
	</tr>
</table>

:::callout {icon="📐" color="purple_bg"}
**핵심 1 — micro 격차는 전부 구성 artifact였습니다.**
- random의 macro G = **네 fold 전부 ±0.002 이내로 소멸** (micro에서는 −0.03~−0.16)
- 왜곡은 부호까지 뒤집음 — l2_norm F-STEP: micro −0.022 → macro **+0.045**
- → stream micro G는 보조 수치로만 사용, 전체 표는 부록 A
:::

:::callout {icon="🔬" color="blue_bg"}
**핵심 2 — 조건을 맞춘 뒤에도 남는 격차가 실제 오염 효과입니다.**
- pca_error F-DS **−0.161**, sensor_range 전 fold **−0.18~−0.43**
- 결정적 대조: clean 학습의 pca_error는 **전 fold |G| < 0.010** → usable 17종의 순수 난이도 격차는 거의 없음
- 메커니즘: seen 유형이 train에 unlabeled 오염으로 포함 → 검출기가 그 패턴을 정상으로 학습 → **정확히 seen 쪽만 무너짐**
:::

표를 읽을 때의 단서 두 가지:

- **약한 검출기는 clean에서도 G가 흔들립니다** — sensor_range는 ffonly에서도 −0.286~+0.190 (F-UNK의 −0.286은 subtle fault 16/19/20이 모두 seen에 몰린 결과). "난이도 격차 ≈ 0"은 충분히 강한 검출기 기준의 진술입니다.
- **nn_distance F-DS의 +0.093** — 표에서 유일하게 큰 양수. 정체는 §3.2에서 밝혀집니다.

### 3.2 오염 피해 — C_dmg

- **정의** — C_dmg = clean(ffonly) seen 점수 − contaminated seen 점수. **클수록 피해가 큼.** §3.1과 같은 macro 척도 사용.
- random 행 = 0이 나와야 정상인 대조 행 · 색 기준: < 0.05 초록 / > 0.15 빨강 / 사이 주황

<table fit-page-width="true" header-row="true" header-column="true">
	<tr>
		<td>모델</td>
		<td>F-STEP</td>
		<td>F-RAND</td>
		<td>F-DS</td>
		<td>F-UNK</td>
		<td>해석</td>
	</tr>
	<tr>
		<td>random (대조 행)</td>
		<td color="green">−0.001</td>
		<td color="green">+0.001</td>
		<td color="green">−0.001</td>
		<td color="green">+0.001</td>
		<td>train 무관, 피해 없음</td>
	</tr>
	<tr>
		<td>pca_error</td>
		<td color="red">0.162</td>
		<td color="orange">0.074</td>
		<td color="red">0.209</td>
		<td color="orange">0.144</td>
		<td>전역 흡수 (fault 방향이 부분공간에 흡수됨)</td>
	</tr>
	<tr>
		<td>l2_norm</td>
		<td color="orange">0.139</td>
		<td color="orange">0.141</td>
		<td color="red">0.231</td>
		<td color="orange">0.079</td>
		<td>표준화 통계 오염 (train 분산 팽창)</td>
	</tr>
	<tr>
		<td>nn_distance</td>
		<td color="orange">0.117</td>
		<td color="orange">0.091</td>
		<td color="green">**0.036**</td>
		<td color="orange">0.095</td>
		<td>국소 흡수 (오염이 점유한 좌표 근방만 가려짐)</td>
	</tr>
	<tr>
		<td>sensor_range</td>
		<td color="red">0.785</td>
		<td color="red">0.858</td>
		<td color="red">0.989</td>
		<td color="red">0.526</td>
		<td>메커니즘 붕괴 (min·max 범위 확장)</td>
	</tr>
</table>

:::callout {icon="🧭" color="blue_bg"}
**핵심 3 — 오염 피해의 형태는 detector의 기하학적 구조가 결정합니다.**
- **pca_error = 전역 흡수** — fault 방향을 부분공간이 통째로 흡수 → 같은 family 전체가 약화
- **nn_distance = 국소 흡수** — 오염 run이 점유한 좌표 근방만 가려짐 → F-DS 피해가 **0.036**에 그침
- **MAE는 전역 흡수형에 가까울 것** → GRL purging이 회복해야 할 피해도 전역적일 가능성이 높음
:::

표 밖의 관찰 네 가지:

- **near-variable spillover** — F-STEP에서 unseen인 IDV11이 0.996→0.722로 동반 하락. 오염된 seen fault와 **같은 물리 변수를 공유**하기 때문 → 전역 흡수의 피해는 family 경계 밖으로도 번집니다.
- **nn_distance 국소성의 전제** — drift·sticking은 run마다 궤적이 달라, 오염 run의 좌표가 test run과 거의 겹치지 않음 → 가려짐이 일어날 자리가 없어 피해 최소.
- **+0.093의 정체 = 오염이 아니라 원래 우위** — nn_distance에게 drift·sticking은 본래 가장 잘 잡히는 유형 (clean에서도 G +0.066). 오염 후 seen은 0.036, unseen은 0.062 내려가 격차가 +0.093으로 벌어진 것.
- **sensor_range = 흡수가 아니라 붕괴** — faulty run 하나가 min·max 범위를 넓히는 순간 탐지 자체가 무력화 (C_dmg 0.53~0.99).

### 3.3 Noisy-label sweep — "라벨 = 제거" 전략의 한계 곡선

- **조건** — 라벨된 n%의 오염 run을 이상적으로 제거(oracle cleaning)하고 나머지는 unlabeled로 잔류
- 표 = seen partition의 pak_auc_f1, 괄호 = 잔류 오염 run 수. 같은 열 안에서는 평가 구성이 같아 직접 비교 가능

<table fit-page-width="true" header-row="true" header-column="true">
	<tr>
		<td>labeled % (잔류 오염 runs)</td>
		<td>pca_error @ F-STEP</td>
		<td>pca_error @ F-DS</td>
		<td>sensor_range @ F-STEP</td>
		<td>sensor_range @ F-DS</td>
	</tr>
	<tr>
		<td>0% (60)</td>
		<td>0.874</td>
		<td>0.747</td>
		<td color="red">0.184</td>
		<td color="red">0.009</td>
	</tr>
	<tr>
		<td>20% (48)</td>
		<td>0.879</td>
		<td>0.748</td>
		<td color="red">0.200</td>
		<td color="red">0.009</td>
	</tr>
	<tr>
		<td>50% (30)</td>
		<td>0.943</td>
		<td>0.757</td>
		<td color="red">0.212</td>
		<td color="red">0.009</td>
	</tr>
	<tr>
		<td>80% (12)</td>
		<td>0.975</td>
		<td color="orange">0.836</td>
		<td color="red">0.304</td>
		<td color="red">0.021</td>
	</tr>
	<tr>
		<td>100% (0, clean)</td>
		<td color="green">**0.9997**</td>
		<td color="green">**0.999**</td>
		<td color="green">**0.946**</td>
		<td color="green">**0.998**</td>
	</tr>
</table>

:::callout {icon="⚠️" color="orange_bg"}
**핵심 4 — 회복 곡선이 심하게 볼록합니다. 이득이 마지막 구간에 몰려 있습니다.**
- 오염의 **80%를 이상적으로 제거해도** pca_error F-DS는 0.836 (clean 0.999에 한참 미달)
- sensor_range는 **100% 직전까지 사실상 0**
- → **잔류 오염 소수가 피해의 대부분**을 만들며, "라벨된 것을 제거"하는 전략은 부분 라벨 환경에서 거의 무가치
:::

- 볼록한 모양의 원인 — **pca_error**: 잔류 fault run 12개만으로도 그 방향이 부분공간에 흡수됨 / **sensor_range**: 단 하나의 unlabeled run이 범위를 넓혀 무력화.

### 3.4 IDV 3/9/15 — 3단계 검증

- **L1** = 단일 feature의 point 값으로 구분되는가 / **L2** = run당 800 sample 집계(mean·std)로 구분되는가 / **L3** = simple 모델의 score로 구분되는가
- 참조선: IDV1 (가장 쉬움) · IDV16/19 (usable 중 가장 subtle)

<table fit-page-width="true" header-row="true" header-column="true">
	<tr>
		<td>Fault</td>
		<td>L1: point 단일 feature AUC</td>
		<td>L2: run 집계 best AUC</td>
		<td>L3: 모델 score ROC (pca@ffonly)</td>
		<td>판정</td>
	</tr>
	<tr>
		<td>IDV3 (D feed 온도 step)</td>
		<td color="red">0.568</td>
		<td color="green">**1.000** (xmeas_21 run-mean)</td>
		<td color="red">0.510</td>
		<td>point 구분 불가 · 시간 집계로 구분 가능</td>
	</tr>
	<tr>
		<td>IDV9 (D feed 온도 random)</td>
		<td color="red">0.514</td>
		<td color="red">0.740 (우연 수준※)</td>
		<td color="red">0.513</td>
		<td>모든 수준에서 구분 불가</td>
	</tr>
	<tr>
		<td>IDV15 (응축기 밸브 sticking)</td>
		<td color="red">0.515</td>
		<td color="green">**0.967** (xmeas_22 run-std)</td>
		<td color="red">0.512</td>
		<td>point 구분 불가 · 시간 집계로 구분 가능</td>
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
		<td>다변량 상관구조로 식별</td>
	</tr>
</table>

※ L2는 52개 feature × 2가지 집계 = 104회 비교의 최댓값이라 selection bias가 있습니다. 신호가 전혀 없어도 (run 20 vs 40에서 AUC 표준편차 ≈ 0.08) 우연만으로 최댓값 기대치가 약 0.72, 상위 5% 경계가 0.77입니다. → IDV9의 0.740은 잡음과 구분되지 않고, IDV3/15의 1.000과 0.967은 실제 신호입니다.

:::callout {icon="💡" color="yellow_bg"}
**핵심 5 — fault 난이도는 4계층으로 나뉩니다.**
- ① **IDV1형** — 단일 feature의 point 값만으로 구분
- ② **IDV16/19형** — point 수준이되 다변량 상관구조 필요 (단일 feature AUC 0.51 ↔ PCA ROC 0.966/0.992)
- ③ **IDV3/15형 — 시간 집계로만 구분** (point에서는 모든 simple 모델이 놓침 · run 집계 AUC 1.000/0.967)
- ④ **IDV9형** — 어느 수준에서도 구분 불가
- → 설계의 "어떤 방법으로도 구분 불가" 문구는 **"point-wise 방법으로는 구분 불가"로 한정**해야 정확
:::

### 3.5 지표 calibration — random floor

- **PA%K의 floor는 상수가 아니라 stream 구성의 함수입니다** — 같은 random이 full stream에서 **0.765** (positive rate 75.8%, region 길이 800), per-fault 평가에서는 **0.48** (positive rate 29.4%)
- **이 stream에서 pak_auc_f1 0.76 이하 = 사실상 random 이하**
- excluded-hard에서 pca_error의 0.79도 prc_auc 0.51 (≈ positive rate 50%)과 함께 보면 **사실상 무신호**
- → **MAE 결과표에는 random 행과 prc_auc 병기가 필수**

---

## 4. 해석

**한 줄 요약 — label-blind 모델의 세계에 type-generalization 격차는 없습니다. 존재하는 것은 오염 격차뿐입니다.**

1. **난이도 가설 기각, 오염 가설 채택** — clean 학습에서 pca_error는 usable 17종 중 14종을 0.99+(최저 0.966)로 잡고 |G| < 0.010. 같은 모델이 오염 train에서는 **seen만 선택적으로 붕괴**. MAE 조건 B가 보일 행동의 예고편입니다.
2. **피해의 형태는 detector 기하가 결정** — 부분공간 방법은 전역 흡수, 인스턴스 방법은 국소 흡수. MAE는 전역형에 가까울 것 → **GRL purging = "전역적으로 흡수될 뻔한 fault 방향을 라벨로 도려내기"**로 정식화할 수 있습니다.
3. **부분 라벨의 진짜 병목 = 잔류 오염** — oracle cleaning은 80% 라벨에서도 피해의 절반을 회복하지 못함 (F-DS 0.836 vs clean 0.999). 라벨이 의미 있으려면 **라벨된 소수의 signature가 unlabeled 다수로 일반화**되어야 합니다.
4. **IDV3/15 = window 모델의 고유 영역** — point-wise 방법은 구조적으로 닿지 못하지만 시간 집계에는 신호가 남음. MAE가 여기서 잡으면 성능 우위가 아니라 **방법론 계층이 다르다는 증거**가 됩니다.

---

## 5. 인사이트

:::callout {icon="1️⃣" color="blue_bg"}
**raw seen/unseen 격차는 type-generalization의 증거가 아닙니다.**
- label-blind 모델조차 격차를 보였고, 방향은 오히려 seen이 나쁜 쪽
- 의미 있는 비교 = **macro per-fault 계산 + 보정값 Ĝ** (Ĝ = 모델의 G − 같은 fold matched control의 G)
- 근거: random의 micro G가 −0.16까지 갔다가 macro에서 사라짐
:::

:::callout {icon="2️⃣" color="purple_bg"}
**라벨의 가치는 인스턴스 제거가 아니라 같은 유형 전체로 번지는 정화에 있습니다.**
- oracle cleaning 곡선의 극단적 볼록성 = "라벨된 인스턴스에만 작용하는 접근"의 상한
- 우위가 생기려면 GRL이 라벨된 signature를 학습해 **unlabeled 동일 유형까지 정화**해야 함
- → MAE noisy-label 실험(설계 표기 P0-4)의 평가 축
:::

:::callout {icon="3️⃣" color="orange_bg"}
**PA%K 절대값은 stream 구성을 떠나면 의미가 없습니다.**
- 같은 random이 full stream 0.765 ↔ per-fault 0.48 — floor가 positive rate·region 길이의 함수
- → 모든 결과표에 **random 행 + prc_auc 병기 필수**
:::

:::callout {icon="4️⃣" color="green_bg"}
**IDV3/15는 point-wise 방법론 전체의 공백 지대입니다.**
- 폐루프 제어가 point 분포를 가리지만 run 집계에는 신호가 남음 (AUC 1.000 / 0.967)
- = window 맥락 없이는 잡히지 않는 anomaly의 실제 사례 → **MAE 진단 케이스**로 사용
- IDV9는 신호가 없어야 정상인 대조 케이스
:::

:::callout {icon="5️⃣" color="yellow_bg"}
**TEP의 변별력은 clean 조건이 아니라 contaminated 조건에 있습니다.**
- clean train에서는 simple 방법조차 거의 만점 → 그 조건의 우위는 가설을 가르지 못함
- 의미 있는 비교 축 3가지 = **오염 내성 · 라벨 회복 · 부분 라벨 유지**
- → supervised skyline이 unseen에서 무너지는지 먼저 확인하는 **Gate 0**의 중요성 증가
:::

---

## 6. MAE 실험에 어떻게 적용할 것인가

1. **Anchor로 재사용** — train/test stream의 run 번호·partition 정의·평가 코드를 MAE 조건 A/B/B0가 그대로 물려받습니다. simple baseline 행은 이 결과를 옮겨 적기만 하면 됩니다.
2. **주 비교 = macro per-fault Ĝ** — fault별 평가 구성이 같아 비교 조건이 완전히 동일. 설계 §4.4(b)의 co-primary 결정이 옳았음을 이번 결과가 실증. stream micro 지표는 보조로만 둡니다.
3. **Sweep 곡선 겹쳐 그리기** — MAE-A(n% labels) · oracle cleaning simple · MAE-B · MAE-B0를 한 그래프에. labeled run 선택 규칙(각 fault의 앞쪽 k runs)도 동일하게 적용해 point 단위 비교를 보장합니다.
4. **excluded-hard 서술 정정** — "어떤 방법으로도 구분 불가" → **"point-wise 방법으로는 구분 불가"**로 한정. IDV3/15는 diagnostic 행으로 분리 보고 (headline 제외는 사전 등록 규칙 그대로 유지).
5. **subtle-set 동결 적용** — IDV **{16, 19, 10, 5, 20}** (post-onset 평균 max|z| 하위 5종). 모델 결과 관측 전에 데이터 통계만으로 확정했으므로 설계 §2.2의 동결 조건 충족. 신호가 약할수록 모델 간 우열이 선명하므로 판별용 하위 분석(discriminative sub-analysis)에 사용합니다.
6. **보고 규칙 고정** — 모든 표에 random 행 + prc_auc 병기 · partition 간 raw 비교 금지(within-fold matched만) · per-run 경계 정책 명시.

---

## 7. MAE 실험에서 기대되는 결과

- **H1** (MAE 핵심 가설) — 라벨은 정상 모델의 정화에만 쓰인다
- **H2** (대립 가설) — 라벨은 seen 유형을 암기하는 implicit classifier를 만들 뿐
- 표의 수치는 seen partition pak_auc_f1 기준 예상 범위, Ĝ는 macro per-fault 기준 (보정 기준 = 인사이트 1의 matched control)

<table fit-page-width="true" header-row="true" header-column="true">
	<tr>
		<td>조건</td>
		<td>H1 (purging 성립) 예상</td>
		<td>H2 (implicit classifier) 예상</td>
		<td>판별 근거</td>
	</tr>
	<tr>
		<td>MAE-B (label-blind, contaminated)</td>
		<td>simple 모델처럼 seen 성능이 깎임. 전역 흡수형이므로 pca_error의 macro C_dmg 0.07~0.21과 유사 또는 그 이상</td>
		<td>H1과 동일 (공통 기준선)</td>
		<td>여기서는 가설이 갈리지 않음</td>
	</tr>
	<tr>
		<td>MAE-A (full labels)</td>
		<td>seen이 B0 ceiling에 근접 (C_dmg 대부분 회복), unseen은 B 대비 손상 없음</td>
		<td>seen은 회복되나 unseen이 B보다 하락 (negative transfer), 또는 skyline과 같은 폭의 unseen 붕괴</td>
		<td>Ĝ_ours ≈ 0 (4/4 folds) vs Ĝ_ours ≈ Ĝ_sup</td>
	</tr>
	<tr>
		<td>MAE-A (n% labels, sweep)</td>
		<td>oracle cleaning floor 곡선 상회, 50% 라벨에서 이미 ceiling 근접 (within-type 일반화 purging)</td>
		<td>oracle cleaning 곡선과 유사 또는 그 이하 (라벨된 인스턴스만 영향)</td>
		<td>**가장 선명한 판별 축**. 볼록한 floor 곡선 위로 얼마나 뜨는가</td>
	</tr>
	<tr>
		<td>IDV3/15 (diagnostic)</td>
		<td>point-wise 방법이 모두 실패하는 지점에서 ROC가 0.5를 유의하게 상회하면 window 모델의 구조적 우위 입증</td>
		<td>가설과 독립적인 보너스 축. IDV9는 신호가 없어야 정상인 대조 케이스라 0.5 근방이 정상</td>
		<td>L2 분석이 상한의 존재를 보증</td>
	</tr>
</table>

:::callout {icon="🚦" color="red_bg"}
**사전 경고 두 가지**
- **Gate 0 먼저** — clean 조건에서는 simple도 거의 만점이므로, supervised skyline이 unseen에서 무너지지 않으면 이 벤치마크는 H1/H2를 변별하지 못합니다. 해석은 skyline 확인에서 시작해야 합니다.
- **MAE-B는 simple과 다를 수 있습니다** — 표현학습의 오염 흡수가 PCA보다 약하거나 강할 수 있고, 그 자체가 중요한 발견입니다. 이 사전 실험의 floor는 예측이 아니라 **좌표**입니다.
:::

---

## 부록 A. stream micro G (참고용)

stream 전체를 한 덩어리로 평가한 micro G입니다. §3.1에서 확인한 구성 artifact가 섞여 있어 **partition 간 비교나 가설 판정에는 사용하지 않습니다.**

<table fit-page-width="true" header-row="true" header-column="true">
	<tr>
		<td>모델</td>
		<td>F-STEP</td>
		<td>F-RAND</td>
		<td>F-DS</td>
		<td>F-UNK</td>
	</tr>
	<tr>
		<td>random</td>
		<td>−0.033</td>
		<td>−0.070</td>
		<td>−0.160</td>
		<td>−0.049</td>
	</tr>
	<tr>
		<td>pca_error</td>
		<td>−0.052</td>
		<td>−0.025</td>
		<td>−0.204</td>
		<td>−0.116</td>
	</tr>
	<tr>
		<td>l2_norm</td>
		<td>−0.022</td>
		<td>−0.068</td>
		<td>−0.121</td>
		<td>−0.073</td>
	</tr>
	<tr>
		<td>nn_distance</td>
		<td>−0.051</td>
		<td>−0.035</td>
		<td>+0.030</td>
		<td>−0.035</td>
	</tr>
	<tr>
		<td>sensor_range</td>
		<td>−0.432</td>
		<td>−0.444</td>
		<td>−0.573</td>
		<td>−0.261</td>
	</tr>
</table>

- train을 보지 않는 random조차 −0.03~−0.16의 격차 · l2_norm F-STEP은 macro와 부호 반대

---

**산출물 위치** — 검증 게이트·전체 표: `analysis_report.md` · 3단계 검증: `idv_hard_report.md` · sweep 20조건: `sweep/` · per-fault 분해: `per_fault_metrics.json` · 사전 등록 설계: `temp/tep_design/80_experiment_design_final.md` · 모든 수치는 기존 baseline 파이프라인과 동일한 평가 코드 `compute_full_metric_set`으로 산출
