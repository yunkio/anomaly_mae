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

:::callout {icon="🧪" color="gray_bg"}
**이 실험은** — simple 5종을 두 가지 train으로 학습시켰습니다: ① fold별 **contaminated** (정상 240 runs + 그 fold의 seen-family fault 60 runs가 라벨 없이 섞임) ② **clean** (정상 240 runs만). 평가는 4 fold가 공유하는 동일한 test (fault 20종 × 20 runs + 정상 40 runs).
**왜** — 라벨을 전혀 쓰지 않는 모델의 seen/unseen 격차를 재서, 격차의 원인이 fault 난이도인지 train 오염인지를 분리합니다 (§2의 Q1). 모델이 label-blind이므로 fold 간 차이는 오직 "train에 어떤 fault가 섞였는가"뿐입니다.
:::

**측정 방법 요약:**

- **용어** — **seen** = 그 fold에서 train에 (라벨 없이) 섞여 있던 fault 유형들 · **unseen** = train에는 전혀 없고 test에서 처음 등장하는 유형들. 예를 들어 F-STEP fold라면 seen = step 6종(IDV1,2,4,5,6,7), unseen = 나머지 usable 11종입니다. 표의 점수는 각 그룹에 속한 fault들의 점수 평균입니다.
- **지표** — pak_auc_f1: PA%K(region 안에서 K% 이상 탐지하면 region 전체 인정) 프로토콜에서 K를 0~100% 훑은 F1 곡선의 AUC
- **격차** — G = seen − unseen. **음수 = 학습 때 접한 seen 유형을 오히려 더 못 잡음**
- **문제: micro 평가의 구성 artifact** — partition별 fault 수가 달라 positive rate가 다름 (seen 41.7~62.5% vs unseen 70.5~73.5%) → F1 계열 지표가 왜곡됨
- **해결: macro per-fault** — fault별 점수(각 fault 20 runs + 동일 FF 40 runs, **positive rate 29.4% 고정**)를 그룹 평균 → seen/unseen이 완전히 같은 조건
- **검증 기준: random 행** — train을 안 보므로, 비교 조건이 정말 같다면 **G = 0이어야 함**

**① 실제 점수 — contaminated train (학습에 seen-family 오염 60 runs 포함)**

- cell = macro per-fault pak_auc_f1의 **seen / unseen** 실제 값
- random의 값이 모든 칸에서 0.481~0.483으로 일정 — per-fault 평가의 비교 조건이 같다는 직접 증거 (random floor ≈ 0.48, §3.5)

<table fit-page-width="true" header-row="true" header-column="true">
	<tr>
		<td>모델 (cell = seen / unseen)</td>
		<td>F-STEP</td>
		<td>F-RAND</td>
		<td>F-DS</td>
		<td>F-UNK</td>
	</tr>
	<tr>
		<td>random</td>
		<td>0.482 / 0.483</td>
		<td>0.483 / 0.481</td>
		<td>0.481 / 0.483</td>
		<td>0.482 / 0.482</td>
	</tr>
	<tr>
		<td>sensor_range</td>
		<td>0.133 / 0.504</td>
		<td>0.010 / 0.333</td>
		<td>0.009 / 0.437</td>
		<td>0.103 / 0.283</td>
	</tr>
	<tr>
		<td>pca_error</td>
		<td>0.837 / 0.930</td>
		<td>0.919 / 0.943</td>
		<td>0.789 / 0.950</td>
		<td>0.844 / 0.971</td>
	</tr>
	<tr>
		<td>l2_norm</td>
		<td>0.791 / 0.746</td>
		<td>0.695 / 0.765</td>
		<td>0.754 / 0.784</td>
		<td>0.675 / 0.774</td>
	</tr>
	<tr>
		<td>nn_distance</td>
		<td>0.844 / 0.877</td>
		<td>0.857 / 0.875</td>
		<td>0.963 / 0.871</td>
		<td>0.792 / 0.888</td>
	</tr>
</table>

**읽는 예** — pca_error × F-STEP의 "0.837 / 0.930": 이 모델은 step fault 6종이 섞인 train으로 학습되었습니다. 0.837은 바로 그 6종(seen)의 탐지 점수 평균이고, 0.930은 train에 없던 나머지 11종(unseen)의 평균입니다. 즉 **학습 때 접한 유형(0.837)을 처음 보는 유형(0.930)보다 오히려 못 잡습니다.** 평균의 내역(IDV1 0.749 · IDV2 0.877 · IDV4 0.619 · IDV5 0.781 · IDV6 0.999 · IDV7 1.000)은 부록 B에서 fault별로 확인할 수 있습니다.

**② 실제 점수 — clean train (ffonly, 오염 0)**

- 같은 모델·같은 평가에서 train만 깨끗할 때의 값 — ①과 같은 칸끼리 비교하면 오염의 효과가 그대로 보입니다
- **주의: clean 모델에게 seen/unseen은 학습 이력이 아니라 "빌려온 그룹 이름"입니다.** clean 모델은 fault를 하나도 본 적이 없습니다. 표 ①과 칸 대 칸 비교가 가능하도록 점수를 같은 fault 그룹(해당 fold의 seen 그룹 / unseen 그룹)으로 묶어 평균 냈을 뿐입니다. clean에서 두 그룹 점수가 거의 같다는 사실 자체가 "그룹 간 원래 난이도 차이는 없다"는 증거입니다

<table fit-page-width="true" header-row="true" header-column="true">
	<tr>
		<td>모델 (cell = seen / unseen)</td>
		<td>F-STEP</td>
		<td>F-RAND</td>
		<td>F-DS</td>
		<td>F-UNK</td>
	</tr>
	<tr>
		<td>random</td>
		<td>0.481 / 0.483</td>
		<td>0.483 / 0.482</td>
		<td>0.481 / 0.482</td>
		<td>0.483 / 0.482</td>
	</tr>
	<tr>
		<td>sensor_range</td>
		<td>0.918 / 0.783</td>
		<td>0.869 / 0.819</td>
		<td>0.998 / 0.808</td>
		<td>0.629 / 0.915</td>
	</tr>
	<tr>
		<td>pca_error</td>
		<td>1.000 / 0.992</td>
		<td>0.993 / 0.995</td>
		<td>0.999 / 0.994</td>
		<td>0.988 / 0.997</td>
	</tr>
	<tr>
		<td>l2_norm</td>
		<td>0.930 / 0.826</td>
		<td>0.836 / 0.871</td>
		<td>0.985 / 0.846</td>
		<td>0.754 / 0.908</td>
	</tr>
	<tr>
		<td>nn_distance</td>
		<td>0.961 / 0.930</td>
		<td>0.948 / 0.938</td>
		<td>0.999 / 0.933</td>
		<td>0.887 / 0.963</td>
	</tr>
</table>

**읽는 예** — 같은 pca_error × F-STEP이 여기서는 "1.000 / 0.992": 오염 없는 train으로 학습하면 같은 step 6종을 1.000으로 잡습니다. 표 ①의 0.837과 비교하면 **seen 쪽 하락은 전부 train 오염 탓**임이 드러납니다. clean 모델은 fold와 무관하게 하나뿐이고 어떤 fault도 본 적이 없지만, 표 ①과 칸 대 칸 비교를 위해 점수를 같은 fault 그룹으로 묶어 평균 내기 때문에 fold별 열이 존재합니다.

**③ 격차 요약 — G = seen − unseen (①에서 파생)**

- 색: |G| ≤ 0.02 초록 / ≤ 0.10 주황 / > 0.10 빨강, 양수 G는 굵게. 마지막 열 = ②에서 계산한 clean G의 4-fold 범위

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
		<td color="green">-0.000</td>
		<td color="green">+0.001</td>
		<td color="green">-0.002</td>
		<td color="green">+0.000</td>
		<td color="green">-0.002 ~ +0.002</td>
	</tr>
	<tr>
		<td>sensor_range</td>
		<td color="red">-0.371</td>
		<td color="red">-0.322</td>
		<td color="red">-0.428</td>
		<td color="red">-0.179</td>
		<td>-0.286 ~ +0.190</td>
	</tr>
	<tr>
		<td>pca_error</td>
		<td color="orange">-0.093</td>
		<td color="orange">-0.025</td>
		<td color="red">-0.161</td>
		<td color="red">-0.127</td>
		<td color="green">-0.009 ~ +0.008</td>
	</tr>
	<tr>
		<td>l2_norm</td>
		<td color="orange">**+0.045**</td>
		<td color="orange">-0.070</td>
		<td color="orange">-0.030</td>
		<td color="orange">-0.099</td>
		<td>-0.153 ~ +0.138</td>
	</tr>
	<tr>
		<td>nn_distance</td>
		<td color="orange">-0.034</td>
		<td color="green">-0.018</td>
		<td color="orange">**+0.093**</td>
		<td color="orange">-0.096</td>
		<td>-0.076 ~ +0.066</td>
	</tr>
</table>

**읽는 예** — pca_error × F-STEP의 −0.093 = 0.837 − 0.930 (표 ①의 두 값의 차). 음수 = seen을 더 못 잡는다는 뜻입니다. 첫 행의 random이 ±0.002로 0에 붙어 있으므로, 0에서 벗어난 G는 평가 왜곡이 아니라 모델과 오염이 만든 실제 격차입니다.

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

:::callout {icon="🧪" color="gray_bg"}
**이 분석은** — §3.1의 두 학습 조건(clean vs contaminated)에서 같은 모델의 **seen 점수를 직접 맞대어** 비교합니다 (새 실험 없이 §3.1 표 ①·②의 재구성).
**왜** — 오염이 깎는 성능의 크기와 형태를 정량화해, 나중에 MAE의 GRL purging이 "얼마나 회복했는지"를 잴 자를 만듭니다.
:::

- **정의** — C_dmg = clean(ffonly) seen 점수 − contaminated seen 점수. **클수록 피해가 큼.** §3.1과 같은 macro 척도 사용.
- cell에 두 원천 점수를 그대로 표기: **clean seen → contaminated seen (피해)**
- random 행 = 0이 나와야 정상인 대조 행 · 색 기준: 피해 < 0.05 초록 / > 0.15 빨강 / 사이 주황

<table fit-page-width="true" header-row="true" header-column="true">
	<tr>
		<td>모델 (cell = clean seen → contaminated seen)</td>
		<td>F-STEP</td>
		<td>F-RAND</td>
		<td>F-DS</td>
		<td>F-UNK</td>
		<td>해석</td>
	</tr>
	<tr>
		<td>random (대조 행)</td>
		<td color="green">0.481 → 0.482  (피해 -0.001)</td>
		<td color="green">0.483 → 0.483  (피해 +0.001)</td>
		<td color="green">0.481 → 0.481  (피해 -0.001)</td>
		<td color="green">0.483 → 0.482  (피해 +0.001)</td>
		<td>train 무관, 피해 없음</td>
	</tr>
	<tr>
		<td>sensor_range</td>
		<td color="red">0.918 → 0.133  (피해 +0.785)</td>
		<td color="red">0.869 → 0.010  (피해 +0.858)</td>
		<td color="red">0.998 → 0.009  (피해 +0.989)</td>
		<td color="red">0.629 → 0.103  (피해 +0.526)</td>
		<td>메커니즘 붕괴 (min·max 범위 확장)</td>
	</tr>
	<tr>
		<td>pca_error</td>
		<td color="red">1.000 → 0.837  (피해 +0.162)</td>
		<td color="orange">0.993 → 0.919  (피해 +0.074)</td>
		<td color="red">0.999 → 0.789  (피해 +0.209)</td>
		<td color="orange">0.988 → 0.844  (피해 +0.144)</td>
		<td>전역 흡수 (fault 방향이 부분공간에 흡수됨)</td>
	</tr>
	<tr>
		<td>l2_norm</td>
		<td color="orange">0.930 → 0.791  (피해 +0.139)</td>
		<td color="orange">0.836 → 0.695  (피해 +0.141)</td>
		<td color="red">0.985 → 0.754  (피해 +0.231)</td>
		<td color="orange">0.754 → 0.675  (피해 +0.079)</td>
		<td>표준화 통계 오염 (train 분산 팽창)</td>
	</tr>
	<tr>
		<td>nn_distance</td>
		<td color="orange">0.961 → 0.844  (피해 +0.117)</td>
		<td color="orange">0.948 → 0.857  (피해 +0.091)</td>
		<td color="green">0.999 → 0.963  (피해 +0.036)</td>
		<td color="orange">0.887 → 0.792  (피해 +0.095)</td>
		<td>국소 흡수 (오염이 점유한 좌표 근방만 가려짐)</td>
	</tr>
</table>

**읽는 예** — pca_error × F-STEP의 "1.000 → 0.837 (피해 +0.162)": 왼쪽 1.000은 clean train일 때의 step 6종 평균(표 ②), 오른쪽 0.837은 같은 6종이 train에 섞였을 때의 평균(표 ①)입니다. 오염 60 runs를 학습에 받아들인 대가로 그 유형의 탐지력 0.162를 잃었다는 뜻입니다.

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

:::callout {icon="🧪" color="gray_bg"}
**이 실험은** — contaminated train의 fault 60 runs 중 "라벨이 있다고 가정한 n%"를 학습에서 이상적으로 제거(oracle cleaning)하고, 나머지 (100−n)%만 unlabeled 오염으로 남긴 채 재학습합니다. n = 0 / 20 / 50 / 80 / 100%, 모델은 pca_error·sensor_range, fold는 오염 다양성이 최대·최소인 F-STEP·F-DS 두 개입니다.
**왜** — 부분 라벨 환경에서 "라벨된 것을 지우는" 가장 단순한 라벨 활용법의 한계 곡선을 재기 위해서입니다 (§2의 Q2). 이 곡선이 MAE GRL이 넘어야 할 기준선이 됩니다.
:::

- 표 = **seen partition의 macro per-fault pak_auc_f1** (§3.1 표 ①·②와 같은 척도 — 0% 행 = 표 ①의 seen, 100% 행 = 표 ②의 seen과 일치), 괄호 = 잔류 오염 run 수

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
		<td color="orange">0.837</td>
		<td color="orange">0.789</td>
		<td color="red">0.133</td>
		<td color="red">0.009</td>
	</tr>
	<tr>
		<td>20% (48)</td>
		<td color="orange">0.818</td>
		<td color="orange">0.793</td>
		<td color="red">0.141</td>
		<td color="red">0.009</td>
	</tr>
	<tr>
		<td>50% (30)</td>
		<td color="green">0.924</td>
		<td color="orange">0.802</td>
		<td color="red">0.149</td>
		<td color="red">0.009</td>
	</tr>
	<tr>
		<td>80% (12)</td>
		<td color="green">0.974</td>
		<td color="orange">0.861</td>
		<td color="red">0.192</td>
		<td color="red">0.020</td>
	</tr>
	<tr>
		<td>100% (0)</td>
		<td color="green">**1.000**</td>
		<td color="green">**0.999**</td>
		<td color="green">**0.918**</td>
		<td color="green">**0.998**</td>
	</tr>
</table>

**읽는 예** — pca_error @ F-DS의 50% 행 "0.802": F-DS train의 fault 60 runs(IDV13 30개 + IDV14 30개) 중 절반(각 15 runs)의 라벨을 안다고 가정하고 그 절반을 학습에서 제거한 뒤 재학습한 결과입니다. 0%(아무것도 제거 안 함)의 0.789와 거의 같고 100%(전부 제거)의 0.999에는 한참 못 미칩니다 — **오염의 절반을 지워도 탐지력은 거의 회복되지 않습니다.**

:::callout {icon="⚠️" color="orange_bg"}
**핵심 4 — 회복 곡선이 심하게 볼록합니다. 이득이 마지막 구간에 몰려 있습니다.**
- 오염의 **80%를 이상적으로 제거해도** pca_error F-DS는 0.861 (clean 0.999에 한참 미달)
- sensor_range는 **100% 직전까지 무신호 수준** — F-DS 0.009~0.020, F-STEP 0.133~0.192 (random floor 0.48에도 한참 미달)
- → **잔류 오염 소수가 피해의 대부분**을 만들며, "라벨된 것을 제거"하는 전략은 부분 라벨 환경에서 거의 무가치
:::

**참고 — 같은 sweep의 unseen 점수** (오염 제거 효과가 seen에 국한됨을 직접 확인할 수 있습니다)

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
		<td color="green">0.930</td>
		<td color="green">0.950</td>
		<td color="orange">0.504</td>
		<td color="orange">0.437</td>
	</tr>
	<tr>
		<td>20% (48)</td>
		<td color="green">0.930</td>
		<td color="green">0.951</td>
		<td color="orange">0.505</td>
		<td color="orange">0.437</td>
	</tr>
	<tr>
		<td>50% (30)</td>
		<td color="green">0.944</td>
		<td color="green">0.952</td>
		<td color="orange">0.509</td>
		<td color="orange">0.438</td>
	</tr>
	<tr>
		<td>80% (12)</td>
		<td color="green">0.942</td>
		<td color="green">0.956</td>
		<td color="orange">0.513</td>
		<td color="orange">0.443</td>
	</tr>
	<tr>
		<td>100% (0)</td>
		<td color="green">**0.992**</td>
		<td color="green">**0.994**</td>
		<td color="orange">**0.783**</td>
		<td color="orange">**0.808**</td>
	</tr>
</table>

**읽는 예** — pca_error @ F-DS의 unseen 열이 0%에서 80%까지 0.950→0.956으로 거의 움직이지 않는 이유: 제거되는 오염이 drift·sticking 유형이므로, 다른 유형(unseen)의 탐지에는 애초에 영향이 작았기 때문입니다.

- unseen은 전 구간 거의 평탄하다가 100% 지점에서만 소폭 상승 (pca: 0.930→0.992 / 0.950→0.994) → **sweep의 회복 곡선은 seen 전용 현상**

- 볼록한 모양의 원인 — **pca_error**: 잔류 fault run 12개만으로도 그 방향이 부분공간에 흡수됨 / **sensor_range**: 단 하나의 unlabeled run이 범위를 넓혀 무력화.

### 3.4 IDV 3/9/15 — 3단계 검증

:::callout {icon="🧪" color="gray_bg"}
**이 검증은** — headline에서 제외한 고난도 fault 3종(IDV 3/9/15)이 "정말 탐지 불가능한지"를 모델 학습 없이 데이터 자체에서 확인합니다. 각 fault의 test run 20개와 정상 run 40개를 놓고, **이 둘을 구분할 정보가 데이터 어디에 남아 있는지**를 시간 단위를 바꿔 가며 세 수준에서 찾습니다.
**왜** — headline 제외 규칙은 "시점 단위" 통계 기준이었습니다. 시점 단위로 안 보이는 fault도 긴 시간 창으로 요약하면 보일 수 있고, 그렇다면 window 길이 500을 쓰는 MAE에게는 잡을 기회가 있습니다 (§2의 Q3).
:::

**세 수준(L1/L2/L3)의 정의** — 위로 갈수록 더 긴 시간을 요약합니다. 모든 수치는 AUC이며, 0.5 = 동전 던지기와 같음, 1.0 = 완벽 구분입니다.

- **L1 — 시점 하나, 센서 하나**: "지금 이 순간의 측정값 하나만 보고 이상 시점인지 알 수 있는가?" 52개 센서 각각에 대해 시점별 값으로 faulty/정상 시점을 가르는 AUC를 재고, **가장 잘 구분하는 센서**의 값을 적습니다. point-wise 탐지기가 쓸 수 있는 정보의 상한입니다.
- **L2 — run 하나(800 시점)를 숫자 하나로 요약**: "오래 지켜본 뒤 요약하면 구분되는가?" 각 run에서 센서별로 800 시점의 평균(run-mean)과 출렁임의 크기(run-std)를 계산해 run당 숫자 하나로 만들고, 그 숫자로 faulty run 20개 vs 정상 run 40개를 가르는 AUC를 잽니다. **가장 잘 구분하는 (센서 × 요약 방식)**의 값을 적습니다. 시간 맥락을 쓰는 window 모델이 도달할 수 있는 상한의 근사입니다.
- **L3 — 실제 모델의 anomaly score**: clean train으로 학습한 simple 모델의 점수로 구분되는가 (표는 pca_error의 ROC를 대표로 표기).
- 참조선: IDV1 (가장 쉬운 fault) · IDV16/19 (usable 중 신호가 가장 약한 fault)

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

**세 fault를 말로 풀면:**

- **IDV3 = D feed 온도의 계단형 상승.** 온도가 오르면 제어 루프가 냉각을 늘려 보상하므로, 매 순간의 센서 값은 거의 정상 범위에 머뭅니다 — 그래서 L1이 0.568(동전 던지기 수준)입니다. 그러나 보상이 완벽하지 않아 반응기 냉각수 출구 온도(xmeas_21)에 아주 작은 **지속 편차**가 남습니다. 한 시점에서는 잡음에 묻혀 안 보이지만, 800 시점을 평균 내면 잡음은 상쇄되고 편차만 남아 faulty run 20개와 정상 run 40개가 **완벽히 갈립니다 (L2 = 1.000)**. 실제 point-wise 모델은 L1의 예고대로 못 잡습니다 (L3 = 0.510).
- **IDV9 = 같은 D feed 온도의 무작위 요동.** IDV3과 같은 변수가 흔들리지만 한 방향이 아니라 무작위라서, 아무리 평균을 내도 0 근처로 상쇄되어 남는 신호가 없습니다. L2가 0.740으로 보이지만 아래 ※의 보정을 보면 우연으로도 나오는 수준입니다 → **어느 수준에서도 신호가 없는, 셋 중 유일한 진짜 불가능 케이스.**
- **IDV15 = 응축기 냉각수 밸브 sticking.** 밸브가 들러붙으면 제어 동작이 거칠어져, 값의 **평균은 그대로인데 출렁임(std)이 커집니다**. 그래서 run-mean이 아니라 run-std(xmeas_22)로 0.967에 분리됩니다 — 신호가 "평균"이 아니라 "변동성"에 있는 경우입니다.

요약하면, "IDV 3/9/15 = 탐지 불가"는 셋을 뭉뚱그린 표현이었습니다. 실제로는 **IDV3·15는 시점 단위에서만 불가능하고 시간 요약에는 신호가 남아 있으며, IDV9만 모든 수준에서 불가능**합니다.

- L3는 pca_error를 대표로 표기했지만 **5모델 전부 동일한 결과**입니다 — IDV3/9/15에서 ROC 0.500~0.532 (random 0.502, sensor 0.500~0.501, pca 0.510~0.513, l2 0.516~0.523, nn 0.518~0.532; 전부 random 수준)

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

:::callout {icon="🧪" color="gray_bg"}
**이 점검은** — 탐지 능력이 0인 random baseline의 점수를 평가 단위별(full stream vs per-fault)로 비교합니다.
**왜** — PA%K 지표의 바닥값이 stream 구성에 따라 달라지므로, 이 페이지의 모든 표를 읽는 눈금을 제공하기 위해서입니다.
:::

- **PA%K의 floor는 상수가 아니라 stream 구성의 함수입니다** — 같은 random이 full stream에서 **0.765** (positive rate 75.8%, region 길이 800), per-fault 평가에서는 **0.48** (positive rate 29.4%)
- **이 stream에서 pak_auc_f1 0.76 이하 = 사실상 random 이하**
- excluded-hard에서 pca_error의 0.79도 prc_auc 0.51 (≈ positive rate 50%)과 함께 보면 **사실상 무신호**
- → **MAE 결과표에는 random 행과 prc_auc 병기가 필수**

---

## 4. 해석

**한 줄 요약 — label-blind 모델의 세계에 type-generalization 격차는 없습니다. 존재하는 것은 오염 격차뿐입니다.**

1. **난이도 가설 기각, 오염 가설 채택** — clean 학습에서 pca_error는 usable 17종 중 14종을 0.99+(최저 0.966)로 잡고 |G| < 0.010. 같은 모델이 오염 train에서는 **seen만 선택적으로 붕괴**. MAE 조건 B가 보일 행동의 예고편입니다.
2. **피해의 형태는 detector 기하가 결정** — 부분공간 방법은 전역 흡수, 인스턴스 방법은 국소 흡수. MAE는 전역형에 가까울 것 → **GRL purging = "전역적으로 흡수될 뻔한 fault 방향을 라벨로 도려내기"**로 정식화할 수 있습니다.
3. **부분 라벨의 진짜 병목 = 잔류 오염** — oracle cleaning은 80% 라벨에서도 피해의 절반가량밖에 회복하지 못함 (F-DS 0.789→0.861, clean 0.999). 라벨이 의미 있으려면 **라벨된 소수의 signature가 unlabeled 다수로 일반화**되어야 합니다.
4. **IDV3/15 = window 모델의 고유 영역** — point-wise 방법은 구조적으로 닿지 못하지만 시간 집계에는 신호가 남음. MAE가 여기서 잡으면 성능 우위가 아니라 **방법론 계층이 다르다는 증거**가 됩니다.

---

## 5. 인사이트

:::callout {icon="1️⃣" color="blue_bg"}
**raw seen/unseen 격차는 type-generalization의 증거가 아닙니다.**
- **설명** — seen/unseen 점수 차이를 그대로 읽으면 "이 모델은 새 유형에 강하다/약하다"로 해석하기 쉽습니다. 그러나 이번 실험은 그 차이의 대부분이 모델의 일반화 능력과 무관한 두 요인 — 평가 구성 차이와 train 오염 — 에서 온다는 것을 보였습니다.
- **근거 1 (평가 구성)** — 탐지 능력이 0인 random조차 stream 통째 평가에서는 F-DS에서 G = −0.160이 나옵니다. seen partition(fault 2종, anomaly 41.7%)과 unseen partition(15종, 73.5%)의 구성이 달라 지표의 바닥값 자체가 다르기 때문입니다. 평가 조건을 맞춘 macro per-fault로 바꾸면 이 격차는 −0.002로 사라집니다.
- **근거 2 (오염)** — 조건을 맞춘 뒤에도 pca_error는 F-DS에서 −0.161이 남는데, 이것의 원인이 "unseen이 어려워서"가 아님은 clean 학습이 증명합니다: 오염만 빼면 같은 모델의 G가 네 fold 전부 |G| < 0.010입니다. 즉 남은 격차는 seen 쪽이 오염으로 망가진 결과입니다.
- **MAE에 적용** — MAE 결과도 같은 함정에 빠질 수 있으므로, ① macro per-fault로 평가하고 ② 같은 fold의 label-blind 대조군(조건 B)과의 차이 **Ĝ = G_모델 − G_대조군**으로만 해석합니다.
:::

:::callout {icon="2️⃣" color="purple_bg"}
**라벨의 가치는 인스턴스 제거가 아니라 같은 유형 전체로 번지는 정화에 있습니다.**
- **설명** — 라벨의 가장 단순한 사용법은 "라벨 붙은 구간을 학습에서 빼는 것"입니다. sweep 실험은 이 방법이 완벽하게 수행된다는 이상적 가정(oracle cleaning) 아래에서조차 한계가 어디인지 쟀습니다.
- **근거** — F-DS에서 오염 60 runs 중 80%(48개)를 완벽히 제거해도 seen 점수는 0.789 → 0.861에 그칩니다 (clean 0.999까지 0.138이 남음). 잔류 12 runs만으로도 PCA가 drift 방향을 정상 부분공간에 흡수해 버리기 때문입니다. sensor_range는 더 극단적입니다 — 단 하나의 unlabeled fault run이 남아도 min·max 범위가 늘어나 0.009~0.020 수준에 머뭅니다.
- **MAE에 주는 기회** — GRL은 라벨된 run을 "빼는" 것이 아니라, 거기서 fault의 signature를 학습해 student 표현에서 지웁니다. 이 정화가 **라벨 없는 같은-유형 run에까지 번진다면**, MAE-A의 sweep 곡선은 위의 oracle cleaning 곡선을 넘어설 것입니다. 넘어서는 폭이 곧 "라벨이 자기 인스턴스 이상을 정화한다"는 증거이며, 이것이 MAE noisy-label 실험(설계 표기 P0-4)의 최우선 판별 축입니다.
:::

:::callout {icon="3️⃣" color="orange_bg"}
**PA%K 절대값은 stream 구성을 떠나면 의미가 없습니다.**
- **설명** — "0.79면 꽤 잘하는 것 아닌가?"라는 직관은 이 지표에서 통하지 않습니다. PA%K의 바닥값(아무 능력 없는 모델이 받는 점수)이 평가 데이터의 구성에 따라 크게 움직이기 때문입니다.
- **근거** — 완전히 같은 random 점수가 full stream 평가에서는 **0.765**를 받습니다 (test의 75.8%가 anomaly이고 region이 800 시점으로 길어 PA%K가 관대해지기 때문). 같은 점수를 per-fault 평가(anomaly 29.4%)로 재면 **0.48**입니다. 실례로 excluded-hard partition에서 pca_error가 받은 0.79는 언뜻 높아 보이지만, prc_auc가 0.51로 positive rate(50%)와 같다는 점까지 보면 **신호가 전혀 없는 점수**입니다.
- **MAE에 적용** — MAE 결과표에는 같은 구성에서 잰 random 행과 prc_auc를 반드시 병기해, 모든 절대값이 "그 표의 바닥값 대비 얼마나 높은지"로 읽히게 합니다.
:::

:::callout {icon="4️⃣" color="green_bg"}
**IDV3/15는 point-wise 방법론 전체의 공백 지대입니다.**
- **설명** — §3.4에서 확인했듯, 매 순간의 값은 정상처럼 보이지만 800 시점을 요약하면 드러나는 fault가 존재합니다. 이런 fault는 시점 단위로 동작하는 방법 전체가 구조적으로 잡을 수 없습니다.
- **근거** — IDV3: 시점 단위 best AUC 0.568(동전 던지기) ↔ run-평균(xmeas_21) AUC **1.000**(완벽 분리). IDV15: 0.515 ↔ run-std(xmeas_22) **0.967**. 실제로 simple 5종 전부가 이 두 fault에서 ROC 0.50~0.53으로 전멸했습니다. 반면 IDV9는 run 요약으로도 0.740(우연 수준)이라 진짜 불가능 케이스입니다.
- **MAE에 주는 기회** — window 500을 한 번에 보는 MAE가 IDV3/15에서 ROC 0.5를 유의하게 넘으면, 그것은 점수 경쟁이 아니라 **"window 맥락이 본질적으로 필요한 이상을 잡는다"는 독립적 증거**가 됩니다. IDV9는 신호가 없어야 정상인 대조 케이스로 함께 보고합니다.
:::

:::callout {icon="5️⃣" color="yellow_bg"}
**TEP의 변별력은 clean 조건이 아니라 contaminated 조건에 있습니다.**
- **설명** — clean train에서는 PCA 같은 단순 방법조차 usable 17종 중 14종을 0.99 이상으로 잡습니다 (표 ②, 최저도 0.966). 이 조건에서 MAE가 높은 점수를 받아도 아무것도 증명하지 못합니다 — 모두가 만점인 시험이기 때문입니다.
- **근거** — 점수가 갈라지는 곳은 오염이 있을 때입니다: 같은 fault를 두고 pca_error는 0.074~0.209를 잃고, sensor_range는 0.526~0.989를 잃고, nn_distance는 F-DS에서 0.036만 잃습니다. 즉 **모델 간 차이는 "얼마나 잘 잡나"가 아니라 "오염에 얼마나 버티나"에서 생깁니다.**
- **MAE에 적용** — 본 실험의 비교 축은 ① 오염 내성(조건 B vs simple) ② 라벨 회복(조건 A가 C_dmg를 얼마나 되돌리나) ③ 부분 라벨 유지(sweep 곡선)의 세 가지입니다. 단, supervised skyline(조건 C)마저 unseen에서 무너지지 않으면 이 벤치마크 자체가 가설을 가르지 못하므로, **skyline부터 확인하는 사전 등록 게이트(Gate 0)**가 해석의 출발점입니다.
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
- **읽는 법** — 각 행은 MAE 본 실험의 조건 하나이고, H1/H2 열은 그 조건에서 두 가설이 각각 예측하는 결과입니다. 실측이 어느 열과 맞는지로 가설을 판정합니다

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

stream 전체를 한 덩어리로 평가한 micro 방식의 실제 점수와 격차입니다. §3.1에서 확인한 구성 artifact가 섞여 있어 **partition 간 비교나 가설 판정에는 사용하지 않습니다.**

<table fit-page-width="true" header-row="true" header-column="true">
	<tr>
		<td>모델 (cell = seen / unseen (G))</td>
		<td>F-STEP</td>
		<td>F-RAND</td>
		<td>F-DS</td>
		<td>F-UNK</td>
	</tr>
	<tr>
		<td>random</td>
		<td>0.713 / 0.746  (G -0.033)</td>
		<td>0.681 / 0.752  (G -0.070)</td>
		<td>0.597 / 0.757  (G -0.160)</td>
		<td>0.701 / 0.749  (G -0.049)</td>
	</tr>
	<tr>
		<td>sensor_range</td>
		<td>0.184 / 0.616  (G -0.432)</td>
		<td>0.011 / 0.455  (G -0.444)</td>
		<td>0.009 / 0.582  (G -0.573)</td>
		<td>0.133 / 0.394  (G -0.261)</td>
	</tr>
	<tr>
		<td>pca_error</td>
		<td>0.874 / 0.925  (G -0.052)</td>
		<td>0.914 / 0.939  (G -0.025)</td>
		<td>0.747 / 0.951  (G -0.204)</td>
		<td>0.858 / 0.974  (G -0.116)</td>
	</tr>
	<tr>
		<td>l2_norm</td>
		<td>0.870 / 0.892  (G -0.022)</td>
		<td>0.834 / 0.902  (G -0.068)</td>
		<td>0.786 / 0.907  (G -0.121)</td>
		<td>0.833 / 0.906  (G -0.073)</td>
	</tr>
	<tr>
		<td>nn_distance</td>
		<td>0.877 / 0.928  (G -0.051)</td>
		<td>0.884 / 0.919  (G -0.035)</td>
		<td>0.962 / 0.931  (G +0.030)</td>
		<td>0.885 / 0.920  (G -0.035)</td>
	</tr>
</table>

**읽는 예** — random × F-DS의 "0.597 / 0.757 (G −0.160)": 탐지 능력이 0인 random조차 stream을 통째로 평가하면 16%p의 격차가 나옵니다. seen partition(fault 2종)과 unseen partition(15종)의 anomaly 비율이 41.7% vs 73.5%로 달라 F1 계열 지표의 바닥값 자체가 다르기 때문입니다. 이것이 micro를 판정에 쓰지 않는 이유입니다.

- train을 보지 않는 random조차 −0.03~−0.16의 격차 · l2_norm F-STEP은 macro와 부호 반대

---

## 부록 B. per-fault 상세 점수 (독립 해석용 원자료)

fault 하나하나의 pak_auc_f1입니다 (각 fault 20 runs + 동일 FF 40 runs, positive rate 29.4% 고정 — 모든 칸이 직접 비교 가능). 왼쪽 5열 = clean train의 5모델, 오른쪽 4열 = contaminated train의 pca_error (★ = 그 fold에서 seen, 즉 train에 오염으로 들어간 fault).

<table fit-page-width="true" header-row="true" header-column="true">
	<tr>
		<td>Fault (family)</td>
		<td>random@clean</td>
		<td>sensor@clean</td>
		<td>pca@clean</td>
		<td>l2@clean</td>
		<td>nn@clean</td>
		<td>pca@F-STEP</td>
		<td>pca@F-RAND</td>
		<td>pca@F-DS</td>
		<td>pca@F-UNK</td>
	</tr>
	<tr>
		<td>IDV1 (Step)</td>
		<td>0.481</td>
		<td>1.000</td>
		<td>1.000</td>
		<td>1.000</td>
		<td>1.000</td>
		<td>0.749 ★</td>
		<td>1.000</td>
		<td>1.000</td>
		<td>1.000</td>
	</tr>
	<tr>
		<td>IDV2 (Step)</td>
		<td>0.485</td>
		<td>1.000</td>
		<td>1.000</td>
		<td>1.000</td>
		<td>1.000</td>
		<td>0.877 ★</td>
		<td>0.998</td>
		<td>1.000</td>
		<td>1.000</td>
	</tr>
	<tr>
		<td>IDV3 (EXCL-HARD)</td>
		<td>0.483</td>
		<td>0.005</td>
		<td>0.609</td>
		<td>0.616</td>
		<td>0.604</td>
		<td>0.604</td>
		<td>0.619</td>
		<td>0.614</td>
		<td>0.615</td>
	</tr>
	<tr>
		<td>IDV4 (Step)</td>
		<td>0.480</td>
		<td>1.000</td>
		<td>1.000</td>
		<td>0.846</td>
		<td>0.997</td>
		<td>0.619 ★</td>
		<td>0.790</td>
		<td>1.000</td>
		<td>0.997</td>
	</tr>
	<tr>
		<td>IDV5 (Step)</td>
		<td>0.482</td>
		<td>0.508</td>
		<td>1.000</td>
		<td>0.739</td>
		<td>0.767</td>
		<td>0.781 ★</td>
		<td>0.987</td>
		<td>1.000</td>
		<td>0.849</td>
	</tr>
	<tr>
		<td>IDV6 (Step)</td>
		<td>0.479</td>
		<td>1.000</td>
		<td>1.000</td>
		<td>1.000</td>
		<td>1.000</td>
		<td>0.999 ★</td>
		<td>1.000</td>
		<td>1.000</td>
		<td>1.000</td>
	</tr>
	<tr>
		<td>IDV7 (Step)</td>
		<td>0.481</td>
		<td>1.000</td>
		<td>1.000</td>
		<td>0.996</td>
		<td>1.000</td>
		<td>1.000 ★</td>
		<td>1.000</td>
		<td>1.000</td>
		<td>1.000</td>
	</tr>
	<tr>
		<td>IDV8 (Random)</td>
		<td>0.484</td>
		<td>0.998</td>
		<td>0.998</td>
		<td>0.986</td>
		<td>0.999</td>
		<td>0.997</td>
		<td>0.974 ★</td>
		<td>0.997</td>
		<td>0.998</td>
	</tr>
	<tr>
		<td>IDV9 (EXCL-HARD)</td>
		<td>0.483</td>
		<td>0.004</td>
		<td>0.612</td>
		<td>0.619</td>
		<td>0.606</td>
		<td>0.606</td>
		<td>0.619</td>
		<td>0.615</td>
		<td>0.617</td>
	</tr>
	<tr>
		<td>IDV10 (Random)</td>
		<td>0.483</td>
		<td>0.556</td>
		<td>0.979</td>
		<td>0.664</td>
		<td>0.847</td>
		<td>0.976</td>
		<td>0.967 ★</td>
		<td>0.942</td>
		<td>0.968</td>
	</tr>
	<tr>
		<td>IDV11 (Random)</td>
		<td>0.481</td>
		<td>0.922</td>
		<td>0.996</td>
		<td>0.707</td>
		<td>0.947</td>
		<td>0.722</td>
		<td>0.741 ★</td>
		<td>0.955</td>
		<td>0.858</td>
	</tr>
	<tr>
		<td>IDV12 (Random)</td>
		<td>0.486</td>
		<td>0.999</td>
		<td>1.000</td>
		<td>0.986</td>
		<td>1.000</td>
		<td>1.000</td>
		<td>0.994 ★</td>
		<td>0.999</td>
		<td>1.000</td>
	</tr>
	<tr>
		<td>IDV13 (Drift·Stick)</td>
		<td>0.483</td>
		<td>0.997</td>
		<td>0.997</td>
		<td>0.991</td>
		<td>0.997</td>
		<td>0.997</td>
		<td>0.996</td>
		<td>0.962 ★</td>
		<td>0.996</td>
	</tr>
	<tr>
		<td>IDV14 (Drift·Stick)</td>
		<td>0.478</td>
		<td>1.000</td>
		<td>1.000</td>
		<td>0.979</td>
		<td>1.000</td>
		<td>1.000</td>
		<td>1.000</td>
		<td>0.617 ★</td>
		<td>0.992</td>
	</tr>
	<tr>
		<td>IDV15 (EXCL-HARD)</td>
		<td>0.483</td>
		<td>0.010</td>
		<td>0.609</td>
		<td>0.620</td>
		<td>0.615</td>
		<td>0.613</td>
		<td>0.622</td>
		<td>0.623</td>
		<td>0.618</td>
	</tr>
	<tr>
		<td>IDV16 (Unknown)</td>
		<td>0.486</td>
		<td>0.246</td>
		<td>0.986</td>
		<td>0.663</td>
		<td>0.791</td>
		<td>0.983</td>
		<td>0.956</td>
		<td>0.962</td>
		<td>0.956 ★</td>
	</tr>
	<tr>
		<td>IDV17 (Unknown)</td>
		<td>0.484</td>
		<td>0.989</td>
		<td>0.999</td>
		<td>0.894</td>
		<td>0.990</td>
		<td>0.998</td>
		<td>0.998</td>
		<td>0.871</td>
		<td>0.754 ★</td>
	</tr>
	<tr>
		<td>IDV18 (Unknown)</td>
		<td>0.483</td>
		<td>0.997</td>
		<td>0.998</td>
		<td>0.995</td>
		<td>0.997</td>
		<td>0.998</td>
		<td>0.998</td>
		<td>0.998</td>
		<td>0.996 ★</td>
	</tr>
	<tr>
		<td>IDV19 (Unknown)</td>
		<td>0.478</td>
		<td>0.188</td>
		<td>0.993</td>
		<td>0.641</td>
		<td>0.786</td>
		<td>0.646</td>
		<td>0.642</td>
		<td>0.626</td>
		<td>0.618 ★</td>
	</tr>
	<tr>
		<td>IDV20 (Unknown)</td>
		<td>0.483</td>
		<td>0.725</td>
		<td>0.966</td>
		<td>0.579</td>
		<td>0.870</td>
		<td>0.916</td>
		<td>0.900</td>
		<td>0.903</td>
		<td>0.897 ★</td>
	</tr>
</table>

**읽는 예** — IDV4 행: clean train의 pca는 1.000인데 pca@F-STEP은 0.619★입니다. IDV4가 train에 섞인(★) fold에서만 무너지고, IDV4가 unseen인 pca@F-DS에서는 1.000 그대로입니다 — 하락이 오염 탓임이 행 하나에서 확인됩니다.

이 표에서 직접 확인할 수 있는 것:

- **오염의 선택적 피해** — pca@F-STEP에서 ★(seen) fault만 하락 (IDV1 1.000→0.749, IDV4 1.000→0.619), 비슷하게 pca@F-DS의 IDV14 1.000→0.617
- **near-variable spillover** — pca@F-STEP의 IDV11(unseen)이 0.996→0.722로 동반 하락 (seen IDV4와 같은 물리 변수)
- **subtle-set의 근거** — clean에서도 sensor_range가 IDV5/10/16/19/20을 놓침 (0.188~0.725); pca는 다섯 종 모두 잡음 (0.966~1.000)
- **IDV3/9/15 (EXCL-HARD)** — 전 모델·전 조건에서 0.6 근방 = per-fault random floor(0.48)를 약간 웃도는 무신호 수준

---

**산출물 위치** — 검증 게이트·전체 표: `analysis_report.md` · 3단계 검증: `idv_hard_report.md` · sweep 20조건: `sweep/` · per-fault 분해: `per_fault_metrics.json` · 사전 등록 설계: `temp/tep_design/80_experiment_design_final.md` · 모든 수치는 기존 baseline 파이프라인과 동일한 평가 코드 `compute_full_metric_set`으로 산출
