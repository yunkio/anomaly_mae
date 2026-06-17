:::callout {icon="🎯" color="blue_bg"}
이 페이지는 TEP fault-type-disjoint 실험의 simple baseline 5종 사전 실행 결과를 보고합니다. 이 실험은 MAE(exp271)의 핵심 주장을 검증하기 위해 설계했습니다. 핵심 주장은 "라벨은 anomaly 유형을 외우는 데 쓰이지 않고 정상 모델을 정화하는 데만 쓰인다"입니다. GPU가 필요 없는 단순 방법 다섯 종을 label-blind 기준선으로 실행했습니다. 다섯 종은 random, sensor_range, pca_error, l2_norm, nn_distance입니다. 목적은 두 가지입니다. 파이프라인 전체가 올바르게 동작함을 확인하는 것, 그리고 MAE 결과를 해석할 때 기준이 될 floor와 ceiling을 미리 확보하는 것입니다.

핵심 발견은 두 가지입니다. 첫째, seen/unseen 성능 격차는 fault 자체의 난이도가 아니라 train에 섞인 오염이 만듭니다. 둘째, 라벨을 인스턴스 제거에만 쓰는 접근은 부분 라벨 환경에서 거의 효과가 없습니다. 두 발견 모두 **MAE의 GRL 기반 purging이 차별화될 지점**을 가리킵니다. GRL은 라벨된 이상이 정상 표현 학습에 스며들지 못하도록 gradient를 반전시키는 정화 메커니즘입니다.
:::

- **실험 번호**: #12 · **결과 디렉토리**: `temp/0610/TEP/results/12_20260610_211815_tep_typegen_simple/`
- **사전 등록 설계**: `temp/tep_design/80_experiment_design_final.md` (2026-06-10 동결) · **스크립트**: `temp/0610/TEP/` (기존 코드 무수정)
- **평가 코드**: 기존 baseline 비교 실험과 동일한 단일 경로 (`compute_full_metric_set`)

---

## 1. 진행한 실험

### 1.1 왜 TEP인가

SWaT, WaDi, PSM, SMD 등 기존 벤치마크의 train/test split에는 공통된 한계가 있습니다. train 라벨에 등장한 anomaly 유형이 test에 그대로 다시 나타난다는 점입니다. 따라서 "학습 때 본 적 없는 유형의 이상도 잡아내는가"라는 질문은 지금까지 한 번도 측정된 적이 없습니다. 이 질문이 바로 MAE 메커니즘 주장의 핵심입니다.

TEP, 곧 Tennessee Eastman Process는 도메인이 공인한 fault 분류 체계를 가진 유일한 표준 벤치마크입니다. Downs & Vogel이 정의한 fault IDV 1~20은 다섯 family로 나뉩니다. 다섯 family는 Step, Random variation, Slow drift, Sticking, Unknown입니다. 그래서 TEP에서만 fault family 단위로 train과 test의 이상 유형을 완전히 분리하는 type-disjoint split이 가능합니다.

### 1.2 데이터 구성

모든 구성은 사전 등록 설계에 따라 run 번호까지 결정론적으로 고정했습니다. 아래 표는 train 두 조건과 공유 test stream의 구성입니다.

<table fit-page-width="true" header-row="true">
	<tr>
		<td>구성</td>
		<td>내용</td>
		<td>규모</td>
	</tr>
	<tr>
		<td>Train (fold별 contaminated)</td>
		<td>FaultFree runs 1~240 + seen family faulty 60 runs. 라벨 양 등화: F-STEP 6종×10 / F-RAND 4종×15 / F-DS 2종×30 / F-UNK 5종×12</td>
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

train 오염은 4-fold 회전으로 주입합니다. 각 fold는 하나의 fault family만 seen으로 두는 hard 설정입니다. fold가 바뀌어도 test stream 자체는 같으므로 fold 간 직접 비교가 가능합니다.

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

IDV 3, 9, 15는 폐루프 제어가 외란을 보상해 버려 탐지가 본질적으로 어려운 fault입니다. 사전 등록한 정량 규칙(post-onset 평균 max|z|가 정상 변동의 2배 미만)에 따라 이 세 fault는 headline 집계에서 제외합니다. 다만 test stream에는 그대로 두어 별도의 평가 구획으로 보고합니다. 이렇게 제외된 난제 fault만 모은 구획을 excluded-hard partition이라 부릅니다.

### 1.3 실험 매트릭스

실험은 세 블록으로 구성됩니다. 아래 표는 각 블록의 조건과 목적입니다.

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
		<td>L1 point 수준 / L2 run 집계 수준 / L3 모델 score 수준의 3단 분리도 검사</td>
		<td>"구분 불가" 주장의 수준별 검증과 window 모델의 구조적 기회 탐색</td>
	</tr>
</table>

:::callout {icon="✅" color="green_bg"}
**검증 게이트 전 항목 PASS.** stream 크기, label 산술, partition 분리, score 무결성, sweep 끝점 일관성을 모두 확인했습니다. train 오염 비율도 16.67% 설계값과 정확히 일치합니다. 기존 loader의 onset off-by-one은 실측 기반 161로 정정했습니다. pca_error의 5-box smoothing은 run마다 재시작하며, 440개 run 전부에서 경계 무결성을 확인했습니다. 상세 항목은 `analysis_report.md` §1에 있습니다.
:::

측정 장치가 건전함을 확인했으니, 이 장치로 정확히 무엇을 재려 했는지부터 분명히 해 두겠습니다.

---

## 2. 실험 의도와 목적

이 실험은 그 자체가 목적이 아니라, MAE 본 실험의 결과를 해석할 좌표계를 미리 구축하는 사전 작업입니다. 본 실험은 다섯 조건으로 구성됩니다. A는 제안 방법, B는 label-blind 대조군, B0는 clean 정상 데이터만 쓰는 참조 조건입니다. C는 라벨을 전부 사용하는 supervised skyline이고, D는 weak baseline입니다. 이 좌표계가 답해야 할 질문은 세 가지입니다.

첫 번째 질문은 **seen/unseen 성능 격차의 원인**입니다. MAE가 unseen 유형에서 성능이 떨어졌을 때, 그 원인이 일반화 실패인지, 원래 어려운 fault인지, train 오염의 부작용인지 구분해야 합니다. 그러려면 라벨을 전혀 쓰지 않는 모델의 격차를 먼저 알아야 합니다. simple 5종은 모두 label-blind이므로, 이들의 격차에는 난이도와 오염 효과만 남습니다. 다만 격차를 의미 있게 재려면 seen과 unseen을 같은 평가 조건에 놓아야 합니다. 그 등화 방법이 §3의 첫 번째 주제입니다.

두 번째 질문은 부분 라벨 환경에서 **"라벨은 곧 오염 제거"라는 전략의 한계**입니다. 실무에서는 이상의 일부만 라벨링됩니다. 이때 label-consuming 방법이 할 수 있는 가장 단순한 행동은 라벨된 인스턴스를 학습에서 제거하는 것입니다. 이 이상적 제거를 oracle cleaning이라 부릅니다. oracle cleaning의 성능 곡선은 MAE의 GRL purging이 넘어야 할 기준선이 됩니다. GRL이 의미를 가지려면, 라벨된 인스턴스에서 학습한 fault signature가 unlabeled 동일 유형 인스턴스까지 정화해야 하기 때문입니다.

세 번째 질문은 **"구분 불가" fault가 정말 모든 수준에서 구분 불가인가**입니다. IDV 3, 9, 15의 제외 규칙은 point 수준 통계로 정의했습니다. window 길이 500의 시간 맥락을 쓰는 MAE에게도 불가능한지는 별개의 질문입니다. 만약 시간 집계 수준에서 분리가 가능하다면, 그곳은 point-wise 방법 전체가 실패하는 지점에서 window 모델이 이기는 구조적 기회가 됩니다.

세 질문에 대한 답은 측정 조건을 등화하는 작업에서 시작합니다.

---

## 3. 결과 분석

### 3.1 측정 설계와 주 결과: macro per-fault G

주 지표는 pak_auc_f1입니다. pak_auc_f1은 PA%K 프로토콜에서 K를 0부터 100%까지 훑으며 얻은 F1 곡선의 면적입니다. region 단위의 부분 탐지를 신용하는 본 파이프라인의 표준 지표입니다. 격차 G는 seen 성능에서 unseen 성능을 뺀 값으로 정의합니다. 따라서 G가 음수라는 것은, 모델이 학습 때 접한 seen 유형을 오히려 더 못 잡는다는 뜻입니다.

격차를 재기 전에 측정 조건부터 따져야 합니다. stream 전체를 한 덩어리로 평가하는 방식을 micro, fault별 점수를 먼저 구해 그룹 평균하는 방식을 macro라 부르겠습니다. seen과 unseen은 fold마다 들어가는 fault 수가 달라, micro로 재면 positive rate가 서로 다릅니다. seen은 41.7~62.5%, unseen은 70.5~73.5%입니다. F1 계열 지표는 positive rate가 높을수록 관대해지므로, micro G에는 모델과 무관한 구성 artifact가 섞입니다.

그래서 주 지표는 macro per-fault G로 정했습니다. fault별 평가 구성은 모두 같습니다. 각 fault는 20개 run에 800 sample씩의 anomaly를 가지며, 동일한 FaultFree 40개 run과 묶어 평가합니다. 그 결과 positive rate가 29.4%로 고정되어, seen과 unseen이 완전히 같은 조건에서 비교됩니다.

이 등화가 옳았는지는 표가 스스로 검증합니다. random은 train을 보지 않으므로, 등화가 제대로 되었다면 G가 0이어야 합니다. 표의 첫 행이 바로 그 검증선입니다. 색은 절대값 기준으로 0.02 이하는 초록, 0.10 이하는 주황, 그 초과는 빨강이며 양수 G는 굵게 표시했습니다. 마지막 열은 clean 학습(ffonly)에서 같은 방식으로 계산한 G의 4-fold 범위입니다.

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
		<td>random (등화 검증선)</td>
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
**판독 1.** random의 macro G는 네 fold 모두 ±0.002 이내로 소멸합니다. 같은 random이 micro에서는 −0.03에서 −0.16까지 벌어졌으므로, micro 격차는 전액 구성 artifact였습니다. **등화의 필요성과 유효성**이 한 행에서 동시에 증명된 셈입니다. 왜곡은 부호까지 뒤집어서, l2_norm의 F-STEP G는 micro −0.022가 macro +0.045로 반전됩니다. 따라서 stream micro G는 보조 수치로 강등하고, 원표는 부록 A에 남깁니다.
:::

:::callout {icon="🔬" color="blue_bg"}
**판독 2.** 등화 후에도 살아남는 격차가 실제 오염 효과입니다. pca_error는 F-DS에서 −0.161을, sensor_range는 전 fold에서 −0.18~−0.43을 기록했습니다. seen 유형은 train에 unlabeled 오염으로 섞여 있습니다. label-blind 검출기는 그 패턴을 정상으로 학습하므로, 정확히 seen 쪽 탐지가 무너집니다. 결정적 대조로, 같은 pca_error가 오염 없이 학습하면 macro G는 네 fold 전부 절대값 0.009 이내입니다. 즉 usable 17종 사이의 순수한 난이도 격차는 거의 없으며, **격차는 오염이 만듭니다**.
:::

다만 "난이도 격차가 거의 없다"는 결론에는 단서가 하나 붙습니다. 검출력 자체가 약한 모델은 clean 조건에서도 G가 흔들립니다. sensor_range는 ffonly에서도 G가 −0.286에서 +0.190까지 출렁입니다. F-UNK fold의 −0.286은 subtle fault 16, 19, 20이 모두 seen에 몰린 결과입니다. 따라서 위 결론은 충분히 강한 검출기를 기준으로 한 진술입니다. 한편 nn_distance가 F-DS에서 기록한 +0.093이라는 유일한 양수의 정체는, 오염 피해를 직접 재 보면 드러납니다.

### 3.2 오염 피해의 정량화: C_dmg

오염이 깎는 성능의 크기는 C_dmg로 잽니다. C_dmg는 clean 학습의 seen 점수에서 contaminated 학습의 seen 점수를 뺀 값으로 정의하며, 값이 클수록 피해가 큽니다. partition 점수는 §3.1과 같은 per-fault macro로 계산하므로 표 전체가 단일한 척도를 공유하고, random 행은 여기서도 음성 대조입니다. 색은 0.05 미만 초록, 0.15 초과 빨강, 그 사이는 주황입니다.

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
		<td>random (음성 대조)</td>
		<td color="green">−0.001</td>
		<td color="green">+0.000</td>
		<td color="green">−0.001</td>
		<td color="green">+0.000</td>
		<td>train 무관, 피해 없음</td>
	</tr>
	<tr>
		<td>pca_error</td>
		<td color="red">0.163</td>
		<td color="orange">0.074</td>
		<td color="red">0.209</td>
		<td color="orange">0.144</td>
		<td>전역 흡수 (fault 방향이 부분공간에 잠식)</td>
	</tr>
	<tr>
		<td>l2_norm</td>
		<td color="orange">0.140</td>
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
		<td>국소 흡수 (점유 좌표 근방만 차폐)</td>
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
**판독 3.** 오염 피해의 형태는 detector의 기하에 따라 전역과 국소로 갈립니다. 부분공간 방법인 pca_error는 fault 방향을 정상 모델에 통째로 흡수하므로 같은 family 전체의 탐지가 약해집니다. 인스턴스 방법인 nn_distance는 오염 run이 점유한 좌표 근방만 가려져, run마다 궤적이 다른 F-DS에서 피해가 0.036에 그칩니다. §3.1의 +0.093이라는 양수 G는 이 국소성의 결과입니다. 표현학습 모델인 MAE는 전역 흡수형에 가까울 것이므로, **GRL purging이 회복해야 할 피해도 전역적**일 가능성이 높습니다.
:::

전역 흡수의 피해는 seen family 안에 머물지 않습니다. F-STEP fold에서 pca_error는 unseen인 IDV11마저 0.996에서 0.722로 끌어내렸습니다. IDV11이 오염된 seen fault와 같은 물리 변수를 공유하기 때문입니다. 이렇게 변수를 공유하는 이웃 fault까지 함께 무너지는 현상을 near-variable spillover라 부릅니다.

sensor_range의 피해는 흡수가 아니라 붕괴입니다. faulty run이 train의 최소·최대 범위를 넓히는 순간 탐지 메커니즘 자체가 무력화되어, C_dmg가 0.53~0.99에 이릅니다. 그렇다면 라벨로 오염의 일부를 제거할 때 피해가 얼마나 회복되는지가 다음 측정입니다.

### 3.3 Noisy-label sweep: 인스턴스 제거 전략의 한계 곡선

§2에서 정의한 oracle cleaning 조건으로 라벨의 한계 곡선을 쟀습니다. 라벨된 비율의 오염 run을 학습에서 이상적으로 제거하고, 나머지는 unlabeled 오염으로 남깁니다. 아래 표는 labeled 비율을 0%에서 100%까지 올렸을 때 seen partition의 pak_auc_f1입니다. 괄호 안 숫자는 학습에 남은 잔류 오염 run의 수입니다. 한 열 안에서는 평가 구성이 같으므로 stream 점수를 그대로 비교할 수 있습니다.

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
		<td color="red">0.010</td>
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
**판독 4.** 회복 곡선은 심하게 볼록합니다. 오염의 80%를 이상적으로 제거해도 F-DS의 pca_error는 0.836으로, clean의 0.999에 한참 못 미칩니다. sensor_range는 100% 직전까지 사실상 0에 머뭅니다. pca_error는 12개의 잔류 fault run만으로도 그 방향을 부분공간에 담아 버립니다. sensor_range는 단 하나의 unlabeled run이 범위를 넓히는 것만으로 무력화됩니다. 결론적으로 **잔류 오염 소수가 피해의 대부분**을 만들며, 라벨된 인스턴스를 제거하는 전략은 부분 라벨 환경에서 거의 가치가 없습니다.
:::

### 3.4 IDV 3, 9, 15의 3단 검증

마지막 블록은 §2의 세 번째 질문, 즉 "구분 불가" 주장의 수준별 검증입니다. L1은 point 수준의 단일 feature 분리도, L2는 run당 800 sample 집계 후의 분리도, L3는 simple 모델 score의 분리도입니다. 참조선으로 가장 쉬운 IDV1과, usable 중 가장 subtle한 IDV16/19를 함께 놓았습니다.

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
		<td>point 비식별 · 시간 집계 분리 가능</td>
	</tr>
	<tr>
		<td>IDV9 (D feed 온도 random)</td>
		<td color="red">0.514</td>
		<td color="red">0.740 (우연 수준※)</td>
		<td color="red">0.513</td>
		<td>완전 비식별</td>
	</tr>
	<tr>
		<td>IDV15 (응축기 밸브 sticking)</td>
		<td color="red">0.515</td>
		<td color="green">**0.967** (xmeas_22 run-std)</td>
		<td color="red">0.512</td>
		<td>point 비식별 · 시간 집계 분리 가능</td>
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

※ L2는 52개 feature와 2가지 집계, 총 104회 비교의 최대값이라 selection bias가 있습니다. run 수 20 대 40에서 우연 최대치는 0.75 수준입니다. 따라서 IDV9의 0.740은 잡음과 구분되지 않고, IDV3/15의 1.000과 0.967은 실제 신호입니다.

:::callout {icon="💡" color="yellow_bg"}
**판독 5.** fault 난이도는 네 계층으로 정밀화됩니다. 차례로 단일 feature point의 IDV1형, 상관구조 point의 IDV16/19형, 시간 집계 전용의 IDV3/15형, 완전 비식별의 IDV9형입니다. IDV16/19는 단일 feature AUC가 0.51 수준인데도 pca_error의 ROC가 0.966과 0.992에 이릅니다. IDV3/15는 point 수준에서 모든 simple 모델이 전멸하지만, 수백 sample의 평균과 표준편차에는 미세하고 지속적인 편이가 남습니다. 따라서 설계의 "어떤 방법으로도 비식별" 문구는 **"point-wise 비식별"로 한정**해야 정확합니다.
:::

### 3.5 평가 지표 calibration: random floor

마지막으로 지표 자체의 눈금을 짚어 둡니다. random의 점수는 stream 구성에 따라 크게 달라집니다. full test stream에서 random의 pak_auc_f1은 0.765에 이릅니다. positive rate가 75.8%로 높고 region이 800 sample로 길어, PA%K가 매우 관대하게 작동하기 때문입니다. 반면 positive rate 29.4%의 per-fault 평가에서는 같은 random이 0.48 수준에 머뭅니다. 즉 PA%K의 floor는 고정 상수가 아니라 stream 구성의 함수입니다.

따라서 이 stream에서 pak_auc_f1이 0.76을 밑돌면 사실상 random보다 못한 것입니다. excluded-hard partition에서 pca_error가 기록한 0.79도 마찬가지입니다. prc_auc 0.51이 positive rate 50%와 거의 같다는 사실을 함께 보면, 사실상 신호가 없는 것입니다. MAE 결과표에 random 행과 prc_auc를 반드시 병기해야 하는 이유입니다. 이상의 관찰을 하나의 인과 서사로 묶으면 다음과 같습니다.

---

## 4. 해석

이번 결과는 한 문장으로 요약됩니다. label-blind 모델의 세계에 type-generalization 격차는 존재하지 않으며, 존재하는 것은 **오염 격차**뿐입니다.

1. 데이터는 난이도 가설을 기각하고 오염 가설을 지지합니다. clean 학습에서 pca_error는 usable 17종 중 14종을 0.99 이상으로 잡았고, 나머지 세 종도 최저 0.966으로 잡았습니다. 이때 macro G는 네 fold 전부 절대값 0.009 이내였습니다. 즉 깨끗한 정상 모델만 있으면 거의 모든 fault가 잡히고, 순수한 난이도 격차는 사실상 없습니다. 반면 같은 모델이 오염된 train에서는 seen partition만 선택적으로 무너졌습니다. 오염으로 주입된 유형이 정확히 seen 유형이기 때문이며, 이 선택적 붕괴가 MAE 조건 B가 보일 기준 행동의 예고편입니다.

2. 오염 피해의 형태는 detector의 기하가 결정합니다. pca_error는 fault 방향을 부분공간에 흡수해 family 전체와 변수 공유 이웃까지 약화시켰습니다. 반면 nn_distance는 오염 run이 점유한 좌표 근방만 가려졌습니다. 표현학습 모델인 MAE는 전자에 가까울 것입니다. 따라서 GRL purging의 가치는 "전역적으로 흡수될 뻔한 fault 방향을 라벨로 도려내는 것"으로 정식화할 수 있습니다.

3. 부분 라벨 환경의 진짜 병목은 잔류 오염입니다. oracle cleaning은 라벨 80%에서도 피해의 절반을 회복하지 못했습니다. F-DS의 0.836은 clean의 0.999에서 한참 멀리 있습니다. 라벨의 가치를 인스턴스 제거로 한정하면 부분 라벨 환경에서는 구조적으로 패배합니다. 라벨이 의미를 가지려면, 라벨된 소수에서 학습한 fault signature가 unlabeled 다수로 일반화되어야 합니다.

4. IDV3와 15는 window 모델의 고유 영토입니다. 두 fault는 point-wise 방법 전체가 구조적으로 닿을 수 없지만, 시간 집계 통계에는 신호가 남아 있습니다. window 길이 500을 쓰는 MAE가 여기서 신호를 잡는다면, 그것은 단순한 성능 우위가 아니라 방법론 계층이 다르다는 증거가 됩니다.

이 해석에서 MAE 본 실험이 따라야 할 행동 원칙을 추리면 다섯 가지입니다.

---

## 5. 인사이트

:::callout {icon="1️⃣" color="blue_bg"}
raw seen/unseen 격차는 **type-generalization의 증거가 아닙니다**. label-blind 모델조차 격차를 보였고, 그 방향은 오히려 seen이 나쁜 쪽이었습니다. 비교가 의미를 가지려면 두 장치가 필요합니다. 하나는 평가 구성을 등화하는 macro per-fault 계산이고, 다른 하나는 같은 fold에서 matched control의 G를 뺀 보정값 Ĝ입니다. random의 micro G가 −0.16까지 내려갔다가 macro에서 소멸한 사실이 등화의 필요성을 보여 줍니다.
:::

:::callout {icon="2️⃣" color="purple_bg"}
라벨의 가치는 제거가 아니라 **일반화 정화**에 있습니다. oracle cleaning 곡선의 극단적 볼록성은 라벨된 인스턴스에만 작용하는 접근의 상한을 보여 줍니다. 우위가 생기려면 GRL이 라벨된 인스턴스의 signature를 학습해 unlabeled 동일 유형까지 정화해야 합니다. 이것이 MAE noisy-label 실험(설계 표기 P0-4)의 평가 축입니다.
:::

:::callout {icon="3️⃣" color="orange_bg"}
PA%K 절대값은 **stream 구성을 떠나면 의미가 없습니다**. 같은 random이 full stream에서는 0.765를, per-fault 평가에서는 0.48을 받습니다. floor가 positive rate와 region 길이의 함수이기 때문입니다. 모든 결과표에는 random 행과 prc_auc를 반드시 병기해야 합니다.
:::

:::callout {icon="4️⃣" color="green_bg"}
IDV3와 15는 **point-wise 방법론 전체의 공백 지대**입니다. 폐루프 제어가 point 분포를 가리지만, run 집계에는 AUC 1.000과 0.967의 신호가 남습니다. 두 fault는 window 맥락이 본질적으로 필요한 이상을 MAE가 잡는지 확인하는 천연 진단 케이스입니다. 어떤 수준에서도 분리되지 않는 IDV9는 negative control로 씁니다.
:::

:::callout {icon="5️⃣" color="yellow_bg"}
TEP의 분별력은 clean이 아니라 **contaminated 체제**에 있습니다. clean train에서는 simple 방법조차 거의 만점이므로, 그곳의 우위는 가설을 가르지 못합니다. 의미 있는 비교 축은 오염 내성, 라벨 회복, 부분 라벨 유지의 세 가지입니다. 같은 이유로, supervised skyline이 unseen에서 무너지는지를 먼저 확인하는 사전 등록 게이트, 즉 Gate 0의 중요성이 한층 커졌습니다.
:::

이 원칙들을 본 실험 프로토콜에는 다음과 같이 반영합니다.

---

## 6. MAE 실험에 어떻게 적용할 것인가

1. 본 실험을 MAE 본 실험의 anchor로 삼습니다. train/test stream의 run 번호, partition 정의, 평가 코드를 MAE 조건 A/B/B0가 그대로 물려받습니다. MAE 결과표의 simple baseline 행에는 이 결과를 옮겨 적기만 하면 됩니다.

2. seen/unseen 주 비교는 macro per-fault Ĝ로 수행합니다. fault별 평가 구성이 같아 비교 조건이 완전히 동일해지기 때문입니다. 이는 사전 등록 설계 §4.4(b)가 per-fault matched 분석을 co-primary로 둔 결정을 실증으로 승격시킵니다. stream micro 지표는 보조로 강등합니다.

3. MAE label sweep(설계 표기 P0-4)에 본 실험의 oracle cleaning floor 곡선을 같은 그래프에 병치합니다. 비교 대상은 MAE-A의 부분 라벨 조건, oracle cleaning simple, MAE-B, MAE-B0입니다. labeled run 선택 규칙(각 fault의 앞쪽 k runs)도 동일하게 적용해 point 단위 비교를 보장합니다.

4. excluded-hard의 서술을 정정합니다. "어떤 방법으로도 비식별"을 "point-wise 비식별"로 한정하고, IDV3/15는 diagnostic 행으로 분리해 따로 보고합니다. headline 집계에서의 제외는 사전 등록된 규칙 그대로 유지합니다.

5. subtle-set을 IDV {16, 19, 10, 5, 20}으로 동결해 discriminative sub-analysis에 사용합니다. 이 집합은 모델 결과를 보기 전에 데이터 통계만으로 확정했으므로, 설계 §2.2의 동결 조건을 충족합니다.

6. 보고 규율을 고정합니다. 모든 표에 random 행과 prc_auc를 병기하고, partition 간 raw 비교를 금지하며, per-run 경계 정책을 명시합니다.

이렇게 설계된 본 실험에서 두 가설이 각각 어떤 결과를 낳을지 미리 적어 둡니다.

---

## 7. MAE 실험에서 기대되는 결과

H1은 "라벨은 정상 모델의 정화에만 쓰인다"는 MAE의 핵심 가설입니다. H2는 "라벨은 seen 유형을 암기하는 implicit classifier를 만들 뿐"이라는 대립 가설입니다. 아래 표의 수치는 seen partition의 pak_auc_f1 기준 예상 범위이며, Ĝ는 모두 macro per-fault 기준입니다. 마지막 행의 diagnostic 축은 두 가설과 독립적인 보너스 검증입니다.

<table fit-page-width="true" header-row="true" header-column="true">
	<tr>
		<td>조건</td>
		<td>H1 (purging 성립) 예상</td>
		<td>H2 (implicit classifier) 예상</td>
		<td>판별 근거</td>
	</tr>
	<tr>
		<td>MAE-B (label-blind, contaminated)</td>
		<td>simple처럼 seen 잠식. 전역 흡수형이므로 pca_error의 macro C_dmg 0.07~0.21과 유사 또는 그 이상</td>
		<td>H1과 동일 (공통 기준선)</td>
		<td>여기서는 가설이 갈리지 않음</td>
	</tr>
	<tr>
		<td>MAE-A (full labels)</td>
		<td>seen이 B0 ceiling에 근접 (C_dmg 대부분 회복), unseen은 B 대비 무손상</td>
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
		<td>point-wise 전멸 지점에서 ROC가 0.5를 유의하게 상회하면 window 모델의 구조적 우위 입증</td>
		<td>가설과 독립적인 보너스 축. IDV9는 negative control로 0.5 근방이 정상</td>
		<td>L2 분석이 상한의 존재를 보증</td>
	</tr>
</table>

:::callout {icon="🚦" color="red_bg"}
**사전 경고 두 가지.** 첫째, clean 체제에서는 simple 방법도 거의 만점입니다. 따라서 supervised skyline이 unseen에서 무너지지 않으면, 이 벤치마크는 H1과 H2를 분별하지 못합니다. 그래서 본 실험의 해석은 skyline 결과를 확인하는 Gate 0에서 시작해야 합니다. 둘째, MAE-B의 행동은 simple과 다를 수 있습니다. 표현학습의 오염 흡수가 PCA보다 약하거나 강할 수 있으며, 그 자체가 중요한 발견입니다. 본 실험이 제공하는 floor 값은 예측이 아니라 **좌표**입니다. MAE 결과가 어느 위치에 떨어지든 해석할 수 있게 만드는 것, 그것이 이 사전 실험의 역할입니다.
:::

---

## 부록 A. stream micro G 원표

아래 표는 stream 전체를 한 덩어리로 평가한 micro G의 원표입니다. §3.1에서 확인했듯 micro G에는 partition 구성 artifact가 섞여 있으므로, 보조 기록으로만 남깁니다.

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

train을 보지 않는 random조차 −0.03~−0.16의 격차를 보이고, l2_norm의 F-STEP은 macro와 부호가 반대입니다. partition 간 비교나 가설 판정에는 이 표를 사용하지 않습니다.

---

실험과 분석의 산출물은 다음 위치에 있습니다. 검증 게이트와 전체 결과표는 `analysis_report.md`에 있습니다. 3단 검증은 `idv_hard_report.md`, sweep 20조건은 `sweep/`, per-fault 분해는 `per_fault_metrics.json`입니다. 사전 등록 설계는 `temp/tep_design/80_experiment_design_final.md`입니다. 모든 수치는 기존 baseline 파이프라인과 동일한 평가 코드 `compute_full_metric_set`으로 산출했습니다.
