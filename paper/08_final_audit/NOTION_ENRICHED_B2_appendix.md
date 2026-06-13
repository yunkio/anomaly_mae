---
phase: 8
agent: spec-enricher-B2
directives: [R3]
last_modified: 2026-06-11
scope: |
  Appendix 전부 + 잔여: TAB-A3 / Table A.4(부분) / TAB-A6 / TAB-A7 / TAB-A8 /
  TAB-B1 / TAB-B2 / TAB-B3 / TAB-B4 / FIG-B1 / ALG-C1 / TXT-001 / TXT-002 /
  권고 실험 R-PROBE + 부모 페이지용 개요 자료(OVERVIEW).
basis: |
  paper/08_final_audit/NOTION_PLACEHOLDER_SPECS.md (r2 — 실행 지침 전부 보존·계승),
  paper/03_blueprint/PAPER_BLUEPRINT.md §6.6–6.8·§8·§14–15,
  paper/01_research_understanding/RESEARCH_SYNTHESIS.md (r3),
  paper/07_latex/appendix_A.tex · appendix_B.tex · appendix_C.tex (확정 캡션·표 구조)
pages: 10 (placeholder) + 1 (OVERVIEW)
---

<!-- PAGE: TAB-A3+TABLE-A4 -->

> 💡 **이 페이지는 두 개의 셋업·통계 표를 묶는다.** 둘 다 학습이 필요 없는 추출·측정 작업이며, 부록 §A의 재현성 기반을 이룬다. 아래에 각각 전체 템플릿 차원으로 명세한다.

# TAB-A3 — 26개 baseline 하이퍼파라미터 전수 표 (Table A.3)

> 💡 26개 baseline 각각의 {Window, LR, Batch, Epochs, Key parameters}를 `comparison/baseline_common.py`의 `MODEL_CONFIGS`에서 그대로 덤프해 채우는 재현성 표. 학습 불필요, 값 발명 절대 금지.

| 항목 | 내용 |
|---|---|
| 위치 | 부록 §A.1, `appendix_A.tex`, `\label{tab:baseline_hparams}` |
| 분류 | `[재사용]` — 코드 상수 추출만 필요 |
| 단일 원천 | `comparison/baseline_common.py`의 `MODEL_CONFIGS` 딕셔너리 |

## 🎯 목적과 의도

이 표는 논문의 **재현성 주장을 떠받치는 기반 표**다. §4.1.2가 "각 baseline은 원 구현 또는 발표 preset의 설정을 유지한다"라고 선언하는데, 그 선언을 26개 방법 각각에 대해 검증 가능한 구체 수치로 펼쳐 보이는 곳이 바로 여기다. 리뷰어 방어 관점에서는 두 갈래 공격을 막는다. 첫째, "baseline을 불리하게 튜닝한 것 아닌가"라는 공정성 공격에 대해, 모든 방법이 자기 원 구현의 preset을 그대로 쓴다는 사실을 표 한 장으로 입증한다. 둘째, "통일 파이프라인과 원 구현 설정이 어디서 어떻게 다른가"라는 재현성 질문에 대해, window·epochs·batch의 이탈 항목을 명시적으로 나열함으로써 답한다. DAGMM의 simplified 표기는 "이 구현은 DAGMM이 아니다"라는 방법-재정의 공격(블루프린트 결정 ⑦)을 선제 차단하는 장치다.

## 🏁 목표와 기대 결과

성공 기준은 단 하나다: **26행의 모든 셀이 `MODEL_CONFIGS`의 실값 또는 "original preset" 표기로 채워지고, 발명된 값이 0개일 것.** 이 표는 성능 결과 표가 아니므로 기대하는 수치 패턴은 없다. 대신 정합성 기준 두 가지를 통과해야 한다. 첫째, Table A.2(`tab:budgets`)와의 모순이 없어야 한다 — 특히 baseline의 batch 열은 모델별로 32–512 범위에서 제각각이므로, A.2의 "model-specific" 서술과 일치해야 하고 특정 단일값(구판의 "512")을 인용해서는 안 된다. 둘째, Window·Epochs 열의 이미 확정된 실값(예: Anomaly Transformer 100/10, NRdetector 100/50, TranAD 10, USAD 12, GDN 15, DCdetector 105)과 새로 덤프한 값이 일치해야 한다. 불일치가 나오면 그것은 표가 아니라 `MODEL_CONFIGS`가 변경된 신호이므로, 변경 이력을 추적해 어느 쪽이 실험 당시 값인지 확인한 뒤 기재한다.

## 🧪 실험 내용과 설계

학습은 전혀 필요 없다. 작업은 1회성 추출 스크립트 하나다. `comparison/baseline_common.py`의 `MODEL_CONFIGS`를 import하여 26개 모델 항목의 {window, lr, batch, epochs, 핵심 모델 파라미터}를 표 형식으로 덤프한다. 잔여 placeholder는 LR·Batch·Key parameters 세 열이며, Window·Epochs 열은 이미 tex에 실값으로 확정되어 있다.

채움 규칙은 다음과 같다. `MODEL_CONFIGS`에 명시된 키는 그 값을 그대로 옮긴다. `MODEL_CONFIGS`에 없는 항목은 **"original preset"으로 표기하고 빈 칸으로 두지 않는다** — 어떤 값도 발명하지 않는다는 A8 원칙이 이 표에서는 "코드에 없는 값은 코드에 없다고 쓴다"로 구현된다. Key parameters 열은 각 방법의 변별적 파라미터(예: PCA 성분 수 50, NN-distance 이웃 수 5, random score 5-run 평균)를 1–2개 골라 기재하되, 역시 코드 등록 항목에서만 가져온다. DAGMM 행은 "DAGMM (simpl.)" 표기를 유지하고 Key parameters 셀에 "GMM omitted"를 남긴다 — TranAD 저장소의 simplified 재구현(GMM energy 항 생략)임을 캡션과 행 양쪽에서 일관되게 알린다.

## 📊 구성과 형태

booktabs 26행 × 6열 {Method, Window, LR, Batch, Epochs, Key parameters}. 행은 tex 확정 구조 그대로 4계층 그룹으로 나눈다: simple/lightweight 9종 → SOTA legacy 6종 → SOTA recent 7종 → weakly supervised 4종, 그룹 사이는 이탤릭 그룹 헤더 행. simple 5종(random score, sensor range, PCA, L2-norm, NN-distance)은 학습 개념이 없으므로 LR·Batch·Epochs가 "—"다. 형태 변경은 불필요하다 — 구조는 tex에서 이미 확정되었고 남은 일은 셀 채움뿐이다.

## 📝 캡션

```
Hyperparameters of all 26 baselines.
Each method retains the settings of its original implementation or publication preset;
deviations from the unified pipeline (window size, epochs, batch size) are listed explicitly.
DAGMM follows the simplified TranAD-repository re-implementation (GMM energy term omitted).
```

## ⚠️ 주의사항·의존성

Table A.2(budgets)와의 정합이 첫째 함정이다: baseline의 batch 열은 "model-specific (original presets)"이 정본이며, 구판 문서들이 인용하던 "512" 단일값을 어디에도 되살리면 안 된다(v2-r3 정정 사항). 둘째, 향후 baseline 큐 재실행 과정에서 어떤 모델의 preset이 바뀌면 **같은 커밋 pass에서 이 표를 갱신**해야 한다 — 표와 코드가 어긋난 채 제출되는 것이 최악의 실패 모드다. 셋째, tranad의 LR은 논문 텍스트(0.01)와 코드 `constants.py`(1e-4)가 다르다는 기존 미결 사안(RESEARCH_SYNTHESIS §⑥)이 있으므로, 이 표에는 **실제 실험이 사용한 코드 값**을 기재하고 필요 시 각주로 원 논문 값과의 차이를 밝힌다.

## 🔢 연결된 NUM 표

| NUM | 관계 |
|---|---|
| (없음) | 이 표에서 파생되는 inline NUM placeholder는 없다. 본문 §4.1.2의 budgets 서술과 정합 의무만 진다. |

---

# Table A.4 (부분) — Per-entity 데이터셋 통계: SMD per-machine 셀 (Table A.4)

> 💡 Table A.4에서 SMD 행의 {#Train, #Test, Train AR} 세 셀만 placeholder다. loader의 분할 산식(`//2`)을 그대로 재사용하는 1회성 측정 스크립트로 28개 machine 통계를 뽑아, 본문 Table 1의 SMD 위임 셀과 **같은 산출물로 동시에** 채운다.

| 항목 | 내용 |
|---|---|
| 위치 | 부록 §A.3, `appendix_A.tex`, `\label{tab:per_entity}` — SMD 행 3종 셀만 잔여 |
| 분류 | `[신규 측정]` — 학습 불필요, 스크립트 1회 |
| 동일 소스 | 본문 TAB-1(Table 1)의 SMD per-machine 위임 셀과 **단일 산출물 공유** |

## 🎯 목적과 의도

Table A.4는 본문 Table 1의 family 요약을 entity 단위로 펼친 표로, 오염 벤치마크 프로토콜의 분할 결과를 entity별로 투명하게 공개하는 재현성 장치다. 나머지 행(SWaT, WaDi A1/A2, PSM, SMAP, MSL)은 전부 실값 확정 상태이고, SMD 행만 "[per-machine]" 위임 셀로 남아 있다. 이 셀을 채우는 일은 단순한 빈칸 메우기가 아니라, §4.1.1 본문이 들고 있는 "Training anomaly ratios range from 0.52% to 6.20% (SMD per-machine values pending…)" 문장의 **pending 꼬리를 해소하는 유일한 경로**다. 프로토콜 방어(블루프린트 §14) 관점에서도, 분할 규칙이 전 데이터셋 단일 산식임을 보이는 논거 ④(통일성)의 증거 표 역할을 한다.

## 🏁 목표와 기대 결과

성공 기준: SMD 28개 machine 각각의 #Train, #Test, Train AR이 산출되어 표(또는 요약 행)에 들어가고, 캡션의 "SMD per-machine rows pending" 이탤릭 문구가 **채움과 동시에 삭제**되는 것. 기대 패턴은 다음과 같다 — SMD machine별 Train AR이 기존 본문 범위 0.52–6.20% 안에 들어오면 §4.1.1 문장은 괄호 절만 떼어내면 된다. 만약 어떤 machine의 Train AR이 이 범위를 벗어나면(더 낮거나 높으면) **범위 수치 자체를 같은 pass에서 수정**해야 한다 — 부분 수정은 금지이며, Table 1·Table A.4·§4.1.1 본문이 한 번에 움직여야 한다. Test AR의 28-machine 평균은 이미 확정된 4.16과 일치해야 한다(불일치 시 산식 검증으로 회귀).

## 🧪 실험 내용과 설계

1회성 측정 스크립트를 작성한다. 산출 규칙은 코드와 글자 단위로 동일해야 한다: `loaders.py:1152-1157`의 SMD 분할 — `test_split = len(test_data)//2`, train = 원본 train 전체 + 원본 test 앞 50%, test = 원본 test 뒤 50% — 을 **그대로 호출**하거나, 같은 산식을 라벨 파일 위에서 직접 재계산한다. 두 방식 중 loader 직접 호출이 안전하다(산식 복제 과정의 off-by-one을 원천 차단). machine별로 #Train = 분할 후 train 길이, #Test = 분할 후 test 길이, Train AR = train 구간 라벨 1 비율(%)을 산출한다. 산출물은 CSV 또는 JSON 한 개로 저장해 Table 1과 Table A.4 양쪽 채움 스크립트가 **같은 파일을 읽게** 한다 — 두 표 간 수치 불일치를 구조적으로 불가능하게 만드는 것이 핵심 설계다.

## 📊 구성과 형태

표 구조는 tex 확정 상태다: {Entity, #Train pts, #Test pts, #Dim., Train AR (%), Test AR (%), Source} 7열. SMD를 28행 전부 펼칠지, 한 행 요약(범위 표기) + 비고로 처리할지는 지면 판단 사안이다 — 펼치면 28행이 추가되므로 부록 분량과 교환 관계에 있다. 어느 쪽을 택하든 #Dim. 열의 "29–36"(per machine)은 유지하고, 펼치는 경우 machine별 실측 차원을 함께 기재할 수 있다(metadata `num_features` 실측 — 잔여 6개 machine은 완주 후 확인).

## 📝 캡션

```
Dataset statistics under the contaminated benchmark protocol, per entity.
Train/test sizes reflect the re-split of Section~\ref{sec:datasets}; Train AR\,/\,Test AR
denote the anomaly ratio of the training\,/\,evaluation portion.
SMAP and MSL sizes are concatenated per-channel totals.
\textit{SMD per-machine rows pending.}
```

채움 완료 시 마지막 문장 `\textit{SMD per-machine rows pending.}`을 삭제한다 — 이 삭제가 resolved 신호다.

## ⚠️ 주의사항·의존성

본문 TAB-1의 SMD 위임 셀과 **반드시 동일 산출물**을 사용한다(두 표 독립 계산 금지). §4.1.1의 Train AR 범위 문장 갱신을 같은 pass에 포함한다. #Dim 열의 단일 원천은 §4.1.1이며 부록 C의 Table C.1(`tab:dimensionality`)과 정합을 유지해야 한다 — SMD 차원 범위(29–36)는 잔여 6 machine 완주 후 범위가 바뀔 수 있음을 인지할 것(RESEARCH_SYNTHESIS §⑥ N5). 측정은 학습과 무관하므로 271canon 완주를 기다릴 필요가 없다 — 지금 바로 실행 가능한 항목이다.

## 🔢 연결된 NUM 표

| NUM | 관계 |
|---|---|
| (없음) | NUM 등록 항목은 없으나, §4.1.1 본문의 Train AR 범위 문구(비-NUM placeholder성 문장)와 Table 1 SMD 셀이 이 측정에 동시 의존한다. |

---

<!-- PAGE: TAB-A6+TAB-A7+TAB-A8 -->

> 💡 **이 페이지는 "TAB-2와 동일 소스" 결과 전수 표 3종을 묶는다.** 세 표 모두 신규 실험이 없거나(메트릭 키 추가 추출) TAB-2를 채우는 실행 묶음에서 자동으로 산출된다 — 공통 전제는 271canon 완주와 baseline 실행 묶음의 완료다. 아래에 각각 전체 템플릿 차원으로 명세한다.

# TAB-A6 — SWaT 이중 조건(full / excl22) 전수 결과 표 (Table A.6)

> 💡 27개 방법 × {full, excl22} 두 평가 조건 × 5지표 전수. 같은 학습 모델·같은 점수에서 평가 마스크만 바꾼 결과이므로, "excl22 기준이 자의적"이라는 공격에 대한 투명성 방어 표다.

| 항목 | 내용 |
|---|---|
| 위치 | 부록 §A.4, `appendix_A.tex`, `\label{tab:swat_dual}` |
| 분류 | `[완주 대기 + TAB-2와 동일 소스]` — 별도 실험 없음 |
| CSMAD 소스 | [271c]의 `SWaT/A1A2_full`·`SWaT/A1A2_excl22` 두 entity metadata |

## 🎯 목적과 의도

본문 Table 2의 SWaT 열은 excl22 조건만 보여준다. 이 선택이 자의적이지 않음을 입증하려면 full 조건 결과를 숨기지 않고 전부 공개해야 하며, 그 공개 장소가 이 표다. 방어하는 공격은 블루프린트 §15의 "SWaT excl22 기준이 자의적이다" 시나리오다: 방어 논리는 ① region #22가 test anomaly 질량의 83.75%를 차지하는 단일 거대 사건이라 full 조건은 사실상 "그 한 사건을 잡았는가"의 지표가 되어 변별력이 낮고, ② excl22 마스크는 모든 baseline에 동일하게 적용되며, ③ full 결과도 이 표에 전수 병기된다는 3단이다. 이 표는 그중 ③을 물리적으로 이행한다. 동시에 §A.4 본문의 region 정의(평가 구간 내 [2,869, 38,769), 35,900 timesteps)와 함께 excl22의 결정론적 식별 가능성을 보여준다.

## 🏁 목표와 기대 결과

성공 기준: 27행(CSMAD + 26 baseline) × 10셀(조건 2 × 지표 5)이 전부 채워지고, 강조 규칙이 본문 표와 통일되는 것. 기대 패턴은 명확하다 — **full 조건 수치가 excl22보다 전반적으로 크게 좋아 보이는 것이 정상이다** (CSMAD [271c] 실측: full `pak_auc_f1` 0.944 vs excl22 0.629). 이 대비 자체가 논증 재료다: 단일 거대 사건이 지표를 부풀린다는 §4.1.1 주장의 실측 증거이며, 캡션의 "같은 모델, 마스크만 차이" 서술이 그 해석 장치다. baseline들 역시 full에서 부풀려진 수치를 보일 것으로 기대된다. 만약 어떤 baseline이 excl22에서 오히려 더 좋다면 그 방법은 region 22를 놓치고 소형 사건들을 잡는다는 뜻이므로, 본문 해석에 쓸 수 있는 관찰이 된다(의무는 아님). CSMAD의 excl22 열 수치는 Table 2의 SWaT 열과 글자 단위로 일치해야 한다.

## 🧪 실험 내용과 설계

별도 실험이 없다. TAB-2를 채우는 실행 묶음에서 자동 산출된다. 갈래별 소스는 다음과 같다.

| 행 그룹 | 소스 | 작업 |
|---|---|---|
| CSMAD | `[재사용]` [271c]의 `SWaT/A1A2_full`·`SWaT/A1A2_excl22` 두 entity | metadata `metrics` dict에서 5지표 추출. **각자 독립 best epoch** — full은 `pak_auc_f1`, excl22는 `excl22_pak_auc_f1` 기준 (271_CONFIG_TRUTH §IV 운영 주의) |
| unsupervised 22종 | `[재사용 — CMP-Q3]` | comparison 파이프라인의 dual 조건 산출(`has_excl22`; 결과 디렉토리 `SWaT/A1A2_full`·`A1A2_excl22`)에서 추출 |
| weakly supervised 4종 | `[신규 실행 — TAB-2 ② 3항과 동일 run]` | weak 4종 Q1 GPU 실행이 완료되면 같은 dual 산출 구조에서 추출 — 이 표를 위한 추가 실행은 없다 |

5지표의 내부 키는 {`pak_auc_f1`, `pak_auc_prc_auc`, `vus_pr`, `vus_roc`, `affiliation_f1_ar`}이다. 모든 지표는 `compute_full_metric_set`이 같은 best epoch에서 일괄 산출하므로 추출만 하면 된다.

## 📊 구성과 형태

좌우 2블록 구조: {Full condition 5열 | excl22 condition 5열}, 블록 사이 공백 열 1개, 총 27행. 열 약칭은 tex 확정대로 {F1, PR, VUS-PR, VUS-ROC, Aff.}. 강조 규칙은 본문 표와 통일(열별 bold = 최고, underline = 2위)을 권장한다. `table*` 2단 폭 + `\footnotesize` + `tabcolsep` 3pt + max-width cap은 PDF QA에서 이미 확정된 레이아웃이므로 변경하지 않는다.

## 📝 캡션

```
SWaT dual-condition results: all five metrics for CSMAD and all baselines under the
full condition and the excl22 condition (Section~\ref{sec:datasets}).
Same trained models and identical scores in both conditions; only the evaluation mask differs.
The excl22 best epoch is selected independently under the shared criterion.
```

## ⚠️ 주의사항·의존성

Affiliation F1은 반드시 `_ar` 변형(`affiliation_f1_ar`)을 사용한다 — §4.1.3 본문 선언과의 R30 정합이며, F1-최적 threshold 변형은 ranking 비사용으로 선언되어 있다. CSMAD의 두 조건은 **독립 best epoch**임을 캡션 마지막 문장이 이미 공개하고 있으므로, 추출 시 full 조건의 best epoch에서 excl22 지표를 읽는 실수(`metrics_excl_region22` 혼용)를 하지 말 것 — excl22 열은 `A1A2_excl22` entity의 headline `metrics`에서 읽는다(두 값 모두 실존하므로 혼용이 가장 위험하다; RESEARCH_SYNTHESIS §④ α-m3 주석). 의존성: weak 4종 미완 시 해당 4행은 TAB-2의 sync 그룹 B fallback 규칙에 연동되어 함께 빠진다(부분 게재 금지).

## 🔢 연결된 NUM 표

| NUM | 관계 |
|---|---|
| (직접 파생 없음) | TAB-2 의존 사슬에 속하며, NUM-012(CSMAD @ SWaT excl22)와 같은 entity 소스를 공유한다(값의 단일 원천은 TAB-2 쪽). |

---

# TAB-A7 — 전 지표 전수 결과 표 (Table A.7)

> 💡 본문 Table 2가 보여주지 않는 나머지 4지표 {PA%K-AUC AUC-PR, VUS-ROC, Affiliation F1, PA F1(oracle)}를 27개 방법 × 7 데이터셋 열로 전수 공개. 신규 실험 0 — metadata에서 metric 키만 추가 추출.

| 항목 | 내용 |
|---|---|
| 위치 | 부록 §A.5, `appendix_A.tex`, `\label{tab:full_metrics}` |
| 분류 | `[완주 대기 — TAB-2와 동일 소스]` — 추가 실험·추가 비용 0 |
| 내부 키 | `pak_auc_prc_auc`, `vus_roc`, `affiliation_f1_ar`, `pa_0_f1` |

## 🎯 목적과 의도

본문 Table 2는 지면 제약상 PA%K-AUC F1과 VUS-PR 2지표로 고정되었다(블루프린트 RT V3 결정). 이 표는 "주 표 2지표 + 전수는 Appendix"라는 §4.1.3의 약속을 이행하는 곳이다. 논증 역할은 두 가지다. 첫째, **지표 선택 공격 방어**: "유리한 지표만 골라 보여준 것 아닌가"라는 공격에 대해 5지표 전수를 공개함으로써 답한다 — 특히 threshold-free 지표(VUS-ROC)와 사건 기반 지표(Affiliation F1)에서도 순위 구도가 유지되는지를 리뷰어가 직접 확인할 수 있게 한다. 둘째, **PA F1(oracle)의 비교 가능성 제공**: 선행 연구 다수가 PA F1을 보고하므로 비교 가능성을 위해 제시하되, oracle threshold 기반임을 명시하고 ranking에서 배제한다는 원칙(R29)을 표 차원에서 구현한다.

## 🏁 목표와 기대 결과

성공 기준: 27 method × 7 데이터셋 열 × 4지표 = 756셀이 모두 채워지고, PA F1 행 전부에 "(oracle)" 라벨이 붙는 것. 기대 패턴: 4지표 간 순위 구도가 본문 2지표와 대체로 일관되면 지표 선택 공격이 무력화된다. PA F1(oracle)은 F1-최적 threshold를 test 라벨로 고르는 지표 특성상 **다른 지표 대비 전반적으로 부풀려진 절대값**을 보일 것으로 기대되며, 이 부풀림 자체가 "oracle 지표를 ranking에 쓰지 않는다"는 §4.1.3 원칙의 시각적 정당화가 된다. 만약 특정 지표에서 순위가 크게 뒤집히는 데이터셋이 있으면 — 예컨대 Affiliation F1에서만 약한 방법 — 이는 본문에서 1문장 관찰로 다룰 수 있는 재료이지 숨길 일이 아니다.

## 🧪 실험 내용과 설계

**신규 실험이 전혀 없다.** TAB-2를 채우는 실행 묶음(271canon 완주분 + [CMP-Q3] 재사용분 + SMD/SMAP/MSL baseline 신규 실행 + weak 4종 신규 실행)의 experiment metadata에서 metric 키 4종만 추가로 추출한다. 단일 평가 루틴 `compute_full_metric_set`이 전 지표를 **같은 best epoch에서** 일괄 산출하므로, 이 표의 한 행과 Table 2의 같은 행은 동일 checkpoint·동일 epoch의 산출물이다 — 추가 비용이 0인 이유다. SMD/SMAP/MSL avg 셀의 집계 규칙은 TAB-2와 동일한 entity 집합·동일 macro 평균이어야 한다. 집계 스크립트는 TAB-2용과 공유하고 metric 키 목록만 늘리는 구현을 권장한다.

## 📊 구성과 형태

method × metric 중첩 행 구조(tex 확정): method당 4개 metric 행 {AUC-PR, VUS-ROC, Aff. F1, PA F1 (oracle)}이 세로로 이어진다. 열은 Table 2와 동일한 7 데이터셋 {SWaT excl22, WaDi A1, WaDi A2, PSM, SMD avg, SMAP avg, MSL avg}. **PA F1 행의 "(oracle)" 라벨은 의무**다 — 생략 시 unfair-threshold 공격이 확실시된다(RESEARCH_SYNTHESIS §④ oracle 표기 의무). 강조(bold/underline)는 지표 행별로 적용할지 생략할지 Phase 7 스타일 판단에 위임하되, 적용한다면 oracle 행은 강조에서 제외하는 쪽이 R29와 정합적이다.

## 📝 캡션

```
Complete multi-metric results for all methods and dataset families: PA\%K-AUC AUC-PR,
VUS-ROC, Affiliation F1, and PA F1 (oracle threshold; reported for comparability only, never
used for ranking --- Section~\ref{sec:metrics}).
PA\%K-AUC F1 and VUS-PR appear in Table~\ref{tab:main_results}.
```

## ⚠️ 주의사항·의존성

키 혼동이 단 하나의 치명 함정이다: PA F1은 F1-최적(oracle) threshold 기반 **`pa_0_f1`**이며, **`pa_0_f1_ar`이라는 키는 존재하지 않는다**(REQUEST-1 RESOLVED). 반면 Affiliation F1은 반대로 `_ar` 변형(`affiliation_f1_ar`)을 쓴다 — 두 지표의 threshold 계열이 서로 다르다는 점을 추출 스크립트에 주석으로 박아둘 것. ranking·본문 서술에 PA F1을 사용하는 것은 어떤 경우에도 금지다(R29). 의존성은 TAB-2와 완전 동일: 271canon 완주(잔여 SMD 6, SMAP 49, MSL 22), baseline SMD/SMAP/MSL 신규 실행, weak 4종 GPU 실행. weak 미완 시 그룹 6 행 4개가 sync 그룹 B 규칙으로 함께 빠진다.

## 🔢 연결된 NUM 표

| NUM | 관계 |
|---|---|
| (직접 파생 없음) | TAB-2 의존 사슬의 일부. 본문 §4.1.3의 "remaining three metrics are in Appendix" 약속 문장이 이 표의 존재에 의존한다. |

---

# TAB-A8 — CSMAD per-entity 전수 결과 표 (Table A.8)

> 💡 SMD 28 + SMAP 54 + MSL 27 = 109개 entity 각각의 {PA%K-AUC F1, VUS-PR}. 271canon 완주가 유일한 전제조건이며, 블록별 macro 평균이 Table 2의 family 열과 자리수까지 일치해야 한다.

| 항목 | 내용 |
|---|---|
| 위치 | 부록 §A.6, `appendix_A.tex`, `\label{tab:per_entity_results}` |
| 분류 | `[완주 대기]` — 신규 실험 없음, 완주 후 집계 스크립트 1회 |
| 소스 | [271c] entity별 `experiment_metadata.json`의 `metrics` dict |

## 🎯 목적과 의도

multi-entity family(SMD/SMAP/MSL)의 Table 2 셀은 macro 평균 한 숫자로 압축되어 있다. 평균은 entity 간 분산을 숨길 수 있으므로, "평균이 소수 entity의 고성능에 끌려간 것 아닌가"라는 집계 공격에 대한 방어는 entity 전수 공개뿐이다. 이 표가 그 공개를 수행한다. 동시에 재현성 장치이기도 하다: 재현 시도자가 특정 machine/channel에서 얻은 수치를 우리 결과와 entity 단위로 직접 대조할 수 있게 한다. 캡션의 "Macro-averages over entities equal the corresponding family columns of Table 2" 문장은 이 표와 본문 표 사이의 **수치 계약**이며, 그 계약의 검증 가능성 자체가 논증 자산이다.

## 🏁 목표와 기대 결과

성공 기준: 109행 × 2지표가 전부 [271c] metadata 실값으로 채워지고, 세 블록의 macro 평균이 Table 2의 SMD/SMAP/MSL avg 셀과 **소수 반올림 자리수까지** 일치하는 것. 기대 패턴: entity 간 성능 분산이 존재하는 것이 자연스럽다 — SMD machine들은 차원(29–36)과 anomaly 구성이 제각각이고 SMAP/MSL channel들은 신호 특성이 이질적이므로, 균일하게 높은 수치보다 분산 있는 분포가 오히려 신뢰할 만한 모양이다. 일부 entity에서 낮은 수치가 나오는 것은 숨길 일이 아니라 평균의 정직성을 보이는 재료다. 다른 패턴(예: 특정 family에서 소수 entity가 평균을 지배)이 관찰되면 본문 §4.2의 family 해석에 1문장 주의를 추가할지 검토한다.

## 🧪 실험 내용과 설계

신규 실험은 없다. 작업 순서는 다음과 같다. 첫째, 271canon 잔여 entity(SMD 6, SMAP 49, MSL 22 — 2026-06-11 실측)의 완주를 기다린다 — **이것이 유일한 전제조건**이다. 둘째, 집계 스크립트가 [271c] 디렉토리를 순회하며 entity별 `experiment_metadata.json`에서 `metrics.pak_auc_f1`과 `metrics.vus_pr`(둘 다 best epoch 기준 — `timing.best_epoch`에서 전 지표가 함께 추출된 값)을 읽어 109행을 생성한다. 셋째, 같은 스크립트 안에서 블록별 macro 평균을 계산해 **Table 2 채움 값과의 일치를 assert로 검증**한다 — 캡션의 계약 문장을 코드 수준에서 보장하라는 spec 권고를 그대로 계승한다. 자리수 처리는 "반올림 후 평균"이 아니라 "평균 후 반올림"으로 통일하고, Table 2 쪽 집계 스크립트와 같은 함수를 쓴다.

## 📊 구성과 형태

3 블록(SMD → SMAP → MSL) 세로 나열, 블록 사이 midrule, 각 블록 머리에 이탤릭 블록 헤더. 행 구조는 {Entity, PA%K-AUC F1, VUS-PR} 3열. entity 명명은 tex stub의 스타일을 따라 "SMD-1-1 / SMAP-A-1 / MSL-C-1" 형식으로 통일한다(내부 디렉토리명 그대로 노출 금지). 각 블록 말미 또는 캡션에 macro 평균 = Table 2 family 열 일치 보장 문구를 유지한다. 109행은 길지만 단일 column `table` 환경으로 처리 가능함이 tex에서 확인되었다(필요 시 2단 분할은 Phase 7 재판단).

## 📝 캡션

```
Per-entity results (PA\%K-AUC F1\,/\,VUS-PR) for SMD (28 machines), SMAP
(54 channels), and MSL (27 channels).
Macro-averages over entities equal the corresponding family columns of
Table~\ref{tab:main_results}.
```

## ⚠️ 주의사항·의존성

이 표의 평균과 Table 2 셀의 **수치 의존성은 단방향이 아니라 동일성**이다 — 두 표가 다른 스크립트로 따로 집계되는 순간 어긋날 위험이 생기므로, 단일 집계 산출물을 공유하라. 부분 완주 상태로 일부 entity만 채우고 나머지를 비워두는 게재는 금지다(TAB-2의 "부분 완주 avg 금지" 규칙이 이 표에는 행 단위로 적용된다). MSL은 27 channels로 표기한다 — SMAP/MSL 합산 81채널 기준의 다른 문서 수치와 혼동하지 말 것. 의존성: 271canon 완주(유일). baseline 실행과는 무관하므로 baseline 큐 지연이 이 표를 막지는 않는다.

## 🔢 연결된 NUM 표

| NUM | 관계 |
|---|---|
| (직접 파생 없음) | Table 2의 SMD/SMAP/MSL avg 셀(N-C 그룹의 입력)과 동일 산출물을 공유한다. sync 그룹 A(N-A, "six")의 성립 조건인 271canon 완주를 이 표가 함께 기다린다. |

---

<!-- PAGE: TAB-B1 -->

# TAB-B1 — Contaminated-training(무절제) 조건 비교 표 (Table B.1)

> 💡 22개 비지도 baseline을 절제 없는 오염 스트림 그대로 학습시킨 결과와, anomaly-excised 조건 대비 변화량 Δ를 공개한다. "절제 조건이 baseline에게 정말 최선이었는가"를 정량화해 학습량 비대칭 인정(R31)을 뒷받침하는 방어 표다.

| 항목 | 내용 |
|---|---|
| 위치 | 부록 §B.1, `appendix_B.tex`, `\label{tab:contaminated}` |
| 분류 | `[신규 실행]` — `comparison/run_baseline_queue.py`, variant `full`(Q1) |
| Δ 기준 | TAB-2 확정본의 anomaly-excised 수치 — **TAB-2 완성 후에만 Δ 산출 가능** |

## 🎯 목적과 의도

본문 비교(Table 2)는 비지도 baseline에게 anomaly-excised 조건 — 라벨로 오염 구간을 절제해 주는, 비지도 패러다임에서의 라벨 최선 활용(R12) — 을 제공한다. 그런데 절제는 학습 데이터의 양을 줄이므로, §4.1.4는 "excised 조건의 baseline이 CSMAD보다 적은 train 볼륨을 받는다"는 비대칭을 정직하게 인정한다(R31). 이 표는 그 인정 문장의 **정량 뒷받침**이다: 같은 22개 방법을 절제 없는 동일 오염 스트림(CSMAD가 받는 것과 동일한 train, 라벨 미사용)에서 학습시켜, 절제가 각 방법에게 실제로 이득이었는지(Δ<0이면 contaminated가 더 나쁨 = 절제가 이득)를 방법별로 보인다. 방어하는 공격은 두 갈래다. 첫째, "절제 때문에 baseline이 데이터를 덜 받아 불리했다"는 볼륨 공격 — Δ가 음수 일색이면 "데이터가 더 많아도 오염 때문에 더 나빴다"는 정면 반박이 된다. 둘째, §15의 "성능 우위가 프로토콜(데이터 추가) 때문 아닌가" 공격의 보조 방어 — 비지도 방법은 같은 추가 데이터를 받아도 라벨을 활용하지 못한다는 protocol-effect 논증(TAB-2 하단 블록·NUM-019)의 contaminated 측 실측을 이 실행 묶음이 공급한다.

## 🏁 목표와 기대 결과

성공 기준: 22개 baseline × 대표 3 family의 contaminated-training F1과 Δ가 채워지고, CSMAD 참조 행이 Table 2에서 복사되는 것. 기대 패턴: **대부분의 비지도 baseline에서 Δ < 0(절제 우세)** — 비지도 방법에게 train 내 anomaly는 순수 오염원이므로, 무절제 조건에서 정상 프로파일 학습이 교란되어 성능이 내려가는 것이 패러다임의 예측이다. 이 패턴이 확인되면 ① 오염이 실제 해를 끼친다는 문제 설정의 실재성, ② Table 2의 excised 조건이 baseline에게 유리한(관대한) 조건이었다는 비교 공정성이 동시에 입증된다. 다른 패턴의 해석: 특정 방법에서 Δ > 0(무절제 우세)이 나오면, 그 방법은 절제로 인한 데이터 손실·경계 단절의 비용이 오염 비용을 상회한 경우다 — 방법별 1문장 관찰로 다루되, 다수 방법이 Δ > 0이면 §4.1.4의 조건 정당화 서술 자체를 재조정해야 한다(침묵 게재 금지). CSMAD 행은 두 조건 모두 동일한 contaminated train이므로 Δ 정의상 "—"다.

## 🧪 실험 내용과 설계

실행은 baseline 비교 파이프라인 한 줄기다. `comparison/run_baseline_queue.py --queue <json>`으로 **22종 × 대표 3 family(SWaT, PSM, SMD) × variant `full`(Q1, contaminated-training)** 큐를 구성해 실행한다. Q1 항목은 각 baseline의 `experiment_configs.py`에 이미 등록되어 있으므로 신규 구현 없이 큐 구성만 하면 된다. 기존 Q1 결과 폴더 `1_20260312_*`는 per-entity 정규화(2026-06-02) 이전의 구버전이므로 재사용 금지 — 전량 재실행이다. SMD 실행 전에 **per-entity 정규화 적용을 반드시 확인**한다(STALE 원인 재발 방지; entity별 train 구간 scaler fit — `entity_norm_segments` 경유). 학습 budget·평가 cadence·best-epoch 기준(매 epoch eval, `pak_auc_f1`)은 main 프로토콜과 동일하게 유지한다 — 변하는 것은 train 데이터 구성(절제 없음)뿐이다.

Δ 산출은 실행과 분리된 후처리다: Δ = (contaminated F1) − (anomaly-excised F1), 양수 = contaminated 우세. 기준값은 **TAB-2 확정본**의 anomaly-excised 수치에서만 가져온다 — SMD avg 열의 기준값은 baseline SMD 신규 실행(실행 대시보드 #1)이 끝나야 존재하므로, 이 표의 완성은 TAB-2 unsupervised 행 완성 이후로 순서가 강제된다. 평가는 main과 동일한 held-out 평가 절반에서 수행되므로 조건 간 비교가 train 구성 차이만 분리한다.

## 📊 구성과 형태

23행(22 baseline + CSMAD 참조) × 6열: {SWaT excl22, PSM, SMD avg} × {F1, Δ}. Δ 열은 부호 명시(+/−). tex 확정 구조는 family당 `\cmidrule` 2열 블록이다. registry 원안(전 family × {F1, VUS-PR, Δ})에서 지면 축소된 형태가 tex에 확정되어 있으므로 **tex가 우선**이다. CSMAD 참조 행의 Δ 셀은 "—".

## 📝 캡션

```
Contaminated-training (no-excision) condition results for all 22 unsupervised
baselines. Each method trains on the identical contaminated training stream used by CSMAD
(no anomaly excision; labels unused) and is evaluated on the identical held-out evaluation
half. Metrics: PA\%K-AUC F1 and VUS-PR per dataset family; $\Delta$ columns give the change
relative to the anomaly-excised condition of Table~\ref{tab:main_results} (positive =
contaminated-training better). The CSMAD row is repeated from Table~\ref{tab:main_results}
for reference, as CSMAD trains on the contaminated stream in both conditions.
```

## ⚠️ 주의사항·의존성

**캡션-표 불일치가 이 placeholder의 고유 함정이다**: 캡션은 "PA%K-AUC F1 and VUS-PR"을 약속하는데 tex 표 stub은 F1/Δ 열만 노출한다. Phase 8 채움 시점에 둘 중 하나로 정합화해야 한다 — 권고는 표를 F1+Δ로 확정하고 캡션의 "and VUS-PR"을 삭제하는 쪽(추가 열 없이 지면 유지)이며, VUS-PR 열을 추가하는 선택도 가능하나 어느 쪽이든 침묵 불일치는 금지다. 둘째, Δ의 기준값 의존성: TAB-2의 anomaly-excised 수치가 확정되기 전에 Δ를 선산출하지 말 것(기준이 바뀌면 Δ 전량 재계산). 셋째, 이 실행 묶음은 NUM-019(protocol-effect 블록의 "같은 추가 데이터에 대한 비지도 변화량")의 contaminated 측 소스로 공유될 수 있다 — N-D 그룹과의 소스 공유를 집계 스크립트에 명시하라. 넷째, SWaT 재실행이 발생하므로 입력 차원 45 일치 검증이 필수다(FEEDBACK-7 — 현 raw CSV 경로는 51을 반환; 상수 6컬럼 필터 확인).

## 🔢 연결된 NUM 표

| NUM | 위치 | 관계 |
|---|---|---|
| NUM-019 | §4.2 protocol-effect 문장 | best unsupervised의 standard→contaminated 변화량 — 비교쌍의 contaminated 측 실측을 이 실행 묶음과 공유 가능 (N-D 그룹, 주 소스는 TAB-2 ② 4항) |

---

<!-- PAGE: TAB-B2 -->

# TAB-B2 — Epoch-budget 민감도 표 (Table B.2)

> 💡 budget 비대칭(CSMAD 500 vs unsup 10 vs weak 50)과 그에 따른 선택-기회 비대칭(100 vs 10 checkpoints)이 비교 결론을 바꾸지 않음을 실측으로 보이는 방어 표. baseline은 budget을 늘려 보고, CSMAD는 줄여 본다.

| 항목 | 내용 |
|---|---|
| 위치 | 부록 §B.2, `appendix_B.tex`, `\label{tab:epoch_sensitivity}` |
| 분류 | `[신규 실행(부분 재사용)]` — baseline 50/100ep 신규, CSMAD 축소분은 exp298/299 재사용 |
| 방어 대상 | §15 "epoch budget 비대칭 불공정"(ADV BLK-005) + "test-set model selection" 시나리오의 보조 실측 |

## 🎯 목적과 의도

§4.1.2는 학습 budget 비대칭(500/50/10 epochs)을 은폐 없이 공개하고, 모든 방법이 "주기 평가 후 best-epoch 선택"이라는 동일 구조를 따른다고 방어한다. 그러나 평가 cadence가 고정이므로 budget 비대칭은 **선택 기회의 비대칭**을 수반한다 — §B.2 본문이 명시하듯 CSMAD는 100개 checkpoint(500 epochs ÷ 5), 비지도 baseline은 10개에서 best를 고른다. "더 많은 추첨 기회가 더 좋은 best를 만든 것 아닌가"라는 공격(ADV BLK-005, test-set selection 시나리오의 파생형)에 대한 정면 실측 답변이 이 표다. 설계는 양방향이다: 대표 비지도 baseline의 budget을 50/100으로 **늘려서** 기회를 더 줘 보고, CSMAD의 budget을 **줄여서** 기회를 빼앗아 본다. 양쪽 모두에서 순위 구도가 유지되면 비대칭은 결과의 원인이 아니다.

## 🏁 목표와 기대 결과

성공 기준: Anomaly Transformer·TranAD의 10/50/100 epochs 성능과 CSMAD의 축소 budget/500 epochs 성능이 채워지고, §B.2 본문의 "selection-frequency effect together with the training-length effect" 서술과 표가 맞물리는 것. 기대 패턴: **baseline은 budget을 5–10배 늘려도 큰 향상이 없거나 과적합으로 오히려 하락**하고(소형 모델의 단기 수렴 — budget 책정의 근거였던 수렴 특성 그대로), **CSMAD는 축소 budget에서도 경쟁력을 유지**하는 것이다. 이 패턴이 확인되면 "budget을 맞춰도 결론이 같다"는 방어가 완성된다. 다른 패턴의 해석: baseline이 50/100 epochs에서 유의미하게 상승해 순위가 좁혀지면 budget 비대칭이 결과에 기여했다는 뜻이므로, §4.1.2의 공개 문구를 강화하고 본문 해석을 보수적으로 수정해야 한다. CSMAD가 축소 budget에서 크게 무너지면 그것은 "장기 수렴이 필요한 설계"라는 사실의 공개 재료이지 은폐 대상이 아니다 — warmup 의존성 서술(§3.5)과 연결해 해석한다.

## 🧪 실험 내용과 설계

네 갈래 소스를 조립한다.

| 셀 그룹 | 소스 | 실행 지침 |
|---|---|---|
| baseline 10 epochs | `[재사용]` [CMP-Q3] | main budget 결과 그대로 — Table 2와 동일 값 |
| baseline 50/100 epochs | `[신규 실행]` | `baseline_common.py`의 epochs override로 2 모델(Anomaly Transformer, TranAD) × 2 budget(50, 100) × 대표 데이터셋(2–3개, **TAB-3 대표 선택과 통일 권장**) 실행. 평가 cadence(매 epoch)·best-epoch 선택 구조는 main과 동일 유지 — checkpoint 수가 budget에 비례해 늘어나는 것이 설계 의도다 |
| CSMAD 500 epochs | `[재사용]` [271c] | main 결과 그대로 |
| CSMAD 축소 budget | `[재사용 — 결정 1건 필요]` | exp298(`num_epochs=300, warmup=150`)·exp299(`num_epochs=200, warmup=100`) 완주분이 실재한다(2026-06-11 실측). 단 tex stub의 열 라벨이 "100 epochs"이므로 다음 중 하나를 결정: **(i) exp299(200ep)를 쓰고 열 라벨을 "reduced (200)"로 수정** — 추가 실행 0, warmup 비율 보존 (권고안), (ii) `num_epochs=100, teacher_only_warmup_epochs=50` 신규 1 run |

핵심 설계 제약: CSMAD의 축소 budget에서는 **warmup도 비례 축소**되어야 한다. warmup=250을 고정한 채 epochs=100으로 줄이면 student가 아예 학습되지 않는 무의미 변형이 된다(student 학습 개시가 epoch 250이므로). exp298/299는 이미 이 비례(절반)를 따르고 있어 그대로 쓸 수 있다 — 권고안 (i)을 채택하면 신규 실행이 0건이 된다.

## 📊 구성과 형태

행 = method {Anomaly Trans., TranAD, CSMAD}, 열 = budget {10, 50, 100(또는 reduced), 500} epochs. 각 method의 비해당 budget 셀은 "—" (baseline 행의 500 셀, CSMAD 행의 10/50 셀). 지표는 PA%K-AUC F1 단일, best-epoch 기준 main과 동일. 권고안 (i) 채택 시 CSMAD 열 라벨을 "reduced (200)"로 바꾸고 캡션의 "a reduced budget" 표현은 그대로 유효하다(캡션이 구체 숫자를 약속하지 않으므로 캡션 수정 불필요 — 열 라벨만 수정).

## 📝 캡션

```
Epoch-budget sensitivity. PA\%K-AUC F1 of representative unsupervised baselines
trained for 10 (main budget), 50, and 100 epochs, and of CSMAD trained for 500 (main budget)
and a reduced budget, on representative datasets; best-epoch selection identical to the main
protocol (Section~\ref{sec:impl}).
```

## ⚠️ 주의사항·의존성

첫째, 위에 적은 warmup 비례 축소가 유일한 치명 함정이다 — 신규 run을 택할 경우 `teacher_only_warmup_epochs`를 반드시 함께 줄일 것. 둘째, baseline 50/100 run의 best-epoch 선택 구조(매 epoch eval 후 best)를 main과 동일하게 유지해야 "선택 기회를 늘려준" 실험이 된다 — cadence를 바꾸면 변인이 둘이 된다. 셋째, 대표 데이터셋은 TAB-3 선택(NUM-020)과 통일을 권장하므로 TAB-3의 대표 데이터셋 확정에 약한 의존이 있다(통일하지 않아도 표는 성립하나 서사 일관성이 떨어진다). 넷째, exp299 재사용 결정 (i)/(ii)는 채움 전에 한 번만 내리면 되는 설계 결정이며, 결정 내용을 DECISION_LOG에 남긴다.

## 🔢 연결된 NUM 표

| NUM | 관계 |
|---|---|
| (직접 파생 없음) | §4.1.2의 budget 공개 문장과 §15 방어 표의 "(옵션) Appendix epoch-budget sensitivity" 항목이 이 표를 가리킨다 — inline NUM은 없으나 본문-부록 상호참조 정합 의무가 있다. |

---

<!-- PAGE: TAB-B3 -->

# TAB-B3 — 추론 연산 비용 표 (Table B.3)

> 💡 leave-one-out 추론(50 마스킹 패턴)의 비용을 single-mask 기준선 대비 {FLOPs, wall-clock, peak memory}로 실측한다. wall-clock 배율 실측값이 NUM-031이 되어 §5의 "approximately 50×" 표현과 동기화된다.

| 항목 | 내용 |
|---|---|
| 위치 | 부록 §B.3, `appendix_B.tex`, `\label{tab:compute}` |
| 분류 | `[신규 측정]` — 학습 불필요, 측정 스크립트 1회 |
| 파생 NUM | **NUM-031** (§B.3 본문의 wall-clock overhead factor) |

## 🎯 목적과 의도

CSMAD의 추론은 윈도당 50개 leave-one-out 마스킹 패턴을 평가하므로 연산량이 단일-pass 대비 약 50배다 — 이 비용은 발표 단계부터 공개된 설계 한계이며, §4.2 분석 텍스트와 §5 한계 문장이 모두 이를 인정한다. 이 표의 논증 역할은 **한계 인정의 정직성을 실측으로 완성**하는 것이다: "비용이 크다"는 추상 인정 대신, FLOPs·wall-clock·메모리 세 축의 측정값과 배율을 제시한다. 특히 wall-clock은 batch 차원 병렬화 덕분에 FLOPs 배율(이론 ~50×)보다 낮게 나올 수 있는데, 이 간극 자체가 유용한 정보다 — "연산량은 50×지만 실제 시간 비용은 그보다 작다"는 서술이 가능해지면 한계의 체감 크기가 정확해진다. 방어하는 공격은 "비용 한계를 숨기거나 과소 서술했다"는 유형이며, 측정 방법까지 본문에 명시함으로써 차단한다.

## 🏁 목표와 기대 결과

성공 기준: 3×3 표가 전부 측정값으로 채워지고, NUM-031(wall-clock 배율)이 §B.3 본문에 들어가는 것. 기대 패턴: FLOPs 배율은 패턴당 1 forward 구조상 **~50× 근방**(leave-one-out 50패턴; batch 확장은 wall-clock 병렬화일 뿐 연산량을 줄이지 않는다 — RESEARCH_SYNTHESIS 표A). wall-clock 배율은 GPU 병렬화·메모리 재사용 효과로 **50×보다 낮을 가능성**이 있다. peak memory는 leave-one-out 쪽이 batch 확장만큼 높게 나오되 `patch_batch_size=2` 분할로 상한이 관리된다. **sync 규칙(중요)**: 측정된 wall-clock 배율이 50보다 유의미하게 낮으면 §5의 "approximately 50×"를 "up to 50×"로 완화한다 — registry §5 audit-trail 규칙이며, §5 문장과 같은 pass에서 수정한다. 배율이 50에 근접하면 표현은 그대로 둔다.

## 🧪 실험 내용과 설계

학습이 필요 없는 측정 스크립트 1회다. [271c] 대표 entity 1–2개의 best checkpoint를 로드해 두 채점 모드를 동일 조건에서 측정한다.

**Leave-one-out 측정**: 현행 evaluator의 추론 경로를 **그대로** 사용한다(50개 마스킹 패턴 batch-병렬, `evaluator.py`의 단일 forward 확장 구현). end-to-end 평가 wall-clock은 해당 entity metadata의 `timing.inference_time`과 교차 검증한다 — 측정 스크립트 값과 운영 기록 값이 크게 어긋나면 측정 조건(batch, device 상태)을 재점검한다.

**Single-mask 측정**: 동일 checkpoint로 윈도당 1-pass(단일 마스킹 패턴) 채점 모드를 측정용으로 구성한다. 이 모드는 비교 기준선일 뿐 **논문 점수 산출에는 사용되지 않음**을 표 각주에 명시한다(미사용 옵션을 검증된 경감책처럼 보이게 하지 말 것 — §5의 complementary masking 서술 규칙과 같은 정신).

**측정 사양**: FLOPs는 분석식 또는 profiler(예: torch profiler) 중 하나로 산출하고 **어느 방법을 썼는지 §B.3 본문에 1줄 명시**한다. peak memory는 `torch.cuda.max_memory_allocated()`를 reset 후 측정한다. 두 모드는 **동일 batch 크기·동일 entity**로 측정해야 배율이 의미를 가진다. 측정 하드웨어는 TXT-001 확정 GPU와 동일해야 하며, 학습 머신과 다르면 각주로 구분 명시한다.

## 📊 구성과 형태

tex 확정 3×3 구조: 행 {Single-mask, Leave-one-out, Overhead ×} × 열 {FLOPs / window, Wall-clock (s/entity), Peak GPU mem. (GB)}. Overhead 행은 비율(×)만 기재하고 memory 열의 Overhead 셀은 "—"(배율 개념이 부적합). 측정 entity가 2개면 본문 또는 각주에 entity별 값의 처리(평균 또는 병기)를 명시한다.

## 📝 캡션

```
Computational cost of CSMAD inference: per-window forward FLOPs, end-to-end wall-clock
evaluation time, and peak GPU memory for leave-one-out masking versus single-mask scoring,
measured on representative datasets (hardware of \ref{sec:appendix_impl}).
```

## ⚠️ 주의사항·의존성

NUM-031 sync 조건이 이 페이지의 존재 이유 절반이다 — 표만 채우고 §5 문장을 잊으면 본문-부록 모순이 생긴다(아래 NUM 표 참조). 하드웨어 표기는 TXT-001 확정값과 같은 페이지(§A.1)를 참조하므로 TXT-001 확인이 선행되어야 한다. 측정 스크립트에서 score 후처리 smoothing 등 어떤 추가 연산도 끼워 넣지 말 것(R34 — 측정 대상은 현행 채점 경로 그대로). [271c] checkpoint 로드 시 SWaT를 쓴다면 입력 차원 45 일치 검증(FEEDBACK-7)이 여기에도 적용된다 — 차원 불일치 시 checkpoint 로드 자체가 실패할 수 있으므로 PSM 등 무리스크 entity를 우선 후보로 권장한다.

## 🔢 연결된 NUM 표

| NUM | 위치 | 정의·규칙 |
|---|---|---|
| NUM-031 | §B.3 본문 (`appendix_B.tex` "the measured wall-clock overhead factor is [X.XX]") | leave-one-out vs single-mask **wall-clock 배율 실측값**. sync 조건: 50보다 유의미하게 낮으면 §5 "approximately 50×" → "up to 50×" 완화를 같은 pass에서 수행 (그룹 N-H) |

---

<!-- PAGE: TAB-B4 -->

# TAB-B4 — 확장 ablation 표 (Table B.4)

> 💡 본문 Table 3 밖의 변형 4행(w/o FM, w/o warmup, symmetric decoder)과 Teacher 깊이 민감도 3행을 다룬다. symmetric decoder 행은 contribution bullet 3의 유일한 정량 근거(NUM-024)를 만드는 **신규 실행 최우선** 항목이다.

| 항목 | 내용 |
|---|---|
| 위치 | 부록 §B.5, `appendix_B.tex`, `\label{tab:extended_ablations}` |
| 분류 | `[재사용(no_fm) + 신규 실행(3종)]` |
| 파생 NUM | **NUM-024**(load-bearing), **NUM-025** |

## 🎯 목적과 의도

본문 Table 3은 라벨 경로 3종의 분해(4행)로 고정되었고, 그 밖의 설계 요소 검증은 전부 이 표가 담당한다. 논증 역할은 행마다 다르다. **Symmetric decoder 행**이 가장 무겁다: contribution bullet 3(비대칭 capacity gap이 신뢰할 만한 anomaly 신호를 만든다)의 **유일한 정량 근거**이며, 블루프린트 §6.7이 "미실행 + load-bearing — Phase 5 진입 전 실행 필수"로 못 박은 항목이다(warmup 공격 패턴이 bullet 3에서 재발하는 것을 차단). **w/o FM 행**은 §12 R10 논증표의 "FM ablation 근거 필요(미존재)" 공백을 정량 해소한다. **w/o warmup 행**은 "warmup ablation 없음" 공격(§15)에 대한 실측 대응이되, warmup은 contribution이 아니므로(블루프린트 결정 ①) 부록 배치가 논증 강도와 맞다. **depth sensitivity 블록**은 "왜 하필 3L/2L인가"라는 설계 선택 질문에 대해 3/2/1 스윕으로 답한다 — gap 크기 선정의 이론적 정당화 부족(RESEARCH_SYNTHESIS 표A)을 실측으로 보완하는 장치다.

## 🏁 목표와 기대 결과

성공 기준: 상단 4행 + 하단 3행 × (TAB-3과 동일한 대표 데이터셋 + Avg) 열이 채워지고, NUM-024/025가 §B.5 문단 2개에 들어가는 것. 기대 패턴: 각 변형 행이 Full 대비 **하락**하는 것 — 특히 symmetric 행의 하락폭(NUM-024)은 capacity gap의 효과를, w/o FM의 하락폭(NUM-025)은 student 표현 붕괴 방지 효과를 정량화한다. depth 블록에서는 3 → 2 → 1로 갈수록 하락이 깊어지는 단조 경향이 나오면 "Teacher가 Student보다 깊어야 한다"는 설계 원리가 스윕 차원에서 지지된다. 다른 패턴의 해석: 어떤 변형이 Full보다 **좋게** 나오면(음수 하락폭) 본문 부호 규약("removal costs X points")을 쓸 수 없으므로 해당 문장을 "improves by"로 재작성하고, symmetric이 그 경우라면 contribution bullet 3의 표현 강도를 반드시 하향한다 — 결과 확인 전 문장 선점은 금지다(A8). w/o warmup이 무해하게 나오면 warmup을 안정화 장치로만 서술하는 현 기조가 오히려 강화된다.

## 🧪 실험 내용과 설계

행별 소스와 실행 지침을 전수 명세한다. 신규 학습은 **3 run × 대표 데이터셋**뿐이다.

| 행 | 소스 | 실행 지침 |
|---|---|---|
| Full model | [271c] `[재사용]` | TAB-3 행1과 **동일 값** — 별도 추출 금지, 같은 집계 산출물 공유 |
| w/o FM loss | **exp285_no_fm** `[재사용]` | `use_feature_matching=False` 단독 diff로 실측 확인된 기존 run, 대표 데이터셋 완주 상태 — metadata 집계만. NUM-025 파생 |
| w/o Teacher warmup (250→0) | `[신규 실행]` | 큐 신규 항목: `teacher_only_warmup_epochs=0`, 그 외 271 canon 동일. **인지 사항**: λ_rev ramp의 분모가 `num_epochs − warmup`이므로 warmup=0이면 sigmoid ramp가 epoch 0부터 시작한다 — 이는 의도된 변형이며 버그가 아니다 |
| Symmetric dec. (2L/2L) | `[신규 실행]` | `num_teacher_decoder_layers=2` (Student 2 유지). **NUM-024 파생 — 신규 실행 중 최우선** (기여 bullet 3 load-bearing) |
| Teacher depth 3 (default) | = Full 행 `[재사용]` | 같은 값의 중복 기재 (블록 비교 가독성용) |
| Teacher depth 2 | = Symmetric run과 **동일 config** | 같은 run으로 두 행을 채운다 — 이중 실행 불필요 |
| Teacher depth 1 | `[신규 실행]` | `num_teacher_decoder_layers=1`, 그 외 271 canon 동일 |

큐 등재는 `configs/queue_dedup_renumbered_v5.json` 형식(`exp_num` / `dataset` 리스트 / `config_override` 공백 구분 키=값)을 따른다. 큐 항목 작성 시 `config_override`에 같은 키를 중복 기재하는 패턴(exp287의 `force_mask_anomaly` True→False last-wins 사례, OBS-2)을 답습하지 말 것 — 키는 1회만, 최종값으로 기재한다.

## 📊 구성과 형태

상단 블록(변형 4행)과 하단 블록(depth 3행)을 midrule로 분리(tex 확정). **열 집합은 TAB-3과 글자 단위 동일**해야 한다 — 대표 데이터셋 선정(NUM-020)이 TAB-3에서 확정되면 이 표가 그대로 따른다. 지표는 PA%K-AUC F1, best epoch 기준 main과 동일. 행 라벨은 tex 확정 표기를 따른다: "w/o FM loss", "w/o Teacher warmup (250→0)", "Symmetric dec. (2L/2L)", "Teacher depth 3 (default)/2/1".

## 📝 캡션

```
Extended ablations: the variants beyond the confirmed rows of
Table~\ref{tab:ablation} --- w/o FM loss, w/o Teacher-only warmup (250$\to$0), and a
symmetric decoder (Teacher 2L\,/\,Student 2L) --- and a Teacher-decoder depth sensitivity
study (3/2/1 layers against the 2-layer Student). PA\%K-AUC F1 on the ablation datasets of
Table~\ref{tab:ablation}.
```

## ⚠️ 주의사항·의존성

첫째, **conditional 게재 규칙**: symmetric-decoder run이 게재 시점까지 미완이면 contribution bullet 3을 "design principle" 수준으로 표현 강도 하향한다(Phase 6 규칙 — landing spot은 이미 §B.5로 확보됨). 미완 행을 placeholder 상태로 본문에 남기는 것은 금지다. 둘째, §B.5의 본문 문단 2개("Symmetric decoder capacity", "FM loss regularizer")가 NUM-024/025를 들고 있으므로 **표와 문단 수치를 같은 pass에서 동시 갱신**한다. 셋째, 열 집합의 TAB-3 종속: NUM-020 확정 전에 이 표의 열을 독자적으로 정하지 말 것(열 불일치 금지). 넷째, depth 2 행과 symmetric 행이 같은 run임을 집계 스크립트에 명시해 두 행이 미래에 어긋나는 것을 방지한다. 다섯째, exp285 재사용 시 단독 diff(`use_feature_matching=False`) 여부를 metadata에서 한 번 더 확인하고 쓴다 — exp290(no_fm+no_grl 복합)과 혼동 금지.

## 🔢 연결된 NUM 표

| NUM | 위치 | 정의·규칙 |
|---|---|---|
| NUM-024 | §B.5 "Symmetric decoder capacity" 문단 | Full 행 − Symmetric 행의 Avg 차 (**기여 bullet 3 load-bearing**). 부호 규약: 양수 하락폭 — 음수면 문장 자체를 "improves by"로 재작성 + bullet 3 강도 하향 (그룹 N-E) |
| NUM-025 | §B.5 "FM loss regularizer" 문단 | Full 행 − w/o FM 행의 Avg 차. exp285 재사용으로 산출 (그룹 N-E) |
| NUM-020 | §4.3 (TAB-3 소유) | ablation 대표 데이터셋 수 — 이 표의 열 집합이 종속되는 외부 결정 |

---

<!-- PAGE: FIG-B1 -->

# FIG-B1 — 파라미터 민감도 2-패널 곡선 (Figure B.1)

> 💡 score 결합비 c(기본 4)와 masking ratio ρ(기본 0.15)의 민감도 곡선. 좌패널은 재학습 없는 재채점(c는 추론 전용 파라미터), 우패널은 ρ별 전체 재학습 — 두 패널의 비용 구조가 본질적으로 다르다.

| 항목 | 내용 |
|---|---|
| 위치 | 부록 §B.4, `appendix_B.tex`, `\label{fig:param_sensitivity}`. 크기 ~3.5 cm ≈ 0.30p |
| 분류 | `[재사용(좌패널: c 재채점)] + [신규 실행(우패널: ρ 재학습)]` |
| 파생 NUM | 없음 (직접) |

## 🎯 목적과 의도

CSMAD는 단일 설정(per-dataset 튜닝 없음)으로 전 entity를 학습한다 — Table A.1 캡션이 이를 명시한다. 이 그림의 논증 역할은 그 단일 설정의 두 핵심 자유도(score 결합비 c=4, masking ratio ρ=0.15)가 **요행으로 고른 값이 아님**을 보이는 것이다. 방어하는 공격은 "하이퍼파라미터를 test 성능으로 튜닝했다 / 기본값이 cherry-pick이다" 유형이다. 기본값 주변에서 성능 곡선이 평탄하면 "이 설계는 c·ρ 선택에 민감하지 않다"는 견고성 서술이 가능해지고, 반대로 민감하다면 그 사실을 공개하고 기본값 선택의 근거를 서술하는 것이 정직한 대응이다. 두 패널은 또한 점수 산식의 두 구성 요소(recon-disc 결합, 마스킹 예산)가 §3의 설계 논증과 연결되는 실측 단면이기도 하다.

## 🏁 목표와 기대 결과

성공 기준: 좌패널 c ∈ {1, 2, 4, 8, 16} 5점 곡선과 우패널 ρ ∈ {0.05, 0.10, 0.15, 0.20, 0.30} 5점 곡선이 대표 데이터셋별로 그려지고, 기본값 위치(c=4, ρ=0.15)가 시각적으로 표시되는 것. 기대 패턴: **기본값 근방의 평탄한(완만한) 곡선** — c는 adaptive scaling이 스케일 차이를 이미 보정한 뒤의 비율이므로 광역에서 완만할 가능성이 높고, ρ는 너무 작으면(0.05 → |M|=2) 학습 신호가 빈약해지고 너무 크면(0.30 → |M|=15) 가시 맥락이 부족해지는 완만한 단봉 형태가 자연스러운 예측이다. 다른 패턴의 해석: c 극단(1 또는 16)에서 급락하면 두 점수 성분의 상보성(둘 다 필요함)이 오히려 강조된다 — §4.5 성분 분해 서사와 연결. ρ에서 기본값이 최적이 아니면 그 사실을 그대로 보고한다(기본값은 main 실험 전 결정된 값이지 사후 최적값 주장이 아님을 본문 1문장으로 명시하면 된다). 어느 경우든 **수치 발명 없이 곡선 확정 후 서술**한다.

## 🧪 실험 내용과 설계

두 패널의 비용이 본질적으로 다르므로 분리 실행한다.

**좌패널 — c sweep `[재사용 + 재채점]`**: c(= `score_recon_disc_ratio`, 기본 4)는 **추론 시에만** 점수식에 들어간다 — `mae_anomaly/scoring.py`의 score = recon + scaled_disc/c. 따라서 재학습이 전혀 필요 없다. [271c]의 best checkpoint(또는 저장된 per-patch score 성분)에 대해 c ∈ {1, 2, 4, 8, 16}(log2 격자, 기본 4 중심)로 재채점 → 재평가만 수행한다. 2026-06에 정비된 eval-recompute 도구 경로를 재사용할 수 있다. 대표 데이터셋은 FIG-3과 동일 선택을 권장한다(권장 SWaT excl22 + PSM). **핵심 제약**: c별로 best epoch을 재선정하지 말 것 — **main run의 best epoch을 고정**한 채 c만 바꿔야 "그 설정 주변의 민감도"가 된다. 재선정하면 test-set selection이 c에도 적용되어 별개 실험이 되어버린다. 본문에 best-epoch 고정 방식을 1줄로 명시하는 것을 권장한다.

**우패널 — ρ sweep `[신규 실행]`**: ρ는 학습 마스킹을 바꾸므로 ρ별 **전체 재학습**이 필요하다. 격자 ρ ∈ {0.05, 0.10, 0.15, 0.20, 0.30} 중 기본 0.15는 [271c]를 재사용하므로 신규는 4 run × 대표 2–3 데이터셋이다. 큐 항목은 `config_override`에 `masking_ratio=<ρ>`만 변경하고 그 외 271 canon 동일(500 epochs, seed 42). ρ 변경 시 |M| = round(50×ρ)로 자동 변동함을 인지한다(0.05 → 2패치, 0.30 → 15패치).

## 📊 구성과 형태

가로 2-패널. 좌: X = c(**log scale 권장**), 우: X = ρ(선형). Y 공통: PA%K-AUC F1. 패널별로 대표 데이터셋당 1선씩, 범례에 데이터셋명. 기본값 위치(c=4, ρ=0.15)에 수직 참조선 또는 강조 마커. 기호는 반드시 ρ를 사용한다(구표기 r_m 금지 — v2-r3 M-5). 높이 ~3.5 cm(≈0.30p)로 두 패널이 한 줄에 들어가는 컴팩트 구성이다.

## 📝 캡션

```
Parameter sensitivity. PA\%K-AUC F1 as a function of (\textit{left}) the score
combination ratio $c$ around its default 4 and (\textit{right}) the masking ratio $\rho$
around its default 0.15, on representative datasets; all other settings fixed to the main
configuration.
```

## ⚠️ 주의사항·의존성

첫째, 좌패널의 best-epoch 고정 규칙(위 🧪)이 이 그림의 과학적 성립 조건이다 — 위반하면 그림 전체가 별개 실험이 된다. 둘째, 우패널은 **현재 어느 큐에도 등재되어 있지 않다** — 신규 등재가 필요하다. 기존 큐 295–303은 전부 다른 변형이다: 295/296/300–303 = window/patch 크기 sweep, 297 = dynamic d_model, 298/299 = epoch-budget 변형이며, 큐 v5 전 32항목에서 `masking_ratio` override는 0건으로 실측 확인되었다(r2 정정). 셋째, c sweep 재채점 시 점수 산식은 `mae_anomaly/scoring.py` 단일 원천만 사용한다 — 다른 곳에 식을 복제하지 말 것(CLAUDE.md API 체크리스트 3항, FM-omission 사고 재발 방지). 넷째, 시각화 코드에 후처리 smoothing을 넣지 말 것(R34).

## 🔢 연결된 NUM 표

| NUM | 관계 |
|---|---|
| (직접 파생 없음) | 대표 데이터셋 선택을 FIG-3(NUM-026)과 통일 권장 — 약한 설계 의존만 존재. |

---

<!-- PAGE: ALG-C1 -->

# ALG-C1 — CSMAD 학습 의사코드 검증 (Algorithm C.1)

> 💡 의사코드 초안은 이미 tex에 있다. 남은 작업은 실험이 아니라 **canonical training loop와의 행 단위 대조 검증** + 캡션의 "(pseudocode placeholder)" 꼬리 제거다. 모든 줄은 trainer.py/model.py/loss.py의 실제 동작에 1:1 대응해야 한다.

| 항목 | 내용 |
|---|---|
| 위치 | 부록 §C.3, `appendix_C.tex`, `\label{alg:training}`, algorithm2e `algorithm*` 2단 폭 ~30줄 |
| 분류 | `[제작 — 코드 대조 검증]` — 실험 소스 없음 |
| 정본 | 271_CONFIG_TRUTH r4 §VIII + `trainer.py` / `model.py` / `loss.py` |

## 🎯 목적과 의도

재현성 주장의 마지막 조각이다: 하이퍼파라미터 전수(Table A.1/A.3)와 수식(§C.1)이 있어도, 학습 절차의 **제어 흐름**(무엇이 언제 켜지고 꺼지는가)은 의사코드만이 전달할 수 있다. 특히 CSMAD는 Teacher-only warmup의 student forward skip, GRL의 λ 이중 구조(손실 가중 λ_GRL vs 반전 계수 λ_rev), batch 내 positive 부재 시 GRL 손실 skip 등 **타이밍·게이트가 본질인 설계**이므로, 이 절차의 부정확한 의사코드는 재현 실패의 직접 원인이 된다. 방어 관점에서는 P3 재리뷰가 두 번 지적한 off-by-one(epoch 표기 규약)과 단일-λ 합산 오서술의 재발 지점을 막는 것이 핵심이다 — 검증 체크리스트의 3·4항이 정확히 그 지점이다.

## 🏁 목표와 기대 결과

성공 기준: 아래 5요소 체크리스트 전 항목 통과 + 캡션의 "(pseudocode placeholder)" 꼬리 제거(이 제거가 resolved 신호다). 의사코드의 어떤 줄도 코드에 없는 행동을 발명하지 않았고, 코드의 어떤 활성 동작도 의사코드가 누락하지 않았음이 확인되어야 한다. 검증 중 불일치가 발견되면 — 예컨대 epoch 경계의 ±1, λ 표기의 병합, 손실 항 누락 — 의사코드를 코드 쪽에 맞춰 수정한다(코드가 ground truth, 역방향 수정 금지). 검증 통과 후 의심 잔여 항목이 있으면 271_CONFIG_TRUTH의 file:line으로 재확인한다.

## 🧪 실험 내용과 설계

실험이 아니라 검증 작업이다. 5요소 체크리스트를 계승한다 — 각 항목을 초안 tex의 해당 줄과 코드를 나란히 놓고 행 단위로 대조한다.

| # | 검증 항목 | 정본·코드 근거 | 초안 상태와 남은 확인 |
|---|---|---|---|
| 1 | 전처리: SWaT constant 6컬럼 제거(45 = 51−6) + per-entity train-구간 min–max | 271_CONFIG_TRUTH r4 §VIII | 초안 반영됨 — 컬럼 목록 {P202, P401, P404, P502, P601, P603}이 §A.1 재현성 노트와 일치하는지 확인 |
| 2 | anomaly-priority masking: priority 식 π_i = 10³·y_i + η_i, argtopk |M| | `model.py` masking 경로, Eq. C.5(`eq:masking_rule`) | 초안 반영됨 — **Eq. C.5와 기호가 글자 단위 일치**하는지 확인 (y^p_i 표기 포함) |
| 3 | Teacher-only gating: **0-based epoch 0–249 동안 학습 경로 student forward 자체 skip** | r4 정본: "student 학습은 0-based epoch 250(= 251번째 epoch)부터"; `trainer.py:526–535` → `model.py:1119` | 초안의 `If e > 250`은 1-based 표기 — 0-based 250 개시와 ±1로 일치하는지 **epoch 표기 규약을 각주 또는 KwIn에 명시**할 것 (off-by-one이 P3 재리뷰 단골) |
| 4 | 손실 조립: L_total = L_recon + L_OD + λ_FM·L_FM + λ_GRL·L_cls; **λ 이중 구조** 분리 표기 — 손실 가중 λ_GRL(grad-ratio clamp[0,10] × 0.2, 직전 epoch smoothing)과 반전 계수 λ_rev(sigmoid ramp 2/(1+e^{−10τ})−1, τ=clip((e−250)/250, 0, 1)) | `trainer.py:752–763`(λ_GRL), `trainer.py:1205-1207`(λ_rev); r4 NEW-B1 | 초안 반영됨 — **단일 λ로 합치지 말 것**. ⚠️ (OBS-1) τ식의 e는 3항과 **동일한 epoch 표기 규약**을 따라야 한다: 위 식은 1-based e 규약에서만 코드(0-based `(epoch−250+1)/250`)와 일치하므로, 3항의 규약 명시가 이 식에도 적용됨을 한 줄로 연동 표기할 것. GRL 손실의 batch 내 positive window 부재 시 skip(`loss.py:293-302`)은 초안 반영됨 |
| 5 | 평가: 5 epoch 간격 test-split 평가 + best PA%K-AUC F1 추적 | config `eval_interval=5`, `best_epoch_metric` | 초안 반영됨 — `e mod 5` 표기 확인 |

검증 방법: tex의 algorithm 블록 30줄을 위 표의 코드 위치와 1:1 대응시키는 대조 시트를 만들고, 대응 없는 줄(발명)·대응 누락(생략)을 0건으로 만든다.

## 📊 구성과 형태

algorithm2e의 `algorithm*`(2단 폭 float, 페이지 상단 배치), `\footnotesize`, ~30줄 — 현 구조를 유지한다(PDF QA에서 단일 column 배치의 overprint 문제가 이미 해결된 형태). KwIn/KwOut, `\tcp` 주석 블록(Preprocessing / masking / Teacher / Student-gated / Loss assembly / Evaluation) 구조도 유지. epoch 표기 규약 명시는 KwIn 줄 또는 각주로 추가한다.

## 📝 캡션

현재 tex 캡션과 확정 시 교체 문구:

```
(현재)  CSMAD Training (pseudocode placeholder)
(확정 시) CSMAD training procedure.
```

"(pseudocode placeholder)" 꼬리의 제거가 이 placeholder의 resolved 신호다.

## ⚠️ 주의사항·의존성

행동 발명 금지가 제1원칙이다 — 모든 줄은 `trainer.py`/`model.py`/`loss.py`의 실제 동작에 1:1 대응해야 하며, 의심 항목은 271_CONFIG_TRUTH의 file:line으로 재확인한다. 의사코드 내 수식 참조(Eq. C.1/C.4/C.5, `eq:ltotal` 등)는 **빌드 후 식 번호를 재확인**한다 — 부록 식 번호는 본문 구성 변경에 따라 밀릴 수 있다. AMP bf16·optimizer 세부는 의사코드 범위 밖이다(Table A.1에 위임) — 검증 중에 "코드에 있으니 추가하자"는 유혹을 따르지 말 것. warmup 구간의 student 상태를 서술할 일이 생기면 "frozen"이 아니라 "forward skipped (training path)"가 정확하다(평가 경로는 full forward — FIG-2 ⑤와 동일 규약).

## 🔢 연결된 NUM 표

| NUM | 관계 |
|---|---|
| (없음) | NUM 파생 없음. Eq. C.5(`eq:masking_rule`)·`eq:ltotal`·`eq:adaptive_weight`·`eq:lcls_app`와의 기호·번호 정합 의무만 진다. |

---

<!-- PAGE: TXT-001+TXT-002 -->

> 💡 **이 페이지는 TXT placeholder 2종(4개소)을 묶는다.** 실험이 아니라 확인·결정 작업이며, 둘 다 "추측 기재 금지"가 핵심 규칙이다.

# TXT-001 — GPU 모델명 (§A.1, 1개소)

> 💡 §A.1 환경 문단의 "[GPU model]"을 실제 실험 수행 GPU 모델명으로 치환한다. metadata에 GPU 필드가 없으므로 실행 호스트 이력 확인이 유일한 경로다 — 추측 기재 금지.

| 항목 | 내용 |
|---|---|
| 위치 | `appendix_A.tex:80` — "All experiments run on [GPU model]" (`% PH:TXT-001`) |
| 분류 | `[신규 측정 — 확인만]` |

## 🎯 목적과 의도

재현성·비용 서술의 기준 하드웨어를 확정한다. TAB-B3(연산 비용)의 측정 하드웨어 표기가 이 값을 참조하므로, 단순 빈칸이 아니라 부록 내 교차 참조의 앵커다.

## 🏁 목표와 기대 결과

성공 기준: 271canon(및 baseline) 실험을 **실제 수행한** GPU 모델명이 확인 절차를 거쳐 기재되는 것. 학습 그룹과 baseline 그룹의 머신이 다르면 그룹별 병기가 올바른 결과다 — 단일 모델명을 강요하지 않는다.

## 🧪 실험 내용과 설계

확인 절차는 다음과 같다. [271c] metadata에는 GPU 모델 필드가 **없다**(2026-06-11 실측 — `timing`/`config`에 부재, `device='cuda'`뿐). 따라서 첫째, 271canon 실행 호스트에서 `nvidia-smi --query-gpu=name`을 확인한다 — 현 machineA 실측은 NVIDIA GeForce RTX 4090이고 271canon이 이 머신에서 실행 중이므로 유력하나, **호스트 이력을 확인한 후에만 기재**한다(전체 실행이 한 머신이었는지 확인; 추측 금지 원칙). 둘째, baseline 실행 머신이 다르면 그룹별로 병기한다. 셋째, 향후 재실행분부터는 metadata에 GPU명 기록 필드를 추가할 것을 권장한다(이 확인 비용 자체를 제거).

## 📊 구성과 형태 / 📝 캡션

표·그림이 아닌 본문 1문장이다. 해당 문장 원문:

```
All experiments run on [GPU model];
code, configurations, and the exact dataset partitions will be released at [URL].
```

## ⚠️ 주의사항·의존성

TAB-B3의 측정 머신이 학습 머신과 다르면 그쪽 각주에서 구분 명시해야 하므로, TXT-001 확정이 TAB-B3 채움보다 선행되는 것이 자연스럽다.

## 🔢 연결된 NUM 표

| NUM | 관계 |
|---|---|
| (없음) | TAB-B3의 하드웨어 참조 문구만 이 값에 의존. |

---

# TXT-002 — 코드 저장소 URL (3개소)

> 💡 Abstract·§A.1·§5의 "[URL]" 3개소를 공개 저장소 URL로 일괄 치환한다. 세 곳 글자 단위 동일이 의무이며, 제출 단계에서는 익명 요건을 먼저 확인한다.

| 항목 | 내용 |
|---|---|
| 위치 | `main.tex:110`(Abstract 말미) · `appendix_A.tex:81`(§A.1) · `sec5_conclusion.tex:31`(§5 말미) |
| 분류 | `[결정 사항]` — 실험·측정 무관, 공개 절차 결정 |

## 🎯 목적과 의도

코드 공개(R25)는 재현성 주장의 전제다. "release upon acceptance" 문구는 이미 확정되어 있으므로, 남은 것은 URL 실값과 그 치환 시점·익명성 처리뿐이다.

## 🏁 목표와 기대 결과

성공 기준: 세 곳이 **글자 단위로 동일한** URL로 치환되는 것. 제출 시점과 게재 시점에 들어가는 값이 다를 수 있다(익명 미러 → 실제 저장소) — 단계별로 두 번의 일괄 치환이 모두 3개소 동시여야 한다.

## 🧪 실험 내용과 설계

절차는 다음과 같다. 첫째, 제출 단계에서 저널의 익명 요건을 확인한다 — 정책에 따라 anonymous.4open.science 등 익명 미러가 필요할 수 있다. 둘째, 게재 확정 시 실제 URL로 일괄 치환한다. 셋째, 치환 시마다 grep으로 3개소를 동시 확인한다: `grep -n "\[URL\]"`를 tex 소스 전체(`main.tex` + 본문·부록 tex — 현 위치 기준 `paper/07_latex/*.tex`)에 걸어 잔존 0건을 확인한다. 공개 전 점검 checklist(RESEARCH_SYNTHESIS §⑦: 공개 branch 결정, 공개 범위 분리 — `configs/` 큐 JSON·`results/`·`temp/`·`paper*/` 비공개 처리, secret/credential 스캔, 재현 진입점 문서화 — SWaT 45-feature 재현성 플래그 포함)가 전부 미결 상태이므로, URL 확정 전에 이 checklist 해소가 선행되어야 한다.

## 📊 구성과 형태 / 📝 캡션

본문 문장 3개소. Abstract·§5는 "will be released at [URL]" 패턴, §A.1은 위 TXT-001 인용 문장과 공유.

## ⚠️ 주의사항·의존성

세 곳 동일 의무가 유일한 규칙이지만 가장 어기기 쉬운 규칙이다 — 한 곳만 고치고 커밋하는 사고를 grep 의식(ritual)으로 차단한다. 익명 요건 위반(실명 저장소 URL을 제출본에 노출)은 desk reject 사유가 될 수 있으므로 제출 직전 최종 확인 항목에 포함한다.

## 🔢 연결된 NUM 표

| NUM | 관계 |
|---|---|
| (없음) | TXT-002 ×3개소 자체가 등록 단위 (REGISTRY TXT 2종/4개소 중 3개소). |

---

<!-- PAGE: R-PROBE -->

# R-PROBE — GRL 억제의 기계적 증거: Probing Classifier (권고 실험, 원고 비반영)

> 💡 rebuttal 대비 전용 권고 실험. frozen checkpoint 위에 소형 probe를 학습시켜 "GRL이 Student 표현에서 anomaly-identity 정보를 실제로 지웠는가"를 AUC로 직접 측정한다. 원고는 한 글자도 바뀌지 않는다 — D-014 (b) 등재 의무 이행 항목.

| 항목 | 내용 |
|---|---|
| 위치 | 원고 placeholder 없음 — Notion '권고 실험' 하위 절로만 발행 |
| 분류 | `[신규 측정]` — 본 모델 학습 불필요, probe만 학습 (§7.4 1행) |
| 근거 | D-014 (b) / p8 리뷰 F-1 BLOCKER 해소 항목 (spec r2 §6R) |

## 🎯 목적과 의도

**목적과 의도가 이 항목의 전부다.** 논문의 핵심 주장 중 하나는 "GRL이 Student decoder에서 anomaly-identity 정보를 능동적으로 억제하여, anomaly 구간에서 Teacher–Student discrepancy가 증폭된다"는 메커니즘 서술이다. 본문의 근거는 성능 ablation(TAB-3 행2 — GRL을 빼면 성능이 떨어진다)인데, 성능 하락은 메커니즘의 **간접 증거**일 뿐이다 — 까다로운 리뷰어는 "성능이 떨어진 것이 정말 '표현에서 정보가 억제'되었기 때문인가, 다른 부수 효과 때문인가"를 물을 수 있다(§15의 "GRL이 student를 망가뜨리지 않는가" 시나리오의 공격적 변형). 이 실험은 그 질문에 대한 **직접적·기계적 증거**를 준비한다: 표현 공간에서 anomaly 정보의 선형 추출 가능성(probe AUC)을 Teacher와 Student 사이에서 비교함으로써, "Student 표현에서는 anomaly가 읽히지 않는다"를 분류 성능이라는 해석 불요의 숫자로 보인다. 원고에는 반영하지 않는다 — rebuttal 단계에서 요구받을 때 즉시 제시할 수 있는 탄약고 역할이며, 따라서 결과가 어떻든 원고 리스크가 없다.

## 🏁 목표와 기대 결과

기대 패턴: **Student probe AUC ≪ Teacher probe AUC** — Teacher의 동일 위치 hidden에서는 anomaly window가 상당히 분류되는 반면(억제가 없으므로 정보 잔존), GRL이 작용한 Student hidden에서는 분류가 어려워야(AUC가 chance 수준에 근접) 억제 성공의 정량 증거가 된다. 확장 대조군까지 수행하면 기대 구도는 "w/o GRL Student probe AUC > with-GRL Student probe AUC" — GRL이 없으면 Student에 anomaly 정보가 잔존함을 보이는 대조다. 다른 패턴의 해석: 두 probe AUC의 차이가 작으면 GRL의 억제가 표현 수준에서 약하다는 뜻이므로, 이 결과는 rebuttal에서 사용하지 않고(원고 무변경이므로 손실 없음) 메커니즘 서술의 표현 강도를 내부적으로 점검하는 입력으로만 쓴다. 성공 기준은 발행이 아니라 **준비 완료**다: 절차·수치가 정리된 내부 노트 1편.

## 🧪 실험 내용과 설계

본 모델의 학습은 일절 없다 — 표현은 전부 frozen이고 probe만 학습한다.

첫째, [271c] 대표 entity(권장: TAB-3 대표 데이터셋과 동일 — 서사 일관성)의 best checkpoint를 동결 로드한다. 둘째, test 윈도들에 대해 두 표현을 추출한다: ① Student decoder의 **final-layer hidden, output projection 직전** — GRL 부착 지점과 정확히 동일한 위치다(FIG-2 ③ⓒ의 명시 라벨과 같은 지점; 다른 층에서 뽑으면 실험의 의미가 사라진다), ② Teacher의 동일 위치 hidden. 셋째, 각 표현 위에 소형 probe — LayerNorm + Linear 1층, **GRL head와 유사한 용량**(용량을 맞춰야 "추출기 능력 차이"가 아니라 "표현 내 정보량 차이"를 측정한다) — 를 anomaly window 이진 분류로 학습한다(표현 frozen, probe 파라미터만 학습). 넷째, 두 probe의 test AUC를 비교한다.

**확장(선택)**: TAB-3 행2(w/o GRL, OD-exclusion 유지) run이 완료되면 그 checkpoint에 동일 probing을 적용해 GRL 유/무 Student probe AUC 차이를 병기한다. 주의 — 기존 exp290은 no_fm+no_grl **복합** 변형이므로 대조군으로 쓸 경우 그 사실을 각주로 반드시 밝힌다(단독 효과로 오인 금지).

## 📊 구성과 형태

산출물은 원고 figure/table이 아니라 rebuttal 대비 내부 노트다. 권장 형태: probe AUC 비교 표 1개 {표현 출처(Teacher / Student / Student w/o GRL)} × {entity} × AUC, 그리고 절차 요약 문단. rebuttal에 첨부할 수 있도록 self-contained로 작성한다.

## 📝 캡션

해당 없음 — 원고 무변경 항목이므로 확정 캡션이 존재하지 않는다. rebuttal 첨부 시의 표 제목은 작성 시점에 자유 작성한다.

## ⚠️ 주의사항·의존성

원고의 어떤 placeholder와도 연결되지 않으며 REGISTRY 커버리지 산식에도 불포함이다(커버리지 불변) — 이 항목 때문에 원고를 고치는 일이 없도록 한다. 추출 지점의 정확성이 실험의 성패다: "output projection 직전"이 아닌 위치에서 뽑은 hidden은 GRL 부착 지점과 다르므로 증거 능력이 없다. probe 용량을 GRL head보다 크게 잡으면 "강한 추출기는 억제된 표현에서도 정보를 캐낸다"는 반론에 노출된다 — 유사 용량 원칙을 지킬 것. 의존성: 기본 실험은 [271c] 완주분만으로 즉시 가능; 확장 대조군만 TAB-3 행2 run(실행 대시보드 #4)에 의존한다.

## 🔢 연결된 NUM 표

| NUM | 관계 |
|---|---|
| (없음) | 원고 placeholder와 무관 (권고 실험, D-014 (b)). §7.4 측정 작업 표에 1행으로 등재됨. |

---

<!-- PAGE: OVERVIEW -->

# Placeholder 전체 지도 · 실행 대시보드 (부모 페이지 자료)

> 💡 원고의 모든 placeholder를 한 페이지에서 조망한다: 분류·개수 전수 지도, 실행 우선순위 대시보드(신규 실행 11 + 신규 측정 5 + 완주 대기 3 + 재사용 8), TAB-2 루트 의존 그래프, 분류 라벨 정의. 하위 페이지는 placeholder별 상세 명세다.

## 📌 분류 라벨 정의

모든 placeholder에는 소스 분류 라벨이 부여되어 있다. 라벨은 "이 빈칸을 채우기 위해 무엇이 필요한가"를 한 단어로 답한다.

| 라벨 | 의미 | 전형적 작업 |
|---|---|---|
| `[재사용]` | 기존 실험 결과 또는 코드 상수에서 **추출만** 하면 됨 — 학습 불필요 | metadata 집계, 코드 덤프, checkpoint 재채점 |
| `[완주 대기]` | 진행 중인 271canon·큐의 잔여 entity **완주 후 집계** (잔여: SMD 6 · SMAP 49 · MSL 22 — 2026-06-11 실측) | 완주 모니터링 → 집계 스크립트 |
| `[신규 실행]` | **새 학습 실험**이 필요 (큐 등재 또는 신규 스크립트) | 큐 항목 작성 → GPU 실행 → 집계 |
| `[신규 측정]` | 학습 없이 **측정·스크립트 1회**로 해결 (통계 추출, 비용 측정, 확인 작업) | 1회성 스크립트 |
| `[제작]` | 실험 무관 — **다이어그램/의사코드 제작·검증**만 필요 | TikZ/벡터 제작, 코드 대조 |

---

## 🗺️ 전체 placeholder 지도 (REGISTRY v3-r1 전수 — 누락 0)

총계: **FIG 5 · body TAB 3(+흡수 1) · appendix TAB 8(+Table A.4 부분) · ALG 1 · NUM 31 · TXT 2종 4개소 · 권고 실험 1**. 담당 열의 B1은 본문 figure/table 페이지(spec-enricher-B1), B2는 본 문서(appendix·잔여)다.

| ID | 한 줄 정의 | 분류 | 담당 |
|---|---|---|---|
| FIG-1 | 3-패널 setting 비교 개념도 (§1) | `[제작]` | B1 |
| FIG-2 | CSMAD 아키텍처 개요 2-패널 (§3.2) | `[제작]` | B1 |
| FIG-3 | label sparsity sweep 곡선 (§4.4) | `[신규 실행]` | B1 |
| FIG-4 | score 성분 분해 정성 시각화 (§4.5) | `[재사용]` | B1 |
| FIG-B1 | 파라미터 민감도 2-패널 (§B.4) | `[재사용(좌)+신규 실행(우)]` | **B2** |
| TAB-1 | 데이터셋 통계 family 요약 (Table 1) | `[재사용+신규 측정(SMD 셀)]` | B1 |
| TAB-2 | main 비교 + protocol-effect 블록 (Table 2) — **의존 그래프 루트** | `[완주 대기+신규 실행]` | B1 |
| TAB-3 | ablation 4행 (Table 3) | `[재사용(행1·3)+신규 실행(행2·4)]` | B1 |
| TAB-4 | (TAB-2 하단 블록으로 흡수 완료 — 별도 페이지 없음, D-010 ①) | 흡수 | B1 기재 |
| TAB-A3 | 26 baseline 하이퍼파라미터 전수 (Table A.3) | `[재사용]` | **B2** |
| Table A.4 | per-entity 통계 — SMD per-machine 셀 3종 (부분) | `[신규 측정]` | **B2** |
| TAB-A6 | SWaT full/excl22 이중 조건 5지표 (Table A.6) | `[완주 대기+TAB-2 동일 소스]` | **B2** |
| TAB-A7 | 잔여 4지표 전수 (Table A.7) | `[완주 대기 — TAB-2 동일 소스]` | **B2** |
| TAB-A8 | CSMAD per-entity 109행 (Table A.8) | `[완주 대기]` | **B2** |
| TAB-B1 | contaminated-training 22종 + Δ (Table B.1) | `[신규 실행]` | **B2** |
| TAB-B2 | epoch-budget 민감도 (Table B.2) | `[신규 실행(부분 재사용)]` | **B2** |
| TAB-B3 | 추론 비용 3×3 (Table B.3) | `[신규 측정]` | **B2** |
| TAB-B4 | 확장 ablation 7행 (Table B.4) | `[재사용(no_fm)+신규 실행(3종)]` | **B2** |
| ALG-C1 | 학습 의사코드 검증 (Algorithm C.1) | `[제작 — 코드 대조]` | **B2** |
| NUM-001~031 | inline 수치 31건 — 파생 소스 단위 8그룹 (아래 그룹 표) | 그룹별 상이 | 그룹별 |
| TXT-001 | GPU 모델명 ×1개소 (§A.1) | `[신규 측정 — 확인]` | **B2** |
| TXT-002 | 코드 저장소 URL ×3개소 (Abstract·§A.1·§5) | `[결정 사항]` | **B2** |
| R-PROBE | GRL probing classifier (권고 — 원고 비반영, D-014 (b)) | `[신규 측정]` | **B2** |

**NUM 8그룹 요약** (31건 전수 — 각 그룹은 소스 실험이 완료되면 그룹 내 전 항목이 동시에 풀린다):

| 그룹 | NUM | 소스 | 분류 | 상세 페이지 |
|---|---|---|---|---|
| N-A (family 수 sync) | 001, 003, 004, 029 | 271canon 완주 + TAB-2 완성 → "six" | `[완주 대기]` | B1 / TAB-2 |
| N-B (baseline 총수 sync) | 002, 005, 030 | weak 4종 GPU 완주 → "26" (미완 시 "22" fallback) | `[신규 실행 의존]` | B1 / TAB-2 |
| N-C (Table 2 본 블록 파생) | 006–013 | TAB-2 완성본 집계 | `[집계만]` | B1 / TAB-2 |
| N-D (protocol-effect 파생) | 014–019 | standard-split run (+019는 contaminated run 공유) | `[신규 실행]` | B1 / TAB-2 |
| N-E (ablation 파생) | 020–023 / **024, 025** | TAB-3 / **TAB-B4** | 혼합 | B1 / **B2 TAB-B4** |
| N-F (sparsity 파생) | 026, 027 | FIG-3 sweep | `[신규 실행]` | B1 / FIG-3 |
| N-G (qualitative 파생) | 028 | FIG-4 제작 (=2) | `[재사용]` | B1 / FIG-4 |
| N-H (cost 파생) | **031** | TAB-B3 측정 (§5 "50×" sync) | `[신규 측정]` | **B2 TAB-B3** |

---

## 🚦 실행 우선순위 대시보드

우선순위 원칙: (1) 본문 핵심 표 의존 → (2) load-bearing 주장 의존 → (3) appendix 방어 실측 순.

### 신규 실행 11건 (GPU 학습 필요 — 우선순위순)

| # | 실험 | 예상 산출 (채워지는 placeholder) | 의존성·선행 조건 | 실행 지침 요약 |
|---|---|---|---|---|
| 1 | baseline SMD/SMAP/MSL 신규 실행 (anomaly-excised) | TAB-2 unsup 행, FIG-3 floor, NUM-008/009/011/016/019, TAB-A6/A7 | per-entity 정규화 적용 확인 (SMD 구버전 `3_20260312_*` 폐기 후 재실행; SMAP/MSL normalonly는 **미실행분 신규**) | `comparison/run_baseline_queue.py`, variant `normalonly` |
| 2 | weakly supervised 4종 GPU 전체 (contaminated-training) | TAB-2 그룹 6, NUM-013, sync 그룹 B="26", TAB-A6/A7 | 없음 (구현·CPU dry-test 완료) | Q1 variant, 50 epochs; **NRdetector 최우선** |
| 3 | standard clean-train split (CSMAD + 대표 baseline 2–3, 대표 3 데이터셋) | TAB-2 하단 블록, NUM-014~019 | **신규 loader variant 구현** (`*_standard` — prefix 미편입, 평가는 동일 test 후반); CSMAD는 동일 config·`use_grl=True` 유지 (자가 비활성, False 금지) | EXECUTION-TODO 항목 3 |
| 4 | ablation 행2 (w/o GRL, OD-exclusion 유지) | TAB-3 행2, NUM-023 | TAB-3 대표 열(NUM-020) 확정 | `use_grl=False anomaly_loss_weight=0.0` — dead-component(dynamic margin) 재활성 차단 |
| 5 | symmetric decoder (Teacher 2L) | TAB-B4 2행(symmetric+depth2), **NUM-024 (bullet 3 load-bearing)** | TAB-3 대표 열 확정 | `num_teacher_decoder_layers=2` |
| 6 | ablation 행4 (w/o OD) | TAB-3 행4, NUM-022 | TAB-3 대표 열 확정 | `use_output_discrepancy=False` — score는 자동 recon-only (`resolve_score_weights`, `scoring.py:105-106`·`249-253`); 표 각주로 명시 |
| 7 | label sparsity sweep (p∈{0.75, 0.5, 0.25, 0.1} × 2–3 데이터셋) | FIG-3, NUM-026/027 | `label_keep_ratio` 파라미터 신설 구현 (NoisyLabel 메커니즘 일반화; region 단위, seed 고정); p=1.0은 [271c] 재사용 | 8–12 run, 271 canon + override |
| 8 | contaminated-training 22종 (대표 3 family) | TAB-B1 (+NUM-019 보조) | Δ 산출은 TAB-2 확정 후; SWaT 차원 45 검증 | Q1 variant 큐 |
| 9 | w/o Teacher warmup (250→0) / Teacher depth 1 | TAB-B4 잔여 2행 | TAB-3 대표 열 확정 | `teacher_only_warmup_epochs=0` / `num_teacher_decoder_layers=1` |
| 10 | epoch-budget 50/100 (Anomaly Trans., TranAD) | TAB-B2 | CSMAD 축소분은 exp298/299 재사용 결정 (i)/(ii) 선행 | `baseline_common.py` epochs override |
| 11 | masking ratio sweep (ρ∈{0.05, 0.1, 0.2, 0.3}) | FIG-B1 우패널 | 큐 미등재 — 신규 등재 필요 (v5 전 32항목 `masking_ratio` override 0건) | `masking_ratio=<ρ>` × 대표 데이터셋, 4 run |

### 신규 측정/스크립트 5건 (학습 불필요 — 즉시 실행 가능)

| 작업 | 예상 산출 | 의존성 |
|---|---|---|
| SMD 28 machine 분할 통계 산출 (loader 산식 재사용) | TAB-1 SMD 셀, Table A.4 SMD 행, §4.1.1 "pending" 문구 해소 | 없음 — 즉시 가능 |
| 추론 비용 측정 (leave-one-out vs single-mask) | TAB-B3, NUM-031 (+§5 "50×" sync) | TXT-001 (하드웨어 표기) 선행 권장 |
| GPU 모델 확인 | TXT-001 | 없음 — 호스트 이력 확인 |
| 저장소 URL 확정 (게재 시) | TXT-002 ×3개소 | 공개 전 checklist (branch·범위·secret·재현 진입점) |
| **R-PROBE** — Student/Teacher hidden probe AUC 비교 | (원고 placeholder 없음 — rebuttal 대비) | 기본: [271c]만으로 가능; 확장 대조군: #4 완료 후 |

### 완주 대기 3건 (진행 중 — 271canon 잔여 SMD 6 / SMAP 49 / MSL 22)

| placeholder | 비고 |
|---|---|
| TAB-2 CSMAD 행 (SMD/SMAP/MSL avg) | **부분 완주 상태로 avg 집계 금지** (sync 그룹 A 보호) |
| TAB-A8 전체, TAB-A6/A7 CSMAD 행 | metadata 집계 스크립트 (단일 산출물 공유 + Table 2 일치 assert) |
| 그룹 N-A (NUM-001/003/004/029) | "six" 확정 조건 — 탈락 family 발생 시 §4.1.1 상수 일괄 수정 |

### 재사용 8건 (실행 불필요 — 추출/제작만)

| placeholder | 소스 |
|---|---|
| FIG-1, FIG-2, ALG-C1 | 다이어그램/의사코드 제작 (정본 대조) |
| FIG-4, NUM-028 | [271c] best checkpoint + `scoring.py` 추출 |
| FIG-B1 좌패널 (c sweep) | [271c] checkpoint 재채점 (c∈{1,2,4,8,16}, best epoch 고정) |
| TAB-1 (SMD 셀 제외) | EXPERIMENT_PROTOCOL_TRUTH §① 실값 — 이미 tex 반영 |
| TAB-A3 | `comparison/baseline_common.py` `MODEL_CONFIGS` 덤프 |
| TAB-3 행3, NUM-021 | exp287_unmask 완주분 |
| TAB-B4 w/o FM 행, NUM-025 | exp285_no_fm 완주분 |
| TAB-B2 CSMAD 축소 budget | exp298/exp299 완주분 (열 라벨 정합화 결정 필요) |

---

## 🕸️ 의존 그래프 — TAB-2가 루트다

TAB-2(main comparison)가 placeholder 의존 그래프의 루트다: NUM 13건(N-C 8 + N-D 6 중 다수), FIG-3의 floor, TAB-B1의 Δ 기준, appendix 결과 표 3종이 전부 이 표의 확정본에서 파생된다. 화살표는 "왼쪽이 끝나야 오른쪽을 채울 수 있다"로 읽는다.

```
[입력 4갈래]                          [루트]                [파생]
271canon 완주 (SMD6·SMAP49·MSL22) ─┐
baseline SMD/SMAP/MSL 신규 (#1) ───┤
weak 4종 GPU (#2) ─────────────────┼──▶ TAB-2 (Table 2) ──▶ NUM-006~013 (N-C 집계)
standard-split run (#3) ───────────┘         │
                                             ├──▶ 하단 protocol-effect 블록 ──▶ NUM-014~019 (N-D)
                                             │         └─(019의 contaminated 측은 TAB-B1 run과 소스 공유)
                                             ├──▶ FIG-3 점선 floor ──▶ NUM-026/027 (N-F; sweep run #7 별도)
                                             ├──▶ TAB-B1 Δ 기준값 (run #8과 결합)
                                             ├──▶ TAB-A6 / TAB-A7 (동일 실행 묶음 metadata에서 키 추가 추출)
                                             ├──▶ TAB-A8 (CSMAD per-entity; 블록 평균 = Table 2 셀 assert)
                                             └──▶ sync 그룹 A "six" (N-A) · sync 그룹 B "26" (N-B)

[독립 사슬]
TAB-3 대표 열 확정 (NUM-020) ──▶ TAB-B4 열 집합 동일 ──▶ NUM-021~025 (N-E)
                                  └─(run #4·#5·#6·#9가 행을 공급; depth2 = symmetric run 공유)
TAB-B3 측정 ──▶ NUM-031 (N-H) ──▶ §5 "approximately 50×" 표현 sync
TAB-1 SMD 측정 스크립트 ══ Table A.4 SMD 행 (동일 산출물 — 불일치 구조적 차단)
FIG-4 (NUM-028) · FIG-B1 좌패널: [271c]만으로 즉시 가능
TXT-001 ──▶ TAB-B3 하드웨어 표기 / TXT-002: 공개 checklist 후 3개소 일괄 치환
R-PROBE: [271c]만으로 기본형 가능 (확장 대조군만 run #4 의존; 원고 무변경)
```

**그래프가 주는 운영 결론 세 가지.** 첫째, 실행 #1–#3이 끝나기 전에는 N-C/N-D의 어떤 NUM도 본문에 선기입할 수 없다 — PSM처럼 이미 산출된 값([271c] `metrics.pak_auc_f1`)이 있어도 **표 전체 확정 전 선기입 금지**가 규칙이다(A8). 둘째, 학습이 필요 없는 작업(측정 5건 + 재사용 8건)은 의존 그래프 바깥이거나 말단이므로 **지금 병렬로 소화 가능**하다. 셋째, 집계에서 Exathlon·Simulation은 절대 배제한다(R33) — 기존 Notion RankAvg류 수치는 재계산 전까지 인용 금지(FEEDBACK-3).

---

## 📂 발행 안내

본 문서(B2)는 appendix·잔여 placeholder 10페이지 + 본 개요를 담당하고, 본문 figure/table(FIG-1~4, TAB-1~3 및 그 NUM 그룹)은 B1 문서가 담당한다. Notion 발행 시 각 `<!-- PAGE: ... -->` 블록을 하위 페이지 1개로 만들고, 본 OVERVIEW 블록은 부모 페이지 본문에 넣는다. 긴 보고서는 반드시 notion-create-pages 경로로 발행한다(마크다운→블록 파싱; update-page의 insert_content는 렌더링 파손). 조건 명칭은 R24 개명 후 표기만 사용한다 — 코드명 normalonly → "anomaly-excised condition", 코드명 full → "contaminated-training condition". 수치 발명 금지(A8)·Gaussian smoothing 언급 금지(R34)는 전 페이지 공통이다.
