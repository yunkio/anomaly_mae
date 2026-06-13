---
phase: 8
agent: notion-singlepage-builder
directives: [R3]
last_modified: 2026-06-13
source_inputs:
  - paper/08_final_audit/NOTION_ENRICHED_B1_body.md
  - paper/08_final_audit/NOTION_ENRICHED_B2_appendix.md
note: |
  단일 Notion 페이지 발행용. frontmatter 아래(H1부터)가 본문이다.
  사실·수치·실행 지침·영문 캡션 무변경 — 한국어 표현·구성만 정제. 수치 창작 금지(A8).
---

# CSMAD 논문 placeholder 실험·Figure 설계서 (단일 통합 페이지)

<callout icon="📐" color="blue_bg">
이 페이지는 TSMAE/CSMAD 논문 원고에 의도적으로 **비워둔 자리(placeholder)** — figure, table, 인라인 수치, 알고리즘, 환경 정보 — 를 **무엇을·왜·어떻게 채우는가**로 완성한 단일 설계서다. 원고의 각 placeholder는 "어떤 reviewer 공격을 막기 위해, 어떤 데이터·config·스크립트로, 어떤 형태의 산출을 만드는가"라는 질문을 남겨 둔 자리이며, 이 페이지 하나만 보고도 실험 설계와 figure 제작에 곧장 착수할 수 있도록 그 답을 전부 담았다.

**읽는 법**: 먼저 **§0 실행 대시보드**로 "오늘 무엇부터 손댈 수 있는가"를 정한다(학습 없이 지금 가능한 일 / GPU가 필요한 신규 실행 / 완주를 기다리는 일이 분리되어 있다). 그다음 해당 placeholder의 토글을 펼치면 **목적·기대·실험 설계·형태·영문 캡션·주의사항**이 통일된 순서로 들어 있다. **§6**은 본문에 흩어진 인라인 수치 31건을 소스 그룹별로 묶어, 어느 실험이 끝나면 어느 수치가 한꺼번에 풀리는지를 보여준다.
</callout>

<table_of_contents/>

---

## 0. 한눈에 보기 — 실행 대시보드

<callout icon="🧭" color="gray_bg">
우선순위 원칙은 **(1) 본문 핵심 표 의존 → (2) load-bearing 주장 의존 → (3) appendix 방어 실측** 순이다. 의존 그래프의 루트는 **TAB-2(main comparison)** 이며, 인라인 수치 다수·FIG-3 floor·appendix 결과 표 3종이 모두 이 표의 확정본에서 파생된다.
</callout>

### ① 소스 분류 라벨 — "이 빈칸을 채우려면 무엇이 필요한가"

모든 placeholder에는 소스 분류 라벨이 붙어 있다. 라벨은 작업 성격을 한 단어로 답한다.

| 라벨 | 의미 | 전형적 작업 |
|---|---|---|
| **[재사용]** | 기존 실험 결과 또는 코드 상수에서 **추출만** — 학습 불필요 | metadata 집계, 코드 덤프, checkpoint 재채점 |
| **[완주 대기]** | 진행 중인 `271canon`·큐의 잔여 entity **완주 후 집계** (잔여: SMD 6 · SMAP 49 · MSL 22 — 2026-06-11 실측) | 완주 모니터링 → 집계 스크립트 |
| **[신규 실행]** | **새 학습 실험**이 필요 (큐 등재 또는 신규 스크립트) | 큐 항목 작성 → GPU 실행 → 집계 |
| **[신규 측정]** | 학습 없이 **측정·스크립트 1회**로 해결 (통계 추출, 비용 측정, 확인) | 1회성 스크립트 |
| **[제작]** | 실험 무관 — **다이어그램/의사코드 제작·검증**만 필요 | TikZ/벡터 제작, 코드 대조 |

### ② 오늘 무엇부터? — 3구역

<callout icon="🟢" color="green_bg">
**즉시 가능 (학습 불필요) — 의존 그래프 바깥 또는 말단이라 지금 병렬 소화 가능**
**측정·스크립트 5건**: ① SMD 28 machine 분할 통계 산출(loader `//2` 산식 재사용 → TAB-1 SMD 셀 + Table A.4 SMD 행 + §4.1.1 "pending" 해소, 의존 없음) ② 추론 비용 측정 leave-one-out vs single-mask(→ TAB-B3, NUM-031; TXT-001 선행 권장) ③ GPU 모델 확인(→ TXT-001; 호스트 이력 확인) ④ 저장소 URL 확정(→ TXT-002 ×3개소; 공개 전 checklist 후) ⑤ **R-PROBE** Student/Teacher hidden probe AUC 비교(원고 무변경, rebuttal 대비; [271c]만으로 기본형 가능).
**재사용 8건**: FIG-1·FIG-2·ALG-C1(제작·정본 대조) / FIG-4·NUM-028([271c]+`scoring.py` 추출) / FIG-B1 좌패널 c sweep([271c] 재채점, best epoch 고정) / TAB-1 SMD 외 셀(이미 tex 반영) / TAB-A3(`MODEL_CONFIGS` 덤프) / TAB-3 행3·NUM-021(exp287_unmask) / TAB-B4 w/o FM 행·NUM-025(exp285_no_fm) / TAB-B2 CSMAD 축소 budget(exp298/299).
</callout>

<callout icon="🔴" color="red_bg">
**GPU 신규 실행 11건 (우선순위순) — 루트부터 풀어야 파생이 열린다**
**#1** baseline SMD/SMAP/MSL 신규(anomaly-excised) → TAB-2 unsup 행·FIG-3 floor·NUM-008/009/011/016/019·TAB-A6/A7 / **#2** weakly supervised 4종 GPU 전체(contaminated-training, NRdetector 최우선) → TAB-2 그룹 6·NUM-013·sync "26"·TAB-A6/A7 / **#3** standard clean-train split(CSMAD+대표 baseline 2–3, 대표 3 데이터셋; 신규 loader variant 필요) → TAB-2 하단 블록·NUM-014~019 / **#4** ablation 행2 w/o GRL(OD-exclusion 유지) → TAB-3 행2·NUM-023 / **#5** symmetric decoder Teacher 2L → TAB-B4 2행·**NUM-024(bullet 3 load-bearing)** / **#6** ablation 행4 w/o OD → TAB-3 행4·NUM-022 / **#7** label sparsity sweep → FIG-3·NUM-026/027 / **#8** contaminated-training 22종(대표 3 family) → TAB-B1 / **#9** w/o warmup·Teacher depth 1 → TAB-B4 잔여 2행 / **#10** epoch-budget 50/100 → TAB-B2 / **#11** masking ratio sweep ρ → FIG-B1 우패널.
</callout>

<callout icon="🟡" color="yellow_bg">
**완주 대기 3건 — `271canon` 잔여 SMD 6 / SMAP 49 / MSL 22 진행 중**
① TAB-2 CSMAD 행(SMD/SMAP/MSL avg) — **부분 완주 상태로 avg 집계 금지**(sync 그룹 A 보호) ② TAB-A8 전체 + TAB-A6/A7 CSMAD 행 — metadata 집계 스크립트(단일 산출물 공유 + Table 2 일치 assert) ③ 그룹 N-A(NUM-001/003/004/029) — "six" 확정 조건, 탈락 family 발생 시 §4.1.1 상수 일괄 수정.
</callout>

### ③ 우선순위 대시보드 — 신규 실행 11건 (작업 / 예상 산출 / 의존성)

| # | 실험 | 예상 산출 (채워지는 placeholder) | 의존성·선행 조건 | 실행 지침 요약 |
|---|---|---|---|---|
| 1 | baseline SMD/SMAP/MSL (anomaly-excised) | TAB-2 unsup 행, FIG-3 floor, NUM-008/009/011/016/019, TAB-A6/A7 | per-entity 정규화 확인 (SMD 구버전 `3_20260312_*` 폐기 후 재실행; SMAP/MSL는 **미실행분 신규**) | `comparison/run_baseline_queue.py`, variant `normalonly` |
| 2 | weakly supervised 4종 GPU 전체 (contaminated-training) | TAB-2 그룹 6, NUM-013, sync "26", TAB-A6/A7 | 없음 (구현·CPU dry-test 완료) | Q1 variant, 50 epochs; **NRdetector 최우선** |
| 3 | standard clean-train split (CSMAD + 대표 baseline 2–3) | TAB-2 하단 블록, NUM-014~019 | **신규 loader variant 구현** (`*_standard`); CSMAD `use_grl=True` 유지 (자가 비활성, False 금지) | EXECUTION-TODO 항목 3 |
| 4 | ablation 행2 (w/o GRL, OD-excl. 유지) | TAB-3 행2, NUM-023 | TAB-3 대표 열(NUM-020) 확정 | `use_grl=False anomaly_loss_weight=0.0` (dead-component 재활성 차단) |
| 5 | symmetric decoder (Teacher 2L) | TAB-B4 2행, **NUM-024 (bullet 3 load-bearing)** | TAB-3 대표 열 확정 | `num_teacher_decoder_layers=2` |
| 6 | ablation 행4 (w/o OD) | TAB-3 행4, NUM-022 | TAB-3 대표 열 확정 | `use_output_discrepancy=False` (score 자동 recon-only; 각주 명시) |
| 7 | label sparsity sweep (p∈{0.75,0.5,0.25,0.1}×2–3 데이터셋) | FIG-3, NUM-026/027 | `label_keep_ratio` 신설 (NoisyLabel 일반화); p=1.0은 [271c] 재사용 | 8–12 run, 271 canon + override |
| 8 | contaminated-training 22종 (대표 3 family) | TAB-B1 (+NUM-019 보조) | Δ 산출은 TAB-2 확정 후; SWaT 차원 45 검증 | Q1 variant 큐 |
| 9 | w/o Teacher warmup (250→0) / Teacher depth 1 | TAB-B4 잔여 2행 | TAB-3 대표 열 확정 | `teacher_only_warmup_epochs=0` / `num_teacher_decoder_layers=1` |
| 10 | epoch-budget 50/100 (Anomaly Trans., TranAD) | TAB-B2 | CSMAD 축소분 exp298/299 재사용 결정 (i)/(ii) 선행 | `baseline_common.py` epochs override |
| 11 | masking ratio sweep (ρ∈{0.05,0.1,0.2,0.3}) | FIG-B1 우패널 | 큐 미등재 — 신규 등재 필요 (v5 전 32항목 `masking_ratio` override 0건) | `masking_ratio=<ρ>` × 대표 데이터셋, 4 run |

### ④ 의존 그래프 — TAB-2가 루트다

TAB-2(main comparison)가 placeholder 의존 그래프의 루트다. 인라인 수치 13건(N-C 8 + N-D 6 중 다수), FIG-3의 floor, TAB-B1의 Δ 기준, appendix 결과 표 3종이 전부 이 표의 확정본에서 파생된다. 화살표는 "왼쪽이 끝나야 오른쪽을 채운다"로 읽는다.

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

<callout icon="⚙️" color="gray_bg">
**그래프가 주는 운영 결론 셋.** ① 실행 #1–#3 완료 전에는 N-C/N-D의 어떤 수치도 본문 선기입 금지 — PSM처럼 이미 산출된 값([271c] `metrics.pak_auc_f1`)이 있어도 **표 전체 확정 전 선기입 금지**가 규칙이다(A8). ② 학습 불필요 작업(측정 5건 + 재사용 8건)은 의존 그래프 바깥/말단이므로 **지금 병렬 소화 가능**. ③ 집계에서 **Exathlon·Simulation은 절대 배제**(R33) — 기존 Notion RankAvg류 수치는 재계산 전까지 인용 금지(FEEDBACK-3).
</callout>

---

## 1. 본문 Figure

### FIG-1 — 학습 패러다임 비교 다이어그램 (Setting-comparison diagram) {toggle="true"}

	| 유형 | 소스 분류 | 우선순위 | 의존성 |
	|---|---|---|---|
	| 개념도 (§1) | **[제작]** — 실험 데이터 없음 | 재사용 묶음 | 없음 (용어 동기화 의무만) |

	<callout icon="💡" color="blue_bg">
	**한 줄 요약** — 오염된 학습 스트림이라는 동일한 입력 위에서 unsupervised / label-aware filtering / CSMAD 세 패러다임이 라벨을 각각 "무시 / 절제 / 통합"하는 방식을 한 장으로 대비시켜, 논문 전체의 문제 설정과 핵심 용어를 시각적으로 고정하는 개념도다.
	</callout>

	**위치·크기** — §1 Introduction, observation 문단 직후 (`sec1_intro.tex`, `\label{fig:setting}`). full-width, 약 5 cm (≈0.40p).

	#### 🎯 목적과 의도

	이 그림은 논문의 중심 논제 — "labeled anomaly는 비지도 방법에게는 오염이지만, 그것을 학습 신호로 통합할 수 있는 방법에게는 가치 있는 정보다" — 를 본문 텍스트보다 먼저, 그리고 텍스트 없이도 전달하기 위해 존재한다. §1의 관찰 문단(labeled anomaly가 드러내는 세 가지 학습 신호 (a)/(b)/(c))이 끝난 직후에 배치되어, 독자가 contribution bullet을 읽기 전에 "기존 패러다임 두 가지가 이 신호를 어떻게 버리는가"를 눈으로 확인하게 만든다.

	논증 구조에서 이 그림이 맡는 역할은 두 가지다. 첫째, 중앙 패널(label-aware filtering)은 "라벨이 있으면 그냥 오염 구간을 걸러내면 되지 않는가"라는 가장 자연스러운 reviewer 반문에 대한 시각적 선제 답변이다. 본문 §1의 문장("the best a label-aware variant can do is exclude confirmed anomaly windows ... filtering contamination rather than learning from it")이 말하는 한계 — 오염은 제거되지만 라벨 정보 자체는 폐기된다 — 를 패널 하나로 보여주며, 이는 곧 본문 비교 실험의 anomaly-excised condition(Table 2의 main 조건)이 왜 "비지도 방법에게 라벨의 최선 활용을 제공하는 조건"인지(블루프린트 §14 논거 ③, R12 논리)와 직결된다.

	둘째, 세 패널 상단의 입력 스트림 띠를 **세 패널에서 동일하게** 그리는 것 자체가 방어 장치다. 블루프린트 §15의 leakage 공격 시나리오("test-prefix 편입은 test label로 학습하는 것")에 대한 방어 논거 ③(모든 비교 모델이 동일한 데이터를 받는다)을, 비교 조건을 설명하기도 전에 그림의 전제로 깔아둔다. 또한 우측 패널의 세 갈래 화살표는 contribution bullet 2의 세 용어(*anomaly-priority masking*, *loss bifurcation*, *gradient-reversal suppression*)를 글자 단위로 고정하는 anchor로, 이후 §3·§4의 모든 서술이 이 세 명칭으로 수렴한다.

	#### 🏁 목표와 기대 결과

	실험이 없는 제작물이므로 성공 기준은 전달력과 정합성으로 정의한다. (1) 비전문 독자가 캡션을 읽지 않고도 세 패널의 차이 — 라벨이 보이지 않음 / 라벨로 구간을 잘라냄 / 라벨이 세 경로로 학습에 흘러 들어감 — 를 읽을 수 있을 것. (2) 그림 내 모든 용어가 본문 표기와 글자 단위로 일치할 것(아래 주의사항). (3) 입력 스트림 띠의 붉은(anomaly) 비율이 실제 train anomaly ratio(0.5–6.2%)를 연상시키는 소수 구간일 것 — 절반이 붉은 그림은 설정 자체를 왜곡한다.

	기대와 다른 패턴이라는 개념은 개념도에는 적용되지 않으나, 대응 규칙은 있다. 만약 Phase 8 채움 과정에서 비교 조건 명칭이나 contribution bullet 2의 용어가 변경되면(예: R24류 개명 재발생), 이 그림은 본문과 동시에 갱신해야 하며 그림만 구표기로 남는 것은 허용되지 않는다.

	#### 🧪 실험 내용과 설계

	**[제작] — 실험 소스 없음.** 학습·측정이 전혀 필요 없고, 다이어그램 제작과 본문 대조 검증만 수행한다.

	- **권장 제작 경로**: TikZ 직접 작성(elsarticle 빌드와 폰트가 일치해 가장 안전) 또는 외부 벡터 도구로 제작 후 PDF 삽입.
	- **공통 입력 띠**: 세 패널 상단에 동일한 입력 스트림 띠(정상 구간 + 붉은 라벨 anomaly 구간)를 그린다. 붉은 구간은 소수(시각적으로 수 % 수준)만 칠한다.
	- **패널 구성**: (좌) unsupervised — 라벨이 모델에 보이지 않아 순수 오염원으로 작용. (중) label-aware filtering — 라벨된 anomaly window를 학습 전에 절제(= anomaly-excised condition; §4.1.4 상호참조). (우) CSMAD — 라벨이 masking·loss·gradient 세 경로로 학습에 유입.

	#### 📊 구성과 형태

	가로 3-패널. 각 패널의 수직 구성은 동일하게 **상단 입력 스트림 띠 → 모델 박스 → 라벨 흐름 글리프**.

	| 패널 | 라벨 흐름 글리프 | 핵심 시각 메시지 |
	|---|---|---|
	| 좌 (unsupervised) | 라벨이 모델에 닿지 않음 (무시됨 표시) | anomaly가 all-normal 가정의 오염원으로만 작용 |
	| 중 (label-aware filtering) | 라벨이 데이터를 잘라내는 가위/절제 표시 | 오염 제거 = 라벨 정보 폐기 |
	| 우 (CSMAD) | 라벨에서 세 갈래 화살표 → masking / loss / gradient | 오염을 학습 신호로 전환 |

	용어는 §1 contribution bullet 2의 표기와 글자 단위로 일치: *anomaly-priority masking*, *loss bifurcation*, *gradient-reversal suppression*.

	#### 📝 캡션 (영문 확정본)

	```latex
	Three training paradigms for multivariate time series anomaly detection under a
	contaminated training stream.
	\textit{Left (unsupervised)}: labeled anomalies are invisible to the model and act purely
	as contamination of the all-normal assumption.
	\textit{Middle (label-aware filtering)}: labeled anomaly windows are excised before
	unsupervised training (the anomaly-excised condition; \S\ref{sec:baselines}) ---
	contamination is removed but the label information is discarded.
	\textit{Right (CSMAD)}: labeled anomalies are integrated into training through three
	paths --- anomaly-priority masking, loss bifurcation, and gradient-reversal suppression
	--- turning contamination into a learning signal.
	```

	#### ⚠️ 주의사항과 의존성

	- 중앙 패널 명칭은 R24 개명 후의 **"anomaly-excised condition"만** 사용한다. 구표기(Q3, normalonly)는 그림·라벨 어디에도 금지.
	- 붉은 구간 비율은 실제 train AR(0.5–6.2%)을 연상시키는 소수 구간만 — 설정 왜곡 금지.
	- 세 패널의 입력 스트림 띠는 픽셀 단위로 동일해야 한다(공정성 논거의 시각화).
	- 그림 용어 ↔ 본문 bullet 2 ↔ §3 소절 제목의 3중 일치 검증을 제작 완료 시점에 1회 수행.

	#### 🔢 연결된 수치 placeholder

	없음 — 이 그림에서 파생되는 NUM placeholder는 없다. 단, 용어 동기화 의무(contribution bullet 2의 3-path 명칭)는 위 주의사항대로 적용된다.


### FIG-2 — CSMAD 아키텍처 개요 (Architecture overview) {toggle="true"}

	| 유형 | 소스 분류 | 우선순위 | 의존성 |
	|---|---|---|---|
	| 아키텍처 다이어그램 (§3.2) | **[제작]** — 구조 상수는 271_CONFIG_TRUTH r4 §VIII에서만 인용 | 재사용 묶음 | 271_CONFIG_TRUTH r4 §VIII |

	<callout icon="💡" color="blue_bg">
	**한 줄 요약** — 학습(좌)·추론(우) 2-패널로 CSMAD의 다섯 기능 블록과 라벨 유입 경로, gradient 차단 구조를 한 장에 담아 §3 전체의 지도 역할을 하는 아키텍처 다이어그램이다.
	</callout>

	**위치·크기** — §3.2 도입부 (`sec3_method.tex`, `\label{fig:architecture}`). full-width, 5 cm = 0.40p (integrator 가정; Phase 7에서 가독성 확인).

	#### 🎯 목적과 의도

	§3의 다섯 소절(문제 정식화 → 마스킹 → 비대칭 디코더 → 라벨 유도 학습 → 채점)은 각각 한 component를 다루기 때문에, 독자가 전체 데이터 흐름을 머리에 그리지 못하면 각 손실 항이 어디에 붙는지 길을 잃는다. 이 그림은 §3.2 도입부에서 그 전체 지도를 제공한다. 입력 윈도가 패치로 갈라져 어느 블록을 거치고, 네 가지 손실(L_recon, L_OD, L_FM, L_cls)이 어느 연결선에서 발생하며, 추론 시에는 무엇이 꺼지는지를 좌우 패널 대비로 보여준다.

	이 그림은 동시에 두 개의 확정된 reviewer 공격에 대한 방어 장치다. 첫째, **"GRL이 Student를(나아가 표현 전체를) 망가뜨리지 않는가"**(블루프린트 §15) — Student의 latent 입력에 stop-gradient 기호(⊥)를 명시해, encoder가 Teacher의 재구성 목적만으로 학습되고 GRL gradient로부터 완전히 차단된다는 §3.2 본문 주장("The adversarial signal therefore cannot corrupt the normal-pattern representation")을 시각적으로 고정한다. 둘째, **GRL 부착 지점의 모호성**(블루프린트 ADV BLK-002 — 과거 리뷰에서 실제로 지적된 재발 지점) — "Student decoder final-layer hidden states, **before output projection**"이라는 명시 라벨을 그림 안에 넣어, 부착 지점을 본문·부록·rebuttal이 공유하는 단일 사실로 만든다. 부수적으로 GRL 박스의 점선 + "training only" 표기는 "추론 시 라벨 미사용"이라는 문제 설정의 약속(§3.1)을 그림 차원에서 반복한다.

	마지막으로 우측 추론 패널은 §3.6의 leave-one-out 채점(50패턴 batch-병렬, σ_i → a_t 평균 집계)을 묘사함으로써, §5 결론의 비용 한계 서술(약 50× forward 연산)과 부록 비용 표(TAB-B3)가 가리키는 대상을 미리 정의한다.

	#### 🏁 목표와 기대 결과

	성공 기준: (1) 그림만 보고 §3의 기호(o^T_i, o^S_i, h^enc, σ_i, a_t)와 손실 연결이 본문 수식과 1:1로 대응될 것. (2) 필수 표기 3건(아래 형태 절의 ⓐⓑⓒ)이 전부 들어 있을 것 — 특히 ⓒ(GRL 부착 지점 라벨)는 생략 시 리뷰 재발 지점이다. (3) 모든 구조 상수(d_model=512, nhead=8, encoder 4L / Teacher 3L / Student 2L, N=50, ρ=0.15 → |M|=8, L=500, patch size 10)가 271_CONFIG_TRUTH r4 §VIII과 일치할 것.

	기대와 다른 상황에 대한 대응: 이 그림은 실험 결과를 담지 않으므로 결과 의존이 없다. 다만 부록 ablation(TAB-B4)의 symmetric decoder run 결과에 따라 contribution bullet 3의 주장 강도가 하향될 수 있는데(Phase 6 규칙), 그 경우에도 이 그림의 구조 자체(3L/2L 비대칭)는 사실 서술이므로 변경 불필요 — 캡션·본문 문구만 조정 대상이다.

	#### 🧪 실험 내용과 설계

	**[제작] — 실험 소스 없음.** 모든 구조 상수는 271_CONFIG_TRUTH r4 §VIII에서 그대로 가져온다. 이 정본 외 출처(코드 default, Notion 스냅샷, 발표자료)에서 수치를 가져오는 것은 금지 — 과거 batch_size(512 vs 1024), d_model(dynamic vs 512 고정) 불일치 사고가 전부 비정본 인용에서 발생했다.

	좌패널(학습)에 담을 데이터 흐름: 윈도(L=500) → N=50 패치 → anomaly-priority masking이 |M|=8 패치를 가림(anomaly 패치 우선) → visible 42패치만 encoder 통과 → 디코더 앞에서 mask token 삽입(Teacher/Student 별도 토큰) → 손실 연결선 4종: L_recon(Teacher 출력), L_OD·L_FM(Teacher↔Student, 정상 masked 패치만), L_cls(GRL classifier → window label).

	우패널(추론): GRL branch 비활성, leave-one-out masking 50패턴을 batch 차원으로 병렬 처리, per-patch score σ_i → point score a_t 평균 집계.

	#### 📊 구성과 형태

	다섯 색상 블록: (1) Patch Embedding(linear), (2) 공유 Transformer Encoder(4L), (3) Teacher Decoder(3L — 진한 색·깊게), (4) Student Decoder(2L — 연한 색·얕게), (5) GRL + AnomalyClassifierHead.

	**필수 표기 3건 (생략 금지)**:

	| # | 표기 | 이유 |
	|---|---|---|
	| ⓐ | GRL 박스 = 점선 + "**training only**" 라벨 | 추론 시 라벨·GRL 미사용 약속의 시각화 |
	| ⓑ | Student latent 입력에 stop-gradient 기호 ⊥ | encoder가 Teacher recon으로만 학습됨 — §15 GRL 방어 |
	| ⓒ | GRL 부착 지점 명시 라벨: "Student decoder final-layer hidden states, **before output projection**" | ADV BLK-002 — 생략 시 리뷰 재발 지점 |

	#### 📝 캡션 (영문 확정본)

	```latex
	CSMAD architecture overview.
	\textit{Left panel (training)}: the input window is split into $N$ patches;
	anomaly-priority masking withholds $|M|$ patches (anomalous patches masked first).
	Visible patches enter the shared Transformer encoder; mask tokens are inserted before
	each decoder.
	The Teacher decoder (darker, deeper) produces reconstructions $\{o^{\mathrm{T}}_i\}$;
	the Student decoder (lighter, shallower) produces $\{o^{\mathrm{S}}_i\}$.
	An AnomalyClassifierHead with gradient reversal (dashed box, labeled \textbf{training only})
	is applied to the Student's final-layer hidden states before the output projection.
	Loss connections: $L_{\mathrm{recon}}$ from Teacher outputs; $L_{\mathrm{OD}}$ and
	$L_{\mathrm{FM}}$ between Teacher and Student on normal masked patches; $L_{\mathrm{cls}}$
	from classifier head to window label.
	The encoder receives no gradient from Student or GRL (stop-gradient $\perp$).
	\textit{Right panel (inference)}: GRL branch inactive; leave-one-out masking patterns
	batch-parallelized; per-patch scores $\sigma_i$ averaged to point-level scores $a_t$.
	```

	#### ⚠️ 주의사항과 의존성

	- 기호는 Table C.2 notation 정본을 따른다 — point score는 s_t가 아니라 **a_t** (v2-r3 정정 반영됨).
	- 학습 좌패널에 warmup(0-based epoch 0–249 동안 Student 학습 경로 forward skip)을 그릴 의무는 없으나, 그릴 경우 "frozen"이 아니라 "**forward skipped (training path)**"가 정확한 서술이다 (271_CONFIG_TRUTH r4 §VIII Training — 평가 경로는 full forward라는 구분 포함).
	- λ를 그림에 표기할 경우 이중 구조(손실 가중 λ_GRL vs 반전 계수 λ_rev)를 단일 λ로 합쳐 쓰지 말 것 — 표기가 번잡하면 그림에서는 생략하고 §3.4 본문에 위임하는 편이 안전하다.

	#### 🔢 연결된 수치 placeholder

	없음 — 이 그림에서 파생되는 NUM placeholder는 없다. 구조 상수의 단일 원천은 271_CONFIG_TRUTH r4 §VIII이며, §4.1.2 Implementation Details의 동일 상수 서술과 일치해야 한다.


### FIG-3 — 라벨 희소화 sweep (Label sparsity sweep) · ★ 미구현 실험 (R32) {toggle="true"}

	| 유형 | 소스 분류 | 우선순위 | 의존성 |
	|---|---|---|---|
	| 결과 곡선 (§4.4) | **[신규 실행]** — 전용 파라미터·스크립트 부재 (`label_ratio`/`sparsity` grep 0건) | 신규 실행 #7 | TAB-2 floor + `label_keep_ratio` 신설 |

	<callout icon="💡" color="blue_bg">
	**한 줄 요약** — 학습 시 라벨이 제공되는 anomaly region 비율 p를 1.0에서 0.1까지 낮추며 CSMAD의 성능 곡선을 그려, "라벨이 희소해져도 점진적으로만 열화하고 unsupervised floor 아래로 떨어지지 않는다"는 abstract·결론의 핵심 주장을 정량적으로 뒷받침하는 그림이다.
	</callout>

	**위치·크기** — §4.4 Results 문단 직후 (`sec4_experiments.tex`, `\label{fig:sparsity}`). ~4 cm ≈ 0.33p.

	#### 🎯 목적과 의도

	main 실험(Table 2)은 train 구간의 모든 anomaly에 라벨이 있는 **라벨 가용성 상한 케이스**다. 그런데 논문의 문제 설정(§3.1, R11)은 "대부분 unlabeled + 소수 labeled"라는 일반 케이스를 가정한다 — 실제 운영 로그는 발생한 fault의 일부만 기록하기 때문이다. 이 간극을 메우지 않으면 "main 결과는 모든 라벨이 주어진 비현실적 조건의 산물"이라는 공격에 노출되고, 더 나아가 §15의 "PU learning이 아닌데 PU라 부른다" 류의 설정 공격에도 취약해진다(블루프린트의 3단 구조 — 설정/상한 구현/일반 케이스 검증 — 에서 이 그림이 세 번째 기둥이다). FIG-3은 라벨 비율 p를 내리며 상한 케이스에서 일반 케이스로, 그리고 비지도 극한으로 연속적으로 이동하는 곡선을 보여줌으로써 설정의 일반성을 직접 검증한다.

	이 그림이 뒷받침하는 본문 주장은 명확히 두 개다. 첫째, abstract와 §5 결론의 "detection performance degrades gracefully ... remaining above the unsupervised floor" — 이 문장의 **유일한 정량 근거**가 이 그림이다. 둘째, §4.4의 "Why graceful degradation is expected" 문단이 제시하는 구조적 논거 3가지(anomaly-priority masking은 labeled 패치에만 작용 / GRL은 batch에 labeled positive가 없으면 손실 자체가 미계산 / 재구성 오차는 라벨-무관 신호)가 실제 곡선과 일치하는지의 검증대다. 점선 floor(각 데이터셋의 best unsupervised baseline)는 "라벨이 거의 없어도 비지도 방법보다 나쁘지 않다"를 시각화해, CSMAD를 도입할 때의 하방 위험이 없음을 보인다.

	블루프린트 §6.8이 명시하듯 이 sweep은 NRdetector의 label-noise sweep과 축의 의미가 다르다(라벨 희소율 vs 잘못된 세그먼트 라벨 비율) — 본문에 1문장 구분이 이미 들어가 있으므로, 그림 설계가 이 구분을 흐리면 안 된다(라벨을 지우는 것이지 데이터를 지우거나 라벨을 틀리게 만드는 것이 아니다).

	#### 🏁 목표와 기대 결과

	**입증하려는 것**: p가 감소할 때 성능이 (i) 연속적·점진적으로 감소하고(절벽형 붕괴 없음), (ii) p→0 부근에서 해당 데이터셋의 unsupervised floor에 접근하되 그 아래로 유의미하게 떨어지지 않으며, (iii) p=1.0 점이 main 설정([271c]의 해당 entity 값)과 정확히 일치한다는 것. (iii)은 기대가 아니라 **검산 조건**이다 — 불일치하면 sweep 파이프라인에 결함이 있는 것이다.

	**기대와 다른 패턴이 나오면**: 곡선이 비단조이거나 특정 p에서 급락하면, 우선 NUM-027의 서술어 후보(gradually/monotonically)를 둘 다 버리고 §4.4 Results 문장을 실제 형상에 맞게 재작성한다(A8 — 곡선 확정 전 서술어 선점 금지). 동시에 "Why graceful degradation is expected" 문단의 구조 논거와 모순되는지 점검한다 — 예컨대 급락 지점이 "batch 내 labeled positive 소멸"과 일치한다면 그것은 논거 2의 예측 범위 안이므로 해석을 보강하면 되고, 논거와 정면 충돌하면 해당 문단 자체를 수정해야 한다. floor 아래로 떨어지는 점이 관찰되면 "without falling below the unsupervised floor" 문장은 유지 불가 — 사실대로 보고하고 한계로 서술한다. 어느 경우든 그림과 본문 서술을 침묵 불일치 상태로 두는 것은 금지다.

	#### 🧪 실험 내용과 설계

	**[신규 실행]** — 전용 파라미터가 코드에 없으므로 소규모 구현 후 실행한다. 단, p=1.0 점은 main 설정과 동일하므로 **[271c] 재사용**(재학습 금지 — 그 점만 추출).

	**구현 — 재사용할 기존 메커니즘 2개 (새로 발명하지 말 것)**:

	1. `mae_anomaly/datasets/noisy.py`의 `NoisyLabelSlidingWindowDataset` — 학습 split에서만 변형 라벨을 반환하고 평가에는 원본 라벨을 쓰는 구조(`use_noisy_labels = (split=='train')`)가 이미 있다. 희소화를 "학습 입력에만" 주입하는 정확한 골격.
	2. `scripts/run_base_experiments.py:397-416`의 `apply_normal50_noise` — train 구간 anomaly **region 단위** 50% 무작위 재라벨(seed=123)의 기존 구현. 이것을 비율 p로 일반화한 `apply_label_sparsity(regions, p, seed)`를 만들고, config에 `label_keep_ratio: float = 1.0`을 추가한다 — **키워드 전용, 기본 1.0 = 현행과 비트 동일** (CLAUDE.md API 체크리스트 2항: 행동을 바꾸는 새 필드의 침묵 기본값 금지 원칙과 정합).

	**조작 단위와 의미**: region 단위 무작위 선택(점 단위 아님) — "기록된 fault 사건" 개념과 일치하며 원고 §4.4 Design 문단("region granularity, as operational logs record faults")과 합치한다. 미선택 region은 **데이터는 train에 그대로 남기고 라벨만 0으로** 둔다(절제 아님 — unlabeled anomaly로 잔류시키는 것이 실험의 핵심). seed 고정: region 선택 seed=123 계열, p별 동일 seed.

	**라벨 영향 경로 확인**: force_mask_anomaly의 우선순위, GRL classifier target, OD 손실의 정상/이상 분기 — 세 경로 전부 `point_labels`를 경유하므로 NoisyLabel 주입 한 곳으로 일괄 제어된다 (EXPERIMENT_PROTOCOL_TRUTH §⑦ 실측). 별도 경로별 처리 불필요.

	**실행 매트릭스**: 대표 데이터셋 2–3개(NUM-026; 권장 SWaT excl22 + PSM, 여유 시 WaDi A1 추가) × p ∈ {0.75, 0.5, 0.25, 0.1} = **8–12 run**. 각 run은 271 canon config 그대로(500 epochs, seed 42), `config_override`에 `label_keep_ratio=<p>`만 추가한 큐 항목으로 등재한다 (`configs/queue_dedup_renumbered_v5.json` 형식: `exp_num` / `dataset` 리스트 / 공백 구분 키=값). 분할·정규화·평가·best-epoch 기준 등 그 외 전부 불변 — 변경되는 것은 학습 라벨뿐이다.

	**집계 규칙**: 각 (데이터셋, p) 점은 해당 run의 best epoch(`pak_auc_f1` 기준; SWaT excl22는 `excl22_pak_auc_f1`) `metrics.pak_auc_f1`. 점선 floor는 Table 2 확정본의 anomaly-excised 조건 best unsupervised baseline 값을 그대로 가져온다.

	#### 📊 구성과 형태

	- **X축**: labeled fraction p (0.1 → 1.0, 선형 눈금).
	- **Y축**: PA%K-AUC F1.
	- **계열**: 데이터셋별 실선 1개 + 같은 색 점선(해당 데이터셋의 unsupervised floor) 1개. 범례에 데이터셋명.
	- **강조**: p=1.0 점은 main 설정과 동일함을 마커로 강조 가능.

	#### 📝 캡션 (영문 확정본 — [N]은 NUM-026 확정 후 치환)

	```latex
	Label sparsity sweep. PA\%K-AUC F1 as a function of the labeled anomaly fraction
	$p \in \{0.1, 0.25, 0.5, 0.75, 1.0\}$ for [N] representative datasets (one line per
	dataset).
	Dashed horizontal lines indicate the performance of the best unsupervised baseline
	(anomaly-excised condition, main protocol) on the corresponding dataset, providing the
	unsupervised floor.
	$p = 1.0$ corresponds to the main experimental setting; $p \to 0$ approximates the fully
	unsupervised limit.
	```

	#### ⚠️ 주의사항과 의존성

	- ⓐ **NUM-026(데이터셋 수)·NUM-027(열화 형상 서술어)이 이 실험에서 파생** — 같은 소스, 동시 확정.
	- ⓑ 점선 floor는 **Table 2 확정값에서만** 가져온다 (TAB-2 의존성). SMD/SMAP/MSL baseline 신규 실행(실행 #1)이 끝나기 전에는 해당 family를 대표 데이터셋으로 쓰는 경우 floor를 확정할 수 없다 — 권장 선택(SWaT excl22, PSM)은 [CMP-Q3] 재사용 가능 family라 이 함정을 피한다.
	- ⓒ p→0 극한과 Table 2 protocol-effect 블록의 "CSMAD (clean)"은 **다른 조건**이다 — clean-split은 prefix 자체가 train에 없는 반면, p=0은 비라벨 anomaly가 train에 남는다. 본문 상호참조 시 "approximates"라는 표현을 유지하고 동일시하지 말 것.
	- ⓓ §4.4 "Why graceful degradation is expected" 문단의 구조 논거(배치에 labeled positive가 없으면 GRL 손실 자체가 미계산 — `loss.py:293-302`)와 결과 해석의 일관성을 곡선 확정 후 확인.
	- 큐 등재 시 `force_mask_anomaly` 키 중복 같은 last-wins 패턴(exp287 원항목의 전례)을 답습하지 말 것 — override는 키당 1회만.

	#### 🔢 연결된 수치 placeholder

	| ID | 본문 위치 | 들어갈 값의 정의 |
	|---|---|---|
	| **NUM-026** | §4.4 Results lead의 [N] + 캡션 [N] 2곳 (동시 치환) | FIG-3 대표 데이터셋 수 — 설계 선택 2 또는 3 (권장: SWaT excl22 + PSM = 2; WaDi A1 추가 시 3) |
	| **NUM-027** | §4.4 Results 문장의 서술어 [gradually / monotonically] | 열화 형상의 정성 서술어 — **곡선 확정 후** 실제 형상에 맞는 쪽 선택. 비단조면 두 단어 모두 버리고 문장 재작성 (A8 — 곡선 없이 단어 선점 금지) |


### FIG-4 — 정성적 score 분해 (Qualitative score decomposition) {toggle="true"}

	| 유형 | 소스 분류 | 우선순위 | 의존성 |
	|---|---|---|---|
	| 정성 시각화 (§4.5) | **[재사용]** + 추출 스크립트 — [271c] 완주분 checkpoint 재사용, 신규 학습 불필요 | 재사용 묶음 | [271c] checkpoint + `scoring.py` |

	<callout icon="💡" color="blue_bg">
	**한 줄 요약** — 대표 anomaly 사건 구간에서 CSMAD 점수를 Teacher 재구성 오차와 Teacher–Student discrepancy로 분해해 나란히 그려, 두 성분이 서로 다른 신호를 낸다는 방법론의 핵심 설계 논리를 실제 데이터 위에서 보여주는 그림이다.
	</callout>

	**위치·크기** — §4.5 lead 직후 (`sec4_experiments.tex`, `\label{fig:decomp}`). full-width, 3.5–4 cm ≈ 0.30p.

	#### 🎯 목적과 의도

	§4.2(얼마나 잘하는가)와 §4.3(어느 component 덕분인가)이 끝난 뒤, §4.5는 "실제로 어떻게 작동하는가"를 보여주는 자리다. CSMAD의 점수는 두 성분의 합 — Teacher 재구성 오차 r_i와 adaptive 스케일링된 Teacher–Student discrepancy — 인데, 이 합산 설계가 의미를 가지려면 두 성분이 **서로 다른 정보를 담는다**는 것을 보여야 한다. 둘이 항상 같은 모양이라면 discrepancy 성분은 군더더기라는 비판(사실상 "recon만으로 충분하지 않은가")이 성립하기 때문이다. 이 그림은 TAB-3 행 4(w/o OD — 자동 recon-only)의 정량 결과와 짝을 이루는 **정성 증거**로, 같은 질문을 평균 수치가 아니라 실제 사건의 시간축 위에서 답한다.

	§12의 R10 논증 두 건이 이 그림에 직접 걸려 있다. (1) asymmetric Teacher–Student — "용량 격차가 비정상 상관 패턴에서 모방 실패를 키운다"는 주장은 행 3(discrepancy)이 anomaly 구간에서 상대적으로 솟는 모양으로 시각화된다. (2) adaptive scoring — 데이터셋마다 recon/disc의 절대 스케일이 크게 다른데도 두 성분이 한 그림에서 비교 가능한 것 자체가 adaptive scaling의 효과다(행 3은 스케일 적용 후 값을 그린다). 추가로 행 4의 threshold 점선은 anomaly-ratio threshold가 실제 점수 분포 위에서 어떻게 작동하는지 보여줘, "threshold selection이 불공정하다"는 §15 공격에 대해 oracle threshold가 아님을 시각적으로 재확인시킨다.

	열 선택(SWaT excl22 포함)도 논증적이다: excl22는 region 22 제거 후 남는 소형·다양한 사건들 위주의 조건이므로, 이 그림이 excl22의 사건들을 다루는 것은 "단일 대형 사건에 의존하지 않는다"는 §4.2의 주장과 호응한다.

	#### 🏁 목표와 기대 결과

	**입증하려는 것**: 캡션의 마지막 문장 그대로 — 재구성 오차는 사건 유형과 무관하게 정상 패턴 이탈에서 상승하고, discrepancy는 용량 격차와 라벨 유도 학습이 증폭하는 구조적 발산을 별도로 포착한다는 것. 성공 기준은 (1) 두 성분의 시간 형상이 사건별로 식별 가능하게 다를 것, (2) 합산 score(행 4)가 GT 음영 구간에서 threshold 점선을 상회할 것, (3) 4행 모두 GT 음영과 시간축이 정확히 정렬될 것.

	**기대와 다른 패턴이 나오면**: 두 성분이 모든 사건에서 사실상 동일 형상이면, §4.5 본문의 해석 문장("The two components respond distinctly...")을 실제 관찰에 맞게 약화·재작성한다 — 수치·관찰 확정 전 해석 강화 금지(RT MINOR-02)가 이 경우의 명령이다. 또한 그런 결과는 TAB-3 행 4의 하락폭 해석과 함께 읽어야 한다(정량 하락이 있는데 정성 그림에서 차이가 안 보이면 사건 선택이 대표성이 없는 것일 수 있으므로 다른 사건 구간으로 교체를 먼저 시도). 후보 열(WaDi A1 vs PSM) 중 시각적 변별이 좋은 쪽을 선택하는 절차 자체가 이 대응의 일부로 설계되어 있다.

	#### 🧪 실험 내용과 설계

	**[재사용] — [271c] 완주분에서 추출만 수행. 신규 학습 불필요.**

	- **점수 추출**: 해당 entity의 best checkpoint를 로드해 evaluator의 **동일 scoring 경로**로 per-timestep 배열 3종을 추출한다: `recon`(Teacher MSE), `scaled_disc = disc × (recon_mean/disc_mean)`, `score = recon + scaled_disc/4.0` (정본 산식: 271_CONFIG_TRUTH §VIII Anomaly Score). 구현의 단일 원천은 `mae_anomaly/scoring.py` — **다른 곳에 식을 복제하지 말 것** (CLAUDE.md API 체크리스트 3항; 2026-05-28 FM-omission 사고의 재발 방지 조항).
	- **threshold 점선**: 해당 entity metadata의 `metrics.anomaly_ratio_threshold` 값을 그대로 사용 (예: [271c] PSM 0.001744). 재계산 금지.
	- **사건 구간 선택**: SWaT excl22는 region 22 마스킹 후 남는 13개 소형 사건 중 **유형이 다른 사건 ≥2개**를 포함하도록 선택한다 (RT MINOR-02 — 사건 규모·유형 대표성). 구간 폭은 사건 길이의 3–5배 컨텍스트를 포함할 것을 권장.
	- **열 2 선택**: WaDi A1 또는 PSM 중 추출 결과를 보고 시각적 변별이 좋은 쪽 — 선택 결과가 NUM-028(=2)과 캡션의 [Dataset-A/B] 치환을 확정한다.

	#### 📊 구성과 형태

	2열(데이터셋) × 4행(분해 단계). 열 내 4행은 X축(timestep) 공유, 행별 Y는 per-trace 정규화.

	| 행 | 내용 | 특기 |
	|---|---|---|
	| 1 | 입력(첫 feature) + GT anomaly 붉은 음영 | 음영은 4행 전체에 연하게 관통 (정렬 확인용) |
	| 2 | Teacher 재구성 오차 (per timestep) | |
	| 3 | Teacher–Student discrepancy (adaptive 스케일 적용 후) | |
	| 4 | 합산 anomaly score + anomaly-ratio threshold 점선 | 점선은 이 행에만 |

	#### 📝 캡션 (영문 확정본 — [Dataset-A/B]는 선택 확정 후 치환)

	```latex
	Qualitative score decomposition on representative anomaly events.
	Each column corresponds to one dataset ([Dataset-A], [Dataset-B]).
	Row~1: multivariate input (first feature shown) with ground-truth anomaly regions shaded in
	red.
	Row~2: Teacher reconstruction error per timestep.
	Row~3: Teacher--Student discrepancy per timestep (adaptively scaled).
	Row~4: combined anomaly score with the anomaly-ratio threshold (dashed horizontal line).
	The decomposition illustrates how the two score components respond differently to anomaly
	characteristics: reconstruction error captures deviations from the learned normal pattern
	regardless of event type, while discrepancy captures structural divergence amplified by the
	capacity gap and label-guided training.
	```

	#### ⚠️ 주의사항과 의존성

	- ⓐ **Gaussian smoothing 절대 금지** (R34). [271c]의 저장 점수는 전부 비평활이므로 추출값을 그대로 그리면 자동 준수되지만, 시각화 코드 단계에서 후처리 smoothing을 끼워 넣지 말 것.
	- ⓑ §4.5 해석 문장("two components respond distinctly...")은 실제 그림 확정 후 사건별 관찰에 맞게 재검토한다 (RT MINOR-02 — 확정 전 해석 강화 금지).
	- ⓒ NUM-028이 이 그림에서 파생 — 그림 제작과 동시 치환.
	- 점수 산식·threshold를 그림 주석에 쓸 경우 §3.6 수식 번호(Eq. dscale/sigma)와 표기 일치 확인.

	#### 🔢 연결된 수치 placeholder

	| ID | 본문 위치 | 들어갈 값의 정의 |
	|---|---|---|
	| **NUM-028** | §4.5 lead의 [N] | FIG-4 데이터셋 수 = **2** (시각화 설계 확정값 — SWaT excl22 + {WaDi A1 또는 PSM}). FIG-4 제작과 동시 치환 |

---

## 2. 본문 Table

### TAB-1 — 데이터셋 통계 (Table 1: Dataset statistics) {toggle="true"}

	| 유형 | 소스 분류 | 우선순위 | 의존성 |
	|---|---|---|---|
	| 통계 표 (§4.1.1) | **[재사용]**(대부분 실값) + **[신규 측정]**(SMD per-machine 셀) | 측정 즉시 가능 | Table A.4와 동일 산출물 |

	<callout icon="💡" color="blue_bg">
	**한 줄 요약** — 6개 벤치마크 family의 재분할 후 train/test 크기와 anomaly 비율을 투명하게 공개해, contaminated benchmark protocol의 실측 기반을 제시하고 프로토콜 방어의 첫 단추를 끼우는 표다.
	</callout>

	**위치** — §4.1.1 (`sec4_experiments.tex`, `\label{tab:datasets}`), ~0.25p.

	#### 🎯 목적과 의도

	이 표는 단순한 데이터셋 소개가 아니라 **프로토콜 방어의 정량 기반**이다. 본 논문의 가장 큰 reviewer 공격면은 test-prefix 편입 프로토콜("test label로 학습하는 leakage 아닌가" — 블루프린트 §15 첫 행)인데, §14의 정면 답변 5논거 중 ②(원본 train에는 labeled anomaly가 구조적으로 부재)와 ④(시간성·전 데이터셋 단일 규칙)는 결국 숫자로 증명된다. Train AR 열이 "training 구간의 anomaly가 전적으로 편입된 prefix에서 유래한다"는 사실을, #Train/#Test 열이 재분할 규칙(`//2`)이 전 데이터셋에 동일하게 적용되었음을 보여준다. 캡션이 "originating from the incorporated test prefix"를 명시하는 것도 같은 이유다.

	또한 이 표는 비교 공정성 논증의 입력값이다. §4.1.4가 인정하는 train 데이터 양적 비대칭(anomaly-excised 조건에서 baseline의 train이 절제분만큼 작다)의 크기가 바로 Train AR 열(0.52–6.20%)이고, FIG-3의 데이터셋 선택 논리(PSM이 train AR 최대 → 라벨 경로 최활성)도 이 표를 근거로 한다. SWaT 행의 dagger(full/excl22 병기)는 §4.1.1 SWaT dual evaluation 문단과 부록 §A.4로 이어지는 excl22 서사의 출발점이다. 요컨대 reviewer가 프로토콜을 공격하려면 가장 먼저 보게 될 표이며, 여기서의 투명성이 이후 모든 방어의 신뢰도를 결정한다.

	#### 🏁 목표와 기대 결과

	이 표의 "성공"은 성능이 아니라 **정합성**이다. (1) 모든 셀이 EXPERIMENT_PROTOCOL_TRUTH §① 실측과 일치, (2) SMD per-machine 산출이 코드의 분할 산식과 동일한 규칙으로 계산됨, (3) 본문·부록의 동일 수치 인용처(§4.1.1 본문 범위 문장, Table A.4, §C.1 차원 표)와 자리수까지 일치.

	**기대와 다른 패턴이 나오면**: SMD per-machine Train AR이 기존 공개 범위(0.52–6.20%)를 벗어나는 machine이 있으면, §4.1.1 본문의 "Training anomaly ratios range from 0.52% to 6.20% (SMD per-machine values pending...)" 문장의 **범위 수치 자체를 같은 pass에서 수정**한다 — 표만 채우고 본문 범위를 방치하는 부분 수정은 금지다. 이 갱신은 §4.1.4의 양적 비대칭 인정 문장(0.52%–6.20%; SMD pending)에도 동일하게 적용된다.

	#### 🧪 실험 내용과 설계

	**대부분 [재사용]** — 다음 실값이 이미 tex에 반영·확정되어 있다 (EXPERIMENT_PROTOCOL_TRUTH §① 실측): SWaT 719,959 / 224,960 / 45 / 1.63 / 19.05·3.68†, WaDi 1,296,001·870,972 / 86,401·86,402 / 123 / 0.52·0.76 / 3.82·3.87, PSM 176,401 / 43,921 / 25 / 6.20 / 30.63, SMAP 355,905 / 217,925 / 25 / 0.70 / 24.54, MSL 95,271 / 36,775 / 55 / 1.70 / 16.72. SMD의 Test AR 평균 4.16도 실값.

	**잔여 [신규 측정]** — SMD 행의 per-machine 위임 셀(#Train, #Test, Train AR): 28개 machine 각각을 산출하는 **1회성 스크립트** (학습 불필요). 산출 규칙은 코드와 동일해야 한다 — `loaders.py:1152-1157`의 분할(`test_split = len(test_data)//2`; train = 원본 train 전체 + test 앞 50%, test = 뒤 50%)을 그대로 호출하거나, 같은 산식으로 라벨 파일에서 직접 계산한다. 본문 표는 "per-machine (§A.3)" 포인터 형태를 유지하므로, 이 산출물의 실제 게재처는 Table A.4(SMD per-machine 행)와 §4.1.1 본문의 "pending" 문구 해소다 — **TAB-1과 Table A.4는 동일 소스 산출물**이며 두 표 간 수치 불일치는 금지.

	#### 📊 구성과 형태

	booktabs 6행 — 형태는 tex 확정, 변경 불필요. SWaT Test AR은 dagger(†)로 full/excl22 병기(캡션에 정의). 확정 셀 미리보기:

	| Family | #Train | #Test | #Dim. | Train AR | Test AR |
	|---|---|---|---|---|---|
	| SWaT (A1+A2) | 719,959 | 224,960 | 45 | 1.63 | 19.05 / 3.68† |
	| WaDi (A1/A2) | 1,296,001 / 870,972 | 86,401 / 86,402 | 123 | 0.52 / 0.76 | 3.82 / 3.87 |
	| PSM | 176,401 | 43,921 | 25 | 6.20 | 30.63 |
	| SMD (×28) | per-machine (§A.3) | per-machine (§A.3) | 29–36 | per-machine | 4.16 (avg) |
	| SMAP (×54) | 355,905 | 217,925 | 25 | 0.70 | 24.54 |
	| MSL (×27) | 95,271 | 36,775 | 55 | 1.70 | 16.72 |

	#### 📝 캡션 (영문 확정본)

	```latex
	Dataset statistics under the contaminated benchmark protocol, summarized per family.
	Train/test sizes reflect the re-split described in \S\ref{sec:datasets}.
	Train AR = anomaly ratio (\%) in the training portion (originating from the incorporated
	test prefix); Test AR = anomaly ratio (\%) in the held-out evaluation portion.
	The WaDi row aggregates the two independent entities A1/A2 (values given as A1\,/\,A2);
	SMD, SMAP, and MSL values are per-entity averages or concatenated totals as indicated.
	SWaT is evaluated under both full and excl22 conditions ($\dagger$: full\,/\,excl22);
	Table~\ref{tab:main_results} uses excl22 (\S\ref{sec:datasets}).
	Per-entity statistics are in \ref{sec:appendix_dataset} (Table~\ref{tab:per_entity}).
	```

	#### ⚠️ 주의사항과 의존성

	- SMD per-machine Train AR 확정 시 §4.1.1 본문 범위 문장("0.52% to 6.20% ... pending")을 **같은 pass에서** 갱신 — SMD 값이 범위를 벗어나면 범위 수치 자체 수정 (부분 수정 금지).
	- #Dim 열은 §4.1.1이 단일 원천 — 부록 §C.2의 Table C.1(입력 차원 표)과 정합 유지 의무.
	- Table A.4(per-entity statistics)와 동일 소스·동일 산식 — 두 표 간 불일치 금지. A.4 캡션의 "SMD per-machine rows pending" 문구도 채움과 동시에 삭제.
	- SMD 차원 29–36은 constant 컬럼 제거 후 수치 — raw 38로 되돌려 쓰지 말 것 (ADV BLK-003).

	#### 🔢 연결된 수치 placeholder

	전용 NUM placeholder 없음 — 잔여 placeholder는 표 안의 SMD per-machine 셀 자체다. 대신 다음 동기화 의무가 NUM에 준해 적용된다.

	| 동기화 대상 | 위치 | 규칙 |
	|---|---|---|
	| Train AR 범위 문장 | §4.1.1 본문 ("range from 0.52% to 6.20% ... pending") | SMD 확정과 같은 pass에서 갱신 (범위 이탈 시 범위 수정) |
	| 양적 비대칭 인정 문장 | §4.1.4 ("0.52\%--6.20\%; SMD pending") | 동일 |
	| Table A.4 SMD 행 | §A.3 | 동일 스크립트 산출물 사용 — 수치 불일치 금지 |


### TAB-2 — Main 비교 결과 + protocol-effect 블록 (Table 2) · ★ 본 논문의 중심 표 {toggle="true"}

	| 유형 | 소스 분류 | 우선순위 | 의존성 |
	|---|---|---|---|
	| 중심 결과 표 (§4.2) | **[완주 대기]**(CSMAD) + **[신규 실행]**(baseline 일부·weak 4종·standard-split) | 신규 실행 #1·#2·#3 | **의존 그래프 루트** — 271canon 완주 + baseline 재실행 |

	<callout icon="💡" color="blue_bg">
	**한 줄 요약** — 26개 baseline(22 unsupervised + 4 weakly supervised)과 CSMAD를 6 family × 2 지표에서 비교하고, 하단 protocol-effect 블록으로 "성능 우위가 프로토콜의 추가 데이터 때문인가, 방법 때문인가"를 분리해 보이는, placeholder 의존 그래프의 루트가 되는 본 논문의 중심 표다.
	</callout>

	**위치·특기** — §4.2 (`sec4_experiments.tex`, `\label{tab:main_results}`), `table*` 2단 폭, ≈0.55p. **TAB-4(protocol-effect analysis)는 이 표의 하단 블록으로 흡수 완료** (D-010 ① — 별도 표·별도 Notion 페이지 없음; 본 토글이 그 명세를 전부 포함).

	#### 🎯 목적과 의도

	이 표는 논문의 중심 논제 — "labeled anomaly를 표현 학습에 직접 통합한 end-to-end 단일 모델이, 같은 라벨을 각자의 패러다임에서 최선으로 활용한 기존 방법들보다 낫다" — 를 입증하는 단일 증거물이다. 행 구성 자체가 논증이다. 22개 unsupervised baseline은 **anomaly-excised condition**(라벨로 오염원을 제거해주는, 그들에게 가장 유리한 조건)에서, 4개 weakly supervised baseline은 그들이 구조적으로 요구하는 **contaminated-training condition**에서 평가된다. 즉 "라벨 있는 우리 vs 라벨 없는 그들"이라는 불공정 구도가 아니라 "같은 라벨을 각자 최선으로 쓴 비교"(블루프린트 §14 논거 ③, R12)임을 표의 조건 표기가 직접 말한다. NRdetector 행은 §1이 "closest prior work"로 지목한 최직접 경쟁자와의 정면 비교다.

	하단 **protocol-effect 블록**(r2에서 TAB-4를 흡수)은 이 논문에서 가장 위험한 reviewer 공격 — "성능 우위가 GRL+distillation 때문이 아니라 test-prefix 편입으로 늘어난 train 데이터 때문 아닌가"(블루프린트 §15, RT BLOCKER-03) — 에 대한 정면 답변이다. 2단 논증 구조다. ① 동일 방법이 standard clean-train split에서도 비지도 SOTA와 경쟁력을 유지한다 → 성능이 프로토콜의 산물이 아니라 방법 자체의 가치임을 보임. ② labeled anomaly가 제공되는 contaminated 조건에서는 CSMAD만 추가 이득을 얻고, 비지도 baseline은 같은 데이터가 추가되어도 라벨을 활용하지 못한다 → 이득이 라벨 활용 능력에 특이적임을 보임. 두 조건의 평가를 동일한 원본 test 뒤 50%로 통일하는 것이 이 분리의 기술적 핵심이다(비교가 train 구성 차이만 반영하게 됨).

	이 표가 §4.2 분석 텍스트의 네 구조(요약 주장 / 데이터셋별 특이점 / protocol-effect 해석 / 비용 한계 인정)를 전부 먹여 살리며, NUM 4개 그룹(N-A·N-B·N-C·N-D), FIG-3의 floor, TAB-B1의 Δ 기준이 모두 여기서 파생된다 — **placeholder 의존 그래프의 루트**다.

	#### 🏁 목표와 기대 결과

	**입증하려는 패턴** (수치 예측이 아니라 방향): (1) CSMAD가 6 family의 다수에서 두 지표 모두 최상위권에 위치하고, 특히 train AR이 가장 높은 PSM(라벨 경로가 가장 강하게 발동)에서 라벨 활용의 이득이 뚜렷할 것. (2) SWaT excl22(소형·다양 사건만 남는 조건)에서도 경쟁력을 유지해 "단일 대형 사건 탐지에 의존하지 않는다"가 성립할 것. (3) protocol-effect 블록에서 clean-split CSMAD가 비지도 대표와 비등하고, contaminated로 옮기면 CSMAD만 유의미하게 상승하며 비지도 baseline의 변화는 그에 못 미칠 것.

	**기대와 다른 패턴이 나오면**: 어떤 family에서 1위를 놓치면 NUM-006의 win 수는 그대로 사실대로 기재하고, §4.2 요약 문장의 강도를 결과에 맞춰 조정한다(과장 금지 — "achieves the highest on [N] of six"는 어떤 N에도 문법적으로 성립하도록 이미 설계되어 있다). protocol-effect에서 clean-split CSMAD가 비지도 대표에 크게 밀리면 2단 논증의 ①이 약화되므로, 분석 문단을 "방법 자체의 경쟁력" 주장에서 "라벨 활용 이득"(② 중심)으로 재구성해야 하며, 반대로 비지도 baseline이 contaminated 조건에서 CSMAD에 준하는 이득을 보이면 NUM-019의 해석("confirming that the gain is specific to methods able to exploit the provided labels")을 그대로 둘 수 없다 — 어느 경우든 표와 본문 문장의 침묵 불일치는 금지이고, 문장 쪽을 결과에 맞춘다.

	#### 🧪 실험 내용과 설계

	**셀 값 정의** — 27 method 행(7개 그룹) × 7 데이터셋 열 {SWaT excl22, WaDi A1, WaDi A2, PSM, SMD avg, SMAP avg, MSL avg} × 2지표 {PA%K-AUC F1, VUS-PR} + 하단 protocol-effect 블록 3행:

	- **CSMAD 행**: [271c] entity별 `experiment_metadata.json`의 `metrics.pak_auc_f1` / `metrics.vus_pr` (best epoch 기준 — 전 지표가 같은 best epoch에서 추출됨). SWaT 열은 `SWaT/A1A2_excl22` entity(독립 best-epoch, `timing.best_epoch_metric='excl22_pak_auc_f1'`). SMD/SMAP/MSL avg = **entity별 best-epoch 지표의 macro 평균**(28/54/27 entity).
	- **unsupervised 22행**: anomaly-excised condition([CMP-Q3] 계열) 동일 키. random 행만 5-run mean(±std는 본문 비표기, §A.1에 명시).
	- **weakly supervised 4행**: contaminated-training condition 단독(구조적으로 excised 불가 — §4.1.4).
	- **protocol-effect 블록**: CSMAD(clean) + 대표 baseline 2–3종(NUM-014)의 standard clean-train split 결과 — 대표 열(SWaT excl22, WaDi A1, PSM — tex stub 기준)만 채우고 나머지는 "—".

	**실험 소스 — 4갈래** (실행 우선순위는 §0 대시보드 #1–#3):

	1. **CSMAD [완주 대기]**: 271canon 잔여 entity 완주(SMD 6, SMAP 49, MSL 22 — 2026-06-11 실측, 큐 진행 중). 완주 후 metadata 집계 스크립트로 macro 평균 산출. **부분 완주 상태로 avg 열을 채우는 것 금지** — sync 그룹 A("six families")가 깨진다.
	2. **unsupervised 22종 [신규 실행(부분)]**: SWaT/WaDi/PSM은 [CMP-Q3](`comparison/results/experiments/6_20260526_085028_baseline_minmax_normalonly_segaware/`) 재사용 가능. **SMD/SMAP/MSL은 `comparison/run_baseline_queue.py`로 전 entity 신규 실행 필수** — SMD normalonly 기존 결과는 per-entity 정규화(2026-06-02) 이전의 구버전 `3_20260312_*`뿐이라 폐기 대상이고, SMAP/MSL normalonly는 어느 결과 폴더에도 부재(미실행)다 (r2 정정 — "STALE 재실행"이 아니라 "SMD 구버전 폐기+재실행 / SMAP·MSL 미실행분 신규 실행"). variant는 `normalonly`(각 baseline의 `experiment_configs.py` 등록 항목 그대로; SMAP/MSL 포함 등록 실재 확인됨). SMD 재실행 시 per-entity 정규화 적용을 실행 전 확인.
	3. **weakly supervised 4종 [신규 실행]**: DeepMIL/WETAS/TreeMIL/NRdetector — 구현·CPU dry-test는 완료, **GPU 전체 실험 미실행**. contaminated-training(full/Q1) variant로 전 데이터셋 실행 (epochs 50, 매 epoch eval — `baseline_common.py` weak preset). NRdetector가 최직접 경쟁자이므로 그룹 6 중 최우선.
	4. **protocol-effect 블록 [신규 실행 + 신규 loader]** (흡수된 TAB-4의 실행 사양 — 블루프린트 §6.6 r3, 코드 근거 포함):
	   - **분할**: train = 원본 train 파일만(test-prefix 미편입, 라벨 anomaly 0), test = **main protocol과 동일한 원본 test 뒤 50%**. 평가 통일이 핵심 — 비교가 train 구성 차이만 분리하게 된다. 현행 loader에 이 variant가 없으므로 loader 함수/variant 추가가 필요하다(예: `*_standard` 키; 기존 `//2` 분할 코드의 train_len에서 prefix 항만 빼는 최소 수정).
	   - **CSMAD 설정**: 271 canon config **그대로, `use_grl=True` 유지**. 라벨 0인 train에서 세 라벨 경로는 코드 수준에서 자가 비활성화된다: anomaly-priority masking은 priority 전부 0 → 무작위 마스킹으로 자연 퇴화, OD 분기는 전 패치 정상(정상 전용과 동일), GRL은 batch 내 positive 부재 시 손실 자체가 계산되지 않음(`loss.py:293-302`). ⚠️ **`use_grl=False`로 끄는 것 금지** — dead component(dynamic margin anomaly loss)가 재활성화되어 비교가 오염된다 (§6.7과 동일한 함정).
	   - **baseline**: 대표 2–3종(NUM-014; 선정 기준 — main 표에서 강한 unsupervised 대표, 예: 최상위 recent 1 + legacy 1)을 동일 standard split에서 학습. 대표 데이터셋 3개(SWaT excl22, WaDi A1, PSM) 한정으로 비용 통제.

	**집계 규칙**: baseline 쪽 SMD/SMAP/MSL avg도 CSMAD와 **동일한 entity 집합·동일 macro 평균 규칙**이어야 한다. 집계에서 Exathlon·Simulation은 절대 배제(R33) — 기존 Notion RankAvg류 수치는 Exathlon 포함 기준이므로 재계산 필수(FEEDBACK-3).

	#### 📊 구성과 형태

	열 구조 (각 데이터셋 열은 F1 / VUS 2칸):

	| Method | Group | SWaT excl22 | WaDi A1 | WaDi A2 | PSM | SMD avg | SMAP avg | MSL avg |
	|---|---|---|---|---|---|---|---|---|
	| (27행 + 블록 3행) | | F1 / VUS | F1 / VUS | F1 / VUS | F1 / VUS | F1 / VUS | F1 / VUS | F1 / VUS |

	행 그룹 (midrule + 이탤릭 그룹 헤더, 조건 명기 — tex 확정):

	| 그룹 | 행 수 | 조건 표기 |
	|---|---|---|
	| Simple / lightweight | 5 (Random score, Sensor range, PCA recon., L2-norm, NN-distance) | anomaly-excised |
	| Lightweight neural | 4 (MLP, MLPMixer, Transformer, GCN-LSTM) | anomaly-excised |
	| SOTA legacy | 6 (Anomaly Trans., TranAD, USAD, DAGMM (simpl.), GDN, OmniAnomaly) | anomaly-excised |
	| SOTA recent | 7 (TFMAE, NPSR, TimesNet, DCdetector, MEMTO, ModernTCN, CATCH) | anomaly-excised |
	| Weakly supervised | 4 (DeepMIL, WETAS, TreeMIL, NRdetector) | contaminated-training |
	| CSMAD (ours) | 1 | contaminated train (excision 없음) |
	| Protocol-effect 블록 | 3 (CSMAD (clean), Baseline A, Baseline B; +C는 NUM-014 확정 시) | standard clean-train split |

	강조 규칙: **Bold = 열별 최고, underline = 2위** (방법 27행 대상; **protocol-effect 블록은 강조 제외**). 블록 행은 대표 3열(SWaT excl22, WaDi A1, PSM)의 F1만 채우고 나머지 "—".

	#### 📝 캡션 (영문 확정본 — [N]은 NUM-014 확정 후 치환)

	```latex
	Main comparison results under the contaminated benchmark protocol
	(anomaly-excised condition for unsupervised baselines; contaminated-training condition for
	weakly supervised baselines; \S\ref{sec:baselines}).
	Reported metrics: PA\%K-AUC F1 and VUS-PR; the remaining three metrics are in
	\ref{sec:appendix_full_results}.
	SWaT column uses the excl22 evaluation condition; full-condition results appear in
	\ref{sec:appendix_swat}.
	SMD, SMAP, and MSL values are macro-averages over all entities (per-entity results in
	\ref{sec:appendix_entity_results}).
	\textbf{Bold} = highest; \underline{underline} = second-highest.
	\textit{Bottom block (protocol effect, \S\ref{sec:main_results})}: CSMAD and [N]
	representative unsupervised baselines under a standard clean-train split (original training
	file only, no labeled anomalies), evaluated on the identical held-out evaluation suffix;
	standard-split CSMAD uses the identical configuration with all label-dependent paths
	automatically inactive in the absence of positive training windows.
	Cells are populated only for the representative protocol-effect dataset columns.
	```

	#### ⚠️ 주의사항과 의존성

	- ⓐ **이 표가 placeholder 의존 그래프의 루트** — NUM-006~013(본 블록), NUM-014~019(하단 블록), FIG-3 점선 floor, TAB-B1 Δ 기준이 전부 이 표에서 파생된다. 이 표의 확정 전에 파생 placeholder를 선기입하는 것 금지 (NUM-010 같은 "현재도 산출돼 있는 값"도 표 전체 확정 전 본문 선기입 금지).
	- ⓑ 집계에서 Exathlon·Simulation 절대 배제 (R33; 기존 Notion RankAvg 재계산 필수 — FEEDBACK-3).
	- ⓒ **weak 4종 미완 시 fallback**: sync 그룹 B 전체가 "22 unsupervised"로 일괄 전환 + Table 2 그룹 6(Weakly supervised) 행 삭제 + §4.1.2–4.1.4 하드코딩("26 baselines / 22 / 4") 동시 수정 — **부분 게재 금지**.
	- ⓓ SWaT 재실행이 발생하면 입력 차원 45 일치 검증 필수 (FEEDBACK-7 — 현 machineA raw CSV 경로는 51을 반환; 불일치 시 checkpoint 로드 실패 가능).
	- ⓔ baseline 쪽 SMD/SMAP/MSL avg도 CSMAD와 동일한 entity 집합·동일 macro 평균 규칙 — 반올림 자리수까지 Table A.8(per-entity)과 일관.
	- **TAB-4 흡수 기록**: protocol-effect analysis는 v2-r2에서 본 표 하단 블록으로 흡수되었다 (D-010 ①). 본문에 `[TAB-4]` 마커는 존재하지 않으며, 별도 Notion 페이지도 생성하지 않는다 — 명세·실행 지침·의존성은 본 토글 🧪 4항에 통합되어 있다.

	#### 🔢 연결된 수치 placeholder

	**그룹 N-A — 데이터셋 family 수 (sync 그룹 A) [완주 대기]** — 4개소 단일 값 동기화 의무:

	| ID | 본문 위치 | 들어갈 값의 정의 |
	|---|---|---|
	| **NUM-001** | Abstract 6문장 (`main.tex`) | family 수 — 6 family 전부 완주 시 "six" |
	| **NUM-003** | Highlights bullet 5 (`main.tex` highlights 블록 + `highlights.txt`) | 동일 값 (sync) |
	| **NUM-004** | §1 contribution bullet 4 (`sec1_intro.tex`) | 동일 값 (sync) |
	| **NUM-029** | §5 결론 (`sec5_conclusion.tex`) | 동일 값 (sync) |

	네 곳이 단일 값으로 동기화되어야 하며, §4.1.1 하드코딩 상수("six ... families", "113 entities / 114 evaluation conditions")·§4.2 "six dataset families"와도 일치 의무. 어느 family라도 제출 시점에 탈락하면 같은 pass에서 §4.1.1 상수까지 일괄 수정 (부분 수정 금지). 소스: 271canon 완주 + baseline 재실행 완료 = **TAB-2 완성이 전제**.

	**그룹 N-B — baseline 총수 (sync 그룹 B) [신규 실행(weak 4종) 의존]**:

	| ID | 본문 위치 | 들어갈 값의 정의 |
	|---|---|---|
	| **NUM-002** | Abstract (`main.tex`) | weak 4종 GPU 완주 시 "26" (22 unsup + 4 weak); 미완 시 세 곳 모두 "22 unsupervised"로 fallback |
	| **NUM-005** | §1 contribution bullet 4 | 동일 값 (sync) |
	| **NUM-030** | §5 결론 | 동일 값 (sync) |

	fallback 시 §4.1.2–4.1.4 하드코딩("26 baselines / 22 / 4")과 Table 2 그룹 6 행을 동시 제거. 소스: 본 토글 🧪 3항(weak 4종 Q1 GPU 실행)과 동일.

	**그룹 N-C — Table 2 본 블록 파생 [집계만 — TAB-2 완성 후]** (전부 신규 실험 없음):

	| ID | 본문 위치 (§4.2) | 들어갈 값의 정의 (집계 규칙) |
	|---|---|---|
	| **NUM-006** | ¶1, [N]×2 | 6 family 중 CSMAD가 1위인 family 수 — PA%K-AUC F1 기준 1개 + VUS-PR 기준 1개. **WaDi 집계 규칙 결정 필요**: 표는 A1/A2 2열인데 본문은 "six families" — 권고: A1·A2 모두 1위일 때만 WaDi family win (보수적), 채택 규칙을 본문 또는 각주 1줄로 명시 |
	| **NUM-007** | ¶1, [X.XX]×2 | CSMAD의 family 평균 (PA%K-AUC F1, VUS-PR) — WaDi는 A1/A2 평균을 family 값으로 한 뒤 6 family 평균 (규칙을 006과 통일) |
	| **NUM-008** | ¶1 | (CSMAD 평균) − (family별 최강 unsupervised의 평균), PA%K-AUC F1 |
	| **NUM-009** | ¶1 | 동일, VUS-PR |
	| **NUM-010** | ¶2 | CSMAD PA%K-AUC F1 @ PSM ([271c] PSM `metrics.pak_auc_f1` — 표 전체 확정 전 본문 선기입 금지) |
	| **NUM-011** | ¶2 | best unsupervised PA%K-AUC F1 @ PSM ([CMP-Q3]) |
	| **NUM-012** | ¶2 | CSMAD PA%K-AUC F1 @ SWaT excl22 ([271c] `SWaT/A1A2_excl22`) |
	| **NUM-013** | ¶3, [X.XX]×2 | NRdetector(contaminated-training) 대비 비교값 — registry 정의는 "margins", tex 문장은 CSMAD **절대값** 형태("CSMAD achieves [X.XX] ... on average"). 채움 시 문장·정의 중 한쪽으로 확정 (권고: 문장을 "outperforms NRdetector by [margin]"으로 고치거나, 절대값 유지 + 본문에 NRdetector 값 병기 — 침묵 불일치 금지) |

	NUM-008/009/011의 "최강 unsupervised"는 family별로 다른 방법일 수 있다 — 평균 산출 규칙(각 family의 best를 뽑아 평균 vs 단일 최강 방법의 평균)을 명시하고 일관 적용 (권고: 전자 — "strongest unsupervised competitor"의 보수적 해석).

	**그룹 N-D — Protocol-effect 블록 파생 [신규 실행 — standard-split run]** (전부 PA%K-AUC F1):

	| ID | 본문 위치 (§4.2 protocol-effect 문단) | 들어갈 값의 정의 |
	|---|---|---|
	| **NUM-014** | 블록 캡션 [N] + 본문 [N] (동시 치환) | 블록 내 대표 baseline 수 (설계 선택 2–3; tex stub은 A/B 2행) |
	| **NUM-015** | "CSMAD remains competitive ([X.XX] ...)" | CSMAD clean-train 평균 (protocol-effect 대표 데이터셋들) |
	| **NUM-016** | "... versus [X.XX] for the best unsupervised competitor" | best unsupervised clean-train 평균 |
	| **NUM-017** | "CSMAD improves to [X.XX]" | CSMAD contaminated 평균 — **Table 2 본 블록의 같은 데이터셋 부분집합 재집계** (신규 실행 아님) |
	| **NUM-018** | "(a gain of [X.XX] points)" | 파생 계산값: 017 − 015 (별도 측정 금지) |
	| **NUM-019** | "the unsupervised baselines show [X.XX] change" | best unsupervised의 조건 간 변화량 (standard → contaminated). **주의**: 비교쌍은 standard-split run vs **contaminated-training(무절제) run** — anomaly-excised가 아니라 "같은 추가 데이터를 받은" 조건. contaminated 쪽은 TAB-B1 실행분과 소스 공유 가능 |


### TAB-3 — Ablation study (Table 3) {toggle="true"}

	| 유형 | 소스 분류 | 우선순위 | 의존성 |
	|---|---|---|---|
	| ablation 표 (§4.3) | **[재사용]**(행 1·3) + **[신규 실행]**(행 2·4) | 신규 실행 #4·#6 | TAB-3 대표 열(NUM-020) 자기 확정 |

	<callout icon="💡" color="blue_bg">
	**한 줄 요약** — 세 가지 라벨 유도 경로(anomaly-priority masking, OD loss, GRL)를 하나씩 제거한 변형과 full model을 대표 데이터셋에서 비교해, contribution bullet 2의 "세 경로 각각이 기여한다"는 주장을 정량 분해하는 표다.
	</callout>

	**위치** — §4.3 (`sec4_experiments.tex`, `\label{tab:ablation}`), half-width, ≈0.20p.

	#### 🎯 목적과 의도

	§4.2가 "얼마나 잘하는가"를 보였다면 §4.3은 "왜 잘하는가 — 어느 component 덕분인가"를 분해한다(블루프린트의 MECE 설계: component-level 서사는 이 소절 전속, §4.2와 중복 금지). 4행 구성은 contribution bullet 2의 세 경로에 1:1 대응한다: 행 3 ↔ anomaly-priority masking(§12 논증 "anomaly-class imbalance 직접 대응"), 행 4 ↔ loss bifurcation의 OD 손실(§12 "정상에서 낮은 discrepancy 유도 → 대비 증폭"), 행 2 ↔ gradient-reversal suppression(§12 "능동 제거"). 각 변형의 하락폭이 곧 해당 경로의 정량 기여이며, §3의 R10 논증("이게 없으면 왜 나빠지는가")의 실측 검증이다.

	이 표에서 가장 정교하게 설계된 것은 **행 2의 정의**다. "w/o GRL"을 단순히 GRL을 끄는 것으로 정의하면, anomaly 패치의 OD-loss 제외(수동 회피)와 GRL(능동 억제)의 효과가 섞여 버린다. 행 2는 OD-exclusion을 **유지한 채** GRL classifier와 reversal만 제거해(RT MAJOR-05), "수동 회피만으로는 부족하고 능동 억제가 추가 기여를 한다"는 §3.5의 핵심 문단("Why gradient reversal is necessary beyond loss bifurcation")과 §1 관찰 문단("Relying only on (b) is insufficient")을 정량적으로 입증한다. 이것이 reviewer의 "GRL이 정말 필요한가 — exclusion만으로 충분하지 않나"라는 공격(이 논문의 novelty 핵심을 겨누는 공격)에 대한 유일한 정량 방어다.

	확장 변형(FM, warmup, symmetric decoder, depth sweep)은 부록 Table B.4로 위임되어 본문 표는 "라벨 경로 3종의 분해"라는 단일 메시지에 집중한다 — warmup이 contribution이 아니라는 Phase 3 결정(블루프린트 결정 ①)과 정합하는 배치다.

	#### 🏁 목표와 기대 결과

	**입증하려는 패턴**: 행 1(full)이 기준선 최고치이고, 세 변형 각각에서 Avg 기준 하락이 관찰되는 것. 특히 행 2의 하락(GRL 순효과)이 0이 아니라는 것이 능동 억제 논증의 성패를 가른다. 하락폭의 부호 규약: 본문이 "removal costs X points" / "the drop is X" 형식이므로 NUM-021/022/023은 **양수 하락폭**으로 기재한다.

	**기대와 다른 패턴이 나오면**: 어떤 변형이 full보다 좋게 나오면(음수 하락), 해당 본문 문단을 "improves by"로 문장 자체를 고쳐야 하고(결과 확인 후 문장 확정 — 침묵 수정 금지), 그 component의 §3 논증과 §12 배치표를 재검토해야 한다. 특히 행 2가 무하락이면 "GRL의 순효과" 주장은 본문에서 유지 불가 — 그 경우 GRL 서사는 §3.5의 구조 논증(맥락 노출 경로 차단)을 정성 수준으로 하향하고, rebuttal 대비 권고 실험 R-PROBE(probing classifier — GRL의 표현 억제 직접 증거)의 우선순위가 올라간다. 데이터셋별로 하락폭이 크게 다른 것은 자연스러운 결과다(train AR이 높을수록 라벨 경로가 활성 — PSM에서 가장 큰 하락이 나오는 패턴이 설계 논리와 정합).

	#### 🧪 실험 내용과 설계

	4행 확정(D-010 ②). 열 = 대표 3–4 데이터셋(NUM-020) + Avg. 지표 = PA%K-AUC F1 (best epoch, main과 동일 기준). 행별 소스와 실행 지침:

	| 행 | 소스 | 실행 지침 |
	|---|---|---|
	| 1. Full model (CSMAD) | **[완주 대기/재사용]** [271c] | 대표 데이터셋 열은 이미 완주분(SWaT·PSM·WaDi)에서 추출 가능 |
	| 2. w/o GRL (OD-excl. 유지) | **[신규 실행]** | **큐에 정확한 변형 부재** (exp290은 no_fm+no_grl 복합 — 행 2 정의와 불일치). 신규 큐 항목: 271 canon 기반 `use_grl=False` + **`anomaly_loss_weight=0.0` 추가로 anomaly-loss 경로 명시 차단**. 이유: `use_grl=False` 단독이면 `grl_disable_anomaly_loss` 게이트가 풀려 dead component인 dynamic-margin anomaly loss가 재활성화되어 비교가 오염된다 (§6.7 함정). 이렇게 OD-exclusion(정상 패치 전용 OD)을 유지한 "GRL 순효과" 변형을 만든다 |
	| 3. w/o anomaly-priority masking | **[재사용]** exp287_unmask (`287_20260603_132835_unmask`) | `force_mask_anomaly=False` 단독 diff — metadata 실측 확인됨. 대표 데이터셋 분 완주 상태 — metadata 집계만. 참고(OBS-2): 큐 원항목 `config_override`에 `force_mask_anomaly` 키가 중복 기재(True→False, last-wins로 net False)되어 있었다 — 단독 diff는 실측으로 확정이나, **신규 큐 항목 작성 시 이 중복 키 패턴 답습 금지** |
	| 4. w/o OD loss | **[신규 실행]** | 신규 큐 항목: `use_output_discrepancy=False`. **score 처리 방침 (코드 확정 사실 — r2 정정)**: 기본 동작은 **자동 recon-only** — `mae_anomaly/scoring.py:105-106`의 `resolve_score_weights`가 `use_output_discrepancy=False`면 `w_disc=0`을 강제하고, `scoring.py:249-253`에서 `w_disc=0` → `student_error=0` → score = Teacher recon만 남는다. 즉 별도 조치 없이 학습·추론 양쪽에서 OD가 일관 제거된다. **이 자동 recon-only 동작을 표 각주로 명시할 것.** disc 성분을 score에 남기는 변형을 원하는 경우에만 별도 채점 경로가 필요 — 침묵 변경 금지 |

	집계 규칙: 각 행의 각 셀은 해당 run의 best epoch `metrics.pak_auc_f1`(SWaT excl22 열은 `excl22_pak_auc_f1` 기준 선정). Avg = 선택된 대표 데이터셋 열의 단순 평균. NUM-021/022/023은 행 1과 각 변형 행의 **Avg 열 차분**.

	#### 📊 구성과 형태

	| Variant | Dataset-A | Dataset-B | Dataset-C | (Dataset-D) | Avg. |
	|---|---|---|---|---|---|
	| 1. Full model (CSMAD) | | | | | |
	| 2. w/o GRL (OD-excl. retained) | | | | | |
	| 3. w/o anomaly-priority masking | | | | | |
	| 4. w/o OD loss | | | | | |

	행 1이 기준선(최고치 기대), 변형 행은 하락폭이 드러나도록 Avg 열 포함. 강조는 통상 Full 행 bold 불필요 — Table 2와 달리 경쟁 비교 표가 아니라 분해 표이므로 (Phase 7 스타일 판단에 위임하되 일관 적용).

	#### 📝 캡션 (영문 확정본)

	```latex
	Ablation study. PA\%K-AUC F1 for each model variant on [3--4 representative datasets].
	Row~2 (w/o GRL) removes the GRL classifier and reversal but retains the anomaly-patch
	OD-loss exclusion, isolating the net effect of active adversarial suppression.
	Extended variants (feature matching, Teacher-only warmup, symmetric decoder) are in
	\ref{sec:extended_ablations} (Table~\ref{tab:extended_ablations}).
	```

	#### ⚠️ 주의사항과 의존성

	- ⓐ **대표 데이터셋 선정(NUM-020)**: 권장 SWaT excl22 + PSM(train AR 최대 — 라벨 경로 가장 활성) + WaDi A1 (+ WaDi A2 또는 SMD 대표 1). 단 **선택된 열은 행 1–4 전부와 부록 TAB-B4에서 글자 단위로 동일**해야 한다 (열 불일치 금지 — B4 캡션이 "the ablation datasets of Table 3"을 약속).
	- ⓑ NUM-021/022/023이 이 표의 Avg 열 차분에서 파생.
	- ⓒ 행 라벨은 "w/o anomaly-priority masking" — 내부 config명 `force_mask_anomaly`를 표에 노출하지 말 것.
	- ⓓ 행 4의 자동 recon-only 각주 의무 (위 실행 지침 참조) — 각주 없이 게재하면 "OD를 학습에서 뺐는데 score에는 남아 있는가"라는 모호성이 생긴다.
	- ⓔ FIG-B1·TAB-B2의 대표 데이터셋 선택도 이 표와의 통일이 권장되어 있다 — NUM-020 확정이 부록 설계의 입력값이 된다.

	#### 🔢 연결된 수치 placeholder

	그룹 N-E 중 TAB-3 소스분 (NUM-024/025는 TAB-B4 소스 — §3 TAB-B4 토글에서 명세):

	| ID | 본문 위치 | 들어갈 값의 정의 |
	|---|---|---|
	| **NUM-020** | §4.3 lead의 [N] + 캡션 "[3--4 representative datasets]" | ablation 대표 데이터셋 수 (설계 선택 3–4 — ⓐ 권고안 확정 시 결정; TAB-B4와 동일 집합) |
	| **NUM-021** | §4.3 "Anomaly-priority masking (Row 3)" 문단 | 행 1 − 행 3의 Avg 차 (w/o anomaly-priority masking 하락폭, 양수 기재) — 소스: [271c] − exp287 **[재사용]** |
	| **NUM-022** | §4.3 "Output discrepancy loss (Row 4)" 문단 | 행 1 − 행 4의 Avg 차 (w/o OD 하락폭, 양수 기재) — 소스: 행 4 **[신규 실행]** |
	| **NUM-023** | §4.3 "GRL adversarial suppression (Row 2)" 문단 | 행 1 − 행 2의 Avg 차 (GRL 순효과, 양수 기재) — 소스: 행 2 **[신규 실행]** |

	부호 규약: 본문이 "removal costs X points" / "the drop is X" 형식이므로 양수 하락폭으로 기재 — 음수면 "improves by"로 문장 자체를 고친다 (결과 확인 후 문장 확정).

---

## 3. 부록 Figure · Table

### TAB-A3 — 26개 baseline 하이퍼파라미터 전수 표 (Table A.3) {toggle="true"}

	| 유형 | 소스 분류 | 우선순위 | 의존성 |
	|---|---|---|---|
	| 재현성 표 (§A.1) | **[재사용]** — 코드 상수 추출만 | 재사용 묶음 | `MODEL_CONFIGS` 딕셔너리 |

	<callout icon="💡" color="blue_bg">
	**한 줄 요약** — 26개 baseline 각각의 {Window, LR, Batch, Epochs, Key parameters}를 `comparison/baseline_common.py`의 `MODEL_CONFIGS`에서 그대로 덤프해 채우는 재현성 표다. 학습 불필요, 값 발명 절대 금지.
	</callout>

	**위치·단일 원천** — 부록 §A.1, `appendix_A.tex`, `\label{tab:baseline_hparams}`. 단일 원천은 `comparison/baseline_common.py`의 `MODEL_CONFIGS`.

	#### 🎯 목적과 의도

	이 표는 논문의 **재현성 주장을 떠받치는 기반 표**다. §4.1.2가 "각 baseline은 원 구현 또는 발표 preset의 설정을 유지한다"라고 선언하는데, 그 선언을 26개 방법 각각에 대해 검증 가능한 구체 수치로 펼쳐 보이는 곳이 바로 여기다. 리뷰어 방어 관점에서는 두 갈래 공격을 막는다. 첫째, "baseline을 불리하게 튜닝한 것 아닌가"라는 공정성 공격에 대해, 모든 방법이 자기 원 구현의 preset을 그대로 쓴다는 사실을 표 한 장으로 입증한다. 둘째, "통일 파이프라인과 원 구현 설정이 어디서 어떻게 다른가"라는 재현성 질문에 대해, window·epochs·batch의 이탈 항목을 명시적으로 나열함으로써 답한다. DAGMM의 simplified 표기는 "이 구현은 DAGMM이 아니다"라는 방법-재정의 공격(블루프린트 결정 ⑦)을 선제 차단하는 장치다.

	#### 🏁 목표와 기대 결과

	성공 기준은 단 하나다: **26행의 모든 셀이 `MODEL_CONFIGS`의 실값 또는 "original preset" 표기로 채워지고, 발명된 값이 0개일 것.** 이 표는 성능 결과 표가 아니므로 기대하는 수치 패턴은 없다. 대신 정합성 기준 두 가지를 통과해야 한다. 첫째, Table A.2(`tab:budgets`)와의 모순이 없어야 한다 — 특히 baseline의 batch 열은 모델별로 32–512 범위에서 제각각이므로, A.2의 "model-specific" 서술과 일치해야 하고 특정 단일값(구판의 "512")을 인용해서는 안 된다. 둘째, Window·Epochs 열의 이미 확정된 실값(예: Anomaly Transformer 100/10, NRdetector 100/50, TranAD 10, USAD 12, GDN 15, DCdetector 105)과 새로 덤프한 값이 일치해야 한다. 불일치가 나오면 그것은 표가 아니라 `MODEL_CONFIGS`가 변경된 신호이므로, 변경 이력을 추적해 어느 쪽이 실험 당시 값인지 확인한 뒤 기재한다.

	#### 🧪 실험 내용과 설계

	학습은 전혀 필요 없다. 작업은 1회성 추출 스크립트 하나다. `comparison/baseline_common.py`의 `MODEL_CONFIGS`를 import하여 26개 모델 항목의 {window, lr, batch, epochs, 핵심 모델 파라미터}를 표 형식으로 덤프한다. 잔여 placeholder는 LR·Batch·Key parameters 세 열이며, Window·Epochs 열은 이미 tex에 실값으로 확정되어 있다.

	채움 규칙은 다음과 같다. `MODEL_CONFIGS`에 명시된 키는 그 값을 그대로 옮긴다. `MODEL_CONFIGS`에 없는 항목은 **"original preset"으로 표기하고 빈 칸으로 두지 않는다** — 어떤 값도 발명하지 않는다는 A8 원칙이 이 표에서는 "코드에 없는 값은 코드에 없다고 쓴다"로 구현된다. Key parameters 열은 각 방법의 변별적 파라미터(예: PCA 성분 수 50, NN-distance 이웃 수 5, random score 5-run 평균)를 1–2개 골라 기재하되, 역시 코드 등록 항목에서만 가져온다. DAGMM 행은 "DAGMM (simpl.)" 표기를 유지하고 Key parameters 셀에 "GMM omitted"를 남긴다 — TranAD 저장소의 simplified 재구현(GMM energy 항 생략)임을 캡션과 행 양쪽에서 일관되게 알린다.

	#### 📊 구성과 형태

	booktabs 26행 × 6열 {Method, Window, LR, Batch, Epochs, Key parameters}. 행은 tex 확정 구조 그대로 4계층 그룹으로 나눈다: simple/lightweight 9종 → SOTA legacy 6종 → SOTA recent 7종 → weakly supervised 4종, 그룹 사이는 이탤릭 그룹 헤더 행. simple 5종(random score, sensor range, PCA, L2-norm, NN-distance)은 학습 개념이 없으므로 LR·Batch·Epochs가 "—"다. 형태 변경은 불필요하다 — 구조는 tex에서 이미 확정되었고 남은 일은 셀 채움뿐이다.

	#### 📝 캡션 (영문 확정본)

	```
	Hyperparameters of all 26 baselines.
	Each method retains the settings of its original implementation or publication preset;
	deviations from the unified pipeline (window size, epochs, batch size) are listed explicitly.
	DAGMM follows the simplified TranAD-repository re-implementation (GMM energy term omitted).
	```

	#### ⚠️ 주의사항과 의존성

	Table A.2(budgets)와의 정합이 첫째 함정이다: baseline의 batch 열은 "model-specific (original presets)"이 정본이며, 구판 문서들이 인용하던 "512" 단일값을 어디에도 되살리면 안 된다(v2-r3 정정 사항). 둘째, 향후 baseline 큐 재실행 과정에서 어떤 모델의 preset이 바뀌면 **같은 커밋 pass에서 이 표를 갱신**해야 한다 — 표와 코드가 어긋난 채 제출되는 것이 최악의 실패 모드다. 셋째, tranad의 LR은 논문 텍스트(0.01)와 코드 `constants.py`(1e-4)가 다르다는 기존 미결 사안(RESEARCH_SYNTHESIS §⑥)이 있으므로, 이 표에는 **실제 실험이 사용한 코드 값**을 기재하고 필요 시 각주로 원 논문 값과의 차이를 밝힌다.

	#### 🔢 연결된 수치 placeholder

	이 표에서 파생되는 inline NUM placeholder는 없다. 본문 §4.1.2의 budgets 서술과 정합 의무만 진다.

### Table A.4 (부분) — Per-entity 데이터셋 통계: SMD per-machine 셀 {toggle="true"}

	| 유형 | 소스 분류 | 우선순위 | 의존성 |
	|---|---|---|---|
	| 통계 표 부분 (§A.3) | **[신규 측정]** — 학습 불필요, 스크립트 1회 | 측정 즉시 가능 | 본문 TAB-1과 단일 산출물 공유 |

	<callout icon="💡" color="blue_bg">
	**한 줄 요약** — Table A.4에서 SMD 행의 {#Train, #Test, Train AR} 세 셀만 placeholder다. loader의 분할 산식(`//2`)을 그대로 재사용하는 1회성 측정 스크립트로 28개 machine 통계를 뽑아, 본문 Table 1의 SMD 위임 셀과 **같은 산출물로 동시에** 채운다.
	</callout>

	**위치** — 부록 §A.3, `appendix_A.tex`, `\label{tab:per_entity}` — SMD 행 3종 셀만 잔여. 동일 소스: 본문 TAB-1의 SMD per-machine 위임 셀과 **단일 산출물 공유**.

	#### 🎯 목적과 의도

	Table A.4는 본문 Table 1의 family 요약을 entity 단위로 펼친 표로, 오염 벤치마크 프로토콜의 분할 결과를 entity별로 투명하게 공개하는 재현성 장치다. 나머지 행(SWaT, WaDi A1/A2, PSM, SMAP, MSL)은 전부 실값 확정 상태이고, SMD 행만 "[per-machine]" 위임 셀로 남아 있다. 이 셀을 채우는 일은 단순한 빈칸 메우기가 아니라, §4.1.1 본문이 들고 있는 "Training anomaly ratios range from 0.52% to 6.20% (SMD per-machine values pending…)" 문장의 **pending 꼬리를 해소하는 유일한 경로**다. 프로토콜 방어(블루프린트 §14) 관점에서도, 분할 규칙이 전 데이터셋 단일 산식임을 보이는 논거 ④(통일성)의 증거 표 역할을 한다.

	#### 🏁 목표와 기대 결과

	성공 기준: SMD 28개 machine 각각의 #Train, #Test, Train AR이 산출되어 표(또는 요약 행)에 들어가고, 캡션의 "SMD per-machine rows pending" 이탤릭 문구가 **채움과 동시에 삭제**되는 것. 기대 패턴: SMD machine별 Train AR이 기존 본문 범위 0.52–6.20% 안에 들어오면 §4.1.1 문장은 괄호 절만 떼어내면 된다. 만약 어떤 machine의 Train AR이 이 범위를 벗어나면(더 낮거나 높으면) **범위 수치 자체를 같은 pass에서 수정**해야 한다 — 부분 수정은 금지이며, Table 1·Table A.4·§4.1.1 본문이 한 번에 움직여야 한다. Test AR의 28-machine 평균은 이미 확정된 4.16과 일치해야 한다(불일치 시 산식 검증으로 회귀).

	#### 🧪 실험 내용과 설계

	1회성 측정 스크립트를 작성한다. 산출 규칙은 코드와 글자 단위로 동일해야 한다: `loaders.py:1152-1157`의 SMD 분할 — `test_split = len(test_data)//2`, train = 원본 train 전체 + 원본 test 앞 50%, test = 원본 test 뒤 50% — 을 **그대로 호출**하거나, 같은 산식을 라벨 파일 위에서 직접 재계산한다. 두 방식 중 loader 직접 호출이 안전하다(산식 복제 과정의 off-by-one을 원천 차단). machine별로 #Train = 분할 후 train 길이, #Test = 분할 후 test 길이, Train AR = train 구간 라벨 1 비율(%)을 산출한다. 산출물은 CSV 또는 JSON 한 개로 저장해 Table 1과 Table A.4 양쪽 채움 스크립트가 **같은 파일을 읽게** 한다 — 두 표 간 수치 불일치를 구조적으로 불가능하게 만드는 것이 핵심 설계다.

	#### 📊 구성과 형태

	표 구조는 tex 확정 상태다: {Entity, #Train pts, #Test pts, #Dim., Train AR (%), Test AR (%), Source} 7열. SMD를 28행 전부 펼칠지, 한 행 요약(범위 표기) + 비고로 처리할지는 지면 판단 사안이다 — 펼치면 28행이 추가되므로 부록 분량과 교환 관계에 있다. 어느 쪽을 택하든 #Dim. 열의 "29–36"(per machine)은 유지하고, 펼치는 경우 machine별 실측 차원을 함께 기재할 수 있다(metadata `num_features` 실측 — 잔여 6개 machine은 완주 후 확인).

	#### 📝 캡션 (영문 확정본)

	```
	Dataset statistics under the contaminated benchmark protocol, per entity.
	Train/test sizes reflect the re-split of Section~\ref{sec:datasets}; Train AR\,/\,Test AR
	denote the anomaly ratio of the training\,/\,evaluation portion.
	SMAP and MSL sizes are concatenated per-channel totals.
	\textit{SMD per-machine rows pending.}
	```

	채움 완료 시 마지막 문장 `\textit{SMD per-machine rows pending.}`을 삭제한다 — 이 삭제가 resolved 신호다.

	#### ⚠️ 주의사항과 의존성

	본문 TAB-1의 SMD 위임 셀과 **반드시 동일 산출물**을 사용한다(두 표 독립 계산 금지). §4.1.1의 Train AR 범위 문장 갱신을 같은 pass에 포함한다. #Dim 열의 단일 원천은 §4.1.1이며 부록 C의 Table C.1(`tab:dimensionality`)과 정합을 유지해야 한다 — SMD 차원 범위(29–36)는 잔여 6 machine 완주 후 범위가 바뀔 수 있음을 인지할 것(RESEARCH_SYNTHESIS §⑥ N5). 측정은 학습과 무관하므로 271canon 완주를 기다릴 필요가 없다 — 지금 바로 실행 가능한 항목이다.

	#### 🔢 연결된 수치 placeholder

	NUM 등록 항목은 없으나, §4.1.1 본문의 Train AR 범위 문구(비-NUM placeholder성 문장)와 Table 1 SMD 셀이 이 측정에 동시 의존한다.

### TAB-A6 — SWaT 이중 조건(full / excl22) 전수 결과 표 (Table A.6) {toggle="true"}

	| 유형 | 소스 분류 | 우선순위 | 의존성 |
	|---|---|---|---|
	| 결과 표 (§A.4) | **[완주 대기 + TAB-2와 동일 소스]** — 별도 실험 없음 | 완주 대기 | 271canon 완주 + baseline 실행 묶음 |

	<callout icon="💡" color="blue_bg">
	**한 줄 요약** — 27개 방법 × {full, excl22} 두 평가 조건 × 5지표 전수. 같은 학습 모델·같은 점수에서 평가 마스크만 바꾼 결과이므로, "excl22 기준이 자의적"이라는 공격에 대한 투명성 방어 표다.
	</callout>

	**위치·CSMAD 소스** — 부록 §A.4, `appendix_A.tex`, `\label{tab:swat_dual}`. CSMAD 소스는 [271c]의 `SWaT/A1A2_full`·`SWaT/A1A2_excl22` 두 entity metadata.

	#### 🎯 목적과 의도

	본문 Table 2의 SWaT 열은 excl22 조건만 보여준다. 이 선택이 자의적이지 않음을 입증하려면 full 조건 결과를 숨기지 않고 전부 공개해야 하며, 그 공개 장소가 이 표다. 방어하는 공격은 블루프린트 §15의 "SWaT excl22 기준이 자의적이다" 시나리오다. 방어 논리는 ① region #22가 test anomaly 질량의 83.75%를 차지하는 단일 거대 사건이라 full 조건은 사실상 "그 한 사건을 잡았는가"의 지표가 되어 변별력이 낮고, ② excl22 마스크는 모든 baseline에 동일하게 적용되며, ③ full 결과도 이 표에 전수 병기된다는 3단이다. 이 표는 그중 ③을 물리적으로 이행한다. 동시에 §A.4 본문의 region 정의(평가 구간 내 [2,869, 38,769), 35,900 timesteps)와 함께 excl22의 결정론적 식별 가능성을 보여준다.

	#### 🏁 목표와 기대 결과

	성공 기준: 27행(CSMAD + 26 baseline) × 10셀(조건 2 × 지표 5)이 전부 채워지고, 강조 규칙이 본문 표와 통일되는 것. 기대 패턴은 명확하다 — **full 조건 수치가 excl22보다 전반적으로 크게 좋아 보이는 것이 정상이다** (CSMAD [271c] 실측: full `pak_auc_f1` 0.944 vs excl22 0.629). 이 대비 자체가 논증 재료다: 단일 거대 사건이 지표를 부풀린다는 §4.1.1 주장의 실측 증거이며, 캡션의 "같은 모델, 마스크만 차이" 서술이 그 해석 장치다. baseline들 역시 full에서 부풀려진 수치를 보일 것으로 기대된다. 만약 어떤 baseline이 excl22에서 오히려 더 좋다면 그 방법은 region 22를 놓치고 소형 사건들을 잡는다는 뜻이므로, 본문 해석에 쓸 수 있는 관찰이 된다(의무는 아님). CSMAD의 excl22 열 수치는 Table 2의 SWaT 열과 글자 단위로 일치해야 한다.

	#### 🧪 실험 내용과 설계

	별도 실험이 없다. TAB-2를 채우는 실행 묶음에서 자동 산출된다. 갈래별 소스는 다음과 같다.

	| 행 그룹 | 소스 | 작업 |
	|---|---|---|
	| CSMAD | **[재사용]** [271c]의 `SWaT/A1A2_full`·`SWaT/A1A2_excl22` 두 entity | metadata `metrics` dict에서 5지표 추출. **각자 독립 best epoch** — full은 `pak_auc_f1`, excl22는 `excl22_pak_auc_f1` 기준 (271_CONFIG_TRUTH §IV 운영 주의) |
	| unsupervised 22종 | **[재사용 — CMP-Q3]** | comparison 파이프라인의 dual 조건 산출(`has_excl22`; 결과 디렉토리 `SWaT/A1A2_full`·`A1A2_excl22`)에서 추출 |
	| weakly supervised 4종 | **[신규 실행 — TAB-2 ② 3항과 동일 run]** | weak 4종 Q1 GPU 실행이 완료되면 같은 dual 산출 구조에서 추출 — 이 표를 위한 추가 실행은 없다 |

	5지표의 내부 키는 {`pak_auc_f1`, `pak_auc_prc_auc`, `vus_pr`, `vus_roc`, `affiliation_f1_ar`}이다. 모든 지표는 `compute_full_metric_set`이 같은 best epoch에서 일괄 산출하므로 추출만 하면 된다.

	#### 📊 구성과 형태

	좌우 2블록 구조: {Full condition 5열 | excl22 condition 5열}, 블록 사이 공백 열 1개, 총 27행. 열 약칭은 tex 확정대로 {F1, PR, VUS-PR, VUS-ROC, Aff.}. 강조 규칙은 본문 표와 통일(열별 bold = 최고, underline = 2위)을 권장한다. `table*` 2단 폭 + `\footnotesize` + `tabcolsep` 3pt + max-width cap은 PDF QA에서 이미 확정된 레이아웃이므로 변경하지 않는다.

	#### 📝 캡션 (영문 확정본)

	```
	SWaT dual-condition results: all five metrics for CSMAD and all baselines under the
	full condition and the excl22 condition (Section~\ref{sec:datasets}).
	Same trained models and identical scores in both conditions; only the evaluation mask differs.
	The excl22 best epoch is selected independently under the shared criterion.
	```

	#### ⚠️ 주의사항과 의존성

	Affiliation F1은 반드시 `_ar` 변형(`affiliation_f1_ar`)을 사용한다 — §4.1.3 본문 선언과의 R30 정합이며, F1-최적 threshold 변형은 ranking 비사용으로 선언되어 있다. CSMAD의 두 조건은 **독립 best epoch**임을 캡션 마지막 문장이 이미 공개하고 있으므로, 추출 시 full 조건의 best epoch에서 excl22 지표를 읽는 실수(`metrics_excl_region22` 혼용)를 하지 말 것 — excl22 열은 `A1A2_excl22` entity의 headline `metrics`에서 읽는다(두 값 모두 실존하므로 혼용이 가장 위험하다; RESEARCH_SYNTHESIS §④ α-m3 주석). 의존성: weak 4종 미완 시 해당 4행은 TAB-2의 sync 그룹 B fallback 규칙에 연동되어 함께 빠진다(부분 게재 금지).

	#### 🔢 연결된 수치 placeholder

	직접 파생 없음. TAB-2 의존 사슬에 속하며, NUM-012(CSMAD @ SWaT excl22)와 같은 entity 소스를 공유한다(값의 단일 원천은 TAB-2 쪽).

### TAB-A7 — 전 지표 전수 결과 표 (Table A.7) {toggle="true"}

	| 유형 | 소스 분류 | 우선순위 | 의존성 |
	|---|---|---|---|
	| 결과 표 (§A.5) | **[완주 대기 — TAB-2와 동일 소스]** — 추가 실험·추가 비용 0 | 완주 대기 | TAB-2와 동일 |

	<callout icon="💡" color="blue_bg">
	**한 줄 요약** — 본문 Table 2가 보여주지 않는 나머지 4지표 {PA%K-AUC AUC-PR, VUS-ROC, Affiliation F1, PA F1(oracle)}를 27개 방법 × 7 데이터셋 열로 전수 공개. 신규 실험 0 — metadata에서 metric 키만 추가 추출.
	</callout>

	**위치·내부 키** — 부록 §A.5, `appendix_A.tex`, `\label{tab:full_metrics}`. 내부 키: `pak_auc_prc_auc`, `vus_roc`, `affiliation_f1_ar`, `pa_0_f1`.

	#### 🎯 목적과 의도

	본문 Table 2는 지면 제약상 PA%K-AUC F1과 VUS-PR 2지표로 고정되었다(블루프린트 RT V3 결정). 이 표는 "주 표 2지표 + 전수는 Appendix"라는 §4.1.3의 약속을 이행하는 곳이다. 논증 역할은 두 가지다. 첫째, **지표 선택 공격 방어**: "유리한 지표만 골라 보여준 것 아닌가"라는 공격에 대해 5지표 전수를 공개함으로써 답한다 — 특히 threshold-free 지표(VUS-ROC)와 사건 기반 지표(Affiliation F1)에서도 순위 구도가 유지되는지를 리뷰어가 직접 확인할 수 있게 한다. 둘째, **PA F1(oracle)의 비교 가능성 제공**: 선행 연구 다수가 PA F1을 보고하므로 비교 가능성을 위해 제시하되, oracle threshold 기반임을 명시하고 ranking에서 배제한다는 원칙(R29)을 표 차원에서 구현한다.

	#### 🏁 목표와 기대 결과

	성공 기준: 27 method × 7 데이터셋 열 × 4지표 = 756셀이 모두 채워지고, PA F1 행 전부에 "(oracle)" 라벨이 붙는 것. 기대 패턴: 4지표 간 순위 구도가 본문 2지표와 대체로 일관되면 지표 선택 공격이 무력화된다. PA F1(oracle)은 F1-최적 threshold를 test 라벨로 고르는 지표 특성상 **다른 지표 대비 전반적으로 부풀려진 절대값**을 보일 것으로 기대되며, 이 부풀림 자체가 "oracle 지표를 ranking에 쓰지 않는다"는 §4.1.3 원칙의 시각적 정당화가 된다. 만약 특정 지표에서 순위가 크게 뒤집히는 데이터셋이 있으면 — 예컨대 Affiliation F1에서만 약한 방법 — 이는 본문에서 1문장 관찰로 다룰 수 있는 재료이지 숨길 일이 아니다.

	#### 🧪 실험 내용과 설계

	**신규 실험이 전혀 없다.** TAB-2를 채우는 실행 묶음(271canon 완주분 + [CMP-Q3] 재사용분 + SMD/SMAP/MSL baseline 신규 실행 + weak 4종 신규 실행)의 experiment metadata에서 metric 키 4종만 추가로 추출한다. 단일 평가 루틴 `compute_full_metric_set`이 전 지표를 **같은 best epoch에서** 일괄 산출하므로, 이 표의 한 행과 Table 2의 같은 행은 동일 checkpoint·동일 epoch의 산출물이다 — 추가 비용이 0인 이유다. SMD/SMAP/MSL avg 셀의 집계 규칙은 TAB-2와 동일한 entity 집합·동일 macro 평균이어야 한다. 집계 스크립트는 TAB-2용과 공유하고 metric 키 목록만 늘리는 구현을 권장한다.

	#### 📊 구성과 형태

	method × metric 중첩 행 구조(tex 확정): method당 4개 metric 행 {AUC-PR, VUS-ROC, Aff. F1, PA F1 (oracle)}이 세로로 이어진다. 열은 Table 2와 동일한 7 데이터셋 {SWaT excl22, WaDi A1, WaDi A2, PSM, SMD avg, SMAP avg, MSL avg}. **PA F1 행의 "(oracle)" 라벨은 의무**다 — 생략 시 unfair-threshold 공격이 확실시된다(RESEARCH_SYNTHESIS §④ oracle 표기 의무). 강조(bold/underline)는 지표 행별로 적용할지 생략할지 Phase 7 스타일 판단에 위임하되, 적용한다면 oracle 행은 강조에서 제외하는 쪽이 R29와 정합적이다.

	#### 📝 캡션 (영문 확정본)

	```
	Complete multi-metric results for all methods and dataset families: PA\%K-AUC AUC-PR,
	VUS-ROC, Affiliation F1, and PA F1 (oracle threshold; reported for comparability only, never
	used for ranking --- Section~\ref{sec:metrics}).
	PA\%K-AUC F1 and VUS-PR appear in Table~\ref{tab:main_results}.
	```

	#### ⚠️ 주의사항과 의존성

	키 혼동이 단 하나의 치명 함정이다: PA F1은 F1-최적(oracle) threshold 기반 **`pa_0_f1`**이며, **`pa_0_f1_ar`이라는 키는 존재하지 않는다**(REQUEST-1 RESOLVED). 반면 Affiliation F1은 반대로 `_ar` 변형(`affiliation_f1_ar`)을 쓴다 — 두 지표의 threshold 계열이 서로 다르다는 점을 추출 스크립트에 주석으로 박아둘 것. ranking·본문 서술에 PA F1을 사용하는 것은 어떤 경우에도 금지다(R29). 의존성은 TAB-2와 완전 동일: 271canon 완주(잔여 SMD 6, SMAP 49, MSL 22), baseline SMD/SMAP/MSL 신규 실행, weak 4종 GPU 실행. weak 미완 시 그룹 6 행 4개가 sync 그룹 B 규칙으로 함께 빠진다.

	#### 🔢 연결된 수치 placeholder

	직접 파생 없음. TAB-2 의존 사슬의 일부. 본문 §4.1.3의 "remaining three metrics are in Appendix" 약속 문장이 이 표의 존재에 의존한다.

### TAB-A8 — CSMAD per-entity 전수 결과 표 (Table A.8) {toggle="true"}

	| 유형 | 소스 분류 | 우선순위 | 의존성 |
	|---|---|---|---|
	| 결과 표 (§A.6) | **[완주 대기]** — 신규 실험 없음, 완주 후 집계 1회 | 완주 대기 | 271canon 완주 (유일) |

	<callout icon="💡" color="blue_bg">
	**한 줄 요약** — SMD 28 + SMAP 54 + MSL 27 = 109개 entity 각각의 {PA%K-AUC F1, VUS-PR}. 271canon 완주가 유일한 전제조건이며, 블록별 macro 평균이 Table 2의 family 열과 자리수까지 일치해야 한다.
	</callout>

	**위치·소스** — 부록 §A.6, `appendix_A.tex`, `\label{tab:per_entity_results}`. 소스는 [271c] entity별 `experiment_metadata.json`의 `metrics` dict.

	#### 🎯 목적과 의도

	multi-entity family(SMD/SMAP/MSL)의 Table 2 셀은 macro 평균 한 숫자로 압축되어 있다. 평균은 entity 간 분산을 숨길 수 있으므로, "평균이 소수 entity의 고성능에 끌려간 것 아닌가"라는 집계 공격에 대한 방어는 entity 전수 공개뿐이다. 이 표가 그 공개를 수행한다. 동시에 재현성 장치이기도 하다: 재현 시도자가 특정 machine/channel에서 얻은 수치를 우리 결과와 entity 단위로 직접 대조할 수 있게 한다. 캡션의 "Macro-averages over entities equal the corresponding family columns of Table 2" 문장은 이 표와 본문 표 사이의 **수치 계약**이며, 그 계약의 검증 가능성 자체가 논증 자산이다.

	#### 🏁 목표와 기대 결과

	성공 기준: 109행 × 2지표가 전부 [271c] metadata 실값으로 채워지고, 세 블록의 macro 평균이 Table 2의 SMD/SMAP/MSL avg 셀과 **소수 반올림 자리수까지** 일치하는 것. 기대 패턴: entity 간 성능 분산이 존재하는 것이 자연스럽다 — SMD machine들은 차원(29–36)과 anomaly 구성이 제각각이고 SMAP/MSL channel들은 신호 특성이 이질적이므로, 균일하게 높은 수치보다 분산 있는 분포가 오히려 신뢰할 만한 모양이다. 일부 entity에서 낮은 수치가 나오는 것은 숨길 일이 아니라 평균의 정직성을 보이는 재료다. 다른 패턴(예: 특정 family에서 소수 entity가 평균을 지배)이 관찰되면 본문 §4.2의 family 해석에 1문장 주의를 추가할지 검토한다.

	#### 🧪 실험 내용과 설계

	신규 실험은 없다. 작업 순서는 다음과 같다. 첫째, 271canon 잔여 entity(SMD 6, SMAP 49, MSL 22 — 2026-06-11 실측)의 완주를 기다린다 — **이것이 유일한 전제조건**이다. 둘째, 집계 스크립트가 [271c] 디렉토리를 순회하며 entity별 `experiment_metadata.json`에서 `metrics.pak_auc_f1`과 `metrics.vus_pr`(둘 다 best epoch 기준 — `timing.best_epoch`에서 전 지표가 함께 추출된 값)을 읽어 109행을 생성한다. 셋째, 같은 스크립트 안에서 블록별 macro 평균을 계산해 **Table 2 채움 값과의 일치를 assert로 검증**한다 — 캡션의 계약 문장을 코드 수준에서 보장하라는 spec 권고를 그대로 계승한다. 자리수 처리는 "반올림 후 평균"이 아니라 "평균 후 반올림"으로 통일하고, Table 2 쪽 집계 스크립트와 같은 함수를 쓴다.

	#### 📊 구성과 형태

	3 블록(SMD → SMAP → MSL) 세로 나열, 블록 사이 midrule, 각 블록 머리에 이탤릭 블록 헤더. 행 구조는 {Entity, PA%K-AUC F1, VUS-PR} 3열. entity 명명은 tex stub의 스타일을 따라 "SMD-1-1 / SMAP-A-1 / MSL-C-1" 형식으로 통일한다(내부 디렉토리명 그대로 노출 금지). 각 블록 말미 또는 캡션에 macro 평균 = Table 2 family 열 일치 보장 문구를 유지한다. 109행은 길지만 단일 column `table` 환경으로 처리 가능함이 tex에서 확인되었다(필요 시 2단 분할은 Phase 7 재판단).

	#### 📝 캡션 (영문 확정본)

	```
	Per-entity results (PA\%K-AUC F1\,/\,VUS-PR) for SMD (28 machines), SMAP
	(54 channels), and MSL (27 channels).
	Macro-averages over entities equal the corresponding family columns of
	Table~\ref{tab:main_results}.
	```

	#### ⚠️ 주의사항과 의존성

	이 표의 평균과 Table 2 셀의 **수치 의존성은 단방향이 아니라 동일성**이다 — 두 표가 다른 스크립트로 따로 집계되는 순간 어긋날 위험이 생기므로, 단일 집계 산출물을 공유하라. 부분 완주 상태로 일부 entity만 채우고 나머지를 비워두는 게재는 금지다(TAB-2의 "부분 완주 avg 금지" 규칙이 이 표에는 행 단위로 적용된다). MSL은 27 channels로 표기한다 — SMAP/MSL 합산 81채널 기준의 다른 문서 수치와 혼동하지 말 것. 의존성: 271canon 완주(유일). baseline 실행과는 무관하므로 baseline 큐 지연이 이 표를 막지는 않는다.

	#### 🔢 연결된 수치 placeholder

	직접 파생 없음. Table 2의 SMD/SMAP/MSL avg 셀(N-C 그룹의 입력)과 동일 산출물을 공유한다. sync 그룹 A(N-A, "six")의 성립 조건인 271canon 완주를 이 표가 함께 기다린다.

### TAB-B1 — Contaminated-training(무절제) 조건 비교 표 (Table B.1) {toggle="true"}

	| 유형 | 소스 분류 | 우선순위 | 의존성 |
	|---|---|---|---|
	| 방어 표 (§B.1) | **[신규 실행]** — `comparison/run_baseline_queue.py`, variant `full`(Q1) | 신규 실행 #8 | TAB-2 확정 (Δ 기준) |

	<callout icon="💡" color="blue_bg">
	**한 줄 요약** — 22개 비지도 baseline을 절제 없는 오염 스트림 그대로 학습시킨 결과와, anomaly-excised 조건 대비 변화량 Δ를 공개한다. "절제 조건이 baseline에게 정말 최선이었는가"를 정량화해 학습량 비대칭 인정(R31)을 뒷받침하는 방어 표다.
	</callout>

	**위치·Δ 기준** — 부록 §B.1, `appendix_B.tex`, `\label{tab:contaminated}`. Δ 기준은 TAB-2 확정본의 anomaly-excised 수치 — **TAB-2 완성 후에만 Δ 산출 가능**.

	#### 🎯 목적과 의도

	본문 비교(Table 2)는 비지도 baseline에게 anomaly-excised 조건 — 라벨로 오염 구간을 절제해 주는, 비지도 패러다임에서의 라벨 최선 활용(R12) — 을 제공한다. 그런데 절제는 학습 데이터의 양을 줄이므로, §4.1.4는 "excised 조건의 baseline이 CSMAD보다 적은 train 볼륨을 받는다"는 비대칭을 정직하게 인정한다(R31). 이 표는 그 인정 문장의 **정량 뒷받침**이다: 같은 22개 방법을 절제 없는 동일 오염 스트림(CSMAD가 받는 것과 동일한 train, 라벨 미사용)에서 학습시켜, 절제가 각 방법에게 실제로 이득이었는지(Δ<0이면 contaminated가 더 나쁨 = 절제가 이득)를 방법별로 보인다. 방어하는 공격은 두 갈래다. 첫째, "절제 때문에 baseline이 데이터를 덜 받아 불리했다"는 볼륨 공격 — Δ가 음수 일색이면 "데이터가 더 많아도 오염 때문에 더 나빴다"는 정면 반박이 된다. 둘째, §15의 "성능 우위가 프로토콜(데이터 추가) 때문 아닌가" 공격의 보조 방어 — 비지도 방법은 같은 추가 데이터를 받아도 라벨을 활용하지 못한다는 protocol-effect 논증(TAB-2 하단 블록·NUM-019)의 contaminated 측 실측을 이 실행 묶음이 공급한다.

	#### 🏁 목표와 기대 결과

	성공 기준: 22개 baseline × 대표 3 family의 contaminated-training F1과 Δ가 채워지고, CSMAD 참조 행이 Table 2에서 복사되는 것. 기대 패턴: **대부분의 비지도 baseline에서 Δ < 0(절제 우세)** — 비지도 방법에게 train 내 anomaly는 순수 오염원이므로, 무절제 조건에서 정상 프로파일 학습이 교란되어 성능이 내려가는 것이 패러다임의 예측이다. 이 패턴이 확인되면 ① 오염이 실제 해를 끼친다는 문제 설정의 실재성, ② Table 2의 excised 조건이 baseline에게 유리한(관대한) 조건이었다는 비교 공정성이 동시에 입증된다. 다른 패턴의 해석: 특정 방법에서 Δ > 0(무절제 우세)이 나오면, 그 방법은 절제로 인한 데이터 손실·경계 단절의 비용이 오염 비용을 상회한 경우다 — 방법별 1문장 관찰로 다루되, 다수 방법이 Δ > 0이면 §4.1.4의 조건 정당화 서술 자체를 재조정해야 한다(침묵 게재 금지). CSMAD 행은 두 조건 모두 동일한 contaminated train이므로 Δ 정의상 "—"다.

	#### 🧪 실험 내용과 설계

	실행은 baseline 비교 파이프라인 한 줄기다. `comparison/run_baseline_queue.py --queue <json>`으로 **22종 × 대표 3 family(SWaT, PSM, SMD) × variant `full`(Q1, contaminated-training)** 큐를 구성해 실행한다. Q1 항목은 각 baseline의 `experiment_configs.py`에 이미 등록되어 있으므로 신규 구현 없이 큐 구성만 하면 된다. 기존 Q1 결과 폴더 `1_20260312_*`는 per-entity 정규화(2026-06-02) 이전의 구버전이므로 재사용 금지 — 전량 재실행이다. SMD 실행 전에 **per-entity 정규화 적용을 반드시 확인**한다(STALE 원인 재발 방지; entity별 train 구간 scaler fit — `entity_norm_segments` 경유). 학습 budget·평가 cadence·best-epoch 기준(매 epoch eval, `pak_auc_f1`)은 main 프로토콜과 동일하게 유지한다 — 변하는 것은 train 데이터 구성(절제 없음)뿐이다.

	Δ 산출은 실행과 분리된 후처리다: Δ = (contaminated F1) − (anomaly-excised F1), 양수 = contaminated 우세. 기준값은 **TAB-2 확정본**의 anomaly-excised 수치에서만 가져온다 — SMD avg 열의 기준값은 baseline SMD 신규 실행(실행 #1)이 끝나야 존재하므로, 이 표의 완성은 TAB-2 unsupervised 행 완성 이후로 순서가 강제된다. 평가는 main과 동일한 held-out 평가 절반에서 수행되므로 조건 간 비교가 train 구성 차이만 분리한다.

	#### 📊 구성과 형태

	23행(22 baseline + CSMAD 참조) × 6열: {SWaT excl22, PSM, SMD avg} × {F1, Δ}. Δ 열은 부호 명시(+/−). tex 확정 구조는 family당 `\cmidrule` 2열 블록이다. registry 원안(전 family × {F1, VUS-PR, Δ})에서 지면 축소된 형태가 tex에 확정되어 있으므로 **tex가 우선**이다. CSMAD 참조 행의 Δ 셀은 "—".

	#### 📝 캡션 (영문 확정본)

	```
	Contaminated-training (no-excision) condition results for all 22 unsupervised
	baselines. Each method trains on the identical contaminated training stream used by CSMAD
	(no anomaly excision; labels unused) and is evaluated on the identical held-out evaluation
	half. Metrics: PA\%K-AUC F1 and VUS-PR per dataset family; $\Delta$ columns give the change
	relative to the anomaly-excised condition of Table~\ref{tab:main_results} (positive =
	contaminated-training better). The CSMAD row is repeated from Table~\ref{tab:main_results}
	for reference, as CSMAD trains on the contaminated stream in both conditions.
	```

	#### ⚠️ 주의사항과 의존성

	**캡션-표 불일치가 이 placeholder의 고유 함정이다**: 캡션은 "PA%K-AUC F1 and VUS-PR"을 약속하는데 tex 표 stub은 F1/Δ 열만 노출한다. Phase 8 채움 시점에 둘 중 하나로 정합화해야 한다 — 권고는 표를 F1+Δ로 확정하고 캡션의 "and VUS-PR"을 삭제하는 쪽(추가 열 없이 지면 유지)이며, VUS-PR 열을 추가하는 선택도 가능하나 어느 쪽이든 침묵 불일치는 금지다. 둘째, Δ의 기준값 의존성: TAB-2의 anomaly-excised 수치가 확정되기 전에 Δ를 선산출하지 말 것(기준이 바뀌면 Δ 전량 재계산). 셋째, 이 실행 묶음은 NUM-019(protocol-effect 블록의 "같은 추가 데이터에 대한 비지도 변화량")의 contaminated 측 소스로 공유될 수 있다 — N-D 그룹과의 소스 공유를 집계 스크립트에 명시하라. 넷째, SWaT 재실행이 발생하므로 입력 차원 45 일치 검증이 필수다(FEEDBACK-7 — 현 raw CSV 경로는 51을 반환; 상수 6컬럼 필터 확인).

	#### 🔢 연결된 수치 placeholder

	| ID | 위치 | 관계 |
	|---|---|---|
	| **NUM-019** | §4.2 protocol-effect 문장 | best unsupervised의 standard→contaminated 변화량 — 비교쌍의 contaminated 측 실측을 이 실행 묶음과 공유 가능 (N-D 그룹, 주 소스는 TAB-2 ② 4항) |

### TAB-B2 — Epoch-budget 민감도 표 (Table B.2) {toggle="true"}

	| 유형 | 소스 분류 | 우선순위 | 의존성 |
	|---|---|---|---|
	| 방어 표 (§B.2) | **[신규 실행(부분 재사용)]** — baseline 50/100ep 신규, CSMAD 축소분은 exp298/299 재사용 | 신규 실행 #10 | TAB-3 대표 열(약한 의존) |

	<callout icon="💡" color="blue_bg">
	**한 줄 요약** — budget 비대칭(CSMAD 500 vs unsup 10 vs weak 50)과 그에 따른 선택-기회 비대칭(100 vs 10 checkpoints)이 비교 결론을 바꾸지 않음을 실측으로 보이는 방어 표다. baseline은 budget을 늘려 보고, CSMAD는 줄여 본다.
	</callout>

	**위치·방어 대상** — 부록 §B.2, `appendix_B.tex`, `\label{tab:epoch_sensitivity}`. 방어 대상: §15 "epoch budget 비대칭 불공정"(ADV BLK-005) + "test-set model selection" 시나리오의 보조 실측.

	#### 🎯 목적과 의도

	§4.1.2는 학습 budget 비대칭(500/50/10 epochs)을 은폐 없이 공개하고, 모든 방법이 "주기 평가 후 best-epoch 선택"이라는 동일 구조를 따른다고 방어한다. 그러나 평가 cadence가 고정이므로 budget 비대칭은 **선택 기회의 비대칭**을 수반한다 — §B.2 본문이 명시하듯 CSMAD는 100개 checkpoint(500 epochs ÷ 5), 비지도 baseline은 10개에서 best를 고른다. "더 많은 추첨 기회가 더 좋은 best를 만든 것 아닌가"라는 공격(ADV BLK-005, test-set selection 시나리오의 파생형)에 대한 정면 실측 답변이 이 표다. 설계는 양방향이다: 대표 비지도 baseline의 budget을 50/100으로 **늘려서** 기회를 더 줘 보고, CSMAD의 budget을 **줄여서** 기회를 빼앗아 본다. 양쪽 모두에서 순위 구도가 유지되면 비대칭은 결과의 원인이 아니다.

	#### 🏁 목표와 기대 결과

	성공 기준: Anomaly Transformer·TranAD의 10/50/100 epochs 성능과 CSMAD의 축소 budget/500 epochs 성능이 채워지고, §B.2 본문의 "selection-frequency effect together with the training-length effect" 서술과 표가 맞물리는 것. 기대 패턴: **baseline은 budget을 5–10배 늘려도 큰 향상이 없거나 과적합으로 오히려 하락**하고(소형 모델의 단기 수렴 — budget 책정의 근거였던 수렴 특성 그대로), **CSMAD는 축소 budget에서도 경쟁력을 유지**하는 것이다. 이 패턴이 확인되면 "budget을 맞춰도 결론이 같다"는 방어가 완성된다. 다른 패턴의 해석: baseline이 50/100 epochs에서 유의미하게 상승해 순위가 좁혀지면 budget 비대칭이 결과에 기여했다는 뜻이므로, §4.1.2의 공개 문구를 강화하고 본문 해석을 보수적으로 수정해야 한다. CSMAD가 축소 budget에서 크게 무너지면 그것은 "장기 수렴이 필요한 설계"라는 사실의 공개 재료이지 은폐 대상이 아니다 — warmup 의존성 서술(§3.5)과 연결해 해석한다.

	#### 🧪 실험 내용과 설계

	네 갈래 소스를 조립한다.

	| 셀 그룹 | 소스 | 실행 지침 |
	|---|---|---|
	| baseline 10 epochs | **[재사용]** [CMP-Q3] | main budget 결과 그대로 — Table 2와 동일 값 |
	| baseline 50/100 epochs | **[신규 실행]** | `baseline_common.py`의 epochs override로 2 모델(Anomaly Transformer, TranAD) × 2 budget(50, 100) × 대표 데이터셋(2–3개, **TAB-3 대표 선택과 통일 권장**) 실행. 평가 cadence(매 epoch)·best-epoch 선택 구조는 main과 동일 유지 — checkpoint 수가 budget에 비례해 늘어나는 것이 설계 의도다 |
	| CSMAD 500 epochs | **[재사용]** [271c] | main 결과 그대로 |
	| CSMAD 축소 budget | **[재사용 — 결정 1건 필요]** | exp298(`num_epochs=300, warmup=150`)·exp299(`num_epochs=200, warmup=100`) 완주분이 실재한다(2026-06-11 실측). 단 tex stub의 열 라벨이 "100 epochs"이므로 다음 중 하나를 결정: **(i) exp299(200ep)를 쓰고 열 라벨을 "reduced (200)"로 수정** — 추가 실행 0, warmup 비율 보존 (권고안), (ii) `num_epochs=100, teacher_only_warmup_epochs=50` 신규 1 run |

	핵심 설계 제약: CSMAD의 축소 budget에서는 **warmup도 비례 축소**되어야 한다. warmup=250을 고정한 채 epochs=100으로 줄이면 student가 아예 학습되지 않는 무의미 변형이 된다(student 학습 개시가 epoch 250이므로). exp298/299는 이미 이 비례(절반)를 따르고 있어 그대로 쓸 수 있다 — 권고안 (i)을 채택하면 신규 실행이 0건이 된다.

	#### 📊 구성과 형태

	행 = method {Anomaly Trans., TranAD, CSMAD}, 열 = budget {10, 50, 100(또는 reduced), 500} epochs. 각 method의 비해당 budget 셀은 "—" (baseline 행의 500 셀, CSMAD 행의 10/50 셀). 지표는 PA%K-AUC F1 단일, best-epoch 기준 main과 동일. 권고안 (i) 채택 시 CSMAD 열 라벨을 "reduced (200)"로 바꾸고 캡션의 "a reduced budget" 표현은 그대로 유효하다(캡션이 구체 숫자를 약속하지 않으므로 캡션 수정 불필요 — 열 라벨만 수정).

	#### 📝 캡션 (영문 확정본)

	```
	Epoch-budget sensitivity. PA\%K-AUC F1 of representative unsupervised baselines
	trained for 10 (main budget), 50, and 100 epochs, and of CSMAD trained for 500 (main budget)
	and a reduced budget, on representative datasets; best-epoch selection identical to the main
	protocol (Section~\ref{sec:impl}).
	```

	#### ⚠️ 주의사항과 의존성

	첫째, 위에 적은 warmup 비례 축소가 유일한 치명 함정이다 — 신규 run을 택할 경우 `teacher_only_warmup_epochs`를 반드시 함께 줄일 것. 둘째, baseline 50/100 run의 best-epoch 선택 구조(매 epoch eval 후 best)를 main과 동일하게 유지해야 "선택 기회를 늘려준" 실험이 된다 — cadence를 바꾸면 변인이 둘이 된다. 셋째, 대표 데이터셋은 TAB-3 선택(NUM-020)과 통일을 권장하므로 TAB-3의 대표 데이터셋 확정에 약한 의존이 있다(통일하지 않아도 표는 성립하나 서사 일관성이 떨어진다). 넷째, exp299 재사용 결정 (i)/(ii)는 채움 전에 한 번만 내리면 되는 설계 결정이며, 결정 내용을 DECISION_LOG에 남긴다.

	#### 🔢 연결된 수치 placeholder

	직접 파생 없음. §4.1.2의 budget 공개 문장과 §15 방어 표의 "(옵션) Appendix epoch-budget sensitivity" 항목이 이 표를 가리킨다 — inline NUM은 없으나 본문-부록 상호참조 정합 의무가 있다.

### TAB-B3 — 추론 연산 비용 표 (Table B.3) {toggle="true"}

	| 유형 | 소스 분류 | 우선순위 | 의존성 |
	|---|---|---|---|
	| 비용 측정 표 (§B.3) | **[신규 측정]** — 학습 불필요, 측정 스크립트 1회 | 측정 즉시 가능 | TXT-001 (하드웨어) 선행 권장 |

	<callout icon="💡" color="blue_bg">
	**한 줄 요약** — leave-one-out 추론(50 마스킹 패턴)의 비용을 single-mask 기준선 대비 {FLOPs, wall-clock, peak memory}로 실측한다. wall-clock 배율 실측값이 NUM-031이 되어 §5의 "approximately 50×" 표현과 동기화된다.
	</callout>

	**위치·파생 NUM** — 부록 §B.3, `appendix_B.tex`, `\label{tab:compute}`. 파생 NUM: **NUM-031**(§B.3 본문의 wall-clock overhead factor).

	#### 🎯 목적과 의도

	CSMAD의 추론은 윈도당 50개 leave-one-out 마스킹 패턴을 평가하므로 연산량이 단일-pass 대비 약 50배다 — 이 비용은 발표 단계부터 공개된 설계 한계이며, §4.2 분석 텍스트와 §5 한계 문장이 모두 이를 인정한다. 이 표의 논증 역할은 **한계 인정의 정직성을 실측으로 완성**하는 것이다: "비용이 크다"는 추상 인정 대신, FLOPs·wall-clock·메모리 세 축의 측정값과 배율을 제시한다. 특히 wall-clock은 batch 차원 병렬화 덕분에 FLOPs 배율(이론 ~50×)보다 낮게 나올 수 있는데, 이 간극 자체가 유용한 정보다 — "연산량은 50×지만 실제 시간 비용은 그보다 작다"는 서술이 가능해지면 한계의 체감 크기가 정확해진다. 방어하는 공격은 "비용 한계를 숨기거나 과소 서술했다"는 유형이며, 측정 방법까지 본문에 명시함으로써 차단한다.

	#### 🏁 목표와 기대 결과

	성공 기준: 3×3 표가 전부 측정값으로 채워지고, NUM-031(wall-clock 배율)이 §B.3 본문에 들어가는 것. 기대 패턴: FLOPs 배율은 패턴당 1 forward 구조상 **~50× 근방**(leave-one-out 50패턴; batch 확장은 wall-clock 병렬화일 뿐 연산량을 줄이지 않는다 — RESEARCH_SYNTHESIS 표A). wall-clock 배율은 GPU 병렬화·메모리 재사용 효과로 **50×보다 낮을 가능성**이 있다. peak memory는 leave-one-out 쪽이 batch 확장만큼 높게 나오되 `patch_batch_size=2` 분할로 상한이 관리된다. **sync 규칙(중요)**: 측정된 wall-clock 배율이 50보다 유의미하게 낮으면 §5의 "approximately 50×"를 "up to 50×"로 완화한다 — registry §5 audit-trail 규칙이며, §5 문장과 같은 pass에서 수정한다. 배율이 50에 근접하면 표현은 그대로 둔다.

	#### 🧪 실험 내용과 설계

	학습이 필요 없는 측정 스크립트 1회다. [271c] 대표 entity 1–2개의 best checkpoint를 로드해 두 채점 모드를 동일 조건에서 측정한다.

	**Leave-one-out 측정**: 현행 evaluator의 추론 경로를 **그대로** 사용한다(50개 마스킹 패턴 batch-병렬, `evaluator.py`의 단일 forward 확장 구현). end-to-end 평가 wall-clock은 해당 entity metadata의 `timing.inference_time`과 교차 검증한다 — 측정 스크립트 값과 운영 기록 값이 크게 어긋나면 측정 조건(batch, device 상태)을 재점검한다.

	**Single-mask 측정**: 동일 checkpoint로 윈도당 1-pass(단일 마스킹 패턴) 채점 모드를 측정용으로 구성한다. 이 모드는 비교 기준선일 뿐 **논문 점수 산출에는 사용되지 않음**을 표 각주에 명시한다(미사용 옵션을 검증된 경감책처럼 보이게 하지 말 것 — §5의 complementary masking 서술 규칙과 같은 정신).

	**측정 사양**: FLOPs는 분석식 또는 profiler(예: torch profiler) 중 하나로 산출하고 **어느 방법을 썼는지 §B.3 본문에 1줄 명시**한다. peak memory는 `torch.cuda.max_memory_allocated()`를 reset 후 측정한다. 두 모드는 **동일 batch 크기·동일 entity**로 측정해야 배율이 의미를 가진다. 측정 하드웨어는 TXT-001 확정 GPU와 동일해야 하며, 학습 머신과 다르면 각주로 구분 명시한다.

	#### 📊 구성과 형태

	tex 확정 3×3 구조: 행 {Single-mask, Leave-one-out, Overhead ×} × 열 {FLOPs / window, Wall-clock (s/entity), Peak GPU mem. (GB)}. Overhead 행은 비율(×)만 기재하고 memory 열의 Overhead 셀은 "—"(배율 개념이 부적합). 측정 entity가 2개면 본문 또는 각주에 entity별 값의 처리(평균 또는 병기)를 명시한다.

	#### 📝 캡션 (영문 확정본)

	```
	Computational cost of CSMAD inference: per-window forward FLOPs, end-to-end wall-clock
	evaluation time, and peak GPU memory for leave-one-out masking versus single-mask scoring,
	measured on representative datasets (hardware of \ref{sec:appendix_impl}).
	```

	#### ⚠️ 주의사항과 의존성

	NUM-031 sync 조건이 이 토글의 존재 이유 절반이다 — 표만 채우고 §5 문장을 잊으면 본문-부록 모순이 생긴다(아래 NUM 표 참조). 하드웨어 표기는 TXT-001 확정값과 같은 섹션(§A.1)을 참조하므로 TXT-001 확인이 선행되어야 한다. 측정 스크립트에서 score 후처리 smoothing 등 어떤 추가 연산도 끼워 넣지 말 것(R34 — 측정 대상은 현행 채점 경로 그대로). [271c] checkpoint 로드 시 SWaT를 쓴다면 입력 차원 45 일치 검증(FEEDBACK-7)이 여기에도 적용된다 — 차원 불일치 시 checkpoint 로드 자체가 실패할 수 있으므로 PSM 등 무리스크 entity를 우선 후보로 권장한다.

	#### 🔢 연결된 수치 placeholder

	| ID | 위치 | 정의·규칙 |
	|---|---|---|
	| **NUM-031** | §B.3 본문 (`appendix_B.tex` "the measured wall-clock overhead factor is [X.XX]") | leave-one-out vs single-mask **wall-clock 배율 실측값**. sync 조건: 50보다 유의미하게 낮으면 §5 "approximately 50×" → "up to 50×" 완화를 같은 pass에서 수행 (그룹 N-H) |

### TAB-B4 — 확장 ablation 표 (Table B.4) {toggle="true"}

	| 유형 | 소스 분류 | 우선순위 | 의존성 |
	|---|---|---|---|
	| 확장 ablation 표 (§B.5) | **[재사용(no_fm)]** + **[신규 실행(3종)]** | 신규 실행 #5(최우선)·#9 | TAB-3 대표 열(NUM-020) 종속 |

	<callout icon="💡" color="blue_bg">
	**한 줄 요약** — 본문 Table 3 밖의 변형 4행(w/o FM, w/o warmup, symmetric decoder)과 Teacher 깊이 민감도 3행을 다룬다. symmetric decoder 행은 contribution bullet 3의 유일한 정량 근거(NUM-024)를 만드는 **신규 실행 최우선** 항목이다.
	</callout>

	**위치·파생 NUM** — 부록 §B.5, `appendix_B.tex`, `\label{tab:extended_ablations}`. 파생 NUM: **NUM-024**(load-bearing), **NUM-025**.

	#### 🎯 목적과 의도

	본문 Table 3은 라벨 경로 3종의 분해(4행)로 고정되었고, 그 밖의 설계 요소 검증은 전부 이 표가 담당한다. 논증 역할은 행마다 다르다. **Symmetric decoder 행**이 가장 무겁다: contribution bullet 3(비대칭 capacity gap이 신뢰할 만한 anomaly 신호를 만든다)의 **유일한 정량 근거**이며, 블루프린트 §6.7이 "미실행 + load-bearing — Phase 5 진입 전 실행 필수"로 못 박은 항목이다(warmup 공격 패턴이 bullet 3에서 재발하는 것을 차단). **w/o FM 행**은 §12 R10 논증표의 "FM ablation 근거 필요(미존재)" 공백을 정량 해소한다. **w/o warmup 행**은 "warmup ablation 없음" 공격(§15)에 대한 실측 대응이되, warmup은 contribution이 아니므로(블루프린트 결정 ①) 부록 배치가 논증 강도와 맞다. **depth sensitivity 블록**은 "왜 하필 3L/2L인가"라는 설계 선택 질문에 대해 3/2/1 스윕으로 답한다 — gap 크기 선정의 이론적 정당화 부족(RESEARCH_SYNTHESIS 표A)을 실측으로 보완하는 장치다.

	#### 🏁 목표와 기대 결과

	성공 기준: 상단 4행 + 하단 3행 × (TAB-3과 동일한 대표 데이터셋 + Avg) 열이 채워지고, NUM-024/025가 §B.5 문단 2개에 들어가는 것. 기대 패턴: 각 변형 행이 Full 대비 **하락**하는 것 — 특히 symmetric 행의 하락폭(NUM-024)은 capacity gap의 효과를, w/o FM의 하락폭(NUM-025)은 student 표현 붕괴 방지 효과를 정량화한다. depth 블록에서는 3 → 2 → 1로 갈수록 하락이 깊어지는 단조 경향이 나오면 "Teacher가 Student보다 깊어야 한다"는 설계 원리가 스윕 차원에서 지지된다. 다른 패턴의 해석: 어떤 변형이 Full보다 **좋게** 나오면(음수 하락폭) 본문 부호 규약("removal costs X points")을 쓸 수 없으므로 해당 문장을 "improves by"로 재작성하고, symmetric이 그 경우라면 contribution bullet 3의 표현 강도를 반드시 하향한다 — 결과 확인 전 문장 선점은 금지다(A8). w/o warmup이 무해하게 나오면 warmup을 안정화 장치로만 서술하는 현 기조가 오히려 강화된다.

	#### 🧪 실험 내용과 설계

	행별 소스와 실행 지침을 전수 명세한다. 신규 학습은 **3 run × 대표 데이터셋**뿐이다.

	| 행 | 소스 | 실행 지침 |
	|---|---|---|
	| Full model | [271c] **[재사용]** | TAB-3 행1과 **동일 값** — 별도 추출 금지, 같은 집계 산출물 공유 |
	| w/o FM loss | **exp285_no_fm [재사용]** | `use_feature_matching=False` 단독 diff로 실측 확인된 기존 run, 대표 데이터셋 완주 상태 — metadata 집계만. NUM-025 파생 |
	| w/o Teacher warmup (250→0) | **[신규 실행]** | 큐 신규 항목: `teacher_only_warmup_epochs=0`, 그 외 271 canon 동일. **인지 사항**: λ_rev ramp의 분모가 `num_epochs − warmup`이므로 warmup=0이면 sigmoid ramp가 epoch 0부터 시작한다 — 이는 의도된 변형이며 버그가 아니다 |
	| Symmetric dec. (2L/2L) | **[신규 실행]** | `num_teacher_decoder_layers=2` (Student 2 유지). **NUM-024 파생 — 신규 실행 중 최우선** (기여 bullet 3 load-bearing) |
	| Teacher depth 3 (default) | = Full 행 **[재사용]** | 같은 값의 중복 기재 (블록 비교 가독성용) |
	| Teacher depth 2 | = Symmetric run과 **동일 config** | 같은 run으로 두 행을 채운다 — 이중 실행 불필요 |
	| Teacher depth 1 | **[신규 실행]** | `num_teacher_decoder_layers=1`, 그 외 271 canon 동일 |

	큐 등재는 `configs/queue_dedup_renumbered_v5.json` 형식(`exp_num` / `dataset` 리스트 / `config_override` 공백 구분 키=값)을 따른다. 큐 항목 작성 시 `config_override`에 같은 키를 중복 기재하는 패턴(exp287의 `force_mask_anomaly` True→False last-wins 사례, OBS-2)을 답습하지 말 것 — 키는 1회만, 최종값으로 기재한다.

	#### 📊 구성과 형태

	상단 블록(변형 4행)과 하단 블록(depth 3행)을 midrule로 분리(tex 확정). **열 집합은 TAB-3과 글자 단위 동일**해야 한다 — 대표 데이터셋 선정(NUM-020)이 TAB-3에서 확정되면 이 표가 그대로 따른다. 지표는 PA%K-AUC F1, best epoch 기준 main과 동일. 행 라벨은 tex 확정 표기를 따른다: "w/o FM loss", "w/o Teacher warmup (250→0)", "Symmetric dec. (2L/2L)", "Teacher depth 3 (default)/2/1".

	#### 📝 캡션 (영문 확정본)

	```
	Extended ablations: the variants beyond the confirmed rows of
	Table~\ref{tab:ablation} --- w/o FM loss, w/o Teacher-only warmup (250$\to$0), and a
	symmetric decoder (Teacher 2L\,/\,Student 2L) --- and a Teacher-decoder depth sensitivity
	study (3/2/1 layers against the 2-layer Student). PA\%K-AUC F1 on the ablation datasets of
	Table~\ref{tab:ablation}.
	```

	#### ⚠️ 주의사항과 의존성

	첫째, **conditional 게재 규칙**: symmetric-decoder run이 게재 시점까지 미완이면 contribution bullet 3을 "design principle" 수준으로 표현 강도 하향한다(Phase 6 규칙 — landing spot은 이미 §B.5로 확보됨). 미완 행을 placeholder 상태로 본문에 남기는 것은 금지다. 둘째, §B.5의 본문 문단 2개("Symmetric decoder capacity", "FM loss regularizer")가 NUM-024/025를 들고 있으므로 **표와 문단 수치를 같은 pass에서 동시 갱신**한다. 셋째, 열 집합의 TAB-3 종속: NUM-020 확정 전에 이 표의 열을 독자적으로 정하지 말 것(열 불일치 금지). 넷째, depth 2 행과 symmetric 행이 같은 run임을 집계 스크립트에 명시해 두 행이 미래에 어긋나는 것을 방지한다. 다섯째, exp285 재사용 시 단독 diff(`use_feature_matching=False`) 여부를 metadata에서 한 번 더 확인하고 쓴다 — exp290(no_fm+no_grl 복합)과 혼동 금지.

	#### 🔢 연결된 수치 placeholder

	| ID | 위치 | 정의·규칙 |
	|---|---|---|
	| **NUM-024** | §B.5 "Symmetric decoder capacity" 문단 | Full 행 − Symmetric 행의 Avg 차 (**기여 bullet 3 load-bearing**). 부호 규약: 양수 하락폭 — 음수면 문장 자체를 "improves by"로 재작성 + bullet 3 강도 하향 (그룹 N-E) |
	| **NUM-025** | §B.5 "FM loss regularizer" 문단 | Full 행 − w/o FM 행의 Avg 차. exp285 재사용으로 산출 (그룹 N-E) |
	| **NUM-020** | §4.3 (TAB-3 소유) | ablation 대표 데이터셋 수 — 이 표의 열 집합이 종속되는 외부 결정 |

### FIG-B1 — 파라미터 민감도 2-패널 곡선 (Figure B.1) {toggle="true"}

	| 유형 | 소스 분류 | 우선순위 | 의존성 |
	|---|---|---|---|
	| 민감도 곡선 (§B.4) | **[재사용(좌패널 c 재채점)]** + **[신규 실행(우패널 ρ 재학습)]** | 좌패널 즉시 / 우패널 #11 | 큐 미등재(우패널) |

	<callout icon="💡" color="blue_bg">
	**한 줄 요약** — score 결합비 c(기본 4)와 masking ratio ρ(기본 0.15)의 민감도 곡선. 좌패널은 재학습 없는 재채점(c는 추론 전용 파라미터), 우패널은 ρ별 전체 재학습 — 두 패널의 비용 구조가 본질적으로 다르다.
	</callout>

	**위치·크기** — 부록 §B.4, `appendix_B.tex`, `\label{fig:param_sensitivity}`. 크기 ~3.5 cm ≈ 0.30p. 파생 NUM: 없음(직접).

	#### 🎯 목적과 의도

	CSMAD는 단일 설정(per-dataset 튜닝 없음)으로 전 entity를 학습한다 — Table A.1 캡션이 이를 명시한다. 이 그림의 논증 역할은 그 단일 설정의 두 핵심 자유도(score 결합비 c=4, masking ratio ρ=0.15)가 **요행으로 고른 값이 아님**을 보이는 것이다. 방어하는 공격은 "하이퍼파라미터를 test 성능으로 튜닝했다 / 기본값이 cherry-pick이다" 유형이다. 기본값 주변에서 성능 곡선이 평탄하면 "이 설계는 c·ρ 선택에 민감하지 않다"는 견고성 서술이 가능해지고, 반대로 민감하다면 그 사실을 공개하고 기본값 선택의 근거를 서술하는 것이 정직한 대응이다. 두 패널은 또한 점수 산식의 두 구성 요소(recon-disc 결합, 마스킹 예산)가 §3의 설계 논증과 연결되는 실측 단면이기도 하다.

	#### 🏁 목표와 기대 결과

	성공 기준: 좌패널 c ∈ {1, 2, 4, 8, 16} 5점 곡선과 우패널 ρ ∈ {0.05, 0.10, 0.15, 0.20, 0.30} 5점 곡선이 대표 데이터셋별로 그려지고, 기본값 위치(c=4, ρ=0.15)가 시각적으로 표시되는 것. 기대 패턴: **기본값 근방의 평탄한(완만한) 곡선** — c는 adaptive scaling이 스케일 차이를 이미 보정한 뒤의 비율이므로 광역에서 완만할 가능성이 높고, ρ는 너무 작으면(0.05 → |M|=2) 학습 신호가 빈약해지고 너무 크면(0.30 → |M|=15) 가시 맥락이 부족해지는 완만한 단봉 형태가 자연스러운 예측이다. 다른 패턴의 해석: c 극단(1 또는 16)에서 급락하면 두 점수 성분의 상보성(둘 다 필요함)이 오히려 강조된다 — §4.5 성분 분해 서사와 연결. ρ에서 기본값이 최적이 아니면 그 사실을 그대로 보고한다(기본값은 main 실험 전 결정된 값이지 사후 최적값 주장이 아님을 본문 1문장으로 명시하면 된다). 어느 경우든 **수치 발명 없이 곡선 확정 후 서술**한다.

	#### 🧪 실험 내용과 설계

	두 패널의 비용이 본질적으로 다르므로 분리 실행한다.

	**좌패널 — c sweep [재사용 + 재채점]**: c(= `score_recon_disc_ratio`, 기본 4)는 **추론 시에만** 점수식에 들어간다 — `mae_anomaly/scoring.py`의 score = recon + scaled_disc/c. 따라서 재학습이 전혀 필요 없다. [271c]의 best checkpoint(또는 저장된 per-patch score 성분)에 대해 c ∈ {1, 2, 4, 8, 16}(log2 격자, 기본 4 중심)로 재채점 → 재평가만 수행한다. 2026-06에 정비된 eval-recompute 도구 경로를 재사용할 수 있다. 대표 데이터셋은 FIG-3과 동일 선택을 권장한다(권장 SWaT excl22 + PSM). **핵심 제약**: c별로 best epoch을 재선정하지 말 것 — **main run의 best epoch을 고정**한 채 c만 바꿔야 "그 설정 주변의 민감도"가 된다. 재선정하면 test-set selection이 c에도 적용되어 별개 실험이 되어버린다. 본문에 best-epoch 고정 방식을 1줄로 명시하는 것을 권장한다.

	**우패널 — ρ sweep [신규 실행]**: ρ는 학습 마스킹을 바꾸므로 ρ별 **전체 재학습**이 필요하다. 격자 ρ ∈ {0.05, 0.10, 0.15, 0.20, 0.30} 중 기본 0.15는 [271c]를 재사용하므로 신규는 4 run × 대표 2–3 데이터셋이다. 큐 항목은 `config_override`에 `masking_ratio=<ρ>`만 변경하고 그 외 271 canon 동일(500 epochs, seed 42). ρ 변경 시 |M| = round(50×ρ)로 자동 변동함을 인지한다(0.05 → 2패치, 0.30 → 15패치).

	#### 📊 구성과 형태

	가로 2-패널. 좌: X = c(**log scale 권장**), 우: X = ρ(선형). Y 공통: PA%K-AUC F1. 패널별로 대표 데이터셋당 1선씩, 범례에 데이터셋명. 기본값 위치(c=4, ρ=0.15)에 수직 참조선 또는 강조 마커. 기호는 반드시 ρ를 사용한다(구표기 r_m 금지 — v2-r3 M-5). 높이 ~3.5 cm(≈0.30p)로 두 패널이 한 줄에 들어가는 컴팩트 구성이다.

	#### 📝 캡션 (영문 확정본)

	```
	Parameter sensitivity. PA\%K-AUC F1 as a function of (\textit{left}) the score
	combination ratio $c$ around its default 4 and (\textit{right}) the masking ratio $\rho$
	around its default 0.15, on representative datasets; all other settings fixed to the main
	configuration.
	```

	#### ⚠️ 주의사항과 의존성

	첫째, 좌패널의 best-epoch 고정 규칙(위 🧪)이 이 그림의 과학적 성립 조건이다 — 위반하면 그림 전체가 별개 실험이 된다. 둘째, 우패널은 **현재 어느 큐에도 등재되어 있지 않다** — 신규 등재가 필요하다. 기존 큐 295–303은 전부 다른 변형이다: 295/296/300–303 = window/patch 크기 sweep, 297 = dynamic d_model, 298/299 = epoch-budget 변형이며, 큐 v5 전 32항목에서 `masking_ratio` override는 0건으로 실측 확인되었다(r2 정정). 셋째, c sweep 재채점 시 점수 산식은 `mae_anomaly/scoring.py` 단일 원천만 사용한다 — 다른 곳에 식을 복제하지 말 것(CLAUDE.md API 체크리스트 3항, FM-omission 사고 재발 방지). 넷째, 시각화 코드에 후처리 smoothing을 넣지 말 것(R34).

	#### 🔢 연결된 수치 placeholder

	직접 파생 없음. 대표 데이터셋 선택을 FIG-3(NUM-026)과 통일 권장 — 약한 설계 의존만 존재.

---

## 4. 알고리즘 · 환경 정보

### ALG-C1 — CSMAD 학습 의사코드 검증 (Algorithm C.1) {toggle="true"}

	| 유형 | 소스 분류 | 우선순위 | 의존성 |
	|---|---|---|---|
	| 의사코드 검증 (§C.3) | **[제작 — 코드 대조 검증]** — 실험 소스 없음 | 재사용 묶음 | trainer/model/loss.py 정본 |

	<callout icon="💡" color="blue_bg">
	**한 줄 요약** — 의사코드 초안은 이미 tex에 있다. 남은 작업은 실험이 아니라 **canonical training loop와의 행 단위 대조 검증** + 캡션의 "(pseudocode placeholder)" 꼬리 제거다. 모든 줄은 `trainer.py`/`model.py`/`loss.py`의 실제 동작에 1:1 대응해야 한다.
	</callout>

	**위치·정본** — 부록 §C.3, `appendix_C.tex`, `\label{alg:training}`, algorithm2e `algorithm*` 2단 폭 ~30줄. 정본: 271_CONFIG_TRUTH r4 §VIII + `trainer.py` / `model.py` / `loss.py`.

	#### 🎯 목적과 의도

	재현성 주장의 마지막 조각이다: 하이퍼파라미터 전수(Table A.1/A.3)와 수식(§C.1)이 있어도, 학습 절차의 **제어 흐름**(무엇이 언제 켜지고 꺼지는가)은 의사코드만이 전달할 수 있다. 특히 CSMAD는 Teacher-only warmup의 student forward skip, GRL의 λ 이중 구조(손실 가중 λ_GRL vs 반전 계수 λ_rev), batch 내 positive 부재 시 GRL 손실 skip 등 **타이밍·게이트가 본질인 설계**이므로, 이 절차의 부정확한 의사코드는 재현 실패의 직접 원인이 된다. 방어 관점에서는 P3 재리뷰가 두 번 지적한 off-by-one(epoch 표기 규약)과 단일-λ 합산 오서술의 재발 지점을 막는 것이 핵심이다 — 검증 체크리스트의 3·4항이 정확히 그 지점이다.

	#### 🏁 목표와 기대 결과

	성공 기준: 아래 5요소 체크리스트 전 항목 통과 + 캡션의 "(pseudocode placeholder)" 꼬리 제거(이 제거가 resolved 신호다). 의사코드의 어떤 줄도 코드에 없는 행동을 발명하지 않았고, 코드의 어떤 활성 동작도 의사코드가 누락하지 않았음이 확인되어야 한다. 검증 중 불일치가 발견되면 — 예컨대 epoch 경계의 ±1, λ 표기의 병합, 손실 항 누락 — 의사코드를 코드 쪽에 맞춰 수정한다(코드가 ground truth, 역방향 수정 금지). 검증 통과 후 의심 잔여 항목이 있으면 271_CONFIG_TRUTH의 file:line으로 재확인한다.

	#### 🧪 실험 내용과 설계

	실험이 아니라 검증 작업이다. 5요소 체크리스트를 계승한다 — 각 항목을 초안 tex의 해당 줄과 코드를 나란히 놓고 행 단위로 대조한다.

	| # | 검증 항목 | 정본·코드 근거 | 초안 상태와 남은 확인 |
	|---|---|---|---|
	| 1 | 전처리: SWaT constant 6컬럼 제거(45 = 51−6) + per-entity train-구간 min–max | 271_CONFIG_TRUTH r4 §VIII | 초안 반영됨 — 컬럼 목록 {P202, P401, P404, P502, P601, P603}이 §A.1 재현성 노트와 일치하는지 확인 |
	| 2 | anomaly-priority masking: priority 식 π_i = 10³·y_i + η_i, argtopk |M| | `model.py` masking 경로, Eq. C.5(`eq:masking_rule`) | 초안 반영됨 — **Eq. C.5와 기호가 글자 단위 일치**하는지 확인 (y^p_i 표기 포함) |
	| 3 | Teacher-only gating: **0-based epoch 0–249 동안 학습 경로 student forward 자체 skip** | r4 정본: "student 학습은 0-based epoch 250(= 251번째 epoch)부터"; `trainer.py:526–535` → `model.py:1119` | 초안의 `If e > 250`은 1-based 표기 — 0-based 250 개시와 ±1로 일치하는지 **epoch 표기 규약을 각주 또는 KwIn에 명시**할 것 (off-by-one이 P3 재리뷰 단골) |
	| 4 | 손실 조립: L_total = L_recon + L_OD + λ_FM·L_FM + λ_GRL·L_cls; **λ 이중 구조** 분리 표기 — 손실 가중 λ_GRL(grad-ratio clamp[0,10] × 0.2, 직전 epoch smoothing)과 반전 계수 λ_rev(sigmoid ramp 2/(1+e^{−10τ})−1, τ=clip((e−250)/250, 0, 1)) | `trainer.py:752–763`(λ_GRL), `trainer.py:1205-1207`(λ_rev); r4 NEW-B1 | 초안 반영됨 — **단일 λ로 합치지 말 것**. ⚠️ (OBS-1) τ식의 e는 3항과 **동일한 epoch 표기 규약**을 따라야 한다: 위 식은 1-based e 규약에서만 코드(0-based `(epoch−250+1)/250`)와 일치하므로, 3항의 규약 명시가 이 식에도 적용됨을 한 줄로 연동 표기할 것. GRL 손실의 batch 내 positive window 부재 시 skip(`loss.py:293-302`)은 초안 반영됨 |
	| 5 | 평가: 5 epoch 간격 test-split 평가 + best PA%K-AUC F1 추적 | config `eval_interval=5`, `best_epoch_metric` | 초안 반영됨 — `e mod 5` 표기 확인 |

	검증 방법: tex의 algorithm 블록 30줄을 위 표의 코드 위치와 1:1 대응시키는 대조 시트를 만들고, 대응 없는 줄(발명)·대응 누락(생략)을 0건으로 만든다.

	#### 📊 구성과 형태

	algorithm2e의 `algorithm*`(2단 폭 float, 페이지 상단 배치), `\footnotesize`, ~30줄 — 현 구조를 유지한다(PDF QA에서 단일 column 배치의 overprint 문제가 이미 해결된 형태). KwIn/KwOut, `\tcp` 주석 블록(Preprocessing / masking / Teacher / Student-gated / Loss assembly / Evaluation) 구조도 유지. epoch 표기 규약 명시는 KwIn 줄 또는 각주로 추가한다.

	#### 📝 캡션 (영문 확정본 — 확정 시 교체)

	```
	(현재)  CSMAD Training (pseudocode placeholder)
	(확정 시) CSMAD training procedure.
	```

	"(pseudocode placeholder)" 꼬리의 제거가 이 placeholder의 resolved 신호다.

	#### ⚠️ 주의사항과 의존성

	행동 발명 금지가 제1원칙이다 — 모든 줄은 `trainer.py`/`model.py`/`loss.py`의 실제 동작에 1:1 대응해야 하며, 의심 항목은 271_CONFIG_TRUTH의 file:line으로 재확인한다. 의사코드 내 수식 참조(Eq. C.1/C.4/C.5, `eq:ltotal` 등)는 **빌드 후 식 번호를 재확인**한다 — 부록 식 번호는 본문 구성 변경에 따라 밀릴 수 있다. AMP bf16·optimizer 세부는 의사코드 범위 밖이다(Table A.1에 위임) — 검증 중에 "코드에 있으니 추가하자"는 유혹을 따르지 말 것. warmup 구간의 student 상태를 서술할 일이 생기면 "frozen"이 아니라 "forward skipped (training path)"가 정확하다(평가 경로는 full forward — FIG-2 ⑤와 동일 규약).

	#### 🔢 연결된 수치 placeholder

	NUM 파생 없음. Eq. C.5(`eq:masking_rule`)·`eq:ltotal`·`eq:adaptive_weight`·`eq:lcls_app`와의 기호·번호 정합 의무만 진다.

### TXT-001 — GPU 모델명 (§A.1, 1개소) {toggle="true"}

	| 유형 | 소스 분류 | 우선순위 | 의존성 |
	|---|---|---|---|
	| 환경 정보 (§A.1) | **[신규 측정 — 확인만]** | 측정 즉시 가능 | TAB-B3 하드웨어 표기가 참조 |

	<callout icon="💡" color="blue_bg">
	**한 줄 요약** — §A.1 환경 문단의 "[GPU model]"을 실제 실험 수행 GPU 모델명으로 치환한다. metadata에 GPU 필드가 없으므로 실행 호스트 이력 확인이 유일한 경로다 — 추측 기재 금지.
	</callout>

	**위치** — `appendix_A.tex:80` — "All experiments run on [GPU model]" (`% PH:TXT-001`).

	#### 🎯 목적과 의도

	재현성·비용 서술의 기준 하드웨어를 확정한다. TAB-B3(연산 비용)의 측정 하드웨어 표기가 이 값을 참조하므로, 단순 빈칸이 아니라 부록 내 교차 참조의 앵커다.

	#### 🏁 목표와 기대 결과

	성공 기준: 271canon(및 baseline) 실험을 **실제 수행한** GPU 모델명이 확인 절차를 거쳐 기재되는 것. 학습 그룹과 baseline 그룹의 머신이 다르면 그룹별 병기가 올바른 결과다 — 단일 모델명을 강요하지 않는다.

	#### 🧪 실험 내용과 설계

	확인 절차는 다음과 같다. [271c] metadata에는 GPU 모델 필드가 **없다**(2026-06-11 실측 — `timing`/`config`에 부재, `device='cuda'`뿐). 따라서 첫째, 271canon 실행 호스트에서 `nvidia-smi --query-gpu=name`을 확인한다 — 현 machineA 실측은 NVIDIA GeForce RTX 4090이고 271canon이 이 머신에서 실행 중이므로 유력하나, **호스트 이력을 확인한 후에만 기재**한다(전체 실행이 한 머신이었는지 확인; 추측 금지 원칙). 둘째, baseline 실행 머신이 다르면 그룹별로 병기한다. 셋째, 향후 재실행분부터는 metadata에 GPU명 기록 필드를 추가할 것을 권장한다(이 확인 비용 자체를 제거).

	#### 📊 구성과 형태 / 📝 캡션 (영문 확정본)

	표·그림이 아닌 본문 1문장이다. 해당 문장 원문:

	```
	All experiments run on [GPU model];
	code, configurations, and the exact dataset partitions will be released at [URL].
	```

	#### ⚠️ 주의사항과 의존성

	TAB-B3의 측정 머신이 학습 머신과 다르면 그쪽 각주에서 구분 명시해야 하므로, TXT-001 확정이 TAB-B3 채움보다 선행되는 것이 자연스럽다.

	#### 🔢 연결된 수치 placeholder

	없음 — TAB-B3의 하드웨어 참조 문구만 이 값에 의존.

### TXT-002 — 코드 저장소 URL (3개소) {toggle="true"}

	| 유형 | 소스 분류 | 우선순위 | 의존성 |
	|---|---|---|---|
	| 공개 정보 (Abstract·§A.1·§5) | **[결정 사항]** — 실험·측정 무관 | 게재 시 | 공개 전 checklist |

	<callout icon="💡" color="blue_bg">
	**한 줄 요약** — Abstract·§A.1·§5의 "[URL]" 3개소를 공개 저장소 URL로 일괄 치환한다. 세 곳 글자 단위 동일이 의무이며, 제출 단계에서는 익명 요건을 먼저 확인한다.
	</callout>

	**위치** — `main.tex:110`(Abstract 말미) · `appendix_A.tex:81`(§A.1) · `sec5_conclusion.tex:31`(§5 말미).

	#### 🎯 목적과 의도

	코드 공개(R25)는 재현성 주장의 전제다. "release upon acceptance" 문구는 이미 확정되어 있으므로, 남은 것은 URL 실값과 그 치환 시점·익명성 처리뿐이다.

	#### 🏁 목표와 기대 결과

	성공 기준: 세 곳이 **글자 단위로 동일한** URL로 치환되는 것. 제출 시점과 게재 시점에 들어가는 값이 다를 수 있다(익명 미러 → 실제 저장소) — 단계별로 두 번의 일괄 치환이 모두 3개소 동시여야 한다.

	#### 🧪 실험 내용과 설계

	절차는 다음과 같다. 첫째, 제출 단계에서 저널의 익명 요건을 확인한다 — 정책에 따라 anonymous.4open.science 등 익명 미러가 필요할 수 있다. 둘째, 게재 확정 시 실제 URL로 일괄 치환한다. 셋째, 치환 시마다 grep으로 3개소를 동시 확인한다: `grep -n "\[URL\]"`를 tex 소스 전체(`main.tex` + 본문·부록 tex — 현 위치 기준 `paper/07_latex/*.tex`)에 걸어 잔존 0건을 확인한다. 공개 전 점검 checklist(RESEARCH_SYNTHESIS §⑦: 공개 branch 결정, 공개 범위 분리 — `configs/` 큐 JSON·`results/`·`temp/`·`paper*/` 비공개 처리, secret/credential 스캔, 재현 진입점 문서화 — SWaT 45-feature 재현성 플래그 포함)가 전부 미결 상태이므로, URL 확정 전에 이 checklist 해소가 선행되어야 한다.

	#### 📊 구성과 형태 / 📝 캡션

	본문 문장 3개소. Abstract·§5는 "will be released at [URL]" 패턴, §A.1은 위 TXT-001 인용 문장과 공유.

	#### ⚠️ 주의사항과 의존성

	세 곳 동일 의무가 유일한 규칙이지만 가장 어기기 쉬운 규칙이다 — 한 곳만 고치고 커밋하는 사고를 grep 의식(ritual)으로 차단한다. 익명 요건 위반(실명 저장소 URL을 제출본에 노출)은 desk reject 사유가 될 수 있으므로 제출 직전 최종 확인 항목에 포함한다.

	#### 🔢 연결된 수치 placeholder

	없음 — TXT-002 ×3개소 자체가 등록 단위 (REGISTRY TXT 2종/4개소 중 3개소).

---

## 5. 권고 실험 (rebuttal 대비)

### R-PROBE — GRL 억제의 기계적 증거: Probing Classifier (원고 비반영) {toggle="true"}

	| 유형 | 소스 분류 | 우선순위 | 의존성 |
	|---|---|---|---|
	| 권고 실험 (원고 placeholder 없음) | **[신규 측정]** — 본 모델 학습 불필요, probe만 학습 | 측정 즉시 가능 | 기본형 [271c]만 / 확장형 #4 |

	<callout icon="💡" color="purple_bg">
	**한 줄 요약** — rebuttal 대비 전용 권고 실험. frozen checkpoint 위에 소형 probe를 학습시켜 "GRL이 Student 표현에서 anomaly-identity 정보를 실제로 지웠는가"를 AUC로 직접 측정한다. **원고는 한 글자도 바뀌지 않는다** — D-014 (b) 등재 의무 이행 항목.
	</callout>

	**위치·근거** — 원고 placeholder 없음 — 본 '권고 실험' 절로만 발행. 근거: D-014 (b) / p8 리뷰 F-1 BLOCKER 해소 항목 (spec r2 §6R).

	#### 🎯 목적과 의도

	**목적과 의도가 이 항목의 전부다.** 논문의 핵심 주장 중 하나는 "GRL이 Student decoder에서 anomaly-identity 정보를 능동적으로 억제하여, anomaly 구간에서 Teacher–Student discrepancy가 증폭된다"는 메커니즘 서술이다. 본문의 근거는 성능 ablation(TAB-3 행2 — GRL을 빼면 성능이 떨어진다)인데, 성능 하락은 메커니즘의 **간접 증거**일 뿐이다 — 까다로운 리뷰어는 "성능이 떨어진 것이 정말 '표현에서 정보가 억제'되었기 때문인가, 다른 부수 효과 때문인가"를 물을 수 있다(§15의 "GRL이 student를 망가뜨리지 않는가" 시나리오의 공격적 변형). 이 실험은 그 질문에 대한 **직접적·기계적 증거**를 준비한다: 표현 공간에서 anomaly 정보의 선형 추출 가능성(probe AUC)을 Teacher와 Student 사이에서 비교함으로써, "Student 표현에서는 anomaly가 읽히지 않는다"를 분류 성능이라는 해석 불요의 숫자로 보인다. 원고에는 반영하지 않는다 — rebuttal 단계에서 요구받을 때 즉시 제시할 수 있는 탄약고 역할이며, 따라서 결과가 어떻든 원고 리스크가 없다.

	#### 🏁 목표와 기대 결과

	기대 패턴: **Student probe AUC ≪ Teacher probe AUC** — Teacher의 동일 위치 hidden에서는 anomaly window가 상당히 분류되는 반면(억제가 없으므로 정보 잔존), GRL이 작용한 Student hidden에서는 분류가 어려워야(AUC가 chance 수준에 근접) 억제 성공의 정량 증거가 된다. 확장 대조군까지 수행하면 기대 구도는 "w/o GRL Student probe AUC > with-GRL Student probe AUC" — GRL이 없으면 Student에 anomaly 정보가 잔존함을 보이는 대조다. 다른 패턴의 해석: 두 probe AUC의 차이가 작으면 GRL의 억제가 표현 수준에서 약하다는 뜻이므로, 이 결과는 rebuttal에서 사용하지 않고(원고 무변경이므로 손실 없음) 메커니즘 서술의 표현 강도를 내부적으로 점검하는 입력으로만 쓴다. 성공 기준은 발행이 아니라 **준비 완료**다: 절차·수치가 정리된 내부 노트 1편.

	#### 🧪 실험 내용과 설계

	본 모델의 학습은 일절 없다 — 표현은 전부 frozen이고 probe만 학습한다.

	첫째, [271c] 대표 entity(권장: TAB-3 대표 데이터셋과 동일 — 서사 일관성)의 best checkpoint를 동결 로드한다. 둘째, test 윈도들에 대해 두 표현을 추출한다: ① Student decoder의 **final-layer hidden, output projection 직전** — GRL 부착 지점과 정확히 동일한 위치다(FIG-2 ③ⓒ의 명시 라벨과 같은 지점; 다른 층에서 뽑으면 실험의 의미가 사라진다), ② Teacher의 동일 위치 hidden. 셋째, 각 표현 위에 소형 probe — LayerNorm + Linear 1층, **GRL head와 유사한 용량**(용량을 맞춰야 "추출기 능력 차이"가 아니라 "표현 내 정보량 차이"를 측정한다) — 를 anomaly window 이진 분류로 학습한다(표현 frozen, probe 파라미터만 학습). 넷째, 두 probe의 test AUC를 비교한다.

	**확장(선택)**: TAB-3 행2(w/o GRL, OD-exclusion 유지) run이 완료되면 그 checkpoint에 동일 probing을 적용해 GRL 유/무 Student probe AUC 차이를 병기한다. 주의 — 기존 exp290은 no_fm+no_grl **복합** 변형이므로 대조군으로 쓸 경우 그 사실을 각주로 반드시 밝힌다(단독 효과로 오인 금지).

	#### 📊 구성과 형태

	산출물은 원고 figure/table이 아니라 rebuttal 대비 내부 노트다. 권장 형태: probe AUC 비교 표 1개 {표현 출처(Teacher / Student / Student w/o GRL)} × {entity} × AUC, 그리고 절차 요약 문단. rebuttal에 첨부할 수 있도록 self-contained로 작성한다.

	#### 📝 캡션

	해당 없음 — 원고 무변경 항목이므로 확정 캡션이 존재하지 않는다. rebuttal 첨부 시의 표 제목은 작성 시점에 자유 작성한다.

	#### ⚠️ 주의사항과 의존성

	원고의 어떤 placeholder와도 연결되지 않으며 REGISTRY 커버리지 산식에도 불포함이다(커버리지 불변) — 이 항목 때문에 원고를 고치는 일이 없도록 한다. 추출 지점의 정확성이 실험의 성패다: "output projection 직전"이 아닌 위치에서 뽑은 hidden은 GRL 부착 지점과 다르므로 증거 능력이 없다. probe 용량을 GRL head보다 크게 잡으면 "강한 추출기는 억제된 표현에서도 정보를 캐낸다"는 반론에 노출된다 — 유사 용량 원칙을 지킬 것. 의존성: 기본 실험은 [271c] 완주분만으로 즉시 가능; 확장 대조군만 TAB-3 행2 run(신규 실행 #4)에 의존한다.

	#### 🔢 연결된 수치 placeholder

	없음 — 원고 placeholder와 무관 (권고 실험, D-014 (b)).

---

## 6. 인라인 수치 placeholder 종합 (NUM 31건)

<callout icon="🔢" color="gray_bg">
본문에 흩어진 인라인 수치 31건을 **파생 소스 단위 8그룹**으로 묶었다. 각 그룹은 소스 실험이 완료되면 그룹 내 전 항목이 **동시에** 풀린다. 수치 창작 금지(A8) — 아래 표는 "들어갈 값의 정의"만 기술하며, 실제 값은 소스 확정 후에만 기입한다. 상세 설계는 각 placeholder 토글의 🔢 절을 본다.
</callout>

**그룹 N-A — 데이터셋 family 수 (sync 그룹 A) · [완주 대기]** — 소스: 271canon 완주 + TAB-2 완성:

| ID | 본문 위치 | 들어갈 값의 정의 |
|---|---|---|
| **NUM-001** | Abstract 6문장 (`main.tex`) | family 수 — 6 family 전부 완주 시 "six" |
| **NUM-003** | Highlights bullet 5 (`main.tex` highlights + `highlights.txt`) | 동일 값 (sync) |
| **NUM-004** | §1 contribution bullet 4 (`sec1_intro.tex`) | 동일 값 (sync) |
| **NUM-029** | §5 결론 (`sec5_conclusion.tex`) | 동일 값 (sync) |

**그룹 N-B — baseline 총수 (sync 그룹 B) · [신규 실행(weak 4종) 의존]** — 소스: weak 4종 GPU 완주:

| ID | 본문 위치 | 들어갈 값의 정의 |
|---|---|---|
| **NUM-002** | Abstract (`main.tex`) | "26" (22 unsup + 4 weak); 미완 시 "22 unsupervised" fallback |
| **NUM-005** | §1 contribution bullet 4 | 동일 값 (sync) |
| **NUM-030** | §5 결론 | 동일 값 (sync) |

**그룹 N-C — Table 2 본 블록 파생 · [집계만 — TAB-2 완성 후]** — 소스: TAB-2 완성본 집계:

| ID | 본문 위치 (§4.2) | 들어갈 값의 정의 (집계 규칙) |
|---|---|---|
| **NUM-006** | ¶1, [N]×2 | 6 family 중 CSMAD 1위 family 수 (F1 기준 1 + VUS-PR 기준 1). WaDi 집계 규칙 명시(권고: A1·A2 모두 1위일 때만 win) |
| **NUM-007** | ¶1, [X.XX]×2 | CSMAD family 평균 (F1, VUS-PR) — WaDi는 A1/A2 평균을 family 값으로 |
| **NUM-008** | ¶1 | (CSMAD 평균) − (family별 최강 unsup 평균), F1 |
| **NUM-009** | ¶1 | 동일, VUS-PR |
| **NUM-010** | ¶2 | CSMAD F1 @ PSM ([271c]; 표 확정 전 선기입 금지) |
| **NUM-011** | ¶2 | best unsup F1 @ PSM ([CMP-Q3]) |
| **NUM-012** | ¶2 | CSMAD F1 @ SWaT excl22 ([271c] `SWaT/A1A2_excl22`) |
| **NUM-013** | ¶3, [X.XX]×2 | NRdetector(contaminated) 대비 비교값 — 문장·정의 중 한쪽으로 확정 (침묵 불일치 금지) |

**그룹 N-D — Protocol-effect 블록 파생 · [신규 실행 — standard-split run]** — 전부 PA%K-AUC F1:

| ID | 본문 위치 (§4.2 protocol-effect) | 들어갈 값의 정의 |
|---|---|---|
| **NUM-014** | 블록 캡션 + 본문 [N] (동시) | 블록 내 대표 baseline 수 (2–3; tex stub은 A/B 2행) |
| **NUM-015** | "CSMAD remains competitive ([X.XX])" | CSMAD clean-train 평균 |
| **NUM-016** | "... versus [X.XX] for the best unsupervised competitor" | best unsup clean-train 평균 |
| **NUM-017** | "CSMAD improves to [X.XX]" | CSMAD contaminated 평균 (Table 2 본 블록 부분집합 재집계; 신규 실행 아님) |
| **NUM-018** | "(a gain of [X.XX] points)" | 파생 계산값: 017 − 015 (별도 측정 금지) |
| **NUM-019** | "the unsupervised baselines show [X.XX] change" | best unsup의 standard→contaminated 변화량. 비교쌍은 standard vs **contaminated-training(무절제)** run. contaminated 쪽은 TAB-B1과 소스 공유 가능 |

**그룹 N-E — ablation 파생 · 혼합** — TAB-3 소스(020–023) + TAB-B4 소스(024,025):

| ID | 본문 위치 | 들어갈 값의 정의 | 소스 |
|---|---|---|---|
| **NUM-020** | §4.3 lead + 캡션 "[3--4 datasets]" | ablation 대표 데이터셋 수 (3–4; TAB-B4와 동일 집합) | TAB-3 결정 |
| **NUM-021** | §4.3 Row 3 문단 | 행 1 − 행 3 Avg 차 (양수 하락폭) | [271c] − exp287 **[재사용]** |
| **NUM-022** | §4.3 Row 4 문단 | 행 1 − 행 4 Avg 차 (양수 하락폭) | 행 4 **[신규 실행]** |
| **NUM-023** | §4.3 Row 2 문단 | 행 1 − 행 2 Avg 차 (GRL 순효과, 양수) | 행 2 **[신규 실행]** |
| **NUM-024** | §B.5 "Symmetric decoder capacity" | Full − Symmetric Avg 차 (**bullet 3 load-bearing**, 양수) | symmetric run **[신규 실행]** |
| **NUM-025** | §B.5 "FM loss regularizer" | Full − w/o FM Avg 차 (양수) | exp285_no_fm **[재사용]** |

**그룹 N-F — sparsity 파생 · [신규 실행]** — 소스: FIG-3 sweep:

| ID | 본문 위치 | 들어갈 값의 정의 |
|---|---|---|
| **NUM-026** | §4.4 Results lead + 캡션 [N] (동시) | FIG-3 대표 데이터셋 수 (2 또는 3; 권장 SWaT excl22 + PSM = 2) |
| **NUM-027** | §4.4 Results 서술어 [gradually/monotonically] | 열화 형상 서술어 — 곡선 확정 후 선택. 비단조면 문장 재작성 (A8) |

**그룹 N-G — qualitative 파생 · [재사용]** — 소스: FIG-4 제작:

| ID | 본문 위치 | 들어갈 값의 정의 |
|---|---|---|
| **NUM-028** | §4.5 lead의 [N] | FIG-4 데이터셋 수 = **2** (SWaT excl22 + {WaDi A1 또는 PSM}) |

**그룹 N-H — cost 파생 · [신규 측정]** — 소스: TAB-B3 측정:

| ID | 본문 위치 | 들어갈 값의 정의 |
|---|---|---|
| **NUM-031** | §B.3 본문 ("the measured wall-clock overhead factor is [X.XX]") | leave-one-out vs single-mask wall-clock 배율 실측값. sync: 50보다 유의미하게 낮으면 §5 "approximately 50×" → "up to 50×" 같은 pass 완화 |

<callout icon="🔤" color="gray_bg">
**TXT placeholder 2종 / 4개소** (수치는 아니나 동일 종합 차원에서 기록) — **TXT-001**(§A.1 GPU 모델명 ×1) · **TXT-002**(Abstract·§A.1·§5 저장소 URL ×3, 세 곳 글자 단위 동일 의무). 상세는 §4 토글 참조.
</callout>

---

## §검증 — 무손실 자가 점검 (작성 후 python 실측)

<callout icon="✅" color="green_bg">
아래 결과는 발행 산출물 파일 자체를 python으로 스캔해 placeholder ID 전수 존재·영문 캡션 코드블록·토글 들여쓰기를 검증한 것이다. (자가 점검 절차와 통과 여부는 최종 보고에 함께 기록한다.)
</callout>

| 검증 항목 | 기대 | 점검 방법 |
|---|---|---|
| FIG ID 5종 | FIG-1·2·3·4·B1 | 각 `### FIG-x` 토글 헤딩 존재 |
| 표 ID 12종 | TAB-1·2·3 / A.4·A6·A7·A8 / B1·B2·B3·B4 (+TAB-A3) | 각 토글 헤딩 존재 |
| ALG | ALG-C1 1건 | §4 토글 존재 |
| R-PROBE | 1건 | §5 토글 존재 |
| NUM | NUM-001~031 = 31건 | §6 그룹 표 + 각 placeholder 🔢 절 |
| TXT | TXT-001/002 = 2종(4개소) | §4 토글 2개 |
| 영문 캡션 | 각 figure/table에 latex 코드블록 또는 plain 코드블록 | figure/table 토글당 1개 이상 |
| 토글 children 들여쓰기 | 탭 들여쓰기 적용 | `{toggle="true"}` 직하 children 전부 탭 시작 |

---

본 페이지는 두 확장판 초안(B1 본문 + B2 부록)을 단일 Notion 페이지로 통합한 발행본이다. 사실·수치·실행 지침·영문 캡션은 원문 그대로 보존했고, 한국어 표현과 페이지 구성만 정제했다. 실험 착수 시에는 §0 대시보드의 우선순위를 따르고, 각 placeholder의 실행 지침·주의사항을 해당 토글에서 확인한다.
