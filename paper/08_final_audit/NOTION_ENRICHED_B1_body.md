---
phase: 8
agent: spec-enricher-B1
directives: [R3]
last_modified: 2026-06-11
scope: |
  본문 Figure/Table 전부 — FIG-1, FIG-2, FIG-3, FIG-4, TAB-1, TAB-2(+TAB-4 흡수 블록), TAB-3
  + 파생 NUM 그룹: N-A(001/003/004/029), N-B(002/005/030), N-C(006–013), N-D(014–019),
  N-E 중 TAB-3분(020–023), N-F(026/027), N-G(028) = NUM 28건.
  (N-E 중 024/025는 TAB-B4 소스 → appendix 담당 B2 / N-H(031)는 TAB-B3 소스 → B2)
basis: |
  사실 기반: paper/08_final_audit/NOTION_PLACEHOLDER_SPECS.md (r2 — 실행 지침·재사용 판정·캡션·의존성 전부 계승)
  목적·논증: paper/03_blueprint/PAPER_BLUEPRINT.md §6/§12/§14/§15 + paper/01_research_understanding/RESEARCH_SYNTHESIS.md
  본문 맥락: paper/07_latex/sec1_intro.tex, sec3_method.tex, sec4_experiments.tex (캡션·배치 확정본)
notes: |
  Notion 발행용 — placeholder 1건 = 페이지 1장. 페이지 경계는 <!-- PAGE: {ID} --> 주석.
  수치 창작 금지(A8): 성능 기대는 방향·패턴으로만 서술. 본문에 등장하는 수치는 전부 r2 명세의 실측·확정값 인용.
---

# NOTION ENRICHED SPECS — B1: 본문 Figure·Table (목적·의도·설계·기대 결과 확장판)

이 문서는 r2 명세(NOTION_PLACEHOLDER_SPECS.md)의 본문 figure/table 항목을 "Notion 페이지만 보고 실험 설계와 figure 제작이 가능한 수준"으로 확장한 것이다. r2의 실행 지침·재사용 판정·캡션 원문·의존성은 전부 보존하고, 그 위에 **목적과 의도**(논증에서의 역할, 방어하는 reviewer 공격)와 **목표·기대 결과**(성공 기준 + 기대와 다를 때의 대응) 차원을 추가했다.

---

<!-- PAGE: FIG-1 -->

# FIG-1 — 학습 패러다임 비교 다이어그램 (Setting-comparison diagram)

> 💡 **한 줄 요약**: 오염된 학습 스트림이라는 동일한 입력 위에서 unsupervised / label-aware filtering / CSMAD 세 패러다임이 라벨을 각각 "무시 / 절제 / 통합"하는 방식을 한 장으로 대비시켜, 논문 전체의 문제 설정과 핵심 용어를 시각적으로 고정하는 개념도.

| 항목 | 내용 |
|---|---|
| 위치 | §1 Introduction, observation 문단 직후 (`sec1_intro.tex`, `\label{fig:setting}`) |
| 크기 | full-width, 약 5 cm (≈0.40p) |
| 소스 분류 | `[제작]` — 실험 데이터 없음, 다이어그램 제작·검증만 필요 |

## 🎯 목적과 의도

이 그림은 논문의 중심 논제 — "labeled anomaly는 비지도 방법에게는 오염이지만, 그것을 학습 신호로 통합할 수 있는 방법에게는 가치 있는 정보다" — 를 본문 텍스트보다 먼저, 그리고 텍스트 없이도 전달하기 위해 존재한다. §1의 관찰 문단(labeled anomaly가 드러내는 세 가지 학습 신호 (a)/(b)/(c))이 끝난 직후에 배치되어, 독자가 contribution bullet을 읽기 전에 "기존 패러다임 두 가지가 이 신호를 어떻게 버리는가"를 눈으로 확인하게 만든다.

논증 구조에서 이 그림이 맡는 역할은 두 가지다. 첫째, 중앙 패널(label-aware filtering)은 "라벨이 있으면 그냥 오염 구간을 걸러내면 되지 않는가"라는 가장 자연스러운 reviewer 반문에 대한 시각적 선제 답변이다. 본문 §1의 문장("the best a label-aware variant can do is exclude confirmed anomaly windows ... filtering contamination rather than learning from it")이 말하는 한계 — 오염은 제거되지만 라벨 정보 자체는 폐기된다 — 를 패널 하나로 보여주며, 이는 곧 본문 비교 실험의 anomaly-excised condition(Table 2의 main 조건)이 왜 "비지도 방법에게 라벨의 최선 활용을 제공하는 조건"인지(블루프린트 §14 논거 ③, R12 논리)와 직결된다.

둘째, 세 패널 상단의 입력 스트림 띠를 **세 패널에서 동일하게** 그리는 것 자체가 방어 장치다. 블루프린트 §15의 leakage 공격 시나리오("test-prefix 편입은 test label로 학습하는 것")에 대한 방어 논거 ③(모든 비교 모델이 동일한 데이터를 받는다)을, 비교 조건을 설명하기도 전에 그림의 전제로 깔아둔다. 또한 우측 패널의 세 갈래 화살표는 contribution bullet 2의 세 용어(*anomaly-priority masking*, *loss bifurcation*, *gradient-reversal suppression*)를 글자 단위로 고정하는 anchor로, 이후 §3·§4의 모든 서술이 이 세 명칭으로 수렴한다.

## 🏁 목표와 기대 결과

실험이 없는 제작물이므로 "성공 기준"은 전달력과 정합성으로 정의한다. (1) 비전문 독자가 캡션을 읽지 않고도 세 패널의 차이 — 라벨이 보이지 않음 / 라벨로 구간을 잘라냄 / 라벨이 세 경로로 학습에 흘러 들어감 — 를 읽을 수 있을 것. (2) 그림 내 모든 용어가 본문 표기와 글자 단위로 일치할 것(아래 주의사항). (3) 입력 스트림 띠의 붉은(anomaly) 비율이 실제 train anomaly ratio(0.5–6.2%)를 연상시키는 소수 구간일 것 — 절반이 붉은 그림은 설정 자체를 왜곡한다.

기대와 다른 패턴이라는 개념은 개념도에는 적용되지 않으나, 대응 규칙은 있다: 만약 Phase 8 채움 과정에서 비교 조건 명칭이나 contribution bullet 2의 용어가 변경되면(예: R24류 개명 재발생), 이 그림은 본문과 동시에 갱신해야 하며 그림만 구표기로 남는 것은 허용되지 않는다.

## 🧪 실험 내용과 설계

**`[제작]` — 실험 소스 없음.** 학습·측정이 전혀 필요 없고, 다이어그램 제작과 본문 대조 검증만 수행한다.

- **권장 제작 경로**: TikZ 직접 작성(elsarticle 빌드와 폰트가 일치해 가장 안전) 또는 외부 벡터 도구로 제작 후 PDF 삽입.
- **공통 입력 띠**: 세 패널 상단에 동일한 입력 스트림 띠(정상 구간 + 붉은 라벨 anomaly 구간)를 그린다. 붉은 구간은 소수(시각적으로 수 % 수준)만 칠한다.
- **패널 구성**: (좌) unsupervised — 라벨이 모델에 보이지 않아 순수 오염원으로 작용. (중) label-aware filtering — 라벨된 anomaly window를 학습 전에 절제(= anomaly-excised condition; §4.1.4 상호참조). (우) CSMAD — 라벨이 masking·loss·gradient 세 경로로 학습에 유입.

## 📊 구성과 형태

가로 3-패널. 각 패널의 수직 구성은 동일하게: **상단 입력 스트림 띠 → 모델 박스 → 라벨 흐름 글리프**.

| 패널 | 라벨 흐름 글리프 | 핵심 시각 메시지 |
|---|---|---|
| 좌 (unsupervised) | 라벨이 모델에 닿지 않음 (무시됨 표시) | anomaly가 all-normal 가정의 오염원으로만 작용 |
| 중 (label-aware filtering) | 라벨이 데이터를 잘라내는 가위/절제 표시 | 오염 제거 = 라벨 정보 폐기 |
| 우 (CSMAD) | 라벨에서 세 갈래 화살표 → masking / loss / gradient | 오염을 학습 신호로 전환 |

용어는 §1 contribution bullet 2의 표기와 글자 단위로 일치: *anomaly-priority masking*, *loss bifurcation*, *gradient-reversal suppression*.

## 📝 캡션 (영문 확정본)

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

## ⚠️ 주의사항과 의존성

- 중앙 패널 명칭은 R24 개명 후의 **"anomaly-excised condition"만** 사용한다. 구표기(Q3, normalonly)는 그림·라벨 어디에도 금지.
- 붉은 구간 비율은 실제 train AR(0.5–6.2%)을 연상시키는 소수 구간만 — 설정 왜곡 금지.
- 세 패널의 입력 스트림 띠는 픽셀 단위로 동일해야 한다(공정성 논거의 시각화).
- 그림 용어 ↔ 본문 bullet 2 ↔ §3 소절 제목의 3중 일치 검증을 제작 완료 시점에 1회 수행.

## 🔢 연결된 수치 placeholder

없음 — 이 그림에서 파생되는 NUM placeholder는 없다. 단, 용어 동기화 의무(contribution bullet 2의 3-path 명칭)는 위 주의사항대로 적용된다.

---

<!-- PAGE: FIG-2 -->

# FIG-2 — CSMAD 아키텍처 개요 (Architecture overview)

> 💡 **한 줄 요약**: 학습(좌)·추론(우) 2-패널로 CSMAD의 다섯 기능 블록과 라벨 유입 경로, gradient 차단 구조를 한 장에 담아 §3 전체의 지도 역할을 하는 아키텍처 다이어그램.

| 항목 | 내용 |
|---|---|
| 위치 | §3.2 도입부 (`sec3_method.tex`, `\label{fig:architecture}`) |
| 크기 | full-width, 5 cm = 0.40p (integrator 가정; Phase 7에서 가독성 확인) |
| 소스 분류 | `[제작]` — 구조 상수는 271_CONFIG_TRUTH r4 §VIII에서만 인용 |

## 🎯 목적과 의도

§3의 다섯 소절(문제 정식화 → 마스킹 → 비대칭 디코더 → 라벨 유도 학습 → 채점)은 각각 한 component를 다루기 때문에, 독자가 전체 데이터 흐름을 머리에 그리지 못하면 각 손실 항이 어디에 붙는지 길을 잃는다. 이 그림은 §3.2 도입부에서 그 전체 지도를 제공한다: 입력 윈도가 패치로 갈라져 어느 블록을 거치고, 네 가지 손실(L_recon, L_OD, L_FM, L_cls)이 어느 연결선에서 발생하며, 추론 시에는 무엇이 꺼지는지를 좌우 패널 대비로 보여준다.

이 그림은 동시에 두 개의 확정된 reviewer 공격에 대한 방어 장치다. 첫째, **"GRL이 Student를(나아가 표현 전체를) 망가뜨리지 않는가"**(블루프린트 §15) — Student의 latent 입력에 stop-gradient 기호(⊥)를 명시해, encoder가 Teacher의 재구성 목적만으로 학습되고 GRL gradient로부터 완전히 차단된다는 §3.2 본문 주장("The adversarial signal therefore cannot corrupt the normal-pattern representation")을 시각적으로 고정한다. 둘째, **GRL 부착 지점의 모호성**(블루프린트 ADV BLK-002 — 과거 리뷰에서 실제로 지적된 재발 지점) — "Student decoder final-layer hidden states, **before output projection**"이라는 명시 라벨을 그림 안에 넣어, 부착 지점을 본문·부록·rebuttal이 공유하는 단일 사실로 만든다. 부수적으로 GRL 박스의 점선 + "training only" 표기는 "추론 시 라벨 미사용"이라는 문제 설정의 약속(§3.1)을 그림 차원에서 반복한다.

마지막으로 우측 추론 패널은 §3.6의 leave-one-out 채점(50패턴 batch-병렬, σ_i → a_t 평균 집계)을 묘사함으로써, §5 결론의 비용 한계 서술(약 50× forward 연산)과 부록 비용 표(TAB-B3)가 가리키는 대상을 미리 정의한다.

## 🏁 목표와 기대 결과

성공 기준: (1) 그림만 보고 §3의 기호(o^T_i, o^S_i, h^enc, σ_i, a_t)와 손실 연결이 본문 수식과 1:1로 대응될 것. (2) 필수 표기 3건(아래 형태 절의 ⓐⓑⓒ)이 전부 들어 있을 것 — 특히 ⓒ(GRL 부착 지점 라벨)는 생략 시 리뷰 재발 지점이다. (3) 모든 구조 상수(d_model=512, nhead=8, encoder 4L / Teacher 3L / Student 2L, N=50, ρ=0.15 → |M|=8, L=500, patch size 10)가 271_CONFIG_TRUTH r4 §VIII과 일치할 것.

기대와 다른 상황에 대한 대응: 이 그림은 실험 결과를 담지 않으므로 결과 의존이 없다. 다만 부록 ablation(TAB-B4)의 symmetric decoder run 결과에 따라 contribution bullet 3의 주장 강도가 하향될 수 있는데(Phase 6 규칙), 그 경우에도 이 그림의 구조 자체(3L/2L 비대칭)는 사실 서술이므로 변경 불필요 — 캡션·본문 문구만 조정 대상이다.

## 🧪 실험 내용과 설계

**`[제작]` — 실험 소스 없음.** 모든 구조 상수는 271_CONFIG_TRUTH r4 §VIII에서 그대로 가져온다. 이 정본 외 출처(코드 default, Notion 스냅샷, 발표자료)에서 수치를 가져오는 것은 금지 — 과거 batch_size(512 vs 1024), d_model(dynamic vs 512 고정) 불일치 사고가 전부 비정본 인용에서 발생했다.

좌패널(학습)에 담을 데이터 흐름: 윈도(L=500) → N=50 패치 → anomaly-priority masking이 |M|=8 패치를 가림(anomaly 패치 우선) → visible 42패치만 encoder 통과 → 디코더 앞에서 mask token 삽입(Teacher/Student 별도 토큰) → 손실 연결선 4종: L_recon(Teacher 출력), L_OD·L_FM(Teacher↔Student, 정상 masked 패치만), L_cls(GRL classifier → window label).

우패널(추론): GRL branch 비활성, leave-one-out masking 50패턴을 batch 차원으로 병렬 처리, per-patch score σ_i → point score a_t 평균 집계.

## 📊 구성과 형태

다섯 색상 블록: (1) Patch Embedding(linear), (2) 공유 Transformer Encoder(4L), (3) Teacher Decoder(3L — 진한 색·깊게), (4) Student Decoder(2L — 연한 색·얕게), (5) GRL + AnomalyClassifierHead.

**필수 표기 3건 (생략 금지)**:

| # | 표기 | 이유 |
|---|---|---|
| ⓐ | GRL 박스 = 점선 + "**training only**" 라벨 | 추론 시 라벨·GRL 미사용 약속의 시각화 |
| ⓑ | Student latent 입력에 stop-gradient 기호 ⊥ | encoder가 Teacher recon으로만 학습됨 — §15 GRL 방어 |
| ⓒ | GRL 부착 지점 명시 라벨: "Student decoder final-layer hidden states, **before output projection**" | ADV BLK-002 — 생략 시 리뷰 재발 지점 |

## 📝 캡션 (영문 확정본)

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

## ⚠️ 주의사항과 의존성

- 기호는 Table C.2 notation 정본을 따른다 — point score는 s_t가 아니라 **a_t** (v2-r3 정정 반영됨).
- 학습 좌패널에 warmup(0-based epoch 0–249 동안 Student 학습 경로 forward skip)을 그릴 의무는 없으나, 그릴 경우 "frozen"이 아니라 "**forward skipped (training path)**"가 정확한 서술이다 (271_CONFIG_TRUTH r4 §VIII Training — 평가 경로는 full forward라는 구분 포함).
- λ를 그림에 표기할 경우 이중 구조(손실 가중 λ_GRL vs 반전 계수 λ_rev)를 단일 λ로 합쳐 쓰지 말 것 — 표기가 번잡하면 그림에서는 생략하고 §3.4 본문에 위임하는 편이 안전하다.

## 🔢 연결된 수치 placeholder

없음 — 이 그림에서 파생되는 NUM placeholder는 없다. 구조 상수의 단일 원천은 271_CONFIG_TRUTH r4 §VIII이며, §4.1.2 Implementation Details의 동일 상수 서술과 일치해야 한다.

---

<!-- PAGE: FIG-3 -->

# FIG-3 — 라벨 희소화 sweep (Label sparsity sweep) ★ 미구현 실험 (R32)

> 💡 **한 줄 요약**: 학습 시 라벨이 제공되는 anomaly region 비율 p를 1.0에서 0.1까지 낮추며 CSMAD의 성능 곡선을 그려, "라벨이 희소해져도 점진적으로만 열화하고 unsupervised floor 아래로 떨어지지 않는다"는 abstract·결론의 핵심 주장을 정량적으로 뒷받침하는 그림.

| 항목 | 내용 |
|---|---|
| 위치 | §4.4 Results 문단 직후 (`sec4_experiments.tex`, `\label{fig:sparsity}`) |
| 크기 | ~4 cm ≈ 0.33p |
| 소스 분류 | `[신규 실행]` — 전용 파라미터·스크립트 현재 부재 (`label_ratio`/`sparsity` grep 0건 실측) |

## 🎯 목적과 의도

main 실험(Table 2)은 train 구간의 모든 anomaly에 라벨이 있는 **라벨 가용성 상한 케이스**다. 그런데 논문의 문제 설정(§3.1, R11)은 "대부분 unlabeled + 소수 labeled"라는 일반 케이스를 가정한다 — 실제 운영 로그는 발생한 fault의 일부만 기록하기 때문이다. 이 간극을 메우지 않으면 "main 결과는 모든 라벨이 주어진 비현실적 조건의 산물"이라는 공격에 노출되고, 더 나아가 §15의 "PU learning이 아닌데 PU라 부른다" 류의 설정 공격에도 취약해진다(블루프린트의 3단 구조 — 설정/상한 구현/일반 케이스 검증 — 에서 이 그림이 세 번째 기둥이다). FIG-3은 라벨 비율 p를 내리며 상한 케이스에서 일반 케이스로, 그리고 비지도 극한으로 연속적으로 이동하는 곡선을 보여줌으로써 설정의 일반성을 직접 검증한다.

이 그림이 뒷받침하는 본문 주장은 명확히 두 개다. 첫째, abstract와 §5 결론의 "detection performance degrades gracefully ... remaining above the unsupervised floor" — 이 문장의 **유일한 정량 근거**가 이 그림이다. 둘째, §4.4의 "Why graceful degradation is expected" 문단이 제시하는 구조적 논거 3가지(anomaly-priority masking은 labeled 패치에만 작용 / GRL은 batch에 labeled positive가 없으면 손실 자체가 미계산 / 재구성 오차는 라벨-무관 신호)가 실제 곡선과 일치하는지의 검증대다. 점선 floor(각 데이터셋의 best unsupervised baseline)는 "라벨이 거의 없어도 비지도 방법보다 나쁘지 않다"를 시각화해, CSMAD를 도입할 때의 하방 위험이 없음을 보인다.

블루프린트 §6.8이 명시하듯 이 sweep은 NRdetector의 label-noise sweep과 축의 의미가 다르다(라벨 희소율 vs 잘못된 세그먼트 라벨 비율) — 본문에 1문장 구분이 이미 들어가 있으므로, 그림 설계가 이 구분을 흐리면 안 된다(라벨을 지우는 것이지 데이터를 지우거나 라벨을 틀리게 만드는 것이 아니다).

## 🏁 목표와 기대 결과

**입증하려는 것**: p가 감소할 때 성능이 (i) 연속적·점진적으로 감소하고(절벽형 붕괴 없음), (ii) p→0 부근에서 해당 데이터셋의 unsupervised floor에 접근하되 그 아래로 유의미하게 떨어지지 않으며, (iii) p=1.0 점이 main 설정([271c]의 해당 entity 값)과 정확히 일치한다는 것. (iii)은 기대가 아니라 **검산 조건**이다 — 불일치하면 sweep 파이프라인에 결함이 있는 것이다.

**기대와 다른 패턴이 나오면**: 곡선이 비단조이거나 특정 p에서 급락하면, 우선 NUM-027의 서술어 후보(gradually/monotonically)를 둘 다 버리고 §4.4 Results 문장을 실제 형상에 맞게 재작성한다(A8 — 곡선 확정 전 서술어 선점 금지). 동시에 "Why graceful degradation is expected" 문단의 구조 논거와 모순되는지 점검한다 — 예컨대 급락 지점이 "batch 내 labeled positive 소멸"과 일치한다면 그것은 논거 2의 예측 범위 안이므로 해석을 보강하면 되고, 논거와 정면 충돌하면 해당 문단 자체를 수정해야 한다. floor 아래로 떨어지는 점이 관찰되면 "without falling below the unsupervised floor" 문장은 유지 불가 — 사실대로 보고하고 한계로 서술한다. 어느 경우든 그림과 본문 서술을 침묵 불일치 상태로 두는 것은 금지다.

## 🧪 실험 내용과 설계

**`[신규 실행]`** — 전용 파라미터가 코드에 없으므로 소규모 구현 후 실행한다. 단, p=1.0 점은 main 설정과 동일하므로 **[271c] 재사용**(재학습 금지 — 그 점만 추출).

**구현 — 재사용할 기존 메커니즘 2개 (새로 발명하지 말 것)**:

1. `mae_anomaly/datasets/noisy.py`의 `NoisyLabelSlidingWindowDataset` — 학습 split에서만 변형 라벨을 반환하고 평가에는 원본 라벨을 쓰는 구조(`use_noisy_labels = (split=='train')`)가 이미 있다. 희소화를 "학습 입력에만" 주입하는 정확한 골격.
2. `scripts/run_base_experiments.py:397-416`의 `apply_normal50_noise` — train 구간 anomaly **region 단위** 50% 무작위 재라벨(seed=123)의 기존 구현. 이것을 비율 p로 일반화한 `apply_label_sparsity(regions, p, seed)`를 만들고, config에 `label_keep_ratio: float = 1.0`을 추가한다 — **키워드 전용, 기본 1.0 = 현행과 비트 동일** (CLAUDE.md API 체크리스트 2항: 행동을 바꾸는 새 필드의 침묵 기본값 금지 원칙과 정합).

**조작 단위와 의미**: region 단위 무작위 선택(점 단위 아님) — "기록된 fault 사건" 개념과 일치하며 원고 §4.4 Design 문단("region granularity, as operational logs record faults")과 합치한다. 미선택 region은 **데이터는 train에 그대로 남기고 라벨만 0으로** 둔다(절제 아님 — unlabeled anomaly로 잔류시키는 것이 실험의 핵심). seed 고정: region 선택 seed=123 계열, p별 동일 seed.

**라벨 영향 경로 확인**: force_mask_anomaly의 우선순위, GRL classifier target, OD 손실의 정상/이상 분기 — 세 경로 전부 `point_labels`를 경유하므로 NoisyLabel 주입 한 곳으로 일괄 제어된다 (EXPERIMENT_PROTOCOL_TRUTH §⑦ 실측). 별도 경로별 처리 불필요.

**실행 매트릭스**: 대표 데이터셋 2–3개(NUM-026; 권장 SWaT excl22 + PSM, 여유 시 WaDi A1 추가) × p ∈ {0.75, 0.5, 0.25, 0.1} = **8–12 run**. 각 run은 271 canon config 그대로(500 epochs, seed 42), `config_override`에 `label_keep_ratio=<p>`만 추가한 큐 항목으로 등재한다 (`configs/queue_dedup_renumbered_v5.json` 형식: `exp_num` / `dataset` 리스트 / 공백 구분 키=값). 분할·정규화·평가·best-epoch 기준 등 그 외 전부 불변 — 변경되는 것은 학습 라벨뿐이다.

**집계 규칙**: 각 (데이터셋, p) 점은 해당 run의 best epoch(`pak_auc_f1` 기준; SWaT excl22는 `excl22_pak_auc_f1`) `metrics.pak_auc_f1`. 점선 floor는 Table 2 확정본의 anomaly-excised 조건 best unsupervised baseline 값을 그대로 가져온다.

## 📊 구성과 형태

- **X축**: labeled fraction p (0.1 → 1.0, 선형 눈금).
- **Y축**: PA%K-AUC F1.
- **계열**: 데이터셋별 실선 1개 + 같은 색 점선(해당 데이터셋의 unsupervised floor) 1개. 범례에 데이터셋명.
- **강조**: p=1.0 점은 main 설정과 동일함을 마커로 강조 가능.

## 📝 캡션 (영문 확정본 — [N]은 NUM-026 확정 후 치환)

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

## ⚠️ 주의사항과 의존성

- ⓐ **NUM-026(데이터셋 수)·NUM-027(열화 형상 서술어)이 이 실험에서 파생** — 같은 소스, 동시 확정.
- ⓑ 점선 floor는 **Table 2 확정값에서만** 가져온다 (TAB-2 의존성). SMD/SMAP/MSL baseline 신규 실행(우선순위표 §7.3 #1)이 끝나기 전에는 해당 family를 대표 데이터셋으로 쓰는 경우 floor를 확정할 수 없다 — 권장 선택(SWaT excl22, PSM)은 [CMP-Q3] 재사용 가능 family라 이 함정을 피한다.
- ⓒ p→0 극한과 Table 2 protocol-effect 블록의 "CSMAD (clean)"은 **다른 조건**이다 — clean-split은 prefix 자체가 train에 없는 반면, p=0은 비라벨 anomaly가 train에 남는다. 본문 상호참조 시 "approximates"라는 표현을 유지하고 동일시하지 말 것.
- ⓓ §4.4 "Why graceful degradation is expected" 문단의 구조 논거(배치에 labeled positive가 없으면 GRL 손실 자체가 미계산 — `loss.py:293-302`)와 결과 해석의 일관성을 곡선 확정 후 확인.
- 큐 등재 시 `force_mask_anomaly` 키 중복 같은 last-wins 패턴(exp287 원항목의 전례)을 답습하지 말 것 — override는 키당 1회만.

## 🔢 연결된 수치 placeholder

| ID | 본문 위치 | 들어갈 값의 정의 |
|---|---|---|
| NUM-026 | §4.4 Results lead의 [N] + 캡션 [N] 2곳 (동시 치환) | FIG-3 대표 데이터셋 수 — 설계 선택 2 또는 3 (권장: SWaT excl22 + PSM = 2; WaDi A1 추가 시 3) |
| NUM-027 | §4.4 Results 문장의 서술어 [gradually / monotonically] | 열화 형상의 정성 서술어 — **곡선 확정 후** 실제 형상에 맞는 쪽 선택. 비단조면 두 단어 모두 버리고 문장 재작성 (A8 — 곡선 없이 단어 선점 금지) |

---

<!-- PAGE: FIG-4 -->

# FIG-4 — 정성적 score 분해 (Qualitative score decomposition)

> 💡 **한 줄 요약**: 대표 anomaly 사건 구간에서 CSMAD 점수를 Teacher 재구성 오차와 Teacher–Student discrepancy로 분해해 나란히 그려, 두 성분이 서로 다른 신호를 낸다는 방법론의 핵심 설계 논리를 실제 데이터 위에서 보여주는 그림.

| 항목 | 내용 |
|---|---|
| 위치 | §4.5 lead 직후 (`sec4_experiments.tex`, `\label{fig:decomp}`) |
| 크기 | full-width, 3.5–4 cm ≈ 0.30p |
| 소스 분류 | `[재사용]` + 추출 스크립트 — [271c] 완주분 checkpoint 재사용, 신규 학습 불필요 |

## 🎯 목적과 의도

§4.2(얼마나 잘하는가)와 §4.3(어느 component 덕분인가)이 끝난 뒤, §4.5는 "실제로 어떻게 작동하는가"를 보여주는 자리다. CSMAD의 점수는 두 성분의 합 — Teacher 재구성 오차 r_i와 adaptive 스케일링된 Teacher–Student discrepancy — 인데, 이 합산 설계가 의미를 가지려면 두 성분이 **서로 다른 정보를 담는다**는 것을 보여야 한다. 둘이 항상 같은 모양이라면 discrepancy 성분은 군더더기라는 비판(사실상 "recon만으로 충분하지 않은가")이 성립하기 때문이다. 이 그림은 TAB-3 행 4(w/o OD — 자동 recon-only)의 정량 결과와 짝을 이루는 **정성 증거**로, 같은 질문을 평균 수치가 아니라 실제 사건의 시간축 위에서 답한다.

§12의 R10 논증 두 건이 이 그림에 직접 걸려 있다. (1) asymmetric Teacher–Student — "용량 격차가 비정상 상관 패턴에서 모방 실패를 키운다"는 주장은 행 3(discrepancy)이 anomaly 구간에서 상대적으로 솟는 모양으로 시각화된다. (2) adaptive scoring — 데이터셋마다 recon/disc의 절대 스케일이 크게 다른데도 두 성분이 한 그림에서 비교 가능한 것 자체가 adaptive scaling의 효과다(행 3은 스케일 적용 후 값을 그린다). 추가로 행 4의 threshold 점선은 anomaly-ratio threshold가 실제 점수 분포 위에서 어떻게 작동하는지 보여줘, "threshold selection이 불공정하다"는 §15 공격에 대해 oracle threshold가 아님을 시각적으로 재확인시킨다.

열 선택(SWaT excl22 포함)도 논증적이다: excl22는 region 22 제거 후 남는 소형·다양한 사건들 위주의 조건이므로, 이 그림이 excl22의 사건들을 다루는 것은 "단일 대형 사건에 의존하지 않는다"는 §4.2의 주장과 호응한다.

## 🏁 목표와 기대 결과

**입증하려는 것**: 캡션의 마지막 문장 그대로 — 재구성 오차는 사건 유형과 무관하게 정상 패턴 이탈에서 상승하고, discrepancy는 용량 격차와 라벨 유도 학습이 증폭하는 구조적 발산을 별도로 포착한다는 것. 성공 기준은 (1) 두 성분의 시간 형상이 사건별로 식별 가능하게 다를 것, (2) 합산 score(행 4)가 GT 음영 구간에서 threshold 점선을 상회할 것, (3) 4행 모두 GT 음영과 시간축이 정확히 정렬될 것.

**기대와 다른 패턴이 나오면**: 두 성분이 모든 사건에서 사실상 동일 형상이면, §4.5 본문의 해석 문장("The two components respond distinctly...")을 실제 관찰에 맞게 약화·재작성한다 — 수치·관찰 확정 전 해석 강화 금지(RT MINOR-02)가 이 경우의 명령이다. 또한 그런 결과는 TAB-3 행 4의 하락폭 해석과 함께 읽어야 한다(정량 하락이 있는데 정성 그림에서 차이가 안 보이면 사건 선택이 대표성이 없는 것일 수 있으므로 다른 사건 구간으로 교체를 먼저 시도). 후보 열(WaDi A1 vs PSM) 중 시각적 변별이 좋은 쪽을 선택하는 절차 자체가 이 대응의 일부로 설계되어 있다.

## 🧪 실험 내용과 설계

**`[재사용]` — [271c] 완주분에서 추출만 수행. 신규 학습 불필요.**

- **점수 추출**: 해당 entity의 best checkpoint를 로드해 evaluator의 **동일 scoring 경로**로 per-timestep 배열 3종을 추출한다: `recon`(Teacher MSE), `scaled_disc = disc × (recon_mean/disc_mean)`, `score = recon + scaled_disc/4.0` (정본 산식: 271_CONFIG_TRUTH §VIII Anomaly Score). 구현의 단일 원천은 `mae_anomaly/scoring.py` — **다른 곳에 식을 복제하지 말 것** (CLAUDE.md API 체크리스트 3항; 2026-05-28 FM-omission 사고의 재발 방지 조항).
- **threshold 점선**: 해당 entity metadata의 `metrics.anomaly_ratio_threshold` 값을 그대로 사용 (예: [271c] PSM 0.001744). 재계산 금지.
- **사건 구간 선택**: SWaT excl22는 region 22 마스킹 후 남는 13개 소형 사건 중 **유형이 다른 사건 ≥2개**를 포함하도록 선택한다 (RT MINOR-02 — 사건 규모·유형 대표성). 구간 폭은 사건 길이의 3–5배 컨텍스트를 포함할 것을 권장.
- **열 2 선택**: WaDi A1 또는 PSM 중 추출 결과를 보고 시각적 변별이 좋은 쪽 — 선택 결과가 NUM-028(=2)과 캡션의 [Dataset-A/B] 치환을 확정한다.

## 📊 구성과 형태

2열(데이터셋) × 4행(분해 단계). 열 내 4행은 X축(timestep) 공유, 행별 Y는 per-trace 정규화.

| 행 | 내용 | 특기 |
|---|---|---|
| 1 | 입력(첫 feature) + GT anomaly 붉은 음영 | 음영은 4행 전체에 연하게 관통 (정렬 확인용) |
| 2 | Teacher 재구성 오차 (per timestep) | |
| 3 | Teacher–Student discrepancy (adaptive 스케일 적용 후) | |
| 4 | 합산 anomaly score + anomaly-ratio threshold 점선 | 점선은 이 행에만 |

## 📝 캡션 (영문 확정본 — [Dataset-A/B]는 선택 확정 후 치환)

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

## ⚠️ 주의사항과 의존성

- ⓐ **Gaussian smoothing 절대 금지** (R34). [271c]의 저장 점수는 전부 비평활이므로 추출값을 그대로 그리면 자동 준수되지만, 시각화 코드 단계에서 후처리 smoothing을 끼워 넣지 말 것.
- ⓑ §4.5 해석 문장("two components respond distinctly...")은 실제 그림 확정 후 사건별 관찰에 맞게 재검토한다 (RT MINOR-02 — 확정 전 해석 강화 금지).
- ⓒ NUM-028이 이 그림에서 파생 — 그림 제작과 동시 치환.
- 점수 산식·threshold를 그림 주석에 쓸 경우 §3.6 수식 번호(Eq. dscale/sigma)와 표기 일치 확인.

## 🔢 연결된 수치 placeholder

| ID | 본문 위치 | 들어갈 값의 정의 |
|---|---|---|
| NUM-028 | §4.5 lead의 [N] | FIG-4 데이터셋 수 = **2** (시각화 설계 확정값 — SWaT excl22 + {WaDi A1 또는 PSM}). FIG-4 제작과 동시 치환 |

---

<!-- PAGE: TAB-1 -->

# TAB-1 — 데이터셋 통계 (Table 1: Dataset statistics)

> 💡 **한 줄 요약**: 6개 벤치마크 family의 재분할 후 train/test 크기와 anomaly 비율을 투명하게 공개해, contaminated benchmark protocol의 실측 기반을 제시하고 프로토콜 방어의 첫 단추를 끼우는 표.

| 항목 | 내용 |
|---|---|
| 위치 | §4.1.1 (`sec4_experiments.tex`, `\label{tab:datasets}`), ~0.25p |
| 소스 분류 | `[재사용]`(대부분 실값 확정) + `[신규 측정]`(SMD per-machine 셀 — 학습 불필요, 스크립트 1회) |

## 🎯 목적과 의도

이 표는 단순한 데이터셋 소개가 아니라 **프로토콜 방어의 정량 기반**이다. 본 논문의 가장 큰 reviewer 공격면은 test-prefix 편입 프로토콜("test label로 학습하는 leakage 아닌가" — 블루프린트 §15 첫 행)인데, §14의 정면 답변 5논거 중 ②(원본 train에는 labeled anomaly가 구조적으로 부재)와 ④(시간성·전 데이터셋 단일 규칙)는 결국 숫자로 증명된다: Train AR 열이 "training 구간의 anomaly가 전적으로 편입된 prefix에서 유래한다"는 사실을, #Train/#Test 열이 재분할 규칙(//2)이 전 데이터셋에 동일하게 적용되었음을 보여준다. 캡션이 "originating from the incorporated test prefix"를 명시하는 것도 같은 이유다.

또한 이 표는 비교 공정성 논증의 입력값이다. §4.1.4가 인정하는 train 데이터 양적 비대칭(anomaly-excised 조건에서 baseline의 train이 절제분만큼 작다)의 크기가 바로 Train AR 열(0.52–6.20%)이고, FIG-3의 데이터셋 선택 논리(PSM이 train AR 최대 → 라벨 경로 최활성)도 이 표를 근거로 한다. SWaT 행의 dagger(full/excl22 병기)는 §4.1.1 SWaT dual evaluation 문단과 부록 §A.4로 이어지는 excl22 서사의 출발점이다. 요컨대 reviewer가 프로토콜을 공격하려면 가장 먼저 보게 될 표이며, 여기서의 투명성이 이후 모든 방어의 신뢰도를 결정한다.

## 🏁 목표와 기대 결과

이 표의 "성공"은 성능이 아니라 **정합성**이다: (1) 모든 셀이 EXPERIMENT_PROTOCOL_TRUTH §① 실측과 일치, (2) SMD per-machine 산출이 코드의 분할 산식과 동일한 규칙으로 계산됨, (3) 본문·부록의 동일 수치 인용처(§4.1.1 본문 범위 문장, Table A.4, §C.1 차원 표)와 자리수까지 일치.

**기대와 다른 패턴이 나오면**: SMD per-machine Train AR이 기존 공개 범위(0.52–6.20%)를 벗어나는 machine이 있으면, §4.1.1 본문의 "Training anomaly ratios range from 0.52% to 6.20% (SMD per-machine values pending...)" 문장의 **범위 수치 자체를 같은 pass에서 수정**한다 — 표만 채우고 본문 범위를 방치하는 부분 수정은 금지다. 이 갱신은 §4.1.4의 양적 비대칭 인정 문장(0.52%–6.20%; SMD pending)에도 동일하게 적용된다.

## 🧪 실험 내용과 설계

**대부분 `[재사용]`** — 다음 실값이 이미 tex에 반영·확정되어 있다 (EXPERIMENT_PROTOCOL_TRUTH §① 실측): SWaT 719,959 / 224,960 / 45 / 1.63 / 19.05·3.68†, WaDi 1,296,001·870,972 / 86,401·86,402 / 123 / 0.52·0.76 / 3.82·3.87, PSM 176,401 / 43,921 / 25 / 6.20 / 30.63, SMAP 355,905 / 217,925 / 25 / 0.70 / 24.54, MSL 95,271 / 36,775 / 55 / 1.70 / 16.72. SMD의 Test AR 평균 4.16도 실값.

**잔여 `[신규 측정]`** — SMD 행의 per-machine 위임 셀(#Train, #Test, Train AR): 28개 machine 각각을 산출하는 **1회성 스크립트** (학습 불필요). 산출 규칙은 코드와 동일해야 한다 — `loaders.py:1152-1157`의 분할(`test_split = len(test_data)//2`; train = 원본 train 전체 + test 앞 50%, test = 뒤 50%)을 그대로 호출하거나, 같은 산식으로 라벨 파일에서 직접 계산한다. 본문 표는 "per-machine (§A.3)" 포인터 형태를 유지하므로, 이 산출물의 실제 게재처는 Table A.4(SMD per-machine 행)와 §4.1.1 본문의 "pending" 문구 해소다 — **TAB-1과 Table A.4는 동일 소스 산출물**이며 두 표 간 수치 불일치는 금지.

## 📊 구성과 형태

booktabs 6행 — 형태는 tex 확정, 변경 불필요. SWaT Test AR은 dagger(†)로 full/excl22 병기(캡션에 정의). 확정 셀 미리보기:

| Family | #Train | #Test | #Dim. | Train AR | Test AR |
|---|---|---|---|---|---|
| SWaT (A1+A2) | 719,959 | 224,960 | 45 | 1.63 | 19.05 / 3.68† |
| WaDi (A1/A2) | 1,296,001 / 870,972 | 86,401 / 86,402 | 123 | 0.52 / 0.76 | 3.82 / 3.87 |
| PSM | 176,401 | 43,921 | 25 | 6.20 | 30.63 |
| SMD (×28) | per-machine (§A.3) | per-machine (§A.3) | 29–36 | per-machine | 4.16 (avg) |
| SMAP (×54) | 355,905 | 217,925 | 25 | 0.70 | 24.54 |
| MSL (×27) | 95,271 | 36,775 | 55 | 1.70 | 16.72 |

## 📝 캡션 (영문 확정본)

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

## ⚠️ 주의사항과 의존성

- SMD per-machine Train AR 확정 시 §4.1.1 본문 범위 문장("0.52% to 6.20% ... pending")을 **같은 pass에서** 갱신 — SMD 값이 범위를 벗어나면 범위 수치 자체 수정 (부분 수정 금지).
- #Dim 열은 §4.1.1이 단일 원천 — 부록 §C.2의 Table C.1(입력 차원 표)과 정합 유지 의무.
- Table A.4(per-entity statistics)와 동일 소스·동일 산식 — 두 표 간 불일치 금지. A.4 캡션의 "SMD per-machine rows pending" 문구도 채움과 동시에 삭제.
- SMD 차원 29–36은 constant 컬럼 제거 후 수치 — raw 38로 되돌려 쓰지 말 것 (ADV BLK-003).

## 🔢 연결된 수치 placeholder

전용 NUM placeholder 없음 — 잔여 placeholder는 표 안의 SMD per-machine 셀 자체다. 대신 다음 동기화 의무가 NUM에 준해 적용된다:

| 동기화 대상 | 위치 | 규칙 |
|---|---|---|
| Train AR 범위 문장 | §4.1.1 본문 ("range from 0.52% to 6.20% ... pending") | SMD 확정과 같은 pass에서 갱신 (범위 이탈 시 범위 수정) |
| 양적 비대칭 인정 문장 | §4.1.4 ("0.52\%--6.20\%; SMD pending") | 동일 |
| Table A.4 SMD 행 | §A.3 | 동일 스크립트 산출물 사용 — 수치 불일치 금지 |

---

<!-- PAGE: TAB-2 -->

# TAB-2 — Main 비교 결과 + protocol-effect 블록 (Table 2: Main comparison results) ★ 본 논문의 중심 표

> 💡 **한 줄 요약**: 26개 baseline(22 unsupervised + 4 weakly supervised)과 CSMAD를 6 family × 2 지표에서 비교하고, 하단 protocol-effect 블록으로 "성능 우위가 프로토콜의 추가 데이터 때문인가, 방법 때문인가"를 분리해 보이는, placeholder 의존 그래프의 루트가 되는 본 논문의 중심 표.

| 항목 | 내용 |
|---|---|
| 위치 | §4.2 (`sec4_experiments.tex`, `\label{tab:main_results}`), table* 2단 폭, ≈0.55p |
| 소스 분류 | `[완주 대기]`(CSMAD) + `[신규 실행]`(baseline 일부·weak 4종·standard-split) |
| 특기 | **TAB-4(protocol-effect analysis)는 이 표의 하단 블록으로 흡수 완료** (D-010 ① — 별도 표·별도 Notion 페이지 없음; 본 페이지가 그 명세를 전부 포함) |

## 🎯 목적과 의도

이 표는 논문의 중심 논제 — "labeled anomaly를 표현 학습에 직접 통합한 end-to-end 단일 모델이, 같은 라벨을 각자의 패러다임에서 최선으로 활용한 기존 방법들보다 낫다" — 를 입증하는 단일 증거물이다. 행 구성 자체가 논증이다: 22개 unsupervised baseline은 **anomaly-excised condition**(라벨로 오염원을 제거해주는, 그들에게 가장 유리한 조건)에서, 4개 weakly supervised baseline은 그들이 구조적으로 요구하는 **contaminated-training condition**에서 평가된다. 즉 "라벨 있는 우리 vs 라벨 없는 그들"이라는 불공정 구도가 아니라 "같은 라벨을 각자 최선으로 쓴 비교"(블루프린트 §14 논거 ③, R12)임을 표의 조건 표기가 직접 말한다. NRdetector 행은 §1이 "closest prior work"로 지목한 최직접 경쟁자와의 정면 비교다.

하단 **protocol-effect 블록**(r2에서 TAB-4를 흡수)은 이 논문에서 가장 위험한 reviewer 공격 — "성능 우위가 GRL+distillation 때문이 아니라 test-prefix 편입으로 늘어난 train 데이터 때문 아닌가"(블루프린트 §15, RT BLOCKER-03) — 에 대한 정면 답변이다. 2단 논증 구조: ① 동일 방법이 standard clean-train split에서도 비지도 SOTA와 경쟁력을 유지한다 → 성능이 프로토콜의 산물이 아니라 방법 자체의 가치임을 보임. ② labeled anomaly가 제공되는 contaminated 조건에서는 CSMAD만 추가 이득을 얻고, 비지도 baseline은 같은 데이터가 추가되어도 라벨을 활용하지 못한다 → 이득이 라벨 활용 능력에 특이적임을 보임. 두 조건의 평가를 동일한 원본 test 뒤 50%로 통일하는 것이 이 분리의 기술적 핵심이다(비교가 train 구성 차이만 반영하게 됨).

이 표가 §4.2 분석 텍스트의 네 구조(요약 주장 / 데이터셋별 특이점 / protocol-effect 해석 / 비용 한계 인정)를 전부 먹여 살리며, NUM 4개 그룹(N-A·N-B·N-C·N-D), FIG-3의 floor, TAB-B1의 Δ 기준이 모두 여기서 파생된다 — **placeholder 의존 그래프의 루트**다.

## 🏁 목표와 기대 결과

**입증하려는 패턴** (수치 예측이 아니라 방향): (1) CSMAD가 6 family의 다수에서 두 지표 모두 최상위권에 위치하고, 특히 train AR이 가장 높은 PSM(라벨 경로가 가장 강하게 발동)에서 라벨 활용의 이득이 뚜렷할 것. (2) SWaT excl22(소형·다양 사건만 남는 조건)에서도 경쟁력을 유지해 "단일 대형 사건 탐지에 의존하지 않는다"가 성립할 것. (3) protocol-effect 블록에서 clean-split CSMAD가 비지도 대표와 비등하고, contaminated로 옮기면 CSMAD만 유의미하게 상승하며 비지도 baseline의 변화는 그에 못 미칠 것.

**기대와 다른 패턴이 나오면**: 어떤 family에서 1위를 놓치면 NUM-006의 win 수는 그대로 사실대로 기재하고, §4.2 요약 문장의 강도를 결과에 맞춰 조정한다(과장 금지 — "achieves the highest on [N] of six"는 어떤 N에도 문법적으로 성립하도록 이미 설계되어 있다). protocol-effect에서 clean-split CSMAD가 비지도 대표에 크게 밀리면 2단 논증의 ①이 약화되므로, 분석 문단을 "방법 자체의 경쟁력" 주장에서 "라벨 활용 이득"(② 중심)으로 재구성해야 하며, 반대로 비지도 baseline이 contaminated 조건에서 CSMAD에 준하는 이득을 보이면 NUM-019의 해석("confirming that the gain is specific to methods able to exploit the provided labels")을 그대로 둘 수 없다 — 어느 경우든 표와 본문 문장의 침묵 불일치는 금지이고, 문장 쪽을 결과에 맞춘다.

## 🧪 실험 내용과 설계

**셀 값 정의** — 27 method 행(7개 그룹) × 7 데이터셋 열 {SWaT excl22, WaDi A1, WaDi A2, PSM, SMD avg, SMAP avg, MSL avg} × 2지표 {PA%K-AUC F1, VUS-PR} + 하단 protocol-effect 블록 3행:

- **CSMAD 행**: [271c] entity별 `experiment_metadata.json`의 `metrics.pak_auc_f1` / `metrics.vus_pr` (best epoch 기준 — 전 지표가 같은 best epoch에서 추출됨). SWaT 열은 `SWaT/A1A2_excl22` entity(독립 best-epoch, `timing.best_epoch_metric='excl22_pak_auc_f1'`). SMD/SMAP/MSL avg = **entity별 best-epoch 지표의 macro 평균**(28/54/27 entity).
- **unsupervised 22행**: anomaly-excised condition([CMP-Q3] 계열) 동일 키. random 행만 5-run mean(±std는 본문 비표기, §A.1에 명시).
- **weakly supervised 4행**: contaminated-training condition 단독(구조적으로 excised 불가 — §4.1.4).
- **protocol-effect 블록**: CSMAD(clean) + 대표 baseline 2–3종(NUM-014)의 standard clean-train split 결과 — 대표 열(SWaT excl22, WaDi A1, PSM — tex stub 기준)만 채우고 나머지는 "—".

**실험 소스 — 4갈래** (실행 우선순위는 §7.3 역인덱스 기준 #1–#3):

1. **CSMAD `[완주 대기]`**: 271canon 잔여 entity 완주(SMD 6, SMAP 49, MSL 22 — 2026-06-11 실측, 큐 진행 중). 완주 후 metadata 집계 스크립트로 macro 평균 산출. **부분 완주 상태로 avg 열을 채우는 것 금지** — sync 그룹 A("six families")가 깨진다.
2. **unsupervised 22종 `[신규 실행(부분)]`**: SWaT/WaDi/PSM은 [CMP-Q3](`comparison/results/experiments/6_20260526_085028_baseline_minmax_normalonly_segaware/`) 재사용 가능. **SMD/SMAP/MSL은 `comparison/run_baseline_queue.py`로 전 entity 신규 실행 필수** — SMD normalonly 기존 결과는 per-entity 정규화(2026-06-02) 이전의 구버전 `3_20260312_*`뿐이라 폐기 대상이고, SMAP/MSL normalonly는 어느 결과 폴더에도 부재(미실행)다 (r2 정정 — "STALE 재실행"이 아니라 "SMD 구버전 폐기+재실행 / SMAP·MSL 미실행분 신규 실행"). variant는 `normalonly`(각 baseline의 `experiment_configs.py` 등록 항목 그대로; SMAP/MSL 포함 등록 실재 확인됨). SMD 재실행 시 per-entity 정규화 적용을 실행 전 확인.
3. **weakly supervised 4종 `[신규 실행]`**: DeepMIL/WETAS/TreeMIL/NRdetector — 구현·CPU dry-test는 완료, **GPU 전체 실험 미실행**. contaminated-training(full/Q1) variant로 전 데이터셋 실행 (epochs 50, 매 epoch eval — `baseline_common.py` weak preset). NRdetector가 최직접 경쟁자이므로 그룹 6 중 최우선.
4. **protocol-effect 블록 `[신규 실행 + 신규 loader]`** (흡수된 TAB-4의 실행 사양 — 블루프린트 §6.6 r3, 코드 근거 포함):
   - **분할**: train = 원본 train 파일만(test-prefix 미편입, 라벨 anomaly 0), test = **main protocol과 동일한 원본 test 뒤 50%**. 평가 통일이 핵심 — 비교가 train 구성 차이만 분리하게 된다. 현행 loader에 이 variant가 없으므로 loader 함수/variant 추가가 필요하다(예: `*_standard` 키; 기존 `//2` 분할 코드의 train_len에서 prefix 항만 빼는 최소 수정).
   - **CSMAD 설정**: 271 canon config **그대로, `use_grl=True` 유지**. 라벨 0인 train에서 세 라벨 경로는 코드 수준에서 자가 비활성화된다: anomaly-priority masking은 priority 전부 0 → 무작위 마스킹으로 자연 퇴화, OD 분기는 전 패치 정상(정상 전용과 동일), GRL은 batch 내 positive 부재 시 손실 자체가 계산되지 않음(`loss.py:293-302`). ⚠️ **`use_grl=False`로 끄는 것 금지** — dead component(dynamic margin anomaly loss)가 재활성화되어 비교가 오염된다 (§6.7과 동일한 함정).
   - **baseline**: 대표 2–3종(NUM-014; 선정 기준 — main 표에서 강한 unsupervised 대표, 예: 최상위 recent 1 + legacy 1)을 동일 standard split에서 학습. 대표 데이터셋 3개(SWaT excl22, WaDi A1, PSM) 한정으로 비용 통제.

**집계 규칙**: baseline 쪽 SMD/SMAP/MSL avg도 CSMAD와 **동일한 entity 집합·동일 macro 평균 규칙**이어야 한다. 집계에서 Exathlon·Simulation은 절대 배제(R33) — 기존 Notion RankAvg류 수치는 Exathlon 포함 기준이므로 재계산 필수(FEEDBACK-3).

## 📊 구성과 형태

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

## 📝 캡션 (영문 확정본 — [N]은 NUM-014 확정 후 치환)

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

## ⚠️ 주의사항과 의존성

- ⓐ **이 표가 placeholder 의존 그래프의 루트** — NUM-006~013(본 블록), NUM-014~019(하단 블록), FIG-3 점선 floor, TAB-B1 Δ 기준이 전부 이 표에서 파생된다. 이 표의 확정 전에 파생 placeholder를 선기입하는 것 금지 (NUM-010 같은 "현재도 산출돼 있는 값"도 표 전체 확정 전 본문 선기입 금지).
- ⓑ 집계에서 Exathlon·Simulation 절대 배제 (R33; 기존 Notion RankAvg 재계산 필수 — FEEDBACK-3).
- ⓒ **weak 4종 미완 시 fallback**: sync 그룹 B 전체가 "22 unsupervised"로 일괄 전환 + Table 2 그룹 6(Weakly supervised) 행 삭제 + §4.1.2–4.1.4 하드코딩("26 baselines / 22 / 4") 동시 수정 — **부분 게재 금지**.
- ⓓ SWaT 재실행이 발생하면 입력 차원 45 일치 검증 필수 (FEEDBACK-7 — 현 machineA raw CSV 경로는 51을 반환; 불일치 시 checkpoint 로드 실패 가능).
- ⓔ baseline 쪽 SMD/SMAP/MSL avg도 CSMAD와 동일한 entity 집합·동일 macro 평균 규칙 — 반올림 자리수까지 Table A.8(per-entity)과 일관.
- **TAB-4 흡수 기록**: protocol-effect analysis는 v2-r2에서 본 표 하단 블록으로 흡수되었다 (D-010 ①). 본문에 `[TAB-4]` 마커는 존재하지 않으며, 별도 Notion 페이지도 생성하지 않는다 — 명세·실행 지침·의존성은 본 페이지 🧪 4항에 통합되어 있다.

## 🔢 연결된 수치 placeholder

**그룹 N-A — 데이터셋 family 수 (sync 그룹 A) `[완주 대기]`** — 4개소 단일 값 동기화 의무:

| ID | 본문 위치 | 들어갈 값의 정의 |
|---|---|---|
| NUM-001 | Abstract 6문장 (`main.tex`) | family 수 — 6 family 전부 완주 시 "six" |
| NUM-003 | Highlights bullet 5 (`main.tex` highlights 블록 + `highlights.txt`) | 동일 값 (sync) |
| NUM-004 | §1 contribution bullet 4 (`sec1_intro.tex`) | 동일 값 (sync) |
| NUM-029 | §5 결론 (`sec5_conclusion.tex`) | 동일 값 (sync) |

네 곳이 단일 값으로 동기화되어야 하며, §4.1.1 하드코딩 상수("six ... families", "113 entities / 114 evaluation conditions")·§4.2 "six dataset families"와도 일치 의무. 어느 family라도 제출 시점에 탈락하면 같은 pass에서 §4.1.1 상수까지 일괄 수정 (부분 수정 금지). 소스: 271canon 완주 + baseline 재실행 완료 = **TAB-2 완성이 전제**.

**그룹 N-B — baseline 총수 (sync 그룹 B) `[신규 실행(weak 4종) 의존]`**:

| ID | 본문 위치 | 들어갈 값의 정의 |
|---|---|---|
| NUM-002 | Abstract (`main.tex`) | weak 4종 GPU 완주 시 "26" (22 unsup + 4 weak); 미완 시 세 곳 모두 "22 unsupervised"로 fallback |
| NUM-005 | §1 contribution bullet 4 | 동일 값 (sync) |
| NUM-030 | §5 결론 | 동일 값 (sync) |

fallback 시 §4.1.2–4.1.4 하드코딩("26 baselines / 22 / 4")과 Table 2 그룹 6 행을 동시 제거. 소스: 본 페이지 🧪 3항(weak 4종 Q1 GPU 실행)과 동일.

**그룹 N-C — Table 2 본 블록 파생 `[집계만 — TAB-2 완성 후]`** (전부 신규 실험 없음):

| ID | 본문 위치 (§4.2) | 들어갈 값의 정의 (집계 규칙) |
|---|---|---|
| NUM-006 | ¶1, [N]×2 | 6 family 중 CSMAD가 1위인 family 수 — PA%K-AUC F1 기준 1개 + VUS-PR 기준 1개. **WaDi 집계 규칙 결정 필요**: 표는 A1/A2 2열인데 본문은 "six families" — 권고: A1·A2 모두 1위일 때만 WaDi family win (보수적), 채택 규칙을 본문 또는 각주 1줄로 명시 |
| NUM-007 | ¶1, [X.XX]×2 | CSMAD의 family 평균 (PA%K-AUC F1, VUS-PR) — WaDi는 A1/A2 평균을 family 값으로 한 뒤 6 family 평균 (규칙을 006과 통일) |
| NUM-008 | ¶1 | (CSMAD 평균) − (family별 최강 unsupervised의 평균), PA%K-AUC F1 |
| NUM-009 | ¶1 | 동일, VUS-PR |
| NUM-010 | ¶2 | CSMAD PA%K-AUC F1 @ PSM ([271c] PSM `metrics.pak_auc_f1` — 표 전체 확정 전 본문 선기입 금지) |
| NUM-011 | ¶2 | best unsupervised PA%K-AUC F1 @ PSM ([CMP-Q3]) |
| NUM-012 | ¶2 | CSMAD PA%K-AUC F1 @ SWaT excl22 ([271c] `SWaT/A1A2_excl22`) |
| NUM-013 | ¶3, [X.XX]×2 | NRdetector(contaminated-training) 대비 비교값 — registry 정의는 "margins", tex 문장은 CSMAD **절대값** 형태("CSMAD achieves [X.XX] ... on average"). 채움 시 문장·정의 중 한쪽으로 확정 (권고: 문장을 "outperforms NRdetector by [margin]"으로 고치거나, 절대값 유지 + 본문에 NRdetector 값 병기 — 침묵 불일치 금지) |

NUM-008/009/011의 "최강 unsupervised"는 family별로 다른 방법일 수 있다 — 평균 산출 규칙(각 family의 best를 뽑아 평균 vs 단일 최강 방법의 평균)을 명시하고 일관 적용 (권고: 전자 — "strongest unsupervised competitor"의 보수적 해석).

**그룹 N-D — Protocol-effect 블록 파생 `[신규 실행 — standard-split run]`** (전부 PA%K-AUC F1):

| ID | 본문 위치 (§4.2 protocol-effect 문단) | 들어갈 값의 정의 |
|---|---|---|
| NUM-014 | 블록 캡션 [N] + 본문 [N] (동시 치환) | 블록 내 대표 baseline 수 (설계 선택 2–3; tex stub은 A/B 2행) |
| NUM-015 | "CSMAD remains competitive ([X.XX] ...)" | CSMAD clean-train 평균 (protocol-effect 대표 데이터셋들) |
| NUM-016 | "... versus [X.XX] for the best unsupervised competitor" | best unsupervised clean-train 평균 |
| NUM-017 | "CSMAD improves to [X.XX]" | CSMAD contaminated 평균 — **Table 2 본 블록의 같은 데이터셋 부분집합 재집계** (신규 실행 아님) |
| NUM-018 | "(a gain of [X.XX] points)" | 파생 계산값: 017 − 015 (별도 측정 금지) |
| NUM-019 | "the unsupervised baselines show [X.XX] change" | best unsupervised의 조건 간 변화량 (standard → contaminated). **주의**: 비교쌍은 standard-split run vs **contaminated-training(무절제) run** — anomaly-excised가 아니라 "같은 추가 데이터를 받은" 조건. contaminated 쪽은 TAB-B1 실행분과 소스 공유 가능 |

---

<!-- PAGE: TAB-3 -->

# TAB-3 — Ablation study (Table 3)

> 💡 **한 줄 요약**: 세 가지 라벨 유도 경로(anomaly-priority masking, OD loss, GRL)를 하나씩 제거한 변형과 full model을 대표 데이터셋에서 비교해, contribution bullet 2의 "세 경로 각각이 기여한다"는 주장을 정량 분해하는 표.

| 항목 | 내용 |
|---|---|
| 위치 | §4.3 (`sec4_experiments.tex`, `\label{tab:ablation}`), half-width, ≈0.20p |
| 소스 분류 | `[재사용]`(행 1·3) + `[신규 실행]`(행 2·4) |

## 🎯 목적과 의도

§4.2가 "얼마나 잘하는가"를 보였다면 §4.3은 "왜 잘하는가 — 어느 component 덕분인가"를 분해한다(블루프린트의 MECE 설계: component-level 서사는 이 소절 전속, §4.2와 중복 금지). 4행 구성은 contribution bullet 2의 세 경로에 1:1 대응한다: 행 3 ↔ anomaly-priority masking(§12 논증 "anomaly-class imbalance 직접 대응"), 행 4 ↔ loss bifurcation의 OD 손실(§12 "정상에서 낮은 discrepancy 유도 → 대비 증폭"), 행 2 ↔ gradient-reversal suppression(§12 "능동 제거"). 각 변형의 하락폭이 곧 해당 경로의 정량 기여이며, §3의 R10 논증("이게 없으면 왜 나빠지는가")의 실측 검증이다.

이 표에서 가장 정교하게 설계된 것은 **행 2의 정의**다. "w/o GRL"을 단순히 GRL을 끄는 것으로 정의하면, anomaly 패치의 OD-loss 제외(수동 회피)와 GRL(능동 억제)의 효과가 섞여 버린다. 행 2는 OD-exclusion을 **유지한 채** GRL classifier와 reversal만 제거해(RT MAJOR-05), "수동 회피만으로는 부족하고 능동 억제가 추가 기여를 한다"는 §3.5의 핵심 문단("Why gradient reversal is necessary beyond loss bifurcation")과 §1 관찰 문단("Relying only on (b) is insufficient")을 정량적으로 입증한다. 이것이 reviewer의 "GRL이 정말 필요한가 — exclusion만으로 충분하지 않나"라는 공격(이 논문의 novelty 핵심을 겨누는 공격)에 대한 유일한 정량 방어다.

확장 변형(FM, warmup, symmetric decoder, depth sweep)은 부록 Table B.4로 위임되어 본문 표는 "라벨 경로 3종의 분해"라는 단일 메시지에 집중한다 — warmup이 contribution이 아니라는 Phase 3 결정(블루프린트 결정 ①)과 정합하는 배치다.

## 🏁 목표와 기대 결과

**입증하려는 패턴**: 행 1(full)이 기준선 최고치이고, 세 변형 각각에서 Avg 기준 하락이 관찰되는 것. 특히 행 2의 하락(GRL 순효과)이 0이 아니라는 것이 능동 억제 논증의 성패를 가른다. 하락폭의 부호 규약: 본문이 "removal costs X points" / "the drop is X" 형식이므로 NUM-021/022/023은 **양수 하락폭**으로 기재한다.

**기대와 다른 패턴이 나오면**: 어떤 변형이 full보다 좋게 나오면(음수 하락), 해당 본문 문단을 "improves by"로 문장 자체를 고쳐야 하고(결과 확인 후 문장 확정 — 침묵 수정 금지), 그 component의 §3 논증과 §12 배치표를 재검토해야 한다. 특히 행 2가 무하락이면 "GRL의 순효과" 주장은 본문에서 유지 불가 — 그 경우 GRL 서사는 §3.5의 구조 논증(맥락 노출 경로 차단)을 정성 수준으로 하향하고, rebuttal 대비 권고 실험 R-PROBE(probing classifier — GRL의 표현 억제 직접 증거)의 우선순위가 올라간다. 데이터셋별로 하락폭이 크게 다른 것은 자연스러운 결과다(train AR이 높을수록 라벨 경로가 활성 — PSM에서 가장 큰 하락이 나오는 패턴이 설계 논리와 정합).

## 🧪 실험 내용과 설계

4행 확정(D-010 ②). 열 = 대표 3–4 데이터셋(NUM-020) + Avg. 지표 = PA%K-AUC F1 (best epoch, main과 동일 기준). 행별 소스와 실행 지침:

| 행 | 소스 | 실행 지침 |
|---|---|---|
| 1. Full model (CSMAD) | `[완주 대기/재사용]` [271c] | 대표 데이터셋 열은 이미 완주분(SWaT·PSM·WaDi)에서 추출 가능 |
| 2. w/o GRL (OD-excl. 유지) | `[신규 실행]` | **큐에 정확한 변형 부재** (exp290은 no_fm+no_grl 복합 — 행 2 정의와 불일치). 신규 큐 항목: 271 canon 기반 `use_grl=False` + **`anomaly_loss_weight=0.0` 추가로 anomaly-loss 경로 명시 차단**. 이유: `use_grl=False` 단독이면 `grl_disable_anomaly_loss` 게이트가 풀려 dead component인 dynamic-margin anomaly loss가 재활성화되어 비교가 오염된다 (§6.7 함정). 이렇게 OD-exclusion(정상 패치 전용 OD)을 유지한 "GRL 순효과" 변형을 만든다 |
| 3. w/o anomaly-priority masking | `[재사용]` **exp287_unmask** (`287_20260603_132835_unmask`) | `force_mask_anomaly=False` 단독 diff — metadata 실측 확인됨. 대표 데이터셋 분 완주 상태 — metadata 집계만. 참고(OBS-2): 큐 원항목 `config_override`에 `force_mask_anomaly` 키가 중복 기재(True→False, last-wins로 net False)되어 있었다 — 단독 diff는 실측으로 확정이나, **신규 큐 항목 작성 시 이 중복 키 패턴 답습 금지** |
| 4. w/o OD loss | `[신규 실행]` | 신규 큐 항목: `use_output_discrepancy=False`. **score 처리 방침 (코드 확정 사실 — r2 정정)**: 기본 동작은 **자동 recon-only** — `mae_anomaly/scoring.py:105-106`의 `resolve_score_weights`가 `use_output_discrepancy=False`면 `w_disc=0`을 강제하고, `scoring.py:249-253`에서 `w_disc=0` → `student_error=0` → score = Teacher recon만 남는다. 즉 별도 조치 없이 학습·추론 양쪽에서 OD가 일관 제거된다. **이 자동 recon-only 동작을 표 각주로 명시할 것.** disc 성분을 score에 남기는 변형을 원하는 경우에만 별도 채점 경로가 필요 — 침묵 변경 금지 |

집계 규칙: 각 행의 각 셀은 해당 run의 best epoch `metrics.pak_auc_f1`(SWaT excl22 열은 `excl22_pak_auc_f1` 기준 선정). Avg = 선택된 대표 데이터셋 열의 단순 평균. NUM-021/022/023은 행 1과 각 변형 행의 **Avg 열 차분**.

## 📊 구성과 형태

| Variant | Dataset-A | Dataset-B | Dataset-C | (Dataset-D) | Avg. |
|---|---|---|---|---|---|
| 1. Full model (CSMAD) | | | | | |
| 2. w/o GRL (OD-excl. retained) | | | | | |
| 3. w/o anomaly-priority masking | | | | | |
| 4. w/o OD loss | | | | | |

행 1이 기준선(최고치 기대), 변형 행은 하락폭이 드러나도록 Avg 열 포함. 강조는 통상 Full 행 bold 불필요 — Table 2와 달리 경쟁 비교 표가 아니라 분해 표이므로 (Phase 7 스타일 판단에 위임하되 일관 적용).

## 📝 캡션 (영문 확정본)

```latex
Ablation study. PA\%K-AUC F1 for each model variant on [3--4 representative datasets].
Row~2 (w/o GRL) removes the GRL classifier and reversal but retains the anomaly-patch
OD-loss exclusion, isolating the net effect of active adversarial suppression.
Extended variants (feature matching, Teacher-only warmup, symmetric decoder) are in
\ref{sec:extended_ablations} (Table~\ref{tab:extended_ablations}).
```

## ⚠️ 주의사항과 의존성

- ⓐ **대표 데이터셋 선정(NUM-020)**: 권장 SWaT excl22 + PSM(train AR 최대 — 라벨 경로 가장 활성) + WaDi A1 (+ WaDi A2 또는 SMD 대표 1). 단 **선택된 열은 행 1–4 전부와 부록 TAB-B4에서 글자 단위로 동일**해야 한다 (열 불일치 금지 — B4 캡션이 "the ablation datasets of Table 3"을 약속).
- ⓑ NUM-021/022/023이 이 표의 Avg 열 차분에서 파생.
- ⓒ 행 라벨은 "w/o anomaly-priority masking" — 내부 config명 `force_mask_anomaly`를 표에 노출하지 말 것.
- ⓓ 행 4의 자동 recon-only 각주 의무 (위 실행 지침 참조) — 각주 없이 게재하면 "OD를 학습에서 뺐는데 score에는 남아 있는가"라는 모호성이 생긴다.
- ⓔ FIG-B1·TAB-B2의 대표 데이터셋 선택도 이 표와의 통일이 권장되어 있다 — NUM-020 확정이 부록 설계의 입력값이 된다.

## 🔢 연결된 수치 placeholder

그룹 N-E 중 TAB-3 소스분 (NUM-024/025는 TAB-B4 소스 — appendix 담당 B2 페이지에서 명세):

| ID | 본문 위치 | 들어갈 값의 정의 |
|---|---|---|
| NUM-020 | §4.3 lead의 [N] + 캡션 "[3--4 representative datasets]" | ablation 대표 데이터셋 수 (설계 선택 3–4 — ⓐ 권고안 확정 시 결정; TAB-B4와 동일 집합) |
| NUM-021 | §4.3 "Anomaly-priority masking (Row 3)" 문단 | 행 1 − 행 3의 Avg 차 (w/o anomaly-priority masking 하락폭, 양수 기재) — 소스: [271c] − exp287 `[재사용]` |
| NUM-022 | §4.3 "Output discrepancy loss (Row 4)" 문단 | 행 1 − 행 4의 Avg 차 (w/o OD 하락폭, 양수 기재) — 소스: 행 4 `[신규 실행]` |
| NUM-023 | §4.3 "GRL adversarial suppression (Row 2)" 문단 | 행 1 − 행 2의 Avg 차 (GRL 순효과, 양수 기재) — 소스: 행 2 `[신규 실행]` |

부호 규약: 본문이 "removal costs X points" / "the drop is X" 형식이므로 양수 하락폭으로 기재 — 음수면 "improves by"로 문장 자체를 고친다 (결과 확인 후 문장 확정).

---

## 부록 (B1 범위 메모) — 커버리지 자가 점검

- **페이지 7장**: FIG-1 · FIG-2 · FIG-3 · FIG-4 · TAB-1 · TAB-2(+TAB-4 흡수 기록 포함) · TAB-3 — 본문 figure/table 전수.
- **NUM 28건 배속**: N-A {001,003,004,029} + N-B {002,005,030} + N-C {006–013} + N-D {014–019} → TAB-2 페이지 / N-E 중 {020–023} → TAB-3 페이지 / N-F {026,027} → FIG-3 페이지 / N-G {028} → FIG-4 페이지. (N-E 잔여 {024,025} = TAB-B4 소스, N-H {031} = TAB-B3 소스 — appendix 담당 B2 산출물에서 커버.)
- r2 명세의 실행 지침·소스 분류 라벨·캡션 원문·주의/의존성은 전 항목 보존·계승 확인.
