---
phase: 3
agent: outline-red-teamer
directives: [R1, R8, R9, R10, R11]
last_modified: 2026-06-11
verdict: CONDITIONAL_PASS_WITH_BLOCKERS
---

# Red-Team Review: PAPER_BLUEPRINT.md + PAGE_BUDGET.md
## Phase 3 — Adversarial Review

---

## 판정 요약

BLOCKER 3건, MAJOR 9건, MINOR 5건, NOTE 3건.

**BLOCKER-01** (설정-구현 간극 미봉): 50% prefix 프로토콜이 "evaluation fidelity" 프레이밍으로 서술되어 있으나, main 실험 자체가 train 구간 내 이상에 라벨이 전부 있는 upper-bound 케이스라는 사실을 reviewr는 "테스트 레이블 사용 금지"와 구분하지 못한다. 방어 서술이 look-ahead/leakage 지적을 아직 막지 못한다.

**BLOCKER-02** (contribution bullet 3번의 warmup): 비대칭 decoder + warmup을 "아키텍처 contribution"으로 별도 bullet에 올렸다. warmup ablation이 없고(REQUEST-F), SDMAE도 2단계 teacher-first 학습을 쓰기 때문에 이 bullet은 novelty가 없으면서 reviewer에게 ablation 요청 빌미를 제공한다.

**BLOCKER-03** (Q3 main table + clean-train 조건 없음): "비지도 baseline에게 가장 유리한 Q3"가 main table 단독이고, 전통적 clean-train 조건(표준 train/test split)은 아예 없다. 이 논문의 핵심 주장 중 하나는 "오염된 train 환경에서 우리 방법이 유리하다"인데, 오염되지 않은 동일 모델 성능을 비교하지 않으면 성능 향상이 방법론 때문인지 프로토콜 때문인지 분리 불가능하다.

Phase 5 진입 전 BLOCKER-01, BLOCKER-02, BLOCKER-03에 대한 서술 수준 결정이 필요하다.

---

## 1. 비판 항목 1: Novelty가 충분히 부각되는가 (R8)

### 공격 시나리오 1-A: "SDMAE가 이미 했다" — 이상 라벨 신호로 모델이 이상을 무시하도록 유도

SDMAE_DOSSIER §3.6-2, §4.1, §7-2가 이미 분석했듯이, SDMAE는 (i) 이상-제거 원본을 재구성 GT로 사용("overlook the anomalies"), (ii) anomaly map을 추가 타깃 채널로 주입, (iii) GT anomaly map을 손실 가중치에 가산한다. 이것이 ablation상 필수 component("mandatory to surpass 90% on Avenue")다.

블루프린트의 contribution bullet 2는 "labeled anomaly를 표현 학습의 기울기에 직접 개입"이라고 주장하지만, SDMAE도 (합성) 이상 라벨 신호를 학습에 직접 주입하여 "student가 이상을 표현하지 못하도록" 유도한다는 점에서 개념적 평행선이 존재한다.

현재 블루프린트의 방어 구조(§0.3 차별점 3축, §15 reviewer 방어 시나리오)는 "주입 계층(타깃/손실 공간 vs 기울기 공간)", "라벨 출처(합성 pseudo vs 실제 운영)", "작동 지점(출력 GT vs 내부 표현)"의 3축으로 방어한다. 이 3축 방어는 SDMAE_DOSSIER §7-2와 정합하며 reviewer rebuttal 수준에서는 충분하다. 그러나 §2.3 Related Work 본문에 이 방어 논리가 명시적으로 들어가 있지 않고 "자연스럽게 언급하고 넘어가는" R9 전략에만 의존한다.

**위험**: 만약 reviewer가 SDMAE §3.6-2를 알고 있다면 "이미 있다"는 공격이 들어올 때, §2.3의 계보 서술 1-2문장과 각주만으로는 방어가 부족하다.

**권고**: §3.5(GRL 소절) 서두에 "SDMAE's anomaly-overlook supervision operates at the target/loss space; our GRL operates at the gradient space of the student's internal representation"을 1문장으로 명시하라. Related Work 각주에 머물지 말고 Method 본문 안에 박아라.

**판정**: MAJOR-01

---

### 공격 시나리오 1-B: "end-to-end"라는 주장의 안전성

contribution bullet 1에서 "contaminated semi-supervised 설정의 최초 통합 솔루션"이라고 하고, bullet 2에서 "end-to-end 단일 모델"이라고 주장한다. NRdetector는 2-stage이므로 end-to-end 구분은 유효하다. 그러나 "최초"라는 주장에 대해 reviewer는 다음을 물을 것이다: WETAS, TreeMIL, DeepMIL도 학습 과정 전체가 단일 모델이다. 이들과의 end-to-end 차이가 구체적으로 무엇인지 §2.2에서 명확히 해야 한다.

현재 블루프린트는 WETAS/DeepMIL/TreeMIL을 "weakly supervised" 계열로 분류하고 §4.1.4 baselines에서 처리하는데, Related Work §2.2의 "last positioning sentence"가 "labeled anomaly를 표현 학습의 기울기에 직접 통합하는 end-to-end 첫 번째 다변량 TSAD 모델이다"라고 쓴다. DeepMIL은 end-to-end 학습이고 시계열은 아니지만, WETAS/TreeMIL은 시계열 도메인이다. WETAS/TreeMIL이 왜 "표현 학습 내부 통합"이 아닌지에 대한 논리가 §2.2에 명시되어야 한다.

**판정**: MAJOR-02

---

### 공격 시나리오 1-C: "contaminated benchmark protocol" 기여의 uniqueness

contribution bullet 1의 절반이 benchmark protocol이다. NRdetector도 "anomalies embedded within the training data" 문제를 동기로 삼고 7:3 split으로 train에 anomaly가 자연 포함된다. 차이는 우리가 의도적으로 test prefix를 train에 편입하는 것이다. 이 차이를 "protocol 설계"라는 기여로 부각하려면, 기존 벤치마크에서 이 설계가 없었다는 점을 구체적 인용으로 뒷받침해야 한다. 단순히 "기존 벤치마크는 clean-train을 가정한다"는 서술만으로는 "protocol이 새롭다"가 증명되지 않는다.

**판정**: MINOR-01

---

## 2. 비판 항목 2: Reviewer Reject 시뮬레이션 (8가지)

### R1: "50% test-prefix 편입은 test set leakage다"

**공격 논리**: test split의 앞 50%를 학습에 쓰면서 그 뒤 50%를 test로 쓴다. test anomaly label을 training에 노출한다 — 이것은 test set을 training에 쓰는 것이다.

**블루프린트 방어 현황 (§14)**: "시간 순서 보존(look-ahead 없음), 추론 시 test label 미사용, 비지도 baseline도 동일 데이터"로 방어한다.

**방어 갭**: "추론 시 label 미사용"은 label leakage에 대한 방어가 아니라 look-ahead에 대한 방어다. 핵심 공격은 다음이다: train 구간 내 anomaly label을 force_mask_anomaly와 GRL target으로 사용하는데, 이 label의 출처가 원본 test split의 앞 50%다. 즉 test 데이터의 ground-truth anomaly label로 모델을 학습시킨다 — 이것은 표준 ML 평가에서 leakage의 정의에 해당한다.

블루프린트가 이 공격에 대해 내세우는 논거는 "evaluation fidelity — 우리 방법의 기여를 평가하려면 labeled anomaly가 train에 있어야 한다"는 것이다. 이는 틀린 말이 아니지만, reviewer 입장에서는 "그 labeled anomaly가 왜 하필 test split에서 가져온 것이어야 하는가? 원본 train split에만 anomaly가 있는 데이터셋을 쓰면 되지 않는가?"라는 반문이 나온다.

**현재 상태**: §14의 5원칙이 이 반문에 정면으로 답하지 않는다.

**권고**: "원본 train split에 anomaly가 거의 없거나 없는 데이터셋(SWaT의 경우 anomaly가 test에만 집중)에서 contaminated semi-supervised 설정을 평가하려면 test prefix 편입이 구조적으로 필수"라는 설명을 §4.1.1 Protocol 방어 문단에 추가하라. 그리고 "이 prefix의 label은 모델이 추론할 새로운 unseen test data(뒤 50%)의 label과 분포가 다를 수 있다"는 것도 한계로 인정하라.

**판정**: BLOCKER-01

---

### R2: "비교가 공정하지 않다 — 비지도 baseline은 test prefix 편입으로 손해 본다"

**공격 논리**: Q3(normalonly) 조건에서 비지도 baseline의 train data는 test prefix 50%에서 anomaly를 제거한 버전이다. 제안 방법은 test prefix 50% 전체(anomaly 포함)를 train으로 쓴다. 즉 제안 방법이 더 많은 train data를 본다.

**블루프린트 방어 현황**: "비지도 baseline에게 labeled anomaly 제거로 가장 유리한 조건 제공"이라고 서술.

**방어 갭**: "오염원 제거"가 비지도에게 유리하다는 논리는 학습 데이터의 질(quality)에 관한 것이다. 그러나 학습 데이터의 양(quantity)에서는 여전히 비지도 baseline이 불리하다 — anomaly 구간 절제로 data가 줄어든다. 이 quantity 비대칭에 대한 방어가 없다.

**권고**: §4.1.4 또는 §4.1.1에 "비지도 baseline의 Q3 train data 크기는 제안 방법의 full contaminated train 대비 [X%] 수준이다. 이 량적 차이는 우리 방법의 이득을 overestimate할 수 있어, Appendix §A.2의 Q1 조건(전체 오염 데이터로 학습한 비지도 baseline)과 비교할 수 있다"는 한 문장을 추가하라.

**판정**: MAJOR-03

---

### R3: "best-epoch selection이 oracle이다 — test split으로 epoch 선정"

**블루프린트 서술 (§4.1.2)**: "Best epoch selected by pak_auc_f1 on the test split."

이것은 RESEARCH_SYNTHESIS §④에서 명시된 사실이다. 이 한 줄이 논문에 그대로 들어가면 reviewer는 즉시 reject 사유로 삼는다: test performance로 model selection하는 것은 test set leakage의 고전적 형태다.

**블루프린트의 방어 구조**: PA%K-AUC는 threshold sweep 적분이라 "oracle threshold가 없다"고 서술하지만, epoch selection 자체가 oracle 문제라는 점에 대한 방어가 없다.

**권고**: test split 기반 epoch selection 서술을 validation split 기반으로 바꾸거나, 이것이 구현상 불가피하다면 "this selection method is consistent across all methods" + "we observe [X] epochs variance has [Y] impact on final score" 형태의 sensitivity 방어를 §4.1.2에 포함하라. 현재 서술 그대로 Phase 5에 들어가면 확실한 reject 포인트가 된다.

**판정**: MAJOR-04

---

### R4: "warmup ablation이 없으면서 warmup을 아키텍처 기여로 bullet에 올린다"

contribution bullet 3: "The asymmetric Teacher(3L)–Student(2L) decoder structure, trained with **teacher-only warmup**, establishes a stable normal reconstruction reference..."

RESEARCH_SYNTHESIS §③ 표A 및 REQUEST-F에 명시된 대로, warmup ablation이 존재하지 않는다. 더구나 SDMAE도 Phase 1 teacher 단독 학습 + Phase 2 student 학습의 2단계 구조를 사용한다 (SDMAE_DOSSIER §3.4). SDMAE의 warmup 대응물을 우리의 아키텍처 contribution으로 bullet에 올리는 것은 SDMAE 유사성 위험과 ablation 부재 위험을 동시에 안는다.

블루프린트 §5.5 CRITICAL NOTE는 "warmup을 독립 기여로 주장하지 않고 학습 안정화 장치로만 서술"이라고 하지만, contribution bullet 3에는 "trained with teacher-only warmup"이 명시적으로 박혀 있다. 이 불일치가 Phase 5 drafter를 혼란에 빠뜨릴 것이다.

**판정**: BLOCKER-02
**권고**: contribution bullet 3에서 "teacher-only warmup"을 제거하거나 괄호 안 부연("(with a warm-up phase for training stability)")으로 격하하라. bullet 3의 novelty는 "asymmetric decoder capacity gap이 contaminated train 환경에서 discrepancy 신호를 안정화한다"는 구조 논리여야 한다.

---

### R5: "실험 수치가 없는 상태에서 'state-of-the-art' 주장"

이것은 EXPERIMENT_EXECUTION_TODO로 처리한다. 단, 주의사항: RESEARCH_SYNTHESIS §① MAJ-007에서 현재 완료된 entity는 37개(SWaT 2 + WaDi 2 + PSM 1 + SMD 22/28 + SMAP 5/54 + MSL 5/27)이다. Phase 5 진입 시점에 SMD 잔여 6 entity, SMAP 잔여 49 entity, MSL 잔여 22 entity, weakly-supervised 4종 전체가 미완료다. 논문 contribution bullet 4의 "six multivariate datasets"라는 주장과 "22 unsupervised baselines"라는 주장은 모두 완주 전까지 placeholder다.

**판정**: EXPERIMENT_EXECUTION_TODO (Phase 3 blocking 불요, Phase 5 진입 전 완주 조건 필수)

---

### R6: "GRL 기여가 불명확하다 — OD loss만으로도 같은 효과 아닌가?"

ablation 변형 목록에 "w/o GRL"과 "w/o OD Loss"가 별도 행으로 있다. 그런데 블루프린트의 R10 논증 배치표(§12)를 보면, GRL의 역할은 "student가 labeled anomaly 패턴을 기억해 잘 복원하면 discrepancy가 작아지는 문제를 gradient 부호 반전으로 해결"이라고 되어 있다. 이 논리 자체는 충분하다. 그런데 §3.5(C) 서술에서 "왜 OD loss의 단순 anomaly patch 제외(grl_disable_anomaly_loss)만으로는 부족하고 GRL이 추가로 필요한가?"에 대한 논거가 약하다.

GRL이 없어도 "anomaly 패치 OD loss = 0"은 student가 anomaly 구간에서 teacher를 따를 의무를 없애는 것이지, student가 anomaly 패턴을 표현에서 능동적으로 제거하는 것이 아니다. 이 차이가 §3.5에 명확히 서술되어야 한다. 현재 블루프린트는 이 차이를 설명하지만 "가장 경제적 메커니즘"이라는 주장만 있고 "유일하게 효과적인 메커니즘"이라는 논리가 없다.

**판정**: MAJOR-05
**권고**: §3.5(C)에 "w/o GRL but with anomaly OD disabled"와 "full model"의 차이를 ablation으로 보여야 한다. 현재 ablation Table 3의 변형 구성에서 이 행은 없다. "w/o GRL"이 이미 있지만, "w/o GRL"은 OD loss도 anomaly 포함(혹은 OD loss = 0)인지 명확히 정의되어야 한다.

---

### R7: "9개 이상 유형 중 어떤 유형에서 GRL/discrepancy가 효과적인가?" — §4.5 정성 분석의 약점

Fig. 4 (Qualitative)는 "두 성분이 각각 어떤 유형의 이상에 더 민감한지 1-2문장 해석"이라고만 되어 있다. 시계열 이상탐지 논문(Anomaly Transformer, DCdetector, Sub-Adjacent Transformer)의 관례는 정성 figure에서 이상 유형별 성공/실패 케이스를 명시한다. "어떤 유형"을 해석한다는 것이 실제 실험 결과에 근거해야 하지만, 현재는 placeholder다.

특히 이 논문은 표준 unsupervised 논문들이 쓰는 SWaT/WaDi 데이터셋과 겹치는데, SWaT excl22는 대부분의 anomaly가 1개 사건(#22)을 제외한 소형 사건들이므로 이것이 "정성 시각화에서 좋게 보이는 이상"인지 확인이 필요하다.

**판정**: MINOR-02 (EXPERIMENT_EXECUTION_TODO 조건부)

---

### R8: "데이터셋이 진부하다 — SWaT, WaDi, PSM, SMD는 2018-2021년 데이터"

CATCH(ICLR 2025)는 10 real-world + 6 synthetic, 총 22개 데이터셋을 사용한다. 이 논문이 SWaT/WaDi/PSM/SMD에 한정한 것은 데이터셋 다양성 측면에서 reviewer 비판을 받을 수 있다. SMAP/MSL은 추가되지만 이것도 2018년 데이터다.

블루프린트에서 Simulation과 Exathlon을 제외(R33)한 이유는 명시되어 있지 않다. Reviewer는 "왜 6개 데이터셋뿐인가? contaminated semi-supervised 설정을 더 다양한 도메인에서 검증하지 않는가?"라고 물을 것이다.

**판정**: MAJOR-06
**권고**: §4.1.1 데이터셋 서술에서 "6개 데이터셋 선택의 근거"를 1문장으로 명시하라. 예: "We focus on datasets with both training-time and test-time anomaly occurrences, enabling evaluation of the contaminated setting. These represent the most widely-used MTSAD benchmarks across industrial and infrastructure domains."

---

## 3. 비판 항목 3: MECE 위반 (R1)

### M1: contribution bullet 중복 — bullet 2와 bullet 3의 경계 모호

Bullet 2: "three orthogonal paths: masking priority, loss bifurcation, gradient reversal"
Bullet 3: "asymmetric Teacher(3L)–Student(2L) decoder structure"

Bullet 3의 asymmetric decoder는 bullet 2의 "discrepancy signal" 형성을 위한 구조적 기반이다. 독자 관점에서 bullet 2와 bullet 3은 같은 메커니즘의 두 측면처럼 읽힌다. "teacher-only warmup"을 bullet 3에서 제거하면 남는 내용(비대칭 decoder)은 bullet 2에 흡수 가능하다.

**권고**: bullet 3을 "아키텍처 기여"로 유지하려면 "비대칭 decoder capacity gap이 어떻게 contaminated semi-supervised 설정에서 discrepancy 신호의 신뢰성을 높이는가"라는 독립 논리가 있어야 한다. 이것을 명확히 하거나 bullet 2에 통합하라.

**판정**: MAJOR-07

---

### M2: §2.1과 §2.2의 경계 — WETAS/DeepMIL/TreeMIL 배치

블루프린트 §4.3은 WETAS/DeepMIL/TreeMIL을 "§2.1의 비지도 클러스터에 통합하거나, §2.2에서 1문장 언급 후 §4 baselines에서 이름+인용"이라고 한다. 이 "또는"이 문제다. WETAS/TreeMIL은 semi-supervised/PU 계열이므로 §2.2에 들어가야 하고, 이들이 §2.1에 들어가면 MECE가 깨진다.

더 큰 문제: §2.2에서 NRdetector를 "거의 유일한 선행 연구"로 부각하면서, WETAS/TreeMIL도 같은 소절에 있으면 "거의 유일"이라는 주장이 약해진다.

**권고**: WETAS/TreeMIL/DeepMIL은 §2.2에서 "이전 weakly-supervised 계열"로 1문장(괄호 인용)으로 처리하고, NRdetector를 "이 설정에서 가장 근접한 심층 학습 기반 선행 연구"로 분리하라. §2.1에는 절대 포함하지 말 것.

**판정**: MINOR-03

---

### M3: §4.2 Main Results와 §4.3 Ablation의 서사 중복

§4.2에 "왜 좋은지 이전 방법 한계와 연결" 구조가 있고, §4.3에도 "이것을 제거하면 왜 성능이 떨어지는가 → §3의 R10 논증과 직접 연결"이 있다. 두 소절 모두 "우리 방법의 각 component가 기여한다"는 같은 논리를 반복한다. 9p 제약에서 이 중복은 분량 낭비다.

**권고**: §4.2 분석 텍스트는 "전체 SOTA 비교"에 집중하고 component-level 기여 설명은 §4.3에만 배치하라. §4.2에서 "이전 방법 한계와 연결"하는 서술은 최소화하라.

**판정**: MINOR-04

---

## 4. 비판 항목 4: Derivative 인상 (R9)

### D1: 옵션 C + 각주 전략이 충분한가?

블루프린트 §2.3은 "In this work, we extend analogous self-distillation principles to the time-series domain, augmented with a contaminated semi-supervised framework..."라는 옵션 C 문구 초안을 제시한다.

이 문구의 "extend analogous ... principles"는 SDMAE를 직접 계보로 인정하는 표현이다. reviewer가 이것을 읽으면 "그럼 TSMAE = SDMAE + semi-supervised 확장인가?"라고 물을 수 있다.

**위험**: R9는 "자연스럽게 언급하고 넘어가는" 방식을 요구하는데, "extend analogous principles"는 오히려 계보를 강조한다. 이것이 과잉인지 아닌지는 Phase 5 drafter의 문장 완성도에 달려 있다. 블루프린트 단계에서 이 위험을 명시해야 한다.

**권고**: "extend ... principles"보다 "we apply the time-series counterpart of this design"이나 "we adapt this architectural paradigm"처럼 계보 계승을 암시하되 강조하지 않는 표현을 Phase 5에서 우선 시도하라. 핵심은 SDMAE를 이 논문의 "parent"가 아닌 "sibling in a family"로 포지셔닝하는 것이다.

SDMAE_DOSSIER §7-2의 방어 3축은 rebuttal 수준으로는 충분하지만, Related Work에서 차이를 "나열하지 않고" 이 3축을 전달하는 것이 Phase 5의 진짜 어려움이다. 블루프린트는 이 어려움을 "각주 1개"로 처리할 수 있다고 가정하는데, 각주만으로 3축 방어를 담기에는 지면이 부족하다.

**판정**: MAJOR-08

---

### D2: 모델명 후보 TS-SDMAE 위험

§10.1 모델명 후보에 "TS-SDMAE"가 포함되어 있다. 이 이름은 R9를 정면으로 위반한다. DECISION_LOG에서 제거 결정이 이루어졌는지 확인되지 않는다.

**권고**: TS-SDMAE를 후보 목록에서 즉시 제거하고 DECISION_LOG에 기록하라.

**판정**: MINOR-05

---

## 5. 비판 항목 5: PU/semi-supervised motivation 설득력 (R11)

### P1: "contaminated semi-supervised"라는 명명의 충돌 위험

블루프린트 §11 결정 ②의 사유는 타당하다. 그러나 이 명칭이 기존 "contaminated training" 문헌(이상탐지 분야에서 train 데이터에 anomaly가 포함된 상황을 지칭하는 일반 용어)과 혼동될 위험이 있다. "contaminated"는 문헌에서 주로 "나쁜 상황"(오염)을 뜻하는데, 이 논문에서는 "실제 운영 환경의 현실적 상황 + 우리가 적극 활용하는 자원"을 뜻한다. 이 의미 반전이 reviewer에게 혼란을 줄 수 있다.

추가로, "semi-supervised"는 보통 labeled data가 소수이고 unlabeled data가 다수인 상황을 의미하는데, main 실험(271)에서는 train 구간의 모든 샘플에 라벨이 존재한다. 이것은 "contaminated semi-supervised"가 아니라 "fully supervised contaminated train"에 가깝다.

**판정**: MAJOR-09
**권고**: §3.1 Problem Formulation에서 "main 실험은 label 가용성 상한 케이스이며, label sparsity sweep이 'semi-supervised' 가정의 일반 케이스를 검증한다"는 것을 명확히 쓰라 (이는 RESEARCH_SYNTHESIS §②-1/②-2/②-3 3단 구조를 논문에 반영하는 것이다). 이 구분 없이 "contaminated semi-supervised"를 명명만 하면 reviewer가 "main 실험이 semi-supervised 설정이 아니다"라고 reject할 수 있다.

---

### P2: Intro Para 3 논증 사슬의 비약

블루프린트 §3.1 Para 3의 핵심 관찰은: "labeled anomaly를 (a) 어느 위치를 주목해야 하는가, (b) 복원에서 무엇을 회피해서는 안 되는가, (c) 표현에서 무엇을 지워야 하는가로 동시에 활용하면 재구성 오차와 표현 불일치라는 두 신호를 모두 증폭할 수 있다"

이 논증은 (c)에서 갑자기 "표현에서 무엇을 지워야 하는가"로 도약한다. (a)는 force_mask_anomaly, (b)는 OD loss의 anomaly patch 제외에 대응하는데, (c)는 GRL adversarial suppression에 대응한다. (c)의 논리적 필연성("왜 표현에서 지워야 하는가?")이 Para 3에서 설명되지 않는다.

**권고**: Para 3에 "(b)만으로는 student가 anomaly 패턴을 기억해 잘 복원할 수 있고 그러면 discrepancy 신호가 약해진다 — (c)가 이 문제를 해결한다"는 bridge 문장이 필요하다.

**판정**: MAJOR-05와 연결 (중복 — 이미 MAJOR-05에서 다루었으나 서술 위치 문제로 MINOR-06으로 재분류)

---

## 6. 비판 항목 6: R10 논증 완성도

### R10-1: "논리 보강 필요 4건"의 블루프린트 처리

§12 R10 논증 배치표의 "논리 보강 필요 4건":
1. Linear patchify: patch_cnn ablation 없음 → "학습 효율과 구현 단순성"으로 방어 계획
2. Transformer encoder Pre-Norm: "시계열 transformer 논문 인용으로 보강" 계획
3. Teacher/student 층 수: ablation Table §B.1에서 정량화 계획
4. FM loss: ablation Table 3 행 5로 정량화 계획

항목 3과 4는 실험이 완료되면 해결된다(EXPERIMENT_EXECUTION_TODO). 항목 1과 2는 논리 수준의 보강이 필요하다.

항목 1(Linear patchify): "학습 효율과 구현 단순성"은 contribution 논증이 아니라 engineering trade-off 인정이다. Reviewer는 "왜 더 효과적인 CNN patchify를 쓰지 않는가?"라고 물을 것이다. 현재 방어는 "patch_cnn ablation 결과 없음"이 핵심 약점이다. 이것은 논문에서 "우리는 linear patchify가 더 낫다"고 주장하지 않고 "we use linear patchify following [MAE 원류]"로 서술하는 것으로 방어할 수 있다. 블루프린트는 §3.3에서 이미 이 경로를 택했다("최소한 CNN과 동등함을 ablation Table 3에서 정량화 목표. 미존재 시 '학습 효율과 구현 단순성' 논리로 방어"). 이것은 올바른 방향이다.

항목 2(Pre-Norm): "시계열 transformer 논문 인용으로 보강"은 전형적인 "기존 연구 인용으로 채우기" 패턴인데, 리뷰어 입장에서 "이 논문에서 Pre-Norm을 선택한 이유"가 명확해지지 않는다. 이 항목은 MINOR 수준이다.

**판정 (전체)**: EXPERIMENT_EXECUTION_TODO(3, 4번) + MINOR(1, 2번). Phase 3 blocking 불요.

---

## 7. 비판 항목 7: 분량 현실성 (R6)

### V1: §4 초과 문제

PAGE_BUDGET §4 Experiments 소절이 이미 −0.34p 초과로 계산되어 있다. 압축 전략이 5개 제시되지만, 이 중 "Table 2 landscape + fontsize small" (0.2p 절약)은 Elsevier elsarticle에서 landscape 처리가 가능한지 템플릿 수준에서 확인이 필요하다. 일부 Elsevier 저널은 landscape table을 지원하지 않고 "supplementary material"로 위임한다.

더 심각한 문제: §4.3 Ablation에 "변형 6(warmup)" 행이 placeholder로 들어간다. 이 행이 "to be updated"로 남아 있으면 Phase 5 drafter가 이것을 포함한 채 분량을 계산하게 된다. warmup 실험이 완료되지 않으면 이 행을 제거해야 하고, 그러면 ablation table의 논증 완결성이 약해진다.

**판정**: MAJOR-03과 연계, MAJOR-10으로 별도 등재

**권고**: §4.3 변형 목록에서 변형 6(warmup)을 Appendix §B.1로 이동하라. Ablation의 main table은 실험이 완료된 변형만 포함하라.

---

### V2: weakly-supervised 4종 미실행 (RESEARCH_SYNTHESIS FEEDBACK-2)

블루프린트 §4.1.4에 NRdetector를 포함한 weakly-supervised 4종이 baseline으로 등재되어 있다. RESEARCH_SYNTHESIS FEEDBACK-2에 따르면 이 4종은 GPU 전체 실험이 미실행이다. 이 baseline들의 결과 없이는 Main Table이 완성되지 않는다. 특히 NRdetector는 이 논문의 가장 직접적인 경쟁자다.

**판정**: EXPERIMENT_EXECUTION_TODO (Phase 3 blocking 불요, Phase 5 진입 전 필수)

---

### V3: Table 2의 열 구성이 확정되지 않음

블루프린트 §4.2: "열 = 데이터셋 × PA%K-AUC F1 (+ VUS-PR, 나머지는 Appendix Table로 위임 or 지면 허용 시 주 테이블 세분)"이라고 한다. "or 지면 허용 시"가 Phase 5 drafter에게 위임된 결정이다. Table 2의 열 수가 확정되지 않으면 PAGE_BUDGET의 "25행×8열" 추정이 근거 없다.

특히 이 논문은 5종 지표를 강조하는데, main table에 지표가 1개(PA%K-AUC F1)만 있으면 "왜 이 지표만 쓰는가?"라는 공격이 나온다. "지면 부족"은 방어가 되지 않는다.

**권고**: main table 열을 "PA%K-AUC F1 + VUS-PR 2개"로 고정하고, 나머지 3개 지표는 Appendix §A.3으로 위임하라. 이것이 지면-정보 trade-off를 명확히 한다.

**판정**: MINOR (PAGE_BUDGET 확정 전에 해소 필요)

---

## 8. 비판 항목 8: 결정 사안 6건의 타당성

### 결정 ①: C1-C4 재구성 → 4 bullet

타당하다. 결정 ①의 재설계는 RESEARCH_SYNTHESIS의 코드-직결 논리와 정합한다. 단, 위 BLOCKER-02(warmup in bullet 3)와 MAJOR-07(bullet 2 vs 3 경계)을 수정하면 4 bullet 구조는 유효하다.

### 결정 ②: "contaminated semi-supervised" 명명

조건부 타당. MAJOR-09의 주의사항(main 실험이 upper-bound 케이스임을 §3.1에서 명확히)이 해소되면 방어 가능하다.

### 결정 ③: SWaT excl22 수치 기준

타당하다. `A1A2_excl22` entity headline(0.62899) 사용이 fair하다는 사유는 RESEARCH_SYNTHESIS §④와 정합한다. "혼용 절대 금지" 원칙이 Phase 5 전체에서 지켜지면 문제없다.

### 결정 ④: Q3 단독 main table

조건부 타당하나 BLOCKER-03과 직결된다. Q3 단독 main table이 "비지도 baseline에게 가장 유리한 조건"이라는 서사는 성립하지만, 동일한 데이터셋에서 표준 clean-train split(즉 50% prefix 편입 없이 원본 train split만 사용)으로 학습했을 때의 제안 방법 성능이 없다. 이것이 없으면 "50% prefix 편입이 성능에 얼마나 기여하는가"를 reviewer가 묻을 때 답이 없다.

**판정**: BLOCKER-03
**근거**: 이 논문의 핵심 주장은 "contaminated semi-supervised 설정에서 labeled anomaly를 활용한다"이다. 그런데 비교는 항상 contaminated 설정(50% prefix 편입) 위에서 이루어진다. 제안 방법이 표준 clean-train 설정에서 얼마나 성능이 나오는지 보여주지 않으면, reviewer는 "성능 우위가 더 많은 데이터를 봐서(50% prefix)인지, GRL+distillation 방법론 때문인지 구분할 수 없다"고 reject할 것이다. 이 비교가 Appendix에라도 있어야 한다.

### 결정 ⑤: 옵션 C (SDMAE 처리) + 각주 전략

조건부 타당. 위 MAJOR-08의 언어 수준 주의사항이 해소되면 유효하다.

### 결정 ⑥: 코드 공개 조건부 포함

타당하다. RESEARCH_SYNTHESIS §⑦의 공개 전 checklist 해소 전에 URL 확정 불가라는 조건이 명시되어 있고, 이것이 Phase 5 진입 전 사용자 확인 사항으로 등재되어 있다.

---

## 9. 추가 발견: Phase 4/5 관련 경고

### PHASE4-001: DAGMM provenance 서술 미확정

블루프린트 §4.1.4는 "DAGMM (simplified variant, following [TranAD repo])"로 표기하고 각주에 "GMM energy 제거"를 명시한다고 한다. RESEARCH_SYNTHESIS §⑥ NOTE-003은 이것을 "방법 재정의"로 reject 사유가 될 수 있다고 경고하고 "Phase 3 시작 시점에 최우선 확정할 것"이라고 했다. 블루프린트는 이 경고를 반영했지만 확정 결정이 DECISION_LOG에 기록되었는지 확인이 안 된다.

**판정**: NOTE-01 — Phase 4 진입 전 DECISION_LOG 기록 필수

### PHASE4-002: baseline best-epoch 선정 조건의 불균형

블루프린트 §4.1.2: "Unsupervised baselines trained for 10 epochs; weakly-supervised for 50 epochs."
10 epoch은 제안 방법의 500 epoch과 비교할 때 극단적으로 짧다. 이것이 "각 방법의 best-effort hyperparameter"라는 서술과 어떻게 정합하는지 설명이 필요하다.

**판정**: NOTE-02 — Phase 5에서 "baseline epoch 설정 근거" 1문장 필수

### PHASE5-001: "최초" 주장이 들어갈 위치에서 스코핑 확인

블루프린트 §0.1, §3.1 Problem Formulation에서 "최초의 단일 모델"이라는 표현이 나온다. Phase 5 drafter가 이것을 그대로 draft에 넣으면 다음 주장이 문제다: "labeled anomaly를 표현 학습의 기울기에 직접 통합하는 end-to-end 첫 번째 다변량 TSAD 모델"

이 스코핑이 성립하려면 "심층 표현 학습 + PU/SSL + 다변량 TSAD + end-to-end + gradient-level integration"이 모두 동시에 성립해야 한다. 이 스코핑은 NRDETECTOR_DOSSIER §5 R20 서술 전략의 권고와 정합한다. 단, Phase 5에서 이 스코핑 외부(예: "최초의 semi-supervised TSAD")로 넘어가면 검증 공격에 취약해진다.

**판정**: NOTE-03 — Phase 5 drafter에게 스코핑 경계 명시 필수

---

## 발견사항 요약 (severity별)

| ID | Severity | 내용 | 수정 방향 |
|----|----------|------|-----------|
| BLOCKER-01 | BLOCKER | 50% prefix 프로토콜의 leakage 방어 갭 — "test split의 ground-truth label로 학습" 공격에 대한 정면 방어 없음 | §4.1.1에 "원본 train에 anomaly가 없는 데이터셋 구조 + 이 prefix label이 새로운 test data label과 다르다"는 인정 추가 |
| BLOCKER-02 | BLOCKER | contribution bullet 3에 warmup이 명시됨 — warmup ablation 없고 SDMAE도 teacher-first warmup 사용 | bullet 3에서 warmup 제거 또는 괄호 내 부연으로 격하; §5.5 CRITICAL NOTE와 일관성 확보 |
| BLOCKER-03 | BLOCKER | Q3 단독 main table에 표준 clean-train 조건 비교 없음 — 방법론 vs 프로토콜 효과 분리 불가 | Appendix에 "제안 방법 + 원본 train split(prefix 없음)" 조건 결과 추가 |
| MAJOR-01 | MAJOR | §2.3 각주로만 방어되는 SDMAE anomaly-overlook 개념적 평행 — §3.5에 직접 서술 없음 | §3.5(GRL 소절) 서두에 1문장 명시 |
| MAJOR-02 | MAJOR | WETAS/TreeMIL이 end-to-end인데 "end-to-end 최초" 주장의 근거 미흡 | §2.2에서 WETAS/TreeMIL이 "표현 학습 내부 통합"이 아닌 이유 명시 |
| MAJOR-03 | MAJOR | Q3에서 비지도 baseline이 데이터 양(quantity)에서도 불리하다는 점 미인정 | §4.1.4에 데이터 량적 비대칭 인정 + Q1 비교로 보완 가능성 언급 |
| MAJOR-04 | MAJOR | test split으로 best-epoch 선정 — oracle 공격에 무방비 | validation split 기반 epoch selection 또는 sensitivity 방어 |
| MAJOR-05 | MAJOR | GRL vs "anomaly OD disabled만"의 차이를 ablation으로 보여야 한다 | ablation Table 3 변형 구성 보완; "w/o GRL" 정의 명확화 |
| MAJOR-06 | MAJOR | 6개 데이터셋 선택 근거 없음 | §4.1.1에 데이터셋 선택 근거 1문장 |
| MAJOR-07 | MAJOR | contribution bullet 2와 3의 경계 모호 — asymmetric decoder가 bullet 2에 흡수 가능 | bullet 3의 독립 novelty 명확화 또는 bullet 2에 통합 |
| MAJOR-08 | MAJOR | 옵션 C 문구 "extend analogous principles"가 SDMAE derivative 인상 강화 위험 | "adapt"/"apply counterpart" 등으로 완화; 각주 내용을 Method 본문으로 이동 고려 |
| MAJOR-09 | MAJOR | "contaminated semi-supervised" 명명 — main 실험이 upper-bound 케이스임을 §3.1에서 명확히 안 하면 "semi-supervised가 아니다" reject 위험 | §3.1에 RESEARCH_SYNTHESIS ②-1/②-2/②-3 3단 구조 반영 |
| MAJOR-10 | MAJOR | ablation Table 3 변형 6(warmup) placeholder — 미완료 실험 행이 분량 계산에 포함됨 | 변형 6을 Appendix §B.1로 이동; main ablation table은 완료된 변형만 |
| MINOR-01 | MINOR | contaminated benchmark protocol 기여의 uniqueness를 기존 문헌 인용으로 뒷받침 필요 | §4.1.1에 "기존 TSAD 벤치마크는 clean-train 가정" 문헌 1-2개 인용 |
| MINOR-02 | MINOR | §4.5 정성 분석의 "어떤 유형에서" 해석이 placeholder | EXPERIMENT_EXECUTION_TODO — 수치 확정 후 이상 유형별 근거 필요 |
| MINOR-03 | MINOR | WETAS/DeepMIL/TreeMIL의 §2.1 vs §2.2 배치 "또는" 모호성 | §2.2에만 배치하고 §2.1에는 절대 포함 불가로 확정 |
| MINOR-04 | MINOR | §4.2와 §4.3의 "왜 좋은지" 논리 중복 | §4.2는 전체 SOTA 집중, component 논리는 §4.3 전용 |
| MINOR-05 | MINOR | 모델명 후보 TS-SDMAE가 목록에 잔존 | 즉시 제거 + DECISION_LOG 기록 |
| NOTE-01 | NOTE | DAGMM provenance 확정이 DECISION_LOG에 기록되지 않았을 가능성 | Phase 4 진입 전 DECISION_LOG 확인 |
| NOTE-02 | NOTE | baseline 10 epoch vs 우리 500 epoch 비대칭의 근거 서술 없음 | Phase 5에서 1문장 추가 |
| NOTE-03 | NOTE | "최초" 스코핑 경계를 Phase 5 drafter에게 명시 필요 | Phase 5 지침에 스코핑 범위 박아두기 |
