---
phase: 2
agent: venue-scout
directives: [T2]
last_modified: 2026-06-11
revision: r2 (fixer — adversarial review paper/99_reviews/p2_venue_corpus_r1.md 반영: S-001 MINOR + S-002 NOTE + V-004 연계; fixlog: p2_fixlog_r2.md; 정정 이력은 말미 부록)
---

> **경고**: 아래 verbatim 인용문(따옴표 처리된 것)은 분석 전용 — 논문 본문으로 복사 금지 (A2)
> **서지 단서**: 서지 정보는 "Phase 4 공식 소스 재검증 필요". 수치·수식은 각 논문 원문 기준이며 재사용 전 출처 명기 필수.

---

# STRUCTURE AND FIGURE PATTERNS — Phase 3 블루프린트용 실행 패턴 모음

이 문서는 Phase 3 (논문 블루프린트)에서 **바로 쓸 수 있는** 섹션 구조·논리 흐름·figure/table 설계 패턴을 정리한다. 논문별 사례와 공통 패턴을 분리하여 제시한다.

---

## A. 섹션 구조 패턴

### A.1 학회 논문 (8–10p 기준) — 공통 패턴

분석 대상: Anomaly Transformer(ICLR 2022), DCdetector(KDD 2023), Sub-Adjacent Transformer(IJCAI 2024), CATCH(ICLR 2025)

**전형적 섹션 순서 (9p 본문 기준)**:
```
1 Introduction          2.0–2.5p
2 Related Work          1.0–1.5p
3 Method/Methodology    2.0–3.0p
4 Experiments           2.5–3.5p
5 Conclusion            0.5p
References              ~0.5–1p
Appendix (선택)         2–8p
```

**분량 비중 특징**:
- Introduction이 전체의 약 22–28%
- Method와 Experiments가 합쳐서 약 55–65%
- Related Work는 10–15% (짧게 유지)
- Conclusion은 단일 단락 수준 (0.3–0.5p)

**KDD vs ICLR 차이**:
| 항목 | KDD (DCdetector) | ICLR (Anomaly Transformer) |
|-----|-----------------|--------------------------|
| 총 페이지 | 17p (2단 ACM 스타일) | 9p 본문 + 4p 부록 |
| Related Work 위치 | §2 (독립, 별도 소절) | §2 (독립) |
| Background 절 | 없음 | 없음 |
| Appendix 정책 | 부록에 hyperparameter 상세 | 부록에 알고리즘 의사코드 + 추가 시각화 |
| 비교 baseline 수 | 26개 | 18개 |

**IJCAI Sub-Adjacent Transformer의 특이점**:
- Related Work를 2개 소절로 압축 (TSAD + Linear Attention만) — 주제 직접 연결 전략
- Ablation 5개를 Experiments 하위 소절로 나란히 배치 (ablation만 1페이지 이상)

---

### A.2 Elsevier 저널 논문 (9–15p 기준) — 관례 패턴

분석 대상: DTAAD(Knowledge-Based Systems 2024), elsarticle 템플릿 관찰

**전형적 섹션 순서 (9p 타깃 기준, elsarticle)**:
```
Abstract             (150–250 words, 자유형식 또는 구조화)
Keywords             (5–8개)
1 Introduction       1.5–2.5p
2 Related Work       1.0–2.0p
3 Background/        0.5–1.0p  [선택: 문제 정의, 예비 정리]
  Preliminaries
4 Methodology        2.0–3.5p
5 Experiments        2.0–4.0p
6 Conclusion         0.5–1.0p
Acknowledgments
References
```

**Elsevier 저널 특유 요소**:
- Highlights (3–5 bullet point, 각 ≤125 chars) — 목차 앞에 위치
- Graphical Abstract (선택적 — 방법 overview 1장짜리 figure)
- "Declaration of Competing Interest" 필수 포함
- 수식 번호 연속적, 위치는 오른쪽 정렬
- 표: 내부 세로선 없이 가로 3선 (booktabs 스타일)
- Figure caption은 figure 아래, table caption은 table 위 (elsarticle 기본값 — 표준적이며 대부분 학회와 동일; fixer r2, S-002 정정: 초판의 "학회와 반대인 경우 있음"은 검증 없는 일반화였음)

**9페이지 타깃에서의 분량 조정 전략**:
- Background/Preliminaries를 짧게 또는 Introduction 안으로 통합 가능
- Related Work를 Introduction §1.2 소절로 통합하면 1p 절약 가능
- Appendix는 Elsevier 저널에서 필수 아님 — 간략화 우선

---

## B. Introduction 논증 전개 패턴

### B.1 공통 4단 구조 (모든 분석 논문에서 확인)

Phase 3에서 바로 사용 가능한 단락 골격:

```
[Para 1] 문제의 중요성·응용 배경 (motivation)
         → "Real-world systems... anomalies... financial loss / safety"

[Para 2] 기존 방법 계보 정리 + 각 계열의 구조적 한계
         → 밀도추정/재구성/예측/대조학습 방법 리뷰 → 공통 약점

[Para 3] 본 논문의 핵심 관찰/통찰 (key observation / insight)
         → "We observe that..."
         → 이것이 왜 기존 방법으로는 해결 안 되는지

[Para 4] 제안 방법 개요 + Contribution 목록
         → "In this paper, we propose..." + bullet 3–4개
```

### B.2 논문별 Para 3 (통찰) 전개 방식 비교

| 논문 | Para 3 핵심 관찰 | 관찰→설계의 논리 연결 |
|-----|----------------|-------------------|
| Anomaly Transformer | "이상은 드물어 전체 시리즈와 비자명한 association을 형성하기 어렵고, 인접 시점에 집중된다" | → prior-association (Gaussian kernel) + series-association 분기 → AssocDis 기준 |
| DCdetector | "정상 시점은 다른 관점에서도 강한 상관을 가지지만, 이상은 그렇지 않다 (permutation invariant)" | → contrastive dual-branch → 재구성 없이 대조 손실만 사용 |
| SDMAE | "teacher와 student는 정상에서 유사, 이상에서 발산 → discrepancy가 이상 신호" | → shared-encoder + asymmetric decoder → discrepancy 이상 점수 |
| CATCH | "주파수 밴드별로 채널 상관 패턴이 다르다 → 고정 채널 전략은 불충분" | → frequency patching + adaptive channel fusion (bi-level 최적화) |
| Sub-Adjacent Transformer | "이상은 즉각 인접(immediate vicinity)보다 sub-adjacent 구간과 더 차이가 난다" | → sub-adjacent attention → 재구성 오차 보강 |

**공통 패턴 추출**: 관찰은 반드시 "정상 vs 이상의 행동 차이"를 서술하고, 그 차이를 증폭/측정하는 메커니즘으로 설계를 이어간다. 빈 claim ("기존 방법은 부족하다")보다 구체적 현상 묘사가 설득력 강함.

### B.3 Contribution 제시 방식

**공통 포맷**: 3–4개 bullet point, 동사로 시작 ("We propose / design / demonstrate")

**강도 표현 패턴**:
- 아키텍처 기여: "We propose/design [모델명], which..."
- 이론적 기여: "We formulate / derive..."
- 실험적 기여: "Extensive experiments demonstrate / show that..."
- 평가 기여: "We achieve state-of-the-art on [N] benchmarks..."

**분량**: bullet당 1–3문장, 전체 contribution 목록 0.5p 미만

**DCdetector 예시 (verbatim 요약, 인용 아님)**:
- Architecture bullet: dual-branch attention → permutation invariant representation
- Optimization bullet: pure contrastive loss without reconstruction
- Performance bullet: SOTA on 7 multivariate + 1 univariate datasets

---

## C. Related Work 조직 패턴

### C.1 소절 구성 방식

| 논문 | 소절 수 | 구성 원리 |
|-----|--------|---------|
| Anomaly Transformer | 2 | (1) Unsupervised TSAD [4 계열] (2) Transformers for TS |
| DCdetector | 2 | (1) TSAD [통계/ML/DL 전반] (2) Contrastive Representation Learning |
| CATCH | 3 | (1) MTSAD 개요 (2) Channel Strategies (3) Frequency Domain Analysis |
| Sub-Adjacent Transformer | 2 | (1) TSAD [재구성/예측/비유사도] (2) Linear Attention |
| NRdetector | 3 | (1) TSAD (2) Learning with Noisy Labels (3) PU Learning |
| DTAAD | 2 | (1) Deep Unsupervised & Weakly Supervised AD (2) Transformers for AD |

**설계 원리**:
1. 첫 소절은 항상 "우리 문제 도메인의 기존 방법 계보" — 이를 통해 우리 방법이 포지셔닝될 공간을 만든다
2. 마지막 소절은 "우리가 쓰는 핵심 기술의 배경" — 기술 선택의 정당화
3. 중간 소절은 "우리가 해결하는 구체적 한계의 문헌"

**Positioning 문장 위치**: 항상 각 소절의 마지막 또는 Related Work 섹션 끝에 "본 논문과의 차이"를 1–2문장으로 명시한다. Anomaly Transformer의 예:
> "This paper is characterized by a new association-based criterion. Different from the random walk and subsequence-based methods..."

**NRdetector 관찰**: baseline으로 사용되는 비지도 모델들은 related work에 개별 논의 없이 괄호 클러스터 인용(10개+)으로 일괄 처리 → 설정·문제 정의를 공유하는 논문만 명제 단위 논의. Phase 5 작성 지침 직결.

### C.2 관련 연구 텍스트 밀도

- 소절당 2–4 단락, 단락당 3–6 문장
- 각 단락은 "계열 정의 → 대표 사례(괄호 인용) → 해당 계열의 한계" 구조
- 자기 방법과의 비교는 최소화 (Related Work에서 우리 방법 자랑은 금지, 포지셔닝만)

---

## D. Method 서술 패턴

### D.1 표준 Method 섹션 구조

```
[문제 정의 / Notation 도입]     반드시 먼저 — 수식 1개~3개
[전체 아키텍처 개요]            Figure + 1–2 단락 산문
[컴포넌트 1: 핵심 모듈]         소절 하나씩
[컴포넌트 2: ...]
[...N]
[손실 함수]                     수식 + 설명
[추론/이상 점수]                 수식 + 설명
```

### D.2 논문별 Method 소절 구성 비교

| 논문 | Method 소절 수 | 핵심 구조 |
|-----|--------------|---------|
| Anomaly Transformer | 2 (Architecture + Minimax Learning) | 아키텍처 → 학습 전략 → 이상 점수 |
| DCdetector | 4 (Overall / Dual Attention / Repr Discrepancy / Anomaly Criterion) | 아키텍처 overview → 핵심 모듈 2개 → 추론 |
| CATCH | 5 (Overview / CFM / TFR Module / Bi-level / Anomaly Scoring) | 전체 → 모듈 3개 → 최적화 → 점수 |
| Sub-Adjacent Transformer | 3 (Problem / Sub-Adjacent / Loss & Score) | 정의 → 핵심 attention → 학습+추론 |
| DTAAD (Elsevier) | 4 (Framework / Training / AD & Diagnosis + Background 별도) | 배경지식 먼저, 그다음 방법 |

**Notation 도입 패턴**:
- 가장 첫 번째 수식: "We denote the multivariate time series as X ∈ ℝ^{T×d}..."
- 기호는 최초 등장 시 정의, 이후 재정의 없음
- 핵심 기호는 Table이나 Glossary로 정리하는 경우도 있음 (DTAAD)

**Architecture Overview Figure** (§D.4에서 상세 다룸): 항상 Method 첫 소절이나 시작 부분에 배치.

**D.3 손실 함수 제시 패턴**

수식 구조:
```
L_total = L_main + λ · L_auxiliary
```
- 각 항을 먼저 개별 소절에서 정의한 후 최종 total loss 수식으로 통합
- λ 같은 trade-off 파라미터는 "when λ=0, the model reduces to..." 형태로 특수 케이스 설명
- Anomaly Transformer: Minimax 2단계 수식을 별도 소절에서 명시적으로 설명

---

## E. 실험 설계 서술 패턴

### E.1 표준 Experiments 섹션 구조

```
[데이터셋]              통계 테이블 + 1–2 단락
[구현 세부 / 설정]      하이퍼파라미터, optimizer, GPU, 시드
[비교 대상 (Baselines)] 계층별 분류 + 각 1문장 설명
[주 결과]               메인 성능 비교 테이블 + 분석 텍스트
[Ablation 연구]         컴포넌트 제거 실험 테이블 + 해석
[추가 분석]             파라미터 민감도 / 효율성 / 시각화
```

### E.2 데이터셋 서술 방식

**공통 패턴**: "Dataset X is a [크기/출처/특성] dataset. It contains [차원] dimensions and [시간 길이] time steps."

Anomaly Transformer의 데이터셋 서술 예시 구조 (verbatim 아님, 패턴 요약):
- 각 데이터셋을 1–2 문장으로 서술
- 출처·규모·응용 도메인을 포함
- 세부 통계는 Appendix 테이블로 위임 ("details in Table X of Appendix")

### E.3 Baseline 계층화 패턴

모든 분석 논문에서 baseline을 계층별로 분류하여 제시:

| 계층 | 예시 | 논거 |
|-----|------|------|
| 고전 통계 방법 | LOF, OCSVM, IsolationForest | 딥러닝 대비 열위 입증 |
| 딥러닝 재구성 기반 | LSTM-VAE, OmniAnomaly, DAGMM | 주요 경쟁자 |
| 딥러닝 예측/연관 기반 | Anomaly Transformer, VAR | 최근 방법 |
| 자기지도/대조학습 기반 | DCdetector, PatchAD | 동류 방법 |

**NRdetector 관찰**: baseline을 3계층(unsupervised / semi-supervised 변형 / weakly supervised)으로 나누고, "주 경쟁자"를 명시 선언 → 독자가 어느 결과에 집중해야 하는지 안내. 계층 간 공정성 담보 설명도 포함.

### E.4 메인 결과 테이블 서술 패턴

**테이블 후 분석 텍스트 구조**:
1. "As shown in Table X, our method achieves..." [전체 요약 1문장]
2. [각 데이터셋별 특이 결과 언급 1–3개]
3. [왜 좋은지 brief 설명 — 이전 분석/insight와 연결]
4. [한계 인정 또는 특수 케이스 설명] (선택)

**DCdetector 실험 텍스트 패턴**: 메인 테이블 서술 후 "It is worth mentioning that..." 형태로 평가 지표 논쟁 (PA vs 멀티지표)을 명시적으로 언급 → 공정성 주장.

### E.5 Ablation 서술 패턴

**공통 구조**: 각 컴포넌트를 하나씩 제거("w/o X") → 성능 저하 측정 → 기여 정량화

**Anomaly Transformer Ablation 텍스트 패턴** (verbatim 아님, 패턴):
- "The association-based criterion brings a remarkable XX% averaged absolute F1 promotion (from YY to ZZ)"
- 수치를 직접 차이로 표현 (화살표 형식: "79.05→94.96")

**CATCH Ablation 조직**: 채널 전략 / 최적화 목표 / 패칭 / 점수 기법 / bi-level 최적화 5개 축을 별도 행으로 → 단일 테이블에 모두 포함

---

## F. Figure/Table 유형 정리

### F.1 Architecture Diagram (아키텍처 다이어그램)

**위치**: Method 섹션 시작부 (대부분 §3 첫 Figure)
**크기**: full-width (단열 논문) 또는 full 2-column (2단 논문), 일반적으로 column-spanning
**내용 구성**:
- 입력 → 인코더 → (중간 모듈) → 디코더 → 출력 흐름
- 색상 블록으로 모듈 구분 (보통 4–6가지 색)
- 화살표로 데이터 흐름 표시
- 소속 수식과 연결하는 레이블 (수식 번호 또는 기호)

**논문별 스타일**:
| 논문 | 다이어그램 스타일 | 특징 |
|-----|----------------|-----|
| Anomaly Transformer Fig.1 | 좌측 Anomaly-Attention 상세 + 우측 전체 스택 | stop-grad 화살표 명시 |
| DCdetector Fig.2 | 3단 계층(a Backbone / b 대조구조 / c 이중 Attention) 분리 | 컴포넌트 4개를 색으로 구분 |
| DCdetector Fig.1 | 3-way 아키텍처 비교 다이어그램 | 경쟁 방법과의 차이를 1장으로 |
| CATCH Fig.2 | 3개 모듈(Forward/CFM/TFR) 병렬 배치 | 입력→FFT→패치→채널 fusion→재구성 |
| SDMAE Fig.1 | 학습 2단계(Phase 1/2) 구분 | 합성 이상 overlay 포함 |

**Phase 5 적용 지침**: TSMAE 아키텍처 다이어그램은 (1) Patchify 모듈 (2) Shared Encoder (3) Teacher Decoder (4) Student Decoder (5) GRL + AnomalyClassifierHead 5개 컴포넌트를 포함해야 함. 학습(force_mask, GRL 활성화)과 추론(GRL 비활성화, score 계산) 두 단계를 별도 패널로 표현하는 방식 고려.

---

### F.2 메인 성능 비교 테이블 (Main Results Table)

**위치**: Experiments 섹션 첫 번째 테이블 (대개 Table 1)
**크기**: full-width, 열이 많을 경우 sideways 또는 fontsize 축소
**내용 구성**:
- 행: 비교 방법 (계층별 정렬, 우리 방법 맨 아래)
- 열: 데이터셋 × 지표 (P/R/F1 또는 AUC/F1 등)
- Bold: 최고값, Underline: 2위 (KDD/ICLR 관례)
- "Ours" 또는 방법 약칭으로 우리 행 표기

**논문별 규모**:
| 논문 | 행(baselines) | 열(datasets×metrics) | 총 셀 수 |
|-----|-------------|---------------------|--------|
| Anomaly Transformer | 18+1 | 5ds×3 = 15 | ~285 |
| DCdetector | 21+1 | 5ds×3 = 15 | ~330 |
| CATCH | 15+1 | 10+6=16 ds × 2 | ~512 |
| NRdetector | 13+1 | 5ds×2 = 10 | ~140 |

> **정정 (fixer r2, S-001)**: CATCH 행 초판 "12+6=18 ds × 2"는 오기. arXiv HTML(2410.12261v3) Table 2 캡션 직접 재확인: "Average A-R (AUC-ROC) and Aff-F (Affiliated-F1) accuracy measures for **10 real-world datasets and 6 synthetic datasets** of different types of anomalies" — 즉 메인 테이블은 10 real + 6 synthetic(이상 유형별) = 16 데이터셋 항목 × 2지표 (실제 지면 배치는 방법이 열, 데이터셋이 행인 전치 구조). abstract의 총 실험 커버리지 22개(10 real + 12 synthetic)와 메인 테이블 항목 수는 다르다.

**캡션 스타일**:
- "Table 1: [방법명/논문 제목] on [데이터셋 목록]. P, R, F1 represent precision, recall, and F1-score... Bold indicates best performance."
- 지표 약어 정의를 캡션에 포함하는 것이 관례

---

### F.3 Ablation 테이블

**위치**: Main Results 뒤, Table 2 또는 Table 3
**크기**: 보통 half-width 또는 column 내
**내용**:
- 행: 모델 변형 (w/o X, w/o Y, full model)
- 열: 주요 데이터셋 3–5개 × F1 (또는 대표 지표)
- Bold: full model 또는 최고값

**구성 원리**: 각 컴포넌트를 하나씩만 제거. 상호작용 ablation (X AND Y 제거)은 선택적.

**Anomaly Transformer 특이 패턴**: Ablation 테이블에 "method variants"를 행으로 나열 (Recon / AssDis / Assoc, 세 가지 anomaly criterion × 두 가지 prior-association 방법 × 두 가지 optimization strategy = 6행). 팩토리얼 조합 방식의 ablation.

---

### F.4 Anomaly Score 시각화 (Qualitative Figure)

**위치**: Experiments §4.2 Model Analysis 또는 부록
**크기**: full-width, 2–3행 × 5열 (행: input TS / 점수1 / 점수2; 열: anomaly type별)
**내용 구성**:
- 행 1: 원본 시계열 (이상 위치 빨간 마킹)
- 행 2: 기준선 anomaly score
- 행 3: 우리 방법 anomaly score
- 붉은 점선 = 탐지 임계값
- 5가지 이상 유형 전부 커버 (point-global / contextual / pattern-shapelet / seasonal / trend)

**논문별 스타일**:
| 논문 | Figure 번호 | 행 구성 | 열 수 |
|-----|------------|--------|------|
| Anomaly Transformer Fig.5 | Fig.5 | Input / Reconstruction criterion / Association criterion | 5 (anomaly types) |
| DCdetector Fig.5 | Fig.5 | Input / AnomalyTrans score / DCdetector score | 5 |
| Sub-Adjacent Transformer Fig.5 | Fig.5 | Input TS + intermediate variables | 5 |

**캡션 패턴**: "Visualization of [criterion/score] for [데이터셋]. Anomalies are labeled by red circles/segments. The failure cases of baselines are bounded by red boxes."

**Phase 5 적용 지침**: TSMAE의 정성 시각화는 최소 (a) SWaT, (b) WaDi A1 또는 PSM에서 선택하고, 행 구성은 (1) 입력 시계열 (2) Teacher 재구성 오차 (3) Teacher-Student discrepancy (4) 최종 합산 점수 4행 형태가 이상적.

---

### F.5 파라미터 민감도 Plot (Sensitivity Plot)

**위치**: Experiments 끝 부분 또는 부록
**크기**: full-width 가로 배열 (4–5 subfigure)
**내용**: X축 = 하이퍼파라미터 값, Y축 = F1 또는 AUC, 선이 여러 데이터셋에 대해 오버레이

**공통 하이퍼파라미터 축**:
- Window size (all papers)
- d_model / hidden dimension (DCdetector, CATCH)
- Encoder layer / decoder layer 수
- Attention head 수
- Loss weight (λ) — Anomaly Transformer

**캡션 패턴**: "Parameter sensitivity for [parameter name]. The model is stable when [범위] and achieves best performance at [최적값]."

---

### F.6 ROC/PR 곡선

**위치**: Experiments 또는 부록
**크기**: 보통 half-width 또는 row of N subfigures
**내용**: 여러 데이터셋별 ROC 곡선을 subplot으로 나열

Anomaly Transformer Fig.3: 5개 subplot (SMD, MSL, SMAP, SWaT, PSM) × 4개 방법 비교선

**Phase 5 적용 지침**: VUS-ROC는 단순 AUC-ROC와 다름 — PA%K-AUC PR이 우리 타깃 지표이므로 curve 기반 figure는 VUS-PR 형태가 적절. 단, 기존 논문들이 주로 F1+PA를 사용하므로 우리 지표 체계를 captioned으로 설명.

---

### F.7 비교 다이어그램 (Comparison Diagram)

**위치**: Introduction 또는 Related Work/Method 시작
**크기**: full-width, 2–3 패널

**전형적 용도**:
- (a) 기존 방법 계보 흐름 vs (b) 우리 방법의 차별점
- DCdetector Fig.1: 재구성 기반 / Anomaly Transformer / DCdetector 3-way 비교 — 한 장으로 차이 전달
- CATCH Table 1: 기존 방법들의 속성 비교 테이블 (5개 속성 체크마크)

**Phase 5 적용 지침**: "비지도 방법 vs 약지도 vs 우리 semi-supervised/PU" 3-way 비교 다이어그램 또는 테이블 (방법군별 label 활용 방식, 설정 비교)이 Introduction 또는 Related Work에 효과적.

---

### F.8 데이터셋 통계 테이블 (Dataset Statistics Table)

**위치**: Experiments 첫 번째 또는 Appendix
**내용 열**: Dataset name | #Train | #Test | #Dimensions | Anomaly Ratio (%) | 출처/응용

**공통 패턴**: 이상 비율(AR)을 명시하는 것이 최근 관례 (NRdetector, CATCH, Sub-Adjacent Transformer 모두 포함)

---

### F.9 효율성/복잡도 비교 (Efficiency Figure)

**위치**: Experiments 분석 절 또는 부록
**형태**: scatter plot (X: FPS or latency, Y: AUC/F1) 또는 bar chart (파라미터 수, FLOPs)

SDMAE Fig.2: 성능(AUC) vs 속도(FPS) scatter — "우리 방법이 Pareto-optimal에 가깝다"는 주장 시각화

DCdetector Fig.7: GPU 메모리 + 실행 시간 bar chart (d_model 크기별)

---

### F.10 Critical Difference Diagram (CD Diagram)

**사용 빈도**: 분석 논문들에서 미사용 (주로 통계 비교 논문에서 사용)
**대안**: 여러 데이터셋의 ranking 또는 win/tie/lose 집계 테이블로 대체하는 경우 많음
**Phase 5 판단**: TSMAE에서는 CD diagram 불필요; 대신 멀티지표(VUS/PA%K/Affiliation) 랭킹 집계 테이블이 더 적합.

---

## G. 공통 패턴 추출 — Phase 3 블루프린트 직결 지침

### G.1 섹션 분량 배분 (9p 타깃, Elsevier 2단 기준)

```
§1 Introduction:         1.5–2.0p   (전체의 ~20%)
§2 Related Work:         1.0–1.5p   (전체의 ~13%)
§3 Method:               2.5–3.0p   (전체의 ~30%)
§4 Experiments:          3.0–3.5p   (전체의 ~35%)
§5 Conclusion:           0.3–0.5p   (전체의 ~5%)
```

### G.2 Introduction 단락 구성 (TSMAE 적용 시)

| 단락 번호 | 역할 | TSMAE 적용 내용 |
|---------|------|----------------|
| Para 1 | 문제 중요성 | 다변량 시계열 이상탐지의 중요성, CPS/산업 센서 응용 |
| Para 2 | 기존 방법 계보 + 한계 | 비지도(재구성/예측/대조) 방법들 → label 활용 불가 한계 |
| Para 3 | 핵심 관찰 | "소수의 labeled anomaly가 존재하지만 기존 비지도는 이를 활용 못 함; 우리는 MAE 내부에 label 신호를 통합한다" |
| Para 4 | 제안 방법 개요 + Contribution | TSMAE 구조 한 문장 + bullet 3–4개 |
| Para 5 | 논문 구성 | "The rest of this paper is organized as follows: §2... §3... §4... §5..." |

### G.3 Contribution Bullet 작성 가이드

분석 논문들에서 공통으로 관찰되는 contribution 구성 (3–4개):
1. **설정/문제 정의 기여**: "We formalize [설정 이름]..."
2. **아키텍처/방법 기여**: "We propose/design [메커니즘]..."
3. **실험 기여**: "Extensive experiments on [N] benchmarks demonstrate..."

TSMAE 적용 시 추가 가능:
- GRL 기반 표현 억제라는 novel mechanism
- contaminated semi-supervised 설정에서의 강건성

### G.4 Related Work 소절 권장 구성 (TSMAE)

```
§2.1 Time Series Anomaly Detection (비지도 계열 전반 + 재구성/대조/예측 분류)
§2.2 Semi-supervised / PU Learning for TSAD (NRdetector, 기타 약지도 방법)
§2.3 Masked Autoencoders and Self-Distillation (MAE, SDMAE → TSMAE 계보)
```

각 소절의 마지막 1–2문장에 "본 논문과의 차이" positioning 문장 필수.

### G.5 Method 섹션 권장 소절 구조 (TSMAE)

```
§3.1 Problem Formulation      (notation 정의, contaminated semi-supervised 설정)
§3.2 Overall Architecture     (전체 아키텍처 figure 포함, 컴포넌트 개요)
§3.3 Patchification           (linear patchify, 10×50 patches, positional encoding)
§3.4 Masked Encoding          (asymmetric MAE, force_mask_anomaly, 15% masking)
§3.5 Teacher-Student          (asymmetric decoder 3L/2L, self-distillation rationale)
§3.6 Gradient Reversal        (GRL 메커니즘, AnomalyClassifierHead, loss 분기)
§3.7 Training Objective       (L_recon + L_disc + L_GRL + adaptive scaling)
§3.8 Anomaly Scoring          (score = recon + scaled_disc/4, inference-time GRL 비활성화)
```

Note: 저널 분량(9p) 제약상 §3.3–§3.6을 2개로 병합 가능.

### G.6 Experiments 섹션 권장 구조 (TSMAE)

```
§4.1 Experimental Setup
  - Datasets (SWaT/WaDi A1/A2/PSM/SMD/SMAP/MSL + 통계 테이블)
  - Implementation Details (seed, epoch, optimizer, 하드웨어)
  - Evaluation Metrics (PA%K-AUC F1, VUS-PR, VUS-ROC, Affiliation F1 + 정당화)
  - Baselines (계층별: Q1 비지도 22개 + Q3 weak reference 4개)
§4.2 Main Results             (Table 1: 메인 성능 비교)
§4.3 Ablation Study           (Table 2: 컴포넌트별 기여)
§4.4 Label Sparsity Analysis  (라벨 희소화 sweep — placeholder 가능)
§4.5 Qualitative Analysis     (anomaly score 시각화 figure)
§4.6 Sensitivity Analysis     (파라미터 민감도 figure, 선택)
```

---

## H. 캡션 스타일 가이드 (논문별 관찰)

### H.1 Figure 캡션

**일반 형식**: "Figure N: [한 문장 주요 내용]. [보조 설명 0–2문장]."

**강조 요소**:
- 핵심 기호 정의: "red circles indicate anomalies"
- 주의사항: "The failure cases of baselines are bounded by red boxes" (Anomaly Transformer 패턴)
- 비교 맥락: "A higher AUC value... indicates better performance" (Anomaly Transformer 패턴)

### H.2 Table 캡션

**일반 형식**: "Table N: [결과 유형] [방법명] on [데이터셋/설정]. [지표 약어 정의]. Bold indicates best, underline indicates second best."

**DCdetector 패턴**: "Performance ranked from lowest to highest. The P, R and F1 are the precision, recall and F1-score. All results are in %, the best ones are in Bold, and the second ones are underlined."

---

## I. 주의사항 및 검증 필요 사항 (Phase 4 연계)

1. **서지 검증 필수**: 이 문서의 모든 venue·연도·저자 정보는 Phase 4에서 공식 소스(ACM DL, IEEE Xplore, OpenReview, arXiv 최종본) 대조 재검증 필요.
2. **Venue 미확정 논문**: PatchAD(2401.09793), MEMTO(2312.02530), DDMT(2310.08800)는 venue가 arXiv preprint 상태 — 최종 게재처 미확인. 인용 시 "arXiv preprint"로 표기하거나 Phase 4 확인 후 갱신.
3. **DCdetector 구조 직접 확인**: PDF 직접 열람 완료 (2026-06-11), 단 KDD 최종본(ACM DL)과 arXiv v3 간 차이 미확인 — Phase 4 교차 확인 권장.
4. **NeurIPS 2024–2025 TSAD 논문** (fixer r2, V-004 갱신): NeurIPS 2024 proceedings 직접 조회로 TSAD 논문 존재 **확인됨** — ① **SARAD** (Dai et al., Main Conference Track, proceedings.neurips.cc 2024/hash/56ad264a…) ② **"The Elephant in the Room: Towards A Reliable TSAD Benchmark"** (Liu & Paparrizos, Datasets & Benchmarks Track — **VUS-PR을 최신뢰 지표로 권고**, 우리 지표 정당화 인용 후보). 상세 구조 분석은 미수행 — VENUE_AND_PAPER_LIST §4 미수록 후보에 등재, Phase 4에서 확인. NeurIPS 2025는 미조회 상태 유지.
5. **ICLR 2024 TSAD 논문**: ModernTCN(ICLR 2024 spotlight)은 이상탐지를 5개 태스크 중 하나로 포함하지만 TSAD 전용 논문은 아님. ICLR 2024에서 TSAD 전용 논문의 존재 여부는 Phase 4 확인 필요.

---

## 부록: 정정 이력

### 2026-06-11 fixer r2 (adversarial review `paper/99_reviews/p2_venue_corpus_r1.md` 반영; fixlog: `p2_fixlog_r2.md`)

1. **[S-001, MINOR]** §F.2 CATCH 행 "12+6=18 ds × 2 / ~576" → "10+6=16 ds × 2 / ~512" — arXiv HTML Table 2 캡션 직접 재확인("10 real-world datasets and 6 synthetic datasets of different types of anomalies"). 리뷰 권고안("10+12=22 ds")이 아니라 원문 테이블 실측 기준으로 정정 + abstract 총 커버리지(22)와의 구분 주석 추가.
2. **[S-002, NOTE]** §A.2 caption 방향 — "(학회와 반대인 경우 있음)" → "(elsarticle 기본값 — 표준적이며 대부분 학회와 동일)".
3. **[V-004 연계, MINOR]** §I.4 — NeurIPS 2024 TSAD 논문 직접 조회 결과(SARAD Main Track, TSB-AD Elephant in the Room D&B Track) 반영, "미확인" 상태 해소.
4. **[S-003, NOTE]** 실행 가능성 평가 — 이상 없음 판정, 조치 불요 (§G.5 Method 8소절 과세분화 위험은 문서 자체 자각 사항으로 Phase 3 압축 결정 사안 유지).
