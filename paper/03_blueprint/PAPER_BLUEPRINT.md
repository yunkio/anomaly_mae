---
phase: 3
agent: blueprint-reviser
directives: [T3, R1, R2, R5, R6, R7, R8, R9, R10, R11, R15, R16, R19, R20, R21, R22, R32]
revision: r3 (fixer — p3_rereview_adversarial_r2.md(NEW-B1/B2 + m1 + n1) + p3_rereview_redteam_r2.md(R2-MAJ-01 + MIN-01..04 + NOTE-01..02) 전수 반영; fixlog: paper/99_reviews/p3_fixlog_r3.md; Phase 1 정본 동반 보강: 271_CONFIG_TRUTH r4 / CODEBASE r4 / SYNTHESIS r3)
last_modified: 2026-06-11
authority: |
  Phase 1 정본 우선순위(271_CONFIG_TRUTH > RESEARCH_SYNTHESIS > 나머지)를 유지.
  본 문서의 contribution 구조·setting 명칭·비교 조건 결정은 DECISION_LOG로 전사 예정.
  실험 수치는 전부 placeholder — 본문에 [X.XX] 또는 [BEST] 형태로만 표기.
  분량 수치의 정본은 PAGE_BUDGET.md — 본 문서 §2는 PAGE_BUDGET §1을 전사한 사본 (ADV BLK-001).
---

# PAPER BLUEPRINT — TSMAE Phase 3

> **r3 핵심 정정 요약** (2026-06-11, fixer; 상세: `paper/99_reviews/p3_fixlog_r3.md`)
> - **GRL 이중 λ 구조 (NEW-B1)**: GRL에는 서로 별개의 λ 2개가 공존하며 271에서 **둘 다 활성** — ① 손실 가중치 λ_GRL = grad-ratio adaptive(clamp[0,10], 직전 epoch) × 0.2 (trainer.py:751–765), ② gradient **반전 계수 λ_rev** = Ganin-style sigmoid ramp `2/(1+exp(−10p))−1`, 0→≈1 (trainer.py:1201–1211 → model.py:1152–1153). r2의 "sigmoid ramp-up 271 미사용" 단정·금지 조항은 **손실 가중치 측면만 본 오류** — 철회·교체 (§5.5/§5.6/§9.2). 근본 원인인 Phase 1 정본 누락은 271_CONFIG_TRUTH §VIII r4로 보강 완료.
> - **warmup 중 student forward skip (NEW-B2)**: 학습 경로에서 student decoder forward 자체가 생략(model.py:1119, trainer.py:526–535) — r2 §5.5의 "forward는 수행되지만 gradient 차단" 서술은 코드와 반대였음 → 교체.
> - **ablation suite 실행 등재 (R2-MAJ-01)**: Table 3 행 2–5·7을 §0.4 Phase 5 진입 조건에 등재 + 행 5·7 conditional placeholder 명시 (행 7은 contribution bullet 3 load-bearing).
> - MINOR/NOTE: 36/113 집계 기준 통일(NEW-m1), §14 "유일한" 완화(R2-MIN-01), Table 4 실행 사양 명기 — use_grl=True 유지 + `loss.py` pos_count==0 skip 인용(R2-MIN-03), 6.2% 상한 어법 통일(R2-MIN-04), Intro Para 3 에코 스코핑(R2-NOTE-02), §B.4 실측 격상 권고(R2-NOTE-01), affiliation 라인 표기 통일(NEW-n1). PAGE_BUDGET fallback 사다리 재정렬은 PAGE_BUDGET r3(R2-MIN-02).

> **r2 핵심 정정 요약** (상세: `paper/99_reviews/p3_blueprint_fixlog_r2.md`)
> - d_model은 **dynamic이 아니라 전 entity 공통 512 고정** (271_CONFIG_TRUTH §II + PSM checkpoint `patch_embed=(512,250)` 직접 실측 2026-06-11 — dynamic이면 256이어야 함). dim_feedforward=2048 고정. NOTION I-3의 dynamic 매핑 표는 Set C preset 문서이며 271 실측과 불일치 (batch_size 512→1024 override와 동급의 stale).
> - ~~GRL λ는 sigmoid ramp-up이 아니라 **trainer inline grad-ratio adaptive λ** (ADV BLK-004).~~ → **r3 정정 (NEW-B1)**: 이 단정은 손실 가중치에 한해서만 참 — sigmoid는 반전 계수 λ_rev로 271에서 실제 사용 (위 r3 요약).
> - SDMAE 구조는 "공유 encoder 이중 decoder"가 아니라 **teacher decoder에서 student가 분기(branch-off)** (ADV MAJ-001).
> - contribution bullet 3에서 warmup 제거 (RT BLOCKER-02), 50% prefix 방어 §14 전면 재구축 (RT BLOCKER-01), 방법-프로토콜 효과 분리 보조분석 main text 격상 (RT BLOCKER-03).

---

## 0. 전략 전제

### 0.1 중심 논제 (Thesis)

기존 다변량 시계열 이상탐지 방법은 학습 데이터 안에 소량 존재하는 **labeled anomaly를 학습 신호로 전혀 활용하지 못한다**. 본 논문은 Masked Autoencoder 위에 비대칭 Teacher–Student 자기증류와 Gradient Reversal Layer를 결합하여, 이 labeled anomaly를 (1) masking 우선순위, (2) 손실 방향 분기, (3) 표현 억제(adversarial suppression)라는 세 독립 경로로 end-to-end로 통합하는, **우리가 아는 한(to our knowledge) 최초의** 단일 모델을 제안한다. 벤치마크 역시 이 설정을 평가할 수 있도록 재설계(원본 test 앞 50% train 편입)하여, contaminated semi-supervised 평가 프로토콜을 표준화한다.

> ⚠️ **"최초" 주장 스코핑 경계 (Phase 5 필수 — RT NOTE-03 / ADV MINOR-006)**: 최초성 주장의 정확한 범위는 **"다변량 TSAD에서 labeled anomaly를 자기지도 표현 학습의 기울기(gradient)에 직접 통합하는 end-to-end 단일 모델"**이며 반드시 "(to our knowledge)"를 동반한다. 이 스코핑 밖으로의 확장("최초의 semi-supervised TSAD", "최초의 contaminated 벤치마크" 등) **절대 금지**. 이 주장은 RESEARCH_SYNTHESIS §②-6에서 INFERENCE 등급 — Phase 4에서 반증 논문 부재 검증 필수 (§16 Phase 4 연계).

### 0.2 논문 포지셔닝 한 줄

"MAE 기반 단일 end-to-end 모델로 labeled anomaly를 표현 학습 자체에 통합한 다변량 시계열 이상탐지 — contaminated semi-supervised 설정의 (to our knowledge) 최초 통합 솔루션."

### 0.3 핵심 차별점 3축 (reviewer-facing)

- **설정**: unsupervised도 완전 지도도 아닌 contaminated semi-supervised — labeled anomaly 소수, unlabeled 다수(가정; main 실험은 label 가용성 상한 케이스 — §5.2). NRdetector와 가장 근접하지만 표현 학습-라벨 분리 vs 통합이 근본 차이.
- **메커니즘**: GRL을 통한 gradient-space adversarial suppression — labeled anomaly 정보를 student 표현에서 능동적으로 제거. 타깃/손실 공간에서 이상을 무시하도록 유도하는 SDMAE와 다른 작동 계층 (이 구분은 §3.5 본문에 1문장 명시 — RT MAJOR-01).
- **평가 프로토콜**: contaminated train(test 앞 50% 편입) + 5종 multi-metric(PA%K-AUC F1, VUS-PR, VUS-ROC, Affiliation F1, PA%K-AUC PR) + SWaT excl22 dual-eval — 단일 임계값/단일 지표 의존 없음.

### 0.4 실험 완료 상태 경고 (Phase 5 진입 조건 — RT R5/V2)

- 현재 완료된 MAE 271은 **학습 단위 36/113** (= SWaT 1 + WaDi 2 + PSM 1 + SMD 22/28 + SMAP 5/54 + MSL 5/27; **평가 단위 기준 37/114** — SWaT full/excl22 dual-eval 2 집계). 분자·분모 집계 기준 혼용 금지 (r3, NEW-m1 — 구판 "37/113"은 분자=평가 단위·분모=학습 단위 혼합; §6.2의 113 학습/114 평가 구분과 정합). SMD 6, SMAP 49, MSL 22 entity 잔여.
- baseline 쪽 SMD/SMAP/MSL은 per-entity 정규화(2026-06-02) 이후 STALE — 재실행 필요.
- **weakly-supervised 4종(DeepMIL/WETAS/TreeMIL/NRdetector) GPU 전체 실험 미실행** (RESEARCH_SYNTHESIS FEEDBACK-2) — NRdetector는 가장 직접적 경쟁자이므로 Main Table 완성의 필수 조건.
- **Ablation suite (Table 3 행 2–5·7) 미실행 — Phase 5 진입 전 실행 필수 (r3 신설, R2-MAJ-01)**: 정본(RESEARCH_SYNTHESIS 표A)상 decoder-depth ablation "필요"(미존재)·FM 제외 ablation "결과 없음" — 행 2–5·7 전부 warmup 행 6과 동급의 미실행 placeholder다. **행 7(symmetric decoder)은 contribution bullet 3(capacity-gap)의 유일한 정량 근거(load-bearing), 최소 행 2(w/o GRL)·행 7은 필수 실행**; 271 canon config 기반(행 2는 anomaly-OD 제외 유지 설계 조건 — §6.7). 미완 행은 행 6과 동일한 conditional 규칙(본문 잔류 금지) + 행 7 미완 시 bullet 3 주장 강도 하향 (§6.7). EXPERIMENT_EXECUTION_TODO 집계에 등재 (p3_fixlog_r3 §EXECUTION-TODO — fixlog r2 §7의 8항목을 대체·확장).
- contribution bullet 4의 "six multivariate datasets"·"22 unsupervised baselines" 주장은 **완주 전까지 전부 placeholder** — Phase 5 진입 전 완주 필수 (EXPERIMENT_EXECUTION_TODO).

---

## 1. Elsevier 필수 요소

### 1.1 Title (후보는 §10)

### 1.2 Abstract (150–200 words, structured 불사용)

4단 구조: (1) 문제 설정 + 동기 1–2문장. (2) 방법 핵심 2–3문장: MAE + 비대칭 Teacher–Student + GRL semi-supervised anomaly suppression. (3) 벤치마크 프로토콜 1문장: contaminated train + multi-metric. (4) 결과 1–2문장: [N] 데이터셋에서 state-of-the-art [요약], 라벨 희소화 강건성 확인. 수치 없음(placeholder 정책).

### 1.3 Keywords (6–7개 예시)

Multivariate time series, Anomaly detection, Semi-supervised learning, Masked autoencoder, Self-distillation, Gradient reversal, Contaminated benchmark

### 1.4 Highlights (5 bullet, 각 ≤125 chars)

- We address anomaly detection under a contaminated semi-supervised setting where labeled anomalies coexist with unlabeled data during training.
- We propose [MODEL], combining asymmetric Teacher–Student MAE with gradient reversal to suppress anomaly information in the student's representation.
- Labeled anomalies guide training via masking priority, loss direction, and adversarial suppression — three orthogonal integration paths.
- A contaminated benchmark protocol (test-prefix incorporation) enables fair evaluation of semi-supervised methods across six real-world datasets.
- The model remains robust under label sparsity, outperforming 22 unsupervised baselines under rigorous multi-metric evaluation.

---

## 2. 섹션 구조 전체 개요

```
§1  Introduction                 1.6p
§2  Related Work                 1.1p
§3  Methodology                  2.7p
§4  Experiments                  3.3p
§5  Conclusion                   0.3p
    References
    Appendix
```

총 본문 9.0p (table/figure 포함). **분량 수치의 정본은 PAGE_BUDGET.md §1** — 위 수치는 그 전사(轉寫)이며, 두 문서가 충돌하면 PAGE_BUDGET.md를 따른다 (ADV BLK-001 — §4가 3.3p로 r2에서 조정됨: RT BLOCKER-03 protocol-effect 보조분석 추가 반영; §1은 1.6p로 0.1p 감축).

---

## 3. §1 Introduction

### 3.1 논증 흐름 (5단락)

**Para 1 — 문제 중요성 (3–4 문장)**
- 다변량 시계열 이상탐지의 산업·안전 응용(CPS, 데이터센터, 우주 telemetry 등).
- 고차원 피처 간 상호의존이 이상을 잠재적으로 다중 채널에 분산시킴.
- 실시간/대규모 환경에서 완전 label 수집이 불가 — unsupervised 방법의 현실적 지배.
- 필요 근거 유형: 응용 사례 1–2개 인용(industry report 또는 survey).

**Para 2 — 기존 방법 계보 + 한계 (4–5 문장)**
- 비지도 계열 4유형 요약: (i) 재구성 기반, (ii) 예측 기반, (iii) 대조학습, (iv) 밀도 추정.
- 공통 한계: "train 데이터는 모두 정상"이라는 암묵적 가정 → labeled anomaly를 학습 신호로 활용하는 경로가 구조적으로 없음.
- "비지도에게 라벨을 주는 최선도 오염원 제거(Q3 normalonly)에 그친다"는 점 1문장.
- 괄호 클러스터 인용(R19 원칙)으로 처리, 개별 설명 불필요.

**Para 3 — 핵심 관찰 + 동기 (4–5 문장) [R11 설정 도입]**
- 현실에서는 labeled anomaly(고장 기록 등)가 소수 존재한다.
- 이것이 비지도 방법에게는 오염원이지만, semi-supervised 방법에게는 귀중한 학습 신호다.
- (RT BLOCKER-01 에코 1문장) 그러나 **본 논문이 평가하는 표준 MTSAD 벤치마크들**의 원본 train split에는 labeled anomaly가 구조적으로 존재하지 않아(정상 운영 구간 또는 라벨 부재), 이 설정을 평가할 수 있는 프로토콜 자체가 부재하다 — §4.1.1에서 상술. **(r3 스코핑, R2-NOTE-02)**: 전칭 표현("기존 공개 벤치마크" 일반) 금지 — "the standard MTSAD benchmarks we evaluate on" 수준으로 한정 (반례 공격 차단 — 예: 본 프로젝트가 보유·제외한 Exathlon; Phase 4 clean-train 문헌 검증과 연동).
- 핵심 관찰: labeled anomaly를 (a) 어느 위치를 주목해야 하는가, (b) 복원에서 무엇을 회피해서는 안 되는가, (c) 표현에서 무엇을 지워야 하는가로 동시에 활용하면 재구성 오차와 표현 불일치라는 두 신호를 모두 증폭할 수 있다.
- **(bridge 문장 — RT MINOR-06)**: (b)만으로는 부족하다 — anomaly 패치를 손실에서 제외해도 student가 학습 중 반복 노출된 anomaly 패턴을 기억해 잘 복원하는 우회로가 남아 discrepancy 신호가 약해질 수 있다. (c)의 표현 수준 적극 제거가 이 우회로를 차단한다.
- 기존 유일 시계열 semi-supervised 연구(NRdetector)도 표현 학습은 라벨-불가지론적 사전학습에 위임 — 라벨이 표현 자체를 형성하지 못한다.

**Para 4 — 제안 방법 개요 + Contribution bullet (4–5 문장 + bullet 4개)**
- "[MODEL] is an end-to-end framework that integrates labeled anomaly information directly into the representation learning process via three orthogonal mechanisms: masking priority, loss bifurcation, and gradient-reversal suppression."
- Contribution bullets: §11 결정 ① 재설계안 사용.

**Para 5 — 논문 구성 1문장**
- "The rest of this paper is organized as follows: §2 reviews related work; §3 describes the proposed [MODEL]; §4 presents experimental results; §5 concludes."

### 3.2 Figure/Table 계획

- **Fig. 1** (§1 초반, ~0.4p): 설정 비교 다이어그램. (a) unsupervised Q1: anomaly가 오염원으로 섞임. (b) unsupervised Q3: labeled anomaly로 오염원 제거(normalonly). (c) [MODEL]: labeled anomaly를 학습 신호로 통합. 3-way 비교로 설정 차별점을 시각적으로 전달. Phase 5 작성 시 이 figure를 Introduction 후반부(Para 3 뒤)에 배치.

---

## 4. §2 Related Work

### 4.1 소절 구성 (MECE R1 검증)

```
§2.1  Multivariate Time Series Anomaly Detection
§2.2  Semi-supervised and PU Learning for Anomaly Detection
§2.3  Masked Autoencoders and Self-Distillation in Anomaly Detection
```

세 소절이 MECE를 만족하는 논거: §2.1은 본 논문이 풀고자 하는 도메인 문제 공간(기존 비지도 한계 포지셔닝), §2.2는 설정 차원에서의 선행 연구 공간(R11/R20 전략), §2.3은 방법 차원에서의 선행 기술 계보(R22/R21 전략). 세 소절 사이에 중복 없음. **WETAS/DeepMIL/TreeMIL은 §2.2 전속 — §2.1 포함 절대 불가** (RT MINOR-03; 4.3 참조). TFMAE는 §2.3 전속 — §2.1 클러스터에서 제외 (ADV MINOR-001).

### 4.2 §2.1 Multivariate Time Series Anomaly Detection

**내용**:
- 비지도 TSAD의 4유형: 재구성 기반(DAGMM, OmniAnomaly, USAD, MEMTO), 예측 기반(GDN), 연관/대조 기반(Anomaly Transformer, DCdetector, CATCH), 자기지도(TranAD, TimesNet). **TFMAE는 이 클러스터에서 제외하고 §2.3에서만 1회 인용** (ADV MINOR-001 — §2.1/§2.3 이중 등장의 약한 MECE 위반 제거).
- **DAGMM 인용 정책 (ADV NOTE-001)**: §2.1 클러스터 인용의 DAGMM은 원논문(Zong et al., ICLR 2018)을 지칭. "simplified variant" 표기는 §4.1.4 baseline 설명과 Appendix에서만 (각주 처리) — related work에서 변형 언급 금지.
- 다변량 설정에서의 추가 도전: 변수 간 상관 포착, 채널 수 이질성.
- 공통 한계: "train = all normal" 가정 — 실세계 contaminated train에서 성능이 이 가정 위반에 민감함.
- 마지막 1–2 문장: 본 논문은 이 가정 없이 labeled anomaly를 학습 신호로 쓰는 방향으로 이 한계를 극복한다.
- 인용 정책(R19): 이 소절에 나오는 모델들은 괄호 클러스터 인용(방법 계열별 1개 클러스터) — 개별 소개 문단 없음. baseline 모델들은 §4 Experiments에서 이름+인용으로 최초 결합.

**단락 수**: 2–3단락.

### 4.3 §2.2 Semi-supervised and PU Learning for Anomaly Detection (R20 핵심)

**내용**:
- PU Learning의 일반 정의: positive(confirmed anomaly) + unlabeled(unknown). 비용민감형(Non-negative Risk Estimator류), 샘플선별형(reliable-negative extraction류) 양대 계열 간략 소개.
- 비시계열 영역에서의 적용 사례 언급(괄호 인용).
- **핵심 강조 (R20)**: 다변량 시계열에서 semi-supervised/PU 적용은 극히 드물다 — 정확한 스코핑: "심층 표현 학습과 통합된 PU/SSL 기반 다변량 TSAD는 거의 없다."
- **weakly-supervised 계열의 배치 확정 (RT MINOR-03 — "또는" 모호성 제거)**: DeepMIL, WETAS, TreeMIL은 **이 소절(§2.2)에서만** "이전 weakly-supervised 계열"로 1문장(괄호 인용) 처리. §2.1 비지도 클러스터 포함 절대 불가(MECE 위반). NRdetector는 이들과 분리하여 "이 설정에서 가장 근접한 심층 학습 기반 선행 연구"로 별도 문장 — "거의 유일" 주장의 정밀 스코핑 유지.
- **end-to-end 차별 논리 명시 (RT MAJOR-02)**: WETAS/TreeMIL/DeepMIL도 단일 모델 학습이지만, 이들의 weak label은 **분류/정렬 목적함수의 지도 신호(출력 결정 수준)**로 쓰인다 — "Weakly supervised approaches ... optimize models to classify segments accurately by leveraging segment-level labels" (NRDETECTOR_DOSSIER §4 원문 인용 구조 참조). 재구성/마스킹 자기지도 pretext 없이 라벨이 곧 목적함수다 (NRDETECTOR_DOSSIER D5). 본 논문의 주장 범위는 "**자기지도(재구성 기반) 표현 학습의 기울기**에 라벨을 통합"이므로 이들과 겹치지 않는다 — 이 구분 논리를 §2.2에 1–2문장으로 명시할 것.
- NRdetector(Wang et al., KDD 2025) 차이점 우선(R20): NRdetector는 사전학습 표현(WETAS/DiCNN)과 PU 분류를 분리(multi-stage, not end-to-end), 라벨이 표현 형성 자체에 개입하지 않음. 세그먼트 단위 weak label vs 본 논문의 point/window 단위 labeled anomaly도 구분. 공통점 3개만 짧게 인정 후 차이의 중심축으로 D1/D3/D5 배치.
- 마지막 포지셔닝 문장: 본 논문은 labeled anomaly를 표현 학습의 기울기에 직접 통합하는, 우리가 아는 한 첫 번째 end-to-end 다변량 TSAD 모델이다 (§0.1 스코핑 경계 준수 + "to our knowledge" 필수).

**단락 수**: 3–4단락.

### 4.4 §2.3 Masked Autoencoders and Self-Distillation in Anomaly Detection (R9/R21/R22)

**내용**:
- Vision MAE(He et al., CVPR 2022)가 patch masking + bidirectional 재구성으로 강한 표현 학습을 보인 원류(R22 — 이것이 본 논문 patch/masking의 직접 계보). 시계열에서 유사한 패치/마스킹 방식을 쓰는 논문들이 있지만 이는 독립 수렴이지 본 논문이 그들을 계승한 것이 아님(R22 원칙 반영).
- Knowledge distillation의 이상탐지 적용: 사전학습 teacher-랜덤초기화 student 격차를 이상 신호로 쓰는 계열(괄호 인용).
- Self-distillation 계보 (**ADV MAJ-001 구조 사실 정정**): Zhang et al. [TPAMI 2022]이 "self-distillation"을 도입, SDMAE(Ristea et al., CVPR 2024)가 이를 anomaly detection에 처음 적용 — 단 SDMAE의 구조는 "공유 encoder 이중 decoder"가 **아니라**, **teacher decoder의 첫 transformer 블록 뒤에서 student decoder가 분기(branch-off)하는 구조**다 (ANCHOR_SDMAE_DOSSIER §3.1: "A student decoder branches out from the teacher after the first transformer block of the main decoder"). 본 논문은 공유 encoder에서 **서로 독립인** 비대칭 Teacher(3L)/Student(2L) decoder 2개가 병렬 분기 — 구조 차이("독립 별도 decoder vs branch-off 분기")는 R21 방어 각주 재료로 활용.
- 옵션 C 문구 (**RT MAJOR-08 — 계보 강조 완화**): "extend analogous self-distillation principles"는 SDMAE를 parent로 격상시킬 위험 → **"we adapt this architectural paradigm to multivariate time series"** 또는 **"we apply the time-series counterpart of this design"** 계열 표현을 Phase 5에서 우선 시도. 초안: "In this work, we adapt this architectural paradigm to multivariate time series, within a contaminated semi-supervised framework that leverages labeled anomalies through targeted masking and gradient-based information suppression." 핵심 포지셔닝: SDMAE는 parent가 아니라 **sibling in a family**.
- R9 원칙 준수: SDMAE를 계보 중 하나로 1–2문장 처리, 차이 나열 없음. 각주 1개 추가(§3 self-distillation 정의 근처)로 R21 방어 — 단 **각주에는 용어 계보+구조 차이(branch-off vs 독립 decoder)만 담고, 작동 계층 차이(타깃/손실 공간 vs gradient 공간)는 §3.5 본문 1문장으로 이동** (RT MAJOR-01/MAJOR-08: 각주 1개에 3축 방어를 모두 담기엔 지면 부족 — 본문 분산 배치).
- TFMAE(Fang et al., ICDE 2024): 시계열 MAE 사례 — 단 1문장 괄호 인용으로 처리 (**유일한 언급 위치** — §2.1 클러스터에서 제외, ADV MINOR-001).

**단락 수**: 2–3단락.

**결정: 옵션 C 채택** (SDMAE를 distillation 계보 내 자연 언급). 각주로 R21 방어(용어 계보+구조 차이) 탑재, 작동 계층 차이는 §3.5 본문.

---

## 5. §3 Methodology

### 5.1 소절 구성

```
§3.1  Problem Formulation and Setting
§3.2  Overall Architecture
§3.3  Patch Embedding and Masking
§3.4  Asymmetric Teacher–Student Decoders
§3.5  Label-Guided Training
§3.6  Anomaly Scoring and Inference
```

Phase 2의 G.5 8소절 안을 6소절로 압축. §3.3은 patchify + masking 통합, §3.5는 GRL + loss 방향 분기 + FM 통합.

### 5.2 §3.1 Problem Formulation and Setting

**내용**:
- Notation 도입: 다변량 시계열 X ∈ R^{T×F}, window W ∈ R^{L×F} (L=500, F 가변), patch P_i ∈ R^{s×F} (s=10, N=50 patches).
- 레이블: y^p ∈ {0,1}^N (패치 단위), y^w ∈ {0,1} (윈도우 단위, window-mode GRL 타깃).
- **Contaminated semi-supervised setting 정의 (R11)**: 학습 데이터 D_train = {(W_i, Y_i)} where Y_i ∈ {0,1}^L — 일부 Y_i에 labeled anomaly 존재, 나머지는 정상. 평가 D_test = 원본 test 뒤 50%, Y_i 일체 미사용(추론 시 label-free).
- train labeled-anomaly 비율 (**ADV MAJ-008 — 단일 상한 "≤6.2%" 폐기, 실측 열거로 교체**): SWaT 1.63%, WaDi A1 0.52% / A2 0.76%, PSM 6.20%, SMAP concat 0.70%, MSL concat 1.70% (EXPERIMENT_PROTOCOL_TRUTH §① 실측); **SMD는 machine별 상이 — 잔여 entity 완주 후 per-machine 비율 확정 전까지 전체 상한 단정 금지** (현재 최대 실측치는 PSM 6.20%이나 SMD 미확정).
- **main 실험 = label 가용성 상한 케이스 명시 (RT MAJOR-09 — RESEARCH_SYNTHESIS §②-1/②-2/②-3 3단 구조 반영 필수)**: ②-1 설정(가정)=대부분 unlabeled + 소수 labeled anomaly / ②-2 main 실험(FACT)=train 구간 내 anomaly 전부 labeled인 **상한(upper-bound) 케이스** / ②-3 라벨 희소화 sweep(§4.4)=일부 anomaly가 unlabeled로 잔류하는 **일반 케이스 검증**. 이 3단 구분 없이 "contaminated semi-supervised"만 명명하면 "main 실험이 semi-supervised가 아니다" reject 위험 — §3.1에서 1–2문장으로 명시.
- "contaminated semi-supervised"라는 명칭을 이 소절에서 공식화. ("semi-supervised"를 대표 명칭으로 쓰되, 더 정확하게는 labeled anomaly가 소수 섞인 contaminated train이라는 점을 명시. R11 Directive 직접 충족.)
- 추론 시 label-free임을 명시.

**R10 논증 배치**: "왜 다변량에서 이 설정이어야 하는가" — 실제 CPS/서버 환경에서 labeled anomaly는 운영 기록(고장 이력)에서 자연 발생하며, 다변량 센서 간 동기 이탈이 이상의 특징이므로 표현 학습이 이 구조 정보를 포착해야 탐지가 가능하다.

### 5.3 §3.2 Overall Architecture

**내용**:
- 전체 흐름 1–2 문단: Input → Patchify → Encoder → [Teacher Decoder | Student Decoder + GRL] → Score.
- **Fig. 2** (full-width, ~1/3p): 아키텍처 다이어그램. 5개 컴포넌트 색 구분: (1) Patch Embedding, (2) Transformer Encoder (shared), (3) Teacher Decoder (3L), (4) Student Decoder (2L), (5) GRL + AnomalyClassifierHead. 학습/추론 두 패널(또는 학습 패널 하나에 "inference-time" 비활성 표시). force_mask_anomaly와 score 수식을 figure 내 레이블로 연결.
- **GRL 위치·추론 비활성 명시 (ADV BLK-002 — Fig. 2 필수 레이블 2건)**:
  1. GRL + AnomalyClassifierHead의 적용 위치는 **student decoder 마지막 층 hidden — output projection 이전** (271_CONFIG_TRUTH §VI "called on student hidden", model.py:1150–1154; NOTION I-3 forward-flow 표의 "Output Projection 다음" 배치는 부정확 — 코드 정본 우선). figure 레이블에 위치를 정확히 표기.
  2. **"GRL: training only (추론 시 비활성)"** 명시 — dashed box 또는 "training only" 주석 필수 (§3.6의 추론 label-free 서술과 시각적으로 정합).
- Encoder는 teacher path gradient로만 학습함을 명시 — 근거 2가지를 구분해 서술: ① student decoder는 encoder latent를 `latent_visible.detach()`로 받아 student 손실 gradient가 encoder로 흐르지 않음, ② GRL gradient 역시 같은 detach로 encoder에서 차단.

### 5.4 §3.3 Patch Embedding and Masking

**내용**:
- Linear patchify: patch P_i = X[is:(i+1)s, :] flatten → Linear(s×F → d_model) + LayerNorm. **d_model = 512 (전 데이터셋·entity 공통 고정값)** — 정정(r2): "F에 따른 동적 결정(dynamic)"은 271 실측과 불일치. 근거: 271_CONFIG_TRUTH §II(114 공통 키에 `d_model=512` 포함, 전 37 entity metadata 동일) + PSM(F=25) checkpoint `patch_embed.weight=(512, 250)` 직접 실측(2026-06-11; dynamic 매핑이었다면 256이어야 함). NOTION I-3의 dynamic 매핑 공식·표는 Set C preset 문서로 271 미적용 — 논문 서술 금지.
- 마스킹: 8 patches (round(50×0.15)) 고정 마스킹. mask_after_encoder=True → 가시 42 패치만 encoder 통과.
- **force_mask_anomaly** 메커니즘: priority_p = 1[patch contains anomaly]×1000 + η_p, masked = TopK_8(priority). anomaly budget ≤8이면 전부 마스킹 + 나머지 random, 초과 시 random 8개.
- **R10 논증 (§3.3 전용)**: "왜 다변량 시계열에서 patch masking인가" — 한 패치(10 timestep × F 피처)는 여러 변수의 시간 구간 전체를 단위로 삼아, 피처 간 상관 구조가 있는 맥락에서 복원을 강제. anomaly는 이 상관에서 이탈하므로 복원 오차가 커진다. force_mask_anomaly는 클래스 불균형 환경에서 anomaly 위치의 복원 회피를 방지하는 직접 대응.
- **R22 원칙**: vision MAE(He et al. 2022)에서 patch/masking 아이디어를 도입했음을 1문장 명시. 시계열 패치 연구들과의 유사점은 독립 수렴 수준임을 §2.3에서 처리 완료.

**수식 번호**: (1) Linear patchify + LayerNorm, (2) masking priority, (3) masked token selection.

### 5.5 §3.4 Asymmetric Teacher–Student Decoders

**내용**:
- Transformer Encoder: 4층, Pre-Norm, GELU, **d_model=512, nhead=8, dim_feedforward=2048** (전 entity 공통 — 271_CONFIG_TRUTH §II/§VIII; "4×d_model" 동적 표현 대신 고정값 서술), dropout=0.15.
- Teacher Decoder (3L) / Student Decoder (2L): 둘 다 self-attention only (use_transformer_encoder_decoder=True, cross-attention 없음). Mask token 별도(shared_mask_token=False). Teacher mask token + Teacher PE → Teacher hidden h_T → Teacher output o_T. Student latent = encoder latent.detach() → Student mask token + Student PE → Student hidden h_S → Student output o_S.
- **비대칭 설계 논리 (R10)**: Teacher가 더 깊어 anomaly 포함 복원을 정확히 학습한 후, Student는 낮은 capacity로 모방을 시도하면 anomaly 패턴에서 모방이 더 크게 실패한다 (capacity gap → discrepancy 신호). 다변량 시계열에서 teacher가 학습한 "정상 변수 간 상관 구조"를 student가 낮은 capacity로 모방할 때 anomaly로 인한 비정상 상관 패턴에서 격차가 커진다.
- **Teacher-only warmup 250 epoch** 설명 (**r3 정정, NEW-B2 — r2의 ADV MINOR-003 처리(NOTION I-4 stale 서술 채택)가 코드와 반대 방향이었음**): warmup(epoch < 250) 동안 **학습 경로에서는 student decoder forward 자체가 생략**된다 — `teacher_only` 전파(trainer.py:526–535, 2026-05-29 변경·271 실행 이전) → model 게이트(model.py:1119); 손실 게이트(loss.py:213)는 이중 방어 (271_CONFIG_TRUTH §VIII r4). student 파라미터는 갱신되지 않으며("frozen"), per-epoch **평가 경로는 teacher_only=False 기본값으로 full forward**를 수행한다. student 학습은 0-based epoch 250부터 개시. 논문 §3.4 서술도 이 사실 기준 — "student forward는 수행되나 gradient만 차단" 식 서술 **금지** (코드 공개 시 즉시 반박됨). **capacity-gap·안정화 논리 재점검 (r3)**: forward-skip 사실은 §11 bullet 3의 capacity-gap 논증·warmup의 안정화 논리와 충돌하지 않음 — 비대칭 capacity는 라벨·학습 단계와 무관한 구조 속성이고, "수렴한 teacher 기준 위에서 student가 모방을 시작한다"는 안정화 서사는 forward-skip 하에서 오히려 더 정확해진다(학습 경로에서 student는 epoch 250 이전에 일절 관여하지 않음).
- **Warmup 종료 후 손실 투입 — GRL 이중 λ 구조 (r3 정정, NEW-B1 — r2의 "sigmoid 미사용" 단정·금지 철회)**: GRL에는 **서로 별개의 λ 2개**가 공존하며 271에서 **둘 다 활성**이다 (271_CONFIG_TRUTH §VIII GRL Details r4):
  1. **손실 가중치 λ_GRL**: GRL·FM **손실 항**은 warmup 종료 직후(epoch 250부터) **ramp 없이 즉시** 투입되며, 유효 가중치는 trainer inline grad-ratio adaptive λ(직전 epoch 값 적용)로 자동 균형 — λ_GRL_adp = clamp(‖∇L_main‖/(‖∇L_GRL‖+1e-4), 0, 10), λ_GRL_eff = λ_GRL_adp × grl_loss_weight(0.2) (trainer.py:751–765). [r2 서술 유지 — 코드 일치 재확인]
  2. **반전 계수 λ_rev**: gradient reversal의 backward 곱셈 계수는 **Ganin et al. (2016)의 sigmoid schedule**을 따라 점진 증가 — λ_rev = 2/(1+exp(−10p))−1, p = clip((epoch−250+1)/250, 0, 1) (student-phase 진행률; 매 epoch train_epoch 전 설정, warmup 중 0.0; trainer.py:1201–1211 → model.py:1152–1153). epoch 250에서 ≈0.02로 시작, 마지막 epoch에서 ≈1.0 — **adversarial suppression 강도는 점진 ramp된다** (r2의 "warmup 종료 직후 suppression 즉시 full 강도" 함의는 오류). student hidden 도달 adversarial gradient = −λ_rev × λ_GRL_eff × ∂L_cls/∂(GRL 출력). FM에는 대응 ramp 없음 — sigmoid는 GRL 반전 계수 전용.
  - **논문 §3.4 서술 방침 (R23/R27 — 일반적 서술 수준)**: 두 메커니즘 모두 method에 기술하되 구현 잡설 없이 일반적 표현으로 — 손실 가중 적응은 "the weight of L_cls is adaptively balanced by a gradient-norm ratio (scaled by 0.2)" 수준, 반전 계수는 "the reversal strength follows the standard sigmoid schedule of Ganin et al. (2016), ramping from 0 to ≈1 over the student phase" 수준. **λ_GRL과 λ_rev를 단일 λ로 합쳐 서술 금지** (§9.2). Ganin et al. 2016 인용 필수 (§16 기등재).
- **CRITICAL NOTE (Phase 5)**: warmup ablation 미존재(RESEARCH_SYNTHESIS REQUEST-F). 논문에서 warmup을 독립 기여로 주장하지 않고, "학습 안정화를 위한 단계적 활성화"로 서술하여 ablation 없이도 방어 가능한 수준으로만 언급. **contribution bullet에서도 warmup 문구 제거 완료 (r2, RT BLOCKER-02 — §11 결정 ① bullet 3 참조). SDMAE도 teacher-first 2단계 학습을 쓰므로 warmup은 novelty 재료가 아님.**

**수식 번호**: (4) teacher/student decoder forward, (5) encoder gradient isolation.

### 5.6 §3.5 Label-Guided Training

**내용**:
- 라벨이 개입하는 3지점을 하나의 소절에 통합하여 "왜 3경로가 필요한가"의 논리를 완성.

**(A) Output Discrepancy (OD) Loss — 정상 패치 전용**:
수식: L_OD = (1/|P_n|) Σ_{p∈P_n} ||o_T^p.detach − o_S^p||² where P_n = masked normal patches.
- grl_disable_anomaly_loss=True → anomaly 패치는 OD loss에서 0. student가 정상 구간에서만 teacher를 따르도록 유도.

**(B) Feature Matching (FM) Loss — 훈련 전용 regularizer**:
수식: L_FM = (1/|P_n|·d) Σ_{p∈P_n} ||h_T^p.detach − h_S^p||²
- Adaptive λ_FM = clamp(||∇L_main||/(||∇L_FM||+1e-4), 0, 10) (직전 epoch 값 적용; trainer inline — trainer.py:639–653).
- 추론 점수에 포함하지 않음(scoring.py fm_active=False hardcoded).

**(C) GRL Anomaly Suppression**:
- **(소절 서두 1문장 — RT MAJOR-01, SDMAE 작동 계층 구분을 Method 본문에 명시)**: "SDMAE's anomaly-overlook supervision operates in the target/loss space (anomaly-removed reconstruction GT and loss weighting); our GRL operates in the gradient space of the student's internal representation." — Related Work 각주에 머물지 않고 §3.5 본문에 박는다.
- AnomalyClassifierHead: 2-layer MLP(d_model → d_model//2=256 → 1; LayerNorm + GELU + Dropout(0.1)).
- **student decoder 마지막 층 hidden(output projection 이전)**에 패치별 독립 적용(풀링 없음), masked 패치만 손실 계산 (ADV BLK-002 위치 표기 통일).
- window-mode 타깃: 윈도우 내 anomaly ≥1 이면 모든 masked 패치 target=1.
- focal-style BCE 변형: L_cls = (1/|P_mask|) Σ (1−exp(−BCE))² × BCE_{w+}(logit, y). 표준 focal loss(Lin et al. 2017) 아님 — pos_weight 내장 BCE 기반 변형.
- **표기·차별 지침 (ADV MAJ-004/NOTE-002 — positive 지침)**: 논문 표기는 **"focal-style BCE variant with class-prior pos_weight"**로 통일. 본 변형이 **본 논문에서 설계한 것**임을 1문장 명시하고, Lin et al. 2017과의 차이를 1문장으로 명기: 표준 focal loss는 p_t를 모델 예측 확률(sigmoid(logit) 기반)로 정의하지만, 본 변형은 p_t := exp(−BCE_{w+})로 pos_weight 반영 BCE로부터 유도한다. 예시 문장: "We design a focal-style variant based on BCE with class-prior pos_weight, rather than adopting the standard focal loss [Lin et al. 2017]."
- GRL gradient reversal (**r3 정정, NEW-B1 — 계수 오귀속 교정**): backward에서 gradient × (−**λ_rev**) — λ_rev = Ganin-style sigmoid ramp 2/(1+exp(−10p))−1 (model.py:129–140, trainer.py:1201–1211; §5.5). **λ_GRL_eff는 backward 곱셈 계수가 아니라 손실 항 가중치다**: λ_GRL_eff = λ_GRL_adp × grl_loss_weight(0.2), λ_GRL_adp = clamp(||∇L_main||/(||∇L_GRL||+1e-4), 0, 10) (직전 epoch 값; trainer inline grad-ratio). student hidden에 실제 도달하는 adversarial gradient = −λ_rev × λ_GRL_eff × ∂L_cls/∂(GRL 출력) — 이중 λ의 역할 분리(§5.5 서술 방침 준수).
- **왜 GRL인가 (R10) + 왜 anomaly-OD 제외만으로는 부족한가 (RT MAJOR-05 보강)**: anomaly 패치를 OD loss에서 제외하는 것(A의 분기)은 student가 anomaly 구간에서 teacher를 따를 "의무"를 없앨 뿐, anomaly 패턴을 표현에서 **능동적으로 제거하지 않는다** — 학습 중 반복 노출된 labeled anomaly 패턴을 student가 기억해 잘 복원하는 우회로가 남고, 그러면 discrepancy 신호가 약해진다. GRL은 gradient 부호 반전으로 이 우회로 자체를 차단한다 — teacher 경로를 건드리지 않고 student 표현에서만 anomaly 정보를 제거하는 가장 경제적 메커니즘. 이 "제외 vs 능동 제거" 구분을 §3.5 본문에 명시 서술하고, ablation에서 "w/o GRL (anomaly-OD 제외는 유지)" 변형으로 정량 분리한다 (§6.7).

**총 손실 수식 (ADV MAJ-007 — 기호 통일: GRL 분류 손실은 전 문서에서 L_cls 단일 표기, "L_GRL" 혼용 금지)**:
L_total = L_recon + L_OD + λ_FM_eff × L_FM + λ_GRL_eff × L_cls

- L_recon = masked timestep MSE, teacher output 기준.
- 각 항을 먼저 개별 정의, 마지막에 L_total로 통합 (표준 패턴).

**수식 번호**: (6)–(10) 순서대로 L_OD, L_FM, L_cls, λ adaptive 공식(λ_FM·λ_GRL 공통형), L_total.

### 5.7 §3.6 Anomaly Scoring and Inference

**내용**:
- Leave-one-out inference: 윈도우당 50개 leave-one-out masking 패턴을 batch 차원 확장으로 병렬 forward. **패치 단위 점수** (ADV MAJ-009 — 집계 계층 명시): 각 패치 p의 recon_p (teacher 복원 오차), disc_p = ||o_T^p − o_S^p||².
- **Adaptive scoring** (score_mode='adaptive', per-patch):
  scaled_disc_p = disc_p × (mean_recon + ε)/(mean_disc + ε)  [ε = 1e-4]
  score_p = recon_p + scaled_disc_p / r,  r = 4  (recon:disc = 4:1)
- **Point-level aggregation (별도 수식)**: 각 timestep을 덮는 모든 (window, patch) 쌍의 score_p **평균(mean 집계)** — evaluator.py mean 산식(bincount-합/coverage), EXPERIMENT_PROTOCOL_TRUTH §④-실행 2항과 정합. 수식 (11)(12)는 per-patch, (13)이 patch→point 집계임을 명확히 구분 (ADV MAJ-009).
- 추론 시 라벨 불사용. GRL classifier 비활성 (Fig. 2 "training only" 표기와 정합 — ADV BLK-002).
- **왜 adaptive scaling인가 (R10)**: 데이터셋·피처 수·anomaly 유형에 따라 recon과 disc의 절대 스케일이 크게 달라진다. adaptive scaling 없으면 한 성분이 score를 지배해 다변량 설정 간 일반화 저하.
- **왜 leave-one-out인가 (R10)**: 한 패치의 이상 여부가 다른 패치 score에 간섭하지 않는 독립 평가. 여러 window의 평균은 단일 window noise를 줄인다. 비용(FLOPs ~50×)은 명시적으로 인정.

**수식 번호**: (11) per-patch scaled_disc, (12) per-patch final score, (13) patch→point mean aggregation.

---

## 6. §4 Experiments

### 6.1 소절 구성 (MECE R1 검증)

```
§4.1  Experimental Setup
  §4.1.1  Datasets and Benchmark Protocol
  §4.1.2  Implementation Details
  §4.1.3  Evaluation Metrics
  §4.1.4  Baselines and Comparison Conditions
§4.2  Main Results  (+ protocol-effect 보조 분석, Table 4 — r2 추가)
§4.3  Ablation Study
§4.4  Label Sparsity Analysis (R32)
§4.5  Qualitative Analysis
```

MECE 논거: §4.1은 설정(what), §4.2는 성능(how well; protocol-effect 분리 포함), §4.3은 기여 원천(why), §4.4는 설정 일반성(robustness), §4.5는 정성적 이해(how it works). 겹치는 주장 없음.

**서사 중복 방지 지침 (RT MINOR-04)**: §4.2 분석 텍스트는 **전체 SOTA 비교·데이터셋별 특이점·protocol-effect 분리**에 집중하고, component-level 기여 설명("이게 빠지면 왜 나빠지는가")은 **§4.3 전용**. §4.2에서 "이전 방법 한계와 연결" 서술은 1문장 이내로 최소화.

### 6.2 §4.1.1 Datasets and Benchmark Protocol

**내용**:
- **6 데이터셋 계열** (ADV MAJ-002 — 계열/entity 구분 명확화): SWaT(A1+A2, 45 features), WaDi(A1 123 / A2 123 features — **두 조건은 독립 entity로 Table 1 별도 행**), PSM(25 features), SMD(28 machines, **constant 컬럼 제거 후 machine별 29–36 features** — raw 38은 제거 전 수치, ADV BLK-003), SMAP(54 channels, 25 features), MSL(27 channels, 55 features). 총 113 학습 단위(=1+2+1+28+54+27; SWaT dual-eval 시 평가 단위 114). Simulation/Exathlon 제외(논문 미포함).
- **데이터셋 선택 근거 1문장 (RT MAJOR-06)**: "We focus on the most widely-used multivariate TSAD benchmarks spanning industrial control (SWaT, WaDi), IT infrastructure (PSM, SMD), and spacecraft telemetry (SMAP, MSL) — datasets in which anomalies occur in realistic operational streams, enabling construction of the contaminated semi-supervised setting." (+ 기존 벤치마크의 clean-train 가정 문헌 1–2개 인용 — Phase 4 수요, RT MINOR-01)
- **Contaminated benchmark protocol 설명 (R13 설계 방어 + RT BLOCKER-01 정면 답변 문단)**:
  - 동기: 기존 벤치마크는 train = 정상으로 구성되어 labeled anomaly를 학습에 반영하는 방법을 평가할 수 없음.
  - **"왜 원본 train 라벨만 쓰지 않는가?" 1문단 정면 답변 (배치 위치 확정 — §4.1.1)**: 원본 train split에는 labeled anomaly가 **구조적으로 존재하지 않는다** — SWaT/WaDi 원본 train은 정상 운영 구간(공격 없음), PSM/SMD는 train 라벨 파일 자체가 부재(분야 표준 "전부 정상" 가정), SMAP/MSL은 train 라벨이 명시적 0 (EXPERIMENT_PROTOCOL_TRUTH §①·② 실측). 따라서 원본 split 그대로는 semi-supervised TSAD의 평가가 **정의상 불가능**하며, test 스트림 앞부분의 train 편입이 이 설정을 평가 가능하게 만드는 구조적 필수 장치다 — 이것이 프로토콜의 존재 이유. (상세 방어 구조는 §14.)
  - 구현: 원본 test 시간순 앞 50%를 train에 편입. 분할 규칙 전 데이터셋 통일(// 2). SMAP/MSL safe-cut 메커니즘(anomaly region ±10 clearance, 4채널만 이동, 최대 +166 steps) 명시 — "negligible at concat scale"은 D-16 단일 채널 관점 과장 금지.
  - 시간성: 뒤 50%만 test → look-ahead 없음.
  - train anomaly ratio 실측치(SWaT 1.63%, WaDi A1 0.52%, A2 0.76%, PSM 6.20%, SMAP 0.70%, MSL 1.70%; SMD per-machine 확정 대기) [수치 확정 후 채움].
  - **한계 인정 1문장 (RT BLOCKER-01)**: 편입된 prefix의 anomaly 분포가 보존된 test 후반 50%의 anomaly 분포와 다를 수 있음을 명시적 limitation으로 인정.
- **Table 1** (§4.1.1, ~1/4p): Dataset statistics table. 열: Dataset | #Train pts | #Test pts | #Dimensions | Train AR(%) | Test AR(%) | Source. WaDi A1/A2 별도 행. SWaT 행: "dual eval (full/excl22) — see §4.1.3". 비고: Simulation/Exathlon 미포함.
- SWaT excl22 프로토콜: region #22는 test anomaly 질량의 83.75%를 차지하여 단일 사건이 recall 대부분을 결정 → 모델 변별력이 낮아짐. excl22는 단일 학습 + eval_mask로 region 22 제외. **주 성능 지표는 excl22 기준**, full은 참고 병기. 수치 기준: `A1A2_excl22` entity headline (`metrics.pak_auc_f1 = 0.62899`) — 결정 사안 ③ 확정값 (재실험 시 갱신 조건은 §11 결정 ③ 참조 — ADV MINOR-002).
- 정규화: per-feature min-max, train fit only. multi-entity는 entity별 독립 fit.
- **완주 상태 주석 (ADV MAJ-002)**: 현재 학습 단위 36/113 완료 (평가 단위 37/114 — §0.4, NEW-m1 기준 통일) — Table 1/Table 2의 수치는 잔여 entity 완주 후 채움을 §6.6 Table 2 설계에 명기.

**R10 논증 (프로토콜 방어)**: "왜 test 앞 50%를 train에 편입해야 하는가" — labeled anomaly가 train에 존재해야 force_mask_anomaly, OD loss 분기, GRL 모두 실제로 발동한다. 기존 clean-train 벤치마크에서는 이 세 경로가 전혀 사용되지 않아 제안 방법의 핵심 기여가 평가에서 지워진다. (정면 방어 5논거 전체 구조는 §14.)

### 6.3 §4.1.2 Implementation Details

**내용**:
- 아키텍처: patch_size=10, num_patches=50, masking_ratio=0.15 (8/50 masked), **d_model=512, dim_feedforward=2048 (전 데이터셋·entity 공통 고정 — ADV BLK-003 관련 r2 정정: dynamic 매핑 서술 폐기; 근거는 §5.4)**, num_encoder_layers=4, nhead=8, dropout=0.15, num_teacher_decoder_layers=3, num_student_decoder_layers=2. 입력 차원 F만 데이터셋별 상이(Table 1 / Appendix §C.1).
- 학습: AdamW (betas=(0.9,0.99), lr=1e-3, weight_decay=1e-3), batch_size=1024, num_epochs=500, teacher_only_warmup_epochs=250, linear LR warmup 10 epochs + CosineAnnealingLR, AMP bf16. GRL classifier lr=1e-4(main×0.1). random_seed=42, **단일 run** (baseline 중 `random`만 5-run mean±std — EXPERIMENT_PROTOCOL_TRUTH §④-실행 1항; 분산/신뢰구간 미보고 사실 명시).
- **SWaT 입력 차원 재현성 플래그 (ADV MAJ-003)**: "SWaT A1+A2 입력 차원 45 = 원본 51 − combined-constant 6 컬럼 {P202,P401,P404,P502,P601,P603} 제거. ⚠️ 현 환경의 raw CSV + loader 경로는 51을 반환 — 재실험/재현 시 45 일치 검증 필수 (EXPERIMENT_PROTOCOL_TRUTH FEEDBACK-7)." 본문 또는 Appendix §C.2 전처리 단계에 1줄 명기.
- 추론: leave-one-out (50 masking patterns, batch-parallelized), score_recon_disc_ratio=4.0.
- 하드웨어: [GPU 모델 — 실험 환경 명시]. 코드 공개: [repository URL] (anonymous/TBD).
- **baseline 학습 설정 — 비대칭 사실 공개 + 방어 (ADV BLK-005/MAJ-011, RT NOTE-02)**: "All baselines use their respective best-effort hyperparameters (see Appendix §A.1)." + epoch 수 비대칭을 **그대로 공개**: MAE 500 epochs(eval 5-epoch 간격) / unsupervised 22종 10 epochs(매 epoch eval) / weakly-supervised 4종 50 epochs. batch size 차이도 공개: MAE 1024 vs baseline 512(원 구현/논문 preset 충실 원칙). **방어 1–2문장**: ① 모든 모델이 "주기 평가 후 best-epoch 선택(기준 pak_auc_f1)" 동일 구조 + early stopping 양쪽 부재 → 각자 budget 완주 후 최적 epoch에서 평가됨, ② epoch budget은 모델군별 수렴 특성(대형 MAE의 warmup 250 포함 장기 수렴 vs 소형 baseline의 단기 수렴)에 맞춘 best-effort, ③ (옵션) Appendix §B.4 epoch-sensitivity placeholder로 budget 민감도 제시 가능. §15 방어 시나리오에 등재. **권고 (r3, R2-NOTE-01)**: epoch 비대칭·test-set selection 두 방어 모두 마지막 보루가 optional placeholder §B.4인데, optional placeholder는 rebuttal 무기가 되지 않는다 — REQUEST-4 (iii) validation-split selection sensitivity 소형 실험(1–2 데이터셋) + 대표 baseline epoch-budget 1점 추가(예: 50 epochs 재실행)를 저비용 EXPERIMENT_EXECUTION_TODO 후보로 등재, §B.4를 실측 sensitivity로 격상하는 것이 rebuttal 화력을 실질적으로 바꾼다.
- **test-set model selection 공개 (ADV MAJ-005, RT MAJOR-04 — 숨김 금지)**: "Best epoch selected by pak_auc_f1 on the test split, **uniformly applied to all methods (MAE + all baselines); no separate validation split exists in this protocol.**" — EXPERIMENT_PROTOCOL_TRUTH §④ M-3/REQUEST-4의 명시 공개 의무 이행. 방어 논거는 §15 신설 행 참조 (전 모델 동일 조건 → 비교 공정성 유지; 일반화 추정의 낙관 편향 가능성은 한계로 인정).
- oracle threshold 주의 명시: "Threshold-dependent metrics use anomaly-ratio threshold (test anomaly ratio), **not** the oracle best-F1 threshold, unless explicitly marked as (oracle)." → PA%K-AUC 계열은 threshold sweep 적분이므로 oracle 이슈 없음. PA F1만 F1-최적(oracle) threshold — "(oracle)" 표기 의무 (§6.4).

### 6.4 §4.1.3 Evaluation Metrics

**내용**:
- 5종 지표 공식 서술 (내부 키 병기 — Phase 5 drafter 키 혼동 방지):
  1. PA%K-AUC F1 (`pak_auc_f1`; Kim et al., AAAI 2022): K=0..100 sweep, per-K optimal threshold, trapz 적분. Main 성능 지표 + best-epoch 선정 기준.
  2. PA%K-AUC PR (`pak_auc_prc_auc`): 동일 K sweep, AUC-PR 적분.
  3. VUS-PR / VUS-ROC (`vus_pr`/`vus_roc`; Paparrizos et al., PVLDB 2022): 3D 곡면 볼륨, threshold-free.
  4. Affiliation F1 (Huet et al., KDD 2022): **사용 키 `affiliation_f1_ar` — AR threshold 기반 (evaluator.py:811–813, 키 할당 :813)** (ADV MINOR-004 — F1-최적 threshold 기반 `affiliation_f1`과 혼용 금지; r3 NEW-n1 — 정본(PROTOCOL_TRUTH REQUEST-1) 표기와 라인 통일). threshold 방어: "not oracle".
- PA F1(`pa_0_f1`; K=0, **F1-최적(oracle) threshold**)은 보조 지표 — "(oracle)" 명시 후 병기. 과대평가 위험(Kim et al. 2022 인용) 인정 + PA%K로 대체 논리.
- NRdetector를 "동일 평가 철학 선행 사례"로 1문장 인용.
- SWaT excl22에서는 excl22 기준 지표를 사용함을 명시.

### 6.5 §4.1.4 Baselines and Comparison Conditions

**내용**:
- 22개 비지도 baseline 계층별 제시 (**ADV MAJ-006 — EXPERIMENT_PROTOCOL_TRUTH §③ 분류로 정정: 5+3+1+6+7=22 정합**):
  - Simple 5 (Sarfraz et al., ICML 2024 QuoVadisTAD): random, sensor_range, pca_error, l2_norm, nn_distance.
  - Neural 3 (QuoVadisTAD): mlp, mlpmixer, transformer.
  - **GCN-LSTM 1** (독립 항목 — SOTA Legacy에 포함하지 않음).
  - **SOTA Legacy 6**: Anomaly Transformer, TranAD, USAD, DAGMM*, GDN, OmniAnomaly. (*DAGMM는 "DAGMM (simplified variant, following [TranAD repo])"로 표기, 각주에 "GMM energy 제거" 명시 — RESEARCH_SYNTHESIS §⑥ DAGMM provenance 결정 반영; related work §2.1에서는 원논문 인용만 — ADV NOTE-001.)
  - SOTA New 7: TFMAE, NPSR, TimesNet, DCdetector, MEMTO, ModernTCN, CATCH.
- 4개 weakly-supervised (Q1-only): DeepMIL, WETAS, TreeMIL, NRdetector. **⚠️ GPU 전체 실험 미실행 — Phase 5 진입 전 완주 필수 (EXPERIMENT_EXECUTION_TODO; §0.4)**.
- **비교 조건 확정 (결정 사안 ④)**:
  - **Main table**: Q3 (minmax normalonly) — 비지도 baseline에게 labeled anomaly로 오염원을 제거해주는 "그들에게 가장 유리한 조건". 제안 모델은 contaminated full train으로 학습(labeled anomaly 제거 없이).
  - **train 데이터 양적 비대칭 인정 (RT MAJOR-03)**: Q3에서 비지도 baseline의 train은 anomaly 구간 절제로 제안 방법의 full contaminated train보다 **양적으로 작다**(절제분 = train anomaly ratio, **실측 완료 데이터셋 기준 0.5–6.2%; SMD per-machine 확정 대기** — r3 R2-MIN-04: §5.2의 "전체 상한 단정 금지" 어법과 통일 + 경계 windowing 손실). 이 양적 차이가 이득을 과대평가할 가능성을 본문 1문장으로 인정하고, 전체 오염 데이터로 학습한 Q1 조건(Appendix §A.2)과 §4.2 protocol-effect 분석(Table 4)이 보완 비교를 제공함을 명시.
  - Q1(full)을 Appendix §A.2에 병기하여 contamination이 비지도에 미치는 영향을 정성적으로 분석.
  - weakly-supervised 4종: Q1에서 실험. "(Q3 = N/A: structurally incompatible — no positive training windows)" 1줄 footnote.
- 단일 baseline 설명 길이: 계열명 1문장 + 대표 인용. 개별 논문 소개 없음(R19).

### 6.6 §4.2 Main Results

**내용**:
- **Table 2** (main results, full-width, ~3/4p): 행 = 비교 방법(계층 구분선), 열 = 데이터셋 × **{PA%K-AUC F1, VUS-PR} 2개 지표로 고정** — 나머지 3개 지표(VUS-ROC, Affiliation F1, PA%K-AUC PR)는 Appendix §A.3 전수 표로 위임 (RT V3 — "지면 허용 시" 위임 문구 삭제, 열 구성 확정; 5지표 강조와의 정합은 §4.1.3에서 "주 표 2지표 + 전수는 Appendix" 명시로 처리). SWaT 열은 excl22. Bold = 최고, Underline = 2위. 우리 모델 행은 맨 아래, "w/ Q3 protocol" 주석. **수치는 전 entity 완주 후 채움 (placeholder — §0.4)**.
- **Protocol-effect 보조 분석 (r2 신설 — RT BLOCKER-03: 방법론 효과 vs 프로토콜 효과 분리, main text 격상)**:
  - **Table 4** (small, half-width, ~0.2p): 행 = {[MODEL], 대표 비지도 baseline 2–3종}; 열 = 대표 2–3 데이터셋 × 2 조건: **(i) standard split** — 원본 train만 사용(test-prefix 미편입; train에 labeled anomaly 없음 → 제안 방법의 3개 라벨 경로 휴면 = 사실상 비지도 모드), **(ii) contaminated** (main protocol). **두 조건 모두 평가는 동일한 원본 test 뒤 50%** — test 통일로 train 구성 차이만 분리.
  - **실행 사양 (r3 신설, R2-MIN-03 — 코드 근거 인용)**: ① standard-split 조건의 제안 방법은 **동일 config 그대로(use_grl=True 유지) 실행** — 라벨 0인 train에서 세 라벨 경로는 코드 수준에서 자가 비활성화된다: force_mask_anomaly priority 전부 0 → 무작위 마스킹으로 자연 퇴화, OD 분기는 전 패치 정상(정상 전용과 동일), **GRL은 batch 단위 positive 부재 시 손실 자체가 계산되지 않음 (`loss.py:293–302` `_pos_count == 0 → grl_cls_loss_tensor=None` skip)**. ⚠️ use_grl=False로 끄는 선택 금지 — §6.7이 경고한 dead-component(dynamic margin anomaly loss) 재활성화 함정에 빠짐 (ablation과 동일한 함정 경고를 Table 4에도 적용). ② contaminated 조건에서 비지도 baseline이 받는 train 데이터 = **Q3(normalonly)** 명시 (main protocol과 동일 — 표 caption "(ii) contaminated (main protocol)"의 암묵 표기를 명시화). 두 설계 조건을 EXPERIMENT_EXECUTION_TODO 항목 3에 추가 (p3_fixlog_r3 집계 반영).
  - **2단 논증 구조**: ① 동일 방법이 표준(clean-train) 조건에서도 비지도 SOTA와 경쟁력 유지 — 성능이 프로토콜 산물이 아니라 방법 자체의 가치임을 보임. ② labeled anomaly가 제공되는 contaminated 조건에서 제안 방법만 추가 이득 — 비지도 baseline은 같은 데이터 추가에도 라벨을 활용하지 못함. "성능 우위가 더 많은 데이터(prefix) 때문인가, GRL+distillation 때문인가"라는 reviewer 질문에 정면 답변.
  - ⚠️ **EXPERIMENT_EXECUTION_TODO**: standard-split 조건의 제안 방법·baseline 실험 미실행 — Phase 5 진입 전 실행 필요 (대표 2–3 데이터셋 한정으로 비용 통제).
  - §4.4 label sparsity의 p→0 극한과 상호 참조 (비지도 극한의 또 다른 절단면).
- 분석 텍스트 4구조 (RT MINOR-04 반영):
  1. "As shown in Table 2, [MODEL] achieves [요약 주장]..."
  2. 데이터셋별 특이 결과 2–3개 (SWaT excl22 해석, PSM 고비율 이상 등).
  3. **Protocol-effect 분리 해석 (Table 4)** — "이전 방법 한계와 연결" 서술은 1문장 이내로 축소, component 기여 설명은 §4.3 전속.
  4. 한계 인정: leave-one-out으로 인한 연산 비용을 인정 후 다음 소절 ablation으로 연결.
- Rank 집계(optional, Appendix): 데이터셋별 ranking 집계 테이블 — VUS/PA%K/Affiliation 지표군 병기.

### 6.7 §4.3 Ablation Study

**내용**:
- **Table 3** (ablation, half-width, ~1/3p): 행 = 모델 변형, 열 = 주요 3–4 데이터셋 × PA%K-AUC F1.
- 변형 구성 (7행):
  1. Full model ([MODEL])
  2. **w/o GRL** — **변형 정의 명확화 (RT MAJOR-05)**: GRL classifier+reversal 제거하되 **anomaly 패치의 OD-loss 제외는 유지** (= "w/o GRL, with anomaly-OD still disabled") — GRL의 순효과(능동 표현 억제)만 분리. ⚠️ 정의 주의: 코드에서 use_grl=False 단독 설정 시 dead-component(dynamic margin anomaly loss)가 재활성화되어 비교가 오염됨 — ablation 실험 config에서 anomaly-loss 경로 차단을 명시적으로 유지할 것 (EXPERIMENT_EXECUTION_TODO 설계 조건).
  3. w/o force_mask_anomaly: anomaly-first masking 제거
  4. w/o OD Loss: output discrepancy loss 제거(L_OD=0)
  5. w/o FM Loss: feature matching regularizer 제거 — **미실행 placeholder (r3, R2-MAJ-01)**: 정본상 "FM 제외 ablation 결과 없음"(RESEARCH_SYNTHESIS 표A) — §0.4 실행 등재. 행 6과 동일한 conditional 규칙 적용(미완 시 본문 잔류 금지); §12 FM 행의 "ablation 근거 필요(미존재 → REQUEST)"의 정량 해소처.
  6. w/o Teacher Warmup: warmup=0 (teacher/student 동시 학습) — **placeholder 유지하되 명시적 conditional (RT MAJOR-10/ADV MAJ-010 절충)**: REQUEST-F 실험이 Phase 5 진입 전 완료되면 본 행 유지, 미완료 시 본 행을 **삭제하고 Appendix §B.1로 강등 또는 전체 생략** — drafter가 placeholder 행을 미완 상태로 본문에 남기는 것 금지. warmup은 contribution이 아니므로(§5.5, §11 결정 ①) 행 삭제가 논증 완결성을 훼손하지 않음.
  7. Symmetric decoder (Teacher 2L/Student 2L): capacity gap 제거 — **미실행 + load-bearing (r3, R2-MAJ-01)**: 정본상 decoder-depth ablation "필요"(미존재; RESEARCH_SYNTHESIS 표A). contribution bullet 3(capacity-gap 재구성, RT B-02)의 **유일한 정량 근거**이므로 **Phase 5 진입 전 실행 필수** (§0.4 등재 — warmup 공격 패턴의 bullet 3 재발 차단). 만약 미완 상태 진입이 불가피하면 행 6 규칙(본문 잔류 금지) 적용 + **bullet 3의 "a reliable anomaly signal" 주장을 정성 수준("intended to provide")으로 하향**하는 drafter 지침 동반.
- 각 행의 논증: "이것을 제거하면 왜 성능이 떨어지는가" → §3의 R10 논증과 직접 연결. **component-level 서사는 이 소절 전속 (§4.2와 중복 금지 — RT MINOR-04)**.

### 6.8 §4.4 Label Sparsity Analysis (R32)

**내용**:
- **동기 (R11 일반 케이스 검증)**: main 실험은 label 가용성 상한 케이스. 실제 환경에서는 labeled anomaly 비율이 낮아 일부 anomaly가 unlabeled 상태로 train에 잔류 — 이것이 R11 설정의 일반 케이스다 (§3.1의 3단 구조와 정합 — RT MAJOR-09).
- **실험 설계**: labeled anomaly 비율 p ∈ {1.0, 0.75, 0.5, 0.25, 0.1} sweep. p=1.0이 main 설정, p→0이 비지도 극한.
- **왜 강건한가의 논리 (R32 요구 — 선험적 설명)**:
  1. force_mask_anomaly는 labeled anomaly 패치만 우선 마스킹 → unlabeled anomaly 패치는 무작위 마스킹에서 낮은 확률로만 마스킹 → teacher 재구성 오차 성분은 라벨 비율에 덜 민감.
  2. GRL은 labeled 샘플에 대해서만 suppression signal이 활성화 → unlabeled anomaly는 student가 잘 복원할 수 있지만, 그 결과 discrepancy가 줄어드는 것은 labeled 쪽에서만 발생 → overall discrepancy 신호의 양적 감소이지 방향 교란 아님.
  3. MAE 재구성 오차 자체는 label-free 자기지도 신호 — anomaly 위치에서 복원이 더 어려워지는 특성은 라벨 비율과 무관.
  4. 따라서 p 감소 시 discrepancy 성분의 신호 강도는 줄지만 recon 성분은 유지 → 전체 성능의 연속적 감소(급격 붕괴 없음).
- **Fig. 3** (~1/4p): X축 = labeled anomaly 비율 p, Y축 = PA%K-AUC F1. 2–3 데이터셋 선 오버레이. placeholder.
- NRdetector의 label-noise sweep(Table 4) 패턴을 참고하되, sweep 축의 의미가 다름(라벨 희소율 vs 세그먼트 노이즈율)을 1문장 구분.
- **코드 근거**: NoisyLabelSlidingWindowDataset + apply_normal50_noise 메커니즘이 존재 → sweep 파라미터 설계 입력.

### 6.9 §4.5 Qualitative Analysis

**내용**:
- **Fig. 4** (full-width, ~1/4–1/3p): anomaly score 시각화. 행: (1) 입력 시계열 + ground truth, (2) Teacher 재구성 오차, (3) Teacher-Student discrepancy, (4) 최종 합산 score. 열: 최소 2 데이터셋(SWaT excl22, WaDi A1 또는 PSM 중 1). 붉은 영역 = anomaly, 점선 = AR threshold.
- 서술: discrepancy와 recon 두 성분이 각각 어떤 유형의 이상에 더 민감한지 1–2 문장 해석. **(RT MINOR-02 — EXPERIMENT_EXECUTION_TODO 조건부)**: 이 해석은 실제 실험 결과(이상 유형/사건별 성공·실패 케이스)에 근거해야 하며 수치 확정 전 작성 금지. SWaT excl22는 region 22 제외 후 소형 사건들 위주이므로, 시각화 구간 선택 시 사건 규모·유형 대표성을 확인할 것.

---

## 7. §5 Conclusion

**내용 (1단락, ~0.3p)**:
- 요약: 문제(labeled anomaly 미활용) → 방법(3경로 통합 MAE + Teacher-Student + GRL) → 프로토콜(contaminated benchmark) → 결과(6 dataset SOTA + label sparsity robustness).
- 한계 1문장: 50×FLOPs inference 비용. Complementary masking(7-pass)을 경감 방향으로 언급하되 **"코드에 구현되어 있으나 본 실험에서는 미사용(eval_complementary_masking=False) — 향후 연구에서 cost-accuracy tradeoff 탐색 가능"** 수식어 필수 (ADV MAJ-012 — 미사용 옵션을 검증된 경감책처럼 서술 금지).
- 향후 연구 1문장: 완전 unsupervised 설정으로의 graceful degradation(GRL 비활성화), 더 큰 피처 수에서의 확장.

---

## 8. Appendix 구성 계획 (R7)

```
Appendix A: Full Results and Additional Tables
  §A.1  Baseline Hyperparameters (데이터셋별 상세 설정 전수)
  §A.2  Q1 (Full-Train) Condition Results (메인에서 Q3 단독 → Q1 보조 병기)
  §A.3  Full Multi-Metric Results (VUS-ROC, Affiliation F1, PA%K-AUC PR 등 전수 — Table 2는 PA%K-AUC F1 + VUS-PR 2지표 고정)
  §A.4  Per-Entity SMD / SMAP / MSL Results (28+54+27 entity 전수)
  §A.5  SWaT Full vs Excl22 상세 비교

Appendix B: Additional Analysis
  §B.1  Ablation — Teacher Decoder Layer Count Sensitivity (3L vs 2L vs 1L — placeholder)
  §B.2  Parameter Sensitivity (score_recon_disc_ratio, masking_ratio)
  §B.3  Computational Cost (inference FLOPs, wall-clock, memory)
  §B.4  Epoch-Budget Sensitivity (placeholder, optional — ADV BLK-005 방어 보조: baseline epoch budget 변화의 성능 영향)

Appendix C: Method Details
  §C.1  Input Dimensionality Table (데이터셋/entity별 F 전수 — SWaT 45=51−6 constant, SMD 29–36=raw 38−constant, WaDi 123=127−4 NaN; d_model=512 전 entity 공통 명기) ← r2 정정: 구 "Dynamic d_model Mapping Table" 폐기 (271은 d_model 고정)
  §C.2  Training Procedure Pseudocode (+ SWaT 45-feature 전처리 단계 명기 — ADV MAJ-003)
  §C.3  Notation Summary Table
```

**Appendix 전략 (R7)**: 본문 9p에서 공간이 부족한 내용(per-entity 전수, Q1 조건, 다중 지표 전수)을 Appendix에 위임하여 본문을 핵심 기여와 주요 데이터셋 결과에 집중. Reviewer가 supplementary에서 재현성을 확인할 수 있도록 hyperparameter 전수와 pseudocode를 확보.

---

## 9. Notation 설계 방침 (R5)

### 9.1 핵심 기호 체계 초안 (일반적, 이해 쉬움)

| 기호 | 의미 | 도입 위치 |
|------|------|---------|
| X ∈ R^{T×F} | 다변량 시계열 (T 타임스텝, F 피처) | §3.1 |
| W ∈ R^{L×F} | 슬라이딩 윈도우 (L=500) | §3.1 |
| P_i ∈ R^{s×F} | i번째 패치 (s=10) | §3.1 |
| N | 패치 수 (=50) | §3.1 |
| z_i ∈ R^d | 패치 i의 임베딩 (d = d_model = **512, 전 entity 공통 고정** — r2 정정) | §3.3 |
| M ⊂ {1,...,N} | 마스킹된 패치 인덱스 집합 (|M|=8) | §3.3 |
| V = {1,...,N} \ M | 가시 패치 인덱스 | §3.3 |
| h_T^i, h_S^i | Teacher/Student decoder hidden at patch i | §3.4 |
| o_T^i, o_S^i | Teacher/Student output at patch i | §3.4 |
| y^w ∈ {0,1} | 윈도우 단위 anomaly label | §3.5 |
| y^p_i ∈ {0,1} | 패치 i의 anomaly label | §3.5 |
| P_n | masked normal patches: {i∈M : y^p_i=0} | §3.5 |
| L_recon, L_OD, L_FM, L_cls | 손실 항 (GRL 분류 손실은 **L_cls 단일 표기** — "L_GRL" 혼용 금지, ADV MAJ-007) | §3.5 |
| λ_FM, λ_GRL | adaptive **loss weights** (trainer inline grad-ratio; λ_GRL은 L_cls에 곱해짐 — 반전 계수 아님, r3) | §3.5 |
| λ_rev | GRL gradient **반전 계수** (Ganin et al. 2016 sigmoid schedule, student-phase 동안 0→≈1 ramp; 손실 가중치 λ_GRL과 별개 — r3 NEW-B1) | §3.4/§3.5 |
| s_t | 타임스텝 t의 최종 anomaly score | §3.6 |

### 9.2 수식 금지 사항

- 발표자료 notation 비계승 (RESEARCH_SYNTHESIS §③ 표A와 정합, 발표 수식 형태 불가).
- 기호 재정의 없음 — 최초 등장 시 정의, 이후 일관 사용. GRL 분류 손실 = L_cls 단일 표기 (총손실 수식 포함 — ADV MAJ-007).
- 논문에서 "standard focal loss"(Lin et al. 2017) 표기 금지 — **"focal-style BCE variant with class-prior pos_weight"**로 표기 (positive 지침). 차이 1문장 명기: 표준 focal은 p_t=모델 예측 확률, 본 변형은 p_t:=exp(−BCE_{w+}) (ADV MAJ-004). 본 변형이 본 논문 설계임을 명시 (ADV NOTE-002).
- GRL adaptive λ: trainer inline grad-ratio로 표기 — "VQGAN-style" 귀속 금지 (Phase 1 BLK-001 정정 반영). **(r3 교체, NEW-B1 — r2의 "sigmoid 서술 금지 — 271 미사용" 조항 철회)**: sigmoid schedule을 **손실 가중치**로 서술하는 것 금지(손실 가중치는 grad-ratio adaptive × 0.2); 단 **반전 계수 λ_rev로는 Ganin schedule(2/(1+e^{−10p})−1)이 271에서 실제 사용** — method에 사용 사실을 명기할 것 (§5.5 서술 방침). λ_GRL과 λ_rev를 단일 λ로 합쳐 서술하는 것 금지.
- AnomalyClassifierHead = "2-layer MLP" 표기 (1-layer 표기 금지, Phase 1 MAJ-004 반영).
- **d_model "dynamic" 표기 금지** — 271은 전 entity d_model=512 고정 (r2 정정, §5.4 근거).

---

## 10. 모델명 / 논문 제목 후보 (R15)

**명명 원칙**: novelty 부각(contaminated semi-supervised + self-distillation MAE + GRL), 불필요 신규 약어 남발 금지, 직관적, 기억하기 쉬운 이름.

### 10.1 모델명 후보

| 후보 | 장점 | 단점 |
|------|------|------|
| **CSMAD** (Contaminated Semi-supervised Masked Anomaly Detector) | 설정 차별점 명시, TSAD 분야 관례와 유사 | 음절 길음, "Contaminated"가 생소할 수 있음 |
| **SemiMAD** (Semi-supervised Masked Anomaly Detector) | 직관적, 설정 명시 | 설정명만 나열, GRL/distillation 숨음 |
| **AnoGRL** | GRL 핵심 기여 부각 | MAE/distillation 맥락 없음, 범용성 낮음 |
| **GLAD** (Gradient-reversal Label-Aware Distillation detector) | GRL + label awareness + distillation 3축 포함, 기억하기 쉬움 | "Distillation" 모호 |

> **r2 제거**: ~~TS-SDMAE (Time-Series Self-Distilled MAE)~~ — R9 정면 위반 + SDMAE와의 naming conflict 위험 (RT MINOR-05 / ADV NOTE-003). 후보 목록에서 **확정 제외** — DECISION_LOG 전사 필요.

**확정 (D-007, 2026-06-11)**: **CSMAD** (차순위 SemiMAD) — DECISION_LOG D-007 참조. Phase 5는 D-007 기준.

### 10.2 논문 제목 후보

| 후보 | 장점 | 단점 |
|------|------|------|
| **Self-Distilled Masked Autoencoders for Semi-supervised Multivariate Time Series Anomaly Detection** | 설정·방법·도메인 3축 포함; SDMAE 계보 자연 수용 | "Self-Distilled MAE" 때문에 SDMAE와 유사감 |
| **Label-Aware Masked Autoencoding with Gradient Reversal for Multivariate Time Series Anomaly Detection** | GRL + label awareness 강조; SDMAE와 차별화 | 제목 길이 과다 |
| **Contaminated-Train Anomaly Detection via Asymmetric Self-Distillation and Gradient Reversal** | 프로토콜 + 방법 양쪽 강조 | "Contaminated-Train"이 다소 기술적 |
| **Leveraging Sparse Anomaly Labels in Masked Autoencoders via Gradient Reversal for Multivariate Time Series** | label sparsity + GRL + MAE + 도메인 포함 | 길이 과다 |
| **Semi-supervised Multivariate Time Series Anomaly Detection with Self-Distilled Masked Autoencoders** | 설정 우선, 방법 후술; 간결 | SDMAE 유사감 잠재 |

**확정 (D-007, 2026-06-11)**: **후보 2** "Label-Aware Masked Autoencoding with Gradient Reversal for Multivariate Time Series Anomaly Detection" (차순위 후보 5). 사유: 후보 1·5는 'Self-Distilled MAE' 전면 배치로 R9 위험 최대화 — 기각. 최종 자구는 Phase 5에서 R9 제약 내 미세 조정 허용. DECISION_LOG D-007 참조.

---

## 11. 결정 사안 처리 (명시적 결정 + 사유)

### 결정 ① C1–C4 contribution 구조 채택/수정/기각

| Notion 항목 | 채택/수정/기각 | 사유 | 재설계안 |
|-----------|------------|------|--------|
| C1 (Context-Aware Masking) | **수정** | MAE 자체의 잘 알려진 기여이지, 본 논문의 핵심 novelty가 아님. force_mask_anomaly가 C1 안에 숨어 있어 novel 기여가 희석됨. | contribution bullet에서 독립 항목으로 두지 않고, "force_mask_anomaly를 통한 labeled anomaly-priority masking"으로 §3.3에 흡수 |
| C2 (Capacity-Gap Self-Distillation) | **수정** | 비대칭 decoder 자체는 SDMAE와 개념 공유(단 SDMAE는 branch-off 구조 — 구조는 상이) — novelty는 "GRL과의 결합으로 discrepancy 신호가 semi-supervised 설정에서 증폭되는 것"이 핵심. | GRL과 묶어 "label-guided discrepancy amplification"으로 재구성 |
| C3 (Discrepancy + FM) | **기각(독립 항목으로)** | FM은 score에서 제외, training regularizer만 — contribution으로 전면에 내세우면 reviewer가 "FM이 contribution인데 score에 없다"고 지적. OD loss는 C2/GRL과 묶어 처리. | C2+C3을 합쳐 "asymmetric distillation + GRL-driven discrepancy amplification" 단일 bullet |
| C4 (Semi-supervised via GRL) | **채택 + 확장** | GRL이 본 논문의 novelty 핵심. 단, C4만으로 부족 — contaminated benchmark protocol도 기여임. | C4를 두 부분으로 확장: (i) GRL-기반 표현 억제 메커니즘, (ii) contaminated benchmark protocol의 설계 |

**재설계된 Contribution 구조 (4 bullet — r2: bullet 3 warmup 제거 + 경계 재정의, RT BLOCKER-02/MAJOR-07, ADV MAJ-010)**:

1. **[설정 + 프로토콜]** "We formalize the *contaminated semi-supervised* setting for multivariate TSAD, where labeled anomalies coexist with unlabeled data in training. We design a benchmark protocol that incorporates the test-prefix into training, enabling evaluation of methods that exploit labeled anomalies — a gap absent in standard benchmarks."

2. **[핵심 메커니즘 — 라벨 신호의 주입]** "We propose [MODEL], which integrates labeled anomalies into MAE representation learning via three orthogonal paths: (i) anomaly-priority masking that prevents evasion of hard reconstruction positions; (ii) loss bifurcation that steers the student toward normal-only mimicry; and (iii) gradient reversal that adversarially suppresses anomaly-specific information from the student's representation."

3. **[아키텍처 — 신호가 발현되는 구조]** "We design an asymmetric Teacher(3L)–Student(2L) decoder architecture in which a deeper teacher establishes a stable normal-reconstruction reference while a capacity-limited student's mimicry fails preferentially on anomalous correlation patterns — making the teacher–student discrepancy a reliable anomaly signal under contaminated training." (~~trained with teacher-only warmup~~ — **r2 삭제**: warmup ablation 부재(REQUEST-F) + SDMAE도 teacher-first 2단계 학습 사용 → novelty 없음·ablation 요구 빌미. warmup은 §3.4의 학습 안정화 장치로만 서술.)

4. **[실험]** "Extensive experiments on six multivariate datasets demonstrate state-of-the-art performance under five rigorous metrics. The model maintains robust detection under label sparsity, validating the framework under the general semi-supervised assumption."

**MECE 검증 (R1 — r2 경계 명문화, RT MAJOR-07)**: 1 = 설정·프로토콜 기여, 2 = **라벨 신호 주입 메커니즘**(라벨이 학습 신호로 개입하는 3경로 — 라벨 의존), 3 = **신호가 발현되는 라벨-무관 구조 기반**(비대칭 capacity gap이 discrepancy 신호의 신뢰성을 만드는 아키텍처 — 라벨 없이도 정의됨), 4 = 실험 기여. bullet 2와 3의 경계 = "라벨 신호의 주입(2) vs 그 신호가 작동하는 구조적 기판(3)" — 중복 없음; 합집합이 논문 기여 전체를 커버.

### 결정 ② Setting 명칭

**확정**: **"contaminated semi-supervised"** 사용.

**사유**:
- "semi-supervised"를 대표 명칭으로 사용하되, 더 정밀하게는 labeled anomaly가 소수 섞인 contaminated train 구조를 "contaminated semi-supervised"로 정식화.
- "PU learning"은 strict PU(positive + unlabeled only) 정의와 main 271 구현(train 라벨 전부 존재) 간 괴리가 있어 primary 명칭으로 부적절. 단 §2.2에서 PU learning과의 관계를 1–2문장 명시(라벨 희소화 sweep이 PU-like 일반 케이스에 해당함).
- "weakly supervised"는 설정의 label granularity(point/window level)를 고려하면 적절할 수 있으나, NRdetector의 "약지도"와 구분이 어려워 혼동 위험.
- "contaminated semi-supervised"는 본 논문의 train protocol 특성(오염된 train에서 소수 labeled anomaly 활용)을 정확히 서술하며, 기존 문헌에서 지배적으로 쓰인 명칭이 아니어서 새로운 포지셔닝이 된다.
- **방어 조건 (RT MAJOR-09)**: main 실험이 label 가용성 상한 케이스임을 §3.1에서 명확히 하는 것이 이 명명의 성립 조건 — §5.2의 3단 구조(②-1/②-2/②-3) 서술 필수.
- **Phase 4 검증 항목 (ADV MINOR-005)**: "기존 문헌에서 지배적으로 쓰인 명칭이 아니다"는 주장은 미검증 — Phase 4에서 "contaminated semi-supervised (time series) anomaly detection" 용어 검색으로 기존 사용 사례 확인 필수. 기존 특정 의미 사용 발견 시 명명 재검토.
- NRdetector dossier의 스코핑 주의(R20: "심층 표현 학습과 통합된 PU/SSL 기반 다변량 TSAD는 거의 없다"로 정밀 스코핑)를 반영.

### 결정 ③ SWaT excl22 수치 기준

**확정**: **`A1A2_excl22` entity headline** (`metrics.pak_auc_f1 = 0.62899`) 사용.

**사유**: excl22 전용 entity가 excl22 기준으로 best-epoch을 선정하므로, 이 entity의 headline 수치가 "excl22 조건에서의 최적 성능"이다. `A1A2_full` metadata의 `metrics_excl_region22.pak_auc_f1 = 0.62730`은 full entity best-epoch 기준의 excl22 수치로, 다른 epoch 선정 기준의 결과물이다. 논문에서 SWaT excl22 성능을 보고할 때는 excl22-dedicated entity 수치(0.62899)를 사용하는 것이 fair. **혼용 절대 금지** — 어느 쪽을 쓰는지 한 번만 결정하고 전체 논문·Appendix에서 일관 사용.

**갱신 조건 (ADV MINOR-002)**: 이 수치는 현재 271 실험의 SWaT excl22 결과 기준. SWaT 입력 차원 45의 재현성 플래그(EXPERIMENT_PROTOCOL_TRUTH FEEDBACK-7)가 해소되지 않은 상태에서 SWaT 재실험이 수행되면 수치가 변동될 수 있음 — 재실험 시 본 기준값 업데이트 필수 (선정 원칙: excl22-dedicated entity headline은 유지).

### 결정 ④ 비교표 main 조건

**확정**: **Q3 (minmax normalonly) 단독을 main table로**, Q1은 Appendix §A.2에 병기. **+ (r2 추가, RT BLOCKER-03) §4.2에 protocol-effect 보조 분석(Table 4: standard split vs contaminated, 제안 방법+대표 baseline) main text 배치** — Q3 단독 main table의 "방법론 vs 프로토콜 효과 분리 불가" 공격에 대한 구조적 보완.

**사유**: Q3은 비지도 baseline에게 labeled anomaly로 오염원을 제거해주는 가장 유리한 조건 → "같은 라벨을 각자 패러다임에서 최선으로 쓴 비교"라는 공정성 서사가 성립. Q1(무처리 오염)을 main table로 쓰면 비지도 baseline에게 불리한 조건을 설정한 것처럼 보여 공정성 시비. Q1은 "contamination sensitivity"를 보여주는 보조 분석으로 Appendix에서 가치가 있음. 단 Q3 단독으로는 "성능 우위가 prefix 데이터 추가 때문인지 방법론 때문인지" 분리가 불가능하므로(RT BLOCKER-03), 표준 split 조건의 소형 비교(Table 4)를 main text에 추가하여 2단 논증(표준 조건 경쟁력 + 라벨 활용 이득)을 완성한다. train 양적 비대칭(Q3 절제분)도 §4.1.4에서 1문장 인정 (RT MAJOR-03).

### 결정 ⑤ R9 포지셔닝 옵션

**확정**: **옵션 C 채택** + Method 내 각주 1개 추가(옵션 B의 각주 방식) + **작동 계층 차이는 §3.5 본문 1문장으로 분산 배치 (r2, RT MAJOR-01/MAJOR-08)**.

**사유**: Related Work §2.3에서 distillation 계보 흐름 내(전통 KD → AD용 KD → SDMAE → 본 논문) 자연 언급. 차이 나열 없음(R9). 언어 수준: "extend analogous principles" 대신 **"adapt this architectural paradigm" / "apply the time-series counterpart"** 계열 — SDMAE를 parent가 아닌 sibling으로 포지셔닝 (RT MAJOR-08). Method §3.4의 self-distillation 정의 근처 각주: "We follow the self-distillation terminology of [SDMAE], adapting it to time series; unlike SDMAE, whose student decoder branches off from the teacher decoder, our teacher and student decoders are independent; the student is additionally trained to exclude anomaly-specific representations via gradient reversal, a mechanism absent in SDMAE's unsupervised video setting." (R21 방어 — 용어 계보 + 구조 차이; ADV MAJ-001의 branch-off 사실 정정 반영). 용어 계보 서술: "self-distillation" 명칭은 Zhang et al. [TPAMI 2022]이 원류, SDMAE가 AD에 적용, 본 논문이 시계열 적응 — "applying/adapting" 표현 사용(coining 표기 금지).

### 결정 ⑥ 코드 공개 문구

**포함하되 조건부**: "Code is available at [URL] (to be released upon acceptance)." — 논문에서 URL placeholder로 포함. 단, RESEARCH_SYNTHESIS §⑦의 공개 전 checklist(branch 정리, 범위 정리, secret 스캔, 재현 진입점 문서화) 해소 전에 URL을 확정하지 않음. Phase 5 본문 작성 전 사용자 확인 필요.

### 결정 ⑦ DAGMM 표기 확정 (r2 신설 — RT NOTE-01: DECISION_LOG 기록 필요 사항의 명시화)

**확정**: baseline 표기 = **"DAGMM (simplified variant, following [TranAD repo])"** + 각주 "GMM energy 제거" (§4.1.4·Appendix §A.1). Related Work §2.1의 클러스터 인용은 원논문(Zong et al., ICLR 2018)만 지칭 — variant 언급 금지 (ADV NOTE-001). 무수식 "DAGMM" 단독 표기는 방법 재정의 reject 사유(RESEARCH_SYNTHESIS §⑥ NOTE-003)이므로 금지. **Phase 4 진입 전 DECISION_LOG 전사 필수.**

### 결정 ⑧ 모델명 후보 TS-SDMAE 제외 (r2 신설 — RT MINOR-05 / ADV NOTE-003)

**확정**: TS-SDMAE를 모델명 후보에서 제외 (§10.1). 사유: R9 정면 위반 + SDMAE와의 명명 혼동/naming conflict 위험 (Elsevier 심사에서 "이름이 SDMAE와 너무 유사" 지적 가능). **DECISION_LOG 전사 필수.**

---

## 12. R10 논증 배치 전수표 (각 component의 "왜 다변량 시계열에서 이래야만 하는가")

| Component | 논증 강도 | 배치 위치 | 논증 요지 |
|-----------|---------|---------|--------|
| Linear patchify | 중간(보강 완료) | §3.3 | 한 패치가 다변량 피처의 시간 구간을 포착; flatten linear로 채널 간 선형 결합이 임베딩에 직접 반영; patch_cnn ablation이 있으면 보강 가능 |
| Patch masking + force_mask | 강함 | §3.3 | 다변량 상관 구조 학습 강제; anomaly-class imbalance 직접 대응 |
| Transformer encoder | 중간(보강 완료) | §3.4 | 패치 간 장거리 의존성(이상 전파 패턴) self-attention으로 포착; positional encoding으로 masked/visible 위치 관계 유지 |
| Asymmetric teacher/student | 중간 + 보강 완료 | §3.4 | 다변량 정상 상관 구조를 낮은 capacity로 모방 시 비정상 이탈에서 격차 커짐 |
| Teacher-only warmup | 낮음(정성적만) | §3.4 | 불안정 teacher → noisy discrepancy 신호 방지; 정성 학습 곡선 근거만. **contribution 서술 금지 (r2)** |
| OD loss (정상 패치만) | 강함 | §3.5 | 정상에서 낮은 discrepancy 유도 → 이상-정상 대비(contrast) 증폭 |
| GRL | 강함 | §3.5 | labeled anomaly 정보를 gradient 부호 반전으로 표현에서 제거; "anomaly-OD 제외(수동 회피)"와 "GRL(능동 제거)"의 구분 논증 포함 (RT MAJOR-05); encoder는 GRL gradient로부터 완전 차단 |
| FM loss | 중간 | §3.5 | hidden 공간 collapsing 방지 regularizer; ablation 근거 필요(미존재 → REQUEST) |
| Adaptive scoring | 강함 | §3.6 | 데이터셋 간 recon/disc 스케일 이질성 → adaptive scaling 없으면 일반화 저하 |
| Leave-one-out inference | 강함 | §3.6 | 패치 간 간섭 없는 독립 평가; window 평균의 noise 감소 효과; 비용 명시 |

**논리 보강 필요 4건 처리 (RESEARCH_SYNTHESIS 표A 원재료)**:
- Linear patchify: patch_cnn ablation 결과 없음 → "최소한 CNN과 동등함을 ablation Table 3에서 정량화" 목표. 미존재 시 "we use linear patchify following [MAE 원류]" + "학습 효율과 구현 단순성" 논리로 방어 (RT R10-1 검토 — 방향 유지 확인).
- Transformer encoder Pre-Norm: "Pre-Norm이 긴 시계열 학습의 안정성에 기여"라는 일반 선행(시계열 transformer 논문 인용)으로 보강.
- Teacher/student 층 수(3L vs 2L): ablation Table 3 §B.1에서 3L/2L/1L 비교로 정량화.
- FM loss: ablation Table 3 행 5로 FM 제거 시 성능 저하 정량화.

---

## 13. Benchmark-Realism Narrative 평가

**질문**: "benchmark-realism narrative (contaminated 프로토콜이 더 현실적이다)"를 논문 중심 서사로 올릴 것인가?

**평가**:
- 강점: 기존 TSAD 벤치마크는 clean-train을 가정하는 암묵적 이상주의 → 실제 운영 환경과 괴리. test-prefix 편입 프로토콜은 이 괴리를 직접 해소. NRdetector도 "anomalies embedded within the training data" 문제를 동기로 삼아 벤치마크 설계를 논거로 사용.
- 약점: 프로토콜 자체는 contribution이지만, 방법론(GRL + 비대칭 distillation) 없이 프로토콜만으로는 성능 우위를 설명할 수 없다 → 프로토콜이 "왜 이 방법이 필요한가"의 논거이지, 논문의 주요 기여가 될 수 없다.
- (RT MINOR-01) "기존 벤치마크에 이 설계가 없었다"는 주장은 구체 문헌 인용으로 뒷받침 필요 — Phase 4 수요 등재 (§16).
- **결론**: benchmark-realism을 Introduction의 Para 3 핵심 관찰로 사용하되(기존 벤치마크의 한계 → 우리 프로토콜 필요성), 논문의 중심 서사는 "labeled anomaly를 표현 학습에 통합한 end-to-end 단일 모델"로 유지. 프로토콜은 contribution bullet 1번(§11 결정 ①)에서 기여로 독립 명시.

---

## 14. 50% test-prefix 프로토콜 방어 서술 (r2 전면 재구축 — RT BLOCKER-01)

> 공격의 정확한 형태를 먼저 인정한다: "train 구간 내 anomaly label(force_mask_anomaly·GRL 타깃)의 출처가 원본 test split의 앞 50%다 — test 데이터의 ground-truth label로 모델을 학습시킨다." 기존 r1의 5원칙(look-ahead 없음·추론 시 라벨 미사용 등)은 이 공격의 **핵심(라벨 출처)**에 정면으로 답하지 않았다. r2는 정면 답변 5논거 + 한계 인정으로 재구축한다.

### 정면 답변 구조 (논문 §4.1.1 방어 문단 + rebuttal 공용)

**논거 ① — 정의: 편입분은 더 이상 test가 아니라 train이다.** 본 프로토콜은 벤치마크의 **재분할(re-split) 정의**다: Train = [원본 train 전체 | 원본 test의 시간순 앞 50%], Test = 원본 test의 뒤 50%. 평가는 보존된 뒤 50%에서만 수행되며, 그 라벨은 학습·추론 어디에도 개입하지 않는다. "test label로 학습"이 아니라 "새 분할 정의 하에서 train에 labeled anomaly가 존재하는 설정" — 어떤 모델도 자신이 평가받는 데이터(뒤 50%)의 정보를 학습에서 보지 않는다.

**논거 ② — 구조적 필연: 원본 train에는 쓸 라벨이 없다 (프로토콜의 존재 이유).** "원본 train split의 라벨만 쓰면 되지 않는가?"에 대한 답: 원본 train split에는 labeled anomaly가 **구조적으로 존재하지 않는다** — EXPERIMENT_PROTOCOL_TRUTH §①·② 실측: SWaT/WaDi 원본 train = 정상 운영 구간(공격 이벤트는 전부 test 스트림에 집중; SWaT train anomaly 1.63%는 전적으로 편입된 A2-front에서 유래), PSM/SMD = train 라벨 파일 자체 부재(분야 표준 "전부 정상" 가정), SMAP/MSL = train 라벨 명시적 0. 따라서 원본 split 그대로는 semi-supervised TSAD 방법의 평가가 **정의상 불가능**하다. test 스트림 앞부분의 train 편입은 **실제 운영 라벨의 분포를 보존하면서** 이 설정을 평가 가능하게 만드는 **가장 직접적인** 구조적 장치이며, 이것이 프로토콜의 존재 이유다. **(r3 완화, R2-MIN-01)**: "유일한" 표기 금지 — synthetic anomaly 주입(SDMAE류)도 "라벨 있는 train"을 만드는 구조적 장치이므로 반례가 된다; 필요 시 "(synthetic injection은 실제 운영 라벨의 활용을 평가하지 못한다)" 1구 병기 (§0.3 라벨 출처 축 — 합성 pseudo vs 실제 운영 — 과 정합).

**논거 ③ — 공정성: 모든 비교 모델이 동일 데이터를 받는다.** 동일 재분할이 전 비교 모델(MAE + 22 비지도 + 4 weakly-supervised)에 적용된다. 비지도 baseline에는 Q3(normalonly)로 같은 라벨의 "그들 패러다임에서의 최선 활용"(오염원 제거)을 제공한다 (R12) — "라벨 있는 우리 vs 라벨 없는 그들"이 아니라 "같은 라벨을 각자 최선으로 쓴 비교".

**논거 ④ — 시간성·통일성: look-ahead 없음 + 전 데이터셋 단일 규칙.** 시간 순서 보존(앞→train, 뒤→test; 미래 데이터로만 평가 — 온라인 운영 부합), 분할 규칙(// 2) 전 데이터셋 통일 (R13), SMAP/MSL safe-cut 메커니즘과 실측 이동량(81채널 중 4채널, max +166 steps) 공개. 데이터셋별 취사선택 없음.

**논거 ⑤ — 선례: 선행 semi/weakly-supervised TSAD도 재분할 위에서 평가한다.** NRdetector(KDD 2025)는 원본 벤치마크 스트림을 7:3 segment split으로 재분할하여 train에 (대부분 unlabeled인) anomaly가 자연 포함된 상태에서 학습·평가한다 (NRDETECTOR_DOSSIER §3.1/D8 — "anomalies embedded within the training data"를 동기로 명시). 즉 "원본 공개 split의 train/test 경계를 그대로 두어야 한다"는 가정 자체가 semi-supervised TSAD 선행 연구에서 이미 유지되지 않는다. (주의: NRdetector 7:3 split의 시간 순서 보존 여부는 원문 미명시 — 단정 인용 금지, "재분할 선례" 수준으로만 인용.)

**+ 한계 인정 (1문장, 필수)**: 편입된 prefix 구간의 anomaly 유형·분포는 보존된 test 후반 50%의 그것과 다를 수 있으며, 이 분포 이동은 본 프로토콜의 한계로 명시한다.

### 배치 지침
- **§4.1.1**: 논거 ②(구조적 필연) 중심의 1문단 정면 답변 + 논거 ①·③·④ 요약 + 한계 인정 (§6.2 반영 완료).
- **§1 Para 3**: 논거 ②의 1문장 에코 ("원본 벤치마크에는 이 설정을 평가할 라벨이 train에 없다") — **스코핑 필수 (r3, R2-NOTE-02)**: "the standard MTSAD benchmarks we evaluate on" 수준으로 한정, 전칭 표현 금지 (§3.1 Para 3 동일 지침).
- **§15 방어 표**: 논거 ①–⑤ 요약 행 갱신 완료.
- 기존 r1 5원칙(동기 명시·공정성 담보·분할 투명성·비표준 인정·재현성)은 논거 ①–⑤에 통합 유지: 비표준임 인정 문구("This protocol differs from the standard clean-train split used in prior TSAD benchmarks. We adopt it deliberately to evaluate labeled-anomaly-aware methods...")와 재현성(분할 코드 공개, seed=42)은 그대로 사용.

---

## 15. 리뷰어 방어 예상 시나리오 요약

| 시나리오 | 방어 논거 |
|---------|--------|
| "test-prefix 편입은 test label로 학습하는 leakage" | **§14 정면 답변 5논거**: ① 재분할 정의상 편입분=train, 평가는 보존 뒤 50%만 ② 원본 train에 labeled anomaly 구조적 부재(실측) → 원본 split로는 semi-supervised 평가 불가 = 프로토콜 존재 이유 ③ 전 모델 동일 데이터(비지도=Q3 최선 활용) ④ 시간 순서 보존+전 데이터셋 통일 규칙 ⑤ NRdetector 등 선행 연구의 재분할 선례. + prefix/test 분포 이동 한계 인정. |
| "SDMAE와 너무 유사" | 도메인(video vs TS), 라벨 출처(합성 vs 실제), 작동 계층(타깃/손실 공간 vs gradient 공간 — §3.5 본문 1문장), 구조(branch-off 분기 vs 독립 비대칭 decoder — Method 각주). §2.3 계보 내 자연 포지셔닝("adapt", sibling 포지션). |
| "PU learning이 아닌데 PU라 함" | "contaminated semi-supervised"를 primary 명칭으로 사용. PU와의 관계는 §2.2에서 명확히 스코핑. main 실험=상한 케이스·sweep=일반 케이스 3단 구조(§3.1) 명시. |
| "test-set model selection (best epoch을 test 지표로 선정)" | **사실 그대로 공개** (§4.1.2 — 숨김 금지, EXPERIMENT_PROTOCOL_TRUTH REQUEST-4): 전 모델(MAE+22 baseline) 동일 프로토콜·동일 기준(pak_auc_f1)·별도 validation split 부재 → **비교 공정성은 유지**. 일반화 추정의 낙관 편향 가능성은 한계로 인정. PA%K-AUC는 K-적분형 지표라 단일 threshold 과적합과는 무관. (옵션) Appendix §B.4 epoch-sensitivity placeholder. |
| "epoch budget 비대칭 (MAE 500 vs unsup 10 vs weak 50) — 불공정" (r2 신설, ADV BLK-005) | ① 전 모델이 "주기 평가(MAE 5-ep/baseline 1-ep 간격) 후 best-epoch 선택" 동일 구조 + early stopping 양쪽 부재 — 각자 budget 완주 후 최적 epoch에서 평가 ② budget은 모델군별 수렴 특성(warmup 250 포함 대형 MAE 장기 수렴 vs 소형 baseline 단기 수렴) 반영한 best-effort, 원 구현 충실 원칙 ③ batch size 차이(1024 vs 512)도 동일 원칙으로 공개 ④ (옵션) Appendix §B.4 epoch-budget sensitivity placeholder. 비대칭 자체를 §4.1.2에 명시 공개(은폐 금지). |
| "threshold selection이 불공정" | PA%K-AUC(threshold sweep 적분), VUS-PR/ROC(threshold-free), Affiliation F1(AR threshold `affiliation_f1_ar`, 비oracle) — oracle 지표는 "(oracle)" 명시 후 보조 사용. |
| "SWaT excl22 기준이 자의적" | excl22 기준 사용 이유(region #22가 83.75% mass → 변별력 낮은 단일사건 탐지 지표) + excl22 dedicated entity를 동일하게 모든 baseline에 적용 + full 결과도 Appendix에 병기. |
| "성능 우위가 프로토콜(데이터 추가) 때문 아닌가" (r2 신설, RT BLOCKER-03) | §4.2 Table 4 protocol-effect 분리: 표준 split에서도 경쟁력 유지(방법 효과) + contaminated에서 제안 방법만 추가 이득(라벨 활용 효과). Q3 train 양적 비대칭도 §4.1.4에서 인정. |
| "warmup ablation 없음" | warmup을 contribution bullet·독립 기여로 올리지 않음(r2 — bullet 3에서 제거). 학습 안정화 장치로만 서술. Fig. training curve 정성 근거만 사용. Table 3 warmup 행은 실험 완료 시에만 포함(conditional). |
| "GRL이 student를 망가뜨리지 않는가" | OD loss(정상 패치 mimicry 강제)와 GRL(anomaly 표현 억제)의 역할 분리 명확히 서술. student는 encoder latent를 detach로 받으므로 encoder는 GRL gradient로부터 완전 차단. + **반전 계수 λ_rev의 Ganin sigmoid ramp(0→≈1)로 suppression 강도가 student-phase 동안 점진 증가** — 학습 초기 student 표현 붕괴를 막는 표준 안정화 장치 (r3 신설, NEW-B1 파생 방어 재료). |
| "'최초' 주장 과장" (r2 신설, RT NOTE-03) | 스코핑 경계 고정(§0.1): "다변량 TSAD에서 labeled anomaly를 자기지도 표현 학습의 기울기에 직접 통합하는 end-to-end 단일 모델" + "(to our knowledge)" — 이 경계 밖 확장 금지. Phase 4 반증 검색으로 사전 검증. |

---

## 16. Phase 4 연계 수요 (r2 신설 — 양 리뷰 합산)

| 검증/인용 수요 | 출처 발견 | 등록 위치 |
|--------------|---------|---------|
| "기존 TSAD 벤치마크는 clean-train 가정" 문헌 1–2개 | RT MINOR-01 | §4.1.1 |
| "contaminated semi-supervised" 용어 기존 사용 여부 검색 | ADV MINOR-005 | §11 결정 ② |
| "end-to-end first" 최초성 반증 논문 부재 검증 | ADV MINOR-006 / RT NOTE-03 | §0.1/§3.1 Thesis |
| NRdetector "거의 유일한 선행" 반증 부재 검증 | ADV §4 표 | §2.2 |
| Zhang et al. TPAMI 2022 self-distillation 원류 서지 확인 | ADV §4 표 | §2.3 |
| Lin et al. 2017 focal loss 수식 대조 (variant 차이 1문장 근거) | ADV MAJ-004 | §3.5 |
| AR threshold의 TSAD 문헌 관행 선례 (확보 전 이 방어 논리 사용 금지) | EXPERIMENT_PROTOCOL_TRUTH §⑤-4 | §4.1.3 |
| 산업 응용/survey, 비지도 4유형 클러스터, GRL 원전(Ganin 2016), 지표 4종 DOI | ADV §4 표 (기존) | §1/§2.1/§3.5/§4.1.3 |

---

## 부록: r3 정정 이력 (2026-06-11, fixer)

전수 처리표는 `paper/99_reviews/p3_fixlog_r3.md` (작업 A 코드 확정 기록 포함). Phase 1 정본 동반 보강: 271_CONFIG_TRUTH r4(§VIII GRL Details λ_rev + warmup forward-skip), CODEBASE_UNDERSTANDING r4, RESEARCH_SYNTHESIS r3. 주요 변경:

1. **[ADV NEW-B1]** GRL 이중 λ 구조 — §5.5 이원 서술(손실 가중치 λ_GRL grad-ratio×0.2 즉시 투입 + 반전 계수 λ_rev Ganin sigmoid ramp 0→≈1), §5.6(C) backward 계수를 λ_rev로 정정(λ_GRL_eff는 손실 항 가중치), §9.1 λ_rev 행 신설, §9.2 "sigmoid 서술 금지" 조항 철회·역할 분리 조항으로 교체, §15 GRL 행에 λ_rev ramp 방어 재료 추가, 논문 method 서술 방침(일반적 수준 — 손실 가중 적응 + 반전 계수 ramp) 명시. 코드 근거: trainer.py:1201–1211, model.py:1152–1153/129–140, trainer.py:751–765.
2. **[ADV NEW-B2]** warmup 중 student forward — §5.5 "forward는 수행되지만 gradient 차단" 서술을 **학습 경로 forward 자체 skip**(model.py:1119, trainer.py:526–535; loss.py:213 이중 방어)으로 역전 교체 + 평가 경로 full forward 구분 + capacity-gap·안정화 논리 재점검(충돌 없음 확인).
3. **[RT R2-MAJ-01]** ablation suite(Table 3 행 2–5·7) §0.4 Phase 5 진입 조건 등재(최소 행 2·7 필수; 행 7 = bullet 3 load-bearing) + §6.7 행 5·7 conditional placeholder 명시 + 미완 시 bullet 3 주장 강도 하향 지침. EXPERIMENT_EXECUTION_TODO 집계는 p3_fixlog_r3가 fixlog r2 §7을 대체·확장.
4. **[MINOR/NOTE 7건]** NEW-m1(§0.4·§6.2 → 36/113 학습 단위·37/114 평가 단위 통일), R2-MIN-01(§14 논거 ② "유일한" 완화), R2-MIN-03(§6.6 Table 4 실행 사양 — use_grl=True 유지 + `loss.py:293–302` pos_count==0 skip 인용 + baseline=Q3 명시), R2-MIN-04(§6.5 "0.5–6.2%" → 실측 완료 기준 + SMD 확정 대기), R2-NOTE-01(§6.3 §B.4 실측 격상 권고 — REQUEST-4 (iii) 소형 실험), R2-NOTE-02(§3.1 Para 3·§14 에코 스코핑), NEW-n1(§6.4 affiliation_f1_ar 라인 811–813 통일). R2-MIN-02(fallback 사다리)는 PAGE_BUDGET r3에서 처리.

---

## 부록: r2 정정 이력 (2026-06-11, blueprint-reviser)

전수 처리표는 `paper/99_reviews/p3_blueprint_fixlog_r2.md`. 주요 변경:
1. **[RT B-01]** §14 전면 재구축 — 정면 답변 5논거(재분할 정의/원본 train 라벨 구조적 부재/공정성/시간성·통일성/NRdetector 선례) + 한계 인정 + §4.1.1/Intro 배치 확정.
2. **[RT B-02 / ADV MAJ-010]** contribution bullet 3에서 "teacher-only warmup" 제거 — 아키텍처 논리(capacity gap → 신뢰성 있는 discrepancy)로 재구성; warmup은 §3.4 안정화 장치 전속.
3. **[RT B-03]** §4.2에 protocol-effect 보조 분석(Table 4: standard split vs contaminated) main text 신설 — 2단 논증(표준 조건 경쟁력 + 라벨 활용 이득); 결정 ④ 갱신.
4. **[ADV BLK-001]** §2 분량 수치를 PAGE_BUDGET §1과 단일화 (PAGE_BUDGET=정본 선언; §1 1.6p/§4 3.3p r2 조정 반영).
5. **[ADV BLK-002]** Fig. 2: GRL 위치(student decoder 마지막 층 hidden, output projection 이전) + "training only(추론 비활성)" 표기 필수화.
6. **[ADV BLK-003 + 추가 발견]** SMD F=29–36(constant 제거 후) 명기; **d_model은 dynamic이 아니라 전 entity 512 고정으로 정정** (271_CONFIG_TRUTH §II + PSM checkpoint patch_embed=(512,250) 실측 — dynamic이면 256); dim_feedforward=2048 고정; Appendix §C.1을 Input Dimensionality Table로 교체.
7. **[ADV BLK-004]** §3.4 GRL λ sigmoid ramp-up 공식 삭제 → trainer inline grad-ratio adaptive λ(직전 epoch, clamp [0,10], ×0.2) + "ramp 없이 즉시 투입"으로 교체.
8. **[ADV BLK-005]** epoch 비대칭(500/10/50)·batch(1024/512) 명시 공개 + §15 공정성 방어 시나리오 신설 + Appendix §B.4 epoch-sensitivity placeholder.
9. 그 외 MAJOR 22건·MINOR 11건·NOTE 6건 처리 — fixlog 참조 (SDMAE branch-off 정정, SOTA Legacy 6 재분류, test-set selection 공개·방어, SWaT 45 재현성 플래그, L_cls 표기 통일, per-patch/집계 수식 분리, focal variant positive 지침, TS-SDMAE 제외, Table 2 열 고정 등).
