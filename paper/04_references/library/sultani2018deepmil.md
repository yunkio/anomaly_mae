---
phase: 4
agent: excerpt-curator-2
directives: [T4]
last_modified: 2026-06-11
key: sultani2018deepmil
verification_status: VERIFIED_A
verified_by_A: source-verifier-A2
verified_note: "All fields confirmed via arXiv + DBLP. Pages 6479-6488, DOI 10.1109/CVPR.2018.00678 confirmed. Abstract verbatim confirmed. All 4 excerpts confirmed in abstract."
card_grade: FULL
excerpt_access: abstract_only
---

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT — verbatim excerpts below are for internal reference only and must not appear in any submitted text -->

> **A2 경고**: 아래 verbatim 발췌는 card 내부 전용입니다. 어떠한 형태로도 원고에 그대로 복사하지 마십시오.

---

## 서지 정보

- **Key**: sultani2018deepmil
- **제목**: Real-World Anomaly Detection in Surveillance Videos
- **저자**: Waqas Sultani, Chen Chen, Mubarak Shah
- **Venue**: CVPR 2018, pp. 6479–6488
- **DOI**: 10.1109/CVPR.2018.00678
- **arXiv**: 1801.04264 (직접 확인 2026-06-11)
- **DBLP**: conf/cvpr/SultaniCS18
- **확인 출처**: arXiv abs + DBLP 직접 열람

---

## Abstract (verbatim)

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT -->

"Surveillance videos are able to capture a variety of realistic anomalies. In this paper, we propose to learn anomalies by exploiting both normal and anomalous videos. To avoid annotating the anomalous segments or clips in training videos, which is very time consuming, we propose to learn anomaly through the deep multiple instance ranking framework by leveraging weakly labeled training videos, i.e. the training labels (anomalous or normal) are at video-level instead of clip-level. In our approach, we consider normal and anomalous videos as bags and video segments as instances in multiple instance learning (MIL), and automatically learn a deep anomaly ranking model that predicts high anomaly scores for anomalous video segments. Furthermore, we introduce sparsity and temporal smoothness constraints in the ranking loss function to better localize anomaly during training. We also introduce a new large-scale first of its kind dataset of 128 hours of videos. It consists of 1900 long and untrimmed real-world surveillance videos, with 13 realistic anomalies such as fighting, road accident, burglary, robbery, etc. as well as normal activities."

---

## 핵심 발췌

### 발췌 1 — 비디오 레벨 약한 감독 정의 (§Abstract)

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT -->

> "the training labels (anomalous or normal) are at video-level instead of clip-level."

- **§위치**: Abstract
- **지지 Claim**: C-023 (DeepMIL: weakly-supervised 계열 — 라벨이 분류 목적함수의 지도 신호로 사용)
- **활용 맥락**: §2.2에서 DeepMIL의 weak supervision 설정을 정의할 때. 우리 설정(point-level 라벨 존재)과의 차이: 우리는 segment/window 내 이상 포인트를 알지만, DeepMIL은 전체 비디오 레벨 레이블만 사용.
- **주의**: 비디오 도메인 — 우리 TSAD와 직접 비교 불가. "시계열의 유사 설정"으로만 서술.

---

### 발췌 2 — MIL 프레임워크 정의 (§Abstract)

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT -->

> "we consider normal and anomalous videos as bags and video segments as instances in multiple instance learning (MIL), and automatically learn a deep anomaly ranking model that predicts high anomaly scores for anomalous video segments."

- **§위치**: Abstract
- **지지 Claim**: C-023, C-070 (DeepMIL baseline 출처)
- **활용 맥락**: §2.2에서 MIL 기반 weakly-supervised AD의 원류로 소개. 비디오 bag = 우리 설정의 시계열 인스턴스, segment/instance = 우리 설정의 윈도우/패치에 대응하는 개념.
- **주의**: "bags and instances"는 MIL 전문 용어 — 우리 논문에서 이 용어를 MIL 맥락 없이 쓰면 혼란 가능.

---

### 발췌 3 — 레이블 수집 비용 절감 동기 (§Abstract)

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT -->

> "To avoid annotating the anomalous segments or clips in training videos, which is very time consuming, we propose to learn anomaly through the deep multiple instance ranking framework."

- **§위치**: Abstract
- **지지 Claim**: C-023 (weakly-supervised 계열 설명)
- **활용 맥락**: §2.2에서 weakly-supervised 접근법의 동기(시간 소모적 세그먼트 단위 어노테이션 회피)를 설명할 때 DeepMIL의 동기로 인용. 우리 설정과의 차이: 우리는 세그먼트 레이블을 비싸더라도 활용하는 것을 목적으로 함.
- **주의**: 이 발췌는 "레이블 없이 학습"하는 동기이므로, 우리 논문의 "레이블을 활용"하는 방향과 반대 — 대조적 서술 필요.

---

### 발췌 4 — 스파시티 + 시간적 평활화 제약 (§Abstract)

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT -->

> "we introduce sparsity and temporal smoothness constraints in the ranking loss function to better localize anomaly during training."

- **§위치**: Abstract
- **지지 Claim**: C-023 (보조)
- **활용 맥락**: DeepMIL의 손실 함수 설계가 분류/순위 목적함수 레벨에서 작동함을 설명. 재구성 표현 학습 없이 목적함수에서 직접 이상 신호를 다룬다는 우리 논문과의 차별화 포인트.
- **주의**: "ranking loss"는 이진 분류가 아닌 ranking 프레임워크 — 정확한 설명 필요.

---

## 활용 절

| 우리 논문 위치 | 활용 방식 | 근거 발췌 |
|-------------|---------|---------|
| §2.2 Related Work | 약한 지도학습 계열 소개 — 비디오 레벨 MIL 원류 | 발췌 2 |
| §2.2 차별화 | 우리 설정(라벨 활용)과 weak supervision(라벨 회피) 대비 | 발췌 1, 3 |
| §4.1.4 baseline | DeepMIL baseline 원논문 인용 | C-070 |

---

## 주의사항

1. **비디오 도메인**: 시계열과 직접 비교 불가 — "analogous setting in video AD" 수준으로만 서술.
2. **MIL과 우리 설정의 차이**: DeepMIL은 분류/순위 목적함수에서 라벨을 사용하고 self-supervised pretext(재구성)가 없음 — 이 점이 우리 논문과 C-023에서 명시한 차별화 포인트.
3. **복사 금지 표현**: "deep multiple instance ranking framework" — 전문 용어이지만 그대로 쓰면 MIL 맥락 전제가 필요. 우리 논문에서는 "MIL-based ranking approach (DeepMIL)" 수준으로 서술.
