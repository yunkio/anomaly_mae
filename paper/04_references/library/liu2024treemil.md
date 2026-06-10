---
phase: 4
agent: excerpt-curator-2
directives: [T4]
last_modified: 2026-06-11
key: liu2024treemil
verification_status: VERIFIED_A
verified_by_A: source-verifier-A2
card_grade: FULL
excerpt_access: abstract_only
corrections:
  - field: authors
    wrong: "Chen Liu, Shibo He, Haoyu Liu, Jiming Li"
    correct: "Chen Liu, Shibo He, Haoyu Liu, Shizhong Li"
    severity: CRITICAL
    confirmed_by: "DBLP conf/icassp/LiuHLL24; arXiv 2401.11235 search"
  - field: pages
    wrong: missing
    correct: "7510–7514"
    severity: MINOR
---

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT — verbatim excerpts below are for internal reference only and must not appear in any submitted text -->

> **A2 경고**: 아래 verbatim 발췌는 card 내부 전용입니다. 어떠한 형태로도 원고에 그대로 복사하지 마십시오.

---

## 서지 정보

- **Key**: liu2024treemil
- **제목**: TreeMIL: A Multi-instance Learning Framework for Time Series Anomaly Detection with Inexact Supervision
- **저자**: Chen Liu, Shibo He, Haoyu Liu, Shizhong Li [VERIFIED_A: "Jiming Li" was wrong — confirmed "Shizhong Li" via DBLP + arXiv]
- **Venue**: IEEE ICASSP 2024 (ICML/NeurIPS 추정은 오류 — 정정 확인)
- **DOI**: 10.1109/ICASSP48485.2024.10447536
- **arXiv**: 2401.11235 (직접 확인 2026-06-11; "accepted by IEEE ICASSP 2024" 명시)
- **ieeexplore**: 문서번호 10447536
- **확인 출처**: arXiv abs 직접 열람 + DBLP (R26 truth [B17] 정합)

---

## Abstract (verbatim)

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT -->

"Time series anomaly detection (TSAD) plays a vital role in various domains such as healthcare, networks, and industry. Considering labels are crucial for detection but difficult to obtain, we turn to TSAD with inexact supervision: only series-level labels are provided during the training phase, while point-level anomalies are predicted during the testing phase. Previous works follow a traditional multi-instance learning (MIL) approach, which focuses on encouraging high anomaly scores at individual time steps. However, time series anomalies are not only limited to individual point anomalies, they can also be collective anomalies, typically exhibiting abnormal patterns over subsequences. To address the challenge of collective anomalies, in this paper, we propose a tree-based MIL framework (TreeMIL). We first adopt an N-ary tree structure to divide the entire series into multiple nodes, where nodes at different levels represent subsequences with different lengths. Then, the subsequence features are extracted to determine the presence of collective anomalies. Finally, we calculate point-level anomaly scores by aggregating features from nodes at different levels. Experiments conducted on seven public datasets and eight baselines demonstrate that TreeMIL achieves an average 32.3% improvement in F1-score compared to previous state-of-the-art methods."

---

## 핵심 발췌

### 발췌 1 — "inexact supervision" 정의 (§Abstract)

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT -->

> "only series-level labels are provided during the training phase, while point-level anomalies are predicted during the testing phase."

- **§위치**: Abstract
- **지지 Claim**: C-023 (TreeMIL: weakly-supervised — 라벨이 분류 목적함수의 지도 신호로 사용, pretext 없음), C-072 (baseline 출처)
- **활용 맥락**: §2.2에서 TreeMIL의 "inexact supervision" 설정을 정의. 시계열 레벨(series-level) 라벨 → 포인트 레벨 예측이라는 설정이 우리 설정(창 내 이상 포인트 라벨 + 패치 레벨 재구성 학습)과 다름을 대비.
- **주의**: "inexact supervision"은 TreeMIL의 용어 — 우리 설정("contaminated semi-supervised")과 혼동하지 않도록 귀속 표기 필수.

---

### 발췌 2 — 집단 이상(collective anomaly) 문제 (§Abstract)

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT -->

> "time series anomalies are not only limited to individual point anomalies, they can also be collective anomalies, typically exhibiting abnormal patterns over subsequences."

- **§위치**: Abstract
- **지지 Claim**: C-023 (보조 — TreeMIL 차별화 포인트 설명)
- **활용 맥락**: §2.2에서 TreeMIL의 기여(집단 이상 처리)를 1문장으로 소개. 우리 모델도 패치 기반 표현으로 집단 이상을 암묵적으로 처리함을 주장할 때 배경 맥락으로 사용.
- **주의**: "collective anomaly" 개념은 TreeMIL 원류가 아닌 시계열 이상탐지 일반 용어 — 출처를 TreeMIL에만 귀속하면 부정확.

---

### 발췌 3 — N-ary tree 구조 (§Abstract)

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT -->

> "We first adopt an N-ary tree structure to divide the entire series into multiple nodes, where nodes at different levels represent subsequences with different lengths."

- **§위치**: Abstract
- **지지 Claim**: C-023, C-072 (TreeMIL 방법론 계보 서술용)
- **활용 맥락**: §2.2에서 TreeMIL 방법론을 1문장으로 서술. 분류 목적함수 레벨에서 작동하며 self-supervised pretext가 없는 구조임을 설명.
- **주의**: 아키텍처 상세(N-ary tree)는 우리 논문에서 불필요 — "tree-based MIL" 수준으로 요약.

---

### 발췌 4 — 기존 MIL 방법의 한계 (§Abstract)

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT -->

> "Previous works follow a traditional multi-instance learning (MIL) approach, which focuses on encouraging high anomaly scores at individual time steps."

- **§위치**: Abstract
- **지지 Claim**: C-023 (weakly-supervised 계열 계보)
- **활용 맥락**: §2.2에서 MIL 기반 방법(DeepMIL 포함)이 포인트 레벨 이상 점수에 집중했다는 한계 서술. TreeMIL이 이를 개선했지만, 두 방법 모두 재구성 pretext 없이 분류/순위 목적함수 수준에서 라벨을 사용한다는 공통점 서술.
- **주의**: WETAS와 DeepMIL이 "previous works"에 해당하는지 verifier 확인 필요.

---

## 활용 절

| 우리 논문 위치 | 활용 방식 | 근거 발췌 |
|-------------|---------|---------|
| §2.2 Related Work | TreeMIL의 inexact supervision 설정 + 방법론 1문장 소개 | 발췌 1, 3 |
| §2.2 차별화 | MIL 계열(DeepMIL/WETAS/TreeMIL)이 분류 목적함수 레벨에서 라벨 사용 — 우리와 차별화 | 발췌 4 |
| §4.1.4 baseline | TreeMIL baseline 원논문 인용 | C-072 |

---

## 주의사항

1. **venue 정정**: ICASSP 2024 (ICML/NeurIPS 추정은 오류) — 인용 시 반드시 "IEEE ICASSP 2024"로 표기.
2. **"inexact supervision" 용어**: TreeMIL의 고유 용어 — 우리 논문에서 이 용어를 사용하면 TreeMIL을 연상시키므로 명확한 귀속 또는 다른 표현 사용.
3. **F1 수치**: "32.3% improvement" — 비교 대상(8 baselines)의 구성이 명확하지 않으므로 이 수치를 우리 논문에서 인용할 때는 원문 맥락 확인 필요.
4. **복사 금지 표현**: "labels are crucial for detection but difficult to obtain" — 우리 논문 §1에서도 유사한 논리를 전개하므로, TreeMIL abstract에서 이 구절을 참조하여 쓰면 독립적 서술로 보이지 않을 위험.
