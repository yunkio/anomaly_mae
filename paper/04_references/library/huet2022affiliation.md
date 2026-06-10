---
phase: 4
agent: excerpt-curator-2
directives: [T4]
last_modified: 2026-06-11
key: huet2022affiliation
verification_status: VERIFIED_A
verified_by_A: 2026-06-11
corrected_fields: []
card_grade: FULL
excerpt_access: abstract_only
---

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT — verbatim excerpts below are for internal reference only and must not appear in any submitted text -->

> **A2 경고**: 아래 verbatim 발췌는 card 내부 전용입니다. 어떠한 형태로도 원고에 그대로 복사하지 마십시오.

---

## 서지 정보

- **Key**: huet2022affiliation
- **제목**: Local Evaluation of Time Series Anomaly Detection Algorithms
- **저자**: Alexis Huet, Jose Manuel Navarro, Dario Rossi
- **Venue**: KDD 2022, pp. 635–645
- **DOI**: 10.1145/3534678.3539339
- **arXiv**: 2206.13167 (직접 확인 2026-06-11)
- **DBLP**: conf/kdd/HuetNR22
- **확인 출처**: arXiv abs + DBLP 직접 열람

---

## Abstract (verbatim)

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT -->

"In recent years, specific evaluation metrics for time series anomaly detection algorithms have been developed to handle the limitations of the classical precision and recall. However, such metrics are heuristically built as an aggregate of multiple desirable aspects, introduce parameters and wipe out the interpretability of the output. In this article, we first highlight the limitations of the classical precision/recall, as well as the main issues of the recent event-based metrics -- for instance, we show that an adversary algorithm can reach high precision and recall on almost any dataset under weak assumption. To cope with the above problems, we propose a theoretically grounded, robust, parameter-free and interpretable extension to precision/recall metrics, based on the concept of ``affiliation'' between the ground truth and the prediction sets. Our metrics leverage measures of duration between ground truth and predictions, and have thus an intuitive interpretation. By further comparison against random sampling, we obtain a normalized precision/recall, quantifying how much a given set of results is better than a random baseline prediction. By construction, our approach keeps the evaluation local regarding ground truth events, enabling fine-grained visualization and interpretation of algorithmic results. We compare our proposal against various public time series anomaly detection datasets, algorithms and metrics. We further derive theoretical properties of the affiliation metrics that give explicit expectations about their behavior and ensure robustness against adversary strategies."

---

## 핵심 발췌

### 발췌 1 — 기존 지표 한계 (§Abstract)

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT -->

> "an adversary algorithm can reach high precision and recall on almost any dataset under weak assumption."

- **§위치**: Abstract
- **지지 Claim**: C-049 (affiliation 지표 도입 동기 — 기존 event-based 지표의 취약성)
- **활용 맥락**: §4.1.3에서 affiliation 지표를 채택한 이유를 설명할 때 기존 지표의 adversarial 취약성 근거로 인용.
- **주의**: "under weak assumption" — 전제 조건이 있는 주장이므로 우리 논문에서 무조건 서술하면 왜곡. "under certain conditions" 수준으로 완화.

---

### 발췌 2 — Affiliation 개념 정의 (§Abstract)

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT -->

> "a theoretically grounded, robust, parameter-free and interpretable extension to precision/recall metrics, based on the concept of ``affiliation'' between the ground truth and the prediction sets."

- **§위치**: Abstract
- **지지 Claim**: C-049 (Affiliation F1: 시간적 근접도 기반 local 평가 지표 제안 논문)
- **활용 맥락**: §4.1.3에서 Affiliation 지표를 소개할 때 1문장 정의 + 인용. "parameter-free"와 "theoretically grounded"는 채택 동기로 부각 가능.
- **주의**: 이 발췌를 그대로 쓰면 표절. 핵심 속성(parameter-free, local, duration-based)만 추출하여 paraphrase.

---

### 발췌 3 — Duration 기반 측정 (§Abstract)

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT -->

> "Our metrics leverage measures of duration between ground truth and predictions, and have thus an intuitive interpretation."

- **§위치**: Abstract
- **지지 Claim**: C-049
- **활용 맥락**: affiliation 지표가 시간적 근접도(duration)를 활용한다는 직관적 설명. PA나 VUS와 비교하여 이 지표가 "지속 시간" 기반 근접도를 반영한다는 서술.
- **주의**: "measures of duration between ground truth and predictions" — 이 표현의 정확한 의미(이상 구간과 예측 간의 시간 거리)는 본문 §2/§3에서 확인 필요.

---

### 발췌 4 — Local 평가 철학 (§Abstract)

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT -->

> "our approach keeps the evaluation local regarding ground truth events, enabling fine-grained visualization and interpretation of algorithmic results."

- **§위치**: Abstract
- **지지 Claim**: C-049
- **활용 맥락**: §4.1.3에서 "local evaluation" 특성을 갖는 affiliation 지표를 소개할 때. PA 같은 전역 평가와의 차이 설명.
- **주의**: "fine-grained visualization" — 우리 논문이 시각화를 강조하지 않는다면 이 측면은 인용 불필요.

---

### 발췌 5 — 랜덤 기준 대비 정규화 (§Abstract)

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT -->

> "By further comparison against random sampling, we obtain a normalized precision/recall, quantifying how much a given set of results is better than a random baseline prediction."

- **§위치**: Abstract
- **지지 Claim**: C-049 (보강)
- **활용 맥락**: affiliation 지표가 랜덤 예측 대비 성능을 정규화한다는 점 — 지표의 신뢰성 측면에서 선택 정당화.
- **주의**: affiliation과 affiliation-normalized의 차이가 있는지 본문에서 확인 필요 — verifier 확인 사항.

---

## 활용 절

| 우리 논문 위치 | 활용 방식 | 근거 발췌 |
|-------------|---------|---------|
| §4.1.3 지표 소개 | Affiliation F1 제안 논문으로 인용 + 1문장 특징 설명 | 발췌 2, 3 |
| §4.1.3 지표 선택 동기 | 기존 event-based 지표의 한계 + duration-based local 평가의 장점 | 발췌 1, 4 |
| §4.1.3 보강 | 랜덤 기준 정규화 특성 (선택적) | 발췌 5 |

---

## 주의사항

1. **수식 미확보**: abstract에 affiliation precision/recall 공식 없음. verifier가 PDF §2/§3에서 공식 발췌 필요. 우리 논문에서 수식 없이 "duration-based local evaluation" 설명으로 충분.
2. **PA와의 관계**: affiliation이 PA 문제를 해결하는지 직접 서술하진 않음 — Kim et al.과 병행 인용 시 역할 구분 필요(Kim = PA 비판, Huet = 대안 지표 제안).
3. **복사 금지 표현**: "theoretically grounded, robust, parameter-free and interpretable extension" — 이 수식어들을 그대로 쓰면 표절. 각 속성을 독립적으로 서술.
