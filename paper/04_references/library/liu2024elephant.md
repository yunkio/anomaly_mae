---
phase: 4
agent: excerpt-curator-2
directives: [T4]
last_modified: 2026-06-11
key: liu2024elephant
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

- **Key**: liu2024elephant
- **제목**: The Elephant in the Room: Towards A Reliable Time-Series Anomaly Detection Benchmark
- **저자**: Qinghua Liu, John Paparrizos
- **Venue**: NeurIPS 2024 Datasets and Benchmarks Track
- **proceedings.neurips.cc**: hash c3f3c690b7a99fba16d0efd35cb83b2c
- **OpenReview**: R6kJtWsTGy
- **확인 출처**: proceedings.neurips.cc 직접 열람 (2026-06-11)

---

## Abstract (verbatim)

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT -->

"Time-series anomaly detection is a fundamental task across scientific fields and industries. However, the field has long faced the 'elephant in the room:' critical issues including flawed datasets, biased evaluation measures, and inconsistent benchmarking practices that have remained largely ignored and unaddressed. We introduce the TSB-AD to systematically tackle these issues in the following three aspects: (i) Dataset Integrity: with 1070 high-quality time series from a diverse collection of 40 datasets (doubling the size of the largest collection and four times the number of existing curated datasets), we provide the first large-scale, heterogeneous, meticulously curated dataset that combines the effort of human perception and model interpretation; (ii) Measure Reliability: by revealing issues and biases in evaluation measures, we identify the most reliable and accurate measure, namely, VUS-PR for anomaly detection in time series to address concerns from the community; and (iii) Comprehensive Benchmarking: with a broad spectrum of 40 detection algorithms, from statistical methods to the latest foundation models, we perform a comprehensive evaluation that includes a thorough hyperparameter tuning and a unified setup for a fair and reproducible comparison. Our findings challenge the conventional wisdom regarding the superiority of advanced neural network architectures, revealing that simpler architectures and statistical methods often yield better performance. The promising performance of neural networks on multivariate cases and foundation models on point anomalies highlights the need for further advancements in these methods."

---

## 핵심 발췌

### 발췌 1 — VUS-PR 최신뢰 지표 권고 (§Abstract)

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT -->

> "we identify the most reliable and accurate measure, namely, VUS-PR for anomaly detection in time series to address concerns from the community"

- **§위치**: Abstract (ii) Measure Reliability
- **지지 Claim**: C-048 (VUS-PR 권고 보강), C-075 (TSB-AD benchmark 인용), C-009 (벤치마크 관행 비판)
- **활용 맥락**: §4.1.3에서 VUS-PR을 주 지표로 채택한 근거. "Liu & Paparrizos (NeurIPS 2024) independently identified VUS-PR as the most reliable measure" 수준의 서술.
- **주의**: "most reliable and accurate" — 우리 논문에서 이 주장을 받아들일 때 전제 맥락(TSB-AD 40 datasets, 40 algorithms 분석)을 함께 서술해야 근거가 성립.

---

### 발췌 2 — 벤치마크 구조적 문제 (§Abstract)

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT -->

> "critical issues including flawed datasets, biased evaluation measures, and inconsistent benchmarking practices that have remained largely ignored and unaddressed."

- **§위치**: Abstract
- **지지 Claim**: C-008, C-009, C-045 (기존 벤치마크의 clean-train 가정 + 평가 관행 비판)
- **활용 맥락**: §1 또는 §4.1.1에서 기존 벤치마크의 문제를 1문장으로 요약할 때. "flawed datasets", "biased evaluation measures", "inconsistent benchmarking"이라는 세 범주를 우리 논문의 프로토콜 설계 동기와 연결.
- **주의**: 이 논문이 지적하는 문제들 중 어느 것이 우리 논문 프로토콜과 직접 관련되는지 서술 시 명확히 할 것 — clean-train 가정 비판이 직접 관련.

---

### 발췌 3 — TSB-AD 규모 (§Abstract)

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT -->

> "1070 high-quality time series from a diverse collection of 40 datasets (doubling the size of the largest collection and four times the number of existing curated datasets)"

- **§위치**: Abstract (i) Dataset Integrity
- **지지 Claim**: C-009, C-075 (benchmark 규모 + 신뢰성)
- **활용 맥락**: TSB-AD를 소개할 때 규모를 간결하게 서술. 단, 우리는 TSB-AD를 실험 데이터셋으로 사용하지 않으므로 평가 관행 비판 맥락으로만 인용.
- **주의**: 우리 논문이 TSB-AD의 40 데이터셋을 사용하지 않으므로 직접 벤치마크 비교 의미로 쓰면 안 됨.

---

### 발췌 4 — 심층 신경망 우위 통념 도전 (§Abstract)

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT -->

> "Our findings challenge the conventional wisdom regarding the superiority of advanced neural network architectures, revealing that simpler architectures and statistical methods often yield better performance."

- **§위치**: Abstract
- **지지 Claim**: C-008, C-009 (벤치마크 신뢰성 비판, 보조)
- **활용 맥락**: 평가 관행의 신뢰성 문제를 강조할 때 보조 서술. 단, 우리 논문은 신경망 아키텍처를 제안하므로 이 발견에 직접 동조하는 서술은 자기 모순이 될 수 있음 — 사용 시 주의.
- **주의**: 이 발췌를 우리 논문에서 인용할 경우 "benchmark reliability" 맥락으로 한정. "복잡한 모델이 불필요하다"는 논지로 쓰면 역효과.

---

## 활용 절

| 우리 논문 위치 | 활용 방식 | 근거 발췌 |
|-------------|---------|---------|
| §4.1.3 지표 채택 근거 | VUS-PR을 주 지표로 쓰는 문헌적 근거 | 발췌 1 |
| §4.1.1 또는 §1 | 기존 벤치마크 관행 비판 (1문장) | 발췌 2 |
| §4.1.1 주석 | TSB-AD benchmark 소개 (괄호 인용) | 발췌 3 |

---

## 주의사항

1. **clean-train 가정 명시 발췌 미확보**: abstract에 "clean train split" 관련 직접 서술 없음 — verifier가 본문(§2 또는 §3)에서 clean-train 가정 비판 발췌 확보 필요. C-008/C-045 지지를 위해 필수.
2. **VUS-PR 원문은 Paparrizos et al. 2022**: VUS-PR을 제안한 논문(paparrizos2022vus)과 "가장 신뢰할 수 있는 지표"로 추천한 논문(liu2024elephant)을 구분. 인용 구조: "VUS-PR [Paparrizos et al. 2022], recently identified as the most reliable measure [Liu & Paparrizos 2024]".
3. **복사 금지 표현**: "critical issues including flawed datasets, biased evaluation measures, and inconsistent benchmarking practices" — 임팩트 있는 구절이나 원문 그대로 쓰면 표절.
