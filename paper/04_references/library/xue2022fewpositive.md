---
phase: 4
agent: excerpt-curator-2
directives: [T4]
last_modified: 2026-06-11
key: xue2022fewpositive
verification_status: VERIFIED_A
verified_by_A: source-verifier-A2
card_grade: FULL
excerpt_access: abstract_only
corrections:
  - field: authors
    wrong: "Yifan Xue, Yijie Yan"
    correct: "Feng Xue, Weizhong Yan"
    severity: CRITICAL
    confirmed_by: "arXiv 2207.00705; DBLP conf/ijcnn/XueY22"
    note: "Both first names AND last names were wrong — neither author name is correct in original card"
  - field: pages
    wrong: missing
    correct: "1–7"
    severity: MINOR
    confirmed_by: "DBLP conf/ijcnn/XueY22"
  - field: doi
    wrong: missing
    correct: "10.1109/IJCNN55064.2022.9892091"
    severity: MINOR
    confirmed_by: "DBLP conf/ijcnn/XueY22"
---

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT — verbatim excerpts below are for internal reference only and must not appear in any submitted text -->

> **A2 경고**: 아래 verbatim 발췌는 card 내부 전용입니다. 어떠한 형태로도 원고에 그대로 복사하지 마십시오.

---

## 서지 정보

- **Key**: xue2022fewpositive
- **제목**: Multivariate Time Series Anomaly Detection with Few Positive Samples
- **저자**: Feng Xue, Weizhong Yan [VERIFIED_A: CRITICAL CORRECTION — original "Yifan Xue, Yijie Yan" was completely wrong; confirmed via arXiv 2207.00705 + DBLP]
- **Venue**: IJCNN 2022
- **DOI**: 10.1109/IJCNN55064.2022.9892091 [VERIFIED_A: was verifier-TODO]
- **Pages**: 1–7 [VERIFIED_A: was missing]
- **arXiv**: 2207.00705 (직접 확인 2026-06-11)
- **확인 출처**: arXiv abs 직접 열람 (scout 2026-06-11)
- **경고**: 이 논문은 C-011/C-025 최초성 주장의 **강한 반증 후보 ①** — 반드시 verifier가 본문을 정독하여 반증 성립 여부 확인

---

## Abstract (verbatim)

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT -->

"Given the scarcity of anomalies in real-world applications, the majority of literature has been focusing on modeling normality. The learned representations enable anomaly detection as the normality model is trained to capture certain key underlying data regularities under normal circumstances. In practical settings, particularly industrial time series anomaly detection, we often encounter situations where a large amount of normal operation data is available along with a small number of anomaly events collected over time. This practical situation calls for methodologies to leverage these small number of anomaly events to create a better anomaly detector. In this paper, we introduce two methodologies to address the needs of this practical situation and compared them with recently developed state of the art techniques. Our proposed methods anchor on representative learning of normal operation with autoregressive (AR) model along with loss components to encourage representations that separate normal versus few positive examples. We applied the proposed methods to two industrial anomaly detection datasets and demonstrated effective performance in comparison with approaches from literature. Our study also points out additional challenges with adopting such methods in practical applications."

---

## 핵심 발췌

### 발췌 1 — 설정 정의 (§Abstract)

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT -->

> "we often encounter situations where a large amount of normal operation data is available along with a small number of anomaly events collected over time."

- **§위치**: Abstract
- **지지 Claim**: C-011, C-025 (반증 후보 — 우리 논문의 설정과 동일한 설정을 선행 서술)
- **활용 맥락**: 우리 논문 §2.2에서 이 설정(소수 labeled anomaly + 다수 normal)이 Xue & Yan (IJCNN 2022)에서 이미 다루어졌음을 인정. 우리 novelty를 이 논문과 차별화하는 서술 필수.
- **주의**: 이 설정 서술이 우리 논문 §1/§2.2의 동기 서술과 거의 동일 — 우리 논문 작성 시 이 발췌에서 영감을 받아 쓰지 않도록 주의. 완전히 독립적인 서술 필요.

---

### 발췌 2 — 표현 학습에 라벨 통합 (§Abstract — 핵심 반증 포인트)

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT -->

> "Our proposed methods anchor on representative learning of normal operation with autoregressive (AR) model along with loss components to encourage representations that separate normal versus few positive examples."

- **§위치**: Abstract
- **지지 Claim**: C-011, C-025 (최강 반증 포인트 — "loss components to encourage representations"은 라벨이 표현 학습 손실에 직접 개입함을 의미)
- **활용 맥락**: §2.2에서 우리 논문의 novelty 범위를 좁히는 근거. Xue & Yan이 AR 기반 표현 학습의 손실에 라벨을 통합했다는 점은 우리 주장("표현 학습 기울기에 labeled anomaly를 통합하는 최초")과 충돌. 차별화 포인트 필요:
  - 우리: masked-reconstruction self-distillation + GRL adversarial integration
  - Xue & Yan: autoregressive representation learning + classification loss 병합
  이 구조적 차이(pretext task, gradient reversal vs direct loss)를 명확히 서술.
- **주의**: 이 발췌는 반증 포인트 — 우리 novelty 주장을 약화시킬 수 있음. CLAIM_CITATION_MAP §5.1 재서술 권고를 따라야 함. 발췌 2의 표현을 우리 논문 서술에 절대 차용 금지.

---

### 발췌 3 — 동기 서술 (§Abstract)

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT -->

> "This practical situation calls for methodologies to leverage these small number of anomaly events to create a better anomaly detector."

- **§위치**: Abstract
- **지지 Claim**: C-011, C-025 (보조 — 동기 서술이 우리와 유사)
- **활용 맥락**: §2.2에서 Xue & Yan의 연구 동기가 우리와 유사함을 인정. 차별화 맥락에서 "같은 실용적 동기를 공유하지만 표현 학습 기제가 다름"을 서술.
- **주의**: "leverage these small number of anomaly events" — 우리 논문 §1 서술과 매우 유사. 독립적 언어로 작성 필수.

---

## 활용 절

| 우리 논문 위치 | 활용 방식 | 근거 발췌 |
|-------------|---------|---------|
| §2.2 차별화 | 우리 novelty 주장의 범위를 좁히는 반증 후보로 명시 인용 + 구조적 차별화 | 발췌 2 |
| §2.2 Related Work | 소수 labeled anomaly + 표현학습 통합 MTSAD의 선행 연구로 인용 | 발췌 1, 3 |

---

## 주의사항

1. **최강 반증 후보**: verifier가 PDF 전문을 읽어 "(a) pretext가 autoregressive인지 masked-reconstruction인지", "(b) 라벨이 gradient에 직접 개입하는지(end-to-end) 아니면 분리 학습인지"를 반드시 확인. 반증이 성립하면 novelty 주장을 즉시 좁혀야 함.
2. **두 방법론 세부**: abstract에서 "two methodologies"를 도입한다고 하지만 구체적 차이는 본문에 있음 — verifier 확인.
3. **arXiv-only 논문**: IEEE DOI verifier-TODO — 인용 시 arXiv 식별자 사용.
4. **복사 금지 표현**: "loss components to encourage representations that separate normal versus few positive examples" — 우리 논문 손실 함수 서술에 이 구절을 참조하여 쓰면 표절 위험 + novelty 주장 약화.
