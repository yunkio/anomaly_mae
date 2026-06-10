---
phase: 4
agent: excerpt-curator-2
directives: [T4]
last_modified: 2026-06-11
key: ruff2020deepsad
verification_status: VERIFIED_A
verified_by_A: source-verifier-A2
verified_note: "All bibliographic fields confirmed via arXiv + OpenReview HkgH0TEYwH + DBLP. Abstract verbatim confirmed. SAD loss objective (§3) remains EXCERPT_UNVERIFIED — requires ICLR paper body."
card_grade: FULL
excerpt_access: abstract_only
---

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT — verbatim excerpts below are for internal reference only and must not appear in any submitted text -->

> **A2 경고**: 아래 verbatim 발췌는 card 내부 전용입니다. 어떠한 형태로도 원고에 그대로 복사하지 마십시오.

---

## 서지 정보

- **Key**: ruff2020deepsad
- **제목**: Deep Semi-Supervised Anomaly Detection
- **저자**: Lukas Ruff, Robert A. Vandermeulen, Nico Görnitz, Alexander Binder, Emmanuel Müller, Klaus-Robert Müller, Marius Kloft
- **Venue**: ICLR 2020
- **arXiv**: 1906.02694 ("Published as a conference paper at ICLR 2020" — 직접 확인 2026-06-11)
- **확인 출처**: arXiv abs 직접 열람

---

## Abstract (verbatim)

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT -->

"Deep approaches to anomaly detection have recently shown promising results over shallow methods on large and complex datasets. Typically anomaly detection is treated as an unsupervised learning problem. In practice however, one may have---in addition to a large set of unlabeled samples---access to a small pool of labeled samples, e.g. a subset verified by some domain expert as being normal or anomalous. Semi-supervised approaches to anomaly detection aim to utilize such labeled samples, but most proposed methods are limited to merely including labeled normal samples. Only a few methods take advantage of labeled anomalies, with existing deep approaches being domain-specific. In this work we present Deep SAD, an end-to-end deep methodology for general semi-supervised anomaly detection. We further introduce an information-theoretic framework for deep anomaly detection based on the idea that the entropy of the latent distribution for normal data should be lower than the entropy of the anomalous distribution, which can serve as a theoretical interpretation for our method. In extensive experiments on MNIST, Fashion-MNIST, and CIFAR-10, along with other anomaly detection benchmark datasets, we demonstrate that our method is on par or outperforms shallow, hybrid, and deep competitors, yielding appreciable performance improvements even when provided with only little labeled data."

---

## 핵심 발췌

### 발췌 1 — labeled anomaly를 쓰는 방법의 희소성 (§Abstract)

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT -->

> "most proposed methods are limited to merely including labeled normal samples. Only a few methods take advantage of labeled anomalies, with existing deep approaches being domain-specific."

- **§위치**: Abstract
- **지지 Claim**: C-021 (비시계열 영역의 semi-supervised AD 사례 + labeled anomaly 활용 방법 희소성), C-011, C-025 (차별화 인용)
- **활용 맥락**: §2.2에서 labeled anomaly를 활용하는 방법이 드물다는 논거. Deep SAD를 "general semi-supervised AD에서 labeled anomaly를 사용한 비시계열 선행 연구"로 소개하면서 우리 모델이 시계열로 이 갭을 채운다고 주장.
- **주의**: Deep SAD는 이미지(MNIST, CIFAR-10) 기반 — "domain-specific"이라는 자기 평가도 주목. 우리 모델이 시계열에 특화되었음을 명시.

---

### 발췌 2 — end-to-end 반지도 이상탐지 (§Abstract)

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT -->

> "In this work we present Deep SAD, an end-to-end deep methodology for general semi-supervised anomaly detection."

- **§위치**: Abstract
- **지지 Claim**: C-021 (end-to-end 선행 사례), C-011, C-025 (최초성 방어 — 비시계열 선행 존재 인정)
- **활용 맥락**: §2.2에서 "end-to-end semi-supervised AD"가 이미 비시계열에서 시도되었음을 인정하면서, 우리 모델이 시계열에 이를 최초로 도입했다고 서술. 최초성 주장의 범위를 "TSAD" 도메인으로 한정하는 근거.
- **주의**: C-011/C-025 반증 후보 맥락에서, 비시계열 선행이 존재함을 명시하는 것이 재서술 권고 사항. Deep SAD는 "end-to-end" 면에서 가장 관련성 높은 비시계열 선행.

---

### 발췌 3 — 정보이론 프레임워크 (§Abstract)

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT -->

> "the entropy of the latent distribution for normal data should be lower than the entropy of the anomalous distribution"

- **§위치**: Abstract
- **지지 Claim**: C-021 (방법론적 차이 이해 — 우리 GRL 방식과 비교)
- **활용 맥락**: Deep SAD의 이론적 기반(엔트로피 기반 hypersphere 목적함수)이 우리 GRL adversarial 방식과 근본적으로 다름을 설명하는 배경 맥락. 직접 인용보다는 "Deep SAD uses an entropy-based objective" 수준의 paraphrase.
- **주의**: 엔트로피 기반 hypersphere 목적함수의 구체적 공식(SAD loss)은 verifier가 PDF §3에서 확보 필요.

---

### 발췌 4 — 소수 라벨 효과 (§Abstract)

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT -->

> "yielding appreciable performance improvements even when provided with only little labeled data."

- **§위치**: Abstract
- **지지 Claim**: C-021 (보조), C-079 (labeled anomaly 비율이 낮아도 유효함의 선행 근거)
- **활용 맥락**: §4.4 label sparsity 분석에서 "소수 라벨로도 성능 향상이 가능하다"는 주장의 비시계열 선행 근거로 활용 가능.
- **주의**: Deep SAD는 이미지 데이터 — 시계열로의 일반화 주장 시 우리 실험 결과가 주 근거여야 함. Deep SAD는 간접 지지로만 사용.

---

## 활용 절

| 우리 논문 위치 | 활용 방식 | 근거 발췌 |
|-------------|---------|---------|
| §2.2 Related Work | labeled anomaly를 쓰는 semi-supervised AD의 비시계열 선행 (괄호 인용) | 발췌 1, 2 |
| §2.2 최초성 방어 | 비시계열에서 end-to-end 선행 존재 인정 + 시계열 도메인 최초성 주장 | 발췌 2 |
| §4.4 보조 | 소수 라벨 효과 선행 근거 (선택적) | 발췌 4 |

---

## 주의사항

1. **도메인 한정**: Deep SAD는 이미지(MNIST/CIFAR-10) — 시계열 성능 외삽 금지. 모든 인용은 "in image/non-time-series domain" 한정.
2. **SAD 목적함수 미확보**: abstract에 수식 없음. verifier가 PDF §3에서 SAD loss 공식(hypersphere center c, 라벨 anomaly를 center에서 멀리 밀어내는 항) 발췌 필요.
3. **최초성 주장 맥락**: Deep SAD를 인용하면 "labeled anomaly 활용 end-to-end AD가 이미 비시계열에 존재했다"고 인정하는 것 — C-011/C-025 재서술 시 이 사실을 솔직하게 서술해야 함.
4. **복사 금지 표현**: "most proposed methods are limited to merely including labeled normal samples" — 임팩트 있는 구절. 우리 논문 §1/§2.2에서 유사한 논지를 전개할 때 독립적으로 서술 필요.
