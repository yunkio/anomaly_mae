---
phase: 4
agent: excerpt-curator-2
directives: [T4]
last_modified: 2026-06-11
key: darban2024dacad
verification_status: VERIFIED_A
verified_by_A: 2026-06-11
corrected_fields: [doi_added]
card_grade: FULL
excerpt_access: abstract_only
---

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT — verbatim excerpts below are for internal reference only and must not appear in any submitted text -->

> **A2 경고**: 아래 verbatim 발췌는 card 내부 전용입니다. 어떠한 형태로도 원고에 그대로 복사하지 마십시오.

---

## 서지 정보

- **Key**: darban2024dacad
- **제목**: DACAD: Domain Adaptation Contrastive Learning for Anomaly Detection in Multivariate Time Series
- **저자**: Zahra Zamanzadeh Darban, Yiyuan Yang, Geoffrey I. Webb, Charu C. Aggarwal, Qingsong Wen, Shirui Pan, Mahsa Salehi
- **Venue**: IEEE Transactions on Knowledge and Data Engineering, vol. 37, no. 8, pp. 4485–4496, August 2025
- **DOI**: 10.1109/TKDE.2025.3569909 [A1 추가 — arXiv journal-ref에서 확인]
- **arXiv**: 2404.11269 (v4: 2025-09-07)
- **확인 출처**: arXiv abs 열람 (journal-ref) + DBLP 검색
- [A1] venue TKDE 2025 최종 확정. DOI 10.1109/TKDE.2025.3569909 추가.

---

## Abstract (verbatim)

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT -->

"In time series anomaly detection (TSAD), the scarcity of labeled data poses a challenge to the development of accurate models. Unsupervised domain adaptation (UDA) offers a solution by leveraging labeled data from a related domain to detect anomalies in an unlabeled target domain. However, existing UDA methods assume consistent anomalous classes across domains. To address this limitation, we propose a novel Domain Adaptation Contrastive learning model for Anomaly Detection in multivariate time series (DACAD), combining UDA with contrastive learning. DACAD utilizes an anomaly injection mechanism that enhances generalization across unseen anomalous classes, improving adaptability and robustness. Additionally, our model employs supervised contrastive loss for the source domain and self-supervised contrastive triplet loss for the target domain, ensuring comprehensive feature representation learning and domain-invariant feature extraction. Finally, an effective Center-based Entropy Classifier (CEC) accurately learns normal boundaries in the source domain. Extensive evaluations on multiple real-world datasets and a synthetic dataset highlight DACAD's superior performance in transferring knowledge across domains and mitigating the challenge of limited labeled data in TSAD."

---

## 핵심 발췌

### 발췌 1 — 도메인 적응 설정 정의 (§Abstract)

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT -->

> "leveraging labeled data from a related domain to detect anomalies in an unlabeled target domain."

- **§위치**: Abstract
- **지지 Claim**: C-011, C-025 (보조 차별화 — DACAD는 transfer/domain-adaptation 설정, 우리는 단일 도메인 설정)
- **활용 맥락**: §2.2에서 DACAD를 "source domain의 labeled anomaly를 contrastive 표현학습에 활용하는 domain-adaptation 방법"으로 소개하되, 우리 설정(train/test가 같은 도메인, labeled anomaly가 표현 학습에 직접 관여)과의 차이를 명시.
- **주의**: DACAD의 labeled anomaly는 source domain에 있고 target domain에는 없음 — 우리 설정(target domain = train domain에 소수 labeled anomaly 존재)과 다른 설정.

---

### 발췌 2 — Supervised contrastive loss for labeled anomaly (§Abstract — 핵심)

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT -->

> "our model employs supervised contrastive loss for the source domain and self-supervised contrastive triplet loss for the target domain, ensuring comprehensive feature representation learning and domain-invariant feature extraction."

- **§위치**: Abstract
- **지지 Claim**: C-011, C-025 (보조 반증 후보 — labeled anomaly가 supervised contrastive loss를 통해 표현 학습에 직접 개입)
- **활용 맥락**: §2.2 차별화 서술: DACAD는 domain-adaptation + contrastive learning으로 labeled anomaly를 표현에 통합하는 반면, 우리는 masked-reconstruction self-distillation + gradient reversal(GRL)로 통합 — 표현 학습 mechanism 및 설정이 다름을 서술.
- **주의**: DACAD의 supervised contrastive loss는 source domain에만 적용 — target domain(우리가 탐지해야 하는 도메인)에는 라벨 없음. 이 점이 우리 설정과의 핵심 차이.

---

### 발췌 3 — Anomaly injection mechanism (§Abstract)

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT -->

> "DACAD utilizes an anomaly injection mechanism that enhances generalization across unseen anomalous classes, improving adaptability and robustness."

- **§위치**: Abstract
- **지지 Claim**: C-011, C-025 (보조)
- **활용 맥락**: §2.2에서 DACAD의 이상 주입(anomaly injection) 메커니즘이 합성 이상을 통한 일반화를 목표로 한다고 설명. 우리 모델은 이상 주입 없이 실제 labeled anomaly를 GRL로 통합 — 이 차이가 설정의 차이(synthetic vs real labels)를 반영.
- **주의**: "anomaly injection"은 SDMAE(합성 이상 생성 + 재구성 감독)와도 유사한 개념이지만 다른 메커니즘 — 혼동 방지.

---

### 발췌 4 — 설정의 cross-domain 성격 (§Abstract)

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT -->

> "DACAD's superior performance in transferring knowledge across domains and mitigating the challenge of limited labeled data in TSAD."

- **§위치**: Abstract
- **지지 Claim**: C-011, C-025 (보조 차별화 — 전이 설정 vs 단일 도메인 설정)
- **활용 맥락**: DACAD가 "cross-domain transfer" 성능을 목표로 한다는 점을 명시 — 우리 논문은 단일 도메인 내에서 labeled anomaly를 활용하는 것이 목적이므로 직접 경쟁 관계가 아님.
- **주의**: DACAD를 보조 반증 후보로 취급(SCOUT에서도 "(보조) 차별화" 분류) — 우리 논문의 core novelty를 직접 반증하는 것이 아닌 인접 방향으로 위치.

---

## 활용 절

| 우리 논문 위치 | 활용 방식 | 근거 발췌 |
|-------------|---------|---------|
| §2.2 Related Work | domain-adaptation + labeled anomaly + contrastive 표현학습 MTSAD 선행으로 소개 | 발췌 1, 2 |
| §2.2 차별화 | transfer 설정(DACAD) vs 단일 도메인 설정(우리) 구분 명시 | 발췌 1, 4 |

---

## 주의사항

1. **FULL-cond 등급**: SCOUT에서 "venue 확정 전 arXiv 표기" 조건부. TKDE 2025 게재 확인이 되었다면(v4 기준) "IEEE TKDE 37(8), 2025"로 인용 가능 — verifier 최종 확인.
2. **보조 차별화 위치**: DACAD는 최강 반증 후보가 아닌 "인접" 후보 — Xue & Yan(강한 반증)과 달리 domain-adaptation 설정 차이로 비교적 쉽게 차별화 가능.
3. **contrastive vs GRL**: DACAD의 supervised contrastive loss와 우리 GRL의 차이는 메커니즘 수준이 아닌 설정 수준(cross-domain vs single-domain, 합성 이상 vs 실제 labeled anomaly)에서 설명하는 것이 더 명확.
4. **복사 금지 표현**: "supervised contrastive loss for the source domain" — 이 구절이 우리 방법 서술에 혼입되면 표절 + 방법론 오서술.
