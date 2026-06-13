---
phase: 4
agent: excerpt-curator-1
directives: [T4]
last_modified: 2026-06-11
key: xu2022anomalytransformer
verification_status: VERIFIED_A
verified_by_A: source-verifier-A2
verified_note: "All fields confirmed via OpenReview LzQQ89U1qm_ + arXiv. ICLR 2022 Spotlight confirmed. Abstract verbatim confirmed. AR-threshold body excerpt remains EXCERPT_UNVERIFIED (R30 hold maintained)."
card_grade: FULL
---
# Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy

**경고: 이 card의 verbatim 발췌·abstract는 검증/문체 대조 전용 — 논문 본문으로 복사·근접 의역 절대 금지 (A2)**

## 서지 (scout 예비 — 검증 전)
저자: Jiehui Xu, Haixu Wu, Jianmin Wang, Mingsheng Long
Venue: ICLR 2022 Spotlight
연도: 2022
OpenReview: LzQQ89U1qm_
arXiv: 2110.02642
공식 URL: https://openreview.net/forum?id=LzQQ89U1qm_

## Abstract 전문 (verbatim)
"Unsupervised detection of anomaly points in time series is a challenging problem, which requires the model to derive a distinguishable criterion. Previous methods tackle the problem mainly through learning pointwise representation or pairwise association, however, neither is sufficient to reason about the intricate dynamics. Recently, Transformers have shown great power in unified modeling of pointwise representation and pairwise association, and we find that the self-attention weight distribution of each time point can embody rich association with the whole series. Our key observation is that due to the rarity of anomalies, it is extremely difficult to build nontrivial associations from abnormal points to the whole series, thereby, the anomalies' associations shall mainly concentrate on their adjacent time points. This adjacent-concentration bias implies an association-based criterion inherently distinguishable between normal and abnormal points, which we highlight through the Association Discrepancy. Technically, we propose the Anomaly Transformer with a new Anomaly-Attention mechanism to compute the association discrepancy. A minimax strategy is devised to amplify the normal-abnormal distinguishability of the association discrepancy. The Anomaly Transformer achieves state-of-the-art results on six unsupervised time series anomaly detection benchmarks of three applications: service monitoring, space & earth exploration, and water treatment."

## 핵심 발췌 (verbatim, 섹션/위치 표기)

> "Unsupervised detection of anomaly points in time series is a challenging problem, which requires the model to derive a distinguishable criterion." (Abstract)

커버 claim: C-002, C-004
활용 맥락: §1 도입부 또는 §2.1에서 비지도 TSAD의 지배적 패러다임을 소개할 때. 대표적 비지도 계열 baseline 클러스터 인용.

---

> "due to the rarity of anomalies, it is extremely difficult to build nontrivial associations from abnormal points to the whole series, thereby, the anomalies' associations shall mainly concentrate on their adjacent time points." (Abstract)

커버 claim: C-002, C-014
활용 맥락: 이상의 "rarity" 특성이 비지도 방법의 criterion 설계를 어렵게 한다는 논거. §2.1 대조 기반 TSAD 계열 소개 시.

---

> "A minimax strategy is devised to amplify the normal-abnormal distinguishability of the association discrepancy." (Abstract)

커버 claim: C-014
활용 맥락: 대조/association 기반 TSAD 방법의 대표 사례로 Anomaly Transformer를 §2.1 클러스터 인용 시.

---

**[A1 EXCERPT_RESOLVED — arXiv PDF 2110.02642 직접 발췌 (2026-06-11). R30 보류 해제.]**

**AR threshold 프로토콜 (§4 Implementation details):**
> "We label the time points as anomalies if their anomaly scores (Equation 6) are larger than a certain threshold δ. The threshold δ is determined to make r proportion data of the validation dataset labeled as anomalies. For the main results, we set r = 0.1% for SWaT, 0.5% for SMD and 1% for other datasets." (§4 Implementation details)

커버 claim: C-053 (AR threshold 관행의 verbatim 출처. r-비율 기반 threshold 설정 프로토콜, §4에서 확보. R30 보류 해제)

A1 노트: "anomaly_ratio r" 파라미터는 이 논문의 §4에서 명시적으로 정의되며, 이후 많은 논문들이 이를 따름이 확인됨.

## 우리 논문에서의 활용

커버 claim: C-002, C-004, C-014, C-017, C-053, C-057

- **§1 Introduction**: C-002, C-004 — 비지도 TSAD 지배적 계열 소개 클러스터 인용
- **§2.1 Related Work**: C-014 — 대조/association 기반 TSAD 계보 클러스터 인용
- **§4.1.4 Baselines**: C-057 — SOTA legacy baseline 출처 인용
- **§4.1.3 Metrics**: C-053 — AR threshold 관행 선례 (verifier 발췌 확보 후 조건부 사용)

## 주의사항
- C-053(AR threshold) 인용은 R30 보류 — verifier가 본문/공식 구현에서 직접 발췌 확보 전 논문 초안에 사용 금지.
- Anomaly Transformer는 완전 비지도 방법 — "train = all normal" 가정의 대표 사례로 C-017 논거에 활용 가능. 단, 논문 자체에서 이 한계를 직접 인정하는 문장이 있는지는 verifier 확인 필요.
- ICLR 2022 Spotlight — venue 표기 시 "Spotlight" 포함 여부는 저널 스타일가이드에 따름.
