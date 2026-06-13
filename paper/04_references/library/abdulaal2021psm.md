---
phase: 4
agent: excerpt-curator-3
directives: [T4, R19]
last_modified: 2026-06-11
key: abdulaal2021psm
verification_status: VERIFIED_A
verified_by_A: 2026-06-11
corrected_fields: [pages]
card_grade: LIGHT
abstract_source: semanticscholar-mirror (공식 dl.acm.org 403 차단)
---
# Practical Approach to Asynchronous Multivariate Time Series Anomaly Detection and Localization
**경고: abstract verbatim은 검증/표절 대조 전용 — 본문 복사 금지 (A2)**

## 서지 (scout 예비 — 검증 전)
- 저자: Ahmed Abdulaal, Zhuanghua Liu, Tomer Lancewicki
- Venue: KDD 2021, pp.2485–2494
- DOI: 10.1145/3447548.3467174
- DBLP: conf/kdd/AbdulaalLL21
- fetch한 페이지: api.semanticscholar.org (DOI 질의, 2026-06-11) — dl.acm.org/doi/10.1145/3447548.3467174 는 403

## Abstract 전문 (verbatim — S2 미러 기준; 공식 페이지 대조는 verifier)
Engineers at eBay utilize robust methods in monitoring IT system signals for anomalies. However, the growing scale of signals, both in volumes and dimensions, overpowers traditional statistical state-space or supervised learning tools. Thus, state-of-the-art methods based on unsupervised deep learning are sought in recent research. However, we experienced flaws when implementing those methods, such as requiring partial supervision and weaknesses to high dimensional datasets, among other reasons discussed in this paper. We propose a practical approach for inferring anomalies from large multivariate sets. We observe an abundance of time series in real-world applications, which exhibit asynchronous and consistent repetitive variations, such as IT, weather, utility, and transportation. Our solution is designed to leverage this behavior. The solution utilizes spectral analysis on the latent representation of a pre-trained autoencoder to extract dominant frequencies across the signals, which are then used in a subsequent network that learns the phase shifts across the signals and produces a synchronized representation of the raw multivariate. Random subsets of the synchronous multivariate are then fed into an array of autoencoders learning to minimize the quantile reconstruction losses, which are then used to infer and localize anomalies based on a majority vote. We benchmark this method against state-of-the-art approaches on public datasets and eBay's data using their referenced evaluation methods. Furthermore, we address the limitations of the referenced evaluation methods and propose a more realistic evaluation method.

## 역할 (커버 claim)
- C-043: §4.1.1 Table 1 — PSM 데이터셋 출처 (실험 섹션 전용).

## 비고
- 데이터셋 약칭: PSM (Pooled Server Metrics, eBay 공개). 우리 실험 5개 base 데이터셋 중 하나.
