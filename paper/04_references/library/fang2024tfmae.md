---
phase: 4
agent: excerpt-curator-3
directives: [T4, R19]
last_modified: 2026-06-11
key: fang2024tfmae
verification_status: VERIFIED_A
verified_by_A: 2026-06-11
corrected_fields: [pages_added]
card_grade: LIGHT
abstract_source: semanticscholar-mirror (공식 ieeexplore.ieee.org 418 차단)
---
# Temporal-Frequency Masked Autoencoders for Time Series Anomaly Detection
**경고: abstract verbatim은 검증/표절 대조 전용 — 본문 복사 금지 (A2)**

## 서지 (scout 예비 — 검증 전)
- 저자: Yuchen Fang, Jiandong Xie, Yan Zhao, Lu Chen, Yunjun Gao, Kai Zheng
- Venue: IEEE ICDE 2024 (International Conference on Data Engineering), pp.1228–1241
- DOI: 10.1109/ICDE60146.2024.00099
- DBLP: conf/icde/FangXZ0G024 / ieeexplore 문서번호 10597757 / arXiv 버전 없음 (ICDE 본이 유일 공식본)
- fetch한 페이지: api.semanticscholar.org (DOI 질의, 2026-06-11) — ieeexplore.ieee.org/document/10597757 은 HTTP 418 차단

## Abstract 전문 (verbatim — S2 미러 기준; 공식 페이지 대조는 verifier)
In the era of observability, massive amounts of time series data have been collected to monitor the running status of the target system, where anomaly detection serves to identify observations that differ significantly from the remaining ones and is of utmost importance to enable value extraction from such data. While existing reconstruction-based methods have demonstrated favorable detection capabilities in the absence of labeled data, they still encounter issues of training bias on abnormal times and distribution shifts within time series. To address these issues, we propose a simple yet effective Temporal-Frequency Masked AutoEncoder (TFMAE) to detect anomalies in time series through a contrastive criterion. Specifically, TFMAE uses two Transformer-based autoencoders that respectively incorporate a window-based temporal masking strategy and an amplitude-based frequency masking strategy to learn knowledge without abnormal bias and reconstruct anomalies by the extracted normal information. Moreover, the dual autoencoder undergoes training through a contrastive objective function, which minimizes the discrepancy of representations from temporal-frequency masked autoencoders to highlight anomalies, as it helps alleviate the negative impact of distribution shifts. Finally, to prevent over-fitting, TFMAE adopts adversarial training during the training phase. Extensive experiments conducted on seven datasets provide evidence that our model is able to surpass the state-of-the-art in terms of anomaly detection accuracy.

## 역할 (커버 claim)
- C-063: §4.1.4 SOTA New baseline 표 출처 (TFMAE).
- C-031: §2.3 단락 3 — 시계열 MAE 사례 1문장 괄호 인용 (§2.3 유일 언급).

## 비고
- 모델 약칭: TFMAE. 우리 실험에서 New SOTA 비교군이자 시계열 MAE 최근접 선행.
- 저자 홈페이지(zheng-kai.com)에 PDF 존재 기록 있음 (CLAIM_CITATION_MAP C-031) — verifier가 공식 abstract 대조 시 활용 가능.
