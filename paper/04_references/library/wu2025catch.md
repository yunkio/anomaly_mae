---
phase: 4
agent: excerpt-curator-3
directives: [T4, R19]
last_modified: 2026-06-11
key: wu2025catch
verification_status: VERIFIED_A
verified_by_A: source-verifier-A2
verified_note: "All fields confirmed via OpenReview m08aK3xxdJ + DBLP. ICLR 2025 poster confirmed. Abstract verbatim confirmed. arXiv 2410.12261 confirmed."
card_grade: LIGHT
---
# CATCH: Channel-Aware Multivariate Time Series Anomaly Detection via Frequency Patching
**경고: abstract verbatim은 검증/표절 대조 전용 — 본문 복사 금지 (A2)**

## 서지 (scout 예비 — 검증 전)
- 저자: Xingjian Wu, Xiangfei Qiu, Zhengyu Li, Yihang Wang, Jilin Hu, Chenjuan Guo, Hui Xiong, Bin Yang
- Venue: ICLR 2025 Poster (venueid ICLR.cc/2025/Conference; ICLR는 DOI 없음)
- 식별자: OpenReview forum m08aK3xxdJ; submission number 4558; arXiv 2410.12261
- fetch한 페이지: https://api2.openreview.net/notes?id=m08aK3xxdJ (OpenReview 공식 API, 2026-06-11) — arxiv.org/abs/2410.12261 abs 페이지는 본문 렌더 실패로 API 사용

## Abstract 전문 (verbatim)
Anomaly detection in multivariate time series is challenging as heterogeneous subsequence anomalies may occur. Reconstruction-based methods, which focus on learning normal patterns in the frequency domain to detect diverse abnormal subsequences, achieve promising results, while still falling short on capturing fine-grained frequency characteristics and channel correlations. To contend with the limitations, we introduce CATCH, a framework based on frequency patching. We propose to patchify the frequency domain into frequency bands, which enhances its ability to capture fine-grained frequency characteristics. To perceive appropriate channel correlations, we propose a Channel Fusion Module (CFM), which features a patch-wise mask generator and a masked-attention mechanism. Driven by a bi-level multi-objective optimization algorithm, the CFM is encouraged to iteratively discover appropriate patch-wise channel correlations, and to cluster relevant channels while isolating adverse effects from irrelevant channels. Extensive experiments on 10 real-world datasets and 12 synthetic datasets demonstrate that CATCH achieves state-of-the-art performance. We make our code and datasets available at https://github.com/decisionintelligence/CATCH.

## 역할 (커버 claim)
- C-069: §4.1.4 SOTA New baseline 표 출처 (CATCH).
- C-002 / C-014 / C-016: §1·§2.1 — 다변량 채널 상관·연관/대조 계보 괄호 클러스터.

## 비고
- 모델 약칭: CATCH. 우리 실험에서 New SOTA 비교군.
