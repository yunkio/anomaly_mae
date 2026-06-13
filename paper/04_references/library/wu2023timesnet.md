---
phase: 4
agent: excerpt-curator-3
directives: [T4, R19]
last_modified: 2026-06-11
key: wu2023timesnet
verification_status: VERIFIED_A
verified_by_A: source-verifier-A2
verified_note: "All fields confirmed via OpenReview ju_Uqw384Oq + DBLP. arXiv 2210.02186 confirmed. ICLR 2023 poster confirmed. Abstract verbatim confirmed."
card_grade: LIGHT
---
# TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis
**경고: abstract verbatim은 검증/표절 대조 전용 — 본문 복사 금지 (A2)**

## 서지 (scout 예비 — 검증 전)
- 저자: Haixu Wu, Tengge Hu, Yong Liu, Hang Zhou, Jianmin Wang, Mingsheng Long
- Venue: ICLR 2023 poster (venueid ICLR.cc/2023/Conference; ICLR는 DOI 없음)
- 식별자: OpenReview forum ju_Uqw384Oq; arXiv 2210.02186은 scout 목록상 [verifier-TODO] (미확인)
- fetch한 페이지: https://api.openreview.net/notes?id=ju_Uqw384Oq (OpenReview 공식 API, 2026-06-11)

## Abstract 전문 (verbatim)
Time series analysis is of immense importance in extensive applications, such as weather forecasting, anomaly detection, and action recognition. This paper focuses on temporal variation modeling, which is the common key problem of extensive analysis tasks. Previous methods attempt to accomplish this directly from the 1D time series, which is extremely challenging due to the intricate temporal patterns. Based on the observation of multi-periodicity in time series, we ravel out the complex temporal variations into the multiple intraperiod- and interperiod-variations. To tackle the limitations of 1D time series in representation capability, we extend the analysis of temporal variations into the 2D space by transforming the 1D time series into a set of 2D tensors based on multiple periods. This transformation can embed the intraperiod- and interperiod-variations into the columns and rows of the 2D tensors respectively, making the 2D-variations to be easily modeled by 2D kernels. Technically, we propose the TimesNet with TimesBlock as a task-general backbone for time series analysis. TimesBlock can discover the multi-periodicity adaptively and extract the complex temporal variations from transformed 2D tensors by a parameter-efficient inception block. Our proposed TimesNet achieves consistent state-of-the-art in five mainstream time series analysis tasks, including short- and long-term forecasting, imputation, classification, and anomaly detection. Code is available at this repository: https://github.com/thuml/TimesNet.

## 역할 (커버 claim)
- C-065: §4.1.4 SOTA New baseline 표 출처 (TimesNet).
- C-004 / C-015: §1·§2.1 자기지도/일반 백본 TSAD 계보 괄호 클러스터.

## 비고
- 모델 약칭: TimesNet. 우리 실험에서 New SOTA 비교군 (AD task로 사용).
