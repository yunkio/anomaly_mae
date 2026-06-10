---
phase: 4
agent: excerpt-curator-3
directives: [T4, R19]
last_modified: 2026-06-11
key: luo2024moderntcn
verification_status: VERIFIED_A
verified_by_A: source-verifier-A2
verified_note: "Author names confirmed: Donghao Luo, Xue Wang (OpenReview lowercase was provisional). ICLR 2024 confirmed. Spotlight designation: DBLP does not confirm spotlight — treat as VERIFY_REQUIRED if used in ms."
card_grade: LIGHT
---
# ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis
**경고: abstract verbatim은 검증/표절 대조 전용 — 본문 복사 금지 (A2)**

## 서지 (scout 예비 — 검증 전)
- 저자 (OpenReview API 표기 그대로): Luo donghao, wang xue
  - 주의: OpenReview 프로필 표기가 소문자/성명 역순 형태 — 논문 PDF상 정식 표기(Donghao Luo, Xue Wang 추정)는 verifier가 camera-ready PDF에서 확정.
- Venue: ICLR 2024 spotlight (venueid ICLR.cc/2024/Conference; ICLR는 DOI 없음)
- 식별자: OpenReview forum vpJMJerXHU; submission number 5228; arXiv 버전 미발견 (OpenReview가 유일 공식본)
- fetch한 페이지: https://api2.openreview.net/notes?id=vpJMJerXHU (OpenReview 공식 API, 2026-06-11)

## Abstract 전문 (verbatim)
Recently, Transformer-based and MLP-based models have emerged rapidly and won dominance in time series analysis. In contrast, convolution is losing steam in time series tasks nowadays for inferior performance. This paper studies the open question of how to better use convolution in time series analysis and makes efforts to bring convolution back to the arena of time series analysis. To this end, we modernize the traditional TCN and conduct time series related modifications to make it more suitable for time series tasks. As the outcome, we propose ModernTCN and successfully solve this open question through a seldom-explored way in time series community. As a pure convolution structure, ModernTCN still achieves the consistent state-of-the-art performance on five mainstream time series analysis tasks while maintaining the efficiency advantage of convolution-based models, therefore providing a better balance of efficiency and performance than state-of-the-art Transformer-based and MLP-based models. Our study further reveals that, compared with previous convolution-based models, our ModernTCN has much larger effective receptive fields (ERFs), therefore can better unleash the potential of convolution in time series analysis. Code is available at this repository: https://github.com/luodhhh/ModernTCN.

## 역할 (커버 claim)
- C-068: §4.1.4 SOTA New baseline 표 출처 (ModernTCN, ICLR 2024 Spotlight).

## 비고
- 모델 약칭: ModernTCN. 우리 실험에서 New SOTA 비교군 (AD task로 사용).
