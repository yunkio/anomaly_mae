---
phase: 4
agent: excerpt-curator-3
directives: [T4, R19]
last_modified: 2026-06-11
key: zong2018dagmm
verification_status: VERIFIED_A
verified_by_A: source-verifier-A2
verified_note: "All fields confirmed via OpenReview BJJLHbb0- + DBLP. Author 'Daeki Cho' confirmed as OpenReview (author-submitted) spelling (DBLP has 'Dae-ki Cho' — OpenReview is authoritative for ICLR). Abstract verbatim confirmed."
card_grade: LIGHT
---
# Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection
**경고: abstract verbatim은 검증/표절 대조 전용 — 본문 복사 금지 (A2)**

## 서지 (scout 예비 — 검증 전)
- 저자: Bo Zong, Qi Song, Martin Renqiang Min, Wei Cheng, Cristian Lumezanu, Daeki Cho, Haifeng Chen
- Venue: ICLR 2018 (International Conference on Learning Representations)
- 식별자: OpenReview forum BJJLHbb0- (ICLR 2018은 DOI 없음 — OpenReview가 공식 식별자); DBLP conf/iclr/ZongSMCLCC18
- fetch한 페이지: https://api.openreview.net/notes?id=BJJLHbb0- (OpenReview 공식 API, 2026-06-11)

## Abstract 전문 (verbatim)
Unsupervised anomaly detection on multi- or high-dimensional data is of great importance in both fundamental machine learning research and industrial applications, for which density estimation lies at the core. Although previous approaches based on dimensionality reduction followed by density estimation have made fruitful progress, they mainly suffer from decoupled model learning with inconsistent optimization goals and incapability of preserving essential information in the low-dimensional space. In this paper, we present a Deep Autoencoding Gaussian Mixture Model (DAGMM) for unsupervised anomaly detection. Our model utilizes a deep autoencoder to generate a low-dimensional representation and reconstruction error for each input data point, which is further fed into a Gaussian Mixture Model (GMM). Instead of using decoupled two-stage training and the standard Expectation-Maximization (EM) algorithm, DAGMM jointly optimizes the parameters of the deep autoencoder and the mixture model simultaneously in an end-to-end fashion, leveraging a separate estimation network to facilitate the parameter learning of the mixture model. The joint optimization, which well balances autoencoding reconstruction, density estimation of latent representation, and regularization, helps the autoencoder escape from less attractive local optima and further reduce reconstruction errors, avoiding the need of pre-training. Experimental results on several public benchmark datasets show that, DAGMM significantly outperforms state-of-the-art anomaly detection techniques, and achieves up to 14% improvement based on the standard F1 score.

## 역할 (커버 claim)
- C-060: §4.1.4 SOTA Legacy baseline 표 출처 (DAGMM) — 각주 "simplified variant following TranAD repo, GMM energy removed".
- C-004 / C-012: §1·§2.1 재구성(+밀도추정) 기반 TSAD 계보 괄호 클러스터.
- C-018: §2.1 주석 — related work는 원논문 인용, 실험은 variant 표기.

## 비고
- 모델 약칭: DAGMM. 우리 실험은 TranAD repo의 단순화 변형(GMM energy 제거)을 사용 — 표/각주에서 명시 (C-082 연계).
