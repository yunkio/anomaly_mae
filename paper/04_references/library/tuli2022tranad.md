---
phase: 4
agent: excerpt-curator-3
directives: [T4, R19]
last_modified: 2026-06-11
key: tuli2022tranad
verification_status: VERIFIED_A
verified_by_A: source-verifier-A2
verified_note: "All fields confirmed via arXiv + DBLP. PVLDB 15(6):1201-1214, DOI 10.14778/3514061.3514067 confirmed. Abstract verbatim confirmed."
card_grade: LIGHT
---
# TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data
**경고: abstract verbatim은 검증/표절 대조 전용 — 본문 복사 금지 (A2)**

## 서지 (scout 예비 — 검증 전)
- 저자: Shreshth Tuli, Giuliano Casale, Nicholas R. Jennings
- Venue: PVLDB 15(6):1201–1214, 2022 (scout 목록; arXiv 페이지 표기는 "Accepted in VLDB 2022")
- DOI: 10.14778/3514061.3514067
- arXiv: 2201.07284 (v1 2022-01-18, v6 2022-05-14)
- DBLP: journals/pvldb/TuliCJ22
- fetch한 페이지: https://arxiv.org/abs/2201.07284 (2026-06-11 직접 열람)

## Abstract 전문 (verbatim)
Efficient anomaly detection and diagnosis in multivariate time-series data is of great importance for modern industrial applications. However, building a system that is able to quickly and accurately pinpoint anomalous observations is a challenging problem. This is due to the lack of anomaly labels, high data volatility and the demands of ultra-low inference times in modern applications. Despite the recent developments of deep learning approaches for anomaly detection, only a few of them can address all of these challenges. In this paper, we propose TranAD, a deep transformer network based anomaly detection and diagnosis model which uses attention-based sequence encoders to swiftly perform inference with the knowledge of the broader temporal trends in the data. TranAD uses focus score-based self-conditioning to enable robust multi-modal feature extraction and adversarial training to gain stability. Additionally, model-agnostic meta learning (MAML) allows us to train the model using limited data. Extensive empirical studies on six publicly available datasets demonstrate that TranAD can outperform state-of-the-art baseline methods in detection and diagnosis performance with data and time-efficient training. Specifically, TranAD increases F1 scores by up to 17%, reducing training times by up to 99% compared to the baselines.

## 역할 (커버 claim)
- C-058: §4.1.4 SOTA Legacy baseline 표 출처 (TranAD).
- C-004 / C-015: §1·§2.1 자기지도 TSAD 계보 괄호 클러스터.
- C-018 / C-082: DAGMM "simplified variant following TranAD repo" provenance 각주 (repo github.com/imperial-qore/TranAD).

## 비고
- 모델 약칭: TranAD. 우리 실험에서 Legacy SOTA 비교군.
- arXiv 페이지는 venue를 "Accepted in VLDB 2022"로만 표기 — PVLDB 권·호·쪽수는 DBLP/scout 목록 기준 (verifier가 vldb.org에서 최종 확정).
