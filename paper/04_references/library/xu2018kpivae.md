---
phase: 4
agent: excerpt-curator-1
directives: [T4]
last_modified: 2026-06-11
key: xu2018kpivae
verification_status: VERIFIED_A
verified_by_A: source-verifier-A1
verified_by_prior: source-verifier-A2
card_grade: FULL
excerpt_access: abstract_pending
corrections:
  - field: authors
    wrong: "24-author list including Taoran Pei, Duogang Feng, Feng Shi, Zijie Zhao, Naichen Shi, Fang Zhou, Yong Cai, Hongyu Li, Fanxi Liu, Guangzhou Ji, Qingwei Lin, Dongmei Zhang (and others)"
    correct: "Haowen Xu; Wenxiao Chen; Nengwen Zhao; Zeyan Li; Jiahao Bu; Zhihan Li; Ying Liu; Youjian Zhao; Dan Pei; Yang Feng; Jie Chen; Zhaogang Wang; Honglin Qiao (13 authors)"
    severity: CRITICAL
    confirmed_by: "arXiv export 1802.03903; DBLP conf/www/XuCZLBLLZPFCWQ18"
    note: "11 spurious author names removed — these are not in the paper"
  - field: pages
    wrong: missing
    correct: "187–196"
    severity: MINOR
    confirmed_by: "DBLP conf/www/XuCZLBLLZPFCWQ18"
---
# Unsupervised Anomaly Detection via Variational Auto-Encoder for Seasonal KPIs in Web Applications

**경고: 이 card의 verbatim 발췌·abstract는 검증/문체 대조 전용 — 논문 본문으로 복사·근접 의역 절대 금지 (A2)**

## 서지 (scout 예비 — 검증 전)
저자: Haowen Xu, Wenxiao Chen, Nengwen Zhao, Zeyan Li, Jiahao Bu, Zhihan Li, Ying Liu, Youjian Zhao, Dan Pei, Yang Feng, Jie Chen, Zhaogang Wang, Honglin Qiao
[VERIFIED_A CRITICAL CORRECTION: Original card listed 24 authors. Correct count is 13 authors per DBLP + arXiv. Removed 11 spurious names.]
Venue: WWW 2018 (The Web Conference 2018)
연도: 2018
Pages: 187–196 [VERIFIED_A: confirmed via DBLP]
DOI: 10.1145/3178876.3185996
arXiv: 1802.03903
공식 URL: https://arxiv.org/abs/1802.03903

## Abstract 전문 (verbatim — arXiv PDF 1802.03903 직접 추출 2026-06-11)

[A1 RESOLVED] arXiv PDF 다운로드 후 ABSTRACT 절 전사:

"To ensure undisrupted business, large Internet companies need to closely monitor various KPIs (e.g., Page Views, number of online users, and number of orders) of its Web applications, to accurately detect anomalies and trigger timely troubleshooting/mitigation. However, anomaly detection for these seasonal KPIs with various patterns and data quality has been a great challenge, especially without labels. In this paper, we proposed Donut, an unsupervised anomaly detection algorithm based on VAE. Thanks to a few of our key techniques, Donut greatly outperforms a state-of-arts supervised ensemble approach and a baseline VAE approach, and its best F-scores range from 0.75 to 0.9 for the studied KPIs from a top global Internet company. We come up with a novel KDE interpretation of reconstruction for Donut, making it the first VAE-based anomaly detection algorithm with solid theoretical explanation."

## 핵심 발췌 (verbatim — A1 EXCERPT_RESOLVED)

**PA 프로토콜 정의 (§4.2 "Metrics" 단락):**
> "We instead use a simple strategy: if any point in an anomaly segment in the ground truth can be detected by a chosen threshold, we say this segment is detected correctly, and all points in this segment are treated as if they can be detected by this threshold. Meanwhile, the points outside the anomaly segments are treated as usual." (§4.2)

커버 claim: C-051 (PA 원전 — §4.2 Metrics 단락에서 verbatim 확보. "if any point … all points in this segment" 규칙이 point adjustment의 정의)

## 우리 논문에서의 활용

커버 claim: C-051

- **§4.1.3 Evaluation Metrics (보조)**: C-051 — PA F1의 원전으로 괄호 인용. "point adjustment는 Xu et al. (WWW 2018)에서 도입" 1문장 + Kim et al. (AAAI 2022)의 PA 비판 인용으로 이어지는 흐름.

## 주의사항
- 저자는 13인(24인이 아님) — 인용 시 "Xu et al." 표기. 단, 다른 Xu et al. 논문들(특히 xu2022anomalytransformer)과 혼동되지 않도록 연도 명시 필수.
- PA(point adjustment) 원전으로 인용되는 이유: K=0에서의 segment-level adjustment가 이 논문에서 실용적으로 제안됨. 우리 논문에서 PA를 비판적으로 언급할 때 이 논문이 PA의 출처라는 점을 명확히 해야 함.
- abstract 전문 미확보 — EXCERPT_UNVERIFIED 상태 유지. verifier 필수 작업.
