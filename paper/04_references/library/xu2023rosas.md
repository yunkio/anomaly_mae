---
phase: 4
agent: excerpt-curator-3
directives: [T4, R19]
last_modified: 2026-06-11
key: xu2023rosas
verification_status: VERIFIED_A
verified_by_A: source-verifier-A2
card_grade: LIGHT
abstract_source: arXiv 2307.13239v1 (confirmed)
corrections:
  - field: authors_5th
    wrong: "Ninghui Liu"
    correct: "Ning Liu"
    severity: CRITICAL
    confirmed_by: "arXiv 2307.13239v1; DBLP journals/ipm/XuWPJLW23"
  - field: doi
    wrong: "10.1016/j.ipm.2023.103459 (S2 외부ID — scout 목록의 verifier-TODO)"
    correct: "10.1016/j.ipm.2023.103459"
    severity: MINOR
    note: "DOI confirmed via DBLP — was listed as TODO, now confirmed"
---
# RoSAS: Deep Semi-Supervised Anomaly Detection with Contamination-Resilient Continuous Supervision
**경고: abstract verbatim은 검증/표절 대조 전용 — 본문 복사 금지 (A2)**

## 서지 (scout 예비 — 검증 전)
- 저자: Hongzuo Xu, Yijie Wang, Guansong Pang, Songlei Jian, Ning Liu, Yongjun Wang
  [VERIFIED_A: 5th author corrected from "Ninghui Liu" → "Ning Liu" per arXiv 2307.13239v1 + DBLP]
- Venue: Information Processing & Management 60(5), 2023 (Elsevier)
- DOI: 10.1016/j.ipm.2023.103459 (S2 외부ID — scout 목록의 "Elsevier DOI [verifier-TODO]" 해소 후보)
- arXiv: 2307.13239 / DBLP: journals/ipm/XuWPJLW23
- fetch한 페이지: api.semanticscholar.org (arXiv ID 질의, 2026-06-11)

## Abstract 전문 (verbatim — S2 미러 기준; 공식 페이지 대조는 verifier)
Semi-supervised anomaly detection methods leverage a few anomaly examples to yield drastically improved performance compared to unsupervised models. However, they still suffer from two limitations: 1) unlabeled anomalies (i.e., anomaly contamination) may mislead the learning process when all the unlabeled data are employed as inliers for model training; 2) only discrete supervision information (such as binary or ordinal data labels) is exploited, which leads to suboptimal learning of anomaly scores that essentially take on a continuous distribution. Therefore, this paper proposes a novel semi-supervised anomaly detection method, which devises contamination-resilient continuous supervisory signals. Specifically, we propose a mass interpolation method to diffuse the abnormality of labeled anomalies, thereby creating new data samples labeled with continuous abnormal degrees. Meanwhile, the contaminated area can be covered by new data samples generated via combinations of data with correct labels. A feature learning-based objective is added to serve as an optimization constraint to regularize the network and further enhance the robustness w.r.t. anomaly contamination. Extensive experiments on 11 real-world datasets show that our approach significantly outperforms state-of-the-art competitors by 20%-30% in AUC-PR and obtains more robust and superior performance in settings with different anomaly contamination levels and varying numbers of labeled anomalies.

## 역할 (커버 claim)
- C-032 (선택): §3.1 — "contaminated semi-supervised" 신조어 정의 각주에서 인접 용어 "contamination-resilient"(RoSAS)와의 구분 괄호 인용 (LIGHT-optional).

## 비고
- 모델 약칭: RoSAS. 우리 설정(오염 + 부분 라벨)과 용어가 인접하나 별개 — 신조어 각주 전용, 비교 실험 대상 아님.
