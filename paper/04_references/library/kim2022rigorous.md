---
phase: 4
agent: excerpt-curator-2
directives: [T4]
last_modified: 2026-06-11
key: kim2022rigorous
verification_status: VERIFIED_A
verified_by_A: 2026-06-11
corrected_fields: [pa_pct_k_excerpt_resolved]
card_grade: FULL
excerpt_access: abstract_only
---

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT — verbatim excerpts below are for internal reference only and must not appear in any submitted text -->

> **A2 경고**: 아래 verbatim 발췌는 card 내부 전용입니다. 어떠한 형태로도 원고에 그대로 복사하지 마십시오.

---

## 서지 정보

- **Key**: kim2022rigorous
- **제목**: Towards a Rigorous Evaluation of Time-Series Anomaly Detection
- **저자**: Siwon Kim, Kukjin Choi, Hyun-Soo Choi, Byunghan Lee, Sungroh Yoon
- **Venue**: AAAI 2022, Proc. AAAI Conf. Artif. Intell. 36(7):7194–7201
- **DOI**: 10.1609/aaai.v36i7.20680
- **확인 출처**: ojs.aaai.org 공식 페이지 직접 열람 (2026-06-11)

---

## Abstract (verbatim)

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT -->

"In recent years, proposed studies on time-series anomaly detection (TAD) report high F1 scores on benchmark TAD datasets, giving the impression of clear improvements in TAD. However, most studies apply a peculiar evaluation protocol called point adjustment (PA) before scoring. In this paper, we theoretically and experimentally reveal that the PA protocol has a great possibility of overestimating the detection performance; even a random anomaly score can easily turn into a state-of-the-art TAD method. Therefore, the comparison of TAD methods after applying the PA protocol can lead to misguided rankings. Furthermore, we question the potential of existing TAD methods by showing that an untrained model obtains comparable detection performance to the existing methods even when PA is forbidden. Based on our findings, we propose a new baseline and an evaluation protocol. We expect that our study will help a rigorous evaluation of TAD and lead to further improvement in future researches."

---

## 핵심 발췌

### 발췌 1 — PA 과대평가 주장 (§Abstract)

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT -->

> "the PA protocol has a great possibility of overestimating the detection performance; even a random anomaly score can easily turn into a state-of-the-art TAD method."

- **§위치**: Abstract
- **지지 Claim**: C-050 (PA F1 과대평가 위험: 무작위 점수도 SOTA로 둔갑 가능)
- **활용 맥락**: §4.1.3에서 PA F1을 주 지표로 쓰지 않는 이유를 설명할 때. "Kim et al. (AAAI 2022) showed that even a random score can achieve state-of-the-art performance under PA" 수준의 서술 근거.
- **주의**: "great possibility of overestimating" — 확실한 사실이 아닌 "가능성"이므로 우리 논문에서도 단정 표현이 아닌 "has been shown to inflate" 류로 완화.

---

### 발췌 2 — PA 기반 순위 문제 (§Abstract)

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT -->

> "the comparison of TAD methods after applying the PA protocol can lead to misguided rankings."

- **§위치**: Abstract
- **지지 Claim**: C-047, C-050 (지표 선택 정당화 — PA 회피 이유)
- **활용 맥락**: §4.1.3 또는 §4.1.4 baseline 비교 설정 설명 시 PA를 사용하지 않는 이유의 문헌 근거. 1문장 괄호 인용으로 충분.
- **주의**: "misguided rankings" — 강한 표현. 우리 논문에서 직접 인용 시 균형 잡힌 서술 필요("evaluation protocol has been criticized for...").

---

### 발췌 3 — PA%K 프로토콜 제안 (§4 "New Evaluation Protocol PA%K")

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT -->

**[A1 EXCERPT_RESOLVED — AAAI 공식 PDF 직접 다운로드 후 발췌 확보 (2026-06-11).]**

> "we propose an alternative evaluation protocol PA%K, which can mitigate the overestimation effect of F1PA and the possibility of underestimation of F1. ... The idea of PA%K is to apply PA to Sm only if the ratio of the number of correctly detected anomalies in Sm to its length exceeds the PA%K threshold K." (§4, "New Evaluation Protocol PA%K")

> "Figure 6: F1 score with PA%K with varying K. If K = 0, it is equal to the F1PA and if K = 100, it is equal to the F1." (Figure 6 caption)

- **§위치**: §4 "New Evaluation Protocol PA%K"
- **지지 Claim**: C-047 (PA%K 프로토콜 정의 — K=0이 standard PA, K=100이 standard F1; PDF에서 verbatim 확보)
- **활용 맥락**: §4.1.3에서 PA%K를 평가 지표로 채택하면서 "proposed by Kim et al. (AAAI 2022)" 명시 필수.
- **A1 확정**: K=0 ↔ F1_PA, K=100 ↔ F1 의 관계가 Figure 6 caption에서 verbatim 확인됨.

---

### 발췌 4 — untrained 모델의 비교 성능 (§Abstract)

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT -->

> "an untrained model obtains comparable detection performance to the existing methods even when PA is forbidden."

- **§위치**: Abstract
- **지지 Claim**: C-050 (PA 과대평가 보강 — 기존 방법의 실질적 성능 의문)
- **활용 맥락**: 평가 지표 섹션에서 "why PA is problematic" 논거의 하나로. 직접 인용보다는 paraphrase + 괄호 인용.
- **주의**: 이 발견은 PA 조건에서의 비교이며, PA 없는 조건에서의 비교임을 정확히 기술해야 함("even when PA is forbidden").

---

## 활용 절

| 우리 논문 위치 | 활용 방식 | 근거 발췌 |
|-------------|---------|---------|
| §4.1.3 지표 설명 | PA%K 프로토콜 제안 논문으로 인용 | 발췌 3 (+ verifier 보강) |
| §4.1.3 PA 비판 | PA 과대평가 문제 1문장 근거 | 발췌 1 |
| §4.1.3 또는 §1 | 평가 신뢰도 문제 논거 | 발췌 2 |

---

## 주의사항

1. **PA%K 정의 미확보**: abstract에 없음 — verifier가 PDF §3/§4에서 PA%K 공식 정의 발췌 필수. 우리 논문에서 PA%K를 소개할 때 자체 정의 서술 후 "(proposed by Kim et al. 2022)" 인용 구조.
2. **PA와 PA%K 구분**: PA(point adjustment, K=0)와 PA%K(K>0 일반화)의 차이를 정확히 서술. Kim et al.이 두 가지 모두 제안했는지 verifier 확인.
3. **복사 금지 표현**: "even a random anomaly score can easily turn into a state-of-the-art TAD method" — 임팩트 있는 표현이지만 원문 그대로 인용은 표절. 반드시 귀속 표기 + paraphrase.
