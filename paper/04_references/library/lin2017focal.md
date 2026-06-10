---
phase: 4
agent: excerpt-curator-2
directives: [T4]
last_modified: 2026-06-11
key: lin2017focal
verification_status: VERIFIED_A
verified_by_A: 2026-06-11
corrected_fields: [doi_added, pages_added, pt_formula_excerpt_resolved]
card_grade: FULL
excerpt_access: abstract_only
---

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT — verbatim excerpts below are for internal reference only and must not appear in any submitted text -->

> **A2 경고**: 아래 verbatim 발췌는 card 내부 전용입니다. 어떠한 형태로도 원고에 그대로 복사하지 마십시오.

---

## 서지 정보

- **Key**: lin2017focal
- **제목**: Focal Loss for Dense Object Detection
- **저자**: Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár
- **Venue**: ICCV 2017, pp.2999–3007
- **DOI**: 10.1109/ICCV.2017.324 [A1 확인 — DBLP 일치]
- **arXiv**: 1708.02002 (직접 확인 2026-06-11)
- **확인 출처**: arXiv abs + DBLP conf/iccv/LinGGHD17
- [A1 추가] pages 2999–3007, DOI 확정

---

## Abstract (verbatim)

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT -->

"The highest accuracy object detectors to date are based on a two-stage approach popularized by R-CNN, where a classifier is applied to a sparse set of candidate object locations. In contrast, one-stage detectors that are applied over a regular, dense sampling of possible object locations have the potential to be faster and simpler, but have trailed the accuracy of two-stage detectors thus far. In this paper, we investigate why this is the case. We discover that the extreme foreground-background class imbalance encountered during training of dense detectors is the central cause. We propose to address this class imbalance by reshaping the standard cross entropy loss such that it down-weights the loss assigned to well-classified examples. Our novel Focal Loss focuses training on a sparse set of hard examples and prevents the vast number of easy negatives from overwhelming the detector during training. To evaluate the effectiveness of our loss, we design and train a simple dense detector we call RetinaNet. Our results show that when trained with the focal loss, RetinaNet is able to match the speed of previous one-stage detectors while surpassing the accuracy of all existing state-of-the-art two-stage detectors."

---

## 핵심 발췌

### 발췌 1 — Focal loss 핵심 아이디어 (§Abstract)

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT -->

> "We propose to address this class imbalance by reshaping the standard cross entropy loss such that it down-weights the loss assigned to well-classified examples."

- **§위치**: Abstract
- **지지 Claim**: C-037 (본 논문의 focal-style BCE variant가 표준 focal loss와 다름을 명시 — 원류 인용 필수)
- **활용 맥락**: §3.5(C)에서 우리의 라벨 손실 함수가 focal loss에서 영감을 받았음을 밝히되, p_t 정의가 다름을 1문장으로 설명. 원류 인용으로만 사용.

**[A1 EXCERPT_RESOLVED — arXiv PDF §3에서 직접 발췌 확보 (2026-06-11):]**

**p_t 정의 (§3.1, Eq.2):**
> "we define pt: pt = p if y=1; pt = 1−p otherwise" (§3.1, Eq.2)
> "CE(p, y) = CE(pt) = − log(pt)" (§3.1, Eq.1 rewritten)

**FL 수식 (§3.2, Eq.4):**
> "FL(pt) = −(1 − pt)^γ log(pt)." (§3.2, Eq.4 — "Focal Loss Definition" 절)

커버 claim: C-037 (p_t 정의 및 FL 수식, §3.1 Eq.2 + §3.2 Eq.4에서 verbatim 확보)

---

### 발췌 2 — 클래스 불균형 문제 (§Abstract)

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT -->

> "the extreme foreground-background class imbalance encountered during training of dense detectors is the central cause"

- **§위치**: Abstract
- **지지 Claim**: C-037 (인용 맥락: 이상탐지에서 normal/anomaly 불균형이 focal loss를 적용하게 된 동기)
- **활용 맥락**: §3.5 방법론에서 클래스 불균형 문제를 언급할 때 focal loss의 원래 문제 설정(전경/배경 불균형)과 우리 설정(정상/이상 불균형)이 유사한 동기를 갖는다고 설명.
- **주의**: Lin et al.의 설정은 object detection — 우리 도메인(TSAD)에 직접 적용되는 것이 아님을 명확히 할 것. "원류에서 영감을 받아 우리가 재설계"라는 서술 구조 필수.

---

### 발췌 3 — 학습 어려운 예제 집중 (§Abstract)

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT -->

> "Our novel Focal Loss focuses training on a sparse set of hard examples and prevents the vast number of easy negatives from overwhelming the detector during training."

- **§위치**: Abstract
- **지지 Claim**: C-037
- **활용 맥락**: 우리 방법이 focal-style 가중치를 쓰는 이유를 설명할 때 "hard example mining" 아이디어의 출처로 인용.
- **주의**: "easy negatives"는 object detection 맥락 — 우리 논문에서 "easy normal samples"로 변환하여 적용하는 것이지, 원문 그대로 인용하는 것은 도메인 불일치.

---

## 활용 절

| 우리 논문 위치 | 활용 방식 | 근거 발췌 |
|-------------|---------|---------|
| §3.5 (C) 본문 | focal-style BCE 설계 시 Lin et al. 원류 인용 (1문장 + 괄호) | 발췌 1 |
| §3.5 (C) 1문장 | p_t 정의가 Lin et al.과 다름을 명시 (차별화 1문장) | C-037 지시사항 |
| §3.5 (C) 동기 | 클래스 불균형 → focal-style 가중치 선택 동기 서술 | 발췌 2 |

---

## 주의사항

1. **p_t 공식 미확보**: abstract에는 수식이 없음. verifier가 PDF §3에서 "FL(p_t) = -(1-p_t)^γ log(p_t)" 및 p_t 정의를 발췌 필수.
2. **차별화 필수**: C-037에 명시된 대로, 우리 논문의 focal-style BCE는 Lin et al.의 표준 focal loss와 p_t 정의가 다름을 반드시 1문장으로 명시해야 함. 차별화 없이 "we use focal loss"라고 쓰면 부정확.
3. **RetinaNet**: 원문의 RetinaNet은 우리 논문과 무관 — 인용 시 모델명 언급 불필요.
4. **복사 금지 표현**: "down-weights the loss assigned to well-classified examples" — 이 표현을 우리 논문에 그대로 쓰면 표절. 동일 아이디어를 독자적으로 표현해야 함.
