---
phase: 4
agent: excerpt-curator-1
directives: [T4]
last_modified: 2026-06-11
key: bekker2020pusurvey
verification_status: VERIFIED_A
verified_by_A: 2026-06-11
corrected_fields: [title_case, excerpt_resolved_SCAR, excerpt_resolved_two_approaches]
card_grade: FULL
---
# Learning from Positive and Unlabeled Data: A Survey

**경고: 이 card의 verbatim 발췌·abstract는 검증/문체 대조 전용 — 논문 본문으로 복사·근접 의역 절대 금지 (A2)**

## 서지 (scout 예비 — 검증 전)
저자: Jessa Bekker, Jesse Davis
Venue: Machine Learning (Springer) — 정식 제목: "Learning from positive and unlabeled data: a survey" (소문자 — Crossref/DBLP 표기 기준)
연도: 2020
권호: vol.109, no.4, pp.719–760
DOI: 10.1007/s10994-020-05877-5
arXiv: 1811.04820 (v3 2020-05-18)
공식 URL: https://arxiv.org/abs/1811.04820
[A1 정정] 제목 대소문자: card 표기 "Learning from Positive and Unlabeled Data: A Survey" → 공식(Crossref/DBLP) 표기 "Learning from positive and unlabeled data: a survey".

## Abstract 전문 (verbatim)
"Learning from positive and unlabeled data or PU learning is the setting where a learner only has access to positive examples and unlabeled data. The assumption is that the unlabeled data can contain both positive and negative examples. This setting has attracted increasing interest within the machine learning literature as this type of data naturally arises in applications such as medical diagnosis and knowledge base completion. This article provides a survey of the current state of the art in PU learning. It proposes seven key research questions that commonly arise in this field and provides a broad overview of how the field has tried to address them."

## 핵심 발췌 (verbatim, 섹션/위치 표기)

> "Learning from positive and unlabeled data or PU learning is the setting where a learner only has access to positive examples and unlabeled data." (Abstract)

커버 claim: C-019
활용 맥락: §2.2 첫 단락에서 PU Learning의 정의를 제시할 때 인용. "positive(확인된 이상) + unlabeled(미지)" 설정의 공식 정의 출처.

---

> "The assumption is that the unlabeled data can contain both positive and negative examples." (Abstract)

커버 claim: C-019, C-020
활용 맥락: unlabeled 데이터가 정상·이상을 모두 포함할 수 있다는 핵심 가정 — 우리 설정의 "contaminated semi-supervised" 맥락과 직결.

---

**[A1 EXCERPT_RESOLVED: arXiv PDF에서 직접 발췌 확보 (2026-06-11). 이하는 공식 PDF verbatim.]**

**SCAR 가정 정의 (§3.1.1):**
> "The Selected Completely At Random (SCAR) assumption lies at the basis of most PU learning methods ... It assumes that the set of labeled examples is a uniform subset of the set of positive examples." (§3.1.1, p.8 of PDF)

> "Definition 1 (Selected Completely At Random (SCAR)) Labeled examples are selected completely at random, independent from their attributes, from the positive distribution. The propensity score e(x), which is the probability for selecting a positive example is constant and equal to the label frequency c: e(x) = Pr(s=1|x,y=1) = Pr(s=1|y=1) = c." (§3.1.1)

커버 claim: C-019 (SCAR 정의의 verbatim 발췌, §3.1.1에서 확보)

---

**양대 접근법 분류 (§5, p.15 of PDF):**
> "Most methods can be divided into the following three categories: Two-step techniques, biased learning and class prior incorporation. The two-step technique consists of two steps: 1) identifying reliable negative examples, and 2) learning based on the labeled positives and reliable negatives. Biased learning considers PU data as fully labeled data with class label noise for the negative class." (§5 Introduction)

커버 claim: C-020 (양대 계열 — Two-step technique = 신뢰음성추출형; biased learning = 비용민감형; §5에서 확보)

## 우리 논문에서의 활용

커버 claim: C-019, C-020

- **§2.2 Related Work (PU Learning 단락)**: C-019 — PU learning의 공식 정의 및 설정을 소개할 때 survey 인용. "labeled anomaly = positive, unlabeled = possibly mixed" 구도 정의.
- **§2.2 계보 정리**: C-020 — PU learning 양대 계열(비용민감형·신뢰음성추출형)을 1–2문장으로 정리하고 du Plessis/Kiryo/Elkan&Noto와 함께 클러스터 인용.

## 주의사항
- Survey 논문이므로 개별 주장보다 "PU learning 전반의 개관" 용도로 인용. 특정 방법론 주장의 근거로 단독 인용하지 말 것.
- SCAR 가정이 우리 설정에 완전히 적용되는지 검토 필요: 우리의 labeled anomaly가 "completely at random"으로 선택된 것이 아닐 수 있음 (특히 SMD처럼 실제 장애 이벤트 기반). 인용 시 "PU learning 설정과 유사하지만 SCAR 가정을 완전히 충족하지 않을 수 있음" 언급 권고.
- arXiv HTML 404로 본문 발췌 미확보 — SCAR 정의·양대 계열 설명의 정확한 section 번호는 verifier 필수 확인.
