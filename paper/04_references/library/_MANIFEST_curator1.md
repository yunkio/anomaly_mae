---
phase: 4
agent: excerpt-curator-1
directives: [T4]
date: 2026-06-11
---

# MANIFEST — excerpt-curator-1 배정 카드 목록

배정 기준: SCOUT_CANDIDATE_LIST §A FULL/FULL-cond 논문 목록 등장 순서에서 홀수 위치(1,3,5,7,9,11,13,15,17,19,21번째)

---

| # | Key | 카드 파일 | 발췌 수 | Abstract 확보 | EXCERPT_UNVERIFIED | 비고 |
|---|-----|----------|---------|--------------|-------------------|------|
| 1 | wang2025nrdetector | library/wang2025nrdetector.md | 6 | 전문 확보 | 아니오 | arXiv HTML 접근 성공 |
| 3 | he2022mae | library/he2022mae.md | 3 | 전문 확보 | 예 (§Method 발췌 없음) | arXiv HTML 404 — abstract만 |
| 5 | ganin2016dann | library/ganin2016dann.md | 2 | 전문 확보 | 예 (GRL 수식 발췌 없음) | JMLR PDF 바이너리 불가 |
| 7 | xu2022anomalytransformer | library/xu2022anomalytransformer.md | 3 | 전문 확보 | 예 (AR-threshold 발췌 없음) | arXiv HTML 404, OpenReview abstract만 |
| 9 | paparrizos2022vus | library/paparrizos2022vus.md | 4 | 전문 확보 | 아니오 | VLDB PDF에서 abstract 확보 |
| 11 | xu2018kpivae | library/xu2018kpivae.md | 0 | 미확보 | 예 (발췌 없음) | arXiv HTML 404, ACM DL 접근 불가 |
| 13 | sarfraz2024quovadis | library/sarfraz2024quovadis.md | 4 | 전문 확보 | 아니오 | PMLR HTML에서 abstract 확보 |
| 15 | lee2021wetas | library/lee2021wetas.md | 4 | 전문 확보 | 부분 (§방법론 section 번호 미확인) | arXiv abstract + ICCV Open Access 403 |
| 17 | bekker2020pusurvey | library/bekker2020pusurvey.md | 2 | 전문 확보 | 예 (본문 발췌 없음) | arXiv HTML 404 |
| 19 | schmidl2022evaluation | library/schmidl2022evaluation.md | 2 | 미확보 | 예 (abstract 전문 없음) | VLDB PDF 추출 불충분 |
| 21 | huang2022slavae | library/huang2022slavae.md | 0 | 미확보 | 예 (발췌 없음) | ACM DL 403, GitHub README만 |

---

## 요약 통계

- 작성 카드 수: **11**
- EXCERPT_UNVERIFIED 카드 수: **8** (he2022mae, ganin2016dann, xu2022anomalytransformer, xu2018kpivae, bekker2020pusurvey, schmidl2022evaluation, huang2022slavae, lee2021wetas 부분)
- Abstract 전문 확보: **8/11** (xu2018kpivae, schmidl2022evaluation, huang2022slavae 미확보)
- "지지 발췌 없음" 발생 claim: **아래 목록 참조**

---

## "지지 발췌 없음" 발생 claim (verifier 우선 작업 신호)

| Claim | 논문 Key | 이유 | verifier 필요 작업 |
|-------|---------|------|------------------|
| C-051 | xu2018kpivae | 본문 접근 불가 (arXiv HTML 404, ACM 403) | arXiv 1802.03903 PDF §4-5에서 PA 프로토콜 정의 발췌 |
| C-036 (GRL 수식) | ganin2016dann | JMLR PDF 바이너리 불가 | JMLR 15-239 PDF에서 GRL 정의 equation 번호 + λ schedule 수식 발췌 |
| C-053 (AR threshold) | xu2022anomalytransformer | abstract에 threshold 서술 없음, HTML 404 | OpenReview 본문 §4 또는 공식 구현 `anomaly_ratio` 파라미터 발췌 (R30 보류 유지) |
| C-011/C-025 (최초성 반증) | huang2022slavae | abstract 미확보, 표현학습 gradient 통합 여부 불명 | ACM DL 또는 사전인쇄본에서 방법론 섹션 정독·발췌 |
| C-019/C-020 (SCAR·양대계열) | bekker2020pusurvey | arXiv HTML 404 | arXiv 1811.04820 PDF §2-3에서 SCAR 정의·양대 접근법 발췌 |
| C-026/C-033 (§Method 발췌) | he2022mae | arXiv HTML 404 | arXiv 2111.06377 PDF §3에서 linear patchify 설명 발췌 |

---

## 접근 실패 URL 기록 (verifier 참고)

- arXiv HTML (404): 2111.06377, 1802.03903, 1505.07818, 2207.00705, 1811.04820, 2108.06816v2, 1906.02694
- ACM DL (403): 10.1145/3485447.3511984 (SLA-VAE)
- ICCV Open Access (403): openaccess.thecvf.com/content/ICCV2021/...
- JMLR PDF (바이너리 디코딩 불가): jmlr.org/papers/volume17/15-239/15-239.pdf
- Anomaly Transformer arXiv HTML (404): 2110.02642
