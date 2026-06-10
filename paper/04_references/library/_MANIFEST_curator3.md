---
phase: 4
agent: excerpt-curator-3
directives: [T4, R19]
last_modified: 2026-06-11
scope: SCOUT_CANDIDATE_LIST §B LIGHT 25편 + §C LIGHT-optional 2편 = 27 cards
---

# _MANIFEST_curator3 — LIGHT 카드 전담 (excerpt-curator-3)

작성일 2026-06-11. 전 카드 `card_grade: LIGHT`, `verification_status: PENDING_VERIFICATION`.
abstract verbatim은 검증/표절 대조 전용 (A2) — 본문 복사 금지.

## 작성 카드 27편

| # | key | fetch 소스 (2026-06-11) | abstract 상태 |
|---|-----|------------------------|--------------|
| 1 | tuli2022tranad | arxiv.org/abs/2201.07284 | verbatim ✓ |
| 2 | audibert2020usad | S2 API 미러 (ACM 403) | verbatim(미러) ⚠ |
| 3 | zong2018dagmm | api.openreview.net (BJJLHbb0-) | verbatim ✓ |
| 4 | deng2021gdn | ojs.aaai.org/16523 (공식) | verbatim ✓ |
| 5 | su2019omnianomaly | S2 API 미러 (ACM 403) | verbatim(미러) ⚠ — "signicantly" 합자 아티팩트 의심 |
| 6 | wu2023timesnet | api.openreview.net (ju_Uqw384Oq) | verbatim ✓ |
| 7 | fang2024tfmae | S2 API 미러 (IEEE 418) | verbatim(미러) ⚠ |
| 8 | lai2023npsr | api2.openreview.net (ljgM3vNqfQ) | verbatim ✓ |
| 9 | song2023memto | arxiv.org/abs/2312.02530 | verbatim ✓ |
| 10 | luo2024moderntcn | api2.openreview.net (vpJMJerXHU) | verbatim ✓ |
| 11 | wu2025catch | api2.openreview.net (m08aK3xxdJ) | verbatim ✓ (arXiv abs 렌더 실패 → API) |
| 12 | yang2023dcdetector | arxiv.org/abs/2306.10347 | verbatim ✓ |
| 13 | goh2016swat | link.springer.com (공식; idp 리다이렉트 경유) | verbatim ✓ |
| 14 | ahmed2017wadi | Crossref+OpenAlex+S2 (ACM 403) | **failed** — OpenAlex 재구성본만 (비-verbatim 참고용) |
| 15 | abdulaal2021psm | S2 API 미러 (ACM 403) | verbatim(미러) ⚠ |
| 16 | hundman2018telemanom | arxiv.org/abs/1802.04431 | verbatim ✓ |
| 17 | jacob2021exathlon | arxiv.org/abs/2010.05073 | verbatim ✓ |
| 18 | duplessis2014pu | papers.nips.cc/paper/5509 (공식) | verbatim ✓ |
| 19 | kiryo2017nnpu | arxiv.org/abs/1703.00593 | verbatim ✓ |
| 20 | elkan2008pu | 저자 camera-ready PDF (cseweb.ucsd.edu) + Crossref 서지 | verbatim ✓ (저자 PDF 1면 전사; ACM 403) |
| 21 | pang2019devnet | arxiv.org/abs/1911.08623 | verbatim ✓ |
| 22 | bergmann2020uninformed | arxiv.org/abs/1911.02357 | verbatim ✓ |
| 23 | deng2022reverse | arxiv.org/abs/2201.10703 | verbatim ✓ |
| 24 | xiong2020prenorm | proceedings.mlr.press/v119/xiong20b (공식) | verbatim ✓ |
| 25 | blazquez2021review | arxiv.org/abs/2002.04236 | verbatim ✓ (arXiv v1 기준 — CSUR 본 대조 필요) |
| 26 | xu2023rosas (§C optional) | S2 API 미러 (arXiv abs 렌더 실패×2 + export 429) | verbatim(미러) ⚠ |
| 27 | wang2022hscl (§C optional) | arxiv.org/abs/2207.11789 | verbatim ✓ |

## 접근 실패 / 강등 내역

1. **ahmed2017wadi — abstract_access: failed.** 시도 경로: dl.acm.org(403) → S2(publisher 보류) → Crossref(abstract 필드 부재) → CySWATER 워크숍 사이트 FORTH PDF(발표 슬라이드본, abstract 없음) → dokumen.pub(점검 중). 카드에는 OpenAlex inverted-index 재구성본을 **비-verbatim 참고용**으로만 수록. 서지(저자 전원·pp.25–28)는 Crossref 공식 메타데이터로 확보.
2. **dl.acm.org 전면 403** (usad/omnianomaly/psm/elkan/wadi) — ACM 수록 5편 중 4편은 S2 미러 또는 저자 PDF로 abstract 확보, WADI만 실패.
3. **ieeexplore 418 차단** (tfmae) — S2 미러로 확보.
4. **arxiv.org abs 페이지 간헐적 본문 렌더 실패** (catch, rosas) — CATCH는 OpenReview API로, RoSAS는 S2 미러로 대체.

## verifier 인계 메모

- ⚠ 미러(S2) 기반 5편(usad, omnianomaly, psm, tfmae, rosas)은 공식 페이지와 철자 단위 diff 필요 (frontmatter `abstract_source`에 명기).
- 저자 표기 보강 필요: usad "F. Guyard"(축약), rosas "Ninghui Liu"(미러 표기), moderntcn "Luo donghao / wang xue"(OpenReview 프로필 소문자) — 모두 카드 서지에 주의 명기.
- 신규 식별자 확보: RoSAS Elsevier DOI 10.1016/j.ipm.2023.103459 (scout 목록 [verifier-TODO] 해소 후보); WADI 쪽수 25–28 + 풀 서지(Crossref); SWaT LNCS 10242, pp.88–99, ISBN; Elkan & Noto pp.213–220.
- ICLR/NeurIPS 계열(dagmm, timesnet, npsr, moderntcn, catch)은 OpenReview 공식 API 직접 확인 — DOI 없음이 정상.
