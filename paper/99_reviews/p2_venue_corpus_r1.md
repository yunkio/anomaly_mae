---
phase: 2
agent: adversarial-reviewer-A
directives: [T2]
last_modified: 2026-06-11
---

# Phase 2 Adversarial Review — venue·구조·corpus
**Reviewer**: adversarial-reviewer-A (venue·구조·corpus)
**Target**: Elsevier 저널, 본문 9페이지
**Review date**: 2026-06-11

---

## 종합 판정

| 문서 | 판정 | 비고 |
|-----|------|------|
| VENUE_AND_PAPER_LIST.md | CONDITIONAL PASS | MAJOR 2건 — 수정 후 통과 가능 |
| STRUCTURE_AND_FIGURE_PATTERNS.md | PASS | MINOR 1건 |
| SENTENCE_CORPUS.md | CONDITIONAL PASS | MAJOR 1건 (MEMTO venue 불일치) |

**Phase 2 전체**: CONDITIONAL PASS — 아래 MAJOR 3건 수정 후 Phase 3 진입 가능. BLOCKER 없음.

---

## Spot-check 기록 (14건)

| #  | 논문 | 확인 항목 | 결과 | 소스 |
|----|------|----------|------|------|
| 1  | Anomaly Transformer | ICLR 2022 Spotlight | **확인** — Accept (Spotlight) | api.openreview.net/notes?forum=LzQQ89U1qm_, ICLR 2022 virtual/spotlight/7024 |
| 2  | Anomaly Transformer | OpenReview ID LzQQ89U1qm_ | **확인** — 제목·venue 일치 | api.openreview.net (Paper803) |
| 3  | DCdetector | KDD 2023 + DOI 10.1145/3580305.3599295 | **확인** | arxiv.org/abs/2306.10347 comment: "Accepted by ACM SIGKDD … KDD 2023" |
| 4  | SDMAE | CVPR 2024 | **확인** | arxiv.org/abs/2306.12041 comment: "Accepted at CVPR 2024" |
| 5  | NRdetector | KDD 2025 + DOI 10.1145/3690624.3709257 | **확인** | arxiv.org/abs/2501.11959 comment: "Accepted by 2025 ACM SIGKDD … KDD'25" |
| 6  | CATCH | ICLR 2025 | **확인** | arxiv.org/abs/2410.12261 comment: "Accepted by ICLR 2025" |
| 7  | Sub-Adjacent Transformer | IJCAI 2024 | **확인** | arxiv.org/abs/2404.18948 comment: "IJCAI 2024" |
| 8  | TSINR | KDD 2025 (SIGKDD) | **확인** | arxiv.org/abs/2411.11641 comment: "Accepted by SIGKDD 2025" |
| 9  | ModernTCN | ICLR 2024 **Spotlight** | **확인** — Accept (spotlight) | api2.openreview.net/notes?forum=vpJMJerXHU (Note: ICLR virtual 검색 결과 "poster" 표시는 spotlight 논문이 포스터 세션 목록에도 등재되기 때문) |
| 10 | MAE (He et al.) | CVPR 2022 | **확인** — pp. 16000-16009 | openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders… |
| 11 | DACR | ICASSP 2024 | **확인** | arxiv.org/abs/2401.11271 comment: "5 pages, 3 figures, accepted at ICASSP 2024" |
| 12 | DTAAD | Knowledge-Based Systems Vol.295, 2024, pp.111849 | **확인** — DOI 10.1016/j.knosys.2024.111849 | arxiv.org/abs/2302.10753 journal_ref |
| 13 | MEMTO | NeurIPS 2023 | **확인 (SENTENCE_CORPUS와 불일치 — 아래 참조)** | papers.nips.cc/paper_files/paper/2023 직접 등재 확인. 단, VENUE_LIST Paper 13은 "venue 미확인"으로 표기 |
| 14 | CATCH 데이터셋 수 | "24개 데이터셋 SOTA" 주장 | **오류** — arXiv abstract: "10 real-world datasets and 12 synthetic datasets" = **22개** | arxiv.org/abs/2410.12261 |

---

## 문서별 발견사항

---

### VENUE_AND_PAPER_LIST.md

#### [V-001] MAJOR — CATCH 데이터셋 수 오류

**위치**: Paper 5 CATCH 선정 사유 "24개 데이터셋 SOTA"

**문제**: arXiv 공식 abstract는 "10 real-world datasets and 12 synthetic datasets" = 22개를 명시한다. 24개 주장은 근거 없음.

**근거**: arxiv.org/abs/2410.12261 — "Extensive experiments on 10 real-world datasets and 12 synthetic datasets"

**권장 수정**: "22개 데이터셋(10 real-world + 12 synthetic) SOTA"로 정정.

**해결 상태**: OPEN

---

#### [V-002] MAJOR — Paper 13 (MEMTO) venue 표기 불일치

**위치**: Paper 13 검증 상태 "venue 미확인, 2024 추정"

**문제**: MEMTO는 NeurIPS 2023 논문으로 공식 proceedings에 게재되어 있다. VENUE_AND_PAPER_LIST가 "venue 미확인"으로 표기하는 반면, SENTENCE_CORPUS는 로스터 표에서 venue를 "NeurIPS 2023"으로 이미 기재하고 있다. 문서 간 모순이다. 연도 추정("2024 추정")도 틀렸다.

**근거**: papers.nips.cc/paper_files/paper/2023 — "MEMTO: Memory-guided Transformer for Multivariate Time Series Anomaly Detection, Junho Song, Keonwoo Kim, Jeonglyul Oh, Sungzoon Cho" 직접 등재. (arXiv 페이지에는 venue comment 없으나 NeurIPS proceedings에서 확인됨.)

**권장 수정**: Paper 13 venue를 "NeurIPS 2023"으로 정정, 연도 추정 삭제, 검증 상태를 "NeurIPS 2023 proceedings 확인"으로 갱신. SENTENCE_CORPUS와 일치시킬 것.

**해결 상태**: OPEN

---

#### [V-003] MINOR — DCdetector 저자 수 비표기 및 TSINR 설명 부정확

**위치**: Paper 2 DCdetector (저자 5인 정상), Paper 7 TSINR "LLM 기반 이상 증폭"

**문제**: TSINR 선정 사유에 "LLM 기반 이상 증폭"이라 표기되어 있다. arXiv abstract 확인 결과 "leverage a pre-trained large language model to amplify the intense fluctuations in anomalies"라는 기술이 실제 있다. 오류는 아니나, 이는 모델의 주된 기여가 아니라 보조 컴포넌트로, 선정 사유 기술이 핵심 기여(INR 기반 재구성)를 누락하고 부차적 특징만 강조하고 있다. Phase 3에서 인용 포지셔닝을 왜곡할 수 있다.

**권장 수정**: "INR(implicit neural representation) 기반 재구성 + LLM 보조 이상 증폭"으로 기술.

**해결 상태**: OPEN (MINOR)

---

#### [V-004] MINOR — NeurIPS 2024/2025 TSAD 논문 부재

**위치**: §I.4 주의사항 — "NeurIPS 2024–2025 직접 accepted TSAD 논문은 명확히 확인되지 않음"

**문제**: 이 사실 자체는 정직한 기재다. 그러나 T2 지시문은 "최근 3년(2024–2026)" 탑티어 커버를 요구한다. NeurIPS 2024 proceedings에 TSAD 논문이 존재하는지 여부를 확인하지 않은 채 "미확인"으로 남겨 두는 것은 T2 충족을 위협하는 누락이다. (실제로 NeurIPS 2024에는 시계열 이상탐지 관련 논문이 복수 등재되었을 가능성이 높다.)

**권장 수정**: Phase 3 전에 NeurIPS 2024 proceedings를 직접 조회하거나, 누락 이유를 명시하는 면책 주석 추가.

**해결 상태**: OPEN (MINOR)

---

#### [V-005] NOTE — Anomaly Transformer "6 benchmarks" vs 본문 표기

**위치**: Paper 1 "동일 벤치마크(SWaT/SMD/PSM/SMAP/MSL)" 선정 사유

**관찰**: arXiv abstract는 "six unsupervised time series anomaly detection benchmarks"를 명시한다. 선정 사유는 5개 데이터셋만 나열하고 있다. 원문에서 6번째 데이터셋(NeurIPS-TS 합성 포함 시)이 NIPS-TS인데, 이를 생략하고 "동일 벤치마크"라고 기재하면 Phase 3에서 TSMAE 비교 설정을 오해할 수 있다.

**권장 수정**: "(SWaT/SMD/PSM/SMAP/MSL + NIPS-TS synthetic, 총 6개 벤치마크)" 명시.

**해결 상태**: NOTE

---

### STRUCTURE_AND_FIGURE_PATTERNS.md

#### [S-001] MINOR — CATCH 데이터셋 수 전파 오류

**위치**: §F.2 Main Results Table, CATCH 행 "열(datasets×metrics): 12+6=18 ds × 2"

**문제**: CATCH의 실험은 10 real-world + 12 synthetic = 22 datasets이나, 이 표에서 "12+6=18"로 기재되어 있다. VENUE 문서의 "24개" 오류와 다른 수치로, 이중으로 틀려 있다.

**근거**: arxiv.org/abs/2410.12261 abstract.

**권장 수정**: "10+12=22 ds × 2"로 정정. 혹은 논문 최신본 기준 실제 테이블 구조를 직접 확인 후 갱신.

**해결 상태**: OPEN (MINOR)

---

#### [S-002] NOTE — Elsevier figure caption 방향 주장

**위치**: §A.2 "Figure caption은 figure 아래, table caption은 table 위 (학회와 반대인 경우 있음)"

**관찰**: elsarticle 템플릿 기본값은 figure caption 아래, table caption 위이며, 이는 대부분의 학회 논문과 동일하다. "학회와 반대"라는 표현은 일부 독자에게 혼란을 줄 수 있다. 실제로 반대인 저널이 있다면 해당 저널을 명시해야 한다. 현재 표현은 검증 없는 일반화다.

**권장 수정**: "figure caption은 figure 아래, table caption은 table 위 (elsarticle 기본값 — 표준적이며 대부분 학회와 동일)"로 수정.

**해결 상태**: NOTE

---

#### [S-003] NOTE — 실행 가능성 평가

**판정**: Phase 3에서 바로 쓸 수 있는 수준 충족.

§G.1–G.6는 9페이지 Elsevier 저널 기준 섹션별 분량 수치, 단락 골격, figure 유형별 배치 위치, TSMAE 직결 권장 소절 구조까지 구체적으로 기술하고 있다. 표면적 요약이 아니라 실행 가능한 패턴이다. §F.4, §F.5 Phase 5 적용 지침은 TSMAE 아키텍처 컴포넌트 이름을 명시하며, Phase 3 narrative-architect가 바로 인용할 수 있는 형태다.

유일한 주의: §G.5 Method 소절이 8개(§3.1–§3.8)로 9페이지 타깃에서 과도하게 세분화될 가능성이 있음. 문서 자체가 "저널 분량 제약상 §3.3–§3.6을 2개로 병합 가능"이라 언급하고 있어 자각된 위험이지만, Phase 3에서 실제 압축 방안을 결정해야 한다.

**해결 상태**: NOTE

---

### SENTENCE_CORPUS.md

#### [C-001] MAJOR — MEMTO venue 불일치 (문서 간 모순)

**위치**: §0.1 논문 로스터, MEMTO 행 — venue "NeurIPS 2023"

**문제**: SENTENCE_CORPUS는 MEMTO를 "NeurIPS 2023"으로 표기하고 있으나, 동일 Phase의 VENUE_AND_PAPER_LIST Paper 13은 "venue 미확인, 2024 추정"으로 기재한다. 같은 Phase 2 산출물 간에 동일 논문의 venue가 상충한다. Phase 6에서 SENTENCE_CORPUS를 기준으로 인용하거나 VENUE를 기준으로 인용하면 한쪽은 반드시 오류가 된다. (NeurIPS 2023이 실제 정확하며, VENUE_AND_PAPER_LIST가 틀림 — Spot-check #13 참조.)

**권장 수정**: VENUE_AND_PAPER_LIST Paper 13을 NeurIPS 2023으로 정정(V-002와 동일 수정). 이후 SENTENCE_CORPUS는 정확하다.

**해결 상태**: OPEN (VENUE 문서 수정으로 해소 가능)

---

#### [C-002] MINOR — RigorEval venue 미확인 상태

**위치**: §0.1 논문 로스터, RigorEval — "AAAI 2022"

**문제**: arXiv 2109.05257 페이지에는 venue comment가 없다. AAAI 2022라는 표기의 직접 소스가 문서에 기재되어 있지 않다. 다만 내용 및 날짜 맥락상 AAAI 2022가 타당하며, AAAI 2022 proceedings 직접 검색으로 확인 가능하다. 사소하지만 "fetch 완료"라는 문서의 신뢰도 주장과 불일치한다.

**권장 수정**: 소스 표에 "AAAI 2022 OJS 또는 proceedings 직접 확인 필요" 주석 추가, 또는 AAAI OJS URL 기재.

**해결 상태**: OPEN (MINOR)

---

#### [C-003] NOTE — verbatim 정확도 (고신뢰 항목)

**확인 결과**: SENTENCE_CORPUS §9 ablation item 2의 인용문 "our proposed Anomaly Transformer surpasses the pure Transformer by 18.34% (76.62→94.96) absolute improvement"를 ar5iv 원문에서 직접 재확인했다. 수치(18.34%, 76.62, 94.96)와 문장 구조가 원문과 완전히 일치한다.

SENTENCE_CORPUS §1 Abstract AnomTr 인용 3문장 ("Our key observation is that…", "Technically, we propose…", "The Anomaly Transformer achieves state-of-the-art results on six unsupervised…")은 arXiv abs 페이지 citation_abstract 필드와 문자 단위 일치 확인.

A2 경고문은 §0 헤더와 §0.1 오프닝에 이중으로 배치되어 있음 — 적정.

**해결 상태**: NOTE (이상 없음)

---

#### [C-004] NOTE — corpus 섹션 유형 10종 커버 확인

§1 Abstract, §2 Intro 도입, §3 Intro 기여, §4 Related Work 포지셔닝, §5 Method notation, §6 Method component, §7 Experiments setup, §8 결과 해석, §9 Ablation, §10 Conclusion — **10종 완전 커버** 확인.

각 섹션당 6–10문장 + 보조 표본. Phase 6 기준 corpus로 충분한 밀도다.

**해결 상태**: NOTE (이상 없음)

---

#### [C-005] NOTE — 잠재적 plagiarism 위험 패턴

**관찰**: §6 Method component item 5의 인용문은 "Each channel in the multivariate time series input is considered as a single time series and divided into patches…" (DCdetector §3 원문)이다. TSMAE의 patchify 구현이 채널 독립 패칭이라면, Phase 5에서 이 문장과 표면적으로 매우 유사한 기술이 생성될 가능성이 높다. SENTENCE_CORPUS A2 경고가 적용되나, Phase 5·6 검사자가 이 항목을 특별히 주목해야 한다.

마찬가지로 §6 item 10의 SDMAE "leverage the reconstruction discrepancy between the teacher and the student with a minimal computational overhead"는 TSMAE teacher-student 서술에서 재사용 유혹이 크다.

**권장 조치**: Phase 5에서 위 두 인용문 원문 표현과의 유사도를 검사 최우선 대상으로 지정.

**해결 상태**: NOTE

---

## T2 충족도 평가

| 기준 | 평가 | 근거 |
|-----|------|------|
| 최근 3년(2024–2026) 학회 커버 | 충족 | ICLR 2025, KDD 2025(×2), CVPR 2024, IJCAI 2024, KDD 2023이 포함됨 |
| 2024–2026 비중 | 충족 — 14편 중 8편이 2023–2025 | Anomaly Transformer(2022), MAE(2022)는 구조 분석·기반 논문으로 선정 사유 명시 |
| 시계열 이상탐지 논문 포함 여부 | **충족** — 14편 중 시계열 이상탐지 직접 대상 11편(Anomaly Transformer, DCdetector, NRdetector, CATCH, Sub-Adjacent Transformer, TSINR, PatchAD, DACR, DTAAD, MEMTO, DDMT) | T2 핵심 요건 충족 |
| 분석 깊이 — 바로 쓸 수 있는 패턴 | 충족 | STRUCTURE 문서 §G 전체가 TSMAE 직결 권장안 제공 |
| 문장 corpus 확보 | 충족 | 10종 섹션 × 6–10문장 + 부록 A/B |
| Elsevier 저널 관례 정확성 | 대체로 충족 (S-002 MINOR 1건) | Highlights, Declaration, booktabs 등 정확히 기술 |

---

## 수정 필요 체크리스트

### Phase 3 진입 전 필수 (MAJOR)

- [ ] **[V-001]** VENUE_AND_PAPER_LIST Paper 5 CATCH 데이터셋 수 "24개" → "22개(10 real-world + 12 synthetic)" 정정
- [ ] **[V-002 / C-001]** VENUE_AND_PAPER_LIST Paper 13 MEMTO venue "미확인, 2024 추정" → "NeurIPS 2023"으로 정정 (SENTENCE_CORPUS와 일치시키기)

### 추적 가능 (MINOR)

- [ ] **[S-001]** STRUCTURE §F.2 CATCH 행 "12+6=18 ds" → "10+12=22 ds" 정정
- [ ] **[V-003]** VENUE TSINR 선정 사유 보완 (INR 주된 기여 명시)
- [ ] **[C-002]** SENTENCE_CORPUS RigorEval venue 확인 소스 추가
- [ ] **[V-004]** NeurIPS 2024 TSAD 논문 조회 후 결과 기재

### Phase 5·6 이관

- [ ] **[C-005]** DCdetector channel-patching 문장 및 SDMAE teacher-student discrepancy 문장을 Phase 5 plagiarism 검사 최우선 대상으로 지정
