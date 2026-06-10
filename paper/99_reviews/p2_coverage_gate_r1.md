---
phase: 2
agent: coverage-auditor
directives: [M10]
last_modified: 2026-06-11
audit_targets:
  - paper/02_venue_study/VENUE_AND_PAPER_LIST.md (r2)
  - paper/02_venue_study/STRUCTURE_AND_FIGURE_PATTERNS.md (r2)
  - paper/02_venue_study/SENTENCE_CORPUS.md
  - paper/02_venue_study/ANCHOR_SDMAE_DOSSIER.md (r2)
  - paper/02_venue_study/NRDETECTOR_DOSSIER.md (r2)
inputs_reviewed:
  - paper/99_reviews/p2_venue_corpus_r1.md
  - paper/99_reviews/p2_dossiers_r1.md
  - paper/99_reviews/p2_fixlog_r2.md
verification_method: "1차 소스 직접 재확인 — arXiv API(2410.12261 abstract), arXiv HTML v3 신규 다운로드(2410.12261 Table 2 캡션), papers.nips.cc 2023 hash 페이지(MEMTO), 리뷰어 원본 덤프 /tmp/dossier_verify/{sdmae,nrdetector}.{html,txt}(타이틀·크기 진본성 확인 후 바이트 단위 grep; SDMAE 2306.12041v2·NRdetector 2501.11959v1), 정본 RESEARCH_SYNTHESIS.md §② 직접 대조"
---

# Phase 2 Coverage Gate Audit (r1) — coverage-auditor

게이트 기준(마스터 §7 Phase 2): "분석의 깊이(표면적 요약이 아닌, Phase 3에서 바로 쓸 수 있는 실행 가능한 패턴인지), 시계열 이상탐지 논문 포함 여부, 문장 corpus 확보 여부, 두 dossier의 완성도." 매핑 Directive(§9.4 기준 Phase 2분): T2, R9(준비), R16(준비), R19(근거 수집), R20(준비), R21(방어논리 확인). §7 "적용 Directive" 목록과 §9.4 매핑은 본 6종에 대해 불일치 없음 (ERRATA 불요).

## 종합 판정: **PASS**

- Directive 6/6 충족 근거 확인 (아래 §1 — COVERAGE_MATRIX 전사용 근거 문자열 포함).
- 게이트 기준 4/4 충족 (§2).
- 수정 spot 재검증 7/7 PASS — MAJOR 4건 + R21 인용 정밀화 + NRdetector 11지표 + (보너스) S-001 캡션 실측 (§3).
- r1 발견 26건(고유 25건) 전수 마감 — 수정 20 / 조치불요 3 / 라우팅 2(이행 확인 1 포함) + 참고 채택 1 (§4).
- 산출물 5종 frontmatter 적합 (§5).

---

## 1. Directive별 충족 근거 판정표 (COVERAGE_MATRIX 전사용)

| ID | Phase 2 요구분 | 판정 | 충족 근거 (파일 §섹션) — 제안 근거 문자열 |
|----|----------------|------|------------------------------------------|
| **T2** | 탑티어 학회 리스트업 + 고평가 논문 논리 흐름·구성 + plot/figure/table 파악 (TSAD 필수 포함) | **충족** | `02_venue_study/VENUE_AND_PAPER_LIST.md` §1(2024–2026 탑티어 학회 리스트, ML/DM/특화 3분류)·§3(고평가 14편 — TSAD 직접 11편, 편별 섹션 구조/figures/tables)·§5(venue 분포); `STRUCTURE_AND_FIGURE_PATTERNS.md` §A–§E(섹션 구조·intro 논증 4단·related work 조직·method/실험 서술 패턴)·§F.1–F.10(figure/table 유형 10종 — 위치·크기·캡션)·§G(TSMAE 직결 지침); `SENTENCE_CORPUS.md` §1–§10(문장 표본) |
| **R9** | (준비) SDMAE 유사/차이 지점 내부 정리 — 포지셔닝 전략 수립용 | **충족** | `ANCHOR_SDMAE_DOSSIER.md` §4.1(구조적 유사점 12행, 리뷰어 위험도)·§4.2(실질 차이점 17행)·§7(포지셔닝 옵션 A/B/C + 권장 C + 예시 초안)·§7-2(anomaly-map 평행 위험 시나리오 + 방어 3축: 주입 계층/라벨 출처/작동 지점)·§8(overextension 위험 6행) |
| **R16** | (준비) NRdetector 실험 구성·논리 상세 분석 | **충족** | `NRDETECTOR_DOSSIER.md` §1(2-stage PU 프레임워크 요약)·§2(PU/semi 정당화 4단 논리 — verbatim 근거)·§3.1–3.5(데이터셋/split, 라벨 40%/e₁ sweep 구조, baseline 3계층+부록 깔때기, 11지표+PA 양보-반박 처리, 구현·ablation 5종) — 차용 포인트("main 1점+sweep 별표", 깔때기 비교, 극단점 환원 해석)까지 운영화 |
| **R19** | (근거 수집) baseline 인용 처리 방식 근거 — NRdetector 방식 | **충족** | `NRDETECTOR_DOSSIER.md` §4(related work 본문 내 baseline 모델명 0건 — 전수 grep 검증, r1 리뷰어 독립 재현 일치; AutoFormer/FEDformer는 실험 섹션이 유일 인용처라는 직접 선례; §4.3 우리 논문 적용 규칙 3조) |
| **R20** | (준비) 공통점보다 차이점 위주 강조 재료 추출 | **충족** | `NRDETECTOR_DOSSIER.md` §5(차이축 D1–D9 표 — 각 행에 "차이가 뒷받침하는 주장" 연결; "거의 없음" 주장의 정밀 스코핑 — 원문 인용 #24 기반 검증-공격 방어; 공통점 3개 최소 인정 + D1/D3/D5 중심축 배치 전략) |
| **R21** | (방어논리 확인) SDMAE가 왜 그 구조를 'self-distilled'라 부르는지 확인 | **충족** | `ANCHOR_SDMAE_DOSSIER.md` §3.5(명명 근거 원문 — "a process known as self-distillation [101]", §1 Introduction 기여 목록 유일 출현; §2 Related Work의 variant 선언)·§5.1(용어 계보: Zhang et al. TPAMI 2022 원류 → SDMAE AD variant → 본 연구 시계열 확장)·§5.3(리뷰어 이의 대응 — 2단 출판 선례)·§9('coining' 표현 Phase 5 금지 플래그 + zhang2022selfdistillation 보조 BibTeX) |

---

## 2. 게이트 기준 4항목 판정

| # | 기준 | 판정 | 근거 |
|---|------|------|------|
| 1 | 분석의 깊이 — Phase 3에서 바로 쓸 수 있는 실행 가능한 패턴 | **PASS** | STRUCTURE §G.1(섹션별 분량 수치)·§G.2(TSMAE intro 단락별 내용)·§G.4–G.6(related work/method/experiments 권장 소절 — TSMAE 컴포넌트 실명), §F.1/F.4/F.7 "Phase 5 적용 지침"; SDMAE dossier §7 초안 문장 3종 + 권장안; NRdet dossier §3.2/§3.3 차용 포인트·§4.3 운영 규칙. r1 리뷰어 S-003 양성 판정("표면적 요약이 아니라 실행 가능한 패턴")과 정합. 단 §G.5 method 8소절 과세분화는 Phase 3 압축 결정 사안으로 명시됨(자각된 위험, 비차단) |
| 2 | 시계열 이상탐지 논문 포함 | **PASS** | VENUE §5: 14편 중 TSAD 직접 대상 11편 (Anomaly Transformer/DCdetector/NRdetector/CATCH/Sub-Adjacent/TSINR/ModernTCN(이상탐지 포함 5태스크)/PatchAD/DACR/DTAAD/MEMTO/DDMT 계열); 2024–2026 venue 커버(ICLR 2025, KDD 2025×2, CVPR 2024, IJCAI 2024, NeurIPS 2024 후보 2편 등재) |
| 3 | 문장 corpus 확보 | **PASS** | SENTENCE_CORPUS: 섹션 유형 10종 × 6–10문장(총 ~105문장, 전건 출처+관용 패턴 해설) + 부록 A(collocation)·부록 B(AI-티 금지 패턴 시드 — corpus 빈도 관찰 기반) + Phase 6 사용 지침·신뢰도 노트. r1 C-003(verbatim 정확도)·C-004(10종 커버) 양성 확인 |
| 4 | 두 dossier 완성도 | **PASS** | r1 리뷰어 B의 verbatim 전수 검증(SDMAE 10/10 + NRdet 26/26 원문 실재, 할루시네이션 0) + 기술 주장 spot 43건+ 일치 + NRdet 완결성 판정; r2에서 MAJOR 2건(S-M1/X-M1) 및 MINOR 11건 전수 해소 — 본 감사의 1차 소스 spot 재검증(§3)에서 핵심 수정 전건 재확인 |

---

## 3. 수정 spot 재검증 (1차 소스 직접 재확인 — 7건 전건 PASS)

검증 자료 진본성: `/tmp/dossier_verify/sdmae.html`(590,917B)·`nrdetector.html`(876,036B)은 `<title>` 태그가 각 논문 제목과 일치하는 arXiv HTML 원본 덤프임을 확인 후 바이트 단위 grep에 사용. CATCH는 arXiv API + v3 HTML을 본 감사에서 신규 다운로드.

| # | 항목 (심각도) | 재검증 방법 | 결과 |
|---|---------------|-------------|------|
| 1 | **V-001** CATCH "24개"→"22개(10 real+12 synthetic)" (MAJOR) | arXiv API `export.arxiv.org/api/query?id_list=2410.12261` abstract 직접 추출 | **PASS** — "Extensive experiments on 10 real-world datasets and 12 synthetic datasets demonstrate that CATCH achieves state-of-the-art performance" 원문 확인. r2 정정값 정확 |
| 2 | **V-002** MEMTO venue "미확인, 2024 추정"→NeurIPS 2023 (MAJOR) | papers.nips.cc/paper_files/paper/2023/hash/b4c898eb… 직접 fetch | **PASS** — 제목·저자 4인(Junho Song, Keonwoo Kim, Jeonglyul Oh, Sungzoon Cho)·"NeurIPS 2023 Main Conference Track" 일치. SENTENCE_CORPUS 로스터와의 모순(C-001) 해소 확인 |
| 3 | **X-M1** 라벨 설정 프레이밍 교정 (MAJOR, 양 dossier 공유) | 정본 `01_research_understanding/RESEARCH_SYNTHESIS.md` §② 직접 대조 | **PASS** — 정본 3단 구조(②-1 설정 가정 / ②-2 main 구현 = "label 가용성 상한(upper-bound) 케이스" / ②-3 희소화 sweep 계획 R32) + ②-6("엄밀한 PU setting이 아니다… 'contaminated semi-supervised'", 명명은 Phase 3 확정)이 SDMAE dossier §4.2/§5.2/§6.2 및 NRdet dossier §5 전제부·D9의 r2 교정 문구와 정확히 정합 |
| 4 | **S-M1** SDMAE anomaly-map 분기 신설 §3.6-2 (MAJOR) | sdmae 원문 덤프 grep — verbatim 5건 | **PASS** — ①"jointly reconstruct the original frames (without anomalies) and the corresponding pixel-level anomaly maps" ②"we add the anomaly map as an additional channel … normal pixels to 0 and abnormal pixels to 1"(MathML 중복 렌더링 "00/1111"은 추출 잔재 — 내용 일치) ③"forcing our model to overlook the anomalies" ④"add the anomaly maps and the gradients together, before computing the weights" ⑤"to surpass the 90% milestone on Avenue, it is mandatory to introduce the prediction of anomaly maps"(§4 Table 2 ablation 논의 단락 실측 — "self-distillation gives the highest boost… However, to surpass…") 전건 원문 실재 |
| 5 | **R21 인용 정밀화** (S-m1/S-m2) | sdmae 원문 덤프 grep + bib101 HTML 직접 추출 | **PASS** — (a) "known as self-distillation [101]" **전문 유일 출현**, 위치는 §1 Introduction의 3대 변경(기여) 단락("Third, we integrate a teacher decoder and a student decoder…" 항목 내) — dossier 출처 교정("Section 1 Introduction, 기여 목록") 정확; (b) bib.bib101 = "Linfeng Zhang, Chenglong Bao, and Kaisheng Ma. Self-Distillation: Towards Efficient and Compact Neural Networks. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(8):4388–4403, 2022" — 권·호·페이지까지 dossier 기재와 한 글자 일치; (c) Supplementary 보강 인용 "the work of Zhang et al. [101], which introduces the form of self-distillation that inspired our work." 원문 실재. 'coining' 금지 플래그의 사실 근거 성립 |
| 6 | **NRdetector 11지표 교정** (N-m1) | nrdetector 원문 덤프 — Table 3 캡션 + 헤더 셀 HTML 파싱 | **PASS** — 헤더 실측: Methods \| **F1, P, R, F1_PA%K, F1_PA, Aff-P, Aff-R, R_A_R, R_A_P, V_ROC, V_PR** = 정확히 11지표, F1_PA 포함; 캡션 "The F1_PA is the F1 score using the PA strategy" 원문 일치. "PA는 main Table 2에서만 배제" 주의 문구의 사실 근거 성립 |
| 7 | (보너스) **S-001** STRUCTURE §F.2 CATCH 행 — 리뷰 권고안(10+12=22)이 아닌 원문 실측(10+6=16) 채택 | arXiv HTML 2410.12261**v3** 본 감사 신규 다운로드, Table 2 캡션 grep | **PASS** — "Average A-R (AUC-ROC) and Aff-F (Affiliated-F1) accuracy measures for 10 real-world datasets and 6 synthetic datasets of different types of anomalies." 캡션 verbatim 확인. fixer의 리뷰-권고 이탈은 리뷰의 대안 지시("논문 최신본 기준 실제 테이블 구조 직접 확인 후 갱신") 경로로 정당 — abstract 총 커버리지(22)와 메인 테이블 항목 수(16)의 구분 주석도 적절 |

추가 관찰 (비차단): 원문 §1의 R21 인용 직후 문장 "During the self-distillation process, the shared encoder is frozen."이 실재 — dossier §4.2의 B-3 채택 행(SDMAE 동결 vs 271 `freeze_teacher_after_warmup=False` 비동결)의 원문 근거와 정합. N-m2(Baselines 단락이 "5.2. Experimental Setting" 헤더 직전 종료 → §5.1 귀속)와 N-m3/N-m4("we compare our method with WETAS (Lee et al., 2021) and TreeMIL (Liu et al., 2024), which are the main baselines…" — 2종 지칭, author-year 괄호 복원)도 원문 라인 실측으로 재확인 완료.

---

## 4. r1 발견 전수 마감 대조 (1:1)

### 4.1 p2_venue_corpus_r1.md (13건)

| ID | 심각도 | fixlog 처리 | 마감 확인 |
|----|--------|------------|----------|
| V-001 | MAJOR | 수정 (VENUE Paper 5 + Tables 행) | **CLOSED** — 본 감사 spot #1 재검증 PASS |
| V-002 | MAJOR | 수정 (Paper 13 + §5 분포 표) | **CLOSED** — spot #2 PASS |
| V-003 | MINOR | 수정 (TSINR 선정 사유 — INR 주기여) | **CLOSED** — VENUE Paper 7 반영 확인 |
| V-004 | MINOR | 수정 (NeurIPS 2024 조회 — SARAD·TSB-AD §4 등재 + STRUCTURE §I.4 갱신; 2025 미조회 명시) | **CLOSED** — 양 문서 반영 확인, 잔여는 Phase 4 이관 명시 |
| V-005 | NOTE | 수정 (6 벤치마크 명시) | **CLOSED** — Paper 1 반영 확인 |
| S-001 | MINOR | 수정 (원문 실측 기준 — 리뷰 권고와 다른 값) | **CLOSED** — spot #7 PASS (이탈 정당) |
| S-002 | NOTE | 수정 (elsarticle 기본값 표현) | **CLOSED** — §A.2 반영 확인 |
| S-003 | NOTE | 조치불요 (양성 평가) | **CLOSED** — STRUCTURE 부록에 기재 확인 |
| C-001 | MAJOR | V-002와 동일 수정으로 해소 (SENTENCE_CORPUS는 원래 정확) | **CLOSED** — 문서 간 모순 해소 확인 |
| C-002 | MINOR | 라우팅 (fixer 쓰기 범위 외 → orchestrator) | **CLOSED (라우팅 이행 확인)** — SENTENCE_CORPUS §0.1 RigorEval 행에 직접 소스 주석("ojs.aaai.org/index.php/AAAI/article/view/20680 — P1 프로토콜 리뷰에서 검증, C-002 주석 2026-06-11; Phase 4 재검증 대상") 실재 — 라우팅된 조치안이 실제 이행됨 |
| C-003 | NOTE | 조치불요 (verbatim 정확 확인) | **CLOSED** |
| C-004 | NOTE | 조치불요 (10종 커버 확인) | **CLOSED** |
| C-005 | NOTE | SENTENCE_CORPUS 미반영(지시 준수) + fixlog §4에 Phase 5 plagiarism-guardian 고위험 목록(H1–H3) 정리·라우팅 | **CLOSED (라우팅)** — orchestrator가 Phase 5 dispatch 시 fixlog §4 포함 의무 — 잔여 추적 항목 ① |

### 4.2 p2_dossiers_r1.md (13건 + 참고 1건)

| ID | 심각도 | fixlog 처리 | 마감 확인 |
|----|--------|------------|----------|
| X-M1 | MAJOR (공유) | 양 dossier 교정 (SDMAE §4.2/§5.2/§6.2, NRdet §5 전제·D9) | **CLOSED** — spot #3 정본 대조 PASS |
| S-M1 | MAJOR | §3.6-2 신설 + §4.1/§4.2 행 신설 + §6.2 단서 + §7-2 신설 + §8 행 + §3.3 예외 정정 | **CLOSED** — spot #4 verbatim 5건 PASS, 요구 조치 5요소 전부 반영 확인 |
| S-m1 | MINOR | 출처 "(Section 3)"→§1 Intro 기여 목록 | **CLOSED** — spot #5(a) PASS |
| S-m2 | MINOR | [101] 복원 + Zhang TPAMI 식별 + §5.1 계보 + 'coining' 완화 + #5–#9 마커 일괄 복원 + 보조 BibTeX | **CLOSED** — spot #5(b)(c) PASS |
| S-m3 | MINOR | teacher decoder proj 128 명기 | **CLOSED** — 원문 "All decoder blocks…" grep 1건 실재 |
| S-m4 | MINOR | §6.1 간접 지지 강등 | **CLOSED** — 문서 반영 확인 |
| S-m5 | MINOR | Poster 확정 (§1/§2) | **CLOSED** — 문서 반영 확인 (CVPR virtual 페이지 — r1·fixlog 이중 확인 기록) |
| S-m6 | MINOR | §4.2 지표 "AUROC"→5지표 체계 | **CLOSED** — NRdet dossier와 정합 확인 |
| N-m1 | MINOR | 11지표 F1_PA 포함·F1-W 분리 + 과장 방지 주의 | **CLOSED** — spot #6 PASS |
| N-m2 | MINOR | 인용 3건 §5.2→§5.1 귀속 교정 | **CLOSED** — 원문 라인 실측(720/721행 순서) 재확인 |
| N-m3 | MINOR | "main baselines" WETAS·TreeMIL 2종 부착 | **CLOSED** — 원문 문장 재확인 |
| N-m4 | MINOR | author-year 괄호 2건 복원 | **CLOSED** — 원문 재확인 |
| N-m5 | MINOR | "고정 추출" INFERENCE 강등 (§3.5/D1/§6) | **CLOSED** — 문서 3개소 반영 확인 |
| (B-3) | 참고 | 채택 — §4.2 동결 차이 행 신설 | **CLOSED** — 원문 "the shared encoder is frozen" 근거 실재 (spot 추가 관찰) |

**대조 결과**: r1 두 리뷰의 발견 26건(고유 25건 — C-001=V-002) 전부가 fixlog에서 수정(20)/조치불요(3)/라우팅(2) 중 하나로 처리되었고, 누락 0건. fixlog §1 집계와 본 대조 일치.

---

## 5. Frontmatter 적합성 (5개 산출물)

§4 규칙(생성 Phase / 작성 agent / 충족 Directive ID / 최종 수정일) 기준:

| 파일 | phase | agent | directives | last_modified | 판정 |
|------|-------|-------|-----------|---------------|------|
| VENUE_AND_PAPER_LIST.md | 2 | venue-scout | [T2] | 2026-06-11 (+revision r2) | **적합** |
| STRUCTURE_AND_FIGURE_PATTERNS.md | 2 | venue-scout | [T2] | 2026-06-11 (+revision r2) | **적합** |
| SENTENCE_CORPUS.md | 2 | corpus-collector | [T2] + related [T6(준비), R4(준비)] | 2026-06-11 | **적합** — related_directives로 Phase 6 연계를 명시한 것은 가산 요소 |
| ANCHOR_SDMAE_DOSSIER.md | 2 | anchor-paper-analyst (SDMAE) | [R9, R21] | 2026-06-11 (+revision r2) | **적합** |
| NRDETECTOR_DOSSIER.md | 2 | nrdetector-analyst | [R16, R19, R20] | 2026-06-11 (+revision r2) | **적합** — agent명이 로스터의 anchor-paper-analyst 세분 페르소나(허용: §5.2 "필요하면 세분화") |

5종 합산 directives = {T2, R9, R21, R16, R19, R20} — Phase 2 매핑 6종과 정확히 1:1 대응, 과·누락 없음.

---

## 6. 잔여 추적 항목 (게이트 비차단 — orchestrator 인입)

1. **C-005 / fixlog §4**: Phase 5 plagiarism-guardian dispatch 프롬프트에 고위험 목록 H1–H3 포함 (orchestrator 의무).
2. **Phase 4 서지 검증 추가 항목**: zhang2022selfdistillation([101]), SARAD, TSB-AD "Elephant in the Room"(VUS-PR 권고 — 인용 우선순위 높음), RigorEval AAAI 2022 재검증, PatchAD/DDMT venue 확정, NeurIPS 2025 TSAD 미조회분, DCdetector KDD 최종본 대조, NRdetector ACM판 대조.
3. **Phase 5 표현 금지 플래그**: "coining the term self-distillation" 계열 사용 금지 (SDMAE dossier §9 — section-drafter 프롬프트에 전달).
4. **Phase 3 결정 사안 재확인**: 설정 명명(semi/PU/contaminated semi-supervised — RESEARCH_SYNTHESIS §②-6), STRUCTURE §G.5 method 소절 압축.
5. COVERAGE_MATRIX.md 갱신: 본 보고서 §1의 근거 문자열로 T2(Phase 2분 DONE), R9/R16/R19/R20/R21(Phase 2 부분 충족 기록 — 후속 Phase 3·5 잔여) 전사 (orchestrator 수행 — 본 감사의 쓰기 범위는 본 파일 1개).

---

## 7. 게이트 판정

**PASS** — BLOCKER 0, MAJOR 0 (r1 MAJOR 4건 전건 수정·1차 소스 재검증 통과), 매핑 Directive 6/6 근거 확보, 게이트 기준 4/4 충족, r1 발견 전수 마감, frontmatter 5/5 적합. Phase 3 진입 가능.
