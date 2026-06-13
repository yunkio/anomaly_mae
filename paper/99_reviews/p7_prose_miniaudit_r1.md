---
phase: 7
agent: prose-miniauditor
directives: [R4, A2, R17]
last_modified: 2026-06-11
scope: "PROSE_DIFF_LOG.md 전 항목 — §1 (D-011 carry-over 7건), §3b (QA 처방 텍스트 2건), §5 (D-013 압축 라운드: §1 6건 + §2 10건 + §3 1건 + §4 10건 + TAB-1 §5.5 + layout §5.6)"
inputs:
  diff: paper/07_latex/PROSE_DIFF_LOG.md
  tex: paper/07_latex/main.tex + sections/*.tex (검사만 — 무수정)
  frozen_source: paper/05_manuscript/MANUSCRIPT_v3.md + PLACEHOLDER_REGISTRY.md (TAB-1/2/3 spec)
  ai_phrasing_basis: paper/02_venue_study/SENTENCE_CORPUS.md (부록 A collocation + 부록 B 금지/자제 목록)
  plagiarism_basis: paper/04_references/library/ 52 cards (abstract/verbatim) + 02_venue_study corpus 105문장 + dossier 2종
  method_truth_basis: paper/01_research_understanding/271_CONFIG_TRUTH.md r4 + EXPERIMENT_PROTOCOL_TRUTH.md r4
verdict: "PASS — BLOCKER 0 / MAJOR 0 / MINOR 7 (ai-phrasing 3, plagiarism 0, method-truth 4). 무손상 검증 PASS."
---

# P7 산문 변경 미니 감사 r1 — ai-phrasing / plagiarism / method-truth 3종

> 검사 대상 = PROSE_DIFF_LOG.md에 기록된 **모든** 산문 변경 문장 (D-011 7건 + QA r2 텍스트 2건 + D-013 압축 27건 + TAB-1 약어화). 각 변경에 대해 3종 검사 전수 수행, diff "before"를 frozen MANUSCRIPT_v3.md와, "after"를 현재 .tex와 대조하여 diff 정본의 정확성 자체도 검증했다.

---

## 0. diff 반영 확인 (전제 검증)

| 항목군 | 확인 결과 |
|---|---|
| D-011 7건 (R2-M1..M3, R2-m2..m5) | 7/7 .tex에 반영 확인 — sec1:22 (M1), appendix_A:85-90 (M2), main.tex:99-101 (M3 "exposing"), sec2:39 (m2 "whereas"), sec4:74-75 (m3 "largest shift: 166"), appendix_A:191 (m4 "convert … into"), sec2:27 (m5 "nor") |
| §3b QA 처방 2건 | \citet 잔존 0건 (전 파일 grep); "Ganin et al.~\cite{ganin2016dann}" 2개소 실재 (sec3:166, appendix_C:12); "Appendix~\ref" 잔존 0건; main.aux 실측 — \ref{sec:appendix_*}가 "Appendix A.x"로 렌더 (elsarticle가 단어를 방출 → 중복 제거 처방 정확) |
| D-013 27건 | 표본 전수 대조 — §1 1.1–1.6, §2 2.1–2.10, §3 3.1, §4 4.1–4.10 모두 log의 after-텍스트와 .tex 일치; before-텍스트는 v3와 일치 (예외: §6 bookkeeping, 아래 M-3) |
| TAB-1 §5.5 | sec4:30-58 — 헤더 약어·"(\%)" 캡션 이동·dagger 셀/캡션 정의 모두 log대로 |

---

## 1. ai-phrasing 검사 — **PASS** (MINOR 3)

### 1.1 금지 패턴 (corpus 부록 B.1/B.2) 전수 스캔
변경·신규 문장 및 전 .tex prose에 대해 금지 어휘(delve/showcase/underscore/pivotal/realm/landscape/seamless/meticulous/holistic/paving the way/unlock/harness/testament/boast/remarkable/imperative/paramount/groundbreaking/revolutionize/"In conclusion"/"It is important to note"/의인화) grep — **검출 0건**. 유일 히트 "additionally"(sec3:151)는 문중 부사이며 diff 외 기존 문장. "novel" 1회(sec2:30)는 NRdetector의 주장 귀속 표현으로 기존 문장·자제 범위 내. Moreover/Furthermore 연쇄 0건.

### 1.2 압축의 부자연 축약 여부
- 1.2 (4-family 포인터화), 2.1–2.2 (가족 정의 압축), 2.4 (PU 압축), 4.8 (sweep 논리 압축), 4.9 (decomposition 압축): 모두 문법적으로 완결, corpus 양성 신호(§B.3-4 "짧은 선언문 허용")와 부합. 부자연 축약 없음.
- 전승 주장·hedge 위치·수치 결합 패턴: 변경 문장 중 성능 주장 문장 없음 (해당 없음).

### 1.3 발견 (전부 MINOR)
- **A-1 (MINOR, §1 ¶1–¶2)**: D-013 1.1·1.2가 §1 도입 연속 두 문단의 첫 문장에 각각 em-dash 쌍 삽입("--- water treatment plants, … ---" / "--- reconstruction-based …, --- "), 4.6에도 1쌍 추가. corpus 부록 B.2는 em-dash 절 연결을 **자제** 판정. 원고 기존 스타일(v3 §4.1 데이터셋 문장·결론 등)에 이미 dash 쌍이 있어 패턴 신설은 아니나, 도입부 밀도가 눈에 띔. 조치 불요(차후 §1 재편집 시 1.1 또는 1.2 중 한쪽을 콤마 구조로 환원 권장).
- **A-2 (MINOR, §3.3 — diff 3.1)**: "It is a training-time mechanism; …"의 문두 대명사 — 직전 문장에 경쟁 명사구 2개("stochastic masking", "the model") 개입. 주어-위치 선행사 우선 + 문단 주제 연속 + "with no label input" 대조로 실독 위험은 낮으나 순간 오부착 가능성 있음. B-7 핵심 구("training-time mechanism; at test time … deterministic leave-one-out … no label input")는 보존 확인.
- **A-3 (MINOR, §1 — diff 1.1)**: "sensor streams **whose reliable anomaly detection** prevents safety incidents" — 행위 명사에 대한 소유 관계사 구문이 다소 압축적. 문법적으로 성립하고 의미 손실 없음; 관찰 기록.

---

## 2. plagiarism 검사 — **PASS** (발견 0)

### 2.1 검사 방법
변경·신규 문장 27건+TAB-1 캡션에서 추출한 변별 n-gram 24개("reliable anomaly detection prevents", "spans four broad families", "trained to reproduce normal", "learned and observed association", "multi-scale contrastive views", "confirmed positive examples", "non-negative risk estimators", "class-prior-based probability correction", "extract reliable negatives", "deviation networks", "active-learning labeling loop", "WETAS-derived backbone", "spatial-domain paradigm", "priority-masked", "rate at which true events are recorded", "thresholded combined score", "fails to replicate the Teacher", "capacity-limited" 등)를 02_venue_study/ (corpus 105문장 + dossier 2종) + 04_references/library/ 52 cards (abstract/verbatim 포함) 전체에 대조 grep + corpus 본문 수동 대조.

### 2.2 결과
- **6+ n-gram 일치 0건**. 유일 히트 "deviation networks" = pang2019devnet 카드의 **논문 제목(고유 방법명 DevNet)** — 방법명 지칭은 표절 아님; 해당 우리 문장("deviation networks with scarce labeled anomalies")은 카드 abstract verbatim("leverage a few … labeled anomalies and a prior probability …")과 어휘 수준에서도 분리됨.
- 도메인 표준 예시("water treatment plants", "spacecraft telemetry")는 corpus 부록 A collocation 범위 내 사용 — GDN/MEMTO 원문장과 구문 불일치 확인.
- §2.1 재구성 정의문은 corpus 기준문(DCdet §2 "Reconstruction-based methods learn a model to reconstruct normal samples…", MEMTO §2 "expects accurate reconstruction…")과 패턴만 공유, 표면 일치 없음 — **압축 과정의 회귀 없음** (압축이 오히려 원천 문장에서 더 멀어지는 방향).
- A2 역방향 검사(corpus 문장의 원고 유입): 변경 문장 중 corpus verbatim과 표면 유사도 높은 문장 없음.

---

## 3. method-truth 검사 — **PASS** (MINOR 4)

### 3.1 §1 4-family 포인터화 (diff 1.2) — 보존 확인
- 4 가족 명칭·**가족별 \cite 키 배정이 v3와 1:1 동일** (reconstruction: zong2018dagmm,su2019omnianomaly,audibert2020usad,song2023memto,wu2025catch / prediction: deng2021gdn / assoc-discrepancy+contrastive: xu2022anomalytransformer,yang2023dcdetector / backbone: tuli2022tranad,wu2023timesnet).
- 포인터 (Section~\ref{sec:related_mtsad}) = §2.1 ✓; §2.1에 per-family 정의 실재 (sec2:11-13) — log의 "정의는 §2.1 full form" 주장 사실 ✓.
- R11 앵커: "the best a label-aware variant can do is exclude confirmed anomaly windows from training, filtering contamination rather than learning from it" — **verbatim 보존** ✓ ("Consequently, these methods" → "These methods consequently" 어순 변경은 의미 동일; M-3 참조).

### 3.2 §4.4 압축 (diff 4.8) — 3-property 논리 보존 검증 ①
- 3-property 문장(First/Second/Third) 자체는 **v3와 byte-동일** (변경은 "omit the term entirely"→"omit it entirely" 1건뿐, 선행사 = 동일 문장 내 "the GRL term" — 명확).
- 압축된 covariation 문장: "fewer patches are priority-masked" ↔ property 1 메커니즘(priority masking은 labeled patch에만 적용 — 직전 문장에 명시), "fewer batches activate the GRL term" ↔ property 2, "the reconstruction term … remains elevated … bounding the degradation from below" ↔ property 3 (label-independence는 property 3 도입부에 유지 — log 주석 사실 ✓). **ARG-02 covariation + 하한 논리 모두 보존**. 인과 연결사 "so"→"and" 전환은 "As $p$ decreases" 지배 하의 공변 서술로 논리 유지 — 발견 아님 (기록만).
- wang2025nrdetector 구분 문장: "varies the rate of *incorrect* segment labels, not the rate at which true events are recorded" — 원문 의미(라벨 오류율 vs 기록율) 정확 보존 ✓. EXPERIMENT_PROTOCOL_TRUTH §⑦ (region 단위, p ∈ {1.0,…,0.1}, p→0 unsupervised 회귀) 정합 ✓.

### 3.3 TAB-1 셀 약어 "19.05\,/\,3.68$^{\dagger}$" — dual-eval 의미 검증 ②
- 수치: **PLACEHOLDER_REGISTRY TAB-1 spec·EXPERIMENT_PROTOCOL_TRUTH §①과 전 셀 byte-동일** (SWaT 719,959/224,960/45/1.63/19.05·3.68; WaDi 1,296,001·870,972/86,401·86,402/123/0.52·0.76/3.82·3.87; PSM 176,401/43,921/25/6.20/30.63; SMD 29–36/4.16avg; SMAP 355,905/217,925/25/0.70/24.54; MSL 95,271/36,775/55/1.70/16.72) — log "byte-identical" 주장 사실 ✓.
- dagger 의미: 캡션 "SWaT is evaluated under both full and excl22 conditions ($\dagger$: full\,/\,excl22); Table 2 uses excl22" — 19.05=full test AR, 3.68=excl22 평가범위 AR (truth §⑥: 단일 학습 + 평가 마스크 2조건, region 22 = anomaly 질량 83.75%) **정확 전달** ✓. dagger가 Test AR 셀에만 부착(학습은 1회) ✓. 본문 "SWaT dual evaluation" 문단(R28 핵심 서술)은 v3와 동일 — 비접촉 ✓. Appendix Table A.4의 완전형 "19.05 (full)/3.68"과 정합 ✓.

### 3.4 TFMAE 우회표현 (diff 4.1)
- "including TFMAE (Section~\ref{sec:related_mae})" — §2.3의 계보 차단 문장("similar masking-based reconstruction objectives in some time-series models \cite{fang2024tfmae} constitute independent developments, **whereas** our design follows directly from vision MAE")은 **비접촉** (R2-m2 적용형 그대로). v3의 동격구 "the time-series MAE variant discussed in Section 2.3" 제거는 새 주장 0건 — 회귀 없음. (§2.3 본문이 TFMAE를 명명하지 않고 \cite로만 지시하는 것은 v3부터의 기존 상태 — 변경에 의한 악화 아님.)

### 3.5 의무 서술 비접촉 spot 확인 ③
| 의무 | 확인 |
|---|---|
| R13 프로토콜 동기 블록 | v3 §4.1.1과 동일 (유일 변경 = 기록된 R2-m3) ✓ |
| R28 excl22 | "trained once but evaluated twice … 83.75% … the model and scores are identical, only the evaluation mask differs" v3와 동일 ✓ |
| R29 지표 상보성 + PA-F1 비판 | 앵커 5구("eliminating sensitivity to the choice of K", "identified as the most reliable single measure", "reporting all five prevents any single failure mode", "labeled (oracle) to indicate", "even a random score can reach state-of-the-art levels") v3·tex 1:1 ✓ |
| R30 threshold 방어 | "(1−α) quantile of the score distribution", "derives from evaluation-set ground truth but is never used in training" ✓ |
| R31 공정성 | "grants each unsupervised method this most favorable use of the labels", "decouples these effects" ✓ |
| R32 강건성 논리 | §3.2 검증 — 3-property byte-보존 ✓ |
| R10 컴포넌트 근거 | §4.3 OD/GRL 문단 v3와 동일 (변경 = 기록된 4.4/4.5만) ✓ |
| R21 계보 각주 | sec2:45-50 footnote — v3 [^sd-fn]과 동일 (layout 변환만) — log "byte-identical" 주장 ✓ |
| D-008 스코핑 문장 2건 | "employs, to our knowledge, the first architecture combining …" (sec1) + "the first end-to-end MTSAD model that integrates …" (sec2) — v3와 1:1 ✓ |
| §5 결론·Abstract | v3와 전문 동일 (D-013 비접촉 범위 준수; Abstract 유일 차이 = 없음, "exposing"은 v3에도 반영돼 있음 — M-3 참조) ✓ |
| \citet 수리 사실성 | "sigmoid schedule of Ganin et al." — 271_CONFIG_TRUTH r4 §VIII λ_rev "Ganin-style sigmoid ramp" 정합 ✓ |

### 3.6 발견 (전부 MINOR)
- **M-1 (MINOR, TAB-1 — log 완전성)**: 본문 TAB-1이 registry spec의 **Source 열(5개 \cite)을 미수록**. 인용 손실은 없음(5키 전부 Datasets 문단 sec4:15-17 + Table A.4 Source 열에 실재; 전체 키 집합 48=48 동일) — 그러나 이 열 제거가 PROSE_DIFF_LOG §2/§5.5 어디에도 기록되지 않음. 07_latex 미추적(git)이라 v1 변환 vs D-013 어느 라운드 소산인지 판별 불가. **조치 권장: PROSE_DIFF_LOG §5.5(또는 §2)에 1줄 추가 기록.**
- **M-2 (MINOR, TAB-1 SMD 행)**: "per-machine (\S A.3)" — 하드코딩 섹션 번호. 현 번호와 일치(렌더 정확)하나 r2 라운드가 다른 5개소에서 제거한 것과 같은 부류의 잠재 stale-ref. 차후 수정 기회에 \ref{sec:appendix_dataset} 권장.
- **M-3 (MINOR, log bookkeeping)**: ① PROSE_DIFF_LOG §1의 R2-M1/M2/M3 "v3 source" 인용이 **현재 frozen MANUSCRIPT_v3.md와 불일치** — v3에는 세 MAJOR 수정형이 이미 반영돼 있음(m2/m3/m4/m5는 구형 유지). 최종 .tex와는 수렴하므로 본문 무영향; 기록 정밀성 문제만. ② §5.1 entry 1.2의 "(R11 anchor sentence preserved verbatim)" — 앵커 후반절은 verbatim이나 전반절은 어순 변경("Consequently, these methods"→"These methods consequently"); before/after 전문이 병기돼 있어 은폐는 아님.
- **M-4 (MINOR, diff 4.10)**: "the Student's capacity-limited, adversarially suppressed **representation fails to replicate the Teacher**" — 엄밀히는 출력 불일치(d_i = ‖o^T−o^S‖², §3.6)이므로 "the Teacher's output"이 정확. 원문도 environ 수준의 환유였고 §3.6 정의로 복원 가능 — 의미 훼손 아님. 차후 재편집 시 "the Teacher's output" 복원 권장.

---

## 4. 무손상 검사 (\cite·PH) — **PASS**

| 항목 | 보고치 (log §5.7) | 재측정 | 판정 |
|---|---|---|---|
| 고유 \cite 키 | 48 unchanged | tex 48 = v3 48, **집합 diff 0건** | ✓ |
| % PH: 마커 분포 | sec4 22, sec1 2, sec5 3, main 4, appendices 5 | 동일 (sec4 22, sec1 2, sec5 3, main 4, A 2+B 3+C 0) | ✓ |
| NUM 고유 ID | 31/31 | 31/31 (occurrence 32 — NUM-014가 TAB-2 캡션 실체화로 의도적 2회: sec4:224 캡션 + :311 본문; v3는 캡션이 registry에 있어 1회. 손실 아닌 보강) | ✓ |
| TXT occurrence | 4/4 | 4/4 (TXT-001, TXT-002) | ✓ |
| FIG/TAB/ALG 마커 | TikZ box 5 / tabular 골격 11 / algorithm2e 1로 실체화 | FIG-1..4+B1 placeholder box 5, TAB 1+2+3+A계 11, algorithm* 1 (appendix_C:115) | ✓ |
| \citet / "(author?)" / Appendix 중복 | 0 / 0 / 0 | grep 0 / 0 / 0 | ✓ |

---

## 5. 판정

| 검사 | 판정 | 발견 |
|---|---|---|
| ai-phrasing | **PASS** | MINOR 3 (A-1 em-dash 밀도, A-2 문두 "It" 선행사, A-3 관계사 구문) — corpus 금지 패턴 도입 0 |
| plagiarism | **PASS** | 0 — 6+ n-gram 일치 없음, 압축 회귀 없음 |
| method-truth | **PASS** | MINOR 4 (M-1 TAB-1 Source 열 미기록, M-2 \S A.3 하드코딩, M-3 log bookkeeping, M-4 환유) — 의미 훼손 0, 의무 서술 비접촉 전수 확인 |
| 무손상 (\cite·PH) | **PASS** | 보고 통계 전 항목 재현 |

**종합: PASS — BLOCKER 0, MAJOR 0, MINOR 7.** Phase 7의 모든 산문 변경은 기술적 의미·의무 서술·인용/마커 무결성을 보존하며 최종본 진입 가능. MINOR 7건은 게이트 비차단 — M-1(로그 1줄 추가)·M-2(\ref 환원)는 차후 .tex/log 수정 기회에 처리 권장.
