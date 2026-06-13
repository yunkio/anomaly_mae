---
phase: 2
agent: fixer (r2)
directives: [T2, R9, R21, R16, R19, R20]
last_modified: 2026-06-11
inputs:
  - paper/99_reviews/p2_venue_corpus_r1.md
  - paper/99_reviews/p2_dossiers_r1.md
modified_files:
  - paper/02_venue_study/VENUE_AND_PAPER_LIST.md
  - paper/02_venue_study/STRUCTURE_AND_FIGURE_PATTERNS.md
  - paper/02_venue_study/ANCHOR_SDMAE_DOSSIER.md
  - paper/02_venue_study/NRDETECTOR_DOSSIER.md
verification_method: "수정 전 전 항목 1차 소스 재확인 — arXiv HTML 원문 직접 대조(SDMAE 2306.12041v2·NRdetector 2501.11959v1: /tmp/dossier_verify/ 원본 덤프 + HTML 재파싱 grep; CATCH 2410.12261v3 신규 다운로드), arXiv abs 페이지(2410.12261, 2411.11641, 2110.02642), papers.nips.cc 2023 proceedings(MEMTO), proceedings.neurips.cc 2024(SARAD)·nips.cc virtual(TSB-AD), cvpr.thecvf.com virtual/2024/poster/30615, 정본 RESEARCH_SYNTHESIS.md §②·271_CONFIG_TRUTH.md §VI"
---

# Phase 2 Fix Log (r2) — 리뷰 r1 전수 처리

## 1. 처리 요약

| 리뷰 | BLOCKER | MAJOR | MINOR | NOTE | 처리 |
|------|---------|-------|-------|------|------|
| p2_venue_corpus_r1 | 0 | 3 (V-001, V-002, C-001) | 4 (V-003, V-004, S-001, C-002) | 5 (V-005, S-002, S-003, C-003, C-004) + C-005 | MAJOR 3/3, MINOR 3/4 수정 + 1 라우팅(C-002), NOTE 2 수정·3 조치불요, C-005 본 fixlog §4에 목록화 |
| p2_dossiers_r1 | 0 | 2 (X-M1, S-M1) | 11 (S-m1–S-m6, N-m1–N-m5) | — | **전수 수정 (13/13)** + 리뷰 참고 항목(B-3 freeze 비대칭) 1건 채택 |

**총 처리: 발견 26건 전수** (고유 25건 — C-001은 V-002와 동일 수정으로 해소): 문서 수정 적용 20건(venue 측 7: V-001·V-002/C-001·V-003·V-004·V-005·S-001·S-002 + dossier 측 13: X-M1·S-M1·S-m1–6·N-m1–5), 조치불요 NOTE 3건 확인 기재(S-003·C-003·C-004), 라우팅 2건(C-002 범위외, C-005 Phase 5 이관 — 본 문서 §4·§5) + 리뷰 참고 항목(B-3) 채택 1건.

---

## 2. 발견 ID별 처리표

### 2.1 p2_venue_corpus_r1.md

| ID | 심각도 | 1차 소스 재확인 결과 | 처리 | 대상 파일·위치 |
|----|--------|---------------------|------|---------------|
| V-001 | MAJOR | arXiv abs 2410.12261 abstract 재확인: "Extensive experiments on **10 real-world datasets and 12 synthetic datasets**" = 22 | "24개" → "22개(10 real + 12 synthetic)" 정정 + abstract verbatim 근거 병기. Paper 5 Tables 행도 메인 테이블 실측 기준으로 동기 정정 | VENUE Paper 5 |
| V-002 / C-001 | MAJOR | papers.nips.cc/paper_files/paper/2023/hash/b4c898eb… 직접 확인 — 제목·저자 4인(Junho Song, Keonwoo Kim, Jeonglyul Oh, Sungzoon Cho)·NeurIPS 36 Main Conference Track 일치 | Paper 13 venue "미확인, 2024 추정" → **NeurIPS 2023** 확정, 검증 상태 갱신, §5 venue 분포 표 갱신(NeurIPS 2023 행 신설, arXiv 미확정 3→2). SENTENCE_CORPUS 로스터("NeurIPS 2023")와의 문서 간 모순 해소 — C-001은 VENUE 측 수정으로 자동 해결(SENTENCE_CORPUS는 원래 정확, 수정 불요) | VENUE Paper 13, §5 |
| V-003 | MINOR | arXiv abs 2411.11641 abstract 재확인 — 주 기여는 INR 기반 재구성(spectral bias·temporal continuity), LLM은 "amplify the intense fluctuations in anomalies" 보조 | TSINR 선정 사유를 "INR 기반 재구성이 주된 기여 + LLM 보조 컴포넌트"로 재기술 | VENUE Paper 7 |
| V-004 | MINOR | NeurIPS 2024 proceedings 직접 조회 — ① **SARAD** (Dai, He, Yang, Leeke — Main Track, proceedings.neurips.cc 확인) ② **"The Elephant in the Room: Towards A Reliable TSAD Benchmark"** (Liu & Paparrizos — D&B Track, **VUS-PR 최신뢰 지표 권고**) 실재 확인 | "미확인" 상태 해소 — VENUE §4 미수록 후보에 2편 등재(Elephant는 평가지표 정당화 인용 후보로 우선순위 표기), STRUCTURE §I.4 주의사항 갱신. NeurIPS 2025는 미조회 상태로 명시 유지 | VENUE §4, STRUCTURE §I.4 |
| V-005 | NOTE | arXiv abs 2110.02642 abstract 재확인: "state-of-the-art results on **six** unsupervised time series anomaly detection benchmarks" | Paper 1 선정 사유에 "총 6개 벤치마크(5개 공유 + NeurIPS-TS)" 명시 | VENUE Paper 1 |
| S-001 | MINOR | arXiv HTML 2410.12261**v3** 신규 다운로드, Table 2 캡션 직접 실측: "Average A-R (AUC-ROC) and Aff-F (Affiliated-F1) accuracy measures for **10 real-world datasets and 6 synthetic datasets** of different types of anomalies", 방법 16열(15+CATCH) | "12+6=18 ds × 2 / ~576" → "**10+6=16 ds × 2 / ~512**" — **리뷰 권고안("10+12=22 ds")이 아니라 원문 테이블 실측 기준으로 정정** (리뷰의 대안 지시 "논문 최신본 기준 실제 테이블 구조 직접 확인 후 갱신" 경로 채택). abstract 총 커버리지 22와 메인 테이블 항목 수의 구분 주석 추가 | STRUCTURE §F.2 |
| S-002 | NOTE | elsarticle 기본값 관찰(리뷰 검증 수용) | "(학회와 반대인 경우 있음)" → "(elsarticle 기본값 — 표준적이며 대부분 학회와 동일)" | STRUCTURE §A.2 |
| S-003 | NOTE | — | **조치 불요** (긍정 평가 NOTE). §G.5 8소절 과세분화 위험은 문서 자체 자각 + Phase 3 압축 결정 사안으로 유지 — 정정 이력에 기재 | STRUCTURE 부록 |
| C-002 | MINOR | — | **범위 외 라우팅**: 수정 대상이 SENTENCE_CORPUS.md(쓰기 허용 4개 파일 외). orchestrator에 라우팅 — 조치안: RigorEval(arXiv 2109.05257) "AAAI 2022" 표기의 직접 소스(AAAI OJS/proceedings URL) 주석 추가 또는 Phase 4 서지 재검증 항목으로 등재 | (미수정 — 라우팅) |
| C-003 | NOTE | — | **조치 불요** — verbatim 정확도 고신뢰 확인(이상 없음) | — |
| C-004 | NOTE | — | **조치 불요** — corpus 10종 섹션 커버 확인(이상 없음) | — |
| C-005 | NOTE | — | SENTENCE_CORPUS에 **반영하지 않음** (지시 준수). 본 fixlog §4에 "Phase 5 plagiarism-guardian dispatch 고위험 목록"으로 정리 — orchestrator 라우팅 | 본 문서 §4 |

### 2.2 p2_dossiers_r1.md

| ID | 심각도 | 1차 소스 재확인 결과 | 처리 | 대상 파일·위치 |
|----|--------|---------------------|------|---------------|
| X-M1 | MAJOR (공유) | 정본 RESEARCH_SYNTHESIS.md §② 재확인 — ②-1 설정(가정): 대부분 unlabeled + 소수 labeled; ②-2 main 271 구현(FACT): "train 구간의 모든 샘플에 라벨이 존재" = label 가용성 **상한 케이스**; ②-3 라벨 희소화 sweep = R32 **계획**(전용 스크립트 미구현); ②-6/⑥ "semi/PU 명명은 Phase 3 결정 사안" | SDMAE dossier §4.2 레이블 설정 행 + §5.2 경로 3 + §6.2 행을 정본 프레이밍으로 교정. NRdetector dossier §5 전제부에 정본 3단 구조 단서 블록 신설 + D9 행 R32 계획 명시. 양쪽 모두 "차이축은 명명과 무관하게 성립" 명시 | SDMAE §4.2/§5.2/§6.2, NRdet §5/D9 |
| S-M1 | MAJOR | arXiv HTML 2306.12041v2 원문 재확인 — 5개 verbatim 전건 실재: ① "jointly reconstruct the original frames (without anomalies) and the corresponding pixel-level anomaly maps" (§1 기여) ② "we add the anomaly map as an additional channel … normal pixels to 0 and abnormal pixels to 1" (§3) ③ "forcing our model to overlook the anomalies" (§3) ④ "add the anomaly maps and the gradients together, before computing the weights" (§3) ⑤ "to surpass the 90% milestone on Avenue, it is mandatory to introduce the prediction of anomaly maps" (§4 ablation). 위치: §3 "Synthetic anomalies" 단락(오프셋 33124 이후) | **§3.6-2 신설**(3요소 + ablation 필수성, verbatim 5건), §4.1 유사점 행 신설("(합성) 이상 라벨 신호의 학습 주입", 위험도 높음), §4.2 주입-계층 차이 행 신설, **§7-2 신설**(GRL과의 개념적 평행 위험 시나리오 + 방어 3축: 주입 계층/라벨 출처/작동 지점 — R9 분석에 통합), §6.2 GRL 행 스코핑 단서, §8 overextension 행 추가, §3.3 모션 가중 서술의 증강-프레임 예외 정정(S-M1-b), §2 Verification Log 행 추가 | SDMAE §2/§3.3/§3.6-2/§4.1/§4.2/§6.2/§7-2/§8 |
| S-m1 | MINOR | 원문 grep — "known as self-distillation [101]" 전문 **유일 출현**은 §1 Introduction 기여 목록("Third, we integrate a teacher decoder and a student decoder…" 항목, 오프셋 ~8188) | 출처 "(Section 3)" → "(Section 1 Introduction, 기여 목록 — 전문 유일 출현)" 교정 + 정정 사유 명기 | SDMAE §3.5 |
| S-m2 | MINOR (R21 핵심) | 원문 reference list 직접 확인 — **[101] = Linfeng Zhang, Chenglong Bao, Kaisheng Ma, "Self-Distillation: Towards Efficient and Compact Neural Networks", IEEE TPAMI 44(8):4388–4403, 2022**. 보강 verbatim (Supplementary §6.2): "the work of Zhang et al. [101], **which introduces the form of self-distillation that inspired our work**" | "[101]" 마커 복원 + [101] 식별 기록, §5.1 **용어 계보 단락 신설**(Zhang 원류 → SDMAE AD variant → 본 연구 시계열 확장 — "SDMAE가 coining" 서술을 "동일 구조에 용어를 사용한 선례 + 선행 계보"로 정확화), §5.2 경로 1·§5.3 bullet 정밀화, §7 옵션 B 초안 "coining the term" → "applying the term … following Zhang et al." 완화 + Phase 5 'coining' 표현 금지 플래그, §8 행 보강, zhang2022selfdistillation 보조 BibTeX 추가. **#5–#9 인용 일괄 복원**: #5 "(as shown in Figure 1)" 말미, #6 "[101]", #7 "Distinct from the aforementioned studies," 두문, #8 "[6, 32]"/"[8, 12, 16, 26, 74, 86]" 클러스터 2곳, #9 "of a given sample" 말미 — 전건 원문 재대조 후 복원 | SDMAE §3.5/§5.1/§5.2/§5.3/§7/§8/§9 |
| S-m3 | MINOR | 원문: "**All decoder blocks** have four attention heads and a projection dimension of 128. The teacher decoder contains three CvT blocks, while the student decoder contains only one block." | §3.1 teacher decoder에 projection 128 명기 + "encoder 256→decoder 128 비대칭은 encoder–decoder 간" 오독 방지 주석 | SDMAE §3.1 |
| S-m4 | MINOR | Table 3 인용문은 점수 결합 전략 근거 (리뷰 판정 수용; 원문 깊이-비대칭 직접 ablation 부재 — grep 무발견과 정합) | §6.1 4행 "지지 가능" → "**간접 지지 (분기 구조 전제하의 결과)**" 강등 + 근거 중복 명시 | SDMAE §6.1 |
| S-m5 | MINOR | cvpr.thecvf.com/virtual/2024/poster/30615 재확인 — "2024 **Poster**" 표기 | §1 발표 유형 Poster 확정 + §2 Verification Log 행 추가 | SDMAE §1/§2 |
| S-m6 | MINOR | 정본 — RESEARCH_SYNTHESIS §④ 지표 표(PA%K-AUC F1 = best-epoch 선정 지표, VUS-PR/ROC, Affiliation F1, PA%K-AUC PR) 재확인 | §4.2 본 연구 지표 "AUROC" → "PA%K-AUC F1·VUS-ROC/PR·Affiliation F1 (+PA%K-AUC PR), roc_auc는 병산·비대표" — NRdetector dossier 서술과 문서 간 정합화 | SDMAE §4.2 |
| (B-3 참고) | 참고 | SDMAE 원문 "we freeze the weights of the shared backbone" + 271_CONFIG_TRUTH §VI `freeze_teacher_after_warmup=False` (INACTIVE, trainer.py:1141–1142 런타임 gate) 재확인 | 리뷰 제안 채택 — §4.2에 "2단계 학습의 teacher/backbone 동결" 차이 행 신설 (수정 의무 아님, 차이점 재료 가치로 추가) | SDMAE §4.2 |
| N-m1 | MINOR | 원문 Table 3 캡션+헤더 행 직접 실측 — 11지표 = F1, P, R, F1_PA%K, **F1_PA**, Aff-P, Aff-R, R_A_R, R_A_P, V_ROC, V_PR; 캡션 "The F1_PA is the F1 score using the PA strategy"; F1-W는 ablation Table 5 캡션에만 등장 | §3.4 11지표 구성 전면 정정(F1_PA 포함, F1-W 분리) + "PA는 main Table 2에서만 배제" 과장 방지 주의 추가 | NRdet §3.4 |
| N-m2 | MINOR | 원문 실측 — Baselines 단락("we compare NRdetector with…" ~ "…main baselines we need to compare.")이 "5.2. Experimental Setting" 헤더 **직전에 종료** → §5.1 소속 | 3건 귀속 교정: §3.3 항목 2(++변형 인용)·항목 3("main baselines")·§4.2(baseline 목록 인용) 전부 §5.2 → **§5.1**. Implementation/Evaluation metrics 인용의 §5.2 귀속은 정확(원문 재확인) — 유지 | NRdet §3.3/§4.2 |
| N-m3 | MINOR | 원문: "we compare our method with **WETAS (Lee et al., 2021) and TreeMIL (Liu et al., 2024)**, which are the main baselines we need to compare." | §3.3 항목 3 — 인용 부착을 3종 나열 직후에서 **WETAS·TreeMIL 2종**으로 이동, 원문 전체 문장 인용으로 교체 | NRdet §3.3 |
| N-m4 | MINOR | 원문: "…unlike TreeMIL **(Liu et al., 2024)** and WETAS **(Lee et al., 2021)**." | §3.1 window-size 인용의 author-year 괄호 2건 복원 | NRdet §3.1 |
| N-m5 | MINOR | 원문 freeze/frozen 전문 grep **0건** 재확인 (HTML 재파싱) | §3.5 "사전학습 후 고정 추출" 단정 → INFERENCE 표기(근거: "pre-trained" + 코드 `pretrained_model/` 구조 방증; 공식 코드 확인 가능 단서), §5 D1 행 동기 수정, §6 한계 노트에 항목 추가 | NRdet §3.5/§5 D1/§6 |

---

## 3. 핵심 확인 사항 (R21 방어 보강)

**[101]의 정체 (S-m2 핵심 산출물)**: SDMAE가 "a process known as self-distillation [101]"에서 귀속시키는 선행 연구는 **Zhang, Linfeng; Bao, Chenglong; Ma, Kaisheng. "Self-Distillation: Towards Efficient and Compact Neural Networks." IEEE TPAMI 44(8):4388–4403, 2022** 이다 (원문 bibliography bib101 직접 확인). SDMAE는 Supplementary §6.2에서 이를 "the work … which introduces the form of self-distillation that inspired our work"라고 재차 명시한다.

**R21 방어 논리에 미치는 효과**: 방어가 "SDMAE 단독 선례"(단일 논문 의존)에서 **2단 용어 계보**(Zhang et al. TPAMI 2022 원류 → SDMAE CVPR 2024의 AD 도메인 variant → 본 연구의 시계열 확장)로 강화된다. 단, "SDMAE가 용어를 coining했다"는 표현은 사실과 어긋나므로 Phase 5에서 금지 ("applying/adopting/extending the term"으로 서술). Phase 4 서지 검증 시 zhang2022selfdistillation 항목 포함 필요.

---

## 4. C-005 — Phase 5 plagiarism-guardian dispatch 시 포함할 고위험 목록 (orchestrator 라우팅)

> 지시에 따라 SENTENCE_CORPUS.md에는 반영하지 않음. Phase 5 plagiarism-guardian 에이전트 dispatch 시 아래 목록을 검사 최우선 대상으로 프롬프트에 포함할 것.

| # | 원문 인용 (corpus 위치) | 위험 사유 | 검사 지침 |
|---|------------------------|----------|----------|
| H1 | DCdetector §3: "Each channel in the multivariate time series input is considered as a single time series and divided into patches…" (SENTENCE_CORPUS §6 Method component item 5) | TSMAE patchify 서술이 채널/패치 분할을 기술할 때 표면 유사 문장이 생성될 확률 최고 | Phase 5/6 산출 본문의 patchify·channel 서술 전 문장을 이 원문과 n-gram·구조 유사도 비교, 최우선 검사 |
| H2 | SDMAE: "leverage the reconstruction discrepancy between the teacher and the student with a minimal computational overhead" (SENTENCE_CORPUS §6 item 10) | TSMAE teacher-student discrepancy 서술에서 재사용 유혹 최대 | discrepancy 관련 전 문장을 이 원문과 비교, "minimal computational overhead"류 표현 직접 재사용 금지 |
| H3 (fixer 추가) | SDMAE §1/§3: "forcing our model to overlook the anomalies" / "known as self-distillation [101]" | 본 r2에서 dossier에 신규 수록된 verbatim(§3.6-2·§3.5) — Phase 5 작성자가 dossier 경유로 접촉할 표현이 늘어남 | GRL/이상 억제·self-distillation 정의 문장에서 "overlook the anomalies", "known as self-distillation" 구문 직접 재사용 검사 |

**비고**: A2 규약(verbatim 본문 복사 금지)이 모든 corpus/dossier 인용에 적용되나, 위 항목들은 "표면적으로 유사한 문장이 자연 생성될" 구조적 유인이 있는 지점이므로 별도 우선 검사 대상이다.

---

## 5. 범위 외 / 후속 라우팅 항목 (orchestrator 인입)

1. **C-002 (MINOR, 미수정)**: SENTENCE_CORPUS.md §0.1 RigorEval "AAAI 2022"의 직접 소스 주석 부재 — 본 fixer의 쓰기 허용 범위(4개 파일) 밖. 조치안: AAAI OJS/proceedings URL 확인 후 소스 표 주석 추가, 또는 Phase 4 서지 재검증 목록에 등재.
2. **Phase 4 서지 검증 추가 항목**: zhang2022selfdistillation (SDMAE [101] — §3 참조), SARAD·TSB-AD Elephant in the Room (NeurIPS 2024 — V-004 신규 등재분).
3. **V-004 잔여**: NeurIPS 2025 TSAD는 미조회 상태 유지 (STRUCTURE §I.4에 명시) — Phase 4 확인 대상.
