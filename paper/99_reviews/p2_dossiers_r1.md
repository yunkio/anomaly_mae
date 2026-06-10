---
phase: 2
agent: adversarial-reviewer-B
directives: [R9, R21, R16, R19, R20]
last_modified: 2026-06-11
review_targets:
  - paper/02_venue_study/ANCHOR_SDMAE_DOSSIER.md
  - paper/02_venue_study/NRDETECTOR_DOSSIER.md
verification_method: "arXiv HTML 원문 직접 다운로드(2306.12041v2, 2501.11959v1) 후 바이트 단위 grep 대조 + WebFetch(arXiv abs ×2, CVPR poster page, GitHub repo ×2) + 정본 대조(271_CONFIG_TRUTH.md, EXPERIMENT_PROTOCOL_TRUTH.md, RESEARCH_SYNTHESIS.md)"
---

# Phase 2 Adversarial Review (B) — Anchor Dossier 2종 (r1)

## 판정 요약

| 문서 | 판정 | BLOCKER | MAJOR | MINOR |
|------|------|---------|-------|-------|
| ANCHOR_SDMAE_DOSSIER.md | **CONDITIONAL PASS** (MAJOR 2건 수정 후 통과) | 0 | 2 (S-M1, X-M1) | 6 |
| NRDETECTOR_DOSSIER.md | **CONDITIONAL PASS** (MAJOR 1건 수정 후 통과) | 0 | 1 (X-M1 공유) | 5 |

**할루시네이션 verbatim: 0건.** 두 dossier의 따옴표 인용 전수(SDMAE 10건, NRdetector 26건)를 arXiv HTML 원문에서 바이트 단위로 재확인 — **모두 원문에 실재**한다. 기술 주장 spot-check는 SDMAE 18건+, NRdetector 25건+ 수행, 조작·허위 주장 발견 없음. 정본(271_CONFIG_TRUTH) 직접 모순 없음(GRL=student hidden 억제 ✓, decoder 3L/2L ✓, anomaly-first masking 15% ✓, score=recon+scaled disc(4:1) ✓). 단, 두 dossier가 공유하는 **본 연구 라벨 설정 전제**가 RESEARCH_SYNTHESIS의 보류 판정과 충돌(X-M1) — Phase 3 인입 전 수정 필요.

NRdetector dossier(세션 중단 의심 문서)의 **완결성: 이상 없음** — §0–§6 전 섹션 실재, 마지막 문장 완결, frontmatter의 "전 인용문 원문 대조 완료" 주장이 본 검증에서 실제로 성립함을 확인.

---

## A. 공유 MAJOR

### X-M1 (MAJOR, 두 문서 공통) — 본 연구 라벨 설정 전제가 Phase 1 정본군의 보류 판정과 충돌
- SDMAE dossier §4.2 표: "레이블 설정 | 반지도/PU 학습 (**소수의 실제 labeled anomaly 활용**)", §5.2 "PU-setting anomaly suppression".
- NRdetector dossier §5 전제: "**소수 labeled anomaly + 대량 unlabeled**".
- 정본 대조: `RESEARCH_SYNTHESIS.md:41` — "구현상 **train 구간의 모든 샘플에 라벨이 존재**하며 … 편입된 test 앞 50% 안의 실제 anomaly에도 전부 라벨이 제공된다"; `:75` — "main 실험 구현은 train 구간 라벨이 전부 존재하는 **상한 케이스**이므로 엄밀한 PU setting이 아니다 … 더 정확하게는 'contaminated semi-supervised'"; `:77` — 설정 명명은 **Phase 3 결정 사안**. 라벨 희소화 sweep은 R32상 "진행할 예정"(`EXPERIMENT_PROTOCOL_TRUTH.md:213-216` — 전용 파라미터·스크립트 grep 0건).
- 271_CONFIG_TRUTH와의 **직접 모순은 아님**(config는 라벨 비율 키 자체가 없음 — BLOCKER 아님). 그러나 두 dossier 모두 미결 사안을 확정 전제로 기재했고, 특히 NRdetector dossier의 R20 차이축 D1–D3("라벨을 표현 학습에 통합")은 라벨 명명과 무관하게 성립하지만, "소수 labeled + 대량 unlabeled"라는 **설정 서술 자체는 main 271 구현(전구간 라벨, 상한 케이스)과 불일치**한다.
- **요구 수정**: 두 dossier의 본 연구 측 설정 서술에 "main 구현은 전구간 라벨 상한 케이스, PU/semi 명명은 Phase 3 결정 사안(RESEARCH_SYNTHESIS §⑤), 희소화 sweep은 계획 단계(R32)" 단서 1줄씩 추가.

---

## B. ANCHOR_SDMAE_DOSSIER.md

### B-1. Verbatim 전수 검증 기록 (10/10 원문 실재)

원문: arXiv HTML 2306.12041v2 직접 다운로드 + 전문 텍스트화 후 대조. 본문 섹션 경계 실측: §1@3254, §2@10742, §3@18411, §4@39216, §5@51195 (정규화 텍스트 오프셋).

| # | 인용 (dossier 위치) | 판정 | 비고 |
|---|---------------------|------|------|
| 1 | "Although our reconstruction loss focuses on tokens with high motion…" (§3.2) | **EXACT** | §3 ✓ |
| 2 | "In the second phase, we freeze the weights of the shared backbone…" (§3.4) | **EXACT** | §3 ✓ |
| 3 | "The main difference is that instead of reconstructing the patches from the real image…" (§3.4) | **EXACT** | §3 ✓ |
| 4 | "To reduce our processing time, we use a shared encoder … known as self-distillation." (§3.5, R21 핵심) | **실재하나 2건 결함** | (a) 원문은 "…self-distillation **[101]**." — 인용 마커 무표기 탈락. (b) **출처 오기**: dossier는 "(Section 3)"이라 했으나 실제 위치는 **§1 Introduction(기여 목록)** @8188 (유일 출현). → S-m1/S-m2 |
| 5 | "A student decoder branches out from the teacher after the first transformer block…" (§3.5) | 실재, 절단 | 원문은 "…extra transformer block **(as shown in Figure 1)**." — ellipsis 없는 말미 절단. §3 ✓ |
| 6 | "Self-distillation attaches multiple classification heads at various depths… two decoders of different depths." (§3.5) | 실재 | 원문 "Self-distillation **[101]** attaches…" — 마커 탈락. §2 ✓ |
| 7 | "To our knowledge, we are the first to introduce a variant of self-distillation in anomaly detection." (§3.5) | 실재 | 원문 두문 "Distinct from the aforementioned studies, to our knowledge…" 절단+대문자화(브래킷 무표기). §2 ✓ |
| 8 | "Knowledge distillation was originally designed to compress…" (§3.5) | 실재 | 원문 "Knowledge distillation **[6, 32]** was…", "Recently adopted in anomaly detection **[8, 12, 16, 26, 74, 86]**, …" — 인용 클러스터 2곳 무표기 제거. §2 ✓ |
| 9 | "since the teacher and student models are both trained on normal data… quantify the anomaly level." (§3.5) | 실재, 절단 | 원문은 "…quantify the anomaly level **of a given sample**." — ellipsis 없는 절단. §3 ✓ |
| 10 | "best micro AUC is obtained when we combine the teacher reconstruction error with the teacher-student discrepancy" (§6.1) | **EXACT** | §4 Table 3 논의 ✓ |

### B-2. 기술 주장 spot-check (18건+ — 전건 원문 일치, 누락 1계열 별도)

검증 일치: 인코더 CvT 3블록/proj 256/4heads ✓("The encoder module is formed of three CvT blocks, each with a projection size of 256 and four attention heads") · teacher decoder 3블록/student 1블록 ✓ · student가 main decoder 첫 블록 뒤 분기 ✓ · 패치 크기 Avenue 16×16, Shanghai/UBnormal 8×8, Ped2 4×4 ✓ · teacher 100ep/student 40ep ✓ · Adam lr 1e-4, batch 100 ✓(dossier 미기재, 무해) · α=0.4/β=0.3/γ=0.3 전 데이터셋 동일 ✓ · o_t = α‖x−x̂‖²+β‖x̂−x̃‖²+γ·ŷ ✓(Eq.6) · w_i = m_i/Σm_j ✓(Eq.2) · m_i = 패치 내 채널별 최대 그래디언트의 채널 평균 ✓(Eq.1) · L_wMSE ✓(Eq.3형) · L_SD가 w_i 가중 포함 ✓ · 랜덤 마스킹+수치 미공개 ✓("masking ratio"/"mask ratio"/비율 수치 전문 grep 0건 — Verification Log의 "미확인" 판정 정당; GitHub README에도 없음) · phase-2 shared encoder/backbone freeze ✓ · UBnormal pixel-level annotation crop+blend ✓ · 증강 확률 0.25 ✓ · [CLS] 분류 헤드 + BCE ✓ · GRL 부재 ✓(원문 GRL/gradient reversal 언급 0건) · 3M params/0.8 GFLOPs/1655 FPS ✓ · 4개 벤치마크 ✓ · 3D 시공간 필터링 + 프레임 max + 시간축 Gaussian ✓ · 합성 0%→88.5 vs 25%→91.3 (§6.2 "합성 기여도 ablation" 주장 지지) ✓.

서지(WebFetch): 제목·저자 6인·순서 ✓, v1 2023-06-21 ✓, v2 2024-03-09 ✓, "Accepted at CVPR 2024" ✓, CVPR virtual poster/30615 페이지 실재·동일 논문 ✓, github.com/ristea/aed-mae 실재 ✓.

### B-3. 발견사항

**S-M1 (MAJOR) — anomaly-map 예측 분기 + "이상 무시(overlook) 재구성 타깃" 전체 누락 (임무 '유사/차이 전수' 미달, R9 위험 분석 약화)**
원문 §1/§3/§4: (i) "task the masked AE model to **jointly reconstruct the original frames (without anomalies) and the corresponding pixel-level anomaly maps**"; (ii) "we add the **anomaly map as an additional channel** to our target image … normal pixels to 0 and abnormal pixels to 1"; (iii) 합성 프레임의 재구성 GT는 **이상 제거된 원본**("forcing our model to overlook the anomalies"); (iv) GT anomaly map을 **모션 그래디언트 가중치에 가산**(Eq.1–2 수정: "we propose to add the anomaly maps and the gradients together, before computing the weights"); (v) ablation상 핵심 — "**to surpass the 90% milestone on Avenue, it is mandatory to introduce the prediction of anomaly maps** in the learning task".
→ dossier §3.1–3.7, §4.1/4.2, §6 어디에도 없음. 문제점 2가지: (a) 유사/차이 표가 "전수"가 아님 — SDMAE도 (합성) 이상 라벨 신호를 **재구성 타깃과 손실 가중에 직접 주입**하는 메커니즘을 가지므로, "이상 정보를 모델이 표현하지 못하게 만드는 학습 신호"라는 관점에서 TSMAE의 GRL 억제와 **개념적 평행선**이 성립한다. 리뷰어가 이를 지적하면 §6.2의 "GRL/레이블 활용은 SDMAE로 지지 불가·독립 justify" 프레임과 §4.2의 GRL 차이 행이 기습당한다. (b) §3.3의 "손실 함수는 모션 그래디언트 크기에 따라 가중" 서술은 합성-증강 프레임에서는 부정확(그래디언트+anomaly map 합산 가중). **요구 수정**: §3.6에 anomaly-map 타깃 채널·overlook 재구성·가중치 가산 3요소 추가, §4.1 유사점 표에 "(합성) 이상 라벨 신호의 학습 주입" 행 신설(위험도 평가 포함), §8 overextension에 대응 문구 추가.

**S-m1 (MINOR)** — R21 핵심 인용(#4)의 출처 오기: "(Section 3)" → 실제 **§1 Introduction**. R21 방어 논리를 논문 각주로 옮길 때 출처 정확성이 직접 노출되는 인용이므로 교정 필수.

**S-m2 (MINOR)** — #4 원문 말미 "[101]" 탈락: SDMAE는 self-distillation이라는 명칭 자체를 **선행 [101](Zhang et al.)에 귀속**시킨다("a process known as self-distillation [101]"). 이 귀속 사실은 R21 방어에 오히려 유리한 재료(용어 계보가 SDMAE 단독 창안이 아님)이므로 인용에 마커를 복원하고 §5.1에 1줄 반영 권장. (#5–#9의 마커 탈락·ellipsis 없는 절단도 일괄: A2상 본문 복사 금지 문서라 치명적이지 않으나, '한 글자 단위' 발췌 규약 위반.)

**S-m3 (MINOR)** — §3.1: 원문 "**All decoder blocks** have four attention heads and a projection dimension of 128" — **teacher decoder도 proj 128**. 현재 표기(student에만 128 병기)는 teacher decoder=256 오독 유발.

**S-m4 (MINOR)** — §6.1 4행: "비대칭 decoder 깊이가 discrepancy 생성에 유효"의 근거로 Table 3 인용 — 해당 인용은 **점수 결합 전략**(teacher recon + discrepancy 조합)의 유효성 근거이지 깊이 비대칭의 근거가 아님(6행과 근거 중복). 깊이 비대칭의 직접 ablation은 원문에 없음 → "지지 가능" → "간접 지지(분기 구조 전제하의 결과)"로 강등 필요.

**S-m5 (MINOR)** — §1 "발표 유형: oral/poster 구분 미확인" — CVPR 페이지(WebFetch)가 **Poster**로 명시. 확인 가능했던 항목.

**S-m6 (MINOR)** — §4.2 "평가 지표 | AUROC on time-series anomaly benchmarks" — 본 연구 대표 지표는 **PA%K-AUC F1(best-epoch 선정 지표) + VUS-ROC/PR + Affiliation** (271_CONFIG_TRUTH §VIII Threshold/Evaluation; EXPERIMENT_PROTOCOL_TRUTH:185 "논문 5지표"). roc_auc도 산출되긴 하나(evaluator.py:835) 대표 서술로 부정확 — NRdetector dossier의 본 연구 서술(VUS/PA%K-AUC/Affiliation, 정확)과 문서 간 불일치이기도 함.

참고(수정 불요): §4.1 "2단계 학습" 유사 행 — SDMAE는 phase 2에서 backbone 동결, 271은 warmup 후 teacher 계속 학습(freeze_teacher_after_warmup=False, 정본 §VI) — 이 비대칭은 오히려 **차이점 재료**로 추가 가치 있음.

### B-4. 임무 완수도 (R21/R9)
- R21 명명 근거: 핵심 원문 4건 확보·전건 실재 ✓, 용례 분석(§5.1–5.3)은 Phase 3 사용 가능 수준 ✓ (단 S-m1/S-m2 교정 전제).
- R9 포지셔닝: 옵션 A/B/C + 권장 + 초안 문장 ✓ 사용 가능. 옵션 C 초안의 "coining the term"류 표현은 옵션 B에만 있고 C에는 없음 — [101] 귀속(S-m2) 반영 시 B 각주 문구의 "coining the term self-distillation" 은 "adopting/extending the term"으로 완화 필요(원문이 선행 귀속이므로 'coining'은 사실과 어긋남).
- 위험도 표기: 유사점 표 전행 존재 ✓.

---

## C. NRDETECTOR_DOSSIER.md

### C-1. Verbatim 전수 검증 기록 (26/26 원문 실재)

원문: arXiv HTML 2501.11959v1 직접 다운로드 + 대조. frontmatter의 "전 인용문 원문 대조 완료" 주장 — **본 검증에서 실제 성립**.

| # | 인용 (dossier 위치) | 판정 |
|---|---------------------|------|
| 1–4 | §2 (a)–(d) 동기 4단 인용 (비지도 한계/라벨 비용/약한 라벨/verified positive) | **전건 EXACT** (§1 ✓) |
| 5 | contribution 1 ("We focus on a novel and practical scenario…") | **EXACT** |
| 6 | PU=SSL 특수형 (§2) | **EXACT** |
| 7 | "The extracted temporal dependencies will be learned through the pre-trained…" (§4.1) | **EXACT** |
| 8 | "Following the pre-processing methods in (Xu et al., 2021)…" | **EXACT** ([Anomaly Transformer]는 브래킷 주석으로 적법) |
| 9 | "We split the set of all segments by 7:3 ratio…" | **EXACT** |
| 10 | window-size 인용 (§3.1) | 실재, 결함: 원문 "unlike TreeMIL **(Liu et al., 2024)** and WETAS **(Lee et al., 2021)**" — 인용 괄호 2건 무표기 탈락 → N-m4 |
| 11 | 라벨 40% 구성 (§3.2, §5.1) | **EXACT** |
| 12 | sweep 서사 ("WEATS [sic]" 포함, §5.3) | **EXACT** — [sic] 처리 적절, 위치 §5.3 ✓(실측 §5.3 구간) |
| 13–15 | "These models are trained by using only normal segments…" / "know as much label information as our method" / "which are the main baselines we need to compare." | **전건 실재** (위치는 N-m2, 지칭은 N-m3) |
| 16–17 | 공정성 인용 2건 ("This ensures that the label information…" / "Note that this kind of labeling…") | **EXACT**, §5.5 ✓(실측 §5.5 본문) |
| 18 | PA 배제 인용 (ellipsis로 인용 클러스터 생략) | 실재 — ellipsis 적법; 말미 절단(원문 "…PA%K (Kim et al., 2022), which actually calculates the AUC of PA%K…")은 dossier 본문이 "K 의존 제거 위해 AUC화"로 별도 정확히 반영 |
| 19 | "Different metrics provide different views…" | **EXACT** |
| 20 | "the extractor here can be replaced with another temporal feature extractor" | **EXACT** (§4.2.1 ✓) |
| 21 | 비지도 괄호 인용 클러스터 (10개 author-year) | **EXACT** — 한 글자 일치 |
| 22 | weak-supervision 명제 인용 | **EXACT** (원문 "(Lee et al., 2021; Liu et al., 2024; Sultani et al., 2018)"; dossier의 [WETAS]/[TreeMIL]/[DeepMIL] 브래킷 매핑 전건 정확) |
| 23 | baseline 목록 인용 (§5.2 Baselines 표기) | **EXACT** (위치는 N-m2) |
| 24 | "PU Learning has been employed in the field of anomaly detection …, including time series (Nguyen et al., 2011; Zhang et al., 2021b)" | **EXACT** — "거의 없음" 스코핑 경고(§5)의 근거 정확 |
| 25 | "based on the Non-negative Risk Estimator (Kiryo et al., 2017)" (§4.2.3) | **EXACT** — Eq.5와의 긴장 지적(§6)도 원문 실측과 일치: Eq.5는 "Based on distribution alignment (Zhao et al., 2022)" + "Though the PU Loss R_pu is biased, Proposition 1 in (Zhao et al., 2022) shows … upper bounded" |
| 26 | "turning the problem into a MIL problem" / "13 competitive baselines" | **EXACT** (후자: §5.3 "13 competitive baselines" 실재) |

### C-2. 기술 주장 spot-check (25건+ — 전건 원문 일치)

Table 1 수치 **전행 일치**: EMG 304,400/130,900/222TPS/8dim/5.8% ✓ · SMD 495,870/212,550/463/38/4.2 ✓ · PSM 61,488/26,353/191/25/27.8 ✓ · MSL 51,610/22,119/99/55/10.5 ✓ · SMAP 299,331/128,286/506/25/12.8 ✓ · 평균 AR 12.22% ✓ · L=100 전 데이터셋·비중첩 ✓ · Eq.5 `R_pu = 2π_P|mean_L f −1| + |mean_Ū f − π_P|` **한 글자 일치** ✓ · Eq.6 L_smooth(인접 점수 차 제곱합) ✓ · Eq.7 L_sep(unlabeled 평균 − labeled 평균; 방향 서술 일치) ✓ · λ₁=λ₂=8×10⁻⁵ ✓ · e₁:=P(Ỹ=0|Y=1) ✓(Table 4 캡션) · Table 4 sweep {0.4, 0.2, 0.0} ✓(표 데이터 실측; main 0.6 별도 ✓) · e₁=0→MIL 환원 ✓ · 13 baselines = 비지도 6 + semi(++) 4 + weak 3 ✓(++ 4종 명단 일치: "AutoFormer++, FEDformer++, TimesNet++, One-fits-all++, except for Anomaly Trasfomer and DCdetector") · Table 2 = F1/F1_PA%K만 ✓(캡션) · DiCNN(WaveNet)·WETAS 프레임워크·GAP ✓ · DiCNN **7층** ✓("both Di-CNN and Transformer are seven layers") · Table 8 Appendix·DiCNN 소폭 우위 ✓("basically the same … but the former performs slightly better") · 6-layer MLP+ReLU+Sigmoid ✓ · Adam lr 1e-4, batch 32 ✓ · baselines "suggested hyperparameters reported in the corresponding previous literature" ✓ · Table 5 서사("단독 사용 해악, 동시 사용 이득") ✓ — 원문 "Both … will degrade the performance … when used individually. However, if … simultaneously, the performance improves." 와 일치 · Table 6 서사(PU Loss 큰 이득 + TC 효과) ✓ · HOC(Zhu et al., 2021c) 임계값 자동화·training-free ✓("training-free automated estimator (Zhu et al., 2023a, 2021c)") · Stage-2 rank→비율 k→pseudo-label→전이 추정 ✓(§4.3 실측) · Fig 3 class prior 둔감("robust with a wide range of class prior")+batch size ✓(§5.6) · cosine similarity 기반 reliable negative + KNN 네트워크 label propagation ✓ · "anomlay" 오탈자 실재 ✓ · 9인 저자+소속(UCSC/HKBU/Squirrel AI/RIPED·CNPC/DUT/BIAI·ZJUT·D5/UCSC) **전건 헤더 일치** ✓ · KDD '25 V.1, Toronto 2025-08-03~07, DOI 10.1145/3690624.3709257, Zenodo 10.5281/zenodo.14676716 ✓(논문 헤더 실측) · GitHub(WebFetch): MIT ✓, main.py/solver.py/data_loader.py/evaluation.py + models/·modules/·metrics/·pretrained_model/ ✓, 동봉 데이터 EMG만 ✓ · "NRdetector"(소문자 d) 표기 ✓.

**R19 핵심 발견 재검증 (전수 grep, 적대적 재현)**: related work 본문 구간(실측 오프셋 11,162–16,325) 내 모델명 출현 **0건** — DCdetector/Anomaly Transformer/TimesNet/One-fits-all/AutoFormer/FEDformer/TreeMIL/DeepMIL 최초 출현 전부 §5 구간(@51,478 이후), WETAS만 §4.2.1(@32,296) ✓ dossier 서술 그대로. **AutoFormer(Wu et al., 2021)·FEDformer(Zhou et al., 2022)는 인용 자체가 related work에 부재**(전문에서 각 1회, §5 baselines 단락뿐) ✓. Yang et al., 2023의 RW 내 1회 출현은 일반 명제 지지 인용("identify anomalies as deviations from these patterns (Yang et al., 2023)") ✓. — **dossier의 핵심 주장 3건 모두 독립 재현 일치.**

### C-3. 발견사항

**X-M1 (MAJOR, 공유)** — §A 참조. §5 전제부("소수 labeled anomaly + 대량 unlabeled") 1줄 수정.

**N-m1 (MINOR)** — §3.4 "평가지표 (11개)" 구성 오류: 원문 Table 3의 11지표는 F1, P, R, F1_PA%K, **F1_PA**, Aff-P, Aff-R, R_A_R, R_A_P, V_ROC, V_PR (캡션 실측: "The F1_PA is the F1 score using the PA strategy"). 즉 **PA를 main 테이블(Table 2)에서만 배제했지 Table 3에는 F1_PA가 포함**된다. dossier는 F1_PA를 누락하고 F1-W를 "+보조"로 나열해 11개 구성이 어긋남(F1-W는 ablation 표 전용). PA 배제 논리를 차용할 Phase 3 문장이 "전 표에서 PA 미사용"으로 과장될 위험.

**N-m2 (MINOR)** — 섹션 귀속 3건 오기: baseline 목록 인용(#23)·++변형 인용(#13)·"main baselines" 인용(#15)은 "(§5.2)"가 아니라 **§5.1** (arXiv v1 실측: Baselines 단락이 "5.2. Experimental Setting" 헤더 직전에 종료). Implementation Details/Evaluation metrics 인용의 §5.2 귀속은 정확.

**N-m3 (MINOR)** — §3.3 항목 3: "which are the main baselines we need to compare"가 DeepMIL·WETAS·TreeMIL 3종 나열 직후에 부착되어 3종 전체를 지칭하는 듯 읽힘. 원문은 "we compare our method with **WETAS … and TreeMIL …**, which are the main baselines" — **WETAS·TreeMIL 2종만** 지칭. (dossier 후속 문장 "이 둘로만 좁힘"이 자체 정정하고 있으나 인용 부착 위치 교정 필요.)

**N-m4 (MINOR)** — window-size 인용(#10)에서 TreeMIL/WETAS 뒤 author-year 괄호 2건이 ellipsis 표기 없이 탈락.

**N-m5 (MINOR)** — §1/§3.5 "사전학습 **후 고정** 추출": 원문에 freeze/frozen 명시 없음(전문 grep 0건). "pre-trained" 표현 + 코드 `pretrained_model/` 구조에서의 **추론**임 — §0은 "방증"으로 옳게 표기했으나 §1·§3.5는 단정 서술. 추론 표기(INFERENCE) 또는 "공식 코드로 확인 가능" 단서로 강등 필요 (§6 한계 노트의 7:3 split 처리와 동일 수준으로).

### C-4. 임무 완수도 (R16/R19/R20) + 완결성
- **R16**: 데이터셋/split/라벨 구성/baseline 3계층/지표 11개/구현/ablation 5종 — 상세·정확(상기 minor 제외) ✓. "main 1점 + sweep 3점", 깔때기 비교 구조, 양보-반박 PA 처리 등 실험 구성 차용 포인트가 실제 Phase 3에서 쓸 수 있는 수준 ✓.
- **R19**: 인용 처리 실제 사례 — 핵심 주장 3건 전수 grep 재현 일치(C-2), 운영 규칙 3조 ✓. 본 review의 독립 재현으로 신뢰도 상향.
- **R20**: 차이축 9개(D1–D9) — 본 연구 측 전제 전건을 정본과 대조: D1(end-to-end/표현 학습 통합) ✓(GRL student hidden, force_mask_anomaly, FM/OD — 271_CONFIG_TRUTH §VI/§VIII), D6(score=recon+discrepancy, threshold-free 지표) ✓, D8(contaminated: test 앞 50% 편입) ✓(EXPERIMENT_PROTOCOL_TRUTH §②), 평가 셋 VUS/PA%K-AUC/Affiliation ✓(evaluator.py:640-770, PROTOCOL:185). "거의 없음" 주장의 정밀 스코핑(§5 주의 단락)은 원문 인용 #24로 뒷받침되는 **모범적 방어** ✓ — 단 X-M1 단서 필요.
- **완결성(세션 중단 의심)**: §0–§6 전 섹션 실재, 말미 문장 완결, frontmatter 검증 주장 성립, 누락 흔적 없음. **완결로 판정.**

---

## D. 요구 조치 목록 (fixer 인입용)

| ID | 문서 | 심각도 | 조치 |
|----|------|--------|------|
| X-M1 | 양쪽 | MAJOR | 본 연구 라벨 설정 전제에 RESEARCH_SYNTHESIS §⑤ 단서(상한 케이스/Phase 3 결정/R32 계획) 추가 |
| S-M1 | SDMAE | MAJOR | anomaly-map 타깃 채널·overlook 재구성·가중치 가산 3요소를 §3.6에 추가, §4.1 유사점 행 신설, §8 대응 추가 |
| S-m1 | SDMAE | MINOR | R21 핵심 인용 출처 "(Section 3)" → "(Section 1, contributions)" |
| S-m2 | SDMAE | MINOR | "[101]" 마커 복원 + §5.1에 용어의 선행 귀속 1줄 + 옵션 B "coining" 표현 완화 |
| S-m3 | SDMAE | MINOR | teacher decoder projection 128 명기 |
| S-m4 | SDMAE | MINOR | §6.1 4행 근거 강등(간접 지지) |
| S-m5 | SDMAE | MINOR | 발표 유형 Poster 확정 기재 |
| S-m6 | SDMAE | MINOR | §4.2 "AUROC" → "PA%K-AUC F1·VUS·Affiliation" (NR dossier와 정합화) |
| N-m1 | NRdetector | MINOR | 11지표 구성에 F1_PA 포함·F1-W 분리 |
| N-m2 | NRdetector | MINOR | 인용 3건 §5.2→§5.1 귀속 교정 |
| N-m3 | NRdetector | MINOR | "main baselines" 인용을 WETAS·TreeMIL 2종에 부착 |
| N-m4 | NRdetector | MINOR | window-size 인용 탈락분 ellipsis 표기 |
| N-m5 | NRdetector | MINOR | "고정 추출" 단정 → 추론 표기 |

검증 부산물(원문 텍스트·대조 스크립트): `/tmp/dossier_verify/` (휘발성, 산출물 아님).
