---
phase: 1
agent: fixer-3
directives: [T1, R2, R26, R5]
last_modified: 2026-06-10
review_input: paper/99_reviews/p1_digests_r1.md (BLOCKER 3 + MAJOR 5 + MINOR 11 = 19건)
targets:
  - paper/01_research_understanding/NOTION_DIGEST.md (r2)
  - paper/01_research_understanding/CONFERENCE_PDF_DIGEST.md (r2)
sources_reverified:
  - "Notion Page 0 덤프: tool-results/mcp-claude_ai_Notion-notion-fetch-1781093695371.txt (75,820 chars decoded, python 슬라이스 재확인)"
  - "Notion Page B 덤프: tool-results/mcp-claude_ai_Notion-notion-fetch-1781093708082.txt (108,461 chars decoded, python 슬라이스 재확인)"
  - "paper/윤기오_대한산업공학회_2026_춘계.pdf (p6–7, 9–16, 19–22, 34 직접 재독)"
  - "paper/99_reviews/p1_reconciliation_r1.md §III (WaDi A2=123 코드 확정 인용)"
disposition: "19/19 처리 — FIXED 18, PARTIALLY-REVISED(리뷰 측 조정) 1, REJECTED 0"
---

# Phase 1 Digest Fix Log (r2) — p1_digests_r1 전수 처리

모든 수정은 원천 재확인 후 적용. 창작 없음 — 각 건에 원천 근거(덤프 문자 오프셋 / PDF 페이지) 명기.

## A. NOTION_DIGEST.md (12건)

| ID | 등급 | 처리 | 내용 / 원천 근거 |
|----|------|------|------------------|
| NB-1 | BLOCKER | **FIXED** | II-3 truth 표 WaDi A2 Features 127→**123**. 근거: Page B 덤프 @18880 `<td>WaDi A2</td><td>123</td>` + Page B 전문 "127" **0회**(regex 전수); PDF p19 표 WaDi A2 = 123 dim(직접 재독); 코드 확정 = `p1_reconciliation_r1.md` §III (exp271 metadata `num_features=123`, raw CSV 124 cols=123+label, 127=all-NaN 4컬럼 drop 이전 원본 수). Page 0의 127 기재(@2793 d_model 표·@35093 num_features 표·@66011 §5.2.1)는 원문 전사로 유지하되 I-3/I-7/I-9에 "원천 간 모순 — 123이 검증값" † 주석 부착, §IV-11 신설. |
| NB-2 | BLOCKER | **FIXED** | II-1 산식에 "**× 22 active models**" 인자 복원 — 원문(Page B @23452 Total 문장) "Pattern A 39 base dataset runs + 2 new SMAP/MSL concat runs = 41 runs per condition × 2 conditions × 22 active models = 1,804 (model, dataset, condition) cells" 정확 전사. 보너스(리뷰 권고): Snapshot callout(@3617 "9 actual base datasets ... = 39 dataset runs (Pattern A)") vs Total(39+2=41)의 원천 내부 모순 주석 추가. |
| NM-1 | MAJOR | **FIXED** | I-10을 두 표로 분리 — [truth 등급 — R26]은 [4],[6]–[10],[12] 7건(= Page B [B2]/[D1]–[D5]/[D8] 동일 항목)만; 방법론 인용 [1] He, [2] Ganin, [3] Kim, [5] Esser, [11] Lin 5건은 "[Notion의 주장 — 검증 완료 주장]"으로 강등 + Phase 4 재확인 필수 명기. 근거: MASTER_ORCHESTRATION_PROMPT [R26] 원문 "비교 대상 모델 reference 및 데이터셋 reference"(L491) — 방법론 인용은 범위 외. Page 0의 검증 주장 원문("DBLP/IEEE Xplore/ACM DL/openaccess.thecvf.com 1차 출처에서 검증") 귀속 표기는 유지. |
| NM-2 | MAJOR | **FIXED** | II-6 상단에 수치 유효성 한정 블록 신설: ① §3 status = 2026-05-22 시점(Page B @26905 status callout), ② 2026-05-25 paper-faithful 재실행 — "영향 모델: **15 종** (9 QuoVadis + 6 non-self_norm SOTA, gcn_lstm 중복)"·"결과 row 는 재실험 완료 후 swap-in 예정"(@101410-101492 변경 이력 원문) + pca_error AGGREGATION FIX/mlp·mlpmixer·transformer REIMPL/neural_base predict Pass 2 fix(@1611-2281 audit 표 원문) + 5번 실험 폐기→6번 재실행(@3285 부근 callout), ③ 2026-06-04 v2 12종(§9 변경 이력), ④ 기존 per-entity STALE 유지. 하단 주의문도 "simple/neural/legacy 행은 column 무관 fidelity 수정 이전 산출"로 확장. |
| NM-3 | MAJOR | **FIXED** | II-2b 신설 (실험 조건 상세): §1.2.1–1.2.5 HP preset 전체(anomaly_transformer win=100·d_model=512, omnianomaly seq=100, memto train_stride=100·2-phase, catch win=192, tranad lr=1e-4 config-layer(2026-06-05)·LeakyReLU, gdn batch=32(run.sh repro) 등 — Page B §1.2 표 @6597-16000 직접 전사); weak label = `max(point label over window)` train-split 한정 leak-free(@11951-12060 원문); provenance 태그 4종([fixed]/[normalization]/[runtime-estimated]/[impl-invented], §1.2.5 callout 원문); 2026-06-04 faithfulness pass v2 핵심(§9 "12 모델 upstream 충실도 재정리" 원문 — timesnet SMAP-script HP 근거, memto FRESH re-init, random seed=None 5-run mean±std, deepmil 유일 leak-free vs WETAS-family fit-on-test); 2026-06-02 boundary-safe TEST windowing(21개 windowing baseline, @102285 원문). |
| NM-4 | MAJOR | **FIXED** | I-부 보강 5건: ① I-1 마스킹 "약 8개"→"round(50×0.15)=8개 고정, batch 균일"(Page 0 @19799 §3.3.2 원문), ② I-3 Force-mask-anomaly priority 공식(`priority_p = 1[anomaly]·1000 + η_p`, TopK_8, budget 초과 시 random subset — @20025 원문 수식), ③ I-3 디코더 구조(`use_transformer_encoder_decoder=True`, 두 디코더 모두 self-attention only TransformerEncoder, cross-attention 없음 — @23022 원문), ④ I-4 teacher_only=True 플래그 메커니즘(@33369 §3.5 원문) + AMP bf16(2026-05-27 사유 @61555)/eval_interval=5/random_seed=42(@61255-61900)/trainer config validation 5종(@16935 callout), ⑤ I-5 GRL adaptive λ anchor(w=student decoder 마지막 weight, @26224 원문). |
| Nm-1 | MINOR | **FIXED** | §IV-12 신설 — Page 0 내부 모순: §2.4(@15892 부근, "recon:disc = 4:1 + FM 점수 제외" 수식 포함) vs §3.6 evaluator(@33883 "정규화 후 1:1 결합")·§4.3.3 anomaly_score_mode 역할란(@45363 "1:1 결합")·§5.3 Top 3(@69030 "1:1 결합") stale text. digest의 4:1 채택은 유지(올바름), 모순 자체를 의심 지점에 기록. |
| Nm-2 | MINOR | **FIXED** | II-1에 원문 불일치 주석 — Snapshot callout 원문 "weakly-supervised **5종** 50 epoch (2026-06-06 통일)"(@3835) vs 제목/§1.1/§6.4의 4종 + `nrdetector_full` 별도 존재(@65501 §6 callout, @92660 §9). 4종 해석 채택 유지. |
| Nm-3 | MINOR | **FIXED** | Q2/Q4(zscore) 폐기 등급 통일 — II-4의 [Notion의 주장]→[검증된 사실 후보] (리뷰 권고대로 사실 후보 채택). 근거: Snapshot(@3700 부근 "Q2/Q4 (zscore) 폐기") + §2(@26790 "Q2/Q4 (zscore 변형) 는 폐기되었음") 양쪽 명시. |
| Nm-4 | MINOR | **FIXED** | II-4 per-entity 예외 목록 전체 복원 — 원문(@25672) "(a) 진짜 단일 entity 변형(PSM/SWaT/WaDi/**simulation**/**`*_simple`**/**단일 machine**) → per-entity ≡ whole-array NO-OP(bit-identical)". |
| Nm-5 | MINOR | **FIXED** | II-3에 TEP 행 추가 — 원문(@22874 §2.1) "TEP (Tennessee Eastman, 참고)" / 52 process variables / Runs "—" / "로컬: 4 RData files 일치" → "보유·검증 완료, 비교 실험 미사용(참고)" 한 줄 + [D6]/[D7] 연결. License: §2.1은 "Harvard Dataverse 공개", §6.4 citation 표(@76792)는 "**CC0 Public Domain Dedication**" — 양쪽 모두 기재. |
| Nm-6 | MINOR | **FIXED** | [B4b](usad official-affiliated PyTorch impl, @48323 원문) / [B5b](TranAD-저자 DAGMM reimpl 코드 인용, @49183 원문) 추가. IV-6 정정: Page B가 이미 **결정**한 사안 — "조치 = scoreboard에서 `dagmm_tranad`로 relabel · energy-DAGMM paper target과 직접 비교 금지"(@94312, 판정 "RELABEL only") — "검토 필요" 수준의 열린 질문 서술을 기결정 반영으로 교체. |

## B. CONFERENCE_PDF_DIGEST.md (7건)

| ID | 등급 | 처리 | 내용 / 원천 근거 |
|----|------|------|------------------|
| PB-1 | BLOCKER | **FIXED** | ①(p20 행)·⑤(Baseline 절) "25종/25개" → "**26종**(+Ours=27)". 근거: PDF p20 표 직접 재실측 — Simple 5(Random/Sensor-Range/PCA/L2-Norm/kNN) + Neural 3(MLP/MLP-Mixer/Transformer) + SOTA 14(GCN-LSTM…CATCH) + Weak 4(DeepMIL/WETAS/TreeMIL/NRDetector) = 26 + Ours. 교차: p22 rank 첨자 (27)까지 실재(예: Anomaly Trans. WaDi A1 0.0806₍₂₇₎), Page B 22 active + 4 weak = 26 정합. 혼동 원인 추정(p34 refs [1]–[25]) 주석화. |
| PM-1 | MAJOR | **FIXED** | ④ 아키텍처 블록에 원천 간 모순 경고 신설 — PDF p12/15/16 "Patchify (1D-CNN)"(직접 재독 확인) vs Notion Page 0 exp271 `patchify_mode='linear'`(Page 0 @12930 부근 callout "본 baseline은 Set C의 patchify_mode='linear'" + I-8 파라미터 표; `'patch_cnn'`은 미사용 옵션 목록 @66039 부근). ⑧에 코드·실험 ID 기준 확정 REQUEST 추가. |
| Pm-1 | MINOR | **FIXED** | ② p6 인용 완화 어미 복원 — 원문(p6 직접 재독) "...semi-supervised learning 접근이 가장 현실적이고 강력한 해결책**이 될 수 있음**". ⑦-6의 "발표체 단정" 사례를 "원문은 이미 hedge — 다만 근거 보강은 여전히 필요"로 조정 (리뷰 권고대로 비판 강도 환원). |
| Pm-2 | MINOR | **FIXED** | ③ "Ristea et al. 계열" → 외부 지식 귀속 표기 — PDF p9–10 직접 재독: 제목 "Self-Distilled Masked Auto-Encoders are Efficient Video Anomaly Detectors (CVPR 2024)"만 있고 저자명 없음; p34 references에도 해당 논문 부재. "(저자명은 PDF에 없음 — 외부 확인, Phase 4 재검증 대상)" 주석. |
| Pm-3 | MINOR | **FIXED** | ④ 정식화 괄호 주석 + ② PU 맥락 서술을 추론으로 정정 — p11 직접 재독: 𝒳ᴺ_lab(labeled normal) 존재, 슬라이드는 라벨 사용 범위 미명시, ℒ_disc의 ℳᴺ 정의만 𝒳ᴬ_lab 참조 → "손실 수식에는 𝒳ᴬ_lab만 등장"으로 교체 + 추론 표기. |
| Pm-4 | MINOR | **FIXED** | ⑤ p19 미전사 column 추가 — p19 표 직접 재독: #Training/#Testing(SWaT 719,959/224,960; excl22 719,959/189,060; WaDi A1 1,296,001/86,401; A2 870,972/86,402; PSM 176,401/43,921), #Anomaly Regions(14/13/7/7/29), **Train AR(%)** SWaT 1.63/WaDi A1 0.52/A2 0.76/PSM 6.20 — contaminated 프로토콜 핵심 통계로 명기. "수치 동결 금지" 단서 동일 적용. |
| Pm-5 | MINOR | **FIXED** | ⑤ p21 metric별 reference 표 추가 — p21 직접 재독: F1 "no single canonical origin"(원문), F1_PA←Xu et al. WWW 2018 [4], F1_PA%K/PRC_PA%K←Kim et al. AAAI 2022 [5], VUS-PR/ROC←Paparrizos et al. PVLDB 2022 [6], Aff-F1←Huet et al. KDD 2022 [7] (p34 [4]–[7]과 대조 일치). Phase 4 공식 소스 재확인 단서 부착. |

## C. 비고

- **REJECTED 0건** — 리뷰의 19건 발견 전부 원천 재확인 결과 타당하여 수정 적용. 유일한 조정: Pm-1은 "digest 수정"과 동시에 리뷰가 지적한 대로 **r1 digest의 자기 비판(⑦-6)이 과했음을 환원**하는 양방향 수정 (FIXED로 분류, 비판 강도 조정 포함).
- **NM-3 tranad lr 표기 주의**: Page B §9의 2026-06-04 v2 항목은 "lr=0.01"로 기재하나, §1.2.3 preset 표(2026-06-05 갱신)는 "**lr=1e-4** (constants.py lr 재현값; paper-text 0.01은 run-code 값 아님; 2026-06-05 config-layer)" — digest II-2b는 최신 상태(§1.2.3)를 채택하고 날짜 출처를 병기했다. 기존 §IV-8(tranad lr 의심 지점)과 정합.
- 후속 검증 의뢰(리뷰 §C-4) 처리 현황: WaDi A2 feature 수는 `p1_reconciliation_r1.md` §III에서 이미 코드 기준 **123 확정** — digest에 반영 완료. patchify mode(linear vs 1D-CNN)는 CONFERENCE_PDF_DIGEST ⑧ REQUEST로 code-digest/271truth 라인에 전달함 (미해소 — 코드 확정 대기).
- paper_legacy/ 미접근, 코드·실험 환경 무변경. 쓰기 대상: 본 fix log + digest 2개만.
