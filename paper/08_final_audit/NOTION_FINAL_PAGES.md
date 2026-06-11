---
phase: 8
agent: notion-polisher-C
directives: [R3]
last_modified: 2026-06-11
scope: |
  NOTION_ENRICHED_B1_body.md(본문 FIG·TAB 7페이지) + NOTION_ENRICHED_B2_appendix.md(appendix 10페이지 + OVERVIEW)
  통합·정제 최종본. 페이지 순서 확정: OVERVIEW → FIG-1~4 → TAB-1~3 → TAB-A 묶음 2 →
  TAB-B1~B4 → FIG-B1 → ALG-C1 → TXT → R-PROBE (총 18 페이지 블록).
basis: |
  사실·수치·실행 지침·영문 캡션: 두 입력 문서에서 무손실 계승 (한국어 표현·구성만 정제).
  템플릿: 전 페이지 동일 차원·순서 — 💡 한 줄 요약 → 메타 표 → 🎯 → 🏁 → 🧪 → 📊 → 📝 → ⚠️ → 🔢.
  각 페이지 메타 표(유형/소스 분류/우선순위/의존성)는 OVERVIEW 대시보드와 일치.
notes: |
  Notion 발행용 — 각 <!-- PAGE: {ID} --> 블록 = 하위 페이지 1장, OVERVIEW 블록 = 부모 페이지 본문.
  발행은 notion-create-pages 경로만 사용한다 (update-page insert_content는 렌더링 파손).
  수치 창작 금지(A8) · Gaussian smoothing 언급 금지(R34) · 조건 명칭은 R24 개명 후 표기만 사용.
---

# NOTION FINAL PAGES — 통합·정제 최종본 (본문 + Appendix 전체)

이 문서는 r2 명세를 확장한 B1(본문)·B2(appendix) 문서를 단일 발행본으로 통합하고, 전 페이지의 구성과 한국어 문장을 정제한 것이다. r2와 B1·B2의 실행 지침·소스 분류·영문 캡션·의존성은 전부 보존했다. 각 페이지는 "그 페이지만 보고 실험 설계와 figure·표 제작이 가능한 수준"을 기준으로 작성되어 있다.

---

<!-- PAGE: OVERVIEW -->

# Placeholder 전체 지도 · 실행 대시보드

> 💡 **한 줄 요약**: 원고의 모든 placeholder — FIG 5 · TAB 12 · ALG 1 · NUM 31 · TXT 2종(4개소) · 권고 실험 1 — 를 한 페이지에서 조망한다. 첫 절의 3구역 대시보드(🟢 즉시 가능 / 🔵 GPU 신규 실행 / ⏳ 대기)가 "오늘 무엇부터 시작하는가"에 답하고, 하위 페이지가 placeholder별 상세 명세를 담는다.

## 🚀 오늘의 시작점 — 3구역 요약

| 구역 | 건수 | 정의 | 첫 행동 |
|---|---|---|---|
| 🟢 즉시 가능 (무학습) | 13건 | 재사용 8 + 측정 5 — GPU·완주 대기와 무관 | SMD 28-machine 분할 통계 스크립트 (의존성 0, 표 2장을 동시 해소) |
| 🔵 GPU 신규 실행 | 11건 | 새 학습 실험 — 큐 등재 후 실행 | #1 baseline SMD/SMAP/MSL (TAB-2 루트의 최대 입력) |
| ⏳ 대기 중 | 3건 + 파생 | 271canon 완주 또는 TAB-2 확정이 선행 조건 | 행동 없음 — 부분 집계 금지 규칙만 준수 |

### 🟢 구역 1 — 지금 바로 가능한 무학습 작업 (13건)

**측정·스크립트 5건** — 학습이 필요 없고, 1회성 스크립트 또는 확인 절차로 끝난다.

| 작업 | 산출 (채워지는 placeholder) | 의존성 |
|---|---|---|
| SMD 28 machine 분할 통계 산출 (loader 산식 재사용) | TAB-1 SMD 셀, Table A.4 SMD 행, §4.1.1 "pending" 문구 해소 | 없음 — 즉시 가능 |
| GPU 모델 확인 (호스트 이력) | TXT-001 | 없음 — 즉시 가능 |
| 추론 비용 측정 (leave-one-out vs single-mask) | TAB-B3, NUM-031 (+§5 "50×" sync) | TXT-001 선행 권장 |
| **R-PROBE** — Student/Teacher hidden probe AUC 비교 | 원고 placeholder 없음 — rebuttal 대비 내부 노트 | 기본형: [271c]만으로 가능. 확장 대조군: 실행 #4 완료 후 |
| 저장소 URL 확정 (게재 시) | TXT-002 ×3개소 | 공개 전 checklist (branch·범위·secret·재현 진입점) |

**재사용·제작 8건** — 기존 결과 추출 또는 다이어그램·의사코드 제작만 필요하다.

| placeholder | 소스 |
|---|---|
| FIG-1, FIG-2, ALG-C1 | 다이어그램·의사코드 제작 (정본 대조 검증) |
| FIG-4, NUM-028 | [271c] best checkpoint + `scoring.py` 추출 |
| FIG-B1 좌패널 (c sweep) | [271c] checkpoint 재채점 (c∈{1,2,4,8,16}, best epoch 고정) |
| TAB-1 (SMD 셀 제외) | EXPERIMENT_PROTOCOL_TRUTH §① 실값 — 이미 tex 반영 |
| TAB-A3 | `comparison/baseline_common.py` `MODEL_CONFIGS` 덤프 |
| TAB-3 행3, NUM-021 | exp287_unmask 완주분 |
| TAB-B4 w/o FM 행, NUM-025 | exp285_no_fm 완주분 |
| TAB-B2 CSMAD 축소 budget | exp298/exp299 완주분 (열 라벨 정합화 결정 필요) |

### 🔵 구역 2 — GPU 신규 실행 11건 (우선순위순)

우선순위 원칙: (1) 본문 핵심 표 의존 → (2) load-bearing 주장 의존 → (3) appendix 방어 실측.

| # | 실험 | 예상 산출 (채워지는 placeholder) | 의존성·선행 조건 | 실행 지침 요약 |
|---|---|---|---|---|
| 1 | baseline SMD/SMAP/MSL 신규 실행 (anomaly-excised) | TAB-2 unsup 행, FIG-3 floor, NUM-008/009/011/016/019, TAB-A6/A7 | per-entity 정규화 적용 확인. SMD 구버전 `3_20260312_*` 폐기 후 재실행, SMAP/MSL normalonly는 **미실행분 신규** | `comparison/run_baseline_queue.py`, variant `normalonly` |
| 2 | weakly supervised 4종 GPU 전체 (contaminated-training) | TAB-2 그룹 6, NUM-013, sync 그룹 B="26", TAB-A6/A7 | 없음 (구현·CPU dry-test 완료) | Q1 variant, 50 epochs. **NRdetector 최우선** |
| 3 | standard clean-train split (CSMAD + 대표 baseline 2–3, 대표 3 데이터셋) | TAB-2 하단 블록, NUM-014~019 | **신규 loader variant 구현** (`*_standard` — prefix 미편입, 평가는 동일 test 후반). CSMAD는 동일 config, `use_grl=True` 유지 (자가 비활성 — False 금지) | EXECUTION-TODO 항목 3 |
| 4 | ablation 행2 (w/o GRL, OD-exclusion 유지) | TAB-3 행2, NUM-023 | TAB-3 대표 열(NUM-020) 확정 | `use_grl=False anomaly_loss_weight=0.0` — dead-component(dynamic margin) 재활성 차단 |
| 5 | symmetric decoder (Teacher 2L) | TAB-B4 2행(symmetric+depth2), **NUM-024 (bullet 3 load-bearing)** | TAB-3 대표 열 확정 | `num_teacher_decoder_layers=2` |
| 6 | ablation 행4 (w/o OD) | TAB-3 행4, NUM-022 | TAB-3 대표 열 확정 | `use_output_discrepancy=False` — score는 자동 recon-only (`resolve_score_weights`, `scoring.py:105-106`·`249-253`). 표 각주로 명시 |
| 7 | label sparsity sweep (p∈{0.75, 0.5, 0.25, 0.1} × 2–3 데이터셋) | FIG-3, NUM-026/027 | `label_keep_ratio` 파라미터 신설 구현 (NoisyLabel 메커니즘 일반화, region 단위, seed 고정). p=1.0은 [271c] 재사용 | 8–12 run, 271 canon + override |
| 8 | contaminated-training 22종 (대표 3 family) | TAB-B1 (+NUM-019 보조) | Δ 산출은 TAB-2 확정 후. SWaT 차원 45 검증 | Q1 variant 큐 |
| 9 | w/o Teacher warmup (250→0) / Teacher depth 1 | TAB-B4 잔여 2행 | TAB-3 대표 열 확정 | `teacher_only_warmup_epochs=0` / `num_teacher_decoder_layers=1` |
| 10 | epoch-budget 50/100 (Anomaly Trans., TranAD) | TAB-B2 | CSMAD 축소분은 exp298/299 재사용 결정 (i)/(ii) 선행 | `baseline_common.py` epochs override |
| 11 | masking ratio sweep (ρ∈{0.05, 0.1, 0.2, 0.3}) | FIG-B1 우패널 | 큐 미등재 — 신규 등재 필요 (v5 전 32항목에서 `masking_ratio` override 0건) | `masking_ratio=<ρ>` × 대표 데이터셋, 4 run |

### ⏳ 구역 3 — 대기 중 (271canon 잔여: SMD 6 · SMAP 49 · MSL 22 — 2026-06-11 실측)

| placeholder | 비고 |
|---|---|
| TAB-2 CSMAD 행 (SMD/SMAP/MSL avg) | **부분 완주 상태로 avg 집계 금지** (sync 그룹 A 보호) |
| TAB-A8 전체, TAB-A6/A7 CSMAD 행 | metadata 집계 스크립트 (단일 산출물 공유 + Table 2 일치 assert) |
| 그룹 N-A (NUM-001/003/004/029) | "six" 확정 조건 — 탈락 family 발생 시 §4.1.1 상수 일괄 수정 |

---

## 📌 분류 라벨 정의

모든 placeholder에는 소스 분류 라벨이 부여되어 있다. 라벨은 "이 빈칸을 채우기 위해 무엇이 필요한가"를 한 단어로 답한다.

| 라벨 | 의미 | 전형적 작업 |
|---|---|---|
| `[재사용]` | 기존 실험 결과 또는 코드 상수에서 **추출만** 하면 됨 — 학습 불필요 | metadata 집계, 코드 덤프, checkpoint 재채점 |
| `[완주 대기]` | 진행 중인 271canon·큐의 잔여 entity **완주 후 집계** (잔여: SMD 6 · SMAP 49 · MSL 22 — 2026-06-11 실측) | 완주 모니터링 → 집계 스크립트 |
| `[신규 실행]` | **새 학습 실험**이 필요 (큐 등재 또는 신규 스크립트) | 큐 항목 작성 → GPU 실행 → 집계 |
| `[신규 측정]` | 학습 없이 **측정·스크립트 1회**로 해결 (통계 추출, 비용 측정, 확인 작업) | 1회성 스크립트 |
| `[제작]` | 실험 무관 — **다이어그램/의사코드 제작·검증**만 필요 | TikZ/벡터 제작, 코드 대조 |

---

## 🗺️ 전체 placeholder 지도 (REGISTRY v3-r1 전수 — 누락 0)

총계: **FIG 5 · body TAB 3(+흡수 1) · appendix TAB 8(+Table A.4 부분) · ALG 1 · NUM 31 · TXT 2종 4개소 · 권고 실험 1**. 담당 열의 B1은 본문 figure/table 페이지(spec-enricher-B1), B2는 appendix·잔여 페이지(spec-enricher-B2)다.

| ID | 한 줄 정의 | 분류 | 담당 |
|---|---|---|---|
| FIG-1 | 3-패널 setting 비교 개념도 (§1) | `[제작]` | B1 |
| FIG-2 | CSMAD 아키텍처 개요 2-패널 (§3.2) | `[제작]` | B1 |
| FIG-3 | label sparsity sweep 곡선 (§4.4) | `[신규 실행]` | B1 |
| FIG-4 | score 성분 분해 정성 시각화 (§4.5) | `[재사용]` | B1 |
| FIG-B1 | 파라미터 민감도 2-패널 (§B.4) | `[재사용(좌)+신규 실행(우)]` | B2 |
| TAB-1 | 데이터셋 통계 family 요약 (Table 1) | `[재사용+신규 측정(SMD 셀)]` | B1 |
| TAB-2 | main 비교 + protocol-effect 블록 (Table 2) — **의존 그래프 루트** | `[완주 대기+신규 실행]` | B1 |
| TAB-3 | ablation 4행 (Table 3) | `[재사용(행1·3)+신규 실행(행2·4)]` | B1 |
| TAB-4 | (TAB-2 하단 블록으로 흡수 완료 — 별도 페이지 없음, D-010 ①) | 흡수 | B1 기재 |
| TAB-A3 | 26 baseline 하이퍼파라미터 전수 (Table A.3) | `[재사용]` | B2 |
| Table A.4 | per-entity 통계 — SMD per-machine 셀 3종 (부분) | `[신규 측정]` | B2 |
| TAB-A6 | SWaT full/excl22 이중 조건 5지표 (Table A.6) | `[완주 대기+TAB-2 동일 소스]` | B2 |
| TAB-A7 | 잔여 4지표 전수 (Table A.7) | `[완주 대기 — TAB-2 동일 소스]` | B2 |
| TAB-A8 | CSMAD per-entity 109행 (Table A.8) | `[완주 대기]` | B2 |
| TAB-B1 | contaminated-training 22종 + Δ (Table B.1) | `[신규 실행]` | B2 |
| TAB-B2 | epoch-budget 민감도 (Table B.2) | `[신규 실행(부분 재사용)]` | B2 |
| TAB-B3 | 추론 비용 3×3 (Table B.3) | `[신규 측정]` | B2 |
| TAB-B4 | 확장 ablation 7행 (Table B.4) | `[재사용(no_fm)+신규 실행(3종)]` | B2 |
| ALG-C1 | 학습 의사코드 검증 (Algorithm C.1) | `[제작 — 코드 대조]` | B2 |
| NUM-001~031 | inline 수치 31건 — 파생 소스 단위 8그룹 (아래 그룹 표) | 그룹별 상이 | 그룹별 |
| TXT-001 | GPU 모델명 ×1개소 (§A.1) | `[신규 측정 — 확인]` | B2 |
| TXT-002 | 코드 저장소 URL ×3개소 (Abstract·§A.1·§5) | `[결정 사항]` | B2 |
| R-PROBE | GRL probing classifier (권고 — 원고 비반영, D-014 (b)) | `[신규 측정]` | B2 |

**NUM 8그룹 요약** — 31건 전수. 각 그룹은 소스 실험이 완료되면 그룹 내 전 항목이 동시에 풀린다.

| 그룹 | NUM | 소스 | 분류 | 상세 페이지 |
|---|---|---|---|---|
| N-A (family 수 sync) | 001, 003, 004, 029 | 271canon 완주 + TAB-2 완성 → "six" | `[완주 대기]` | TAB-2 |
| N-B (baseline 총수 sync) | 002, 005, 030 | weak 4종 GPU 완주 → "26" (미완 시 "22" fallback) | `[신규 실행 의존]` | TAB-2 |
| N-C (Table 2 본 블록 파생) | 006–013 | TAB-2 완성본 집계 | `[집계만]` | TAB-2 |
| N-D (protocol-effect 파생) | 014–019 | standard-split run (+019는 contaminated run 공유) | `[신규 실행]` | TAB-2 |
| N-E (ablation 파생) | 020–023 / **024, 025** | TAB-3 / **TAB-B4** | 혼합 | TAB-3 / TAB-B4 |
| N-F (sparsity 파생) | 026, 027 | FIG-3 sweep | `[신규 실행]` | FIG-3 |
| N-G (qualitative 파생) | 028 | FIG-4 제작 (=2) | `[재사용]` | FIG-4 |
| N-H (cost 파생) | **031** | TAB-B3 측정 (§5 "50×" sync) | `[신규 측정]` | TAB-B3 |

---

## 🕸️ 의존 그래프 — TAB-2가 루트다

TAB-2(main comparison)가 placeholder 의존 그래프의 루트다. NUM 13건(N-C 8 + N-D 6 중 다수), FIG-3의 floor, TAB-B1의 Δ 기준, appendix 결과 표 3종이 전부 이 표의 확정본에서 파생된다. 화살표는 "왼쪽이 끝나야 오른쪽을 채울 수 있다"로 읽는다.

```
[입력 4갈래]                          [루트]                [파생]
271canon 완주 (SMD6·SMAP49·MSL22) ─┐
baseline SMD/SMAP/MSL 신규 (#1) ───┤
weak 4종 GPU (#2) ─────────────────┼──▶ TAB-2 (Table 2) ──▶ NUM-006~013 (N-C 집계)
standard-split run (#3) ───────────┘         │
                                             ├──▶ 하단 protocol-effect 블록 ──▶ NUM-014~019 (N-D)
                                             │         └─(019의 contaminated 측은 TAB-B1 run과 소스 공유)
                                             ├──▶ FIG-3 점선 floor ──▶ NUM-026/027 (N-F; sweep run #7 별도)
                                             ├──▶ TAB-B1 Δ 기준값 (run #8과 결합)
                                             ├──▶ TAB-A6 / TAB-A7 (동일 실행 묶음 metadata에서 키 추가 추출)
                                             ├──▶ TAB-A8 (CSMAD per-entity; 블록 평균 = Table 2 셀 assert)
                                             └──▶ sync 그룹 A "six" (N-A) · sync 그룹 B "26" (N-B)

[독립 사슬]
TAB-3 대표 열 확정 (NUM-020) ──▶ TAB-B4 열 집합 동일 ──▶ NUM-021~025 (N-E)
                                  └─(run #4·#5·#6·#9가 행을 공급; depth2 = symmetric run 공유)
TAB-B3 측정 ──▶ NUM-031 (N-H) ──▶ §5 "approximately 50×" 표현 sync
TAB-1 SMD 측정 스크립트 ══ Table A.4 SMD 행 (동일 산출물 — 불일치 구조적 차단)
FIG-4 (NUM-028) · FIG-B1 좌패널: [271c]만으로 즉시 가능
TXT-001 ──▶ TAB-B3 하드웨어 표기 / TXT-002: 공개 checklist 후 3개소 일괄 치환
R-PROBE: [271c]만으로 기본형 가능 (확장 대조군만 run #4 의존; 원고 무변경)
```

**그래프가 주는 운영 결론 세 가지.**

1. 실행 #1–#3이 끝나기 전에는 N-C/N-D의 어떤 NUM도 본문에 선기입할 수 없다. PSM처럼 이미 산출된 값([271c] `metrics.pak_auc_f1`)이 있어도 **표 전체 확정 전 선기입 금지**가 규칙이다 (A8).
2. 학습이 필요 없는 작업(측정 5건 + 재사용 8건)은 의존 그래프 바깥이거나 말단이므로 **지금 병렬로 소화할 수 있다**.
3. 집계에서 Exathlon·Simulation은 절대 배제한다 (R33). 기존 Notion RankAvg류 수치는 Exathlon 포함 기준이므로 재계산 전까지 인용 금지다 (FEEDBACK-3).

---

## 📂 발행 안내

1. **페이지 구조**: 각 `<!-- PAGE: ... -->` 블록을 하위 페이지 1장으로 만들고, 본 OVERVIEW 블록은 부모 페이지 본문에 넣는다.
2. **발행 경로**: 긴 보고서는 반드시 notion-create-pages 경로로 발행한다 (마크다운→블록 파싱). update-page의 insert_content는 렌더링을 파손한다.
3. **조건 명칭**: R24 개명 후 표기만 사용한다 — 코드명 normalonly → "anomaly-excised condition", 코드명 full → "contaminated-training condition".
4. **공통 금지 규칙**: 수치 발명 금지(A8) · Gaussian smoothing 언급 금지(R34)는 전 페이지에 적용된다.

---

<!-- PAGE: FIG-1 -->

# FIG-1 — 학습 패러다임 비교 다이어그램 (Setting-comparison diagram)

> 💡 **한 줄 요약**: 동일한 오염 학습 스트림 위에서 unsupervised / label-aware filtering / CSMAD 세 패러다임이 라벨을 각각 "무시 / 절제 / 통합"하는 방식을 한 장으로 대비시킨 개념도. 논문 전체의 문제 설정과 핵심 용어를 시각적으로 고정한다.

| 유형 | 소스 분류 | 우선순위 | 의존성 |
|---|---|---|---|
| 본문 Figure (§1) | `[제작]` | 🟢 즉시 가능 — 무학습 제작 | 없음. contribution bullet 2 용어 동기화 의무만 |

| 항목 | 내용 |
|---|---|
| 위치 | §1 Introduction, observation 문단 직후 (`sec1_intro.tex`, `\label{fig:setting}`) |
| 크기 | full-width, 약 5 cm (≈0.40p) |
| 작업 성격 | 실험 데이터 없음 — 다이어그램 제작과 본문 대조 검증만 필요 |

## 🎯 목적과 의도

논문의 중심 논제는 "labeled anomaly는 비지도 방법에게는 오염이지만, 학습 신호로 통합할 수 있는 방법에게는 가치 있는 정보다"이다. 이 그림은 그 논제를 본문 텍스트보다 먼저, 텍스트 없이도 전달한다. §1의 관찰 문단 — labeled anomaly가 드러내는 세 가지 학습 신호 (a)/(b)/(c) — 직후에 배치되어, 독자는 contribution bullet을 읽기 전에 "기존 두 패러다임이 이 신호를 어떻게 버리는가"를 눈으로 먼저 확인하게 된다.

**논증 역할 ① — 중앙 패널은 가장 자연스러운 반문에 대한 시각적 선제 답변이다.** "라벨이 있으면 오염 구간을 걸러내면 되지 않는가"라는 reviewer 반문에 대해, 중앙 패널(label-aware filtering)은 본문 §1의 문장("the best a label-aware variant can do is exclude confirmed anomaly windows ... filtering contamination rather than learning from it")이 말하는 한계 — 오염은 제거되지만 라벨 정보 자체는 폐기된다 — 를 패널 하나로 보여준다. 이 한계는 본문 비교 실험의 절제 조건(anomaly-excised condition; Table 2의 main 조건)이 왜 "비지도 방법에게 라벨의 최선 활용을 제공하는 조건"인지(블루프린트 §14 논거 ③, R12 논리)와 직결된다.

**논증 역할 ② — 동일한 입력 띠 자체가 방어 장치다.** 세 패널 상단의 입력 스트림 띠를 **세 패널에서 동일하게** 그리는 것은, 블루프린트 §15의 leakage 공격 시나리오("test-prefix 편입은 test label로 학습하는 것")에 대한 방어 논거 ③ — 모든 비교 모델이 동일한 데이터를 받는다 — 를 비교 조건을 설명하기도 전에 그림의 전제로 깔아 두는 장치다. 우측 패널의 세 갈래 화살표는 contribution bullet 2의 세 용어(*anomaly-priority masking*, *loss bifurcation*, *gradient-reversal suppression*)를 글자 단위로 고정하는 anchor이며, 이후 §3·§4의 모든 서술이 이 세 명칭으로 수렴한다.

## 🏁 목표와 기대 결과

실험이 없는 제작물이므로 성공 기준은 전달력과 정합성으로 정의한다.

1. 비전문 독자가 캡션을 읽지 않고도 세 패널의 차이 — 라벨이 보이지 않음 / 라벨로 구간을 잘라냄 / 라벨이 세 경로로 학습에 흘러 들어감 — 를 읽을 수 있을 것.
2. 그림 내 모든 용어가 본문 표기와 글자 단위로 일치할 것 (⚠️ 절 참조).
3. 입력 스트림 띠의 붉은(anomaly) 구간 비율이 실제 train anomaly ratio(0.5–6.2%)를 연상시키는 소수 구간일 것 — 절반이 붉은 그림은 설정 자체를 왜곡한다.

개념도이므로 "기대와 다른 실험 결과"는 발생하지 않지만, 대응 규칙은 하나 있다. Phase 8 채움 과정에서 비교 조건 명칭이나 contribution bullet 2의 용어가 변경되면(예: R24류 개명 재발생) 이 그림을 본문과 **동시에** 갱신한다 — 그림만 구표기로 남는 상태는 허용되지 않는다.

## 🧪 실험 내용과 설계

**`[제작]` — 실험 소스 없음.** 학습·측정이 전혀 필요 없고, 다이어그램 제작과 본문 대조 검증만 수행한다.

- **권장 제작 경로**: TikZ 직접 작성(elsarticle 빌드와 폰트가 일치해 가장 안전) 또는 외부 벡터 도구로 제작 후 PDF 삽입.
- **공통 입력 띠**: 세 패널 상단에 동일한 입력 스트림 띠(정상 구간 + 붉은 라벨 anomaly 구간)를 그린다. 붉은 구간은 소수(시각적으로 수 % 수준)만 칠한다.
- **패널 구성**: (좌) unsupervised — 라벨이 모델에 보이지 않아 순수 오염원으로 작용. (중) label-aware filtering — 라벨된 anomaly window를 학습 전에 절제(= anomaly-excised condition; §4.1.4 상호참조). (우) CSMAD — 라벨이 masking·loss·gradient 세 경로로 학습에 유입.

## 📊 구성과 형태

가로 3-패널. 각 패널의 수직 구성은 동일하다: **상단 입력 스트림 띠 → 모델 박스 → 라벨 흐름 글리프**.

| 패널 | 라벨 흐름 글리프 | 핵심 시각 메시지 |
|---|---|---|
| 좌 (unsupervised) | 라벨이 모델에 닿지 않음 (무시됨 표시) | anomaly가 all-normal 가정의 오염원으로만 작용 |
| 중 (label-aware filtering) | 라벨이 데이터를 잘라내는 가위/절제 표시 | 오염 제거 = 라벨 정보 폐기 |
| 우 (CSMAD) | 라벨에서 세 갈래 화살표 → masking / loss / gradient | 오염을 학습 신호로 전환 |

용어는 §1 contribution bullet 2의 표기와 글자 단위로 일치시킨다: *anomaly-priority masking*, *loss bifurcation*, *gradient-reversal suppression*.

## 📝 캡션 (영문 확정본)

```latex
Three training paradigms for multivariate time series anomaly detection under a
contaminated training stream.
\textit{Left (unsupervised)}: labeled anomalies are invisible to the model and act purely
as contamination of the all-normal assumption.
\textit{Middle (label-aware filtering)}: labeled anomaly windows are excised before
unsupervised training (the anomaly-excised condition; \S\ref{sec:baselines}) ---
contamination is removed but the label information is discarded.
\textit{Right (CSMAD)}: labeled anomalies are integrated into training through three
paths --- anomaly-priority masking, loss bifurcation, and gradient-reversal suppression
--- turning contamination into a learning signal.
```

## ⚠️ 주의사항과 의존성

- 중앙 패널 명칭은 R24 개명 후의 **"anomaly-excised condition"만** 사용한다. 구표기(Q3, normalonly)는 그림·라벨 어디에도 금지.
- 붉은 구간 비율은 실제 train AR(0.5–6.2%)을 연상시키는 소수 구간만 — 설정 왜곡 금지.
- 세 패널의 입력 스트림 띠는 픽셀 단위로 동일해야 한다 (공정성 논거의 시각화).
- 그림 용어 ↔ 본문 bullet 2 ↔ §3 소절 제목의 3중 일치 검증을 제작 완료 시점에 1회 수행한다.

## 🔢 연결된 수치 placeholder

없음 — 이 그림에서 파생되는 NUM placeholder는 없다. 단, 용어 동기화 의무(contribution bullet 2의 3-path 명칭)는 위 주의사항대로 적용된다.

---

<!-- PAGE: FIG-2 -->

# FIG-2 — CSMAD 아키텍처 개요 (Architecture overview)

> 💡 **한 줄 요약**: 학습(좌)·추론(우) 2-패널로 CSMAD의 다섯 기능 블록, 라벨 유입 경로, gradient 차단 구조를 한 장에 담아 §3 전체의 지도 역할을 하는 아키텍처 다이어그램.

| 유형 | 소스 분류 | 우선순위 | 의존성 |
|---|---|---|---|
| 본문 Figure (§3.2) | `[제작]` | 🟢 즉시 가능 — 무학습 제작 | 구조 상수는 271_CONFIG_TRUTH r4 §VIII에서만 인용 |

| 항목 | 내용 |
|---|---|
| 위치 | §3.2 도입부 (`sec3_method.tex`, `\label{fig:architecture}`) |
| 크기 | full-width, 5 cm = 0.40p (integrator 가정; Phase 7에서 가독성 확인) |
| 작업 성격 | 다이어그램 제작 + 정본 상수 대조 검증 |

## 🎯 목적과 의도

§3의 다섯 소절(문제 정식화 → 마스킹 → 비대칭 디코더 → 라벨 유도 학습 → 채점)은 각각 한 component를 다룬다. 독자가 전체 데이터 흐름을 머리에 그리지 못하면 각 손실 항이 어디에 붙는지 길을 잃기 때문에, 이 그림이 §3.2 도입부에서 전체 지도를 제공한다: 입력 윈도가 패치로 갈라져 어느 블록을 거치는지, 네 가지 손실(L_recon, L_OD, L_FM, L_cls)이 어느 연결선에서 발생하는지, 추론 시에는 무엇이 꺼지는지를 좌우 패널 대비로 보여준다.

**방어 역할 ① — "GRL이 Student를(나아가 표현 전체를) 망가뜨리지 않는가"** (블루프린트 §15). Student의 latent 입력에 stop-gradient 기호(⊥)를 명시해, encoder가 Teacher의 재구성 목적만으로 학습되고 GRL gradient로부터 완전히 차단된다는 §3.2 본문 주장("The adversarial signal therefore cannot corrupt the normal-pattern representation")을 시각적으로 고정한다.

**방어 역할 ② — GRL 부착 지점의 모호성 차단** (블루프린트 ADV BLK-002 — 과거 리뷰에서 실제로 지적된 재발 지점). "Student decoder final-layer hidden states, **before output projection**"이라는 명시 라벨을 그림 안에 넣어, 부착 지점을 본문·부록·rebuttal이 공유하는 단일 사실로 만든다. 부수적으로 GRL 박스의 점선 + "training only" 표기는 "추론 시 라벨 미사용"이라는 문제 설정의 약속(§3.1)을 그림 차원에서 반복한다.

마지막으로 우측 추론 패널은 §3.6의 leave-one-out 채점(50패턴 batch-병렬, σ_i → a_t 평균 집계)을 묘사한다. 이로써 §5 결론의 비용 한계 서술(약 50× forward 연산)과 부록 비용 표(TAB-B3)가 가리키는 대상을 미리 정의한다.

## 🏁 목표와 기대 결과

성공 기준은 세 가지다.

1. 그림만 보고 §3의 기호(o^T_i, o^S_i, h^enc, σ_i, a_t)와 손실 연결이 본문 수식과 1:1로 대응될 것.
2. 필수 표기 3건(📊 절의 ⓐⓑⓒ)이 전부 들어 있을 것 — 특히 ⓒ(GRL 부착 지점 라벨)는 생략 시 리뷰 재발 지점이다.
3. 모든 구조 상수(d_model=512, nhead=8, encoder 4L / Teacher 3L / Student 2L, N=50, ρ=0.15 → |M|=8, L=500, patch size 10)가 271_CONFIG_TRUTH r4 §VIII과 일치할 것.

이 그림은 실험 결과를 담지 않으므로 결과 의존이 없다. 단 하나의 연동 규칙: 부록 ablation(TAB-B4)의 symmetric decoder run 결과에 따라 contribution bullet 3의 주장 강도가 하향될 수 있는데(Phase 6 규칙), 그 경우에도 이 그림의 구조 자체(3L/2L 비대칭)는 사실 서술이므로 변경 불필요 — 캡션·본문 문구만 조정 대상이다.

## 🧪 실험 내용과 설계

**`[제작]` — 실험 소스 없음.** 모든 구조 상수는 271_CONFIG_TRUTH r4 §VIII에서 그대로 가져온다. 정본 외 출처(코드 default, Notion 스냅샷, 발표자료)에서 수치를 가져오는 것은 금지다 — 과거 batch_size(512 vs 1024), d_model(dynamic vs 512 고정) 불일치 사고가 전부 비정본 인용에서 발생했다.

**좌패널(학습)에 담을 데이터 흐름**: 윈도(L=500) → N=50 패치 → anomaly-priority masking이 |M|=8 패치를 가림(anomaly 패치 우선) → visible 42패치만 encoder 통과 → 디코더 앞에서 mask token 삽입(Teacher/Student 별도 토큰) → 손실 연결선 4종: L_recon(Teacher 출력), L_OD·L_FM(Teacher↔Student, 정상 masked 패치만), L_cls(GRL classifier → window label).

**우패널(추론)**: GRL branch 비활성. leave-one-out masking 50패턴을 batch 차원으로 병렬 처리하고, per-patch score σ_i를 point score a_t로 평균 집계한다.

## 📊 구성과 형태

다섯 색상 블록: (1) Patch Embedding(linear), (2) 공유 Transformer Encoder(4L), (3) Teacher Decoder(3L — 진한 색·깊게), (4) Student Decoder(2L — 연한 색·얕게), (5) GRL + AnomalyClassifierHead.

**필수 표기 3건 (생략 금지)**:

| # | 표기 | 이유 |
|---|---|---|
| ⓐ | GRL 박스 = 점선 + "**training only**" 라벨 | 추론 시 라벨·GRL 미사용 약속의 시각화 |
| ⓑ | Student latent 입력에 stop-gradient 기호 ⊥ | encoder가 Teacher recon으로만 학습됨 — §15 GRL 방어 |
| ⓒ | GRL 부착 지점 명시 라벨: "Student decoder final-layer hidden states, **before output projection**" | ADV BLK-002 — 생략 시 리뷰 재발 지점 |

## 📝 캡션 (영문 확정본)

```latex
CSMAD architecture overview.
\textit{Left panel (training)}: the input window is split into $N$ patches;
anomaly-priority masking withholds $|M|$ patches (anomalous patches masked first).
Visible patches enter the shared Transformer encoder; mask tokens are inserted before
each decoder.
The Teacher decoder (darker, deeper) produces reconstructions $\{o^{\mathrm{T}}_i\}$;
the Student decoder (lighter, shallower) produces $\{o^{\mathrm{S}}_i\}$.
An AnomalyClassifierHead with gradient reversal (dashed box, labeled \textbf{training only})
is applied to the Student's final-layer hidden states before the output projection.
Loss connections: $L_{\mathrm{recon}}$ from Teacher outputs; $L_{\mathrm{OD}}$ and
$L_{\mathrm{FM}}$ between Teacher and Student on normal masked patches; $L_{\mathrm{cls}}$
from classifier head to window label.
The encoder receives no gradient from Student or GRL (stop-gradient $\perp$).
\textit{Right panel (inference)}: GRL branch inactive; leave-one-out masking patterns
batch-parallelized; per-patch scores $\sigma_i$ averaged to point-level scores $a_t$.
```

## ⚠️ 주의사항과 의존성

- 기호는 Table C.2 notation 정본을 따른다 — point score는 s_t가 아니라 **a_t** (v2-r3 정정 반영됨).
- 학습 좌패널에 warmup(0-based epoch 0–249 동안 Student 학습 경로 forward skip)을 그릴 의무는 없다. 그릴 경우 "frozen"이 아니라 "**forward skipped (training path)**"가 정확한 서술이다 (271_CONFIG_TRUTH r4 §VIII Training — 평가 경로는 full forward라는 구분 포함).
- λ를 그림에 표기할 경우 이중 구조(손실 가중 λ_GRL vs 반전 계수 λ_rev)를 단일 λ로 합쳐 쓰지 말 것. 표기가 번잡하면 그림에서는 생략하고 §3.4 본문에 위임하는 편이 안전하다.

## 🔢 연결된 수치 placeholder

없음 — 이 그림에서 파생되는 NUM placeholder는 없다. 구조 상수의 단일 원천은 271_CONFIG_TRUTH r4 §VIII이며, §4.1.2 Implementation Details의 동일 상수 서술과 일치해야 한다.

---
