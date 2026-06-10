---
phase: 3
agent: blueprint-reviser
directives: [R6]
revision: r3 (fixer — p3_rereview_redteam_r2.md R2-MIN-02(fallback 사다리 재정렬) + R2-MAJ-01(Table 3 행 5·7 conditional) 반영; fixlog: paper/99_reviews/p3_fixlog_r3.md)
last_modified: 2026-06-11
authority: |
  Phase 2 STRUCTURE_AND_FIGURE_PATTERNS.md §G.1 배분안 참조 기준.
  Elsevier elsarticle 1-column 본문 기준으로 조정 (학회 2-column과 다름).
  총 9.0p 목표, 8.5p 이상 채움. Table/Figure 크기는 넉넉하게 가정.
  모든 분량은 Phase 5 drafter의 통제 기준 — text 단락 단위로 슬랙 명시.
  **본 문서가 섹션별 분량 수치의 단일 정본 (ADV BLK-001)** — PAPER_BLUEPRINT.md §2는 본 문서 §1의 전사이며, 충돌 시 본 문서를 따른다.
---

# PAGE BUDGET — TSMAE Elsevier elsarticle 9p 배분

> **r2 변경 요약**: §4에 protocol-effect 보조분석(Table 4, RT BLOCKER-03) 추가 → §4 3.2→3.3p, §1 1.7→1.6p로 상쇄 (총 9.0p 유지). Table 2 열 구성 확정(데이터셋 × {PA%K-AUC F1, VUS-PR} — RT V3). Table 2 landscape 조판의 elsarticle 지원 여부 Phase 5 확인 플래그 (RT V1). Appendix §B.4(epoch-sensitivity placeholder)·§C.1 명칭 변경(d_model 고정 정정) 반영.

---

## 0. 기본 전제

### 0.1 단위 변환 가이드 (Phase 5 drafter용)

| 요소 | 1-column elsarticle 기준 |
|------|------------------------|
| 본문 텍스트 1페이지 | ~650–700 words (11pt, line-spacing 1.15, 상하여백 포함) |
| 수식 1개 (inline 제외, 독립 display) | ~2–3줄 = 0.04–0.06p |
| Figure (full-width, ~4-5cm 높이) | ~0.33p |
| Figure (full-width, ~6-7cm 높이) | ~0.5p |
| Table (booktabs, 10행×5열) | ~0.25p |
| Table (booktabs, 30행×8열) | ~0.5–0.7p |
| 섹션 헤더(§X.Y) | ~0.03p |
| Abstract (200 words) | ~0.3p |
| Keywords 1줄 | ~0.03p |

> **주의**: elsarticle은 기본 1-column 조판이므로 2-column 학회 논문 대비 같은 페이지에 텍스트 밀도가 낮다. Table이 full-width를 차지하므로 큰 테이블은 분량 계획에 적극 반영.

### 0.2 총 목표 및 슬랙

- 목표: **9.0p 이하, 8.5p 이상 채움** (0.5p 슬랙 허용)
- Front matter (abstract + keywords) 포함 기준: 총 ~9.3p — abstract/keywords는 본문 9p 카운트에서 제외하되, 실제 인쇄 시 약 0.3p 추가.
- Conclusion + Acknowledgments + References는 본문 9p 카운트 외 (Appendix도 별도).

---

## 1. 섹션별 분량 배분 (총 9.0p) — **단일 정본**

```
§1 Introduction          1.6p   (r2: 1.7 → 1.6)
§2 Related Work          1.1p
§3 Methodology           2.7p
§4 Experiments           3.3p   (r2: 3.2 → 3.3, protocol-effect 보조분석 추가)
§5 Conclusion            0.3p
─────────────────────────────
합계                     9.0p
```

---

## 2. 섹션별 세부 배분

### §1 Introduction — 1.6p 합계

| 구성 요소 | 예상 크기 | 배분 |
|---------|---------|------|
| Para 1 (문제 중요성) | ~4문장 | 0.20p |
| Para 2 (기존 방법 계보 + 한계) | ~5문장 | 0.25p |
| Para 3 (핵심 관찰 + 동기 + bridge 문장 + 프로토콜 에코 1문장) | ~5문장 | 0.22p |
| Fig. 1 (설정 비교 3-way 다이어그램, full-width, 약 5cm 높이) | 1 figure | 0.40p |
| Para 4 (제안 방법 + Contribution bullet 4개) | ~4문장 + 4 bullet | 0.35p |
| Para 5 (논문 구성) | ~1문장 | 0.05p |
| 섹션 헤더 × 1 | — | 0.05p |
| **합계** | | **1.52p** |

슬랙: 0.08p (r2: §1 목표 1.6p로 축소 — 슬랙 0.20→0.08; Para 1–2를 합쳐 0.42p 이내 유지).

> Drafter 지침: Fig. 1은 반드시 §1 Para 3 직후 또는 Para 4 앞에 배치. Para 1–2를 압축 — 기존 방법 계보 서술은 괄호 클러스터 인용으로 짧게.

---

### §2 Related Work — 1.1p 합계

| 구성 요소 | 예상 크기 | 배분 |
|---------|---------|------|
| §2.1 헤더 | — | 0.03p |
| §2.1 MTSAD 비지도 계보 (2–3 단락) | ~300 words | 0.45p |
| §2.2 헤더 | — | 0.03p |
| §2.2 Semi-supervised/PU (2–3 단락; weakly-supervised 1문장 + NRdetector 분리 + end-to-end 차별 1–2문장) | ~250 words | 0.35p |
| §2.3 헤더 | — | 0.03p |
| §2.3 MAE + Self-distillation (1–2 단락) | ~150 words | 0.22p |
| **합계** | | **1.11p** |

슬랙: −0.01p (초과 주의) — §2.1을 2단락으로 압축하거나, §2.3을 1단락으로 통합하면 1.0p 내외로 유지 가능.

> Drafter 지침: §2.2가 논문 포지셔닝 핵심 — 이 소절에서 0.35p를 사용해도 됨 (RT MAJOR-02의 end-to-end 차별 논리 1–2문장 포함). §2.1은 괄호 클러스터 인용으로 개별 모델 소개 없이 압축 (TFMAE는 §2.3 전속 — ADV MINOR-001). §2.3은 짧게 유지(각주 1개로 R21 용어계보+구조차이 방어; 작동 계층 차이는 §3.5 본문으로 이동 — 각주 비대화 방지).

---

### §3 Methodology — 2.7p 합계

| 구성 요소 | 예상 크기 | 배분 |
|---------|---------|------|
| §3.1 헤더 | — | 0.03p |
| §3.1 Problem Formulation + Notation (notation table or inline; 상한 케이스/sweep 3단 구조 1–2문장 포함) | ~160 words + 수식 3개 | 0.32p |
| §3.2 헤더 | — | 0.03p |
| §3.2 Overall Architecture 산문 | ~100 words | 0.15p |
| Fig. 2 (아키텍처 다이어그램, full-width, 약 6cm 높이; GRL "training only" 표기 포함) | 1 figure | 0.50p |
| §3.3 헤더 | — | 0.03p |
| §3.3 Patch Embedding and Masking (~200 words + 수식 3개) | | 0.33p |
| §3.4 헤더 | — | 0.03p |
| §3.4 Asymmetric Teacher–Student (~200 words + 수식 2개) | | 0.30p |
| §3.5 헤더 | — | 0.03p |
| §3.5 Label-Guided Training (~310 words + 수식 5개; SDMAE 계층구분 1문장 + GRL 필요성 논증 포함) | | 0.53p |
| §3.6 헤더 | — | 0.03p |
| §3.6 Anomaly Scoring (~150 words + 수식 3개; per-patch/집계 구분) | | 0.27p |
| **합계** | | **2.58p** |

슬랙: 0.12p — 수식 추가 또는 §3.5 GRL 상세화에 활용 가능.

> Drafter 지침: Fig. 2가 §3 분량의 ~19%를 차지 — figure 크기를 5cm로 줄이면 0.1p 절약 가능. §3.1 notation은 table 형태로 제시하면 compact함(Appendix §C.3에 위임도 가능). 수식은 inline이 아닌 display format으로 번호 부여 — 수식 13개 전후 예상 (per-patch score 2 + 집계 1 분리 — ADV MAJ-009).

---

### §4 Experiments — 3.3p 합계

| 구성 요소 | 예상 크기 | 배분 |
|---------|---------|------|
| §4.1 헤더 + §4.1.1 헤더 | — | 0.06p |
| §4.1.1 Datasets + Protocol (~230 words; 정면 방어 문단 + 한계 인정 + 선택 근거 1문장 포함) | | 0.32p |
| Table 1 (Dataset stats, 9행×7열, booktabs) | 1 table | 0.28p |
| §4.1.2 헤더 + Implementation (~170 words; epoch/batch 비대칭 공개 + test-set selection 공개 + SWaT 재현성 1줄) | | 0.27p |
| §4.1.3 헤더 + Metrics (~150 words + 지표 list) | | 0.22p |
| §4.1.4 헤더 + Baselines (~110 words; 양적 비대칭 인정 1문장 포함) | | 0.16p |
| §4.2 헤더 | — | 0.03p |
| Table 2 (Main results, ~27행 × [데이터셋 × {PA%K-AUC F1, VUS-PR}], large, sideways or fontsize 축소) | 1 large table | 0.70p |
| §4.2 분석 텍스트 (~180 words; protocol-effect 해석 포함) | | 0.25p |
| **Table 4 (Protocol-effect: standard vs contaminated, half-width 소형 — r2 신설, RT BLOCKER-03)** | 1 small table | 0.20p |
| §4.3 헤더 | — | 0.03p |
| Table 3 (Ablation, 8행×5열) | 1 table | 0.25p |
| §4.3 분석 텍스트 (~100 words) | | 0.15p |
| §4.4 헤더 | — | 0.03p |
| Fig. 3 (Label sparsity sweep, full-width, 약 4cm 높이) | 1 figure | 0.33p |
| §4.4 서술 (~100 words) | | 0.15p |
| §4.5 헤더 | — | 0.03p |
| Fig. 4 (Qualitative anomaly score, full-width, 약 4–5cm 높이) | 1 figure | 0.35p |
| §4.5 서술 (~80 words) | | 0.12p |
| **합계** | | **3.93p** |

슬랙: −0.63p (초과 — 압축 필수)

**압축 전략 (우선순위 순; 합계 ~0.65p)**:
1. Table 2를 landscape(sideways) + fontsize small로 조판하면 최대 0.2p 절약. **⚠️ Phase 5 확인 필수 (RT V1)**: elsarticle/대상 저널이 sideways table을 본문에 허용하는지 템플릿 수준에서 검증 — 미지원 시 **fallback 사다리 (우선순위 순 — r3 재정렬, R2-MIN-02)**: (a) fontsize \small + tabcolsep 축소 + 데이터셋 열 약어 → (b) 전략 2(Table 4의 Table 2 흡수) 병용 → (c) **최후 수단**: 지표 1열(PA%K-AUC F1)로 줄이고 VUS-PR을 Appendix §A.3 이동(−0.15p 추가) — 단 (c)는 RT V3 확정(열 구성 2지표 **고정**)을 재개방하고 "왜 이 지표만 쓰는가" 공격을 부활시키므로 **V3 재결정(orchestrator) 없이 적용 금지**.
2. Table 4를 Table 2의 하단 블록(조건 열 2개 추가 행 그룹)으로 흡수하면 0.15p 절약 — drafter 재량 (별도 표 유지가 가독성 우선).
3. §4.1.3 지표 서술을 compressed list 형태로 줄이면 0.05p 절약.
4. Fig. 4 높이를 3.5–4cm로 줄이면 0.05–0.1p 절약.
5. §4.2 분석 텍스트를 150 words로 줄이면 0.05p 절약 (component 서사는 §4.3 전속 — RT MINOR-04로 자연 단축).
6. §4.1.2 impl details 일부(optimizer 세부 등)를 Appendix §A.1로 위임하면 0.1p 절약.

**압축 후 예상**: 3.3–3.4p 범위 내 조정 가능 (전략 1+3+4+5+6 적용 시 ~3.43p; 전략 2 병용 시 3.28p).

> Drafter 지침: Table 2가 §4 분량의 약 21% 차지 — 이 테이블의 크기와 형태(landscape vs portrait)가 전체 분량을 크게 결정. 실험 수치가 확정된 후 실제 테이블 크기를 측정하여 조정. Table 2 열 구성은 **데이터셋 × {PA%K-AUC F1, VUS-PR} 고정** (RT V3 — "지면 허용 시 세분" 위임 폐기); 나머지 3지표는 Appendix §A.3. §4.5는 figure 크기로 분량 조절.

---

### §5 Conclusion — 0.3p 합계

| 구성 요소 | 예상 크기 | 배분 |
|---------|---------|------|
| §5 헤더 | — | 0.03p |
| 1단락 (요약 + 한계 + 향후연구; complementary masking "구현됐으나 미사용" 수식어 포함) | ~150 words | 0.27p |
| **합계** | | **0.30p** |

---

## 3. Figure/Table 전수 목록 및 크기 사양

| 번호 | 유형 | 내용 | 배치 섹션 | 예상 높이 | 예상 면적(p) |
|-----|------|------|---------|---------|-----------|
| Fig. 1 | Figure | 설정 비교 3-way (unsupervised Q1/Q3 vs [MODEL]) | §1 | ~5cm | 0.40p |
| Table 1 | Table | Dataset statistics (9행×7열; WaDi A1/A2 별도 행) | §4.1.1 | — | 0.28p |
| Fig. 2 | Figure | Architecture diagram (5 components; GRL=student hidden, output projection 이전 + "training only" 표기) | §3.2 | ~6cm | 0.50p |
| Table 2 | Table | Main results (~27행 × 데이터셋×{PA%K-AUC F1, VUS-PR}, sideways — Phase 5 템플릿 확인) | §4.2 | — | 0.70p |
| Table 3 | Table | Ablation (7–8행×5열; warmup 행(6)·FM 행(5)·symmetric decoder 행(7)은 실험 완료 시에만 — conditional, r3 R2-MAJ-01; 행 7은 contribution bullet 3 load-bearing — BLUEPRINT §0.4 Phase 5 진입 조건) | §4.3 | — | 0.25p |
| Table 4 | Table | Protocol-effect (standard split vs contaminated; [MODEL]+대표 baseline 2–3 × 데이터셋 2–3) — r2 신설 | §4.2 | — | 0.20p |
| Fig. 3 | Figure | Label sparsity sweep (X: p, Y: PA%K-AUC F1, 2–3 datasets) | §4.4 | ~4cm | 0.33p |
| Fig. 4 | Figure | Qualitative anomaly score (4행 × 2열, SWaT excl22 + WaDi/PSM) | §4.5 | ~4–5cm | 0.35p |

**Figure 합계**: 4개 = 1.58p
**Table 합계**: 4개 = 1.43p
**Figure + Table 합계**: 3.01p (전체 9p의 약 33%)

> Drafter 지침: Figure/Table이 전체의 ~33% 차지 — 나머지 67%(≈6.0p)가 텍스트+수식. Table 2가 분량에 가장 큰 영향 — 실제 실험 수치 확정 후 행/열 수 조정. Table 4는 압축 전략 2(Table 2 흡수)로 면적 절약 가능.

---

## 4. 앞부분(Abstract + Keywords) 분량

| 구성 요소 | 배분 |
|---------|------|
| Abstract (~180 words) | 0.26p |
| Keywords (7개, 1줄) | 0.03p |
| Highlights (5 bullet) | 0.10p |
| 합계 | 0.39p |

> 이 0.39p는 본문 9p 카운트 외 — 실제 제출 시 abstract page와 별도 처리. Elsevier elsarticle 기준으로 abstract/keywords는 첫 페이지에 위치하며 본문 페이지 카운트에서 별도 처리됨.

---

## 5. Appendix 분량 (본문 9p 카운트 외)

예상 Appendix 분량: ~4–5p

| 소절 | 내용 | 예상 크기 |
|-----|------|---------|
| §A.1 Baseline Hyperparameters | 22+4 baseline 전수 파라미터 table | 1.0–1.5p |
| §A.2 Q1 Condition Results | Q1 조건 비교 table | 0.5p |
| §A.3 Full Multi-Metric Results | VUS-ROC/Affiliation F1/PA%K-AUC PR 등 전수 (Table 2 미수록 3지표 포함) | 0.5–0.7p |
| §A.4 Per-Entity SMD/SMAP/MSL | 109 entity 전수 결과 | 0.5–1.0p |
| §A.5 SWaT Full vs Excl22 | dual eval 상세 | 0.2p |
| §B.1 Decoder Depth Ablation | 3L/2L/1L 비교 (+ warmup 변형이 main Table 3에서 빠질 경우 수용처) | 0.2p |
| §B.2 Parameter Sensitivity | score ratio, masking ratio | 0.3p |
| §B.3 Computational Cost | FLOPs/wall-clock/memory | 0.2p |
| §B.4 Epoch-Budget Sensitivity (r2 신설, optional placeholder) | baseline epoch budget 민감도 — ADV BLK-005 방어 보조 | 0.2p |
| §C.1 Input Dimensionality Table (r2 명칭 변경) | 데이터셋/entity별 F 전수 (SWaT 45, SMD 29–36, WaDi 123) + d_model=512 전 entity 공통 명기 — 구 "Dynamic d_model Mapping" 폐기 (271은 d_model 고정) | 0.1p |
| §C.2 Training Pseudocode | Algorithm block (+ SWaT constant-컬럼 제거 전처리 단계) | 0.2p |
| §C.3 Notation Summary | 기호 전수 표 | 0.1p |
| **합계** | | **~4.0–5.0p** |

---

## 6. 전체 페이지 예산 요약

| 구성 | 배분 |
|-----|------|
| §1 Introduction | 1.6p |
| §2 Related Work | 1.1p |
| §3 Methodology | 2.7p |
| §4 Experiments | 3.3p |
| §5 Conclusion | 0.3p |
| **본문 합계** | **9.0p** |
| Abstract/Keywords/Highlights | +0.39p |
| References | +~0.8p (25–35개 인용) |
| Appendix | +4.0–5.0p |

**실질 제출 볼륨**: 본문 9p + 약 5–6p(references+appendix) = ~14–15p 예상. Elsevier 저널 제출 규정에서 supplementary material로 처리 가능.

---

## 7. 분량 위험 요소 및 완화 전략

| 위험 | 원인 | 완화 |
|-----|------|------|
| §4 초과 | Table 2 크기, Table 4 신설, Fig. 4 높이 | §2 압축 전략 1–6 (Table 2 landscape — **elsarticle 지원 Phase 5 확인 필수, 미지원 시 fallback**; Table 4의 Table 2 흡수; Fig. 4 3.5–4cm) |
| §3 미달 | 수식이 예상보다 짧을 경우 | §3.5 GRL 상세화(AnomalyClassifierHead 구조 수식 추가) 또는 notation table §3.1에 포함 |
| §2 초과 | §2.2 PU/SSL 소절 확장 욕구 | §2.2를 3단락으로 제한; 상세 PU learning 설명은 BLUEPRINT §4.3 지침 준수 |
| 전체 초과 | 수식 번호가 많아 display 줄 증가 | 수식 중 일부(λ adaptive 공식 등) inline으로 대체, 핵심 12–13개만 독립 display |
| landscape 미지원 (r2 신설) | 일부 Elsevier 저널은 sideways table을 supplementary로 위임 | fallback 사다리(r3, R2-MIN-02): fontsize/tabcolsep/약어 → Table 4 흡수(전략 2) → 지표 1열화는 **최후 수단 + V3 재결정 필요** |

---

## 8. 단어수 환산 가이드 (Phase 5 drafter 분량 통제 기준)

| 섹션 | 텍스트 배분(p) | 수식+구조 공제 후 순 텍스트 | 권장 단어 수 |
|-----|------------|----------------------|-----------|
| §1 Introduction | 1.6p | ~1.0p (Fig. 1 공제) | 650–750 words |
| §2 Related Work | 1.1p | ~1.1p (figure 없음) | 700–780 words |
| §3 Methodology | 2.7p | ~1.6p (Fig. 2 + 수식 공제) | 1,000–1,100 words |
| §4 Experiments | 3.3p | ~1.2p (Tables+Figs 공제) | 780–870 words |
| §5 Conclusion | 0.3p | ~0.3p | 150–180 words |
| **합계** | **9.0p** | **~5.2p** | **3,280–3,680 words** |

> 이 단어 수는 본문 텍스트만 기준 — caption, 수식 내 기호, table 내 숫자는 제외. 실제 원고 Word count는 4,500–5,500 words 예상(caption, 수식 텍스트 포함 시).

---

## 9. 분량 체크포인트 (Phase 5 집필 중 사용)

Phase 5 drafter가 집필 중 분량을 검증하기 위한 체크포인트:

- §1 완료 후: 1.4–1.8p 범위 확인.
- §2 완료 후: 누적 2.4–3.0p 범위 확인.
- §3 완료 후: 누적 4.9–6.0p 범위 확인. Fig. 2 크기 확정.
- §4.1–4.2 완료 후: 누적 7.2–8.1p 범위 확인. Table 2/Table 4 크기 확정 (landscape 지원 여부 이 시점까지 해소).
- §4.3–4.5 + §5 완료 후: 총 8.5–9.0p 범위 확인.
- 초과 시: 먼저 §4 압축 전략 1–6 순서 적용 → §4.5 Fig. 4 높이 축소 → §4.2 분석 텍스트 단축 → §3.5 수식 일부 inline화 순으로 조정.
- 미달 시: §3.1 notation table 본문 내 추가 → §4.4 label sparsity 서술 확장 → §2.2 PU 계열 상세화 순.

---

## 부록: r3 정정 이력 (2026-06-11, fixer)

1. **[R2-MIN-02]** §2 압축 전략 1·§7 위험 표의 landscape-미지원 fallback을 우선순위 사다리로 재정렬 — 지표 1열화(VUS-PR §A.3 이동)는 RT V3 확정(2지표 고정)을 재개방하므로 **최후 수단 + V3 재결정 필요**로 격하 (fontsize/tabcolsep/약어 → Table 4 흡수 → 1열화 순).
2. **[R2-MAJ-01 파급]** §3 Table 3 행 conditional 표기를 행 6(warmup) 단독에서 **행 5(FM)·행 7(symmetric decoder) 포함**으로 확장 — 행 7은 contribution bullet 3 load-bearing (BLUEPRINT §0.4 r3 등재와 정합).

---

## 부록: r2 정정 이력 (2026-06-11, blueprint-reviser)

1. **[ADV BLK-001]** 본 문서를 분량 수치 단일 정본으로 선언 (frontmatter); BLUEPRINT §2가 본 문서 §1을 전사하도록 통일. §1 세부 합계(1.50→1.52p)·§3(2.55→2.58p) 미세 재계산.
2. **[RT BLOCKER-03]** §4에 Table 4(protocol-effect) + 분석 텍스트 반영 → §4 3.2→3.3p, §1 1.7→1.6p 상쇄 (총 9.0p 유지). 압축 전략 6개로 확장 + 압축 후 예상 재계산.
3. **[RT V1]** Table 2 landscape 조판의 elsarticle/저널 지원 여부를 Phase 5 확인 플래그로 명시 + fallback 전략 추가 (§2, §7).
4. **[RT V3]** Table 2 열 구성 확정: 데이터셋 × {PA%K-AUC F1, VUS-PR} — "지면 허용 시" drafter 위임 문구 폐기.
5. **[ADV BLK-005]** Appendix §B.4 Epoch-Budget Sensitivity (optional placeholder) 신설.
6. **[ADV BLK-003/r2 추가 발견]** §C.1을 "Input Dimensionality Table"로 교체 — 271은 d_model=512 전 entity 고정 (dynamic 매핑 폐기; 근거는 BLUEPRINT §5.4).
7. **[RT MAJOR-10/ADV MAJ-010]** Table 3 warmup 행 conditional 표기 (§3 전수 목록) — 미완료 시 §B.1 강등/생략.
