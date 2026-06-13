---
phase: 8
agent: singlepage-reviewer
directives: [R3]
last_modified: 2026-06-13
target: paper/08_final_audit/NOTION_FINAL_SINGLEPAGE.md
fact_basis:
  - paper/08_final_audit/NOTION_PLACEHOLDER_SPECS.md (r2 — 정본)
  - paper/05_manuscript/PLACEHOLDER_REGISTRY.md (v3-r1)
  - paper/07_latex/sec4_experiments.tex / appendix_A.tex / appendix_B.tex / appendix_C.tex / sec1_intro.tex / sec3_method.tex (캡션 원문)
  - paper/01_research_understanding/{RESEARCH_SYNTHESIS,EXPERIMENT_PROTOCOL_TRUTH}.md (83.75% 등 파생 사실 provenance)
verdict: PUBLISH (조건부 — MINOR 1건 권고 수정; BLOCKER/MAJOR 0)
---

# Phase 8 단일 페이지 독립 검수 — NOTION_FINAL_SINGLEPAGE.md (r1)

발행 직전 최종 검증. 4축(사실 보존 / 실행성 / 한국어 / 렌더 안전) + placeholder 전수 + 캡션 글자 대조 + 기계 점검.
대상은 1,527행. 검수자 임무는 "깨뜨리는 것"이었으나, **깨지는 BLOCKER·MAJOR를 찾지 못했다.** 발견은 렌더 MINOR 1건 + 관찰 2건뿐.

---

## 판정 요약

| 항목 | 결과 |
|---|---|
| **종합 판정** | **PUBLISH (발행 가능)** — MINOR 1건만 권고 수정 |
| BLOCKER | **0** |
| MAJOR | **0** |
| MINOR | **1** (F-S1 — 표 셀 내 4중 백틱 코드펜스 표기, 렌더 리스크) |
| 관찰(OBS) | **2** (비차단) |
| 렌더 기계 점검 | **PASS** (토글 누수 0, 태그 균형, 펜스 균형, page/column 태그 0, H5/H6 0, raw `$` math 0) |

---

## 축 ① 사실·실행 지침 보존 (BLOCKER급 대상) — 위반 0

r2 spec(정본)과 단일 페이지를 핵심 4건 + 무작위 8건 정밀 대조. **사실 누락·왜곡·창작 수치 0건.**

### 핵심 4건 정밀 대조 (전부 일치)

- **TAB-2 (의존 그래프 루트)**: 27 method 행(7그룹)·7 데이터셋 열·2지표 정의, CSMAD `metrics.pak_auc_f1`/`metrics.vus_pr`(best epoch), SWaT 열 `A1A2_excl22` 독립 best-epoch(`excl22_pak_auc_f1`), SMD/SMAP/MSL macro 평균(28/54/27), 4갈래 실험 소스, protocol-effect 블록 `use_grl=True` 유지·`use_grl=False` 금지, 하단 블록 대표 3열(SWaT excl22/WaDi A1/PSM)만 채움 — r2 §2 TAB-2 전 항목 일치. "루트" 명시 5회, Exathlon/Simulation 배제(R33), weak 4종 미완 fallback("22 unsupervised" + 그룹 6 행 삭제) 보존.
- **TAB-B1 (contaminated-training)**: 22 baseline + CSMAD 참조, tex 확정 열 {SWaT excl22, PSM, SMD avg}×{F1, Δ}("tex가 우선"), Δ = contaminated − anomaly-excised(양수=contaminated 우세), `1_20260312_*` 구버전 폐기·전량 재실행, **캡션-표 불일치 함정**("and VUS-PR" vs F1/Δ stub) 보존, NUM-019 소스 공유 — r2 §2 TAB-B1 일치.
- **TAB-3 행2 (w/o GRL)**: "exp290은 no_fm+no_grl 복합 — 행2 정의와 불일치", 신규 항목 `use_grl=False` + `anomaly_loss_weight=0.0`, `grl_disable_anomaly_loss` 게이트 재활성 차단 논리, OD-exclusion 유지 — r2 F-2 인접 정정 포함 일치. (`anomaly_loss_weight=0.0` ×2, `grl_disable_anomaly_loss` ×1 모두 보존.)
- **FIG-3 (sparsity sweep)**: 재사용 메커니즘 2개(`NoisyLabelSlidingWindowDataset`, `apply_normal50_noise` `397-416` seed=123), `label_keep_ratio: float = 1.0`(키워드 전용·비트 동일), region 단위·미선택 region 라벨만 0(절제 아님), 8–12 run, p=1.0 [271c] 재사용, floor는 TAB-2 확정값에서만, `loss.py:293-302` 논거 — r2 §1 FIG-3 일치.

### 무작위 8건 (TAB-1·FIG-4·TAB-A6·TAB-A7·TAB-A8·TAB-B2·TAB-B4·ALG-C1) — 전부 일치

- 데이터셋 통계 전수(719,959 / 1,296,001 / 86,401·86,402 / dim 123 / 4.16 / 0.52%–6.20% / 29–36)가 r2·tex와 자리수까지 일치. `loaders.py:1152-1157` `//2` 분할 산식 보존.
- FIG-4 score 산식 `score = recon + scaled_disc/4.0`, `scaled_disc = disc × (recon_mean/disc_mean)`, PSM threshold 0.001744, scoring.py 단일 원천, Gaussian smoothing 금지(R34), NUM-028=2 — 일치.
- TAB-A6: full/excl22 독립 best-epoch, 0.944 vs 0.629 실측, **83.75%**(region 22 mass) — RESEARCH_SYNTHESIS §④·EXPERIMENT_PROTOCOL_TRUTH §⑥·appendix_A.tex:298·sec4:88에서 provenance 확인됨(35,900/42,864=0.83753). **창작 아님.**
- TAB-A7: `pa_0_f1`(oracle)·`pa_0_f1_ar` 부재(REQUEST-1 RESOLVED)·`affiliation_f1_ar`·R29 보존.
- TAB-A8: 109행(28+54+27), macro=Table 2 동일성(단방향 아닌 동일성) assert 권고, MSL 27 channels 보존.
- TAB-B2: exp298(`num_epochs=300, warmup=150`)·exp299(`num_epochs=200, warmup=100`) 실재, 결정 (i)/(ii), warmup 비례 축소 함정 보존.
- TAB-B4: 신규 3 run, NUM-024 load-bearing 최우선, depth2=symmetric 동일 run, exp285 단독 diff 확인 보존.
- ALG-C1: off-by-one(0-based 250 / 1-based) 규약, λ 이중구조(λ_GRL vs λ_rev), `trainer.py:1205-1207`·`loss.py:293-302`, OBS-1 τ식 연동 표기 보존.

### r2 정정 회귀 점검 (consolidation 과정에서 사실이 되돌아갔는지) — 회귀 0

- **F-3**: 캡션 "Teacher 2L\,/\,Student 2L" 정확(L1172) — "2L/2L" 회귀 없음.
- **F-4**: [CMP-Q3] = `6_20260526_*`(SWaT/WaDi/PSM만), SMD `3_20260312_*` 폐기, SMAP/MSL 미실행 — 정확. "STALE 재실행" 문구는 명시적 "~이 아니라" 정정 주석 안에서만 등장(회귀 아님).
- **F-5**: 큐 295/296/300–303(window/patch), 297(dynamic d_model), 298/299(epoch), masking_ratio override 0건 — 정확.
- **F-1**: R-PROBE 권고 실험 등재(§5 토글 + §0 측정 5건 + 커버리지 노트) — 정확.

**축 ① 발견: 0건.**

---

## 축 ② 영문 캡션 무변경 — 글자 대조 6건+ 전부 일치 (위반 0)

표본을 6건에서 **전 17개 figure/table + ALG**로 확대해 .tex `\caption{}` 원문과 글자 대조했다.

| placeholder | .tex 소스:line | 결과 |
|---|---|---|
| FIG-1 | sec1_intro.tex:48–57 | 일치 |
| FIG-2 | sec3_method.tex:69–83 | 일치 |
| FIG-3 | sec4_experiments.tex:444–451 | 일치 |
| FIG-4 | sec4_experiments.tex:489–499 | 일치 |
| TAB-1 | sec4_experiments.tex:33+ | 일치 |
| TAB-2 | sec4_experiments.tex:208–223 | 일치(`\textbf{Bold}`/`\underline{}` 포함) |
| TAB-3 | sec4_experiments.tex:344–348 | 일치 |
| TAB-A3 | appendix_A.tex:107–110 | 일치 |
| Table A.4 | appendix_A.tex:224–228 (`\textit{SMD per-machine rows pending.}` 포함) | 일치 |
| TAB-A6 | appendix_A.tex:320–323 | 일치 |
| TAB-A7 | appendix_A.tex:347–350 | 일치 |
| TAB-A8 | appendix_A.tex:375–378 | 일치 |
| TAB-B1 | appendix_B.tex:23–29 | 일치 |
| TAB-B2 | appendix_B.tex:64–67 | 일치 |
| TAB-B3 | appendix_B.tex:96–98 | 일치 |
| TAB-B4 | appendix_B.tex:162–166 (`Teacher 2L\,/\,Student 2L`) | 일치 |
| FIG-B1 | appendix_B.tex:131–134 | 일치 |
| ALG-C1 | appendix_C.tex:119 ("(pseudocode placeholder)" → "training procedure." 교체 안내) | 일치 |

캡션은 모두 `\caption{}` 본문(tikz placeholder 안내문이 아님)을 옮겼고, `$...$` LaTeX math는 전부 ```` ```latex ```` 코드펜스 안에 있어 Notion 렌더에서 깨지지 않는다. **축 ② 발견: 0건.**

---

## 축 ③ placeholder 전수 + 8차원 (누락 0)

기계 census 결과:

- **FIG 5/5**: FIG-1·2·3·4·B1 토글 존재.
- **TAB 12/12**: TAB-1·2·3 + A.4·A6·A7·A8 + B1·B2·B3·B4 + (보너스)TAB-A3 토글 존재. **TAB-4는 toggle 없음 — 의도된 흡수**(D-010 ①, TAB-2 하단 블록; 단일 페이지가 그 명세를 전부 포함, "TAB-4 흡수 기록" 노트로 차단).
- **ALG 1/1**: ALG-C1.
- **R-PROBE 1건**: §5 토글.
- **NUM 31/31**: NUM-001…031 전부 존재(누락 [], 기계 확인). 8그룹(N-A~N-H = 4+3+8+6+6+2+1+1=31) + 각 소속 placeholder 토글 🔢 절 동시 등장.
- **TXT 2종/4개소**: TXT-001(§A.1 ×1) + TXT-002(Abstract·§A.1·§5 ×3) 토글 2개.

**8차원 무손실**: 21개 토글 전부 {메타표·💡한 줄 요약·🎯목적·🏁목표·🧪실험·📊형태·📝캡션·⚠️주의·🔢연결} 완비(기계 확인 7/7 + 요약 + 메타). TXT-001/002·R-PROBE는 "📊 구성과 형태 / 📝 캡션"을 한 헤더로 병합했으나 두 차원 내용 모두 존재. **축 ③ 발견: 0건.**

---

## 축 ④ 실행성 (사용자 핵심 요구) — "토글만 읽고 실험 설계/그림 제작 가능한가" 5건 시뮬레이션

| placeholder | 시뮬레이션 결과 |
|---|---|
| **FIG-3** | 가능. 재사용 메커니즘 2개 file:line, 신설 config 키(`label_keep_ratio`), 조작 단위(region·라벨만 0), seed(123), 실행 매트릭스(2–3 데이터셋 × p 4점 = 8–12 run), 큐 형식(`queue_dedup_renumbered_v5.json`), 집계(best epoch `pak_auc_f1`/`excl22_pak_auc_f1`), floor 출처(TAB-2) — **축·데이터 조건·config 전부 명시.** 곧장 큐 작성 가능. |
| **TAB-3 행4 (w/o OD)** | 가능. `use_output_discrepancy=False` 단일 키, 자동 recon-only 메커니즘 file:line(`scoring.py:105-106`·`249-253`), 각주 의무 명시. |
| **TAB-B2 CSMAD 축소** | 가능. 결정 (i)/(ii) 양자택일 + 권고안(i), exp299 재사용 시 열 라벨 "reduced (200)" 수정·캡션 무수정 명시. |
| **FIG-4** | 가능. checkpoint 로드 → 동일 scoring 경로 → 3배열 추출(산식 포함) → threshold(`anomaly_ratio_threshold`) → 사건 선택(≥2 유형) → 열2 선택(NUM-028 확정). 도면 행·열·축·정규화·음영 규약 명시. |
| **FIG-B1** | 가능. 좌(c sweep, 재채점, best-epoch 고정 규칙) / 우(ρ sweep, 재학습, `masking_ratio` override, |M|=round(50×ρ)) 분리, 격자·축·기본값 마커·큐 미등재 경고 명시. |

빠진 정보 없음. 설득력도 reviewer 공격 ↔ 방어 논거(블루프린트 §14/§15·R10/R12·ADV BLK 등) 매핑이 각 🎯절에 명시돼 "왜 이 실험인가"가 reviewer 납득 논리로 읽힌다. **축 ④ 발견: 0건.**

---

## 축 ⑤ 한국어 품질 — 위반 0 (MAJOR 없음)

- 모호어 스캔: 적절히/적절한/잘 /등등/되어진/지는다 — **0건.**
- 번역투 스캔: 에 의해/으로 인해/통해서/에 대해서 — 0건. "함으로써" ×5는 자연스러운 격식 한국어(번역투 아님).
- 명사 나열·비문: 표본 정독(§0 대시보드, 각 🎯목적, callout) 결과 비문 없음. 문장은 "주장 → 근거 → 방어 대상"의 reviewer-설득 구조로 흐른다.

**축 ⑤ 발견: 0건.**

---

## 렌더 안전 (Notion-flavored md) — 기계 점검 PASS

python으로 발행본 파일 직접 스캔:

| 점검 | 결과 |
|---|---|
| ① toggle children 탭 들여쓰기 | **PASS** — 21개 토글 전부 children 탭 시작, **누수(토글 밖 샘) 0건** |
| ② `<callout>` 균형 | **PASS** — open 30 / close 30 + 내부 탭 들여쓰기 정상 |
| ③ 표 셀 블록 침입 | 1건 (아래 F-S1) — `<callout>`/`<page>` 침입은 0; 코드펜스 표기 1건 |
| ④ heading 5/6 | **PASS** — H5 0, H6 0 (토글 내부는 H4 `####`, Notion 정상) |
| ⑤ `<page>` 태그(단일 페이지 금지) | **PASS** — 0건 |
| ⑥ columns 태그 균형 | **PASS** — `<column>`/`<columns>` 0건 |
| ⑦ 코드펜스 fence 균형 | **PASS** — 40 fence-line = 20 pairs(```latex ×7 + ``` ×33), figure/table 토글당 정확히 1 caption pair |
| ⑧ 인라인 math 백틱 | **PASS** — 펜스 밖 raw `$` 0건 (모든 LaTeX math는 ```latex 펜스 내부) |
| `<table_of_contents/>` | 1건(정상, H1 직하) |

---

## 발견 (severity·위치·수정안)

### F-S1 [MINOR · 렌더] — 자가 점검 표 셀 내 4중 백틱 코드펜스 표기

- **위치**: L1521 — `§검증` 자가 점검 표의 셀: `| 영문 캡션 | 각 figure/table에 ```` ```latex ```` 또는 ```` ``` ```` 코드블록 | … |`
- **문제**: Notion-flavored markdown 표 셀은 4중 백틱으로 감싼 리터럴 삼중 백틱(```` ```latex ````)을 안정적으로 inline code로 파싱하지 못한다. 셀 안의 리터럴 ` ``` `가 코드펜스 opener로 오인되면 표 행 또는 그 이하 렌더가 깨질 수 있다. (펜스 균형 자체는 셀이 한 줄이라 깨지지 않으나, 표 셀 inline 렌더가 리스크.)
- **영향**: 본문 placeholder 내용이 아니라 **메타 자가 점검 표 1행**이라 정보 손실 위험은 낮음. 그러나 "발행 직전 렌더 안전" 기준에서는 제거가 안전.
- **수정안**: 해당 셀의 백틱 표기를 일반 텍스트로 치환 — 예: `각 figure/table에 latex 코드블록(또는 plain 코드블록)` 또는 `각 figure/table에 코드펜스(\`\`\`latex / \`\`\`) 1개 이상`을 단일 백틱 inline으로. 권고: 백틱을 모두 빼고 `각 figure/table에 latex(또는 plain) 코드블록 1개 이상`.

### OBS-1 [관찰 · 비차단] — 토글 내부 H4(`####`) 사용

- 21개 토글이 내부 소절을 `#### 🎯 목적과 의도` 등 H4로 구성. Notion에서 toggle children H4는 정상 렌더(H5/H6 아님). **조치 불요** — 명세의 "heading 5/6 사용" 금지 항목에 해당하지 않음을 확인 기록.

### OBS-2 [관찰 · 비차단] — TXT-002·R-PROBE 캡션 펜스 부재

- 기계 점검상 TXT-002·R-PROBE는 caption 코드펜스 0개. 이는 정상 — TXT-002는 TXT-001의 본문 문장을 공유("📝 캡션" 절에서 명시)하고, R-PROBE는 원고 무변경 항목으로 "해당 없음 — 확정 캡션 없음"을 명시. **조치 불요.**

---

## 결론

NOTION_FINAL_SINGLEPAGE.md는 **발행 가능(PUBLISH)**하다.

- **사실 보존**: 핵심 4건 + 무작위 8건 + r2 정정 4건 회귀 점검 전부 통과 — 누락·왜곡·창작 0건. 83.75% 등 enrichment 수치도 provenance 확인됨.
- **캡션**: 17개 figure/table + ALG 전부 .tex `\caption{}` 글자 대조 일치.
- **전수/8차원**: NUM 31/31, FIG 5, TAB 12(+보너스 A3), ALG 1, TXT 2종/4개소, R-PROBE 1 — 누락 토글·누락 차원 0. TAB-4 흡수는 의도된 설계.
- **실행성**: 무작위 5건 전부 "토글만 읽고 설계/제작 가능" 통과.
- **한국어**: 모호어·번역투·비문 0건. 설득 논리 정합.
- **렌더**: 토글 누수 0, 태그 균형, 펜스 균형 — 기계 PASS. **유일한 권고 수정은 L1521 표 셀의 4중 백틱(MINOR)** — 제거 후 발행 권장(미수정 시에도 본문 내용 손실 위험은 낮음).

**축별 발견 수**: ①사실 0 · ②캡션 0 · ③전수/차원 0 · ④실행성 0 · ⑤한국어 0 · 렌더 MINOR 1 + 관찰 2.
**렌더 안전**: 8개 기계 점검 중 7 PASS, 1 MINOR(표 셀 백틱). **발행 가능 — MINOR 1건 권고 수정 후.**
