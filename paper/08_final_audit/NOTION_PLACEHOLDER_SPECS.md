---
phase: 8
agent: placeholder-spec-writer
directives: [R3]
last_modified: 2026-06-11
revision: r2 (p8 spec-fixer)
review_applied: |
  paper/99_reviews/p8_notion_spec_review_r1.md — F-1 BLOCKER (D-014 (b) R-PROBE 등재),
  F-2 MAJOR (TAB-3 행4 w/o OD 코드-사실 정정: scoring.py resolve_score_weights 직접 재확인),
  F-3/F-4/F-5 MINOR + OBS-1/OBS-2 전수 반영. 상세: §9 정정 이력.
registry_basis: paper/05_manuscript/PLACEHOLDER_REGISTRY.md (v3-r1)
truth_basis: |
  paper/01_research_understanding/EXPERIMENT_PROTOCOL_TRUTH.md (r4),
  paper/01_research_understanding/271_CONFIG_TRUTH.md (r4),
  paper/07_latex/sections/*.tex (캡션·배치 확정본),
  paper/99_reviews/p3_fixlog_r3.md §3 (EXPERIMENT_EXECUTION_TODO 집계),
  configs/queue_dedup_renumbered_v5.json + results/experiments/ 실측 (2026-06-11)
coverage: FIG 5 / body TAB 3 (+TAB-4 흡수 블록) / appendix TAB 8 (+Table A.4 부분) / ALG 1 / NUM 31 / TXT 2종 4개소 — REGISTRY v3-r1 전수
---

# NOTION PLACEHOLDER SPECS — placeholder별 실험·figure 명세 (Notion 발행 초안)

## 0. 개요 — 이 문서의 용도와 읽는 법

이 문서는 원고(MANUSCRIPT_v3 / LaTeX 확정본)에 남아 있는 **모든 placeholder**에 대해, "그 자리에 정확히 무엇이 들어가야 하고, 그 값을 얻으려면 어떤 실험을 어떤 설정으로 돌려야 하는가"를 재현 가능한 수준으로 명세한다. Notion 하위 페이지 발행 전 초안이며, 발행 시에는 Figure/Table별로 하위 페이지를 만들어 본 문서의 해당 절을 그대로 옮긴다 (R3).

**각 항목의 5요소** — 모든 Figure/Table/Algorithm 절은 다음 다섯 항목으로 구성된다.

1. **들어갈 내용**: 지표·값의 정의 (어느 조건, 어느 데이터셋, 어느 집계).
2. **실험 소스**: 기존 결과 재사용 가능 여부와, 신규 실행이 필요하면 config·스크립트 수준의 실행 지침.
3. **형태**: 시각화/표의 축·계열·비교 대상·강조 규칙.
4. **캡션**: LaTeX 확정본(`paper/07_latex/sections/*.tex`)의 영문 캡션 원문 그대로.
5. **주의사항**: placeholder 간 의존성, 집계 규칙, 함정.

**소스 분류 라벨** (모든 항목에 부여):

| 라벨 | 의미 |
|---|---|
| `[재사용]` | 기존 실험 결과/코드 상수에서 추출만 하면 됨 — 학습 불필요 |
| `[완주 대기]` | 진행 중인 271canon·큐의 잔여 entity 완주 후 집계 (SMD 6, SMAP 49, MSL 22 잔여 — 2026-06-11 실측) |
| `[신규 실행]` | 새 학습 실험 필요 (큐 등재 또는 신규 스크립트) |
| `[신규 측정]` | 학습 없이 측정/스크립트 1회 실행으로 해결 (통계 추출, 비용 측정 등) |
| `[제작]` | 실험 무관 — 다이어그램/의사코드 제작·검증만 필요 |

**공통 데이터 소스 경로** (이하 본문에서 약칭 사용):

- **[271c]** = `results/experiments/271_20260602_020545_271canon_baseline/` — CSMAD(271canon) 정본. entity별 `experiment_metadata.json`의 `metrics` dict(153키)가 모든 CSMAD 수치의 단일 원천. best epoch은 `timing.best_epoch` (기준 `pak_auc_f1`; SWaT excl22는 `excl22_pak_auc_f1`로 독립 선정).
- **[CMP-Q3]** = `comparison/results/experiments/6_20260526_085028_baseline_minmax_normalonly_segaware/` — unsupervised 22종의 anomaly-excised(코드명 normalonly/Q3) 조건 최신 결과. 단, **이 폴더에는 SWaT/WaDi/PSM만 존재** (2026-06-11 실측). SMD normalonly는 구버전 `3_20260312_*`뿐(per-entity 정규화(2026-06-02) 이전 — 폐기 대상)이고, **SMAP/MSL normalonly는 어느 결과 폴더에도 부재(미실행)** — 따라서 SMD는 구버전 폐기 후 재실행, SMAP/MSL은 미실행분 신규 실행이 필수다 (r2 정정 — "세 family STALE 재실행"이 아님).
- **[CMP-Q1]** = contaminated-training(코드명 full/Q1) 조건 — `comparison/run_baseline_queue.py`로 재실행 필요 (기존 `1_20260312_*`는 구버전).
- **실행 스크립트**: CSMAD 계열 = `scripts/run_base_experiments.py` + 큐 `configs/queue_dedup_renumbered_v5.json` (항목 형식: `exp_num` / `dataset` 리스트 / `config_override` 공백 구분 키=값); baseline 계열 = `comparison/run_baseline_queue.py --queue <json>`.
- **금지 사항 (전 항목 공통)**: 수치 발명 금지 (A8 — 실험 확정 전 본문 기입 금지), Gaussian smoothing 언급 금지 (R34), 조건 명칭은 R24 개명 후 표기만 사용 (코드 normalonly → "anomaly-excised condition", 코드 full → "contaminated-training condition").

---

## 1. Figures

### FIG-1 — Setting-comparison diagram `[제작]`

- **위치**: §1 Introduction, observation 문단 직후 (`sec1_intro.tex`, `\label{fig:setting}`). 크기: full-width, ~5 cm (≈0.40p).

**① 들어갈 내용** — 오염된 학습 스트림 하에서의 세 가지 학습 패러다임을 대비하는 3-패널 개념도. 실험 데이터가 들어가는 그림이 아니라 **설정(setting)을 시각적으로 정의하는 다이어그램**이다. 세 패널: (좌) unsupervised — 라벨이 모델에 보이지 않아 순수 오염원으로 작용, (중) label-aware filtering — 라벨된 anomaly window를 학습 전에 절제(= anomaly-excised condition; §4.1.4), (우) CSMAD — 라벨이 anomaly-priority masking · loss bifurcation · gradient-reversal suppression 세 경로로 학습에 통합.

**② 실험 소스** — 없음. 제작만 필요. 세 패널 상단의 입력 스트림 띠(정상 구간 + 붉은 라벨 anomaly 구간)는 **세 패널에서 동일하게** 그린다. 권장 제작 경로: TikZ 직접 작성(elsarticle 빌드와 폰트 일치) 또는 외부 벡터 도구 제작 후 PDF 삽입.

**③ 형태** — 가로 3-패널. 각 패널: 상단 입력 스트림 띠 → 모델 박스 → 라벨 흐름 글리프(무시됨 / 절제됨 / 세 갈래 화살표가 masking·loss·gradient로 유입). 용어는 §1 contribution bullet 2의 표기와 글자 단위로 일치시킬 것: *anomaly-priority masking*, *loss bifurcation*, *gradient-reversal suppression*.

**④ 캡션 (확정본, sec1_intro.tex)**:
> "Three training paradigms for multivariate time series anomaly detection under a contaminated training stream. *Left (unsupervised)*: labeled anomalies are invisible to the model and act purely as contamination of the all-normal assumption. *Middle (label-aware filtering)*: labeled anomaly windows are excised before unsupervised training (the anomaly-excised condition; §\ref{sec:baselines}) --- contamination is removed but the label information is discarded. *Right (CSMAD)*: labeled anomalies are integrated into training through three paths --- anomaly-priority masking, loss bifurcation, and gradient-reversal suppression --- turning contamination into a learning signal."

**⑤ 주의** — 중앙 패널 명칭은 R24 개명 후의 "anomaly-excised condition"만 사용 (구표기 Q3/normalonly 금지). 붉은 구간의 비율은 실제 train AR(0.5–6.2%)을 연상시키도록 소수 구간만 칠할 것 — 절반이 붉은 그림은 설정 왜곡이다.

---

### FIG-2 — CSMAD architecture overview `[제작]`

- **위치**: §3.2 도입부 (`sec3_method.tex`, `\label{fig:architecture}`). 크기: full-width, 5 cm = 0.40p (integrator 가정; Phase 7에서 가독성 확인).

**① 들어갈 내용** — 학습(좌)/추론(우) 2-패널 아키텍처 개요. 다섯 색상 블록: (1) Patch Embedding(linear), (2) 공유 Transformer Encoder(4L), (3) Teacher Decoder(3L, 진한 색·깊게), (4) Student Decoder(2L, 연한 색·얕게), (5) GRL + AnomalyClassifierHead. 좌패널: 윈도(L=500) → N=50 패치, anomaly-priority masking이 |M|=8 패치를 가림(anomaly 우선), visible 42패치만 encoder 통과, 디코더 앞에서 mask token 삽입(teacher/student 별도 토큰), 손실 연결선 L_recon(Teacher), L_OD·L_FM(Teacher↔Student, 정상 masked 패치), L_cls(classifier→window label). 우패널: GRL 비활성, leave-one-out masking 50패턴 batch-병렬, per-patch score σ_i → point score a_t 평균 집계.

**② 실험 소스** — 없음. 모든 구조 상수는 271_CONFIG_TRUTH r4 §VIII에서 그대로 가져온다 (d_model=512, nhead=8, 4/3/2 layers, ρ=0.15→|M|=8, N=50). 수치를 그림에 쓸 때 이 정본 외 출처 금지.

**③ 형태** — 필수 표기 3건: ⓐ GRL 박스는 점선 + "**training only**" 라벨, ⓑ Student latent 입력에 stop-gradient 기호 ⊥ (encoder는 Teacher recon으로만 학습됨을 시각화), ⓒ **GRL 부착 지점을 명시 라벨**: "Student decoder final-layer hidden states, **before output projection**" (v2-r3 / 블루프린트 ADV BLK-002 — 생략 시 리뷰 재발 지점).

**④ 캡션 (확정본, sec3_method.tex)**:
> "CSMAD architecture overview. *Left panel (training)*: the input window is split into $N$ patches; anomaly-priority masking withholds $|M|$ patches (anomalous patches masked first). Visible patches enter the shared Transformer encoder; mask tokens are inserted before each decoder. The Teacher decoder (darker, deeper) produces reconstructions $\{o^{\mathrm{T}}_i\}$; the Student decoder (lighter, shallower) produces $\{o^{\mathrm{S}}_i\}$. An AnomalyClassifierHead with gradient reversal (dashed box, labeled **training only**) is applied to the Student's final-layer hidden states before the output projection. Loss connections: $L_{\mathrm{recon}}$ from Teacher outputs; $L_{\mathrm{OD}}$ and $L_{\mathrm{FM}}$ between Teacher and Student on normal masked patches; $L_{\mathrm{cls}}$ from classifier head to window label. The encoder receives no gradient from Student or GRL (stop-gradient $\perp$). *Right panel (inference)*: GRL branch inactive; leave-one-out masking patterns batch-parallelized; per-patch scores $\sigma_i$ averaged to point-level scores $a_t$."

**⑤ 주의** — 기호는 Table C.2 notation 정본을 따른다 (point score는 s_t가 아니라 **a_t** — v2-r3 정정 반영됨). 학습 좌패널에 warmup(epoch<250 student forward skip)을 그려 넣을 필요는 없으나, 그릴 경우 "frozen"이 아니라 "forward skipped (training path)"가 정확한 서술이다 (271_CONFIG_TRUTH r4 §VIII Training).

---

### FIG-3 — Label sparsity sweep `[신규 실행]` ★ 미구현 실험 (R32)

- **위치**: §4.4 Results 문단 직후 (`sec4_experiments.tex`, `\label{fig:sparsity}`). 크기: ~4 cm ≈ 0.33p.

**① 들어갈 내용** — 라벨된 anomaly region 비율 p ∈ {0.1, 0.25, 0.5, 0.75, 1.0}에 대한 CSMAD의 PA%K-AUC F1 곡선 (대표 데이터셋별 1개 실선). 데이터셋별로 점선 수평선 = 해당 데이터셋의 **best unsupervised baseline (anomaly-excised condition, main protocol)** 성능 — Table 2 완성본에서 그대로 가져오는 "unsupervised floor". p=1.0 점은 main 설정과 동일하므로 **[271c]의 해당 entity 값과 일치해야 한다** (별도 재학습 불필요 — 그 점만 재사용).

**② 실험 소스 — 신규 실행 (전용 파라미터·스크립트 현재 부재: `label_ratio`/`sparsity` grep 0건, EXPERIMENT_PROTOCOL_TRUTH §⑦ 실측)**. 구현·실행 지침:

- **재사용할 기존 메커니즘 2개** (구현 기반 — 새로 발명하지 말 것):
  1. `mae_anomaly/datasets/noisy.py` `NoisyLabelSlidingWindowDataset` — 학습 split에서만 noisy 라벨을 반환하고 평가에는 원본 라벨 사용 (`use_noisy_labels = (split=='train')`). 희소화를 "학습 입력에만" 주입하는 정확한 구조가 이미 있다.
  2. `scripts/run_base_experiments.py:397-416` `apply_normal50_noise` — train 구간 anomaly **region 단위** 50% 무작위 재라벨(seed=123)의 기존 구현. 이것을 비율 p 파라미터로 일반화한 `apply_label_sparsity(regions, p, seed)`를 만들고, config에 `label_keep_ratio: float = 1.0` (키워드 전용, 기본 1.0 = 현행과 비트 동일) 추가.
- **조작 단위**: region 단위 무작위 선택 (점 단위 아님 — "기록된 fault 사건" 개념과 일치, 원고 §4.4 Design 문단과 합치). 미선택 region은 **데이터는 그대로 두고 라벨만 0** (절제 아님). seed 고정(region 선택 seed=123 계열, p별 동일 seed).
- **라벨 영향 경로 확인**: force_mask_anomaly, GRL classifier target, OD 정상/이상 분기 — 전부 `point_labels` 경유이므로 NoisyLabel 주입 한 곳으로 일괄 제어된다 (EXPERIMENT_PROTOCOL_TRUTH §⑦).
- **실행 매트릭스**: 대표 데이터셋 2–3개 (NUM-026; 권장 SWaT excl22 + PSM, 여유 시 WaDi A1) × p ∈ {0.75, 0.5, 0.25, 0.1} = 8–12 run. **p=1.0은 [271c] 재사용**. 각 run은 271 canon config 그대로 (500 epochs, seed 42), `config_override`에 `label_keep_ratio=<p>`만 추가한 큐 항목으로 등재 (`configs/queue_dedup_renumbered_v5.json` 형식).
- 그 외 모든 것(분할·정규화·평가·best-epoch 기준) 불변 — 변경은 학습 라벨뿐.

**③ 형태** — X축: labeled fraction p (0.1→1.0, 선형). Y축: PA%K-AUC F1. 데이터셋별 실선 1개 + 같은 색 점선(floor) 1개. 범례에 데이터셋명. p=1.0 main 설정 점 표시(마커 강조 가능).

**④ 캡션 (확정본, sec4_experiments.tex — [N]은 NUM-026 확정 후 치환)**:
> "Label sparsity sweep. PA\%K-AUC F1 as a function of the labeled anomaly fraction $p \in \{0.1, 0.25, 0.5, 0.75, 1.0\}$ for [N] representative datasets (one line per dataset). Dashed horizontal lines indicate the performance of the best unsupervised baseline (anomaly-excised condition, main protocol) on the corresponding dataset, providing the unsupervised floor. $p = 1.0$ corresponds to the main experimental setting; $p \to 0$ approximates the fully unsupervised limit."

**⑤ 주의** — ⓐ NUM-026(데이터셋 수)·NUM-027(열화 형상 서술어)이 이 실험에서 파생 — 같은 소스. ⓑ 점선 floor는 Table 2 확정값에서만 가져온다 (TAB-2 의존성 — SMD/SMAP/MSL baseline 신규 실행(§7.3 #1) 완료 전 floor 확정 불가). ⓒ p→0 극한은 Table 2 protocol-effect 블록의 "CSMAD (clean)"과 **다른 조건**이다 — clean-split은 prefix 자체가 train에 없는 반면 p=0은 비라벨 anomaly가 train에 남는다. 본문 상호참조 시 "approximates"라는 표현 유지, 동일시 금지. ⓓ §4.4 "Why graceful degradation is expected" 문단의 구조 논거(배치에 positive 없으면 GRL 손실 자체 미계산 — `loss.py:293-302`)와 결과 해석의 일관성 확인.

---

### FIG-4 — Qualitative score decomposition `[재사용]` + 추출 스크립트

- **위치**: §4.5 lead 직후 (`sec4_experiments.tex`, `\label{fig:decomp}`). 크기: full-width, 3.5–4 cm ≈ 0.30p.

**① 들어갈 내용** — 대표 anomaly 사건 구간에 대한 CSMAD score의 성분 분해 시각화. 2열(데이터셋) × 4행: 행1 입력(첫 feature) + GT anomaly 붉은 음영, 행2 Teacher 재구성 오차(per timestep), 행3 Teacher–Student discrepancy(adaptive 스케일 적용 후, per timestep), 행4 합산 anomaly score + anomaly-ratio threshold 점선. 데이터셋: 열1 = SWaT excl22, 열2 = WaDi A1 또는 PSM 중 시각적 변별이 좋은 쪽 (결과 확인 후 선택; 개수 = NUM-028 = 2).

**② 실험 소스 — [271c] 완주분 재사용, 신규 학습 불필요.** 추출 지침:

- 해당 entity의 best checkpoint를 로드해 evaluator의 **동일 scoring 경로**로 per-timestep 배열 3종을 추출: `recon`(Teacher MSE), `scaled_disc = disc × (recon_mean/disc_mean)`, `score = recon + scaled_disc/4.0` (정본 산식: 271_CONFIG_TRUTH §VIII Anomaly Score; 구현 단일 원천 `mae_anomaly/scoring.py` — **다른 곳에 식 복제 금지**, CLAUDE.md API 체크리스트 3항).
- threshold 점선 값은 해당 entity metadata의 `metrics.anomaly_ratio_threshold` 그대로 (예: [271c] PSM 0.001744).
- 사건 구간 선택: excl22 마스킹 후 남는 13개 소형 사건 중 유형이 다른 사건 ≥2개를 포함하도록 (RT MINOR-02 — 사건 규모·유형 대표성 확인). 구간 폭은 사건 길이의 3–5배 컨텍스트 포함 권장.

**③ 형태** — 열 내 4행 X축 공유(timestep), 행별 Y 정규화(per-trace normalized). 행4에만 점선 threshold. GT 음영은 4행 전체에 연하게 관통시켜 정렬 확인 가능하게.

**④ 캡션 (확정본, sec4_experiments.tex — [Dataset-A/B]는 선택 확정 후 치환)**:
> "Qualitative score decomposition on representative anomaly events. Each column corresponds to one dataset ([Dataset-A], [Dataset-B]). Row 1: multivariate input (first feature shown) with ground-truth anomaly regions shaded in red. Row 2: Teacher reconstruction error per timestep. Row 3: Teacher--Student discrepancy per timestep (adaptively scaled). Row 4: combined anomaly score with the anomaly-ratio threshold (dashed horizontal line). The decomposition illustrates how the two score components respond differently to anomaly characteristics: reconstruction error captures deviations from the learned normal pattern regardless of event type, while discrepancy captures structural divergence amplified by the capacity gap and label-guided training."

**⑤ 주의** — ⓐ **Gaussian smoothing 절대 금지** (R34) — [271c] 저장 점수는 전부 비평활이므로 추출값을 그대로 그리면 자동 준수되나, 시각화 코드에서 후처리 smoothing을 넣지 말 것. ⓑ §4.5 해석 문장("two components respond distinctly…")은 실제 그림 확정 후 사건별 관찰에 맞게 재검토 (RT MINOR-02 — 수치/관찰 확정 전 해석 강화 금지). ⓒ NUM-028이 이 그림에서 파생.

---

### FIG-B1 — Parameter sensitivity `[재사용(좌패널)] + [신규 실행(우패널)]`

- **위치**: Appendix §B.4 (`appendix_B.tex`, `\label{fig:param_sensitivity}`). 크기: ~3.5 cm ≈ 0.30p.

**① 들어갈 내용** — 2-패널 민감도 곡선. (좌) score 결합비 c (= `score_recon_disc_ratio`, 기본 4) 변화에 따른 PA%K-AUC F1; (우) masking ratio ρ (기본 0.15) 변화에 따른 PA%K-AUC F1. 대표 데이터셋별 1선.

**② 실험 소스 — 두 패널의 비용이 본질적으로 다르다**:

- **좌패널 (c sweep) `[재사용 + 재채점]`**: c는 **추론 시에만** 점수식에 들어간다 (`scoring.py` — score = recon + scaled_disc/c). 따라서 **재학습 불필요** — [271c] best checkpoint(또는 저장된 per-patch score 성분)에 대해 c ∈ {1, 2, 4, 8, 16} (log2 격자, 기본 4 중심)로 재채점→재평가만 수행. 기존 eval-recompute 도구 경로(2026-06 eval-recompute 툴링) 재사용 가능. 대표 데이터셋은 FIG-3과 동일 선택 권장.
- **우패널 (ρ sweep) `[신규 실행]`**: ρ는 학습 마스킹을 바꾸므로 **ρ별 전체 재학습 필요**. 권장 격자 ρ ∈ {0.05, 0.10, 0.15, 0.20, 0.30} (기본 0.15는 [271c] 재사용 → 신규 4 run × 대표 2–3 데이터셋). 큐 항목 `config_override`: `masking_ratio=<ρ>`만 변경, 그 외 271 canon 동일. 주의: ρ 변경 시 |M| = round(50×ρ)로 자동 변동 (0.05→2, 0.30→15패치).

**③ 형태** — 좌: X=c(log scale 권장), 우: X=ρ(선형). Y 공통: PA%K-AUC F1. 패널별 데이터셋 1선씩, 기본값 위치(c=4, ρ=0.15)에 수직 참조선 또는 마커.

**④ 캡션 (확정본, appendix_B.tex)**:
> "Parameter sensitivity. PA\%K-AUC F1 as a function of (*left*) the score combination ratio $c$ around its default 4 and (*right*) the masking ratio $\rho$ around its default 0.15, on representative datasets; all other settings fixed to the main configuration."

**⑤ 주의** — ⓐ 기호는 ρ (구표기 r_m 금지 — v2-r3 M-5). ⓑ c sweep 재채점 시 best-epoch을 c별로 재선정하지 말 것 — **main run의 best epoch 고정** 후 c만 바꿔야 "그 설정 주변의 민감도"가 된다 (재선정하면 test-set selection이 c에도 적용되어 별개 실험이 됨; 본문 한 줄로 고정 방식 명시 권장). ⓒ 우패널은 큐 미등재 — 신규 등재 필요 (기존 큐 295–303 중 295/296/300–303은 window/patch 크기 sweep, 297은 dynamic d_model, 298/299는 epoch-budget 변형 — masking-ratio sweep 항목은 없음; 큐 v5 전 32항목 `masking_ratio` override 0건 실측, r2 범위 정정).

---

## 2. Body Tables

### TAB-1 — Dataset statistics (Table 1) `[재사용 + 신규 측정(SMD 셀)]`

- **위치**: §4.1.1 (`sec4_experiments.tex`, `\label{tab:datasets}`). ~0.25p.

**① 들어갈 내용** — 6 family 행 × {#Train, #Test, #Dim., Train AR, Test AR} 열. **대부분 실값으로 이미 확정** (EXPERIMENT_PROTOCOL_TRUTH §① 실측: SWaT 719,959/224,960/45/1.63/19.05·3.68†, WaDi 1,296,001·870,972/86,401·86,402/123/0.52·0.76/3.82·3.87, PSM 176,401/43,921/25/6.20/30.63, SMAP 355,905/217,925/25/0.70/24.54, MSL 95,271/36,775/55/1.70/16.72). 잔여 placeholder는 **SMD 행의 per-machine 위임 셀**: 본문 표는 "per-machine (§A.3)" 포인터를 유지하되 Test AR 평균 4.16은 실값.

**② 실험 소스 — `[신규 측정]` (학습 불필요)**: SMD 28개 machine 각각의 #Train/#Test/Train AR을 산출하는 1회성 스크립트. 산출 규칙은 코드와 동일하게: `loaders.py:1152-1157` 분할(`test_split = len(test_data)//2`; train = orig train 전체 + test 앞 50%, test = 뒤 50%)을 그대로 호출하거나 같은 산식으로 라벨 파일에서 직접 계산. 결과는 Table A.4(SMD per-machine 행)와 §4.1.1 본문의 "SMD per-machine values pending" 문구 해소에 함께 쓴다 — **TAB-1과 Table A.4는 동일 소스 산출물**.

**③ 형태** — booktabs 6행. SWaT Test AR은 dagger(†)로 full/excl22 병기 (캡션에 정의). 변경 불필요 — 형태는 tex 확정.

**④ 캡션 (확정본, sec4_experiments.tex)**:
> "Dataset statistics under the contaminated benchmark protocol, summarized per family. Train/test sizes reflect the re-split described in §\ref{sec:datasets}. Train AR = anomaly ratio (\%) in the training portion (originating from the incorporated test prefix); Test AR = anomaly ratio (\%) in the held-out evaluation portion. The WaDi row aggregates the two independent entities A1/A2 (values given as A1\,/\,A2); SMD, SMAP, and MSL values are per-entity averages or concatenated totals as indicated. SWaT is evaluated under both full and excl22 conditions ($\dagger$: full\,/\,excl22); Table~\ref{tab:main_results} uses excl22 (§\ref{sec:datasets}). Per-entity statistics are in \ref{sec:appendix_dataset} (Table~\ref{tab:per_entity})."

**⑤ 주의** — SMD per-machine Train AR이 확정되면 §4.1.1 본문 "Training anomaly ratios range from 0.52% to 6.20% (SMD per-machine values pending…)" 문장의 범위 수치를 같은 pass에서 갱신 (SMD 값이 0.52–6.20 범위를 벗어나면 범위 자체 수정). #Dim 열은 §4.1.1이 단일 원천 (C.2 Table C.1과 정합 유지).

---

### TAB-2 — Main comparison results + protocol-effect 블록 (Table 2) `[완주 대기 + 신규 실행]` ★ 본 논문의 중심 표

- **위치**: §4.2 (`sec4_experiments.tex`, `\label{tab:main_results}`). table* 2단 폭, ≈0.55p. TAB-4는 이 표 하단 블록으로 흡수 완료 (D-010 ① — 별도 표 없음).

**① 들어갈 내용** — 27 method 행(7개 그룹) × 7 데이터셋 열 {SWaT excl22, WaDi A1, WaDi A2, PSM, SMD avg, SMAP avg, MSL avg} × 2지표 {PA%K-AUC F1, VUS-PR} + 하단 protocol-effect 블록. 셀 값 정의:

- **CSMAD 행**: [271c] entity별 metadata `metrics.pak_auc_f1` / `metrics.vus_pr` (best epoch 기준 — 전 지표가 같은 best epoch에서 추출됨). SWaT 열은 `SWaT/A1A2_excl22` entity (독립 best-epoch, `timing.best_epoch_metric='excl22_pak_auc_f1'`). SMD/SMAP/MSL avg = **entity별 best-epoch 지표의 macro 평균** (28/54/27 entity).
- **unsupervised 22행**: anomaly-excised condition([CMP-Q3]) 동일 키. random 행만 5-run mean (±std는 본문 비표기, §A.1에 명시).
- **weakly supervised 4행**: contaminated-training condition 단독 (구조적으로 excised 불가 — §4.1.4).
- **하단 protocol-effect 블록**: CSMAD(clean) + 대표 baseline 2–3종(NUM-014)의 standard clean-train split 결과 — 대표 열(SWaT excl22, WaDi A1, PSM — tex stub 기준)만 채우고 나머지 "—".

**② 실험 소스 — 4갈래**:

1. **CSMAD `[완주 대기]`**: 271canon 잔여 entity 완주 (SMD 6, SMAP 49, MSL 22 — 큐 진행 중). 완주 후 metadata 집계 스크립트로 macro 평균 산출. **부분 완주 상태로 avg 열을 채우지 말 것** — sync 그룹 A("six families")가 깨진다.
2. **unsupervised 22종 `[신규 실행(부분)]`**: SWaT/WaDi/PSM은 [CMP-Q3] 재사용 가능; **SMD/SMAP/MSL은 `comparison/run_baseline_queue.py`로 전 entity 신규 실행 필수** — SMD normalonly는 구버전 `3_20260312_*`(per-entity 정규화(2026-06-02) 이전)뿐이라 폐기 대상이고, SMAP/MSL normalonly는 어느 결과 폴더에도 부재(미실행)다 ([N-COMP] §3 red callout; r2 정정 — "STALE 재실행"이 아니라 "SMD 구버전 폐기+재실행 / SMAP·MSL 미실행분 신규 실행"). variant는 `normalonly` (각 baseline의 `experiment_configs.py` 등록 항목 그대로; SMAP/MSL 포함 등록 실재 확인).
3. **weakly supervised 4종 `[신규 실행]`**: DeepMIL/WETAS/TreeMIL/NRdetector — 구현·CPU dry-test 완료, **GPU 전체 실험 미실행**. Q1(full) variant로 전 데이터셋 실행 (epochs 50, eval 매 epoch — `baseline_common.py` weak preset). NRdetector가 최직접 경쟁자이므로 그룹 6 중 최우선.
4. **protocol-effect 블록 `[신규 실행 + 신규 loader]`**: standard clean-train split 실험 (EXECUTION-TODO 항목 3). 설계 조건 (블루프린트 §6.6 r3 — 코드 근거 포함):
   - **분할**: train = 원본 train 파일만 (test-prefix 미편입, 라벨 anomaly 0), test = **main protocol과 동일한 원본 test 뒤 50%** (평가 통일이 핵심 — 비교가 train 구성 차이만 분리하게 됨). 현행 loader에 이 variant 없음 → loader 함수/variant 추가 필요 (예: `*_standard` 키; 기존 `//2` 분할 코드의 train_len에서 prefix 항만 빼는 최소 수정).
   - **CSMAD 설정**: 271 canon config **그대로, use_grl=True 유지** — 라벨 0 train에서 세 라벨 경로는 코드 수준 자가 비활성 (priority 전부 0 → 무작위 마스킹 퇴화; OD 전 패치 정상; GRL은 batch 내 positive 부재 시 손실 자체 미계산 `loss.py:293-302`). ⚠️ **use_grl=False로 끄는 것 금지** — dead component(dynamic margin anomaly loss)가 재활성화되어 비교 오염 (§6.7 함정).
   - **baseline**: 대표 2–3종 (NUM-014; 선정 기준 — main 표에서 강한 unsupervised 대표, 예: 최상위 recent 1 + legacy 1)을 동일 standard split에서 학습. 대표 데이터셋 3개(SWaT excl22, WaDi A1, PSM) 한정으로 비용 통제.

**③ 형태** — **Bold = 열별 최고, underline = 2위** (방법 27행 대상; protocol-effect 블록은 강조 제외). 그룹 구분 midrule + 이탤릭 그룹 헤더(조건 명기 포함 — 이미 tex 확정). 하단 블록 행: CSMAD (clean) / Baseline A / Baseline B (+C는 NUM-014 확정 시).

**④ 캡션 (확정본, sec4_experiments.tex — [N]은 NUM-014 확정 후 치환)**:
> "Main comparison results under the contaminated benchmark protocol (anomaly-excised condition for unsupervised baselines; contaminated-training condition for weakly supervised baselines; §\ref{sec:baselines}). Reported metrics: PA\%K-AUC F1 and VUS-PR; the remaining three metrics are in \ref{sec:appendix_full_results}. SWaT column uses the excl22 evaluation condition; full-condition results appear in \ref{sec:appendix_swat}. SMD, SMAP, and MSL values are macro-averages over all entities (per-entity results in \ref{sec:appendix_entity_results}). **Bold** = highest; underline = second-highest. *Bottom block (protocol effect, §\ref{sec:main_results})*: CSMAD and [N] representative unsupervised baselines under a standard clean-train split (original training file only, no labeled anomalies), evaluated on the identical held-out evaluation suffix; standard-split CSMAD uses the identical configuration with all label-dependent paths automatically inactive in the absence of positive training windows. Cells are populated only for the representative protocol-effect dataset columns."

**⑤ 주의** — ⓐ NUM-006~013(본 블록)·NUM-014~019(하단 블록)·FIG-3 점선 floor·TAB-B1 Δ 기준이 전부 이 표에서 파생 — **이 표가 placeholder 의존 그래프의 루트**. ⓑ 집계에서 Exathlon·Simulation 절대 배제 (R33; 기존 Notion RankAvg 재계산 필수 — FEEDBACK-3). ⓒ weak 4종 미완 시 sync 그룹 B 전체가 "22 unsupervised"로 일괄 fallback + 그룹 6 행 삭제 (부분 게재 금지). ⓓ SWaT 재실행이 발생하면 입력 차원 45 일치 검증 필수 (FEEDBACK-7 — 현 raw CSV 경로는 51 반환). ⓔ baseline 쪽 SMD/SMAP/MSL avg도 CSMAD와 동일한 entity 집합·동일 macro 평균 규칙이어야 함.

---

### TAB-3 — Ablation study (Table 3) `[재사용(행1·3) + 신규 실행(행2·4)]`

- **위치**: §4.3 (`sec4_experiments.tex`, `\label{tab:ablation}`). half-width, ≈0.20p.

**① 들어갈 내용** — 4행 확정 (D-010 ②): 1 Full model(CSMAD) / 2 w/o GRL(OD-exclusion 유지) / 3 w/o anomaly-priority masking / 4 w/o OD loss. 열 = 대표 3–4 데이터셋(NUM-020) + Avg. 지표 = PA%K-AUC F1 (best epoch, main과 동일 기준).

**② 실험 소스 — 행별로 갈린다**:

| 행 | 소스 | 실행 지침 |
|---|---|---|
| 1 Full | `[완주 대기/재사용]` [271c] | 대표 데이터셋 열은 이미 완주분(SWaT·PSM·WaDi)에서 추출 가능 |
| 2 w/o GRL | `[신규 실행]` | **큐에 정확한 변형 부재** (exp290은 no_fm+no_grl 복합 — 행2 정의와 불일치). 신규 큐 항목: 271 canon 기반 `use_grl=False` + **anomaly-loss 경로 명시 차단** — use_grl=False 단독이면 grl_disable_anomaly_loss 게이트가 풀려 dynamic-margin anomaly loss가 재활성화됨 (§6.7 함정). `anomaly_loss_weight=0.0` 추가로 OD-exclusion(정상 패치 전용 OD)을 유지한 "GRL 순효과" 변형을 만들 것 |
| 3 w/o masking | `[재사용]` **exp287_unmask** (`287_20260603_132835_unmask`, `force_mask_anomaly=False` 단독 diff — 실측 확인) | 대표 데이터셋 분 완주 상태 — metadata 집계만. 참고(OBS-2): 큐 원항목 `config_override`에 `force_mask_anomaly` 키 중복(True→False, last-wins로 net False) — metadata 실측으로 단독 diff 확정이나, 신규 큐 항목 작성 시 이 중복 키 패턴 답습 금지 |
| 4 w/o OD | `[신규 실행]` | 신규 큐 항목: `use_output_discrepancy=False`. **score 처리 방침 (코드 확정 사실 — r2 정정)**: 기본 동작은 **자동 recon-only**다 — `mae_anomaly/scoring.py:105-106` `resolve_score_weights`가 `use_output_discrepancy=False`면 `w_disc`를 0으로 강제하고, `scoring.py:249-253`에서 `w_disc=0` → `student_error=0` → score = Teacher recon만 남는다 (구판의 "OD 학습 제거 후에도 추론 score는 disc 성분 포함" 서술은 코드와 반대 — 폐기). 즉 별도 조치 없이 학습·추론 양쪽에서 OD가 일관 제거되며, **이 자동 recon-only 동작을 표 각주로 명시**할 것. disc 성분을 score에 남기는 변형을 원하는 경우에만 별도 채점 경로가 필요 — 침묵 변경 금지 |

**③ 형태** — 행1이 기준선(최고치 기대), 변형 행은 하락폭이 드러나게 Avg 열 포함. 강조는 통상 Full 행 bold 불필요 (Table 2와 달리 비교 표가 아니라 분해 표) — Phase 7 스타일 판단에 위임하되 일관 적용.

**④ 캡션 (확정본, sec4_experiments.tex)**:
> "Ablation study. PA\%K-AUC F1 for each model variant on [3--4 representative datasets]. Row~2 (w/o GRL) removes the GRL classifier and reversal but retains the anomaly-patch OD-loss exclusion, isolating the net effect of active adversarial suppression. Extended variants (feature matching, Teacher-only warmup, symmetric decoder) are in \ref{sec:extended_ablations} (Table~\ref{tab:extended_ablations})."

**⑤ 주의** — ⓐ 대표 데이터셋 선정(NUM-020): 권장 SWaT excl22 + PSM(train AR 최대 — 라벨 경로 가장 활성) + WaDi A1 (+ WaDi A2 또는 SMD 대표 1) — 단 **선택된 열은 행 1–4 전부와 TAB-B4에서 동일해야 함** (열 불일치 금지). ⓑ NUM-021/022/023이 이 표에서 파생 (Avg 열 차분). ⓒ 행 라벨은 "w/o anomaly-priority masking" (내부명 force_mask_anomaly 금지).

---

### TAB-4 — Protocol-effect analysis `[흡수 완료 — 별도 작업 없음]`

v2-r2에서 **TAB-2 하단 블록으로 흡수** (D-010 ①). 본문 `[TAB-4]` 마커 부재 — 명세·실행 지침·의존성은 전부 위 TAB-2 ② 4항에 통합 기재했다. Notion 페이지도 별도 생성하지 않고 TAB-2 페이지에 "흡수" 한 줄로 기록.

---

## 3. Appendix Tables

### TAB-A3 — Per-baseline hyperparameters (Table A.3) `[재사용 — 코드 추출]`

- **위치**: §A.1 (`appendix_A.tex`, `\label{tab:baseline_hparams}`).

**① 들어갈 내용** — 26 baseline의 {Window, LR, Batch, Epochs, Key parameters}. Window·Epochs 열은 이미 실값 확정 (예: Anomaly Trans. 100/10, NRdetector 100/50); 잔여 [X.XX]는 LR·Batch·Key parameters.

**② 실험 소스 — `[재사용]` 학습 불필요**: **`comparison/baseline_common.py` MODEL_CONFIGS가 단일 원천** (batch 32–512 범위, 모델별 원 구현 preset). 추출 스크립트로 26개 모델 항목을 덤프해 채운다. **어떤 값도 발명 금지 (A8)** — MODEL_CONFIGS에 없는 항목은 "original preset" 표기로 남긴다. DAGMM은 simplified 표기 유지 (GMM energy 생략 각주).

**③ 형태** — 4계층 그룹(simple 9 / legacy 6 / recent 7 / weak 4), tex 확정 구조 유지.

**④ 캡션 (확정본, appendix_A.tex)**:
> "Hyperparameters of all 26 baselines. Each method retains the settings of its original implementation or publication preset; deviations from the unified pipeline (window size, epochs, batch size) are listed explicitly. DAGMM follows the simplified TranAD-repository re-implementation (GMM energy term omitted)."

**⑤ 주의** — Table A.2(budgets)와의 정합: baseline batch 열은 "model-specific (original presets)"이 정본 (구판 "512" 단일값 인용 금지 — v2-r3 정정). 큐 재실행으로 preset이 바뀌면 같은 pass에서 이 표 갱신.

### Table A.4 — Per-entity dataset statistics (부분 placeholder) `[신규 측정]`

- **위치**: §A.3 (`appendix_A.tex`, `\label{tab:per_entity}`). SMD 행의 [per-machine] 셀 3종(#Train, #Test, Train AR)만 placeholder — 나머지 행 전부 실값 확정.
- **실행**: TAB-1 ②와 **동일한 1회성 스크립트** 산출물 사용 (같은 소스 — 두 표 간 수치 불일치 금지). 28 machine 전 행을 펼칠지(28행 추가) 요약할지는 지면 판단이나, 캡션의 "SMD per-machine rows pending" 문구는 채움과 동시에 삭제.

### TAB-A6 — SWaT dual-condition results (Table A.6) `[완주 대기 + TAB-2와 동일 소스]`

- **위치**: §A.4 (`appendix_A.tex`, `\label{tab:swat_dual}`).

**① 들어갈 내용** — 27 method × {full, excl22} × 5지표 {PA%K-AUC F1, PA%K-AUC AUC-PR, VUS-PR, VUS-ROC, Affiliation F1}. CSMAD는 [271c] `SWaT/A1A2_full`·`SWaT/A1A2_excl22` 두 entity의 metadata에서 (각자 독립 best epoch — full은 `pak_auc_f1`, excl22는 `excl22_pak_auc_f1` 기준, 271_CONFIG_TRUTH §IV 운영 주의). baseline은 comparison 파이프라인의 dual 조건 산출 (`has_excl22`; 결과 디렉토리 `SWaT/A1A2_full`·`A1A2_excl22`).

**② 실험 소스** — TAB-2와 동일 실행 묶음에서 자동 산출 (별도 실험 없음): CSMAD `[재사용]`, unsupervised SWaT 분 `[재사용 — CMP-Q3]`, weak 4종 `[신규 실행 — TAB-2 ② 3항과 동일 run]`.

**③ 형태** — 좌우 5열×2 블록 (full | excl22), 27행. 강조 규칙은 본문 표와 통일(열별 bold/underline) 권장.

**④ 캡션 (확정본, appendix_A.tex)**:
> "SWaT dual-condition results: all five metrics for CSMAD and all baselines under the full condition and the excl22 condition (Section~\ref{sec:datasets}). Same trained models and identical scores in both conditions; only the evaluation mask differs. The excl22 best epoch is selected independently under the shared criterion."

**⑤ 주의** — Affiliation F1은 `_ar` 변형(`affiliation_f1_ar`) 사용 — R30 정합 (§4.1.3 본문 선언과 일치; F1-최적 threshold 변형은 ranking 비사용 선언됨). full 조건 수치가 excl22보다 크게 좋아 보이는 것이 정상([271c] 실측 0.944 vs 0.629) — 캡션의 "같은 모델, 마스크만 차이" 서술이 이 대비의 해석 장치다.

### TAB-A7 — Full multi-metric results (Table A.7) `[완주 대기 — TAB-2와 동일 소스]`

- **위치**: §A.5 (`appendix_A.tex`, `\label{tab:full_metrics}`).

**① 들어갈 내용** — 27 method × 7 데이터셋 열 × 4지표 {PA%K-AUC AUC-PR(내부키 `pak_auc_prc_auc`), VUS-ROC(`vus_roc`), Affiliation F1(`affiliation_f1_ar`), PA F1(oracle)(`pa_0_f1`)}. Table 2의 2지표를 뺀 나머지 전수.

**② 실험 소스** — **신규 실험 없음**: TAB-2를 채우는 실행 묶음의 metadata에서 metric 키만 추가로 추출 (`compute_full_metric_set`이 전 지표를 같은 best epoch에서 산출하므로 추가 비용 0).

**③ 형태** — method × metric 중첩 행 구조 (tex 확정: method당 4 metric 행). PA F1 행은 "(oracle)" 라벨 의무.

**④ 캡션 (확정본, appendix_A.tex)**:
> "Complete multi-metric results for all methods and dataset families: PA\%K-AUC AUC-PR, VUS-ROC, Affiliation F1, and PA F1 (oracle threshold; reported for comparability only, never used for ranking --- Section~\ref{sec:metrics}). PA\%K-AUC F1 and VUS-PR appear in Table~\ref{tab:main_results}."

**⑤ 주의** — PA F1은 F1-최적(oracle) threshold 기반 `pa_0_f1`이며 **`pa_0_f1_ar`은 존재하지 않음** (REQUEST-1 RESOLVED) — 키 혼동 금지. ranking·서술에 PA F1 사용 금지 (R29).

### TAB-A8 — Per-entity results (Table A.8) `[완주 대기]`

- **위치**: §A.6 (`appendix_A.tex`, `\label{tab:per_entity_results}`).

**① 들어갈 내용** — CSMAD의 entity별 {PA%K-AUC F1, VUS-PR}: SMD 28 + SMAP 54 + MSL 27 = 109행. 소스는 [271c] entity별 metadata — **271canon 완주가 유일한 전제조건** (잔여 SMD 6, SMAP 49, MSL 22).

**② 실험 소스** — `[완주 대기]` 후 집계 스크립트. 신규 실험 없음.

**③ 형태** — 3 블록(SMD/SMAP/MSL) 세로 나열. 각 블록 말미 또는 캡션에 macro 평균 = Table 2 family 열과의 일치 보장 문구 (실제 일치 검증을 채움 스크립트에 assert로 포함 권장).

**④ 캡션 (확정본, appendix_A.tex)**:
> "Per-entity results (PA\%K-AUC F1\,/\,VUS-PR) for SMD (28 machines), SMAP (54 channels), and MSL (27 channels). Macro-averages over entities equal the corresponding family columns of Table~\ref{tab:main_results}."

**⑤ 주의** — TAB-2의 SMD/SMAP/MSL avg 셀과 **수치 의존성: 이 표의 평균 = Table 2 셀** (소수 반올림 자리수까지 일관 처리). entity 명명은 tex stub의 "SMD-1-1 / SMAP-A-1 / MSL-C-1" 스타일로 통일.

### TAB-B1 — Contaminated-training condition comparison (Table B.1) `[신규 실행]`

- **위치**: §B.1 (`appendix_B.tex`, `\label{tab:contaminated}`).

**① 들어갈 내용** — 22 unsupervised baseline의 contaminated-training(무절제, 라벨 미사용) 조건 결과 + Δ(= contaminated − anomaly-excised, 양수 = contaminated 우세) + CSMAD 참조 행(Table 2 반복 — CSMAD는 두 조건 모두 contaminated train이므로). tex 확정 열: {SWaT excl22, PSM, SMD avg} × {F1, Δ} (registry 원안의 families×{F1,VUS-PR,Δ}에서 지면 축소된 형태 — **tex가 우선**).

**② 실험 소스** — `[신규 실행]`: `comparison/run_baseline_queue.py`로 22종 × 대표 3 family × variant `full`(Q1) 실행. `experiment_configs.py`에 Q1 항목이 이미 등록되어 있으므로 큐 구성만 하면 됨. SMD는 per-entity 정규화 적용 확인 후 실행 (STALE 원인 재발 방지). Δ 기준값은 TAB-2 확정본의 anomaly-excised 수치 — **TAB-2 완성 후에만 Δ 산출 가능**.

**③ 형태** — 23행(22 + CSMAD 참조) × 6열. Δ 열은 부호 표기(+/−).

**④ 캡션 (확정본, appendix_B.tex)**:
> "Contaminated-training (no-excision) condition results for all 22 unsupervised baselines. Each method trains on the identical contaminated training stream used by CSMAD (no anomaly excision; labels unused) and is evaluated on the identical held-out evaluation half. Metrics: PA\%K-AUC F1 and VUS-PR per dataset family; $\Delta$ columns give the change relative to the anomaly-excised condition of Table~\ref{tab:main_results} (positive = contaminated-training better). The CSMAD row is repeated from Table~\ref{tab:main_results} for reference, as CSMAD trains on the contaminated stream in both conditions."

**⑤ 주의** — 캡션이 "PA%K-AUC F1 and VUS-PR"을 약속하는데 tex 표 stub은 F1/Δ만 노출 — Phase 8 채움 시점에 **캡션과 열 구성 중 한쪽을 정합화**할 것 (권고: 표를 F1+Δ로 확정하고 캡션의 "and VUS-PR" 삭제, 또는 열 추가). 이 표는 R31 volume-asymmetry 인정 문장(§4.1.4)의 정량 뒷받침이다.

### TAB-B2 — Epoch-budget sensitivity (Table B.2) `[신규 실행(부분 재사용)]`

- **위치**: §B.2 (`appendix_B.tex`, `\label{tab:epoch_sensitivity}`).

**① 들어갈 내용** — 대표 unsupervised baseline(tex stub: Anomaly Transformer, TranAD)의 10(main)/50/100 epochs 성능 + CSMAD의 축소 budget/500(main) 성능. 지표 PA%K-AUC F1, best-epoch 기준 main과 동일. §4.1.2 epoch 비대칭 공개(500/50/10)의 방어 실측.

**② 실험 소스**:
- baseline 10 epochs: [CMP-Q3] `[재사용]`. 50/100 epochs: `[신규 실행]` — `baseline_common.py` epochs override로 2 모델 × 2 budget × 대표 데이터셋(2–3개, TAB-3 선택과 통일 권장).
- CSMAD 500: [271c] `[재사용]`. **축소 budget: 기존 큐 결과 재사용 가능** — exp298(`num_epochs=300, warmup=150`)·exp299(`num_epochs=200, warmup=100`) 완주분 실재 (2026-06-11 실측). 단 tex stub의 열 라벨이 "100 epochs"이므로, (i) exp299(200ep)를 쓰고 열 라벨을 "reduced (200)"로 수정하거나 (ii) `num_epochs=100, teacher_only_warmup_epochs=50` 신규 1 run — **둘 중 하나를 결정** (권고: (i) — 추가 실행 0, warmup 비율 보존).

**③ 형태** — 행 = method, 열 = budget. CSMAD 행의 비해당 budget 셀 "—".

**④ 캡션 (확정본, appendix_B.tex)**:
> "Epoch-budget sensitivity. PA\%K-AUC F1 of representative unsupervised baselines trained for 10 (main budget), 50, and 100 epochs, and of CSMAD trained for 500 (main budget) and a reduced budget, on representative datasets; best-epoch selection identical to the main protocol (Section~\ref{sec:impl})."

**⑤ 주의** — CSMAD 축소 budget에서 warmup도 비례 축소해야 student가 학습된다 (warmup=250 고정 + epochs=100이면 student 미학습 — 무의미 변형). exp298/299가 이미 이 비례를 따른다. baseline 50/100 epochs run도 best-epoch 선택 구조(매 epoch eval 후 best)는 main과 동일하게.

### TAB-B3 — Computational cost (Table B.3) `[신규 측정]`

- **위치**: §B.3 (`appendix_B.tex`, `\label{tab:compute}`).

**① 들어갈 내용** — {Single-mask, Leave-one-out, Overhead×} 3행 × {FLOPs/window, Wall-clock(s/entity), Peak GPU mem(GB)} 3열. **wall-clock overhead 비율이 NUM-031**.

**② 실험 소스** — `[신규 측정]` 학습 불필요: [271c] 대표 entity 1–2개의 best checkpoint로 측정 스크립트 1회. 측정 사양:
- **Leave-one-out**: 현행 evaluator 추론 경로 그대로 (50 masking patterns batch-병렬) — end-to-end 평가 wall-clock은 metadata `timing.inference_time`과 교차 검증.
- **Single-mask**: 동일 checkpoint로 윈도당 1-pass(단일 마스킹 패턴) 채점 모드를 측정용으로 구성 (비교 기준선 — 논문 점수 산출에는 미사용임을 표 각주에 명시).
- FLOPs: 분석식 또는 profiler(예: torch profiler) — 측정 방법을 §B.3 본문 한 줄에 명시. Peak memory: `torch.cuda.max_memory_allocated()` reset 후 측정. 동일 batch 크기·동일 entity로 두 모드 측정.

**③ 형태** — tex 확정 3×3 구조. Overhead 행은 비율(×)만.

**④ 캡션 (확정본, appendix_B.tex)**:
> "Computational cost of CSMAD inference: per-window forward FLOPs, end-to-end wall-clock evaluation time, and peak GPU memory for leave-one-out masking versus single-mask scoring, measured on representative datasets (hardware of \ref{sec:appendix_impl})."

**⑤ 주의** — **NUM-031 sync 조건**: 측정 wall-clock 배율이 50보다 유의미하게 낮으면 §5의 "approximately 50×"를 "up to 50×"로 완화 (registry §5 audit-trail 규칙). 하드웨어 표기는 TXT-001 확정값과 동일 페이지(§A.1) 참조 — 측정 머신과 학습 머신이 다르면 각주로 구분 명시.

### TAB-B4 — Extended ablations (Table B.4) `[재사용(no_fm) + 신규 실행(3종)]`

- **위치**: §B.5 (`appendix_B.tex`, `\label{tab:extended_ablations}`).

**① 들어갈 내용** — 상단 4행 {Full, w/o FM, w/o Teacher warmup(250→0), Symmetric dec.(2L/2L)} + 하단 depth sensitivity 3행 {Teacher depth 3(default)/2/1, Student 2L 고정}. 열 = TAB-3과 **동일한** 대표 데이터셋 + Avg. 지표 PA%K-AUC F1.

**② 실험 소스**:

| 행 | 소스 | 지침 |
|---|---|---|
| Full | [271c] `[재사용]` | TAB-3 행1과 동일 값 |
| w/o FM | **exp285_no_fm** `[재사용]` (`use_feature_matching=False` 단독 diff — 실측 확인, 대표 데이터셋 완주) | metadata 집계만; NUM-025 파생 |
| w/o warmup | `[신규 실행]` | 큐 신규 항목: `teacher_only_warmup_epochs=0` (그 외 271 canon). λ_rev ramp 분모가 num_epochs−warmup이므로 warmup=0이면 ramp가 epoch 0부터 시작 — 의도된 변형임을 인지 |
| Symmetric 2L/2L | `[신규 실행]` | `num_teacher_decoder_layers=2` (Student 2 유지). **NUM-024(기여 bullet 3의 load-bearing 정량 근거) 파생 — 신규 실행 중 최우선** |
| depth 3 | = Full 행 `[재사용]` | 중복 기재 |
| depth 2 | = Symmetric run과 **동일 config** | 같은 run으로 두 행을 채움 (이중 실행 불필요) |
| depth 1 | `[신규 실행]` | `num_teacher_decoder_layers=1` |

요컨대 신규 학습은 **3 run × 대표 데이터셋** (w/o warmup, teacher 2L, teacher 1L).

**③ 형태** — 상단/하단 블록 midrule 분리 (tex 확정). 열 집합은 TAB-3과 글자 단위 동일.

**④ 캡션 (확정본, appendix_B.tex)**:
> "Extended ablations: the variants beyond the confirmed rows of Table~\ref{tab:ablation} --- w/o FM loss, w/o Teacher-only warmup (250$\to$0), and a symmetric decoder (Teacher 2L\,/\,Student 2L) --- and a Teacher-decoder depth sensitivity study (3/2/1 layers against the 2-layer Student). PA\%K-AUC F1 on the ablation datasets of Table~\ref{tab:ablation}."

**⑤ 주의** — symmetric-decoder run이 게재 시점까지 미완이면 contribution bullet 3은 "design principle" 수준으로 표현 강도 하향 (Phase 6 규칙 — landing spot은 이미 B.5). §B.5 본문 문단 2개가 NUM-024/NUM-025를 들고 있다 — 표와 문단 수치 동시 갱신.

---

## 4. Algorithm

### ALG-C1 — CSMAD Training pseudocode (Algorithm C.1) `[제작 — 코드 대조 검증]`

- **위치**: §C.3 (`appendix_C.tex`, `\label{alg:training}`). 초안이 이미 tex에 작성되어 있음 ("pseudocode placeholder" 캡션 상태) — 남은 작업은 **canonical training loop와의 행 단위 대조 검증 + 캡션의 "(pseudocode placeholder)" 꼬리 제거**.

**① 들어갈 내용** — 5요소 검증 체크리스트 (정본: 271_CONFIG_TRUTH r4 §VIII + trainer.py/model.py/loss.py):
1. 전처리: SWaT constant 6컬럼 제거(45=51−6) + per-entity train-구간 min–max — 초안 반영됨.
2. anomaly-priority masking: priority 식 π_i = 10³·y_i + η_i, argtopk |M| — Eq. C.5와 기호 일치 확인.
3. Teacher-only gating: **0-based epoch 0–249 동안 학습 경로 student forward 자체 skip** — 초안의 `If e > 250`(1-based)이 0-based 250 개시와 ±1 일치하는지 epoch 표기 규약을 각주 또는 KwIn에 명시 (off-by-one이 P3 재리뷰 단골 — r4 정본: "student 학습은 0-based epoch 250(=251번째 epoch)부터").
4. 손실 조립: L_total = L_recon + L_OD + λ_FM·L_FM + λ_GRL·L_cls; **λ 이중 구조** — 손실 가중 λ_GRL(grad-ratio clamp[0,10] × 0.2, prev-epoch smoothing)과 반전 계수 λ_rev(sigmoid ramp 2/(1+e^{−10τ})−1, τ=clip((e−250)/250,0,1)) 분리 표기 — 초안 반영됨, 단일 λ로 합치지 말 것 (r4 NEW-B1). ⚠️ (OBS-1) τ식의 e는 **3항과 동일한 epoch 표기 규약을 따라야 한다** — 위 식은 1-based e 규약에서만 코드(0-based `(epoch−250+1)/250`, `trainer.py:1205-1207`)와 일치하므로, 3항의 규약 명시(각주/KwIn)가 이 식에도 적용됨을 한 줄로 연동 표기할 것 (off-by-one 재발 차단). GRL 손실은 batch 내 positive window 부재 시 skip — 초안 반영됨.
5. 평가: 5 epoch 간격 test-split 평가 + best PA%K-AUC F1 추적 — 초안 반영됨.

**② 실험 소스** — 없음 (`[제작]`). 행동 발명 금지 — 모든 줄은 trainer.py/model.py/loss.py의 실제 동작에 1:1 대응해야 하며, 의심 항목은 271_CONFIG_TRUTH의 file:line으로 재확인.

**③ 형태** — algorithm2e 2단 폭(algorithm*), ~30줄. 현 구조 유지.

**④ 캡션** — 현 tex: "CSMAD Training (pseudocode placeholder)" → 확정 시 "**CSMAD training procedure.**" 류로 교체 (placeholder 꼬리 제거가 resolved 신호).

**⑤ 주의** — 의사코드의 수식 참조(Eq. C.1/C.4/C.5, eq:ltotal 등)는 본문 식 번호와 빌드 후 재확인. AMP bf16·optimizer 세부는 의사코드 범위 밖 (Table A.1에 위임) — 추가하지 말 것.

---

## 5. Inline NUM placeholders — 소스 실험 단위 그룹 (31건 전수)

> 31건을 개별 절이 아니라 **파생 소스 단위 8개 그룹**으로 묶는다. 각 그룹의 소스 실험이 완료되면 그룹 내 전 항목이 동시에 풀린다. 누락 0 보장: NUM-001…031 전부 아래 표 중 정확히 한 곳에 등장.

### 그룹 N-A — 데이터셋 family 수 (sync 그룹 A): NUM-001, 003, 004, 029 `[완주 대기]`

- **위치**: Abstract 6문장(001), Highlights bullet 5(003), §1 기여 bullet 4(004), §5 4문장(029).
- **들어갈 값**: 6 family 전부 완주 시 "six". **네 곳이 단일 값으로 동기화**되어야 하며, §4.1.1의 하드코딩 상수("six … families", "113 entities / 114 evaluation conditions")·§4.2 "six dataset families"와도 일치 의무.
- **소스**: 271canon 완주 + baseline 재실행 완료 = TAB-2 완성이 전제. 어느 family라도 제출 시점에 탈락하면 **같은 pass에서 §4.1.1 상수까지 일괄 수정** (부분 수정 금지).

### 그룹 N-B — baseline 총수 (sync 그룹 B): NUM-002, 005, 030 `[신규 실행(weak 4종) 의존]`

- **위치**: Abstract(002), §1 bullet 4(005), §5(030).
- **들어갈 값**: weak 4종 GPU 실험 완주 시 "26"(22 unsup + 4 weak); 미완 시 **세 곳 모두 "22 unsupervised"로 fallback** + §4.1.2–4.1.4 하드코딩("26 baselines / 22 / 4")·Table 2 그룹 6 행 동시 제거.
- **소스**: TAB-2 ② 3항(weak 4종 Q1 GPU 실행)과 동일.

### 그룹 N-C — Table 2 본 블록 파생: NUM-006 ~ NUM-013 `[집계만 — TAB-2 완성 후]`

| ID | 위치(§4.2) | 정의 (집계 규칙) |
|---|---|---|
| 006 | ¶1, [N]×2 | 6 family 중 CSMAD가 1위인 family 수 — PA%K-AUC F1 기준 1개, VUS-PR 기준 1개. **WaDi 집계 규칙 결정 필요**: 표는 A1/A2 2열인데 본문은 "six families" — 권고: A1·A2 모두 1위일 때만 WaDi family win (보수적), 채택 규칙을 본문 또는 각주 1줄로 명시 |
| 007 | ¶1, [X.XX]×2 | CSMAD의 family 평균 (PA%K-AUC F1, VUS-PR) — WaDi는 A1/A2 평균을 family 값으로 한 뒤 6 family 평균 (규칙을 006과 통일) |
| 008 | ¶1 | (CSMAD 평균) − (family별 최강 unsupervised의 평균), PA%K-AUC F1 |
| 009 | ¶1 | 동일, VUS-PR |
| 010 | ¶2 | CSMAD PA%K-AUC F1 @ PSM ([271c] PSM `metrics.pak_auc_f1` — 현재도 산출돼 있으나 **표 전체 확정 전 본문 선기입 금지**) |
| 011 | ¶2 | best unsupervised PA%K-AUC F1 @ PSM ([CMP-Q3]) |
| 012 | ¶2 | CSMAD PA%K-AUC F1 @ SWaT excl22 ([271c] `SWaT/A1A2_excl22`) |
| 013 | ¶3, [X.XX]×2 | NRdetector(contaminated-training) 대비 비교값 — registry 정의는 "margins", tex 문장은 "CSMAD achieves [X.XX] … and [X.XX] … on average"로 **CSMAD 절대값** 형태. 채움 시 문장·정의 중 하나로 확정 (권고: 문장을 "outperforms NRdetector by [margin]…"으로 고치든지, 절대값+본문에 NRdetector 값 병기 — Phase 6 결정, 침묵 불일치 금지) |

- **소스**: 전부 TAB-2 완성본에서 파생 — 신규 실험 없음. 008/009/011의 "최강 unsupervised"는 family별로 다른 방법일 수 있음 — 평균 산출 규칙(각 family의 best를 뽑아 평균 vs 단일 최강 방법의 평균)을 명시하고 일관 적용 (권고: 전자 — 문장 "strongest unsupervised competitor"의 보수적 해석).

### 그룹 N-D — Protocol-effect 블록 파생: NUM-014 ~ NUM-019 `[신규 실행 — standard-split run]`

| ID | 정의 |
|---|---|
| 014 | 블록 내 대표 baseline 수 (설계 선택 2–3; tex stub는 A/B 2행) — 캡션·본문 [N] 동시 치환 |
| 015 | CSMAD clean-train 평균 (protocol-effect 대표 데이터셋들) |
| 016 | best unsupervised clean-train 평균 |
| 017 | CSMAD contaminated 평균 — **Table 2 본 블록의 같은 데이터셋 부분집합 재집계** (신규 실행 아님) |
| 018 | 파생: 017 − 015 (계산값 — 별도 측정 금지) |
| 019 | best unsupervised의 조건 간 변화량 (standard→contaminated; contaminated 쪽은 TAB-B1 또는 Q3? — **주의**: 본문 문장은 "the unsupervised baselines show [X.XX] change on the same added data"이므로 비교쌍은 standard-split run vs **contaminated-training(무절제) run** — anomaly-excised가 아니라 같은 추가 데이터를 받은 조건. TAB-B1 실행분과 소스 공유 가능) |

- **소스**: TAB-2 ② 4항의 standard-split 실험 + (019 한정) contaminated-training baseline run. 모두 PA%K-AUC F1.

### 그룹 N-E — Ablation 파생: NUM-020 ~ NUM-023 (TAB-3) / NUM-024, 025 (TAB-B4)

| ID | 정의 | 소스 |
|---|---|---|
| 020 | ablation 데이터셋 수 (설계 선택 3–4 — TAB-3 ⑤ 권고안 확정 시 결정) | 설계 + TAB-3 |
| 021 | 행1 − 행3 Avg 차 (w/o anomaly-priority masking 하락폭) | [271c] − exp287 `[재사용]` |
| 022 | 행1 − 행4 Avg 차 (w/o OD 하락폭) | 행4 `[신규 실행]` |
| 023 | 행1 − 행2 Avg 차 (GRL 순효과) | 행2 `[신규 실행]` |
| 024 | 행1 − symmetric Avg 차 — **기여 bullet 3 load-bearing** | symmetric run `[신규 실행]` (§B.5 문단) |
| 025 | 행1 − no_fm Avg 차 | exp285 `[재사용]` (§B.5 문단) |

- 부호 규약: 본문이 "removal costs X points" / "the drop is X" 형식이므로 **양수 하락폭**으로 기재 (음수면 "improves by"로 문장 자체를 고쳐야 함 — 결과 확인 후 문장 확정).

### 그룹 N-F — Sparsity sweep 파생: NUM-026, 027 `[신규 실행 — FIG-3]`

- **026** (§4.4 lead, [N]): FIG-3 데이터셋 수 (설계 2–3) — 캡션 [N] 2곳과 동시 치환.
- **027** (§4.4 Results, 서술어): 열화 형상 정성 서술어 [gradually / monotonically] — **FIG-3 곡선 확정 후** 실제 형상에 맞는 쪽 선택 (비단조면 두 단어 다 버리고 문장 재작성; A8 — 곡선 없이 단어 선점 금지).

### 그룹 N-G — Qualitative 파생: NUM-028 `[재사용 — FIG-4]`

- **028** (§4.5 lead, [N]): FIG-4 데이터셋 수 = 2 (시각화 설계 확정값 — SWaT excl22 + {WaDi A1 | PSM}). FIG-4 제작과 동시 치환.

### 그룹 N-H — Cost 측정 파생: NUM-031 `[신규 측정 — TAB-B3]`

- **031** (§B.3): leave-one-out vs single-mask **wall-clock 배율 실측값**. TAB-B3 측정과 동일 소스. **sync 조건**: 50보다 유의미하게 낮으면 §5 "approximately 50×" → "up to 50×" 완화 (§5 문장과 같은 pass 수정).

---

## 6. TXT placeholders

### TXT-001 — GPU 모델 (1개소) `[신규 측정 — 확인만]`

- **위치**: §A.1 Environment 문단 (`appendix_A.tex:80` "All experiments run on [GPU model]").
- **들어갈 내용**: 271canon(및 baseline) 실험을 실제 수행한 GPU 모델명.
- **확인 절차**: [271c] metadata에는 GPU 모델 필드가 **없다** (2026-06-11 실측 — `timing`/`config`에 부재; `device='cuda'`뿐). 따라서 ① 271canon 실행 호스트에서 `nvidia-smi --query-gpu=name` 확인 (현 machineA 실측: NVIDIA GeForce RTX 4090 — 271canon이 이 머신에서 실행 중이므로 유력하나, **호스트 이력 확인 후 기재** — 추측 금지 원칙), ② baseline 실행 머신이 다르면 그룹별 병기. 향후 재실행분은 metadata에 GPU명 기록 필드 추가 권장 (확인 비용 제거).

### TXT-002 — 코드 저장소 URL (3개소) `[결정 사항]`

- **위치**: Abstract 말미(`main.tex:110`), §A.1 Environment(`appendix_A.tex:81`), §5 말미(`sec5_conclusion.tex:31`).
- **들어갈 내용**: 공개 저장소 URL — **세 곳 글자 단위 동일** 의무. "release upon acceptance" 문구는 이미 확정.
- **절차**: 제출 단계에서는 익명 요건 확인(저널 정책에 따라 anonymous.4open.science 등 익명 미러 필요 가능) → 게재 확정 시 실제 URL로 일괄 치환. 치환 시 grep으로 3개소 동시 확인 (`grep -n "\[URL\]" sections/ main.tex`).

---

## 6R. 권고 실험 (rebuttal 대비, 원고 비반영) — D-014 (b)

> 이 절의 항목은 원고의 어떤 placeholder와도 연결되지 않는다 (원고 무변경). 리뷰 대응(rebuttal) 대비용 권고 실험으로, Notion 발행 시 '권고 실험' 하위 절로 함께 발행한다.

### 권고 실험 R-PROBE — GRL 억제의 기계적 증거 (probing classifier) `[신규 측정]`

- **목적**: rebuttal 대비 — GRL이 Student 표현에서 anomaly-identity 정보를 실제로 억제했다는 직접 증거. 원고 무변경, Notion 명세에만 등재.
- **절차**: [271c] 대표 entity(권장: TAB-3 대표 데이터셋과 동일) best checkpoint를 동결하고, test 윈도에 대해 ① Student decoder **final-layer hidden (output projection 직전 — GRL 부착 지점과 동일, FIG-2 ③ⓒ)**과 ② Teacher 동일 위치 hidden을 추출. 각 표현 위에 소형 probe(LayerNorm + Linear 1층, GRL head와 유사 용량)를 anomaly window 분류로 학습(표현은 frozen, probe만 학습) → probe AUC 비교. 기대: Student probe AUC ≪ Teacher probe AUC (억제 성공의 정량 증거).
- **확장(선택)**: TAB-3 행2(w/o GRL) run 완료 후 동일 probing을 적용해 GRL 유/무 Student probe AUC 차이를 병기 — "GRL이 없으면 Student에 anomaly 정보가 잔존"의 대조군. (exp290은 no_fm 복합이므로 대조군으로 쓸 경우 각주 필수.)
- **분류**: 학습 불필요, probe만 학습 → `[신규 측정]` 등급; §7.4 표에 1행 등재. 산출물은 본문 placeholder와 무관(원고 무변경) — Notion 페이지 '권고 실험' 하위 절로 발행.

---

## 7. 실행 우선순위 요약표

> "무엇을 돌려야 표가 채워지는가"의 역인덱스. 우선순위는 (1) 본문 핵심 표 의존 → (2) load-bearing 주장 의존 → (3) appendix 방어 실측 순.

### 7.1 재사용 가능 (실행 불필요 — 추출/제작만)

| placeholder | 소스 |
|---|---|
| FIG-1, FIG-2, ALG-C1 | 다이어그램/의사코드 제작 (정본 대조) |
| FIG-4, NUM-028 | [271c] best checkpoint + scoring.py 추출 |
| FIG-B1 좌패널 (c sweep) | [271c] checkpoint 재채점 (c∈{1,2,4,8,16}) |
| TAB-1 (SMD 셀 제외) | EXPERIMENT_PROTOCOL_TRUTH §① 실값 — 이미 tex 반영 |
| TAB-A3 | `comparison/baseline_common.py` MODEL_CONFIGS 덤프 |
| TAB-3 행3, NUM-021 | exp287_unmask 완주분 |
| TAB-B4 w/o FM 행, NUM-025 | exp285_no_fm 완주분 |
| TAB-B2 CSMAD 축소 budget | exp298/exp299 완주분 (열 라벨 정합화 필요) |

### 7.2 완주 대기 (진행 중 — 271canon 잔여 SMD 6 / SMAP 49 / MSL 22)

| placeholder | 비고 |
|---|---|
| TAB-2 CSMAD 행 (SMD/SMAP/MSL avg) | 부분 집계 금지 |
| TAB-A8 전체, TAB-A7·A6 CSMAD 행 | metadata 집계 스크립트 |
| 그룹 N-A (NUM-001/003/004/029) | "six" 확정 조건 |

### 7.3 신규 실행 필요 (우선순위순)

| # | 실험 | 채워지는 placeholder | 실행 지침 요약 |
|---|---|---|---|
| 1 | baseline SMD/SMAP/MSL 신규 실행 (anomaly-excised; SMD = 구버전(per-entity 정규화 이전) 폐기 후 재실행, SMAP/MSL = normalonly 미실행분 신규 실행 — r2 정정) | TAB-2 unsup 행, FIG-3 floor, NUM-008/009/011/016/019, TAB-A6/A7 | `comparison/run_baseline_queue.py`, variant normalonly |
| 2 | weakly supervised 4종 GPU 전체 (contaminated-training) | TAB-2 그룹 6, NUM-013, sync 그룹 B="26", TAB-A6/A7 | Q1 variant, 50 epochs; NRdetector 최우선 |
| 3 | standard clean-train split (CSMAD + 대표 baseline 2–3, 대표 3 데이터셋) | TAB-2 하단 블록, NUM-014~019 | 신규 loader variant; CSMAD는 동일 config·use_grl=True 유지 (자가 비활성) |
| 4 | ablation 행2 (w/o GRL, OD-exclusion 유지) | TAB-3 행2, NUM-023 | `use_grl=False anomaly_loss_weight=0.0` — dead-component 재활성 차단 |
| 5 | symmetric decoder (Teacher 2L) | TAB-B4 2행(symmetric+depth2), **NUM-024 (bullet 3 load-bearing)** | `num_teacher_decoder_layers=2` |
| 6 | ablation 행4 (w/o OD) | TAB-3 행4, NUM-022 | `use_output_discrepancy=False` — score는 자동 recon-only (`resolve_score_weights`가 w_disc=0 강제, scoring.py:105-106·249-253); 이 동작을 표 각주로 명시 (r2 정정) |
| 7 | label sparsity sweep (p∈{0.75,0.5,0.25,0.1} × 2–3 데이터셋) | FIG-3, NUM-026/027 | NoisyLabelSlidingWindowDataset + region 단위 재라벨 일반화 (`label_keep_ratio` 신설); p=1.0은 [271c] 재사용 |
| 8 | contaminated-training 22종 (대표 3 family) | TAB-B1 (+NUM-019 보조) | Q1 variant 큐 |
| 9 | w/o Teacher warmup (250→0) / Teacher depth 1 | TAB-B4 잔여 2행 | `teacher_only_warmup_epochs=0` / `num_teacher_decoder_layers=1` |
| 10 | epoch-budget 50/100 (Anomaly Trans., TranAD) | TAB-B2 | baseline epochs override |
| 11 | masking ratio sweep (ρ∈{0.05,0.1,0.2,0.3}) | FIG-B1 우패널 | `masking_ratio=<ρ>` 큐 4 run × 대표 데이터셋 |

### 7.4 신규 측정/스크립트 (학습 불필요)

| 작업 | 채워지는 placeholder |
|---|---|
| SMD 28 machine 분할 통계 산출 (loader 산식 재사용) | TAB-1 SMD 셀, Table A.4 SMD 행, §4.1.1 "pending" 문구 해소 |
| 추론 비용 측정 (leave-one-out vs single-mask) | TAB-B3, NUM-031 (+§5 "50×" sync) |
| GPU 모델 확인 | TXT-001 |
| 저장소 URL 확정 (게재 시) | TXT-002 ×3개소 |
| **R-PROBE** — Student/Teacher hidden probing classifier AUC 비교 ([271c] checkpoint 동결, probe만 학습 — §6R) | (placeholder 없음 — 권고 실험, rebuttal 대비 / D-014 (b)) |

---

## 8. 전수 커버리지 체크 (REGISTRY v3-r1 대조)

- **FIG 5/5**: FIG-1 (§1.1) · FIG-2 (§1.2) · FIG-3 (§1.3) · FIG-4 (§1.4) · FIG-B1 (§1.5) ✓
- **body TAB 3/3 + 흡수 1**: TAB-1 · TAB-2(+TAB-4 흡수 블록 — §2 audit 기재) · TAB-3 ✓
- **appendix TAB 8/8 + 부분 1**: TAB-A3 · TAB-A6 · TAB-A7 · TAB-A8 · TAB-B1 · TAB-B2 · TAB-B3 · TAB-B4 + Table A.4 부분(SMD 셀) ✓
- **ALG 1/1**: ALG-C1 ✓
- **NUM 31/31**: N-A {001,003,004,029} + N-B {002,005,030} + N-C {006–013} + N-D {014–019} + N-E {020–025} + N-F {026,027} + N-G {028} + N-H {031} = 4+3+8+6+6+2+1+1 = **31** ✓
- **TXT 2종/4개소**: TXT-001 ×1 (§A.1) + TXT-002 ×3 (Abstract·§A.1·§5) ✓
- **(REGISTRY 외)** 권고 실험 R-PROBE 1건 (§6R, §7.4 1행) — D-014 (b) 등재 의무 이행; 원고 placeholder와 무관 (커버리지 산식 불변) ✓

---

## 9. 정정 이력

### r2 (2026-06-11, p8 spec-fixer) — p8_notion_spec_review_r1.md 전수 반영

| 발견 | Severity | 처리 |
|---|---|---|
| F-1 | BLOCKER | D-014 (b) 권고 실험 R-PROBE(GRL probing classifier) 신설 등재 — §6R + §7.4 1행 (검수자 권고안 채택: [271c] checkpoint 동결 + Student/Teacher hidden(GRL 부착 지점 동일 — output projection 직전) probe AUC 비교 + w/o GRL 대조군 확장) |
| F-2 | MAJOR | TAB-3 행4(w/o OD) 전제 정정 — `scoring.py:105-106` `resolve_score_weights` 직접 재확인: `use_output_discrepancy=False` → `w_disc=0` 강제 → `scoring.py:249-253`에서 score 자동 recon-only. 구판의 "추론 score는 disc 성분 포함" 서술(코드와 반대)을 폐기하고 "자동 recon-only + 표 각주 명시"로 재서술 (§2 TAB-3 행4, §7.3 #6) |
| F-3 | MINOR | TAB-B4 ④ 캡션 전사 정정: "(Teacher 2L\,/\,2L)" → "(Teacher 2L\,/\,Student 2L)" (`appendix_B.tex:157` 원문 그대로) |
| F-4 | MINOR | [CMP-Q3] 서술 정정 (§0·TAB-2 ② 2·§7.3 #1·FIG-3 ⑤ⓑ): `6_20260526_*`에는 SWaT/WaDi/PSM만 존재; SMD normalonly = 구버전 `3_20260312_*`(폐기 대상), SMAP/MSL normalonly = 미존재(신규 실행) — "STALE 재실행" 서술 폐기. 실행 결론(전 entity 실행)은 불변 |
| F-5 | MINOR | FIG-B1 ⑤ⓒ 큐 범위 정정: 295–303 중 295/296/300–303 = window/patch sweep, 297 = dynamic d_model, 298/299 = epoch budget (큐 v5 직접 실측) — masking-ratio 항목 없음 결론 불변 |
| OBS-1 | 관찰 | ALG-C1 ④ τ식에 epoch 표기 규약 연동 표기 추가 (1-based e ↔ 코드 0-based `trainer.py:1205-1207`) |
| OBS-2 | 관찰 | TAB-3 행3(exp287)에 큐 `force_mask_anomaly` 키 중복(last-wins) 경고 추가 — 신규 큐 작성 시 답습 금지 |
