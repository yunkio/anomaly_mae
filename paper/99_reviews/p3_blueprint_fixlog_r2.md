---
phase: 3
agent: blueprint-reviser
inputs:
  - paper/99_reviews/p3_blueprint_redteam_r1.md (BLOCKER 3 / MAJOR 10 / MINOR 5 / NOTE 3 + 비번호 V1·V3)
  - paper/99_reviews/p3_blueprint_adversarial_r1.md (BLOCKER 5 / MAJOR 12 / MINOR 6 / NOTE 3)
outputs:
  - paper/03_blueprint/PAPER_BLUEPRINT.md (r2)
  - paper/03_blueprint/PAGE_BUDGET.md (r2)
verification_basis: |
  모든 기술 사실 수정은 정본 재확인 후 적용 — 271_CONFIG_TRUTH.md(r3, 1순위) > RESEARCH_SYNTHESIS.md(r2)
  > EXPERIMENT_PROTOCOL_TRUTH.md(r3) > NOTION_DIGEST.md(r3) > 02_venue_study dossiers.
  추가 실측 1건: PSM checkpoint patch_embed shape 직접 조회 (2026-06-11, 본 fixlog §3 참조).
last_modified: 2026-06-11
---

# Phase 3 Blueprint Fix Log — r2

## 1. 처리 요약

| 구분 | redteam r1 | adversarial r1 | 합계 |
|------|-----------|----------------|------|
| BLOCKER | 3/3 처리 | 5/5 처리 | 8/8 |
| MAJOR | 10/10 처리 | 12/12 처리 | 22/22 |
| MINOR | 5/5 처리 (+V1·V3 비번호 2건) | 6/6 처리 | 13/13 |
| NOTE | 3/3 처리 | 3/3 처리 | 6/6 |
| **총계** | **23** | **26** | **49** |

전건 수용(부분 수용 2건 — RT MAJOR-04, RT MAJOR-10: 사유 명기). 기각 0건.
추가로 리뷰가 잡지 못한 정본 충돌 1건을 검증 중 발견·정정 (**d_model dynamic → 512 고정**, §3).

---

## 2. BLOCKER 처리표 (8건)

| ID | 발견 내용 | 처리 | 반영 위치 |
|----|----------|------|---------|
| **RT BLOCKER-01** | 50% prefix 프로토콜의 leakage 방어 갭 — "test split의 ground-truth label로 학습" 공격에 정면 답변 없음 | **§14 전면 재구축** — 정면 답변 5논거: ① 재분할 정의상 편입분=train, 평가는 보존 뒤 50%만(그 라벨 일체 미사용) ② 원본 train split에 labeled anomaly 구조적 부재 — SWaT/WaDi 원본 train=정상 운영 구간(SWaT train anomaly 1.63%는 전부 편입 A2-front 유래), PSM/SMD train 라벨 파일 부재, SMAP/MSL train 라벨 명시적 0 (EXPERIMENT_PROTOCOL_TRUTH §①·② 실측 인용) → 원본 split로는 semi-supervised TSAD 평가 정의상 불가능 = 프로토콜 존재 이유 ③ 전 모델 동일 데이터(비지도=Q3 normalonly 최선 활용, R12) ④ 시간 순서 보존 + //2 전 데이터셋 통일(R13) ⑤ NRdetector 7:3 재분할 선례(NRDETECTOR_DOSSIER §3.1/D8; 시간 순서 보존 여부 미명시 — 단정 인용 금지 주의 포함). + prefix/test 분포 이동 **한계 인정 1문장**. "why not original-train labels only?" 1문단 답은 §4.1.1 배치 + Intro Para 3 에코 1문장 | BLUEPRINT §14, §6.2, §3.1 Para 3, §15 행 1 |
| **RT BLOCKER-02** | contribution bullet 3에 "teacher-only warmup" 명시 — ablation 부재(REQUEST-F) + SDMAE도 teacher-first 2단계 학습 → novelty 없음·ablation 요구 빌미·§5.5 CRITICAL NOTE와 자기모순 | bullet 3에서 warmup 문구 **삭제** — 아키텍처 논리(deeper teacher의 안정 기준 + capacity-limited student의 선택적 모방 실패 → contaminated train에서 신뢰성 있는 discrepancy)로 재구성. warmup은 §3.4 학습 안정화 장치 전속 + §12 표에 "contribution 서술 금지" 명기. ablation Table 3의 warmup=0 행은 **placeholder로 유지** (orchestrator 지침) — 단 명시적 conditional (§6.7; RT MAJOR-10 참조) | BLUEPRINT §11 결정 ① bullet 3, §5.5, §6.7, §12, §15 |
| **RT BLOCKER-03** | Q3 단독 main table — 표준 clean-train 조건 비교 부재로 방법론 효과 vs 프로토콜(데이터 추가) 효과 분리 불가 | **§4.2 내 protocol-effect 보조 분석 신설 (main text 격상, Appendix 아님)** — Table 4(half-width): [MODEL]+대표 비지도 baseline 2–3종 × 대표 2–3 데이터셋 × 2조건 {standard split(원본 train만, prefix 미편입 — 라벨 경로 휴면), contaminated(main)}; **두 조건 모두 평가는 동일한 원본 test 뒤 50%로 통일**(train 구성 차이만 분리). 2단 논증: ① 표준 조건에서도 경쟁력 유지(방법 효과) ② contaminated에서 제안 방법만 추가 이득(라벨 활용 효과). EXPERIMENT_EXECUTION_TODO(standard-split 실험 미실행 — Phase 5 진입 전 실행, 대표 데이터셋 한정으로 비용 통제). 분량 영향 PAGE_BUDGET 반영(§4 3.2→3.3p, §1 1.7→1.6p 상쇄). 결정 ④ 본문 갱신. *해석 주*: orchestrator 표기 "Q1(클린/표준 조건)"은 redteam 원문의 요구(원본 split 조건)로 구현 — 프로젝트 정의상 Q1(=contaminated full, 라벨 미사용)과 구분되며, Q1 병기는 §A.2에 기존대로 유지 | BLUEPRINT §6.6, §6.1, §11 결정 ④, §15 신설 행; PAGE_BUDGET §1·§2·§3 |
| **ADV BLK-001** | BLUEPRINT §2(~1.8/~1.2/~2.8/~2.8/~0.3)와 PAGE_BUDGET §1(1.7/1.1/2.7/3.2/0.3) 분량 수치 불일치 + §4 세부 합계 초과 | **PAGE_BUDGET을 분량 단일 정본으로 선언** (양 문서 frontmatter 명기); BLUEPRINT §2를 PAGE_BUDGET §1 전사로 교체 (r2 값: 1.6/1.1/2.7/3.3/0.3 = 9.0). §4 세부 재계산 + 압축 전략 6개로 확장(합계 ~0.65p) → 압축 후 3.3–3.4p | BLUEPRINT frontmatter·§2; PAGE_BUDGET frontmatter·§1·§2 |
| **ADV BLK-002** | Fig. 2 설계에 GRL 추론 시 비활성 표시 누락 + GRL 위치(student hidden, output projection 이전) 불명확 — NOTION I-3 forward flow(Output Projection 다음)와 코드 정본 충돌 | §5.3에 Fig. 2 필수 레이블 2건 명시: ① GRL+AnomalyClassifierHead 위치 = **student decoder 마지막 층 hidden, output projection 이전** (271_CONFIG_TRUTH §VI model.py:1150–1154 — NOTION 배치 부정확 판정, 코드 정본 우선) ② **"GRL: training only (추론 시 비활성)"** dashed box/주석 필수. §5.6(C)·§5.7에도 위치·비활성 표기 동기화. encoder gradient 차단 서술의 근거 2분화(student detach / GRL detach) 정밀화 | BLUEPRINT §5.3, §5.6(C), §5.7 |
| **ADV BLK-003** | SMD 실제 입력 F=29–36(constant 제거 후) vs NOTION F=38 혼동 — d_model 표 정정 필요 | §6.2에 "SMD 28 machines, constant 컬럼 제거 후 29–36 (raw 38은 제거 전)" 명기; Appendix §C.1을 **Input Dimensionality Table**(SWaT 45=51−6, SMD 29–36=38−constant, WaDi 123=127−4 NaN)로 교체. **+ 검증 중 추가 발견·정정 (§3 상세): d_model 자체가 dynamic이 아니라 전 entity 512 고정** — 리뷰의 전제("런타임 d_model 동적 결정")까지 정정 | BLUEPRINT §5.4, §5.5, §6.2, §6.3, §8(C.1), §9.1, §9.2; PAGE_BUDGET §5 |
| **ADV BLK-004** | §5.5에 GRL λ sigmoid ramp-up 공식(λ=2/(1+exp(−10p))−1) 잔존 — 271 실제는 trainer inline grad-ratio adaptive λ. §9.2 방침과 자기모순 | sigmoid ramp-up 서술 **삭제**, 교체: "warmup 종료 직후(epoch 250) GRL·FM 손실 ramp 없이 즉시 투입; λ_GRL_adp = clamp(‖∇L_main‖/(‖∇L_GRL‖+1e-4), 0, 10) (직전 epoch 값), λ_GRL_eff = λ_GRL_adp × grl_loss_weight(0.2)" (271_CONFIG_TRUTH §VIII GRL Details/Training, trainer.py:751–765 — "ramp 없음"도 정본 명기 사항). §9.2에 "sigmoid ramp-up(Ganin schedule) 서술 금지 — 271 미사용" 추가. §5.6의 λ 표기도 clamp(…+1e-4) 형태로 정밀화 | BLUEPRINT §5.5, §5.6, §9.2 |
| **ADV BLK-005** | epoch 비대칭(MAE 500 vs unsup 10 vs weak 50) 공정성 공격 방어 계획 부재 (+ MAJ-011 batch 1024 vs 512) | §6.3에 비대칭 사실 **그대로 공개**(epoch 500/10/50 + eval 간격 5/1 + batch 1024/512) + 방어 1–2문장; §15에 공정성 공격 시나리오 신설: ① 전 모델 "주기 평가 후 best-epoch 선택(pak_auc_f1)" 동일 구조 + early stopping 양쪽 부재(EXPERIMENT_PROTOCOL_TRUTH §④-실행 3항 ⓐⓑⓒ) ② 모델군별 수렴 특성 반영 best-effort budget ③ Appendix §B.4 epoch-budget sensitivity placeholder(optional) 신설 | BLUEPRINT §6.3, §8(B.4), §15 신설 행; PAGE_BUDGET §5 |

---

## 3. 검증 중 추가 발견 (리뷰 외 정본 충돌 — 정정 적용)

**d_model "dynamic" 서술은 271 실측과 불일치 — 전 entity 512 고정으로 정정.**

- 블루프린트 r1은 §5.4/§5.5/§6.3/§9.1/§C.1에서 "d_model=dynamic(F→{128,192,256,384,512}, cap=512)"로 서술했고, ADV BLK-003도 이 전제("실제 각 entity의 런타임 d_model은 동적으로 결정된다")를 수용한 채 SMD F 범위만 정정 요구했다.
- 그러나 **1순위 정본 271_CONFIG_TRUTH §II**는 `d_model=512`·`dim_feedforward=2048`을 **전 37 entity metadata 공통 114키**에 포함한다 (PSM F=25, SMD F=29–36 포함).
- **직접 실측 (2026-06-11, 본 reviser)**: `results/experiments/271_20260602_020545_271canon_baseline/PSM/experiment_metadata.json` → `config.d_model=512, dim_feedforward=2048, num_features=25`; `PSM/best_model.pt` → **`patch_embed.weight = (512, 250)`** = Linear(10×25 → 512). dynamic 매핑(min d ≥ 10×25=250)이었다면 **256**이어야 함 → 271 런타임 모델이 d_model=512임이 checkpoint 수준에서 확정. SMD machine-1-4·SWaT full metadata도 동일(512/2048).
- NOTION I-3의 dynamic 매핑 공식·표("Set C")는 271에 미적용된 preset 문서 — batch_size 512(Set C)→1024(271 override)와 동급의 stale (RESEARCH_SYNTHESIS §⑥ N2 선례와 동일 패턴). NOTION 표의 "SMD F=38, d_model=384" 행은 이중으로 부정확(F는 raw, d_model은 미적용 매핑).
- 적용: 전 위치에서 d_model=512(고정)·dim_feedforward=2048로 교체, "dynamic" 표기 금지를 §9.2에 등재, §C.1을 Input Dimensionality Table로 개편. ADV BLK-004의 "dim_feedforward=4×d_model 서술 자체는 맞다" 판정도 동적 함의가 있어 고정값 직접 표기로 대체.

---

## 4. MAJOR 처리표 (22건)

### redteam (10건)

| ID | 발견 | 처리 |
|----|------|------|
| RT MAJOR-01 | SDMAE anomaly-overlook 평행선 방어가 §2.3 각주에만 의존 — Method 본문 부재 | §5.6(C) 서두에 1문장 명시: "SDMAE's anomaly-overlook supervision operates in the target/loss space; our GRL operates in the gradient space of the student's internal representation." 각주는 용어계보+구조차이로 축소(분산 배치 — MAJOR-08과 연동) |
| RT MAJOR-02 | WETAS/TreeMIL도 end-to-end인데 "최초" 근거 미흡 | §4.3(§2.2)에 차별 논리 1–2문장 명시: weakly-supervised 계열의 weak label은 분류/정렬 목적함수의 지도 신호(출력 결정 수준)이고 자기지도 pretext 부재(NRDETECTOR_DOSSIER 원문 구조·D5) — 본 논문의 스코핑("자기지도 표현 학습의 기울기에 통합")과 비중첩 |
| RT MAJOR-03 | Q3에서 비지도 baseline의 train 데이터 양(quantity) 불리 미인정 | §6.5에 양적 비대칭 인정 1문장(절제분=train AR 0.5–6.2% 수준 + windowing 손실) + Q1(§A.2)·Table 4(§4.2)가 보완 비교 제공 명시 |
| RT MAJOR-04 | test split 기반 best-epoch 선정 = oracle 공격 무방비 | **부분 수용**: "validation split으로 변경" 권고는 기각 — 실제 271+baseline 프로토콜이 test-split 선정(전 모델 동일)이며 사후 서술 변경은 정본 위반(EXPERIMENT_PROTOCOL_TRUTH §④ M-3 "반드시 공개" 의무). 채택안: §6.3 명시 공개("uniformly applied to all methods; no separate validation split") + §15 방어 행(공정성 유지/낙관 편향 한계 인정/PA%K 적분 특성) + §B.4 sensitivity placeholder 옵션 (= 리뷰가 제시한 대안 경로 "sensitivity 방어"의 채택) |
| RT MAJOR-05 | "GRL vs anomaly-OD 제외만"의 차이 논증·ablation 분리 부족; "w/o GRL" 정의 불명 | §5.6(C)에 "수동 회피(OD 제외) vs 능동 제거(GRL)" 구분 논증 명시; §6.7 변형 2를 "w/o GRL, anomaly-OD 제외 유지"로 정의 확정 + 코드 함정 경고(use_grl=False 단독 시 dead-component dynamic margin 재활성화 → ablation config에서 차단 유지 — EXPERIMENT_EXECUTION_TODO 설계 조건); Intro Para 3 bridge 문장(MINOR-06 연동) |
| RT MAJOR-06 | 6개 데이터셋 선택 근거 부재 | §6.2에 선택 근거 1문장(산업제어/IT인프라/우주 telemetry 3도메인 + 운영 스트림 내 anomaly 발생으로 contaminated 설정 구성 가능) + clean-train 가정 문헌 인용 Phase 4 수요 |
| RT MAJOR-07 | bullet 2와 3의 경계 모호 — asymmetric decoder가 bullet 2에 흡수 가능 | bullet 3을 독립 논리로 재구성(라벨-무관 구조 기반: capacity gap → discrepancy 신뢰성) + MECE 검증문에 경계 명문화("라벨 신호의 주입(2) vs 신호가 발현되는 구조적 기판(3)") — warmup 제거(B-02)와 동시 적용 |
| RT MAJOR-08 | 옵션 C 문구 "extend analogous principles"가 SDMAE를 parent로 격상 | "adapt this architectural paradigm" / "apply the time-series counterpart" 계열로 교체 (sibling 포지셔닝); 각주 1개에 3축을 다 담는 부담 해소 — 작동 계층 차이는 §3.5 본문 이동, 각주는 용어계보+구조차이 전속 (§11 결정 ⑤ 갱신) |
| RT MAJOR-09 | "contaminated semi-supervised" 명명 — main 실험이 상한 케이스임을 §3.1 미명시 시 reject 위험 | §5.2에 RESEARCH_SYNTHESIS §②-1(설정)/②-2(main=상한 케이스 FACT)/②-3(sweep=일반 케이스) 3단 구조 명시 서술 의무화 + §11 결정 ②에 방어 조건으로 등재 |
| RT MAJOR-10 | ablation 변형 6(warmup) placeholder 행이 미완 상태로 분량·논증에 유입 | **부분 수용** (orchestrator 지침으로 행 유지): 행은 placeholder로 유지하되 **명시적 conditional** — REQUEST-F 실험 완료 시에만 main Table 3 포함, 미완료 시 삭제 후 §B.1 강등/생략(drafter가 미완 placeholder를 본문에 남기는 것 금지). warmup은 contribution이 아니므로 행 삭제가 논증 완결성 훼손하지 않음을 명기. PAGE_BUDGET §3 표에도 conditional 표기 |

### adversarial (12건)

| ID | 발견 | 처리 |
|----|------|------|
| ADV MAJ-001 | SDMAE를 "공유 encoder 이중 decoder 구조"로 오기술 — 실제는 teacher decoder 첫 블록 뒤 branch-off | §4.4 정정: branch-off 구조 명기(ANCHOR_SDMAE_DOSSIER §3.1 원문 인용) + 본 논문의 "독립 비대칭 decoder vs branch-off 분기" 구조 차이를 R21 방어 각주 재료로 등재 (§11 결정 ⑤ 각주 초안에 반영). §11 결정 ① C2 행의 표현도 "개념 공유(구조는 상이)"로 정밀화 |
| ADV MAJ-002 | "6계열" 표기 모호(나열은 7개) + 잔여 entity 미완료 사실 §6.2 미반영 | §6.2를 "6 데이터셋 계열(WaDi A1/A2는 독립 entity·Table 1 별도 행), 총 113 학습 단위(산식 명기)"로 재서술 + §0.4 완주 상태 경고 신설(37/113, baseline STALE, weak 4종 미실행) + §6.6 Table 2에 "완주 후 수치 채움" 명기 |
| ADV MAJ-003 | SWaT 45-feature 재현성 플래그 미반영 | §6.3에 1줄 명기(45=51−constant 6 {P202,P401,P404,P502,P601,P603}; 현 환경 loader는 51 반환 — 재실험 전 검증 필수, FEEDBACK-7) + §C.2 전처리 단계 등재 + §11 결정 ③ 갱신 조건 연동(MINOR-002) |
| ADV MAJ-004 | focal-style BCE 변형의 positive 표기 지침·Lin et al. 차이 설명 자리 부재 | §5.6(C)에 확정 표기 "focal-style BCE variant with class-prior pos_weight" + 차이 1문장(표준: p_t=모델 예측 확률 / 본 변형: p_t:=exp(−BCE_{w+})) + 예시 문장 제공; §9.2 금지사항을 positive 지침으로 보강 |
| ADV MAJ-005 | test-set model selection 방어 계획 부재 (RT MAJOR-04와 동근) | RT MAJOR-04 처리와 통합 — §6.3 공개 문구 + §15 방어 행 + §B.4 placeholder |
| ADV MAJ-006 | "SOTA Legacy 6" 제목에 7개 나열(GCN-LSTM 포함) — 내부 모순 | EXPERIMENT_PROTOCOL_TRUTH §③(r2 정정본) 분류로 교체: Simple 5 + Neural 3 + **GCN-LSTM 1(독립)** + SOTA Legacy 6(anomaly_transformer, tranad, usad, dagmm, gdn, omnianomaly) + SOTA New 7 = 22 정합. NOTION II-2의 "legacy 7"은 r2 정정 이전 오기로 판정 |
| ADV MAJ-007 | 총손실 수식 L_GRL vs 본문 L_cls 기호 혼용 | L_total = L_recon + L_OD + λ_FM_eff·L_FM + **λ_GRL_eff·L_cls**로 통일; §9.1 notation 표에 L_recon/L_OD/L_FM/L_cls 행 신설 + "L_GRL 혼용 금지" 명기 |
| ADV MAJ-008 | "ratio ≤ 6.2%" 상한 — SMD per-machine 미확정 상태에서 단정 | §5.2를 실측 열거형으로 교체(SWaT 1.63 / WaDi 0.52·0.76 / PSM 6.20 / SMAP 0.70 / MSL 1.70%) + "SMD per-machine 확정 전 전체 상한 단정 금지" 명기; §6.2 실측치 목록에도 SMD 대기 명기 |
| ADV MAJ-009 | score 수식의 per-patch/per-timestep 계층 불명확 | §5.7 재서술: recon_p·disc_p per-patch 명기, 수식 (11)(12)=per-patch, (13)=patch→point **mean 집계**(bincount-합/coverage, EXPERIMENT_PROTOCOL_TRUTH §④-실행 2항 정합) 구분 확정. ε=1e-4는 정본 일치 확인(271_CONFIG_TRUTH §VIII) — 변경 없음 |
| ADV MAJ-010 | warmup이 bullet 3에 있으면서 Table 3 placeholder — 연쇄 수정 위험 | RT BLOCKER-02 + RT MAJOR-10 처리로 해소 (bullet에서 제거 → Table 행 삭제 시에도 contribution 연쇄 수정 불필요한 구조 확보) |
| ADV MAJ-011 | batch_size 1024(MAE) vs 512(baseline preset) 차이 방어 부재 | ADV BLK-005 처리에 통합 — §6.3 공개 + §15 공정성 행("원 구현 충실 원칙"으로 동일 프레임) |
| ADV MAJ-012 | Conclusion의 complementary masking(7-pass)이 비활성 옵션임이 불명확 | §7에 수식어 의무화: "코드에 구현되어 있으나 본 실험 미사용(eval_complementary_masking=False) — 향후 연구에서 cost-accuracy tradeoff 탐색 가능" |

---

## 5. MINOR 처리표 (13건)

| ID | 발견 | 처리 |
|----|------|------|
| RT MINOR-01 | contaminated protocol 기여의 uniqueness 인용 부재 | §6.2 선택근거 문장에 "기존 벤치마크 clean-train 가정" 문헌 1–2개 인용 수요 명기 + §16 Phase 4 연계 표 등재 + §13에 주의 추가 |
| RT MINOR-02 | §4.5 정성 분석 "어떤 유형" 해석이 placeholder | §6.9에 EXPERIMENT_EXECUTION_TODO 조건 명기(수치 확정 전 작성 금지 + 유형/사건 근거 의무) + SWaT excl22 소형사건 대표성 확인 주의 |
| RT MINOR-03 | WETAS/DeepMIL/TreeMIL의 §2.1/§2.2 배치 "또는" 모호 | §4.1·§4.3에서 "§2.2 전속, §2.1 포함 절대 불가"로 확정; NRdetector와 분리 서술로 "거의 유일" 스코핑 보존 |
| RT MINOR-04 | §4.2/§4.3 component 서사 중복 | §6.1에 중복 방지 지침 신설 + §6.6 분석 구조 수정("한계 연결" 1문장 이내, component 서사 §4.3 전속) + PAGE_BUDGET 압축 전략 5와 연동 |
| RT MINOR-05 | 모델명 후보 TS-SDMAE 잔존 | §10.1에서 제거(취소선+사유) + §11 결정 ⑧ 신설(DECISION_LOG 전사 필수) — ADV NOTE-003과 통합 처리 |
| RT V1 (비번호) | Table 2 landscape의 elsarticle 지원 미확인 | PAGE_BUDGET §2 압축 전략 1·§7·§9에 Phase 5 템플릿 확인 플래그 + fallback(fontsize/tabcolsep/지표 1열화) 명기 |
| RT V3 (비번호) | Table 2 열 구성 미확정("지면 허용 시" drafter 위임) | 열 구성 확정: 데이터셋 × {PA%K-AUC F1, VUS-PR}; 나머지 3지표 §A.3 위임 — BLUEPRINT §6.6 + PAGE_BUDGET §2·§3 |
| ADV MINOR-001 | TFMAE가 §2.1과 §2.3에 이중 등장 | §2.1 자기지도 클러스터에서 TFMAE 제외, §2.3 1문장 인용을 유일 언급으로 확정 (§4 baselines에서는 이름+인용 결합 — 기존 R19 정책 유지) |
| ADV MINOR-002 | excl22 기준값 0.62899의 재실험 시 갱신 지침 부재 | §11 결정 ③에 갱신 조건 신설(FEEDBACK-7 해소·재실험 시 수치 업데이트, 선정 원칙은 유지) + §6.2 참조 연결 |
| ADV MINOR-003 | warmup 중 "student frozen" vs "forward 수행·손실 비활성" 표현 충돌 | §5.5를 정밀 표현으로 확정: "forward는 수행되나 student 관련 손실항 전부 비활성(teacher_only=True 게이트) → gradient 미흐름; 'frozen'을 forward 중단으로 오독 금지" |
| ADV MINOR-004 | Affiliation F1의 사용 threshold 키 미명기 | §6.4에 `affiliation_f1_ar`(AR threshold, evaluator.py:809–813) 확정 + F1-최적 `affiliation_f1`과 혼용 금지 명기; 5지표 전부 내부 키 병기 |
| ADV MINOR-005 | "contaminated semi-supervised" 기존 문헌 사용 여부 미검증 | §11 결정 ②에 Phase 4 용어 검색 검증 항목 명기 + §16 연계 표 등재 |
| ADV MINOR-006 | Thesis의 "최초" 단언 — INFERENCE 등급 주장 | §0.1에 "(to our knowledge)" 의무화 + 스코핑 경계 박스 신설(RT NOTE-03 통합) + Phase 4 반증 검색 §16 등재; §4.3 포지셔닝 문장에도 동일 적용 |

---

## 6. NOTE 처리표 (6건)

| ID | 발견 | 처리 |
|----|------|------|
| RT NOTE-01 | DAGMM provenance 확정의 DECISION_LOG 기록 불확실 | §11 **결정 ⑦ 신설** — 표기 확정("DAGMM (simplified variant, following [TranAD repo])" + GMM energy 제거 각주; §2.1은 원논문만) + "Phase 4 진입 전 DECISION_LOG 전사 필수" 명기. (DECISION_LOG 자체는 본 reviser 쓰기 범위 외 — orchestrator 전사 요청) |
| RT NOTE-02 | baseline 10 epoch vs 500 epoch 근거 서술 부재 | ADV BLK-005 처리에 통합 (§6.3 공개+방어 문장, §15 행) |
| RT NOTE-03 | "최초" 스코핑 경계의 Phase 5 명시 필요 | §0.1 스코핑 경계 박스 + §15 신설 행("'최초' 주장 과장" 시나리오) — ADV MINOR-006과 통합 |
| ADV NOTE-001 | §2.1 클러스터 인용의 DAGMM이 원논문/variant 중 무엇인지 불명 | §4.2에 인용 정책 명기(§2.1=Zong et al. 원논문만; variant 표기는 §4.1.4/Appendix 전속) — 결정 ⑦에 포함 |
| ADV NOTE-002 | focal-style 변형의 독창성 명시 부재 | §5.6(C)에 "본 논문에서 설계한 변형" 1문장 명시 + 예시 문장 ("We design a focal-style variant based on BCE with class-prior pos_weight, rather than adopting the standard focal loss [Lin et al. 2017].") |
| ADV NOTE-003 | TS-SDMAE naming conflict 위험 | RT MINOR-05와 통합 — §10.1 제거 + 결정 ⑧ |

---

## 7. EXPERIMENT_EXECUTION_TODO 집계 (Phase 5 진입 전 실행/완주 필수 — 블루프린트 §0.4·각 절에 표식)

1. MAE 271 잔여 entity 완주 (SMD 6, SMAP 49, MSL 22) + baseline SMD/SMAP/MSL 재실행(per-entity 정규화 STALE).
2. weakly-supervised 4종(DeepMIL/WETAS/TreeMIL/NRdetector) GPU 전체 실험 — NRdetector는 최직접 경쟁자.
3. **Protocol-effect 실험 (r2 신설, RT BLOCKER-03)**: standard split(원본 train만) 조건의 [MODEL]+대표 baseline, 대표 2–3 데이터셋 — Table 4 입력.
4. Label sparsity sweep (R32, p ∈ {1.0,…,0.1}) — Fig. 3 입력.
5. Warmup ablation (REQUEST-F) — 완료 시에만 Table 3 행 6 유지 (conditional).
6. "w/o GRL" ablation 변형의 config 설계 조건: anomaly-OD 제외 유지(dead-component dynamic margin 재활성화 차단) — RT MAJOR-05.
7. (optional) Epoch-budget sensitivity — §B.4 placeholder, ADV BLK-005 방어 보조.
8. §4.5 정성 figure의 유형별 해석 — 수치 확정 후 (RT MINOR-02).

---

## 8. 잔여/이관 사항

- **DECISION_LOG 전사 필요 (orchestrator)**: 결정 ⑦(DAGMM 표기), 결정 ⑧(TS-SDMAE 제외), 결정 ④ 갱신(Table 4 보조분석), PAGE_BUDGET 정본 선언(BLK-001), d_model=512 정정(§3).
- **Phase 4 검증 수요**: 블루프린트 §16 표로 일원화 (clean-train 가정 문헌 / 용어 기존 사용 / 최초성 반증 / Lin et al. 수식 대조 / AR threshold 관행 등).
- redteam R10-1 항목 1·2(Linear patchify·Pre-Norm 논리 보강)는 리뷰 스스로 "현 방향 올바름/MINOR" 판정 — §12에 MAE-원류 귀속 표현 보강 외 현행 유지.
