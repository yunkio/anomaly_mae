---
phase: 3
agent: adversarial-reviewer
directives: [T3, R1, R2, R5, R6, R7, R8, R9, R10, R11, R15, R16, R19, R20, R21, R22, R32]
last_modified: 2026-06-11
authority: |
  정본 우선순위: 271_CONFIG_TRUTH.md > RESEARCH_SYNTHESIS.md > EXPERIMENT_PROTOCOL_TRUTH.md
  > NOTION_DIGEST.md > 02_venue_study dossiers.
  본 리뷰는 PAPER_BLUEPRINT.md + PAGE_BUDGET.md를 대조 검증 대상으로 삼는다.
---

# Phase 3 블루프린트 Adversarial Review — r1

> 검증 범위: PAPER_BLUEPRINT.md 전문 + PAGE_BUDGET.md 전문
> 대조 정본: 271_CONFIG_TRUTH.md (r3), RESEARCH_SYNTHESIS.md (r2),
>            EXPERIMENT_PROTOCOL_TRUTH.md (r3), NOTION_DIGEST.md (r3),
>            ANCHOR_SDMAE_DOSSIER.md (r2), NRDETECTOR_DOSSIER.md,
>            STRUCTURE_AND_FIGURE_PATTERNS.md (r2)
> 검토일: 2026-06-11

---

## 1. Directive 체크리스트 (Phase 3 해당 17종 전수)

| Directive | 블루프린트 충족 절 | 충족 여부 | 미충족/위반 사항 |
|-----------|----------------|-----------|----------------|
| T3 (진실 문서 활용) | 전체 구조가 정본 참조 표명 | 부분 충족 | d_model, 학습 epoch 분리, 기타 수치 오류 — 아래 개별 항목 참조 |
| R1 (MECE contribution) | §11 결정 ①, §4/§6 MECE 논거 | 충족 | Contribution 4개 MECE 논거 명기 |
| R2 (C1-C4 선판단) | §11 결정 ① — 채택/수정/기각 명기 | 충족 | C1 수정·C2 수정·C3 기각·C4 채택 확장 — 4개 모두 처리됨 |
| R5 (notation 방침) | §9 Notation 설계 방침 | 충족 | 기호 표 + 수식 금지사항 명기 |
| R6 (9p 배분) | PAGE_BUDGET.md 전문 | 부분 충족 | §4 산술 오류 — 세부 합계 불일치 (BLK-001) |
| R7 (appendix) | §8 Appendix 구성 계획 | 충족 | 주요 위임 항목 전수 열거 |
| R8 (코드 공개 문구) | §11 결정 ⑥ | 충족 | 조건부 포함 + checklist 연계 명기 |
| R9 (SDMAE 포지셔닝) | §4.4, §11 결정 ⑤ | 부분 충족 | 옵션 C + 각주 방식 채택 명기되나, 계보 서술의 선행 귀속이 SDMAE 원문과 불일치 (MAJ-001) |
| R10 (다변량 시계열 논증) | §12 R10 논증 배치 전수표 | 충족 | 10개 component 전수 배치 명기 |
| R11 (설정 도입) | §5.2, §3 Para 3 | 충족 | contaminated semi-supervised 공식화, label-free 추론 명시 |
| R15 (제목·모델명 후보) | §10 | 충족 | 모델명 5개 + 제목 5개 후보, 불필요 신규 약어 금지 명기 |
| R16 (related work 스코핑) | §4.3 | 충족 | 다변량 TSAD semi-supervised 희소성 스코핑 명기 |
| R19 (괄호 클러스터 인용) | §4.2, §4.3, §6.5 | 충족 | 개별 소개 금지 + 클러스터 인용 방침 명기 |
| R20 (NRdetector 스코핑) | §4.3 | 충족 | 차이 우선 + 공통점 간략 + D1/D3/D5 배치 명기 |
| R21 (self-distillation 용어 방어) | §4.4, §11 결정 ⑤ | 부분 충족 | 각주 방어 계획 있으나 계보 귀속 오류 (MAJ-001) |
| R22 (MAE 아이디어 귀속) | §5.4, §4.4 | 충족 | He et al. 2022 명기 + 독립 수렴 구분 명기 |
| R32 (label sparsity analysis) | §6.8 | 충족 | 동기·설계·논리·figure 계획 전수 명기 |

---

## 2. 발견사항 (severity별)

---

### BLK-001

**Severity**: BLOCKER
**Artifact**: PAGE_BUDGET.md §2 §4 + 섹션 구조 개요

**문제**: PAGE_BUDGET.md의 섹션 합산이 PAPER_BLUEPRINT.md 선언과 불일치하며, §4(Experiments) 세부 내역 합계가 목표치를 초과한다.

PAPER_BLUEPRINT.md §2의 섹션 구조 선언:
- §1 Introduction ~1.8p
- §2 Related Work ~1.2p
- §3 Methodology ~2.8p
- §4 Experiments ~2.8p
- §5 Conclusion ~0.3p

PAGE_BUDGET.md §1 섹션 배분:
- §1 Introduction 1.7p
- §2 Related Work 1.1p
- §3 Methodology 2.7p
- §4 Experiments 3.2p
- §5 Conclusion 0.3p
- 합계 9.0p

불일치 1: BLUEPRINT §2의 §4 "~2.8p" vs PAGE_BUDGET §1의 §4 "3.2p" — 0.4p 차이. BLUEPRINT를 보는 Phase 5 drafter가 §4에 2.8p 예산을 할당하면 PAGE_BUDGET와 충돌한다.

불일치 2: PAGE_BUDGET §2 §4(Experiments) 세부 내역 합계 = 3.54p, 목표는 3.2p. 문서 자체가 "슬랙: −0.34p (초과 — 압축 필요)"라고 명기하고 압축 전략을 제시했으나, 압축 후 예상치("3.2–3.4p")는 여전히 목표치 상단을 초과한다.

불일치 3: BLUEPRINT §2의 §1 Introduction "~1.8p", §2 Related Work "~1.2p", §3 Methodology "~2.8p"은 PAGE_BUDGET 값(1.7p, 1.1p, 2.7p)과 각각 차이가 있다.

**권장 수정**: BLUEPRINT §2의 섹션별 수치를 PAGE_BUDGET §1 값과 일치시키거나, 두 문서 중 하나를 AUTHORITATIVE로 지정하고 나머지에 "PAGE_BUDGET.md 참조" 위임 문구를 붙여라. 두 문서가 서로 다른 수치를 갖는 채로 병존하면 Phase 5 drafter가 어느 것을 따라야 할지 모른다.

**해결 상태**: OPEN

---

### BLK-002

**Severity**: BLOCKER
**Artifact**: BLUEPRINT §5.3 §3.2 Overall Architecture

**문제**: "Encoder는 teacher path gradient로만 학습함을 명시(latent.detach() for student)"라는 서술이 잘못되었다.

271_CONFIG_TRUTH.md §VI "Output Discrepancy Loss" 행과 §VIII "Total loss": `reconstruction_loss + normal_loss + adaptive_grl_weight * grl_cls_loss + adaptive_fm_weight * fm_loss`. Total loss에 `normal_loss`(OD loss)가 포함되어 있으며, 이 normal_loss는 `||teacher_out.detach - student_out||²`에서 teacher output이 detach되므로 gradient는 student decoder에만 흐른다. Encoder(shared)는 teacher reconstruction loss를 통해 학습되며, student decoder는 encoder로부터 `latent_visible.detach()`를 받으므로 student loss gradient는 encoder에 흐르지 않는다.

따라서 "encoder는 teacher path gradient로만 학습한다"는 서술 자체는 맞다. 그런데 블루프린트 §5.3의 해당 문장은 "(latent.detach() for student)"라는 괄호 보충으로 그 근거를 `latent.detach()`에서 찾는다. 이것은 student decoder가 encoder latent를 detach해서 받는다는 구현 사실이고, 실제로 그렇다(RESEARCH_SYNTHESIS 표A "Transformer Encoder" 행: "encoder는 teacher path gradient만으로 학습 (student는 latent_visible.detach())").

표면적으로는 맞는 서술처럼 보이지만, Fig. 2 설계(§5.3)에서 "force_mask_anomaly와 score 수식을 figure 내 레이블로 연결"이라는 계획은 §3.6의 추론 score 수식을 아키텍처 다이어그램에 연결하겠다는 것이다. 271_CONFIG_TRUTH §VIII score formula에서 FM은 score에서 제외(scoring.py:237 fm_active=False)되는데, §5.7(§3.6 Anomaly Scoring)에서는 이를 명기했다. 이 부분은 일관성이 있다.

실제 BLOCKER는 아래다: §5.3에서 "5개 컴포넌트 색 구분: (1) Patch Embedding, (2) Transformer Encoder (shared), (3) Teacher Decoder (3L), (4) Student Decoder (2L), (5) GRL + AnomalyClassifierHead"라고 명기했는데, RESEARCH_SYNTHESIS 표A 및 271_CONFIG_TRUTH §VI에 따르면 GRL과 AnomalyClassifierHead는 student decoder 마지막 층 hidden에 적용된다. 그런데 "shared" Encoder에 대해 색 구분 컴포넌트가 별도인 것은 맞지만, NOTION_DIGEST I-3 forward flow에는 GRL이 Student Output Projection 다음에 위치한다고 기재되어 있고, 271_CONFIG_TRUTH §VI "GRL classifier" 행에는 "model.py:1150-1154 — called on student hidden"이라고 되어 있다. student decoder 마지막 층 hidden이므로 Output Projection 이전이다. 이 위치 불일치가 Fig. 2 설계에서 시각적 오류로 이어질 수 있다. Phase 5 figure 작성 시 직접적인 오류를 유발한다.

더 심각한 BLOCKER: §5.3은 "학습/추론 두 패널(또는 학습 패널 하나에 'inference-time' 비활성 표시)"을 제안하는데, GRL classifier는 추론 시 비활성("추론 시 라벨 불사용. GRL classifier 비활성" — §5.7)이다. 이것이 Fig. 2에 명시되어야 하는데 §5.3 설계 계획에는 이 항목이 빠져 있다.

**권장 수정**: §5.3 Fig. 2 설계에 GRL classifier의 추론 시 비활성(dashed box 또는 "training only" 주석) 명시를 추가하라. GRL의 위치(student hidden, output projection 이전)를 figure 레이블 계획에 명기하라.

**해결 상태**: OPEN

---

### BLK-003

**Severity**: BLOCKER
**Artifact**: BLUEPRINT §5.3 §3.2 / §6.3 §4.1.2

**문제**: d_model 서술에 271 실험과 모순이 있다.

BLUEPRINT §5.4 §3.3 및 §9.1 Notation 표에서는 "d_model=dynamic(F→{128,192,256,384,512}, cap=512)"라고 기술한다. 이것은 NOTION_DIGEST I-3의 d_model 매핑 공식("min{d ∈ {128,192,256,384,512} : d ≥ 10F}, cap=512")과 일치한다.

그러나 BLUEPRINT §6.3 §4.1.2 Implementation Details에는 "d_model=dynamic(F→{128,192,256,384,512}, cap=512)"라고 올바르게 기재되어 있으면서, §5.5 §3.4 Asymmetric Teacher-Student Decoders 첫 bullet에서 "d_model=dynamic"이라고 쓰는 대신 "d_model 동적 결정(min{128,192,256,384,512 : d ≥ 10F}, cap 512)"으로 반복 정의한다.

271_CONFIG_TRUTH §II에서 `d_model=512`는 config의 기본값이며, 실제 각 entity의 런타임 d_model은 동적으로 결정된다. 문제는 §6.3에서 "d_model=dynamic" 뒤에 parenthetical "(F→{128,192,256,384,512}, cap=512)"를 쓰고 있는데 SMD의 d_model이 논의 대상이다:

NOTION_DIGEST I-3 d_model 표에서 "SMD: F=38, d_model=384"이다. 그러나 271_CONFIG_TRUTH §II에서는 SMD machine들의 num_features가 29–36이며 38이 아니다(§III §3a "SMD/machine-3-3: 36"이 최대). 따라서 F=38인 SMD machine은 실제로 없으므로 d_model=384인 SMD entity도 없다. 실제로는 F=29–36이므로 d_model = min{d: d≥10×36}=384는 machine-3-3에만, 나머지는 F≤35이면 d_model=512(≥10×36=360인 최소값은 384, 10×35=350→384, 10×30=300→384... 10×29=290→384)... 실제로 {128,192,256,384,512}에서 d≥290인 최소값은 384이므로 SMD 전 machine이 d_model=384이다. NOTION_DIGEST 표의 "F=38"은 raw features(NaN drop 전)이고 실제 투입은 29–36이다. 블루프린트가 이 미묘한 점을 혼동없이 서술하기 위해서는 "SMD의 실제 F는 29–36(constant 제거 후)"임을 어딘가에 명기해야 한다. 현재 §6.2 §4.1.1에서 "SMD(28 machines, 29–36 features per machine)"이라고 올바르게 쓴 부분이 있으나, Implementation Details §6.3의 d_model 서술은 이 실제 F 범위와 연결되지 않은 채 "F에 따라 동적 결정"이라고만 되어 있다.

더 심각한 것은 BLUEPRINT §5.5에서 "Transformer Encoder: 4층, Pre-Norm, GELU, d_model=dynamic, nhead=8, dim_feedforward=4×d_model, dropout=0.15"라고 하는데, 271_CONFIG_TRUTH §VIII 에서 "dim_feedforward: 2048" (for d_model=512)이다. SWaT/WaDi의 경우 d_model=512이므로 4×512=2048이 맞다. 하지만 PSM(F=25)의 경우 d_model=256이므로 dim_feedforward=4×256=1024이다. 이 "4×d_model" 공식이 맞는지 271 config에서 확인하면: 271_CONFIG_TRUTH §II `dim_feedforward=2048`이고 `d_model=512`이므로 2048/512=4가 맞다. 그러나 이것은 config에서 dim_feedforward가 고정 2048이 아니라 d_model에 연동되어 설정된다는 것을 함의한다. NOTION_DIGEST I-3에서도 dim_feedforward = 4×d_model로 확인된다. 서술 자체는 맞다.

**권장 수정**: SMD 실제 투입 F 범위(29–36)가 d_model 표에서 "F=38" NOTION 오기와 혼동되지 않도록 §6.3 또는 Appendix §C.1 d_model 매핑 표에서 "SMD: F=29–36(constant 제거 후), d_model=384"로 명기하라.

**해결 상태**: OPEN

---

### BLK-004

**Severity**: BLOCKER
**Artifact**: BLUEPRINT §5.6 §3.5 GRL λ sigmoid ramp-up 서술

**문제**: GRL adaptive lambda 공식이 config 사실과 불일치한다.

BLUEPRINT §5.5 §3.4에서 "GRL λ sigmoid ramp-up: λ = 2/(1+exp(-10p))−1, p = max(0,(epoch-250)/250)"이라고 쓴다. 이 공식은 NOTION_DIGEST I-4 "GRL λ Sigmoid Ramp-up (Ganin et al. 2016 schedule)" 표에 있는 것이다.

그러나 271_CONFIG_TRUTH §VIII GRL Details에서 "Lambda balancing: Adaptive (grl_adaptive_lambda=True): lambda = ||grad_main|| / (||grad_grl|| + 1e-4), clamped [0, 10], smoothed via prev-epoch average"라고 하며, RESEARCH_SYNTHESIS 표A GRL 행에서 "λ_GRL = clamp(‖∇L_main‖/(‖∇L_GRL‖+1e-4), 0, 10), 직전 epoch 값 적용 (trainer.py:752–763)"이라고 확정되어 있다.

즉, 271 실험에서 실제로 사용된 GRL lambda는 sigmoid ramp-up이 아니라 trainer inline grad-ratio adaptive lambda이다. Sigmoid ramp-up(Ganin et al. 2016 schedule)은 NOTION_DIGEST에 서술되어 있고 코드에도 존재할 수 있으나, 271 실행 경로에서의 실제 GRL lambda는 adaptive grad-ratio이다.

RESEARCH_SYNTHESIS 표A GRL 행의 명기: "backward: -lambda × grad" — 이 lambda가 어떤 공식으로 계산되는지가 핵심이며, 271에서는 sigmoid가 아니라 adaptive grad-ratio이다. 271_CONFIG_TRUTH §VII #25도 "bare adaptive_lambda=True (discriminator 전용) — grl_adaptive_lambda와 fm_adaptive_lambda와 별개"임을 명기하고, §VIII GRL Details에서 trainer inline grad-ratio를 명시한다.

BLUEPRINT §5.5에서 "warmup 종료 직후 GRL λ sigmoid ramp-up"이라고 서술하는 것은 271 실제 동작과 다른 설명이다. 이 오류가 논문 §3.4에 그대로 들어가면 reviewer가 코드와 대조했을 때 재현 불가능한 서술이 된다.

추가로 BLUEPRINT §9.2 수식 금지사항에 "GRL adaptive λ: 수식 제시 시 'VQGAN-style'로 귀속 삭제 — trainer inline grad-ratio로 표기 (BLK-001 정정 반영)"이라고 되어 있다. 이것은 정정 의도가 있음을 보여주나, §5.5에서 여전히 "sigmoid ramp-up: λ = 2/(1+exp(-10p))−1"이 남아 있어 §9.2의 방침과 모순된다.

**권장 수정**: §5.5의 warmup 이후 GRL lambda 서술을 "λ_GRL_eff = adaptive_lambda × grl_loss_weight(0.2), adaptive_lambda = clamp(‖∇L_main‖/(‖∇L_GRL‖+1e-4), 0, 10) [직전 epoch 값 적용]"으로 교체하라. sigmoid ramp-up 수식은 논문 §3.4에서 삭제하라.

**해결 상태**: OPEN

---

### BLK-005

**Severity**: BLOCKER
**Artifact**: BLUEPRINT §6.3 §4.1.2 Implementation Details

**문제**: baseline epoch 수 서술이 사실과 반대이다.

BLUEPRINT §6.3에서 "1문장: 'Unsupervised baselines trained for 10 epochs; weakly-supervised for 50 epochs.'"라고 명기한다.

EXPERIMENT_PROTOCOL_TRUTH §④-실행 3항(r3 정정, RB-1): "unsupervised baseline 22종 = 10 epochs (baseline_common.py:272... '2026-06-06: unsupervised unified to 10') / weakly-supervised 4종 = 50 epochs"이다.

이것은 블루프린트 서술과 일치한다. 그러나 같은 절에서 "Best epoch selected by pak_auc_f1 on the test split"이라고 쓰는데, EXPERIMENT_PROTOCOL_TRUTH §④ M-3에서 "best epoch도 test 지표로 선정된다 — 즉 test-set model selection이다 (전 모델 동일 조건)"이라고 명기되어 있고, "반드시 공개해야 할 프로토콜 사실"이라고 강조되어 있다.

그러나 블루프린트의 해당 서술은 "Best epoch selected by pak_auc_f1 on the test split"으로 정직하게 명기하고 있다. 이것은 맞다. 문제는 이 사실이 §6.3의 짧은 1문장으로만 처리되고 있어, 이 test-set model selection이 논문에서 어떻게 정당화될 것인지에 대한 방어 계획이 없다는 점이다(EXPERIMENT_PROTOCOL_TRUTH REQUEST-4).

더 심각한 문제: MAE는 500 epochs인데 unsupervised baseline은 10 epochs이다. 이 비대칭(500 vs 10)은 reviewer의 공정성 공격 대상이 된다. 블루프린트 어디에도 이 비대칭에 대한 방어 계획이 없다.

**권장 수정**: §6.3 또는 §6.5 baselines 설명에 epoch 수의 비대칭(MAE 500 / unsupervised 10 / weakly-supervised 50)과 이에 대한 방어 논리를 1–2문장으로 명기하라. 또한 §15 리뷰어 방어 시나리오에 "unsupervised baseline 10 epochs vs MAE 500 epochs" 공정성 공격 대응 시나리오를 추가하라.

**해결 상태**: OPEN

---

### MAJ-001

**Severity**: MAJOR
**Artifact**: BLUEPRINT §4.4 §2.3 self-distillation 계보 서술

**문제**: "self-distillation" 용어 귀속에서 SDMAE 원문과 불일치하는 서술이 있다.

BLUEPRINT §4.4에서 "Self-distillation 계보: Zhang et al. [TPAMI 2022]이 'self-distillation'을 도입, SDMAE(Ristea et al., CVPR 2024)가 이를 anomaly detection의 공유 encoder 이중 decoder 구조에 처음 적용"이라고 쓴다.

ANCHOR_SDMAE_DOSSIER §2 Verification Log: "'self-distillation' 용어의 선행 귀속 [101] = Zhang et al., TPAMI 2022 — 확인 (fixer r2, S-m2) arxiv.org/html/2306.12041v2 reference list bib101 직접 확인"이다.

그러나 SDMAE의 구조는 "공유 encoder 이중 decoder"가 아니다. ANCHOR_SDMAE_DOSSIER §3.1: "Teacher decoder: CvT 3블록, Student decoder: CvT 1블록 — teacher decoder에서 중간 분기(branch-off) 형태로 연결". 즉 SDMAE는 공유 인코더에서 teacher/student decoder가 독립된 것이 아니라, student decoder가 teacher decoder에서 분기된다. 본 논문의 비대칭 Teacher–Student(독립 decoder 2개)와 구조가 다르다.

BLUEPRINT가 SDMAE를 "공유 encoder 이중 decoder 구조"로 묘사하면 실제 SDMAE 구조와 다른 서술이 된다. 이것은 R21(용어 계보 방어)에서 reviewer가 SDMAE 논문과 대조할 때 틀린 사실로 지적받을 수 있다.

또한 BLUEPRINT §4.4의 계보 설명에서 옵션 C 초안 문장: "In this work, we extend analogous self-distillation principles to the time-series domain, augmented with a contaminated semi-supervised framework that leverages labeled anomalies through targeted masking and gradient-based information suppression." — "extend ... principles"는 R22 원칙("analogous ... applying/extending"으로 coining 표기 금지)에 부합하나, SDMAE가 "공유 encoder 이중 decoder"라는 잘못된 전제에서 기술되면 plagiarism 위험보다 사실 오류가 더 심각하다.

**권장 수정**: §4.4에서 SDMAE 구조를 "teacher decoder에서 student decoder가 분기되는 branch-off 구조"로 정정하라. "공유 encoder 이중 decoder"라는 표현을 삭제하라. 본 논문의 독립 비대칭 decoder와의 구조적 차이("독립 별도 decoder vs. branch-off 분기")를 각주에서 명기하면 R21 방어로도 활용 가능하다.

**해결 상태**: OPEN

---

### MAJ-002

**Severity**: MAJOR
**Artifact**: BLUEPRINT §6.2 §4.1.1 Datasets — 데이터셋 목록

**문제**: 논문 포함 데이터셋 목록이 현재 실험 완료 상태와 불일치하며, 총 entity 수 서술이 잘못되었다.

BLUEPRINT §6.2에서 "6계열 데이터셋: SWaT(A1+A2, 45 features), WaDi A1(123 features), WaDi A2(123 features), PSM(25 features), SMD(28 machines, 29–36 features per machine), SMAP(54 channels, 25 features), MSL(27 channels, 55 features). Simulation/Exathlon 제외"라고 한다.

그러나 이것은 6계열이 아니라 7계열이다(SWaT, WaDi A1, WaDi A2, PSM, SMD, SMAP, MSL). 여기서 WaDi A1과 A2를 "WaDi"로 묶으면 6계열이 된다. RESEARCH_SYNTHESIS §④ 데이터셋 표는 6행(SWaT, WaDi A1, WaDi A2, PSM, SMD, SMAP, MSL)을 나열하되 WaDi를 2행으로 표기한다. BLUEPRINT §6.2의 "6계열" 표현은 WaDi A1/A2를 1계열로 합산한 것으로 보이나 feature 수와 train/test 길이가 달라 논문 Table 1에서 별도 행으로 처리될 것이다. 이 모호함을 해소해야 한다.

더 중요한 문제: RESEARCH_SYNTHESIS §① 정정(fixer-2, MAJ-007): "현재 완료된 MAE 271 entity는 37개 (SMD 22/28, SMAP 5/54, MSL 5/27 — 잔여 실행 진행 중)". BLUEPRINT에는 이 "잔여 entity 미완료" 사실이 전혀 언급되어 있지 않다. §6.4 TABLE 2 설계에서 "행 = 비교 방법"이라고 쓰지만, 미완료 entity의 수치는 placeholder([BEST])로 처리되어야 함을 명기하지 않았다. placeholder 정책(R3/A8)은 §0의 frontmatter에 "실험 수치는 전부 placeholder"라고 선언되어 있으므로 구조적으로는 커버되지만, §6.2에서 데이터셋을 소개할 때 "현재 37/113 entity 완료" 상태를 알 수 없다.

**권장 수정**: §6.2 첫 줄에서 "6계열" 대신 "6 데이터셋 계열(WaDi는 A1/A2 두 조건 독립)"이라고 표현하거나, 7계열이면 7계열로 쓰라. 또한 §6.4 Table 2 설계 근처에서 "잔여 entity 완주 후 수치 채움"을 명기하라.

**해결 상태**: OPEN

---

### MAJ-003

**Severity**: MAJOR
**Artifact**: BLUEPRINT §6.3 §4.1.2 Implementation Details

**문제**: SWaT feature 수 재현성 플래그가 blueprint에 전혀 반영되어 있지 않다.

EXPERIMENT_PROTOCOL_TRUTH §⑧ FEEDBACK-7: "학습된 271 SWaT 모델의 입력 차원은 45... 현 machineA의 raw CSV(51 features) + 현행 load_swat_a1a2_raw(constant 제거 코드 없음) 경로는 51을 반환한다. 재실험 전 반드시 feature 수 45 일치 여부를 반드시 확인할 것."

RESEARCH_SYNTHESIS §⑧-6도 "SWaT feature 수 재현성 플래그" 항목에서 동일 경고.

그러나 BLUEPRINT §6.3에서 "SWaT(A1+A2, 45 features)"로만 기재하고, 재현 시 현 환경에서 SWaT feature 수가 51로 반환될 수 있다는 경고를 Appendix 설계에도 본문 어디에도 언급하지 않는다. 이것은 Phase 5 drafter 또는 reader가 "45 features"를 당연히 재현 가능하다고 가정하게 만든다. 재현성 claim(§6.3 코드 공개 서술)이 흔들린다.

**권장 수정**: §6.3 Implementation Details에 "SWaT A1+A2 입력 차원 45 (원본 51 − constant 6개 제거; 재현 시 constant 컬럼 제거 코드 검증 필요)" 한 줄을 추가하라. 또는 Appendix §C.2 Training Pseudocode에 SWaT preprocessing 단계로 명기하라.

**해결 상태**: OPEN

---

### MAJ-004

**Severity**: MAJOR
**Artifact**: BLUEPRINT §5.6 §3.5 Label-Guided Training — GRL focal loss 표기

**문제**: GRL classifier loss 표기에서 표준 focal loss(Lin et al. 2017) 혼용 위험이 있다.

BLUEPRINT §5.6 (C) GRL Anomaly Suppression에서 "focal-style BCE 변형: L_cls = (1/|P_mask|) Σ (1-exp(-BCE))² × BCE_{w+}(logit, y). 표준 focal loss(Lin et al. 2017) 아님 — pos_weight 내장 BCE 기반 변형"이라고 올바르게 명기했다.

그러나 BLUEPRINT §9.2 수식 금지사항에도 "논문에서 'standard focal loss'(Lin et al. 2017) 표기 금지 — 'focal-style BCE variant'로 표기"라고 정확하게 명기했다. 이것은 RESEARCH_SYNTHESIS 표A GRL 행(BLK-004 정정)의 지침과 일치한다.

그런데 §5.6에서 "focal-style BCE 변형"이라고 했지만, 수식 (8)로 번호 매겨진 λ adaptive 공식 계획에서 이 "variant"의 정확한 이름이 아직 결정되지 않았다. Phase 5 drafter가 논문을 쓸 때 "focal loss"라는 표현이 자연스럽게 쓰일 위험이 있다. 플래그는 있으나 positive 지침("이렇게 쓰라")이 구체적이지 않다.

또한, pos_weight 내장 BCE 기반이라는 점에서 Lin et al. 2017과의 차이가 무엇인지를 논문 내 1문장으로 설명해야 할 자리가 계획에 없다. "표준 focal loss가 아님"을 주장하려면 왜 아닌지를 1문장 이상 설명해야 reviewer 공격을 방어할 수 있다.

**권장 수정**: §5.6 또는 §9.2에 "이 변형을 논문에서 'focal-style BCE variant with class-prior pos_weight'로 표기하고, 각주나 1문장으로 Lin et al. 2017 표준 focal loss와의 차이(표준: p_t = sigmoid(logit)·y+(1-sigmoid(logit))·(1-y)로 정의; 본 변형: p_t:=exp(-BCE_{w+}))를 명기한다"는 지침을 추가하라.

**해결 상태**: OPEN

---

### MAJ-005

**Severity**: MAJOR
**Artifact**: BLUEPRINT §6.3 §4.1.2 + §6.6 §4.2 Main Results

**문제**: test-set model selection(oracle best epoch) 방어 계획이 없다.

EXPERIMENT_PROTOCOL_TRUTH §④ M-3: "per-epoch 평가는 test split 위에서 수행되며, best epoch도 test 지표로 선정된다 — 즉 test-set model selection이다 (전 모델 동일 조건)... 논문 experiments 섹션에 반드시 공개해야 할 프로토콜 사실 (숨기면 리뷰어 단골 공격 지점)."

BLUEPRINT §6.3에서 "Best epoch selected by pak_auc_f1 on the test split"이라고 정직하게 명기한다. 그러나 §15 리뷰어 방어 예상 시나리오에 이 항목이 없다. 또한 §6.6 §4.2 분석 텍스트 4구조에도 이 점에 대한 방어 서술 위치가 없다.

Reviewer는 "no separate validation set, test-set model selection"을 공정성 문제로 제기할 것이 확실하다. 전 모델에 동일 적용이므로 공정성은 유지되지만, 이에 대한 방어 논거가 blueprint 어디에도 없다.

**권장 수정**: §15 리뷰어 방어 시나리오에 "test-set model selection (best epoch by pak_auc_f1 on test split) — 전 모델 동일 프로토콜, 별도 validation split 없음, PA%K-AUC 적분형 지표로 epoch-wise overfitting 위험 최소화" 항목을 추가하라.

**해결 상태**: OPEN

---

### MAJ-006

**Severity**: MAJOR
**Artifact**: BLUEPRINT §6.5 §4.1.4 Baselines — SOTA Legacy 기재

**문제**: SOTA Legacy 6개 목록이 EXPERIMENT_PROTOCOL_TRUTH §③과 불일치한다.

BLUEPRINT §6.5에서 "SOTA Legacy 6: GCN-LSTM, Anomaly Transformer, TranAD, USAD, DAGMM*, GDN, OmniAnomaly"라고 7개를 열거하면서 "SOTA Legacy 6"이라는 제목을 쓴다.

EXPERIMENT_PROTOCOL_TRUTH §③: "SOTA legacy 6 (anomaly_transformer, tranad, usad, dagmm, gdn, omnianomaly — 정정 r2: 초판 '7'은 오기, 6이어야 5+3+1+6+7=22로 총계 정합)"이다. 즉 GCN-LSTM은 SOTA Legacy가 아니라 Neural 3(QuoVadisTAD) 카테고리에서 분리된 독립 항목으로, STANDARD_BASELINES에서는 "Neural 3 + GCN-LSTM 1"로 분류된다.

BLUEPRINT의 열거에는 GCN-LSTM(1)이 SOTA Legacy에 들어가 7개로 열거되어 있다. NOTION_DIGEST II-2에서 "SOTA Legacy 7개"라고 되어 있는데, 이것은 r2 정정 이전의 오기다. EXPERIMENT_PROTOCOL_TRUTH r2 정정이 6개로 확정했다. 블루프린트 §6.5가 SOTA Legacy 6이라고 제목을 쓰면서 7개를 나열하는 것은 명백한 내부 모순이다.

NOTION_DIGEST II-2의 "SOTA Legacy 7개" 표는 GCN-LSTM이 포함된 반면, EXPERIMENT_PROTOCOL_TRUTH는 GCN-LSTM을 "GCN-LSTM 1" 독립 항목으로 분리해 "5+3+1+6+7=22" 산식을 구성한다. 블루프린트는 EXPERIMENT_PROTOCOL_TRUTH의 더 최신 분류를 따라야 한다.

**권장 수정**: §6.5의 "SOTA Legacy 6" 목록에서 GCN-LSTM을 제외하고, GCN-LSTM을 Neural 3(QuoVadisTAD) 카테고리 뒤 "SOTA Legacy 이전 도입 독립 모델 1: GCN-LSTM"으로 분리하거나, NOTION_DIGEST 분류대로 SOTA Legacy 7(GCN-LSTM 포함)으로 제목을 수정하라. 어느 쪽이든 22개 합산이 맞아야 한다.

**해결 상태**: OPEN

---

### MAJ-007

**Severity**: MAJOR
**Artifact**: BLUEPRINT §5.6 §3.5 FM Loss 수식 / §9 Notation

**문제**: FM loss 수식에서 FM이 score에서 제외된다는 정보와 논문 서술의 연결이 약하고, 수식 번호 계획에 일관성이 없다.

BLUEPRINT §5.6에서 FM loss를 "(B) Feature Matching (FM) Loss — 훈련 전용 regularizer"로 분리하고, "추론 점수에 포함하지 않음(scoring.py fm_active=False hardcoded)"을 명기한다. 이것은 271_CONFIG_TRUTH §VII #6과 일치한다.

그러나 §5.6의 "총 손실 수식: L_total = L_recon + L_OD + λ_FM_eff × L_FM + λ_GRL_eff × L_GRL"에서 subscript 명칭이 §5.6 본문 기술과 불일치한다. 본문에서는 "L_cls"라고 쓰는 GRL loss를 총 손실 수식에서 "L_GRL"이라고 쓴다. 수식 번호 계획에서 "(6)–(10) 순서대로 L_OD, L_FM, L_cls, λ adaptive 공식, L_total"이라고 했는데 총 손실 수식에서는 L_GRL로 표기하므로 L_cls ≠ L_GRL 혼용이 발생한다.

271_CONFIG_TRUTH §VIII "Total loss": "reconstruction_loss + normal_loss + adaptive_grl_weight * grl_cls_loss + adaptive_fm_weight * fm_loss"에서는 "grl_cls_loss"라는 코드 변수명이 쓰인다. 논문에서 일관된 표기(L_cls 또는 L_GRL 중 하나)를 사용해야 한다.

**권장 수정**: §5.6의 총 손실 수식을 "L_total = L_recon + L_OD + λ_FM_eff × L_FM + λ_GRL_eff × L_cls"로 통일하거나, L_GRL_cls로 표기를 결정하고 §9 Notation 표에 등재하라.

**해결 상태**: OPEN

---

### MAJ-008

**Severity**: MAJOR
**Artifact**: BLUEPRINT §3.2 Para 2 한계 인정 + §5.2 §3.1

**문제**: "labeled anomaly ratio ≤ 6.2%"의 출처와 정확성에 문제가 있다.

BLUEPRINT §5.2에서 "Contaminated semi-supervised setting 정의: train 데이터 D_train = {(W_i, Y_i)} where Y_i ∈ {0,1}^L — 일부 Y_i에 labeled anomaly 존재(ratio ≤ 6.2%), 나머지는 정상"이라고 쓴다.

EXPERIMENT_PROTOCOL_TRUTH §① 표에서 train anomaly ratio: SWaT 1.63%, WaDi A1 0.52%, A2 0.76%, PSM 6.20%, SMAP concat 0.70%, MSL concat 1.70%. 따라서 6.20%(PSM)가 최대값이므로 "≤ 6.2%"는 PSM을 기준으로 한 상한이다. 그러나 이것은 SMAP/MSL의 SMD 개별 machine 수치가 아직 완전히 확인되지 않은 상태("잔여 entity 완주 후")에서의 상한이다. SMD per-machine 비율은 "machine별 상이"로만 기재되어 있으며(§① SMD 행), 만약 일부 SMD machine의 train anomaly ratio가 6.20%를 초과한다면 "≤ 6.2%"는 틀린 상한이 된다.

271_CONFIG_TRUTH §III §3b에서 grl_pos_weight 범위 3.14 (SMAP/T-1) to 999.0 (SMD/machine-1-5)라고 하는데, 999.0은 patch_ratio ≥ 0.001에서 유도된 값이다. 즉 SMD/machine-1-5의 anomaly patch ratio가 매우 낮다는 의미로 해석할 수 있다. 하지만 이것이 train anomaly ratio가 아니라 test anomaly ratio를 기반으로 계산된 값일 수도 있다. PSM의 6.20%가 가장 높은 train anomaly ratio임을 확신하려면 SMD per-machine train anomaly ratio를 확인해야 한다.

**권장 수정**: §5.2의 "ratio ≤ 6.2%"를 "ratio ≤ ~6.2% (데이터셋별: SWaT 1.63%, WaDi 0.52–0.76%, PSM 6.20%, SMAP 0.70%, MSL 1.70% — SMD per-machine 확인 후 수정)"으로 쓰거나, SMD per-machine train anomaly ratio 확인 후 상한을 업데이트하라.

**해결 상태**: OPEN

---

### MAJ-009

**Severity**: MAJOR
**Artifact**: BLUEPRINT §5.7 §3.6 score formula + §5.6 FM

**문제**: 점수 수식의 ε 값이 정본과 불일치한다.

BLUEPRINT §5.7에서 "scaled_disc = disc × (mean_recon + ε)/(mean_disc + ε) [ε = 1e-4]"이라고 쓴다. 271_CONFIG_TRUTH §VIII Anomaly Score 절:
```
recon_mean = mean(recon) + 1e-4
disc_mean  = mean(disc) + 1e-4
scaled_disc = disc * (recon_mean / disc_mean)
student_error = scaled_disc / score_recon_disc_ratio   # = scaled_disc / 4.0
score = recon + student_error
```
ε = 1e-4는 일치한다.

그러나 NOTION_DIGEST I-6 Adaptive Scoring Formula에는 "scaled_disc = disc × (mean_recon / mean_disc)"로 ε이 없다. 이것은 NOTION 스냅샷의 stale이며, 271_CONFIG_TRUTH가 우선이다. 블루프린트가 1e-4를 명기한 것은 맞다.

문제는 §5.7의 수식 표기: "score = recon + scaled_disc / r, r = 4"라고 쓰는데, 수식에서 recon과 scaled_disc의 의미가 모호하다. 271_CONFIG_TRUTH에서 recon = teacher reconstruction error (MSE on masked positions, per timestep)이다. "per timestep"이 맞다면 점수가 타임스텝 단위로 계산된다는 것인데, §5.7 "Point-level aggregation: 각 timestep을 덮는 (window, patch) 쌍들의 score 평균"이라고 되어 있다. 이 집계가 수식 (13)으로 설계되어 있는데, 수식 (11)(12)의 recon/disc가 이미 per-timestep인지 per-patch인지가 불명확하다.

RESEARCH_SYNTHESIS 표A 추론 점수 행: "각 패치 p의 score: recon_p, disc_p = ||o_T^p - o_S^p||²"라고 하므로 per-patch이다. 이후 patch score → point score 집계가 별도로 이뤄진다. 블루프린트 §5.7의 수식 (11)–(12)가 per-patch인지, 수식 (13)이 patch→point 집계인지 명확히 해야 한다.

**권장 수정**: §5.7 수식 설계에서 "수식 (11): per-patch scaled_disc, 수식 (12): per-patch final score, 수식 (13): patch→point aggregation"이라고 명기하고, (13)이 EXPERIMENT_PROTOCOL_TRUTH §④-실행 2항("mean 집계, mean 산식 evaluator.py:278-280 bincount-합/coverage")과 일치함을 확인하라.

**해결 상태**: OPEN

---

### MAJ-010

**Severity**: MAJOR
**Artifact**: BLUEPRINT §6.7 §4.3 Ablation Study — Table 3 설계

**문제**: Table 3에서 "6. w/o Teacher Warmup: warmup=0 (teacher/student 동시 학습, 단 ablation 실험 미실행 → REQUEST-F, placeholder)"이라고 명기하는데, RESEARCH_SYNTHESIS REQUEST-F(NOTE-001 격상)에서 "warmup ablation 부재를 CRITICAL RISK로 격상"했다.

BLUEPRINT §5.5 §3.4에서도 "CRITICAL NOTE (Phase 5): warmup ablation 미존재(RESEARCH_SYNTHESIS REQUEST-F). 논문에서 warmup을 독립 기여로 주장하지 않고, '학습 안정화를 위한 단계적 활성화'로 서술하여 ablation 없이도 방어 가능한 수준으로만 언급"이라고 밝혔다.

그러나 동시에 Contribution bullet 3번(§11 결정 ①)에서 "The asymmetric Teacher(3L)–Student(2L) decoder structure, trained with teacher-only warmup, establishes a stable normal reconstruction reference before the student and GRL are activated"라고 warmup을 contribution bullet에 명기한다. Warmup을 contribution bullet에 올리면 reviewer는 ablation을 요구할 것이다.

더불어 Table 3에 warmup 변형을 행으로 두면서 "placeholder"라고 하는 것은 논문 제출 시 실험 수치가 없다는 의미다. Phase 5에서 실험을 수행하지 못하면 이 행 전체를 삭제해야 한다. 삭제 시 Contribution bullet 3번도 warmup 관련 서술을 제거해야 하는 연쇄 수정이 필요하다.

**권장 수정**: Contribution bullet 3번에서 "teacher-only warmup"을 contribution의 핵심으로 서술하지 말고, "안정적인 teacher 기준 확립을 위한 단계적 학습 절차"로 downgrade하고 ablation이 없어도 방어 가능한 수준으로 Bullet 3을 수정하라. Table 3 행 6의 존재 여부를 Phase 5 실험 가용성에 명시적으로 conditional로 표기하라.

**해결 상태**: OPEN

---

### MAJ-011

**Severity**: MAJOR
**Artifact**: BLUEPRINT §6.3 §4.1.2 — batch_size 오기

**문제**: Implementation Details의 batch_size가 정본과 불일치한다.

BLUEPRINT §6.3에서 "batch_size=1024"라고 쓴다. 이것은 271_CONFIG_TRUTH §II `batch_size=1024`와 일치한다.

그러나 NOTION_DIGEST I-4 "검증된 사실 후보 — 학습 파라미터"에서 "batch_size=512 (Set C)"라고 기재하고, I-8 핵심 파라미터 목록에서도 "batch_size=512"라고 한다. NOTION_DIGEST II-2b §1.2.2 Neural 3에서도 "epochs=10·bs=512(paper)"라고 한다.

271_CONFIG_TRUTH §II는 metadata에서 직접 실측한 `batch_size=1024`를 확정값으로 제시하고 있으며, RESEARCH_SYNTHESIS §③ 표A도 별도 batch_size 기재가 없어 config truth를 따른다. NOTION의 512는 "Set C preset default"이고 271이 override해서 1024를 사용한다(RESEARCH_SYNTHESIS §⑥ N2: "Notion batch_size=512 (Set C preset), 271 metadata batch_size=1024 (override)").

블루프린트 §6.3의 "batch_size=1024"는 맞다. 그런데 baseline들은 "bs=512(paper)"를 사용하므로, 논문에서 MAE batch_size=1024와 baseline batch_size=512의 차이가 공정성 문제로 제기될 수 있다. 블루프린트는 이 차이에 대한 방어 계획이 없다.

**권장 수정**: §6.3에서 baseline 학습 설정과 MAE 학습 설정의 batch_size 차이(MAE=1024 vs baseline=512)를 명기하고, 이 차이가 비교 공정성에 미치는 영향을 §15 방어 시나리오에 추가하라.

**해결 상태**: OPEN

---

### MAJ-012

**Severity**: MAJOR
**Artifact**: BLUEPRINT §5.2 §3.1 — leave-one-out 추론 비용 서술

**문제**: §5.7에서 "비용(FLOPs ~50×)은 명시적으로 인정"이라고 쓰는데, RESEARCH_SYNTHESIS 표A에서 "forward 연산량(FLOPs)이 단일-pass 대비 ~50×라는 비용 한계는 유효 (발표자료 p13에서 공개; batch 확장은 wall-clock 병렬화일 뿐 연산량을 줄이지 않음)"이라고 확인된다. 이것은 맞다.

그러나 §7 Conclusion에서 "한계 1문장: 50×FLOPs inference 비용. Complementary masking(7-pass)으로 경감 가능한 방향 언급"이라고 쓴다. 271_CONFIG_TRUTH §VI에서 `eval_complementary_masking=False`이고 §VII #12에서 "Complementary masking at inference — INACTIVE"이다. Complementary masking은 비활성화된 option으로, 이를 "향후 연구"로 언급하는 것은 괜찮다. 그러나 "7-pass"라는 구체적인 숫자는 `eval_complementary_k=7` (config)에서 온 것인데, 이 값이 271에서 비활성화된 INACTIVE 옵션의 설정값이라는 것을 논문에서 어떻게 제시할지 불명확하다. "7-pass 옵션이 코드에 존재하지만 271에서 미사용"이라고 서술해야 정확하다.

**권장 수정**: §7 Conclusion의 "Complementary masking(7-pass)" 서술에 "코드에 구현되어 있으나 본 실험에서는 미사용 — 향후 연구에서 inference cost 대 accuracy tradeoff 탐색 가능"이라는 수식어를 붙이는 것을 지침으로 명기하라.

**해결 상태**: OPEN

---

### MINOR-001

**Severity**: MINOR
**Artifact**: BLUEPRINT §4.4 §2.3 — TFMAE 인용

**문제**: §4.4에서 "TFMAE(Fang et al., ICDE 2024): 시계열 MAE 사례 — 단 1문장 괄호 인용으로 처리"라고 하면서, §4.2 §2.1에서도 "SOTA Legacy 6: ... TFMAE"가 비지도 방법 계보에 포함되어 있다. §2.1과 §2.3에서 TFMAE가 두 번 나오는 구조에 대한 처리 방침이 없다. Related Work 소절 간 "비지도 계열 사례(§2.1)"와 "시계열 MAE 사례(§2.3)" 양쪽에 같은 모델이 언급되는 것은 R1 MECE를 약하게 위반할 수 있다.

**권장 수정**: §2.3에서 TFMAE를 시계열 MAE 사례로 인용할 때 "§4 Experiments baselines 참조"로 forward reference 처리하거나, §2.1에서 계열 인용 시 TFMAE를 빼고 §2.3에서만 1회 언급하는 방침을 명기하라.

**해결 상태**: OPEN

---

### MINOR-002

**Severity**: MINOR
**Artifact**: BLUEPRINT §6.2 §4.1.1 — SWaT excl22 수치 기준

**문제**: §6.2에서 "수치 기준: A1A2_excl22 entity headline (metrics.pak_auc_f1 = 0.62899)"이라고 명기한다. RESEARCH_SYNTHESIS §④ excl22 부분: "A1A2_excl22 entity 자체 headline metrics.pak_auc_f1은 0.62899(best epoch을 excl22_pak_auc_f1로 별도 선정 — 271_CONFIG_TRUTH §IV r2 주석 정합). 두 값 모두 실존 — 논문 표가 어느 쪽 기준인지는 Phase 3 결정 사안 (혼용 금지)." BLUEPRINT §11 결정 ③에서 0.62899로 확정했다. 이것은 올바른 결정이다.

단, 이 수치가 SWaT 모델이 추가 실험 없이 변하지 않는다는 전제 하에서만 유효하다. EXPERIMENT_PROTOCOL_TRUTH FEEDBACK-7(SWaT feature 수 재현성 플래그)이 해소되지 않으면, SWaT 실험을 재실행했을 때 다른 수치가 나올 수 있다. 그런데 블루프린트에서는 이 수치를 "기준 선택 기록"으로만 표기하고(frontier 정책: placeholder가 아닌 "결정된 기준값 표기"), 재실험 시 업데이트 여부에 대한 지침이 없다.

**권장 수정**: §11 결정 ③에 "이 수치는 현재 271 실험의 SWaT excl22 결과 기준. SWaT feature 수(45) 재현 확인 후 변동 시 업데이트 필요"를 명기하라.

**해결 상태**: OPEN

---

### MINOR-003

**Severity**: MINOR
**Artifact**: BLUEPRINT §5.5 §3.4 — warmup epochs during warmup student forward

**문제**: §5.5에서 "epoch < 250에서 student forward는 수행되나 discrepancy/FM/GRL 손실 비활성(teacher_only=True)"이라고 쓴다. 이것은 NOTION_DIGEST I-4 "Epoch 0~249: 단계... Student forward는 수행되지만, loss.py의 teacher_only=True 플래그로 모든 discrepancy/FM/GRL 손실 항이 비활성화"와 일치한다.

그런데 271_CONFIG_TRUTH §VIII Training에서 "Teacher-only warmup: 250 epochs (first 250 epochs train teacher decoder only; student frozen)"이라고 "student frozen"이라는 표현이 쓰인다. "student forward는 수행되나 손실 비활성"과 "student frozen"은 서로 다른 표현이다. "frozen"이 gradient 흐름이 없다는 의미라면 forward 수행 여부와 무관할 수 있으나, Phase 5 drafter가 논문을 쓸 때 "student frozen"이라는 표현을 그대로 쓰면 "forward가 완전히 중단된다"는 오해를 줄 수 있다.

**권장 수정**: §5.5에 "warmup 기간 중 student decoder forward는 수행되지만 backward(gradient)는 흐르지 않음 — loss.py teacher_only=True 게이트로 student 관련 loss항 0"이라고 명확하게 기술하라.

**해결 상태**: OPEN

---

### MINOR-004

**Severity**: MINOR
**Artifact**: BLUEPRINT §6.4 §4.1.3 — PA F1 threshold 표기

**문제**: §6.4에서 "PA F1(K=0, oracle best-F1 threshold)은 보조 지표 — '(oracle)' 명시 후 병기"라고 쓴다. EXPERIMENT_PROTOCOL_TRUTH §⑧ REQUEST-1 RESOLVED에서 "PA F1 → pa_0_f1(F1-최적 threshold)만 존재하므로 옵션 (i) 채택 시 본문에 'PA F1은 F1-최적 threshold 기준' 명기 필요"라고 확정되었다. 블루프린트의 "(oracle)" 표기 방침은 이 확정 사항과 일치한다.

단, 블루프린트 §6.4의 "Affiliation F1 (AR threshold 기반). threshold 방어: 'not oracle'"이라는 서술에서 Affiliation F1에 사용하는 threshold가 AR threshold(`affiliation_f1_ar`)인지 F1-최적 threshold(`affiliation_f1`)인지 명기가 없다. EXPERIMENT_PROTOCOL_TRUTH REQUEST-1 RESOLVED에서 "R30 정합 보고안: affiliation-F1 → affiliation_f1_ar 사용 가능"이라고 했는데 블루프린트가 "AR threshold 기반"이라고 쓴 것은 일치한다. 그러나 §9.1 Notation 표에 Affiliation F1에 사용되는 threshold 변수가 없어, Phase 5 drafter가 어느 변수를 써야 할지 모를 수 있다.

**권장 수정**: §6.4에 "Affiliation F1 사용 키: affiliation_f1_ar (AR threshold 기준, evaluator.py:809-813)"을 명기하라.

**해결 상태**: OPEN

---

### MINOR-005

**Severity**: MINOR
**Artifact**: BLUEPRINT §11 결정 ② Setting 명칭 / Phase 4 연계

**문제**: "contaminated semi-supervised"가 기존 문헌에서 쓰인 용어인지 확인이 필요하다.

BLUEPRINT §11 결정 ②에서 "'contaminated semi-supervised'는 본 논문의 train protocol 특성을 정확히 서술하며, 기존 문헌에서 지배적으로 쓰인 명칭이 아니어서 새로운 포지셔닝이 된다"라고 쓴다.

이 명칭이 "기존 문헌에서 지배적으로 쓰인 명칭이 아님"이라는 주장은 Phase 4 인용 검색에서 검증되지 않은 사항이다. 만약 기존 TSAD 또는 anomaly detection 문헌에서 "contaminated semi-supervised"를 이미 특정 의미로 쓰는 논문이 있다면, 본 논문의 포지셔닝이 달라진다. 이 검증이 Phase 4 연계 항목으로 명시되어야 한다.

**권장 수정**: §11 결정 ② 말미에 "Phase 4에서 '(contaminated semi-supervised) time series anomaly detection' 검색으로 기존 사용 사례 확인 필요"를 명기하라.

**해결 상태**: OPEN

---

### MINOR-006

**Severity**: MINOR
**Artifact**: BLUEPRINT §3.1 Thesis — "최초의 단일 모델" 주장

**문제**: 논제에서 "이 labeled anomaly를 ... 세 독립 경로로 end-to-end로 통합하는 최초의 단일 모델을 제안한다"라고 쓴다.

"최초(first)"라는 주장은 Phase 4에서 검증되어야 할 미지의 영역이다. RESEARCH_SYNTHESIS §② §②-6: "INFERENCE: 다변량 시계열에서 labeled anomaly를 표현 학습의 기울기에 직접 통합하는 end-to-end 첫 번째 다변량 TSAD 모델이다"라고 INFERENCE 등급으로 표기했다. 즉 Phase 1에서도 FACT가 아닌 INFERENCE로 분류한 주장이 BLUEPRINT의 Thesis에 확정적으로 들어가 있다.

이것은 Phase 4 관련 논문 검색 전에 "최초"를 단언하는 것으로, 반증 논문이 발견될 경우 Thesis 전체가 흔들린다.

**권장 수정**: §3.1 Thesis의 "최초의 단일 모델"을 "최초의 통합 단일 모델(to our knowledge)"로 완화하거나, Phase 4 관련 논문 검색으로 "최초" 주장을 검증하도록 Phase 4 연계 항목에 명시하라.

**해결 상태**: OPEN

---

### NOTE-001

**Severity**: NOTE
**Artifact**: BLUEPRINT §4.3 §2.2 — DAGMM provenance

**문제**: §6.5에서 "DAGMM는 'DAGMM (simplified variant, following [TranAD repo])'으로 표기, 각주에 'GMM energy 제거' 명시"라고 한다. 이것은 RESEARCH_SYNTHESIS §⑥ "DAGMM 구현 provenance: TranAD repo reimplementation (GMM energy 제거 simplified variant)"를 반영한다. 그러나 §4.2 §2.1의 비지도 4유형에서 DAGMM이 "재구성 기반(DAGMM, ...)"으로 분류될 때, 이것이 "simplified variant (GMM energy 제거)"임을 related work에서 어떻게 처리할지 방침이 없다. §4.2에서 클러스터 인용으로 처리하면 괄호 안에 "DAGMM"이 들어가는데, 이것이 원논문 DAGMM(Zong et al. ICLR 2018)을 가리키는지 simplified variant를 가리키는지 불명확해진다.

**권장 수정**: §4.2 비지도 4유형 클러스터 인용에서 DAGMM은 "Zong et al. [DAGMM]"으로 원논문만 인용하고, baseline 설명(§4.1.4)에서만 "simplified variant"임을 각주로 처리하는 방침을 명기하라.

**해결 상태**: OPEN

---

### NOTE-002

**Severity**: NOTE
**Artifact**: BLUEPRINT §5.6 §3.5 / §9.2 — focal-style BCE variant의 plagiarism 위험

**문제**: NOTION_DIGEST I-5에서 "(1-p_t)^2 × BCE_{w+}(l_p, y_p) where p_t = exp(-BCE)"라는 수식이 verbatim 서술로 제시되어 있다. 이것이 Notion 페이지 원문의 수식이며, 논문에 그대로 들어갈 경우 Notion 서술이 "원저자가 직접 쓴 것"이므로 plagiarism 문제는 없다. 그러나 이 수식이 SDMAE나 Lin et al. 2017 등 외부 논문의 수식을 변형한 것이라는 설명이 없다. 독창적인 수식임을 논문에서 명확히 해야 한다.

**권장 수정**: §5.6에서 이 focal-style 변형이 본 논문에서 새롭게 설계된 것임을 1문장으로 명기하라: "We design a focal-style variant based on BCE with class-prior pos_weight (Section 5.6), rather than the standard focal loss [Lin et al. 2017]."

**해결 상태**: OPEN

---

### NOTE-003

**Severity**: NOTE
**Artifact**: BLUEPRINT §10 모델명 후보 — plagiarism 위험

**문제**: 모델명 후보 중 "TS-SDMAE (Time-Series Self-Distilled MAE)"는 SDMAE(Ristea et al. CVPR 2024)와 이름이 과도하게 유사하다. 블루프린트도 "SDMAE 유사도 지나치게 부각 (R9 위험)"이라고 단점으로 명기했다. 그러나 이것이 단순 R9 이슈를 넘어서 명명 자체에서 SDMAE와의 혼동을 일으킬 수 있는 potential plagiarism/naming conflict 위험을 갖는다. Elsevier 심사 과정에서 "이름이 SDMAE와 너무 유사하다"는 지적을 받을 수 있다.

**권장 수정**: TS-SDMAE를 모델명 후보에서 제외하거나, DECISION_LOG에서 최종 선택 시 이 후보를 배제하도록 권장 사항을 명기하라.

**해결 상태**: OPEN

---

## 3. 검증 통과 항목 (clean)

- placeholder 정책(R3/A8): frontmatter에서 "실험 수치는 전부 placeholder — [X.XX] 또는 [BEST]"로 선언. 청사진 전체에서 구체 실험 수치 없음 (0.62899는 "기준 선택 기록"으로 한정 표기). 통과.
- 미사용 component 유입 검사: dynamic margin, Gaussian smoothing, RevIN, EMA, SCAD — 블루프린트 본문 어디에도 언급 없음. GRL 수식 내 "focal-style"도 "표준 focal loss(Lin et al. 2017) 아님" 명기. §9.2 수식 금지사항에 VQGAN-style 귀속 금지·standard focal loss 표기 금지·1-layer MLP 표기 금지 모두 명기. 통과.
- R11 설정 도입: §5.2에서 contaminated semi-supervised 정의, 추론 시 label-free 명시. §3 Para 3에서 동기 서술. 통과.
- R19 괄호 클러스터 인용 정책: §4.2/§4.3/§6.5에서 개별 소개 금지 + 클러스터 인용 명기. 통과.
- R20 NRdetector 스코핑: §4.3에서 차이 우선(D1/D3/D5) + 공통점 3개 간략 인정 + 마지막 포지셔닝 문장. 통과.
- R22 MAE 귀속: §5.4에서 "vision MAE(He et al. 2022)에서 patch/masking 아이디어를 도입했음을 1문장 명시" + §4.4에서 독립 수렴 구분. 통과.
- Elsevier 요소: §1 Abstract 계획(150-200 words, 4단 구조), Keywords(6-7개), Highlights(5 bullet, 각 ≤125 chars) 명기. 통과.
- R7 Appendix: §8에서 A/B/C 3부 전수 설계. 통과.
- R32 Label sparsity analysis: §6.8에서 동기·설계·선험적 논리(4개)·Fig. 3·NRdetector 구분 전수 명기. 통과.

---

## 4. Phase 4 연계 수요 (인용 수요 주장 단위 요약)

| 주장 | 필요 근거 유형 | 수요 등록 위치 |
|-----|--------------|--------------|
| 산업·안전 응용(CPS, 데이터센터) 동기 | industry report 또는 survey 1–2개 인용 | §3.1 Para 1 |
| "비지도 방법의 현실적 지배" | TSAD survey 인용 | §3.1 Para 2 |
| 비지도 4유형 분류(재구성·예측·대조·밀도) | 각 유형 대표 논문 클러스터 인용 | §4.2 §2.1 |
| "다변량 TSAD에서 semi-supervised/PU 극히 드물다" 스코핑 | 검색으로 선행 사례 전수 확인 | §4.3 §2.2 |
| "contaminated semi-supervised" 명칭 기존 사용 여부 | 용어 검색 검증 | §11 결정 ② (MINOR-005) |
| NRdetector를 "거의 유일한 선행 연구"로 주장 | 반증 논문 부재 검증 | §4.3 §2.2 |
| "end-to-end first" 최초성 주장 | 반증 논문 부재 검증 | §3.1 Thesis (MINOR-006) |
| Zhang et al. [TPAMI 2022] self-distillation 원류 | DOI/서지 확인 | §4.4 §2.3 |
| GRL 원전 (Ganin et al. 2016 JMLR) | DOI 확인 | §5.6 §3.5 |
| "표준 focal loss(Lin et al. 2017) 아님" 방어 | Lin et al. 2017 수식 대조 | §5.6 (MAJ-004) |
| AR threshold가 TSAD 문헌 표준 관행 | 선행 연구 사례 확인 (EXPERIMENT_PROTOCOL_TRUTH §⑤ 근거 보류 항목) | §6.4 §4.1.3 |
| PA%K-AUC F1 (Kim et al. AAAI 2022) | DOI 확인됨 | §6.4 §4.1.3 |
| VUS-PR/ROC (Paparrizos et al. PVLDB 2022) | DOI 확인됨 | §6.4 §4.1.3 |
| Affiliation F1 (Huet et al. KDD 2022) | DOI 확인됨 | §6.4 §4.1.3 |
| "PA F1 과대평가 위험" (Kim et al. 2022) | 원문 인용 | §6.4 §4.1.3 |

---

## 5. 판정 요약

BLOCKER 5건, MAJOR 12건, MINOR 6건, NOTE 3건.

BLOCKER 5건:
- BLK-001: PAGE_BUDGET과 BLUEPRINT의 섹션별 분량 수치 불일치 (§4: ~2.8p vs 3.2p)
- BLK-002: GRL 위치(student hidden, output projection 이전) + Fig. 2 설계에서 GRL 추론 시 비활성 표시 누락
- BLK-003: SMD 실제 투입 F 범위(29–36) vs NOTION F=38 혼동 가능 — d_model 표에 명기 필요
- BLK-004: GRL adaptive lambda가 sigmoid ramp-up이 아닌 trainer inline grad-ratio인데 §5.5에 sigmoid ramp-up 공식이 남아 있음
- BLK-005: baseline epoch 비대칭(MAE 500 vs unsupervised 10) 방어 계획 없음

모든 BLOCKER는 Phase 5 집필 전에 해소되어야 한다.
