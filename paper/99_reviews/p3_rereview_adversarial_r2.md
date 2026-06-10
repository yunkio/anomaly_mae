---
phase: 3
agent: rereviewer-adversarial
directives: [T3, R1, R2, R5, R6, R7, R8, R9, R10, R11, R15, R16, R19, R20, R21, R22, R32]
last_modified: 2026-06-11
inputs:
  - paper/99_reviews/p3_blueprint_adversarial_r1.md (B5/M12/m6/N3)
  - paper/99_reviews/p3_blueprint_fixlog_r2.md
  - paper/03_blueprint/PAPER_BLUEPRINT.md (r2)
  - paper/03_blueprint/PAGE_BUDGET.md (r2)
verification_basis: |
  정본 1순위 271_CONFIG_TRUTH.md(r3) > RESEARCH_SYNTHESIS.md(r2) > EXPERIMENT_PROTOCOL_TRUTH.md(r3)
  + 코드 spot (mae_anomaly/ read-only) + checkpoint/metadata 직접 재실측.
  red-team 재리뷰어 산출물 미열람 (독립 작업).
direct_measurements:
  - "PSM metadata: config.d_model=512, dim_feedforward=2048, num_features=25 (2026-06-11 재실측)"
  - "PSM best_model.pt: patch_embed.weight=(512,250), patch_embed.bias=(512,) (2026-06-11 재실측)"
  - "trainer.py:1202-1211 — GRL sigmoid λ schedule (use_grl만으로 게이트, 271 활성)"
  - "model.py:1119 — student forward gate `and not teacher_only` (warmup 중 학습 경로 student forward skip)"
---

# Phase 3 블루프린트 재리뷰 (adversarial 정합성, r2)

> 검증 범위: r1 BLOCKER 5건 해소 재추적(1차 소스 전건 재확인) + MAJOR 12건 마감 대조
> + r2 추가·변경 서술 전수 신규 오류 검사 + Directive 체크리스트 재실행 + placeholder(A8) 정책.
> 검토일: 2026-06-11.

---

## 1. BLOCKER 5건 해소 재추적 (각 1차 소스 재확인)

### ADV-B01 — 분량 수치 통일 · **해소 확인 (RESOLVED)**

기계적 대조 수행:
- BLUEPRINT §2: §1 1.6 / §2 1.1 / §3 2.7 / §4 3.3 / §5 0.3 = 9.0p.
- PAGE_BUDGET §1·§6: 동일 수치, 합계 9.0p. **두 문서 전건 일치.**
- 단일 정본 선언: PAGE_BUDGET frontmatter("본 문서가 섹션별 분량 수치의 단일 정본") + BLUEPRINT frontmatter·§2("충돌하면 PAGE_BUDGET을 따른다") — 양방향 명기 확인.
- 세부 합계 산술 전건 재계산 (재리뷰어 독립 검산):
  - §1: 0.20+0.25+0.22+0.40+0.35+0.05+0.05 = **1.52** ✓ (문서 기재 1.52, 슬랙 0.08)
  - §2: 0.03+0.45+0.03+0.35+0.03+0.22 = **1.11** ✓
  - §3: 13개 항목 합 = **2.58** ✓ (슬랙 0.12)
  - §4: 19개 항목 합 = **3.93** ✓ (초과 −0.63p 자인)
  - Figure 1.58 + Table 1.43 = 3.01 ✓; §6 합계 9.0 ✓.
- r1의 잔여 우려("압축 후도 목표 상단 초과"): r2 압축 전략 6개 합 ~0.65p ≥ 초과분 0.63p; 전략 1+3+4+5+6 ≈ 3.43p, 전략 2 병용 시 3.28p ≤ 3.3p — 문서 자체가 경로를 수치로 제시. 수용 가능.

### ADV-B02 — GRL 위치 + Fig. 2 training-only 표기 · **해소 확인 (RESOLVED, 코드 재실측)**

- `model.py:1149-1154` 직접 확인: `if self.training and hasattr(self, 'anomaly_classifier'):` →
  `cls_logits = self.anomaly_classifier(student_hidden, lambda_grl)`. GRL+AnomalyClassifierHead는
  **student decoder 출력 hidden**에 적용되고, `student_output_projection(student_hidden)`은 **:1165**에서
  그 이후 수행 — "마지막 층 hidden, output projection 이전" 서술 정확.
- `self.training` 게이트로 추론 시 비활성(training only) — §5.3 필수 레이블 2건(위치 + dashed box) 모두 신설 확인.
- NOTION I-3 "Output Projection 다음" 배치를 부정확으로 판정한 것 — 코드 정본 우선 판정 타당.
- §5.6(C)·§5.7 위치·비활성 표기 동기화 확인.

### ADV-B03 — SMD F=29–36 + reviser 추가 발견 "d_model 전 entity 512 고정" · **해소 확인 + 추가 발견 재검증 CONFIRMED**

reviser의 핵심 주장을 본 재리뷰어가 **독립 재실측**:
- `results/experiments/271_20260602_020545_271canon_baseline/PSM/experiment_metadata.json` →
  `config.d_model=512`, `dim_feedforward=2048`, `num_features=25` — **일치**.
- `PSM/best_model.pt` → **`patch_embed.weight = (512, 250)`** = Linear(10×25→512) — **reviser 주장 사실**.
  dynamic 매핑(min{d∈{128,…,512}: d≥10F}=min d≥250)이었다면 **256**이어야 함 → 271 런타임 d_model=512 확정.
- 271_CONFIG_TRUTH §II: `d_model=512`·`dim_feedforward=2048`이 **전 37 entity 공통 114키**에 포함 — 정본 정합.
  (부록 1의 SWaT checkpoint `(512,450)` 실측도 동방향 보강 증거.)
- r1 리뷰(본인 전신)의 전제("런타임 d_model은 동적 결정")까지 정정한 것은 **리뷰를 넘어선 올바른 정정** — NOTION I-3
  dynamic 표는 Set C preset stale (batch 512→1024 override 선례와 동급) 판정 타당.
- SMD F=29–36: 271_CONFIG_TRUTH §III-3a 실측(최소 29 machine-3-10, 최대 36 machine-3-3)과 일치.
  §6.2 "constant 제거 후 29–36, raw 38" / §C.1 입력 차원 표(SWaT 45=51−6, WaDi 123=127−4 NaN) —
  EXPERIMENT_PROTOCOL_TRUTH §①(SWaT constant 6 {P202,P401,P404,P502,P601,P603}, WaDi FEEDBACK-2 RESOLVED)와 전건 일치.
- 파급 적용(§5.4/§5.5/§6.3/§9.1/§9.2/§C.1, dim_feedforward 고정 표기, AnomalyClassifierHead hidden=256 정합) 확인.

### ADV-B04 — GRL λ 공식 교체 · **부분 해소 — 교체 공식은 정확하나, 삭제·금지 판단이 코드와 모순 (신규 BLOCKER NEW-B1 파생)**

**정확한 부분 (코드 재실측으로 확인):**
- `trainer.py:751-765`: `_grl_lambda_adp = (‖∇L_main‖/(‖∇L_GRL‖+1e-4)).clamp(0,10)`,
  `_grl_effective = self._prev_epoch_grl_lambda * _grl_w` (`_grl_w = grl_loss_weight = 0.2`),
  `loss += _grl_effective * _grl_cls_loss`. — **×0.2 계수 실재**, 직전 epoch 값 적용 실재.
  블루프린트 §5.5의 "λ_GRL_eff = λ_GRL_adp × grl_loss_weight(0.2)" 공식은 trainer 코드와 일치.
- FM도 동형 확인 (`trainer.py:639-653`, ×fm_loss_weight 1.0, prev-epoch).
- "GRL·FM **손실 항**이 ramp 없이 warmup 종료 직후 즉시 투입" — `not teacher_only` 게이트만 존재, 정본
  (271_CONFIG_TRUTH §VIII r2 정정)과 일치. **손실 항에 한해서는** 맞다.

**모순 부분 → §3 NEW-B1 (BLOCKER) 참조.** 요지: sigmoid ramp-up은 "271 실행 경로 미사용"이 아니다 —
reversal **계수**로 실제 사용된다. r1 BLK-004의 판정 자체가 불완전했고 r2가 이를 금지 조항으로 codify했다.

### ADV-B05 — epoch 비대칭 방어 신설 · **해소 확인 (RESOLVED)**

- §6.3: MAE 500(eval 5-ep 간격) / unsupervised 22종 10(매 epoch eval) / weak 4종 50, batch 1024 vs 512 명시 공개
  — EXPERIMENT_PROTOCOL_TRUTH r3 RB-1 실측과 **전건 일치** (early stopping 양쪽 부재, `random`만 5-run mean±std 예외 포함).
- §15 신설 행(공정성 시나리오, 방어 ①②③④) + Appendix §B.4 placeholder + PAGE_BUDGET §5 등재 확인.
- test-set model selection 공개 문구(§6.3)와 §15 별도 행 — M-3/REQUEST-4의 "반드시 공개" 의무 이행 확인.

---

## 2. MAJOR 12건 마감 대조 (fixlog 1:1 + 개정본 spot)

| ID | fixlog 처리 | 개정본 spot 검증 | 판정 |
|----|-----------|----------------|------|
| MAJ-001 | SDMAE branch-off 정정 | §4.4: "teacher decoder의 첫 transformer 블록 뒤에서 student decoder가 분기" — ANCHOR_SDMAE_DOSSIER §3.1 verbatim("A student decoder branches out from the teacher after the first transformer block of the main decoder")과 일치. 결정 ⑤ 각주 초안에 branch-off vs 독립 decoder 반영, 결정 ① C2 행 "구조는 상이" 정밀화 | **마감** |
| MAJ-002 | 계열/entity 구분 + 완주 경고 | §6.2 "총 113 학습 단위(=1+2+1+28+54+27; dual-eval 시 114)" — PROTOCOL_TRUTH §① 산식과 일치. §0.4 신설, §6.6 "완주 후 채움" 명기. 단 §0.4 "37/113" 혼합 기준 → NEW-m1 | **마감 (잔여 MINOR 1)** |
| MAJ-003 | SWaT 45 플래그 | §6.3: "45=51−constant 6 {P202,P401,P404,P502,P601,P603}; 현 loader 51 반환" — FEEDBACK-7과 컬럼명까지 일치. §C.2 + 결정 ③ 갱신 조건 연동 | **마감** |
| MAJ-004 | focal variant positive 지침 | §5.6(C)·§9.2: "focal-style BCE variant with class-prior pos_weight" + p_t 차이 1문장 + 예시 문장. 수식 (1−exp(−BCE))²×BCE_{w+} — loss.py:337-340·SYNTHESIS 표A 일치 | **마감** |
| MAJ-005 | test-set selection 방어 | §6.3 공개 문구("uniformly applied…; no separate validation split") + §15 행 + §B.4 — M-3 정합 | **마감** |
| MAJ-006 | SOTA Legacy 6 재분류 | §6.5: Simple 5 + Neural 3 + GCN-LSTM 1(독립) + Legacy 6(anomaly_transformer/tranad/usad/dagmm/gdn/omnianomaly) + New 7(tfmae/npsr/timesnet/dcdetector/memto/moderntcn/catch) = 22 — PROTOCOL_TRUTH §③(r2 정정본) 전건 일치 | **마감** |
| MAJ-007 | L_cls 표기 통일 | §5.6 총손실 L_total = L_recon + L_OD + λ_FM_eff·L_FM + λ_GRL_eff·L_cls + §9.1 행 신설 + 혼용 금지 | **마감** |
| MAJ-008 | AR 상한 실측 열거 | §5.2: SWaT 1.63 / WaDi 0.52·0.76 / PSM 6.20 / SMAP 0.70 / MSL 1.70% + SMD 단정 금지 — PROTOCOL_TRUTH §① 일치 | **마감** |
| MAJ-009 | per-patch/집계 분리 | §5.7: (11)(12) per-patch / (13) point mean 집계(bincount-합/coverage) — §④-실행 2항·evaluator.py:278-280 정합; ε=1e-4 유지(정본 일치) | **마감** |
| MAJ-010 | warmup 연쇄 제거 | bullet 3 warmup 삭제 + Table 3 행 6 명시적 conditional — 행 삭제가 contribution을 건드리지 않는 구조 확보 | **마감** |
| MAJ-011 | batch 비대칭 | ADV BLK-005 통합 처리 — §6.3 공개 + §15 | **마감** |
| MAJ-012 | complementary masking 수식어 | §7: "구현되어 있으나 본 실험 미사용(eval_complementary_masking=False)" — 271_CONFIG_TRUTH §VII #12 정합 | **마감** |

**MINOR 6건**: m1(TFMAE §2.3 전속) ✓ / m2(0.62899 갱신 조건, 결정 ③) ✓ / **m3(warmup 표현) — 구현됐으나 방향이 코드와 반대 → NEW-B2** / m4(`affiliation_f1_ar`, evaluator.py 키 실측 확인) ✓ / m5(용어 Phase 4 검증, 결정 ②+§16) ✓ / m6(to our knowledge + 스코핑 박스, §0.1) ✓.
**NOTE 3건**: N1(DAGMM 결정 ⑦) ✓ / N2(variant 설계 명시 문장) ✓ / N3(TS-SDMAE 제외, 결정 ⑧ + §10.1 취소선) ✓.

---

## 3. 신규 오류 (r2 추가·변경 서술의 정본·코드 대조)

### NEW-B1 — **BLOCKER**: "GRL sigmoid ramp-up 271 미사용" 단정은 코드와 모순 — sigmoid는 **reversal 계수**로 271에서 실제 활성

**Artifact**: BLUEPRINT §5.5(취소선 단락), §5.6(C) "backward에서 gradient × (−λ_GRL_eff)", §9.2("sigmoid ramp-up 공식(Ganin schedule) 서술 금지 — 271 미사용"), §15 일부 파급.

**코드 실측 (2026-06-11)**:
```
trainer.py:1202-1211  (게이트: if getattr(self.config, 'use_grl', False) — 271은 True, 추가 플래그 없음)
    _p = max(0.0, min((_student_epoch + 1) / _student_total, 1.0))   # _student_total = 500-250 = 250
    self.model._grl_lambda = 2.0 / (1.0 + math.exp(-10.0 * _p)) - 1.0   # warmup 중에는 0.0
model.py:1152-1153
    lambda_grl = getattr(self, '_grl_lambda', 0.0)
    cls_logits = self.anomaly_classifier(student_hidden, lambda_grl)
model.py:133-140 (GradientReversalFunction)
    backward: return -ctx.lambda_ * grad_output   # ← 이 lambda_가 sigmoid 값
```
`self.model._grl_lambda` 대입 지점은 코드 전체에서 trainer.py:1209/1211 **뿐** (grep 전수). 즉 271에는 **λ가 2개 공존**한다:
1. **손실 가중치** = `_prev_epoch_grl_lambda(grad-ratio clamp[0,10]) × 0.2` — 블루프린트 서술 **정확**.
2. **반전(reversal) 계수** λ_rev = Ganin sigmoid `2/(1+exp(−10p))−1`, p=(epoch−250+1)/250 — epoch 250에서 ≈0.02로 시작해 500에서 ≈1.0까지 ramp. **271 실행 경로에서 매 epoch 실제 설정·사용됨.** NOTION I-4의 sigmoid 표("0→0.9999")가 사실상 정확했다.

**오류 판정**:
- §5.5의 "이 공식은 271 실행 경로에서 미사용. 논문 §3.4 서술 금지"는 **허위 부재 진술** (r1 BLK-004의 판정 자체가 불완전 — 손실 가중치 측면만 보고 reversal 계수 측면을 누락; r2가 이를 검증 없이 금지 조항으로 codify).
- §5.6(C) "GRL gradient reversal: backward에서 gradient × (−λ_GRL_eff)"는 **계수 오귀속** — backward 곱셈 계수는 λ_rev(sigmoid)이고, λ_GRL_eff는 손실 항 가중치다. student hidden에 실제 도달하는 adversarial gradient는 −λ_rev × λ_GRL_eff × ∂L_cls.
- 파급: 이대로 논문 §3.4–3.5에 들어가면 "warmup 종료 직후 suppression 즉시 full 강도"라는 잘못된 메커니즘 서술이 되고, 코드 공개 시(결정 ⑥) 리뷰어가 trainer.py에서 즉시 반박 가능. §9.2의 금지 조항은 실재 메커니즘의 서술을 막는다.
- **정본 비고**: 271_CONFIG_TRUTH §VIII·RESEARCH_SYNTHESIS 표A는 reversal 계수 스케줄을 **미등재** (둘 다 손실 가중치만 기술 — "ramp 없음" 서술도 손실 항 투입에 한정되어 그 자체는 참). 즉 정본의 누락이 r1 오판의 근본 원인 — **Phase 1 정본 보강(escalation) 필요**: 271_CONFIG_TRUTH §VIII GRL Details에 reversal 계수 행(`trainer.py:1202-1211` + `model.py:129-140`) 추가.

**권장 수정**: §5.5를 "GRL·FM **손실 항**은 epoch 250부터 ramp 없이 즉시 투입(가중치 = grad-ratio adaptive × 0.2); 단 GRL **반전 계수** λ_rev는 Ganin et al. (2016)의 sigmoid schedule(2/(1+e^{−10p})−1, p = student-phase 진행률)로 0→≈1 ramp — adversarial suppression 강도는 점진 증가"로 이원 서술. §5.6(C) backward 계수를 λ_rev로 정정, §9.2 금지 조항을 "sigmoid를 **손실 가중치**로 서술하는 것 금지(손실 가중치는 grad-ratio×0.2); reversal 계수로는 사용 사실 명기"로 교체. Ganin et al. 2016 인용 수요 §16 유지(이미 등재).

### NEW-B2 — **BLOCKER**: warmup 중 "student forward는 수행된다"는 서술이 코드와 반대 — 학습 경로에서 student forward 자체가 skip

**Artifact**: BLUEPRINT §5.5 (ADV MINOR-003 처리분): "epoch < 250에서 student decoder의 forward는 수행되지만 student 관련 손실항(OD/FM/GRL)이 전부 비활성(loss.py `teacher_only=True` 게이트)되어 student로 gradient가 흐르지 않는다 — 'frozen'을 'forward 중단'으로 오독하지 않도록".

**코드 실측 (2026-06-11)**:
```
trainer.py:526-535  # "2026-05-29: propagate teacher_only so model can skip student
                    #  decoder / GRL classifier / SCAD head forward during warmup."
    teacher_output, student_output, mask = self.model(..., teacher_only=teacher_only)
model.py:1119
    if self.config.use_student and self.student_decoder is not None and not teacher_only:
        ...  # student forward
    else:   # teacher_only mode (warmup): student_output=None,
            # self._student_hidden = None, self._grl_cls_logits = None
loss.py:186-195  # "student_output is None (teacher_only mode in model forward —
                 #  skips student decoder during warmup for compute savings)"
```
2026-05-29 변경(271 실행 2026-06-02 **이전**)으로, warmup 중 **학습 경로의 student forward는 수행되지 않는다**
(forward 중단이 맞다). loss 게이트(loss.py:213 `not teacher_only`)는 이중 방어로 존재. 평가/시각화 경로만
teacher_only=False 기본값으로 full forward를 유지한다.

**오류 판정**: r1 MINOR-003의 권고 자체가 NOTION I-4의 stale 서술("forward는 수행되지만 손실 비활성")을 그대로
따른 오류였고, r2가 검증 없이 구현했다. "'frozen'을 forward 중단으로 오독 금지"라는 지침은 **정반대 오독을 강제**한다.
271_CONFIG_TRUTH §VIII "student frozen"이 오히려 실제 동작에 가깝다. 논문 §3.4에 들어가면 코드 대조 시 즉시 반박되는
신규 사실 오류.

**권장 수정**: §5.5 해당 단락을 "warmup(epoch<250) 동안 **학습 경로에서는 student decoder forward 자체가 생략**된다
(`teacher_only` 전파, model.py:1119; 손실 게이트 loss.py:213은 이중 방어). student 파라미터는 갱신되지 않으며
('frozen'), per-epoch 평가 경로는 full forward를 수행한다"로 교체. ADV MINOR-003 처리 방향을 역전.

### NEW-m1 — MINOR: §0.4 "37/113" — 분자·분모 집계 기준 혼합

분자 37은 평가 단위(SWaT full/excl22를 2로 집계 — RESEARCH_SYNTHESIS §① MAJ-007 기준), 분모 113은 학습 단위
(SWaT=1 — PROTOCOL_TRUTH §① 산식). 정합 표기는 **36/113(학습 단위)** 또는 **37/114(평가 단위)**. §6.2의
"113 학습 단위 / dual-eval 시 평가 단위 114" 구분 자체는 정확하므로 §0.4만 한쪽 기준으로 통일하면 된다.

### NEW-n1 — NOTE: §6.4 `affiliation_f1_ar` 라인 표기 809–813

정본(PROTOCOL_TRUTH REQUEST-1 RESOLVED)은 `evaluator.py:811-813`, 실측 키 할당은 :813. 블루프린트의 "809–813"은
포함 범위라 실해는 없으나 정본 표기와 통일 권장.

### 그 외 r2 추가·변경 기술 서술 전수 검사 — 이상 없음 (확인 목록)

- §14 논거 ② 원본 train 라벨 구조: SWaT/WaDi train 파일 전부 정상(SWaT train anomaly 11,757pts 전부 A2-front 유래 — A1 495,000 정상 + A2 449,919 attack 구성과 정합), PSM/SMD train 라벨 파일 부재, SMAP/MSL 명시적 zeros(loaders.py:2602-2604) — PROTOCOL_TRUTH §② r2와 전건 일치.
- §14 논거 ④/§6.2 safe-cut: clearance 10·무제한 탐색·81채널 중 4채널(전부 MSL: D-16/M-1/M-2/S-2)·max +166 steps·SMAP 0건 — §② 실측 표 일치. "negligible 과장 금지" 주의 유지.
- §14 논거 ⑤ NRdetector 7:3 segment split + 시간 순서 보존 미명시 단정 금지 — NRDETECTOR_DOSSIER §3.1/D8 verbatim 일치.
- §4.4 Zhang et al. TPAMI 2022 = [101] 귀속 — ANCHOR_SDMAE_DOSSIER §2/§5.1 일치.
- force_mask 우선순위 식(anomaly×1000+noise, TopK_8) — model.py:986-996 실측 일치.
- AnomalyClassifierHead 2-layer MLP(512→256→1, LayerNorm/GELU/Dropout 0.1), classifier lr=1e-4(×0.1), AdamW(0.9,0.99)/lr 1e-3/wd 1e-3, bf16, linear warmup 10 + cosine, batch 1024, 8/42 패치, score ε=1e-4·ratio 4.0, fm_active=False(scoring.py:237) — 전부 271_CONFIG_TRUTH §II/§VI/§VIII 일치.
- excl22 83.75%·0.62899 vs 0.62730 구분(결정 ③) — SYNTHESIS §④ α-m3 일치.
- §16 AR threshold 행 "확보 전 방어 논리 사용 금지" — PROTOCOL_TRUTH §⑤-4 r2 정정 일치.
- PAGE_BUDGET r2 변경분(§B.4, §C.1 개편, Table 3 conditional, Table 2 열 고정, landscape 플래그) — BLUEPRINT와 상호 정합.

---

## 4. Directive 체크리스트 재실행 (r1 미충족·부분충족 항목 중심)

| Directive | r1 판정 | r2 판정 | 근거 |
|-----------|--------|--------|------|
| T3 (진실 문서 활용) | 부분 충족 | **부분 충족 (잔존)** | d_model 512 정정·epoch 비대칭·SOTA 분류 등 r1 지적 전건 해소. 단 신규 코드-모순 2건(NEW-B1/B2)이 발생 — 둘 다 정본/NOTION의 불완전성에서 파생된 것이나 1차 소스(코드) 기준 미충족 |
| R6 (9p 배분) | 부분 충족 (BLK-001) | **충족** | 두 문서 수치 일치 + 단일 정본 선언 + 산술 검산 통과 |
| R9 (SDMAE 포지셔닝) | 부분 충족 (MAJ-001) | **충족** | branch-off 사실 정정 + sibling 포지셔닝("adapt this architectural paradigm") + 각주/본문 분산 배치 |
| R21 (용어 방어) | 부분 충족 (MAJ-001) | **충족** | 계보(Zhang→SDMAE→본 논문) + 구조 차이 각주 초안 — dossier 원문 정합 |
| R1, R2, R5, R7, R8, R10, R11, R15, R16, R19, R20, R22, R32 | 충족 | **충족 유지** | r2 변경이 깨뜨린 항목 없음 — R1(bullet 2/3 경계 명문화로 강화), R5(L_cls·d=512 반영; 단 λ 기호 체계는 NEW-B1 해소 시 λ_rev 추가 필요), R11(②-1/2/3 3단 구조 명시로 강화), R15(TS-SDMAE 제외 후 모델명 4 + 제목 5 후보 유지), R32(전수 유지) 개별 spot 확인 |

---

## 5. Placeholder 정책 (A8) 재검사

**통과.** r2 추가분의 수치는 전부 (a) 데이터셋 통계(train/test AR, 83.75%, +166 steps, 113/114 산식,
35,900pts 등), (b) config·코드 사실(512/2048/0.2/1e-4/8패치 등), (c) 기준 선택 기록(0.62899; 비교 근거로서의
0.62730 — 결정 ③ 사유 한정) — **모델 성능 실험 수치의 신규 유입 0건**. Table 2/3/4·Fig 3 전부 placeholder 유지,
§0.4가 "완주 전까지 전부 placeholder"를 재선언.

---

## 6. 종합 판정

| 구분 | 결과 |
|------|------|
| r1 BLOCKER 5건 | **4건 완전 해소** (B01/B02/B03/B05) + **1건 부분 해소** (B04 — 교체 공식 자체는 코드 일치·정확하나, sigmoid "미사용" 단정·금지가 코드와 모순 → NEW-B1) |
| r1 MAJOR 12건 | **12/12 마감 확인** (MAJ-002 잔여 MINOR 1건 — NEW-m1) |
| r1 MINOR/NOTE | 6/6·3/3 처리 확인, 단 MINOR-003 처리가 코드와 반대 방향 (→ NEW-B2) |
| 신규 오류 | **BLOCKER 2** (NEW-B1 GRL sigmoid reversal 계수 실재 / NEW-B2 warmup student-forward skip), **MINOR 1** (NEW-m1 37/113 혼합 기준), NOTE 1 |
| reviser 추가 발견 (d_model 512) | **독립 재실측으로 CONFIRMED** — checkpoint patch_embed=(512,250) 직접 확인 |
| Directive | R6/R9/R21 충족 격상; T3만 신규 2건으로 부분 충족 잔존 |
| Placeholder (A8) | 통과 |

**판정: REVISE (r3 필요 — 국소 수정 2건).** r2는 r1 발견을 사실상 전건 올바르게 해소했고 d_model 정정이라는
가치 있는 추가 발견까지 검증했으나, GRL 메커니즘의 이원 구조(손실 가중치 grad-ratio×0.2 vs 반전 계수 sigmoid
ramp)와 warmup 중 student forward skip이라는 두 코드 사실에서 **새 허위 부재/반대 진술**이 들어갔다. 두 건 모두
§5.5/§5.6/§9.2 국소 수정으로 해소 가능하며 구조 변경은 불요. 추가로 **Phase 1 정본 escalation 권고**:
271_CONFIG_TRUTH §VIII GRL Details에 reversal 계수 sigmoid schedule(trainer.py:1202-1211, model.py:129-140)
및 warmup 중 student-forward skip(model.py:1119, trainer.py:526-535) 등재 — 이번 오류 2건의 근본 원인은
정본의 해당 메커니즘 누락이다.
