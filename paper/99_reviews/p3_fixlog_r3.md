---
phase: 3
agent: fixer (Phase 1 정본 보강 + Phase 3 블루프린트 r3)
directives: [T3, R1, R5, R10, R11, R23, R27, R32]
last_modified: 2026-06-11
inputs:
  - paper/99_reviews/p3_rereview_adversarial_r2.md (NEW-B1, NEW-B2, NEW-m1, NEW-n1)
  - paper/99_reviews/p3_rereview_redteam_r2.md (R2-MAJ-01, R2-MIN-01..04, R2-NOTE-01..02)
outputs:
  - paper/01_research_understanding/271_CONFIG_TRUTH.md (r4 — §VIII 보강 + 부록 4)
  - paper/01_research_understanding/CODEBASE_UNDERSTANDING.md (r4 — §1/§2.5/§2.6/§5.3 + 부록 3)
  - paper/01_research_understanding/RESEARCH_SYNTHESIS.md (r3 — §③-3/표A warmup·GRL 행 + 정정 이력)
  - paper/03_blueprint/PAPER_BLUEPRINT.md (r3)
  - paper/03_blueprint/PAGE_BUDGET.md (r3)
verification_basis: |
  전 수정 항목에 대해 코드 1차 소스(mae_anomaly/, read-only) 직접 재확인 + 271 metadata 재추출
  (PSM experiment_metadata.json: use_grl=True, grl_adaptive_lambda=True, grl_loss_weight=0.2,
  fm_adaptive_lambda=True, fm_loss_weight=1.0, teacher_only_warmup_epochs=250, num_epochs=500,
  grl_target_mode='window' — 2026-06-11 재실측). 구 fixlog(p3_blueprint_fixlog_r2.md)는 본 phase
  쓰기 범위 외라 미수정 — §3 EXPERIMENT_EXECUTION_TODO 집계는 본 문서가 대체·확장.
---

# Phase 3 Fixlog r3 — 재리뷰(adversarial r2 + redteam r2) 발견 전수 처리

> 핵심: 재리뷰 신규 BLOCKER 2건(NEW-B1/B2)의 근본 원인이 **Phase 1 정본의 메커니즘 누락**으로
> 판정되어, §6.3 회귀 프로토콜에 따라 정본(271_CONFIG_TRUTH r4 + CODEBASE r4 + SYNTHESIS r3)과
> 블루프린트(r3)를 함께 수정했다. 모든 수정은 아래 §1 코드 확정 기록에 근거한다.

---

## 1. 작업 A — 코드 사실 확정 기록 (수정 전 직접 확인; 2026-06-11)

### A-1. GRL 반전 계수 λ_rev — Ganin-style sigmoid ramp, 매 epoch 설정 (확정)

```
trainer.py:1201        # "GRL lambda: set BEFORE train_epoch so the current epoch uses the correct value"
trainer.py:1202        if getattr(self.config, 'use_grl', False):      ← 게이트는 use_grl뿐 (271=True, 추가 플래그 없음)
trainer.py:1204        _student_start = self.config.teacher_only_warmup_epochs        # 271: 250
trainer.py:1205        _student_total = max(self.config.num_epochs - _student_start, 1)  # 500-250 = 250
trainer.py:1206        _student_epoch = epoch - _student_start          # 0-indexed within student phase
trainer.py:1207        _p = max(0.0, min((_student_epoch + 1) / _student_total, 1.0))
trainer.py:1208-1209   if _student_epoch < 0: self.model._grl_lambda = 0.0             # warmup 중 0
trainer.py:1211        else: self.model._grl_lambda = 2.0/(1.0 + math.exp(-10.0*_p)) - 1.0
```
- 진행 변수 p = clip((epoch−250+1)/250, 0, 1) — **student-phase 진행률**. 0-based epoch 250에서
  p=0.004 → λ_rev≈0.0200, epoch 499에서 p=1.0 → λ_rev≈0.99991 (0→≈1 단조 ramp).
- `model._grl_lambda` **대입 지점은 trainer.py:1209/1211 뿐** (grep 전수 — 그 외 매치는 history/로깅:
  trainer.py:319,1292,1294 및 분석 스크립트). 재리뷰 적시 라인(1202-1211)과 일치.

### A-2. λ_rev 소비처 — GradientReversal backward 곱셈 계수 (확정)

```
model.py:1149-1150     if self.training and hasattr(self, 'anomaly_classifier'):   ← training-only
model.py:1152          lambda_grl = getattr(self, '_grl_lambda', 0.0)
model.py:1153          cls_logits = self.anomaly_classifier(student_hidden, lambda_grl)
model.py:129-139       class GradientReversalFunction: forward=x.clone(); backward: return -ctx.lambda_ * grad_output
```
- 즉 backward에서 gradient에 곱해지는 계수 = **λ_rev(sigmoid)**이며, 손실 가중치가 아니다.
- student hidden 도달 adversarial gradient = −λ_rev × λ_GRL_eff × ∂L_cls/∂(GRL 출력).

### A-3. GRL 손실 가중치 λ_GRL — grad-ratio clamp[0,10] × 0.2, 직전 epoch 적용 (확정)

```
trainer.py:747         elif use_grl and not teacher_only and 'grl_cls_loss' in loss_tensors:
trainer.py:749         _grl_w = getattr(self.config, 'grl_loss_weight', 1.0)          # 271 metadata: 0.2
trainer.py:760         _grl_lambda_adp = (_main_g.norm() / (_grl_g.norm() + 1e-4)).clamp(0.0, 10.0).detach()
trainer.py:762         _grl_effective = self._prev_epoch_grl_lambda * _grl_w           ← ×0.2 계수 실재
trainer.py:763         loss = loss + _grl_effective * _grl_cls_loss
trainer.py:190         self._prev_epoch_grl_lambda = 1.0   (초기값)
trainer.py:1317-1319   _grl_l = epoch_losses.get('grl_lambda', 0.0); if _grl_l > 0: self._prev_epoch_grl_lambda = _grl_l
```
- 공식 λ_GRL = clamp(‖∇L_main‖/(‖∇L_GRL‖+1e-4), 0, 10) (w = student decoder 마지막 파라미터),
  적용값은 **직전 epoch 집계값** × grl_loss_weight(0.2). 프롬프트 명세(×0.2 포함) 그대로 확인.

### A-4. Teacher-only warmup 중 학습 경로 student decoder forward skip (확정)

```
trainer.py:526-535     # "2026-05-29: propagate teacher_only so model can skip student decoder /
                       #  GRL classifier / SCAD head forward during warmup." → model(..., teacher_only=teacher_only)
model.py:1119          if self.config.use_student and self.student_decoder is not None and not teacher_only:
loss.py:193            if student_output is None: ... (teacher_only mode — student recon 0.0 sentinel)
loss.py:213            if self.use_discrepancy and not teacher_only:   ← 손실 게이트 (이중 방어)
```
- 2026-05-29 변경은 271 실행(2026-06-02) **이전** — 271 warmup(0-based epoch 0–249) 동안 학습
  경로의 student forward는 수행되지 않는다. **student 학습 개시 = 0-based epoch 250(=251번째 epoch).**
- 평가/시각화 경로는 `teacher_only=False` 기본값으로 full forward 유지 (trainer.py:532-533 주석).
- λ_rev도 warmup 중 0.0 (trainer.py:1209) — 이중으로 비활성.

### A-5. FM λ — 손실 가중 단일 구조, reversal 계수 없음 (확정: sigmoid는 GRL 전용)

```
trainer.py:639         if fm_adaptive_lambda and not teacher_only and 'fm_loss' in loss_tensors:
trainer.py:647         _fm_lambda = (_main_g_fm.norm() / (_fm_g.norm() + 1e-4)).clamp(0.0, 10.0).detach()
trainer.py:652         loss = loss + self._prev_epoch_fm_lambda * _fm_w * _fm_loss_tensor   # _fm_w = fm_loss_weight = 1.0
```
- FM에는 model-level 곱셈 계수/ramp가 존재하지 않음 — sigmoid `_grl_lambda`의 소비처는
  model.py:1152(anomaly_classifier) **단일**(grep 전수). 이중 λ 구조는 **GRL 전용**.

### 보조 확인 (블루프린트 수정에 인용)

- `loss.py:293-302` — `_pos_count == 0 → grl_cls_loss_tensor=None` (batch positive 부재 시 GRL 손실
  미계산; Table 4 standard-split "라벨 경로 자가 비활성" 코드 근거).
- `evaluator.py:811-813` — `affiliation_f1_ar` 키 할당 :813 (NEW-n1 라인 표기).

---

## 2. 발견 ID별 처리표

### 2-1. Phase 1 정본 보강 (escalation — NEW-B1/B2의 근본 원인 해소)

| 대상 | 처리 내용 | 근거 (§1) |
|------|----------|----------|
| 271_CONFIG_TRUTH **r4** | §VIII GRL Details: **이중 λ 구조** 행 + **λ_rev** 행(정확 공식·warmup 0·epoch 250≈0.020→499≈0.9999·소비처·대입 지점 전수·FM 무대응) + student-hidden 도달 gradient 행 신설; Lambda balancing 행을 "손실 가중치 λ_GRL"로 명칭 명확화 + file:line 보강(:751–765/:760/:762–763/:1317–1319); §VIII Training warmup 행에 **학습 경로 forward skip**(trainer.py:526–535, model.py:1119, loss.py:193/213) + student 학습 개시 epoch + 평가 경로 구분; "ramp 없음" 서술을 **손실 항 한정**으로 정밀화; Loss Components GRL 행의 `-lambda × grad` lambda를 λ_rev로 명시. frontmatter last_modified 갱신 + 헤더 r4 정정 노트 + 부록 4 | A-1~A-5 |
| CODEBASE_UNDERSTANDING **r4** | §1 GRL: λ_rev bullet 신설 + λ_GRL bullet에 "손실 항 가중치(반전 계수 아님)" 주의; §2.6: "(별개 메커니즘) λ_rev — adaptive λ 3경로와 독립" 절 신설; §2.5/§5.3: anomaly-ramp 271 no-op 주석(잔존 모순 제거) + warmup forward-skip file:line 보강 + 평가 경로 구분 + λ_rev warmup 0; 부록 3 신설 | A-1~A-5 |
| RESEARCH_SYNTHESIS **r3** | §③-3: 반전 계수 λ_rev sigmoid ramp 1구 병기; 표A warmup 행: "전부 비활성" → 학습 경로 forward skip + "anomaly loss ramp 50ep"(271 no-op) 삭제 + 손실 항 즉시 투입/λ_rev ramp 이원 서술; 표A GRL 행: backward `−λ_rev × grad` + 정확 공식 + 이중 λ 구조; 정정 이력 추가 | A-1, A-2, A-4 |

grep 전수 점검: 세 정본에서 `sigmoid|ramp|frozen|teacher_only|reversal|_grl_lambda` 관련 서술 전수 확인 —
잔존 모순 없음 (CODEBASE §5.3은 r3 시점에 이미 "skipped"로 정확했고 file:line만 보강).

### 2-2. 블루프린트 r3 (PAPER_BLUEPRINT.md)

| ID | 등급 | 처리 | 위치 |
|----|------|------|------|
| **NEW-B1** | BLOCKER | §5.5 "Warmup 종료 후 손실 투입" 단락을 **이중 λ 구조 이원 서술**로 교체(손실 가중치 즉시 투입 서술은 유지 — 코드 일치 재확인; 반전 계수 λ_rev Ganin sigmoid ramp 0→≈1 신설; "sigmoid 미사용·서술 금지" 단정 철회); §5.6(C) backward 계수를 λ_rev로 정정(λ_GRL_eff = 손실 항 가중치; 도달 gradient = −λ_rev×λ_GRL_eff×∂L_cls); §9.1 λ_rev 행 신설 + λ_FM/λ_GRL 행 "loss weights" 명확화; §9.2 금지 조항을 "sigmoid를 손실 가중치로 서술 금지 + λ_rev 사용 사실 명기 + 단일 λ 합산 금지"로 교체; §15 GRL 행에 λ_rev ramp 방어 재료 추가; 논문 §3.4 서술 방침 명시(R23/R27 — 손실 가중 적응·반전 계수 ramp를 일반적 표현으로, Ganin 2016 인용 §16 기등재); 헤더 r2 요약의 해당 행 취소선+정정 | §5.5, §5.6(C), §9.1, §9.2, §15, 헤더 |
| **NEW-B2** | BLOCKER | §5.5 warmup 단락을 "학습 경로 student forward 자체 생략"(model.py:1119, trainer.py:526–535; loss.py:213 이중 방어)으로 역전 교체 — r2의 ADV MINOR-003 처리(NOTION I-4 stale 채택)가 코드와 반대였음을 명기; 평가 경로 full forward 구분; "forward 수행+gradient 차단" 식 서술 금지 지침; **capacity-gap·안정화 논리 재점검**: bullet 3 논증과 충돌 없음 확인(비대칭 capacity는 구조 속성; 안정화 서사는 forward-skip 하에서 더 정확) | §5.5 |
| **R2-MAJ-01** | MAJOR | §0.4에 "Ablation suite(Table 3 행 2–5·7) 미실행 — Phase 5 진입 전 실행 필수(최소 행 2·7), 271 canon config 기반" bullet 신설(행 7 = bullet 3 load-bearing 명시); §6.7 행 5(FM)·행 7(symmetric)에 미실행 + conditional 규칙(행 6과 동일 — 미완 본문 잔류 금지) + 행 7 미완 시 bullet 3 "reliable signal" 주장 정성 수준 하향 지침; PAGE_BUDGET §3 Table 3 행 conditional 확장; EXPERIMENT_EXECUTION_TODO 집계는 본 fixlog §3이 대체·확장 | §0.4, §6.7; PAGE_BUDGET §3 |
| **NEW-m1** | MINOR | §0.4 "37/113" → **학습 단위 36/113 + 평가 단위 37/114** 병기·기준 혼용 금지; §6.2 완주 주석 동일 통일 | §0.4, §6.2 |
| **R2-MIN-01** | MINOR | §14 논거 ② "유일한 구조적 장치" → "실제 운영 라벨의 분포를 보존하는 **가장 직접적인** 구조적 장치" + synthetic injection 반례 차단 1구(§0.3 라벨 출처 축 정합) + "유일한" 표기 금지 지침 | §14 |
| **R2-MIN-02** | MINOR | PAGE_BUDGET §2 전략 1·§7: fallback을 우선순위 사다리로 재정렬 — fontsize/tabcolsep/약어 → Table 4 흡수 → 지표 1열화는 **최후 수단 + V3 재결정 필요**(RT V3 재개방 차단) | PAGE_BUDGET §2, §7 |
| **R2-MIN-03** | MINOR | §6.6 Table 4 **실행 사양** 신설: ① standard-split은 동일 config 그대로(**use_grl=True 유지**) — 라벨 경로 자가 비활성의 코드 근거 인용(`loss.py:293–302` `_pos_count==0 → grl_cls_loss_tensor=None` skip + force_mask 퇴화 + OD 전 패치 정상), use_grl=False 금지(dead-component dynamic margin 함정 — §6.7 경고를 Table 4에도 적용) ② contaminated 조건 baseline train = **Q3** 명시. TODO 항목 3에 두 설계 조건 추가 (red-team §1.3 권고 포함 처리) | §6.6 |
| **R2-MIN-04** | MINOR | §6.5 "데이터셋별 0.5–6.2% 수준" → "**실측 완료 데이터셋 기준** 0.5–6.2%; SMD per-machine 확정 대기" — §5.2의 상한 단정 금지 어법과 통일 | §6.5 |
| **R2-NOTE-01** | NOTE | §6.3 epoch-비대칭 방어 bullet에 권고 추가: REQUEST-4 (iii) validation-split sensitivity 소형 실험(1–2 데이터셋) + 대표 baseline epoch-budget 1점 — §B.4를 optional placeholder에서 실측으로 격상(두 방어의 마지막 보루 실질화); TODO 후보 등재(§3) | §6.3 |
| **R2-NOTE-02** | NOTE | §3.1 Para 3 에코 + §14 배치 지침에 스코핑 의무: "the standard MTSAD benchmarks we evaluate on" — 전칭 금지(Exathlon 반례 차단), Phase 4 clean-train 검증 연동 | §3.1, §14 |
| **NEW-n1** | NOTE | §6.4 `affiliation_f1_ar` 라인 "809–813" → "**811–813** (키 할당 :813)" — 정본(PROTOCOL_TRUTH REQUEST-1) 표기 통일 (evaluator.py 직접 재확인) | §6.4 |

---

## 3. EXPERIMENT_EXECUTION_TODO 집계 (r3 — fixlog r2 §7의 8항목을 대체·확장; Phase 5 진입 전 필수)

1. MAE 271 잔여 entity 완주 (SMD 6, SMAP 49, MSL 22) + baseline SMD/SMAP/MSL 재실행(per-entity 정규화 STALE).
2. weakly-supervised 4종(DeepMIL/WETAS/TreeMIL/NRdetector) GPU 전체 실험 — NRdetector는 최직접 경쟁자.
3. **Protocol-effect 실험 (Table 4)**: standard split 조건의 [MODEL]+대표 baseline, 대표 2–3 데이터셋.
   **설계 조건 (r3, R2-MIN-03)**: ① [MODEL]은 동일 config 그대로(use_grl=True 유지; `loss.py:293–302`
   pos_count==0 skip으로 라벨 경로 자가 비활성 — use_grl=False 금지) ② contaminated 조건 baseline train=Q3.
4. Label sparsity sweep (R32, p ∈ {1.0,…,0.1}) — Fig. 3 입력.
5. Warmup ablation (REQUEST-F) — 완료 시에만 Table 3 행 6 유지 (conditional).
6. **Ablation suite 실행 (r3 신설, R2-MAJ-01)**: Table 3 행 2–5·7 — **최소 행 2(w/o GRL)·행 7(symmetric
   decoder) 필수** (행 7 = contribution bullet 3 load-bearing). 271 canon config 기반. 행 2 설계 조건:
   anomaly-OD 제외 유지(dead-component dynamic margin 재활성화 차단 — 구 항목 6 흡수). 미완 행은
   conditional 규칙(본문 잔류 금지) + 행 7 미완 시 bullet 3 주장 강도 하향 (BLUEPRINT §0.4/§6.7).
7. (optional → **권고 격상**, R2-NOTE-01) Epoch-budget sensitivity + REQUEST-4 (iii) validation-split
   selection sensitivity 소형 실험(1–2 데이터셋) — §B.4를 실측으로 격상.
8. §4.5 정성 figure의 유형별 해석 — 수치 확정 후 (RT MINOR-02).

---

## 4. 검증 요약

- **코드 1차 소스 확인**: §1 A-1~A-5 전건 — 두 재리뷰의 file:line 적시와 전부 일치 (불일치 0건).
  재리뷰가 인용하지 않은 보조 사실(prev-epoch 갱신 trainer.py:1317–1319, 초기값 :190, FM 소비처
  단일성 grep)도 독립 확인.
- **정본 우선순위 준수**: 수정은 271_CONFIG_TRUTH(1순위)부터 하향 동기화 — CODEBASE/SYNTHESIS의
  관련 서술 grep 전수 후 모순 잔존 없음.
- **r2 서술의 보존 범위**: NEW-B1에서 "손실 항 ramp 없이 즉시 투입 + grad-ratio×0.2" 부분은 코드
  일치가 재확인되어 **유지** — 철회 대상은 "sigmoid 271 미사용" 단정·금지 조항과 §5.6(C) 계수
  오귀속에 한정. NEW-B2는 방향 역전 교체.
- placeholder 정책(A8): 본 r3 수정으로 신규 유입된 수치는 전부 코드 상수(0.2/250/500/2/(1+e^-10p)-1
  등)·집계 산식(36/113·37/114) — 모델 성능 수치 0건.
