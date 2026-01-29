# Phase 2 실험 계획서

> Phase 1 분석 결과를 기반으로 도출한 Phase 2 실험 계획
> 참조: [PHASE1_ANALYSIS.md](./PHASE1_ANALYSIS.md)

---

## Default Setting (고정 - Phase 1 Baseline)

| Parameter | Value |
|-----------|-------|
| d_model | 64 |
| nhead | 2 |
| seq_length | 100 |
| patch_size | 10 |
| masking_ratio | 0.2 |
| num_encoder_layers | 1 |
| num_teacher_decoder_layers | 2 |
| num_student_decoder_layers | 1 |
| dim_feedforward | 256 |
| dropout | 0.1 |
| lambda_disc | 0.5 |
| learning_rate | 0.002 |
| margin_type | dynamic |
| dynamic_margin_k | 1.5 |
| masking_strategy | patch |
| patchify_mode | patch_cnn |

---

## 실험 그룹 개요

| Group | 목적 | 관련 관점 | 실험 수 |
|-------|------|----------|---------|
| G1 | Disc Ratio 극대화 | (1) | 5 |
| G2 | DC_NA + RC_NA 동시 극대화 | (2) | 5 |
| G3 | DC_NA+RC_NA 높은 모델 × scoring × window | (3) | 4 |
| G4 | Disturbing Normal 분리 극대화 | (4) | 4 |
| G5 | PA%80 + disc_ratio 동시 최적화 | (5) | 3 |
| G6 | Window × Depth × Masking 관계 | (6) | 7 |
| G7 | Scoring Mode 차이 검증 | (7) | 3 |
| G8 | 성능 + DC_DA 동시 최적화 | (8) | 3 |
| G9 | Softplus Margin 테스트 | (9) | 5 |
| G10 | mask_after vs mask_before 비교 | (10) | 4 |
| G11 | Combined vs Recon-only 분석 | (11) | 3 |
| G12 | 추가 Insight 도출 | (12) | 4 |
| **합계** | | | **50** |

---

## 상세 실험 계획

### Group 1: Disc Ratio 극대화 (관점 1)

**Phase 1 근거**: disc_ratio 최고 모델은 nhead=1 + mr=0.10 + d128 + mask_after (DR=10.80). DC_NA=3.02.
**가설**: 낮은 attention head 수가 student의 표현력을 제한하여 teacher와의 discrepancy를 증폭. 여기에 "easy win" 파라미터(lr, dropout)를 결합하면 추가 개선 가능.

| ID | 변경 파라미터 (default 대비) | 검증 가설 |
|----|---------------------------|----------|
| P2-001 | d_model=128, nhead=1, mr=0.10, lr=0.005, dropout=0.2 | Easy win 전체 결합 시 disc_ratio 15+ 가능? |
| P2-002 | d_model=128, nhead=1, mr=0.08, lr=0.005, dropout=0.2 | mr=0.08(ROC 최고) + disc_ratio 극대화 조건 |
| P2-003 | d_model=128, nhead=1, mr=0.10, t_dec=3, s_dec=1, lr=0.005 | Decoder depth 증가 + disc 극대화 |
| P2-004 | d_model=128, nhead=1, mr=0.10, lambda_disc=1.0, lr=0.003 | Lambda 증가가 disc_ratio에 미치는 영향 |
| P2-005 | d_model=256, nhead=1, mr=0.10, lr=0.005 | 더 큰 모델에서도 disc 극대화 패턴 재현? |

### Group 2: DC_NA + RC_NA 동시 극대화 (관점 2)

**Phase 1 근거**: 067_optimal_v3 (d128, w500, nhead=4, mr=0.15, t2s1)가 DC_NA=4.12, RC_NA=3.01로 동시 최고. 핵심 조건: mask_after + all_patches + seq=500.
**가설**: seq=500이 충분한 temporal context를 제공하여 disc와 recon 분리를 동시에 높임. 여기에 세부 파라미터 조정으로 추가 개선 가능.

| ID | 변경 파라미터 (default 대비) | 검증 가설 |
|----|---------------------------|----------|
| P2-006 | d_model=128, seq=500, patch=20, nhead=4, mr=0.15, t_dec=2, lr=0.003 | 067 + 높은 LR |
| P2-007 | d_model=128, seq=500, patch=20, nhead=4, mr=0.15, t_dec=2, dropout=0.2 | 067 + 높은 dropout |
| P2-008 | d_model=128, seq=500, patch=20, nhead=4, mr=0.12, t_dec=2 | mr 미세 감소 → disc 우세 방향 이동? |
| P2-009 | d_model=128, seq=500, patch=20, nhead=8, mr=0.15, t_dec=2 | nhead 증가 → recon 개선 유지 + disc 변화? |
| P2-010 | d_model=128, seq=500, patch=20, nhead=4, mr=0.15, t_dec=3, s_dec=1 | Decoder 깊이 증가의 영향 |

### Group 3: DC_NA+RC_NA 높은 모델 × scoring × window (관점 3)

**Phase 1 근거**: DC_NA와 RC_NA 동시 높은 모델은 100% seq=500 + mask_after + all_patches. scoring mode 변경 시 ROC 0.1~0.4% 차이.
**가설**: 이 모델들에서 window를 변경하면 disc/recon balance가 변하고, scoring mode의 효과도 달라질 것.

| ID | 변경 파라미터 (default 대비) | 검증 가설 |
|----|---------------------------|----------|
| P2-011 | d_model=128, nhead=4, mr=0.15, t_dec=2 (w=100) | 067 params를 w100에 적용 → 안정성 vs 분리도 |
| P2-012 | d_model=128, seq=1000, patch=20, nhead=4, mr=0.15, t_dec=2 | w1000으로 확장 시 d128이 성능 유지? |
| P2-013 | d_model=128, seq=500, patch=20, nhead=1, mr=0.10, t_dec=2 | disc 극대화 모델 + w500 → DC_DA 개선? |
| P2-014 | d_model=128, seq=1000, patch=20, nhead=1, mr=0.10, t_dec=2 | disc 극대화 모델 + w1000 |

### Group 4: Disturbing Normal 분리 극대화 (관점 4)

**Phase 1 근거**: DC_DA 상위: 067(2.47), 008_w500_p5(2.21), 066(2.17), 070(2.16). 공통: w500, mask_after.
DC_DA 높은 모델 특성: mask_after(94.9%), all_patches(91.9%), d_model=128(16.1%), masking_ratio=0.15(10.3%).
**가설**: 긴 window에서 disturbing normal의 "일시성"과 anomaly의 "지속성"을 구분 가능. d128이 이 패턴을 더 정밀하게 포착.

| ID | 변경 파라미터 (default 대비) | 검증 가설 |
|----|---------------------------|----------|
| P2-015 | d_model=128, seq=500, patch=20, nhead=4, mr=0.15, t_dec=3, s_dec=1 | Deeper decoder가 disturbing 구분에 도움? |
| P2-016 | d_model=128, seq=500, patch=20, nhead=4, mr=0.10, t_dec=2 | mr 감소 → disc 증가 + disturbing 구분? |
| P2-017 | d_model=128, seq=500, patch=20, nhead=8, mr=0.15, t_dec=2 | nhead 증가 → attention 다양성 → disturbing 구분? |
| P2-018 | d_model=128, seq=500, patch=10, mr=0.15, t_dec=2, lr=0.003 | Smaller patch + LR boost → disturbing 구분? |

### Group 5: PA%80 + disc_ratio 동시 최적화 (관점 5)

**Phase 1 근거**: PA%80 F1 top은 전부 mask_before + last_patch + default (DR 낮음). 하지만 PA%80 + DR 동시 높은 모델은 mask_after + d128 + adaptive + 낮은 mr.
**가설**: PA%80을 높이려면 score의 temporal consistency가 중요. mask_after + d128에서 높은 DR을 유지하면서 PA%80을 개선하려면 t_dec 증가 + lr/dropout 최적화가 필요.

| ID | 변경 파라미터 (default 대비) | 검증 가설 |
|----|---------------------------|----------|
| P2-019 | d_model=128, nhead=1, mr=0.10, t_dec=3, s_dec=2, lr=0.005, dropout=0.2 | PA%80 최적화 + disc 극대화 결합 |
| P2-020 | d_model=128, nhead=4, mr=0.08, t_dec=3, s_dec=1, lr=0.003 | mr=0.08(ROC 최고) + t3s1(PA80 최적 depth) |
| P2-021 | d_model=128, nhead=16, mr=0.10, t_dec=4, s_dec=1, lr=0.003, dropout=0.2 | 기존 PA%80 모델 + easy win |

### Group 6: Window × Depth × Masking 관계 (관점 6)

**Phase 1 근거**:
- seq=500 + d128 + t_dec=2: ROC 0.952, DR 6.35
- seq=500 + d128 + t_dec=3: ROC 0.927, DR 3.68 (깊으면 오히려 하락!)
- seq=100 + t_dec=3: ROC 최적 영역
**가설**: 큰 window에서는 shallow decoder가 유리하고, 작은 window에서는 deeper decoder가 유리. masking_ratio도 window 크기에 따라 최적값이 다를 것.

| ID | 변경 파라미터 (default 대비) | 검증 가설 |
|----|---------------------------|----------|
| P2-022 | d_model=128, nhead=4, mr=0.10, t_dec=3, s_dec=1 (w=100) | w100 + t3s1 + 낮은 mr |
| P2-023 | d_model=128, nhead=4, mr=0.08, t_dec=3, s_dec=1 (w=100) | w100 + t3s1 + mr=0.08 |
| P2-024 | d_model=128, seq=500, patch=20, nhead=4, mr=0.10, t_dec=2, s_dec=1 | w500 + shallow + 낮은 mr |
| P2-025 | d_model=128, seq=500, patch=20, nhead=4, mr=0.08, t_dec=2, s_dec=1 | w500 + shallow + mr=0.08 |
| P2-026 | d_model=128, seq=500, patch=20, nhead=4, mr=0.08, t_dec=3, s_dec=1 | w500 + mid depth + mr=0.08 |
| P2-027 | d_model=128, seq=1000, patch=20, nhead=4, mr=0.10, t_dec=2, s_dec=1 | w1000 + shallow + 낮은 mr |
| P2-028 | d_model=128, seq=1000, patch=20, nhead=4, mr=0.08, t_dec=2, s_dec=1 | w1000 + shallow + mr=0.08 |

### Group 7: Scoring Mode 차이 검증 (관점 7)

**Phase 1 근거**: Adaptive/normalized가 가장 큰 차이를 보인 조건: patch_size=5(+8.3% ROC), 높은 masking_ratio(+3%), 긴 window(+6~11%). disc_ratio > 7에서 adaptive > default.
**가설**: Scoring mode 차이는 disc signal의 강도에 비례. disc_ratio가 극대화된 조건에서 scoring 차이도 극대화.

| ID | 변경 파라미터 (default 대비) | 검증 가설 |
|----|---------------------------|----------|
| P2-029 | d_model=128, patch_size=5, nhead=4, mr=0.20, lr=0.003 | Patch5 + d128에서 scoring 차이 극대화 |
| P2-030 | d_model=128, nhead=16, mr=0.20, lr=0.003 | nhead16 + d128에서 scoring 차이 |
| P2-031 | d_model=128, nhead=4, mr=0.40, lr=0.003 | 높은 mr에서 scoring mode 효과 |

### Group 8: 성능 + DC_DA 동시 최적화 (관점 8)

**Phase 1 근거**: ROC 상위 + DC_DA 상위가 겹치는 모델: 067(ROC 0.987, DC_DA 2.47). 조건: d128, w500, nhead=4, mr=0.15.
**가설**: 067 기반에서 세부 파라미터 조정으로 DC_DA를 더 높이면서 ROC 유지 가능.

| ID | 변경 파라미터 (default 대비) | 검증 가설 |
|----|---------------------------|----------|
| P2-032 | d_model=128, seq=500, patch=20, nhead=4, mr=0.15, t_dec=2, lr=0.003 | 067 + LR boost → DC_DA 개선? |
| P2-033 | d_model=128, seq=500, patch=20, nhead=4, mr=0.15, t_dec=2, dropout=0.2 | 067 + dropout → 과적합 방지 → DC_DA? |
| P2-034 | d_model=128, seq=500, patch=20, nhead=4, mr=0.12, t_dec=2, lr=0.003, dropout=0.2 | 067 + mr 미세조정 + easy win |

### Group 9: Softplus Margin 테스트 (관점 9)

**Phase 1 근거**: Phase 1은 모두 dynamic margin. Softplus margin은 미검증.
**가설**: Softplus는 gradient가 더 smooth하여 학습 안정성이 높을 수 있으나, dynamic margin의 adaptive 특성이 없어 성능이 다를 수 있음.

| ID | 변경 파라미터 (default 대비) | 검증 가설 |
|----|---------------------------|----------|
| P2-035 | margin_type=softplus (나머지 default) | Baseline에 softplus 적용 |
| P2-036 | d_model=128, seq=500, patch=20, nhead=4, mr=0.15, t_dec=2, margin_type=softplus | 최고 ROC 모델에 softplus |
| P2-037 | d_model=128, nhead=1, mr=0.10, margin_type=softplus | 최고 disc_ratio 모델에 softplus |
| P2-038 | d_model=128, nhead=1, mr=0.10, t_dec=3, s_dec=1, lr=0.005, margin_type=softplus | PA%80 최적 + softplus |
| P2-039 | d_model=128, nhead=16, mr=0.10, t_dec=4, s_dec=1, margin_type=softplus | High capacity + softplus |

### Group 10: mask_after vs mask_before 비교 (관점 10)

**Phase 1 근거**:
- seq=100~200: mask_after 우세
- seq=500: ROC 수렴, PA%80은 mask_before 우세
- seq=1000+: mask_before 우세
- mask_after: DR1 4.47 vs mask_before: DR1 1.62 (모든 seq에서 mask_after DR 높음)
**가설**: 특정 hyperparameter에서 mask_after/before 성능 gap이 확대되거나 역전되는 조건이 있을 것.

| ID | 변경 파라미터 (default 대비) | 검증 가설 | 비고 |
|----|---------------------------|----------|------|
| P2-040 | d_model=128, seq=200, patch=10, mr=0.10 | w200에서 mask_after 우위 재현 + d128 효과 | both mask |
| P2-041 | d_model=128, seq=500, patch=20, mr=0.10 | w500에서 mask_before 우위가 d128에서도? | both mask |
| P2-042 | d_model=128, seq=500, patch=20, nhead=4, mr=0.15 | 067 조건에서 mask mode 차이 | both mask |
| P2-043 | d_model=64, seq=500, patch=20, nhead=1, mr=0.20 | Small model + w500에서 mask 차이 | both mask |

### Group 11: Combined Scoring vs Recon-only (관점 11)

**Phase 1 근거**: Recon-dominant(MB+default+last) F1=0.818, PA80=0.711 vs Disc-enhanced(MA+adaptive+all) F1=0.634, PA80=0.555. 그러나 최적 모델(067)은 disc-enhanced에서 ROC 0.987.
**가설**: disc benefit은 모델이 충분히 강할 때(d128+, disc_ratio 높을 때) 극대화됨. d256이나 patch5에서 disc benefit이 더 클 것.

| ID | 변경 파라미터 (default 대비) | 검증 가설 |
|----|---------------------------|----------|
| P2-044 | d_model=256, nhead=8, mr=0.15, lr=0.003 | d256에서 disc benefit 유지? |
| P2-045 | d_model=128, patch_size=5, nhead=4, lr=0.003 | Patch5 + d128에서 disc benefit 극대화 |
| P2-046 | d_model=128, seq=500, patch=10, nhead=4, mr=0.15, lr=0.003 | w500 + smaller patch에서 disc benefit |

### Group 12: 추가 Insight 도출 (관점 12)

**Phase 1에서 미검증된 조합들**:
- 모든 easy win 동시 적용
- nhead=6 (Phase 1에서 ROC 최고였으나 sample size 제한)
- s_dec=2 효과 추가 검증

| ID | 변경 파라미터 (default 대비) | 검증 가설 |
|----|---------------------------|----------|
| P2-047 | d_model=128, nhead=4, mr=0.08, t_dec=3, s_dec=1, lr=0.005, dropout=0.2, lambda_disc=1.0 | 모든 easy win 동시 적용 |
| P2-048 | d_model=128, nhead=6, mr=0.12, t_dec=3, s_dec=1, lr=0.003 | nhead=6 최적 여부 검증 |
| P2-049 | d_model=128, nhead=8, mr=0.08, t_dec=3, s_dec=2, lr=0.003, dropout=0.2 | s_dec=2 효과 + nhead=8 |
| P2-050 | d_model=128, seq=500, patch=20, nhead=4, mr=0.15, t_dec=2, lr=0.005, dropout=0.2, lambda_disc=1.0 | 067 + 모든 easy win 동시 적용 |

---

## 실험 우선순위

### Tier 1 (최우선 - 핵심 가설 검증)
- **P2-047**: 모든 easy win 결합 → 빠른 성능 상한 확인
- **P2-050**: 최고 모델 + 모든 easy win → 0.99+ ROC 가능?
- **P2-036**: 최고 모델 + softplus → margin type 효과 확인
- **P2-001**: Disc ratio 극대화 + easy win → DR 15+ 가능?

### Tier 2 (높은 우선순위 - 핵심 관계 규명)
- **P2-006~010**: DC_NA + RC_NA 동시 극대화 탐색
- **P2-022~028**: Window × Depth × Masking 관계 체계적 규명
- **P2-040~043**: mask_after/before crossover point 정밀 파악

### Tier 3 (보통 우선순위 - 세부 최적화)
- **P2-015~018**: Disturbing normal 분리
- **P2-019~021**: PA%80 + disc_ratio 동시 최적화
- **P2-029~031**: Scoring mode 차이 요인 분석

### Tier 4 (탐색적)
- **P2-035, 037~039**: Softplus margin 추가 테스트
- **P2-044~046**: Combined scoring benefit 분석

---

## 예상 총 실험 규모

- Base configs: 50
- x2 mask modes: 100 (단, G10은 이미 both mask이므로 실제 96)
- x3 scoring modes: ~300
- x2 inference modes: ~600
- 예상 총: **약 600개 실험**

---

## Phase 1 핵심 교훈 요약

1. **disc_ratio와 recon_ratio는 trade-off 관계** (-0.55 correlation). 최적 모델은 균형점.
2. **mask_after는 disc 강화, mask_before는 recon 강화**. 둘 다 장단점 있음.
3. **seq=500 + d128 + mask_after + all_patches**에서만 disc와 recon 동시 극대화 가능.
4. **Easy win: lr=0.003~0.005, dropout=0.2, lambda=1.0** → 아직 최적 모델에 미적용.
5. **PA%80은 recon-dominant 전략(mask_before + last_patch + default)이 유리**.
6. **Adaptive scoring은 disc_ratio > 7일 때만 확실히 유리**.
7. **깊은 decoder는 w100에서 유리, w500에서는 shallow가 유리** (overfitting 방지).
8. **masking_ratio 0.08~0.15이 sweet spot**. 0.05 이하 / 0.50 이상은 성능 급감.
