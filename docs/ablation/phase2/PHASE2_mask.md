# mask_after가 discrepancy metrics를 극적으로 향상시키는 이유

## 핵심 메커니즘: 정보 비대칭의 증폭

두 모드의 차이는 **encoder가 무엇을 보는가**에서 시작됩니다.

### mask_before (기본 모드)
```
[visible patches] + [mask tokens] → Encoder → latent (전체 seq_len)
                                              ↓
                                    Teacher Decoder → teacher_output
                                    Student Decoder → student_output (latent.detach())
```

Encoder가 **mask token을 포함한 전체 시퀀스**에 self-attention을 수행합니다. Mask token이 encoder 내에서 visible patch들의 정보를 attention으로 흡수하므로, encoder 출력의 masked 위치에도 상당한 contextual 정보가 이미 존재합니다.

결과: Teacher와 Student 모두 **이미 정보가 풍부한 latent**를 받아서 디코딩 → 두 디코더의 출력 차이가 작아짐 → **disc_d가 낮음**.

### mask_after (MAE 스타일)
```
[visible patches만] → Encoder → latent_visible (num_keep만)
                                       ↓
              mask token 삽입 후 unshuffle → Teacher Decoder (teacher_mask_token)
              latent_visible.detach() + mask token 삽입 → Student Decoder (student_mask_token)
```

Encoder는 **visible patch만** 처리합니다. Masked 위치에 대한 정보가 encoder 출력에 **전혀 없습니다**. Decoder에 전달되는 masked 위치는 순수한 학습 가능 mask token뿐입니다.

## 왜 disc_d가 증폭되는가: 3가지 메커니즘

### 1. Encoder representation의 순도 차이

mask_before에서 encoder는 mask token과 visible patch을 함께 처리합니다. Self-attention에 의해 mask token 위치에도 주변 visible patch 정보가 leak됩니다. 이 "leaked information"은 teacher와 student 모두에게 동일하게 전달되므로, 두 디코더가 비슷한 출력을 만들 수 있습니다.

mask_after에서는 encoder가 visible만 처리하므로, masked 위치의 복원은 **전적으로 decoder의 능력에 의존**합니다. Teacher decoder(deeper)와 student decoder(shallower)의 능력 차이가 직접적으로 출력 차이로 나타납니다.

### 2. Mask token의 역할 변화

mask_before: mask token은 encoder를 거치며 contextual representation이 됨 → decoder 입력 시 이미 유용한 정보 포함
mask_after: mask token은 decoder 입력 시 **raw learnable parameter** 그대로 → decoder가 latent_visible의 cross-attention만으로 복원해야 함

이 차이로 인해 mask_after에서는 decoder 깊이(td)의 영향이 극대화됩니다. 코드에서 확인 가능:

- `model.py:498-501`: Teacher는 `latent_visible` + `teacher_mask_token`으로 unshuffle
- `model.py:521-522`: Student는 `latent_visible.detach()` + `student_mask_token`으로 unshuffle

Teacher와 Student가 **별도의 mask token**을 사용하므로 (`shared_mask_token=False`), masked 위치에서의 출발점 자체가 다릅니다.

### 3. Reconstruction quality와의 trade-off

Phase 2 데이터가 명확하게 보여줍니다:

| Metric | mask_after wins | mask_before wins | Mean diff |
|--------|:-:|:-:|-----|
| disc_d | **164**/300 | 136 | **+0.71** |
| recon_d | 4 | **296**/300 | **-0.96** |

mask_after에서 recon_d가 0.96이나 하락하는 이유: encoder가 visible만 처리하므로, masked 위치의 reconstruction은 decoder의 cross-attention이 visible latent에서 정보를 가져오는 것에 전적으로 의존합니다. 이것은 teacher/student 모두에게 **더 어려운 과제**입니다.

- **Teacher**: 더 깊은 decoder로 이 어려운 과제를 잘 수행 → 정상 데이터에서 좋은 복원
- **Student**: 더 얕은 decoder로 이 어려운 과제를 수행 → 정상에서도 복원 부족
- **이상에서**: Teacher도 복원 실패 → teacher-student 차이가 더 극대화

결과적으로 mask_after는 **disc signal을 키우되 recon signal을 죽이는** 근본적 trade-off를 만듭니다.

## Phase 1 vs Phase 2에서의 차이

| | Phase 1 | Phase 2 |
|--|---------|---------|
| disc_d 비율 | mask_after 3× higher | mask_after +0.71 (mean) |
| roc에서 우위 | mask_after wins (with normalized scoring) | **mask_before wins** 214/300 |
| 핵심 차이 | λ=0.5, d=64, enc=1, scoring 3종 비교 | λ=2.0, d=128, enc=1, scoring 2종 |

Phase 1에서 mask_after가 roc에서도 승리할 수 있었던 이유: **normalized scoring**이 disc signal을 자동 스케일링하여 recon 약점을 보상했습니다. Phase 2에서는 λ=2.0으로 disc weight가 이미 높아졌지만, recon_d의 절대적 하락(-0.96)을 상쇄하기에는 부족했습니다.

단, Phase 2에서도 **p=5 + mask_after** 조합은 roc=0.990으로 전체 1위를 차지합니다. 이 경우 disc_d=4.64로 극도로 높아서 recon_d 하락을 완전히 압도합니다. 이것이 가능한 조건: 100개의 미세한 패치 → teacher-student 정보 격차 극대화.

## 결론

mask_after가 discrepancy를 극적으로 높이는 근본 원인은 **encoder 단계에서 masked 위치의 정보를 완전히 차단**하여, decoder 능력 차이(teacher > student)가 출력 차이로 직결되도록 만들기 때문입니다. 이것은 original MAE 논문의 설계 의도(encoder는 visible만 처리)와 일치하며, self-distillation에서는 teacher-student 간 능력 비대칭을 최대한 활용하는 효과를 냅니다.
