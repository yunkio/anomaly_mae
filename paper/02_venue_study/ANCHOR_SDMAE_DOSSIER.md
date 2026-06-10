---
phase: 2
agent: anchor-paper-analyst (SDMAE)
directives: [R9, R21]
last_modified: 2026-06-11
revision: r2 (fixer — adversarial review paper/99_reviews/p2_dossiers_r1.md 전수 반영: S-M1/X-M1 MAJOR + S-m1–S-m6 MINOR; 전 수정은 arXiv HTML 2306.12041v2 원문 재확인 후 적용; fixlog: p2_fixlog_r2.md; 정정 이력은 말미 부록)
---

**경고: 이 문서의 verbatim 발췌는 분석 전용 — 논문 본문으로 복사 금지 (A2)**

---

# Core Anchor Reference Dossier: SDMAE

## 1. Source Card

| 항목 | 내용 |
|------|------|
| 제목 | Self-Distilled Masked Auto-Encoders are Efficient Video Anomaly Detectors |
| 저자 | Nicolae-Catalin Ristea, Florinel-Alin Croitoru, Radu Tudor Ionescu, Marius Popescu, Fahad Shahbaz Khan, Mubarak Shah |
| 게재 venue | CVPR 2024 (IEEE/CVF Conference on Computer Vision and Pattern Recognition) |
| 발표 유형 | 정규 conference paper, **Poster** (CVPR 2024 virtual 페이지 "2024 Poster" 직접 확인 — fixer r2, S-m5) |
| arXiv ID | 2306.12041 |
| arXiv DOI | https://doi.org/10.48550/arXiv.2306.12041 |
| 공식 페이지 | https://cvpr.thecvf.com/virtual/2024/poster/30615 |
| arXiv 페이지 | https://arxiv.org/abs/2306.12041 |
| HTML 풀텍스트 | https://arxiv.org/html/2306.12041v2 |
| 코드 저장소 | https://github.com/ristea/aed-mae |
| v1 제출일 | 2023년 6월 21일 |
| v2 (CVPR 최종) | 2024년 3월 9일 |
| 약칭 (본 문서용) | SDMAE |

---

## 2. Verification Log

| 검증 항목 | 상태 | 출처 |
|-----------|------|------|
| CVPR 2024 게재 | 확인 | cvpr.thecvf.com/virtual/2024/poster/30615 |
| 저자 전원 (6인) | 확인 | arxiv.org/abs/2306.12041 |
| arXiv ID 2306.12041 | 확인 | arxiv.org/abs/2306.12041 |
| 코드 공개 | 확인 | github.com/ristea/aed-mae |
| conference paper (not workshop) | 확인 | CVPR main track |
| 평가 벤치마크 (Avenue/ShanghaiTech/UBnormal/Ped2) | 확인 | arxiv.org/html/2306.12041v2 |
| 3M 파라미터, 0.8 GFLOPs | 확인 | arxiv.org/html/2306.12041v2 |
| 1655 FPS | 확인 | arxiv.org/html/2306.12041v2 |
| 마스킹 비율 (숫자) | 미확인 — 논문에서 수치 미공개 | arxiv.org/html/2306.12041v2 반복 조회 |
| 발표 유형 Poster | 확인 (fixer r2, S-m5) | cvpr.thecvf.com/virtual/2024/poster/30615 — "2024 Poster" 표기 |
| anomaly-map 예측 분기 (클린 프레임 + 픽셀 anomaly map 동시 재구성) | 확인 (fixer r2, S-M1 — §3.6-2 신설) | arxiv.org/html/2306.12041v2 §1 기여 목록 + §3 "Synthetic anomalies" 단락 + §4 ablation 논의 |
| "self-distillation" 용어의 선행 귀속 [101] = Zhang et al., TPAMI 2022 | 확인 (fixer r2, S-m2) | arxiv.org/html/2306.12041v2 reference list bib101 직접 확인 |

---

## 3. 방법 구조 상세 분석

### 3.1 아키텍처 개요

SDMAE는 비디오 프레임 단위의 masked autoencoder를 기반으로 한다. 표준 ViT 블록 대신 convolutional vision transformer (CvT) 블록을 사용하며, 전체 구성은:

- **공유 인코더**: CvT 3블록, projection 크기 256, 어텐션 헤드 4개
- **Teacher decoder**: CvT 3블록 (4 어텐션 헤드, **projection 크기 128**)
- **Student decoder**: CvT 1블록 (4 어텐션 헤드, projection 크기 128) — teacher decoder의 첫 번째 블록 이후에서 분기

정정 (fixer r2, S-m3): decoder projection 128은 teacher·student **공통**이다 — 원문: "All decoder blocks have four attention heads and a projection dimension of 128." (초판처럼 student에만 128을 병기하면 teacher decoder=256으로 오독될 수 있음. 256→128 비대칭은 encoder–decoder 간이지 teacher–student 간이 아님.)

Student decoder는 독립적인 separate 네트워크가 아니라, teacher decoder에서 중간 분기(branch-off) 형태로 연결된다.

데이터셋별 패치 크기:
- Avenue: 16×16
- ShanghaiTech, UBnormal: 8×8
- UCSD Ped2: 4×4

### 3.2 마스킹 전략

입력 이미지를 non-overlapping token으로 분할하고 임의 선택된 일부 토큰을 제거한다. 마스킹된 토큰은 mask token으로 대체하여 decoder에 전달된다. **마스킹 비율은 논문에 수치로 명시되지 않음** (논문 전체 검색 결과 미발견). 중요한 점: 마스킹 토큰 선택은 여전히 랜덤이지만, 손실 함수는 모션 그래디언트 크기에 따라 가중된다 — 즉, 마스킹 자체가 이상 위치 중심으로 구성되는 것이 아님.

Verbatim: "Although our reconstruction loss focuses on tokens with high motion, the masked tokens are still chosen randomly." (Section 3)

### 3.3 모션 그래디언트 가중치

가중치 w_i^(t)는 각 패치 내 최대 그래디언트 크기의 채널별 평균으로 정의되며, 모든 토큰에 대해 정규화:

w_i^(t) = m_i^(t) / Σ_{j=1}^{n} m_j^(t)

가중 MSE 손실 (Teacher 학습용):
L_wMSE(x_t, θ_T) = (1/n) Σ_{i=1}^{n} w_i^(t) · ||p_i^(t) - p̂_i^(t)||²

이는 배경보다 움직이는 전경 객체에 손실을 집중시키는 역할 — **시계열 도메인에 직접 대응 없음**.

**정정 (fixer r2, S-M1-b)**: "손실이 모션 그래디언트 크기에 따라 가중된다"는 서술은 **원본(비증강) 프레임에 한해** 정확하다. 합성 이상이 합성된 증강 프레임에서는 GT anomaly map이 그래디언트에 **가산**된 후 가중치가 계산된다 (§3.6-2 요소 ③ — 원문: "we propose to add the anomaly maps and the gradients together, before computing the weights as in Eq. (2)").

### 3.4 Self-Distillation 적용 방식 (R21 핵심)

**2단계 학습 절차:**

**Phase 1 (Teacher 학습, 100 에폭)**:
Teacher가 가중 MSE 손실 L_wMSE로 원본 프레임의 마스킹된 패치를 재구성하도록 훈련.

**Phase 2 (Student 학습, 40 에폭)**:
Verbatim: "In the second phase, we freeze the weights of the shared backbone and train only the student decoder via self-distillation." (Section 3)

Student의 목표는 원본 프레임이 아닌 teacher의 출력을 재구성하는 것:

Verbatim: "The main difference is that instead of reconstructing the patches from the real image, the student learns to reconstruct the ones produced by the teacher." (Section 3)

Self-Distillation 손실:
L_SD(x̂_t, θ_S) = (1/n) Σ_{i=1}^{n} w_i^(t) · ||p̂_i^(t) - p̃_i^(t)||²

여기서 p̂ = teacher 재구성, p̃ = student 재구성.

### 3.5 Self-Distillation 명명 근거 (R21 핵심 원문)

핵심 정의 verbatim:
"To reduce our processing time, we use a shared encoder for the teacher and student models, leading to a process known as self-distillation [101]." (**Section 1 Introduction, 기여 목록** — 전문 유일 출현)

**정정 (fixer r2, S-m1/S-m2)**: 초판은 위 인용의 출처를 "(Section 3)"으로 오기했고 말미 인용 마커 "[101]"을 무표기 탈락시켰다. 원문 재확인 결과 이 문장은 §1 Introduction의 기여(contribution) 목록("Third, we integrate a teacher decoder and a student decoder…" 항목) 안에 있으며, **[101] = Linfeng Zhang, Chenglong Bao, Kaisheng Ma, "Self-Distillation: Towards Efficient and Compact Neural Networks", IEEE TPAMI 44(8):4388–4403, 2022** (reference list 직접 확인). 즉 SDMAE는 "self-distillation"이라는 명칭 자체를 선행 연구 [101]에 귀속시킨다. 보강 근거 (Supplementary §6.2 Extended Related Work): "…does not even cite the work of Zhang et al. [101], **which introduces the form of self-distillation that inspired our work**."

구조적 설명 verbatim:
"A student decoder branches out from the teacher after the first transformer block of the main decoder, adding only one extra transformer block (as shown in Figure 1)." (Section 3 — fixer r2: 말미 괄호구 절단 복원)

Related work에서 전통적 self-distillation 정의와 자신의 용례 구분:
"Self-distillation [101] attaches multiple classification heads at various depths to boost the classification performance of a neural classifier. In contrast, we integrate self-distillation into a masked AE, employing two decoders of different depths." (Section 2, Related Work — fixer r2: [101] 마커 복원)

"Distinct from the aforementioned studies, to our knowledge, we are the first to introduce a variant of self-distillation in anomaly detection." (Section 2, Related Work — fixer r2: 두문 절단·무표기 대문자화 복원)

전통적 knowledge distillation과의 차이를 명시:
"Knowledge distillation [6, 32] was originally designed to compress one or multiple large models (teachers) into a lighter neural network (student). Recently adopted in anomaly detection [8, 12, 16, 26, 74, 86], knowledge distillation was deemed useful due to the possibility of leveraging the representation discrepancy between the teacher and the student networks, which is larger in the case of anomalies." (Section 2 — fixer r2: 인용 클러스터 마커 2곳 복원)

**SDMAE의 self-distillation 핵심 로직**: "since the teacher and student models are both trained on normal data, their reconstructions should be very similar for normal test samples. However, their behavior is not guaranteed to be similar on abnormal examples. Therefore, the magnitude of the teacher-student output gap (discrepancy) can serve as a means to quantify the anomaly level of a given sample." (Section 3 — fixer r2: 말미 절단 복원)

### 3.6 합성 이상 데이터 보강 및 분류 헤드

학습 영상에 UBnormal 데이터셋의 픽셀 레벨 annotated 이상 이벤트를 overlay하여 합성 이상을 생성 (증강 확률 25%). 공유 인코더의 최종 [CLS] 토큰에 이진 분류 헤드(binary cross-entropy loss)를 적용하여 합성 이상 포함 여부를 분류하도록 훈련. 이 분류 헤드는 추론 점수에도 기여함.

**중요**: 이 분류 헤드는 표준 classifier이며 **Gradient Reversal Layer(GRL)가 없음**. 우리 TSMAE의 GRL 기반 이상 정보 억제 설계와 근본적으로 다름.

### 3.6-2 Anomaly-map 예측 분기 + "이상 무시(overlook)" 재구성 타깃 (fixer r2 신설, S-M1)

초판이 전체 누락했던 핵심 메커니즘 (arXiv HTML 원문 §1 기여 목록·§3 "Synthetic anomalies" 단락·§4 ablation 논의 직접 재확인). 합성 이상 증강(§3.6)은 분류 헤드에만 쓰이는 것이 아니라, **재구성 타깃과 손실 가중치에 (합성) 이상 라벨 신호를 직접 주입하는 3요소**로 확장된다:

**① Anomaly-map 타깃 채널 — 클린 프레임 + 픽셀 anomaly map 동시 재구성**
Verbatim: "task the masked AE model to jointly reconstruct the original frames (without anomalies) and the corresponding pixel-level anomaly maps." (Section 1, 기여 목록)
Verbatim: "we add the anomaly map as an additional channel to our target image. In the anomaly map, we set normal pixels to 0 and abnormal pixels to 1. This change implies that, in Eq. (3) and Eq. (4), all patches will have an additional channel." (Section 3)

**② "이상 무시" 재구성 GT — 합성 증강 프레임의 재구성 ground-truth는 이상이 제거된 원본**
Verbatim: "consider the original training frames (without superimposed anomalies) as the ground-truth, essentially forcing our model to overlook the anomalies." (Section 3)

**③ GT anomaly map의 손실 가중치 가산 (Eq. 1–2 수정)** — 합성 이상은 모션 그래디언트가 낮을 수 있으므로 GT anomaly map을 그래디언트에 가산한 후 가중치 계산:
Verbatim: "we propose to add the anomaly maps and the gradients together, before computing the weights as in Eq. (2)." (Section 3)

**Ablation상 필수 component** — anomaly-map 예측은 부속 옵션이 아니라 Avenue 90% 돌파의 필수 요소로 보고됨:
Verbatim: "to surpass the 90% milestone on Avenue, it is mandatory to introduce the prediction of anomaly maps in the learning task." (Section 4, Table 2 ablation 논의)

**TSMAE 관점 함의 (R9 위험 분석 직결)**: SDMAE도 (합성) 이상 라벨 신호를 학습에 직접 주입해 "모델이 이상을 표현/복원하지 못하도록(overlook)" 유도한다 — TSMAE의 GRL 억제와 **개념적 평행선**이 성립한다. 유사/차이 정리는 §4.1 신설 행·§4.2, 위험도 평가와 방어 구조는 §7-2, overextension 경고는 §8 참조.

### 3.7 이상 점수 공식 (추론)

o_t = α · ||x_t - x̂_t||² + β · ||x̂_t - x̃_t||² + γ · ŷ_t

- x_t: 원본 프레임
- x̂_t: teacher 재구성
- x̃_t: student 재구성
- ŷ_t: 분류 헤드 출력
- α=0.4, β=0.3, γ=0.3 (모든 데이터셋 동일)

공간적: 픽셀 레벨 점수에 3D 시공간 필터링 적용. 프레임 레벨: 픽셀별 최댓값 + 시간축 가우시안 스무딩.

---

## 4. 유사/차이 지점 내부 정리 (R9 전략용)

### 4.1 구조적 유사점 (리뷰어 위험도 포함)

| 유사 항목 | 구체 내용 | 위험도 |
|-----------|-----------|--------|
| **마스킹된 autoencoder 기반** | 두 방법 모두 masked AE를 핵심 구조로 사용 | 높음 |
| **Teacher decoder + Student decoder 이중 decoder** | 동일한 "두 개의 decoder" 구조 | 높음 |
| **Teacher-student discrepancy를 이상 점수에 사용** | o_t에 ||teacher_out - student_out||² 항목 포함 | 높음 |
| **2단계 학습 (teacher-first warmup 후 student 학습)** | Phase 1: teacher 단독 학습, Phase 2: student 학습 | 높음 |
| **teacher 재구성 오차 + teacher-student discrepancy를 점수로 결합** | 두 항목을 선형 결합한 점수 공식 | 높음 |
| **이상 라벨 신호의 학습 주입 — "이상을 표현하지 않도록" 유도** (fixer r2 신설, S-M1) | SDMAE: (합성) 이상 라벨을 재구성 GT(이상-제거 원본, "overlook the anomalies")·anomaly-map 타깃 채널·손실 가중치 가산에 직접 주입 (§3.6-2). TSMAE: 실제 labeled anomaly를 GRL로 student hidden에서 adversarial 억제 | **높음** — 리뷰어가 "라벨 신호로 이상 무시/억제를 유도하는 학습"을 동일 아이디어로 묶을 수 있음. 방어 3축(주입 계층·라벨 출처·작동 지점)은 §7-2 |
| **비대칭 decoder 깊이 (teacher 더 깊음)** | SDMAE: teacher 3블록/student 1블록; TSMAE: teacher 3L/student 2L | 중간 |
| **공유 encoder** | 인코더 가중치를 두 decoder가 공유 | 중간 |
| **"self-distillation"이라는 용어 사용** | 동일한 용어 사용 의도 | 중간 |
| **Transformer 기반 encoder** | 두 방법 모두 Transformer 블록 사용 | 낮음 (범용) |
| **Patchify 전처리** | 입력을 patch 단위로 분할 | 낮음 (MAE 표준) |

### 4.2 실질적 차이점

| 차이 항목 | SDMAE | TSMAE | 중요도 |
|-----------|-------|-------|--------|
| **도메인** | 비디오 프레임 (공간, RGB 이미지 패치) | 다변량 시계열 (시간축, 센서 채널) | 매우 높음 |
| **레이블 설정** | 비지도 학습 (합성 이상으로 pseudo-supervision) | 실제 anomaly 라벨 활용 — **설정(가정)은 "대부분 unlabeled + 소수 labeled anomaly"(R11)이고, main 271 구현은 train 구간 라벨이 전부 존재하는 label 가용성 상한 케이스. "반지도/PU" 명명은 Phase 3 결정 사안이며 라벨 희소화 sweep은 계획 단계(R32)** (RESEARCH_SYNTHESIS §②-1/②-2/②-3/②-6 — fixer r2, X-M1 정정: 초판의 "반지도/PU 학습" 확정 서술은 정본 보류 판정과 충돌) | 높음 |
| **이상 레이블 활용 방식** | UBnormal에서 가져온 합성 이상을 오버레이하여 분류 헤드 학습 **+ 재구성 타깃/가중치에 주입 (§3.6-2)** | 실제 labeled anomaly 위치의 patch를 student 학습에서 제외 | 높음 |
| **이상 라벨 신호의 주입 계층** (fixer r2 신설, S-M1) | **타깃/손실 공간** — 이상-제거 GT 재구성 + anomaly-map 타깃 채널 + 가중치 가산 (라벨은 UBnormal 합성 pseudo-label) | **기울기 공간** — GRL gradient reversal로 student hidden의 이상 표현을 adversarial 억제 (라벨은 실제 운영 라벨) | 높음 |
| **GRL (Gradient Reversal Layer)** | 없음 — 표준 이진 분류 헤드 | 있음 — student hidden에서 이상 정보를 능동적으로 억제 | 높음 |
| **분류 헤드 목적** | 합성 이상 포함 여부 이진 분류 (감지 보조) | GRL로 이상 표현 억제 (정상 재구성 강제) | 높음 |
| **마스킹 전략** | 랜덤 마스킹 + 모션 기반 손실 가중치 | labeled anomaly 패치 우선 마스킹 (15%) | 높음 |
| **Student 학습 타깃** | Teacher 출력 재구성 (teacher-student mimicry) | Teacher 출력 모방 (단, known-anomaly patch 제외 처리) | 중간 |
| **모션 그래디언트 가중치** | 있음 (비디오 전경 강조) | 없음 (시계열에 해당 없음) | 중간 |
| **합성 이상 생성** | 있음 (UBnormal에서 crop하여 overlay) | 없음 | 중간 |
| **점수 구성 3항 (γ·ŷ_t)** | 있음 (분류 헤드 출력 직접 포함) | 없음 | 중간 |
| **적응적 스케일 조정** | 고정 계수 (α=0.4, β=0.3, γ=0.3) | 적응형 스케일 조정 (4:1 비율) | 중간 |
| **처리 단위** | 단일 프레임 (2D 공간) | 시간 window (1D 시간축, 다변량) | 높음 |
| **평가 지표/벤치마크** | (micro/macro) AUC on video surveillance benchmarks | **PA%K-AUC F1(best-epoch 선정 지표) + VUS-ROC/PR + Affiliation F1 (+PA%K-AUC PR)** on time-series benchmarks (SWaT, WaDi, PSM, SMD 등) — roc_auc도 병산되나 대표 지표 아님 (fixer r2, S-m6 정정: 초판 "AUROC" 단독 표기는 271_CONFIG_TRUTH §VIII·EXPERIMENT_PROTOCOL_TRUTH "논문 5지표"와 불일치; NRdetector dossier의 본 연구 서술과 정합화) | 높음 |
| **입력 모달리티** | RGB 이미지 (224×224 등) | 다변량 숫자 시계열 (8+ 채널) | 매우 높음 |
| **2단계 학습의 teacher/backbone 동결** (fixer r2 신설 — 리뷰 B-3 참고 항목 채택) | Phase 2에서 shared backbone **동결** ("we freeze the weights of the shared backbone") | warmup 후에도 teacher 계속 학습 (`freeze_teacher_after_warmup=False` — 271_CONFIG_TRUTH §VI INACTIVE 확인) | 중간 — §4.1 "2단계 학습" 유사 행의 비대칭을 보강하는 차이점 재료 |

---

## 5. R21 방어 논리 원재료

### 5.1 SDMAE의 "self-distilled" 용례 요약

SDMAE는 "self-distillation"을 다음 구조로 정의한다:
1. 하나의 shared encoder에서 두 decoder(teacher/student)가 분기
2. Student가 외부 teacher 모델이 아닌 동일 네트워크 내 teacher decoder의 출력을 학습 타깃으로 삼음
3. 즉, 지식이 같은 모델(architecture) 내부에서 전달됨 — "self" (외부 모델 없음)

이 정의는 전통적 self-distillation (Born Again Networks: 동일 네트워크를 iterative하게 재훈련; BYOL류: momentum encoder와 online encoder 간 자기 지도)과 다르지만, SDMAE는 이 구조를 "a variant of self-distillation"이라 부르며 **anomaly detection 도메인에서 동일 구조에 이 용어를 사용한 선례**가 되었다 — 단, 용어 자체를 창안한 것은 아니다(아래 용어 계보).

**용어 계보 (fixer r2, S-m2 — R21 방어 보강)**: SDMAE는 "self-distillation"이라는 명칭을 자체 창안(coining)하지 않고 **선행 연구 [101] = Zhang, Bao & Ma, "Self-Distillation: Towards Efficient and Compact Neural Networks" (IEEE TPAMI 44(8):4388–4403, 2022)에 명시적으로 귀속**시킨다 ("a process known as self-distillation [101]", §1; Supplementary §6.2: "the work of Zhang et al. [101], which introduces the form of self-distillation that inspired our work"). 따라서 R21 방어는 "SDMAE 단독 선례"가 아니라 **용어 계보(Zhang et al. 원류 → SDMAE의 AD 도메인 variant → 본 연구의 시계열 확장)** 위에 선다 — 단일 논문 의존보다 강한 방어 구조다. 정확한 서술: "SDMAE가 이 용어를 coining했다"(초판 뉘앙스)가 아니라 "SDMAE가 동일 구조에 이 용어를 사용한 선례이며, 용어의 선행 계보([101])가 존재한다".

### 5.2 우리 TSMAE 용례와의 연속성

TSMAE에서 "self-distillation"이 의미하는 것:
- 동일한 shared encoder 내에서 teacher decoder(3L)와 student decoder(2L)가 함께 존재
- Student는 외부 모델이 아닌 동일 네트워크의 teacher decoder 출력을 학습 목표로 삼음
- 이는 SDMAE의 용례와 **구조적으로 동일한 논리**에 해당

방어 논리 경로 (fixer r2, S-m2/X-M1 정밀화):
1. SDMAE (CVPR 2024)가 anomaly detection에서 이 구조에 "self-distillation"(variant) 용어를 **적용한 선례**임을 명시 — 용어 자체의 원류는 Zhang et al. [101] (TPAMI 2022)로 거슬러 올라간다 (§5.1 용어 계보; SDMAE 원문도 "first to introduce **a variant of** self-distillation in anomaly detection"으로 한정)
2. 우리 구조는 동일한 architectural rationale (shared encoder + asymmetric dual decoder, 내부 지식 전달)을 따름
3. 도메인(video vs 시계열)과 목적(efficiency vs label-informed anomaly suppression — 설정 명명 "semi/PU"는 Phase 3 결정 사안, RESEARCH_SYNTHESIS §②-6)이 다르지만, self-distillation 개념의 적용 방식은 동일 선례 범위에 속함

### 5.3 리뷰어 이의 대응

예상 이의: "self-distillation은 보통 teacher와 student가 별개의 독립적 모델이어야 하는 것 아닌가?"

대응:
- SDMAE는 CVPR 2024에서 공유 encoder 기반 분기 구조를 명시적으로 self-distillation이라 칭하고 출판됨
- 이 선례가 우리 용어 선택의 정당성 근거로 직접 인용 가능
- 용어의 경계는 SDMAE가 이미 확장한 범위 내에 있음
- **용어 계보가 존재 (fixer r2, S-m2)**: Zhang et al. [101] (TPAMI 2022)이 "self-distillation"을 도입했고, SDMAE가 이를 AD의 shared-encoder 분기 구조로 변형·적용 — 우리 용례는 이 확장 계보의 시계열 연장선이다. "독립 모델이어야 한다"는 이의는 이미 두 단계의 출판 선례(원류 + AD variant)에 의해 약화됨

---

## 6. Claim Support Map

### 6.1 SDMAE로 지지 가능한 주장

| 우리 논문 주장 | SDMAE 지지 가능 여부 | 근거 |
|---------------|----------------------|------|
| MAE 기반 anomaly detection이 효과적임 | 지지 가능 | CVPR 2024 결과, 4개 벤치마크 SOTA급 성능 |
| Teacher-student discrepancy가 anomaly signal로 유효 | 지지 가능 | Section 3의 이론적 근거 + 실험 결과 |
| "Self-distillation"이라는 용어가 이 구조에 적절히 적용됨 | 지지 가능 (R21 핵심) | SDMAE가 동일 논리로 동일 용어 사용, CVPR 2024 출판 |
| 비대칭 decoder 깊이(teacher > student)가 discrepancy 생성에 유효 | **간접 지지 (분기 구조 전제하의 결과)** — fixer r2, S-m4 강등: 깊이 비대칭 자체의 직접 ablation은 원문에 없음 | Table 3 인용("best micro AUC is obtained when we combine the teacher reconstruction error with the teacher-student discrepancy")은 **점수 결합 전략**의 유효성 근거이지 깊이 비대칭의 근거가 아님 (아래 "결합 점수" 행과 근거 중복) |
| Teacher 재구성 오차 + teacher-student discrepancy 결합 점수가 유효 | 지지 가능 | SDMAE 점수 공식과 원리 동일 |
| 2단계 학습(teacher-first warmup → student 학습) 구조가 유효 | 지지 가능 | SDMAE의 Phase 1/Phase 2 동일 절차 |

### 6.2 SDMAE로 지지할 수 없는 주장

| 우리 논문 주장 | 이유 |
|---------------|------|
| 우리 방법이 시계열 anomaly detection에서 유효하다 | SDMAE는 비디오 도메인 — 직접 전이 불가, 우리 실험 결과로 입증해야 함 |
| GRL 기반 이상 정보 억제가 유효하다 | SDMAE에 GRL 없음 — 독립적으로 justify 필요. **단 (fixer r2, S-M1): "이상 라벨 신호를 주입해 모델이 이상을 표현/복원하지 못하게 유도"하는 일반 아이디어 수준에서는 SDMAE의 anomaly-map 분기·overlook 재구성(§3.6-2)이 합성-라벨 기반 평행 선례 — 우리 novelty 주장은 "기울기 공간의 adversarial 억제(GRL) + 실제 라벨"로 스코핑해야 안전 (§7-2)** |
| 실제 labeled anomaly를 활용하는 설정에서의 성능 (명명은 Phase 3 결정 — X-M1) | SDMAE는 합성 이상 기반 비지도 설정 — 다름 |
| 마스킹이 labeled anomaly 위치 우선이어야 한다 | SDMAE는 랜덤 마스킹 — 우리 설계 근거는 별도 제시 필요 |
| 우리 방법이 SDMAE보다 우월하다 | 도메인이 달라 직접 비교 불가 |
| 합성 이상 없이도 성능이 유지된다 | SDMAE에서 합성 이상 기여도가 ablation에 나타남 — 우리 설정은 다름 |

---

## 7. R9 포지셔닝 전략 제안

### 배경 제약 (R9 Directive 요지)
"차이점을 나열하는 방식은 오히려 해당 차이점 말고는 매우 유사한 방법론이라고 받아들여져 novelty가 적어보일 수 있음. 숨기지는 않되, 자연스럽게 언급하고 넘어가는 방식으로, 해당 논문을 지나치게 강조하지 않도록."

### 옵션 A: Related Work에서 1-2문장, 흐름 내 자연 삽입

**위치**: Related Work의 "Knowledge Distillation in Anomaly Detection" 소단락 말미

**톤**: 선행 연구로 자연스럽게 언급, 비교 강조 없이 흐름 연결

**예시 초안**:
"Recent work has explored masked autoencoders for anomaly detection, including Ristea et al. [SDMAE], who integrate a shared-encoder dual-decoder structure into video anomaly detection — an approach they term self-distillation. Our work extends this paradigm to multivariate time-series under a semi-supervised PU setting, where known anomaly information actively guides both masking and decoder training through gradient reversal."

**장점**: 자연스럽게 인용, 우리 방법으로 흐름 연결, 차이를 "나열"하지 않고 한 문장에 포함
**단점**: 리뷰어가 두 방법을 직접 비교하려 할 경우 불충분하다고 느낄 수 있음

### 옵션 B: Related Work에서 언급 후 Method에 각주 1개

**위치**: Related Work 1문장 + Method §의 self-distillation 설명 아래 각주

**Related Work 문장 (1문장)**:
"Ristea et al. [SDMAE] adopt a masked AE with dual decoders for video anomaly detection, applying the term self-distillation — following Zhang et al. [101의 우리 측 인용 키] — to shared-encoder architectures where a student decoder mimics a deeper teacher decoder."

(정정, fixer r2 S-m2: 초판 초안의 "coining the term self-distillation"은 사실과 어긋남 — SDMAE 원문이 용어를 선행 [101]에 귀속하므로 "applying/adopting/extending the term"으로 완화. 'coining' 계열 표현은 Phase 5에서 사용 금지.)

**각주 (Method 내)**:
"We follow the self-distillation terminology of [SDMAE], extending it to the time-series domain where the student decoder is additionally trained to exclude anomaly-specific information via gradient reversal — a mechanism absent in [SDMAE]'s unsupervised video setting."

**장점**: 용어 선택의 근거를 각주에서 방어, R21 방어 논리를 Method 내에 자연스럽게 탑재
**단점**: 각주가 주의를 끌 경우 오히려 비교를 유도할 수 있음

### 옵션 C: Related Work에서 Knowledge Distillation 계보 맥락으로 언급 (권장)

**위치**: Related Work 소단락, distillation 계보 흐름 내

**구성**: (1) 전통적 KD → (2) 이상탐지에서의 teacher-student (기존 방법들) → (3) SDMAE → (4) 우리 방법으로 이어지는 흐름에서 자연스럽게 포함

**예시 초안**:
"Knowledge distillation has been applied to anomaly detection by exploiting the representation gap between pre-trained teacher and randomly initialized student networks [...]. A related line of work integrates this idea directly into reconstruction-based models: Ristea et al. [SDMAE] embed a dual-decoder structure within a masked autoencoder for video anomaly detection. In this work, we apply analogous self-distillation principles to the time-series domain, augmented with a PU learning framework that leverages scarce labeled anomalies through targeted masking and gradient-based information suppression."

**장점**: SDMAE가 계보 중 하나로 자연스럽게 위치함, 우리 방법이 독립적 기여로 읽힘, 차이 나열 없음
**단점**: SDMAE를 "related line" 수준으로 처리하기 때문에 리뷰어가 더 깊은 비교를 요청할 수 있음

**권장 옵션: C** — 계보 흐름 내 삽입이 자연스러우며, 우리 방법의 contribution이 독자적으로 읽힘. 각주(옵션 B 각주)를 Method 내 self-distillation 정의 근처에 추가하여 R21 방어 논리도 확보.

### 7-2. 추가 위험 분석 (fixer r2 신설, S-M1): anomaly-map 분기와 TSMAE GRL의 개념적 평행

**위험 시나리오**: 리뷰어가 §3.6-2의 메커니즘(이상-제거 GT 재구성 + anomaly-map 예측 + 가중치 가산 — ablation상 필수 component)을 들어 "SDMAE도 이상 라벨 신호를 학습 신호로 주입해 모델이 이상을 '무시'하도록 유도한다 — TSMAE의 GRL과 본질적으로 같은 아이디어"라고 주장할 수 있다. 초판처럼 이 분기를 누락한 채 "SDMAE에 GRL 없음 — 라벨 활용은 SDMAE로 지지 불가·독립 justify"(§6.2)와 §4.2의 GRL 차이 행만 내세우면 이 지점에서 기습당한다.

**방어 구조 (차이 3축)**:
1. **주입 계층**: SDMAE는 타깃/손실 공간(재구성 GT 교체 + 타깃 채널 추가 + 가중치 가산) — 표준적인 지도 신호의 형태다. TSMAE는 기울기 공간(GRL gradient reversal로 student hidden에서 이상 표현을 adversarial하게 억제) — 신호의 부호 자체를 뒤집는 적대적 메커니즘으로, SDMAE에는 대응물이 없다.
2. **라벨 출처/의미**: SDMAE의 라벨은 UBnormal에서 합성·오버레이한 pseudo-label로, 설정은 비지도로 유지된다. TSMAE의 라벨은 실제 운영 환경의 labeled anomaly로, 라벨 활용 자체가 문제 설정의 핵심 자원이다 (설정 명명은 Phase 3 결정 — X-M1 단서).
3. **작동 지점**: SDMAE는 "출력이 무엇이어야 하는가"(이상 없는 프레임 + anomaly map)를 지정한다. TSMAE는 "내부 표현이 무엇을 담아서는 안 되는가"(student hidden의 anomaly-identity)를 지정한다 — discrepancy 신호의 형성 기전이 다르다.

**서술 지침 (Phase 3/5)**: GRL 기여의 novelty는 "이상 라벨 신호의 학습 주입"이라는 일반 아이디어가 아니라 위 3축으로 스코핑된 메커니즘 수준에서 주장할 것. R9 원칙(차이 나열 금지)에 따라 본문/Related Work에서 anomaly-map 분기를 상술할 필요는 없으나, 리뷰어 rebuttal 대비 재료로 본 절과 §3.6-2를 유지한다.

---

## 8. Overextension 위험 분석

| 위험 유형 | 구체 내용 |
|-----------|-----------|
| 도메인 전이 과잉 주장 | SDMAE 성능을 근거로 "MAE가 일반적으로 anomaly detection에 효과적"이라고 쓰는 것 — 비디오에 한정된 결과임 |
| GRL 정당화에 SDMAE 인용 | SDMAE에 GRL 없음 — GRL 기여는 별도 justify 필요 |
| 레이블 활용 정당화에 SDMAE 인용 | SDMAE는 합성 이상(비지도) 사용 — 실제 labeled anomaly 활용 효과는 SDMAE로 지지 불가 |
| "우리가 더 낫다"는 암시 | 도메인이 달라 직접 비교는 부적절 |
| self-distillation 용어가 완전히 확립됐다고 주장 | SDMAE가 "variant"라고 표현 — 표준 정의로 인용하면 과장. 또한 SDMAE가 용어를 coining했다고 쓰는 것도 과장 (선행 [101] 귀속 — fixer r2, S-m2) |
| "이상 라벨 신호 주입으로 이상 무시를 유도"를 TSMAE 고유 아이디어로 주장 (fixer r2 신설, S-M1) | SDMAE의 anomaly-map 분기·overlook 재구성·가중치 가산(§3.6-2)이 합성-라벨 기반의 평행 선례 — 일반 아이디어 수준의 고유성 주장은 기습 위험. §7-2의 3축(기울기 공간 adversarial 억제 + 실제 라벨 + 내부 표현 지정)으로 스코핑 필수 |

---

## 9. Phase 5 Usage Instructions

### 인용 방식
SDMAE는 Related Work에서 1회 자연스럽게 인용한다. Method 내 self-distillation 용어 정의 근처에 각주 또는 문장으로 R21 방어 논리를 탑재한다.

### 금지 사항
- Method 전반에 걸쳐 SDMAE와의 차이점을 항목 형식으로 나열하지 않는다 (R9).
- SDMAE 성능 수치를 우리 주장의 근거로 인용하지 않는다 (도메인 불일치).
- GRL 기여 또는 PU 설정 효과를 SDMAE로 justify하지 않는다.

### 필수 포함 사항
- SDMAE의 self-distillation 용례를 R21 방어 논리의 primary reference로 인용한다.
- arXiv 2306.12041 / CVPR 2024를 정확히 표기한다.
- Verbatim 발췌 ("Distinct from the aforementioned studies, to our knowledge, we are the first to introduce a variant of self-distillation in anomaly detection") 등은 dossier 내부 분석용으로만 유지하고 논문 본문에는 그대로 복사하지 않는다 (A2 규약).
- "coining the term self-distillation" 계열 표현 사용 금지 — SDMAE는 용어를 선행 [101](Zhang et al., TPAMI 2022)에 귀속 (fixer r2, S-m2). "applying/adopting/extending the term"으로 서술할 것.

### 레퍼런스 키
논문 본문에서 사용할 BibTeX 키 (제안): `ristea2024sdmae`

### 주요 BibTeX 정보
```
@inproceedings{ristea2024sdmae,
  title={Self-Distilled Masked Auto-Encoders are Efficient Video Anomaly Detectors},
  author={Ristea, Nicolae-C{\u{a}}t{\u{a}}lin and Croitoru, Florinel-Alin and Ionescu, Radu Tudor and Popescu, Marius and Khan, Fahad Shahbaz and Shah, Mubarak},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```

### 보조 레퍼런스 (R21 용어 계보용, fixer r2 — Phase 4 서지 재검증 대상)

```
@article{zhang2022selfdistillation,
  title={Self-Distillation: Towards Efficient and Compact Neural Networks},
  author={Zhang, Linfeng and Bao, Chenglong and Ma, Kaisheng},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={44},
  number={8},
  pages={4388--4403},
  year={2022}
}
```
(SDMAE reference [101] — SDMAE가 "self-distillation" 명칭을 귀속시키는 원류 논문. R21 각주에서 SDMAE와 함께 인용하면 용어 계보 방어가 완성된다.)

---

## 부록: 정정 이력

### 2026-06-11 fixer r2 (adversarial review `paper/99_reviews/p2_dossiers_r1.md` 전수 반영; fixlog: `p2_fixlog_r2.md`; 전 항목 arXiv HTML 2306.12041v2 원문 재확인 후 적용)

1. **[S-M1, MAJOR]** anomaly-map 예측 분기 + "이상 무시(overlook)" 재구성 타깃 전체 누락 해소 — §3.6-2 신설(타깃 채널·overlook GT·가중치 가산 3요소 + "90% milestone … mandatory" ablation 근거, verbatim 5건), §4.1 유사점 행 신설(위험도 높음), §4.2 주입-계층 차이 행 신설, §6.2 GRL 행 스코핑 단서, §7-2 위험 분석(방어 3축) 신설, §8 overextension 행 추가, §3.3 모션 가중 서술의 증강-프레임 예외 정정.
2. **[X-M1, MAJOR]** §4.2 레이블 설정 행 — "반지도/PU 학습" 확정 서술을 정본 프레이밍(설정 가정 / main 271 = 상한 케이스 / sweep 계획 R32 / 명명은 Phase 3 결정 — RESEARCH_SYNTHESIS §②)으로 교정. §5.2 경로 3·§6.2 해당 행의 "PU-setting/PU·semi" 확정 표현도 동기 완화.
3. **[S-m1]** R21 핵심 인용 출처 "(Section 3)" → **§1 Introduction 기여 목록** (전문 유일 출현 실측).
4. **[S-m2]** "[101]" 마커 복원 + **[101] = Zhang, Bao & Ma, TPAMI 44(8), 2022** 식별·기록, §5.1 용어 계보 단락 신설, §5.3 계보 대응 bullet 추가, 옵션 B 초안 "coining" → "applying … following Zhang et al." 완화, 보조 BibTeX 추가. #5–#9 인용의 마커 탈락·무표기 절단 일괄 복원(괄호구·두문·인용 클러스터·말미 절단).
5. **[S-m3]** §3.1 — teacher decoder도 projection 128 명기 ("All decoder blocks …" 원문 직접 확인).
6. **[S-m4]** §6.1 — "비대칭 decoder 깊이" 행을 간접 지지로 강등 (Table 3 인용은 점수 결합 전략의 근거).
7. **[S-m5]** §1/§2 — 발표 유형 **Poster** 확정 (CVPR virtual 페이지 직접 확인).
8. **[S-m6]** §4.2 — 본 연구 대표 지표를 "AUROC" → "PA%K-AUC F1·VUS-ROC/PR·Affiliation F1"로 정정 (NRdetector dossier와 문서 간 정합화).
9. **[리뷰 B-3 참고 항목 채택]** §4.2 — phase-2 backbone 동결(SDMAE) vs teacher 비동결(271, `freeze_teacher_after_warmup=False`) 차이 행 신설.
