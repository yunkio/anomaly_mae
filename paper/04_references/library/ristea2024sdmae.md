---
phase: 4
agent: excerpt-curator-2
directives: [T4]
last_modified: 2026-06-11
key: ristea2024sdmae
verification_status: VERIFIED_A
verified_by_A: source-verifier-A2
card_grade: FULL
excerpt_access: full_html
corrections:
  - field: publisher_doi
    wrong: "[verifier-TODO]"
    correct: "10.1109/CVPR52733.2024.01513"
    severity: MINOR
    confirmed_by: "DBLP conf/cvpr/RisteaCIPKS24"
  - field: pages
    wrong: missing
    correct: "15984–15995"
    severity: MINOR
    confirmed_by: "DBLP conf/cvpr/RisteaCIPKS24"
---

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT — verbatim excerpts below are for internal reference only and must not appear in any submitted text -->

> **A2 경고**: 아래 verbatim 발췌는 card 내부 전용입니다. 어떠한 형태로도 원고에 그대로 복사하지 마십시오.

---

## 서지 정보

- **Key**: ristea2024sdmae
- **제목**: Self-Distilled Masked Auto-Encoders are Efficient Video Anomaly Detectors
- **저자**: Nicolae-Catalin Ristea, Florinel-Alin Croitoru, Radu Tudor Ionescu, Marius Popescu, Fahad Shahbaz Khan, Mubarak Shah
- **Venue**: CVPR 2024
- **arXiv**: 2306.12041 (v1: 2023-06-21; v2: 2024-03-09)
- **Publisher DOI**: 10.1109/CVPR52733.2024.01513 [VERIFIED_A: confirmed via DBLP]
- **Pages**: 15984–15995 [VERIFIED_A: confirmed via DBLP]
- **확인 출처**: arXiv abs + HTML 직접 열람 (2026-06-11)

---

## Abstract (verbatim)

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT -->

"We propose an efficient abnormal event detection model based on a lightweight masked auto-encoder (AE) applied at the video frame level. The novelty of the proposed model is threefold. First, we introduce an approach to weight tokens based on motion gradients, thus shifting the focus from the static background scene to the foreground objects. Second, we integrate a teacher decoder and a student decoder into our architecture, leveraging the discrepancy between the outputs given by the two decoders to improve anomaly detection. Third, we generate synthetic abnormal events to augment the training videos, and task the masked AE model to jointly reconstruct the original frames (without anomalies) and the corresponding pixel-level anomaly maps. Our design leads to an efficient and effective model, as demonstrated by the extensive experiments carried out on four benchmarks: Avenue, ShanghaiTech, UBnormal and UCSD Ped2. The empirical results show that our model achieves an excellent trade-off between speed and accuracy, obtaining competitive AUC scores, while processing 1655 FPS."

---

## 핵심 발췌

### 발췌 1 — 자기증류 novelty 주장 (§ Related Work / §3)

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT -->

> "To our knowledge, we are the first to introduce a variant of self-distillation in anomaly detection."

- **§위치**: Related Work 섹션 (self-distillation 관련 단락)
- **지지 Claim**: C-029 (SDMAE가 self-distillation을 anomaly detection에 처음 적용한다는 계보 서술)
- **활용 맥락**: 우리 논문에서 "SDMAE applies self-distillation to anomaly detection" 서술 시 이 주장이 출처. 단, 우리 논문은 이를 시계열 도메인으로 확장하므로 계보의 중간 단계로 인용.
- **주의**: SDMAE 자신의 novelty 주장이므로 우리 논문은 "applies" 수준으로만 서술 — "first to introduce SD in AD"를 우리가 주장하면 안 됨.

---

### 발췌 2 — Student decoder 분기 구조 (§3 Architecture)

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT -->

> "the student decoder branches out from the teacher after the first transformer block of the main decoder, adding only one extra transformer block"

- **§위치**: §3 (Architecture description)
- **지지 Claim**: C-030 (SDMAE 구조와 본 논문의 구조적 차이 방어)
- **활용 맥락**: 우리 논문 각주 또는 §2.3에서 SDMAE 아키텍처를 정확히 기술하기 위한 근거. 우리 모델은 공유 encoder에서 독립 비대칭 decoder 2개가 병렬 분기하는 반면, SDMAE는 teacher decoder 첫 블록 이후 branch-off — 이 구조적 차이를 방어하는 발췌.
- **주의**: 이 문장의 정확한 의미는 "student가 teacher decoder의 첫 transformer block 이후에 분기"임. 우리 모델과의 차이를 설명할 때만 사용.

---

### 발췌 3 — 이상 재구성 억제 메커니즘 (§3 Anomaly Supervision)

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT -->

> "we force our model to reconstruct the original training frames (without anomalies) to limit its ability to reconstruct anomalies, hence generating higher errors when anomalies occur."

- **§위치**: §3 (Anomaly overlook / synthetic anomaly supervision)
- **지지 Claim**: C-035 (SDMAE의 anomaly supervision이 타깃/손실 공간에서 작동 — 우리 GRL은 gradient 공간에서 작동, 작동 계층 차이 방어)
- **활용 맥락**: 우리 §3.5 또는 §2.3에서 SDMAE의 supervison 방식이 재구성 타깃 수준에서 이상을 억제한다는 것을 설명할 때. 우리 GRL은 gradient reversal로 표현 공간에서 직접 작동한다는 차별화의 근거.
- **주의**: "anomaly overlook"이라는 용어는 원문에 없음 — 이 현상을 지칭할 때 원문 표현("force to reconstruct original frames without anomalies")에 기반하여 paraphrase.

---

### 발췌 4 — Self-distillation 정의 (§3)

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT -->

> "a novel variant of self-distillation with a shared encoder and two decoders, a teacher and a student, where the student learns to distill knowledge from the already optimized teacher."

- **§위치**: §3 (self-distillation 방법 도입부)
- **지지 Claim**: C-029, C-034 (self-distillation 용어 계보, SDMAE AD 적용 계보)
- **활용 맥락**: §2.3에서 SDMAE의 self-distillation 구조를 간결하게 소개할 때. 공유 encoder + teacher/student decoder 구조가 우리 모델과 공통점이 있음을 인정하면서, 분기 방식의 차이를 강조.
- **주의**: "student learns to distill knowledge from the already optimized teacher" — SDMAE의 2단계 학습(encoder 동결 후 student 학습)을 전제함. 우리 모델의 joint training과 다른 점.

---

### 발췌 5 — Teacher/student 구조 도입 동기 (§Abstract / §3)

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT -->

> "we integrate a teacher decoder and a student decoder into our architecture, leveraging the discrepancy between the outputs given by the two decoders to improve anomaly detection."

- **§위치**: Abstract (§3 본문과 일치)
- **지지 Claim**: C-029 (계보 서술), C-034 (용어 계보)
- **활용 맥락**: teacher-student discrepancy를 anomaly 신호로 사용하는 원리가 SDMAE에서 도입되었음을 명시. 우리 모델도 동일 원리를 채택하지만 시계열 패치 도메인에 적용하고 추가로 GRL 기반 라벨 통합.
- **주의**: 이 아이디어가 우리 논문의 novelty가 아니라 SDMAE에서 온 것임을 명확히 해야 함.

---

## 활용 절

| 우리 논문 위치 | 활용 방식 | 근거 발췌 |
|-------------|---------|---------|
| §2.3 Related Work | SDMAE가 video AD에서 self-distillation을 도입한 선행 연구로 소개 | 발췌 1, 5 |
| §2.3 각주 또는 본문 | SDMAE student decoder 분기 구조를 정확히 기술 (우리 구조와 차이 방어) | 발췌 2 |
| §3.5 Method | SDMAE supervision 방식(재구성 타깃 수준)과 우리 GRL(gradient 수준)의 차별화 | 발췌 3 |
| §3.4 / §2.3 | self-distillation 용어 계보: Zhang TPAMI 2022 → SDMAE(AD 적용) → 본 논문(TSAD 확장) | 발췌 4 |

---

## 주의사항

1. **도메인 차이**: SDMAE는 비디오 프레임 이상탐지 — 우리는 다변량 시계열. 직접 성능 비교는 불가하며, 방법론적 계보 인용만 유효.
2. **novelty 주장 충돌 방지**: 발췌 1의 SDMAE 자체 주장("we are the first")을 우리 논문에서 다시 인용할 때, SDMAE가 video AD에서의 first임을 명시 — 시계열 AD에서의 first는 우리 논문이 주장할 수 있으나 SDMAE 선행 존재는 인정.
3. **publisher DOI**: verifier 확인 전 arXiv 식별자로만 인용.
4. **복사 금지 표현**: "we are the first to introduce a variant of self-distillation in anomaly detection" — 우리 논문에서 이 문장을 그대로 쓰면 SDMAE의 주장을 표절하거나 우리 자신의 주장과 충돌. 반드시 paraphrase + 귀속 표기.
